"""RAG query endpoint — standard JSON + SSE streaming + chat history."""
import json
import logging
import uuid
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import ChatMessage, ChatSession, SessionLocal, User, get_db
from app import llm, vector

log = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: Optional[str] = None
    document_ids: Optional[List[str]] = None


class SourceOut(BaseModel):
    document_id: str
    document_name: str
    page: int
    chunk_text: str
    score: float


class QueryResponse(BaseModel):
    session_id: str
    message_id: str
    answer: str
    sources: List[SourceOut]


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    sources: List[SourceOut]

    @field_validator("sources", mode="before")
    @classmethod
    def _parse_sources(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                log.warning("Corrupt sources JSON in chat_messages; returning empty list")
                return []
        return v or []


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    messages: List[MessageOut] = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_create_session(session_id: Optional[str], question: str,
                            user: User, db: Session) -> ChatSession:
    if session_id:
        s = db.query(ChatSession).filter(
            ChatSession.id == session_id, ChatSession.user_id == user.id
        ).first()
        if not s:
            raise HTTPException(404, "Chat session not found")
        return s
    s = ChatSession(id=str(uuid.uuid4()), title=question[:60], user_id=user.id)
    db.add(s)
    db.commit()
    db.refresh(s)
    return s


def _get_session_or_404(session_id: str, user_id: str, db: Session) -> ChatSession:
    s = db.query(ChatSession).filter(
        ChatSession.id == session_id, ChatSession.user_id == user_id
    ).first()
    if not s:
        raise HTTPException(404, "Session not found")
    return s


def _history(session: ChatSession) -> List[dict]:
    return [{"role": m.role, "content": m.content} for m in session.messages]


def _save_messages(session_id: str, question: str,
                   answer: str, sources: List[dict], db: Session) -> str:
    db.add(ChatMessage(session_id=session_id, role="user", content=question))
    msg_id = str(uuid.uuid4())
    db.add(ChatMessage(id=msg_id, session_id=session_id, role="assistant",
                       content=answer, sources=json.dumps(sources)))
    db.commit()
    return msg_id


def _search_or_404(question: str, doc_ids: Optional[List[str]]) -> List[dict]:
    sources = vector.search(question, doc_ids=doc_ids)
    if not sources:
        raise HTTPException(404, "No relevant documents found. Upload a PDF first.")
    return sources


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sources = _search_or_404(body.question, body.document_ids)
    session = _get_or_create_session(body.session_id, body.question, current_user, db)
    history = _history(session)
    full_answer = ""
    async for token in llm.stream_answer(body.question, sources, history):
        full_answer += token
    msg_id = _save_messages(session.id, body.question, full_answer, sources, db)
    return QueryResponse(session_id=session.id, message_id=msg_id,
                         answer=full_answer, sources=sources)


@router.post("/stream", response_class=StreamingResponse)
async def stream(
    body: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Server-Sent Events streaming endpoint."""
    sources = _search_or_404(body.question, body.document_ids)
    session = _get_or_create_session(body.session_id, body.question, current_user, db)
    session_id = session.id
    history = _history(session)

    async def event_stream():
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        full_answer = ""
        # open a fresh session for the generator — the request-scoped db may be closed
        # before StreamingResponse finishes iterating
        gen_db = SessionLocal()
        try:
            async for token in llm.stream_answer(body.question, sources, history):
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            msg_id = _save_messages(session_id, body.question, full_answer, sources, gen_db)
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'message_id': msg_id})}\n\n"
        except Exception as e:
            log.error("Stream error for session %s: %s", session_id, e)
            yield f"data: {json.dumps({'type': 'error', 'detail': 'Stream interrupted'})}\n\n"
        finally:
            gen_db.close()

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no"})


@router.get("/sessions", response_model=List[SessionOut])
def list_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(ChatSession).filter(ChatSession.user_id == current_user.id).all()


@router.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: str, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    return _get_session_or_404(session_id, current_user.id, db)


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    s = _get_session_or_404(session_id, current_user.id, db)
    db.delete(s)
    db.commit()

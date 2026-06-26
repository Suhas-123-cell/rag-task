"""RAG query endpoint — standard JSON + SSE streaming + chat history."""
import json
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import ChatMessage, ChatSession, User, get_db
from app import llm, vector

router = APIRouter(prefix="/chat", tags=["chat"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    document_ids: Optional[List[str]] = None   # restrict search to these docs


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
    id: str
    role: str
    content: str
    sources: List[SourceOut]

    class Config:
        from_attributes = True


class SessionOut(BaseModel):
    id: str
    title: str
    messages: List[MessageOut] = []

    class Config:
        from_attributes = True


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


def _history(session: ChatSession) -> List[dict]:
    return [{"role": m.role, "content": m.content} for m in session.messages]


def _save_messages(session: ChatSession, question: str,
                   answer: str, sources: List[dict], db: Session) -> str:
    db.add(ChatMessage(session_id=session.id, role="user", content=question))
    msg_id = str(uuid.uuid4())
    db.add(ChatMessage(id=msg_id, session_id=session.id, role="assistant",
                       content=answer, sources=json.dumps(sources)))
    db.commit()
    return msg_id


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(
    body: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    sources = vector.search(body.question, doc_ids=body.document_ids)
    if not sources:
        raise HTTPException(404, "No relevant documents found. Upload a PDF first.")
    session = _get_or_create_session(body.session_id, body.question, current_user, db)
    history = _history(session)
    answer = await llm.get_answer(body.question, sources, history)
    msg_id = _save_messages(session, body.question, answer, sources, db)
    return QueryResponse(session_id=session.id, message_id=msg_id,
                         answer=answer, sources=sources)


@router.post("/stream")
async def stream(
    body: QueryRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Server-Sent Events streaming endpoint."""
    sources = vector.search(body.question, doc_ids=body.document_ids)
    if not sources:
        raise HTTPException(404, "No relevant documents found. Upload a PDF first.")
    session = _get_or_create_session(body.session_id, body.question, current_user, db)
    history = _history(session)

    async def event_stream():
        # First event: sources metadata
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
        full_answer = ""
        async for token in llm.stream_answer(body.question, sources, history):
            full_answer += token
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
        msg_id = _save_messages(session, body.question, full_answer, sources, db)
        yield f"data: {json.dumps({'type': 'done', 'session_id': session.id, 'message_id': msg_id})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"X-Accel-Buffering": "no"})


@router.get("/sessions", response_model=List[SessionOut])
def list_sessions(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(ChatSession).filter(ChatSession.user_id == current_user.id).all()


@router.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: str, db: Session = Depends(get_db),
                current_user: User = Depends(get_current_user)):
    s = db.query(ChatSession).filter(
        ChatSession.id == session_id, ChatSession.user_id == current_user.id
    ).first()
    if not s:
        raise HTTPException(404, "Session not found")
    for m in s.messages:
        try:
            m.sources = json.loads(m.sources or "[]")
        except Exception:
            m.sources = []
    return s


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str, db: Session = Depends(get_db),
                   current_user: User = Depends(get_current_user)):
    s = db.query(ChatSession).filter(
        ChatSession.id == session_id, ChatSession.user_id == current_user.id
    ).first()
    if not s:
        raise HTTPException(404, "Session not found")
    db.delete(s)
    db.commit()

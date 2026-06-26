"""PDF upload, extraction, chunking, and embedding pipeline."""
import logging
import os
import re
import uuid
from typing import List, Literal, Tuple

import fitz  # PyMuPDF
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import settings
from app.database import Document, User, get_db
from app import vector

log = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    original_name: str
    num_pages: int
    num_chunks: int
    file_size: int
    status: Literal["processing", "ready", "error"]


# ── PDF helpers ───────────────────────────────────────────────────────────────

def _extract(file_path: str) -> Tuple[List[dict], int]:
    with fitz.open(file_path) as doc:
        pages = []
        for i, page in enumerate(doc):
            text = re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", page.get_text("text"))).strip()
            if text:
                pages.append({"text": text, "page": i + 1})
        return pages, len(doc)


def _chunk(pages: List[dict]) -> List[dict]:
    """Semantic chunking: split at paragraph boundaries, accumulate to CHUNK_SIZE words."""
    chunks, size = [], settings.CHUNK_SIZE
    for p in pages:
        paragraphs = [s.strip() for s in re.split(r"\n\n+", p["text"]) if s.strip()]
        current, current_words = [], 0
        for para in paragraphs:
            para_words = len(para.split())
            if current and current_words + para_words > size:
                chunks.append({"text": " ".join(current), "page": p["page"]})
                # keep last paragraph as overlap for context continuity
                current = [current[-1]]
                current_words = len(current[0].split())
            current.append(para)
            current_words += para_words
        if current:
            chunks.append({"text": " ".join(current), "page": p["page"]})
    return chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_doc_or_404(doc_id: str, user_id: str, db: Session) -> Document:
    doc = db.query(Document).filter(Document.id == doc_id, Document.owner_id == user_id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentOut, status_code=201)
async def upload(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    content = await file.read()
    if len(content) > settings.MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.MAX_FILE_MB} MB limit")

    doc_id = str(uuid.uuid4())
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.pdf")
    with open(save_path, "wb") as f:
        f.write(content)

    doc = Document(
        id=doc_id,
        original_name=file.filename,
        file_path=save_path,
        file_size=len(content),
        owner_id=current_user.id,
        status="processing",
    )
    db.add(doc)
    db.commit()

    try:
        pages, num_pages = _extract(save_path)
        chunks = _chunk(pages)
        vector.add_chunks(doc_id, file.filename, chunks)
        doc.num_pages = num_pages
        doc.num_chunks = len(chunks)
        doc.status = "ready"
    except Exception as e:
        log.error("PDF processing failed for %s: %s", doc_id, e)
        # clean up partial state so disk and vector store stay consistent
        vector.delete_doc(doc_id)
        if os.path.exists(save_path):
            os.remove(save_path)
        doc.status = "error"
        db.commit()
        raise HTTPException(500, "PDF processing failed. Check server logs.")

    db.commit()
    db.refresh(doc)
    return doc


@router.get("/", response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Document).filter(Document.owner_id == current_user.id).all()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return _get_doc_or_404(doc_id, current_user.id, db)


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = _get_doc_or_404(doc_id, current_user.id, db)
    vector.delete_doc(doc_id)
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    db.delete(doc)
    db.commit()

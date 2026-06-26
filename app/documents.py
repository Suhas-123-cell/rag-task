"""PDF upload, extraction, chunking, and embedding pipeline."""
import os
import re
import uuid
from typing import List, Tuple

import fitz  # PyMuPDF
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.config import settings
from app.database import Document, User, get_db
from app import vector

router = APIRouter(prefix="/documents", tags=["documents"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class DocumentOut(BaseModel):
    id: str
    original_name: str
    num_pages: int
    num_chunks: int
    file_size: int
    status: str

    class Config:
        from_attributes = True


# ── PDF helpers ───────────────────────────────────────────────────────────────

def _extract(file_path: str) -> Tuple[List[dict], int]:
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", page.get_text("text"))).strip()
        if text:
            pages.append({"text": text, "page": i + 1})
    total = len(doc)
    doc.close()
    return pages, total


def _chunk(pages: List[dict]) -> List[dict]:
    chunks, size, overlap = [], settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    for p in pages:
        words = p["text"].split()
        start = 0
        while start < len(words):
            end = min(start + size, len(words))
            chunks.append({"text": " ".join(words[start:end]), "page": p["page"]})
            if end == len(words):
                break
            start += size - overlap
    return chunks


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
        doc.status = "error"
        db.commit()
        raise HTTPException(500, f"Processing failed: {e}")

    db.commit()
    db.refresh(doc)
    return doc


@router.get("/", response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Document).filter(Document.owner_id == current_user.id).all()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = db.query(Document).filter(Document.id == doc_id, Document.owner_id == current_user.id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = db.query(Document).filter(Document.id == doc_id, Document.owner_id == current_user.id).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    vector.delete_doc(doc_id)
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
    db.delete(doc)
    db.commit()

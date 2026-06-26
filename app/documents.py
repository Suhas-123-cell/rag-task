"""PDF upload, extraction, chunking, and embedding pipeline."""
import logging
import os
import re
import uuid
from functools import lru_cache
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


# ── Storage (S3 or local) ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def s3_client():
    import boto3  # lazy: only needed when S3_BUCKET is set
    return boto3.client(
        "s3",
        region_name=settings.AWS_REGION,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    )


def storage_save(doc_id: str, content: bytes) -> str:
    """Upload to S3 if configured, else write to local disk. Returns the stored ref."""
    if settings.S3_BUCKET:
        key = f"documents/{doc_id}.pdf"
        s3_client().put_object(Bucket=settings.S3_BUCKET, Key=key, Body=content,
                               ContentType="application/pdf")
        return key
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}.pdf")
    with open(path, "wb") as f:
        f.write(content)
    return path


def storage_delete(file_ref: str) -> None:
    if settings.S3_BUCKET:
        try:
            s3_client().delete_object(Bucket=settings.S3_BUCKET, Key=file_ref)
        except Exception as e:
            log.warning("S3 delete failed for %s: %s", file_ref, e)
    elif os.path.exists(file_ref):
        os.remove(file_ref)


# ── PDF helpers ───────────────────────────────────────────────────────────────

def extract_pdf(content: bytes) -> Tuple[List[dict], int]:
    with fitz.open(stream=content, filetype="pdf") as doc:
        pages = []
        for i, page in enumerate(doc):
            text = re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", page.get_text("text"))).strip()
            if text:
                pages.append({"text": text, "page": i + 1})
        return pages, len(doc)


def chunk_pages(pages: List[dict]) -> List[dict]:
    """Semantic chunking: split at paragraph boundaries, accumulate to CHUNK_SIZE words."""
    chunks, size = [], settings.CHUNK_SIZE
    for p in pages:
        paragraphs = [s.strip() for s in re.split(r"\n\n+", p["text"]) if s.strip()]
        current, current_words = [], 0
        for para in paragraphs:
            para_words = len(para.split())
            if current and current_words + para_words > size:
                chunks.append({"text": " ".join(current), "page": p["page"]})
                current = [current[-1]]
                current_words = len(current[0].split())
            current.append(para)
            current_words += para_words
        if current:
            chunks.append({"text": " ".join(current), "page": p["page"]})
    return chunks


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_doc_or_404(doc_id: str, user_id: str, db: Session) -> Document:
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
    file_ref = storage_save(doc_id, content)

    doc = Document(
        id=doc_id,
        original_name=file.filename,
        file_path=file_ref,
        file_size=len(content),
        owner_id=current_user.id,
        status="processing",
    )
    db.add(doc)
    db.commit()

    try:
        pages, num_pages = extract_pdf(content)
        chunks = chunk_pages(pages)
        vector.add_chunks(doc_id, file.filename, chunks)
        doc.num_pages = num_pages
        doc.num_chunks = len(chunks)
        doc.status = "ready"
    except Exception as e:
        log.error("PDF processing failed for %s: %s", doc_id, e)
        vector.delete_doc(doc_id)
        storage_delete(file_ref)
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
    return get_doc_or_404(doc_id, current_user.id, db)


@router.delete("/{doc_id}", status_code=204)
def delete_document(doc_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    doc = get_doc_or_404(doc_id, current_user.id, db)
    vector.delete_doc(doc_id)
    storage_delete(doc.file_path)
    db.delete(doc)
    db.commit()

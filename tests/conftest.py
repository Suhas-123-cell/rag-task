"""Shared fixtures — in-memory DB, mocked vector/LLM, test client."""
import os
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("SECRET_KEY", "test-secret-key-for-pytest-only-32x")
os.environ.setdefault("GEMINI_API_KEY", "test-key")

import fitz
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app

# ── In-memory SQLite — StaticPool keeps one connection so tables persist ──────

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestSession()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def client(fresh_db):
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def mock_vector():
    """Stub all ChromaDB calls — no real vector DB needed."""
    source = {
        "document_id": "doc-1",
        "document_name": "sample.pdf",
        "page": 1,
        "chunk_text": "The mitochondria is the powerhouse of the cell.",
        "score": 0.92,
    }
    with patch("app.documents.vector") as dv, patch("app.chat.vector") as cv:
        dv.add_chunks = MagicMock()
        dv.delete_doc = MagicMock()
        cv.search = MagicMock(return_value=[source])
        cv.delete_doc = MagicMock()
        yield dv, cv


@pytest.fixture()
def mock_llm():
    """Stub LLM so tests don't need an API key."""
    async def fake_stream(*args, **kwargs):
        for token in ["Hello ", "world."]:
            yield token

    with patch("app.chat.llm") as m:
        m.stream_answer = fake_stream
        yield m


@pytest.fixture()
def sample_pdf(tmp_path):
    """Single-page PDF with synthetic text, created via PyMuPDF."""
    path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "The mitochondria is the powerhouse of the cell.")
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture()
def registered_user(client):
    """Register a user and return (auth_headers, response_json)."""
    resp = client.post("/auth/register", json={
        "email": "test@example.com",
        "username": "testuser",
        "password": "password123",
    })
    assert resp.status_code == 201
    token = resp.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}, resp.json()


@pytest.fixture()
def uploaded_doc(client, registered_user, sample_pdf, mock_vector):
    """Upload a PDF and return the document JSON."""
    headers, _ = registered_user
    with open(sample_pdf, "rb") as f:
        resp = client.post("/documents/upload",
                           files={"file": ("sample.pdf", f, "application/pdf")},
                           headers=headers)
    assert resp.status_code == 201
    return resp.json(), headers

# RAG Document Assistant

AI-powered Document Assistant that answers questions from PDF files using Retrieval-Augmented Generation (RAG).

## Features

| Feature | Status |
|---|---|
| PDF Upload API | ✅ |
| Text Extraction (PyMuPDF) | ✅ |
| Sliding-window Chunking | ✅ |
| Local Embeddings (all-MiniLM-L6-v2) | ✅ |
| ChromaDB Vector Store | ✅ |
| Question Answering with Source References | ✅ |
| Chat History API | ✅ |
| Multi-document Support | ✅ Bonus |
| JWT Authentication | ✅ Bonus |
| SSE Streaming Responses | ✅ Bonus |
| Docker / docker-compose | ✅ Bonus |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client (HTTP / SSE)                    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   FastAPI  (app/main.py)                  │
│  ┌──────────┐  ┌─────────────┐  ┌───────────────────┐  │
│  │  /auth   │  │ /documents  │  │      /chat        │  │
│  │ register │  │  upload     │  │  query  (JSON)    │  │
│  │  login   │  │  list       │  │  stream (SSE)     │  │
│  │   me     │  │  delete     │  │  sessions+history │  │
│  └────┬─────┘  └──────┬──────┘  └────────┬──────────┘  │
└───────│───────────────│──────────────────│─────────────┘
        │               │                  │
        ▼               ▼                  ▼
  ┌──────────┐   ┌──────────────┐   ┌──────────────────┐
  │  SQLite  │   │ PDF Pipeline │   │   RAG Pipeline   │
  │ (users + │   │  PyMuPDF →   │   │ vector.search()  │
  │ history) │   │  chunker →   │   │ → llm.stream()   │
  └──────────┘   │  ChromaDB    │   └──────────────────┘
                 └──────┬───────┘           │
                        ▼                   ▼
                 ┌──────────────┐   ┌──────────────────┐
                 │   ChromaDB   │   │ Gemini / OpenAI  │
                 │  (persisted) │   │  (configurable)  │
                 └──────────────┘   └──────────────────┘
```

### Data Flow

**Upload:**
```
PDF → PyMuPDF (text per page)
    → sliding-window chunker (800 words, 150 overlap)
    → SentenceTransformer (all-MiniLM-L6-v2) → embeddings
    → ChromaDB (stored with document_id + page metadata)
```

**Query:**
```
Question → embed → ChromaDB cosine search (top-5 chunks)
         → build prompt (context + last 6 messages)
         → Gemini / OpenAI (stream tokens via SSE)
         → persist to SQLite chat history
         → return answer + source references (doc name + page)
```

## Quick Start

```bash
git clone https://github.com/Suhas-123-cell/rag-task.git
cd rag-task
cp .env.example .env        # add your GEMINI_API_KEY
pip install -r requirements.txt
uvicorn app.main:app --reload
# → http://localhost:8000/docs
```

### Docker

```bash
docker compose up --build
```

## API Reference

### Auth
| Method | Endpoint | Description |
|---|---|---|
| POST | `/auth/register` | Register → returns JWT |
| POST | `/auth/login` | Login (form data) → JWT |
| GET | `/auth/me` | Current user info |

### Documents
| Method | Endpoint | Description |
|---|---|---|
| POST | `/documents/upload` | Upload PDF (multipart) |
| GET | `/documents/` | List documents |
| GET | `/documents/{id}` | Get metadata |
| DELETE | `/documents/{id}` | Delete doc + embeddings |

### Chat
| Method | Endpoint | Description |
|---|---|---|
| POST | `/chat/query` | Ask → JSON response + sources |
| POST | `/chat/stream` | Ask → SSE stream |
| GET | `/chat/sessions` | List sessions |
| GET | `/chat/sessions/{id}` | Full session history |
| DELETE | `/chat/sessions/{id}` | Delete session |

### SSE Stream Format
```
data: {"type": "sources", "sources": [...]}
data: {"type": "token", "content": "Hello"}
data: {"type": "done", "session_id": "...", "message_id": "..."}
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | `gemini` or `openai` |
| `GEMINI_API_KEY` | — | Free key at aistudio.google.com |
| `OPENAI_API_KEY` | — | OpenAI key |
| `CHUNK_SIZE` | `800` | Words per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K` | `5` | Retrieved chunks per query |
| `MAX_FILE_MB` | `50` | Max upload size |

## Tech Stack

- **FastAPI** — async REST + auto OpenAPI docs
- **ChromaDB** — persistent vector store (cosine similarity)
- **sentence-transformers** — local embeddings, no API key
- **PyMuPDF** — fast, accurate PDF text extraction
- **SQLAlchemy + SQLite** — users and chat history
- **python-jose + passlib** — JWT auth
- **Google Gemini / OpenAI** — configurable LLM backend

"""Full API test suite — auth, documents, chat."""
from unittest.mock import MagicMock, patch


# ── Health ────────────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Auth: register ────────────────────────────────────────────────────────────

def test_register_success(client):
    r = client.post("/auth/register", json={
        "email": "user@example.com", "username": "user1", "password": "securepass",
    })
    assert r.status_code == 201
    data = r.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["username"] == "user1"


def test_register_duplicate_username(client):
    payload = {"email": "a@example.com", "username": "dup", "password": "password1"}
    client.post("/auth/register", json=payload)
    r = client.post("/auth/register", json={**payload, "email": "b@example.com"})
    assert r.status_code == 400


def test_register_duplicate_email(client):
    payload = {"email": "same@example.com", "username": "user1", "password": "password1"}
    client.post("/auth/register", json=payload)
    r = client.post("/auth/register", json={**payload, "username": "user2"})
    assert r.status_code == 400


def test_register_short_password_rejected(client):
    r = client.post("/auth/register", json={
        "email": "x@example.com", "username": "abc", "password": "short",
    })
    assert r.status_code == 422


def test_register_short_username_rejected(client):
    r = client.post("/auth/register", json={
        "email": "x@example.com", "username": "ab", "password": "password1",
    })
    assert r.status_code == 422


# ── Auth: login ───────────────────────────────────────────────────────────────

def test_login_success(client, registered_user):
    r = client.post("/auth/login", data={"username": "testuser", "password": "password123"})
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_login_wrong_password(client, registered_user):
    r = client.post("/auth/login", data={"username": "testuser", "password": "wrongpass"})
    assert r.status_code == 401


def test_login_unknown_user(client):
    r = client.post("/auth/login", data={"username": "nobody", "password": "password123"})
    assert r.status_code == 401


# ── Auth: me ──────────────────────────────────────────────────────────────────

def test_me_returns_user(client, registered_user):
    headers, _ = registered_user
    r = client.get("/auth/me", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["username"] == "testuser"
    assert body["email"] == "test@example.com"
    assert "id" in body


def test_me_unauthenticated(client):
    assert client.get("/auth/me").status_code == 401


def test_me_invalid_token(client):
    r = client.get("/auth/me", headers={"Authorization": "Bearer notavalidtoken"})
    assert r.status_code == 401


# ── Documents: upload ─────────────────────────────────────────────────────────

def test_upload_pdf_success(client, registered_user, sample_pdf, mock_vector):
    headers, _ = registered_user
    with open(sample_pdf, "rb") as f:
        r = client.post("/documents/upload",
                        files={"file": ("sample.pdf", f, "application/pdf")},
                        headers=headers)
    assert r.status_code == 201
    body = r.json()
    assert body["original_name"] == "sample.pdf"
    assert body["status"] == "ready"
    assert body["num_pages"] == 1
    assert body["num_chunks"] >= 1


def test_upload_non_pdf_rejected(client, registered_user, tmp_path):
    headers, _ = registered_user
    txt = tmp_path / "file.txt"
    txt.write_text("hello")
    with open(txt, "rb") as f:
        r = client.post("/documents/upload",
                        files={"file": ("file.txt", f, "text/plain")},
                        headers=headers)
    assert r.status_code == 400


def test_upload_requires_auth(client, sample_pdf):
    with open(sample_pdf, "rb") as f:
        r = client.post("/documents/upload",
                        files={"file": ("sample.pdf", f, "application/pdf")})
    assert r.status_code == 401


# ── Documents: list / get / delete ───────────────────────────────────────────

def test_list_documents_empty(client, registered_user):
    headers, _ = registered_user
    r = client.get("/documents/", headers=headers)
    assert r.status_code == 200
    assert r.json() == []


def test_list_documents_after_upload(client, uploaded_doc):
    doc, headers = uploaded_doc
    r = client.get("/documents/", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.json()[0]["id"] == doc["id"]


def test_get_document_success(client, uploaded_doc):
    doc, headers = uploaded_doc
    r = client.get(f"/documents/{doc['id']}", headers=headers)
    assert r.status_code == 200
    assert r.json()["id"] == doc["id"]


def test_get_document_not_found(client, registered_user):
    headers, _ = registered_user
    assert client.get("/documents/nonexistent-id", headers=headers).status_code == 404


def test_delete_document_success(client, uploaded_doc, mock_vector):
    doc, headers = uploaded_doc
    assert client.delete(f"/documents/{doc['id']}", headers=headers).status_code == 204
    assert client.get(f"/documents/{doc['id']}", headers=headers).status_code == 404


def test_delete_document_not_found(client, registered_user):
    headers, _ = registered_user
    assert client.delete("/documents/nonexistent-id", headers=headers).status_code == 404


def test_user_cannot_access_other_users_document(client, sample_pdf, mock_vector):
    h1 = {"Authorization": "Bearer " + client.post("/auth/register", json={
        "email": "u1@example.com", "username": "user1", "password": "password123"
    }).json()["access_token"]}
    h2 = {"Authorization": "Bearer " + client.post("/auth/register", json={
        "email": "u2@example.com", "username": "user2", "password": "password123"
    }).json()["access_token"]}

    with open(sample_pdf, "rb") as f:
        doc = client.post("/documents/upload",
                          files={"file": ("sample.pdf", f, "application/pdf")},
                          headers=h1).json()

    assert client.get(f"/documents/{doc['id']}", headers=h2).status_code == 404


# ── Chat: query ───────────────────────────────────────────────────────────────

def test_query_success(client, uploaded_doc, mock_vector, mock_llm):
    doc, headers = uploaded_doc
    r = client.post("/chat/query",
                    json={"question": "What is the powerhouse of the cell?"},
                    headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Hello world."
    assert len(body["sources"]) == 1
    assert body["sources"][0]["document_name"] == "sample.pdf"
    assert "session_id" in body
    assert "message_id" in body


def test_query_empty_question_rejected(client, registered_user):
    headers, _ = registered_user
    r = client.post("/chat/query", json={"question": ""}, headers=headers)
    assert r.status_code == 422


def test_query_no_documents_returns_404(client, registered_user):
    headers, _ = registered_user
    with patch("app.chat.vector") as mv:
        mv.search = MagicMock(return_value=[])
        r = client.post("/chat/query", json={"question": "anything?"}, headers=headers)
    assert r.status_code == 404


def test_query_continues_existing_session(client, uploaded_doc, mock_vector, mock_llm):
    doc, headers = uploaded_doc
    r1 = client.post("/chat/query", json={"question": "First?"}, headers=headers)
    session_id = r1.json()["session_id"]
    r2 = client.post("/chat/query",
                     json={"question": "Second?", "session_id": session_id},
                     headers=headers)
    assert r2.status_code == 200
    assert r2.json()["session_id"] == session_id


def test_query_requires_auth(client):
    assert client.post("/chat/query", json={"question": "hello?"}).status_code == 401


# ── Chat: stream ──────────────────────────────────────────────────────────────

def test_stream_returns_sse(client, uploaded_doc, mock_vector, mock_llm):
    doc, headers = uploaded_doc
    with client.stream("POST", "/chat/stream",
                       json={"question": "What is the powerhouse?"},
                       headers=headers) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        raw = r.read().decode()
    assert '"type": "sources"' in raw
    assert '"type": "token"' in raw
    assert '"type": "done"' in raw


# ── Chat: session history ─────────────────────────────────────────────────────

def test_list_sessions_empty(client, registered_user):
    headers, _ = registered_user
    r = client.get("/chat/sessions", headers=headers)
    assert r.status_code == 200
    assert r.json() == []


def test_list_sessions_after_query(client, uploaded_doc, mock_vector, mock_llm):
    doc, headers = uploaded_doc
    client.post("/chat/query", json={"question": "Test?"}, headers=headers)
    r = client.get("/chat/sessions", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_get_session_with_messages(client, uploaded_doc, mock_vector, mock_llm):
    doc, headers = uploaded_doc
    qr = client.post("/chat/query", json={"question": "What is the cell?"}, headers=headers)
    session_id = qr.json()["session_id"]

    r = client.get(f"/chat/sessions/{session_id}", headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert body["id"] == session_id
    assert len(body["messages"]) == 2          # user + assistant
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][1]["role"] == "assistant"
    assert body["messages"][1]["sources"][0]["page"] == 1


def test_get_session_not_found(client, registered_user):
    headers, _ = registered_user
    assert client.get("/chat/sessions/nonexistent-id", headers=headers).status_code == 404


def test_delete_session(client, uploaded_doc, mock_vector, mock_llm):
    doc, headers = uploaded_doc
    qr = client.post("/chat/query", json={"question": "Delete me?"}, headers=headers)
    session_id = qr.json()["session_id"]
    assert client.delete(f"/chat/sessions/{session_id}", headers=headers).status_code == 204
    assert client.get(f"/chat/sessions/{session_id}", headers=headers).status_code == 404

"""ChromaDB vector store + sentence-transformer embeddings."""
from functools import lru_cache
from typing import List, Dict, Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def embedding_model() -> SentenceTransformer:
    return SentenceTransformer(settings.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def chroma_collection():
    client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
    return client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def embed(texts: List[str]) -> List[List[float]]:
    return embedding_model().encode(texts, batch_size=32, show_progress_bar=False).tolist()


def add_chunks(doc_id: str, doc_name: str, chunks: List[Dict]) -> None:
    col = chroma_collection()
    texts = [c["text"] for c in chunks]
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metas = [{"document_id": doc_id, "document_name": doc_name,
               "page": c["page"], "chunk_index": i}
              for i, c in enumerate(chunks)]
    embeddings = embed(texts)
    # insert in batches of 100 to stay within chroma limits
    for i in range(0, len(ids), 100):
        col.add(ids=ids[i:i+100], embeddings=embeddings[i:i+100],
                documents=texts[i:i+100], metadatas=metas[i:i+100])


def search(query: str, top_k: int = None,
           doc_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    col = chroma_collection()
    count = col.count()
    if count == 0:
        return []
    k = min(top_k or settings.TOP_K, count)
    where = None
    if doc_ids:
        where = ({"document_id": doc_ids[0]} if len(doc_ids) == 1
                 else {"document_id": {"$in": doc_ids}})
    results = col.query(
        query_embeddings=embed([query]),
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "document_id": m["document_id"],
            "document_name": m["document_name"],
            "page": m["page"],
            "chunk_text": d,
            "score": round(1 - dist, 4),   # cosine distance → similarity
        }
        for d, m, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def delete_doc(doc_id: str) -> None:
    col = chroma_collection()
    res = col.get(where={"document_id": doc_id})
    if res["ids"]:
        col.delete(ids=res["ids"])

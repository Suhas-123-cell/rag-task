"""LLM — Groq streaming via llama-3.1-8b-instant."""
from typing import AsyncGenerator, Dict, List, Optional

from openai import AsyncOpenAI

from app.config import settings


def build_prompt(question: str, sources: List[Dict], history: List[Dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[{s['document_name']}, page {s['page']}]\n{s['chunk_text'][:1500]}" for s in sources
    )
    hist = ""
    if history:
        hist = "\nConversation so far:\n" + "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-6:]
        ) + "\n"
    return (
        "You are a precise document assistant. Answer using ONLY the context below. "
        "If the answer is not in the context, say so clearly. "
        "Always cite the document name and page number.\n"
        f"{hist}\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


async def stream_answer(
    question: str,
    sources: List[Dict],
    history: Optional[List[Dict]] = None,
) -> AsyncGenerator[str, None]:
    prompt = build_prompt(question, sources, history or [])
    client = AsyncOpenAI(api_key=settings.GROQ_API_KEY,
                         base_url="https://api.groq.com/openai/v1")
    stream = await client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

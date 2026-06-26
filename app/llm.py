"""LLM provider abstraction — Gemini (default) or OpenAI, streaming-first."""
from typing import List, Dict, AsyncGenerator

from app.config import settings


def _build_prompt(question: str, sources: List[Dict], history: List[Dict]) -> str:
    context = "\n\n---\n\n".join(
        f"[{s['document_name']}, page {s['page']}]\n{s['chunk_text']}" for s in sources
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
    history: List[Dict] = None,
) -> AsyncGenerator[str, None]:
    prompt = _build_prompt(question, sources, history or [])
    if settings.LLM_PROVIDER == "openai":
        async for token in _openai_stream(prompt):
            yield token
    else:
        async for token in _gemini_stream(prompt):
            yield token


async def get_answer(question: str, sources: List[Dict], history: List[Dict] = None) -> str:
    full = ""
    async for token in stream_answer(question, sources, history):
        full += token
    return full


async def _gemini_stream(prompt: str) -> AsyncGenerator[str, None]:
    import google.generativeai as genai
    genai.configure(api_key=settings.GEMINI_API_KEY)
    model = genai.GenerativeModel(settings.GEMINI_MODEL)
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


async def _openai_stream(prompt: str) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    stream = await client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

"""LLM provider abstraction — Gemini (default) or OpenAI, streaming-first."""
import asyncio
from typing import AsyncGenerator, Dict, List, Optional

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
    if settings.LLM_PROVIDER == "openai":
        async for token in openai_stream(prompt):
            yield token
    elif settings.LLM_PROVIDER == "groq":
        async for token in groq_stream(prompt):
            yield token
    else:
        async for token in gemini_stream(prompt):
            yield token



async def gemini_stream(prompt: str) -> AsyncGenerator[str, None]:
    from google import genai
    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    # new SDK is synchronous; run in thread to avoid blocking the event loop
    def collect() -> List[str]:
        chunks = []
        for chunk in client.models.generate_content_stream(
            model=settings.GEMINI_MODEL, contents=prompt
        ):
            if chunk.text:
                chunks.append(chunk.text)
        return chunks
    for chunk in await asyncio.to_thread(collect):
        yield chunk


async def groq_stream(prompt: str) -> AsyncGenerator[str, None]:
    from openai import AsyncOpenAI
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


async def openai_stream(prompt: str) -> AsyncGenerator[str, None]:
    # pyrefly: ignore [missing-import]
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

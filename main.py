# main.py
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()

from rag import get_components, run_pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "https://ai-portfolio-chat.vercel.app",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_ready = False
try:
    get_components()
    rag_ready = True
    logger.info("RAG components loaded successfully")
except Exception as e:
    logger.error("RAG startup failed: %s", e, exc_info=True)
    logger.error("Verify OPENROUTER_API_KEY, INDEX_NAME, PINECONE_API_KEY and connectivity.")


class Message(BaseModel):
    role: str
    content: str = ""


class ChatRequest(BaseModel):
    messages: List[Message]


def _clean_messages(messages: List[Message]) -> List[Dict[str, str]]:
    cleaned = [
        {"role": "assistant" if m.role == "assistant" else "user", "content": m.content.strip()}
        for m in messages
        if m.content and m.content.strip()
    ]
    return cleaned


def _line(event: Dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


@app.post("/chat/events")
async def chat_events(data: ChatRequest):
    """NDJSON stream of real pipeline events: stages, sources, tokens, done."""
    messages = _clean_messages(data.messages)

    async def generate() -> AsyncGenerator[str, None]:
        if not rag_ready:
            yield _line({"type": "error", "message": "The agent is offline. Please try again shortly."})
            return
        if not messages or messages[-1]["role"] != "user":
            yield _line({"type": "error", "message": "Send a question to start."})
            return
        async for event in run_pipeline(messages):
            yield _line(event)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.post("/chat/stream")
async def chat_stream(data: ChatRequest):
    """Plain-text token stream (compat)."""
    messages = _clean_messages(data.messages)

    async def generate() -> AsyncGenerator[str, None]:
        if not rag_ready or not messages:
            yield "RAG chain not available"
            return
        async for event in run_pipeline(messages):
            if event["type"] == "token":
                yield event["content"]
            elif event["type"] == "error":
                yield event["message"]

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.post("/chat")
async def chat(data: ChatRequest):
    """Non-streaming answer (compat)."""
    messages = _clean_messages(data.messages)
    if not rag_ready:
        return PlainTextResponse("RAG chain not available", status_code=503)
    if not messages:
        return PlainTextResponse("No messages provided", status_code=400)

    parts: List[str] = []
    async for event in run_pipeline(messages):
        if event["type"] == "token":
            parts.append(event["content"])
        elif event["type"] == "error":
            return PlainTextResponse(event["message"], status_code=500)
    return PlainTextResponse("".join(parts))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if rag_ready else "degraded",
        "rag_available": rag_ready,
        "env": {
            "OPENROUTER_API_KEY_set": bool(os.getenv("OPENROUTER_API_KEY")),
            "INDEX_NAME_set": bool(os.getenv("INDEX_NAME")),
            "PINECONE_API_KEY_set": bool(os.getenv("PINECONE_API_KEY")),
        },
    }

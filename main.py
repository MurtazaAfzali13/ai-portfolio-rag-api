# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, AsyncGenerator, Optional
import asyncio
import json
import logging
import os

from dotenv import load_dotenv

from rag import create_retrieval_chain_with_lcel

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# CORS
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

# RAG CHAIN
missing_env = [key for key in ("OPENROUTER_API_KEY", "INDEX_NAME") if not os.getenv(key)]
if missing_env:
    logger.error("Missing required environment variables at startup: %s", ", ".join(missing_env))

try:
    chain = create_retrieval_chain_with_lcel()
    logger.info("RAG chain loaded successfully")
except Exception as e:
    logger.error("Error loading RAG chain: %s", e, exc_info=True)
    logger.error(
        "RAG startup failed. Verify OPENROUTER_API_KEY, INDEX_NAME, and OpenRouter/Pinecone connectivity."
    )
    chain = None


# Request Models
class Message(BaseModel):
    role: str
    content: str
    text: Optional[str] = None


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = "gpt-4o"
    webSearch: Optional[bool] = False


class StreamChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    model: Optional[str] = "gpt-4o"
    webSearch: Optional[bool] = False


def extract_answer(result: Any) -> str:
    """استخراج پاسخ از نتیجه RAG"""
    if result is None:
        return "متاسفانه پاسخی یافت نشد."

    if isinstance(result, dict):
        if "answer" in result:
            return str(result["answer"])
        if "result" in result:
            return str(result["result"])
        if "output" in result:
            return str(result["output"])
        if "text" in result:
            return str(result["text"])
        return json.dumps(result, ensure_ascii=False)

    if isinstance(result, str):
        return result

    return str(result)


# ✅ ENDPOINT جدید با پشتیبانی از streaming
@app.post("/chat/stream")
async def chat_stream(request: StreamChatRequest):
    logger.info(f"Received streaming request: {request.question[:50]}...")

    if chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not available")

    async def generate() -> AsyncGenerator[str, None]:
        try:
            # اجرای RAG
            logger.info(f"Invoking RAG chain with question: {request.question}")
            result = chain.invoke({"question": request.question})

            # استخراج پاسخ
            answer = extract_answer(result)
            logger.info(f"Got answer of length: {len(answer)}")

            # ارسال پاسخ به صورت تکه‌تکه
            chunk_size = 30
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.02)  # تاخیر برای افکت تایپ

        except Exception as e:
            logger.error(f"Error in streaming: {e}", exc_info=True)
            yield "Server error while processing request."

    return StreamingResponse(
        generate(),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# endpoint قبلی برای سازگاری
@app.post("/chat")
async def chat(data: ChatRequest):
    try:
        logger.info(f"Received chat request")

        if chain is None:
            return PlainTextResponse("RAG chain not available", status_code=503)

        # گرفتن آخرین پیام کاربر
        if not data.messages:
            return PlainTextResponse("No messages provided", status_code=400)

        last_message = data.messages[-1]
        question = last_message.content or last_message.text or ""

        if not question:
            return PlainTextResponse("Empty message", status_code=400)

        logger.info(f"Processing question: {question[:50]}...")

        # اجرای RAG
        result = chain.invoke({"question": question})
        answer = extract_answer(result)

        return PlainTextResponse(answer)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return PlainTextResponse("Server error while processing request.", status_code=500)


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if chain is not None else "degraded",
        "rag_available": chain is not None,
        "env": {
            "OPENROUTER_API_KEY_set": bool(os.getenv("OPENROUTER_API_KEY")),
            "INDEX_NAME_set": bool(os.getenv("INDEX_NAME")),
        },
    }
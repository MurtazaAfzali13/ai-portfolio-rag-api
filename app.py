from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from typing import List, Dict, Any

from rag import create_retrieval_chain_with_lcel

app = FastAPI()

# =========================
# CORS
# =========================
origins = [
    "http://localhost:3000",
    "https://ai-portfolio-chat.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# RAG CHAIN
# =========================
chain = create_retrieval_chain_with_lcel()

# =========================
# Request Model
# =========================
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]

# =========================
# Extract Question
# =========================
def extract_question(message: dict) -> str | None:
    if message.get("content"):
        return message["content"]

    if "parts" in message:
        return " ".join(
            part.get("text", "")
            for part in message["parts"]
            if part.get("type") == "text"
        )

    return None

# =========================
# Extract Answer from RAG
# =========================
def extract_answer(result: Any) -> str:
    if isinstance(result, dict):
        if "answer" in result:
            return result["answer"]
        if "result" in result:
            return result["result"]
        if "output" in result:
            return result["output"]
        return str(result)

    if isinstance(result, str):
        return result

    return str(result)

# =========================
# MAIN CHAT ENDPOINT
# =========================
@app.post("/chat")
async def chat(data: ChatRequest):
    try:
        messages = data.messages

        # گرفتن آخرین پیام کاربر
        user_messages = [m for m in messages if m.get("role") == "user"]

        if not user_messages:
            return PlainTextResponse("No user message found")

        question = extract_question(user_messages[-1])

        if not question:
            return PlainTextResponse("Empty message")

        # اجرای RAG
        result = chain.invoke({"question": question})

        # استخراج پاسخ
        answer = extract_answer(result)

        # 🔥 مهم: فقط متن خام برگردان
        return PlainTextResponse(answer)

    except Exception as e:
        return PlainTextResponse(f"Server error: {str(e)}")
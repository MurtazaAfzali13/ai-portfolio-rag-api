from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from main import create_retrieval_chain_with_lcel

app = FastAPI()

# =========================
# CORS (خیلی مهم)
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
    messages: list

# =========================
# Helper
# =========================
def extract_question(message: dict):

    if "content" in message and message["content"]:
        return message["content"]

    if "parts" in message:
        return " ".join(
            part.get("text", "")
            for part in message["parts"]
            if part.get("type") == "text"
        )

    return None

# =========================
# MAIN CHAT
# =========================
@app.post("/chat")
async def chat(data: ChatRequest):
    messages = data.messages

    user_messages = [
        m for m in messages
        if m.get("role") == "user"
    ]

    if not user_messages:
        return {"error": "No user message found"}

    question = extract_question(user_messages[-1])

    if not question:
        return {"error": "Empty message"}

    result = chain.invoke({"question": question})

    return {
        "content": result
    }
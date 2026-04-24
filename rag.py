# main.py
from operator import itemgetter
import logging
import os
from typing import Tuple

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

logger = logging.getLogger(__name__)


def _build_openrouter_headers() -> dict:
    """Build OpenRouter attribution headers."""
    app_url = os.getenv("OPENROUTER_SITE_URL") or os.getenv("APP_URL") or "https://railway.app"
    app_name = os.getenv("OPENROUTER_APP_NAME") or "LangChain RAG API"
    return {
        "HTTP-Referer": app_url,
        "X-Title": app_name,
    }


def _validate_required_env() -> Tuple[str, str]:
    """Validate required env vars and return resolved values."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    index_name = os.getenv("INDEX_NAME")

    if not openrouter_key:
        logger.error("Missing required environment variable: OPENROUTER_API_KEY")
    if not index_name:
        logger.error("Missing required environment variable: INDEX_NAME")

    if not openrouter_key or not index_name:
        raise RuntimeError(
            "RAG configuration is incomplete. Please set OPENROUTER_API_KEY and INDEX_NAME."
        )

    return openrouter_key, index_name


def _build_components():
    """Create LangChain/OpenRouter components with explicit error handling."""
    openrouter_key, index_name = _validate_required_env()
    default_headers = _build_openrouter_headers()

    logger.info(
        "Initializing RAG components (index=%s, openrouter_headers=%s)",
        index_name,
        list(default_headers.keys()),
    )

    embeddings = OpenAIEmbeddings(
        openai_api_key=openrouter_key,
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers=default_headers,
    )

    llm = ChatOpenAI(
        openai_api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        default_headers=default_headers,
        temperature=0,
    )

    vectorstore = PineconeVectorStore(
        index_name=index_name,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever, llm

prompt_template = ChatPromptTemplate.from_template(
"""
You are a personal AI assistant for Murtaza Afzali.

Answer using ONLY the provided context.

Important instructions:
- Focus on the MOST relevant and important information
- When possible, connect skills to real projects
- If a project is mentioned in the context, include it in your answer
- Avoid generic summaries
- Make the answer feel personal and specific

Context:
{context}

Question:
{question}

Answer:
"""
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def retrieval_chain_without_lcel(query: str):
    retriever, llm = _build_components()
    docs = retriever.invoke(query)
    context = format_docs(docs)

    messages = prompt_template.format_messages(
        context=context,
        question=query
    )

    response = llm.invoke(messages)
    return response.content


def create_retrieval_chain_with_lcel():
    retriever, llm = _build_components()

    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        ) | prompt_template | llm | StrOutputParser()
    )
    return retrieval_chain



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Retrieving from knowledge base...")

   
    query = "What is Murtaza's expertise in AI and RAG?" 
    # یا: "Tell me about his flagship video analysis project"
    # یا: "What technologies does he use for frontend?"

    print("\n" + "=" * 70)
    print("AI PORTFOLIO CHATBOT: LCEL Implementation")
    print("=" * 70)
    
    chain_with_lcel = create_retrieval_chain_with_lcel()
    
    # اجرای زنجیره با سوال مرتبط با تو
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    
    print(f"\nUser Question: {query}")
    print("\nAI Answer:")
    print(result_with_lcel)
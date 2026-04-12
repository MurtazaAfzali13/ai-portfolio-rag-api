# main.py
from operator import itemgetter
import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()


print("Initializing components...")


OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]


embeddings = OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1" # یا سرویس embedding دیگر
)


llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini"  # یا هر مدل OpenRouter
)

# 🔹 Pinecone
vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

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
    docs = retriever.invoke(query)
    context = format_docs(docs)

    messages = prompt_template.format_messages(
        context=context,
        question=query
    )

    response = llm.invoke(messages)
    return response.content

def create_retrieval_chain_with_lcel():
  
    retrieval_chain=(
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
        ) | prompt_template | llm | StrOutputParser()
    )
    return retrieval_chain



if __name__ == "__main__":
    print("Retrieving from Murtaza's Knowledge Base...")

   
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
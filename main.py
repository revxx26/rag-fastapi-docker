import os
from fastapi import FastAPI
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI(
    title="RAG Company Assistant")

loader = PyPDFLoader("sample.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "API KEY TIDAK DITEMUKAN! Set OPENAI_API_KEY atau OPENROUTER_API_KEY."
    )

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)


vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0
)

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question

    # cari dokumen relevan dulu
    docs_relevant = vector_db.similarity_search(question, k=3)

    # gabungkan jadi satu konteks
    context = "\n\n".join([doc.page_content for doc in docs_relevant])

    # kirim ke LLM
    prompt = f"""
    Jawab pertanyaan berdasarkan konteks di bawah ini.
    Jika tidak ada di konteks, bilang "Tidak ada di dokumen."

    KONTEKS:
    {context}

    PERTANYAAN:
    {question}

     """

    answer =  llm.invoke(prompt).content

    return {
        "answer": answer,
        "question": question
        }


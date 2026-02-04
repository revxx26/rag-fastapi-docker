import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ============ 1) LOAD PDF ============
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

# ============ 2) CHUNKING ============
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# ============ 3) VECTOR DB ============
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1"
)

vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

vector_db.persist()

# ============ 4) SETUP LLM ============
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini",
    temperature=0
)

# ============ 5) CHAT LOOP (simple) ============
while True:
    question = input("\nTanya tentang perusahaan (ketik exit untuk keluar): ")

    if question.lower() == "exit":
        break

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

    answer = llm.invoke(prompt).content
    print("\nJawaban AI:\n", answer)

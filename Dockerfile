FROM python:3.11-slim

WORKDIR /app

# Copy hanya file penting (bukan venv)
COPY main.py sample.pdf ./

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    langchain-community \
    langchain-openai \
    langchain-text-splitters \
    pypdf \
    chromadb \
    sentence-transformers

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

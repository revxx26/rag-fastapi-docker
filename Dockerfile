# ========== STAGE 1: install dependencies ==========
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --upgrade pip

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    langchain-community \
    langchain-openai \
    langchain-text-splitters \
    pypdf \
    chromadb \
    sentence-transformers

# ========== STAGE 2: runtime image (kecil) ==========
FROM python:3.11-slim

WORKDIR /app

# Copy hanya library yang sudah terinstall
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy file app kamu saja
COPY main.py sample.pdf ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

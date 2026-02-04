FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    fastapi uvicorn \
    langchain langchain-community langchain-openai \
    langchain-text-splitters pypdf chromadb \
    sentence-transformers

ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV OPENAI_BASE_URL=${OPENAI_BASE_URL}


EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
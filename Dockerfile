FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    fastapi uvicorn \
    langchain langchain-community langchain-openai\
    langchain-text-splitters pypdf chromadb openai


EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
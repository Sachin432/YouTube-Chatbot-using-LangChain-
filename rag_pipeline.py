import os
import requests
from pathlib import Path
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------------------------
# Load environment variables (FORCED, Windows-safe)
# -------------------------------------------------
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

if not OLLAMA_BASE_URL or not OLLAMA_MODEL:
    raise RuntimeError("Missing OLLAMA_BASE_URL or OLLAMA_MODEL in .env")

# -------------------------------------------------
# Embeddings (lightweight & stable)
# -------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -------------------------------------------------
# Prompt
# -------------------------------------------------
PROMPT = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY using the provided transcript context.
If the context is insufficient, say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"],
)

# -------------------------------------------------
# Local Ollama call (NO API KEY)
# -------------------------------------------------
def call_gemma(prompt: str) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,   # gemma3:1b
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.RequestException as e:
        raise RuntimeError(
            "Failed to call local Ollama. "
            "Ensure Ollama is running and the model is pulled."
        ) from e

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------
# Build RAG Chain
# -------------------------------------------------
def build_chain(transcript_text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([transcript_text])

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | PROMPT
        | RunnableLambda(lambda x: call_gemma(x.to_string()))
        | StrOutputParser()
    )

    return chain

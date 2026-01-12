import requests
import os
from dotenv import load_dotenv
from pathlib import Path

# Force-load .env
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=True)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")

print("OLLAMA_BASE_URL =", OLLAMA_BASE_URL)
print("OLLAMA_MODEL    =", OLLAMA_MODEL)

if not OLLAMA_BASE_URL or not OLLAMA_MODEL:
    raise RuntimeError("Missing Ollama configuration")

print("\nSending test prompt to Ollama...\n")

response = requests.post(
    f"{OLLAMA_BASE_URL}/api/generate",
    json={
        "model": OLLAMA_MODEL,
        "prompt": "Say hello in one short sentence.",
        "stream": False
    },
    timeout=120
)

print("HTTP status:", response.status_code)
print("Raw response:", response.text)

response.raise_for_status()

print("\nModel response:")
print(response.json()["response"])

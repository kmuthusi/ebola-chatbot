"""Show locally available Ollama models and test the embedding model."""

import os
from pathlib import Path

import ollama
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> None:
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:latest")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
    models = ollama.list().models
    names = [model.model for model in models]
    print("Installed models:")
    for name in names:
        print(f"  {name}")
    print(f"Embedding model: {embed_model} ({'installed' if embed_model in names else 'missing'})")
    print(f"Chat model: {chat_model} ({'installed' if chat_model in names else 'missing'})")
    embedding = ollama.embed(model=embed_model, input="health check")
    print(f"Embedding check: {len(embedding['embeddings'][0])} dimensions")


if __name__ == "__main__":
    main()

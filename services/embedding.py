"""
Embedding Service - Uses OpenAI text-embedding-3-small.
"""

from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts into vectors."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [item.embedding for item in response.data]


def embed_query(query: str) -> list[float]:
    """Embed a single query text."""
    return embed_texts([query])[0]

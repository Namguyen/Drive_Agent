"""
Vector Store Service - Qdrant for agent long-term memory.
"""

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from config import QDRANT_HOST, QDRANT_PORT, MEMORY_COLLECTION, EMBEDDING_DIM

_client = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def ensure_collection():
    """Ensure the Qdrant collection exists."""
    client = _get_client()
    if not client.collection_exists(MEMORY_COLLECTION):
        client.create_collection(
            collection_name=MEMORY_COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def save_memory(text: str, embedding: list[float], metadata: dict = None):
    """Save a single memory entry to the vector store."""
    ensure_collection()
    point_id = str(uuid.uuid4())
    _get_client().upsert(
        collection_name=MEMORY_COLLECTION,
        points=[PointStruct(id=point_id, vector=embedding, payload={"text": text, **(metadata or {})})],
    )
    return point_id


def search_memory(query_vector: list[float], top_k: int = 5) -> list[dict]:
    """Search memory by semantic similarity."""
    ensure_collection()
    hits = _get_client().query_points(
        collection_name=MEMORY_COLLECTION, query=query_vector, limit=top_k,
    ).points
    return [
        {
            "text": h.payload.get("text", ""),
            "score": round(h.score, 4),
            "metadata": {k: v for k, v in h.payload.items() if k != "text"},
        }
        for h in hits
    ]


def list_all_memories(limit: int = 100) -> list[dict]:
    """List all stored memories."""
    ensure_collection()
    points, _ = _get_client().scroll(collection_name=MEMORY_COLLECTION, limit=limit)
    return [
        {
            "id": str(p.id),
            "text": p.payload.get("text", ""),
            "metadata": {k: v for k, v in p.payload.items() if k != "text"},
        }
        for p in points
    ]

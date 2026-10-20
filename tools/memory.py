"""
Memory Tools - Long-term memory using RAG (embed + vector search).
"""

from datetime import datetime
from registry.models import ToolDefinition
from services import embedding, vectorstore


# ============================================================
# Tool 1: SAVE MEMORY
# ============================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def _chunk(text: str) -> list[str]:
    """Split long text (e.g. a whole file) so each vector covers one focused passage."""
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, max(len(text) - CHUNK_OVERLAP, 1), step)]


def save_memory(content: str, category: str = "general") -> dict:
    """Save information to long-term memory."""
    chunks = _chunk(content)
    vectors = embedding.embed_texts(chunks)
    timestamp = datetime.now().isoformat()
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        vectorstore.save_memory(chunk, vector, {
            "category": category, "timestamp": timestamp, "chunk_index": i,
        })
    return {
        "status": "saved",
        "content_preview": content[:100],
        "category": category,
        "chunks": len(chunks),
    }


save_memory_tool = ToolDefinition(
    name="save_memory",
    description=(
        "Save important information to long-term memory for future reference. "
        "Use this to remember facts, user preferences, conversation insights, "
        "or any information that should persist across conversations. "
        "Provide a category to organize memories (e.g., 'user_preference', 'fact', 'task')."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The information to remember.",
            },
            "category": {
                "type": "string",
                "description": "Category for organizing the memory (e.g., 'user_preference', 'fact', 'task', 'general').",
            },
        },
        "required": ["content"],
    },
    required_scopes=["memory:write"],
    handler=save_memory,
)


# ============================================================
# Tool 2: SEARCH MEMORY
# ============================================================

def search_memory(query: str, top_k: int = 5) -> dict:
    """Search long-term memory for relevant information."""
    memories = vectorstore.search_memory(embedding.embed_query(query), top_k)
    return {"query": query, "results_count": len(memories), "memories": memories}


search_memory_tool = ToolDefinition(
    name="search_memory",
    description=(
        "Search long-term memory for previously saved information. "
        "Uses semantic search to find the most relevant memories. "
        "Use this to recall facts, user preferences, or past interactions."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant memories.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (default: 5).",
            },
        },
        "required": ["query"],
    },
    required_scopes=["memory:read"],
    handler=search_memory,
)


ALL_MEMORY_TOOLS = [save_memory_tool, search_memory_tool]

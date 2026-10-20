"""
File Reader Service - Convert files to Markdown using MarkItDown.
"""

import os
from markitdown import MarkItDown

MAX_CHARS = 15000


def read_file(file_path: str) -> dict:
    """Read a local file and convert its content to Markdown using MarkItDown."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: '{file_path}'")

    content = MarkItDown().convert(file_path).text_content or ""
    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS] + f"\n\n[... truncated, total {len(content)} chars]"
    return {"file_name": os.path.basename(file_path), "content": content}

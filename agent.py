"""
Agent Core - Connects Claude LLM with the Tool Registry.
Handles the full conversation loop with tool use.
"""

import json
from collections.abc import Iterator

import anthropic
from config import LLM_MODEL
from registry import ToolRegistry, ToolDefinition

from tools.google_drive import ALL_DRIVE_TOOLS
from tools.read_file import ALL_READ_FILE_TOOLS
from tools.memory import ALL_MEMORY_TOOLS

ALL_TOOLS: list[ToolDefinition] = ALL_DRIVE_TOOLS + ALL_READ_FILE_TOOLS + ALL_MEMORY_TOOLS

SYSTEM_PROMPT = """\
You are a powerful AI assistant with access to the following capabilities:

1. **Google Drive**: You can list all files and read their contents (supports many formats: PDF, DOCX, XLSX, PPTX, images, etc.).
2. **Long-term Memory**: You can save and search information across conversations using semantic search (RAG).

Guidelines:
- When the user asks to see files from Google Drive, use list_drive_files first, then read_drive_file for each file.
- Proactively save important information to memory (user preferences, key facts, task results).
- Before answering questions about past interactions, user preferences, or concepts from previously saved files, search_memory first.
- When the user asks to save/remember a Drive file's content, call save_drive_file_to_memory with its file ID. Never copy file content into save_memory yourself.
- read_drive_file returns at most 15000 chars; for questions about a long file, save it to memory and use search_memory.
- When showing a file's content, include the content itself in your reply, not just a summary.
- If a tool returns an error (e.g. missing permission), explain it to the user plainly.
- Always respond in the same language as the user's message.
- Be concise and helpful.
"""


class Agent:
    def __init__(self, service_api_key: str = "sk-admin-001"):
        self.client = anthropic.Anthropic()
        self.model = LLM_MODEL
        self.service_api_key = service_api_key
        self.conversation_history: list[dict] = []

        # Initialize Tool Registry with all tools
        self.registry = ToolRegistry()
        for tool in ALL_TOOLS:
            self.registry.register(tool)

    def get_tools_for_claude(self) -> list[dict]:
        return self.registry.list_tools()

    def run(self, user_message: str) -> str:
        """Run the agent loop and return the full text response."""
        return "".join(
            event["delta"] for event in self.run_stream(user_message)
            if event["type"] == "text"
        )

    def run_stream(self, user_message: str) -> Iterator[dict]:
        """
        Run the full agent loop, yielding events as they happen:
        1. Add user message to history
        2. Stream Claude's reply → {"type": "text", "delta": ...}
        3. If Claude uses tools → run through Registry pipeline
           → {"type": "tool", "name": ..., "status": "success" | "failed"}
        4. Loop until Claude gives final text response
        """
        print(f"\n{'#'*60}")
        print(f"  USER: {user_message}")
        print(f"{'#'*60}")

        self.conversation_history.append({
            "role": "user",
            "content": user_message,
        })

        tools = self.get_tools_for_claude()

        while True:
            print(f"\n>>> Calling Claude LLM (stream)...")
            with self.client.messages.stream(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=self.conversation_history,
            ) as stream:
                for delta in stream.text_stream:
                    yield {"type": "text", "delta": delta}
                response = stream.get_final_message()

            print(f">>> Claude stop_reason: {response.stop_reason}")

            self.conversation_history.append({
                "role": "assistant",
                "content": response.content,
            })

            # Final response - anything but tool_use (end_turn, max_tokens, ...) ends the loop
            if response.stop_reason != "tool_use":
                print(f"\n{'#'*60}")
                print(f"  ASSISTANT: done ({response.stop_reason})")
                print(f"{'#'*60}\n")
                return

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    # Run through the 6-step Registry pipeline
                    result = self.registry.call(
                        tool_name=block.name,
                        arguments=block.input,
                        api_key=self.service_api_key,
                    )
                    yield {
                        "type": "tool",
                        "name": block.name,
                        "status": "failed" if "error" in result else "success",
                    }

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False, default=str),
                    })

            self.conversation_history.append({
                "role": "user",
                "content": tool_results,
            })

    def clear_history(self):
        self.conversation_history = []
        print("[Agent] Conversation history cleared.")

    def get_audit_log(self) -> list[dict]:
        return self.registry.get_audit_log()

"""
Tool Registry - 6-step pipeline (run order):
1. Validate Schema
2. Check Authentication
3. Check Scopes
4. Check Rate Limit
5. Execute Tool
6. Audit Log
"""

import time
import json
from datetime import datetime
from typing import Any
from .models import ToolDefinition


# ============================================================
# Step 1: VALIDATE SCHEMA
# ============================================================

def validate_schema(tool: ToolDefinition, arguments: dict) -> dict:
    """Validate tool arguments against the tool's input_schema."""
    properties = tool.input_schema.get("properties", {})
    for field in tool.input_schema.get("required", []):
        if field not in arguments:
            raise ValueError(f"Missing required field: '{field}'")

    type_map = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for field, value in arguments.items():
        spec = properties.get(field)
        if spec is None:
            continue
        expected = type_map.get(spec.get("type"))
        # bool is a subclass of int, so reject it explicitly for integer/number
        if expected and (not isinstance(value, expected)
                         or (isinstance(value, bool) and spec["type"] != "boolean")):
            raise ValueError(f"Field '{field}' must be {spec['type']}, got {type(value).__name__}")
        if "enum" in spec and value not in spec["enum"]:
            raise ValueError(f"Field '{field}' must be one of {spec['enum']}")
    return arguments


# ============================================================
# Step 2: CHECK AUTHENTICATION
# ============================================================

USER_DB: dict[str, dict] = {
    "sk-admin-001": {
        "user_id": "user_admin",
        "role": "admin",
        "scopes": [
            "drive:read",
            "memory:read", "memory:write",
        ],
    },
    "sk-user-002": {
        "user_id": "user_standard",
        "role": "user",
        "scopes": [
            "drive:read",
            "memory:read",
        ],
    },
    "sk-guest-003": {
        "user_id": "user_guest",
        "role": "guest",
        "scopes": ["drive:read", "memory:read"],
    },
}


def check_authentication(api_key: str) -> dict:
    """Authenticate user by API key."""
    if not api_key:
        raise PermissionError("API key is required")
    user = USER_DB.get(api_key)
    if user is None:
        raise PermissionError("Invalid API key")
    return user


# ============================================================
# Step 3: CHECK SCOPES
# ============================================================

def check_scopes(user: dict, tool: ToolDefinition) -> bool:
    """Check if user has required scopes for the tool."""
    missing = set(tool.required_scopes) - set(user["scopes"])
    if missing:
        raise PermissionError(
            f"User '{user['user_id']}' ({user['role']}) lacks scopes: {sorted(missing)}"
        )
    return True


# ============================================================
# Step 4: RATE LIMIT
# ============================================================

class RateLimiter:
    def __init__(self, max_calls: int = 10, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = {}

    def check(self, user_id: str) -> bool:
        """Check if user is within rate limit."""
        now = time.time()
        bucket = [t for t in self._buckets.get(user_id, []) if now - t < self.window_seconds]
        if len(bucket) >= self.max_calls:
            self._buckets[user_id] = bucket
            raise RuntimeError(
                f"Rate limit exceeded: {self.max_calls} calls per {self.window_seconds}s"
            )
        bucket.append(now)
        self._buckets[user_id] = bucket
        return True


rate_limiter = RateLimiter(max_calls=20, window_seconds=60)


# ============================================================
# Step 5: AUDIT LOG
# ============================================================

AUDIT_LOG: list[dict] = []


def audit_log(user: dict, tool_name: str, arguments: dict, result: Any = None, error: str = None,
              steps: list[dict] = None):
    """Record tool call in the audit log."""
    user = user or {}
    # Truncate so large file contents don't bloat the log
    result_preview = None if result is None else json.dumps(result, ensure_ascii=False, default=str)[:500]
    entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user.get("user_id", "unknown"),
        "role": user.get("role", "unknown"),
        "tool": tool_name,
        "arguments": arguments,
        "result": result_preview,
        "error": error,
        "status": "failed" if error else "success",
        "steps": steps or [],
    }
    AUDIT_LOG.append(entry)
    return entry


# ============================================================
# Step 6: EXECUTE TOOL
# ============================================================

def execute_tool(tool: ToolDefinition, arguments: dict) -> Any:
    """Execute the tool handler with validated arguments."""
    return tool.handler(**arguments)


# ============================================================
# TOOL REGISTRY
# ============================================================

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        self._tools[tool.name] = tool
        print(f"[Registry] Registered tool: '{tool.name}'")

    def get_tool(self, name: str) -> ToolDefinition:
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"[Registry] Tool '{name}' not found. Available: {list(self._tools.keys())}")
        return tool

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def call(self, tool_name: str, arguments: dict, api_key: str) -> dict:
        """
        Run the 6-step pipeline: Schema → Auth → Scopes → Rate Limit → Execute → Audit.
        """
        print(f"\n{'='*60}")
        print(f"  TOOL CALL: {tool_name}")
        print(f"  Arguments: {json.dumps(arguments, ensure_ascii=False)}")
        print(f"{'='*60}")

        steps: list[dict] = []
        user = None
        result = None
        error_msg = None
        current = "validate_schema"

        def ok(name: str):
            steps.append({"step": len(steps) + 1, "name": name, "status": "ok"})
            print(f"  [Step {len(steps)}] {name}: OK")

        try:
            tool = self.get_tool(tool_name)
            validate_schema(tool, arguments)
            ok(current)

            current = "check_authentication"
            user = check_authentication(api_key)
            ok(current)

            current = "check_scopes"
            check_scopes(user, tool)
            ok(current)

            current = "rate_limit"
            rate_limiter.check(user["user_id"])
            ok(current)

            current = "execute_tool"
            result = execute_tool(tool, arguments)
            ok(current)
        except Exception as e:
            error_msg = f"{type(e).__name__}: {e}"
            steps.append({"step": len(steps) + 1, "name": current, "status": "failed", "error": error_msg})
            print(f"  [Step {len(steps)}] {current}: FAILED - {error_msg}")

        # Step 6 always runs, so failed/denied calls are audited too
        steps.append({"step": 6, "name": "audit_log", "status": "ok"})
        audit_log(user, tool_name, arguments, result=result, error=error_msg, steps=steps)
        print("  [Step 6] audit_log: OK")

        return {"error": error_msg} if error_msg else {"result": result}

    def get_audit_log(self) -> list[dict]:
        return AUDIT_LOG

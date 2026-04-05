import logging
from typing import Any, Dict

from reachy_mini_conversation_app.memory import MemoryStore, get_default_memory_user_id
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class ForgetMemory(Tool):
    """Delete a saved durable fact from memory."""

    name = "forget_memory"
    description = (
        "Forget one or more saved memories when the user asks Reachy to stop remembering something "
        "or to correct outdated information."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "integer",
                "description": "Specific memory ID to remove, if known.",
            },
            "query": {
                "type": "string",
                "description": "Search phrase to match memories that should be forgotten.",
            },
            "category": {
                "type": "string",
                "description": "Optional category filter when forgetting by query.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to forget when using a query.",
                "minimum": 1,
                "maximum": 20,
            },
            "user_id": {
                "type": "string",
                "description": "Optional memory owner ID. Omit for the default household user.",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Forget one or more memories from the local store."""
        default_user_id = deps.active_memory_user_id or get_default_memory_user_id()
        user_id = (kwargs.get("user_id") or default_user_id).strip() or default_user_id
        limit = int(kwargs.get("limit") or 10)
        store = MemoryStore()

        try:
            forgotten = store.forget_memories(
                user_id=user_id,
                memory_id=kwargs.get("memory_id"),
                query=kwargs.get("query"),
                category=kwargs.get("category"),
                limit=limit,
            )
        except ValueError as exc:
            return {"error": str(exc)}

        logger.info(
            "Tool call: forget_memory user_id=%s memory_id=%s query=%s",
            user_id,
            kwargs.get("memory_id"),
            (kwargs.get("query") or "")[:160],
        )
        return {"count": len(forgotten), "forgotten": [entry.as_dict() for entry in forgotten]}

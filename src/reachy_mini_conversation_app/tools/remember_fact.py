import logging
from typing import Any, Dict

from reachy_mini_conversation_app.memory import (
    MEMORY_CATEGORIES,
    MemoryStore,
    get_default_memory_user_id,
)
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class RememberFact(Tool):
    """Persist a durable user fact or preference."""

    name = "remember_fact"
    description = (
        "Save a durable fact for future conversations, such as the user's name, preferences, "
        "relationships, routines, ongoing projects, or explicit remember-this requests. "
        "Do not use for fleeting small talk, secrets, or one-off details."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The durable fact to remember in concise natural language.",
            },
            "category": {
                "type": "string",
                "enum": list(MEMORY_CATEGORIES),
                "description": "The type of memory being stored.",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional short tags for later retrieval.",
            },
            "user_id": {
                "type": "string",
                "description": "Optional memory owner ID. Omit for the default household user.",
            },
        },
        "required": ["fact"],
        "additionalProperties": False,
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Save a durable fact in the local memory store."""
        fact = (kwargs.get("fact") or "").strip()
        if not fact:
            return {"error": "fact must be a non-empty string"}

        default_user_id = deps.active_memory_user_id or get_default_memory_user_id()
        user_id = (kwargs.get("user_id") or default_user_id).strip() or default_user_id
        store = MemoryStore()
        action, entry = store.remember_fact(
            user_id=user_id,
            fact=fact,
            category=kwargs.get("category"),
            tags=kwargs.get("tags"),
        )
        logger.info("Tool call: remember_fact user_id=%s fact=%s", user_id, fact[:160])
        return {"status": action, "memory": entry.as_dict()}

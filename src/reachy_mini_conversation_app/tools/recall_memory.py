import logging
from typing import Any, Dict

from reachy_mini_conversation_app.memory import MemoryStore, get_default_memory_user_id
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)


class RecallMemory(Tool):
    """Search durable memories from past conversations."""

    name = "recall_memory"
    description = (
        "Search saved long-term memories from earlier conversations. Use this when prior facts about the "
        "user, household, preferences, or ongoing projects may help."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Optional search phrase. Leave empty to list the most recent memories.",
            },
            "category": {
                "type": "string",
                "description": "Optional category filter such as identity, preference, project, or relationship.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return.",
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
        """Return relevant memories from the local memory store."""
        default_user_id = deps.active_memory_user_id or get_default_memory_user_id()
        user_id = (kwargs.get("user_id") or default_user_id).strip() or default_user_id
        limit = int(kwargs.get("limit") or 5)
        store = MemoryStore()
        memories = store.recall_memories(
            user_id=user_id,
            query=kwargs.get("query"),
            category=kwargs.get("category"),
            limit=limit,
        )
        logger.info("Tool call: recall_memory user_id=%s query=%s", user_id, (kwargs.get("query") or "")[:160])
        return {"count": len(memories), "memories": [entry.as_dict() for entry in memories]}

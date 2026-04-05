from pathlib import Path
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.forget_memory import ForgetMemory
from reachy_mini_conversation_app.tools.recall_memory import RecallMemory
from reachy_mini_conversation_app.tools.remember_fact import RememberFact


@pytest.mark.asyncio
async def test_memory_tools_save_recall_and_forget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Memory tools should provide a complete CRUD flow."""
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_MEMORY_DB_PATH", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_MEMORY_USER_ID", "walter")

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())

    remember_result = await RememberFact()(
        deps,
        fact="Walter likes short answers.",
        category="preference",
        tags=["style"],
    )
    assert remember_result["status"] == "saved"

    recall_result = await RecallMemory()(deps, query="short")
    assert recall_result["count"] == 1
    assert recall_result["memories"][0]["fact"] == "Walter likes short answers."

    forget_result = await ForgetMemory()(deps, memory_id=recall_result["memories"][0]["id"])
    assert forget_result["count"] == 1

    recall_after_forget = await RecallMemory()(deps, query="short")
    assert recall_after_forget == {"count": 0, "memories": []}


@pytest.mark.asyncio
async def test_memory_tools_use_active_memory_user_from_deps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Memory tools should default to the handler's active user, not only the global config."""
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_MEMORY_DB_PATH", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_MEMORY_USER_ID", "default")

    deps = ToolDependencies(
        reachy_mini=MagicMock(),
        movement_manager=MagicMock(),
        active_memory_user_id="marie",
    )

    remember_result = await RememberFact()(deps, fact="Marie likes tea.", category="preference")
    assert remember_result["memory"]["user_id"] == "marie"

    recall_result = await RecallMemory()(deps, query="tea")
    assert recall_result["count"] == 1
    assert recall_result["memories"][0]["user_id"] == "marie"

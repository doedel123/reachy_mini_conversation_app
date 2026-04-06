from pathlib import Path

import pytest

import reachy_mini_conversation_app.prompts as prompts_mod
from reachy_mini_conversation_app.memory import MemoryStore, normalize_memory_user_id
from reachy_mini_conversation_app.config import config


def test_memory_store_round_trip(tmp_path: Path) -> None:
    """Facts should be saved, found, updated, and forgotten."""
    store = MemoryStore(tmp_path / "memory.sqlite3")

    action, entry = store.remember_fact(
        user_id="walter",
        fact="Walter prefers concise answers.",
        category="preference",
        tags=["style"],
    )
    assert action == "saved"
    assert entry.id > 0

    action, updated = store.remember_fact(
        user_id="walter",
        fact="Walter prefers concise answers.",
        category="preference",
        tags=["reachy", "style"],
    )
    assert action == "updated"
    assert updated.id == entry.id
    assert updated.tags == ["reachy", "style"]

    recalled = store.recall_memories(user_id="walter", query="concise")
    assert [memory.fact for memory in recalled] == ["Walter prefers concise answers."]

    forgotten = store.forget_memories(user_id="walter", memory_id=entry.id)
    assert [memory.id for memory in forgotten] == [entry.id]
    assert store.recall_memories(user_id="walter", query="concise") == []


def test_prompt_includes_stored_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Session instructions should append a compact stored-memory section."""
    store = MemoryStore(tmp_path / "memory.sqlite3")
    store.remember_fact(
        user_id="walter",
        fact="Walter uses Home Assistant.",
        category="project",
        tags=["home-assistant"],
    )

    monkeypatch.setattr(config, "REACHY_MINI_MEMORY_DB_PATH", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setattr(config, "REACHY_MINI_MEMORY_USER_ID", "walter")
    monkeypatch.setattr(config, "REACHY_MINI_MEMORY_PROMPT_LIMIT", 12)
    monkeypatch.setattr(config, "REACHY_MINI_CUSTOM_PROFILE", None)

    prompt = prompts_mod.get_session_instructions()

    assert "## STORED MEMORY" in prompt
    assert "Walter uses Home Assistant." in prompt


def test_normalize_memory_user_id() -> None:
    """Spoken names should normalize into stable multi-user IDs."""
    assert normalize_memory_user_id("Walter Voll") == "walter-voll"
    assert normalize_memory_user_id("  Marie-Claire  ") == "marie-claire"
    assert normalize_memory_user_id("") == config.REACHY_MINI_MEMORY_USER_ID


def test_memory_store_persists_last_active_user(tmp_path: Path) -> None:
    """The most recently identified user should survive across app restarts."""
    store = MemoryStore(tmp_path / "memory.sqlite3")

    assert store.get_last_active_user_id(default="default") == "default"

    saved_user_id = store.set_last_active_user_id("Walter Voll")
    assert saved_user_id == "walter-voll"

    reopened = MemoryStore(tmp_path / "memory.sqlite3")
    assert reopened.get_last_active_user_id(default="default") == "walter-voll"

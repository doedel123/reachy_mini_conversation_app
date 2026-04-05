import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.memory import MemoryStore
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _build_handler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> rt_mod.OpenaiRealtimeHandler:
    monkeypatch.setattr(rt_mod.config, "REACHY_MINI_MEMORY_DB_PATH", str(tmp_path / "memory.sqlite3"))
    monkeypatch.setattr(rt_mod.config, "REACHY_MINI_MEMORY_USER_ID", "default")
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return rt_mod.OpenaiRealtimeHandler(deps)


@pytest.mark.parametrize(
    ("transcript", "expected"),
    [
        ("ich bin Walter", "Walter"),
        ("ich heiße Walter", "Walter"),
        ("mein name ist Marie-Claire", "Marie-Claire"),
        ("I am Alice", "Alice"),
        ("je suis Jean", "Jean"),
        ("das ist Walter", None),
    ],
)
def test_extract_introduced_name(transcript: str, expected: str | None) -> None:
    """Explicit self-introductions should trigger multi-user memory switching."""
    assert rt_mod._extract_introduced_name(transcript) == expected


@pytest.mark.asyncio
async def test_switch_active_memory_user_updates_session_and_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A spoken name should switch the active user and persist identity memory."""
    handler = _build_handler(tmp_path, monkeypatch)
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda user_id=None: f"instructions:{user_id}")
    monkeypatch.setattr(rt_mod, "get_session_voice", lambda: "cedar")

    session_update = AsyncMock()
    handler.connection = SimpleNamespace(session=SimpleNamespace(update=session_update))

    user_id = await handler._switch_active_memory_user("Walter")

    assert user_id == "walter"
    assert handler.active_memory_user_id == "walter"
    assert handler.deps.active_memory_user_id == "walter"
    session_update.assert_awaited_once()

    store = MemoryStore(tmp_path / "memory.sqlite3")
    memories = store.recall_memories(user_id="walter", category="identity")
    assert [memory.fact for memory in memories] == ["The user's name is Walter."]


@pytest.mark.asyncio
async def test_memory_summarizer_stores_durable_facts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Completed turns should be condensed into durable memories with the cheap side model."""
    handler = _build_handler(tmp_path, monkeypatch)
    monkeypatch.setattr(rt_mod.config, "REACHY_MINI_MEMORY_SUMMARIZER_MODEL", "gpt-5-mini")

    class FakeResponses:
        async def create(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                output_text='{"save":[{"fact":"Walter prefers concise answers.","category":"preference","tags":["style"]}]}'
            )

    handler.client = SimpleNamespace(responses=FakeResponses())
    handler.active_memory_user_id = "walter"
    handler.deps.active_memory_user_id = "walter"

    await handler._summarize_and_store_memory(
        memory_user_id="walter",
        user_transcript="Ich mag kurze Antworten.",
        assistant_transcript="Verstanden, ich halte mich kurz.",
    )

    store = MemoryStore(tmp_path / "memory.sqlite3")
    memories = store.recall_memories(user_id="walter", query="concise")
    assert [memory.fact for memory in memories] == ["Walter prefers concise answers."]


@pytest.mark.asyncio
async def test_memory_summary_task_runs_in_background(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Scheduling the summarizer should not block the handler."""
    handler = _build_handler(tmp_path, monkeypatch)
    invoked = asyncio.Event()

    async def fake_summarize(*, memory_user_id: str, user_transcript: str, assistant_transcript: str) -> None:
        assert memory_user_id == "default"
        assert user_transcript == "Ich heiße Walter."
        assert assistant_transcript == "Freut mich, Walter."
        invoked.set()

    handler._summarize_and_store_memory = fake_summarize  # type: ignore[method-assign]

    handler._schedule_memory_summary(
        user_transcript="Ich heiße Walter.",
        assistant_transcript="Freut mich, Walter.",
    )

    await asyncio.wait_for(invoked.wait(), timeout=1.0)

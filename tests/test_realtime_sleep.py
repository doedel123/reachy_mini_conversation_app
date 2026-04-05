import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _build_handler() -> rt_mod.OpenaiRealtimeHandler:
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return rt_mod.OpenaiRealtimeHandler(deps)


def test_voice_activity_detected_after_multiple_loud_frames() -> None:
    """Sleeping sessions should only wake after consecutive loud frames."""
    handler = _build_handler()
    loud = np.ones(1024, dtype=np.float32) * 0.2

    assert handler._voice_activity_detected(loud) is False
    assert handler._voice_activity_detected(loud) is True


def test_should_auto_sleep_session_honors_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idle sleep should depend on configured timeout and pending work."""
    handler = _build_handler()
    monkeypatch.setattr(rt_mod.config, "REACHY_MINI_IDLE_SESSION_TIMEOUT_S", 300.0)

    assert handler._should_auto_sleep_session(100.0) is False
    assert handler._should_auto_sleep_session(301.0) is True

    handler._response_done_event.clear()
    assert handler._should_auto_sleep_session(600.0) is False


@pytest.mark.asyncio
async def test_receive_reconnects_on_voice_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    """When sleeping, incoming speech should request a reconnect."""
    handler = _build_handler()
    handler.connection = None
    handler._connected_event.set()
    ensure_connection = AsyncMock(return_value=True)
    handler._ensure_connection_for_voice_activity = ensure_connection  # type: ignore[method-assign]

    loud = np.ones(1024, dtype=np.float32) * 0.2
    await handler.receive((24000, loud))
    await handler.receive((24000, loud))

    ensure_connection.assert_awaited_once()

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


def test_voice_activity_detected_honors_configured_wake_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wake-up sensitivity should follow the dedicated wake threshold config."""
    handler = _build_handler()
    medium = np.ones(1024, dtype=np.float32) * 0.06

    monkeypatch.setattr(rt_mod.config, "REACHY_MINI_WAKE_DB_THRESHOLD", -20.0)
    assert handler._voice_activity_detected(medium) is False
    assert handler._voice_activity_detected(medium) is False

    monkeypatch.setattr(rt_mod.config, "REACHY_MINI_WAKE_DB_THRESHOLD", -26.0)
    handler._wake_above_threshold_frames = 0
    assert handler._voice_activity_detected(medium) is False
    assert handler._voice_activity_detected(medium) is True


def test_should_auto_sleep_session_honors_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idle sleep should depend on configured timeout and pending work."""
    handler = _build_handler()
    monkeypatch.setattr(rt_mod.config, "REACHY_MINI_IDLE_SESSION_TIMEOUT_S", 300.0)

    assert handler._should_auto_sleep_session(100.0) is False
    assert handler._should_auto_sleep_session(301.0) is True

    handler._response_done_event.clear()
    assert handler._should_auto_sleep_session(600.0) is False


@pytest.mark.asyncio
async def test_sleep_session_sets_sleep_flag_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Auto-sleep should mark the RTC as sleeping and emit a clear log line."""
    handler = _build_handler()
    handler.connection = MagicMock()
    handler.connection.close = AsyncMock()

    with caplog.at_level("INFO"):
        await handler._sleep_session_due_to_inactivity(321.0)

    assert handler._rtc_sleeping is True
    handler.connection.close.assert_awaited_once()
    assert "RTC auto-sleep" in caplog.text


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


@pytest.mark.asyncio
async def test_ensure_connection_for_voice_activity_clears_sleep_flag() -> None:
    """Successful wake-up should clear the sleeping flag."""
    handler = _build_handler()
    handler._rtc_sleeping = True
    handler.client = MagicMock()
    handler.connection = MagicMock()
    handler._connected_event.set()

    connected = await handler._ensure_connection_for_voice_activity()

    assert connected is True
    assert handler._rtc_sleeping is False


@pytest.mark.asyncio
async def test_receive_does_not_mark_activity_from_raw_frames_while_connected() -> None:
    """While connected, raw mic frames should not reset idle timers by themselves."""
    handler = _build_handler()
    handler.connection = MagicMock()
    handler.connection.input_audio_buffer = MagicMock()
    handler.connection.input_audio_buffer.append = AsyncMock()
    handler.last_activity_time = 1.0

    await handler.receive((24000, np.ones(1024, dtype=np.int16) * 12000))
    await handler.receive((24000, np.ones(1024, dtype=np.int16) * 12000))

    assert handler.last_activity_time == 1.0


@pytest.mark.asyncio
async def test_receive_does_not_mark_activity_for_silence() -> None:
    """Silent mic frames should not keep the realtime session artificially awake."""
    handler = _build_handler()
    handler.connection = MagicMock()
    handler.connection.input_audio_buffer = MagicMock()
    handler.connection.input_audio_buffer.append = AsyncMock()
    handler.last_activity_time = 5.0

    await handler.receive((24000, np.zeros(1024, dtype=np.int16)))

    assert handler.last_activity_time == 5.0


@pytest.mark.asyncio
async def test_receive_does_not_mark_activity_for_single_noise_spike() -> None:
    """One noisy frame should not keep the session alive indefinitely."""
    handler = _build_handler()
    handler.connection = MagicMock()
    handler.connection.input_audio_buffer = MagicMock()
    handler.connection.input_audio_buffer.append = AsyncMock()
    handler.last_activity_time = 9.0

    await handler.receive((24000, np.ones(1024, dtype=np.int16) * 12000))

    assert handler.last_activity_time == 9.0

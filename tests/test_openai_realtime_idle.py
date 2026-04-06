import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.tools.background_tool_manager import ToolNotification
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.tool_constants import ToolState


@pytest.mark.asyncio
async def test_emit_skips_idle_when_response_is_pending() -> None:
    """Idle behavior must not interrupt a queued assistant follow-up."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    deps.movement_manager.is_idle.return_value = True

    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler.last_activity_time = asyncio.get_event_loop().time() - 30.0
    await handler._safe_response_create(response={"instructions": "answer now"})

    send_idle_signal = MagicMock()
    handler.send_idle_signal = send_idle_signal  # type: ignore[method-assign]

    emit_task = asyncio.create_task(handler.emit())
    await asyncio.sleep(0.05)
    emit_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await emit_task

    send_idle_signal.assert_not_called()


@pytest.mark.asyncio
async def test_emit_skips_idle_while_tool_is_running() -> None:
    """Idle behavior must not start while a background tool is still running."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    deps.movement_manager.is_idle.return_value = True

    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler.last_activity_time = asyncio.get_event_loop().time() - 30.0
    object.__setattr__(handler.tool_manager, "get_running_tools", MagicMock(return_value=[MagicMock()]))

    send_idle_signal = MagicMock()
    handler.send_idle_signal = send_idle_signal  # type: ignore[method-assign]

    emit_task = asyncio.create_task(handler.emit())
    await asyncio.sleep(0.05)
    emit_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await emit_task

    send_idle_signal.assert_not_called()


@pytest.mark.asyncio
async def test_emit_idle_signal_does_not_reset_activity_clock() -> None:
    """Idle gestures should not count as fresh conversation activity."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    deps.movement_manager.is_idle.return_value = True

    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler.last_activity_time = asyncio.get_event_loop().time() - 20.0
    send_idle_signal = AsyncMock()
    handler.send_idle_signal = send_idle_signal  # type: ignore[method-assign]

    emit_task = asyncio.create_task(handler.emit())
    await asyncio.sleep(0.05)
    emit_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await emit_task

    send_idle_signal.assert_awaited_once()
    assert handler._last_idle_signal_time > 0.0
    assert asyncio.get_event_loop().time() - handler.last_activity_time >= 20.0


@pytest.mark.asyncio
async def test_idle_tool_result_does_not_mark_recent_activity() -> None:
    """Idle-triggered tool completions must not postpone RTC auto-sleep."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler.last_activity_time = 1.0
    handler.connection = SimpleNamespace(
        conversation=SimpleNamespace(item=SimpleNamespace(create=AsyncMock())),
    )

    await handler._handle_tool_result(
        ToolNotification(
            id="idle_call",
            tool_name="play_emotion",
            is_idle_tool_call=True,
            status=ToolState.COMPLETED,
            result={"status": "queued"},
        )
    )

    assert handler.last_activity_time == 1.0


@pytest.mark.asyncio
async def test_emit_skips_idle_when_idle_behavior_is_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Idle gestures should be fully disabled when configured off."""
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    deps.movement_manager.is_idle.return_value = True

    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler.last_activity_time = asyncio.get_event_loop().time() - 30.0
    monkeypatch.setattr(rt_mod.config, "REACHY_MINI_ENABLE_IDLE_BEHAVIOR", False)

    send_idle_signal = AsyncMock()
    handler.send_idle_signal = send_idle_signal  # type: ignore[method-assign]

    emit_task = asyncio.create_task(handler.emit())
    await asyncio.sleep(0.05)
    emit_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await emit_task

    send_idle_signal.assert_not_called()

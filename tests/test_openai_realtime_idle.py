import asyncio
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


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

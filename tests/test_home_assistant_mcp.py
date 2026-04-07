import pytest

import reachy_mini_conversation_app.config as config_mod
from reachy_mini_conversation_app.tools.core_tools import get_tool_specs


def test_get_tool_specs_appends_home_assistant_mcp_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The optional Home Assistant MCP server should be exposed as a remote tool when enabled."""
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_HOME_ASSISTANT_MCP_ENABLED", True)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_HOME_ASSISTANT_MCP_URL", "https://ha.example/api/mcp")
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_HOME_ASSISTANT_MCP_TOKEN", "secret-token")
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_HOME_ASSISTANT_MCP_SERVER_LABEL", "home_assistant")
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_HOME_ASSISTANT_MCP_ALLOWED_TOOLS", "get_state,call_service")
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_HOME_ASSISTANT_MCP_REQUIRE_APPROVAL", "never")

    specs = get_tool_specs()
    mcp_specs = [spec for spec in specs if spec.get("type") == "mcp"]

    assert len(mcp_specs) == 1
    assert mcp_specs[0]["server_label"] == "home_assistant"
    assert mcp_specs[0]["server_url"] == "https://ha.example/api/mcp"
    assert mcp_specs[0]["headers"] == {"Authorization": "Bearer secret-token"}
    assert mcp_specs[0]["allowed_tools"] == ["get_state", "call_service"]
    assert mcp_specs[0]["require_approval"] == "never"
    assert "server_description" not in mcp_specs[0]


def test_get_tool_specs_omits_home_assistant_mcp_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """No remote MCP server should be advertised unless the feature is enabled."""
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_HOME_ASSISTANT_MCP_ENABLED", False)
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_HOME_ASSISTANT_MCP_URL", "https://ha.example/api/mcp")
    monkeypatch.setattr(config_mod.config, "REACHY_MINI_HOME_ASSISTANT_MCP_TOKEN", "secret-token")

    specs = get_tool_specs()

    assert all(spec.get("type") != "mcp" for spec in specs)

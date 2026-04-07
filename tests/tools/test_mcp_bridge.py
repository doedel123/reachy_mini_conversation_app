import json

import httpx
import pytest

from reachy_mini_conversation_app.tools.mcp_bridge import MCPBridge, _parse_jsonrpc_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bridge(**kwargs):
    defaults = dict(server_url="https://ha.test/api/mcp", token="tok", server_label="ha")
    defaults.update(kwargs)
    return MCPBridge(**defaults)


_DUMMY_REQUEST = httpx.Request("POST", "https://ha.test/api/mcp")


def _json_response(body: dict, status_code: int = 200, headers: dict | None = None) -> httpx.Response:
    content = json.dumps(body).encode()
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers={"content-type": "application/json", **(headers or {})},
        request=_DUMMY_REQUEST,
    )


def _sse_response(body: dict, status_code: int = 200, headers: dict | None = None) -> httpx.Response:
    text = f"data: {json.dumps(body)}\n\n"
    return httpx.Response(
        status_code=status_code,
        content=text.encode(),
        headers={"content-type": "text/event-stream", **(headers or {})},
        request=_DUMMY_REQUEST,
    )


# ---------------------------------------------------------------------------
# _parse_jsonrpc_response
# ---------------------------------------------------------------------------

class TestParseJsonrpcResponse:
    def test_plain_json(self):
        resp = _json_response({"jsonrpc": "2.0", "result": {"ok": True}, "id": 1})
        assert _parse_jsonrpc_response(resp) == {"jsonrpc": "2.0", "result": {"ok": True}, "id": 1}

    def test_sse(self):
        payload = {"jsonrpc": "2.0", "result": {"tools": []}, "id": 2}
        resp = _sse_response(payload)
        assert _parse_jsonrpc_response(resp) == payload

    def test_sse_multiple_data_lines(self):
        lines = "data: {\"jsonrpc\":\"2.0\",\"id\":1}\n\ndata: {\"jsonrpc\":\"2.0\",\"result\":{\"ok\":true},\"id\":2}\n\n"
        resp = httpx.Response(200, content=lines.encode(), headers={"content-type": "text/event-stream"})
        parsed = _parse_jsonrpc_response(resp)
        assert parsed["result"] == {"ok": True}

    def test_sse_empty_raises(self):
        resp = httpx.Response(200, content=b"", headers={"content-type": "text/event-stream"})
        with pytest.raises(RuntimeError, match="No JSON-RPC payload"):
            _parse_jsonrpc_response(resp)


# ---------------------------------------------------------------------------
# MCPBridge.get_function_specs
# ---------------------------------------------------------------------------

class TestGetFunctionSpecs:
    def test_converts_to_function_type(self):
        bridge = _make_bridge()
        bridge._tools = {
            "get_state": {
                "name": "get_state",
                "description": "Get state",
                "inputSchema": {"type": "object", "properties": {"entity_id": {"type": "string"}}},
            }
        }
        specs = bridge.get_function_specs()
        assert len(specs) == 1
        assert specs[0]["type"] == "function"
        assert specs[0]["name"] == "ha_get_state"
        assert specs[0]["description"] == "Get state"
        assert specs[0]["parameters"]["properties"]["entity_id"]["type"] == "string"

    def test_prefix_from_label(self):
        bridge = _make_bridge(server_label="my_home")
        bridge._tools = {"ping": {"name": "ping", "description": "", "inputSchema": {}}}
        assert bridge.get_function_specs()[0]["name"] == "my_home_ping"


# ---------------------------------------------------------------------------
# MCPBridge.has_tool
# ---------------------------------------------------------------------------

class TestHasTool:
    def test_matches_prefixed(self):
        bridge = _make_bridge()
        bridge._tools = {"get_state": {}}
        assert bridge.has_tool("ha_get_state") is True

    def test_rejects_wrong_prefix(self):
        bridge = _make_bridge()
        bridge._tools = {"get_state": {}}
        assert bridge.has_tool("other_get_state") is False

    def test_rejects_unknown_tool(self):
        bridge = _make_bridge()
        bridge._tools = {"get_state": {}}
        assert bridge.has_tool("ha_unknown") is False


# ---------------------------------------------------------------------------
# MCPBridge.call_tool
# ---------------------------------------------------------------------------

class TestCallTool:
    @pytest.mark.asyncio
    async def test_proxies_call(self, monkeypatch):
        bridge = _make_bridge()
        bridge._tools = {"turn_on": {"name": "turn_on"}}
        bridge._client = httpx.AsyncClient()
        bridge._session_id = "sess-1"

        captured_request = {}

        async def mock_post(url, *, json, headers, **kw):
            captured_request["url"] = url
            captured_request["json"] = json
            captured_request["headers"] = headers
            return _json_response({
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": "Light turned on"}],
                },
                "id": json["id"],
            })

        monkeypatch.setattr(bridge._client, "post", mock_post)

        result = await bridge.call_tool("ha_turn_on", {"entity_id": "light.kitchen"})

        assert result == {"result": "Light turned on"}
        assert captured_request["json"]["method"] == "tools/call"
        assert captured_request["json"]["params"]["name"] == "turn_on"
        assert captured_request["json"]["params"]["arguments"] == {"entity_id": "light.kitchen"}
        assert captured_request["headers"]["Mcp-Session-Id"] == "sess-1"

        await bridge._client.aclose()

    @pytest.mark.asyncio
    async def test_returns_error_on_is_error(self, monkeypatch):
        bridge = _make_bridge()
        bridge._tools = {"fail": {"name": "fail"}}
        bridge._client = httpx.AsyncClient()

        async def mock_post(url, *, json, headers, **kw):
            return _json_response({
                "jsonrpc": "2.0",
                "result": {
                    "content": [{"type": "text", "text": "Something went wrong"}],
                    "isError": True,
                },
                "id": json["id"],
            })

        monkeypatch.setattr(bridge._client, "post", mock_post)

        result = await bridge.call_tool("ha_fail", {})
        assert result == {"error": "Something went wrong"}

        await bridge._client.aclose()


# ---------------------------------------------------------------------------
# MCPBridge.connect
# ---------------------------------------------------------------------------

class TestConnect:
    @pytest.mark.asyncio
    async def test_discovers_tools(self, monkeypatch):
        bridge = _make_bridge(allowed_tools=["get_state"])

        async def mock_post(_self, url, *, json, headers, **kw):
            method = json.get("method")
            if method == "initialize":
                return _json_response(
                    {"jsonrpc": "2.0", "result": {"serverInfo": {"name": "HA"}}, "id": json["id"]},
                    headers={"mcp-session-id": "s1"},
                )
            if method == "notifications/initialized":
                return httpx.Response(202, content=b"", headers={"content-type": "application/json"}, request=_DUMMY_REQUEST)
            if method == "tools/list":
                return _json_response({
                    "jsonrpc": "2.0",
                    "result": {
                        "tools": [
                            {"name": "get_state", "description": "Get", "inputSchema": {}},
                            {"name": "call_service", "description": "Call", "inputSchema": {}},
                        ]
                    },
                    "id": json["id"],
                })
            return _json_response({"jsonrpc": "2.0", "result": {}, "id": json.get("id")})

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        await bridge.connect()

        assert "get_state" in bridge._tools
        assert "call_service" not in bridge._tools  # filtered out
        assert bridge._session_id == "s1"

        await bridge.close()

    @pytest.mark.asyncio
    async def test_no_filter_discovers_all(self, monkeypatch):
        bridge = _make_bridge(allowed_tools=None)

        async def mock_post(_self, url, *, json, headers, **kw):
            method = json.get("method")
            if method == "initialize":
                return _json_response({"jsonrpc": "2.0", "result": {"serverInfo": {}}, "id": json["id"]})
            if method == "notifications/initialized":
                return httpx.Response(202, content=b"", headers={"content-type": "application/json"}, request=_DUMMY_REQUEST)
            if method == "tools/list":
                return _json_response({
                    "jsonrpc": "2.0",
                    "result": {"tools": [
                        {"name": "a", "description": "", "inputSchema": {}},
                        {"name": "b", "description": "", "inputSchema": {}},
                    ]},
                    "id": json["id"],
                })
            return _json_response({"jsonrpc": "2.0", "result": {}, "id": json.get("id")})

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        await bridge.connect()
        assert len(bridge._tools) == 2

        await bridge.close()

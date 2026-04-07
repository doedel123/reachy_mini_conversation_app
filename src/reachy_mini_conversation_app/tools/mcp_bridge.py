"""Lightweight MCP client that bridges a remote MCP server to local function-tool dispatch.

Speaks JSON-RPC 2.0 over Streamable HTTP (the transport used by Home Assistant's MCP
integration).  At startup it discovers available tools via ``tools/list`` and exposes
them as ``"type": "function"`` specs that the OpenAI Realtime API can consume.  Tool
calls are proxied back to the MCP server via ``tools/call``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = "2025-03-26"
_CLIENT_INFO = {"name": "reachy-mcp-bridge", "version": "1.0.0"}


def _parse_jsonrpc_response(response: httpx.Response) -> dict[str, Any]:
    """Extract a JSON-RPC result from either a plain JSON or SSE response."""
    content_type = response.headers.get("content-type", "")

    if "text/event-stream" in content_type:
        # SSE: look for the last `data:` line that contains a JSON-RPC result.
        last_data: dict[str, Any] | None = None
        for line in response.text.splitlines():
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                if not payload:
                    continue
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        last_data = parsed
                except json.JSONDecodeError:
                    continue
        if last_data is not None:
            return last_data
        raise RuntimeError("No JSON-RPC payload found in SSE response")

    # Default: plain JSON.
    return response.json()  # type: ignore[no-any-return]


class MCPBridge:
    """Bridges a single remote MCP server to local function-tool dispatch."""

    def __init__(
        self,
        *,
        server_url: str,
        token: str,
        server_label: str = "home_assistant",
        allowed_tools: list[str] | None = None,
    ) -> None:
        self._server_url = server_url
        self._token = token
        self._prefix = server_label + "_"
        self._allowed_tools = set(allowed_tools) if allowed_tools else None
        self._session_id: str | None = None
        self._next_id = 1
        self._tools: dict[str, dict[str, Any]] = {}  # MCP name -> MCP tool schema
        self._client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Initialize the MCP session and discover available tools."""
        self._client = httpx.AsyncClient(timeout=30.0)

        # 1. initialize
        init_result = await self._request(
            "initialize",
            {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": _CLIENT_INFO,
            },
        )
        logger.info("MCP initialize response: %s", init_result.get("result", {}).get("serverInfo"))

        # 2. notifications/initialized (no id → notification)
        await self._notify("notifications/initialized", {})

        # 3. tools/list
        tools_result = await self._request("tools/list", {})
        raw_tools: list[dict[str, Any]] = tools_result.get("result", {}).get("tools", [])

        for tool in raw_tools:
            name = tool.get("name", "")
            if not name:
                continue
            if self._allowed_tools is not None and name not in self._allowed_tools:
                continue
            self._tools[name] = tool

        logger.info(
            "MCP bridge connected – discovered %d tool(s): %s",
            len(self._tools),
            ", ".join(sorted(self._tools)),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # Public API consumed by core_tools
    # ------------------------------------------------------------------

    def get_function_specs(self) -> list[dict[str, Any]]:
        """Return discovered MCP tools as ``type=function`` specs."""
        specs: list[dict[str, Any]] = []
        for mcp_name, mcp_tool in self._tools.items():
            specs.append(
                {
                    "type": "function",
                    "name": f"{self._prefix}{mcp_name}",
                    "description": mcp_tool.get("description", ""),
                    "parameters": mcp_tool.get("inputSchema", {"type": "object", "properties": {}}),
                }
            )
        return specs

    def has_tool(self, prefixed_name: str) -> bool:
        """Check whether *prefixed_name* belongs to this bridge."""
        if not prefixed_name.startswith(self._prefix):
            return False
        return prefixed_name[len(self._prefix):] in self._tools

    async def call_tool(self, prefixed_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Proxy a tool call to the remote MCP server and return the result."""
        mcp_name = prefixed_name[len(self._prefix):]
        result = await self._request(
            "tools/call",
            {"name": mcp_name, "arguments": arguments},
        )

        # Extract content from the MCP response.
        content_items = result.get("result", {}).get("content", [])
        texts = [item.get("text", "") for item in content_items if item.get("type") == "text"]
        is_error = result.get("result", {}).get("isError", False)

        combined = "\n".join(texts) if texts else json.dumps(result.get("result", {}))
        if is_error:
            return {"error": combined}
        return {"result": combined}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {self._token}",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and return the parsed response."""
        assert self._client is not None, "MCPBridge not connected"
        request_id = self._next_id
        self._next_id += 1

        body = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id,
        }

        resp = await self._client.post(self._server_url, json=body, headers=self._headers())
        resp.raise_for_status()

        # Capture session id if returned.
        session_id = resp.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

        return _parse_jsonrpc_response(resp)

    async def _notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC notification (no ``id``, no response expected)."""
        assert self._client is not None, "MCPBridge not connected"

        body = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        resp = await self._client.post(self._server_url, json=body, headers=self._headers())
        # Notifications may return 200 or 202; both are acceptable.
        if resp.status_code not in (200, 202, 204):
            logger.warning("MCP notification %s returned status %d", method, resp.status_code)

        session_id = resp.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id

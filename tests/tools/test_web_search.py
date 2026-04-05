"""Tests for the web_search tool."""

from typing import Any
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.tools.web_search as web_search_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.tools.web_search import WebSearch, _extract_sources


def test_extract_sources_deduplicates_urls() -> None:
    """Source extraction should return unique URLs with optional titles."""
    payload = {
        "output": [
            {
                "sources": [
                    {"title": "Weather", "url": "https://weather.example/koln"},
                    {"title": "Weather Duplicate", "url": "https://weather.example/koln"},
                    {"url": "https://restaurants.example/maastricht"},
                ]
            }
        ]
    }

    assert _extract_sources(payload) == [
        {"title": "Weather", "url": "https://weather.example/koln"},
        {"url": "https://restaurants.example/maastricht"},
    ]


@pytest.mark.asyncio
async def test_web_search_tool_returns_answer_and_sources(monkeypatch: Any) -> None:
    """The tool should return output text plus extracted sources."""

    class FakeResponse:
        output_text = "It is 12 C and cloudy in Koln."

        def model_dump(self) -> dict[str, Any]:
            return {
                "output": [
                    {
                        "action": {
                            "sources": [
                                {"title": "Weather.com", "url": "https://weather.com/koln"},
                            ]
                        }
                    }
                ]
            }

    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> FakeResponse:
            self.calls.append(kwargs)
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.responses = FakeResponses()

    fake_client = FakeClient(api_key="test-key")
    monkeypatch.setattr(web_search_mod, "AsyncOpenAI", lambda api_key: fake_client)
    monkeypatch.setattr(web_search_mod.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(web_search_mod.config, "WEB_SEARCH_MODEL", "gpt-5-mini")

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    result = await WebSearch()(
        deps,
        query="Wetter in Koln heute",
        allowed_domains=["weather.com", "  "],
        user_location={"city": "Cologne", "country": "DE"},
    )

    assert result == {
        "answer": "It is 12 C and cloudy in Koln.",
        "sources": [{"title": "Weather.com", "url": "https://weather.com/koln"}],
    }
    assert fake_client.responses.calls[0]["model"] == "gpt-5-mini"
    assert fake_client.responses.calls[0]["tools"][0] == {
        "type": "web_search",
        "search_context_size": "medium",
        "filters": {"allowed_domains": ["weather.com"]},
        "user_location": {"type": "approximate", "city": "Cologne", "country": "DE"},
    }


@pytest.mark.asyncio
async def test_web_search_tool_retries_with_preview_tool(monkeypatch: Any) -> None:
    """The tool should fall back to the preview variant if needed."""

    class FakeResponse:
        output_text = "A fallback result."

        def model_dump(self) -> dict[str, Any]:
            return {}

    class FakeResponses:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> FakeResponse:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RuntimeError("unknown tool")
            return FakeResponse()

    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.responses = FakeResponses()

    fake_client = FakeClient(api_key="test-key")
    monkeypatch.setattr(web_search_mod, "AsyncOpenAI", lambda api_key: fake_client)
    monkeypatch.setattr(web_search_mod.config, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(web_search_mod.config, "WEB_SEARCH_MODEL", "gpt-5-mini")

    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    result = await WebSearch()(deps, query="Good restaurant in Maastricht")

    assert result == {"answer": "A fallback result."}
    assert fake_client.responses.calls[0]["tools"][0]["type"] == "web_search"
    assert fake_client.responses.calls[1]["tools"][0]["type"] == "web_search_preview"

import os
import logging
from typing import Any, Dict

from openai import AsyncOpenAI

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT_SIZE = "medium"
_MAX_SOURCES = 5


def _clean_allowed_domains(raw_domains: Any) -> list[str]:
    """Normalize optional domain filters."""
    if not isinstance(raw_domains, list):
        return []

    domains: list[str] = []
    for item in raw_domains:
        if isinstance(item, str):
            value = item.strip()
            if value:
                domains.append(value)
    return domains


def _build_user_location(raw_location: Any) -> dict[str, str] | None:
    """Normalize optional approximate user location for the search tool."""
    if not isinstance(raw_location, dict):
        return None

    location: dict[str, str] = {"type": "approximate"}
    for key in ("city", "region", "country", "timezone"):
        value = raw_location.get(key)
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                location[key] = cleaned

    return location if len(location) > 1 else None


def _extract_sources(payload: Any) -> list[dict[str, str]]:
    """Collect source URLs and titles from a response payload."""
    found: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            url = node.get("url")
            if isinstance(url, str) and url not in seen_urls:
                source: dict[str, str] = {"url": url}
                title = node.get("title")
                if isinstance(title, str) and title.strip():
                    source["title"] = title.strip()
                found.append(source)
                seen_urls.add(url)

            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(payload)
    return found[:_MAX_SOURCES]


class WebSearch(Tool):
    """Search the live web for current and location-sensitive information."""

    name = "web_search"
    description = (
        "Search the live web for current or location-sensitive information such as weather, news, "
        "restaurants, opening hours, schedules, prices, events, and other up-to-date facts."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The exact question or search request to answer with up-to-date web results.",
            },
            "search_context_size": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "How much search context to use. Use high for harder research; medium is default.",
            },
            "allowed_domains": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional allowlist of domains to search, for example official weather or restaurant sites.",
            },
            "user_location": {
                "type": "object",
                "description": "Optional approximate user location to improve local results.",
                "properties": {
                    "city": {"type": "string"},
                    "region": {"type": "string"},
                    "country": {"type": "string"},
                    "timezone": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Run a web search through the OpenAI Responses API."""
        del deps

        query = (kwargs.get("query") or "").strip()
        if not query:
            return {"error": "query must be a non-empty string"}

        api_key = (os.getenv("OPENAI_API_KEY") or config.OPENAI_API_KEY or "").strip()
        if not api_key:
            return {"error": "OPENAI_API_KEY is required for web_search"}

        search_context_size = kwargs.get("search_context_size") or _DEFAULT_CONTEXT_SIZE
        if search_context_size not in {"low", "medium", "high"}:
            search_context_size = _DEFAULT_CONTEXT_SIZE

        tool: dict[str, Any] = {
            "type": "web_search",
            "search_context_size": search_context_size,
        }

        allowed_domains = _clean_allowed_domains(kwargs.get("allowed_domains"))
        if allowed_domains:
            tool["filters"] = {"allowed_domains": allowed_domains}

        user_location = _build_user_location(kwargs.get("user_location"))
        if user_location is not None:
            tool["user_location"] = user_location

        logger.info("Tool call: web_search query=%s", query[:160])
        client = AsyncOpenAI(api_key=api_key)

        try:
            response = await client.responses.create(
                model=config.WEB_SEARCH_MODEL,
                input=query,
                tools=[tool],
                include=["web_search_call.action.sources"],
                max_tool_calls=1,
            )
        except Exception as exc:
            if tool["type"] != "web_search":
                raise

            logger.warning("web_search failed, retrying with preview tool: %s", exc)
            preview_tool = {
                "type": "web_search_preview",
                "search_context_size": search_context_size,
            }
            if user_location is not None:
                preview_tool["user_location"] = user_location

            response = await client.responses.create(
                model=config.WEB_SEARCH_MODEL,
                input=query,
                tools=[preview_tool],
                include=["web_search_call.action.sources"],
                max_tool_calls=1,
            )

        answer = (getattr(response, "output_text", "") or "").strip()
        payload = response.model_dump() if hasattr(response, "model_dump") else {}
        result: Dict[str, Any] = {"answer": answer or "No answer returned from web search."}

        sources = _extract_sources(payload)
        if sources:
            result["sources"] = sources

        return result

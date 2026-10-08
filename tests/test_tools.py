"""Unit tests for the LangChain Anakin tools.

These test the tool <-> SDK wiring (field names, method calls, output
shape) with the Anakin client's methods mocked — not live HTTP. The SDK's
own request/response behavior is anakin-py's test responsibility.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from anakin import Anakin, WireAuthRequiredError
from anakin.models import (
    AgenticSearchData,
    AgenticSearchResult,
    CrawlPage,
    CrawlResult,
    Document,
    MapResult,
    SearchResult,
    SearchResultItem,
    WireActionMatch,
    WireResult,
)

import langchain_anakin.tools as tools_module
from langchain_anakin import (
    AnakinAgenticSearchTool,
    AnakinCrawlTool,
    AnakinMapTool,
    AnakinScrapeTool,
    AnakinSearchTool,
    AnakinWireDiscoverTool,
    AnakinWireRunTool,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANAKIN_API_KEY", "ask_test_dummy")


@pytest.fixture
def async_client(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the per-call AsyncAnakin with a mock async context manager."""
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(tools_module, "new_async_client", lambda *_: client)
    return client


def _doc(**fields: object) -> Document:
    return Document(id="job_1", url="https://example.com", status="completed", **fields)


# ─── client construction ──────────────────────────────────────────────────────


def test_tool_builds_a_keyed_client_from_the_env() -> None:
    tool = AnakinScrapeTool()

    assert isinstance(tool.client, Anakin)
    assert tool.client.api_key_configured


def test_tool_without_a_key_builds_a_keyless_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANAKIN_API_KEY")

    tool = AnakinScrapeTool()

    assert not tool.client.api_key_configured


def test_keyed_only_tool_without_a_key_reports_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANAKIN_API_KEY")

    result = AnakinSearchTool().invoke({"prompt": "anything"})

    assert result[0]["error"].startswith("ConfigurationError")


def test_explicit_client_is_kept() -> None:
    client = MagicMock(spec=Anakin)

    tool = AnakinScrapeTool(client=client)

    assert tool.client is client


# ─── scrape ───────────────────────────────────────────────────────────────────


def test_scrape_tool_returns_markdown() -> None:
    tool = AnakinScrapeTool()
    tool.client = MagicMock()
    tool.client.scrape.return_value = _doc(markdown="# Hello")

    result = tool.invoke({"url": "https://example.com"})

    assert result == "# Hello"
    tool.client.scrape.assert_called_once_with(
        "https://example.com",
        generate_json=False,
        output_schema=None,
        country="us",
        use_browser=False,
        force_fresh=False,
    )


def test_scrape_tool_falls_back_to_summary_when_no_markdown() -> None:
    tool = AnakinScrapeTool()
    tool.client = MagicMock()
    tool.client.scrape.return_value = _doc(summary="A concise summary.")

    result = tool.invoke({"url": "https://example.com"})

    assert result == "A concise summary."


def test_scrape_tool_returns_extracted_json_for_an_output_schema() -> None:
    schema = {"type": "object", "properties": {"price": {"type": "number"}}}
    tool = AnakinScrapeTool()
    tool.client = MagicMock()
    tool.client.scrape.return_value = _doc(markdown="# Shop", generatedJson={"price": 9.5})

    result = tool.invoke({"url": "https://example.com", "output_schema": schema})

    assert result == '{"price": 9.5}'
    assert tool.client.scrape.call_args.kwargs["output_schema"] == schema


def test_scrape_tool_surfaces_errors_as_text_not_an_exception() -> None:
    tool = AnakinScrapeTool()
    tool.client = MagicMock()
    tool.client.scrape.side_effect = RuntimeError("boom")

    result = tool.invoke({"url": "https://example.com"})

    assert result == "RuntimeError: boom"


async def test_scrape_tool_async(async_client: MagicMock) -> None:
    async_client.scrape = AsyncMock(return_value=_doc(markdown="# Async"))

    result = await AnakinScrapeTool().ainvoke({"url": "https://example.com"})

    assert result == "# Async"
    async_client.__aexit__.assert_awaited_once()


# ─── search ───────────────────────────────────────────────────────────────────


def test_search_tool_returns_result_dicts() -> None:
    tool = AnakinSearchTool()
    tool.client = MagicMock()
    tool.client.search.return_value = SearchResult(
        id="search_1",
        results=[
            SearchResultItem(url="https://example.com", title="Example", snippet="...")
        ],
    )

    result = tool.invoke({"prompt": "what is anakin"})

    assert result == [
        {
            "url": "https://example.com",
            "title": "Example",
            "snippet": "...",
            "date": None,
            "last_updated": None,
        }
    ]
    tool.client.search.assert_called_once_with("what is anakin", limit=5)


async def test_search_tool_async(async_client: MagicMock) -> None:
    async_client.search = AsyncMock(
        return_value=SearchResult(id="s", results=[SearchResultItem(url="https://a.com")])
    )

    result = await AnakinSearchTool().ainvoke({"prompt": "q", "limit": 3})

    assert result[0]["url"] == "https://a.com"
    async_client.search.assert_awaited_once_with("q", limit=3)


def test_agentic_search_tool_returns_structured_result() -> None:
    tool = AnakinAgenticSearchTool()
    tool.client = MagicMock()
    tool.client.agentic_search.return_value = AgenticSearchResult(
        id="job_2",
        status="completed",
        generatedJson=AgenticSearchData(summary="Findings.", structured_data={"a": 1}),
    )

    result = tool.invoke({"prompt": "research this"})

    assert result["id"] == "job_2"
    assert result["generated_json"]["summary"] == "Findings."
    tool.client.agentic_search.assert_called_once_with(
        "research this", use_browser=True, schema=None
    )


# ─── map / crawl ──────────────────────────────────────────────────────────────


def test_map_tool_returns_links() -> None:
    tool = AnakinMapTool()
    tool.client = MagicMock()
    tool.client.map.return_value = MapResult(
        id="map_1", links=["https://example.com/a", "https://example.com/b"]
    )

    result = tool.invoke({"url": "https://example.com", "search": "pricing"})

    assert result == ["https://example.com/a", "https://example.com/b"]
    tool.client.map.assert_called_once_with(
        "https://example.com", search="pricing", limit=100, include_subdomains=False
    )


def test_crawl_tool_returns_page_markdown() -> None:
    tool = AnakinCrawlTool()
    tool.client = MagicMock()
    tool.client.crawl.return_value = CrawlResult(
        id="crawl_1",
        results=[
            CrawlPage(url="https://example.com/a", status="completed", markdown="# A"),
            CrawlPage(url="https://example.com/b", status="failed", error="timeout"),
        ],
    )

    result = tool.invoke(
        {"url": "https://example.com", "max_pages": 2, "include_patterns": ["/a*"]}
    )

    assert result == [
        {"url": "https://example.com/a", "markdown": "# A"},
        {"url": "https://example.com/b", "markdown": None, "error": "timeout"},
    ]
    tool.client.crawl.assert_called_once_with(
        "https://example.com",
        max_pages=2,
        depth=1,
        include_patterns=["/a*"],
        exclude_patterns=(),
        use_browser=False,
    )


# ─── wire ─────────────────────────────────────────────────────────────────────


def test_wire_discover_tool_returns_compact_matches() -> None:
    tool = AnakinWireDiscoverTool()
    tool.client = MagicMock()
    tool.client.wire.discover.return_value = [
        WireActionMatch(
            action_id="walmart_search",
            catalog="walmart",
            name="Search",
            params={"required": ["query"]},
            credits=1,
        )
    ]

    result = tool.invoke({"query": "phones on walmart"})

    assert result == [
        {
            "action_id": "walmart_search",
            "site": "walmart",
            "name": "Search",
            "params": {"required": ["query"]},
            "auth_required": False,
            "credits": 1,
        }
    ]
    tool.client.wire.discover.assert_called_once_with(
        "phones on walmart", catalog=None, limit=5
    )


def test_wire_run_tool_uses_wire_run_with_a_key() -> None:
    tool = AnakinWireRunTool()
    tool.client = MagicMock()
    tool.client.api_key_configured = True
    tool.client.wire.run.return_value = WireResult(
        job_id="w1", status="completed", data=[{"name": "Phone"}], credits_used=1
    )

    result = tool.invoke({"action_id": "walmart_search", "params": {"query": "phones"}})

    assert result == {"status": "completed", "data": [{"name": "Phone"}], "credits_used": 1}
    tool.client.wire.run.assert_called_once_with(
        "walmart_search", {"query": "phones"}, credential_id=None
    )
    tool.client.wire.zero_touch.assert_not_called()


def test_wire_run_tool_falls_back_to_zero_touch_without_a_key() -> None:
    tool = AnakinWireRunTool()
    tool.client = MagicMock()
    tool.client.api_key_configured = False
    tool.client.wire.zero_touch.return_value = WireResult(
        status="completed", data={"ok": True}, trial={"remaining_credits": 4}
    )

    result = tool.invoke({"action_id": "walmart_search", "params": {"query": "x"}})

    assert result["data"] == {"ok": True}
    assert result["trial"]["remaining_credits"] == 4
    tool.client.wire.zero_touch.assert_called_once_with("walmart_search", {"query": "x"})
    tool.client.wire.run.assert_not_called()


def test_wire_run_tool_surfaces_connect_url() -> None:
    tool = AnakinWireRunTool()
    tool.client = MagicMock()
    tool.client.api_key_configured = True
    tool.client.wire.run.side_effect = WireAuthRequiredError(
        "Account not connected", connect_url="https://anakin.io/connect/linkedin"
    )

    result = tool.invoke({"action_id": "li_profile"})

    assert "https://anakin.io/connect/linkedin" in result["error"]


async def test_wire_run_tool_async(async_client: MagicMock) -> None:
    async_client.api_key_configured = True
    async_client.wire.run = AsyncMock(
        return_value=WireResult(job_id="w1", status="completed", data={"a": 1})
    )

    result = await AnakinWireRunTool().ainvoke({"action_id": "a", "params": {"q": 1}})

    assert result["data"] == {"a": 1}
    async_client.wire.run.assert_awaited_once_with("a", {"q": 1}, credential_id=None)


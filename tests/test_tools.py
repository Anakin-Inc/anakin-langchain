"""Unit tests for the LangChain Anakin tools.

These test the tool <-> SDK wiring (field names, method calls, output
shape) with the Anakin client's methods mocked — not live HTTP. The SDK's
own request/response behavior is anakin-py's test responsibility.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from anakin.models import (
    AgenticSearchData,
    AgenticSearchResult,
    Document,
    SearchResult,
    SearchResultItem,
)

from langchain_anakin import (
    AnakinAgenticSearchTool,
    AnakinScrapeTool,
    AnakinSearchTool,
)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANAKIN_API_KEY", "ask_test_dummy")


def test_scrape_tool_returns_markdown() -> None:
    tool = AnakinScrapeTool()
    tool.client = MagicMock()
    tool.client.scrape.return_value = Document(
        id="job_1",
        url="https://example.com",
        status="completed",
        cached=False,
        duration_ms=100,
        markdown="# Hello",
    )

    result = tool.invoke({"url": "https://example.com"})

    assert result == "# Hello"
    tool.client.scrape.assert_called_once_with(
        "https://example.com",
        generate_json=False,
        country="us",
        use_browser=False,
        force_fresh=False,
    )


def test_scrape_tool_falls_back_to_summary_when_no_markdown() -> None:
    tool = AnakinScrapeTool()
    tool.client = MagicMock()
    tool.client.scrape.return_value = Document(
        id="job_1",
        url="https://example.com",
        status="completed",
        cached=False,
        duration_ms=100,
        summary="A concise summary.",
    )

    result = tool.invoke({"url": "https://example.com"})

    assert result == "A concise summary."


def test_scrape_tool_surfaces_errors_as_text_not_an_exception() -> None:
    tool = AnakinScrapeTool()
    tool.client = MagicMock()
    tool.client.scrape.side_effect = RuntimeError("boom")

    result = tool.invoke({"url": "https://example.com"})

    assert "boom" in result


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

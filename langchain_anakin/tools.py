"""Tools for the Anakin web data platform."""

from __future__ import annotations

from typing import Any

from anakin import Anakin
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator

from langchain_anakin._utilities import initialize_client


class AnakinScrapeTool(BaseTool):  # type: ignore[override]
    """Scrape a URL to markdown, HTML, and optional AI-extracted JSON.

    Setup:
        ```bash
        pip install -U langchain-anakin
        export ANAKIN_API_KEY="your-api-key"
        ```

    Instantiation:
        ```python
        from langchain_anakin import AnakinScrapeTool

        tool = AnakinScrapeTool()
        ```

    Invocation:
        ```python
        tool.invoke({"url": "https://example.com"})
        ```
    """

    name: str = "anakin_scrape"
    description: str = (
        "Scrape a single URL and return its content as clean markdown, "
        "optionally with AI-extracted structured JSON. Input should be a "
        "fully-qualified URL."
    )
    client: Anakin = Field(default=None)  # type: ignore[assignment]
    anakin_api_key: SecretStr = Field(default=SecretStr(""))
    anakin_base_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment and construct the Anakin client."""
        return initialize_client(values)

    def _run(
        self,
        url: str,
        generate_json: bool = False,
        country: str = "us",
        use_browser: bool = False,
        force_fresh: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Scrape a URL.

        Args:
            url: The URL to scrape.
            generate_json: AI-extract structured JSON from the page content.
            country: Proxy country code, e.g. "us", "gb", "de".
            use_browser: Use a headless browser — best for JS-heavy sites.
            force_fresh: Bypass the cache and force a fresh extraction.
            run_manager: The run manager for callbacks.
        """
        try:
            doc = self.client.scrape(
                url,
                generate_json=generate_json,
                country=country,
                use_browser=use_browser,
                force_fresh=force_fresh,
            )
            return doc.markdown or doc.summary or ""
        except Exception as e:  # noqa: BLE001 — tool errors surface to the agent as text
            return repr(e)


class AnakinSearchTool(BaseTool):  # type: ignore[override]
    """AI-powered web search with citations — synchronous, no polling.

    Setup:
        ```bash
        pip install -U langchain-anakin
        export ANAKIN_API_KEY="your-api-key"
        ```

    Instantiation:
        ```python
        from langchain_anakin import AnakinSearchTool

        tool = AnakinSearchTool()
        ```

    Invocation:
        ```python
        tool.invoke({"prompt": "latest developments in AI agents"})
        ```
    """

    name: str = "anakin_search"
    description: str = (
        "AI-powered web search. Input should be a search query or "
        "question. Output is a JSON array of results with URL, title, "
        "snippet, and date."
    )
    client: Anakin = Field(default=None)  # type: ignore[assignment]
    anakin_api_key: SecretStr = Field(default=SecretStr(""))
    anakin_base_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment and construct the Anakin client."""
        return initialize_client(values)

    def _run(
        self,
        prompt: str,
        limit: int = 5,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict]:
        """Search the web.

        Args:
            prompt: The search query or question.
            limit: Maximum number of results to return (default 5, max 20).
            run_manager: The run manager for callbacks.
        """
        try:
            result = self.client.search(prompt, limit=limit)
            return [r.model_dump() for r in result.results]
        except Exception as e:  # noqa: BLE001
            return [{"error": repr(e)}]


class AnakinAgenticSearchTool(BaseTool):  # type: ignore[override]
    """Multi-stage AI research pipeline: search, scrape citations, extract
    structured data. Slower than `AnakinSearchTool` (can take minutes) but
    returns a synthesized summary plus structured data rather than raw
    results.

    Setup:
        ```bash
        pip install -U langchain-anakin
        export ANAKIN_API_KEY="your-api-key"
        ```

    Instantiation:
        ```python
        from langchain_anakin import AnakinAgenticSearchTool

        tool = AnakinAgenticSearchTool()
        ```

    Invocation:
        ```python
        tool.invoke({"prompt": "Compare pricing across the top 5 CRM platforms"})
        ```
    """

    name: str = "anakin_agentic_search"
    description: str = (
        "Run a multi-stage AI research pipeline: search the web, scrape "
        "citation sources, and extract structured data with a summary. "
        "Slower than a plain search (can take several minutes) — use for "
        "research questions that need synthesis across multiple sources, "
        "not quick lookups."
    )
    client: Anakin = Field(default=None)  # type: ignore[assignment]
    anakin_api_key: SecretStr = Field(default=SecretStr(""))
    anakin_base_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment and construct the Anakin client."""
        return initialize_client(values)

    def _run(
        self,
        prompt: str,
        use_browser: bool = True,
        schema: dict | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict:
        """Run an agentic research pipeline.

        Args:
            prompt: The research question or topic (max 8KB).
            use_browser: Use a headless browser for citation scraping.
            schema: Optional JSON Schema (max 50KB) for the structured
                output. Leave unset to let the pipeline choose its own.
            run_manager: The run manager for callbacks.
        """
        try:
            result = self.client.agentic_search(
                prompt, use_browser=use_browser, schema=schema
            )
            return result.model_dump()
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)}

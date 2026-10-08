"""Tools for the Anakin web data platform."""

from __future__ import annotations

import json
from typing import Any

from anakin import Anakin, AsyncAnakin
from anakin.models import (
    CrawlResult,
    Document,
    MapResult,
    WireActionMatch,
    WireResult,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import Field, SecretStr, model_validator

from langchain_anakin._utilities import initialize_client, new_async_client


def _error_text(e: Exception) -> str:
    """Render an SDK error for the agent, keeping any "fix it here" URL."""
    text = f"{type(e).__name__}: {e}"
    connect_url = getattr(e, "connect_url", None)  # WireAuthRequiredError
    if connect_url:
        text += f" Connect the account at {connect_url}"
    return text


class _AnakinBaseTool(BaseTool):  # type: ignore[override]
    """Shared client wiring. With no API key the SDK runs keyless."""

    client: Anakin = Field(default=None)  # type: ignore[assignment]
    anakin_api_key: SecretStr = Field(default=SecretStr(""))
    anakin_base_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: dict) -> Any:
        """Validate the environment and construct the Anakin client."""
        return initialize_client(values)

    def _async_client(self) -> AsyncAnakin:
        return new_async_client(self.anakin_api_key, self.anakin_base_url)


# ─── Scrape ───────────────────────────────────────────────────────────────────


def _format_document(doc: Document, structured: bool) -> str:
    if structured and doc.generated_json is not None:
        return json.dumps(doc.generated_json)
    return doc.markdown or doc.summary or ""


class AnakinScrapeTool(_AnakinBaseTool):  # type: ignore[override]
    """Scrape a URL to markdown, or AI-extract structured JSON from it.

    Works without an API key (free per-IP allowance); set one for higher
    limits.

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
        "Scrape a single URL and return its content as clean markdown. "
        "Set generate_json or pass an output_schema to get AI-extracted "
        "structured JSON instead. Input should be a fully-qualified URL."
    )

    def _run(
        self,
        url: str,
        generate_json: bool = False,
        output_schema: dict | None = None,
        country: str = "us",
        use_browser: bool = False,
        force_fresh: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        """Scrape a URL.

        Args:
            url: The URL to scrape.
            generate_json: AI-extract structured JSON from the page content.
            output_schema: JSON Schema of the fields to extract (implies
                generate_json).
            country: Proxy country code, e.g. "us", "gb", "de".
            use_browser: Use a headless browser — best for JS-heavy sites.
            force_fresh: Bypass the cache and force a fresh extraction.
            run_manager: The run manager for callbacks.
        """
        try:
            doc = self.client.scrape(
                url,
                generate_json=generate_json,
                output_schema=output_schema,
                country=country,
                use_browser=use_browser,
                force_fresh=force_fresh,
            )
            return _format_document(doc, generate_json or output_schema is not None)
        except Exception as e:  # noqa: BLE001 — tool errors surface to the agent as text
            return _error_text(e)

    async def _arun(
        self,
        url: str,
        generate_json: bool = False,
        output_schema: dict | None = None,
        country: str = "us",
        use_browser: bool = False,
        force_fresh: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        try:
            async with self._async_client() as client:
                doc = await client.scrape(
                    url,
                    generate_json=generate_json,
                    output_schema=output_schema,
                    country=country,
                    use_browser=use_browser,
                    force_fresh=force_fresh,
                )
            return _format_document(doc, generate_json or output_schema is not None)
        except Exception as e:  # noqa: BLE001
            return _error_text(e)


# ─── Search ───────────────────────────────────────────────────────────────────


class AnakinSearchTool(_AnakinBaseTool):  # type: ignore[override]
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
            return [{"error": _error_text(e)}]

    async def _arun(
        self,
        prompt: str,
        limit: int = 5,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[dict]:
        try:
            async with self._async_client() as client:
                result = await client.search(prompt, limit=limit)
            return [r.model_dump() for r in result.results]
        except Exception as e:  # noqa: BLE001
            return [{"error": _error_text(e)}]


class AnakinAgenticSearchTool(_AnakinBaseTool):  # type: ignore[override]
    """Multi-stage AI research pipeline: search, scrape citations, extract
    structured data. Slower than `AnakinSearchTool` (typically 1-5 minutes)
    but returns a synthesized summary plus structured data rather than raw
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
        "Slower than a plain search (typically 1-5 minutes) — use for "
        "research questions that need synthesis across multiple sources, "
        "not quick lookups."
    )

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
            return {"error": _error_text(e)}

    async def _arun(
        self,
        prompt: str,
        use_browser: bool = True,
        schema: dict | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict:
        try:
            async with self._async_client() as client:
                result = await client.agentic_search(
                    prompt, use_browser=use_browser, schema=schema
                )
            return result.model_dump()
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


# ─── Map / Crawl ──────────────────────────────────────────────────────────────


def _format_map(result: MapResult) -> list[str]:
    return result.links


def _format_crawl(result: CrawlResult) -> list[dict]:
    return [
        {"url": p.url, "markdown": p.markdown, "error": p.error} if p.error
        else {"url": p.url, "markdown": p.markdown}
        for p in result.pages
    ]


class AnakinMapTool(_AnakinBaseTool):  # type: ignore[override]
    """Discover the URLs on a website without scraping their content.

    Instantiation:
        ```python
        from langchain_anakin import AnakinMapTool

        tool = AnakinMapTool()
        ```

    Invocation:
        ```python
        tool.invoke({"url": "https://docs.anakin.io", "search": "pricing"})
        ```
    """

    name: str = "anakin_map"
    description: str = (
        "List the URLs on a website (a site map) without fetching page "
        "content. Optionally filter by a search term. Use it to find the "
        "right page before scraping, or to plan a crawl."
    )

    def _run(
        self,
        url: str,
        search: str | None = None,
        limit: int = 100,
        include_subdomains: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[str]:
        """Map a website.

        Args:
            url: The site's starting URL.
            search: Only return URLs relevant to this term.
            limit: Maximum number of URLs to return (max 5000).
            include_subdomains: Also follow links onto subdomains.
            run_manager: The run manager for callbacks.
        """
        try:
            result = self.client.map(
                url, search=search, limit=limit, include_subdomains=include_subdomains
            )
            return _format_map(result)
        except Exception as e:  # noqa: BLE001
            return [_error_text(e)]

    async def _arun(
        self,
        url: str,
        search: str | None = None,
        limit: int = 100,
        include_subdomains: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[str]:
        try:
            async with self._async_client() as client:
                result = await client.map(
                    url, search=search, limit=limit, include_subdomains=include_subdomains
                )
            return _format_map(result)
        except Exception as e:  # noqa: BLE001
            return [_error_text(e)]


class AnakinCrawlTool(_AnakinBaseTool):  # type: ignore[override]
    """Crawl a website and return each page as markdown.

    Instantiation:
        ```python
        from langchain_anakin import AnakinCrawlTool

        tool = AnakinCrawlTool()
        ```

    Invocation:
        ```python
        tool.invoke({"url": "https://docs.anakin.io", "max_pages": 5})
        ```
    """

    name: str = "anakin_crawl"
    description: str = (
        "Crawl a website starting from a URL and return the markdown of "
        "each page found. Slower and more expensive than a single scrape; "
        "keep max_pages small and use include_patterns to stay focused."
    )

    def _run(
        self,
        url: str,
        max_pages: int = 10,
        depth: int = 1,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        use_browser: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict]:
        """Crawl a website.

        Args:
            url: The starting URL.
            max_pages: Maximum number of pages to crawl.
            depth: How many links deep to follow from the starting URL.
            include_patterns: Only crawl URLs matching these glob patterns,
                e.g. ["/blog/*"].
            exclude_patterns: Skip URLs matching these glob patterns.
            use_browser: Use a headless browser — best for JS-heavy sites.
            run_manager: The run manager for callbacks.
        """
        try:
            result = self.client.crawl(
                url,
                max_pages=max_pages,
                depth=depth,
                include_patterns=include_patterns or (),
                exclude_patterns=exclude_patterns or (),
                use_browser=use_browser,
            )
            return _format_crawl(result)
        except Exception as e:  # noqa: BLE001
            return [{"error": _error_text(e)}]

    async def _arun(
        self,
        url: str,
        max_pages: int = 10,
        depth: int = 1,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        use_browser: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[dict]:
        try:
            async with self._async_client() as client:
                result = await client.crawl(
                    url,
                    max_pages=max_pages,
                    depth=depth,
                    include_patterns=include_patterns or (),
                    exclude_patterns=exclude_patterns or (),
                    use_browser=use_browser,
                )
            return _format_crawl(result)
        except Exception as e:  # noqa: BLE001
            return [{"error": _error_text(e)}]


# ─── Wire ─────────────────────────────────────────────────────────────────────


def _format_matches(matches: list[WireActionMatch]) -> list[dict]:
    # The live resolve endpoint omits name/description, so drop empty fields
    # rather than hand the agent a row of nulls.
    rows = [
        {
            "action_id": m.action_id,
            "site": m.catalog_slug,
            "name": m.name,
            "description": m.description,
            "params": m.params,
            "auth_required": m.auth_required,
            "credits": m.credits,
        }
        for m in matches
    ]
    return [{k: v for k, v in row.items() if v is not None} for row in rows]


def _format_wire_result(result: WireResult) -> dict:
    out: dict[str, Any] = {
        "status": result.status,
        "data": result.data,
        "credits_used": result.credits_used,
    }
    if result.files:
        out["files"] = [f.model_dump() for f in result.files]
    if result.trial is not None:
        out["trial"] = result.trial.model_dump()
    return out


class AnakinWireDiscoverTool(_AnakinBaseTool):  # type: ignore[override]
    """Find a pre-built Wire action for a task on a specific website.

    Wire has ready-made, structured actions for hundreds of sites (search a
    store, read a profile, list jobs, ...). Pair with `AnakinWireRunTool`.
    Works without an API key.

    Instantiation:
        ```python
        from langchain_anakin import AnakinWireDiscoverTool

        tool = AnakinWireDiscoverTool()
        ```

    Invocation:
        ```python
        tool.invoke({"query": "search products on walmart"})
        ```
    """

    name: str = "anakin_wire_discover"
    description: str = (
        "Find pre-built Wire actions that return structured data from a "
        "specific website (e.g. 'search products on walmart', 'get a "
        "LinkedIn profile'). Returns action IDs with their parameters; "
        "run one with anakin_wire_run. Prefer this over scraping when a "
        "matching action exists."
    )

    def _run(
        self,
        query: str,
        site: str | None = None,
        limit: int = 5,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict]:
        """Find Wire actions by intent.

        Args:
            query: What you want to do, in plain language.
            site: Restrict to one site's catalog slug, e.g. "walmart".
            limit: Maximum number of actions to return.
            run_manager: The run manager for callbacks.
        """
        try:
            matches = self.client.wire.discover(query, catalog=site, limit=limit)
            return _format_matches(matches)
        except Exception as e:  # noqa: BLE001
            return [{"error": _error_text(e)}]

    async def _arun(
        self,
        query: str,
        site: str | None = None,
        limit: int = 5,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[dict]:
        try:
            async with self._async_client() as client:
                matches = await client.wire.discover(query, catalog=site, limit=limit)
            return _format_matches(matches)
        except Exception as e:  # noqa: BLE001
            return [{"error": _error_text(e)}]


class AnakinWireRunTool(_AnakinBaseTool):  # type: ignore[override]
    """Run a Wire action and return its structured result.

    With an API key this runs any action (`client.wire.run`). Without one it
    falls back to keyless Zero Touch (`client.wire.zero_touch`), which covers
    read-only actions on a free per-IP allowance.

    Instantiation:
        ```python
        from langchain_anakin import AnakinWireRunTool

        tool = AnakinWireRunTool()
        ```

    Invocation:
        ```python
        tool.invoke({"action_id": "walmart_search", "params": {"query": "phones"}})
        ```
    """

    name: str = "anakin_wire_run"
    description: str = (
        "Run a Wire action by its action_id (from anakin_wire_discover) "
        "with its parameters, and return the structured data it produces."
    )

    def _run(
        self,
        action_id: str,
        params: dict | None = None,
        credential_id: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict:
        """Run a Wire action.

        Args:
            action_id: The action to run, from anakin_wire_discover.
            params: The action's parameters.
            credential_id: A saved site login, for actions that need one.
            run_manager: The run manager for callbacks.
        """
        try:
            if self.client.api_key_configured:
                result = self.client.wire.run(
                    action_id, params, credential_id=credential_id
                )
            else:
                result = self.client.wire.zero_touch(action_id, params)
            return _format_wire_result(result)
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        action_id: str,
        params: dict | None = None,
        credential_id: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict:
        try:
            async with self._async_client() as client:
                if client.api_key_configured:
                    result = await client.wire.run(
                        action_id, params, credential_id=credential_id
                    )
                else:
                    result = await client.wire.zero_touch(action_id, params)
            return _format_wire_result(result)
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

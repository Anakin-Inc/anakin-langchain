"""Tools for the Anakin web data platform."""

from __future__ import annotations

import asyncio
import json
import re
import time
import warnings
from typing import Any, Literal
from urllib.parse import quote

import httpx
from anakin import (
    Anakin,
    AnakinError,
    AnakinPermissionError,
    AsyncAnakin,
    AuthenticationError,
    ConfigurationError,
    ConflictError,
    InsufficientCreditsError,
    InvalidRequestError,
    NotFoundError,
    RateLimitError,
    ServerError,
    UnprocessableEntityError,
)
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
from langchain_core.tools.base import create_schema_from_function
from pydantic import BaseModel, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_anakin._utilities import initialize_client, new_async_client

DEFAULT_BASE_URL = "https://api.anakin.io/v1"


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

    @model_validator(mode="after")
    def _describe_arguments(self) -> Self:
        """Give the model each argument's description from the `_run` docstring.

        LangChain's default schema for a `BaseTool` subclass has argument names
        and types only, so the "Args:" text never reached the model.
        """
        if self.args_schema is None:
            with warnings.catch_warnings():
                # Agentic search's `schema` argument shadows pydantic's deprecated
                # BaseModel.schema(), which pydantic warns about. It works as a
                # field, and renaming it would break 0.3.0 callers.
                warnings.filterwarnings(
                    "ignore", message='Field name "schema"', category=UserWarning
                )
                self.args_schema = create_schema_from_function(
                    self.name, self._run, parse_docstring=True
                )
        return self

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
        session_id: str | None = None,
        session_name: str | None = None,
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
            session_id: A saved browser session (from anakin_session_list) for
                pages behind a login. Use with use_browser.
            session_name: A saved browser session's name, instead of session_id.
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
                session_id=session_id,
                session_name=session_name,
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
        session_id: str | None = None,
        session_name: str | None = None,
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
                    session_id=session_id,
                    session_name=session_name,
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
        depth: int = 2,
        limit_per_level: int = 100,
        include_external_links: bool = False,
        use_browser: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[str]:
        """Map a website.

        Args:
            url: The site's starting URL.
            search: Only return URLs relevant to this term.
            limit: Maximum number of URLs to return (max 5000).
            include_subdomains: Also follow links onto subdomains.
            depth: How many links deep to follow from the starting URL (max 5).
            limit_per_level: Maximum URLs collected per link depth.
            include_external_links: Also return links to other sites.
            use_browser: Use a headless browser — best for JS-heavy sites.
            run_manager: The run manager for callbacks.
        """
        try:
            result = self.client.map(
                url,
                search=search,
                limit=limit,
                include_subdomains=include_subdomains,
                depth=depth,
                limit_per_level=limit_per_level,
                include_external_links=include_external_links,
                use_browser=use_browser,
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
        depth: int = 2,
        limit_per_level: int = 100,
        include_external_links: bool = False,
        use_browser: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[str]:
        try:
            async with self._async_client() as client:
                result = await client.map(
                    url,
                    search=search,
                    limit=limit,
                    include_subdomains=include_subdomains,
                    depth=depth,
                    limit_per_level=limit_per_level,
                    include_external_links=include_external_links,
                    use_browser=use_browser,
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
        country: str = "us",
        session_id: str | None = None,
        session_name: str | None = None,
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
            country: Proxy country code, e.g. "us", "gb", "de".
            session_id: A saved browser session (from anakin_session_list) for
                sites behind a login.
            session_name: A saved browser session's name, instead of session_id.
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
                country=country,
                session_id=session_id,
                session_name=session_name,
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
        country: str = "us",
        session_id: str | None = None,
        session_name: str | None = None,
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
                    country=country,
                    session_id=session_id,
                    session_name=session_name,
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
    store, read a profile, list jobs, ...). Pair with
    `AnakinWireReadActionTool` and `AnakinWireWriteActionTool`. Works without
    an API key.

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
        "LinkedIn profile'). Returns action IDs with their parameters and "
        "type; run a read action with anakin_wire_read_action and a write "
        "action with anakin_wire_write_action. Prefer this over scraping when "
        "a matching action exists."
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

    Deprecated since 0.4.0: use `AnakinWireReadActionTool` for actions that
    read data and `AnakinWireWriteActionTool` for actions that change
    something, which refuses payments.

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

    def __init__(self, **kwargs: Any) -> None:
        warnings.warn(
            "AnakinWireRunTool is deprecated and will be removed in a future "
            "release. Use AnakinWireReadActionTool for actions that read data "
            "and AnakinWireWriteActionTool for actions that change something.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)
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


# ─── Shared helpers for the tools below ───────────────────────────────────────


def _jsonable(value: Any) -> Any:
    """SDK models (and containers of them) as plain JSON-ready data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


# Same rule as Anakin's hosted MCP server: refuse Wire writes, Wire builds and
# browser tasks that look like they complete a payment or move money. Adding to
# a cart or filling in a form is fine.
_FINANCIAL = re.compile(
    r"\b(payments?|pay\s?now|checkout|purchase|place\s?order|buy\s?now|wire\s?transfer"
    r"|remit(?:tance)?|payout|charge\s?card|transfer\s?funds)\b",
    re.IGNORECASE,
)


def _refusal(kind: str, *parts: Any) -> dict | None:
    """The agent-facing refusal for a payment-like request, or None."""
    text = " ".join(str(p) for p in parts if p is not None)
    # \b does not split snake_case or kebab-case ids such as "shop_checkout".
    if _FINANCIAL.search(re.sub(r"[_-]+", " ", text)):
        return {
            "error": (
                f"Refused: this {kind} looks like a payment or money transfer. "
                "Complete payments directly on the site, or create the tool "
                "with allow_financial_actions=True."
            )
        }
    return None


# ─── Wire: catalog, actions, identities, builds ───────────────────────────────


class AnakinWireCatalogTool(_AnakinBaseTool):  # type: ignore[override]
    """Browse the Wire catalog: every supported site, or one site's actions.

    Works without an API key.

    Instantiation:
        ```python
        from langchain_anakin import AnakinWireCatalogTool

        tool = AnakinWireCatalogTool()
        ```

    Invocation:
        ```python
        tool.invoke({"slug": "walmart"})
        ```
    """

    name: str = "anakin_wire_catalog"
    description: str = (
        "Browse the Wire catalog. With no slug, list every supported website "
        "and its action count. With a site's slug (e.g. 'walmart'), list that "
        "site's actions with their exact parameters, type (read or write), "
        "auth mode and credit cost, plus its login fields."
    )

    def _run(
        self,
        slug: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """Browse the Wire catalog.

        Args:
            slug: A site's catalog slug, e.g. "walmart". Omit to list all sites.
            run_manager: The run manager for callbacks.
        """
        try:
            if slug:
                return _jsonable(self.client.wire.catalog(slug))
            return _jsonable(self.client.wire.catalogs())
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        slug: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        try:
            async with self._async_client() as client:
                if slug:
                    return _jsonable(await client.wire.catalog(slug))
                return _jsonable(await client.wire.catalogs())
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


class AnakinWireReadActionTool(_AnakinBaseTool):  # type: ignore[override]
    """Run a Wire action that reads data and changes nothing on the site.

    With an API key this runs any read action (`client.wire.run`). Without one
    it uses keyless Zero Touch (`client.wire.zero_touch`) on a free per-IP
    allowance.

    Instantiation:
        ```python
        from langchain_anakin import AnakinWireReadActionTool

        tool = AnakinWireReadActionTool()
        ```

    Invocation:
        ```python
        tool.invoke({"action_id": "walmart_search", "params": {"query": "phones"}})
        ```
    """

    name: str = "anakin_wire_read_action"
    description: str = (
        "Run a Wire READ action: one that extracts data and changes nothing on "
        "the site (search listings, product details, profiles, dashboard "
        "data). Find the action_id with anakin_wire_discover or "
        "anakin_wire_catalog and check its type is 'read' first."
    )

    def _run(
        self,
        action_id: str,
        params: dict | None = None,
        credential_id: str | None = None,
        identity_id: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict:
        """Run a Wire read action.

        Args:
            action_id: The action to run, from anakin_wire_discover or
                anakin_wire_catalog.
            params: The action's parameters, matching its schema.
            credential_id: A saved site login (from anakin_wire_identities),
                for actions that need one.
            identity_id: An identity whose credential to use, instead of
                credential_id.
            run_manager: The run manager for callbacks.
        """
        try:
            if self.client.api_key_configured:
                result = self.client.wire.run(
                    action_id,
                    params or {},
                    credential_id=credential_id,
                    identity_id=identity_id,
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
        identity_id: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict:
        try:
            async with self._async_client() as client:
                if client.api_key_configured:
                    result = await client.wire.run(
                        action_id,
                        params or {},
                        credential_id=credential_id,
                        identity_id=identity_id,
                    )
                else:
                    result = await client.wire.zero_touch(action_id, params)
            return _format_wire_result(result)
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


class AnakinWireWriteActionTool(_AnakinBaseTool):  # type: ignore[override]
    """Run a Wire action that changes something on the site.

    Refuses anything that looks like a payment or money transfer unless
    created with `allow_financial_actions=True`. Needs an API key.

    Instantiation:
        ```python
        from langchain_anakin import AnakinWireWriteActionTool

        tool = AnakinWireWriteActionTool()
        ```

    Invocation:
        ```python
        tool.invoke({
            "action_id": "shop_add_to_cart",
            "params": {"sku": "123"},
            "credential_id": "cred_...",
        })
        ```
    """

    name: str = "anakin_wire_write_action"
    description: str = (
        "Run a Wire WRITE action: one that changes state on the site (submit a "
        "form, add to a cart, post content, update settings). Check its type "
        "is 'write' with anakin_wire_discover or anakin_wire_catalog, and "
        "confirm with the user first. Payments and money transfers are refused."
    )
    allow_financial_actions: bool = False

    def _run(
        self,
        action_id: str,
        params: dict | None = None,
        credential_id: str | None = None,
        identity_id: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict:
        """Run a Wire write action.

        Args:
            action_id: The action to run (type "write").
            params: The action's parameters, matching its schema.
            credential_id: The user's saved login for the site (from
                anakin_wire_identities or anakin_wire_login).
            identity_id: An identity whose credential to use, instead of
                credential_id.
            run_manager: The run manager for callbacks.
        """
        if not self.allow_financial_actions and (
            refused := _refusal("action", action_id, params)
        ):
            return refused
        try:
            result = self.client.wire.run(
                action_id,
                params or {},
                credential_id=credential_id,
                identity_id=identity_id,
            )
            return _format_wire_result(result)
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        action_id: str,
        params: dict | None = None,
        credential_id: str | None = None,
        identity_id: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict:
        if not self.allow_financial_actions and (
            refused := _refusal("action", action_id, params)
        ):
            return refused
        try:
            async with self._async_client() as client:
                result = await client.wire.run(
                    action_id,
                    params or {},
                    credential_id=credential_id,
                    identity_id=identity_id,
                )
            return _format_wire_result(result)
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


class AnakinWireIdentitiesTool(_AnakinBaseTool):  # type: ignore[override]
    """List the user's saved Wire identities and their credentials.

    Instantiation:
        ```python
        from langchain_anakin import AnakinWireIdentitiesTool

        tool = AnakinWireIdentitiesTool()
        ```

    Invocation:
        ```python
        tool.invoke({})
        ```
    """

    name: str = "anakin_wire_identities"
    description: str = (
        "List the user's saved Wire identities (site logins) and their "
        "credentials. A credential's id is the credential_id that Wire actions "
        "behind a login need; check its status is 'active'."
    )

    def _run(
        self,
        catalog_id: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """List saved Wire identities.

        Args:
            catalog_id: Only identities for this catalog.
            run_manager: The run manager for callbacks.
        """
        try:
            return _jsonable(self.client.wire.identities(catalog_id=catalog_id))
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        catalog_id: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        try:
            async with self._async_client() as client:
                return _jsonable(await client.wire.identities(catalog_id=catalog_id))
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


class AnakinWireLoginTool(_AnakinBaseTool):  # type: ignore[override]
    """Sign in to a Wire site for the user and get a credential ID.

    Instantiation:
        ```python
        from langchain_anakin import AnakinWireLoginTool

        tool = AnakinWireLoginTool()
        ```

    Invocation:
        ```python
        tool.invoke({
            "catalog_slug": "neb",
            "params": {"email": "...", "password": "..."},
        })
        ```
    """

    name: str = "anakin_wire_login"
    description: str = (
        "Sign in to a site for the user and get a credential_id usable right "
        "away with Wire actions. Only sign in to accounts the user owns, with "
        "details the user provided; never invent credentials. The password is "
        "not stored, only an encrypted session."
    )

    def _run(
        self,
        catalog_slug: str,
        params: dict | None = None,
        identity_name: str | None = None,
        source_id: str | None = None,
        source_ref: dict | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """Sign in to a Wire site.

        Args:
            catalog_slug: The site's catalog slug.
            params: Login fields from the catalog's login schema, e.g. email and
                password.
            identity_name: A name for the identity (required with source_id).
            source_id: A connected 1Password identity source, instead of params.
            source_ref: The 1Password item locator: vault_id, item_id, fields.
            run_manager: The run manager for callbacks.
        """
        try:
            return _jsonable(
                self.client.wire.login(
                    catalog_slug,
                    params=params,
                    identity_name=identity_name,
                    source_id=source_id,
                    source_ref=source_ref,
                )
            )
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        catalog_slug: str,
        params: dict | None = None,
        identity_name: str | None = None,
        source_id: str | None = None,
        source_ref: dict | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        try:
            async with self._async_client() as client:
                return _jsonable(
                    await client.wire.login(
                        catalog_slug,
                        params=params,
                        identity_name=identity_name,
                        source_id=source_id,
                        source_ref=source_ref,
                    )
                )
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


class AnakinWireBuildTool(_AnakinBaseTool):  # type: ignore[override]
    """Request new Wire actions for a site that isn't in the catalog yet.

    Charges credits (refunded if the build fails). Refuses builds that look
    like payments unless created with `allow_financial_actions=True`.

    Instantiation:
        ```python
        from langchain_anakin import AnakinWireBuildTool

        tool = AnakinWireBuildTool()
        ```

    Invocation:
        ```python
        tool.invoke({
            "website_url": "https://example.com",
            "goal": "Extract product name, price and stock from a product page",
        })
        ```
    """

    name: str = "anakin_wire_build"
    description: str = (
        "Request new Wire actions for a site that isn't in the catalog yet. "
        "Wire builds and tests them; track progress with "
        "anakin_wire_build_status. Charges credits (refunded if the build "
        "fails), and a login build costs much more, so confirm with the user "
        "first. Only use after anakin_wire_discover and anakin_wire_catalog "
        "find nothing."
    )
    allow_financial_actions: bool = False

    def _run(
        self,
        website_url: str,
        goal: str,
        actions: list[str] | None = None,
        catalog_id: str | None = None,
        country: str | None = None,
        credential: dict | None = None,
        visibility: Literal["private", "public"] | None = None,
        force: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """Request a Wire build.

        Args:
            website_url: The site to build actions for.
            goal: What the action should extract or do, as specifically as
                possible.
            actions: Separate capabilities to build as separate actions.
            catalog_id: Add to an existing catalog instead of creating one.
            country: Two-letter country for geo-gated sites.
            credential: A login for actions behind a sign-in, e.g. type "plain"
                with username and password. Used once, never stored.
            visibility: "private" (the default) or "public".
            force: Build even if similar actions already exist.
            run_manager: The run manager for callbacks.
        """
        if not self.allow_financial_actions and (
            refused := _refusal("build", goal, website_url, actions)
        ):
            return refused
        try:
            return _jsonable(
                self.client.wire.build(
                    website_url,
                    goal,
                    actions=actions,
                    catalog_id=catalog_id,
                    country=country,
                    credential=credential,
                    visibility=visibility,
                    force=force,
                )
            )
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        website_url: str,
        goal: str,
        actions: list[str] | None = None,
        catalog_id: str | None = None,
        country: str | None = None,
        credential: dict | None = None,
        visibility: Literal["private", "public"] | None = None,
        force: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        if not self.allow_financial_actions and (
            refused := _refusal("build", goal, website_url, actions)
        ):
            return refused
        try:
            async with self._async_client() as client:
                return _jsonable(
                    await client.wire.build(
                        website_url,
                        goal,
                        actions=actions,
                        catalog_id=catalog_id,
                        country=country,
                        credential=credential,
                        visibility=visibility,
                        force=force,
                    )
                )
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


class AnakinWireBuildStatusTool(_AnakinBaseTool):  # type: ignore[override]
    """Check on Wire builds: one build by ID, or the user's recent builds.

    Instantiation:
        ```python
        from langchain_anakin import AnakinWireBuildStatusTool

        tool = AnakinWireBuildStatusTool()
        ```

    Invocation:
        ```python
        tool.invoke({"build_id": "build_..."})
        ```
    """

    name: str = "anakin_wire_build_status"
    description: str = (
        "Check Wire builds. With build_id, return that build: its status (poll "
        "about every 30 seconds while 'pending' or 'processing'), the "
        "published actions, and a skipped list of anything it could not "
        "deliver. Without build_id, list recent builds."
    )

    def _run(
        self,
        build_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        page: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """Check Wire builds.

        Args:
            build_id: The id from anakin_wire_build's response.
            status: List mode only: filter by status.
            limit: List mode only: maximum builds per page.
            page: List mode only: 1-based page, to see builds past the first page.
            run_manager: The run manager for callbacks.
        """
        try:
            if build_id:
                return _jsonable(self.client.wire.get_build(build_id))
            return _jsonable(self.client.wire.builds(status=status, page=page, limit=limit))
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        build_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
        page: int | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        try:
            async with self._async_client() as client:
                if build_id:
                    return _jsonable(await client.wire.get_build(build_id))
                return _jsonable(
                    await client.wire.builds(status=status, page=page, limit=limit)
                )
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


# ─── Monitors ─────────────────────────────────────────────────────────────────


class AnakinMonitorCreateTool(_AnakinBaseTool):  # type: ignore[override]
    """Watch a page, a whole site or a Wire action on a schedule.

    Instantiation:
        ```python
        from langchain_anakin import AnakinMonitorCreateTool

        tool = AnakinMonitorCreateTool()
        ```

    Invocation:
        ```python
        tool.invoke({"url": "https://example.com/pricing", "interval_minutes": 1440})
        ```
    """

    name: str = "anakin_monitor_create"
    description: str = (
        "Watch a page, a whole site or a Wire action on a schedule and record "
        "changes, optionally alerting a webhook or email. Every check costs "
        "credits: full_page 2, specific_data 3, ai_mode +1. Tell the user the "
        "credits per day ((1440 / interval_minutes) x cost per check) before "
        "creating."
    )

    def _run(
        self,
        url: str,
        interval_minutes: int,
        scope: Literal["page", "site", "wire"] = "page",
        watch_mode: Literal["full_page", "specific_data"] = "full_page",
        watch_format: Literal["markdown", "html", "cleaned_html"] | None = None,
        output_schema: dict | None = None,
        ai_mode: bool = False,
        ai_goal: str | None = None,
        alert_webhook_url: str | None = None,
        alert_emails: list[str] | None = None,
        use_browser: bool = False,
        country: str | None = None,
        session_id: str | None = None,
        max_pages: int | None = None,
        max_depth: int | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        wire_action_id: str | None = None,
        wire_catalog_slug: str | None = None,
        wire_params: dict | None = None,
        wire_credential_id: str | None = None,
        wire_watch_paths: list[str] | None = None,
        is_active: bool | None = None,
        expires_at: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """Create a monitor.

        Args:
            url: The page or site to watch (for wire scope, the site's URL).
            interval_minutes: Minutes between checks (minimum 15).
            scope: "page", "site" (tracks pages added, removed and changed) or
                "wire".
            watch_mode: "full_page", or "specific_data" to track only
                output_schema fields.
            watch_format: full_page mode: compare "markdown" (the default),
                "html" or "cleaned_html".
            output_schema: JSON Schema of the fields to track (required for
                specific_data).
            ai_mode: Filter out trivial changes and summarize real ones.
            ai_goal: With ai_mode, which changes count, e.g. "only price drops".
            alert_webhook_url: Webhook to notify on change.
            alert_emails: Emails to notify on change.
            use_browser: Render with a headless browser.
            country: Two-letter country to check from.
            session_id: A saved browser session for pages behind a login.
            max_pages: Site scope: pages per check.
            max_depth: Site scope: link depth to crawl (1-5, default 2).
            include_patterns: Site scope: only these URLs.
            exclude_patterns: Site scope: skip these URLs.
            wire_action_id: Wire scope: the action to run each check.
            wire_catalog_slug: Wire scope: the action's catalog slug.
            wire_params: Wire scope: the action's parameters.
            wire_credential_id: Wire scope: credential for actions behind a login.
            wire_watch_paths: Wire scope: JSON paths to compare instead of the
                whole result.
            is_active: False creates the monitor paused.
            expires_at: Date the monitor pauses itself (ISO 8601 or YYYY-MM-DD).
            run_manager: The run manager for callbacks.
        """
        options = _monitor_options(locals())
        try:
            return _jsonable(self.client.monitors.create(url, interval_minutes, **options))
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        url: str,
        interval_minutes: int,
        scope: Literal["page", "site", "wire"] = "page",
        watch_mode: Literal["full_page", "specific_data"] = "full_page",
        watch_format: Literal["markdown", "html", "cleaned_html"] | None = None,
        output_schema: dict | None = None,
        ai_mode: bool = False,
        ai_goal: str | None = None,
        alert_webhook_url: str | None = None,
        alert_emails: list[str] | None = None,
        use_browser: bool = False,
        country: str | None = None,
        session_id: str | None = None,
        max_pages: int | None = None,
        max_depth: int | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        wire_action_id: str | None = None,
        wire_catalog_slug: str | None = None,
        wire_params: dict | None = None,
        wire_credential_id: str | None = None,
        wire_watch_paths: list[str] | None = None,
        is_active: bool | None = None,
        expires_at: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        options = _monitor_options(locals())
        try:
            async with self._async_client() as client:
                return _jsonable(
                    await client.monitors.create(url, interval_minutes, **options)
                )
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


def _monitor_options(arguments: dict[str, Any]) -> dict[str, Any]:
    """The monitor options a caller set, from a `_run`/`_arun` locals()."""
    skip = {"self", "url", "interval_minutes", "run_manager"}
    return {k: v for k, v in arguments.items() if k not in skip and v is not None}


class AnakinMonitorListTool(_AnakinBaseTool):  # type: ignore[override]
    """List the user's monitors, or fetch one monitor.

    Instantiation:
        ```python
        from langchain_anakin import AnakinMonitorListTool

        tool = AnakinMonitorListTool()
        ```

    Invocation:
        ```python
        tool.invoke({})
        ```
    """

    name: str = "anakin_monitor_list"
    description: str = (
        "List the user's website monitors, or with monitor_id, one monitor's "
        "settings and status (next and last check, active state, cost per "
        "check, alerts)."
    )

    def _run(
        self,
        monitor_id: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """List monitors.

        Args:
            monitor_id: Return just this monitor.
            run_manager: The run manager for callbacks.
        """
        try:
            if monitor_id:
                return _jsonable(self.client.monitors.get(monitor_id))
            return _jsonable(self.client.monitors.list())
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        monitor_id: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        try:
            async with self._async_client() as client:
                if monitor_id:
                    return _jsonable(await client.monitors.get(monitor_id))
                return _jsonable(await client.monitors.list())
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


class AnakinMonitorChangesTool(_AnakinBaseTool):  # type: ignore[override]
    """Get the changes a monitor detected.

    Instantiation:
        ```python
        from langchain_anakin import AnakinMonitorChangesTool

        tool = AnakinMonitorChangesTool()
        ```

    Invocation:
        ```python
        tool.invoke({"monitor_id": "mon_..."})
        ```
    """

    name: str = "anakin_monitor_changes"
    description: str = (
        "Get the changes a monitor detected, each with its diff or summary. "
        "Find the monitor_id with anakin_monitor_list."
    )

    def _run(
        self,
        monitor_id: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """Get a monitor's changes.

        Args:
            monitor_id: The monitor, from anakin_monitor_list or
                anakin_monitor_create.
            run_manager: The run manager for callbacks.
        """
        try:
            return _jsonable(self.client.monitors.changes(monitor_id))
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        monitor_id: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        try:
            async with self._async_client() as client:
                return _jsonable(await client.monitors.changes(monitor_id))
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


class AnakinMonitorControlTool(_AnakinBaseTool):  # type: ignore[override]
    """Pause, resume, run now or delete a monitor.

    Instantiation:
        ```python
        from langchain_anakin import AnakinMonitorControlTool

        tool = AnakinMonitorControlTool()
        ```

    Invocation:
        ```python
        tool.invoke({"monitor_id": "mon_...", "action": "pause"})
        ```
    """

    name: str = "anakin_monitor_control"
    description: str = (
        "Pause, resume, run now or delete a monitor. run_now is billed like a "
        "check. delete permanently removes the monitor and its history: "
        "confirm with the user first."
    )

    def _run(
        self,
        monitor_id: str,
        action: Literal["pause", "resume", "run_now", "delete"],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """Control a monitor.

        Args:
            monitor_id: The monitor.
            action: "pause", "resume", "run_now" or "delete".
            run_manager: The run manager for callbacks.
        """
        try:
            monitors = self.client.monitors
            if action == "delete":
                monitors.delete(monitor_id)
                return {"id": monitor_id, "deleted": True}
            if action in ("pause", "resume", "run_now"):
                return _jsonable(getattr(monitors, action)(monitor_id))
            return {"error": f'Unknown action "{action}": use pause, resume, run_now or delete.'}
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        monitor_id: str,
        action: Literal["pause", "resume", "run_now", "delete"],
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        try:
            async with self._async_client() as client:
                if action == "delete":
                    await client.monitors.delete(monitor_id)
                    return {"id": monitor_id, "deleted": True}
                if action in ("pause", "resume", "run_now"):
                    return _jsonable(await getattr(client.monitors, action)(monitor_id))
            return {"error": f'Unknown action "{action}": use pause, resume, run_now or delete.'}
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


# ─── AI visibility ────────────────────────────────────────────────────────────


def _visibility(result: Any, include_full_content: bool) -> Any:
    out = _jsonable(result)
    if not include_full_content and isinstance(out, dict):
        for engine in out.get("results", []):
            engine.pop("full_content", None)
    return out


class AnakinAIVisibilitySearchTool(_AnakinBaseTool):  # type: ignore[override]
    """Ask several AI answer engines the same question and compare.

    Instantiation:
        ```python
        from langchain_anakin import AnakinAIVisibilitySearchTool

        tool = AnakinAIVisibilitySearchTool()
        ```

    Invocation:
        ```python
        tool.invoke({"query": "best web scraping API"})
        ```
    """

    name: str = "anakin_ai_visibility_search"
    description: str = (
        "Ask several AI answer engines (such as ChatGPT, Gemini and Google AI "
        "Overview) the same question and compare their answers, with a "
        "synthesis of where they agree. For brand and AI-SEO visibility checks. "
        "Takes 1-2 minutes; billed per engine."
    )

    def _run(
        self,
        query: str,
        sources: list[str] | None = None,
        country: str = "us",
        include_full_content: bool = False,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """Compare AI engines' answers.

        Args:
            query: The question to ask every engine.
            sources: Engine slugs to ask (see anakin_ai_visibility_sources); all
                enabled engines when omitted.
            country: Two-letter country for geo-specific answers.
            include_full_content: Also return each engine's full answer, which
                is long. Summaries and the synthesis are always returned.
            run_manager: The run manager for callbacks.
        """
        try:
            result = self.client.ai_visibility.search(query, sources=sources, country=country)
            return _visibility(result, include_full_content)
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        query: str,
        sources: list[str] | None = None,
        country: str = "us",
        include_full_content: bool = False,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        try:
            async with self._async_client() as client:
                result = await client.ai_visibility.search(
                    query, sources=sources, country=country
                )
            return _visibility(result, include_full_content)
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


class AnakinAIVisibilitySourcesTool(_AnakinBaseTool):  # type: ignore[override]
    """List the AI answer engines that AI visibility search can ask.

    Instantiation:
        ```python
        from langchain_anakin import AnakinAIVisibilitySourcesTool

        tool = AnakinAIVisibilitySourcesTool()
        ```

    Invocation:
        ```python
        tool.invoke({})
        ```
    """

    name: str = "anakin_ai_visibility_sources"
    description: str = (
        "List the AI answer engines anakin_ai_visibility_search can ask, with "
        "their slugs."
    )

    def _run(self, run_manager: CallbackManagerForToolRun | None = None) -> Any:
        """List AI answer engines.

        Args:
            run_manager: The run manager for callbacks.
        """
        try:
            return _jsonable(self.client.ai_visibility.sources())
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(self, run_manager: AsyncCallbackManagerForToolRun | None = None) -> Any:
        try:
            async with self._async_client() as client:
                return _jsonable(await client.ai_visibility.sources())
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


# ─── Saved browser sessions and browser tasks ─────────────────────────────────


class AnakinSessionListTool(_AnakinBaseTool):  # type: ignore[override]
    """List the user's saved browser sessions (logins captured in the dashboard).

    Instantiation:
        ```python
        from langchain_anakin import AnakinSessionListTool

        tool = AnakinSessionListTool()
        ```

    Invocation:
        ```python
        tool.invoke({"domain": "amazon.com"})
        ```
    """

    name: str = "anakin_session_list"
    description: str = (
        "List the user's saved browser sessions (encrypted login states created "
        "in the Anakin dashboard). Pass a session's id as session_id to scrape, "
        "crawl, monitor or browser-task tools to reach pages behind that login."
    )

    def _run(
        self,
        domain: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """List saved browser sessions.

        Args:
            domain: Only sessions for this site, e.g. "amazon.com".
            run_manager: The run manager for callbacks.
        """
        try:
            return _jsonable(self.client.sessions.list(domain=domain))
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        domain: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        try:
            async with self._async_client() as client:
                return _jsonable(await client.sessions.list(domain=domain))
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


class AnakinSessionDeleteTool(_AnakinBaseTool):  # type: ignore[override]
    """Permanently delete a saved browser session.

    Instantiation:
        ```python
        from langchain_anakin import AnakinSessionDeleteTool

        tool = AnakinSessionDeleteTool()
        ```

    Invocation:
        ```python
        tool.invoke({"session_id": "sess_..."})
        ```
    """

    name: str = "anakin_session_delete"
    description: str = (
        "Permanently delete a saved browser session. The user must log in again "
        "in the dashboard to recreate it, and monitors using it lose access. "
        "Confirm with the user first."
    )

    def _run(
        self,
        session_id: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict:
        """Delete a saved browser session.

        Args:
            session_id: The session to delete, from anakin_session_list.
            run_manager: The run manager for callbacks.
        """
        try:
            self.client.sessions.delete(session_id)
            return {"id": session_id, "deleted": True}
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        session_id: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict:
        try:
            async with self._async_client() as client:
                await client.sessions.delete(session_id)
            return {"id": session_id, "deleted": True}
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}


BROWSER_TASK_POLL_SECONDS = 3.0
BROWSER_TASK_MAX_SECONDS = 360.0


def _browser_task_body(
    prompt: str,
    url: str | None,
    session_id: str | None,
    output_schema: dict | None,
    max_steps: int | None,
    timeout_ms: int | None,
) -> dict[str, Any]:
    optional = {
        "url": url,
        "session_id": session_id,
        "output_schema": output_schema,
        "max_steps": max_steps,
        "timeout_ms": timeout_ms,
    }
    return {"prompt": prompt, "async": True, **{k: v for k, v in optional.items() if v is not None}}


# The SDK's error class for each HTTP status, so a browser task fails with the
# same error names as every other tool.
_STATUS_ERRORS: dict[int, type[AnakinError]] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    402: InsufficientCreditsError,
    403: AnakinPermissionError,
    404: NotFoundError,
    409: ConflictError,
    422: UnprocessableEntityError,
    429: RateLimitError,
}


def _checked(res: httpx.Response) -> dict[str, Any]:
    """The JSON body of an Anakin API response, or the SDK's error for it."""
    try:
        data = res.json() if res.content else {}
    except ValueError:
        data = {}
    if res.status_code >= 400:
        detail = data.get("message") or data.get("error") if isinstance(data, dict) else None
        if isinstance(detail, dict):
            detail = detail.get("message")
        default = ServerError if res.status_code >= 500 else AnakinError
        raise _STATUS_ERRORS.get(res.status_code, default)(
            detail or f"Anakin API returned {res.status_code}",
            status_code=res.status_code,
            body=data,
        )
    return data if isinstance(data, dict) else {"result": data}


def _task_outcome(job: dict[str, Any]) -> tuple[bool, Any]:
    """(finished, result) for a browser-task job; raises if it failed."""
    status = job.get("status")
    if status == "completed":
        return True, job.get("result", job)
    if status in ("failed", "timed_out"):
        raise AnakinError(
            f"Browser task {status}: {job.get('error') or 'unknown error'}",
            code=str(status).upper(),
        )
    return False, None


class AnakinBrowserTaskTool(_AnakinBaseTool):  # type: ignore[override]
    """Have an AI agent drive a real cloud browser through a multi-step task.

    Calls the REST API directly (`POST /v1/ai/evaluate`, then polls the job),
    since `anakin-sdk` has no method for it. Needs an API key. Refuses tasks
    that look like payments unless created with `allow_financial_actions=True`.

    Instantiation:
        ```python
        from langchain_anakin import AnakinBrowserTaskTool

        tool = AnakinBrowserTaskTool()
        ```

    Invocation:
        ```python
        tool.invoke({
            "prompt": "Find the cheapest 65-inch TV and list its specs",
            "url": "https://example-store.com",
        })
        ```
    """

    name: str = "anakin_browser_task"
    description: str = (
        "Have an AI agent drive a real cloud browser for a multi-step task "
        "(navigate, click, type, extract). Runs up to about 5 minutes. Use only "
        "when scraping can't do it and no Wire action exists. Never put "
        "passwords in the prompt: use session_id. Payments are refused."
    )
    allow_financial_actions: bool = False

    def _api_base(self) -> str:
        return (self.anakin_base_url or DEFAULT_BASE_URL).rstrip("/")

    def _api_key(self) -> str:
        key = self.anakin_api_key.get_secret_value()
        if not key:
            raise ConfigurationError(
                "POST /ai/evaluate needs an Anakin API key. Pass anakin_api_key=... "
                "or set ANAKIN_API_KEY. Get a free key (300 credits) at "
                "https://anakin.io/signup."
            )
        return key

    def _run(
        self,
        prompt: str,
        url: str | None = None,
        session_id: str | None = None,
        output_schema: dict | None = None,
        max_steps: int | None = None,
        timeout_ms: int | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> Any:
        """Run a browser task.

        Args:
            prompt: The task, specific about the goal and what to return.
            url: Where to start.
            session_id: A saved browser session (from anakin_session_list) for
                logged-in tasks.
            output_schema: JSON Schema for a structured result.
            max_steps: Cap on browser steps.
            timeout_ms: Task timeout in milliseconds.
            run_manager: The run manager for callbacks.
        """
        if not self.allow_financial_actions and (refused := _refusal("task", prompt, url)):
            return refused
        body = _browser_task_body(prompt, url, session_id, output_schema, max_steps, timeout_ms)
        try:
            headers = {"X-API-Key": self._api_key()}
            with httpx.Client(base_url=self._api_base(), headers=headers, timeout=30.0) as http:
                job = _checked(http.post("/ai/evaluate", json=body))
                workflow_id = job.get("workflow_id")
                if not workflow_id:
                    return job
                deadline = time.monotonic() + BROWSER_TASK_MAX_SECONDS
                while time.monotonic() < deadline:
                    job = _checked(http.get(f"/ai/jobs/{quote(str(workflow_id), safe='')}"))
                    finished, result = _task_outcome(job)
                    if finished:
                        return result
                    time.sleep(BROWSER_TASK_POLL_SECONDS)
            raise AnakinError("Browser task still running after 6 minutes.", code="TIMEOUT")
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

    async def _arun(
        self,
        prompt: str,
        url: str | None = None,
        session_id: str | None = None,
        output_schema: dict | None = None,
        max_steps: int | None = None,
        timeout_ms: int | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> Any:
        if not self.allow_financial_actions and (refused := _refusal("task", prompt, url)):
            return refused
        body = _browser_task_body(prompt, url, session_id, output_schema, max_steps, timeout_ms)
        try:
            headers = {"X-API-Key": self._api_key()}
            async with httpx.AsyncClient(
                base_url=self._api_base(), headers=headers, timeout=30.0
            ) as http:
                job = _checked(await http.post("/ai/evaluate", json=body))
                workflow_id = job.get("workflow_id")
                if not workflow_id:
                    return job
                deadline = time.monotonic() + BROWSER_TASK_MAX_SECONDS
                while time.monotonic() < deadline:
                    job = _checked(
                        await http.get(f"/ai/jobs/{quote(str(workflow_id), safe='')}")
                    )
                    finished, result = _task_outcome(job)
                    if finished:
                        return result
                    await asyncio.sleep(BROWSER_TASK_POLL_SECONDS)
            raise AnakinError("Browser task still running after 6 minutes.", code="TIMEOUT")
        except Exception as e:  # noqa: BLE001
            return {"error": _error_text(e)}

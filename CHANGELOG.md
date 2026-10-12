# Changelog

## [0.4.0] - unreleased

One tool for each of the 22 tools on Anakin's hosted MCP server
(`mcp.anakin.io`). 0.3.0 covered 8.

### Added
- Wire: `AnakinWireCatalogTool`, `AnakinWireReadActionTool`,
  `AnakinWireWriteActionTool`, `AnakinWireIdentitiesTool`,
  `AnakinWireLoginTool`, `AnakinWireBuildTool`, `AnakinWireBuildStatusTool`.
- Monitors: `AnakinMonitorCreateTool`, `AnakinMonitorListTool`,
  `AnakinMonitorChangesTool`, `AnakinMonitorControlTool`.
- AI visibility: `AnakinAIVisibilitySearchTool`, `AnakinAIVisibilitySourcesTool`.
- Browser: `AnakinSessionListTool`, `AnakinSessionDeleteTool`,
  `AnakinBrowserTaskTool`.
- `AnakinToolkit`: every tool, sharing one client. `read_only=True` leaves out
  the 7 tools that change something or start a long paid job.
- Like the hosted server, the Wire write, Wire build and browser task tools
  refuse anything that looks like a payment or money transfer unless created
  with `allow_financial_actions=True`.
- The hosted server's remaining parameters on existing tools: `session_id` and
  `session_name` on scrape and crawl, `country` on crawl, and `depth`,
  `limit_per_level`, `include_external_links` and `use_browser` on map.
- `httpx` is now a direct dependency (`AnakinBrowserTaskTool` calls the REST
  API, since `anakin-sdk` has no browser-task method).

### Changed
- Every argument's description from the docstrings now reaches the model in
  the tool's schema. LangChain's default schema for a `BaseTool` subclass had
  names and types only.

### Deprecated
- `AnakinWireRunTool`: use `AnakinWireReadActionTool` for actions that read
  data and `AnakinWireWriteActionTool` for actions that change something. It
  still works in 0.4.0 and warns when created.

## [0.3.0] - 2026-10-09

Tracks `anakin-sdk` 0.2.0.

### Added
- `AnakinMapTool`, `AnakinCrawlTool`, `AnakinWireDiscoverTool`, `AnakinWireRunTool`.
- Async support on every tool (`ainvoke`), backed by `anakin.AsyncAnakin`.
- `AnakinScrapeTool(output_schema=...)` for schema-guided JSON extraction.
- Keyless mode: with no API key, scrape, Wire discovery and read-only Wire
  runs (via Zero Touch) work; other tools return a `ConfigurationError`
  message with a signup link.
- A pre-built `client=` passed to a tool is kept instead of replaced.

### Changed
- Requires `anakin-sdk>=0.2.0,<0.3.0` (was `<0.2.0`, which blocked the update).
- Allows `langchain-core` 1.x (was `<1.0.0`, which conflicted with
  `langchain.agents.create_agent` used in the README).
- `AnakinScrapeTool` returns the extracted JSON (as a JSON string) when
  `generate_json` or `output_schema` is set. It used to return markdown and
  drop the extraction.
- Error text is now `"<ErrorClass>: <message>"` instead of `repr(e)`, and a
  Wire `connect_url` is appended when a site account needs connecting.

## [0.2.0]

Initial release on PyPI: `AnakinScrapeTool`, `AnakinSearchTool`,
`AnakinAgenticSearchTool`.

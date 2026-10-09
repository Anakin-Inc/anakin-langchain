# Changelog

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

# langchain-anakin

LangChain tools for [Anakin](https://anakin.io): web scraping, site mapping
and crawling, AI search, multi-stage agentic research, and Wire (pre-built
structured actions for hundreds of sites). Built on the official
[`anakin-sdk`](https://github.com/Anakin-Inc/anakin-py) Python client (not a
separate HTTP implementation).

## Install

```bash
pip install -U langchain-anakin
export ANAKIN_API_KEY="ask-..."   # free at https://anakin.io/signup
```

No key yet? `AnakinScrapeTool`, `AnakinWireDiscoverTool` and
`AnakinWireRunTool` (read-only actions) work without one on a free per-IP
allowance. The other tools return a `ConfigurationError` message with a
signup link.

## Tools

| Tool | What it does | Needs key |
|---|---|---|
| `AnakinScrapeTool` | Scrape a URL to markdown, or AI-extract JSON (`generate_json` / `output_schema`) | No |
| `AnakinSearchTool` | Synchronous AI-powered web search with citations | Yes |
| `AnakinAgenticSearchTool` | Multi-stage research: search, scrape citations, extract structured data with a summary (1-5 min) | Yes |
| `AnakinMapTool` | List the URLs on a site, optionally filtered by a search term | Yes |
| `AnakinCrawlTool` | Crawl a site and return each page's markdown | Yes |
| `AnakinWireDiscoverTool` | Find a pre-built Wire action for a task on a site | No |
| `AnakinWireRunTool` | Run a Wire action and return its structured data | No (read-only actions) |

```python
from langchain_anakin import (
    AnakinScrapeTool,
    AnakinSearchTool,
    AnakinWireDiscoverTool,
    AnakinWireRunTool,
)

AnakinScrapeTool().invoke({"url": "https://example.com"})

AnakinScrapeTool().invoke({
    "url": "https://example.com/product",
    "output_schema": {"type": "object", "properties": {"price": {"type": "number"}}},
})

AnakinSearchTool().invoke({"prompt": "latest developments in AI agents"})

actions = AnakinWireDiscoverTool().invoke({"query": "search products on walmart"})
AnakinWireRunTool().invoke({"action_id": "walmart_search", "params": {"query": "phones"}})
```

Every tool also supports async (`await tool.ainvoke(...)`), backed by
`anakin.AsyncAnakin`.

Use with an agent:

```python
from langchain.agents import create_agent
from langchain_anakin import (
    AnakinScrapeTool,
    AnakinSearchTool,
    AnakinWireDiscoverTool,
    AnakinWireRunTool,
)

agent = create_agent(
    model,
    tools=[
        AnakinSearchTool(),
        AnakinScrapeTool(),
        AnakinWireDiscoverTool(),
        AnakinWireRunTool(),
    ],
)
```

Errors (bad key, no credits, a Wire site that needs a connected account)
come back to the agent as text instead of raising, including any URL it
needs to fix the problem.

## Development

Requires `anakin-sdk` 0.2.x. Until 0.2.0 is on PyPI, `[tool.uv.sources]` in
`pyproject.toml` points at the sibling checkout `../anakin-py`; drop that
override once it is published.

```bash
uv sync --group test
uv run pytest
```

## License

Apache-2.0.

# langchain-anakin

LangChain tools for [Anakin](https://anakin.io) — web scraping, AI search,
and multi-stage agentic research, built on the official
[`anakin-sdk`](https://github.com/Anakin-Inc/anakin-py) Python client (not a
separate HTTP implementation).

## Install

```bash
pip install -U langchain-anakin
export ANAKIN_API_KEY="ask-..."   # free at https://anakin.io/dashboard
```

## Tools

| Tool | What it does |
|---|---|
| `AnakinScrapeTool` | Scrape a URL to markdown, optionally AI-extract structured JSON |
| `AnakinSearchTool` | Synchronous AI-powered web search with citations |
| `AnakinAgenticSearchTool` | Multi-stage research: search, scrape citations, extract structured data with a summary |

```python
from langchain_anakin import AnakinScrapeTool, AnakinSearchTool, AnakinAgenticSearchTool

scrape = AnakinScrapeTool()
scrape.invoke({"url": "https://example.com"})

search = AnakinSearchTool()
search.invoke({"prompt": "latest developments in AI agents"})

research = AnakinAgenticSearchTool()
research.invoke({"prompt": "Compare pricing across the top 5 CRM platforms"})
```

Use with an agent:

```python
from langchain.agents import create_agent  # or your preferred agent constructor
from langchain_anakin import AnakinScrapeTool, AnakinSearchTool

agent = create_agent(model, tools=[AnakinScrapeTool(), AnakinSearchTool()])
```

## Development

This package depends on `anakin-sdk`, which isn't published to PyPI yet —
`pyproject.toml`'s `[tool.uv.sources]` points at a local path
(`../anakin-py`) for development. Once `anakin-sdk` is live on PyPI, drop
that override.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../anakin-py -e ".[test]"
pytest
```

## License

Apache-2.0.

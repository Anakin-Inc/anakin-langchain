# langchain-anakin

LangChain tools for [Anakin](https://anakin.io): web scraping, site mapping
and crawling, AI search, multi-stage agentic research, Wire (pre-built
structured actions for hundreds of sites), website monitors, AI-engine answer
comparison, saved browser sessions and AI-driven browser tasks. One tool for
each of the 22 tools on Anakin's hosted MCP server (`mcp.anakin.io`), built on
the official [`anakin-sdk`](https://github.com/Anakin-Inc/anakin-py) Python
client (not a separate HTTP implementation).

## Install

```bash
pip install -U langchain-anakin
export ANAKIN_API_KEY="ask-..."   # free at https://anakin.io/signup
```

No key yet? `AnakinScrapeTool`, `AnakinWireDiscoverTool`,
`AnakinWireCatalogTool` and `AnakinWireReadActionTool` work without one on a
free per-IP allowance. The other tools return a `ConfigurationError` message
with a signup link.

## Tools

| Tool | What it does | Needs key |
|---|---|---|
| **Web data** | | |
| `AnakinScrapeTool` | Scrape a URL to markdown, or AI-extract JSON (`generate_json` / `output_schema`) | No |
| `AnakinSearchTool` | Synchronous AI-powered web search with citations | Yes |
| `AnakinMapTool` | List the URLs on a site, optionally filtered by a search term | Yes |
| `AnakinCrawlTool` | Crawl a site and return each page's markdown | Yes |
| `AnakinAgenticSearchTool` | Multi-stage research: search, scrape citations, extract structured data with a summary (1-5 min) | Yes |
| **Wire: pre-built site actions** | | |
| `AnakinWireDiscoverTool` | Find a pre-built Wire action for a task on a site | No |
| `AnakinWireCatalogTool` | Every supported site, or one site's actions with their parameters | No |
| `AnakinWireReadActionTool` | Run an action that reads data | No |
| `AnakinWireWriteActionTool` | Run an action that changes something on the site \* | Yes |
| `AnakinWireIdentitiesTool` | The user's saved site logins | Yes |
| `AnakinWireLoginTool` | Sign in to a site and get a credential \* | Yes |
| `AnakinWireBuildTool` | Request actions for a site Wire doesn't cover yet \* | Yes |
| `AnakinWireBuildStatusTool` | Progress of those requests | Yes |
| **Monitors** | | |
| `AnakinMonitorCreateTool` | Watch a page, a site or a Wire action on a schedule \* | Yes |
| `AnakinMonitorListTool` | The user's monitors, or one monitor's settings | Yes |
| `AnakinMonitorChangesTool` | Changes a monitor detected | Yes |
| `AnakinMonitorControlTool` | Pause, resume, run now or delete a monitor \* | Yes |
| **AI visibility** | | |
| `AnakinAIVisibilitySearchTool` | Ask ChatGPT, Gemini and Google AI Overview the same question and compare | Yes |
| `AnakinAIVisibilitySourcesTool` | The answer engines available | Yes |
| **Browser** | | |
| `AnakinSessionListTool` | Saved browser logins, for pages behind a sign-in | Yes |
| `AnakinSessionDeleteTool` | Delete a saved browser login \* | Yes |
| `AnakinBrowserTaskTool` | An AI agent that drives a real browser through a multi-step task \* | Yes |

\* Left out by `AnakinToolkit(read_only=True)`.

Tool names follow the hosted server's, with an `anakin_` prefix
(`anakin_scrape`, `anakin_wire_read_action`, ...). Each argument's description
reaches the model in the tool's schema.

`AnakinWireRunTool` (0.3.0's single Wire runner) still works but is deprecated:
use `AnakinWireReadActionTool` and `AnakinWireWriteActionTool`.

## Usage

```python
from langchain_anakin import (
    AnakinScrapeTool,
    AnakinSearchTool,
    AnakinWireDiscoverTool,
    AnakinWireReadActionTool,
)

AnakinScrapeTool().invoke({"url": "https://example.com"})

AnakinScrapeTool().invoke({
    "url": "https://example.com/product",
    "output_schema": {"type": "object", "properties": {"price": {"type": "number"}}},
})

AnakinSearchTool().invoke({"prompt": "latest developments in AI agents"})

actions = AnakinWireDiscoverTool().invoke({"query": "search products on walmart"})
AnakinWireReadActionTool().invoke({"action_id": "walmart_search", "params": {"query": "phones"}})
```

Every tool also supports async (`await tool.ainvoke(...)`), backed by
`anakin.AsyncAnakin`.

Use with an agent. `AnakinToolkit` gives the agent every tool, sharing one
client:

```python
from langchain.agents import create_agent
from langchain_anakin import AnakinToolkit

agent = create_agent(model, tools=AnakinToolkit().get_tools())
```

To give the agent fewer tools (and a shorter prompt), pass individual tools
instead, as in the usage example above.

### Read-only agents

`AnakinToolkit(read_only=True)` leaves out the 7 tools that change something
on a website or in the user's Anakin account, or start a long paid job (marked
\* above). Use it for agents that should only look things up.

### Payments

Like the hosted MCP server, `AnakinWireWriteActionTool`, `AnakinWireBuildTool`
and `AnakinBrowserTaskTool` refuse anything that looks like it completes a
payment or moves money (pay, checkout, purchase, place order, buy now, wire
transfer, payout). Adding to a cart or filling in a form is allowed. Create
those tools, or the toolkit, with `allow_financial_actions=True` to turn the
check off.

### Errors

Errors (bad key, no credits, a Wire site that needs a connected account)
come back to the agent as text instead of raising, including any URL it
needs to fix the problem.

## Development

Requires `anakin-sdk` 0.2.x.

```bash
uv sync --group test
uv run pytest
```

## License

Apache-2.0.

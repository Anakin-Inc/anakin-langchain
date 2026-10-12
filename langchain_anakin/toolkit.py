"""Every Anakin tool in one toolkit, sharing one API client."""

from __future__ import annotations

import os
from typing import Any

from anakin import Anakin
from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import Field, SecretStr

from langchain_anakin._utilities import client_args
from langchain_anakin.tools import (
    AnakinAgenticSearchTool,
    AnakinAIVisibilitySearchTool,
    AnakinAIVisibilitySourcesTool,
    AnakinBrowserTaskTool,
    AnakinCrawlTool,
    AnakinMapTool,
    AnakinMonitorChangesTool,
    AnakinMonitorControlTool,
    AnakinMonitorCreateTool,
    AnakinMonitorListTool,
    AnakinScrapeTool,
    AnakinSearchTool,
    AnakinSessionDeleteTool,
    AnakinSessionListTool,
    AnakinWireBuildStatusTool,
    AnakinWireBuildTool,
    AnakinWireCatalogTool,
    AnakinWireDiscoverTool,
    AnakinWireIdentitiesTool,
    AnakinWireLoginTool,
    AnakinWireReadActionTool,
    AnakinWireWriteActionTool,
)

# One tool per tool on Anakin's hosted MCP server (mcp.anakin.io), in its order.
TOOLS: tuple[type[BaseTool], ...] = (
    AnakinScrapeTool,
    AnakinSearchTool,
    AnakinMapTool,
    AnakinCrawlTool,
    AnakinAgenticSearchTool,
    AnakinWireDiscoverTool,
    AnakinWireCatalogTool,
    AnakinWireReadActionTool,
    AnakinWireWriteActionTool,
    AnakinWireIdentitiesTool,
    AnakinWireLoginTool,
    AnakinWireBuildTool,
    AnakinWireBuildStatusTool,
    AnakinMonitorCreateTool,
    AnakinMonitorListTool,
    AnakinMonitorChangesTool,
    AnakinMonitorControlTool,
    AnakinAIVisibilitySearchTool,
    AnakinAIVisibilitySourcesTool,
    AnakinSessionListTool,
    AnakinSessionDeleteTool,
    AnakinBrowserTaskTool,
)

# Tools that change something on a website or in the user's Anakin account, or
# start a long paid job. `read_only=True` leaves them out.
WRITE_TOOLS = frozenset(
    {
        "anakin_wire_write_action",
        "anakin_wire_login",
        "anakin_wire_build",
        "anakin_monitor_create",
        "anakin_monitor_control",
        "anakin_session_delete",
        "anakin_browser_task",
    }
)


class AnakinToolkit(BaseToolkit):
    """Every Anakin tool, sharing one API client.

    Setup:
        ```bash
        pip install -U langchain-anakin
        export ANAKIN_API_KEY="your-api-key"
        ```

    Instantiation:
        ```python
        from langchain_anakin import AnakinToolkit

        toolkit = AnakinToolkit()
        ```

    Tools:
        ```python
        tools = toolkit.get_tools()
        ```

    Use within an agent:
        ```python
        from langchain.agents import create_agent

        agent = create_agent(model, tools=AnakinToolkit(read_only=True).get_tools())
        ```
    """

    anakin_api_key: SecretStr = Field(default=SecretStr(""))
    anakin_base_url: str | None = None
    read_only: bool = False
    """Leave out the 7 tools that change something or start a long paid job."""
    allow_financial_actions: bool = False
    """Let Wire writes, Wire builds and browser tasks run payment-like requests."""

    def get_tools(self) -> list[BaseTool]:
        """The toolkit's tools: all 22, or the 15 read-only ones."""
        key = self.anakin_api_key.get_secret_value() or os.environ.get("ANAKIN_API_KEY", "")
        client = Anakin(**client_args(SecretStr(key), self.anakin_base_url))
        tools: list[BaseTool] = []
        for tool_cls in TOOLS:
            if self.read_only and tool_cls.model_fields["name"].default in WRITE_TOOLS:
                continue
            kwargs: dict[str, Any] = {
                "client": client,
                "anakin_api_key": key,
                "anakin_base_url": self.anakin_base_url,
            }
            if "allow_financial_actions" in tool_cls.model_fields:
                kwargs["allow_financial_actions"] = self.allow_financial_actions
            tools.append(tool_cls(**kwargs))
        return tools

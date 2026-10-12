"""Tests for the tools added in 0.4.0 and for `AnakinToolkit`.

The SDK clients here are `create_autospec` mocks of real `Anakin` /
`AsyncAnakin` instances, so a call with an argument the SDK doesn't take
fails the test instead of passing silently.
"""

from __future__ import annotations

import inspect
import json
import re
import warnings
from typing import Any
from unittest.mock import AsyncMock, create_autospec

import httpx
import pytest
from anakin import Anakin, AsyncAnakin
from anakin.models import AIVisibilitySearch, WireResult
from langchain_core.messages import ToolMessage

import langchain_anakin.tools as tools_module
from langchain_anakin import (
    AnakinAIVisibilitySearchTool,
    AnakinAIVisibilitySourcesTool,
    AnakinBrowserTaskTool,
    AnakinMonitorChangesTool,
    AnakinMonitorControlTool,
    AnakinMonitorCreateTool,
    AnakinMonitorListTool,
    AnakinSessionDeleteTool,
    AnakinSessionListTool,
    AnakinToolkit,
    AnakinWireBuildStatusTool,
    AnakinWireBuildTool,
    AnakinWireCatalogTool,
    AnakinWireIdentitiesTool,
    AnakinWireLoginTool,
    AnakinWireReadActionTool,
    AnakinWireRunTool,
    AnakinWireWriteActionTool,
)
from langchain_anakin.toolkit import WRITE_TOOLS

# Each tool on Anakin's hosted MCP server (mcp.anakin.io) and its parameters,
# in snake_case, as of October 2026.
HOSTED = {
    "scrape": {"url", "country", "use_browser", "generate_json", "force_fresh",
               "session_id", "session_name"},
    "search": {"prompt", "limit"},
    "map": {"url", "limit", "depth", "limit_per_level", "search", "include_subdomains",
            "include_external_links", "use_browser"},
    "crawl": {"url", "max_pages", "depth", "include_patterns", "exclude_patterns", "country",
              "use_browser", "session_id", "session_name"},
    "agentic_search": {"prompt", "schema", "use_browser"},
    "wire_discover": {"q", "limit"},
    "wire_catalog": {"slug"},
    "wire_read_action": {"action_id", "params", "credential_id", "identity_id"},
    "wire_write_action": {"action_id", "params", "credential_id", "identity_id"},
    "wire_identities": {"catalog_id"},
    "wire_login": {"catalog_slug", "params", "identity_name", "source_id", "source_ref"},
    "wire_build": {"website_url", "goal", "actions", "catalog_id", "country", "credential",
                   "visibility", "force"},
    "wire_build_status": {"id", "status", "limit", "page", "include_events"},
    "monitor_create": {"url", "interval_minutes", "scope", "watch_mode", "watch_format",
                       "output_schema", "ai_mode", "ai_goal", "alert_webhook_url",
                       "alert_emails", "use_browser", "country", "session_id", "max_pages",
                       "max_depth", "include_patterns", "exclude_patterns", "wire_action_id",
                       "wire_catalog_slug", "wire_params", "wire_credential_id",
                       "wire_watch_paths", "is_active", "expires_at"},
    "monitor_list": {"id"},
    "monitor_changes": {"id"},
    "monitor_control": {"id", "action"},
    "ai_visibility_search": {"query", "sources", "country", "include_full_content"},
    "ai_visibility_sources": set(),
    "session_list": {"domain"},
    "session_delete": {"id"},
    "browser_task": {"prompt", "url", "session_id", "output_schema", "max_steps", "timeout_ms"},
}  # fmt: skip
# Where a tool here names a hosted parameter differently.
RENAMED = {
    ("wire_discover", "q"): "query",
    ("wire_build_status", "id"): "build_id",
    ("monitor_list", "id"): "monitor_id",
    ("monitor_changes", "id"): "monitor_id",
    ("monitor_control", "id"): "monitor_id",
    ("session_delete", "id"): "session_id",
}
# Hosted parameters the Python SDK has no option for.
NOT_IN_SDK = {("wire_build_status", "include_events")}


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANAKIN_API_KEY", "ask_test_dummy")


def sdk() -> Any:
    """A mock with the real sync client's method signatures."""
    return create_autospec(Anakin(api_key="ask_test_dummy"))


@pytest.fixture
def async_sdk(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The per-call AsyncAnakin, as a mock with the real async signatures."""
    client = create_autospec(AsyncAnakin(api_key="ask_test_dummy"))
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    monkeypatch.setattr(tools_module, "new_async_client", lambda *_: client)
    return client


def resolve(client: Any, path: str) -> Any:
    for part in path.split("."):
        client = getattr(client, part)
    return client


# ─── toolkit ──────────────────────────────────────────────────────────────────


def test_toolkit_has_one_tool_per_hosted_tool_in_order() -> None:
    tools = AnakinToolkit().get_tools()

    assert [t.name for t in tools] == [f"anakin_{name}" for name in HOSTED]
    assert len({id(t.client) for t in tools}) == 1  # type: ignore[attr-defined]


def test_read_only_toolkit_drops_exactly_the_write_tools() -> None:
    tools = AnakinToolkit(read_only=True).get_tools()

    assert len(WRITE_TOOLS) == 7
    assert {t.name for t in tools} == {f"anakin_{n}" for n in HOSTED} - WRITE_TOOLS


def test_toolkit_passes_allow_financial_actions_to_the_guarded_tools() -> None:
    tools = AnakinToolkit(allow_financial_actions=True).get_tools()

    guarded = {t.name for t in tools if getattr(t, "allow_financial_actions", False)}
    assert guarded == {"anakin_wire_write_action", "anakin_wire_build", "anakin_browser_task"}


def test_tools_take_every_hosted_parameter() -> None:
    tools = {t.name: t for t in AnakinToolkit().get_tools()}
    for name, hosted in HOSTED.items():
        params = set(inspect.signature(tools[f"anakin_{name}"]._run).parameters)
        expected = {RENAMED.get((name, p), p) for p in hosted if (name, p) not in NOT_IN_SDK}
        assert expected <= params, f"{name} is missing {sorted(expected - params)}"


# LangChain itself builds this schema, and pydantic warns that agentic search's
# `schema` argument shadows BaseModel.schema(), as it did in 0.3.0.
@pytest.mark.filterwarnings('ignore:Field name "schema":UserWarning')
def test_every_argument_reaches_the_model_with_a_description() -> None:
    for tool in AnakinToolkit().get_tools():
        properties = tool.tool_call_schema.model_json_schema().get("properties", {})
        assert "run_manager" not in properties
        for arg, schema in properties.items():
            assert schema.get("description"), f"{tool.name}.{arg} has no description"


# ─── each tool calls the SDK the way the SDK expects ──────────────────────────

CALLS = [
    (AnakinWireCatalogTool, {}, "wire.catalogs", ((), {})),
    (AnakinWireCatalogTool, {"slug": "walmart"}, "wire.catalog", (("walmart",), {})),
    (AnakinWireIdentitiesTool, {"catalog_id": "c1"}, "wire.identities",
     ((), {"catalog_id": "c1"})),
    (AnakinWireLoginTool, {"catalog_slug": "neb", "params": {"email": "e"}}, "wire.login",
     (("neb",), {"params": {"email": "e"}, "identity_name": None, "source_id": None,
                 "source_ref": None})),
    (AnakinWireBuildTool, {"website_url": "https://x.com", "goal": "list jobs"}, "wire.build",
     (("https://x.com", "list jobs"), {"actions": None, "catalog_id": None, "country": None,
                                       "credential": None, "visibility": None, "force": False})),
    (AnakinWireBuildStatusTool, {"build_id": "b1"}, "wire.get_build", (("b1",), {})),
    (AnakinWireBuildStatusTool, {"status": "pending", "page": 2}, "wire.builds",
     ((), {"status": "pending", "page": 2, "limit": None})),
    (AnakinMonitorListTool, {}, "monitors.list", ((), {})),
    (AnakinMonitorListTool, {"monitor_id": "m1"}, "monitors.get", (("m1",), {})),
    (AnakinMonitorChangesTool, {"monitor_id": "m1"}, "monitors.changes", (("m1",), {})),
    (AnakinMonitorControlTool, {"monitor_id": "m1", "action": "pause"}, "monitors.pause",
     (("m1",), {})),
    (AnakinMonitorControlTool, {"monitor_id": "m1", "action": "run_now"}, "monitors.run_now",
     (("m1",), {})),
    (AnakinAIVisibilitySourcesTool, {}, "ai_visibility.sources", ((), {})),
    (AnakinSessionListTool, {"domain": "amazon.com"}, "sessions.list",
     ((), {"domain": "amazon.com"})),
]  # fmt: skip


@pytest.mark.parametrize(("tool_cls", "args", "path", "expected"), CALLS)
def test_tools_call_the_sdk(tool_cls: Any, args: dict, path: str, expected: tuple) -> None:
    tool = tool_cls()
    tool.client = sdk()
    resolve(tool.client, path).return_value = {"ok": True}

    assert tool.invoke(args) == {"ok": True}
    resolve(tool.client, path).assert_called_once_with(*expected[0], **expected[1])


@pytest.mark.parametrize(("tool_cls", "args", "path", "expected"), CALLS)
async def test_tools_call_the_async_sdk(
    async_sdk: Any, tool_cls: Any, args: dict, path: str, expected: tuple
) -> None:
    resolve(async_sdk, path).return_value = {"ok": True}

    assert await tool_cls().ainvoke(args) == {"ok": True}
    resolve(async_sdk, path).assert_awaited_once_with(*expected[0], **expected[1])


def test_wire_read_action_runs_with_a_key() -> None:
    tool = AnakinWireReadActionTool()
    tool.client = sdk()
    tool.client.api_key_configured = True
    tool.client.wire.run.return_value = WireResult(status="completed", data=[1], credits_used=1)

    result = tool.invoke({"action_id": "walmart_search", "params": {"q": "tv"}, "identity_id": "i1"})

    assert result == {"status": "completed", "data": [1], "credits_used": 1}
    tool.client.wire.run.assert_called_once_with(
        "walmart_search", {"q": "tv"}, credential_id=None, identity_id="i1"
    )


def test_wire_read_action_without_a_key_uses_zero_touch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANAKIN_API_KEY")
    tool = AnakinWireReadActionTool()
    tool.client = sdk()
    tool.client.api_key_configured = False
    tool.client.wire.zero_touch.return_value = WireResult(status="completed", data={"ok": 1})

    assert tool.invoke({"action_id": "walmart_search"})["data"] == {"ok": 1}
    tool.client.wire.zero_touch.assert_called_once_with("walmart_search", None)
    tool.client.wire.run.assert_not_called()


def test_wire_write_action_without_a_key_explains_how_to_get_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANAKIN_API_KEY")

    result = AnakinWireWriteActionTool().invoke({"action_id": "shop_add_to_cart"})

    assert result["error"].startswith("ConfigurationError")
    assert "API key" in result["error"]


async def test_wire_write_action_async(async_sdk: Any) -> None:
    async_sdk.wire.run.return_value = WireResult(status="completed", data={"added": True})

    result = await AnakinWireWriteActionTool().ainvoke(
        {"action_id": "shop_add_to_cart", "params": {"sku": "1"}, "credential_id": "c1"}
    )

    assert result["data"] == {"added": True}
    async_sdk.wire.run.assert_awaited_once_with(
        "shop_add_to_cart", {"sku": "1"}, credential_id="c1", identity_id=None
    )


# ─── payments are refused unless allowed ──────────────────────────────────────


@pytest.mark.parametrize(
    ("action_id", "params"),
    [
        ("shop_checkout", {}),
        ("shop.place-order", {}),
        ("bank.send", {"note": "wire transfer to vendor"}),
        ("store.order", {"step": "buy now"}),
    ],
)
def test_write_actions_that_look_like_payments_are_refused(action_id: str, params: dict) -> None:
    tool = AnakinWireWriteActionTool()
    tool.client = sdk()

    result = tool.invoke({"action_id": action_id, "params": params})

    assert result["error"].startswith("Refused: this action looks like a payment")
    tool.client.wire.run.assert_not_called()


def test_ordinary_writes_run_and_the_guard_can_be_turned_off() -> None:
    tool = AnakinWireWriteActionTool()
    tool.client = sdk()
    tool.client.wire.run.return_value = WireResult(status="completed")
    tool.invoke({"action_id": "shop_add_to_cart"})
    tool.client.wire.run.assert_called_once()

    allowed = AnakinWireWriteActionTool(allow_financial_actions=True)
    allowed.client = sdk()
    allowed.client.wire.run.return_value = WireResult(status="completed")
    allowed.invoke({"action_id": "shop_checkout"})
    allowed.client.wire.run.assert_called_once()


def test_wire_build_refuses_payment_builds() -> None:
    tool = AnakinWireBuildTool()
    tool.client = sdk()

    result = tool.invoke({"website_url": "https://shop.example", "goal": "pay now with saved card"})

    assert result["error"].startswith("Refused: this build")
    tool.client.wire.build.assert_not_called()


# ─── monitors, AI visibility, sessions ────────────────────────────────────────


def test_monitor_create_sends_only_the_options_given_and_the_sdk_accepts_them() -> None:
    from anakin._ops import monitor_body  # the SDK's own option-name check

    tool = AnakinMonitorCreateTool()
    tool.client = sdk()
    tool.client.monitors.create.return_value = {"id": "m1"}
    every_option = {
        "scope": "site", "watch_mode": "specific_data", "watch_format": "html",
        "output_schema": {"type": "object"}, "ai_mode": True, "ai_goal": "price drops",
        "alert_webhook_url": "https://hook", "alert_emails": ["a@b.c"], "use_browser": True,
        "country": "gb", "session_id": "s1", "max_pages": 5, "max_depth": 2,
        "include_patterns": ["/p/*"], "exclude_patterns": ["/x"], "wire_action_id": "a1",
        "wire_catalog_slug": "shop", "wire_params": {"q": 1}, "wire_credential_id": "c1",
        "wire_watch_paths": ["$.price"], "is_active": False, "expires_at": "2026-12-31",
    }  # fmt: skip

    tool.invoke({"url": "https://a.com", "interval_minutes": 60, **every_option})

    call = tool.client.monitors.create.call_args
    assert call.args == ("https://a.com", 60)
    assert call.kwargs == every_option
    monitor_body("https://a.com", 60, call.kwargs)  # raises TypeError on an unknown option

    tool.client.monitors.create.reset_mock()
    tool.invoke({"url": "https://a.com", "interval_minutes": 60})
    assert tool.client.monitors.create.call_args.kwargs == {
        "scope": "page", "watch_mode": "full_page", "ai_mode": False, "use_browser": False,
    }  # fmt: skip


def test_monitor_control_delete() -> None:
    tool = AnakinMonitorControlTool()
    tool.client = sdk()

    assert tool.invoke({"monitor_id": "m1", "action": "delete"}) == {"id": "m1", "deleted": True}
    tool.client.monitors.delete.assert_called_once_with("m1")


def test_session_delete() -> None:
    tool = AnakinSessionDeleteTool()
    tool.client = sdk()

    assert tool.invoke({"session_id": "s1"}) == {"id": "s1", "deleted": True}
    tool.client.sessions.delete.assert_called_once_with("s1")


def test_ai_visibility_search_leaves_out_full_answers_unless_asked() -> None:
    tool = AnakinAIVisibilitySearchTool()
    tool.client = sdk()
    tool.client.ai_visibility.search.return_value = AIVisibilitySearch.model_validate(
        {
            "search_id": "v1",
            "status": "completed",
            "synthesis": "Both name Anakin.",
            "results": [{"source": "chatgpt", "status": "completed", "full_content": "long"}],
        }
    )

    short = tool.invoke({"query": "best scraping api"})
    assert short["synthesis"] == "Both name Anakin."
    assert "full_content" not in short["results"][0]
    full = tool.invoke({"query": "best scraping api", "include_full_content": True})
    assert full["results"][0]["full_content"] == "long"
    tool.client.ai_visibility.search.assert_called_with(
        "best scraping api", sources=None, country="us"
    )


def test_a_tool_call_comes_back_as_a_tool_message() -> None:
    tool = AnakinSessionListTool()
    tool.client = sdk()
    tool.client.sessions.list.return_value = [{"id": "s1", "domain": "amazon.com"}]

    message = tool.invoke(
        {"type": "tool_call", "id": "call_1", "name": tool.name, "args": {"domain": "amazon.com"}}
    )

    assert isinstance(message, ToolMessage)
    assert message.tool_call_id == "call_1"
    assert json.loads(message.content)[0]["id"] == "s1"  # type: ignore[arg-type]


# ─── browser tasks talk to the REST API directly ──────────────────────────────


def mock_api(monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]) -> list[httpx.Request]:
    seen: list[httpx.Request] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    transport = httpx.MockTransport(handler)
    real_client, real_async_client = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(
        tools_module.httpx, "Client", lambda **kw: real_client(transport=transport, **kw)
    )
    monkeypatch.setattr(
        tools_module.httpx,
        "AsyncClient",
        lambda **kw: real_async_client(transport=transport, **kw),
    )
    monkeypatch.setattr(tools_module.time, "sleep", lambda _: None)
    monkeypatch.setattr(tools_module.asyncio, "sleep", AsyncMock())
    return seen


def task_responses() -> list[httpx.Response]:
    """A browser task that is accepted, still running, then completed."""
    return [
        httpx.Response(202, json={"workflow_id": "wf_1", "status": "queued"}),
        httpx.Response(200, json={"status": "running"}),
        httpx.Response(200, json={"status": "completed", "result": {"price": "$9"}}),
    ]


def test_browser_task_starts_then_polls_until_completed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = mock_api(monkeypatch, task_responses())

    result = AnakinBrowserTaskTool().invoke(
        {"prompt": "find the price", "url": "https://a.com", "max_steps": 10}
    )

    assert result == {"price": "$9"}
    assert seen[0].method == "POST"
    assert seen[0].url.path == "/v1/ai/evaluate"
    assert seen[0].headers["X-API-Key"] == "ask_test_dummy"
    assert json.loads(seen[0].content) == {
        "prompt": "find the price", "async": True, "url": "https://a.com", "max_steps": 10,
    }  # fmt: skip
    assert seen[1].url.path == "/v1/ai/jobs/wf_1"


async def test_browser_task_async(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = mock_api(monkeypatch, task_responses())

    assert await AnakinBrowserTaskTool().ainvoke({"prompt": "find the price"}) == {"price": "$9"}
    assert [r.url.path for r in seen] == ["/v1/ai/evaluate", "/v1/ai/jobs/wf_1", "/v1/ai/jobs/wf_1"]


def test_browser_task_failure_and_api_errors_come_back_as_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_api(
        monkeypatch,
        [
            httpx.Response(202, json={"workflow_id": "wf_1"}),
            httpx.Response(200, json={"status": "failed", "error": "page blocked"}),
            httpx.Response(401, json={"message": "Invalid API key"}),
        ],
    )
    tool = AnakinBrowserTaskTool()

    failed = tool.invoke({"prompt": "do it"})["error"]
    assert failed.startswith("AnakinError: Browser task failed: page blocked")
    unauthorized = tool.invoke({"prompt": "do it"})["error"]
    assert unauthorized.startswith("AuthenticationError: Invalid API key | status=401")


def test_browser_task_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANAKIN_API_KEY")
    seen = mock_api(monkeypatch, [])

    result = AnakinBrowserTaskTool().invoke({"prompt": "do it"})

    assert result["error"].startswith("ConfigurationError")
    assert seen == []


@pytest.mark.parametrize(
    ("prompt", "url"),
    [
        ("Buy now the cheapest TV and pay with the saved card", None),
        ("Finish the order", "https://shop.example/checkout"),
    ],
)
def test_browser_task_refuses_payment_tasks_before_calling_the_api(
    monkeypatch: pytest.MonkeyPatch, prompt: str, url: str | None
) -> None:
    seen = mock_api(monkeypatch, [])

    result = AnakinBrowserTaskTool().invoke({"prompt": prompt, "url": url})

    assert result["error"].startswith("Refused: this task")
    assert seen == []


# ─── the old single Wire runner ───────────────────────────────────────────────


def test_wire_run_tool_is_deprecated_but_still_runs() -> None:
    with pytest.warns(DeprecationWarning, match="AnakinWireReadActionTool"):
        tool = AnakinWireRunTool()
    tool.client = sdk()
    tool.client.api_key_configured = True
    tool.client.wire.run.return_value = WireResult(status="completed", data=[1])

    assert tool.invoke({"action_id": "walmart_search"})["data"] == [1]


def test_new_tools_do_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AnakinToolkit().get_tools()


@pytest.mark.filterwarnings('ignore:Field name "schema":UserWarning')
def test_every_tool_converts_to_a_function_definition_for_chat_models() -> None:
    # What chat model integrations do in bind_tools().
    from langchain_core.utils.function_calling import convert_to_openai_tool

    for tool in AnakinToolkit().get_tools():
        function = convert_to_openai_tool(tool)["function"]
        assert function["name"] == tool.name
        assert function["description"]
        properties = function["parameters"].get("properties", {})
        assert all(p.get("description") for p in properties.values()), tool.name
        json.dumps(function)


def test_no_wrapped_docstring_line_reads_as_a_new_argument() -> None:
    # langchain-core 0.3.0 (our lowest supported version) parses an indented
    # continuation line like "Default: ..." in an Args section as a new
    # argument, and every tool then fails to build.
    source = inspect.getsource(tools_module)
    risky = [line for line in source.splitlines() if re.match(r"^ {16}[A-Za-z_]+: ", line)]
    assert risky == []

import os
from typing import Any

from anakin import Anakin, AsyncAnakin
from langchain_core.utils import convert_to_secret_str
from pydantic import SecretStr


def client_args(api_key: SecretStr, base_url: str | None) -> dict[str, Any]:
    """Constructor kwargs shared by the sync and async Anakin clients.

    An empty key is passed as `None`, which puts the SDK in keyless "Zero
    Touch" mode: `scrape` and Wire discovery still work, and every other
    call raises `anakin.ConfigurationError` with a signup link.
    """
    args: dict[str, Any] = {"api_key": api_key.get_secret_value() or None}
    if base_url:
        args["base_url"] = base_url
    return args


def initialize_client(values: dict) -> dict:
    """Initialize the Anakin SDK client from a LangChain tool's field values.

    Mirrors langchain-exa's `initialize_client` pattern: pull the API key
    from the field or `ANAKIN_API_KEY`, wrap it as a SecretStr, construct
    the client once (`anakin.Anakin`, the same official SDK client used
    everywhere else in this codebase — no separate HTTP implementation
    here).
    """
    anakin_api_key = (
        values.get("anakin_api_key") or os.environ.get("ANAKIN_API_KEY") or ""
    )
    values["anakin_api_key"] = convert_to_secret_str(anakin_api_key)
    if values.get("client") is None:
        values["client"] = Anakin(
            **client_args(values["anakin_api_key"], values.get("anakin_base_url"))
        )
    return values


def new_async_client(api_key: SecretStr, base_url: str | None) -> AsyncAnakin:
    """A fresh `AsyncAnakin` for one `_arun` call.

    Not cached on the tool: an httpx async pool is bound to the event loop
    that first used it, and a tool instance can outlive that loop.
    """
    return AsyncAnakin(**client_args(api_key, base_url))

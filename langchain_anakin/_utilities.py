import os

from anakin import Anakin
from langchain_core.utils import convert_to_secret_str


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
    args = {"api_key": values["anakin_api_key"].get_secret_value()}
    if values.get("anakin_base_url"):
        args["base_url"] = values["anakin_base_url"]
    values["client"] = Anakin(**args)
    return values

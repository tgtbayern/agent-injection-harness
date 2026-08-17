"""Model access. One gateway, one client, no per-vendor branching."""

from __future__ import annotations

import os

from .base import LLMClient, LLMResponse, ProviderError, ToolCall, parse_json_action
from .mock import MockClient
from .openai_compat import OpenAICompatClient, explain
from .probe import ProbeResult, probe_model

DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")


def build_client(config: dict) -> LLMClient:
    """Build a client from a model config dict.

    `{"model_name": "mock"}` yields the offline scripted client, so the whole
    pipeline runs with no key and no network -- that is what the tests use.
    """
    model = config.get("model_name") or config.get("model") or "mock"
    if model == "mock":
        return MockClient(
            seed=int(config.get("seed", 0)),
            susceptibility=float(config.get("susceptibility", 0.75)),
        )
    api_key = config.get("api_key") or os.getenv("LLM_API_KEY", "")
    return OpenAICompatClient(
        model=model,
        api_key=api_key,
        base_url=config.get("base_url") or DEFAULT_BASE_URL,
        tool_mode=config.get("tool_mode", "native"),
        display_name=config.get("display_name"),
        group=config.get("group"),
    )


__all__ = [
    "DEFAULT_BASE_URL",
    "LLMClient",
    "LLMResponse",
    "MockClient",
    "OpenAICompatClient",
    "ProbeResult",
    "ProviderError",
    "ToolCall",
    "build_client",
    "explain",
    "parse_json_action",
    "probe_model",
]

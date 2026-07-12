from __future__ import annotations

from typing import Any

from .base import LLMClient, LLMResponse, Message, ToolCall, ToolChoice, ToolResultBlock, Usage
from .config import LLMConfig
from .factory import ProviderFactory, register_provider, registered_providers, resolve_llm
from .policy import ProviderPolicy, ProviderPolicyViolation
from .scripted import ScriptedClient

__all__ = [
    "AnthropicClient",
    "GeminiClient",
    "LLMClient",
    "LLMConfig",
    "LLMResponse",
    "Message",
    "OpenAIClient",
    "ProviderFactory",
    "ProviderPolicy",
    "ProviderPolicyViolation",
    "register_provider",
    "registered_providers",
    "resolve_llm",
    "ScriptedClient",
    "ToolCall",
    "ToolChoice",
    "ToolResultBlock",
    "Usage",
]


def __getattr__(name: str) -> Any:
    # Lazy: importing governed.llm must not require the anthropic/openai/
    # google-genai SDKs to be installed unless you actually construct their
    # clients (directly, or indirectly via resolve_llm/LLMConfig).
    if name == "AnthropicClient":
        from .anthropic_client import AnthropicClient

        return AnthropicClient
    if name == "OpenAIClient":
        from .openai_client import OpenAIClient

        return OpenAIClient
    if name == "GeminiClient":
        from .gemini_client import GeminiClient

        return GeminiClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

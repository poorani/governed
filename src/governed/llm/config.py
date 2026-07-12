"""LLM configuration: describe a model by data, not by constructing a client.

``LLMConfig`` is the config-driven alternative to importing a provider's
``LLMClient`` subclass directly. Pass one to ``AgentConfig(llm=...)`` and
``resolve_llm`` (see ``factory.py``) instantiates the right adapter -- the
caller never imports ``AnthropicClient``, ``OpenAIClient``, or any vendor SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["LLMConfig"]


@dataclass
class LLMConfig:
    """Provider-agnostic description of a model. Resolved by ``resolve_llm``.

    ``extra`` is forwarded as keyword arguments to the adapter's constructor --
    the escape hatch for provider-specific knobs (``extra_headers`` on
    Anthropic, an injected ``client=`` test double, and so on) without growing
    this dataclass per provider.
    """

    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

"""Maps a provider name to an adapter. The one place that knows all of them.

``AgentConfig(llm=LLMConfig(provider="openai", model="gpt-4.1"))`` resolves
here into an ``OpenAIClient`` -- the caller never imports the adapter class.
Swapping providers is a config edit, not a code change.

Built-in providers are registered behind small builder functions that import
the vendor SDK only when actually invoked, so importing this module never
requires ``anthropic``, ``openai``, or ``google-genai`` to be installed.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import LLMClient
from .config import LLMConfig
from .policy import ProviderPolicy

__all__ = ["ProviderFactory", "register_provider", "registered_providers", "resolve_llm"]

#: ``(config) -> a ready-to-use LLMClient``.
ProviderFactory = Callable[[LLMConfig], LLMClient]

_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Add or replace the adapter used for ``LLMConfig(provider=name, ...)``.

    Call this to plug in a provider governed doesn't ship: ``factory``
    receives the ``LLMConfig`` and must return a ready-to-use ``LLMClient``.
    Provider names are matched case-insensitively.
    """
    _REGISTRY[name.lower()] = factory


def registered_providers() -> list[str]:
    return sorted(_REGISTRY)


def resolve_llm(
    llm: LLMClient | LLMConfig, *, policy: ProviderPolicy | None = None
) -> LLMClient:
    """Return a usable ``LLMClient``, constructing one from config if needed.

    Already have an ``LLMClient`` -- built it yourself, or it's
    ``ScriptedClient`` in a test? It passes through unchanged, and ``policy``
    is not consulted (see ``ProviderPolicy``'s docstring for why: an opaque,
    already-built client has no provider/model this function can check). Have
    an ``LLMConfig`` instead? The provider named in it is looked up in the
    registry and instantiated -- after ``policy.check`` passes, if a policy
    was given. This is what lets ``AgentConfig(llm=...)`` accept either: the
    agent never needs to know which one it got.
    """
    if isinstance(llm, LLMClient):
        return llm
    if not isinstance(llm, LLMConfig):
        raise TypeError(
            f"AgentConfig.llm must be an LLMClient or LLMConfig, got {type(llm).__name__}"
        )
    if policy is not None:
        policy.check(llm)
    key = llm.provider.lower()
    factory = _REGISTRY.get(key)
    if factory is None:
        raise ValueError(
            f"Unknown LLM provider {llm.provider!r}. Registered providers: "
            f"{registered_providers()}. Call register_provider() to add your own."
        )
    return factory(llm)


def _build_anthropic(cfg: LLMConfig) -> LLMClient:
    from .anthropic_client import AnthropicClient

    return AnthropicClient(
        model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url, **cfg.extra
    )


def _build_openai(cfg: LLMConfig) -> LLMClient:
    from .openai_client import OpenAIClient

    return OpenAIClient(
        model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url, **cfg.extra
    )


def _build_gemini(cfg: LLMConfig) -> LLMClient:
    from .gemini_client import GeminiClient

    return GeminiClient(
        model=cfg.model, api_key=cfg.api_key, base_url=cfg.base_url, **cfg.extra
    )


register_provider("anthropic", _build_anthropic)
register_provider("openai", _build_openai)
register_provider("gemini", _build_gemini)

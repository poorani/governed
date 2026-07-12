"""Model/provider governance: which providers and models a deployment may use.

Complements ``governed.security.policy.GovernancePolicy`` (which governs
tools and actions) with the analogous restriction on the *model* side. A
``ProviderPolicy`` is checked by ``resolve_llm`` before an adapter is ever
constructed -- pass it once via ``AgentConfig(provider_policy=...)`` and every
``LLMConfig`` resolved through that config is validated against it. That is
what makes "only these providers, only these models" an enforced property of
a deployment instead of a convention someone has to remember to follow.

Scope, stated plainly: this only intercepts the config-driven path
(``LLMConfig`` resolved through ``resolve_llm``). An already-constructed
``LLMClient`` passed directly to ``AgentConfig(llm=...)`` bypasses it, because
there is no reliable way to recover "which provider and model is this opaque
object" from an arbitrary client instance. Deployments that need this policy
enforced should standardise on ``LLMConfig`` as their integration contract
rather than constructing adapters directly -- see ``docs/RESPONSIBLE_AI.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import LLMConfig

__all__ = ["ProviderPolicy", "ProviderPolicyViolation"]


class ProviderPolicyViolation(Exception):
    """Raised by ``resolve_llm`` when an ``LLMConfig`` violates a ``ProviderPolicy``."""


@dataclass
class ProviderPolicy:
    """``None`` on a field means "no restriction on that axis."

    ``allowed_models`` is keyed by lower-cased provider name. A provider
    absent from this dict is unrestricted on model, so long as it clears
    ``allowed_providers``; a provider present in it is restricted to exactly
    the models listed.
    """

    allowed_providers: frozenset[str] | None = None
    allowed_models: dict[str, frozenset[str]] = field(default_factory=dict)

    def check(self, config: LLMConfig) -> None:
        provider = config.provider.lower()
        if self.allowed_providers is not None and provider not in self.allowed_providers:
            raise ProviderPolicyViolation(
                f"Provider {config.provider!r} is not permitted by ProviderPolicy. "
                f"Allowed providers: {sorted(self.allowed_providers)}."
            )
        models = self.allowed_models.get(provider)
        if models is not None and config.model not in models:
            raise ProviderPolicyViolation(
                f"Model {config.model!r} is not permitted for provider "
                f"{config.provider!r} by ProviderPolicy. Allowed models: {sorted(models)}."
            )

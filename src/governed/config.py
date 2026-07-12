"""Configuration: one dataclass, sane defaults, no global state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .llm.base import LLMClient
from .llm.config import LLMConfig
from .llm.factory import resolve_llm
from .llm.policy import ProviderPolicy
from .memory.optimizer import CircuitBreakerConfig, CostConfig
from .memory.store import InMemoryStore, StateStore
from .memory.transcript import CompactionConfig
from .observability.decision_ledger import DecisionLedgerConfig
from .observability.events import Subscriber
from .skills.loader import SkillConfig, SkillLibrary
from .tools.base import DANGEROUS, ToolConfig, ToolSpec

#: ``(spec, arguments) -> bool``. Return False to deny the call.
ApprovalFn = Callable[[ToolSpec, dict[str, Any]], bool]

ApprovalPolicy = Literal["never", "dangerous", "always"]


@dataclass
class Budget:
    """Hard stops. Exceeding any of them ends the run with status `exhausted`."""

    max_iterations: int = 20
    max_tokens: int = 500_000
    max_tool_calls: int = 100
    max_wall_seconds: float = 900.0
    #: Bail out after this many consecutive `failure` evaluations.
    max_consecutive_failures: int = 3
    #: Retries allowed per phase when the model breaks the output contract.
    max_contract_retries: int = 2


def auto_approve(spec: ToolSpec, args: dict[str, Any]) -> bool:
    return True


def deny_all(spec: ToolSpec, args: dict[str, Any]) -> bool:
    return False


def cli_approve(spec: ToolSpec, args: dict[str, Any]) -> bool:
    """Blocking terminal prompt. Use for interactive sessions."""
    import json
    import sys

    print(f"\nApprove `{spec.name}` [{spec.safety.value}]?", file=sys.stderr)
    print(json.dumps(args, indent=2)[:2000], file=sys.stderr)
    return input("  y/N > ").strip().lower() in ("y", "yes")


@dataclass
class ObservabilityConfig:
    """Groups everything ``AgentConfig`` otherwise takes as five separate
    fields (``trace_path``, ``console``, ``verbose``, ``subscribers``,
    ``decision_ledger``) into one config-driven object -- for bootstrapping
    from a dict/YAML file where a nested ``observability: {...}`` block reads
    more naturally than five top-level keys.

    Passing this to ``AgentConfig(observability=...)`` **replaces** those five
    fields outright (see ``AgentConfig.__post_init__``); set them there, not
    both places, the same rule ``guardrails``/``approval_policy`` already
    follow for their own overlap.
    """

    trace_path: str | Path | None = None
    console: bool = True
    verbose: bool = False
    subscribers: list[Subscriber] = field(default_factory=list)
    decision_ledger: DecisionLedgerConfig | None = None


@dataclass
class FeatureToggleConfig:
    """Coarse on/off switches for subsystems that otherwise each have their
    own independent enablement convention (``guardrails=None`` vs.
    ``GuardrailConfig(enabled=)`` vs. ``DecisionLedgerConfig(enabled=)`` vs.
    "just don't add a ``TelemetryCollector`` to ``subscribers``"). This is a
    convenience layer over those, not a replacement -- every toggle here only
    fills a gap left unset; it never overrides an explicit, more specific
    configuration. Resolved in ``Agent.__init__`` (not here), because doing
    so needs ``GuardrailConfig``/``ContentSafetyScanner``, and this module is
    kept free of a ``security`` import on purpose -- see ``guardrails``'s
    field comment below.

    There is deliberately no ``memory`` toggle: session state is structural
    to the agent loop, not an optional subsystem, so there is nothing to turn
    off.
    """

    #: Build a default ``GuardrailConfig()`` if neither ``guardrails`` nor
    #: ``governance`` produced one. A no-op once either is set.
    guardrails: bool = False
    #: Add a zero-dependency ``ContentSafetyScanner(KeywordSafetyProvider())``
    #: to whatever guardrails end up in effect, if none is already there.
    #: Implies ``guardrails`` is effectively on too (a scanner needs a gateway
    #: to run in). For anything beyond the keyword-based reference provider
    #: (bias detection in particular), configure
    #: ``GuardrailConfig(content_safety_scanners=[...])`` directly instead.
    content_safety: bool = False
    #: Add a ``PIIScanner()`` to whatever guardrails end up in effect, if one
    #: isn't already present. Implies ``guardrails`` is effectively on too,
    #: same as ``content_safety``. Detects SSNs, payment card numbers,
    #: emails, and phone numbers in tool arguments/results and escalates to
    #: ``RiskTier.WARNING`` -- it never blocks. See ``PIIScanner``'s
    #: docstring for exactly what it does and doesn't catch.
    pii_detection: bool = False
    #: ``DecisionLedgerConfig(enabled=True)`` (default JSONL store under the
    #: workspace) if ``decision_ledger`` wasn't already set.
    decision_ledger: bool = False
    #: Attach a ``TelemetryCollector``, exposed back as ``Agent.telemetry``.
    telemetry: bool = False


@dataclass
class AgentConfig:
    """Everything the agent needs. ``llm`` is the only required field.

    ``llm`` accepts either a ready-made ``LLMClient`` (``AnthropicClient(...)``,
    ``ScriptedClient(...)``, your own subclass) or an ``LLMConfig`` -- a plain
    ``provider``/``model``/``api_key``/``base_url`` description that gets
    resolved into the right adapter automatically. The two are equivalent;
    ``LLMConfig`` exists so the provider can be swapped by editing config
    (a dict from YAML/env, say) rather than by changing which class the
    caller imports and constructs.
    """

    llm: LLMClient | LLMConfig

    # Model/provider governance. ``None`` (default) permits any provider the
    # factory knows about, or any provider you've registered yourself --
    # today's behaviour, unchanged. Set a ``ProviderPolicy`` to restrict a
    # deployment to an explicit allowlist of providers and models; it is
    # checked once, here, before ``llm`` is resolved into a client. Only
    # takes effect for the ``LLMConfig`` path -- see ``ProviderPolicy``'s
    # docstring.
    provider_policy: ProviderPolicy | None = None

    # Where the agent may touch the disk. Created if absent.
    workspace: str | Path = "./workspace"

    # Tools. Leave as None to get `default_tools(skills)`. Accepts a live
    # `list[Tool]`, or a `ToolConfig` -- the data-only description resolved
    # by `governed.tools.resolve_tools`, for config-driven bootstrapping.
    tools: list[Any] | ToolConfig | None = None

    # Skills. Directories are scanned for `*/SKILL.md`. `skills` accepts a
    # live `SkillLibrary`, or a `SkillConfig` (data-only, resolved by
    # `governed.skills.resolve_skills`) as an alternative to `skills_dirs`.
    skills_dirs: list[str | Path] = field(default_factory=lambda: ["./skills"])
    skills: SkillLibrary | SkillConfig | None = None

    # Limits.
    budget: Budget = field(default_factory=Budget)
    tool_timeout_s: float = 60.0

    # Human in the loop.
    approval_policy: ApprovalPolicy = "never"
    approval_fn: ApprovalFn = auto_approve

    # Guardrails. ``None`` disables the gateway entirely and falls back to the
    # ToolSafety-based approval above. Set it to a GuardrailConfig for the
    # three-tier risk policy, the scanners, and HITL approval.
    guardrails: Any | None = None

    # Governance. A ``GovernancePolicy`` (governed.security.GovernancePolicy)
    # naming an explicit tool allowlist, deployment-mandated sensitive
    # operations, and an approval threshold. If set, it is folded into
    # ``guardrails`` (building one if you didn't supply one) -- see
    # ``GovernancePolicy.apply``. Typed loosely here, like ``guardrails``
    # above, to keep this module free of a ``security`` import; ``Agent``
    # validates the type at construction.
    governance: Any | None = None

    # Memory.
    store: StateStore = field(default_factory=InMemoryStore)
    compaction: CompactionConfig = field(default_factory=CompactionConfig)
    #: Fold old context through chunked, recursive summarisation rather than one
    #: oversized summarise call. Required once a prefix can exceed the window.
    recursive_compaction: bool = True
    checkpoint_every_iteration: bool = True

    # Money. Costing is pure arithmetic over the provider's own token counts, so
    # it is on by default. The circuit breaker's dollar ceiling is not: only you
    # know what this task is worth.
    cost: CostConfig = field(default_factory=CostConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)

    # Observability.
    trace_path: str | Path | None = None
    console: bool = True
    verbose: bool = False
    subscribers: list[Subscriber] = field(default_factory=list)

    # The decision ledger: an immutable, hash-chained record of every
    # iteration's plan/rationale/tool/safety-checks/evidence, plus a
    # guaranteed final-outcome record, independent of `trace_path`. Disabled
    # by default. See `governed.observability.decision_ledger` and
    # docs/RESPONSIBLE_AI.md.
    decision_ledger: DecisionLedgerConfig | None = None

    # Grouped alternative to trace_path/console/verbose/subscribers/
    # decision_ledger above -- set this *instead of* those five fields, not
    # alongside them; see ObservabilityConfig's docstring. Primarily for
    # config-driven bootstrapping (a nested `observability: {...}` block).
    observability: ObservabilityConfig | None = None

    # Coarse enable/disable switches for guardrails, content-safety
    # screening, the decision ledger, and telemetry, filled in only where
    # the more specific field above was left unset. See
    # FeatureToggleConfig's docstring for exactly what each toggle does.
    features: FeatureToggleConfig | None = None

    # Model knobs.
    max_tokens_per_call: int = 4096
    temperature: float = 0.0
    #: Appended verbatim to the system prompt. Domain rules, tone, constraints.
    extra_instructions: str = ""

    def __post_init__(self) -> None:
        self.llm = resolve_llm(self.llm, policy=self.provider_policy)
        self.workspace = Path(self.workspace)
        if self.observability is not None:
            self.trace_path = self.observability.trace_path
            self.console = self.observability.console
            self.verbose = self.observability.verbose
            self.subscribers = self.observability.subscribers
            self.decision_ledger = self.observability.decision_ledger
        if (
            self.guardrails is not None or self.governance is not None
        ) and self.approval_policy != "never":
            raise ValueError(
                "guardrails/governance supersede approval_policy: the RiskPolicy decides "
                "which calls need a human, and the approver (from GuardrailConfig or "
                "GovernancePolicy) is who gets asked. Leave approval_policy='never' (its "
                "default) when either is set."
            )
        if self.approval_policy == "always" and self.approval_fn is auto_approve:
            raise ValueError(
                "approval_policy='always' with the default auto_approve fn is a no-op. "
                "Supply approval_fn (e.g. governed.cli_approve)."
            )

    def resolve_approval(self) -> ApprovalFn:
        """Wrap ``approval_fn`` with the policy so tools see a single callable."""
        if self.approval_policy == "never":
            return auto_approve
        if self.approval_policy == "always":
            return self.approval_fn

        fn = self.approval_fn

        def gated(spec: ToolSpec, args: dict[str, Any]) -> bool:
            if spec.safety in DANGEROUS or spec.requires_approval:
                return fn(spec, args)
            return True

        return gated

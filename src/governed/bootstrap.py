"""Bootstrap an ``Agent`` from plain data -- a dict, JSON, or YAML file --
instead of Python code that imports and constructs each config object by
hand. This is what makes "config-first" literal: everything expressible here
is JSON-safe (strings, numbers, bools, lists, nested objects), so it can come
from a config file, an environment-driven templating step, or a secret
manager, not just a Python call site.

::

    from governed import Agent
    from governed.bootstrap import agent_config_from_yaml

    config = agent_config_from_yaml("agent.yaml")
    result = Agent(config).run("Profile data/sales.csv and report the top 3 regions.")

Scope, stated plainly, the same way ``ProviderPolicy`` and
``KeywordSafetyProvider`` are upfront about theirs: this resolves the parts
of ``AgentConfig`` that are genuinely data -- ``llm``, ``provider_policy``,
``governance``, ``features``, ``tools`` (via ``ToolConfig``), ``skills`` (via
``SkillConfig``), ``observability`` (including typed decision-ledger stores
and sinks, and typed event-trace subscribers -- ``http``/``otel`` for
external monitoring, ``console``/``logging`` for the built-in sinks),
``budget``, ``cost``, ``circuit_breaker``, ``compaction``, ``store``, and the
plain scalar fields. It does **not** attempt to resolve ``guardrails`` (a
``GuardrailConfig`` holds callables and scanner instances -- code, not data)
or anything else requiring a live Python object with no name this loader
knows -- a custom ``Tool``, a real ``Approver``, a hand-written ``Subscriber``.
Reach those through ``overrides``, applied after everything else:

::

    agent_config_from_mapping(data, overrides={"guardrails": my_guardrail_config})

For the fully data-driven guardrail path, use ``governance`` (an explicit
tool allowlist, sensitive operations, an approval threshold) and
``features.guardrails``/``features.content_safety`` instead -- both are
entirely expressible as data and cover the common case.

Every ``{"type": name, ...}`` this module resolves does so through a
registry you can extend: ``register_decision_ledger_sink``/
``register_decision_ledger_store``/``register_event_sink``/
``register_state_store`` here, plus ``governed.tools.register_tool`` and
``governed.skills.register_skill_source`` for the tool/skill side of
config-driven bootstrapping. All five follow the same shape
``governed.llm.factory.register_provider`` established: a factory function,
registered under a name, selectable from data alone from then on -- a
plugin author writes the factory once; a deployment picks it by name in a
config file with no Python glue.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import (
    AgentConfig,
    ApprovalFn,
    Budget,
    FeatureToggleConfig,
    ObservabilityConfig,
    auto_approve,
    cli_approve,
    deny_all,
)
from .llm.config import LLMConfig
from .llm.policy import ProviderPolicy
from .memory.optimizer import CircuitBreakerConfig, CostConfig
from .memory.store import InMemoryStore, JSONFileStore, StateStore
from .memory.transcript import CompactionConfig
from .observability.decision_ledger import (
    DecisionLedgerConfig,
    DecisionLedgerSink,
    DecisionLedgerStore,
    HttpDecisionLedgerSink,
    InMemoryDecisionLedger,
    JSONLDecisionLedger,
    OTelDecisionLedgerSink,
)
from .observability.events import EventType, Subscriber
from .observability.logger import ConsoleSink, HttpEventSink, LoggingSink, OTelEventSink
from .security.guardrails import RiskTier
from .security.policy import GovernancePolicy
from .skills.loader import SkillConfig
from .tools.base import ToolConfig

__all__ = [
    "agent_config_from_json",
    "agent_config_from_mapping",
    "agent_config_from_yaml",
    "register_decision_ledger_sink",
    "register_decision_ledger_store",
    "register_event_sink",
    "register_state_store",
    "registered_decision_ledger_sinks",
    "registered_decision_ledger_stores",
    "registered_event_sinks",
    "registered_state_stores",
]

#: Fields no amount of plain data can express -- see the module docstring.
_UNSUPPORTED_FROM_DATA = frozenset({"guardrails"})


def agent_config_from_mapping(
    data: dict[str, Any], *, overrides: dict[str, Any] | None = None
) -> AgentConfig:
    """Build an ``AgentConfig`` from a plain (JSON/YAML-shaped) mapping.

    ``data["llm"]`` is required (``{"provider": ..., "model": ..., ...}``).
    Every other top-level key is optional and maps to the matching
    ``AgentConfig`` field, nested dicts resolved into the corresponding
    dataclass -- see the module docstring for exactly which fields, and for
    the ``overrides`` escape hatch for anything not expressible as data.
    """
    unsupported = _UNSUPPORTED_FROM_DATA & set(data)
    if unsupported:
        raise ValueError(
            f"{sorted(unsupported)} cannot be built from plain data -- approvers, "
            "scanners, and risk policies are code, not JSON. Use `governance` + "
            "`features.guardrails`/`features.content_safety` for the data-driven "
            "path, or pass a live GuardrailConfig via "
            "overrides={'guardrails': ...}."
        )

    if "llm" not in data:
        raise ValueError(
            "agent_config_from_mapping requires an 'llm' key, e.g. "
            '{"llm": {"provider": "anthropic", "model": "claude-sonnet-5"}}.'
        )

    kwargs: dict[str, Any] = {"llm": _llm_config(data["llm"])}

    if data.get("provider_policy") is not None:
        kwargs["provider_policy"] = _provider_policy(data["provider_policy"])
    if data.get("governance") is not None:
        kwargs["governance"] = _governance(data["governance"])
    if data.get("features") is not None:
        kwargs["features"] = FeatureToggleConfig(**data["features"])
    if data.get("tools") is not None:
        kwargs["tools"] = _tool_config(data["tools"])
    if data.get("skills") is not None:
        kwargs["skills"] = SkillConfig(
            dirs=list(data["skills"].get("dirs", ["./skills"])),
            enabled=data["skills"].get("enabled", True),
            source=data["skills"].get("source", "directory"),
        )
    if "skills_dirs" in data:
        kwargs["skills_dirs"] = list(data["skills_dirs"])
    if data.get("observability") is not None:
        kwargs["observability"] = _observability_config(data["observability"])
    if data.get("budget") is not None:
        kwargs["budget"] = Budget(**data["budget"])
    if data.get("cost") is not None:
        # `pricing_overrides` needs live `ModelPricing` instances -- not
        # resolved from data here; use `overrides` for that.
        kwargs["cost"] = CostConfig(
            enabled=data["cost"].get("enabled", True), batch=data["cost"].get("batch", False)
        )
    if data.get("circuit_breaker") is not None:
        kwargs["circuit_breaker"] = CircuitBreakerConfig(**data["circuit_breaker"])
    if data.get("compaction") is not None:
        kwargs["compaction"] = CompactionConfig(**data["compaction"])
    if data.get("store") is not None:
        kwargs["store"] = _state_store(data["store"])
    if data.get("approval_fn") is not None:
        kwargs["approval_fn"] = _approval_fn(data["approval_fn"])

    for key in (
        "workspace",
        "tool_timeout_s",
        "approval_policy",
        "recursive_compaction",
        "checkpoint_every_iteration",
        "max_tokens_per_call",
        "temperature",
        "extra_instructions",
    ):
        if key in data:
            kwargs[key] = data[key]

    kwargs.update(overrides or {})
    return AgentConfig(**kwargs)


def agent_config_from_json(
    path: str | Path, *, overrides: dict[str, Any] | None = None
) -> AgentConfig:
    """``agent_config_from_mapping`` from a JSON file on disk."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return agent_config_from_mapping(data, overrides=overrides)


def agent_config_from_yaml(
    path: str | Path, *, overrides: dict[str, Any] | None = None
) -> AgentConfig:
    """``agent_config_from_mapping`` from a YAML file on disk.

    Requires PyYAML (``pip install 'governed[yaml]'``); imported lazily, so
    the core has no hard dependency on it -- same rule as everywhere else in
    this framework an optional format is involved.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "agent_config_from_yaml requires the `pyyaml` package: "
            "pip install 'governed[yaml]'"
        ) from exc
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return agent_config_from_mapping(data, overrides=overrides)


# ---------------------------------------------------------------------------
# Field-level resolvers
# ---------------------------------------------------------------------------


def _llm_config(d: dict[str, Any]) -> LLMConfig:
    return LLMConfig(
        provider=d["provider"],
        model=d["model"],
        api_key=d.get("api_key"),
        base_url=d.get("base_url"),
        extra=dict(d.get("extra", {})),
    )


def _provider_policy(d: dict[str, Any]) -> ProviderPolicy:
    allowed_providers = d.get("allowed_providers")
    return ProviderPolicy(
        allowed_providers=(
            frozenset(allowed_providers) if allowed_providers is not None else None
        ),
        allowed_models={k: frozenset(v) for k, v in d.get("allowed_models", {}).items()},
    )


def _governance(d: dict[str, Any]) -> GovernancePolicy:
    allowed_tools = d.get("allowed_tools")
    threshold = d.get("approval_threshold", "WARNING")
    return GovernancePolicy(
        allowed_tools=frozenset(allowed_tools) if allowed_tools is not None else None,
        sensitive_operations=frozenset(d.get("sensitive_operations", [])),
        approval_threshold=RiskTier[threshold] if isinstance(threshold, str) else threshold,
    )


def _tool_config(d: dict[str, Any]) -> ToolConfig:
    # `extra` (live Tool instances) isn't resolvable from data -- add them
    # via overrides={"tools": ToolConfig(names=[...], extra=[MyTool()])}.
    names = d.get("names")
    return ToolConfig(
        include_code_execution=d.get("include_code_execution", True),
        include_data_analysis=d.get("include_data_analysis", True),
        names=list(names) if names is not None else None,
    )


_SINK_BUILDERS: dict[str, Callable[[dict[str, Any]], DecisionLedgerSink]] = {
    "http": lambda d: HttpDecisionLedgerSink(d["url"], headers=d.get("headers")),
    "otel": lambda d: OTelDecisionLedgerSink(
        d["endpoint"],
        headers=d.get("headers"),
        service_name=d.get("service_name", "governed"),
    ),
}

_DECISION_STORE_BUILDERS: dict[str, Callable[[dict[str, Any]], DecisionLedgerStore]] = {
    "jsonl": lambda d: JSONLDecisionLedger(d["path"]),
    "memory": lambda d: InMemoryDecisionLedger(),
}


def _event_types(d: dict[str, Any]) -> set[EventType] | None:
    """``["tool.call", "run.end"]`` (an ``EventType`` value) or
    ``["TOOL_CALL", "RUN_END"]`` (the member name) both work -- accepting
    the raw ``EventType`` values means this reads the same as the JSONL
    trace it's filtering."""
    names = d.get("event_types")
    if names is None:
        return None
    return {n if isinstance(n, EventType) else EventType(n) for n in names}


#: Event-trace subscribers nameable from data -- "http"/"otel" reach the same
#: external monitoring systems `_SINK_BUILDERS` above reaches for the
#: decision ledger; "console"/"logging" name the two built-in sinks that
#: otherwise only turn on via `AgentConfig.console`/a hand-built `LoggingSink`.
_EVENT_SINK_BUILDERS: dict[str, Callable[[dict[str, Any]], Subscriber]] = {
    "http": lambda d: HttpEventSink(
        d["url"], headers=d.get("headers"), event_types=_event_types(d)
    ),
    "otel": lambda d: OTelEventSink(
        d["endpoint"],
        headers=d.get("headers"),
        service_name=d.get("service_name", "governed"),
        event_types=_event_types(d),
    ),
    "console": lambda d: ConsoleSink(verbose=d.get("verbose", False)),
    "logging": lambda d: LoggingSink(),
}

_STATE_STORE_BUILDERS: dict[str, Callable[[dict[str, Any]], StateStore]] = {
    "memory": lambda d: InMemoryStore(),
    "json_file": lambda d: JSONFileStore(d["directory"]),
}


def _resolve_typed(
    d: dict[str, Any], builders: dict[str, Callable[[dict[str, Any]], Any]], what: str
) -> Any:
    kind = d.get("type")
    if not isinstance(kind, str) or kind.lower() not in builders:
        raise ValueError(
            f"Unknown {what} type {kind!r}. Known: {sorted(builders)}. Register your own "
            "via the matching register_*() function, or pass a live instance via "
            "`overrides` for anything else."
        )
    return builders[kind.lower()](d)


# ---------------------------------------------------------------------------
# Plugin registries: config-driven registration of decision-ledger sinks and
# stores, event-trace sinks, and state stores -- the same "register a
# factory, select it by name from data" pattern as
# `governed.llm.factory.register_provider`, `governed.tools.register_tool`,
# and `governed.skills.register_skill_source`. Each registry seeds from the
# built-ins already used above; registering an existing name replaces it.
# ---------------------------------------------------------------------------


def register_decision_ledger_sink(
    name: str, builder: Callable[[dict[str, Any]], DecisionLedgerSink]
) -> None:
    """Make ``{"type": name, ...}`` resolvable inside
    ``observability.decision_ledger.sinks``. ``builder`` takes the sink's own
    dict (everything except ``"type"``) and returns a ready ``DecisionLedgerSink``.
    """
    _SINK_BUILDERS[name.lower()] = builder


def registered_decision_ledger_sinks() -> list[str]:
    return sorted(_SINK_BUILDERS)


def register_decision_ledger_store(
    name: str, builder: Callable[[dict[str, Any]], DecisionLedgerStore]
) -> None:
    """Make ``{"type": name, ...}`` resolvable as
    ``observability.decision_ledger.store``."""
    _DECISION_STORE_BUILDERS[name.lower()] = builder


def registered_decision_ledger_stores() -> list[str]:
    return sorted(_DECISION_STORE_BUILDERS)


def register_event_sink(name: str, builder: Callable[[dict[str, Any]], Subscriber]) -> None:
    """Make ``{"type": name, ...}`` resolvable inside
    ``observability.subscribers`` -- the event-trace counterpart of
    ``register_decision_ledger_sink``. ``builder`` takes the sink's own dict
    and returns any ``Subscriber`` (``Event -> None``); it doesn't have to be
    HTTP-shaped like ``HttpEventSink``/``OTelEventSink`` at all.
    """
    _EVENT_SINK_BUILDERS[name.lower()] = builder


def registered_event_sinks() -> list[str]:
    return sorted(_EVENT_SINK_BUILDERS)


def register_state_store(name: str, builder: Callable[[dict[str, Any]], StateStore]) -> None:
    """Make ``{"type": name, ...}`` resolvable as ``store`` -- e.g. a Redis-
    or Postgres-backed ``StateStore`` your deployment supplies."""
    _STATE_STORE_BUILDERS[name.lower()] = builder


def registered_state_stores() -> list[str]:
    return sorted(_STATE_STORE_BUILDERS)


def _decision_ledger_config(d: dict[str, Any]) -> DecisionLedgerConfig:
    store = (
        _resolve_typed(d["store"], _DECISION_STORE_BUILDERS, "decision ledger store")
        if d.get("store") is not None
        else None
    )
    sinks = [
        _resolve_typed(s, _SINK_BUILDERS, "decision ledger sink") for s in d.get("sinks", [])
    ]
    return DecisionLedgerConfig(enabled=d.get("enabled", False), store=store, sinks=sinks)


def _observability_config(d: dict[str, Any]) -> ObservabilityConfig:
    decision_ledger = (
        _decision_ledger_config(d["decision_ledger"])
        if d.get("decision_ledger") is not None
        else None
    )
    # A hand-written Subscriber still isn't resolvable from data -- add one
    # via overrides={"observability": ObservabilityConfig(subscribers=[...])}.
    subscribers = [
        _resolve_typed(s, _EVENT_SINK_BUILDERS, "event sink") for s in d.get("subscribers", [])
    ]
    return ObservabilityConfig(
        trace_path=d.get("trace_path"),
        console=d.get("console", True),
        verbose=d.get("verbose", False),
        subscribers=subscribers,
        decision_ledger=decision_ledger,
    )


def _state_store(d: dict[str, Any]) -> StateStore:
    store: StateStore = _resolve_typed(d, _STATE_STORE_BUILDERS, "store")
    return store


_APPROVAL_FNS: dict[str, ApprovalFn] = {
    "auto": auto_approve,
    "cli": cli_approve,
    "deny": deny_all,
}


def _approval_fn(name: str) -> ApprovalFn:
    try:
        return _APPROVAL_FNS[name]
    except KeyError:
        raise ValueError(
            f"Unknown approval_fn {name!r}. Known: {sorted(_APPROVAL_FNS)}. Pass a live "
            "callable via overrides={'approval_fn': ...} instead."
        ) from None

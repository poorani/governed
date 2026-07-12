"""Bootstrapping an AgentConfig from plain data -- the "config only, no
Python glue" path -- via governed.bootstrap.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from governed import Agent, GuardrailConfig
from governed.bootstrap import (
    agent_config_from_json,
    agent_config_from_mapping,
    agent_config_from_yaml,
)
from governed.llm.anthropic_client import AnthropicClient
from governed.llm.openai_client import OpenAIClient


def _fake_anthropic_sdk() -> object:
    from types import SimpleNamespace

    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **_: response))


def test_llm_is_required() -> None:
    with pytest.raises(ValueError, match="llm"):
        agent_config_from_mapping({})


def test_minimal_mapping_resolves_a_working_llm_client() -> None:
    cfg = agent_config_from_mapping(
        {"llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "sk-test"}}
    )
    assert isinstance(cfg.llm, OpenAIClient)
    assert cfg.llm.model == "gpt-4.1"


def test_provider_policy_is_resolved_and_enforced() -> None:
    cfg = agent_config_from_mapping(
        {
            "llm": {
                "provider": "anthropic",
                "model": "claude-sonnet-5",
                "extra": {"client": _fake_anthropic_sdk()},
            },
            "provider_policy": {"allowed_providers": ["anthropic"]},
        }
    )
    assert isinstance(cfg.llm, AnthropicClient)

    with pytest.raises(Exception, match="openai"):
        agent_config_from_mapping(
            {
                "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
                "provider_policy": {"allowed_providers": ["anthropic"]},
            }
        )


def test_governance_is_resolved_from_data() -> None:
    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "governance": {
                "allowed_tools": ["file_system", "submit"],
                "sensitive_operations": ["file_system:delete"],
                "approval_threshold": "SAFE",
            },
        }
    )
    assert cfg.governance is not None
    assert cfg.governance.allowed_tools == frozenset({"file_system", "submit"})
    assert cfg.governance.sensitive_operations == frozenset({"file_system:delete"})
    assert cfg.governance.approval_threshold.name == "SAFE"


def test_features_and_tools_and_skills_and_observability_are_resolved() -> None:
    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "features": {"decision_ledger": True, "telemetry": True},
            "tools": {"names": ["file_system", "submit"]},
            "skills": {"dirs": [], "enabled": False},
            "observability": {
                "console": False,
                "decision_ledger": {"enabled": True, "store": {"type": "memory"}},
            },
        }
    )
    assert cfg.features is not None and cfg.features.decision_ledger
    assert cfg.tools is not None
    assert cfg.skills is not None
    assert cfg.console is False
    assert cfg.decision_ledger is not None and cfg.decision_ledger.enabled


def test_budget_cost_circuit_breaker_compaction_are_resolved() -> None:
    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "budget": {"max_iterations": 3},
            "cost": {"enabled": False},
            "circuit_breaker": {"max_usd": 2.5},
            "compaction": {"keep_iterations": 1},
        }
    )
    assert cfg.budget.max_iterations == 3
    assert cfg.cost.enabled is False
    assert cfg.circuit_breaker.max_usd == 2.5
    assert cfg.compaction.keep_iterations == 1


def test_plain_scalar_fields_pass_through() -> None:
    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "workspace": "./somewhere",
            "temperature": 0.5,
            "max_tokens_per_call": 2048,
            "extra_instructions": "be terse",
        }
    )
    assert str(cfg.workspace).endswith("somewhere")
    assert cfg.temperature == 0.5
    assert cfg.max_tokens_per_call == 2048
    assert cfg.extra_instructions == "be terse"


def test_approval_fn_resolves_by_name() -> None:
    from governed.config import cli_approve

    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "approval_fn": "cli",
        }
    )
    assert cfg.approval_fn is cli_approve


def test_unknown_approval_fn_name_raises() -> None:
    with pytest.raises(ValueError, match="nonsense"):
        agent_config_from_mapping(
            {
                "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
                "approval_fn": "nonsense",
            }
        )


def test_decision_ledger_sink_types_resolve() -> None:
    cfg = agent_config_from_mapping(
        {
            "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
            "observability": {
                "decision_ledger": {
                    "enabled": True,
                    "store": {"type": "memory"},
                    "sinks": [
                        {"type": "http", "url": "https://logs.example/ingest"},
                        {"type": "otel", "endpoint": "https://collector.example:4318"},
                    ],
                }
            },
        }
    )
    assert cfg.decision_ledger is not None
    assert len(cfg.decision_ledger.sinks) == 2


def test_unknown_sink_type_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        agent_config_from_mapping(
            {
                "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
                "observability": {
                    "decision_ledger": {"sinks": [{"type": "bogus", "url": "x"}]}
                },
            }
        )


def test_guardrails_key_is_rejected_with_a_clear_message() -> None:
    with pytest.raises(ValueError, match="governance"):
        agent_config_from_mapping(
            {
                "llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"},
                "guardrails": {"enabled": True},
            }
        )


def test_overrides_win_and_reach_the_unsupported_fields() -> None:
    guardrails = GuardrailConfig(enabled=True)
    cfg = agent_config_from_mapping(
        {"llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"}},
        overrides={"guardrails": guardrails, "temperature": 0.9},
    )
    assert cfg.guardrails is guardrails
    assert cfg.temperature == 0.9


def test_resulting_config_actually_boots_an_agent(tmp_path: Path) -> None:
    cfg = agent_config_from_mapping(
        {
            "llm": {
                "provider": "anthropic",
                "model": "claude-sonnet-5",
                "extra": {"client": _fake_anthropic_sdk()},
            },
            "tools": {"names": ["file_system", "submit"]},
            "skills": {"dirs": [], "enabled": False},
            "features": {"decision_ledger": True},
        },
        overrides={"workspace": tmp_path / "workspace"},
    )
    agent = Agent(cfg)
    assert sorted(agent.registry.names) == ["file_system", "submit"]
    assert agent._decision_ledger_config is not None


def test_agent_config_from_json_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    path.write_text(
        json.dumps({"llm": {"provider": "openai", "model": "gpt-4.1", "api_key": "x"}})
    )
    cfg = agent_config_from_json(path)
    assert isinstance(cfg.llm, OpenAIClient)


def test_agent_config_from_yaml_reads_a_file(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "agent.yaml"
    path.write_text(
        "llm:\n  provider: openai\n  model: gpt-4.1\n  api_key: x\n"
        "budget:\n  max_iterations: 4\n"
    )
    cfg = agent_config_from_yaml(path)
    assert isinstance(cfg.llm, OpenAIClient)
    assert cfg.budget.max_iterations == 4

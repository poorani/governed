"""Config-first composition: ToolConfig/SkillConfig (data-only alternatives
to live tool/skill objects), ObservabilityConfig (grouped observability
settings), and FeatureToggleConfig (coarse, gap-filling on/off switches) --
in isolation and wired through a real Agent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from governed import (
    Agent,
    AgentConfig,
    AllowTierApprover,
    Budget,
    FeatureToggleConfig,
    GuardrailConfig,
    InMemoryStore,
    LLMResponse,
    ObservabilityConfig,
    PIIScanner,
    RiskTier,
    SkillConfig,
    ToolConfig,
    default_tools,
    resolve_skills,
    resolve_tools,
)
from governed.llm import ScriptedClient, ToolCall, Usage
from governed.tools import (
    CodeExecutionTool,
    FileSystemTool,
    LoadSkillTool,
    ScratchpadTool,
    SubmitTool,
)

# ---------------------------------------------------------------------------
# ToolConfig / resolve_tools
# ---------------------------------------------------------------------------


def test_default_tool_config_matches_default_tools() -> None:
    names = {t.name for t in resolve_tools(ToolConfig(), skills=None)}
    assert names == {t.name for t in default_tools(None)}


def test_tool_config_names_selects_an_explicit_subset() -> None:
    tools = resolve_tools(ToolConfig(names=["file_system", "submit"]), skills=None)
    assert {t.name for t in tools} == {"file_system", "submit"}


def test_tool_config_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="bogus"):
        resolve_tools(ToolConfig(names=["bogus"]), skills=None)


def test_tool_config_load_skill_without_skills_raises() -> None:
    with pytest.raises(ValueError, match="load_skill"):
        resolve_tools(ToolConfig(names=["load_skill", "submit"]), skills=None)


def test_tool_config_load_skill_with_skills_resolves() -> None:
    skills = resolve_skills(SkillConfig(dirs=[]))
    tools = resolve_tools(ToolConfig(names=["submit", "load_skill"]), skills=skills)
    assert any(isinstance(t, LoadSkillTool) for t in tools)


def test_tool_config_extra_appends_custom_instances() -> None:
    class MyTool(SubmitTool):
        name = "my_submit"

    tools = resolve_tools(ToolConfig(names=["submit"], extra=[MyTool()]), skills=None)
    assert {t.name for t in tools} == {"submit", "my_submit"}


def test_tool_config_can_drop_code_execution() -> None:
    tools = resolve_tools(ToolConfig(include_code_execution=False), skills=None)
    assert not any(isinstance(t, CodeExecutionTool) for t in tools)


# ---------------------------------------------------------------------------
# SkillConfig / resolve_skills
# ---------------------------------------------------------------------------


def test_skill_config_resolves_via_dirs(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: a demo skill\n---\nbody text\n"
    )
    library = resolve_skills(SkillConfig(dirs=[str(tmp_path / "skills")]))
    assert library.names == {"demo"}


def test_skill_config_disabled_returns_empty_library(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: demo\ndescription: x\n---\nbody\n")
    library = resolve_skills(SkillConfig(dirs=[str(tmp_path / "skills")], enabled=False))
    assert library.names == set()


def test_resolve_skills_none_returns_empty_library() -> None:
    assert resolve_skills(None).names == set()


# ---------------------------------------------------------------------------
# ObservabilityConfig
# ---------------------------------------------------------------------------


def test_observability_config_overwrites_the_individual_fields() -> None:
    cfg = AgentConfig(
        llm=ScriptedClient([LLMResponse(text="unused")]),
        observability=ObservabilityConfig(trace_path="./traces/x.jsonl", console=False),
    )
    assert cfg.trace_path == "./traces/x.jsonl"
    assert cfg.console is False


def test_no_observability_config_leaves_defaults_untouched() -> None:
    cfg = AgentConfig(llm=ScriptedClient([LLMResponse(text="unused")]))
    assert cfg.trace_path is None
    assert cfg.console is True


# ---------------------------------------------------------------------------
# FeatureToggleConfig, wired through a real Agent
# ---------------------------------------------------------------------------


def _agent(tmp_path: Path, features: FeatureToggleConfig | None, **kw) -> Agent:
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    return Agent(
        AgentConfig(
            llm=ScriptedClient([LLMResponse(text="unused")]),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            features=features,
            **kw,
        )
    )


def test_no_features_is_a_no_op(tmp_path: Path) -> None:
    agent = _agent(tmp_path, None)
    assert agent.gateway is None
    assert agent.decision_ledger is None
    assert agent.telemetry is None


def test_feature_toggle_guardrails_builds_a_default_gateway(tmp_path: Path) -> None:
    agent = _agent(tmp_path, FeatureToggleConfig(guardrails=True))
    assert agent.gateway is not None


def test_feature_toggle_never_overrides_explicit_guardrails(tmp_path: Path) -> None:
    approver = AllowTierApprover(RiskTier.DANGER)
    explicit = GuardrailConfig(approver=approver)
    agent = _agent(tmp_path, FeatureToggleConfig(guardrails=True), guardrails=explicit)
    assert agent.gateway is not None
    assert agent.gateway.approver is approver  # untouched, not replaced


def test_feature_toggle_content_safety_adds_scanner_to_existing_guardrails(
    tmp_path: Path,
) -> None:
    explicit = GuardrailConfig(scan_results=False)  # a real, specific setting
    agent = _agent(tmp_path, FeatureToggleConfig(content_safety=True), guardrails=explicit)
    assert agent.gateway is not None
    # the content-safety scanner was added to the arguments path...
    assert len(agent.gateway.scanners) >= 1
    # ...without reverting the explicit scan_results=False: content-safety
    # scanners are unconditionally screened both ways, but the *built-in*
    # InjectionScanner/SecretExfiltrationScanner respect the flag, so exactly
    # one result scanner (content-safety) should be present, not three.
    assert len(agent.gateway.result_scanners) == 1


def test_feature_toggle_content_safety_alone_builds_guardrails_too(tmp_path: Path) -> None:
    agent = _agent(tmp_path, FeatureToggleConfig(content_safety=True))
    assert agent.gateway is not None


def test_feature_toggle_pii_detection_alone_builds_guardrails_too(tmp_path: Path) -> None:
    agent = _agent(tmp_path, FeatureToggleConfig(pii_detection=True))
    assert agent.gateway is not None
    assert any(isinstance(s, PIIScanner) for s in agent.gateway.scanners)


def test_feature_toggle_pii_detection_adds_scanner_to_existing_guardrails(
    tmp_path: Path,
) -> None:
    explicit = GuardrailConfig(scan_results=False)
    agent = _agent(tmp_path, FeatureToggleConfig(pii_detection=True), guardrails=explicit)
    assert agent.gateway is not None
    assert any(isinstance(s, PIIScanner) for s in agent.gateway.scanners)


def test_feature_toggle_pii_detection_does_not_duplicate_an_explicit_scanner(
    tmp_path: Path,
) -> None:
    explicit = GuardrailConfig(extra_scanners=[PIIScanner()])
    agent = _agent(tmp_path, FeatureToggleConfig(pii_detection=True), guardrails=explicit)
    assert agent.gateway is not None
    assert sum(isinstance(s, PIIScanner) for s in agent.gateway.scanners) == 1


def test_feature_toggle_decision_ledger_uses_a_default_store(tmp_path: Path) -> None:
    agent = _agent(tmp_path, FeatureToggleConfig(decision_ledger=True))
    assert agent._decision_ledger_config is not None
    assert agent._decision_ledger_config.enabled is True
    assert agent._decision_ledger_store is not None


def test_feature_toggle_decision_ledger_does_not_override_explicit_config(
    tmp_path: Path,
) -> None:
    from governed import DecisionLedgerConfig, InMemoryDecisionLedger

    store = InMemoryDecisionLedger()
    explicit = DecisionLedgerConfig(enabled=True, store=store)
    agent = _agent(
        tmp_path, FeatureToggleConfig(decision_ledger=True), decision_ledger=explicit
    )
    assert agent._decision_ledger_store is store


def test_feature_toggle_telemetry_attaches_a_collector(tmp_path: Path) -> None:
    agent = _agent(tmp_path, FeatureToggleConfig(telemetry=True))
    assert agent.telemetry is not None


def _plan(step: str, tool: str, why: str, done: list[str]) -> LLMResponse:
    return LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "count words",
                "steps": [{"id": "s1", "description": "submit", "done": "s1" in done}],
                "next_action": {
                    "step_id": step,
                    "tool": tool,
                    "rationale": why,
                    "success_criteria": "the run ends",
                },
            }
        )
        + "</plan>",
        usage=Usage(200, 20),
    )


def test_feature_toggle_telemetry_actually_receives_events(tmp_path: Path) -> None:
    script = [
        _plan("s1", "submit", "done", []),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    "c1",
                    "submit",
                    {
                        "answer": "done",
                        "status": "complete",
                        "confidence": 1.0,
                        "evidence": ["nothing to do"],
                        "unmet_requirements": [],
                    },
                )
            ],
            usage=Usage(150, 15),
        ),
    ]
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(script),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
            tools=[FileSystemTool(), ScratchpadTool(), SubmitTool()],
            features=FeatureToggleConfig(telemetry=True),
        )
    )
    result = agent.run("do nothing")
    assert result.ok
    assert agent.telemetry is not None
    assert agent.telemetry.llm.count > 0  # actually observed LLM_RESPONSE events
    assert "submit" in agent.telemetry.tools
    assert agent.telemetry.session.started_ts is not None

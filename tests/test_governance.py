"""Governance: allowed-tool policy, sensitive-operation escalation, approval
thresholds, and the audit report -- exercised both in isolation (against
``GovernancePolicy`` directly) and end-to-end (through a real ``Agent`` run
driven by ``ScriptedClient``, so no network or API key is needed).
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
    GovernancePolicy,
    GovernanceViolation,
    GuardrailConfig,
    InMemoryStore,
    LLMResponse,
    RiskPolicy,
    RiskTier,
    build_audit_report,
)
from governed.llm import ScriptedClient, ToolCall, Usage
from governed.tools import CodeExecutionTool, FileSystemTool, ScratchpadTool, SubmitTool

# ---------------------------------------------------------------------------
# GovernancePolicy in isolation
# ---------------------------------------------------------------------------


def test_allowed_tools_none_means_unrestricted() -> None:
    GovernancePolicy().enforce_allowed_tools([FileSystemTool(), SubmitTool()])  # no raise


def test_allowed_tools_permits_an_explicit_allowlist() -> None:
    policy = GovernancePolicy(allowed_tools=frozenset({"file_system"}))
    policy.enforce_allowed_tools([FileSystemTool(), SubmitTool()])  # submit is implicit


def test_allowed_tools_rejects_anything_outside_the_allowlist() -> None:
    policy = GovernancePolicy(allowed_tools=frozenset({"file_system"}))
    with pytest.raises(GovernanceViolation, match="execute_code"):
        policy.enforce_allowed_tools([FileSystemTool(), CodeExecutionTool(), SubmitTool()])


def test_sensitive_operations_can_only_raise_the_tier() -> None:
    policy = GovernancePolicy(sensitive_operations=frozenset({"file_system:write"}))
    cfg = policy.apply(None)

    spec = FileSystemTool().spec()
    # Ordinarily WARNING; governance forces it to DANGER.
    assert cfg.risk_policy.assess(spec, {"operation": "write", "path": "a"}) is RiskTier.DANGER
    # Reads are untouched -- the escalation is scoped to the named operation.
    assert cfg.risk_policy.assess(spec, {"operation": "read", "path": "a"}) is RiskTier.SAFE


def test_sensitive_operations_by_whole_tool_name() -> None:
    policy = GovernancePolicy(sensitive_operations=frozenset({"scratchpad"}))
    cfg = policy.apply(None)
    spec = ScratchpadTool().spec()
    assert cfg.risk_policy.assess(spec, {"action": "read"}) is RiskTier.DANGER


def test_apply_builds_an_approver_from_the_threshold_when_none_is_set() -> None:
    policy = GovernancePolicy(approval_threshold=RiskTier.SAFE)
    cfg = policy.apply(None)
    assert isinstance(cfg.approver, AllowTierApprover)
    assert cfg.approver.ceiling is RiskTier.SAFE


def test_apply_layers_on_top_of_an_explicit_guardrail_config_without_discarding_it() -> None:
    base_policy = RiskPolicy(tool_tiers={"custom_tool": RiskTier.WARNING})
    base = GuardrailConfig(risk_policy=base_policy, scan_results=False)
    policy = GovernancePolicy(sensitive_operations=frozenset({"file_system:delete"}))

    merged = policy.apply(base)

    assert merged.scan_results is False  # untouched
    assert merged.risk_policy.tool_tiers["custom_tool"] is RiskTier.WARNING  # preserved
    spec = FileSystemTool().spec()
    assert merged.risk_policy.assess(spec, {"operation": "delete"}) is RiskTier.DANGER


def test_apply_does_not_let_a_caller_approver_be_silently_replaced() -> None:
    approver = AllowTierApprover(RiskTier.DANGER)
    base = GuardrailConfig(approver=approver)
    merged = GovernancePolicy(approval_threshold=RiskTier.SAFE).apply(base)
    assert merged.approver is approver


# ---------------------------------------------------------------------------
# End to end, through Agent construction
# ---------------------------------------------------------------------------


def test_agent_construction_rejects_a_disallowed_tool(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    with pytest.raises(GovernanceViolation, match="execute_code"):
        Agent(
            AgentConfig(
                llm=ScriptedClient([LLMResponse(text="unused")]),
                workspace=ws,
                skills_dirs=[],
                store=InMemoryStore(),
                tools=[FileSystemTool(), CodeExecutionTool(), SubmitTool()],
                governance=GovernancePolicy(allowed_tools=frozenset({"file_system"})),
            )
        )


def test_agent_construction_rejects_a_governance_of_the_wrong_type(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    with pytest.raises(TypeError, match="GovernancePolicy"):
        Agent(
            AgentConfig(
                llm=ScriptedClient([LLMResponse(text="unused")]),
                workspace=ws,
                skills_dirs=[],
                store=InMemoryStore(),
                governance={"allowed_tools": ["file_system"]},  # not a GovernancePolicy
            )
        )


def test_governance_and_approval_policy_conflict_is_rejected() -> None:
    with pytest.raises(ValueError, match="supersede"):
        AgentConfig(
            llm=ScriptedClient([LLMResponse(text="unused")]),
            governance=GovernancePolicy(),
            approval_policy="always",
        )


# ---------------------------------------------------------------------------
# End to end, through a real run
# ---------------------------------------------------------------------------


def _plan(step: str, tool: str, why: str, done: list[str]) -> LLMResponse:
    return LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "write to a file, then report the outcome",
                "steps": [
                    {"id": "s1", "description": "write", "done": "s1" in done},
                    {"id": "s2", "description": "report", "done": "s2" in done},
                ],
                "next_action": {
                    "step_id": step,
                    "tool": tool,
                    "rationale": why,
                    "success_criteria": "the tool call returns",
                },
            }
        )
        + "</plan>",
        usage=Usage(300, 40),
    )


def _eval(outcome: str, evidence: str, status: str, nxt: str, done: list[str]) -> LLMResponse:
    return LLMResponse(
        text="<evaluation>"
        + json.dumps(
            {
                "outcome": outcome,
                "evidence": evidence,
                "completed_step_ids": done,
                "goal_status": status,
                "next_step": nxt,
            }
        )
        + "</evaluation>",
        usage=Usage(250, 30),
    )


def _submit_call(answer: str, status: str) -> LLMResponse:
    return LLMResponse(
        tool_calls=[
            ToolCall(
                "csubmit",
                "submit",
                {
                    "answer": answer,
                    "status": status,
                    "confidence": 0.5,
                    "evidence": [answer],
                    "unmet_requirements": [],
                },
            )
        ],
        usage=Usage(200, 20),
    )


def _write_script() -> list[LLMResponse]:
    return [
        _plan("s1", "file_system", "write the file", []),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    "c1",
                    "file_system",
                    {"operation": "write", "path": "a.txt", "content": "hi"},
                )
            ],
            usage=Usage(300, 30),
        ),
        _eval("failure", "the write was blocked", "in_progress", "report it", []),
        _plan("s2", "submit", "report the outcome", []),
        _submit_call("could not write: blocked by policy", "blocked"),
    ]


def test_sensitive_operation_without_a_human_present_is_denied(tmp_path: Path) -> None:
    """file_system(write) is WARNING-tier by default -- runs unattended. A
    deployment that names it a sensitive operation, and supplies no approver
    to say yes, gets it denied instead: the whole point of the feature."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(_write_script()),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
            tools=[FileSystemTool(), SubmitTool()],
            governance=GovernancePolicy(
                sensitive_operations=frozenset({"file_system:write"}),
                approval_threshold=RiskTier.WARNING,
            ),
        )
    )
    result = agent.run("write hi to a.txt")

    assert not (ws / "a.txt").exists()
    call = result.state.iterations[0].tool_calls[0]
    assert not call.ok
    assert call.error_code == "policy_violation"


def test_sensitive_operation_runs_once_a_human_approves(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(
                [
                    _plan("s1", "file_system", "write the file", []),
                    LLMResponse(
                        tool_calls=[
                            ToolCall(
                                "c1",
                                "file_system",
                                {"operation": "write", "path": "a.txt", "content": "hi"},
                            )
                        ],
                        usage=Usage(300, 30),
                    ),
                    _eval("success", "wrote the file", "complete", "submit", ["s1"]),
                    _plan("s2", "submit", "report the outcome", ["s1"]),
                    _submit_call("wrote a.txt", "complete"),
                ]
            ),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
            tools=[FileSystemTool(), SubmitTool()],
            governance=GovernancePolicy(
                sensitive_operations=frozenset({"file_system:write"}),
                approver=AllowTierApprover(RiskTier.DANGER),
            ),
        )
    )
    result = agent.run("write hi to a.txt")

    assert result.ok
    assert (ws / "a.txt").read_text() == "hi"


# ---------------------------------------------------------------------------
# Audit report
# ---------------------------------------------------------------------------


def test_audit_report_captures_plan_action_and_approval_history(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(_write_script()),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
            tools=[FileSystemTool(), SubmitTool()],
            governance=GovernancePolicy(
                sensitive_operations=frozenset({"file_system:write"}),
                approval_threshold=RiskTier.WARNING,
            ),
        )
    )
    result = agent.run("write hi to a.txt")
    report = build_audit_report(agent, result)

    assert report.status == "blocked"
    assert report.denied_count == 1
    assert report.iterations[0].tool == "file_system"
    assert report.iterations[0].tool_calls[0]["ok"] is False
    assert report.governance == {
        "allowed_tools": None,
        "sensitive_operations": ["file_system:write"],
        "approval_threshold": "WARNING",
    }
    # Both renderers should run without raising, and actually say something.
    assert "file_system" in report.to_markdown()
    assert json.loads(report.to_json())["status"] == "blocked"

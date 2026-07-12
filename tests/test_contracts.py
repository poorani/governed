from __future__ import annotations

import json

import pytest

from governed.contracts import (
    ContractViolation,
    Phase,
    parse_evaluation,
    parse_plan,
    validate_tool_choice,
)


def _plan_text(
    tool: str = "file_system", criteria: str = "the file now exists on disk"
) -> str:
    return (
        "<plan>"
        + json.dumps(
            {
                "goal_restatement": "write a file",
                "steps": [{"id": "s1", "description": "write it", "done": False}],
                "next_action": {
                    "step_id": "s1",
                    "tool": tool,
                    "rationale": "need to create the file first",
                    "success_criteria": criteria,
                },
            }
        )
        + "</plan>"
    )


def test_parse_plan_happy_path() -> None:
    plan = parse_plan(_plan_text())
    assert plan.next_action.tool == "file_system"
    assert plan.steps[0].id == "s1"


def test_parse_plan_missing_tags_is_a_violation() -> None:
    with pytest.raises(ContractViolation) as exc:
        parse_plan("just some prose, no tags")
    assert exc.value.phase is Phase.ANALYZE


def test_parse_plan_invalid_json_is_a_violation() -> None:
    with pytest.raises(ContractViolation):
        parse_plan("<plan>{not json</plan>")


def test_parse_plan_next_action_must_reference_a_declared_step() -> None:
    text = (
        "<plan>"
        + json.dumps(
            {
                "goal_restatement": "x",
                "steps": [{"id": "s1", "description": "d", "done": False}],
                "next_action": {
                    "step_id": "s99",
                    "tool": "file_system",
                    "rationale": "r",
                    "success_criteria": "a checkable condition",
                },
            }
        )
        + "</plan>"
    )
    with pytest.raises(ContractViolation):
        parse_plan(text)


def _eval_text(
    evidence: str = "exit code 0 and 'tests: 14 passed'", ids: list[str] | None = None
) -> str:
    return (
        "<evaluation>"
        + json.dumps(
            {
                "outcome": "success",
                "evidence": evidence,
                "completed_step_ids": ids or ["s1"],
                "goal_status": "in_progress",
                "next_step": "read it back",
            }
        )
        + "</evaluation>"
    )


def test_parse_evaluation_happy_path() -> None:
    ev = parse_evaluation(_eval_text(), {"s1"})
    assert ev.outcome == "success"


def test_parse_evaluation_short_evidence_is_a_violation() -> None:
    with pytest.raises(ContractViolation):
        parse_evaluation(_eval_text(evidence="it worked"), {"s1"})


def test_parse_evaluation_unknown_step_id_is_a_violation() -> None:
    with pytest.raises(ContractViolation):
        parse_evaluation(_eval_text(ids=["s1", "s99"]), {"s1"})


# -- ACT contract -------------------------------------------------------


def test_validate_tool_choice_accepts_matching_call() -> None:
    plan = parse_plan(_plan_text(tool="file_system"))
    validate_tool_choice(plan, ["file_system"], {"file_system", "execute_code"})


def test_validate_tool_choice_rejects_mismatched_tool() -> None:
    plan = parse_plan(_plan_text(tool="file_system"))
    with pytest.raises(ContractViolation) as exc:
        validate_tool_choice(plan, ["execute_code"], {"file_system", "execute_code"})
    assert exc.value.phase is Phase.ACT
    assert "file_system" in exc.value.reason


def test_validate_tool_choice_rejects_no_call() -> None:
    plan = parse_plan(_plan_text())
    with pytest.raises(ContractViolation):
        validate_tool_choice(plan, [], {"file_system"})


def test_validate_tool_choice_rejects_multiple_calls() -> None:
    plan = parse_plan(_plan_text())
    with pytest.raises(ContractViolation):
        validate_tool_choice(
            plan, ["file_system", "execute_code"], {"file_system", "execute_code"}
        )

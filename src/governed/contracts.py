"""The Plan / Evaluation schemas that make each phase of the loop falsifiable.

``parse_plan`` and ``parse_evaluation`` turn raw model text into validated
objects or raise ``ContractViolation`` -- caught by ``Agent._with_retries``,
which feeds the violation back to the model as a corrective message rather than
crashing the run. ``validate_tool_choice`` is the ACT-phase enforcement that a
plan cannot be acted on by any tool other than the one it named.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

__all__ = [
    "ContractViolation",
    "Evaluation",
    "NextAction",
    "Phase",
    "Plan",
    "PlanStep",
    "parse_evaluation",
    "parse_plan",
    "validate_tool_choice",
]


class Phase(str, Enum):
    ANALYZE = "analyze"
    ACT = "act"
    EXECUTE = "execute"
    OBSERVE = "observe"


class ContractViolation(Exception):
    """A phase's output did not satisfy its contract.

    ``feedback`` is what gets pushed back to the model verbatim -- write it as
    an instruction, not a diagnosis: tell the model what to do differently, not
    just what it did wrong.
    """

    def __init__(self, phase: Phase, reason: str, feedback: str = "") -> None:
        super().__init__(f"[{phase.value}] {reason}")
        self.phase = phase
        self.reason = reason
        self.feedback = feedback or reason


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PlanStep(BaseModel):
    id: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    done: bool = False


class NextAction(BaseModel):
    step_id: str = Field(..., min_length=1)
    tool: str = Field(..., min_length=1)
    rationale: str = Field(
        ..., min_length=1, description="Why this tool, with these args, now."
    )
    success_criteria: str = Field(
        ..., min_length=1, description="The observable condition that proves it worked."
    )


class Plan(BaseModel):
    goal_restatement: str = Field(..., min_length=1)
    steps: list[PlanStep] = Field(..., min_length=1)
    next_action: NextAction

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class Evaluation(BaseModel):
    outcome: str = Field(..., pattern="^(success|partial|failure)$")
    evidence: str = Field(..., min_length=1)
    completed_step_ids: list[str] = Field(default_factory=list)
    goal_status: str = Field(..., pattern="^(complete|in_progress|blocked)$")
    next_step: str = Field(..., min_length=1)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_MIN_EVIDENCE_CHARS = 10


def _extract_tag(text: str, tag: str) -> str | None:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def parse_plan(text: str) -> Plan:
    raw = _extract_tag(text, "plan")
    if raw is None:
        raise ContractViolation(
            Phase.ANALYZE,
            "no <plan> block found",
            feedback="Wrap your plan in <plan>...</plan> tags containing exactly one JSON "
            "object with goal_restatement, steps, and next_action.",
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractViolation(
            Phase.ANALYZE,
            f"invalid JSON in <plan>: {exc}",
            feedback=f"Your <plan> block was not valid JSON ({exc}). Emit exactly one "
            "well-formed JSON object matching the schema.",
        ) from exc

    try:
        plan = Plan.model_validate(data)
    except ValidationError as exc:
        raise ContractViolation(
            Phase.ANALYZE,
            f"plan failed schema validation: {exc}",
            feedback=f"Your plan JSON did not match the required schema:\n{exc}\n"
            "Re-emit a complete <plan> block with all required fields.",
        ) from exc

    step_ids = {s.id for s in plan.steps}
    if plan.next_action.step_id not in step_ids:
        raise ContractViolation(
            Phase.ANALYZE,
            "next_action.step_id does not reference a declared step",
            feedback=f"next_action.step_id must be one of your declared step ids: "
            f"{sorted(step_ids)}.",
        )
    return plan


def parse_evaluation(text: str, valid_step_ids: set[str]) -> Evaluation:
    raw = _extract_tag(text, "evaluation")
    if raw is None:
        raise ContractViolation(
            Phase.OBSERVE,
            "no <evaluation> block found",
            feedback="Wrap your evaluation in <evaluation>...</evaluation> tags containing "
            "exactly one JSON object with outcome, evidence, completed_step_ids, "
            "goal_status, and next_step.",
        )
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractViolation(
            Phase.OBSERVE,
            f"invalid JSON in <evaluation>: {exc}",
            feedback=f"Your <evaluation> block was not valid JSON ({exc}). Emit exactly one "
            "well-formed JSON object matching the schema.",
        ) from exc

    try:
        evaluation = Evaluation.model_validate(data)
    except ValidationError as exc:
        raise ContractViolation(
            Phase.OBSERVE,
            f"evaluation failed schema validation: {exc}",
            feedback=f"Your evaluation JSON did not match the required schema:\n{exc}\n"
            "Re-emit a complete <evaluation> block with all required fields.",
        ) from exc

    if len(evaluation.evidence.strip()) < _MIN_EVIDENCE_CHARS:
        raise ContractViolation(
            Phase.OBSERVE,
            "evidence too short to be evidence",
            feedback="`evidence` must quote or cite the actual tool output that justifies "
            "your outcome. 'it worked' is not evidence; 'exit code 0 and stdout contains "
            '"14 passed"\' is.',
        )
    bad = set(evaluation.completed_step_ids) - valid_step_ids
    if bad:
        raise ContractViolation(
            Phase.OBSERVE,
            f"completed_step_ids references unknown steps: {sorted(bad)}",
            feedback=f"completed_step_ids must be a subset of this plan's step ids: "
            f"{sorted(valid_step_ids)}.",
        )
    return evaluation


def validate_tool_choice(
    plan: Plan, called_names: list[str], available_names: set[str]
) -> None:
    """The ACT-phase enforcement: the model may only call the tool its plan named."""
    if not called_names:
        raise ContractViolation(
            Phase.ACT,
            "no tool called",
            feedback=f"You must call exactly the tool your plan committed to: "
            f"`{plan.next_action.tool}`.",
        )
    if len(called_names) > 1:
        raise ContractViolation(
            Phase.ACT,
            f"called multiple tools in one iteration: {called_names}",
            feedback="Call exactly one tool per iteration -- the one your plan committed to. "
            "If you need several, do them one at a time across iterations.",
        )
    called = called_names[0]
    if called != plan.next_action.tool:
        raise ContractViolation(
            Phase.ACT,
            f"plan committed to '{plan.next_action.tool}' but called {called_names}",
            feedback=f"Your plan committed to `{plan.next_action.tool}`. Call that tool "
            "with matching arguments, or go back and revise your plan first.",
        )
    if called not in available_names:
        raise ContractViolation(
            Phase.ACT,
            f"'{called}' is not a registered tool",
            feedback=f"`{called}` is not registered. Available tools: "
            f"{sorted(available_names)}.",
        )

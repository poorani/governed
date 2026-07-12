"""One call, one structured, compliance-facing summary of a run.

Everything here is derived from data the framework already collects --
``SessionState.iterations`` (the plan/action/evidence/evaluation history
described in ``agent.py``'s module docstring), ``Gateway.decisions`` (the
guardrail audit trail), ``CostLedger`` (spend), and whatever ``GovernancePolicy``
was in effect -- reassembled into the shape a reviewer who was not in the room
actually wants: what was configured, what happened, what a human had to
approve, and what it cost. Nothing here is computed twice; this module is a
*view* over existing state, not a second bookkeeping system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..agent import Agent, RunResult

__all__ = ["AuditReport", "IterationSummary", "build_audit_report"]


@dataclass
class IterationSummary:
    index: int
    tool: str | None
    step_id: str | None
    rationale: str | None
    tool_calls: list[dict[str, Any]]
    evaluation_outcome: str | None
    evaluation_evidence: str | None
    violations: list[dict[str, Any]]


@dataclass
class AuditReport:
    """A structured record of what a run did and why.

    Built once, after a run finishes, by ``build_audit_report``. Everything
    on it is JSON-serialisable (``to_dict``) or renderable as a short
    narrative (``to_markdown``) for a human reviewer.
    """

    run_id: str
    session_id: str
    goal: str
    status: str
    model: str
    iterations: list[IterationSummary]
    guardrail_decisions: list[dict[str, Any]]
    approved_count: int
    denied_count: int
    cost_usd: float
    cost_by_phase: dict[str, float]
    final_answer: str
    confidence: float
    evidence: list[str]
    unmet_requirements: list[str]
    governance: dict[str, Any] | None = field(default=None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "goal": self.goal,
            "status": self.status,
            "model": self.model,
            "iterations": [vars(it) for it in self.iterations],
            "guardrail_decisions": self.guardrail_decisions,
            "approved_count": self.approved_count,
            "denied_count": self.denied_count,
            "cost_usd": self.cost_usd,
            "cost_by_phase": self.cost_by_phase,
            "final_answer": self.final_answer,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "unmet_requirements": self.unmet_requirements,
            "governance": self.governance,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_markdown(self) -> str:
        out: list[str] = [
            f"# Audit report -- run `{self.run_id}`",
            "",
            f"**Goal:** {self.goal}",
            f"**Status:** {self.status}  ·  **Model:** {self.model}",
            f"**Cost:** ${self.cost_usd:.4f}  ·  "
            f"**Human approvals:** {self.approved_count} granted, {self.denied_count} denied",
            "",
        ]
        if self.governance is not None:
            allowed = self.governance.get("allowed_tools")
            out += [
                "## Governance policy in effect",
                "",
                f"- Allowed tools: {allowed if allowed is not None else 'unrestricted'}",
                f"- Sensitive operations: "
                f"{self.governance.get('sensitive_operations') or 'none declared'}",
                f"- Approval threshold: {self.governance.get('approval_threshold')}",
                "",
            ]
        out += ["## Iterations", ""]
        for it in self.iterations:
            out.append(
                f"**{it.index}.** `{it.tool}` for step `{it.step_id}` -- {it.rationale}"
            )
            for tc in it.tool_calls:
                status = "ok" if tc.get("ok") else f"FAILED ({tc.get('error_code')})"
                out.append(f"  - `{tc.get('tool')}` {status}")
            if it.evaluation_outcome:
                out.append(f"  -> {it.evaluation_outcome}: {it.evaluation_evidence}")
            out.append("")
        if self.guardrail_decisions:
            out += ["## Guardrail decisions", ""]
            for d in self.guardrail_decisions:
                verdict = "BLOCKED" if d.get("blocked") else "allowed"
                reason = d.get("reason") or ""
                out.append(f"- `{d.get('tool')}` [{d.get('tier')}] {verdict} -- {reason}")
            out.append("")
        out += [
            "## Outcome",
            "",
            self.final_answer,
            "",
            f"*confidence={self.confidence}, evidence={self.evidence}, "
            f"unmet_requirements={self.unmet_requirements}*",
        ]
        return "\n".join(out)


def build_audit_report(agent: Agent, result: RunResult) -> AuditReport:
    """Assemble the accountability record for one finished run.

    ``agent`` and ``result`` are exactly what ``Agent.run``/``resume`` already
    gave you -- this reads their existing state, it doesn't re-run anything or
    ask the model for a self-report.
    """
    state = result.state
    iterations: list[IterationSummary] = []
    if state is not None:
        for it in state.iterations:
            next_action = (it.plan or {}).get("next_action", {})
            evaluation = it.evaluation or {}
            iterations.append(
                IterationSummary(
                    index=it.index,
                    tool=next_action.get("tool"),
                    step_id=next_action.get("step_id"),
                    rationale=next_action.get("rationale"),
                    tool_calls=[
                        {
                            "tool": tc.tool,
                            "arguments": tc.arguments,
                            "ok": tc.ok,
                            "error_code": tc.error_code,
                            "duration_ms": tc.duration_ms,
                        }
                        for tc in it.tool_calls
                    ],
                    evaluation_outcome=evaluation.get("outcome"),
                    evaluation_evidence=evaluation.get("evidence"),
                    violations=list(it.violations),
                )
            )

    decisions = list(agent.gateway.decisions) if agent.gateway is not None else []
    denied = sum(1 for d in decisions if d.get("blocked"))
    approved = sum(1 for d in decisions if d.get("tier") == "DANGER" and not d.get("blocked"))

    governance_summary: dict[str, Any] | None = None
    if agent.governance is not None:
        g = agent.governance
        governance_summary = {
            "allowed_tools": sorted(g.allowed_tools) if g.allowed_tools is not None else None,
            "sensitive_operations": sorted(g.sensitive_operations),
            "approval_threshold": str(g.approval_threshold),
        }

    return AuditReport(
        run_id=state.run_id if state is not None else "",
        session_id=result.session_id,
        goal=state.goal if state is not None else "",
        status=result.status,
        model=agent.llm.model,
        iterations=iterations,
        guardrail_decisions=decisions,
        approved_count=approved,
        denied_count=denied,
        cost_usd=round(agent.ledger.total_usd, 6),
        cost_by_phase={k: round(v, 6) for k, v in agent.ledger.by_phase().items()},
        final_answer=result.answer,
        confidence=result.confidence,
        evidence=list(result.evidence),
        unmet_requirements=list(result.unmet_requirements),
        governance=governance_summary,
    )

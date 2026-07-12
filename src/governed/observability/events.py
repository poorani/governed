"""Every observable thing the agent does is an ``Event``.

Events are the only mechanism by which the agent talks to the outside world
mid-run. Logging, cost accounting, progress bars, and human-approval UIs are all
just subscribers. Nothing in the core imports a logging framework.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    RUN_START = "run.start"
    RUN_END = "run.end"

    ITERATION_START = "iteration.start"
    ITERATION_END = "iteration.end"

    PHASE_START = "phase.start"
    PHASE_END = "phase.end"

    LLM_REQUEST = "llm.request"
    LLM_RESPONSE = "llm.response"

    PLAN_CREATED = "plan.created"
    EVALUATION_CREATED = "evaluation.created"
    CONTRACT_VIOLATION = "contract.violation"

    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_DECIDED = "approval.decided"

    RISK_ASSESSED = "guardrail.risk"
    GUARDRAIL_FINDING = "guardrail.finding"
    GUARDRAIL_BLOCKED = "guardrail.blocked"

    COST_RECORDED = "cost.recorded"
    COST_WARNING = "cost.warning"
    CIRCUIT_OPEN = "cost.circuit_open"

    SKILL_LOADED = "skill.loaded"
    COMPACTION = "memory.compaction"
    BUDGET_EXCEEDED = "budget.exceeded"
    CANCELLED = "run.cancelled"
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    run_id: str
    iteration: int = 0
    phase: str = ""
    #: Free-form payload. Kept JSON-serialisable so the JSONL sink is lossless.
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "type": self.type.value,
            "run_id": self.run_id,
            "iteration": self.iteration,
            "phase": self.phase,
            "data": self.data,
        }


Subscriber = Callable[[Event], None]


class EventBus:
    """Synchronous fan-out. A failing subscriber can never break a run."""

    def __init__(self, subscribers: list[Subscriber] | None = None) -> None:
        self._subs: list[Subscriber] = list(subscribers or [])

    def subscribe(self, fn: Subscriber) -> Subscriber:
        self._subs.append(fn)
        return fn

    def emit(self, event: Event) -> Event:
        for sub in self._subs:
            # A failing subscriber must never break a run.
            with contextlib.suppress(Exception):
                sub(event)
        return event

"""``SessionState`` -- the transcript, scratchpad, and iteration history of one run.

Round-trips through JSON so a ``StateStore`` can persist it and ``Agent.resume``
can pick it back up. Three tiers, deliberately kept separate (see the module
docstring in ``agent.py`` and the README's "Memory and state" section):

* ``transcript`` -- the raw message list. Large, lossy under compaction.
* ``scratchpad`` -- small facts the model explicitly chose to keep. Never
  compacted.
* ``iterations`` -- the structured plan -> calls -> evaluation history. Never
  sent to the model; read by the trace and by ``resume``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from ..contracts import Evaluation, Plan
from ..llm.base import Message, ToolCall, ToolResultBlock, Usage

__all__ = [
    "IterationRecord",
    "RunStatus",
    "SessionState",
    "ToolCallRecord",
    "evaluation_from_dict",
    "plan_from_dict",
]

RunStatus = Literal[
    "running", "complete", "partial", "blocked", "exhausted", "failed", "cancelled"
]


@dataclass
class ToolCallRecord:
    call_id: str
    tool: str
    arguments: dict[str, Any]
    rationale: str
    step_id: str
    ok: bool
    result_preview: str
    error_code: str | None = None
    duration_ms: int = 0
    artifacts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class IterationRecord:
    index: int
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    plan: dict[str, Any] | None = None
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    evaluation: dict[str, Any] | None = None
    violations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return (self.ended_at - self.started_at) if self.ended_at else 0.0


@dataclass
class SessionState:
    goal: str
    #: The stable identifier for this run of work. Used as the store key and
    #: carried through resume; ``run_id`` and ``session_id`` are the same value
    #: unless a caller explicitly assigns a memorable ``session_id``.
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: RunStatus = "running"
    iteration: int = 0
    iterations: list[IterationRecord] = field(default_factory=list)
    transcript: list[Message] = field(default_factory=list)
    scratchpad: dict[str, Any] = field(default_factory=dict)
    #: The rolling summary produced by compaction. Prepended to the system
    #: prompt; never re-injected into the transcript itself.
    summary: str = ""
    summarized_through: int = 0
    tool_call_count: int = 0
    usage: Usage = field(default_factory=Usage)
    final_answer: dict[str, Any] | None = None
    created_at: float = field(default_factory=time.time)

    # -- mutation, called only by Agent ------------------------------------

    def begin_iteration(self) -> IterationRecord:
        self.iteration += 1
        rec = IterationRecord(index=self.iteration)
        self.iterations.append(rec)
        return rec

    @property
    def current(self) -> IterationRecord | None:
        return self.iterations[-1] if self.iterations else None

    def add_message(self, message: Message) -> None:
        self.transcript.append(message)

    def record_usage(self, usage: Usage) -> None:
        self.usage.input_tokens += usage.input_tokens
        self.usage.output_tokens += usage.output_tokens
        self.usage.cache_read_tokens += usage.cache_read_tokens
        self.usage.cache_write_tokens += usage.cache_write_tokens

    def consecutive_failures(self) -> int:
        """Trailing run of `failure` evaluations, most recent first."""
        n = 0
        for rec in reversed(self.iterations):
            if rec.evaluation is None:
                continue
            if rec.evaluation.get("outcome") == "failure":
                n += 1
            else:
                break
        return n

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "status": self.status,
            "iteration": self.iteration,
            "iterations": [_iteration_to_dict(it) for it in self.iterations],
            "transcript": [_message_to_dict(m) for m in self.transcript],
            "scratchpad": self.scratchpad,
            "summary": self.summary,
            "summarized_through": self.summarized_through,
            "tool_call_count": self.tool_call_count,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
                "cache_read_tokens": self.usage.cache_read_tokens,
                "cache_write_tokens": self.usage.cache_write_tokens,
            },
            "final_answer": self.final_answer,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionState:
        state = cls(
            goal=data["goal"],
            session_id=data.get("session_id", uuid.uuid4().hex[:12]),
            run_id=data.get("run_id", uuid.uuid4().hex[:12]),
            status=data.get("status", "running"),
            iteration=data.get("iteration", 0),
            scratchpad=data.get("scratchpad", {}),
            summary=data.get("summary", ""),
            summarized_through=data.get("summarized_through", 0),
            tool_call_count=data.get("tool_call_count", 0),
            final_answer=data.get("final_answer"),
            created_at=data.get("created_at", time.time()),
        )
        u = data.get("usage") or {}
        state.usage = Usage(
            input_tokens=u.get("input_tokens", 0),
            output_tokens=u.get("output_tokens", 0),
            cache_read_tokens=u.get("cache_read_tokens", 0),
            cache_write_tokens=u.get("cache_write_tokens", 0),
        )
        state.iterations = [_iteration_from_dict(d) for d in data.get("iterations", [])]
        state.transcript = [_message_from_dict(d) for d in data.get("transcript", [])]
        return state


# ---------------------------------------------------------------------------
# (de)serialisation helpers
# ---------------------------------------------------------------------------


def _message_to_dict(m: Message) -> dict[str, Any]:
    return {
        "role": m.role,
        "text": m.text,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
        ],
        "tool_results": [
            {"call_id": tr.call_id, "content": tr.content, "is_error": tr.is_error}
            for tr in m.tool_results
        ],
        "meta": m.meta,
    }


def _message_from_dict(d: dict[str, Any]) -> Message:
    return Message(
        role=d["role"],
        text=d.get("text", ""),
        tool_calls=[ToolCall(**tc) for tc in d.get("tool_calls", [])],
        tool_results=[ToolResultBlock(**tr) for tr in d.get("tool_results", [])],
        meta=d.get("meta", {}),
    )


def _iteration_to_dict(it: IterationRecord) -> dict[str, Any]:
    return {
        "index": it.index,
        "started_at": it.started_at,
        "ended_at": it.ended_at,
        "plan": it.plan,
        "tool_calls": [vars(tc) for tc in it.tool_calls],
        "evaluation": it.evaluation,
        "violations": it.violations,
    }


def _iteration_from_dict(d: dict[str, Any]) -> IterationRecord:
    return IterationRecord(
        index=d["index"],
        started_at=d.get("started_at", time.time()),
        ended_at=d.get("ended_at"),
        plan=d.get("plan"),
        tool_calls=[ToolCallRecord(**tc) for tc in d.get("tool_calls", [])],
        evaluation=d.get("evaluation"),
        violations=d.get("violations", []),
    )


def plan_from_dict(data: dict[str, Any] | None) -> Plan | None:
    """Reconstruct a typed ``Plan`` from an ``IterationRecord.plan`` dict."""
    return Plan.model_validate(data) if data else None


def evaluation_from_dict(data: dict[str, Any] | None) -> Evaluation | None:
    """Reconstruct a typed ``Evaluation`` from an ``IterationRecord.evaluation`` dict."""
    return Evaluation.model_validate(data) if data else None

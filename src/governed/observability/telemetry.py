"""Enterprise telemetry: per-call LLM/tool instrumentation and session timing.

``TelemetryCollector`` is a plain ``EventBus`` subscriber -- wired in exactly
like ``TraceLogger``, via ``AgentConfig(subscribers=[...])``. Nothing in the
core loop imports this module or calls back into it; it only reads the events
``Agent`` already emits. That is the same discipline the rest of
``observability`` follows: logging, cost accounting, and this are all just
listeners on one bus, and a failing listener can never break a run.

Five questions this answers that reading the trace by hand does not:

1. **LLM instrumentation.** Request count, latency (``latency_ms``), status
   (``"ok"`` or the provider's exception type / HTTP status code, best-effort
   -- see the caveat below), and cumulative token usage, in aggregate and
   broken out by phase (ANALYZE / ACT / OBSERVE / compaction).
2. **Tool / dependency instrumentation.** Latency and success rate per tool
   name. This is the number that tells you whether a tool wrapping a
   third-party API or a database is degrading, as distinct from the model
   being slow -- the two look identical in wall-clock alone.
3. **Session timing.** Wall-clock duration split into active processing time
   and human-in-the-loop idle wait time: the gap between an
   ``APPROVAL_REQUESTED`` event and its matching ``APPROVAL_DECIDED``, which
   is exactly the span during which a human (or a webhook) was the
   bottleneck, not the agent. A twenty-minute run where a human was at lunch
   for eighteen of them should not read as a slow agent.
4. **Cost.** The ledger's running total (in USD), read straight off
   ``COST_RECORDED`` events -- no separate wiring to ``CostLedger`` required.
5. **Safety posture.** How often the framework actually refused to do
   something: blocked guardrail calls (by tool), circuit breaker trips (by
   reason), and budget exhaustions (by resource) -- cumulative across every
   run this collector has observed, which is what an operations dashboard
   wants to answer "is this safe to leave unattended," not a single run.

Caveat on "status codes": ``LLMClient.complete`` is a provider-agnostic
interface -- it does not standardise on HTTP. On success, status is always
``"ok"``. On failure, ``Agent`` best-effort extracts an HTTP status code from
the raised exception (``anthropic``/``openai`` SDK errors expose
``.status_code``) and falls back to the exception's type name. A self-hosted
or custom ``LLMClient`` that raises plain exceptions will show up as
``"error:<ExceptionType>"`` rather than an HTTP code -- that is expected, not
a bug.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .events import Event, EventType

__all__ = [
    "LLMCallStats",
    "ToolCallStats",
    "SessionTiming",
    "SafetyStats",
    "TelemetryCollector",
]


@dataclass
class LLMCallStats:
    """Aggregate stats over a set of LLM calls -- overall, or scoped to one phase."""

    count: int = 0
    error_count: int = 0
    total_latency_ms: int = 0
    max_latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    #: e.g. {"ok": 12, "error:RateLimitError": 1, "http_529": 1}
    status_codes: dict[str, int] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.count if self.count else 0.0

    @property
    def success_rate(self) -> float:
        return (self.count - self.error_count) / self.count if self.count else 1.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def _observe(
        self,
        *,
        latency_ms: int,
        ok: bool,
        status_key: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.count += 1
        self.error_count += 0 if ok else 1
        self.total_latency_ms += latency_ms
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.status_codes[status_key] = self.status_codes.get(status_key, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "error_count": self.error_count,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "max_latency_ms": self.max_latency_ms,
            "success_rate": round(self.success_rate, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "status_codes": dict(self.status_codes),
        }


@dataclass
class ToolCallStats:
    """Latency and success rate for one tool.

    This is what matters when the tool wraps a third-party API or a database:
    it tells you whether *the dependency* is degrading, which a run's overall
    wall-clock time cannot distinguish from the model thinking slowly.
    """

    tool: str
    count: int = 0
    error_count: int = 0
    total_latency_ms: int = 0
    max_latency_ms: int = 0
    #: e.g. {"timeout": 1, "invalid_input": 2}
    error_codes: dict[str, int] = field(default_factory=dict)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.count if self.count else 0.0

    @property
    def success_rate(self) -> float:
        return (self.count - self.error_count) / self.count if self.count else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "count": self.count,
            "error_count": self.error_count,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "max_latency_ms": self.max_latency_ms,
            "success_rate": round(self.success_rate, 4),
            "error_codes": dict(self.error_codes),
        }


@dataclass
class SessionTiming:
    """One run's wall clock, split into active processing and HITL idle wait.

    ``idle_wait_s`` accumulates every ``APPROVAL_REQUESTED`` -> ``APPROVAL_DECIDED``
    gap in the run -- a blocking terminal prompt, a webhook poll, whatever the
    approver does. ``active_s`` is what is left: time the agent itself spent
    planning, calling tools, and calling the model.
    """

    started_ts: float | None = None
    ended_ts: float | None = None
    idle_wait_s: float = 0.0
    #: The ledger's running total for this run, read off ``COST_RECORDED``.
    cost_usd: float = 0.0

    @property
    def total_s(self) -> float:
        if self.started_ts is None:
            return 0.0
        end = self.ended_ts if self.ended_ts is not None else time.time()
        return max(0.0, end - self.started_ts)

    @property
    def active_s(self) -> float:
        return max(0.0, self.total_s - self.idle_wait_s)

    def to_dict(self) -> dict[str, Any]:
        total = self.total_s
        return {
            "total_s": round(total, 3),
            "active_s": round(self.active_s, 3),
            "idle_wait_s": round(self.idle_wait_s, 3),
            "idle_ratio": round(self.idle_wait_s / total, 4) if total else 0.0,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class SafetyStats:
    """Cumulative guardrail / circuit-breaker activity across every run this
    collector has observed -- not scoped to one session, on purpose. The
    question this answers is operational ("how often does this deployment
    refuse a dangerous action, and what kind"), which is naturally a
    fleet-wide number, not a per-run one.
    """

    blocked_calls: int = 0
    #: e.g. {"execute_code": 3, "file_system": 1}
    blocked_by_tool: dict[str, int] = field(default_factory=dict)
    circuit_trips: int = 0
    #: e.g. {"cost_ceiling": 2, "stalled": 1}
    circuit_trip_reasons: dict[str, int] = field(default_factory=dict)
    budget_exhaustions: int = 0
    #: e.g. {"max_iterations=20": 4}
    budget_exhaustions_by_resource: dict[str, int] = field(default_factory=dict)
    #: Runs stopped by Agent.cancel() rather than any budget/circuit limit.
    cancellations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked_calls": self.blocked_calls,
            "blocked_by_tool": dict(self.blocked_by_tool),
            "circuit_trips": self.circuit_trips,
            "circuit_trip_reasons": dict(self.circuit_trip_reasons),
            "budget_exhaustions": self.budget_exhaustions,
            "budget_exhaustions_by_resource": dict(self.budget_exhaustions_by_resource),
            "cancellations": self.cancellations,
        }


def _error_status_key(data: dict[str, Any]) -> str:
    code = data.get("status_code")
    if code:
        return f"http_{code}"
    err = data.get("error_type")
    return f"error:{err}" if err else "error"


def _overall_tool_success_rate(tools: dict[str, ToolCallStats]) -> float:
    total = sum(t.count for t in tools.values())
    if not total:
        return 1.0
    errors = sum(t.error_count for t in tools.values())
    return (total - errors) / total


class TelemetryCollector:
    """Subscribe this to an ``Agent``'s event bus for LLM, tool, and
    session-timing telemetry, aggregated for free from events the agent
    already emits.

    ::

        telemetry = TelemetryCollector()
        agent = Agent(AgentConfig(llm=..., subscribers=[telemetry]))
        result = agent.run(goal)

        print(telemetry.summary())
        telemetry.to_dict()   # ship this to your own metrics backend

    One collector can be reused across multiple ``Agent.run()`` calls, or
    shared by multiple agents -- sessions are keyed by ``run_id``, and
    ``.session`` is a convenience accessor for the most recently started one.
    """

    def __init__(self) -> None:
        self.llm = LLMCallStats()
        self.llm_by_phase: dict[str, LLMCallStats] = {}
        self.tools: dict[str, ToolCallStats] = {}
        self.sessions: dict[str, SessionTiming] = {}
        self.safety = SafetyStats()
        self._pending_approvals: dict[str, float] = {}
        self._last_run_id: str | None = None

    @property
    def session(self) -> SessionTiming:
        """The most recently started session. The common case: one collector,
        one run, read this after ``agent.run()`` returns."""
        if self._last_run_id is None:
            return SessionTiming()
        return self.sessions[self._last_run_id]

    def __call__(self, event: Event) -> None:
        if event.type is EventType.RUN_START:
            self._on_run_start(event)
        elif event.type is EventType.RUN_END:
            self._on_run_end(event)
        elif event.type is EventType.LLM_RESPONSE:
            self._on_llm_response(event)
        elif event.type is EventType.TOOL_RESULT:
            self._on_tool_result(event)
        elif event.type is EventType.APPROVAL_REQUESTED:
            self._pending_approvals[event.run_id] = event.ts
        elif event.type is EventType.APPROVAL_DECIDED:
            self._on_approval_decided(event)
        elif event.type is EventType.COST_RECORDED:
            self._on_cost_recorded(event)
        elif event.type is EventType.GUARDRAIL_BLOCKED:
            self._on_guardrail_blocked(event)
        elif event.type is EventType.CIRCUIT_OPEN:
            self._on_circuit_open(event)
        elif event.type is EventType.BUDGET_EXCEEDED:
            self._on_budget_exceeded(event)
        elif event.type is EventType.CANCELLED:
            self.safety.cancellations += 1

    # -- handlers -----------------------------------------------------------

    def _on_run_start(self, event: Event) -> None:
        self.sessions[event.run_id] = SessionTiming(started_ts=event.ts)
        self._last_run_id = event.run_id

    def _on_run_end(self, event: Event) -> None:
        timing = self.sessions.setdefault(event.run_id, SessionTiming(started_ts=event.ts))
        timing.ended_ts = event.ts

    def _on_llm_response(self, event: Event) -> None:
        d = event.data
        ok = d.get("status") == "ok"
        status_key = "ok" if ok else _error_status_key(d)
        kwargs: dict[str, Any] = {
            "latency_ms": int(d.get("latency_ms", 0)),
            "ok": ok,
            "status_key": status_key,
            "input_tokens": int(d.get("input_tokens", 0)),
            "output_tokens": int(d.get("output_tokens", 0)),
        }
        self.llm._observe(**kwargs)
        phase = event.phase or "unknown"
        self.llm_by_phase.setdefault(phase, LLMCallStats())._observe(**kwargs)

    def _on_tool_result(self, event: Event) -> None:
        d = event.data
        tool = str(d.get("tool", "unknown"))
        stats = self.tools.setdefault(tool, ToolCallStats(tool=tool))
        stats.count += 1
        latency = int(d.get("duration_ms", 0))
        stats.total_latency_ms += latency
        stats.max_latency_ms = max(stats.max_latency_ms, latency)
        if not d.get("ok"):
            stats.error_count += 1
            code = str(d.get("error_code") or "unknown")
            stats.error_codes[code] = stats.error_codes.get(code, 0) + 1

    def _on_approval_decided(self, event: Event) -> None:
        started = self._pending_approvals.pop(event.run_id, None)
        if started is None:
            return
        timing = self.sessions.setdefault(event.run_id, SessionTiming())
        timing.idle_wait_s += max(0.0, event.ts - started)

    def _on_cost_recorded(self, event: Event) -> None:
        timing = self.sessions.setdefault(event.run_id, SessionTiming())
        timing.cost_usd = float(event.data.get("run_usd", timing.cost_usd))

    def _on_guardrail_blocked(self, event: Event) -> None:
        self.safety.blocked_calls += 1
        tool = str(event.data.get("tool", "unknown"))
        self.safety.blocked_by_tool[tool] = self.safety.blocked_by_tool.get(tool, 0) + 1

    def _on_circuit_open(self, event: Event) -> None:
        self.safety.circuit_trips += 1
        reason = str(event.data.get("reason", "unknown"))
        counts = self.safety.circuit_trip_reasons
        counts[reason] = counts.get(reason, 0) + 1

    def _on_budget_exceeded(self, event: Event) -> None:
        self.safety.budget_exhaustions += 1
        which = str(event.data.get("which", "unknown"))
        counts = self.safety.budget_exhaustions_by_resource
        counts[which] = counts.get(which, 0) + 1

    # -- reporting ------------------------------------------------------

    def summary(self) -> str:
        lines = [
            f"LLM: {self.llm.count} requests, {self.llm.avg_latency_ms:.0f}ms avg "
            f"({self.llm.max_latency_ms}ms max), {self.llm.success_rate:.1%} ok, "
            f"{self.llm.total_tokens} tokens"
        ]
        for phase, phase_stats in sorted(self.llm_by_phase.items()):
            lines.append(
                f"  {phase:<12} {phase_stats.count:>4} req  "
                f"{phase_stats.avg_latency_ms:>7.0f}ms avg  "
                f"{phase_stats.success_rate:>6.1%} ok"
            )
        if self.tools:
            lines.append("Tools:")
            for name, tool_stats in sorted(self.tools.items()):
                lines.append(
                    f"  {name:<16} {tool_stats.count:>4} calls  "
                    f"{tool_stats.avg_latency_ms:>7.0f}ms avg  "
                    f"{tool_stats.success_rate:>6.1%} ok"
                )
        for run_id, timing in self.sessions.items():
            lines.append(
                f"Session {run_id}: {timing.total_s:.1f}s total = "
                f"{timing.active_s:.1f}s active + {timing.idle_wait_s:.1f}s HITL idle, "
                f"${timing.cost_usd:.4f}"
            )
        if (
            self.safety.blocked_calls
            or self.safety.circuit_trips
            or self.safety.budget_exhaustions
            or self.safety.cancellations
        ):
            lines.append(
                f"Safety: {self.safety.blocked_calls} blocked, "
                f"{self.safety.circuit_trips} circuit trips, "
                f"{self.safety.budget_exhaustions} budget exhaustions, "
                f"{self.safety.cancellations} cancelled"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm": self.llm.to_dict(),
            "llm_by_phase": {k: v.to_dict() for k, v in self.llm_by_phase.items()},
            "tools": {k: v.to_dict() for k, v in self.tools.items()},
            "sessions": {k: v.to_dict() for k, v in self.sessions.items()},
            "safety": self.safety.to_dict(),
        }

    def overview(self) -> dict[str, Any]:
        """A one-shot, dashboard-shaped rollup of the most recent session --
        total run time, active vs. idle, tokens, cost, and safety events.
        This is the shape most metrics backends want; ``to_dict()`` is the
        full structure if you need the per-phase / per-tool / per-run detail.
        """
        timing = self.session
        return {
            "total_run_time_s": round(timing.total_s, 3),
            "active_time_s": round(timing.active_s, 3),
            "idle_wait_s": round(timing.idle_wait_s, 3),
            "llm_requests": self.llm.count,
            "llm_success_rate": round(self.llm.success_rate, 4),
            "total_tokens": self.llm.total_tokens,
            "total_cost_usd": round(timing.cost_usd, 6),
            "tool_calls": sum(t.count for t in self.tools.values()),
            "tool_success_rate": round(_overall_tool_success_rate(self.tools), 4),
            "blocked_calls": self.safety.blocked_calls,
            "circuit_trips": self.safety.circuit_trips,
            "budget_exhaustions": self.safety.budget_exhaustions,
        }

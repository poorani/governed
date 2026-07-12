"""Tracing.

The design goal is answering, after the fact and without re-running anything:

    "Which tool did it call at iteration 4, what arguments, *why*, what came
     back, and what did it conclude from that?"

The ``why`` is not inferred. It is the ``rationale`` from the plan that
authorised the call, carried through on the ``tool.call`` event. Because the ACT
phase contract forbids calling a tool the plan did not commit to, every tool
call in the trace has a real, pre-registered justification attached.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, TextIO

from .events import Event, EventBus, EventType
from .exporters import (
    HttpTransport,
    default_http_transport,
    otlp_log_record,
    otlp_resource_logs,
)

_ICON = {
    EventType.RUN_START: "\u25b6",
    EventType.RUN_END: "\u25a0",
    EventType.PLAN_CREATED: "\u25cb",
    EventType.TOOL_CALL: "\u2192",
    EventType.TOOL_RESULT: "\u2190",
    EventType.EVALUATION_CREATED: "\u2713",
    EventType.CONTRACT_VIOLATION: "\u26a0",
    EventType.BUDGET_EXCEEDED: "\u26a0",
    EventType.ERROR: "\u2717",
    EventType.GUARDRAIL_FINDING: "\u26a1",
    EventType.GUARDRAIL_BLOCKED: "\u26d4",
    EventType.CIRCUIT_OPEN: "\u26d4",
    EventType.COST_RECORDED: "$",
    EventType.COST_WARNING: "\u26a0",
    EventType.CANCELLED: "\u26d4",
}


class JSONLSink:
    """Append-only, one JSON object per line. The canonical record of a run."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = self.path.open("a", encoding="utf-8")

    def __call__(self, event: Event) -> None:
        self._fh.write(json.dumps(event.to_dict(), default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class ConsoleSink:
    """Human-readable running commentary. Terse by design."""

    def __init__(self, stream: TextIO | None = None, verbose: bool = False) -> None:
        self.stream = stream or sys.stderr
        self.verbose = verbose

    def __call__(self, event: Event) -> None:
        line = self._format(event)
        if line:
            print(line, file=self.stream, flush=True)

    def _format(self, e: Event) -> str:
        icon = _ICON.get(e.type, " ")
        it = f"[{e.iteration:>2}]" if e.iteration else "[  ]"
        d = e.data

        if e.type is EventType.RUN_START:
            return f"{icon} {it} run {e.run_id} :: {_clip(d.get('goal', ''), 100)}"
        if e.type is EventType.PLAN_CREATED:
            na = d.get("next_action", {})
            return (
                f"{icon} {it} plan -> {na.get('tool')} :: {_clip(na.get('rationale', ''), 80)}"
            )
        if e.type is EventType.TOOL_CALL:
            return (
                f"{icon} {it} {d.get('tool')}({_clip(json.dumps(d.get('arguments', {})), 90)})"
            )
        if e.type is EventType.TOOL_RESULT:
            status = "ok" if d.get("ok") else f"FAIL {d.get('error_code')}"
            return f"{icon} {it} {d.get('tool')} {status} in {d.get('duration_ms')}ms"
        if e.type is EventType.EVALUATION_CREATED:
            return (
                f"{icon} {it} {d.get('outcome')} / goal {d.get('goal_status')} "
                f":: {_clip(d.get('next_step', ''), 70)}"
            )
        if e.type is EventType.CONTRACT_VIOLATION:
            return f"{icon} {it} contract violation in {e.phase}: {d.get('reason')}"
        if e.type is EventType.GUARDRAIL_FINDING:
            return (
                f"{icon} {it} {d.get('rule_id')} [{d.get('severity')}] in "
                f"{d.get('source')}: {_clip(d.get('message', ''), 70)}"
            )
        if e.type is EventType.GUARDRAIL_BLOCKED:
            return f"{icon} {it} BLOCKED {d.get('tool')}: {_clip(d.get('reason', ''), 80)}"
        if e.type is EventType.CIRCUIT_OPEN:
            return f"{icon} {it} circuit breaker: {d.get('reason')} -- {d.get('detail')}"
        if e.type is EventType.COST_WARNING:
            return f"{icon} {it} {d.get('message')}"
        if e.type is EventType.COST_RECORDED and self.verbose:
            return f"{icon} {it} ${d.get('run_usd', 0):.4f} total ({d.get('phase')})"
        if e.type is EventType.BUDGET_EXCEEDED:
            return f"{icon} {it} budget exceeded: {d.get('which')}"
        if e.type is EventType.CANCELLED:
            reason = d.get("reason")
            return f"{icon} {it} cancelled" + (f": {reason}" if reason else "")
        if e.type is EventType.ERROR:
            return f"{icon} {it} {d.get('message')}"
        if e.type is EventType.RUN_END:
            return (
                f"{icon} {it} {d.get('status')} in {d.get('iterations')} iterations, "
                f"{d.get('total_tokens')} tokens"
            )
        if self.verbose:
            return f"  {it} {e.type.value} {_clip(json.dumps(d, default=str), 120)}"
        return ""


class LoggingSink:
    """Bridge to the stdlib ``logging`` module for apps that already have one."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("governed")

    def __call__(self, event: Event) -> None:
        level = logging.ERROR if event.type is EventType.ERROR else logging.INFO
        self.logger.log(level, event.type.value, extra={"governed_event": event.to_dict()})


class HttpEventSink:
    """Streams every event as a JSON POST to any HTTP ingestion endpoint --
    the event-trace counterpart to ``HttpDecisionLedgerSink`` (same shape,
    same transport, different stream). Splunk's HTTP Event Collector,
    Datadog's logs intake, New Relic's Log API, and Dynatrace's generic
    ingest API all accept this directly: arbitrary structured JSON, one
    event per request, authenticated by a header.

    ``event_types`` optionally scopes down what gets shipped -- useful when
    only e.g. ``{EventType.TOOL_CALL, EventType.RUN_END}`` matters to the
    downstream system and every ``COST_RECORDED`` (emitted once per
    completion) would just be noise and spend. ``None`` (the default) ships
    everything, same as any other ``Subscriber``.

    ``transport`` is injectable so this is testable without a network. A
    failing sink never breaks a run -- ``EventBus.emit`` already suppresses
    subscriber exceptions.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        event_types: set[EventType] | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        self.url = url
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.event_types = event_types
        self.transport = transport or default_http_transport

    def __call__(self, event: Event) -> None:
        if self.event_types is not None and event.type not in self.event_types:
            return
        self.transport(self.url, event.to_dict(), self.headers)


class OTelEventSink:
    """Streams every event as an OpenTelemetry log record over OTLP/HTTP --
    the event-trace counterpart to ``OTelDecisionLedgerSink``. Hand-built
    against OTLP's documented JSON wire format, not the ``opentelemetry-sdk``
    package -- no new dependency, works the moment you point it at a
    Collector or any backend with native OTLP ingestion (Datadog, New Relic,
    Dynatrace, and recent Splunk all have one today).

    ``event_types`` filters which events are shipped, the same as
    ``HttpEventSink``'s.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        service_name: str = "governed",
        event_types: set[EventType] | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.service_name = service_name
        self.event_types = event_types
        self.transport = transport or default_http_transport

    def __call__(self, event: Event) -> None:
        if self.event_types is not None and event.type not in self.event_types:
            return
        log_record = otlp_log_record(
            ts=event.ts,
            body=json.dumps(event.to_dict(), default=str),
            attributes={
                "governed.run_id": event.run_id,
                "governed.event_type": event.type.value,
                "governed.iteration": event.iteration,
                "governed.phase": event.phase,
            },
        )
        body = otlp_resource_logs(
            service_name=self.service_name,
            scope_name="governed.trace",
            log_records=[log_record],
        )
        self.transport(f"{self.endpoint}/v1/logs", body, self.headers)


class TraceLogger:
    """Convenience wrapper that wires the standard sinks onto an ``EventBus``."""

    def __init__(
        self,
        run_id: str,
        *,
        jsonl_path: str | Path | None = None,
        console: bool = True,
        verbose: bool = False,
        extra_subscribers: list[Any] | None = None,
    ) -> None:
        self.run_id = run_id
        self.bus = EventBus()
        self._jsonl: JSONLSink | None = None

        if jsonl_path:
            self._jsonl = JSONLSink(jsonl_path)
            self.bus.subscribe(self._jsonl)
        if console:
            self.bus.subscribe(ConsoleSink(verbose=verbose))
        for sub in extra_subscribers or []:
            self.bus.subscribe(sub)

    def emit(
        self,
        type: EventType,
        *,
        iteration: int = 0,
        phase: str = "",
        **data: Any,
    ) -> Event:
        return self.bus.emit(
            Event(type=type, run_id=self.run_id, iteration=iteration, phase=phase, data=data)
        )

    def close(self) -> None:
        if self._jsonl:
            self._jsonl.close()


# --------------------------------------------------------------------------
# Post-hoc rendering
# --------------------------------------------------------------------------


def read_trace(jsonl_path: str | Path) -> list[dict[str, Any]]:
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def trace_to_markdown(jsonl_path: str | Path) -> str:
    """Render a run as a reviewable narrative: plan, action, why, result, verdict."""
    out: list[str] = []
    for rec in read_trace(jsonl_path):
        t, d, it = rec["type"], rec["data"], rec["iteration"]
        if t == EventType.RUN_START.value:
            out += [f"# Run `{rec['run_id']}`", "", f"**Goal:** {d.get('goal')}", ""]
        elif t == EventType.ITERATION_START.value:
            out += [f"## Iteration {it}", ""]
        elif t == EventType.PLAN_CREATED.value:
            na = d.get("next_action", {})
            steps = "\n".join(
                f"  {'x' if s.get('done') else ' '} `{s['id']}` {s['description']}"
                for s in d.get("steps", [])
            )
            out += [
                "**Plan**",
                "",
                steps,
                "",
                f"**Next action:** `{na.get('tool')}` for step `{na.get('step_id')}`",
                f"**Why:** {na.get('rationale')}",
                f"**Success looks like:** {na.get('success_criteria')}",
                "",
            ]
        elif t == EventType.TOOL_CALL.value:
            args = json.dumps(d.get("arguments", {}), indent=2)
            out += [f"**Called** `{d.get('tool')}`", "", "```json", args, "```", ""]
        elif t == EventType.TOOL_RESULT.value:
            head = "Result" if d.get("ok") else f"Result (FAILED: {d.get('error_code')})"
            out += [
                f"**{head}** ({d.get('duration_ms')}ms)",
                "",
                "```",
                _clip(d.get("preview", ""), 800),
                "```",
                "",
            ]
        elif t == EventType.EVALUATION_CREATED.value:
            out += [
                f"**Evaluation:** {d.get('outcome')} -- goal is {d.get('goal_status')}",
                f"> {d.get('evidence')}",
                "",
                f"**Next:** {d.get('next_step')}",
                "",
            ]
        elif t == EventType.CONTRACT_VIOLATION.value:
            out += [
                f"> [!WARNING] Contract violation in `{rec['phase']}`: {d.get('reason')}",
                "",
            ]
        elif t == EventType.RUN_END.value:
            out += [
                "---",
                "",
                f"## Outcome: {d.get('status')}",
                "",
                f"{d.get('answer', '')}",
                "",
                f"*{d.get('iterations')} iterations, {d.get('total_tokens')} tokens, "
                f"{d.get('duration_s', 0):.1f}s*",
            ]
    return "\n".join(out)


def _clip(text: str, n: int) -> str:
    text = str(text).replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "\u2026"

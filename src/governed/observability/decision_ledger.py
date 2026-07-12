"""An immutable, tamper-evident record of every decision an agent made.

Where `TraceLogger`/`AuditReport` answer "what happened, readably," the
decision ledger answers a stricter question: "can I prove, after the fact,
that this record has not been altered?" One `DecisionRecord` is written per
completed iteration -- the plan, the model's stated rationale, the tool(s)
selected and their outcomes, every safety check performed during EXECUTE, and
the evaluation's evidence -- plus one guaranteed final record at the end of
every run, however it ended, carrying the run's terminal outcome.

Each record is chained to the one before it: `entry_hash` is a SHA-256 over
the record's own content plus the previous record's `entry_hash`. Edit,
delete, or reorder any entry and every hash from that point forward stops
matching what `verify_chain` recomputes. Read the honesty this project
applies everywhere else in `security/guardrails.py`'s module docstring and
apply it here too: this is *detection*, not a lock. Nothing stops someone
with write access to the underlying store from regenerating the entire chain
from scratch, consistently, after the fact. What it does guarantee is that
the far more common tampering scenario -- touching up *one* inconvenient
entry after the fact, leaving the rest alone -- is always detectable, because
that one edit invalidates every hash after it.

Two independent, composable, optional pieces, wired through
``AgentConfig(decision_ledger=DecisionLedgerConfig(...))``:

* ``DecisionLedgerStore`` -- durable, append-only storage. ``JSONLDecisionLedger``
  (a hash-chained JSONL file) is the reference implementation;
  ``InMemoryDecisionLedger`` exists for tests.
* ``DecisionLedgerSink`` -- fan-out to external observability systems, the
  same shape as ``Subscriber`` for the event trace. ``HttpDecisionLedgerSink``
  posts each record as JSON to any HTTP ingestion endpoint (Splunk HEC,
  Datadog's logs intake, New Relic's Log API, and Dynatrace's ingest API all
  accept this shape); ``OTelDecisionLedgerSink`` speaks OTLP/HTTP's documented
  JSON wire format directly -- no ``opentelemetry-sdk`` dependency, so it
  works the moment you point it at a Collector or any backend with native
  OTLP ingestion, which by now includes all four vendors above.

Neither piece changes the agent loop. Records are built from data ``Agent``
already collects (``IterationRecord``, ``Gateway.decisions``), written
alongside the existing per-iteration checkpoint.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .exporters import (
    HttpTransport,
    default_http_transport,
    otlp_log_record,
    otlp_resource_logs,
)

__all__ = [
    "GENESIS_HASH",
    "DecisionLedger",
    "DecisionLedgerConfig",
    "DecisionLedgerSink",
    "DecisionLedgerStore",
    "DecisionRecord",
    "HttpDecisionLedgerSink",
    "InMemoryDecisionLedger",
    "JSONLDecisionLedger",
    "OTelDecisionLedgerSink",
    "TamperDetected",
    "export_decisions",
    "verify_chain",
]

#: The hash a chain's first entry declares as its predecessor. Recognisable
#: on sight as "nothing came before this," the same convention block-based
#: ledgers use for a genesis entry.
GENESIS_HASH = "0" * 64


# ---------------------------------------------------------------------------
# The record and its hash chain
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionRecord:
    """One immutable entry. Construct these only via ``DecisionLedger.record``
    -- it is what maintains correct, monotonic ``seq``/``prev_hash`` state;
    building one by hand produces a record that will fail ``verify_chain``.
    """

    seq: int
    run_id: str
    session_id: str
    iteration: int
    ts: float
    goal: str
    #: The ANALYZE-phase plan in effect for this iteration, or ``None`` for
    #: the run-end record (see this module's docstring).
    plan: dict[str, Any] | None
    #: The model's stated reason for the tool it committed to -- carried
    #: straight from ``Plan.next_action.rationale``, never inferred.
    rationale: str
    tool: str
    tool_calls: list[dict[str, Any]]
    #: ``Gateway.decisions`` entries produced while this iteration's calls
    #: were screened -- risk tier, findings, blocked/fallback, approved_by.
    safety_checks: list[dict[str, Any]]
    #: The OBSERVE-phase evaluation -- outcome, evidence, goal_status -- or
    #: ``None`` on the iteration that called `submit` (there is no further
    #: evaluation of a run that just ended) and on the run-end record.
    evaluation: dict[str, Any] | None
    violations: list[dict[str, Any]]
    #: Populated only on the iteration that submitted, and on the guaranteed
    #: run-end record.
    final: dict[str, Any] | None
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "iteration": self.iteration,
            "ts": self.ts,
            "goal": self.goal,
            "plan": self.plan,
            "rationale": self.rationale,
            "tool": self.tool,
            "tool_calls": self.tool_calls,
            "safety_checks": self.safety_checks,
            "evaluation": self.evaluation,
            "violations": self.violations,
            "final": self.final,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DecisionRecord:
        return cls(
            seq=d["seq"],
            run_id=d["run_id"],
            session_id=d.get("session_id", ""),
            iteration=d["iteration"],
            ts=d["ts"],
            goal=d.get("goal", ""),
            plan=d.get("plan"),
            rationale=d.get("rationale", ""),
            tool=d.get("tool", ""),
            tool_calls=d.get("tool_calls", []),
            safety_checks=d.get("safety_checks", []),
            evaluation=d.get("evaluation"),
            violations=d.get("violations", []),
            final=d.get("final"),
            prev_hash=d["prev_hash"],
            entry_hash=d["entry_hash"],
        )


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, default=str).encode("utf-8")


def _compute_hash(prev_hash: str, fields: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(prev_hash.encode("utf-8"))
    h.update(_canonical(fields))
    return h.hexdigest()


class TamperDetected(Exception):
    """Raised by ``verify_chain`` on the first entry that fails to verify."""

    def __init__(self, seq: int, reason: str) -> None:
        super().__init__(f"decision ledger entry seq={seq} failed verification: {reason}")
        self.seq = seq
        self.reason = reason


def verify_chain(records: Iterable[DecisionRecord]) -> None:
    """Recompute the hash chain; raise ``TamperDetected`` on the first entry
    that doesn't match. Deliberately returns nothing rather than a bool --
    silence is the only trustworthy "yes," and a caller who forgets to check
    a boolean return is a common way tamper detection rots into theatre.
    """
    prev = GENESIS_HASH
    for r in records:
        if r.prev_hash != prev:
            raise TamperDetected(
                r.seq,
                f"prev_hash mismatch (expected {prev[:12]}…, got {r.prev_hash[:12]}…)",
            )
        fields = r.to_dict()
        fields.pop("entry_hash")
        recomputed = _compute_hash(r.prev_hash, fields)
        if recomputed != r.entry_hash:
            raise TamperDetected(r.seq, "entry_hash does not match recomputed content hash")
        prev = r.entry_hash


class DecisionLedger:
    """Owns the hash chain for one run. Wraps a ``DecisionLedgerStore`` plus
    ``DecisionLedgerSink``s, the same shape ``TraceLogger`` wraps an
    ``EventBus`` plus sinks -- this is the only thing that should ever
    construct a ``DecisionRecord``.

    Resume-safe: on construction it reads any existing entries for
    ``run_id`` from ``store`` and continues the chain from the last one,
    rather than resetting to genesis -- a resumed run's ledger is one
    continuous chain, not two chains pretending to be one.
    """

    def __init__(
        self,
        run_id: str,
        session_id: str,
        *,
        store: DecisionLedgerStore,
        sinks: Sequence[DecisionLedgerSink] = (),
    ) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self.store = store
        self.sinks = list(sinks)
        existing = store.read(run_id)
        if existing:
            last = existing[-1]
            self._seq = last.seq
            self._prev_hash = last.entry_hash
        else:
            self._seq = 0
            self._prev_hash = GENESIS_HASH

    def record(
        self,
        *,
        iteration: int,
        goal: str,
        plan: dict[str, Any] | None,
        rationale: str,
        tool: str,
        tool_calls: list[dict[str, Any]],
        safety_checks: list[dict[str, Any]],
        evaluation: dict[str, Any] | None,
        violations: list[dict[str, Any]],
        final: dict[str, Any] | None = None,
    ) -> DecisionRecord:
        self._seq += 1
        fields: dict[str, Any] = {
            "seq": self._seq,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "iteration": iteration,
            "ts": time.time(),
            "goal": goal,
            # Deep-copied so a caller mutating their own plan/evaluation dict
            # after this call cannot retroactively change what was hashed.
            "plan": copy.deepcopy(plan),
            "rationale": rationale,
            "tool": tool,
            "tool_calls": copy.deepcopy(tool_calls),
            "safety_checks": copy.deepcopy(safety_checks),
            "evaluation": copy.deepcopy(evaluation),
            "violations": copy.deepcopy(violations),
            "final": copy.deepcopy(final),
            "prev_hash": self._prev_hash,
        }
        entry_hash = _compute_hash(self._prev_hash, fields)
        record = DecisionRecord(entry_hash=entry_hash, **fields)
        self._prev_hash = entry_hash

        self.store.append(record)
        for sink in self.sinks:
            # A failing sink must never break a run -- same rule as EventBus.
            with contextlib.suppress(Exception):
                sink(record)
        return record


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------


class DecisionLedgerStore(Protocol):
    def append(self, record: DecisionRecord) -> None: ...
    def read(self, run_id: str) -> list[DecisionRecord]: ...


class InMemoryDecisionLedger:
    """For tests. Not append-only in any way that survives the process."""

    def __init__(self) -> None:
        self._records: list[DecisionRecord] = []

    def append(self, record: DecisionRecord) -> None:
        self._records.append(record)

    def read(self, run_id: str) -> list[DecisionRecord]:
        return [r for r in self._records if r.run_id == run_id]


class JSONLDecisionLedger:
    """Hash-chained JSONL file -- the reference "structured append-only
    registry." One run's entries may share a file with other runs' (each
    line carries its own ``run_id``); ``read`` filters by it.

    Append-only in the sense the OS gives ``mode="a"``: nothing here stops a
    process with filesystem access from rewriting the file. The hash chain
    is what turns "nothing stops it" into "it's detectable if it happens" --
    see this module's docstring.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: DecisionRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), default=str) + "\n")
            fh.flush()

    def read(self, run_id: str) -> list[DecisionRecord]:
        if not self.path.exists():
            return []
        out: list[DecisionRecord] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("run_id") == run_id:
                    out.append(DecisionRecord.from_dict(d))
        return out

    def verify(self, run_id: str) -> None:
        """Convenience: read this run's entries back and verify the chain."""
        verify_chain(self.read(run_id))


# ---------------------------------------------------------------------------
# Streaming to external observability systems
# ---------------------------------------------------------------------------


class DecisionLedgerSink(Protocol):
    def __call__(self, record: DecisionRecord) -> None: ...


class HttpDecisionLedgerSink:
    """Streams each record as a JSON POST to any HTTP ingestion endpoint.

    Splunk's HTTP Event Collector, Datadog's logs intake, New Relic's Log
    API, and Dynatrace's generic ingest API all accept exactly this shape --
    arbitrary structured JSON, one event per request, authenticated by a
    header. Point ``url``/``headers`` at whichever one your deployment uses
    (the API key/token goes in ``headers``, per that vendor's docs). If your
    account needs a specific envelope around the payload rather than the raw
    record, inject your own ``transport``.

    ``transport`` is injectable so this is testable without a network.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        self.url = url
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.transport = transport or default_http_transport

    def __call__(self, record: DecisionRecord) -> None:
        self.transport(self.url, record.to_dict(), self.headers)


class OTelDecisionLedgerSink:
    """Streams each record as an OpenTelemetry log record over OTLP/HTTP.

    Hand-built against OTLP's documented JSON wire format, not the
    ``opentelemetry-sdk`` package -- so there is no new dependency, and no
    exposure to that SDK's Logs API, which has moved more across versions
    than Traces/Metrics have. Point ``endpoint`` at an OTel Collector's
    ``/v1/logs`` route, or directly at any backend with native OTLP
    ingestion -- Datadog, New Relic, Dynatrace, and recent Splunk all have
    one today, which makes this one sink a path into all four.

    ``transport`` is injectable so this is testable without a network, same
    seam as ``HttpDecisionLedgerSink``.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        headers: dict[str, str] | None = None,
        service_name: str = "governed",
        transport: HttpTransport | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.service_name = service_name
        self.transport = transport or default_http_transport

    def __call__(self, record: DecisionRecord) -> None:
        log_record = otlp_log_record(
            ts=record.ts,
            body=json.dumps(record.to_dict(), default=str),
            attributes={
                "governed.run_id": record.run_id,
                "governed.iteration": record.iteration,
                "governed.tool": record.tool,
                "governed.entry_hash": record.entry_hash,
                "governed.prev_hash": record.prev_hash,
            },
        )
        body = otlp_resource_logs(
            service_name=self.service_name,
            scope_name="governed.decision_ledger",
            log_records=[log_record],
        )
        self.transport(f"{self.endpoint}/v1/logs", body, self.headers)


def export_decisions(records: Iterable[DecisionRecord], sink: DecisionLedgerSink) -> int:
    """Replay historical records through a sink -- e.g. backfilling a
    store's history into an observability system configured after the run
    already happened. Returns the count sent."""
    n = 0
    for r in records:
        sink(r)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DecisionLedgerConfig:
    """Configures the decision ledger. Disabled by default -- this is
    additive instrumentation, not a replacement for ``trace_path``, and nothing
    about a deployment's behaviour changes by leaving it off.
    """

    enabled: bool = False
    #: Storage backend. ``Agent`` defaults this to a `JSONLDecisionLedger`
    #: under the workspace if ``enabled`` and no store is given.
    store: DecisionLedgerStore | None = None
    #: Streamed to every sink, in addition to being persisted to ``store``.
    sinks: list[DecisionLedgerSink] = field(default_factory=list)

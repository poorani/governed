from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from governed import (
    Agent,
    AgentConfig,
    Budget,
    InMemoryStore,
    LLMResponse,
    TelemetryCollector,
)
from governed.llm import ScriptedClient, ToolCall, Usage
from governed.observability.events import Event, EventType


def _plan(step: str, tool: str, done: list[str]) -> LLMResponse:
    return LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "g",
                "steps": [
                    {"id": "s1", "description": "act", "done": "s1" in done},
                    {"id": "s2", "description": "submit", "done": "s2" in done},
                ],
                "next_action": {
                    "step_id": step,
                    "tool": tool,
                    "rationale": "why",
                    "success_criteria": "checkable",
                },
            }
        )
        + "</plan>",
        usage=Usage(500, 100),
    )


def _eval(done: list[str]) -> LLMResponse:
    return LLMResponse(
        text="<evaluation>"
        + json.dumps(
            {
                "outcome": "success",
                "evidence": "the tool returned successfully",
                "completed_step_ids": done,
                "goal_status": "complete",
                "next_step": "submit",
            }
        )
        + "</evaluation>",
        usage=Usage(400, 80),
    )


def _run(tmp_path: Path, telemetry: TelemetryCollector) -> None:
    submit_args = {"answer": "done", "status": "complete", "confidence": 0.9}
    write_args = {"operation": "write", "path": "a.txt", "content": "x"}
    script = [
        _plan("s1", "file_system", []),
        LLMResponse(
            tool_calls=[ToolCall("c1", "file_system", write_args)],
            usage=Usage(300, 30),
        ),
        _eval(["s1"]),
        _plan("s2", "submit", ["s1"]),
        LLMResponse(tool_calls=[ToolCall("c2", "submit", submit_args)], usage=Usage(200, 20)),
    ]
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(script, model="claude-sonnet-5"),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
            console=False,
            subscribers=[telemetry],
        )
    )
    result = agent.run("write a.txt")
    assert result.ok


# -- LLM instrumentation -----------------------------------------------------


def test_llm_calls_are_counted_with_latency_and_tokens(tmp_path: Path) -> None:
    telemetry = TelemetryCollector()
    _run(tmp_path, telemetry)

    # 5 scripted completions: analyze, act, observe, analyze, act (submit is terminal).
    assert telemetry.llm.count == 5
    assert telemetry.llm.error_count == 0
    assert telemetry.llm.success_rate == 1.0
    assert telemetry.llm.input_tokens == 500 + 300 + 400 + 500 + 200
    assert telemetry.llm.status_codes == {"ok": 5}
    assert telemetry.llm.max_latency_ms >= 0


def test_llm_calls_are_broken_out_by_phase(tmp_path: Path) -> None:
    telemetry = TelemetryCollector()
    _run(tmp_path, telemetry)

    assert set(telemetry.llm_by_phase) == {"analyze", "act", "observe"}
    assert telemetry.llm_by_phase["analyze"].count == 2
    assert telemetry.llm_by_phase["act"].count == 2
    assert telemetry.llm_by_phase["observe"].count == 1


def test_llm_error_is_recorded_with_a_status_key() -> None:
    telemetry = TelemetryCollector()
    telemetry(
        Event(
            type=EventType.LLM_RESPONSE,
            run_id="r1",
            phase="analyze",
            data={"latency_ms": 42, "status": "error", "error_type": "RateLimitError"},
        )
    )
    assert telemetry.llm.count == 1
    assert telemetry.llm.error_count == 1
    assert telemetry.llm.success_rate == 0.0
    assert telemetry.llm.status_codes == {"error:RateLimitError": 1}


def test_llm_http_status_code_wins_over_exception_type() -> None:
    telemetry = TelemetryCollector()
    telemetry(
        Event(
            type=EventType.LLM_RESPONSE,
            run_id="r1",
            phase="act",
            data={
                "latency_ms": 10,
                "status": "error",
                "error_type": "APIStatusError",
                "status_code": 529,
            },
        )
    )
    assert telemetry.llm.status_codes == {"http_529": 1}


# -- tool instrumentation -----------------------------------------------


def test_tool_calls_are_counted_with_latency_and_success_rate(tmp_path: Path) -> None:
    telemetry = TelemetryCollector()
    _run(tmp_path, telemetry)

    assert "file_system" in telemetry.tools
    fs = telemetry.tools["file_system"]
    assert fs.count == 1
    assert fs.error_count == 0
    assert fs.success_rate == 1.0

    assert "submit" in telemetry.tools
    assert telemetry.tools["submit"].count == 1


def test_tool_failure_is_recorded_by_error_code() -> None:
    telemetry = TelemetryCollector()
    failure = {"tool": "http_get", "ok": False, "error_code": "timeout", "duration_ms": 5000}
    telemetry(Event(type=EventType.TOOL_RESULT, run_id="r1", data=failure))
    telemetry(
        Event(
            type=EventType.TOOL_RESULT,
            run_id="r1",
            data={"tool": "http_get", "ok": True, "duration_ms": 120},
        )
    )
    stats = telemetry.tools["http_get"]
    assert stats.count == 2
    assert stats.error_count == 1
    assert stats.success_rate == 0.5
    assert stats.error_codes == {"timeout": 1}
    assert stats.max_latency_ms == 5000


# -- session timing: active vs. HITL idle -----------------------------------


def test_session_timing_tracks_total_and_active_time(tmp_path: Path) -> None:
    telemetry = TelemetryCollector()
    _run(tmp_path, telemetry)

    timing = telemetry.session
    assert timing.total_s >= 0.0
    # The file_system write is DANGEROUS-tier, so it still passes through the
    # APPROVAL_REQUESTED/DECIDED audit pair -- auto-granted instantly under the
    # default policy, so idle time is on the order of microseconds, not zero.
    assert timing.idle_wait_s < 0.05
    assert timing.active_s == pytest.approx(timing.total_s, abs=0.05)


def test_hitl_wait_is_separated_from_active_time() -> None:
    telemetry = TelemetryCollector()
    run_id = "r1"

    telemetry(Event(type=EventType.RUN_START, run_id=run_id, ts=1000.0))
    telemetry(Event(type=EventType.APPROVAL_REQUESTED, run_id=run_id, ts=1001.0))
    # Simulates a human taking 30s to approve a DANGER-tier call.
    telemetry(Event(type=EventType.APPROVAL_DECIDED, run_id=run_id, ts=1031.0))
    telemetry(Event(type=EventType.RUN_END, run_id=run_id, ts=1040.0))

    timing = telemetry.sessions[run_id]
    assert timing.total_s == 40.0
    assert timing.idle_wait_s == 30.0
    assert timing.active_s == 10.0


def test_multiple_approvals_in_one_run_accumulate_idle_time() -> None:
    telemetry = TelemetryCollector()
    run_id = "r1"

    telemetry(Event(type=EventType.RUN_START, run_id=run_id, ts=0.0))
    telemetry(Event(type=EventType.APPROVAL_REQUESTED, run_id=run_id, ts=1.0))
    telemetry(Event(type=EventType.APPROVAL_DECIDED, run_id=run_id, ts=6.0))  # 5s
    telemetry(Event(type=EventType.APPROVAL_REQUESTED, run_id=run_id, ts=10.0))
    telemetry(Event(type=EventType.APPROVAL_DECIDED, run_id=run_id, ts=17.0))  # 7s
    telemetry(Event(type=EventType.RUN_END, run_id=run_id, ts=20.0))

    assert telemetry.sessions[run_id].idle_wait_s == 12.0


def test_unmatched_approval_decided_is_ignored_not_a_crash() -> None:
    telemetry = TelemetryCollector()
    telemetry(Event(type=EventType.APPROVAL_DECIDED, run_id="orphan", ts=time.time()))
    assert telemetry.sessions == {} or telemetry.sessions["orphan"].idle_wait_s == 0.0


# -- cost -----------------------------------------------------------------


def test_cost_is_read_off_cost_recorded_events(tmp_path: Path) -> None:
    telemetry = TelemetryCollector()
    _run(tmp_path, telemetry)

    # claude-sonnet-5 has a real rate card; five priced completions cost > $0.
    assert telemetry.session.cost_usd > 0.0


def test_cost_tracks_the_ledgers_running_total_not_a_sum() -> None:
    telemetry = TelemetryCollector()
    run_id = "r1"
    telemetry(Event(type=EventType.RUN_START, run_id=run_id, ts=0.0))
    telemetry(
        Event(
            type=EventType.COST_RECORDED,
            run_id=run_id,
            phase="analyze",
            data={"run_usd": 0.01},
        )
    )
    telemetry(
        Event(type=EventType.COST_RECORDED, run_id=run_id, phase="act", data={"run_usd": 0.03})
    )
    # The second event's run_usd is already the cumulative total from the
    # ledger -- summing the two would double count.
    assert telemetry.sessions[run_id].cost_usd == 0.03


# -- safety -----------------------------------------------------------------


def test_guardrail_blocked_is_counted_by_tool() -> None:
    telemetry = TelemetryCollector()
    telemetry(
        Event(type=EventType.GUARDRAIL_BLOCKED, run_id="r1", data={"tool": "execute_code"})
    )
    telemetry(
        Event(type=EventType.GUARDRAIL_BLOCKED, run_id="r1", data={"tool": "execute_code"})
    )
    telemetry(
        Event(type=EventType.GUARDRAIL_BLOCKED, run_id="r1", data={"tool": "file_system"})
    )

    assert telemetry.safety.blocked_calls == 3
    assert telemetry.safety.blocked_by_tool == {"execute_code": 2, "file_system": 1}


def test_circuit_open_is_counted_by_reason() -> None:
    telemetry = TelemetryCollector()
    telemetry(Event(type=EventType.CIRCUIT_OPEN, run_id="r1", data={"reason": "cost_ceiling"}))
    telemetry(Event(type=EventType.CIRCUIT_OPEN, run_id="r2", data={"reason": "cost_ceiling"}))
    telemetry(Event(type=EventType.CIRCUIT_OPEN, run_id="r3", data={"reason": "stalled"}))

    assert telemetry.safety.circuit_trips == 3
    assert telemetry.safety.circuit_trip_reasons == {"cost_ceiling": 2, "stalled": 1}


def test_budget_exceeded_is_counted_by_resource() -> None:
    telemetry = TelemetryCollector()
    telemetry(
        Event(type=EventType.BUDGET_EXCEEDED, run_id="r1", data={"which": "max_iterations=20"})
    )
    assert telemetry.safety.budget_exhaustions == 1
    assert telemetry.safety.budget_exhaustions_by_resource == {"max_iterations=20": 1}


def test_safety_stats_are_cumulative_across_runs_not_per_session() -> None:
    telemetry = TelemetryCollector()
    telemetry(
        Event(type=EventType.GUARDRAIL_BLOCKED, run_id="r1", data={"tool": "execute_code"})
    )
    telemetry(
        Event(type=EventType.GUARDRAIL_BLOCKED, run_id="r2", data={"tool": "execute_code"})
    )
    assert telemetry.safety.blocked_calls == 2


# -- reporting ------------------------------------------------------------


def test_summary_and_to_dict_do_not_raise(tmp_path: Path) -> None:
    telemetry = TelemetryCollector()
    _run(tmp_path, telemetry)

    text = telemetry.summary()
    assert "LLM" in text
    assert "file_system" in text


def test_overview_is_a_flat_dashboard_shaped_dict(tmp_path: Path) -> None:
    telemetry = TelemetryCollector()
    _run(tmp_path, telemetry)

    overview = telemetry.overview()
    assert overview["llm_requests"] == 5
    assert overview["total_tokens"] == telemetry.llm.total_tokens
    assert overview["total_cost_usd"] > 0.0
    assert overview["tool_calls"] == 2  # file_system write + submit
    assert overview["tool_success_rate"] == 1.0
    assert overview["blocked_calls"] == 0
    assert overview["total_run_time_s"] >= 0.0

    payload = telemetry.to_dict()
    assert payload["llm"]["count"] == 5
    assert "sessions" in payload

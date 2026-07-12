"""The decision ledger: hash-chained, tamper-evident records of every
iteration's decision path, plus its storage backends, streaming sinks, and
wiring into a real Agent run.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from governed import (
    GENESIS_HASH,
    Agent,
    AgentConfig,
    AllowTierApprover,
    Budget,
    DecisionLedger,
    DecisionLedgerConfig,
    GuardrailConfig,
    HttpDecisionLedgerSink,
    InMemoryDecisionLedger,
    InMemoryStore,
    JSONLDecisionLedger,
    LLMResponse,
    OTelDecisionLedgerSink,
    RiskTier,
    TamperDetected,
    export_decisions,
    verify_chain,
)
from governed.llm import ScriptedClient, ToolCall, Usage
from governed.tools import FileSystemTool, SubmitTool

# ---------------------------------------------------------------------------
# DecisionLedger + hash chain
# ---------------------------------------------------------------------------


def _record_three(store: InMemoryDecisionLedger, run_id: str = "run-1") -> DecisionLedger:
    ledger = DecisionLedger(run_id, "sess-1", store=store)
    for i in range(3):
        ledger.record(
            iteration=i + 1,
            goal="do the thing",
            plan={"next_action": {"tool": "file_system"}},
            rationale=f"step {i + 1}",
            tool="file_system",
            tool_calls=[{"tool": "file_system", "ok": True}],
            safety_checks=[],
            evaluation={"outcome": "success"},
            violations=[],
        )
    return ledger


def test_first_entry_chains_to_genesis() -> None:
    store = InMemoryDecisionLedger()
    _record_three(store)
    records = store.read("run-1")
    assert records[0].prev_hash == GENESIS_HASH
    assert records[0].seq == 1


def test_entries_chain_to_the_previous_hash() -> None:
    store = InMemoryDecisionLedger()
    _record_three(store)
    records = store.read("run-1")
    assert records[1].prev_hash == records[0].entry_hash
    assert records[2].prev_hash == records[1].entry_hash
    assert records[0].entry_hash != records[1].entry_hash  # distinct content


def test_verify_chain_passes_for_untampered_records() -> None:
    store = InMemoryDecisionLedger()
    _record_three(store)
    verify_chain(store.read("run-1"))  # does not raise


def test_verify_chain_detects_altered_content() -> None:
    store = InMemoryDecisionLedger()
    _record_three(store)
    records = store.read("run-1")
    tampered = replace(records[0], rationale="a different rationale, written after the fact")
    with pytest.raises(TamperDetected) as exc:
        verify_chain([tampered, records[1], records[2]])
    assert exc.value.seq == 1


def test_verify_chain_detects_a_deleted_entry() -> None:
    store = InMemoryDecisionLedger()
    _record_three(store)
    records = store.read("run-1")
    with pytest.raises(TamperDetected):
        verify_chain([records[0], records[2]])  # records[1] removed


def test_verify_chain_detects_reordering() -> None:
    store = InMemoryDecisionLedger()
    _record_three(store)
    records = store.read("run-1")
    with pytest.raises(TamperDetected):
        verify_chain([records[1], records[0], records[2]])


def test_record_deep_copies_inputs_against_later_caller_mutation() -> None:
    store = InMemoryDecisionLedger()
    ledger = DecisionLedger("run-1", "sess-1", store=store)
    plan = {"next_action": {"tool": "file_system"}}
    record = ledger.record(
        iteration=1,
        goal="g",
        plan=plan,
        rationale="r",
        tool="file_system",
        tool_calls=[],
        safety_checks=[],
        evaluation=None,
        violations=[],
    )
    plan["next_action"]["tool"] = "mutated after the fact"
    assert record.plan is not None
    assert record.plan["next_action"]["tool"] == "file_system"


def test_resuming_a_run_continues_the_same_chain_not_a_new_one() -> None:
    store = InMemoryDecisionLedger()
    first = DecisionLedger("run-1", "sess-1", store=store)
    first.record(
        iteration=1,
        goal="g",
        plan=None,
        rationale="",
        tool="file_system",
        tool_calls=[],
        safety_checks=[],
        evaluation=None,
        violations=[],
    )
    # A fresh DecisionLedger for the same run_id (what _drive does on resume).
    second = DecisionLedger("run-1", "sess-1", store=store)
    second.record(
        iteration=2,
        goal="g",
        plan=None,
        rationale="",
        tool="submit",
        tool_calls=[],
        safety_checks=[],
        evaluation=None,
        violations=[],
    )
    records = store.read("run-1")
    assert len(records) == 2
    assert records[1].seq == 2
    assert records[1].prev_hash == records[0].entry_hash
    verify_chain(records)  # one continuous, valid chain


# ---------------------------------------------------------------------------
# JSONLDecisionLedger
# ---------------------------------------------------------------------------


def test_jsonl_ledger_round_trips(tmp_path: Path) -> None:
    store = JSONLDecisionLedger(tmp_path / "ledger.jsonl")
    _record_three(store)
    records = store.read("run-1")
    assert len(records) == 3
    assert [r.rationale for r in records] == ["step 1", "step 2", "step 3"]
    store.verify("run-1")  # does not raise


def test_jsonl_ledger_filters_by_run_id(tmp_path: Path) -> None:
    store = JSONLDecisionLedger(tmp_path / "ledger.jsonl")
    _record_three(store, run_id="run-a")
    _record_three(store, run_id="run-b")
    assert len(store.read("run-a")) == 3
    assert len(store.read("run-b")) == 3
    assert {r.run_id for r in store.read("run-a")} == {"run-a"}


def test_jsonl_ledger_detects_tampering_written_to_disk(tmp_path: Path) -> None:
    """The actual promise: edit one line in the file after the fact, and
    verification catches it -- not just in-memory objects."""
    path = tmp_path / "ledger.jsonl"
    store = JSONLDecisionLedger(path)
    _record_three(store)

    lines = path.read_text().splitlines()
    first = json.loads(lines[0])
    first["rationale"] = "quietly edited after the run"
    lines[0] = json.dumps(first)
    path.write_text("\n".join(lines) + "\n")

    with pytest.raises(TamperDetected):
        store.verify("run-1")


# ---------------------------------------------------------------------------
# Streaming sinks
# ---------------------------------------------------------------------------


def test_http_sink_posts_the_record_as_json() -> None:
    sent = []

    def transport(url, payload, headers):
        sent.append((url, payload, headers))

    sink = HttpDecisionLedgerSink(
        "https://logs.example/ingest", headers={"X-Api-Key": "k"}, transport=transport
    )
    store = InMemoryDecisionLedger()
    ledger = DecisionLedger("run-1", "sess-1", store=store, sinks=[sink])
    record = ledger.record(
        iteration=1,
        goal="g",
        plan=None,
        rationale="r",
        tool="submit",
        tool_calls=[],
        safety_checks=[],
        evaluation=None,
        violations=[],
    )

    assert len(sent) == 1
    url, payload, headers = sent[0]
    assert url == "https://logs.example/ingest"
    assert payload == record.to_dict()
    assert headers["X-Api-Key"] == "k"


def test_otel_sink_posts_a_valid_otlp_logs_shape() -> None:
    sent = []
    sink = OTelDecisionLedgerSink(
        "https://collector.example:4318", transport=lambda u, p, h: sent.append((u, p, h))
    )
    store = InMemoryDecisionLedger()
    ledger = DecisionLedger("run-1", "sess-1", store=store, sinks=[sink])
    ledger.record(
        iteration=1,
        goal="g",
        plan=None,
        rationale="r",
        tool="submit",
        tool_calls=[],
        safety_checks=[],
        evaluation=None,
        violations=[],
    )

    url, body, _ = sent[0]
    assert url == "https://collector.example:4318/v1/logs"
    log_record = body["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    assert "stringValue" in log_record["body"]
    attr_keys = {a["key"] for a in log_record["attributes"]}
    assert "governed.run_id" in attr_keys
    assert "governed.entry_hash" in attr_keys


def test_a_failing_sink_does_not_break_recording() -> None:
    def broken_sink(record):
        raise RuntimeError("downstream is down")

    store = InMemoryDecisionLedger()
    ledger = DecisionLedger("run-1", "sess-1", store=store, sinks=[broken_sink])
    record = ledger.record(
        iteration=1,
        goal="g",
        plan=None,
        rationale="r",
        tool="submit",
        tool_calls=[],
        safety_checks=[],
        evaluation=None,
        violations=[],
    )
    assert store.read("run-1") == [record]  # persisted despite the sink failing


def test_export_decisions_replays_history_through_a_sink() -> None:
    store = InMemoryDecisionLedger()
    _record_three(store)
    received = []
    n = export_decisions(store.read("run-1"), received.append)
    assert n == 3
    assert len(received) == 3


# ---------------------------------------------------------------------------
# End to end through a real Agent run
# ---------------------------------------------------------------------------


def _plan(step: str, tool: str, why: str, done: list[str]) -> LLMResponse:
    return LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "write a file, then report",
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


def _script() -> list[LLMResponse]:
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
        _eval("success", "wrote the file successfully", "complete", "submit", ["s1"]),
        _plan("s2", "submit", "report the outcome", ["s1"]),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    "c2",
                    "submit",
                    {
                        "answer": "wrote a.txt",
                        "status": "complete",
                        "confidence": 0.9,
                        "evidence": ["wrote the file successfully"],
                        "unmet_requirements": [],
                    },
                )
            ],
            usage=Usage(200, 20),
        ),
    ]


def test_decision_ledger_is_disabled_by_default(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(_script()),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
            tools=[FileSystemTool(), SubmitTool()],
        )
    )
    agent.run("write hi to a.txt")
    assert agent.decision_ledger is None


def test_agent_run_writes_one_record_per_iteration_plus_a_run_end_record(
    tmp_path: Path,
) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    store = InMemoryDecisionLedger()
    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(_script()),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
            tools=[FileSystemTool(), SubmitTool()],
            decision_ledger=DecisionLedgerConfig(enabled=True, store=store),
        )
    )
    result = agent.run("write hi to a.txt")

    records = store.read(result.state.run_id)
    # iteration 1 (file_system, evaluated), iteration 2 (submit), run-end.
    assert len(records) == 3
    assert records[0].tool == "file_system"
    assert records[0].rationale == "write the file"
    assert records[0].evaluation is not None
    assert records[0].evaluation["outcome"] == "success"
    assert records[1].tool == "submit"
    assert records[1].final is not None
    assert records[1].final["status"] == "complete"
    assert records[2].tool == "__run_end__"
    assert records[2].final is not None
    assert records[2].final["answer"] == "wrote a.txt"
    verify_chain(records)


def test_decision_ledger_captures_safety_checks(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    ws.mkdir()
    store = InMemoryDecisionLedger()
    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(_script()),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
            tools=[FileSystemTool(), SubmitTool()],
            guardrails=GuardrailConfig(approver=AllowTierApprover(RiskTier.WARNING)),
            decision_ledger=DecisionLedgerConfig(enabled=True, store=store),
        )
    )
    result = agent.run("write hi to a.txt")

    records = store.read(result.state.run_id)
    write_record = records[0]
    assert write_record.safety_checks  # the file_system write was screened
    assert write_record.safety_checks[0]["tier"] == "WARNING"


def test_decision_ledger_default_store_protects_its_own_file(tmp_path: Path) -> None:
    """The ledger's own JSONL file is registered as framework-owned, the
    same way trace_path is -- the agent cannot write over its own audit log."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    ledger_path = ws / ".governed" / "decisions" / "ledger.jsonl"

    script = [
        _plan("s1", "file_system", "overwrite the ledger", []),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    "c1",
                    "file_system",
                    {
                        "operation": "write",
                        "path": str(ledger_path),
                        "content": "nothing to see here",
                    },
                )
            ],
            usage=Usage(300, 30),
        ),
        _eval("failure", "the write was refused", "in_progress", "report", []),
        _plan("s2", "submit", "report the outcome", []),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    "c2",
                    "submit",
                    {
                        "answer": "could not overwrite the ledger",
                        "status": "blocked",
                        "confidence": 0.5,
                        "evidence": ["refused"],
                        "unmet_requirements": [],
                    },
                )
            ],
            usage=Usage(200, 20),
        ),
    ]

    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(script),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
            tools=[FileSystemTool(), SubmitTool()],
            guardrails=GuardrailConfig(approver=AllowTierApprover(RiskTier.DANGER)),
            decision_ledger=DecisionLedgerConfig(enabled=True),
        )
    )
    result = agent.run("overwrite the ledger file")

    call = result.state.iterations[0].tool_calls[0]
    assert not call.ok
    assert "framework-owned" in call.result_preview

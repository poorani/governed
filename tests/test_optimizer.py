from __future__ import annotations

import json

import pytest

from governed import (
    Agent,
    AgentConfig,
    Budget,
    CircuitBreakerConfig,
    CostConfig,
    InMemoryStore,
    LLMResponse,
    ScriptedClient,
)
from governed.llm import ToolCall, Usage
from governed.memory import CompactionConfig
from governed.memory.optimizer import (
    PRICING,
    CircuitBreaker,
    CircuitOpen,
    CostLedger,
    ModelPricing,
    RecursiveCompactor,
    compaction_for,
    resolve_pricing,
)

# -- pricing ---------------------------------------------------------------


def test_versioned_model_ids_resolve_by_longest_prefix():
    assert resolve_pricing("claude-sonnet-4-6-20260219") is PRICING["claude-sonnet-4-6"]
    assert resolve_pricing("claude-opus-4-8") is PRICING["claude-opus-4-8"]


def test_unknown_model_resolves_to_none_rather_than_guessing():
    assert resolve_pricing("some-finetune-v3") is None


def test_overrides_win():
    mine = ModelPricing(1.0, 2.0)
    assert resolve_pricing("some-finetune-v3", {"some-finetune": mine}) is mine


def test_input_and_output_are_priced_separately():
    p = PRICING["claude-opus-4-8"]  # $5 / $25
    assert p.cost(1_000_000, 0) == pytest.approx(5.0)
    assert p.cost(0, 1_000_000) == pytest.approx(25.0)
    assert p.cost(1_000_000, 200_000) == pytest.approx(10.0)


def test_batch_halves_and_cache_reads_are_a_tenth():
    p = PRICING["claude-opus-4-8"]
    assert p.cost(1_000_000, 0, batch=True) == pytest.approx(2.5)
    assert p.cost(0, 0, cache_read_tokens=1_000_000) == pytest.approx(0.5)


def test_compaction_window_follows_the_model():
    assert compaction_for("claude-opus-4-8").context_window_tokens == 1_000_000
    assert compaction_for("claude-haiku-4-5").context_window_tokens == 200_000
    assert compaction_for("claude-opus-4-8").trigger_tokens == 750_000  # 75%
    assert compaction_for("unknown-model").context_window_tokens == 128_000


# -- ledger ----------------------------------------------------------------


def test_ledger_attributes_spend_to_the_phase():
    led = CostLedger()
    led.record("claude-sonnet-4-6", Usage(1_000_000, 0), phase="analyze")  # $3
    led.record("claude-sonnet-4-6", Usage(0, 1_000_000), phase="observe")  # $15
    assert led.total_usd == pytest.approx(18.0)
    assert led.by_phase() == {"observe": pytest.approx(15.0), "analyze": pytest.approx(3.0)}


def test_unpriced_model_is_zero_and_says_so_once():
    warned: list[str] = []
    led = CostLedger()
    led.on_unpriced = warned.append
    led.record("mystery", Usage(9_999, 9_999))
    led.record("mystery", Usage(9_999, 9_999))

    assert led.total_usd == 0.0
    assert warned == ["mystery"]  # warned, but only once
    assert led.unpriced_models == {"mystery"}
    assert "$0" in led.summary()


def test_seed_restores_spend_so_resume_cannot_reset_the_meter():
    led = CostLedger()
    led.seed(1.75)
    assert led.total_usd == 1.75
    led.seed(0.10)  # a lower seed never lowers the total
    assert led.total_usd == 1.75


def test_disabled_ledger_prices_nothing():
    led = CostLedger(CostConfig(enabled=False))
    led.record("claude-opus-4-8", Usage(1_000_000, 1_000_000))
    assert led.total_usd == 0.0


# -- circuit breaker -------------------------------------------------------


def test_cost_ceiling_trips_and_is_exhaustion_not_failure():
    led = CostLedger()
    br = CircuitBreaker(CircuitBreakerConfig(max_usd=2.00), led)
    led.record("claude-opus-4-8", Usage(100_000, 0))  # $0.50
    br.check_cost()  # under, fine
    led.record("claude-opus-4-8", Usage(400_000, 0))  # $2.50 total

    with pytest.raises(CircuitOpen) as e:
        br.check_cost()
    assert e.value.reason == "cost_ceiling"
    assert e.value.terminal_status == "exhausted"  # try again with more money


def test_cost_warning_fires_once_at_the_ratio():
    warnings = []
    led = CostLedger()
    br = CircuitBreaker(CircuitBreakerConfig(max_usd=1.00, warn_at_ratio=0.75), led)
    br.on_warn = lambda r, d: warnings.append(r)

    led.record("claude-opus-4-8", Usage(160_000, 0))  # $0.80
    br.check_cost()
    br.check_cost()
    assert warnings == ["cost_warning"]


def test_no_ceiling_means_no_trip():
    br = CircuitBreaker(CircuitBreakerConfig(max_usd=None), CostLedger())
    br.check_cost()


def test_identical_tool_calls_trip_the_loop_detector():
    br = CircuitBreaker(CircuitBreakerConfig(max_identical_tool_calls=3))
    args = {"operation": "read", "path": "a"}
    br.observe_tool_call("file_system", args)
    br.observe_tool_call("file_system", args)
    br.observe_tool_call("file_system", {"operation": "read", "path": "b"})  # different: fine

    reordered = dict(
        reversed(list(args.items()))
    )  # key order is irrelevant to the fingerprint
    with pytest.raises(CircuitOpen, match="repeated_tool_call"):
        br.observe_tool_call("file_system", reordered)


class _Action:
    def __init__(self, step="s1", tool="file_system"):
        self.step_id, self.tool = step, tool


class _Plan:
    def __init__(self, step="s1", tool="file_system"):
        self.next_action = _Action(step, tool)


class _Eval:
    def __init__(self, completed=(), outcome="partial"):
        self.completed_step_ids, self.outcome = list(completed), outcome


def test_stall_is_repetition_without_progress():
    br = CircuitBreaker(CircuitBreakerConfig(max_stalled_iterations=2))
    br.observe_iteration(_Plan(), _Eval())  # first: nothing to repeat
    br.observe_iteration(_Plan(), _Eval())  # repeat 1
    with pytest.raises(CircuitOpen, match="stalled"):
        br.observe_iteration(_Plan(), _Eval())  # repeat 2


def test_progress_resets_the_stall_counter():
    br = CircuitBreaker(CircuitBreakerConfig(max_stalled_iterations=3))
    br.observe_iteration(_Plan(), _Eval())
    br.observe_iteration(_Plan(), _Eval())  # repeat 1
    br.observe_iteration(_Plan(), _Eval(completed=["s1"]))  # a step completed: reset
    br.observe_iteration(_Plan(), _Eval())
    br.observe_iteration(_Plan(), _Eval())  # only repeat 1 again


def test_repeating_a_step_after_moving_on_is_not_a_stall():
    br = CircuitBreaker(CircuitBreakerConfig(max_stalled_iterations=1))
    br.observe_iteration(_Plan("s1"), _Eval())
    br.observe_iteration(_Plan("s2"), _Eval())
    br.observe_iteration(_Plan("s1"), _Eval())  # retrying s1 is legitimate


def test_stalling_is_blocked_not_exhausted():
    br = CircuitBreaker(CircuitBreakerConfig(max_stalled_iterations=1))
    br.observe_iteration(_Plan(), _Eval())
    with pytest.raises(CircuitOpen) as e:
        br.observe_iteration(_Plan(), _Eval())
    assert e.value.terminal_status == "blocked"  # a bigger budget will not help


# -- recursive compaction --------------------------------------------------


class _Summarizer:
    """Counts completions and returns a marker so the fold depth is observable."""

    model = "claude-haiku-4-5"

    def __init__(self):
        self.calls = 0
        self.usage_reported = []

    def complete(self, **kw):
        self.calls += 1
        return LLMResponse(text=f"summary-{self.calls}", usage=Usage(100, 50))

    def count_tokens(self, s: str) -> int:
        return len(s) // 4


def _turns(n: int, size: int = 4_000):
    from governed.llm import Message

    out = []
    for i in range(n):
        m = Message(role="user", text="x" * size)
        m.meta["iteration"] = i
        out.append(m)
    return out


def test_small_history_folds_in_one_call():
    llm = _Summarizer()
    c = RecursiveCompactor(llm, CompactionConfig(keep_iterations=1), chunk_tokens=10_000)
    _kept, summary = c.compact(_turns(3), "", current_iteration=3)
    assert llm.calls == 1 and summary == "summary-1"


def test_large_history_chunks_and_then_merges():
    llm = _Summarizer()
    # 10 turns x ~1000 tokens each against a 2500-token chunk budget: several
    # level-1 summaries, then a merge. One call summarises nothing twice.
    c = RecursiveCompactor(
        llm, CompactionConfig(keep_iterations=3), chunk_tokens=2_500, max_depth=1
    )
    kept, _summary = c.compact(_turns(10), "", current_iteration=10)
    assert llm.calls > 1
    assert kept  # the recent window survives verbatim


def test_the_fold_is_metered():
    llm = _Summarizer()
    seen: list[Usage] = []
    c = RecursiveCompactor(llm, CompactionConfig(keep_iterations=1), meter=seen.append)
    c.compact(_turns(3), "", current_iteration=3)
    assert seen and seen[0].output_tokens == 50  # summarisation is not free


def test_compaction_never_orphans_a_tool_call():
    from governed.llm import Message
    from governed.llm import ToolCall as TC

    llm = _Summarizer()
    msgs = _turns(2)
    opener = Message(role="assistant", text="", tool_calls=[TC("c1", "t", {})])
    opener.meta["iteration"] = 1
    msgs.insert(1, opener)

    c = RecursiveCompactor(llm, CompactionConfig(keep_iterations=0))
    kept, _ = c.compact(msgs, "", current_iteration=5)
    # The split may not land between the tool_use and its result.
    assert not (kept and kept[0].tool_calls)


# -- end to end ------------------------------------------------------------


def _plan(tool="file_system", step="s1"):
    return LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "g",
                "steps": [
                    {"id": "s1", "description": "act"},
                    {"id": "s2", "description": "sub"},
                ],
                "next_action": {
                    "step_id": step,
                    "tool": tool,
                    "rationale": "because",
                    "success_criteria": "exists",
                },
            }
        )
        + "</plan>",
        usage=Usage(1_000, 200),
    )


def _write():
    args = {"operation": "write", "path": "a.txt", "content": "abc"}
    return LLMResponse(
        text="",
        tool_calls=[ToolCall("c1", "file_system", args)],
        usage=Usage(1_000, 50),
    )


def _eval(status="in_progress", done=("s1",)):
    return LLMResponse(
        text="<evaluation>"
        + json.dumps(
            {
                "outcome": "success",
                "evidence": "the tool reported Created a.txt (3 bytes, 1 lines).",
                "completed_step_ids": list(done),
                "goal_status": status,
                "next_step": "carry on",
            }
        )
        + "</evaluation>",
        usage=Usage(800, 150),
    )


def _agent(script, tmp_path, **cfg):
    return Agent(
        AgentConfig(
            llm=ScriptedClient(script, model="claude-sonnet-4-6"),
            workspace=tmp_path / "ws",
            skills_dirs=[],
            store=cfg.pop("store", None) or InMemoryStore(),
            console=False,
            budget=Budget(max_iterations=cfg.pop("max_iterations", 8)),
            **cfg,
        )
    )


def test_a_completed_run_reports_its_cost(tmp_path):
    submit_args = {"answer": "done", "status": "complete", "confidence": 0.9}
    submit = LLMResponse(
        text="",
        tool_calls=[ToolCall("s", "submit", submit_args)],
        usage=Usage(500, 100),
    )
    agent = _agent(
        [_plan(), _write(), _eval("complete"), _plan("submit", "s2"), submit], tmp_path
    )
    result = agent.run("write a.txt")

    assert result.ok and result.cost_usd > 0
    assert agent.ledger.total_usd == pytest.approx(result.cost_usd)
    assert set(agent.ledger.by_phase()) == {"analyze", "act", "observe"}


def test_the_dollar_ceiling_terminates_the_run_safely(tmp_path):
    script = []
    for _ in range(6):
        script += [_plan(), _write(), _eval()]

    agent = _agent(
        script,
        tmp_path,
        # One ANALYZE completion costs 1000 in + 200 out on Sonnet = $0.006.
        circuit_breaker=CircuitBreakerConfig(max_usd=0.01),
    )
    result = agent.run("write a.txt forever")

    assert result.status == "exhausted"
    assert "circuit breaker" in result.answer and "cost_ceiling" in result.answer
    assert result.state is not None  # state was checkpointed, not lost
    assert result.cost_usd >= 0.01


def test_cost_survives_resume(tmp_path):
    store = InMemoryStore()
    script = []
    for _ in range(6):
        script += [_plan(), _write(), _eval()]

    agent = _agent(script, tmp_path, store=store, max_iterations=2)
    first = agent.run("write a.txt")
    assert first.status == "exhausted" and first.cost_usd > 0

    # A fresh Agent, a fresh ledger -- and the meter picks up where it stopped.
    agent2 = _agent(script, tmp_path, store=store, max_iterations=4)
    agent2.ledger.seed(store.load(first.session_id).scratchpad["_cost_usd"])
    assert agent2.ledger.total_usd == pytest.approx(first.cost_usd)


def test_submit_is_exempt_from_the_loop_detector(tmp_path):
    """Terminating a run is not a suspicious repetition."""
    submit_args = {"answer": "d", "status": "complete", "confidence": 0.9}
    submit = LLMResponse(
        text="",
        tool_calls=[ToolCall("s", "submit", submit_args)],
        usage=Usage(10, 10),
    )
    agent = _agent(
        [_plan("submit", "s2"), submit],
        tmp_path,
        circuit_breaker=CircuitBreakerConfig(max_identical_tool_calls=1),
    )
    assert agent.run("just submit").ok


def test_repeated_identical_calls_trip_the_breaker_in_a_real_run(tmp_path):
    script = []
    for _ in range(4):
        script += [_plan(), _write(), _eval(done=())]

    agent = _agent(
        script,
        tmp_path,
        circuit_breaker=CircuitBreakerConfig(max_identical_tool_calls=2),
    )
    result = agent.run("write a.txt")
    assert result.status == "blocked"
    assert "repeated_tool_call" in result.answer

"""Agent.cancel(): a cooperative kill switch for a run in progress.

`Agent.run()`/`resume()` are synchronous and blocking, so "cancel a running
agent" only makes sense from another thread holding a reference to the same
`Agent` -- these tests exercise that directly, plus the two checkpoints
`_drive` checks (top of the loop, and after EXECUTE/before OBSERVE) without
needing real threads for the deterministic ones: a scripted `LLMClient`
calls `agent.cancel()` as a side effect of one of its own responses, which
is a faithful stand-in for "another thread cancelled while this call was in
flight" without any timing dependence.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from governed import Agent, AgentConfig, Budget, InMemoryStore, LLMResponse
from governed.llm import ScriptedClient, ToolCall, Usage


def _plan(step: str, tool: str, done: list[str]) -> LLMResponse:
    return LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "do the thing",
                "steps": [{"id": "s1", "description": "do it", "done": "s1" in done}],
                "next_action": {
                    "step_id": step,
                    "tool": tool,
                    "rationale": "test",
                    "success_criteria": "the call returns without error",
                },
            }
        )
        + "</plan>",
        usage=Usage(200, 40),
    )


def _act(tool: str, call_id: str, args: dict[str, object]) -> LLMResponse:
    return LLMResponse(tool_calls=[ToolCall(call_id, tool, args)], usage=Usage(200, 20))


def _eval(status: str, nxt: str, done: list[str]) -> LLMResponse:
    return LLMResponse(
        text="<evaluation>"
        + json.dumps(
            {
                "outcome": "success",
                "evidence": "scratchpad list returned successfully",
                "completed_step_ids": done,
                "goal_status": status,
                "next_step": nxt,
            }
        )
        + "</evaluation>",
        usage=Usage(150, 30),
    )


class _CancelAfterNCalls(ScriptedClient):
    """Calls `agent.cancel()` as a side effect of its Nth response, standing
    in for "another thread cancelled while this LLM call was in flight" --
    same effect as real cross-thread cancellation, deterministic to test."""

    def __init__(self, responses: list[LLMResponse], *, cancel_after: int) -> None:
        super().__init__(responses, model="test-cancel")
        self.cancel_after = cancel_after
        self.agent: Agent | None = None

    def complete(self, **kwargs: object) -> LLMResponse:
        resp = super().complete(**kwargs)  # type: ignore[arg-type]
        if len(self.calls) == self.cancel_after:
            assert self.agent is not None, "attach .agent before calling run()"
            self.agent.cancel("test cancel")
        return resp


def _agent(tmp_path: Path, client: ScriptedClient, *, max_iterations: int = 10) -> Agent:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return Agent(
        AgentConfig(
            llm=client,
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=max_iterations),
        )
    )


def test_cancel_between_execute_and_observe_skips_the_observe_call(tmp_path: Path) -> None:
    # ANALYZE -> ACT -> [cancel fires here, as a side effect of the ACT
    # response] -> EXECUTE (real, fast) -> checkpoint should catch it before
    # OBSERVE ever gets called.
    client = _CancelAfterNCalls(
        [
            _plan("s1", "scratchpad", done=[]),
            _act("scratchpad", "c1", {"action": "list"}),
        ],
        cancel_after=2,
    )
    agent = _agent(tmp_path, client)
    client.agent = agent

    result = agent.run("do the thing")

    assert result.status == "cancelled"
    assert len(client.calls) == 2  # OBSERVE (a 3rd call) never happened


def test_cancel_at_top_of_loop_skips_the_next_iteration(tmp_path: Path) -> None:
    # A full first iteration (ANALYZE, ACT, OBSERVE) that does NOT complete
    # the goal, cancel fires as a side effect of the OBSERVE response, and
    # the top-of-loop checkpoint should stop before ANALYZE #2.
    client = _CancelAfterNCalls(
        [
            _plan("s1", "scratchpad", done=[]),
            _act("scratchpad", "c1", {"action": "list"}),
            _eval("in_progress", "keep going", done=[]),
            _plan("s1", "scratchpad", done=[]),  # would be call #4 if not cancelled
        ],
        cancel_after=3,
    )
    agent = _agent(tmp_path, client)
    client.agent = agent

    result = agent.run("do the thing")

    assert result.status == "cancelled"
    assert len(client.calls) == 3


def test_cancelled_run_still_produces_a_normal_run_result(tmp_path: Path) -> None:
    client = _CancelAfterNCalls(
        [_plan("s1", "scratchpad", done=[]), _act("scratchpad", "c1", {"action": "list"})],
        cancel_after=2,
    )
    agent = _agent(tmp_path, client)
    client.agent = agent

    result = agent.run("do the thing")

    assert result.ok is False
    assert "cancel" in result.answer.lower()
    assert result.state is not None
    assert result.state.status == "cancelled"


def test_cancel_before_run_starts_has_no_effect(tmp_path: Path) -> None:
    """cancel() only means something once a run is in flight; calling it
    with nothing running yet, then starting a fresh run(), must not
    pre-cancel that run -- run()/resume() clear stale cancellation first."""
    client = ScriptedClient(
        [
            _plan("s1", "submit", done=["s1"]),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "c1",
                        "submit",
                        {
                            "answer": "done",
                            "status": "complete",
                            "confidence": 1.0,
                            "evidence": ["ok"],
                            "unmet_requirements": [],
                        },
                    )
                ],
                usage=Usage(100, 10),
            ),
        ],
        model="test",
    )
    agent = _agent(tmp_path, client)
    agent.cancel("premature")
    result = agent.run("do the thing")

    assert result.status == "complete"


def test_a_second_run_is_not_cancelled_by_the_first(tmp_path: Path) -> None:
    """A cancelled run must not leave the Agent instance permanently
    cancelled -- run() clears the flag, so a reused Agent's next run is
    unaffected."""
    first_client = _CancelAfterNCalls(
        [_plan("s1", "scratchpad", done=[]), _act("scratchpad", "c1", {"action": "list"})],
        cancel_after=2,
    )
    agent = _agent(tmp_path, first_client)
    first_client.agent = agent
    first = agent.run("first goal")
    assert first.status == "cancelled"

    # Swap in a fresh script for a second, uninterrupted run on the same
    # Agent instance.
    agent.llm = ScriptedClient(
        [
            _plan("s1", "submit", done=["s1"]),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "c2",
                        "submit",
                        {
                            "answer": "done",
                            "status": "complete",
                            "confidence": 1.0,
                            "evidence": ["ok"],
                            "unmet_requirements": [],
                        },
                    )
                ],
                usage=Usage(100, 10),
            ),
        ],
        model="test",
    )
    second = agent.run("second goal")
    assert second.status == "complete"


def test_resume_accepts_a_cancelled_session(tmp_path: Path) -> None:
    store = InMemoryStore()
    client = _CancelAfterNCalls(
        [_plan("s1", "scratchpad", done=[]), _act("scratchpad", "c1", {"action": "list"})],
        cancel_after=2,
    )
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent = Agent(
        AgentConfig(llm=client, workspace=ws, skills_dirs=[], store=store, budget=Budget())
    )
    client.agent = agent
    cancelled = agent.run("do the thing")
    assert cancelled.status == "cancelled"

    agent.llm = ScriptedClient(
        [
            _plan("s1", "submit", done=["s1"]),
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        "c3",
                        "submit",
                        {
                            "answer": "resumed and done",
                            "status": "complete",
                            "confidence": 1.0,
                            "evidence": ["ok"],
                            "unmet_requirements": [],
                        },
                    )
                ],
                usage=Usage(100, 10),
            ),
        ],
        model="test",
    )
    resumed = agent.resume(cancelled.session_id)
    assert resumed.status == "complete"
    assert resumed.answer == "resumed and done"


def test_cancel_from_a_real_background_thread(tmp_path: Path) -> None:
    """The realistic usage pattern: run() on a worker thread, cancel() from
    the caller. A tiny per-call sleep gives the main thread a window to call
    cancel() before the script would otherwise run to completion."""

    class _SlowNonConvergingClient(ScriptedClient):
        def complete(self, **kwargs: object) -> LLMResponse:
            time.sleep(0.02)
            return super().complete(**kwargs)  # type: ignore[arg-type]

    # Enough iterations that, uncancelled, this would run for a while --
    # cancellation should stop it well short of exhausting the script.
    script: list[LLMResponse] = []
    for _ in range(20):
        script.append(_plan("s1", "scratchpad", done=[]))
        script.append(_act("scratchpad", "c1", {"action": "list"}))
        script.append(_eval("in_progress", "keep going", done=[]))

    client = _SlowNonConvergingClient(script, model="test-slow")
    agent = _agent(tmp_path, client, max_iterations=20)

    results: list[object] = []
    thread = threading.Thread(target=lambda: results.append(agent.run("do the thing")))
    thread.start()
    time.sleep(0.05)
    agent.cancel("stop from the main thread")
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(results) == 1
    result = results[0]
    assert result.status == "cancelled"  # type: ignore[attr-defined]
    assert result.iterations < 20  # type: ignore[attr-defined]

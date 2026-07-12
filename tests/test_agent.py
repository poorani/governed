from __future__ import annotations

import json
from pathlib import Path

from governed import Agent, AgentConfig, Budget, InMemoryStore, LLMResponse
from governed.llm import ScriptedClient, ToolCall, Usage


def _plan(step: str, tool: str, why: str, done: list[str]) -> LLMResponse:
    return LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "write hi to hello.txt and verify it",
                "steps": [
                    {"id": "s1", "description": "write", "done": "s1" in done},
                    {"id": "s2", "description": "verify", "done": "s2" in done},
                ],
                "next_action": {
                    "step_id": step,
                    "tool": tool,
                    "rationale": why,
                    "success_criteria": "the tool call returns without error",
                },
            }
        )
        + "</plan>",
        usage=Usage(500, 100),
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
        usage=Usage(400, 80),
    )


def _agent(tmp_path: Path, script: list[LLMResponse]) -> tuple[Agent, ScriptedClient]:
    client = ScriptedClient(script, model="claude-sonnet-5")
    ws = tmp_path / "workspace"
    ws.mkdir()
    agent = Agent(
        AgentConfig(
            llm=client,
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
        )
    )
    return agent, client


def test_tools_are_withheld_during_analyze_and_observe(tmp_path: Path) -> None:
    script = [
        _plan("s1", "file_system", "create the file", []),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    "c1",
                    "file_system",
                    {"operation": "write", "path": "f.txt", "content": "hi"},
                )
            ],
            usage=Usage(300, 30),
        ),
        _eval("success", "wrote 2 characters", "complete", "submit", ["s1"]),
        _plan("s2", "submit", "goal complete", ["s1"]),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    "c2",
                    "submit",
                    {
                        "answer": "done",
                        "status": "complete",
                        "confidence": 0.9,
                        "evidence": ["wrote 2 characters"],
                        "unmet_requirements": [],
                    },
                )
            ],
            usage=Usage(300, 30),
        ),
    ]
    agent, client = _agent(tmp_path, script)
    result = agent.run("write hi to f.txt")

    assert result.ok
    assert result.status == "complete"
    assert client.calls[0]["tool_names"] == []  # ANALYZE: no tools offered
    assert client.calls[1]["tool_choice"] == "required"  # ACT
    assert client.calls[2]["tool_names"] == []  # OBSERVE: no tools offered


def test_contract_violation_is_retried_not_executed(tmp_path: Path) -> None:
    script = [
        _plan("s1", "file_system", "create the file", []),
        # Violates: plan committed to file_system.
        LLMResponse(
            tool_calls=[ToolCall("bad", "execute_code", {"language": "python", "code": "1"})],
            usage=Usage(300, 30),
        ),
        # Retry, now correct.
        LLMResponse(
            tool_calls=[
                ToolCall(
                    "c1",
                    "file_system",
                    {"operation": "write", "path": "f.txt", "content": "hi"},
                )
            ],
            usage=Usage(300, 30),
        ),
        _eval("success", "wrote 2 characters", "complete", "submit", ["s1"]),
        _plan("s2", "submit", "goal complete", ["s1"]),
        LLMResponse(
            tool_calls=[
                ToolCall(
                    "c2",
                    "submit",
                    {
                        "answer": "done",
                        "status": "complete",
                        "confidence": 0.9,
                        "evidence": ["wrote 2 characters"],
                        "unmet_requirements": [],
                    },
                )
            ],
            usage=Usage(300, 30),
        ),
    ]
    agent, _client = _agent(tmp_path, script)
    result = agent.run("write hi to f.txt")

    assert result.ok
    # The rejected execute_code call never touched the filesystem or ended the run.
    assert result.state is not None
    first_iter = result.state.iterations[0]
    assert len(first_iter.violations) == 1
    assert first_iter.tool_calls[0].tool == "file_system"


def test_missing_submit_tool_raises_at_construction(tmp_path: Path) -> None:
    import pytest

    from governed.tools import FileSystemTool

    ws = tmp_path / "workspace"
    ws.mkdir()
    with pytest.raises(ValueError, match="submit"):
        Agent(
            AgentConfig(
                llm=ScriptedClient([], model="scripted"),
                workspace=ws,
                skills_dirs=[],
                tools=[FileSystemTool()],
            )
        )

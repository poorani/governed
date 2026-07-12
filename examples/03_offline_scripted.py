"""The full loop, offline, no API key -- and a deliberate contract violation.

At iteration 2 the scripted model's first attempt calls `execute_code` when its
own plan committed to `file_system`. Watch the ACT contract catch it, feed the
violation back, and the model correct itself on the retry -- the call that
violated the contract never executes.

    python examples/03_offline_scripted.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from governed import Agent, AgentConfig, Budget, InMemoryStore, LLMResponse
from governed.llm import ToolCall, Usage
from governed.observability import trace_to_markdown


def plan(step: str, tool: str, why: str, done: list[str]) -> LLMResponse:
    return LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "Write 'hi' to hello.txt, then verify it.",
                "steps": [
                    {"id": "s1", "description": "Write hello.txt", "done": "s1" in done},
                    {
                        "id": "s2",
                        "description": "Read it back to verify",
                        "done": "s2" in done,
                    },
                    {"id": "s3", "description": "Report the result", "done": False},
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
        usage=Usage(1_200, 220),
    )


def evaluation(
    outcome: str, evidence: str, status: str, nxt: str, done: list[str]
) -> LLMResponse:
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
        usage=Usage(1_000, 150),
    )


SCRIPT = [
    plan("s1", "file_system", "nothing exists yet; create the file", []),
    LLMResponse(
        tool_calls=[
            ToolCall(
                "c1",
                "file_system",
                {"operation": "write", "path": "hello.txt", "content": "hi"},
            )
        ],
        usage=Usage(1_300, 60),
    ),
    evaluation(
        "success", "Wrote 2 characters to hello.txt", "in_progress", "read it back", ["s1"]
    ),
    plan("s2", "file_system", "verify the write by reading the file", ["s1"]),
    # Deliberately breaks the ACT contract: the plan above committed to
    # `file_system`, not `execute_code`. This call is rejected and never runs.
    LLMResponse(
        tool_calls=[
            ToolCall(
                "c2",
                "execute_code",
                {"language": "python", "code": "print(open('hello.txt').read())"},
            )
        ],
        usage=Usage(1_350, 70),
    ),
    # The retry, after seeing the violation feedback, calls the right tool.
    LLMResponse(
        tool_calls=[ToolCall("c3", "file_system", {"operation": "read", "path": "hello.txt"})],
        usage=Usage(1_400, 55),
    ),
    evaluation("success", "hello.txt contains 'hi'", "complete", "submit", ["s1", "s2"]),
    plan("s3", "submit", "goal complete; report the result", ["s1", "s2"]),
    LLMResponse(
        tool_calls=[
            ToolCall(
                "c4",
                "submit",
                {
                    "answer": "Wrote 'hi' to hello.txt and verified its contents "
                    "by reading it back.",
                    "status": "complete",
                    "confidence": 0.95,
                    "evidence": ["hello.txt contains 'hi'"],
                    "unmet_requirements": [],
                },
            )
        ],
        usage=Usage(1_100, 90),
    ),
]


def main() -> None:
    ws = Path(tempfile.mkdtemp()) / "workspace"
    ws.mkdir(parents=True)
    trace_path = Path(tempfile.mkdtemp()) / "run.jsonl"

    from governed.llm import ScriptedClient

    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(SCRIPT, model="claude-sonnet-5"),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
            trace_path=trace_path,
        )
    )

    result = agent.run("Write 'hi' to hello.txt, then verify it.")

    print(f"\nstatus={result.status}  confidence={result.confidence}")
    print(f"answer: {result.answer}")
    print(f"hello.txt contains: {(ws / 'hello.txt').read_text()!r}")

    print("\n--- rendered trace ---\n")
    print(trace_to_markdown(trace_path))


if __name__ == "__main__":
    main()

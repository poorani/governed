"""Adding a custom tool, exercised offline with ScriptedClient (no API key needed).

Shows the whole contract: four class attributes, one Pydantic Input model, one
run() method, and a clean ToolExecutionError on the bad-input path.

    python examples/02_custom_tool.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from governed import (
    Agent,
    AgentConfig,
    Budget,
    InMemoryStore,
    LLMResponse,
    Tool,
    ToolContext,
    ToolErrorCode,
    ToolExecutionError,
    ToolResult,
    ToolSafety,
    default_tools,
)
from governed.llm import ScriptedClient, ToolCall, Usage


class WordCountTool(Tool):
    """A trivial custom tool: count words in a string. No network, no disk."""

    name = "word_count"
    description = "Count the words in a piece of text. Splits on whitespace."
    safety = ToolSafety.READ_ONLY
    returns = "The word count, as an integer in the response text."

    class Input(BaseModel):
        text: str = Field(..., min_length=1, description="The text to count words in.")

    def run(self, args: Input, ctx: ToolContext) -> ToolResult:
        if not args.text.strip():
            raise ToolExecutionError(
                ToolErrorCode.INVALID_INPUT, "Text is empty after stripping."
            )
        n = len(args.text.split())
        return ToolResult.success(str(n), data={"word_count": n})


SCRIPT = [
    LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "Count the words in the given sentence.",
                "steps": [{"id": "s1", "description": "Count the words", "done": False}],
                "next_action": {
                    "step_id": "s1",
                    "tool": "word_count",
                    "rationale": "word_count does exactly this, directly",
                    "success_criteria": "a numeric count is returned",
                },
            }
        )
        + "</plan>",
        usage=Usage(900, 140),
    ),
    LLMResponse(
        tool_calls=[
            ToolCall(
                "c1", "word_count", {"text": "the quick brown fox jumps over the lazy dog"}
            )
        ],
        usage=Usage(950, 40),
    ),
    LLMResponse(
        text="<evaluation>"
        + json.dumps(
            {
                "outcome": "success",
                "evidence": "word_count returned 9",
                "completed_step_ids": ["s1"],
                "goal_status": "complete",
                "next_step": "submit",
            }
        )
        + "</evaluation>",
        usage=Usage(850, 60),
    ),
    LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "Count the words in the given sentence.",
                "steps": [{"id": "s1", "description": "Count the words", "done": True}],
                "next_action": {
                    "step_id": "s1",
                    "tool": "submit",
                    "rationale": "count obtained, report it",
                    "success_criteria": "run ends",
                },
            }
        )
        + "</plan>",
        usage=Usage(800, 100),
    ),
    LLMResponse(
        tool_calls=[
            ToolCall(
                "c2",
                "submit",
                {
                    "answer": "The sentence contains 9 words.",
                    "status": "complete",
                    "confidence": 1.0,
                    "evidence": ["word_count returned 9"],
                    "unmet_requirements": [],
                },
            )
        ],
        usage=Usage(700, 40),
    ),
]


def main() -> None:
    ws = Path(tempfile.mkdtemp()) / "workspace"
    ws.mkdir(parents=True)

    agent = Agent(
        AgentConfig(
            llm=ScriptedClient(SCRIPT, model="claude-sonnet-5"),
            workspace=ws,
            skills_dirs=[],
            store=InMemoryStore(),
            budget=Budget(max_iterations=6),
            # The custom tool alongside the framework's defaults. `submit` must
            # always be present -- Agent.__init__ raises without it.
            tools=[*default_tools(include_code_execution=False), WordCountTool()],
        )
    )

    result = agent.run("Count the words in: 'the quick brown fox jumps over the lazy dog'")
    print(result.status, "--", result.answer)


if __name__ == "__main__":
    main()

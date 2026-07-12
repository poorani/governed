"""The sixty-second example. Requires ANTHROPIC_API_KEY and a real model call.

export ANTHROPIC_API_KEY=sk-...
python examples/01_basic.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from governed import AgentConfig, Budget, JSONFileStore
from governed.agent import Agent


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run this example.", file=sys.stderr)
        raise SystemExit(1)

    from governed.llm import AnthropicClient

    here = Path(__file__).parent
    agent = Agent(
        AgentConfig(
            llm=AnthropicClient(model="claude-sonnet-5"),
            workspace=here / "workspace",
            skills_dirs=[str(here.parent / "skills")],
            budget=Budget(max_iterations=12, max_tokens=200_000),
            store=JSONFileStore(here / ".governed" / "sessions"),
            trace_path=here / "traces" / "01_basic.jsonl",
        )
    )

    result = agent.run(
        "Create a file called notes.md in the workspace containing a three-item "
        "markdown checklist for setting up a new Python project. Then read it "
        "back to confirm it was written correctly."
    )

    print(result.status)
    print(result.confidence)
    print(result.answer)
    print(result.unmet_requirements)
    print(f"${result.cost_usd:.4f} across {result.iterations} iterations")


if __name__ == "__main__":
    main()

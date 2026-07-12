"""Bootstrapping an agent from config alone -- no Python-side construction of
LLMConfig/GovernancePolicy/ToolConfig/etc. The dict below is exactly what a
YAML or JSON file would deserialize into; `agent_config_from_yaml("agent.yaml")`
would do the same thing from a real file. Also demonstrates the plugin
registries: a custom LLM provider and a custom tool, neither shipped by
governed, both selectable by name from config once registered. Runs fully
offline, no API key.

    python examples/05_config_driven.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pydantic import BaseModel

from governed import (
    Agent,
    LLMResponse,
    Tool,
    ToolResult,
    ToolSafety,
    register_provider,
    register_tool,
)
from governed.bootstrap import agent_config_from_mapping
from governed.llm import ScriptedClient, ToolCall, Usage
from governed.tools.base import ToolContext

# --------------------------------------------------------------------------
# In production this whole block doesn't exist -- `{"provider": "anthropic",
# "model": "claude-sonnet-5", "api_key": ...}` in the config below would
# resolve straight to a real AnthropicClient via the built-in factory (see
# "Configuring the LLM by config" in the README). Registering a stand-in
# provider is what makes *this* example runnable offline, and it doubles as
# a demonstration of the same extension point a real custom provider uses:
# implement resolve, hand it to register_provider(name, ...), and it becomes
# selectable from config by name, same as "anthropic" or "openai" are.
# --------------------------------------------------------------------------

SCRIPT = [
    LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "write hi to hello.txt, then report",
                "steps": [
                    {"id": "s1", "description": "write the file", "done": False},
                    {"id": "s2", "description": "report", "done": False},
                ],
                "next_action": {
                    "step_id": "s1",
                    "tool": "file_system",
                    "rationale": "nothing exists yet; create the file",
                    "success_criteria": "the write returns without error",
                },
            }
        )
        + "</plan>",
        usage=Usage(300, 40),
    ),
    LLMResponse(
        tool_calls=[
            ToolCall(
                "c1",
                "file_system",
                {"operation": "write", "path": "hello.txt", "content": "hi"},
            )
        ],
        usage=Usage(300, 30),
    ),
    LLMResponse(
        text="<evaluation>"
        + json.dumps(
            {
                "outcome": "success",
                "evidence": "wrote 2 characters to hello.txt",
                "completed_step_ids": ["s1"],
                "goal_status": "complete",
                "next_step": "submit",
            }
        )
        + "</evaluation>",
        usage=Usage(250, 30),
    ),
    LLMResponse(
        text="<plan>"
        + json.dumps(
            {
                "goal_restatement": "write hi to hello.txt, then report",
                "steps": [
                    {"id": "s1", "description": "write the file", "done": True},
                    {"id": "s2", "description": "report", "done": False},
                ],
                "next_action": {
                    "step_id": "s2",
                    "tool": "submit",
                    "rationale": "the file is written; report it",
                    "success_criteria": "the run ends",
                },
            }
        )
        + "</plan>",
        usage=Usage(280, 30),
    ),
    LLMResponse(
        tool_calls=[
            ToolCall(
                "c2",
                "submit",
                {
                    "answer": "Wrote 'hi' to hello.txt.",
                    "status": "complete",
                    "confidence": 0.9,
                    "evidence": ["wrote 2 characters to hello.txt"],
                    "unmet_requirements": [],
                },
            )
        ],
        usage=Usage(200, 20),
    ),
]

register_provider("scripted-demo", lambda cfg: ScriptedClient(SCRIPT, model=cfg.model))


# --------------------------------------------------------------------------
# The tool/skill/observability-sink equivalent of the provider registration
# above: `register_tool(name, factory)` makes a tool selectable by name from
# `ToolConfig(names=[...])` / a config file's `tools.names`, the same way
# `register_provider` makes a provider selectable by name from `LLMConfig`.
# `governed.skills.register_skill_source` and
# `governed.bootstrap.register_event_sink`/`register_decision_ledger_sink`/
# `register_decision_ledger_store`/`register_state_store` follow the exact
# same pattern for skill loaders and observability sinks/stores -- see
# tests/test_plugin_registries.py for one of each.
# --------------------------------------------------------------------------


class PingTool(Tool):
    """A trivial custom tool, registered as a plugin rather than passed as a
    live instance -- proof that a third-party tool is selectable from config
    alone, with the caller never importing `PingTool` directly."""

    name = "ping"
    description = "Health-check tool. Always replies 'pong'."
    safety = ToolSafety.READ_ONLY
    returns = "The string 'pong'."

    class Input(BaseModel):
        pass

    def run(self, args: Input, ctx: ToolContext) -> ToolResult:
        return ToolResult.success("pong")


register_tool("ping", PingTool)


def main() -> None:
    ws = Path(tempfile.mkdtemp()) / "workspace"
    ws.mkdir(parents=True)

    # This is exactly what a YAML/JSON config file deserializes into --
    # agent_config_from_yaml("agent.yaml") reads this same shape from disk.
    # See docs/RESPONSIBLE_AI.md and "Config-first bootstrapping" in the
    # README for the full field reference.
    config_data = {
        "llm": {
            "provider": "scripted-demo",  # in production: "anthropic" / "openai" / "gemini"
            "model": "demo-model",
        },
        "tools": {
            # "ping" isn't a built-in -- it resolves because of the
            # register_tool("ping", PingTool) plugin registration above.
            "names": ["file_system", "submit", "ping"],
        },
        "skills": {
            "dirs": [],
            "enabled": False,
        },
        "governance": {
            "allowed_tools": ["file_system", "submit", "ping"],
            "sensitive_operations": ["file_system:delete"],
            "approval_threshold": "WARNING",
        },
        "features": {
            "content_safety": True,  # adds a zero-dependency keyword scanner
            "decision_ledger": True,  # tamper-evident record, in-memory here
            "telemetry": True,
        },
        "observability": {
            "console": True,
            "decision_ledger": {
                "enabled": True,
                "store": {"type": "memory"},
            },
        },
        "budget": {
            "max_iterations": 6,
        },
    }

    config = agent_config_from_mapping(config_data, overrides={"workspace": ws})
    agent = Agent(config)

    print("--- tools resolved from config + the plugin registry ---")
    print(f"  {sorted(agent.registry.names)}  ('ping' is a plugin, not a built-in)")

    result = agent.run("Write 'hi' to hello.txt, then report what you did.")

    print(f"\nstatus={result.status}  confidence={result.confidence}")
    print(f"answer: {result.answer}")
    print(f"hello.txt contains: {(ws / 'hello.txt').read_text()!r}")

    print("\n--- decision ledger (tamper-evident, config-enabled) ---")
    assert agent.decision_ledger is not None
    for record in agent.decision_ledger.store.read(result.state.run_id):
        print(f"  seq={record.seq} tool={record.tool!r} rationale={record.rationale!r}")

    print("\n--- telemetry (config-enabled) ---")
    assert agent.telemetry is not None
    print(f"  LLM calls: {agent.telemetry.llm.count}")
    print(f"  tools called: {sorted(agent.telemetry.tools)}")

    print("\n--- content-safety scanner (config-enabled via features) ---")
    assert agent.gateway is not None
    print(f"  arg scanners: {[type(s).__name__ for s in agent.gateway.scanners]}")


if __name__ == "__main__":
    main()

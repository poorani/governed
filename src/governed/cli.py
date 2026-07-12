"""Command-line entry point. Deliberately thin: ``bootstrap.py`` already does
the parsing and resolution work, so this is argparse plus
``Agent(agent_config_from_yaml(...)).run(goal)``.

    governed agent.yaml "Profile data/sales.csv and report the top 3 regions."
    governed agent.json --goal "..." --workspace ./scratch --json

Exit code is 0 for a completed run, 1 for any other terminal status
(blocked, budget-exhausted, failed, cancelled), 2 for a CLI or config error.

The first Ctrl-C asks the agent to stop at its next checkpoint --
``Agent.cancel()``, the same cooperative kill switch available to any
embedder -- so the run still finalizes normally: trace, decision ledger, and
a real ``RunResult`` with ``status="cancelled"``, not a bare stack trace. A
second Ctrl-C means "I don't want to wait for a checkpoint" and force-quits
immediately, same as pressing it against any other CLI.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from types import FrameType

from . import __version__
from .agent import Agent
from .bootstrap import agent_config_from_json, agent_config_from_yaml
from .config import AgentConfig

__all__ = ["main"]


def _install_cancel_on_sigint(agent: Agent) -> None:
    """First Ctrl-C cancels gracefully; the handler then swaps itself out
    for Python's default (raise ``KeyboardInterrupt`` immediately), so a
    second Ctrl-C still force-quits an unresponsive run."""

    def handler(signum: int, frame: FrameType | None) -> None:
        print(
            "\ngoverned: cancelling -- finishing the current step, then stopping "
            "(press Ctrl-C again to force-quit)",
            file=sys.stderr,
        )
        agent.cancel("interrupted (Ctrl-C)")
        signal.signal(signal.SIGINT, signal.default_int_handler)

    signal.signal(signal.SIGINT, handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="governed",
        description=(
            "Run a governed agent from a config file -- see "
            "governed.bootstrap.agent_config_from_mapping for the config shape."
        ),
    )
    parser.add_argument("config", type=Path, help="Path to a YAML or JSON AgentConfig file.")
    parser.add_argument("goal", nargs="?", help="The goal to run. Reads stdin if omitted.")
    parser.add_argument("--workspace", help="Override the config's workspace directory.")
    parser.add_argument(
        "--json", action="store_true", help="Print the result as JSON instead of text."
    )
    parser.add_argument("--version", action="version", version=f"governed {__version__}")
    return parser


def _load_config(path: Path, overrides: dict[str, object]) -> AgentConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return agent_config_from_yaml(path, overrides=overrides or None)
    if suffix == ".json":
        return agent_config_from_json(path, overrides=overrides or None)
    raise ValueError(
        f"Unrecognized config extension {suffix!r} for {path} -- expected .yaml, .yml, "
        "or .json"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    goal = args.goal or sys.stdin.read().strip()
    if not goal:
        parser.error("a goal is required (positionally, or piped via stdin)")

    overrides: dict[str, object] = {}
    if args.workspace:
        overrides["workspace"] = args.workspace

    try:
        config = _load_config(args.config, overrides)
        agent = Agent(config)
    except (ValueError, ImportError, FileNotFoundError) as exc:
        print(f"governed: {exc}", file=sys.stderr)
        return 2

    _install_cancel_on_sigint(agent)
    result = agent.run(goal)

    if args.json:
        print(
            json.dumps(
                {
                    "status": result.status,
                    "answer": result.answer,
                    "confidence": result.confidence,
                    "evidence": result.evidence,
                    "unmet_requirements": result.unmet_requirements,
                    "iterations": result.iterations,
                    "total_tokens": result.total_tokens,
                    "cost_usd": result.cost_usd,
                    "duration_s": result.duration_s,
                    "session_id": result.session_id,
                },
                indent=2,
            )
        )
    else:
        print(f"[{result.status}] {result.answer}")
        if result.evidence:
            print("\nEvidence:")
            for item in result.evidence:
                print(f"  - {item}")
        if result.unmet_requirements:
            print("\nUnmet requirements:")
            for item in result.unmet_requirements:
                print(f"  - {item}")
        print(
            f"\n{result.iterations} iteration(s), {result.total_tokens} tokens, "
            f"${result.cost_usd:.4f}, {result.duration_s:.1f}s, session={result.session_id}"
        )

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())

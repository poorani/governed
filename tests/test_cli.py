"""The `governed` console script: argparse + bootstrap.agent_config_from_*
+ Agent.run, wired end to end. No network, no real API key -- the LLM is a
ScriptedClient registered under a throwaway provider name, the same pattern
`test_llm_factory.py` and `examples/05_config_driven.py` already use.
"""

from __future__ import annotations

import io
import json
import signal
from pathlib import Path

import pytest

from governed.cli import _install_cancel_on_sigint, build_parser, main
from governed.llm import LLMResponse, ScriptedClient, ToolCall, Usage
from governed.llm.factory import _REGISTRY

PLAN = LLMResponse(
    text="<plan>"
    + json.dumps(
        {
            "goal_restatement": "say hi",
            "steps": [{"id": "s1", "description": "report", "done": False}],
            "next_action": {
                "step_id": "s1",
                "tool": "submit",
                "rationale": "just answer",
                "success_criteria": "done",
            },
        }
    )
    + "</plan>",
    usage=Usage(100, 20),
)

SUBMIT = LLMResponse(
    tool_calls=[
        ToolCall(
            "c1",
            "submit",
            {
                "answer": "hi",
                "status": "complete",
                "confidence": 0.9,
                "evidence": ["trivial"],
                "unmet_requirements": [],
            },
        )
    ],
    usage=Usage(100, 10),
)


@pytest.fixture
def scripted_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        _REGISTRY, "scripted-demo", lambda cfg: ScriptedClient([PLAN, SUBMIT], model=cfg.model)
    )


def _write_config(tmp_path: Path, workspace: Path) -> Path:
    config_path = tmp_path / "agent.json"
    config_path.write_text(
        json.dumps(
            {
                "llm": {"provider": "scripted-demo", "model": "demo-model"},
                "tools": {"names": ["submit"]},
                "skills": {"dirs": [], "enabled": False},
                "workspace": str(workspace),
                "budget": {"max_iterations": 4},
            }
        )
    )
    return config_path


def test_run_with_positional_goal_exits_zero(
    tmp_path: Path, scripted_provider: None, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(tmp_path, tmp_path / "ws")
    rc = main([str(config_path), "say hi"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[complete] hi" in out


def test_run_json_output_is_well_formed(
    tmp_path: Path, scripted_provider: None, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = _write_config(tmp_path, tmp_path / "ws")
    rc = main([str(config_path), "say hi", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert payload["answer"] == "hi"
    assert payload["confidence"] == 0.9


def test_goal_falls_back_to_stdin(
    tmp_path: Path,
    scripted_provider: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path, tmp_path / "ws")
    monkeypatch.setattr("sys.stdin", io.StringIO("say hi\n"))
    rc = main([str(config_path)])
    assert rc == 0
    assert "[complete] hi" in capsys.readouterr().out


def test_workspace_override_wins_over_config(tmp_path: Path, scripted_provider: None) -> None:
    config_path = _write_config(tmp_path, tmp_path / "ws-from-config")
    override_ws = tmp_path / "ws-from-flag"
    rc = main([str(config_path), "say hi", "--workspace", str(override_ws)])
    assert rc == 0
    assert override_ws.exists()
    assert not (tmp_path / "ws-from-config").exists()


def test_missing_config_file_exits_two(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["/nonexistent/agent.yaml", "say hi"])
    assert rc == 2
    assert "not found" in capsys.readouterr().err.lower()


def test_unrecognized_extension_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bogus = tmp_path / "agent.toml"
    bogus.write_text("llm = {}")
    rc = main([str(bogus), "say hi"])
    assert rc == 2
    assert "unrecognized" in capsys.readouterr().err.lower()


def test_missing_goal_and_empty_stdin_is_a_parser_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = _write_config(tmp_path, tmp_path / "ws")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    with pytest.raises(SystemExit) as exc_info:
        main([str(config_path)])
    assert exc_info.value.code == 2


def test_build_parser_exposes_config_and_goal_positionals() -> None:
    parser = build_parser()
    args = parser.parse_args(["agent.yaml", "do the thing", "--workspace", "/tmp/ws"])
    assert args.config == Path("agent.yaml")
    assert args.goal == "do the thing"
    assert args.workspace == "/tmp/ws"


def test_sigint_handler_cancels_then_restores_the_default(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """First Ctrl-C should call agent.cancel() and swap itself out for
    Python's default handler; a second Ctrl-C is then a normal, immediate
    KeyboardInterrupt. Invokes the installed handler directly rather than
    sending a real OS signal, so this doesn't depend on signal-delivery
    timing across threads."""

    class _FakeAgent:
        def __init__(self) -> None:
            self.cancelled_with: list[str] = []

        def cancel(self, reason: str = "") -> None:
            self.cancelled_with.append(reason)

    original = signal.getsignal(signal.SIGINT)
    try:
        agent = _FakeAgent()
        _install_cancel_on_sigint(agent)  # type: ignore[arg-type]

        installed = signal.getsignal(signal.SIGINT)
        installed(signal.SIGINT, None)  # type: ignore[misc]

        assert agent.cancelled_with == ["interrupted (Ctrl-C)"]
        assert "cancelling" in capsys.readouterr().err.lower()
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    finally:
        signal.signal(signal.SIGINT, original)

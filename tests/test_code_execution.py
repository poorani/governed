"""CodeExecutionTool's pluggable ExecutionBackend: the default SubprocessBackend
(exercised for real -- no network, fast, deterministic) and
DockerCodeExecutionBackend (exercised through a fake subprocess.Popen double,
since CI has no Docker daemon -- the same style test_llm_factory.py uses to
verify wire-level logic without a real vendor SDK).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from governed.tools.base import ToolContext
from governed.tools.code_execution import (
    BackendTimeout,
    BackendUnavailable,
    CodeExecutionTool,
    DockerCodeExecutionBackend,
    ExecResult,
    SubprocessBackend,
    _Input,
)
from governed.tools.errors import ToolErrorCode, ToolExecutionError

if sys.platform == "win32":  # pragma: no cover
    pytest.skip("subprocess semantics tested here are POSIX-specific", allow_module_level=True)

# ---------------------------------------------------------------------------
# SubprocessBackend / CodeExecutionTool default behaviour (real subprocess)
# ---------------------------------------------------------------------------


def test_default_backend_is_subprocess() -> None:
    assert isinstance(CodeExecutionTool().backend, SubprocessBackend)


def test_python_success(ctx: ToolContext) -> None:
    tool = CodeExecutionTool()
    result = tool.run(_Input(language="python", code="print('hi')", timeout_s=10), ctx)
    assert result.ok
    assert "[exit code 0]" in result.content
    assert "hi" in result.content


def test_bash_success(ctx: ToolContext) -> None:
    tool = CodeExecutionTool()
    result = tool.run(_Input(language="bash", code="echo hi", timeout_s=10), ctx)
    assert result.ok
    assert "hi" in result.content


def test_nonzero_exit_is_execution_failed(ctx: ToolContext) -> None:
    tool = CodeExecutionTool()
    with pytest.raises(ToolExecutionError) as exc_info:
        tool.run(_Input(language="python", code="import sys; sys.exit(3)", timeout_s=10), ctx)
    assert exc_info.value.error.code is ToolErrorCode.EXECUTION_FAILED


def test_timeout_is_reported_with_partial_output(ctx: ToolContext) -> None:
    tool = CodeExecutionTool()
    code = "import sys, time; print('before', flush=True); time.sleep(10)"
    with pytest.raises(ToolExecutionError) as exc_info:
        tool.run(_Input(language="python", code=code, timeout_s=1), ctx)
    assert exc_info.value.error.code is ToolErrorCode.TIMEOUT
    assert "before" in exc_info.value.error.details["partial_output"]


def test_output_over_limit_is_truncated(ctx: ToolContext) -> None:
    tool = CodeExecutionTool()
    code = "print('x' * 25_000)"
    result = tool.run(_Input(language="python", code=code, timeout_s=10), ctx)
    assert result.truncated


def test_env_is_stripped_to_allowlist(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOVERNED_TEST_SECRET", "leaked-if-you-see-this")
    tool = CodeExecutionTool()
    code = "import os, sys; sys.exit(1 if 'GOVERNED_TEST_SECRET' in os.environ else 0)"
    result = tool.run(_Input(language="python", code=code, timeout_s=10), ctx)
    assert result.ok


def test_missing_interpreter_is_dependency_missing(
    ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(*a: object, **k: object) -> None:
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(subprocess, "Popen", _raise)
    tool = CodeExecutionTool()
    with pytest.raises(ToolExecutionError) as exc_info:
        tool.run(_Input(language="python", code="1+1", timeout_s=10), ctx)
    assert exc_info.value.error.code is ToolErrorCode.DEPENDENCY_MISSING


# ---------------------------------------------------------------------------
# DockerCodeExecutionBackend (fake subprocess.Popen -- no real Docker needed)
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, cmd: list[str], output: str = "", returncode: int = 0) -> None:
        self.cmd = cmd
        self._output = output
        self.returncode = returncode

    def communicate(self, timeout: float | None = None) -> tuple[str, None]:
        return self._output, None


class _HangingProc(_FakeProc):
    def __init__(self, cmd: list[str]) -> None:
        super().__init__(cmd)
        self._first_call = True

    def communicate(self, timeout: float | None = None) -> tuple[str, None]:
        if self._first_call:
            self._first_call = False
            raise subprocess.TimeoutExpired(self.cmd, timeout or 0)
        return "partial", None


def test_docker_backend_builds_an_isolated_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakeProc:
        captured["cmd"] = cmd
        return _FakeProc(cmd, output="ok", returncode=0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    backend = DockerCodeExecutionBackend()
    result = backend.run(language="python", code="print(1)", workspace=tmp_path, timeout_s=10)

    assert result == ExecResult(returncode=0, output="ok")
    cmd = captured["cmd"]
    assert cmd[:2] == ["docker", "run"]
    assert "--rm" in cmd
    assert "none" in cmd  # --network none
    assert "--read-only" in cmd
    assert "--cap-drop" in cmd
    assert f"{tmp_path}:/workspace:rw" in cmd
    assert "python:3.12-slim" in cmd
    assert cmd[-3:] == ["python3", "-c", "print(1)"]


def test_docker_backend_respects_custom_image_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakeProc:
        captured["cmd"] = cmd
        return _FakeProc(cmd, output="", returncode=0)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    backend = DockerCodeExecutionBackend(image="python:3.11-alpine", memory="256m", cpus="0.5")
    backend.run(language="bash", code="echo hi", workspace=tmp_path, timeout_s=10)

    cmd = captured["cmd"]
    assert "python:3.11-alpine" in cmd
    assert "256m" in cmd
    assert "0.5" in cmd


def test_docker_backend_timeout_kills_the_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kill_calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _HangingProc:
        return _HangingProc(cmd)

    def fake_run(cmd: list[str], **kwargs: object) -> None:
        kill_calls.append(cmd)

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "run", fake_run)
    backend = DockerCodeExecutionBackend()

    with pytest.raises(BackendTimeout) as exc_info:
        backend.run(
            language="python", code="while True: pass", workspace=tmp_path, timeout_s=1
        )

    assert exc_info.value.partial_output == "partial"
    assert kill_calls and kill_calls[0][:2] == ["docker", "kill"]


def test_docker_backend_missing_cli_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_popen(cmd: list[str], **kwargs: object) -> None:
        raise FileNotFoundError("docker: command not found")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    backend = DockerCodeExecutionBackend()

    with pytest.raises(BackendUnavailable):
        backend.run(language="python", code="1+1", workspace=tmp_path, timeout_s=10)


def test_code_execution_tool_delegates_to_a_custom_backend(
    tmp_path: Path, ctx: ToolContext
) -> None:
    class _StubBackend:
        def run(self, **kwargs: object) -> ExecResult:
            return ExecResult(returncode=0, output="stubbed")

    tool = CodeExecutionTool(backend=_StubBackend())
    result = tool.run(_Input(language="python", code="ignored", timeout_s=10), ctx)
    assert "stubbed" in result.content


def test_code_execution_tool_translates_backend_timeout(ctx: ToolContext) -> None:
    class _TimeoutBackend:
        def run(self, **kwargs: object) -> ExecResult:
            raise BackendTimeout(partial_output="so far so good")

    tool = CodeExecutionTool(backend=_TimeoutBackend())
    with pytest.raises(ToolExecutionError) as exc_info:
        tool.run(_Input(language="python", code="x", timeout_s=1), ctx)
    assert exc_info.value.error.code is ToolErrorCode.TIMEOUT
    assert exc_info.value.error.details["partial_output"] == "so far so good"


def test_code_execution_tool_translates_backend_unavailable(ctx: ToolContext) -> None:
    class _UnavailableBackend:
        def run(self, **kwargs: object) -> ExecResult:
            raise BackendUnavailable("docker not installed")

    tool = CodeExecutionTool(backend=_UnavailableBackend())
    with pytest.raises(ToolExecutionError) as exc_info:
        tool.run(_Input(language="python", code="x", timeout_s=1), ctx)
    assert exc_info.value.error.code is ToolErrorCode.DEPENDENCY_MISSING


def test_docker_backend_selectable_via_register_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plugin-registry path from `governed.register_tool`: a deployment
    swaps the sandboxed backend in for the whole fleet by replacing the
    `execute_code` factory, with no change to `ToolConfig(names=[...])`."""
    from governed.tools import _TOOL_REGISTRY

    monkeypatch.setitem(
        _TOOL_REGISTRY,
        "execute_code",
        lambda: CodeExecutionTool(backend=DockerCodeExecutionBackend()),
    )
    tool = _TOOL_REGISTRY["execute_code"]()
    assert isinstance(tool, CodeExecutionTool)
    assert isinstance(tool.backend, DockerCodeExecutionBackend)

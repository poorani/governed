"""Code execution: Python or bash, in the workspace, dispatched to a
pluggable ``ExecutionBackend``.

``SubprocessBackend`` -- the default -- is a guardrail, not a jail: the
subprocess runs as the same OS user as the agent. On POSIX,
``RLIMIT_CPU``/``RLIMIT_AS``/``RLIMIT_NOFILE``/``RLIMIT_FSIZE`` bound it, the
process gets its own session so a timeout can kill the whole tree, and the
environment is stripped to an allowlist -- the agent's own API keys are not
passed through. Network egress is **not** blocked; that stops an agent's
mistakes, not an adversary's intent.

For untrusted goals, use ``DockerCodeExecutionBackend`` instead --
``CodeExecutionTool(backend=DockerCodeExecutionBackend())`` -- which runs the
same code inside a throwaway, no-network, read-only, resource-capped
container: real namespace/cgroup isolation instead of same-user resource
limits. Or write your own ``ExecutionBackend`` (one method, ``run``) and pass
it the same way. See ``docs/RESPONSIBLE_AI.md`` §7 for the boundary this
distinction matters for.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from .base import Tool, ToolContext, ToolResult, ToolSafety, env_allowlist
from .errors import ToolErrorCode, ToolExecutionError

__all__ = [
    "BackendTimeout",
    "BackendUnavailable",
    "CodeExecutionTool",
    "DockerCodeExecutionBackend",
    "ExecResult",
    "ExecutionBackend",
    "SubprocessBackend",
]

_MAX_OUTPUT = 20_000
_CPU_SECONDS = 30
_MEMORY_BYTES = 1_024 * 1_024 * 1_024  # 1 GiB
_MAX_OPEN_FILES = 256
_MAX_FILE_SIZE = 50 * 1_024 * 1_024  # 50 MB


class _Input(BaseModel):
    language: Literal["python", "bash"]
    code: str = Field(..., description="The source to run.")
    timeout_s: int = Field(30, ge=1, le=300, description="Wall-clock timeout.")


def _clip(text: str | None) -> str:
    text = text or ""
    return text if len(text) <= _MAX_OUTPUT else text[:_MAX_OUTPUT] + "\n[output truncated]"


# ---------------------------------------------------------------------------
# Execution backends
# ---------------------------------------------------------------------------


@dataclass
class ExecResult:
    returncode: int
    output: str


class BackendTimeout(Exception):
    """Raised by an ``ExecutionBackend`` when the wall-clock timeout is hit.
    Whatever output was captured before the kill is preserved."""

    def __init__(self, partial_output: str) -> None:
        super().__init__("execution timed out")
        self.partial_output = partial_output


class BackendUnavailable(Exception):
    """Raised when the backend's own runtime isn't reachable -- no
    interpreter on ``PATH``, no ``docker`` CLI, no daemon listening."""


class ExecutionBackend(Protocol):
    """Runs one piece of source and returns what happened. Implementations
    own timeout enforcement themselves -- raise ``BackendTimeout`` /
    ``BackendUnavailable`` rather than letting the underlying exception
    (``subprocess.TimeoutExpired``, ``FileNotFoundError``, ...) escape, so
    ``CodeExecutionTool`` can translate either into the same model-facing
    ``ToolExecutionError`` regardless of which backend is in play."""

    def run(
        self,
        *,
        language: Literal["python", "bash"],
        code: str,
        workspace: Path,
        timeout_s: int,
    ) -> ExecResult: ...


def _limit_resources() -> None:  # pragma: no cover -- POSIX only, exercised via subprocess
    try:
        import resource
    except ImportError:
        pass  # Windows: no resource module. The timeout and workspace cwd still hold.
    else:
        # Each limit is applied independently and best-effort: some platforms
        # enforce their own ceiling below what we ask for here (notably
        # macOS, where RLIMIT_AS often can't be tightened at all -- "current
        # limit exceeds maximum limit" even though getrlimit reports
        # unlimited) and raising from preexec_fn aborts the whole subprocess
        # launch, not just the one limit. A limit that silently doesn't
        # apply is consistent with this being a guardrail, not a jail; a
        # child process that never starts is a worse failure mode than that.
        for res, value in (
            (resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS)),
            (resource.RLIMIT_AS, (_MEMORY_BYTES, _MEMORY_BYTES)),
            (resource.RLIMIT_NOFILE, (_MAX_OPEN_FILES, _MAX_OPEN_FILES)),
            (resource.RLIMIT_FSIZE, (_MAX_FILE_SIZE, _MAX_FILE_SIZE)),
        ):
            with contextlib.suppress(ValueError, OSError):
                resource.setrlimit(res, value)
    # No os.setsid() here: SubprocessBackend already launches with
    # start_new_session=True, which calls setsid() itself before exec. A
    # second setsid() call in preexec_fn hits the child *after* that -- it
    # is already a session leader -- and raises EPERM, which aborts the
    # whole subprocess launch (a preexec_fn exception kills process
    # creation, not just the one call). One setsid() is enough to give
    # _kill_tree's killpg() a process group to target.


class SubprocessBackend:
    """Default backend: same-user subprocess, resource-limited. See the
    module docstring for exactly what this does and does not bound."""

    def run(
        self,
        *,
        language: Literal["python", "bash"],
        code: str,
        workspace: Path,
        timeout_s: int,
    ) -> ExecResult:
        cmd = [sys.executable, "-c", code] if language == "python" else ["bash", "-c", code]

        env = env_allowlist()
        popen_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            popen_kwargs["preexec_fn"] = _limit_resources
            popen_kwargs["start_new_session"] = True

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(workspace),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                **popen_kwargs,
            )
        except FileNotFoundError as exc:
            raise BackendUnavailable(f"Interpreter for {language!r} not found: {exc}") from exc

        try:
            output, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            output, _ = proc.communicate()
            raise BackendTimeout(partial_output=_clip(output)) from None

        return ExecResult(returncode=proc.returncode, output=output or "")

    @staticmethod
    def _kill_tree(proc: subprocess.Popen[str]) -> None:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(proc.pid), 9)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError):
            pass


@dataclass
class DockerCodeExecutionBackend:
    """Runs code inside a throwaway Docker container: no network, a
    read-only root filesystem, and memory/CPU/process caps -- real
    namespace and cgroup isolation instead of ``SubprocessBackend``'s
    same-user resource limits.

    Shells out to the ``docker`` CLI (no SDK dependency, consistent with the
    rest of the core: the dependency tree stays whatever the *deployment*
    already has, not something this package pulls in). Requires ``docker``
    on ``PATH`` and a reachable daemon; raises ``BackendUnavailable`` if
    either is missing, the same failure ``CodeExecutionTool`` already
    translates into ``ToolErrorCode.DEPENDENCY_MISSING`` for
    ``SubprocessBackend``.

    The workspace is bind-mounted read-write at ``/workspace`` (code that
    writes output files still works); nothing else on the host is reachable.
    No host environment variable is passed through -- ``docker run`` does not
    inherit the caller's environment unless told to with ``-e``, and this
    backend never does. Swap ``docker_bin="podman"`` for a rootless
    alternative; both speak the same CLI surface used here.
    """

    image: str = "python:3.12-slim"
    memory: str = "512m"
    cpus: str = "1"
    pids_limit: int = 128
    docker_bin: str = "docker"
    #: Extra arguments spliced in before the image name -- a seccomp profile,
    #: an additional read-only mount, etc.
    extra_args: list[str] = field(default_factory=list)

    def run(
        self,
        *,
        language: Literal["python", "bash"],
        code: str,
        workspace: Path,
        timeout_s: int,
    ) -> ExecResult:
        container = f"governed-exec-{uuid.uuid4().hex[:12]}"
        interpreter = ["python3", "-c", code] if language == "python" else ["bash", "-c", code]
        cmd = [
            self.docker_bin,
            "run",
            "--rm",
            "--name",
            container,
            "--network",
            "none",
            "--memory",
            self.memory,
            "--cpus",
            self.cpus,
            "--pids-limit",
            str(self.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,size=64m",
            "-v",
            f"{workspace}:/workspace:rw",
            "-w",
            "/workspace",
            *self.extra_args,
            self.image,
            *interpreter,
        ]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
        except FileNotFoundError as exc:
            raise BackendUnavailable(f"{self.docker_bin!r} CLI not found: {exc}") from exc

        try:
            output, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            subprocess.run(
                [self.docker_bin, "kill", container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            output, _ = proc.communicate()
            raise BackendTimeout(partial_output=_clip(output)) from None

        return ExecResult(returncode=proc.returncode, output=output or "")


class CodeExecutionTool(Tool):
    name = "execute_code"
    description = (
        "Run Python or bash code in the workspace directory. Use for computation, "
        "scripting, and anything that isn't better served by file_system or "
        "analyze_data. Output is captured and truncated to 20k characters. "
        "The process is time- and resource-limited and cannot see the host's "
        "environment variables or API keys."
    )
    safety = ToolSafety.EXECUTES_CODE
    returns = "Combined stdout/stderr, prefixed with the exit code."
    Input = _Input

    def __init__(self, backend: ExecutionBackend | None = None) -> None:
        self.backend = backend or SubprocessBackend()

    def run(self, args: _Input, ctx: ToolContext) -> ToolResult:
        try:
            result = self.backend.run(
                language=args.language,
                code=args.code,
                workspace=ctx.workspace,
                timeout_s=args.timeout_s,
            )
        except BackendUnavailable as exc:
            raise ToolExecutionError(ToolErrorCode.DEPENDENCY_MISSING, str(exc)) from exc
        except BackendTimeout as exc:
            raise ToolExecutionError(
                ToolErrorCode.TIMEOUT,
                f"Execution exceeded {args.timeout_s}s and was terminated.",
                remediation="Reduce the amount of work per call, or raise timeout_s "
                "(capped at 300s).",
                partial_output=exc.partial_output,
            ) from exc

        truncated = len(result.output) > _MAX_OUTPUT
        text = _clip(result.output)
        header = f"[exit code {result.returncode}]\n"
        if result.returncode != 0:
            raise ToolExecutionError(
                ToolErrorCode.EXECUTION_FAILED,
                f"{args.language} exited with code {result.returncode}.",
                remediation="Read the output for the traceback/error and fix the code "
                "before retrying.",
                output=text,
            )
        return ToolResult.success(
            header + text, data={"exit_code": result.returncode}, truncated=truncated
        )


def make_scratch_dir() -> Path:
    """Unused by default, exposed for subclasses that want a per-call scratch dir."""
    return Path(tempfile.mkdtemp(prefix="governed-"))

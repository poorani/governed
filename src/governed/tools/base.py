"""The tool contract every default and custom tool implements.

A tool is four class attributes, one Pydantic model, one method. ``ToolRegistry``
(``registry.py``) is the single chokepoint that validates input, gates side
effects, bounds runtime, and normalises errors -- no individual tool has to.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .errors import ToolError, ToolErrorCode, ToolExecutionError

__all__ = [
    "DANGEROUS",
    "Artifact",
    "SandboxViolation",
    "Tool",
    "ToolConfig",
    "ToolContext",
    "ToolResult",
    "ToolSafety",
    "ToolSpec",
]


class ToolSafety(str, Enum):
    """The declared blast radius of a tool *class*.

    This is a coarse instrument -- it cannot distinguish
    ``file_system(operation="read")`` from ``file_system(operation="delete")``.
    For per-call risk, see ``security.guardrails.RiskPolicy``, which is strictly
    more informed because it sees the arguments.
    """

    READ_ONLY = "read_only"
    MUTATES_STATE = "mutates_state"
    EXECUTES_CODE = "executes_code"
    NETWORK = "network"


#: Safety classes that require approval under the simple ``approval_policy``
#: gate (``AgentConfig.approval_policy="dangerous"``). Guardrails supersede this.
DANGEROUS: frozenset[ToolSafety] = frozenset(
    {ToolSafety.MUTATES_STATE, ToolSafety.EXECUTES_CODE, ToolSafety.NETWORK}
)


@dataclass(frozen=True)
class ToolSpec:
    """The model-facing and gateway-facing description of a tool.

    Produced fresh from a ``Tool`` instance by ``Tool.spec()``. Kept separate
    from the ``Tool`` object itself so the registry and gateway can reason about
    a tool without holding a reference to its (possibly stateful) implementation.
    """

    name: str
    description: str
    safety: ToolSafety
    returns: str
    input_schema: dict[str, Any]
    requires_approval: bool = False

    def to_llm_schema(self) -> dict[str, Any]:
        """The shape every ``LLMClient.complete(tools=...)`` expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class Artifact:
    """A file the tool produced, so the final answer can cite it."""

    path: str
    description: str = ""
    bytes: int = 0


@dataclass
class ToolResult:
    """What a tool returns. Never raises past the registry -- see ``invoke``."""

    ok: bool
    content: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: ToolError | None = None
    artifacts: list[Artifact] = field(default_factory=list)
    truncated: bool = False
    duration_ms: int = 0

    @classmethod
    def success(
        cls,
        content: str,
        *,
        data: dict[str, Any] | None = None,
        artifacts: list[Artifact] | None = None,
        truncated: bool = False,
    ) -> ToolResult:
        return cls(
            ok=True,
            content=content,
            data=data or {},
            artifacts=artifacts or [],
            truncated=truncated,
        )

    @classmethod
    def failure(cls, error: ToolError, *, duration_ms: int = 0) -> ToolResult:
        return cls(ok=False, error=error, duration_ms=duration_ms)

    def to_model_text(self) -> str:
        """What actually goes back into the transcript as a tool_result block."""
        if self.ok:
            text = self.content
            if self.truncated:
                text += "\n\n[output truncated]"
            return text
        return self.error.to_model_text() if self.error else "ERROR: unknown failure"


class SandboxViolation(ToolExecutionError):
    def __init__(self, path: str) -> None:
        super().__init__(
            ToolErrorCode.UNSAFE_OPERATION,
            f"Path escapes the workspace sandbox: {path}",
            remediation="Use a path relative to the workspace root. Absolute paths, "
            "'..' traversal, and symlinks resolving outside the workspace are refused.",
        )


@dataclass
class ToolContext:
    """Everything a tool's ``run`` needs besides its own parsed arguments."""

    workspace: Path
    scratchpad: dict[str, Any]
    run_id: str
    iteration: int = 0
    #: ``(ToolSpec, args) -> bool``. Populated by the Agent; tools rarely call
    #: this directly -- the registry/gateway already gate on ``ToolSafety``.
    approve: Callable[[ToolSpec, dict[str, Any]], bool] = field(
        default=lambda spec, args: True
    )
    #: Mutated by control tools (``submit``) to signal the loop should stop.
    signals: dict[str, Any] = field(default_factory=dict)

    def resolve(self, raw: str) -> Path:
        """The sandbox. Every filesystem-touching tool routes through this.

        Rejects absolute paths, ``..`` traversal, and symlinks resolving outside
        the workspace root. This is a real boundary, not a convention.
        """
        if Path(raw).is_absolute():
            raise SandboxViolation(raw)
        candidate = (self.workspace / raw).resolve()
        try:
            candidate.relative_to(self.workspace.resolve())
        except ValueError:
            raise SandboxViolation(raw) from None
        return candidate


class Tool(ABC):
    """Subclass this. Four attributes, one Pydantic ``Input`` model, one method."""

    name: str
    description: str
    safety: ToolSafety
    returns: str = ""
    #: Terminal tools (``submit``) end the run and are exempt from the circuit
    #: breaker's repetition/stall detectors.
    terminal: bool = False
    #: Doubles as the JSON Schema sent to the LLM.
    Input: type[BaseModel]

    def spec(self) -> ToolSpec:
        schema = self.Input.model_json_schema()
        schema.pop("title", None)
        return ToolSpec(
            name=self.name,
            description=self.description,
            safety=self.safety,
            returns=self.returns,
            input_schema=schema,
            requires_approval=self.safety in DANGEROUS,
        )

    @abstractmethod
    def run(self, args: Any, ctx: ToolContext) -> ToolResult: ...


def read_bounded(path: Path, max_bytes: int = 200_000) -> tuple[str, bool]:
    """Read a file, truncating rather than blowing the context budget."""
    size = path.stat().st_size
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        text = fh.read(max_bytes)
    return text, size > max_bytes


def env_allowlist(extra: dict[str, str] | None = None) -> dict[str, str]:
    """A minimal, safe environment for subprocesses. No inherited API keys."""
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "TMPDIR")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(extra or {})
    return env


@dataclass
class ToolConfig:
    """Declarative tool selection, for config-driven bootstrapping.

    ``AgentConfig(tools=...)`` already accepts a live ``list[Tool]``; this is
    the data-only alternative -- a plain, JSON/YAML-safe description that
    ``governed.tools.resolve_tools`` turns into one. Resolved the same way
    ``LLMConfig`` is: pass it and the framework builds the list, no import of
    the concrete ``Tool`` classes required at the call site.

    ``names=None`` (the default) reproduces ``default_tools()`` exactly,
    gated by ``include_code_execution``/``include_data_analysis``.  Set
    ``names`` to an explicit subset of the built-in registry (see
    ``governed.tools.BUILTIN_TOOLS``) to hand-pick tools by name instead --
    e.g. ``ToolConfig(names=["file_system", "submit"])`` for a read/write-only,
    no-shell deployment expressed entirely in data. ``extra`` is the escape
    hatch for custom ``Tool`` instances this format can't name -- the same
    role ``LLMConfig.extra`` plays for provider-specific adapter kwargs.
    """

    include_code_execution: bool = True
    include_data_analysis: bool = True
    names: list[str] | None = None
    extra: list[Any] = field(default_factory=list)

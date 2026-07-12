from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .base import (
    DANGEROUS,
    Artifact,
    SandboxViolation,
    Tool,
    ToolConfig,
    ToolContext,
    ToolResult,
    ToolSafety,
    ToolSpec,
)
from .code_execution import (
    BackendTimeout,
    BackendUnavailable,
    CodeExecutionTool,
    DockerCodeExecutionBackend,
    ExecResult,
    ExecutionBackend,
    SubprocessBackend,
)
from .control import LoadSkillTool, ScratchpadTool, SubmitTool
from .data_analysis import DataAnalysisTool
from .errors import ToolError, ToolErrorCode, ToolExecutionError
from .filesystem import FileSystemTool
from .registry import ToolRegistry

if TYPE_CHECKING:
    from ..skills.loader import SkillLibrary

__all__ = [
    "DANGEROUS",
    "Artifact",
    "BUILTIN_TOOLS",
    "BackendTimeout",
    "BackendUnavailable",
    "CodeExecutionTool",
    "DataAnalysisTool",
    "DockerCodeExecutionBackend",
    "ExecResult",
    "ExecutionBackend",
    "FileSystemTool",
    "LoadSkillTool",
    "SandboxViolation",
    "ScratchpadTool",
    "SubmitTool",
    "SubprocessBackend",
    "Tool",
    "ToolConfig",
    "ToolContext",
    "ToolError",
    "ToolErrorCode",
    "ToolExecutionError",
    "ToolFactory",
    "ToolRegistry",
    "ToolResult",
    "ToolSafety",
    "ToolSpec",
    "default_tools",
    "register_tool",
    "registered_tool_names",
    "resolve_tools",
]

#: ``() -> a ready-to-use Tool``. See ``register_tool``.
ToolFactory = Callable[[], Tool]


def default_tools(
    skills: SkillLibrary | None = None,
    *,
    include_code_execution: bool = True,
    include_data_analysis: bool = True,
) -> list[Tool]:
    """The tools every agent gets unless you say otherwise.

    ``submit`` and ``scratchpad`` are always included -- a tool set without a
    terminal tool can never end a run successfully, and ``Agent.__init__``
    raises if you build one without ``submit``. ``execute_code`` and
    ``analyze_data`` can be dropped for untrusted goals or when their
    dependencies aren't installed::

        default_tools(include_code_execution=False)
    """
    tools: list[Tool] = [FileSystemTool(), ScratchpadTool(), SubmitTool()]
    if include_code_execution:
        tools.append(CodeExecutionTool())
    if include_data_analysis:
        tools.append(DataAnalysisTool())
    if skills is not None:
        tools.append(LoadSkillTool(skills))
    return tools


#: Built-in tools nameable from data (``ToolConfig(names=[...])``, or a
#: bootstrap config file). Only the no-argument ones -- ``load_skill`` needs a
#: ``SkillLibrary`` and is special-cased in ``resolve_tools`` below. This dict
#: itself is not consulted at resolve time -- it only seeds ``_TOOL_REGISTRY``
#: below -- so mutating it after import has no effect; use ``register_tool``.
BUILTIN_TOOLS: dict[str, ToolFactory] = {
    "file_system": FileSystemTool,
    "scratchpad": ScratchpadTool,
    "submit": SubmitTool,
    "execute_code": CodeExecutionTool,
    "analyze_data": DataAnalysisTool,
}

_TOOL_REGISTRY: dict[str, ToolFactory] = dict(BUILTIN_TOOLS)


def register_tool(name: str, factory: ToolFactory) -> None:
    """Add or replace the tool built by ``ToolConfig(names=[name, ...])`` --
    the same pattern ``register_provider`` uses for LLM adapters.

    ``factory`` takes no arguments and returns a ready-to-use ``Tool``
    instance; names are matched case-insensitively, and registering an
    existing name (including a built-in one) replaces it. For a tool that
    needs per-deployment configuration -- a database handle, an API client --
    close over it in the factory at registration time::

        register_tool("crm_lookup", lambda: CrmLookupTool(client=my_crm_client))

    A tool that needs something only known *per run* (not at plugin-register
    time) can't go through this seam -- pass a live instance via
    ``ToolConfig(extra=[...])`` or ``AgentConfig(tools=[...])`` instead.
    """
    _TOOL_REGISTRY[name.lower()] = factory


def registered_tool_names() -> list[str]:
    """Every name ``ToolConfig(names=[...])`` currently resolves -- the
    built-ins plus anything added via ``register_tool``. (``load_skill`` is
    handled separately, since it needs a ``SkillLibrary``, and is always
    accepted even though it isn't in this list.)
    """
    return sorted(_TOOL_REGISTRY)


def resolve_tools(config: ToolConfig, skills: SkillLibrary | None = None) -> list[Tool]:
    """Turn a ``ToolConfig`` into a ``list[Tool]`` -- the ``ToolConfig``
    analogue of ``resolve_llm`` for ``LLMConfig``.

    ``config.names is None`` (the default) defers straight to
    ``default_tools()``, so a bare ``ToolConfig()`` behaves exactly like
    leaving ``AgentConfig.tools`` unset. Naming an explicit subset bypasses
    ``default_tools()`` entirely -- only the named tools are built, plus
    ``config.extra``, so ``submit`` must be named explicitly if you use this
    form (``Agent.__init__`` still enforces that it's present). Names are
    looked up in ``_TOOL_REGISTRY``, so anything added via ``register_tool``
    is selectable here too, from config alone.
    """
    if config.names is None:
        tools = default_tools(
            skills,
            include_code_execution=config.include_code_execution,
            include_data_analysis=config.include_data_analysis,
        )
        return [*tools, *config.extra]

    names = list(config.names)
    unknown = sorted(set(names) - set(_TOOL_REGISTRY) - {"load_skill"})
    if unknown:
        raise ValueError(
            f"ToolConfig.names contains unknown tool(s) {unknown}. "
            f"Known names: {sorted({*_TOOL_REGISTRY, 'load_skill'})}. "
            "Call register_tool() to add your own, or pass a live instance "
            "via ToolConfig.extra."
        )

    tools = [_TOOL_REGISTRY[name]() for name in names if name in _TOOL_REGISTRY]
    if "load_skill" in names:
        if skills is None:
            raise ValueError(
                "ToolConfig.names includes 'load_skill' but no SkillLibrary was provided."
            )
        tools.append(LoadSkillTool(skills))
    return [*tools, *config.extra]

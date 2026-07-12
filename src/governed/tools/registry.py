"""``ToolRegistry`` -- the single dispatch chokepoint every tool call passes through.

Four guarantees, so no individual tool has to provide them:

1. **Validated input.** Pydantic parses the arguments before ``run`` ever sees
   them. A failure becomes a model-facing ``ToolError`` naming the exact field.
2. **Gated side effects.** Anything in ``DANGEROUS`` passes through
   ``ctx.approve`` first. (Superseded by ``security.guardrails.GuardedRegistry``
   when guardrails are enabled -- see that module.)
3. **Bounded runtime.** Every call is wall-clocked in a worker thread and
   abandoned -- not killed, Python cannot kill a thread, but the caller moves on
   -- at the timeout.
4. **No escaping exceptions.** ``invoke`` never raises. A buggy tool cannot
   crash the loop.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import Any

from pydantic import ValidationError

from .base import DANGEROUS, Tool, ToolContext, ToolResult, ToolSpec
from .errors import ToolError, ToolErrorCode

__all__ = ["ToolRegistry"]


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = (), default_timeout_s: float = 60.0) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name!r}")
            self._tools[tool.name] = tool
        self.default_timeout_s = default_timeout_s
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="governed-tool")
        self._shutdown = False

    @property
    def names(self) -> set[str]:
        return set(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def schemas(self) -> list[dict[str, Any]]:
        return [t.spec().to_llm_schema() for t in self._tools.values()]

    def invoke(
        self,
        name: str,
        raw_args: dict[str, Any],
        ctx: ToolContext,
        timeout_s: float | None = None,
    ) -> ToolResult:
        started = time.monotonic()
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self._tools)) or "(none)"
            return ToolResult.failure(
                ToolError(
                    ToolErrorCode.NOT_FOUND,
                    f"No tool named `{name}` is registered.",
                    remediation=f"Available tools: {available}.",
                )
            )

        spec = tool.spec()

        try:
            args = tool.Input.model_validate(raw_args)
        except ValidationError as exc:
            return ToolResult.failure(
                ToolError(
                    ToolErrorCode.INVALID_INPUT,
                    f"Arguments for `{name}` failed validation:\n"
                    + "\n".join(
                        f"  - `{'.'.join(str(p) for p in e['loc'])}`: {e['msg']}"
                        for e in exc.errors()
                    ),
                    remediation="Re-read the tool's input schema and call it again with "
                    "corrected arguments.",
                ),
                duration_ms=_ms(started),
            )

        if spec.safety in DANGEROUS:
            try:
                approved = ctx.approve(spec, raw_args)
            except Exception as exc:
                return ToolResult.failure(
                    ToolError(
                        ToolErrorCode.INTERNAL,
                        f"Approval callback raised {type(exc).__name__}: {exc}",
                        remediation="Fix the approval callback; treat as denied for now.",
                    ),
                    duration_ms=_ms(started),
                )
            if not approved:
                return ToolResult.failure(
                    ToolError(
                        ToolErrorCode.APPROVAL_DENIED,
                        f"`{name}` requires human approval and was denied.",
                        remediation="Do not retry this call. Choose a different approach, "
                        "or call `submit` explaining what you could not do.",
                    ),
                    duration_ms=_ms(started),
                )

        timeout = timeout_s if timeout_s is not None else self.default_timeout_s
        if self._shutdown:
            # Agent._drive shuts this down in its `finally` at the end of
            # every run, so a bounded-lifetime executor never outlives the
            # run it served -- but the Agent itself is meant to be reusable
            # (`agent.resume(...)` right after `agent.run(...)`, same
            # instance, is the documented pattern). Recreate on next use
            # rather than making every caller construct a fresh Agent per
            # run just to get a live executor.
            self._executor = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="governed-tool"
            )
            self._shutdown = False
        future = self._executor.submit(self._run_safely, tool, args, ctx)
        try:
            result = future.result(timeout=timeout)
        except FutureTimeout:
            return ToolResult.failure(
                ToolError(
                    ToolErrorCode.TIMEOUT,
                    f"`{name}` did not complete within {timeout:g}s.",
                    remediation="Break the task into smaller calls, or increase "
                    "AgentConfig.tool_timeout_s if this is expected to be slow.",
                ),
                duration_ms=_ms(started),
            )
        result.duration_ms = _ms(started)
        return result

    def _run_safely(self, tool: Tool, args: Any, ctx: ToolContext) -> ToolResult:
        from .errors import ToolExecutionError

        try:
            return tool.run(args, ctx)
        except ToolExecutionError as exc:
            return ToolResult.failure(exc.error)
        except Exception as exc:
            return ToolResult.failure(
                ToolError(
                    ToolErrorCode.INTERNAL,
                    f"`{tool.name}` raised {type(exc).__name__}: {exc}",
                    remediation="This is a bug in the tool, not something you can fix by "
                    "retrying. Try a different approach or report it in your final answer.",
                )
            )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._shutdown = True


def _ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)

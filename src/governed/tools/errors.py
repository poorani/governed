"""Tool errors are *model-facing artifacts*, not stack traces.

Every failure is normalised into a ``ToolError`` carrying (a) a stable machine
code, (b) a human message, and (c) a ``remediation`` hint telling the model what
to do differently. Raw exceptions never reach the LLM: they would leak host
paths and waste tokens on tracebacks the model cannot act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"  # schema validation failed
    NOT_FOUND = "not_found"  # path/resource missing
    PERMISSION_DENIED = "permission_denied"  # sandbox or OS refusal
    UNSAFE_OPERATION = "unsafe_operation"  # blocked by the sandbox
    POLICY_VIOLATION = "policy_violation"  # blocked by a guardrail
    SAFETY_FALLBACK = "safety_fallback"  # redirected to a safer path by content safety
    APPROVAL_DENIED = "approval_denied"  # human said no
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"  # output/memory/size cap hit
    EXECUTION_FAILED = "execution_failed"  # tool ran, task failed (e.g. non-zero exit)
    DEPENDENCY_MISSING = "dependency_missing"
    INTERNAL = "internal"  # bug in the tool itself


#: Codes where retrying the identical call is pointless -- the agent must change
#: its approach. Surfaced to the model so it does not thrash.
TERMINAL_CODES: frozenset[ToolErrorCode] = frozenset(
    {
        ToolErrorCode.UNSAFE_OPERATION,
        ToolErrorCode.POLICY_VIOLATION,
        ToolErrorCode.SAFETY_FALLBACK,
        ToolErrorCode.APPROVAL_DENIED,
        ToolErrorCode.DEPENDENCY_MISSING,
    }
)


@dataclass
class ToolError:
    code: ToolErrorCode
    message: str
    remediation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def retryable(self) -> bool:
        return self.code not in TERMINAL_CODES

    def to_model_text(self) -> str:
        lines = [f"ERROR [{self.code.value}]: {self.message}"]
        if self.remediation:
            lines.append(f"How to fix: {self.remediation}")
        if not self.retryable:
            lines.append("Do not retry this call as-is. Change your approach.")
        for k, v in self.details.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "remediation": self.remediation,
            "retryable": self.retryable,
            "details": self.details,
        }


class ToolExecutionError(Exception):
    """Raise inside ``Tool.run`` to return a clean, structured failure.

    Anything else that escapes ``run`` is caught by the registry and wrapped as
    ``ToolErrorCode.INTERNAL``.
    """

    def __init__(
        self,
        code: ToolErrorCode,
        message: str,
        remediation: str | None = None,
        **details: Any,
    ) -> None:
        super().__init__(message)
        self.error = ToolError(code, message, remediation, details)

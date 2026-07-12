"""The three control tools: ``submit``, ``scratchpad``, ``load_skill``.

None of these touch the outside world. They exist to give the model explicit,
schema-checked ways to end a run, persist a fact past compaction, and pull a
skill body into context on demand.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, model_validator

from .base import Tool, ToolContext, ToolResult, ToolSafety
from .errors import ToolErrorCode, ToolExecutionError

if TYPE_CHECKING:
    from ..skills.loader import SkillLibrary

__all__ = ["RESERVED_SCRATCHPAD_PREFIX", "LoadSkillTool", "ScratchpadTool", "SubmitTool"]

#: Keys under this prefix are framework-owned (e.g. the cost ledger checkpoint).
#: The model cannot write or delete them, with or without guardrails enabled.
RESERVED_SCRATCHPAD_PREFIX = "_"


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------


class _SubmitInput(BaseModel):
    answer: str = Field(..., min_length=1, description="The final answer or report.")
    status: Literal["complete", "partial", "blocked"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="Your own calibration, 0-1.")
    evidence: list[str] = Field(
        default_factory=list, description="Concrete evidence backing the answer."
    )
    unmet_requirements: list[str] = Field(
        default_factory=list, description="Anything the goal asked for that you did not do."
    )

    @model_validator(mode="after")
    def _complete_means_complete(self) -> _SubmitInput:
        if self.status == "complete" and self.unmet_requirements:
            raise ValueError(
                "status='complete' with a non-empty unmet_requirements is a contradiction. "
                "Use status='partial' if something is missing, or clear "
                "unmet_requirements if it is not."
            )
        return self


class SubmitTool(Tool):
    name = "submit"
    description = (
        "End the run with a final, structured answer. Required: a status "
        "(complete/partial/blocked), a calibrated confidence, cited evidence, and "
        "anything you did not manage to do. This is the only way a run ends "
        "successfully -- call it as soon as the goal is met, not after."
    )
    safety = ToolSafety.READ_ONLY
    returns = "Confirmation that the run is ending."
    terminal = True
    Input = _SubmitInput

    def run(self, args: _SubmitInput, ctx: ToolContext) -> ToolResult:
        ctx.signals["submitted"] = {
            "answer": args.answer,
            "status": args.status,
            "confidence": args.confidence,
            "evidence": args.evidence,
            "unmet_requirements": args.unmet_requirements,
        }
        return ToolResult.success(f"Submitted with status={args.status}.")


# ---------------------------------------------------------------------------
# scratchpad
# ---------------------------------------------------------------------------


class _ScratchpadInput(BaseModel):
    action: Literal["read", "write", "delete", "list"]
    key: str | None = Field(None, description="Required for read/write/delete.")
    value: Any = Field(None, description="Required for write. Any JSON-serialisable value.")

    @model_validator(mode="after")
    def _require_key(self) -> _ScratchpadInput:
        if self.action in ("read", "write", "delete") and not self.key:
            raise ValueError(f"`key` is required when action={self.action!r}")
        return self


class ScratchpadTool(Tool):
    name = "scratchpad"
    description = (
        "Remember a fact across the whole run, immune to context compaction. Use "
        "this for anything you'll need many iterations from now: a schema, an ID, "
        "an approach that already failed. The transcript gets summarised as it "
        "grows; the scratchpad never does."
    )
    safety = ToolSafety.MUTATES_STATE
    returns = "The stored value, a confirmation, or the list of known keys."
    Input = _ScratchpadInput

    def run(self, args: _ScratchpadInput, ctx: ToolContext) -> ToolResult:
        if (
            args.key
            and args.key.startswith(RESERVED_SCRATCHPAD_PREFIX)
            and args.action
            in (
                "write",
                "delete",
            )
        ):
            raise ToolExecutionError(
                ToolErrorCode.UNSAFE_OPERATION,
                f"Scratchpad key {args.key!r} is reserved for the framework.",
                remediation="Choose a key that does not start with '_'.",
            )

        if args.action == "list":
            keys = sorted(
                k for k in ctx.scratchpad if not k.startswith(RESERVED_SCRATCHPAD_PREFIX)
            )
            return ToolResult.success(", ".join(keys) or "(empty)")

        assert args.key is not None  # enforced by _ScratchpadInput's model_validator

        if args.action == "read":
            if args.key not in ctx.scratchpad:
                raise ToolExecutionError(
                    ToolErrorCode.NOT_FOUND, f"No scratchpad key {args.key!r}."
                )
            return ToolResult.success(str(ctx.scratchpad[args.key]))

        if args.action == "write":
            ctx.scratchpad[args.key] = args.value
            return ToolResult.success(f"Stored `{args.key}`.")

        # delete
        ctx.scratchpad.pop(args.key, None)
        return ToolResult.success(f"Deleted `{args.key}` (if it existed).")


# ---------------------------------------------------------------------------
# load_skill
# ---------------------------------------------------------------------------


class _LoadSkillInput(BaseModel):
    name: str = Field(..., description="The skill's name, as it appears in the skill index.")


class LoadSkillTool(Tool):
    name = "load_skill"
    description = (
        "Pull the full body of a skill (a written procedure for a recurring task) "
        "into context. Check the skill index in the system prompt first -- only "
        "load a skill when its description matches what you are about to do."
    )
    safety = ToolSafety.READ_ONLY
    returns = "The skill's full Markdown body."
    Input = _LoadSkillInput

    def __init__(self, library: SkillLibrary) -> None:
        self.library = library

    def run(self, args: _LoadSkillInput, ctx: ToolContext) -> ToolResult:
        skill = self.library.get(args.name)
        if skill is None:
            available = ", ".join(sorted(self.library.names)) or "(none)"
            raise ToolExecutionError(
                ToolErrorCode.NOT_FOUND,
                f"No skill named {args.name!r}.",
                remediation=f"Available skills: {available}.",
            )
        return ToolResult.success(skill.body, data={"version": skill.version})

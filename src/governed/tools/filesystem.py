"""File system management, sandboxed to the workspace.

One of the three default tools. Everything routes through ``ToolContext.resolve``,
which is the sandbox: absolute paths, ``..`` traversal, and symlinks resolving
outside the workspace all fail before any I/O happens.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .base import Artifact, Tool, ToolContext, ToolResult, ToolSafety, read_bounded
from .errors import ToolErrorCode, ToolExecutionError

__all__ = ["FileSystemTool"]

_MAX_READ_BYTES = 200_000
_MAX_LIST_ENTRIES = 500


class _Input(BaseModel):
    operation: Literal["read", "write", "append", "list", "glob", "delete", "mkdir", "stat"]
    path: str = Field(..., description="Path relative to the workspace root.")
    content: str | None = Field(None, description="Required for write/append.")
    pattern: str | None = Field(
        None, description="Glob pattern, required for operation='glob'."
    )
    recursive: bool = Field(False, description="For delete/list: recurse into directories.")

    @model_validator(mode="after")
    def _require_content_for_writes(self) -> _Input:
        if self.operation in ("write", "append") and self.content is None:
            raise ValueError("`content` is required when operation='write' or 'append'")
        if self.operation == "glob" and not self.pattern:
            raise ValueError("`pattern` is required when operation='glob'")
        return self


class FileSystemTool(Tool):
    name = "file_system"
    description = (
        "Read, write, and manage files inside the sandboxed workspace. Operations: "
        "read, write, append, list, glob, delete, mkdir, stat. Paths are always "
        "relative to the workspace root; absolute paths and '..' are refused."
    )
    safety = ToolSafety.MUTATES_STATE
    returns = "File content, a directory listing, or a confirmation, depending on operation."
    Input = _Input

    def run(self, args: _Input, ctx: ToolContext) -> ToolResult:
        method: Callable[[_Input, ToolContext], ToolResult] = getattr(
            self, f"_{args.operation}"
        )
        return method(args, ctx)

    # -- reads --------------------------------------------------------------

    def _read(self, args: _Input, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        if not p.is_file():
            raise ToolExecutionError(
                ToolErrorCode.NOT_FOUND,
                f"No such file: {args.path}",
                remediation="Check the path with operation='list' or 'glob' first.",
            )
        text, truncated = read_bounded(p, _MAX_READ_BYTES)
        return ToolResult.success(text, data={"bytes": p.stat().st_size}, truncated=truncated)

    def _list(self, args: _Input, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        if not p.is_dir():
            raise ToolExecutionError(
                ToolErrorCode.NOT_FOUND, f"No such directory: {args.path}"
            )
        it = p.rglob("*") if args.recursive else p.iterdir()
        entries = sorted(
            str(e.relative_to(ctx.workspace)) + ("/" if e.is_dir() else "") for e in it
        )
        truncated = len(entries) > _MAX_LIST_ENTRIES
        entries = entries[:_MAX_LIST_ENTRIES]
        return ToolResult.success("\n".join(entries) or "(empty)", truncated=truncated)

    def _glob(self, args: _Input, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        assert args.pattern is not None  # enforced by _Input's model_validator
        matches = sorted(str(m.relative_to(ctx.workspace)) for m in p.glob(args.pattern))
        truncated = len(matches) > _MAX_LIST_ENTRIES
        matches = matches[:_MAX_LIST_ENTRIES]
        return ToolResult.success("\n".join(matches) or "(no matches)", truncated=truncated)

    def _stat(self, args: _Input, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        if not p.exists():
            raise ToolExecutionError(ToolErrorCode.NOT_FOUND, f"No such path: {args.path}")
        st = p.stat()
        kind = "directory" if p.is_dir() else "file"
        return ToolResult.success(
            f"{kind}, {st.st_size} bytes, modified {st.st_mtime:.0f}",
            data={"kind": kind, "bytes": st.st_size, "mtime": st.st_mtime},
        )

    # -- writes ---------------------------------------------------------

    def _write(self, args: _Input, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args.content or "", encoding="utf-8")
        return ToolResult.success(
            f"Wrote {len(args.content or '')} characters to {args.path}",
            artifacts=[Artifact(path=args.path, bytes=len(args.content or ""))],
        )

    def _append(self, args: _Input, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(args.content or "")
        return ToolResult.success(
            f"Appended {len(args.content or '')} characters to {args.path}"
        )

    def _mkdir(self, args: _Input, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        p.mkdir(parents=True, exist_ok=True)
        return ToolResult.success(f"Created directory {args.path}")

    def _delete(self, args: _Input, ctx: ToolContext) -> ToolResult:
        p = ctx.resolve(args.path)
        if not p.exists():
            raise ToolExecutionError(ToolErrorCode.NOT_FOUND, f"No such path: {args.path}")
        if p.is_dir():
            if not args.recursive and any(p.iterdir()):
                raise ToolExecutionError(
                    ToolErrorCode.UNSAFE_OPERATION,
                    f"{args.path} is a non-empty directory.",
                    remediation="Pass recursive=true to delete it and its contents.",
                )
            shutil.rmtree(p)
        else:
            p.unlink()
        return ToolResult.success(f"Deleted {args.path}")

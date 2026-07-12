from __future__ import annotations

from pathlib import Path

import pytest

from governed.tools.base import SandboxViolation, ToolContext
from governed.tools.control import ScratchpadTool, SubmitTool
from governed.tools.errors import ToolErrorCode
from governed.tools.filesystem import FileSystemTool
from governed.tools.registry import ToolRegistry

# -- sandbox ----------------------------------------------------------------


def test_resolve_rejects_absolute_path(ctx: ToolContext) -> None:
    with pytest.raises(SandboxViolation):
        ctx.resolve("/etc/passwd")


def test_resolve_rejects_parent_traversal(ctx: ToolContext) -> None:
    with pytest.raises(SandboxViolation):
        ctx.resolve("../outside.txt")


def test_resolve_rejects_symlink_escape(workspace: Path, ctx: ToolContext) -> None:
    outside = workspace.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope")
    link = workspace / "escape"
    link.symlink_to(outside)
    with pytest.raises(SandboxViolation):
        ctx.resolve("escape/secret.txt")


def test_resolve_allows_relative_path_inside_workspace(ctx: ToolContext) -> None:
    p = ctx.resolve("a/b/c.txt")
    assert p.parent.parent.name == "a"


# -- file_system --------------------------------------------------------


def test_write_then_read_round_trips(ctx: ToolContext) -> None:
    reg = ToolRegistry([FileSystemTool(), SubmitTool()])
    w = reg.invoke(
        "file_system", {"operation": "write", "path": "hello.txt", "content": "hi"}, ctx
    )
    assert w.ok
    r = reg.invoke("file_system", {"operation": "read", "path": "hello.txt"}, ctx)
    assert r.ok
    assert r.content == "hi"


def test_read_missing_file_is_not_found(ctx: ToolContext) -> None:
    reg = ToolRegistry([FileSystemTool(), SubmitTool()])
    result = reg.invoke("file_system", {"operation": "read", "path": "nope.txt"}, ctx)
    assert not result.ok
    assert result.error.code is ToolErrorCode.NOT_FOUND
    assert result.error.retryable  # NOT_FOUND isn't terminal -- a corrected path can retry


def test_delete_nonempty_dir_requires_recursive(ctx: ToolContext, workspace: Path) -> None:
    (workspace / "d").mkdir()
    (workspace / "d" / "f.txt").write_text("x")
    reg = ToolRegistry([FileSystemTool(), SubmitTool()])
    blocked = reg.invoke("file_system", {"operation": "delete", "path": "d"}, ctx)
    assert not blocked.ok
    ok = reg.invoke(
        "file_system", {"operation": "delete", "path": "d", "recursive": True}, ctx
    )
    assert ok.ok


# -- invalid input --------------------------------------------------------


def test_invalid_input_is_a_model_facing_error_not_a_crash(ctx: ToolContext) -> None:
    reg = ToolRegistry([FileSystemTool(), SubmitTool()])
    # missing `content`, required for a write
    result = reg.invoke("file_system", {"operation": "write", "path": "x.txt"}, ctx)
    assert not result.ok
    assert result.error.code is ToolErrorCode.INVALID_INPUT
    assert "content" in result.error.message


def test_unknown_tool_is_not_found(ctx: ToolContext) -> None:
    reg = ToolRegistry([SubmitTool()])
    result = reg.invoke("does_not_exist", {}, ctx)
    assert not result.ok
    assert result.error.code is ToolErrorCode.NOT_FOUND


# -- submit -----------------------------------------------------------------


def test_submit_complete_with_unmet_requirements_is_rejected(ctx: ToolContext) -> None:
    reg = ToolRegistry([SubmitTool()])
    result = reg.invoke(
        "submit",
        {
            "answer": "done",
            "status": "complete",
            "confidence": 0.9,
            "evidence": ["x"],
            "unmet_requirements": ["something left"],
        },
        ctx,
    )
    assert not result.ok
    assert result.error.code is ToolErrorCode.INVALID_INPUT


def test_submit_sets_signal(ctx: ToolContext) -> None:
    reg = ToolRegistry([SubmitTool()])
    result = reg.invoke(
        "submit",
        {"answer": "done", "status": "complete", "confidence": 0.9, "evidence": ["x"]},
        ctx,
    )
    assert result.ok
    assert ctx.signals["submitted"]["status"] == "complete"


# -- scratchpad reserved keys -------------------------------------------


def test_scratchpad_refuses_reserved_key_write(ctx: ToolContext) -> None:
    reg = ToolRegistry([ScratchpadTool()])
    result = reg.invoke("scratchpad", {"action": "write", "key": "_cost_usd", "value": 0}, ctx)
    assert not result.ok
    assert result.error.code is ToolErrorCode.UNSAFE_OPERATION

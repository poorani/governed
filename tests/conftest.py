from __future__ import annotations

from pathlib import Path

import pytest

from governed.tools.base import ToolContext


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture
def ctx(workspace: Path) -> ToolContext:
    return ToolContext(workspace=workspace, scratchpad={}, run_id="test-run", iteration=1)

"""DataAnalysisTool: every operation, every file format it dispatches on, and
both failure paths (pandas missing, file missing/unparseable/malformed
query). Requires pandas -- skipped entirely if it isn't installed, the same
optional-dependency posture the tool itself takes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from governed.tools.base import ToolContext
from governed.tools.data_analysis import DataAnalysisTool, _Input
from governed.tools.errors import ToolErrorCode, ToolExecutionError

pd = pytest.importorskip("pandas")

pytestmark = pytest.mark.filterwarnings("ignore")

_ROWS = [
    {"region": "US", "revenue": 100, "units": 3},
    {"region": "US", "revenue": 200, "units": 5},
    {"region": "EU", "revenue": 150, "units": 2},
    {"region": "EU", "revenue": None, "units": 1},
]


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ToolContext(workspace=ws, scratchpad={}, run_id="t", iteration=1)


@pytest.fixture
def csv_path(ctx: ToolContext) -> str:
    pd.DataFrame(_ROWS).to_csv(ctx.workspace / "sales.csv", index=False)
    return "sales.csv"


def _run(ctx: ToolContext, **kwargs: object) -> object:
    return DataAnalysisTool().run(_Input(**kwargs), ctx)  # type: ignore[arg-type]


def test_profile_reports_shape_and_columns(ctx: ToolContext, csv_path: str) -> None:
    result = _run(ctx, operation="profile", path=csv_path)
    assert result.ok
    assert "shape: 4 rows x 3 columns" in result.content
    assert "region" in result.content
    assert "nulls=" in result.content


def test_head_renders_and_flags_truncation(ctx: ToolContext, csv_path: str) -> None:
    result = _run(ctx, operation="head", path=csv_path, n=2)
    assert result.ok
    assert result.truncated
    assert "showing 2 of 4 rows" in result.content


def test_head_not_truncated_when_n_covers_everything(ctx: ToolContext, csv_path: str) -> None:
    result = _run(ctx, operation="head", path=csv_path, n=10)
    assert result.ok
    assert not result.truncated


def test_describe_returns_summary_statistics(ctx: ToolContext, csv_path: str) -> None:
    result = _run(ctx, operation="describe", path=csv_path)
    assert result.ok
    assert "revenue" in result.content


def test_query_filters_rows(ctx: ToolContext, csv_path: str) -> None:
    result = _run(ctx, operation="query", path=csv_path, expression="region == 'US'")
    assert result.ok
    assert "US" in result.content
    assert "EU" not in result.content


def test_query_requires_expression() -> None:
    with pytest.raises(ValueError, match="expression"):
        _Input(operation="query", path="x.csv")


def test_query_with_bad_expression_is_invalid_input(ctx: ToolContext, csv_path: str) -> None:
    with pytest.raises(ToolExecutionError) as exc_info:
        _run(ctx, operation="query", path=csv_path, expression="not a valid ( expr")
    assert exc_info.value.error.code is ToolErrorCode.INVALID_INPUT


def test_aggregate_groups_and_sums(ctx: ToolContext, csv_path: str) -> None:
    result = _run(
        ctx,
        operation="aggregate",
        path=csv_path,
        group_by="region",
        agg={"revenue": "sum"},
    )
    assert result.ok
    assert "region" in result.content


def test_aggregate_requires_group_by_and_agg() -> None:
    with pytest.raises(ValueError, match="group_by"):
        _Input(operation="aggregate", path="x.csv")


def test_aggregate_with_unknown_column_is_invalid_input(
    ctx: ToolContext, csv_path: str
) -> None:
    with pytest.raises(ToolExecutionError) as exc_info:
        _run(
            ctx,
            operation="aggregate",
            path=csv_path,
            group_by="not_a_column",
            agg={"revenue": "sum"},
        )
    assert exc_info.value.error.code is ToolErrorCode.INVALID_INPUT


def test_value_counts_counts_each_value(ctx: ToolContext, csv_path: str) -> None:
    result = _run(ctx, operation="value_counts", path=csv_path, column="region")
    assert result.ok
    assert "US" in result.content and "EU" in result.content


def test_value_counts_requires_column() -> None:
    with pytest.raises(ValueError, match="column"):
        _Input(operation="value_counts", path="x.csv")


def test_value_counts_with_unknown_column_is_invalid_input(
    ctx: ToolContext, csv_path: str
) -> None:
    with pytest.raises(ToolExecutionError) as exc_info:
        _run(ctx, operation="value_counts", path=csv_path, column="nope")
    assert exc_info.value.error.code is ToolErrorCode.INVALID_INPUT


def test_correlate_all_numeric_columns(ctx: ToolContext, csv_path: str) -> None:
    result = _run(ctx, operation="correlate", path=csv_path)
    assert result.ok
    assert "revenue" in result.content


def test_correlate_restricted_to_named_columns(ctx: ToolContext, csv_path: str) -> None:
    result = _run(ctx, operation="correlate", path=csv_path, columns=["revenue", "units"])
    assert result.ok


def test_missing_file_is_not_found(ctx: ToolContext) -> None:
    with pytest.raises(ToolExecutionError) as exc_info:
        _run(ctx, operation="profile", path="nope.csv")
    assert exc_info.value.error.code is ToolErrorCode.NOT_FOUND


def test_unsupported_extension_is_invalid_input(ctx: ToolContext) -> None:
    (ctx.workspace / "data.txt").write_text("not tabular")
    with pytest.raises(ToolExecutionError) as exc_info:
        _run(ctx, operation="profile", path="data.txt")
    assert exc_info.value.error.code is ToolErrorCode.INVALID_INPUT


def test_malformed_file_is_execution_failed(ctx: ToolContext) -> None:
    (ctx.workspace / "bad.parquet").write_bytes(b"not actually parquet")
    with pytest.raises(ToolExecutionError) as exc_info:
        _run(ctx, operation="profile", path="bad.parquet")
    assert exc_info.value.error.code is ToolErrorCode.EXECUTION_FAILED


def test_tsv_uses_tab_separator(ctx: ToolContext) -> None:
    (ctx.workspace / "sales.tsv").write_text("region\trevenue\nUS\t100\nEU\t150\n")
    result = _run(ctx, operation="head", path="sales.tsv", n=10)
    assert result.ok
    assert "US" in result.content


def test_jsonl_reads_line_delimited_records(ctx: ToolContext) -> None:
    (ctx.workspace / "sales.jsonl").write_text(
        '{"region": "US", "revenue": 100}\n{"region": "EU", "revenue": 150}\n'
    )
    result = _run(ctx, operation="head", path="sales.jsonl", n=10)
    assert result.ok
    assert "US" in result.content


def test_parquet_round_trips(ctx: ToolContext) -> None:
    pd.DataFrame(_ROWS).to_parquet(ctx.workspace / "sales.parquet")
    result = _run(ctx, operation="head", path="sales.parquet", n=10)
    assert result.ok
    assert "region" in result.content


def test_pandas_missing_is_dependency_missing(
    ctx: ToolContext, csv_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pandas", None)
    with pytest.raises(ToolExecutionError) as exc_info:
        _run(ctx, operation="profile", path=csv_path)
    assert exc_info.value.error.code is ToolErrorCode.DEPENDENCY_MISSING

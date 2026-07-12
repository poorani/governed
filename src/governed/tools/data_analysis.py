"""Bounded, aggregate-don't-dump exploration of tabular data.

An agent *could* do all of this in ``execute_code`` -- and would burn an
iteration on read-and-print boilerplate, then dump a 50k-row dataframe straight
into its own context. This tool exists to bound the output: 50 rows max,
explicit truncation notices, and operations (``profile``, ``aggregate``,
``value_counts``, ``correlate``) that summarise rather than enumerate.

Requires ``pandas`` (``pip install 'governed[data]'``). Its absence is a
``dependency_missing`` error, not an import crash at module load, so the rest of
the framework works without it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .base import Tool, ToolContext, ToolResult, ToolSafety
from .errors import ToolErrorCode, ToolExecutionError

__all__ = ["DataAnalysisTool"]

_MAX_ROWS = 50
_READERS = {
    ".csv": "read_csv",
    ".tsv": "read_csv",
    ".json": "read_json",
    ".jsonl": "read_json",
    ".parquet": "read_parquet",
    ".xlsx": "read_excel",
    ".xls": "read_excel",
}


class _Input(BaseModel):
    operation: Literal[
        "profile", "head", "describe", "query", "aggregate", "value_counts", "correlate"
    ]
    path: str = Field(..., description="Path to a tabular file, relative to the workspace.")
    n: int = Field(10, ge=1, le=_MAX_ROWS, description="Row count for 'head'.")
    expression: str | None = Field(
        None, description="Pandas query expression, required for operation='query'."
    )
    column: str | None = Field(None, description="Target column for value_counts.")
    columns: list[str] | None = Field(None, description="Column subset for correlate.")
    group_by: str | list[str] | None = Field(None, description="Group key(s) for aggregate.")
    agg: dict[str, str] | None = Field(
        None, description="{column: function} for aggregate, e.g. {'revenue': 'sum'}."
    )

    @model_validator(mode="after")
    def _require_op_args(self) -> _Input:
        if self.operation == "query" and not self.expression:
            raise ValueError("`expression` is required when operation='query'")
        if self.operation == "value_counts" and not self.column:
            raise ValueError("`column` is required when operation='value_counts'")
        if self.operation == "aggregate" and not (self.group_by and self.agg):
            raise ValueError("`group_by` and `agg` are required when operation='aggregate'")
        return self


class DataAnalysisTool(Tool):
    name = "analyze_data"
    description = (
        "Profile and query tabular data (csv, tsv, json, jsonl, parquet, xlsx) without "
        "loading it into your own context by hand. Operations: profile (shape, dtypes, "
        "nulls, cardinality -- always start here), head, describe, query (pandas "
        "boolean expression), aggregate (group_by + agg), value_counts, correlate. "
        "Output is capped at 50 rows; use aggregate or value_counts instead of "
        "dumping raw rows for anything larger."
    )
    safety = ToolSafety.READ_ONLY
    returns = "A bounded, text-rendered table or summary. Never the full dataset."
    Input = _Input

    def run(self, args: _Input, ctx: ToolContext) -> ToolResult:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ToolExecutionError(
                ToolErrorCode.DEPENDENCY_MISSING,
                "pandas is not installed.",
                remediation="Install with `pip install 'governed[data]'`, or use "
                "execute_code for a one-off if that dependency can't be added.",
            ) from exc

        p = ctx.resolve(args.path)
        if not p.is_file():
            raise ToolExecutionError(ToolErrorCode.NOT_FOUND, f"No such file: {args.path}")
        df = self._load(pd, p)

        method: Callable[[Any, Any, _Input], ToolResult] = getattr(self, f"_{args.operation}")
        return method(pd, df, args)

    # -- loading --------------------------------------------------------

    def _load(self, pd: Any, p: Path) -> Any:
        suffix = p.suffix.lower()
        reader_name = _READERS.get(suffix)
        if reader_name is None:
            raise ToolExecutionError(
                ToolErrorCode.INVALID_INPUT,
                f"Unsupported file type: {suffix or '(none)'}",
                remediation=f"Supported: {', '.join(sorted(_READERS))}.",
            )
        try:
            if suffix == ".tsv":
                return pd.read_csv(p, sep="\t")
            if suffix == ".jsonl":
                return pd.read_json(p, lines=True)
            return getattr(pd, reader_name)(p)
        except Exception as exc:
            raise ToolExecutionError(
                ToolErrorCode.EXECUTION_FAILED,
                f"Failed to parse {p.name} as {suffix}: {exc}",
                remediation="Check the file is well-formed for its extension.",
            ) from exc

    # -- operations -------------------------------------------------------

    def _profile(self, pd: Any, df: Any, args: _Input) -> ToolResult:
        lines = [f"shape: {df.shape[0]} rows x {df.shape[1]} columns", "", "columns:"]
        for col in df.columns:
            s = df[col]
            lines.append(
                f"  {col:<24} {s.dtype!s:<10} nulls={s.isna().sum():<6} unique={s.nunique()}"
            )
        return ToolResult.success("\n".join(lines))

    def _head(self, pd: Any, df: Any, args: _Input) -> ToolResult:
        return self._render(df.head(args.n), len(df))

    def _describe(self, pd: Any, df: Any, args: _Input) -> ToolResult:
        return ToolResult.success(df.describe(include="all").to_string())

    def _query(self, pd: Any, df: Any, args: _Input) -> ToolResult:
        try:
            result = df.query(args.expression, engine="python")
        except Exception as exc:
            raise ToolExecutionError(
                ToolErrorCode.INVALID_INPUT,
                f"Query expression failed: {exc}",
                remediation="Use pandas query syntax, e.g. "
                "\"revenue > 1000 and region == 'US'\".",
            ) from exc
        return self._render(result.head(_MAX_ROWS), len(result))

    def _aggregate(self, pd: Any, df: Any, args: _Input) -> ToolResult:
        try:
            grouped = df.groupby(args.group_by).agg(args.agg).reset_index()
        except Exception as exc:
            raise ToolExecutionError(
                ToolErrorCode.INVALID_INPUT,
                f"Aggregation failed: {exc}",
                remediation="Check group_by and agg reference real column names.",
            ) from exc
        return self._render(grouped.head(_MAX_ROWS), len(grouped))

    def _value_counts(self, pd: Any, df: Any, args: _Input) -> ToolResult:
        if args.column not in df.columns:
            raise ToolExecutionError(
                ToolErrorCode.INVALID_INPUT, f"No column named {args.column!r}."
            )
        counts = df[args.column].value_counts().head(_MAX_ROWS)
        return self._render(counts.reset_index(), len(counts))

    def _correlate(self, pd: Any, df: Any, args: _Input) -> ToolResult:
        subset = df[args.columns] if args.columns else df.select_dtypes(include="number")
        return ToolResult.success(subset.corr(numeric_only=True).to_string())

    # -- rendering --------------------------------------------------------

    def _render(self, frame: Any, total_rows: int) -> ToolResult:
        truncated = total_rows > len(frame)
        text = frame.to_string(index=False)
        if truncated:
            text += f"\n\n[showing {len(frame)} of {total_rows} rows]"
        return ToolResult.success(text, truncated=truncated)

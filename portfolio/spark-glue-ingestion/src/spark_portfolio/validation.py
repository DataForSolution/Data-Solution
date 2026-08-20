"""DataFrame contract checks that work with Spark DataFrames and light test doubles."""
from __future__ import annotations

from collections.abc import Mapping, Sequence


def _row_mapping(row, columns: Sequence[str]) -> dict[str, object]:
    if row is None:
        return {}
    if hasattr(row, "asDict"):
        return dict(row.asDict(recursive=True))
    if isinstance(row, Mapping):
        return dict(row)
    try:
        return {col: row[i] for i, col in enumerate(columns)}
    except Exception as exc:
        raise TypeError("first row must be Spark Row-like, mapping-like, or sequence-like") from exc


def header_match_fraction(row, columns: Sequence[str]) -> float:
    """Fraction of non-null first-row strings that equal their column names case-insensitively."""
    mapping = _row_mapping(row, columns)
    compared = 0
    matched = 0
    for col in columns:
        value = mapping.get(col)
        if value is None:
            continue
        if isinstance(value, str):
            compared += 1
            normalized_value = value.strip().lower()
            normalized_col = str(col).strip().lower()
            if normalized_value == normalized_col:
                matched += 1
    return matched / compared if compared else 0.0


def dataframe_audit(df, *, header_match_threshold: float = 0.5) -> dict[str, object]:
    """Collect bounded ingestion diagnostics without converting the dataset to pandas."""
    if not 0 <= header_match_threshold <= 1:
        raise ValueError("header_match_threshold must be within [0, 1]")
    columns = list(df.columns)
    if not columns:
        raise ValueError("DataFrame must contain at least one column")
    row_count = int(df.count())
    first = df.first() if row_count else None
    match_fraction = header_match_fraction(first, columns) if first is not None else 0.0
    return {
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "first_row": _row_mapping(first, columns),
        "header_match_fraction": match_fraction,
        "embedded_header_suspected": bool(first is not None and match_fraction >= header_match_threshold),
    }


def validate_dataframe_contract(
    df,
    *,
    required_columns: Sequence[str] = (),
    expected_row_count: int | None = None,
    header_match_threshold: float = 0.5,
) -> dict[str, object]:
    """Fail closed on missing columns, count mismatch, or an embedded header row."""
    audit = dataframe_audit(df, header_match_threshold=header_match_threshold)
    missing = [c for c in required_columns if c not in audit["columns"]]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if expected_row_count is not None and audit["row_count"] != int(expected_row_count):
        raise ValueError(
            f"unexpected row count: observed {audit['row_count']}, expected {int(expected_row_count)}"
        )
    if audit["embedded_header_suspected"]:
        raise ValueError(
            "embedded header row suspected: first-row string values match DataFrame column names"
        )
    return audit

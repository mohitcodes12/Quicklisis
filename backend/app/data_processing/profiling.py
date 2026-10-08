from __future__ import annotations

import re

import numpy as np
import pandas as pd


DATE_NAME_HINTS = re.compile(
    r"(^|[_\-\s])(date|time|datetime|timestamp|dob|birth|joined|created|updated|start|end)([_\-\s]|$)",
    re.IGNORECASE,
)


def _detect_excel_header_row(path: str) -> int:
    """
    Detect a simple title/preamble row before the real Excel header.

    This is intentionally conservative:
    - Normal Excel files whose first row is already the header keep header=0.
    - If an early row has one or two populated cells and the following row
      clearly contains multiple populated cells, the following row is used
      as the header.
    """
    preview = pd.read_excel(path, header=None, nrows=10)

    if preview.empty:
        return 0

    max_rows = min(len(preview), 9)

    for row_index in range(max_rows):
        current = preview.iloc[row_index]
        current_non_empty = int(current.notna().sum())

        if row_index + 1 >= len(preview):
            break

        next_row = preview.iloc[row_index + 1]
        next_non_empty = int(next_row.notna().sum())

        # Most common case: a title such as
        # "Tanvi's Chocolate Sales Tracker" in the first row,
        # followed by the real column headers.
        if current_non_empty <= 1 and next_non_empty >= 2:
            return row_index + 1

        # Also handle a short two-cell preamble/title followed by a
        # substantially wider header row.
        if (
            current_non_empty <= 2
            and next_non_empty >= 3
            and next_non_empty >= current_non_empty + 2
        ):
            return row_index + 1

    return 0


def load_dataframe(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xlsx", ".xls")):
        header_row = _detect_excel_header_row(path)
        return pd.read_excel(path, header=header_row)
    raise ValueError("Unsupported file type. Use CSV or XLSX.")


def _is_date_like(series: pd.Series, name: str = "") -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    non_null = series.dropna()
    if non_null.empty:
        return False

    # Excel date serials are numeric. We only classify them as dates when
    # the column name gives a strong enough date/time signal, preventing
    # ordinary numbers such as salaries from being misclassified.
    if pd.api.types.is_numeric_dtype(series):
        if DATE_NAME_HINTS.search(str(name)):
            values = pd.to_numeric(non_null, errors="coerce")
            values = values[np.isfinite(values)]
            if len(values):
                return bool(values.between(20000, 60000).mean() >= 0.8)
        return False

    sample = non_null.astype(str).head(200)
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.8)


def infer_type(series: pd.Series, name: str = "") -> str:
    if _is_date_like(series, name):
        return "date"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    unique = series.nunique(dropna=True)
    threshold = max(20, min(50, int(len(series) * 0.2)))
    return "categorical" if unique <= threshold else "text"


def profile_dataframe(df: pd.DataFrame) -> dict:
    columns = []
    for col in df.columns:
        s = df[col]
        name = str(col)
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        col_type = infer_type(s, name)
        is_id = unique == len(df) and len(df) > 0

        item = {
            "name": name,
            "type": col_type,
            "missing": missing,
            "missing_percent": round((missing / len(df) * 100), 2) if len(df) else 0,
            "unique": unique,
            "potential_id": is_id,
        }

        if col_type in {"integer", "float", "numeric"}:
            numeric = pd.to_numeric(s, errors="coerce")
            item["stats"] = {
                "mean": _safe_float(numeric.mean()),
                "median": _safe_float(numeric.median()),
                "min": _safe_float(numeric.min()),
                "max": _safe_float(numeric.max()),
                "std": _safe_float(numeric.std()),
            }

        columns.append(item)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_percent": round(df.duplicated().mean() * 100, 2) if len(df) else 0,
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024**2, 3),
        "columns_detail": columns,
    }


def _safe_float(value):
    if pd.isna(value) or not np.isfinite(value):
        return None
    return round(float(value), 4)

from __future__ import annotations

import re
from typing import Any

import pandas as pd


# ============================================================
# QUICKLISIS CLEANING ENGINE
# ============================================================
#
# Philosophy:
# 1. Detect problems
# 2. Explain the problem
# 3. Recommend an action
# 4. Allow the application to apply safe cleaning
#
# IMPORTANT:
# We do not blindly delete useful data.
# ============================================================


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def _is_empty_value(value: Any) -> bool:
    """Return True when a scalar value should be treated as missing."""
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    try:
        result = pd.isna(value)
        return bool(result) if not hasattr(result, "__len__") else False
    except (TypeError, ValueError):
        return False


def _clean_column_name(column: Any) -> str:
    """
    Normalize a column name.

    Examples:
        " Customer Name " -> "customer_name"
        "Sales Amount"    -> "sales_amount"
    """
    name = str(column).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^a-zA-Z0-9_]", "", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_").lower()

    return name or "column"


def _unique_column_names(columns: list[str]) -> list[str]:
    """Make cleaned column names unique."""
    result: list[str] = []
    used: dict[str, int] = {}

    for column in columns:
        if column not in used:
            used[column] = 0
            result.append(column)
        else:
            used[column] += 1
            result.append(f"{column}_{used[column]}")

    return result


def _looks_like_unnamed(column: Any) -> bool:
    """Detect Excel-generated columns such as Unnamed: 0."""
    return bool(re.match(r"^Unnamed:\s*\d+$", str(column), re.IGNORECASE))


def _is_text_series(series: pd.Series) -> bool:
    """Return True for object/string columns."""
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def _non_empty_series(series: pd.Series) -> pd.Series:
    """Return values that are not empty according to QuickLisis rules."""
    return series[~series.map(_is_empty_value)]


def _is_numeric_like(series: pd.Series) -> bool:
    """Check whether most non-empty values can be interpreted as numbers."""
    non_empty = _non_empty_series(series)

    if non_empty.empty:
        return False

    converted = pd.to_numeric(
        non_empty.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .str.replace(r"^\$", "", regex=True),
        errors="coerce",
    )

    return bool(converted.notna().mean() >= 0.80)


def _is_date_like(series: pd.Series) -> bool:
    """Check whether most non-empty values can be interpreted as dates."""
    non_empty = _non_empty_series(series)

    if non_empty.empty:
        return False

    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    sample = non_empty.astype(str).head(100)

    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        parsed = pd.to_datetime(sample, errors="coerce")

    return bool(parsed.notna().mean() >= 0.80)


# ============================================================
# DETECTION
# ============================================================

def detect_empty_columns(df: pd.DataFrame) -> list[dict]:
    """Detect columns containing no usable data."""
    issues = []

    for column in df.columns:
        empty_count = int(df[column].map(_is_empty_value).sum())

        if empty_count == len(df):
            issues.append(
                {
                    "type": "empty_column",
                    "severity": "high",
                    "column": str(column),
                    "count": empty_count,
                    "message": f"Column '{column}' contains no usable values.",
                    "suggestion": "Remove this column.",
                    "action": "remove",
                }
            )

    return issues


def detect_unnamed_columns(df: pd.DataFrame) -> list[dict]:
    """
    Detect Excel-generated Unnamed: X columns.

    Empty unnamed columns are handled by detect_empty_columns.
    Unnamed columns containing data are flagged for manual review.
    """
    issues = []

    for column in df.columns:
        if not _looks_like_unnamed(column):
            continue

        non_empty_count = int(_non_empty_series(df[column]).shape[0])

        if non_empty_count == 0:
            # Avoid duplicate reporting with empty_column.
            continue

        issues.append(
            {
                "type": "unnamed_column",
                "severity": "medium",
                "column": str(column),
                "count": non_empty_count,
                "message": (
                    f"'{column}' contains data even though it looks like "
                    "an Excel-generated unnamed column."
                ),
                "suggestion": "Review this column before removing it.",
                "action": "review",
            }
        )

    return issues


def detect_empty_rows(df: pd.DataFrame) -> list[dict]:
    """Detect rows where every value is empty."""
    if df.empty:
        return []

    empty_mask = df.apply(
        lambda row: all(_is_empty_value(value) for value in row),
        axis=1,
    )
    count = int(empty_mask.sum())

    if count == 0:
        return []

    return [
        {
            "type": "empty_rows",
            "severity": "high",
            "column": None,
            "count": count,
            "message": f"{count} completely empty row(s) found.",
            "suggestion": "Remove completely empty rows.",
            "action": "remove",
        }
    ]


def detect_duplicates(df: pd.DataFrame) -> list[dict]:
    """Detect exact duplicate rows."""
    if df.empty:
        return []

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        return []

    return [
        {
            "type": "duplicate_rows",
            "severity": "medium",
            "column": None,
            "count": duplicate_count,
            "message": f"{duplicate_count} duplicate row(s) found.",
            "suggestion": "Remove exact duplicate rows.",
            "action": "remove",
        }
    ]


def detect_missing_values(df: pd.DataFrame) -> list[dict]:
    """
    Detect missing values column-by-column.

    Missing values are reported but NOT automatically filled.
    """
    issues = []

    if df.empty:
        return issues

    for column in df.columns:
        series = df[column]
        missing_count = int(series.map(_is_empty_value).sum())

        if missing_count == 0:
            continue

        percentage = round((missing_count / len(df)) * 100, 2)

        if percentage >= 50:
            severity = "high"
        elif percentage >= 20:
            severity = "medium"
        else:
            severity = "low"

        if pd.api.types.is_numeric_dtype(series):
            suggestion = (
                "Review missing values; median imputation may be appropriate."
            )
        elif _is_date_like(series):
            suggestion = (
                "Review missing dates; forward-fill or domain-specific "
                "replacement may be appropriate."
            )
        else:
            suggestion = (
                "Review missing values; mode or explicit 'Unknown' "
                "may be appropriate."
            )

        issues.append(
            {
                "type": "missing_values",
                "severity": severity,
                "column": str(column),
                "count": missing_count,
                "percentage": percentage,
                "message": (
                    f"Column '{column}' contains {missing_count} "
                    f"missing value(s) ({percentage}%)."
                ),
                "suggestion": suggestion,
                "action": "review",
            }
        )

    return issues


def detect_whitespace(df: pd.DataFrame) -> list[dict]:
    """Detect leading/trailing whitespace in text values."""
    issues = []

    for column in df.columns:
        series = df[column]

        if not _is_text_series(series):
            continue

        non_empty = _non_empty_series(series)

        if non_empty.empty:
            continue

        stripped = non_empty.astype(str).str.strip()
        count = int((stripped != non_empty.astype(str)).sum())

        if count > 0:
            issues.append(
                {
                    "type": "whitespace",
                    "severity": "low",
                    "column": str(column),
                    "count": count,
                    "message": (
                        f"Column '{column}' contains {count} value(s) "
                        "with unnecessary whitespace."
                    ),
                    "suggestion": "Trim leading and trailing whitespace.",
                    "action": "trim",
                }
            )

    return issues


def detect_column_names(df: pd.DataFrame) -> list[dict]:
    """Detect column names that can be normalized."""
    issues = []

    cleaned_names = [_clean_column_name(column) for column in df.columns]

    for original, cleaned in zip(df.columns, cleaned_names):
        original_string = str(original)

        if original_string != cleaned:
            issues.append(
                {
                    "type": "column_name",
                    "severity": "low",
                    "column": original_string,
                    "count": 1,
                    "message": f"Column name '{original_string}' can be normalized.",
                    "suggestion": f"Rename to '{cleaned}'.",
                    "action": "rename",
                    "new_name": cleaned,
                }
            )

    return issues


def detect_numeric_like_columns(df: pd.DataFrame) -> list[dict]:
    """Detect text columns that mostly contain numbers."""
    issues = []

    for column in df.columns:
        series = df[column]

        if not _is_text_series(series):
            continue

        if not _is_numeric_like(series):
            continue

        issues.append(
            {
                "type": "numeric_type",
                "severity": "medium",
                "column": str(column),
                "count": int(_non_empty_series(series).shape[0]),
                "message": (
                    f"Column '{column}' is stored as text but mostly "
                    "contains numeric values."
                ),
                "suggestion": "Convert the column to numeric values.",
                "action": "convert_numeric",
            }
        )

    return issues


def detect_date_columns(df: pd.DataFrame) -> list[dict]:
    """Detect text columns that mostly contain dates."""
    issues = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        if not _is_text_series(series):
            continue

        if not _is_date_like(series):
            continue

        issues.append(
            {
                "type": "date_type",
                "severity": "medium",
                "column": str(column),
                "count": int(_non_empty_series(series).shape[0]),
                "message": (
                    f"Column '{column}' appears to contain date values "
                    "but is not stored as a date type."
                ),
                "suggestion": "Convert values to a standard datetime format.",
                "action": "convert_date",
            }
        )

    return issues


# ============================================================
# MAIN DETECTION FUNCTION
# ============================================================

def detect_cleaning_issues(df: pd.DataFrame) -> dict:
    """
    Run the complete QuickLisis cleaning inspection.

    This function DOES NOT modify the DataFrame.

    Redundant issues are suppressed. For example, a completely empty
    'Unnamed: 0' column is reported as one removal issue instead of
    being reported again as unnamed, missing, and rename issues.
    """
    if df.empty:
        return {
            "issues": [],
            "summary": {
                "total_issues": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            },
            "message": "Dataset is empty.",
        }

    issues: list[dict] = []
    handled_columns: set[str] = set()

    # Structural issues first.
    empty_columns = detect_empty_columns(df)
    issues.extend(empty_columns)
    handled_columns.update(str(issue["column"]) for issue in empty_columns)

    # Do not duplicate empty unnamed columns.
    for issue in detect_unnamed_columns(df):
        column = str(issue["column"])
        if column not in handled_columns:
            issues.append(issue)

    # Dataset-level issues.
    issues.extend(detect_empty_rows(df))
    issues.extend(detect_duplicates(df))

    # Column-level issues.
    for issue in detect_missing_values(df):
        column = issue.get("column")
        if column is not None and str(column) in handled_columns:
            continue
        issues.append(issue)

    for issue in detect_whitespace(df):
        column = str(issue["column"])
        if column not in handled_columns:
            issues.append(issue)

    for issue in detect_column_names(df):
        column = str(issue["column"])
        if column not in handled_columns:
            issues.append(issue)

    for issue in detect_numeric_like_columns(df):
        column = str(issue["column"])
        if column not in handled_columns:
            issues.append(issue)

    for issue in detect_date_columns(df):
        column = str(issue["column"])
        if column not in handled_columns:
            issues.append(issue)

    summary = {
        "total_issues": len(issues),
        "high": sum(issue["severity"] == "high" for issue in issues),
        "medium": sum(issue["severity"] == "medium" for issue in issues),
        "low": sum(issue["severity"] == "low" for issue in issues),
    }

    return {
        "issues": issues,
        "summary": summary,
        "message": (
            f"QuickLisis detected {summary['total_issues']} potential "
            "data-quality issue(s)."
        ),
    }


# ============================================================
# CLEANING OPERATIONS
# ============================================================

def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns containing no usable values."""
    result = df.copy()

    empty_columns = [
        column
        for column in result.columns
        if result[column].map(_is_empty_value).all()
    ]

    if empty_columns:
        result = result.drop(columns=empty_columns)

    return result


def remove_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where every value is empty."""
    if df.empty:
        return df.copy()

    result = df.copy()

    mask = result.apply(
        lambda row: all(_is_empty_value(value) for value in row),
        axis=1,
    )

    return result.loc[~mask].reset_index(drop=True)


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def trim_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """Trim unnecessary whitespace from text columns."""
    result = df.copy()

    for column in result.columns:
        series = result[column]

        if _is_text_series(series):
            result[column] = series.map(
                lambda value: value.strip()
                if isinstance(value, str)
                else value
            )

    return result


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all column names and keep them unique."""
    result = df.copy()

    cleaned_names = [
        _clean_column_name(column)
        for column in result.columns
    ]

    result.columns = _unique_column_names(cleaned_names)

    return result


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert strongly numeric-like text columns to numeric."""
    result = df.copy()

    for column in result.columns:
        series = result[column]

        if not _is_text_series(series):
            continue

        if not _is_numeric_like(series):
            continue

        converted = pd.to_numeric(
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .str.replace(r"^\$", "", regex=True),
            errors="coerce",
        )

        original_valid = int(_non_empty_series(series).shape[0])
        converted_valid = int(converted.notna().sum())

        if original_valid > 0 and converted_valid / original_valid >= 0.80:
            result[column] = converted

    return result


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert strongly date-like text columns to datetime."""
    result = df.copy()

    for column in result.columns:
        series = result[column]

        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        if not _is_text_series(series):
            continue

        if not _is_date_like(series):
            continue

        try:
            converted = pd.to_datetime(
                series,
                errors="coerce",
                format="mixed",
            )
        except (TypeError, ValueError):
            converted = pd.to_datetime(series, errors="coerce")

        original_valid = int(_non_empty_series(series).shape[0])
        converted_valid = int(converted.notna().sum())

        if original_valid > 0 and converted_valid / original_valid >= 0.80:
            result[column] = converted

    return result


# ============================================================
# COMPLETE CLEANING FUNCTION
# ============================================================

def clean_dataframe(
    df: pd.DataFrame,
    *,
    remove_empty: bool = True,
    remove_duplicates_flag: bool = True,
    trim_text: bool = True,
    normalize_names: bool = True,
    convert_numeric: bool = True,
    convert_dates: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply safe automatic cleaning operations.

    Missing-value imputation is intentionally NOT automatic.
    """
    original_rows = len(df)
    original_columns = len(df.columns)
    original_duplicate_count = int(df.duplicated().sum())

    result = df.copy()

    if remove_empty:
        result = remove_empty_columns(result)
        result = remove_empty_rows(result)

    if remove_duplicates_flag:
        result = remove_duplicates(result)

    if trim_text:
        result = trim_text_values(result)

    if normalize_names:
        result = normalize_column_names(result)

    if convert_numeric:
        result = convert_numeric_columns(result)

    if convert_dates:
        result = convert_date_columns(result)

    result = result.reset_index(drop=True)

    summary = {
        "original_rows": original_rows,
        "cleaned_rows": len(result),
        "rows_removed": original_rows - len(result),
        "original_columns": original_columns,
        "cleaned_columns": len(result.columns),
        "columns_removed": original_columns - len(result.columns),
        "duplicates_removed": original_duplicate_count,
    }

    return result, summary


# ============================================================
# SAFE PREVIEW
# ============================================================

def cleaning_preview(df: pd.DataFrame) -> dict:
    """
    Generate a before/after-style preview without modifying
    the original DataFrame.
    """
    issues = detect_cleaning_issues(df)
    cleaned_df, summary = clean_dataframe(df)

    before = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": [str(c) for c in df.columns],
    }

    after = {
        "rows": len(cleaned_df),
        "columns": len(cleaned_df.columns),
        "column_names": [str(c) for c in cleaned_df.columns],
    }

    return {
        "issues": issues["issues"],
        "issue_summary": issues["summary"],
        "before": before,
        "after": after,
        "cleaning_summary": summary,
    }

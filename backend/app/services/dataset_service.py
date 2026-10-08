from pathlib import Path
import uuid

import pandas as pd

from app.data_processing.profiling import load_dataframe, profile_dataframe
from app.analytics.eda import eda, recommend_charts, chart_data

DATASETS: dict[str, dict] = {}


def register_dataset(path: Path, original_name: str) -> dict:
    df = load_dataframe(str(path))
    if df.empty:
        raise ValueError("The uploaded dataset is empty.")

    dataset_id = uuid.uuid4().hex[:12]

    DATASETS[dataset_id] = {
        "path": str(path),
        "name": original_name,
        "original_df": df.copy(),
        "df": df.copy(),
        "is_cleaned": False,
    }

    return dataset_summary(dataset_id)


def get_df(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in DATASETS:
        raise KeyError("Dataset not found")

    return DATASETS[dataset_id]["df"].copy()


def get_original_df(dataset_id: str) -> pd.DataFrame:
    if dataset_id not in DATASETS:
        raise KeyError("Dataset not found")

    return DATASETS[dataset_id]["original_df"].copy()


def replace_dataset_df(dataset_id: str, df: pd.DataFrame) -> None:
    if dataset_id not in DATASETS:
        raise KeyError("Dataset not found")

    DATASETS[dataset_id]["df"] = df.copy()
    DATASETS[dataset_id]["is_cleaned"] = True


def dataset_summary(dataset_id: str) -> dict:
    df = get_df(dataset_id)

    return {
        "id": dataset_id,
        "name": DATASETS[dataset_id]["name"],
        "profile": profile_dataframe(df),
        "is_cleaned": DATASETS[dataset_id].get("is_cleaned", False),
    }


def get_eda(dataset_id: str) -> dict:
    df = get_df(dataset_id)

    return {
        "eda": eda(df),
        "charts": recommend_charts(df),
    }


def chart_data_for_dataset(
    dataset_id: str,
    chart_type: str,
    x: str = "",
    y: str = "",
    aggregation: str = "mean",
):
    return chart_data(
        get_df(dataset_id),
        chart_type,
        x,
        y,
        aggregation,
    )


def preview(dataset_id: str, limit: int = 100) -> dict:
    df = get_df(dataset_id)

    preview_df = df.head(limit).copy()
    preview_df = preview_df.astype(object).where(
        pd.notna(preview_df),
        None,
    )

    return {
        "columns": [str(column) for column in preview_df.columns],
        "rows": preview_df.to_dict(orient="records"),
        "total_rows": len(df),
    }


def data_page(
    dataset_id: str,
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """
    Return a paginated section of the current dataset.

    limit:
        Number of rows to return, capped at 1000.

    offset:
        Starting row index.
    """

    df = get_df(dataset_id)

    # Keep pagination values safe.
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))

    page_df = df.iloc[offset:offset + limit].copy()

    # Convert NaN/NaT values to None so the response is JSON serializable.
    page_df = page_df.astype(object).where(
        pd.notna(page_df),
        None,
    )

    return {
        "columns": [str(column) for column in page_df.columns],
        "rows": page_df.to_dict(orient="records"),
        "total_rows": len(df),
        "offset": offset,
        "limit": limit,
    }
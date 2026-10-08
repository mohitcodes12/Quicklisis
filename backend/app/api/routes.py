from pathlib import Path

import uuid

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from pydantic import BaseModel, Field

from app.core.config import MAX_FILE_SIZE_MB, UPLOAD_DIR

from app.services.dataset_service import (

    register_dataset,

    dataset_summary,

    get_eda,

    preview,

    chart_data_for_dataset,

    get_df,

    replace_dataset_df,
    data_page,

)

from app.data_processing.profiling import profile_dataframe

from app.data_processing.cleaning import (


    cleaning_preview,

    clean_dataframe,

)

from app.ai.analyst import ask

router = APIRouter(prefix="/api")

# ============================================================

# REQUEST MODELS

# ============================================================

class AskRequest(BaseModel):

    dataset_id: str = Field(min_length=1)

    question: str = Field(min_length=1, max_length=1000)

# ============================================================

# HEALTH

# ============================================================

@router.get("/health")

def health():

    return {

        "status": "ok",

        "service": "quicklisis-api",

    }

# ============================================================

# DATASET UPLOAD

# ============================================================

@router.post("/datasets/upload")

async def upload_dataset(file: UploadFile = File(...)):

    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in {".csv", ".xlsx"}:

        raise HTTPException(

            status_code=400,

            detail="Only CSV and XLSX files are supported.",

        )

    data = await file.read()

    if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:

        raise HTTPException(

            status_code=413,

            detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.",

        )

    safe_name = (

        Path(file.filename).name

        if file.filename

        else "dataset.csv"

    )

    path = (

        UPLOAD_DIR

        / f"{uuid.uuid4().hex}_{safe_name}"

    )

    path.write_bytes(data)

    try:

        return register_dataset(path, safe_name)

    except Exception as exc:

        path.unlink(missing_ok=True)

        raise HTTPException(

            status_code=400,

            detail=f"Could not read dataset: {exc}",

        ) from exc

# ============================================================

# DATASET SUMMARY

# ============================================================

@router.get("/datasets/{dataset_id}")

def dataset(dataset_id: str):

    try:

        return dataset_summary(dataset_id)

    except KeyError:

        raise HTTPException(

            status_code=404,

            detail="Dataset not found",

        )

# ============================================================

# DATA PREVIEW

# ============================================================

@router.get("/datasets/{dataset_id}/preview")

def dataset_preview(
    dataset_id: str,
    limit: int = Query(100, ge=1, le=10000),
):

    try:

        return preview(dataset_id, limit)

    except KeyError:

        raise HTTPException(

            status_code=404,

            detail="Dataset not found",

        )

# ============================================================

# ============================================================

# FULL DATA / PAGINATED DATA

# ============================================================

@router.get("/datasets/{dataset_id}/data")
def dataset_data(
    dataset_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    try:
        return data_page(dataset_id, limit, offset)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load dataset data: {exc}") from exc

# EDA

# ============================================================

@router.get("/datasets/{dataset_id}/eda")

def dataset_eda(dataset_id: str):

    try:

        return get_eda(dataset_id)

    except KeyError:

        raise HTTPException(

            status_code=404,

            detail="Dataset not found",

        )

# ============================================================

# CHART DATA

# ============================================================

@router.get("/datasets/{dataset_id}/chart-data")

def dataset_chart_data(

    dataset_id: str,

    type: str,

    x: str = "",

    y: str = "",
    aggregation: str = "mean",
):

    try:

        return {

            "data": chart_data_for_dataset(

                dataset_id,

                type,

                x,

                y,

                aggregation,

            )

        }

    except KeyError:

        raise HTTPException(

            status_code=404,

            detail="Dataset not found",

        )

    except Exception as exc:

        raise HTTPException(

            status_code=400,

            detail=f"Could not generate chart data: {exc}",

        ) from exc

# ============================================================

# CLEANING REPORT

# ============================================================

@router.get("/datasets/{dataset_id}/cleaning-report")

def dataset_cleaning_report(dataset_id: str):

    """

    Analyze the current active dataset without modifying it.

    Returns:

    - detected cleaning issues

    - severity summary

    - before dimensions

    - after dimensions

    - cleaning summary

    """

    try:

        df = get_df(dataset_id)

    except KeyError:

        raise HTTPException(

            status_code=404,

            detail="Dataset not found",

        )

    try:

        report = cleaning_preview(df)

        summary = report.get(

            "issue_summary",

            report.get(

                "summary",

                {

                    "total_issues": 0,

                    "high": 0,

                    "medium": 0,

                    "low": 0,

                },

            ),

        )

        return {

            "issues": report.get("issues", []),

            "summary": summary,

            "message": (

                f"QuickLisis detected "

                f"{len(report.get('issues', []))} "

                "potential data-quality issue(s)."

            ),

            "before": report.get("before", {}),

            "after": report.get("after", {}),

            "cleaning_summary": report.get(

                "cleaning_summary",

                {},

            ),

        }

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=f"Unable to generate cleaning report: {exc}",

        ) from exc

# ============================================================

# CLEANING PREVIEW

# ============================================================

@router.get("/datasets/{dataset_id}/cleaning-preview")

def dataset_cleaning_preview(dataset_id: str):

    """

    Generate a before/after cleaning preview.

    This does NOT modify the active dataset.

    """

    try:

        df = get_df(dataset_id)

    except KeyError:

        raise HTTPException(

            status_code=404,

            detail="Dataset not found",

        )

    try:

        report = cleaning_preview(df)

        summary = report.get(

            "issue_summary",

            report.get(

                "summary",

                {

                    "total_issues": 0,

                    "high": 0,

                    "medium": 0,

                    "low": 0,

                },

            ),

        )

        return {

            "issues": report.get("issues", []),

            "summary": summary,

            "message": (

                f"QuickLisis detected "

                f"{len(report.get('issues', []))} "

                "potential data-quality issue(s)."

            ),

            "before": report.get("before", {}),

            "after": report.get("after", {}),

            "cleaning_summary": report.get(

                "cleaning_summary",

                {},

            ),

        }

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=f"Unable to generate cleaning preview: {exc}",

        ) from exc

# ============================================================

# APPLY CLEANING

# ============================================================

@router.post("/datasets/{dataset_id}/clean")

def clean_dataset(dataset_id: str):

    """

    Apply QuickLisis deterministic cleaning operations.

    The cleaning engine:

    - removes completely empty columns

    - removes completely empty rows

    - removes exact duplicates

    - trims text whitespace

    - normalizes column names

    - converts numeric-like columns

    - converts date-like columns

    IMPORTANT:

    The cleaned DataFrame becomes the active dataset.

    """

    # --------------------------------------------------------

    # Get current dataset

    # --------------------------------------------------------

    try:

        df = get_df(dataset_id)

    except KeyError:

        raise HTTPException(

            status_code=404,

            detail="Dataset not found",

        )

    # --------------------------------------------------------

    # Apply deterministic Pandas cleaning

    # --------------------------------------------------------

    try:

        cleaned_df, summary = clean_dataframe(df)

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=f"Unable to clean dataset: {exc}",

        ) from exc

    # --------------------------------------------------------

    # IMPORTANT:

    # Replace the active dataset with the cleaned DataFrame.

    # --------------------------------------------------------

    try:

        replace_dataset_df(

            dataset_id,

            cleaned_df,

        )

    except KeyError:

        raise HTTPException(

            status_code=404,

            detail="Dataset not found",

        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=f"Unable to save cleaned dataset: {exc}",

        ) from exc

    # --------------------------------------------------------

    # Generate fresh profile

    # --------------------------------------------------------

    try:

        profile = profile_dataframe(cleaned_df)

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=f"Unable to profile cleaned dataset: {exc}",

        ) from exc

    # --------------------------------------------------------

    # Generate fresh preview

    # --------------------------------------------------------

    try:

        preview_result = preview(dataset_id)

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=f"Unable to generate cleaned preview: {exc}",

        ) from exc

    # --------------------------------------------------------

    # Return everything the frontend needs

    # --------------------------------------------------------

    return {

        "id": dataset_id,

        "message": "Dataset cleaned successfully.",

        "profile": profile,

        "preview": preview_result,

        "cleaning_summary": summary,

        "is_cleaned": True,

    }

# ============================================================

# AI DATA ANALYST

# ============================================================

@router.post("/ai/ask")

def ai_ask(payload: AskRequest):

    try:

        return ask(

            payload.dataset_id,

            payload.question,

        )

    except KeyError:

        raise HTTPException(

            status_code=404,

            detail="Dataset not found",

        )

    except Exception as exc:

        raise HTTPException(

            status_code=500,

            detail=f"Unable to analyze dataset: {exc}",

        ) from exc
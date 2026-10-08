from __future__ import annotations







import re



from typing import Any







import numpy as np



import pandas as pd











DATE_NAME_HINTS = re.compile(



    r"(^|[_**\-**\s])(date|time|datetime|timestamp|dob|birth|joined|created|updated|start|end)([_**\-**\s]|$)",



    re.IGNORECASE,



)











def _clean_number(value: Any):



    if value is None or pd.isna(value):



        return None



    try:



        value = float(value)



    except (TypeError, ValueError):



        return None



    if not np.isfinite(value):



        return None



    return round(value, 4)











def _is_date_like(series: pd.Series, name: str = "") -> bool:



    if pd.api.types.is_datetime64_any_dtype(series):



        return True







    non_null = series.dropna()



    if non_null.empty:



        return False







    # Excel frequently stores dates as serial numbers. Only treat numeric



    # values as dates when the column name provides a reasonable date/time hint.



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











def _column_kind(series: pd.Series, name: str) -> str:



    if _is_date_like(series, name):



        return "date"



    if pd.api.types.is_bool_dtype(series):



        return "boolean"



    if pd.api.types.is_numeric_dtype(series):



        if re.search(r"(^|[_**\-**\s])(id|identifier|index|key)([_**\-**\s]|$)", str(name), re.IGNORECASE):



            return "id"



        return "numeric"







    unique = series.nunique(dropna=True)



    non_null_count = max(len(series.dropna()), 1)



    if unique / non_null_count >= 0.98 and non_null_count >= 10:



        return "id"



    if unique <= max(20, min(50, int(len(series) * 0.2))):



        return "categorical"



    return "text"











def _safe_series_numeric(series: pd.Series) -> pd.Series:



    return pd.to_numeric(series, errors="coerce")











def _numeric_columns(df: pd.DataFrame) -> list[str]:



    return [



        str(c)



        for c in df.columns



        if _column_kind(df[c], str(c)) == "numeric"



    ]











def _date_columns(df: pd.DataFrame) -> list[str]:



    return [



        str(c)



        for c in df.columns



        if _column_kind(df[c], str(c)) == "date"



    ]











def _categorical_columns(df: pd.DataFrame) -> list[str]:



    return [



        str(c)



        for c in df.columns



        if _column_kind(df[c], str(c)) in {"categorical", "boolean"}



    ]











def eda(df: pd.DataFrame) -> dict:



    numeric_names = _numeric_columns(df)



    numeric = df[numeric_names] if numeric_names else pd.DataFrame(index=df.index)







    numeric_stats = []



    for col in numeric.columns:



        s = _safe_series_numeric(numeric[col]).dropna()



        numeric_stats.append(



            {



                "column": str(col),



                "mean": _clean_number(s.mean()) if not s.empty else None,



                "median": _clean_number(s.median()) if not s.empty else None,



                "min": _clean_number(s.min()) if not s.empty else None,



                "max": _clean_number(s.max()) if not s.empty else None,



                "std": _clean_number(s.std()) if not s.empty else None,



            }



        )







    categories = []



    for col in _categorical_columns(df):



        vc = df[col].astype("string").fillna("Unknown").value_counts(dropna=False).head(10)



        categories.append(



            {



                "column": str(col),



                "values": [



                    {"label": str(k), "count": int(v)}



                    for k, v in vc.items()



                ],



            }



        )







    correlation = {}



    if len(numeric.columns) >= 2:



        corr = numeric.apply(pd.to_numeric, errors="coerce").corr().round(3)



        correlation = {



            str(c): {



                str(k): _clean_number(v)



                for k, v in corr[c].items()



            }



            for c in corr.columns



        }







    return {



        "numeric_stats": numeric_stats,



        "categorical_distributions": categories,



        "correlation": correlation,



    }











def recommend_charts(df: pd.DataFrame) -> list[dict]:

    """Return up to four diverse, dataset-aware chart configurations."""

    numeric = _numeric_columns(df)

    dates = _date_columns(df)

    categorical = _categorical_columns(df)

    charts: list[dict] = []



    if dates and numeric:

        charts.append({

            "type": "line",

            "analysis": "trend",

            "x": dates[0],

            "y": numeric[0],

            "aggregation": "mean",

            "title": f"{numeric[0]} over {dates[0]}",

            "insight": f"Shows how {numeric[0]} changes across {dates[0]}.",

        })



    if categorical:

        charts.append({

            "type": "bar",

            "analysis": "count",

            "x": categorical[0],

            "y": "",

            "aggregation": "count",

            "title": f"{categorical[0]} distribution",

            "insight": f"Shows the most common values and concentration in {categorical[0]}.",

        })



    if numeric:

        charts.append({

            "type": "histogram",

            "analysis": "distribution",

            "x": numeric[0],

            "y": "",

            "aggregation": "count",

            "title": f"Distribution of {numeric[0]}",

            "insight": f"Shows the spread and concentration of {numeric[0]}.",

        })



    if len(numeric) >= 2:

        charts.append({

            "type": "scatter",

            "analysis": "relationship",

            "x": numeric[0],

            "y": numeric[1],

            "aggregation": "mean",

            "title": f"{numeric[0]} vs {numeric[1]}",

            "insight": f"Shows the relationship between {numeric[0]} and {numeric[1]}.",

        })

    elif len(numeric) >= 1 and categorical:

        charts.append({

            "type": "bar",

            "analysis": "average",

            "x": categorical[0],

            "y": numeric[0],

            "aggregation": "mean",

            "title": f"Average {numeric[0]} by {categorical[0]}",

            "insight": f"Compares average {numeric[0]} across {categorical[0]}.",

        })

    elif len(numeric) >= 2:

        charts.append({

            "type": "heatmap",

            "analysis": "correlation",

            "x": "",

            "y": "",

            "aggregation": "mean",

            "title": "Numeric correlation heatmap",

            "insight": "Highlights positive and negative relationships between numeric fields.",

        })



    if len(charts) < 4 and len(numeric) >= 2 and not any(c["type"] == "heatmap" for c in charts):

        charts.append({

            "type": "heatmap",

            "analysis": "correlation",

            "x": "",

            "y": "",

            "aggregation": "mean",

            "title": "Numeric correlation heatmap",

            "insight": "Highlights positive and negative relationships between numeric fields.",

        })



    if len(charts) < 4 and numeric and categorical:

        charts.append({

            "type": "pie",

            "analysis": "share",

            "x": categorical[0],

            "y": numeric[0],

            "aggregation": "sum",

            "title": f"{categorical[0]} share",

            "insight": f"Shows how {numeric[0]} is distributed across {categorical[0]}.",

        })



    if len(charts) < 4 and numeric:

        charts.append({

            "type": "area",

            "analysis": "trend",

            "x": dates[0] if dates else numeric[0],

            "y": numeric[0],

            "aggregation": "mean",

            "title": f"{numeric[0]} area view",

            "insight": f"Shows the overall movement and magnitude of {numeric[0]}.",

        })



    return charts[:4]





def _top_categories(series: pd.Series, limit: int = 15) -> pd.Series:
    return series.astype("string").fillna("Unknown").value_counts().head(limit)


def _grouped_values(df: pd.DataFrame, x: str, y: str, aggregation: str, limit: int = 15) -> list[dict]:

    if not x or x not in df.columns:

        return []

    if aggregation == "count" or not y or y not in df.columns:

        counts = _top_categories(df[x], limit)

        return [{"x": str(k), "y": int(v)} for k, v in counts.items()]



    work = pd.DataFrame({

        "x": df[x].astype("string").fillna("Unknown"),

        "y": _safe_series_numeric(df[y]),

    }).dropna(subset=["y"])

    if work.empty:

        return []



    grouped = work.groupby("x", dropna=False)["y"]

    if aggregation == "sum":

        values = grouped.sum()

    elif aggregation == "median":

        values = grouped.median()

    else:

        values = grouped.mean()

    values = values.sort_values(ascending=False).head(limit)

    return [{"x": str(k), "y": _clean_number(v)} for k, v in values.items()]





def chart_data(

    df_or_dataset,

    chart_type: str,

    x: str = "",

    y: str = "",

    aggregation: str = "mean",

):

    df = df_or_dataset



    if x and x not in df.columns:

        return []

    if y and y not in df.columns:

        return []



    if chart_type in {"bar", "horizontal_bar", "grouped_bar", "stacked_bar"}:

        return _grouped_values(df, x, y, aggregation)



    if chart_type in {"line", "area"} and x in df.columns and y in df.columns:

        raw_x = df[x]

        values = _safe_series_numeric(df[y])



        # A line/area chart can also use a categorical X field. Aggregate it
        # instead of attempting to parse labels such as A/B/C as dates.
        if _column_kind(raw_x, str(x)) in {"categorical", "boolean", "text", "id"}:

            return _grouped_values(df, x, y, aggregation)

        # Keep genuinely numeric X values numeric. Numeric fields are converted
        # to dates only when they are actually Excel serial dates.
        if pd.api.types.is_numeric_dtype(raw_x) and not _is_date_like(raw_x, str(x)):

            work = pd.DataFrame({"x": _safe_series_numeric(raw_x), "y": values}).dropna()

            if work.empty:

                return []

            grouped = work.groupby("x", dropna=False)["y"]

            if aggregation == "sum":

                result = grouped.sum()

            elif aggregation == "median":

                result = grouped.median()

            else:

                result = grouped.mean()

            result = result.dropna().sort_index()

            return [{"x": _clean_number(idx), "y": _clean_number(value)} for idx, value in result.items()]



        if pd.api.types.is_numeric_dtype(raw_x) and _is_date_like(raw_x, str(x)):

            numeric_dates = pd.to_numeric(raw_x, errors="coerce")

            dates = pd.to_datetime(numeric_dates, errors="coerce", unit="D", origin="1899-12-30")

        else:

            dates = pd.to_datetime(raw_x, errors="coerce", format="mixed")



        work = pd.DataFrame({"x": dates, "y": values}).dropna()

        if work.empty:

            return []

        span_days = (work["x"].max() - work["x"].min()).days

        freq = "D" if span_days <= 120 else "W" if span_days <= 730 else "ME"

        grouped = work.set_index("x")["y"]

        if aggregation == "sum":

            grouped = grouped.resample(freq).sum()

        elif aggregation == "median":

            grouped = grouped.resample(freq).median()

        else:

            grouped = grouped.resample(freq).mean()

        grouped = grouped.dropna()

        return [{"x": idx.strftime("%Y-%m-%d"), "y": _clean_number(value)} for idx, value in grouped.items()]



    if chart_type == "scatter" and x in df.columns and y in df.columns:

        work = pd.DataFrame({"x": _safe_series_numeric(df[x]), "y": _safe_series_numeric(df[y])}).dropna().head(500)

        return [{"x": _clean_number(a), "y": _clean_number(b)} for a, b in zip(work["x"], work["y"])]



    if chart_type == "histogram" and x in df.columns:

        vals = _safe_series_numeric(df[x]).dropna()

        if vals.empty:

            return []

        bins = min(15, max(5, int(np.sqrt(len(vals)))))

        if vals.nunique() == 1:

            return [{"x": _clean_number(vals.iloc[0]), "y": int(len(vals))}]

        counts, edges = np.histogram(vals, bins=bins)

        return [{"x": _clean_number((edges[i] + edges[i + 1]) / 2), "y": int(counts[i])} for i in range(len(counts))]



    if chart_type in {"pie", "donut", "radar", "treemap", "funnel"}:

        values = _grouped_values(df, x, y, aggregation)

        if chart_type in {"treemap", "funnel"}:

            return [{"name": item["x"], "value": item["y"]} for item in values]

        return values



    if chart_type == "heatmap":

        numeric_names = _numeric_columns(df)

        if len(numeric_names) < 2:

            return []

        corr = df[numeric_names].apply(pd.to_numeric, errors="coerce").corr()

        return [

            {"x": str(col), "y": str(row), "value": _clean_number(corr.loc[row, col])}

            for row in corr.index

            for col in corr.columns

            if pd.notna(corr.loc[row, col])

        ]



    return []



"""Tests for the analyst. Run from your backend folder with:  pytest -q test_analyst.py

EDIT THIS ONE LINE if your analyst file lives somewhere else:
"""
import importlib
import json
import os

import numpy as np
import pandas as pd
import pytest

ANALYST_MODULE = os.getenv("ANALYST_MODULE", "app.ai.analyst")
analyst = importlib.import_module(ANALYST_MODULE)


# ------------------------------------------------------------------ sample data (shaped like your 92-row file)
@pytest.fixture
def df():
    rng = np.random.default_rng(7)
    n = 92
    d = pd.DataFrame({
        "name": [f"Person {i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "age": rng.integers(21, 60, n),
        "date_joined": pd.to_datetime("2023-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
        "rating": rng.choice(["Poor", "Below Average", "Average", "Good", "Excellent"], n),
        "department": rng.choice(["Sales", "IT", "HR", "Finance", "Ops"], n),
        "salary": rng.integers(30000, 90000, n),
        "country": "India",
    })
    d.loc[[3, 10, 20, 30], "gender"] = None
    # make the "oldest with Average rating" a deliberate tie
    d.loc[[5, 6], ["age", "rating"]] = [[63, "Average"], [63, " average "]]
    return d


# ------------------------------------------------------------------ fake Gemini
class FakeGemini:
    """Scripted replacement for _gemini_request. script maps a prompt marker to a list of replies."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def __call__(self, prompt, *, json_mode=False, max_tokens=2048):
        for marker, replies in self.script.items():
            if prompt.startswith(marker):
                self.calls.append(marker)
                if not replies:
                    return None
                reply = replies.pop(0) if len(replies) > 1 else replies[0]
                return None if reply is None else (reply if isinstance(reply, str) else json.dumps(reply))
        self.calls.append("UNSCRIPTED")
        return None


@pytest.fixture
def setup(monkeypatch, df):
    monkeypatch.setattr(analyst, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(analyst, "get_df", lambda dataset_id: df)
    monkeypatch.setattr(analyst, "VERIFY_WITH_AI", True)
    monkeypatch.setattr(analyst, "ON_DISPUTE", "direct")

    def install(script):
        fake = FakeGemini(script)
        monkeypatch.setattr(analyst, "_gemini_request", fake)
        return fake
    return install


OLDEST_AVG_PLAN = {
    "intent": "analysis", "can_answer": True,
    "filters": [{"column": "rating", "op": "eq", "value": "Average"}],
    "extreme": {"column": "age", "mode": "max"}, "select": ["name", "age", "rating"],
}


# ================================================================== executor (pure pandas, no LLM)
def test_oldest_with_average_rating_keeps_ties_and_messy_labels(df):
    res = analyst.execute_plan(df, OLDEST_AVG_PLAN)
    truth = df[df.rating.str.strip().str.lower() == "average"]
    truth = truth[truth.age == truth.age.max()]
    assert sorted(res.df["name"]) == sorted(truth["name"])
    assert res.scalar == {"operation": "max", "column": "age", "value": 63}
    assert len(res.df) >= 2  # the deliberate tie


def test_group_mean_sorted_matches_pandas(df):
    plan = {"group_by": ["department"], "metrics": [{"column": "salary", "agg": "mean", "as": "avg_salary"}],
            "sort": {"by": "avg_salary", "order": "desc"}, "limit": 3}
    res = analyst.execute_plan(df, plan)
    expected = df.groupby("department")["salary"].mean().sort_values(ascending=False).head(3)
    assert list(res.df["department"]) == list(expected.index)
    assert np.allclose(res.df["avg_salary"], expected.values)


def test_scalar_mean_and_count(df):
    res = analyst.execute_plan(df, {"metrics": [{"column": "salary", "agg": "mean"}]})
    assert res.scalar["value"] == pytest.approx(df.salary.mean(), rel=1e-5)
    res = analyst.execute_plan(df, {"filters": [{"column": "department", "op": "eq", "value": "sales"}],
                                    "metrics": [{"column": None, "agg": "count", "as": "n"}]})
    assert res.scalar["value"] == int((df.department == "Sales").sum())


def test_average_salary_is_not_average_age(df):
    """Regression for the 'average' contains 'age' bug."""
    res = analyst.execute_plan(df, {"metrics": [{"column": "salary", "agg": "mean"}]})
    assert res.scalar["column"] == "salary"


def test_monthly_trend_bucket(df):
    plan = {"group_by": [{"column": "date_joined", "bucket": "month"}],
            "metrics": [{"column": None, "agg": "count", "as": "joins"}]}
    res = analyst.execute_plan(df, plan)
    assert res.df["joins"].sum() == len(df)
    assert res.df.iloc[0, 0].count("-") == 1  # '2023-01'


def test_filters_in_between_contains_null(df):
    assert analyst.execute_plan(df, {"filters": [{"column": "department", "op": "in", "value": ["IT", "hr"]}]}).matched_rows == \
        int(df.department.isin(["IT", "HR"]).sum())
    assert analyst.execute_plan(df, {"filters": [{"column": "age", "op": "between", "value": [30, 40]}]}).matched_rows == \
        int(df.age.between(30, 40).sum())
    assert analyst.execute_plan(df, {"filters": [{"column": "name", "op": "contains", "value": "person 1"}]}).matched_rows > 0
    assert analyst.execute_plan(df, {"filters": [{"column": "gender", "op": "is_null"}]}).matched_rows == 4
    assert analyst.execute_plan(df, {"filters": [{"column": "date_joined", "op": "gte", "value": "2024-01-01"}]}).matched_rows == \
        int((df.date_joined >= "2024-01-01").sum())


def test_no_match_returns_empty_result_not_error(df):
    res = analyst.execute_plan(df, {"filters": [{"column": "rating", "op": "eq", "value": "Terrible"}],
                                    "extreme": {"column": "age", "mode": "max"}})
    assert res.matched_rows == 0 and res.result_rows == 0


def test_with_ties_on_limit_one(df):
    plan = {"group_by": ["department"], "metrics": [{"column": None, "agg": "count", "as": "n"}],
            "sort": {"by": "n", "order": "desc", "with_ties": True}, "limit": 1}
    res = analyst.execute_plan(df, plan)
    counts = df.groupby("department").size()
    assert len(res.df) == int((counts == counts.max()).sum())


@pytest.mark.parametrize("plan", [
    {"filters": [{"column": "salery", "op": "eq", "value": 1}]},                       # unknown column
    {"metrics": [{"column": "name", "agg": "mean"}]},                                   # mean of text
    {"filters": [{"column": "age", "op": "DROP TABLE", "value": 1}]},                   # unknown operator
    {"extreme": {"column": "name", "mode": "max"}},                                     # max of text
    {"extreme": {"column": "age", "mode": "max"}, "metrics": [{"column": "age", "agg": "mean"}]},
    {"group_by": [{"column": "name", "bucket": "month"}], "metrics": [{"column": None, "agg": "count"}]},
    {"filters": [{"column": "age", "op": "gt", "value": "abc"}]},
    {"sort": {"by": "nope"}},
    "not a dict",
])
def test_invalid_plans_are_rejected_cleanly(df, plan):
    with pytest.raises(analyst.PlanError):
        analyst.execute_plan(df, plan)


def test_plan_cannot_run_code(df):
    """Strings are only ever compared as data - nothing is evaluated."""
    plan = {"filters": [{"column": "name", "op": "eq", "value": "__import__('os').system('echo hacked')"}]}
    assert analyst.execute_plan(df, plan).matched_rows == 0


# ================================================================== full pipeline with scripted Gemini
def test_happy_path_verified(setup):
    fake = setup({
        "QUERY_PLAN_REQUEST": [OLDEST_AVG_PLAN],
        "REVIEW_REQUEST": [{"verdict": "correct", "problem": "",
                            "answer": "The oldest people rated Average are aged 63."}],
    })
    out = analyst.ask("1", "name of the oldest who has an average rating")
    assert out["calculated"] is True and out["verification"] == "verified"
    assert out["ai_provider"] == "gemini-explanation"
    assert out["value"] == 63 and out["column"] == "age" and out["operation"] == "max"
    assert {r["name"] for r in out["table"]["rows"]} >= {"Person 5", "Person 6"}
    assert fake.calls == ["QUERY_PLAN_REQUEST", "REVIEW_REQUEST"]


def test_reviewer_rejects_then_repair_succeeds(setup):
    wrong = {"intent": "analysis", "can_answer": True, "metrics": [{"column": "age", "agg": "mean"}]}
    fake = setup({
        "QUERY_PLAN_REQUEST": [wrong, OLDEST_AVG_PLAN],
        "REVIEW_REQUEST": [
            {"verdict": "incorrect", "problem": "The user wants a name, but only an average age was computed.", "answer": ""},
            {"verdict": "correct", "problem": "", "answer": "The oldest people rated Average are aged 63."},
        ],
    })
    out = analyst.ask("1", "name of the oldest who has an average rating")
    assert out["verification"] == "verified" and out["value"] == 63
    assert fake.calls == ["QUERY_PLAN_REQUEST", "REVIEW_REQUEST", "QUERY_PLAN_REQUEST", "REVIEW_REQUEST"]


def test_invalid_plan_is_repaired(setup):
    bad = {"intent": "analysis", "can_answer": True, "extreme": {"column": "agee", "mode": "max"}}
    setup({"QUERY_PLAN_REQUEST": [bad, OLDEST_AVG_PLAN],
           "REVIEW_REQUEST": [{"verdict": "correct", "problem": "", "answer": "The oldest is aged 63."}]})
    out = analyst.ask("1", "who is the oldest")
    assert out["calculated"] is True and out["value"] == 63


def test_pandas_cannot_answer_so_gemini_reads_data(setup):
    fake = setup({
        "QUERY_PLAN_REQUEST": [{"intent": "analysis", "can_answer": False, "reason": "needs interpretation"}],
        "DIRECT_ANALYSIS_REQUEST": [{"answer": "Salaries look fairly evenly spread.", "can_determine": True}],
    })
    out = analyst.ask("1", "describe the salary pattern")
    assert out["calculated"] is False and out["ai_provider"] == "gemini-direct"
    assert out["verification"] == "unverified" and "not verified" in out["answer"]
    assert fake.calls == ["QUERY_PLAN_REQUEST", "DIRECT_ANALYSIS_REQUEST"]


def test_missing_information_is_stated_not_invented(setup):
    setup({
        "QUERY_PLAN_REQUEST": [{"intent": "analysis", "can_answer": False, "reason": "no age group column"}],
        "DIRECT_ANALYSIS_REQUEST": [{"answer": "The dataset has no purchase data, so this cannot be answered.",
                                     "can_determine": False}],
    })
    out = analyst.ask("1", "which age group buys the most?")
    assert out["calculated"] is False and "cannot be answered" in out["answer"]
    assert "not verified" not in out["answer"]


def test_persistent_dispute_falls_back_to_direct_and_keeps_attempt(setup):
    wrong = {"intent": "analysis", "can_answer": True, "metrics": [{"column": "age", "agg": "mean"}]}
    setup({
        "QUERY_PLAN_REQUEST": [wrong],
        "REVIEW_REQUEST": [{"verdict": "incorrect", "problem": "wrong measure", "answer": ""}],
        "DIRECT_ANALYSIS_REQUEST": [{"answer": "Person 5 and Person 6 (age 63).", "can_determine": True}],
    })
    out = analyst.ask("1", "who is the oldest with an average rating")
    assert out["ai_provider"] == "gemini-direct" and out["verification"] == "disputed"
    assert "computed_attempt" in out


def test_on_dispute_computed_keeps_pandas_result(setup, monkeypatch):
    monkeypatch.setattr(analyst, "ON_DISPUTE", "computed")
    wrong = {"intent": "analysis", "can_answer": True, "metrics": [{"column": "age", "agg": "mean"}]}
    setup({"QUERY_PLAN_REQUEST": [wrong],
           "REVIEW_REQUEST": [{"verdict": "incorrect", "problem": "x", "answer": ""}]})
    out = analyst.ask("1", "q")
    assert out["calculated"] is True and out["verification"] == "disputed"


def test_hallucinated_number_in_explanation_is_discarded(setup):
    setup({
        "QUERY_PLAN_REQUEST": [OLDEST_AVG_PLAN],
        "REVIEW_REQUEST": [{"verdict": "correct", "problem": "",
                            "answer": "The oldest is aged 63 and earns 99,999 on average."}],
    })
    out = analyst.ask("1", "oldest with average rating")
    assert "99,999" not in out["answer"]
    assert out["ai_provider"] == "deterministic-analysis" and "63" in out["answer"]


def test_gemini_down_never_crashes(setup):
    setup({})  # every call returns None
    out = analyst.ask("1", "what is the average salary?")
    assert out["calculated"] is False and "try again" in out["answer"]


def test_review_unavailable_still_returns_computed_answer(setup):
    setup({"QUERY_PLAN_REQUEST": [{"intent": "analysis", "can_answer": True,
                                   "metrics": [{"column": "salary", "agg": "mean"}]}]})
    out = analyst.ask("1", "average salary")
    assert out["calculated"] is True and out["verification"] == "unverified"
    assert out["column"] == "salary"


def test_verification_can_be_switched_off(setup, monkeypatch):
    monkeypatch.setattr(analyst, "VERIFY_WITH_AI", False)
    fake = setup({"QUERY_PLAN_REQUEST": [OLDEST_AVG_PLAN]})
    out = analyst.ask("1", "oldest with average rating")
    assert out["calculated"] is True and fake.calls == ["QUERY_PLAN_REQUEST"]


def test_llm_call_budget_is_respected(setup):
    wrong = {"intent": "analysis", "can_answer": True, "metrics": [{"column": "age", "agg": "mean"}]}
    fake = setup({
        "QUERY_PLAN_REQUEST": [wrong],
        "REVIEW_REQUEST": [{"verdict": "incorrect", "problem": "x", "answer": ""}],
        "DIRECT_ANALYSIS_REQUEST": [None],
    })
    analyst.ask("1", "q")
    assert len(fake.calls) <= analyst.MAX_LLM_CALLS


def test_group_result_keeps_old_rows_shape_for_frontend(setup):
    setup({
        "QUERY_PLAN_REQUEST": [{"intent": "analysis", "can_answer": True, "group_by": ["department"],
                                "metrics": [{"column": "salary", "agg": "sum", "as": "total_salary"}],
                                "sort": {"by": "total_salary", "order": "desc"}, "limit": 3}],
        "REVIEW_REQUEST": [{"verdict": "correct", "problem": "", "answer": "Here are the top departments."}],
    })
    out = analyst.ask("1", "top 3 departments by salary")
    assert len(out["rows"]) == 3 and set(out["rows"][0]) == {"label", "value"}


def test_greeting_and_chat_do_not_run_analysis(setup):
    fake = setup({"CONVERSATION_REQUEST": ["Hi! Ask me about your data."]})
    out = analyst.ask("1", "Hello!")
    assert out["ai_provider"] == "gemini-conversation" and fake.calls == ["CONVERSATION_REQUEST"]
    fake = setup({"QUERY_PLAN_REQUEST": [{"intent": "chat"}], "CONVERSATION_REQUEST": ["Pandas is a library."]})
    out = analyst.ask("1", "what is pandas?")
    assert out["calculated"] is False


def test_empty_question_and_missing_key(setup, monkeypatch):
    setup({})
    assert analyst.ask("1", "   ")["calculated"] is False
    monkeypatch.setattr(analyst, "GEMINI_API_KEY", "")
    assert "GEMINI_API_KEY" in analyst.ask("1", "average salary")["answer"]


def test_plan_json_wrapped_in_markdown_fence_is_accepted(setup):
    fenced = "```json\n" + json.dumps(OLDEST_AVG_PLAN) + "\n```"
    setup({"QUERY_PLAN_REQUEST": [fenced],
           "REVIEW_REQUEST": [{"verdict": "correct", "problem": "", "answer": "The oldest is aged 63."}]})
    assert analyst.ask("1", "oldest with average rating")["value"] == 63


def test_prompt_contains_category_values_so_average_maps_to_label(setup, monkeypatch, df):
    captured = {}

    def fake(prompt, **kw):
        captured.setdefault("prompts", []).append(prompt)
        return None
    monkeypatch.setattr(analyst, "_gemini_request", fake)
    analyst.ask("1", "oldest with average rating")
    assert "Below Average" in captured["prompts"][0] and "Excellent" in captured["prompts"][0]
    assert "rating" in captured["prompts"][0]


def test_large_dataset_is_never_sent_whole(monkeypatch):
    big = pd.DataFrame({"a": range(5000), "b": ["x"] * 5000})
    assert analyst._data_block(big) is None
    small = pd.DataFrame({"a": range(10)})
    assert analyst._data_block(small) is not None


def test_transport_returns_none_on_truncation_and_http_errors(monkeypatch):
    monkeypatch.setattr(analyst, "GEMINI_API_KEY", "k")
    monkeypatch.setattr(analyst.time, "sleep", lambda s: None)

    class R:
        def __init__(self, status=200, payload=None, text=""):
            self.status_code, self._p, self.text = status, payload, text
            self.ok = status < 400

        def json(self):
            return self._p
    seen = {}

    def post_trunc(url, json, headers, timeout):
        seen["url"], seen["headers"] = url, headers
        return R(payload={"candidates": [{"finishReason": "MAX_TOKENS",
                                          "content": {"parts": [{"text": "The average age across the"}]}}]})
    monkeypatch.setattr(analyst.requests, "post", post_trunc)
    assert analyst._gemini_request("hi") is None                      # truncated text is discarded
    assert "key=" not in seen["url"] and seen["headers"]["x-goog-api-key"] == "k"   # key not in URL

    calls = {"n": 0}

    def post_429_then_ok(url, json, headers, timeout):
        calls["n"] += 1
        return R(429) if calls["n"] < 3 else R(payload={"candidates": [{"finishReason": "STOP",
                                                                         "content": {"parts": [{"text": "ok"}]}}]})
    monkeypatch.setattr(analyst.requests, "post", post_429_then_ok)
    assert analyst._gemini_request("hi") == "ok" and calls["n"] == 3  # retried twice

    monkeypatch.setattr(analyst.requests, "post", lambda *a, **k: R(400, text="bad"))
    assert analyst._gemini_request("hi") is None
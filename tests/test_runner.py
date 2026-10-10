"""Tests for the §7.2 cost guard, §9 runner semantics (resume, aborts, concurrency).

All runner tests run offline against fake contestants.
"""

import json
import threading
import time

import pytest

from tierline_mvp import contestants, runner


class FakeContestant:
    """Contract-conformant contestant for offline runner tests."""

    def __init__(self, name="Fake Model", cost=0.01, label="balance",
                 error=None, delay=0.0, http_402_after=None):
        self.name = name
        self.cost = cost
        self.label = label
        self.error = error
        self.delay = delay
        self.http_402_after = http_402_after
        self.calls = 0
        self._lock = threading.Lock()

    def predict(self, state_text, options, definitions, prompt_version):
        with self._lock:
            self.calls += 1
            n = self.calls
        if self.delay:
            time.sleep(self.delay)
        if self.http_402_after is not None and n > self.http_402_after:
            return {"label": None, "probs": None, "latency_ms": 1, "cost_usd": 0.0,
                    "raw": None, "error": "HTTP 402: insufficient credits"}
        return {"label": self.label, "probs": None, "latency_ms": 1,
                "cost_usd": self.cost, "raw": self.label, "error": self.error}


def make_rows(n):
    return [{"row_id": f"bnk-{i:04d}", "text": f"utterance {i}", "intent": "balance"}
            for i in range(1, n + 1)]


def make_task(tmp_path, monkeypatch, contestant, rows, *, cap=10.0, workers=1,
              version="V1", key="fake"):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    meter = runner.CostMeter(cap)
    stop_event = threading.Event()
    state = {"reason": None, "detail": None, "lock": threading.Lock()}
    task = runner.TaskRun(key, version, contestant, rows, meter, workers,
                          threading.Semaphore(10), stop_event, state)
    return task, meter, state


def file_records(tmp_path, key="fake", version="V1"):
    path = tmp_path / f"v1-{key}-{version}.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# --- resume semantics (§9) ---------------------------------------------------

def test_run_task_scores_all_rows_then_resume_skips(tmp_path, monkeypatch):
    task, meter, state = make_task(tmp_path, monkeypatch, FakeContestant(), make_rows(3))
    summary = task.run()
    assert summary["scored"] == 3 and summary["errors"] == 0
    records = file_records(tmp_path)
    assert len(records) == 3
    assert all(r["error"] is None for r in records)
    assert meter.total_usd == pytest.approx(0.03)

    task2, _, _ = make_task(tmp_path, monkeypatch, FakeContestant(), make_rows(3))
    summary2 = task2.run()
    assert summary2["pending"] == 0 and summary2["scored"] == 0
    assert len(file_records(tmp_path)) == 3  # unchanged


def test_resume_retests_errored_rows_and_last_wins(tmp_path, monkeypatch):
    rows = make_rows(3)
    # pre-seed: row 1 ok, row 2 errored
    task0, _, _ = make_task(tmp_path, monkeypatch, FakeContestant(), rows[:1])
    task0.run()
    path = tmp_path / "v1-fake-V1.jsonl"
    errored = dict(file_records(tmp_path)[0])
    errored.update(row_id="bnk-0002", label=None, raw=None, error="HTTP 500")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(errored) + "\n")

    pending = runner.load_pending(rows, path)
    assert [r["row_id"] for r in pending] == ["bnk-0002", "bnk-0003"]

    task, _, _ = make_task(tmp_path, monkeypatch, FakeContestant(), rows)
    task.run()
    records = file_records(tmp_path)
    assert len(records) == 4  # 1 ok + 1 error (kept, append-only) + 2 new successes
    last_by_id = {}
    for r in records:
        last_by_id[r["row_id"]] = r
    assert all(r["error"] is None for r in last_by_id.values())  # last occurrence wins


# --- cost guard + aborts ------------------------------------------------------

def test_budget_guard_stops_before_call_that_would_exceed(tmp_path, monkeypatch):
    task, meter, state = make_task(tmp_path, monkeypatch, FakeContestant(cost=0.01),
                                   make_rows(5), cap=0.015)
    summary = task.run()
    assert summary["scored"] == 2  # 0.01 + 0.01 crosses the 0.015 cap at call 3
    assert state["reason"] == runner.BUDGET_EXCEEDED
    assert len(file_records(tmp_path)) == 2
    assert meter.total_usd == pytest.approx(0.02)


def test_http_402_aborts_whole_task(tmp_path, monkeypatch):
    contestant = FakeContestant(http_402_after=2)
    task, meter, state = make_task(tmp_path, monkeypatch, contestant, make_rows(5))
    summary = task.run()
    assert contestant.calls == 3  # 2 ok + the 402 row; rows 4-5 never called
    assert state["reason"] == runner.ABORTED_402
    records = file_records(tmp_path)
    assert len(records) == 3
    assert records[-1]["error"] and "402" in records[-1]["error"]


# --- concurrency --------------------------------------------------------------

def test_concurrent_workers_score_every_row_exactly_once(tmp_path, monkeypatch):
    contestant = FakeContestant(delay=0.005)
    task, meter, state = make_task(tmp_path, monkeypatch, contestant, make_rows(12),
                                   cap=10.0, workers=4, key="gemini-3.8-flash")
    summary = task.run()
    assert summary["scored"] == 12 and summary["errors"] == 0
    records = file_records(tmp_path, key="gemini-3.8-flash")
    ids = [r["row_id"] for r in records]
    assert len(ids) == len(set(ids)) == 12  # no duplicates, no losses
    assert state["reason"] is None


# --- run_grid orchestration (fake builders, no network) -----------------------

def test_run_grid_runs_all_model_version_tasks(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "RESULTS_DIR", tmp_path)
    made = {}

    def fake_build(key):
        made[key] = FakeContestant(name=f"Fake {key}", cost=0.001)
        return made[key]

    monkeypatch.setattr(contestants, "build_contestant", fake_build)
    result = runner.run_grid(["fake-a", "fake-b"], ["V1"], make_rows(4),
                             workers=2, use_balance_guard=False)
    assert set(result["summaries"]) == {"v1-fake-a-V1", "v1-fake-b-V1"}
    assert all(s["scored"] == 4 for s in result["summaries"].values())
    assert result["reason"] is None
    assert result["spent_usd"] == pytest.approx(0.008)


def test_cap_must_be_positive():
    with pytest.raises(ValueError):
        runner.CostMeter(0)
    with pytest.raises(ValueError):
        runner.CostMeter(-1)


def test_check_passes_below_cap():
    meter = runner.CostMeter(10.0)
    meter.charge(3.0)
    meter.check()  # no raise


def test_check_aborts_at_cap_before_next_call():
    meter = runner.CostMeter(0.01)
    meter.charge(0.0078)
    meter.check()  # 0.0078 < 0.01 -> may proceed
    meter.charge(0.0070)
    with pytest.raises(runner.BudgetExceeded) as excinfo:
        meter.check()
    message = str(excinfo.value)
    assert "$0.01" in message and "$0.0148" in message and "aborting before the next call" in message


def test_charge_accumulates_and_counts():
    meter = runner.CostMeter(10.0)
    meter.charge(1.0)
    meter.charge(0.25)
    assert meter.total_usd == pytest.approx(1.25)
    assert meter.calls == 2


def test_roster_keys_match_amended_design_doc():
    expected_openrouter = {"gemini-3.8-flash", "gpt-6.1-sol",
                           "claude-5.5-opus-thinking", "kimi-k3", "gemma-4-31b"}
    assert set(contestants.OPENROUTER_MODELS) == expected_openrouter
    # 2026-10-10 session-5b amendment: S1 slots are the benchmark-selected five
    assert contestants.HOSTED_S1_KEYS == ("jev",)
    assert set(contestants.LOCAL_S1_KEYS) == {
        "winnow-e4b", "decider-4b", "bekko-400m", "opendecider-small-mlx"}
    assert set(contestants.LEGACY_S1_KEYS) == {
        "decider-2b", "opendecider-nano", "laya-421m", "von", "lev"}
    assert len(contestants.ALL_CONTESTANT_KEYS) == 10
    assert not (set(contestants.ALL_CONTESTANT_KEYS) & set(contestants.LEGACY_S1_KEYS))
    # legacy keys still build (reported, not featured); bekko routes to its script
    for legacy in contestants.LEGACY_S1_KEYS:
        contestants.build_contestant(legacy)  # does not raise


def test_model_entries_are_wellformed():
    required = {"display_name", "slug", "reasoning_effort", "max_tokens",
                "price_prompt_per_1m_usd", "price_completion_per_1m_usd"}
    slugs = set()
    for key, meta in contestants.OPENROUTER_MODELS.items():
        assert required <= set(meta), f"{key} missing config keys"
        assert meta["max_tokens"] >= 2048
        assert meta["price_prompt_per_1m_usd"] > 0
        assert meta["price_completion_per_1m_usd"] > 0
        slugs.add(meta["slug"])
    assert len(slugs) == len(contestants.OPENROUTER_MODELS), "slugs must be unique"


def test_display_names_match_design_doc_roster():
    names = {meta["display_name"] for meta in contestants.OPENROUTER_MODELS.values()}
    assert names == {
        "Gemini 3.8 Flash, High",
        "GPT-6.1 Sol, Max Effort",
        "Claude 5.5 Opus Thinking, Max Effort",
        "Kimi K3",
        "Gemma-4-31B-it",
    }


def test_reasoning_efforts_recorded():
    efforts = {k: m["reasoning_effort"] for k, m in contestants.OPENROUTER_MODELS.items()}
    assert efforts == {
        "gemini-3.8-flash": "high",
        "gpt-6.1-sol": "max",
        "claude-5.5-opus-thinking": "max",
        "kimi-k3": "high",
        "gemma-4-31b": None,  # model supports `reasoning` but not `reasoning_effort`
    }
"""Tests for the §7.2 cost guard (runner.CostMeter)."""

import pytest

from tierline_mvp import contestants, runner


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


def test_all_four_openrouter_models_configured():
    expected_keys = {"gemini-3.8-flash", "gpt-6.1-sol",
                     "claude-5.5-opus-thinking", "kimi-k3"}
    assert set(contestants.OPENROUTER_MODELS) == expected_keys


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
    }


def test_reasoning_efforts_recorded():
    efforts = {k: m["reasoning_effort"] for k, m in contestants.OPENROUTER_MODELS.items()}
    assert efforts == {
        "gemini-3.8-flash": "high",
        "gpt-6.1-sol": "max",
        "claude-5.5-opus-thinking": "max",
        "kimi-k3": "high",
    }
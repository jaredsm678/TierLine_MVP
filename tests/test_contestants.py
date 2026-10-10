"""Offline tests for the OpenRouter adapter: payload shape, retry, cost.

Uses httpx.MockTransport so no network and no real API key are involved
(the key comes from the test environment).
"""

import os

import httpx
import pytest

from tierline_mvp import contestants, prompts, schema


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")


def make_contestant(handler) -> contestants.OpenRouterContestant:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    c = contestants.OpenRouterContestant("gemini-3.8-flash", client=client)
    c.BACKOFF_S = (0.0, 0.0)  # no sleeping in tests
    return c


def ok_body(label="pay_bill", usage=None):
    body = {"model": "google/gemini-3.8-flash",
            "choices": [{"message": {"content": f" {label} "}}],
            "usage": usage if usage is not None else
                     {"prompt_tokens": 1000, "completion_tokens": 10}}
    return httpx.Response(200, json=body)


def test_payload_shape():
    payload = contestants.build_payload(
        "google/gemini-3.8-flash", "V1", "an utterance",
        reasoning_effort="high", max_tokens=2048,
    )
    assert payload["model"] == "google/gemini-3.8-flash"
    assert payload["messages"] == [
        {"role": "user", "content": prompts.render_prompt("V1", "an utterance")}
    ]  # single message, no system role (§6)
    assert payload["temperature"] == 0
    assert payload["max_tokens"] == 2048
    assert payload["reasoning"] == {"effort": "high"}
    assert payload["usage"] == {"include": True}


def test_payload_without_reasoning_effort():
    payload = contestants.build_payload(
        "some/slug", "V2", "u", reasoning_effort=None, max_tokens=2048
    )
    assert "reasoning" not in payload


def test_predict_success_parses_label_and_cost():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return ok_body(label="pay_bill",
                       usage={"prompt_tokens": 1000, "completion_tokens": 10,
                              "cost": 0.002})

    c = make_contestant(handler)
    result = c.predict("utterance", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert result["label"] == "pay_bill"          # trimmed
    assert result["raw"] == " pay_bill "
    assert result["probs"] is None
    assert result["error"] is None
    assert result["cost_usd"] == 0.002            # provider-reported cost wins
    assert result["latency_ms"] >= 0
    assert len(calls) == 1
    assert calls[0].headers["Authorization"] == "Bearer test-key"


def test_cost_fallback_computed_from_token_pricing():
    usage = {"prompt_tokens": 1000, "completion_tokens": 100}
    expected = 1000 * 0.75 / 1e6 + 100 * 3.75 / 1e6
    c = make_contestant(lambda request: ok_body(usage=usage))
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert result["cost_usd"] == pytest.approx(expected)


def test_retry_on_429_then_success():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return ok_body()

    c = make_contestant(handler)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert result["error"] is None and result["label"] == "pay_bill"
    assert len(calls) == 2


def test_gives_up_after_three_5xx_attempts():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, json={"error": "boom"})

    c = make_contestant(handler)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert len(calls) == 3
    assert result["label"] is None
    assert result["error"] is not None and "gave up after 3 attempts" in result["error"]
    assert result["cost_usd"] == 0.0


def test_non_retryable_4xx_fails_immediately():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": "bad key"})

    c = make_contestant(handler)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert len(calls) == 1
    assert result["error"] is not None and "401" in result["error"]


def test_network_error_retried_then_recorded():
    import httpx as _hx
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) < 3:
            raise _hx.ConnectError("conn refused")
        return ok_body()

    c = make_contestant(handler)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert result["error"] is None and result["label"] == "pay_bill"
    assert len(calls) == 3


def test_parity_guard_rejects_mismatched_options():
    c = make_contestant(lambda request: ok_body())
    with pytest.raises(ValueError, match="frozen schema"):
        c.predict("u", list(schema.OPTIONS)[:-1], schema.DEFINITIONS, "V1")


def test_empty_content_is_label_none_without_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": ""}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 0},
        })

    c = make_contestant(handler)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert result["label"] is None and result["error"] is None


def test_openrouter_balance_parses_credits_and_key_limit(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/credits"):
            return httpx.Response(200, json={"data": {"total_credits": 10.0, "total_usage": 0.19}})
        return httpx.Response(200, json={"data": {"limit_remaining": 9.81}})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client  # capture before patching: the lambda must not call itself
    monkeypatch.setattr(contestants.httpx, "Client",
                        lambda *a, **k: real_client(transport=transport))
    balance = contestants.openrouter_balance()
    assert balance == {"account_remaining_usd": 9.81, "key_limit_remaining_usd": 9.81}


def test_unknown_contestant_key_raises():
    with pytest.raises(KeyError, match="unknown contestant"):
        contestants.build_contestant("not-a-model")


def test_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(contestants.Path, "exists", lambda self: False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        contestants.api_key()
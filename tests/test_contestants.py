"""Offline tests for the OpenRouter adapter: payload shape, retry, cost.

Uses httpx.MockTransport so no network and no real API key are involved
(the key comes from the test environment).
"""

import json
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


def test_rate_limit_storm_survives_within_six_attempts():
    # models the 429 storm seen in the first concurrent run: several 429s in a
    # row, then success — must survive without giving up.
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= 4:
            return httpx.Response(429, json={"error": "rate limited"})
        return ok_body()

    c = make_contestant(handler)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert result["error"] is None and result["label"] == "pay_bill"
    assert len(calls) == 5


def test_gives_up_after_six_5xx_attempts():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, json={"error": "boom"})

    c = make_contestant(handler)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert len(calls) == contestants.OpenRouterContestant.MAX_ATTEMPTS == 6
    assert result["label"] is None
    assert result["error"] is not None and "gave up after 6 attempts" in result["error"]
    assert result["cost_usd"] == 0.0


def test_429_retry_after_header_is_honored(monkeypatch):
    calls = []
    sleeps = []
    monkeypatch.setattr(contestants.time, "sleep", lambda s: sleeps.append(s))

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "rate limited"},
                                  headers={"Retry-After": "0"})
        return ok_body()

    c = make_contestant(handler)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert result["error"] is None and len(calls) == 2
    assert sleeps == [0.0]  # Retry-After overrode the default ladder


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


# --- Jev adapter (TypeSafe official API; offline via MockTransport) ----------

def jev_body(choice="balance", input_tokens=2000):
    return httpx.Response(200, json={
        "model": "jev-1.13.0",
        "answers": {"intent": {"type": "choice", "choice": choice,
                               "confidence": 0.9,
                               "probabilities": {"balance": 0.9, "pay_bill": 0.1}}},
        "usage": {"input_tokens": input_tokens, "output_tokens": 31},
    })


def make_jev(handler, monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", "jev-test-key")
    c = contestants.JevContestant(
        client=httpx.Client(transport=httpx.MockTransport(handler)))
    c.BACKOFF_S = (0.0, 0.0, 0.0, 0.0, 0.0)
    return c


def test_jev_parses_choice_probs_and_cost(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        return jev_body(choice="balance", input_tokens=2500)

    c = make_jev(handler, monkeypatch)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert result["label"] == "balance"
    assert result["probs"] == {"balance": 0.9, "pay_bill": 0.1}
    assert result["error"] is None
    assert result["cost_usd"] == pytest.approx(2500 * 0.042 / 1e6)
    payload = calls[0]
    assert payload["model"] == "jev-1.13.0"
    assert payload["state"] == "u"
    q = payload["questions"]["intent"]
    assert q["type"] == "choice"
    assert q["criteria"] == {o: schema.DEFINITIONS[o] for o in schema.OPTIONS}


def test_jev_retries_529_then_succeeds(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(529, json={"detail": "overloaded"})
        return jev_body()

    c = make_jev(handler, monkeypatch)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert result["error"] is None and len(calls) == 2


def test_jev_non_retryable_4xx_becomes_error_row(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"detail": "bad key"})

    c = make_jev(handler, monkeypatch)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert result["label"] is None
    assert result["error"] is not None and "401" in result["error"]


def test_jev_gives_up_after_six_attempts(monkeypatch):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(529, json={"detail": "overloaded"})

    c = make_jev(handler, monkeypatch)
    result = c.predict("u", list(schema.OPTIONS), schema.DEFINITIONS, "V1")
    assert len(calls) == contestants.JevContestant.MAX_ATTEMPTS == 6
    assert result["error"] is not None and "gave up after 6 attempts" in result["error"]


def test_jev_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    monkeypatch.setattr(contestants.Path, "exists", lambda self: False)
    with pytest.raises(RuntimeError, match="JEV_API_KEY"):
        contestants.jev_api_key()


def test_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(contestants.Path, "exists", lambda self: False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        contestants.api_key()
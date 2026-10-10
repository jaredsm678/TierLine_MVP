"""Contestant adapter contract and OpenRouter LLM adapters (design doc §7).

Session 2 ships the shared OpenRouter plumbing plus one adapter
(Gemini 3.8 Flash, High). The remaining LLM adapters (session 3) join the
OPENROUTER_MODELS table; the System-One and local-Gemma adapters (session 4)
get their own classes behind build_contestant.

Adapter contract (§7.2) — every contestant implements exactly this:

    predict(state_text, options, definitions, prompt_version) -> dict with keys
    {"label": str|None, "probs": dict[str,float]|None, "latency_ms": int,
     "cost_usd": float, "raw": str|None, "error": str|None}

OpenRouter call rules (§7.2): POST /v1/chat/completions; single user message
(no system role); temperature 0; max_tokens 2048 default; reasoning/effort set
where supported and recorded; retry 3 attempts with exponential backoff on
429/5xx/network errors; after 3 failures the row is recorded with an error
(excluded from accuracy by the runner, counted in the summary).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

from tierline_mvp import prompts, schema

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Pinned at setup (§7.1): slugs verified against the public OpenRouter models
# API on 2026-10-09; prices are USD per 1M tokens from the same snapshot.
# The served model id + date are also recorded in the run MANIFEST (session 5).
OPENROUTER_MODELS: dict[str, dict] = {
    "gemini-3.8-flash": {
        "display_name": "Gemini 3.8 Flash, High",
        "slug": "google/gemini-3.8-flash",
        "reasoning_effort": "high",
        "max_tokens": 2048,
        "price_prompt_per_1m_usd": 0.75,
        "price_completion_per_1m_usd": 3.75,
    },
    "gpt-6.1-sol": {
        "display_name": "GPT-6.1 Sol, Max Effort",
        "slug": "openai/gpt-6.1-sol",
        "reasoning_effort": "max",
        "max_tokens": 8192,  # thinking headroom (§7.2)
        "price_prompt_per_1m_usd": 2.00,
        "price_completion_per_1m_usd": 10.00,
    },
    "claude-5.5-opus-thinking": {
        # Roster display name per §7.1; there is no separate "-thinking" slug:
        # thinking is enabled via the reasoning parameter on this model.
        "display_name": "Claude 5.5 Opus Thinking, Max Effort",
        "slug": "anthropic/claude-opus-5.5",
        "reasoning_effort": "max",
        "max_tokens": 16384,  # max-effort thinking needs generous headroom
        "price_prompt_per_1m_usd": 4.00,
        "price_completion_per_1m_usd": 20.00,
    },
    "kimi-k3": {
        # Roster gives no effort level for Kimi; §7.1 rationale says the four
        # API LLMs run at high-reasoning settings, so effort=high (recorded).
        "display_name": "Kimi K3",
        "slug": "moonshotai/kimi-k3",
        "reasoning_effort": "high",
        "max_tokens": 8192,
        "price_prompt_per_1m_usd": 0.64,
        "price_completion_per_1m_usd": 13.50,
    },
}

RESULT_KEYS = ("label", "probs", "latency_ms", "cost_usd", "raw", "error")


def error_result(message: str) -> dict:
    return {"label": None, "probs": None, "latency_ms": 0, "cost_usd": 0.0,
            "raw": None, "error": message}


def api_key() -> str:
    """OPENROUTER_API_KEY from the environment, or a git-ignored .env file.

    §10 mandates env vars only; the value is never logged or written to
    results. The .env fallback exists because the app's shell may not inherit
    the user's interactive environment; .env is in .gitignore.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENROUTER_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set: export it as an env var or put "
        "'OPENROUTER_API_KEY=...' in a git-ignored .env at the repo root"
    )


def build_payload(
    slug: str,
    prompt_version: str,
    utterance: str,
    *,
    reasoning_effort: str | None,
    max_tokens: int,
) -> dict:
    """The request body sent to OpenRouter (pure; unit-tested offline)."""
    payload = {
        "model": slug,
        # §6: single message, no system role, so every provider gets
        # equivalent input.
        "messages": [{"role": "user", "content": prompts.render_prompt(prompt_version, utterance)}],
        "temperature": 0,        # §2 row 8
        "max_tokens": max_tokens,
        # OpenRouter cost accounting: returns usage with an authoritative cost.
        "usage": {"include": True},
    }
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}  # recorded per §7.2
    return payload


class Contestant:
    """§7.2 adapter contract."""

    name: str = "unset"

    def predict(
        self,
        state_text: str,
        options: list[str],
        definitions: dict[str, str],
        prompt_version: str,
    ) -> dict:
        raise NotImplementedError


class OpenRouterContestant(Contestant):
    MAX_ATTEMPTS = 3
    BACKOFF_S = (1.0, 2.0)  # exponential backoff before attempts 2 and 3

    def __init__(self, key: str, client: httpx.Client | None = None):
        meta = OPENROUTER_MODELS[key]
        self.key_id = key
        self.name = meta["display_name"]
        self.slug = meta["slug"]
        self.reasoning_effort = meta["reasoning_effort"]
        self.max_tokens = meta["max_tokens"]
        self.price_prompt = meta["price_prompt_per_1m_usd"]
        self.price_completion = meta["price_completion_per_1m_usd"]
        self._client = client

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=120)
        return self._client

    def predict(
        self,
        state_text: str,
        options: list[str],
        definitions: dict[str, str],
        prompt_version: str,
    ) -> dict:
        # Parity guard: LLM prompts render from the frozen constants (§5/§6);
        # whatever the caller passes must match them exactly.
        if list(options) != list(schema.OPTIONS) or dict(definitions) != schema.DEFINITIONS:
            raise ValueError("options/definitions do not match the frozen schema")

        payload = build_payload(
            self.slug, prompt_version, state_text,
            reasoning_effort=self.reasoning_effort, max_tokens=self.max_tokens,
        )
        headers = {"Authorization": f"Bearer {api_key()}"}

        last_error = None
        for attempt in range(self.MAX_ATTEMPTS):
            if attempt:
                time.sleep(self.BACKOFF_S[min(attempt - 1, len(self.BACKOFF_S) - 1)])
            started = time.perf_counter()
            try:
                resp = self._http().post(OPENROUTER_URL, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                last_error = f"network error: {type(exc).__name__}"
                continue
            latency_ms = int((time.perf_counter() - started) * 1000)
            if resp.status_code == 200:
                return self._parse(resp.json(), latency_ms)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
                continue
            # Other 4xx: not retryable (bad request, auth, ...).
            return error_result(f"HTTP {resp.status_code}: {resp.text[:200]}")
        return error_result(f"gave up after {self.MAX_ATTEMPTS} attempts: {last_error}")

    def _parse(self, data: dict, latency_ms: int) -> dict:
        # §8: the final answer is the returned message content, with thinking
        # stripped by the provider; scoring.py normalizes further (session 6).
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content")
        label = content.strip() if isinstance(content, str) and content.strip() else None
        return {
            "label": label,
            "probs": None,  # LLMs: no native distribution over the 16 options
            "latency_ms": latency_ms,
            "cost_usd": self._cost_usd(data.get("usage") or {}),
            "raw": content,
            "error": None,
        }

    def _cost_usd(self, usage: dict) -> float:
        cost = usage.get("cost")
        if isinstance(cost, (int, float)):
            return float(cost)
        prompt_tokens = usage.get("prompt_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or 0
        return (
            prompt_tokens * self.price_prompt / 1e6
            + completion_tokens * self.price_completion / 1e6
        )


def build_contestant(key: str, client: httpx.Client | None = None) -> Contestant:
    """Instantiate an adapter by roster key (session 4 adds the local models)."""
    if key in OPENROUTER_MODELS:
        return OpenRouterContestant(key, client=client)
    raise KeyError(f"unknown contestant {key!r}; known: {sorted(OPENROUTER_MODELS)}")
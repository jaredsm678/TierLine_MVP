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

import json
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

    def _endpoint(self) -> str:
        return OPENROUTER_URL

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {api_key()}"}

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
        headers = self._headers()

        last_error = None
        for attempt in range(self.MAX_ATTEMPTS):
            if attempt:
                time.sleep(self.BACKOFF_S[min(attempt - 1, len(self.BACKOFF_S) - 1)])
            started = time.perf_counter()
            try:
                resp = self._http().post(self._endpoint(), headers=headers, json=payload)
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


class LocalOpenAIContestant(OpenRouterContestant):
    """LLM contestants served by an OpenAI-compatible local server (§7.1: the
    local Gemma via LM Studio). Same prompt path, contract, and retry logic as
    the OpenRouter adapters; no API key and zero cost. The runner must keep
    this single-worker: concurrent requests have wedged LM Studio queues
    before (§10.3)."""

    def __init__(self, *, name: str, slug: str,
                 base_url: str = "http://localhost:1234/v1", max_tokens: int = 2048):
        self.key_id = "gemma-4-12b"
        self.name = name
        self.slug = slug
        self.reasoning_effort = None  # no reasoning-effort control locally
        self.max_tokens = max_tokens
        self.price_prompt = 0.0
        self.price_completion = 0.0
        self._client = None
        self._base_url = base_url.rstrip("/")

    def _endpoint(self) -> str:
        return f"{self._base_url}/chat/completions"

    def _headers(self) -> dict:
        return {"Authorization": "Bearer lm-studio"}  # server ignores the value


# ---------------------------------------------------------------------------
# System-One adapters (§7.1 rows 1-5; local CPU)
#
# Shared shape (§7.2): the utterance is the state; one Choice question whose
# criteria map every option to its frozen §5 definition, so the S1 models see
# exactly the information the LLM prompt carries. Native label + native
# probability distribution are recorded (probs; calibration analysis is
# POST-MVP — scoring uses the label only). prompt_version is accepted and
# ignored: S1 models have a fixed native input format, so wording variants do
# not apply (§6.2, disclosed).
# ---------------------------------------------------------------------------

# §5 task framing, verbatim — the S1 equivalent of the prompt's framing line.
S1_INSTRUCTIONS = "Given one customer request, decide which of the following handlers should take it."


class SystemOneContestant(Contestant):
    instructions = S1_INSTRUCTIONS

    def __init__(self):
        self._model = None  # lazy: weights download on first use (§10.2)

    def predict(
        self,
        state_text: str,
        options: list[str],
        definitions: dict[str, str],
        prompt_version: str,
    ) -> dict:
        # Same parity guard as the LLM adapters: the frozen schema is the only
        # option set any contestant may see.
        if list(options) != list(schema.OPTIONS) or dict(definitions) != schema.DEFINITIONS:
            raise ValueError("options/definitions do not match the frozen schema")
        started = time.perf_counter()
        try:
            label, probs, raw = self._system_one(state_text, list(options), dict(definitions))
        except Exception as exc:  # noqa: BLE001 — recorded as an error row (§7.2)
            return error_result(f"{type(exc).__name__}: {exc}")
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {"label": label, "probs": probs, "latency_ms": latency_ms,
                "cost_usd": 0.0, "raw": raw, "error": None}

    def _criteria(self, options: list[str], definitions: dict[str, str]) -> dict[str, str]:
        return {option: definitions[option] for option in options}

    def _system_one(self, state_text: str, options: list[str],
                    definitions: dict[str, str]) -> tuple[str, dict | None, str]:
        raise NotImplementedError


class OpenDeciderContestant(SystemOneContestant):
    name = "opendecider-nano"

    def _system_one(self, state_text, options, definitions):
        import opendecider as od
        if self._model is None:
            self._model = od.load("manjunathshiva/opendecider-nano", device="cpu")
        question = {"intent": od.Choice(
            instructions=self.instructions,
            criteria=self._criteria(options, definitions),
        )}
        resp = self._model.system_one(state_text, question)
        answer = resp["answers"]["intent"]
        return answer["choice"], answer.get("probabilities"), _json(resp)


class LayaContestant(SystemOneContestant):
    name = "laya-421m typed-decisions"

    def _system_one(self, state_text, options, definitions):
        from laya import LayaRouter
        if self._model is None:
            self._model = LayaRouter(
                criteria=self._criteria(options, definitions),
                instructions=self.instructions,
                model="typed-decisions",  # §7.1: convaiinnovations/laya, typed-decisions revision
            )
        label = self._model.invoke(state_text)
        decision = self._model.last_decision
        answer = decision["answers"]["route"]  # LayaRouter names its question "route"
        return answer.get("choice", label), answer.get("probabilities"), _json(decision)


class DeciderContestant(SystemOneContestant):
    name = "decider-2b"

    def _system_one(self, state_text, options, definitions):
        from decider.infer import Decider
        if self._model is None:
            self._model = Decider("Mapika/decider-2b", device="cpu")
        question = {"intent": {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": self._criteria(options, definitions),
        }}
        resp = self._model.system_one(state_text, question)
        answer = resp["answers"]["intent"]
        return answer["choice"], answer.get("probabilities"), _json(resp)


class LevContestant(SystemOneContestant):
    name = "lev"

    def _system_one(self, state_text, options, definitions):
        import lev
        if self._model is None:
            self._model = lev.load("interfaze-ai/lev")
        question = {"intent": {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": self._criteria(options, definitions),
        }}
        resp = self._model.system_one(state_text, question)
        answer = resp.answers["intent"]
        return answer.choice, dict(answer.probabilities), _json(_dump(resp))


class VonContestant(SystemOneContestant):
    name = "von"

    def _system_one(self, state_text, options, definitions):
        import von
        if self._model is None:
            von.set_backend("von-1.3")  # doc pins von-1.3.0; served id recorded in raw
            self._model = von
        question = {"intent": von.Choice(
            instructions=self.instructions,
            criteria=self._criteria(options, definitions),
        )}
        resp = von.system_one(state_text, question)
        answer = resp.answers["intent"]
        return answer.choice, dict(answer.probabilities), _json(_dump(resp))


def _dump(obj):
    """Pydantic v2 models -> plain dict; anything else passes through."""
    return obj.model_dump() if hasattr(obj, "model_dump") else obj


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# Roster keys beyond the OpenRouter table (§7.1 rows 1-5 and row 10).
LOCAL_CONTESTANT_KEYS = (
    "opendecider-nano",
    "laya-421m",
    "decider-2b",
    "lev",
    "von",
    "gemma-4-12b",
)


def build_contestant(key: str, client: httpx.Client | None = None) -> Contestant:
    """Instantiate an adapter by roster key (all ten contestants, §7.1)."""
    if key in OPENROUTER_MODELS:
        return OpenRouterContestant(key, client=client)
    if key == "gemma-4-12b":
        # Served model id verified against LM Studio's /v1/models at smoke time.
        return LocalOpenAIContestant(name="Gemma-4-12B-it, 4-bit MLX", slug="gemma-4-12b-it")
    if key == "opendecider-nano":
        return OpenDeciderContestant()
    if key == "laya-421m":
        return LayaContestant()
    if key == "decider-2b":
        return DeciderContestant()
    if key == "lev":
        return LevContestant()
    if key == "von":
        return VonContestant()
    raise KeyError(f"unknown contestant {key!r}; known: "
                   f"{sorted(list(OPENROUTER_MODELS) + list(LOCAL_CONTESTANT_KEYS))}")
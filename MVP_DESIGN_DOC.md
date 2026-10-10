# TierLine-MVP — Design Document

**Version:** 1.0 · **Date:** 2026-10-09 · **Owner:** Jared Santamaria
**Purpose:** This document is the complete, self-contained specification for building the TierLine MVP. It is written so that a fresh conversation in an empty repository can implement the entire MVP from this file alone, with no other context.
**Status:** All decisions below are LOCKED. Anything marked POST-MVP is explicitly out of scope.
**Amendment (2026-10-09, session 4):** Roster row 10 changed by owner decision — Gemma moves from the
local LM Studio build (Gemma-4-12B-it, 4-bit MLX) to the OpenRouter-hosted `google/gemma-4-31b-it`.
Rationale: the Kaggle Gemma competition (Gemma 4 Developer Agent Competition) mandates
`gemma-4-31b-it-qat-w4a16-ct` — a 31B-it QAT quantization — so the doc's original 12B pick never
matched the referenced competition; 31B-it does. Quantization/serving-stack differences vs the
competition's W4A16 build are disclosed.

---

## 1. What the MVP is (one paragraph)

A difficulty-stratified evaluation of **System-One decision models vs. frontier LLMs** on a fixed routing task built from the **CLINC150** dataset's banking domain: 150 test utterances (75 "difficult" from a published TF-IDF-difficult subset, 75 ordinary), 16 routing options (15 banking intents + `out_of_scope`), one frozen prompt (plus one reworded variant), ten contestants, final-answer scoring, and a small set of pre-registered analyses. The MVP's job: (1) verify the harness reproduces the published difficult-subset accuracy drop, (2) measure whether the System-One vs. LLM gap changes between ordinary and difficult items, (3) produce honest per-model numbers with confidence intervals. Fine-grained boundary mapping is the post-MVP expansion.

---

## 2. Locked decisions (do not revisit during the build)

| # | Decision | Value |
|---|---|---|
| 1 | Dataset | CLINC150 (`data_full.json`), **banking domain only**, **test split only** |
| 2 | Option set | The 15 banking intents + `out_of_scope` (16 options, prompt-only for oos) |
| 3 | Eval set | 150 rows = 15 intents × (5 difficult + 5 non-difficult) |
| 4 | Difficult rows | Taken verbatim from Qi et al.'s released TF-IDF-difficult subset (no recomputation) |
| 5 | Difficulty buckets | Binary: difficult / non-difficult. Multi-level bands are POST-MVP |
| 6 | oos gold rows | **None.** oos is a prompt option only; false-abstention is measured, abstention-recall is deferred |
| 7 | Prompt | 2 versions (V1 canonical, V2 rewording); definitions byte-identical; options in fixed alphabetical order |
| 8 | Scoring | Final answer; trim→lowercase→strip quotes/punct→exact match vs 16 labels; else invalid = **wrong**; temperature 0 |
| 9 | Roster | 5 System-One (local) + 5 LLMs (OpenRouter), incl. Gemma-4-31B |
| 10 | Budget | Hard cap **$10**; expected $3–6; runner aborts past cap |
| 11 | Harness | Built from scratch; Python 3.12; stdlib + `httpx` only; no frameworks |
| 12 | Results | Append-only JSONL, resume-safe by `(run_id, case_id)`; MANIFEST with SHA-256s |
| 13 | Dataset gold | CLINC's own intent labels are ground truth, accepted as-is (no verification pass) |
| 14 | Analysis | Pre-registered set in §11; expanded after first results |
| 15 | Repo | New empty repo, name `tierline-mvp` |

**Rules of engagement for the build:** no scope creep; prompts and definitions are frozen text from §6 (do not paraphrase while coding); the budget guard is mandatory; the user reviews each build session's output before the next begins.

---

## 3. Why this dataset and task (justification, for the writeup)

- **CLINC150 is canonical.** Larson et al., *"An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,"* EMNLP-IJCNLP 2019 (ACL D19-1131; arXiv:1909.02027) — ~856+ citations. It introduced the out-of-scope (OOS) prediction task and is a standard member of the ATIS/SNIPS/BANKING77/CLINC150 benchmark quartet. Still actively used in the LLM era (e.g., *"Intent Detection in the Age of LLMs,"* Arora et al. 2024, arXiv:2410.01627; *"When Do LLMs Replace Fine-Tuned NLU?,"* arXiv:2608.20371).
- **The task shape matches the System-One use case.** TypeSafe's own Jev documentation demos a support-ticket triage decision (state + typed question); the ecosystem's canonical typed-decision example is a Choice over `{billing, technical, sales, needs_review}` with per-option criteria and an explicit "can't classify" home. A short utterance routed over a small labeled option set with an abstention option is exactly that geometry.
- **Published performance landscape (sets expectations):** fine-tuned encoders reach 96.9–97.1% in-scope accuracy on full CLINC150 (Larson 2019; IBM NAACL-Industry 2021); Claude Haiku zero-shot reaches 88.5% on the full 151-way task — statistically tied with fine-tuned RoBERTa (89.1%) — with far better OOS recall (85.6 vs 58.1) (arXiv:2608.20371). Consequence: ordinary CLINC items will be near-ceiling for strong LLMs; separation is expected on the difficult subset and in abstention behavior. **No published work evaluates Jev-class System-One models on CLINC150** — that is this MVP's novel measurement.
- **Domain-scoped sub-schema evaluation is precedented:** arXiv:2608.20371 runs schema-swap experiments on 75-intent halves of CLINC150; this MVP evaluates the banking domain (15 intents) exhaustively — no cherry-picking within the domain.

---

## 4. Dataset construction (exact recipe)

### 4.1 Source artifacts

| Artifact | URL | Role |
|---|---|---|
| CLINC150 `data_full.json` | `https://github.com/clinc/oos-eval` → `data/data_full.json` | utterances + intent labels + splits. SHA-256 of the copy already used in this project: `36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0` |
| CLINC150 `domains.json` | same repo → `data/domains.json` | official intent→domain mapping (10 domains × 15 intents). Banking = `freeze_account, routing, pin_change, bill_due, pay_bill, account_blocked, interest_rate, min_payment, bill_balance, transfer, order_checks, balance, spending_history, transactions, report_fraud` |
| IBM difficult subset | `https://raw.githubusercontent.com/haodeqi/BenchmarkingIntentDetection/master/CLINC150/clinc150-tfidf-testset.tsv` | 750 rows, TSV `utterance\tintent`, = 5 least-TF-IDF-similar test utterances per intent, all 150 intents. Local copy: `mvp-collection/sources/clinc150-tfidf-testset.tsv` |

Note: the IBM repo README states a 620-row test set; the actual released TSV has **750 rows** (verified: exactly 5 per intent for all 150 intents, 0 shortfalls). Cite the TSV as the artifact of record.

### 4.2 The IBM difficult-subset recipe (specify verbatim in the writeup)

From *"Benchmarking Commercial Intent Detection Services with Practice-Driven Evaluations"* (NAACL Industry Track 2021; code: `github.com/haodeqi/BenchmarkingIntentDetection`):

1. For each intent *i* (processed independently): tokenize all of intent *i*'s **training** utterances (whitespace tokenizer, punctuation ignored), concatenate, fit a TF-IDF vector space on that intent's own training text.
2. Transform each **test** utterance of intent *i* with that transformer; compute cosine similarity against the intent's training utterances.
3. Keep the **5 least-similar** test utterances per intent → the difficult test subset.
4. Models trained on full train sets are evaluated on this subset only.

Their published result (CLINC150): BERT-large 97.1% (full) → 89.5% (difficult); RoBERTa-base 97.0 → 88.4; Watson Assistant enhanced 95.7 → 86.0; RASA 89.4 → 68.3. **The MVP reproduces this drop pattern with its own roster.**

### 4.3 Construction procedure (implement exactly)

1. Load `data_full.json`; take the `test` split. Assert SHA-256 of the file matches the value above.
2. Restrict to the 15 banking intents (labels listed in §2 row 2). Every utterance is a `[text, intent]` pair.
3. Load the IBM TSV; parse each line as `utterance \t intent`; keep rows whose intent is one of the 15 banking intents → **75 difficult rows** (verified: exactly 5 per intent).
4. Verify: every difficult utterance must exist verbatim in the CLINC test split (assert; log any miss as a build failure).
5. For each of the 15 intents: from that intent's test rows **excluding** the 5 difficult ones (25 remain), sample **5 rows** with `random.Random(42)` (sort candidates by original file order first for determinism). → **75 non-difficult rows**.
6. Light dedup: normalize each row's text (lowercase, strip punctuation/whitespace); if a normalized duplicate pair exists, keep the first, replace the dropped row with the next candidate from the same intent's pool (same seed continuation).
7. Emit `data/eval_v0.jsonl` (150 rows) and `data/MANIFEST.json` (SHA-256 of eval file + both source artifacts, row counts per intent × difficulty group, seed, construction code version).

### 4.4 Row schema (`data/eval_v0.jsonl`, one JSON object per line)

```json
{
  "row_id": "bnk-0001",
  "split_source": "test",
  "intent": "pay_bill",
  "difficulty": "difficult | non_difficult",
  "source": "ibm_tfidf_testset | clinc_test_random",
  "text": "the customer's utterance, verbatim"
}
```

- `difficulty` comes solely from membership in the IBM subset. There is no other difficulty signal in the MVP.
- `out_of_scope` never appears as `intent` (no oos gold rows — see Limitations).

---

## 5. The task and option set

- **Task framing (routing):** "Given one customer request, decide which of the following handlers should take it."
- **16 options, frozen alphabetical order (identical in every prompt, every model, both versions):**

```
account_blocked, balance, bill_balance, bill_due, freeze_account, interest_rate,
min_payment, order_checks, out_of_scope, pay_bill, pin_change, report_fraud,
routing, spending_history, transactions, transfer
```

- **Definitions (frozen, pre-registered).** Methodology: authored following the *dependent + cleansed* intent-description design of Hong et al., *"Exploring the Use of Natural Language Descriptions of Intents for Large Language Models in Zero-shot Intent Classification,"* SIGDIAL 2024 (ACL 2024.sigdial-1.39) — each definition explicitly distinguishes the option from its neighbors. Disclosed simplification vs. their optimal recipe: single-author (not LLM-generated + independently cleansed), scoped to the 16-option sub-schema rather than the full 150-intent space, and not empirically ablated. Their paper's finding that **test-time description quality matters most** is the reason these are frozen and reviewed. Full reproduction of their generation recipe across all domains is the full-project plan.

```
- account_blocked: The customer reports that their account is blocked, locked, or frozen — or asks why.
  (A request to impose a block is `freeze_account`.)
- balance: The customer asks how much money is in an account right now — a point-in-time amount.
  (A list of activity is `transactions`; a spending total is `spending_history`; an amount owed on a bill is `bill_balance`.)
- bill_balance: The customer asks how much they owe on a bill — the total amount due.
  (The due date is `bill_due`; the smallest allowed payment is `min_payment`.)
- bill_due: The customer asks when a payment is due — the date or deadline.
  (The amount owed is `bill_balance`; making the payment is `pay_bill`.)
- freeze_account: The customer asks to block, freeze, or restrict activity on their account.
  (An already-blocked account being asked about is `account_blocked`.)
- interest_rate: The customer asks about an interest rate — charged or earned — on any account, loan, or card.
  (An amount owed or available is not a rate question.)
- min_payment: The customer asks for the minimum amount they are required to pay.
  (The full amount owed is `bill_balance`; the due date is `bill_due`.)
- order_checks: The customer wants to order checkbooks or more checks.
  (Paying money out is `transfer` or `pay_bill`, not ordering checks.)
- out_of_scope: The request is none of the 15 banking requests above — for example: device commands, other
  services or companies, general knowledge questions, or small talk.
- pay_bill: The customer wants to make a payment toward a bill, or asks for help paying one.
  (The due date is `bill_due`; the amount owed is `bill_balance`.)
- pin_change: The customer wants to change, set, or reset a PIN — including having forgotten it.
- report_fraud: The customer reports unauthorized or suspicious charges or activity, or asks how to report fraud.
  (An ordinary lookup of their own charges is `transactions`.)
- routing: The customer asks for a bank or account routing number.
- spending_history: The customer asks for totals or summaries of spending, usually by category or time period.
  (Individual entries are `transactions`; the current amount available is `balance`.)
- transactions: The customer asks to see individual account activity — a list or detail of charges, purchases, or deposits.
  (A point-in-time amount is `balance`; a spending summary is `spending_history`.)
- transfer: The customer asks to move money between accounts.
  (Viewing past activity is `transactions`; paying a bill is `pay_bill`.)
```

---

## 6. Prompt system (frozen text)

Common rules: single message (no system role) so every provider receives equivalent input; utterance fenced in triple quotes; temperature 0; the 16 definitions are byte-identical across versions; option order alphabetical in both versions.

### 6.1 Version V1 (canonical)

```
You will route one customer request to the correct handler.

The 16 handlers are:
<§5 definitions block, verbatim>

Request:
"""
<utterance>
"""

Reply with exactly one handler name from the list above — nothing else.
```

### 6.2 Version V2 (rewording — framing only; definitions and order byte-identical)

```
Which of the following handlers should take this request?

Handlers and their criteria:
<§5 definitions block, verbatim>

The request:
"""
<utterance>
"""

Answer with a single handler name and nothing else.
```

- V1 runs on **all 10 contestants**. V2 runs on **LLM contestants only** (System-One models have a fixed native input format; wording variants do not apply — disclosed).
- The **names-only ablation** (prompt with label list, no definitions) is POST-MVP.

---

## 7. Contestant roster & adapter contract

### 7.1 Roster (10 contestants)

| # | Class | Model | Where | Access |
|---|---|---|---|---|
| 1 | System-One | opendecider-nano (`manjunathshiva/opendecider-nano`) | local CPU | pip `opendecider` |
| 2 | System-One | laya-421m typed-decisions (`convaiinnovations/laya`) | local CPU | pip `laya` |
| 3 | System-One | decider-2b (`Mapika/decider-2b`) | local CPU | pip `decider-ai` |
| 4 | System-One | lev (`interfaze-ai/lev`) | local CPU | pip `lev` (github.com/Abhinavexists/lev) |
| 5 | System-One | von (`wfzyx/von`, von-1.3.0) | local CPU | pip `von` (von-sdk) |
| 6 | LLM | GPT-6.1 Sol, Max Effort | OpenRouter | API |
| 7 | LLM | Claude 5.5 Opus Thinking, Max Effort | OpenRouter | API |
| 8 | LLM | Kimi K3 | OpenRouter | API |
| 9 | LLM | Gemini 3.8 Flash, High | OpenRouter | API |
| 10 | LLM | Gemma-4-31B-it | OpenRouter | API |

- Leaderboard names are display names; **verify exact OpenRouter model slugs at setup** and record the served ID + date in the MANIFEST.
- Rationale (one line each, for the writeup): 5 S1 models = the open System-One class with published/verified runtimes; the 4 API LLMs = one per major lab at high-reasoning settings (frontier anchor); Gemma-4-31B-it = the open-weights LLM, size-matched to the variant mandated by the Kaggle Gemma 4 Developer Agent Competition (`gemma-4-31b-it-qat-w4a16-ct`), served via OpenRouter (amended 2026-10-09; was local 12B — see Amendment note).

### 7.2 Adapter contract (every contestant implements exactly this)

```python
class Contestant:
    name: str                      # pinned display name
    def predict(self, state_text: str, options: list[str],
                definitions: dict[str, str], prompt_version: str) -> dict
    # returns: {"label": str|None, "probs": dict[str,float]|None,
    #           "latency_ms": int, "cost_usd": float, "raw": str|None,
    #           "error": str|None}
```

- **System-One adapters:** pass `state_text` as the state and a single Choice question whose `criteria` map each option to **its frozen definition** (information parity with the LLM prompt). Read the native label + native probability distribution; record probs under `probs` (analysis of calibration is POST-MVP — scoring uses the label only).
- **LLM adapters (OpenRouter):** `POST /v1/chat/completions`, single user message = rendered prompt, `temperature: 0`, `max_tokens` per-model (2048 default; thinking models need headroom), reasoning/effort parameter set to the model's Max/High setting where supported and recorded. Retry: 3 attempts, exponential backoff, on 429/5xx/network. After 3 failures: row recorded with `error`, **excluded from accuracy**, counted in the run summary.
- **Gemma adapter:** LM Studio local server (OpenAI-compatible), same contract, no API key.
- **Cost meter:** sum `cost_usd` across API calls; if the running total would exceed **$10**, the runner aborts before the next call with a clear message. Estimated total: ~$3–6 (1,200 calls: 150 rows × 2 versions × 4 API models).

---

## 8. Scoring (exact rules)

1. Take the model's final answer (LLM: the returned message content, thinking stripped by the provider; S1: the native label field).
2. Normalize: trim whitespace → lowercase → strip surrounding quotes → strip trailing punctuation (`. , ! ? : ;`).
3. **Valid** iff the normalized answer is exactly one of the 16 option strings. Otherwise: **invalid → scored wrong**, and counted in the invalid-output rate.
4. Correct iff valid AND normalized answer == the row's `intent`.
5. An `oos` prediction on an in-scope row is scored wrong, and additionally counted in the model's **oos-selection rate** (false abstention).
6. No probability pooling, no per-model threshold tuning, no rescoring. Every scored row appears in exactly one accuracy number per (model, version, difficulty group).

---

## 9. Harness architecture (build from scratch, six small modules)

```
tierline_mvp/
├── data/eval_v0.jsonl        # frozen eval set (built by scripts/build_dataset.py)
├── data/MANIFEST.json        # sha256 + counts + seed + code version
├── src/tierline_mvp/
│   ├── schema.py             # row validation, option list, definitions (frozen text)
│   ├── prompts.py            # V1/V2 renderers (§6 text lives here, verbatim)
│   ├── contestants.py        # adapter contract + 10 adapters (§7)
│   ├── runner.py             # CLI: --model --version --workers; resume; cost guard
│   ├── scoring.py            # §8 normalization + aggregation
│   └── report.py             # tables + CIs
├── results/runs/<run_id>.jsonl
├── results/summary.md
├── tests/                    # schema, prompt rendering, scoring edge cases
└── pyproject.toml            # python 3.12; deps: httpx (+ per-model pip packages)
```

- **Results row schema:** `{run_id, row_id, model, prompt_version, label, probs, latency_ms, cost_usd, raw, error, ts}` — append-only; a rerun with the same `run_id` skips rows already present (resume-safety).
- **`run_id`** = `v1-{model}-{prompt_version}` (deterministic; one file per model×version).
- **Ordering:** rows run in file order; API models may use concurrency ≤ 4; local models sequential.
- **No dashboard, no database, no config framework.** One config dict in code, pinned.

---

## 10. Environment & setup checklist (execute before Session 1)

1. **Python 3.12** venv; `pip install httpx pytest`.
2. **System-One packages:** `pip install laya opendecider decider-ai lev von` (weights download on first use; verify each imports and answers one smoke row).
3. ~~Gemma-4-12B-it (MLX 4-bit) in LM Studio~~ — **obsolete after the 2026-10-09 amendment:** Gemma runs via OpenRouter (§7.1 row 10); no local server needed. (The LM Studio single-worker note still applies to any future local LLM.)
4. **OpenRouter account** → API key → env var `OPENROUTER_API_KEY`. Verify each of the 4 model slugs resolves with a 1-token call; record served IDs and prices in `results/MANIFEST.json`.
5. **Download artifacts** (§4.1) into `data/sources/`; verify SHA-256 of `data_full.json` against the value in §4.1.
6. Write `data/eval_v0.jsonl` + `MANIFEST.json` per §4.3; assert 150 rows, 15 intents × (5 difficult + 5 non-difficult).

**API key handling:** env vars only; never committed; never printed to logs or results.

---

## 11. Pre-registered analysis (freeze before any model runs)

Primary outputs of `report.py` — per contestant, per prompt version, with **Wilson 95% CIs** on every proportion:

1. **Difficult-subset replication:** accuracy on difficult rows vs non-difficult rows (per model, and per class S1-pooled vs LLM-pooled, descriptive). Expected: a drop on difficult rows; **if no model drops, suspect the harness before the thesis.**
2. **Class gap by difficulty group:** S1-group accuracy vs LLM-group accuracy within difficult rows and within non-difficult rows. The MVP's boundary question in miniature: *does the S1↔LLM gap change between ordinary and difficult items?* Direction is NOT pre-committed — a tie at D1 with a gap on difficult rows, a flat gap, or an S1 win anywhere are all reportable findings.
3. **False-abstention rate:** fraction of in-scope rows answered `out_of_scope`, per model. (Context for the writeup: the original CLINC paper and follow-ups show OOS handling is the weak axis for trained classifiers; LLMs reach high OOS recall when oos is an explicit option — arXiv:2608.20371 reports 85.6 recall vs 58.1 fine-tuned.)
4. **Invalid-output rate** per model (Hayk protocol: invalid = wrong, reported separately).
5. **V1-vs-V2 spread** per LLM model (paired, same rows) — wording sensitivity.

Statistical discipline: same 150 rows for every model (paired design); Wilson CIs everywhere; McNemar for paired model contrasts where useful; **no fine-grained ranking claims** — at this n, only large gaps (≥ ~15–20 points) are interpretable, consistent with the literature's finding that even large studies cannot cleanly separate top models.

**Explicit expansion clause:** after the first full results are viewed, additional analyses (per-intent breakdowns, confusion structure, per-cluster behavior of the balance/transactions/spending_history family, calibration of S1 native probabilities, cost/latency tables) are permitted and expected — they are exploratory and must be labeled as such. The five pre-registered items above are the confirmatory set.

---

## 12. Limitations (state verbatim in the writeup)

1. **oos is schema-relative.** The MVP evaluates the banking sub-schema, so `out_of_scope` means "none of these 15 intents." Utterances from other CLINC domains would also be oos under this schema but are not represented in the eval set; the oos gold rows are absent entirely. Therefore the MVP measures **false abstention only**, not abstention recall. The full oos picture returns when other domains are added, and via published hard-negative oos data (Li, Larson & Leach, LREC 2024 — their released CLINC150 hard-negative oos set is the intended source; their results show oos difficulty is real: BERT oos AUROC 0.968 general → 0.914 hard-negative).
2. **Two difficulty buckets.** The MVP uses the published binary difficult/non-difficult split. Multi-level graded difficulty (human rubric bands, empirical/IRT difficulty per Rodríguez et al. 2021, training-dynamics bands per Cosma et al. 2024) is the full-project design; the MVP validates the harness on the binary split first.
3. **Single-domain scope.** Banking only, by design; the remaining 9 domains are the expansion plan. Domain choice is disclosed as a scope cut, not a claim that banking represents all support routing.
4. **Dataset gold accepted as-is.** CLINC150's crowdsourced labels occasionally contain noise (e.g., a weather utterance labeled `transfer` exists inside the difficult subset). Per convention — and because the dataset is the canonical reference for hundreds of papers — its labels are treated as ground truth without a verification pass.
5. **Definitions are single-author and sub-schema-scoped** (§5 methodology note): a disclosed simplification of the Hong et al. (2024) generate-then-cleanse recipe, which the full project adopts.
6. **Pretraining contamination.** CLINC150 is public and old enough that LLMs have likely seen it; this applies equally to all contestants and is disclosed. Absolute numbers may be inflated; comparisons remain internally valid.
7. **Small-n CIs.** 150 rows / 75 per difficulty group support directional findings and large-gap detection, not fine-grained model rankings.
8. **Zero-shot only.** No few-shot examples in prompts (matches the production "no labeled data" regime; few-shot is post-MVP).
9. **Latency is measured under load (amendment, 2026-10-09).** The MVP's LLM grid runs concurrently (≤4 calls in flight per model, 5 models at once) on a laptop shared with other work, so recorded latencies are contention-polluted upper bounds, not clean serving numbers. Latency stays in the results schema for directionality; any latency-sensitive analysis (e.g., the post-MVP cost/latency tables) must be re-measured on a quiet machine.

---

## 13. Build plan (six sessions, each ends with user review)

| Session | Deliverable | Acceptance criteria (user checks) |
|---|---|---|
| 1 | Repo scaffold, `schema.py`, `scripts/build_dataset.py` → `eval_v0.jsonl` + MANIFEST | 150 rows; counts 15×(5+5); SHA-256 checks pass; spot-check 10 rows vs sources |
| 2 | `prompts.py` (V1/V2 renderers) + one LLM adapter (Gemini 3.8 Flash via OpenRouter) end-to-end on 5 rows | Rendered prompts match §6 byte-for-byte; 5/5 rows scored; costs logged |
| 3 | Remaining 3 LLM adapters + invalid-handling + retry/cost guard | 5-row smoke per model; invalid rows demonstrably scored wrong; abort logic demoed at a $0.01 test cap |
| 4 | 5 System-One adapters (+ Gemma local) | Smoke row per model; parity confirmed (same definitions in criteria); native probs recorded |
| 5 | `runner.py` full run + resume demo (kill mid-run, restart, no duplicates) | 150×2×10 grid complete or budget-aborted cleanly; MANIFEST updated |
| 6 | `scoring.py` + `report.py` → `results/summary.md` | All 5 pre-registered outputs present with CIs; numbers recomputable from JSONL |

Session rule: each session is one conversation; the user reviews the diff and outputs before the next session starts. Keep modules small and readable — the user wants to understand every choice well enough to criticize it.

---

## 14. Post-MVP roadmap (the full project)

1. **Domain expansion:** add the remaining 9 CLINC domains one at a time (each: 15 intents × difficult+non-difficult rows), ending at the full 151-way schema. Each domain is a self-contained block; the expansion studies label-space size/scope effects on the S1↔LLM gap (framing from Arora et al. 2024).
2. **Graded difficulty (Metric A):** human rubric flags F1–F5 → band formula, validated by the monotonicity gate (accuracy declining across bands). This revives the already-drafted rubric machinery from the previous repository.
3. **Empirical difficulty (Metric C):** reference-panel outcomes → per-case empirical difficulty; IRT fit via `py-irt` (Rodríguez et al. 2021). The accuracy-vs-difficulty figure has literature precedent as the *system characteristic curve* (Martínez-Plumed et al. 2022), which also warns why outcome-only difficulty is circular — the citation that justifies keeping the outcome-free rubric.
4. **Two-tier oos:** general oos rows (easy) + **released CLINC150 hard-negative oos rows** (Li, Larson & Leach, LREC 2024; BERT oos AUROC 0.968 → 0.914 on hard-negatives) → abstention recall finally measured, with published provenance.
5. **Definitions recipe (full Hong et al.):** LLM-generated dependent descriptions over the full label space + human cleansing, replacing the single-author criteria.
6. **Calibration & mechanisms:** S1 native-probability calibration (ECE/reliability); confidence-gated cascade and router mechanisms (the Jev ecosystem's own deployment pattern) as the decision-layer payoff.
7. **Prompt-order robustness, names-only ablation, few-shot conditions** — each a cheap additive run on the frozen harness.

---

## 15. Related work & citations (as they appear above)

- Larson et al. 2019, *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*, EMNLP-IJCNLP (ACL D19-1131). Dataset + oos task.
- *Benchmarking Commercial Intent Detection Services with Practice-Driven Evaluations*, NAACL Industry 2021 — TF-IDF difficult subsets (adopted here); full-set vs difficult results.
- Hong et al. 2024, *Exploring the Use of Natural Language Descriptions of Intents for LLMs…*, SIGDIAL (ACL 2024.sigdial-1.39) — description-quality methodology (independent < dependent < cleansed).
- *When Do LLMs Replace Fine-Tuned NLU?* arXiv:2608.20371 — Claude Haiku 88.5 vs RoBERTa 89.1 (tie) on CLINC150; OOS recall 85.6 vs 58.1; schema-swap protocol.
- Arora et al. 2024, *Intent Detection in the Age of LLMs*, arXiv:2410.01627 — adaptive-ICL recipe; label-space size/scope effects on OOS.
- Li, Larson & Leach 2024, *Generating Hard-Negative Out-of-Scope Data with ChatGPT*, LREC-COLING — released CLINC150 hard-negative oos set.
- Zhang et al. 2022 — CLINC in-domain-OOS splits (CLINC-Single-Domain-OOS).
- Arora et al. 2020, *HINT3* — production difficulty filtering via ELMo + entailment.
- Rodríguez et al. 2021, *Evaluation Examples are not Equally Informative* — IRT item difficulty (Metric C anchor).
- Martínez-Plumed et al. 2022, *When AI Difficulty is Easy*, AAAI — difficulty-metric pitfalls; system characteristic curves.
- Swayamdipta et al. 2020 (*Dataset Cartography*); Cosma et al. 2024 (EMNLP) — training-dynamics difficulty bands with monotonic degradation.
- Kiela et al. 2021, *Dynabench*; Nie et al. 2019, *ANLI*; McCoy et al. 2019, *HANS* — model-in-the-loop and heuristic-challenge difficulty families.
- TypeSafe Jev materials — docs quickstart (support-ticket demo), helpdesk-ticket routing walkthrough, choice-over-`{billing, technical, sales, needs_review}` example, "give can't-classify a home" guidance, published jaggedness (literal reading of negations — the failure mode this benchmark's difficulty construction targets).

---

## 16. Appendix — banking domain facts (for build-time assertions)

- Banking intents (15): `freeze_account, routing, pin_change, bill_due, pay_bill, account_blocked, interest_rate, min_payment, bill_balance, transfer, order_checks, balance, spending_history, transactions, report_fraud`
- IBM difficult subset: 750 rows total; banking = 75 rows (5 per intent, verified 2026-10-09 against the released TSV).
- Eval set: 150 rows = 75 difficult + 75 non-difficult. No oos rows. 16 prompt options.
- CLINC150 splits: train 15,000 / val 3,000 / test 4,500 in-scope + 100/100/1,000 oos. **Only test is used.**
- Estimated LLM spend: 1,200 calls ≈ $3–6 (cap $10).

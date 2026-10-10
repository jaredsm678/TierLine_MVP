# TierLine MVP

A difficulty-stratified evaluation of **System-One decision models vs. frontier
LLMs** on a fixed routing task built from the **CLINC150** dataset's banking
domain: 150 test utterances (75 "difficult" from a published TF-IDF-difficult
subset, 75 ordinary), 16 routing options (15 banking intents + `out_of_scope`),
two frozen prompts, ten contestants, final-answer scoring, and a small set of
pre-registered analyses.

**[MVP_DESIGN_DOC.md](MVP_DESIGN_DOC.md) is the single source of truth** — all
decisions there are locked. This README only covers how to run the code.

## Layout

```
MVP_DESIGN_DOC.md            the spec (§4.3 dataset recipe, §5 definitions, ...)
src/tierline_mvp/schema.py   frozen constants: 16 options, definitions, row schema
src/tierline_mvp/prompts.py  V1/V2 prompt renderers (frozen §6 text)
src/tierline_mvp/contestants.py  adapter contract + OpenRouter LLM adapters (§7)
src/tierline_mvp/scoring.py  §8 normalization/validity (aggregation: session 6)
src/tierline_mvp/runner.py   cost guard now; full CLI + resume (session 5)
scripts/fetch_sources.py     downloads frozen source artifacts into data/sources/
scripts/build_dataset.py     builds data/eval_v0.jsonl + data/MANIFEST.json (§4.3)
scripts/smoke.py             per-model 5-row smoke with §8 scoring + cost cap
data/eval_v0.jsonl           the frozen eval set (150 rows)
data/MANIFEST.json           SHA-256s, counts, seed, construction notes
tests/                       schema, dataset, prompts, adapters, scoring tests
results/                     run outputs (session 5)
```

## Setup (design doc §10)

```bash
python3.12 -m venv .venv
.venv/bin/pip install httpx pytest
```

## Build the eval set

```bash
.venv/bin/python scripts/fetch_sources.py    # downloads + SHA-256-verifies sources
.venv/bin/python scripts/build_dataset.py    # asserts §4.3/§16 invariants, emits data/
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

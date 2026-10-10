#!/usr/bin/env python3
"""Build the frozen MVP eval set per MVP_DESIGN_DOC.md §4.3 (session 1).

Usage:
    python scripts/build_dataset.py

Reads  data/sources/data_full.json and data/sources/clinc150-tfidf-testset.tsv
Writes data/eval_v0.jsonl (150 rows) and data/MANIFEST.json

Construction procedure (§4.3, implemented exactly):
 1. Load data_full.json, take the test split; assert its SHA-256 (§4.1 pin).
 2. Restrict to the 15 banking intents.
 3. Load the IBM TSV; keep banking rows (expect 75 = 5 per intent).
 4. Assert every difficult utterance exists verbatim in the CLINC test split.
 5. For each intent: sample 5 non-difficult rows from the remaining test rows
    with random.Random(42), candidates in original file order.
 6. Light dedup on normalized text: keep the first, replace the dropped row
    with the next candidate from the same intent's pool (same RNG continuing).
 7. Emit data/eval_v0.jsonl + data/MANIFEST.json.

Disclosed implementation choices (also recorded in the MANIFEST):
- One shared random.Random(42) is consumed across intents, which are
  processed in the frozen alphabetical option order of schema.OPTIONS.
- Dedup is global across the eval set: a sampled row is rejected if its
  normalized text matches any difficult row or any previously accepted row.
  Keep-first is applied in draw order; replacements are drawn one at a time
  from the same intent's remaining pool with the same RNG.
- All test rows whose text equals a difficult utterance are excluded from the
  non-difficult pool (guards against verbatim duplicates of difficult items
  leaking into the ordinary bucket); any exclusion beyond the 5 canonical
  rows is logged.
"""

from __future__ import annotations

import hashlib
import json
import random
import string
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tierline_mvp import schema  # noqa: E402  (path shim must run first)

DATA_DIR = ROOT / "data"
SOURCES_DIR = DATA_DIR / "sources"
EVAL_PATH = DATA_DIR / "eval_v0.jsonl"
MANIFEST_PATH = DATA_DIR / "MANIFEST.json"

CONSTRUCTION_CODE_VERSION = "session-1/1"

# §4.3.6 dedup key: lowercase, strip punctuation/whitespace.
_STRIP_TABLE = {ord(c): None for c in string.punctuation + string.whitespace}
_STRIP_TABLE.update({ord(c): None for c in "‘’“”–—…"})


def normalize_text(text: str) -> str:
    return text.lower().translate(_STRIP_TABLE)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    raise SystemExit(f"BUILD FAILURE: {msg}")


# ---------------------------------------------------------------------------
# Step 1-2: CLINC150 data_full.json -> banking rows of the test split
# ---------------------------------------------------------------------------

def load_clinc(path: Path) -> dict:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    pinned = schema.SOURCE_ARTIFACTS["data_full.json"]["sha256"]
    if digest != pinned:
        fail(f"data_full.json SHA-256 mismatch:\n  got      {digest}\n  expected {pinned}")

    data = json.loads(raw.decode("utf-8"))
    expected_sizes = {
        "train": schema.EXPECTED_TRAIN_ROWS,
        "val": schema.EXPECTED_VAL_ROWS,
        "test": schema.EXPECTED_TEST_ROWS,
        **{f"oos_{k}": v for k, v in schema.EXPECTED_OOS_ROWS.items()},
    }
    for key, n in expected_sizes.items():
        actual = len(data.get(key, []))
        if actual != n:
            fail(f"data_full.json split {key!r} has {actual} rows, expected {n} (§16)")

    for key in ("test",):
        for i, item in enumerate(data[key]):
            if not (isinstance(item, list) and len(item) == 2
                    and isinstance(item[0], str) and isinstance(item[1], str)):
                fail(f"data_full.json {key}[{i}] is not a [text, intent] string pair")
    return data


def banking_test_rows(test_split: list) -> list[tuple[int, str, str]]:
    """[(original index, text, intent)] for the 15 banking intents, file order."""
    rows = [
        (i, text, intent)
        for i, (text, intent) in enumerate(test_split)
        if intent in schema.BANKING_INTENTS
    ]
    if len(rows) != 15 * schema.EXPECTED_TEST_ROWS_PER_INTENT:
        fail(f"banking test rows: got {len(rows)}, expected 450")
    per_intent = Counter(intent for _, _, intent in rows)
    for intent in schema.BANKING_INTENTS:
        if per_intent[intent] != schema.EXPECTED_TEST_ROWS_PER_INTENT:
            fail(f"intent {intent!r}: {per_intent[intent]} test rows, expected 30")
    return rows


# ---------------------------------------------------------------------------
# Step 3-4: IBM difficult TSV
# ---------------------------------------------------------------------------

def load_difficult_tsv(path: Path) -> list[tuple[str, str]]:
    """All (utterance, intent) rows of the IBM TSV, file order.

    Pure parser: the §16 row-count assertion (750) lives in main() so this
    stays unit-testable on small fixtures.
    """
    rows: list[tuple[str, str]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split("\t", 1)
        if len(parts) != 2:
            fail(f"TSV line {lineno}: expected 2 tab-separated fields")
        utterance, intent = parts[0], parts[1].strip()
        if lineno == 1 and utterance == "utterance" and intent == "intent":
            continue  # header
        rows.append((utterance, intent))
    return rows


def banking_difficult_of(tsv_rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Banking rows of the TSV, file order (§4.3.3: expect 75 = 5 per intent)."""
    rows = [(u, i) for u, i in tsv_rows if i in schema.BANKING_INTENTS]
    if len(rows) != 75:
        fail(f"TSV banking rows: got {len(rows)}, expected 75")
    per_intent = Counter(intent for _, intent in rows)
    for intent in schema.BANKING_INTENTS:
        if per_intent[intent] != schema.EXPECTED_DIFFICULT_PER_INTENT:
            fail(f"TSV intent {intent!r}: {per_intent[intent]} difficult rows, expected 5")
    return rows


def verify_difficult_in_test(
    difficult: list[tuple[str, str]], test_split: list
) -> dict:
    """§4.3.4: every difficult utterance must exist verbatim in the test split.

    Also cross-checks the TSV's intent label against CLINC's own label for the
    same text (gold = CLINC labels, §2 row 13). Returns a small report dict.
    """
    by_text: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for i, (text, intent) in enumerate(test_split):
        by_text[text].append((i, intent))

    misses, mismatches = [], []
    multi_occurrence = {}
    for utterance, intent in difficult:
        hits = by_text.get(utterance)
        if not hits:
            misses.append((utterance, intent))
            continue
        if len(hits) > 1:
            multi_occurrence[utterance] = len(hits)
        if intent not in {lab for _, lab in hits}:
            mismatches.append((utterance, intent, sorted({lab for _, lab in hits})))
    if misses:
        details = "\n".join(f"  - intent={i!r} text={u!r}" for u, i in misses[:10])
        fail(f"{len(misses)} difficult utterance(s) not found verbatim in the "
             f"CLINC test split:\n{details}")
    if mismatches:
        details = "\n".join(f"  - TSV={i!r} CLINC={labs!r} text={u!r}" for u, i, labs in mismatches[:10])
        fail(f"{len(mismatches)} TSV/CLINC intent mismatch(es) on difficult rows:\n{details}")
    return {"multi_occurrence_difficult_texts": multi_occurrence}


# ---------------------------------------------------------------------------
# Steps 5-6: non-difficult sampling + dedup
# ---------------------------------------------------------------------------

def sample_nondifficult(
    banking_rows: list[tuple[int, str, str]],
    difficult_texts: list[str],
    seed: int = schema.SEED,
    per_intent: int = schema.EXPECTED_NON_DIFFICULT_PER_INTENT,
    intents: tuple[str, ...] = schema.BANKING_INTENTS,
) -> tuple[dict[str, list[tuple[int, str]]], dict]:
    """§4.3.5-6. Returns {intent: [(original index, text)]} plus stats.

    `intents` is the processing order for the shared RNG; the real build uses
    the frozen alphabetical option order (schema.BANKING_INTENTS). Exposed as
    a parameter only so small synthetic fixtures can be unit-tested.
    """
    difficult_text_set = set(difficult_texts)
    rng = random.Random(seed)

    # Global dedup state: difficult rows are kept verbatim (locked decision #4),
    # so their normalized texts pre-occupy the used-set.
    used: set[str] = {normalize_text(t) for t in difficult_texts}

    picked: dict[str, list[tuple[int, str]]] = {}
    replacements: dict[str, int] = {}
    extra_excluded: dict[str, int] = {}

    for intent in intents:  # frozen alphabetical order in the real build
        pool = [
            (i, text) for i, text, lab in banking_rows
            if lab == intent and text not in difficult_text_set
        ]
        expected_pool = schema.EXPECTED_TEST_ROWS_PER_INTENT - schema.EXPECTED_DIFFICULT_PER_INTENT
        extra_excluded[intent] = len(pool) - expected_pool  # >0 only if test-split dups of difficult texts exist
        if len(pool) < per_intent:
            fail(f"intent {intent!r}: non-difficult pool has {len(pool)} rows, need {per_intent}")

        chosen: list[tuple[int, str]] = []
        chosen_idx: set[int] = set()

        def accept(cand: tuple[int, str]) -> bool:
            norm = normalize_text(cand[1])
            if norm in used:
                return False
            chosen.append(cand)
            chosen_idx.add(cand[0])
            used.add(norm)
            return True

        # First draw (§4.3.5): pool is already in original file order.
        for cand in rng.sample(pool, per_intent):
            accept(cand)

        # Replacements (§4.3.6): same RNG, same intent's remaining pool.
        n_repl = 0
        while len(chosen) < per_intent:
            remaining = [c for c in pool if c[0] not in chosen_idx]
            if not remaining:
                fail(f"intent {intent!r}: non-difficult pool exhausted by dedup")
            if not accept(rng.sample(remaining, 1)[0]):
                n_repl += 1
            if n_repl > len(pool):
                fail(f"intent {intent!r}: dedup could not converge")
        replacements[intent] = n_repl
        picked[intent] = sorted(chosen, key=lambda c: c[0])  # file order for emission

    stats = {
        "replacements_per_intent": replacements,
        "extra_pool_exclusions_per_intent": extra_excluded,
        "total_replacements": sum(replacements.values()),
    }
    return picked, stats


# ---------------------------------------------------------------------------
# Step 7: rows, emission, MANIFEST
# ---------------------------------------------------------------------------

def make_row(n: int, intent: str, difficulty: str, text: str) -> dict:
    return schema.validate_row({
        "row_id": f"bnk-{n:04d}",
        "split_source": schema.SPLIT_SOURCE,
        "intent": intent,
        "difficulty": difficulty,
        "source": schema.SOURCE_FOR_DIFFICULTY[difficulty],
        "text": text,
    })


def construct_eval_set(
    banking_rows: list[tuple[int, str, str]],
    difficult: list[tuple[str, str]],
    seed: int = schema.SEED,
) -> tuple[list[dict], dict]:
    """Difficult rows first (intent-alphabetical, then TSV order), then
    non-difficult rows (intent-alphabetical, then original file order).
    Returns (rows, sampling_stats)."""
    difficult_by_intent: dict[str, list[str]] = defaultdict(list)
    for utterance, intent in difficult:
        difficult_by_intent[intent].append(utterance)

    nondifficult, stats = sample_nondifficult(
        banking_rows, [u for u, _ in difficult], seed=seed,
        intents=schema.BANKING_INTENTS,
    )

    rows: list[dict] = []
    n = 0
    for intent in schema.BANKING_INTENTS:
        for utterance in difficult_by_intent[intent]:
            n += 1
            rows.append(make_row(n, intent, schema.DIFFICULTY_DIFFICULT, utterance))
    for intent in schema.BANKING_INTENTS:
        for _, text in nondifficult[intent]:
            n += 1
            rows.append(make_row(n, intent, schema.DIFFICULTY_NON_DIFFICULT, text))

    if len(rows) != schema.EXPECTED_EVAL_ROWS:
        fail(f"eval set has {len(rows)} rows, expected {schema.EXPECTED_EVAL_ROWS}")
    per_intent = Counter(r["intent"] for r in rows)
    for intent in schema.BANKING_INTENTS:
        if per_intent[intent] != 10:
            fail(f"intent {intent!r}: {per_intent[intent]} eval rows, expected 10")
    norms = [normalize_text(r["text"]) for r in rows]
    if len(set(norms)) != len(norms):
        fail("normalized duplicate texts remain in the eval set")
    return rows, stats


def build_manifest(
    rows: list[dict], stats: dict, clinc_report: dict, domains_note: str
) -> dict:
    sources = {}
    for name, meta in schema.SOURCE_ARTIFACTS.items():
        path = SOURCES_DIR / name
        entry = {"role": meta["role"], "url": meta["urls"][0], "pinned_sha256": meta["sha256"]}
        if path.exists():
            digest = sha256_file(path)
            entry.update({
                "sha256": digest,
                "bytes": path.stat().st_size,
                "matches_pin": None if meta["sha256"] is None else digest == meta["sha256"],
            })
        else:
            entry["sha256"] = None
        sources[name] = entry

    row_counts = {
        intent: {
            "difficult": sum(1 for r in rows if r["intent"] == intent and r["difficulty"] == "difficult"),
            "non_difficult": sum(1 for r in rows if r["intent"] == intent and r["difficulty"] == "non_difficult"),
        }
        for intent in schema.BANKING_INTENTS
    }

    build_script = ROOT / "scripts" / "build_dataset.py"
    return {
        "design_doc": "MVP_DESIGN_DOC.md v1.0 (2026-10-09)",
        "eval_file": "data/eval_v0.jsonl",
        "eval_sha256": sha256_file(EVAL_PATH),
        "eval_rows": len(rows),
        "totals": {
            "difficult": sum(1 for r in rows if r["difficulty"] == "difficult"),
            "non_difficult": sum(1 for r in rows if r["difficulty"] == "non_difficult"),
        },
        "row_counts_per_intent": row_counts,
        "sources": sources,
        "seed": schema.SEED,
        "construction_code_version": CONSTRUCTION_CODE_VERSION,
        "build_dataset_sha256": sha256_file(build_script),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "sampling_stats": stats,
        "clinc_test_report": clinc_report,
        "notes": [
            "Row order: all difficult rows first (intents in frozen alphabetical "
            "option order; TSV file order within intent), then non-difficult rows "
            "(intents alphabetical; original CLINC test-file order within intent). "
            "row_id reflects this order.",
            "Non-difficult sampling consumes one shared random.Random(42) across "
            "intents, processed in frozen alphabetical option order; candidates "
            "are in original file order (§4.3.5).",
            "Dedup (§4.3.6) is global: a sampled row is rejected if its normalized "
            "text (lowercase, punctuation/whitespace stripped, plus common unicode "
            "quotes/dashes) matches a difficult row or any previously accepted "
            "row; keep-first in draw order; replacements drawn one at a time from "
            "the same intent's remaining pool with the same RNG.",
            "All test rows whose text equals a difficult utterance are excluded "
            "from the non-difficult pool; counts beyond the canonical 5 per "
            "intent are in sampling_stats.extra_pool_exclusions_per_intent.",
            "Definitions are stored as single logical lines (the doc hard-wraps "
            "§5 for display) and markdown backticks around intent names are "
            "dropped; see tierline_mvp/schema.py.",
            domains_note,
        ],
    }


def main() -> None:
    data_full = SOURCES_DIR / "data_full.json"
    tsv = SOURCES_DIR / "clinc150-tfidf-testset.tsv"
    if not data_full.exists() or not tsv.exists():
        raise SystemExit("Missing source artifacts. Run: python scripts/fetch_sources.py")

    data = load_clinc(data_full)
    test_split = data["test"]

    tsv_rows = load_difficult_tsv(tsv)
    if len(tsv_rows) != schema.EXPECTED_TSV_ROWS:
        fail(f"TSV has {len(tsv_rows)} data rows, expected {schema.EXPECTED_TSV_ROWS}")
    difficult = banking_difficult_of(tsv_rows)
    clinc_report = verify_difficult_in_test(difficult, test_split)
    if clinc_report["multi_occurrence_difficult_texts"]:
        print(f"note: {len(clinc_report['multi_occurrence_difficult_texts'])} difficult "
              f"text(s) occur multiple times in the test split; all occurrences are "
              f"excluded from the non-difficult pool")

    banking_rows = banking_test_rows(test_split)
    rows, stats = construct_eval_set(banking_rows, difficult)

    domains_path = SOURCES_DIR / "domains.json"
    if domains_path.exists():
        domains = json.loads(domains_path.read_text(encoding="utf-8"))
        # domains.json maps {domain: [intents]}; cross-check only — the frozen
        # list in schema.py is authoritative.
        banking_from_domains = sorted(domains.get("banking", []))
        if banking_from_domains != sorted(schema.BANKING_INTENTS):
            print(f"note: domains.json banking set differs from the frozen list: "
                  f"{banking_from_domains}")
        domains_note = ("domains.json present; banking intent cross-check "
                        + ("passed" if banking_from_domains == sorted(schema.BANKING_INTENTS)
                           else "FAILED (see build log)"))
    else:
        domains_note = "domains.json unavailable at build time; banking intent list taken from the frozen §2/§16 list in tierline_mvp/schema.py"

    EVAL_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    manifest = build_manifest(rows, stats, clinc_report, domains_note)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {EVAL_PATH.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"eval sha256: {manifest['eval_sha256']}")
    print(f"replacements (dedup swaps): {stats['total_replacements']} "
          f"{stats['replacements_per_intent']}")
    extra = {k: v for k, v in stats["extra_pool_exclusions_per_intent"].items() if v}
    if extra:
        print(f"extra pool exclusions (test-split duplicates of difficult texts): {extra}")


if __name__ == "__main__":
    main()

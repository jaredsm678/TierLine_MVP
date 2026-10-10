"""Tests for scripts/build_dataset.py: normalization, verification, sampling."""

import pytest

from tierline_mvp import schema
import build_dataset as bd


# --- normalize_text (§4.3.6 dedup key) -------------------------------------

def test_normalize_text_strips_punct_whitespace_and_lowercases():
    assert bd.normalize_text("  Hello, WORLD!! ") == "helloworld"
    assert bd.normalize_text("I can't pay — my bill?") == "icantpaymybill"


# --- difficult TSV parsing --------------------------------------------------

def test_load_difficult_tsv_parses_and_skips_header(tmp_path):
    tsv = tmp_path / "t.tsv"
    tsv.write_text(
        "utterance\tintent\n"
        "how do I pay\ttransfer\n"
        "check my balance\tbalance\n"
        "what is my routing number\trouting\n",
        encoding="utf-8",
    )
    rows = bd.load_difficult_tsv(tsv)
    assert rows == [("how do I pay", "transfer"),
                    ("check my balance", "balance"),
                    ("what is my routing number", "routing")]


def test_load_difficult_tsv_fails_on_malformed_line(tmp_path):
    tsv = tmp_path / "t.tsv"
    tsv.write_text("utterance\tintent\nno tab here\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="2 tab-separated fields"):
        bd.load_difficult_tsv(tsv)


def test_banking_difficult_of_filters_and_counts():
    tsv_rows = [(f"u-{i}-{k}", intent)
                for i, intent in enumerate(schema.BANKING_INTENTS)
                for k in range(5)]
    tsv_rows.append(("what is the weather", "translate"))  # non-banking, dropped
    difficult = bd.banking_difficult_of(tsv_rows)
    assert len(difficult) == 75
    counts = {}
    for _, intent in difficult:
        counts[intent] = counts.get(intent, 0) + 1
    assert set(counts.values()) == {5}


def test_banking_difficult_of_fails_on_wrong_count():
    tsv_rows = [(f"u-{i}", intent)
                for i, intent in enumerate(schema.BANKING_INTENTS)]  # 1 per intent
    with pytest.raises(SystemExit, match="expected 75"):
        bd.banking_difficult_of(tsv_rows)


# --- §4.3.4 verbatim membership + label cross-check -------------------------

def test_verify_difficult_passes_on_consistent_data():
    test_rows = [["pay my bill", "pay_bill"], ["pay my bill", "pay_bill"], ["balance?", "balance"]]
    report = bd.verify_difficult_in_test([("pay my bill", "pay_bill")], test_rows)
    assert report["multi_occurrence_difficult_texts"] == {"pay my bill": 2}


def test_verify_difficult_fails_on_missing_text():
    with pytest.raises(SystemExit, match="not found verbatim"):
        bd.verify_difficult_in_test([("nope", "pay_bill")], [["pay my bill", "pay_bill"]])


def test_verify_difficult_fails_on_label_mismatch():
    with pytest.raises(SystemExit, match="mismatch"):
        bd.verify_difficult_in_test([("pay my bill", "transfer")], [["pay my bill", "pay_bill"]])


# --- §4.3.5-6 sampling + dedup (synthetic data, no network) ------------------

def make_banking_rows():
    """Synthetic banking slice: 2 intents x 8 test rows (file-order indices).

    Deliberate traps: "PAY UP!!" (balance) vs "pay up" (routing) normalize
    equal; "D1"/"D2" are the difficult rows and must never be resampled.
    """
    rows = []
    idx = 0
    for intent, texts in (
        ("balance", ["D1", "how much money", "PAY UP!!", "whats my balance",
                     "balance please", "dupe text", "show funds", "available?"]),
        ("routing", ["D2", "routing number", "pay up", "aba number",
                     "get my routing", "bank code", "routing help", "wire num"]),
    ):
        for t in texts:
            rows.append((idx, t, intent))
            idx += 1
    return rows


SYNTH_INTENTS = ("balance", "routing")


def test_sample_nondifficult_is_deterministic():
    a, _ = bd.sample_nondifficult(make_banking_rows(), ["D1", "D2"], seed=42,
                                  per_intent=2, intents=SYNTH_INTENTS)
    b, _ = bd.sample_nondifficult(make_banking_rows(), ["D1", "D2"], seed=42,
                                  per_intent=2, intents=SYNTH_INTENTS)
    assert a == b


def test_sample_nondifficult_shape_and_difficult_exclusion():
    picked, stats = bd.sample_nondifficult(make_banking_rows(), ["D1", "D2"],
                                           seed=42, per_intent=2,
                                           intents=SYNTH_INTENTS)
    assert set(picked) == {"balance", "routing"}
    for intent, chosen in picked.items():
        assert len(chosen) == 2
        for _, text in chosen:
            assert text not in ("D1", "D2")
    assert set(stats["replacements_per_intent"]) == {"balance", "routing"}


def test_no_normalized_duplicates_within_or_across_intents():
    picked, _ = bd.sample_nondifficult(make_banking_rows(), ["D1", "D2"],
                                       seed=42, per_intent=5,
                                       intents=SYNTH_INTENTS)
    norms = [bd.normalize_text(t) for chosen in picked.values() for _, t in chosen]
    assert len(norms) == len(set(norms)), "dedup failed: duplicate normalized text kept"
    assert all(t not in ("D1", "D2") for chosen in picked.values() for _, t in chosen)


def test_difficult_rows_are_never_resampled_other_seed():
    picked, _ = bd.sample_nondifficult(make_banking_rows(), ["D1", "D2"],
                                       seed=123, per_intent=5,
                                       intents=SYNTH_INTENTS)
    for chosen in picked.values():
        assert all(t not in ("D1", "D2") for _, t in chosen)


# --- construct_eval_set: frozen 5+5 shape on synthetic data ------------------

def make_full_synthetic_slice():
    """15 intents x (5 difficult rows + 10-candidate pool), file-order indices."""
    difficult = []
    banking_rows = []
    idx = 0
    for n, intent in enumerate(schema.BANKING_INTENTS):
        for k in range(5):
            difficult.append((f"D{n}-{k}", intent))
            banking_rows.append((idx, f"D{n}-{k}", intent))
            idx += 1
        for j in range(10):
            banking_rows.append((idx, f"{intent} candidate {j}", intent))
            idx += 1
    return banking_rows, difficult


def test_construct_eval_set_full_shape_and_row_schema():
    banking_rows, difficult = make_full_synthetic_slice()
    rows, stats = bd.construct_eval_set(banking_rows, difficult, seed=42)
    assert len(rows) == 150
    assert [r["row_id"] for r in rows] == [f"bnk-{i:04d}" for i in range(1, 151)]
    for r in rows:
        schema.validate_row(r)
    assert all(r["difficulty"] == "difficult" for r in rows[:75])
    assert all(r["difficulty"] == "non_difficult" for r in rows[75:])
    assert all(r["source"] == "ibm_tfidf_testset" for r in rows[:75])
    assert all(r["source"] == "clinc_test_random" for r in rows[75:])
    assert stats["total_replacements"] == 0
    # intents appear in frozen alphabetical order within each block
    block_intents = [r["intent"] for r in rows[:75]]
    assert block_intents == [i for i in schema.BANKING_INTENTS for _ in range(5)]


def test_construct_eval_set_fails_on_undersized_pool():
    difficult = []
    banking_rows = []
    idx = 0
    for n, intent in enumerate(schema.BANKING_INTENTS):
        for k in range(5):
            difficult.append((f"D{n}-{k}", intent))
            banking_rows.append((idx, f"D{n}-{k}", intent))
            idx += 1
        for j in range(2):  # pool of 2, needs 5
            banking_rows.append((idx, f"{intent} candidate {j}", intent))
            idx += 1
    with pytest.raises(SystemExit, match="pool"):
        bd.construct_eval_set(banking_rows, difficult, seed=42)
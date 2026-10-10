"""Tests for the frozen constants and row validation in tierline_mvp.schema."""

import pytest

from tierline_mvp import schema

# Transcribed from design doc §5 — the options block, byte for byte.
EXPECTED_OPTIONS = (
    "account_blocked",
    "balance",
    "bill_balance",
    "bill_due",
    "freeze_account",
    "interest_rate",
    "min_payment",
    "order_checks",
    "out_of_scope",
    "pay_bill",
    "pin_change",
    "report_fraud",
    "routing",
    "spending_history",
    "transactions",
    "transfer",
)


def test_options_exact_frozen_order():
    assert schema.OPTIONS == EXPECTED_OPTIONS
    assert list(schema.OPTIONS) == sorted(schema.OPTIONS)


def test_banking_intents_are_options_minus_oos():
    assert len(schema.BANKING_INTENTS) == 15
    assert set(schema.BANKING_INTENTS) == set(EXPECTED_OPTIONS) - {"out_of_scope"}
    assert "out_of_scope" not in schema.BANKING_INTENTS


def test_definitions_complete_and_clean():
    assert set(schema.DEFINITIONS) == set(EXPECTED_OPTIONS)
    for name, text in schema.DEFINITIONS.items():
        assert text.strip(), f"empty definition for {name}"
        assert "`" not in text, f"markdown backtick leaked into {name}"
        assert "\n" not in text, f"{name} must be a single logical line"


def make_valid_row(**overrides):
    row = {
        "row_id": "bnk-0001",
        "split_source": "test",
        "intent": "pay_bill",
        "difficulty": "difficult",
        "source": "ibm_tfidf_testset",
        "text": "i need to pay my bill",
    }
    row.update(overrides)
    return row


def test_validate_row_accepts_valid_row():
    assert schema.validate_row(make_valid_row()) == make_valid_row()


def test_validate_row_roundtrip_both_difficulties():
    assert schema.validate_row(make_valid_row(
        row_id="bnk-0075", intent="balance",
        difficulty="non_difficult", source="clinc_test_random",
    ))


@pytest.mark.parametrize("overrides", [
    {"intent": "transfer_money"},                     # not an option
    {"intent": "out_of_scope"},                       # never a gold intent
    {"difficulty": "hard"},
    {"source": "clinc_test"},
    {"row_id": "bank-0001"},
    {"row_id": "bnk-1"},
    {"text": "   "},
    {"split_source": "train"},
    # source/difficulty 1:1 pairing violated
    {"difficulty": "non_difficult"},
])
def test_validate_row_rejects(overrides):
    with pytest.raises(ValueError):
        schema.validate_row(make_valid_row(**overrides))


def test_definitions_mention_neighbor_options():
    # spot-check that cross-references survive transcription un-backticked
    assert "freeze_account" in schema.DEFINITIONS["account_blocked"]
    assert "account_blocked" in schema.DEFINITIONS["freeze_account"]
    assert "bill_balance" in schema.DEFINITIONS["balance"]


def test_validate_row_rejects_missing_and_extra_fields():
    row = make_valid_row()
    del row["text"]
    with pytest.raises(ValueError, match="missing fields"):
        schema.validate_row(row)
    with pytest.raises(ValueError, match="unexpected fields"):
        schema.validate_row(make_valid_row(label="extra"))


def test_section16_constants():
    assert schema.SEED == 42
    assert schema.EXPECTED_TEST_ROWS == 4500
    assert schema.EXPECTED_TSV_ROWS == 750
    assert schema.EXPECTED_EVAL_ROWS == 150
    assert schema.EXPECTED_DIFFICULT_PER_INTENT == 5
    assert schema.EXPECTED_NON_DIFFICULT_PER_INTENT == 5
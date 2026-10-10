"""Tests for §8 per-row scoring: normalization, validity, outcome flags."""

import pytest

from tierline_mvp import scoring, schema


@pytest.mark.parametrize("raw,expected", [
    ("pay_bill", "pay_bill"),
    ("  Pay_Bill  ", "pay_bill"),              # trim + lowercase
    ("'pay_bill'", "pay_bill"),                # surrounding quotes
    ('"pay_bill"', "pay_bill"),
    ("“pay_bill”", "pay_bill"),                # curly quotes
    ("''pay_bill''", "pay_bill"),              # nested straight quotes
    ("“‘pay_bill’”", "pay_bill"),              # nested curly quotes
    ("pay_bill.", "pay_bill"),                 # trailing period
    ("pay_bill!!!", "pay_bill"),               # trailing punctuation run
    ("OUT_OF_SCOPE;", "out_of_scope"),         # case + trailing punctuation
])
def test_normalize_answer(raw, expected):
    assert scoring.normalize_answer(raw) == expected


@pytest.mark.parametrize("raw", [
    "`pay_bill`",                    # backticks are not quotes (literal §8)
    "the handler is pay_bill",       # extra words -> not an exact option
    "pay bill",                      # underscore missing -> not an option
    "' pay_bill '",                  # spaces inside quotes survive -> invalid
    "pay_bill's",                    # trailing punctuation strip doesn't create a match
    "",                              # empty
    "   ",
])
def test_normalize_or_validate_rejects_noncompliant(raw):
    assert not scoring.is_valid(scoring.normalize_answer(raw))


def test_quote_stripping_requires_matching_pairs():
    # asymmetric quote: nothing to strip as a pair
    assert scoring.normalize_answer("\"pay_bill'") == "\"pay_bill'"


def test_literal_single_pass_order_quotes_before_punctuation():
    # §8 order is quotes THEN trailing punctuation, single pass: the period
    # is outermost when the quote step runs, so the quotes survive and the
    # answer is invalid. Deliberate literal reading of the locked spec.
    assert scoring.normalize_answer("'pay_bill'.") == "'pay_bill'"
    assert not scoring.is_valid(scoring.normalize_answer("'pay_bill'."))


def test_score_correct():
    assert scoring.score("  Pay_Bill. ", "pay_bill") == {
        "valid": True, "correct": True, "oos_selection": False,
    }


def test_score_wrong_but_valid():
    verdict = scoring.score("balance", "pay_bill")
    assert verdict == {"valid": True, "correct": False, "oos_selection": False}


def test_score_invalid_is_wrong():
    verdict = scoring.score("the answer is pay_bill", "pay_bill")
    assert verdict == {"valid": False, "correct": False, "oos_selection": False}


def test_score_oos_selection_flag():
    # §8 rule 5: oos on an in-scope row is wrong AND counted as oos-selection
    verdict = scoring.score("out_of_scope", "pay_bill")
    assert verdict == {"valid": True, "correct": False, "oos_selection": True}


def test_score_none_label():
    assert scoring.score(None, "pay_bill") == {
        "valid": False, "correct": False, "oos_selection": False,
    }


def test_every_option_is_scorable():
    for option in schema.BANKING_INTENTS:
        verdict = scoring.score(option, option)
        assert verdict == {"valid": True, "correct": True, "oos_selection": False}


def test_oos_option_scores_as_valid_false_abstention():
    verdict = scoring.score("out_of_scope", "pay_bill")
    assert verdict == {"valid": True, "correct": False, "oos_selection": True}
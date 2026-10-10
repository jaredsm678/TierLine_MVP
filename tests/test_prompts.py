"""Byte-for-byte tests for the frozen prompt renderers (design doc §6).

The expected literals below are transcribed from §6.1/§6.2 with the §5
definitions block as one line per option (the doc hard-wraps for display;
see the transcription notes in schema.py). If a prompt ever changes, these
tests must change with it — deliberately loud.
"""

import pytest

from tierline_mvp import prompts, schema

BLOCK = "\n".join([
    "- account_blocked: The customer reports that their account is blocked, locked, or frozen — or asks why. (A request to impose a block is freeze_account.)",
    "- balance: The customer asks how much money is in an account right now — a point-in-time amount. (A list of activity is transactions; a spending total is spending_history; an amount owed on a bill is bill_balance.)",
    "- bill_balance: The customer asks how much they owe on a bill — the total amount due. (The due date is bill_due; the smallest allowed payment is min_payment.)",
    "- bill_due: The customer asks when a payment is due — the date or deadline. (The amount owed is bill_balance; making the payment is pay_bill.)",
    "- freeze_account: The customer asks to block, freeze, or restrict activity on their account. (An already-blocked account being asked about is account_blocked.)",
    "- interest_rate: The customer asks about an interest rate — charged or earned — on any account, loan, or card. (An amount owed or available is not a rate question.)",
    "- min_payment: The customer asks for the minimum amount they are required to pay. (The full amount owed is bill_balance; the due date is bill_due.)",
    "- order_checks: The customer wants to order checkbooks or more checks. (Paying money out is transfer or pay_bill, not ordering checks.)",
    "- out_of_scope: The request is none of the 15 banking requests above — for example: device commands, other services or companies, general knowledge questions, or small talk.",
    "- pay_bill: The customer wants to make a payment toward a bill, or asks for help paying one. (The due date is bill_due; the amount owed is bill_balance.)",
    "- pin_change: The customer wants to change, set, or reset a PIN — including having forgotten it.",
    "- report_fraud: The customer reports unauthorized or suspicious charges or activity, or asks how to report fraud. (An ordinary lookup of their own charges is transactions.)",
    "- routing: The customer asks for a bank or account routing number.",
    "- spending_history: The customer asks for totals or summaries of spending, usually by category or time period. (Individual entries are transactions; the current amount available is balance.)",
    "- transactions: The customer asks to see individual account activity — a list or detail of charges, purchases, or deposits. (A point-in-time amount is balance; a spending summary is spending_history.)",
    "- transfer: The customer asks to move money between accounts. (Viewing past activity is transactions; paying a bill is pay_bill.)",
])

UTTERANCE = "how much is in my account"

EXPECTED_V1 = (
    "You will route one customer request to the correct handler.\n"
    "\n"
    "The 16 handlers are:\n"
    + BLOCK + "\n"
    "\n"
    "Request:\n"
    "\"\"\"\n"
    + UTTERANCE + "\n"
    "\"\"\"\n"
    "\n"
    "Reply with exactly one handler name from the list above — nothing else."
)

EXPECTED_V2 = (
    "Which of the following handlers should take this request?\n"
    "\n"
    "Handlers and their criteria:\n"
    + BLOCK + "\n"
    "\n"
    "The request:\n"
    "\"\"\"\n"
    + UTTERANCE + "\n"
    "\"\"\"\n"
    "\n"
    "Answer with a single handler name and nothing else."
)


def test_v1_byte_for_byte():
    assert prompts.render_prompt("V1", UTTERANCE) == EXPECTED_V1


def test_v2_byte_for_byte():
    assert prompts.render_prompt("V2", UTTERANCE) == EXPECTED_V2


def test_definitions_block_identical_across_versions():
    v1 = prompts.render_prompt("V1", UTTERANCE)
    v2 = prompts.render_prompt("V2", UTTERANCE)
    assert BLOCK in v1 and BLOCK in v2


def test_option_bullets_in_frozen_alphabetical_order():
    text = prompts.render_definitions_block()
    positions = [text.index(f"- {name}:") for name in schema.OPTIONS]
    assert positions == sorted(positions)
    assert text.count("- ") == 16


def test_utterance_verbatim_no_trimming():
    prompt = prompts.render_prompt("V1", "  padded utterance  ")
    assert '"""\n  padded utterance  \n"""' in prompt


def test_unknown_version_raises():
    with pytest.raises(ValueError, match="unknown prompt version"):
        prompts.render_prompt("V3", UTTERANCE)


def test_no_trailing_newline_after_final_instruction():
    # The prompt ends with the instruction line itself (no trailing blank).
    assert prompts.render_prompt("V1", UTTERANCE).endswith("nothing else.")
    assert prompts.render_prompt("V2", UTTERANCE).endswith("nothing else.")
"""Frozen schema constants and row validation for the TierLine MVP.

Everything in this module is transcribed from MVP_DESIGN_DOC.md (§2, §4.1,
§4.4, §5, §16). Per the design doc's rules of engagement, the option list and
the definitions are frozen text: do not paraphrase, reorder, or re-wrap.

Two transcription decisions (disclosed, easy to revert):
- The doc hard-wraps §5 definition lines for display; each definition is
  stored here as a single logical line (wrap points joined with one space).
- The doc's markdown backticks around intent names inside definitions are
  dropped, so prompt text reads as plain prose.

Session 1 also parks the frozen source-artifact registry (§4.1) here, so the
URLs and the pinned SHA-256 live in exactly one place.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# §5: the 16 routing options, frozen alphabetical order (identical in every
# prompt, every model, both prompt versions)
# ---------------------------------------------------------------------------

OPTIONS: tuple[str, ...] = (
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

# §2 row 6: oos is a prompt option only; it never appears as a gold intent.
ABSTENTION_OPTION = "out_of_scope"

# §2 row 2: the 15 banking intents = the option set minus out_of_scope.
BANKING_INTENTS: tuple[str, ...] = tuple(
    o for o in OPTIONS if o != ABSTENTION_OPTION
)

# ---------------------------------------------------------------------------
# §5: frozen definitions — the single source of truth for prompt V1 and V2.
# ---------------------------------------------------------------------------

DEFINITIONS: dict[str, str] = {
    "account_blocked": (
        "The customer reports that their account is blocked, locked, or frozen — "
        "or asks why. (A request to impose a block is freeze_account.)"
    ),
    "balance": (
        "The customer asks how much money is in an account right now — a "
        "point-in-time amount. (A list of activity is transactions; a spending "
        "total is spending_history; an amount owed on a bill is bill_balance.)"
    ),
    "bill_balance": (
        "The customer asks how much they owe on a bill — the total amount due. "
        "(The due date is bill_due; the smallest allowed payment is min_payment.)"
    ),
    "bill_due": (
        "The customer asks when a payment is due — the date or deadline. "
        "(The amount owed is bill_balance; making the payment is pay_bill.)"
    ),
    "freeze_account": (
        "The customer asks to block, freeze, or restrict activity on their "
        "account. (An already-blocked account being asked about is "
        "account_blocked.)"
    ),
    "interest_rate": (
        "The customer asks about an interest rate — charged or earned — on any "
        "account, loan, or card. (An amount owed or available is not a rate "
        "question.)"
    ),
    "min_payment": (
        "The customer asks for the minimum amount they are required to pay. "
        "(The full amount owed is bill_balance; the due date is bill_due.)"
    ),
    "order_checks": (
        "The customer wants to order checkbooks or more checks. (Paying money "
        "out is transfer or pay_bill, not ordering checks.)"
    ),
    "out_of_scope": (
        "The request is none of the 15 banking requests above — for example: "
        "device commands, other services or companies, general knowledge "
        "questions, or small talk."
    ),
    "pay_bill": (
        "The customer wants to make a payment toward a bill, or asks for help "
        "paying one. (The due date is bill_due; the amount owed is "
        "bill_balance.)"
    ),
    "pin_change": (
        "The customer wants to change, set, or reset a PIN — including having "
        "forgotten it."
    ),
    "report_fraud": (
        "The customer reports unauthorized or suspicious charges or activity, "
        "or asks how to report fraud. (An ordinary lookup of their own charges "
        "is transactions.)"
    ),
    "routing": (
        "The customer asks for a bank or account routing number."
    ),
    "spending_history": (
        "The customer asks for totals or summaries of spending, usually by "
        "category or time period. (Individual entries are transactions; the "
        "current amount available is balance.)"
    ),
    "transactions": (
        "The customer asks to see individual account activity — a list or "
        "detail of charges, purchases, or deposits. (A point-in-time amount is "
        "balance; a spending summary is spending_history.)"
    ),
    "transfer": (
        "The customer asks to move money between accounts. (Viewing past "
        "activity is transactions; paying a bill is pay_bill.)"
    ),
}

# ---------------------------------------------------------------------------
# §4.4: eval row schema
# ---------------------------------------------------------------------------

SPLIT_SOURCE = "test"
DIFFICULTY_DIFFICULT = "difficult"
DIFFICULTY_NON_DIFFICULT = "non_difficult"
SOURCE_IBM_TFIDF = "ibm_tfidf_testset"
SOURCE_CLINC_RANDOM = "clinc_test_random"

DIFFICULTIES = (DIFFICULTY_DIFFICULT, DIFFICULTY_NON_DIFFICULT)
SOURCES = (SOURCE_IBM_TFIDF, SOURCE_CLINC_RANDOM)

# source <-> difficulty is 1:1 by construction (§4.4)
SOURCE_FOR_DIFFICULTY = {
    DIFFICULTY_DIFFICULT: SOURCE_IBM_TFIDF,
    DIFFICULTY_NON_DIFFICULT: SOURCE_CLINC_RANDOM,
}

_ROW_FIELDS = frozenset(
    {"row_id", "split_source", "intent", "difficulty", "source", "text"}
)
_ROW_ID_RE = re.compile(r"^bnk-\d{4}$")


def validate_row(row: dict) -> dict:
    """Validate one eval row against §4.4. Returns the row or raises ValueError."""
    keys = set(row)
    missing = _ROW_FIELDS - keys
    if missing:
        raise ValueError(f"row {row.get('row_id', '?')}: missing fields {sorted(missing)}")
    extra = keys - _ROW_FIELDS
    if extra:
        raise ValueError(f"row {row.get('row_id', '?')}: unexpected fields {sorted(extra)}")

    rid = row["row_id"]
    if not isinstance(rid, str) or not _ROW_ID_RE.match(rid):
        raise ValueError(f"row: bad row_id {rid!r} (expected 'bnk-NNNN')")

    if row["split_source"] != SPLIT_SOURCE:
        raise ValueError(f"row {rid}: split_source must be {SPLIT_SOURCE!r}")
    if row["intent"] not in BANKING_INTENTS:
        raise ValueError(
            f"row {rid}: intent {row['intent']!r} is not a banking intent "
            f"(out_of_scope never appears as gold intent, §2 row 6)"
        )
    if row["difficulty"] not in DIFFICULTIES:
        raise ValueError(f"row {rid}: bad difficulty {row['difficulty']!r}")
    if row["source"] not in SOURCES:
        raise ValueError(f"row {rid}: bad source {row['source']!r}")
    if row["source"] != SOURCE_FOR_DIFFICULTY[row["difficulty"]]:
        raise ValueError(
            f"row {rid}: source {row['source']!r} does not match difficulty "
            f"{row['difficulty']!r}"
        )
    text = row["text"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"row {rid}: text must be a non-empty string")
    return row


# ---------------------------------------------------------------------------
# §4.1: frozen source artifacts (URLs + the one pinned SHA-256)
# ---------------------------------------------------------------------------

SOURCE_ARTIFACTS: dict[str, dict] = {
    "data_full.json": {
        "role": "CLINC150 utterances, intent labels, splits",
        "urls": [
            "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
            "https://raw.githubusercontent.com/clinc/oos-eval/main/data/data_full.json",
        ],
        # Pinned in §4.1; the build asserts against this value.
        "sha256": "36923c3705a59e08fe9c3883d8bc2dd966ef93e22cb78ac41171782a698d56e0",
        "required": True,
    },
    "clinc150-tfidf-testset.tsv": {
        "role": "IBM NAACL-Industry 2021 TF-IDF difficult test subset (750 rows)",
        "urls": [
            "https://raw.githubusercontent.com/haodeqi/BenchmarkingIntentDetection/master/CLINC150/clinc150-tfidf-testset.tsv",
            "https://raw.githubusercontent.com/haodeqi/BenchmarkingIntentDetection/main/CLINC150/clinc150-tfidf-testset.tsv",
        ],
        "sha256": None,  # not pinned in the doc; recorded in the MANIFEST at build time
        "required": True,
    },
    "domains.json": {
        "role": "official CLINC150 intent->domain mapping (cross-check only; "
        "the 15 banking intents are frozen in this module)",
        "urls": [
            "https://raw.githubusercontent.com/clinc/oos-eval/master/data/domains.json",
            "https://raw.githubusercontent.com/clinc/oos-eval/main/data/domains.json",
        ],
        "sha256": None,
        "required": False,  # auxiliary; build proceeds (with a logged note) if absent
    },
}

# ---------------------------------------------------------------------------
# §16: build-time assertion constants (appendix facts)
# ---------------------------------------------------------------------------

SEED = 42
EXPECTED_TRAIN_ROWS = 15_000
EXPECTED_VAL_ROWS = 3_000
EXPECTED_TEST_ROWS = 4_500
EXPECTED_OOS_ROWS = {"train": 100, "val": 100, "test": 1_000}
EXPECTED_TEST_ROWS_PER_INTENT = 30  # 4500 in-scope / 150 intents
EXPECTED_TSV_ROWS = 750
EXPECTED_DIFFICULT_PER_INTENT = 5
EXPECTED_NON_DIFFICULT_PER_INTENT = 5
EXPECTED_EVAL_ROWS = 150  # 15 intents x (5 difficult + 5 non-difficult)
"""§8 scoring rules: normalization, validity, and the per-row outcome.

Session 3 implements the per-row part of §8 (invalid-handling); aggregation
into accuracy tables and CIs is session 6 (report.py).

§8 rules, implemented literally:
1. Final answer = the model's returned label (LLM: message content, thinking
   stripped by the provider; S1: native label field). The adapter already
   trims it; trimming again here is idempotent.
2. Normalize: trim whitespace -> lowercase -> strip surrounding quotes ->
   strip trailing punctuation (. , ! ? : ;). No other tolerance: backticks,
   interior whitespace, or extra words make the answer invalid, exactly as
   the locked §8 text implies ("strip quotes/punct" names only these steps).
3. Valid iff the normalized answer is exactly one of the 16 option strings;
   otherwise invalid -> scored wrong, counted in the invalid-output rate.
4. Correct iff valid AND the normalized answer equals the row's intent.
5. An out_of_scope prediction on an in-scope row is wrong and additionally
   counted in the oos-selection rate (false abstention).
"""

from __future__ import annotations

from tierline_mvp import schema

# §8 rule 2 — exactly these six trailing punctuation characters.
TRAILING_PUNCTUATION = ".,!?:;"

# Surrounding-quote pairs: straight quotes pair with themselves; curly
# quotes have distinct opening/closing characters.
QUOTE_PAIRS = {("\"", "\""), ("'", "'"), ("“", "”"), ("‘", "’")}


def normalize_answer(label: str) -> str:
    """§8 rule 2, applied as a literal single pass in the doc's order:
    trim -> lowercase -> strip surrounding quotes -> strip trailing
    punctuation. Consequence (pinned by tests): an answer decorated as
    'pay_bill'. keeps its quotes (the period is outermost when the quote
    step runs) and is therefore invalid. If the user amends the spec to
    iterate to a fixpoint, change this function and its tests together.
    """
    s = label.strip()
    s = s.lower()
    while len(s) >= 2 and (s[0], s[-1]) in QUOTE_PAIRS:
        s = s[1:-1]
    s = s.rstrip(TRAILING_PUNCTUATION)
    return s


def is_valid(normalized: str) -> bool:
    """§8 rule 3: exactly one of the 16 option strings (oos counts as valid)."""
    return normalized in schema.OPTIONS


def score(label: str | None, intent: str) -> dict:
    """§8 rules 3-5 for one row (intent is always a banking intent here,
    since the MVP has no oos gold rows)."""
    if label is None:
        return {"valid": False, "correct": False, "oos_selection": False}
    normalized = normalize_answer(label)
    valid = is_valid(normalized)
    return {
        "valid": valid,
        "correct": valid and normalized == intent,
        # §8 rule 5: false abstention on an in-scope row.
        "oos_selection": valid and normalized == schema.ABSTENTION_OPTION,
    }
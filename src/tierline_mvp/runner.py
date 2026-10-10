"""Runner module (design doc §9).

Session 3 delivers the cost guard only; the full CLI (argparse --model
--version --workers, append-only JSONL results, resume-safety by
(run_id, case_id)) is session 5 and will live in this module too.

Cost guard contract (§7.2): the runner holds a hard budget cap ($10 for the
MVP). Before initiating each API call it checks the running total; once the
total has reached the cap it aborts with a clear message instead of making
the next call. Because a call's cost is only known after it returns, the
overage is bounded by a single call's cost — the guard can never prevent the
call that crosses the cap, only everything after it.
"""

from __future__ import annotations


class BudgetExceeded(Exception):
    """Raised before a call that must not be made because the cap is hit."""


class CostMeter:
    """Accumulates spent USD and enforces the hard cap (§7.2, §2 row 10)."""

    def __init__(self, cap_usd: float):
        if cap_usd <= 0:
            raise ValueError(f"cap must be positive, got {cap_usd!r}")
        self.cap_usd = cap_usd
        self.total_usd = 0.0
        self.calls = 0

    def check(self) -> None:
        """Call before each API call; raises BudgetExceeded once the cap is hit."""
        if self.total_usd >= self.cap_usd:
            raise BudgetExceeded(
                f"budget cap ${self.cap_usd:.2f} reached: spent ${self.total_usd:.4f} "
                f"over {self.calls} call(s); aborting before the next call"
            )

    def charge(self, cost_usd: float) -> None:
        """Record the cost of a completed call (errors cost $0.00)."""
        self.total_usd += cost_usd
        self.calls += 1
#!/usr/bin/env python3
"""Session-2 smoke: run a few eval rows through one OpenRouter adapter (§13).

Usage:
    python scripts/smoke.py [--model gemini-3.8-flash] [--version V1|V2] [--rows 5]

Row selection: the file is laid out as 75 difficult rows first (5 per intent,
in frozen alphabetical intent order), then 75 non-difficult rows; picking
every 10th row from the start therefore lands on the first difficult row of
5 distinct intents (intents #1, 3, 5, 7, 9). Scoring/normalization (§8) lands
in session 6; this script reports the raw returned label plus a strict
exact-match precheck.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tierline_mvp import contestants, schema  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-3.8-flash",
                    choices=sorted(contestants.OPENROUTER_MODELS))
    ap.add_argument("--version", default="V1", choices=("V1", "V2"))
    ap.add_argument("--rows", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            (ROOT / "data" / "eval_v0.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.rows > 15:
        raise SystemExit("smoke is capped at 15 rows (one per intent)")
    picked = [rows[i * 10] for i in range(args.rows)]

    contestant = contestants.build_contestant(args.model)
    print(f"model={contestant.name}  slug={contestant.slug}  "
          f"reasoning={contestant.reasoning_effort}  version={args.version}  "
          f"rows={len(picked)}")

    total_cost = 0.0
    scored = 0
    exact = 0
    for row in picked:
        result = contestant.predict(
            row["text"], list(schema.OPTIONS), schema.DEFINITIONS, args.version
        )
        total_cost += result["cost_usd"]
        label = result["label"]
        valid = label in schema.OPTIONS if label else False
        exact += int(valid and label == row["intent"])
        scored += int(result["error"] is None)
        status = "ok" if result["error"] is None else f"ERROR: {result['error']}"
        print(f"  {row['row_id']}  {row['intent']:16} -> {str(label):16} "
              f"valid={str(valid):5} latency={result['latency_ms']:>6}ms "
              f"cost=${result['cost_usd']:.6f}  {status}")
        if not valid and result["error"] is None:
            print(f"      raw: {str(result['raw'])[:200]!r}")

    print(f"--- {scored}/{len(picked)} rows scored (no API error); "
          f"{exact}/{len(picked)} exact matches (pre-normalization); "
          f"total cost ${total_cost:.6f}")


if __name__ == "__main__":
    main()
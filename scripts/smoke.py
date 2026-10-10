#!/usr/bin/env python3
"""Per-model smoke: run a few eval rows through one OpenRouter adapter (§13).

Usage:
    python scripts/smoke.py [--model gemini-3.8-flash] [--version V1|V2] [--rows 5]
                            [--cap-usd 0.01]

Row selection: the file is laid out as 75 difficult rows first (5 per intent,
in frozen alphabetical intent order), then 75 non-difficult rows; picking
every 10th row from the start therefore lands on the first difficult row of
5 distinct intents (intents #1, 3, 5, 7, 9).

Rows are scored with the §8 per-row rules (scoring.score) and, if --cap-usd
is given, the run demonstrates the §7.2 cost guard: once the running total
reaches the cap, the script aborts with a clear message before the next call.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tierline_mvp import contestants, runner, schema, scoring  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gemini-3.8-flash",
                    choices=sorted(list(contestants.OPENROUTER_MODELS)
                                   + list(contestants.LOCAL_CONTESTANT_KEYS)))
    ap.add_argument("--version", default="V1", choices=("V1", "V2"))
    ap.add_argument("--rows", type=int, default=5)
    ap.add_argument("--cap-usd", type=float, default=None,
                    help="budget cap for this smoke (e.g. 0.01 to demo the abort)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            (ROOT / "data" / "eval_v0.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.rows > 15:
        raise SystemExit("smoke is capped at 15 rows (one per intent)")
    picked = [rows[i * 10] for i in range(args.rows)]

    contestant = contestants.build_contestant(args.model)
    meter = runner.CostMeter(args.cap_usd) if args.cap_usd else None
    slug = getattr(contestant, "slug", "(native)")
    effort = getattr(contestant, "reasoning_effort", None) or "n/a"
    print(f"model={contestant.name}  slug={slug}  "
          f"reasoning={effort}  version={args.version}  "
          f"rows={len(picked)}" + (f"  cap=${args.cap_usd:.2f}" if meter else ""))

    # Persist every smoke result (incl. raw output) in the §9 results-row
    # schema. Smoke files are diagnostics, not the frozen run store: each run
    # overwrites results/smoke/<model>-<version>.jsonl; append-only runs with
    # resume-safety arrive in session 5 under results/runs/.
    out_path = ROOT / "results" / "smoke" / f"{args.model}-{args.version}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = out_path.open("w", encoding="utf-8")
    print(f"saving raw results to {out_path.relative_to(ROOT)}")

    def result_row(row, result) -> dict:
        return {
            "run_id": f"smoke-{args.model}-{args.version}",
            "row_id": row["row_id"],
            "model": contestant.name,
            "prompt_version": args.version,
            "label": result["label"],
            "probs": result["probs"],
            "latency_ms": result["latency_ms"],
            "cost_usd": result["cost_usd"],
            "raw": result["raw"],
            "error": result["error"],
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    scored = 0
    correct = 0
    oos_selections = 0
    try:
        for row in picked:
            if meter:
                meter.check()  # §7.2: abort BEFORE the call that must not happen
            result = contestant.predict(
                row["text"], list(schema.OPTIONS), schema.DEFINITIONS, args.version
            )
            verdict = scoring.score(result["label"], row["intent"])
            scored += int(result["error"] is None)
            correct += int(verdict["correct"])
            oos_selections += int(verdict["oos_selection"])
            if meter:
                meter.charge(result["cost_usd"])
            out_file.write(json.dumps(result_row(row, result), ensure_ascii=False) + "\n")
            out_file.flush()
            status = "ok" if result["error"] is None else f"ERROR: {result['error']}"
            print(f"  {row['row_id']}  {row['intent']:16} -> {str(result['label']):16} "
                  f"valid={str(verdict['valid']):5} correct={str(verdict['correct']):5} "
                  f"oos_sel={str(verdict['oos_selection']):5} "
                  f"latency={result['latency_ms']:>6}ms cost=${result['cost_usd']:.6f}  {status}")
            if not verdict["valid"] and result["error"] is None:
                print(f"      raw: {str(result['raw'])[:200]!r}")
    except runner.BudgetExceeded as exc:
        out_file.close()
        print(f"BUDGET ABORT: {exc}")
        if meter:
            print(f"--- spent ${meter.total_usd:.4f} of the ${meter.cap_usd:.2f} test cap "
                  f"({meter.calls} call(s) completed); remaining rows not called")
        return
    out_file.close()

    total_cost = meter.total_usd if meter else float("nan")
    cost_note = f"total cost ${total_cost:.6f}" if meter else "(no cap set)"
    print(f"--- {scored}/{len(picked)} rows scored (no API error); "
          f"{correct}/{len(picked)} correct (§8); "
          f"{oos_selections}/{len(picked)} oos selections; {cost_note}")


if __name__ == "__main__":
    main()
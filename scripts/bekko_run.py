#!/usr/bin/env python3
"""Run bekko-system-one-v0-400m on the MVP eval set (S1 roster slot 4).

Runs in a SEPARATE venv (.venv-bekko) because bekko pins
torch>=2.10,<2.11 / transformers==5.17.0 / sentence-transformers==6.1.0,
which conflict with the main venv. Usage:

    .venv-bekko/bin/python scripts/bekko_run.py [--limit N]

Writes results/runs/v1-bekko-400m-V1.jsonl in the §9 results-row schema
(resume-aware: rows already recorded with error=None are skipped). The input
format mirrors S1MB's bekko_v0 adapter (state_json + decisions with
criteria/description_json), loaded through
inference_v0.BekkoSentenceTransformer with trust_remote_code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tierline_mvp"))

EVAL_PATH = ROOT / "data" / "eval_v0.jsonl"
OUT_PATH = ROOT / "results" / "runs" / "v1-bekko-400m-V1.jsonl"
REPO = "hotchpotch/bekko-system-one-v0-400m"

# The frozen §5 task framing + definitions, imported from the main package if
# importable, else transcribed fallback is NOT allowed — fail hard instead.
try:
    from tierline_mvp import schema  # noqa: E402

    OPTIONS = list(schema.OPTIONS)
    DEFINITIONS = dict(schema.DEFINITIONS)
    INSTRUCTIONS = "Given one customer request, decide which of the following handlers should take it."
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "tierline_mvp not importable from the bekko venv; run with "
        "PYTHONPATH=src so the frozen constants load from the main package"
    ) from exc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    import importlib

    torch = importlib.import_module("torch")
    dynamic = importlib.import_module("transformers.dynamic_module_utils")
    from huggingface_hub import snapshot_download

    print("downloading/loading bekko checkpoint (trust_remote_code)...")
    local_dir = snapshot_download(REPO)
    model_class = dynamic.get_class_from_dynamic_module(
        "inference_v0.BekkoSentenceTransformer", local_dir, local_files_only=True
    )
    model = model_class(local_dir, device="cpu", trust_remote_code=True).eval()
    print("loaded; warming up capacity validation...")

    rows = [json.loads(line) for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        rows = rows[: args.limit]

    done: set[str] = set()
    if OUT_PATH.exists():
        for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                if record.get("error") is None:
                    done.add(record["row_id"])
    pending = [r for r in rows if r["row_id"] not in done]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    scored = 0
    started = time.perf_counter()
    out_file = OUT_PATH.open("a", encoding="utf-8")
    for row in pending:
        native_input = {
            "state_json": json.dumps(row["text"], ensure_ascii=False),
            "decisions": [{
                "id": "intent",
                "kind": "judgment",
                "type": "choice",
                "instructions_json": json.dumps(INSTRUCTIONS, ensure_ascii=False),
                "system_prompt": None,
                "criteria": [
                    {"id": option,
                     "description_json": json.dumps(DEFINITIONS[option], ensure_ascii=False),
                     "value": None}
                    for option in OPTIONS
                ],
                "documents": [],
                "scoring": None,
            }],
        }
        t0 = time.perf_counter()
        try:
            # model.predict returns one result dict per input case (S1MB adapter
            # zips chunk with results) — index the single case.
            result = model.predict([native_input], batch_size=1, show_progress_bar=False)[0]
            answer = result["intent"]
            label = answer["selected_id"]
            probs = answer.get("probabilities")
            error = None
        except Exception as exc:  # noqa: BLE001 — recorded as an error row
            label, probs, error = None, None, f"{type(exc).__name__}: {exc}"
        latency_ms = int((time.perf_counter() - t0) * 1000)
        record = {
            "run_id": "v1-bekko-400m-V1",
            "row_id": row["row_id"],
            "model": "bekko-system-one-v0-400m",
            "prompt_version": "V1",
            "label": label,
            "probs": probs,
            "latency_ms": latency_ms,
            "cost_usd": 0.0,
            "raw": json.dumps(result, ensure_ascii=False, default=str) if error is None else None,
            "error": error,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        out_file.flush()
        scored += error is None
        if scored % 25 == 0:
            print(f"  {scored}/{len(pending)} scored...")

    out_file.close()
    print(f"done: {scored} newly scored, {len(pending) - scored} errors, "
          f"{round(time.perf_counter() - started, 1)}s")


if __name__ == "__main__":
    main()
"""Runner (design doc §9): CLI, append-only JSONL results, resume, cost guard.

Session 5 deliverable. Semantics:

- run_id = v1-{model_key}-{prompt_version}; one JSONL file per (model, version)
  under results/runs/ with the §9 results-row schema.
- Resume-safety: rows already present with error=None are skipped; rows whose
  recorded outcome was an error are re-run (an errored row is effectively
  absent). Appends are append-only; when a row appears twice (error then
  success after a resume), the LAST occurrence wins at analysis time.
- Ordering: rows are dispatched in file order. With concurrency >1 the
  appends land in completion order; row_id is the join key.
- Concurrency: ≤4 calls in flight per model (§9), shared by that model's
  versions; a global semaphore caps total in-flight calls. Local System-One
  models run one model at a time with workers=1 (latency_ms is a recorded
  field; contention would distort it).
- Budget: the §7.2 CostMeter enforces the doc cap ($10). The cap is bounded
  by the live OpenRouter balance minus a reserve (probed at preflight and
  re-probed every `--balance-every` completed calls). HTTP 402 anywhere
  aborts the whole run: every subsequent call would fail identically, so
  continuing would only manufacture error rows.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # allow `python src/tierline_mvp/runner.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tierline_mvp import contestants, schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "runs"
DOC_CAP_USD = 10.0

LLM_KEYS = tuple(contestants.OPENROUTER_MODELS)  # §7.1 rows 6-10
S1_KEYS = tuple(contestants.HOSTED_S1_KEYS) + tuple(contestants.LOCAL_S1_KEYS)  # §7.1 rows 1-5

BUDGET_EXCEEDED = "budget"
ABORTED_402 = "http_402"
ABORTED_BALANCE = "balance"


class BudgetExceeded(Exception):
    """Raised before a call that must not be made because the cap is hit."""


class CostMeter:
    """Accumulates spent USD and enforces the hard cap (§7.2). Thread-safe."""

    def __init__(self, cap_usd: float):
        if cap_usd <= 0:
            raise ValueError(f"cap must be positive, got {cap_usd!r}")
        self.cap_usd = cap_usd
        self.total_usd = 0.0
        self.calls = 0
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self.total_usd >= self.cap_usd:
                raise BudgetExceeded(
                    f"budget cap ${self.cap_usd:.2f} reached: spent ${self.total_usd:.4f} "
                    f"over {self.calls} call(s); aborting before the next call"
                )

    def charge(self, cost_usd: float) -> None:
        with self._lock:
            self.total_usd += cost_usd
            self.calls += 1


def run_id_for(key: str, version: str) -> str:
    return f"v1-{key}-{version}"


def run_path(key: str, version: str) -> Path:
    return RESULTS_DIR / f"{run_id_for(key, version)}.jsonl"


def load_pending(rows: list[dict], path: Path) -> list[dict]:
    """Rows not yet successfully recorded (§9 resume-safety; errored rows re-run)."""
    done: set[str] = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error") is None:
                done.add(record["row_id"])
    return [row for row in rows if row["row_id"] not in done]


def result_row(key: str, display_name: str, version: str, row: dict, result: dict) -> dict:
    return {
        "run_id": run_id_for(key, version),
        "row_id": row["row_id"],
        "model": display_name,
        "prompt_version": version,
        "label": result["label"],
        "probs": result["probs"],
        "latency_ms": result["latency_ms"],
        "cost_usd": result["cost_usd"],
        "raw": result["raw"],
        "error": result["error"],
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


class TaskRun:
    """One (model, version) grid slice; thread-safe appends."""

    def __init__(self, key: str, version: str, contestant, rows: list[dict],
                 meter: CostMeter, workers: int, global_slots: threading.Semaphore,
                 stop_event: threading.Event, state: dict):
        self.key = key
        self.version = version
        self.contestant = contestant
        self.rows = rows
        self.meter = meter
        self.workers = workers
        self.global_slots = global_slots
        self.stop_event = stop_event
        self.state = state  # shared: {"reason": None|str, "detail": str, lock}
        self.run_id = run_id_for(key, version)
        self.path = run_path(key, version)
        self.display_name = contestant.name
        self._write_lock = threading.Lock()

    def _append(self, record: dict) -> None:
        with self._write_lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                fh.flush()

    def _fail(self, reason: str, detail: str) -> None:
        with self.state["lock"]:
            if self.state["reason"] is None:
                self.state["reason"] = reason
                self.state["detail"] = detail
        self.stop_event.set()

    def run(self) -> dict:
        pending = load_pending(self.rows, self.path)
        summary = {"run_id": self.run_id, "pending": len(pending),
                   "scored": 0, "errors": 0, "cost_usd": 0.0, "skipped_done": len(self.rows) - len(pending)}
        if not pending or self.stop_event.is_set():
            return summary

        def work(row: dict) -> dict:
            if self.stop_event.is_set():
                raise _Stopped()
            try:
                self.meter.check()
            except BudgetExceeded as exc:
                self._fail(BUDGET_EXCEEDED, str(exc))
                raise _Stopped() from exc
            with self.global_slots:
                result = self.contestant.predict(
                    row["text"], list(schema.OPTIONS), schema.DEFINITIONS, self.version
                )
            self.meter.charge(result["cost_usd"])
            self._append(result_row(self.key, self.display_name, self.version, row, result))
            if result["error"] and result["error"].startswith("HTTP 402"):
                self._fail(ABORTED_402, f"OpenRouter returned 402 on {row['row_id']}")
                raise _Stopped()
            return result

        started = time.perf_counter()
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(work, row) for row in pending]  # file order
                for future in list(futures):
                    try:
                        result = future.result()
                    except _Stopped:
                        break
                    if result["error"] is None:
                        summary["scored"] += 1
                    else:
                        summary["errors"] += 1
                    summary["cost_usd"] += result["cost_usd"]
        finally:
            summary["elapsed_s"] = round(time.perf_counter() - started, 1)
        summary["cost_usd"] = round(summary["cost_usd"], 6)
        return summary


class _Stopped(Exception):
    """Internal: worker unwinds after a cooperative stop."""


def probe_served_model(contestant) -> dict:
    """One 1-token call to verify auth and record the served model id (§7.1)."""
    payload = {
        "model": contestant.slug,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    resp = contestant._http().post(contestants.OPENROUTER_URL,
                                   headers=contestant._headers(), json=payload)
    resp.raise_for_status()
    data = resp.json()
    return {"slug": contestant.slug, "served_model": data.get("model"),
            "provider": data.get("provider"), "verified_at":
                datetime.now(timezone.utc).isoformat(timespec="seconds")}


def update_run_manifest(entries: dict) -> None:
    path = RESULTS_DIR / "MANIFEST.json"
    manifest = {}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.update(entries)
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_grid(keys: list[str], versions: list[str], rows: list[dict], *,
             workers: int = 4, max_concurrent: int = 20, cap_usd: float = DOC_CAP_USD,
             reserve_usd: float = 0.5, limit: int | None = None,
             balance_every: int = 25, use_balance_guard: bool = True) -> dict:
    """Run the grid for the given roster keys; returns summaries + abort info."""
    if limit is not None:
        rows = rows[:limit]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    api_keys = [k for k in keys if k in contestants.OPENROUTER_MODELS]
    if use_balance_guard and api_keys:
        balance = contestants.openrouter_balance()
        limits = [v for v in balance.values() if v is not None]
        ceiling = min(limits) if limits else cap_usd
        cap_usd = min(cap_usd, ceiling - reserve_usd)
        print(f"balance: account=${balance['account_remaining_usd']:.2f} "
              f"key=${balance['key_limit_remaining_usd']} -> effective cap ${cap_usd:.2f} "
              f"(doc cap ${DOC_CAP_USD:.2f}, reserve ${reserve_usd:.2f})")
        if cap_usd <= 0:
            raise SystemExit("refusing to start: effective budget after reserve is $0")
        update_run_manifest({"balance_at_preflight": balance, "effective_cap_usd": cap_usd})
        for key in api_keys:
            served = probe_served_model(contestants.build_contestant(key))
            print(f"served check: {key} -> {served['served_model']} via {served['provider']}")
            update_run_manifest({f"served_{key}": served})

    meter = CostMeter(cap_usd)
    stop_event = threading.Event()
    state = {"reason": None, "detail": None, "lock": threading.Lock()}
    global_slots = threading.Semaphore(max_concurrent)
    summaries: dict[str, dict] = {}

    def model_worker(key: str) -> None:
        contestant = contestants.build_contestant(key)
        for version in versions:
            if stop_event.is_set():
                return
            task = TaskRun(key, version, contestant, rows, meter,
                           workers if key in contestants.OPENROUTER_MODELS else 1,
                           global_slots, stop_event, state)
            summaries[task.run_id] = task.run()
            # Coarse pacing check after each task completes: the meter and the
            # HTTP-402 detector are the real-time guards; this probe catches a
            # balance that has drifted low between tasks.
            if use_balance_guard and key in contestants.OPENROUTER_MODELS:
                balance = contestants.openrouter_balance()
                limits = [v for v in balance.values() if v is not None]
                remaining = min(limits) if limits else float("inf")
                if remaining <= reserve_usd:
                    state["reason"] = ABORTED_BALANCE
                    state["detail"] = f"balance ${remaining:.2f} at/below reserve ${reserve_usd:.2f}"
                    stop_event.set()

    threads = [threading.Thread(target=model_worker, args=(k,), name=k) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    total_cost = meter.total_usd
    print(f"\n=== grid summary (spent ${total_cost:.4f} of ${meter.cap_usd:.2f} cap) ===")
    for run_id, s in summaries.items():
        print(f"  {run_id:42} scored={s['scored']:3} errors={s['errors']:2} "
              f"cost=${s['cost_usd']:.4f} elapsed={s.get('elapsed_s', 0):>6}s "
              f"(skipped {s['skipped_done']} done)")
    if state["reason"]:
        print(f"ABORT: {state['reason']}: {state['detail']}")
    return {"summaries": summaries, "spent_usd": total_cost,
            "reason": state["reason"], "detail": state["detail"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="TierLine MVP runner (§9)")
    ap.add_argument("--keys", nargs="+", default=list(LLM_KEYS),
                    help="roster keys (default: the five OpenRouter LLMs)")
    ap.add_argument("--versions", nargs="+", default=["V1", "V2"], choices=["V1", "V2"])
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent calls per model (§9: API models ≤ 4)")
    ap.add_argument("--max-concurrent", type=int, default=20,
                    help="global in-flight cap across all tasks")
    ap.add_argument("--limit", type=int, default=None, help="only the first N rows (sanity runs)")
    ap.add_argument("--cap-usd", type=float, default=DOC_CAP_USD)
    ap.add_argument("--reserve-usd", type=float, default=0.5)
    ap.add_argument("--balance-every", type=int, default=25)
    ap.add_argument("--no-balance-guard", action="store_true",
                    help="skip the live balance probe (offline/local runs)")
    args = ap.parse_args()

    rows = [json.loads(line) for line in
            (ROOT / "data" / "eval_v0.jsonl").read_text(encoding="utf-8").splitlines()]
    run_grid(args.keys, args.versions, rows, workers=args.workers,
             max_concurrent=args.max_concurrent, cap_usd=args.cap_usd,
             reserve_usd=args.reserve_usd, limit=args.limit,
             balance_every=args.balance_every,
             use_balance_guard=not args.no_balance_guard)


if __name__ == "__main__":
    main()
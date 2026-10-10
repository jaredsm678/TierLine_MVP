#!/usr/bin/env python3
"""Download the frozen source artifacts into data/sources/ (design doc §4.1, §10.5).

Usage:
    python scripts/fetch_sources.py

Idempotent: skips files already present with a matching (or unpinned) SHA-256.
Fails hard if a required artifact's SHA-256 contradicts its pinned value.
domains.json is auxiliary: if it cannot be downloaded the build may still
proceed (the banking intent list is frozen in tierline_mvp.schema), and the
gap is recorded in the MANIFEST notes.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tierline_mvp import schema  # noqa: E402  (path shim must run first)

DEST = ROOT / "data" / "sources"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, meta: dict) -> None:
    dest = DEST / name
    pinned = meta["sha256"]

    if dest.exists():
        digest = sha256_file(dest)
        if pinned is None or digest == pinned:
            print(f"[skip]    {name}: present, sha256 {digest}")
            return
        print(f"[refetch] {name}: present but sha256 {digest} != pin {pinned}")

    last_err = None
    for url in meta["urls"]:
        try:
            print(f"[get ]    {name}: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "tierline-mvp/0.1"})
            with urllib.request.urlopen(req, timeout=120) as resp, dest.open("wb") as fh:
                while chunk := resp.read(1 << 20):
                    fh.write(chunk)
            last_err = None
            break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = e
            print(f"          failed: {e}")
    if last_err is not None:
        if meta["required"]:
            raise SystemExit(f"ERROR: could not download required artifact {name}: {last_err}")
        print(f"[warn]    {name}: unavailable ({last_err}); build will record this in the MANIFEST")
        return

    digest = sha256_file(dest)
    if pinned is not None and digest != pinned:
        dest.unlink()
        raise SystemExit(
            f"ERROR: SHA-256 mismatch for {name}:\n  got      {digest}\n  expected {pinned}"
        )
    print(f"[ok  ]    {name}: sha256 {digest}")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, meta in schema.SOURCE_ARTIFACTS.items():
        fetch(name, meta)
    print("done.")


if __name__ == "__main__":
    main()

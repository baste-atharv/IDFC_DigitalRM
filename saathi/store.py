"""Append-only logs: conversations, RM handoffs (raise_hand), consents, submitted leads, tickets, compliance flags.
Each record carries the hash of the previous one, so later edits are detectable (same idea as the PS7 audit log)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

from . import config as C


def append(kind: str, **rec) -> dict:
    path = C.LOGS / f"{kind}.jsonl"
    prev = "GENESIS"
    if path.exists():
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if lines:
            prev = json.loads(lines[-1])["hash"]
    rec = dict(ts=datetime.now().isoformat(timespec="seconds"), **rec, prev_hash=prev)
    rec["hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    return rec


def read(kind: str) -> list[dict]:
    path = C.LOGS / f"{kind}.jsonl"
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def verify(kind: str) -> bool:
    prev = "GENESIS"
    for e in read(kind):
        h = e.pop("hash")
        if e["prev_hash"] != prev or hashlib.sha256(json.dumps(e, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest() != h:
            return False
        prev = h
    return True

"""Durable append-only JSON-lines ledger.

P1 guarantees: fsync per record, per-file locks so concurrent writers
never interleave partial lines, size-based rotation, correlation IDs
linking grid_fire -> entry -> exit, and ``ts_monotonic_ns`` on every
record for safe duration math when wall clock jumps.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from typing import Any, Dict, Optional

from detectors.base import Direction, SignalFire


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:16]


class LedgerStore:
    DEFAULT_MAX_BYTES = 50 * 1024 * 1024

    def __init__(self, directory: str, max_bytes: int = DEFAULT_MAX_BYTES):
        self._dir = directory
        self._max_bytes = max_bytes
        os.makedirs(self._dir, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}
        self._locks_mutex = threading.Lock()

    def log_signal_fire(self, fire: SignalFire) -> None:
        self._append("signal_fires.jsonl", {
            "ts": fire.timestamp,
            "ts_monotonic_ns": time.monotonic_ns(),
            "detector": fire.detector_name,
            "market": fire.market,
            "token_id": fire.token_id,
            "direction": fire.direction.value,
            "confidence": fire.confidence,
            "meta": fire.meta,
        })

    def log_grid_fire(self, market, direction, n_signals, agreement,
                      correlation_id):
        self._append("grid_fires.jsonl", {
            "ts": time.time(),
            "ts_monotonic_ns": time.monotonic_ns(),
            "correlation_id": correlation_id,
            "market": market,
            "direction": direction.value,
            "n_signals": n_signals,
            "agreement": agreement,
        })

    def log_entry(self, market, token_id, direction, size, price, mode,
                  meta, correlation_id):
        self._append("entries.jsonl", {
            "ts": time.time(),
            "ts_monotonic_ns": time.monotonic_ns(),
            "correlation_id": correlation_id,
            "market": market,
            "token_id": token_id,
            "direction": direction,
            "size": size,
            "price": price,
            "mode": mode,
            "meta": meta,
        })

    def log_exit(self, market, token_id, direction, size, price,
                 pnl_usdc, mode, correlation_id):
        self._append("exits.jsonl", {
            "ts": time.time(),
            "ts_monotonic_ns": time.monotonic_ns(),
            "correlation_id": correlation_id,
            "market": market,
            "token_id": token_id,
            "direction": direction,
            "size": size,
            "price": price,
            "pnl_usdc": pnl_usdc,
            "mode": mode,
        })

    def log_block(self, reason: str, market: str, meta: Optional[dict] = None) -> None:
        self._append("audit_log.jsonl", {
            "ts": time.time(),
            "ts_monotonic_ns": time.monotonic_ns(),
            "event": "block",
            "reason": reason,
            "market": market,
            "meta": meta or {},
        })

    def log_audit(self, event: str, **fields) -> None:
        rec = {
            "ts": time.time(),
            "ts_monotonic_ns": time.monotonic_ns(),
            "event": event,
        }
        rec.update(fields)
        self._append("audit_log.jsonl", rec)

    def _append(self, filename: str, record: Dict[str, Any]) -> None:
        data = (json.dumps(record, default=str) + "\n").encode("utf-8")
        with self._get_lock(filename):
            self._rotate_if_needed(filename)
            path = os.path.join(self._dir, filename)
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, data)
                os.fsync(fd)
            finally:
                os.close(fd)

    def _rotate_if_needed(self, filename: str) -> None:
        path = os.path.join(self._dir, filename)
        try:
            size = os.path.getsize(path)
        except FileNotFoundError:
            return
        if size < self._max_bytes:
            return
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        stem, ext = os.path.splitext(filename)
        os.rename(path, os.path.join(self._dir, f"{stem}.{stamp}{ext}"))

    def _get_lock(self, filename: str) -> threading.Lock:
        with self._locks_mutex:
            lk = self._locks.get(filename)
            if lk is None:
                lk = threading.Lock()
                self._locks[filename] = lk
            return lk

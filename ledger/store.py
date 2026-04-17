"""
Ledger store.

Append-only JSON-lines files for signal fires, grid fires, entries,
exits, and PnL.  Designed for post-run analysis, not real-time reads.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from detectors.base import Direction, SignalFire


class LedgerStore:
    def __init__(self, directory: str):
        self._dir = directory
        os.makedirs(self._dir, exist_ok=True)

    # ── writers ──────────────────────────────────────────────────────

    def log_signal_fire(self, fire: SignalFire) -> None:
        self._append("signal_fires.jsonl", {
            "ts": fire.timestamp,
            "detector": fire.detector_name,
            "market": fire.market,
            "token_id": fire.token_id,
            "direction": fire.direction.value,
            "confidence": fire.confidence,
            "meta": fire.meta,
        })

    def log_grid_fire(
        self, market: str, direction: Direction, n_signals: int, agreement: float
    ) -> None:
        self._append("grid_fires.jsonl", {
            "ts": time.time(),
            "market": market,
            "direction": direction.value,
            "n_signals": n_signals,
            "agreement": agreement,
        })

    def log_entry(self, market: str, token_id: str, direction: str,
                  size: float, price: float, mode: str, meta: dict) -> None:
        self._append("entries.jsonl", {
            "ts": time.time(),
            "market": market,
            "token_id": token_id,
            "direction": direction,
            "size": size,
            "price": price,
            "mode": mode,
            "meta": meta,
        })

    def log_exit(self, market: str, token_id: str, direction: str,
                 size: float, price: float, pnl_usdc: float, mode: str) -> None:
        self._append("exits.jsonl", {
            "ts": time.time(),
            "market": market,
            "token_id": token_id,
            "direction": direction,
            "size": size,
            "price": price,
            "pnl_usdc": pnl_usdc,
            "mode": mode,
        })

    # ── internal ─────────────────────────────────────────────────────

    def _append(self, filename: str, record: Dict[str, Any]) -> None:
        path = os.path.join(self._dir, filename)
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")

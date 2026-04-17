"""
Whale feed poller.

Polls the public Polymarket Data API for recent activity from
configured whale wallets and injects trades into the whale detector.

This poller is a no-op if ``config.whale_wallets`` is empty.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, Set

import requests

if TYPE_CHECKING:
    from detectors.whale import WhaleDetector
    from grid.config import GridConfig


DATA_API = "https://data-api.polymarket.com"


class WhalePoller:
    def __init__(self, config: "GridConfig", detector: "WhaleDetector"):
        self.config = config
        self._det = detector
        # Track already-seen trade IDs so we don't double-fire.
        self._seen: Set[str] = set()
        # Bound the seen-set size.
        self._max_seen = 10_000

    def poll(self) -> None:
        if not self.config.whale_wallets:
            return

        for wallet in self.config.whale_wallets:
            try:
                resp = requests.get(
                    f"{DATA_API}/activity",
                    params={"user": wallet},
                    timeout=10,
                )
                resp.raise_for_status()
                rows = resp.json()
            except Exception as exc:
                print(f"[whale] poll {wallet[:10]}… failed: {exc}")
                continue

            if not isinstance(rows, list):
                continue

            for row in rows:
                tid = row.get("id", "")
                if tid in self._seen:
                    continue
                self._seen.add(tid)
                if len(self._seen) > self._max_seen:
                    # Evict oldest half (set is unordered, but good enough)
                    to_remove = list(self._seen)[:self._max_seen // 2]
                    for r in to_remove:
                        self._seen.discard(r)

                market = row.get("conditionId", "") or row.get("market", "")
                token_id = row.get("asset", "")
                side = row.get("side", "BUY")
                size = float(row.get("size", 0))

                if market and size > 0:
                    self._det.record_whale_trade(market, token_id, wallet, side, size)

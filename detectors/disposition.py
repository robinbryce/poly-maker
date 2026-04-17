"""
Disposition detector.

Fires when recent taker aggression is strongly one-sided, inferred
from the direction and size of fills in a short window.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

from detectors.base import BaseDetector, Direction, SignalFire
from grid.config import GridConfig


class DispositionDetector(BaseDetector):
    name = "disposition"

    WINDOW = 60  # seconds

    def __init__(self, config: GridConfig):
        self.config = config
        # Per-market deques of (timestamp, signed_size)
        # positive = taker buying, negative = taker selling
        self._fills: dict[str, deque] = defaultdict(deque)

    def on_event(self, event: dict) -> List[SignalFire]:
        if event.get("event_type") != "trade":
            return []

        market = event.get("market", "")
        token_id = event.get("asset_id", "")
        size = float(event.get("size", 0))
        side = event.get("side", "").upper()
        now = time.time()

        signed = size if side == "BUY" else -size
        self._fills[market].append((now, signed))
        self._prune(market, now)

        buy_vol, sell_vol = self._directional(market, now)
        total = buy_vol + sell_vol
        if total == 0:
            return []

        ratio = max(buy_vol, sell_vol) / total
        if ratio < self.config.disposition_threshold:
            return []

        direction = Direction.BUY if buy_vol > sell_vol else Direction.SELL
        confidence = ratio

        return [SignalFire(
            detector_name=self.name,
            market=market,
            token_id=token_id,
            direction=direction,
            confidence=confidence,
            meta={"buy_vol": buy_vol, "sell_vol": sell_vol, "ratio": ratio},
        )]

    def reset(self, market: str) -> None:
        self._fills.pop(market, None)

    # ── helpers ──────────────────────────────────────────────────────

    def _prune(self, market: str, now: float) -> None:
        q = self._fills[market]
        cutoff = now - self.WINDOW
        while q and q[0][0] < cutoff:
            q.popleft()

    def _directional(self, market: str, now: float):
        cutoff = now - self.WINDOW
        buy = sum(s for t, s in self._fills[market] if t >= cutoff and s > 0)
        sell = sum(abs(s) for t, s in self._fills[market] if t >= cutoff and s < 0)
        return buy, sell

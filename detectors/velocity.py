"""
Velocity detector.

Fires when the rate of midpoint price change in a short window exceeds
a threshold, normalised against the market's own recent movement.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

from detectors.base import BaseDetector, Direction, SignalFire
from grid.config import GridConfig


class VelocityDetector(BaseDetector):
    name = "velocity"

    SHORT_WINDOW = 30   # seconds – measure acceleration here
    LONG_WINDOW = 300   # seconds – normalisation baseline

    def __init__(self, config: GridConfig):
        self.config = config
        # Per-market deques of (timestamp, midpoint)
        self._prices: dict[str, deque] = defaultdict(deque)

    def on_event(self, event: dict) -> List[SignalFire]:
        """Expects events that carry a midpoint snapshot for a market."""
        if event.get("event_type") not in ("book", "price_change"):
            return []

        market = event.get("market", "")
        token_id = event.get("asset_id", "")
        mid = event.get("midpoint")
        if mid is None:
            return []

        mid = float(mid)
        now = time.time()
        self._prices[market].append((now, mid))
        self._prune(market, now)

        short_delta = self._price_delta(market, now, self.SHORT_WINDOW)
        long_delta = self._price_delta(market, now, self.LONG_WINDOW)

        if short_delta is None:
            return []

        short_rate = abs(short_delta) / self.SHORT_WINDOW
        long_rate = (abs(long_delta) / self.LONG_WINDOW) if long_delta is not None else 0

        # The raw rate must exceed the config threshold AND be notably
        # faster than the longer-term baseline.
        if short_rate < self.config.velocity_threshold:
            return []
        if long_rate > 0 and short_rate < long_rate * 2:
            return []

        direction = Direction.BUY if short_delta > 0 else Direction.SELL
        confidence = min(short_rate / max(self.config.velocity_threshold, 1e-9), 1.0)

        return [SignalFire(
            detector_name=self.name,
            market=market,
            token_id=token_id,
            direction=direction,
            confidence=confidence,
            meta={"short_rate": short_rate, "long_rate": long_rate, "delta": short_delta},
        )]

    def reset(self, market: str) -> None:
        self._prices.pop(market, None)

    # ── helpers ──────────────────────────────────────────────────────

    def _prune(self, market: str, now: float) -> None:
        q = self._prices[market]
        cutoff = now - self.LONG_WINDOW
        while q and q[0][0] < cutoff:
            q.popleft()

    def _price_delta(self, market: str, now: float, window: float):
        cutoff = now - window
        points = [(t, p) for t, p in self._prices[market] if t >= cutoff]
        if len(points) < 2:
            return None
        return points[-1][1] - points[0][1]

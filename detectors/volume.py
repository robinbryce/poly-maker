"""
Volume detector.

Fires when the rolling trade volume in a short window exceeds a
configurable multiplier of the market's own longer-term baseline.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

from detectors.base import BaseDetector, Direction, SignalFire
from grid.config import GridConfig


class VolumeDetector(BaseDetector):
    name = "volume"

    # Rolling windows (seconds)
    SHORT_WINDOW = 60
    LONG_WINDOW = 900  # 15 min baseline
    MIN_BASELINE_TRADES = 5  # need this many trades in the long window

    def __init__(self, config: GridConfig):
        self.config = config
        # Per-market deques of (timestamp, size, side)
        self._trades: dict[str, deque] = defaultdict(deque)

    def on_event(self, event: dict) -> List[SignalFire]:
        if event.get("event_type") != "trade":
            return []

        market = event.get("market", "")
        token_id = event.get("asset_id", "")
        size = float(event.get("size", 0))
        side = event.get("side", "").upper()
        now = time.time()

        self._trades[market].append((now, size, side))
        self._prune(market, now)

        short_vol = self._window_volume(market, now, self.SHORT_WINDOW)
        long_vol = self._window_volume(market, now, self.LONG_WINDOW)
        long_count = self._window_count(market, now, self.LONG_WINDOW)
        baseline = long_vol / (self.LONG_WINDOW / self.SHORT_WINDOW) if long_vol > 0 else 0

        if (baseline <= 0
                or long_count < self.MIN_BASELINE_TRADES
                or short_vol < baseline * self.config.volume_spike_multiplier):
            return []

        # Determine dominant direction in the short window
        buy_vol, sell_vol = self._directional_volume(market, now, self.SHORT_WINDOW)
        if buy_vol > sell_vol:
            direction = Direction.BUY
        elif sell_vol > buy_vol:
            direction = Direction.SELL
        else:
            return []

        confidence = min(short_vol / (baseline * self.config.volume_spike_multiplier), 1.0)
        return [SignalFire(
            detector_name=self.name,
            market=market,
            token_id=token_id,
            direction=direction,
            confidence=confidence,
            meta={"short_vol": short_vol, "baseline": baseline},
        )]

    def reset(self, market: str) -> None:
        self._trades.pop(market, None)

    # ── helpers ──────────────────────────────────────────────────────

    def _prune(self, market: str, now: float) -> None:
        q = self._trades[market]
        cutoff = now - self.LONG_WINDOW
        while q and q[0][0] < cutoff:
            q.popleft()

    def _window_volume(self, market: str, now: float, window: float) -> float:
        cutoff = now - window
        return sum(size for ts, size, _ in self._trades[market] if ts >= cutoff)

    def _window_count(self, market: str, now: float, window: float) -> int:
        cutoff = now - window
        return sum(1 for ts, _, _ in self._trades[market] if ts >= cutoff)

    def _directional_volume(self, market: str, now: float, window: float):
        cutoff = now - window
        buy = sum(s for ts, s, side in self._trades[market] if ts >= cutoff and side == "BUY")
        sell = sum(s for ts, s, side in self._trades[market] if ts >= cutoff and side == "SELL")
        return buy, sell

"""
Per-market rolling signal state.

Maintains the latest fire from each detector for every market, with
staleness pruning.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List

from detectors.base import Direction, SignalFire


class MarketState:
    """Rolling signal map for a single market."""

    def __init__(self, staleness_secs: float):
        self.staleness_secs = staleness_secs
        # detector_name -> latest SignalFire
        self._signals: Dict[str, SignalFire] = {}

    def update(self, fire: SignalFire) -> None:
        self._signals[fire.detector_name] = fire

    def active_signals(self) -> List[SignalFire]:
        now = time.time()
        return [
            f for f in self._signals.values()
            if (now - f.timestamp) < self.staleness_secs
        ]

    def dominant_direction(self) -> tuple[Direction | None, float]:
        """Return (direction, agreement_ratio) or (None, 0) if no consensus."""
        active = self.active_signals()
        if not active:
            return None, 0.0
        buy = sum(1 for f in active if f.direction == Direction.BUY)
        sell = len(active) - buy
        total = len(active)
        if buy > sell:
            return Direction.BUY, buy / total
        elif sell > buy:
            return Direction.SELL, sell / total
        return None, 0.0

    def clear(self) -> None:
        self._signals.clear()


class GridState:
    """Container for all per-market state objects."""

    def __init__(self, staleness_secs: float):
        self.staleness_secs = staleness_secs
        self._markets: Dict[str, MarketState] = {}

    def get(self, market: str) -> MarketState:
        if market not in self._markets:
            self._markets[market] = MarketState(self.staleness_secs)
        return self._markets[market]

    def remove(self, market: str) -> None:
        self._markets.pop(market, None)

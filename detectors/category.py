"""
Category detector.

Fires when the aggregate activity across a market's Gamma category
is unusually elevated, suggesting a regime-level event.  Category
metadata is injected by ``feeds.gamma``.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

from detectors.base import BaseDetector, Direction, SignalFire
from grid.config import GridConfig


class CategoryDetector(BaseDetector):
    name = "category"

    SPIKE_MULTIPLIER = 2.5  # category volume must be this × baseline

    def __init__(self, config: GridConfig):
        self.config = config
        # Mapping from market -> category tag (set by feeds.gamma)
        self._market_category: Dict[str, str] = {}
        # {category: {"recent_volume": float, "baseline_volume": float, "ts": float}}
        self._category_stats: Dict[str, dict] = {}
        # Latest midpoints for direction inference
        self._midpoints: Dict[str, float] = {}

    def set_market_category(self, market: str, category: str) -> None:
        self._market_category[market] = category

    def set_category_stats(
        self, category: str, recent_volume: float, baseline_volume: float
    ) -> None:
        self._category_stats[category] = {
            "recent_volume": recent_volume,
            "baseline_volume": baseline_volume,
            "ts": time.time(),
        }

    def on_event(self, event: dict) -> List[SignalFire]:
        market = event.get("market", "")
        mid = event.get("midpoint")
        if mid is not None:
            self._midpoints[market] = float(mid)
        return []  # evaluation happens on poll()

    def poll(self) -> List[SignalFire]:
        fires: List[SignalFire] = []
        for market, cat in self._market_category.items():
            stats = self._category_stats.get(cat)
            if stats is None:
                continue
            baseline = stats["baseline_volume"]
            if baseline <= 0:
                continue
            ratio = stats["recent_volume"] / baseline
            if ratio < self.SPIKE_MULTIPLIER:
                continue

            # Category is hot — fire with the market's recent price direction
            mid = self._midpoints.get(market)
            # Default to BUY (momentum follows category attention)
            direction = Direction.BUY
            confidence = min(ratio / (self.SPIKE_MULTIPLIER * 2), 1.0)

            fires.append(SignalFire(
                detector_name=self.name,
                market=market,
                token_id="",
                direction=direction,
                confidence=confidence,
                meta={"category": cat, "ratio": ratio},
            ))
        return fires

    def reset(self, market: str) -> None:
        self._midpoints.pop(market, None)

"""
News / oracle-delta detector.

Fires when a public data feed (NOAA, AP, CoinGecko, Reuters-style)
has moved and the market price has not yet adjusted by a configurable
delta.  The actual feed polling is done by ``feeds.oracle``; this
detector compares the latest feed value against the market midpoint.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from detectors.base import BaseDetector, Direction, SignalFire
from grid.config import GridConfig


class NewsDetector(BaseDetector):
    name = "news"

    def __init__(self, config: GridConfig):
        self.config = config
        # Populated by feeds.oracle via set_feed_value()
        # {market: {"value": float, "ts": float, "source": str}}
        self._feed_values: Dict[str, dict] = {}
        # Latest observed midpoints from the websocket layer
        self._midpoints: Dict[str, float] = {}

    # ── streaming side: track midpoints ─────────────────────────────

    def on_event(self, event: dict) -> List[SignalFire]:
        if event.get("event_type") not in ("book", "price_change"):
            return []

        market = event.get("market", "")
        mid = event.get("midpoint")
        if mid is None:
            return []

        self._midpoints[market] = float(mid)
        return self._evaluate(market, event.get("asset_id", ""))

    # ── polling side: called after feeds.oracle refreshes ───────────

    def poll(self) -> List[SignalFire]:
        fires: List[SignalFire] = []
        for market in list(self._feed_values):
            if market in self._midpoints:
                fires.extend(self._evaluate(market, ""))
        return fires

    # ── feed injection (called by the oracle feed poller) ───────────

    def set_feed_value(self, market: str, value: float, source: str) -> None:
        self._feed_values[market] = {
            "value": value,
            "ts": time.time(),
            "source": source,
        }

    # ── core logic ──────────────────────────────────────────────────

    def _evaluate(self, market: str, token_id: str) -> List[SignalFire]:
        feed = self._feed_values.get(market)
        mid = self._midpoints.get(market)
        if feed is None or mid is None:
            return []

        delta_cents = (feed["value"] - mid) * 100
        if abs(delta_cents) < self.config.news_delta_cents:
            return []

        direction = Direction.BUY if delta_cents > 0 else Direction.SELL
        confidence = min(abs(delta_cents) / (self.config.news_delta_cents * 3), 1.0)

        return [SignalFire(
            detector_name=self.name,
            market=market,
            token_id=token_id,
            direction=direction,
            confidence=confidence,
            meta={"delta_cents": delta_cents, "source": feed["source"]},
        )]

    def reset(self, market: str) -> None:
        self._feed_values.pop(market, None)
        self._midpoints.pop(market, None)

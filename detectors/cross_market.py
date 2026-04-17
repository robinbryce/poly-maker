"""
Cross-market detector.

Fires when the implied probability of a Polymarket market diverges
from a configured external reference price by more than a threshold.
The reference values are injected by ``feeds.cross_market``.
"""

from __future__ import annotations

import time
from typing import Dict, List

from detectors.base import BaseDetector, Direction, SignalFire
from grid.config import GridConfig


class CrossMarketDetector(BaseDetector):
    name = "cross_market"

    def __init__(self, config: GridConfig):
        self.config = config
        # Injected by feeds.cross_market
        # {market: {"ref_price": float, "ts": float, "source": str}}
        self._refs: Dict[str, dict] = {}
        self._midpoints: Dict[str, float] = {}

    def on_event(self, event: dict) -> List[SignalFire]:
        if event.get("event_type") not in ("book", "price_change"):
            return []

        market = event.get("market", "")
        mid = event.get("midpoint")
        if mid is not None:
            self._midpoints[market] = float(mid)

        return self._evaluate(market, event.get("asset_id", ""))

    def poll(self) -> List[SignalFire]:
        fires: List[SignalFire] = []
        for market in list(self._refs):
            if market in self._midpoints:
                fires.extend(self._evaluate(market, ""))
        return fires

    def set_reference(self, market: str, ref_price: float, source: str) -> None:
        self._refs[market] = {
            "ref_price": ref_price,
            "ts": time.time(),
            "source": source,
        }

    def _evaluate(self, market: str, token_id: str) -> List[SignalFire]:
        ref = self._refs.get(market)
        mid = self._midpoints.get(market)
        if ref is None or mid is None:
            return []

        delta_cents = (ref["ref_price"] - mid) * 100
        if abs(delta_cents) < self.config.cross_market_delta_cents:
            return []

        direction = Direction.BUY if delta_cents > 0 else Direction.SELL
        confidence = min(abs(delta_cents) / (self.config.cross_market_delta_cents * 3), 1.0)

        return [SignalFire(
            detector_name=self.name,
            market=market,
            token_id=token_id,
            direction=direction,
            confidence=confidence,
            meta={"delta_cents": delta_cents, "source": ref["source"]},
        )]

    def reset(self, market: str) -> None:
        self._refs.pop(market, None)
        self._midpoints.pop(market, None)

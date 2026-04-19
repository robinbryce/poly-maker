"""
News / oracle-delta detector.

Fires when a public data feed (NOAA, AP, CoinGecko, Reuters-style)
has moved and the market price has not yet adjusted by a configurable
delta.  The actual feed polling is done by ``feeds.oracle``; this
detector compares the latest feed value against the market midpoint.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

from detectors.base import BaseDetector, Direction, SignalFire
from grid.config import GridConfig


class NewsDetector(BaseDetector):
    name = "news"

    def __init__(
        self,
        config: GridConfig,
        hours_to_resolution: Optional[Callable[[str], Optional[float]]] = None,
    ):
        self.config = config
        # Populated by feeds.oracle via set_feed_value()
        # {market: {"value": float, "ts": float, "source": str}}
        self._feed_values: Dict[str, dict] = {}
        # Latest observed midpoints from the websocket layer
        self._midpoints: Dict[str, float] = {}
        # P4: optional hours-to-resolution callback used for
        # time-weighting.  Defaults to None — no weighting — when not
        # provided, which preserves pre-P4 behaviour for callers that
        # don't know the market's resolution time.
        self._hours_to_resolution = hours_to_resolution

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

    # ── time weighting ─────────────────────────────────────────────

    def _time_weight(self, market: str) -> float:
        """Return the confidence multiplier for a news fire.

        A news gap near resolution is a stronger bet than the same
        gap 18 hours out.  Within ``news_short_horizon_hours`` the
        weight is 1.0; beyond ``news_long_horizon_hours`` it floors
        at ``news_far_weight``; linear in between.  When we can't
        determine the horizon the midpoint of near and far weights
        is used so news still contributes something on unknown
        markets.
        """
        short_h = float(getattr(self.config, "news_short_horizon_hours", 0.0) or 0.0)
        long_h = float(getattr(self.config, "news_long_horizon_hours", 0.0) or 0.0)
        far = float(getattr(self.config, "news_far_weight", 1.0))
        # Defensive: if the config somehow has near > far, swap so the
        # interpolation stays well-defined.
        if long_h < short_h:
            short_h, long_h = long_h, short_h

        if self._hours_to_resolution is None:
            return 1.0
        hours_left = self._hours_to_resolution(market)
        if hours_left is None:
            return (1.0 + far) / 2.0
        if hours_left <= short_h:
            return 1.0
        if hours_left >= long_h or long_h <= short_h:
            return far
        t = (hours_left - short_h) / (long_h - short_h)
        return 1.0 - t * (1.0 - far)

    # ── core logic ───────────────────────────────────────────────

    def _evaluate(self, market: str, token_id: str) -> List[SignalFire]:
        feed = self._feed_values.get(market)
        mid = self._midpoints.get(market)
        if feed is None or mid is None:
            return []

        delta_cents = (feed["value"] - mid) * 100
        if abs(delta_cents) < self.config.news_delta_cents:
            return []

        direction = Direction.BUY if delta_cents > 0 else Direction.SELL
        base_conf = min(abs(delta_cents) / (self.config.news_delta_cents * 3), 1.0)
        time_weight = self._time_weight(market)
        confidence = max(0.0, min(1.0, base_conf * time_weight))

        return [SignalFire(
            detector_name=self.name,
            market=market,
            token_id=token_id,
            direction=direction,
            confidence=confidence,
            meta={
                "delta_cents": delta_cents,
                "source": feed["source"],
                "time_weight": time_weight,
            },
        )]

    def reset(self, market: str) -> None:
        self._feed_values.pop(market, None)
        self._midpoints.pop(market, None)

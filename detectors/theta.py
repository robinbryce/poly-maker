"""
Theta detector.

Fires when a market is within its resolution window and the price
suggests a strong lean that hasn't fully converged.  End dates are
injected by ``feeds.gamma``.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

from detectors.base import BaseDetector, Direction, SignalFire
from grid.config import GridConfig


class ThetaDetector(BaseDetector):
    name = "theta"

    def __init__(self, config: GridConfig):
        self.config = config
        # {market: end_date_epoch}  set by feeds.gamma
        self._end_dates: Dict[str, float] = {}
        self._midpoints: Dict[str, float] = {}

    def set_end_date(self, market: str, end_epoch: float) -> None:
        self._end_dates[market] = end_epoch

    def on_event(self, event: dict) -> List[SignalFire]:
        market = event.get("market", "")
        mid = event.get("midpoint")
        if mid is not None:
            self._midpoints[market] = float(mid)
        return []  # evaluation on poll

    def poll(self) -> List[SignalFire]:
        now = time.time()
        fires: List[SignalFire] = []
        for market, end_epoch in self._end_dates.items():
            hours_left = (end_epoch - now) / 3600
            if hours_left <= 0 or hours_left > self.config.theta_hours:
                continue

            mid = self._midpoints.get(market)
            if mid is None:
                continue

            # Strong lean: price > 0.75 or < 0.25 within the window
            if mid > 0.75:
                direction = Direction.BUY
                confidence = (mid - 0.75) / 0.25  # 0 at 0.75, 1 at 1.0
            elif mid < 0.25:
                direction = Direction.SELL
                confidence = (0.25 - mid) / 0.25  # 0 at 0.25, 1 at 0.0
            else:
                continue

            # Scale confidence by time pressure (closer = stronger)
            time_factor = 1.0 - (hours_left / self.config.theta_hours)
            confidence = min(confidence * (0.5 + 0.5 * time_factor), 1.0)

            fires.append(SignalFire(
                detector_name=self.name,
                market=market,
                token_id="",
                direction=direction,
                confidence=confidence,
                meta={"hours_left": hours_left, "midpoint": mid},
            ))
        return fires

    def reset(self, market: str) -> None:
        self._end_dates.pop(market, None)
        self._midpoints.pop(market, None)

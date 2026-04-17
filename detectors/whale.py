"""
Whale detector.

Fires when a wallet from the configured watchlist makes a trade in a
tracked market.  The trades are injected by ``feeds.whale`` via
``record_whale_trade()``.

This detector is a no-op if ``config.whale_wallets`` is empty.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import List

from detectors.base import BaseDetector, Direction, SignalFire
from grid.config import GridConfig


class WhaleDetector(BaseDetector):
    name = "whale"

    WINDOW = 300  # seconds to keep whale activity

    def __init__(self, config: GridConfig):
        self.config = config
        # {market: deque of (ts, wallet, side, size)}
        self._activity: dict[str, deque] = defaultdict(deque)

    def record_whale_trade(
        self, market: str, token_id: str, wallet: str, side: str, size: float
    ) -> List[SignalFire]:
        """Called by the whale feed poller when a watched wallet trades."""
        now = time.time()
        self._activity[market].append((now, wallet, side.upper(), size))
        self._prune(market, now)
        return self._evaluate(market, token_id, now)

    def poll(self) -> List[SignalFire]:
        """Re-evaluate all markets with recent whale activity."""
        now = time.time()
        fires: List[SignalFire] = []
        for market in list(self._activity):
            self._prune(market, now)
            if self._activity[market]:
                fires.extend(self._evaluate(market, "", now))
        return fires

    def _evaluate(self, market: str, token_id: str, now: float) -> List[SignalFire]:
        if not self.config.whale_wallets:
            return []

        cutoff = now - self.WINDOW
        buy_size = sum(
            sz for ts, _, side, sz in self._activity[market]
            if ts >= cutoff and side == "BUY"
        )
        sell_size = sum(
            sz for ts, _, side, sz in self._activity[market]
            if ts >= cutoff and side == "SELL"
        )
        total = buy_size + sell_size
        if total == 0:
            return []

        direction = Direction.BUY if buy_size > sell_size else Direction.SELL
        confidence = max(buy_size, sell_size) / total

        return [SignalFire(
            detector_name=self.name,
            market=market,
            token_id=token_id,
            direction=direction,
            confidence=confidence,
            meta={"buy_size": buy_size, "sell_size": sell_size},
        )]

    def reset(self, market: str) -> None:
        self._activity.pop(market, None)

    def _prune(self, market: str, now: float) -> None:
        q = self._activity[market]
        cutoff = now - self.WINDOW
        while q and q[0][0] < cutoff:
            q.popleft()

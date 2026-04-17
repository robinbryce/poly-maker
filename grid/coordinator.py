"""
Grid coordinator.

Ingests signal fires, maintains rolling per-market state, and decides
when to trigger an entry.  Enforces kill-switch, risk caps, and the
one-position-per-market rule.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

from detectors.base import Direction, SignalFire
from grid.config import GridConfig
from grid.state import GridState

if TYPE_CHECKING:
    from ledger.store import LedgerStore


class Coordinator:
    def __init__(
        self,
        config: GridConfig,
        on_entry: Callable[[str, str, Direction, float, dict], None],
        ledger: Optional["LedgerStore"] = None,
    ):
        self.config = config
        self._state = GridState(config.signal_staleness_secs)
        self._on_entry = on_entry  # (market, token_id, direction, confidence, meta)
        self._ledger = ledger

        # Tracks markets with an active grid-managed position or pending entry.
        self._open_markets: Set[str] = set()

        # Risk counters (reset daily by the main loop).
        self.daily_loss_usdc: float = 0.0
        self.consecutive_losses: int = 0

    # ── ingest fires ────────────────────────────────────────────────

    def ingest(self, fires: List[SignalFire]) -> None:
        """Called by the event bus with new signal fires."""
        for fire in fires:
            ms = self._state.get(fire.market)
            ms.update(fire)

            if self._ledger:
                self._ledger.log_signal_fire(fire)

        # Evaluate every market that received a fire this tick.
        seen: set = set()
        for fire in fires:
            if fire.market not in seen:
                seen.add(fire.market)
                self._evaluate(fire.market)

    # ── core evaluation ─────────────────────────────────────────────

    def _evaluate(self, market: str) -> None:
        if self.config.kill_switch:
            return

        if market in self._open_markets:
            return

        if len(self._open_markets) >= self.config.max_open_positions:
            return

        if self.daily_loss_usdc >= self.config.daily_loss_cap_usdc:
            return

        if self.consecutive_losses >= self.config.consecutive_loss_cap:
            return

        ms = self._state.get(market)
        active = ms.active_signals()
        if len(active) < self.config.min_signals:
            return

        direction, agreement = ms.dominant_direction()
        if direction is None or agreement < self.config.direction_threshold:
            return

        # All checks passed — trigger entry.
        avg_confidence = sum(f.confidence for f in active) / len(active)

        # Prefer token_id from the first fire that has one.
        token_id = ""
        for f in active:
            if f.token_id:
                token_id = f.token_id
                break

        meta = {
            "n_signals": len(active),
            "agreement": agreement,
            "detectors": [f.detector_name for f in active],
        }

        self._open_markets.add(market)

        if self._ledger:
            self._ledger.log_grid_fire(market, direction, len(active), agreement)

        self._on_entry(market, token_id, direction, avg_confidence, meta)

    # ── position lifecycle hooks ────────────────────────────────────

    def mark_closed(self, market: str, pnl_usdc: float) -> None:
        """Called when a grid-managed position is exited."""
        self._open_markets.discard(market)
        self._state.get(market).clear()

        if pnl_usdc < 0:
            self.daily_loss_usdc += abs(pnl_usdc)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    def reset_daily(self) -> None:
        self.daily_loss_usdc = 0.0
        self.consecutive_losses = 0

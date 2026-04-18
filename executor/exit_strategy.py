"""Pluggable exit-decision strategies for paper mode.

Replaces the hard-coded ``+10% TP / -5% SL`` rule in
:class:`executor.paper.PaperExecutor` so exit behaviour can be
reasoned about per market rather than per percent-of-entry.

Two strategies are shipped:

* :class:`CentThresholdStrategy` (default) \u2014 absolute-cent TP and SL
  with optional tightening when the market is close to resolution.
  This is a better fit for Polymarket probability markets where a
  10% move at mid=0.995 is an order of magnitude larger than a 10%
  move at mid=0.50.

* :class:`PercentageStrategy` \u2014 the legacy rule, retained so the
  paper log can be compared 1:1 against pre-P3 runs if needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from detectors.base import Direction

if TYPE_CHECKING:
    from grid.config import GridConfig


@dataclass
class ExitDecision:
    """Result of an :meth:`ExitStrategy.evaluate` call."""
    exit_now: bool
    reason: str                 # "tp", "sl", "hold", ...
    current_price: float
    pnl_pct: float              # signed, relative to entry price
    pnl_usdc: float             # signed USDC PnL at the decision point


class ExitStrategy(ABC):
    """Abstract base.  Implementations must be pure functions of the
    inputs \u2014 no I/O, no state mutation."""

    @abstractmethod
    def evaluate(
        self,
        position: dict,
        current_price: float,
        hours_to_resolution: Optional[float],
    ) -> ExitDecision:
        ...


def _pnl_from_move(
    entry_price: float,
    current_price: float,
    direction: str,
    size: float,
) -> tuple[float, float]:
    """Return ``(pnl_pct, pnl_usdc)`` for a move from entry to current.

    ``direction`` is the string stored on a paper position
    (``"BUY"`` or ``"SELL"``).
    """
    if entry_price <= 0:
        return 0.0, 0.0
    if direction == "BUY":
        pnl_pct = (current_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - current_price) / entry_price
    pnl_usdc = pnl_pct * size * entry_price
    return pnl_pct, pnl_usdc


class CentThresholdStrategy(ExitStrategy):
    """Exit on an absolute cent move, tightened near resolution.

    * Take profit when the favourable cent move exceeds
      ``config.exit_tp_cents``.
    * Stop loss when the adverse cent move exceeds
      ``config.exit_sl_cents``.
    * When ``hours_to_resolution`` is within
      ``config.exit_tighten_hours``, multiply both thresholds by
      ``config.exit_tighten_factor`` so the strategy banks PnL faster
      as the market approaches resolution.
    """

    def __init__(self, config: "GridConfig"):
        self.config = config

    def evaluate(
        self,
        position: dict,
        current_price: float,
        hours_to_resolution: Optional[float],
    ) -> ExitDecision:
        entry_price = float(position["entry_price"])
        direction = position["direction"]
        size = float(position["size"])

        # Favourable move in cents (positive = making money).
        if direction == "BUY":
            cents_move = (current_price - entry_price) * 100.0
        else:
            cents_move = (entry_price - current_price) * 100.0

        tp = float(self.config.exit_tp_cents)
        sl = float(self.config.exit_sl_cents)
        tighten_hours = float(self.config.exit_tighten_hours)
        tighten_factor = float(self.config.exit_tighten_factor)
        if (
            hours_to_resolution is not None
            and hours_to_resolution >= 0
            and hours_to_resolution <= tighten_hours
        ):
            tp *= tighten_factor
            sl *= tighten_factor

        pnl_pct, pnl_usdc = _pnl_from_move(
            entry_price, current_price, direction, size
        )

        if cents_move >= tp:
            return ExitDecision(
                exit_now=True, reason="tp",
                current_price=current_price,
                pnl_pct=pnl_pct, pnl_usdc=pnl_usdc,
            )
        if cents_move <= -sl:
            return ExitDecision(
                exit_now=True, reason="sl",
                current_price=current_price,
                pnl_pct=pnl_pct, pnl_usdc=pnl_usdc,
            )
        return ExitDecision(
            exit_now=False, reason="hold",
            current_price=current_price,
            pnl_pct=pnl_pct, pnl_usdc=pnl_usdc,
        )


class PercentageStrategy(ExitStrategy):
    """Legacy rule: exit on +``tp_pct`` / \u2212``sl_pct`` of entry.

    Retained so we can A/B compare behaviour if needed.
    """

    def __init__(self, tp_pct: float = 0.10, sl_pct: float = 0.05):
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct

    def evaluate(
        self,
        position: dict,
        current_price: float,
        hours_to_resolution: Optional[float],
    ) -> ExitDecision:
        entry_price = float(position["entry_price"])
        direction = position["direction"]
        size = float(position["size"])
        pnl_pct, pnl_usdc = _pnl_from_move(
            entry_price, current_price, direction, size
        )
        if pnl_pct >= self.tp_pct:
            reason = "tp"
            exit_now = True
        elif pnl_pct <= -self.sl_pct:
            reason = "sl"
            exit_now = True
        else:
            reason = "hold"
            exit_now = False
        return ExitDecision(
            exit_now=exit_now, reason=reason,
            current_price=current_price,
            pnl_pct=pnl_pct, pnl_usdc=pnl_usdc,
        )

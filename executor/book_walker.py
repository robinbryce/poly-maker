"""Order-book walk simulator.

Used by the paper executor to model slippage.  This module is
deliberately kept pure (no globals, no side effects) so it is easy
to unit-test and to reuse from report tooling.

The book shape is the one used throughout the rest of the codebase:

    book = {
        "bids": {price: size, ...},  # buyers
        "asks": {price: size, ...},  # sellers
    }

For a ``BUY`` fill we consume asks from the lowest price up; for a
``SELL`` fill we consume bids from the highest price down.  ``size``
is in tokens.  ``walk_book`` returns a :class:`FillResult` summarising
the walk (VWAP, slippage against best price, levels consumed) or
``None`` when the relevant side of the book is empty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from detectors.base import Direction


@dataclass
class FillResult:
    """Summary of a simulated book walk."""
    requested_size: float
    filled_size: float
    vwap: float
    best_price: float
    worst_price: float
    levels_consumed: int
    slippage_bps: float

    @property
    def is_full_fill(self) -> bool:
        # Tolerate float fuzz on size comparisons.
        return self.filled_size >= self.requested_size - 1e-9


def walk_book(
    book: Dict[str, Dict[float, float]],
    direction: Direction,
    size: float,
) -> Optional[FillResult]:
    """Simulate taking ``size`` tokens against ``book`` on ``direction``.

    Returns ``None`` when the requested side of the book is empty or
    ``size`` is non-positive.  Returns a partial :class:`FillResult`
    when the book exists but can't fill the full ``size``.
    """
    if size <= 0:
        return None

    if direction == Direction.BUY:
        side = book.get("asks") or {}
        levels = sorted(
            ((float(p), float(s)) for p, s in side.items() if float(s) > 0),
            key=lambda x: x[0],
        )
    else:
        side = book.get("bids") or {}
        levels = sorted(
            ((float(p), float(s)) for p, s in side.items() if float(s) > 0),
            key=lambda x: x[0],
            reverse=True,
        )

    if not levels:
        return None

    best_price = levels[0][0]
    remaining = size
    filled = 0.0
    cost = 0.0
    worst_price = best_price
    levels_consumed = 0

    for price, avail in levels:
        if remaining <= 0:
            break
        take = min(avail, remaining)
        cost += take * price
        filled += take
        remaining -= take
        worst_price = price
        levels_consumed += 1

    if filled <= 0:
        return None

    vwap = cost / filled
    # Slippage is the distance from the best price, in basis points
    # (1 bp = 0.01%).  Direction-agnostic: we always measure the
    # adverse move so BUY slippage_bps > 0 when vwap > best, SELL > 0
    # when vwap < best.
    if best_price > 0:
        if direction == Direction.BUY:
            slip_bps = max(0.0, (vwap - best_price) / best_price * 10_000.0)
        else:
            slip_bps = max(0.0, (best_price - vwap) / best_price * 10_000.0)
    else:
        slip_bps = 0.0

    return FillResult(
        requested_size=size,
        filled_size=filled,
        vwap=vwap,
        best_price=best_price,
        worst_price=worst_price,
        levels_consumed=levels_consumed,
        slippage_bps=slip_bps,
    )

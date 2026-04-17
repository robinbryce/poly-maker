"""
Paper executor.

Simulates fills against the last observed order book and logs every
decision to the ledger.  No real orders are placed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from detectors.base import Direction

if TYPE_CHECKING:
    from grid.config import GridConfig
    from ledger.store import LedgerStore

import poly_data.global_state as global_state


class PaperExecutor:
    def __init__(self, config: "GridConfig", ledger: "LedgerStore"):
        self.config = config
        self._ledger = ledger
        # {market: {"token_id", "direction", "entry_price", "size"}}
        self._positions: dict[str, dict] = {}

    def enter(
        self, market: str, token_id: str, direction: Direction,
        confidence: float, meta: dict
    ) -> None:
        """Simulate an entry at the current best price."""
        price = self._best_price(market, direction)
        if price is None:
            print(f"[paper] no book for {market}, skipping entry")
            return

        size = min(self.config.max_entry_usdc / price, self.config.max_entry_usdc)
        self._positions[market] = {
            "token_id": token_id,
            "direction": direction.value,
            "entry_price": price,
            "size": size,
        }

        self._ledger.log_entry(
            market, token_id, direction.value, size, price, "paper", meta
        )
        print(f"[paper] ENTER {direction.value} {market[:12]}… "
              f"size={size:.2f} price={price:.4f} signals={meta.get('detectors')}")

    def check_exits(self) -> list[tuple[str, float]]:
        """Check open paper positions for exit conditions.  Returns list of (market, pnl)."""
        closed: list[tuple[str, float]] = []
        for market in list(self._positions):
            pos = self._positions[market]
            dir_ = pos["direction"]
            exit_dir = Direction.SELL if dir_ == "BUY" else Direction.BUY
            price = self._best_price(market, exit_dir)
            if price is None:
                continue

            # Simple exit: take profit at 10% or stop loss at -5%
            entry = pos["entry_price"]
            if dir_ == "BUY":
                pnl_pct = (price - entry) / entry
            else:
                pnl_pct = (entry - price) / entry

            if pnl_pct >= 0.10 or pnl_pct <= -0.05:
                pnl_usdc = pnl_pct * pos["size"] * entry
                self._ledger.log_exit(
                    market, pos["token_id"], dir_, pos["size"], price, pnl_usdc, "paper"
                )
                tag = "TP" if pnl_pct >= 0 else "SL"
                print(f"[paper] EXIT {tag} {market[:12]}… pnl={pnl_usdc:.2f} USDC")
                del self._positions[market]
                closed.append((market, pnl_usdc))

        return closed

    @property
    def open_count(self) -> int:
        return len(self._positions)

    def _best_price(self, market: str, direction: Direction) -> float | None:
        book = global_state.all_data.get(market)
        if not book:
            return None
        if direction == Direction.BUY:
            asks = book.get("asks")
            if asks and len(asks) > 0:
                return float(min(asks.keys()))
        else:
            bids = book.get("bids")
            if bids and len(bids) > 0:
                return float(max(bids.keys()))
        return None

"""
Live executor.

Wraps the existing ``PolymarketClient`` to place real orders.
**Disabled by default** — only active when ``config.mode == "live"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from detectors.base import Direction

if TYPE_CHECKING:
    from grid.config import GridConfig
    from ledger.store import LedgerStore

import poly_data.global_state as global_state


class LiveExecutor:
    def __init__(self, config: "GridConfig", ledger: "LedgerStore"):
        self.config = config
        self._ledger = ledger

    def enter(
        self, market: str, token_id: str, direction: Direction,
        confidence: float, meta: dict, correlation_id: str,
    ) -> None:
        if self.config.mode != "live":
            print("[live] mode is not 'live', refusing to place order")
            self._ledger.log_block("live_mode_off", market)
            return
        if self.config.kill_switch:
            print("[live] kill switch active, refusing to place order")
            self._ledger.log_block("kill_switch", market)
            return
        if not getattr(self.config, "live_armed", False):
            print("[live] live_armed is False — refusing to place order")
            self._ledger.log_block("not_armed", market)
            return

        client = global_state.client
        if client is None:
            print("[live] no client initialised")
            self._ledger.log_block("no_client", market)
            return

        action = "BUY" if direction == Direction.BUY else "SELL"
        book = global_state.all_data.get(market)
        if not book:
            print(f"[live] no book for {market}")
            self._ledger.log_block("no_book", market)
            return

        if direction == Direction.BUY:
            asks = book.get("asks")
            if not asks:
                return
            price = float(min(asks.keys()))
        else:
            bids = book.get("bids")
            if not bids:
                return
            price = float(max(bids.keys()))

        size = min(self.config.max_entry_usdc / price, self.config.max_entry_usdc)
        resp = client.create_order(token_id, action, price, size, neg_risk=False)
        print(f"[live] ENTER {action} {market[:12]}… size={size:.2f} "
              f"price={price:.4f} resp={resp} cid={correlation_id[:8]}…")
        self._ledger.log_entry(
            market, token_id, action, size, price, "live", meta, correlation_id,
        )

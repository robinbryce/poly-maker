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
                print(f"[live] no asks for {market}")
                self._ledger.log_block("no_asks", market)
                return
            price = float(min(asks.keys()))
        else:
            bids = book.get("bids")
            if not bids:
                print(f"[live] no bids for {market}")
                self._ledger.log_block("no_bids", market)
                return
            price = float(max(bids.keys()))

        # Pre-trade book-drift guard: refuse if the book has moved
        # more than book_drift_bps since the fire moment.
        fire_price = meta.get("fire_price") if isinstance(meta, dict) else None
        drift_limit = float(getattr(self.config, "book_drift_bps", 0.0) or 0.0)
        if fire_price and drift_limit > 0:
            try:
                fp = float(fire_price)
            except (TypeError, ValueError):
                fp = 0.0
            if fp > 0:
                drift_bps = abs(price - fp) / fp * 10_000.0
                if drift_bps > drift_limit:
                    print(f"[live] book drift {drift_bps:.1f}bps > "
                          f"{drift_limit:.1f}bps for {market[:12]}… "
                          f"refusing order")
                    self._ledger.log_block(
                        "book_drift_exceeded", market,
                        {"drift_bps": drift_bps,
                         "limit_bps": drift_limit,
                         "fire_price": fp,
                         "price_now": price,
                         "correlation_id": correlation_id},
                    )
                    return

        size = min(self.config.max_entry_usdc / price, self.config.max_entry_usdc)
        resp = client.create_order(token_id, action, price, size, neg_risk=False)
        print(f"[live] ENTER {action} {market[:12]}… size={size:.2f} "
              f"price={price:.4f} resp={resp} cid={correlation_id[:8]}…")
        self._ledger.log_entry(
            market, token_id, action, size, price, "live", meta, correlation_id,
        )

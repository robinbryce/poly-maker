"""
Live executor.

Wraps the existing ``PolymarketClient`` to place real orders.
**Disabled by default** — only active when ``config.mode == "live"``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from detectors.base import Direction
from grid.metrics import counters

if TYPE_CHECKING:
    from grid.config import GridConfig
    from ledger.store import LedgerStore

import poly_data.global_state as global_state

logger = logging.getLogger(__name__)


class LiveExecutor:
    def __init__(self, config: "GridConfig", ledger: "LedgerStore"):
        self.config = config
        self._ledger = ledger

    def enter(
        self, market: str, token_id: str, direction: Direction,
        confidence: float, meta: dict, correlation_id: str,
    ) -> None:
        if self.config.mode != "live":
            logger.warning("mode is not 'live', refusing to place order",
                           extra={"market": market})
            counters.labelled_incr("live.entries", {"outcome": "mode_off"})
            self._ledger.log_block("live_mode_off", market)
            return
        if self.config.kill_switch:
            logger.warning("kill switch active, refusing to place order",
                           extra={"market": market})
            counters.labelled_incr("live.entries", {"outcome": "kill_switch"})
            self._ledger.log_block("kill_switch", market)
            return
        if not getattr(self.config, "live_armed", False):
            logger.warning("live_armed is False, refusing to place order",
                           extra={"market": market})
            counters.labelled_incr("live.entries", {"outcome": "not_armed"})
            self._ledger.log_block("not_armed", market)
            return

        client = global_state.client
        if client is None:
            logger.error("no client initialised", extra={"market": market})
            counters.labelled_incr("live.entries", {"outcome": "no_client"})
            self._ledger.log_block("no_client", market)
            return

        action = "BUY" if direction == Direction.BUY else "SELL"
        book = global_state.all_data.get(market)
        if not book:
            logger.info("no book for %s", market, extra={"market": market})
            counters.labelled_incr("live.entries", {"outcome": "no_book"})
            self._ledger.log_block("no_book", market)
            return

        if direction == Direction.BUY:
            asks = book.get("asks")
            if not asks:
                logger.info("no asks for %s", market, extra={"market": market})
                counters.labelled_incr("live.entries", {"outcome": "no_asks"})
                self._ledger.log_block("no_asks", market)
                return
            price = float(min(asks.keys()))
        else:
            bids = book.get("bids")
            if not bids:
                logger.info("no bids for %s", market, extra={"market": market})
                counters.labelled_incr("live.entries", {"outcome": "no_bids"})
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
                    logger.warning(
                        "book drift %.1fbps > %.1fbps for %s, refusing order",
                        drift_bps, drift_limit, market[:12],
                        extra={
                            "market": market,
                            "drift_bps": drift_bps,
                            "limit_bps": drift_limit,
                            "correlation_id": correlation_id,
                        },
                    )
                    counters.labelled_incr(
                        "live.entries",
                        {"outcome": "book_drift_exceeded"},
                    )
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
        logger.info(
            "ENTER %s %s… size=%.2f price=%.4f resp=%s cid=%s…",
            action, market[:12], size, price, resp, correlation_id[:8],
            extra={
                "event": "live_enter",
                "market": market,
                "direction": action,
                "size": size,
                "price": price,
                "correlation_id": correlation_id,
            },
        )
        counters.labelled_incr("live.entries", {"outcome": "ok"})
        self._ledger.log_entry(
            market, token_id, action, size, price, "live", meta, correlation_id,
        )

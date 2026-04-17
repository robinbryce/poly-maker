"""Paper executor with correlation-id propagation, asyncio lock, and
snapshot/restore of open positions.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List, Tuple

from detectors.base import Direction

if TYPE_CHECKING:
    from grid.config import GridConfig
    from ledger.store import LedgerStore

import poly_data.global_state as global_state


class PaperExecutor:
    TAKE_PROFIT_PCT = 0.10
    STOP_LOSS_PCT = -0.05

    def __init__(self, config: "GridConfig", ledger: "LedgerStore"):
        self.config = config
        self._ledger = ledger
        self._positions: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    def snapshot(self) -> dict:
        return {"positions": dict(self._positions)}

    def restore(self, data: dict) -> None:
        self._positions = dict(data.get("positions", {}))

    async def enter(
        self, market: str, token_id: str, direction: Direction,
        confidence: float, meta: dict, correlation_id: str,
    ) -> None:
        async with self._lock:
            self._enter_inner(market, token_id, direction, meta, correlation_id)

    def enter_sync(
        self, market: str, token_id: str, direction: Direction,
        confidence: float, meta: dict, correlation_id: str,
    ) -> None:
        self._enter_inner(market, token_id, direction, meta, correlation_id)

    def _enter_inner(
        self, market: str, token_id: str, direction: Direction,
        meta: dict, correlation_id: str,
    ) -> None:
        price = self._best_price(market, direction)
        if price is None:
            print(f"[paper] no book for {market}, skipping entry")
            return
        size = min(self.config.max_entry_usdc / price, self.config.max_entry_usdc)
        self._positions[market] = {
            "correlation_id": correlation_id,
            "token_id": token_id,
            "direction": direction.value,
            "entry_price": price,
            "size": size,
        }
        self._ledger.log_entry(
            market, token_id, direction.value, size, price, "paper", meta,
            correlation_id,
        )
        print(f"[paper] ENTER {direction.value} {market[:12]}… "
              f"size={size:.2f} price={price:.4f} "
              f"signals={meta.get('detectors')} cid={correlation_id[:8]}…")

    async def check_exits(self) -> List[Tuple[str, float]]:
        async with self._lock:
            return self._check_exits_inner()

    def check_exits_sync(self) -> List[Tuple[str, float]]:
        return self._check_exits_inner()

    def _check_exits_inner(self) -> List[Tuple[str, float]]:
        closed: List[Tuple[str, float]] = []
        for market in list(self._positions):
            pos = self._positions[market]
            dir_ = pos["direction"]
            exit_dir = Direction.SELL if dir_ == "BUY" else Direction.BUY
            price = self._best_price(market, exit_dir)
            if price is None:
                continue
            entry = pos["entry_price"]
            if dir_ == "BUY":
                pnl_pct = (price - entry) / entry
            else:
                pnl_pct = (entry - price) / entry
            if pnl_pct >= self.TAKE_PROFIT_PCT or pnl_pct <= self.STOP_LOSS_PCT:
                pnl_usdc = pnl_pct * pos["size"] * entry
                self._ledger.log_exit(
                    market, pos["token_id"], dir_, pos["size"], price,
                    pnl_usdc, "paper", pos["correlation_id"],
                )
                tag = "TP" if pnl_pct >= 0 else "SL"
                print(f"[paper] EXIT {tag} {market[:12]}… pnl={pnl_usdc:.2f} USDC "
                      f"cid={pos['correlation_id'][:8]}…")
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

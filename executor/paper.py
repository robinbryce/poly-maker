"""Paper executor with correlation-id propagation, asyncio lock, and
snapshot/restore of open positions.

P1.1: ``enter`` now returns an ``ExecutionResult`` with ``ok=True``
on success or ``ok=False`` plus a ``reason`` on decline.  The
coordinator uses that to avoid orphaning markets in ``_open_markets``
when the executor silently declines (e.g. missing book).

P3:
* Entries run through :func:`executor.book_walker.walk_book` so the
  simulated fill honours the observable book.  Walks that exceed
  ``config.max_slippage_bps`` are refused with
  ``slippage_exceeded``.
* The live-executor pre-trade drift guard introduced in P2 is
  ported to paper: if ``meta["fire_price"]`` is present and the
  post-walk VWAP drifts further than ``config.book_drift_bps``
  from it, the entry is refused with ``book_drift_exceeded``.
* Exits delegate to a pluggable
  :class:`executor.exit_strategy.ExitStrategy` —
  ``CentThresholdStrategy`` by default, which replaces the hard-
  coded ±10%/−5% rule.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

from detectors.base import Direction
from executor.book_walker import FillResult, walk_book
from executor.exit_strategy import CentThresholdStrategy, ExitStrategy

if TYPE_CHECKING:
    from grid.config import GridConfig
    from ledger.store import LedgerStore

import poly_data.global_state as global_state


@dataclass
class ExecutionResult:
    ok: bool
    reason: Optional[str] = None
    meta: Optional[dict] = None

    @classmethod
    def success(cls, **meta) -> "ExecutionResult":
        return cls(ok=True, meta=meta or None)

    @classmethod
    def declined(cls, reason: str, **meta) -> "ExecutionResult":
        return cls(ok=False, reason=reason, meta=meta or None)


class PaperExecutor:
    def __init__(
        self,
        config: "GridConfig",
        ledger: "LedgerStore",
        exit_strategy: Optional[ExitStrategy] = None,
        hours_to_resolution: Optional[Callable[[str], Optional[float]]] = None,
    ):
        self.config = config
        self._ledger = ledger
        self._positions: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._exit_strategy: ExitStrategy = (
            exit_strategy or CentThresholdStrategy(config)
        )
        self._hours_to_resolution = (
            hours_to_resolution or (lambda _m: None)
        )

    def snapshot(self) -> dict:
        return {"positions": dict(self._positions)}

    def restore(self, data: dict) -> None:
        self._positions = dict(data.get("positions", {}))

    async def enter(
        self, market: str, token_id: str, direction: Direction,
        confidence: float, meta: dict, correlation_id: str,
    ) -> ExecutionResult:
        async with self._lock:
            return self._enter_inner(market, token_id, direction, meta, correlation_id)

    def enter_sync(
        self, market: str, token_id: str, direction: Direction,
        confidence: float, meta: dict, correlation_id: str,
    ) -> ExecutionResult:
        return self._enter_inner(market, token_id, direction, meta, correlation_id)

    def _enter_inner(
        self, market: str, token_id: str, direction: Direction,
        meta: dict, correlation_id: str,
    ) -> ExecutionResult:
        if market in self._positions:
            return ExecutionResult.declined("already_held")

        book = global_state.all_data.get(market)
        if not book:
            print(f"[paper] no book for {market}, skipping entry")
            self._ledger.log_block("no_book", market,
                                   {"correlation_id": correlation_id})
            return ExecutionResult.declined("no_book")

        best = self._best_price(market, direction)
        if best is None or best <= 0 or best >= 1:
            self._ledger.log_block(
                "invalid_price", market,
                {"price": best, "correlation_id": correlation_id},
            )
            return ExecutionResult.declined("invalid_price", price=best)

        requested_size = min(
            self.config.max_entry_usdc / best,
            self.config.max_entry_usdc,
        )
        if requested_size <= 0:
            return ExecutionResult.declined("zero_size", price=best)

        fill = walk_book(book, direction, requested_size)
        if fill is None or fill.filled_size <= 0:
            self._ledger.log_block(
                "no_liquidity", market,
                {"correlation_id": correlation_id},
            )
            return ExecutionResult.declined("no_liquidity")

        slip_limit = float(self.config.max_slippage_bps)
        if slip_limit > 0 and fill.slippage_bps > slip_limit:
            print(f"[paper] slippage {fill.slippage_bps:.1f}bps > "
                  f"{slip_limit:.1f}bps for {market[:12]}… refusing")
            self._ledger.log_block(
                "slippage_exceeded", market,
                {"slippage_bps": fill.slippage_bps,
                 "limit_bps": slip_limit,
                 "best_price": fill.best_price,
                 "vwap": fill.vwap,
                 "requested_size": fill.requested_size,
                 "filled_size": fill.filled_size,
                 "correlation_id": correlation_id},
            )
            return ExecutionResult.declined(
                "slippage_exceeded",
                slippage_bps=fill.slippage_bps,
                limit_bps=slip_limit,
            )

        # Pre-trade book-drift guard (paper mirror of the live guard).
        fire_price = meta.get("fire_price") if isinstance(meta, dict) else None
        drift_limit = float(getattr(self.config, "book_drift_bps", 0.0) or 0.0)
        if fire_price and drift_limit > 0:
            try:
                fp = float(fire_price)
            except (TypeError, ValueError):
                fp = 0.0
            if fp > 0:
                drift_bps = abs(fill.vwap - fp) / fp * 10_000.0
                if drift_bps > drift_limit:
                    print(f"[paper] book drift {drift_bps:.1f}bps > "
                          f"{drift_limit:.1f}bps for {market[:12]}… "
                          f"refusing")
                    self._ledger.log_block(
                        "book_drift_exceeded", market,
                        {"drift_bps": drift_bps,
                         "limit_bps": drift_limit,
                         "fire_price": fp,
                         "vwap": fill.vwap,
                         "correlation_id": correlation_id},
                    )
                    return ExecutionResult.declined(
                        "book_drift_exceeded",
                        drift_bps=drift_bps, limit_bps=drift_limit,
                    )

        # Record the simulated fill at VWAP and whatever size the book
        # could actually absorb inside the slippage budget.
        filled_size = fill.filled_size
        vwap = fill.vwap

        # Enrich meta with fill diagnostics so the ledger captures the
        # slippage cost for this paper trade.
        fill_meta = dict(meta) if isinstance(meta, dict) else {}
        fill_meta.update({
            "best_price": fill.best_price,
            "vwap": vwap,
            "slippage_bps": fill.slippage_bps,
            "levels_consumed": fill.levels_consumed,
            "requested_size": fill.requested_size,
        })

        self._positions[market] = {
            "correlation_id": correlation_id,
            "token_id": token_id,
            "direction": direction.value,
            "entry_price": vwap,
            "size": filled_size,
        }
        self._ledger.log_entry(
            market, token_id, direction.value, filled_size, vwap, "paper",
            fill_meta, correlation_id,
        )
        print(f"[paper] ENTER {direction.value} {market[:12]}… "
              f"size={filled_size:.2f} vwap={vwap:.4f} "
              f"slip={fill.slippage_bps:.1f}bps "
              f"signals={fill_meta.get('detectors')} "
              f"cid={correlation_id[:8]}…")
        return ExecutionResult.success(
            price=vwap, size=filled_size,
            slippage_bps=fill.slippage_bps,
        )

    def has_position(self, market: str) -> bool:
        return market in self._positions

    def open_markets(self) -> set:
        return set(self._positions.keys())

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
            hours_left = self._hours_to_resolution(market)
            decision = self._exit_strategy.evaluate(
                pos, float(price), hours_left,
            )
            if not decision.exit_now:
                continue
            self._ledger.log_exit(
                market, pos["token_id"], dir_, pos["size"], price,
                decision.pnl_usdc, "paper", pos["correlation_id"],
            )
            tag = decision.reason.upper()
            print(f"[paper] EXIT {tag} {market[:12]}… "
                  f"pnl={decision.pnl_usdc:.2f} USDC "
                  f"cid={pos['correlation_id'][:8]}…")
            del self._positions[market]
            closed.append((market, decision.pnl_usdc))
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

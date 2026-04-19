"""Grid coordinator with asyncio lock, audit log, correlation IDs,
UTC daily reset, and per-category concentration caps (P1).

All mutating paths acquire a single ``asyncio.Lock``.  A ``_sync``
variant is retained for tests that don't run an event loop.
"""

from __future__ import annotations

import asyncio
import datetime
from collections import Counter
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

from detectors.base import Direction, SignalFire
from grid.config import GridConfig
from grid.metrics import counters
from grid.state import GridState
from ledger.store import new_correlation_id

if TYPE_CHECKING:
    from ledger.store import LedgerStore


class Coordinator:
    def __init__(
        self,
        config: GridConfig,
        on_entry,  # (market, token_id, direction, confidence, meta, correlation_id)
        ledger: Optional["LedgerStore"] = None,
        category_of: Optional[Callable[[str], Optional[str]]] = None,
        get_midpoint: Optional[Callable[[str], Optional[float]]] = None,
    ):
        self.config = config
        self._state = GridState(config.signal_staleness_secs)
        self._on_entry = on_entry
        self._ledger = ledger
        self._category_of = category_of or (lambda _m: None)
        self._get_midpoint = get_midpoint or (lambda _m: None)

        self._open_markets: Set[str] = set()
        self._open_categories: Dict[str, str] = {}

        self.daily_loss_usdc: float = 0.0
        self.consecutive_losses: int = 0
        self._last_reset_utc_date = self._today_utc()

        # P4: fire-quality counters.  fires_by_detector counts every
        # signal fire ingested; grid_contributions_by_detector counts
        # the times each detector appeared in a grid fire's active set.
        # fire_quality_report() divides them for an operator-visible
        # contribution rate.
        self.fires_by_detector: Counter = Counter()
        self.grid_contributions_by_detector: Counter = Counter()

        # P6: lazy lock allocation.  Py3.9's ``asyncio.Lock()`` binds
        # to the running loop at construction time, which means
        # constructing a ``Coordinator`` outside an async context (or
        # after ``asyncio.run()`` tore one down) raised.  Defer until
        # the first async method actually needs it.
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    # snapshot / restore --------------------------------------------

    def snapshot(self) -> dict:
        return {
            "open_markets": sorted(self._open_markets),
            "open_categories": dict(self._open_categories),
            "daily_loss_usdc": self.daily_loss_usdc,
            "consecutive_losses": self.consecutive_losses,
            "last_reset_utc_date": self._last_reset_utc_date,
            "fires_by_detector": dict(self.fires_by_detector),
            "grid_contributions_by_detector":
                dict(self.grid_contributions_by_detector),
        }

    def restore(self, data: dict) -> None:
        self._open_markets = set(data.get("open_markets", []))
        self._open_categories = dict(data.get("open_categories", {}))
        self.daily_loss_usdc = float(data.get("daily_loss_usdc", 0.0))
        self.consecutive_losses = int(data.get("consecutive_losses", 0))
        self._last_reset_utc_date = data.get(
            "last_reset_utc_date", self._today_utc()
        )
        self.fires_by_detector = Counter(data.get("fires_by_detector", {}))
        self.grid_contributions_by_detector = Counter(
            data.get("grid_contributions_by_detector", {})
        )

    # fire quality --------------------------------------------------

    def fire_quality_report(self) -> Dict[str, Dict[str, float]]:
        """Per-detector fires, grid contributions and contribution
        rate.  Rate is ``contributions / fires`` when ``fires > 0``.
        Detectors with zero fires are included so the operator can
        see that they are silent.
        """
        names = set(self.fires_by_detector) | set(
            self.grid_contributions_by_detector
        )
        report: Dict[str, Dict[str, float]] = {}
        for name in sorted(names):
            fires = int(self.fires_by_detector.get(name, 0))
            contrib = int(self.grid_contributions_by_detector.get(name, 0))
            rate = (contrib / fires) if fires > 0 else 0.0
            report[name] = {
                "fires": fires,
                "contributions": contrib,
                "quality_pct": rate * 100.0,
            }
        return report

    # ingest --------------------------------------------------------

    async def ingest(self, fires: List[SignalFire]) -> None:
        async with self._get_lock():
            self._maybe_daily_reset()
            for fire in fires:
                self.fires_by_detector[fire.detector_name] += 1
                counters.labelled_incr(
                    "grid.fires", {"detector": fire.detector_name}
                )
                self._state.get(fire.market).update(fire)
                if self._ledger:
                    self._ledger.log_signal_fire(fire)
            seen: set = set()
            for fire in fires:
                if fire.market in seen:
                    continue
                seen.add(fire.market)
                await self._evaluate(fire.market)

    def ingest_sync(self, fires: List[SignalFire]) -> None:
        self._maybe_daily_reset()
        for fire in fires:
            self.fires_by_detector[fire.detector_name] += 1
            counters.labelled_incr(
                "grid.fires", {"detector": fire.detector_name}
            )
            self._state.get(fire.market).update(fire)
            if self._ledger:
                self._ledger.log_signal_fire(fire)
        seen: set = set()
        for fire in fires:
            if fire.market in seen:
                continue
            seen.add(fire.market)
            self._evaluate_sync(fire.market)

    # evaluate ------------------------------------------------------

    async def _evaluate(self, market: str) -> None:
        d = self._decide(market)
        if d is None:
            return
        direction, _, active, token_id, cid, category, avg_conf, meta, \
            agreement = d
        counters.incr("grid.grid_fires")
        if self._ledger:
            self._ledger.log_grid_fire(
                market, direction, len(active), agreement, cid,
            )
        result = self._on_entry(
            market, token_id, direction, avg_conf, meta, cid,
        )
        if asyncio.iscoroutine(result):
            result = await result
        self._commit_or_decline(market, category, cid, result)

    def _evaluate_sync(self, market: str) -> None:
        d = self._decide(market)
        if d is None:
            return
        direction, _, active, token_id, cid, category, avg_conf, meta, \
            agreement = d
        counters.incr("grid.grid_fires")
        if self._ledger:
            self._ledger.log_grid_fire(
                market, direction, len(active), agreement, cid,
            )
        result = self._on_entry(
            market, token_id, direction, avg_conf, meta, cid,
        )
        self._commit_or_decline(market, category, cid, result)

    def _decide(self, market: str):
        if self.config.kill_switch:
            self._audit("kill_switch", market)
            return None
        if market in self._open_markets:
            self._audit("already_open", market)
            return None
        if len(self._open_markets) >= self.config.max_open_positions:
            self._audit("max_open_positions", market,
                        {"limit": self.config.max_open_positions})
            return None
        if self.daily_loss_usdc >= self.config.daily_loss_cap_usdc:
            self._audit("daily_loss_cap", market,
                        {"loss_usdc": self.daily_loss_usdc})
            return None
        if self.consecutive_losses >= self.config.consecutive_loss_cap:
            self._audit("consecutive_loss_cap", market,
                        {"losses": self.consecutive_losses})
            return None

        ms = self._state.get(market)
        active = ms.active_signals()
        if len(active) < self.config.min_signals:
            return None

        direction, agreement = ms.dominant_direction()
        if direction is None or agreement < self.config.direction_threshold:
            self._audit("direction_threshold", market,
                        {"agreement": agreement})
            return None

        category = self._category_of(market)
        max_cat = getattr(self.config, "max_open_per_category", 0)
        if category and max_cat > 0:
            in_cat = sum(1 for c in self._open_categories.values()
                         if c == category)
            if in_cat >= max_cat:
                self._audit("category_concentration", market,
                            {"category": category, "in_category": in_cat})
                return None

        avg_confidence = sum(f.confidence for f in active) / len(active)
        token_id = ""
        for f in active:
            if f.token_id:
                token_id = f.token_id
                break
        cid = new_correlation_id()
        meta = {
            "n_signals": len(active),
            "agreement": agreement,
            "detectors": [f.detector_name for f in active],
            "category": category or "",
        }
        # Capture the midpoint at fire time so the live executor can
        # detect drift between the fire moment and the order moment.
        fire_price = self._get_midpoint(market)
        if fire_price is not None:
            meta["fire_price"] = float(fire_price)
        # P4: credit every active detector with this grid fire so
        # fire_quality_report() can separate contributors from noise.
        for f in active:
            self.grid_contributions_by_detector[f.detector_name] += 1
        return (direction, agreement, active, token_id, cid, category,
                avg_confidence, meta, agreement)

    def _commit_or_decline(self, market, category, cid, result) -> None:
        """Commit market to _open_markets only if the executor returned
        success.  If it declined, audit-log the reason and do not track
        the market as open."""
        ok = True
        reason = None
        meta = None
        if result is None:
            # Tests that use a no-op lambda as on_entry return None;
            # treat that as success.
            ok = True
        elif hasattr(result, "ok"):
            ok = bool(result.ok)
            reason = getattr(result, "reason", None)
            meta = getattr(result, "meta", None)

        if ok:
            self._open_markets.add(market)
            if category:
                self._open_categories[market] = category
        else:
            self._audit("executor_declined", market,
                        {"reason": reason, "correlation_id": cid,
                         "meta": meta})

    # lifecycle -----------------------------------------------------

    async def mark_closed(self, market: str, pnl_usdc: float) -> None:
        async with self._get_lock():
            self._mark_closed_inner(market, pnl_usdc)

    def mark_closed_sync(self, market: str, pnl_usdc: float) -> None:
        self._mark_closed_inner(market, pnl_usdc)

    def _mark_closed_inner(self, market: str, pnl_usdc: float) -> None:
        self._open_markets.discard(market)
        self._open_categories.pop(market, None)
        self._state.get(market).clear()
        if pnl_usdc < 0:
            self.daily_loss_usdc += abs(pnl_usdc)
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

    # daily reset ---------------------------------------------------

    def _maybe_daily_reset(self) -> None:
        today = self._today_utc()
        if today != self._last_reset_utc_date:
            self._last_reset_utc_date = today
            self.daily_loss_usdc = 0.0
            self.consecutive_losses = 0
            if self._ledger:
                self._ledger.log_audit("daily_reset", utc_date=today)

    @staticmethod
    def _today_utc() -> str:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

    def reset_daily(self) -> None:
        self.daily_loss_usdc = 0.0
        self.consecutive_losses = 0
        self._last_reset_utc_date = self._today_utc()

    # reconciliation ------------------------------------------------

    async def reconcile_open_markets(self, actually_open) -> int:
        async with self._get_lock():
            return self._reconcile_inner(actually_open)

    def reconcile_open_markets_sync(self, actually_open) -> int:
        return self._reconcile_inner(actually_open)

    def _reconcile_inner(self, actually_open) -> int:
        actual = set(actually_open)
        orphans = [m for m in list(self._open_markets) if m not in actual]
        for m in orphans:
            self._open_markets.discard(m)
            self._open_categories.pop(m, None)
            self._state.get(m).clear()
            if self._ledger:
                self._ledger.log_audit("orphan_evicted", market=m)
        return len(orphans)

    @property
    def open_markets(self) -> set:
        return set(self._open_markets)

    # audit ---------------------------------------------------------

    def _audit(self, reason: str, market: str, meta: Optional[dict] = None) -> None:
        counters.labelled_incr("grid.blocks", {"reason": reason})
        if self._ledger:
            self._ledger.log_block(reason, market, meta)

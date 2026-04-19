"""Safety-focused guardrail tests.

These tests assert invariants that must hold for the grid to be safe:

  * Paper mode must never place a real order.
  * Live mode must refuse to enter when mode != 'live' or when the
    kill switch is set.
  * Coordinator must block entries on kill switch, on
    max_open_positions, and on consecutive-loss cap.
  * Signal staleness must cause old fires to be evicted from the
    rolling market state so they don't count toward the N-signal rule.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from detectors.base import Direction, SignalFire
from executor.live import LiveExecutor
from executor.paper import PaperExecutor
from grid.config import GridConfig
from grid.coordinator import Coordinator
from grid.state import MarketState


# paper-mode isolation ----------------------------------------------

class TestPaperExecutorIsolation:
    def test_paper_never_touches_authenticated_client(self, tmp_path):
        import poly_data.global_state as gs
        from ledger.store import LedgerStore

        sentinel = MagicMock()
        gs.client = sentinel
        gs.all_data = {
            "mkt1": {"bids": {0.40: 100.0}, "asks": {0.42: 100.0}}
        }

        cfg = GridConfig(max_entry_usdc=10.0)
        paper = PaperExecutor(cfg, LedgerStore(str(tmp_path)))
        paper.enter_sync("mkt1", "tok1", Direction.BUY, 0.9,
                         {"detectors": ["a", "b", "c"]}, "cid123")

        sentinel.create_order.assert_not_called()
        sentinel.cancel_all_asset.assert_not_called()
        sentinel.cancel_all_market.assert_not_called()


class TestPublicClientBlocksWrites:
    def test_public_client_raises_on_write_methods(self):
        from poly_data.public_client import PublicPolymarketClient
        c = PublicPolymarketClient.__new__(PublicPolymarketClient)
        with pytest.raises(RuntimeError):
            c.create_order("tok", "BUY", 0.5, 10)
        with pytest.raises(RuntimeError):
            c.cancel_all_asset("tok")
        with pytest.raises(RuntimeError):
            c.cancel_all_market("mkt")
        with pytest.raises(RuntimeError):
            c.merge_positions(1, "cid", False)


# live-mode gating --------------------------------------------------

class TestLiveExecutorGating:
    def test_refuses_when_mode_is_paper(self, tmp_path, caplog):
        import poly_data.global_state as gs
        from ledger.store import LedgerStore

        gs.client = MagicMock()
        gs.all_data = {"mkt1": {"bids": {0.5: 100}, "asks": {0.51: 100}}}

        cfg = GridConfig(mode="paper", max_entry_usdc=10.0)
        live = LiveExecutor(cfg, LedgerStore(str(tmp_path)))
        with caplog.at_level("WARNING", logger="executor.live"):
            live.enter("mkt1", "tok1", Direction.BUY, 0.9, {}, "cid")

        gs.client.create_order.assert_not_called()
        assert any("mode is not" in r.getMessage() for r in caplog.records)

    def test_refuses_when_kill_switch_on(self, tmp_path, caplog):
        import poly_data.global_state as gs
        from ledger.store import LedgerStore

        gs.client = MagicMock()
        gs.all_data = {"mkt1": {"bids": {0.5: 100}, "asks": {0.51: 100}}}

        cfg = GridConfig(mode="live", kill_switch=True,
                         live_armed=True, max_entry_usdc=10.0)
        live = LiveExecutor(cfg, LedgerStore(str(tmp_path)))
        with caplog.at_level("WARNING", logger="executor.live"):
            live.enter("mkt1", "tok1", Direction.BUY, 0.9, {}, "cid")

        gs.client.create_order.assert_not_called()
        assert any("kill switch" in r.getMessage() for r in caplog.records)


# coordinator guardrails --------------------------------------------

def _fires(market, direction=Direction.BUY, n=3, ts=None):
    now = ts if ts is not None else time.time()
    return [
        SignalFire(f"d{i}", market, "tok", direction, 0.9, now)
        for i in range(n)
    ]


class TestCoordinatorGuardrails:
    def test_kill_switch_blocks_entry(self):
        entries = []
        cfg = GridConfig(min_signals=3, kill_switch=True)
        coord = Coordinator(cfg, lambda *a: entries.append(a))
        coord.ingest_sync(_fires("mkt1"))
        assert entries == []

    def test_max_open_positions_blocks_additional_markets(self):
        entries = []
        cfg = GridConfig(min_signals=3, max_open_positions=2)
        coord = Coordinator(cfg, lambda *a: entries.append(a))

        coord.ingest_sync(_fires("mkt1"))
        coord.ingest_sync(_fires("mkt2"))
        assert len(entries) == 2

        coord.ingest_sync(_fires("mkt3"))
        assert len(entries) == 2

    def test_consecutive_loss_cap_blocks_entries(self):
        entries = []
        cfg = GridConfig(min_signals=3, consecutive_loss_cap=2,
                         daily_loss_cap_usdc=1e9)
        coord = Coordinator(cfg, lambda *a: entries.append(a))

        coord.ingest_sync(_fires("mkt1"))
        coord.mark_closed_sync("mkt1", -1.0)
        coord.ingest_sync(_fires("mkt2"))
        coord.mark_closed_sync("mkt2", -1.0)

        coord.ingest_sync(_fires("mkt3"))
        assert len(entries) == 2

    def test_winning_trade_resets_consecutive_loss_count(self):
        entries = []
        cfg = GridConfig(min_signals=3, consecutive_loss_cap=2,
                         daily_loss_cap_usdc=1e9)
        coord = Coordinator(cfg, lambda *a: entries.append(a))

        coord.ingest_sync(_fires("mkt1"))
        coord.mark_closed_sync("mkt1", -1.0)
        coord.ingest_sync(_fires("mkt2"))
        coord.mark_closed_sync("mkt2", +2.0)

        coord.ingest_sync(_fires("mkt3"))
        coord.mark_closed_sync("mkt3", -1.0)
        coord.ingest_sync(_fires("mkt4"))
        assert len(entries) == 4

    def test_reset_daily_clears_counters(self):
        cfg = GridConfig(min_signals=3, daily_loss_cap_usdc=10.0)
        coord = Coordinator(cfg, lambda *a: None)

        coord.ingest_sync(_fires("mkt1"))
        coord.mark_closed_sync("mkt1", -50.0)
        assert coord.daily_loss_usdc >= 50.0
        assert coord.consecutive_losses >= 1

        coord.reset_daily()
        assert coord.daily_loss_usdc == 0
        assert coord.consecutive_losses == 0


# signal staleness --------------------------------------------------

class TestStaleness:
    def test_stale_signals_excluded_from_dominant_direction(self):
        ms = MarketState(staleness_secs=60.0)
        now = time.time()
        ms.update(SignalFire("old_buyer", "mkt", "tok",
                             Direction.BUY, 0.9, now - 90))
        ms.update(SignalFire("fresh1", "mkt", "tok",
                             Direction.SELL, 0.9, now - 10))
        ms.update(SignalFire("fresh2", "mkt", "tok",
                             Direction.SELL, 0.9, now - 5))

        active = ms.active_signals()
        assert len(active) == 2
        direction, agreement = ms.dominant_direction()
        assert direction == Direction.SELL
        assert agreement == 1.0

    def test_stale_signals_dont_count_toward_min_signals(self):
        entries = []
        cfg = GridConfig(min_signals=3, signal_staleness_secs=30.0)
        coord = Coordinator(cfg, lambda *a: entries.append(a))

        now = time.time()
        coord.ingest_sync([
            SignalFire("volume",      "mkt1", "tok", Direction.BUY, 0.9, now),
            SignalFire("velocity",    "mkt1", "tok", Direction.BUY, 0.9, now - 120),
            SignalFire("disposition", "mkt1", "tok", Direction.BUY, 0.9, now - 120),
        ])
        assert entries == []

"""Phase 3 tests.

Covers:

* :mod:`executor.book_walker` \u2014 walk direction, VWAP, slippage bps,
  empty-book and partial-fill handling.
* ``PaperExecutor`` slippage guard \u2014 refuses when ``max_slippage_bps``
  exceeded, accepts inside the budget and ledgers the VWAP + slip.
* ``PaperExecutor`` drift guard \u2014 refuses when the post-walk VWAP
  has drifted more than ``book_drift_bps`` from ``meta['fire_price']``.
* :mod:`executor.exit_strategy` \u2014 ``CentThresholdStrategy`` fires TP
  on a favourable cent move, SL on adverse, tightens near
  resolution, and ``PercentageStrategy`` preserves legacy behaviour.
* ``PaperExecutor.check_exits`` \u2014 drives a custom strategy and ledgers
  the right PnL.
* ``Coordinator`` category cap \u2014 same-category second fire is blocked
  with audit reason ``category_concentration``.
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock

import pytest

from detectors.base import Direction, SignalFire
from executor.book_walker import FillResult, walk_book
from executor.exit_strategy import (
    CentThresholdStrategy,
    ExitDecision,
    PercentageStrategy,
)
from executor.paper import PaperExecutor
from grid.config import GridConfig
from grid.coordinator import Coordinator
from ledger.store import LedgerStore


# --------------------------------------------------------------------
# book walker
# --------------------------------------------------------------------

class TestBookWalker:
    def test_buy_walks_asks_ascending(self):
        book = {"asks": {0.60: 5.0, 0.62: 10.0, 0.65: 20.0}, "bids": {}}
        fr = walk_book(book, Direction.BUY, 7.0)
        assert fr is not None
        assert fr.filled_size == pytest.approx(7.0)
        # 5 @ 0.60 + 2 @ 0.62 = 3.0 + 1.24 = 4.24 → vwap 4.24/7 ≈ 0.6057
        assert fr.vwap == pytest.approx((5 * 0.60 + 2 * 0.62) / 7.0)
        assert fr.best_price == 0.60
        assert fr.levels_consumed == 2
        # slippage vs best 0.60:  (vwap - 0.60) / 0.60 * 10_000
        expected = (fr.vwap - 0.60) / 0.60 * 10_000
        assert fr.slippage_bps == pytest.approx(expected)
        assert fr.is_full_fill

    def test_sell_walks_bids_descending(self):
        book = {"asks": {}, "bids": {0.40: 5.0, 0.38: 10.0, 0.30: 20.0}}
        fr = walk_book(book, Direction.SELL, 7.0)
        assert fr is not None
        assert fr.filled_size == pytest.approx(7.0)
        assert fr.vwap == pytest.approx((5 * 0.40 + 2 * 0.38) / 7.0)
        assert fr.best_price == 0.40
        expected = (0.40 - fr.vwap) / 0.40 * 10_000
        assert fr.slippage_bps == pytest.approx(expected)

    def test_empty_book_returns_none(self):
        assert walk_book({"asks": {}, "bids": {}}, Direction.BUY, 1.0) is None
        assert walk_book({"asks": {}, "bids": {}}, Direction.SELL, 1.0) is None

    def test_zero_or_negative_size_returns_none(self):
        book = {"asks": {0.5: 10.0}, "bids": {0.49: 10.0}}
        assert walk_book(book, Direction.BUY, 0) is None
        assert walk_book(book, Direction.BUY, -1.0) is None

    def test_partial_fill_when_book_too_thin(self):
        book = {"asks": {0.5: 3.0, 0.6: 1.0}, "bids": {}}
        fr = walk_book(book, Direction.BUY, 100.0)
        assert fr is not None
        assert fr.filled_size == pytest.approx(4.0)
        assert not fr.is_full_fill
        assert fr.levels_consumed == 2

    def test_single_level_zero_slippage(self):
        book = {"asks": {0.5: 100.0}, "bids": {}}
        fr = walk_book(book, Direction.BUY, 10.0)
        assert fr is not None
        assert fr.vwap == pytest.approx(0.5)
        assert fr.slippage_bps == pytest.approx(0.0)


# --------------------------------------------------------------------
# PaperExecutor: slippage + drift guards
# --------------------------------------------------------------------

def _audit_reasons(path):
    if not os.path.exists(path):
        return []
    tags = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("reason") is not None:
                tags.append(rec["reason"])
            if rec.get("event") is not None:
                tags.append(rec["event"])
    return tags


class TestPaperSlippageGuard:
    def test_refuses_when_slippage_exceeds_budget(self, tmp_path):
        import poly_data.global_state as gs
        # Thin book: best ask at 0.50, next level at 0.56 → a 25 USDC
        # BUY walks past the first level and averages well above 0.50.
        gs.all_data = {
            "mkt1": {
                "asks": {0.50: 5.0, 0.56: 100.0, 0.60: 100.0},
                "bids": {0.48: 100.0},
            }
        }
        cfg = GridConfig(max_entry_usdc=25.0, max_slippage_bps=50.0)
        led = LedgerStore(str(tmp_path))
        paper = PaperExecutor(cfg, led)

        result = paper.enter_sync(
            "mkt1", "tok1", Direction.BUY, 0.9,
            {"detectors": ["a", "b", "c"]}, "cid-slip",
        )
        assert result.ok is False
        assert result.reason == "slippage_exceeded"
        assert not paper.has_position("mkt1")
        reasons = _audit_reasons(tmp_path / "audit_log.jsonl")
        assert "slippage_exceeded" in reasons

    def test_accepts_and_fills_at_vwap_inside_budget(self, tmp_path):
        import poly_data.global_state as gs
        # Deep book at 0.50 → walk is trivial, slippage ~0.
        gs.all_data = {
            "mkt1": {
                "asks": {0.50: 1000.0},
                "bids": {0.49: 1000.0},
            }
        }
        cfg = GridConfig(max_entry_usdc=25.0, max_slippage_bps=50.0)
        led = LedgerStore(str(tmp_path))
        paper = PaperExecutor(cfg, led)

        result = paper.enter_sync(
            "mkt1", "tok1", Direction.BUY, 0.9,
            {"detectors": ["a", "b", "c"]}, "cid-ok",
        )
        assert result.ok is True
        pos = paper._positions["mkt1"]
        assert pos["entry_price"] == pytest.approx(0.50)
        # Existing sizing formula is min(max_entry_usdc / price,
        # max_entry_usdc): at price=0.5 that floors at 25 tokens.
        assert pos["size"] == pytest.approx(25.0)

        # The entry ledger record should carry the fill diagnostics.
        entries = [
            json.loads(l)
            for l in (tmp_path / "entries.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert entries
        meta = entries[-1]["meta"]
        assert "slippage_bps" in meta
        assert "vwap" in meta
        assert meta["levels_consumed"] >= 1


class TestPaperDriftGuard:
    def test_refuses_when_fire_price_drift_exceeds_limit(self, tmp_path):
        import poly_data.global_state as gs
        gs.all_data = {
            "mkt1": {
                "asks": {0.60: 1000.0},
                "bids": {0.59: 1000.0},
            }
        }
        # Fire happened at 0.50, fill VWAP lands at 0.60 →
        # drift = (0.60 - 0.50) / 0.50 * 10_000 = 2000 bps > 100 bps.
        cfg = GridConfig(
            max_entry_usdc=25.0, max_slippage_bps=10_000,
            book_drift_bps=100.0,
        )
        led = LedgerStore(str(tmp_path))
        paper = PaperExecutor(cfg, led)
        result = paper.enter_sync(
            "mkt1", "tok1", Direction.BUY, 0.9,
            {"fire_price": 0.50, "detectors": []}, "cid-drift",
        )
        assert result.ok is False
        assert result.reason == "book_drift_exceeded"
        assert not paper.has_position("mkt1")
        reasons = _audit_reasons(tmp_path / "audit_log.jsonl")
        assert "book_drift_exceeded" in reasons

    def test_accepts_when_drift_within_limit(self, tmp_path):
        import poly_data.global_state as gs
        gs.all_data = {
            "mkt1": {
                "asks": {0.502: 1000.0},
                "bids": {0.500: 1000.0},
            }
        }
        # VWAP 0.502 vs fire 0.500 → drift 40 bps, inside 100 bps.
        cfg = GridConfig(
            max_entry_usdc=25.0, max_slippage_bps=10_000,
            book_drift_bps=100.0,
        )
        led = LedgerStore(str(tmp_path))
        paper = PaperExecutor(cfg, led)
        result = paper.enter_sync(
            "mkt1", "tok1", Direction.BUY, 0.9,
            {"fire_price": 0.500, "detectors": []}, "cid-clean",
        )
        assert result.ok is True
        assert paper.has_position("mkt1")


# --------------------------------------------------------------------
# Exit strategies
# --------------------------------------------------------------------

def _pos(direction="BUY", entry_price=0.50, size=50.0):
    return {
        "correlation_id": "cidA",
        "token_id": "tok",
        "direction": direction,
        "entry_price": entry_price,
        "size": size,
    }


class TestCentThresholdStrategy:
    def test_tp_fires_on_favourable_cent_move(self):
        cfg = GridConfig(exit_tp_cents=3.0, exit_sl_cents=2.0)
        s = CentThresholdStrategy(cfg)
        d = s.evaluate(_pos("BUY", 0.50), 0.534, hours_to_resolution=24)
        assert d.exit_now is True
        assert d.reason == "tp"
        assert d.pnl_usdc > 0

    def test_sl_fires_on_adverse_cent_move(self):
        cfg = GridConfig(exit_tp_cents=3.0, exit_sl_cents=2.0)
        s = CentThresholdStrategy(cfg)
        d = s.evaluate(_pos("BUY", 0.50), 0.478, hours_to_resolution=24)
        assert d.exit_now is True
        assert d.reason == "sl"
        assert d.pnl_usdc < 0

    def test_hold_when_inside_band(self):
        cfg = GridConfig(exit_tp_cents=3.0, exit_sl_cents=2.0)
        s = CentThresholdStrategy(cfg)
        d = s.evaluate(_pos("BUY", 0.50), 0.515, hours_to_resolution=24)
        assert d.exit_now is False
        assert d.reason == "hold"

    def test_sell_direction(self):
        cfg = GridConfig(exit_tp_cents=3.0, exit_sl_cents=2.0)
        s = CentThresholdStrategy(cfg)
        # SELL from 0.70 → favourable when price drops
        d = s.evaluate(_pos("SELL", 0.70), 0.665, hours_to_resolution=24)
        assert d.reason == "tp"
        assert d.pnl_usdc > 0

    def test_tighten_near_resolution(self):
        cfg = GridConfig(
            exit_tp_cents=4.0, exit_sl_cents=4.0,
            exit_tighten_hours=1.0, exit_tighten_factor=0.5,
        )
        s = CentThresholdStrategy(cfg)
        # Outside tighten window: 3c move does not fire tp (needs 4c)
        d_far = s.evaluate(
            _pos("BUY", 0.50), 0.53, hours_to_resolution=24,
        )
        assert d_far.reason == "hold"
        # Inside tighten window: threshold halved to 2c, so 3c fires tp
        d_near = s.evaluate(
            _pos("BUY", 0.50), 0.53, hours_to_resolution=0.5,
        )
        assert d_near.reason == "tp"


class TestPercentageStrategy:
    def test_legacy_rule_preserved(self):
        s = PercentageStrategy(tp_pct=0.10, sl_pct=0.05)
        # BUY 0.50 → 10% up = 0.55
        assert s.evaluate(_pos("BUY", 0.50), 0.55, None).reason == "tp"
        # BUY 0.50 → 5% down = 0.475
        assert s.evaluate(_pos("BUY", 0.50), 0.475, None).reason == "sl"
        # No trigger
        assert s.evaluate(_pos("BUY", 0.50), 0.52, None).reason == "hold"


# --------------------------------------------------------------------
# PaperExecutor.check_exits with strategy object
# --------------------------------------------------------------------

class TestCheckExitsWithStrategy:
    def test_strategy_drives_exit_and_ledger_records_pnl(self, tmp_path):
        import poly_data.global_state as gs
        # Deep book so entry is clean.
        gs.all_data = {
            "mkt1": {
                "asks": {0.50: 1000.0},
                "bids": {0.49: 1000.0},
            }
        }
        cfg = GridConfig(
            max_entry_usdc=25.0, max_slippage_bps=200.0,
            exit_tp_cents=3.0, exit_sl_cents=2.0,
        )
        led = LedgerStore(str(tmp_path))
        paper = PaperExecutor(cfg, led)
        paper.enter_sync(
            "mkt1", "tok1", Direction.BUY, 0.9, {}, "cidX",
        )
        # Move best bid up so the SELL-exit price is > entry + 3c.
        gs.all_data["mkt1"]["bids"] = {0.54: 1000.0}
        closed = paper.check_exits_sync()
        assert len(closed) == 1
        market, pnl = closed[0]
        assert market == "mkt1"
        assert pnl > 0
        exits = [
            json.loads(l)
            for l in (tmp_path / "exits.jsonl").read_text().splitlines()
            if l.strip()
        ]
        assert exits and exits[-1]["pnl_usdc"] == pytest.approx(pnl)


# --------------------------------------------------------------------
# Category concentration cap wired via coordinator + detector
# --------------------------------------------------------------------

def _fires(market, n=3):
    now = time.time()
    return [
        SignalFire(f"d{i}", market, "tok", Direction.BUY, 0.9, now)
        for i in range(n)
    ]


class TestCategoryCap:
    def test_second_fire_same_category_is_blocked(self, tmp_path):
        led = LedgerStore(str(tmp_path))
        cfg = GridConfig(
            min_signals=3, max_open_positions=10,
            max_open_per_category=1,
            daily_loss_cap_usdc=1e9,
        )
        categories = {"mkt1": "crypto", "mkt2": "crypto", "mkt3": "sports"}
        captured = []
        coord = Coordinator(
            cfg, lambda *a: captured.append(a), ledger=led,
            category_of=lambda m: categories.get(m),
        )

        coord.ingest_sync(_fires("mkt1"))  # crypto → accepted
        coord.ingest_sync(_fires("mkt2"))  # crypto → blocked
        coord.ingest_sync(_fires("mkt3"))  # sports → accepted

        accepted = [c[0] for c in captured]
        assert "mkt1" in accepted
        assert "mkt2" not in accepted
        assert "mkt3" in accepted

        reasons = _audit_reasons(tmp_path / "audit_log.jsonl")
        assert "category_concentration" in reasons

    def test_same_category_allowed_up_to_cap(self, tmp_path):
        led = LedgerStore(str(tmp_path))
        cfg = GridConfig(
            min_signals=3, max_open_positions=10,
            max_open_per_category=2,
            daily_loss_cap_usdc=1e9,
        )
        categories = {"a": "crypto", "b": "crypto", "c": "crypto"}
        captured = []
        coord = Coordinator(
            cfg, lambda *a: captured.append(a), ledger=led,
            category_of=lambda m: categories.get(m),
        )
        coord.ingest_sync(_fires("a"))
        coord.ingest_sync(_fires("b"))
        coord.ingest_sync(_fires("c"))

        accepted = [c[0] for c in captured]
        assert accepted == ["a", "b"]
        reasons = _audit_reasons(tmp_path / "audit_log.jsonl")
        assert reasons.count("category_concentration") == 1

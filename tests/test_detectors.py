"""
Happy-path tests for every detector and the grid coordinator.

Run with:  uv run pytest tests/test_detectors.py -v
"""

import time
import pytest

from grid.config import GridConfig
from detectors.base import Direction, SignalFire


# ── helpers ─────────────────────────────────────────────────────────

def _trade_event(market="mkt1", token="tok1", side="BUY", size=100):
    return {
        "event_type": "trade",
        "market": market,
        "asset_id": token,
        "side": side,
        "size": size,
    }


def _book_event(market="mkt1", token="tok1", midpoint=0.5):
    return {
        "event_type": "book",
        "market": market,
        "asset_id": token,
        "midpoint": midpoint,
    }


# ── VolumeDetector ──────────────────────────────────────────────────

class TestVolumeDetector:
    def test_fires_on_spike(self):
        from detectors.volume import VolumeDetector
        cfg = GridConfig(volume_spike_multiplier=2.0)
        det = VolumeDetector(cfg)

        # Build a 15-min baseline of 10 trades each 1 unit.
        now = time.time()
        for i in range(10):
            e = _trade_event(size=1)
            det._trades["mkt1"].append((now - 800 + i, 1, "BUY"))

        # Now fire a spike: 30 units in the short window.
        fires = []
        for _ in range(30):
            fires.extend(det.on_event(_trade_event(size=1)))

        assert len(fires) > 0
        assert fires[-1].detector_name == "volume"
        assert fires[-1].direction == Direction.BUY

    def test_no_fire_without_baseline(self):
        from detectors.volume import VolumeDetector
        # A single trade lands in both the short AND long window, so
        # baseline is technically non-zero.  With the default 3× spike
        # multiplier, 1 trade cannot exceed 3× its own normalised
        # baseline.  Verify no fire.
        det = VolumeDetector(GridConfig(volume_spike_multiplier=3.0))
        fires = det.on_event(_trade_event(size=1))
        # A single data point produces baseline == short_vol (normalised),
        # so spike ratio == 1 which is below the 3× multiplier.
        assert fires == []


# ── VelocityDetector ────────────────────────────────────────────────

class TestVelocityDetector:
    def test_fires_on_rapid_price_move(self):
        from detectors.velocity import VelocityDetector
        cfg = GridConfig(velocity_threshold=0.001)
        det = VelocityDetector(cfg)

        now = time.time()
        # Seed a baseline price 4 min ago.
        det._prices["mkt1"].append((now - 240, 0.50))
        # Rapid upward move.
        det._prices["mkt1"].append((now - 2, 0.50))
        fires = det.on_event(_book_event(midpoint=0.60))

        assert len(fires) == 1
        assert fires[0].direction == Direction.BUY

    def test_no_fire_on_slow_move(self):
        from detectors.velocity import VelocityDetector
        cfg = GridConfig(velocity_threshold=0.01)
        det = VelocityDetector(cfg)
        # Slow drift: 0.001 change across 30s → rate 3.3e-5 < threshold.
        now = time.time()
        det._prices["mkt1"].append((now - 30, 0.500))
        fires = det.on_event(_book_event(midpoint=0.501))
        assert fires == []


# ── DispositionDetector ─────────────────────────────────────────────

class TestDispositionDetector:
    def test_fires_on_one_sided_aggression(self):
        from detectors.disposition import DispositionDetector
        cfg = GridConfig(disposition_threshold=0.6)
        det = DispositionDetector(cfg)

        fires = []
        for _ in range(8):
            fires.extend(det.on_event(_trade_event(side="BUY", size=10)))
        for _ in range(2):
            fires.extend(det.on_event(_trade_event(side="SELL", size=10)))

        assert any(f.direction == Direction.BUY for f in fires)

    def test_no_fire_when_balanced(self):
        from detectors.disposition import DispositionDetector
        cfg = GridConfig(disposition_threshold=0.7)
        det = DispositionDetector(cfg)

        # Interleave buys and sells so the running ratio stays near 50%.
        # The first event is always 100% one-sided, so we skip checking
        # the very first fire and instead look at the final state.
        for _ in range(10):
            det.on_event(_trade_event(side="BUY", size=10))
            det.on_event(_trade_event(side="SELL", size=10))

        # At this point buys == sells.  A fresh BUY should not fire
        # because the ratio is ≈ 0.52, well below 0.7.
        final_fires = det.on_event(_trade_event(side="BUY", size=10))
        assert final_fires == []


# ── NewsDetector ────────────────────────────────────────────────────

class TestNewsDetector:
    def test_fires_on_feed_vs_market_divergence(self):
        from detectors.news import NewsDetector
        cfg = GridConfig(news_delta_cents=3.0)
        det = NewsDetector(cfg)

        det.set_feed_value("mkt1", 0.60, "test_source")
        fires = det.on_event(_book_event(midpoint=0.50))

        assert len(fires) == 1
        assert fires[0].direction == Direction.BUY
        assert fires[0].meta["delta_cents"] == pytest.approx(10.0)

    def test_no_fire_when_aligned(self):
        from detectors.news import NewsDetector
        cfg = GridConfig(news_delta_cents=5.0)
        det = NewsDetector(cfg)

        det.set_feed_value("mkt1", 0.51, "test")
        fires = det.on_event(_book_event(midpoint=0.50))
        assert fires == []


# ── CrossMarketDetector ─────────────────────────────────────────────

class TestCrossMarketDetector:
    def test_fires_on_divergence(self):
        from detectors.cross_market import CrossMarketDetector
        cfg = GridConfig(cross_market_delta_cents=4.0)
        det = CrossMarketDetector(cfg)

        det.set_reference("mkt1", 0.70, "kalshi")
        fires = det.on_event(_book_event(midpoint=0.50))

        assert len(fires) == 1
        assert fires[0].direction == Direction.BUY


# ── WhaleDetector ───────────────────────────────────────────────────

class TestWhaleDetector:
    def test_fires_on_whale_trade(self):
        from detectors.whale import WhaleDetector
        cfg = GridConfig(whale_wallets=["0xabc"])
        det = WhaleDetector(cfg)

        fires = det.record_whale_trade("mkt1", "tok1", "0xabc", "BUY", 500)
        assert len(fires) == 1
        assert fires[0].direction == Direction.BUY

    def test_no_fire_without_wallets(self):
        from detectors.whale import WhaleDetector
        cfg = GridConfig(whale_wallets=[])
        det = WhaleDetector(cfg)

        fires = det.record_whale_trade("mkt1", "tok1", "0xabc", "BUY", 500)
        assert fires == []


# ── ThetaDetector ───────────────────────────────────────────────────

class TestThetaDetector:
    def test_fires_near_resolution_with_strong_lean(self):
        from detectors.theta import ThetaDetector
        cfg = GridConfig(theta_hours=48.0)
        det = ThetaDetector(cfg)

        # Market resolves in 12 hours, price at 0.85 → strong BUY lean.
        det.set_end_date("mkt1", time.time() + 12 * 3600)
        det.on_event(_book_event(midpoint=0.85))

        fires = det.poll()
        assert len(fires) == 1
        assert fires[0].direction == Direction.BUY

    def test_no_fire_when_price_is_neutral(self):
        from detectors.theta import ThetaDetector
        cfg = GridConfig(theta_hours=48.0)
        det = ThetaDetector(cfg)

        det.set_end_date("mkt1", time.time() + 12 * 3600)
        det.on_event(_book_event(midpoint=0.50))  # neutral price

        fires = det.poll()
        assert fires == []


# ── CategoryDetector ────────────────────────────────────────────────

class TestCategoryDetector:
    def test_fires_when_category_is_hot(self):
        from detectors.category import CategoryDetector
        det = CategoryDetector(GridConfig())

        det.set_market_category("mkt1", "crypto")
        det.set_category_stats("crypto", recent_volume=1000, baseline_volume=100)
        det.on_event(_book_event(midpoint=0.60))

        fires = det.poll()
        assert len(fires) == 1
        assert fires[0].market == "mkt1"


# ── Coordinator integration ─────────────────────────────────────────

class TestCoordinator:
    def test_triggers_entry_on_3_aligned_signals(self):
        from grid.coordinator import Coordinator

        entries = []

        def capture_entry(market, token_id, direction, confidence, meta, cid):
            entries.append((market, direction, meta, cid))

        cfg = GridConfig(min_signals=3, direction_threshold=0.6)
        coord = Coordinator(cfg, capture_entry)

        now = time.time()
        fires = [
            SignalFire("volume", "mkt1", "tok1", Direction.BUY, 0.8, now),
            SignalFire("velocity", "mkt1", "tok1", Direction.BUY, 0.9, now),
            SignalFire("disposition", "mkt1", "tok1", Direction.BUY, 0.7, now),
        ]
        coord.ingest_sync(fires)

        assert len(entries) == 1
        assert entries[0][1] == Direction.BUY
        assert entries[0][2]["n_signals"] == 3
        assert len(entries[0][3]) > 0  # correlation_id

    def test_no_entry_with_only_2_signals(self):
        from grid.coordinator import Coordinator

        entries = []
        cfg = GridConfig(min_signals=3)
        coord = Coordinator(cfg, lambda *a: entries.append(a))

        now = time.time()
        coord.ingest_sync([
            SignalFire("volume", "mkt1", "tok1", Direction.BUY, 0.8, now),
            SignalFire("velocity", "mkt1", "tok1", Direction.BUY, 0.9, now),
        ])
        assert entries == []

    def test_kill_switch_blocks_entry(self):
        from grid.coordinator import Coordinator

        entries = []
        cfg = GridConfig(min_signals=3, kill_switch=True)
        coord = Coordinator(cfg, lambda *a: entries.append(a))

        now = time.time()
        coord.ingest_sync([
            SignalFire("a", "mkt1", "tok1", Direction.BUY, 0.8, now),
            SignalFire("b", "mkt1", "tok1", Direction.BUY, 0.9, now),
            SignalFire("c", "mkt1", "tok1", Direction.BUY, 0.7, now),
        ])
        assert entries == []

    def test_one_position_per_market(self):
        from grid.coordinator import Coordinator

        entries = []
        cfg = GridConfig(min_signals=3)
        coord = Coordinator(cfg, lambda *a: entries.append(a))

        now = time.time()
        batch = [
            SignalFire("a", "mkt1", "tok1", Direction.BUY, 0.8, now),
            SignalFire("b", "mkt1", "tok1", Direction.BUY, 0.9, now),
            SignalFire("c", "mkt1", "tok1", Direction.BUY, 0.7, now),
        ]
        coord.ingest_sync(batch)
        assert len(entries) == 1

        coord.ingest_sync(batch)
        assert len(entries) == 1

    def test_daily_loss_cap(self):
        from grid.coordinator import Coordinator

        entries = []
        cfg = GridConfig(min_signals=3, daily_loss_cap_usdc=10.0)
        coord = Coordinator(cfg, lambda *a: entries.append(a))

        now = time.time()
        batch = [
            SignalFire("a", "mkt1", "tok1", Direction.BUY, 0.8, now),
            SignalFire("b", "mkt1", "tok1", Direction.BUY, 0.9, now),
            SignalFire("c", "mkt1", "tok1", Direction.BUY, 0.7, now),
        ]
        coord.ingest_sync(batch)
        assert len(entries) == 1

        coord.mark_closed_sync("mkt1", -15.0)

        batch2 = [
            SignalFire("a", "mkt2", "tok2", Direction.SELL, 0.8, now),
            SignalFire("b", "mkt2", "tok2", Direction.SELL, 0.9, now),
            SignalFire("c", "mkt2", "tok2", Direction.SELL, 0.7, now),
        ]
        coord.ingest_sync(batch2)
        assert len(entries) == 1

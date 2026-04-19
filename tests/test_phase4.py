"""Phase 4 tests.

Covers:

* ``VelocityDetector`` refuses to fire when ``top_of_book_size`` is
  missing is fine (no regression) / below the depth gate, and fires
  when above.  The raised default threshold is honoured.
* ``NewsDetector`` confidence is scaled by the time-to-resolution
  weight: full weight inside ``news_short_horizon_hours``, floor at
  ``news_far_weight`` beyond ``news_long_horizon_hours``, linear in
  between, midpoint when horizon is unknown.
* ``Coordinator.fire_quality_report`` counts fires per detector,
  credits only detectors that actually contributed to a grid fire,
  and survives snapshot / restore.
* ``scripts/report.fire_quality`` reconstructs the same shape from
  the JSON-lines ledger without a live coordinator.
"""

from __future__ import annotations

import json
import time
from typing import List, Optional

import pytest

from detectors.base import Direction, SignalFire
from detectors.news import NewsDetector
from detectors.velocity import VelocityDetector
from grid.config import GridConfig
from grid.coordinator import Coordinator
from ledger.store import LedgerStore


# --------------------------------------------------------------------
# Velocity depth gate
# --------------------------------------------------------------------

def _price_event(market: str, midpoint: float, depth: Optional[float] = None):
    ev = {
        "event_type": "book",
        "market": market,
        "asset_id": "tok1",
        "midpoint": midpoint,
    }
    if depth is not None:
        ev["top_of_book_size"] = depth
    return ev


class TestVelocityDepthGate:
    def _prime(self, det: VelocityDetector, market: str, price_series, depth=None):
        """Feed a ramp of prices so the detector has enough history."""
        now = time.time()
        # Manually backdate samples so the short-window delta is large.
        for i, p in enumerate(price_series):
            det._prices[market].append((now - (len(price_series) - i) * 5, p))

    def test_fires_when_depth_above_gate(self):
        cfg = GridConfig(
            velocity_threshold=0.001, velocity_min_book_depth=50.0,
        )
        det = VelocityDetector(cfg)
        # Synthetic history with a strong up-ramp → big short-window
        # rate, matched only by a small long-window rate.
        self._prime(det, "mkt1", [0.50, 0.52, 0.55, 0.58, 0.62, 0.66])
        fires = det.on_event(_price_event("mkt1", 0.70, depth=100.0))
        assert fires, "expected velocity fire on deep book"

    def test_does_not_fire_when_depth_below_gate(self):
        cfg = GridConfig(
            velocity_threshold=0.001, velocity_min_book_depth=50.0,
        )
        det = VelocityDetector(cfg)
        self._prime(det, "mkt2", [0.50, 0.52, 0.55, 0.58, 0.62, 0.66])
        # Best side only has 5 tokens → thin book, should be gated out
        fires = det.on_event(_price_event("mkt2", 0.70, depth=5.0))
        assert fires == []

    def test_fires_when_depth_missing(self):
        """Events without a ``top_of_book_size`` field must not regress \u2014
        the gate only kicks in when the field is present."""
        cfg = GridConfig(
            velocity_threshold=0.001, velocity_min_book_depth=50.0,
        )
        det = VelocityDetector(cfg)
        self._prime(det, "mkt3", [0.50, 0.52, 0.55, 0.58, 0.62, 0.66])
        fires = det.on_event(_price_event("mkt3", 0.70))
        assert fires, "events without depth must not be gated out"

    def test_default_threshold_is_raised(self):
        cfg = GridConfig()
        assert cfg.velocity_threshold == 0.005


# --------------------------------------------------------------------
# News time-to-resolution weight
# --------------------------------------------------------------------

class TestNewsTimeWeight:
    def _make(self, hours_left, *,
              short=1.0, long_=18.0, far=0.3):
        cfg = GridConfig(
            news_delta_cents=5.0,
            news_short_horizon_hours=short,
            news_long_horizon_hours=long_,
            news_far_weight=far,
        )
        det = NewsDetector(cfg, hours_to_resolution=lambda _m: hours_left)
        # Enough delta to fire at full weight: 30c above the market.
        det.set_feed_value("mkt1", 0.80, source="coingecko")
        det._midpoints["mkt1"] = 0.50
        return det

    def test_near_resolution_full_weight(self):
        det = self._make(hours_left=0.5)
        fires = det._evaluate("mkt1", "tok")
        assert len(fires) == 1
        assert fires[0].meta["time_weight"] == pytest.approx(1.0)

    def test_far_from_resolution_floor(self):
        det = self._make(hours_left=50.0)
        fires = det._evaluate("mkt1", "tok")
        assert len(fires) == 1
        assert fires[0].meta["time_weight"] == pytest.approx(0.3)
        # Confidence below the unweighted base.
        unweighted = self._make(hours_left=0.5)
        unweighted_fires = unweighted._evaluate("mkt1", "tok")
        assert fires[0].confidence < unweighted_fires[0].confidence

    def test_linear_between_horizons(self):
        # short=1, long=18, far=0.3.  hours_left = 9.5 sits at t=0.5 of
        # the span, weight = 1 - 0.5 * (1 - 0.3) = 0.65.
        det = self._make(hours_left=9.5)
        fires = det._evaluate("mkt1", "tok")
        assert fires[0].meta["time_weight"] == pytest.approx(0.65)

    def test_unknown_horizon_uses_midpoint(self):
        det = self._make(hours_left=None)
        fires = det._evaluate("mkt1", "tok")
        # Midpoint weight = (1 + 0.3) / 2 = 0.65
        assert fires[0].meta["time_weight"] == pytest.approx(0.65)

    def test_no_callback_preserves_pre_p4_behaviour(self):
        cfg = GridConfig(news_delta_cents=5.0, news_far_weight=0.3)
        det = NewsDetector(cfg)  # no hours_to_resolution
        det.set_feed_value("mkt1", 0.80, source="coingecko")
        det._midpoints["mkt1"] = 0.50
        fires = det._evaluate("mkt1", "tok")
        assert fires[0].meta["time_weight"] == pytest.approx(1.0)


# --------------------------------------------------------------------
# Coordinator fire-quality counters
# --------------------------------------------------------------------

def _fires(market, detectors=("a", "b", "c"), direction=Direction.BUY):
    now = time.time()
    return [
        SignalFire(d, market, "tok", direction, 0.9, now)
        for d in detectors
    ]


class TestFireQualityCounters:
    def test_counts_fires_per_detector(self):
        cfg = GridConfig(min_signals=3, daily_loss_cap_usdc=1e9)
        coord = Coordinator(cfg, lambda *a: None)
        coord.ingest_sync(_fires("m1", detectors=("a", "b", "c")))
        coord.ingest_sync(_fires("m2", detectors=("a", "b")))
        assert coord.fires_by_detector["a"] == 2
        assert coord.fires_by_detector["b"] == 2
        assert coord.fires_by_detector["c"] == 1

    def test_counts_contributions_only_on_grid_fires(self):
        cfg = GridConfig(min_signals=3, daily_loss_cap_usdc=1e9)
        captured = []
        coord = Coordinator(cfg, lambda *a: captured.append(a))
        # 3 fires on m1 → grid fire → credit a, b, c
        coord.ingest_sync(_fires("m1", detectors=("a", "b", "c")))
        # Only 2 fires on m2 → no grid fire → no credit
        coord.ingest_sync(_fires("m2", detectors=("d", "e")))
        assert captured  # sanity: m1 fired
        r = coord.fire_quality_report()
        assert r["a"]["contributions"] == 1
        assert r["b"]["contributions"] == 1
        assert r["c"]["contributions"] == 1
        assert r["d"]["contributions"] == 0
        assert r["e"]["contributions"] == 0
        assert r["a"]["quality_pct"] == pytest.approx(100.0)
        assert r["d"]["quality_pct"] == pytest.approx(0.0)

    def test_snapshot_round_trip(self):
        cfg = GridConfig(min_signals=3, daily_loss_cap_usdc=1e9)
        c1 = Coordinator(cfg, lambda *a: None)
        c1.ingest_sync(_fires("m1", detectors=("a", "b", "c")))
        c1.ingest_sync(_fires("m2", detectors=("a",)))
        snap = c1.snapshot()

        c2 = Coordinator(cfg, lambda *a: None)
        c2.restore(snap)
        r = c2.fire_quality_report()
        assert r["a"]["fires"] == 2
        assert r["a"]["contributions"] == 1
        assert r["b"]["fires"] == 1
        assert r["b"]["contributions"] == 1


# --------------------------------------------------------------------
# Offline fire_quality in scripts.report
# --------------------------------------------------------------------

class TestReportFireQuality:
    def test_reconstructs_contributions_from_ledger(self):
        from scripts.report import fire_quality

        base = 1000.0
        signal_fires = [
            {"ts": base + 0,  "detector": "velocity", "market": "m1"},
            {"ts": base + 1,  "detector": "news",     "market": "m1"},
            {"ts": base + 2,  "detector": "theta",    "market": "m1"},
            {"ts": base + 10, "detector": "velocity", "market": "m2"},
            # Stale on m2 (too far before the grid fire below):
            {"ts": base + 10, "detector": "news",     "market": "m2"},
        ]
        grid_fires = [
            {"ts": base + 3,   "market": "m1", "direction": "BUY",
             "n_signals": 3, "agreement": 1.0,
             "correlation_id": "c1"},
            # Grid fire on m2 400s after the only signals there \u2014 both
            # signals fall outside the staleness window of 300s and so
            # should not be credited.
            {"ts": base + 410, "market": "m2", "direction": "SELL",
             "n_signals": 2, "agreement": 1.0,
             "correlation_id": "c2"},
        ]
        report = fire_quality(signal_fires, grid_fires)
        assert report["velocity"]["contributions"] == 1
        assert report["news"]["contributions"] == 1
        assert report["theta"]["contributions"] == 1
        # m2's stale fires should not be credited.
        assert report["velocity"]["fires"] == 2
        assert report["news"]["fires"] == 2

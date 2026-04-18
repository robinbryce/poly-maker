"""Phase 2 tests.

Covers:

* ``GridConfig.validate_for_deployment`` gating for live mode
* ``GridConfig.reload_runtime_from`` updates runtime fields only and
  leaves deployment fields alone
* ``Coordinator`` stamps ``fire_price`` into entry meta when a
  ``get_midpoint`` callback is supplied
* ``LiveExecutor`` enforces the pre-trade book-drift guard using
  ``meta["fire_price"]`` and logs ``book_drift_exceeded``
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import MagicMock

import pytest

from detectors.base import Direction, SignalFire
from executor.live import LiveExecutor
from grid.config import DeploymentConfig, GridConfig, RuntimeConfig
from grid.coordinator import Coordinator
from ledger.store import LedgerStore


# --------------------------------------------------------------------
# deployment / runtime split
# --------------------------------------------------------------------

class TestConfigViews:
    def test_deployment_and_runtime_views_are_typed(self):
        cfg = GridConfig()
        dep = cfg.deployment()
        rt = cfg.runtime()
        assert isinstance(dep, DeploymentConfig)
        assert isinstance(rt, RuntimeConfig)

    def test_split_covers_every_field(self):
        cfg = GridConfig()
        from dataclasses import fields
        declared = {f.name for f in fields(cfg)}
        covered = cfg.DEPLOYMENT_FIELDS | cfg.RUNTIME_FIELDS
        assert declared <= covered


# --------------------------------------------------------------------
# validate_for_deployment
# --------------------------------------------------------------------

class TestValidateForDeployment:
    def test_paper_mode_passes(self, monkeypatch):
        monkeypatch.delenv("PK", raising=False)
        cfg = GridConfig(mode="paper")
        cfg.validate_for_deployment()

    def test_live_requires_live_armed(self, monkeypatch):
        monkeypatch.setenv("PK", "0xdeadbeef")
        cfg = GridConfig(mode="live", live_armed=False)
        with pytest.raises(ValueError, match="live_armed"):
            cfg.validate_for_deployment()

    def test_live_requires_pk(self, monkeypatch):
        monkeypatch.delenv("PK", raising=False)
        cfg = GridConfig(mode="live", live_armed=True)
        with pytest.raises(ValueError, match="PK"):
            cfg.validate_for_deployment()

    def test_live_passes_when_both_locks_set(self, monkeypatch):
        monkeypatch.setenv("PK", "0xdeadbeef")
        cfg = GridConfig(mode="live", live_armed=True)
        cfg.validate_for_deployment()

    def test_unknown_mode_rejected(self):
        cfg = GridConfig(mode="simulation")
        with pytest.raises(ValueError, match="mode must"):
            cfg.validate_for_deployment()


# --------------------------------------------------------------------
# reload_runtime_from
# --------------------------------------------------------------------

class TestReloadRuntimeFrom:
    def test_runtime_fields_applied(self, tmp_path):
        cfg = GridConfig(min_signals=3, velocity_threshold=0.002)
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({
            "min_signals": 4,
            "velocity_threshold": 0.005,
        }))

        changed = cfg.reload_runtime_from(str(p))
        assert changed == {"min_signals", "velocity_threshold"}
        assert cfg.min_signals == 4
        assert cfg.velocity_threshold == 0.005

    def test_deployment_fields_ignored(self, tmp_path):
        cfg = GridConfig(mode="paper", state_dir="state",
                         ledger_dir="ledger_data")
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({
            "mode": "live",
            "state_dir": "/tmp/newstate",
            "ledger_dir": "/tmp/newledger",
            "velocity_threshold": 0.009,
        }))

        changed = cfg.reload_runtime_from(str(p))
        assert changed == {"velocity_threshold"}
        # Deployment fields must not have moved.
        assert cfg.mode == "paper"
        assert cfg.state_dir == "state"
        assert cfg.ledger_dir == "ledger_data"
        # Runtime field moved.
        assert cfg.velocity_threshold == 0.009

    def test_unchanged_fields_not_reported(self, tmp_path):
        cfg = GridConfig(min_signals=3)
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({"min_signals": 3, "velocity_threshold":
                                 cfg.velocity_threshold}))
        changed = cfg.reload_runtime_from(str(p))
        assert changed == set()

    def test_unknown_keys_tolerated(self, tmp_path):
        cfg = GridConfig()
        p = tmp_path / "cfg.json"
        p.write_text(json.dumps({
            "__doc_comment": "operator notes",
            "min_signals": 5,
        }))
        changed = cfg.reload_runtime_from(str(p))
        assert changed == {"min_signals"}
        assert cfg.min_signals == 5


# --------------------------------------------------------------------
# coordinator stamps fire_price in meta
# --------------------------------------------------------------------

def _fires(market, n=3):
    now = time.time()
    return [
        SignalFire(f"d{i}", market, "tok", Direction.BUY, 0.9, now)
        for i in range(n)
    ]


class TestCoordinatorFirePrice:
    def test_fire_price_stamped_when_midpoint_available(self):
        captured = []
        cfg = GridConfig(min_signals=3, daily_loss_cap_usdc=1e9)
        coord = Coordinator(
            cfg, lambda *a: captured.append(a),
            get_midpoint=lambda _m: 0.42,
        )
        coord.ingest_sync(_fires("mkt1"))
        assert captured, "expected an entry"
        meta = captured[0][4]
        assert meta.get("fire_price") == 0.42

    def test_fire_price_omitted_when_midpoint_none(self):
        captured = []
        cfg = GridConfig(min_signals=3, daily_loss_cap_usdc=1e9)
        coord = Coordinator(
            cfg, lambda *a: captured.append(a),
            get_midpoint=lambda _m: None,
        )
        coord.ingest_sync(_fires("mkt1"))
        assert captured, "expected an entry"
        meta = captured[0][4]
        assert "fire_price" not in meta

    def test_fire_price_omitted_when_callback_not_provided(self):
        captured = []
        cfg = GridConfig(min_signals=3, daily_loss_cap_usdc=1e9)
        coord = Coordinator(cfg, lambda *a: captured.append(a))
        coord.ingest_sync(_fires("mkt1"))
        assert captured
        assert "fire_price" not in captured[0][4]


# --------------------------------------------------------------------
# live executor drift guard
# --------------------------------------------------------------------

def _live_cfg(book_drift_bps=100.0):
    return GridConfig(
        mode="live", live_armed=True, kill_switch=False,
        max_entry_usdc=10.0, book_drift_bps=book_drift_bps,
    )


def _audit_reasons(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l).get("reason") for l in f if l.strip()]


class TestLiveDriftGuard:
    def test_refuses_when_drift_exceeds_limit(self, tmp_path, capsys):
        import poly_data.global_state as gs

        gs.client = MagicMock()
        # Fire happened at 0.50 (meta), book now shows best-ask 0.60
        # → drift = |0.60 - 0.50| / 0.50 * 10_000 = 2000 bps,
        # exceeds the default 100 bps guard.
        gs.all_data = {
            "mkt1": {"bids": {0.58: 100}, "asks": {0.60: 100}},
        }

        cfg = _live_cfg(book_drift_bps=100.0)
        led = LedgerStore(str(tmp_path))
        live = LiveExecutor(cfg, led)

        live.enter(
            "mkt1", "tok1", Direction.BUY, 0.9,
            {"fire_price": 0.50}, "cid-drift",
        )

        gs.client.create_order.assert_not_called()
        reasons = _audit_reasons(tmp_path / "audit_log.jsonl")
        assert "book_drift_exceeded" in reasons
        assert "book drift" in capsys.readouterr().out

    def test_allows_when_drift_within_limit(self, tmp_path):
        import poly_data.global_state as gs

        gs.client = MagicMock()
        gs.client.create_order.return_value = {"orderID": "abc"}
        # Fire at 0.500, book now 0.502 → drift = 40 bps, < 100 bps.
        gs.all_data = {
            "mkt1": {"bids": {0.500: 100}, "asks": {0.502: 100}},
        }

        cfg = _live_cfg(book_drift_bps=100.0)
        led = LedgerStore(str(tmp_path))
        live = LiveExecutor(cfg, led)

        live.enter(
            "mkt1", "tok1", Direction.BUY, 0.9,
            {"fire_price": 0.500}, "cid-ok",
        )

        gs.client.create_order.assert_called_once()
        reasons = _audit_reasons(tmp_path / "audit_log.jsonl")
        assert "book_drift_exceeded" not in reasons

    def test_no_drift_guard_when_fire_price_missing(self, tmp_path):
        import poly_data.global_state as gs

        gs.client = MagicMock()
        gs.client.create_order.return_value = {"orderID": "xyz"}
        gs.all_data = {
            "mkt1": {"bids": {0.20: 100}, "asks": {0.90: 100}},
        }

        cfg = _live_cfg(book_drift_bps=100.0)
        led = LedgerStore(str(tmp_path))
        live = LiveExecutor(cfg, led)

        # No fire_price in meta → guard is a no-op, the order proceeds
        # even though the mid has moved wildly.
        live.enter("mkt1", "tok1", Direction.BUY, 0.9, {}, "cid-none")

        gs.client.create_order.assert_called_once()
        reasons = _audit_reasons(tmp_path / "audit_log.jsonl")
        assert "book_drift_exceeded" not in reasons

    def test_no_asks_audit_reason(self, tmp_path, capsys):
        import poly_data.global_state as gs

        gs.client = MagicMock()
        gs.all_data = {"mkt1": {"bids": {0.5: 100}, "asks": {}}}

        cfg = _live_cfg()
        led = LedgerStore(str(tmp_path))
        live = LiveExecutor(cfg, led)

        live.enter(
            "mkt1", "tok1", Direction.BUY, 0.9,
            {"fire_price": 0.5}, "cid-nask",
        )
        gs.client.create_order.assert_not_called()
        reasons = _audit_reasons(tmp_path / "audit_log.jsonl")
        assert "no_asks" in reasons

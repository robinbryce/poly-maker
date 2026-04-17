"""Phase 1 tests: ledger durability, correlation IDs, snapshot
round-trip, wall-clock daily reset, audit log for blocked decisions.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import pytest

from detectors.base import Direction, SignalFire
from executor.paper import PaperExecutor
from grid.config import GridConfig
from grid.coordinator import Coordinator
from grid.snapshot import SnapshotStore
from ledger.store import LedgerStore, new_correlation_id


# --------------------------------------------------------------------
# ledger durability
# --------------------------------------------------------------------

class TestLedgerDurability:
    def test_concurrent_writers_do_not_interleave(self, tmp_path):
        led = LedgerStore(str(tmp_path))
        n_per_writer = 200
        n_writers = 6

        def write(writer_id):
            for i in range(n_per_writer):
                led.log_audit("ping", writer_id=writer_id, i=i)

        threads = [threading.Thread(target=write, args=(w,))
                   for w in range(n_writers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every line must be valid JSON (no torn writes).
        path = tmp_path / "audit_log.jsonl"
        count = 0
        with open(path) as f:
            for line in f:
                json.loads(line)
                count += 1
        assert count == n_per_writer * n_writers

    def test_rotation_on_size(self, tmp_path):
        led = LedgerStore(str(tmp_path), max_bytes=500)
        for i in range(200):
            led.log_audit("noise", i=i, padding="x" * 80)
        files = sorted(os.listdir(tmp_path))
        rotated = [f for f in files if f.startswith("audit_log.") and f != "audit_log.jsonl"]
        assert len(rotated) >= 1, files

    def test_ts_monotonic_ns_is_present(self, tmp_path):
        led = LedgerStore(str(tmp_path))
        led.log_audit("test")
        rec = json.loads((tmp_path / "audit_log.jsonl").read_text().strip())
        assert "ts_monotonic_ns" in rec


# --------------------------------------------------------------------
# correlation IDs
# --------------------------------------------------------------------

class TestCorrelationIds:
    def test_fire_to_entry_to_exit_share_cid(self, tmp_path):
        led = LedgerStore(str(tmp_path))
        cfg = GridConfig(min_signals=3, max_entry_usdc=10.0,
                         daily_loss_cap_usdc=1e9)
        captured = []
        coord = Coordinator(cfg, lambda *a: captured.append(a), ledger=led)
        now = time.time()
        coord.ingest_sync([
            SignalFire("a", "mkt1", "tok1", Direction.BUY, 0.9, now),
            SignalFire("b", "mkt1", "tok1", Direction.BUY, 0.9, now),
            SignalFire("c", "mkt1", "tok1", Direction.BUY, 0.9, now),
        ])
        assert captured, "coordinator should have emitted an entry"
        cid = captured[0][5]
        assert len(cid) > 0

        grid_line = json.loads(
            (tmp_path / "grid_fires.jsonl").read_text().strip().splitlines()[0]
        )
        assert grid_line["correlation_id"] == cid


# --------------------------------------------------------------------
# audit log for blocks
# --------------------------------------------------------------------

class TestAuditLog:
    def test_kill_switch_block_is_recorded(self, tmp_path):
        led = LedgerStore(str(tmp_path))
        cfg = GridConfig(min_signals=3, kill_switch=True)
        coord = Coordinator(cfg, lambda *a: None, ledger=led)
        now = time.time()
        coord.ingest_sync([
            SignalFire("a", "mkt1", "tok", Direction.BUY, 0.9, now),
            SignalFire("b", "mkt1", "tok", Direction.BUY, 0.9, now),
            SignalFire("c", "mkt1", "tok", Direction.BUY, 0.9, now),
        ])
        audit = (tmp_path / "audit_log.jsonl").read_text().splitlines()
        reasons = [json.loads(l).get("reason") for l in audit]
        assert "kill_switch" in reasons

    def test_max_open_positions_block_is_recorded(self, tmp_path):
        led = LedgerStore(str(tmp_path))
        cfg = GridConfig(min_signals=3, max_open_positions=1)
        coord = Coordinator(cfg, lambda *a: None, ledger=led)
        now = time.time()
        fires_a = [SignalFire(f"d{i}", "mkt1", "tok", Direction.BUY, 0.9, now) for i in range(3)]
        fires_b = [SignalFire(f"d{i}", "mkt2", "tok", Direction.BUY, 0.9, now) for i in range(3)]
        coord.ingest_sync(fires_a)
        coord.ingest_sync(fires_b)
        audit = (tmp_path / "audit_log.jsonl").read_text().splitlines()
        reasons = [json.loads(l).get("reason") for l in audit]
        assert "max_open_positions" in reasons


# --------------------------------------------------------------------
# snapshot round-trip
# --------------------------------------------------------------------

class TestSnapshotRoundTrip:
    def test_coordinator_snapshot_restores_state(self, tmp_path):
        led = LedgerStore(str(tmp_path / "ledger"))
        cfg = GridConfig(min_signals=3, daily_loss_cap_usdc=1e9)
        c1 = Coordinator(cfg, lambda *a: None, ledger=led)
        now = time.time()
        c1.ingest_sync([
            SignalFire(f"d{i}", "mkt1", "tok", Direction.BUY, 0.9, now)
            for i in range(3)
        ])
        c1.mark_closed_sync("mkt1", -5.0)

        snap = c1.snapshot()
        # Write through SnapshotStore.
        store = SnapshotStore(str(tmp_path / "state"))
        store.save({"coordinator": snap})

        loaded = store.load()
        c2 = Coordinator(cfg, lambda *a: None, ledger=led)
        c2.restore(loaded["coordinator"])
        assert c2.daily_loss_usdc == 5.0
        assert c2.consecutive_losses == 1

    def test_paper_executor_snapshot_restores_positions(self, tmp_path):
        import poly_data.global_state as gs
        gs.all_data = {"mkt1": {"bids": {0.4: 100}, "asks": {0.5: 100}}}
        led = LedgerStore(str(tmp_path / "ledger"))
        cfg = GridConfig(max_entry_usdc=10.0)
        p1 = PaperExecutor(cfg, led)
        p1.enter_sync("mkt1", "tok1", Direction.BUY, 0.9,
                      {"detectors": []}, "cidAAA")
        snap = p1.snapshot()

        p2 = PaperExecutor(cfg, led)
        p2.restore(snap)
        assert p2.open_count == 1
        assert "mkt1" in p2._positions
        assert p2._positions["mkt1"]["correlation_id"] == "cidAAA"


# --------------------------------------------------------------------
# UTC daily reset
# --------------------------------------------------------------------

class TestDailyReset:
    def test_wall_clock_day_change_triggers_reset(self, tmp_path):
        led = LedgerStore(str(tmp_path))
        cfg = GridConfig(min_signals=3, daily_loss_cap_usdc=1e9)
        coord = Coordinator(cfg, lambda *a: None, ledger=led)
        now = time.time()
        fires = [SignalFire(f"d{i}", "mkt1", "tok", Direction.BUY, 0.9, now)
                 for i in range(3)]
        coord.ingest_sync(fires)
        coord.mark_closed_sync("mkt1", -7.0)
        assert coord.daily_loss_usdc == 7.0

        # Simulate a day roll by moving the recorded date backwards.
        coord._last_reset_utc_date = "1999-01-01"
        fires2 = [SignalFire(f"d{i}", "mkt2", "tok", Direction.BUY, 0.9, now)
                  for i in range(3)]
        coord.ingest_sync(fires2)
        assert coord.daily_loss_usdc == 0.0
        assert coord.consecutive_losses == 0

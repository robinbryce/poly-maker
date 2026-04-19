"""Phase 6 tests: integration, chaos, invariants, ledger round-trip.

Depth coverage that the per-phase happy-path suites don't reach:

* :class:`TestIntegration` \u2014 full stack from coordinator ingest to
  paper executor enter to ledger lines, asserting the correlation
  id and P3 fill diagnostics stitch through the pipeline end to
  end.  A second test drives the real asyncio path via
  ``await coordinator.ingest(...)``.

* :class:`TestChaos` \u2014 simulate a crash mid-position by dropping the
  coordinator / paper executor, re-building them against the same
  ledger directory, restoring from snapshot, and reconciling.
  Asserts no double entry even when the same fires arrive again.

* :class:`TestInvariants` \u2014 a seeded random walk over 200 ingest /
  mark_closed steps that checks coordinator invariants
  (``max_open_positions``, ``max_open_per_category``,
  ``consecutive_loss_cap``, ``kill_switch``) hold at every step.

* :class:`TestLedgerRoundTrip` \u2014 confirms every field written by
  ``LedgerStore`` survives ``scripts.report.load_jsonl`` and that
  rotation does not lose any records.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import pytest

from detectors.base import Direction, SignalFire
from executor.paper import PaperExecutor
from grid.config import GridConfig
from grid.coordinator import Coordinator
from ledger.store import LedgerStore


# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------

def _seed_book(market: str = "m1"):
    """Install a deep, stable book so paper entries fill cleanly."""
    import poly_data.global_state as gs
    gs.all_data = {
        market: {"bids": {0.49: 1000.0}, "asks": {0.50: 1000.0}},
    }


def _fires_for(market: str, detectors=("velocity", "news", "theta"),
               direction=Direction.BUY, now=None):
    """Build N synthetic fires on the same market + direction."""
    ts = now if now is not None else time.time()
    return [
        SignalFire(d, market, "tok1", direction, 0.9, ts)
        for d in detectors
    ]


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# --------------------------------------------------------------------
# Integration
# --------------------------------------------------------------------

class TestIntegration:
    def test_full_stack_sync_path(self, tmp_path):
        _seed_book("m1")
        ledger_dir = tmp_path / "ledger"
        led = LedgerStore(str(ledger_dir))
        cfg = GridConfig(
            min_signals=3, max_entry_usdc=10.0,
            max_slippage_bps=200.0, daily_loss_cap_usdc=1e9,
        )
        paper = PaperExecutor(cfg, led)

        def _on_entry(market, token_id, direction,
                      confidence, meta, correlation_id):
            return paper.enter_sync(
                market, token_id, direction, confidence,
                meta, correlation_id,
            )

        coord = Coordinator(cfg, _on_entry, led)
        coord.ingest_sync(_fires_for("m1"))

        # Ledger shape -------------------------------------------------
        sig = _read_jsonl(ledger_dir / "signal_fires.jsonl")
        grid = _read_jsonl(ledger_dir / "grid_fires.jsonl")
        entries = _read_jsonl(ledger_dir / "entries.jsonl")
        assert len(sig) == 3
        assert len(grid) == 1
        assert len(entries) == 1

        # Correlation id stitches all three ledger layers together.
        cid = grid[0]["correlation_id"]
        assert entries[0]["correlation_id"] == cid

        # P3 fill diagnostics travelled through the meta dict.
        emeta = entries[0]["meta"]
        for k in ("best_price", "vwap", "slippage_bps",
                  "levels_consumed", "requested_size"):
            assert k in emeta, f"missing {k} in entry meta"

        # Executor + coordinator agree on the open set.
        assert paper.open_count == 1
        assert "m1" in paper.open_markets()
        assert coord.open_markets == {"m1"}

    def test_full_stack_async_path(self, tmp_path):
        """Same scenario but through ``await coordinator.ingest``."""
        _seed_book("m1")
        ledger_dir = tmp_path / "ledger"
        led = LedgerStore(str(ledger_dir))
        cfg = GridConfig(
            min_signals=3, max_entry_usdc=10.0,
            max_slippage_bps=200.0, daily_loss_cap_usdc=1e9,
        )
        paper = PaperExecutor(cfg, led)

        async def _on_entry(*args):
            return await paper.enter(*args)

        coord = Coordinator(cfg, _on_entry, led)

        async def _drive():
            await coord.ingest(_fires_for("m1"))

        asyncio.run(_drive())

        assert paper.open_count == 1
        assert coord.open_markets == {"m1"}
        assert len(_read_jsonl(ledger_dir / "entries.jsonl")) == 1


# --------------------------------------------------------------------
# Chaos
# --------------------------------------------------------------------

class TestChaos:
    def test_crash_restore_does_not_double_enter(self, tmp_path):
        _seed_book("m1")
        ledger_dir = tmp_path / "ledger"
        state_dir = tmp_path / "state"
        os.makedirs(state_dir, exist_ok=True)

        cfg = GridConfig(
            min_signals=3, max_entry_usdc=10.0,
            max_slippage_bps=200.0, daily_loss_cap_usdc=1e9,
            state_dir=str(state_dir), ledger_dir=str(ledger_dir),
        )

        # --- run 1: produce the entry and snapshot the state -----
        led1 = LedgerStore(str(ledger_dir))
        paper1 = PaperExecutor(cfg, led1)

        def _on_entry_1(*args):
            return paper1.enter_sync(*args)

        coord1 = Coordinator(cfg, _on_entry_1, led1)
        coord1.ingest_sync(_fires_for("m1"))
        assert paper1.open_count == 1
        snapshot = {
            "coordinator": coord1.snapshot(),
            "paper_executor": paper1.snapshot(),
        }

        # --- crash: drop the in-memory state ---------------------
        del coord1, paper1, led1

        # --- run 2: fresh instances, restore, reconcile, re-fire -
        led2 = LedgerStore(str(ledger_dir))
        paper2 = PaperExecutor(cfg, led2)
        paper2.restore(snapshot["paper_executor"])

        def _on_entry_2(*args):
            return paper2.enter_sync(*args)

        coord2 = Coordinator(cfg, _on_entry_2, led2)
        coord2.restore(snapshot["coordinator"])
        coord2.reconcile_open_markets_sync(paper2.open_markets())

        # Drive the same fires again.
        coord2.ingest_sync(_fires_for("m1"))

        entries = _read_jsonl(Path(ledger_dir) / "entries.jsonl")
        assert len(entries) == 1, (
            "exactly one entry should exist across both runs"
        )
        audit = _read_jsonl(Path(ledger_dir) / "audit_log.jsonl")
        reasons = [r.get("reason") or r.get("event") for r in audit]
        assert "already_open" in reasons

    def test_reconcile_evicts_orphans_then_allows_new_fires(self, tmp_path):
        _seed_book("m1")
        ledger_dir = tmp_path / "ledger"
        led = LedgerStore(str(ledger_dir))
        cfg = GridConfig(
            min_signals=3, max_entry_usdc=10.0,
            max_slippage_bps=200.0, daily_loss_cap_usdc=1e9,
        )
        paper = PaperExecutor(cfg, led)

        def _on_entry(*args):
            return paper.enter_sync(*args)

        coord = Coordinator(cfg, _on_entry, led)
        # Restore a "stale" snapshot that says m1 is open but no
        # matching paper position exists.
        coord.restore({"open_markets": ["m1"]})
        assert coord.open_markets == {"m1"}
        assert paper.open_count == 0

        evicted = coord.reconcile_open_markets_sync(paper.open_markets())
        assert evicted == 1
        assert coord.open_markets == set()

        # A fresh fire on m1 is now accepted.
        coord.ingest_sync(_fires_for("m1"))
        assert paper.open_count == 1


# --------------------------------------------------------------------
# Randomised invariants
# --------------------------------------------------------------------

class TestInvariants:
    MARKETS = ["a", "b", "c", "d", "e"]
    CATEGORIES = {"a": "crypto", "b": "crypto", "c": "sports",
                  "d": "politics", "e": "crypto"}

    def _step(self, coord, rng):
        market = rng.choice(self.MARKETS)
        op = rng.choices(["ingest", "close"], weights=[3, 1])[0]
        if op == "ingest":
            coord.ingest_sync(_fires_for(market))
        else:
            pnl = rng.uniform(-5.0, 5.0)
            coord.mark_closed_sync(market, pnl)

    def _invariants_hold(self, coord, cfg):
        assert len(coord.open_markets) <= cfg.max_open_positions
        if cfg.max_open_per_category > 0:
            by_cat: Dict[str, int] = {}
            for m in coord.open_markets:
                c = self.CATEGORIES.get(m)
                if c:
                    by_cat[c] = by_cat.get(c, 0) + 1
            for c, n in by_cat.items():
                assert n <= cfg.max_open_per_category, (
                    f"category {c} has {n} > cap {cfg.max_open_per_category}"
                )

    def test_random_walk_200_steps(self):
        cfg = GridConfig(
            min_signals=3, max_open_positions=3,
            max_open_per_category=2, consecutive_loss_cap=4,
            daily_loss_cap_usdc=1e9,
        )
        coord = Coordinator(
            cfg, lambda *a: None,
            category_of=lambda m: self.CATEGORIES.get(m),
        )
        rng = random.Random(0xE197)
        for _ in range(200):
            self._step(coord, rng)
            self._invariants_hold(coord, cfg)

    def test_random_walk_1000_steps_longer_seed(self):
        cfg = GridConfig(
            min_signals=3, max_open_positions=3,
            max_open_per_category=2, consecutive_loss_cap=5,
            daily_loss_cap_usdc=1e9,
        )
        coord = Coordinator(
            cfg, lambda *a: None,
            category_of=lambda m: self.CATEGORIES.get(m),
        )
        rng = random.Random(0xBEEF)
        for _ in range(1000):
            self._step(coord, rng)
            self._invariants_hold(coord, cfg)

    def test_kill_switch_blocks_every_fire(self):
        cfg = GridConfig(
            min_signals=3, kill_switch=True,
            daily_loss_cap_usdc=1e9,
        )
        captured = []
        coord = Coordinator(cfg, lambda *a: captured.append(a))
        rng = random.Random(0xDEAD)
        for _ in range(100):
            coord.ingest_sync(_fires_for(rng.choice(self.MARKETS)))
        assert captured == [], "no entries should pass with kill_switch"
        assert coord.open_markets == set()

    def test_consecutive_loss_cap_halts_after_N_losses(self):
        cfg = GridConfig(
            min_signals=3, consecutive_loss_cap=3,
            max_open_positions=10, daily_loss_cap_usdc=1e9,
        )
        captured = []
        coord = Coordinator(cfg, lambda *a: captured.append(a))

        # Lose 3 in a row, then confirm the next fire is blocked.
        for i, m in enumerate(["w", "x", "y"]):
            coord.ingest_sync(_fires_for(m))
            coord.mark_closed_sync(m, -1.0)
        before = len(captured)
        coord.ingest_sync(_fires_for("z"))
        after = len(captured)
        assert after == before, "fires after the cap should be blocked"


# --------------------------------------------------------------------
# Ledger writer ↔ report reader round-trip
# --------------------------------------------------------------------

class TestLedgerRoundTrip:
    def test_every_ledger_method_round_trips(self, tmp_path):
        from scripts.report import load_jsonl

        led = LedgerStore(str(tmp_path))

        # Write one of each kind.
        fire = SignalFire(
            "velocity", "m1", "tok", Direction.BUY, 0.9, time.time(),
            meta={"short_rate": 0.01},
        )
        led.log_signal_fire(fire)
        led.log_grid_fire("m1", Direction.BUY, 3, 1.0, "cid-grid")
        led.log_entry(
            "m1", "tok", "BUY", 25.0, 0.5, "paper",
            {"detectors": ["velocity", "news", "theta"], "fire_price": 0.5},
            "cid-entry",
        )
        led.log_exit(
            "m1", "tok", "BUY", 25.0, 0.53, 0.75, "paper", "cid-exit",
        )
        led.log_block("kill_switch", "m1", {"note": "blocked"})
        led.log_audit("runtime_reload", changed=["min_signals"])

        # Read back through the report reader.
        sig = load_jsonl(tmp_path / "signal_fires.jsonl")
        grid = load_jsonl(tmp_path / "grid_fires.jsonl")
        entries = load_jsonl(tmp_path / "entries.jsonl")
        exits = load_jsonl(tmp_path / "exits.jsonl")
        audits = load_jsonl(tmp_path / "audit_log.jsonl")

        assert sig and sig[0]["detector"] == "velocity"
        assert sig[0]["meta"] == {"short_rate": 0.01}
        assert "ts_monotonic_ns" in sig[0]

        assert grid and grid[0]["correlation_id"] == "cid-grid"

        assert entries and entries[0]["correlation_id"] == "cid-entry"
        assert entries[0]["meta"]["fire_price"] == 0.5
        assert entries[0]["meta"]["detectors"] == [
            "velocity", "news", "theta",
        ]

        assert exits and exits[0]["correlation_id"] == "cid-exit"
        assert exits[0]["pnl_usdc"] == pytest.approx(0.75)

        # audit_log.jsonl contains both log_block and log_audit records.
        reasons = {r.get("reason") for r in audits if r.get("reason")}
        events = {r.get("event") for r in audits if r.get("event")}
        assert "kill_switch" in reasons
        assert "runtime_reload" in events

    def test_rotation_preserves_records(self, tmp_path):
        from scripts.report import load_jsonl

        # Tiny rotation threshold so a handful of audits triggers it.
        led = LedgerStore(str(tmp_path), max_bytes=512)
        n = 200
        for i in range(n):
            led.log_audit("beat", i=i, padding="x" * 60)

        files = sorted(os.listdir(tmp_path))
        rotated = [
            f for f in files
            if f.startswith("audit_log.") and f != "audit_log.jsonl"
        ]
        assert rotated, "expected at least one rotation"

        total: List[dict] = []
        for f in rotated + ["audit_log.jsonl"]:
            total.extend(load_jsonl(tmp_path / f))
        assert len(total) == n
        indices = sorted(r["i"] for r in total)
        assert indices == list(range(n))

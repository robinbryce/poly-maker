"""Phase 5 tests: structured logging, metrics, smarter monitor."""

from __future__ import annotations

import json
import logging
import os
import threading
import time

import pytest


# --------------------------------------------------------------------
# Structured logging
# --------------------------------------------------------------------

class TestLoggingSetup:
    def test_configure_logging_installs_three_handlers(self, tmp_path):
        from grid.logging_setup import configure_logging

        root = configure_logging(level="DEBUG", log_dir=str(tmp_path))
        grid_handlers = [
            h for h in root.handlers if getattr(h, "_grid_managed", False)
        ]
        assert len(grid_handlers) == 3

    def test_configure_logging_is_idempotent(self, tmp_path):
        from grid.logging_setup import configure_logging

        configure_logging(log_dir=str(tmp_path))
        configure_logging(log_dir=str(tmp_path))
        root = logging.getLogger()
        grid_handlers = [
            h for h in root.handlers if getattr(h, "_grid_managed", False)
        ]
        assert len(grid_handlers) == 3

    def test_json_formatter_emits_expected_fields(self, tmp_path):
        from grid.logging_setup import configure_logging

        configure_logging(
            level="INFO", log_dir=str(tmp_path),
            install_stdout=False, install_human_file=False,
        )
        lg = logging.getLogger("grid.test")
        lg.info("hello", extra={"market": "mkt1", "count": 3})

        # Flush handlers so the file sees the record before we read.
        for h in logging.getLogger().handlers:
            h.flush()

        path = tmp_path / "grid.jsonl"
        assert path.exists()
        last = path.read_text().splitlines()[-1]
        rec = json.loads(last)
        for key in ("ts", "iso", "level", "logger", "message"):
            assert key in rec
        assert rec["level"] == "INFO"
        assert rec["logger"] == "grid.test"
        assert rec["message"] == "hello"
        assert rec["market"] == "mkt1"
        assert rec["count"] == 3


# --------------------------------------------------------------------
# Counters
# --------------------------------------------------------------------

class TestCounters:
    def test_plain_incr_and_snapshot(self):
        from grid.metrics import Counters

        c = Counters()
        c.incr("ws.reconnects")
        c.incr("ws.reconnects", n=2)
        assert c.get("ws.reconnects") == 3
        snap = c.snapshot()
        assert snap["plain"]["ws.reconnects"] == 3

    def test_labelled_incr_is_keyed_by_labels(self):
        from grid.metrics import Counters

        c = Counters()
        c.labelled_incr("http.status",
                        {"host": "api.coingecko.com", "code": "429"})
        c.labelled_incr("http.status",
                        {"host": "api.coingecko.com", "code": "429"})
        c.labelled_incr("http.status",
                        {"host": "api.coingecko.com", "code": "200"})
        assert c.get_labelled(
            "http.status",
            {"host": "api.coingecko.com", "code": "429"},
        ) == 2
        assert c.get_labelled(
            "http.status",
            {"host": "api.coingecko.com", "code": "200"},
        ) == 1
        snap = c.snapshot()
        labelled = snap["labelled"]["http.status"]
        # Two distinct label combos.
        assert len(labelled) == 2

    def test_write_snapshot_produces_valid_json(self, tmp_path):
        from grid.metrics import Counters

        c = Counters()
        c.incr("a", 5)
        c.labelled_incr("b", {"k": "v"}, n=7)
        path = tmp_path / "metrics.json"
        c.write_snapshot(str(path))
        data = json.loads(path.read_text())
        assert data["plain"]["a"] == 5
        assert data["labelled"]["b"][0]["labels"] == {"k": "v"}
        assert data["labelled"]["b"][0]["value"] == 7

    def test_thread_safety_smoke(self):
        from grid.metrics import Counters

        c = Counters()
        N_THREADS = 8
        N_INCR = 5000

        def worker():
            for _ in range(N_INCR):
                c.incr("x")
                c.labelled_incr("y", {"label": "one"})

        threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert c.get("x") == N_THREADS * N_INCR
        assert c.get_labelled("y", {"label": "one"}) == N_THREADS * N_INCR


# --------------------------------------------------------------------
# Monitor anomaly detection
# --------------------------------------------------------------------

def _samples(deltas):
    return [{"signal_fires_delta": int(d)} for d in deltas]


class TestMonitorAnomaly:
    def test_no_anomaly_on_steady_stream(self):
        from scripts.monitor import detect_anomaly

        # 100 fires per sample is steady.
        anomalies = detect_anomaly(_samples([100] * 15))
        assert anomalies == []

    def test_stuck_loop_flagged_on_spike_vs_median(self):
        from scripts.monitor import detect_anomaly

        # Median of prior ~100, latest 900 → 9× → over 5× threshold.
        deltas = [100] * 12 + [900]
        anomalies = detect_anomaly(_samples(deltas))
        kinds = [a["kind"] for a in anomalies]
        assert "stuck_loop_suspected" in kinds

    def test_stuck_loop_not_flagged_when_absolute_too_small(self):
        from scripts.monitor import detect_anomaly

        # Even if ratio is huge, the absolute delta must be material.
        deltas = [1] * 12 + [10]  # 10x median, but < 50 absolute
        anomalies = detect_anomaly(_samples(deltas))
        assert [a["kind"] for a in anomalies] == []

    def test_dead_loop_flagged_after_three_zeros(self):
        from scripts.monitor import detect_anomaly

        anomalies = detect_anomaly(_samples([50, 0, 0, 0]))
        assert "no_signal_flow" in [a["kind"] for a in anomalies]

    def test_no_dead_loop_when_latest_is_nonzero(self):
        from scripts.monitor import detect_anomaly

        anomalies = detect_anomaly(_samples([0, 0, 5]))
        assert "no_signal_flow" not in [a["kind"] for a in anomalies]

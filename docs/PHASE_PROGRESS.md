# Detector-grid remediation progress

Working document. Each phase commits as a discrete unit so we can
stop, review, and decide whether the next phase is safe to start.

## Where we are

* **Phase 1 committed at `cb796d8`.** Sixty-nine tests green.
  * Atomic ledger with fsync, per-file locks, size-based rotation
  * Asyncio-locked coordinator and paper executor
  * Correlation IDs flowing grid_fire → entry → exit
  * `ts_monotonic_ns` on every ledger record
  * `audit_log.jsonl` capturing every block decision with reason
  * Coordinator and paper-executor snapshot / restore
  * SIGTERM / SIGINT handlers that save state before exit
  * Periodic 60 s snapshots while running
  * Background thread replaced by asyncio tasks; HTTP polling via
    `asyncio.to_thread` so shared state is only mutated from the
    event loop
  * UTC-midnight daily reset independent of process uptime
  * `live_armed` second lock: live trading requires
    `mode == "live"` *and* `live_armed: true`

## What a validation run should demonstrate

The paper run after `cb796d8` should produce evidence for each
invariant before we move to Phase 2.

* `logs/grid-latest.log` → grid started, configured HTTP throttling,
  seeded tokens via gamma discovery.
* `ledger_data/signal_fires.jsonl` → every line parses as JSON,
  carries `ts_monotonic_ns`, no torn lines even under load.
* `ledger_data/audit_log.jsonl` → block entries from the
  coordinator explaining why grid fires didn't convert.
* `state/grid_state.json` → exists and updates roughly every 60 s.
* Full ledger round-trip: `uv run python scripts/report.py` reads
  every file cleanly.
* SIGTERM behaviour: sending `^C` writes a `shutdown` audit row and
  leaves a valid snapshot on disk.

## What to watch for as a regression

* Any truncated JSON line in the ledger.
* A Python `Traceback` in the grid window.
* `state/grid_state.json` not appearing or not updating.
* A second start immediately after a `^C` restart failing to
  `[grid] restored state from snapshot ...`.

## Remaining phases (not yet started)

### Phase 2 — live-safety gates
* Separate `DeploymentConfig` (immutable: mode, PK path, state dir)
  from `RuntimeConfig` (thresholds, risk caps).
* Pre-trade book-drift guard on `LiveExecutor`: abort if the book
  has moved more than X bps since the triggering signal.
* Config-driven audit-log channel with explicit event types.

### Phase 3 — honest paper
* Slippage: walk the book to fill size; refuse if price impact
  exceeds a configured threshold.
* Replace fixed `+10 % / -5 %` TP/SL with a strategy object.
* Per-category concentration caps enabled in `grid_config.json`.
* Pre-entry price-drift guard on both paper and live.

### Phase 4 — detector quality
* Raise `velocity_threshold` and add a minimum-book-depth gate.
* `signal_fire_quality` metric: fraction of fires that contribute
  to a grid fire.
* News detector weighted by time-to-resolution, not binary value.

### Phase 5 — observability
* Replace `print` with structured `logging`; JSON formatter.
* Size-rotating log handlers for `logs/grid-latest.log`.
* Counters module (per-detector fires, blocks-by-reason).
* Makefile / justfile with `test`, `run-paper`, `run-live`,
  `report`, `tail-ledger`.
* Monitor script detects "stuck in loop" via rolling-median fire
  rate.

### Phase 6 — test depth
* Integration test driving synthetic websocket events through the
  full stack, asserting ledger contents.
* Chaos test: crash between entry and exit, assert restart
  preserves `_open_markets` and reconciles.
* Property-based tests on coordinator invariants.
* Ledger writer → report reader round-trip.

## Go / no-go checklist before starting Phase 2

* Phase 1 validation run has produced a clean `audit_log.jsonl`
  with block reasons.
* `state/grid_state.json` has been written at least once.
* `^C` followed by a restart shows the "restored state from
  snapshot" banner.
* No `Traceback` in the grid window during the run.
* `uv run pytest tests/ -q` still reports 69 passed.

Tick the list, then: **"P2"** to start Phase 2.

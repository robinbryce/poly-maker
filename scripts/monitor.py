"""
Stability monitor for a running grid_main process.

Periodically snapshots:
  * whether grid_main is still alive (PID + uptime)
  * RSS memory of the process
  * ledger line counts (signal / grid / entry / exit fires)
  * log tail: total lines, recent 429 count, recent exception count
  * delta counters since last snapshot

Writes one compact JSON line per interval to ``logs/monitor.jsonl``
and prints a single human-readable line to stdout.  Designed to run
in its own tmux window next to grid_main so you can leave it going
for an hour and have a clean audit trail when you come back.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "ledger_data"
LOG = ROOT / "logs" / "grid-latest.log"
OUT = ROOT / "logs" / "monitor.jsonl"

LEDGERS = ("signal_fires", "grid_fires", "entries", "exits")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def _pgrep() -> tuple[int | None, float, int]:
    """Return (pid, uptime_secs, rss_kb) for the running python grid_main."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", r"python.*grid_main\.py"], text=True
        )
    except subprocess.CalledProcessError:
        return None, 0.0, 0

    pids = [int(p) for p in out.split() if p.isdigit()]
    if not pids:
        return None, 0.0, 0

    # Prefer the actual python process (lowest pid tends to be the shell wrapper)
    for pid in pids:
        try:
            ps = subprocess.check_output(
                ["ps", "-o", "etime=,rss=,command=", "-p", str(pid)], text=True
            ).strip()
        except subprocess.CalledProcessError:
            continue
        if "python" in ps and "grid_main" in ps:
            parts = ps.split(None, 2)
            etime = _parse_etime(parts[0])
            rss = int(parts[1])
            return pid, etime, rss
    return pids[0], 0.0, 0


def _parse_etime(s: str) -> float:
    # ps etime: [[dd-]hh:]mm:ss
    if "-" in s:
        days, rest = s.split("-", 1)
        days_f = float(days) * 86400
    else:
        days_f = 0.0
        rest = s
    parts = [float(x) for x in rest.split(":")]
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h = 0.0
        m, sec = parts
    else:
        return days_f + parts[0]
    return days_f + h * 3600 + m * 60 + sec


def _tail_counts(path: Path, since_line: int) -> dict:
    """Count interesting patterns in lines [since_line:] of the log."""
    out = {"new_lines": 0, "new_429": 0, "new_exceptions": 0,
           "new_failed": 0, "new_cooldown": 0}
    if not path.exists():
        return out
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            if i < since_line:
                continue
            out["new_lines"] += 1
            if "429" in line:
                out["new_429"] += 1
            if "Traceback" in line or "Error" in line:
                out["new_exceptions"] += 1
            if "failed:" in line:
                out["new_failed"] += 1
            if "cooldown" in line.lower():
                out["new_cooldown"] += 1
    return out


def snapshot(prev: dict | None) -> dict:
    pid, uptime, rss = _pgrep()
    now_ts = time.time()
    iso = datetime.now(timezone.utc).isoformat(timespec="seconds")

    ledger_counts = {name: count_lines(LEDGER / f"{name}.jsonl") for name in LEDGERS}

    log_total = count_lines(LOG)
    since = (prev or {}).get("log_total", 0)
    tail = _tail_counts(LOG, since)

    deltas = {}
    if prev:
        for name in LEDGERS:
            deltas[f"{name}_delta"] = ledger_counts[name] - prev.get(name, 0)

    return {
        "ts": now_ts,
        "iso": iso,
        "pid": pid,
        "alive": pid is not None,
        "uptime_secs": uptime,
        "rss_kb": rss,
        **ledger_counts,
        **deltas,
        "log_total": log_total,
        **tail,
    }


def fmt(snap: dict) -> str:
    alive = "alive" if snap["alive"] else "DEAD"
    return (
        f"{snap['iso']} {alive:5s} pid={snap['pid']} up={snap['uptime_secs']:.0f}s "
        f"rss={snap['rss_kb']}kB | "
        f"fires={snap['signal_fires']}(+{snap.get('signal_fires_delta', 0)}) "
        f"grid={snap['grid_fires']}(+{snap.get('grid_fires_delta', 0)}) "
        f"entries={snap['entries']}(+{snap.get('entries_delta', 0)}) "
        f"exits={snap['exits']}(+{snap.get('exits_delta', 0)}) | "
        f"log_lines=+{snap['new_lines']} 429=+{snap['new_429']} "
        f"fails=+{snap['new_failed']} exc=+{snap['new_exceptions']} "
        f"cool=+{snap['new_cooldown']}"
    )


# ── anomaly detection (pure helpers, unit-tested) ─────────────────

STUCK_WINDOW = 12       # number of prior samples for the median.
STUCK_MULT = 5.0        # current delta > mult * median = suspicious.
STUCK_MIN_ABS = 50      # but only if the delta is materially large.
DEAD_CONSEC = 3         # consecutive zero-delta samples = dead loop.


def detect_anomaly(samples: list[dict]) -> list[dict]:
    """Inspect the tail of ``samples`` (oldest first) and return any
    anomalies triggered by the latest sample.

    Each anomaly is ``{"kind": str, "detail": str, "metrics": {...}}``
    where ``kind`` is ``"stuck_loop_suspected"`` or
    ``"no_signal_flow"``.
    """
    out: list[dict] = []
    if not samples:
        return out
    latest = samples[-1]
    delta = int(latest.get("signal_fires_delta", 0) or 0)

    # Stuck loop: latest delta >> rolling median of prior deltas.
    prior = [
        int(s.get("signal_fires_delta", 0) or 0)
        for s in samples[-(STUCK_WINDOW + 1):-1]
    ]
    if prior:
        med = _median(prior)
        if (
            delta >= STUCK_MIN_ABS
            and med > 0
            and delta > med * STUCK_MULT
        ):
            out.append({
                "kind": "stuck_loop_suspected",
                "detail": (
                    f"signal_fires_delta={delta} > {STUCK_MULT:.0f}× "
                    f"median({med:.0f}) over last {len(prior)} samples"
                ),
                "metrics": {
                    "delta": delta, "median": med,
                    "samples": len(prior),
                },
            })

    # Dead loop: last N samples all zero delta.
    tail = samples[-DEAD_CONSEC:]
    if len(tail) == DEAD_CONSEC and all(
        int(s.get("signal_fires_delta", 0) or 0) == 0 for s in tail
    ):
        out.append({
            "kind": "no_signal_flow",
            "detail": (
                f"signal_fires_delta was zero for the last "
                f"{DEAD_CONSEC} samples"
            ),
            "metrics": {"consecutive_zero": DEAD_CONSEC},
        })

    return out


def _median(xs: list[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2:
        return float(s[mid])
    return (s[mid - 1] + s[mid]) / 2.0


# ── main loop ────────────────────────────────────────────────────────────

def main(interval: float = 60.0, *, anomaly: bool = True) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = None
    history: list[dict] = []
    print(f"[monitor] writing snapshots to {OUT} every {interval:.0f}s "
          f"anomaly={anomaly}")
    while True:
        try:
            snap = snapshot(prev)
        except Exception as exc:
            snap = {"ts": time.time(), "iso": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc)}
        with open(OUT, "a") as f:
            f.write(json.dumps(snap) + "\n")
        if "error" in snap:
            print(f"[monitor] error: {snap['error']}")
        else:
            print(fmt(snap))
            if anomaly:
                history.append(snap)
                # Keep the tail bounded.
                if len(history) > STUCK_WINDOW + 4:
                    history = history[-(STUCK_WINDOW + 4):]
                for a in detect_anomaly(history):
                    print(f"[monitor] ANOMALY {a['kind']}: {a['detail']}")
        prev = snap
        time.sleep(interval)


if __name__ == "__main__":
    args = sys.argv[1:]
    interval = float(args[0]) if args and args[0].replace(".", "").isdigit() else 60.0
    anomaly = "--no-anomaly" not in args
    main(interval, anomaly=anomaly)

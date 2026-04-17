"""
Paper-mode financial performance report.

Reads the JSON-lines ledger written by the grid and prints a compact
summary of:

  * signal funnel (signal fires -> grid fires -> entries -> exits)
  * realised PnL stats (win rate, avg/max win/loss, hold time, streaks)
  * max drawdown (on the realised equity curve)
  * mark-to-market of any open paper positions (live book fetch)
  * "what-if" replay at different min-signal thresholds, to answer
    "would we have traded more if we'd set min_signals=2?"

The last piece is crucial during the paper run: if the grid is firing
zero trades, we need to know whether that's structural (no signals
ever coincide) or just a conservative threshold.

Usage:
    uv run python scripts/report.py
    uv run python scripts/report.py --replay 2
    uv run python scripts/report.py --ledger-dir ledger_data
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out: List[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


# ── section: signal funnel ──────────────────────────────────────────

def signal_funnel(signal_fires, grid_fires, entries, exits) -> dict:
    per_det = collections.Counter(f["detector"] for f in signal_fires)
    by_dir = collections.Counter(f["direction"] for f in signal_fires)
    by_mkt = collections.Counter(f["market"] for f in signal_fires)
    return {
        "signal_fires":   len(signal_fires),
        "by_detector":    dict(per_det),
        "by_direction":   dict(by_dir),
        "markets_firing": len(by_mkt),
        "grid_fires":     len(grid_fires),
        "entries":        len(entries),
        "exits":          len(exits),
        "conversion_grid_to_entry": (
            len(entries) / len(grid_fires) if grid_fires else 0.0
        ),
        "conversion_entry_to_exit": (
            len(exits) / len(entries) if entries else 0.0
        ),
    }


# ── section: realised PnL ───────────────────────────────────────────

def realised_pnl_stats(exits: List[dict]) -> dict:
    if not exits:
        return {"count": 0}

    pnls = [float(e["pnl_usdc"]) for e in exits]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    # Streaks
    cur_streak = 0
    best_win_streak = 0
    worst_loss_streak = 0
    for p in pnls:
        if p > 0:
            cur_streak = cur_streak + 1 if cur_streak > 0 else 1
            best_win_streak = max(best_win_streak, cur_streak)
        elif p < 0:
            cur_streak = cur_streak - 1 if cur_streak < 0 else -1
            worst_loss_streak = min(worst_loss_streak, cur_streak)
        else:
            cur_streak = 0

    # Max drawdown on the realised equity curve
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

    return {
        "count":            len(pnls),
        "win_rate":         len(wins) / len(pnls),
        "gross_profit":     sum(wins),
        "gross_loss":       sum(losses),
        "net_pnl_usdc":     sum(pnls),
        "best_trade":       max(pnls),
        "worst_trade":      min(pnls),
        "avg_win":          statistics.mean(wins) if wins else 0.0,
        "avg_loss":         statistics.mean(losses) if losses else 0.0,
        "median_pnl":       statistics.median(pnls),
        "profit_factor":    (sum(wins) / -sum(losses)) if losses else float("inf"),
        "max_drawdown":     max_dd,
        "best_win_streak":  best_win_streak,
        "worst_loss_streak": abs(worst_loss_streak),
    }


def hold_time_stats(entries: List[dict], exits: List[dict]) -> dict:
    """Match exits to their entries by market + direction and compute
    hold durations in seconds."""
    by_market: Dict[str, List[dict]] = collections.defaultdict(list)
    for e in entries:
        by_market[e["market"]].append(e)

    holds: List[float] = []
    for x in exits:
        pool = by_market.get(x["market"], [])
        if not pool:
            continue
        # FIFO — earliest open entry for this market.
        entry = pool.pop(0)
        holds.append(float(x["ts"]) - float(entry["ts"]))

    if not holds:
        return {"count": 0}
    return {
        "count":       len(holds),
        "min_secs":    min(holds),
        "max_secs":    max(holds),
        "mean_secs":   statistics.mean(holds),
        "median_secs": statistics.median(holds),
    }


# ── section: mark-to-market of open paper positions ────────────────

def mark_to_market(entries, exits) -> dict:
    """Figure out which paper positions are still open (entry present,
    no matching exit yet) and value them against the current order
    book on Polymarket."""
    by_market: Dict[str, List[dict]] = collections.defaultdict(list)
    for e in entries:
        by_market[e["market"]].append(e)
    for x in exits:
        pool = by_market.get(x["market"], [])
        if pool:
            pool.pop(0)

    open_positions = [p for lst in by_market.values() for p in lst]
    if not open_positions:
        return {"open": 0}

    from grid import http_client  # lazy to avoid import cost when no opens

    total_unrealised = 0.0
    details = []
    for p in open_positions:
        entry_price = float(p["price"])
        size = float(p["size"])
        direction = p["direction"]
        token_id = p.get("token_id")

        mark_price = None
        if token_id:
            try:
                resp = http_client.get(
                    "https://clob.polymarket.com/midpoint",
                    params={"token_id": token_id}, timeout=10,
                )
                if resp.status_code == 200:
                    mark_price = float(resp.json().get("mid", 0))
            except Exception:
                pass

        if mark_price is None:
            continue
        if direction == "BUY":
            pnl = (mark_price - entry_price) * size
        else:
            pnl = (entry_price - mark_price) * size
        total_unrealised += pnl
        details.append({
            "market":      p["market"],
            "direction":   direction,
            "size":        size,
            "entry_price": entry_price,
            "mark_price":  mark_price,
            "unrealised":  pnl,
        })

    return {
        "open":        len(open_positions),
        "marked":      len(details),
        "unrealised_total_usdc": total_unrealised,
        "positions":   details,
    }


# ── section: what-if replay at a different min-signals ─────────────

def replay(
    signal_fires: List[dict],
    *,
    min_signals: int,
    staleness_secs: float = 300.0,
    direction_threshold: float = 0.6,
) -> dict:
    """Re-run the coordinator logic offline against the recorded
    signal_fires with a different min_signals threshold.  Returns how
    many grid fires would have been triggered and on which markets.
    Does **not** replay executor behaviour (no PnL estimate)."""
    fires = sorted(signal_fires, key=lambda f: f["ts"])
    state: Dict[str, Dict[str, dict]] = collections.defaultdict(dict)  # market -> detector -> fire
    open_markets: set = set()
    triggers: List[dict] = []

    for f in fires:
        market = f["market"]
        state[market][f["detector"]] = f

        # Prune stale
        now_ts = f["ts"]
        active = {
            d: s for d, s in state[market].items()
            if (now_ts - s["ts"]) < staleness_secs
        }
        state[market] = active

        if market in open_markets:
            continue
        if len(active) < min_signals:
            continue
        buy = sum(1 for s in active.values() if s["direction"] == "BUY")
        sell = len(active) - buy
        total = len(active)
        if buy > sell:
            direction = "BUY"
            agreement = buy / total
        elif sell > buy:
            direction = "SELL"
            agreement = sell / total
        else:
            continue
        if agreement < direction_threshold:
            continue

        open_markets.add(market)
        triggers.append({
            "ts":        now_ts,
            "market":    market,
            "direction": direction,
            "n_signals": len(active),
            "detectors": list(active.keys()),
        })

    return {
        "min_signals": min_signals,
        "triggers":    len(triggers),
        "markets":     len({t["market"] for t in triggers}),
        "sample":      triggers[:5],
    }


# ── formatting ──────────────────────────────────────────────────────

def _section(title: str) -> None:
    print(f"\n── {title} ─" + "─" * max(0, 60 - len(title)))


def _usd(v: float) -> str:
    return f"${v:,.2f}"


def print_report(ledger_dir: Path, replay_with: Optional[List[int]] = None) -> None:
    signal_fires = load_jsonl(ledger_dir / "signal_fires.jsonl")
    grid_fires   = load_jsonl(ledger_dir / "grid_fires.jsonl")
    entries      = load_jsonl(ledger_dir / "entries.jsonl")
    exits        = load_jsonl(ledger_dir / "exits.jsonl")

    funnel = signal_funnel(signal_fires, grid_fires, entries, exits)
    pnl = realised_pnl_stats(exits)
    holds = hold_time_stats(entries, exits)
    mtm = mark_to_market(entries, exits)

    _section("Signal funnel")
    print(f"signal_fires:    {funnel['signal_fires']:,}")
    print(f"  by detector:   {funnel['by_detector']}")
    print(f"  by direction:  {funnel['by_direction']}")
    print(f"  distinct markets firing: {funnel['markets_firing']}")
    print(f"grid_fires:      {funnel['grid_fires']}")
    print(f"entries:         {funnel['entries']}  "
          f"(grid_fires -> entries: {funnel['conversion_grid_to_entry']*100:.0f}%)")
    print(f"exits:           {funnel['exits']}  "
          f"(entries -> exits: {funnel['conversion_entry_to_exit']*100:.0f}%)")

    _section("Realised PnL (paper)")
    if pnl["count"] == 0:
        print("no closed trades yet")
    else:
        print(f"trades:          {pnl['count']}")
        print(f"win rate:        {pnl['win_rate']*100:.1f}%")
        print(f"net PnL:         {_usd(pnl['net_pnl_usdc'])}")
        print(f"gross profit:    {_usd(pnl['gross_profit'])}")
        print(f"gross loss:      {_usd(pnl['gross_loss'])}")
        print(f"avg win:         {_usd(pnl['avg_win'])}")
        print(f"avg loss:        {_usd(pnl['avg_loss'])}")
        print(f"best trade:      {_usd(pnl['best_trade'])}")
        print(f"worst trade:     {_usd(pnl['worst_trade'])}")
        print(f"profit factor:   {pnl['profit_factor']:.2f}")
        print(f"max drawdown:    {_usd(pnl['max_drawdown'])}")
        print(f"streaks (W/L):   {pnl['best_win_streak']} / {pnl['worst_loss_streak']}")

    _section("Hold time")
    if holds["count"] == 0:
        print("no closed trades yet")
    else:
        print(f"trades:          {holds['count']}")
        print(f"mean hold:       {holds['mean_secs']/60:.1f} min")
        print(f"median hold:     {holds['median_secs']/60:.1f} min")
        print(f"min / max hold:  {holds['min_secs']/60:.1f} / {holds['max_secs']/60:.1f} min")

    _section("Open paper positions (mark-to-market)")
    if mtm["open"] == 0:
        print("no open paper positions")
    else:
        print(f"open positions:  {mtm['open']}")
        print(f"marked:          {mtm['marked']}")
        print(f"unrealised:      {_usd(mtm['unrealised_total_usdc'])}")
        for p in mtm["positions"]:
            print(f"  {p['direction']} {p['market'][:12]}…  "
                  f"entry={p['entry_price']:.3f} mark={p['mark_price']:.3f} "
                  f"size={p['size']:.1f}  pnl={_usd(p['unrealised'])}")

    if replay_with:
        _section("What-if replay (offline)")
        print("Replays the recorded signal_fires with a different min_signals")
        print("to answer: 'is the grid too conservative?'")
        print()
        for n in replay_with:
            r = replay(signal_fires, min_signals=n)
            print(f"min_signals={n}:  would have triggered {r['triggers']} grid fire(s) "
                  f"on {r['markets']} market(s)")
        if signal_fires:
            # Show a sample at the lowest replay threshold
            r = replay(signal_fires, min_signals=min(replay_with))
            if r["sample"]:
                print()
                print(f"Sample (min_signals={min(replay_with)}):")
                for t in r["sample"]:
                    print(f"  {t['direction']} on {t['market'][:18]}…  "
                          f"n={t['n_signals']} detectors={t['detectors']}")


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ledger-dir", default="ledger_data",
                   help="Directory containing the JSON-lines ledger files.")
    p.add_argument("--replay", nargs="*", type=int, default=[2, 3, 4],
                   help="min_signals values to replay.")
    args = p.parse_args(argv)
    print_report(ROOT / args.ledger_dir, replay_with=args.replay)


if __name__ == "__main__":
    main()

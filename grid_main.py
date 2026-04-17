"""
grid_main.py — separate entrypoint for the detector grid.

Runs alongside (or instead of) the existing market-maker main.py.
Connects to the same Polymarket websocket feeds, fans events to all
eight detectors, and enters positions when >=3 signals align.

Starts in **paper mode** by default.  Set GRID_MODE=live (or supply
a grid_config.json with "mode": "live") to enable real execution.
"""

import asyncio
import gc
import json
import os
import sys
import threading
import time
import traceback

from dotenv import load_dotenv

load_dotenv()

# ── config ──────────────────────────────────────────────────────────

from grid.config import GridConfig

_cfg_path = os.environ.get("GRID_CONFIG", "grid_config.json")
if os.path.isfile(_cfg_path):
    config = GridConfig.from_json(_cfg_path)
    print(f"[grid] loaded config from {_cfg_path}")
else:
    config = GridConfig.from_env()
    print("[grid] loaded config from environment (defaults where unset)")

print(f"[grid] mode={config.mode}  kill_switch={config.kill_switch}  "
      f"min_signals={config.min_signals}")

# ── infrastructure ──────────────────────────────────────────────────

from ledger.store import LedgerStore

ledger = LedgerStore(config.ledger_dir)

# ── detectors ───────────────────────────────────────────────────────

from detectors.volume import VolumeDetector
from detectors.velocity import VelocityDetector
from detectors.disposition import DispositionDetector
from detectors.news import NewsDetector
from detectors.cross_market import CrossMarketDetector
from detectors.whale import WhaleDetector
from detectors.category import CategoryDetector
from detectors.theta import ThetaDetector

volume_det = VolumeDetector(config)
velocity_det = VelocityDetector(config)
disposition_det = DispositionDetector(config)
news_det = NewsDetector(config)
cross_market_det = CrossMarketDetector(config)
whale_det = WhaleDetector(config)
category_det = CategoryDetector(config)
theta_det = ThetaDetector(config)

all_detectors = [
    volume_det, velocity_det, disposition_det,
    news_det, cross_market_det,
    whale_det, category_det, theta_det,
]

# ── event bus ───────────────────────────────────────────────────────

from grid.event_bus import EventBus

bus = EventBus(all_detectors)

# ── executor ────────────────────────────────────────────────────────

from detectors.base import Direction
from executor.paper import PaperExecutor
from executor.live import LiveExecutor

paper_exec = PaperExecutor(config, ledger)
live_exec = LiveExecutor(config, ledger)


def on_entry(market: str, token_id: str, direction: Direction,
             confidence: float, meta: dict) -> None:
    """Dispatch to the appropriate executor based on mode."""
    if config.mode == "live":
        live_exec.enter(market, token_id, direction, confidence, meta)
    else:
        paper_exec.enter(market, token_id, direction, confidence, meta)


# ── coordinator ─────────────────────────────────────────────────────

from grid.coordinator import Coordinator

coordinator = Coordinator(config, on_entry, ledger)
bus.set_fire_callback(coordinator.ingest)

# ── feeds ───────────────────────────────────────────────────────────

from feeds.gamma import GammaPoller
from feeds.oracle import OraclePoller
from feeds.cross_market import CrossMarketPoller
from feeds.whale import WhalePoller

# The tracked markets list will be populated from global_state after
# the initial data load.
import poly_data.global_state as global_state

gamma_poller = GammaPoller([], category_det, theta_det)
oracle_poller = OraclePoller(config, news_det)
cross_market_poller = CrossMarketPoller(config, cross_market_det)
whale_poller = WhalePoller(config, whale_det)

# ── websocket integration ───────────────────────────────────────────

import websockets

from poly_data.polymarket_client import PolymarketClient
from poly_data.data_utils import update_markets, update_positions, update_orders
from poly_data.data_processing import process_data, process_user_data


def _enrich_event(json_data: dict) -> dict:
    """Add midpoint to events when we can compute it from the book."""
    market = json_data.get("market", "")
    book = global_state.all_data.get(market)
    if book and "midpoint" not in json_data:
        bids = book.get("bids")
        asks = book.get("asks")
        if bids and asks and len(bids) > 0 and len(asks) > 0:
            best_bid = float(max(bids.keys()))
            best_ask = float(min(asks.keys()))
            json_data["midpoint"] = (best_bid + best_ask) / 2
    return json_data


async def connect_grid_market_ws(tokens: list) -> None:
    """Market websocket that feeds both the existing processor and the
    detector grid event bus."""
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as ws:
        await ws.send(json.dumps({"assets_ids": tokens}))
        print(f"[grid] subscribed to {len(tokens)} market tokens")
        try:
            while True:
                raw = await ws.recv()
                data = json.loads(raw)
                # Feed existing market-maker processor
                process_data(data, trade=False)
                # Feed the detector bus
                if isinstance(data, list):
                    for d in data:
                        bus.dispatch(_enrich_event(d))
                else:
                    bus.dispatch(_enrich_event(data))
        except websockets.ConnectionClosed:
            print("[grid] market ws closed")
        except Exception as exc:
            print(f"[grid] market ws error: {exc}")
            traceback.print_exc()
        finally:
            await asyncio.sleep(5)


async def connect_grid_user_ws() -> None:
    """User websocket that feeds both the existing processor and the
    detector bus (for trade events → disposition / volume detectors)."""
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as ws:
        msg = {
            "type": "user",
            "auth": {
                "apiKey": global_state.client.client.creds.api_key,
                "secret": global_state.client.client.creds.api_secret,
                "passphrase": global_state.client.client.creds.api_passphrase,
            },
        }
        await ws.send(json.dumps(msg))
        print("[grid] user ws authenticated")
        try:
            while True:
                raw = await ws.recv()
                rows = json.loads(raw)
                process_user_data(rows)
                # Forward trade events to detectors
                if isinstance(rows, list):
                    for r in rows:
                        if r.get("event_type") == "trade":
                            bus.dispatch(r)
                elif isinstance(rows, dict) and rows.get("event_type") == "trade":
                    bus.dispatch(rows)
        except websockets.ConnectionClosed:
            print("[grid] user ws closed")
        except Exception as exc:
            print(f"[grid] user ws error: {exc}")
            traceback.print_exc()
        finally:
            await asyncio.sleep(5)


# ── periodic background tasks ───────────────────────────────────────

def _background_polling() -> None:
    """Background thread: polls feeds, checks paper exits, resets daily."""
    last_daily_reset = time.time()
    while True:
        time.sleep(10)
        try:
            # Feed polls
            gamma_poller.poll()
            oracle_poller.poll()
            cross_market_poller.poll()
            whale_poller.poll()

            # Let polling detectors evaluate
            bus.poll_all()

            # Paper exit checks
            closed = paper_exec.check_exits()
            for market, pnl in closed:
                coordinator.mark_closed(market, pnl)

            # Daily risk reset
            if time.time() - last_daily_reset > 86400:
                coordinator.reset_daily()
                last_daily_reset = time.time()
                print("[grid] daily risk counters reset")

        except Exception:
            print("[grid] background polling error")
            traceback.print_exc()
        gc.collect()


# ── main ────────────────────────────────────────────────────────────

async def main() -> None:
    # Initialise the Polymarket client and load initial state.
    global_state.client = PolymarketClient()
    global_state.all_tokens = []
    update_markets()
    update_positions()
    update_orders()

    # Populate tracked markets for the gamma poller.
    tracked = []
    if global_state.df is not None:
        tracked = global_state.df["condition_id"].tolist()
    gamma_poller._tracked = tracked

    print(f"[grid] {len(global_state.all_tokens)} tokens, "
          f"{len(tracked)} tracked markets")

    # Start background polling thread.
    t = threading.Thread(target=_background_polling, daemon=True)
    t.start()

    # Main loop: maintain websocket connections.
    while True:
        try:
            await asyncio.gather(
                connect_grid_market_ws(global_state.all_tokens),
                connect_grid_user_ws(),
            )
        except Exception:
            traceback.print_exc()
        await asyncio.sleep(1)
        gc.collect()


if __name__ == "__main__":
    asyncio.run(main())

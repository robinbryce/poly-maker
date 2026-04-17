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
from grid import http_client
from grid.snapshot import SnapshotStore

ledger = LedgerStore(config.ledger_dir)
snapshot_store = SnapshotStore(config.state_dir)

# Install the configured rate-limited HTTP session so every feed
# module's ``http_client.get(...)`` respects the per-host throttles.
http_client.configure(config.http_throttle)
print(f"[grid] http_throttle configured for "
      f"{len([h for h in config.http_throttle if h != 'default'])} hosts "
      f"(plus default)")

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


async def on_entry(market: str, token_id: str, direction: Direction,
                   confidence: float, meta: dict, correlation_id: str) -> None:
    """Dispatch to the appropriate executor based on mode."""
    if config.mode == "live":
        live_exec.enter(market, token_id, direction, confidence, meta,
                        correlation_id)
    else:
        await paper_exec.enter(market, token_id, direction, confidence, meta,
                               correlation_id)


# ── coordinator ─────────────────────────────────────────────────────

from grid.coordinator import Coordinator

coordinator = Coordinator(config, on_entry, ledger)

# The event bus fires into a sync callback, so we wrap the async
# coordinator.ingest in a task-scheduling shim.
def _bus_fire_callback(fires):
    loop = asyncio.get_event_loop()
    asyncio.ensure_future(coordinator.ingest(fires))

bus.set_fire_callback(_bus_fire_callback)

# ── feeds ───────────────────────────────────────────────────────────

from feeds.gamma_discovery import GammaDiscoveryPoller
from feeds.oracle import OraclePoller
from feeds.cross_market import CrossMarketPoller
from feeds.whale import WhalePoller

# The tracked markets list will be populated from global_state after
# the initial data load.
import poly_data.global_state as global_state

oracle_poller = OraclePoller(config, news_det)
cross_market_poller = CrossMarketPoller(config, cross_market_det)
whale_poller = WhalePoller(config, whale_det)
gamma_discovery = GammaDiscoveryPoller(
    oracle_poller, news_det,
    theta_detector=theta_det,
    category_detector=category_det,
)

# Register any oracle mappings declared in the config file so the news
# detector has something to compare market midpoints against.
for _cid, _spec in config.oracle_mappings.items():
    _kwargs = {k: v for k, v in _spec.items() if k != "source"}
    oracle_poller.set_mapping(_cid, _spec.get("source", "coingecko"), **_kwargs)
if config.oracle_mappings:
    print(f"[grid] registered {len(config.oracle_mappings)} oracle mappings")

# ── websocket integration ───────────────────────────────────────────

import websockets

from poly_data.data_processing import process_data, process_user_data
# Live mode (mode="live") requires the authenticated client + sheet
# bootstrap.  Paper mode uses a credential-free public client.
if config.mode == "live":
    from poly_data.polymarket_client import PolymarketClient
    from poly_data.data_utils import (
        update_markets, update_positions, update_orders,
    )
from poly_data.public_client import PublicPolymarketClient


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
    detector grid event bus.  Closes itself when the discovery poller
    signals that new subscriptions are pending."""
    uri = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    async with websockets.connect(uri, ping_interval=5, ping_timeout=None) as ws:
        await ws.send(json.dumps({"assets_ids": tokens}))
        print(f"[grid] subscribed to {len(tokens)} market tokens")

        async def _watch_dirty() -> None:
            """Close the ws when gamma_discovery has new tokens to
            subscribe to, forcing the main loop to reconnect."""
            while True:
                await asyncio.sleep(5)
                if gamma_discovery.subscription_dirty.is_set():
                    gamma_discovery.subscription_dirty.clear()
                    print("[grid] subscription update pending — reconnecting")
                    await ws.close()
                    return

        watcher = asyncio.create_task(_watch_dirty())
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
            watcher.cancel()
            await asyncio.sleep(1)


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

async def _periodic_pollers() -> None:
    """Async task: drives feed polling via to_thread (blocking HTTP
    stays off the event loop), checks paper exits on the loop so
    coordinator/paper state is only mutated from asyncio."""
    last_discovery = 0.0
    DISCOVERY_INTERVAL = 300.0
    while True:
        await asyncio.sleep(10)
        try:
            now = time.time()
            if now - last_discovery > DISCOVERY_INTERVAL:
                await asyncio.to_thread(gamma_discovery.poll)
                last_discovery = now

            await asyncio.to_thread(oracle_poller.poll)
            await asyncio.to_thread(cross_market_poller.poll)
            await asyncio.to_thread(whale_poller.poll)
            await asyncio.to_thread(bus.poll_all)

            closed = await paper_exec.check_exits()
            for market, pnl in closed:
                await coordinator.mark_closed(market, pnl)

            # Wall-clock daily reset is handled inside coordinator.ingest
            # now, not here.
        except Exception:
            print("[grid] periodic poller error")
            traceback.print_exc()
        gc.collect()


async def _periodic_snapshot() -> None:
    """Every 60s write the current grid state to disk so a crash
    restart resumes correctly."""
    while True:
        await asyncio.sleep(60)
        try:
            _save_snapshot()
        except Exception:
            traceback.print_exc()


def _save_snapshot() -> None:
    snapshot_store.save({
        "coordinator": coordinator.snapshot(),
        "paper_executor": paper_exec.snapshot(),
    })


# ── main ────────────────────────────────────────────────────────────

async def main() -> None:
    global_state.all_tokens = []

    if config.mode == "live":
        # Live mode: needs the authenticated client, sheet-driven markets,
        # and the user websocket for fill tracking.
        global_state.client = PolymarketClient()
        update_markets()
        update_positions()
        update_orders()

        tracked = []
        if global_state.df is not None:
            tracked = global_state.df["condition_id"].tolist()
        print(f"[grid] live mode: {len(global_state.all_tokens)} tokens, "
              f"{len(tracked)} tracked markets")
    else:
        # Paper mode: no PK, no sheet.  Seed the token list with a
        # blocking discovery pass so the market ws has something to
        # subscribe to on first connect.
        global_state.client = PublicPolymarketClient()
        print("[grid] paper mode: no credentials required, "
              "bootstrapping via gamma discovery…")
        gamma_discovery.poll()
        # The ws hasn't started yet, so the dirty flag set by the
        # bootstrap discovery would trigger a pointless reconnect on
        # first tick — clear it now.
        gamma_discovery.subscription_dirty.clear()
        print(f"[grid] paper mode: seeded {len(global_state.all_tokens)} tokens "
              f"from gamma_discovery")
        if not global_state.all_tokens:
            print("[grid] WARNING: no tokens discovered — the market ws will "
                  "idle until discovery finds a qualifying threshold market.")

    # Restore state if a prior run left a snapshot.
    prior = snapshot_store.load()
    if prior:
        try:
            coordinator.restore(prior.get("coordinator", {}))
            paper_exec.restore(prior.get("paper_executor", {}))
            print(f"[grid] restored state from snapshot: "
                  f"open_markets={len(coordinator._open_markets)} "
                  f"paper_positions={paper_exec.open_count}")
        except Exception:
            print("[grid] failed to restore snapshot")
            traceback.print_exc()

    # Install signal handlers to drain on SIGTERM / SIGINT.
    import signal
    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    def _request_shutdown(sig_name):
        print(f"[grid] received {sig_name}, saving snapshot and exiting")
        try:
            _save_snapshot()
            ledger.log_audit("shutdown", signal=sig_name)
        finally:
            stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown, sig.name)
        except NotImplementedError:
            # Windows or some restricted envs.
            pass

    # Async background tasks.
    poll_task = asyncio.create_task(_periodic_pollers())
    snap_task = asyncio.create_task(_periodic_snapshot())

    async def ws_loop():
        while not stop_event.is_set():
            try:
                tasks = [connect_grid_market_ws(global_state.all_tokens)]
                if config.mode == "live":
                    tasks.append(connect_grid_user_ws())
                await asyncio.gather(*tasks)
            except Exception:
                traceback.print_exc()
            await asyncio.sleep(1)

    ws_task = asyncio.create_task(ws_loop())
    await stop_event.wait()
    for t in (poll_task, snap_task, ws_task):
        t.cancel()
    # Final flush.
    _save_snapshot()


if __name__ == "__main__":
    asyncio.run(main())

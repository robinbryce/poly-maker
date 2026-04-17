"""
Grid configuration.

Loadable from environment variables or a JSON file.  Every field has a
safe default so the grid can start in paper / read-only mode without
any external config.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class GridConfig:
    # ── execution mode ──────────────────────────────────────────────
    # "paper" (default) or "live".  Live requires explicit opt-in.
    mode: str = "paper"

    # Emergency stop: when True the grid will not open new positions
    # and will cancel any pending grid-managed orders.
    kill_switch: bool = False

    # ── signal thresholds ───────────────────────────────────────────
    # Minimum number of aligned, non-stale signals to trigger entry.
    min_signals: int = 3

    # Seconds after which a signal fire is considered stale.
    signal_staleness_secs: float = 300.0

    # Minimum directional agreement ratio (0-1) among active signals.
    direction_threshold: float = 0.6

    # ── risk limits ─────────────────────────────────────────────────
    # Maximum USDC notional per grid-managed entry.
    max_entry_usdc: float = 25.0

    # Maximum number of concurrent grid-managed positions.
    max_open_positions: int = 5

    # Daily cumulative loss cap in USDC.  Grid pauses when hit.
    daily_loss_cap_usdc: float = 100.0

    # Consecutive losing trades before the grid pauses.
    consecutive_loss_cap: int = 5

    # ── detector-specific knobs ─────────────────────────────────────
    # Volume: multiplier over rolling baseline to fire.
    volume_spike_multiplier: float = 3.0

    # Velocity: minimum absolute price change per second (normalised).
    velocity_threshold: float = 0.002

    # Disposition: net taker aggression ratio to fire.
    disposition_threshold: float = 0.6

    # Cross-market: minimum divergence in cents to fire.
    cross_market_delta_cents: float = 5.0

    # News / oracle: minimum feed-vs-market delta in cents to fire.
    news_delta_cents: float = 5.0

    # Theta: hours-before-resolution pressure window.
    theta_hours: float = 48.0

    # ── whale detector ──────────────────────────────────────────────
    # Full wallet addresses to watch.  Empty until the operator
    # supplies them — the whale detector is a no-op until then.
    whale_wallets: List[str] = field(default_factory=list)

    # ── cross-market reference map ──────────────────────────────────
    # Maps a Polymarket condition_id to an external reference:
    #   { "condition_abc": {"source": "coingecko", "pair": "bitcoin"} }
    cross_market_refs: Dict[str, dict] = field(default_factory=dict)

    # ── oracle mappings for the news detector ───────────────────────
    # Maps a Polymarket condition_id to an oracle source descriptor:
    #   { "0x<cid>": {"source": "coingecko", "id": "bitcoin", "threshold": 62000} }
    oracle_mappings: Dict[str, dict] = field(default_factory=dict)

    # ── feed poll intervals (seconds) ───────────────────────────────
    gamma_poll_interval: float = 60.0
    oracle_poll_interval: float = 30.0
    whale_poll_interval: float = 30.0

    # ── ledger ──────────────────────────────────────────────────────
    ledger_dir: str = "ledger_data"

    # ── HTTP throttling (see grid/http_client.py) ───────────────────
    # Mapping of host -> ThrottleConfig fields.  The reserved "default"
    # key supplies the fallback for any unlisted host.  Sensible
    # defaults are baked in so paper mode works out of the box.
    http_throttle: Dict[str, dict] = field(default_factory=lambda: {
        "default": {
            "min_interval_secs": 1.0,
            "max_retries": 4,
            "backoff_base": 1.0,
            "backoff_max": 60.0,
            "jitter": 0.25,
        },
        "api.coingecko.com": {
            # CoinGecko free tier caps at ~20 req/min.  If a penalty-
            # box kicks in we cool the host down for 5 min so the
            # oracle poll loop stays responsive.
            "min_interval_secs": 3.5,
            "max_retries": 3,
            "backoff_base": 2.0,
            "backoff_max": 60.0,
            "cooldown_on_final_429_secs": 300.0,
        },
        "gamma-api.polymarket.com": {
            "min_interval_secs": 1.0,
            "max_retries": 4,
        },
        "data-api.polymarket.com": {
            "min_interval_secs": 1.0,
            "max_retries": 4,
        },
    })

    # ── factory helpers ─────────────────────────────────────────────
    @classmethod
    def from_json(cls, path: str) -> "GridConfig":
        with open(path) as f:
            raw = json.load(f)
        return cls(**{k: v for k, v in raw.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "GridConfig":
        """Build config from GRID_* environment variables."""
        overrides: dict = {}
        prefix = "GRID_"
        for key, fld in cls.__dataclass_fields__.items():
            env_key = prefix + key.upper()
            val = os.environ.get(env_key)
            if val is None:
                continue
            if fld.type in ("float", float):
                overrides[key] = float(val)
            elif fld.type in ("int", int):
                overrides[key] = int(val)
            elif fld.type in ("bool", bool):
                overrides[key] = val.lower() in ("1", "true", "yes")
            elif fld.type == "str":
                overrides[key] = val
            # Lists and dicts are only loaded from JSON, not env vars.
        return cls(**overrides)

"""
Grid configuration.

Loadable from environment variables or a JSON file.  Every field has
a safe default so the grid can start in paper / read-only mode
without any external config.

P2 split
--------
The fields on ``GridConfig`` come in two flavours:

* **Deployment** fields (``DEPLOYMENT_FIELDS``) — mode, directories,
  HTTP throttling, oracle / whale / cross-market feed setup, poll
  intervals.  Changing any of these requires a process restart.
  ``validate_for_deployment()`` asserts that the deployment fields
  are coherent (e.g. live mode has a ``PK`` set).

* **Runtime** fields (``RUNTIME_FIELDS``) — detector thresholds, risk
  caps, book drift guard.  These can be reloaded on ``SIGHUP`` via
  ``reload_runtime_from(path)`` without dropping open positions or
  resetting daily counters.

``GridConfig`` stays flat so all existing call sites keep working;
``DeploymentConfig`` and ``RuntimeConfig`` are lightweight views that
expose only the relevant subset of fields for inspection or audit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, FrozenSet, List, Set


@dataclass(frozen=True)
class DeploymentConfig:
    """Immutable view of the deployment-level config slice."""
    mode: str
    kill_switch: bool
    live_armed: bool
    state_dir: str
    ledger_dir: str
    whale_wallets: List[str]
    cross_market_refs: Dict[str, dict]
    oracle_mappings: Dict[str, dict]
    gamma_poll_interval: float
    oracle_poll_interval: float
    whale_poll_interval: float
    http_throttle: Dict[str, dict]


@dataclass
class RuntimeConfig:
    """Reloadable view of the runtime-level config slice."""
    min_signals: int
    signal_staleness_secs: float
    direction_threshold: float
    max_entry_usdc: float
    max_open_positions: int
    daily_loss_cap_usdc: float
    consecutive_loss_cap: int
    max_open_per_category: int
    volume_spike_multiplier: float
    velocity_threshold: float
    disposition_threshold: float
    cross_market_delta_cents: float
    news_delta_cents: float
    theta_hours: float
    book_drift_bps: float
    max_slippage_bps: float
    exit_tp_cents: float
    exit_sl_cents: float
    exit_tighten_hours: float
    exit_tighten_factor: float


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

    # Max open positions in a single category (0 = disabled).
    max_open_per_category: int = 0

    # P1 hard second-lock for live mode: both mode="live" AND
    # live_armed=True must hold for orders to actually be placed.
    live_armed: bool = False

    # Snapshot / state directory for graceful restart.
    state_dir: str = "state"

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

    # Live pre-trade book-drift guard.  Refuse the order when the
    # book has moved more than this many basis points (1 bp = 0.01%)
    # since the fire moment.  Default 100 bps = 1%.
    book_drift_bps: float = 100.0

    # P3: maximum tolerable slippage when walking the book on a
    # paper-mode entry.  If the VWAP of the simulated fill sits
    # further than this many bps from the best price the paper
    # executor refuses the entry with reason slippage_exceeded.
    max_slippage_bps: float = 50.0

    # P3: cent-based take-profit / stop-loss thresholds, used by the
    # default CentThresholdStrategy.  These are absolute cent moves
    # from the entry price, so they behave consistently across a
    # mid=0.05 market and a mid=0.50 market.
    exit_tp_cents: float = 3.0
    exit_sl_cents: float = 2.0

    # P3: when the market is within this many hours of resolution,
    # tighten the TP/SL thresholds by exit_tighten_factor so the
    # strategy banks PnL faster near resolution.
    exit_tighten_hours: float = 1.0
    exit_tighten_factor: float = 0.5

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

    # ── deployment / runtime split (P2) ─────────────────────────────
    # Fields that describe *where* and *how* the process is running.
    # Changing any of these requires a full restart.
    DEPLOYMENT_FIELDS: ClassVar[FrozenSet[str]] = frozenset({
        "mode", "kill_switch", "live_armed", "state_dir", "ledger_dir",
        "whale_wallets", "cross_market_refs", "oracle_mappings",
        "gamma_poll_interval", "oracle_poll_interval",
        "whale_poll_interval", "http_throttle",
    })

    # Fields that can be live-reloaded via SIGHUP.  Every non-
    # deployment field belongs here by construction (see the self-
    # check in ``__post_init__``).
    RUNTIME_FIELDS: ClassVar[FrozenSet[str]] = frozenset({
        "min_signals", "signal_staleness_secs", "direction_threshold",
        "max_entry_usdc", "max_open_positions", "daily_loss_cap_usdc",
        "consecutive_loss_cap", "max_open_per_category",
        "volume_spike_multiplier", "velocity_threshold",
        "disposition_threshold", "cross_market_delta_cents",
        "news_delta_cents", "theta_hours", "book_drift_bps",
        "max_slippage_bps", "exit_tp_cents", "exit_sl_cents",
        "exit_tighten_hours", "exit_tighten_factor",
    })

    def __post_init__(self) -> None:
        all_declared = {f.name for f in fields(self)}
        covered = self.DEPLOYMENT_FIELDS | self.RUNTIME_FIELDS
        missing = all_declared - covered
        if missing:
            raise RuntimeError(
                f"GridConfig fields missing from deployment/runtime "
                f"split: {sorted(missing)}"
            )

    # ── typed views ─────────────────────────────────────────────────
    def deployment(self) -> DeploymentConfig:
        return DeploymentConfig(**{
            k: getattr(self, k) for k in self.DEPLOYMENT_FIELDS
        })

    def runtime(self) -> RuntimeConfig:
        return RuntimeConfig(**{
            k: getattr(self, k) for k in self.RUNTIME_FIELDS
        })

    # ── validation ──────────────────────────────────────────────────
    def validate_for_deployment(self) -> None:
        """Fail fast if the deployment config is inconsistent.

        Paper mode is always accepted.  Live mode requires both the
        ``live_armed`` flag and a ``PK`` environment variable.  This
        is the startup-time guard; ``LiveExecutor.enter`` re-checks at
        order-placement time too.
        """
        if self.mode not in ("paper", "live"):
            raise ValueError(
                f"mode must be 'paper' or 'live', got {self.mode!r}"
            )
        if self.mode == "live":
            if not self.live_armed:
                raise ValueError(
                    "live mode requires live_armed=true in the config "
                    "(second lock)"
                )
            if not os.environ.get("PK"):
                raise ValueError(
                    "live mode requires the PK environment variable "
                    "to be set"
                )

    # ── runtime reload (SIGHUP) ─────────────────────────────────────
    def reload_runtime_from(self, path: str) -> Set[str]:
        """Reload runtime-only fields from ``path``.

        Deployment fields in the file are ignored so a SIGHUP can
        never silently change mode / credentials / state dirs while
        the process is live.  Returns the set of field names that
        actually changed.
        """
        with open(path) as f:
            raw = json.load(f)
        changed: Set[str] = set()
        for key, value in raw.items():
            if key not in self.RUNTIME_FIELDS:
                continue
            if getattr(self, key) != value:
                setattr(self, key, value)
                changed.add(key)
        return changed

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

"""
Thread-safe rate-limited HTTP client.

Wraps ``requests.Session`` with:

*  per-host minimum spacing between requests (serialised via a per-host
   lock so multiple threads never hammer the same upstream)
*  transparent retry on ``429`` and 5xx responses
*  honouring of the ``Retry-After`` response header (integer seconds)
*  exponential backoff with jitter when no ``Retry-After`` is supplied
*  retry on transient network errors

Used by every ``feeds/*`` module so poll cadence and upstream limits
are decoupled: a feed can call ``poll()`` as often as it likes and the
HTTP client will space requests to each host correctly.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlparse

import requests


# ── configuration ───────────────────────────────────────────────────

@dataclass
class ThrottleConfig:
    """Per-host rate-limiting configuration."""

    # Minimum seconds between two successive requests to the same host.
    min_interval_secs: float = 1.0

    # Number of retries on 429 / 5xx / transient network errors.
    # Total attempts = max_retries + 1.
    max_retries: int = 4

    # Exponential backoff base (seconds).
    backoff_base: float = 1.0

    # Cap on a single backoff delay.
    backoff_max: float = 60.0

    # Fractional jitter applied to backoff delays (0.25 = +/-25%).
    jitter: float = 0.25

    # Circuit breaker: when > 0 and all retries to this host result
    # in a final 429, suppress further requests to the host for this
    # many seconds.  During the cooldown the session returns a
    # synthetic 429 response without touching the network.
    cooldown_on_final_429_secs: float = 0.0


# ── core session ────────────────────────────────────────────────────

class RateLimitedSession:
    """Thread-safe HTTP session with per-host throttling and retries."""

    def __init__(
        self,
        default: Optional[ThrottleConfig] = None,
        overrides: Optional[Dict[str, ThrottleConfig]] = None,
        session: Optional[requests.Session] = None,
        sleep_fn=time.sleep,
        monotonic_fn=time.monotonic,
    ):
        self._session = session or requests.Session()
        self._default = default or ThrottleConfig()
        self._overrides: Dict[str, ThrottleConfig] = dict(overrides or {})

        self._last_ts: Dict[str, float] = {}
        self._host_locks: Dict[str, threading.Lock] = {}
        self._locks_mutex = threading.Lock()

        # Circuit-breaker state: host -> epoch seconds until which
        # requests should short-circuit with a synthetic 429.
        self._cooldown_until: Dict[str, float] = {}

        # Injected for testability.
        self._sleep = sleep_fn
        self._monotonic = monotonic_fn

        # Counters (handy for diagnostics / tests).
        self.retries: int = 0
        self.throttle_waits: int = 0
        self.short_circuits: int = 0

    # ── public API ──────────────────────────────────────────────────

    def config_for(self, host: str) -> ThrottleConfig:
        return self._overrides.get(host, self._default)

    def get(self, url: str, **kwargs) -> requests.Response:
        return self._request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> requests.Response:
        return self._request("POST", url, **kwargs)

    # ── core request loop ───────────────────────────────────────────

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        host = urlparse(url).netloc
        cfg = self.config_for(host)

        # Short-circuit if this host is currently cooling down.
        cooldown_end = self._cooldown_until.get(host, 0.0)
        if cooldown_end and self._monotonic() < cooldown_end:
            self.short_circuits += 1
            return self._synthetic_429(cooldown_end - self._monotonic())

        last_response: Optional[requests.Response] = None
        last_exc: Optional[Exception] = None

        for attempt in range(cfg.max_retries + 1):
            self._throttle(host, cfg.min_interval_secs)

            try:
                resp = self._session.request(method, url, **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= cfg.max_retries:
                    raise
                self.retries += 1
                self._sleep(self._backoff_delay(cfg, attempt))
                continue

            if resp.status_code == 429 or 500 <= resp.status_code < 600:
                last_response = resp
                if attempt >= cfg.max_retries:
                    # Final 429: optionally trip the circuit breaker so
                    # subsequent requests during the cooldown short-
                    # circuit without hitting the network.
                    if resp.status_code == 429 and cfg.cooldown_on_final_429_secs > 0:
                        self._cooldown_until[host] = (
                            self._monotonic() + cfg.cooldown_on_final_429_secs
                        )
                    return resp
                self.retries += 1
                delay = self._retry_after(resp)
                if delay is None:
                    delay = self._backoff_delay(cfg, attempt)
                self._sleep(delay)
                continue

            return resp

        if last_response is not None:
            return last_response
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unreachable")  # pragma: no cover

    # ── throttling helpers ──────────────────────────────────────────

    def _throttle(self, host: str, min_interval: float) -> None:
        if min_interval <= 0:
            return
        lock = self._get_host_lock(host)
        with lock:
            now = self._monotonic()
            prev = self._last_ts.get(host)
            if prev is not None:
                wait = min_interval - (now - prev)
                if wait > 0:
                    self.throttle_waits += 1
                    self._sleep(wait)
                    now = self._monotonic()
            self._last_ts[host] = now

    def _get_host_lock(self, host: str) -> threading.Lock:
        with self._locks_mutex:
            if host not in self._host_locks:
                self._host_locks[host] = threading.Lock()
            return self._host_locks[host]

    def _backoff_delay(self, cfg: ThrottleConfig, attempt: int) -> float:
        delay = min(cfg.backoff_base * (2 ** attempt), cfg.backoff_max)
        if cfg.jitter > 0:
            delay *= 1.0 + random.uniform(-cfg.jitter, cfg.jitter)
        return max(delay, 0.0)

    @staticmethod
    def _synthetic_429(seconds_remaining: float) -> requests.Response:
        """Build a fake 429 response used while a host is cooling down."""
        r = requests.Response()
        r.status_code = 429
        r.headers["Retry-After"] = str(int(max(seconds_remaining, 1)))
        r._content = b'{"error":"circuit-breaker cooldown"}'
        r.url = "circuit-breaker://cooldown"
        return r

    @staticmethod
    def _retry_after(resp: requests.Response) -> Optional[float]:
        val = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
        if not val:
            return None
        try:
            return max(float(val), 0.0)
        except (TypeError, ValueError):
            return None


# ── module-level default session + configure hook ──────────────────

_default_session: Optional[RateLimitedSession] = None
_configure_lock = threading.Lock()


def default() -> RateLimitedSession:
    """Return the process-wide default session, creating a permissive
    one on first use if the main loop hasn't configured it yet."""
    global _default_session
    if _default_session is None:
        with _configure_lock:
            if _default_session is None:
                _default_session = RateLimitedSession()
    return _default_session


def configure(throttle_config: Dict[str, dict]) -> RateLimitedSession:
    """(Re)build the default session from a ``{host -> config-dict}``
    mapping as shipped in ``grid_config.json``.

    The ``default`` key (if present) becomes the fallback configuration
    for any host without an explicit override.
    """
    global _default_session
    raw_default = dict(throttle_config.get("default", {}))
    default_cfg = ThrottleConfig(**raw_default)

    overrides: Dict[str, ThrottleConfig] = {}
    for host, host_cfg in throttle_config.items():
        if host == "default":
            continue
        merged = {**raw_default, **host_cfg}
        overrides[host] = ThrottleConfig(**merged)

    with _configure_lock:
        _default_session = RateLimitedSession(default_cfg, overrides)
    return _default_session


# Convenience wrappers so callers can write ``http_client.get(...)``.

def get(url: str, **kwargs) -> requests.Response:
    return default().get(url, **kwargs)


def post(url: str, **kwargs) -> requests.Response:
    return default().post(url, **kwargs)

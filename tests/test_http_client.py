"""
Tests for the rate-limited HTTP client.

All tests use a fake session that records calls, returns scripted
responses, and a fake clock so the real ``time.sleep`` is never
invoked.  This keeps the suite deterministic and fast.
"""

from __future__ import annotations

import threading
from typing import List, Optional
from unittest.mock import MagicMock

import pytest
import requests

from grid.http_client import (
    RateLimitedSession,
    ThrottleConfig,
    configure,
    default,
)


# ── helpers ─────────────────────────────────────────────────────────

class FakeClock:
    """Deterministic replacement for time.monotonic + time.sleep."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_log: List[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, secs: float) -> None:
        self.sleep_log.append(secs)
        if secs > 0:
            self.now += secs


def make_response(status: int, headers: Optional[dict] = None) -> MagicMock:
    r = MagicMock(spec=requests.Response)
    r.status_code = status
    r.headers = headers or {}
    return r


class FakeSession:
    """Scripted session.  Pops from ``responses`` on each call."""

    def __init__(self, responses: List):
        self.responses = list(responses)
        self.calls: List[tuple] = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_session(responses, *, overrides=None, default_cfg=None, clock=None):
    clock = clock or FakeClock()
    sess = RateLimitedSession(
        default=default_cfg or ThrottleConfig(
            min_interval_secs=1.0, max_retries=3,
            backoff_base=1.0, backoff_max=10.0, jitter=0.0,
        ),
        overrides=overrides or {},
        session=FakeSession(responses),
        sleep_fn=clock.sleep,
        monotonic_fn=clock.monotonic,
    )
    return sess, clock


# ── throttling ──────────────────────────────────────────────────────

class TestThrottling:
    def test_first_request_does_not_sleep(self):
        sess, clock = make_session([make_response(200)])
        sess.get("https://example.com/a")
        assert sess.throttle_waits == 0

    def test_second_request_waits_min_interval(self):
        sess, clock = make_session([make_response(200), make_response(200)])
        sess.get("https://example.com/a")
        sess.get("https://example.com/b")
        # One throttle wait of ~1.0s
        assert sess.throttle_waits == 1
        assert any(0.9 <= s <= 1.1 for s in clock.sleep_log)

    def test_different_hosts_do_not_block_each_other(self):
        sess, clock = make_session(
            [make_response(200), make_response(200)],
        )
        sess.get("https://example.com/a")
        sess.get("https://other.com/b")
        assert sess.throttle_waits == 0


# ── 429 handling ────────────────────────────────────────────────────

class TestRateLimit429:
    def test_retries_and_succeeds(self):
        sess, _ = make_session([
            make_response(429),
            make_response(429),
            make_response(200),
        ])
        resp = sess.get("https://example.com/a")
        assert resp.status_code == 200
        assert sess.retries == 2

    def test_surfaces_final_429_after_max_retries(self):
        # Total attempts = max_retries + 1 = 4 with default, so script 4 429s.
        sess, _ = make_session(
            [make_response(429)] * 4,
            default_cfg=ThrottleConfig(
                min_interval_secs=0.0, max_retries=3,
                backoff_base=0.1, backoff_max=1.0, jitter=0.0,
            ),
        )
        resp = sess.get("https://example.com/a")
        assert resp.status_code == 429
        assert sess.retries == 3

    def test_honors_retry_after_integer(self):
        sess, clock = make_session([
            make_response(429, {"Retry-After": "7"}),
            make_response(200),
        ])
        sess.get("https://example.com/a")
        # 7s from Retry-After should appear in the sleep log.
        assert 7.0 in clock.sleep_log

    def test_exponential_backoff_without_retry_after(self):
        sess, clock = make_session(
            [make_response(429), make_response(429), make_response(200)],
            default_cfg=ThrottleConfig(
                min_interval_secs=0.0, max_retries=3,
                backoff_base=1.0, backoff_max=10.0, jitter=0.0,
            ),
        )
        sess.get("https://example.com/a")
        # Expect backoffs of 1.0 (attempt=0) and 2.0 (attempt=1).
        assert 1.0 in clock.sleep_log
        assert 2.0 in clock.sleep_log


# ── 5xx handling ────────────────────────────────────────────────────

class Test5xx:
    def test_retries_on_503(self):
        sess, _ = make_session(
            [make_response(503), make_response(200)],
            default_cfg=ThrottleConfig(
                min_interval_secs=0.0, max_retries=3,
                backoff_base=0.1, backoff_max=1.0, jitter=0.0,
            ),
        )
        resp = sess.get("https://example.com/a")
        assert resp.status_code == 200
        assert sess.retries == 1

    def test_does_not_retry_on_4xx_other_than_429(self):
        sess, _ = make_session([make_response(404)])
        resp = sess.get("https://example.com/a")
        assert resp.status_code == 404
        assert sess.retries == 0


# ── transient network errors ────────────────────────────────────────

class TestNetworkErrors:
    def test_retries_on_connection_error(self):
        sess, _ = make_session(
            [
                requests.ConnectionError("nope"),
                make_response(200),
            ],
            default_cfg=ThrottleConfig(
                min_interval_secs=0.0, max_retries=3,
                backoff_base=0.1, backoff_max=1.0, jitter=0.0,
            ),
        )
        resp = sess.get("https://example.com/a")
        assert resp.status_code == 200
        assert sess.retries == 1

    def test_raises_after_all_attempts_fail(self):
        sess, _ = make_session(
            [requests.ConnectionError("boom")] * 4,
            default_cfg=ThrottleConfig(
                min_interval_secs=0.0, max_retries=3,
                backoff_base=0.1, backoff_max=1.0, jitter=0.0,
            ),
        )
        with pytest.raises(requests.ConnectionError):
            sess.get("https://example.com/a")


# ── per-host overrides ──────────────────────────────────────────────

class TestPerHostOverrides:
    def test_override_picked_up_by_host(self):
        overrides = {
            "api.coingecko.com": ThrottleConfig(
                min_interval_secs=5.0, max_retries=7,
                backoff_base=2.0, backoff_max=30.0, jitter=0.0,
            ),
        }
        sess, _ = make_session(
            [make_response(200)], overrides=overrides,
        )
        cfg = sess.config_for("api.coingecko.com")
        assert cfg.min_interval_secs == 5.0
        assert cfg.max_retries == 7

    def test_default_used_for_unknown_host(self):
        overrides = {
            "api.coingecko.com": ThrottleConfig(min_interval_secs=5.0),
        }
        sess, _ = make_session(
            [make_response(200)], overrides=overrides,
        )
        cfg = sess.config_for("example.com")
        # Default min_interval_secs from make_session fixture is 1.0.
        assert cfg.min_interval_secs == 1.0


# ── configure() module-level helper ────────────────────────────────

class TestConfigure:
    def test_configure_merges_default_into_overrides(self):
        sess = configure({
            "default": {"min_interval_secs": 2.0, "max_retries": 3},
            "api.example.com": {"min_interval_secs": 5.0},
        })
        cfg = sess.config_for("api.example.com")
        assert cfg.min_interval_secs == 5.0
        # max_retries should inherit from default.
        assert cfg.max_retries == 3

    def test_configure_replaces_default_session(self):
        sess1 = configure({"default": {"min_interval_secs": 1.0}})
        sess2 = configure({"default": {"min_interval_secs": 2.0}})
        assert default() is sess2
        assert sess1 is not sess2


# ── circuit breaker ──────────────────────────────────────────────

class TestCircuitBreaker:
    def test_cooldown_trips_after_final_429(self):
        cfg = ThrottleConfig(
            min_interval_secs=0.0, max_retries=1,
            backoff_base=0.1, backoff_max=1.0, jitter=0.0,
            cooldown_on_final_429_secs=60.0,
        )
        # Exhaust retries with persistent 429s, then a would-be 200.
        sess, clock = make_session(
            [make_response(429), make_response(429), make_response(200)],
            default_cfg=cfg,
        )
        resp1 = sess.get("https://rate.example.com/")
        assert resp1.status_code == 429

        # Next call must short-circuit to a synthetic 429 without
        # touching the underlying session (the 200 is never reached).
        resp2 = sess.get("https://rate.example.com/")
        assert resp2.status_code == 429
        assert resp2.headers.get("Retry-After") is not None
        assert sess.short_circuits == 1

    def test_cooldown_expires(self):
        cfg = ThrottleConfig(
            min_interval_secs=0.0, max_retries=1,
            backoff_base=0.1, backoff_max=1.0, jitter=0.0,
            cooldown_on_final_429_secs=10.0,
        )
        sess, clock = make_session(
            [make_response(429), make_response(429), make_response(200)],
            default_cfg=cfg,
        )
        sess.get("https://rate.example.com/")       # trips breaker
        clock.now += 20                              # advance past cooldown
        resp = sess.get("https://rate.example.com/") # should reach the 200
        assert resp.status_code == 200

    def test_cooldown_is_per_host(self):
        cfg = ThrottleConfig(
            min_interval_secs=0.0, max_retries=1,
            backoff_base=0.1, backoff_max=1.0, jitter=0.0,
            cooldown_on_final_429_secs=60.0,
        )
        sess, _ = make_session(
            [make_response(429), make_response(429),   # host A
             make_response(200)],                       # host B
            default_cfg=cfg,
        )
        sess.get("https://a.example.com/")  # trips A
        resp = sess.get("https://b.example.com/")
        assert resp.status_code == 200  # B is unaffected


# ── thread safety ──────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_same_host_calls_are_serialised(self):
        """Two concurrent threads to the same host must not race the
        min-interval check."""
        # 10 scripted 200s, one per call.
        sess, clock = make_session(
            [make_response(200) for _ in range(10)],
            default_cfg=ThrottleConfig(
                min_interval_secs=0.05, max_retries=0,
                backoff_base=0.0, backoff_max=0.0, jitter=0.0,
            ),
        )
        # Make the FakeClock thread-safe for the duration of this test.
        orig_sleep = clock.sleep
        clock_lock = threading.Lock()
        def safe_sleep(s):
            with clock_lock:
                orig_sleep(s)
        sess._sleep = safe_sleep

        def run():
            for _ in range(5):
                sess.get("https://same-host.com/x")

        t1 = threading.Thread(target=run)
        t2 = threading.Thread(target=run)
        t1.start(); t2.start()
        t1.join(); t2.join()

        # 10 total calls, 9 should have triggered a throttle wait.
        assert sess.throttle_waits == 9

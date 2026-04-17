"""
Happy-path tests for the GammaDiscoveryPoller.

These tests exercise the parsing, registration, and eviction logic in
isolation — no network calls, no real Gamma traffic.  Discovery itself
(``_discover``) is bypassed by calling ``_register`` and
``_evict_expired`` directly with synthetic entries.
"""

from __future__ import annotations

import time

import pytest

from detectors.news import NewsDetector
from feeds.gamma_discovery import GammaDiscoveryPoller
from feeds.oracle import OraclePoller
from grid.config import GridConfig

import poly_data.global_state as global_state


@pytest.fixture
def cfg():
    return GridConfig()


@pytest.fixture
def wired(cfg):
    news_det = NewsDetector(cfg)
    oracle = OraclePoller(cfg, news_det)
    disc = GammaDiscoveryPoller(oracle, news_det)
    # Reset shared global state between tests.
    global_state.all_tokens = []
    return disc, oracle, news_det


# ── parser ──────────────────────────────────────────────────────────

class TestParser:
    def test_parses_btc_threshold_market(self, wired):
        disc, _, _ = wired
        m = {
            "conditionId": "0xabc",
            "question": "Will the price of Bitcoin be above $62,000 on April 17?",
            "endDate": "2099-01-01T00:00:00Z",
            "clobTokenIds": '["1111","2222"]',
            "volumeNum": 100_000,
        }
        parsed = disc._parse_market(m)
        assert parsed is not None
        assert parsed["coin_id"] == "bitcoin"
        assert parsed["threshold"] == 62000
        assert parsed["token_ids"] == ["1111", "2222"]

    def test_parses_eth_with_k_suffix(self, wired):
        disc, _, _ = wired
        m = {
            "conditionId": "0xeth",
            "question": "Will the price of Ethereum be above $2k on May 1?",
            "endDate": "2099-01-01T00:00:00Z",
            "clobTokenIds": "[]",
            "volumeNum": 50_000,
        }
        parsed = disc._parse_market(m)
        assert parsed is not None
        assert parsed["coin_id"] == "ethereum"
        assert parsed["threshold"] == 2_000

    def test_rejects_non_threshold_question(self, wired):
        disc, _, _ = wired
        m = {
            "conditionId": "0xelection",
            "question": "Will the Democrats win the 2028 election?",
            "endDate": "2099-01-01T00:00:00Z",
            "clobTokenIds": "[]",
        }
        assert disc._parse_market(m) is None

    def test_rejects_unknown_asset(self, wired):
        disc, _, _ = wired
        m = {
            "conditionId": "0xasdf",
            "question": "Will the price of Potato be above $10 on Friday?",
            "endDate": "2099-01-01T00:00:00Z",
            "clobTokenIds": "[]",
        }
        assert disc._parse_market(m) is None


# ── token extraction ────────────────────────────────────────────────

class TestTokenExtraction:
    def test_stringified_json_array(self, wired):
        disc, _, _ = wired
        assert disc._extract_token_ids({"clobTokenIds": '["1","2"]'}) == ["1", "2"]

    def test_real_list(self, wired):
        disc, _, _ = wired
        assert disc._extract_token_ids({"clobTokenIds": ["a", "b"]}) == ["a", "b"]

    def test_missing_returns_empty(self, wired):
        disc, _, _ = wired
        assert disc._extract_token_ids({}) == []

    def test_tokens_fallback(self, wired):
        disc, _, _ = wired
        m = {"tokens": [{"token_id": "x"}, {"token_id": "y"}]}
        assert disc._extract_token_ids(m) == ["x", "y"]


# ── registration and eviction ───────────────────────────────────────

class TestRegister:
    def test_registers_oracle_mapping_and_subscriptions(self, wired):
        disc, oracle, _ = wired
        entry = {
            "condition_id": "0xfresh",
            "coin_id": "bitcoin",
            "threshold": 62000,
            "end_epoch": time.time() + 3600,
            "token_ids": ["111", "222"],
            "question": "Will BTC be above $62,000 today?",
            "volume": 100_000,
        }
        added = disc._register(entry)
        assert added is True
        assert "0xfresh" in oracle._mappings
        assert oracle._mappings["0xfresh"]["id"] == "bitcoin"
        assert oracle._mappings["0xfresh"]["threshold"] == 62000
        assert "111" in global_state.all_tokens
        assert "222" in global_state.all_tokens

    def test_second_register_of_same_cid_is_noop(self, wired):
        disc, oracle, _ = wired
        entry = {
            "condition_id": "0xdup",
            "coin_id": "ethereum",
            "threshold": 1700,
            "end_epoch": time.time() + 3600,
            "token_ids": ["a", "b"],
            "question": "Eth",
            "volume": 10_000,
        }
        assert disc._register(entry) is True
        # Re-register — no new tokens appended.
        assert disc._register(entry) is False
        assert global_state.all_tokens.count("a") == 1

    def test_evicts_expired(self, wired):
        disc, oracle, news = wired
        entry = {
            "condition_id": "0xstale",
            "coin_id": "bitcoin",
            "threshold": 50000,
            "end_epoch": time.time() - 10,  # already past
            "token_ids": ["z"],
            "question": "BTC stale",
            "volume": 10_000,
        }
        # Force register despite being in the past (caller normally
        # filters this — we just want to exercise eviction).
        disc._registered["0xstale"] = {
            "end_epoch": entry["end_epoch"],
            "token_ids": entry["token_ids"],
            "question": entry["question"],
            "coin_id": entry["coin_id"],
            "threshold": entry["threshold"],
            "volume": entry["volume"],
        }
        oracle.set_mapping("0xstale", "coingecko", id="bitcoin", threshold=50000)
        news.set_feed_value("0xstale", 1.0, "coingecko:bitcoin")

        disc._evict_expired()
        assert "0xstale" not in disc._registered
        assert "0xstale" not in oracle._mappings
        assert "0xstale" not in news._feed_values

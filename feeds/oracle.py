"""
Oracle feed poller.

Fetches public external data (CoinGecko prices, etc.) and injects
values into the news detector so it can compare feed vs market price.

This is a stub — concrete feed adapters should be added per-source
as the operator configures market-to-feed mappings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import requests

if TYPE_CHECKING:
    from detectors.news import NewsDetector
    from grid.config import GridConfig


COINGECKO_SIMPLE = "https://api.coingecko.com/api/v3/simple/price"


class OraclePoller:
    def __init__(self, config: "GridConfig", news_detector: "NewsDetector"):
        self.config = config
        self._news_det = news_detector
        # Operator-supplied mapping: {market_condition_id: {"source": "coingecko", "id": "bitcoin", ...}}
        # Loaded from config or a separate JSON file.  Starts empty.
        self._mappings: Dict[str, dict] = {}

    def set_mapping(self, market: str, source: str, **kwargs) -> None:
        self._mappings[market] = {"source": source, **kwargs}

    def poll(self) -> None:
        # Group by source for batching
        coingecko_ids: Dict[str, str] = {}  # coin_id -> market
        for market, mapping in self._mappings.items():
            if mapping["source"] == "coingecko":
                coingecko_ids[mapping.get("id", "")] = market

        if coingecko_ids:
            self._poll_coingecko(coingecko_ids)

    def _poll_coingecko(self, ids_to_market: Dict[str, str]) -> None:
        ids_csv = ",".join(ids_to_market.keys())
        try:
            resp = requests.get(
                COINGECKO_SIMPLE,
                params={"ids": ids_csv, "vs_currencies": "usd"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[oracle] coingecko poll failed: {exc}")
            return

        for coin_id, market in ids_to_market.items():
            price = data.get(coin_id, {}).get("usd")
            if price is not None:
                # Normalise to 0–1 probability if the mapping provides a threshold.
                mapping = self._mappings[market]
                threshold = mapping.get("threshold")
                if threshold is not None:
                    # e.g. "BTC above 100k" → value = price / threshold, clamped 0–1
                    value = min(max(float(price) / float(threshold), 0.0), 1.0)
                else:
                    value = float(price)  # raw — caller must interpret
                self._news_det.set_feed_value(market, value, f"coingecko:{coin_id}")

"""
Cross-market feed poller.

Fetches reference prices from external prediction-market or financial
data sources and injects them into the cross_market detector.

Currently supports a simple HTTP JSON endpoint adapter.  Add venue-
specific adapters (Kalshi, Manifold, PredictIt) as needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import requests

if TYPE_CHECKING:
    from detectors.cross_market import CrossMarketDetector
    from grid.config import GridConfig


class CrossMarketPoller:
    def __init__(self, config: "GridConfig", detector: "CrossMarketDetector"):
        self.config = config
        self._det = detector

    def poll(self) -> None:
        for market, ref in self.config.cross_market_refs.items():
            source = ref.get("source", "")
            url = ref.get("url", "")
            json_path = ref.get("json_path", "")  # dot-separated path
            if not url:
                continue
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                value = self._extract(data, json_path)
                if value is not None:
                    self._det.set_reference(market, float(value), source)
            except Exception as exc:
                print(f"[cross_market] poll {source} failed: {exc}")

    @staticmethod
    def _extract(data: dict, path: str):
        """Walk a dot-separated JSON path to extract a value."""
        if not path:
            return data
        parts = path.split(".")
        obj = data
        for p in parts:
            if isinstance(obj, dict):
                obj = obj.get(p)
            elif isinstance(obj, list) and p.isdigit():
                obj = obj[int(p)]
            else:
                return None
        return obj

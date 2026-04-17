"""
Gamma API feed poller.

Fetches market metadata (categories, end dates, volume) from the
public Gamma REST API and injects it into the category and theta
detectors.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List

import requests

if TYPE_CHECKING:
    from detectors.category import CategoryDetector
    from detectors.theta import ThetaDetector


GAMMA_BASE = "https://gamma-api.polymarket.com"


class GammaPoller:
    def __init__(
        self,
        tracked_markets: List[str],
        category_detector: "CategoryDetector",
        theta_detector: "ThetaDetector",
    ):
        self._tracked = tracked_markets
        self._cat_det = category_detector
        self._theta_det = theta_detector
        # {category: [volume_samples]}  for baseline computation
        self._category_volumes: Dict[str, List[float]] = {}

    def poll(self) -> None:
        try:
            resp = requests.get(
                f"{GAMMA_BASE}/markets",
                params={"limit": 100, "active": "true"},
                timeout=15,
            )
            resp.raise_for_status()
            markets = resp.json()
        except Exception as exc:
            print(f"[gamma] poll failed: {exc}")
            return

        cat_volumes: Dict[str, float] = {}

        for m in markets:
            cid = m.get("conditionID", "")
            tag = m.get("category", "")
            end_iso = m.get("endDate", "")
            vol = float(m.get("volumeNum", 0) or 0)

            if cid in self._tracked:
                if tag:
                    self._cat_det.set_market_category(cid, tag)
                if end_iso:
                    try:
                        end_dt = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
                        self._theta_det.set_end_date(cid, end_dt.timestamp())
                    except ValueError:
                        pass

            if tag:
                cat_volumes[tag] = cat_volumes.get(tag, 0) + vol

        # Update category baselines (simple exponential moving average)
        for tag, vol in cat_volumes.items():
            history = self._category_volumes.setdefault(tag, [])
            history.append(vol)
            # Keep last 20 samples for baseline
            if len(history) > 20:
                history.pop(0)
            baseline = sum(history) / len(history)
            self._cat_det.set_category_stats(tag, vol, baseline)

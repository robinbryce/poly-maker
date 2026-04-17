"""
Gamma-based market auto-discovery — the "lights out" half of the
detector grid.

Periodically queries the Polymarket Gamma API for active threshold
markets on a configured set of assets and, for each new discovery:

    1. parses the asset and threshold out of the question text
    2. registers a binary-margin oracle mapping on the OraclePoller
       so the news detector can compare feed vs market midpoint
    3. adds the market's YES and NO clob token ids to
       ``global_state.all_tokens`` so the market websocket picks them
       up on next reconnect
    4. sets the ``subscription_dirty`` event so the main loop can
       force a websocket reconnect to pick up new subscriptions

When an existing mapping's resolution time passes, it is pruned so
stale markets don't linger.

No manual editing of ``grid_config.json`` or the Selected Markets
sheet is required for a discovered market to produce fires.
"""

from __future__ import annotations

import datetime
import json
import re
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional

from grid import http_client

import poly_data.global_state as global_state

if TYPE_CHECKING:
    from detectors.news import NewsDetector
    from feeds.oracle import OraclePoller


GAMMA_BASE = "https://gamma-api.polymarket.com"

# Common spoken / slug names → CoinGecko ids.  Extend as needed.
ASSET_TO_COINGECKO: Dict[str, str] = {
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "solana": "solana",
    "sol": "solana",
    "ripple": "ripple",
    "xrp": "ripple",
    "dogecoin": "dogecoin",
    "doge": "dogecoin",
    "cardano": "cardano",
    "ada": "cardano",
    "avalanche": "avalanche-2",
    "avax": "avalanche-2",
}

# Matches "... price of <asset> be above $<num>[k]? on ..."
_QUESTION_RE = re.compile(
    r"price of\s+(?P<asset>[A-Za-z]+)\s+be\s+above\s+"
    r"\$?(?P<num>[\d,\.]+)\s*(?P<suffix>k|K|m|M)?",
)


class GammaDiscoveryPoller:
    """Discovers active threshold markets and wires them into the grid.

    Designed to run on the shared background-polling interval.  All
    HTTP and parse failures are swallowed with a log message so that
    transient Gamma outages can't take the grid down.

    Two discovery paths:

    *  Threshold markets ("Will the price of X be above $Y on <date>?"):
       registered with the oracle poller so the news detector can
       compare feed vs market midpoint.
    *  Top-volume markets: registered only for websocket subscription
       so the volume / disposition / whale detectors have something
       with actual trade flow to observe.
    """

    def __init__(
        self,
        oracle_poller: "OraclePoller",
        news_detector: "NewsDetector",
        search_terms: Optional[List[str]] = None,
        lookahead_hours: float = 48.0,
        min_volume_usdc: float = 10_000.0,
        margin: float = 0.01,
        top_volume_count: int = 30,
        top_volume_min_usdc: float = 100_000.0,
        top_volume_lookahead_days: float = 14.0,
    ):
        self._oracle = oracle_poller
        self._news_det = news_detector
        self._search_terms = search_terms or [
            "bitcoin", "ethereum", "solana",
        ]
        self._lookahead_hours = lookahead_hours
        self._min_volume = min_volume_usdc
        self._margin = margin

        # Top-volume discovery knobs.
        self._top_volume_count = top_volume_count
        self._top_volume_min_usdc = top_volume_min_usdc
        self._top_volume_lookahead_days = top_volume_lookahead_days

        # condition_id -> {end_epoch, token_ids, question, coin_id, threshold}
        self._registered: Dict[str, dict] = {}
        # condition_id -> {end_epoch, token_ids, question, volume} for
        # markets tracked for subscription but without oracle mapping.
        self._tracked_only: Dict[str, dict] = {}

        # Set whenever we append new token ids to global_state.all_tokens.
        # The asyncio layer polls this and forces a websocket reconnect
        # so the new subscriptions take effect.
        self.subscription_dirty = threading.Event()

    # ── public API ──────────────────────────────────────────────────

    def poll(self) -> None:
        """One discovery cycle.  Intended to be called from the shared
        background polling loop."""
        added_any_tokens = False

        # 1. Threshold markets (register with oracle + subscribe).
        for entry in self._discover_threshold():
            if self._register(entry):
                added_any_tokens = True

        # 2. Top-volume markets (subscribe only).
        for entry in self._discover_top_volume():
            if self._track_only(entry):
                added_any_tokens = True

        if added_any_tokens:
            self.subscription_dirty.set()

        self._evict_expired()

    def all_tracked_cids(self) -> List[str]:
        """Every condition_id this poller currently considers tracked
        (threshold + top-volume).  Fed into the main gamma_poller so
        the category and theta detectors receive metadata for them."""
        return list(self._registered.keys()) + list(self._tracked_only.keys())

    def snapshot(self) -> List[dict]:
        """Return a copy of the current registrations for logging."""
        return [
            {"condition_id": cid, **meta}
            for cid, meta in self._registered.items()
        ]

    # ── discovery ────────────────────────────────────────────────────────

    def _discover_threshold(self) -> List[dict]:
        out: List[dict] = []
        now = time.time()
        horizon = now + self._lookahead_hours * 3600

        for term in self._search_terms:
            try:
                resp = http_client.get(
                    f"{GAMMA_BASE}/public-search",
                    params={"q": f"{term} above", "limit": 40},
                    timeout=15,
                )
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:
                print(f"[gamma_discovery] search '{term}' failed: {exc}")
                continue

            events = payload.get("events", []) if isinstance(payload, dict) else []
            for ev in events:
                for m in (ev.get("markets") or []):
                    parsed = self._parse_market(m)
                    if parsed is None:
                        continue
                    if parsed["end_epoch"] <= now:
                        continue
                    if parsed["end_epoch"] > horizon:
                        continue
                    if parsed["volume"] < self._min_volume:
                        continue
                    out.append(parsed)
        return out

    def _discover_top_volume(self) -> List[dict]:
        """Fetch the highest-volume active markets from Gamma.  These
        are registered for subscription only — no oracle mapping.
        """
        try:
            resp = http_client.get(
                f"{GAMMA_BASE}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "order": "volumeNum",
                    "ascending": "false",
                    "limit": self._top_volume_count,
                },
                timeout=15,
            )
            resp.raise_for_status()
            markets = resp.json()
        except Exception as exc:
            print(f"[gamma_discovery] top-volume poll failed: {exc}")
            return []

        now = time.time()
        horizon = now + self._top_volume_lookahead_days * 86400

        out: List[dict] = []
        for m in markets or []:
            cid = m.get("conditionId") or ""
            if not cid:
                continue
            # Skip if already covered by threshold discovery.
            if cid in self._registered:
                continue

            end_iso = m.get("endDate") or ""
            try:
                end_dt = datetime.datetime.fromisoformat(
                    end_iso.replace("Z", "+00:00"))
                end_epoch = end_dt.timestamp()
            except Exception:
                continue
            if end_epoch <= now or end_epoch > horizon:
                continue

            volume = 0.0
            for key in ("volumeNum", "volume"):
                v = m.get(key)
                if v is not None:
                    try:
                        volume = float(v)
                        break
                    except (TypeError, ValueError):
                        continue
            if volume < self._top_volume_min_usdc:
                continue

            token_ids = self._extract_token_ids(m)
            if not token_ids:
                continue

            out.append({
                "condition_id": cid,
                "end_epoch":    end_epoch,
                "token_ids":    token_ids,
                "question":     m.get("question") or "",
                "volume":       volume,
            })
        return out

    def _parse_market(self, m: dict) -> Optional[dict]:
        cid = m.get("conditionId") or ""
        q = m.get("question") or ""
        if not cid or not q:
            return None

        match = _QUESTION_RE.search(q)
        if match is None:
            return None

        asset = match.group("asset").lower()
        coin_id = ASSET_TO_COINGECKO.get(asset)
        if coin_id is None:
            return None

        num_str = match.group("num").replace(",", "")
        try:
            num = float(num_str)
        except ValueError:
            return None
        suffix = (match.group("suffix") or "").lower()
        if suffix == "k":
            num *= 1_000
        elif suffix == "m":
            num *= 1_000_000
        if num <= 0:
            return None

        end_iso = m.get("endDate") or ""
        try:
            end_dt = datetime.datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
            end_epoch = end_dt.timestamp()
        except Exception:
            return None

        token_ids = self._extract_token_ids(m)

        volume = 0.0
        for key in ("volumeNum", "volume"):
            v = m.get(key)
            if v is not None:
                try:
                    volume = float(v)
                    break
                except (TypeError, ValueError):
                    continue

        return {
            "condition_id": cid,
            "coin_id": coin_id,
            "threshold": num,
            "end_epoch": end_epoch,
            "token_ids": token_ids,
            "question": q,
            "volume": volume,
        }

    @staticmethod
    def _extract_token_ids(m: dict) -> List[str]:
        """Gamma returns ``clobTokenIds`` as a stringified JSON array."""
        raw = m.get("clobTokenIds")
        if isinstance(raw, list):
            return [str(t) for t in raw if t]
        if isinstance(raw, str):
            try:
                arr = json.loads(raw)
                return [str(t) for t in arr if t]
            except (TypeError, ValueError):
                return []
        # Fall back to the less common ``tokens`` shape.
        tokens = m.get("tokens") or []
        if isinstance(tokens, list):
            return [str(t.get("token_id", "")) for t in tokens if t.get("token_id")]
        return []

    # ── registration / eviction ─────────────────────────────────────

    def _register(self, entry: dict) -> bool:
        """Register one discovered market.  Returns True if any new
        token ids were appended to the global subscription list."""
        cid = entry["condition_id"]
        if cid in self._registered:
            # Already known — refresh end_epoch in case it changed.
            self._registered[cid]["end_epoch"] = entry["end_epoch"]
            return False

        self._oracle.set_mapping(
            cid,
            source="coingecko",
            id=entry["coin_id"],
            threshold=entry["threshold"],
            margin=self._margin,
        )

        added = 0
        for tid in entry["token_ids"]:
            if tid and tid not in global_state.all_tokens:
                global_state.all_tokens.append(tid)
                added += 1

        self._registered[cid] = {
            "end_epoch": entry["end_epoch"],
            "token_ids": entry["token_ids"],
            "question": entry["question"],
            "coin_id": entry["coin_id"],
            "threshold": entry["threshold"],
            "volume": entry["volume"],
        }

        end_human = datetime.datetime.fromtimestamp(
            entry["end_epoch"], tz=datetime.timezone.utc
        ).isoformat()
        print(
            f"[gamma_discovery] +register {entry['coin_id']} > "
            f"${entry['threshold']:,.0f} ends {end_human}  "
            f"vol={entry['volume']:,.0f}  new_tokens={added}  "
            f"cid={cid[:12]}…"
        )
        return added > 0

    def _track_only(self, entry: dict) -> bool:
        """Register a market for subscription without an oracle
        mapping.  Returns True if any new token ids were added."""
        cid = entry["condition_id"]
        if cid in self._tracked_only or cid in self._registered:
            return False

        added = 0
        for tid in entry["token_ids"]:
            if tid and tid not in global_state.all_tokens:
                global_state.all_tokens.append(tid)
                added += 1

        self._tracked_only[cid] = {
            "end_epoch": entry["end_epoch"],
            "token_ids": entry["token_ids"],
            "question":  entry["question"],
            "volume":    entry["volume"],
        }
        print(
            f"[gamma_discovery] +track  vol={entry['volume']:>12,.0f}  "
            f"new_tokens={added}  cid={cid[:12]}…  q={entry['question'][:60]}"
        )
        return added > 0

    def _evict_expired(self) -> None:
        now = time.time()
        # Threshold / oracle-backed markets.
        expired = [
            cid for cid, meta in self._registered.items()
            if meta["end_epoch"] <= now
        ]
        for cid in expired:
            meta = self._registered.pop(cid)
            self._oracle._mappings.pop(cid, None)
            try:
                self._news_det._feed_values.pop(cid, None)
                self._news_det._midpoints.pop(cid, None)
            except AttributeError:
                pass
            print(
                f"[gamma_discovery] -expire {meta['coin_id']} > "
                f"${meta['threshold']:,.0f}  cid={cid[:12]}…"
            )

        # Top-volume markets (no oracle state to drop).
        expired = [
            cid for cid, meta in self._tracked_only.items()
            if meta["end_epoch"] <= now
        ]
        for cid in expired:
            meta = self._tracked_only.pop(cid)
            print(
                f"[gamma_discovery] -expire top-volume  "
                f"vol={meta['volume']:,.0f}  cid={cid[:12]}…"
            )

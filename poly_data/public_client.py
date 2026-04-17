"""
Public (credential-free) Polymarket client.

Used by the detector grid in paper mode.  Wraps the CLOB client at
authentication Level 0 so it can pull order books and market data
without requiring a private key, proxy wallet, or API keys.

Deliberately exposes only the read-only subset of methods the grid
actually needs.  Any attempt to create or post an order will raise.
"""

from __future__ import annotations

import pandas as pd
from py_clob_client.client import ClobClient
from py_clob_client.constants import POLYGON


class PublicPolymarketClient:
    """Read-only Polymarket client for paper / simulation runs."""

    def __init__(self, host: str = "https://clob.polymarket.com") -> None:
        # Level 0 — no key, no creds.  All read endpoints are public.
        self.client = ClobClient(host=host, chain_id=POLYGON)
        # Placeholder attrs so code paths that inspect these don't
        # explode when we're running without credentials.
        self.browser_wallet = ""
        self.creds = None

    # ── read-only market data ────────────────────────────────────────

    def get_order_book(self, market: str):
        book = self.client.get_order_book(market)
        return (
            pd.DataFrame(book.bids).astype(float),
            pd.DataFrame(book.asks).astype(float),
        )

    # ── write-path stubs (paper mode must never call these) ──────────

    def create_order(self, *args, **kwargs):  # noqa: D401
        raise RuntimeError(
            "PublicPolymarketClient cannot place orders — paper mode only."
        )

    def cancel_all_asset(self, *args, **kwargs):
        raise RuntimeError(
            "PublicPolymarketClient cannot cancel orders — paper mode only."
        )

    def cancel_all_market(self, *args, **kwargs):
        raise RuntimeError(
            "PublicPolymarketClient cannot cancel orders — paper mode only."
        )

    def merge_positions(self, *args, **kwargs):
        raise RuntimeError(
            "PublicPolymarketClient cannot merge positions — paper mode only."
        )

    # ── auth-required reads that paper mode doesn't need ─────────────

    def get_all_orders(self):
        # Return an empty frame so call-sites that inspect it don't fail.
        return pd.DataFrame()

    def get_all_positions(self):
        return pd.DataFrame()

    def get_usdc_balance(self):
        return 0.0

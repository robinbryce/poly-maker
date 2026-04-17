"""Persistent snapshot of grid runtime state.

Written atomically on graceful shutdown, read on startup so the
coordinator and paper executor can resume with correct invariants
(one-position-per-market, daily loss counters, open paper positions,
circuit-breaker cooldowns).
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, Optional


SNAPSHOT_FILE = "grid_state.json"


class SnapshotStore:
    def __init__(self, directory: str):
        self._dir = directory
        os.makedirs(self._dir, exist_ok=True)
        self._path = os.path.join(self._dir, SNAPSHOT_FILE)

    def load(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self._path):
            return None
        try:
            with open(self._path) as f:
                return json.load(f)
        except Exception:
            return None

    def save(self, data: Dict[str, Any]) -> None:
        """Atomic write: write to a tempfile, fsync, rename."""
        data = {**data, "snapshot_ts": time.time()}
        fd, tmp = tempfile.mkstemp(prefix=".snap_", dir=self._dir)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, default=str, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise

    def clear(self) -> None:
        try:
            os.unlink(self._path)
        except FileNotFoundError:
            pass

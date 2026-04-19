"""Thread-safe metric counters for the grid.

Two flavours of counter are supported:

* **Unlabelled** \u2014 ``counters.incr("ws.reconnects")``: a single
  integer per name.
* **Labelled** \u2014
  ``counters.labelled_incr("http.status", {"host": "api.coingecko.com", "code": "429"})``:
  a named counter keyed by a frozen set of label pairs, StatsD-style.

``snapshot()`` returns a plain dict you can pass to
``json.dumps(..., default=str)``.  ``write_snapshot(path)`` writes
the snapshot to a file atomically via a sibling ``.tmp`` + ``rename``.

A module-level singleton ``counters`` lets any module do
``from grid.metrics import counters`` without plumbing references
through the system.  The class itself is instantiable too, which is
useful in tests.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, Mapping, Optional, Tuple


_LabelKey = Tuple[Tuple[str, str], ...]


def _freeze_labels(labels: Mapping[str, str]) -> _LabelKey:
    return tuple(sorted((str(k), str(v)) for k, v in labels.items()))


class Counters:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._plain: Dict[str, int] = {}
        self._labelled: Dict[str, Dict[_LabelKey, int]] = {}

    # ── increments ─────────────────────────────────────────────────

    def incr(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._plain[name] = self._plain.get(name, 0) + int(n)

    def labelled_incr(
        self, name: str, labels: Mapping[str, str], n: int = 1,
    ) -> None:
        key = _freeze_labels(labels)
        with self._lock:
            bucket = self._labelled.setdefault(name, {})
            bucket[key] = bucket.get(key, 0) + int(n)

    # ── reads ──────────────────────────────────────────────────────

    def get(self, name: str) -> int:
        with self._lock:
            return int(self._plain.get(name, 0))

    def get_labelled(self, name: str, labels: Mapping[str, str]) -> int:
        key = _freeze_labels(labels)
        with self._lock:
            return int(self._labelled.get(name, {}).get(key, 0))

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        """Return a plain-dict view safe to JSON-serialise.

        Labelled counters are nested under ``"labelled"`` with labels
        re-expanded to a dict to keep the output readable.
        """
        with self._lock:
            plain = dict(self._plain)
            labelled = {
                name: [
                    {"labels": dict(key), "value": int(value)}
                    for key, value in buckets.items()
                ]
                for name, buckets in self._labelled.items()
            }
        return {"plain": plain, "labelled": labelled}

    def write_snapshot(self, path: str) -> None:
        """Atomically write the snapshot to ``path`` as JSON."""
        snap = self.snapshot()
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f, default=str, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def reset(self) -> None:
        with self._lock:
            self._plain.clear()
            self._labelled.clear()


# Module-level singleton so any module can import and use this without
# dependency plumbing.  Tests can instantiate a fresh Counters().
counters = Counters()

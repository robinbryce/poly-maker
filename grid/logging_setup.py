"""Structured logging for the grid.

Installs three handlers on the root logger:

* **stdout** (``StreamHandler``) \u2014 a concise human formatter, so the
  tmux window keeps its familiar at-a-glance output.
* **logs/grid.log** (``TimedRotatingFileHandler``) \u2014 same human
  formatter, rotated daily, last 7 files retained.
* **logs/grid.jsonl** (``TimedRotatingFileHandler``) \u2014 JSON-lines
  formatter with ``ts``, ``level``, ``logger``, ``message`` and any
  ``extra={}`` fields, rotated daily, last 7 files retained.

The stdout + JSON split means ``task tail:log`` stays readable while
ops tooling can consume the JSON stream directly.

Usage:

    from grid.logging_setup import configure_logging
    configure_logging(level="INFO", log_dir="logs")
    logger = logging.getLogger(__name__)
    logger.info("something happened", extra={"market": "0x..."})
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from typing import Any, Dict, Optional


_RESERVED_LOGRECORD_ATTRS = {
    # Standard LogRecord attributes we never want duplicated in the
    # JSON payload.
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "message",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Format a LogRecord as a single JSON line.

    Picks up any keyword arguments passed via ``logger.info(..., extra={})``
    and emits them as siblings of the core fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": record.created,
            "iso": time.strftime(
                "%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)
            ) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_ATTRS or key.startswith("_"):
                continue
            # Defensive: skip values that are not JSON-serialisable.
            try:
                json.dumps(value, default=str)
            except (TypeError, ValueError):
                continue
            payload[key] = value
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Concise human-readable formatter.

    Keeps the feel of the pre-P5 ``print()`` output so operators
    watching a tmux pane don't notice the switch.
    """

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        short = record.name.split(".")[-1]
        return f"{ts} {record.levelname[0]} {short}: {record.getMessage()}"


def configure_logging(
    *,
    level: str = "INFO",
    log_dir: str = "logs",
    human_filename: str = "grid.log",
    json_filename: str = "grid.jsonl",
    retention_days: int = 7,
    install_stdout: bool = True,
    install_human_file: bool = True,
    install_json_file: bool = True,
) -> logging.Logger:
    """Configure the root logger; return it for convenience.

    Idempotent: re-running with the same arguments replaces our
    previously installed handlers rather than stacking more.  Keeps
    third-party root handlers intact.
    """
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    try:
        root.setLevel(getattr(logging, level.upper()))
    except AttributeError:
        root.setLevel(logging.INFO)

    # Remove any of our previously installed handlers (tagged below).
    root.handlers = [
        h for h in root.handlers
        if not getattr(h, "_grid_managed", False)
    ]

    if install_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(HumanFormatter())
        sh._grid_managed = True  # type: ignore[attr-defined]
        root.addHandler(sh)

    if install_human_file:
        fh = logging.handlers.TimedRotatingFileHandler(
            os.path.join(log_dir, human_filename),
            when="midnight", utc=True, backupCount=retention_days,
        )
        fh.setFormatter(HumanFormatter())
        fh._grid_managed = True  # type: ignore[attr-defined]
        root.addHandler(fh)

    if install_json_file:
        jh = logging.handlers.TimedRotatingFileHandler(
            os.path.join(log_dir, json_filename),
            when="midnight", utc=True, backupCount=retention_days,
        )
        jh.setFormatter(JsonFormatter())
        jh._grid_managed = True  # type: ignore[attr-defined]
        root.addHandler(jh)

    return root

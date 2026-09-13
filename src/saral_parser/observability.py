"""Bounded, structured, per-job debug logging."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from langsmith import trace


class JobLogger:
    """Write JSON lines for one job without placing document content in normal logs."""

    def __init__(self, document_id: str, log_path: Path) -> None:
        self.document_id = document_id
        self.log_path = log_path
        self._logger = logging.getLogger(f"saral_parser.{document_id}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False
        self._logger.handlers.clear()
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)

    def event(self, level: int, stage: str, event: str, **details: Any) -> None:
        """Log a JSON-safe event with a stable document/job identifier."""
        payload = {
            "document_id": self.document_id,
            "stage": stage,
            "event": event,
            "details": details,
            "timestamp_epoch_seconds": round(time.time(), 3),
        }
        self._logger.log(level, json.dumps(payload, default=str, ensure_ascii=False))

    @contextmanager
    def stage(self, name: str, **configuration: Any) -> Iterator[None]:
        """Emit stage start, failure, and completion timing events."""
        started = time.perf_counter()
        self.event(logging.INFO, name, "started", configuration=configuration)
        try:
            with trace(
                f"docling.{name}",
                run_type="tool",
                inputs={"document_id": self.document_id, "configuration": configuration},
                metadata={"document_id": self.document_id, "component": "docling"},
                tags=["saral", "parser", "docling"],
            ):
                yield
        except Exception as exc:
            self.event(
                logging.ERROR,
                name,
                "failed",
                duration_seconds=round(time.perf_counter() - started, 3),
                error=repr(exc),
            )
            raise
        else:
            self.event(
                logging.INFO,
                name,
                "completed",
                duration_seconds=round(time.perf_counter() - started, 3),
            )

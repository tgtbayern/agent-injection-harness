"""Observability.

Every model call becomes a span written to a JSONL file, which is enough to
replay any turn offline and is what the replay UI reads. If `langfuse` is
installed and configured, the same spans are mirrored there; if it is not, the
harness works exactly the same. Tracing must never be a hard dependency of
running an experiment.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Span:
    name: str
    game_id: str
    player_id: int | None = None
    round: int | None = None
    step: int | None = None
    started_at: float = field(default_factory=time.time)
    duration_ms: int = 0
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "game_id": self.game_id,
            "player_id": self.player_id,
            "round": self.round,
            "step": self.step,
            "ts": self.started_at,
            "duration_ms": self.duration_ms,
            **self.data,
        }


class Tracer:
    def __init__(self, path: str | os.PathLike | None = None, game_id: str | None = None,
                 langfuse_client=None):
        self.game_id = game_id or uuid.uuid4().hex[:8]
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._langfuse = langfuse_client
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def span(self, name: str, **kwargs) -> Span:
        return Span(name=name, game_id=self.game_id, **kwargs)

    def emit(self, span: Span) -> None:
        record = span.to_dict()
        if self.path:
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        if self._langfuse is not None:  # pragma: no cover -- optional integration
            try:
                self._langfuse.trace(
                    name=span.name, id=f"{self.game_id}-{span.name}", metadata=record
                )
            except Exception:  # noqa: BLE001 -- tracing must never break a run
                pass

    def log(self, name: str, **data) -> None:
        self.emit(self.span(name, **{k: v for k, v in data.items()
                                     if k in {"player_id", "round", "step"}}) )

    @staticmethod
    def maybe_langfuse():  # pragma: no cover -- optional integration
        """Return a Langfuse client if the package and keys are both present."""
        if not os.getenv("LANGFUSE_PUBLIC_KEY"):
            return None
        try:
            from langfuse import Langfuse

            return Langfuse()
        except Exception:  # noqa: BLE001
            return None

"""Langfuse tracing.

Degrades to a no-op when no keys are set, so a clone with an empty .env still
runs. Observability that breaks the app when it is absent does not get kept.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager

from rag_agent.config import get_settings

log = logging.getLogger(__name__)


class _NullSpan:
    def update(self, **_): ...
    def end(self, **_): ...


class Tracer:
    def __init__(self):
        cfg = get_settings()
        self.enabled = bool(cfg.langfuse_public_key and cfg.langfuse_secret_key)
        self.client = None
        if not self.enabled:
            log.warning("Langfuse keys absent, tracing disabled")
            return
        from langfuse import Langfuse

        self.client = Langfuse(
            public_key=cfg.langfuse_public_key,
            secret_key=cfg.langfuse_secret_key,
            host=cfg.langfuse_host,
        )

    @contextmanager
    def trace(self, name: str, **metadata):
        if not self.enabled:
            yield _NullSpan()
            return
        span = self.client.trace(name=name, metadata=metadata)
        try:
            yield span
        except Exception as exc:
            span.update(level="ERROR", status_message=str(exc))
            raise
        finally:
            self.client.flush()

    def score(self, trace_id: str, name: str, value: float, comment: str = "") -> None:
        """Push a grader score back onto the trace so retrieval quality and
        grounding are queryable next to latency and token cost."""
        if self.enabled:
            self.client.score(trace_id=trace_id, name=name, value=value, comment=comment)

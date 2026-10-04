"""Answer cache.

Keyed on the normalised question, not the rewritten one: the rewrite is a
function of the retry count, so caching on it would miss every first attempt.
"""
from __future__ import annotations

import hashlib
import json

import redis.asyncio as redis

from rag_agent.config import get_settings


def _key(question: str) -> str:
    norm = " ".join(question.lower().split())
    return "rag:answer:" + hashlib.sha256(norm.encode()).hexdigest()[:32]


class AnswerCache:
    def __init__(self, client: redis.Redis | None = None):
        cfg = get_settings()
        self.ttl = cfg.cache_ttl_seconds
        self.client = client or redis.from_url(cfg.redis_url, decode_responses=True)

    async def get(self, question: str) -> dict | None:
        raw = await self.client.get(_key(question))
        return json.loads(raw) if raw else None

    async def set(self, question: str, payload: dict) -> None:
        await self.client.setex(_key(question), self.ttl, json.dumps(payload))

"""Redis: cache embeddings, session state, rate-limit counters."""

from __future__ import annotations

import io
import json
import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class RedisCache:
    """Async Redis cache for embeddings, sessions, and rate limiting.

    Uses ``redis.asyncio`` for non-blocking I/O. Numpy arrays are
    serialized to bytes via ``np.save`` / ``np.load`` with ``BytesIO``.

    Parameters
    ----------
    redis_url : str
        Redis connection URL (e.g. ``redis://localhost:6379``).
    """

    def __init__(self, redis_url: str = "redis://localhost:6379") -> None:
        self._redis_url = redis_url
        self._client = None

    async def _get_client(self):
        """Lazy-init the async Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis

                self._client = aioredis.from_url(
                    self._redis_url,
                    decode_responses=False,  # We need bytes for numpy
                )
                logger.info("Connected to Redis at %s", self._redis_url)
            except Exception as exc:
                logger.error("Failed to connect to Redis: %s", exc)
                raise
        return self._client

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            client = await self._get_client()
            return await client.ping()
        except Exception:
            return False

    # ── Embedding cache ─────────────────────────────────────────────────────

    async def get_embedding(self, text_hash: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding by text hash.

        Parameters
        ----------
        text_hash : str
            Hash of the text whose embedding is cached.

        Returns
        -------
        np.ndarray | None
            Cached embedding array, or None if not found.
        """
        client = await self._get_client()
        key = f"emb:{text_hash}"
        data = await client.get(key)
        if data is None:
            return None
        return _deserialize_numpy(data)

    async def set_embedding(
        self,
        text_hash: str,
        embedding: np.ndarray,
        ttl: int = 3600,
    ) -> None:
        """Cache an embedding by text hash.

        Parameters
        ----------
        text_hash : str
            Hash of the text.
        embedding : np.ndarray
            Embedding vector to cache.
        ttl : int
            Time-to-live in seconds (default: 1 hour).
        """
        client = await self._get_client()
        key = f"emb:{text_hash}"
        data = _serialize_numpy(embedding)
        await client.setex(key, ttl, data)

    # ── Session state ───────────────────────────────────────────────────────

    async def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve session data.

        Parameters
        ----------
        session_id : str
            Session identifier.

        Returns
        -------
        dict | None
            Session data or None.
        """
        client = await self._get_client()
        key = f"session:{session_id}"
        data = await client.get(key)
        if data is None:
            return None
        return json.loads(data)

    async def set_session(
        self,
        session_id: str,
        data: dict,
        ttl: int = 1800,
    ) -> None:
        """Store session data.

        Parameters
        ----------
        session_id : str
            Session identifier.
        data : dict
            Session data (must be JSON-serializable).
        ttl : int
            Time-to-live in seconds (default: 30 minutes).
        """
        client = await self._get_client()
        key = f"session:{session_id}"
        await client.setex(key, ttl, json.dumps(data))

    # ── Rate limiting ───────────────────────────────────────────────────────

    async def incr_rate(self, key: str, window_seconds: int = 60) -> int:
        """Increment a sliding window rate counter.

        Parameters
        ----------
        key : str
            Rate limit key (e.g. ``rl:<tenant_id>``).
        window_seconds : int
            Window size in seconds.

        Returns
        -------
        int
            Current count within the window.
        """
        client = await self._get_client()
        full_key = f"rate:{key}"

        pipe = client.pipeline()
        pipe.incr(full_key)
        pipe.expire(full_key, window_seconds)
        results = await pipe.execute()

        return int(results[0])

    # ── Job state ───────────────────────────────────────────────────────────

    async def get_job(self, job_id: str) -> Optional[dict]:
        """Get job status."""
        client = await self._get_client()
        data = await client.get(f"job:{job_id}")
        if data is None:
            return None
        return json.loads(data)

    async def set_job(self, job_id: str, data: dict, ttl: int = 86400) -> None:
        """Set job status (TTL: 24 hours)."""
        client = await self._get_client()
        await client.setex(f"job:{job_id}", ttl, json.dumps(data))

    # ── Webhook registrations ───────────────────────────────────────────────

    async def get_webhooks(self, tenant_id: str) -> list[dict]:
        """Get all webhook registrations for a tenant."""
        client = await self._get_client()
        data = await client.get(f"webhooks:{tenant_id}")
        if data is None:
            return []
        return json.loads(data)

    async def set_webhooks(self, tenant_id: str, webhooks: list[dict]) -> None:
        """Store webhook registrations for a tenant."""
        client = await self._get_client()
        await client.set(f"webhooks:{tenant_id}", json.dumps(webhooks))

    # ── API key storage ─────────────────────────────────────────────────────

    async def get_api_key_tenant(self, key_hash: str) -> Optional[str]:
        """Look up the tenant_id for an API key hash."""
        client = await self._get_client()
        data = await client.get(f"apikey:{key_hash}")
        if data is None:
            return None
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)

    async def set_api_key(self, key_hash: str, tenant_id: str) -> None:
        """Store an API key hash → tenant_id mapping."""
        client = await self._get_client()
        await client.set(f"apikey:{key_hash}", tenant_id.encode("utf-8"))

    async def delete_api_key(self, key_hash: str) -> None:
        """Delete an API key mapping."""
        client = await self._get_client()
        await client.delete(f"apikey:{key_hash}")


def _serialize_numpy(arr: np.ndarray) -> bytes:
    """Serialize a numpy array to bytes using np.save."""
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return buf.getvalue()


def _deserialize_numpy(data: bytes) -> np.ndarray:
    """Deserialize a numpy array from bytes."""
    buf = io.BytesIO(data)
    return np.load(buf, allow_pickle=False)

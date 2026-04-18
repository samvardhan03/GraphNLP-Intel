"""FastAPI dependency injection: db session, current tenant, api key."""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header, HTTPException, Request

from graphnlp.api.auth.api_keys import verify_api_key
from graphnlp.storage.neo4j_store import Neo4jGraphStore
from graphnlp.storage.redis_cache import RedisCache

# Module-level singletons set during app startup
_redis_cache: Optional[RedisCache] = None
_neo4j_store: Optional[Neo4jGraphStore] = None


def set_stores(redis: RedisCache, neo4j: Neo4jGraphStore) -> None:
    """Set module-level store references during app startup."""
    global _redis_cache, _neo4j_store
    _redis_cache = redis
    _neo4j_store = neo4j


async def get_redis() -> RedisCache:
    """FastAPI dependency: get the Redis cache instance."""
    if _redis_cache is None:
        raise HTTPException(status_code=503, detail="Redis service unavailable")
    return _redis_cache


async def get_neo4j_store() -> Neo4jGraphStore:
    """FastAPI dependency: get the Neo4j store instance."""
    if _neo4j_store is None:
        raise HTTPException(status_code=503, detail="Neo4j service unavailable")
    return _neo4j_store


async def get_current_tenant(request: Request) -> str:
    """FastAPI dependency: get the current tenant_id from request state."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Tenant context not established")
    return tenant_id

"""API key generation and verification for multi-tenant authentication."""

from __future__ import annotations

import hashlib
import logging
import uuid

from fastapi import Header, HTTPException

logger = logging.getLogger(__name__)

# Module-level reference to the Redis cache; set during app startup
_redis_cache = None


def set_redis_cache(cache) -> None:
    """Set the module-level Redis cache reference (called during app startup)."""
    global _redis_cache
    _redis_cache = cache


def _hash_key(raw_key: str) -> str:
    """Hash an API key using SHA-256.  Never store raw keys."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


async def generate_api_key(tenant_id: str) -> str:
    """Generate a new API key for a tenant.

    The raw key is returned once; only the SHA-256 hash is stored.

    Parameters
    ----------
    tenant_id : str
        Tenant identifier to associate with the key.

    Returns
    -------
    str
        The raw API key (``sk-<uuid4>``).  Store securely — it cannot be
        retrieved from the system after this call.
    """
    if _redis_cache is None:
        raise RuntimeError("Redis cache not initialized")

    raw_key = f"sk-{uuid.uuid4()}"
    key_hash = _hash_key(raw_key)

    await _redis_cache.set_api_key(key_hash, tenant_id)
    logger.info("Generated API key for tenant '%s'", tenant_id)

    return raw_key


async def verify_api_key(
    authorization: str = Header(..., alias="Authorization"),
) -> str:
    """FastAPI dependency to verify an API key from the Authorization header.

    Expects the header format: ``Bearer sk-<uuid>``.

    Parameters
    ----------
    authorization : str
        The ``Authorization`` header value.

    Returns
    -------
    str
        The tenant_id associated with the verified key.

    Raises
    ------
    HTTPException
        401 if the key is missing, malformed, or invalid.
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    # Parse "Bearer <key>" format
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Authorization header must be in format: Bearer <api_key>",
        )

    raw_key = parts[1].strip()
    if not raw_key.startswith("sk-"):
        raise HTTPException(status_code=401, detail="Invalid API key format")

    if _redis_cache is None:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")

    key_hash = _hash_key(raw_key)
    tenant_id = await _redis_cache.get_api_key_tenant(key_hash)

    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return tenant_id

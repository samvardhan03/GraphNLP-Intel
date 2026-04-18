"""ASGI middleware: resolve X-Tenant-ID or API key → tenant context."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Paths that don't require tenant context
_PUBLIC_PATHS = {"/health", "/health/", "/docs", "/openapi.json", "/redoc"}

# Module-level Redis cache reference
_redis_cache = None


def set_redis_cache(cache) -> None:
    """Set the module-level Redis cache reference."""
    global _redis_cache
    _redis_cache = cache


class TenantMiddleware(BaseHTTPMiddleware):
    """Resolve tenant context from request headers.

    Checks (in order):
    1. ``X-Tenant-ID`` header (direct tenant specification)
    2. ``Authorization: Bearer sk-...`` header (API key → tenant lookup)

    Sets ``request.state.tenant_id`` on every request.
    Public paths (``/health``, ``/docs``) are exempt from tenant requirements.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Allow public paths without tenant
        if request.url.path in _PUBLIC_PATHS:
            request.state.tenant_id = None
            return await call_next(request)

        tenant_id = None

        # 1. Check X-Tenant-ID header
        tenant_header = request.headers.get("X-Tenant-ID")
        if tenant_header:
            tenant_id = tenant_header.strip()

        # 2. Check Authorization header (API key → tenant)
        if tenant_id is None:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer sk-"):
                raw_key = auth_header.split(" ", 1)[1].strip()
                tenant_id = await self._resolve_tenant_from_key(raw_key)

        # Reject unauthenticated requests to protected endpoints
        if tenant_id is None:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Missing tenant context. "
                    "Provide X-Tenant-ID header or Authorization: Bearer <api_key>"
                },
            )

        request.state.tenant_id = tenant_id
        return await call_next(request)

    @staticmethod
    async def _resolve_tenant_from_key(raw_key: str) -> Optional[str]:
        """Look up tenant_id from an API key via Redis."""
        if _redis_cache is None:
            return None

        try:
            key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
            return await _redis_cache.get_api_key_tenant(key_hash)
        except Exception as exc:
            logger.warning("Tenant resolution from API key failed: %s", exc)
            return None

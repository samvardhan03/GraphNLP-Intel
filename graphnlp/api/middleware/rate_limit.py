"""ASGI middleware: per-tenant sliding window rate limiting."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from graphnlp.config import get_settings

logger = logging.getLogger(__name__)

# Module-level Redis cache reference
_redis_cache = None


def set_redis_cache(cache) -> None:
    """Set the module-level Redis cache reference."""
    global _redis_cache
    _redis_cache = cache


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant sliding window rate limiting middleware.

    Uses Redis to track request counts per tenant within a 60-second window.
    Returns 429 Too Many Requests with a ``Retry-After`` header when exceeded.

    The ``/health`` endpoint is exempt from rate limiting.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ("/health", "/health/"):
            return await call_next(request)

        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id is None or _redis_cache is None:
            return await call_next(request)

        try:
            settings = get_settings()
            key = f"rl:{tenant_id}"
            window = 60  # seconds

            count = await _redis_cache.incr_rate(key, window)

            if count > settings.rate_limit_per_minute:
                logger.warning(
                    "Rate limit exceeded for tenant '%s': %d/%d",
                    tenant_id,
                    count,
                    settings.rate_limit_per_minute,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded",
                        "limit": settings.rate_limit_per_minute,
                        "window_seconds": window,
                    },
                    headers={"Retry-After": str(window)},
                )
        except Exception as exc:
            # Don't block requests if Redis is down — log and continue
            logger.warning("Rate limiting check failed: %s", exc)

        response = await call_next(request)
        return response

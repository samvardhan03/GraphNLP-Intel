"""FastAPI application factory with lifecycle management."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from graphnlp.api.routes import analyze, graph, health, stream, webhooks
from graphnlp.api.middleware.rate_limit import RateLimitMiddleware
from graphnlp.api.middleware.tenant import TenantMiddleware
from graphnlp.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: startup and shutdown."""
    settings = get_settings()

    # Initialize Redis cache
    redis_cache = None
    try:
        from graphnlp.storage.redis_cache import RedisCache

        redis_cache = RedisCache(settings.redis_url)
        # Set Redis on middleware and auth modules
        from graphnlp.api.middleware.rate_limit import set_redis_cache as set_rl_redis
        from graphnlp.api.middleware.tenant import set_redis_cache as set_tenant_redis
        from graphnlp.api.auth.api_keys import set_redis_cache as set_auth_redis

        set_rl_redis(redis_cache)
        set_tenant_redis(redis_cache)
        set_auth_redis(redis_cache)
        logger.info("Redis cache initialized")
    except Exception as exc:
        logger.warning("Redis initialization failed: %s (continuing without cache)", exc)

    # Initialize Neo4j store
    neo4j_store = None
    try:
        from graphnlp.storage.neo4j_store import Neo4jGraphStore

        neo4j_store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
        )
        logger.info("Neo4j store initialized")
    except Exception as exc:
        logger.warning("Neo4j initialization failed: %s (continuing without store)", exc)

    # Set stores in deps module
    from graphnlp.api.deps import set_stores

    set_stores(redis_cache, neo4j_store)

    yield

    # Shutdown: close connections
    if redis_cache:
        try:
            await redis_cache.close()
        except Exception:
            pass
    if neo4j_store:
        try:
            await neo4j_store.close()
        except Exception:
            pass

    logger.info("Application shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="GraphNLP Intel API",
        version="0.1.0",
        description="Hybrid Graph-NLP Intelligence Platform API",
        lifespan=lifespan,
    )

    # Add middleware (order matters: outermost first)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(TenantMiddleware)

    # Include route modules
    app.include_router(health.router)
    app.include_router(analyze.router, prefix="/v1")
    app.include_router(graph.router, prefix="/v1")
    app.include_router(stream.router, prefix="/v1")
    app.include_router(webhooks.router, prefix="/v1")

    # Global exception handler for service unavailability
    @app.exception_handler(503)
    async def service_unavailable_handler(request: Request, exc):
        return JSONResponse(
            status_code=503,
            content={"detail": "Service temporarily unavailable. Please try later."},
        )

    return app


app = create_app()

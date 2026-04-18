"""GET /health — liveness + readiness probes."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Liveness probe — always returns 200 if the service is running."""
    return {
        "status": "healthy",
        "service": "graphnlp-intel",
        "version": "0.1.0",
    }

"""Celery app — broker: Redis, backend: Redis."""

from __future__ import annotations

from graphnlp.config import get_settings

settings = get_settings()

try:
    from celery import Celery

    celery_app = Celery(
        "graphnlp",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["graphnlp.queue.tasks"],
    )

    celery_app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_track_started=True,
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        result_expires=86400,  # 24 hours
    )

except ImportError:
    # Celery not installed — provide a stub
    celery_app = None

"""Celery tasks: process_documents, dispatch_webhook."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from graphnlp.queue.worker import celery_app

    if celery_app is None:
        raise ImportError("Celery not configured")

    @celery_app.task(bind=True, max_retries=3, acks_late=True)
    def process_documents(
        self,
        documents: list[str],
        domain: str,
        graph_id: str,
        tenant_id: str,
    ) -> dict[str, Any]:
        """Full pipeline: ingest → extract → build graph → save to Neo4j.

        Runs with exponential backoff on retry.
        """
        from graphnlp.config import get_settings
        from graphnlp.storage.redis_cache import RedisCache

        settings = get_settings()

        # Update job status to "processing"
        _update_job_status_sync(settings.redis_url, graph_id, "processing", tenant_id)

        try:
            from graphnlp.pipeline import Pipeline

            pipe = Pipeline(domain=domain)
            result = pipe.run(documents)

            # Save to Neo4j
            try:
                from graphnlp.storage.neo4j_store import Neo4jGraphStore

                store = Neo4jGraphStore(
                    uri=settings.neo4j_uri,
                    user=settings.neo4j_user,
                    password=settings.neo4j_password,
                )
                asyncio.get_event_loop().run_until_complete(
                    store.save(graph_id, result.graph, tenant_id)
                )
            except Exception as exc:
                logger.warning("Failed to save to Neo4j: %s", exc)

            # Update job status to "ready"
            _update_job_status_sync(settings.redis_url, graph_id, "ready", tenant_id)

            logger.info(
                "Task complete: graph_id=%s, nodes=%d, edges=%d",
                graph_id,
                result.graph.number_of_nodes(),
                result.graph.number_of_edges(),
            )

            return {
                "graph_id": graph_id,
                "status": "ready",
                "node_count": result.graph.number_of_nodes(),
                "edge_count": result.graph.number_of_edges(),
            }

        except Exception as exc:
            logger.error("Task failed (graph_id=%s): %s", graph_id, exc)
            _update_job_status_sync(
                settings.redis_url, graph_id, "failed", tenant_id, error=str(exc)
            )
            # Exponential backoff retry
            raise self.retry(
                exc=exc,
                countdown=2**self.request.retries,
            )

    @celery_app.task(max_retries=2)
    def dispatch_webhook(
        tenant_id: str,
        event: str,
        payload: dict,
    ) -> dict:
        """HTTP POST to all registered webhooks for a tenant."""
        try:
            from graphnlp.webhooks.dispatcher import dispatch

            asyncio.get_event_loop().run_until_complete(
                dispatch(tenant_id, event, payload)
            )
            return {"status": "dispatched", "event": event}

        except Exception as exc:
            logger.error("Webhook dispatch failed: %s", exc)
            return {"status": "failed", "error": str(exc)}

except ImportError:
    # Celery not installed — provide stub functions
    def process_documents(*args, **kwargs):
        raise RuntimeError("Celery is not installed. Install with: pip install celery[redis]")

    def dispatch_webhook(*args, **kwargs):
        raise RuntimeError("Celery is not installed. Install with: pip install celery[redis]")


def _update_job_status_sync(
    redis_url: str,
    graph_id: str,
    status: str,
    tenant_id: str,
    error: str | None = None,
) -> None:
    """Synchronously update job status in Redis (for Celery worker context)."""
    try:
        import redis

        r = redis.from_url(redis_url)
        import json

        data = {"status": status, "graph_id": graph_id, "tenant_id": tenant_id}
        if error:
            data["error"] = error
        r.setex(f"job:{graph_id}", 86400, json.dumps(data))
    except Exception as exc:
        logger.warning("Failed to update job status in Redis: %s", exc)

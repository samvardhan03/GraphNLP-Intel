"""HTTP POST to registered URLs when graph topology/sentiment shifts."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def dispatch(
    tenant_id: str,
    event: str,
    payload: dict[str, Any],
) -> None:
    """Load registered webhook URLs from Redis and POST the payload to each.

    Payloads are signed with ``X-GraphNLP-Signature: sha256=<hmac>`` using
    the tenant's webhook secret (if configured).

    This is fire-and-forget: failures are logged but don't raise exceptions.

    Parameters
    ----------
    tenant_id : str
        Tenant whose webhooks should be invoked.
    event : str
        Event type (e.g. ``graph.changed``, ``sentiment.alert``).
    payload : dict
        JSON-serializable event payload.
    """
    try:
        import httpx
    except ImportError:
        logger.error("httpx is required for webhook dispatch")
        return

    # Load webhook registrations from Redis
    try:
        from graphnlp.config import get_settings
        from graphnlp.storage.redis_cache import RedisCache

        settings = get_settings()
        cache = RedisCache(settings.redis_url)
        webhooks = await cache.get_webhooks(tenant_id)
        await cache.close()
    except Exception as exc:
        logger.error("Failed to load webhooks for tenant '%s': %s", tenant_id, exc)
        return

    if not webhooks:
        return

    # Prepare payload body
    body = json.dumps(
        {
            "event": event,
            "tenant_id": tenant_id,
            "data": payload,
        },
        default=str,
    )

    async with httpx.AsyncClient(timeout=5.0) as client:
        for webhook in webhooks:
            # Filter by subscribed events
            subscribed_events = webhook.get("events", [])
            if event not in subscribed_events:
                continue

            if not webhook.get("active", True):
                continue

            url = webhook.get("url", "")
            secret = webhook.get("secret")

            headers = {
                "Content-Type": "application/json",
                "X-GraphNLP-Event": event,
            }

            # HMAC signature
            if secret:
                signature = hmac.new(
                    secret.encode("utf-8"),
                    body.encode("utf-8"),
                    hashlib.sha256,
                ).hexdigest()
                headers["X-GraphNLP-Signature"] = f"sha256={signature}"

            try:
                response = await client.post(url, content=body, headers=headers)
                logger.info(
                    "Webhook dispatched: %s → %s (status=%d)",
                    event,
                    url,
                    response.status_code,
                )
            except Exception as exc:
                logger.warning(
                    "Webhook delivery failed: %s → %s: %s",
                    event,
                    url,
                    exc,
                )

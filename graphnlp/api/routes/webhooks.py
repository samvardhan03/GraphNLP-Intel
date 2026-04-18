"""Webhook registration and management routes."""

from __future__ import annotations

import uuid
from typing import Optional
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from graphnlp.api.deps import get_current_tenant, get_redis

router = APIRouter(tags=["webhooks"])

# Valid webhook event types
VALID_EVENTS = {"graph.changed", "sentiment.alert", "job.completed", "job.failed"}


class WebhookRegisterRequest(BaseModel):
    """Request body for webhook registration."""

    url: str = Field(..., description="HTTPS webhook URL")
    events: list[str] = Field(
        ..., min_length=1, description="List of event types to subscribe to"
    )
    secret: Optional[str] = Field(
        None, description="Optional secret for HMAC signature verification"
    )


class WebhookResponse(BaseModel):
    """Response for a webhook registration."""

    id: str
    url: str
    events: list[str]
    active: bool = True


@router.post("/webhooks", response_model=WebhookResponse)
async def register_webhook(
    body: WebhookRegisterRequest,
    tenant_id: str = Depends(get_current_tenant),
    redis=Depends(get_redis),
):
    """Register a new webhook for the current tenant."""
    # Validate URL is HTTPS
    parsed = urlparse(body.url)
    if parsed.scheme != "https":
        raise HTTPException(
            status_code=400,
            detail="Webhook URL must use HTTPS",
        )

    # Validate event types
    invalid = set(body.events) - VALID_EVENTS
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event types: {invalid}. Valid: {VALID_EVENTS}",
        )

    webhook_id = str(uuid.uuid4())
    webhook = {
        "id": webhook_id,
        "url": body.url,
        "events": body.events,
        "secret": body.secret,
        "active": True,
    }

    # Store in Redis (per-tenant list)
    existing = await redis.get_webhooks(tenant_id)
    existing.append(webhook)
    await redis.set_webhooks(tenant_id, existing)

    return WebhookResponse(id=webhook_id, url=body.url, events=body.events)


@router.get("/webhooks", response_model=list[WebhookResponse])
async def list_webhooks(
    tenant_id: str = Depends(get_current_tenant),
    redis=Depends(get_redis),
):
    """List all registered webhooks for the current tenant."""
    webhooks = await redis.get_webhooks(tenant_id)
    return [
        WebhookResponse(
            id=w["id"],
            url=w["url"],
            events=w["events"],
            active=w.get("active", True),
        )
        for w in webhooks
    ]


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: str,
    tenant_id: str = Depends(get_current_tenant),
    redis=Depends(get_redis),
):
    """Unregister a webhook."""
    webhooks = await redis.get_webhooks(tenant_id)
    updated = [w for w in webhooks if w.get("id") != webhook_id]

    if len(updated) == len(webhooks):
        raise HTTPException(status_code=404, detail="Webhook not found")

    await redis.set_webhooks(tenant_id, updated)
    return {"status": "deleted", "id": webhook_id}

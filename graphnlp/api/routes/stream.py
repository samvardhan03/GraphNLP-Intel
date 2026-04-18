"""WebSocket /v1/stream — real-time document processing."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["stream"])
logger = logging.getLogger(__name__)


@router.websocket("/stream")
async def stream_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time document processing.

    Client sends JSON messages: ``{"document": "...", "domain": "finance"}``
    Server responds with partial graph updates as each document is processed.
    """
    await websocket.accept()

    # Get tenant from query params or initial handshake
    tenant_id = websocket.query_params.get("tenant_id", "default")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            document = msg.get("document", "")
            domain = msg.get("domain", "generic")

            if not document.strip():
                await websocket.send_json({"error": "Empty document"})
                continue

            # Process document in background to avoid blocking
            try:
                result = await _process_document(document, domain, tenant_id)
                await websocket.send_json(result)
            except Exception as exc:
                logger.error("Stream processing failed: %s", exc)
                await websocket.send_json({"error": str(exc)})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected (tenant: %s)", tenant_id)


async def _process_document(
    document: str,
    domain: str,
    tenant_id: str,
) -> dict:
    """Process a single document through the pipeline.

    Runs in a thread pool to avoid blocking the event loop.
    """
    from concurrent.futures import ThreadPoolExecutor

    def _run():
        from graphnlp.pipeline import Pipeline

        pipe = Pipeline(domain=domain)
        result = pipe.run([document])
        return result.summary()

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as pool:
        summary = await loop.run_in_executor(pool, _run)

    return {"status": "processed", "summary": summary}

"""POST /v1/analyze — submit documents for NLP pipeline processing."""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from graphnlp.api.deps import get_current_tenant, get_neo4j_store, get_redis

router = APIRouter(tags=["analyze"])


class AnalyzeRequest(BaseModel):
    """Request body for the analyze endpoint."""

    documents: list[str] = Field(..., min_length=1, description="List of document texts")
    domain: str = Field("generic", description="Domain adapter to use")
    async_mode: bool = Field(False, alias="async", description="Run asynchronously via Celery")

    model_config = {"populate_by_name": True}


class AnalyzeResponse(BaseModel):
    """Response from the analyze endpoint."""

    job_id: str
    graph_id: str
    status: str  # "queued" | "processing" | "ready" | "failed"


class JobStatusResponse(BaseModel):
    """Response from the job status polling endpoint."""

    job_id: str
    status: str
    graph_id: Optional[str] = None
    error: Optional[str] = None


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_documents(
    body: AnalyzeRequest,
    tenant_id: str = Depends(get_current_tenant),
    redis=Depends(get_redis),
):
    """Submit documents for analysis.

    If ``async: true``, enqueues a Celery task and returns immediately.
    Otherwise, runs the pipeline synchronously (recommended for < 10 docs).
    """
    job_id = str(uuid.uuid4())
    graph_id = str(uuid.uuid4())

    if body.async_mode:
        # Enqueue Celery task
        try:
            from graphnlp.queue.tasks import process_documents

            process_documents.delay(
                documents=body.documents,
                domain=body.domain,
                graph_id=graph_id,
                tenant_id=tenant_id,
            )
            status = "queued"
        except Exception as exc:
            # Celery not available — fall back to sync
            status = await _run_sync_pipeline(
                body.documents, body.domain, graph_id, tenant_id
            )
    else:
        status = await _run_sync_pipeline(
            body.documents, body.domain, graph_id, tenant_id
        )

    # Store job state in Redis
    await redis.set_job(
        job_id,
        {"status": status, "graph_id": graph_id, "tenant_id": tenant_id},
    )

    return AnalyzeResponse(job_id=job_id, graph_id=graph_id, status=status)


@router.get("/analyze/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    tenant_id: str = Depends(get_current_tenant),
    redis=Depends(get_redis),
):
    """Poll the status of an analysis job."""
    job = await redis.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Verify tenant owns this job
    if job.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        graph_id=job.get("graph_id"),
        error=job.get("error"),
    )


async def _run_sync_pipeline(
    documents: list[str],
    domain: str,
    graph_id: str,
    tenant_id: str,
) -> str:
    """Run the NLP pipeline synchronously."""
    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        from graphnlp.pipeline import Pipeline

        def _run():
            pipe = Pipeline(domain=domain)
            result = pipe.run(documents)
            return result

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(pool, _run)

        # Save to Neo4j if available
        try:
            from graphnlp.api.deps import _neo4j_store

            if _neo4j_store is not None:
                await _neo4j_store.save(graph_id, result.graph, tenant_id)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Failed to save to Neo4j: %s", exc)

        return "ready"

    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Pipeline execution failed: %s", exc)
        return "failed"

"""Sync + async HTTP client wrapping the GraphNLP Intel API."""

from __future__ import annotations

import time
from typing import Any, Optional

import httpx


class GraphNLPClient:
    """Synchronous Python SDK for the GraphNLP Intel API.

    Parameters
    ----------
    api_key : str
        API key in ``sk-<uuid>`` format.
    base_url : str
        Base URL of the GraphNLP API.
    timeout : float
        Request timeout in seconds.
    max_retries : int
        Number of retries for failed requests.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.graphnlp.io",
        timeout: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._max_retries = max_retries
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> dict:
        """Make an HTTP request with retry logic."""
        url = f"{self.base_url}{path}"
        last_exc = None

        for attempt in range(self._max_retries):
            try:
                r = httpx.request(
                    method,
                    url,
                    headers=self._headers,
                    timeout=self._timeout,
                    **kwargs,
                )
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    retry_after = int(exc.response.headers.get("Retry-After", 5))
                    time.sleep(retry_after)
                    last_exc = exc
                    continue
                raise
            except httpx.RequestError as exc:
                last_exc = exc
                time.sleep(2**attempt)
                continue

        raise last_exc or RuntimeError("Request failed after retries")

    def analyze(
        self,
        documents: list[str],
        domain: str = "generic",
        async_mode: bool = False,
    ) -> dict:
        """Submit documents for analysis.

        Parameters
        ----------
        documents : list[str]
            List of document texts.
        domain : str
            Domain adapter name.
        async_mode : bool
            If True, run asynchronously (returns job_id for polling).

        Returns
        -------
        dict
            ``{job_id, graph_id, status}``
        """
        return self._request(
            "POST",
            "/v1/analyze",
            json={"documents": documents, "domain": domain, "async": async_mode},
        )

    def poll_job(self, job_id: str, poll_interval: float = 2.0, max_wait: float = 300.0) -> dict:
        """Poll a job until completion.

        Parameters
        ----------
        job_id : str
            Job ID from an analyze() call.
        poll_interval : float
            Seconds between polls.
        max_wait : float
            Maximum seconds to wait.

        Returns
        -------
        dict
            Final job status.
        """
        start = time.time()
        while time.time() - start < max_wait:
            status = self._request("GET", f"/v1/analyze/{job_id}")
            if status.get("status") in ("ready", "failed"):
                return status
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job_id} did not complete within {max_wait}s")

    def get_graph(self, graph_id: str) -> dict:
        """Fetch the full graph as D3-compatible JSON."""
        return self._request("GET", f"/v1/graph/{graph_id}")

    def get_summary(self, graph_id: str) -> dict:
        """Get a graph summary: communities, sentiments, anomalies."""
        return self._request("GET", f"/v1/graph/{graph_id}/summary")

    def download_html(self, graph_id: str) -> str:
        """Download the Pyvis HTML visualization."""
        url = f"{self.base_url}/v1/graph/{graph_id}/html"
        r = httpx.get(url, headers=self._headers, timeout=self._timeout)
        r.raise_for_status()
        return r.text

    def list_webhooks(self) -> list[dict]:
        """List registered webhooks."""
        return self._request("GET", "/v1/webhooks")

    def register_webhook(
        self, url: str, events: list[str], secret: Optional[str] = None
    ) -> dict:
        """Register a new webhook."""
        body: dict[str, Any] = {"url": url, "events": events}
        if secret:
            body["secret"] = secret
        return self._request("POST", "/v1/webhooks", json=body)

    def delete_webhook(self, webhook_id: str) -> dict:
        """Unregister a webhook."""
        return self._request("DELETE", f"/v1/webhooks/{webhook_id}")


class AsyncGraphNLPClient:
    """Async Python SDK for the GraphNLP Intel API.

    Uses ``httpx.AsyncClient`` for non-blocking requests.

    Parameters
    ----------
    api_key : str
        API key in ``sk-<uuid>`` format.
    base_url : str
        Base URL of the GraphNLP API.
    timeout : float
        Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.graphnlp.io",
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def close(self):
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def analyze(
        self,
        documents: list[str],
        domain: str = "generic",
        async_mode: bool = False,
    ) -> dict:
        """Submit documents for analysis."""
        r = await self._client.post(
            "/v1/analyze",
            json={"documents": documents, "domain": domain, "async": async_mode},
        )
        r.raise_for_status()
        return r.json()

    async def get_graph(self, graph_id: str) -> dict:
        """Fetch the full graph as D3-compatible JSON."""
        r = await self._client.get(f"/v1/graph/{graph_id}")
        r.raise_for_status()
        return r.json()

    async def get_summary(self, graph_id: str) -> dict:
        """Get graph summary."""
        r = await self._client.get(f"/v1/graph/{graph_id}/summary")
        r.raise_for_status()
        return r.json()

"""Graph API routes — fetch, summarize, and export knowledge graphs."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse

from graphnlp.api.deps import get_current_tenant, get_neo4j_store, get_redis

router = APIRouter(tags=["graph"])


@router.get("/graph/{graph_id}")
async def get_graph(
    graph_id: str,
    tenant_id: str = Depends(get_current_tenant),
    store=Depends(get_neo4j_store),
):
    """Return the full graph as D3-compatible JSON.

    Format: ``{nodes: [...], links: [...]}``
    """
    try:
        graph = await store.load(graph_id, tenant_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load graph: {exc}")

    if graph.number_of_nodes() == 0:
        raise HTTPException(status_code=404, detail="Graph not found or empty")

    from graphnlp.viz.d3_export import export_d3_json

    return export_d3_json(graph)


@router.get("/graph/{graph_id}/summary")
async def get_graph_summary(
    graph_id: str,
    tenant_id: str = Depends(get_current_tenant),
    store=Depends(get_neo4j_store),
):
    """Return a summary: top clusters, sentiment scores, anomalies."""
    try:
        graph = await store.load(graph_id, tenant_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load graph: {exc}")

    if graph.number_of_nodes() == 0:
        raise HTTPException(status_code=404, detail="Graph not found or empty")

    from graphnlp.graph.community import CommunityDetector
    from graphnlp.graph.gnn import GraphGNN

    # Run community detection
    detector = CommunityDetector()
    communities = detector.detect(graph)

    # Run sentiment analysis
    gnn = GraphGNN()
    sentiments = gnn.run(graph)

    # Top communities
    top = detector.top_communities(graph, n=5, sentiments=sentiments)

    # Anomalies: nodes with extreme sentiment
    anomalies = [
        {"node": node, "sentiment": score, "type": graph.nodes[node].get("type", "MISC")}
        for node, score in sorted(sentiments.items(), key=lambda x: abs(x[1]), reverse=True)
        if abs(score) > 0.5
    ][:10]

    return {
        "graph_id": graph_id,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "communities": top,
        "anomalies": anomalies,
        "avg_sentiment": (
            sum(sentiments.values()) / len(sentiments) if sentiments else 0.0
        ),
    }


@router.get("/graph/{graph_id}/html", response_class=HTMLResponse)
async def get_graph_html(
    graph_id: str,
    tenant_id: str = Depends(get_current_tenant),
    store=Depends(get_neo4j_store),
):
    """Return an interactive Pyvis HTML visualization for download."""
    try:
        graph = await store.load(graph_id, tenant_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Failed to load graph: {exc}")

    if graph.number_of_nodes() == 0:
        raise HTTPException(status_code=404, detail="Graph not found or empty")

    from graphnlp.graph.community import CommunityDetector
    from graphnlp.graph.gnn import GraphGNN
    from graphnlp.viz.pyvis_renderer import render_html

    detector = CommunityDetector()
    communities = detector.detect(graph)
    gnn = GraphGNN()
    sentiments = gnn.run(graph)

    html = render_html(graph, sentiments, communities)
    return HTMLResponse(content=html, media_type="text/html")

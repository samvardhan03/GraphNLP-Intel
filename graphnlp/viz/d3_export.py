"""Export graph as D3.js-compatible JSON (nodes + links)."""

from __future__ import annotations

from typing import Any

import networkx as nx


def export_d3_json(
    graph: nx.DiGraph,
    sentiment: dict[str, float] | None = None,
    communities: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Export a knowledge graph as D3.js force-directed JSON.

    Parameters
    ----------
    graph : nx.DiGraph
        Knowledge graph.
    sentiment : dict[str, float] | None
        Node sentiment scores.
    communities : dict[str, int] | None
        Community assignments.

    Returns
    -------
    dict
        D3-compatible format: ``{"nodes": [...], "links": [...]}``.
    """
    sentiment = sentiment or {}
    communities = communities or {}

    nodes: list[dict[str, Any]] = []
    for node_id in graph.nodes():
        data = graph.nodes[node_id]
        nodes.append(
            {
                "id": str(node_id),
                "label": data.get("label", str(node_id)),
                "type": data.get("type", "MISC"),
                "mention_count": data.get("mention_count", 1),
                "sentiment": sentiment.get(str(node_id), 0.0),
                "community": communities.get(str(node_id), 0),
            }
        )

    links: list[dict[str, Any]] = []
    for src, dst, data in graph.edges(data=True):
        links.append(
            {
                "source": str(src),
                "target": str(dst),
                "predicate": data.get("predicate", ""),
                "confidence": data.get("confidence", 1.0),
                "weight": data.get("weight", 0.5),
            }
        )

    return {
        "nodes": nodes,
        "links": links,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(links),
        },
    }

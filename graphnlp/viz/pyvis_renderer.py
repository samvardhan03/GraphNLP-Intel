"""Render NetworkX graph → self-contained interactive HTML via Pyvis."""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)


def _sentiment_to_color(score: float) -> str:
    """Map a sentiment score in [-1, 1] to a diverging color (red → amber → green)."""
    # Clamp
    score = max(-1.0, min(1.0, score))

    if score >= 0.3:
        # Green spectrum
        intensity = int(100 + 155 * min(score, 1.0))
        return f"rgba(0, {intensity}, 80, 0.9)"
    elif score <= -0.3:
        # Red spectrum
        intensity = int(100 + 155 * min(abs(score), 1.0))
        return f"rgba({intensity}, 40, 40, 0.9)"
    else:
        # Amber / neutral
        return "rgba(255, 193, 7, 0.9)"


def _mention_to_size(count: int, min_size: int = 10, max_size: int = 40) -> int:
    """Map mention count to node size."""
    return max(min_size, min(max_size, min_size + count * 3))


def render_html(
    graph: nx.DiGraph,
    sentiment: dict[str, float] | None = None,
    communities: dict[str, int] | None = None,
) -> str:
    """Render a knowledge graph as self-contained interactive Pyvis HTML.

    Parameters
    ----------
    graph : nx.DiGraph
        Knowledge graph with node attributes.
    sentiment : dict[str, float] | None
        Node sentiment scores in [-1, 1].
    communities : dict[str, int] | None
        Community assignments per node.

    Returns
    -------
    str
        Complete HTML string with embedded JS visualization.
    """
    try:
        from pyvis.network import Network
    except ImportError:
        raise ImportError("pyvis is required for visualization. Install with: pip install pyvis")

    sentiment = sentiment or {}
    communities = communities or {}

    # Create Pyvis network
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#e0e0e0",
        directed=True,
        select_menu=True,
        filter_menu=True,
    )

    # Configure physics
    net.set_options("""
    {
        "physics": {
            "barnesHut": {
                "gravitationalConstant": -3000,
                "centralGravity": 0.3,
                "springLength": 120,
                "springConstant": 0.04,
                "damping": 0.09
            },
            "stabilization": {
                "iterations": 150
            }
        },
        "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "navigationButtons": true
        },
        "edges": {
            "smooth": {
                "type": "continuous",
                "forceDirection": "none"
            },
            "arrows": {
                "to": {"enabled": true, "scaleFactor": 0.5}
            }
        }
    }
    """)

    # Add nodes
    for node_id in graph.nodes():
        data = graph.nodes[node_id]
        label = data.get("label", str(node_id))
        node_type = data.get("type", "MISC")
        mention_count = data.get("mention_count", 1)
        sent_score = sentiment.get(str(node_id), 0.0)
        community = communities.get(str(node_id), 0)

        color = _sentiment_to_color(sent_score)
        size = _mention_to_size(mention_count)

        title = (
            f"<b>{label}</b><br>"
            f"Type: {node_type}<br>"
            f"Sentiment: {sent_score:.3f}<br>"
            f"Mentions: {mention_count}<br>"
            f"Community: {community}"
        )

        net.add_node(
            str(node_id),
            label=label,
            title=title,
            color=color,
            size=size,
            group=community,
            shape="dot",
            borderWidth=2,
            borderWidthSelected=4,
        )

    # Add edges
    for src, dst, data in graph.edges(data=True):
        predicate = data.get("predicate", "")
        confidence = data.get("confidence", 1.0)
        weight = data.get("weight", 0.5)

        edge_width = max(1, int(weight * 4))
        edge_color = f"rgba(150, 150, 200, {min(confidence, 1.0) * 0.7:.2f})"

        net.add_edge(
            str(src),
            str(dst),
            title=f"{predicate} (conf: {confidence:.2f})",
            label=predicate if len(predicate) < 20 else "",
            width=edge_width,
            color=edge_color,
        )

    return net.generate_html()

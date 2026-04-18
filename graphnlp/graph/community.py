"""Louvain + Girvan-Newman community detection on knowledge graphs."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# Threshold below which we use Girvan-Newman instead of Louvain
_SMALL_GRAPH_THRESHOLD = 100


class CommunityDetector:
    """Detect communities in a knowledge graph.

    Primary algorithm: Louvain (via ``python-louvain``).
    Fallback: Girvan-Newman (for small graphs < 100 nodes or when
    ``python-louvain`` is not installed).
    """

    def detect(self, graph: nx.Graph | nx.DiGraph) -> dict[str, int]:
        """Assign each node to a community.

        Parameters
        ----------
        graph : nx.Graph | nx.DiGraph
            Input graph.  For directed graphs, an undirected view is used.

        Returns
        -------
        dict[str, int]
            Mapping of ``{node_id: community_id}``.
        """
        if graph.number_of_nodes() == 0:
            return {}

        # Work on undirected graph for community detection
        G = graph.to_undirected() if graph.is_directed() else graph

        # Remove isolated nodes for better community detection
        if G.number_of_edges() == 0:
            return {n: 0 for n in G.nodes()}

        # Choose algorithm based on graph size and availability
        if G.number_of_nodes() < _SMALL_GRAPH_THRESHOLD:
            try:
                return self._girvan_newman(G)
            except Exception:
                pass

        # Primary: Louvain
        try:
            return self._louvain(G)
        except ImportError:
            logger.warning("python-louvain not installed; using Girvan-Newman fallback")
            return self._girvan_newman(G)

    def top_communities(
        self,
        graph: nx.Graph | nx.DiGraph,
        n: int = 5,
        sentiments: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-N communities by size.

        Parameters
        ----------
        graph : nx.Graph | nx.DiGraph
            Input graph.
        n : int
            Number of top communities to return.
        sentiments : dict[str, float] | None
            Optional node sentiment scores for computing average sentiment.

        Returns
        -------
        list[dict]
            Each dict has: ``id``, ``size``, ``top_nodes``, ``avg_sentiment``,
            ``dominant_type``.
        """
        communities = self.detect(graph)
        if not communities:
            return []

        sentiments = sentiments or {}

        # Group nodes by community
        groups: dict[int, list[str]] = {}
        for node, cid in communities.items():
            groups.setdefault(cid, []).append(node)

        # Build community info
        results: list[dict[str, Any]] = []
        for cid, members in groups.items():
            # Top nodes by mention count
            top_nodes = sorted(
                members,
                key=lambda n: graph.nodes[n].get("mention_count", 0)
                if n in graph.nodes
                else 0,
                reverse=True,
            )[:5]

            # Average sentiment
            member_sentiments = [sentiments.get(m, 0.0) for m in members]
            avg_sentiment = (
                sum(member_sentiments) / len(member_sentiments)
                if member_sentiments
                else 0.0
            )

            # Dominant entity type
            type_counts: Counter = Counter()
            for m in members:
                if m in graph.nodes:
                    etype = graph.nodes[m].get("type", "MISC")
                    type_counts[etype] += 1
            dominant_type = type_counts.most_common(1)[0][0] if type_counts else "MISC"

            results.append(
                {
                    "id": cid,
                    "size": len(members),
                    "top_nodes": top_nodes,
                    "avg_sentiment": round(avg_sentiment, 4),
                    "dominant_type": dominant_type,
                }
            )

        # Sort by size descending, return top N
        results.sort(key=lambda c: c["size"], reverse=True)
        return results[:n]

    @staticmethod
    def _louvain(G: nx.Graph) -> dict[str, int]:
        """Community detection using the Louvain algorithm."""
        import community as community_louvain

        partition = community_louvain.best_partition(G)
        return partition

    @staticmethod
    def _girvan_newman(G: nx.Graph) -> dict[str, int]:
        """Community detection using Girvan-Newman (edge betweenness removal).

        Returns communities from the first meaningful split.
        """
        from networkx.algorithms.community import girvan_newman

        comp = girvan_newman(G)
        # Take the first level of communities
        try:
            communities = next(comp)
        except StopIteration:
            return {n: 0 for n in G.nodes()}

        result: dict[str, int] = {}
        for cid, community in enumerate(communities):
            for node in community:
                result[node] = cid

        return result

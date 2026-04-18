"""Customer feedback schema: PRODUCT, ISSUE, SENTIMENT, FEATURE."""

from __future__ import annotations

import re

import networkx as nx

from graphnlp.adapters.base import DomainAdapter, register_adapter


@register_adapter
class FeedbackAdapter(DomainAdapter):
    """Domain adapter for customer feedback / review analysis.

    Entity types: PRODUCT, ISSUE, SENTIMENT, FEATURE, PERSON, ORG.
    """

    domain = "feedback"
    entity_types = ["PRODUCT", "ISSUE", "SENTIMENT", "FEATURE", "PERSON", "ORG"]

    def preprocess(self, text: str) -> str:
        """Clean feedback text: normalize ratings, strip formatting."""
        processed = text

        # Normalize star ratings: "5/5", "4 out of 5", "★★★★" → "[RATING:X]"
        processed = re.sub(r"(\d)\s*/\s*5", r"[RATING:\1]", processed)
        processed = re.sub(r"(\d)\s+out\s+of\s+5", r"[RATING:\1]", processed, flags=re.IGNORECASE)
        processed = re.sub(r"[★]{1,5}", lambda m: f"[RATING:{len(m.group())}]", processed)

        # Collapse excessive punctuation
        processed = re.sub(r"[!]{2,}", "!", processed)
        processed = re.sub(r"[?]{2,}", "?", processed)

        return processed.strip()

    def postprocess(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Add REPORTED_IN edges from ISSUE to PRODUCT nodes."""
        issue_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") == "ISSUE"
        ]
        product_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") in ("PRODUCT", "ORG")
        ]

        for issue in issue_nodes:
            for product in product_nodes:
                if not graph.has_edge(issue, product) and not graph.has_edge(product, issue):
                    # Check for any path <= 2 hops
                    neighbors = set(graph.successors(issue)) | set(graph.predecessors(issue))
                    prod_neighbors = set(graph.successors(product)) | set(graph.predecessors(product))
                    if neighbors & prod_neighbors:
                        graph.add_edge(
                            issue, product, predicate="REPORTED_IN", confidence=0.7, weight=0.5
                        )

        return graph

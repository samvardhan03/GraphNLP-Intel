"""Email/transaction schema: SENDER, RECIPIENT, SUBJECT, AMOUNT, MERCHANT."""

from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx

from graphnlp.adapters.base import DomainAdapter, register_adapter

# Regex to strip quoted reply threads
_QUOTED_RE = re.compile(r"^>+\s*.*$", re.MULTILINE)
_REPLY_HEADER_RE = re.compile(
    r"^-+\s*(Original Message|Forwarded message)\s*-+.*$",
    re.MULTILINE | re.IGNORECASE,
)
# HTML tag stripping
_HTML_TAG_RE = re.compile(r"<[^>]+>")
# Email header patterns to strip
_HEADER_RE = re.compile(
    r"^(From|To|Cc|Bcc|Subject|Date|Sent|Received|Reply-To):.*$",
    re.MULTILINE | re.IGNORECASE,
)


@register_adapter
class EmailAdapter(DomainAdapter):
    """Domain adapter for email and transaction analysis.

    Entity types: SENDER, RECIPIENT, MERCHANT, AMOUNT, DATE, CATEGORY.

    Preprocessing:
    - Strip email headers, HTML tags, quoted reply threads
    - Extract email metadata

    Postprocessing:
    - Add PAID_TO edges from SENDER to MERCHANT nodes
    - Aggregate spend by merchant
    """

    domain = "email"
    entity_types = ["SENDER", "RECIPIENT", "MERCHANT", "AMOUNT", "DATE", "CATEGORY"]

    def preprocess(self, text: str) -> str:
        """Strip email headers, HTML tags, and quoted reply threads."""
        processed = text

        # Strip HTML tags
        processed = _HTML_TAG_RE.sub(" ", processed)

        # Strip email headers
        processed = _HEADER_RE.sub("", processed)

        # Strip quoted replies
        processed = _QUOTED_RE.sub("", processed)

        # Strip reply/forward headers
        processed = _REPLY_HEADER_RE.sub("", processed)

        # Collapse whitespace
        processed = re.sub(r"\n{3,}", "\n\n", processed)
        processed = re.sub(r"[ \t]+", " ", processed)

        return processed.strip()

    def postprocess(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Add PAID_TO edges and aggregate spend data."""
        graph = self._add_payment_edges(graph)
        return graph

    @staticmethod
    def _add_payment_edges(graph: nx.DiGraph) -> nx.DiGraph:
        """Add PAID_TO edges from SENDER nodes to MERCHANT nodes."""
        sender_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") in ("SENDER", "PERSON")
        ]
        merchant_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") in ("MERCHANT", "ORG")
        ]

        for sender in sender_nodes:
            for merchant in merchant_nodes:
                if graph.has_edge(sender, merchant):
                    continue
                # Check if they're connected through an AMOUNT node
                sender_neighbors = set(graph.successors(sender))
                merchant_neighbors = set(graph.predecessors(merchant))
                common = sender_neighbors & merchant_neighbors
                amount_intermediaries = [
                    n for n in common
                    if graph.nodes[n].get("type") in ("AMOUNT", "MONEY")
                ]
                if amount_intermediaries:
                    graph.add_edge(
                        sender,
                        merchant,
                        predicate="PAID_TO",
                        confidence=0.8,
                        weight=0.7,
                    )

        return graph

    @staticmethod
    def monthly_spend_summary(graph: nx.DiGraph) -> dict[str, float]:
        """Aggregate spend by merchant from the graph.

        Looks for PAID_TO edges and sums AMOUNT nodes connected
        to the same MERCHANT.

        Returns
        -------
        dict[str, float]
            Mapping of merchant name → total spend amount.
        """
        spend: dict[str, float] = defaultdict(float)

        merchant_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") in ("MERCHANT", "ORG")
        ]

        for merchant in merchant_nodes:
            # Look for AMOUNT nodes connected to this merchant
            predecessors = graph.predecessors(merchant)
            for pred in predecessors:
                node_data = graph.nodes.get(pred, {})
                if node_data.get("type") in ("AMOUNT", "MONEY"):
                    # Try to parse the amount from the label
                    amount_text = node_data.get("label", pred)
                    try:
                        amount = float(
                            re.sub(r"[^\d.]", "", amount_text)
                        )
                        spend[merchant] += amount
                    except (ValueError, TypeError):
                        continue

        return dict(spend)

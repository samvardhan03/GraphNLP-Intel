"""Finance schema: ORG, TICKER, METRIC, AMOUNT, DATE, PERCENT."""

from __future__ import annotations

import re
from collections import Counter

import networkx as nx

from graphnlp.adapters.base import DomainAdapter, register_adapter

# Common financial abbreviations → expansions
_ABBR_MAP = {
    "Q1": "first quarter",
    "Q2": "second quarter",
    "Q3": "third quarter",
    "Q4": "fourth quarter",
    "YoY": "year over year",
    "QoQ": "quarter over quarter",
    "MoM": "month over month",
    "EPS": "earnings per share",
    "P/E": "price to earnings",
    "EBITDA": "earnings before interest taxes depreciation and amortization",
    "ROI": "return on investment",
    "ROE": "return on equity",
    "IPO": "initial public offering",
    "M&A": "mergers and acquisitions",
    "CEO": "chief executive officer",
    "CFO": "chief financial officer",
    "FY": "fiscal year",
    "YTD": "year to date",
    "AUM": "assets under management",
    "NAV": "net asset value",
}

# Regex for currency normalization: $1,234.56 → 1234.56 USD
_CURRENCY_RE = re.compile(r"\$\s*([\d,]+\.?\d*)")

# Regex for percentage values
_PERCENT_RE = re.compile(r"(\d+\.?\d*)\s*%")


@register_adapter
class FinanceAdapter(DomainAdapter):
    """Domain adapter for financial text analysis.

    Entity types: ORG, TICKER, METRIC, AMOUNT, DATE, PERCENT.

    Preprocessing:
    - Normalize currency strings ($1,234.56 → 1234.56 USD)
    - Expand common financial abbreviations

    Postprocessing:
    - Add COMPETITOR_OF edges between ORGs in the same community
    - Add CORRELATES_WITH edges between co-occurring TICKERs
    """

    domain = "finance"
    entity_types = ["ORG", "TICKER", "METRIC", "AMOUNT", "DATE", "PERCENT"]

    def preprocess(self, text: str) -> str:
        """Normalize currency strings and expand financial abbreviations."""
        processed = text

        # Normalize currency: $1,234.56 → 1234.56 USD
        def _normalize_currency(match: re.Match) -> str:
            amount = match.group(1).replace(",", "")
            return f"{amount} USD"

        processed = _CURRENCY_RE.sub(_normalize_currency, processed)

        # Expand abbreviations (word-boundary aware)
        for abbr, expansion in _ABBR_MAP.items():
            processed = re.sub(
                rf"\b{re.escape(abbr)}\b",
                expansion,
                processed,
            )

        return processed.strip()

    def postprocess(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Add finance-specific edges to the knowledge graph."""
        graph = self._add_competitor_edges(graph)
        graph = self._add_correlation_edges(graph)
        return graph

    @staticmethod
    def _add_competitor_edges(graph: nx.DiGraph) -> nx.DiGraph:
        """Add COMPETITOR_OF edges between ORG nodes that share a community.

        ORGs that appear in the same community (via Louvain) are assumed
        to be competitors or closely related.
        """
        org_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") == "ORG"
        ]

        # Find ORG pairs that share at least one neighbor
        for i, org1 in enumerate(org_nodes):
            neighbors1 = set(graph.predecessors(org1)) | set(graph.successors(org1))
            for org2 in org_nodes[i + 1 :]:
                if graph.has_edge(org1, org2) or graph.has_edge(org2, org1):
                    continue
                neighbors2 = set(graph.predecessors(org2)) | set(graph.successors(org2))
                shared = neighbors1 & neighbors2
                if shared:
                    graph.add_edge(
                        org1,
                        org2,
                        predicate="COMPETITOR_OF",
                        confidence=0.7,
                        weight=len(shared) / max(len(neighbors1), len(neighbors2), 1),
                    )

        return graph

    @staticmethod
    def _add_correlation_edges(graph: nx.DiGraph) -> nx.DiGraph:
        """Add CORRELATES_WITH edges between TICKER nodes that co-occur frequently."""
        ticker_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") in ("TICKER", "ORG")
        ]

        # Co-occurrence: tickers connected through shared edges/contexts
        for i, t1 in enumerate(ticker_nodes):
            for t2 in ticker_nodes[i + 1 :]:
                if graph.has_edge(t1, t2) or graph.has_edge(t2, t1):
                    continue
                # Check if they share a common predicate/edge pattern
                preds1 = {
                    d.get("predicate", "")
                    for _, _, d in graph.edges(t1, data=True)
                }
                preds2 = {
                    d.get("predicate", "")
                    for _, _, d in graph.edges(t2, data=True)
                }
                shared_preds = preds1 & preds2 - {""}
                if shared_preds:
                    graph.add_edge(
                        t1,
                        t2,
                        predicate="CORRELATES_WITH",
                        confidence=0.6,
                        weight=0.5,
                    )

        return graph

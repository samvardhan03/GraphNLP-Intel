"""Graph diff: detect topology / sentiment shifts between snapshots."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class GraphChange:
    """Summary of changes between two graph snapshots."""

    added_nodes: list[str] = field(default_factory=list)
    removed_nodes: list[str] = field(default_factory=list)
    added_edges: list[tuple[str, str]] = field(default_factory=list)
    removed_edges: list[tuple[str, str]] = field(default_factory=list)
    sentiment_shifts: dict[str, tuple[float, float]] = field(default_factory=dict)
    new_communities: list[int] = field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "low"


class GraphDiff:
    """Detect significant changes between two graph snapshots.

    This is used to determine when to trigger webhook dispatch for
    subscribers interested in graph topology or sentiment changes.
    """

    def diff(
        self,
        old: nx.DiGraph,
        new: nx.DiGraph,
        old_sentiments: dict[str, float] | None = None,
        new_sentiments: dict[str, float] | None = None,
        old_communities: dict[str, int] | None = None,
        new_communities: dict[str, int] | None = None,
    ) -> GraphChange:
        """Compute the diff between two graph snapshots.

        Parameters
        ----------
        old : nx.DiGraph
            Previous graph snapshot.
        new : nx.DiGraph
            Current graph snapshot.
        old_sentiments : dict[str, float] | None
            Sentiment scores from the old graph.
        new_sentiments : dict[str, float] | None
            Sentiment scores from the new graph.
        old_communities : dict[str, int] | None
            Community assignments from the old graph.
        new_communities : dict[str, int] | None
            Community assignments from the new graph.

        Returns
        -------
        GraphChange
            Structured diff with severity assessment.
        """
        change = GraphChange()

        old_nodes = set(old.nodes())
        new_nodes = set(new.nodes())

        # Node changes
        change.added_nodes = sorted(new_nodes - old_nodes)
        change.removed_nodes = sorted(old_nodes - new_nodes)

        # Edge changes
        old_edges = set(old.edges())
        new_edges = set(new.edges())
        change.added_edges = sorted(new_edges - old_edges)
        change.removed_edges = sorted(old_edges - new_edges)

        # Sentiment shifts (for nodes present in both snapshots)
        old_sentiments = old_sentiments or {}
        new_sentiments = new_sentiments or {}
        common_nodes = old_nodes & new_nodes

        for node in common_nodes:
            old_score = old_sentiments.get(node, 0.0)
            new_score = new_sentiments.get(node, 0.0)
            if abs(new_score - old_score) > 0.1:  # Meaningful shift
                change.sentiment_shifts[node] = (old_score, new_score)

        # New communities
        if old_communities is not None and new_communities is not None:
            old_cids = set(old_communities.values())
            new_cids = set(new_communities.values())
            change.new_communities = sorted(new_cids - old_cids)

        # Compute severity
        change.severity = self._compute_severity(change, old_nodes, new_nodes)

        return change

    def is_significant(self, change: GraphChange, threshold: float = 0.3) -> bool:
        """Determine if a graph change is significant enough to trigger notifications.

        Parameters
        ----------
        change : GraphChange
            Computed graph diff.
        threshold : float
            Sentiment shift threshold above which a change is significant.

        Returns
        -------
        bool
            True if the change is significant.
        """
        if change.severity == "high":
            return True

        if change.severity == "medium" and threshold <= 0.3:
            return True

        # Check individual sentiment shifts against threshold
        for _node, (old_score, new_score) in change.sentiment_shifts.items():
            if abs(new_score - old_score) >= threshold:
                return True

        return False

    @staticmethod
    def _compute_severity(
        change: GraphChange,
        old_nodes: set[str],
        new_nodes: set[str],
    ) -> Literal["low", "medium", "high"]:
        """Compute severity level for a graph change.

        Severity = "high" if:
        - Any sentiment shift > 0.5
        - New community forms
        - > 20% nodes changed (added or removed)

        Severity = "medium" if:
        - Any sentiment shift > 0.3
        - > 10% nodes changed
        - Multiple new edges

        Otherwise "low".
        """
        # Check sentiment shifts
        max_shift = 0.0
        for _node, (old_score, new_score) in change.sentiment_shifts.items():
            shift = abs(new_score - old_score)
            max_shift = max(max_shift, shift)

        # Check node churn ratio
        total_nodes = len(old_nodes | new_nodes)
        if total_nodes > 0:
            changed = len(change.added_nodes) + len(change.removed_nodes)
            churn_ratio = changed / total_nodes
        else:
            churn_ratio = 0.0

        # HIGH severity
        if max_shift > 0.5:
            return "high"
        if change.new_communities:
            return "high"
        if churn_ratio > 0.2:
            return "high"

        # MEDIUM severity
        if max_shift > 0.3:
            return "medium"
        if churn_ratio > 0.1:
            return "medium"
        if len(change.added_edges) > 5 or len(change.removed_edges) > 5:
            return "medium"

        return "low"

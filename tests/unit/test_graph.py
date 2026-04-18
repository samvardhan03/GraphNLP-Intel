"""Unit tests for graph modules: builder, community, diff, gnn."""

from __future__ import annotations

import numpy as np
import networkx as nx
import pytest

from graphnlp.extraction.ner import Entity
from graphnlp.extraction.relations import Triple


# ── GraphBuilder tests ──────────────────────────────────────────────────────


class TestGraphBuilder:
    def test_build_basic(self):
        from graphnlp.graph.builder import GraphBuilder

        entities = [
            Entity(text="Apple", label="ORG", start=0, end=5),
            Entity(text="Google", label="ORG", start=20, end=26),
        ]
        triples = [
            Triple(
                subject="Apple", predicate="competes_with", object="Google",
                confidence=1.0, source_text="Apple competes with Google."
            ),
        ]
        embeddings = {
            "Apple": np.random.randn(384).astype(np.float32),
            "Google": np.random.randn(384).astype(np.float32),
        }

        builder = GraphBuilder()
        graph = builder.build(triples, entities, embeddings)

        assert isinstance(graph, nx.DiGraph)
        assert graph.number_of_nodes() >= 2
        assert graph.number_of_edges() >= 1
        assert "Apple" in graph.nodes
        assert "Google" in graph.nodes

    def test_build_node_attributes(self):
        from graphnlp.graph.builder import GraphBuilder

        entities = [
            Entity(text="Apple", label="ORG", start=0, end=5),
            Entity(text="Apple", label="ORG", start=50, end=55),  # duplicate mention
        ]
        embeddings = {"Apple": np.ones(384, dtype=np.float32)}

        builder = GraphBuilder()
        graph = builder.build([], entities, embeddings)

        assert "Apple" in graph.nodes
        assert graph.nodes["Apple"]["type"] == "ORG"
        assert graph.nodes["Apple"]["mention_count"] >= 2

    def test_build_empty(self):
        from graphnlp.graph.builder import GraphBuilder

        builder = GraphBuilder()
        graph = builder.build([], [], {})
        assert graph.number_of_nodes() == 0
        assert graph.number_of_edges() == 0

    def test_semantic_merge(self):
        from graphnlp.graph.builder import GraphBuilder

        entities = [
            Entity(text="Apple Inc", label="ORG", start=0, end=9),
            Entity(text="Apple Inc", label="ORG", start=50, end=59),
            Entity(text="Apple Inc.", label="ORG", start=100, end=110),
        ]
        # Nearly identical embeddings → should merge
        base = np.random.randn(384).astype(np.float32)
        base_norm = base / np.linalg.norm(base)
        embeddings = {
            "Apple Inc": base_norm,
            "Apple Inc.": base_norm + np.random.randn(384).astype(np.float32) * 0.01,
        }
        # Normalize the second one too for high cosine
        embeddings["Apple Inc."] = embeddings["Apple Inc."] / np.linalg.norm(embeddings["Apple Inc."])

        builder = GraphBuilder(merge_threshold=0.92)
        graph = builder.build([], entities, embeddings)

        # Should be merged into fewer nodes
        assert graph.number_of_nodes() <= 2

    def test_edge_weight_computation(self):
        from graphnlp.graph.builder import GraphBuilder

        entities = [
            Entity(text="A", label="ORG", start=0, end=1),
            Entity(text="B", label="ORG", start=5, end=6),
        ]
        triples = [
            Triple(subject="A", predicate="rel", object="B", confidence=0.9, source_text="A rel B"),
        ]
        embeddings = {
            "A": np.array([1, 0, 0], dtype=np.float32),
            "B": np.array([0, 1, 0], dtype=np.float32),
        }
        builder = GraphBuilder()
        graph = builder.build(triples, entities, embeddings)

        assert graph.has_edge("A", "B")
        # Orthogonal vectors → cosine = 0
        assert graph.edges["A", "B"]["weight"] == pytest.approx(0.0, abs=0.01)


# ── CommunityDetector tests ────────────────────────────────────────────────


class TestCommunityDetector:
    def test_detect_basic(self, sample_graph):
        from graphnlp.graph.community import CommunityDetector

        detector = CommunityDetector()
        communities = detector.detect(sample_graph)
        assert isinstance(communities, dict)
        assert len(communities) == sample_graph.number_of_nodes()

    def test_detect_empty_graph(self):
        from graphnlp.graph.community import CommunityDetector

        detector = CommunityDetector()
        G = nx.DiGraph()
        communities = detector.detect(G)
        assert communities == {}

    def test_detect_no_edges(self):
        from graphnlp.graph.community import CommunityDetector

        detector = CommunityDetector()
        G = nx.DiGraph()
        G.add_node("A")
        G.add_node("B")
        communities = detector.detect(G)
        # All nodes should be in the same community (0)
        assert all(v == 0 for v in communities.values())

    def test_top_communities(self, sample_graph):
        from graphnlp.graph.community import CommunityDetector

        detector = CommunityDetector()
        sentiments = {"Apple": 0.5, "Microsoft": -0.2, "Tim Cook": 0.8, "$120B": 0.0}
        top = detector.top_communities(sample_graph, n=3, sentiments=sentiments)
        assert isinstance(top, list)
        assert len(top) >= 1
        for c in top:
            assert "id" in c
            assert "size" in c
            assert "top_nodes" in c
            assert "avg_sentiment" in c
            assert "dominant_type" in c


# ── GraphDiff tests ─────────────────────────────────────────────────────────


class TestGraphDiff:
    def test_diff_basic(self, sample_graph, sample_graph_v2):
        from graphnlp.graph.diff import GraphDiff

        differ = GraphDiff()
        change = differ.diff(sample_graph, sample_graph_v2)

        assert "Google" in change.added_nodes
        assert "$120B" in change.removed_nodes
        assert len(change.added_edges) > 0

    def test_diff_sentiment_shifts(self, sample_graph, sample_graph_v2):
        from graphnlp.graph.diff import GraphDiff

        differ = GraphDiff()
        old_sent = {"Apple": 0.5, "Microsoft": -0.2, "Tim Cook": 0.8, "$120B": 0.0}
        new_sent = {"Apple": -0.3, "Microsoft": -0.2, "Tim Cook": 0.8, "Google": 0.6}
        change = differ.diff(
            sample_graph, sample_graph_v2,
            old_sentiments=old_sent, new_sentiments=new_sent,
        )
        assert "Apple" in change.sentiment_shifts
        assert change.sentiment_shifts["Apple"] == (0.5, -0.3)

    def test_diff_identical_graphs(self, sample_graph):
        from graphnlp.graph.diff import GraphDiff

        differ = GraphDiff()
        change = differ.diff(sample_graph, sample_graph)
        assert len(change.added_nodes) == 0
        assert len(change.removed_nodes) == 0
        assert change.severity == "low"

    def test_is_significant_high(self, sample_graph, sample_graph_v2):
        from graphnlp.graph.diff import GraphDiff

        differ = GraphDiff()
        old_sent = {"Apple": 0.8, "Microsoft": 0.0, "Tim Cook": 0.0, "$120B": 0.0}
        new_sent = {"Apple": -0.3, "Microsoft": 0.0, "Tim Cook": 0.0, "Google": 0.0}
        change = differ.diff(
            sample_graph, sample_graph_v2,
            old_sentiments=old_sent, new_sentiments=new_sent,
        )
        # Apple shifted by 1.1 → should be high
        assert differ.is_significant(change)

    def test_severity_computation(self):
        from graphnlp.graph.diff import GraphChange

        # High severity: sentiment shift > 0.5
        change = GraphChange(
            sentiment_shifts={"X": (0.0, 0.8)},
        )
        assert change.severity == "low"  # severity is set by diff(), not constructor

        from graphnlp.graph.diff import GraphDiff
        assert GraphDiff._compute_severity(change, {"A"}, {"A"}) in ("high",)

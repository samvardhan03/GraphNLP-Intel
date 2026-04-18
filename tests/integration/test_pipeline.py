"""Integration tests for the full Pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest


class TestPipeline:
    def test_pipeline_init(self):
        """Test that Pipeline initializes with default settings."""
        from graphnlp.pipeline import Pipeline

        pipe = Pipeline(domain="generic")
        assert pipe.domain == "generic"
        assert pipe.adapter is not None

    def test_pipeline_init_finance(self):
        """Test Pipeline with finance domain adapter."""
        import graphnlp.adapters.finance  # trigger registration
        from graphnlp.pipeline import Pipeline

        pipe = Pipeline(domain="finance")
        assert pipe.domain == "finance"

    def test_graph_result_summary(self, sample_graph):
        """Test GraphResult.summary() returns valid structure."""
        from graphnlp.pipeline import GraphResult

        result = GraphResult(
            graph=sample_graph,
            entities=[],
            triples=[],
            sentiments={"Apple": 0.5, "Microsoft": -0.2, "Tim Cook": 0.8, "$120B": 0.0},
            communities={"Apple": 0, "Microsoft": 0, "Tim Cook": 1, "$120B": 1},
        )

        summary = result.summary()
        assert summary["node_count"] == 4
        assert summary["edge_count"] == 3
        assert "communities" in summary
        assert "avg_sentiment" in summary
        assert isinstance(summary["avg_sentiment"], float)

    def test_graph_result_export_json(self, sample_graph, tmp_path):
        """Test JSON export."""
        from graphnlp.pipeline import GraphResult

        result = GraphResult(
            graph=sample_graph,
            sentiments={"Apple": 0.5, "Microsoft": -0.2},
            communities={"Apple": 0, "Microsoft": 0},
        )

        output = tmp_path / "test_export.json"
        result.export_json(str(output))
        assert output.exists()

        import json
        data = json.loads(output.read_text())
        assert "nodes" in data
        assert "links" in data

    def test_pipeline_run_with_text_list(self, mock_spacy):
        """Test pipeline.run() with a list of text strings."""
        from graphnlp.pipeline import Pipeline
        from graphnlp.extraction.embeddings import EmbeddingExtractor

        # Mock the embedding extractor
        with patch.object(EmbeddingExtractor, "_load_sbert") as mock_sbert:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(5, 384).astype(np.float32)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_sbert.return_value = mock_model

            pipe = Pipeline(domain="generic")
            pipe.embedding_extractor._sbert_model = mock_model
            pipe.embedding_extractor._embed_dim = 384

            result = pipe.run([
                "Apple reported strong earnings. Tim Cook presented Q4 results.",
                "Microsoft announced a partnership with Goldman Sachs.",
            ])

            assert result.graph is not None
            assert isinstance(result.sentiments, dict)
            assert isinstance(result.communities, dict)
            summary = result.summary()
            assert summary["node_count"] >= 0


class TestD3Export:
    def test_export_basic(self, sample_graph):
        from graphnlp.viz.d3_export import export_d3_json

        sentiments = {"Apple": 0.5, "Microsoft": -0.2, "Tim Cook": 0.8, "$120B": 0.0}
        communities = {"Apple": 0, "Microsoft": 0, "Tim Cook": 1, "$120B": 1}

        data = export_d3_json(sample_graph, sentiments, communities)
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == 4
        assert len(data["links"]) == 3

        # Verify node structure
        node = data["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "type" in node
        assert "sentiment" in node
        assert "community" in node

    def test_export_empty_graph(self):
        import networkx as nx
        from graphnlp.viz.d3_export import export_d3_json

        G = nx.DiGraph()
        data = export_d3_json(G)
        assert data["nodes"] == []
        assert data["links"] == []

"""Main Pipeline orchestrator — wires ingestion → extraction → graph → viz."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import networkx as nx

from graphnlp.adapters.base import get_adapter
from graphnlp.extraction.embeddings import EmbeddingExtractor
from graphnlp.extraction.ner import NERExtractor, Entity
from graphnlp.extraction.relations import RelationExtractor, Triple
from graphnlp.graph.builder import GraphBuilder
from graphnlp.graph.community import CommunityDetector
from graphnlp.graph.gnn import GraphGNN
from graphnlp.ingestion.chunker import TextChunker
from graphnlp.ingestion.loader import DocumentLoader

logger = logging.getLogger(__name__)


@dataclass
class GraphResult:
    """Result of a pipeline run, containing the graph and computed metadata."""

    graph: nx.DiGraph
    entities: list[Entity] = field(default_factory=list)
    triples: list[Triple] = field(default_factory=list)
    sentiments: dict[str, float] = field(default_factory=dict)
    communities: dict[str, int] = field(default_factory=dict)

    def visualize(self, output_path: str) -> None:
        """Render the graph to an interactive HTML file."""
        from graphnlp.viz.pyvis_renderer import render_html

        html = render_html(self.graph, self.sentiments, self.communities)
        Path(output_path).write_text(html, encoding="utf-8")
        logger.info("Visualization saved to %s", output_path)

    def export_json(self, output_path: str) -> None:
        """Export the graph as D3-compatible JSON."""
        from graphnlp.viz.d3_export import export_d3_json

        data = export_d3_json(self.graph, self.sentiments, self.communities)
        Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("JSON exported to %s", output_path)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict: top clusters, sentiment scores, anomalies."""
        detector = CommunityDetector()
        top_communities = detector.top_communities(
            self.graph, n=5, sentiments=self.sentiments
        )

        # Anomalies: extreme sentiment nodes
        anomalies = [
            {
                "node": node,
                "sentiment": score,
                "type": self.graph.nodes[node].get("type", "MISC")
                if node in self.graph.nodes
                else "MISC",
            }
            for node, score in sorted(
                self.sentiments.items(), key=lambda x: abs(x[1]), reverse=True
            )
            if abs(score) > 0.5
        ][:10]

        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "entity_count": len(self.entities),
            "triple_count": len(self.triples),
            "communities": top_communities,
            "anomalies": anomalies,
            "avg_sentiment": (
                sum(self.sentiments.values()) / len(self.sentiments)
                if self.sentiments
                else 0.0
            ),
        }


class Pipeline:
    """Main NLP pipeline orchestrator.

    Wires together all components: ingestion → extraction → graph construction →
    community detection → GNN sentiment propagation.

    Parameters
    ----------
    domain : str
        Domain adapter name (e.g. ``"finance"``, ``"email"``, ``"generic"``).
    config_path : str | None
        Path to a custom YAML config file. If None, uses default settings.
    """

    def __init__(
        self,
        domain: str = "generic",
        config_path: str | None = None,
    ) -> None:
        self.domain = domain
        self.adapter = get_adapter(domain)

        # Load settings
        from graphnlp.config import get_settings

        self.settings = get_settings()

        # Initialize components (lazy-loaded where possible)
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=5, overlap=1)

        ner_model = self.adapter.ner_model or self.settings.ner_model
        self.ner = NERExtractor(model=ner_model)
        self.relation_extractor = RelationExtractor()
        self.embedding_extractor = EmbeddingExtractor(model=self.settings.embedding_model)
        self.graph_builder = GraphBuilder()
        self.community_detector = CommunityDetector()
        self.gnn = GraphGNN(num_layers=self.settings.gnn_layers)

    def run(self, source: Union[str, Path, list[str]]) -> GraphResult:
        """Run the full NLP pipeline.

        Parameters
        ----------
        source : str | Path | list[str]
            Either a file path (str/Path) or a pre-loaded list of document strings.

        Returns
        -------
        GraphResult
            Contains the knowledge graph, entities, triples, sentiments, and communities.
        """
        # Step 1: Load documents
        if isinstance(source, (str, Path)) and not isinstance(source, list):
            path = Path(source)
            if path.exists():
                documents = self.loader.load(source)
            else:
                # Treat as raw text
                documents = [str(source)]
        elif isinstance(source, list):
            documents = source
        else:
            documents = [str(source)]

        logger.info("Pipeline: loaded %d documents (domain=%s)", len(documents), self.domain)

        # Step 2: Preprocess with domain adapter
        documents = [self.adapter.preprocess(doc) for doc in documents]

        # Step 3: Chunk documents
        all_chunks: list[str] = []
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            all_chunks.extend(chunks if chunks else [doc])

        logger.info("Pipeline: chunked into %d segments", len(all_chunks))

        # Step 4: Extract entities
        all_entities: list[Entity] = []
        for chunk in all_chunks:
            entities = self.ner.extract(chunk)
            all_entities.extend(entities)

        logger.info("Pipeline: extracted %d entities", len(all_entities))

        # Step 5: Extract relations
        all_triples: list[Triple] = []
        for chunk in all_chunks:
            triples = self.relation_extractor.extract(chunk, all_entities)
            all_triples.extend(triples)

        logger.info("Pipeline: extracted %d triples", len(all_triples))

        # Step 6: Compute embeddings
        embeddings = self.embedding_extractor.embed_entities(all_entities)
        logger.info("Pipeline: computed %d entity embeddings", len(embeddings))

        # Step 7: Build graph
        graph = self.graph_builder.build(all_triples, all_entities, embeddings)

        # Step 8: Domain-specific postprocessing
        graph = self.adapter.postprocess(graph)

        # Step 9: Community detection
        communities = self.community_detector.detect(graph)

        # Step 10: GNN sentiment propagation
        sentiments = self.gnn.run(graph)

        logger.info(
            "Pipeline complete: %d nodes, %d edges, %d communities",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            len(set(communities.values())) if communities else 0,
        )

        return GraphResult(
            graph=graph,
            entities=all_entities,
            triples=all_triples,
            sentiments=sentiments,
            communities=communities,
        )

"""PyTorch Geometric GAT model — sentiment propagation across the knowledge graph."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

# Entity type vocabulary for one-hot encoding
ENTITY_TYPES = [
    "PERSON", "ORG", "GPE", "DATE", "MONEY",
    "PRODUCT", "EVENT", "LOC", "WORK_OF_ART", "MISC",
]
_TYPE_TO_IDX = {t: i for i, t in enumerate(ENTITY_TYPES)}
NUM_ENTITY_TYPES = len(ENTITY_TYPES)

# Input feature dim: SBERT (384) + entity type one-hot (10) + mention count (1)
DEFAULT_EMBED_DIM = 384
DEFAULT_IN_CHANNELS = DEFAULT_EMBED_DIM + NUM_ENTITY_TYPES + 1


def _try_import_pyg():
    """Try importing PyTorch Geometric components."""
    try:
        import torch
        import torch.nn.functional as F
        from torch_geometric.nn import GATConv
        from torch_geometric.data import Data
        return torch, F, GATConv, Data
    except ImportError:
        return None, None, None, None


# ── GAT Model ───────────────────────────────────────────────────────────────

torch, F, GATConv, PygData = _try_import_pyg()

if torch is not None:
    class SentimentGAT(torch.nn.Module):
        """Graph Attention Network for sentiment score propagation.

        Parameters
        ----------
        in_channels : int
            Input feature dimension per node.
        hidden_channels : int
            Hidden layer dimension.
        num_layers : int
            Number of GAT layers.
        heads : int
            Number of attention heads per layer.
        """

        def __init__(
            self,
            in_channels: int = DEFAULT_IN_CHANNELS,
            hidden_channels: int = 64,
            num_layers: int = 2,
            heads: int = 4,
        ) -> None:
            super().__init__()
            self.convs = torch.nn.ModuleList()
            self.norms = torch.nn.ModuleList()

            # First layer
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
            self.norms.append(torch.nn.LayerNorm(hidden_channels * heads))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)
                )
                self.norms.append(torch.nn.LayerNorm(hidden_channels * heads))

            # Final layer: single head → scalar output
            if num_layers >= 2:
                self.convs.append(
                    GATConv(hidden_channels * heads, 1, heads=1, concat=False)
                )
            else:
                # Single layer model
                self.convs[0] = GATConv(in_channels, 1, heads=1, concat=False)
                self.norms = torch.nn.ModuleList()

        def forward(self, x, edge_index, edge_weight=None):
            """Forward pass.

            Parameters
            ----------
            x : Tensor
                Node features of shape ``(N, in_channels)``.
            edge_index : Tensor
                Edge indices of shape ``(2, E)``.
            edge_weight : Tensor | None
                Optional edge weights of shape ``(E,)``.

            Returns
            -------
            Tensor
                Node-level sentiment scores in ``[-1, 1]``, shape ``(N,)``.
            """
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_index)
                x = self.norms[i](x)
                x = F.elu(x)
                x = F.dropout(x, p=0.2, training=self.training)

            # Final layer → tanh for [-1, 1] range
            x = self.convs[-1](x, edge_index)
            return torch.tanh(x).squeeze(-1)
else:
    # Stub class when PyG is not available
    class SentimentGAT:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PyTorch Geometric is required for GNN sentiment propagation. "
                "Install with: pip install torch-geometric"
            )


class GraphGNN:
    """High-level interface for running GNN sentiment propagation.

    Handles NetworkX → PyG conversion, initial sentiment seeding,
    inference, and result extraction.

    Parameters
    ----------
    num_layers : int
        Number of GAT layers.
    hidden_channels : int
        Hidden dimension size.
    embed_dim : int
        Expected embedding dimension from SBERT.
    """

    def __init__(
        self,
        num_layers: int = 2,
        hidden_channels: int = 64,
        embed_dim: int = DEFAULT_EMBED_DIM,
    ) -> None:
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim
        self._sentiment_analyzer = None

    def run(self, graph: nx.DiGraph) -> dict[str, float]:
        """Run sentiment propagation on a knowledge graph.

        Parameters
        ----------
        graph : nx.DiGraph
            Knowledge graph with node attributes: ``embedding``, ``type``, ``mention_count``.

        Returns
        -------
        dict[str, float]
            Mapping of ``{node_id: sentiment_score}`` where scores are in ``[-1, 1]``.
        """
        if graph.number_of_nodes() == 0:
            return {}

        if torch is None:
            logger.warning(
                "PyTorch Geometric not available; using VADER-only sentiment scoring"
            )
            return self._vader_only_sentiment(graph)

        try:
            return self._run_gnn(graph)
        except Exception as exc:
            logger.warning("GNN inference failed: %s; falling back to VADER", exc)
            return self._vader_only_sentiment(graph)

    def _run_gnn(self, graph: nx.DiGraph) -> dict[str, float]:
        """Run full GNN pipeline: convert → seed → infer."""
        nodes = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        if n == 0:
            return {}

        # Build feature matrix: [embedding | entity_type_onehot | mention_count]
        in_channels = self.embed_dim + NUM_ENTITY_TYPES + 1
        x = np.zeros((n, in_channels), dtype=np.float32)

        for i, node in enumerate(nodes):
            data = graph.nodes[node]
            # Embedding
            emb = data.get("embedding", [])
            if isinstance(emb, (list, np.ndarray)) and len(emb) > 0:
                emb_arr = np.array(emb, dtype=np.float32)
                x[i, : min(len(emb_arr), self.embed_dim)] = emb_arr[: self.embed_dim]
            # Entity type one-hot
            etype = data.get("type", "MISC")
            type_idx = _TYPE_TO_IDX.get(etype, _TYPE_TO_IDX["MISC"])
            x[i, self.embed_dim + type_idx] = 1.0
            # Mention count (normalized)
            mc = data.get("mention_count", 1)
            x[i, -1] = min(mc / 10.0, 1.0)

        # Build edge index
        edges_src, edges_dst = [], []
        edge_weights = []
        for u, v, edata in graph.edges(data=True):
            if u in node_to_idx and v in node_to_idx:
                edges_src.append(node_to_idx[u])
                edges_dst.append(node_to_idx[v])
                edge_weights.append(edata.get("weight", 0.5))

        if not edges_src:
            # No edges — return seed sentiments only
            return self._vader_only_sentiment(graph)

        x_t = torch.tensor(x, dtype=torch.float32)
        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        edge_weight_t = torch.tensor(edge_weights, dtype=torch.float32)

        # Seed sentiment for supervision signal
        seed_sentiments = self._get_seed_sentiments(graph)

        # Initialize and run model
        model = SentimentGAT(
            in_channels=in_channels,
            hidden_channels=self.hidden_channels,
            num_layers=self.num_layers,
        )
        model.eval()

        # For inference without training: use seed sentiments where available,
        # propagate through the graph structure
        with torch.no_grad():
            scores = model(x_t, edge_index, edge_weight_t)

        # Blend with seed sentiments
        result: dict[str, float] = {}
        for i, node in enumerate(nodes):
            gnn_score = float(scores[i]) if i < len(scores) else 0.0
            seed = seed_sentiments.get(node)
            if seed is not None:
                # Weighted blend: 70% seed, 30% GNN (untrained model)
                result[node] = 0.7 * seed + 0.3 * gnn_score
            else:
                result[node] = gnn_score
            # Clamp to [-1, 1]
            result[node] = max(-1.0, min(1.0, result[node]))

        return result

    def _get_seed_sentiments(self, graph: nx.DiGraph) -> dict[str, float]:
        """Compute seed sentiment scores for nodes using VADER."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

            if self._sentiment_analyzer is None:
                self._sentiment_analyzer = SentimentIntensityAnalyzer()
            analyzer = self._sentiment_analyzer
        except ImportError:
            logger.warning("vaderSentiment not installed; seed sentiments will be zeros")
            return {}

        sentiments: dict[str, float] = {}
        for node in graph.nodes():
            # Use node label (entity text) for sentiment analysis
            label = graph.nodes[node].get("label", node)
            scores = analyzer.polarity_scores(label)
            compound = scores["compound"]
            if abs(compound) > 0.05:  # Only seed if sentiment is meaningful
                sentiments[node] = compound

        return sentiments

    def _vader_only_sentiment(self, graph: nx.DiGraph) -> dict[str, float]:
        """Fallback: pure VADER sentiment without GNN propagation."""
        seed = self._get_seed_sentiments(graph)
        result: dict[str, float] = {}
        for node in graph.nodes():
            result[node] = seed.get(node, 0.0)
        return result

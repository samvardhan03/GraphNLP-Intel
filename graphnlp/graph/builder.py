"""NetworkX graph construction from SVO triples + embeddings."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import networkx as nx
import numpy as np

from graphnlp.extraction.ner import Entity
from graphnlp.extraction.relations import Triple

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class GraphBuilder:
    """Build a knowledge graph from extracted triples, entities, and embeddings.

    Nodes represent unique entities; edges represent relations.
    Semantically similar entities (cosine > threshold) are merged.

    Parameters
    ----------
    merge_threshold : float
        Cosine similarity threshold above which entities are merged.
    """

    def __init__(self, merge_threshold: float = 0.92) -> None:
        self.merge_threshold = merge_threshold

    def build(
        self,
        triples: list[Triple],
        entities: list[Entity],
        embeddings: dict[str, np.ndarray],
    ) -> nx.DiGraph:
        """Construct a directed knowledge graph.

        Parameters
        ----------
        triples : list[Triple]
            SVO relation triples.
        entities : list[Entity]
            Named entities extracted from text.
        embeddings : dict[str, np.ndarray]
            Mapping of entity text → embedding vector.

        Returns
        -------
        nx.DiGraph
            Knowledge graph with node/edge attributes.
        """
        G = nx.DiGraph()

        # Count entity mentions for sizing
        mention_counts: Counter = Counter()
        entity_types: dict[str, str] = {}
        for ent in entities:
            mention_counts[ent.text] += 1
            entity_types[ent.text] = ent.label

        # Merge semantically similar entities
        merge_map = self._compute_merge_map(entities, embeddings)

        # Add nodes (using canonical forms after merging)
        added_nodes: set[str] = set()
        for ent in entities:
            canonical = merge_map.get(ent.text, ent.text)
            if canonical not in added_nodes:
                emb = embeddings.get(canonical, embeddings.get(ent.text))
                G.add_node(
                    canonical,
                    label=canonical,
                    type=entity_types.get(canonical, entity_types.get(ent.text, "MISC")),
                    embedding=emb.tolist() if emb is not None else [],
                    mention_count=mention_counts.get(canonical, 0)
                    + mention_counts.get(ent.text, 0),
                )
                added_nodes.add(canonical)

        # Add edges from triples
        for triple in triples:
            subj = merge_map.get(triple.subject, triple.subject)
            obj = merge_map.get(triple.object, triple.object)

            # Ensure both nodes exist
            if subj not in G:
                G.add_node(subj, label=subj, type="MISC", embedding=[], mention_count=1)
            if obj not in G:
                G.add_node(obj, label=obj, type="MISC", embedding=[], mention_count=1)

            if subj == obj:
                continue

            # Edge weight = cosine similarity between subject and object embeddings
            weight = self._compute_edge_weight(subj, obj, embeddings, merge_map)

            if G.has_edge(subj, obj):
                # Accumulate: keep highest confidence, average weight
                existing = G.edges[subj, obj]
                existing["confidence"] = max(
                    existing.get("confidence", 0), triple.confidence
                )
                existing["weight"] = (existing.get("weight", 0) + weight) / 2
                preds = existing.get("predicates", [])
                if triple.predicate not in preds:
                    preds.append(triple.predicate)
                existing["predicates"] = preds
                existing["predicate"] = preds[0]
            else:
                G.add_edge(
                    subj,
                    obj,
                    predicate=triple.predicate,
                    predicates=[triple.predicate],
                    confidence=triple.confidence,
                    weight=weight,
                )

        logger.info(
            "Built graph with %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
        )
        return G

    def _compute_merge_map(
        self,
        entities: list[Entity],
        embeddings: dict[str, np.ndarray],
    ) -> dict[str, str]:
        """Map entity texts to their canonical (most-mentioned) surface form.

        Entities with cosine similarity > ``merge_threshold`` are merged.
        """
        unique_texts = list({e.text for e in entities})
        if len(unique_texts) <= 1:
            return {}

        # Count mentions per entity text
        mention_counts: Counter = Counter(e.text for e in entities)

        # Build pairwise similarity for entities that have embeddings
        merge_map: dict[str, str] = {}
        merged_groups: list[set[str]] = []

        texts_with_emb = [t for t in unique_texts if t in embeddings]

        for i, t1 in enumerate(texts_with_emb):
            for t2 in texts_with_emb[i + 1 :]:
                sim = _cosine_similarity(embeddings[t1], embeddings[t2])
                if sim >= self.merge_threshold:
                    # Find or create a merge group
                    found_group = None
                    for group in merged_groups:
                        if t1 in group or t2 in group:
                            group.add(t1)
                            group.add(t2)
                            found_group = group
                            break
                    if found_group is None:
                        merged_groups.append({t1, t2})

        # For each group, pick the most-mentioned surface form as canonical
        for group in merged_groups:
            canonical = max(group, key=lambda t: mention_counts.get(t, 0))
            for text in group:
                if text != canonical:
                    merge_map[text] = canonical

        return merge_map

    @staticmethod
    def _compute_edge_weight(
        subj: str,
        obj: str,
        embeddings: dict[str, np.ndarray],
        merge_map: dict[str, str],
    ) -> float:
        """Compute edge weight as cosine similarity between endpoint embeddings."""
        subj_key = merge_map.get(subj, subj)
        obj_key = merge_map.get(obj, obj)

        subj_emb = embeddings.get(subj_key, embeddings.get(subj))
        obj_emb = embeddings.get(obj_key, embeddings.get(obj))

        if subj_emb is not None and obj_emb is not None:
            return _cosine_similarity(subj_emb, obj_emb)
        return 0.5  # default weight when embeddings are missing

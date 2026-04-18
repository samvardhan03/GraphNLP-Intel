"""Neo4j backend — persist and query knowledge graphs with tenant isolation."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import networkx as nx

from graphnlp.storage.base import GraphStore

logger = logging.getLogger(__name__)


class Neo4jGraphStore(GraphStore):
    """Neo4j-backed graph store with full tenant isolation.

    Every Cypher query is scoped by ``tenant_id`` to prevent cross-tenant data leakage.
    Uses ``MERGE`` on ``(entity_text, tenant_id)`` to avoid duplicate nodes.

    Parameters
    ----------
    uri : str
        Neo4j Bolt URI.
    user : str
        Neo4j username.
    password : str
        Neo4j password.
    """

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._driver = None

    async def _get_driver(self):
        """Lazy-init the async Neo4j driver with connection pooling."""
        if self._driver is None:
            try:
                from neo4j import AsyncGraphDatabase

                self._driver = AsyncGraphDatabase.driver(
                    self._uri,
                    auth=(self._user, self._password),
                    max_connection_pool_size=50,
                )
                logger.info("Connected to Neo4j at %s", self._uri)
            except Exception as exc:
                logger.error("Failed to connect to Neo4j: %s", exc)
                raise
        return self._driver

    async def close(self) -> None:
        """Close the Neo4j driver."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    async def save(self, graph_id: str, graph: nx.DiGraph, tenant_id: str) -> None:
        """Upsert all nodes and edges for this graph_id.

        Uses MERGE to avoid duplicates. All data is scoped by tenant_id.
        """
        driver = await self._get_driver()

        async with driver.session() as session:
            # Save nodes
            for node_id, data in graph.nodes(data=True):
                await session.run(
                    """
                    MERGE (n:Entity {name: $name, tenant_id: $tenant_id})
                    SET n.graph_id = $graph_id,
                        n.label = $label,
                        n.type = $type,
                        n.embedding = $embedding,
                        n.mention_count = $mention_count,
                        n.updated_at = datetime()
                    """,
                    name=str(node_id),
                    tenant_id=tenant_id,
                    graph_id=graph_id,
                    label=data.get("label", str(node_id)),
                    type=data.get("type", "MISC"),
                    embedding=_serialize_embedding(data.get("embedding")),
                    mention_count=data.get("mention_count", 1),
                )

            # Save edges
            for src, dst, data in graph.edges(data=True):
                await session.run(
                    """
                    MATCH (a:Entity {name: $src, tenant_id: $tenant_id})
                    MATCH (b:Entity {name: $dst, tenant_id: $tenant_id})
                    MERGE (a)-[r:RELATES_TO {graph_id: $graph_id}]->(b)
                    SET r.predicate = $predicate,
                        r.confidence = $confidence,
                        r.weight = $weight,
                        r.updated_at = datetime()
                    """,
                    src=str(src),
                    dst=str(dst),
                    tenant_id=tenant_id,
                    graph_id=graph_id,
                    predicate=data.get("predicate", ""),
                    confidence=data.get("confidence", 1.0),
                    weight=data.get("weight", 0.5),
                )

            # Save graph metadata
            await session.run(
                """
                MERGE (g:GraphMeta {graph_id: $graph_id, tenant_id: $tenant_id})
                SET g.node_count = $node_count,
                    g.edge_count = $edge_count,
                    g.updated_at = datetime()
                """,
                graph_id=graph_id,
                tenant_id=tenant_id,
                node_count=graph.number_of_nodes(),
                edge_count=graph.number_of_edges(),
            )

        logger.info(
            "Saved graph '%s' for tenant '%s' (%d nodes, %d edges)",
            graph_id,
            tenant_id,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    async def load(self, graph_id: str, tenant_id: str) -> nx.DiGraph:
        """Load graph from Neo4j, return as networkx DiGraph."""
        driver = await self._get_driver()
        G = nx.DiGraph()

        async with driver.session() as session:
            # Load nodes
            result = await session.run(
                """
                MATCH (n:Entity {graph_id: $graph_id, tenant_id: $tenant_id})
                RETURN n.name AS name, n.label AS label, n.type AS type,
                       n.embedding AS embedding, n.mention_count AS mention_count
                """,
                graph_id=graph_id,
                tenant_id=tenant_id,
            )
            async for record in result:
                name = record["name"]
                G.add_node(
                    name,
                    label=record["label"] or name,
                    type=record["type"] or "MISC",
                    embedding=_deserialize_embedding(record["embedding"]),
                    mention_count=record["mention_count"] or 1,
                )

            # Load edges
            result = await session.run(
                """
                MATCH (a:Entity {tenant_id: $tenant_id})-[r:RELATES_TO {graph_id: $graph_id}]->(b:Entity {tenant_id: $tenant_id})
                RETURN a.name AS src, b.name AS dst,
                       r.predicate AS predicate, r.confidence AS confidence, r.weight AS weight
                """,
                graph_id=graph_id,
                tenant_id=tenant_id,
            )
            async for record in result:
                G.add_edge(
                    record["src"],
                    record["dst"],
                    predicate=record["predicate"] or "",
                    confidence=record["confidence"] or 1.0,
                    weight=record["weight"] or 0.5,
                )

        logger.info(
            "Loaded graph '%s' for tenant '%s' (%d nodes, %d edges)",
            graph_id,
            tenant_id,
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        return G

    async def list_graphs(self, tenant_id: str) -> list[dict]:
        """List all graph_ids for a tenant with metadata."""
        driver = await self._get_driver()

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (g:GraphMeta {tenant_id: $tenant_id})
                RETURN g.graph_id AS graph_id, g.node_count AS node_count,
                       g.edge_count AS edge_count, g.updated_at AS updated_at
                ORDER BY g.updated_at DESC
                """,
                tenant_id=tenant_id,
            )
            graphs: list[dict] = []
            async for record in result:
                graphs.append(
                    {
                        "graph_id": record["graph_id"],
                        "node_count": record["node_count"],
                        "edge_count": record["edge_count"],
                        "updated_at": str(record["updated_at"]) if record["updated_at"] else None,
                    }
                )
            return graphs

    async def delete(self, graph_id: str, tenant_id: str) -> None:
        """Delete a graph and all its nodes/edges."""
        driver = await self._get_driver()

        async with driver.session() as session:
            # Delete edges
            await session.run(
                """
                MATCH (a:Entity {tenant_id: $tenant_id})-[r:RELATES_TO {graph_id: $graph_id}]->(b)
                DELETE r
                """,
                graph_id=graph_id,
                tenant_id=tenant_id,
            )
            # Delete nodes (only if they belong exclusively to this graph)
            await session.run(
                """
                MATCH (n:Entity {graph_id: $graph_id, tenant_id: $tenant_id})
                WHERE NOT EXISTS {
                    MATCH (n)-[r:RELATES_TO]->()
                    WHERE r.graph_id <> $graph_id
                }
                DELETE n
                """,
                graph_id=graph_id,
                tenant_id=tenant_id,
            )
            # Delete metadata
            await session.run(
                """
                MATCH (g:GraphMeta {graph_id: $graph_id, tenant_id: $tenant_id})
                DELETE g
                """,
                graph_id=graph_id,
                tenant_id=tenant_id,
            )

        logger.info("Deleted graph '%s' for tenant '%s'", graph_id, tenant_id)


def _serialize_embedding(embedding: Any) -> list[float]:
    """Serialize an embedding to a list of floats for Neo4j storage."""
    if embedding is None:
        return []
    if isinstance(embedding, list):
        return [float(v) for v in embedding]
    try:
        import numpy as np

        if isinstance(embedding, np.ndarray):
            return embedding.tolist()
    except ImportError:
        pass
    return []


def _deserialize_embedding(data: Any) -> list[float]:
    """Deserialize an embedding from Neo4j storage."""
    if data is None:
        return []
    if isinstance(data, list):
        return [float(v) for v in data]
    return []

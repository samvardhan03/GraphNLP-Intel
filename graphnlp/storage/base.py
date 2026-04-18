"""Abstract GraphStore interface for graph persistence backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import networkx as nx


class GraphStore(ABC):
    """Abstract base class for knowledge graph persistence."""

    @abstractmethod
    async def save(self, graph_id: str, graph: nx.DiGraph, tenant_id: str) -> None:
        """Upsert all nodes and edges for this graph_id."""

    @abstractmethod
    async def load(self, graph_id: str, tenant_id: str) -> nx.DiGraph:
        """Load graph from store, return as networkx DiGraph."""

    @abstractmethod
    async def list_graphs(self, tenant_id: str) -> list[dict]:
        """List all graph_ids for a tenant with metadata."""

    @abstractmethod
    async def delete(self, graph_id: str, tenant_id: str) -> None:
        """Delete a graph and all its nodes/edges."""

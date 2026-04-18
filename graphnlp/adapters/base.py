"""Abstract DomainAdapter — entity schema + pre/post hooks for domain-specific processing."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import networkx as nx


class DomainAdapter(ABC):
    """Abstract base class for domain-specific NLP adapters.

    Domain adapters customize the NLP pipeline for specific use cases
    (e.g. finance, email, IT incidents) by defining entity types,
    text preprocessing, and graph post-processing hooks.

    Subclasses must define:
    - ``domain``: domain name string
    - ``entity_types``: list of expected entity labels
    - ``ner_model``: spaCy/HF model override (or None for default)
    """

    domain: str = "generic"
    entity_types: list[str] = []
    ner_model: str | None = None

    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Domain-specific text cleaning before NER.

        Parameters
        ----------
        text : str
            Raw input text.

        Returns
        -------
        str
            Cleaned text ready for NER extraction.
        """

    @abstractmethod
    def postprocess(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Domain-specific graph enrichment after building.

        Parameters
        ----------
        graph : nx.DiGraph
            Knowledge graph constructed from extracted triples.

        Returns
        -------
        nx.DiGraph
            Enriched graph with domain-specific edges/attributes.
        """

    def entity_schema(self) -> dict[str, Any]:
        """Return a JSON schema describing entity types for this domain.

        Returns
        -------
        dict
            JSON schema with entity type definitions.
        """
        return {
            "domain": self.domain,
            "entity_types": [
                {"label": t, "description": f"{self.domain} entity type: {t}"}
                for t in self.entity_types
            ],
        }


class GenericAdapter(DomainAdapter):
    """Pass-through adapter for generic text processing."""

    domain = "generic"
    entity_types = ["PERSON", "ORG", "GPE", "DATE", "MONEY", "PRODUCT", "EVENT"]

    def preprocess(self, text: str) -> str:
        """No-op preprocessing for generic domain."""
        return text.strip()

    def postprocess(self, graph: nx.DiGraph) -> nx.DiGraph:
        """No-op post-processing for generic domain."""
        return graph


# Registry of available domain adapters
_ADAPTER_REGISTRY: dict[str, type[DomainAdapter]] = {
    "generic": GenericAdapter,
}


def register_adapter(cls: type[DomainAdapter]) -> type[DomainAdapter]:
    """Register a domain adapter class in the global registry."""
    _ADAPTER_REGISTRY[cls.domain] = cls
    return cls


def get_adapter(domain: str) -> DomainAdapter:
    """Get an instantiated adapter for the specified domain.

    Falls back to ``GenericAdapter`` if the domain is not registered.
    """
    cls = _ADAPTER_REGISTRY.get(domain, GenericAdapter)
    return cls()

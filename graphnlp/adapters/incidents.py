"""IT incident schema: SERVICE, ERROR, IMPACT, SEVERITY."""

from __future__ import annotations

import re

import networkx as nx

from graphnlp.adapters.base import DomainAdapter, register_adapter

# Common severity/priority patterns
_SEVERITY_RE = re.compile(
    r"\b(P[0-4]|SEV[- ]?[0-4]|critical|major|minor|low|high|medium)\b",
    re.IGNORECASE,
)

# Error code patterns
_ERROR_CODE_RE = re.compile(r"\b(ERR[-_]?\d{3,5}|HTTP\s*[45]\d{2}|OOM|SIGKILL|SIGSEGV)\b")

# Timestamp normalization
_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")


@register_adapter
class IncidentAdapter(DomainAdapter):
    """Domain adapter for IT incident / ops log analysis.

    Entity types: SERVICE, ERROR, IMPACT, SEVERITY, PERSON, DATE.
    """

    domain = "incidents"
    entity_types = ["SERVICE", "ERROR", "IMPACT", "SEVERITY", "PERSON", "DATE"]

    def preprocess(self, text: str) -> str:
        """Clean incident logs: normalize severity tags, extract error codes."""
        processed = text

        # Normalize severity tags to a standard format
        def _normalize_sev(match: re.Match) -> str:
            raw = match.group(1).upper().replace(" ", "").replace("-", "")
            sev_map = {
                "P0": "[SEV:CRITICAL]", "P1": "[SEV:HIGH]",
                "P2": "[SEV:MEDIUM]", "P3": "[SEV:LOW]", "P4": "[SEV:LOW]",
                "SEV0": "[SEV:CRITICAL]", "SEV1": "[SEV:HIGH]",
                "SEV2": "[SEV:MEDIUM]", "SEV3": "[SEV:LOW]", "SEV4": "[SEV:LOW]",
                "CRITICAL": "[SEV:CRITICAL]", "MAJOR": "[SEV:HIGH]",
                "HIGH": "[SEV:HIGH]", "MEDIUM": "[SEV:MEDIUM]",
                "MINOR": "[SEV:LOW]", "LOW": "[SEV:LOW]",
            }
            return sev_map.get(raw, match.group(0))

        processed = _SEVERITY_RE.sub(_normalize_sev, processed)

        # Strip ANSI escape codes
        processed = re.sub(r"\x1b\[[0-9;]*m", "", processed)

        # Collapse repeated log lines
        lines = processed.split("\n")
        deduped: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and (not deduped or stripped != deduped[-1]):
                deduped.append(stripped)
        processed = "\n".join(deduped)

        return processed.strip()

    def postprocess(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Add AFFECTS edges from ERROR to SERVICE nodes, ESCALATED_TO for people."""
        error_nodes = [
            n for n, d in graph.nodes(data=True) if d.get("type") == "ERROR"
        ]
        service_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") in ("SERVICE", "ORG", "PRODUCT")
        ]

        for error in error_nodes:
            for service in service_nodes:
                if not graph.has_edge(error, service):
                    neighbors = set(graph.successors(error)) | set(graph.predecessors(error))
                    svc_neighbors = set(graph.successors(service)) | set(graph.predecessors(service))
                    if neighbors & svc_neighbors:
                        graph.add_edge(
                            error, service, predicate="AFFECTS", confidence=0.75, weight=0.6
                        )

        return graph

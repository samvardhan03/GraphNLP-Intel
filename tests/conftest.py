"""Shared pytest fixtures for graphnlp-intel tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import numpy as np
import networkx as nx
import pytest


FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── Sample data fixtures ────────────────────────────────────────────────────


@pytest.fixture
def sample_finance_csv() -> Path:
    return FIXTURES_DIR / "sample_finance.csv"


@pytest.fixture
def sample_emails_json() -> Path:
    return FIXTURES_DIR / "sample_emails.json"


@pytest.fixture
def sample_email_raw() -> str:
    data = json.loads((FIXTURES_DIR / "sample_emails.json").read_text())
    return data[0]["raw"]


@pytest.fixture
def sample_texts() -> list[str]:
    return [
        "Apple Inc reported revenue of $120 billion in Q4 2024. CEO Tim Cook announced new AI initiatives.",
        "Goldman Sachs acquired a 5% stake in Microsoft Corporation for $2.3 billion.",
        "The Federal Reserve raised interest rates by 0.25% citing persistent inflation concerns.",
        "Amazon Web Services signed a deal worth $500 million with JPMorgan Chase for cloud infrastructure.",
        "Elon Musk sold 10 million shares of Tesla stock valued at approximately $2.1 billion.",
    ]


# ── Mock NLP models ─────────────────────────────────────────────────────────


@dataclass
class MockEntity:
    text: str
    label_: str
    start_char: int
    end_char: int


class MockDoc:
    """Mock spaCy Doc object."""

    def __init__(self, text: str, ents: list[MockEntity] | None = None):
        self.text = text
        self.ents = ents or []
        self._sents = [self]

    @property
    def sents(self):
        # Simple sentence splitting for tests
        sentences = self.text.split(". ")
        result = []
        for s in sentences:
            mock = MagicMock()
            mock.text = s.strip()
            result.append(mock)
        return result if result else [MagicMock(text=self.text)]


class MockNlp:
    """Mock spaCy NLP pipeline."""

    def __init__(self):
        self.pipe_names = ["ner"]

    def __call__(self, text: str) -> MockDoc:
        ents = []
        # Simple pattern matching for test entities
        entity_patterns = {
            "Apple": ("ORG", 0),
            "Goldman Sachs": ("ORG", 0),
            "Microsoft": ("ORG", 0),
            "Amazon": ("ORG", 0),
            "Tesla": ("ORG", 0),
            "JPMorgan Chase": ("ORG", 0),
            "Tim Cook": ("PERSON", 0),
            "Elon Musk": ("PERSON", 0),
            "$120 billion": ("MONEY", 0),
            "$2.3 billion": ("MONEY", 0),
            "$500 million": ("MONEY", 0),
        }
        for pattern, (label, _) in entity_patterns.items():
            idx = text.find(pattern)
            if idx >= 0:
                ents.append(
                    MockEntity(
                        text=pattern,
                        label_=label,
                        start_char=idx,
                        end_char=idx + len(pattern),
                    )
                )
        return MockDoc(text, ents)


@pytest.fixture
def mock_spacy():
    """Patch spaCy load to return a mock NLP pipeline.

    Works even when spaCy is not installed by injecting a mock module.
    """
    import sys

    mock_nlp = MockNlp()

    spacy_installed = "spacy" in sys.modules or _can_import("spacy")

    if spacy_installed:
        with patch("spacy.load", return_value=mock_nlp) as m:
            yield m
    else:
        # Create a fake spacy module so patches and imports work
        fake_spacy = MagicMock()
        fake_spacy.load = MagicMock(return_value=mock_nlp)
        fake_spacy.blank = MagicMock(return_value=mock_nlp)
        sys.modules["spacy"] = fake_spacy
        try:
            yield fake_spacy.load
        finally:
            sys.modules.pop("spacy", None)


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


@pytest.fixture
def mock_embeddings():
    """Return mock embeddings for test entities."""
    np.random.seed(42)
    entities = [
        "Apple", "Goldman Sachs", "Microsoft", "Amazon",
        "Tesla", "Tim Cook", "Elon Musk", "JPMorgan Chase",
    ]
    return {e: np.random.randn(384).astype(np.float32) for e in entities}


# ── Sample graph fixture ────────────────────────────────────────────────────


@pytest.fixture
def sample_graph() -> nx.DiGraph:
    """A small test graph with entities and relations."""
    G = nx.DiGraph()
    G.add_node("Apple", label="Apple", type="ORG", embedding=[0.1] * 384, mention_count=5)
    G.add_node("Microsoft", label="Microsoft", type="ORG", embedding=[0.2] * 384, mention_count=3)
    G.add_node("Tim Cook", label="Tim Cook", type="PERSON", embedding=[0.3] * 384, mention_count=2)
    G.add_node("$120B", label="$120B", type="MONEY", embedding=[0.4] * 384, mention_count=1)

    G.add_edge("Apple", "Microsoft", predicate="competes_with", confidence=0.8, weight=0.7)
    G.add_edge("Tim Cook", "Apple", predicate="leads", confidence=1.0, weight=0.9)
    G.add_edge("Apple", "$120B", predicate="reported", confidence=1.0, weight=0.6)

    return G


@pytest.fixture
def sample_graph_v2() -> nx.DiGraph:
    """A modified version of sample_graph for diff testing."""
    G = nx.DiGraph()
    G.add_node("Apple", label="Apple", type="ORG", embedding=[0.1] * 384, mention_count=7)
    G.add_node("Microsoft", label="Microsoft", type="ORG", embedding=[0.2] * 384, mention_count=3)
    G.add_node("Tim Cook", label="Tim Cook", type="PERSON", embedding=[0.3] * 384, mention_count=2)
    G.add_node("Google", label="Google", type="ORG", embedding=[0.5] * 384, mention_count=4)

    G.add_edge("Apple", "Microsoft", predicate="competes_with", confidence=0.8, weight=0.7)
    G.add_edge("Tim Cook", "Apple", predicate="leads", confidence=1.0, weight=0.9)
    G.add_edge("Apple", "Google", predicate="competes_with", confidence=0.75, weight=0.65)

    return G

"""Unit tests for extraction modules: NER, relations, embeddings."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ── NERExtractor tests ──────────────────────────────────────────────────────


class TestNERExtractor:
    def test_extract_with_mock_spacy(self, mock_spacy):
        from graphnlp.extraction.ner import NERExtractor

        extractor = NERExtractor(model="en_core_web_sm")
        entities = extractor.extract(
            "Apple Inc reported revenue of $120 billion. Tim Cook made the announcement."
        )
        assert len(entities) > 0
        labels = {e.label for e in entities}
        assert "ORG" in labels or "PERSON" in labels

    def test_extract_empty_text(self, mock_spacy):
        from graphnlp.extraction.ner import NERExtractor

        extractor = NERExtractor(model="en_core_web_sm")
        entities = extractor.extract("")
        assert entities == []

    def test_extract_no_entities(self, mock_spacy):
        from graphnlp.extraction.ner import NERExtractor

        extractor = NERExtractor(model="en_core_web_sm")
        entities = extractor.extract("The quick brown fox jumps over the lazy dog.")
        # May or may not find entities depending on mock
        assert isinstance(entities, list)

    def test_entity_sorted_by_start(self, mock_spacy):
        from graphnlp.extraction.ner import NERExtractor

        extractor = NERExtractor(model="en_core_web_sm")
        entities = extractor.extract(
            "Tim Cook leads Apple which competes with Microsoft."
        )
        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                assert entities[i].start <= entities[i + 1].start

    def test_entity_dataclass(self):
        from graphnlp.extraction.ner import Entity

        e = Entity(text="Apple", label="ORG", start=0, end=5, confidence=0.95)
        assert e.text == "Apple"
        assert e.label == "ORG"
        assert e.confidence == 0.95

    def test_entity_overlap_detection(self):
        from graphnlp.extraction.ner import Entity

        e1 = Entity(text="New York", label="GPE", start=0, end=8)
        e2 = Entity(text="York City", label="GPE", start=4, end=13)
        e3 = Entity(text="London", label="GPE", start=20, end=26)

        assert e1.overlaps(e2)
        assert e2.overlaps(e1)
        assert not e1.overlaps(e3)


# ── RelationExtractor tests ────────────────────────────────────────────────


class TestRelationExtractor:
    def test_triple_dataclass(self):
        from graphnlp.extraction.relations import Triple

        t = Triple(
            subject="Apple",
            predicate="acquired",
            object="Beats",
            confidence=1.0,
            source_text="Apple acquired Beats.",
        )
        assert t.subject == "Apple"
        assert t.predicate == "acquired"
        assert t.object == "Beats"

    def test_extract_empty_text(self):
        from graphnlp.extraction.relations import RelationExtractor

        extractor = RelationExtractor()
        triples = extractor.extract("")
        assert triples == []

    def test_parse_rebel_output(self):
        from graphnlp.extraction.relations import _parse_rebel_output

        text = "<triplet> Barack Obama <subj> president of <obj> United States"
        result = _parse_rebel_output(text)
        assert len(result) == 1
        assert result[0] == ("Barack Obama", "president of", "United States")

    def test_parse_rebel_multiple(self):
        from graphnlp.extraction.relations import _parse_rebel_output

        text = (
            "<triplet> Apple <subj> develops <obj> iPhone "
            "<triplet> Tim Cook <subj> CEO of <obj> Apple"
        )
        result = _parse_rebel_output(text)
        assert len(result) == 2


# ── EmbeddingExtractor tests ───────────────────────────────────────────────


class TestEmbeddingExtractor:
    def test_embed_texts_mock(self):
        from graphnlp.extraction.embeddings import EmbeddingExtractor

        with patch.object(EmbeddingExtractor, "_load_sbert") as mock_load:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.random.randn(3, 384).astype(np.float32)
            mock_model.get_sentence_embedding_dimension.return_value = 384
            mock_load.return_value = mock_model

            extractor = EmbeddingExtractor()
            extractor._sbert_model = mock_model
            extractor._embed_dim = 384

            result = extractor.embed_texts(["hello", "world", "test"])
            assert result.shape == (3, 384)
            assert result.dtype == np.float32

    def test_embed_empty(self):
        from graphnlp.extraction.embeddings import EmbeddingExtractor

        extractor = EmbeddingExtractor()
        extractor._embed_dim = 384
        result = extractor.embed_texts([])
        assert result.shape == (0, 384)

    def test_embed_entities_mock(self):
        from graphnlp.extraction.embeddings import EmbeddingExtractor
        from graphnlp.extraction.ner import Entity

        with patch.object(EmbeddingExtractor, "_embed_sbert") as mock_embed:
            mock_embed.return_value = np.random.randn(2, 384).astype(np.float32)

            extractor = EmbeddingExtractor()
            extractor._embed_dim = 384

            entities = [
                Entity(text="Apple", label="ORG", start=0, end=5),
                Entity(text="Google", label="ORG", start=10, end=16),
            ]
            result = extractor.embed_entities(entities)
            assert isinstance(result, dict)
            assert "Apple" in result
            assert "Google" in result

    def test_text_hash_deterministic(self):
        from graphnlp.extraction.embeddings import EmbeddingExtractor

        h1 = EmbeddingExtractor._text_hash("hello world")
        h2 = EmbeddingExtractor._text_hash("hello world")
        h3 = EmbeddingExtractor._text_hash("different text")
        assert h1 == h2
        assert h1 != h3

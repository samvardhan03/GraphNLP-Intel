"""Unit tests for ingestion modules: loader, chunker, email_parser."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


# ── DocumentLoader tests ────────────────────────────────────────────────────


class TestDocumentLoader:
    def setup_method(self):
        from graphnlp.ingestion.loader import DocumentLoader
        self.loader = DocumentLoader()

    def test_load_csv(self, sample_finance_csv):
        docs = self.loader.load(sample_finance_csv)
        assert len(docs) == 20
        assert "Amazon" in docs[0] or "Payment" in docs[0]

    def test_load_csv_with_column(self, sample_finance_csv):
        docs = self.loader.load(sample_finance_csv, column="description")
        assert len(docs) == 20

    def test_load_csv_invalid_column_fallback(self, sample_finance_csv):
        # Should fall back to auto-detect
        docs = self.loader.load(sample_finance_csv, column="nonexistent")
        assert len(docs) > 0

    def test_load_json_list_of_dicts(self, tmp_path):
        data = [
            {"text": "Document one about finance."},
            {"text": "Document two about technology."},
            {"text": "Document three about healthcare."},
        ]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        docs = self.loader.load(path)
        assert len(docs) == 3
        assert "finance" in docs[0]

    def test_load_json_list_of_strings(self, tmp_path):
        data = ["Hello world", "Second document"]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(data))
        docs = self.loader.load(path)
        assert len(docs) == 2

    def test_load_txt(self, tmp_path):
        path = tmp_path / "test.txt"
        path.write_text("This is a plain text document.\nWith multiple lines.")
        docs = self.loader.load(path)
        assert len(docs) == 1
        assert "plain text" in docs[0]

    def test_load_markdown(self, tmp_path):
        path = tmp_path / "test.md"
        path.write_text("# Title\n\nSome markdown content.")
        docs = self.loader.load(path)
        assert len(docs) == 1

    def test_load_unsupported_type(self, tmp_path):
        path = tmp_path / "test.xyz"
        path.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file type"):
            self.loader.load(path)

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            self.loader.load("/nonexistent/path/file.txt")

    def test_load_empty_txt(self, tmp_path):
        path = tmp_path / "empty.txt"
        path.write_text("")
        docs = self.loader.load(path)
        assert docs == []


# ── TextChunker tests ───────────────────────────────────────────────────────


class TestTextChunker:
    def setup_method(self):
        from graphnlp.ingestion.chunker import TextChunker
        self.chunker = TextChunker(chunk_size=3, overlap=1)

    def test_chunk_basic(self):
        text = (
            "First sentence is here. Second sentence follows. "
            "Third sentence now. Fourth sentence appears. "
            "Fifth sentence ends it."
        )
        chunks = self.chunker.chunk(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk) >= 20

    def test_chunk_empty(self):
        assert self.chunker.chunk("") == []
        assert self.chunker.chunk("   ") == []

    def test_chunk_single_sentence(self):
        text = "This is a single long sentence that should be returned as one chunk."
        chunks = self.chunker.chunk(text)
        assert len(chunks) == 1

    def test_chunk_overlap(self):
        from graphnlp.ingestion.chunker import TextChunker
        chunker = TextChunker(chunk_size=2, overlap=1)
        text = (
            "First sentence is here. Second sentence follows. "
            "Third sentence now. Fourth sentence appears."
        )
        chunks = chunker.chunk(text)
        assert len(chunks) >= 2

    def test_invalid_params(self):
        from graphnlp.ingestion.chunker import TextChunker
        with pytest.raises(ValueError):
            TextChunker(chunk_size=0)
        with pytest.raises(ValueError):
            TextChunker(chunk_size=3, overlap=3)
        with pytest.raises(ValueError):
            TextChunker(chunk_size=3, overlap=-1)


# ── EmailParser tests ───────────────────────────────────────────────────────


class TestEmailParser:
    def setup_method(self):
        from graphnlp.ingestion.email_parser import EmailParser
        self.parser = EmailParser()

    def test_parse_raw_email(self, sample_email_raw):
        result = self.parser.parse(sample_email_raw)
        assert result.sender == "billing@amazon.com"
        assert "john@company.com" in result.recipients
        assert "Amazon" in result.subject
        assert result.message_id != ""

    def test_extract_amounts(self, sample_email_raw):
        result = self.parser.parse(sample_email_raw)
        assert len(result.amounts) > 0
        assert 234.56 in result.amounts

    def test_to_text(self, sample_email_raw):
        result = self.parser.parse(sample_email_raw)
        text = result.to_text()
        assert "Subject:" in text
        assert "From:" in text
        assert len(text) > 50

    def test_parse_multipart(self, sample_emails_json):
        data = json.loads(sample_emails_json.read_text())
        # The Deloitte email (index 2) is multipart
        result = self.parser.parse(data[2]["raw"])
        assert "Deloitte" in result.subject or "DLT" in result.subject
        assert 25000.0 in result.amounts
        # Should prefer plain text body over HTML
        assert "<html>" not in result.body

    def test_parse_all_fixtures(self, sample_emails_json):
        data = json.loads(sample_emails_json.read_text())
        for item in data:
            result = self.parser.parse(item["raw"])
            assert result.sender != ""
            assert result.subject != ""
            assert result.body != ""

"""Sentence / paragraph splitting with configurable overlap."""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Minimum chunk length in characters — skip very short sentences
_MIN_CHUNK_LEN = 20


class TextChunker:
    """Split long text into overlapping sentence windows.

    Parameters
    ----------
    chunk_size : int
        Number of sentences per chunk.
    overlap : int
        Number of sentences shared between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 5, overlap: int = 1) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._nlp = None  # lazy-loaded spaCy pipeline

    def _get_nlp(self):
        """Lazy-load a lightweight spaCy pipeline with sentencizer."""
        if self._nlp is None:
            try:
                import spacy

                try:
                    self._nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
                except OSError:
                    # Model not installed — use a blank pipeline with sentencizer
                    logger.warning(
                        "en_core_web_sm not found; falling back to blank sentencizer"
                    )
                    self._nlp = spacy.blank("en")
                    self._nlp.add_pipe("sentencizer")
            except ImportError:
                logger.warning("spaCy not installed; using regex sentence splitting")
                self._nlp = None
        return self._nlp

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy sentencizer or regex fallback."""
        nlp = self._get_nlp()

        if nlp is not None:
            doc = nlp(text)
            sents = [sent.text.strip() for sent in doc.sents]
        else:
            # Regex fallback: split on sentence-ending punctuation
            raw = re.split(r"(?<=[.!?])\s+", text)
            sents = [s.strip() for s in raw]

        # Filter out very short sentences
        return [s for s in sents if len(s) >= _MIN_CHUNK_LEN]

    def chunk(self, text: str) -> List[str]:
        """Split *text* into overlapping sentence windows.

        Returns
        -------
        list[str]
            Each element is a chunk formed by joining ``chunk_size`` sentences.
        """
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        # If fewer sentences than chunk_size, return everything as one chunk
        if len(sentences) <= self.chunk_size:
            return [" ".join(sentences)]

        chunks: list[str] = []
        step = self.chunk_size - self.overlap
        for i in range(0, len(sentences), step):
            window = sentences[i : i + self.chunk_size]
            chunk_text = " ".join(window)
            if len(chunk_text) >= _MIN_CHUNK_LEN:
                chunks.append(chunk_text)
            # Stop if this window reached the end
            if i + self.chunk_size >= len(sentences):
                break

        return chunks

"""spaCy + HuggingFace NER: entities, types, spans."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """A named entity extracted from text."""

    text: str
    label: str  # e.g. ORG, PERSON, MONEY, DATE
    start: int
    end: int
    confidence: float = 1.0

    def overlaps(self, other: Entity) -> bool:
        """Check if this entity span overlaps with another."""
        return self.start < other.end and other.start < self.end


class NERExtractor:
    """Named Entity Recognition using spaCy and optionally HuggingFace.

    Parameters
    ----------
    model : str
        spaCy model name (e.g. ``en_core_web_trf``).
    hf_model : str | None
        Optional HuggingFace token classification model.  If provided,
        results are merged with spaCy output (HF overrides on overlap).
    """

    def __init__(
        self,
        model: str = "en_core_web_trf",
        hf_model: Optional[str] = None,
    ) -> None:
        self._model_name = model
        self._hf_model_name = hf_model
        self._nlp = None
        self._hf_pipeline = None

    def _load_spacy(self):
        """Lazy-load spaCy pipeline."""
        if self._nlp is None:
            try:
                import spacy

                self._nlp = spacy.load(self._model_name)
                logger.info("Loaded spaCy model: %s", self._model_name)
            except OSError:
                import spacy

                logger.warning(
                    "spaCy model '%s' not found; falling back to en_core_web_sm",
                    self._model_name,
                )
                try:
                    self._nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("No spaCy model available; using blank pipeline")
                    self._nlp = spacy.blank("en")
        return self._nlp

    def _load_hf(self):
        """Lazy-load HuggingFace NER pipeline."""
        if self._hf_pipeline is None and self._hf_model_name:
            try:
                from transformers import pipeline as hf_pipeline

                self._hf_pipeline = hf_pipeline(
                    "ner",
                    model=self._hf_model_name,
                    aggregation_strategy="simple",
                )
                logger.info("Loaded HuggingFace NER model: %s", self._hf_model_name)
            except Exception as exc:
                logger.warning(
                    "Failed to load HuggingFace model '%s': %s",
                    self._hf_model_name,
                    exc,
                )
                self._hf_model_name = None  # Don't retry
        return self._hf_pipeline

    def extract(self, text: str) -> list[Entity]:
        """Extract named entities from *text*.

        Returns
        -------
        list[Entity]
            Deduplicated entities sorted by start offset.
        """
        if not text or not text.strip():
            return []

        # Primary: spaCy extraction
        entities = self._extract_spacy(text)

        # Optional: HuggingFace extraction (overrides spaCy on overlap)
        if self._hf_model_name:
            hf_entities = self._extract_hf(text)
            entities = self._merge_entities(entities, hf_entities)

        # Deduplicate overlapping spans
        entities = self._deduplicate(entities)

        # Sort by start offset
        entities.sort(key=lambda e: (e.start, -e.end))
        return entities

    def _extract_spacy(self, text: str) -> list[Entity]:
        """Extract entities using spaCy."""
        nlp = self._load_spacy()
        doc = nlp(text)
        return [
            Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0,
            )
            for ent in doc.ents
        ]

    def _extract_hf(self, text: str) -> list[Entity]:
        """Extract entities using HuggingFace pipeline."""
        pipe = self._load_hf()
        if pipe is None:
            return []

        try:
            results = pipe(text)
        except Exception as exc:
            logger.warning("HuggingFace NER failed: %s", exc)
            return []

        entities: list[Entity] = []
        for r in results:
            entities.append(
                Entity(
                    text=r.get("word", text[r["start"]:r["end"]]),
                    label=r.get("entity_group", r.get("entity", "MISC")),
                    start=r["start"],
                    end=r["end"],
                    confidence=float(r.get("score", 0.85)),
                )
            )
        return entities

    @staticmethod
    def _merge_entities(
        spacy_ents: list[Entity],
        hf_ents: list[Entity],
    ) -> list[Entity]:
        """Merge spaCy and HF entities; HF overrides spaCy on overlapping spans."""
        merged: list[Entity] = list(hf_ents)  # HF has priority

        for sp_ent in spacy_ents:
            # Keep spaCy entity only if it doesn't overlap with any HF entity
            has_overlap = any(sp_ent.overlaps(hf) for hf in hf_ents)
            if not has_overlap:
                merged.append(sp_ent)

        return merged

    @staticmethod
    def _deduplicate(entities: list[Entity]) -> list[Entity]:
        """Remove overlapping entities, keeping the one with highest confidence."""
        if not entities:
            return []

        # Sort by start, then by confidence descending
        sorted_ents = sorted(entities, key=lambda e: (e.start, -e.confidence))
        result: list[Entity] = [sorted_ents[0]]

        for ent in sorted_ents[1:]:
            last = result[-1]
            if ent.overlaps(last):
                # Keep the one with higher confidence
                if ent.confidence > last.confidence:
                    result[-1] = ent
            else:
                result.append(ent)

        return result

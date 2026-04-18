"""Relation extraction — subject-verb-object triples from text."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from graphnlp.extraction.ner import Entity

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    """A subject–predicate–object triple extracted from text."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0
    source_text: str = ""


class RelationExtractor:
    """Extract subject-verb-object triples from text.

    Uses spaCy dependency parsing by default, with an optional HuggingFace
    REBEL model (``Babelscape/rebel-large``) for improved extraction.

    Parameters
    ----------
    use_hf : bool
        If True, use the REBEL model for relation extraction.
    spacy_model : str
        spaCy model to use for dependency parsing.
    """

    def __init__(
        self,
        use_hf: bool = False,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self._use_hf = use_hf
        self._spacy_model = spacy_model
        self._nlp = None
        self._hf_pipeline = None

    def _load_spacy(self):
        """Lazy-load spaCy pipeline for dependency parsing."""
        if self._nlp is None:
            try:
                import spacy

                try:
                    self._nlp = spacy.load(self._spacy_model)
                except OSError:
                    self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                import spacy

                self._nlp = spacy.blank("en")
                self._nlp.add_pipe("sentencizer")
        return self._nlp

    def _load_hf(self):
        """Lazy-load HuggingFace REBEL model for relation extraction."""
        if self._hf_pipeline is None and self._use_hf:
            try:
                from transformers import pipeline as hf_pipeline

                self._hf_pipeline = hf_pipeline(
                    "text2text-generation",
                    model="Babelscape/rebel-large",
                    max_length=512,
                )
                logger.info("Loaded REBEL model for relation extraction")
            except Exception as exc:
                logger.warning("Failed to load REBEL model: %s", exc)
                self._use_hf = False
        return self._hf_pipeline

    def extract(
        self,
        text: str,
        entities: list[Entity] | None = None,
    ) -> list[Triple]:
        """Extract relation triples from *text*.

        Parameters
        ----------
        text : str
            Input text to extract relations from.
        entities : list[Entity] | None
            Known entities to filter triples by.  If provided, only
            triples whose subject and object match a known entity span
            are returned.

        Returns
        -------
        list[Triple]
            Extracted triples with confidence scores.
        """
        if not text or not text.strip():
            return []

        entity_texts = {e.text.lower() for e in entities} if entities else None

        # Dependency parse extraction (primary)
        triples = self._extract_dep(text, entity_texts)

        # Optional HF REBEL extraction
        if self._use_hf:
            hf_triples = self._extract_rebel(text, entity_texts)
            triples.extend(hf_triples)

        return triples

    def _extract_dep(
        self,
        text: str,
        entity_texts: set[str] | None = None,
    ) -> list[Triple]:
        """Extract SVO triples using spaCy dependency parsing."""
        nlp = self._load_spacy()
        doc = nlp(text)
        triples: list[Triple] = []

        for sent in doc.sents:
            for token in sent:
                # Look for verbs that have both subject and object
                if token.pos_ != "VERB":
                    continue

                subjects: list[str] = []
                objects: list[str] = []

                for child in token.children:
                    # Subject dependencies
                    if child.dep_ in ("nsubj", "nsubjpass"):
                        subj_span = _get_span_text(child)
                        subjects.append(subj_span)
                    # Object dependencies
                    elif child.dep_ in ("dobj", "attr", "pobj"):
                        obj_span = _get_span_text(child)
                        objects.append(obj_span)
                    # Prepositional objects
                    elif child.dep_ == "prep":
                        for grandchild in child.children:
                            if grandchild.dep_ == "pobj":
                                obj_span = _get_span_text(grandchild)
                                objects.append(obj_span)

                # Create triples from subject × object pairs
                for subj in subjects:
                    for obj in objects:
                        if subj == obj:
                            continue
                        # Filter by known entities if provided
                        if entity_texts is not None:
                            if subj.lower() not in entity_texts or obj.lower() not in entity_texts:
                                continue
                        triples.append(
                            Triple(
                                subject=subj,
                                predicate=token.lemma_,
                                object=obj,
                                confidence=1.0,
                                source_text=sent.text.strip(),
                            )
                        )

        return triples

    def _extract_rebel(
        self,
        text: str,
        entity_texts: set[str] | None = None,
    ) -> list[Triple]:
        """Extract triples using the REBEL model (Babelscape/rebel-large).

        REBEL outputs triples in the format:
        ``<triplet> subject <subj> object <obj> predicate``
        """
        pipe = self._load_hf()
        if pipe is None:
            return []

        try:
            # REBEL works best on shorter texts
            outputs = pipe(text, return_text=True, max_length=512)
        except Exception as exc:
            logger.warning("REBEL extraction failed: %s", exc)
            return []

        triples: list[Triple] = []
        for output in outputs:
            generated = output.get("generated_text", "")
            parsed = _parse_rebel_output(generated)
            for subj, pred, obj in parsed:
                if entity_texts is not None:
                    if subj.lower() not in entity_texts or obj.lower() not in entity_texts:
                        continue
                triples.append(
                    Triple(
                        subject=subj,
                        predicate=pred,
                        object=obj,
                        confidence=0.85,
                        source_text=text[:200],
                    )
                )

        return triples


def _get_span_text(token) -> str:
    """Get the full span text for a token including its compound/modifier children."""
    span_tokens = [token]
    for child in token.children:
        if child.dep_ in ("compound", "amod", "det"):
            span_tokens.append(child)
    span_tokens.sort(key=lambda t: t.i)
    return " ".join(t.text for t in span_tokens)


def _parse_rebel_output(text: str) -> list[tuple[str, str, str]]:
    """Parse REBEL model output into (subject, predicate, object) tuples.

    REBEL format: ``<triplet> subject <subj> predicate <obj> object``
    """
    triples: list[tuple[str, str, str]] = []
    # Split on <triplet> markers
    parts = re.split(r"<triplet>\s*", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Parse: subject <subj> predicate <obj> object
        subj_match = re.match(r"(.+?)\s*<subj>\s*(.+?)\s*<obj>\s*(.+)", part)
        if subj_match:
            subj = subj_match.group(1).strip()
            pred = subj_match.group(2).strip()
            obj = subj_match.group(3).strip()
            if subj and pred and obj:
                triples.append((subj, pred, obj))
    return triples

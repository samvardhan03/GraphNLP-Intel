"""SBERT dense embeddings + SONAR multilingual concept embeddings."""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Optional

import numpy as np

from graphnlp.extraction.ner import Entity

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Compute dense text embeddings using Sentence-Transformers (SBERT) with
    an optional SONAR multilingual fallback.

    Parameters
    ----------
    model : str
        Sentence-Transformers model name or path.
    use_sonar : bool
        If True, attempt to use Meta SONAR for multilingual embeddings.
        Falls back to SBERT if SONAR is not installed.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        use_sonar: bool = False,
    ) -> None:
        self._model_name = model
        self._use_sonar = use_sonar
        self._sbert_model = None
        self._sonar_model = None
        self._embed_dim: int | None = None

    def _load_sbert(self):
        """Lazy-load Sentence-Transformers model."""
        if self._sbert_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._sbert_model = SentenceTransformer(self._model_name)
                self._embed_dim = self._sbert_model.get_sentence_embedding_dimension()
                logger.info(
                    "Loaded SBERT model '%s' (dim=%d)",
                    self._model_name,
                    self._embed_dim,
                )
            except Exception as exc:
                logger.error("Failed to load SBERT model: %s", exc)
                raise
        return self._sbert_model

    def _load_sonar(self):
        """Lazy-load SONAR multilingual encoder (optional)."""
        if self._sonar_model is None and self._use_sonar:
            try:
                from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

                self._sonar_model = TextToEmbeddingModelPipeline(
                    encoder="text_sonar_basic_encoder",
                    tokenizer="text_sonar_basic_encoder",
                )
                logger.info("Loaded SONAR multilingual encoder")
            except ImportError:
                logger.warning(
                    "SONAR not installed; falling back to SBERT. "
                    "Install with: pip install sonar-space"
                )
                self._use_sonar = False
            except Exception as exc:
                logger.warning("Failed to load SONAR: %s; falling back to SBERT", exc)
                self._use_sonar = False
        return self._sonar_model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings for a batch of texts.

        Parameters
        ----------
        texts : list[str]
            Input texts to embed.

        Returns
        -------
        np.ndarray
            Shape ``(N, D)`` float32 array of embeddings.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        # Try cached results first
        cached_results: dict[int, np.ndarray] = {}
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._get_cached(text)
            if cached is not None:
                cached_results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Compute uncached embeddings
        if uncached_texts:
            if self._use_sonar:
                new_embeddings = self._embed_sonar(uncached_texts)
            else:
                new_embeddings = self._embed_sbert(uncached_texts)

            # Cache new results
            for idx, text in enumerate(uncached_texts):
                self._set_cached(text, new_embeddings[idx])
                cached_results[uncached_indices[idx]] = new_embeddings[idx]

        # Assemble in order
        result = np.stack([cached_results[i] for i in range(len(texts))])
        return result.astype(np.float32)

    def embed_entities(self, entities: list[Entity]) -> dict[str, np.ndarray]:
        """Compute embeddings for a list of entities.

        Parameters
        ----------
        entities : list[Entity]
            Entities whose ``.text`` will be embedded.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from entity text to its embedding vector.
        """
        unique_texts = list({e.text for e in entities})
        if not unique_texts:
            return {}

        embeddings = self.embed_texts(unique_texts)
        return {text: emb for text, emb in zip(unique_texts, embeddings)}

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        if self._embed_dim is None:
            self._load_sbert()
        return self._embed_dim or 384

    def _embed_sbert(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings using SBERT (batched, GPU if available)."""
        model = self._load_sbert()
        embeddings = model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _embed_sonar(self, texts: list[str]) -> np.ndarray:
        """Compute embeddings using SONAR multilingual encoder."""
        sonar = self._load_sonar()
        if sonar is None:
            return self._embed_sbert(texts)

        try:
            embeddings = sonar.predict(texts, source_lang="eng_Latn")
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            logger.warning("SONAR encoding failed: %s; falling back to SBERT", exc)
            return self._embed_sbert(texts)

    # ── Embedding cache ─────────────────────────────────────────────────────

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    @lru_cache(maxsize=1024)
    def _cached_embed(text_hash: str) -> Optional[bytes]:
        """LRU cache placeholder — values are set by _set_cached."""
        return None

    def _get_cached(self, text: str) -> Optional[np.ndarray]:
        """Try to retrieve a cached embedding."""
        key = self._text_hash(text)
        data = EmbeddingExtractor._cached_embed(key)
        if data is not None:
            return np.frombuffer(data, dtype=np.float32).copy()
        return None

    def _set_cached(self, text: str, embedding: np.ndarray) -> None:
        """Store an embedding in the LRU cache."""
        key = self._text_hash(text)
        # Force-update the cache by manipulating the underlying dict
        EmbeddingExtractor._cached_embed.cache_clear()
        # Re-insert by calling with the hash; we use a workaround since
        # lru_cache doesn't support direct writes.  Instead, we keep a
        # separate dict for actual storage.
        if not hasattr(EmbeddingExtractor, "_embed_store"):
            EmbeddingExtractor._embed_store: dict[str, bytes] = {}
        EmbeddingExtractor._embed_store[key] = embedding.astype(np.float32).tobytes()

    def _get_cached(self, text: str) -> Optional[np.ndarray]:  # noqa: F811
        """Try to retrieve a cached embedding from the internal store."""
        key = self._text_hash(text)
        store = getattr(EmbeddingExtractor, "_embed_store", {})
        data = store.get(key)
        if data is not None:
            return np.frombuffer(data, dtype=np.float32).copy()
        return None

    def _set_cached(self, text: str, embedding: np.ndarray) -> None:  # noqa: F811
        """Store an embedding in the internal cache store."""
        key = self._text_hash(text)
        if not hasattr(EmbeddingExtractor, "_embed_store"):
            EmbeddingExtractor._embed_store = {}
        # Enforce max cache size
        if len(EmbeddingExtractor._embed_store) >= 1024:
            # Evict oldest entry
            oldest = next(iter(EmbeddingExtractor._embed_store))
            del EmbeddingExtractor._embed_store[oldest]
        EmbeddingExtractor._embed_store[key] = embedding.astype(np.float32).tobytes()

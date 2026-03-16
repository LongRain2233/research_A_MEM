"""
Local embedding encoder using sentence-transformers.

Aligns with Implementation Plan §5.1:
    Embedding: sentence-transformers (all-MiniLM-L6-v2)

Note: ChromaDB uses its own embedding function internally.
This encoder is exposed for direct embedding operations outside ChromaDB
(e.g., pre-computing vectors for similarity checks, evaluation metrics).
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingEncoder:
    """Local text embedding encoder."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"EmbeddingEncoder loaded: model={model_name}, dim={self._dim}")

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dim

    def encode(self, text: Union[str, list[str]]) -> np.ndarray:
        """
        Encode text(s) into dense vector(s).

        Args:
            text: Single string or list of strings.

        Returns:
            numpy array of shape (dim,) for single text, or (N, dim) for batch.
        """
        return self._model.encode(text, normalize_embeddings=True)

    def cosine_similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity between two texts."""
        vec_a = self.encode(text_a)
        vec_b = self.encode(text_b)
        return float(np.dot(vec_a, vec_b))

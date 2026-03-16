"""
MemoryBank baseline adapter.

MemoryBank uses an Ebbinghaus forgetting curve mechanism for memory decay.
Source: A-Mem §4.1, Table 1

The Ebbinghaus curve models memory retention as:
    R(t) = e^(-t/S)
where S is the stability parameter that increases with successful recalls.

This adapter implements a simplified MemoryBank for benchmark comparison.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from phaseforget.evaluation.benchmark import BaselineAdapter

logger = logging.getLogger(__name__)


@dataclass
class MemoryBankEntry:
    """A memory entry with Ebbinghaus forgetting curve parameters."""
    id: str
    content: str
    created_at: float = field(default_factory=time.time)
    last_recalled: float = field(default_factory=time.time)
    recall_count: int = 0
    stability: float = 1.0  # S parameter, increases with recalls

    def retention_score(self, now: Optional[float] = None) -> float:
        """Compute retention R(t) = exp(-t/S) where t is time since last recall."""
        now = now or time.time()
        elapsed = (now - self.last_recalled) / 3600  # hours
        return math.exp(-elapsed / self.stability)

    def recall(self):
        """Mark this memory as recalled, strengthening stability."""
        self.last_recalled = time.time()
        self.recall_count += 1
        # Stability grows with each successful recall (spacing effect)
        self.stability *= 1.5


class MemoryBankAdapter(BaselineAdapter):
    """
    Simplified MemoryBank baseline with Ebbinghaus forgetting curve.

    Implements the core forgetting mechanism: memories with low retention
    scores are naturally forgotten over time. No active eviction is needed.
    """

    def __init__(
        self,
        retention_threshold: float = 0.3,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Args:
            retention_threshold: Minimum retention score to keep a memory.
            embedding_model: Model for semantic search.
        """
        self._threshold = retention_threshold
        self._embedding_model = embedding_model
        self._memories: dict[str, MemoryBankEntry] = {}
        self._encoder = None
        self._vectors: dict[str, np.ndarray] = {}

    def _ensure_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self._embedding_model)

    def name(self) -> str:
        return "MemoryBank"

    async def add_interaction(self, content: str) -> None:
        """Add an interaction to MemoryBank."""
        self._ensure_encoder()

        entry_id = str(len(self._memories))
        entry = MemoryBankEntry(id=entry_id, content=content)
        self._memories[entry_id] = entry

        vec = self._encoder.encode(content, normalize_embeddings=True)
        self._vectors[entry_id] = vec

        # Prune memories below retention threshold
        self._prune_forgotten()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search MemoryBank with retention-weighted similarity."""
        self._ensure_encoder()

        if not self._memories:
            return []

        query_vec = self._encoder.encode(query, normalize_embeddings=True)
        now = time.time()

        scored = []
        for mid, entry in self._memories.items():
            if mid not in self._vectors:
                continue
            cosine_sim = float(np.dot(query_vec, self._vectors[mid]))
            retention = entry.retention_score(now)
            # Combined score: semantic relevance * memory retention
            combined = cosine_sim * retention
            scored.append({
                "id": mid,
                "content": entry.content,
                "score": combined,
                "retention": retention,
            })

        scored.sort(key=lambda x: x["score"], reverse=True)

        # Mark top-k as recalled (strengthens their stability)
        for hit in scored[:top_k]:
            self._memories[hit["id"]].recall()

        return scored[:top_k]

    def _prune_forgotten(self):
        """Remove memories below retention threshold (Ebbinghaus forgetting)."""
        now = time.time()
        to_remove = [
            mid for mid, entry in self._memories.items()
            if entry.retention_score(now) < self._threshold
        ]
        for mid in to_remove:
            del self._memories[mid]
            self._vectors.pop(mid, None)

        if to_remove:
            logger.debug(f"MemoryBank pruned {len(to_remove)} forgotten memories")

    def get_memory_count(self) -> int:
        """Return current number of active memories."""
        return len(self._memories)

    def reset(self) -> None:
        """Reset all state."""
        self._memories.clear()
        self._vectors.clear()

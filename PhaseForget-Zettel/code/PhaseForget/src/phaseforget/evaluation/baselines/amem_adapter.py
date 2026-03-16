"""
A-Mem baseline adapter.

Wraps the A-Mem (Agentic Memory) system as a baseline for comparison.
A-Mem is the no-forgetting absolute baseline (Source: A-Mem original paper).

This adapter expects the A-Mem system to be importable from the sibling
code directory (A-mem-sys).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

from phaseforget.evaluation.benchmark import BaselineAdapter

logger = logging.getLogger(__name__)


class AMemAdapter(BaselineAdapter):
    """Adapter wrapping the A-Mem agentic memory system."""

    def __init__(
        self,
        amem_path: Optional[str] = None,
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize A-Mem adapter.

        Args:
            amem_path: Path to the A-mem-sys directory.
                       Defaults to sibling directory in code/.
            llm_backend: LLM backend for A-Mem (openai, ollama, etc.)
            llm_model: Model name for LLM calls.
            embedding_model: Sentence-transformer model name.
        """
        if amem_path is None:
            amem_path = str(
                Path(__file__).resolve().parents[4] / "A-mem-sys"
            )

        # Add A-Mem to path if not already importable
        if amem_path not in sys.path:
            sys.path.insert(0, amem_path)

        self._backend = llm_backend
        self._model = llm_model
        self._embedding_model = embedding_model
        self._system = None

    def _ensure_initialized(self):
        """Lazy initialization of A-Mem system."""
        if self._system is not None:
            return

        try:
            from agentic_memory.memory_system import AgenticMemorySystem

            self._system = AgenticMemorySystem(
                model_name=self._embedding_model,
                llm_backend=self._backend,
                llm_model=self._model,
            )
            logger.info("A-Mem baseline system initialized")
        except ImportError as e:
            logger.error(
                f"Failed to import A-Mem. Ensure A-mem-sys is at the expected path: {e}"
            )
            raise

    def name(self) -> str:
        return "A-Mem"

    async def add_interaction(self, content: str) -> None:
        """Add an interaction to A-Mem (synchronous internally)."""
        self._ensure_initialized()
        self._system.add_note(content=content)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search A-Mem using agentic search."""
        self._ensure_initialized()
        results = self._system.search(query=query, k=top_k)
        # Normalize output format
        hits = []
        if results and "ids" in results and results["ids"]:
            for i, doc_ids in enumerate(results["ids"]):
                for j, doc_id in enumerate(doc_ids):
                    hits.append({
                        "id": doc_id,
                        "content": results.get("documents", [[]])[i][j]
                        if results.get("documents") else "",
                        "score": 1.0 - results.get("distances", [[0]])[i][j]
                        if results.get("distances") else 0.0,
                    })
        return hits[:top_k]

    def reset(self) -> None:
        """Reset A-Mem state for a new evaluation run."""
        self._system = None

"""
MemGPT baseline adapter.

Wraps MemGPT (OS-level memory paging with archival/recall) as a baseline.
Source: A-Mem §2.1, §4.1

MemGPT uses a tiered memory architecture:
    - Core memory: always in context
    - Recall memory: conversation search
    - Archival memory: long-term vector store

This adapter provides a standardized interface for benchmark comparison.
"""

from __future__ import annotations

import logging
from typing import Optional

from phaseforget.evaluation.benchmark import BaselineAdapter

logger = logging.getLogger(__name__)


class MemGPTAdapter(BaselineAdapter):
    """Adapter for the MemGPT memory system baseline."""

    def __init__(
        self,
        agent_name: str = "phaseforget_eval",
        model: str = "gpt-4o-mini",
    ):
        """
        Initialize MemGPT adapter.

        Args:
            agent_name: Name for the MemGPT agent instance.
            model: LLM model to use within MemGPT.
        """
        self._agent_name = agent_name
        self._model = model
        self._client = None
        self._agent = None
        self._archival_store: list[dict] = []

    def _ensure_initialized(self):
        """Lazy initialization of MemGPT client."""
        if self._client is not None:
            return

        try:
            from letta import create_client

            self._client = create_client()
            # Check for existing agent or create new
            existing = [
                a for a in self._client.list_agents()
                if a.name == self._agent_name
            ]
            if existing:
                self._agent = existing[0]
            else:
                self._agent = self._client.create_agent(
                    name=self._agent_name,
                    llm_config={"model": self._model},
                )
            logger.info(f"MemGPT baseline initialized: agent={self._agent_name}")
        except ImportError:
            logger.warning(
                "MemGPT (letta) not installed. Using fallback in-memory store. "
                "Install with: pip install letta"
            )
            self._client = "fallback"

    def name(self) -> str:
        return "MemGPT"

    async def add_interaction(self, content: str) -> None:
        """Add interaction via MemGPT archival memory."""
        self._ensure_initialized()

        if self._client == "fallback":
            self._archival_store.append({"content": content, "id": str(len(self._archival_store))})
            return

        try:
            self._client.insert_archival_memory(
                agent_id=self._agent.id,
                memory=content,
            )
        except Exception as e:
            logger.error(f"MemGPT add_interaction failed: {e}")
            self._archival_store.append({"content": content, "id": str(len(self._archival_store))})

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search MemGPT archival memory."""
        self._ensure_initialized()

        if self._client == "fallback":
            # Simple substring matching fallback
            hits = []
            for item in self._archival_store:
                if any(w in item["content"].lower() for w in query.lower().split()):
                    hits.append({"id": item["id"], "content": item["content"], "score": 0.5})
            return hits[:top_k]

        try:
            results = self._client.search_archival_memory(
                agent_id=self._agent.id,
                query=query,
                count=top_k,
            )
            return [
                {"id": str(i), "content": r.text, "score": 0.5}
                for i, r in enumerate(results)
            ]
        except Exception as e:
            logger.error(f"MemGPT search failed: {e}")
            return []

    def reset(self) -> None:
        """Reset MemGPT state."""
        if self._client and self._client != "fallback" and self._agent:
            try:
                self._client.delete_agent(self._agent.id)
            except Exception:
                pass
        self._client = None
        self._agent = None
        self._archival_store.clear()

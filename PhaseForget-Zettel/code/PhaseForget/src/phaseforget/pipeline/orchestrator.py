"""
PhaseForget-Zettel System Orchestrator.

This is the top-level facade that wires together all three modules
(M_state, M_trigger, M_renorm) and the dual-track storage into a
single coherent system.

Aligns with the complete data flow in Research Design §5 and
the sprint delivery milestones in Implementation Plan §5.2.

Data Flow (per interaction round):
    [New interaction (c_n, t_n)]
        → M_state:   create note m_n, update retrieved utilities
        → M_trigger: build topology, accumulate evidence
        → M_renorm:  (async) if triggered, execute renormalization
    [Output: evolved memory M']
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from phaseforget.config.settings import Settings, get_settings
from phaseforget.llm.base import BaseLLMClient
from phaseforget.llm.providers.litellm_client import LiteLLMClient
from phaseforget.memory.models.note import MemoryNote
from phaseforget.memory.renorm.renorm_engine import RenormalizationEngine, RenormResult
from phaseforget.memory.state.state_manager import StateManager
from phaseforget.memory.trigger.trigger_engine import TriggerEngine, TriggerResult
from phaseforget.storage.cold_track.chroma_store import ChromaColdTrack
from phaseforget.storage.hot_track.sqlite_store import SQLiteHotTrack

logger = logging.getLogger(__name__)


class PhaseForgetSystem:
    """
    Top-level orchestrator for the PhaseForget-Zettel memory system.

    Usage:
        system = PhaseForgetSystem()
        await system.initialize()
        note = await system.add_interaction("User said something important")
        results = system.search("something important")
        await system.close()
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm_client: Optional[BaseLLMClient] = None,
    ):
        self._settings = settings or get_settings()
        self._llm = llm_client
        self._interaction_count = 0

        # Storage layers (initialized in .initialize())
        self._cold: Optional[ChromaColdTrack] = None
        self._hot: Optional[SQLiteHotTrack] = None

        # Processing modules (initialized in .initialize())
        self._state_mgr: Optional[StateManager] = None
        self._trigger: Optional[TriggerEngine] = None
        self._renorm: Optional[RenormalizationEngine] = None
        self._pending_renorm_tasks: set[asyncio.Task] = set()

    async def initialize(self) -> None:
        """
        Initialize all components. Must be called before any operations.
        Wires up the dependency graph: Settings → Storage → LLM → Modules.
        """
        s = self._settings

        # 1. LLM client
        if self._llm is None:
            self._llm = LiteLLMClient(
                model=s.llm_model,
                api_key=s.llm_api_key or None,
                base_url=s.llm_base_url or None,
            )

        # 2. Storage layers
        self._cold = ChromaColdTrack(
            persist_dir=s.chroma_persist_dir,
            embedding_model=s.embedding_model,
        )
        self._hot = SQLiteHotTrack(db_path=s.sqlite_db_path)
        await self._hot.initialize()

        # 3. Processing modules
        self._state_mgr = StateManager(
            cold_track=self._cold,
            hot_track=self._hot,
            llm_client=self._llm,
            settings=s,
        )
        self._trigger = TriggerEngine(
            cold_track=self._cold,
            hot_track=self._hot,
            settings=s,
        )
        self._renorm = RenormalizationEngine(
            cold_track=self._cold,
            hot_track=self._hot,
            llm_client=self._llm,
            state_manager=self._state_mgr,
            settings=s,
        )

        # 4. Restore interaction_count from persisted state (断点续传)
        hot_stats = await self._hot.get_stats()
        self._interaction_count = hot_stats.get("interaction_count", 0)

        logger.info(
            f"PhaseForgetSystem initialized: resumed from "
            f"interaction_count={self._interaction_count}, "
            f"total_notes={hot_stats.get('total_notes', 0)}"
        )

    async def close(self) -> None:
        """Shutdown and release resources."""
        await self.wait_for_pending_renorm()
        if self._hot:
            await self._hot.close()
        logger.info("PhaseForgetSystem closed")

    # ── Core API ─────────────────────────────────────────────────────────

    async def add_interaction(
        self,
        content: str,
        retrieved_ids: Optional[list[str]] = None,
        adopted_ids: Optional[list[str]] = None,
    ) -> MemoryNote:
        """
        Process a new interaction through the full pipeline.

        This is the main entry point per conversation turn:
            1. Update utilities for previously retrieved notes.
            2. Create new atomic note.
            3. Evaluate trigger and potentially fire renormalization.
            4. Periodically apply global decay.

        Args:
            content:       Raw interaction text c_n.
            retrieved_ids: IDs of notes retrieved for this turn's context.
            adopted_ids:   Subset of retrieved_ids actually used in response.

        Returns:
            The newly created MemoryNote.
        """
        # Persist count to SQLite so it survives restarts
        self._interaction_count = await self._hot.increment_interaction_count()

        # Step 1: Update utilities for previously retrieved notes
        if retrieved_ids:
            await self._state_mgr.update_utility_on_retrieval(
                note_ids=retrieved_ids,
                adopted_ids=adopted_ids or [],
            )

        # Step 2: Create new note (async metadata extraction)
        note = await self._state_mgr.create_note(content=content)

        logger.debug(
            f"[Round {self._interaction_count}] note={note.id} "
            f"retrieved={len(retrieved_ids or [])} adopted={len(adopted_ids or [])}"
        )

        # Step 3: Evaluate trigger
        trigger_result = await self._trigger.process(note)

        # Step 3b: If triggered, execute renormalization asynchronously
        if trigger_result.triggered and trigger_result.anchor_v:
            logger.info(
                f"[Round {self._interaction_count}] Renorm triggered: "
                f"anchor={trigger_result.anchor_v}"
            )
            task = asyncio.create_task(self._safe_renormalize(trigger_result.anchor_v))
            self._pending_renorm_tasks.add(task)
            task.add_done_callback(self._pending_renorm_tasks.discard)

        # Step 4: Periodic global decay
        if self._interaction_count % self._settings.decay_interval_rounds == 0:
            logger.info(f"[Round {self._interaction_count}] Applying global decay")
            await self._state_mgr.apply_global_decay()

        return note

    async def add_note(
        self,
        content: str,
        is_abstract: bool = False,
        tags: Optional[list[str]] = None,
        skip_metadata: bool = False,
    ) -> MemoryNote:
        """
        Convenience method to add a note without full pipeline processing.
        Used by renormalization closed-loop injection and direct note insertion.
        """
        return await self._state_mgr.create_note(
            content=content,
            is_abstract=is_abstract,
            tags=tags,
            skip_metadata=skip_metadata,
        )

    async def _safe_renormalize(self, anchor_v: str) -> Optional[RenormResult]:
        """Execute renormalization with error handling."""
        try:
            return await self._renorm.execute(anchor_v)
        except Exception as e:
            logger.error(f"Renormalization failed for {anchor_v}: {e}", exc_info=True)
            return None

    async def force_renormalize(self, anchor_v: str) -> Optional[RenormResult]:
        """Manually trigger renormalization on a specific anchor. For testing."""
        return await self._renorm.execute(anchor_v)

    async def wait_for_pending_renorm(self) -> int:
        """
        Wait until all currently scheduled renormalization tasks are finished.

        Returns:
            Number of tasks that were pending when this wait started.
        """
        if not self._pending_renorm_tasks:
            return 0

        pending = list(self._pending_renorm_tasks)
        await asyncio.gather(*pending, return_exceptions=True)
        return len(pending)

    async def update_retrieval_feedback(
        self,
        note_ids: list[str],
        adopted_ids: list[str],
    ) -> None:
        """
        Public API for utility feedback updates from evaluation/retrieval loops.

        Args:
            note_ids: Retrieved note IDs used as candidate evidence.
            adopted_ids: Subset of note_ids deemed actually used in response.
        """
        if not note_ids:
            return
        await self._state_mgr.update_utility_on_retrieval(
            note_ids=note_ids,
            adopted_ids=adopted_ids,
        )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Search for relevant memories using semantic similarity.

        Args:
            query:  Natural language query text.
            top_k:  Number of results (defaults to settings.retrieval_top_k).

        Returns:
            List of matching notes with scores.
        """
        k = top_k or self._settings.retrieval_top_k
        return self._cold.search(
            query_text=query,
            top_k=k,
            min_similarity=self._settings.search_min_similarity or None,
        )

    async def search_with_graph(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_abstract_ratio: float = 0.3,
    ) -> list[dict]:
        """
        Search with link-graph expansion (aligned with A-MEM find_related_memories_raw).

        1. ChromaDB Top-K semantic recall.
        2. For each hit, fetch first-degree neighbors from the Zettel link topology.
        3. Retrieve neighbor content from ChromaDB.
        4. Merge and deduplicate, preserving original Top-K order first.
        5. Limit abstract (Sigma/Delta) nodes to *max_abstract_ratio* of results
           so that synthesized summaries do not dilute concrete factual evidence
           needed for temporal and multi-hop reasoning.

        This restores the graph traversal capability that A-MEM uses at QA time.
        """
        k = top_k or self._settings.retrieval_top_k
        seed_hits = self._cold.search(
            query_text=query,
            top_k=k,
            min_similarity=self._settings.search_min_similarity or None,
        )

        if not seed_hits:
            return []

        seen_ids: set[str] = set()
        ordered_results: list[dict] = []

        for hit in seed_hits:
            hid = hit["id"]
            if hid not in seen_ids:
                seen_ids.add(hid)
                ordered_results.append(hit)

        neighbor_ids_to_fetch: list[str] = []
        for hit in seed_hits:
            try:
                neighbors = await self._hot.get_first_degree_neighbors(hit["id"])
                for nid in neighbors:
                    if nid not in seen_ids:
                        seen_ids.add(nid)
                        neighbor_ids_to_fetch.append(nid)
            except Exception as e:
                logger.debug(f"Graph expansion failed for {hit['id']}: {e}")

        if neighbor_ids_to_fetch:
            neighbor_notes = self._cold.get_by_ids(neighbor_ids_to_fetch)
            for note_data in neighbor_notes:
                content = note_data.get("content", "")
                metadata = note_data.get("metadata", {})
                ordered_results.append({
                    "id": note_data["id"],
                    "score": 0.0,
                    "content": content,
                    "metadata": metadata,
                })

        # ── Abstract node ratio cap ───────────────────────────────────────
        # Separate concrete vs abstract results, then limit abstract count
        # to avoid diluting factual evidence with synthesized summaries.
        concrete: list[dict] = []
        abstract: list[dict] = []
        for r in ordered_results:
            rid = r["id"]
            try:
                is_abs = await self._hot.is_abstract(rid)
            except Exception:
                is_abs = False
            if is_abs:
                abstract.append(r)
            else:
                concrete.append(r)

        max_abs = max(1, int(len(ordered_results) * max_abstract_ratio))
        capped_results = concrete + abstract[:max_abs]

        return capped_results

    async def get_stats(self) -> dict:
        """Get system statistics (all values read from persistent storage)."""
        hot_stats = await self._hot.get_stats()
        cold_count = self._cold.count()
        return {
            **hot_stats,
            "cold_track_count": cold_count,
        }

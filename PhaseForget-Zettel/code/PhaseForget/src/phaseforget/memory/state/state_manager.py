"""
Module M_state: Memory State Representation & Utility Tracking.

Aligns with Implementation Plan §3 Module 1 and Research Design §3 Module 1.

Responsibilities:
    1. Async metadata extraction & enhanced vectorization pipeline.
    2. Bidirectional utility momentum update with silent decay.

Data Flow:
    Input:  Raw interaction c_n, timestamp t_n, retrieval feedback r_j
    Output: Complete MemoryNote m_n, updated utility scores u_j*
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from phaseforget.config.settings import Settings
from phaseforget.llm.base import BaseLLMClient
from phaseforget.llm.prompts.templates import METADATA_EXTRACTION_PROMPT
from phaseforget.memory.models.note import MemoryNote
from phaseforget.storage.cold_track.chroma_store import ChromaColdTrack
from phaseforget.storage.hot_track.sqlite_store import SQLiteHotTrack

logger = logging.getLogger(__name__)


class StateManager:
    """
    Handles note creation with async metadata enrichment, and
    manages utility score updates for existing notes.
    """

    def __init__(
        self,
        cold_track: ChromaColdTrack,
        hot_track: SQLiteHotTrack,
        llm_client: BaseLLMClient,
        settings: Settings,
    ):
        self._cold = cold_track
        self._hot = hot_track
        self._llm = llm_client
        self._settings = settings

    async def create_note(
        self,
        content: str,
        is_abstract: bool = False,
        tags: Optional[list[str]] = None,
        skip_metadata: bool = False,
    ) -> MemoryNote:
        """
        Create a new atomic note and persist it to both tracks.

        Steps:
            1. Extract metadata (K_n, G_n, X_n) via LLM (async).
            2. Build enhanced text and write to ChromaDB (cold track).
            3. Initialize state record in SQLite (hot track) with U_init.

        Args:
            content:       Raw interaction text c_n.
            is_abstract:   Whether this is a higher-order Sigma/Delta node.
            tags:          Optional pre-set tags (e.g., ["Macro", "Evolved_From_xxx"]).
            skip_metadata: If True, skip LLM metadata extraction (for testing).

        Returns:
            The fully constructed MemoryNote.
        """
        note = MemoryNote(
            content=content,
            is_abstract=is_abstract,
            utility=self._settings.u_init,
            tags=tags or [],
        )

        # Async metadata extraction via LLM
        if not skip_metadata:
            try:
                meta = await self._llm.generate_json(
                    METADATA_EXTRACTION_PROMPT.format(content=content),
                    system_prompt="You are a metadata extraction engine. Output ONLY a valid JSON object with keys: keywords, tags, context. No explanations, no markdown.",
                )
                # Validate and coerce each field to its expected type so that
                # a malformed-but-parsed response never silently corrupts state.
                raw_kw = meta.get("keywords", [])
                note.keywords = raw_kw if isinstance(raw_kw, list) else []
                if not tags:
                    raw_tags = meta.get("tags", [])
                    note.tags = raw_tags if isinstance(raw_tags, list) else []
                raw_ctx = meta.get("context", "")
                note.context = raw_ctx if isinstance(raw_ctx, str) else ""
            except Exception as e:
                logger.warning(f"Metadata extraction failed, proceeding without: {e}")
                # Ensure note fields keep clean defaults even on unexpected failure
                note.keywords = []
                if not tags:
                    note.tags = []
                note.context = ""

        # Persist to dual-track storage
        self._cold.add_note(note)
        await self._hot.insert_state(
            note_id=note.id,
            utility=note.utility,
            is_abstract=is_abstract,
        )

        logger.info(f"Created note {note.id} (abstract={is_abstract})")
        return note

    async def _build_semantic_links(self, note: MemoryNote) -> None:
        """
        Find semantically similar existing notes and insert bidirectional links.
        Mirrors A-MEM's process_memory / strengthen action: after each note is
        written, immediately connect it to its top-K neighbours so the link graph
        is populated for graph-expansion retrieval.
        """
        link_k = getattr(self._settings, "link_top_k", 5)
        try:
            candidates = self._cold.search(
                query_text=note.build_enhanced_text(),
                top_k=link_k + 1,  # +1 because the note itself may appear
            )
            neighbor_ids = [
                c["id"] for c in candidates
                if c["id"] != note.id
            ][:link_k]

            if neighbor_ids:
                await self._hot.insert_links(note.id, neighbor_ids)
                logger.debug(
                    f"Semantic links built: {note.id} -> {neighbor_ids}"
                )
        except Exception as e:
            logger.debug(f"Semantic link building skipped for {note.id}: {e}")

    async def update_utility_on_retrieval(
        self,
        note_ids: list[str],
        adopted_ids: list[str],
    ) -> None:
        """
        Update utility scores for all retrieved notes.

        Aligns with Implementation Plan §3 Module 1:
            u_i <- u_i + eta * (r_i - u_i)
            where r_i = 1 if adopted, r_i = 0 if retrieved but not adopted.

        Args:
            note_ids:    All note IDs that were retrieved into LLM context.
            adopted_ids: Subset of note_ids that were actually used/adopted.
        """
        eta = self._settings.eta
        for nid in note_ids:
            reward = 1.0 if nid in adopted_ids else 0.0
            await self._hot.update_utility(nid, reward=reward, eta=eta)
            logger.debug(f"Utility updated: {nid}, reward={reward}")

    async def apply_global_decay(self) -> int:
        """
        Apply time-based decay to prevent silent node freezing.
        Should be called periodically (e.g., every decay_interval_rounds).

        Returns number of affected notes.
        """
        affected = await self._hot.apply_global_decay(
            decay_factor=self._settings.decay_factor
        )
        logger.info(f"Global decay applied to {affected} notes")
        return affected

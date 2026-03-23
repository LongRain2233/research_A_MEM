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
from phaseforget.llm.prompts.templates import EVOLUTION_PROMPT, METADATA_EXTRACTION_PROMPT
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
        self._evo_cnt: int = 0  # counts A-MEM-style evolutions for consolidate trigger

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

        # A-MEM alignment: run process_memory to build links and evolve neighbors.
        # Skipped for abstract notes (Sigma/Delta) to avoid recursive evolution.
        if not is_abstract and not skip_metadata:
            evolved = await self._process_memory(note)
            if evolved:
                self._evo_cnt += 1
                evo_threshold = getattr(self._settings, "evo_threshold", 100)
                if self._evo_cnt % evo_threshold == 0:
                    await self.consolidate_memories()

        logger.info(f"Created note {note.id} (abstract={is_abstract})")
        return note

    async def _process_memory(self, note: MemoryNote) -> bool:
        """
        A-MEM process_memory alignment: on every note creation, find top-K
        semantic neighbors and let the LLM decide whether to:
          - strengthen: add links from this note to suggested neighbors, update its tags.
          - update_neighbor: rewrite the context and tags of existing neighbors
            so their embeddings stay semantically current.

        Returns True if the LLM decided to evolve (should_evolve=true).
        """
        link_k = getattr(self._settings, "link_top_k", 5)
        try:
            candidates = self._cold.search(
                query_text=note.build_enhanced_text(),
                top_k=link_k + 1,
            )
            neighbors = [c for c in candidates if c["id"] != note.id][:link_k]
        except Exception as e:
            logger.debug(f"process_memory: neighbor search failed for {note.id}: {e}")
            return False

        if not neighbors:
            return False

        # Build neighbor description string (mirrors A-MEM find_related_memories)
        neighbor_lines = []
        for idx, n in enumerate(neighbors):
            meta = n.get("metadata", {})
            kw = meta.get("keywords", [])
            tags = meta.get("tags", [])
            ctx = meta.get("context", "")
            ts = meta.get("timestamp", "")
            neighbor_lines.append(
                f"memory index:{idx}\t talk start time:{ts}\t "
                f"memory content: {meta.get('content', n.get('content',''))}\t "
                f"memory context: {ctx}\t "
                f"memory keywords: {kw}\t "
                f"memory tags: {tags}"
            )
        neighbor_str = "\n".join(neighbor_lines)

        try:
            resp = await self._llm.generate_json(
                EVOLUTION_PROMPT.format(
                    context=note.context,
                    content=note.content,
                    keywords=note.keywords,
                    nearest_neighbors_memories=neighbor_str,
                    neighbor_number=len(neighbors),
                ),
                system_prompt="You are a memory evolution agent. Output ONLY a valid JSON object. No explanations, no markdown.",
            )
        except Exception as e:
            logger.debug(f"process_memory: LLM call failed for {note.id}: {e}")
            return False

        should_evolve = resp.get("should_evolve", False)
        if not should_evolve:
            return False

        actions = resp.get("actions", [])

        if "strengthen" in actions:
            suggested = resp.get("suggested_connections", [])
            new_tags = resp.get("tags_to_update", [])
            link_targets = [
                neighbors[i]["id"] for i in suggested
                if isinstance(i, int) and i < len(neighbors)
            ]
            if link_targets:
                await self._hot.insert_links(note.id, link_targets)
                logger.debug(f"process_memory strengthen: {note.id} -> {link_targets}")
            if new_tags and isinstance(new_tags, list):
                note.tags = new_tags
                self._cold.update_note(note)

        if "update_neighbor" in actions:
            new_contexts = resp.get("new_context_neighborhood", [])
            new_tags_nb = resp.get("new_tags_neighborhood", [])
            for i, neighbor in enumerate(neighbors):
                nb_meta = neighbor.get("metadata", {})
                updated_ctx = new_contexts[i] if i < len(new_contexts) else nb_meta.get("context", "")
                updated_tags = new_tags_nb[i] if i < len(new_tags_nb) else nb_meta.get("tags", [])
                if not isinstance(updated_tags, list):
                    updated_tags = nb_meta.get("tags", [])

                # Reconstruct a MemoryNote shell just to call update_note
                from datetime import datetime as _dt
                try:
                    ts_raw = nb_meta.get("timestamp", "")
                    ts = _dt.fromisoformat(ts_raw) if ts_raw else _dt.utcnow()
                except ValueError:
                    ts = _dt.utcnow()
                nb_note = MemoryNote(
                    id=neighbor["id"],
                    content=nb_meta.get("content", neighbor.get("content", "")),
                    keywords=nb_meta.get("keywords", []) if isinstance(nb_meta.get("keywords"), list) else [],
                    tags=updated_tags,
                    context=updated_ctx,
                    timestamp=ts,
                )
                self._cold.update_note(nb_note)
                logger.debug(f"process_memory update_neighbor: {neighbor['id']} context/tags refreshed")

        return True

    async def consolidate_memories(self) -> None:
        """
        A-MEM consolidate_memories alignment: after evo_threshold evolutions,
        rebuild every note's embedding from its current (possibly updated)
        context, keywords, and tags so the vector space reflects the latest
        semantic state of all memories.

        Uses ChromaDB's update() (via update_note) rather than a full reset,
        because ChromaDB persists to disk and a reset would be destructive.
        We batch-fetch all notes and re-upsert their enhanced text.
        """
        logger.info("consolidate_memories: rebuilding all note embeddings...")
        try:
            total = self._cold.count()
            if total == 0:
                return
            # Fetch all IDs, then batch-get their metadata and re-upsert
            # ChromaDB doesn't expose a list-all-IDs API directly; we use
            # get() with a large limit via the underlying collection.
            result = self._cold._collection.get(
                include=["documents", "metadatas"],
                limit=total,
            )
            if not result["ids"]:
                return
            for i, nid in enumerate(result["ids"]):
                meta = result["metadatas"][i] if result["metadatas"] else {}
                import json as _json
                from datetime import datetime as _dt
                kw = meta.get("keywords", [])
                if isinstance(kw, str):
                    try:
                        kw = _json.loads(kw)
                    except Exception:
                        kw = []
                tags = meta.get("tags", [])
                if isinstance(tags, str):
                    try:
                        tags = _json.loads(tags)
                    except Exception:
                        tags = []
                ts_raw = meta.get("timestamp", "")
                try:
                    ts = _dt.fromisoformat(ts_raw) if ts_raw else _dt.utcnow()
                except ValueError:
                    ts = _dt.utcnow()
                nb_note = MemoryNote(
                    id=nid,
                    content=meta.get("content", ""),
                    keywords=kw,
                    tags=tags,
                    context=meta.get("context", ""),
                    timestamp=ts,
                )
                self._cold.update_note(nb_note)
            logger.info(f"consolidate_memories: rebuilt embeddings for {total} notes")
        except Exception as e:
            logger.warning(f"consolidate_memories failed: {e}")

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

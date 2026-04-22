"""
Module M_trigger: Topology Construction & Evidence Pool Accumulation.

Aligns with Implementation Plan §3 Module 2 and Research Design §3 Module 2.

Responsibilities:
    1. Recall neighbors with absolute similarity lower bound (theta_sim).
    2. Anti-ghost-read: synchronize with SQLite valid state.
    3. Establish Zettel explicit links (L_n).
    4. Accumulate evidence pool and check trigger threshold.
    5. Multi-scale evidence propagation to parent nodes.

Data Flow:
    Input:  New note m_n, global memory M
    Output: Boolean trigger b_trigger, evidence set I_v^new
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from phaseforget.config.settings import Settings
from phaseforget.memory.models.note import MemoryNote
from phaseforget.storage.cold_track.chroma_store import ChromaColdTrack
from phaseforget.storage.hot_track.sqlite_store import SQLiteHotTrack

logger = logging.getLogger(__name__)


@dataclass
class TriggerResult:
    """Result of the trigger evaluation for a new note."""
    triggered: bool
    anchor_v: Optional[str] = None
    evidence_ids: list[str] = None

    def __post_init__(self):
        if self.evidence_ids is None:
            self.evidence_ids = []


class TriggerEngine:
    """
    Evaluates whether a new note triggers renormalization by:
        1. Finding valid semantic neighbors.
        2. Building topology links.
        3. Accumulating evidence and checking threshold.
    """

    def __init__(
        self,
        cold_track: ChromaColdTrack,
        hot_track: SQLiteHotTrack,
        settings: Settings,
    ):
        self._cold = cold_track
        self._hot = hot_track
        self._settings = settings

    async def process(self, note: MemoryNote) -> TriggerResult:
        """
        Execute the complete trigger flow for a new note.

        Aligns with Implementation Plan §3 Module 2 pseudocode.

        Returns:
            TriggerResult indicating whether renormalization should fire.
        """
        # Step 1: Recall neighbors with absolute similarity lower bound
        neighbors_raw = self._cold.search(
            query_text=note.build_enhanced_text(),
            top_k=self._settings.retrieval_top_k,
            min_similarity=self._settings.theta_sim,
        )

        # Exclude self from results
        neighbors_raw = [n for n in neighbors_raw if n["id"] != note.id]

        if not neighbors_raw:
            logger.debug(f"Note {note.id}: no valid neighbors above theta_sim")
            return TriggerResult(triggered=False)

        # Step 2: Anti-ghost-read - filter against SQLite valid state
        raw_ids = [n["id"] for n in neighbors_raw]
        valid_ids = await self._hot.filter_valid_ids(raw_ids)
        valid_neighbors = [n for n in neighbors_raw if n["id"] in valid_ids]

        if not valid_neighbors:
            logger.debug(f"Note {note.id}: all neighbors filtered by ghost-read check")
            return TriggerResult(triggered=False)

        # Step 3: Establish Zettel explicit links (L_n)
        neighbor_ids = [n["id"] for n in valid_neighbors]
        cooldown_states = await self._hot.get_cooldown_states(neighbor_ids)

        # Step 4: Determine anchor node v (highest similarity, not in cooldown)
        anchor_v = None
        for neighbor in valid_neighbors:
            nid = neighbor["id"]
            if not cooldown_states.get(nid, False):
                anchor_v = nid
                break

        if anchor_v is None:
            logger.debug(f"Note {note.id}: all candidate anchors in cooldown")
            return TriggerResult(triggered=False)

        # Step 3/5/6 share one short transaction to avoid repeated commits
        async with self._hot.transaction():
            await self._hot.insert_links(
                source_id=note.id,
                target_ids=neighbor_ids,
                commit=False,
            )
            logger.debug(f"Note {note.id}: linked to {len(neighbor_ids)} neighbors")

            # Step 5: Accumulate evidence pool for anchor_v and parent nodes
            parent_ids = await self._hot.get_parent_nodes(anchor_v)
            pool_sizes = await self._hot.append_evidence_to_anchors(
                anchor_ids=[anchor_v, *parent_ids],
                evidence_id=note.id,
                commit=False,
            )

            # Step 6: Check trigger threshold |I_v^new| > theta_sum
            pool_size = pool_sizes.get(anchor_v, 0)
            if pool_size > self._settings.theta_sum:
                # Set cooldown to prevent cascading (§4.2)
                await self._hot.set_cooldown(
                    anchor_v,
                    self._settings.t_cool,
                    commit=False,
                )

        if pool_size > self._settings.theta_sum:
            logger.info(
                f"TRIGGER: anchor={anchor_v}, pool_size={pool_size} > "
                f"theta_sum={self._settings.theta_sum}"
            )
            return TriggerResult(
                triggered=True,
                anchor_v=anchor_v,
            )

        logger.debug(
            f"Note {note.id}: pool_size={pool_size}/{self._settings.theta_sum} "
            f"for anchor={anchor_v} (not triggered)"
        )
        return TriggerResult(triggered=False, anchor_v=anchor_v)

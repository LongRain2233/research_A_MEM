"""
Module M_renorm: Renormalization, Closed-Loop Injection & Utility-Aware Pruning.

Aligns with Implementation Plan §3 Module 3 and Research Design §3 Module 3.

Responsibilities:
    1. Projection Operator P: Select and prioritize review domain.
    2. Synthesis Operator S: Generate Sigma (order parameter) and Delta (correction term).
    3. Closed-loop injection: Write Sigma/Delta back as new abstract notes.
    4. Utility-aware eviction with entailment gating and topology inheritance.

Data Flow:
    Input:  Trigger signal, I_v^new, utility scores {u_j}
    Output: Higher-order Sigma/Delta nodes, pruned memory M'
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from phaseforget.config.settings import Settings
from phaseforget.llm.base import BaseLLMClient
from phaseforget.llm.prompts.templates import (
    ENTAILMENT_PROMPT,
    RENORMALIZATION_PROMPT,
)
from phaseforget.memory.models.note import MemoryNote
from phaseforget.memory.state.state_manager import StateManager
from phaseforget.storage.cold_track.chroma_store import ChromaColdTrack
from phaseforget.storage.hot_track.sqlite_store import SQLiteHotTrack

logger = logging.getLogger(__name__)


@dataclass
class RenormResult:
    """Result of a renormalization operation."""
    sigma_id: str = ""
    delta_id: str = ""
    evicted_ids: list[str] = field(default_factory=list)
    review_domain_size: int = 0


class RenormalizationEngine:
    """
    Executes the renormalization → injection → eviction pipeline.
    This is the core "phase transition" engine of PhaseForget-Zettel.
    """

    def __init__(
        self,
        cold_track: ChromaColdTrack,
        hot_track: SQLiteHotTrack,
        llm_client: BaseLLMClient,
        state_manager: StateManager,
        settings: Settings,
    ):
        self._cold = cold_track
        self._hot = hot_track
        self._llm = llm_client
        self._state = state_manager
        self._settings = settings
        # Per-anchor in-progress set replaces the old global depth counter.
        # Allows concurrent renorms on different anchors (genuine parallelism)
        # while still preventing duplicate work on the same anchor.
        # True cascade is impossible: Sigma/Delta notes are created via
        # state_manager.create_note() which bypasses TriggerEngine entirely.
        self._in_progress: set[str] = set()

    async def execute(self, anchor_v: str) -> RenormResult:
        """
        Execute the full renormalization pipeline for a given anchor node.

        Steps (aligned with Implementation Plan §3 Module 3):
            1. Pop evidence pool and expand review domain.
            2. Projection operator P: prioritize abstract nodes + high utility.
            3. Synthesis operator S: generate Sigma and Delta via LLM.
            4. Closed-loop injection: write back as new notes.
            5. Utility-aware eviction with redundancy gating.

        Args:
            anchor_v: The anchor node ID that triggered renormalization.

        Returns:
            RenormResult with IDs of new nodes and evicted nodes.
        """
        # Skip if this specific anchor is already being processed.
        # Different anchors are allowed to run concurrently.
        if anchor_v in self._in_progress:
            logger.info(
                f"Renorm[{anchor_v}]: already in progress for this anchor, "
                "skipping duplicate trigger"
            )
            return RenormResult()

        if self._in_progress:
            logger.info(
                f"Renorm[{anchor_v}]: starting concurrently alongside "
                f"{len(self._in_progress)} other anchor(s)"
            )

        self._in_progress.add(anchor_v)
        try:
            return await self._execute_inner(anchor_v, RenormResult())
        finally:
            self._in_progress.discard(anchor_v)

    async def _execute_inner(
        self, anchor_v: str, result: RenormResult
    ) -> RenormResult:
        """Inner execution logic separated for depth tracking."""

        # ── Step 1: Pop evidence pool & expand review domain ─────────────
        evidence_ids = await self._hot.pop_evidence_pool(anchor_v)
        historical_ids = await self._hot.get_first_degree_neighbors(anchor_v)
        historical_ids.append(anchor_v)

        review_domain_ids = list(set(evidence_ids + historical_ids))
        result.review_domain_size = len(review_domain_ids)
        logger.info(
            f"Renorm[{anchor_v}]: evidence={len(evidence_ids)}, "
            f"review_domain={len(review_domain_ids)}"
        )

        # Fetch full note content from cold track
        raw_notes = self._cold.get_by_ids(review_domain_ids)
        if not raw_notes:
            logger.warning(f"Renorm[{anchor_v}]: no notes found in review domain")
            return result

        # ── Step 2: Projection Operator P ────────────────────────────────
        # Prioritize abstract nodes, then by utility score (descending)
        note_states = await self._hot.get_note_states([note["id"] for note in raw_notes])
        scored_notes = []
        for note_data in raw_notes:
            nid = note_data["id"]
            state = note_states.get(nid, {})
            is_abs = state.get("is_abstract", False)
            utility = state.get("utility", 0.0) or 0.0
            scored_notes.append({
                **note_data,
                "is_abstract": is_abs,
                "utility": utility,
            })

        projected_notes = sorted(
            scored_notes,
            key=lambda x: (x["is_abstract"], x["utility"]),
            reverse=True,
        )[: self._settings.projection_max_notes]

        projected_ids = {n["id"] for n in projected_notes}

        # ── Step 3: Synthesis Operator S ─────────────────────────────────
        notes_text = self._format_notes_for_llm(projected_notes)
        try:
            synth_result = await self._llm.generate_json(
                RENORMALIZATION_PROMPT.format(notes_text=notes_text),
                system_prompt="You are a knowledge synthesis engine. Output ONLY a valid JSON object with keys: sigma, delta. No explanations, no markdown.",
            )
        except Exception as e:
            logger.warning(f"Renorm[{anchor_v}]: synthesis LLM call failed ({e}), skipping renorm")
            return result

        sigma_text = synth_result.get("sigma", "") if synth_result else ""
        delta_text = synth_result.get("delta", "") if synth_result else ""

        if not sigma_text:
            logger.warning(f"Renorm[{anchor_v}]: LLM returned empty Sigma")
            return result

        # ── Step 4: Closed-loop injection ────────────────────────────────
        sigma_note = await self._state.create_note(
            content=sigma_text,
            is_abstract=True,
            tags=["Macro", f"Evolved_From_{anchor_v}"],
        )
        result.sigma_id = sigma_note.id
        logger.info(f"Renorm[{anchor_v}]: Sigma node created → {sigma_note.id}")

        if delta_text:
            delta_note = await self._state.create_note(
                content=delta_text,
                is_abstract=False,
                tags=["Conflict", f"Evolved_From_{anchor_v}"],
            )
            result.delta_id = delta_note.id
            logger.info(f"Renorm[{anchor_v}]: Delta node created → {delta_note.id}")

        # ── Step 5: Utility-aware eviction ───────────────────────────────
        eviction_stats = {
            "total_candidates": len(scored_notes),
            "utility_too_high": 0,
            "entailment_false": 0,
            "evicted": 0,
        }
        for note_data in scored_notes:
            nid = note_data["id"]

            # Check eviction condition: utility < theta_evict
            utility = note_data.get("utility", 0.0)
            if utility >= self._settings.theta_evict:
                eviction_stats["utility_too_high"] += 1
                continue

            # Entailment gate: check if Sigma logically covers this note
            # Sigma is the premise (broader synthesis), old note is the hypothesis
            # (we ask: does sigma subsume the old note's information?)
            entailment = await self._check_entailment(
                premise=sigma_text,
                hypothesis=note_data.get("content", ""),
            )
            if not entailment:
                eviction_stats["entailment_false"] += 1
                logger.info(
                    f"Renorm[{anchor_v}]: entailment=False for {nid} "
                    f"(utility={utility:.3f}, content={note_data.get('content', '')[:60]}...)"
                )
                continue

            # Execute eviction: topology inheritance + physical delete
            async with self._hot.transaction():
                await self._hot.inherit_links(
                    source_id=nid,
                    new_target_id=sigma_note.id,
                    commit=False,
                )
                self._cold.delete(nid)
                await self._hot.delete_state(nid, commit=False)

            result.evicted_ids.append(nid)
            eviction_stats["evicted"] += 1
            logger.info(f"Renorm[{anchor_v}]: evicted {nid} (utility={utility:.3f})")

        logger.info(
            f"Renorm[{anchor_v}] complete: "
            f"sigma={result.sigma_id}, delta={result.delta_id}, "
            f"eviction_stats={eviction_stats}"
        )
        return result

    async def _check_entailment(self, premise: str, hypothesis: str) -> bool:
        """
        LLM-based redundancy check (replaces previous strict entailment):
        Does the premise (new Sigma summary) fully cover the core facts of
        the hypothesis (old note)?
        """
        try:
            response = await self._llm.generate_json(
                ENTAILMENT_PROMPT.format(premise=premise, hypothesis=hypothesis),
                system_prompt="You must respond with ONLY a valid JSON object.",
            )
            return response.get("redundant", False) is True
        except Exception as e:
            logger.warning(f"Entailment check failed, defaulting to False: {e}")
            return False

    @staticmethod
    def _format_notes_for_llm(notes: list[dict]) -> str:
        """Format projected notes into a text block for the renormalization prompt."""
        lines = []
        for i, note in enumerate(notes, 1):
            abs_flag = " [ABSTRACT]" if note.get("is_abstract") else ""
            utility = note.get("utility", 0.0)
            content = note.get("content", note.get("metadata", {}).get("content", ""))
            lines.append(
                f"[Note {i}]{abs_flag} (utility={utility:.2f})\n{content}"
            )
        return "\n\n---\n\n".join(lines)

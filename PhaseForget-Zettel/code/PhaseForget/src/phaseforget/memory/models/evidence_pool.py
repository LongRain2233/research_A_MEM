"""
Evidence Pool model.

Aligns with Implementation Plan §2.2 Evidence_Pool table and
Research Design §3 Module 2: I_v^new set accumulation.

The evidence pool precisely tracks which new note IDs have accumulated
around a given anchor node v, enabling threshold-triggered renormalization.
"""

from pydantic import BaseModel, Field


class EvidencePool(BaseModel):
    """
    Tracks the incremental evidence set I_v^new for a given anchor node.

    Attributes:
        anchor_v:     The anchor node ID around which evidence accumulates.
        evidence_ids: Ordered list of new note IDs in this pool (JSON array in SQLite).
    """

    anchor_v: str
    evidence_ids: list[str] = Field(default_factory=list)

    @property
    def size(self) -> int:
        """Returns |I_v^new|, the cardinality of the evidence set."""
        return len(set(self.evidence_ids))

    def append(self, evidence_id: str) -> None:
        """Add a new evidence ID to the pool (deduplicating)."""
        if evidence_id not in self.evidence_ids:
            self.evidence_ids.append(evidence_id)

    def pop_all(self) -> list[str]:
        """Pop and return all evidence IDs, resetting the pool."""
        ids = self.evidence_ids.copy()
        self.evidence_ids.clear()
        return ids

"""
Atomic Memory Note (Zettel) data model.

Aligns with Research Design §3 Module 1 and A-Mem §3.1 Eq.(1):
    m_i = {c_i, t_i, K_i, G_i, X_i, e_i, L_i}

Extended with utility tracking from Agent KB §3.2.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MemoryNote(BaseModel):
    """
    Atomic Zettelkasten memory note.

    Attributes:
        id:           Unique identifier (UUID).
        content:      Raw interaction text c_n.
        timestamp:    Creation time t_n.
        keywords:     Extracted keywords K_n (via LLM).
        tags:         Classification tags G_n (via LLM).
        context:      Contextual description X_n (via LLM).
        is_abstract:  Whether this note is a higher-order Sigma/Delta node.
        utility:      Dynamic utility score u_i (default U_init=0.5).
        last_accessed: Last retrieval timestamp.
        cooldown_until: Timestamp until which this node cannot trigger renormalization.
        evolution_source: If abstract, the anchor_v ID from which it was evolved.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    keywords: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    context: str = Field(default="")
    is_abstract: bool = Field(default=False)
    utility: float = Field(default=0.5)
    last_accessed: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    evolution_source: Optional[str] = None

    def build_enhanced_text(self) -> str:
        """
        Build the enhanced text for embedding, aligning with Implementation Plan §2.1:
            e_n = f_enc(concat(c_n, K_n, G_n, X_n))
        """
        parts = [self.content]
        if self.keywords:
            parts.append(f"keywords: {', '.join(self.keywords)}")
        if self.tags:
            parts.append(f"tags: {', '.join(self.tags)}")
        if self.context:
            parts.append(f"context: {self.context}")
        return " ".join(parts)

    def is_in_cooldown(self, now: Optional[datetime] = None) -> bool:
        """Check whether this note is currently in cooldown period."""
        if self.cooldown_until is None:
            return False
        now = now or datetime.utcnow()
        return now < self.cooldown_until

"""
Unit tests for core data models.
"""

from datetime import datetime, timedelta

from phaseforget.memory.models.note import MemoryNote
from phaseforget.memory.models.evidence_pool import EvidencePool


class TestMemoryNote:

    def test_default_creation(self):
        note = MemoryNote(content="Test content")
        assert note.content == "Test content"
        assert note.utility == 0.5
        assert note.is_abstract is False
        assert note.id is not None

    def test_enhanced_text_building(self):
        note = MemoryNote(
            content="User likes Python",
            keywords=["Python", "programming"],
            tags=["Technology"],
            context="Discussion about programming languages",
        )
        text = note.build_enhanced_text()
        assert "User likes Python" in text
        assert "Python" in text
        assert "Technology" in text
        assert "programming languages" in text

    def test_cooldown_check(self):
        note = MemoryNote(content="Test")
        assert note.is_in_cooldown() is False

        note.cooldown_until = datetime.utcnow() + timedelta(hours=1)
        assert note.is_in_cooldown() is True

        note.cooldown_until = datetime.utcnow() - timedelta(hours=1)
        assert note.is_in_cooldown() is False


class TestEvidencePool:

    def test_append_and_size(self):
        pool = EvidencePool(anchor_v="node_1")
        pool.append("ev_1")
        pool.append("ev_2")
        pool.append("ev_1")  # duplicate
        assert pool.size == 2

    def test_pop_all(self):
        pool = EvidencePool(anchor_v="node_1", evidence_ids=["a", "b", "c"])
        popped = pool.pop_all()
        assert len(popped) == 3
        assert pool.size == 0

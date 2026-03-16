"""
Integration tests for ChromaDB Cold Track.
Tests enhanced embedding, cosine similarity search, and physical delete.
"""

import pytest

from phaseforget.memory.models.note import MemoryNote
from phaseforget.storage.cold_track.chroma_store import ChromaColdTrack


@pytest.fixture
def cold_track():
    """Create a fresh in-memory ChromaDB cold track for each test."""
    ct = ChromaColdTrack(
        persist_dir="",  # in-memory
        collection_name="test_memories",
        embedding_model="all-MiniLM-L6-v2",
    )
    yield ct
    ct.reset()


class TestChromaColdTrack:

    def test_add_and_count(self, cold_track):
        note = MemoryNote(content="Python is a programming language")
        cold_track.add_note(note)
        assert cold_track.count() == 1

    def test_add_with_enhanced_metadata(self, cold_track):
        note = MemoryNote(
            content="User enjoys hiking",
            keywords=["hiking", "outdoor", "nature"],
            tags=["Hobby", "Lifestyle"],
            context="Discussion about weekend activities",
        )
        cold_track.add_note(note)
        results = cold_track.get_by_ids([note.id])
        assert len(results) == 1
        assert "hiking" in results[0]["content"]

    def test_search_returns_similarity_scores(self, cold_track):
        notes = [
            MemoryNote(content="I love machine learning and AI research"),
            MemoryNote(content="My favorite food is sushi"),
            MemoryNote(content="Deep learning models for NLP tasks"),
        ]
        for n in notes:
            cold_track.add_note(n)

        results = cold_track.search("artificial intelligence", top_k=3)
        assert len(results) > 0
        # AI-related notes should score higher
        assert results[0]["score"] > 0

    def test_search_with_min_similarity(self, cold_track):
        cold_track.add_note(MemoryNote(content="Machine learning algorithms"))
        cold_track.add_note(MemoryNote(content="Cooking recipe for pasta"))

        # High threshold should filter out unrelated notes
        results = cold_track.search(
            "deep learning neural networks",
            top_k=10,
            min_similarity=0.5,
        )
        # Only ML-related note should pass
        for r in results:
            assert r["score"] >= 0.5

    def test_physical_delete(self, cold_track):
        note = MemoryNote(content="Temporary note to delete")
        cold_track.add_note(note)
        assert cold_track.count() == 1

        cold_track.delete(note.id)
        assert cold_track.count() == 0

    def test_batch_delete(self, cold_track):
        ids = []
        for i in range(5):
            note = MemoryNote(content=f"Note number {i}")
            cold_track.add_note(note)
            ids.append(note.id)

        assert cold_track.count() == 5
        cold_track.delete_many(ids[:3])
        assert cold_track.count() == 2

    def test_get_by_ids(self, cold_track):
        notes = [
            MemoryNote(content="First note"),
            MemoryNote(content="Second note"),
        ]
        for n in notes:
            cold_track.add_note(n)

        retrieved = cold_track.get_by_ids([notes[0].id, notes[1].id])
        assert len(retrieved) == 2

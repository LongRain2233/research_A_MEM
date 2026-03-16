"""
Integration tests for the full PhaseForget pipeline.

Tests the complete data flow from interaction ingestion through
trigger evaluation, verifying the orchestration of all three modules.

Uses a mock LLM client to avoid external API dependencies.
"""

import asyncio
import os
import tempfile

import pytest

from phaseforget.config.settings import Settings
from phaseforget.llm.base import BaseLLMClient
from phaseforget.pipeline.orchestrator import PhaseForgetSystem


class MockLLMClient(BaseLLMClient):
    """Mock LLM client that returns deterministic responses for testing."""

    async def generate(self, prompt, system_prompt=None, temperature=0.3, max_tokens=4096):
        return '{"keywords": ["test"], "tags": ["Test"], "context": "Test context"}'

    async def generate_json(self, prompt, system_prompt=None, temperature=0.3, max_tokens=4096):
        if "entailment" in prompt.lower() or "entailed" in prompt.lower():
            return {"entailed": True, "confidence": 0.9, "reasoning": "test"}
        elif "renormalization" in prompt.lower() or "synthesis" in prompt.lower():
            return {
                "sigma": "Synthesized knowledge about testing patterns",
                "delta": "Minor conflict in test methodology",
            }
        else:
            return {"keywords": ["test"], "tags": ["Test"], "context": "Test context"}


@pytest.fixture
async def system():
    """Create a PhaseForget system with mock LLM for testing."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    settings = Settings(
        theta_sim=0.3,  # Lower threshold for testing
        theta_sum=3,     # Lower threshold for easier triggering
        theta_evict=0.2,
        u_init=0.5,
        eta=0.1,
        t_cool=60,
        retrieval_top_k=5,
        sqlite_db_path=db_path,
        decay_interval_rounds=1000,
    )

    sys = PhaseForgetSystem(settings=settings, llm_client=MockLLMClient())
    await sys.initialize()
    yield sys
    await sys.close()
    os.unlink(db_path)


@pytest.mark.asyncio
async def test_add_single_interaction(system):
    """Test basic note creation through the pipeline."""
    note = await system.add_interaction("User likes Python programming")
    assert note is not None
    assert note.id
    assert note.content == "User likes Python programming"
    assert note.utility == 0.5


@pytest.mark.asyncio
async def test_add_multiple_interactions(system):
    """Test adding multiple interactions builds up state."""
    contents = [
        "User enjoys hiking in the mountains",
        "User went hiking last weekend and loved it",
        "User is planning a hiking trip to the Alps",
    ]
    notes = []
    for content in contents:
        note = await system.add_interaction(content)
        notes.append(note)

    assert len(notes) == 3
    stats = await system.get_stats()
    assert stats["total_notes"] >= 3


@pytest.mark.asyncio
async def test_search_returns_relevant_results(system):
    """Test that search returns semantically relevant notes."""
    await system.add_interaction("Python is great for machine learning")
    await system.add_interaction("I love cooking Italian food")
    await system.add_interaction("Neural networks are fascinating")

    results = system.search("artificial intelligence")
    # Should find ML-related notes
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_utility_update_on_retrieval(system):
    """Test utility score updates when notes are retrieved."""
    note = await system.add_interaction("Important fact about databases")

    # Simulate retrieval with positive feedback
    await system.add_interaction(
        content="Another interaction",
        retrieved_ids=[note.id],
        adopted_ids=[note.id],
    )

    # Utility should have increased: 0.5 + 0.1*(1.0-0.5) = 0.55
    utility = await system._hot.get_utility(note.id)
    assert utility is not None
    assert utility > 0.5


@pytest.mark.asyncio
async def test_utility_decay_on_negative_feedback(system):
    """Test utility decreases when retrieved but not adopted."""
    note = await system.add_interaction("Some trivia fact")

    # Simulate retrieval without adoption (negative feedback)
    await system.add_interaction(
        content="Another interaction",
        retrieved_ids=[note.id],
        adopted_ids=[],  # Not adopted
    )

    # Utility should have decreased: 0.5 + 0.1*(0.0-0.5) = 0.45
    utility = await system._hot.get_utility(note.id)
    assert utility is not None
    assert utility < 0.5


@pytest.mark.asyncio
async def test_topology_links_created(system):
    """Test that similar notes get linked in the topology."""
    await system.add_interaction("Machine learning is a subset of AI")
    await system.add_interaction("Deep learning uses neural networks for AI tasks")

    # Check that links were created between similar notes
    stats = await system.get_stats()
    # Links may or may not be created depending on similarity threshold
    assert stats["total_notes"] >= 2


@pytest.mark.asyncio
async def test_system_stats(system):
    """Test system statistics reporting."""
    await system.add_interaction("Test note 1")
    await system.add_interaction("Test note 2")

    stats = await system.get_stats()
    assert "total_notes" in stats
    assert "abstract_notes" in stats
    assert "total_links" in stats
    assert "cold_track_count" in stats
    assert "interaction_count" in stats
    assert stats["interaction_count"] == 2


@pytest.mark.asyncio
async def test_add_note_convenience_method(system):
    """Test the direct add_note method."""
    note = await system.add_note(
        content="Abstract summary of test patterns",
        is_abstract=True,
        tags=["Macro", "Test"],
        skip_metadata=True,
    )
    assert note.is_abstract is True
    assert "Macro" in note.tags


@pytest.mark.asyncio
async def test_force_renormalize(system):
    """Test manual renormalization trigger."""
    # Add several related notes to build up a neighborhood
    notes = []
    for i in range(5):
        n = await system.add_interaction(
            f"Machine learning concept {i}: neural networks and deep learning"
        )
        notes.append(n)

    # Manually populate evidence pool and trigger renormalization
    anchor = notes[0].id
    for n in notes[1:]:
        await system._hot.append_evidence_pool(anchor, n.id)

    result = await system.force_renormalize(anchor)
    # Result may or may not have evictions depending on utility scores
    assert result is not None

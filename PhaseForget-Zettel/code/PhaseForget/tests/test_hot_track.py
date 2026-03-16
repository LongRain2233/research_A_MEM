"""
Integration tests for SQLite Hot Track.
Tests the three core tables and cascade delete behavior (§4.1).
"""

import asyncio
import os
import tempfile

import pytest

from phaseforget.storage.hot_track.sqlite_store import SQLiteHotTrack


@pytest.fixture
async def hot_track():
    """Create a temporary SQLite hot track for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    ht = SQLiteHotTrack(db_path=path)
    await ht.initialize()
    yield ht
    await ht.close()
    os.unlink(path)


@pytest.mark.asyncio
async def test_insert_and_get_utility(hot_track):
    await hot_track.insert_state("note_1", utility=0.5)
    u = await hot_track.get_utility("note_1")
    assert u == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_utility_momentum_update(hot_track):
    await hot_track.insert_state("note_1", utility=0.5)
    # Positive feedback: u = 0.5 + 0.1 * (1.0 - 0.5) = 0.55
    await hot_track.update_utility("note_1", reward=1.0, eta=0.1)
    u = await hot_track.get_utility("note_1")
    assert u == pytest.approx(0.55, abs=1e-6)


@pytest.mark.asyncio
async def test_cascade_delete_clears_links(hot_track):
    """Verify §4.1: ON DELETE CASCADE clears dangling links."""
    await hot_track.insert_state("A")
    await hot_track.insert_state("B")
    await hot_track.insert_state("C")
    await hot_track.insert_links("A", ["B", "C"])

    # Delete B - links from A→B should be cascaded
    await hot_track.delete_state("B")
    neighbors = await hot_track.get_first_degree_neighbors("A")
    assert "B" not in neighbors
    assert "C" in neighbors


@pytest.mark.asyncio
async def test_evidence_pool_operations(hot_track):
    await hot_track.insert_state("anchor_1")
    await hot_track.append_evidence_pool("anchor_1", "ev_1")
    await hot_track.append_evidence_pool("anchor_1", "ev_2")
    await hot_track.append_evidence_pool("anchor_1", "ev_1")  # duplicate

    size = await hot_track.get_pool_size("anchor_1")
    assert size == 2

    ids = await hot_track.pop_evidence_pool("anchor_1")
    assert set(ids) == {"ev_1", "ev_2"}
    assert await hot_track.get_pool_size("anchor_1") == 0


@pytest.mark.asyncio
async def test_cooldown_management(hot_track):
    await hot_track.insert_state("node_1")
    assert await hot_track.is_in_cooldown("node_1") is False

    await hot_track.set_cooldown("node_1", cooldown_seconds=3600)
    assert await hot_track.is_in_cooldown("node_1") is True


@pytest.mark.asyncio
async def test_filter_valid_ids(hot_track):
    await hot_track.insert_state("valid_1")
    await hot_track.insert_state("valid_2")

    result = await hot_track.filter_valid_ids(
        ["valid_1", "ghost_1", "valid_2", "ghost_2"]
    )
    assert set(result) == {"valid_1", "valid_2"}

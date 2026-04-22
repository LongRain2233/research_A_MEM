"""
Hot Track - SQLite State Management Store.

Aligns with Implementation Plan §2.2, managing three core logical tables:
    - Memory_State:   utility scores, abstraction flags, cooldown locks
    - Memory_Links:   explicit Zettel topology links with ON DELETE CASCADE
    - Evidence_Pool:  incremental evidence ID sets (JSON arrays)

Also implements boundary defenses from §4:
    - §4.1 Dangling link prevention via CASCADE
    - §4.2 Cooldown management
    - §4.3 Cold start state initialization
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

# ── Schema DDL ───────────────────────────────────────────────────────────────

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS Memory_State (
    id                TEXT PRIMARY KEY,
    utility_score     REAL NOT NULL DEFAULT 0.5,
    is_abstract       INTEGER NOT NULL DEFAULT 0,
    last_evolved_at   TEXT,
    cooldown_until    TEXT,
    created_at        TEXT NOT NULL,
    last_accessed_at  TEXT
);

CREATE TABLE IF NOT EXISTS Memory_Links (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES Memory_State(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES Memory_State(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS Evidence_Pool (
    anchor_v      TEXT PRIMARY KEY,
    evidence_ids  TEXT NOT NULL DEFAULT '[]',
    FOREIGN KEY (anchor_v) REFERENCES Memory_State(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS System_Meta (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_state_decay
ON Memory_State(is_abstract, last_accessed_at);

CREATE INDEX IF NOT EXISTS idx_memory_state_cooldown
ON Memory_State(cooldown_until);

CREATE INDEX IF NOT EXISTS idx_memory_links_source
ON Memory_Links(source_id);

CREATE INDEX IF NOT EXISTS idx_memory_links_target
ON Memory_Links(target_id);
"""


class SQLiteHotTrack:
    """
    Async SQLite store for high-frequency state operations.

    All methods are async to avoid blocking the main interaction loop.
    Uses aiosqlite for non-blocking database access.
    """

    def __init__(self, db_path: str = "./data/phaseforget.db"):
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._db_lock = asyncio.Lock()
        self._transaction_owner: Optional[asyncio.Task] = None
        self._transaction_depth = 0

    def _current_task(self) -> Optional[asyncio.Task]:
        try:
            return asyncio.current_task()
        except RuntimeError:
            return None

    def _in_owned_transaction(self) -> bool:
        return self._transaction_owner is self._current_task()

    @asynccontextmanager
    async def _guard_connection(self):
        if self._in_owned_transaction():
            yield
            return

        async with self._db_lock:
            yield

    @asynccontextmanager
    async def transaction(self):
        """
        Group multiple SQLite operations into a single atomic commit.

        This lock is connection-wide because aiosqlite uses one shared
        connection; letting other tasks interleave queries mid-transaction
        would expose partial state and defeat batching.
        """
        current_task = self._current_task()
        if self._transaction_owner is current_task:
            self._transaction_depth += 1
            try:
                yield
            finally:
                self._transaction_depth -= 1
            return

        async with self._db_lock:
            self._transaction_owner = current_task
            self._transaction_depth = 1
            await self._db.execute("BEGIN")
            try:
                yield
            except Exception:
                await self._db.rollback()
                raise
            else:
                await self._db.commit()
            finally:
                self._transaction_depth = 0
                self._transaction_owner = None

    async def _commit_if_needed(self, commit: bool) -> None:
        if commit and not self._in_owned_transaction():
            await self._db.commit()

    async def initialize(self) -> None:
        """Open connection and create tables if not exist."""
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(SCHEMA_DDL)
        await self._db.commit()
        logger.info(f"SQLiteHotTrack initialized: {self._db_path}")

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ── Memory_State Operations ──────────────────────────────────────────

    async def insert_state(
        self,
        note_id: str,
        utility: float = 0.5,
        is_abstract: bool = False,
        commit: bool = True,
    ) -> None:
        """Insert a new memory state record (cold-start with U_init)."""
        now = datetime.utcnow().isoformat()
        async with self._guard_connection():
            await self._db.execute(
                """INSERT OR IGNORE INTO Memory_State
                   (id, utility_score, is_abstract, created_at, last_accessed_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (note_id, utility, int(is_abstract), now, now),
            )
            await self._commit_if_needed(commit)

    async def update_utility(
        self, note_id: str, reward: float, eta: float = 0.1, commit: bool = True
    ) -> None:
        """
        Momentum utility update: u_i <- u_i + eta * (r_i - u_i)
        Aligns with Implementation Plan §3 Module 1.
        """
        now = datetime.utcnow().isoformat()
        async with self._guard_connection():
            await self._db.execute(
                """UPDATE Memory_State
                   SET utility_score = utility_score + ? * (? - utility_score),
                       last_accessed_at = ?
                   WHERE id = ?""",
                (eta, reward, now, note_id),
            )
            await self._commit_if_needed(commit)

    async def update_utilities_batch(
        self,
        note_rewards: dict[str, float],
        eta: float = 0.1,
        commit: bool = True,
    ) -> None:
        """Batch utility updates for a retrieval feedback set."""
        if not note_rewards:
            return

        now = datetime.utcnow().isoformat()
        rows = [(eta, reward, now, note_id) for note_id, reward in note_rewards.items()]
        async with self._guard_connection():
            await self._db.executemany(
                """UPDATE Memory_State
                   SET utility_score = utility_score + ? * (? - utility_score),
                       last_accessed_at = ?
                   WHERE id = ?""",
                rows,
            )
            await self._commit_if_needed(commit)

    async def apply_utility_penalty(
        self, note_id: str, penalty_factor: float = 0.9, commit: bool = True
    ) -> None:
        """Apply multiplicative penalty to edge nodes during renormalization."""
        async with self._guard_connection():
            await self._db.execute(
                "UPDATE Memory_State SET utility_score = utility_score * ? WHERE id = ?",
                (penalty_factor, note_id),
            )
            await self._commit_if_needed(commit)

    async def apply_global_decay(
        self,
        decay_factor: float = 0.95,
        grace_period_hours: float = 1.0,
        commit: bool = True,
    ) -> int:
        """
        Apply time-based decay to nodes not recently accessed and not abstract.

        Abstract (Sigma/Delta) nodes are excluded because they represent
        synthesized higher-order knowledge that should not erode over time.
        Recently accessed nodes (within *grace_period_hours*) are also
        excluded to avoid punishing actively useful memories.

        Anti-freeze mechanism from Implementation Plan §3 Module 1.
        Returns number of affected rows.
        """
        cutoff = (
            datetime.utcnow() - timedelta(hours=grace_period_hours)
        ).isoformat()
        async with self._guard_connection():
            cursor = await self._db.execute(
                """UPDATE Memory_State
                   SET utility_score = utility_score * ?
                   WHERE is_abstract = 0
                     AND (last_accessed_at IS NULL OR last_accessed_at < ?)""",
                (decay_factor, cutoff),
            )
            await self._commit_if_needed(commit)
            return cursor.rowcount

    async def get_utility(self, note_id: str) -> Optional[float]:
        """Get current utility score for a note."""
        async with self._guard_connection():
            cursor = await self._db.execute(
                "SELECT utility_score FROM Memory_State WHERE id = ?", (note_id,)
            )
            row = await cursor.fetchone()
            return row[0] if row else None

    async def is_abstract(self, note_id: str) -> bool:
        """Check if a note is marked as abstract (higher-order Sigma/Delta)."""
        async with self._guard_connection():
            cursor = await self._db.execute(
                "SELECT is_abstract FROM Memory_State WHERE id = ?", (note_id,)
            )
            row = await cursor.fetchone()
            return bool(row[0]) if row else False

    async def get_note_states(self, note_ids: list[str]) -> dict[str, dict]:
        """
        Batch fetch utility / abstract / cooldown state for multiple notes.
        """
        if not note_ids:
            return {}

        placeholders = ",".join("?" for _ in note_ids)
        async with self._guard_connection():
            cursor = await self._db.execute(
                f"""SELECT id, utility_score, is_abstract, cooldown_until
                    FROM Memory_State
                    WHERE id IN ({placeholders})""",
                note_ids,
            )
            rows = await cursor.fetchall()

        return {
            row[0]: {
                "utility": row[1],
                "is_abstract": bool(row[2]),
                "cooldown_until": row[3],
            }
            for row in rows
        }

    async def filter_valid_ids(self, ids: list[str]) -> list[str]:
        """
        Anti-ghost-read: filter IDs to only those existing in Memory_State.
        Aligns with Implementation Plan §3 Module 2 ghost ID interception.
        """
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        async with self._guard_connection():
            cursor = await self._db.execute(
                f"SELECT id FROM Memory_State WHERE id IN ({placeholders})", ids
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def delete_state(self, note_id: str, commit: bool = True) -> None:
        """Delete a memory state (CASCADE clears links and evidence pools)."""
        async with self._guard_connection():
            await self._db.execute("DELETE FROM Memory_State WHERE id = ?", (note_id,))
            await self._commit_if_needed(commit)

    # ── Cooldown Management (§4.2 Cascading Phase Transition Suppression) ─

    async def set_cooldown(
        self, note_id: str, cooldown_seconds: int, commit: bool = True
    ) -> None:
        """Set cooldown period for a node to prevent cascading renormalization."""
        until = (datetime.utcnow() + timedelta(seconds=cooldown_seconds)).isoformat()
        async with self._guard_connection():
            await self._db.execute(
                """UPDATE Memory_State
                   SET cooldown_until = ?, last_evolved_at = ?
                   WHERE id = ?""",
                (until, datetime.utcnow().isoformat(), note_id),
            )
            await self._commit_if_needed(commit)

    async def is_in_cooldown(self, note_id: str) -> bool:
        """Check if a node is currently in cooldown."""
        async with self._guard_connection():
            cursor = await self._db.execute(
                "SELECT cooldown_until FROM Memory_State WHERE id = ?", (note_id,)
            )
            row = await cursor.fetchone()
            if not row or not row[0]:
                return False
            cooldown_until = datetime.fromisoformat(row[0])
            return datetime.utcnow() < cooldown_until

    async def get_cooldown_states(self, note_ids: list[str]) -> dict[str, bool]:
        """Batch check cooldown flags for multiple notes."""
        if not note_ids:
            return {}

        states = await self.get_note_states(note_ids)
        now = datetime.utcnow()
        cooldowns = {nid: False for nid in note_ids}
        for nid, state in states.items():
            raw = state.get("cooldown_until")
            cooldowns[nid] = bool(raw and datetime.fromisoformat(raw) > now)
        return cooldowns

    # ── Memory_Links Operations (Zettel Topology) ────────────────────────

    async def insert_links(
        self, source_id: str, target_ids: list[str], commit: bool = True
    ) -> None:
        """Create Zettel links from source to multiple targets (L_n)."""
        if not target_ids:
            return
        now = datetime.utcnow().isoformat()
        async with self._guard_connection():
            await self._db.executemany(
                """INSERT OR IGNORE INTO Memory_Links (source_id, target_id, created_at)
                   VALUES (?, ?, ?)""",
                [(source_id, tid, now) for tid in target_ids],
            )
            await self._commit_if_needed(commit)

    async def get_first_degree_neighbors(self, note_id: str) -> list[str]:
        """Get all directly linked neighbor IDs (both directions)."""
        async with self._guard_connection():
            cursor = await self._db.execute(
                """SELECT DISTINCT target_id FROM Memory_Links WHERE source_id = ?
                   UNION
                   SELECT DISTINCT source_id FROM Memory_Links WHERE target_id = ?""",
                (note_id, note_id),
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def get_parent_nodes(self, note_id: str) -> list[str]:
        """Get nodes that link TO this note (upstream in topology)."""
        async with self._guard_connection():
            cursor = await self._db.execute(
                "SELECT source_id FROM Memory_Links WHERE target_id = ?", (note_id,)
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]

    async def inherit_links(
        self, source_id: str, new_target_id: str, commit: bool = True
    ) -> None:
        """
        Topology inheritance: transfer all links of source_id to new_target_id.
        Used during eviction to preserve graph connectivity.
        Aligns with Implementation Plan §3 Module 3.
        """
        neighbors = await self.get_first_degree_neighbors(source_id)
        valid_targets = [n for n in neighbors if n != new_target_id]
        if valid_targets:
            await self.insert_links(new_target_id, valid_targets, commit=commit)

    # ── Evidence_Pool Operations (§3 Module 2) ───────────────────────────

    async def append_evidence_pool(
        self, anchor_v: str, evidence_id: str, commit: bool = True
    ) -> None:
        """Append a new evidence ID to the anchor's incremental pool."""
        await self.append_evidence_to_anchors([anchor_v], evidence_id, commit=commit)

    async def append_evidence_to_anchors(
        self,
        anchor_ids: list[str],
        evidence_id: str,
        commit: bool = True,
    ) -> dict[str, int]:
        """
        Append the same evidence id to multiple anchors and return new pool sizes.
        """
        unique_anchor_ids = list(dict.fromkeys(anchor_ids))
        if not unique_anchor_ids:
            return {}

        placeholders = ",".join("?" for _ in unique_anchor_ids)
        async with self._guard_connection():
            cursor = await self._db.execute(
                f"""SELECT anchor_v, evidence_ids
                    FROM Evidence_Pool
                    WHERE anchor_v IN ({placeholders})""",
                unique_anchor_ids,
            )
            existing_rows = await cursor.fetchall()
            existing = {row[0]: json.loads(row[1]) for row in existing_rows}

            pool_sizes: dict[str, int] = {}
            updates: list[tuple[str, str]] = []
            inserts: list[tuple[str, str]] = []
            for anchor_id in unique_anchor_ids:
                ids = existing.get(anchor_id, [])
                if evidence_id not in ids:
                    ids = ids + [evidence_id]
                pool_sizes[anchor_id] = len(set(ids))
                payload = json.dumps(ids)
                if anchor_id in existing:
                    updates.append((payload, anchor_id))
                else:
                    inserts.append((anchor_id, payload))

            if updates:
                await self._db.executemany(
                    "UPDATE Evidence_Pool SET evidence_ids = ? WHERE anchor_v = ?",
                    updates,
                )
            if inserts:
                await self._db.executemany(
                    "INSERT INTO Evidence_Pool (anchor_v, evidence_ids) VALUES (?, ?)",
                    inserts,
                )
            await self._commit_if_needed(commit)
            return pool_sizes

    async def get_pool_size(self, anchor_v: str) -> int:
        """Get |I_v^new| for the given anchor node."""
        async with self._guard_connection():
            cursor = await self._db.execute(
                "SELECT evidence_ids FROM Evidence_Pool WHERE anchor_v = ?",
                (anchor_v,),
            )
            row = await cursor.fetchone()
            if not row:
                return 0
            return len(set(json.loads(row[0])))

    async def pop_evidence_pool(self, anchor_v: str, commit: bool = True) -> list[str]:
        """Pop and return all evidence IDs, resetting the pool."""
        async with self._guard_connection():
            cursor = await self._db.execute(
                "SELECT evidence_ids FROM Evidence_Pool WHERE anchor_v = ?",
                (anchor_v,),
            )
            row = await cursor.fetchone()
            if not row:
                return []

            ids = json.loads(row[0])
            await self._db.execute(
                "UPDATE Evidence_Pool SET evidence_ids = '[]' WHERE anchor_v = ?",
                (anchor_v,),
            )
            await self._commit_if_needed(commit)
            return ids

    # ── System Meta (断点续传 / interaction counter) ─────────────────────

    async def increment_interaction_count(self) -> int:
        """Persist interaction count to survive restarts."""
        async with self._guard_connection():
            cursor = await self._db.execute(
                "SELECT value FROM System_Meta WHERE key = 'interaction_count'"
            )
            row = await cursor.fetchone()
            new_count = (int(row[0]) + 1) if row else 1
            await self._db.execute(
                """INSERT INTO System_Meta (key, value) VALUES ('interaction_count', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (str(new_count),),
            )
            await self._commit_if_needed(True)
            return new_count

    async def get_interaction_count(self) -> int:
        """Read persisted interaction count."""
        async with self._guard_connection():
            cursor = await self._db.execute(
                "SELECT value FROM System_Meta WHERE key = 'interaction_count'"
            )
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    # ── Bulk Query Helpers ───────────────────────────────────────────────

    async def get_all_states(self) -> list[dict]:
        """Retrieve all memory state records."""
        async with self._guard_connection():
            cursor = await self._db.execute(
                "SELECT id, utility_score, is_abstract, cooldown_until FROM Memory_State"
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "utility": r[1],
                    "is_abstract": bool(r[2]),
                    "cooldown_until": r[3],
                }
                for r in rows
            ]

    async def get_stats(self) -> dict:
        """Return basic statistics about the hot track."""
        async with self._guard_connection():
            cursor = await self._db.execute("SELECT COUNT(*) FROM Memory_State")
            total = (await cursor.fetchone())[0]
            cursor = await self._db.execute(
                "SELECT COUNT(*) FROM Memory_State WHERE is_abstract = 1"
            )
            abstract = (await cursor.fetchone())[0]
            cursor = await self._db.execute("SELECT COUNT(*) FROM Memory_Links")
            links = (await cursor.fetchone())[0]
            cursor = await self._db.execute(
                "SELECT value FROM System_Meta WHERE key = 'interaction_count'"
            )
            row = await cursor.fetchone()
            interaction_count = int(row[0]) if row else 0
            return {
                "total_notes": total,
                "abstract_notes": abstract,
                "total_links": links,
                "interaction_count": interaction_count,
            }

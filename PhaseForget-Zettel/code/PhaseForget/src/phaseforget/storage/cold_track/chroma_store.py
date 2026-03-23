"""
Cold Track - ChromaDB Semantic Vector Store.

Aligns with Implementation Plan §2.1:
    - Read-heavy, append-only writes + physical deletes.
    - No metadata UPDATE operations (avoids I/O lock contention).
    - Enhanced embedding: content || keywords || tags || context.

Reference implementation pattern: A-Mem retrievers.py
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from phaseforget.memory.models.note import MemoryNote

logger = logging.getLogger(__name__)


class ChromaColdTrack:
    """
    Semantic vector store backed by ChromaDB.

    Responsibilities:
        - Store enhanced embeddings for memory notes.
        - Perform Top-K cosine similarity recall.
        - Execute physical deletes upon eviction.
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "phaseforget_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        if persist_dir:
            import os
            os.makedirs(persist_dir, exist_ok=True)
            # PersistentClient ensures data survives restarts (断点续传基础)
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            # Empty persist_dir → ephemeral in-memory client (used in tests)
            self._client = chromadb.EphemeralClient()
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            f"ChromaColdTrack initialized: collection='{collection_name}', "
            f"persist_dir='{persist_dir}', model='{embedding_model}'"
        )

    def add_note(self, note: MemoryNote) -> None:
        """
        Append a memory note with enhanced embedding.
        Aligns with e_n = f_enc(c_n || K_n || G_n || X_n).
        """
        enhanced_text = note.build_enhanced_text()
        metadata = {
            "content": note.content,
            "keywords": json.dumps(note.keywords),
            "tags": json.dumps(note.tags),
            "context": note.context,
            "is_abstract": str(note.is_abstract),
            "timestamp": note.timestamp.isoformat(),
        }
        self._collection.add(
            documents=[enhanced_text],
            metadatas=[metadata],
            ids=[note.id],
        )
        logger.debug(f"Cold track: added note {note.id}")

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        min_similarity: Optional[float] = None,
    ) -> list[dict]:
        """
        Perform Top-K cosine similarity search.

        Returns list of dicts with keys: id, score, content, metadata.
        Optionally filters by absolute similarity lower bound (theta_sim).
        """
        # ChromaDB raises ValueError when querying an empty collection
        total = self._collection.count()
        if total == 0:
            return []

        n = min(top_k, total)
        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}", exc_info=True)
            return []

        hits = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        for i, doc_id in enumerate(results["ids"][0]):
            # ChromaDB with hnsw:space=cosine returns cosine distance (1 - cosine_sim)
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            if min_similarity is not None and similarity < min_similarity:
                continue

            metadata = results["metadatas"][0][i] if results["metadatas"] else {}
            # Deserialize JSON fields
            for key in ("keywords", "tags"):
                if key in metadata and isinstance(metadata[key], str):
                    try:
                        metadata[key] = json.loads(metadata[key])
                    except json.JSONDecodeError:
                        pass

            hits.append({
                "id": doc_id,
                "score": similarity,
                "content": results["documents"][0][i] if results["documents"] else "",
                "metadata": metadata,
            })

        return hits

    def get_by_ids(self, ids: list[str]) -> list[dict]:
        """Retrieve notes by their IDs."""
        if not ids:
            return []
        results = self._collection.get(
            ids=ids,
            include=["documents", "metadatas"],
        )
        notes = []
        for i, doc_id in enumerate(results["ids"]):
            metadata = results["metadatas"][i] if results["metadatas"] else {}
            # Deserialize JSON fields (same as search method)
            for key in ("keywords", "tags"):
                if key in metadata and isinstance(metadata[key], str):
                    try:
                        metadata[key] = json.loads(metadata[key])
                    except json.JSONDecodeError:
                        pass
            notes.append({
                "id": doc_id,
                "content": results["documents"][i] if results["documents"] else "",
                "metadata": metadata,
            })
        return notes

    def update_note(self, note: MemoryNote) -> None:
        """
        Update an existing note's embedding and metadata in-place.
        Called after process_memory mutates a neighbor's context/tags so that
        the vector space reflects the updated semantic state.
        """
        enhanced_text = note.build_enhanced_text()
        metadata = {
            "content": note.content,
            "keywords": json.dumps(note.keywords),
            "tags": json.dumps(note.tags),
            "context": note.context,
            "is_abstract": str(note.is_abstract),
            "timestamp": note.timestamp.isoformat(),
        }
        self._collection.update(
            documents=[enhanced_text],
            metadatas=[metadata],
            ids=[note.id],
        )
        logger.debug(f"Cold track: updated note {note.id}")

    def delete(self, note_id: str) -> None:
        """Physical delete - permanently remove a note from the vector store."""
        self._collection.delete(ids=[note_id])
        logger.debug(f"Cold track: deleted note {note_id}")

    def delete_many(self, note_ids: list[str]) -> None:
        """Batch physical delete."""
        if note_ids:
            self._collection.delete(ids=note_ids)
            logger.debug(f"Cold track: batch deleted {len(note_ids)} notes")

    def count(self) -> int:
        """Return total number of stored notes."""
        return self._collection.count()

    def reset(self) -> None:
        """
        Reset by dropping and recreating the collection.
        Avoids client.reset() which is disabled by default in newer ChromaDB.
        """
        name = self._collection.name
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning("Cold track: collection reset (delete + recreate)")

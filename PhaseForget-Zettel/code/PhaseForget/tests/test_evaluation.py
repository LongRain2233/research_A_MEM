"""
Tests for dataset loaders and baseline adapters.
"""

import json
import os
import tempfile

import pytest

from phaseforget.evaluation.loaders.locomo import LoCoMoLoader
from phaseforget.evaluation.loaders.personamem import PersonaMemLoader
from phaseforget.evaluation.loaders.dialsim import DialSimLoader
from phaseforget.evaluation.baselines.memorybank_adapter import MemoryBankAdapter


class TestLoCoMoLoader:

    def _make_official_record(self):
        """Build a minimal locomo10.json-style record for testing."""
        return {
            "sample_id": "test_sample_0",
            "qa": [
                {
                    "question": "When did Alice go to the support group?",
                    "answer": "7 May 2023",
                    "evidence": ["D1:3"],
                    "category": 2,
                },
                {
                    "question": "What is Alice's hobby?",
                    "answer": "painting",
                    "evidence": ["D1:5"],
                    "category": 1,
                },
                {
                    # adversarial — should be excluded
                    "question": "What did Alice NOT do?",
                    "evidence": ["D1:1"],
                    "category": 5,
                    "adversarial_answer": "painting",
                },
            ],
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "1:00 pm on 7 May, 2023",
                "session_1": [
                    {"speaker": "Alice", "dia_id": "D1:1", "text": "Hey Bob!"},
                    {"speaker": "Bob",   "dia_id": "D1:2", "text": "Hey Alice! What's new?"},
                    {"speaker": "Alice", "dia_id": "D1:3", "text": "I went to a support group."},
                ],
                "session_2_date_time": "2:00 pm on 8 May, 2023",
                "session_2": [
                    {"speaker": "Alice", "dia_id": "D2:1", "text": "I started painting recently."},
                    {"speaker": "Bob",   "dia_id": "D2:2", "text": "That sounds wonderful!"},
                ],
            },
            "event_summary": "Alice and Bob discuss hobbies.",
            "session_summary": "Two sessions of conversation.",
            "sample_id": "test_sample_0",
        }

    def test_load_official_format(self):
        """Test loading the official locomo10.json format.

        A-MEM evaluation protocol: all session_N turns are merged into one
        benchmark session per record, so the system sees the full conversation
        history before answering questions.
        """
        data = [self._make_official_record()]
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump(data, f)

        loader = LoCoMoLoader()
        sessions = loader.load(path)
        os.unlink(path)

        # 1 record → 1 merged benchmark session (not one per session_N)
        assert len(sessions) == 1

        # All turns merged: session_1 (3 turns) + session_2 (2 turns) = 5 turns
        assert len(sessions[0]["dialogue"]) == 5

        # QA excludes adversarial (category 5)
        assert len(sessions[0]["questions"]) == 2
        assert sessions[0]["questions"][0]["answer"] == "7 May 2023"

    def test_speaker_role_mapping(self):
        """speaker_a → user, speaker_b → assistant (across merged dialogue)."""
        data = [self._make_official_record()]
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump(data, f)

        loader = LoCoMoLoader()
        sessions = loader.load(path)
        os.unlink(path)

        dialogue = sessions[0]["dialogue"]
        assert dialogue[0]["role"] == "user"       # Alice = speaker_a
        assert dialogue[1]["role"] == "assistant"  # Bob = speaker_b

    def test_adversarial_excluded(self):
        """Category-5 (adversarial) QA should be excluded."""
        data = [self._make_official_record()]
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump(data, f)

        loader = LoCoMoLoader()
        sessions = loader.load(path)
        os.unlink(path)

        categories = [q["category"] for q in sessions[0]["questions"]]
        assert 5 not in categories

    def test_session_id_format(self):
        """session_id should be the sample_id (one session per record)."""
        data = [self._make_official_record()]
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump(data, f)

        loader = LoCoMoLoader()
        sessions = loader.load(path)
        os.unlink(path)

        assert sessions[0]["session_id"] == "test_sample_0"

    def test_load_missing_file(self):
        loader = LoCoMoLoader()
        result = loader.load("/nonexistent/path.json")
        assert result == []

    def test_multi_record_file(self):
        """Multiple top-level records produce one merged session each."""
        data = [self._make_official_record(), self._make_official_record()]
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump(data, f)

        loader = LoCoMoLoader()
        sessions = loader.load(path)
        os.unlink(path)

        # 2 records × 1 merged session each = 2 sessions
        assert len(sessions) == 2


class TestPersonaMemLoader:

    def test_load_multi_session(self):
        data = [
            {
                "user_id": "user_1",
                "sessions": [
                    {
                        "session_id": "s1",
                        "dialogue": [
                            {"role": "user", "content": "I like jazz", "created_at": "2025-01-01"},
                        ],
                    },
                    {
                        "session_id": "s2",
                        "dialogue": [
                            {"role": "user", "content": "Actually I prefer rock now", "created_at": "2025-02-01"},
                        ],
                    },
                ],
                "eval_questions": [
                    {"question": "Music preference?", "answer": "rock"},
                ],
            }
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump(data, f)

        loader = PersonaMemLoader()
        sessions = loader.load(path)
        os.unlink(path)

        assert len(sessions) == 1
        # Sessions should be flattened into sequential dialogue
        assert len(sessions[0]["dialogue"]) == 2


class TestDialSimLoader:

    def test_load_episode_format(self):
        data = [
            {
                "episode_id": "ep1",
                "dialogue": [
                    {"speaker": "Alice", "text": "Have you seen Bob?"},
                    {"speaker": "Charlie", "text": "He went to the store."},
                ],
                "questions": [
                    {"question": "Where is Bob?", "answer": "the store"},
                ],
            }
        ]
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        with open(path, "w") as f:
            json.dump(data, f)

        loader = DialSimLoader()
        sessions = loader.load(path)
        os.unlink(path)

        assert len(sessions) == 1
        assert sessions[0]["dialogue"][0]["speaker"] == "Alice"


class TestMemoryBankBaseline:

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        adapter = MemoryBankAdapter(retention_threshold=0.01)

        await adapter.add_interaction("Machine learning is important")
        await adapter.add_interaction("I love cooking pasta")
        await adapter.add_interaction("Neural networks for AI")

        results = adapter.search("artificial intelligence", top_k=3)
        assert len(results) > 0
        assert adapter.get_memory_count() == 3

    @pytest.mark.asyncio
    async def test_reset(self):
        adapter = MemoryBankAdapter()
        await adapter.add_interaction("Test content")
        assert adapter.get_memory_count() == 1

        adapter.reset()
        assert adapter.get_memory_count() == 0

"""
DialSim dataset loader.

DialSim is a long-range multi-party dialogue QA dataset based on TV show
transcripts. Source: A-Mem §4.1

Expected data format (JSON):
[
  {
    "episode_id": "...",
    "dialogue": [{"speaker": "...", "text": "...", "timestamp": "..."}, ...],
    "questions": [{"question": "...", "answer": "..."}, ...]
  },
  ...
]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from phaseforget.evaluation.benchmark import DatasetLoader

logger = logging.getLogger(__name__)


class DialSimLoader(DatasetLoader):
    """Loader for the DialSim multi-party dialogue dataset."""

    def name(self) -> str:
        return "DialSim"

    def load(self, path: str) -> list[dict[str, Any]]:
        """Load DialSim dataset and normalize to unified format."""
        file_path = Path(path)
        if not file_path.exists():
            logger.error(f"DialSim dataset file not found: {path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict):
            raw_data = list(raw_data.values()) if all(
                isinstance(v, dict) for v in raw_data.values()
            ) else [raw_data]

        results = []
        for episode in raw_data:
            dialogue = []
            for turn in episode.get("dialogue", episode.get("transcript", [])):
                speaker = turn.get("speaker", turn.get("role", "unknown"))
                dialogue.append({
                    "role": "user" if speaker != "assistant" else "assistant",
                    "content": turn.get("text", turn.get("content", "")),
                    "created_at": turn.get("timestamp", turn.get("created_at", "")),
                    "speaker": speaker,
                })

            questions = []
            for qa in episode.get("questions", episode.get("qa_pairs", [])):
                questions.append({
                    "question": qa.get("question", qa.get("query", "")),
                    "answer": qa.get("answer", qa.get("response", "")),
                    "type": qa.get("type", "multi_party"),
                })

            results.append({
                "session_id": episode.get("episode_id", episode.get("id", "")),
                "dialogue": dialogue,
                "questions": questions,
            })

        logger.info(f"DialSim loaded: {len(results)} episodes")
        return results

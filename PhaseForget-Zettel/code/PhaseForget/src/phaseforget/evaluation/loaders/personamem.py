"""
PersonaMem dataset loader.

PersonaMem is for evaluating user preference evolution (Concept Drift)
and personalization maintenance under conflicting dialogue.
Source: RGMem §1

Expected data format (JSON):
[
  {
    "user_id": "...",
    "sessions": [
      {
        "session_id": "...",
        "dialogue": [{"role": "...", "content": "...", "created_at": "..."}, ...],
        "persona_state": {"key": "value", ...}
      },
      ...
    ],
    "eval_questions": [
      {"question": "...", "answer": "...", "session_context": "..."},
      ...
    ]
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


class PersonaMemLoader(DatasetLoader):
    """Loader for the PersonaMem preference evolution dataset."""

    def name(self) -> str:
        return "PersonaMem"

    def load(self, path: str) -> list[dict[str, Any]]:
        """
        Load PersonaMem dataset.

        Flattens multi-session user data into a sequential dialogue with
        interleaved persona state snapshots for evolution tracking.
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.error(f"PersonaMem dataset file not found: {path}")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict):
            raw_data = [raw_data]

        results = []
        for user_data in raw_data:
            # Flatten all sessions into a single sequential dialogue
            full_dialogue = []
            sessions = user_data.get("sessions", [])
            for session in sessions:
                for turn in session.get("dialogue", []):
                    full_dialogue.append({
                        "role": turn.get("role", "user").lower(),
                        "content": turn.get("content", ""),
                        "created_at": turn.get("created_at", ""),
                        "session_id": session.get("session_id", ""),
                    })

            questions = []
            for qa in user_data.get("eval_questions", user_data.get("questions", [])):
                questions.append({
                    "question": qa.get("question", ""),
                    "answer": qa.get("answer", ""),
                    "type": qa.get("type", "preference_tracking"),
                })

            results.append({
                "session_id": user_data.get("user_id", "unknown"),
                "dialogue": full_dialogue,
                "questions": questions,
            })

        logger.info(f"PersonaMem loaded: {len(results)} users")
        return results

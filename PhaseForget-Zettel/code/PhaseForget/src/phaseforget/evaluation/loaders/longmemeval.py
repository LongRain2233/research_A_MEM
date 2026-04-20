"""
LongMemEval dataset loader.

LongMemEval is a long-horizon conversational memory QA benchmark where each
top-level record contains one evaluation question plus a long history of
prior chat sessions (`haystack_sessions`).

Expected data format (JSON):
[
  {
    "question_id": "...",
    "question_type": "temporal-reasoning",
    "question": "...",
    "question_date": "2023/05/30 (Tue) 23:31",
    "answer": "...",
    "haystack_session_ids": ["session_1", "session_2", ...],
    "haystack_dates": ["2023/05/20 ...", "2023/05/20 ...", ...],
    "haystack_sessions": [
      [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
      ],
      ...
    ],
    "answer_session_ids": ["session_k"]
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

_QUESTION_TYPE_TO_CATEGORY = {
    "single-session-assistant": 1,
    "single-session-user": 1,
    "temporal-reasoning": 2,
    "multi-session": 3,
    "knowledge-update": 4,
    "single-session-preference": 5,
}


class LongMemEvalLoader(DatasetLoader):
    """Loader for the LongMemEval conversational memory benchmark."""

    def __init__(self, record_indices: list[int] | None = None):
        self._record_indices = record_indices

    def name(self) -> str:
        return "LongMemEval"

    def load(self, path: str) -> list[dict[str, Any]]:
        file_path = Path(path)
        if not file_path.exists():
            logger.error(f"LongMemEval dataset file not found: {path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LongMemEval JSON: {e}")
            return []

        if isinstance(raw_data, dict):
            if all(isinstance(v, dict) for v in raw_data.values()):
                records = list(raw_data.values())
            else:
                records = [raw_data]
        elif isinstance(raw_data, list):
            records = raw_data
        else:
            logger.error(f"Unexpected LongMemEval root type: {type(raw_data)}")
            return []

        if self._record_indices is not None:
            total = len(records)
            selected = []
            for idx in self._record_indices:
                if 0 <= idx < total:
                    selected.append(records[idx])
                else:
                    logger.warning(
                        "LongMemEval record index %s out of range [0, %s], skipped",
                        idx,
                        total - 1,
                    )
            records = selected
            logger.info(
                "LongMemEval: using record_indices=%s -> %s/%s records selected",
                self._record_indices,
                len(records),
                total,
            )

        sessions = []
        for rec_idx, record in enumerate(records):
            session = self._normalize_record(record, rec_idx)
            if session is not None:
                sessions.append(session)

        logger.info("LongMemEval: loaded %s sessions from %s", len(sessions), path)
        return sessions

    def _normalize_record(
        self, record: dict[str, Any], default_index: int
    ) -> dict[str, Any] | None:
        question = str(record.get("question", "")).strip()
        answer = str(record.get("answer", "")).strip()
        if not question or not answer:
            logger.warning(
                "LongMemEval record %s skipped: missing question or answer",
                record.get("question_id", default_index),
            )
            return None

        haystack_sessions = record.get("haystack_sessions", [])
        haystack_dates = record.get("haystack_dates", [])
        haystack_session_ids = record.get("haystack_session_ids", [])

        dialogue: list[dict[str, Any]] = []
        for sess_idx, session_turns in enumerate(haystack_sessions):
            if not isinstance(session_turns, list):
                logger.debug(
                    "LongMemEval record %s session %s is not a list, skipped",
                    record.get("question_id", default_index),
                    sess_idx,
                )
                continue

            created_at = (
                str(haystack_dates[sess_idx]).strip()
                if sess_idx < len(haystack_dates)
                else ""
            )
            source_session_id = (
                str(haystack_session_ids[sess_idx]).strip()
                if sess_idx < len(haystack_session_ids)
                else f"session_{sess_idx}"
            )

            for turn_idx, turn in enumerate(session_turns):
                if not isinstance(turn, dict):
                    continue
                content = str(turn.get("content", "")).strip()
                if not content:
                    continue

                role = str(turn.get("role", "user")).strip().lower() or "user"
                normalized_role = role if role in {"user", "assistant"} else "user"
                speaker = str(
                    turn.get("speaker")
                    or ("assistant" if normalized_role == "assistant" else "user")
                ).strip()

                dialogue.append(
                    {
                        "role": normalized_role,
                        "speaker": speaker,
                        "content": content,
                        "created_at": created_at,
                        "session_id": source_session_id,
                        "turn_index": turn_idx,
                    }
                )

        if not dialogue:
            logger.warning(
                "LongMemEval record %s skipped: no dialogue turns found",
                record.get("question_id", default_index),
            )
            return None

        question_type = str(record.get("question_type", "")).strip()
        question_payload = {
            "question": question,
            "answer": answer,
            "category": _QUESTION_TYPE_TO_CATEGORY.get(question_type),
            "question_type": question_type,
            "question_date": str(record.get("question_date", "")).strip(),
            "answer_session_ids": record.get("answer_session_ids", []),
        }

        session_id = str(record.get("question_id", f"longmemeval_{default_index}"))
        return {
            "session_id": session_id,
            "dialogue": dialogue,
            "questions": [question_payload],
            "meta": {
                "question_id": session_id,
                "question_type": question_type,
                "question_date": str(record.get("question_date", "")).strip(),
                "haystack_session_count": len(haystack_sessions),
                "answer_session_ids": record.get("answer_session_ids", []),
            },
        }

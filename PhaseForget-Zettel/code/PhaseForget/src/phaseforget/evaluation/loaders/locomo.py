"""
LoCoMo dataset loader.

LoCoMo is used for evaluating long-context dependency and multi-hop reasoning.
Source: A-Mem §4.1, RGMem §1

Official LoCoMo data format (locomo10.json):
[
  {
    "sample_id": "...",
    "qa": [
      {
        "question": "...",
        "answer": "...",
        "evidence": ["D1:3", "D2:7"],
        "category": 1
      },
      ...
      // category 5 = adversarial (answer intentionally wrong, excluded)
    ],
    "conversation": {
      "speaker_a": "Caroline",
      "speaker_b": "Melanie",
      "session_1_date_time": "1:56 pm on 8 May, 2023",
      "session_1": [
        {"speaker": "Caroline", "dia_id": "D1:1", "text": "..."},
        {"speaker": "Melanie",  "dia_id": "D1:2", "text": "..."},
        ...
      ],
      "session_2_date_time": "...",
      "session_2": [...],
      ...
    },
    "event_summary": "...",
    "observation": "...",
    "session_summary": "..."
  },
  ...
]

Each top-level record is one complete conversation. Each session_N inside
`conversation` becomes one benchmark session. QA covers the entire
conversation and is attached to every session.

Category guide:
  1 = single-hop factual
  2 = temporal reasoning
  3 = multi-hop reasoning
  4 = detailed comprehension
  5 = adversarial (excluded — answer is intentionally wrong)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from phaseforget.evaluation.benchmark import DatasetLoader

logger = logging.getLogger(__name__)


class LoCoMoLoader(DatasetLoader):
    """
    Loader for the official LoCoMo long-context conversation dataset.

    Converts locomo10.json format into benchmark sessions where each
    session_N in `conversation` becomes one independent benchmark session,
    all sharing the full QA question set (category 1-4 only).

    Args:
        record_indices: Optional list of 0-based record indices to load.
            e.g. [0, 2, 4] loads only the 1st, 3rd, 5th records.
            None (default) loads all records.
            locomo10.json has 10 records (indices 0-9).
    """

    def __init__(self, record_indices: list[int] | None = None):
        self._record_indices = record_indices

    def name(self) -> str:
        return "LoCoMo"

    def load(self, path: str) -> list[dict[str, Any]]:
        """
        Load LoCoMo dataset from the official JSON file.

        Returns a flat list of benchmark sessions. Each item has:
            session_id : str   e.g. "sample_0_session_1"
            dialogue   : list  [{"role": "user"|"assistant", "content": "...",
                                 "speaker": "Caroline", "dia_id": "D1:1"}, ...]
            questions  : list  [{"question": "...", "answer": "...",
                                 "category": 1, "evidence": [...]}]
        """
        file_path = Path(path)
        if not file_path.exists():
            logger.error(f"LoCoMo dataset file not found: {path}")
            return []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LoCoMo JSON: {e}")
            return []

        # Support both list-of-records and single-record formats
        if isinstance(raw_data, dict):
            records = [raw_data]
        elif isinstance(raw_data, list):
            records = raw_data
        else:
            logger.error(f"Unexpected LoCoMo root type: {type(raw_data)}")
            return []

        # Filter by record_indices if specified
        if self._record_indices is not None:
            total = len(records)
            selected = []
            for idx in self._record_indices:
                if 0 <= idx < total:
                    selected.append(records[idx])
                else:
                    logger.warning(
                        f"LoCoMo record index {idx} out of range [0, {total - 1}], skipped"
                    )
            records = selected
            logger.info(
                f"LoCoMo: using record_indices={self._record_indices} → "
                f"{len(records)}/{total} records selected"
            )

        sessions = []
        for rec_idx, record in enumerate(records):
            sample_id = record.get("sample_id", str(rec_idx))
            rec_sessions = self._extract_sessions(record, sample_id)
            sessions.extend(rec_sessions)

        logger.info(f"LoCoMo: loaded {len(sessions)} sessions from {path}")
        return sessions

    # ── Internal helpers ──────────────────────────────────────────────────

    def _extract_sessions(
        self, record: dict, sample_id: str
    ) -> list[dict[str, Any]]:
        """Parse one top-level LoCoMo record into benchmark sessions."""

        # ── 1. Extract QA (exclude adversarial category 5) ───────────────
        questions = []
        for qa in record.get("qa", []):
            category = qa.get("category", 0)
            if category == 5:
                continue  # adversarial — answer is intentionally wrong
            answer = qa.get("answer")
            if answer is None:
                continue
            questions.append({
                "question": qa["question"],
                "answer": str(answer),
                "category": category,
                "evidence": qa.get("evidence", []),
            })

        if not questions:
            logger.warning(f"LoCoMo record {sample_id}: no valid QA found")

        # ── 2. Extract dialogue sessions from `conversation` ──────────────
        conv = record.get("conversation", {})

        # speaker_a treated as "user", speaker_b treated as "assistant"
        speaker_a = conv.get("speaker_a", "")
        speaker_b = conv.get("speaker_b", "")

        # Gather session_N keys (integers only, value must be a list)
        session_keys = sorted(
            [
                k for k in conv
                if k.startswith("session_")
                and k.split("_")[1].isdigit()
                and isinstance(conv[k], list)
            ],
            key=lambda x: int(x.split("_")[1]),
        )

        if not session_keys:
            logger.warning(
                f"LoCoMo record {sample_id}: no session_N dialogue found. "
                f"Available keys: {list(conv.keys())}"
            )
            return []

        # ── 3. Merge all session_N turns into one benchmark session ──────
        # Following A-MEM evaluation protocol: the entire conversation history
        # (all sessions combined in chronological order) is fed to the memory
        # system before evaluating QA. This ensures the system has access to
        # all context when answering questions that span multiple sessions.
        all_dialogue = []
        for key in session_keys:
            session_num = key.split("_")[1]
            date_time = conv.get(f"session_{session_num}_date_time", "")

            for turn in conv[key]:
                speaker = turn.get("speaker", "")
                text = turn.get("text", "").strip()
                if not text:
                    continue

                # Map speaker name to role
                if speaker_a and speaker == speaker_a:
                    role = "user"
                elif speaker_b and speaker == speaker_b:
                    role = "assistant"
                else:
                    role = "user"  # unknown speaker defaults to user

                all_dialogue.append({
                    "role": role,
                    "speaker": speaker,
                    "content": text,
                    "created_at": date_time,
                    "dia_id": turn.get("dia_id", ""),
                    "session_key": key,
                })

        if not all_dialogue:
            logger.warning(f"LoCoMo record {sample_id}: no dialogue turns found")
            return []

        logger.info(
            f"LoCoMo record {sample_id}: merged {len(session_keys)} sessions → "
            f"{len(all_dialogue)} turns, {len(questions)} QA questions, "
            f"speakers=[{speaker_a}, {speaker_b}]"
        )
        return [{
            "session_id": sample_id,
            "dialogue": all_dialogue,
            "questions": questions,
            "meta": {
                "sample_id": sample_id,
                "num_sessions": len(session_keys),
                "speaker_a": speaker_a,
                "speaker_b": speaker_b,
            },
        }]

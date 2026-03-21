"""
Benchmark runner for PhaseForget-Zettel.

Aligns with Research Design §7 and Implementation Plan §5.2 Sprint 4:
    - Datasets: LoCoMo, PersonaMem, DialSim
    - Baselines: A-Mem, MemGPT, MemoryBank
    - Metrics: F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT Similarity,
               Memory Usage (MB), Retrieval Time (us)

Execution flow per dataset session:
    1. Feed dialogue turns sequentially to each system.
    2. After all turns, query each system with evaluation questions.
    3. Retrieve relevant memories, use LLM to generate an answer.
    4. Measure retrieval time and compute all QA metrics.
    5. Record memory usage at regular intervals.
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from phaseforget.evaluation.metrics import (
    EvalMetrics,
    RetrievalTimer,
    compute_bleu1,
    compute_f1,
    compute_rouge_l,
    compute_rouge2,
    compute_meteor,
    compute_sbert,
)

logger = logging.getLogger(__name__)

# ── Prompt Templates (aligned with A-MEM evaluation protocol) ──────────

_QUERY_EXPANSION_PROMPT = """\
Given the following question, generate several search keywords separated by commas.
Question: {question}
Output a JSON object: {{"keywords": "keyword1, keyword2, keyword3"}}"""

_ANSWER_PROMPT_DEFAULT = """\
Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:

Respond with a JSON object: {{"answer": "<your short answer>"}}"""

_ANSWER_PROMPT_TEMPORAL = """\
Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.

Question: {question} Short answer:

Respond with a JSON object: {{"answer": "<your short answer>"}}"""

_ANSWER_PROMPT_ADVERSARIAL = """\
Based on the context: {context}, answer the following question. {question}

Select the correct answer: {option_a} or {option_b}  Short answer:

Respond with a JSON object: {{"answer": "<selected answer>"}}"""


class DatasetLoader(ABC):
    """Abstract loader for evaluation datasets."""

    @abstractmethod
    def load(self, path: str) -> list[dict[str, Any]]:
        """
        Load dataset from path.
        Each item should have at minimum: 'dialogue' (list of turns),
        'questions' (list of QA pairs with 'question' and 'answer' keys).
        """
        ...

    @abstractmethod
    def name(self) -> str:
        """Return dataset name (LoCoMo, PersonaMem, DialSim)."""
        ...


class BaselineAdapter(ABC):
    """Abstract adapter for baseline memory systems."""

    @abstractmethod
    async def add_interaction(self, content: str) -> None:
        """Add an interaction to the baseline system."""
        ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[dict]:
        """Search the baseline system."""
        ...

    @abstractmethod
    def name(self) -> str:
        """Return baseline name (A-Mem, MemGPT, MemoryBank)."""
        ...


def _get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


class BenchmarkRunner:
    """
    Runs a full benchmark comparing PhaseForget against baselines.

    Collects F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT Similarity,
    Memory Usage, and Retrieval Time metrics across multiple datasets.

    LLM answer generation:
        If llm_client is provided, retrieved memories are fed into the LLM
        to generate a natural language answer before metric computation.
        This matches the evaluation protocol in A-MEM (Xu et al., 2025).
        If llm_client is None, falls back to concatenating retrieved text.

    Supports checkpoint/resume (断点续传): if a run is interrupted, re-running
    with the same checkpoint_path will skip already-completed sessions.

    Usage:
        runner = BenchmarkRunner(system, llm_client=llm, checkpoint_path="ckpt.json")
        runner.register_baseline(MemoryBankAdapter())
        results = await runner.run(LoCoMoLoader(), "path/to/locomo.json")
        runner.print_report(results)
    """

    def __init__(
        self,
        system,
        llm_client=None,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Args:
            system:          A PhaseForgetSystem instance (the system under test).
            llm_client:      Optional LLM client for answer generation. If provided,
                             retrieved memories are used as context for LLM to generate
                             answers (matches A-MEM evaluation protocol).
            checkpoint_path: Optional path for saving/resuming benchmark progress.
        """
        self._system = system
        self._llm_client = llm_client
        self._baselines: list[BaselineAdapter] = []
        self._memory_sample_interval = 50
        self._checkpoint_path = Path(
            checkpoint_path or "./data/bench_checkpoint.json"
        )
        self._sbert_model = None  # lazy-loaded once on first use

    def register_baseline(self, baseline: BaselineAdapter) -> None:
        """Register a baseline system for comparison."""
        self._baselines.append(baseline)

    def _get_sbert_model(self):
        """Load SBERT model once and cache it."""
        if self._sbert_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("SBERT model loaded for similarity scoring")
            except Exception as e:
                logger.warning(f"Failed to load SBERT model, SBERT scores will be 0: {e}")
        return self._sbert_model

    async def _expand_query(self, question: str) -> str:
        """Extract search keywords from a question via LLM (A-MEM protocol)."""
        if self._llm_client is None:
            return question
        try:
            result = await self._llm_client.generate_json(
                _QUERY_EXPANSION_PROMPT.format(question=question),
                system_prompt="You must respond with a JSON object.",
                temperature=0.1,
                max_tokens=256,
            )
            keywords = result.get("keywords", "")
            if keywords and keywords.strip():
                return keywords.strip()
        except Exception as e:
            logger.debug(f"Query expansion failed, using raw question: {e}")
        return question

    async def _generate_answer(
        self,
        question: str,
        retrieved: list[dict],
        category: int | None = None,
        reference: str = "",
    ) -> str:
        """
        Generate answer using LLM with retrieved memories as context.

        Implements category-specific prompt engineering aligned with A-MEM:
            - Category 2 (temporal): instruct model to use conversation dates
            - Category 5 (adversarial): binary choice format
            - Default: short-phrase answer using exact context words
        """
        if not retrieved:
            return ""

        # Build context aligned with A-MEM's memory_str format:
        # "talk start time:<ts> memory content:<c> memory context:<ctx> memory keywords:<kw> memory tags:<tags>"
        ctx_lines = []
        for r in retrieved:
            meta = r.get("metadata", {})
            raw_content = meta.get("content", r.get("content", ""))
            ctx_summary = meta.get("context", "")
            # keywords/tags may be list (deserialized) or str; normalize to str
            kw_raw = meta.get("keywords", "")
            tags_raw = meta.get("tags", "")
            keywords = ", ".join(kw_raw) if isinstance(kw_raw, list) else str(kw_raw)
            tags = ", ".join(tags_raw) if isinstance(tags_raw, list) else str(tags_raw)
            timestamp = meta.get("timestamp", meta.get("created_at", ""))
            parts = []
            if timestamp:
                parts.append(f"talk start time:{timestamp}")
            parts.append(f"memory content:{raw_content}")
            if ctx_summary:
                parts.append(f"memory context:{ctx_summary}")
            if keywords:
                parts.append(f"memory keywords:{keywords}")
            if tags:
                parts.append(f"memory tags:{tags}")
            ctx_lines.append(" ".join(parts))
        context = "\n".join(ctx_lines)

        if self._llm_client is not None:
            # A-MEM default temperature=0.7; adversarial uses self-configured value
            temperature = 0.7

            if category == 5 and reference:
                options = ["Not mentioned in the conversation", reference]
                if random.random() < 0.5:
                    options.reverse()
                prompt = _ANSWER_PROMPT_ADVERSARIAL.format(
                    context=context,
                    question=question,
                    option_a=options[0],
                    option_b=options[1],
                )
                temperature = 0.5
            elif category == 2:
                prompt = _ANSWER_PROMPT_TEMPORAL.format(
                    context=context,
                    question=question,
                )
            else:
                prompt = _ANSWER_PROMPT_DEFAULT.format(
                    context=context,
                    question=question,
                )

            try:
                # Use generate_json + system prompt to enforce structured output,
                # aligned with A-MEM's response_format={"type": "json_schema"} protocol.
                result = await self._llm_client.generate_json(
                    prompt=prompt,
                    system_prompt="You must respond with a JSON object.",
                    temperature=temperature,
                    max_tokens=256,
                )
                # A-MEM parses: parsed.get("short_answer") or parsed.get("answer")
                answer = result.get("short_answer") or result.get("answer") or ""
                if answer and str(answer).strip():
                    logger.debug(f"LLM generated answer: {str(answer).strip()[:80]}")
                    return str(answer).strip()
                else:
                    logger.debug("LLM returned empty JSON answer, using retrieval fallback")
            except Exception as e:
                logger.debug(f"LLM answer generation failed: {type(e).__name__}: {e}, using retrieval fallback")

        fallback = " ".join(r.get("content", "")[:200] for r in retrieved[:3])
        logger.debug(f"Using retrieval fallback answer (len={len(fallback)}): {fallback[:80]}")
        return fallback

    def _score_all(self, prediction: str, reference: str, metrics: EvalMetrics) -> None:
        """Compute all 6 QA metrics and append to the metrics object."""
        metrics.f1_scores.append(compute_f1(prediction, reference))
        metrics.bleu_scores.append(compute_bleu1(prediction, reference))
        metrics.rouge_l_scores.append(compute_rouge_l(prediction, reference))
        metrics.rouge2_scores.append(compute_rouge2(prediction, reference))
        metrics.meteor_scores.append(compute_meteor(prediction, reference))
        sbert_model = self._get_sbert_model()
        if sbert_model is not None:
            metrics.sbert_scores.append(compute_sbert(prediction, reference, sbert_model))

    # ── Checkpoint helpers ────────────────────────────────────────────────

    def _load_checkpoint(self) -> dict:
        if self._checkpoint_path.exists():
            try:
                with open(self._checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(
                    f"Checkpoint loaded from {self._checkpoint_path}: "
                    f"completed_sessions={data.get('completed_sessions', [])}"
                )
                return data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint, starting fresh: {e}")
        return {}

    def _save_checkpoint(self, completed_sessions: list, partial_metrics: dict) -> None:
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "completed_sessions": completed_sessions,
            "partial_metrics": {
                name: m.to_dict()
                for name, m in partial_metrics.items()
            },
        }
        try:
            with open(self._checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _restore_metrics(self, checkpoint: dict) -> dict[str, EvalMetrics]:
        restored: dict[str, EvalMetrics] = {}
        for name, raw in checkpoint.get("partial_metrics", {}).items():
            restored[name] = EvalMetrics.from_dict(raw)
        return restored

    def _append_failed_scores(self, metrics: EvalMetrics) -> None:
        """Append zeroed QA metrics when scoring fails."""
        for lst in (
            metrics.f1_scores,
            metrics.bleu_scores,
            metrics.rouge_l_scores,
            metrics.rouge2_scores,
            metrics.meteor_scores,
        ):
            lst.append(0.0)

    def _score_with_optional_category(
        self,
        metrics: EvalMetrics,
        prediction: str,
        reference: str,
        retrieval_time_us: float,
        category: int | None,
    ) -> None:
        """Score overall metrics and mirror scores into category bucket if provided."""
        metrics.retrieval_times_us.append(retrieval_time_us)
        self._score_all(prediction, reference, metrics)

        if category is not None:
            cat_metrics = metrics.category_metrics(category)
            cat_metrics.retrieval_times_us.append(retrieval_time_us)
            self._score_all(prediction, reference, cat_metrics)

    def clear_checkpoint(self) -> None:
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()
            logger.info(f"Checkpoint cleared: {self._checkpoint_path}")

    # ── Main run loop ─────────────────────────────────────────────────────

    async def run(
        self,
        dataset_loader: DatasetLoader,
        dataset_path: str,
        max_sessions: Optional[int] = None,
    ) -> dict[str, EvalMetrics]:
        """
        Execute a full benchmark on a dataset with checkpoint/resume support.

        For each session in the dataset:
            1. Feed dialogue turns sequentially to all systems.
            2. For each QA question: retrieve → LLM generate → compute 6 metrics.
            3. Save checkpoint after each session.

        Args:
            dataset_loader: A DatasetLoader instance for the target dataset.
            dataset_path:   Path to the dataset file/directory.
            max_sessions:   Limit the number of sessions to evaluate.

        Returns:
            Dict mapping system name to its EvalMetrics.
        """
        dataset_name = dataset_loader.name()
        use_llm = self._llm_client is not None
        logger.info(
            f"Starting benchmark: dataset={dataset_name}, "
            f"baselines={[b.name() for b in self._baselines]}, "
            f"llm_generation={'enabled' if use_llm else 'disabled (retrieval fallback)'}, "
            f"checkpoint={self._checkpoint_path}"
        )

        sessions = dataset_loader.load(dataset_path)
        if max_sessions:
            sessions = sessions[:max_sessions]

        if not sessions:
            logger.warning(f"No sessions loaded from {dataset_path}")
            return {}

        checkpoint = self._load_checkpoint()
        completed_sessions: list = checkpoint.get("completed_sessions", [])
        all_systems = ["PhaseForget"] + [b.name() for b in self._baselines]

        if completed_sessions:
            metrics = self._restore_metrics(checkpoint)
            for name in all_systems:
                if name not in metrics:
                    metrics[name] = EvalMetrics()
            skipped = len([s for s in sessions
                           if s.get("session_id", str(sessions.index(s)))
                           in completed_sessions])
            logger.info(f"Resuming benchmark: {skipped}/{len(sessions)} sessions already done")
        else:
            metrics = {name: EvalMetrics() for name in all_systems}

        for session_idx, session in enumerate(sessions):
            session_id = session.get("session_id", str(session_idx))

            if session_id in completed_sessions:
                logger.debug(f"Skipping completed session {session_id}")
                continue

            dialogue = session.get("dialogue", [])
            questions = session.get("questions", [])

            logger.info(
                f"Session {session_idx + 1}/{len(sessions)} "
                f"(id={session_id}): {len(dialogue)} turns, {len(questions)} questions"
            )

            # ── Phase 1: Feed dialogue turns ──────────────────────────
            for turn_idx, turn in enumerate(dialogue):
                raw_text = turn.get("content", "")
                if not raw_text.strip():
                    continue

                # Align with A-MEM ingestion: "Speaker X says: <text>"
                speaker = turn.get("speaker", "")
                date_time = turn.get("created_at", "")
                content = f"Speaker {speaker} says: {raw_text}" if speaker else raw_text
                if date_time:
                    content = f"[{date_time}] {content}"

                try:
                    await self._system.add_interaction(content=content)
                except Exception as e:
                    logger.error(
                        f"PhaseForget failed on session={session_id} turn={turn_idx}: {e}",
                        exc_info=True,
                    )

                for baseline in self._baselines:
                    try:
                        await baseline.add_interaction(content=content)
                    except Exception as e:
                        logger.warning(
                            f"Baseline '{baseline.name()}' failed on "
                            f"session={session_id} turn={turn_idx}: {e}"
                        )

                if turn_idx > 0 and turn_idx % self._memory_sample_interval == 0:
                    mem_mb = _get_process_memory_mb()
                    if mem_mb > 0:
                        metrics["PhaseForget"].memory_usage_mb.append(mem_mb)

            # ── Phase 2: Evaluate QA questions ────────────────────────
            for qa in questions:
                question = qa.get("question", "")
                reference = qa.get("answer", "")
                raw_category = qa.get("category")
                category = raw_category if isinstance(raw_category, int) else None
                if not question or not reference:
                    continue

                # Evaluate PhaseForget: keyword expand → graph search → generate → score
                try:
                    expanded_query = await self._expand_query(question)
                    with RetrievalTimer() as timer:
                        pf_results = await self._system.search_with_graph(expanded_query)
                    prediction = await self._generate_answer(
                        question, pf_results,
                        category=category,
                        reference=reference,
                    )
                    self._score_with_optional_category(
                        metrics=metrics["PhaseForget"],
                        prediction=prediction,
                        reference=reference,
                        retrieval_time_us=timer.elapsed_us,
                        category=category,
                    )
                except Exception as e:
                    logger.warning(f"PhaseForget QA failed for '{question[:50]}': {e}")
                    m = metrics["PhaseForget"]
                    self._append_failed_scores(m)
                    if category is not None:
                        self._append_failed_scores(m.category_metrics(category))

                # Evaluate each baseline: retrieve → generate → score
                for baseline in self._baselines:
                    try:
                        with RetrievalTimer() as timer:
                            bl_results = baseline.search(question, top_k=5)
                        prediction = await self._generate_answer(question, bl_results)
                        self._score_with_optional_category(
                            metrics=metrics[baseline.name()],
                            prediction=prediction,
                            reference=reference,
                            retrieval_time_us=timer.elapsed_us,
                            category=category,
                        )
                    except Exception as e:
                        logger.warning(f"Baseline {baseline.name()} QA failed: {e}")
                        # Append zeros to keep sample counts aligned
                        m = metrics[baseline.name()]
                        self._append_failed_scores(m)
                        if category is not None:
                            self._append_failed_scores(m.category_metrics(category))

            # Record final memory snapshot for this session
            mem_mb = _get_process_memory_mb()
            if mem_mb > 0:
                for name in all_systems:
                    metrics[name].memory_usage_mb.append(mem_mb)

            completed_sessions.append(session_id)
            self._save_checkpoint(completed_sessions, metrics)
            logger.info(
                f"Session {session_id} complete "
                f"({len(completed_sessions)}/{len(sessions)}), checkpoint saved"
            )

        logger.info(f"Benchmark complete: {dataset_name}")
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()
            logger.info("Checkpoint file removed after successful completion")
        return metrics

    @staticmethod
    def print_report(results: dict[str, EvalMetrics]) -> str:
        """
        Format benchmark results into a human-readable report.

        Displays all 6 QA metrics (F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT)
        plus retrieval latency and sample count.

        Returns the report string (also prints to stdout).
        """
        lines = [
            "=" * 100,
            "PhaseForget-Zettel Benchmark Report (with 6 QA Metrics)",
            "=" * 100,
            "",
            f"{'System':<20} {'F1':>8} {'BLEU':>8} {'ROUGE-L':>8} "
            f"{'ROUGE2':>8} {'METEOR':>8} {'SBERT':>8} "
            f"{'RetTime(us)':>12} {'N':>6}",
            "-" * 100,
        ]

        for name, m in results.items():
            n = len(m.f1_scores)
            lines.append(
                f"{name:<20} "
                f"{m.avg_f1:>8.4f} {m.avg_bleu:>8.4f} "
                f"{m.avg_rouge_l:>8.4f} {m.avg_rouge2:>8.4f} "
                f"{m.avg_meteor:>8.4f} {m.avg_sbert:>8.2f} "
                f"{m.avg_retrieval_time_us:>12.1f} {n:>6}"
            )

        has_category_breakdown = any(m.by_category for m in results.values())
        if has_category_breakdown:
            lines.extend([
                "",
                "Category Breakdown (from QA.category)",
                "-" * 100,
                f"{'System':<20} {'Cat':>5} {'F1':>8} {'BLEU':>8} {'ROUGE-L':>8} "
                f"{'ROUGE2':>8} {'METEOR':>8} {'SBERT':>8} {'RetTime(us)':>12} {'N':>6}",
                "-" * 100,
            ])
            for name, m in results.items():
                for cat in sorted(m.by_category):
                    cat_m = m.by_category[cat]
                    n = len(cat_m.f1_scores)
                    lines.append(
                        f"{name:<20} {cat:>5} "
                        f"{cat_m.avg_f1:>8.4f} {cat_m.avg_bleu:>8.4f} "
                        f"{cat_m.avg_rouge_l:>8.4f} {cat_m.avg_rouge2:>8.4f} "
                        f"{cat_m.avg_meteor:>8.4f} {cat_m.avg_sbert:>8.2f} "
                        f"{cat_m.avg_retrieval_time_us:>12.1f} {n:>6}"
                    )

        lines.extend(["", "=" * 100])
        report = "\n".join(lines)
        print(report)
        return report

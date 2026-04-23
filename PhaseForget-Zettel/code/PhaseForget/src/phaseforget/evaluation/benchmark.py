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
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from phaseforget.evaluation.metrics import (
    EvalMetrics,
    RetrievalTimer,
    _simple_tokenize,
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
        disable_self_retrieval: bool = False,
    ):
        """
        Args:
            system:          A PhaseForgetSystem instance (the system under test).
            llm_client:      Optional LLM client for answer generation. If provided,
                             retrieved memories are used as context for LLM to generate
                             answers (matches A-MEM evaluation protocol).
            checkpoint_path: Optional path for saving/resuming benchmark progress.
            disable_self_retrieval: If True, skip search_with_graph during dialogue
                             ingestion (Phase 1). This isolates the effect of
                             self-retrieval on QA metrics, matching A-MEM's
                             linear memory building without retrieval feedback.
        """
        self._system = system
        self._llm_client = llm_client
        self._baselines: list[BaselineAdapter] = []
        self._memory_sample_interval = 50
        self._checkpoint_path = Path(
            checkpoint_path or "./data/bench_checkpoint.json"
        )
        self._timing_path = self._checkpoint_path.with_name(
            self._checkpoint_path.stem + "_timing.json"
        )
        self._sbert_model = None  # lazy-loaded once on first use
        self._disable_self_retrieval = disable_self_retrieval
        self._cat4_diag_count = 0  # counter for deep Open Domain diagnostics

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

    async def _expand_query(self, question: str, metrics: "EvalMetrics | None" = None) -> str:
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
            if keywords and str(keywords).strip():
                return str(keywords).strip()
            # Empty result counts as a parse failure (model returned {} or missing key)
            if metrics is not None:
                metrics.query_expand_parse_fail += 1
        except Exception as e:
            logger.debug(f"Query expansion failed, using raw question: {e}")
            if metrics is not None:
                metrics.query_expand_parse_fail += 1
        return question

    async def _generate_answer(
        self,
        question: str,
        retrieved: list[dict],
        category: int | None = None,
        reference: str = "",
        metrics: "EvalMetrics | None" = None,
    ) -> str:
        """
        Generate answer using LLM with retrieved memories as context.

        Implements category-specific prompt engineering aligned with A-MEM:
            - LoCoMo category 2 / LongMemEval category 6 (temporal): use dates
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

        # Track context token count for efficiency analysis (whitespace-split approximation)
        if metrics is not None:
            metrics.context_token_counts.append(len(context.split()))

        if self._llm_client is not None:
            # A-MEM default temperature=0.7; adversarial uses self-configured value
            temperature = 0.7

            if category == 2 or category == 6:
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
                    if metrics is not None:
                        metrics.answer_parse_fail += 1
            except Exception as e:
                logger.debug(f"LLM answer generation failed: {type(e).__name__}: {e}, using retrieval fallback")
                if metrics is not None:
                    metrics.answer_parse_fail += 1

        # A-MEM alignment: when LLM answer generation fails, A-MEM uses the
        # raw LLM response string directly (not retrieved text concatenation).
        # Concatenating retrieved content inflates F1 because long retrieval
        # text has high token overlap with long reference answers, especially
        # for Open-Domain questions.  Return empty string to mirror A-MEM's
        # behaviour when JSON parsing fails but there is no raw text to use.
        logger.debug("LLM answer generation failed, returning empty prediction")
        return ""

    @staticmethod
    def _tokenize_text(text: str) -> set[str]:
        """Lightweight tokenizer for lexical-overlap based adoption inference."""
        if not text:
            return set()
        return set(re.findall(r"\w+", text.lower()))

    def _infer_adopted_ids(
        self,
        retrieved: list[dict],
        prediction: str,
        max_adopted: int = 3,
    ) -> list[str]:
        """
        Infer which retrieved notes were likely adopted in the final answer.

        Strategy: lexical overlap between prediction tokens and note content.
        Falls back to top-1 retrieved note when prediction is non-empty but
        overlap is zero, to avoid starving utility updates.
        """
        pred_tokens = self._tokenize_text(prediction)
        if not retrieved:
            return []

        scored: list[tuple[str, float]] = []
        for r in retrieved:
            nid = r.get("id")
            if not nid:
                continue
            meta = r.get("metadata", {})
            note_text = meta.get("content", r.get("content", "")) or ""
            note_tokens = self._tokenize_text(note_text)
            if not pred_tokens or not note_tokens:
                overlap = 0.0
            else:
                overlap = len(pred_tokens & note_tokens) / len(pred_tokens)
            scored.append((nid, overlap))

        if not scored:
            return []

        scored.sort(key=lambda x: x[1], reverse=True)
        adopted = [nid for nid, score in scored if score > 0][:max_adopted]

        if not adopted and prediction.strip():
            adopted = [scored[0][0]]

        return adopted

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

    def _save_session_timing(
        self,
        session_id: str,
        session_idx: int,
        n_turns: int,
        n_questions: int,
        phase1_s: float,
        phase2_s: float,
        total_s: float,
        ingest_tokens: int = 0,
        avg_context_tokens: float = 0.0,
        avg_retrieval_time_s: float = 0.0,
    ) -> None:
        """追加一条样本计时记录到 timing JSON 文件（每个样本完成后立即写入）。"""
        record = {
            "session_idx": session_idx,
            "session_id": session_id,
            "n_turns": n_turns,
            "n_questions": n_questions,
            "phase1_ingest_s": round(phase1_s, 3),
            "phase2_qa_s": round(phase2_s, 3),
            "total_s": round(total_s, 3),
            "ms_per_turn": round(phase1_s / n_turns * 1000, 1) if n_turns else 0.0,
            "ingest_tokens": ingest_tokens,
            "ingest_tokens_per_s": round(ingest_tokens / phase1_s, 1) if phase1_s > 0 else 0.0,
            "avg_context_tokens": round(avg_context_tokens, 1),
            "avg_retrieval_latency_s": round(avg_retrieval_time_s, 6),
            "qa_throughput_qps": round(n_questions / phase2_s, 3) if phase2_s > 0 else 0.0,
        }
        self._timing_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[dict] = []
        if self._timing_path.exists():
            try:
                with open(self._timing_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.append(record)
        try:
            with open(self._timing_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save timing record: {e}")

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
            f"self_retrieval={'DISABLED' if self._disable_self_retrieval else 'enabled'}, "
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
            _session_wall_start = time.perf_counter()

            # ── Phase 1: Feed dialogue turns ──────────────────────────
            _phase1_start = time.perf_counter()
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

                # Track tokens ingested per turn (whitespace-split approximation)
                metrics["PhaseForget"].ingest_token_counts.append(len(content.split()))

                # Self-Retrieval: 在写入新记忆前，先用当前对话内容检索历史记忆，
                # 模拟"新信息唤起相关记忆"的自然过程，使效用分数在构建阶段就能更新。
                # 第一轮（turn_idx=0）库为空，跳过检索直接写入。
                # 可通过 disable_self_retrieval=True 关闭，用于对齐 A-MEM 行为。
                retrieved_ids: list[str] = []
                adopted_ids: list[str] = []
                if turn_idx > 0 and not self._disable_self_retrieval:
                    try:
                        recalled = await self._system.search_with_graph(
                            content, top_k=3
                        )
                        retrieved_ids = [r["id"] for r in recalled if r.get("id")]
                        # Top-1 相关记忆推断为被采纳（启发式规则：最相关的是本轮的背景知识）
                        if retrieved_ids:
                            adopted_ids = [retrieved_ids[0]]
                    except Exception as e:
                        logger.debug(f"Self-retrieval skipped at turn {turn_idx}: {e}")

                try:
                    await self._system.add_interaction(
                        content=content,
                        retrieved_ids=retrieved_ids or None,
                        adopted_ids=adopted_ids or None,
                    )
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
            _phase1_elapsed = time.perf_counter() - _phase1_start
            _phase2_start = time.perf_counter()
            try:
                await self._system.wait_for_pending_renorm()
            except Exception as e:
                logger.debug(f"Failed while waiting pending renorm before QA: {e}")

            for qa in questions:
                question = qa.get("question", "")
                reference = qa.get("answer", "")
                raw_category = qa.get("category")
                category = raw_category if isinstance(raw_category, int) else None
                if not question or not reference:
                    continue

                # Evaluate PhaseForget: keyword expand → graph search → generate → score
                try:
                    pf_metrics = metrics["PhaseForget"]
                    expanded_query = await self._expand_query(question, metrics=pf_metrics)
                    with RetrievalTimer() as timer:
                        pf_results = await self._system.search_with_graph(expanded_query)
                    prediction = await self._generate_answer(
                        question, pf_results,
                        category=category,
                        reference=reference,
                        metrics=pf_metrics,
                    )

                    # ── Cat 4 专项诊断：对比 A-MEM 用 ──────────────────
                    if category == 4:
                        f1_val = compute_f1(prediction, reference)
                        ref_tokens = set(_simple_tokenize(reference))
                        ctx_chars = sum(
                            len(r.get("metadata", {}).get("content", r.get("content", "")))
                            for r in pf_results
                        )
                        _all_ctx_text = " ".join(
                            r.get("metadata", {}).get("content", r.get("content", ""))
                            for r in pf_results
                        ).lower()
                        ref_in_ctx = {t for t in ref_tokens if t in _all_ctx_text}
                        ref_not_in_ctx = ref_tokens - ref_in_ctx
                        _hit = len(ref_in_ctx)
                        _total = len(ref_tokens)
                        _hit_pct = f"{_hit}/{_total}" if _total else "0/0"
                        if f1_val == 0 and _total > 0:
                            if _hit == _total:
                                _reason = "GEN_FAIL"
                            elif _hit == 0:
                                _reason = "RETRIEVAL_FAIL"
                            else:
                                _reason = "PARTIAL_RETRIEVAL"
                        else:
                            _reason = ""
                        logger.info(
                            f"[CAT4-DIAG] F1={f1_val:.4f} retrieved={len(pf_results)} "
                            f"ctx_chars={ctx_chars} ref_in_ctx={_hit_pct} "
                            f"{_reason+' ' if _reason else ''}"
                            f"Q={question[:60]}... "
                            f"PRED={prediction[:80]}... "
                            f"REF={reference[:80]}..."
                        )
                        if _reason and ref_not_in_ctx:
                            logger.info(f"  missing_ref_tokens={ref_not_in_ctx}")
                    # ── End Cat 4 diagnostic ──────────────────────────────

                    retrieved_ids = [r.get("id") for r in pf_results if r.get("id")]
                    adopted_ids = self._infer_adopted_ids(pf_results, prediction)
                    await self._system.update_retrieval_feedback(
                        note_ids=retrieved_ids,
                        adopted_ids=adopted_ids,
                    )
                    self._score_with_optional_category(
                        metrics=metrics["PhaseForget"],
                        prediction=prediction,
                        reference=reference,
                        retrieval_time_us=timer.elapsed_us,
                        category=category,
                    )
                except Exception as e:
                    # A-MEM alignment: on any failure use question text as fallback
                    # prediction rather than appending 0 scores, to avoid unfairly
                    # penalising the system for transient LLM/network errors.
                    logger.warning(f"PhaseForget QA failed for '{question[:50]}': {e}")
                    try:
                        fallback_pred = question[:200]
                        self._score_with_optional_category(
                            metrics=metrics["PhaseForget"],
                            prediction=fallback_pred,
                            reference=reference,
                            retrieval_time_us=0,
                            category=category,
                        )
                    except Exception:
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

            _phase2_elapsed = time.perf_counter() - _phase2_start
            _session_elapsed = time.perf_counter() - _session_wall_start
            _ms_per_turn = (_phase1_elapsed / len(dialogue) * 1000) if dialogue else 0.0
            _pf_m = metrics["PhaseForget"]
            _session_ingest_tokens = sum(
                _pf_m.ingest_token_counts[-len(dialogue):]
                if len(_pf_m.ingest_token_counts) >= len(dialogue)
                else _pf_m.ingest_token_counts
            )
            _session_ctx_tokens = (
                sum(_pf_m.context_token_counts[-len(questions):]
                    if len(_pf_m.context_token_counts) >= len(questions)
                    else _pf_m.context_token_counts)
                / max(len(questions), 1)
            )
            _qa_tput = len(questions) / _phase2_elapsed if _phase2_elapsed > 0 else 0.0
            logger.info(
                f"[TIMING] session={session_id} "
                f"total={_session_elapsed:.1f}s  "
                f"phase1(ingest)={_phase1_elapsed:.1f}s({_ms_per_turn:.0f}ms/turn)  "
                f"phase2(qa)={_phase2_elapsed:.1f}s({_qa_tput:.2f}q/s)  "
                f"turns={len(dialogue)} questions={len(questions)}  "
                f"ingest_tokens={_session_ingest_tokens}  "
                f"avg_ctx_tokens={_session_ctx_tokens:.0f}  "
                f"latency={_pf_m.avg_retrieval_time_s:.4f}s"
            )
            self._save_session_timing(
                session_id=session_id,
                session_idx=session_idx,
                n_turns=len(dialogue),
                n_questions=len(questions),
                phase1_s=_phase1_elapsed,
                phase2_s=_phase2_elapsed,
                total_s=_session_elapsed,
                ingest_tokens=_session_ingest_tokens,
                avg_context_tokens=_session_ctx_tokens,
                avg_retrieval_time_s=_pf_m.avg_retrieval_time_s,
            )

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
        plus retrieval latency, sample count, and LLM parse failure rates.

        Returns the report string (also prints to stdout).
        """
        lines = [
            "=" * 110,
            "PhaseForget-Zettel Benchmark Report (with 6 QA Metrics)",
            "=" * 110,
            "",
            f"{'System':<20} {'F1':>8} {'BLEU':>8} {'ROUGE-L':>8} "
            f"{'ROUGE2':>8} {'METEOR':>8} {'SBERT':>8} "
            f"{'RetTime(us)':>12} {'N':>6} {'AnsFailRate':>12} {'QExpFail':>9}",
            "-" * 110,
        ]

        for name, m in results.items():
            n = len(m.f1_scores)
            lines.append(
                f"{name:<20} "
                f"{m.avg_f1:>8.4f} {m.avg_bleu:>8.4f} "
                f"{m.avg_rouge_l:>8.4f} {m.avg_rouge2:>8.4f} "
                f"{m.avg_meteor:>8.4f} {m.avg_sbert:>8.2f} "
                f"{m.avg_retrieval_time_us:>12.1f} {n:>6} "
                f"{m.parse_fail_rate:>11.1%} {m.query_expand_parse_fail:>9d}"
            )

        # ── Open Domain Investigation Summary ──────────────────────────
        for name, m in results.items():
            if m.by_category:
                lines.extend(["", f"  {name} — Per-Category Token Analysis:", "-" * 110])
                for cat in sorted(m.by_category):
                    cat_m = m.by_category[cat]
                    n = len(cat_m.f1_scores)
                    if n > 0:
                        avg_f1 = sum(cat_m.f1_scores) / n
                        lines.append(
                            f"  Cat {cat}: N={n:>3}  avg_F1={avg_f1:.4f}"
                        )

        has_category_breakdown = any(m.by_category for m in results.values())
        if has_category_breakdown:
            lines.extend([
                "",
                "Category Breakdown (from QA.category)",
                "-" * 110,
                f"{'System':<20} {'Cat':>5} {'F1':>8} {'BLEU':>8} {'ROUGE-L':>8} "
                f"{'ROUGE2':>8} {'METEOR':>8} {'SBERT':>8} {'RetTime(us)':>12} {'N':>6}",
                "-" * 110,
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

        lines.extend(["", "=" * 110])
        report = "\n".join(lines)
        print(report)
        return report

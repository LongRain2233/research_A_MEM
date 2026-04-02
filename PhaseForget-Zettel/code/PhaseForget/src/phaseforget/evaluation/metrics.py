"""
Evaluation metrics for PhaseForget-Zettel.

Aligns with Research Design §7.2:
    - QA quality:    F1, BLEU-1, ROUGE-L, ROUGE-2, METEOR, SBERT Similarity
    - System efficiency: Memory Usage (MB), Retrieval Time (us)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EvalMetrics:
    """Container for evaluation metrics across a benchmark run."""
    f1_scores: list[float] = field(default_factory=list)
    bleu_scores: list[float] = field(default_factory=list)
    rouge_l_scores: list[float] = field(default_factory=list)
    rouge2_scores: list[float] = field(default_factory=list)
    meteor_scores: list[float] = field(default_factory=list)
    sbert_scores: list[float] = field(default_factory=list)
    retrieval_times_us: list[float] = field(default_factory=list)
    memory_usage_mb: list[float] = field(default_factory=list)
    by_category: dict[int, "EvalMetrics"] = field(default_factory=dict, repr=False)
    # LLM parse failure counters — help diagnose model output quality issues
    query_expand_parse_fail: int = 0
    answer_parse_fail: int = 0

    @property
    def avg_f1(self) -> float:
        return sum(self.f1_scores) / len(self.f1_scores) if self.f1_scores else 0.0

    @property
    def avg_bleu(self) -> float:
        return sum(self.bleu_scores) / len(self.bleu_scores) if self.bleu_scores else 0.0

    @property
    def avg_rouge_l(self) -> float:
        return sum(self.rouge_l_scores) / len(self.rouge_l_scores) if self.rouge_l_scores else 0.0

    @property
    def avg_rouge2(self) -> float:
        return sum(self.rouge2_scores) / len(self.rouge2_scores) if self.rouge2_scores else 0.0

    @property
    def avg_meteor(self) -> float:
        return sum(self.meteor_scores) / len(self.meteor_scores) if self.meteor_scores else 0.0

    @property
    def avg_sbert(self) -> float:
        return sum(self.sbert_scores) / len(self.sbert_scores) if self.sbert_scores else 0.0

    @property
    def avg_retrieval_time_us(self) -> float:
        return (
            sum(self.retrieval_times_us) / len(self.retrieval_times_us)
            if self.retrieval_times_us
            else 0.0
        )

    def category_metrics(self, category: int) -> "EvalMetrics":
        """Get or create metric bucket for a QA category."""
        if category not in self.by_category:
            self.by_category[category] = EvalMetrics()
        return self.by_category[category]

    @property
    def parse_fail_rate(self) -> float:
        """Fraction of QA answer calls that failed to parse, over total scored samples."""
        n = len(self.f1_scores)
        return self.answer_parse_fail / n if n > 0 else 0.0

    def to_dict(self, include_categories: bool = True) -> dict:
        """Serialize metrics for checkpoint persistence."""
        data = {
            "f1_scores": self.f1_scores,
            "bleu_scores": self.bleu_scores,
            "rouge_l_scores": self.rouge_l_scores,
            "rouge2_scores": self.rouge2_scores,
            "meteor_scores": self.meteor_scores,
            "sbert_scores": self.sbert_scores,
            "retrieval_times_us": self.retrieval_times_us,
            "memory_usage_mb": self.memory_usage_mb,
            "query_expand_parse_fail": self.query_expand_parse_fail,
            "answer_parse_fail": self.answer_parse_fail,
        }
        if include_categories and self.by_category:
            data["by_category"] = {
                str(cat): m.to_dict(include_categories=False)
                for cat, m in sorted(self.by_category.items())
            }
        return data

    @classmethod
    def from_dict(cls, raw: dict) -> "EvalMetrics":
        """Restore metrics from checkpoint payload."""
        m = cls()
        m.f1_scores = raw.get("f1_scores", [])
        m.bleu_scores = raw.get("bleu_scores", [])
        m.rouge_l_scores = raw.get("rouge_l_scores", [])
        m.rouge2_scores = raw.get("rouge2_scores", [])
        m.meteor_scores = raw.get("meteor_scores", [])
        m.sbert_scores = raw.get("sbert_scores", [])
        m.retrieval_times_us = raw.get("retrieval_times_us", [])
        m.memory_usage_mb = raw.get("memory_usage_mb", [])
        m.query_expand_parse_fail = raw.get("query_expand_parse_fail", 0)
        m.answer_parse_fail = raw.get("answer_parse_fail", 0)

        for cat_raw, cat_data in raw.get("by_category", {}).items():
            try:
                cat = int(cat_raw)
            except (TypeError, ValueError):
                continue
            m.by_category[cat] = cls.from_dict(cat_data)
        return m


def _simple_tokenize(text: str) -> list[str]:
    """Tokenize matching A-MEM's simple_tokenize: lowercase + strip punctuation."""
    text = str(text).lower()
    for ch in (".", ",", "!", "?"):
        text = text.replace(ch, " ")
    return text.split()


def compute_f1(prediction: str, reference: str) -> float:
    """
    Set-based token-level F1 score (aligned with A-MEM utils.py).

    Uses set() intersection (not Counter) so duplicate tokens are
    collapsed, and strips common punctuation before splitting.
    """
    pred_tokens = set(_simple_tokenize(prediction))
    ref_tokens = set(_simple_tokenize(reference))

    if not pred_tokens or not ref_tokens:
        return 0.0

    common_tokens = pred_tokens & ref_tokens

    if not common_tokens:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_bleu1(prediction: str, reference: str) -> float:
    """
    BLEU-1 score using nltk sentence_bleu with smoothing (aligned with A-MEM).

    A-MEM uses nltk.translate.bleu_score.sentence_bleu with
    SmoothingFunction().method1 and unigram weights (1,0,0,0).
    """
    try:
        import nltk
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

        pred_tokens = nltk.word_tokenize(prediction.lower())
        ref_tokens = [nltk.word_tokenize(reference.lower())]

        if not pred_tokens or not ref_tokens[0]:
            return 0.0

        return float(
            sentence_bleu(
                ref_tokens,
                pred_tokens,
                weights=(1, 0, 0, 0),
                smoothing_function=SmoothingFunction().method1,
            )
        )
    except Exception:
        return 0.0


def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score based on longest common subsequence (stemmer=True per A-MEM)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure
    except Exception:
        return 0.0


def compute_rouge2(prediction: str, reference: str) -> float:
    """ROUGE-2 F1 score based on bigram overlap (stemmer=True per A-MEM)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores["rouge2"].fmeasure
    except Exception:
        return 0.0


def compute_meteor(prediction: str, reference: str) -> float:
    """METEOR score considering synonyms and paraphrases."""
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        # Ensure wordnet data is available (downloaded once, silently)
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()
        if not pred_tokens or not ref_tokens:
            return 0.0
        return float(meteor_score([ref_tokens], pred_tokens))
    except Exception:
        return 0.0


def compute_sbert(prediction: str, reference: str, model) -> float:
    """
    SBERT cosine similarity in 0-1 range (aligned with A-MEM).

    Args:
        model: A loaded SentenceTransformer model (passed in to avoid repeated loading).
    """
    try:
        from sentence_transformers import util
        emb1 = model.encode(prediction, convert_to_tensor=True)
        emb2 = model.encode(reference, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2).item())
    except Exception:
        return 0.0


class RetrievalTimer:
    """Context manager for measuring retrieval latency in microseconds."""

    def __init__(self):
        self.elapsed_us: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_us = (time.perf_counter() - self._start) * 1e6

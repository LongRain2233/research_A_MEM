"""
Unit tests for evaluation metrics.
"""

from phaseforget.evaluation.metrics import (
    EvalMetrics,
    RetrievalTimer,
    compute_bleu1,
    compute_f1,
)
from phaseforget.evaluation.benchmark import BenchmarkRunner


class TestF1Score:

    def test_perfect_match(self):
        assert compute_f1("hello world", "hello world") == 1.0

    def test_partial_overlap(self):
        f1 = compute_f1("hello world foo", "hello world bar")
        assert 0 < f1 < 1

    def test_no_overlap(self):
        assert compute_f1("hello", "world") == 0.0

    def test_empty_strings(self):
        assert compute_f1("", "hello") == 0.0
        assert compute_f1("hello", "") == 0.0

    def test_case_insensitive(self):
        assert compute_f1("Hello World", "hello world") == 1.0


class TestBLEU1:

    def test_perfect_match(self):
        assert compute_bleu1("the cat sat", "the cat sat") == 1.0

    def test_partial_match(self):
        b = compute_bleu1("the cat sat on mat", "the cat sat")
        assert 0 < b < 1

    def test_no_match(self):
        assert compute_bleu1("hello", "world") == 0.0

    def test_empty(self):
        assert compute_bleu1("", "something") == 0.0


class TestRetrievalTimer:

    def test_timer_measures_elapsed(self):
        with RetrievalTimer() as timer:
            _ = sum(range(1000))
        assert timer.elapsed_us > 0


class TestEvalMetrics:

    def test_avg_f1(self):
        m = EvalMetrics(f1_scores=[0.5, 0.8, 0.6])
        assert abs(m.avg_f1 - 0.6333) < 0.01

    def test_empty_metrics(self):
        m = EvalMetrics()
        assert m.avg_f1 == 0.0
        assert m.avg_bleu == 0.0
        assert m.avg_retrieval_time_us == 0.0

    def test_category_bucket_create_and_reuse(self):
        m = EvalMetrics()
        c1 = m.category_metrics(1)
        c1.f1_scores.append(0.9)
        assert m.category_metrics(1).avg_f1 == 0.9
        assert 1 in m.by_category

    def test_to_dict_and_from_dict_with_categories(self):
        m = EvalMetrics(
            f1_scores=[0.7],
            bleu_scores=[0.6],
            retrieval_times_us=[100.0],
        )
        c2 = m.category_metrics(2)
        c2.f1_scores.append(0.5)
        c2.bleu_scores.append(0.4)
        c2.retrieval_times_us.append(80.0)

        raw = m.to_dict()
        restored = EvalMetrics.from_dict(raw)

        assert restored.avg_f1 == 0.7
        assert 2 in restored.by_category
        assert restored.by_category[2].avg_f1 == 0.5
        assert restored.by_category[2].avg_bleu == 0.4

    def test_print_report_includes_category_breakdown(self):
        m = EvalMetrics(f1_scores=[1.0], bleu_scores=[1.0], retrieval_times_us=[50.0])
        c1 = m.category_metrics(1)
        c1.f1_scores.append(1.0)
        c1.bleu_scores.append(1.0)
        c1.retrieval_times_us.append(50.0)

        report = BenchmarkRunner.print_report({"PhaseForget": m})
        assert "Category Breakdown" in report
        assert "PhaseForget" in report

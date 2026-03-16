"""
Unit tests for evaluation metrics.
"""

from phaseforget.evaluation.metrics import (
    EvalMetrics,
    RetrievalTimer,
    compute_bleu1,
    compute_f1,
)


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

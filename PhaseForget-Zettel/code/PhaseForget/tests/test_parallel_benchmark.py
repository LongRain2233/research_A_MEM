from phaseforget.utils.parallel_benchmark import (
    aggregate_run_summaries,
    choose_recommendation,
    parse_parallel_values,
    summarize_gpu_samples,
)


def test_parse_parallel_values_sorts_and_deduplicates():
    assert parse_parallel_values("4,2,4,1") == [1, 2, 4]


def test_summarize_gpu_samples_extracts_peak_ratio():
    summary = summarize_gpu_samples(
        [
            {
                "gpu_util": 40.0,
                "mem_util": 20.0,
                "memory_used_mb": 2000.0,
                "memory_used_ratio": 0.25,
                "temperature_c": 60.0,
                "power_w": 80.0,
            },
            {
                "gpu_util": 90.0,
                "mem_util": 35.0,
                "memory_used_mb": 5000.0,
                "memory_used_ratio": 0.625,
                "temperature_c": 72.0,
                "power_w": 110.0,
            },
        ]
    )

    assert summary["gpu_util_avg"] == 65.0
    assert summary["vram_peak_mb"] == 5000.0
    assert summary["vram_peak_ratio"] == 0.625
    assert summary["temp_peak_c"] == 72.0


def test_choose_recommendation_prefers_smallest_within_tolerance():
    rows = aggregate_run_summaries(
        [
            {
                "parallel": 1,
                "throughput_trials_per_hour": 10.0,
                "error_rate": 0.0,
                "elapsed_seconds": 360.0,
                "successful_trials": 1,
                "failed_trials": 0,
                "vram_peak_ratio": 0.30,
                "gpu_util_avg": 20.0,
            },
            {
                "parallel": 2,
                "throughput_trials_per_hour": 18.2,
                "error_rate": 0.0,
                "elapsed_seconds": 200.0,
                "successful_trials": 1,
                "failed_trials": 0,
                "vram_peak_ratio": 0.50,
                "gpu_util_avg": 45.0,
            },
            {
                "parallel": 4,
                "throughput_trials_per_hour": 19.0,
                "error_rate": 0.0,
                "elapsed_seconds": 190.0,
                "successful_trials": 1,
                "failed_trials": 0,
                "vram_peak_ratio": 0.65,
                "gpu_util_avg": 70.0,
            },
        ]
    )

    recommendation = choose_recommendation(rows, throughput_tolerance_pct=5.0)

    assert recommendation is not None
    assert recommendation["best_parallel"] == 4
    assert recommendation["recommended_parallel"] == 2


def test_choose_recommendation_filters_out_invalid_rows():
    rows = aggregate_run_summaries(
        [
            {
                "parallel": 1,
                "throughput_trials_per_hour": 10.0,
                "error_rate": 0.0,
                "elapsed_seconds": 360.0,
                "successful_trials": 1,
                "failed_trials": 0,
                "vram_peak_ratio": 0.30,
                "gpu_util_avg": 20.0,
            },
            {
                "parallel": 2,
                "throughput_trials_per_hour": 22.0,
                "error_rate": 0.5,
                "elapsed_seconds": 170.0,
                "successful_trials": 1,
                "failed_trials": 1,
                "vram_peak_ratio": 0.40,
                "gpu_util_avg": 60.0,
            },
            {
                "parallel": 4,
                "throughput_trials_per_hour": 20.0,
                "error_rate": 0.0,
                "elapsed_seconds": 180.0,
                "successful_trials": 1,
                "failed_trials": 0,
                "vram_peak_ratio": 0.95,
                "gpu_util_avg": 80.0,
            },
        ]
    )

    recommendation = choose_recommendation(
        rows,
        throughput_tolerance_pct=5.0,
        max_error_rate=0.0,
        max_vram_utilization=0.90,
    )

    assert recommendation is not None
    assert recommendation["recommended_parallel"] == 1

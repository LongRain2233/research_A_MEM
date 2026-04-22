from __future__ import annotations

import math
import statistics
from typing import Any


def parse_parallel_values(raw: str) -> list[int]:
    values = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    if not values:
        raise ValueError("parallel values cannot be empty")
    if any(value < 1 for value in values):
        raise ValueError("parallel values must be >= 1")
    return values


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return ordered[low]
    weight = idx - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _stdev(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else 0.0 if values else None


def summarize_gpu_samples(samples: list[dict[str, float]]) -> dict[str, float | int | None]:
    gpu_utils = [sample["gpu_util"] for sample in samples if sample.get("gpu_util") is not None]
    mem_utils = [sample["mem_util"] for sample in samples if sample.get("mem_util") is not None]
    vram_used = [sample["memory_used_mb"] for sample in samples if sample.get("memory_used_mb") is not None]
    vram_ratios = [sample["memory_used_ratio"] for sample in samples if sample.get("memory_used_ratio") is not None]
    temps = [sample["temperature_c"] for sample in samples if sample.get("temperature_c") is not None]
    powers = [sample["power_w"] for sample in samples if sample.get("power_w") is not None]

    return {
        "gpu_sample_count": len(samples),
        "gpu_util_avg": _mean(gpu_utils),
        "gpu_util_p95": percentile(gpu_utils, 0.95),
        "mem_util_avg": _mean(mem_utils),
        "mem_util_p95": percentile(mem_utils, 0.95),
        "vram_peak_mb": max(vram_used) if vram_used else None,
        "vram_peak_ratio": max(vram_ratios) if vram_ratios else None,
        "temp_peak_c": max(temps) if temps else None,
        "power_avg_w": _mean(powers),
    }


def summarize_run(
    *,
    parallel: int,
    repeat_index: int,
    elapsed_seconds: float,
    results: list[dict[str, Any]],
    return_code: int,
    gpu_samples: list[dict[str, float]],
) -> dict[str, Any]:
    total_trials = len(results)
    failed_trials = sum(1 for item in results if "error" in item.get("metrics", {}))
    successful_trials = total_trials - failed_trials
    throughput = (successful_trials / elapsed_seconds * 3600.0) if elapsed_seconds > 0 else 0.0

    trial_elapsed_values = [
        float(item.get("elapsed_seconds", 0.0))
        for item in results
        if isinstance(item.get("elapsed_seconds"), (float, int))
    ]
    gpu_summary = summarize_gpu_samples(gpu_samples)

    summary = {
        "parallel": parallel,
        "repeat_index": repeat_index,
        "elapsed_seconds": elapsed_seconds,
        "return_code": return_code,
        "total_trials": total_trials,
        "successful_trials": successful_trials,
        "failed_trials": failed_trials,
        "error_rate": (failed_trials / total_trials) if total_trials else 0.0,
        "throughput_trials_per_hour": throughput,
        "mean_trial_seconds": _mean(trial_elapsed_values),
        "median_trial_seconds": statistics.median(trial_elapsed_values) if trial_elapsed_values else None,
    }
    summary.update(gpu_summary)
    return summary


def aggregate_run_summaries(run_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in run_summaries:
        grouped.setdefault(int(item["parallel"]), []).append(item)

    rows: list[dict[str, Any]] = []
    for parallel in sorted(grouped):
        group = grouped[parallel]
        throughput_values = [float(item["throughput_trials_per_hour"]) for item in group]
        error_rates = [float(item["error_rate"]) for item in group]
        elapsed_values = [float(item["elapsed_seconds"]) for item in group]
        vram_peak_ratios = [
            float(item["vram_peak_ratio"])
            for item in group
            if item.get("vram_peak_ratio") is not None
        ]
        gpu_utils = [
            float(item["gpu_util_avg"])
            for item in group
            if item.get("gpu_util_avg") is not None
        ]

        rows.append(
            {
                "parallel": parallel,
                "repeats": len(group),
                "throughput_mean": _mean(throughput_values),
                "throughput_stdev": _stdev(throughput_values),
                "elapsed_mean_seconds": _mean(elapsed_values),
                "error_rate_mean": _mean(error_rates),
                "error_rate_max": max(error_rates) if error_rates else None,
                "successful_trials_mean": _mean([float(item["successful_trials"]) for item in group]),
                "failed_trials_mean": _mean([float(item["failed_trials"]) for item in group]),
                "vram_peak_ratio_max": max(vram_peak_ratios) if vram_peak_ratios else None,
                "gpu_util_avg_mean": _mean(gpu_utils),
            }
        )

    baseline = None
    if rows:
        baseline = rows[0]["throughput_mean"]

    for row in rows:
        throughput = row["throughput_mean"]
        if baseline and throughput is not None and baseline > 0:
            speedup = throughput / baseline
            row["speedup_vs_parallel_1"] = speedup
            row["efficiency_vs_parallel_1"] = speedup / row["parallel"]
        else:
            row["speedup_vs_parallel_1"] = None
            row["efficiency_vs_parallel_1"] = None

    return rows


def choose_recommendation(
    rows: list[dict[str, Any]],
    *,
    throughput_tolerance_pct: float = 5.0,
    max_error_rate: float = 0.0,
    max_vram_utilization: float = 0.90,
) -> dict[str, Any] | None:
    valid_rows = []
    for row in rows:
        error_rate = row.get("error_rate_max")
        vram_ratio = row.get("vram_peak_ratio_max")
        if error_rate is not None and error_rate > max_error_rate:
            continue
        if vram_ratio is not None and vram_ratio > max_vram_utilization:
            continue
        valid_rows.append(row)

    if not valid_rows:
        return None

    best_row = max(valid_rows, key=lambda item: item["throughput_mean"] or 0.0)
    best_throughput = best_row["throughput_mean"] or 0.0
    floor_throughput = best_throughput * (1.0 - throughput_tolerance_pct / 100.0)

    candidates = [
        row for row in valid_rows
        if (row["throughput_mean"] or 0.0) >= floor_throughput
    ]
    recommended = min(candidates, key=lambda item: item["parallel"]) if candidates else best_row

    return {
        "recommended_parallel": recommended["parallel"],
        "recommended_row": recommended,
        "best_parallel": best_row["parallel"],
        "best_row": best_row,
        "throughput_floor": floor_throughput,
        "throughput_tolerance_pct": throughput_tolerance_pct,
        "max_error_rate": max_error_rate,
        "max_vram_utilization": max_vram_utilization,
    }

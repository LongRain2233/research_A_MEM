"""
并发 benchmark 工具：自动 sweep 不同 --max-parallel，采集吞吐与 GPU 指标，
并给出可迁移到其他 GPU 的推荐并发值。

示例：
    python benchmark_parallelism.py \
      --parallel-values 1,2,4,6 \
      --repeats 2 \
      --warmup 1 \
      --output data/parallel_bench/full_compare_report.json \
      -- \
      --theta-sim-values 0.75,0.8 \
      --theta-sum-values 10,3,5,9999 \
      --decay-interval-values 50 \
      --theta-evict-values 0.35
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

from phaseforget.utils.parallel_benchmark import (  # noqa: E402
    aggregate_run_summaries,
    choose_recommendation,
    parse_parallel_values,
    summarize_run,
)


def _resolve_path(raw: str | None, default_name: str) -> Path:
    if raw:
        path = Path(raw)
    else:
        path = Path(default_name)
    if not path.is_absolute():
        path = _ROOT / path
    return path


def _strip_remainder_prefix(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def _validate_forwarded_args(args: list[str]) -> None:
    blocked_flags = {
        "--max-parallel",
        "--output",
        "--show-results",
        "--clear-results",
    }
    for flag in blocked_flags:
        if flag in args:
            raise ValueError(f"请不要在转发参数中包含 {flag}，benchmark 脚本会自动注入。")

    for idx, arg in enumerate(args):
        if arg == "--search-type" and idx + 1 < len(args) and args[idx + 1] == "bayesian":
            raise ValueError("bayesian 模式不会使用 --max-parallel，不能用于并发 sweep。")


def _load_result_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _parse_float(raw: str) -> float | None:
    raw = raw.strip()
    if not raw or raw in {"-", "N/A", "[N/A]", "Not Supported", "[Not Supported]"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_gpu_samples(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []

    samples: list[dict[str, float]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 9:
                continue

            gpu_util = _parse_float(parts[2])
            mem_util = _parse_float(parts[3])
            memory_used_mb = _parse_float(parts[4])
            memory_total_mb = _parse_float(parts[5])
            temperature_c = _parse_float(parts[6])
            power_w = _parse_float(parts[7])
            sm_clock_mhz = _parse_float(parts[8])
            mem_clock_mhz = _parse_float(parts[9]) if len(parts) > 9 else None

            memory_used_ratio = None
            if memory_used_mb is not None and memory_total_mb not in (None, 0):
                memory_used_ratio = memory_used_mb / memory_total_mb

            samples.append(
                {
                    "gpu_util": gpu_util,
                    "mem_util": mem_util,
                    "memory_used_mb": memory_used_mb,
                    "memory_total_mb": memory_total_mb,
                    "memory_used_ratio": memory_used_ratio,
                    "temperature_c": temperature_c,
                    "power_w": power_w,
                    "sm_clock_mhz": sm_clock_mhz,
                    "mem_clock_mhz": mem_clock_mhz,
                }
            )
    return samples


def _start_gpu_monitor(
    *,
    gpu_index: int,
    sample_interval_sec: int,
    output_path: Path,
) -> tuple[subprocess.Popen[str] | None, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None, None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(output_path, "w", encoding="utf-8")
    command = [
        nvidia_smi,
        "-i",
        str(gpu_index),
        "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.sm,clocks.mem",
        "--format=csv,noheader,nounits",
        "-l",
        str(sample_interval_sec),
    ]
    process = subprocess.Popen(
        command,
        stdout=handle,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=str(_ROOT),
    )
    return process, handle


def _stop_gpu_monitor(process: subprocess.Popen[str] | None, handle: Any) -> None:
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    if handle is not None:
        handle.close()


def _print_run_summary(summary: dict[str, Any]) -> None:
    throughput = summary["throughput_trials_per_hour"]
    error_rate = summary["error_rate"] * 100.0
    vram_ratio = summary.get("vram_peak_ratio")
    vram_text = f"{vram_ratio * 100:.1f}%" if vram_ratio is not None else "N/A"
    gpu_util = summary.get("gpu_util_avg")
    gpu_text = f"{gpu_util:.1f}%" if gpu_util is not None else "N/A"
    print(
        f"    完成: {summary['successful_trials']}/{summary['total_trials']} trials | "
        f"耗时 {summary['elapsed_seconds']:.1f}s | "
        f"吞吐 {throughput:.2f} trials/h | "
        f"错误率 {error_rate:.1f}% | "
        f"GPU均值 {gpu_text} | VRAM峰值 {vram_text}"
    )


def _run_single_case(
    *,
    python_exe: str,
    target_script: Path,
    forwarded_args: list[str],
    parallel: int,
    repeat_index: int,
    raw_dir: Path,
    gpu_index: int,
    gpu_sample_interval_sec: int,
    warmup: bool,
) -> dict[str, Any]:
    run_tag = f"p{parallel:02d}_{'warmup' if warmup else f'r{repeat_index:02d}'}"
    run_dir = raw_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.json"
    log_path = run_dir / "run.log"
    gpu_log_path = run_dir / "gpu.csv"

    command = [
        python_exe,
        str(target_script),
        *forwarded_args,
        "--max-parallel",
        str(parallel),
        "--output",
        str(results_path),
    ]

    print(
        f"  {'预热' if warmup else '正式'}运行: parallel={parallel} "
        f"{'' if warmup else f'repeat={repeat_index}'}"
    )
    start = time.perf_counter()
    monitor_process, monitor_handle = _start_gpu_monitor(
        gpu_index=gpu_index,
        sample_interval_sec=gpu_sample_interval_sec,
        output_path=gpu_log_path,
    )
    try:
        with open(log_path, "w", encoding="utf-8") as log_handle:
            completed = subprocess.run(
                command,
                cwd=str(_ROOT),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
    finally:
        _stop_gpu_monitor(monitor_process, monitor_handle)

    elapsed_seconds = time.perf_counter() - start
    results = _load_result_items(results_path)
    gpu_samples = _parse_gpu_samples(gpu_log_path)
    summary = summarize_run(
        parallel=parallel,
        repeat_index=repeat_index,
        elapsed_seconds=elapsed_seconds,
        results=results,
        return_code=completed.returncode,
        gpu_samples=gpu_samples,
    )
    summary.update(
        {
            "warmup": warmup,
            "command": command,
            "results_path": str(results_path),
            "log_path": str(log_path),
            "gpu_log_path": str(gpu_log_path),
        }
    )
    _print_run_summary(summary)
    return summary


def _print_aggregate_table(rows: list[dict[str, Any]]) -> None:
    print("\n汇总结果")
    print("-" * 96)
    print(
        f"{'并发':>4} {'重复':>4} {'吞吐均值':>12} {'吞吐方差':>12} "
        f"{'加速比':>8} {'效率':>8} {'错误率':>8} {'VRAM峰值':>10} {'GPU均值':>10}"
    )
    print("-" * 96)
    for row in rows:
        throughput_mean = row.get("throughput_mean")
        throughput_stdev = row.get("throughput_stdev")
        speedup = row.get("speedup_vs_parallel_1")
        efficiency = row.get("efficiency_vs_parallel_1")
        error_rate = row.get("error_rate_max")
        vram_peak = row.get("vram_peak_ratio_max")
        gpu_util = row.get("gpu_util_avg_mean")
        print(
            f"{row['parallel']:>4} {row['repeats']:>4} "
            f"{throughput_mean:>12.2f} {throughput_stdev:>12.2f} "
            f"{speedup:>8.2f} {efficiency:>8.2f} "
            f"{(error_rate * 100.0 if error_rate is not None else 0.0):>7.1f}% "
            f"{(f'{vram_peak * 100:.1f}%' if vram_peak is not None else 'N/A'):>10} "
            f"{(f'{gpu_util:.1f}%' if gpu_util is not None else 'N/A'):>10}"
        )
    print("-" * 96)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="自动测试不同并发档位，并给出推荐的 --max-parallel。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parallel-values",
        required=True,
        help="要测试的并发档位，例如 1,2,4,6",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="每个并发档位的正式重复次数（默认: 2）",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="每个并发档位正式测量前的预热次数（默认: 1）",
    )
    parser.add_argument(
        "--target-script",
        default="hyperparameter_search.py",
        help="被测脚本路径，默认是 hyperparameter_search.py",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="用于启动被测脚本的 Python 解释器",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="使用 nvidia-smi 采集时的 GPU 编号（默认: 0）",
    )
    parser.add_argument(
        "--gpu-sample-interval",
        type=int,
        default=1,
        help="GPU 采样间隔，单位秒（默认: 1）",
    )
    parser.add_argument(
        "--throughput-tolerance-pct",
        type=float,
        default=5.0,
        help="在最佳吞吐的多少百分比内，优先选更小并发（默认: 5%%）",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=0.0,
        help="允许的最大错误率（默认: 0.0）",
    )
    parser.add_argument(
        "--max-vram-utilization",
        type=float,
        default=0.90,
        help="允许的最大显存占用比例（默认: 0.90）",
    )
    parser.add_argument(
        "--output",
        default="data/parallel_bench/report.json",
        help="汇总报告输出路径",
    )
    parser.add_argument(
        "forwarded_args",
        nargs=argparse.REMAINDER,
        help="传给 hyperparameter_search.py 的参数，前面用 -- 隔开",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    parallel_values = parse_parallel_values(args.parallel_values)
    forwarded_args = _strip_remainder_prefix(list(args.forwarded_args))
    _validate_forwarded_args(forwarded_args)

    target_script = _resolve_path(args.target_script, "hyperparameter_search.py")
    if not target_script.exists():
        raise FileNotFoundError(f"找不到被测脚本: {target_script}")

    output_path = _resolve_path(args.output, "data/parallel_bench/report.json")
    raw_dir = output_path.parent / f"{output_path.stem}_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PhaseForget 并发 benchmark")
    print("=" * 72)
    print(f"并发档位       : {parallel_values}")
    print(f"正式重复次数   : {args.repeats}")
    print(f"预热次数       : {args.warmup}")
    print(f"目标脚本       : {target_script}")
    print(f"汇总报告       : {output_path}")
    print(f"原始日志目录   : {raw_dir}")
    print(f"转发参数       : {forwarded_args}")
    print("=" * 72)

    measured_runs: list[dict[str, Any]] = []

    for parallel in parallel_values:
        print(f"\n[并发 {parallel}]")
        for warmup_idx in range(1, args.warmup + 1):
            _run_single_case(
                python_exe=args.python_exe,
                target_script=target_script,
                forwarded_args=forwarded_args,
                parallel=parallel,
                repeat_index=warmup_idx,
                raw_dir=raw_dir,
                gpu_index=args.gpu_index,
                gpu_sample_interval_sec=max(1, args.gpu_sample_interval),
                warmup=True,
            )

        for repeat_index in range(1, args.repeats + 1):
            summary = _run_single_case(
                python_exe=args.python_exe,
                target_script=target_script,
                forwarded_args=forwarded_args,
                parallel=parallel,
                repeat_index=repeat_index,
                raw_dir=raw_dir,
                gpu_index=args.gpu_index,
                gpu_sample_interval_sec=max(1, args.gpu_sample_interval),
                warmup=False,
            )
            measured_runs.append(summary)

    rows = aggregate_run_summaries(measured_runs)
    recommendation = choose_recommendation(
        rows,
        throughput_tolerance_pct=args.throughput_tolerance_pct,
        max_error_rate=args.max_error_rate,
        max_vram_utilization=args.max_vram_utilization,
    )

    _print_aggregate_table(rows)

    report = {
        "created_at_epoch": time.time(),
        "config": {
            "parallel_values": parallel_values,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "target_script": str(target_script),
            "python_exe": args.python_exe,
            "gpu_index": args.gpu_index,
            "gpu_sample_interval": args.gpu_sample_interval,
            "throughput_tolerance_pct": args.throughput_tolerance_pct,
            "max_error_rate": args.max_error_rate,
            "max_vram_utilization": args.max_vram_utilization,
            "forwarded_args": forwarded_args,
        },
        "runs": measured_runs,
        "summary": rows,
        "recommendation": recommendation,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    if recommendation is None:
        print("\n没有找到满足阈值的并发档位，请放宽阈值或查看原始日志。")
    else:
        recommended = recommendation["recommended_row"]
        best = recommendation["best_row"]
        print("\n推荐结论")
        print("-" * 72)
        print(
            f"推荐并发       : {recommendation['recommended_parallel']}  "
            f"(在最佳吞吐 {best['throughput_mean']:.2f} trials/h 的 "
            f"{100 - args.throughput_tolerance_pct:.1f}% 以上)"
        )
        print(
            f"对应吞吐       : {recommended['throughput_mean']:.2f} trials/h | "
            f"错误率上限 {recommended['error_rate_max'] * 100:.1f}% | "
            f"VRAM峰值 {recommended['vram_peak_ratio_max'] * 100:.1f}%"
            if recommended.get("vram_peak_ratio_max") is not None
            else (
                f"对应吞吐       : {recommended['throughput_mean']:.2f} trials/h | "
                f"错误率上限 {recommended['error_rate_max'] * 100:.1f}% | "
                "VRAM峰值 N/A"
            )
        )
        print(f"最佳吞吐并发   : {recommendation['best_parallel']}")
        print("-" * 72)

    print(f"\n完整报告已保存到: {output_path}")


if __name__ == "__main__":
    main()

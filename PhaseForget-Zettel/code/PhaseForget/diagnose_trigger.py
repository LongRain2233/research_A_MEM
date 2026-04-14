"""
触发诊断脚本（并行版）
=====================
验证遗忘机制在不同参数组合下是否能被激活。

共 30 组参数：
  - 1 组对照（原始参数，已知从不触发）
  - 20 组主网格（theta_sim × theta_sum，扫描触发边界）
  - 4 组 t_cool 灵敏度测试
  - 5 组 theta_evict 灵敏度测试

每组只跑 record_index=0（1条对话，约419轮），每组约1h。
4进程并行，30组约需 7-8h，10h内可跑完。

结果保存到 data/diag_trigger_results.json，支持断点续跑。

用法：
    python diagnose_trigger.py
    python diagnose_trigger.py --max-parallel 2  # 资源受限时用更少进程
    python diagnose_trigger.py --show-results    # 只看结果不运行
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import logging
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

logger = logging.getLogger("diagnose_trigger")

RESULTS_PATH = _ROOT / "data" / "diag_trigger_results.json"
DATA_PATH = str(_ROOT / "dataset" / "locomo10.json")
RECORD_INDICES = [0]   # 只跑第0条记录
MAX_PARALLEL = 4


# ── 30 组参数定义 ─────────────────────────────────────────────────────────────

def build_configs() -> list[dict]:
    configs = []

    # ── 对照组（原始参数，已知 trigger=0）─────────────────────────────────────
    configs.append({
        "group": "control",
        "label": "对照组-原始参数",
        "theta_sim": 0.80,
        "theta_sum": 20,
        "theta_evict": 0.35,
        "t_cool": 3600,
        "decay_interval_rounds": 50,
        "decay_factor": 0.95,
    })

    # ── 主网格（theta_sim × theta_sum），固定 evict=0.35, t_cool=60 ────────────
    # 4 × 5 = 20 组
    for sim in [0.50, 0.55, 0.60, 0.65]:
        for tsum in [3, 5, 8, 12, 15]:
            configs.append({
                "group": "main_grid",
                "label": f"网格-sim{sim}-sum{tsum}",
                "theta_sim": sim,
                "theta_sum": tsum,
                "theta_evict": 0.35,
                "t_cool": 60,
                "decay_interval_rounds": 50,
                "decay_factor": 0.90,
            })

    # ── t_cool 灵敏度（固定 sim=0.60, sum=5）────────────────────────────────
    # 4 组
    for t_cool in [15, 30, 120, 300]:
        configs.append({
            "group": "tcool_sensitivity",
            "label": f"t_cool-{t_cool}s",
            "theta_sim": 0.60,
            "theta_sum": 5,
            "theta_evict": 0.35,
            "t_cool": t_cool,
            "decay_interval_rounds": 50,
            "decay_factor": 0.90,
        })

    # ── theta_evict 灵敏度（固定 sim=0.60, sum=5, t_cool=60）────────────────
    # 5 组
    for evict in [0.15, 0.25, 0.45, 0.55, 0.65]:
        configs.append({
            "group": "evict_sensitivity",
            "label": f"evict-{evict}",
            "theta_sim": 0.60,
            "theta_sum": 5,
            "theta_evict": evict,
            "t_cool": 60,
            "decay_interval_rounds": 50,
            "decay_factor": 0.90,
        })

    return configs  # 合计 1 + 20 + 4 + 5 = 30 组


# ── 结果文件 I/O ──────────────────────────────────────────────────────────────

def _load_results() -> list[dict]:
    if RESULTS_PATH.exists():
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_results(results: list[dict]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _config_key(cfg: dict) -> tuple:
    return (
        cfg["theta_sim"],
        cfg["theta_sum"],
        cfg["theta_evict"],
        cfg["t_cool"],
        cfg["decay_interval_rounds"],
    )


# ── 日志分析 ──────────────────────────────────────────────────────────────────

def _parse_log_stats(log_path: str) -> dict:
    """从日志文件提取触发/驱逐统计，并记录最大证据池大小。"""
    stats = {
        "trigger_count": 0,
        "renorm_complete_count": 0,
        "evicted_count": 0,
        "max_pool_size": 0,
        "no_neighbor_count": 0,
        "entailment_false_count": 0,
    }
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "TRIGGER:" in line:
                    stats["trigger_count"] += 1
                if "Renorm[" in line and "complete:" in line:
                    stats["renorm_complete_count"] += 1
                    # 解析驱逐数：eviction_stats={'evicted': N}
                    m = re.search(r"'evicted':\s*(\d+)", line)
                    if m:
                        stats["evicted_count"] += int(m.group(1))
                if "no valid neighbors" in line:
                    stats["no_neighbor_count"] += 1
                if "entailment=False" in line:
                    stats["entailment_false_count"] += 1
                # pool_size=N/theta_sum（DEBUG级）
                m = re.search(r"pool_size=(\d+)/", line)
                if m:
                    stats["max_pool_size"] = max(
                        stats["max_pool_size"], int(m.group(1))
                    )
    except FileNotFoundError:
        pass
    return stats


# ── 单次实验 ──────────────────────────────────────────────────────────────────

async def run_single_trial(cfg: dict, trial_id: str) -> dict:
    """在当前进程中运行一组参数，返回完整结果字典。"""
    from phaseforget.config.settings import Settings
    from phaseforget.pipeline.orchestrator import PhaseForgetSystem
    from phaseforget.evaluation.loaders.locomo import LoCoMoLoader
    from phaseforget.evaluation.benchmark import BenchmarkRunner
    from phaseforget.utils.logger import setup_logging

    exp_id = f"diag_{trial_id}"
    base_data = _ROOT / "data" / exp_id
    base_data.mkdir(parents=True, exist_ok=True)
    log_file = str(base_data / "phaseforget.log")

    # DEBUG 级日志，捕获 pool_size 信息
    setup_logging(level="DEBUG", log_file=log_file)

    settings = Settings(
        experiment_id=exp_id,
        theta_sim=cfg["theta_sim"],
        theta_sum=cfg["theta_sum"],
        theta_evict=cfg["theta_evict"],
        t_cool=cfg["t_cool"],
        decay_interval_rounds=cfg["decay_interval_rounds"],
        decay_factor=cfg["decay_factor"],
        chroma_persist_dir=str(base_data / "chroma_db"),
        sqlite_db_path=str(base_data / "phaseforget.db"),
        log_level="DEBUG",
        log_file=log_file,
    )

    system = PhaseForgetSystem(settings=settings)
    await system.initialize()

    loader = LoCoMoLoader(record_indices=RECORD_INDICES)
    runner = BenchmarkRunner(
        system,
        llm_client=system._llm,
        checkpoint_path=str(base_data / "ckpt.json"),
    )

    t0 = time.time()
    bench_results = await runner.run(loader, DATA_PATH)
    elapsed = time.time() - t0

    pf = bench_results.get("PhaseForget")
    mem_stats = await system.get_stats()
    await system.close()

    log_stats = _parse_log_stats(log_file)

    # 清理 Chroma/DB，只保留日志
    for p in [base_data / "chroma_db", base_data / "phaseforget.db", base_data / "ckpt.json"]:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        except Exception:
            pass

    by_category = {}
    if pf:
        raw = {}
        for cat, cat_m in pf.by_category.items():
            raw[str(cat)] = {
                "avg_f1": cat_m.avg_f1,
                "avg_sbert": cat_m.avg_sbert,
                "n": len(cat_m.f1_scores),
            }
        by_category = raw

    return {
        "trial_id": trial_id,
        "group": cfg["group"],
        "label": cfg["label"],
        "params": {
            "theta_sim": cfg["theta_sim"],
            "theta_sum": cfg["theta_sum"],
            "theta_evict": cfg["theta_evict"],
            "t_cool": cfg["t_cool"],
            "decay_interval_rounds": cfg["decay_interval_rounds"],
            "decay_factor": cfg["decay_factor"],
        },
        "trigger": {
            "trigger_count": log_stats["trigger_count"],
            "renorm_complete_count": log_stats["renorm_complete_count"],
            "evicted_count": log_stats["evicted_count"],
            "max_pool_size": log_stats["max_pool_size"],
            "entailment_false_count": log_stats["entailment_false_count"],
            "no_neighbor_count": log_stats["no_neighbor_count"],
        },
        "memory": {
            "total_notes": mem_stats.get("total_notes", 0),
            "abstract_notes": mem_stats.get("abstract_notes", 0),
        },
        "metrics": {
            "avg_f1": pf.avg_f1 if pf else 0.0,
            "avg_sbert": pf.avg_sbert if pf else 0.0,
            "avg_rouge_l": pf.avg_rouge_l if pf else 0.0,
            "n_questions": len(pf.f1_scores) if pf else 0,
            "by_category": by_category,
        },
        "elapsed_s": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        "log_path": log_file,
    }


def _run_in_subprocess(payload: dict) -> dict:
    """子进程入口。"""
    return asyncio.run(run_single_trial(payload["cfg"], payload["trial_id"]))


# ── 结果展示 ──────────────────────────────────────────────────────────────────

def print_leaderboard(results: list[dict]) -> None:
    if not results:
        print("暂无结果。")
        return

    sep = "=" * 110
    print(f"\n{sep}")
    print("  触发诊断结果总览（按触发次数降序）")
    print(sep)
    print(
        f"{'#':>3} {'分组':<20} {'sim':>5} {'sum':>5} {'evict':>6} "
        f"{'t_cool':>7} {'触发':>5} {'Renorm':>7} {'驱逐':>5} "
        f"{'Sigma':>6} {'最大池':>7} {'F1':>7} {'耗时(s)':>8}"
    )
    print("-" * 110)

    sorted_r = sorted(results, key=lambda x: x["trigger"]["trigger_count"], reverse=True)
    for i, r in enumerate(sorted_r, 1):
        p = r["params"]
        t = r["trigger"]
        m = r["metrics"]
        mem = r["memory"]
        print(
            f"{i:>3} {r['label']:<20} "
            f"{p['theta_sim']:>5.2f} {p['theta_sum']:>5} {p['theta_evict']:>6.2f} "
            f"{p['t_cool']:>7} {t['trigger_count']:>5} {t['renorm_complete_count']:>7} "
            f"{t['evicted_count']:>5} {mem['abstract_notes']:>6} "
            f"{t['max_pool_size']:>7} {m['avg_f1']:>7.4f} {r['elapsed_s']:>8.1f}"
        )
    print(sep)

    # 找出能触发的最优参数
    triggered = [r for r in results if r["trigger"]["trigger_count"] > 0]
    not_triggered = [r for r in results if r["trigger"]["trigger_count"] == 0]
    print(f"\n  能触发重整化的参数组合: {len(triggered)}/{len(results)}")
    print(f"  从未触发的参数组合:     {len(not_triggered)}/{len(results)}")

    if triggered:
        best_by_trigger = max(triggered, key=lambda x: x["trigger"]["trigger_count"])
        best_by_f1 = max(triggered, key=lambda x: x["metrics"]["avg_f1"])
        print(f"\n  触发次数最多: {best_by_trigger['label']}")
        print(f"    trigger={best_by_trigger['trigger']['trigger_count']}, "
              f"evicted={best_by_trigger['trigger']['evicted_count']}, "
              f"F1={best_by_trigger['metrics']['avg_f1']:.4f}")
        print(f"\n  触发组中F1最高: {best_by_f1['label']}")
        print(f"    trigger={best_by_f1['trigger']['trigger_count']}, "
              f"evicted={best_by_f1['trigger']['evicted_count']}, "
              f"F1={best_by_f1['metrics']['avg_f1']:.4f}")

    # 触发边界分析（针对主网格）
    grid = [r for r in results if r["group"] == "main_grid"]
    if grid:
        print(f"\n{'─'*110}")
        print("  主网格触发边界（+ = 触发, · = 未触发）")
        print(f"  {'theta_sim/sum':>14} " +
              "  ".join(f"{s:>5}" for s in [3, 5, 8, 12, 15]))
        print(f"  {'':>14}─────────────────────────────────")
        for sim in [0.50, 0.55, 0.60, 0.65]:
            row = f"  {sim:>14.2f} "
            for tsum in [3, 5, 8, 12, 15]:
                match = next(
                    (r for r in grid
                     if abs(r["params"]["theta_sim"] - sim) < 0.001
                     and r["params"]["theta_sum"] == tsum),
                    None
                )
                if match is None:
                    row += "  ?   "
                elif match["trigger"]["trigger_count"] > 0:
                    row += f"  +{match['trigger']['trigger_count']:<3} "
                else:
                    row += "  ·    "
            print(row)

    print(f"\n完整结果保存于: {RESULTS_PATH}")
    print(sep)


def print_trial_summary(r: dict) -> None:
    t = r["trigger"]
    m = r["metrics"]
    status = "✓触发" if t["trigger_count"] > 0 else "✗未触发"
    print(
        f"  {status}  trigger={t['trigger_count']} renorm={t['renorm_complete_count']} "
        f"evicted={t['evicted_count']} maxPool={t['max_pool_size']} "
        f"F1={m['avg_f1']:.4f}  耗时={r['elapsed_s']}s"
    )


# ── 主流程 ────────────────────────────────────────────────────────────────────

async def main(max_parallel: int = MAX_PARALLEL, show_only: bool = False) -> None:
    configs = build_configs()
    all_results = _load_results()

    if show_only:
        print_leaderboard(all_results)
        return

    # 只跳过真正成功的组合（trigger_count != -1 且没有 error 字段）
    completed_keys = {
        _config_key(r["params"])
        for r in all_results
        if "error" not in r and r.get("trigger", {}).get("trigger_count", -1) != -1
    }

    pending: list[tuple[int, dict, str]] = []
    for i, cfg in enumerate(configs):
        if _config_key(cfg) in completed_keys:
            continue
        trial_id = f"{i+1:03d}_{cfg['theta_sim']:.2f}_{cfg['theta_sum']}_{cfg['theta_evict']:.2f}_{cfg['t_cool']}_{time.time_ns()}"
        pending.append((i + 1, cfg, trial_id))

    skipped = len(configs) - len(pending)
    total = len(configs)

    print("=" * 80)
    print("  PhaseForget 触发诊断 —— 并行版")
    print("=" * 80)
    print(f"  总组合数:   {total}")
    print(f"  已完成:     {skipped}（断点续跑）")
    print(f"  待运行:     {len(pending)}")
    print(f"  并行进程:   {max_parallel}")
    print(f"  数据记录:   record_index={RECORD_INDICES}")
    print(f"  结果路径:   {RESULTS_PATH}")
    print("=" * 80)

    if not pending:
        print("\n所有组合已完成，直接输出结果：")
        print_leaderboard(all_results)
        return

    if max_parallel == 1:
        for trial_num, cfg, trial_id in pending:
            print(
                f"\n[{trial_num}/{total}] {cfg['label']}  "
                f"sim={cfg['theta_sim']} sum={cfg['theta_sum']} "
                f"evict={cfg['theta_evict']} t_cool={cfg['t_cool']}  "
                f"开始: {datetime.now().strftime('%H:%M:%S')}"
            )
            try:
                result = await run_single_trial(cfg, trial_id)
                all_results.append(result)
                _save_results(all_results)
                print_trial_summary(result)
            except Exception as e:
                logger.error(f"Trial {trial_num} 失败: {e}", exc_info=True)
    else:
        # 并行路径（与 hyperparameter_search.py 完全一致的有序落盘逻辑）
        ordered: dict[int, tuple[int, dict, dict]] = {}
        next_flush = 0

        with ProcessPoolExecutor(max_workers=max_parallel) as executor:
            future_map: dict = {}
            for idx, (trial_num, cfg, trial_id) in enumerate(pending):
                print(
                    f"[{trial_num}/{total}] {cfg['label']}  "
                    f"sim={cfg['theta_sim']} sum={cfg['theta_sum']} "
                    f"evict={cfg['theta_evict']} t_cool={cfg['t_cool']}  "
                    f"已提交 {datetime.now().strftime('%H:%M:%S')}"
                )
                fut = executor.submit(
                    _run_in_subprocess,
                    {"cfg": cfg, "trial_id": trial_id},
                )
                future_map[fut] = idx

            for fut in as_completed(future_map):
                idx = future_map[fut]
                trial_num, cfg, _ = pending[idx]
                try:
                    result = fut.result()
                except Exception as e:
                    result = {
                        "trial_id": "unknown",
                        "group": cfg["group"],
                        "label": cfg["label"],
                        "params": cfg,
                        "trigger": {
                            "trigger_count": -1,
                            "renorm_complete_count": -1,
                            "evicted_count": -1,
                            "max_pool_size": -1,
                            "entailment_false_count": -1,
                            "no_neighbor_count": -1,
                        },
                        "memory": {"total_notes": -1, "abstract_notes": -1},
                        "metrics": {"avg_f1": 0.0, "avg_sbert": 0.0, "avg_rouge_l": 0.0, "n_questions": 0, "by_category": {}},
                        "elapsed_s": 0.0,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                    }
                ordered[idx] = (trial_num, cfg, result)

                # 按提交顺序依次落盘，保证 JSON 文件中结果有序
                while next_flush in ordered:
                    flush_num, flush_cfg, flush_result = ordered.pop(next_flush)
                    print(
                        f"[{flush_num}/{total}] {flush_cfg['label']}  "
                        f"已完成 {datetime.now().strftime('%H:%M:%S')}"
                    )
                    print_trial_summary(flush_result)
                    all_results.append(flush_result)
                    _save_results(all_results)
                    next_flush += 1

    print("\n所有实验完成！")
    print_leaderboard(all_results)


def main_entry() -> None:
    global RESULTS_PATH

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="PhaseForget 触发诊断 —— 并行版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--max-parallel", type=int, default=MAX_PARALLEL,
        help=f"并行进程数（默认: {MAX_PARALLEL}）",
    )
    parser.add_argument(
        "--show-results", action="store_true",
        help="只显示已有结果，不运行新实验",
    )
    parser.add_argument(
        "--output", type=str, default=str(RESULTS_PATH),
        help=f"结果保存路径（默认: {RESULTS_PATH}）",
    )
    args = parser.parse_args()

    # 更新全局结果路径
    RESULTS_PATH = Path(args.output)

    asyncio.run(main(max_parallel=args.max_parallel, show_only=args.show_results))


if __name__ == "__main__":
    main_entry()

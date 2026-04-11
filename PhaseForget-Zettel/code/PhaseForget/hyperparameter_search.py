"""
PhaseForget-Zettel 超参数自动搜索脚本
=====================================

功能：
  1. 支持只选 locomo10.json 中部分记录（通过 --record-indices 参数）
  2. 自动网格搜索 / 随机搜索三个最重要的超参数：
       - theta_sim   (拓扑邻居相似度阈值)
       - theta_sum   (证据池触发重整化阈值)
       - theta_evict (低效笔记驱逐阈值)
  3. 每次搜索使用独立的 experiment_id，数据完全隔离
  4. 结果自动保存到 data/hparam_search_results.json 并打印排行榜

典型用法：
  # 快速搜索：只用记录 0 和 1，网格搜索
  python hyperparameter_search.py --record-indices 0,1 --search-type grid

  # 随机搜索：用前3条记录，搜索20组组合
  python hyperparameter_search.py --record-indices 0,1,2 --search-type random --n-trials 20

  # 指定参数范围（覆盖默认值）
  python hyperparameter_search.py --record-indices 0 \\
      --theta-sim-values 0.5,0.65,0.8 \\
      --theta-sum-values 3,5,8 \\
      --theta-evict-values 0.2,0.35,0.5

  # 查看帮助
  python hyperparameter_search.py --help
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import logging
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── 确保 src 在路径中 ───────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

logger = logging.getLogger("hparam_search")


def _check_network(timeout: float = 5.0) -> bool:
    """检查能否连接到 HuggingFace（模型下载源）。"""
    import socket

    host = "huggingface.co"
    try:
        socket.create_connection((host, 443), timeout=timeout)
        return True
    except OSError:
        return False


def _ensure_model_cached() -> None:
    """确保 all-MiniLM-L6-v2 模型已缓存，无网络则提前退出。"""
    from pathlib import Path

    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    # 模型缓存目录特征：存在 transformers 相关的文件夹
    marker = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    if marker.exists():
        return  # 已缓存

    if _check_network():
        # 有网但未缓存，触发下载
        print("[INFO] 首次运行，正在下载 all-MiniLM-L6-v2 模型（约 90MB）...")
        print("[INFO] 下载完成后下次运行无需网络。")
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[INFO] 模型下载完成！")
    else:
        print("[ERROR] 检测不到网络连接（HuggingFace 不可达）")
        print("[ERROR] 请先开启 VPN 后再运行本脚本")
        sys.exit(1)


# ── 默认搜索空间（较大步长，用于快速定位合理区间）────────────────────────────

DEFAULT_THETA_SIM_VALUES = [0.5, 0.65, 0.75, 0.85]
DEFAULT_THETA_SUM_VALUES = [3, 5, 8, 12]
DEFAULT_THETA_EVICT_VALUES = [0.15, 0.3, 0.45, 0.6]
DEFAULT_DECAY_INTERVAL_VALUES = [50]


# ── 结果存储路径 ────────────────────────────────────────────────────────────
# 默认路径，可以通过 --output 参数覆盖
DEFAULT_RESULTS_PATH = _ROOT / "data" / "hparam_search_results.json"
RESULTS_PATH: Path = DEFAULT_RESULTS_PATH


def _load_existing_results() -> list[dict]:
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


def _composite_score(metrics: dict) -> float:
    """
    综合评分 = 0.4*F1 + 0.3*ROUGE-L + 0.2*METEOR + 0.1*BLEU
    聚焦于最能反映记忆质量的指标组合。
    """
    return (
        0.4 * metrics.get("avg_f1", 0.0)
        + 0.3 * metrics.get("avg_rouge_l", 0.0)
        + 0.2 * metrics.get("avg_meteor", 0.0)
        + 0.1 * metrics.get("avg_bleu", 0.0)
    )


def _run_trial_in_subprocess(payload: dict[str, Any]) -> dict[str, Any]:
    """
    子进程入口：在独立进程中执行单次 trial，避免环境变量和资源冲突。
    """
    return asyncio.run(
        run_single_trial(
            theta_sim=payload["theta_sim"],
            theta_sum=payload["theta_sum"],
            theta_evict=payload["theta_evict"],
            decay_interval_rounds=payload["decay_interval_rounds"],
            record_indices=payload["record_indices"],
            data_path=payload["data_path"],
            trial_id=payload["trial_id"],
            include_adversarial=payload["include_adversarial"],
            disable_self_retrieval=payload.get("disable_self_retrieval", False),
        )
    )


async def run_single_trial(
    theta_sim: float,
    theta_sum: int,
    theta_evict: float,
    decay_interval_rounds: int,
    record_indices: list[int],
    data_path: str,
    trial_id: str,
    include_adversarial: bool = False,
    disable_self_retrieval: bool = False,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    运行单次超参数组合的实验，返回评估结果字典。

    使用独立的 experiment_id 隔离数据，实验结束后清理临时存储。
    """
    from phaseforget.config.settings import Settings
    from phaseforget.pipeline.orchestrator import PhaseForgetSystem
    from phaseforget.evaluation.loaders.locomo import LoCoMoLoader
    from phaseforget.evaluation.benchmark import BenchmarkRunner
    from phaseforget.utils.logger import setup_logging

    experiment_id = f"hps_{trial_id}"
    base_data = _ROOT / "data" / experiment_id

    # 构造 Settings（直接注入超参数，不依赖 .env）
    env_overrides = {
        "EXPERIMENT_ID": experiment_id,
        "THETA_SIM": str(theta_sim),
        "THETA_SUM": str(theta_sum),
        "THETA_EVICT": str(theta_evict),
        "DECAY_INTERVAL_ROUNDS": str(decay_interval_rounds),
        "CHROMA_PERSIST_DIR": f"./data/{experiment_id}/chroma_db",
        "SQLITE_DB_PATH": f"./data/{experiment_id}/phaseforget.db",
        "LOG_LEVEL": "INFO",
        "LOG_FILE": f"./data/{experiment_id}/phaseforget.log",
    }
    if extra_env:
        env_overrides.update(extra_env)

    old_env = {}
    for k, v in env_overrides.items():
        old_env[k] = os.environ.get(k)
        os.environ[k] = v

    start_time = time.time()
    result_metrics = {}

    try:
        log_file_path = f"./data/{experiment_id}/phaseforget.log"
        settings = Settings(
            experiment_id=experiment_id,
            theta_sim=theta_sim,
            theta_sum=theta_sum,
            theta_evict=theta_evict,
            decay_interval_rounds=decay_interval_rounds,
            chroma_persist_dir=f"./data/{experiment_id}/chroma_db",
            sqlite_db_path=f"./data/{experiment_id}/phaseforget.db",
            log_level="INFO",
            log_file=log_file_path,
        )

        # 初始化日志文件（之前缺少这一步导致日志只打印到控制台）
        setup_logging(level="INFO", log_file=log_file_path)

        system = PhaseForgetSystem(settings=settings)
        await system.initialize()

        loader = LoCoMoLoader(
            record_indices=record_indices,
            include_adversarial=include_adversarial,
        )
        ckpt_path = str(base_data / "bench_checkpoint.json")
        runner = BenchmarkRunner(
            system,
            llm_client=system._llm,
            checkpoint_path=ckpt_path,
            disable_self_retrieval=disable_self_retrieval,
        )

        bench_results = await runner.run(
            dataset_loader=loader,
            dataset_path=data_path,
        )

        pf = bench_results.get("PhaseForget")
        if pf:
            result_metrics = {
                "avg_f1": pf.avg_f1,
                "avg_bleu": pf.avg_bleu,
                "avg_rouge_l": pf.avg_rouge_l,
                "avg_rouge2": pf.avg_rouge2,
                "avg_meteor": pf.avg_meteor,
                "avg_sbert": pf.avg_sbert,
                "avg_retrieval_time_us": pf.avg_retrieval_time_us,
                "n_questions": len(pf.f1_scores),
            }
            # 添加按类别的指标
            if pf.by_category:
                result_metrics["by_category"] = {}
                for cat, cat_m in sorted(pf.by_category.items()):
                    result_metrics["by_category"][str(cat)] = {
                        "avg_f1": cat_m.avg_f1,
                        "avg_bleu": cat_m.avg_bleu,
                        "avg_rouge_l": cat_m.avg_rouge_l,
                        "avg_rouge2": cat_m.avg_rouge2,
                        "avg_meteor": cat_m.avg_meteor,
                        "avg_sbert": cat_m.avg_sbert,
                        "n_questions": len(cat_m.f1_scores),
                    }

        # ── Diagnostic: dump memory stats to verify forget mechanism ────
        try:
            stats = await system.get_stats()
            logger.info(
                f"[TRIAL-STATS] trial={trial_id} "
                f"total_notes={stats.get('total_notes', '?')} "
                f"abstract_notes={stats.get('abstract_notes', '?')} "
                f"total_links={stats.get('total_links', '?')} "
                f"interactions={stats.get('interaction_count', '?')}"
            )
            result_metrics["memory_stats"] = stats
        except Exception as e2:
            logger.debug(f"Stats collection failed: {e2}")

        await system.close()

    except Exception as e:
        logger.error(f"Trial {trial_id} failed: {e}", exc_info=True)
        result_metrics = {"error": str(e)}

    finally:
        # 恢复环境变量
        for k, old_v in old_env.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v

        # 保留日志文件用于诊断，但清理 chroma_db 和 phaseforget.db
        chroma_dir = base_data / "chroma_db"
        db_file = base_data / "phaseforget.db"
        if chroma_dir.exists():
            try:
                shutil.rmtree(chroma_dir)
            except Exception as e:
                logger.warning(f"Failed to clean up {chroma_dir}: {e}")
        if db_file.exists():
            try:
                db_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to clean up {db_file}: {e}")

    elapsed = time.time() - start_time
    return {
        "trial_id": trial_id,
        "params": {
            "theta_sim": theta_sim,
            "theta_sum": theta_sum,
            "theta_evict": theta_evict,
            "decay_interval_rounds": decay_interval_rounds,
        },
        "disable_self_retrieval": disable_self_retrieval,
        "record_indices": record_indices,
        "metrics": result_metrics,
        "composite_score": _composite_score(result_metrics),
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
    }


def build_grid(
    theta_sim_values: list[float],
    theta_sum_values: list[int],
    theta_evict_values: list[float],
    decay_interval_values: list[int],
) -> list[dict]:
    """生成笛卡尔积网格搜索参数组合列表。"""
    combos = []
    for ts, tsum, te, decay_intv in itertools.product(
        theta_sim_values, theta_sum_values, theta_evict_values, decay_interval_values
    ):
        combos.append({
            "theta_sim": ts,
            "theta_sum": tsum,
            "theta_evict": te,
            "decay_interval_rounds": decay_intv,
        })
    return combos


def build_random(
    theta_sim_values: list[float],
    theta_sum_values: list[int],
    theta_evict_values: list[float],
    decay_interval_values: list[int],
    n_trials: int,
    seed: int = 42,
) -> list[dict]:
    """随机采样参数组合。"""
    rng = random.Random(seed)
    all_combos = build_grid(
        theta_sim_values, theta_sum_values, theta_evict_values, decay_interval_values
    )
    if n_trials >= len(all_combos):
        return all_combos
    return rng.sample(all_combos, n_trials)


def print_leaderboard(results: list[dict], top_n: int = 10) -> None:
    """打印超参数搜索结果排行榜。"""
    valid = [r for r in results if "error" not in r.get("metrics", {})]
    if not valid:
        print("暂无有效结果。")
        return

    sorted_results = sorted(valid, key=lambda x: x["composite_score"], reverse=True)

    print("\n" + "=" * 100)
    print(f"  超参数搜索排行榜 (Top {min(top_n, len(sorted_results))})")
    print("=" * 100)
    print(
        f"{'排名':<4} {'theta_sim':>10} {'theta_sum':>10} {'theta_evict':>12} {'decay':>8} "
        f"{'综合分':>8} {'F1':>8} {'ROUGE-L':>8} {'METEOR':>8} {'BLEU':>8} "
        f"{'样本数':>6} {'耗时(s)':>8}"
    )
    print("-" * 100)

    for rank, r in enumerate(sorted_results[:top_n], 1):
        p = r["params"]
        m = r["metrics"]
        print(
            f"{rank:<4} {p['theta_sim']:>10.2f} {p['theta_sum']:>10} {p['theta_evict']:>12.2f} "
            f"{p.get('decay_interval_rounds', DEFAULT_DECAY_INTERVAL_VALUES[0]):>8} "
            f"{r['composite_score']:>8.4f} {m.get('avg_f1', 0):>8.4f} "
            f"{m.get('avg_rouge_l', 0):>8.4f} {m.get('avg_meteor', 0):>8.4f} "
            f"{m.get('avg_bleu', 0):>8.4f} {m.get('n_questions', 0):>6} "
            f"{r.get('elapsed_seconds', 0):>8.1f}"
        )

    print("=" * 100)

    if sorted_results:
        best = sorted_results[0]
        bp = best["params"]
        bm = best["metrics"]
        print(f"\n最佳超参数组合（综合分 {best['composite_score']:.4f}）：")
        print(f"  theta_sim   = {bp['theta_sim']}")
        print(f"  theta_sum   = {bp['theta_sum']}")
        print(f"  theta_evict = {bp['theta_evict']}")
        print(f"  decay_intv  = {bp.get('decay_interval_rounds', DEFAULT_DECAY_INTERVAL_VALUES[0])}")
        print(f"\n对应的 .env 配置：")
        print(f"  THETA_SIM={bp['theta_sim']}")
        print(f"  THETA_SUM={bp['theta_sum']}")
        print(f"  THETA_EVICT={bp['theta_evict']}")
        print(f"  DECAY_INTERVAL_ROUNDS={bp.get('decay_interval_rounds', DEFAULT_DECAY_INTERVAL_VALUES[0])}")

        # 显示按类别的细分指标
        by_cat = bm.get("by_category", {})
        if by_cat:
            print(f"\n最佳结果的类别细分（Category Breakdown）：")
            print("-" * 80)
            cat_names = {
                "1": "Single-hop (单跳)",
                "2": "Temporal (时间)",
                "3": "Multi-hop (多跳)",
                "4": "Open-domain (开放域)",
                "5": "Adversarial (对抗性)",
            }
            print(f"{'类别':<25} {'F1':>8} {'ROUGE-L':>8} {'METEOR':>8} {'BLEU':>8} {'样本数':>6}")
            print("-" * 80)
            for cat in sorted(by_cat.keys(), key=int):
                cat_data = by_cat[cat]
                cat_name = cat_names.get(cat, f"Category {cat}")
                print(
                    f"{cat_name:<25} "
                    f"{cat_data.get('avg_f1', 0):>8.4f} "
                    f"{cat_data.get('avg_rouge_l', 0):>8.4f} "
                    f"{cat_data.get('avg_meteor', 0):>8.4f} "
                    f"{cat_data.get('avg_bleu', 0):>8.4f} "
                    f"{cat_data.get('n_questions', 0):>6}"
                )
            print("-" * 80)
            print()


async def main_async(args: argparse.Namespace) -> None:
    # 解析 record_indices
    record_indices: list[int] | None = None
    if args.record_indices:
        try:
            record_indices = [int(x.strip()) for x in args.record_indices.split(",")]
        except ValueError:
            print(f"[ERROR] --record-indices 必须是逗号分隔的整数，收到: {args.record_indices}")
            sys.exit(1)

    # 解析超参数搜索范围
    def parse_floats(s: str) -> list[float]:
        return [float(x.strip()) for x in s.split(",")]

    def parse_ints(s: str) -> list[int]:
        return [int(x.strip()) for x in s.split(",")]

    theta_sim_values = parse_floats(args.theta_sim_values) if args.theta_sim_values else DEFAULT_THETA_SIM_VALUES
    theta_sum_values = parse_ints(args.theta_sum_values) if args.theta_sum_values else DEFAULT_THETA_SUM_VALUES
    theta_evict_values = parse_floats(args.theta_evict_values) if args.theta_evict_values else DEFAULT_THETA_EVICT_VALUES
    decay_interval_values = (
        parse_ints(args.decay_interval_values)
        if args.decay_interval_values
        else DEFAULT_DECAY_INTERVAL_VALUES
    )

    # 构建参数组合
    if args.search_type == "grid":
        combos = build_grid(
            theta_sim_values, theta_sum_values, theta_evict_values, decay_interval_values
        )
    else:
        combos = build_random(
            theta_sim_values, theta_sum_values, theta_evict_values, decay_interval_values,
            n_trials=args.n_trials, seed=args.seed
        )

    total = len(combos)
    print(f"\n{'='*60}")
    print(f"  PhaseForget 超参数搜索")
    print(f"{'='*60}")
    print(f"  搜索模式     : {args.search_type}")
    print(f"  参数组合数   : {total}")
    print(f"  theta_sim    : {theta_sim_values}")
    print(f"  theta_sum    : {theta_sum_values}")
    print(f"  theta_evict  : {theta_evict_values}")
    print(f"  decay_intv   : {decay_interval_values}")
    print(f"  数据集记录   : {record_indices if record_indices else '全部(0-9)'}")
    print(f"  包含对抗题   : {args.include_adversarial}")
    print(f"  关闭自检索   : {args.disable_self_retrieval}")
    print(f"  数据集路径   : {args.data_path}")
    print(f"  结果保存至   : {RESULTS_PATH}")
    print(f"{'='*60}\n")

    # 加载已有结果（支持断点续搜）
    all_results = _load_existing_results()
    completed_keys = {
        (
            r["params"]["theta_sim"],
            r["params"]["theta_sum"],
            r["params"]["theta_evict"],
            r["params"].get("decay_interval_rounds", DEFAULT_DECAY_INTERVAL_VALUES[0]),
        )
        for r in all_results
        if r.get("record_indices") == record_indices
    }

    pending = [
        c for c in combos
        if (
            c["theta_sim"],
            c["theta_sum"],
            c["theta_evict"],
            c["decay_interval_rounds"],
        ) not in completed_keys
    ]
    skipped = total - len(pending)
    if skipped > 0:
        print(f"[断点续搜] 已跳过 {skipped} 组已完成的组合，剩余 {len(pending)} 组。\n")

    if not pending:
        print("没有待运行的参数组合。")
    else:
        max_parallel = max(1, args.max_parallel)
        print(f"  并行进程数   : {max_parallel}\n")

        # 输出统一的结果摘要，保持串行和并行模式一致。
        def print_trial_summary(result: dict[str, Any]) -> None:
            if "error" in result.get("metrics", {}):
                print(f"  [FAILED] {result['metrics']['error']}")
                return
            m = result["metrics"]
            print(
                f"  综合分={result['composite_score']:.4f}  "
                f"F1={m.get('avg_f1', 0):.4f}  "
                f"ROUGE-L={m.get('avg_rouge_l', 0):.4f}  "
                f"METEOR={m.get('avg_meteor', 0):.4f}  "
                f"耗时={result['elapsed_seconds']}s"
            )
            by_cat = m.get("by_category", {})
            if by_cat:
                cat_summary = []
                for cat in sorted(by_cat.keys(), key=int):
                    cat_data = by_cat[cat]
                    cat_summary.append(f"C{cat}:F1={cat_data.get('avg_f1', 0):.3f}")
                print(f"    类别细分: {' | '.join(cat_summary)}")

        # 先构建任务列表，确保 trial_id 在并行时也稳定且唯一。
        tasks: list[tuple[int, dict[str, Any], str]] = []
        for trial_num, combo in enumerate(pending, skipped + 1):
            ts, tsum, te, decay_intv = (
                combo["theta_sim"],
                combo["theta_sum"],
                combo["theta_evict"],
                combo["decay_interval_rounds"],
            )
            trial_id = f"{trial_num:04d}_{ts:.2f}_{tsum}_{te:.2f}_{decay_intv}_{time.time_ns()}"
            tasks.append((trial_num, combo, trial_id))

        if max_parallel == 1:
            # 保留串行路径，便于调试与资源受限场景。
            for trial_num, combo, trial_id in tasks:
                ts, tsum, te, decay_intv = (
                    combo["theta_sim"],
                    combo["theta_sum"],
                    combo["theta_evict"],
                    combo["decay_interval_rounds"],
                )
                print(
                    f"[{trial_num}/{total}] theta_sim={ts} theta_sum={tsum} theta_evict={te} decay={decay_intv} "
                    f"  开始时间: {datetime.now().strftime('%H:%M:%S')}"
                )
                result = await run_single_trial(
                    theta_sim=ts,
                    theta_sum=tsum,
                    theta_evict=te,
                    decay_interval_rounds=decay_intv,
                    record_indices=record_indices,
                    data_path=args.data_path,
                    trial_id=trial_id,
                    include_adversarial=args.include_adversarial,
                    disable_self_retrieval=args.disable_self_retrieval,
                )
                all_results.append(result)
                _save_results(all_results)
                print_trial_summary(result)
        else:
            # 并行路径：子进程并发执行，主进程按任务顺序落盘和打印，结果秩序稳定。
            ordered_results: dict[int, tuple[int, dict[str, Any], dict[str, Any]]] = {}
            next_to_flush = 0

            with ProcessPoolExecutor(max_workers=max_parallel) as executor:
                future_to_idx = {}
                for idx, (trial_num, combo, trial_id) in enumerate(tasks):
                    ts, tsum, te, decay_intv = (
                        combo["theta_sim"],
                        combo["theta_sum"],
                        combo["theta_evict"],
                        combo["decay_interval_rounds"],
                    )
                    print(
                        f"[{trial_num}/{total}] theta_sim={ts} theta_sum={tsum} theta_evict={te} decay={decay_intv} "
                        f"  已提交  时间: {datetime.now().strftime('%H:%M:%S')}"
                    )
                    payload = {
                        "theta_sim": ts,
                        "theta_sum": tsum,
                        "theta_evict": te,
                        "decay_interval_rounds": decay_intv,
                        "record_indices": record_indices,
                        "data_path": args.data_path,
                        "trial_id": trial_id,
                        "include_adversarial": args.include_adversarial,
                        "disable_self_retrieval": args.disable_self_retrieval,
                    }
                    fut = executor.submit(_run_trial_in_subprocess, payload)
                    future_to_idx[fut] = idx

                for fut in as_completed(future_to_idx):
                    idx = future_to_idx[fut]
                    trial_num, combo, _ = tasks[idx]
                    try:
                        result = fut.result()
                    except Exception as e:
                        # 子进程级异常兜底，保证整个搜索不中断。
                        result = {
                            "trial_id": "unknown",
                            "params": {
                                "theta_sim": combo["theta_sim"],
                                "theta_sum": combo["theta_sum"],
                                "theta_evict": combo["theta_evict"],
                                "decay_interval_rounds": combo["decay_interval_rounds"],
                            },
                            "record_indices": record_indices,
                            "metrics": {"error": f"subprocess failure: {e}"},
                            "composite_score": 0.0,
                            "elapsed_seconds": 0.0,
                            "timestamp": datetime.now().isoformat(),
                        }
                    ordered_results[idx] = (trial_num, combo, result)

                    while next_to_flush in ordered_results:
                        flush_trial_num, flush_combo, flush_result = ordered_results.pop(next_to_flush)
                        ts, tsum, te, decay_intv = (
                            flush_combo["theta_sim"],
                            flush_combo["theta_sum"],
                            flush_combo["theta_evict"],
                            flush_combo["decay_interval_rounds"],
                        )
                        print(
                            f"[{flush_trial_num}/{total}] theta_sim={ts} theta_sum={tsum} theta_evict={te} decay={decay_intv} "
                            f"  已完成  时间: {datetime.now().strftime('%H:%M:%S')}"
                        )
                        all_results.append(flush_result)
                        _save_results(all_results)
                        print_trial_summary(flush_result)
                        next_to_flush += 1

    print("\n所有实验完成！")
    print_leaderboard(all_results)
    print(f"完整结果已保存到: {RESULTS_PATH}\n")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="PhaseForget-Zettel 超参数自动搜索工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── 数据集参数 ────────────────────────────────────────────────────────
    parser.add_argument(
        "--data-path",
        default="dataset/locomo10.json",
        help="locomo10.json 路径（默认: dataset/locomo10.json）",
    )
    parser.add_argument(
        "--record-indices",
        type=str,
        default=None,
        help=(
            "逗号分隔的记录索引（0-9），例如 '0,1,2'。"
            "不指定则使用全部10条记录。"
            "建议快速搜索时只用1-2条记录。"
        ),
    )
    parser.add_argument(
        "--include-adversarial",
        action="store_true",
        help=(
            "是否在 LoCoMo 评估中包含 category=5 的对抗问题。"
            "默认不包含（遵循常见协议）。"
        ),
    )
    parser.add_argument(
        "--disable-self-retrieval",
        action="store_true",
        help=(
            "关闭对话写入阶段的自检索（self-retrieval）。"
            "用于诊断：对齐 A-MEM 的线性写入行为，"
            "检验自检索对 Open Domain F1 偏高的影响。"
        ),
    )

    # ── 搜索策略 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--search-type",
        choices=["grid", "random"],
        default="grid",
        help="搜索类型：grid=网格搜索（全覆盖），random=随机搜索（默认: grid）",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=12,
        help="随机搜索时的试验次数（默认: 12，仅 --search-type random 有效）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机搜索的随机种子（默认: 42）",
    )

    # ── 超参数搜索空间 ────────────────────────────────────────────────────
    parser.add_argument(
        "--theta-sim-values",
        type=str,
        default=None,
        help=f"theta_sim 候选值，逗号分隔（默认: {','.join(map(str, DEFAULT_THETA_SIM_VALUES))}）",
    )
    parser.add_argument(
        "--theta-sum-values",
        type=str,
        default=None,
        help=f"theta_sum 候选值，逗号分隔（默认: {','.join(map(str, DEFAULT_THETA_SUM_VALUES))}）",
    )
    parser.add_argument(
        "--theta-evict-values",
        type=str,
        default=None,
        help=f"theta_evict 候选值，逗号分隔（默认: {','.join(map(str, DEFAULT_THETA_EVICT_VALUES))}）",
    )
    parser.add_argument(
        "--decay-interval-values",
        type=str,
        default=None,
        help=(
            "decay_interval_rounds 候选值，逗号分隔"
            f"（默认: {','.join(map(str, DEFAULT_DECAY_INTERVAL_VALUES))}）"
        ),
    )

    # ── 工具命令 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--show-results",
        action="store_true",
        help="只显示已有的搜索结果排行榜，不运行新实验",
    )
    parser.add_argument(
        "--clear-results",
        action="store_true",
        help="清除已保存的搜索结果（谨慎使用）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "自定义结果文件保存路径（默认: data/hparam_search_results.json）。"
            "例如: --output experiments/run_1/results.json"
        ),
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help=(
            "并行执行的最大进程数（默认: 1=串行）。"
            "建议从 2 开始尝试，避免机器过载。"
        ),
    )

    args = parser.parse_args()

    # 处理自定义输出路径
    global RESULTS_PATH
    if args.output:
        RESULTS_PATH = Path(args.output)
        # 如果是相对路径，基于项目根目录
        if not RESULTS_PATH.is_absolute():
            RESULTS_PATH = _ROOT / RESULTS_PATH

    # 工具命令不需要网络，提前处理
    if args.clear_results:
        if RESULTS_PATH.exists():
            RESULTS_PATH.unlink()
            print(f"已清除结果文件: {RESULTS_PATH}")
        else:
            print("结果文件不存在，无需清除。")
        return

    if args.show_results:
        results = _load_existing_results()
        if not results:
            print("尚无搜索结果。请先运行实验。")
        else:
            print_leaderboard(results)
        return

    # 切换到脚本所在目录（确保相对路径正确）
    os.chdir(_ROOT)

    # 真正运行实验前检查：确保模型已缓存或网络可用
    _ensure_model_cached()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

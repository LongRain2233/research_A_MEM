"""
PhaseForget-Zettel 超参数自动搜索脚本
=====================================

功能：
  1. 支持多数据集评测（LoCoMo / LongMemEval / PersonaMem / DialSim）
  2. 支持只选数据集中的部分记录（通过 --record-indices 参数）
  3. 三种搜索策略：
       - grid:     笛卡尔积网格搜索（全覆盖）
       - random:   随机采样
       - bayesian: Optuna TPE 贝叶斯优化（推荐，需 pip install optuna）
  4. 搜索 7 个核心超参数：
       - theta_sim            拓扑邻居相似度阈值
       - theta_sum            证据池触发重整化阈值
       - theta_evict          低效笔记驱逐阈值
       - eta                  效用分动量学习率
       - decay_factor         全局衰减系数
       - decay_interval_rounds 衰减频率
       - retrieval_top_k      检索 Top-K
  5. 每次搜索使用独立的 experiment_id，数据完全隔离
  6. 结果自动保存到 data/hparam_search_results.json 并打印排行榜
  7. 贝叶斯模式支持热启动：自动注入已有结果作为先验观测

典型用法：
  # 贝叶斯搜索（推荐）：只用记录 0，搜索 25 次
  python hyperparameter_search.py --record-indices 0 --search-type bayesian --n-trials 25

  # 快速搜索：只用记录 0 和 1，网格搜索
  python hyperparameter_search.py --record-indices 0,1 --search-type grid

  # 随机搜索：用前3条记录，搜索20组组合
  python hyperparameter_search.py --record-indices 0,1,2 --search-type random --n-trials 20

  # 在 LongMemEval 上搜索
  python hyperparameter_search.py --dataset longmemeval --record-indices 0,1,2 --search-type random --n-trials 20

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
    marker = cache_dir / "models--sentence-transformers--all-MiniLM-L6-v2"
    if marker.exists():
        return

    if _check_network():
        print("[INFO] 首次运行，正在下载 all-MiniLM-L6-v2 模型（约 90MB）...")
        print("[INFO] 下载完成后下次运行无需网络。")
        from sentence_transformers import SentenceTransformer

        SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("[INFO] 模型下载完成！")
    else:
        print("[ERROR] 检测不到网络连接（HuggingFace 不可达）")
        print("[ERROR] 请先开启 VPN 后再运行本脚本")
        sys.exit(1)


# ── 默认搜索空间 ──────────────────────────────────────────────────────────────

DEFAULT_THETA_SIM_VALUES = [0.7,0.75,0.8,0.85, 0.9]
DEFAULT_THETA_SUM_VALUES = [5, 10, 15,20,25,30,35, 40]
DEFAULT_THETA_EVICT_VALUES = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
DEFAULT_DECAY_INTERVAL_VALUES = [50]
DEFAULT_DECAY_FACTOR_VALUES = [0.85]
DEFAULT_ETA_VALUES = [0.1]
DEFAULT_RETRIEVAL_TOP_K_VALUES = [10]

# ── 贝叶斯搜索连续空间的默认范围 ──────────────────────────────────────────────

BAYESIAN_RANGES = {
    "theta_sim":             (0.70, 0.90),
    "theta_sum":             (5, 40),
    "theta_evict":           (0.15, 0.50),
    "eta":                   (0.05, 0.30),
    "decay_factor":          (0.75, 0.95),
    "decay_interval_rounds": (20, 100),
    "retrieval_top_k":       (5, 20),
}


# ── 结果存储路径 ──────────────────────────────────────────────────────────────
DEFAULT_RESULTS_PATH = _ROOT / "data" / "hparam_search_results.json"
RESULTS_PATH: Path = DEFAULT_RESULTS_PATH

DEFAULT_DATA_PATHS = {
    "locomo": "dataset/locomo10.json",
    "longmemeval": "dataset/longmemeval_m_cleaned.json",
    "personamem": "dataset/personamem.json",
    "dialsim": "dataset/dialsim.json",
}

DATASETS_WITH_RECORD_SELECTION = {"locomo", "longmemeval"}

DATASET_CATEGORY_LABELS = {
    "locomo": {
        "1": "Single-hop (单跳)",
        "2": "Temporal (时间)",
        "3": "Multi-hop (多跳)",
        "4": "Open-domain (开放域)",
        "5": "Adversarial (对抗性)",
    },
    "longmemeval": {
        "1": "Single-session user",
        "2": "Single-session assistant",
        "3": "Single-session preference",
        "4": "Multi-session",
        "5": "Knowledge update",
        "6": "Temporal reasoning",
        "7": "Abstention",
    },
}


def _build_dataset_loader(
    dataset: str,
    record_indices: list[int] | None,
    include_adversarial: bool,
    include_abstention: bool = False,
):
    from phaseforget.evaluation.loaders import (
        DialSimLoader,
        LoCoMoLoader,
        LongMemEvalLoader,
        PersonaMemLoader,
    )

    if dataset == "locomo":
        return LoCoMoLoader(
            record_indices=record_indices,
            include_adversarial=include_adversarial,
        )
    if dataset == "longmemeval":
        return LongMemEvalLoader(
            record_indices=record_indices,
            include_abstention=include_abstention,
        )
    if dataset == "personamem":
        return PersonaMemLoader()
    if dataset == "dialsim":
        return DialSimLoader()
    raise ValueError(f"Unsupported dataset: {dataset}")


def _result_matches_dataset(result: dict[str, Any], dataset: str) -> bool:
    stored_dataset = result.get("dataset")
    if stored_dataset is None:
        return dataset == "locomo"
    return stored_dataset == dataset


def _filter_results_for_dataset(results: list[dict], dataset: str) -> list[dict]:
    return [r for r in results if _result_matches_dataset(r, dataset)]


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
    """
    return (
        0.4 * metrics.get("avg_f1", 0.0)
        + 0.3 * metrics.get("avg_rouge_l", 0.0)
        + 0.2 * metrics.get("avg_meteor", 0.0)
        + 0.1 * metrics.get("avg_bleu", 0.0)
    )


def _run_trial_in_subprocess(payload: dict[str, Any]) -> dict[str, Any]:
    """子进程入口：在独立进程中执行单次 trial。"""
    return asyncio.run(
        run_single_trial(
            theta_sim=payload["theta_sim"],
            theta_sum=payload["theta_sum"],
            theta_evict=payload["theta_evict"],
            decay_interval_rounds=payload["decay_interval_rounds"],
            decay_factor=payload["decay_factor"],
            eta=payload.get("eta", 0.1),
            retrieval_top_k=payload.get("retrieval_top_k", 10),
            dataset=payload["dataset"],
            record_indices=payload["record_indices"],
            data_path=payload["data_path"],
            trial_id=payload["trial_id"],
            include_adversarial=payload["include_adversarial"],
            include_abstention=payload.get("include_abstention", False),
            disable_self_retrieval=payload.get("disable_self_retrieval", False),
        )
    )


async def run_single_trial(
    theta_sim: float,
    theta_sum: int,
    theta_evict: float,
    decay_interval_rounds: int,
    decay_factor: float,
    eta: float = 0.1,
    retrieval_top_k: int = 10,
    dataset: str = "locomo",
    record_indices: list[int] | None = None,
    data_path: str = DEFAULT_DATA_PATHS["locomo"],
    trial_id: str = "",
    include_adversarial: bool = False,
    include_abstention: bool = False,
    disable_self_retrieval: bool = False,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    运行单次超参数组合的实验，返回评估结果字典。

    使用独立的 experiment_id 隔离数据，实验结束后清理临时存储。
    """
    from phaseforget.config.settings import Settings
    from phaseforget.evaluation.benchmark import BenchmarkRunner
    from phaseforget.pipeline.orchestrator import PhaseForgetSystem
    from phaseforget.utils.logger import setup_logging

    experiment_id = f"hps_{trial_id}"
    base_data = _ROOT / "data" / experiment_id

    env_overrides = {
        "EXPERIMENT_ID": experiment_id,
        "THETA_SIM": str(theta_sim),
        "THETA_SUM": str(theta_sum),
        "THETA_EVICT": str(theta_evict),
        "DECAY_INTERVAL_ROUNDS": str(decay_interval_rounds),
        "DECAY_FACTOR": str(decay_factor),
        "ETA": str(eta),
        "RETRIEVAL_TOP_K": str(retrieval_top_k),
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
    _trial_succeeded = False

    try:
        log_file_path = f"./data/{experiment_id}/phaseforget.log"
        settings = Settings(
            experiment_id=experiment_id,
            theta_sim=theta_sim,
            theta_sum=theta_sum,
            theta_evict=theta_evict,
            decay_interval_rounds=decay_interval_rounds,
            decay_factor=decay_factor,
            eta=eta,
            retrieval_top_k=retrieval_top_k,
            chroma_persist_dir=f"./data/{experiment_id}/chroma_db",
            sqlite_db_path=f"./data/{experiment_id}/phaseforget.db",
            log_level="INFO",
            log_file=log_file_path,
        )

        setup_logging(level="INFO", log_file=log_file_path)

        system = PhaseForgetSystem(settings=settings)
        await system.initialize()

        loader = _build_dataset_loader(
            dataset=dataset,
            record_indices=record_indices,
            include_adversarial=include_adversarial,
            include_abstention=include_abstention,
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
                "avg_retrieval_latency_s": round(pf.avg_retrieval_time_s, 6),
                "avg_context_tokens": round(pf.avg_context_tokens, 1),
                "total_ingest_tokens": pf.total_ingest_tokens,
                "n_questions": len(pf.f1_scores),
                "answer_parse_fail": pf.answer_parse_fail,
                "query_expand_parse_fail": pf.query_expand_parse_fail,
                "parse_fail_rate": round(pf.parse_fail_rate, 4),
                "peak_memory_mb": round(max(pf.memory_usage_mb), 1) if pf.memory_usage_mb else 0.0,
            }
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

        # ── 聚合 timing JSON：Build Time / Ingest 吞吐率 / QA 吞吐率 ──────
        timing_file = base_data / "bench_checkpoint_timing.json"
        timing_records: list[dict] = []
        if timing_file.exists():
            try:
                with open(timing_file, "r", encoding="utf-8") as _tf:
                    timing_records = json.load(_tf)
            except Exception as _e:
                logger.debug(f"Failed to read timing file: {_e}")

        _build_time_s = sum(r.get("phase1_ingest_s", 0.0) for r in timing_records)
        _qa_time_s    = sum(r.get("phase2_qa_s", 0.0)    for r in timing_records)
        _total_turns  = sum(r.get("n_turns", 0)           for r in timing_records)
        _total_qs     = sum(r.get("n_questions", 0)       for r in timing_records)
        _build_time_h = _build_time_s / 3600.0
        _ingest_tpm   = (_total_turns / _build_time_s * 60) if _build_time_s > 0 else 0.0
        _qa_tput_qps  = (_total_qs / _qa_time_s) if _qa_time_s > 0 else 0.0

        result_metrics.update({
            "build_time_s":         round(_build_time_s, 1),
            "build_time_h":         round(_build_time_h, 6),
            "qa_time_s":            round(_qa_time_s, 1),
            "total_turns_ingested": _total_turns,
            "ingest_throughput_tpm": round(_ingest_tpm, 2),
            "qa_throughput_qps":    round(_qa_tput_qps, 3),
        })

        # ── 记忆系统 Token 占用（ChromaDB 所有 note 内容词数）────────────
        _memory_token_count = 0
        try:
            _memory_token_count = system.get_memory_token_count()
            result_metrics["memory_token_count"] = _memory_token_count
            _ingest_tokens = result_metrics.get("total_ingest_tokens", 0)
            result_metrics["memory_compression_ratio"] = round(
                _ingest_tokens / _memory_token_count, 3
            ) if _memory_token_count > 0 else None
        except Exception as _e:
            logger.debug(f"Memory token count failed: {_e}")

        # ── 系统统计（笔记数 / 链接数 / 抽象比）────────────────────────
        try:
            stats = await system.get_stats()
            _total_notes    = stats.get("total_notes", 0)
            _abstract_notes = stats.get("abstract_notes", 0)
            _total_links    = stats.get("total_links", 0)
            _interactions   = stats.get("interaction_count", 0)
            result_metrics["memory_stats"] = stats
            result_metrics["notes_per_turn"] = round(
                _total_notes / _total_turns, 3
            ) if _total_turns > 0 else None
            result_metrics["abstract_ratio"] = round(
                _abstract_notes / _total_notes, 3
            ) if _total_notes > 0 else None

            # ── 综合效率日志块 ─────────────────────────────────────────
            logger.info(
                f"\n{'='*70}\n"
                f"  [TRIAL-METRICS] {trial_id}\n"
                f"{'='*70}\n"
                f"  质量指标:\n"
                f"    综合分        = {_composite_score(result_metrics):.4f}\n"
                f"    Avg F1        = {result_metrics.get('avg_f1', 0):.4f}\n"
                f"    Avg ROUGE-L   = {result_metrics.get('avg_rouge_l', 0):.4f}\n"
                f"    Avg METEOR    = {result_metrics.get('avg_meteor', 0):.4f}\n"
                f"    Avg SBERT     = {result_metrics.get('avg_sbert', 0):.4f}\n"
                f"    ParseFailRate = {result_metrics.get('parse_fail_rate', 0):.2%}\n"
                f"  效率指标:\n"
                f"    Build Time    = {_build_time_h:.4f} h  ({_build_time_s:.1f} s)\n"
                f"    Ingest Rate   = {_ingest_tpm:.1f} turns/min\n"
                f"    QA Latency    = {result_metrics.get('avg_retrieval_latency_s', 0):.4f} s/query\n"
                f"    QA Throughput = {_qa_tput_qps:.3f} q/s\n"
                f"    Tokens/Query  = {result_metrics.get('avg_context_tokens', 0):.0f} tokens\n"
                f"  记忆系统:\n"
                f"    Total Notes   = {_total_notes}  (abstract={_abstract_notes})\n"
                f"    Total Links   = {_total_links}\n"
                f"    Notes/Turn    = {result_metrics.get('notes_per_turn', 0):.3f}\n"
                f"    Abstract Ratio= {result_metrics.get('abstract_ratio', 0):.1%}\n"
                f"    Ingest Tokens = {result_metrics.get('total_ingest_tokens', 0):,}\n"
                f"    Memory Tokens = {_memory_token_count:,}\n"
                f"    Compression   = {result_metrics.get('memory_compression_ratio', 'N/A')}x\n"
                f"    Peak Mem      = {result_metrics.get('peak_memory_mb', 0):.1f} MB\n"
                f"{'='*70}"
            )
        except Exception as e2:
            logger.debug(f"Stats collection failed: {e2}")

        await system.close()
        _trial_succeeded = True

    except Exception as e:
        logger.error(f"Trial {trial_id} failed: {e}", exc_info=True)
        result_metrics = {"error": str(e)}

    finally:
        for k, old_v in old_env.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v

        chroma_dir = base_data / "chroma_db"
        db_file = base_data / "phaseforget.db"
        if _trial_succeeded:
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
        else:
            logger.info(
                f"Trial {trial_id} 未完成，DB 已保留以支持断点续传: {base_data}"
            )

    elapsed = time.time() - start_time
    return {
        "trial_id": trial_id,
        "dataset": dataset,
        "data_path": data_path,
        "params": {
            "theta_sim": theta_sim,
            "theta_sum": theta_sum,
            "theta_evict": theta_evict,
            "decay_interval_rounds": decay_interval_rounds,
            "decay_factor": decay_factor,
            "eta": eta,
            "retrieval_top_k": retrieval_top_k,
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
    decay_factor_values: list[float],
    eta_values: list[float] | None = None,
    retrieval_top_k_values: list[int] | None = None,
) -> list[dict]:
    """生成笛卡尔积网格搜索参数组合列表。"""
    eta_vals = eta_values or DEFAULT_ETA_VALUES
    topk_vals = retrieval_top_k_values or DEFAULT_RETRIEVAL_TOP_K_VALUES
    combos = []
    for ts, tsum, te, decay_intv, df, eta, topk in itertools.product(
        theta_sim_values, theta_sum_values, theta_evict_values,
        decay_interval_values, decay_factor_values,
        eta_vals, topk_vals,
    ):
        combos.append({
            "theta_sim": ts,
            "theta_sum": tsum,
            "theta_evict": te,
            "decay_interval_rounds": decay_intv,
            "decay_factor": df,
            "eta": eta,
            "retrieval_top_k": topk,
        })
    return combos


def build_random(
    theta_sim_values: list[float],
    theta_sum_values: list[int],
    theta_evict_values: list[float],
    decay_interval_values: list[int],
    decay_factor_values: list[float],
    n_trials: int,
    seed: int = 42,
    eta_values: list[float] | None = None,
    retrieval_top_k_values: list[int] | None = None,
) -> list[dict]:
    """随机采样参数组合。"""
    rng = random.Random(seed)
    all_combos = build_grid(
        theta_sim_values, theta_sum_values, theta_evict_values,
        decay_interval_values, decay_factor_values,
        eta_values, retrieval_top_k_values,
    )
    if n_trials >= len(all_combos):
        return all_combos
    return rng.sample(all_combos, n_trials)


# ── 贝叶斯优化 (Optuna TPE) ─────────────────────────────────────────────────

async def _run_bayesian_search(
    n_trials: int,
    dataset: str,
    record_indices: list[int] | None,
    data_path: str,
    include_adversarial: bool,
    include_abstention: bool,
    disable_self_retrieval: bool,
    all_results: list[dict],
    seed: int = 42,
    ranges: dict[str, tuple] | None = None,
) -> list[dict]:
    """
    Optuna TPE 贝叶斯超参数优化。

    特性：
      - 自动将已有结果作为先验观测注入 study（热启动）
      - 使用 TPE sampler，对非平滑目标函数更鲁棒
      - 每完成一个 trial 立即保存结果
      - 支持自定义搜索范围

    Returns:
        新增的结果列表（也已追加到 all_results 中）。
    """
    try:
        import optuna
        from optuna.distributions import FloatDistribution, IntDistribution
    except ImportError:
        print("[ERROR] 贝叶斯搜索需要 optuna 库。请运行: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    r = ranges or BAYESIAN_RANGES

    distributions = {
        "theta_sim":             FloatDistribution(r["theta_sim"][0], r["theta_sim"][1]),
        "theta_sum":             IntDistribution(r["theta_sum"][0], r["theta_sum"][1]),
        "theta_evict":           FloatDistribution(r["theta_evict"][0], r["theta_evict"][1]),
        "eta":                   FloatDistribution(r["eta"][0], r["eta"][1]),
        "decay_factor":          FloatDistribution(r["decay_factor"][0], r["decay_factor"][1]),
        "decay_interval_rounds": IntDistribution(r["decay_interval_rounds"][0], r["decay_interval_rounds"][1]),
        "retrieval_top_k":       IntDistribution(r["retrieval_top_k"][0], r["retrieval_top_k"][1]),
    }

    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # ── 热启动：注入已有结果 ──────────────────────────────────────────────
    warm_count = 0
    for prev in all_results:
        if "error" in prev.get("metrics", {}):
            continue
        if not _result_matches_dataset(prev, dataset):
            continue
        if prev.get("record_indices") != record_indices:
            continue
        p = prev["params"]
        try:
            trial_params = {
                "theta_sim":             p["theta_sim"],
                "theta_sum":             p["theta_sum"],
                "theta_evict":           p["theta_evict"],
                "eta":                   p.get("eta", 0.1),
                "decay_factor":          p.get("decay_factor", 0.85),
                "decay_interval_rounds": p.get("decay_interval_rounds", 50),
                "retrieval_top_k":       p.get("retrieval_top_k", 10),
            }
            # 确保参数在合法范围内
            in_range = True
            for key, dist in distributions.items():
                val = trial_params[key]
                if val < dist.low or val > dist.high:
                    in_range = False
                    break
            if not in_range:
                continue

            study.add_trial(
                optuna.trial.create_trial(
                    params=trial_params,
                    distributions=distributions,
                    values=[prev["composite_score"]],
                )
            )
            warm_count += 1
        except Exception as e:
            logger.debug(f"Failed to inject warm-start trial: {e}")

    if warm_count > 0:
        print(f"[贝叶斯热启动] 已注入 {warm_count} 个历史观测点\n")

    # ── 搜索循环 ──────────────────────────────────────────────────────────
    new_results: list[dict] = []
    ri_str = "all" if record_indices is None else "_".join(map(str, sorted(record_indices)))

    for i in range(n_trials):
        trial = study.ask(distributions)
        params = trial.params

        theta_sim = round(params["theta_sim"], 3)
        theta_sum = params["theta_sum"]
        theta_evict = round(params["theta_evict"], 3)
        eta = round(params["eta"], 3)
        decay_factor = round(params["decay_factor"], 3)
        decay_interval = params["decay_interval_rounds"]
        topk = params["retrieval_top_k"]

        trial_id = (
            f"{dataset}_bay_{theta_sim:.2f}_{theta_sum}_{theta_evict:.2f}_"
            f"{eta:.2f}_{decay_factor:.2f}_{decay_interval}_{topk}_{ri_str}"
        )

        print(
            f"\n[Bayesian {i+1}/{n_trials}] "
            f"θ_sim={theta_sim} θ_sum={theta_sum} θ_evict={theta_evict} "
            f"η={eta} decay={decay_factor} intv={decay_interval} topK={topk}  "
            f"时间: {datetime.now().strftime('%H:%M:%S')}"
        )

        result = await run_single_trial(
            theta_sim=theta_sim,
            theta_sum=theta_sum,
            theta_evict=theta_evict,
            decay_interval_rounds=decay_interval,
            decay_factor=decay_factor,
            eta=eta,
            retrieval_top_k=topk,
            dataset=dataset,
            record_indices=record_indices,
            data_path=data_path,
            trial_id=trial_id,
            include_adversarial=include_adversarial,
            include_abstention=include_abstention,
            disable_self_retrieval=disable_self_retrieval,
        )

        score = result["composite_score"]
        study.tell(trial, score)

        all_results.append(result)
        new_results.append(result)
        _save_results(all_results)

        if "error" not in result.get("metrics", {}):
            m = result["metrics"]
            print(
                f"  综合分={score:.4f}  "
                f"F1={m.get('avg_f1', 0):.4f}  "
                f"ROUGE-L={m.get('avg_rouge_l', 0):.4f}  "
                f"METEOR={m.get('avg_meteor', 0):.4f}  "
                f"耗时={result['elapsed_seconds']}s"
            )
            print(
                f"    BuildTime={m.get('build_time_h', 0):.4f}h  "
                f"Latency={m.get('avg_retrieval_latency_s', 0):.4f}s  "
                f"Tokens/Q={m.get('avg_context_tokens', 0):.0f}  "
                f"MemTokens={m.get('memory_token_count', 0):,}  "
                f"IngestRate={m.get('ingest_throughput_tpm', 0):.1f}t/min  "
                f"PeakMem={m.get('peak_memory_mb', 0):.0f}MB"
            )
            by_cat = m.get("by_category", {})
            if by_cat:
                cat_summary = []
                for cat in sorted(by_cat.keys(), key=int):
                    cat_data = by_cat[cat]
                    cat_summary.append(f"C{cat}:F1={cat_data.get('avg_f1', 0):.3f}")
                print(f"    类别细分: {' | '.join(cat_summary)}")
        else:
            print(f"  [FAILED] {result['metrics']['error']}")

        # 打印当前最优
        if study.best_trial is not None:
            bp = study.best_params
            print(
                f"  [当前最优] score={study.best_value:.4f}  "
                f"θ_sim={bp['theta_sim']:.3f} θ_sum={bp['theta_sum']} "
                f"θ_evict={bp['theta_evict']:.3f} η={bp['eta']:.3f} "
                f"decay={bp['decay_factor']:.3f} intv={bp['decay_interval_rounds']} "
                f"topK={bp['retrieval_top_k']}"
            )

    # ── 打印 Optuna 参数重要度分析 ────────────────────────────────────────
    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"\n{'='*60}")
        print("  超参数重要度排名（fANOVA）")
        print(f"{'='*60}")
        for param, imp in importances.items():
            bar = "█" * int(imp * 40)
            print(f"  {param:<25} {imp:>6.1%}  {bar}")
        print(f"{'='*60}")
    except Exception as e:
        logger.debug(f"Importance analysis skipped: {e}")

    return new_results


def print_leaderboard(
    results: list[dict],
    top_n: int = 10,
    dataset: str | None = None,
) -> None:
    """打印超参数搜索结果排行榜（含全部 7 个参数）。"""
    scoped = _filter_results_for_dataset(results, dataset) if dataset else results
    valid = [r for r in scoped if "error" not in r.get("metrics", {})]
    if not valid:
        print("暂无有效结果。")
        return

    sorted_results = sorted(valid, key=lambda x: x["composite_score"], reverse=True)

    _LB_WIDTH = 185
    print("\n" + "=" * _LB_WIDTH)
    dataset_label = f" [{dataset}]" if dataset else ""
    print(f"  超参数搜索排行榜{dataset_label} (Top {min(top_n, len(sorted_results))})")
    print("=" * _LB_WIDTH)
    print(
        f"{'排名':<4} {'θ_sim':>6} {'θ_sum':>6} {'θ_evict':>8} {'η':>6} "
        f"{'decay':>6} {'intv':>5} {'topK':>5} "
        f"{'综合分':>8} {'F1':>8} {'ROUGE-L':>8} {'METEOR':>8} {'BLEU':>8} "
        f"{'样本数':>6} {'耗时(s)':>8} "
        f"{'BuildTime(h)':>13} {'Latency(s)':>11} {'Tokens/Q':>9} "
        f"{'MemTokens':>10} {'Ingest(t/m)':>11} {'PeakMem(MB)':>12}"
    )
    print("-" * _LB_WIDTH)

    for rank, r in enumerate(sorted_results[:top_n], 1):
        p = r["params"]
        m = r["metrics"]
        print(
            f"{rank:<4} "
            f"{p['theta_sim']:>6.2f} {p['theta_sum']:>6} {p['theta_evict']:>8.2f} "
            f"{p.get('eta', 0.1):>6.2f} "
            f"{p.get('decay_factor', 0.85):>6.2f} "
            f"{p.get('decay_interval_rounds', 50):>5} "
            f"{p.get('retrieval_top_k', 10):>5} "
            f"{r['composite_score']:>8.4f} {m.get('avg_f1', 0):>8.4f} "
            f"{m.get('avg_rouge_l', 0):>8.4f} {m.get('avg_meteor', 0):>8.4f} "
            f"{m.get('avg_bleu', 0):>8.4f} {m.get('n_questions', 0):>6} "
            f"{r.get('elapsed_seconds', 0):>8.1f} "
            f"{m.get('build_time_h', 0):>13.4f} "
            f"{m.get('avg_retrieval_latency_s', 0):>11.4f} "
            f"{m.get('avg_context_tokens', 0):>9.0f} "
            f"{m.get('memory_token_count', 0):>10,} "
            f"{m.get('ingest_throughput_tpm', 0):>11.1f} "
            f"{m.get('peak_memory_mb', 0):>12.1f}"
        )

    print("=" * _LB_WIDTH)

    if sorted_results:
        best = sorted_results[0]
        bp = best["params"]
        bm = best["metrics"]
        print(f"\n最佳超参数组合（综合分 {best['composite_score']:.4f}）：")
        print(f"  theta_sim            = {bp['theta_sim']}")
        print(f"  theta_sum            = {bp['theta_sum']}")
        print(f"  theta_evict          = {bp['theta_evict']}")
        print(f"  eta                  = {bp.get('eta', 0.1)}")
        print(f"  decay_factor         = {bp.get('decay_factor', 0.85)}")
        print(f"  decay_interval_rounds= {bp.get('decay_interval_rounds', 50)}")
        print(f"  retrieval_top_k      = {bp.get('retrieval_top_k', 10)}")

        print(f"\n最佳结果的效率指标：")
        print(f"  Build Time           = {bm.get('build_time_h', 0):.4f} h  ({bm.get('build_time_s', 0):.1f} s)")
        print(f"  Ingest Rate          = {bm.get('ingest_throughput_tpm', 0):.1f} turns/min")
        print(f"  QA Latency           = {bm.get('avg_retrieval_latency_s', 0):.4f} s/query  ({bm.get('avg_retrieval_time_us', 0):.1f} μs)")
        print(f"  QA Throughput        = {bm.get('qa_throughput_qps', 0):.3f} q/s")
        print(f"  Tokens/Query         = {bm.get('avg_context_tokens', 0):.0f} tokens (LLM context)")
        print(f"  Ingest Tokens        = {bm.get('total_ingest_tokens', 0):,}  (input to memory system)")
        print(f"  Memory Tokens        = {bm.get('memory_token_count', 0):,}  (stored in ChromaDB)")
        _comp = bm.get('memory_compression_ratio')
        print(f"  Compression Ratio    = {_comp:.3f}x" if _comp else f"  Compression Ratio    = N/A")
        _ms = bm.get("memory_stats", {})
        if _ms:
            print(f"  Total Notes          = {_ms.get('total_notes', '?')}  (abstract={_ms.get('abstract_notes', '?')})")
            print(f"  Total Links          = {_ms.get('total_links', '?')}")
            print(f"  Notes/Turn           = {bm.get('notes_per_turn', 'N/A')}")
            print(f"  Abstract Ratio       = {bm.get('abstract_ratio', 0):.1%}" if bm.get('abstract_ratio') is not None else f"  Abstract Ratio       = N/A")
        print(f"  Peak Memory          = {bm.get('peak_memory_mb', 0):.1f} MB")
        print(f"  Parse Fail Rate      = {bm.get('parse_fail_rate', 0):.2%}")

        print(f"\n对应的 .env 配置：")
        print(f"  THETA_SIM={bp['theta_sim']}")
        print(f"  THETA_SUM={bp['theta_sum']}")
        print(f"  THETA_EVICT={bp['theta_evict']}")
        print(f"  ETA={bp.get('eta', 0.1)}")
        print(f"  DECAY_FACTOR={bp.get('decay_factor', 0.85)}")
        print(f"  DECAY_INTERVAL_ROUNDS={bp.get('decay_interval_rounds', 50)}")
        print(f"  RETRIEVAL_TOP_K={bp.get('retrieval_top_k', 10)}")

        by_cat = bm.get("by_category", {})
        if by_cat:
            print(f"\n最佳结果的类别细分（Category Breakdown）：")
            print("-" * 80)
            dataset_key = best.get("dataset", "locomo")
            cat_names = DATASET_CATEGORY_LABELS.get(dataset_key, {})
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

        # 固定参数提醒
        print(f"\n[提示] 以下参数当前为固定值，未纳入搜索：")
        print(f"  u_init               = 0.5    （初始效用分，与 theta_evict 耦合）")
        print(f"  t_cool               = 3600s  （重整化冷却期）")
        print(f"  link_top_k           = 5      （创建链接 Top-K）")
        print(f"  projection_max_notes = 15     （重整化投影最大节点数）")
        print(f"  max_abstract_ratio   = 0.3    （检索中 Sigma 节点占比上限）")
        print(f"  llm_temperature      = 0.7    （答案生成温度，影响分数波动）")
        print()


async def main_async(args: argparse.Namespace) -> None:
    record_indices: list[int] | None = None
    if args.record_indices:
        try:
            record_indices = [int(x.strip()) for x in args.record_indices.split(",")]
        except ValueError:
            print(f"[ERROR] --record-indices 必须是逗号分隔的整数，收到: {args.record_indices}")
            sys.exit(1)
    if record_indices is not None and args.dataset not in DATASETS_WITH_RECORD_SELECTION:
        print(
            f"[ERROR] 数据集 {args.dataset} 不支持 --record-indices。"
            f"当前仅支持: {', '.join(sorted(DATASETS_WITH_RECORD_SELECTION))}"
        )
        sys.exit(1)

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
    decay_factor_values = (
        parse_floats(args.decay_factor_values)
        if args.decay_factor_values
        else DEFAULT_DECAY_FACTOR_VALUES
    )
    eta_values = parse_floats(args.eta_values) if args.eta_values else DEFAULT_ETA_VALUES
    retrieval_top_k_values = (
        parse_ints(args.retrieval_top_k_values)
        if args.retrieval_top_k_values
        else DEFAULT_RETRIEVAL_TOP_K_VALUES
    )

    all_results = _load_existing_results()
    dataset_results = _filter_results_for_dataset(all_results, args.dataset)

    # ── 贝叶斯搜索走独立路径 ──────────────────────────────────────────────
    if args.search_type == "bayesian":
        print(f"\n{'='*60}")
        print(f"  PhaseForget 贝叶斯超参数搜索 (Optuna TPE)")
        print(f"{'='*60}")
        print(f"  数据集       : {args.dataset}")
        print(f"  搜索次数     : {args.n_trials}")
        print(f"  随机种子     : {args.seed}")
        print(f"  数据集记录   : {record_indices if record_indices else '全部'}")
        print(f"  包含对抗题   : {args.include_adversarial}")
        print(f"  包含拒答题   : {args.include_abstention}")
        print(f"  关闭自检索   : {args.disable_self_retrieval}")
        print(f"  数据集路径   : {args.data_path}")
        print(f"  结果保存至   : {RESULTS_PATH}")
        print(f"  已有结果     : {len(dataset_results)} 条")

        ranges = dict(BAYESIAN_RANGES)
        if args.theta_sim_values:
            vals = parse_floats(args.theta_sim_values)
            ranges["theta_sim"] = (min(vals), max(vals))
        if args.theta_sum_values:
            vals = parse_ints(args.theta_sum_values)
            ranges["theta_sum"] = (min(vals), max(vals))
        if args.theta_evict_values:
            vals = parse_floats(args.theta_evict_values)
            ranges["theta_evict"] = (min(vals), max(vals))
        if args.eta_values:
            vals = parse_floats(args.eta_values)
            ranges["eta"] = (min(vals), max(vals))
        if args.decay_factor_values:
            vals = parse_floats(args.decay_factor_values)
            ranges["decay_factor"] = (min(vals), max(vals))
        if args.decay_interval_values:
            vals = parse_ints(args.decay_interval_values)
            ranges["decay_interval_rounds"] = (min(vals), max(vals))
        if args.retrieval_top_k_values:
            vals = parse_ints(args.retrieval_top_k_values)
            ranges["retrieval_top_k"] = (min(vals), max(vals))

        print(f"\n  搜索范围:")
        for k, (lo, hi) in sorted(ranges.items()):
            print(f"    {k:<25} [{lo}, {hi}]")
        print(f"{'='*60}\n")

        await _run_bayesian_search(
            n_trials=args.n_trials,
            dataset=args.dataset,
            record_indices=record_indices,
            data_path=args.data_path,
            include_adversarial=args.include_adversarial,
            include_abstention=args.include_abstention,
            disable_self_retrieval=args.disable_self_retrieval,
            all_results=all_results,
            seed=args.seed,
            ranges=ranges,
        )

        print("\n所有实验完成！")
        print_leaderboard(all_results, dataset=args.dataset)
        print(f"完整结果已保存到: {RESULTS_PATH}\n")
        return

    # ── Grid / Random 搜索 ────────────────────────────────────────────────
    if args.search_type == "grid":
        combos = build_grid(
            theta_sim_values, theta_sum_values, theta_evict_values,
            decay_interval_values, decay_factor_values,
            eta_values, retrieval_top_k_values,
        )
    else:
        combos = build_random(
            theta_sim_values, theta_sum_values, theta_evict_values,
            decay_interval_values, decay_factor_values,
            n_trials=args.n_trials, seed=args.seed,
            eta_values=eta_values,
            retrieval_top_k_values=retrieval_top_k_values,
        )

    total = len(combos)
    print(f"\n{'='*60}")
    print(f"  PhaseForget 超参数搜索")
    print(f"{'='*60}")
    print(f"  数据集       : {args.dataset}")
    print(f"  搜索模式     : {args.search_type}")
    print(f"  参数组合数   : {total}")
    print(f"  theta_sim    : {theta_sim_values}")
    print(f"  theta_sum    : {theta_sum_values}")
    print(f"  theta_evict  : {theta_evict_values}")
    print(f"  eta          : {eta_values}")
    print(f"  decay_intv   : {decay_interval_values}")
    print(f"  decay_factor : {decay_factor_values}")
    print(f"  top_k        : {retrieval_top_k_values}")
    print(f"  数据集记录   : {record_indices if record_indices else '全部'}")
    print(f"  包含对抗题   : {args.include_adversarial}")
    print(f"  包含拒答题   : {args.include_abstention}")
    print(f"  关闭自检索   : {args.disable_self_retrieval}")
    print(f"  数据集路径   : {args.data_path}")
    print(f"  结果保存至   : {RESULTS_PATH}")
    print(f"{'='*60}\n")

    completed_keys = {
        (
            r["params"]["theta_sim"],
            r["params"]["theta_sum"],
            r["params"]["theta_evict"],
            r["params"].get("decay_interval_rounds", DEFAULT_DECAY_INTERVAL_VALUES[0]),
            r["params"].get("decay_factor", DEFAULT_DECAY_FACTOR_VALUES[0]),
            r["params"].get("eta", 0.1),
            r["params"].get("retrieval_top_k", 10),
        )
        for r in all_results
        if _result_matches_dataset(r, args.dataset) and r.get("record_indices") == record_indices
    }

    pending = [
        c for c in combos
        if (
            c["theta_sim"],
            c["theta_sum"],
            c["theta_evict"],
            c["decay_interval_rounds"],
            c["decay_factor"],
            c["eta"],
            c["retrieval_top_k"],
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
            print(
                f"    BuildTime={m.get('build_time_h', 0):.4f}h  "
                f"Latency={m.get('avg_retrieval_latency_s', 0):.4f}s  "
                f"Tokens/Q={m.get('avg_context_tokens', 0):.0f}  "
                f"MemTokens={m.get('memory_token_count', 0):,}  "
                f"IngestRate={m.get('ingest_throughput_tpm', 0):.1f}t/min  "
                f"PeakMem={m.get('peak_memory_mb', 0):.0f}MB"
            )
            by_cat = m.get("by_category", {})
            if by_cat:
                cat_summary = []
                for cat in sorted(by_cat.keys(), key=int):
                    cat_data = by_cat[cat]
                    cat_summary.append(f"C{cat}:F1={cat_data.get('avg_f1', 0):.3f}")
                print(f"    类别细分: {' | '.join(cat_summary)}")

        tasks: list[tuple[int, dict[str, Any], str]] = []
        ri_str = "all" if record_indices is None else "_".join(map(str, sorted(record_indices)))
        for trial_num, combo in enumerate(pending, skipped + 1):
            ts = combo["theta_sim"]
            tsum = combo["theta_sum"]
            te = combo["theta_evict"]
            decay_intv = combo["decay_interval_rounds"]
            df = combo["decay_factor"]
            eta = combo["eta"]
            topk = combo["retrieval_top_k"]
            trial_id = (
                f"{args.dataset}_{ts:.2f}_{tsum}_{te:.2f}_{decay_intv}_{df:.2f}_"
                f"{eta:.2f}_{topk}_{ri_str}"
            )
            tasks.append((trial_num, combo, trial_id))

        if max_parallel == 1:
            for trial_num, combo, trial_id in tasks:
                print(
                    f"[{trial_num}/{total}] θ_sim={combo['theta_sim']} θ_sum={combo['theta_sum']} "
                    f"θ_evict={combo['theta_evict']} η={combo['eta']} "
                    f"decay={combo['decay_factor']} intv={combo['decay_interval_rounds']} "
                    f"topK={combo['retrieval_top_k']}  "
                    f"开始时间: {datetime.now().strftime('%H:%M:%S')}"
                )
                result = await run_single_trial(
                    theta_sim=combo["theta_sim"],
                    theta_sum=combo["theta_sum"],
                    theta_evict=combo["theta_evict"],
                    decay_interval_rounds=combo["decay_interval_rounds"],
                    decay_factor=combo["decay_factor"],
                    eta=combo["eta"],
                    retrieval_top_k=combo["retrieval_top_k"],
                    dataset=args.dataset,
                    record_indices=record_indices,
                    data_path=args.data_path,
                    trial_id=trial_id,
                    include_adversarial=args.include_adversarial,
                    include_abstention=args.include_abstention,
                    disable_self_retrieval=args.disable_self_retrieval,
                )
                all_results.append(result)
                _save_results(all_results)
                print_trial_summary(result)
        else:
            ordered_results: dict[int, tuple[int, dict[str, Any], dict[str, Any]]] = {}
            next_to_flush = 0

            with ProcessPoolExecutor(max_workers=max_parallel) as executor:
                future_to_idx = {}
                for idx, (trial_num, combo, trial_id) in enumerate(tasks):
                    print(
                        f"[{trial_num}/{total}] θ_sim={combo['theta_sim']} θ_sum={combo['theta_sum']} "
                        f"θ_evict={combo['theta_evict']} η={combo['eta']} "
                        f"decay={combo['decay_factor']} intv={combo['decay_interval_rounds']} "
                        f"topK={combo['retrieval_top_k']}  "
                        f"已提交  时间: {datetime.now().strftime('%H:%M:%S')}"
                    )
                    payload = {
                        "theta_sim": combo["theta_sim"],
                        "theta_sum": combo["theta_sum"],
                        "theta_evict": combo["theta_evict"],
                        "decay_interval_rounds": combo["decay_interval_rounds"],
                        "decay_factor": combo["decay_factor"],
                        "eta": combo["eta"],
                        "retrieval_top_k": combo["retrieval_top_k"],
                        "dataset": args.dataset,
                        "record_indices": record_indices,
                        "data_path": args.data_path,
                        "trial_id": trial_id,
                        "include_adversarial": args.include_adversarial,
                        "include_abstention": args.include_abstention,
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
                        result = {
                            "trial_id": "unknown",
                            "dataset": args.dataset,
                            "data_path": args.data_path,
                            "params": {
                                "theta_sim": combo["theta_sim"],
                                "theta_sum": combo["theta_sum"],
                                "theta_evict": combo["theta_evict"],
                                "decay_interval_rounds": combo["decay_interval_rounds"],
                                "decay_factor": combo["decay_factor"],
                                "eta": combo["eta"],
                                "retrieval_top_k": combo["retrieval_top_k"],
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
                        print(
                            f"[{flush_trial_num}/{total}] θ_sim={flush_combo['theta_sim']} "
                            f"θ_sum={flush_combo['theta_sum']} θ_evict={flush_combo['theta_evict']} "
                            f"η={flush_combo['eta']} decay={flush_combo['decay_factor']} "
                            f"intv={flush_combo['decay_interval_rounds']} "
                            f"topK={flush_combo['retrieval_top_k']}  "
                            f"已完成  时间: {datetime.now().strftime('%H:%M:%S')}"
                        )
                        all_results.append(flush_result)
                        _save_results(all_results)
                        print_trial_summary(flush_result)
                        next_to_flush += 1

    print("\n所有实验完成！")
    print_leaderboard(all_results, dataset=args.dataset)
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
        "--dataset",
        choices=sorted(DEFAULT_DATA_PATHS.keys()),
        default="locomo",
        help="要运行搜索的数据集（默认: locomo）",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="数据集路径（不指定时按 --dataset 自动选择默认路径）",
    )
    parser.add_argument(
        "--record-indices",
        type=str,
        default=None,
        help=(
            "逗号分隔的记录索引，例如 '0,1,2'。"
            "仅 locomo / longmemeval 支持；不指定则使用全部记录。"
        ),
    )
    parser.add_argument(
        "--include-adversarial",
        action="store_true",
        help="包含 LoCoMo category=5 的对抗问题（默认不包含）。",
    )
    parser.add_argument(
        "--include-abstention",
        action="store_true",
        help="包含 LongMemEval abstention 拒答题（默认不包含）。",
    )
    parser.add_argument(
        "--disable-self-retrieval",
        action="store_true",
        help="关闭对话写入阶段的自检索（self-retrieval），用于对齐 A-MEM 行为。",
    )

    # ── 搜索策略 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--search-type",
        choices=["grid", "random", "bayesian"],
        default="grid",
        help="搜索类型：grid=网格搜索，random=随机搜索，bayesian=Optuna TPE（默认: grid）",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=25,
        help="随机/贝叶斯搜索的试验次数（默认: 25）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )

    # ── 超参数搜索空间 ────────────────────────────────────────────────────
    # 对 grid/random：这些值是离散候选列表
    # 对 bayesian：如果提供，将用 min/max 覆盖默认连续范围
    parser.add_argument(
        "--theta-sim-values", type=str, default=None,
        help=f"theta_sim 候选值（默认: {','.join(map(str, DEFAULT_THETA_SIM_VALUES))}）",
    )
    parser.add_argument(
        "--theta-sum-values", type=str, default=None,
        help=f"theta_sum 候选值（默认: {','.join(map(str, DEFAULT_THETA_SUM_VALUES))}）",
    )
    parser.add_argument(
        "--theta-evict-values", type=str, default=None,
        help=f"theta_evict 候选值（默认: {','.join(map(str, DEFAULT_THETA_EVICT_VALUES))}）",
    )
    parser.add_argument(
        "--eta-values", type=str, default=None,
        help=f"eta 候选值（默认: {','.join(map(str, DEFAULT_ETA_VALUES))}）",
    )
    parser.add_argument(
        "--decay-interval-values", type=str, default=None,
        help=f"decay_interval_rounds 候选值（默认: {','.join(map(str, DEFAULT_DECAY_INTERVAL_VALUES))}）",
    )
    parser.add_argument(
        "--decay-factor-values", type=str, default=None,
        help=f"decay_factor 候选值（默认: {','.join(map(str, DEFAULT_DECAY_FACTOR_VALUES))}）",
    )
    parser.add_argument(
        "--retrieval-top-k-values", type=str, default=None,
        help=f"retrieval_top_k 候选值（默认: {','.join(map(str, DEFAULT_RETRIEVAL_TOP_K_VALUES))}）",
    )

    # ── 工具命令 ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--show-results", action="store_true",
        help="只显示已有的搜索结果排行榜，不运行新实验",
    )
    parser.add_argument(
        "--clear-results", action="store_true",
        help="清除已保存的搜索结果（谨慎使用）",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="自定义结果文件保存路径（默认: data/hparam_search_results.json）",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=1,
        help="并行执行的最大进程数（默认: 1=串行，仅 grid/random 有效）",
    )

    args = parser.parse_args()
    if not args.data_path:
        args.data_path = DEFAULT_DATA_PATHS[args.dataset]

    global RESULTS_PATH
    if args.output:
        RESULTS_PATH = Path(args.output)
        if not RESULTS_PATH.is_absolute():
            RESULTS_PATH = _ROOT / RESULTS_PATH

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
            print_leaderboard(results, dataset=args.dataset)
        return

    os.chdir(_ROOT)
    _ensure_model_cached()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

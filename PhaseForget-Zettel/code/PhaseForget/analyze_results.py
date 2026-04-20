"""
分析 sum_a1_a2.json 超参数搜索结果
关注: memory_stats, F1/BLEU, abstract_notes=0 异常, 最佳超参数组合
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "sum_a1_a2.json"

data = json.load(open(DATA_PATH, encoding="utf-8"))
print(f"共 {len(data)} 组实验\n")

# ── 展平为 DataFrame ───────────────────────────────────────────────────────────
rows = []
for t in data:
    p = t["params"]
    m = t["metrics"]
    ms = m["memory_stats"]
    bc = m["by_category"]
    row = {
        "trial_id": t["trial_id"],
        "theta_sim": p["theta_sim"],
        "theta_sum": p["theta_sum"],
        "theta_evict": p["theta_evict"],
        "decay_interval_rounds": p["decay_interval_rounds"],
        # 整体指标
        "avg_f1": m["avg_f1"],
        "avg_bleu": m["avg_bleu"],
        "avg_rouge_l": m["avg_rouge_l"],
        "avg_rouge2": m["avg_rouge2"],
        "avg_meteor": m["avg_meteor"],
        "avg_sbert": m["avg_sbert"],
        "composite_score": t["composite_score"],
        # 分类指标
        **{f"cat{c}_f1":   bc[c]["avg_f1"]   for c in bc},
        **{f"cat{c}_bleu": bc[c]["avg_bleu"]  for c in bc},
        # 记忆统计
        "total_notes": ms["total_notes"],
        "abstract_notes": ms["abstract_notes"],
        "total_links": ms["total_links"],
        "interaction_count": ms["interaction_count"],
        "cold_track_count": ms["cold_track_count"],
        # 辅助
        "abstract_ratio": ms["abstract_notes"] / max(ms["total_notes"], 1),
        "links_per_note": ms["total_links"] / max(ms["total_notes"], 1),
        "elapsed_s": t["elapsed_seconds"],
    }
    rows.append(row)

df = pd.DataFrame(rows)

# ── 1. 概览 ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("【1】整体分布概览")
print("=" * 60)
cols_show = ["avg_f1", "avg_bleu", "composite_score",
             "total_notes", "abstract_notes", "abstract_ratio", "total_links"]
print(df[cols_show].describe().round(4).to_string())

# ── 2. abstract_notes=0 异常分析 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("【2】abstract_notes = 0 的异常情况")
print("=" * 60)
zero_abs = df[df["abstract_notes"] == 0]
nonzero_abs = df[df["abstract_notes"] > 0]
print(f"abstract_notes=0  的试验数: {len(zero_abs)} / {len(df)} ({100*len(zero_abs)/len(df):.1f}%)")
print(f"abstract_notes>0  的试验数: {len(nonzero_abs)}")

print("\n--- 异常组 vs 正常组 指标对比 ---")
for col in ["avg_f1", "avg_bleu", "composite_score", "total_notes", "total_links"]:
    print(f"  {col:22s}  零抽象={zero_abs[col].mean():.4f}  非零抽象={nonzero_abs[col].mean():.4f}")

print("\n--- abstract_notes=0 时，各超参的分布 ---")
for param in ["theta_sim", "theta_sum", "theta_evict", "decay_interval_rounds"]:
    vc = zero_abs[param].value_counts().sort_index()
    print(f"  {param}: {dict(vc)}")

print("\n--- theta_sum 与 abstract_notes 的关系（均值）---")
print(df.groupby("theta_sum")["abstract_notes"].agg(["mean", "min", "max", "count"]).to_string())

# ── 3. 各超参对核心指标的影响 ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("【3】各超参对核心指标的影响（均值）")
print("=" * 60)
params = ["theta_sim", "theta_sum", "theta_evict", "decay_interval_rounds"]
kpi = ["avg_f1", "avg_bleu", "composite_score", "abstract_notes", "abstract_ratio"]
for p in params:
    print(f"\n  -- {p} --")
    print(df.groupby(p)[kpi].mean().round(4).to_string())

# ── 4. 过滤 abstract_notes>0 后的最佳组合 ─────────────────────────────────────
print("\n" + "=" * 60)
print("【4】过滤 abstract_notes>0 后的 Top-15（按 composite_score 降序）")
print("=" * 60)
valid = df[df["abstract_notes"] > 0].copy()
valid_sorted = valid.sort_values("composite_score", ascending=False)
top_cols = ["trial_id", "theta_sim", "theta_sum", "theta_evict",
            "decay_interval_rounds", "avg_f1", "avg_bleu", "composite_score",
            "abstract_notes", "abstract_ratio", "total_notes", "total_links"]
print(valid_sorted[top_cols].head(15).to_string(index=False))

# ── 5. 全量 Top-15（含 abstract=0）──────────────────────────────────────────
print("\n" + "=" * 60)
print("【5】全量 Top-15（含 abstract=0，按 composite_score）")
print("=" * 60)
all_sorted = df.sort_values("composite_score", ascending=False)
print(all_sorted[top_cols].head(15).to_string(index=False))

# ── 6. 分类别 F1 分析 ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("【6】分类别 F1 分析（valid 组，各类均值）")
print("=" * 60)
cat_cols = [c for c in df.columns if c.startswith("cat") and c.endswith("_f1")]
print("类别说明: cat1=单跳简单, cat2=单跳复杂, cat3=多跳, cat4=摘要/总结（猜测）")
for p in params:
    print(f"\n  -- {p} --")
    print(valid.groupby(p)[cat_cols].mean().round(4).to_string())

# ── 7. 相关性热力图数据 ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("【7】超参数 & 记忆统计 与核心指标的 Pearson 相关系数")
print("=" * 60)
feat_cols = ["theta_sim", "theta_sum", "theta_evict", "decay_interval_rounds",
             "total_notes", "abstract_notes", "abstract_ratio", "total_links", "links_per_note"]
tgt_cols = ["avg_f1", "avg_bleu", "avg_sbert", "composite_score"]
corr = df[feat_cols + tgt_cols].corr()[tgt_cols].loc[feat_cols]
print(corr.round(4).to_string())

# ── 8. abstract_notes=0 的直接原因假设 ────────────────────────────────────────
print("\n" + "=" * 60)
print("【8】theta_sum 阈值与 abstract_notes 关系（关键！）")
print("=" * 60)
pivot = df.pivot_table(
    values="abstract_notes", index="theta_sum",
    columns="theta_sim", aggfunc="mean"
).round(2)
print("  行=theta_sum, 列=theta_sim, 值=abstract_notes 均值")
print(pivot.to_string())

print("\n  行=theta_sum, 列=theta_evict, 值=abstract_notes 均值")
pivot2 = df.pivot_table(
    values="abstract_notes", index="theta_sum",
    columns="theta_evict", aggfunc="mean"
).round(2)
print(pivot2.to_string())

# ── 9. 推荐最佳超参 ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("【9】推荐最佳超参（valid 组，多指标综合）")
print("=" * 60)
# 归一化后加权
valid2 = valid.copy()
for col in ["avg_f1", "avg_bleu", "avg_sbert"]:
    valid2[col + "_norm"] = (valid2[col] - valid2[col].min()) / (valid2[col].max() - valid2[col].min() + 1e-9)
valid2["abstract_ratio_norm"] = (valid2["abstract_ratio"] - valid2["abstract_ratio"].min()) / \
                                 (valid2["abstract_ratio"].max() - valid2["abstract_ratio"].min() + 1e-9)
valid2["score_custom"] = (
    0.4 * valid2["avg_f1_norm"] +
    0.2 * valid2["avg_bleu_norm"] +
    0.2 * valid2["avg_sbert_norm"] +
    0.2 * valid2["abstract_ratio_norm"]
)
best = valid2.sort_values("score_custom", ascending=False).head(10)
print(best[["theta_sim", "theta_sum", "theta_evict", "decay_interval_rounds",
            "avg_f1", "avg_bleu", "avg_sbert", "abstract_notes", "abstract_ratio",
            "score_custom"]].to_string(index=False))

print("\n[完成] 分析结束\n")

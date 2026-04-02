#!/usr/bin/env python3
"""分析 baseline_check_changge_deacy.json 中特定参数组合的实验结果"""

import json
import sys

def find_matching_results(filepath, target_params):
    """查找匹配指定参数的所有实验结果"""

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    matching_results = []

    for entry in data:
        params = entry.get('params', {})

        # 检查是否匹配所有目标参数
        match = all(
            params.get(key) == value
            for key, value in target_params.items()
        )

        if match:
            matching_results.append(entry)

    return matching_results

def format_result(result, index=1):
    """格式化输出单个实验结果"""

    params = result.get('params', {})
    metrics = result.get('metrics', {})
    by_category = metrics.get('by_category', {})

    output = []
    output.append(f"\n{'='*70}")
    output.append(f"匹配结果 #{index}")
    output.append(f"{'='*70}")

    # 基本信息
    output.append(f"\n[基本信息]")
    output.append(f"  trial_id: {result.get('trial_id', 'N/A')}")
    output.append(f"  record_indices: {result.get('record_indices', 'N/A')}")
    output.append(f"  timestamp: {result.get('timestamp', 'N/A')}")
    output.append(f"  elapsed_seconds: {result.get('elapsed_seconds', 'N/A'):.1f}")

    # 参数
    output.append(f"\n[参数配置]")
    for key, value in params.items():
        output.append(f"  {key}: {value}")

    # 总体指标
    output.append(f"\n[总体指标]")
    output.append(f"  composite_score (综合得分): {result.get('composite_score', 'N/A')}")
    output.append(f"  avg_f1:     {metrics.get('avg_f1', 'N/A'):.4f}")
    output.append(f"  avg_bleu:   {metrics.get('avg_bleu', 'N/A'):.4f}")
    output.append(f"  avg_rouge_l: {metrics.get('avg_rouge_l', 'N/A'):.4f}")
    output.append(f"  avg_rouge2: {metrics.get('avg_rouge2', 'N/A'):.4f}")
    output.append(f"  avg_meteor: {metrics.get('avg_meteor', 'N/A'):.4f}")
    output.append(f"  avg_sbert:  {metrics.get('avg_sbert', 'N/A'):.2f}")
    output.append(f"  avg_retrieval_time_us: {metrics.get('avg_retrieval_time_us', 'N/A'):.2f}")
    output.append(f"  n_questions: {metrics.get('n_questions', 'N/A')}")

    # 分类指标
    category_names = {
        "1": "Single-hop",
        "2": "Temporal",
        "3": "Multi-hop",
        "4": "Open Domain"
    }

    output.append(f"\n[分类指标]")
    for cat_id, cat_metrics in sorted(by_category.items(), key=lambda x: int(x[0])):
        cat_name = category_names.get(cat_id, f"Category {cat_id}")
        output.append(f"\n  {cat_name} (n={cat_metrics.get('n_questions', 'N/A')}):")
        output.append(f"    F1:    {cat_metrics.get('avg_f1', 'N/A'):.4f}")
        output.append(f"    BLEU:  {cat_metrics.get('avg_bleu', 'N/A'):.4f}")
        output.append(f"    ROUGE-L: {cat_metrics.get('avg_rouge_l', 'N/A'):.4f}")
        output.append(f"    SBERT: {cat_metrics.get('avg_sbert', 'N/A'):.2f}")

    return '\n'.join(output)

def main():
    filepath = "data/baseline_check_changge_deacy.json"

    # 用户选中的参数组合
    target_params = {
        "theta_sim": 0.7,
        "theta_sum": 40,
        "theta_evict": 0.45,
        "decay_interval_rounds": 80
    }

    print(f"正在查找匹配的参数组合...")
    print(f"目标参数: {target_params}")

    results = find_matching_results(filepath, target_params)

    if not results:
        print(f"\n[!] 未找到匹配的实验结果")
        return

    print(f"\n[OK] 找到 {len(results)} 条匹配的实验结果")

    for i, result in enumerate(results, 1):
        print(format_result(result, i))

    # 如果有多个结果，输出统计对比
    if len(results) > 1:
        print(f"\n{'='*70}")
        print("多结果对比")
        print(f"{'='*70}")

        for i, result in enumerate(results, 1):
            params = result.get('params', {})
            print(f"\n结果 #{i}: {result.get('trial_id')}")
            print(f"  综合得分: {result.get('composite_score', 'N/A'):.4f}")
            print(f"  F1: {result.get('metrics', {}).get('avg_f1', 'N/A'):.4f}")

if __name__ == "__main__":
    main()

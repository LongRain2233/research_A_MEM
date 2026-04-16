"""
结果导出脚本
用法：
  python export_results.py --input data/fix_forget.json --output data/results_export.csv
  python export_results.py --input data/fix_forget.json --output data/results_report.txt --format txt
  python export_results.py --input data/fix_forget.json --output data/results_summary.json --format summary
"""

import argparse
import json
import csv
from pathlib import Path


def export_csv(data, output_path):
    """导出为 CSV 格式"""
    fieldnames = [
        'trial_id',
        'theta_sim',
        'theta_sum',
        'theta_evict',
        'decay_interval_rounds',
        'composite_score',
        'avg_f1',
        'avg_bleu',
        'avg_rouge_l',
        'avg_rouge2',
        'avg_meteor',
        'avg_sbert',
        'avg_retrieval_time_us',
        'n_questions',
        'elapsed_seconds',
        'timestamp',
    ]

    # 添加类别指标
    categories = {}
    for r in data:
        if 'by_category' in r.get('metrics', {}):
            for cat in r['metrics']['by_category'].keys():
                if cat not in categories:
                    categories[cat] = []
                    for metric in ['avg_f1', 'avg_bleu', 'avg_rouge_l', 'avg_meteor', 'n_questions']:
                        fieldnames.append(f'cat_{cat}_{metric}')

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in data:
            params = r.get('params', {})
            metrics = r.get('metrics', {})

            row = {
                'trial_id': r.get('trial_id', ''),
                'theta_sim': params.get('theta_sim', ''),
                'theta_sum': params.get('theta_sum', ''),
                'theta_evict': params.get('theta_evict', ''),
                'decay_interval_rounds': params.get('decay_interval_rounds', ''),
                'composite_score': r.get('composite_score', ''),
                'avg_f1': metrics.get('avg_f1', ''),
                'avg_bleu': metrics.get('avg_bleu', ''),
                'avg_rouge_l': metrics.get('avg_rouge_l', ''),
                'avg_rouge2': metrics.get('avg_rouge2', ''),
                'avg_meteor': metrics.get('avg_meteor', ''),
                'avg_sbert': metrics.get('avg_sbert', ''),
                'avg_retrieval_time_us': metrics.get('avg_retrieval_time_us', ''),
                'n_questions': metrics.get('n_questions', ''),
                'elapsed_seconds': r.get('elapsed_seconds', ''),
                'timestamp': r.get('timestamp', ''),
            }

            # 添加类别指标
            if 'by_category' in metrics:
                for cat, cat_metrics in metrics['by_category'].items():
                    for metric in ['avg_f1', 'avg_bleu', 'avg_rouge_l', 'avg_meteor', 'n_questions']:
                        row[f'cat_{cat}_{metric}'] = cat_metrics.get(metric, '')

            writer.writerow(row)

    print(f"CSV 导出完成: {output_path}")


def export_txt(data, output_path):
    """导出为文本报告"""
    cat_names = {
        "1": "Single-hop (单跳)",
        "2": "Temporal (时间)",
        "3": "Multi-hop (多跳)",
        "4": "Open-domain (开放域)",
        "5": "Adversarial (对抗性)",
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("  PhaseForget 超参数搜索结果报告\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"总记录数: {len(data)}\n")

        # 排除错误的结果
        valid = [r for r in data if "error" not in r.get("metrics", {})]
        f.write(f"有效记录数: {len(valid)}\n")
" + "\n")

        if not valid:
            f.write("没有有效的实验结果。\n")
            return

        # 按综合分数排序
        sorted_results = sorted(valid, key=lambda x: x["composite_score"], reverse=True)

        # 排行榜
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"  实验结果排行榜 (Top {min(20, len(sorted_results))})\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'排名':<4} {'trial_id':<30} {'theta_sim':>10} {'theta_sum':>8} "
                f"{'theta_evict':>12} {'decay':>6} {'综合分':>8} "
                f"{'F1':>8} {'ROUGE-L':>8} {'METEOR':>8} {'BLEU':>8} "
                f"{'样本数':>6} {'耗时(s)':>8}\n")
        f.write("-" * 100 + "\n")

        for rank, r in enumerate(sorted_results[:20], 1):
            p = r["params"]
            m = r["metrics"]
            f.write(
                f"{rank:<4} {r['trial_id'][:30]:<30} {p['theta_sim']:>10.2f} {p['theta_sum']:>8} "
"
                f"{p['theta_evict']:>12.2f} {p.get('decay_interval_rounds', '-'):>6} "
                f"{r['composite_score']:>8.4f} {m.get('avg_f1', 0):>8.4f} "
                f"{m.get('avg_rouge_l', 0):>8.4f} {m.get('avg_meteor', 0):>8.4f} "
                f"{m.get('avg_bleu', 0):>8.4f} {m.get('n_questions', 0):>6} "
                f"{r.get('elapsed_seconds', 0):>8.1f}\n"
            )

        # 最佳结果
        f.write("\n" + "=" * 100 + "\n")
        f.write("  最佳实验结果\n")
        f.write("=" * 100 + "\n")
        best = sorted_results[0]
        bp = best["params"]
        bm = best["metrics"]
        f.write(f"trial_id: {best['trial_id']}\n")
        f.write(f"综合分数: {best['composite_score']:.4f}\n")
        f.write(f"theta_sim: {bp['theta_sim']}\n")
        f.write(f"theta_sum: {bp['theta_sum']}\n")
        f.write(f"theta_evict: {bp['theta_evict']}\n")
        f.write(f"decay_interval_rounds: {bp.get('decay_interval_rounds', 'N/A')}\n")
        f.write(f"avg_f1: {bm.get('avg_f1', 'N/A'):.4f}\n")
        f.write(f"avg_rouge_l: {bm.get('avg_rouge_l', 'N/A'):.4f}\n")
        f.write(f"avg_meteor: {bm.get('avg_meteor', 'N/A'):.4f}\n")
        f.write(f"avg_bleu: {bm.get('avg_bleu', 'N/A'):.4f}\n")
        f.write(f"avg_sbert: {bm.get('avg_sbert', 'N/A'):.4f}\n")
        f.write(f"n_questions: {bm.get('n_questions', 'N/A')}\n")
        f.write(f"elapsed_seconds: {best.get('elapsed_seconds', 'N/A')}\n")
        f.write(f"timestamp: {best.get('timestamp', 'N/A')}\n")

        # 类别细分
        by_cat = bm.get("by_category", {})
        if by_cat:
            f.write("\n" + "-" * 80 + "\n")
            f.write("类别细分 (Category Breakdown):\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'类别':<25} {'F1':>8} {'ROUGE-L':>8} {'METEOR':>8} {'BLEU':>8} {'样本数':>6}\n")
            f.write("-" * 80 + "\n")
            for cat in sorted(by_cat.keys(), key=int):
                cat_data = by_cat[cat]
                cat_name = cat_names.get(cat, f"Category {cat}")
                f.write(
                    f"{cat_name:<25} "
                    f"{cat_data.get('avg_f1', 0):>8.4f} "
                    f"{cat_data.get('avg_rouge_l', 0):>8.4f} "
                    f"{cat_data.get('avg_meteor', 0):>8.4f} "
                    f"{cat_data.get('avg_bleu', 0):>8.4f} "
                    f"{cat_data.get('n_questions', 0):>6}\n"
                )

        # 所有实验结果
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"  所有实验结果 ({len(sorted_results)} 条)\n")
        f.write("=" * 100 + "\n")

        for rank, r in enumerate(sorted_results, 1):
            p = r["params"]
            m = r["metrics"]
            f.write(f"\n[{rank}] trial_id: {r['trial_id']}\n")
            f.write(f"    参数: theta_sim={p['theta_sim']}, theta_sum={p['theta_sum']}, "
                   f"theta_evict={p['theta_evict']}, decay={p.get('decay_interval_rounds', 'N/A')}\n")
            f.write(f"    综合分数: {r['composite_score']:.4f}\n")
            f.write(f"    指标: F1={m.get('avg_f1', 0):.4f}, "
                   f"ROUGE-L={m.get('avg_rouge_l', 0):.4f}, "
                   f"METEOR={m.get('avg_meteor', 0):.4f}, "
                   f"BLEU={m.get('avg_bleu', 0):.4f}, "
                   f"SBERT={m.get('avg_sbert', 0):.4f}\n")
            f.write(f"    样本数: {m.get('n_questions', 0)}, 耗时: {r.get('elapsed_seconds', 0):.1f}s\n")

            by_cat = m.get("by_category", {})
            if by_cat:
                f.write("    类别: ")
                cat_parts = []
                for cat in sorted(by_cat.keys(), key=int):
                    cat_data = by_cat[cat]
                    cat_parts.append(
                        f"C{cat}(F1={cat_data.get('avg_f1', 0):.3f})"
                    )
                f.write(" | ".join(cat_parts) + "\n")

    print(f"文本报告导出完成: {output_path}")


def export_summary(data, output_path):
    """导出为汇总 JSON"""
    valid = [r for r in data if "error" not in r.get("metrics", {})]
    sorted_results = sorted(valid, key=lambda x: x["composite_score"], reverse=True)

    summary = {
        "total_trials": len(data),
        "valid_trials": len(valid),
        "best_result": sorted_results[0] if sorted_results else None,
        "top_10": sorted_results[:10],
        "params_ranges": {
            "theta_sim": sorted(set(r["params"]["theta_sim"] for r in data)),
            "theta_sum": sorted(set(r["params"]["theta_sum"] for r in data)),
            "theta_evict": sorted(set(r["params"]["theta_evict"] for r in data)),
            "decay_interval": sorted(set(
                r["params"].get("decay_interval_rounds", 0) for r in data
            )),
        },
        "avg_composite_score": sum(r["composite_score"] for r in valid) / len(valid) if valid else 0,
        "avg_elapsed_seconds": sum(r.get("elapsed_seconds", 0) for r in valid) / len(valid) if valid else 0,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"汇总 JSON 导出完成: {output_path}")


def export_markdown(data, output_path):
    """导出为 Markdown 格式"""
    cat_names = {
        "1": "Single-hop (单跳)",
        "2": "Temporal (时间)",
        "3": "Multi-hop (多跳)",
        "4": "Open-domain (开放域)",
        "5": "Adversarial (对抗性)",
    }

    valid = [r for r in data if "error" not in r.get("metrics", {})]
    sorted_results = sorted(valid, key=lambda x: x["composite_score"], reverse=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# PhaseForget 超参数搜索结果\n\n")
        f.write(f"**总记录数**: {len(data)}\n\n")
        f.write(f"**有效记录数**: {len(valid)}\n\n")

        if not sorted_results:
            f.write("没有有效的实验结果。\n")
            return

        # 最佳结果
        best = sorted_results[0]
        bp = best["params"]
        bm = best["metrics"]

        f.write("## 最佳结果\n\n")
        f.write(f"- **综合分数**: {best['composite_score']:.4f}\n")
        f.write(f"- **theta_sim**: {bp['theta_sim']}\n")
        f.write(f"- **theta_sum**: {bp['theta_sum']}\n")
        f.write(f"- **theta_evict**: {bp['theta_evict']}\n")
        f.write(f"- **decay_interval**: {bp.get('decay_interval_rounds', 'N/A')}\n")
        f.write(f"- **F1**: {bm.get('avg_f1', 0):.4f}\n")
        f.write(f"- **ROUGE-L**: {bm.get('avg_rouge_l', 0):.4f}\n")
        f.write(f"- **METEOR**: {bm.get('avg_meteor', 0):.4f}\n")
        f.write(f"- **BLEU**: {bm.get('avg_bleu', 0):.4f}\n")
        f.write(f"- **SBERT**: {bm.get('avg_sbert', 0):.4f}\n")
        f.write(f"- **样本数**: {bm.get('n_questions', 0)}\n")
        f.write(f"- **耗时**: {best.get('elapsed_seconds', 0):.1f}s\n\n")

        # 排行榜
        f.write("## 排行榜 (Top 20)\n\n")
        f.write("| 排名 | theta_sim | theta_sum | theta_evict | decay | 综合分 | F1 | ROUGE-L | METEOR | BLEU | 耗时(s) |\n")
        f.write("|------|-----------|-----------|-------------|-------|--------|----|---------|--------|------|----------|\n")

        for rank, r in enumerate(sorted_results[:20], 1):
            p = r["params"]
            m = r["metrics"]
            f.write(
                f"| {rank} | {p['theta_sim']:.2f} | {p['theta_sum']} | {p['theta_evict']:.2f} | "
                f"{p.get('decay_interval_rounds', '-')} | {r['composite_score']:.4f} | "
                f"{m.get('avg_f1', 0):.4f} | {m.get('avg_rouge_l', 0):.4f} | "
                f"{m.get('avg_meteor', 0):.4f} | {m.get('avg_bleu', 0):.4f} | "
                f"{r.get('elapsed_seconds', 0):.1f} |\n"
            )

        # 类别分析
        f.write("\n## 类别分析\n\n")

        for cat_num, cat_name in cat_names.items():
            f.write(f"### {cat_name}\n\n")
            cat_scores = []
            for r in sorted_results:
                if 'by_category' in r.get('metrics', {}):
                    if cat_num in r['metrics']['by_category']:
                        cat_data = r['metrics']['by_category'][cat_num]
                        cat_scores.append({
                            'result': r,
                            'f1': cat_data.get('avg_f1', 0),
                            'rouge_l': cat_data.get('avg_rouge_l', 0),
                        })

            if cat_scores:
                cat_scores_sorted = sorted(cat_scores, key=lambda x: x['f1'], reverse=True)
                f.write("| 排名 | theta_sim | theta_sum | theta_evict | decay | F1 | ROUGE-L |\n")
                f.write("|------|-----------|-----------|-------------|-------|----|---------|\n")

                for rank, item in enumerate(cat_scores_sorted[:10], 1):
                    p = item['result']['params']
                    f.write(
                        f"| {rank} | {p['theta_sim']:.2f} | {p['theta_sum']} | {p['theta_evict']:.2f} | "
                        f"{p.get('decay_interval_rounds', '-')} | {item['f1']:.4f} | "
                        f"{item['rouge_l']:.4f} |\n"
                    )
            else:
                f.write("该类别没有数据。\n")
            f.write("\n")

    print(f"Markdown 报告导出完成: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="导出超参数搜索结果")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出文件路径")
    parser.add_argument(
        "--format",
        choices=["csv", "txt", "json", "summary", "md", "markdown"],
        default="csv",
        help="输出格式：csv=CSV表格，txt=文本报告，json=原始JSON，summary=汇总JSON，md/markdown=Markdown报告"
    )

    args = parser.parse_args()

    # 读取输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误：输入文件不存在: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"读取到 {len(data)} 条记录")

    # 根据格式导出
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "csv":
        export_csv(data, output_path)
    elif args.format == "txt":
        export_txt(data, output_path)
    elif args.format == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"JSON 导出完成: {output_path}")
    elif args.format == "summary":
        export_summary(data, output_path)
    elif args.format in ["md", "markdown"]:
        export_markdown(data, output_path)


if __name__ == "__main__":
    main()

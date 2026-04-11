import json

def summarize_metrics(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 80)
    print("A-Mem 实验结果总结 - F1 & BLEU 指标")
    print("=" * 80)
    print(f"模型: {data.get('model', 'N/A')}")
    print(f"数据集: {data.get('dataset', 'N/A')}")
    print(f"总问题数: {data.get('total_questions', 'N/A')}")
    print()

    category_dist = data.get('category_distribution', {})
    print("类别分布:")
    for cat, count in sorted(category_dist.items()):
        print(f"  类别 {cat}: {count} 条")
    print()

    categories = ["category_1", "category_2", "category_3", "category_4", "category_5"]
    category_names = ["类别1", "类别2", "类别3", "类别4", "类别5"]

    print("=" * 80)
    print(f"{'类别':<10} {'F1 Mean':<12} {'BLEU-1':<12} {'BLEU-2':<12} {'BLEU-3':<12} {'BLEU-4':<12}")
    print("=" * 80)

    for cat, name in zip(categories, category_names):
        if cat in data.get('aggregate_metrics', {}):
            metrics = data['aggregate_metrics'][cat]
            f1_mean = metrics.get('f1', {}).get('mean', 0)
            bleu1 = metrics.get('bleu1', {}).get('mean', 0)
            bleu2 = metrics.get('bleu2', {}).get('mean', 0)
            bleu3 = metrics.get('bleu3', {}).get('mean', 0)
            bleu4 = metrics.get('bleu4', {}).get('mean', 0)

            print(f"{name:<10} {f1_mean:>10.4f}   {bleu1:>10.4f}   {bleu2:>10.4f}   {bleu3:>10.4f}   {bleu4:>10.4f}")

    if "overall" in data.get('aggregate_metrics', {}):
        overall = data['aggregate_metrics']['overall']
        f1_mean = overall.get('f1', {}).get('mean', 0)
        bleu1 = overall.get('bleu1', {}).get('mean', 0)
        bleu2 = overall.get('bleu2', {}).get('mean', 0)
        bleu3 = overall.get('bleu3', {}).get('mean', 0)
        bleu4 = overall.get('bleu4', {}).get('mean', 0)

        print("-" * 80)
        print(f"{'总体':<10} {f1_mean:>10.4f}   {bleu1:>10.4f}   {bleu2:>10.4f}   {bleu3:>10.4f}   {bleu4:>10.4f}")

    print("=" * 80)
    print()

    print("=" * 80)
    print("详细指标 (包含Std)")
    print("=" * 80)

    for cat, name in zip(categories, category_names):
        if cat in data.get('aggregate_metrics', {}):
            metrics = data['aggregate_metrics'][cat]
            print(f"\n{name}:")
            print(f"  F1:    mean={metrics.get('f1', {}).get('mean', 0):.4f}, std={metrics.get('f1', {}).get('std', 0):.4f}")
            print(f"  BLEU-1: mean={metrics.get('bleu1', {}).get('mean', 0):.4f}, std={metrics.get('bleu1', {}).get('std', 0):.4f}")
            print(f"  BLEU-2: mean={metrics.get('bleu2', {}).get('mean', 0):.4f}, std={metrics.get('bleu2', {}).get('std', 0):.4f}")
            print(f"  BLEU-3: mean={metrics.get('bleu3', {}).get('mean', 0):.4f}, std={metrics.get('bleu3', {}).get('std', 0):.4f}")
            print(f"  BLEU-4: mean={metrics.get('bleu4', {}).get('mean', 0):.4f}, std={metrics.get('bleu4', {}).get('std', 0):.4f}")

    if "overall" in data.get('aggregate_metrics', {}):
        overall = data['aggregate_metrics']['overall']
        print(f"\n总体:")
        print(f"  F1:    mean={overall.get('f1', {}).get('mean', 0):.4f}, std={overall.get('f1', {}).get('std', 0):.4f}")
        print(f"  BLEU-1: mean={overall.get('bleu1', {}).get('mean', 0):.4f}, std={overall.get('bleu1', {}).get('std', 0):.4f}")
        print(f"  BLEU-2: mean={overall.get('bleu2', {}).get('mean', 0):.4f}, std={overall.get('bleu2', {}).get('std', 0):.4f}")
        print(f"  BLEU-3: mean={overall.get('bleu3', {}).get('mean', 0):.4f}, std={overall.get('bleu3', {}).get('std', 0):.4f}")
        print(f"  BLEU-4: mean={overall.get('bleu4', {}).get('mean', 0):.4f}, std={overall.get('bleu4', {}).get('std', 0):.4f}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    json_path = r"D:\research\research_A_MEM\A-mem-ollma\A-mem\results\amem_full10.json"
    summarize_metrics(json_path)

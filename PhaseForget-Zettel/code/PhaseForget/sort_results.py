import json
import os
import sys

def sort_results(filepath=None):
    if filepath is None:
        filepath = r'D:\research\research_A_MEM\PhaseForget-Zettel\code\PhaseForget\data\full_compare_ollama.json'
    
    if not os.path.exists(filepath):
        print(f"文件未找到: {filepath}")
        return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 过滤掉没有 metrics 的条目
    data = [entry for entry in data if 'metrics' in entry]

    # 按 avg_f1 降序排列
    sorted_data = sorted(data, key=lambda x: x['metrics'].get('avg_f1', 0), reverse=True)

    print(f"排序文件: {filepath}")
    print(f"{'排名':<5} | {'总分 F1':<10} | {'总分 BLEU':<10} | {'实验 ID (Trial ID)'}")
    print("=" * 110)

    for i, entry in enumerate(sorted_data):
        trial_id = entry.get('trial_id', 'N/A')
        metrics = entry['metrics']
        avg_f1 = metrics.get('avg_f1', 0)
        avg_bleu = metrics.get('avg_bleu', 0)
        params = entry.get('params', {})
        by_category = metrics.get('by_category', {})
        memory_stats = metrics.get('memory_stats', {})

        # 格式化分种类 F1 和 BLEU
        cat_f1_str = ", ".join([f"类别 {cat} F1: {m.get('avg_f1', 0):.4f}" for cat, m in by_category.items()])
        cat_bleu_str = ", ".join([f"类别 {cat} BLEU: {m.get('avg_bleu', 0):.4f}" for cat, m in by_category.items()])
        
        print(f"{i+1:<5} | {avg_f1:<10.4f} | {avg_bleu:<10.4f} | {trial_id}")
        print(f"      参数 (Params): {params}")
        print(f"      分类 F1: {cat_f1_str}")
        print(f"      分类 BLEU: {cat_bleu_str}")
        print(f"      内存统计 (Memory Stats): {memory_stats}")
        print("-" * 110)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    sort_results(path)

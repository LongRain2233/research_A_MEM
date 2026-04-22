import json
import os

filepath = 'D:/research/research_A_MEM/PhaseForget-Zettel/code/PhaseForget/data/full_compare.json'

with open(filepath, 'r', encoding='utf-8') as f:
    data = json.load(f)

data = [entry for entry in data if 'metrics' in entry]
sorted_data = sorted(data, key=lambda x: x['metrics'].get('avg_f1', 0), reverse=True)

print("排名 | 总分(F1) | BLEU | 实验ID")
print("=" * 100)

for i, entry in enumerate(sorted_data):
    trial_id = entry.get('trial_id', 'N/A')
    metrics = entry['metrics']
    avg_f1 = metrics.get('avg_f1', 0)
    avg_bleu = metrics.get('avg_bleu', 0)
    by_category = metrics.get('by_category', {})
    memory_stats = metrics.get('memory_stats', {})
    
    cat_scores = []
    for cat, m in by_category.items():
        cat_f1 = m.get('avg_f1', 0)
        cat_bleu = m.get('avg_bleu', 0)
        cat_scores.append(f"Cat{cat}: F1={cat_f1:.3f} BLEU={cat_bleu:.3f}")
    
    print(f"{i+1:>2} | {avg_f1:.4f} | {avg_bleu:.4f} | {trial_id}")
    print(f"     分类: {', '.join(cat_scores)}")
    print(f"     内存: {memory_stats}")
    print("-" * 100)

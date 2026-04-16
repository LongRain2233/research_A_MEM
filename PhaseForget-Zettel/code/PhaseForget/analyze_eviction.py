import json
from collections import defaultdict

def analyze_file(filepath, label):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"Total experiments: {len(data)}")

    valid = [d for d in data if 'error' not in d.get('metrics', {})]
    print(f"Valid: {len(valid)}")

    if not valid:
        return

    # evicted 计算：total_notes - cold_track_count
    # interaction_count = 总对话轮数 = 总笔记数（每轮写一条）
    # total_notes = memory_stats 中的总数（含 Sigma）
    # cold_track_count = ChromaDB 中的笔记数（原始 + Sigma，驱逐后减少）
    # evicted = 原始笔记数 - 留在 cold_track 中的笔记数

    # 但我们没有记录"原始笔记总数"，用 interaction_count 代替
    for d in valid:
        mem = d.get('metrics', {}).get('memory_stats', {})
        interaction_count = mem.get('interaction_count', 0)
        cold_count = mem.get('cold_track_count', 0)
        d['_evicted'] = interaction_count - cold_count

    valid.sort(key=lambda x: x.get('metrics', {}).get('avg_f1', 0), reverse=True)

    print("\n--- Top 5 by F1 ---")
    print(f"{'sim':<5} {'sum':<5} {'evict':<6} {'evicted':<8} {'F1':<8}")
    for d in valid[:5]:
        p = d['params']
        m = d['metrics']
        print(f"{p.get('theta_sim','?'):<5.2f} {p.get('theta_sum','?'):<5} {p.get('theta_evict','?'):<6.2f} {d['_evicted']:<8} {m.get('avg_f1',0):<8.4f}")

    valid.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
    print("\n--- Top 5 by Composite Score ---")
    print(f"{'sim':<5} {'sum':<5} {'evict':<6} {'evicted':<8} {'F1':<8} {'Comp':<8}")
    for d in valid[:5]:
        p = d['params']
        m = d['metrics']
        print(f"{p.get('theta_sim','?'):<5.2f} {p.get('theta_sum','?'):<5} {p.get('theta_evict','?'):<6.2f} {d['_evicted']:<8} {m.get('avg_f1',0):<8.4f} {d.get('composite_score',0):<8.4f}")

    # 统计有驱逐和无驱逐的 F1 均值
    with_eviction = [d for d in valid if d['_evicted'] > 0]
    no_eviction = [d for d in valid if d['_evicted'] == 0]
    avg_f1_evicted = sum(d['metrics']['avg_f1'] for d in with_eviction) / len(with_eviction) if with_eviction else 0
    avg_f1_no = sum(d['metrics']['avg_f1'] for d in no_eviction) / len(no_eviction) if no_eviction else 0
    print(f"\n--- Eviction Impact ---")
    print(f"  With eviction ({len(with_eviction)} exp): avg F1 = {avg_f1_evicted:.4f}")
    print(f"  No eviction ({len(no_eviction)} exp):  avg F1 = {avg_f1_no:.4f}")

    # sum 维度聚合
    print("\n--- theta_sum vs eviction (avg) ---")
    by_sum = defaultdict(list)
    for d in valid:
        by_sum[d['params'].get('theta_sum', '?')].append(d)
    for k in sorted(by_sum.keys(), key=lambda x: int(x) if isinstance(x, int) else 999):
        group = by_sum[k]
        avg_ev = sum(d['_evicted'] for d in group) / len(group)
        avg_f1 = sum(d['metrics']['avg_f1'] for d in group) / len(group)
        print(f"  sum={k}: count={len(group)}, avg_evicted={avg_ev:.1f}, avg_F1={avg_f1:.4f}")

if __name__ == "__main__":
    analyze_file(r"D:\research\research_A_MEM\PhaseForget-Zettel\code\PhaseForget\data\sum_a1_a2.json", "sum_a1_a2.json (已修复代码的搜索结果)")
    analyze_file(r"D:\research\research_A_MEM\PhaseForget-Zettel\code\PhaseForget\data\fix_forget.json", "fix_forget.json")

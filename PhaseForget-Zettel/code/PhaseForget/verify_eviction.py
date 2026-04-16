import json

for fname in ['sum_a1_a2.json', 'fix_forget.json']:
    with open(f'data/{fname}', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n{'='*60}")
    print(f"  {fname}")
    print(f"{'='*60}")

    # 理解字段含义：
    # interaction_count = 对话轮数（固定=675），等于"原本应该有的原子笔记数"
    # total_notes = SQLite中Memory_State的记录数 = 剩余原子笔记 + Sigma节点
    # abstract_notes = Sigma节点数
    # cold_track_count = ChromaDB中的笔记数

    # 所以：
    # 剩余原子笔记 = total_notes - abstract_notes
    # 被驱逐的原子笔记数 = interaction_count - (total_notes - abstract_notes)
    evicted_list = []
    for d in data:
        mem = d['metrics']['memory_stats']
        ic = mem['interaction_count']
        tn = mem['total_notes']
        ab = mem['abstract_notes']
        evicted = ic - (tn - ab)  # 真正被驱逐的原子笔记数
        evicted_list.append(evicted)

    print(f"实验数: {len(data)}")
    print(f"evicted 范围: {min(evicted_list)} ~ {max(evicted_list)}")
    print(f"evicted > 0 的实验数: {sum(1 for e in evicted_list if e > 0)}")
    print(f"evicted = 0 的实验数: {sum(1 for e in evicted_list if e == 0)}")
    print(f"evicted 平均值: {sum(evicted_list)/len(evicted_list):.1f}")

    # 按 sum 聚合
    from collections import defaultdict
    by_sum = defaultdict(list)
    for d, ev in zip(data, evicted_list):
        by_sum[d['params']['theta_sum']].append(ev)

    print(f"\n按 theta_sum 分组驱逐数：")
    for k in sorted(by_sum.keys()):
        vals = by_sum[k]
        print(f"  sum={k}: count={len(vals)}, avg_evicted={sum(vals)/len(vals):.1f}, max={max(vals)}")

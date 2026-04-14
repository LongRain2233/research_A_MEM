import json

def load_data(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

old = load_data('data/diag_trigger_results.json')
new = load_data('data/diag_trigger_results_v2.json')

with open('data/compare_output.txt', 'w', encoding='utf-8') as f:
    f.write('=== 总体统计对比 ===\n')
    old_triggered = sum(1 for r in old if r['trigger']['trigger_count'] > 0)
    new_triggered = sum(1 for r in new if r['trigger']['trigger_count'] > 0)
    f.write(f'能触发的参数组数: 旧={old_triggered}/30, 新={new_triggered}/30\n')

    old_evicted = sum(r['trigger']['evicted_count'] for r in old)
    new_evicted = sum(r['trigger']['evicted_count'] for r in new)
    f.write(f'总驱逐笔记数:     旧={old_evicted}, 新={new_evicted}\n')

    old_renorm = sum(r['trigger']['renorm_complete_count'] for r in old)
    new_renorm = sum(r['trigger']['renorm_complete_count'] for r in new)
    f.write(f'总Renorm完成数:   旧={old_renorm}, 新={new_renorm}\n')

    f.write('\n=== 触发与驱逐情况抽样对比 ===\n')
    for nr in sorted([r for r in new if r['trigger']['trigger_count']>0], key=lambda x: x['trigger']['trigger_count'], reverse=True)[:5]:
        match = next((o for o in old if o['label'] == nr['label']), None)
        if match:
            nt, ot = nr['trigger'], match['trigger']
            f.write(f"[{nr['label']}]\n")
            f.write(f"  触发: 旧={ot['trigger_count']:<3} -> 新={nt['trigger_count']:<3}\n")
            f.write(f"  完成: 旧={ot['renorm_complete_count']:<3} -> 新={nt['renorm_complete_count']:<3}\n")
            f.write(f"  驱逐: 旧={ot['evicted_count']:<3} -> 新={nt['evicted_count']:<3}\n")
            f.write(f"  F1:   旧={match['metrics']['avg_f1']:.4f} -> 新={nr['metrics']['avg_f1']:.4f}\n")

    f.write('\n=== Theta Evict 灵敏度组驱逐数对比 ===\n')
    for nr in [r for r in new if r['group'] == 'evict_sensitivity']:
        match = next((o for o in old if o['label'] == nr['label']), None)
        if match:
            f.write(f"[{nr['label']}] 驱逐数: 旧={match['trigger']['evicted_count']} -> 新={nr['trigger']['evicted_count']}\n")

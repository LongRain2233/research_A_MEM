import json
from collections import defaultdict

def analyze_results(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        data = [data]
        
    print(f"Total experiments: {len(data)}\n")
    
    # Collect metrics for analysis
    valid_data = [d for d in data if 'error' not in d.get('metrics', {})]
    print(f"Valid experiments (no errors): {len(valid_data)}\n")
    
    if not valid_data:
        return
        
    # Sort by F1 score
    valid_data.sort(key=lambda x: x.get('metrics', {}).get('avg_f1', 0), reverse=True)
    
    print("=== Top 10 Configurations by F1 Score ===")
    print(f"{'Rank':<5} {'sim':<5} {'sum':<5} {'evict':<6} {'t_cool':<7} {'F1':<8} {'SBERT':<8} {'Total_Notes':<12} {'Cold_Notes':<12} {'Evicted':<8}")
    for i, d in enumerate(valid_data[:10], 1):
        p = d['params']
        m = d['metrics']
        mem = m.get('memory_stats', {})
        total_notes = mem.get('interaction_count', 0)
        cold_notes = mem.get('cold_track_count', 0)
        evicted = total_notes - cold_notes if total_notes > 0 else "N/A"
        
        # t_cool might not be present in some runs if it wasn't tuned, but let's check
        t_cool = p.get('t_cool', 'N/A')
        
        print(f"{i:<5} {p.get('theta_sim', 'N/A'):<5.2f} {p.get('theta_sum', 'N/A'):<5} {p.get('theta_evict', 'N/A'):<6.2f} {t_cool:<7} {m.get('avg_f1', 0):<8.4f} {m.get('avg_sbert', 0):<8.4f} {total_notes:<12} {cold_notes:<12} {evicted:<8}")

    print("\n=== Top 5 Configurations by Composite Score ===")
    valid_data_comp = sorted(valid_data, key=lambda x: x.get('composite_score', 0), reverse=True)
    print(f"{'Rank':<5} {'sim':<5} {'sum':<5} {'evict':<6} {'t_cool':<7} {'Comp_Score':<10} {'F1':<8} {'Evicted':<8}")
    for i, d in enumerate(valid_data_comp[:5], 1):
        p = d['params']
        m = d['metrics']
        mem = m.get('memory_stats', {})
        total_notes = mem.get('interaction_count', 0)
        cold_notes = mem.get('cold_track_count', 0)
        evicted = total_notes - cold_notes if total_notes > 0 else "N/A"
        t_cool = p.get('t_cool', 'N/A')
        print(f"{i:<5} {p.get('theta_sim', 'N/A'):<5.2f} {p.get('theta_sum', 'N/A'):<5} {p.get('theta_evict', 'N/A'):<6.2f} {t_cool:<7} {d.get('composite_score', 0):<10.4f} {m.get('avg_f1', 0):<8.4f} {evicted:<8}")


    # Analyze impact of theta_sum on F1 and Eviction
    print("\n=== Impact of theta_sum (Averages) ===")
    sum_stats = defaultdict(lambda: {'f1': [], 'evicted': [], 'count': 0})
    for d in valid_data:
        tsum = d['params'].get('theta_sum', 'N/A')
        f1 = d['metrics'].get('avg_f1', 0)
        mem = d['metrics'].get('memory_stats', {})
        total_notes = mem.get('interaction_count', 0)
        cold_notes = mem.get('cold_track_count', 0)
        evicted = total_notes - cold_notes if total_notes > 0 else 0
        
        sum_stats[tsum]['f1'].append(f1)
        sum_stats[tsum]['evicted'].append(evicted)
        sum_stats[tsum]['count'] += 1
        
    print(f"{'theta_sum':<10} {'Count':<8} {'Avg_F1':<10} {'Avg_Evicted':<12}")
    for k in sorted(sum_stats.keys(), key=lambda x: float(x) if isinstance(x, (int, float)) else 999):
        v = sum_stats[k]
        avg_f1 = sum(v['f1'])/v['count'] if v['count'] > 0 else 0
        avg_evicted = sum(v['evicted'])/v['count'] if v['count'] > 0 else 0
        print(f"{k:<10} {v['count']:<8} {avg_f1:<10.4f} {avg_evicted:<12.1f}")
        
    # Analyze impact of theta_sim on F1 and Eviction
    print("\n=== Impact of theta_sim (Averages) ===")
    sim_stats = defaultdict(lambda: {'f1': [], 'evicted': [], 'count': 0})
    for d in valid_data:
        tsim = d['params'].get('theta_sim', 'N/A')
        f1 = d['metrics'].get('avg_f1', 0)
        mem = d['metrics'].get('memory_stats', {})
        total_notes = mem.get('interaction_count', 0)
        cold_notes = mem.get('cold_track_count', 0)
        evicted = total_notes - cold_notes if total_notes > 0 else 0
        
        sim_stats[tsim]['f1'].append(f1)
        sim_stats[tsim]['evicted'].append(evicted)
        sim_stats[tsim]['count'] += 1
        
    print(f"{'theta_sim':<10} {'Count':<8} {'Avg_F1':<10} {'Avg_Evicted':<12}")
    for k in sorted(sim_stats.keys(), key=lambda x: float(x) if isinstance(x, (int, float)) else 999):
        v = sim_stats[k]
        avg_f1 = sum(v['f1'])/v['count'] if v['count'] > 0 else 0
        avg_evicted = sum(v['evicted'])/v['count'] if v['count'] > 0 else 0
        print(f"{k:<10} {v['count']:<8} {avg_f1:<10.4f} {avg_evicted:<12.1f}")

if __name__ == "__main__":
    analyze_results(r"D:\research\research_A_MEM\PhaseForget-Zettel\code\PhaseForget\data\sum_a1_a2.json")

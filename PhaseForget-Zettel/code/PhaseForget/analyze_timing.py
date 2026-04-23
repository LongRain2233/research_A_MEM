import json
from pathlib import Path

data_dir = Path(r'D:\research\research_A_MEM\PhaseForget-Zettel\code\PhaseForget\data')
folders = list(data_dir.glob('hps_longmemeval_*_3_12'))

summary = []
for f in folders:
    timing_file = f / 'bench_checkpoint_timing.json'
    if timing_file.exists():
        try:
            with open(timing_file, 'r', encoding='utf-8') as j:
                records = json.load(j)
                parts = f.name.split('_')
                tsim = parts[2]
                tsum = parts[3]
                
                if records:
                    avg_total = sum(r['total_s'] for r in records) / len(records)
                    avg_p1 = sum(r['phase1_ingest_s'] for r in records) / len(records)
                    avg_p2 = sum(r['phase2_qa_s'] for r in records) / len(records)
                    avg_mpt = sum(r.get('ms_per_turn', 0) for r in records) / len(records)
                    summary.append({
                        'tsim': tsim,
                        'tsum': tsum,
                        'total': avg_total,
                        'p1': avg_p1,
                        'p2': avg_p2,
                        'mpt': avg_mpt
                    })
        except Exception as e:
            print(f"Error reading {timing_file}: {e}")

summary.sort(key=lambda x: (float(x['tsim']), int(x['tsum'])))

print("-" * 75)
print(f"{'theta_sim':<10} | {'theta_sum':<10} | {'Total(s)':<10} | {'P1_Ingest(s)':<12} | {'P2_QA(s)':<10} | {'ms/turn':<10}")
print("-" * 75)
for s in summary:
    print(f"{s['tsim']:<10} | {s['tsum']:<10} | {s['total']:<10.1f} | {s['p1']:<12.1f} | {s['p2']:<10.1f} | {s['mpt']:<10.1f}")
print("-" * 75)

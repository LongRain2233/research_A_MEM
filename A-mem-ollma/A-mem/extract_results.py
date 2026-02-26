import json

with open(r'D:\research\A-mem-ollma\A-mem\results\eval_qwen2.5_1.5b.json', 'r') as f:
    data = json.load(f)

agg = data['aggregate_metrics']
cat_names = {
    'overall':    'Overall',
    'category_1': 'Cat1: Multi-hop',
    'category_2': 'Cat2: Temporal',
    'category_3': 'Cat3: Open-domain',
    'category_4': 'Cat4: Single-hop',
    'category_5': 'Cat5: Adversarial'
}

print('='*70)
print(f"{'Category':<24} {'F1':>8} {'Exact':>8} {'ROUGE-1':>8} {'METEOR':>8} {'N':>6}")
print('-'*70)
for key in ['overall', 'category_1', 'category_2', 'category_3', 'category_4', 'category_5']:
    if key in agg:
        d = agg[key]
        f1    = d['f1']['mean'] * 100
        em    = d['exact_match']['mean'] * 100
        r1    = d['rouge1_f']['mean'] * 100
        met   = d['meteor']['mean'] * 100
        cnt   = d['f1']['count']
        name  = cat_names.get(key, key)
        print(f"{name:<24} {f1:>7.1f}% {em:>7.1f}% {r1:>7.1f}% {met:>7.1f}% {cnt:>6}")
print('='*70)

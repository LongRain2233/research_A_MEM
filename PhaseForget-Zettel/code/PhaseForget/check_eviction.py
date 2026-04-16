import json

with open('data/sum_a1_a2.json', encoding='utf-8') as f:
    data = json.load(f)

d = data[0]
mem = d['metrics']['memory_stats']

print("=== sum_a1_a2.json 示例结构 ===")
print("trial_id:", d.get('trial_id'))
print()
print("memory_stats:")
for k, v in mem.items():
    print(f"  {k}: {v}")
print()
print("params:")
for k, v in d['params'].items():
    print(f"  {k}: {v}")
print()

# Key question: does total_notes == cold_track_count for ALL experiments?
print("=== 检验：total_notes == cold_track_count 吗？===")
same = mem['total_notes'] == mem['cold_track_count']
print(f"  结果: {same}")
print(f"  total_notes - cold_track_count = {mem['total_notes'] - mem['cold_track_count']}")

diff_count = sum(
    1 for d in data
    if d['metrics']['memory_stats']['total_notes'] != d['metrics']['memory_stats']['cold_track_count']
)
print()
print(f"所有 {len(data)} 个实验中，total_notes != cold_track_count 的有: {diff_count} 个")

# Now let's check fix_forget.json
with open('data/fix_forget.json', encoding='utf-8') as f:
    fix_data = json.load(f)

fix_diff = sum(
    1 for d in fix_data
    if d['metrics']['memory_stats']['total_notes'] != d['metrics']['memory_stats']['cold_track_count']
)
print(f"fix_forget.json 中有差异的有: {fix_diff} / {len(fix_data)} 个")

# Show a few examples where they differ
print()
print("=== fix_forget.json 中有差异的示例（前5个）===")
count = 0
for d in fix_data:
    mem_d = d['metrics']['memory_stats']
    if mem_d['total_notes'] != mem_d['cold_track_count']:
        diff = mem_d['total_notes'] - mem_d['cold_track_count']
        p = d['params']
        print(f"  trial={d['trial_id'][:20]}... sim={p.get('theta_sim')} sum={p.get('theta_sum')} evict={p.get('theta_evict')}")
        print(f"    total_notes={mem_d['total_notes']}, cold_track={mem_d['cold_track_count']}, 差值={diff}")
        count += 1
        if count >= 5:
            break

"""
分析 fix_forget.json 的参数组合
从 trial_id 解析参数，并统计唯一组合
"""
import json
from pathlib import Path
from collections import defaultdict

# 加载数据
data_path = Path(__file__).parent / "data" / "fix_forget.json"
with open(data_path, "r", encoding="utf-8") as f:
    results = json.load(f)

print(f"总实验数: {len(results)}")
print("=" * 60)

# 用于去重和统计
unique_params = set()
all_trials = []

for r in results:
    trial_id = r["trial_id"]
    # 解析 trial_id: 格式 0001_0.50_10_0.25_20_时间戳
    parts = trial_id.split("_")
    if len(parts) >= 5:
        trial_num = parts[0]
        theta_sim = parts[1]
        theta_sum = parts[2]
        theta_evict = parts[3]
        decay = parts[4]
    else:
        # fallback: 从 params 获取
        theta_sim = r["params"]["theta_sim"]
        theta_sum = r["params"]["theta_sum"]
        theta_evict = r["params"]["theta_evict"]
        decay = r["params"]["decay_interval_rounds"]

    key = (theta_sim, theta_sum, theta_evict, decay)
    unique_params.add(key)
    all_trials.append({
        "trial_id": trial_id,
        "theta_sim": theta_sim,
        "theta_sum": theta_sum,
        "theta_evict": theta_evict,
        "decay": decay,
        "record_indices": tuple(r.get("record_indices", [])),
        "disable_self_retrieval": r.get("disable_self_retrieval", False),
    })

print(f"唯一参数组合数: {len(unique_params)}")
print("=" * 60)

# 统计每个参数的不同取值
theta_sims = set()
theta_sums = set()
theta_evicts = set()
decays = set()
records = set()
self_retrievals = set()

for key in unique_params:
    theta_sims.add(float(key[0]))
    theta_sums.add(int(key[1]))
    theta_evicts.add(float(key[2]))
    decays.add(int(key[3]))

for t in all_trials:
    records.add(t["record_indices"])
    self_retrievals.add(t["disable_self_retrieval"])

print("参数取值范围:")
print(f"  theta_sim:      {sorted(theta_sims)}")
print(f"  theta_sum:      {sorted(theta_sums)}")
print(f"  theta_evict:    {sorted(theta_evicts)}")
print(f"  decay:          {sorted(decays)}")
print(f"  record_indices: {sorted(records, key=lambda x: str(x))}")
print(f"  disable_self_retrieval: {sorted(self_retrievals)}")
print("=" * 60)

# 计算理论笛卡尔积数量
theoretical_total = len(theta_sims) * len(theta_sums) * len(theta_evicts) * len(decays)
print(f"理论笛卡尔积组合数: {theoretical_total}")
print(f"实际唯一组合数:    {len(unique_params)}")
print(f"覆盖率:            {len(unique_params)/theoretical_total*100:.1f}%")
print("=" * 60)

# 按参数值排序显示所有唯一组合
print("\n所有唯一参数组合:")
print(f"{'theta_sim':>10} {'theta_sum':>10} {'theta_evict':>12} {'decay':>8}")
print("-" * 45)
for p in sorted(unique_params, key=lambda x: (float(x[0]), int(x[1]), float(x[2]), int(x[3]))):
    print(f"{p[0]:>10} {p[1]:>10} {p[2]:>12} {p[3]:>8}")

print("\n" + "=" * 60)
print("根据分析，输入参数应该是:")
print(f"  --theta-sim-values    {','.join(map(str, sorted(theta_sims)))}")
print(f"  --theta-sum-values    {','.join(map(str, sorted(theta_sums)))}")
print(f"  --theta-evict-values  {','.join(map(str, sorted(theta_evicts)))}")
print(f"  --decay-interval-values {','.join(map(str, sorted(decays)))}")
print(f"  --record-indices       5 (或根据实际数据)")
print(f"  --search-type          grid")

#!/usr/bin/env python3
"""分析 baseline_check_changge_deacy.json 中所有实验参数组合"""

import json
from collections import defaultdict

def analyze_all_params(filepath):
    """分析文件中所有实验的参数组合"""

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 收集所有参数值
    param_values = defaultdict(set)
    all_combinations = []

    for entry in data:
        params = entry.get('params', {})

        # 收集每个参数的所有取值
        for key, value in params.items():
            param_values[key].add(value)

        # 记录完整的参数组合
        combo = {
            'trial_id': entry.get('trial_id', 'N/A'),
            'theta_sim': params.get('theta_sim'),
            'theta_sum': params.get('theta_sum'),
            'theta_evict': params.get('theta_evict'),
            'decay_interval_rounds': params.get('decay_interval_rounds'),
            'composite_score': entry.get('composite_score', 'N/A'),
            'record_indices': entry.get('record_indices', [])
        }
        all_combinations.append(combo)

    return param_values, all_combinations, len(data)

def main():
    filepath = "data/baseline_check_changge_deacy.json"

    param_values, all_combinations, total_count = analyze_all_params(filepath)

    print("=" * 70)
    print("实验参数分析报告")
    print("=" * 70)

    print(f"\n总实验组数: {total_count}")

    # 按theta_sim分组
    sim_07 = [c for c in all_combinations if c['theta_sim'] == 0.7]
    sim_08 = [c for c in all_combinations if c['theta_sim'] == 0.8]

    print(f"\n  - theta_sim=0.7 的实验: {len(sim_07)} 组")
    print(f"  - theta_sim=0.8 的实验: {len(sim_08)} 组")

    print("\n" + "=" * 70)
    print("一、各参数的取值范围")
    print("=" * 70)

    for param_name in ['theta_sim', 'theta_sum', 'theta_evict', 'decay_interval_rounds']:
        values = sorted(param_values[param_name])
        print(f"\n{param_name}:")
        print(f"  取值: {values}")
        print(f"  共 {len(values)} 个不同值")

    print("\n" + "=" * 70)
    print("二、完整的参数组合")
    print("=" * 70)

    # 先显示 theta_sim=0.8 的组合（表现更好的）
    print(f"\n【theta_sim = 0.8】 共 {len(sim_08)} 组")
    print("-" * 70)

    # 按theta_sum分组显示
    for theta_sum in sorted(set(c['theta_sum'] for c in sim_08)):
        subset = [c for c in sim_08 if c['theta_sum'] == theta_sum]
        print(f"\n  theta_sum = {theta_sum} ({len(subset)} 组):")

        for combo in sorted(subset, key=lambda x: (x['theta_evict'], x['decay_interval_rounds'])):
            print(f"    theta_evict={combo['theta_evict']}, decay_interval={combo['decay_interval_rounds']:<3} "
                  f"-> score={combo['composite_score']:.4f}  (trial: {combo['trial_id']})")

    # 再显示 theta_sim=0.7 的组合
    print(f"\n\n【theta_sim = 0.7】 共 {len(sim_07)} 组")
    print("-" * 70)

    for theta_sum in sorted(set(c['theta_sum'] for c in sim_07)):
        subset = [c for c in sim_07 if c['theta_sum'] == theta_sum]
        print(f"\n  theta_sum = {theta_sum} ({len(subset)} 组):")

        for combo in sorted(subset, key=lambda x: (x['theta_evict'], x['decay_interval_rounds'])):
            print(f"    theta_evict={combo['theta_evict']}, decay_interval={combo['decay_interval_rounds']:<3} "
                  f"-> score={combo['composite_score']:.4f}  (trial: {combo['trial_id']})")

    print("\n" + "=" * 70)
    print("三、record_indices 信息")
    print("=" * 70)

    record_indices_set = set()
    for combo in all_combinations:
        record_indices_set.add(tuple(combo['record_indices']))

    print(f"\n所有实验使用的 record_indices:")
    for indices in sorted(record_indices_set):
        count = sum(1 for c in all_combinations if tuple(c['record_indices']) == indices)
        print(f"  {list(indices)}: {count} 组实验")

    print("\n" + "=" * 70)
    print("四、参数网格汇总")
    print("=" * 70)

    # 计算笛卡尔积
    theta_sim_vals = sorted(param_values['theta_sim'])
    theta_sum_vals = sorted(param_values['theta_sum'])
    theta_evict_vals = sorted(param_values['theta_evict'])
    decay_interval_vals = sorted(param_values['decay_interval_rounds'])

    print(f"\ntheta_sim 值: {theta_sim_vals}")
    print(f"theta_sum 值: {theta_sum_vals}")
    print(f"theta_evict 值: {theta_evict_vals}")
    print(f"decay_interval_rounds 值: {decay_interval_vals}")

    expected_count = len(theta_sim_vals) * len(theta_sum_vals) * len(theta_evict_vals) * len(decay_interval_vals)
    print(f"\n理论组合数: {len(theta_sim_vals)} x {len(theta_sum_vals)} x {len(theta_evict_vals)} x {len(decay_interval_vals)} = {expected_count}")
    print(f"实际实验数: {total_count}")

    if total_count == expected_count:
        print("[OK] 所有参数组合均已覆盖")
    else:
        print(f"[!] 缺少 {expected_count - total_count} 组实验")

if __name__ == "__main__":
    main()

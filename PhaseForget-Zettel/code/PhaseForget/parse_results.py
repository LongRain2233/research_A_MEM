#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析所有 hparam_record5_grid*.json 实验结果
并给出统一排名
"""

import json
import sys
from pathlib import Path

def parse_multiple_results(filepaths):
    """解析多个实验结果JSON文件并合并排名"""
    all_data = []
    for filepath in filepaths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.extend(data)
        except Exception as e:
            print(f"读取文件 {filepath} 时出错: {e}")

    # 排序
    sorted_data = sorted(all_data, key=lambda x: x.get('composite_score', 0), reverse=True)

    print("\n### 所有实验统一排名 (按总分降序)")
    print("\n| 排名 | Trial ID | 总分 | F1-C1 | F1-C2 | F1-C3 | F1-C4 | Abstract | 来源 |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for rank, record in enumerate(sorted_data, 1):
        trial_id = record.get('trial_id', 'N/A')
        composite_score = record.get('composite_score', 0)
        metrics = record.get('metrics', {})
        by_category = metrics.get('by_category', {})
        abstract_notes = metrics.get('memory_stats', {}).get('abstract_notes', 0)
        f1s = [by_category.get(str(i), {}).get('avg_f1', 0) for i in range(1, 5)]
        
        # 简单判断来源：theta_sum=99999 为 no-forget，否则为 grid
        params = record.get('params', {})
        theta_sum = params.get('theta_sum', 0)
        source = "no-forget" if theta_sum == 99999 else "grid"
        
        print(f"| {rank} | `{trial_id}` | **{composite_score:.4f}** | {f1s[0]:.4f} | {f1s[1]:.4f} | {f1s[2]:.4f} | {f1s[3]:.4f} | {abstract_notes} | {source} |")

def main():
    base_path = Path("d:/research/research_A_MEM/PhaseForget-Zettel/code/PhaseForget/data")
    files = [
        base_path / "hparam_record5_grid.json",
        base_path / "hparam_record5_grid_withoutforget.json"
    ]
    
    existing_files = [f for f in files if f.exists()]
    if not existing_files:
        print("错误: 未找到任何实验结果文件")
        sys.exit(1)
        
    parse_multiple_results(existing_files)

if __name__ == "__main__":
    main()

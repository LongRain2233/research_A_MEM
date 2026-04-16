import json

with open('data/sum_a1_a2.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 按composite_score排序，取前20
sorted_data = sorted(data, key=lambda x: x.get('composite_score', 0), reverse=True)[:20]

md_content = '# Top 20 参数组合得分排名\n\n'
md_content += '| 排名 | trial_id | theta_sim | theta_sum | theta_evict | decay | 综合得分 | F1 | BLEU | RougeL | Rouge2 | METEOR | SBERT |\n'
md_content += '|------|----------|-----------|-----------|-------------|-------|----------|-----|------|--------|-------|-------|------|\n'

for i, item in enumerate(sorted_data, 1):
    params = item['params']
    metrics = item['metrics']
    md_content += f'| {i} | {item["trial_id"]} | {params["theta_sim"]} | {params["theta_sum"]} | {params["theta_evict"]} | {params["decay_interval_rounds"]} | {item["composite_score"]:.4f} | {metrics["avg_f1"]:.4f} | {metrics["avg_bleu"]:.4f} | {metrics["avg_rouge_l"]:.4f} | {metrics["avg_rouge2"]:.4f} | {metrics["avg_meteor"]:.4f} | {metrics["avg_sbert"]:.4f} |\n'

md_content += '\n\n# 各分类详细得分\n\n'

for i, item in enumerate(sorted_data, 1):
    params = item['params']
    md_content += f'## 排名 {i}: {item["trial_id"]}\n\n'
    md_content += f'- **参数**: theta_sim={params["theta_sim"]}, theta_sum={params["theta_sum"]}, theta_evict={params["theta_evict"]}, decay={params["decay_interval_rounds"]}\n'
    md_content += f'- **综合得分**: {item["composite_score"]:.4f}\n\n'
    
    md_content += '| 类别 | F1 | BLEU | RougeL | SBERT | 样本数 |\n'
    md_content += '|------|-----|------|--------|-------|------|\n'
    
    for cat, cat_data in sorted(item['metrics']['by_category'].items()):
        md_content += f'| {cat} | {cat_data["avg_f1"]:.4f} | {cat_data["avg_bleu"]:.4f} | {cat_data["avg_rouge_l"]:.4f} | {cat_data["avg_sbert"]:.4f} | {cat_data["n_questions"]} |\n'
    
    md_content += '\n'

with open('data/top20_results.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

print('已生成 data/top20_results.md')

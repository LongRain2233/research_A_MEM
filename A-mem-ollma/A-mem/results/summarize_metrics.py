import json

# 读取评估结果文件
with open('eval_qwen2.5_3b_record5.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 获取每个类别的指标
categories = ['category_1', 'category_2', 'category_3', 'category_4', 'category_5', 'overall']

print("=" * 90)
print("Qwen2.5:3b 模型在 locomo_record_idx5 数据集上的评估结果")
print("=" * 90)
print()

# 打印表头
print(f"{'类别':<15} {'F1 Mean':<12} {'BLEU-1':<12} {'BLEU-2':<12} {'BLEU-3':<12} {'BLEU-4':<12}")
print("-" * 90)

for cat in categories:
    metrics = data['aggregate_metrics'][cat]
    f1_mean = metrics['f1']['mean']
    bleu1 = metrics['bleu1']['mean']
    bleu2 = metrics['bleu2']['mean']
    bleu3 = metrics['bleu3']['mean']
    bleu4 = metrics['bleu4']['mean']

    cat_display = cat.replace('category_', 'Category ')
    if cat == 'overall':
        cat_display = 'Overall'

    print(f"{cat_display:<15} {f1_mean:<12.4f} {bleu1:<12.4f} {bleu2:<12.4f} {bleu3:<12.4f} {bleu4:<12.4f}")

print()
print("=" * 80)

# 额外输出各类型的样本数量
print("\n各类型样本数量:")
dist = data['category_distribution']
for cat_id, count in sorted(dist.items(), key=lambda x: int(x[0])):
    print(f"  Category {cat_id}: {count} 个样本")
print(f"  总计: {data['total_questions']} 个问题")

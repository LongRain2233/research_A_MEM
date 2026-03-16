"""
单个样本的记忆构建和评测演示。

流程：
1. 加载 locomo10.json 第一条样本
2. 构建记忆 (add all dialogue turns)
3. 查看记忆网络状态
4. 评测 QA
"""

import asyncio
import json
from pathlib import Path

from phaseforget.config.settings import get_settings
from phaseforget.pipeline.orchestrator import PhaseForgetSystem
from phaseforget.evaluation.loaders.locomo import LoCoMoLoader
from phaseforget.evaluation.metrics import (
    compute_f1, compute_bleu1, compute_rouge_l, compute_rouge2, compute_meteor, compute_sbert
)
from sentence_transformers import SentenceTransformer


async def main():
    print("=" * 80)
    print("单样本记忆构建和评测演示")
    print("=" * 80)

    # Step 0: 加载数据
    print("\n[Step 0] 加载 locomo10.json...")
    loader = LoCoMoLoader()
    sessions = loader.load("dataset/locomo10.json")

    if not sessions:
        print("❌ 没有加载到数据")
        return

    # 只用第一个样本
    session = sessions[0]
    sample_id = session["session_id"]
    dialogue = session["dialogue"]
    questions = session["questions"]

    print(f"✅ 加载成功")
    print(f"   样本 ID: {sample_id}")
    print(f"   对话轮数: {len(dialogue)}")
    print(f"   QA 数量: {len(questions)}")

    # Step 1: 初始化系统
    print("\n[Step 1] 初始化 PhaseForgetSystem...")
    settings = get_settings()
    system = PhaseForgetSystem(settings=settings)
    await system.initialize()
    print("✅ 系统初始化完成")

    # Step 2: 构建记忆
    print("\n[Step 2] 构建记忆网络 (添加 {} 轮对话)...".format(len(dialogue)))

    for idx, turn in enumerate(dialogue):
        content = turn.get("content", "")
        if not content.strip():
            continue

        try:
            note = await system.add_interaction(content=content)
            if (idx + 1) % 50 == 0:
                print(f"   进度: {idx + 1}/{len(dialogue)} 轮")
        except Exception as e:
            print(f"   ❌ Turn {idx} 失败: {e}")

    print(f"✅ 记忆构建完成")

    # Step 3: 查看记忆状态
    print("\n[Step 3] 记忆网络状态:")
    stats = await system.get_stats()
    print(f"   总 note 数: {stats.get('total_notes', 0)}")
    print(f"   Cold track count: {stats.get('cold_track_count', 0)}")
    print(f"   Interaction count: {stats.get('interaction_count', 0)}")

    # Step 4: 评测 QA
    print(f"\n[Step 4] 评测 QA (共 {len(questions)} 个问题)...")
    print("=" * 80)
    print(f"{'问题':<50} {'答案':<20} {'F1':>8} {'BLEU':>8}")
    print("-" * 80)

    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

    f1_scores = []
    bleu_scores = []
    rouge_l_scores = []
    rouge2_scores = []
    meteor_scores = []
    sbert_scores = []

    for qa_idx, qa in enumerate(questions[:10]):  # 只显示前 10 个
        question = qa.get("question", "")
        reference = qa.get("answer", "")

        if not question or not reference:
            continue

        # 检索
        retrieved = system.search(question, top_k=5)

        # 用检索到的文本作为预测答案 (LLM 失败时的降级方案)
        if retrieved:
            prediction = " ".join(r.get("content", "")[:100] for r in retrieved[:3])
        else:
            prediction = ""

        # 计算指标
        f1 = compute_f1(prediction, reference)
        bleu = compute_bleu1(prediction, reference)
        rouge_l = compute_rouge_l(prediction, reference)
        rouge2 = compute_rouge2(prediction, reference)
        meteor = compute_meteor(prediction, reference)
        sbert = compute_sbert(prediction, reference, sbert_model)

        f1_scores.append(f1)
        bleu_scores.append(bleu)
        rouge_l_scores.append(rouge_l)
        rouge2_scores.append(rouge2)
        meteor_scores.append(meteor)
        sbert_scores.append(sbert)

        q_short = question[:47] + "..." if len(question) > 50 else question
        a_short = str(reference)[:17]
        print(f"{q_short:<50} {a_short:<20} {f1:>8.4f} {bleu:>8.4f}")

    # 汇总结果
    print("=" * 80)
    print(f"\n[汇总] 前 10 个 QA 的指标平均值:")
    print(f"  F1         = {sum(f1_scores) / len(f1_scores):.4f}")
    print(f"  BLEU-1     = {sum(bleu_scores) / len(bleu_scores):.4f}")
    print(f"  ROUGE-L    = {sum(rouge_l_scores) / len(rouge_l_scores):.4f}")
    print(f"  ROUGE-2    = {sum(rouge2_scores) / len(rouge2_scores):.4f}")
    print(f"  METEOR     = {sum(meteor_scores) / len(meteor_scores):.4f}")
    print(f"  SBERT      = {sum(sbert_scores) / len(sbert_scores):.2f}")

    # Step 5: 示例检索详情
    print(f"\n[Step 5] 详细检索示例 (第一个问题):")
    if questions:
        qa = questions[0]
        q = qa.get("question", "")
        a = qa.get("answer", "")
        cat = qa.get("category", 0)

        print(f"  问题: {q}")
        print(f"  标准答案: {a}")
        print(f"  分类: {cat} ({['', '单跳', '时间', '多跳', '详细', '对抗'][min(cat, 5)]}")

        retrieved = system.search(q, top_k=5)
        print(f"\n  检索到 {len(retrieved)} 条相关记忆:")
        for idx, res in enumerate(retrieved[:5]):
            print(f"    [{idx+1}] (相似度 {res['score']:.3f}) {res['content'][:80]}")

    # 关闭系统
    await system.close()
    print("\n✅ 测试完成，系统已关闭")


if __name__ == "__main__":
    asyncio.run(main())

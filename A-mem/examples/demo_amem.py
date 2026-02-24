# -*- coding: utf-8 -*-
"""
A-MEM 记忆系统交互式体验脚本
================================
演示 A-MEM 的核心能力：
  1. 自动元数据提取（关键词、上下文、标签）
  2. 记忆进化（自动建立关联、更新邻居）
  3. 语义搜索 + 邻居扩展检索
  4. 记忆更新与删除

用法：
  python examples/demo_amem.py
"""

import os
import sys
import io
import time

# 强制 stdout 使用 UTF-8，避免 Windows GBK 编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from agentic_memory.memory_system import AgenticMemorySystem

# -- 配置 --------------------------------------------------------
LLM_BACKEND = "openrouter"
LLM_MODEL = "deepseek/deepseek-chat-v3-0324"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def separator(title=""):
    width = 60
    if title:
        pad = width - len(title) - 7
        if pad < 2:
            pad = 2
        print(f"\n{'=' * 5} {title} {'=' * pad}")
    else:
        print("=" * width)


def print_memory(mem, prefix=""):
    """格式化打印一条记忆的详情"""
    print(f"{prefix}  ID      : {mem.id[:16]}...")
    print(f"{prefix}  内容    : {mem.content[:80]}")
    print(f"{prefix}  关键词  : {mem.keywords}")
    print(f"{prefix}  上下文  : {mem.context}")
    print(f"{prefix}  标签    : {mem.tags}")
    print(f"{prefix}  链接数  : {len(mem.links)}")
    if mem.links:
        print(f"{prefix}  链接到  : {[lid[:12] + '...' for lid in mem.links]}")


def print_search_result(r, idx):
    """格式化打印一条搜索结果"""
    tag = "[邻居]" if r.get("is_neighbor") else "[直接命中]"
    score = f" (距离: {r['score']:.4f})" if "score" in r else ""
    print(f"  [{idx}] {tag}{score}")
    print(f"       内容: {r['content'][:80]}")
    print(f"       标签: {r.get('tags', [])}")
    print(f"       上下文: {r.get('context', 'N/A')}")


# ================================================================
#  主流程
# ================================================================
def main():
    separator("初始化 A-MEM 系统")
    print(f"  后端: {LLM_BACKEND}")
    print(f"  模型: {LLM_MODEL}")
    print(f"  Embedding: {EMBEDDING_MODEL}")

    t0 = time.time()
    memory_system = AgenticMemorySystem(
        model_name=EMBEDDING_MODEL,
        llm_backend=LLM_BACKEND,
        llm_model=LLM_MODEL,
    )
    print(f"  [OK] 初始化完成 ({time.time() - t0:.1f}s)")

    # --- 第一阶段：逐条添加记忆，观察自动元数据提取 ---
    separator("第一阶段：添加记忆 & 自动元数据提取")
    print("  系统会用 LLM 自动为每条记忆提取关键词、上下文和标签。\n")

    memories_to_add = [
        "Transformer 模型的核心创新是自注意力机制(Self-Attention)，"
        "它能并行处理序列中所有位置之间的关系，解决了 RNN 的长距离依赖问题。",

        "BERT 是一种双向编码器模型，通过掩码语言模型(MLM)和下一句预测(NSP)进行预训练，"
        "在 NLU 任务上取得了突破性表现。",

        "GPT 系列采用自回归解码器架构，通过不断增大模型规模和训练数据，"
        "涌现出了推理、代码生成等强大的 few-shot 能力。",

        "强化学习从人类反馈(RLHF)是将 LLM 与人类偏好对齐的关键技术，"
        "ChatGPT 的成功很大程度上归功于 RLHF 训练。",

        "检索增强生成(RAG)通过在推理时动态检索外部知识来减少 LLM 的幻觉，"
        "是目前企业级 AI 应用最主流的架构。",

        "向量数据库(如 ChromaDB, Pinecone)通过高效的近似最近邻搜索，"
        "为 RAG 和记忆系统提供语义检索能力。",

        "多模态大模型(如 GPT-4V, Gemini)能同时处理文本、图像、音频等多种模态，"
        "开启了通用人工智能的新方向。",
    ]

    memory_ids = []
    for i, content in enumerate(memories_to_add):
        print(f"  [+] 添加第 {i + 1}/{len(memories_to_add)} 条记忆...")
        t0 = time.time()
        mid = memory_system.add_note(content)
        elapsed = time.time() - t0
        memory_ids.append(mid)

        mem = memory_system.read(mid)
        print(f"      完成 ({elapsed:.1f}s)")
        print(f"      关键词: {mem.keywords}")
        print(f"      标签  : {mem.tags}")
        print(f"      链接数: {len(mem.links)}")
        print()

    # --- 第二阶段：检查记忆进化（链接建立情况） ---
    separator("第二阶段：查看记忆进化结果")
    print("  A-MEM 的核心特性：添加新记忆时自动分析与已有记忆的关系，")
    print("  决定是否 strengthen(加强连接) 或 update_neighbor(更新邻居标签)。\n")

    linked_count = 0
    for mid in memory_ids:
        mem = memory_system.read(mid)
        if mem.links:
            linked_count += 1
    print(f"  [统计] 共 {len(memory_ids)} 条记忆，{linked_count} 条建立了链接关系\n")

    # 打印每条记忆的详情
    for i, mid in enumerate(memory_ids):
        mem = memory_system.read(mid)
        print(f"  -- 记忆 #{i + 1} --")
        print_memory(mem, prefix="  ")
        print()

    # --- 第三阶段：语义搜索 + 邻居扩展 ---
    separator("第三阶段：语义搜索演示")
    print("  search_agentic 不仅返回直接匹配，还会通过链接返回相关邻居记忆。\n")

    queries = [
        "注意力机制是什么",
        "如何减少大模型的幻觉",
        "RLHF 人类偏好对齐",
    ]

    for query in queries:
        print(f"  [?] 搜索: \"{query}\"")
        results = memory_system.search_agentic(query, k=3)
        if results:
            for idx, r in enumerate(results):
                print_search_result(r, idx + 1)
        else:
            print("     (无结果)")
        print()

    # --- 第四阶段：更新与删除 ---
    separator("第四阶段：记忆更新与删除")

    # 更新第一条记忆
    target_id = memory_ids[0]
    old_mem = memory_system.read(target_id)
    print(f"  [~] 更新记忆 #1...")
    print(f"      旧标签: {old_mem.tags}")

    memory_system.update(
        target_id,
        tags=old_mem.tags + ["经典论文", "2017"],
        context="Transformer 是 Vaswani et al. 2017 提出的革命性架构，深刻影响了整个 NLP 和 AI 领域。"
    )
    updated_mem = memory_system.read(target_id)
    print(f"      新标签: {updated_mem.tags}")
    print(f"      新上下文: {updated_mem.context}")

    # 删除最后一条记忆
    last_id = memory_ids[-1]
    last_mem = memory_system.read(last_id)
    print(f"\n  [-] 删除记忆 #{len(memory_ids)}: \"{last_mem.content[:30]}...\"")
    memory_system.delete(last_id)
    print(f"      已删除，剩余记忆数: {len(memory_system.memories)}")

    # --- 第五阶段：再次搜索，观察变化 ---
    separator("第五阶段：删除后再次搜索")
    query = "多模态"
    print(f"  [?] 搜索: \"{query}\"（刚删除了多模态相关记忆）")
    results = memory_system.search_agentic(query, k=3)
    if results:
        for idx, r in enumerate(results):
            print_search_result(r, idx + 1)
    else:
        print("     (无直接匹配，多模态记忆已被删除)")
    print()

    # --- 总结 ---
    separator("体验总结")
    total = len(memory_system.memories)
    linked = sum(1 for m in memory_system.memories.values() if m.links)
    print(f"""
  A-MEM 记忆系统的核心能力:

  1. 智能元数据提取
     添加记忆时，LLM 自动提取关键词、上下文、标签

  2. 记忆进化 (Zettelkasten 方法)
     新记忆添加时自动与已有记忆建立语义关联
     - strengthen:       加强连接，建立链接
     - update_neighbor:  更新邻居的标签和上下文

  3. 语义搜索 + 图扩展
     search_agentic 先做向量检索，再沿链接返回邻居记忆

  4. CRUD 完整操作
     支持添加、读取、更新、删除

  当前统计:
     后端:          {LLM_BACKEND} / {LLM_MODEL}
     记忆总数:      {total}
     有链接的记忆:  {linked}
""")
    print("=" * 60)
    print("  演示结束!")
    print("=" * 60)


if __name__ == "__main__":
    main()

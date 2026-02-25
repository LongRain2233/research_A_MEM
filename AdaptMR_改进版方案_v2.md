# AdaptMR-v2：面向 Agent 记忆的查询感知自适应检索

> 改进版研究方案说明文档  
> 原始版本：2026-02-13 | 修订版本：2026-02-25  
> **核心调整说明：** 基于学术评审反馈，将主贡献从"五策略框架"重构为"实证分析（主） + 轻量级 Prompt-based 自适应检索（辅）"，全部实验通过 LLM API 完成，无需 GPU，单人可在 11 周内完成。
>
> **文档结构：**  §0-6 为学术方案正文（研究设计）；§7-10 为工程支撑（时间线/风险/投稿/文献）；**§11 为逐步执行手册（可直接操作）**。

---

## 0. 方案定位与贡献重构

### 0.1 评审反馈的核心问题

原方案（v1）存在以下需要解决的关键问题：

| 问题 | 具体表现 | 本版本的应对 |
|------|---------|------------|
| 创新性不足 | "分类→路由→策略"范式已被 Adaptive-RAG、Self-RAG 等工作覆盖 | 主贡献转向**首个对 Agent 记忆系统的查询类型感知失败分析**；方法论层面采用 LLM Planning 替代硬编码策略路由，规避 benchmark overfitting |
| 与评测集过度绑定 | 5 种类型直接来自 LongMemEval，方法为 benchmark 量身定制 | 将失败模式分析从数据集归纳出发，独立建立分类体系；增加跨数据集泛化实验 |
| 实验公平性 | AdaptMR 使用大量额外 LLM 调用，与基线不公平 | 明确报告各方法 LLM 调用次数与 token 消耗；增加 token-budget-matched baseline |
| 工程量超出单人能力 | 5 种策略 + 分类器 + 融合机制 + 5 组实验，17 周 | 方法大幅简化，实验精简为 4 组，工期缩短至 11 周 |
| 缺失关键相关工作 | 未与 Adaptive RAG 系列比较 | 补充 Related Work，增加与 Adaptive-RAG 等方法的实验对比 |

### 0.2 重构后的两个核心贡献

```
贡献 1（主）：对现有 Agent 记忆系统的查询类型感知失败分析
──────────────────────────────────────────────────────
首次系统性地揭示：不同类型的记忆查询在现有系统中的失败模式存在
根本性差异。这是一个独立的、有价值的 empirical findings 贡献，
不依赖于任何新方法的提出。

贡献 2（辅）：Query-Aware Retrieval Planning (QARP)
──────────────────────────────────────────────────────
基于上述分析，提出一种轻量级的 LLM 自主检索规划方法：
不通过人工定义固定分类树，而是让 LLM 根据查询特征自主生成
检索计划并执行后处理。全部通过 API 调用实现，无需 GPU。
```

---

## 1. 研究背景：为什么需要"查询感知"的记忆检索

### 1.1 现有系统的统一检索范式

当前主流 Agent 记忆系统（Mem0、A-MEM、MemGPT、H-MEM 等）在检索记忆时，几乎都采用**同一种方式**：

```
用户输入 query → 转为 embedding 向量 → 在记忆库中计算 cosine similarity
             → 返回 top-K 条最相似记忆 → 送入 LLM 生成回答
```

这种方式称为**统一向量检索（Unified Embedding Retrieval）**，其核心假设为：

> "语义上最相似的记忆 = 最有用的记忆"

### 1.2 这个假设在哪些场景下失效？

用一个具体场景说明。假设 Agent 的记忆库存储了以下 8 条记忆：

| 编号 | 时间 | 记忆内容 |
|------|------|---------|
| M1 | 1月5日 | 用户说："我住在北京朝阳区" |
| M2 | 2月10日 | 用户说："我最近在找杭州的工作" |
| M3 | 3月15日 | 用户说："我养了一只叫小白的猫" |
| M4 | 4月20日 | 用户说："我搬到杭州西湖区了" |
| M5 | 5月1日 | 用户说："推荐一下杭州的日料店" → Agent 推荐了"鹤矢日料" |
| M6 | 6月8日 | 用户说："我最近开始跑步了，每天5公里" |
| M7 | 7月12日 | 用户说："推荐一下杭州的川菜馆" → Agent 推荐了"蜀乡情" |
| M8 | 8月3日 | 用户说："今天跑了10公里，比上个月进步不少" |

对 5 种不同类型的查询，统一向量检索的表现如下：

---

#### 场景 A：事实提取 —— "我的猫叫什么名字？"

| 统一向量检索 | 结果 |
|-------------|------|
| 搜索"猫 名字"相关的 embedding | 找到 M3（"叫小白的猫"），相似度最高 |
| **效果** | ✅ **正确**。语义匹配在简单事实查询上表现良好 |

**结论：向量检索对事实提取类问题通常有效。**

---

#### 场景 B：时序推理 —— "我是先开始跑步还是先搬到杭州的？"

| 统一向量检索 | 结果 |
|-------------|------|
| 搜索"跑步 搬家 杭州"相关的 embedding | 可能返回 M4、M6、M8 |
| 返回给 LLM 的信息 | M4："搬到杭州"、M6："开始跑步"、M8："跑了10公里" |
| **问题** | ❌ **向量检索不携带时间顺序信息**。LLM 收到 3 条记忆，但呈现顺序是按相似度而非时间排列，无法支持"先后"判断 |

**统一向量检索缺陷：忽略时间维度，无法自然支持时序推理。**

---

#### 场景 C：知识更新 —— "我现在住在哪里？"

| 统一向量检索 | 结果 |
|-------------|------|
| 搜索"住在哪里 住址" | 返回 M1（"北京朝阳区"，相似度 0.92）和 M4（"杭州西湖区"，相似度 0.89） |
| **问题** | ❌ **M1 的相似度可能比 M4 更高**（M1 直接用"住在"，M4 用"搬到"），LLM 可能回答已过时的"北京" |

**统一向量检索缺陷：不区分信息新旧，旧信息因措辞更匹配而优先级更高。**

---

#### 场景 D：多跳推理 —— "我提到过的餐厅里哪种菜系最多？"

| 统一向量检索 | 结果 |
|-------------|------|
| 搜索"餐厅 菜系 次数" | 可能返回 M5、M7，也可能引入 M6（"跑步"因健康关联被检索） |
| **问题** | ❌ **需要先穷举所有餐厅记忆，再聚合统计**。单次 top-K 可能遗漏记忆，也可能引入噪声 |

**统一向量检索缺陷：单次检索无法支持"先收集、再聚合"的多步推理。**

---

#### 场景 E：拒绝判断 —— "我跟你说过我的血型吗？"

| 统一向量检索 | 结果 |
|-------------|------|
| 搜索"血型" | 无相关记忆，但 top-K 仍强制返回 K 条（如 M6"跑步"、M8"10公里"因健康关联被返回） |
| **问题** | ❌ **LLM 可能基于弱相关记忆臆造答案**（如"根据您经常运动的习惯来看..."） |

**统一向量检索缺陷：永远返回 K 条结果，从不说"我没找到相关记忆"。**

---

### 1.3 核心研究问题

> **RQ1（分析）：** 现有 Agent 记忆系统在不同查询类型上的失败率是否存在系统性差异？失败的根本原因是什么？  
> **RQ2（方法）：** 是否可以通过让 LLM 自主规划检索策略，在不引入专用训练的前提下，弥补统一检索在特定类型上的不足？

---

### 1.4 与现有自适应检索工作的关键区别

> ⚠️ **重要定位说明（审稿防雷）**

自适应检索在 RAG 领域已有大量工作，本研究须明确区别于以下方法：

| 相关工作 | 核心思路 | 与本研究的区别 |
|---------|---------|--------------|
| **Adaptive-RAG** (NAACL 2024) | 按查询复杂度决定是否检索（0/1/多步） | 针对通用 QA，不涉及 Agent 个人记忆的特殊属性（时序、冲突、无关） |
| **Self-RAG** (ICLR 2024) | 通过反思 token 动态触发和评估检索 | 训练依赖，针对通用知识检索，非个人记忆场景 |
| **FLARE** (EMNLP 2023) | 基于生成置信度判断是否需要检索 | 解决"何时检索"问题，非"如何从个人记忆中检索" |
| **IRCoT** (ACL 2023) | 交错检索与推理的 chain-of-thought | 针对多跳 QA，无记忆更新/冲突/拒绝场景 |

**本研究的独特性：** 上述工作均针对"静态知识库检索"；而 Agent 个人记忆库具有三个独特属性：
1. **记忆随时间更新**：同一实体的属性会发生变化（住址、工作等），产生新旧冲突
2. **记忆高度个性化**：查询往往需要跨越多条零散的个人事件进行聚合推理
3. **缺失信息是常态**：用户从未提及的信息不应被推断或捏造

这三个属性使得直接迁移通用 RAG 的自适应方法无法充分应对 Agent 记忆场景的挑战。

---

## 2. 主贡献一：查询类型感知的失败模式分析

> 这是本论文的**核心学术贡献**。即使没有后续的方法提出，这部分分析本身也构成独立的学术价值。

### 2.1 分析框架

在三个数据集（LongMemEval、LoCoMo、MSC 子集）上，对现有主流 Agent 记忆系统进行系统性评测，并从以下三个维度展开分析：

**维度 1：各查询类型的准确率差异**

对每个系统，分别报告 5 种查询类型上的准确率，回答：
- 哪种类型的失败率最高？
- 不同系统的失败模式是否一致？（是系统设计问题还是通用问题？）

**维度 2：失败案例的根本原因分类**

对失败案例进行手工标注（各类型抽取 30-50 个失败样本），将失败原因归类为：

| 失败类型 | 定义 | 典型场景 |
|---------|------|---------|
| **检索缺失（Retrieval Miss）** | 正确答案所在记忆未被检索到 | 多跳推理中遗漏相关记忆 |
| **时序混乱（Temporal Disorder）** | 检索到了相关记忆但顺序/时间信息丢失 | 时序推理中无法判断先后 |
| **冲突记忆干扰（Conflict Noise）** | 同时检索到新旧矛盾记忆，LLM 选择了旧的 | 知识更新类查询 |
| **无关记忆幻觉（Hallucination from Noise）** | 基于弱相关记忆捏造答案 | 拒绝判断类查询 |
| **上下文过载（Context Overload）** | 检索到的记忆过多，LLM 忽略了关键信息 | 多跳推理聚合阶段 |

**维度 3：统一检索与 Oracle 策略的性能上界差距**

通过 Oracle 分析（为每个查询手工选择最优检索操作）计算性能上界，量化"如果检索策略是最优的，性能能提升多少"。

### 2.2 分析价值

这一分析直接回答以下研究问题，具有独立发表价值：
- 现有 Agent 记忆系统是否存在查询类型盲区？（对哪种类型系统性地表现差？）
- 失败的瓶颈在检索阶段还是生成阶段？
- 统一检索范式的性能上界在哪里？

---

## 3. 主贡献二：Query-Aware Retrieval Planning (QARP)

> 基于失败分析的发现，提出轻量级方法。**核心设计原则：无需 GPU，无需训练，全部通过 LLM API 实现，可在任何现有记忆系统上即插即用。**

### 3.1 方法概述

QARP 的核心思路：**不预定义固定的分类-策略映射，而是让 LLM 根据查询特征自主生成检索计划（Retrieval Plan），再按计划执行检索和后处理。**

```
┌──────────────────────────────────────────────────────┐
│                    用户查询 (Query)                    │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│          [步骤1] Retrieval Planner (LLM)              │
│  输入：Query                                          │
│  输出：结构化检索计划 (JSON)                            │
│   ├── 检索关键词（改写后）                              │
│   ├── 是否需要时间排序                                  │
│   ├── 是否需要过滤旧信息                                │
│   ├── 是否需要分步检索（及子查询列表）                   │
│   └── 相关性阈值（是否允许拒绝回答）                     │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│          [步骤2] Plan-guided Retrieval                │
│  根据计划执行向量检索 + 规则后处理                       │
│   ├── 单步检索 或 多步迭代检索                          │
│   ├── 时间排序 / 最新优先过滤（若计划指定）              │
│   └── 相关性置信度判断（若计划指定阈值）                 │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│          [步骤3] Post-retrieval Refinement (LLM)     │
│  输入：Query + 检索计划 + 候选记忆                      │
│  输出：精炼后的记忆集合（含元信息标注）                   │
│   ├── 过滤不相关记忆                                    │
│   ├── 标注时间信息（如需要）                            │
│   ├── 标注"此信息已更新"（如检测到冲突）                 │
│   └── 若无相关记忆：输出 [NO_RELEVANT_MEMORY] 信号      │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│                  LLM 生成最终回答                       │
└──────────────────────────────────────────────────────┘
```

### 3.2 核心实现：Retrieval Planner

这是 QARP 的核心创新点——**不用规则分类，让 LLM 自主分析并生成结构化检索指令**。

```python
RETRIEVAL_PLANNER_PROMPT = """你是一个记忆检索规划器。根据用户查询的特征，
生成一个结构化的检索计划，指导后续从个人记忆库中检索信息。

用户查询：{query}

记忆库概况（最近10条记忆的摘要）：{memory_summary}

请分析查询的检索需求，输出以下 JSON 格式的检索计划：

{{
  "retrieval_keywords": ["改写后的检索关键词1", "关键词2"],
  "is_multi_step": false,          // 是否需要分步检索
  "sub_queries": [],               // 若分步，列出子查询序列
  "requires_temporal_order": false, // 检索结果是否需要按时间排序
  "prefer_latest": false,          // 是否优先返回最新版本（知识更新场景）
  "relevance_threshold": 0.7,      // 相关性置信度阈值（0-1），低于此值拒绝回答
  "post_processing_hint": ""       // 对后处理阶段的自然语言提示
}}

分析规则：
- 若查询涉及"先后/什么时候/最近一次"等时序词 → requires_temporal_order: true
- 若查询涉及"现在/目前/最新"等更新词 → prefer_latest: true
- 若查询需要综合多条信息（如"所有/哪些/统计"）→ is_multi_step: true
- 若查询在判断"是否说过/是否知道" → relevance_threshold 设为 0.8（严格阈值）
- 其他情况 → relevance_threshold 默认 0.65

只返回 JSON，不需要解释。"""
```

**与原方案（v1）硬编码路由的关键区别：**

| 维度 | v1（硬编码路由） | v2（LLM Planning） |
|------|----------------|-------------------|
| 类型分类 | 硬映射到 5 种固定类型 | LLM 理解查询意图，生成连续的参数空间 |
| 混合类型处理 | 需要额外的多策略融合模块 | 自然支持（如 `requires_temporal_order: true` 同时 `prefer_latest: true`） |
| 泛化能力 | 受限于预定义分类体系 | 不依赖固定分类，可泛化到任意查询 |
| Benchmark 绑定 | 5 种类型直接来自 LongMemEval | 分类逻辑在 prompt 中，可跨数据集适用 |
| GPU 需求 | 微调分类器需要 GPU | 完全 API-based，无需 GPU |

### 3.3 Plan-guided Retrieval 执行

```python
def qarp_retrieve(query, memory_store, llm, K=5):
    """
    QARP 主函数：生成检索计划并按计划执行检索
    全部通过 LLM API 实现，无需 GPU
    """
    # ===== Step 1: 生成检索计划 =====
    memory_summary = memory_store.get_recent_summary(n=10)
    plan_prompt = RETRIEVAL_PLANNER_PROMPT.format(
        query=query, memory_summary=memory_summary
    )
    plan = json.loads(llm.call(plan_prompt))  # 1 次 LLM 调用
    
    # ===== Step 2: 按计划执行检索 =====
    if plan["is_multi_step"] and plan["sub_queries"]:
        # 多步检索（对应多跳推理场景）
        all_candidates = []
        for sub_q in plan["sub_queries"]:
            hits = memory_store.semantic_search(sub_q, top_k=K)
            all_candidates.extend(hits)
        candidates = deduplicate(all_candidates)
    else:
        # 单步检索（大多数场景）
        search_query = " ".join(plan["retrieval_keywords"])
        candidates = memory_store.semantic_search(search_query, top_k=2*K)
    
    # 置信度筛选（对应拒绝判断场景）
    if candidates and candidates[0].similarity < plan["relevance_threshold"]:
        return {"memories": [], "has_relevant_memory": False,
                "plan": plan}
    
    # 时间排序（对应时序推理场景）
    if plan["requires_temporal_order"]:
        candidates.sort(key=lambda m: m.timestamp)
    
    # 最新优先（对应知识更新场景）
    if plan["prefer_latest"]:
        candidates = filter_to_latest(candidates, similarity_threshold=0.85)
    
    # ===== Step 3: 后处理精炼 =====
    refinement_prompt = f"""基于以下检索计划，对候选记忆进行精炼和标注：

用户查询：{query}
检索计划提示：{plan["post_processing_hint"]}
候选记忆（{len(candidates)} 条）：
{format_memories(candidates)}

请完成以下工作：
1. 过滤掉与查询无关的记忆（标注原因）
2. 若涉及时序，为每条记忆添加时间标签
3. 若存在同一话题的新旧记忆冲突，标注哪条更新并说明原因
4. 若候选记忆整体与查询不相关，返回 [NO_RELEVANT_MEMORY]

返回精炼后的记忆列表（JSON格式）。"""
    
    refined_result = llm.call(refinement_prompt)  # 1 次 LLM 调用
    
    return {
        "memories": parse_refined_memories(refined_result),
        "has_relevant_memory": "[NO_RELEVANT_MEMORY]" not in refined_result,
        "plan": plan
    }


def filter_to_latest(candidates, similarity_threshold=0.85):
    """
    对语义相似度 > threshold 的记忆，识别同主题的新旧版本，
    仅保留最新版本（无需 LLM 调用，纯时间戳逻辑）
    """
    groups = {}
    ungrouped = []
    
    for mem in candidates:
        placed = False
        for key, group in groups.items():
            if group[0].similarity_to(mem) > similarity_threshold:
                group.append(mem)
                placed = True
                break
        if not placed:
            groups[id(mem)] = [mem]
            ungrouped.append(mem)
    
    result = []
    for group in groups.values():
        if len(group) > 1:
            # 保留时间戳最新的，附加"已更新"标注
            group.sort(key=lambda m: m.timestamp, reverse=True)
            latest = group[0]
            latest.update_note = f"(已更新，此前为：{group[-1].content[:30]}...)"
            result.append(latest)
        else:
            result.extend(group)
    
    return result
```

### 3.4 LLM 调用次数与 Token 成本分析

> 针对实验公平性问题，明确 QARP 的额外开销：

| 查询类型 | QARP 额外 LLM 调用次数 | 估算 Token 消耗（GPT-4o-mini） | 额外成本/次 |
|---------|----------------------|-------------------------------|-----------|
| 事实提取 | 2 次（plan + refine） | ~400 tokens | ~$0.00012 |
| 时序推理 | 2 次 | ~500 tokens | ~$0.00015 |
| 知识更新 | 2 次 | ~500 tokens | ~$0.00015 |
| 多跳推理 | 2 次（plan 含子查询生成） | ~600 tokens | ~$0.00018 |
| 拒绝判断 | 1-2 次（低置信度时可提前结束） | ~300 tokens | ~$0.00009 |

**公平性保证措施：**
- 在实验中设置 **Token-Budget-Matched Baseline**：给基线方法（A-MEM、Mem0）同等 token 预算用于 LLM reranking，若 QARP 在相同 token 预算下仍更优，则说明贡献来自检索策略而非算力投入
- 绘制 **Accuracy vs. Token Budget** 的帕累托前沿曲线

---

## 4. 实验方案设计（精简版，单人可执行）

### 4.1 数据集

| 数据集 | 用途 | 查询数量 | 使用方式 |
|--------|------|---------|---------|
| **LongMemEval** | 主评估集 | 500 | 完整评测 + 失败案例标注 |
| **LoCoMo** | 泛化性验证 | ~300 | 跨数据集评测，验证方法不过拟合 LongMemEval |

> MSC 数据集由于缺乏查询类型标注，本版本暂不使用，可作为 future work。

### 4.2 基线方法（精简至可执行范围）

| 基线 | 说明 | 额外 LLM 调用 |
|------|------|--------------|
| **Unified-Embedding** | 标准 top-K 向量检索，所有查询同一策略 | 0 次 |
| **A-MEM** | Zettelkasten 结构化记忆 + 检索（已有代码） | 0 次 |
| **Mem0** | 动态记忆提取 + 向量检索（开源） | 0 次 |
| **Adaptive-RAG (适配版)** | 将 Adaptive-RAG 的复杂度分级策略迁移到记忆场景 | ~1 次 |
| **Token-Matched Baseline** | Unified-Embedding + 同等 token 预算的 LLM Reranking | 2 次 |
| **QARP (本方法)** | 本论文提出的方法 | 2 次 |
| **Oracle-Strategy** | 为每个查询手工选择最优检索操作（性能上界） | — |

> **说明：** 相比 v1 方案，移除了 Zep（工程部署复杂）和 LongMemEval-RAG（与 Oracle 重叠）。增加了 Adaptive-RAG 适配版和 Token-Matched Baseline 以应对审稿人关于公平性和相关工作的质疑。

### 4.3 评估指标

| 指标 | 计算方式 | 核心地位 |
|------|---------|---------|
| **各查询类型准确率（Per-type Accuracy）** | 每种查询类型的单独准确率 | ⭐⭐⭐ 最核心指标 |
| **整体 QA 准确率** | 正确回答数 / 总查询数 | ⭐⭐⭐ |
| **检索召回率 Recall@K** | 正确答案所在记忆被检索到的比例 | ⭐⭐ 检索质量 |
| **LLM 调用次数 / Token 消耗** | 各方法的额外 LLM 开销 | ⭐⭐ 公平性证明 |
| **端到端延迟** | 从查询到回答的平均时间 | ⭐ 实用性参考 |

### 4.4 实验设计（4 组，精简可执行）

#### 实验 1：失败模式分析（主贡献实验）

**目的：** 系统性证明"统一检索在不同查询类型上存在显著的失败率差异"，建立研究动机，回答 RQ1。

```
对 Unified-Embedding / A-MEM / Mem0 分别在 LongMemEval 上做完整评测
→ 按 5 种查询类型拆分准确率
→ 对各类型失败案例各抽取 40 个，手工标注失败原因（5 类）
→ 计算各失败类型的分布比例
→ 画出："失败率热力图 × 系统 × 查询类型"

预期发现：
  - 事实提取：所有系统表现较好（~70-80%）
  - 时序推理 + 知识更新：所有系统系统性表现差（~30-50%）
  - 失败原因以"时序混乱"和"冲突记忆干扰"为主
  → 这证明问题不是某个系统设计缺陷，而是统一检索范式的通病
```

**⚠️ 关键决策节点：若实验 1 发现各类型差距 < 5%，整个研究立论崩塌，需要重新选题。因此实验 1 必须在方案确定后立即执行。**

#### 实验 2：Oracle 上界分析

**目的：** 量化"最优检索策略"与"统一检索"之间的性能天花板差距，论证自适应检索的潜力空间。

```
对 LongMemEval 的各查询类型，手工执行最优检索操作：
  - 时序类：按时间戳排序后检索，显式标注时间
  - 更新类：仅返回每个话题的最新记忆
  - 多跳类：分步检索 + 手工聚合
  - 拒绝类：严格阈值过滤

→ 计算 Oracle 准确率
→ 与 Unified-Embedding 对比，画出"潜力空间条形图"
→ 预期：差距 ≥ 10%（尤其在时序和更新类型上）
```

#### 实验 3：QARP 主对比实验

**目的：** 展示 QARP 相对基线的整体效果，回答 RQ2。

```
在 LongMemEval（主）和 LoCoMo（泛化验证）上：
  - 对比 6 个基线 + QARP
  - 报告：整体准确率 + 各类型准确率 + LLM Token 消耗
  
重点展示：
  (1) QARP vs Unified-Embedding：整体和各类型的提升
  (2) QARP vs Token-Matched Baseline：相同 token 预算下，QARP 是否更优
  (3) QARP vs Adaptive-RAG 适配版：在记忆场景的特定优势
  (4) QARP 在 LoCoMo 上的性能（跨数据集泛化）
```

#### 实验 4：消融实验

**目的：** 验证 Retrieval Plan 各参数项的贡献。

| 消融变体 | 移除内容 | 预期影响 |
|---------|---------|---------|
| w/o Planning | 直接用 query 做向量检索（无 plan 生成） | 整体下降，尤其时序/更新类型 |
| w/o Temporal Ordering | 忽略 `requires_temporal_order` 参数 | 时序推理类下降 |
| w/o Latest Filtering | 忽略 `prefer_latest` 参数 | 知识更新类下降 |
| w/o Relevance Threshold | 固定阈值 = 0（永远返回结果） | 拒绝判断类下降 |
| w/o Multi-step | 忽略 `is_multi_step`，强制单步检索 | 多跳推理类下降 |
| w/o Post-refinement | 移除 Step 3 精炼 | 整体轻微下降 |

> **注意：** 消融实验同时验证了 QARP 设计的每个组件均有贡献，为 Reviewer 提供了"方法各部分确实有效"的证据。

---

## 5. 相关工作定位（新增，原方案缺失）

### 5.1 Agent 记忆系统

（保留原方案中的 Mem0、A-MEM、MemGPT、H-MEM、Zep、AriGraph、RMM、LightMem、PREMem 等引用）

### 5.2 自适应检索（重点补充，原方案缺失此节）

- **Adaptive-RAG (NAACL 2024)**：按查询复杂度分配检索资源，但针对通用知识库，不处理记忆的时序冲突和拒绝场景
- **Self-RAG (ICLR 2024)**：训练依赖型自适应检索，不适用于 API-only 场景
- **FLARE (EMNLP 2023)**：解决"何时检索"，不解决"如何从个人记忆中检索"
- **IRCoT (ACL 2023)**：多跳推理的交错检索，是 QARP 多步检索的重要参考，但无冲突/拒绝处理

**与本工作的关系：** 本工作将自适应检索的核心思想（查询敏感的检索策略）首次系统性地应用于 Agent 个人记忆场景，并通过 LLM Retrieval Planning 的方式避免了训练依赖和固定分类树的局限性。

### 5.3 查询感知检索的相关工作

- **Query2Doc / HyDE**：查询改写类方法，关注如何表示查询，本工作关注如何执行检索操作
- **RRR (Rewrite-Retrieve-Read)**：与 QARP 在"先分析再检索"的思路上相似，但无记忆管理场景的针对性设计

---

## 6. 论文叙事结构

```
Title: AdaptMR: Understanding and Addressing Query-Type-Specific 
       Retrieval Failures in Agent Memory Systems

Abstract:
  Agent 记忆系统普遍采用统一向量检索 
  → 我们首次系统分析：不同查询类型的失败率存在显著差异，失败模式各不相同
  → 基于分析，提出 QARP：LLM 自主生成检索计划的轻量级自适应方法
  → 无需 GPU，即插即用，在 LongMemEval 上显著提升（尤其时序/更新/拒绝类型）

Section 1 - Introduction:
  - Agent 长期记忆的重要性
  - 统一检索范式及其核心假设
  - 核心 Motivation：我们发现该假设对特定类型的查询系统性失效
  - 贡献声明：(1) 首个查询类型感知失败分析；(2) QARP 方法；(3) 跨系统实证评测

Section 2 - Related Work:
  - Agent 记忆系统
  - 自适应检索（Adaptive RAG 系列）← 新增，原方案缺失
  - 与本工作的定位区别

Section 3 - Failure Analysis (主贡献)
  - 3.1 实验设置（数据集、评测系统、失败类型分类体系）
  - 3.2 各查询类型的准确率差异（实验1结果）
  - 3.3 失败原因深入分析（失败模式标注结果）
  - 3.4 Oracle 上界分析（实验2结果）
  - 3.5 核心发现总结：统一检索的三类系统性缺陷

Section 4 - Method: QARP
  - 4.1 方法概述与设计原则
  - 4.2 Retrieval Planner：LLM 自主生成检索计划
  - 4.3 Plan-guided Retrieval 执行机制
  - 4.4 与 Adaptive RAG 方法的区别

Section 5 - Experiments
  - 5.1 实验设置（基线、指标、公平性保证）
  - 5.2 主实验结果（实验3：对比实验）
  - 5.3 消融实验（实验4）
  - 5.4 效率与成本分析（token 消耗对比）
  - 5.5 案例分析（各类型的具体检索对比）

Section 6 - Conclusion
```

---

## 7. 时间线（单人、无 GPU，11 周可完成）

| 阶段 | 时间 | 任务 | 关键产出 |
|------|------|------|---------|
| **[立即] 验证假设** | 第 1-2 周 | 跑实验 1（A-MEM + Uniform baseline 在 LongMemEval 上按类型分析） | **决策数据**：若各类型差异 < 5%，需重新选题 |
| **分析深化** | 第 3-4 周 | 补充 Mem0 实验 + 失败案例手工标注 + Oracle 上界分析（实验2） | 主贡献完整数据 |
| **方法实现** | 第 5-7 周 | 实现 QARP（Planner + Plan-guided Retrieval + Refinement） | 可运行代码 |
| **对比实验** | 第 8-9 周 | 跑实验 3（主对比）+ 实验 4（消融）+ LoCoMo 泛化实验 | 完整实验结果 |
| **论文写作** | 第 10-11 周 | 写作 + 图表制作 + 修改 | 论文初稿 |

**总计：约 11 周（2.5 个月），全部通过 API 调用完成，无需 GPU。**

---

## 8. 关键风险与应对

| 风险 | 概率 | 应对方案 |
|------|------|---------|
| 实验 1 发现各类型差距不显著（< 5%） | 中 | 研究方向不成立，需提前止损。此为最高优先级验证项 |
| QARP 在部分类型上不如基线 | 中低 | 可接受，只要在失败率最高的 1-2 个类型上有显著提升即可支撑 Claim |
| LLM Planner 生成质量不稳定（JSON 格式错误） | 高 | 添加输出解析校验 + fallback 到统一检索；在论文中报告成功率 |
| Reviewer 认为创新性不足（empirical 贡献是否够） | 中 | 强调"首个"：首个对 Agent 记忆查询类型失败的系统分析；同时确保 QARP 的实验效果扎实 |
| 跨数据集泛化结果不好 | 中 | 不掩盖，诚实讨论。分析原因（数据集分布差异），作为 limitation 处理，不影响主贡献 |

---

## 9. 投稿目标建议

| 会议/期刊 | 适合度 | 理由 |
|-----------|--------|------|
| **COLING 2025/2026** | ★★★★★ | 接受实证分析 + 轻量方法类工作，对 Agent NLP 应用友好 |
| **ECIR 2026** | ★★★★★ | 信息检索专业会议，对"检索策略分析与优化"天然契合 |
| **EMNLP Findings 2025** | ★★★★ | Findings 对非顶级创新但扎实的实证工作接受度高 |
| **NAACL 2026** | ★★★ | 需要方法贡献更突出，当前版本偏分析 |
| **IPM / Information Sciences（期刊）** | ★★★★ | CCF B 期刊，时间宽裕，允许更充分的实证讨论 |

---

## 10. 可引用文献（补充更新）

| 引用目的 | 文献 | 备注 |
|---------|------|------|
| Agent 记忆领域综述 | Memory in the Age of AI Agents | `3S9QUJWQ_Hu等` |
| 记忆流+反思 | Generative Agents | `BM2EFQ6T_Park等` |
| OS式记忆管理 | MemGPT | `CHHWJSWN_Packer等` |
| 工业级记忆 | Mem0 | `BHW9GDA8_Chhikara等` |
| 结构化记忆 | A-MEM | `7PAIEXBT_Xu等` |
| 时序知识图谱 | Zep | `D9VGYK42_Rasmussen等` |
| 双重检索 | AriGraph | `GTEIPZMQ_Anokhin等` |
| RL 记忆检索 | RMM | `Q3PUFMNM_Tan等` |
| 层次化检索 | H-MEM | `RZI3M53F_Sun&Zeng` |
| 轻量级记忆 | LightMem | `IDEYWTW4_Fang等` |
| 推理前移 | PREMem | `AB57ZFXD_Kim等` |
| **自适应 RAG（新增）** | Adaptive-RAG (Jeong et al., NAACL 2024) | 最直接的竞争性工作，需在论文中明确区分 |
| **自适应 RAG（新增）** | Self-RAG (Asai et al., ICLR 2024) | 需与本方法的 training-free 特性做对比 |
| **交错检索推理（新增）** | IRCoT (Trivedi et al., ACL 2023) | 多步检索的参考工作 |
| **LLM Planner（新增）** | ReAct (Yao et al., ICLR 2023) | QARP 的 Planning 机制与 ReAct 有概念联系 |

---

---

## 11. 详细执行手册（逐步操作，可直接照做）

> 本节是面向实际操作的行动指南，与 §4（实验方案）和 §7（时间线）一一对应。  
> **当前实验状态（2026-02-25）：** `test_advanced_adaptmr.py` 正在运行，已完成 Sample 0~2（共 497 题），当前 Sample 3/10 正在构建记忆（Turn ~480/629），预计今晚完成全部 10 个 sample。

---

### 全局路线图

```
【现在】等待当前实验跑完（Sample 3~10）
          │
          ▼
【第1~2周】第1阶段：实验1——失败模式分析  ← ⚠️ 关键决策节点
          │
          ├── 差距 < 5%  →  ❌ 立论不成立，重新选题
          └── 差距 ≥ 10% →  ✅ 继续
          │
          ▼
【第3周】第2阶段：实验2——Oracle 上界分析
          │
          ▼
【第4~6周】第3阶段：实现 QARP 方法代码
          │
          ▼
【第7~8周】第4阶段：实验3+4——对比实验 + 消融实验
          │
          ▼
【第9~11周】第5阶段：论文撰写
```

---

### 第 0 步：等待当前实验（现在进行中）

**你需要做的：什么都不做，让 `test_advanced_adaptmr.py` 跑完。**

完成标志：日志文件末尾出现 `FINAL COMPARISON RESULTS` 汇总表格。

日志文件位置：
```
D:\research\research_A_MEM\A-mem-ollma\A-mem\logs\adaptmr_compare_qwen2.5_1.5b_ollama_<timestamp>.log
```

---

### 第 1 阶段：实验 1——失败模式分析（第 1~2 周）

#### Step 1-A：提取各类型准确率（实验跑完后立即执行，约 1 小时）

在 `D:\research\research_A_MEM\A-mem-ollma\A-mem\` 目录下新建并运行以下脚本：

```python
# 文件名：analyze_exp1_accuracy.py
import json
import numpy as np
from collections import defaultdict

# 读取实验结果（根据实际文件名调整）
result_file = "results/adaptmr_test_run.json"
with open(result_file, "r", encoding="utf-8") as f:
    data = json.load(f)

CAT_NAMES = {
    1: "Multi-hop Reasoning",
    2: "Temporal Reasoning",
    3: "Open-domain Factual",
    4: "Single-hop Factual",
    5: "Adversarial (Abstention)"
}

# 只看 baseline 结果
cat_f1 = defaultdict(list)
for item in data["baseline_results"]:
    cat_f1[item["category"]].append(item["metrics"]["f1"])

print(f"\n{'='*70}")
print(f"【实验1结果】Baseline（统一向量检索）按查询类型准确率")
print(f"{'='*70}")
print(f"{'类型':<32} {'F1均值':>8} {'样本数':>7} {'失败率(F1<0.3)':>14}")
print(f"{'-'*70}")

results_for_decision = {}
all_f1 = []
for cat in sorted(cat_f1.keys()):
    f1s = cat_f1[cat]
    mean = np.mean(f1s)
    fail_rate = sum(1 for x in f1s if x < 0.3) / len(f1s)
    all_f1.extend(f1s)
    results_for_decision[cat] = mean
    print(f"  Cat{cat} {CAT_NAMES[cat]:<27} {mean*100:>7.1f}%  "
          f"{len(f1s):>6}    {fail_rate*100:>10.1f}%")

print(f"{'-'*70}")
print(f"  {'Overall':<31} {np.mean(all_f1)*100:>7.1f}%  {len(all_f1):>6}")

best  = max(results_for_decision.values())
worst = min(results_for_decision.values())
gap   = (best - worst) * 100
print(f"\n【关键指标】最好类型 vs 最差类型 F1 差距 = {gap:.1f}%")

# ====== 决策判断 ======
print(f"\n{'='*70}")
if gap >= 10:
    print(f"✅ 差距 {gap:.1f}% ≥ 10%  →  立论成立，继续推进！")
elif gap >= 5:
    print(f"⚠️  差距 {gap:.1f}% 在 5~10% 之间  →  立论偏弱，建议补充 LongMemEval 数据再判断")
else:
    print(f"❌ 差距 {gap:.1f}% < 5%  →  统一检索无显著类型盲区，需重新选题！")
print(f"{'='*70}")
```

**运行命令：**
```bash
cd D:\research\research_A_MEM\A-mem-ollma\A-mem
python analyze_exp1_accuracy.py
```

**⚠️ 这是整个项目的决策节点：**

| 结果 | 行动 |
|------|------|
| 差距 ≥ 10% | ✅ 立论成立，执行 Step 1-B |
| 差距 5~10% | ⚠️ 先补充 LongMemEval 数据集的实验再判断 |
| 差距 < 5%  | ❌ 立论不成立，停止，需重新选题 |

---

#### Step 1-B：导出失败案例（约 1 小时）

```python
# 文件名：export_failures.py
import json
from collections import defaultdict

with open("results/adaptmr_test_run.json", "r", encoding="utf-8") as f:
    data = json.load(f)

CAT_NAMES = {1:"multihop", 2:"temporal", 3:"opendomain", 4:"singlehop", 5:"adversarial"}

failures = defaultdict(list)
for item in data["baseline_results"]:
    if item["metrics"]["f1"] < 0.3:   # "失败"定义：F1 < 0.3
        failures[item["category"]].append({
            "question":     item["question"],
            "prediction":   item["prediction"],   # baseline 的错误答案
            "reference":    item["reference"],    # 正确答案
            "f1":           round(item["metrics"]["f1"], 3),
            "sample_id":    item["sample_id"],
            # ↓ 你需要手工填写的字段
            "failure_type": "",    # 填入下方5种代码之一
            "failure_note": ""     # 可选：1句备注说明
        })

output = {}
for cat, cases in sorted(failures.items()):
    sampled = cases[:40]   # 每类最多取 40 条
    key = f"cat{cat}_{CAT_NAMES[cat]}"
    output[key] = sampled
    print(f"Cat{cat} {CAT_NAMES[cat]}: 共 {len(cases)} 个失败，导出 {len(sampled)} 条")

with open("results/failure_cases_annotation.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
print("\n✅ 已导出到 results/failure_cases_annotation.json")
```

---

#### Step 1-C：手工标注失败原因（核心工作，预计 2~3 天）

用 VS Code 打开 `results/failure_cases_annotation.json`，逐条填写 `"failure_type"` 字段。

**5 种失败类型代码及判断标准：**

| 代码 | 名称 | 怎么判断 |
|------|------|---------|
| `MISS` | 检索缺失 | 答案明显存在于对话历史，但 prediction 完全错——说明根本没检索到 |
| `TEMPORAL` | 时序混乱 | 内容方向对但时间/顺序错——比如答了"March"但应该是"June" |
| `CONFLICT` | 旧信息干扰 | 答了一个"过时的"版本——比如旧地址、旧工作、已更新前的信息 |
| `HALLUC` | 噪声幻觉 | Cat5 中本应回答"Not mentioned"，但 LLM 基于无关记忆编造了内容 |
| `GEN` | 生成失败 | 记忆看起来被正确检索到，但 LLM 输出时语言组织失误（与检索无关） |

**操作流程：**
```
打开 failure_cases_annotation.json
→ 每条记录：
    读 "question"（问题是什么）
    读 "reference"（正确答案是什么）
    读 "prediction"（baseline 答错了什么）
→ 对照上表判断属于哪种失败类型
→ 填入 "failure_type": "MISS"（或其他代码）
→ 可选：在 "failure_note" 写 1 句说明
```

**完成标准：** 每个类型至少标注 20 条，共约 100 条有效标注。

---

#### Step 1-D：统计分布并生成论文图（约 2 小时）

```python
# 文件名：analyze_exp1_failure.py
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open("results/failure_cases_annotation.json", "r", encoding="utf-8") as f:
    data = json.load(f)

CAT_LABELS = {
    "cat1_multihop":    "Cat1\nMulti-hop",
    "cat2_temporal":    "Cat2\nTemporal",
    "cat3_opendomain":  "Cat3\nOpen-domain",
    "cat4_singlehop":   "Cat4\nSingle-hop",
    "cat5_adversarial": "Cat5\nAdversarial"
}
FAILURE_TYPES = ["MISS", "TEMPORAL", "CONFLICT", "HALLUC", "GEN"]
COLORS        = ["#e74c3c", "#e67e22", "#f39c12", "#9b59b6", "#95a5a6"]

cat_keys = [k for k in CAT_LABELS if k in data]
counts = {ft: [] for ft in FAILURE_TYPES}
for cat_key in cat_keys:
    annotated = [c for c in data[cat_key] if c.get("failure_type")]
    total = max(len(annotated), 1)
    for ft in FAILURE_TYPES:
        counts[ft].append(
            sum(1 for c in annotated if c["failure_type"] == ft) / total * 100
        )

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cat_keys))
bottoms = np.zeros(len(cat_keys))
for ft, color in zip(FAILURE_TYPES, COLORS):
    ax.bar(x, counts[ft], bottom=bottoms, label=ft,
           color=color, alpha=0.85, width=0.6)
    bottoms += np.array(counts[ft])

ax.set_xticks(x)
ax.set_xticklabels([CAT_LABELS[k] for k in cat_keys], fontsize=11)
ax.set_ylabel("Failure Mode Distribution (%)", fontsize=11)
ax.set_title(
    "Figure 2: Failure Mode Analysis by Query Type\n"
    "(Baseline: Unified Embedding Retrieval)", fontsize=11
)
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig("results/fig2_failure_distribution.png", dpi=150, bbox_inches='tight')
print("✅ 图已保存：results/fig2_failure_distribution.png")
```

**第 1 阶段完成物清单：**
- [ ] 📊 各类型准确率表格（论文 Table 1 素材）
- [ ] 📊 Figure 2：失败原因分布堆叠图（`fig2_failure_distribution.png`）
- [ ] 📝 `failure_cases_annotation.json`（200 条标注，论文 case study 素材）

---

### 第 2 阶段：实验 2——Oracle 上界分析（第 3 周）

> **好消息：** 你已有的 `test_adaptmr_comparison.py` 中，AdaptMR 使用了 ground-truth category 做路由（`STRATEGY_MAP[cat]`），**这本质上就是 Oracle 策略**。因此实验 2 不需要额外跑实验，直接从已有结果分析即可。

#### Step 2-A：生成 Oracle vs Baseline 对比（约 2 小时）

```python
# 文件名：analyze_exp2_oracle.py
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

with open("results/adaptmr_test_run.json", "r", encoding="utf-8") as f:
    data = json.load(f)

CAT_NAMES = {1:"Multi-hop", 2:"Temporal", 3:"Open-domain", 4:"Single-hop", 5:"Adversarial"}

baseline_f1 = defaultdict(list)
oracle_f1   = defaultdict(list)

for b, a in zip(data["baseline_results"], data["adaptmr_results"]):
    cat = b["category"]
    baseline_f1[cat].append(b["metrics"]["f1"])
    oracle_f1[cat].append(a["metrics"]["f1"])  # AdaptMR = Oracle（使用真实类别路由）

print(f"\n{'='*75}")
print(f"【实验2】Oracle 上界：统一检索 vs 最优策略对比")
print(f"{'='*75}")
print(f"{'类型':<30} {'Baseline F1':>12} {'Oracle F1':>11} {'提升空间':>10}")
print(f"{'-'*75}")

all_base, all_oracle = [], []
gains = {}
for cat in sorted(baseline_f1.keys()):
    b = np.mean(baseline_f1[cat]) * 100
    o = np.mean(oracle_f1[cat]) * 100
    gain = o - b
    gains[cat] = gain
    all_base.extend(baseline_f1[cat])
    all_oracle.extend(oracle_f1[cat])
    marker = "🔴" if gain > 5 else ("🟡" if gain > 0 else "🟢")
    print(f"  Cat{cat} {CAT_NAMES[cat]:<25} {b:>10.1f}%  {o:>10.1f}%  {marker}{gain:>+7.1f}%")

print(f"{'-'*75}")
b_all = np.mean(all_base)   * 100
o_all = np.mean(all_oracle) * 100
print(f"  {'Overall':<29} {b_all:>10.1f}%  {o_all:>10.1f}%   {o_all-b_all:>+7.1f}%")
print(f"\n结论：Oracle 策略整体提升 {o_all-b_all:.1f}%，"
      f"提升最大的类型：Cat{max(gains, key=gains.get)} {CAT_NAMES[max(gains, key=gains.get)]}"
      f"（{max(gains.values()):+.1f}%）")

# 绘制"潜力空间"条形图
cats  = sorted(baseline_f1.keys())
x     = np.arange(len(cats))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
b_vals = [np.mean(baseline_f1[c]) * 100 for c in cats]
o_vals = [np.mean(oracle_f1[c])   * 100 for c in cats]
ax.bar(x - width/2, b_vals, width, label='Baseline (Unified)',  color='#3498db', alpha=0.85)
ax.bar(x + width/2, o_vals, width, label='Oracle (Best Strategy)', color='#2ecc71', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"Cat{c}\n{CAT_NAMES[c]}" for c in cats], fontsize=9)
ax.set_ylabel("F1 Score (%)", fontsize=11)
ax.set_title("Figure 3: Performance Gap Between Unified Retrieval and Oracle Strategy\n"
             "(Gap = Potential Improvement Space for Adaptive Retrieval)", fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 100)
# 标注提升空间
for i, (b, o) in enumerate(zip(b_vals, o_vals)):
    if o > b:
        ax.annotate(f'+{o-b:.1f}%', xy=(i + width/2, o + 1),
                    ha='center', fontsize=8, color='#27ae60')
plt.tight_layout()
plt.savefig("results/fig3_oracle_gap.png", dpi=150, bbox_inches='tight')
print("✅ 图已保存：results/fig3_oracle_gap.png")
```

**第 2 阶段完成物清单：**
- [ ] 📊 Figure 3：Oracle vs Baseline 性能差距图（`fig3_oracle_gap.png`）
- [ ] ✅ 验证：Oracle 整体高于 Baseline ≥ 10%（若不到 5% 说明策略本身无效）

---

### 第 3 阶段：实现 QARP 方法（第 4~6 周）

#### Step 3-A：申请 LLM API Key（第 4 周初，1 天）

LoCoMo 是英文数据集，推荐使用 **OpenAI GPT-4o-mini**：
- 成本估算：500 题 × 2 次调用 × ~500 tokens = 50 万 tokens ≈ **$0.075**，约 **¥0.5**
- 备选：DeepSeek API（国内访问稳定，成本更低）

#### Step 3-B：新建 `qarp_retrieval.py` 并实现三个核心函数（第 4~5 周，约 1 周编码）

```python
# qarp_retrieval.py —— 放到 A-mem 目录下
import json
import openai  # 或 import anthropic / from openai import OpenAI

# ===================================================
# 核心 Prompt：Retrieval Planner
# ===================================================
RETRIEVAL_PLANNER_PROMPT = """You are a memory retrieval planner for a personal AI assistant.
Analyze the user's query and generate a structured retrieval plan.

Query: {query}

Output a JSON retrieval plan with this exact format:
{{
  "retrieval_keywords": ["keyword1", "keyword2"],
  "is_multi_step": false,
  "sub_queries": [],
  "requires_temporal_order": false,
  "prefer_latest": false,
  "relevance_threshold": 0.65,
  "post_processing_hint": ""
}}

Rules:
- If query contains temporal words (when, before, after, first, last time) → requires_temporal_order: true
- If query asks for current/latest state (currently, now, recently changed) → prefer_latest: true
- If query needs to aggregate multiple items (all, which ones, how many times) → is_multi_step: true with sub_queries
- If query checks existence (did I ever, do you know my) → relevance_threshold: 0.80
- Default relevance_threshold: 0.65

Return JSON only, no explanation."""


def generate_retrieval_plan(query: str, llm_client) -> dict:
    """Step 1: LLM 生成结构化检索计划（1 次 API 调用）"""
    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": RETRIEVAL_PLANNER_PROMPT.format(query=query)}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # Fallback：返回默认计划
        return {
            "retrieval_keywords": query.split()[:5],
            "is_multi_step": False,
            "sub_queries": [],
            "requires_temporal_order": False,
            "prefer_latest": False,
            "relevance_threshold": 0.65,
            "post_processing_hint": ""
        }


def plan_guided_retrieval(plan: dict, store, k: int = 10) -> tuple:
    """Step 2: 按计划执行检索（0 次额外 LLM 调用，纯向量操作）"""
    
    if plan.get("is_multi_step") and plan.get("sub_queries"):
        # 多步检索：分步执行子查询，合并去重
        all_indices, all_scores = [], []
        seen = set()
        for sub_q in plan["sub_queries"]:
            idx, scores = store.semantic_search(sub_q, k=k)
            for i, s in zip(idx, scores):
                if i not in seen:
                    all_indices.append(i)
                    all_scores.append(s)
                    seen.add(i)
        indices, scores = all_indices[:2*k], all_scores[:2*k]
    else:
        # 单步检索
        search_query = " ".join(plan.get("retrieval_keywords", []))
        if not search_query.strip():
            search_query = "memory"
        indices, scores = store.semantic_search(search_query, k=2*k)
    
    # 置信度过滤（拒绝判断）
    threshold = plan.get("relevance_threshold", 0.65)
    if len(scores) > 0 and float(scores[0]) < threshold:
        return [], scores, True   # has_no_relevant=True
    
    # 时间排序（时序推理）
    if plan.get("requires_temporal_order"):
        from datetime import datetime
        pairs = list(zip(indices, scores))
        def get_ts(idx):
            from test_adaptmr_comparison import _parse_timestamp
            m = store.get_memory(idx)
            ts = getattr(m, 'timestamp', '')
            return _parse_timestamp(ts) or datetime.min
        pairs.sort(key=lambda x: get_ts(x[0]))
        indices = [p[0] for p in pairs[:k]]
        scores  = [p[1] for p in pairs[:k]]
    
    # 最新优先过滤（知识更新）
    if plan.get("prefer_latest"):
        indices, scores = _filter_to_latest(indices, scores, store)
    
    return indices[:k], scores[:k], False


def post_retrieval_refinement(query: str, plan: dict, indices: list,
                               store, llm_client) -> str:
    """Step 3: LLM 精炼候选记忆（1 次 API 调用）"""
    if not indices:
        return "[NO_RELEVANT_MEMORY]"
    
    # 格式化候选记忆
    memories_text = ""
    for idx in indices:
        memories_text += store.format_memory(idx, include_time_prefix=True) + "\n"
    
    hint = plan.get("post_processing_hint", "")
    prompt = f"""Based on the retrieval plan, refine the candidate memories for the query.

Query: {query}
Hint: {hint}

Candidate memories:
{memories_text}

Instructions:
1. Remove memories clearly unrelated to the query
2. If temporal order matters, ensure memories are presented chronologically with explicit timestamps
3. If there are conflicting old/new versions of the same fact, keep only the latest and note it was updated
4. If NO memory is actually relevant to the query, return exactly: [NO_RELEVANT_MEMORY]

Return the refined memory context as plain text."""

    response = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content


def qarp_retrieve(query: str, store, llm_client, k: int = 10) -> dict:
    """QARP 主函数：串联三个步骤，总计 2 次 LLM API 调用"""
    plan = generate_retrieval_plan(query, llm_client)               # 第 1 次 LLM 调用
    indices, scores, no_relevant = plan_guided_retrieval(plan, store, k)  # 0 次 LLM
    
    if no_relevant:
        return {
            "context": "[NO_RELEVANT_MEMORY]",
            "has_relevant_memory": False,
            "plan": plan,
            "llm_calls": 1
        }
    
    context = post_retrieval_refinement(query, plan, indices, store, llm_client)  # 第 2 次 LLM 调用
    return {
        "context": context,
        "has_relevant_memory": "[NO_RELEVANT_MEMORY]" not in context,
        "plan": plan,
        "llm_calls": 2
    }


def _filter_to_latest(indices, scores, store, sim_threshold=0.82):
    """保留同主题记忆中时间戳最新的版本（纯规则，无 LLM）"""
    from sklearn.metrics.pairwise import cosine_similarity
    from datetime import datetime
    import numpy as np
    
    if len(indices) <= 1:
        return indices, scores
    
    embs = store.embeddings[indices]
    sim_matrix = cosine_similarity(embs)
    
    kept = []
    skip = set()
    for i in range(len(indices)):
        if i in skip:
            continue
        group = [i]
        for j in range(i+1, len(indices)):
            if j not in skip and sim_matrix[i][j] > sim_threshold:
                group.append(j)
                skip.add(j)
        if len(group) > 1:
            # 取时间戳最新的
            def get_ts(pos):
                from test_adaptmr_comparison import _parse_timestamp
                m = store.get_memory(indices[pos])
                ts = getattr(m, 'timestamp', '')
                return _parse_timestamp(ts) or datetime.min
            best = max(group, key=get_ts)
            kept.append(best)
        else:
            kept.append(i)
    
    return [indices[i] for i in kept], [scores[i] for i in kept]
```

#### Step 3-C：集成到评测脚本（第 6 周，约 3 天）

在 `test_adaptmr_comparison.py` 的 `run_comparison()` 函数中，在原有 baseline + AdaptMR(Oracle) 两路对比基础上，新增 QARP 第三路：

```python
# 在 run_comparison() 主循环中添加（示意）
from qarp_retrieval import qarp_retrieve
import openai

llm_client = openai.OpenAI(api_key="YOUR_KEY")

# --- QARP retrieval ---
qarp_result = qarp_retrieve(qa.question, store, llm_client, k=retrieve_k)
qarp_context = qarp_result["context"]
qarp_pred = generate_answer(llm_controller, qarp_context, qa.question, cat, qa.final_answer)
qarp_token_count += qarp_result["llm_calls"] * 450  # 估算每次调用 token
```

**第 3 阶段完成物清单：**
- [ ] `qarp_retrieval.py`（QARP 实现代码，可运行）
- [ ] 修改后的评测脚本（支持三方对比：Baseline / Oracle / QARP）

---

### 第 4 阶段：对比实验 + 消融实验（第 7~8 周）

#### Step 4-A：实验 3 主对比实验（第 7 周）

在完整 LoCoMo 10 个 sample 上运行三方对比：

```bash
cd D:\research\research_A_MEM\A-mem-ollma\A-mem
python test_adaptmr_comparison.py \
    --mode three_way \
    --output results/exp3_main_comparison.json
```

**预期输出（论文 Table 2 雏形）：**

```
Category          Baseline F1    Oracle F1    QARP F1    QARP vs Baseline
Cat1 Multi-hop       28.3%          41.2%       36.5%        +8.2%
Cat2 Temporal        31.5%          52.8%       47.1%       +15.6%
Cat3 Open-domain     52.1%          55.3%       53.8%        +1.7%
Cat4 Single-hop      55.8%          57.2%       56.1%        +0.3%
Cat5 Adversarial     22.4%          48.9%       41.3%       +18.9%
Overall              38.0%          51.1%       47.0%        +9.0%
```

**关键判断标准：**
- QARP 整体高于 Baseline → ✅ 方法有效
- QARP 在 Cat2/Cat5 改善最大 → ✅ Planning 抓住了关键失败类型
- QARP 与 Oracle 差距合理（< 10%）→ ✅ LLM Planning 接近最优

#### Step 4-B：实验 4 消融实验（第 8 周）

在 `qarp_retrieval.py` 中添加消融变体函数，逐个关闭 Plan 参数：

```python
def qarp_retrieve_ablation(query, store, llm_client, k=10, ablate=None):
    """
    ablate 参数：
      "no_planning"    - 直接用原始 query 做向量检索，跳过 LLM Planning
      "no_temporal"    - 强制 requires_temporal_order=False
      "no_latest"      - 强制 prefer_latest=False
      "no_threshold"   - 强制 relevance_threshold=0（永远返回结果）
      "no_multistep"   - 强制 is_multi_step=False
      "no_refinement"  - 跳过 Step 3 后处理精炼
    """
    if ablate == "no_planning":
        # 完全退化为统一检索
        indices, scores = store.semantic_search(query, k=k)
        context = "\n".join(store.format_memory(i) for i in indices)
        return {"context": context, "has_relevant_memory": True, "llm_calls": 0}
    
    plan = generate_retrieval_plan(query, llm_client)
    
    if ablate == "no_temporal":   plan["requires_temporal_order"] = False
    if ablate == "no_latest":     plan["prefer_latest"] = False
    if ablate == "no_threshold":  plan["relevance_threshold"] = 0.0
    if ablate == "no_multistep":  plan["is_multi_step"] = False
    
    indices, scores, no_relevant = plan_guided_retrieval(plan, store, k)
    
    if ablate == "no_refinement":
        context = "\n".join(store.format_memory(i) for i in indices)
        return {"context": context, "has_relevant_memory": True, "llm_calls": 1}
    
    context = post_retrieval_refinement(query, plan, indices, store, llm_client)
    return {"context": context, "has_relevant_memory": True, "llm_calls": 2}
```

**第 4 阶段完成物清单：**
- [ ] 📊 Table 2：主对比实验结果（Baseline / Oracle / QARP × 5 类型 + Overall）
- [ ] 📊 Table 3：消融实验结果（6 种变体 × 5 类型）
- [ ] 📊 Figure 4：Token 消耗 vs 性能帕累托曲线（公平性证明）

---

### 第 5 阶段：论文撰写（第 9~11 周）

#### 推荐写作顺序

```
第9周前半段：Section 3（失败分析）
             ↑ 数据最扎实，先写，建立信心
第9周后半段：Section 4（QARP 方法描述）
             ↑ 对照代码写，不会出错
第10周前半段：Section 5（实验结果）
             ↑ 填表格、写分析段落
第10周后半段：Section 2（相关工作）
             ↑ 对照 §5 相关工作节填写
第11周前半段：Section 1（Introduction）
             ↑ 全文写完后再提炼，最准确
第11周后半段：Abstract + 全文润色
```

#### 每一节需要的素材

| 章节 | 核心素材 | 来源文件 |
|------|---------|---------|
| Section 3.2 | 各类型准确率表 | `analyze_exp1_accuracy.py` 输出 |
| Section 3.3 | 失败原因分布图 | `fig2_failure_distribution.png` |
| Section 3.4 | Oracle 上界图 | `fig3_oracle_gap.png` |
| Section 4 | QARP 架构图 + Prompt 代码 | `qarp_retrieval.py` |
| Section 5.2 | 主对比结果表 | `exp3_main_comparison.json` |
| Section 5.3 | 消融结果表 | 消融实验 JSON |
| Section 5.4 | Token 消耗对比 | 实验过程统计 |
| Section 5.5 | 3~5 个 case 对比 | `failure_cases_annotation.json` |

---

### 汇总时间表

| 周次 | 具体任务 | 关键产出 | 决策点 |
|------|---------|---------|-------|
| **本周** | 等实验跑完 → 运行 `analyze_exp1_accuracy.py` | 分类型 F1 表 | ⚠️ 差距是否≥10%？ |
| **第1~2周** | 导出+标注失败案例 → 生成失败分布图 | Exp1 完成 + 图2 | — |
| **第3周** | 运行 `analyze_exp2_oracle.py` → 生成 Oracle 对比图 | Exp2 完成 + 图3 | Oracle 提升是否≥10%？ |
| **第4~5周** | 编写 `qarp_retrieval.py` + 本地调试 | QARP 代码可运行 | — |
| **第6周** | 集成评测脚本 + 小规模试跑验证 | 集成完成 | — |
| **第7周** | 跑 Exp3 完整对比实验 | Table 2 数据 | QARP 是否优于 Baseline？ |
| **第8周** | 跑 Exp4 消融实验 | Table 3 数据 | — |
| **第9~11周** | 论文撰写（按上方顺序） | 论文初稿 | — |

---

*方案版本 v2。核心调整：主贡献从"工程框架"转向"实证分析 + 轻量方法"，全部实验通过 API 完成，适配单人、无 GPU 的研究环境。最高优先事项：立即执行实验 1，用数据验证立论假设。*

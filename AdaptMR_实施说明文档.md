# AdaptMR 实施说明文档：从问题到论文的完整路径

> 创建日期：2026-02-25  
> 定位：基于 A-MEM 代码基座，结合 O-Mem / SimpleMem / Zep / RMM 等论文的技术洞察，设计并实现面向 Agent 记忆的查询感知自适应检索框架  
> 目标：CCF-B 会议（EMNLP / COLING / NAACL）

---

## 一、要解决什么问题

### 1.1 一句话概括

> 现有 Agent 记忆系统对所有查询都采用相同的检索流程，导致在复杂查询（时序推理、知识更新、多跳聚合、拒绝判断）上性能严重不足。AdaptMR 的目标是让检索方式根据查询特征动态调整。

### 1.2 文献证据：统一检索的天花板

从已有论文中，可以收集到以下关键证据支撑 AdaptMR 的研究动机：

**证据 1：O-Mem（Wang et al., 2025）的子任务不均匀表现**

O-Mem 使用了三路并行检索（语义向量 + 话题索引 + IDF 关键词），是目前检索多样性最强的系统之一，但其 LoCoMo 各子任务性能仍然极不均匀：

| 子任务 | F1 | 说明 |
|--------|-----|------|
| Single-hop | 54.89 | 相对较好 |
| Temporal | 57.48 | 尚可 |
| Multi-hop | 42.64 | 明显偏低 |
| Open-domain | **30.58** | 远低于其他子任务 |

即使三路并行，O-Mem 对 Open-domain 和 Multi-hop 仍然力不从心。原因在于三路检索是**固定分工**的（每个组件负责不同类型信息），而不是**根据查询类型动态调整检索逻辑**。

**证据 2：SimpleMem（Liu et al., ICML 2026）的子任务不均匀表现**

SimpleMem 实现了意图感知检索规划（LLM 推断意图 + 动态检索深度 k + 三层并行），在 LoCoMo 上相较 Mem0 的提升同样不均匀：

| 子任务 | 相较 Mem0 的 F1 提升 |
|--------|---------------------|
| Multi-hop | +13.3 |
| Temporal | +9.7 |
| Open-domain | **+3.3**（最低） |

SimpleMem 的自适应仅限"参数级"——调整检索深度 k 和查询改写，对所有查询类型仍执行**同一套三路并行检索流程**，不会因为查询是时序类就加入时间排序，不会因为查询涉及知识更新就做新旧记忆消解。

**证据 3：A-MEM（Xu et al., NeurIPS 2025）的单一检索方式**

A-MEM 采用标准语义 top-K 检索。虽然其 Zettelkasten 结构化存储（笔记间有链接关系）提供了潜在的图遍历能力，但检索时并未利用这一结构——仅做向量相似度排序，链接结构在检索阶段被完全忽略。

**证据 4：Zep（Rasmussen et al., 2025）的多检索方式但无自动路由**

Zep 提供了三种搜索方式（语义搜索 / 全文搜索 / 图遍历），是文献中检索方式最多样的系统。但 Zep 依赖**人工预定义规则**进行策略切换，没有实现基于查询特征的自动策略选择。这从另一个角度说明"多种检索方式"和"自动选择最优方式"是两个不同的问题。

**证据 5：RMM（Tan et al., ACL 2025）优化了排序但没有优化策略**

RMM 用强化学习训练了一个记忆重排序器（回顾性反思模块），优化的是检索结果的**排序**，而非检索**流程本身**。它假设语义检索已经返回了候选集，再对候选集做优化排列。但如果检索流程本身就选错了方向（比如时序查询用语义检索），重排序无法弥补源头的错误。

**证据 6：LongMemEval 基准已区分 5 种查询类型**

LongMemEval（Bai et al., 2024）将查询明确划分为 5 种类型（Single-hop / Multi-hop / Temporal / Knowledge Update / Open-domain），但**没有任何论文**专门研究不同查询类型的最优检索策略。这是一个被明确标注但未被解决的空白。

### 1.3 核心假设

> 不同类型的查询存在**根本不同的最优检索逻辑**，用一种统一流程处理所有查询必然在部分类型上触及性能天花板。

这个假设需要通过 **Oracle 实验**来验证（见第四节）。

---

## 二、怎么解决

### 2.1 方案演进：从"分类 + 路由"到"检索规划"

原始 AdaptMR 方案（v1，见 `AdaptMR_查询感知自适应记忆检索_详细方案.md`）采用的是**查询分类 + 策略路由**架构：用分类器判断查询类型 → 路由到 5 种预定义检索策略之一。

这一方案的问题在于：
- 5 种策略是硬编码的，不能处理混合类型查询（如"我最近一次去的餐厅叫什么？"同时涉及时序和事实提取）
- 本质上仍是"选一个函数执行"，创新深度不够
- 容易被 reviewer 质疑为"工程拼装"

**升级方案：检索规划（Retrieval Planning）**

核心思路来自两个启发：

1. **SimpleMem 的"意图感知检索规划"**：SimpleMem 证明了让 LLM 参与检索决策的可行性——它用 LLM 推断查询意图并调整检索参数。但 SimpleMem 的"规划"仅限于选择检索深度和改写查询，没有涉及**多步检索逻辑**。
2. **RAG 领域的自适应检索**（Self-RAG, FLARE 等）：RAG 领域已经出现了"让模型自主决定何时检索、检索什么"的思路，但这些方法尚未被引入 Agent 记忆系统。

升级方案的关键转变：

| 维度 | v1（分类 + 路由） | v2（检索规划） |
|------|------------------|---------------|
| 核心机制 | 分类器选 1/5 策略 | LLM 生成多步检索计划 |
| 灵活性 | 只能处理 5 种预定义类型 | 可处理任意复杂查询（检索原语自由组合） |
| 深度 | 每种策略是独立函数 | 计划可迭代——后步利用前步结果 |
| 可分析性 | 只能看"选对了没" | 可以分析计划质量、步骤必要性、规划效率 |
| 论文贡献 | 工程拼装嫌疑 | 新问题定义 + 新框架 + 深入分析 |

### 2.2 检索原语设计

检索规划的基础是一组**检索原语（Retrieval Primitives）**——不可再分的基本检索操作。LLM 生成的检索计划由这些原语组合而成。

原语设计的灵感来源：

| 原语 | 功能 | 灵感来源 |
|------|------|---------|
| `SEMANTIC_SEARCH(query, k)` | 语义向量 top-k 检索 | A-MEM 默认方式；几乎所有系统的基础检索 |
| `KEYWORD_MATCH(word)` | 精确关键词匹配，优先选稀有词 | **O-Mem 的 IDF 情节检索**：用逆文档频率选取最稀有词作为线索，捕获语义检索遗漏的特殊实体。O-Mem 证明了在 LLM 时代传统稀疏检索仍有独特价值 |
| `TIME_FILTER(start, end)` | 按时间范围过滤记忆 | **Zep 的双时间线模型**：Zep 区分了事件发生时间和事件记录时间，证明时间维度在记忆检索中不可或缺 |
| `TIME_SORT(memories, order)` | 按时间排序（正序/倒序） | LongMemEval 的 Temporal 类查询需求 |
| `LINK_EXPAND(memory, depth)` | 沿笔记链接扩展关联记忆 | **A-MEM 的 Zettelkasten 链接结构**：A-MEM 的笔记间有明确的链接关系，但其检索时未利用这一结构。LINK_EXPAND 将这一结构优势释放出来 |
| `RERANK(query, memories)` | 用 LLM 对候选记忆重排序 | **RMM 的 RL 重排序器**：RMM 证明了在初始检索后做精细排序的价值。此处简化为 LLM 重排序 |
| `DECOMPOSE(query)` | 将复杂查询分解为子查询 | **AriGraph 的双重检索**：AriGraph 在知识图谱和情景记忆上分别检索再合并，暗示了"分解 → 分别检索 → 聚合"的模式 |
| `CONFIDENCE_CHECK(memories, threshold)` | 判断检索结果是否真正相关 | 原始 AdaptMR 方案的拒绝策略（策略 5）：向量检索永远返回 top-K，不会说"没找到"，需要额外的相关性判断 |
| `CLUSTER_BY_TOPIC(memories)` | 按主题聚类 | **O-Mem 的 LLM 增强近邻图聚类**：O-Mem 用连通分量分析聚合同类属性，证明了在记忆系统中做主题聚类的可行性 |
| `KEEP_LATEST(cluster)` | 同主题取最新 | 原始 AdaptMR 方案的知识更新策略（策略 3）：需要从同主题的多条记忆中取最新版本 |

### 2.3 检索计划的生成与执行

**生成**：给定用户查询，由 LLM 生成一个结构化 JSON 检索计划，包含多个步骤，每步调用一个检索原语。

**执行**：计划执行器逐步执行计划中的原语操作，后步可引用前步结果。

**示例**——查询"我是先开始跑步还是先搬到杭州的？"：

```json
{
  "query_analysis": "用户需要比较两个事件的时间先后",
  "plan": [
    {"step": 1, "op": "SEMANTIC_SEARCH", "args": {"query": "开始跑步", "k": 3}},
    {"step": 2, "op": "SEMANTIC_SEARCH", "args": {"query": "搬到杭州", "k": 3}},
    {"step": 3, "op": "TIME_SORT", "args": {"memories": "step1 + step2", "order": "asc"}}
  ],
  "output_instruction": "返回两个事件的记忆及时间戳，按时间正序排列"
}
```

对比：统一语义检索会搜索"跑步 搬家 杭州"返回一批记忆，但不保证携带时间信息，也不保证 LLM 能正确推断先后关系。

**示例**——查询"我现在住在哪里？"：

```json
{
  "query_analysis": "涉及可能更新的信息（住址），需要取最新",
  "plan": [
    {"step": 1, "op": "SEMANTIC_SEARCH", "args": {"query": "住在哪里 住址 搬家", "k": 10}},
    {"step": 2, "op": "CLUSTER_BY_TOPIC", "args": {"memories": "step1"}},
    {"step": 3, "op": "KEEP_LATEST", "args": {"clusters": "step2"}}
  ],
  "output_instruction": "返回最新的住址信息，标注更新历史"
}
```

对比：统一检索可能同时返回"住在北京"（旧，但措辞直接导致相似度高）和"搬到杭州"（新，但措辞不同导致相似度略低），LLM 可能选错。

**示例**——查询"我跟你说过我的血型吗？"：

```json
{
  "query_analysis": "需要判断是否存在相关记忆",
  "plan": [
    {"step": 1, "op": "SEMANTIC_SEARCH", "args": {"query": "血型", "k": 5}},
    {"step": 2, "op": "CONFIDENCE_CHECK", "args": {"memories": "step1", "threshold": 0.7}}
  ],
  "output_instruction": "如果置信度低，返回 NO_RELEVANT_MEMORY 信号"
}
```

对比：统一检索永远返回 top-K，即使没有任何相关记忆也会返回 K 条"最不离谱的"，导致 LLM 可能臆造答案。

### 2.4 与 A-MEM 代码的集成方式

A-MEM 是我们的代码基座（已复现），AdaptMR 仅替换其检索层：

```
A-MEM 代码结构
│
├── 记忆存储层 ────── 【完全保留】
│   ├── 笔记创建（Note）
│   ├── 笔记链接（Links）
│   └── 记忆库管理
│
├── 记忆写入层 ────── 【完全保留】
│   ├── 信息提取
│   └── 笔记生成
│
├── 记忆检索层 ────── 【替换为 RetrievePlan 模块】
│   └── 原始：semantic_search(query, top_k)
│       ↓ 替换为 ↓
│       新增模块：
│       ├── plan_generator.py    （检索规划器，~100 行）
│       ├── plan_executor.py     （计划执行器，~200 行）
│       ├── primitives.py        （10 个检索原语，~300 行）
│       └── aggregator.py        （结果聚合，~50 行）
│
└── 回答生成层 ────── 【保留，微调 prompt 格式】
    └── 将检索结果 + 计划元信息送入 LLM
```

新增代码量约 650 行，其余完全复用 A-MEM。

**论文写作中的关键定位**：A-MEM 是"实验载体"，不是"基座"。论文标题和摘要中不出现 A-MEM，主角是 RetrievePlan 框架本身。论文应说明 RetrievePlan 可与任何记忆系统集成（通过在 Mem0 上也验证来证明通用性）。

---

## 三、为什么这么做（理论支撑）

### 3.1 来自文献的理论基础

**（1）"检索即规划"的合理性——来自 RAG 领域的类比**

RAG 领域已经认识到单次检索不够：Self-RAG 让模型自主决定是否需要检索，FLARE 在生成过程中动态触发检索。这些工作将检索从"静态操作"升级为"动态决策"。AdaptMR 将这一思路引入 Agent 记忆领域，进一步从"动态决策"升级为"多步规划"。

**（2）多步检索的必要性——来自 O-Mem 和 SimpleMem 的实证**

O-Mem 和 SimpleMem 分别代表了两种不同的"增强检索"思路：
- O-Mem：多路**并行**检索（三个组件同时检索，合并结果）
- SimpleMem：多层**并行**检索（三种索引同时检索，集合去重）

两者的共同局限是：并行的多路检索仍然是**单轮**的——它们增加了"检索的宽度"（同时用多种方式找），但没有增加"检索的深度"（基于第一轮结果做进一步检索）。AdaptMR 的多步检索计划增加的正是这个缺失的维度。

**（3）A-MEM 链接结构的未利用价值**

A-MEM 的 Zettelkasten 结构在笔记间建立了语义链接，理论上可以通过图遍历发现间接相关的记忆。但 A-MEM 的检索代码仅使用向量相似度 top-K，完全没有利用链接结构。AdaptMR 的 `LINK_EXPAND` 原语将这一结构优势释放出来，使得 A-MEM 的存储设计和检索设计真正对齐。

**（4）IDF 检索的独特价值——来自 O-Mem 的验证**

O-Mem 的情节记忆用 IDF 最稀有词而非语义向量作为检索线索，证明了传统信息检索技术在 LLM 记忆系统中仍有不可替代的价值。具体而言：语义检索擅长找"意思相近"的记忆，但对含有特殊命名实体或罕见词的查询（如"鶴矢日料在哪里？"），IDF 关键词匹配的精准度远高于语义检索。AdaptMR 将 IDF 检索作为原语之一纳入框架。

### 3.2 与现有工作的差异定位

| 系统 | 检索方式 | AdaptMR 与之的区别 |
|------|---------|-------------------|
| **A-MEM** | 统一语义 top-K | AdaptMR 将其作为原语之一（`SEMANTIC_SEARCH`），并释放其未利用的链接结构（`LINK_EXPAND`） |
| **O-Mem** | 三路并行（语义 + 话题 + IDF） | O-Mem 的三路是固定分工，不根据查询调整；AdaptMR 根据查询动态组合原语，且纳入了 O-Mem 的 IDF 技术作为原语 |
| **SimpleMem** | 三层并行 + 自适应 k | SimpleMem 自适应仅限检索参数（深度 k 和查询改写），不改变检索逻辑；AdaptMR 实现策略级自适应——不同查询走根本不同的检索流程 |
| **Zep** | 三种搜索方式 | Zep 依赖人工预定义规则切换策略；AdaptMR 由 LLM 自动生成检索计划 |
| **RMM** | RL 优化重排序 | RMM 优化的是检索后的排序；AdaptMR 优化的是检索流程本身 |

**核心差异**：没有任何现有系统将记忆检索视为"需要规划的多步推理过程"。

---

## 四、怎么做（分阶段行动计划）

### 第一阶段：问题验证（第 1-2 周）

**目标**：用第一手数据确认核心假设——"统一检索在不同查询类型上的最优策略确实不同"。

#### Week 1：跑通评测 + 错误分析

**Day 1-2：在 LoCoMo 上跑 A-MEM 完整评测**

输出：每道题的查询、检索结果、生成回答、正确答案、F1 分数。按四个子任务分开：Single-hop / Multi-hop / Temporal / Open-domain。

**Day 3-5：人工分析 60 个错误案例（每类 15 个）**

对每个错误 case 填写分析表：

| # | 查询 | 子任务类型 | 应检索的记忆 | 实际检索到的 | 失败原因 |
|---|------|----------|------------|------------|---------|
| 1 | ... | Temporal | M3, M7 | M3, M12 | 时序信息缺失 |
| 2 | ... | Multi-hop | M2, M5, M9 | M5 | 单次检索遗漏 |

失败原因标签池：
- 语义不匹配
- 需要多条记忆但只检索到部分
- 时序信息缺失 / 错乱
- 检索到噪声淹没正确记忆
- 新旧信息未区分
- 无相关记忆但强行返回
- 其他

**这张表 = 论文 Section 3（Motivation Study）的核心数据。**

#### Week 2：Oracle 实验

**Day 1-2：实现 4 种基础检索策略**

```python
# 策略 1：A-MEM 默认语义检索（已有）
def strategy_semantic(query, memory_store, k=5):
    return memory_store.semantic_search(query, top_k=k)

# 策略 2：语义 + 时间排序
def strategy_temporal(query, memory_store, k=10):
    candidates = memory_store.semantic_search(query, top_k=k)
    return sorted(candidates, key=lambda m: m.timestamp)

# 策略 3：语义 + 链接扩展（利用 A-MEM 的 Zettelkasten 结构）
def strategy_link_expand(query, memory_store, k=3):
    seeds = memory_store.semantic_search(query, top_k=k)
    expanded = []
    for seed in seeds:
        expanded.extend(seed.linked_notes)
    return rerank(query, list(set(seeds + expanded)))

# 策略 4：关键词精确匹配（借鉴 O-Mem 的 IDF 思路）
def strategy_keyword(query, memory_store):
    words = tokenize(query)
    rarest = min(words, key=lambda w: doc_freq(w, memory_store))
    return memory_store.get_by_keyword(rarest)
```

**Day 3-4：对 LoCoMo 每条查询穷举所有策略**

```python
for query, answer in locomo_test:
    for strategy in [strategy_semantic, strategy_temporal,
                     strategy_link_expand, strategy_keyword]:
        score = evaluate(strategy(query, memory), answer)
        record(query, strategy, score)
```

**Day 5：分析 Oracle 结果**

Oracle = 对每条查询选取得分最高的策略。期望得到两张关键表格：

**表 1：各查询类型的最优策略分布**

```
            | 语义检索 | 时序检索 | 链接扩展 | 关键词  |
Single-hop  |  75%    |   5%    |  10%    |  10%   |
Multi-hop   |  20%    |   5%    |  60%    |  15%   |
Temporal    |  10%    |  70%    |  10%    |  10%   |
Open-domain |  40%    |  10%    |  20%    |  30%   |
```

（以上为预估值，实际需要实验数据填充）

**表 2：Oracle vs 最佳单一策略**

```
            | 最佳单一策略 | Oracle | 差距   |
Single-hop  |    52.0     |  58.0  |  6.0  |
Multi-hop   |    35.0     |  55.0  | 20.0  |
Temporal    |    40.0     |  62.0  | 22.0  |
Open-domain |    28.0     |  40.0  | 12.0  |
Average     |    38.8     |  53.8  | 15.0  |
```

**判断节点**：
- Oracle − 最佳单一策略 **≥ 10%** → 前提成立，继续推进
- 差距 **< 5%** → 前提不成立，需要换方向或重新审视策略设计

这个实验的意义参考 Generative Agents（Park et al., 2023）在消融实验中验证三重评分（相关性 / 近期性 / 重要性）各自贡献的思路——先证明"不同维度确实有独立贡献"，再构建统一框架。

---

### 第二阶段：方案实现（第 3-5 周）

基于第一阶段数据发现来设计方案，做到数据驱动而非拍脑袋。

#### Week 3：实现检索规划器

`plan_generator.py`：核心是一个 prompt，指导 LLM 根据查询生成结构化检索计划。

规划器的 prompt 需要包含：
- 可用原语列表及每个原语的功能说明
- 规划原则（简单查询不过度规划、时序查询需要时间排序、更新查询需要新旧消解等）
- Few-shot 示例（覆盖各查询类型的典型计划）

关键设计决策：
- 计划格式使用 JSON（结构化、易解析）
- 每步有 `step`（序号）、`op`（原语名称）、`args`（参数，可引用前步结果）
- 有 `output_instruction` 指导结果呈现

#### Week 4：实现检索原语 + 计划执行器

`primitives.py`：10 个检索原语的具体实现。其中：
- `SEMANTIC_SEARCH`：直接调用 A-MEM 已有的向量检索接口
- `KEYWORD_MATCH`：参考 O-Mem 的 IDF 实现，统计词频并选取最稀有词
- `LINK_EXPAND`：利用 A-MEM 的笔记链接关系做图遍历
- `CLUSTER_BY_TOPIC`：参考 O-Mem 的 LLM 增强近邻图聚类思路
- 其余原语相对简单（排序、过滤、阈值判断等）

`plan_executor.py`：逐步执行计划中的原语操作。关键逻辑是"参数解引用"——当某步的参数引用了前步结果时，需要将引用替换为实际数据。

#### Week 5：集成到 A-MEM + 初步调试

替换 A-MEM 的检索函数：

```python
# 原始 A-MEM
results = memory_store.semantic_search(query, top_k=5)

# 替换为 AdaptMR
plan = generate_plan(query, llm)
results = execute_plan(plan, memory_store)
```

调试重点：
- 检查 LLM 生成的计划是否格式正确、原语调用是否合法
- 检查前后步引用是否正确解析
- 对简单查询（事实提取），计划应退化为 1 步 `SEMANTIC_SEARCH`，不应引入额外开销

---

### 第三阶段：实验评估（第 6-9 周）

#### Week 6-7：主对比实验

**实验设置：**

| 数据集 | 查询数量 | 查询类型标注 |
|--------|---------|------------|
| **LoCoMo** | ~300 | 4 种（Single-hop / Multi-hop / Temporal / Open） |
| **LongMemEval** | 500 | 7 种细分类型 |

**对比方法：**

| 方法 | 类型 | 说明 |
|------|------|------|
| A-MEM | 基线 | 统一语义检索（已复现） |
| A-MEM + RetrievePlan | **我们的方法** | 规划式检索 |
| Mem0（开源版） | 基线 | 工业级记忆系统 |
| Mem0 + RetrievePlan | **我们的方法** | 证明框架通用性 |
| RAG baseline | 基线 | 纯向量检索（最简） |
| Oracle | 上界 | 每条选最优策略 |

**为什么不对比 O-Mem / SimpleMem？**

1. O-Mem 和 SimpleMem 的核心贡献在存储设计（三组件架构 / 语义压缩），与 AdaptMR 的检索层优化**正交**
2. 如果 reviewer 问，回应："RetrievePlan 是检索层框架，与存储设计正交，可与任何存储架构结合。我们选择 A-MEM 和 Mem0 是因为它们有公开可复现的代码"
3. 在 A-MEM 和 Mem0 两个不同系统上都有提升，已足够证明通用性

**评估指标：**

| 指标 | 意义 |
|------|------|
| 各类型 F1 / Accuracy | 核心指标——分类型展示差异化提升 |
| 整体 F1 / Accuracy | 总体效果 |
| Recall@K | 检索质量 |
| Precision@K | 检索精度 |
| Token 使用量 | 效率开销 |
| 端到端延迟 | 实用性 |

**核心看点**：RetrievePlan 在 Multi-hop、Temporal、Open-domain 上显著优于基线，在 Single-hop 上持平或略有提升（因为简单查询的计划退化为 1 步语义检索）。

#### Week 8：消融实验

| 消融变体 | 移除内容 | 验证什么 |
|---------|---------|---------|
| w/o Planning | 直接用 LLM 分类 + 路由（退化为 v1 方案） | 规划 vs 路由的差距 |
| w/o Multi-step | 只执行计划的第一步 | 多步检索的价值 |
| w/o Link Expand | 去掉链接扩展原语 | A-MEM 链接结构的价值 |
| w/o Keyword Match | 去掉关键词原语 | IDF 思路（来自 O-Mem）的价值 |
| w/o Confidence Check | 去掉置信度检查 | 拒绝判断能力的价值 |
| Fixed Plan | 为每种类型固定一个计划模板 | 动态规划 vs 静态模板 |

消融实验的设计参考了 O-Mem 的 Token 控制消融方法论——在相同 token 预算下对比不同配置，排除"检索量增加导致性能提升"的混淆因素。

#### Week 9：深入分析

- **检索计划质量分析**：人工评估 50 个计划的合理性（合理率、常见错误模式）
- **规划长度 vs 性能**：1 步计划 vs 2 步 vs 3 步的性能变化
- **错误分析**：规划失败的典型模式——是规划器生成了错误计划？还是执行器执行错误？还是原语不够丰富？
- **效率分析**：额外 LLM 调用（规划 + 执行中的 RERANK/CONFIDENCE_CHECK）带来的延迟和 token 开销

---

### 第四阶段：论文撰写（第 10-13 周）

#### Week 10-11：初稿

重点参考以下论文的写作结构：
- **PREMem**（Kim et al., EMNLP 2025）：同样是"对现有记忆系统的检索/写入层做改进"的定位，写作结构清晰
- **RMM**（Tan et al., ACL 2025）：同样聚焦于检索优化，且在 Motivation 部分做了详细的现有系统性能分析

#### Week 12：图表制作 + 案例展示

关键图表：
1. 各系统在不同查询类型上的性能雷达图
2. Oracle 分析图（各类型最优策略分布）
3. 典型案例的检索计划 vs 统一检索的对比
4. 消融实验柱状图
5. 规划长度 vs 性能散点图

#### Week 13：修改打磨

---

## 五、做成什么样（预期产出）

### 5.1 论文结构

```
Title: RetrievePlan: Planning-Based Adaptive Memory Retrieval for LLM Agents

Abstract:
  现有系统统一 top-K 检索 → 复杂查询失败
  → 提出"检索即规划"新范式 → 原语组合 + 多步执行
  → 在 A-MEM 和 Mem0 上即插即用验证，一致提升

Section 1 - Introduction:
  1.1 Agent 记忆的重要性（引用 Memory Survey [Hu et al., 2026]、
      Generative Agents [Park et al., 2023]）
  1.2 统一检索的局限（motivating examples）
  1.3 核心洞察："记忆检索不是单步操作，而是多步规划过程"
  1.4 贡献：
      (1) 提出"检索即规划"新范式
      (2) 设计 10 种检索原语 + 规划器 + 执行器
      (3) 在 A-MEM 和 Mem0 上即插即用验证
      (4) 系统分析不同查询类型的最优检索策略

Section 2 - Related Work:
  2.1 Agent 记忆系统
      A-MEM [Xu et al., 2025]、O-Mem [Wang et al., 2025]、
      Mem0 [Chhikara et al., 2025]、SimpleMem [Liu et al., 2026]、
      MemGPT [Packer et al., 2023]
      → 聚焦指出：它们在检索层大同小异
  2.2 检索增强生成（Self-RAG, FLARE）
      → 指出：RAG 的自适应思想尚未被引入 Agent 记忆
  2.3 LLM 用于规划
      → 指出：规划思想可以用于检索流程设计

Section 3 - Motivation Study（第一阶段数据）:
  3.1 实验设置
  3.2 各系统在不同查询类型上的性能分析
  3.3 Oracle 上界分析
  3.4 错误案例分析（60 个 case 的分类统计）
  3.5 核心发现总结

Section 4 - Method: RetrievePlan:
  4.1 问题形式化
  4.2 检索原语设计（10 种原语 + 设计动机）
  4.3 检索规划器（Plan Generator）
  4.4 计划执行器（Plan Executor）
  4.5 与现有系统的集成（即插即用性）

Section 5 - Experiments:
  5.1 实验设置（数据集、基线、指标）
  5.2 主实验结果
  5.3 消融实验
  5.4 检索计划质量分析
  5.5 效率分析
  5.6 案例展示

Section 6 - Conclusion
```

### 5.2 预期实验结果

```
| Method                | Single-hop | Multi-hop | Temporal | Open   | Avg  |
|-----------------------|-----------|-----------|----------|--------|------|
| A-MEM                 | xx.x      | xx.x      | xx.x     | xx.x   | xx.x |
| A-MEM + RetrievePlan  | xx.x      | xx.x ↑    | xx.x ↑↑  | xx.x ↑ | xx.x |
| Mem0                  | xx.x      | xx.x      | xx.x     | xx.x   | xx.x |
| Mem0 + RetrievePlan   | xx.x      | xx.x ↑    | xx.x ↑↑  | xx.x ↑ | xx.x |
| RAG baseline          | xx.x      | xx.x      | xx.x     | xx.x   | xx.x |
| Oracle                | xx.x      | xx.x      | xx.x     | xx.x   | xx.x |
```

核心故事线：
- RetrievePlan 在 Multi-hop 和 Temporal 上**大幅提升**（因为这些类型最需要多步检索）
- 在 Single-hop 上**基本持平**（简单查询的计划退化为 1 步，不引入额外开销）
- 在 A-MEM 和 Mem0 两个系统上**一致有效** → 通用框架

---

## 六、资源需求与风险

### 6.1 资源需求

| 资源 | 需求 | 说明 |
|------|------|------|
| GPU | **无需 GPU** | 全部基于 API 调用 |
| API 开销 | ~$200-400 | GPT-4o-mini 用于规划 + 生成 + 评估 |
| 代码基座 | A-MEM（已复现） | 保留存储/写入层，替换检索层 |
| 新增代码量 | ~650 行 | 规划器 + 执行器 + 原语 + 聚合器 |
| 时间 | **约 3.5 个月（13 周）** | 见第四节时间线 |

### 6.2 风险与应对

| 风险 | 级别 | 应对 |
|------|------|------|
| Oracle 差距太小 | 🔴 高 | Week 2 即可判断；若差距 < 5% 则重审策略设计或换方向 |
| LLM 生成的计划质量不稳定 | 🟡 中 | 设计固定模板作为 fallback；用 few-shot 示例优化 prompt |
| 多步检索增加延迟 | 🟡 中 | 简单查询退化为 1 步（不过度规划）；统计平均步数，展示大部分查询仅 1-2 步 |
| Reviewer 质疑"工程拼装" | 🟢 低 | Motivation Study 提供深入分析；消融实验证明每个原语的独立贡献 |

### 6.3 目标会议

| 优先级 | 会议 | 截稿时间（预估） | 适配理由 |
|--------|------|-----------------|---------|
| 🥇 首选 | EMNLP 2026 | 2026 年 6 月 | 检索增强是 NLP 核心主题；记忆系统热度高 |
| 🥈 备选 | COLING 2026 | 2026 年 7 月 | 接受率较高；系统分析类论文友好 |
| 🥉 备选 | NAACL 2027 | 2026 年 10 月 | 时间更充裕 |

---

## 七、关键参考文献

以下是 AdaptMR 方案设计中直接引用或借鉴的论文及其具体贡献：

| 论文 | 在 AdaptMR 中的角色 | 借鉴的具体技术/观点 |
|------|-------------------|-------------------|
| **A-MEM** (Xu et al., NeurIPS 2025) | 代码基座 + 基线 | Zettelkasten 链接结构 → `LINK_EXPAND` 原语；统一语义检索的局限 → 研究动机 |
| **O-Mem** (Wang et al., 2025) | 策略池技术来源 + 子任务分析证据 | IDF 最稀有词检索 → `KEYWORD_MATCH` 原语；近邻图聚类 → `CLUSTER_BY_TOPIC` 原语；子任务不均匀表现 → Motivation 证据 |
| **SimpleMem** (Liu et al., ICML 2026) | 差异化定位参照 + 子任务分析证据 | 意图感知检索规划 → 证明 LLM 参与检索决策可行，但其"参数级"自适应不足 → AdaptMR 做"策略级"自适应；OpenDomain 仅 +3.3 F1 → Motivation 证据 |
| **Zep** (Rasmussen et al., 2025) | 多检索方式的先例 | 语义/全文/图三种搜索 → 证明多种检索方式有价值但需自动路由；双时间线模型 → `TIME_FILTER` 原语 |
| **RMM** (Tan et al., ACL 2025) | 排序优化的先例 | RL 重排序 → `RERANK` 原语的灵感来源；前瞻性/回顾性反思 → 规划器和执行器分离的架构参考 |
| **AriGraph** (Anokhin et al., 2025) | 双重检索的先例 | 语义+情景记忆分别检索 → `DECOMPOSE` 原语的灵感 |
| **LongMemEval** (Bai et al., 2024) | 评估基准 + 查询类型分类 | 5 种查询类型定义 → 检索原语设计的需求来源 |
| **LoCoMo** (Maharana et al., 2024) | 评估基准 | 长对话记忆 QA → 主评估数据集 |
| **Generative Agents** (Park et al., 2023) | 基础范式 | 三重评分检索 → 多维度检索评估的早期先例 |
| **MemGPT** (Packer et al., 2023) | Related Work | OS 式记忆管理 → 检索层仍为统一向量检索的例证 |
| **Mem0** (Chhikara et al., 2025) | 通用性验证基线 | 工业级系统 + RetrievePlan → 证明框架与存储设计正交 |
| **PREMem** (Kim et al., EMNLP 2025) | 论文写作参考 | "对记忆系统某一层做改进"的定位和写作结构参考 |
| **Self-RAG / FLARE** | Related Work | RAG 领域自适应检索 → 证明自适应思想可行但尚未进入 Agent 记忆 |
| **Memory Survey** (Hu et al., 2026) | 综述引用 | 三维分类框架 → Introduction 中引用 |

---

## 八、立即行动清单

```
□ 本周（Week 1）：
  □ 跑 A-MEM 在 LoCoMo 上的完整评测
  □ 收集每道题的检索结果和得分
  □ 人工分析 60 个错误案例，填写错误分析表

□ 下周（Week 2）：
  □ 实现 4 种基础检索策略
  □ 跑 Oracle 实验
  □ 分析 Oracle 结果，判断前提是否成立

□ 判断节点：
  □ Oracle 差距 ≥ 10% → 继续 Week 3-13
  □ Oracle 差距 < 5% → 停止，回顾数据，考虑换方向
```

---

*文档完毕。核心原则：每个设计决策都有文献支撑，每个组件都有来源论文的技术启发，从数据出发验证前提，不拍脑袋设计方案。*

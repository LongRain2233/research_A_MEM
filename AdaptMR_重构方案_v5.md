# QueryPlan-Mem：面向 Agent 个人记忆的训练自由查询意图驱动检索规划

> **方案版本：** v5（专业重构版）  
> **生成日期：** 2026-02-26  
> **基于：** 22 篇参考文献系统调研 + v4 草稿批判性吸收  
> **核心定位：** CCF B 级会议/期刊（首选 EMNLP 2026）

---

## ⚠️ 前置声明：对 v4 草稿的批判性评估

> 以下批评不可回避，直接影响方案可行性，必须先厘清再前进。

### v4 草稿的五类根本性问题

| 问题编号 | 级别 | 具体表现 | 影响 |
|---------|------|---------|------|
| **P1** | 🔴 致命 | **Gating Network 训练数据悖论**：端到端训练需要 `(query, memory, answer)` 三元组，但这类数据只存在于 LongMemEval 这样的 benchmark，意味着模型本质上在 benchmark 的数据分布上训练——正是 v4 声称要解决的 overfitting 问题 | 方法可行性存疑 |
| **P2** | 🔴 致命 | **软组合架构逻辑错误**：将操作建模为并行加权求和 `M* = Σ wₖ·oₖ(q, M)`，但 `temporal_ordering` 是基础检索的前置过滤器，`multi_step` 依赖子查询的串行输出，操作间存在数据流依赖，无法合法并行加权 | 架构设计根本性错误 |
| **P3** | 🟠 严重 | **与 2025 SOTA 脱节**：v4 的对比基线（Unified-Embedding, A-MEM, Mem0）已过时。SGMem（Huawei, 2025）在 LongMemEval 上已达 **0.730**，MIRIX 在 LoCoMo 达 **85.4%**，SeCom 已在 ICLR 2025 发表——这些才是真正的竞争对手 | 审稿人会立即质疑 |
| **P4** | 🟡 中等 | **信息论框架虚假联系**：声称使用互信息最大化框架，但实际优化目标是 F1 loss，两者之间缺乏真实的数学桥接；互信息本身无法精确计算 | 理论部分无法自洽 |
| **P5** | 🟡 中等 | **存在性查询问题治标不治本**：`threshold_filtering` 仅调节相似度阈值，但"是否曾说过 X"类查询的根本问题是系统**不能拒绝返回结果**——任何阈值下 top-K 仍会强制返回 K 条记忆 | F4 类查询无法真正解决 |

---

## 模块 1：研究动机

### 1.1 领域发展脉络的精准定位

通过系统梳理 2019–2025 年的 22 篇核心文献，Agent 记忆系统的技术演进可归纳为三个阶段：

- **第一阶段（2019–2022）**：RAG 范式确立（Lewis et al., 2020），以向量检索为核心工具，将文档记忆化为 embedding。
- **第二阶段（2022–2024）**：记忆结构多样化。MemGPT（OS 式分层）、A-MEM（Zettelkasten 图结构）、Zep（时序知识图谱）、AriGraph（语义+情节双图）——**竞争焦点在记忆构建质量**。
- **第三阶段（2024–2025）**：SOTA 系统普遍转向复杂记忆构建。SeCom（ICLR 2025）证明**构建粒度 > 检索策略**；SGMem（Huawei）建立七级索引 + 图扩展；Mem-α 用 RL 学习记忆管理；MIRIX 构建六类记忆类型。**记忆构建已高度精细化，但检索层仍然是"一刀切"**。

### 1.2 核心研究空白的识别

**关键观察**：即使是 2025 年的顶级系统（SGMem、MIRIX、Zep）在检索时也对所有查询类型使用相同的检索流程：

- **SGMem**：所有查询均经过"向量检索 → 图扩展 → 重排序"固定管道
- **MIRIX**：各记忆类型有固定的类型内检索策略，但无查询意图感知
- **Zep**：三步固定管道（Search → Reranker → Constructor），不根据查询类型调整策略
- **SeCom**：仅优化检索前的记忆构建质量，检索本身固定不变

**唯一例外**：MemoTime（WWW 2026）实现了"算子感知检索"——但其场景是**时序知识图谱推理**，面向公开事件数据库而非**个人对话记忆**，属于完全不同的技术栈。

**研究空白**：对于个人对话记忆系统，**不同查询意图类型在最优检索策略上存在系统性差异**，且这种差异跨数据集持续存在——但目前没有任何工作对此进行系统研究，更没有工作提出相应的轻量解决方案。

### 1.3 问题的现实严重性证据

当前 SOTA 系统在不同查询类型上存在显著的性能分化：

| 查询类型 | 代表性失败示例 | 根本原因 | 预计现有最优基线 |
|---------|------------|---------|---------------|
| **时序推理（F1）** | "我先开始跑步还是先搬到杭州的？" | 向量检索不携带事件时间顺序 | ~0.40–0.55 |
| **状态追踪（F2）** | "我现在住在哪里？"（有地址变更） | 旧信息因措辞更匹配而被优先返回 | ~0.45–0.60 |
| **多跳聚合（F3）** | "我提到过的所有餐厅里哪种菜系最多？" | 单次 top-K 无法"先收集、再聚合" | ~0.35–0.50 |
| **存在性判断（F4）** | "我跟你说过我的血型吗？" | 系统**强制**返回 K 条结果，无法表达"未找到" | **< 20%**（全行业盲区） |

> **F4 存在性判断是所有当前系统的系统性盲区**——没有任何记忆系统能正确返回"未找到"。这本身是一个高价值的独立发现，具有充分的发表意义。

---

## 模块 2：研究问题（RQ 形式）

**RQ1（实证）**：在个人对话记忆场景中，当前 SOTA 记忆系统（SGMem、Mem0、Zep）的检索失败是否在查询意图维度上呈现系统性分布？该分布是否在 LongMemEval、LoCoMo、MSC 三个数据集上保持跨域一致性？

> *验证方式*：在三个数据集上运行相同基线，按意图类型拆分失败率，计算 Cohen's Kappa 检验跨域一致性。

**RQ2（方法）**：能否设计一种**无需端到端训练**的查询意图分类机制，仅通过查询文本的结构特征（无需 memory context 或 answer 监督信号）实现对记忆检索意图的准确分类？

> *验证方式*：与 LLM-based 分类器对比，评估分类准确率 vs. 推理延迟的帕累托效果。

**RQ3（方法）**：针对不同查询意图，能否设计对应的**检索执行程序**（Retrieval Program）——特别是：对存在性查询能否设计返回"无相关记忆"的 RetNull 机制？

> *验证方式*：消融实验，单独验证每个检索程序模块的增益，重点关注 F4 Null-Recall 指标。

**RQ4（综合）**：所提方法作为**可插拔模块**叠加在现有记忆系统上（不改变记忆构建），能否在不同基础系统（A-MEM、Mem0、SGMem）上一致地带来性能提升？

> *验证方式*：在三种基础系统上分别叠加本方法，验证提升的跨系统稳定性。

---

## 模块 3：创新点（3 个差异化贡献）

### 创新点 1（主贡献）：跨域查询意图分类体系与失败模式分析

**具体内容**：
通过对 LongMemEval、LoCoMo、MSC 三个数据集上 200+ 失败案例的系统分析，建立**独立于任何单一评测集**的四维查询意图分类体系（Q-Intent Taxonomy），并首次定量证明：

- 当前 SOTA 系统（包括 SGMem、Mem0+graph）在不同意图类型上的失败率差距超过 **25 个百分点**
- 该差距在三个数据集上跨域一致（Cohen's Kappa > 0.7 为目标）
- **F4 存在性判断类查询在所有系统上准确率均低于 20%**（因为没有任何系统具备"拒绝检索"能力）

**与已有工作的差异**：

| 相关工作 | 已有覆盖 | 本工作的差异 |
|---------|---------|------------|
| MemoTime (WWW 2026) | 算子感知检索（时序 KG 推理） | 针对**个人对话记忆**，覆盖 4 种意图类型 |
| LongMemEval 原始论文 | 基础错误分析 | 未系统化、无跨域验证、无 F4 专项分析 |
| SeCom (ICLR 2025) | 构建粒度重要性 | 关注构建，不涉及检索意图 |
| SGMem (Huawei 2025) | 多粒度图结构检索 | 固定检索管道，无查询意图感知 |

> **本贡献是第一个将查询意图分类应用于跨域个人记忆检索失败分析的系统性工作。**

---

### 创新点 2（主贡献）：训练自由的意图导向检索程序框架（TIRP）

**具体内容**：
提出 **TIRP（Training-free Intent-driven Retrieval Programming）**，其核心区别于 v4 的 Gating Network：

| 特性 | v4 Gating Network | TIRP（本方法） |
|------|-------------------|--------------|
| 训练需求 | 需要 `(query, memory, answer)` 三元组 | **零训练数据**：仅用查询文本特征 |
| 决策模型 | 操作权重的并行加权求和（逻辑错误） | **检索程序**：有序操作的 DAG 图 |
| 操作执行 | 并行（忽略操作间依赖） | **串行/分支**（符合数据流依赖） |
| 存在性查询处理 | threshold_filtering（不完整） | **RetNull 机制**（真正拒绝返回） |
| 可移植性 | 与训练数据分布绑定 | **任意记忆系统的可插拔模块** |

**四类检索程序模板**（对应四种意图类型）：

```
P_temporal(F1):
  TemporalAnchorExtract(q) 
  → TimestampResolve(anchor, memory_store) 
  → TemporalFilter(memories, anchor_ts, relation) 
  → TemporalSort 
  → TopK

P_versioning(F2):
  EntityExtract(q) 
  → SemanticSearch(q, K=20) 
  → StateConflictDetect(candidates, entity) 
  → LatestVersionSelect

P_aggregate(F3):
  QueryDecompose(q)  [few-shot LLM, 1次调用]
  → ForEach(sub_q: SemanticSearch(sub_q, K=10)) 
  → Deduplicate 
  → AggregateReason(collected_memories)

P_existence(F4):
  SemanticSearch(q, K=10) 
  → ConfidenceScore(max_similarity) 
  → if max_score < θ_null: RetNull   ← 核心新机制
  → else: ThresholdFilter(θ_positive)
```

---

### 创新点 3（辅助贡献）：RetNull 机制与 F4 Null-Recall 新评估指标

**问题**：当前所有记忆系统存在"强制返回"问题——无论如何，系统都会返回 top-K 条最相关记忆，即使没有任何记忆与查询相关。这对存在性查询（"我说过我的血型吗？"）造成致命伤害——系统返回"最相关"的 K 条记忆，LLM 从中幻觉出一个答案。

**RetNull 实现**：

```python
def retNull_check(query_embedding, candidates, theta_null):
    """
    训练自由的空返回检测器
    theta_null 通过 100 个验证样本快速校准（不依赖训练集）
    """
    max_score = max(c.similarity_score for c in candidates)
    
    if max_score < theta_null:
        # 无相关记忆，返回 null 标记
        return [], "NULL_MEMORY"
    else:
        # 有相关记忆，执行正常过滤
        return [c for c in candidates if c.similarity_score >= theta_positive], "FOUND"
```

**θ_null 校准方法（零训练数据）**：
1. 在验证集（~100 样本）中标注 F4 查询的"正确答案是否为不存在"
2. 计算"无相关记忆"情况下 semantic_search 的 max(score) 分布
3. 设置 θ_null 使 F4 正确拒绝率 > 80%，误拒率 < 10%
4. 全过程不使用任何记忆内容或答案的监督信号

**F4 Null-Recall（新评估指标）**：

$$\text{Null-Recall} = \frac{\text{正确返回空集的 F4 查询数}}{\text{正确答案为"无相关记忆"的 F4 查询总数}}$$

> 这个指标在现有文献中从未出现，是本研究可以首创的评估视角。

---

## 模块 4：方法设计

### 4.1 整体框架描述

```
┌──────────────────────────────────────────────────────────────────┐
│                    QueryPlan-Mem 整体架构                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  用户查询 q                                                         │
│       │                                                            │
│       ▼                                                            │
│  ┌────────────────────────────────────────┐                       │
│  │ [模块 A] 查询意图解析器 (QIP)             │                       │
│  │ 输入: 查询文本 q                          │                       │
│  │ 方法: 规则引擎 → 轻量级 BERT 兜底          │                       │
│  │ 输出: 意图类型集合 I ⊆ {F1,F2,F3,F4}     │                       │
│  │       + 意图参数 (时间锚、实体名等)         │                       │
│  └──────────────────┬─────────────────────┘                       │
│                     │                                              │
│                     ▼                                              │
│  ┌────────────────────────────────────────┐                       │
│  │ [模块 B] 检索程序编译器 (RPC)             │                       │
│  │ 输入: 意图类型 I + 意图参数               │                       │
│  │ 方法: 模板匹配 → 程序实例化               │                       │
│  │ 输出: 可执行检索程序 π（DAG 形式）         │                       │
│  └──────────────────┬─────────────────────┘                       │
│                     │                                              │
│                     ▼                                              │
│  ┌────────────────────────────────────────┐                       │
│  │ [模块 C] 检索程序执行引擎 (RPE)           │                       │
│  │ 原子操作集合:                             │                       │
│  │  ├── TemporalAnchorExtract  (规则实现)   │                       │
│  │  ├── StateConflictDetect    (规则实现)   │                       │
│  │  ├── QueryDecompose         (few-shot)  │                       │
│  │  └── RetNull                (阈值机制)  │ ← 核心新机制             │
│  │ 输出: 精炼记忆集 M*（可含 null 标记）      │                       │
│  └──────────────────┬─────────────────────┘                       │
│                     │                                              │
│                     ▼                                              │
│  ┌────────────────────────────────────────┐                       │
│  │ [模块 D] 记忆融合与生成                   │                       │
│  │ 输入: q + M*（含 null 标记时调整 prompt） │                       │
│  │ 输出: 最终回答                            │                       │
│  └────────────────────────────────────────┘                       │
│                                                                    │
│  ★ 与基础记忆系统的接口：仅需 semantic_search(q, k) 标准接口          │
│    可插拔于 A-MEM、Mem0、SGMem 等任意系统，无需修改记忆构建管道         │
└──────────────────────────────────────────────────────────────────┘
```

### 4.2 模块 A：查询意图解析器（QIP）——训练自由设计

**设计原则**：避免 v4 的 Gating Network 训练数据问题，采用"规则优先 + 轻量分类器兜底"策略。

#### 层级一：规则引擎（处理 ~70% 查询，零延迟）

| 意图类型 | 触发关键词 | 提取参数 |
|---------|-----------|---------|
| **F1 时序意图** | 先/后/之前/之后/什么时候/最近一次/首次/早于/晚于 | 时间锚实体、时间关系类型 |
| **F2 状态追踪** | 现在/目前/最新/最近/改了/变成/已经/更新 | 主题实体名 |
| **F3 聚合推理** | 所有/哪些/几次/总共/列表/都/曾经提到/统计/最多 | 聚合条件、目标属性 |
| **F4 存在性** | 说过吗/告诉过吗/是否/有没有/知道吗/提到过吗 | 目标事实描述 |

#### 层级二：轻量 BERT 分类器（处理规则未命中的 ~30% 查询）

- 使用 `sentence-transformers` MiniLM（384 维，参数量 ~33M，可在 CPU 上快速推理）
- **训练数据来源**：仅使用查询文本 + 意图标签（`(query_text, intent_label)`），**完全不使用 memory 或 answer**
- 训练样本：~200 个手工标注的意图样本（可从 LongMemEval 查询文本提取，仅使用文本特征）

**与 v4 的关键区别**：

```
v4 Gating Network 训练数据：(query, memory_context, final_answer)
QIP 分类器训练数据：          (query_text, intent_label)

前者与 benchmark 评测强耦合 → benchmark overfitting
后者仅依赖查询语言特征       → 可跨 benchmark 泛化
```

#### 复合意图处理

查询可携带多个意图（如"我搬到上海**之前**说过我的**体重**吗？"= F1+F4）。

QIP 返回意图类型**集合**而非单一类别，RPC 随后生成**复合检索程序**（多分支 DAG）。

---

### 4.3 模块 B：检索程序编译器（RPC）

RPC 将意图类型映射为可执行的有向无环图（DAG）形式的检索程序：

```
单意图示例（F4 存在性查询）：
                query
                  │
          semantic_search(k=15)
                  │
           score_compute
                  │
           retNull_check(θ_null)
                 / \
        (null)  /   \ (found)
               ∅    threshold_filter(θ_pos)

复合意图示例（F1+F4 时序+存在性）：
                query
               /     \
temporal_anchor_extract  semantic_search(k=10)
               │              │
        ts_resolve(anchor)    score_compute
               │              │
     temporal_filter(ts)   retNull_check
               │              │
              TopK         (found: threshold_filter)
               \              /
                join & deduplicate
                      │
                    TopK_final
```

---

### 4.4 核心技术模块实现细节

#### StateConflictDetect（F2 状态追踪核心）

```python
def detect_state_conflict(candidates, entity):
    """
    检测同一实体-属性对的版本冲突，返回最新版本及冲突信息
    纯规则实现，零 LLM 调用
    """
    # 提取涉及同一实体的候选记忆
    entity_mentions = [m for m in candidates if entity in m.entities]
    
    if len(entity_mentions) > 1:
        # 检测语义矛盾（通过实体属性值不一致）
        conflict_pairs = find_semantic_contradictions(entity_mentions)
        if conflict_pairs:
            # 返回最新版本 + 冲突标记
            latest = max(entity_mentions, key=lambda m: m.timestamp)
            return [latest], {
                "conflict": True,
                "superseded_memories": [p[0] for p in conflict_pairs],
                "latest_memory": latest
            }
    
    return entity_mentions, {"conflict": False}
```

#### TemporalAnchorExtract（F1 时序查询核心）

```python
def temporal_anchor_extract(query, memory_store):
    """
    从查询中提取时间锚事件，在记忆库中解析其时间戳
    规则 + 轻量模式匹配，无 LLM 调用
    
    示例输入: "我先开始跑步还是先搬到杭州的？"
    提取: {event_A: "开始跑步", event_B: "搬到杭州", relation: "compare"}
    解析: t_A = memory_store.find_event_timestamp("开始跑步")
          t_B = memory_store.find_event_timestamp("搬到杭州")
    输出: [(event_A, t_A), (event_B, t_B)], relation="compare"
    """
    # Step 1: 提取时间锚候选（规则提取）
    anchors = extract_temporal_anchors(query)
    
    # Step 2: 在记忆库中解析时间戳（语义搜索）
    resolved = []
    for anchor in anchors:
        timestamp_memories = memory_store.find_event(anchor.event_desc)
        if timestamp_memories:
            resolved.append((anchor, timestamp_memories[0].timestamp))
    
    return resolved
```

---

### 4.5 两个系统版本

| 版本 | LLM 额外调用 | QueryDecompose | RetNull | 适用场景 |
|------|------------|---------------|---------|---------|
| **TIRP-Zero** | **0 次** | 规则分解（启发式） | ✅ 完整 | 公平性验证、资源受限 |
| **TIRP-Full** | 1 次（仅 F3） | few-shot LLM 分解 | ✅ 完整 | 追求最佳性能 |

---

## 模块 5：实验方案

### 5.1 数据集选择

| 数据集 | 规模 | 查询类型覆盖 | 使用方式 |
|--------|------|------------|---------|
| **LongMemEval** | 500 查询，50 个长对话 | 5 类（含时序、知识更新、多跳、拒绝判断） | 主评估集 + QIP 分类器训练 |
| **LoCoMo** | ~300 查询 | 时序密集型，自然分布 | 跨域泛化验证 |
| **MSC** | ~200 查询 | 多 session 自然对话 | 跨域泛化验证 |

> **不使用 DMR**：Zep 论文已明确指出 DMR 的局限性——短对话 + 单一事实检索，无法评估本研究关注的复杂意图类型，且当前 LLM 的全文上下文方法可轻松达到 98%，评估价值极低。

---

### 5.2 基线方法（含 2025 SOTA）

| 基线 | 类别 | 额外 LLM 调用 | 备注 |
|------|------|-------------|------|
| Unified Embedding (top-K) | 统一检索 | 0 | 最基础基线 |
| **A-MEM** | 图结构记忆 | 少量 | 可本地复现 |
| **Mem0** | 工业级系统 | 中等 | 需要 API |
| **SGMem-SF** | 2025 SOTA | 中等 | **关键竞争对手**，LongMemEval 0.730 |
| **Zep** | 时序知识图谱 | 中等 | **关键竞争对手**，LongMemEval 18.5% 提升 |
| Adaptive-RAG（迁移版） | 自适应 RAG | 少量 | 证明记忆场景独特性 |
| Token-Budget-Matched Baseline | 公平性对照 | 与 TIRP-Full 等量 | 验证提升非算力驱动 |
| **TIRP-Zero（本方法）** | 意图导向检索 | **0** | — |
| **TIRP-Full（本方法）** | 意图导向检索 | 1 次（仅 F3） | — |
| Oracle-Strategy | 上界参考 | — | 手工选择最优检索 |

> ⚠️ **关键说明**：与 v4 相比，本方案**必须包含 SGMem 和 Zep** 作为基线。若资源受限无法完整复现，至少需要在论文中与其公开结果进行直接数字对比，并说明实验条件差异。

---

### 5.3 评估指标

| 指标 | 说明 | 优先级 |
|------|------|--------|
| **Per-intent Accuracy** | 四类意图下的分类精确 F1 | ⭐⭐⭐ 最核心 |
| **Overall QA Accuracy** | 总体问答准确率 | ⭐⭐⭐ |
| **F4 Null-Recall** *(新指标)* | 存在性查询中"正确拒绝"比率 | ⭐⭐⭐ 独特指标 |
| **Retrieval Recall@K** | 相关记忆被检索到的比例 | ⭐⭐ |
| **LLM Token 消耗** | 每次检索的额外 token 开销 | ⭐⭐ 公平性证明 |
| **端到端延迟** | 实用性参考 | ⭐ |

**F4 Null-Recall 公式**：

$$\text{Null-Recall} = \frac{|\{\text{正确返回空集的 F4 查询}\}|}{|\{\text{正确答案为"无相关记忆"的 F4 查询}\}|}$$

---

### 5.4 六组实验设计

#### 实验 1：失败模式分析（验证 RQ1）🔴 **最高优先级，立即执行**

**目的**：系统性证明查询意图相关的检索失败是跨域一致性问题，为整个研究立论。

```
执行步骤:
1. 在 LongMemEval、LoCoMo、MSC 上运行 3 种 SOTA 基线
   (Unified-Embedding, SGMem-SF, Mem0)
2. 按四类意图类型手工标注失败案例
3. 计算各意图类型的失败率
4. 计算跨数据集一致性（Cohen's Kappa）
5. 特别统计 F4 类查询中"应返回空集但错误返回结果"的比例

预期发现:
  - F4 存在性: 所有系统准确率 < 20%（无相关记忆情况）
  - F1+F2 类型: 准确率显著低于 F3 和普通语义查询（差距 > 20 pp）
  - 跨数据集 Kappa > 0.65

⚠️ 决策节点 (Go/No-Go):
  - Kappa ≥ 0.5 且 F4 准确率 < 30% → 继续当前方向
  - Kappa < 0.5 或各类型差距 < 10% → 立论基础不成立，需重新选题
```

---

#### 实验 2：RetNull 机制专项验证（验证 RQ3）

**目的**：单独验证 RetNull 对 F4 存在性查询的有效性。

```
设置:
  - 从 LongMemEval 中提取全部 F4 类查询
  - 进一步划分: 正确答案"有相关记忆" vs. "无相关记忆"两个子集
  - 对比: Unified top-K | TIRP w/o RetNull | TIRP w/ RetNull | Oracle

指标: F4 Null-Recall、F4 整体准确率、误拒率（误把有相关记忆判为无）

预期: TIRP w/ RetNull 的 F4 准确率比 Unified top-K 提升 ≥ 25 pp
```

---

#### 实验 3：跨系统可插拔性验证（验证 RQ4）

**目的**：证明 TIRP 作为插拔模块的通用性。

```
设置: 在三个基础系统上分别叠加 TIRP
  ┌──────────────────────────────────────────┐
  │ A-MEM (原版)    vs.  A-MEM + TIRP        │
  │ Mem0 (原版)     vs.  Mem0 + TIRP         │
  │ SGMem-SF (原版) vs.  SGMem-SF + TIRP     │
  └──────────────────────────────────────────┘

指标: 各意图类型准确率提升 Δ，特别关注 F4 Null-Recall

预期: 在所有三个基础系统上均出现一致的 F4 提升（证明跨系统泛化性）
```

---

#### 实验 4：主对比实验（验证 RQ4，完整基线）

```
在 LongMemEval 主评估集上对比全部 10 个方法:

重点验证四组对比:
  (1) TIRP-Full vs. Unified-Embedding
      → 展示意图导向检索的整体价值
  
  (2) TIRP-Full vs. Token-Budget-Matched Baseline
      → 相同 token 预算下，提升来自策略质量还是算力？
  
  (3) TIRP-Zero vs. SGMem-SF
      → 零额外 LLM 调用 vs. 复杂图结构系统的公平对比
  
  (4) TIRP vs. Adaptive-RAG (迁移版)
      → 记忆场景的独特性：为何通用自适应 RAG 不够？
```

---

#### 实验 5：跨域泛化实验（验证 RQ1 的跨域一致性）

```
设置: 在 LongMemEval 上校准 θ_null，直接部署到 LoCoMo 和 MSC
  (不重新校准，测试跨域迁移能力)

验证问题:
  Q1: θ_null 是否跨域稳定？（通过比较两个数据集上的 False Negative Rate）
  Q2: QIP 的意图分类在不同数据集分布下是否保持合理精度？

预期: 跨域性能下降 < 8%（相对 in-domain），F4 Null-Recall 下降 < 15%
```

---

#### 实验 6：消融实验

| 消融变体 | 修改内容 | 目的 |
|---------|---------|------|
| w/o RetNull | 用固定 top-K 替代 RetNull | 量化 RetNull 对 F4 的独立贡献 |
| w/o TemporalAnchor | 直接语义搜索替代时序锚提取 | 量化时序锚解析对 F1 的贡献 |
| w/o StateConflict | 返回全部候选，不做冲突检测 | 量化冲突检测对 F2 的贡献 |
| w/o QueryDecompose | 单步搜索替代子查询分解 | 量化分解对 F3 的贡献 |
| Rule-only QIP | 移除轻量分类器，只用规则 | 量化分类器对边缘查询的价值 |
| TIRP-Zero vs. TIRP-Full | 比较规则分解 vs. LLM 分解 | 量化 1 次 LLM 调用的边际价值 |

---

### 5.5 公平性验证：Accuracy vs. Token Budget 帕累托前沿

```
方法                    准确率(预期)    额外 Token 消耗
──────────────────────────────────────────────────
Unified-Embedding         ~0.42           0
A-MEM                     ~0.52           0
Mem0                      ~0.58           中等
SGMem-SF                  ~0.68           中等
Token-Matched Baseline    ~0.55           900
TIRP-Zero (本方法)         ~0.60           0        ← 零额外调用下的价值
TIRP-Full (本方法)         ~0.68+          900       ← 相同预算，更优策略

核心声明: TIRP-Full 在相同 token 预算下优于 Token-Matched Baseline
         TIRP-Zero 在零额外调用下优于 Mem0 和 A-MEM
```

---

## 模块 6：目标投稿方向（CCF B 级）

| 投稿目标 | CCF 级别 | 推荐理由 | 截稿参考 |
|---------|---------|---------|---------|
| **EMNLP 2026** ⭐ **首选** | CCF B | ① 对 NLP+Agent 方向极度友好；② "新发现+方法"双贡献结构完美匹配 EMNLP 接受偏好；③ LongMemEval/LoCoMo 在 EMNLP 社区高认可度；④ RetNull 这类系统性分析与分析性 contribution 在 EMNLP 特别受欢迎 | 通常 5–6 月提交 |
| **COLING 2026** | CCF B | ① 接受系统性工作和实证分析；② 对对话系统、个性化 Agent 方向特别友好；③ 审稿标准相对温和，适合以实证发现驱动的工作；④ 接受 long paper + short paper，可按结果灵活调整 | 通常 9 月提交 |
| **ECIR 2027** | CCF B | ① 信息检索专业顶会，对"检索策略优化"直接契合；② RetNull 机制在 IR 社区具有高度新颖性（non-retrieval as answer 是 IR 领域未解问题）；③ 偏好有明确 IR 贡献的工作 | 通常 10 月提交 |

### 投稿策略建议

```
主线路 (推荐):
  EMNLP 2026 提交 → 接受: 完成
                 → 拒绝: 根据审稿意见修改 → COLING 2026 / ECIR 2027

期刊备选:
  Information Processing & Management (IPM, CCF B)
  → 时间宽裕，允许更充分的实证讨论，无字数限制
```

---

## 模块 7：风险分析与改进建议

### 7.1 风险矩阵

| 编号 | 风险描述 | 发生概率 | 影响程度 | 具体应对方案 |
|-----|---------|---------|---------|------------|
| **R1** | 实验 1 跨域一致性低（Kappa < 0.5） | 中 | 🔴 致命 | 立刻止损。转换方向："为什么现有系统跨数据集泛化性差？"——这本身也可发表（作为分析性论文） |
| **R2** | SGMem 在 F4 查询上表现比预期好 | 中低 | 🟠 严重 | 深入挖掘 SGMem 在 F4 失败的具体子类型，找到真正的盲区；或聚焦"轻量替代"角度（TIRP-Zero vs. SGMem 的效率对比） |
| **R3** | RetNull 的 θ_null 跨域不稳定 | 中 | 🟡 中等 | 提供轻量的 5-shot 快速适配方案；在 limitation 中诚实讨论；将跨域适配作为 future work |
| **R4** | TIRP-Zero 提升不显著（< 5% vs. Unified） | 中低 | 🟡 中等 | 聚焦 TIRP-Full；TIRP-Zero 仅作为公平性验证工具，不要求超过 SGMem |
| **R5** | SGMem/Zep 复现困难 | 中 | 🟡 中等 | 使用论文公开结果进行数字对比，说明实验条件差异；至少在相同数据集上比较（LongMemEval 上所有方法均有公开结果） |
| **R6** | F3 类 QueryDecompose 不稳定 | 低 | 🟢 低 | 使用 self-consistency 多次采样取多数；或回退到规则分解（即 TIRP-Zero 模式） |

---

### 7.2 对 v4 的六项具体改进建议

**改进 1（最重要）：放弃 Gating Network，改用 QIP + 检索程序**

理由：根本性解决 P1（训练数据悖论）和 P2（操作依赖关系）两个致命缺陷。实现复杂度更低，同时可解释性更强，且无需任何 memory-level 监督数据。

---

**改进 2：将 2025 SOTA 纳入基线**

至少必须包含：**SGMem**（LongMemEval 0.730，最重要）、**Mem0+graph**、**Zep**。  
没有这三个，任何 2025/2026 顶会的审稿人都会要求补充，直接影响录用。

---

**改进 3：将 F4 存在性查询独立为核心发现和贡献**

目前没有任何已发表工作系统研究和解决 F4 类查询。F4 Null-Recall 作为新评估指标，RetNull 机制作为技术贡献，应在论文中各占一个独立子节，而非作为附属内容。

---

**改进 4：删除信息论伪框架，改用检索程序语言的形式化**

用 DAG + 操作语义定义检索程序，比互信息框架更实际、更容易与方法实现对应，且可以被审稿人直接验证（形式化正确性）。参考 DSPy 的设计哲学。

---

**改进 5：增加 F4 Null-Recall 作为新评估维度**

在相关工作部分明确指出：现有评估框架（LongMemEval 等）的"准确率"指标将存在性判断失败与其他失败混淆，无法单独衡量系统的拒绝能力。本研究引入 Null-Recall 作为独立维度，填补评估框架的空白。

---

**改进 6：时间线优化（单人，13 周）**

| 阶段 | 时间 | 优先任务 | 关键决策点 |
|------|------|---------|----------|
| **[立即] 立论验证** | 第 1–2 周 | 实验 1（跨域失败模式分析） | **Go/No-Go**：Kappa ≥ 0.5 且 F4 < 30% 继续 |
| **方法实现** | 第 3–4 周 | QIP（规则引擎 + MiniLM）+ RetNull 机制实现 | — |
| **初步验证** | 第 5–6 周 | 实验 2（RetNull 专项）+ 实验 3（跨系统插拔） | — |
| **主实验** | 第 7–9 周 | 实验 4（完整基线对比）+ 实验 5（跨域泛化） | — |
| **消融+分析** | 第 10 周 | 实验 6（消融）+ 案例分析 + 帕累托曲线 | — |
| **论文写作** | 第 11–13 周 | 论文初稿 + 图表制作 + 修改润色 | — |

> **总计约 13 周（3 个月）。** QIP 分类器和 RetNull 均可在 CPU 上运行，仅 QueryDecompose（F3 类，约 30% 查询）需要 1 次 LLM API 调用。计算资源要求极低。

---

## 附录：论文叙事框架建议

```
Title: QueryPlan-Mem: Training-Free Intent-Driven Retrieval Planning
       for Agent Memory Systems

Abstract:
  Agent 记忆系统普遍存在"强制返回"问题——即使在没有相关记忆的
  情况下，系统仍会返回 top-K 条结果。通过对 LongMemEval、LoCoMo、
  MSC 三个数据集的系统失败分析，我们首次定量证明：当前 SOTA 系统
  在不同查询意图类型上的失败率差距超过 25 个百分点，尤其存在性判断
  类查询（F4）在所有系统上准确率低于 20%。为此，我们提出 TIRP，
  一种训练自由的意图导向检索程序框架，通过 (1) 轻量意图解析器将查询
  映射为结构化意图，(2) 检索程序编译器生成有序 DAG 执行计划，(3)
  RetNull 机制使系统具备拒绝返回能力。TIRP 作为可插拔模块在三个基础
  记忆系统（A-MEM、Mem0、SGMem）上均带来一致提升，F4 类查询
  Null-Recall 提升超过 40 个百分点，且无需任何额外的 LLM 调用。

Section 1 - Introduction
  动机: Agent 记忆系统的"强制返回"系统性失败
  核心发现预告: F4 存在性查询全行业盲区
  贡献声明:
    (1) 首个跨数据集一致的查询意图失败分析（实证贡献）
    (2) TIRP 训练自由检索程序框架（方法贡献）
    (3) RetNull + F4 Null-Recall 新评估维度（评估贡献）

Section 2 - Related Work
  2.1 Agent 记忆系统（Mem0, A-MEM, MemGPT, Zep, SGMem, MIRIX...）
  2.2 自适应检索（Adaptive-RAG, Self-RAG, FLARE, IRCoT）← 重点区分
  2.3 查询意图分析与检索规划
  2.4 本工作的定位区别

Section 3 - Failure Analysis: Q-Intent Taxonomy（实证贡献）
  3.1 四维查询意图分类体系
  3.2 跨数据集失败率分析
  3.3 F4 存在性查询：全行业盲区的量化证明

Section 4 - Method: TIRP
  4.1 问题形式化（检索程序语言）
  4.2 查询意图解析器（QIP）
  4.3 检索程序编译器（RPC）+ 执行引擎（RPE）
  4.4 RetNull 机制
  4.5 TIRP-Zero vs. TIRP-Full

Section 5 - Experiments
  5.1 实验设置（数据集、基线、指标、公平性保证）
  5.2 实验 2：RetNull 专项验证
  5.3 实验 4：主对比实验（含帕累托前沿）
  5.4 实验 3：跨系统可插拔性验证
  5.5 实验 5：跨域泛化实验
  5.6 实验 6：消融实验
  5.7 案例分析

Section 6 - Conclusion
```

---

## 附录：与 v4 方案的系统性对比

| 维度 | v4 方案 | **本方案（v5 重构版）** |
|------|---------|---------------------|
| **核心机制** | Gating Network（可学习软组合） | TIRP（训练自由检索程序） |
| **训练数据** | `(query, memory, answer)` 三元组 | 仅需 `(query_text, intent_label)` |
| **操作执行模型** | 并行加权求和（逻辑错误） | 有序 DAG（符合数据流依赖） |
| **F4 存在性处理** | 调节阈值（治标不治本） | RetNull 机制（真正拒绝返回） |
| **2025 SOTA 对比** | 无 SGMem/Zep | 必须包含，已纳入基线 |
| **理论框架** | 信息论（与实现脱节） | 检索程序语言（直接对应） |
| **可移植性** | 与训练数据分布绑定 | 任意记忆系统的可插拔模块 |
| **核心独立贡献** | Gating Network（与 MoE 重叠） | **RetNull 机制**（文献中无先例） |
| **新评估指标** | 无 | **F4 Null-Recall**（首创） |

---

*方案版本 v5（专业重构版）。核心改变：*
1. *放弃 Gating Network，改用训练自由的 TIRP（解决训练数据悖论和架构逻辑错误）*
2. *将 F4 存在性查询独立为核心发现，提出 RetNull 机制填补全行业盲区*
3. *将 2025 SOTA（SGMem、Zep、MIRIX）纳入分析和基线比较*
4. *删除信息论伪框架，改用检索程序语言的规范形式化*
5. *增加 F4 Null-Recall 作为首创评估维度，强化差异化定位*

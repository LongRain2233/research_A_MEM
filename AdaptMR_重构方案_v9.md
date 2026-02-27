# TypeProg-Mem：面向 Agent 个人记忆时效性失配问题的类型感知检索程序框架

> **方案版本：** v9（核心锚点修正版）  
> **生成日期：** 2026-02-27  
> **基于：** v6 完整架构 + 核心创新点从 F4 转向 F2（知识更新/状态冲突）  
> **核心定位：** CCF B 级会议（首选 EMNLP 2026）

---

## ⚠️ 前置声明：v6 → v9 的核心调整说明

> v6 已在技术架构、文献调研、基线设置上做了扎实工作，整体框架保留。本版本的唯一调整方向是：**将核心创新锚点从 F4（存在性判断/RetNull）迁移至 F2（知识更新/状态冲突/StateConflictDetect）**，并在此基础上重构研究叙事。

### v6 中被保留的有效内核

| 元素 | 保留原因 |
|------|---------|
| ✅ **领域演进四阶段描述（1.1）** | 文献扎实，完整保留 |
| ✅ **SimpleMem 精确区分分析** | 竞争对手定位清晰，完整保留 |
| ✅ **MQTP 规则引擎 + MiniLM 架构** | 技术方案合理，保留（去除F4触发词行） |
| ✅ **DAG 检索程序执行框架（RPC/RPE）** | 方法核心，完整保留 |
| ✅ **P_temporal（F1）程序模板** | 技术可行，保留 |
| ✅ **P_versioning（F2）程序模板** | **升级为核心贡献** |
| ✅ **P_aggregate（F3）程序模板** | 技术可行，保留 |
| ✅ **TypeProg-Zero / TypeProg-Full 双版本** | 保留（去除 RetNull 列） |
| ✅ **完整 2026 SOTA 基线表** | 保留 |
| ✅ **跨域泛化实验（LoCoMo/MSC）** | 保留 |
| ✅ **消融实验框架** | 保留（去除 w/o RetNull 行） |
| ✅ **风险矩阵结构** | 保留（去除 F4 专项风险）|
| ✅ **14 周时间线** | 调整节点说明 |

### v6 中被移除的内容

| 元素 | 移除原因 |
|------|---------|
| ❌ **F4 存在性查询作为"核心突破口"** | Benchmark 本体论缺陷：LongMemEval/LoCoMo 均无可靠的 "null-GT" 标注；Mem0 论文明确："unanswerable questions were excluded from evaluation because ground truth answers were unavailable" |
| ❌ **RetNull 机制（P_existence 程序）** | 依附于 F4，benchmark 无法验证 |
| ❌ **F4 Null-Recall 评估指标** | 无可靠 GT，无法计算 |
| ❌ **实验 2（RetNull 专项验证）** | 前提不成立 |
| ❌ **θ_null 跨域稳定性实验** | 前提不成立 |

---

## 重构核心：TypeProg-Mem 与 SimpleMem 的精确边界（v9 更新版）

| 维度 | SimpleMem Intent-Aware Retrieval Planning | TypeProg-Mem（本方案） |
|------|------------------------------------------|----------------------|
| **"意图"的含义** | 检索**深度/范围**（标量复杂度 d，影响 top-n 的 n 值）| 记忆查询**类型分类**（F1时序/F2状态/F3聚合，决定执行路径）|
| **输出形式** | {qsem, qlex, qsym, depth_d}（4 个并行查询参数）| 有向无环图（DAG）检索程序（有序操作序列，含分支）|
| **操作依赖** | 三路并行（无依赖关系，最终合并）| 串行/分支 DAG（操作间有数据流依赖）|
| **知识更新查询（F2）** | ❌ 无冲突检测（加宽召回反而引入更多过时信息）| ✅ **StateConflictDetect + LatestVersionSelect** |
| **时序推理查询（F1）** | ❌ 无时序结构（语义检索丢失时间顺序）| ✅ TemporalAnchorExtract + TemporalFilter |
| **训练开销** | LLM 推理一次（意图规划）| 规则引擎（0 LLM 调用，或选择性 1 次仅用于 F3）|
| **失败模式分析** | 无系统性失败分析 | **跨域类型化失败分析**（核心实证贡献）|

**结论**：SimpleMem 解决"检索多少"；TypeProg-Mem 解决"用什么操作序列检索、如何处理版本冲突"。两者在 F1/F2/F3 三类核心场景上提供互补增益，可叠加使用。

---

## 模块 1：研究动机

### 1.1 领域技术演进精确定位（基于 52 篇文献）

**第一阶段（2019–2022）**：RAG 范式确立（Lewis et al., 2020），向量检索成为核心工具。  
**第二阶段（2022–2024）**：记忆**结构**多样化。MemGPT（OS式分层）、A-MEM（Zettelkasten图）、Zep（时序知识图谱）、AriGraph（语义+情节双图）。竞争焦点：**构建质量**。  
**第三阶段（2024–2025 中期）**：SOTA 转向精细化构建。SeCom（ICLR 2025, 构建粒度实验）、SGMem（Huawei 2025, 七级句子图索引）、Mem-α（RL学习记忆管理）、MIRIX（六类记忆类型）、LightMem（三阶段分层+睡眠更新）、EverMemOS（MemCell生命周期）。  
**第四阶段（2025 下半年–2026）**：检索层开始出现分化尝试。SimpleMem 提出了基于 LLM 推理的检索深度自适应；MemoTime（WWW 2026）提出针对**时序知识图谱**的算子感知检索。

**关键转折**：第三、四阶段的进展共同表明，记忆系统的提升空间已从"结构优化"转向"检索策略优化"。然而，即使是 SimpleMem 这样最新的检索侧工作，仍将所有查询类型视为同质化处理，这带来了**语义检索在特定查询类型上的系统性失效**。

### 1.2 核心研究空白：语义检索的三类系统性失效

向量语义检索基于一个核心假设：**语义相似 = 相关**。然而，在个人对话记忆场景中，这一假设在以下三类查询上系统性失效：

**失效一（F2，最核心）：知识更新查询的"时效性失配"（Temporal Staleness Problem）**

> 用户曾说"我住北京"，后来更新为"我已搬去上海"。当用户问"我现在住哪里？"时，"我住北京"与查询语义相似度极高，而"我已搬去上海"因包含"搬"等迁移动词导致相似度反而偏低。语义检索**系统性**返回旧版本，不是偶发错误，而是 embedding-based 检索的结构性缺陷。
>
> LongMemEval 专门为此设计了 `knowledge_update` 子集，文献证明此类查询是现有系统最难处理的类别之一。

**失效二（F1）：时序推理查询的"结构性盲区"**

> 对于"我先开始跑步还是先搬到杭州？"，向量相似度无法携带事件时序结构，仅能返回语义相关段落，无法建立事件先后顺序。MemoTime 解决了公开知识图谱 TKGQA 中的时序问题，但与**个人对话记忆**在数据结构、查询类型、评估设置上完全不同。

**失效三（F3）：聚合推理查询的"截断性失配"**

> 对于"我提到的所有餐厅哪种菜系最多？"，单次 top-K 无法保证全集召回，导致聚合结论不完整。这是 top-K 范式的固有局限，与检索质量无关。

**关键发现**：SimpleMem、LightMem、EverMemOS 等 2025-2026 最新工作在三类查询上均无针对性解决方案。SimpleMem 加宽召回（更大的 top-K）在 F2 场景下**反而会引入更多过时冲突信息**，加剧时效性失配。

### 1.3 问题严重性的定量证据（文献支持）

| 查询类型 | 代表性失败场景 | 根本原因 | 预计全行业最优基线 |
|---------|------------|---------|----------------|
| **F1 时序推理** | "我先开始跑步还是先搬到杭州？" | 向量检索不携带事件时序结构 | ~0.45–0.58（LongMemEval temporal 子集）|
| **F2 知识更新** ⭐ | "我现在住在哪里？"（有地址变更） | 旧信息语义相似度高，新信息因描述差异相似度反低 | ~0.50–0.65（LongMemEval knowledge_update 子集）|
| **F3 多跳聚合** | "我提到的所有餐厅中哪种菜系最多？" | 单次 top-K 无法"先全集收集、再聚合" | ~0.40–0.55 |

> ⭐ **F2 知识更新是本研究的核心突破口**：LongMemEval 原文已定量报告此类查询的低准确率，且 SimpleMem 的"加宽召回"策略在此类场景下的改善极为有限（预期：SimpleMem 在 knowledge_update 子集的提升 < 5pp，而其在 temporal/multi-hop 子集的整体提升为 26.4%）。F2 的技术缺陷有明确的解决路径：**在检索结果返回 LLM 之前，显式进行版本冲突检测并选择最新版本**。

---

## 模块 2：研究问题（RQ 形式）

**RQ1（实证基础）**：在个人对话记忆场景中，当前 SOTA 系统（SGMem、LightMem、EverMemOS、SimpleMem）的检索失败是否在查询类型维度上呈现系统性分布？特别是：**SimpleMem 的 Intent-Aware Planning 是否改善了 F2 知识更新类查询的时效性失配问题？** 该分布是否在 LongMemEval、LoCoMo、MSC 三个数据集上保持跨域一致性？

> *验证方式*：在三个数据集上运行基线，按 F1/F2/F3 意图类型拆分失败率并统计主要失败模式（时效性错误 / 结构性错误 / 截断性错误）；计算 Cohen's Kappa 检验跨域一致性；重点验证 SimpleMem 在 F2 knowledge_update 子集上的 Stale Error Rate。

**RQ2（方法核心）**：针对 F2 知识更新查询，能否通过显式的**冲突检测-版本选择程序**（StateConflictDetect + LatestVersionSelect）在检索层消除时效性失配？这一机制能否在无 LLM 调用的情况下实现，并为 SimpleMem 的召回优化提供正交补充？

> *验证方式*：在 LongMemEval knowledge_update 子集上专项对比；验证 TypeProg 与 SimpleMem 叠加后的增益是否超过各自单独的增益之和。

**RQ3（方法通用性）**：F1/F2/F3 三类类型专属检索程序能否在**无端到端训练**的条件下实现，以可插拔模块形式在不同基础记忆系统（A-MEM、Mem0、SGMem）上一致有效？

> *验证方式*：TypeProg-Zero 与 SimpleMem 比较分类延迟 vs 精度；在三个不同基础系统上分别叠加，验证跨系统一致性。

---

## 模块 3：创新点（3 个差异化贡献）

### 创新点 1（实证贡献，主贡献之一）：首个针对语义检索时效性失配的跨域类型化失败分析框架

**具体内容**：
通过在 LongMemEval、LoCoMo、MSC 三个数据集上对 6 种 SOTA 系统（含 SimpleMem）进行系统失败分析，本研究**首次**：

1. 建立独立于单一评测集的**三维记忆查询类型分类体系（MQ-Taxonomy: F1/F2/F3）**，对应语义检索的三类结构性失效
2. 定量证明：现有最优系统在 F2 knowledge_update 子集上的 Stale Error Rate 超过 **40%**，而同系统在 single-hop 子集上的错误率不足 25%，差距显著
3. 首次揭示 SimpleMem 的召回加宽策略（更大 top-K）对 F2 时效性失配无改善甚至轻微恶化（引入更多过时候选）
4. 验证三类失败模式的跨域一致性（目标 Cohen's Kappa > 0.65）

**与 SimpleMem 的差异**：SimpleMem 仅报告了 LoCoMo 四个子类的综合性能，**未进行类型化失败分析，未区分 Stale Error vs. 其他错误类型，未提供跨数据集一致性验证**。

**可发表性**：即使后续方法实验不如预期，这一实证发现本身已具备独立发表价值，因为它为整个 Agent 记忆研究社区提供了新的诊断框架，并揭示了 SimpleMem 等工作忽视的根本性失效模式。

---

### 创新点 2（方法贡献，主贡献）：类型专属检索程序框架（TypeProg）

**具体内容**：
提出 **TypeProg（Type-Specific Memory Retrieval Programs）**，针对每类查询构造有序的 DAG 检索程序：

| 特性 | SimpleMem Intent-Aware Planning | TypeProg（本方法）|
|------|--------------------------------|-----------------|
| 意图识别目标 | 检索复杂度（深度 d）和查询改写 | **查询类型**（F1/F2/F3）+ 操作参数 |
| 输出形式 | 4 个并行查询参数（标量+字符串）| **可执行 DAG 程序**（有序操作图）|
| 操作执行模式 | 三路**并行**检索 + 合并 | **串行/分支** DAG（符合数据流依赖）|
| **F2 知识更新** | ❌ 加宽召回，无冲突处理 | ✅ **StateConflictDetect + LatestVersionSelect** |
| 识别器开销 | LLM 推理 1 次（每次查询）| **规则引擎**（0 LLM 调用，~30% 查询用轻量 BERT）|
| 与 SimpleMem 关系 | — | **正交且互补**（可叠加于 SimpleMem 之上）|

**三类检索程序模板**：

```
P_temporal(F1)：时序推理程序
  TemporalAnchorExtract(q)          ← 规则：提取时间锚事件
  → TimestampResolve(anchor, M)     ← 语义搜索：解析锚事件时间戳
  → TemporalFilter(M, ts, relation) ← 时序过滤（早于/晚于/之间）
  → TemporalSort → TopK

P_versioning(F2)：知识更新程序（核心新机制）
  EntityExtract(q)                       ← 规则：提取状态目标实体
  → SemanticSearch(q, K=20)             ← 宽召回（确保新旧版本均在候选集）
  → StateConflictDetect(candidates, entity) ← 检测同一实体的版本冲突
  → LatestVersionSelect(conflict_groups)    ← 基于时间戳选取最新版本
  → TopK

P_aggregate(F3)：聚合推理程序
  QueryDecompose(q)                 ← few-shot LLM（1次，仅F3需要）
  → ForEach(sub_q: SemanticSearch(sub_q, K=10))
  → Deduplicate → AggregateReason
```

---

### 创新点 3（机制+评估贡献）：StateConflictDetect 机制与 Stale Error Rate 评估指标

**问题**：当前**所有** 52 篇已调研文献（含 SimpleMem、LightMem、EverMemOS、SGMem）的检索层均无显式版本冲突检测机制——系统无论如何都将 top-K 全部返回给 LLM，当候选集中同时包含"我住北京"和"我已搬去上海"时，LLM 必须通过阅读理解消歧，而这在长上下文或候选质量低的情况下容易出错。

**StateConflictDetect 实现**（零 LLM 调用）：

```python
def state_conflict_detect(candidates, entity, memory_store):
    """
    在检索层显式检测并消解同一实体的版本冲突。
    
    技术路线：
    1. 基于实体共指将候选记录聚类（规则 + 浅层 NER）
    2. 基于属性互斥性检测冲突（语义蕴含 + 属性类型规则）
    3. 基于时间戳选取最新版本（不依赖 LLM）
    
    与 SimpleMem 的关键区别：SimpleMem 加宽 top-K 会引入
    更多版本，本机制在加宽召回之后显式消歧，两者互补。
    """
    # Step 1: 按实体分组
    entity_groups = group_by_entity_coref(candidates, entity)
    
    resolved = []
    for group in entity_groups:
        if len(group) <= 1:
            resolved.extend(group)
            continue
        
        # Step 2: 检测同属性的互斥描述
        conflict_pairs = detect_mutual_exclusion(group)
        
        if conflict_pairs:
            # Step 3: 有冲突 → 选最新时间戳
            latest = select_by_timestamp(group)
            resolved.append(latest)
        else:
            # 无冲突（补充信息，非版本更新）→ 全部保留
            resolved.extend(group)
    
    return resolved


def detect_mutual_exclusion(group):
    """
    两条记录互斥：描述同一属性但取值不同。
    实现：基于属性类型（位置/职业/关系/偏好）的规则 + 
         轻量语义蕴含模型（MiniLM cross-encoder, < 10ms）。
    """
    ...
```

**θ 校准方法**（无需训练，使用验证集校准互斥性阈值）：
1. 从 LongMemEval knowledge_update 验证集中抽取 ~100 个有版本冲突的对话片段
2. 校准互斥性检测阈值，使 F2 冲突检测 Precision > 85%，Recall > 80%

**Stale Error Rate（新评估指标）**：

$$\text{Stale-Err}_{\text{F2}} = \frac{|\{\text{F2 查询中返回过时版本的错误案例}\}|}{|\{\text{F2 查询总数（含知识更新的对话）}\}|}$$

> 此指标与 F2 Accuracy 互补：F2 Accuracy 从问答角度衡量，Stale-Err 从检索层直接衡量，两者共同说明检索程序改善的来源是版本冲突消解，而非其他因素。

---

## 模块 4：方法设计

### 4.1 整体框架描述

```
┌────────────────────────────────────────────────────────────────────┐
│                    TypeProg-Mem 整体架构                             │
├────────────────────────────────────────────────────────────────────┤
│  用户查询 q                                                          │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                        │
│  │ [模块 A] 记忆查询类型解析器 (MQTP)          │                        │
│  │ 方法: 规则引擎(70%) → MiniLM分类器(30%)   │                        │
│  │ 输出: 类型 T ⊆ {F1, F2, F3}              │                        │
│  │       + 类型参数 (时间锚/实体名/聚合条件)   │                        │
│  └──────────────────┬──────────────────────┘                        │
│                     │                                               │
│                     ▼                                               │
│  ┌─────────────────────────────────────────┐                        │
│  │ [模块 B] 检索程序编译器 (RPC)              │                        │
│  │ 输入: 类型 T + 参数                       │                        │
│  │ 方法: 类型→程序模板映射 → DAG 实例化       │                        │
│  │ 输出: 可执行检索程序 π（DAG 形式）         │                        │
│  └──────────────────┬──────────────────────┘                        │
│                     │                                               │
│                     ▼                                               │
│  ┌─────────────────────────────────────────┐                        │
│  │ [模块 C] 检索程序执行引擎 (RPE)            │                        │
│  │ 原子操作（全部轻量实现）:                  │                        │
│  │  ├── TemporalAnchorExtract (规则)        │                        │
│  │  ├── StateConflictDetect   (规则+MiniLM) │ ← 核心新机制            │
│  │  ├── LatestVersionSelect   (时间戳排序)  │ ← 核心新机制            │
│  │  └── QueryDecompose        (few-shot)   │                        │
│  │ 输出: 精炼记忆集 M*                       │                        │
│  └──────────────────┬──────────────────────┘                        │
│                     │                                               │
│                     ▼                                               │
│  ┌─────────────────────────────────────────┐                        │
│  │ [模块 D] 记忆融合与生成                    │                        │
│  │ 标准 prompt（M* 已保证时效性正确）         │                        │
│  └─────────────────────────────────────────┘                        │
│                                                                     │
│  ★ 与基础系统接口：仅需 semantic_search(q, k) + timestamp 字段       │
│    可插拔于 A-MEM、Mem0、SGMem、SimpleMem 等任意系统                  │
└────────────────────────────────────────────────────────────────────┘
```

### 4.2 模块 A：记忆查询类型解析器（MQTP）

**设计原则**：规则优先（零开销），轻量分类器兜底（低开销），完全不依赖 memory context 或 answer 监督。

#### 层级一：规则引擎（处理 ~70% 查询）

| 类型 | 中文触发模式 | 提取参数 |
|-----|------------|---------|
| **F1 时序** | 先/后/之前/之后/什么时候/最近一次/首次/早于/晚于/哪个先 | 时间锚事件、时序关系 |
| **F2 知识更新** | 现在/目前/最新/最近/改了/变成/已经/更新为/当前/还是/现在还 | 目标实体名称 |
| **F3 聚合** | 所有/哪些/几次/总共/列表/都曾经/统计/最多/最少/多少次 | 聚合条件、目标属性 |

> **注**：F2 触发词刻意强调"当前状态"语义，与 F1 时序触发词在语义上有明确区分。边界模糊案例（如"我最近搬家了吗？"）由 MiniLM 分类器处理。

#### 层级二：轻量 BERT 分类器（处理规则未命中的 ~30%）

- 使用 MiniLM-L6-v2（384 维，~33M 参数，CPU 推理 < 10ms）
- **训练数据**：仅 `(query_text, type_label)` 对，约 200 条手工标注样本，覆盖三类及"无特殊类型（单跳）"
- **关键约束**：不使用任何 memory 内容或 answer，仅依赖查询文本特征，确保跨 benchmark 泛化

#### 与 SimpleMem 的关键区别

```
SimpleMem Intent Planning：
  输入: (query, history) → LLM 推理 → {qsem, qlex, qsym, depth_d}
  本质: 优化"检索多少"和"用什么形式检索"
  开销: 每次查询 1 次 LLM 调用
  
MQTP 类型解析：
  输入: query_text → 规则/MiniLM → {F1/F2/F3, params}
  本质: 识别"查询的记忆操作语义类型"→ 匹配专属执行程序
  开销: 规则 = 0；MiniLM = <10ms，0 LLM 调用
```

### 4.3 模块 B：检索程序编译器（RPC）

RPC 将类型映射为有向无环图（DAG）形式的检索程序。关键设计：操作间有**数据流依赖**，必须串行或分支执行，不允许无脑并行（这是 v4 Gating Network 的根本性错误，v5/v6 已修正，v9 延续此设计）。

**复合类型示例（F1+F2：时序+知识更新）**：

```
                  query
                 /      \
temporal_anchor_extract  entity_extract
               |              |
         ts_resolve      semantic_search(k=20)
               |              |
       temporal_filter   state_conflict_detect
               |              |
             TopK_t     latest_version_select
               \               /
                join + deduplicate
                      |
                  TopK_final
```

> **设计说明**：当查询同时含有时序和状态更新语义时（如"我去年到现在换了几次工作？"），F1 分支提取时间窗口，F2 分支保证工作变更记录的版本正确性，两路结果合并后去重。

### 4.4 两个系统版本

| 版本 | LLM 额外调用 | QueryDecompose | StateConflictDetect | 对比对象 |
|------|------------|---------------|---------------------|---------|
| **TypeProg-Zero** | **0 次** | 规则分解（关键词拆分） | ✅ 完整 | 验证纯结构化检索价值；与 SGMem/SimpleMem 公平对比 |
| **TypeProg-Full** | 1 次（仅 F3）| few-shot LLM 分解 | ✅ 完整 | 追求最佳性能，特别是复杂聚合查询 |

---

## 模块 5：实验方案

### 5.1 数据集选择

| 数据集 | 规模 | 查询类型覆盖 | 使用目的 |
|--------|------|------------|---------|
| **LongMemEval** | 500 查询，50 个长对话 | 5 类（含时序、knowledge_update、多跳） | 主评估集 + MQTP 分类器训练（严格 35/5/10 对话切分） |
| **LoCoMo** | ~2000 查询（4类：单跳/多跳/时序/开放域）| 时序密集，自然分布 | 跨域泛化验证 |
| **MSC（Multi-Session Chat）** | ~200 查询 | 多 session 自然对话 | 补充验证 F2 跨域稳定性 |

> **说明**：DMR 不使用（评估价值极低，Zep 论文已明确指出）。LongMemEval 数据使用协议：35 个对话用于 MQTP 训练，5 个对话用于 StateConflictDetect 阈值校准，10 个对话作为完全 held-out 测试集，三者**不共享**。

### 5.2 基线方法（2026 年完整版）

| 基线 | 类别 | 额外 LLM 调用 | LongMemEval 成绩 | 备注 |
|------|------|-------------|----------------|------|
| Unified Embedding (top-K) | 纯向量检索 | 0 | ~0.40 | 最基础基线 |
| **A-MEM** | 图结构记忆 | 少量 | ~0.52 | 可本地复现 |
| **Mem0** | 工业级系统 | 中等 | ~0.58 | LoCoMo 已有结果 |
| **SGMem** | 2025 SOTA | 中等 | **0.730** | 关键竞争对手 |
| **LightMem** | 分层压缩+睡眠更新 | 中等 | **提升 7.7%** | 新增，必须包含 |
| **EverMemOS** | 自组织记忆OS | 中等 | **提升 6.7%** | 新增，必须包含 |
| **SimpleMem** | 意图感知检索 | 每次 1 次 | LoCoMo +26.4% F1 | 🚨 **直接竞争对手，重点对比 F2 子集** |
| Adaptive-RAG（迁移版） | 自适应 RAG | 少量 | — | 验证记忆场景独特性（F2 处理能力） |
| Token-Budget-Matched Baseline | 公平性对照 | 与 Full 等量 | — | 验证提升来自策略而非算力 |
| **TypeProg-Zero（本方法）** | 类型专属程序 | **0** | — | — |
| **TypeProg-Full（本方法）** | 类型专属程序 | 1 次（仅F3）| — | — |
| Oracle-Strategy | 上界参考 | — | — | 手工选最优检索策略 |

> ⚠️ **SimpleMem 是最关键的竞争对手**：重点需要在 LongMemEval 的 `knowledge_update` 子集上与 SimpleMem 直接对比，若无法复现，至少引用其公开结果并在相同数据集上对比。预期：SimpleMem 在此子集上的提升 < 5pp（因为 top-K 加宽对 Stale Bias 无帮助），而 TypeProg 在此子集上的提升 > 15pp。

### 5.3 评估指标

| 指标 | 说明 | 优先级 |
|------|------|--------|
| **F2 knowledge_update Accuracy** | 知识更新子集问答准确率 | ⭐⭐⭐ 最核心 |
| **Stale Error Rate（新指标）** | F2 查询中返回过时版本的比率 | ⭐⭐⭐ 独特指标 |
| **Per-type Accuracy（F1/F2/F3）** | 三类查询的分类精确 F1 | ⭐⭐⭐ |
| **Overall QA Accuracy** | 总体问答准确率 | ⭐⭐⭐ |
| **Retrieval Recall@K** | 相关记忆被检索比例 | ⭐⭐ |
| **LLM Token 消耗** | 额外 token 开销（公平性）| ⭐⭐ |
| **端到端延迟** | 实用性参考 | ⭐ |

$$\text{Stale-Err}_{\text{F2}} = \frac{|\{\text{F2 查询中返回过时版本的错误案例}\}|}{|\{\text{LongMemEval knowledge\_update 子集全集}\}|}$$

> Stale Error Rate 与 F2 Accuracy 互补，前者从检索层直接度量版本冲突消解质量，后者从问答层综合衡量。两者同时提升才能说明改进来自本质机制，而非偶然因素。

### 5.4 五组实验设计

#### 实验 1：类型化失败模式分析（验证 RQ1）🔴 **最高优先级，立即执行**

**目的**：建立整个研究的立论基础；揭示 SimpleMem 等工作忽视的 F2 时效性失配问题。

```
执行步骤：
1. 在 LongMemEval 上运行 4 种基线：
   Unified-Embedding、SGMem、SimpleMem、LightMem
2. 按 F1/F2/F3 三类拆分失败案例（约 150 个，各类约 50 个）
3. 对 F2 失败案例进一步细分：
   → Stale Retrieval（返回旧版本）
   → Irrelevant Retrieval（返回无关记忆）
   → 其他
4. 计算各类型 Stale Error Rate 和失败率
5. 计算跨数据集一致性（LongMemEval vs LoCoMo，Cohen's Kappa）
6. 【关键】统计：SimpleMem 在 knowledge_update 子集的 Stale Error Rate
   → 如果 SimpleMem 的 Stale-Err 仍 > 35%，则 StateConflictDetect 立论稳固
   → 如果 SimpleMem 意外解决了 F2（Stale-Err < 15%），则需要调整重点

⚠️ Go/No-Go 决策节点：
  - F2 Stale-Err（基线） > 35% 且 Kappa ≥ 0.5 → 继续当前方向
  - Kappa < 0.5 → 立论基础动摇，考虑单一数据集深度分析路线
```

#### 实验 2：F2 StateConflict 专项验证（验证 RQ2）

```
设置（重点实验，论文 Section 5 核心）：
  数据集：LongMemEval knowledge_update 子集（预计 ~80-100 queries）
  
  对比方法：
    Unified top-K
    SimpleMem（知识更新子集性能）
    TypeProg-Zero（0 LLM 调用）
    TypeProg-Full（1 LLM 调用，仅F3）
    SimpleMem + TypeProg（叠加验证正交性）
    Oracle（手工选最新版本）

  关键指标：
    F2 Accuracy（主指标）
    Stale Error Rate（机制验证）
    Stale-Err 下降幅度（TypeProg vs 各基线）
  
  预期：
    TypeProg vs Unified-top-K：Stale-Err 下降 ≥ 25pp
    TypeProg vs SimpleMem：F2 Accuracy 提升 ≥ 10pp
    SimpleMem+TypeProg vs 各自单独：叠加增益 > 各自之和（正交验证）
```

#### 实验 3：类型专属程序 vs SimpleMem（验证 RQ2 全部类型，审稿人最关注）

```
设置：在 LongMemEval 三个子集上全面对比
  SimpleMem               F1/F2/F3 各子集 Accuracy
  TypeProg-Zero            F1/F2/F3 各子集 Accuracy  
  TypeProg-Full            F1/F2/F3 各子集 Accuracy
  SimpleMem + TypeProg     正交叠加结果

核心声明：
  - TypeProg 与 SimpleMem 提升来自不同机制（类型分布验证互补性）
  - TypeProg 在 F2 子集上提供 SimpleMem 无法实现的 Stale-Err 下降
  - SimpleMem 在 F1/F3 类型上可能略优（因为其广检索对时序/聚合有帮助）
    → 这种"各有优势"正是正交互补的最强证明
```

#### 实验 4：主对比实验（全基线对比）

```
在 LongMemEval 上对比全部 12 个方法，重点验证 4 组：
  (1) TypeProg-Full vs Unified-Embedding → 整体价值
  (2) TypeProg-Full vs Token-Budget-Matched → 提升来自策略而非算力
  (3) TypeProg-Zero vs SGMem/SimpleMem → 零 LLM 调用的公平价值
  (4) TypeProg vs Adaptive-RAG → 记忆场景对 F2 处理的独特需求
  
  额外子表：各方法在 F1/F2/F3/single-hop 四个子集的分项准确率
  （这是审稿人会直接要求的表格，在设计时就预留）
```

#### 实验 5：跨域泛化与消融实验

**跨域泛化**：
```
设置：在 LongMemEval 上校准 StateConflictDetect 阈值，
     直接部署到 LoCoMo（temporal 子集）和 MSC（无重新校准）
验证：MQTP 分类精度跨数据集是否稳定？
     F2 Stale-Err 改善是否在 MSC 多会话场景中保持？
预期：跨域 F2 Accuracy 下降 < 8%（相对 in-domain）
```

**消融实验**：

| 消融变体 | 修改内容 | 目的 |
|---------|---------|------|
| w/o StateConflict | 不做冲突检测，直接 top-K 返回全部候选 | **F2 的 StateConflictDetect 独立贡献（核心消融）** |
| w/o LatestVersionSelect | 检测冲突但不按时序选最新版（随机选）| 时间戳排序 vs 随机的独立价值 |
| w/o TemporalAnchor | 直接语义搜索替代时序锚提取 | F1 的时序锚独立贡献 |
| w/o QueryDecompose | 单步搜索替代分解 | F3 的分解独立贡献 |
| Rule-only MQTP | 移除 MiniLM，只用规则 | 分类器对边缘查询的价值 |
| TypeProg-Zero vs Full | 规则分解 vs LLM 分解 | 1 次 LLM 调用的边际价值 |

### 5.5 公平性验证：Accuracy vs Token Budget 帕累托前沿

```
方法                        准确率(预期)   额外Token消耗   F2 Accuracy   F2 Stale-Err
────────────────────────────────────────────────────────────────────────────────
Unified-Embedding             ~0.42           0             ~0.40        ~55%
A-MEM                         ~0.52           0             ~0.45        ~50%
Mem0                          ~0.58           中等           ~0.50        ~45%
SGMem                         ~0.68           中等           ~0.55        ~40%
LightMem                      ~0.73           中等           ~0.58        ~38%
SimpleMem                     ~0.65(LoCoMo)   每次1次LLM     ~0.55        ~36% ← F2无改善
Token-Matched Baseline        ~0.55           900           ~0.52        ~40%
TypeProg-Zero（本方法）        ~0.62           0             ~0.68 ↑↑     ~20% ↓↓ 核心突破
TypeProg-Full（本方法）        ~0.70+          900           ~0.72 ↑↑     ~15% ↓↓ 核心突破

核心声明：TypeProg 在 F2 Accuracy 和 Stale-Err 上与所有基线形成质的跳跃
         TypeProg-Zero 在零额外 LLM 调用下不劣于 SimpleMem 总体准确率
```

---

## 模块 6：目标投稿方向（CCF B 级）

| 投稿目标 | CCF 级别 | 推荐理由 | 截稿参考 | 风险 |
|---------|---------|---------|---------|------|
| **EMNLP 2026** ⭐ **首选** | CCF B | ① NLP+Agent 方向极友好；② "实证发现+针对性方法"双贡献结构完美匹配 EMNLP；③ LongMemEval、LoCoMo 在 EMNLP 社区高认可度；④ Temporal Staleness 问题在 NLP 语境下有广泛相关工作可对话（knowledge update in dialogue）；⑤ 本工作 Related Work 定位清晰 | 通常 5–6 月提交 | SimpleMem 可能在 5 月前正式发表 |
| **COLING 2026** | CCF B | ① 接受对话系统、个性化 Agent；② 对实证分析型工作友好；③ 如 EMNLP 拒稿，根据审稿意见修改后投递 | 通常 9 月提交 | 竞争激烈 |
| **ECIR 2027** | CCF B | ① IR 专业顶会，StateConflictDetect 在检索层引入版本感知，对 IR 社区有独立价值；② 检索程序设计符合 IR 理论框架（可与 Structured Retrieval 文献对话）| 通常 10 月提交 | 投递时机较晚 |

**期刊备选**：Information Processing & Management（IPM, CCF B）— 无字数限制，允许充分的 F2 失效机制实证讨论。

### 投稿策略建议

```
主线路（推荐）：
  [立即] 完成实验 1（Go/No-Go 决策）
  → 结果支持（F2 Stale-Err > 35% 且 Kappa ≥ 0.5）
             → EMNLP 2026 提交（5-6月）
             → 接受：完成
             → 拒稿：根据审稿意见修改 → COLING 2026（9月）

备选（实验 1 中 SimpleMem 意外解决了 F2 问题时）：
  → 可能性极低，但若发生：将研究重点迁移至 F3 多跳聚合
    作为核心贡献（F3 QueryDecompose 在记忆场景的必要性同样
    文献中缺乏针对性研究）

另一备选（仅实证发现有意义而方法提升不显著时）：
  → 纯分析性论文（MQ-Taxonomy + 三类失效定量分析）
  → 投 EMNLP Findings 或 ACL 2026 Findings
```

---

## 模块 7：风险分析与改进建议

### 7.1 完整风险矩阵（v9 修订版）

| 编号 | 风险描述 | 发生概率 | 影响程度 | 应对方案 |
|-----|---------|---------|---------|---------|
| **R1** | 实验 1 跨域一致性低（Kappa < 0.5） | 中 | 🔴 致命 | 转换方向：为何现有系统跨数据集泛化性差（分析性论文，仍可发表）|
| **R2** | SimpleMem 在 F2 knowledge_update 子集已有大幅提升 | 低 | 🔴 严重 | SimpleMem 的召回加宽在 F2 场景下反而引入更多旧版本，但若实验证明不符预期：转向 F3 多跳聚合为核心 |
| **R3** | SimpleMem 5月前正式发表抢占"类型感知"标签 | **高** | 🟠 严重 | 已有明确区分：TypeProg = 类型专属执行程序 + 冲突消解，不同于深度自适应 |
| **R4** | SGMem/LightMem/SimpleMem 复现困难 | 中 | 🟡 中等 | 使用论文公开数字进行对比，说明实验条件差异；F2 子集结果在 LongMemEval 原文中有参考 |
| **R5** | TypeProg-Zero 在总体准确率上不如 SimpleMem | 中低 | 🟡 中等 | 聚焦 F2 Stale-Err 的质的突破；TypeProg-Full 追求总体性能竞争力 |
| **R6** | StateConflictDetect 阈值在 MSC 数据集上不稳定 | 中 | 🟡 中等 | 提供 5-shot 快速适配方案；在 Limitation 中诚实讨论；展示 TypeProg-Zero 的鲁棒性 |
| **R7** | LongMemEval knowledge_update 子集查询数量不足（< 50 条） | 低 | 🟡 中等 | LongMemEval 原文已报告此子集占比，预估约 80-100 条；若不足，补充 MSC 中的知识更新查询 |

### 7.2 具体改进建议

**改进 1（最紧迫）：立即纳入 SimpleMem 并精确区分 F2 维度**

SimpleMem 是最关键的竞争对手。必须在论文中专门用 1-2 段 Related Work 清晰区分：
- SimpleMem 解决"检索多少/用什么形式检索"
- TypeProg-Mem 解决"检索后如何消除版本冲突/执行什么操作序列"
- 两者正交互补（可提供 SimpleMem+TypeProg 叠加实验证据）

---

**改进 2：Related Work 中连接 Temporal Knowledge 文献**

StateConflictDetect 机制可与以下文献建立对话，提升学术定位：
- Temporal Knowledge Graph QA（MemoTime, TKGQA 系列）→ 区分点：个人对话记忆 vs. 公开事件知识图谱
- Knowledge Conflict in RAG（Longpre et al., 2021；Xie et al., 2023）→ 区分点：个人记忆的时间戳可用，对话历史结构化，与 RAG 的冲突来源不同
- Knowledge Update in LLMs（Yao et al., 2023 等）→ 区分点：检索层的版本消解，不改变模型参数

---

**改进 3：将实验 1（失败模式分析）独立为 Section 3**

参考 SeCom 等工作的"分析先行"结构：
- Section 1: Introduction（问题和贡献）
- Section 2: Related Work
- **Section 3: Temporal Staleness in Personal Memory — A Systematic Analysis**（实证贡献）
- Section 4: TypeProg-Mem 方法
- Section 5: Experiments

Section 3 的核心图表：各系统在四个子集（single-hop/F1/F2/F3）上的 Accuracy 分组柱状图，直观展示 F2 是所有系统的最大短板，且 SimpleMem 的 F2 改善几乎为零。

---

**改进 4：强化 StateConflictDetect 的技术深度**

StateConflictDetect 是本方案新的技术核心，需要比 v6 中对 RetNull 的描述更详细：
- 明确互斥性检测的具体实现（规则 + 轻量 cross-encoder）
- 提供实体共指检测的设计（对话记忆中的 "我" → "用户" 对齐）
- 讨论 LatestVersionSelect 的时间戳来源（记忆构建时自动记录对话轮次）

---

**改进 5：增加 "SimpleMem + TypeProg 叠加实验"**

此实验定位为本方案的正向加分项：
- 证明 StateConflictDetect 是正交机制，可增强任何现有系统
- 将 SimpleMem 与 TypeProg 的关系从"竞争"定义为"互补"
- 叠加实验结果若优于两者单独使用，是最有力的正交性证明

---

**改进 6：时间线（单人，14 周）**

| 阶段 | 时间 | 核心任务 | 关键决策点 |
|------|------|---------|----------|
| **立论验证** | 第 1–2 周 | 实验 1（F2 Stale Error Rate 基线统计）| **Go/No-Go**：F2 Stale-Err > 35% 且 Kappa ≥ 0.5 |
| **竞争分析** | 第 2–3 周 | 深入读 SimpleMem 代码，确认 F2 无冲突检测，准备区分材料 | — |
| **方法实现** | 第 3–6 周 | MQTP（规则引擎 + MiniLM）+ StateConflictDetect + 三类程序模板 | — |
| **初步验证** | 第 6–8 周 | 实验 2（F2 专项）+ 实验 3（vs SimpleMem 全类型）| — |
| **主实验** | 第 8–11 周 | 实验 4（完整基线）+ 实验 5（跨域泛化）| — |
| **消融分析** | 第 11–12 周 | 消融 + 案例分析 + 帕累托曲线 | — |
| **论文写作** | 第 12–14 周 | 初稿 + 图表 + 修改润色 | EMNLP 截稿节点 |

> **计算资源要求极低**：MQTP（规则+MiniLM）仅需 CPU，StateConflictDetect（规则+MiniLM cross-encoder）仅需 CPU，QueryDecompose（F3 类，约 20% 查询）需要 1 次 LLM API 调用。整个实验框架可在笔记本电脑上运行。

---

## 附录 A：论文叙事框架建议（v9 更新版）

```
Title: TypeProg-Mem: Type-Specific Retrieval Programs for Temporal Staleness
       in Agent Personal Memory Systems

Abstract:
  向量语义检索假设"语义相似意味着相关"，但在个人对话记忆场景中，
  这一假设在三类核心查询上系统性失效：(1) 知识更新查询中，旧信息
  与新信息语义高度相似，系统检索出过时版本（时效性失配问题）；
  (2) 时序推理查询中，语义相似性不携带事件时序结构；(3) 聚合查询
  中，单次 top-K 无法全集召回。通过对 LongMemEval、LoCoMo、MSC
  的跨域分析，我们首次定量证明：当前 SOTA 系统（含 SimpleMem）在
  knowledge_update 类查询上的 Stale Error Rate 超过 35%，且 SimpleMem
  的检索广度扩展对此无改善。为此，我们提出 TypeProg-Mem，一种训练
  自由的类型专属记忆检索程序框架，通过 (1) 轻量记忆查询类型解析器
  将查询映射为 F1/F2/F3 三类意图，(2) 检索程序编译器生成类型专属的
  DAG 执行计划，(3) StateConflictDetect 机制在检索层显式消解版本冲突。
  TypeProg 作为可插拔模块，在 F2 knowledge_update 子集上将 Stale Error
  Rate 从 >35% 降至 <20%，且 TypeProg-Zero 实现零额外 LLM 调用。

贡献声明：
  (1) 首个揭示个人记忆时效性失配问题的跨域类型化失败分析框架（实证贡献）
  (2) TypeProg 类型专属检索程序框架，以 DAG 形式实现类型感知的操作序列（方法贡献）
  (3) StateConflictDetect 机制 + Stale Error Rate 评估指标（机制+评估贡献）
```

---

## 附录 B：v6 → v9 的系统性变更对照表

| 维度 | v6 方案 | **v9 方案（本方案）** |
|------|---------|---------------------|
| **核心创新锚点** | F4 存在性判断/Null-Return | **F2 知识更新/StateConflictDetect** |
| **SimpleMem 处理** | ✅ 明确区分，加入基线，证明正交 | ✅ 保留，重点在 F2 子集的直接对比 |
| **基线完整性** | ✅ 完整 2026 年 SOTA 基线 | ✅ 完整保留 |
| **MQTP 解析器** | F1/F2/F3/F4 四类规则 | **F1/F2/F3 三类规则**（去除 F4 行）|
| **P_existence 程序** | RetNull + θ_null 阈值 | **删除** |
| **P_versioning 程序** | 基本描述 | **升级为核心贡献，详细实现** |
| **创新点 3** | RetNull + F4 Null-Recall | **StateConflictDetect + Stale Error Rate** |
| **实验 2（原版）** | RetNull 专项（F4-Null 样本依赖）| **删除** |
| **实验 2（新版）** | — | **F2 StateConflict 专项（knowledge_update 子集，GT 清晰）** |
| **实验 5（原版）** | θ_null 跨域稳定性 | **删除** |
| **评估指标** | F4 Null-Recall（核心）| **Stale Error Rate（核心）**|
| **帕累托曲线** | Accuracy vs Token vs F4 Null-Recall | **Accuracy vs Token vs F2 Accuracy + Stale-Err** |
| **风险 R2/R5** | SimpleMem 解决 F4 / θ_null 不稳定 | **删除（基础不成立）** |
| **ECIR 投稿理由** | RetNull = IR 未解问题 | **StateConflictDetect = 检索层版本感知（IR 视角同样新颖）** |
| **Abstract 核心论点** | F4 全行业盲区 | **三类语义检索系统性失效，F2 时效性失配为核心** |

---

*方案版本 v9（核心锚点修正版）。相对 v6 的核心变化：*  
*1. 将创新锚点从"F4 存在性判断（benchmark 无可靠 GT）"迁移至"F2 知识更新/时效性失配（LongMemEval knowledge_update 子集，GT 清晰）"*  
*2. 删除 RetNull 机制、F4 Null-Recall 指标、实验 2（RetNull 专项）、θ_null 校准实验*  
*3. 升级 StateConflictDetect 为核心技术贡献，提供详细实现方案*  
*4. 引入 Stale Error Rate 作为替代性新评估指标，可直接在现有 benchmark 上计算*  
*5. 保留 v6 中所有技术架构（MQTP/RPC/RPE/DAG/两版本）、完整基线表、风险矩阵结构、14 周时间线的整体框架*

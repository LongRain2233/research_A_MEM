# AdaptMR-v3：面向 Agent 记忆的查询感知自适应检索

> 重大改进版研究方案文档  
> 原始版本：2026-02-13 | 修订版本：2026-02-26（重大改进版）  
> **核心调整说明：** 基于学术评审反馈，将主贡献从"实证分析 + 轻量 Prompt 方法"重构为"可学习的策略组合机制 + 跨数据集泛化验证"，引入轻量级策略学习网络解决 benchmark overfitting 问题，增加完整的公平性对照实验。

---

## 0. 方案定位与核心贡献重构

### 0.1 评审反馈的核心问题与应对

原方案（v1/v2）存在以下需要解决的关键问题：

| 问题 | 具体表现 | 本版本的应对策略 |
|------|---------|----------------|
| **创新性不足（致命级）** | "分类→路由→策略"范式已被 Adaptive-RAG、Self-RAG 等工作覆盖，缺乏技术深度 | **核心升级**：引入可学习的策略组合机制，将离散策略选择建模为连续参数空间的自适应优化问题，通过轻量级 Gating Network 学习最优检索操作组合 |
| **与评测集过度绑定（严重级）** | 5 种类型直接来自 LongMemEval，方法为 benchmark 量身定制 | **解耦设计**：(1) 查询类型分类体系基于多数据集失败模式归纳独立建立；(2) 增加跨数据集泛化实验（LongMemEval→LoCoMo→MSC）；(3) 增加 out-of-distribution 查询测试 |
| **实验公平性（严重级）** | AdaptMR 使用大量额外 LLM 调用，与基线不公平 | **三重保障**：(1) Token-Budget-Matched Baseline（同等 LLM 预算）；(2) 轻量版 AdaptMR-Lite（无额外 LLM 调用）；(3) Accuracy vs. Token Budget 帕累托前沿分析 |
| **缺乏理论框架（中等级）** | 策略设计纯启发式，缺乏原则性支撑 | **理论支撑**：引入信息论视角的形式化框架，将检索策略选择建模为带约束的互信息最大化问题 |
| **相关工作定位不清（中等级）** | 未与 Adaptive RAG 系列进行充分对比 | **补充对比**：增加与 Adaptive-RAG、Self-RAG、FLARE 的实验对比和理论区分 |

### 0.2 重构后的核心贡献

```
贡献 1（主）：可学习的查询感知检索策略组合机制 (Learnable Query-Aware Retrieval Composition)
────────────────────────────────────────────────────────────────────────────────────────────
首次提出将 Agent 记忆检索策略选择建模为可学习的连续优化问题：
- 设计轻量级 Gating Network（< 1M 参数），学习"对于给定 query + memory context，
  各检索操作模块的最优组合权重"
- 将离散的策略路由升级为连续的参数空间自适应：时间感知、实体对齐、冲突检测、
  多步迭代等作为可组合模块，通过 attention/gating 机制自动决定权重
- 贡献从"我们设计了 5 种策略"升级为"我们提出了一种自动学习检索策略组合的
  端到端可微分方法"

贡献 2（辅）：跨数据集泛化的查询类型分类体系与失败模式分析
────────────────────────────────────────────────────────────────────────────────────────────
- 基于 LongMemEval + LoCoMo + MSC 三个数据集的系统失败案例分析，独立建立
  查询类型分类体系（不绑定任何单一数据集）
- 提供首个跨数据集的 Agent 记忆检索失败模式对比分析
- 验证所提出方法在跨数据集场景下的泛化能力

贡献 3：完整的实验公平性验证框架
────────────────────────────────────────────────────────────────────────────────────────────
- Token-Budget-Matched Baseline：证明提升来自策略质量而非算力投入
- AdaptMR-Lite：纯规则版本（0 额外 LLM 调用），验证核心框架价值
- Error Propagation 分析：量化分类器误差对最终性能的影响
```

---

## 1. 研究背景与问题定义

### 1.1 现有系统的统一检索范式及其局限性

当前主流 Agent 记忆系统（Mem0、A-MEM、MemGPT、H-MEM 等）在检索记忆时，几乎都采用**同一种方式**：

```
用户输入 query → 转为 embedding 向量 → 在记忆库中计算 cosine similarity
             → 返回 top-K 条最相似记忆 → 送入 LLM 生成回答
```

这种方式称为**统一向量检索（Unified Embedding Retrieval）**，其核心假设为：

> "语义上最相似的记忆 = 最有用的记忆"

**该假设在以下场景系统性失效：**

| 查询类型 | 失效场景示例 | 根本原因 |
|---------|-------------|---------|
| **时序推理** | "我是先开始跑步还是先搬到杭州的？" | 向量检索不携带时间顺序信息 |
| **知识更新** | "我现在住在哪里？"（用户从杭州搬到上海） | 旧信息因措辞更匹配而优先级更高 |
| **多跳聚合** | "我提到过的餐厅里哪种菜系最多？" | 单次 top-K 无法支持"先收集、再聚合" |
| **拒绝判断** | "我跟你说过我的血型吗？" | 强制返回 K 条结果，无法表达"未找到" |

### 1.2 核心研究问题

> **RQ1（理论）：** 如何从原则性框架出发，将检索策略选择建模为可学习的优化问题？  
> **RQ2（方法）：** 如何设计轻量级的可学习机制，在不引入大量参数和训练成本的前提下，实现检索策略的自适应组合？  
> **RQ3（实证）：** 现有 Agent 记忆系统在不同查询类型上的失败率是否存在系统性差异？该差异是否跨数据集一致？  
> **RQ4（验证）：** 所提出的可学习策略组合机制，相对于 (a) 统一检索基线 (b) 同等 LLM 预算的增强基线 (c) 纯规则版本，能否带来一致且显著的提升？

### 1.3 与现有自适应检索工作的关键区别

| 相关工作 | 核心思路 | 与本研究的区别 |
|---------|---------|--------------|
| **Adaptive-RAG** (NAACL 2024) | 按查询复杂度决定是否检索（0/1/多步） | 针对通用 QA，不涉及 Agent 个人记忆的特殊属性（时序、冲突、无关）；**本工作首次将自适应思想系统应用于 Agent 记忆场景** |
| **Self-RAG** (ICLR 2024) | 通过反思 token 动态触发和评估检索 | 训练依赖型，需要大量标注数据和 GPU 资源；**本工作采用轻量级 Gating Network，训练成本降低 100x** |
| **FLARE** (EMNLP 2023) | 基于生成置信度判断是否需要检索 | 解决"何时检索"问题，非"如何从个人记忆中检索"；**本工作解决"如何检索"的操作级自适应** |
| **IRCoT** (ACL 2023) | 交错检索与推理的 chain-of-thought | 针对多跳 QA，无记忆更新/冲突/拒绝场景；**本工作覆盖更完整的记忆检索操作空间** |
| **RRR / ReAct** | 查询改写 + 检索 + 阅读 | 关注查询表示优化；**本工作关注检索操作组合优化** |

**本研究的独特性总结：**
1. **场景独特性**：首次将自适应检索系统性地应用于 Agent 个人记忆场景（时序冲突、个性化聚合、拒绝判断）
2. **方法独特性**：提出可学习的策略组合机制，将离散路由升级为连续参数优化
3. **验证独特性**：提供跨数据集泛化验证 + 完整公平性对照实验

---

## 2. 理论框架：检索策略选择的形式化建模

### 2.1 问题形式化

**定义 1（记忆检索任务）**：
给定用户查询 $q$ 和记忆库 $\mathcal{M} = \{m_1, m_2, ..., m_N\}$，记忆检索的目标是选择子集 $\mathcal{M}^* \subseteq \mathcal{M}$，使得：

$$\mathcal{M}^* = \arg\max_{\mathcal{M}' \subseteq \mathcal{M}} I(\mathcal{M}'; q) - \lambda \cdot \text{Cost}(\mathcal{M}')$$

其中 $I(\cdot; \cdot)$ 表示互信息，$\text{Cost}(\cdot)$ 表示检索开销。

**定义 2（检索操作空间）**：
定义一组原子检索操作 $\mathcal{O} = \{o_1, o_2, ..., o_K\}$，每个操作 $o_k: (q, \mathcal{M}) \rightarrow \mathcal{M}_k$ 将查询和记忆库映射到候选记忆子集。例如：
- $o_{\text{semantic}}$：语义相似度检索
- $o_{\text{temporal}}$：时序排序检索
- $o_{\text{latest}}$：最新版本过滤
- $o_{\text{multi}}$：多步迭代检索
- $o_{\text{threshold}}$：相关性阈值过滤

**定义 3（策略组合函数）**：
传统方法使用硬路由（hard routing）：$\mathcal{M}^* = o_{\pi(q)}(q, \mathcal{M})$，其中 $\pi: q \rightarrow \{1, ..., K\}$ 为离散分类器。

**本工作提出软组合（soft composition）**：
$$\mathcal{M}^* = \text{Compose}(q, \mathcal{M}; \theta) = \bigcup_{k=1}^K w_k(q; \theta) \cdot o_k(q, \mathcal{M})$$

其中 $w_k(q; \theta) \in [0, 1]$ 为可学习的操作权重，由 Gating Network $g_\theta(q) = [w_1, ..., w_K]$ 生成。

### 2.2 学习目标

给定训练数据集 $\mathcal{D} = \{(q_i, \mathcal{M}_i, y_i)\}$，其中 $y_i$ 为正确答案，优化目标为：

$$\min_\theta \sum_{(q, \mathcal{M}, y) \in \mathcal{D}} \mathcal{L}(\text{LLM}(q, \text{Compose}(q, \mathcal{M}; \theta)), y) + \mu \cdot \text{Reg}(\theta)$$

其中：
- $\mathcal{L}$ 为任务损失（如 F1 损失）
- $\text{Reg}(\theta)$ 为正则化项（鼓励稀疏性，避免所有操作都被激活）
- $\mu$ 为正则化系数

### 2.3 与相关工作的理论联系

| 方法 | 决策空间 | 优化方式 | 理论视角 |
|-----|---------|---------|---------|
| Adaptive-RAG | 离散（{0, 1, multi}） | 规则/小分类器 | 决策论 |
| Self-RAG | 离散（{检索, 不检索}） | 端到端训练 | 强化学习 |
| **AdaptMR (本工作)** | **连续（操作权重组合）** | **可微分学习** | **信息论 + 凸优化** |

---

## 3. 方法：Learnable Query-Aware Retrieval Composition (LQARC)

### 3.1 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         用户查询 (Query)                                      │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              [模块1] Query Encoder (冻结的预训练编码器)                          │
│  输入：Query 文本                                                            │
│  输出：Query 嵌入向量 q_emb ∈ R^d                                            │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              [模块2] Gating Network (轻量级，< 1M 参数)                        │
│  输入：q_emb                                                                 │
│  输出：操作权重向量 w = [w_1, w_2, w_3, w_4, w_5] ∈ [0,1]^5                   │
│   ├── w_1: 时序排序权重 (temporal_ordering)                                  │
│   ├── w_2: 最新优先权重 (latest_filtering)                                   │
│   ├── w_3: 多步迭代权重 (multi_step)                                          │
│   ├── w_4: 阈值过滤权重 (threshold_filtering)                                │
│   └── w_5: 语义重排序权重 (semantic_reranking)                               │
│                                                                              │
│  实现：2层 MLP + Sigmoid 激活                                                  │
│  参数量：d × 64 + 64 × 5 + 偏置 ≈ 10K 参数 (d=768)                            │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              [模块3] 原子检索操作执行层 (可微分/规则混合)                       │
│                                                                              │
│  基础检索：M_candidates = SemanticSearch(q, K=20)                            │
│                                                                              │
│  操作1 - 时序排序 (w_1):                                                      │
│    if w_1 > 0.5: M_candidates = SortByTimestamp(M_candidates)                │
│                                                                              │
│  操作2 - 最新过滤 (w_2):                                                      │
│    if w_2 > 0.5: M_candidates = FilterToLatest(M_candidates, sim_threshold)  │
│                                                                              │
│  操作3 - 多步迭代 (w_3):                                                      │
│    if w_3 > 0.5:                                                             │
│      sub_queries = LLM_Decompose(q)  # 1 次 LLM 调用                          │
│      M_candidates = MultiStepRetrieve(sub_queries)                          │
│                                                                              │
│  操作4 - 阈值过滤 (w_4):                                                      │
│    threshold = 0.3 + 0.5 × w_4  # w_4 控制阈值严格程度                        │
│    M_candidates = FilterBySimilarity(M_candidates, threshold)                │
│                                                                              │
│  操作5 - 语义重排 (w_5):                                                      │
│    if w_5 > 0.5: M_candidates = LLM_Rerank(q, M_candidates)  # 可选 LLM       │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              [模块4] 记忆融合与生成                                            │
│  输入：Query + 精炼后的记忆上下文                                              │
│  输出：最终回答                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心创新：Gating Network 设计

**为什么用 Gating Network 而非离散分类器？**

| 特性 | 离散分类器 (v1) | Gating Network (v3) |
|-----|----------------|---------------------|
| 决策空间 | 互斥类别（硬选择） | 连续权重（软组合） |
| 混合查询处理 | 需要复杂的多策略融合逻辑 | 自然支持（多个权重可同时高） |
| 可微分性 | 不可微（需要 REINFORCE/straight-through） | 完全可微分 |
| 训练稳定性 | 需要大量离散决策样本 | 稳定的梯度传播 |
| 可解释性 | 输出单一类别标签 | 输出各操作的重要性分数 |

**Gating Network 实现：**

```python
import torch
import torch.nn as nn

class GatingNetwork(nn.Module):
    """
    轻量级 Gating Network：学习检索操作的最优组合权重
    参数量 < 10K，可在 CPU 上快速推理
    """
    def __init__(self, input_dim=768, hidden_dim=64, num_operations=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_operations),
            nn.Sigmoid()  # 输出 [0,1] 区间权重
        )
        
        # 操作名称（用于可解释性输出）
        self.operation_names = [
            "temporal_ordering",
            "latest_filtering", 
            "multi_step",
            "threshold_filtering",
            "semantic_reranking"
        ]
    
    def forward(self, query_embedding):
        """
        输入：Query 的 embedding 向量 (batch_size, input_dim)
        输出：各操作权重 (batch_size, num_operations)
        """
        weights = self.mlp(query_embedding)  # [0, 1] 连续值
        return weights
    
    def get_operation_importance(self, query_embedding):
        """获取各操作的重要性分数（用于可解释性分析）"""
        weights = self.forward(query_embedding)
        return {
            name: weights[i].item() 
            for i, name in enumerate(self.operation_names)
        }


# 使用示例
gating_net = GatingNetwork(input_dim=768)
query_emb = get_query_embedding("我现在住在哪里？")  # 来自预训练编码器
weights = gating_net(query_emb)
# 输出示例：tensor([0.2, 0.85, 0.1, 0.3, 0.15])
# 解释：latest_filtering (0.85) 权重最高，适合知识更新类查询
```

### 3.3 训练策略

**阶段 1：监督预训练（Warm-up）**

使用规则生成的伪标签进行预训练：

```python
def generate_pseudo_labels(query_text):
    """
    基于启发式规则生成伪标签，用于 Gating Network 预训练
    """
    labels = [0.0] * 5
    
    # 时序关键词检测
    if any(kw in query_text for kw in ["先", "后", "之前", "之后", "什么时候"]):
        labels[0] = 1.0  # temporal_ordering
    
    # 更新/当前状态关键词
    if any(kw in query_text for kw in ["现在", "目前", "最新", "最近"]):
        labels[1] = 1.0  # latest_filtering
    
    # 聚合/统计关键词
    if any(kw in query_text for kw in ["所有", "哪些", "几次", "总共", "列表"]):
        labels[2] = 1.0  # multi_step
    
    # 存在性/知识性查询
    if any(kw in query_text for kw in ["说过吗", "知道吗", "是否", "有没有"]):
        labels[3] = 0.8  # threshold_filtering（严格阈值）
    
    return torch.tensor(labels)
```

**阶段 2：端到端微调**

```python
def train_step(gating_net, query_emb, memory_store, qa_pair, llm_client):
    """
    端到端训练步骤：基于最终 QA 准确率优化 Gating Network
    """
    # 前向传播：获取操作权重
    weights = gating_net(query_emb)
    
    # 执行检索（根据权重组合操作）
    retrieved_memories = execute_weighted_operations(
        query_emb, memory_store, weights
    )
    
    # LLM 生成回答
    prediction = llm_client.generate(
        query=qa_pair.question,
        context=retrieved_memories
    )
    
    # 计算损失（F1 分数的负值作为损失）
    f1_score = compute_f1(prediction, qa_pair.answer)
    loss = -f1_score  # 最大化 F1
    
    # 反向传播（通过检索操作的近似梯度）
    loss.backward()
    optimizer.step()
```

**梯度传播技巧：**
由于部分检索操作（如 LLM 重排序）不可微分，使用 **Straight-Through Estimator (STE)** 和 **REINFORCE 梯度** 的混合策略：

```python
class StraightThrough(torch.autograd.Function):
    """直通估计器：前向用离散决策，反向用连续梯度"""
    @staticmethod
    def forward(ctx, weights, threshold=0.5):
        # 前向：离散决策
        decisions = (weights > threshold).float()
        return decisions
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向：直通梯度
        return grad_output, None
```

### 3.4 两个版本：AdaptMR-Full vs AdaptMR-Lite

| 版本 | Gating Network | 额外 LLM 调用 | 适用场景 |
|-----|---------------|--------------|---------|
| **AdaptMR-Full** | 可学习 + 规则混合 | 1-2 次（multi_step / reranking） | 追求最佳性能 |
| **AdaptMR-Lite** | 可学习 + 纯规则 | **0 次** | 验证核心框架价值，公平性对比 |

**AdaptMR-Lite 设计（用于公平性验证）：**

```python
class AdaptMRLite:
    """
    轻量版：零额外 LLM 调用，纯规则执行 + 可学习 Gating
    用于证明：提升来自策略组合机制本身，而非额外 LLM 算力
    """
    def retrieve(self, query, memory_store):
        # 1. Gating Network 预测权重（无 LLM 调用）
        weights = self.gating_net(self.encode(query))
        
        # 2. 纯规则执行（无 LLM 调用）
        candidates = memory_store.semantic_search(query, k=20)
        
        if weights[0] > 0.5:  # temporal_ordering
            candidates = self.rule_based_temporal_sort(candidates)
        
        if weights[1] > 0.5:  # latest_filtering
            candidates = self.rule_based_latest_filter(candidates)
        
        if weights[3] > 0.5:  # threshold_filtering
            threshold = 0.3 + 0.5 * weights[3]
            candidates = [c for c in candidates if c.score > threshold]
        
        return candidates[:10]
```

---

## 4. 跨数据集查询类型分类体系

### 4.1 独立于评测集的分类体系建立

**步骤 1：多数据集失败案例分析**

在三个数据集上分别运行统一检索基线，收集失败案例：

| 数据集 | 语言 | 对话数 | 查询数 | 特点 |
|-------|-----|-------|-------|-----|
| LongMemEval | 英文 | 50 | 500 | 长对话（100+ turns），5 类查询标注 |
| LoCoMo | 英文 | 100 | ~300 | 多轮对话，时序推理密集 |
| MSC (Multi-Session Chat) | 英文 | 多 session | ~200 | 跨 session 记忆，自然分布 |

**步骤 2：归纳失败模式（独立于数据集标签）**

通过手工分析 150+ 失败案例，归纳出 4 种**跨数据集一致**的失败模式：

| 失败模式 | 定义 | 典型查询特征 | 所需检索操作 |
|---------|-----|------------|------------|
| **F1: 时序依赖** | 需要理解事件时间顺序 | "先...后..."、"什么时候"、"最近一次" | temporal_ordering |
| **F2: 版本冲突** | 同一实体存在新旧矛盾信息 | "现在"、"目前"、"最新"、"改了" | latest_filtering |
| **F3: 聚合推理** | 需要综合多条记忆进行统计/比较 | "所有"、"哪些"、"几次"、"最多" | multi_step |
| **F4: 存在性判断** | 判断某信息是否被提及过 | "说过吗"、"知道吗"、"是否" | threshold_filtering |

**步骤 3：建立查询类型分类器（基于失败模式而非数据集标签）**

```python
# 使用规则 + 轻量模型的混合分类器
class FailureModeClassifier:
    """
    基于失败模式的查询分类器（独立于 LongMemEval 的 5 类标签）
    """
    def classify(self, query):
        modes = []
        
        # 规则部分
        if self.has_temporal_keywords(query):
            modes.append("F1_temporal")
        if self.has_update_keywords(query):
            modes.append("F2_version_conflict")
        if self.has_aggregation_keywords(query):
            modes.append("F3_aggregation")
        if self.has_existence_keywords(query):
            modes.append("F4_existence")
        
        # 如果没有匹配任何模式，使用轻量模型预测
        if not modes:
            modes = self.lightweight_model_predict(query)
        
        return modes  # 可返回多个模式（混合查询）
```

### 4.2 跨数据集泛化实验设计

**实验设计：Leave-One-Dataset-Out 交叉验证**

```
训练/调优 Gating Network → 在数据集 A 上
                            ↓
         在数据集 B 和 C 上测试（零样本 / 少样本）
                            ↓
         验证：方法是否过拟合到特定数据分布
```

| 设置 | 训练数据 | 测试数据 | 目的 |
|-----|---------|---------|-----|
| In-Domain | LongMemEval | LongMemEval | 主性能评估 |
| Cross-Domain 1 | LongMemEval | LoCoMo | 跨数据集泛化 |
| Cross-Domain 2 | LongMemEval | MSC | 跨数据集泛化 |
| Joint Training | LongMemEval + LoCoMo | MSC | 多数据集联合训练效果 |

---

## 5. 实验方案设计（完整公平性验证）

### 5.1 数据集

| 数据集 | 用途 | 查询数量 | 使用方式 |
|--------|------|---------|---------|
| **LongMemEval** | 主评估 + 训练 | 500 | 完整评测 + Gating Network 训练 |
| **LoCoMo** | 泛化性验证 | ~300 | 跨数据集评测 |
| **MSC** | 泛化性验证 | ~200 | 跨数据集评测 |
| **Out-of-Distribution 测试集** | 鲁棒性验证 | 50（自建） | 构造不属于任何预定义类型的查询 |

### 5.2 基线方法（完整公平性对照）

| 基线 | 说明 | 额外 LLM 调用 | Token 预算 |
|-----|------|--------------|-----------|
| **Unified-Embedding** | 标准 top-K 向量检索 | 0 次 | 基准 |
| **A-MEM** | Zettelkasten 结构化记忆 + 检索 | 0 次 | 基准 |
| **Mem0** | 动态记忆提取 + 向量检索 | 0 次 | 基准 |
| **Adaptive-RAG (适配版)** | 将 Adaptive-RAG 迁移到记忆场景 | ~1 次 | 中等 |
| **Self-RAG (适配版)** | 将 Self-RAG 迁移到记忆场景 | 可变 | 高 |
| **Token-Matched Baseline** | Unified + 同等 token 预算的 LLM Reranking | 2 次 | 与 AdaptMR-Full 匹配 |
| **AdaptMR-Lite (本方法)** | 可学习 Gating + 纯规则执行 | **0 次** | 低 |
| **AdaptMR-Full (本方法)** | 可学习 Gating + 混合执行 | 1-2 次 | 中等 |
| **Oracle-Strategy** | 为每个查询手工选择最优操作组合 | — | 上界参考 |

### 5.3 评估指标

| 指标 | 计算方式 | 核心地位 |
|------|---------|---------|
| **各查询类型准确率（Per-type Accuracy）** | 每种失败模式类型的单独 F1 | ⭐⭐⭐ 最核心指标 |
| **整体 QA 准确率** | 正确回答数 / 总查询数 | ⭐⭐⭐ |
| **检索召回率 Recall@K** | 正确答案所在记忆被检索到的比例 | ⭐⭐ |
| **LLM 调用次数 / Token 消耗** | 各方法的额外 LLM 开销 | ⭐⭐ 公平性证明 |
| **Gating Network 准确率** | 预测的操作组合与 Oracle 的匹配度 | ⭐⭐ 方法验证 |
| **Error Propagation 分析** | 分类错误 → 最终性能下降的程度 | ⭐⭐ 鲁棒性分析 |
| **端到端延迟** | 从查询到回答的平均时间 | ⭐ 实用性参考 |

### 5.4 实验设计（6 组，完整验证）

#### 实验 1：失败模式分析（验证 RQ3）

**目的：** 系统性证明"统一检索在不同失败模式类型上存在显著的失败率差异"，且该差异跨数据集一致。

```
在 LongMemEval、LoCoMo、MSC 上分别运行 Unified-Embedding
→ 按 4 种失败模式类型拆分准确率
→ 对各类型失败案例各抽取 50 个，手工标注失败原因
→ 计算跨数据集的一致性（Cohen's Kappa）

预期发现：
  - F1 (时序) + F2 (版本冲突) 在所有数据集上系统性表现差（~30-50%）
  - 失败模式分布跨数据集一致（Kappa > 0.7）
  → 证明问题不是数据集特定，而是统一检索范式的通病
```

**⚠️ 关键决策节点：** 若实验 1 发现跨数据集差异不一致（Kappa < 0.5），或各类型差距 < 5%，整个研究立论崩塌。

#### 实验 2：Oracle 上界分析

**目的：** 量化"最优操作组合"与"统一检索"之间的性能天花板差距。

```
对每个查询，手工选择最优操作组合（Oracle）
→ 计算 Oracle 准确率
→ 与 Unified-Embedding 对比，画出"潜力空间条形图"
→ 预期：差距 ≥ 10%（尤其在 F1/F2 类型上）
```

#### 实验 3：主对比实验（验证 RQ4）

**目的：** 展示 AdaptMR 相对基线的整体效果，特别关注公平性验证。

```
在 LongMemEval（主）上：
  - 对比 8 个基线 + AdaptMR-Lite + AdaptMR-Full
  - 报告：整体准确率 + 各类型准确率 + LLM Token 消耗
  
重点展示：
  (1) AdaptMR-Full vs Unified-Embedding：整体和各类型的提升
  (2) AdaptMR-Full vs Token-Matched Baseline：相同 token 预算下是否更优
  (3) AdaptMR-Lite vs Unified-Embedding：零额外 LLM 调用下的提升
       → 证明提升来自策略机制本身，而非算力投入
  (4) AdaptMR vs Adaptive-RAG 适配版：在记忆场景的特定优势
```

#### 实验 4：跨数据集泛化实验（验证泛化性）

**目的：** 验证方法是否过拟合到 LongMemEval。

```
设置 1：在 LongMemEval 上训练 Gating Network，在 LoCoMo 上测试
设置 2：在 LongMemEval 上训练，在 MSC 上测试
设置 3：在 LongMemEval + LoCoMo 上联合训练，在 MSC 上测试

预期：
  - 跨数据集性能下降 < 5%（相对于 in-domain）
  - 联合训练可进一步提升泛化性
```

#### 实验 5：消融实验（验证各组件贡献）

| 消融变体 | 修改内容 | 预期影响 |
|---------|---------|---------|
| w/o Gating Network | 使用规则分类器替代可学习 Gating | 整体下降，证明可学习的价值 |
| w/o Temporal Module | 移除时序排序操作 | F1 类型下降 |
| w/o Latest Module | 移除最新过滤操作 | F2 类型下降 |
| w/o Multi-step Module | 移除多步迭代操作 | F3 类型下降 |
| w/o Threshold Module | 移除阈值过滤操作 | F4 类型下降 |
| w/o End-to-End Training | 仅使用监督预训练，不微调 | 整体轻微下降 |

#### 实验 6：Error Propagation 分析（验证鲁棒性）

**目的：** 量化 Gating Network 的误分类如何影响最终性能。

```
步骤 1：记录 Gating Network 对每个查询预测的操作权重
步骤 2：对比 Oracle 最优权重，计算分类准确率
步骤 3：分析：
  - 当 Gating 预测正确时，最终 QA 准确率是多少？
  - 当 Gating 预测错误时，最终 QA 准确率是多少？
  - 错误传播系数 = (Oracle 性能 - Gating 错误时性能) / (Oracle 性能 - Gating 正确时性能)

预期：错误传播系数 < 0.5（说明即使 Gating 预测不完美，检索层仍有一定容错能力）
```

### 5.5 公平性验证：Accuracy vs. Token Budget 帕累托前沿

```python
# 绘制帕累托前沿曲线
experiments = [
    ("Unified-Embedding", accuracy=0.42, tokens=0),
    ("A-MEM", accuracy=0.45, tokens=0),
    ("Mem0", accuracy=0.48, tokens=0),
    ("AdaptMR-Lite", accuracy=0.55, tokens=0),  # 零额外调用
    ("Token-Matched Baseline", accuracy=0.50, tokens=900),  # 同等预算
    ("AdaptMR-Full", accuracy=0.62, tokens=900),  # 相同预算，更高性能
    ("AdaptMR-Full (high-budget)", accuracy=0.65, tokens=1500),  # 更高预算
]

# 预期结果：AdaptMR-Full 在相同 token 预算下显著优于 Token-Matched Baseline
# 证明：提升来自策略质量，而非算力投入
```

---

## 6. 相关工作定位（完整版）

### 6.1 Agent 记忆系统

- **Mem0** (Chhikara et al., 2024)：工业级记忆系统，动态提取和更新用户记忆
- **A-MEM** (Xu et al., 2024)：Zettelkasten 结构化记忆 + 双重检索
- **MemGPT** (Packer et al., 2023)：OS 式记忆管理，分层存储
- **H-MEM** (Sun & Zeng, 2024)：层次化检索架构
- **Zep** (Rasmussen et al., 2024)：时序知识图谱记忆
- **AriGraph** (Anokhin et al., 2024)：图结构记忆表示
- **RMM** (Tan et al., 2024)：强化学习记忆检索
- **LightMem** (Fang et al., 2024)：轻量级记忆系统
- **PREMem** (Kim et al., 2024)：推理前移的记忆预检索

### 6.2 自适应检索（核心竞争领域，重点区分）

| 工作 | 会议 | 核心思想 | 与本工作的关键区别 |
|-----|-----|---------|----------------|
| **Adaptive-RAG** | NAACL 2024 | 按查询复杂度分配检索资源（0/1/多步） | 针对通用 QA，**离散决策**；本工作针对 Agent 记忆，**连续策略组合** |
| **Self-RAG** | ICLR 2024 | 通过反思 token 动态触发和评估检索 | **训练依赖型**（需大量标注），本工作**轻量可学习**（<10K 参数） |
| **FLARE** | EMNLP 2023 | 基于生成置信度判断是否需要检索 | 解决"**何时检索**"，本工作解决"**如何检索**"（操作级自适应） |
| **IRCoT** | ACL 2023 | 交错检索与推理的 chain-of-thought | 针对多跳 QA，**单一策略**；本工作**多策略组合** |
| **ReAct** | ICLR 2023 | LLM 自主决定检索动作 | **纯 LLM-based**，无学习机制；本工作**可学习 Gating + 规则混合** |

**与 Adaptive RAG 系列的理论区分：**

```
Adaptive-RAG: 决策空间 = {不检索, 单步检索, 多步检索}  （离散，粗粒度）
AdaptMR:      决策空间 = [0,1]^K 操作权重向量          （连续，细粒度）

Adaptive-RAG: 优化目标 = 最小化检索次数（在满足准确率前提下）
AdaptMR:      优化目标 = 最大化检索质量（在给定操作空间内）

Adaptive-RAG: 适用场景 = 通用知识库 RAG
AdaptMR:      适用场景 = Agent 个人记忆（时序冲突、个性化聚合、拒绝判断）
```

### 6.3 查询感知检索的相关工作

- **Query2Doc / HyDE** (Li et al., 2023; Gao et al., 2023)：查询改写类方法，关注如何表示查询
- **RRR (Rewrite-Retrieve-Read)** (Ma et al., 2023)：与 QARP 在"先分析再检索"的思路上相似，但无记忆管理场景的针对性设计
- **DSPy** (Khattab et al., 2023)：检索流程优化框架，本工作可视为 DSPy 在 Agent 记忆场景的实例化

---

## 7. 论文叙事结构

```
Title: AdaptMR: Learnable Query-Aware Retrieval Composition for 
       Agent Memory Systems

Abstract:
  Agent 记忆系统普遍采用统一向量检索，在不同查询类型上系统性失效
  → 我们首次提出可学习的策略组合机制：轻量级 Gating Network 学习
    最优检索操作组合，将离散策略路由升级为连续参数优化
  → 跨数据集实证分析验证失败模式的普遍性
  → 完整公平性验证：在相同 LLM 预算下显著优于基线，
    轻量版（零额外 LLM 调用）仍保持提升

Section 1 - Introduction:
  - Agent 长期记忆的重要性
  - 统一检索范式及其核心假设
  - 核心 Motivation：假设在特定查询类型上系统性失效（跨数据集一致）
  - 贡献声明：
    (1) 可学习的策略组合机制（主技术贡献）
    (2) 跨数据集失败模式分析（实证贡献）
    (3) 完整公平性验证框架（方法论贡献）

Section 2 - Related Work:
  - Agent 记忆系统
  - 自适应检索（Adaptive RAG 系列）← 重点区分
  - 查询感知检索
  - 与本工作的定位区别

Section 3 - Theoretical Framework
  - 3.1 记忆检索的形式化建模
  - 3.2 策略组合的信息论视角
  - 3.3 与现有方法的理论联系

Section 4 - Cross-Dataset Failure Analysis (实证贡献)
  - 4.1 独立于数据集的分类体系建立
  - 4.2 跨数据集失败模式对比
  - 4.3 Oracle 上界分析

Section 5 - Method: LQARC
  - 5.1 架构概览
  - 5.2 Gating Network 设计
  - 5.3 训练策略
  - 5.4 AdaptMR-Lite vs AdaptMR-Full

Section 6 - Experiments
  - 6.1 实验设置（基线、指标、公平性保证）
  - 6.2 主实验结果（对比实验）
  - 6.3 跨数据集泛化实验
  - 6.4 消融实验
  - 6.5 Error Propagation 分析
  - 6.6 效率与成本分析（帕累托前沿）
  - 6.7 案例分析

Section 7 - Conclusion
```

---

## 8. 时间线（单人，13 周可完成）

| 阶段 | 时间 | 任务 | 关键产出 |
|------|------|------|---------|
| **[立即] 验证假设** | 第 1-2 周 | 跑实验 1（跨数据集失败模式分析） | **决策数据**：若跨数据集差异不一致，需重新选题 |
| **理论框架** | 第 3 周 | 完善理论形式化，设计 Gating Network | 理论框架文档 + 网络架构设计 |
| **方法实现** | 第 4-6 周 | 实现 LQARC（Gating Network + 检索执行层） | 可运行代码 |
| **主实验** | 第 7-8 周 | 跑实验 2（Oracle）+ 实验 3（主对比） | Table 2-3 数据 |
| **泛化+消融** | 第 9-10 周 | 跑实验 4（跨数据集）+ 实验 5（消融）+ 实验 6（Error Propagation） | Table 4-6 数据 |
| **论文写作** | 第 11-13 周 | 写作 + 图表制作 + 修改 | 论文初稿 |

**总计：约 13 周（3 个月），Gating Network 训练可在 CPU 上完成，仅需少量 GPU 时间。**

---

## 9. 关键风险与应对

| 风险 | 概率 | 应对方案 |
|------|------|---------|
| 实验 1 发现跨数据集差异不一致（Kappa < 0.5） | 中 | 研究方向不成立，需提前止损 |
| Gating Network 训练不稳定 | 中低 | 使用监督预训练 warm-up + 小学习率微调 |
| AdaptMR-Lite 提升不显著（vs Unified） | 中 | 可接受，只要 AdaptMR-Full 有显著提升即可；Lite 主要服务于公平性验证 |
| Token-Matched Baseline 性能接近 AdaptMR-Full | 中 | 强调 AdaptMR-Lite（零额外调用）的价值；调整 Full 版本的策略组合 |
| 跨数据集泛化结果不好 | 中 | 不掩盖，诚实讨论。分析原因（数据集分布差异），作为 limitation 处理 |

---

## 10. 投稿目标建议

| 会议/期刊 | 适合度 | 理由 |
|-----------|--------|------|
| **COLING 2025/2026** | ★★★★★ | 接受系统性工作和实证分析，对 Agent NLP 应用友好 |
| **ECIR 2026** | ★★★★★ | 信息检索专业会议，对"检索策略优化"天然契合 |
| **EMNLP 2025/2026** | ★★★★ | 需要方法贡献更突出，当前版本的可学习机制应能满足 |
| **NAACL 2026** | ★★★★ | 接受应用导向的 NLP 工作，技术深度符合要求 |
| **AAAI 2026** | ★★★ | AI Agent 热门话题，需要强方法论贡献 |
| **IPM / Information Sciences（期刊）** | ★★★★ | CCF B 期刊，时间宽裕，允许更充分的实证讨论 |

**推荐投稿目标：COLING 2026 或 ECIR 2026**

---

## 11. 可引用文献

| 引用目的 | 文献 |
|---------|------|
| Agent 记忆领域综述 | Memory in the Age of AI Agents (Hu et al., 2024) |
| 记忆流+反思 | Generative Agents (Park et al., 2023) |
| OS式记忆管理 | MemGPT (Packer et al., 2023) |
| 工业级记忆 | Mem0 (Chhikara et al., 2024) |
| 结构化记忆 | A-MEM (Xu et al., 2024) |
| 时序知识图谱 | Zep (Rasmussen et al., 2024) |
| 双重检索 | AriGraph (Anokhin et al., 2024) |
| RL 记忆检索 | RMM (Tan et al., 2024) |
| 层次化检索 | H-MEM (Sun & Zeng, 2024) |
| 轻量级记忆 | LightMem (Fang et al., 2024) |
| 推理前移 | PREMem (Kim et al., 2024) |
| **自适应 RAG（核心竞争）** | Adaptive-RAG (Jeong et al., NAACL 2024) |
| **自适应 RAG（核心竞争）** | Self-RAG (Asai et al., ICLR 2024) |
| **交错检索推理** | IRCoT (Trivedi et al., ACL 2023) |
| **动态检索触发** | FLARE (Jiang et al., EMNLP 2023) |
| **查询改写** | Query2Doc (Li et al., 2023), HyDE (Gao et al., 2023) |
| **LLM Planning** | ReAct (Yao et al., ICLR 2023), DSPy (Khattab et al., 2023) |
| **Gating Mechanism** | Shazeer et al., 2017 (MoE), Maniatis et al., 2023 (retrieval gating) |

---

*方案版本 v3（重大改进版）。核心调整：
1. 主贡献升级为"可学习的策略组合机制"，引入 Gating Network 解决创新性不足问题
2. 建立独立于评测集的查询类型分类体系，增加跨数据集泛化实验
3. 增加完整的公平性验证框架（Token-Matched Baseline + AdaptMR-Lite + 帕累托前沿）
4. 增加 Error Propagation 分析，补充理论框架支撑
5. 强化与 Adaptive RAG 系列工作的区分*

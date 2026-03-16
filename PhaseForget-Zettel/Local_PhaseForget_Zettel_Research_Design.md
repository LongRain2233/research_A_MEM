# Local PhaseForget-Zettel: 局部阈值驱动与效用感知的Zettelkasten记忆重整化机制

**Local PhaseForget-Zettel: Threshold-Driven Local Renormalization and Utility-Aware Forgetting in Zettelkasten Memory**

---

## 1. 课题元数据

| 字段 | 内容 |
|------|------|
| **课题名称** | **Local PhaseForget-Zettel: 局部阈值驱动与效用感知的Zettelkasten记忆重整化机制**<br>(Local PhaseForget-Zettel: Threshold-Driven Local Renormalization and Utility-Aware Forgetting in Zettelkasten Memory) |
| **研究类型** | 融合设计型 (Fusion Design) |
| **方法来源** | 1. **A-Mem** (Xu 等, 2025): §3.1-§3.3 (原子笔记与链接生成) <br>2. **RGMem** (Tian 等, 2026): §3.1-§3.3 (多尺度理论与节点级抽象算子 $\mathcal{R}_{K2}$)<br>3. **Agent KB** (Tang 等, 2025): §3.2 (自适应效用驱逐策略) |

---

## 2. 研究背景（严格溯源）

### 2.1 当前方法局限性
当前先进的显式记忆系统（如 A-Mem）虽实现了图数据库存储和Zettelkasten式链接，但其交互历史无界增长（"interaction histories grow without bound"），而模型的推理受限于有限上下文窗口（引自 **RGMem §1 Introduction**）。A-Mem自身缺乏动态遗忘机制，只依靠静态相似度操作，随着时间推移导致计算资源的线性攀升（引自 **A-Mem §2.1**）。

### 2.2 研究缺口
长对话个性化任务面临"稳定性-可塑性困境（stability–plasticity dilemma）"。现有系统难以在适应新证据的同时保持宏观用户特征的稳定，缺乏原则性的抽象和矛盾管理机制，导致噪声和不一致性随时间累积（引自 **RGMem §2.2 Explicit Memory and Retrieval-Augmented Generation**）。

### 2.3 本文切入点
摒弃原方案中缺乏严密物理学支撑的"全局阈值相变"与"静态黑盒权重"，将 **RGMem** 的节点级抽象算子（$\mathcal{R}_{K2}$）与 **Agent KB** 的效用分数（Utility Score）无缝融合至 **A-Mem** 的笔记网络中，实现以局部邻居网络（Local Community）为中心的自适应重整化与遗忘。

---

## 3. 核心方法论（公式级精确）

本方案由三个基于文献严格支撑的模块构成：

### 模块 1：记忆状态表示与效用追踪模块 ($\mathcal{M}_{\text{state}}$)

**功能定义**：若系统接收原始交互文本 $c_n$ 与时间戳 $t_n$，则调用语言模型（基于系统级模板 $P_{s1}$）生成上下文描述 $X_n$、关键词 $K_n$ 与标签 $G_n$，并调用文本编码器 $f_{\text{enc}}$ 拼接时间戳生成高维密集向量 $e_n = f_{\text{enc}}(c_n \parallel t_n \parallel K_n \parallel G_n \parallel X_n)$，组合封装为原子笔记 $m_n$ 存入系统；若在问答环节历史笔记 $m_j$ 被检索命中，则依据当前检索的奖励信号 $r_j \in \{0, 1\}$，执行基于学习率 $\eta$ 的动量更新，重置其效用分数 $u_j$。

**基础数据结构**直接继承自 **A-Mem (§3.1, Eq. 1)**，每个记忆笔记表示为：
$$
m_i = \{c_i, t_i, K_i, G_i, X_i, e_i, L_i\}
$$

**效用更新机制**引入 **Agent KB (§3.2, Self-Evolving Memory)** 中的"自适应驱逐策略（adaptive eviction policy）"，为每个 $m_i$ 附加一个动态效用分数 $u_i$。在每次检索交互后，通过奖励信号 $r_i$（如检索成功或执行增益）进行动量更新：
$$
u_i \gets u_i + \eta(r_i - u_i) \quad \text{(来源: Agent KB §3.2)}
$$

**输入输出规范**：
- **输入**：原始交互 $c_n$、时间戳 $t_n$、检索奖励信号 $r_j$、原始效用分数 $u_j$
- **输出**：原子笔记元组 $m_n = \{c_n, t_n, K_n, G_n, X_n, e_n, L_n\}$、更新后的效用分数 $u_j^*$

---

### 模块 2：局部证据累积与阈值触发模块 ($\mathcal{M}_{\text{trigger}}$)

**功能定义**：若新笔记 $m_n$ 完成构建，则计算 $e_n$ 与全局记忆库 $\mathcal{M}$ 中所有元素向量的余弦相似度 $s_{n,j}$，提取Top-$k$节点形成局部邻接集 $L_n$。指定核心拓扑节点 $v$，收集局部邻域内未被处理的混合尺度证据，更新集合 $\mathcal{I}_v^{\text{new}}$。若该集合内的证据势 $|\mathcal{I}_v^{\text{new}}|$ 严格大于临界超参数 $\theta_{\text{sum}}$，则生成激活信号。

**链接构建与局部邻居检索**直接采用 **A-Mem (§3.2, Eq. 4-5)**：
$$
s_{n, j} = \frac{e_n \cdot e_j}{|e_n| |e_j|}
$$
$$
\mathcal{M}_{\text{near}}^n = \{m_j \mid \text{rank}(s_{n, j}) \leq k, m_j \in \mathcal{M}\}
$$

**局部证据累积**严格采用 **RGMem (§3.3.2, Eq. 7)** 的混合尺度输入集定义：
$$
\mathcal{I}_v^{\text{new}} = \left\{\mathcal{T}_{e_i}^{(1), \text{new}} \right\}_{e_i \in N(v)} \cup \left\{d_j^{\text{new}} \right\}_{j \in D(v)} \quad \text{(来源: RGMem Eq.7)}
$$

**阈值触发条件**：为防止过早抽象导致信息丢失，局部重整化操作仅在节点 $v$ 累积的新证据数量超过临界阈值 $\theta_{\text{sum}}$ 时触发：
$$
b_{\text{trigger}} = \mathbb{1}\left[|\mathcal{I}_v^{\text{new}}| > \theta_{\text{sum}}\right]
$$
（"executed only when sufficient new evidence accumulates for a node, controlled by a threshold $\theta_{\text{sum}}$"，来源: **RGMem §3.3.2**）

**输入输出规范**：
- **输入**：新笔记向量 $e_n$、全局记忆库 $\mathcal{M}$、局部累积阈值 $\theta_{\text{sum}}$
- **输出**：布尔型触发变量 $b_{\text{trigger}} \in \{0, 1\}$、混合尺度输入集 $\mathcal{I}_v^{\text{new}}$

---

### 模块 3：节点级重整化与效用驱逐模块 ($\mathcal{M}_{\text{renorm}}$)

**功能定义**：若 $b_{\text{trigger}} = 1$，则对 $\mathcal{I}_v^{\text{new}}$ 依次执行投影选择函数 $\mathbb{P}$ 与非线性聚合函数 $\mathbb{S}$，输出代表局部网络主导模式的序参量 $\Sigma_v$ 与代表差异特征的修正项 $\Delta_v$。若底层笔记 $m_j \in \mathcal{I}_v^{\text{new}}$ 的语义被 $\Sigma_v$ 逻辑蕴含（Entailment），且其效用分数 $u_j$ 严格小于驱逐阈值 $\theta_{\text{evict}}$，则从 $\mathcal{M}$ 中执行物理删除操作。

**投影-选择算子 ($\mathbb{P}$)**：过滤并优先处理已聚合的关系级理论而非原始微观观察：
$$
D'_v = \mathbb{P}\left(\mathcal{I}_v^{\text{new}}\right) \quad \text{(来源: RGMem Eq.9)}
$$

**综合-重整化算子 ($\mathbb{S}$)**：执行非线性聚合生成更新后的高阶表示：
$$
\Sigma_v^{(2, t+1)} = \text{Agg}_{\text{common}}\left(D'_v, \Sigma_v^{(2, t)}\right) \quad \text{(来源: RGMem Eq.11)}
$$
$$
\Delta_v^{(2, t+1)} = \text{Extract}_{\text{salient}}\left(D'_v, \Delta_v^{(2, t)}\right) \quad \text{(来源: RGMem Eq.12)}
$$

其中：
- **序参量 $\Sigma_v$**：捕获跨多种情境持续存在的主导模式（来源: **RGMem §3.3.2**）
- **修正项 $\Delta_v$**：保留显著但非普适的信号，明确表示档案内部的冲突或过渡行为（来源: **RGMem §3.3.2**）

**效用驱逐决策**：结合 **Agent KB (§3.2)** 的驱逐机制，对于已经被吸收到 $\Sigma_v$ 中且效用分数 $u_j$ 极低的底层原子笔记执行安全删除。

**输入输出规范**：
- **输入**：触发信号 $b_{\text{trigger}}$、输入集 $\mathcal{I}_v^{\text{new}}$、驱逐阈值 $\theta_{\text{evict}}$、节点效用分数集合 $\{u_j\}$
- **输出**：高阶重整化表示 $(\Sigma_v^{(2, t+1)}, \Delta_v^{(2, t+1)})$、剪枝更新后的全局记忆库 $\mathcal{M}'$

---

## 4. 模块关系矩阵 (Relationship Matrix)

| 源模块 | 目标模块 | 通信协议与数据流载荷 | 依赖/触发条件 |
| :--- | :--- | :--- | :--- |
| **外部环境** | $\mathcal{M}_{\text{state}}$ | 同步 RPC / 传入 $(c_n, t_n)$ | 若产生新对话轮次，则触发 |
| $\mathcal{M}_{\text{state}}$ | **全局存储** | 数据库插入 (DB Insert) / 传入新笔记 $m_n$ | 若 $m_n$ 实例化完成，则触发 |
| **全局存储** | $\mathcal{M}_{\text{trigger}}$ | 内存对象传参 / 传入全量 $\mathcal{M}$ 向量 | 若进入邻居检索阶段，则触发 |
| $\mathcal{M}_{\text{state}}$ | $\mathcal{M}_{\text{trigger}}$ | 内存对象传参 / 传入 $(m_n, \mathcal{M})$ | 若 $m_n$ 实例化完成，则触发 |
| $\mathcal{M}_{\text{trigger}}$ | $\mathcal{M}_{\text{renorm}}$ | 事件驱动 (Event-driven) / 传入 $\{\mathcal{I}_v^{\text{new}}, \{u_j\}\}$ | 若 $|\mathcal{I}_v^{\text{new}}| > \theta_{\text{sum}}$ 且未冷却，则触发 ($b_{\text{trigger}}=1$) |
| $\mathcal{M}_{\text{renorm}}$ | **全局存储** | 数据库覆写 (DB Overwrite) / 传入 $\mathcal{M}'$ | 若驱逐算子遍历结束，则触发 |

---

## 5. 完整数据流图 (Data Flow Diagram)

```
[新交互流输入: (c_n, t_n)] 
       │
       ▼
(模块 M_state: 构建原子表示)
 ├─ 检索命中历史集 R_hit, 获取反馈 r_j
 ├─ 更新命中笔记效用: u_j ← u_j + η(r_j - u_j)
 ├─ 生成辅助元数据: K_n, G_n, X_n ← LLM
 ├─ 生成密集向量: e_n ← f_enc
 └─ 输出: 完整笔记对象 m_n
       │
       ▼
(模块 M_trigger: 拓扑映射与证据追踪)
 ├─ 计算全局余弦相似度: s_{n,j}
 ├─ 提取 Top-k 形成子图结构: L_n
 ├─ 更新局部证据缓存池: I_v^{new} ← 融合生成混合尺度证据(N(v), D(v))
 └─ 逻辑判断: |I_v^{new}| > θ_sum ?
       │
       ├─ [若判断为 False] ──→ (终止当前计算图，等待下一轮交互)
       │
       └─ [若判断为 True ] ──→ (输出 b_trigger = 1，继续向下流转)
              │
              ▼
(模块 M_renorm: 多尺度演化与自适应驱逐)
 ├─ 投影滤波: D'_v ← P(I_v^{new})
 ├─ 抽象生成: Σ_v ← Agg_common(D'_v)
 ├─ 冲突保留: Δ_v ← Extract_salient(D'_v)
 │
 ├─ [遍历原始输入集 I_v^{new} 中的每个底层笔记 m_j]
 │     ├─ 逻辑判断: Entailment(m_j, Σ_v) == True ∧ (u_j < θ_evict) ?
 │     ├─ [若为 True ] ──→ 执行 Hard_Delete(m_j)
 │     └─ [若为 False] ──→ 保留 m_j 状态
 │
 └─ 清空局部证据缓存: I_v^{new} ← ∅
       │
       ▼
[输出最终记忆库状态 M']
```

---

## 6. 核心算法伪代码

```latex
\begin{algorithm}[H]
\caption{Local PhaseForget-Zettel: Threshold-Driven Renormalization and Utility-Aware Eviction}
\label{alg:phase_forget}
\begin{algorithmic}[1]
\REQUIRE Interaction stream $\mathcal{D}$, Global memory base $\mathcal{M}$, Evidence threshold $\theta_{\text{sum}}$, Eviction threshold $\theta_{\text{evict}}$, Utility learning rate $\eta$
\ENSURE Evolved and pruned memory state $\mathcal{M}'$

\FOR{each input tuple $(c_n, t_n) \in \mathcal{D}$}
    \STATE \textbf{/* Module $\mathcal{M}_{\text{state}}$: Representation \& Utility */}
    \STATE $R_{\text{hit}}, \{r_j\} \leftarrow \text{Retrieve}(\mathcal{M}, c_n)$ \COMMENT{Retrieve context and get feedback}
    \FOR{each retrieved historical note $m_j \in R_{\text{hit}}$}
        \STATE $u_j \leftarrow u_j + \eta(r_j - u_j)$ \COMMENT{Agent KB utility momentum update}
    \ENDFOR
    \STATE $K_n, G_n, X_n \leftarrow \text{LLM}(c_n \parallel t_n \parallel P_{s1})$
    \STATE $e_n \leftarrow f_{\text{enc}}(\text{concat}(c_n, t_n, K_n, G_n, X_n))$
    \STATE $m_n \leftarrow \{c_n, t_n, K_n, G_n, X_n, e_n, \emptyset\}$
    \STATE $\mathcal{M} \leftarrow \mathcal{M} \cup \{m_n\}$
    
    \STATE \textbf{/* Module $\mathcal{M}_{\text{trigger}}$: Local Accumulation */}
    \FOR{each $m_j \in \mathcal{M} \setminus \{m_n\}$}
        \STATE $s_{n, j} \leftarrow \frac{e_n \cdot e_j}{|e_n| |e_j|}$
    \ENDFOR
    \STATE $L_n \leftarrow \{m_j \mid \text{rank}(s_{n,j}) \leq k\}$
    \STATE $m_n.L \leftarrow L_n$ \COMMENT{Update topology links in note}
    \STATE Define local target node $v$ based on $L_n$ \COMMENT{Fallback to next if $v$ is in cooldown $T_{\text{cool}}$}
    \STATE $\mathcal{I}_v^{\text{new}} \leftarrow \mathcal{I}_v^{\text{new}} \cup \{ \mathcal{T}_{e_i}^{(1), \text{new}} \}_{e_i \in N(v)} \cup \{ d_j^{\text{new}} \}_{j \in D(v)}$
    
    \STATE \textbf{/* Module $\mathcal{M}_{\text{renorm}}$: Renormalization \& Eviction */}
    \IF{$|\mathcal{I}_v^{\text{new}}| > \theta_{\text{sum}}$ \AND $\text{Cooldown}(v) = \text{FALSE}$}
        \STATE $D'_v \leftarrow \mathbb{P}(\mathcal{I}_v^{\text{new}})$ \COMMENT{Filter evidence by projection $\mathbb{P}$}
        \STATE $\Sigma_v^{(2, t+1)} \leftarrow \text{Agg}_{\text{common}}(D'_v, \Sigma_v^{(2, t)})$ \COMMENT{Extract macroscopic invariant}
        \STATE $\Delta_v^{(2, t+1)} \leftarrow \text{Extract}_{\text{salient}}(D'_v, \Delta_v^{(2, t)})$ \COMMENT{Preserve salient tensions}
        
        \FOR{each underlying note $m_j \in \mathcal{I}_v^{\text{new}}$}
            \IF{$\text{Entailment}(m_j, \Sigma_v^{(2, t+1)}) = \text{TRUE}$ \AND $u_j < \theta_{\text{evict}}$}
                \STATE $\Sigma_v^{(2, t+1)}.L \leftarrow \Sigma_v^{(2, t+1)}.L \cup (m_j.L \setminus \{\Sigma_v^{(2, t+1)}\})$ \COMMENT{Topology inheritance with DAG self-loop check}
                \STATE $\mathcal{M} \leftarrow \mathcal{M} \setminus \{m_j\}$ \COMMENT{Safely prune absorbed \& useless nodes}
            \ENDIF
        \ENDFOR
        \STATE $\mathcal{I}_v^{\text{new}} \leftarrow \emptyset$ \COMMENT{Reset local counter}
    \ENDIF
\ENDFOR

\STATE $\mathcal{M}' \leftarrow \mathcal{M}$
\RETURN $\mathcal{M}'$
\end{algorithmic}
\end{algorithm}
```

---

## 7. 实验设计（可复现标准）

### 7.1 数据集 (Datasets)

必须采用文献中明确验证过长期记忆与偏好演化（Concept Drift）的数据集：

1.  **LoCoMo**：用于评估长上下文依赖和多跳推理（来源：A-Mem §4.1, RGMem §1）
2.  **PersonaMem**：针对用户偏好不断演变（evolving user states）和冲突对话环境下的个性化维持（来源：RGMem §1 摘要及实验设定）
3.  **DialSim**：基于电视节目的长程多方对话问答（来源：A-Mem §4.1）

### 7.2 评估指标 (Evaluation Metrics)

1.  **问答质量指标**：F1 score, BLEU-1（来源：A-Mem §4.1 表1）
2.  **系统效率指标**：Memory Usage (MB) 和 Retrieval Time ($\mu$s) 随对话轮次/记忆规模(1,000至1,000,000)的增长曲线（来源：A-Mem §4.5 表4）

### 7.3 基线系统对照组 (Baselines)

全部选用文献中开源且已被比较的代表性基线：

1.  **A-Mem (原版)**：作为无遗忘机制的绝对基准（来源：A-Mem原文）
2.  **MemGPT**：具备基于操作系统层面的内存分页缓存机制的SOTA（来源：A-Mem §2.1, §4.1）
3.  **MemoryBank**：利用艾宾浩斯遗忘曲线机制的基准（来源：A-Mem §4.1, 表1）

---

## 8. 文献溯源矩阵 (Traceability Matrix)

| 设计要素 | 来源文献 | 具体章节/公式 | 使用方式 |
| :--- | :--- | :--- | :--- |
| **基础数据结构(原子笔记)** | A-Mem | §3.1, Eq.(1)(2)(3) | 直接引用，作为记忆存储的底层节点 $m_i$ |
| **链接构建(邻居检索)** | A-Mem | §3.2, Eq.(4)(5) | 直接引用，利用余弦相似度构建局部聚类集 |
| **动态遗忘依据(效用分数)** | Agent KB | §3.2, "Self-Evolving Memory" | 适应性修改，替换原方案黑盒权重，使用 $u_j \gets u_j + \eta(r_j - u_j)$ 评估笔记存留价值 |
| **相变触发条件(局部积累)** | RGMem | §3.3.2, "Node-Level Abstraction..." | 适应性修改，摒弃全局判断，采用局部证据积累量超 $\theta_{\text{sum}}$ 触发重整化 |
| **记忆合并策略(主次分离)** | RGMem | §3.3.2, Eq.(11)(12) | 直接引用，将高度重叠笔记合并为 $\Sigma_v$ 和 $\Delta_v$，取代原方案昂贵的双轨制与桥梁中心度计算 |
| **投影选择算子($\mathbb{P}$)** | RGMem | §3.3.2, Eq.(9) | 直接引用，过滤并优先处理已聚合的关系级理论 |
| **序参量与修正项** | RGMem | §3.3.2, "Order Parameter $\Sigma$ / Correction Term $\Delta$" | 直接引用，分离主导模式与冲突信号 |

---

## 9. 质量检查清单 (Quality Checklist)

- [x] **所有方法公式均能在参考文献中找到等价或高度相似的形式**（A-Mem Eq 1-5; Agent KB 效用更新公式; RGMem Eq 7, 9, 11, 12 全部落实对应）
- [x] **所有核心概念均已在参考文献中定义**（"效用分数/Utility"见Agent KB，"相变/Renormalization Group / Order parameter $\Sigma$"见RGMem）
- [x] **实验数据集已在 $\ge 1$ 篇参考文献中使用**（PersonaMem 源自 RGMem，LoCoMo/DialSim 源自 A-Mem）
- [x] **未引入任何参考文献中未提及的超参数或架构设计**（已删除了原先主观构造的静态权重 $\alpha, \beta, \gamma$，指数衰减 $\lambda$，和全局相变图算法）
- [x] **已明确区分"直接引用"、"适应性修改"和"受启发设计"三类引用**（见文献溯源矩阵最后一列）

---

## 10. 与原方案的差异对照

| 原方案要素 | 问题诊断 | 本方案修正 | 修正依据 |
| :--- | :--- | :--- | :--- |
| 全局阈值触发 $\text{TriggerForget} = \mathbb{1}[\frac{1}{|\mathcal{M}|}\sum \text{FP}(m_i) > \theta_{\text{phase}}]$ | 缺乏物理学相变严密性，全局平均导致频繁误触发 | 局部证据积累触发 $b_{\text{trigger}} = \mathbb{1}[|\mathcal{I}_v^{\text{new}}| > \theta_{\text{sum}}]$ | RGMem §3.3.2 |
| 静态权重 $\text{FP}(m_i) = \alpha \cdot \text{Red} + \beta \cdot (1-\text{Util}) + \gamma \cdot \text{Stale}$ | 黑箱参数，跨任务域缺乏鲁棒性 | 动态效用分数 $u_j \gets u_j + \eta(r_j - u_j)$ | Agent KB §3.2 |
| LLM双轨制合并（规则+LLM） | 文献未支撑，成本高昂 | RGMem重整化算子 $\mathcal{R}_{K2} = \mathbb{S} \circ \mathbb{P}$ | RGMem §3.3.2 |
| 桥梁节点全局连通分量分析 | 计算复杂度过高（$O(|V||E|)$） | 局部序参量抽象与效用驱逐联合决策 | RGMem Eq.11-12 + Agent KB §3.2 |
| 指数衰减陈旧度 $\text{Staleness}(m_i) = 1 - \exp(-\lambda \cdot \Delta t_i)$ | 文献未支撑，无法区分语义过时与暂时沉默 | 基于效用反馈的动态权重调整 | Agent KB §3.2 |

---

*文档生成时间：2026-03-07*  
*基于文献：A-Mem (Xu et al., 2025), RGMem (Tian et al., 2026), Agent KB (Tang et al., 2025)*

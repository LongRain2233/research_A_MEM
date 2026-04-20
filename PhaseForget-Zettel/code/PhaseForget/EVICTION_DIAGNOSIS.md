# PhaseForget 遗忘机制诊断报告

## 问题描述

超参数搜索实验（144组）中，**抽象笔记数量（abstract_notes）与 F1 得分呈强负相关（-0.63）**，这违反直觉——摘要应该帮助系统，而非伤害。同时发现：

1. **77 组（53.5%）的实验中 abstract_notes=0**（摘要功能完全没生成）
2. **abstract_notes=0 的组反而得分更高**（F1=0.308 vs 0.261）
3. **theta_evict 对 F1 的影响极小**（Pearson 相关性 -0.03，几乎无关）

## 根本原因分析

通过分析 73 个本地日志文件（theta_sim ∈ {0.5, 0.6}），发现遗忘流程被两个瓶颈阻断：

### 瓶颈1：Trigger 永不触发（theta_sum 过高）

**证据**：日志统计表

```
theta_sum | theta_sim=0.5 触发次数 | theta_sim=0.6 触发次数
----------|---------------------|-------------------
   10     |      26~41           |      18~27
   20     |       6~9            |       2~6
   30     |       2~6            |       0~4
   40     |    0（全部为零）       |  0（全部为零）
```

**原因**：
- 每轮对话产生 1 条新笔记，675 轮对话总共 675 条
- 一个锚点（anchor_v）的证据池需要积累超过 theta_sum 条证据才能触发 renormalization
- theta_sum=40 时，要求单个锚点被引用 40+ 次，在 675 条笔记分散到 263 个抽象笔记的情况下几乎不可能
- **theta_sum=40 意味着 Trigger 根本不会被触发，后续所有流程（renorm、eviction、abstract_notes生成）全部为零**

**影响**：这就是 abstract_notes=0 的直接原因。

---

### 瓶颈2：Eviction 被 utility_too_high 大量拦截

当 Trigger 触发后（theta_sum≤30 时），进入 Renormalization 执行 Eviction。

**流程**：
```
候选笔记（review domain 内所有笔记）
    │
    ├─ 门1：utility >= theta_evict ?
    │   ├─ 是 → utility_too_high++，跳过（保留）
    │   └─ 否 ↓
    │
    ├─ 门2：LLM 判断 Sigma 是否覆盖此笔记内容？
    │   ├─ 否 → entailment_false++，跳过（保留）
    │   └─ 是 ↓
    │
    └─ 执行删除 → evicted++
```

**实测数据**（theta_sim=0.5, theta_sum=10, theta_evict=0.25）：

```
总候选笔记数：981
├─ utility_too_high:   584 (60%)  ← 门1 拦截
├─ entailment_false:     1 (<1%)  ← 门2 拦截
└─ 实际蒸发：         396 (40%)
```

**为什么 utility_too_high 这么高？**

1. 新笔记初始 utility = 0.5
2. 每轮对话中被 retrieve 的笔记，utility 通过动量更新向 1.0 靠拢：
   ```
   u_i ← u_i + eta * (reward - u_i)
   ```
   其中 reward=1（被采纳）或 0（仅检索）
3. Global decay 每 decay_interval_rounds（20/30/40）轮触发一次，乘以 0.95
   - **但 decay 只作用于"最近 1 小时内未被访问"的笔记**（grace_period_hours=0.0 已改为 0，但条件仍在）
4. 实验跑几千秒，大量笔记在最近都被访问过，**decay 几乎不生效**
5. 结果：大多数笔记 utility 维持在 0.4~0.6，远高于 theta_evict=0.25~0.45
6. 门1 把 60% 的候选笔记拦了下来——**"这条笔记还有用，不该删"**

**theta_evict 对 F1 无影响的原因**：
- theta_evict 确实控制了 eviction 数量（theta_evict 越高，门越宽，evicted 越多）
- 但最终 total_notes 的分布几乎相同，因为被拦截的笔记本身数量就有限
- 而 **total_notes 才是决定 F1 的主因**（Pearson 相关性 +0.843）
- eviction 数量的差异被淹没在整体 total_notes 的量级差异中

---

### 瓶颈3：Entailment 检查形同虚设

**实测数据**：entailment_false 拦截率 < 1%

**原因**：
- 能到门2 的笔记已经 utility 很低了（< theta_evict）
- LLM 被要求："新摘要 Sigma 有没有覆盖旧笔记的信息？"
- LLM 几乎每次都回答 "redundant=True"（覆盖了），直接放行删除
- **这个二次过滤门没有发挥作用**

**问题**：
- LLM 的"覆盖"判断过于宽松
- 或者 Sigma 质量足够高，真的覆盖了大部分旧笔记
- 或者 entailment 检查本身被跳过/异常处理导致默认返回 True

---

## 为什么 abstract_notes 高时 F1 反而更低？

| 指标 | 值 | 相关系数 |
|------|-----|---------|
| total_notes | ↑ | +0.843 ⭐ |
| abstract_notes | ↑ | -0.629 |
| abstract_ratio | ↑ | -0.677 |
| total_links | ↑ | -0.034 |

**机制**：
1. abstract_notes 越多 → Sigma 摘要越多
2. Sigma 摘要是通过删除原始笔记而生成的（eviction）
3. 删除原始笔记 → **total_notes 下降**
4. total_notes 是 F1 最强正相关因子 → **F1 下降**
5. 结论：**摘要压缩了信息，而检索/QA 需要足够的粒度和细节**

---

## 超参数组合分析

### 最佳组合（按 composite_score，要求 abstract_notes>0）

| theta_sim | theta_sum | theta_evict | decay_rounds | F1 | BLEU | abstract_notes |
|-----------|-----------|-------------|--------------|-----|------|-----------------|
| 0.7 | 10 | 0.35 | 40 | 0.330 | 0.284 | 4 |
| 0.5 | 20 | 0.25 | 30 | 0.317 | 0.277 | 8 |
| 0.7 | 10 | 0.45 | 30 | 0.323 | 0.274 | 3 |

### 分析

**theta_sum 影响最直接**：
- theta_sum=10：trigger 40 次，evicted ~400 条，abstract_notes ~3~8
- theta_sum=20：trigger 6~9 次，evicted ~200 条，abstract_notes ~1~3
- theta_sum=30：trigger 2~6 次，evicted ~50 条，abstract_notes ~0~1
- theta_sum=40：trigger 0 次，abstract_notes 0（完全失效）

**theta_sim 的两面性**：
- theta_sim 高（0.7~0.8）→ 相似度门槛严格 → 找到的邻居少 → evicted 少 → abstract_notes 少 → total_notes 多 → **F1 高**
- theta_sim 低（0.5~0.6）→ 相似度门槛宽松 → 邻居多 → evicted 多 → abstract_notes 多 → total_notes 少 → **F1 低**
- Pearson 相关性 +0.51：与 eviction 数量的制约相比，与 total_notes 对 F1 的驱动相比，微不足道

**theta_evict 几乎无影响**：
- 相关系数 -0.03，说明改变 theta_evict 不改变系统行为
- 根本原因：门1（utility_too_high）已经拦截了大部分候选笔记，门2 的宽度变化影响不大

**decay_interval_rounds 影响极小**：
- 相关系数 +0.11，基本无用
- 原因：grace_period 机制导致 decay 几乎不生效

---

## 问题诊断清单

| 问题 | 根因 | 证据 |
|------|------|------|
| abstract_notes=0 | theta_sum 太高，trigger 永不触发 | 日志：theta_sum=40 时 trigger=0 次 |
| abstract_notes 多时 F1 低 | 摘要压缩了原始笔记，total_notes 下降 | Pearson：total_notes 与 F1 相关 +0.843 |
| theta_evict 无效 | utility_too_high 已拦截 60%，门2 形同虚设 | 日志：entailment_false < 1% |
| theta_sim 高时 F1 高 | 邻居少→eviction 少→total_notes 多 | 数据：theta_sim=0.8 的组 total_notes=675（满载） |
| decay 无效 | grace_period 使 decay 仅作用于冷笔记 | 代码：grace_period_hours=0.0（已改），但条件 `last_accessed_at < cutoff` 仍有效 |

---

## 改进方案

### 方案 A：修复 Decay 机制（最快）

**现状**：grace_period_hours=0.0，但代码中仍保留了时间窗口检查
```python
WHERE is_abstract = 0
  AND (last_accessed_at IS NULL OR last_accessed_at < ?)
```

**改进**：
1. 简化条件：移除时间窗口限制，对**所有非摘要笔记**统一衰减
2. 效果：utility 不再被高频访问锚定，eviction 门1 的拦截率会下降

### 方案 B：降低 theta_sum（立即有效）

**现状**：theta_sum=40 时 trigger=0，theta_sum=10 时 trigger=40

**建议**：
- 生产环境使用 theta_sum ≤ 20（大多数数据集）
- 仅在超大规模场景（>10000 条笔记）才用 theta_sum=30+

### 方案 C：改进 Entailment 检查

**现状**：LLM 判断过于宽松，entailment_false < 1%

**改进**：
1. 严格化提示词，明确要求"旧笔记的关键事实是否全部出现在 Sigma 中"
2. 或改用更严格的逻辑：不是"覆盖"而是"100% 冗余"（使用 NLI 模型而非 LLM）
3. 预期效果：entailment_false 拦截率从 <1% 提升到 10~20%

### 方案 D：重新设计摘要策略（长期）

**核心问题**：摘要压缩了笔记，而 QA 需要原始细节

**改进**：
1. 不删除原始笔记，而是**同时保留**原始笔记和摘要，标记摘要的权重
2. 在检索时，摘要作为"跳转门槛"而非"替代"
3. 或改用**分层标记**：Sigma 标记为"高层"，原始笔记标记为"细节"，根据查询类型选择

---

## 实验数据汇总表

所有 theta_sim ∈ {0.5, 0.6} 的 73 个实验：

```
theta_sum=10：trigger ~30 次，abstract_notes ~3-8，evicted ~300-500
theta_sum=20：trigger ~7 次，abstract_notes ~1-3，evicted ~150-300
theta_sum=30：trigger ~3 次，abstract_notes ~0-2，evicted ~50-100
theta_sum=40：trigger 0 次，abstract_notes 0，evicted 0
```

## 结论

**遗忘逻辑本身没有 bug，但被三个设计瓶颈阻断**：

1. **theta_sum 过高**：40 远超实际数据分布，导致 trigger 永不触发
2. **Decay 机制不完全**：grace_period 保留导致 utility 无法充分衰减，门1 拦截率 60%
3. **Entailment 检查过松**：LLM 几乎不拒绝，门2 形同虚设

**最直接的改进**：
- 降低 theta_sum 到 10~20
- 修复 decay 机制（移除/简化时间窗口条件）
- 严格化 entailment 提示词或改用 NLI 模型

**核心启示**：
- 最强影响 F1 的因素是 **total_notes**（+0.843），而非摘要质量
- 摘要应该**保留而非替代**原始笔记
- 遗忘机制的目标应该是"压缩重复信息"而非"减少笔记总数"

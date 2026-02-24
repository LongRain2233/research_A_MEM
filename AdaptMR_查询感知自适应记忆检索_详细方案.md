# AdaptMR：面向 Agent 记忆的查询感知自适应检索框架

> 详细研究方案说明文档  
> 生成日期：2026-02-13

---

## 1. 研究背景：为什么需要"专属检索策略"

### 1.1 现有系统的统一检索范式

当前主流 Agent 记忆系统（Mem0、A-MEM、MemGPT、H-MEM 等）在检索记忆时，几乎都采用**同一种方式**：

```
用户输入 query → 转为 embedding 向量 → 在记忆库中计算 cosine similarity → 返回 top-K 条最相似记忆 → 送入 LLM 生成回答
```

这种方式称为**统一向量检索（Unified Embedding Retrieval）**，它的核心假设是：

> "语义上最相似的记忆 = 最有用的记忆"

### 1.2 这个假设什么时候会失败？

用一个具体场景说明。假设 Agent 的记忆库中存储了以下 8 条记忆（按时间排列）：

| 编号 | 时间 | 记忆内容 |
|------|------|---------|
| M1 | 1月5日 | 用户说："我住在北京朝阳区" |
| M2 | 2月10日 | 用户说："我最近在找杭州的工作" |
| M3 | 3月15日 | 用户说："我养了一只叫小白的猫" |
| M4 | 4月20日 | 用户说："我搬到杭州西湖区了" |
| M5 | 5月1日 | 用户说："推荐一下杭州的日料店" → Agent推荐了"�的矢日料" |
| M6 | 6月8日 | 用户说："我最近开始跑步了，每天5公里" |
| M7 | 7月12日 | 用户说："推荐一下杭州的川菜馆" → Agent推荐了"蜀乡情" |
| M8 | 8月3日 | 用户说："今天跑了10公里，比上个月进步不少" |

现在看 5 种不同类型的查询，统一向量检索分别会遇到什么问题：

---

#### 场景 A：事实提取 —— "我的猫叫什么名字？"

| 统一向量检索 | 结果 |
|-------------|------|
| 搜索 "猫 名字" 相关的 embedding | 找到 M3（"叫小白的猫"），相似度最高 |
| **效果** | ✅ **正确**。语义匹配在这种简单事实查询上表现良好 |

**结论：向量检索对事实提取类问题通常有效。**

---

#### 场景 B：时序推理 —— "我是先开始跑步还是先搬到杭州的？"

| 统一向量检索 | 结果 |
|-------------|------|
| 搜索 "跑步 搬家 杭州" 相关的 embedding | 可能返回 M4、M6、M8（都和搬家/跑步相关） |
| 返回给 LLM 的信息 | M4: "搬到杭州"、M6: "开始跑步"、M8: "跑了10公里" |
| **问题** | ❌ **向量检索不携带时间顺序信息**。LLM 收到 3 条记忆，但不知道 M4 在 M6 之前还是之后（记忆的呈现顺序可能是按相似度排的，不是按时间排的） |

**统一向量检索的缺陷：忽略了时间维度，无法自然支持"先后"判断。**

---

#### 场景 C：知识更新 —— "我现在住在哪里？"

| 统一向量检索 | 结果 |
|-------------|------|
| 搜索 "住在哪里 住址" 相关的 embedding | 返回 M1（"北京朝阳区"，相似度 0.92）和 M4（"杭州西湖区"，相似度 0.89） |
| 返回给 LLM 的信息 | 两条矛盾的记忆同时出现 |
| **问题** | ❌ **M1 的相似度可能比 M4 更高**（因为 M1 直接说"住在"，而 M4 说"搬到"），导致 LLM 可能回答"北京" |

**统一向量检索的缺陷：不区分新旧信息的优先级，旧答案可能因"措辞更直接"而相似度更高。**

---

#### 场景 D：多跳推理 —— "我提到过的餐厅里，哪种菜系我提到的次数最多？"

| 统一向量检索 | 结果 |
|-------------|------|
| 搜索 "餐厅 菜系 次数" 相关的 embedding | 可能返回 M5（"日料店"）和 M7（"川菜馆"），但也可能返回 M6（"跑步"因为向量空间中"活动"和"餐厅"可能有些关联） |
| **问题** | ❌ **需要先找到所有提到餐厅的记忆，再聚合统计**。但一次 top-K 检索可能遗漏部分餐厅记忆，也可能引入不相关记忆 |

**统一向量检索的缺陷：单次检索无法支持"先收集、再聚合"的多步推理。**

---

#### 场景 E：拒绝回答 —— "我跟你说过我的血型吗？"

| 统一向量检索 | 结果 |
|-------------|------|
| 搜索 "血型" 相关的 embedding | 没有任何记忆与血型相关，但 top-K 仍会返回 K 条"最不相关中最相关的"记忆（如 M6"跑步"、M8"10公里"，因为都和健康相关） |
| 返回给 LLM 的信息 | K 条不太相关的记忆 |
| **问题** | ❌ **LLM 可能基于这些弱相关记忆臆造答案**（如"根据您经常运动的习惯来看..."） |

**统一向量检索的缺陷：永远返回 K 条结果，不会说"我没找到相关记忆"。**

---

### 1.3 核心洞察

> **不同类型的查询，需要不同的"翻记忆本"方式。**
> 用一种方法解决所有问题，必然在某些类型上表现很差。

这就是 **AdaptMR** 要解决的问题。

---

## 2. 解决方案：五种查询类型的专属检索策略

### 2.1 整体架构

```
                    ┌─────────────────────────────┐
                    │        用户查询 (Query)       │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │   查询类型分类器 (Classifier)  │
                    │  识别查询属于哪种类型          │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │  事实提取   │ │  时序推理   │ │  知识更新   │  ...
              │  Strategy  │ │  Strategy  │ │  Strategy  │
              └─────┬─────┘ └─────┬─────┘ └─────┬─────┘
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    检索到的记忆 + 元信息       │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │      LLM 生成最终回答         │
                    └─────────────────────────────┘
```

整个系统分为三个核心模块：
1. **查询类型分类器**：判断查询属于哪种类型
2. **策略池**：5 种专属检索策略
3. **策略路由器**：将查询路由到对应策略（或多策略融合）

---

### 2.2 模块一：查询类型分类器

#### 分类体系

基于 LongMemEval 的定义，将查询分为 **5 种类型**：

| 类型ID | 类型名称 | 定义 | 典型问句模式 |
|--------|---------|------|-------------|
| T1 | **事实提取（Factual Extraction）** | 从历史对话中直接提取某个具体事实 | "我的XX是什么？"、"你记得我说过XX吗？" |
| T2 | **时序推理（Temporal Reasoning）** | 需要理解事件的时间先后关系 | "我是先XX还是先XX？"、"XX之前发生了什么？"、"我最近一次XX是什么时候？" |
| T3 | **知识更新（Knowledge Update）** | 涉及已变化/更新过的信息 | "我现在的XX是什么？"（隐含信息可能已更新）、"我最近换了XX" |
| T4 | **多跳推理（Multi-hop Reasoning）** | 需要综合多条记忆进行推理 | "我提到过的XX中哪个最XX？"、"综合来看XX" |
| T5 | **拒绝判断（Abstention）** | 需要判断记忆中是否有相关信息，没有则拒绝回答 | "我有没有跟你说过XX？"、"你知道我的XX吗？" |

#### 实现方案（三选一，从易到难）

**方案 A：基于 LLM 的 Few-shot 分类（最简单，推荐新手）**

```python
CLASSIFIER_PROMPT = """你是一个查询类型分类器。根据用户的查询，判断它属于以下哪种类型：

1. factual_extraction: 直接提取某个具体事实
   示例："我的猫叫什么名字？"、"你记得我说过喜欢什么颜色吗？"

2. temporal_reasoning: 需要理解时间先后关系
   示例："我是先换的工作还是先搬的家？"、"上个月我跟你聊了什么？"

3. knowledge_update: 涉及可能已更新的信息
   示例："我现在住在哪里？"、"我最近的工作是什么？"

4. multi_hop: 需要综合多条信息推理
   示例："我提到过的餐厅里哪个评价最好？"、"我和小王的共同爱好是什么？"

5. abstention: 需要判断是否有相关信息
   示例："我有没有跟你说过我的血型？"、"你知道我父亲的职业吗？"

用户查询：{query}

请只返回类型名称（如 factual_extraction），不需要解释。"""
```

调用成本：每次分类仅需 1 次 API 调用（~100 tokens），成本约 $0.0001/次。

**方案 B：基于轻量级模型的微调分类器（适合有少量 GPU 的情况）**

```python
# 使用 BERT-base 或 DeBERTa-v3-base 微调
# 训练数据：LongMemEval 的 500 个标注查询 + 自行扩展
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=5  # 5种查询类型
)
# 微调后模型仅 ~400MB，推理延迟 <50ms
```

**方案 C：规则 + LLM 混合（平衡效率和准确率）**

```python
def classify_query(query):
    # 第一层：关键词规则快速判断（覆盖 ~60% 的简单情况）
    temporal_keywords = ["先", "后", "之前", "之后", "什么时候", "最近一次", "上次"]
    update_keywords = ["现在", "目前", "最新", "换了", "改了"]
    abstention_keywords = ["有没有说过", "你知道吗", "我提过吗"]
    
    if any(kw in query for kw in temporal_keywords):
        return "temporal_reasoning"
    if any(kw in query for kw in update_keywords):
        return "knowledge_update"
    if any(kw in query for kw in abstention_keywords):
        return "abstention"
    
    # 第二层：复杂查询走 LLM 分类（处理剩余 ~40%）
    return llm_classify(query)
```

#### 分类器评估

在正式实验前需要先评估分类器本身的准确率：
- 使用 LongMemEval 的 500 个带标注的查询作为测试集
- 目标：分类准确率 ≥ 85%（低于此则分类器本身的误差会影响下游策略效果）
- 上界分析：用 Oracle（人工正确分类）的结果作为性能上界，展示"如果分类完美，策略自适应最多能提升多少"

---

### 2.3 模块二：五种专属检索策略的详细设计

---

#### 策略 1：事实提取检索（Factual Retrieval）

**适用查询：** "我的猫叫什么名字？"、"你记得我喜欢什么颜色吗？"

**核心思路：** 标准语义检索已经足够好，但可以通过**实体聚焦**进一步提升精度。

**实现步骤：**

```
Step 1: 从查询中提取关键实体/概念
        "我的猫叫什么名字？" → 提取实体 ["猫", "名字"]

Step 2: 语义向量检索 top-2K 候选记忆（K 的 2 倍，留出筛选空间）

Step 3: 实体匹配重排序——在候选记忆中，同时包含提取实体的记忆排名提升
        - M3 ("小白的猫") 包含"猫" → 排名提升
        - M6 ("开始跑步") 不包含"猫" → 排名不变

Step 4: 返回 top-K 条重排序后的记忆
```

**与统一检索的区别：** 增加了实体匹配的重排序步骤，减少语义相近但实体不相关的记忆干扰。

**技术实现：**

```python
def factual_retrieval(query, memory_store, K=5):
    # Step 1: 提取查询中的关键实体
    entities = extract_entities(query)  # 用 LLM 或 NER 工具提取
    
    # Step 2: 语义检索 top-2K
    candidates = memory_store.semantic_search(query, top_k=2*K)
    
    # Step 3: 实体匹配重排序
    for mem in candidates:
        entity_overlap = len(set(entities) & set(mem.entities))
        mem.score = mem.similarity * (1 + 0.3 * entity_overlap)  # 加权
    
    # Step 4: 按新分数排序，返回 top-K
    candidates.sort(key=lambda m: m.score, reverse=True)
    return candidates[:K]
```

---

#### 策略 2：时序推理检索（Temporal Retrieval）

**适用查询：** "我是先换工作还是先搬家？"、"上个月我们聊了什么？"、"最近一次去医院是什么时候？"

**核心思路：** 时间维度是第一优先级。检索时必须**保留和暴露时间信息**。

**实现步骤：**

```
Step 1: 从查询中提取时间约束
        "上个月我们聊了什么？" → 时间窗口 = [上月1日, 上月末]
        "我是先换工作还是先搬家？" → 无具体时间窗口，但需提取事件 ["换工作", "搬家"]

Step 2: 根据有无明确时间窗口，走不同子路径

  Path A (有明确时间窗口):
    → 先按时间戳过滤，仅保留时间窗口内的记忆
    → 在过滤后的子集内做语义检索
    → 按时间正序排列返回（最早→最晚）
  
  Path B (无明确时间窗口，比较先后):
    → 对每个事件分别做语义检索（"换工作" → M_work, "搬家" → M_move）
    → 返回每个事件的最相关记忆 + 对应时间戳
    → 额外返回时间关系标注："M_work.timestamp vs M_move.timestamp"

Step 3: 在返回给 LLM 的 prompt 中，显式标注每条记忆的时间信息
        格式："[2024-04-20] 用户说：我搬到杭州西湖区了"
```

**与统一检索的关键区别：**
1. 加入了**时间窗口过滤**——先缩小搜索范围
2. 返回记忆时**按时间排序**而非按相似度排序
3. 每条记忆都**显式携带时间戳**

**技术实现：**

```python
def temporal_retrieval(query, memory_store, K=5):
    # Step 1: 提取时间约束和事件
    time_window = extract_time_window(query)  # 返回 (start, end) 或 None
    events = extract_events(query)             # 返回事件列表，如 ["换工作", "搬家"]
    
    if time_window:
        # Path A: 有明确时间窗口
        filtered_memories = memory_store.filter_by_time(
            start=time_window[0], end=time_window[1]
        )
        results = semantic_search_within(query, filtered_memories, top_k=K)
        results.sort(key=lambda m: m.timestamp)  # 按时间正序
        
    elif len(events) >= 2:
        # Path B: 比较事件先后
        results = []
        for event in events:
            event_memories = memory_store.semantic_search(event, top_k=3)
            best_match = event_memories[0]
            results.append(best_match)
        results.sort(key=lambda m: m.timestamp)  # 按时间正序
        
    else:
        # 默认：语义检索 + 时间排序
        results = memory_store.semantic_search(query, top_k=K)
        results.sort(key=lambda m: m.timestamp)
    
    # 为每条记忆标注时间信息
    for mem in results:
        mem.display = f"[{mem.timestamp}] {mem.content}"
    
    return results
```

---

#### 策略 3：知识更新检索（Update-Aware Retrieval）

**适用查询：** "我现在住在哪里？"、"我目前的工作是什么？"

**核心思路：** 当同一主题存在新旧两条记忆时，**优先返回最新版本**，并显式标注"此信息已更新"。

**实现步骤：**

```
Step 1: 语义检索 top-2K 候选记忆

Step 2: 对候选记忆按主题聚类
        例如搜索 "住在哪里" 返回：
        - 簇A（住址相关）: M1 [1月] "北京朝阳区", M4 [4月] "杭州西湖区"
        - 簇B（其他）: M2 [2月] "找杭州的工作"

Step 3: 对每个主题簇，执行"更新检测"
        - 检查簇内的记忆是否存在更新关系（同一属性的新值覆盖旧值）
        - 若存在更新：仅保留最新版本，并标注"(已更新，原为：旧值)"
        - 若不存在更新：正常返回

Step 4: 返回经过更新消解后的记忆
        返回："[4月20日] 用户住在杭州西湖区 (更新于4月，此前为北京朝阳区)"
```

**与统一检索的关键区别：**
1. 增加了**主题聚类**——识别"关于同一件事的多条记忆"
2. 增加了**更新检测**——判断是否存在新旧覆盖关系
3. 返回**消解后的单一答案**而非多条矛盾记忆

**技术实现：**

```python
def update_aware_retrieval(query, memory_store, K=5):
    # Step 1: 语义检索候选
    candidates = memory_store.semantic_search(query, top_k=2*K)
    
    # Step 2: 主题聚类（使用 embedding 相似度聚类同主题记忆）
    clusters = cluster_by_topic(candidates, similarity_threshold=0.85)
    
    # Step 3: 每个簇内做更新检测
    resolved_memories = []
    for cluster in clusters:
        if len(cluster) > 1:
            # 同主题存在多条记忆 → 可能是更新
            # 按时间排序，取最新
            cluster.sort(key=lambda m: m.timestamp, reverse=True)
            newest = cluster[0]
            oldest = cluster[-1]
            
            # 用 LLM 判断是否是更新关系
            is_update = check_update_relation(newest, oldest)
            
            if is_update:
                newest.display = (
                    f"[{newest.timestamp}] {newest.content} "
                    f"(已更新，此前为：{oldest.content})"
                )
                resolved_memories.append(newest)
            else:
                resolved_memories.extend(cluster)
        else:
            resolved_memories.extend(cluster)
    
    return resolved_memories[:K]


def check_update_relation(mem_new, mem_old):
    """用 LLM 判断两条记忆是否是同一信息的新旧版本"""
    prompt = f"""判断以下两条记忆是否描述了同一信息的更新：
    
旧记忆 [{mem_old.timestamp}]: {mem_old.content}
新记忆 [{mem_new.timestamp}]: {mem_new.content}

如果新记忆是对旧记忆的更新（同一属性的新值），返回 YES。
如果是不同的信息，返回 NO。"""
    
    return llm_call(prompt).strip() == "YES"
```

---

#### 策略 4：多跳推理检索（Multi-hop Retrieval）

**适用查询：** "我提到过的餐厅里哪种菜系最多？"、"我和小王有什么共同爱好？"

**核心思路：** 单次检索不够，需要**分步检索 + 中间结果聚合**。

**实现步骤：**

```
Step 1: 将复杂查询分解为多个子查询
        "我提到过的餐厅里哪种菜系最多？"
        → 子查询1: "用户提到过哪些餐厅？"
        → 子查询2: "每个餐厅是什么菜系？"

Step 2: 逐步执行子查询
        子查询1 → 检索到 M5("日料店-鶴矢")、M7("川菜馆-蜀乡情")
        子查询2 → 基于子查询1的结果，提取菜系信息 → ["日料", "川菜"]

Step 3: 聚合中间结果
        统计：日料 1次，川菜 1次 → 无明显最多
        或若有更多记忆 → 川菜 3次 > 日料 1次 → 回答"川菜"

Step 4: 将所有检索到的记忆 + 中间推理结果一并返回给 LLM
        提供：原始记忆条目 + 聚合后的结构化信息
```

**与统一检索的关键区别：**
1. **查询分解**——将一个复杂查询拆成多个简单子查询
2. **迭代检索**——后续检索基于前序结果
3. **中间聚合**——在检索阶段就完成部分推理，减轻 LLM 负担

**技术实现：**

```python
def multihop_retrieval(query, memory_store, K=10, max_hops=3):
    # Step 1: 用 LLM 分解查询
    sub_queries = decompose_query(query)
    # 例如返回: ["用户提到过哪些餐厅？", "每个餐厅分别是什么菜系？"]
    
    all_memories = []
    intermediate_results = {}
    
    # Step 2: 逐步执行子查询
    for i, sub_q in enumerate(sub_queries):
        if i == 0:
            # 第一步：直接语义检索
            memories = memory_store.semantic_search(sub_q, top_k=K)
        else:
            # 后续步：基于前序结果构造增强查询
            enhanced_query = f"{sub_q}\n\n已知信息：{intermediate_results}"
            memories = memory_store.semantic_search(enhanced_query, top_k=K)
        
        all_memories.extend(memories)
        
        # 用 LLM 从检索结果中提取中间答案
        intermediate_results[sub_q] = extract_intermediate_answer(
            sub_q, memories
        )
    
    # Step 3: 去重并返回
    unique_memories = deduplicate(all_memories)
    
    # 附加中间推理结果作为 context
    context = {
        "memories": unique_memories[:K],
        "intermediate_reasoning": intermediate_results
    }
    return context


def decompose_query(query):
    """用 LLM 将复杂查询分解为子查询序列"""
    prompt = f"""将以下复杂问题分解为 2-3 个简单的子问题，按执行顺序排列：

问题：{query}

要求：
1. 每个子问题应该可以通过搜索记忆来回答
2. 后续子问题可以依赖前序子问题的结果
3. 返回 JSON 列表格式

示例：
问题："我和小王有什么共同爱好？"
子问题：["我的爱好有哪些？", "小王的爱好有哪些？"]"""
    
    return json.loads(llm_call(prompt))
```

---

#### 策略 5：拒绝判断检索（Abstention-Aware Retrieval）

**适用查询：** "我有没有跟你说过我的血型？"、"你知道我父亲的名字吗？"

**核心思路：** 检索后需要**判断"是否真的找到了相关记忆"**，而不是强行返回 top-K。

**实现步骤：**

```
Step 1: 语义检索 top-K 候选记忆

Step 2: 计算 top-1 结果的"相关性置信度"
        - 如果 top-1 相似度 > 阈值 θ_high (如 0.85)：有相关记忆，正常返回
        - 如果 top-1 相似度 < 阈值 θ_low (如 0.60)：很可能无相关记忆
        - 如果介于两者之间：不确定，需进一步判断

Step 3: 对"不确定"区间的结果，用 LLM 做二次判断
        prompt: "以下记忆是否包含关于 [查询主题] 的信息？"
        如果 LLM 判断 "否" → 返回"无相关记忆"信号

Step 4: 根据判断结果，返回不同内容
        - 有相关记忆 → 正常返回 top-K
        - 无相关记忆 → 返回特殊标记 [NO_RELEVANT_MEMORY]
```

**与统一检索的关键区别：**
1. 增加了**置信度阈值判断**——不盲目返回 top-K
2. 增加了**相关性二次确认**——用 LLM 验证检索结果是否真的相关
3. 支持**返回"无记忆"信号**——允许 Agent 坦诚回答"我不记得你提过这件事"

**技术实现：**

```python
def abstention_aware_retrieval(query, memory_store, K=5, 
                                theta_high=0.85, theta_low=0.60):
    # Step 1: 语义检索
    candidates = memory_store.semantic_search(query, top_k=K)
    
    if not candidates:
        return {"memories": [], "has_relevant_memory": False}
    
    top1_score = candidates[0].similarity
    
    # Step 2: 置信度判断
    if top1_score >= theta_high:
        # 高置信度：有相关记忆
        return {"memories": candidates, "has_relevant_memory": True}
    
    elif top1_score < theta_low:
        # 低置信度：很可能无相关记忆
        return {"memories": [], "has_relevant_memory": False}
    
    else:
        # 不确定区间：LLM 二次确认
        topic = extract_topic(query)
        is_relevant = llm_relevance_check(topic, candidates[:3])
        
        if is_relevant:
            return {"memories": candidates, "has_relevant_memory": True}
        else:
            return {"memories": [], "has_relevant_memory": False}


def llm_relevance_check(topic, candidate_memories):
    """用 LLM 判断候选记忆是否真的与查询主题相关"""
    memories_text = "\n".join([m.content for m in candidate_memories])
    prompt = f"""以下记忆内容中，是否包含关于"{topic}"的信息？

记忆内容：
{memories_text}

如果包含，返回 YES。如果不包含（即使有间接关联），返回 NO。"""
    
    return llm_call(prompt).strip() == "YES"
```

---

### 2.4 模块三：策略路由与融合

#### 单策略路由（基础版）

```python
STRATEGY_MAP = {
    "factual_extraction": factual_retrieval,
    "temporal_reasoning": temporal_retrieval,
    "knowledge_update": update_aware_retrieval,
    "multi_hop": multihop_retrieval,
    "abstention": abstention_aware_retrieval,
}

def adaptive_retrieve(query, memory_store, K=5):
    # 1. 分类
    query_type = classify_query(query)
    
    # 2. 路由到对应策略
    strategy = STRATEGY_MAP[query_type]
    
    # 3. 执行检索
    return strategy(query, memory_store, K)
```

#### 多策略融合（进阶版，可选）

有些查询可能同时涉及多种类型（如"我最近一次去过的餐厅叫什么？"既有时序又有事实提取）。此时可以：

```python
def adaptive_retrieve_fusion(query, memory_store, K=5):
    # 1. 分类——返回各类型的概率分布
    type_probs = classify_query_with_probs(query)
    # 例如: {"temporal": 0.6, "factual": 0.3, "update": 0.1, ...}
    
    # 2. 对概率 > 阈值的类型，分别执行检索
    all_results = {}
    for qtype, prob in type_probs.items():
        if prob > 0.2:  # 阈值
            strategy = STRATEGY_MAP[qtype]
            results = strategy(query, memory_store, K)
            all_results[qtype] = (prob, results)
    
    # 3. 加权融合
    # 对所有候选记忆，按"策略权重 × 策略内分数"重新排序
    scored_memories = []
    for qtype, (prob, results) in all_results.items():
        for mem in results:
            scored_memories.append((mem, prob * mem.score))
    
    scored_memories.sort(key=lambda x: x[1], reverse=True)
    return [m for m, s in scored_memories[:K]]
```

---

## 3. 实验方案设计

### 3.1 数据集

| 数据集 | 用途 | 查询数量 | 查询类型标注 | 获取方式 |
|--------|------|---------|------------|---------|
| **LongMemEval** | 主评估集 | 500 | ✅ 有（7种细分类型） | [HuggingFace](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) |
| **LoCoMo** | 辅助评估集 | ~300 | ✅ 有（4种类型） | [GitHub](https://github.com/snap-stanford/locomo) |
| **MSC** (可选) | 补充评估 | ~5000 | ❌ 需手工标注子集 | [ParlAI](https://parl.ai/projects/msc/) |

### 3.2 基线方法

| 基线 | 类型 | 说明 | 来源 |
|------|------|------|------|
| **Unified-Embedding** | 统一向量检索 | 标准 top-K 语义检索，所有查询用同一策略 | 自行实现（最基础baseline） |
| **Mem0** | 工业级记忆系统 | 动态记忆提取 + 向量检索 | [GitHub 开源](https://github.com/mem0ai/mem0) |
| **A-MEM** | 学术级记忆系统 | Zettelkasten结构化笔记 + 检索 | [GitHub 开源](https://github.com/agentic-mem/a-mem) |
| **Zep** | 图记忆系统 | 时序KG + 多种搜索（语义/全文/图） | [GitHub 开源](https://github.com/getzep/graphiti) |
| **LongMemEval-RAG** | 论文官方baseline | LongMemEval论文中的三阶段记忆框架 | [官方代码](https://github.com/xiaowu0162/LongMemEval) |
| **Oracle-Strategy** | 策略上界 | 为每个查询手工选择最优策略（性能上界） | 自行实现 |

### 3.3 评估指标

| 指标 | 计算方式 | 意义 |
|------|---------|------|
| **QA 准确率（Overall Accuracy）** | 正确回答数 / 总查询数 | 整体效果 |
| **各类型准确率（Per-type Accuracy）** | 每种查询类型的单独准确率 | 核心指标——展示不同类型的差异化提升 |
| **检索召回率 Recall@K** | 正确答案所在记忆被检索到的比例 | 检索质量 |
| **检索精确率 Precision@K** | 检索到的 K 条记忆中真正相关的比例 | 检索噪声 |
| **Token 使用量** | 检索+生成阶段消耗的总 token 数 | 效率 |
| **端到端延迟** | 从查询到回答的总时间 | 实用性 |

### 3.4 实验设计（共 5 组核心实验）

#### 实验 1：现有系统的查询类型性能分析（Motivation 实验）

**目的：** 证明"统一检索在某些查询类型上确实很差"，建立研究动机。

```
对 Mem0, A-MEM, Zep 分别在 LongMemEval 上跑完整评估
→ 按 5 种查询类型拆分结果
→ 画出每种系统在各类型上的准确率柱状图
→ 预期发现：所有系统在"事实提取"上表现好（~70-80%），
            在"时序推理"和"知识更新"上表现差（~30-50%）
```

**这个实验本身就是重要的分析贡献。**

#### 实验 2：Oracle 策略上界分析

**目的：** 证明"如果我们能为每种类型选择最优策略，性能上限很高"。

```
对每种查询类型，手工测试所有 5 种策略
→ 选出每种类型的最优策略
→ 组合成 Oracle 策略（上界）
→ 计算 Oracle 与统一检索之间的性能差距
→ 预期：差距 ≥ 10-15% → 证明自适应策略的潜力
```

#### 实验 3：AdaptMR 完整对比实验

**目的：** 展示 AdaptMR 的整体效果。

```
在 LongMemEval 和 LoCoMo 上，对比：
  - AdaptMR（完整版）
  - 所有 6 个基线
→ 报告整体准确率 + 各类型准确率 + 检索效率
→ 预期：AdaptMR 在整体上提升 5-10%，在弱势类型上提升 15-25%
```

#### 实验 4：消融实验

**目的：** 验证每个组件的贡献。

| 消融变体 | 移除内容 | 预期效果 |
|---------|---------|---------|
| w/o Classifier | 移除分类器，随机选策略 | 整体大幅下降 |
| w/o Temporal Strategy | 移除时序策略，统一向量检索 | 时序类查询下降 |
| w/o Update Strategy | 移除更新策略 | 知识更新类查询下降 |
| w/o Multihop Strategy | 移除多跳策略 | 多跳推理类查询下降 |
| w/o Abstention Strategy | 移除拒绝策略 | 拒绝类查询下降 |
| w/o Entity Reranking | 移除事实策略中的实体重排 | 事实提取略微下降 |

#### 实验 5：效率分析

**目的：** 展示 AdaptMR 的额外开销是可接受的。

```
对比各方法的：
  - 平均检索延迟（ms）
  - 平均 Token 使用量
  - API 调用次数
→ 预期：AdaptMR 的额外开销主要来自分类器（~100ms + ~100 tokens）
         多跳策略会增加 2-3 倍延迟（因为迭代检索），但仅占查询总量的 ~15%
         整体平均额外开销 < 30%
```

---

## 4. 实施时间线

| 阶段 | 时间 | 任务 | 产出 |
|------|------|------|------|
| **准备** | 第1-2周 | 复现 LongMemEval 官方 baseline；部署 Mem0/A-MEM 跑通评估 | 基线结果 |
| **分析** | 第3-4周 | 完成实验1和2（按类型分析 + Oracle上界分析） | Motivation 数据 + 策略可行性验证 |
| **实现** | 第5-8周 | 实现 5 种策略 + 查询分类器 + 路由器 | AdaptMR 框架代码 |
| **实验** | 第9-13周 | 完成实验3-5（完整对比 + 消融 + 效率） | 全部实验结果 |
| **写作** | 第14-17周 | 论文撰写 + 图表制作 + 反复修改 | 完整论文初稿 |

**总计：约 4 个月（17 周）**

---

## 5. 论文叙事结构（参考）

```
Title: AdaptMR: Query-Aware Adaptive Memory Retrieval for LLM Agents

Abstract: 
  现有Agent记忆系统采用统一检索策略 → 不同查询类型性能差异大 
  → 我们提出 AdaptMR → 5种专属策略 + 自适应路由 → 显著提升

Section 1 - Introduction:
  - Agent 长期记忆的重要性（引用 Memory Survey, Generative Agents）
  - 现有系统统一检索的局限（引用 Mem0, A-MEM, MemGPT 的单一检索方式）
  - LongMemEval 的 5 种查询类型暴露了不同的检索需求
  - 我们的贡献：(1) 系统分析；(2) 5 种策略设计；(3) 自适应框架

Section 2 - Related Work:
  - Agent 记忆系统（Mem0, A-MEM, MemGPT, Zep, H-MEM, LightMem）
  - 记忆检索优化（RMM 的 RL 重排序, AriGraph 的双重检索）
  - 查询理解与自适应检索（RAG 领域的相关工作）

Section 3 - Motivation Study:
  - 实验1的结果：各系统在不同查询类型上的性能分析
  - 实验2的结果：Oracle 上界证明策略自适应的潜力
  - 核心发现：统一检索在时序/更新/多跳类型上显著劣于专属策略

Section 4 - Method: AdaptMR
  - 4.1 查询类型分类器
  - 4.2 五种专属检索策略
  - 4.3 策略路由与融合机制

Section 5 - Experiments:
  - 5.1 实验设置（数据集、基线、指标）
  - 5.2 主实验结果（实验3）
  - 5.3 消融实验（实验4）
  - 5.4 效率分析（实验5）
  - 5.5 案例分析（各类型的具体检索对比）

Section 6 - Conclusion
```

---

## 6. 已有文献中可引用的关键支撑

| 引用目的 | 文献 | 文件名 |
|---------|------|--------|
| Agent 记忆领域综述 | Memory in the Age of AI Agents | `3S9QUJWQ_Hu等` |
| 记忆流+反思开创性工作 | Generative Agents | `BM2EFQ6T_Park等` |
| OS式记忆管理 | MemGPT | `CHHWJSWN_Packer等` |
| 工业级记忆系统 | Mem0 | `BHW9GDA8_Chhikara等` |
| 结构化记忆 | A-MEM | `7PAIEXBT_Xu等` |
| 时序知识图谱（多种检索方式的存在证据） | Zep | `D9VGYK42_Rasmussen等` |
| 知识图谱+情景记忆双重检索 | AriGraph | `GTEIPZMQ_Anokhin等` |
| RL优化记忆检索排序 | RMM | `Q3PUFMNM_Tan等` |
| 层次化检索 | H-MEM | `RZI3M53F_Sun&Zeng` |
| 轻量级记忆（效率对比） | LightMem | `IDEYWTW4_Fang等` |
| 推理前移（写入端优化对比） | PREMem | `AB57ZFXD_Kim等` |

---

*文档完毕。本方案的核心优势在于：问题定义清晰、实验设计干净、与现有工作形成强对比、全部可通过 API 调用完成、3-4 个月可完成核心实验。*

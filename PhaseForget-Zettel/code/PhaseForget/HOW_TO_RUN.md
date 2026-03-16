# PhaseForget-Zettel 运行指南

## 目录

1. [环境要求](#1-环境要求)
2. [安装步骤](#2-安装步骤)
3. [配置 .env 文件](#3-配置-env-文件)
4. [数据目录初始化](#4-数据目录初始化)
5. [运行模式](#5-运行模式)
   - [交互式 Demo](#51-交互式-demo)
   - [查看系统统计](#52-查看系统统计)
   - [运行基准测试](#53-运行基准测试)
6. [运行测试套件](#6-运行测试套件)
7. [超参数调优参考](#7-超参数调优参考)
8. [LLM 提供商配置](#8-llm-提供商配置)
9. [常见问题与排查](#9-常见问题与排查)

---

## 1. 环境要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | >= 3.10 | 必须，使用 `match` 语句和 asyncio 特性 |
| pip | >= 23.0 | 推荐升级到最新版 |
| Git | 任意 | 克隆项目用 |

> **Windows 用户注意**：ChromaDB 在 Windows 上需要 Visual C++ Build Tools。如果安装时报错，请先安装 [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)。

---

## 2. 安装步骤

### 2.1 进入项目目录

```bash
cd D:\research\PhaseForget-Zettel\code\PhaseForget
```

### 2.2 创建虚拟环境（推荐）

```bash
python -m venv .venv

# Windows 激活
.venv\Scripts\activate

# Linux/macOS 激活
source .venv/bin/activate
```

### 2.3 安装核心依赖

```bash
pip install -e ".[dev]"
```

这会安装：
- `chromadb` — 向量数据库（冷轨道）
- `aiosqlite` — 异步 SQLite（热轨道）
- `sentence-transformers` — 本地嵌入模型（all-MiniLM-L6-v2）
- `litellm` — 多提供商 LLM 网关
- `pydantic` / `pydantic-settings` — 配置管理
- `pytest` / `pytest-asyncio` — 测试框架

### 2.4 安装评估依赖（可选，运行基准测试时需要）

```bash
pip install -e ".[dev,eval]"
```

额外安装：`scikit-learn`、`matplotlib`、`pandas`

### 2.5 验证安装

```bash
python -c "import phaseforget; print('安装成功')"
```

---

## 3. 配置 .env 文件

### 3.1 复制模板

```bash
cp .env.template .env
```

### 3.2 编辑 .env

用文本编辑器打开 `.env`，按以下说明填写：

```env
# ── LLM 提供商配置 ────────────────────────────────────────
# 使用 OpenAI
LLM_PROVIDER=litellm
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
LLM_BASE_URL=

# 使用 Claude (Anthropic)
# LLM_MODEL=claude-3-5-haiku-20241022
# LLM_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 使用本地 Ollama（无需 API Key）
# LLM_MODEL=ollama/qwen2.5:7b
# LLM_BASE_URL=http://localhost:11434
# LLM_API_KEY=ollama

# ── 嵌入模型 ──────────────────────────────────────────────
# 首次运行会自动从 HuggingFace 下载（约 90MB）
EMBEDDING_MODEL=all-MiniLM-L6-v2

# ── 存储路径 ──────────────────────────────────────────────
CHROMA_PERSIST_DIR=./data/chroma_db
SQLITE_DB_PATH=./data/phaseforget.db

# ── 日志 ──────────────────────────────────────────────────
LOG_LEVEL=INFO
LOG_FILE=./data/phaseforget.log
```

### 3.3 关键字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `LLM_MODEL` | 是 | LiteLLM 格式的模型名称，见[第8节](#8-llm-提供商配置) |
| `LLM_API_KEY` | 是* | API 密钥，使用 Ollama 时填 `ollama` |
| `LLM_BASE_URL` | 否 | 仅非标准端点时需要（如本地 Ollama） |
| `EMBEDDING_MODEL` | 否 | 默认 `all-MiniLM-L6-v2`，纯本地运行无需联网 |
| `CHROMA_PERSIST_DIR` | 否 | ChromaDB 持久化目录，默认 `./data/chroma_db` |
| `SQLITE_DB_PATH` | 否 | SQLite 数据库路径，默认 `./data/phaseforget.db` |

---

## 4. 数据目录初始化

首次运行前，确保 `data/` 目录存在：

```bash
mkdir -p data/chroma_db
```

> **注意**：`data/` 目录已在 `.gitignore` 中，不会被提交到版本库。系统在首次 `initialize()` 时也会自动创建必要的目录和数据库表结构。

---

## 5. 运行模式

### 5.1 交互式 Demo

交互式 REPL，用于手动测试系统功能：

```bash
python -m phaseforget demo
```

**可用命令：**

```
> /add 用户喜欢用 Python 做机器学习        # 添加一条记忆
> /search 人工智能                          # 语义检索
> /stats                                    # 查看系统统计
> /quit                                     # 退出
>（直接输入文本）                           # 等同于 /add
```

**示例会话：**

```
PhaseForget-Zettel Interactive Demo
Commands: /add <text> | /search <query> | /stats | /quit
--------------------------------------------------
> /add 用户喜欢徒步旅行，上周去了泰山
  Added note: 3f8a2c1d-...
> /add 用户计划下个月去黄山徒步
  Added note: 7b1e4f2a-...
> /add 用户说他更喜欢春天爬山，天气凉爽
  Added note: 9c2d5e3b-...
> /search 户外运动偏好
  [0.847] 用户喜欢徒步旅行，上周去了泰山
  [0.823] 用户计划下个月去黄山徒步
> /stats
  total_notes: 3
  abstract_notes: 0
  total_links: 2
  cold_track_count: 3
  interaction_count: 3
> /quit
```

### 5.2 查看系统统计

```bash
python -m phaseforget stats
```

输出示例：

```
PhaseForget-Zettel System Statistics
----------------------------------------
  total_notes: 42
  abstract_notes: 7
  total_links: 38
  cold_track_count: 42
  interaction_count: 156
```

| 字段 | 含义 |
|------|------|
| `total_notes` | SQLite 中的记忆节点总数 |
| `abstract_notes` | 经过 Renormalization 生成的抽象节点（Sigma）数量 |
| `total_links` | Zettelkasten 图中的拓扑链接数 |
| `cold_track_count` | ChromaDB 中的向量条目数 |
| `interaction_count` | 系统处理的总交互轮次 |

### 5.3 运行基准测试

基准测试需要准备数据集文件。支持三种数据集格式：

#### LoCoMo（长对话多跳问答）

```bash
python -m phaseforget bench \
  --dataset locomo \
  --data-path /path/to/locomo_data.json \
  --max-sessions 10
```

**数据格式**（JSON，支持两种格式）：

```json
[
  {
    "dialogue": [
      {"role": "user", "content": "你好", "created_at": "2025-01-01T00:00:00"},
      {"role": "assistant", "content": "你好！", "created_at": "2025-01-01T00:00:01"}
    ],
    "questions": [
      {"question": "用户说了什么？", "answer": "你好"}
    ]
  }
]
```

#### PersonaMem（偏好演变追踪）

```bash
python -m phaseforget bench \
  --dataset personamem \
  --data-path /path/to/personamem_data.json
```

**数据格式：**

```json
[
  {
    "user_id": "user_1",
    "sessions": [
      {
        "session_id": "s1",
        "dialogue": [
          {"role": "user", "content": "我喜欢爵士乐", "created_at": "2025-01-01"}
        ]
      }
    ],
    "eval_questions": [
      {"question": "用户的音乐偏好？", "answer": "爵士乐"}
    ]
  }
]
```

#### DialSim（多方对话）

```bash
python -m phaseforget bench \
  --dataset dialsim \
  --data-path /path/to/dialsim_data.json
```

**数据格式：**

```json
[
  {
    "episode_id": "ep1",
    "dialogue": [
      {"speaker": "Alice", "text": "你见过 Bob 吗？"},
      {"speaker": "Charlie", "text": "他去超市了。"}
    ],
    "questions": [
      {"question": "Bob 在哪里？", "answer": "超市"}
    ]
  }
]
```

#### 基准测试输出示例

```
============================================================
BENCHMARK RESULTS
============================================================
Dataset: locomo | Sessions: 10 | Turns: 847

System              F1      BLEU-1  Retrieval(µs)  Memory(MB)
------------------------------------------------------------
PhaseForget         0.634   0.571   1823           48.2
MemoryBank          0.521   0.463   312             12.1
------------------------------------------------------------

Renormalization events: 23
Eviction events: 17
============================================================
```

#### `--max-sessions` 参数

用于快速验证时限制处理的会话数量：

```bash
# 仅处理前 3 个会话（快速冒烟测试）
python -m phaseforget bench --dataset locomo --data-path data.json --max-sessions 3
```

---

## 6. 运行测试套件

### 6.1 运行全部测试

```bash
pytest tests/ -v
```

### 6.2 运行单个测试文件

```bash
pytest tests/test_models.py -v          # 数据模型测试
pytest tests/test_hot_track.py -v       # SQLite 热轨道测试
pytest tests/test_cold_track.py -v      # ChromaDB 冷轨道测试
pytest tests/test_metrics.py -v         # 评估指标测试
pytest tests/test_evaluation.py -v      # 数据集加载器测试
pytest tests/test_pipeline_integration.py -v  # 全流水线集成测试
```

### 6.3 运行特定测试函数

```bash
pytest tests/test_pipeline_integration.py::test_add_single_interaction -v
pytest tests/test_metrics.py::TestF1Score::test_perfect_match -v
```

### 6.4 生成覆盖率报告

```bash
pytest tests/ --cov=phaseforget --cov-report=html
# 报告生成于 htmlcov/index.html
```

### 6.5 测试说明

| 测试文件 | 描述 | 是否需要 LLM API |
|----------|------|-----------------|
| `test_models.py` | MemoryNote、EvidencePool 数据结构 | 否 |
| `test_hot_track.py` | SQLite CRUD、CASCADE 级联删除 | 否 |
| `test_cold_track.py` | ChromaDB 向量搜索、删除 | 否（但需要下载嵌入模型） |
| `test_metrics.py` | F1、BLEU-1 计算、计时器 | 否 |
| `test_evaluation.py` | 数据集格式解析、MemoryBank 基线 | 否 |
| `test_pipeline_integration.py` | 全流水线集成（使用 Mock LLM） | 否（使用 Mock） |

> **所有测试均使用 Mock LLM**，无需配置真实 API Key 即可运行测试。

---

## 7. 超参数调优参考

所有超参数可通过 `.env` 文件覆盖默认值，变量名为大写形式。

| 参数 | 默认值 | .env 变量名 | 调优建议 |
|------|--------|------------|----------|
| `theta_sim` | 0.75 | `THETA_SIM` | 增大→更严格的邻居过滤，减少噪声链接；减小→更多连接，适合领域词汇多的场景 |
| `theta_sum` | 5 | `THETA_SUM` | 增大→更少的 Renormalization 事件，内存保留更多细节；减小→更激进的合并 |
| `theta_evict` | 0.3 | `THETA_EVICT` | 增大→更激进的驱逐，内存保持紧凑；减小→更保守，保留更多历史 |
| `u_init` | 0.5 | `U_INIT` | 新笔记初始效用分数，通常无需调整 |
| `eta` | 0.1 | `ETA` | 效用动量学习率，增大→对最近反馈更敏感 |
| `t_cool` | 3600 | `T_COOL` | Renormalization 后冷却时间（秒），防止级联触发 |
| `retrieval_top_k` | 10 | `RETRIEVAL_TOP_K` | ChromaDB 召回候选数，增大→更全面但更慢 |
| `decay_interval_rounds` | 100 | `DECAY_INTERVAL_ROUNDS` | 每 N 轮应用全局衰减，防止静默节点冻结 |
| `decay_factor` | 0.95 | `DECAY_FACTOR` | 衰减乘数，减小→更快遗忘未被访问的节点 |

### 场景推荐配置

**短期对话（快速遗忘）：**
```env
THETA_EVICT=0.4
T_COOL=600
DECAY_FACTOR=0.90
DECAY_INTERVAL_ROUNDS=20
```

**长期知识管理（保守遗忘）：**
```env
THETA_EVICT=0.15
THETA_SUM=10
T_COOL=7200
DECAY_FACTOR=0.98
DECAY_INTERVAL_ROUNDS=200
```

**测试与调试（激进触发）：**
```env
THETA_SIM=0.3
THETA_SUM=3
T_COOL=60
```

---

## 8. 超参数自动搜索（新功能）

`hyperparameter_search.py` 脚本支持针对 locomo10.json 中部分或全部记录进行超参数自动迭代搜索，帮助快速定位最优超参数区间。

### 三个最重要的超参数

| 超参数 | 作用 | 推荐搜索范围 |
|--------|------|------------|
| `theta_sim` | 控制拓扑邻居相似度阈值——直接决定哪些记忆被关联和触发重整化，对召回质量影响最大 | 0.5 ~ 0.85 |
| `theta_sum` | 证据池积累触发重整化的阈值——控制记忆压缩激进程度 | 3 ~ 12 |
| `theta_evict` | 效用分驱逐阈值——决定哪些历史笔记被删除，直接影响记忆保留质量 | 0.15 ~ 0.6 |

### 快速开始

```bash
# 进入项目目录
cd D:\research\PhaseForget-Zettel\code\PhaseForget

# 只用记录0和1做快速网格搜索（4×4×4=64组，每组需要几分钟）
python hyperparameter_search.py --record-indices 0,1 --search-type grid

# 只用第0条记录做随机搜索（快速验证，20组）
python hyperparameter_search.py --record-indices 0 --search-type random --n-trials 20

# 指定更细粒度的搜索范围（在初步搜索后缩小范围）
python hyperparameter_search.py --record-indices 0,1,2 \
    --theta-sim-values 0.6,0.7,0.75,0.8 \
    --theta-sum-values 4,5,6,7 \
    --theta-evict-values 0.25,0.3,0.35,0.4

# 查看已有搜索结果排行榜（不运行新实验）
python hyperparameter_search.py --show-results

# 清除已有搜索结果重新开始
python hyperparameter_search.py --clear-results
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data-path` | locomo10.json 路径 | `dataset/locomo10.json` |
| `--record-indices` | 逗号分隔的记录索引（0-9），如 `0,1,2` | 全部10条 |
| `--search-type` | `grid`=网格搜索全覆盖；`random`=随机采样 | `grid` |
| `--n-trials` | 随机搜索时的试验次数 | `12` |
| `--theta-sim-values` | theta_sim 候选值，逗号分隔 | `0.5,0.65,0.75,0.85` |
| `--theta-sum-values` | theta_sum 候选值，逗号分隔 | `3,5,8,12` |
| `--theta-evict-values` | theta_evict 候选值，逗号分隔 | `0.15,0.3,0.45,0.6` |
| `--show-results` | 只显示排行榜 | — |
| `--clear-results` | 清除已保存结果 | — |

### 搜索结果

- 每次实验结果自动保存到 `data/hparam_search_results.json`
- 支持**断点续搜**：中断后重新运行会跳过已完成的组合
- 每个实验使用独立 experiment_id 隔离，结束后自动清理临时数据
- 综合评分公式：`0.4×F1 + 0.3×ROUGE-L + 0.2×METEOR + 0.1×BLEU`

### 推荐工作流

```
第1轮：大范围扫描（--record-indices 0）
  → theta_sim: 0.5, 0.65, 0.75, 0.85
  → theta_sum: 3, 5, 8, 12
  → theta_evict: 0.15, 0.3, 0.45, 0.6

第2轮：缩小范围（根据排行榜Top3区域细化，--record-indices 0,1）
  → 在最优区间附近±1档做更细的网格搜索

第3轮：验证（--record-indices 0,1,2,3）用更多数据确认最优参数
```

### 在 bench 命令中使用 --record-indices

标准 bench 命令也支持只选部分记录：

```bash
# 只对记录0和记录2进行 benchmark
python -m phaseforget bench \
    --dataset locomo \
    --data-path dataset/locomo10.json \
    --record-indices 0,2

# 对记录0到4做评估（共5条）
python -m phaseforget bench \
    --dataset locomo \
    --data-path dataset/locomo10.json \
    --record-indices 0,1,2,3,4
```

---

## 9. LLM 提供商配置

PhaseForget 通过 LiteLLM 支持多种 LLM 提供商，只需修改 `.env` 中的 `LLM_MODEL` 和 `LLM_API_KEY`：

### OpenAI

```env
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-xxxxxxxx
```

### Anthropic Claude

```env
LLM_MODEL=claude-3-5-haiku-20241022
LLM_API_KEY=sk-ant-xxxxxxxx
```

### DeepSeek

```env
LLM_MODEL=deepseek/deepseek-chat
LLM_API_KEY=sk-xxxxxxxx
LLM_BASE_URL=https://api.deepseek.com
```

### 本地 Ollama（完全离线）

```bash
# 先启动 Ollama 并拉取模型
ollama serve
ollama pull qwen2.5:7b
```

```env
LLM_MODEL=ollama/qwen2.5:7b
LLM_BASE_URL=http://localhost:11434
LLM_API_KEY=ollama
```

### 通义千问 (DashScope)

```env
LLM_MODEL=dashscope/qwen-turbo
LLM_API_KEY=sk-xxxxxxxx
```

> **提示**：LLM 只在 Renormalization（笔记合并）和元数据提取时调用，正常的搜索和工具调用均为本地操作。若想完全离线运行，使用 Ollama 方案即可。

---

## 10. 常见问题与排查

### Q1: `ModuleNotFoundError: No module named 'phaseforget'`

**原因**：包未以开发模式安装，或虚拟环境未激活。

```bash
# 确认虚拟环境已激活
.venv\Scripts\activate      # Windows
source .venv/bin/activate    # Linux/macOS

# 重新安装
pip install -e ".[dev]"
```

### Q2: `pydantic_settings ImportError`

**原因**：`pydantic-settings` 是单独的包，需要显式安装。

```bash
pip install pydantic-settings>=2.0.0
```

### Q3: ChromaDB 安装失败（Windows）

**原因**：需要 C++ 编译工具。

```bash
# 方案1：安装 Visual C++ Build Tools
# 下载: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# 方案2：使用预编译版本
pip install chromadb --prefer-binary
```

### Q4: 嵌入模型下载失败

**原因**：网络问题，无法访问 HuggingFace。

```bash
# 方案1：使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方案2：手动指定本地模型路径
EMBEDDING_MODEL=/path/to/local/all-MiniLM-L6-v2
```

### Q5: `asyncio.TimeoutError` 在 LLM 调用时

**原因**：LLM API 响应超时（默认 120 秒）。

```bash
# 检查网络连接
# 或切换到更快的模型
LLM_MODEL=gpt-4o-mini   # 比 gpt-4o 快很多
```

### Q6: SQLite `FOREIGN KEY constraint failed`

**原因**：数据目录不存在，或数据库文件损坏。

```bash
mkdir -p data
# 如果数据库损坏，删除重建
rm data/phaseforget.db
python -m phaseforget stats   # 会自动重建表结构
```

### Q7: `pytest` 异步测试报错 `ScopeMismatch`

**原因**：`pytest-asyncio` 版本兼容问题。

```bash
pip install "pytest-asyncio>=0.23"
```

确认 `pyproject.toml` 中有：
```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

### Q8: Demo 模式下 `add_interaction` 很慢

**原因**：首次运行会加载嵌入模型（约 3-5 秒），后续交互会很快。如果每次都慢，检查是否触发了 LLM 元数据提取。可临时调低 `LOG_LEVEL=DEBUG` 查看详细日志。

### Q9: 基准测试报 `No sessions loaded`

**原因**：数据文件格式不匹配。检查数据格式是否符合[第5.3节](#53-运行基准测试)中的结构，或查看日志文件 `data/phaseforget.log` 获取详细错误信息。

### Q10: Windows 上路径包含中文报错

```bash
# 在命令开头设置编码
set PYTHONIOENCODING=utf-8
python -m phaseforget demo
```

---

## 附录：项目文件结构速查

```
PhaseForget/
├── .env                    # 你的配置文件（从 .env.template 复制）
├── .env.template           # 配置模板
├── pyproject.toml          # 项目依赖声明
├── hyperparameter_search.py  # 超参数自动搜索脚本（新功能）
├── data/                   # 运行时数据（gitignored）
│   ├── chroma_db/          # ChromaDB 向量存储
│   ├── phaseforget.db      # SQLite 状态数据库
│   ├── phaseforget.log     # 日志文件
│   └── hparam_search_results.json  # 超参数搜索结果（自动生成）
├── src/phaseforget/        # 源码
│   ├── config/settings.py  # 所有超参数定义
│   ├── pipeline/           # 系统入口（PhaseForgetSystem）
│   ├── memory/             # 三大核心模块
│   ├── storage/            # 冷/热双轨存储
│   ├── llm/                # LLM 抽象层
│   └── evaluation/         # 基准测试框架
│       └── loaders/locomo.py  # 支持 record_indices 参数（新功能）
└── tests/                  # 测试套件
```

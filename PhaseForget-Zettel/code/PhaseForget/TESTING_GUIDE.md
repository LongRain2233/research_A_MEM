# PhaseForget-Zettel 完整测试指南

本文档提供从环境准备到完整验证的系统化测试方案。
python hyperparameter_search.py --record-indices 0,1 --theta-sim-values 0.65,0.7,0.8 --theta-sum-values 10,20,30,40,50 --theta-evict-values 0.2,0.3,0.4
## 目录

1. [阶段 1：环境准备](#阶段-1环境准备)
2. [阶段 2：配置 LLM](#阶段-2配置-llm)
3. [阶段 3：单元测试](#阶段-3单元测试)
4. [阶段 4：快速冒烟测试](#阶段-4快速冒烟测试)
5. [阶段 5：交互式 Demo](#阶段-5交互式-demo)
6. [阶段 6：基准测试](#阶段-6基准测试)
7. [完整一键测试脚本](#完整一键测试脚本)
8. [测试清单](#测试清单)
9. [预期故障排查](#预期故障排查)

---

## 阶段 1：环境准备

**预计时间**：5 分钟

### 1.1 进入项目目录

```bash
cd D:\research\PhaseForget-Zettel\code\PhaseForget
```

### 1.2 升级 pip

```bash
python -m pip install --upgrade pip
```

### 1.3 创建虚拟环境

```bash
python -m venv .venv
```

激活虚拟环境：

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

### 1.4 安装依赖

```bash
pip install -e ".[dev]"
```

此命令会安装：
- `chromadb` — 向量数据库（冷轨道）
- `aiosqlite` — 异步 SQLite（热轨道）
- `sentence-transformers` — 本地嵌入模型
- `litellm` — 多提供商 LLM 网关
- `pydantic` / `pydantic-settings` — 配置管理
- `pytest` / `pytest-asyncio` — 测试框架

### 1.5 验证安装

逐行执行以下命令，验证所有依赖可用：

```bash
python -c "import phaseforget; print('✓ phaseforget 导入成功')"
python -c "import chromadb; print('✓ ChromaDB 可用')"
python -c "import aiosqlite; print('✓ aiosqlite 可用')"
python -c "import litellm; print('✓ litellm 可用')"
python -c "import sentence_transformers; print('✓ sentence-transformers 可用')"
```

**预期输出**：所有行都显示 ✓

---

## 阶段 2：配置 LLM

**预计时间**：3 分钟

### 2.1 复制 .env 模板

```bash
cp .env.template .env
```

### 2.2 选择一个 LLM 方案

#### 方案 A：使用 OpenAI（需要国外账户或翻墙）

编辑 `.env`：

```env
LLM_PROVIDER=litellm
LLM_MODEL=gpt-4o-mini
LLM_API_KEY=sk-your-api-key-here
LLM_BASE_URL=

EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./data/chroma_db
SQLITE_DB_PATH=./data/phaseforget.db
LOG_LEVEL=INFO
LOG_FILE=./data/phaseforget.log
```

#### 方案 B：使用 Ollama（完全离线，推荐用于测试）

**第 1 步**：下载并启动 Ollama

```bash
# 从 https://ollama.ai 下载安装

# 启动 Ollama 服务（保持这个终端打开）
ollama serve
```

**第 2 步**：在另一个终端拉取模型

```bash
ollama pull qwen2.5:7b
```

模型大小约 4GB，首次下载约需 5-10 分钟。

**第 3 步**：编辑 `.env`

```env
LLM_PROVIDER=litellm
LLM_MODEL=ollama/qwen2.5:7b
LLM_BASE_URL=http://localhost:11434
LLM_API_KEY=ollama

EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./data/chroma_db
SQLITE_DB_PATH=./data/phaseforget.db
LOG_LEVEL=INFO
LOG_FILE=./data/phaseforget.log
```

#### 方案 C：使用 Claude（Anthropic）

```env
LLM_PROVIDER=litellm
LLM_MODEL=claude-3-5-haiku-20241022
LLM_API_KEY=sk-ant-your-api-key-here
LLM_BASE_URL=

EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./data/chroma_db
SQLITE_DB_PATH=./data/phaseforget.db
LOG_LEVEL=INFO
LOG_FILE=./data/phaseforget.log
```

### 2.3 验证配置

```bash
# 检查 .env 文件被正确读取
python -c "from phaseforget.config.settings import get_settings; s = get_settings(); print(f'LLM Model: {s.llm_model}'); print(f'Embedding: {s.embedding_model}')"
```

**预期输出**：
```
LLM Model: gpt-4o-mini
Embedding: all-MiniLM-L6-v2
```

---

## 阶段 3：单元测试

**预计时间**：10 分钟

这些测试 **不需要 LLM API**，使用 Mock 或本地计算。

### 3.1 运行所有单元测试

```bash
pytest tests/ -v
```

### 3.2 分类运行测试

如果你想逐个检查各部分：

```bash
# 数据模型测试（无 I/O）
pytest tests/test_models.py -v

# 热轨道测试（SQLite 操作）
pytest tests/test_hot_track.py -v

# 冷轨道测试（ChromaDB 操作）
pytest tests/test_cold_track.py -v

# 评估指标测试（F1、BLEU-1 计算）
pytest tests/test_metrics.py -v

# 数据集加载器测试
pytest tests/test_evaluation.py -v

# 全流水线集成测试（使用 Mock LLM）
pytest tests/test_pipeline_integration.py -v
```

### 3.3 生成覆盖率报告

```bash
pytest tests/ --cov=phaseforget --cov-report=html
```

打开 `htmlcov/index.html` 查看覆盖率详情。

### 3.4 预期结果

```
============================= test session starts ==============================
...
============================== 15 passed in 2.34s ===============================
```

所有测试应该通过（绿色 ✓）。

### 3.5 常见问题

| 问题 | 解决方案 |
|------|--------|
| `asyncio.TimeoutError` | 某个异步操作超时，重新运行试试 |
| `ScopeMismatch: pytest-asyncio` | 升级 `pip install "pytest-asyncio>=0.23"` |
| `ModuleNotFoundError: phaseforget` | 检查虚拟环境是否激活，重新 `pip install -e ".[dev]"` |

---

## 阶段 4：快速冒烟测试

**预计时间**：2 分钟

验证系统初始化和基本统计：

```bash
python -m phaseforget stats
```

### 预期输出

```
PhaseForget-Zettel System Statistics
----------------------------------------
  total_notes: 0
  abstract_notes: 0
  total_links: 0
  cold_track_count: 0
  interaction_count: 0
```

### 关键检查点

- ✓ 命令正常执行，无异常
- ✓ 所有值都是 0（首次运行）
- ✓ `data/` 目录被创建
- ✓ `data/phaseforget.log` 文件存在

---

## 阶段 5：交互式 Demo

**预计时间**：5 分钟

启动交互式 REPL 环境：

```bash
python -m phaseforget demo
```

### 测试脚本

按顺序输入以下命令：

```
> /add 用户喜欢 Python 编程
  Added note: 550e8400-e29b-41d4-a716-446655440000

> /add Python 是一个强大的编程语言
  Added note: 550e8400-e29b-41d4-a716-446655440001

> /add 用户在学习机器学习框架 TensorFlow
  Added note: 550e8400-e29b-41d4-a716-446655440002

> /search Python 编程
  [0.847] 用户喜欢 Python 编程
  [0.721] Python 是一个强大的编程语言

> /stats
  total_notes: 3
  abstract_notes: 0
  total_links: 2
  cold_track_count: 3
  interaction_count: 3

> /quit
```

### 关键检查点

| 检查项 | 预期 |
|--------|------|
| 能正常添加笔记 | 每条 `/add` 都返回有效 ID |
| 搜索返回相关结果 | 返回 2-3 条结果，score 降序 |
| score 在有效范围 | 0.0 ~ 1.0 之间 |
| 统计数字正确 | `total_notes == 3`，`interaction_count == 3` |
| 没有异常错误 | 日志只有 INFO/DEBUG，无 ERROR/EXCEPTION |

### 可能的问题

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| `asyncio` 错误 | 虚拟环境问题 | 重新激活 `.venv` |
| 第一条 `/add` 很慢（3-5 秒） | 嵌入模型首次加载 | 正常现象，耐心等待 |
| 搜索结果为空 | ChromaDB 集合为空或未正确持久化 | 检查 `./data/chroma_db/` 目录 |
| 中文乱码 | 终端编码问题 | 设置 `set PYTHONIOENCODING=utf-8` |
| LLM 连接超时 | 网络或 API 响应慢 | 检查 `.env` 中的 LLM 配置，或换更快的模型 |

### 输出日志位置

实时日志可在以下位置查看：

- **终端输出**：`[INFO]` 开头的消息
- **日志文件**：`./data/phaseforget.log`

查看日志文件：
```bash
tail -f ./data/phaseforget.log
```

---

## 阶段 6：基准测试

**预计时间**：30+ 分钟（取决于数据集大小）

这是验证核心算法和性能的关键阶段。

### 6.1 准备测试数据集

#### 6.1.1 创建最小测试数据

创建文件 `test_data.json`：

```bash
cat > test_data.json << 'EOF'
[
  {
    "dialogue": [
      {"role": "user", "content": "用户喜欢徒步旅行", "created_at": "2025-01-01T00:00:00"},
      {"role": "assistant", "content": "很有趣！", "created_at": "2025-01-01T00:00:01"},
      {"role": "user", "content": "特别是春天在山里徒步", "created_at": "2025-01-02T00:00:00"},
      {"role": "assistant", "content": "春天确实最美。", "created_at": "2025-01-02T00:00:01"}
    ],
    "questions": [
      {"question": "用户喜欢什么户外活动？", "answer": "徒步旅行"},
      {"question": "用户最喜欢什么季节徒步？", "answer": "春天"}
    ]
  },
  {
    "dialogue": [
      {"role": "user", "content": "今天电影很有意思", "created_at": "2025-01-03T00:00:00"},
      {"role": "assistant", "content": "哪部电影？", "created_at": "2025-01-03T00:00:01"},
      {"role": "user", "content": "一部科幻电影，关于太空探险", "created_at": "2025-01-03T00:00:02"}
    ],
    "questions": [
      {"question": "用户看了什么类型的电影？", "answer": "科幻电影"}
    ]
  }
]
EOF
```

#### 6.1.2 其他数据集格式

**PersonaMem 格式**（用户偏好演变）：

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
      },
      {
        "session_id": "s2",
        "dialogue": [
          {"role": "user", "content": "现在我更喜欢摇滚乐", "created_at": "2025-02-01"}
        ]
      }
    ],
    "eval_questions": [
      {"question": "用户的音乐偏好？", "answer": "摇滚乐"}
    ]
  }
]
```

**DialSim 格式**（多方对话）：

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

### 6.2 运行 Benchmark

#### 6.2.1 第一次运行（冷启动）

```bash
python -m phaseforget bench \
  --dataset locomo \
  --data-path test_data.json \
  --max-sessions 2
```

**参数说明**：

| 参数 | 说明 |
|------|------|
| `--dataset` | 数据集类型：`locomo`、`personamem`、`dialsim` |
| `--data-path` | 数据文件路径 |
| `--max-sessions` | 仅处理前 N 个 session（快速测试用） |
| `--reset-checkpoint` | 忽略现有 checkpoint，从头开始 |

#### 6.2.2 预期输出

```
Starting benchmark: dataset=locomo, baselines=['MemoryBank'], checkpoint=./data/bench_locomo_checkpoint.json
Session 1/2 (id=0): 4 turns, 2 questions
Session 1/2 complete (1/2), checkpoint saved
Session 2/2 (id=1): 3 turns, 1 question
Session 2/2 complete (2/2), checkpoint saved
Benchmark complete: locomo
Checkpoint file removed after successful completion

======================================================================
PhaseForget-Zettel Benchmark Report
======================================================================

System              Avg F1     Avg BLEU-1  Avg Retrieval(us)  Samples
----------------------------------------------------------------------
PhaseForget         0.6234     0.5421      2847               3
MemoryBank          0.4521     0.3891      412                3

======================================================================
```

### 6.3 关键检查点

- ✓ 没有中途崩溃（即使数据格式不完美也继续）
- ✓ PhaseForget 的 F1 通常 ≥ MemoryBank（相关数据时）
- ✓ 日志显示了 Renormalization 触发事件
- ✓ 生成了 checkpoint 文件 `./data/bench_locomo_checkpoint.json`
- ✓ 最后自动删除了 checkpoint 文件

### 6.4 测试断点续传

#### 6.4.1 模拟中断

运行 benchmark，然后按 Ctrl+C 中断：

```bash
python -m phaseforget bench \
  --dataset locomo \
  --data-path test_data.json
```

按 Ctrl+C（约 30 秒后）暂停。

#### 6.4.2 验证 Checkpoint 已保存

```bash
# 查看 checkpoint 文件是否存在
ls -la ./data/bench_locomo_checkpoint.json

# 查看内容
cat ./data/bench_locomo_checkpoint.json
```

#### 6.4.3 从中断点继续

重新运行同样命令：

```bash
python -m phaseforget bench \
  --dataset locomo \
  --data-path test_data.json
```

**预期行为**：

```
Checkpoint loaded from ./data/bench_locomo_checkpoint.json: completed_sessions=['0']
Resuming benchmark: 1/2 sessions already done
Session 2/2 (id=1): ...
Checkpoint file removed after successful completion
```

系统应该从第 2 个 session 继续，而不是从头开始。

### 6.5 强制重新运行（清除 Checkpoint）

```bash
python -m phaseforget bench \
  --dataset locomo \
  --data-path test_data.json \
  --reset-checkpoint
```

这会忽略现有 checkpoint，从头开始处理所有 session。

---

## 完整一键测试脚本

将以下内容保存为 `run_tests.sh`（Linux/macOS）或 `run_tests.bat`（Windows）：

### Windows 批处理脚本 (run_tests.bat)

```batch
@echo off
setlocal enabledelayedexpansion

echo ========== PhaseForget-Zettel 完整测试 ==========

REM 激活虚拟环境
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo 错误：无法激活虚拟环境
    exit /b 1
)

REM 1. 单元测试
echo.
echo [1/5] 运行单元测试...
pytest tests/test_models.py tests/test_hot_track.py tests/test_cold_track.py tests/test_metrics.py tests/test_pipeline_integration.py -v --tb=short
if errorlevel 1 (
    echo 单元测试失败
    exit /b 1
)

REM 2. 统计信息
echo.
echo [2/5] 检查系统统计...
python -m phaseforget stats

REM 3. 创建测试数据
echo.
echo [3/5] 准备基准测试数据...
(
    echo [
    echo   {
    echo     "dialogue": [
    echo       {"role": "user", "content": "用户喜欢徒步旅行", "created_at": "2025-01-01T00:00:00"},
    echo       {"role": "assistant", "content": "很有趣！", "created_at": "2025-01-01T00:00:01"}
    echo     ],
    echo     "questions": [
    echo       {"question": "用户喜欢什么？", "answer": "徒步旅行"}
    echo     ]
    echo   }
    echo ]
) > test_data.json

REM 4. 运行基准测试
echo.
echo [4/5] 运行基准测试...
python -m phaseforget bench --dataset locomo --data-path test_data.json --max-sessions 1

REM 5. 显示日志摘要
echo.
echo [5/5] 日志摘要：
echo 完整日志位置：./data/phaseforget.log
tail -20 ./data/phaseforget.log 2>nul || type .\data\phaseforget.log | findstr /E ".*"

echo.
echo ========== 所有测试完成 ==========
echo.
echo 文件检查清单：
if exist ./data/chroma_db/. (
    echo   [✓] ChromaDB 向量库（./data/chroma_db/）
) else (
    echo   [✗] ChromaDB 向量库缺失
)
if exist ./data/phaseforget.db (
    echo   [✓] SQLite 数据库（./data/phaseforget.db）
) else (
    echo   [✗] SQLite 数据库缺失
)
if exist ./data/phaseforget.log (
    echo   [✓] 日志文件（./data/phaseforget.log）
) else (
    echo   [✗] 日志文件缺失
)

pause
```

### Linux/macOS Shell 脚本 (run_tests.sh)

```bash
#!/bin/bash

set -e  # Exit on any error

echo "========== PhaseForget-Zettel 完整测试 =========="

# 激活虚拟环境
source .venv/bin/activate

# 1. 单元测试
echo -e "\n[1/5] 运行单元测试..."
pytest tests/test_models.py tests/test_hot_track.py tests/test_cold_track.py \
  tests/test_metrics.py tests/test_pipeline_integration.py -v --tb=short

# 2. 统计信息
echo -e "\n[2/5] 检查系统统计..."
python -m phaseforget stats

# 3. 创建测试数据
echo -e "\n[3/5] 准备基准测试数据..."
cat > test_data.json << 'EOF'
[
  {
    "dialogue": [
      {"role": "user", "content": "用户喜欢徒步旅行", "created_at": "2025-01-01T00:00:00"},
      {"role": "assistant", "content": "很有趣！", "created_at": "2025-01-01T00:00:01"}
    ],
    "questions": [
      {"question": "用户喜欢什么？", "answer": "徒步旅行"}
    ]
  }
]
EOF

# 4. 运行基准测试
echo -e "\n[4/5] 运行基准测试..."
python -m phaseforget bench --dataset locomo --data-path test_data.json --max-sessions 1

# 5. 显示日志摘要
echo -e "\n[5/5] 日志摘要（最后 20 行）："
tail -20 ./data/phaseforget.log || echo "日志文件不存在"

echo -e "\n========== 所有测试完成 =========="
echo ""
echo "文件检查清单："
[ -d "./data/chroma_db/" ] && echo "  [✓] ChromaDB 向量库（./data/chroma_db/）" || echo "  [✗] ChromaDB 向量库缺失"
[ -f "./data/phaseforget.db" ] && echo "  [✓] SQLite 数据库（./data/phaseforget.db）" || echo "  [✗] SQLite 数据库缺失"
[ -f "./data/phaseforget.log" ] && echo "  [✓] 日志文件（./data/phaseforget.log）" || echo "  [✗] 日志文件缺失"
```

运行脚本：

```bash
# Windows
run_tests.bat

# Linux/macOS
chmod +x run_tests.sh
./run_tests.sh
```

---

## 测试清单

完成各个测试阶段后，在此清单中打勾：

### 基础环境

- [ ] 虚拟环境成功创建并激活
- [ ] `pip install -e ".[dev]"` 完成无错
- [ ] 所有导入验证命令通过（✓ 符号）

### LLM 配置

- [ ] `.env` 文件复制成功
- [ ] 选定了 LLM 提供商（OpenAI / Ollama / Claude）
- [ ] `.env` 中的 `LLM_MODEL` 和 `LLM_API_KEY` 已填写

### 单元测试

- [ ] `pytest tests/test_models.py -v` 通过
- [ ] `pytest tests/test_hot_track.py -v` 通过
- [ ] `pytest tests/test_cold_track.py -v` 通过
- [ ] `pytest tests/test_metrics.py -v` 通过
- [ ] `pytest tests/test_evaluation.py -v` 通过
- [ ] `pytest tests/test_pipeline_integration.py -v` 通过
- [ ] 所有测试总计通过（无 FAILED）

### 冒烟测试

- [ ] `python -m phaseforget stats` 命令执行成功
- [ ] 输出显示 0 个 notes（首次运行）
- [ ] `./data/` 目录已创建
- [ ] `./data/phaseforget.log` 文件存在

### Demo 测试

- [ ] `python -m phaseforget demo` 启动成功
- [ ] `/add` 命令能成功添加笔记
- [ ] `/search` 命令能返回相关结果（score > 0.7）
- [ ] `/stats` 显示正确的计数
- [ ] `/quit` 正常退出

### 基准测试

- [ ] `test_data.json` 文件创建成功
- [ ] `python -m phaseforget bench` 命令开始执行
- [ ] Benchmark 没有中途异常退出
- [ ] 生成了最终报告表格
- [ ] `./data/bench_locomo_checkpoint.json` 文件存在（Benchmark 期间）

### 断点续传测试

- [ ] 第一次 Benchmark 运行时创建 checkpoint
- [ ] 中断后重新运行，自动加载 checkpoint
- [ ] 日志显示 `Resuming benchmark` 消息
- [ ] 仅处理未完成的 session
- [ ] 成功完成后自动删除 checkpoint 文件

### 数据持久化

- [ ] `./data/chroma_db/` 目录包含向量库数据
- [ ] `./data/phaseforget.db` 文件大小 > 0 KB
- [ ] SQLite 包含 4 个表：Memory_State, Memory_Links, Evidence_Pool, System_Meta
- [ ] 重启后 `interaction_count` 不归零

---

## 预期故障排查

### 症状：所有 pytest 超时

**根因**：LLM 响应太慢

**解决方案**：
1. 检查网络连接
2. 换更小的模型：
   ```env
   LLM_MODEL=gpt-4o-mini  # 比 gpt-4o 快
   ```
3. 如使用 Ollama，确保已启动 `ollama serve`

---

### 症状：ChromaDB 查询返回空或异常

**根因**：向量库未正确持久化或初始化

**解决方案**：
1. 检查 `./data/chroma_db/` 目录是否存在
2. 删除目录重新初始化：
   ```bash
   rm -rf ./data/chroma_db
   python -m phaseforget stats  # 重新初始化
   ```
3. 检查 ChromaDB 版本：
   ```bash
   python -c "import chromadb; print(chromadb.__version__)"
   ```

---

### 症状：日志重复输出

**根因**：`setup_logging` 被调用多次添加了重复 handler

**解决方案**：
已在 `logger.py` 中修复。如仍有问题，检查是否在使用最新代码：
```bash
git pull
pip install -e ".[dev]" --force-reinstall
```

---

### 症状：Benchmark 中途崩溃

**根因**：数据格式不预期或 LLM 返回错误

**解决方案**：
1. 查看完整日志：
   ```bash
   tail -100 ./data/phaseforget.log | grep ERROR
   ```
2. 确认数据格式正确（参考 § 6.1.2）
3. 重新运行会从 checkpoint 继续（无需重复处理已完成的 session）

---

### 症状：`interaction_count` 重启后归零

**根因**：未使用最新代码的 SQLite 持久化

**解决方案**：
1. 确认代码已更新
2. 删除旧数据库重新开始：
   ```bash
   rm ./data/phaseforget.db
   python -m phaseforget stats
   ```
3. 检查 SQLite 中是否有 `System_Meta` 表：
   ```bash
   sqlite3 ./data/phaseforget.db ".tables"
   ```
   应显示：`Evidence_Pool  Memory_Links  Memory_State  System_Meta`

---

### 症状：集合为空时 search 异常

**根因**：ChromaDB 在空集合上调用 `query()` 抛异常

**解决方案**：
已在 `chroma_store.py` 中修复（检查 count 并异常处理）。确认使用最新代码：
```bash
grep -n "total == 0" src/phaseforget/storage/cold_track/chroma_store.py
```

---

### 症状：中文输入显示乱码

**根因**：终端编码设置

**解决方案**（Windows）：
```batch
set PYTHONIOENCODING=utf-8
python -m phaseforget demo
```

**解决方案**（Linux/macOS）：
```bash
export PYTHONIOENCODING=utf-8
python -m phaseforget demo
```

---

### 症状：Ollama 连接拒绝

**根因**：未启动 Ollama 服务或地址不对

**解决方案**：
1. 启动 Ollama：
   ```bash
   ollama serve
   ```
2. 验证连接：
   ```bash
   curl http://localhost:11434/api/tags
   ```
3. 检查 `.env` 中的端口是否正确：
   ```env
   LLM_BASE_URL=http://localhost:11434
   ```

---

## 常用调试命令

### 查看实时日志

```bash
tail -f ./data/phaseforget.log
```

### 检查数据库内容

```bash
sqlite3 ./data/phaseforget.db

# SQLite 交互式命令
sqlite> SELECT COUNT(*) FROM Memory_State;
sqlite> SELECT id, utility_score FROM Memory_State LIMIT 5;
sqlite> .quit
```

### 检查 ChromaDB 集合

```bash
python << 'EOF'
from phaseforget.storage.cold_track.chroma_store import ChromaColdTrack
cold = ChromaColdTrack()
print(f"Total notes in ChromaDB: {cold.count()}")
EOF
```

### 清空所有数据（重新开始）

```bash
rm -rf ./data/
mkdir -p ./data/
python -m phaseforget stats  # 重新初始化
```

### 生成故障报告

```bash
python << 'EOF'
import os
import subprocess
from pathlib import Path

report = []
report.append("=== PhaseForget-Zettel 诊断报告 ===\n")

# Python 版本
report.append(f"Python 版本: {subprocess.check_output(['python', '--version']).decode().strip()}")

# 包版本
packages = ['chromadb', 'aiosqlite', 'litellm', 'sentence-transformers', 'pytest']
report.append("\n依赖包版本:")
for pkg in packages:
    try:
        version = subprocess.check_output(['pip', 'show', pkg], universal_newlines=True)
        for line in version.split('\n'):
            if 'Version:' in line:
                report.append(f"  {pkg}: {line.split('Version:')[1].strip()}")
                break
    except:
        report.append(f"  {pkg}: NOT INSTALLED")

# 文件检查
report.append("\n文件系统:")
report.append(f"  ./data/ 存在: {Path('./data').exists()}")
report.append(f"  ./data/chroma_db/ 存在: {Path('./data/chroma_db').exists()}")
report.append(f"  ./data/phaseforget.db 存在: {Path('./data/phaseforget.db').exists()}")
report.append(f"  ./data/phaseforget.log 存在: {Path('./data/phaseforget.log').exists()}")

# 配置检查
report.append("\n配置:")
try:
    from phaseforget.config.settings import get_settings
    s = get_settings()
    report.append(f"  LLM 模型: {s.llm_model}")
    report.append(f"  嵌入模型: {s.embedding_model}")
    report.append(f"  theta_sim: {s.theta_sim}")
    report.append(f"  theta_sum: {s.theta_sum}")
except Exception as e:
    report.append(f"  配置加载失败: {e}")

# 日志末尾
report.append("\n最后 10 行日志:")
log_file = Path('./data/phaseforget.log')
if log_file.exists():
    with open(log_file) as f:
        lines = f.readlines()[-10:]
        for line in lines:
            report.append(f"  {line.rstrip()}")
else:
    report.append("  日志文件不存在")

report = "\n".join(report)
print(report)

# 保存到文件
with open('./diagnostic_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)
print("\n报告已保存至 ./diagnostic_report.txt")
EOF
```

---

## 下一步建议

### 如果所有测试都通过

1. **探索超参数调优**：见 `HOW_TO_RUN.md` 第 7 章
2. **准备真实数据集**：使用你自己的对话数据
3. **集成到应用**：参考 `PhaseForgetSystem` 的 API 设计

### 如果遇到问题

1. **收集诊断信息**：运行上面的"生成故障报告"命令
2. **检查日志**：`./data/phaseforget.log` 包含详细错误信息
3. **隔离问题**：逐阶段运行测试，找到失败点
4. **查看源码注释**：每个模块都有详细的中英文说明

---

## 相关文档

- `README.md` — 项目概述和架构
- `HOW_TO_RUN.md` — 详细运行指南和配置参考
- 代码注释 — 每个模块都有 docstring 说明


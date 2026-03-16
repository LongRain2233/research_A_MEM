# 快速开始：使用不同模型运行实验

## 问题解决

当你想用不同的 LLM 模型（Ollama、GPT-4o、Claude 等）运行同一个基准测试时，需要区分它们的实验数据。

## 解决方案

使用 `--experiment-id` 参数为每个实验命名，自动隔离存储数据。

## 使用示例

### 1. 用 Ollama (Qwen 2.5) 运行实验

```bash
# 第一次运行：构建记忆网络 + 评测
python -m phaseforget bench \
  --experiment-id "qwen2.5-exp1" \
  --dataset locomo \
  --data-path dataset/locomo10.json

# 数据自动保存到: ./data/qwen2.5-exp1/
#   - chroma_db/     (记忆向量数据库)
#   - phaseforget.db (记忆元数据)
#   - phaseforget.log (日志)
```

### 2. 用另一个模型（比如 GPT-4o）运行同样的数据

只需改变 `experiment_id` 和模型配置：

```bash
# 配置 GPT-4o
export OPENAI_API_KEY="your-key"

# 运行实验，自动使用不同的存储目录
python -m phaseforget bench \
  --experiment-id "gpt4o-exp1" \
  --dataset locomo \
  --data-path dataset/locomo10.json

# 数据自动保存到: ./data/gpt4o-exp1/
```

### 3. 再用 Claude 运行

```bash
# 配置 Claude
export ANTHROPIC_API_KEY="your-key"

python -m phaseforget bench \
  --experiment-id "claude-exp1" \
  --dataset locomo \
  --data-path dataset/locomo10.json

# 数据自动保存到: ./data/claude-exp1/
```

## 查看文件结构

运行多个实验后，数据目录会这样组织：

```
./data/
├── qwen2.5-exp1/
│   ├── chroma_db/           # 向量数据库
│   ├── phaseforget.db       # 元数据和状态
│   ├── phaseforget.log      # 运行日志
│   └── bench_locomo_checkpoint.json  # 进度检查点
├── gpt4o-exp1/
│   ├── chroma_db/
│   ├── phaseforget.db
│   ├── phaseforget.log
│   └── bench_locomo_checkpoint.json
└── claude-exp1/
    ├── chroma_db/
    ├── phaseforget.db
    ├── phaseforget.log
    └── bench_locomo_checkpoint.json
```

## 中断和恢复

### 中断实验

按 Ctrl+C 暂停：

```bash
^C
```

### 恢复实验

用相同的 `--experiment-id` 重新运行，会自动从检查点恢复：

```bash
python -m phaseforget bench \
  --experiment-id "qwen2.5-exp1" \
  --dataset locomo \
  --data-path dataset/locomo10.json
```

### 重新开始

如果想重新开始（清除检查点），添加 `--reset-checkpoint`：

```bash
python -m phaseforget bench \
  --experiment-id "qwen2.5-exp1" \
  --dataset locomo \
  --data-path dataset/locomo10.json \
  --reset-checkpoint
```

## 环境变量方式

也可以用环境变量而不是命令行参数：

```bash
# 设置默认实验 ID
export PHASEFORGET_EXPERIMENT_ID="qwen2.5-exp1"

# 现在不需要 --experiment-id 参数
python -m phaseforget bench \
  --dataset locomo \
  --data-path dataset/locomo10.json
```

## 清理实验数据

删除某个实验的所有数据：

```bash
# Linux/Mac
rm -rf ./data/qwen2.5-exp1/

# Windows
rmdir /s /q .\data\qwen2.5-exp1\
```

## 技术细节

实验隔离通过以下方式实现：

1. **Settings 类** (`src/phaseforget/config/settings.py`):
   - 添加 `experiment_id` 字段
   - `get_settings()` 函数根据 experiment_id 自动命名存储目录

2. **CLI 参数** (`src/phaseforget/__main__.py`):
   - 所有命令都支持 `--experiment-id` 参数
   - 参数会传递给 `get_settings(experiment_id)`

3. **自动隔离**:
   - ChromaDB 向量数据库
   - SQLite 元数据数据库
   - 日志文件
   - 检查点文件

都会自动存储在独立的实验目录中，不需要手动干预。

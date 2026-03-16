# Experiment Tracking and Model Differentiation

## Overview

PhaseForget now supports running multiple experiments with different LLM models while keeping their data completely isolated. Each experiment gets its own namespaced storage directories.

## How It Works

When you specify an `--experiment-id`, all storage paths are automatically namespaced:

```
./data/
├── default/                    # Default experiment
│   ├── chroma_db/
│   ├── phaseforget.db
│   └── phaseforget.log
├── qwen2.5-exp1/              # Experiment 1: Qwen 2.5
│   ├── chroma_db/
│   ├── phaseforget.db
│   └── phaseforget.log
├── gpt4o-exp1/                # Experiment 2: GPT-4o
│   ├── chroma_db/
│   ├── phaseforget.db
│   └── phaseforget.log
└── claude-exp1/               # Experiment 3: Claude
    ├── chroma_db/
    ├── phaseforget.db
    └── phaseforget.log
```

Each experiment is completely isolated — shared memory pools, checkpoints, and logs stay separate.

## Usage Examples

### Run experiment with Ollama (Qwen 2.5)

```bash
# Set Ollama as LLM provider in environment or .env
export PHASEFORGET_LLM_BASE_URL="http://localhost:11434"
export PHASEFORGET_LLM_MODEL="qwen2.5"

# Run benchmark with experiment namespacing
python -m phaseforget bench \
  --experiment-id "qwen2.5-exp1" \
  --dataset locomo \
  --data-path dataset/locomo10.json
```

### Run experiment with GPT-4o

```bash
export OPENAI_API_KEY="your-key-here"
export PHASEFORGET_LLM_PROVIDER="litellm"
export PHASEFORGET_LLM_MODEL="gpt-4o-mini"

python -m phaseforget bench \
  --experiment-id "gpt4o-exp1" \
  --dataset locomo \
  --data-path dataset/locomo10.json
```

### Run experiment with Claude

```bash
export ANTHROPIC_API_KEY="your-key-here"
export PHASEFORGET_LLM_PROVIDER="litellm"
export PHASEFORGET_LLM_MODEL="claude-opus-4-6"

python -m phaseforget bench \
  --experiment-id "claude-exp1" \
  --dataset locomo \
  --data-path dataset/locomo10.json
```

### Run demo with specific experiment

```bash
python -m phaseforget demo --experiment-id "my-demo"
```

### Run stats for specific experiment

```bash
python -m phaseforget stats --experiment-id "qwen2.5-exp1"
```

### Using environment variable for experiment-id

Instead of passing `--experiment-id` every time, set the environment variable:

```bash
export PHASEFORGET_EXPERIMENT_ID="qwen2.5-exp1"
python -m phaseforget bench --dataset locomo --data-path dataset/locomo10.json
```

## Checkpoint Behavior

Each experiment maintains its own checkpoint file:
- Default: `./data/bench_locomo_checkpoint.json`
- With experiment-id: `./data/<experiment-id>/bench_locomo_checkpoint.json`

This means:
- **Resume support per experiment**: If you interrupt an experiment run and restart with the same `--experiment-id`, it will resume from the checkpoint
- **Fresh start**: Run with a different `--experiment-id` or use `--reset-checkpoint` to start fresh

## Comparing Experiments

After running multiple experiments, you'll have separate result files and logs in each directory:

```bash
# View Qwen results
cat ./data/qwen2.5-exp1/phaseforget.log | grep "Benchmark complete"

# View GPT-4o results
cat ./data/gpt4o-exp1/phaseforget.log | grep "Benchmark complete"

# View Claude results
cat ./data/claude-exp1/phaseforget.log | grep "Benchmark complete"
```

## Clean Up an Experiment

To delete all data for a specific experiment:

```bash
rm -rf ./data/qwen2.5-exp1/
```

## Implementation Details

The experiment tracking is implemented in:
1. `src/phaseforget/config/settings.py`: Adds `experiment_id` field and namespace logic
2. `src/phaseforget/__main__.py`: Exposes `--experiment-id` CLI argument to all commands

No other code changes needed — the system automatically handles storage path namespacing.

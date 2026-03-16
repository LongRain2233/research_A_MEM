# PhaseForget-Zettel

**Threshold-Driven Local Renormalization and Utility-Aware Forgetting in Zettelkasten Memory**

## Architecture Overview

```
PhaseForget/
├── src/phaseforget/
│   ├── __main__.py          # CLI entry point (demo, stats, bench)
│   ├── config/              # Global settings & hyperparameter registry (§3.4)
│   │   ├── settings.py      # Pydantic-based unified configuration
│   │   └── hyperparams.py   # Hyperparameter documentation & introspection
│   │
│   ├── storage/             # Dual-track storage layer (§2)
│   │   ├── cold_track/      # ChromaDB semantic vector store (§2.1)
│   │   │   └── chroma_store.py   # Cosine similarity, enhanced embedding
│   │   └── hot_track/       # SQLite state management (§2.2)
│   │       └── sqlite_store.py   # Memory_State + Memory_Links + Evidence_Pool
│   │
│   ├── memory/              # Core processing modules (§3)
│   │   ├── models/          # Data structures
│   │   │   ├── note.py          # MemoryNote (m_i = {c,t,K,G,X,e,L})
│   │   │   └── evidence_pool.py # Evidence set I_v^new
│   │   ├── state/           # M_state: Representation & Utility Tracking
│   │   │   └── state_manager.py
│   │   ├── trigger/         # M_trigger: Topology & Evidence Accumulation
│   │   │   └── trigger_engine.py
│   │   └── renorm/          # M_renorm: Renormalization & Eviction
│   │       └── renorm_engine.py  # With recursive depth limiting (§4.2)
│   │
│   ├── llm/                 # LLM abstraction layer (§5.1)
│   │   ├── base.py          # Abstract LLM client interface
│   │   ├── providers/       # Backend implementations
│   │   │   └── litellm_client.py  # Async with timeout handling
│   │   └── prompts/         # Structured prompt templates
│   │       └── templates.py     # Metadata, Renormalization, Entailment
│   │
│   ├── embedding/           # Local embedding encoder (§5.1)
│   │   └── encoder.py       # SentenceTransformer wrapper
│   │
│   ├── pipeline/            # System orchestrator (§5 Data Flow)
│   │   └── orchestrator.py  # PhaseForgetSystem - top-level facade
│   │
│   ├── evaluation/          # Benchmark framework (§7)
│   │   ├── metrics.py       # F1, BLEU-1, retrieval time measurement
│   │   ├── benchmark.py     # BenchmarkRunner with full QA evaluation loop
│   │   ├── loaders/         # Dataset loaders
│   │   │   ├── locomo.py        # LoCoMo (long-context multi-hop)
│   │   │   ├── personamem.py    # PersonaMem (preference evolution)
│   │   │   └── dialsim.py       # DialSim (multi-party dialogue)
│   │   └── baselines/       # Baseline system adapters
│   │       ├── amem_adapter.py      # A-Mem (no forgetting baseline)
│   │       ├── memgpt_adapter.py    # MemGPT (OS-level paging)
│   │       └── memorybank_adapter.py # MemoryBank (Ebbinghaus curve)
│   │
│   └── utils/               # Cross-cutting concerns
│       ├── logger.py        # Centralized logging
│       └── exceptions.py    # Exception hierarchy
│
├── tests/                   # Unit & integration tests
│   ├── conftest.py          # Shared fixtures
│   ├── test_models.py       # MemoryNote, EvidencePool
│   ├── test_hot_track.py    # SQLite operations, CASCADE
│   ├── test_cold_track.py   # ChromaDB search, delete
│   ├── test_metrics.py      # F1, BLEU-1, timer
│   ├── test_evaluation.py   # Dataset loaders, baselines
│   └── test_pipeline_integration.py  # Full pipeline with mock LLM
│
├── data/                    # Runtime data (gitignored)
├── pyproject.toml           # Project configuration
└── .env.template            # Environment variable template
```

## Module-to-Theory Mapping

| Module | Implementation Plan Section | Core Algorithm |
|--------|---------------------------|----------------|
| `memory.state` | §3 Module 1 (M_state) | u_i <- u_i + eta(r_i - u_i) |
| `memory.trigger` | §3 Module 2 (M_trigger) | b_trigger = 1[|I_v^new| > theta_sum] |
| `memory.renorm` | §3 Module 3 (M_renorm) | Sigma = Agg(P(I_v^new)), evict if entailed & u < theta_evict |
| `storage.cold_track` | §2.1 Cold Track | ChromaDB cosine similarity, append-only + physical delete |
| `storage.hot_track` | §2.2 Hot Track | SQLite with ON DELETE CASCADE, 3 tables |
| `pipeline` | §5 Data Flow | Full interaction loop orchestration |
| `evaluation` | §7 Experiments | LoCoMo/PersonaMem/DialSim + 3 baselines |

## Boundary Defenses Implemented

| Defense | Plan Section | Implementation |
|---------|-------------|----------------|
| Dangling Links | §4.1 | ON DELETE CASCADE in Memory_Links DDL |
| Cascading Phase Transitions | §4.2 | Cooldown timer + recursive depth limit (MAX_DEPTH=1) |
| Cold Start | §4.3 | Absolute theta_sim (0.75) lower bound |
| Ghost Read | §3 Module 2 | filter_valid_ids() cross-store sync |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Configure
cp .env.template .env
# Edit .env with your LLM API key

# Run tests
pytest tests/ -v

# Interactive demo
python -m phaseforget demo

# Run benchmark
python -m phaseforget bench --dataset locomo --data-path path/to/data.json
```

## Tech Stack

- **Python 3.10+** with `asyncio`
- **ChromaDB** - semantic vector storage (cold track, cosine distance)
- **SQLite + aiosqlite** - async state management (hot track)
- **LiteLLM** - multi-provider LLM gateway with timeout handling
- **sentence-transformers** - local embedding (all-MiniLM-L6-v2)
- **Pydantic v2 + pydantic-settings** - data validation & .env config

## Sprint Milestones

| Sprint | Deliverable | Status |
|--------|-------------|--------|
| 1 | Dual-track storage + 3-table schema + CASCADE | Complete |
| 2 | Trigger engine + topology + evidence accumulation | Complete |
| 3 | Renormalization pipeline + Sigma/Delta injection + eviction | Complete |
| 4 | Benchmark evaluation + 3 datasets + 3 baselines | Complete |

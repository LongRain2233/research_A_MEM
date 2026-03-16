"""
Global settings and hyperparameter registry.

Aligns with Section 3.4 of the implementation plan: all tunable hyperparameters
are centralized here with their recommended defaults.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for PhaseForget-Zettel system."""

    # ── Core Hyperparameters (Implementation Plan §3.4) ──────────────────

    theta_sim: float = Field(
        default=0.75,
        description="Cosine similarity lower bound for neighbor filtering"
    )
    theta_sum: int = Field(
        default=5,
        description="Evidence pool accumulation threshold to trigger renormalization"
    )
    theta_evict: float = Field(
        default=0.3,
        description="Utility score threshold below which eviction is considered"
    )
    u_init: float = Field(
        default=0.5,
        description="Initial utility score for newly created notes"
    )
    eta: float = Field(
        default=0.1,
        description="Utility momentum learning rate"
    )
    t_cool: int = Field(
        default=3600,
        description="Cooldown period in seconds after renormalization"
    )

    # ── Retrieval Parameters ─────────────────────────────────────────────

    retrieval_top_k: int = Field(default=10, description="Top-K neighbors for ChromaDB recall")
    projection_max_notes: int = Field(default=15, description="Max notes fed into renormalization LLM")
    search_min_similarity: float = Field(
        default=0.0,
        description=(
            "Minimum similarity for user-facing search(). "
            "Kept at 0.0 so results always show. "
            "theta_sim is used only for internal topology filtering."
        )
    )

    # ── Decay Parameters (Silent Node Anti-Freeze) ───────────────────────

    decay_interval_rounds: int = Field(
        default=100,
        description="Apply global decay every N interaction rounds"
    )
    decay_factor: float = Field(
        default=0.95,
        description="Multiplicative decay applied to unretrieved nodes"
    )

    # ── LLM Configuration ────────────────────────────────────────────────

    llm_provider: str = Field(default="litellm")
    llm_model: str = Field(default="gpt-4o-mini")
    llm_api_key: str = Field(default="")
    llm_base_url: str = Field(default="")
    llm_temperature: float = Field(default=0.3)
    llm_max_tokens: int = Field(default=4096)

    # ── Embedding Configuration ──────────────────────────────────────────

    embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # ── Storage Paths ────────────────────────────────────────────────────

    experiment_id: str = Field(
        default="default",
        description="Experiment identifier to namespace storage directories (e.g. 'qwen2.5-exp1', 'gpt4o-exp1')"
    )
    chroma_persist_dir: str = Field(default="./data/chroma_db")
    sqlite_db_path: str = Field(default="./data/phaseforget.db")

    # ── Logging ──────────────────────────────────────────────────────────

    log_level: str = Field(default="INFO")
    log_file: str = Field(default="./data/phaseforget.log")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


def get_settings(experiment_id: str = None) -> Settings:
    """
    Returns a Settings instance with experiment_id-namespaced storage paths.

    If experiment_id is provided, storage paths are updated to:
        ./data/<experiment_id>/chroma_db/
        ./data/<experiment_id>/phaseforget.db

    This allows running multiple experiments with different models while keeping
    their data isolated.

    Args:
        experiment_id: Optional experiment identifier. If None, uses environment
                      variable PHASEFORGET_EXPERIMENT_ID or 'default'.
    """
    if experiment_id is None:
        experiment_id = os.getenv("PHASEFORGET_EXPERIMENT_ID", "default")

    settings = Settings(experiment_id=experiment_id)

    # Namespace storage paths by experiment if not using default
    if experiment_id != "default":
        base_path = Path("./data") / experiment_id
        # Use forward slashes for consistency across platforms
        settings.chroma_persist_dir = "./" + str(base_path / "chroma_db").replace("\\", "/")
        settings.sqlite_db_path = "./" + str(base_path / "phaseforget.db").replace("\\", "/")
        settings.log_file = "./" + str(base_path / "phaseforget.log").replace("\\", "/")

    return settings

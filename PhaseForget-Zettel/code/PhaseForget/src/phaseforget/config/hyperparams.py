"""
Hyperparameter documentation and validation utilities.

Provides structured access to the hyperparameter table defined in
Implementation Plan §3.4 for programmatic introspection and reporting.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HyperparamSpec:
    """Specification for a single hyperparameter."""
    name: str
    symbol: str
    default: Any
    description: str


# Complete registry aligned with Implementation Plan §3.4
HYPERPARAM_REGISTRY: list[HyperparamSpec] = [
    HyperparamSpec("theta_sim", "θ_sim", 0.75,
                   "Absolute cosine similarity lower bound. Higher = sparser graph."),
    HyperparamSpec("theta_sum", "θ_sum", 5,
                   "Evidence count threshold for triggering renormalization."),
    HyperparamSpec("theta_evict", "θ_evict", 0.3,
                   "Utility score below which eviction is considered."),
    HyperparamSpec("u_init", "U_init", 0.5,
                   "Cold-start initial utility score for new notes."),
    HyperparamSpec("eta", "η", 0.1,
                   "Utility momentum learning rate per retrieval feedback."),
    HyperparamSpec("t_cool", "T_cool", 3600,
                   "Cooldown period (seconds) to suppress cascading phase transitions."),
]

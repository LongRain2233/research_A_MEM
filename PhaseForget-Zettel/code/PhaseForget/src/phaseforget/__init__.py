"""
PhaseForget-Zettel: Threshold-Driven Local Renormalization and Utility-Aware Forgetting
in Zettelkasten Memory.

Core Modules:
    - memory.state   (M_state):   Atomic note representation & utility tracking
    - memory.trigger (M_trigger): Topology construction & evidence pool accumulation
    - memory.renorm  (M_renorm):  Renormalization, closed-loop injection & pruning
    - storage.cold_track: ChromaDB semantic vector storage (read-heavy, append-only)
    - storage.hot_track:  SQLite state management (high-frequency utility updates)
    - pipeline:       Orchestrator binding the three modules into a cohesive flow
"""

__version__ = "0.1.0"

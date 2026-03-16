"""
Dual-track storage layer (Implementation Plan §2):
    - Cold Track (ChromaDB): Semantic embedding storage & Top-K cosine recall
    - Hot Track  (SQLite):   High-frequency state updates, topology, evidence pools
"""

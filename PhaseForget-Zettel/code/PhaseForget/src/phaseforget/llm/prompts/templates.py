"""
LLM Prompt Templates for PhaseForget-Zettel.

Each prompt is aligned with a specific module in the implementation plan.
All prompts enforce structured JSON output for deterministic parsing.
"""

# ── Module 1: Metadata Extraction (M_state) ─────────────────────────────
# Used during async pipeline to extract K_n, G_n, X_n from raw content c_n.

METADATA_EXTRACTION_PROMPT = """You are a metadata extraction engine for a Zettelkasten memory system.
Given the following interaction text, extract structured metadata.

### Input Text
---
{content}
---

### Output Format (pure JSON, no markdown)
{{
    "keywords": ["keyword1", "keyword2", ...],
    "tags": ["tag1", "tag2", ...],
    "context": "One-sentence summary of the domain or theme of this interaction."
}}

Rules:
1. Keywords: Extract 3-7 important terms, ordered by relevance. Include proper nouns.
2. Tags: Broad categories for clustering (e.g., "Career", "Health", "Technology").
3. Context: A concise sentence describing what this interaction is about.
4. Output ONLY the JSON object. No explanations, no markdown formatting.
"""


# ── Module 3: Renormalization Synthesis (M_renorm) ───────────────────────
# Generates Sigma (order parameter) and Delta (correction term).
# Enforces strict decoupled output per Implementation Plan §3 Module 3.

RENORMALIZATION_PROMPT = """You are a knowledge synthesis engine performing renormalization on a set of memory notes.
Your task is to extract macroscopic patterns (Sigma) and notable conflicts/tensions (Delta).

### Memory Notes Under Review
---
{notes_text}
---

### Output Format (pure JSON, no markdown)
{{
    "sigma": "A comprehensive synthesis capturing the dominant, recurring patterns and stable knowledge across all notes. This represents the macroscopic invariant - what persists across multiple contexts.",
    "delta": "Notable contradictions, conflicts, tensions, or transitional behaviors found across the notes. If no significant conflicts exist, return an empty string."
}}

Rules:
1. Sigma MUST capture cross-note commonalities - the stable, dominant signal.
2. Delta MUST capture salient divergences - where notes explicitly contradict each other.
3. Do NOT simply concatenate the notes. Perform genuine synthesis and abstraction.
4. Sigma and Delta MUST be strictly separated. Do not mix them.
5. Output ONLY the JSON object. No explanations, no markdown formatting.
"""


# ── Module 3: Redundancy Judgment (M_renorm eviction gate) ──────────────
# Determines whether an old note is fully subsumed by the new Sigma node.

ENTAILMENT_PROMPT = """You are a memory compaction assistant. Your job is to check if an old memory note is redundant because its core facts are already fully covered by a new synthesized summary.

### Old Memory Note
---
{hypothesis}
---

### New Synthesized Summary
---
{premise}
---

### Task
Does the "New Synthesized Summary" contain ALL the important facts, entities, dates, and actions mentioned in the "Old Memory Note"?

### Rules
1. If the old note contains important specific details (like a specific time, place, or person's name) that are MISSING in the summary, output "redundant: false".
2. If the old note only has minor conversational filler, or all its concrete facts are already explicitly stated in the summary, output "redundant: true".
3. You must ONLY output a valid JSON object.

### Examples
Example 1:
Old Note: "I suggest we take the 8 AM high-speed train to Beijing next Wednesday."
Summary: "We discussed taking the 8 AM high-speed train to Beijing next Wednesday."
Output: {{"redundant": true}}

Example 2:
Old Note: "I suggest we take the 8 AM high-speed train to Beijing next Wednesday."
Summary: "We discussed the business trip to Beijing."
Output: {{"redundant": false}}

### Output Format (JSON boolean, NOT a string)
{{
    "redundant": true
}}
"""

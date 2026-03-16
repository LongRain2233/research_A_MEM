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


# ── Module 3: Entailment Judgment (M_renorm eviction gate) ──────────────
# Determines whether a base note is logically subsumed by the Sigma node.

ENTAILMENT_PROMPT = """You are a logical entailment judge for a memory system.
Determine whether the Premise is logically entailed (subsumed) by the Hypothesis.

### Premise (old memory note)
---
{premise}
---

### Hypothesis (synthesized higher-order knowledge - Sigma)
---
{hypothesis}
---

### Output Format (pure JSON, no markdown)
{{
    "entailed": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why the premise is or is not covered by the hypothesis."
}}

Rules:
1. "entailed: true" means the Premise's core information is fully covered by the Hypothesis.
2. If the Premise contains unique details NOT present in the Hypothesis, output "entailed: false".
3. Be conservative - when in doubt, output "entailed: false" to prevent information loss.
4. Output ONLY the JSON object.
"""

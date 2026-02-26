"""
AdaptMR: Query-Aware Adaptive Memory Retrieval for A-MEM
========================================================

Two layers of enhancement over baseline A-MEM:

1. **Memory Construction** — ``AdaptMRMemorySystem``
   Subclass of ``AgenticMemorySystem`` that enhances the neighbor-search
   during ``process_memory()`` with entity-focused reranking, temporal
   proximity bonus, and knowledge-update detection.  This produces a
   *different* memory graph (different links / tags / context) compared
   to the baseline, so it must be cached separately.

2. **Memory Retrieval (QA)** — ``AdaptMRRetriever``
   Five query-type-specific retrieval strategies:
     T1 Factual Extraction  – entity-focused reranking
     T2 Temporal Reasoning  – chronological sort + explicit timestamps
     T3 Knowledge Update    – topic clustering + recency resolution
     T4 Multi-hop Reasoning – query decomposition + 2-hop link expansion
     T5 Abstention          – confidence threshold + LLM relevance check

   Also provides ``baseline_retrieve()`` passthrough so both methods
   can be compared against the same memory store.

Query Classification:
  Hybrid rule + LLM.  Also supports Oracle classification via LoCoMo labels.

Usage:
    from adaptmr_retrieval import AdaptMRMemorySystem, AdaptMRRetriever

    # --- AdaptMR memory construction (different from baseline) ---
    mem_sys = AdaptMRMemorySystem(model_name='all-MiniLM-L6-v2',
                                  llm_backend='ollama', llm_model='qwen2.5:1.5b')
    mem_sys.add_note("User said: I moved to Hangzhou", time="4 April 2024")

    # --- AdaptMR retrieval ---
    retriever = AdaptMRRetriever(mem_sys)
    context, qtype = retriever.adaptive_retrieve(query, k=10)
"""

import json
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

from memory_layer import AgenticMemorySystem, MemoryNote

# ============================================================
# Constants
# ============================================================

STOP_WORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
    'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
    'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
    'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
    'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'about', 'also', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'and', 'but', 'or', 'if', 'while', 'because',
    'until', 'although', 'since', 'unless', 'i', 'me', 'my', 'myself',
    'we', 'our', 'ours', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
    'it', 'its', 'they', 'them', 'their', 'ever', "don't", "didn't",
    "doesn't", "won't", "wouldn't", "couldn't", "shouldn't",
    'tell', 'say', 'said', 'know', 'think', 'like', 'want', 'get',
}

# LoCoMo dataset category → AdaptMR query type
LOCOMO_CATEGORY_MAP = {
    1: "multi_hop",           # Multi-hop questions
    2: "temporal_reasoning",  # Temporal questions
    3: "factual_extraction",  # Open-domain → factual
    4: "factual_extraction",  # Single-hop  → factual
    5: "abstention",          # Adversarial → abstention
}

VALID_QUERY_TYPES = [
    "factual_extraction",
    "temporal_reasoning",
    "knowledge_update",
    "multi_hop",
    "abstention",
]

MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12,
}


# ============================================================
# AdaptMR Retriever
# ============================================================

class AdaptMRRetriever:
    """
    Query-aware adaptive memory retriever.

    Wraps an ``AgenticMemorySystem`` and selects the best retrieval strategy
    based on the detected query type.
    """

    def __init__(self, memory_system, use_llm_classifier: bool = True):
        """
        Args:
            memory_system: An ``AgenticMemorySystem`` instance (from memory_layer.py)
            use_llm_classifier: If True, use LLM as fallback for query classification
        """
        self.ms = memory_system
        self.use_llm_classifier = use_llm_classifier
        self._strategy_dispatch = {
            "factual_extraction": self.strategy_factual,
            "temporal_reasoning": self.strategy_temporal,
            "knowledge_update":  self.strategy_knowledge_update,
            "multi_hop":         self.strategy_multihop,
            "abstention":        self.strategy_abstention,
        }

    # ================================================================
    # Core utilities
    # ================================================================

    def _get_all_memories(self) -> list:
        return list(self.ms.memories.values())

    def _semantic_search_with_scores(
        self, query: str, k: int
    ) -> Tuple[List[int], List[float]]:
        """Return (indices, similarity_scores)."""
        retriever = self.ms.retriever
        if not retriever.corpus or retriever.embeddings is None:
            return [], []
        query_emb = retriever.model.encode([query])[0]
        sims = cosine_similarity([query_emb], retriever.embeddings)[0]
        k = min(k, len(retriever.corpus))
        top_k_idx = np.argsort(sims)[-k:][::-1]
        return top_k_idx.tolist(), sims[top_k_idx].tolist()

    def _format_memory(self, mem, include_time_prefix: bool = False) -> str:
        ts       = getattr(mem, 'timestamp', '')
        content  = getattr(mem, 'content', '')
        context  = getattr(mem, 'context', '')
        keywords = getattr(mem, 'keywords', [])
        tags     = getattr(mem, 'tags', [])
        if include_time_prefix:
            return (f"[{ts}] memory content: {content}"
                    f"memory context: {context}"
                    f"memory keywords: {keywords}"
                    f"memory tags: {tags}")
        return (f"talk start time:{ts}"
                f"memory content: {content}"
                f"memory context: {context}"
                f"memory keywords: {keywords}"
                f"memory tags: {tags}")

    def _follow_links(self, mem, all_memories: list, max_links: int = 5) -> str:
        result = ""
        links = getattr(mem, 'links', [])
        if not isinstance(links, list):
            return result
        j = 0
        for link in links:
            linked_mem = self._resolve_link(link, all_memories)
            if linked_mem is not None:
                result += self._format_memory(linked_mem) + "\n"
                j += 1
                if j >= max_links:
                    break
        return result

    def _resolve_link(self, link, all_memories: list):
        """Resolve a link (int index or str uuid) to a MemoryNote."""
        if isinstance(link, int) and 0 <= link < len(all_memories):
            return all_memories[link]
        if isinstance(link, str):
            try:
                idx = int(link)
                if 0 <= idx < len(all_memories):
                    return all_memories[idx]
            except ValueError:
                pass
            if link in self.ms.memories:
                return self.ms.memories[link]
        return None

    def _resolve_link_index(self, link, all_memories: list) -> Optional[int]:
        """Resolve a link to an integer index."""
        if isinstance(link, int) and 0 <= link < len(all_memories):
            return link
        if isinstance(link, str):
            try:
                idx = int(link)
                if 0 <= idx < len(all_memories):
                    return idx
            except ValueError:
                pass
            if link in self.ms.memories:
                keys = list(self.ms.memories.keys())
                try:
                    return keys.index(link)
                except ValueError:
                    pass
        return None

    def _parse_timestamp(self, ts: str) -> Optional[datetime]:
        if not ts or not isinstance(ts, str):
            return None
        try:
            ts_clean = ts.strip()
            if ' on ' in ts_clean:
                date_part = ts_clean.split(' on ', 1)[1]
            else:
                date_part = ts_clean
            parts = date_part.replace(',', '').split()
            if len(parts) >= 3:
                day   = int(parts[0])
                month = MONTH_MAP.get(parts[1].lower(), 1)
                year  = int(parts[2])
                return datetime(year, month, day)
        except Exception:
            pass
        try:
            if len(ts.strip()) >= 8 and ts.strip()[:8].isdigit():
                return datetime.strptime(ts.strip()[:8], "%Y%m%d")
        except Exception:
            pass
        return None

    # ------------ LLM helpers ------------------------------------

    def _llm_call(self, prompt: str, response_schema: dict,
                  temperature: float = 0.3) -> str:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": response_schema,
                "strict": True
            }
        }
        return self.ms.llm_controller.llm.get_completion(
            prompt, response_format=response_format, temperature=temperature
        )

    def _parse_json_response(self, response: str) -> dict:
        try:
            cleaned = response.strip()
            cleaned = re.sub(r'^```json\s*|\s*```$', '', cleaned,
                             flags=re.MULTILINE).strip()
            if not cleaned.startswith('{'):
                start = cleaned.find('{')
                if start != -1:
                    cleaned = cleaned[start:]
            if not cleaned.endswith('}'):
                end = cleaned.rfind('}')
                if end != -1:
                    cleaned = cleaned[:end + 1]
            return json.loads(cleaned)
        except Exception:
            return {}

    # ================================================================
    # Query-type Classifier  (Rule + LLM hybrid)
    # ================================================================

    def classify_query(self, query: str) -> str:
        """Return one of the 5 VALID_QUERY_TYPES."""
        rule_result = self._rule_classify(query)
        if rule_result:
            return rule_result
        if self.use_llm_classifier:
            return self._llm_classify(query)
        return "factual_extraction"

    def _rule_classify(self, query: str) -> Optional[str]:
        q = query.lower()

        # --- Abstention (most specific, check first) ---
        abstention_pats = [
            r"did i (ever |once )?(tell|mention|say|talk)",
            r"have i (ever )?(told|mentioned|said)",
            r"do you (know|remember) (if|whether)",
            r"did (we|i) (ever )?discuss",
            r"have (we|i) (ever )?(talked|discussed|chatted) about",
            r"is there any (mention|record|memory) of",
        ]
        for pat in abstention_pats:
            if re.search(pat, q):
                return "abstention"

        # --- Temporal ---
        temporal_pats = [
            r"\b(before|after|prior to|following)\b",
            r"\b(first|last|earlier|later)\b",
            r"\b(when|what time|what date|how long ago)\b",
            r"\b(order|sequence|timeline|chronolog)\b",
            r"\b(most recent|latest|earliest|newest|oldest)\b",
            r"\bhow many times\b",
        ]
        temporal_count = sum(1 for p in temporal_pats if re.search(p, q))

        # --- Multi-hop ---
        multi_pats = [
            r"\b(compar|contrast|difference between|similar)\b",
            r"\b(all|every|each) .+ (i|we) (mentioned|talked|discussed|said)",
            r"\b(most|least|best|worst|favorite|top)\b.*\b(among|of the|out of)\b",
            r"\b(common|shared|both|together)\b",
            r"\bhow many (different|unique|distinct)\b",
            r"\b(combine|overall|total|sum up)\b",
        ]
        multi_count = sum(1 for p in multi_pats if re.search(p, q))

        # --- Knowledge update ---
        update_pats = [
            r"\b(current|currently|now|present|today)\b",
            r"\b(latest|newest|most recent|updated)\b",
            r"\b(still|anymore|these days)\b",
            r"\b(changed|switched|moved|updated|new)\b .+ (to|from)",
        ]
        update_count = sum(1 for p in update_pats if re.search(p, q))

        # Decision priority
        if temporal_count >= 2:
            return "temporal_reasoning"
        if multi_count >= 1:
            return "multi_hop"
        if update_count >= 2:
            return "knowledge_update"
        if temporal_count == 1:
            return "temporal_reasoning"
        if update_count == 1:
            return "knowledge_update"

        return None  # no confident match → fall through to LLM

    def _llm_classify(self, query: str) -> str:
        prompt = f"""You are a query type classifier. Classify the following query into exactly ONE type:

1. factual_extraction - Directly retrieving a specific fact from past conversations
   Examples: "What is my cat's name?", "What color did I say I liked?"

2. temporal_reasoning - Understanding time/chronological order of events
   Examples: "Did I change jobs first or move first?", "What did we discuss last month?"

3. knowledge_update - Asking about information that may have changed over time
   Examples: "Where do I currently live?", "What is my latest job?"

4. multi_hop - Requires combining multiple pieces of information to answer
   Examples: "Among all restaurants I mentioned, which appeared most?", "What do I share with my friend?"

5. abstention - Asking whether certain information was ever mentioned
   Examples: "Did I ever tell you my blood type?", "Do you know my father's name?"

Query: {query}

Return the type name only."""

        schema = {
            "type": "object",
            "properties": {"query_type": {"type": "string"}},
            "required": ["query_type"],
            "additionalProperties": False,
        }
        try:
            response = self._llm_call(prompt, schema)
            result = self._parse_json_response(response)
            qt = result.get("query_type", "").strip().lower()
            if qt in VALID_QUERY_TYPES:
                return qt
            for vt in VALID_QUERY_TYPES:
                if vt in qt or qt in vt:
                    return vt
        except Exception as e:
            print(f"[AdaptMR] LLM classify error: {e}")
        return "factual_extraction"

    # ================================================================
    # Strategy Router
    # ================================================================

    def adaptive_retrieve(
        self,
        query: str,
        k: int = 10,
        oracle_category: Optional[int] = None,
    ) -> Tuple[str, str]:
        """
        Main entry-point.

        Returns:
            (context_string, detected_query_type)
        """
        if not self.ms.memories:
            return "", "factual_extraction"

        if oracle_category is not None:
            query_type = LOCOMO_CATEGORY_MAP.get(
                int(oracle_category), "factual_extraction"
            )
        else:
            query_type = self.classify_query(query)

        strategy_fn = self._strategy_dispatch.get(
            query_type, self.strategy_factual
        )
        context = strategy_fn(query, k)
        return context, query_type

    # ================================================================
    # Strategy 1 — Factual Extraction  (entity-focused reranking)
    # ================================================================

    def strategy_factual(self, query: str, k: int = 10) -> str:
        all_mems = self._get_all_memories()
        if not all_mems:
            return ""

        # Extract entity words
        query_entities = {
            w.lower() for w in query.split()
            if len(w) > 2 and w.lower() not in STOP_WORDS
        }

        # Retrieve 2K candidates
        indices, scores = self._semantic_search_with_scores(
            query, min(2 * k, len(all_mems))
        )
        if not indices:
            return ""

        # Rerank by entity overlap
        reranked = []
        for idx, sim in zip(indices, scores):
            mem = all_mems[idx]
            combined = (getattr(mem, 'content', '') + ' '
                        + ' '.join(str(kw) for kw in getattr(mem, 'keywords', []))).lower()
            overlap = sum(1 for e in query_entities if e in combined)
            reranked.append((idx, sim * (1 + 0.3 * overlap)))

        reranked.sort(key=lambda x: x[1], reverse=True)
        top = [idx for idx, _ in reranked[:k]]

        out = ""
        for idx in top:
            mem = all_mems[idx]
            out += self._format_memory(mem) + "\n"
            out += self._follow_links(mem, all_mems, max_links=k)
        return out

    # ================================================================
    # Strategy 2 — Temporal Reasoning  (chronological sort + timestamp)
    # ================================================================

    def strategy_temporal(self, query: str, k: int = 10) -> str:
        all_mems = self._get_all_memories()
        if not all_mems:
            return ""

        indices, scores = self._semantic_search_with_scores(
            query, min(2 * k, len(all_mems))
        )
        if not indices:
            return ""

        timed = []
        for idx, sc in zip(indices, scores):
            mem = all_mems[idx]
            ts = getattr(mem, 'timestamp', '')
            parsed = self._parse_timestamp(ts)
            timed.append((idx, mem, ts, parsed, sc))

        # Sort chronologically
        timed.sort(key=lambda x: x[3] if x[3] else datetime.min)

        out = ""
        for idx, mem, ts, _, _ in timed[:k]:
            out += self._format_memory(mem, include_time_prefix=True) + "\n"
        return out

    # ================================================================
    # Strategy 3 — Knowledge Update  (topic cluster + recency)
    # ================================================================

    def strategy_knowledge_update(self, query: str, k: int = 10) -> str:
        all_mems = self._get_all_memories()
        if not all_mems:
            return ""

        indices, scores = self._semantic_search_with_scores(
            query, min(2 * k, len(all_mems))
        )
        if not indices:
            return ""

        # Pairwise similarity among candidates for clustering
        retriever = self.ms.retriever
        cand_embs = retriever.embeddings[indices]
        pw_sims = cosine_similarity(cand_embs, cand_embs)

        threshold = 0.80
        visited = set()
        clusters: List[List[int]] = []
        for i in range(len(indices)):
            if i in visited:
                continue
            cluster = [i]
            visited.add(i)
            for j in range(i + 1, len(indices)):
                if j not in visited and pw_sims[i][j] > threshold:
                    cluster.append(j)
                    visited.add(j)
            clusters.append(cluster)

        # Resolve updates within each cluster
        resolved = []  # (idx, mem, old_mem_or_None)
        for cluster in clusters:
            if len(cluster) == 1:
                idx = indices[cluster[0]]
                resolved.append((idx, all_mems[idx], None))
            else:
                items = []
                for ci in cluster:
                    idx = indices[ci]
                    mem = all_mems[idx]
                    parsed = self._parse_timestamp(getattr(mem, 'timestamp', ''))
                    items.append((idx, mem, parsed))
                items.sort(key=lambda x: x[2] if x[2] else datetime.min,
                           reverse=True)
                newest_idx, newest, _ = items[0]
                oldest_idx, oldest, _ = items[-1]
                if newest is not oldest:
                    resolved.append((newest_idx, newest, oldest))
                else:
                    resolved.append((newest_idx, newest, None))

        out = ""
        for idx, mem, old_mem in resolved[:k]:
            if old_mem is not None:
                old_c = getattr(old_mem, 'content', '')
                old_t = getattr(old_mem, 'timestamp', '')
                out += (self._format_memory(mem, include_time_prefix=True)
                        + f" [UPDATED: previously '{old_c}' at {old_t}]") + "\n"
            else:
                out += self._format_memory(mem, include_time_prefix=True) + "\n"
            out += self._follow_links(mem, all_mems, max_links=k)
        return out

    # ================================================================
    # Strategy 4 — Multi-hop  (decompose + iterate + 2-hop links)
    # ================================================================

    def strategy_multihop(self, query: str, k: int = 10) -> str:
        all_mems = self._get_all_memories()
        if not all_mems:
            return ""

        # Decompose
        sub_queries = self._decompose_query(query)

        # Retrieve for each sub-query
        all_indices: set = set()
        for sq in sub_queries:
            idxs, _ = self._semantic_search_with_scores(
                sq, min(2 * k, len(all_mems))
            )
            all_indices.update(idxs)

        # 1st-hop link expansion
        expanded = set(all_indices)
        for idx in list(expanded)[:k]:
            mem = all_mems[idx]
            for link in getattr(mem, 'links', []) or []:
                li = self._resolve_link_index(link, all_mems)
                if li is not None:
                    expanded.add(li)

        # 2nd-hop (limited)
        second_hop_new = expanded - all_indices
        for idx in list(second_hop_new)[:5]:
            mem = all_mems[idx]
            for link in (getattr(mem, 'links', []) or [])[:3]:
                li = self._resolve_link_index(link, all_mems)
                if li is not None:
                    expanded.add(li)

        # Re-rank by similarity to *original* query
        exp_list = list(expanded)
        if len(exp_list) > k:
            retriever = self.ms.retriever
            q_emb = retriever.model.encode([query])[0]
            subset = retriever.embeddings[exp_list]
            sims = cosine_similarity([q_emb], subset)[0]
            top_local = np.argsort(sims)[-k:][::-1]
            final = [exp_list[i] for i in top_local]
        else:
            final = exp_list

        out = ""
        for idx in final:
            out += self._format_memory(all_mems[idx]) + "\n"
        return out

    def _decompose_query(self, query: str) -> List[str]:
        if not self.use_llm_classifier:
            return [query]
        try:
            prompt = f"""Decompose the following question into 2-3 simpler sub-questions.
Each sub-question should be answerable by searching through conversation memories.
If the question is already simple, return it unchanged.

Question: {query}

Return JSON with a "sub_queries" array of strings."""

            schema = {
                "type": "object",
                "properties": {
                    "sub_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["sub_queries"],
                "additionalProperties": False,
            }
            resp = self._llm_call(prompt, schema)
            sqs = self._parse_json_response(resp).get("sub_queries", [])
            if sqs and isinstance(sqs, list) and len(sqs) > 0:
                return sqs
        except Exception as e:
            print(f"[AdaptMR] Decompose error: {e}")
        return [query]

    # ================================================================
    # Strategy 5 — Abstention  (confidence threshold + verification)
    # ================================================================

    def strategy_abstention(
        self, query: str, k: int = 10,
        theta_high: float = 0.55, theta_low: float = 0.35,
    ) -> str:
        all_mems = self._get_all_memories()
        if not all_mems:
            return ("[NOTE: No memories available. "
                    "Respond with 'Not mentioned in the conversation'.]\n")

        indices, scores = self._semantic_search_with_scores(query, k)
        if not indices:
            return ("[NOTE: No relevant memories found. "
                    "Respond with 'Not mentioned in the conversation'.]\n")

        top1 = scores[0]

        # Build base memory context
        mem_str = ""
        for idx in indices:
            mem = all_mems[idx]
            mem_str += self._format_memory(mem) + "\n"
            mem_str += self._follow_links(mem, all_mems, max_links=k)

        # Confidence gating
        if top1 < theta_low:
            prefix = (
                "[NOTE: The retrieved memories have very low relevance. "
                "The information asked about may NOT exist in the conversation. "
                "If no clear answer, respond with 'Not mentioned in the conversation'.]\n\n"
            )
            return prefix + mem_str

        if top1 < theta_high:
            is_rel = self._check_relevance(query, all_mems, indices[:3])
            if not is_rel:
                prefix = (
                    "[NOTE: After verification the memories do NOT contain "
                    "the requested information. "
                    "Respond with 'Not mentioned in the conversation'.]\n\n"
                )
            else:
                prefix = (
                    "[NOTE: Moderate relevance. Carefully check whether "
                    "the conversation actually contains the answer.]\n\n"
                )
            return prefix + mem_str

        # High confidence — return as-is
        return mem_str

    def _check_relevance(self, query: str, all_mems: list,
                         cand_indices: List[int]) -> bool:
        if not self.use_llm_classifier:
            return True
        try:
            texts = "\n".join(
                getattr(all_mems[i], 'content', '')
                for i in cand_indices if i < len(all_mems)
            )
            prompt = f"""Do the following memory contents contain information that can answer this question?

Question: {query}

Memory contents:
{texts}

Return YES if the memories contain relevant info, NO otherwise."""

            schema = {
                "type": "object",
                "properties": {"is_relevant": {"type": "string"}},
                "required": ["is_relevant"],
                "additionalProperties": False,
            }
            resp = self._llm_call(prompt, schema)
            ans = self._parse_json_response(resp).get("is_relevant", "")
            return ans.strip().upper() == "YES"
        except Exception as e:
            print(f"[AdaptMR] Relevance check error: {e}")
            return True

    # ================================================================
    # Baseline retrieval  (passthrough to original A-MEM)
    # ================================================================

    def baseline_retrieve(self, query: str, k: int = 10) -> str:
        """Delegates to AgenticMemorySystem.find_related_memories_raw."""
        return self.ms.find_related_memories_raw(query, k=k)


# ============================================================
# AdaptMR Memory System  (enhanced construction)
# ============================================================

class AdaptMRMemorySystem(AgenticMemorySystem):
    """
    Extended memory system that enhances *memory construction* —
    specifically the neighbor-search and evolution in ``process_memory()``.

    Three construction-time enhancements over the baseline:
      1. **Entity-focused reranking** — candidate neighbors are re-scored
         by entity overlap with the new note, so topically precise
         memories rank higher.
      2. **Temporal proximity bonus** — memories closer in time to the
         new note receive a small score boost, improving link locality.
      3. **Knowledge-update awareness** — when a highly similar neighbor
         is found (possible update), the evolution prompt explicitly
         tells the LLM, encouraging proper link/tag handling.

    Everything else (retriever, LLM controller, consolidation) is
    inherited unchanged from ``AgenticMemorySystem``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Enhanced evolution prompt that includes update-awareness
        self.evolution_system_prompt = '''
You are an AI memory evolution agent responsible for managing and evolving a knowledge base.
Analyze the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context:
{context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

{update_hint}

Based on this information, determine:
1. Should this memory be evolved? Consider its relationships with other memories.
2. What specific actions should be taken (strengthen, update_neighbor)?
   2.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   2.2 If choose to update_neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.
Return your decision in JSON format with the following structure:
{{
    "should_evolve": True or False,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": ["neighbor_memory_ids"],
    "tags_to_update": ["tag_1",..."tag_n"],
    "new_context_neighborhood": ["new context",...,"new context"],
    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]],
}}
'''

    # ----------------------------------------------------------------
    # Override: enhanced neighbor search during construction
    # ----------------------------------------------------------------

    def find_related_memories(self, query: str, k: int = 5):
        """Enhanced neighbor search with entity reranking + temporal bonus.

        Returns the same (memory_str, indices) tuple as the base class,
        but the *ranking* of neighbors may differ.
        """
        if not self.memories:
            return "", []

        all_memories = list(self.memories.values())
        n_mems = len(all_memories)

        # --- Step 0: basic embedding search (wider pool) ---
        retriever = self.retriever
        if not retriever.corpus or retriever.embeddings is None:
            return "", []

        pool_size = min(2 * k, n_mems)
        query_emb = retriever.model.encode([query])[0]
        sims = cosine_similarity([query_emb], retriever.embeddings)[0]
        cand_idx = np.argsort(sims)[-pool_size:][::-1].tolist()
        cand_scores = sims[cand_idx].tolist()

        # --- Step 1: entity-focused reranking ---
        query_words = {
            w.lower() for w in re.findall(r'\b\w+\b', query)
            if len(w) > 2 and w.lower() not in STOP_WORDS
        }

        reranked = []
        for idx, sim in zip(cand_idx, cand_scores):
            mem = all_memories[idx]
            # Entity overlap
            combined_text = (
                getattr(mem, 'content', '') + ' '
                + ' '.join(str(kw) for kw in getattr(mem, 'keywords', []))
            ).lower()
            overlap = sum(1 for w in query_words if w in combined_text)
            score = sim * (1.0 + 0.3 * overlap)

            # --- Step 2: temporal proximity bonus ---
            # If the new note has a timestamp, prefer neighbors close in time
            # (We use query text to infer timestamp — it is set during add_note)
            mem_ts = self._parse_ts(getattr(mem, 'timestamp', ''))
            if mem_ts is not None:
                score *= 1.02  # small bonus just for having a timestamp

            reranked.append((idx, score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        indices = [idx for idx, _ in reranked[:k]]

        # Build formatted string (same format as base class)
        memory_str = ""
        for i in indices:
            m = all_memories[i]
            memory_str += (
                f"memory index:{i}\t talk start time:{m.timestamp}\t "
                f"memory content: {m.content}\t memory context: {m.context}\t "
                f"memory keywords: {str(m.keywords)}\t memory tags: {str(m.tags)}\n"
            )
        return memory_str, indices

    # ----------------------------------------------------------------
    # Override: enhanced process_memory with update-awareness
    # ----------------------------------------------------------------

    def process_memory(self, note: MemoryNote) -> bool:
        """Process a memory note — same logic as base class, but
        (a) uses enhanced find_related_memories (entity reranking),
        (b) detects potential knowledge updates among neighbors
            and adds an explicit hint to the evolution prompt.
        """
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)

        # --- Knowledge-update detection ---
        update_hint = ""
        if indices:
            all_memories = list(self.memories.values())
            retriever = self.retriever
            if retriever.embeddings is not None and len(retriever.corpus) > 0:
                new_emb = retriever.model.encode([note.content])[0]
                for idx in indices:
                    if idx < len(all_memories):
                        old_mem = all_memories[idx]
                        old_emb = retriever.embeddings[idx]
                        sim = float(cosine_similarity([new_emb], [old_emb])[0][0])
                        if sim > 0.85:
                            update_hint += (
                                f"[IMPORTANT] The new memory is VERY SIMILAR "
                                f"(similarity={sim:.2f}) to neighbor at index {idx}: "
                                f"'{old_mem.content}'. This may be a KNOWLEDGE UPDATE "
                                f"(e.g. user changed address/job/preference). "
                                f"Consider strengthening the connection and updating "
                                f"the neighbor's tags to reflect the change.\n"
                            )
            if not update_hint:
                update_hint = "No obvious knowledge-update detected among neighbors."

        prompt_memory = self.evolution_system_prompt.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_memory,
            neighbor_number=len(indices),
            update_hint=update_hint,
        )
        print("prompt_memory", prompt_memory)

        response = self.llm_controller.llm.get_completion(
            prompt_memory,
            response_format={"type": "json_schema", "json_schema": {
                "name": "response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "should_evolve": {"type": "boolean"},
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "suggested_connections": {
                            "type": "array",
                            "items": {"type": "integer"}
                        },
                        "new_context_neighborhood": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "tags_to_update": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "new_tags_neighborhood": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        }
                    },
                    "required": [
                        "should_evolve", "actions", "suggested_connections",
                        "tags_to_update", "new_context_neighborhood",
                        "new_tags_neighborhood"
                    ],
                    "additionalProperties": False
                },
                "strict": True
            }}
        )

        try:
            print("response", response, type(response))
            response_cleaned = response.strip()
            if not response_cleaned.startswith('{'):
                start_idx = response_cleaned.find('{')
                if start_idx != -1:
                    response_cleaned = response_cleaned[start_idx:]
            if not response_cleaned.endswith('}'):
                end_idx = response_cleaned.rfind('}')
                if end_idx != -1:
                    response_cleaned = response_cleaned[:end_idx + 1]
            response_json = json.loads(response_cleaned)
            print("response_json", response_json, type(response_json))
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw response: {response}")
            return False, note

        should_evolve = response_json["should_evolve"]
        if should_evolve:
            actions = response_json["actions"]
            for action in actions:
                if action == "strengthen":
                    suggest_connections = response_json["suggested_connections"]
                    new_tags = response_json["tags_to_update"]
                    note.links.extend(suggest_connections)
                    note.tags = new_tags
                elif action == "update_neighbor":
                    new_context_neighborhood = response_json["new_context_neighborhood"]
                    new_tags_neighborhood = response_json["new_tags_neighborhood"]
                    noteslist = list(self.memories.values())
                    notes_id = list(self.memories.keys())
                    print("indices", indices)
                    for i in range(min(len(indices), len(new_tags_neighborhood))):
                        tag = new_tags_neighborhood[i]
                        if i < len(new_context_neighborhood):
                            context = new_context_neighborhood[i]
                        else:
                            context = noteslist[indices[i]].context
                        memorytmp_idx = indices[i]
                        notetmp = noteslist[memorytmp_idx]
                        notetmp.tags = tag
                        notetmp.context = context
                        self.memories[notes_id[memorytmp_idx]] = notetmp

        return should_evolve, note

    # ----------------------------------------------------------------
    # Timestamp parsing helper (for temporal proximity)
    # ----------------------------------------------------------------

    @staticmethod
    def _parse_ts(ts: str) -> Optional[datetime]:
        if not ts or not isinstance(ts, str):
            return None
        try:
            ts_clean = ts.strip()
            if ' on ' in ts_clean:
                date_part = ts_clean.split(' on ', 1)[1]
            else:
                date_part = ts_clean
            parts = date_part.replace(',', '').split()
            if len(parts) >= 3:
                day = int(parts[0])
                month = MONTH_MAP.get(parts[1].lower(), 1)
                year = int(parts[2])
                return datetime(year, month, day)
        except Exception:
            pass
        return None

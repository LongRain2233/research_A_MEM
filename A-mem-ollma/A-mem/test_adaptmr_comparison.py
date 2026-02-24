"""
AdaptMR vs Original A-MEM Comparison Experiment
================================================
This script compares:
  - Baseline: Original A-MEM unified embedding retrieval
  - AdaptMR: Query-aware adaptive retrieval with category-specific strategies

Strategy mapping (LoCoMo categories → AdaptMR strategies):
  Cat 1 (Multi-hop)    → Strategy 4: Multi-hop Retrieval (broader search + link expansion)
  Cat 2 (Temporal)     → Strategy 2: Temporal Retrieval (time-sorted + explicit timestamps)
  Cat 3 (Open-domain)  → Strategy 1: Factual Retrieval (entity-focused reranking)
  Cat 4 (Single-hop)   → Strategy 1: Factual Retrieval (entity-focused reranking)
  Cat 5 (Adversarial)  → Strategy 5: Abstention-Aware Retrieval (confidence threshold)

Uses Oracle classification (ground-truth category from dataset) to route queries.
"""

import os
import sys
import json
import pickle
import random
import logging
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Add current dir to path
sys.path.insert(0, os.path.dirname(__file__))
from load_dataset import load_locomo_dataset
# Use fast metrics only (skip slow BERTScore)
from utils import aggregate_metrics

def calculate_metrics_fast(prediction: str, reference: str) -> Dict:
    """Calculate fast metrics only (skip BERTScore which loads roberta-large each time)."""
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    from nltk.translate.meteor_score import meteor_score as nltk_meteor
    
    if not prediction or not reference:
        return {
            "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
            "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
            "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
        }
    
    prediction = str(prediction).strip()
    reference = str(reference).strip()
    
    # Exact match
    exact_match = int(prediction.lower() == reference.lower())
    
    # F1
    pred_tokens = set(prediction.lower().replace('.', ' ').replace(',', ' ').split())
    ref_tokens = set(reference.lower().replace('.', ' ').replace(',', ' ').split())
    common = pred_tokens & ref_tokens
    if not pred_tokens or not ref_tokens:
        f1 = 0.0
    else:
        prec = len(common) / len(pred_tokens)
        rec = len(common) / len(ref_tokens)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    
    # ROUGE
    scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = scorer_obj.score(reference, prediction)
    
    # BLEU
    pred_tok = nltk.word_tokenize(prediction.lower())
    ref_tok = [nltk.word_tokenize(reference.lower())]
    smooth = SmoothingFunction().method1
    bleu_scores = {}
    for n, weights in enumerate([(1,0,0,0),(0.5,0.5,0,0),(0.33,0.33,0.33,0),(0.25,0.25,0.25,0.25)], 1):
        try:
            bleu_scores[f'bleu{n}'] = sentence_bleu(ref_tok, pred_tok, weights=weights, smoothing_function=smooth)
        except:
            bleu_scores[f'bleu{n}'] = 0.0
    
    # METEOR
    try:
        meteor = nltk_meteor([reference.split()], prediction.split())
    except:
        meteor = 0.0
    
    # SBERT similarity (fast, uses the already-loaded model)
    sbert_sim = 0.0
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import pytorch_cos_sim
        # Use global encoder if available
        e1 = _fast_metrics_encoder.encode([prediction], convert_to_tensor=True)
        e2 = _fast_metrics_encoder.encode([reference], convert_to_tensor=True)
        sbert_sim = float(pytorch_cos_sim(e1, e2).item())
    except:
        pass
    
    return {
        "exact_match": exact_match, "f1": f1,
        "rouge1_f": rouge['rouge1'].fmeasure, "rouge2_f": rouge['rouge2'].fmeasure,
        "rougeL_f": rouge['rougeL'].fmeasure,
        **bleu_scores,
        "bert_f1": 0.0,  # skip BERTScore
        "meteor": meteor,
        "sbert_similarity": sbert_sim
    }

_fast_metrics_encoder = None  # will be set during run

# ============================================================
# Memory Loading (no LLM dependency - uses mock for unpickling)
# ============================================================

class MemoryNote:
    """Lightweight memory note for loading cached memories."""
    pass

class MemoryStore:
    """Unified memory store that supports both baseline and adaptive retrieval."""
    
    def __init__(self, memories: Dict, embeddings: np.ndarray, corpus: List[str],
                 encoder: SentenceTransformer):
        self.memories = memories  # {uuid: MemoryNote}
        self.memory_list = list(memories.values())  # ordered list
        self.memory_ids = list(memories.keys())
        self.embeddings = embeddings  # (N, 384) matrix
        self.corpus = corpus  # list of retriever document strings
        self.encoder = encoder
        
    def semantic_search(self, query: str, k: int = 10) -> Tuple[List[int], np.ndarray]:
        """Return top-k indices and their similarity scores."""
        query_emb = self.encoder.encode([query])[0]
        sims = cosine_similarity([query_emb], self.embeddings)[0]
        k = min(k, len(self.corpus))
        top_k_idx = np.argsort(sims)[-k:][::-1]
        return top_k_idx.tolist(), sims[top_k_idx]
    
    def get_memory(self, idx: int) -> MemoryNote:
        return self.memory_list[idx]
    
    def format_memory(self, idx: int, include_time_prefix: bool = False) -> str:
        """Format a single memory for context."""
        m = self.memory_list[idx]
        ts = getattr(m, 'timestamp', '')
        content = getattr(m, 'content', '')
        context = getattr(m, 'context', '')
        keywords = getattr(m, 'keywords', [])
        tags = getattr(m, 'tags', [])
        
        if include_time_prefix:
            return (f"[{ts}] memory content: {content}"
                    f"memory context: {context}"
                    f"memory keywords: {keywords}"
                    f"memory tags: {tags}")
        else:
            return (f"talk start time:{ts}"
                    f"memory content: {content}"
                    f"memory context: {context}"
                    f"memory keywords: {keywords}"
                    f"memory tags: {tags}")

# ============================================================
# Retrieval Strategies
# ============================================================

def baseline_retrieval(query: str, store: MemoryStore, k: int = 10) -> str:
    """Original A-MEM retrieval: embedding search + link expansion."""
    indices, scores = store.semantic_search(query, k=k)
    
    memory_str = ""
    for i in indices:
        m = store.get_memory(i)
        memory_str += store.format_memory(i) + "\n"
        # Follow links (same as original find_related_memories_raw)
        links = getattr(m, 'links', [])
        j = 0
        for link in links:
            note = _resolve_link(store, link)
            if note is None:
                continue
            note_idx = _find_note_index(store, note)
            if note_idx is not None:
                memory_str += store.format_memory(note_idx) + "\n"
            j += 1
            if j >= k:
                break
    return memory_str


def strategy_factual(query: str, store: MemoryStore, k: int = 10) -> str:
    """Strategy 1: Factual Retrieval with entity-focused reranking.
    
    For Single-hop (Cat 4) and Open-domain (Cat 3).
    Extracts key nouns from query, retrieves 2*K candidates,
    then reranks by entity overlap.
    """
    # Step 1: Extract key entities from query (simple approach: split into words, 
    # keep nouns/proper nouns - use lowercase content words > 3 chars)
    query_words = set(w.lower() for w in query.split() 
                      if len(w) > 3 and w.lower() not in STOP_WORDS)
    
    # Step 2: Retrieve 2*K candidates
    indices, scores = store.semantic_search(query, k=min(2*k, len(store.corpus)))
    
    # Step 3: Rerank by entity overlap
    reranked = []
    for idx, sim_score in zip(indices, scores):
        m = store.get_memory(idx)
        content = getattr(m, 'content', '').lower()
        keywords_list = getattr(m, 'keywords', [])
        keywords_text = ' '.join(str(kw).lower() for kw in keywords_list)
        
        # Count entity overlap
        combined_text = content + ' ' + keywords_text
        overlap = sum(1 for w in query_words if w in combined_text)
        
        # Combined score: similarity * (1 + 0.3 * overlap)
        new_score = float(sim_score) * (1 + 0.3 * overlap)
        reranked.append((idx, new_score))
    
    # Sort by new score
    reranked.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in reranked[:k]]
    
    # Step 4: Format output (same format as baseline for fair comparison)
    memory_str = ""
    for i in top_indices:
        m = store.get_memory(i)
        memory_str += store.format_memory(i) + "\n"
        # Also follow links
        links = getattr(m, 'links', [])
        j = 0
        for link in links:
            note = _resolve_link(store, link)
            if note is None:
                continue
            note_idx = _find_note_index(store, note)
            if note_idx is not None:
                memory_str += store.format_memory(note_idx) + "\n"
            j += 1
            if j >= k:
                break
    return memory_str


def strategy_temporal(query: str, store: MemoryStore, k: int = 10) -> str:
    """Strategy 2: Temporal Retrieval with time-aware sorting.
    
    For Temporal (Cat 2).
    Key differences from baseline:
    1. Retrieve candidates normally
    2. Sort by timestamp (chronological) instead of similarity
    3. Explicitly prefix each memory with timestamp
    """
    # Step 1: Retrieve candidates (wider search)
    indices, scores = store.semantic_search(query, k=min(2*k, len(store.corpus)))
    
    # Step 2: Parse timestamps and sort chronologically
    memories_with_time = []
    for idx in indices:
        m = store.get_memory(idx)
        ts = getattr(m, 'timestamp', '')
        parsed_time = _parse_timestamp(ts)
        memories_with_time.append((idx, ts, parsed_time))
    
    # Sort by parsed timestamp (chronological order)
    memories_with_time.sort(key=lambda x: x[2] if x[2] else datetime.min)
    
    # Take top-k after sorting
    selected = memories_with_time[:k]
    
    # Step 3: Format with explicit time prefix
    memory_str = ""
    for idx, ts, _ in selected:
        memory_str += store.format_memory(idx, include_time_prefix=True) + "\n"
    
    return memory_str


def strategy_multihop(query: str, store: MemoryStore, k: int = 10) -> str:
    """Strategy 4: Multi-hop Retrieval with broader search + link expansion.
    
    For Multi-hop (Cat 1).
    Key differences from baseline:
    1. Retrieve more candidates (3*K)
    2. Aggressively follow links to expand neighborhood
    3. Deduplicate and return diverse set
    """
    # Step 1: Broader initial retrieval
    indices, scores = store.semantic_search(query, k=min(3*k, len(store.corpus)))
    
    # Step 2: Expand through links (2 hops)
    all_indices = set(indices.tolist() if isinstance(indices, np.ndarray) else indices)
    
    # First hop: follow links from initial results
    for idx in list(all_indices)[:k]:
        m = store.get_memory(idx)
        links = getattr(m, 'links', [])
        for link in links:
            note = _resolve_link(store, link)
            if note is not None:
                note_idx = _find_note_index(store, note)
                if note_idx is not None:
                    all_indices.add(note_idx)
    
    # Second hop: follow links from first-hop results (limited)
    first_hop_new = all_indices - set(indices.tolist() if isinstance(indices, np.ndarray) else indices)
    for idx in list(first_hop_new)[:5]:
        m = store.get_memory(idx)
        links = getattr(m, 'links', [])
        for link in links[:3]:  # limit
            note = _resolve_link(store, link)
            if note is not None:
                note_idx = _find_note_index(store, note)
                if note_idx is not None:
                    all_indices.add(note_idx)
    
    # Step 3: Score all expanded results and take top
    all_indices_list = list(all_indices)
    if len(all_indices_list) > k:
        # Re-score by similarity
        query_emb = store.encoder.encode([query])[0]
        subset_embs = store.embeddings[all_indices_list]
        sims = cosine_similarity([query_emb], subset_embs)[0]
        top_k_local = np.argsort(sims)[-k:][::-1]
        final_indices = [all_indices_list[i] for i in top_k_local]
    else:
        final_indices = all_indices_list
    
    # Step 4: Format output
    memory_str = ""
    for i in final_indices:
        memory_str += store.format_memory(i) + "\n"
    
    return memory_str


def strategy_abstention(query: str, store: MemoryStore, k: int = 10,
                        theta_high: float = 0.55, theta_low: float = 0.35) -> str:
    """Strategy 5: Abstention-Aware Retrieval with confidence threshold.
    
    For Adversarial (Cat 5).
    Key differences from baseline:
    1. Check similarity confidence of top results
    2. If low confidence, prepend a signal that info may not exist
    3. If high confidence, return normally
    """
    indices, scores = store.semantic_search(query, k=k)
    
    top1_score = float(scores[0]) if len(scores) > 0 else 0.0
    
    # Build memory context (same as baseline)
    memory_str = ""
    for i in indices:
        m = store.get_memory(i)
        memory_str += store.format_memory(i) + "\n"
        links = getattr(m, 'links', [])
        j = 0
        for link in links:
            note = _resolve_link(store, link)
            if note is None:
                continue
            note_idx = _find_note_index(store, note)
            if note_idx is not None:
                memory_str += store.format_memory(note_idx) + "\n"
            j += 1
            if j >= k:
                break
    
    # Confidence-based prefix
    if top1_score < theta_low:
        memory_str = ("[NOTE: The retrieved memories have very low relevance to the question. "
                      "The information asked about may NOT exist in the conversation history. "
                      "If you cannot find a clear answer, respond with 'Not mentioned in the conversation'.]\n\n"
                      + memory_str)
    elif top1_score < theta_high:
        memory_str = ("[NOTE: The retrieved memories have moderate relevance. "
                      "Carefully check if the conversation actually contains the requested information. "
                      "If not clearly stated, respond with 'Not mentioned in the conversation'.]\n\n"
                      + memory_str)
    
    return memory_str


# ============================================================
# Helper Functions
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
    'it', 'its', 'they', 'them', 'their', 'what', 'ever', 'did', 'does',
    'don', 'didn', 'doesn', 'won', 'wouldn', 'couldn', 'shouldn',
}

def _resolve_link(store: MemoryStore, link) -> Optional[MemoryNote]:
    """Resolve a link to a MemoryNote."""
    if isinstance(link, int):
        if 0 <= link < len(store.memory_list):
            return store.memory_list[link]
        return None
    if isinstance(link, str):
        try:
            idx = int(link)
            if 0 <= idx < len(store.memory_list):
                return store.memory_list[idx]
        except ValueError:
            pass
        if link.startswith("memory_"):
            try:
                idx = int(link.split("_", 1)[1])
                if 0 <= idx < len(store.memory_list):
                    return store.memory_list[idx]
            except (ValueError, IndexError):
                pass
        if link in store.memories:
            return store.memories[link]
    return None

def _find_note_index(store: MemoryStore, note: MemoryNote) -> Optional[int]:
    """Find the index of a note in the memory list."""
    note_id = getattr(note, 'id', None)
    if note_id and note_id in store.memories:
        try:
            return store.memory_ids.index(note_id)
        except ValueError:
            pass
    # Fallback: check by identity
    for i, m in enumerate(store.memory_list):
        if m is note:
            return i
    return None

MONTH_MAP = {
    'january': 1, 'february': 2, 'march': 3, 'april': 4,
    'may': 5, 'june': 6, 'july': 7, 'august': 8,
    'september': 9, 'october': 10, 'november': 11, 'december': 12
}

def _parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse LoCoMo-style timestamps like '1:56 pm on 8 May, 2023'."""
    if not ts or not isinstance(ts, str):
        return None
    try:
        # Try common LoCoMo format: "H:MM am/pm on D Month, YYYY"
        ts_clean = ts.strip()
        # Extract date part after "on "
        if ' on ' in ts_clean:
            date_part = ts_clean.split(' on ', 1)[1]
        else:
            date_part = ts_clean
        
        # Parse "8 May, 2023" or "21 August, 2022"
        parts = date_part.replace(',', '').split()
        if len(parts) >= 3:
            day = int(parts[0])
            month = MONTH_MAP.get(parts[1].lower(), 1)
            year = int(parts[2])
            return datetime(year, month, day)
    except:
        pass
    return None


# ============================================================
# LLM Interface (reuse from memory_layer)
# ============================================================

def create_llm(model: str, backend: str):
    """Create LLM controller for answer generation."""
    from memory_layer import LLMController
    return LLMController(backend=backend, model=model)


def generate_keywords(llm_controller, question: str) -> str:
    """Generate search keywords from question using LLM."""
    prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

            Question: {question}

            Format your response as a JSON object with a "keywords" field containing the selected text. 

            Example response format:
            {{"keywords": "keyword1, keyword2, keyword3"}}"""
    
    response = llm_controller.llm.get_completion(prompt, response_format={"type": "json_schema", "json_schema": {
        "name": "response",
        "schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "string"}
            },
            "required": ["keywords"],
            "additionalProperties": False
        },
        "strict": True
    }})
    try:
        return json.loads(response)["keywords"]
    except:
        return response.strip()


def generate_answer(llm_controller, context: str, question: str, category: int,
                    answer: str, temperature_c5: float = 0.5) -> str:
    """Generate answer using LLM with category-specific prompts."""
    temperature = 0.7
    
    if category == 5:
        answer_tmp = []
        if random.random() < 0.5:
            answer_tmp.append('Not mentioned in the conversation')
            answer_tmp.append(answer)
        else:
            answer_tmp.append(answer)
            answer_tmp.append('Not mentioned in the conversation')
        user_prompt = f"""
            Based on the context: {context}, answer the following question. {question} 
            
            Select the correct answer: {answer_tmp[0]} or {answer_tmp[1]}  Short answer:
            """
        temperature = temperature_c5
    elif category == 2:
        user_prompt = f"""
            Based on the context: {context}, answer the following question. Use DATE of CONVERSATION to answer with an approximate date.
            Please generate the shortest possible answer, using words from the conversation where possible, and avoid using any subjects.   

            Question: {question} Short answer:
            """
    elif category == 3:
        user_prompt = f"""
            Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

            Question: {question} Short answer:
            """
    else:
        user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

            Question: {question} Short answer:
            """
    
    response = llm_controller.llm.get_completion(
        user_prompt, 
        response_format={"type": "json_schema", "json_schema": {
            "name": "response",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False
            },
            "strict": True
        }},
        temperature=temperature
    )
    
    try:
        return json.loads(response)["answer"]
    except:
        return response.strip()


# ============================================================
# Main Evaluation
# ============================================================

STRATEGY_MAP = {
    1: ("multi_hop", strategy_multihop),
    2: ("temporal", strategy_temporal),
    3: ("factual", strategy_factual),      # Open-domain → factual
    4: ("factual", strategy_factual),      # Single-hop → factual
    5: ("abstention", strategy_abstention), # Adversarial → abstention
}

def load_memory_store(memories_dir: str, sample_idx: int, 
                      encoder: SentenceTransformer) -> Optional[MemoryStore]:
    """Load cached memories and create a MemoryStore."""
    mem_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
    ret_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
    emb_file = os.path.join(memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy")
    
    if not all(os.path.exists(f) for f in [mem_file, ret_file, emb_file]):
        return None
    
    with open(mem_file, 'rb') as f:
        memories = pickle.load(f)
    with open(ret_file, 'rb') as f:
        ret_state = pickle.load(f)
    embeddings = np.load(emb_file)
    
    return MemoryStore(memories, embeddings, ret_state['corpus'], encoder)


def run_comparison(dataset_path: str, memories_dir: str, model: str, backend: str,
                   output_path: str, retrieve_k: int = 10, temperature_c5: float = 0.5,
                   ratio: float = 1.0):
    """Run the full comparison experiment."""
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"adaptmr_comparison_{timestamp}.log")
    
    logger = logging.getLogger('adaptmr')
    logger.setLevel(logging.INFO)
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info("=" * 60)
    logger.info("AdaptMR vs Baseline Comparison Experiment")
    logger.info(f"Model: {model}, Backend: {backend}, K: {retrieve_k}")
    logger.info("=" * 60)
    
    # Load encoder
    logger.info("Loading SentenceTransformer encoder...")
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Set global encoder for fast SBERT metrics
    global _fast_metrics_encoder
    _fast_metrics_encoder = encoder
    
    # Load LLM
    logger.info(f"Initializing LLM ({model} via {backend})...")
    llm_controller = create_llm(model, backend)
    
    # Load dataset
    logger.info(f"Loading dataset from {dataset_path}")
    samples = load_locomo_dataset(dataset_path)
    
    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
    logger.info(f"Using {len(samples)} samples")
    
    # Checkpoint support
    checkpoint_file = os.path.join(os.path.dirname(output_path), "adaptmr_checkpoint.json")
    
    baseline_results = []
    adaptmr_results = []
    baseline_metrics_all = []
    adaptmr_metrics_all = []
    all_categories = []
    completed_samples = set()
    total_q = 0
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                ckpt = json.load(f)
            baseline_results = ckpt.get("baseline_results", [])
            adaptmr_results = ckpt.get("adaptmr_results", [])
            baseline_metrics_all = ckpt.get("baseline_metrics", [])
            adaptmr_metrics_all = ckpt.get("adaptmr_metrics", [])
            all_categories = ckpt.get("all_categories", [])
            completed_samples = set(ckpt.get("completed_samples", []))
            total_q = ckpt.get("total_q", 0)
            logger.info(f"[Checkpoint] Resumed: {len(completed_samples)} samples done, {total_q} questions")
        except Exception as e:
            logger.info(f"[Checkpoint] Failed to load: {e}, starting fresh")
    
    def save_checkpoint():
        ckpt = {
            "baseline_results": baseline_results,
            "adaptmr_results": adaptmr_results,
            "baseline_metrics": baseline_metrics_all,
            "adaptmr_metrics": adaptmr_metrics_all,
            "all_categories": all_categories,
            "completed_samples": list(completed_samples),
            "total_q": total_q,
        }
        with open(checkpoint_file, 'w') as f:
            json.dump(ckpt, f)
    
    # Main evaluation loop
    allow_categories = [1, 2, 3, 4, 5]
    
    for sample_idx, sample in enumerate(samples):
        if sample_idx in completed_samples:
            logger.info(f"[Skip] Sample {sample_idx} already completed")
            continue
        
        # Load memories for this sample
        store = load_memory_store(memories_dir, sample_idx, encoder)
        if store is None:
            logger.warning(f"[Skip] No cached memories for sample {sample_idx}")
            continue
        
        logger.info(f"\n{'='*40}")
        logger.info(f"Processing sample {sample_idx + 1}/{len(samples)} "
                     f"({len(store.memory_list)} memories, {len(sample.qa)} QAs)")
        
        for qa in sample.qa:
            if int(qa.category) not in allow_categories:
                continue
            
            total_q += 1
            cat = int(qa.category)
            all_categories.append(cat)
            
            # Generate keywords (shared between baseline and adaptmr)
            keywords = generate_keywords(llm_controller, qa.question)
            
            # --- Baseline retrieval ---
            baseline_context = baseline_retrieval(keywords, store, k=retrieve_k)
            baseline_pred = generate_answer(
                llm_controller, baseline_context, qa.question, cat,
                qa.final_answer, temperature_c5
            )
            
            # --- AdaptMR retrieval ---
            strategy_name, strategy_fn = STRATEGY_MAP[cat]
            adaptmr_context = strategy_fn(keywords, store, k=retrieve_k)
            adaptmr_pred = generate_answer(
                llm_controller, adaptmr_context, qa.question, cat,
                qa.final_answer, temperature_c5
            )
            
            # Calculate metrics
            ref = qa.final_answer if qa.final_answer else ""
            b_metrics = calculate_metrics_fast(baseline_pred, str(ref)) if ref else _empty_metrics()
            a_metrics = calculate_metrics_fast(adaptmr_pred, str(ref)) if ref else _empty_metrics()
            
            baseline_metrics_all.append(b_metrics)
            adaptmr_metrics_all.append(a_metrics)
            
            # Store results
            result_base = {
                "sample_id": sample_idx, "question": qa.question,
                "prediction": baseline_pred, "reference": ref,
                "category": cat, "strategy": "baseline",
                "metrics": b_metrics
            }
            result_adapt = {
                "sample_id": sample_idx, "question": qa.question,
                "prediction": adaptmr_pred, "reference": ref,
                "category": cat, "strategy": strategy_name,
                "metrics": a_metrics
            }
            baseline_results.append(result_base)
            adaptmr_results.append(result_adapt)
            
            # Log
            b_f1 = b_metrics.get('f1', 0)
            a_f1 = a_metrics.get('f1', 0)
            diff = a_f1 - b_f1
            marker = "+" if diff > 0 else ("=" if diff == 0 else "-")
            logger.info(f"  Q{total_q} [Cat{cat}/{strategy_name}] "
                        f"F1: {b_f1:.3f}→{a_f1:.3f} ({marker}{diff:.3f}) "
                        f"| {qa.question[:60]}...")
            
            if total_q % 10 == 0:
                logger.info(f"  --- Progress: {total_q} questions processed ---")
        
        completed_samples.add(sample_idx)
        save_checkpoint()
        logger.info(f"[Checkpoint] Sample {sample_idx} saved ({total_q} total questions)")
    
    # ============================================================
    # Final Results
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON RESULTS")
    logger.info("=" * 60)
    
    if not baseline_metrics_all:
        logger.info("No results to report.")
        return
    
    # Aggregate
    baseline_agg = aggregate_metrics(baseline_metrics_all, all_categories)
    adaptmr_agg = aggregate_metrics(adaptmr_metrics_all, all_categories)
    
    cat_names = {1: 'Multi-hop', 2: 'Temporal', 3: 'Open-domain', 4: 'Single-hop', 5: 'Adversarial'}
    strategy_names = {1: 'multi_hop', 2: 'temporal', 3: 'factual', 4: 'factual', 5: 'abstention'}
    
    logger.info(f"\n{'Category':<20} {'Baseline F1':>12} {'AdaptMR F1':>12} {'Delta':>8} {'Strategy':<12}")
    logger.info("-" * 70)
    
    for key in sorted(baseline_agg.keys()):
        b_f1 = baseline_agg[key].get('f1', {}).get('mean', 0) * 100
        a_f1 = adaptmr_agg[key].get('f1', {}).get('mean', 0) * 100
        delta = a_f1 - b_f1
        count = baseline_agg[key].get('f1', {}).get('count', 0)
        
        if key == 'overall':
            display = 'Overall'
            strat = 'adaptive'
        else:
            cat_num = int(key.split('_')[1])
            display = f'Cat{cat_num} {cat_names.get(cat_num, "")}'
            strat = strategy_names.get(cat_num, '')
        
        sign = '+' if delta > 0 else ''
        logger.info(f"{display:<20} {b_f1:>11.2f}% {a_f1:>11.2f}% {sign}{delta:>7.2f}% {strat:<12} (n={count})")
    
    # Also show BLEU and SBERT
    logger.info(f"\n{'Category':<20} {'BL Bleu1':>10} {'AM Bleu1':>10} {'BL SBERT':>10} {'AM SBERT':>10}")
    logger.info("-" * 65)
    for key in sorted(baseline_agg.keys()):
        b_bleu = baseline_agg[key].get('bleu1', {}).get('mean', 0) * 100
        a_bleu = adaptmr_agg[key].get('bleu1', {}).get('mean', 0) * 100
        b_sbert = baseline_agg[key].get('sbert_similarity', {}).get('mean', 0) * 100
        a_sbert = adaptmr_agg[key].get('sbert_similarity', {}).get('mean', 0) * 100
        
        if key == 'overall':
            display = 'Overall'
        else:
            cat_num = int(key.split('_')[1])
            display = f'Cat{cat_num} {cat_names.get(cat_num, "")}'
        
        logger.info(f"{display:<20} {b_bleu:>9.2f}% {a_bleu:>9.2f}% {b_sbert:>9.2f}% {a_sbert:>9.2f}%")
    
    # Save final results
    final = {
        "experiment": "AdaptMR vs Baseline Comparison",
        "model": model,
        "backend": backend,
        "retrieve_k": retrieve_k,
        "total_questions": total_q,
        "baseline_aggregate": baseline_agg,
        "adaptmr_aggregate": adaptmr_agg,
        "baseline_results": baseline_results,
        "adaptmr_results": adaptmr_results,
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_path}")
    
    # Cleanup checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    return final


def _empty_metrics():
    return {
        "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
        "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
        "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
    }


def main():
    parser = argparse.ArgumentParser(description="AdaptMR vs Baseline Comparison")
    parser.add_argument("--dataset", type=str, default="data/locomo10.json")
    parser.add_argument("--memories_dir", type=str, 
                        default="cached_memories_advanced_ollama_qwen2.5_1.5b")
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b")
    parser.add_argument("--backend", type=str, default="ollama")
    parser.add_argument("--output", type=str, default="results/adaptmr_comparison.json")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--retrieve_k", type=int, default=10)
    parser.add_argument("--temperature_c5", type=float, default=0.5)
    args = parser.parse_args()
    
    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, args.dataset)
    memories_dir = os.path.join(base_dir, args.memories_dir)
    output_path = os.path.join(base_dir, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    run_comparison(
        dataset_path=dataset_path,
        memories_dir=memories_dir,
        model=args.model,
        backend=args.backend,
        output_path=output_path,
        retrieve_k=args.retrieve_k,
        temperature_c5=args.temperature_c5,
        ratio=args.ratio,
    )


if __name__ == "__main__":
    main()

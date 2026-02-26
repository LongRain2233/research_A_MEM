"""
Download all papers cited in Section 5.3.3 "Retrieval Strategies" of
"Memory in the Age of AI Agents" (arXiv:2512.13564)

Papers are downloaded from arXiv PDF endpoints.
"""

import requests
import os
import time
import sys

sys.stdout.reconfigure(encoding='utf-8')

OUTPUT_DIR = r"D:\research\research_A_MEM\papers_533"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All papers cited in Section 5.3.3 with confirmed arXiv IDs
# Format: (paper_name, arxiv_id, citation_key)
PAPERS_533 = [
    # ===== Lexical Retrieval =====
    ("Agent KB - Cross-domain Experience for Agentic Problem Solving",
     "2507.06229", "Tang_2025d_AgentKB"),
    
    ("Mem-alpha - Learning Memory Construction via Reinforcement Learning",
     "2509.25911", "Wang_2025p_MemAlpha"),
    
    # ===== Semantic Retrieval =====
    ("Sentence-BERT - Sentence Embeddings using Siamese BERT-Networks",
     "1908.10084", "Reimers_2019_SentenceBERT"),
    
    ("CLIP - Learning Transferable Visual Models from Natural Language",
     "2103.00020", "Radford_2021_CLIP"),
    
    ("RAG - Retrieval-Augmented Generation for Knowledge-Intensive NLP",
     "2005.11401", "Lewis_2020_RAG"),
    
    # ===== Graph Retrieval =====
    ("AriGraph - Learning Knowledge Graph World Models with Episodic Memory",
     "2407.04363", "Anokhin_2024_AriGraph"),
    
    ("EMG-RAG - Retrieval-Augmented Generation on Editable Memory Graphs",
     "2409.19401", "Wang_2024l_EMGRAG"),
    
    ("Mem0 - Building Production-Ready AI Agents with Scalable Long-Term Memory",
     "2504.19413", "Chhikara_2025_Mem0"),
    
    ("SGMem - Sentence Graph Memory for Long-Term Conversational Agents",
     "2509.21212", "Wu_2025h_SGMem"),
    
    ("HippoRAG - Neurobiologically Inspired Long-Term Memory for LLMs",
     "2405.14831", "Gutierrez_2024_HippoRAG"),
    
    ("D-SMART - Enhancing LLM Dialogue Consistency via Dynamic Structured Memory",
     "2510.13363", "Lei_2025_DSMART"),
    
    ("Zep - A Temporal Knowledge Graph Architecture for Agent Memory",
     "2501.13956", "Rasmussen_2025_Zep"),
    
    ("MemoTime - Memory-Augmented Temporal KG Enhanced LLM Reasoning",
     "2510.13614", "Tan_2025b_MemoTime"),
    
    # ===== Generative Retrieval =====
    ("DSI - Transformer Memory as a Differentiable Search Index",
     "2202.00433", "Tay_2022_DSI"),
    
    ("NCI - A Neural Corpus Indexer for Document Retrieval",
     "2206.02743", "Wang_2022b_NCI"),
    
    # ===== Hybrid Retrieval =====
    ("MIRIX - Multi-Agent Memory System for LLM-Based Agents",
     "2507.07957", "Wang_2025_MIRIX"),
    
    ("Semantic Anchoring in Agentic Memory",
     "2508.12630", "Chatterjee_2025_SemanticAnchoring"),
    
    ("Lyfe Agents - Generative Agents for Low-Cost Real-Time Social Interactions",
     "2310.02172", "Kaiya_2023_LyfeAgents"),
    
    ("MemoriesDB - Temporal-Semantic-Relational Database for Long-Term Agent Memory",
     "2511.06179", "Ward_2025_MemoriesDB"),
]

# Papers without arXiv IDs (need alternative download)
PAPERS_NO_ARXIV = [
    ("SeCom - Memory Construction and Retrieval for Personalized Conversational Agents",
     "https://openreview.net/forum?id=xKDZAW0He3", "Pan_2025_SeCom"),
    ("CAM - A Constructivist View of Agentic Memory for LLM-Based Reading Comprehension",
     "October 2025, no URL found", "Li_2025g_CAM"),
    ("MAICC - Mixed-Utility Scoring Function for Agent Memory Retrieval",
     "Jiang et al., 2025c - not found in references", "Jiang_2025c_MAICC"),
]


def download_arxiv_pdf(arxiv_id, output_path, paper_name):
    """Download a paper from arXiv"""
    # Clean up the arXiv ID (remove trailing periods)
    arxiv_id = arxiv_id.strip().rstrip('.')
    
    # Try multiple URL formats
    urls_to_try = [
        f"https://arxiv.org/pdf/{arxiv_id}",
        f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        f"https://export.arxiv.org/pdf/{arxiv_id}",
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    for url in urls_to_try:
        try:
            print(f"  Trying: {url}")
            response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
            
            if response.status_code == 200 and len(response.content) > 10000:
                # Check if it's a PDF (starts with %PDF)
                if response.content[:4] == b'%PDF':
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    size_kb = len(response.content) / 1024
                    print(f"  ✓ Downloaded: {os.path.basename(output_path)} ({size_kb:.0f} KB)")
                    return True
                else:
                    print(f"  Response is not a PDF (got: {response.content[:50]})")
            else:
                print(f"  HTTP {response.status_code}, size: {len(response.content)} bytes")
        except Exception as e:
            print(f"  Error: {e}")
    
    return False


print("="*70)
print("Downloading papers from Section 5.3.3: Retrieval Strategies")
print(f"Output directory: {OUTPUT_DIR}")
print("="*70 + "\n")

success_count = 0
fail_list = []

for i, (paper_name, arxiv_id, cite_key) in enumerate(PAPERS_533, 1):
    output_filename = f"{cite_key}_{arxiv_id}.pdf"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"[{i}/{len(PAPERS_533)}] {paper_name}")
    print(f"  arXiv: {arxiv_id}")
    
    if os.path.exists(output_path):
        size_kb = os.path.getsize(output_path) / 1024
        print(f"  ✓ Already exists ({size_kb:.0f} KB), skipping")
        success_count += 1
    else:
        success = download_arxiv_pdf(arxiv_id, output_path, paper_name)
        if success:
            success_count += 1
        else:
            fail_list.append((paper_name, arxiv_id))
    
    print()
    time.sleep(2)  # Be polite to arXiv servers

print("="*70)
print(f"Download complete: {success_count}/{len(PAPERS_533)} papers downloaded")
print(f"Output directory: {OUTPUT_DIR}")
print()

if fail_list:
    print(f"FAILED ({len(fail_list)} papers):")
    for name, arxiv_id in fail_list:
        print(f"  - {name} (arXiv:{arxiv_id})")
    print()

print("Papers NOT available on arXiv:")
for name, url, cite_key in PAPERS_NO_ARXIV:
    print(f"  - {name}")
    print(f"    URL: {url}")

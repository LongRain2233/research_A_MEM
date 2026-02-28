# Abstract

To support long-term interaction in complex environments, LLM agents require memory systems that manage historical experiences. Existing approaches either retain full interaction histories via passive context extension, leading to substantial redundancy, or rely on iterative reasoning to filter noise, incurring high token costs. To address this challenge, we introduce SimpleMem, an efficient memory framework based on semantic lossless compression. We propose a three-stage pipeline designed to maximize information density and token utilization: (1) Semantic Structured Compression, which distills unstructured interactions into compact, multi-view indexed memory units; (2) Online Semantic Synthesis, an intra-session process that instantly integrates related context into unified abstract representations to eliminate redundancy; and (3) Intent-Aware Retrieval Planning, which infers search intent to dynamically determine retrieval scope and construct precise context efficiently. Experiments on benchmark datasets show that our method consistently outperforms baseline approaches in accuracy, retrieval efficiency, and inference cost, achieving an average F1 improvement of $2 6 . 4 \%$ in LoCoMo while reducing inference-time token consumption by up to $3 0 \times$ , demonstrating a superior balance between performance and efficiency. Code is available at https://github.com/aiming-lab/SimpleMem.

# 1. Introduction

Large Language Model (LLM) agents have recently demonstrated remarkable capabilities across a wide range of tasks (Xia et al., 2025; Team et al., 2025; Qiu et al., 2025). However, constrained by fixed context windows, existing

*Equal contribution 1UNC-Chapel Hill 2University of California, Berkeley 3University of California, Santa Cruz. Correspondence to: Jiaqi Liu <jqliu@cs.unc.edu>, Mingyu Ding <md@cs.unc.edu>, Huaxiu Yao <huaxiu@cs.unc.edu>.

Preprint. January 30, 2026.

agents exhibit significant limitations when engaging in longcontext and multi-turn interaction scenarios (Liu et al., 2023; Wang et al., 2024a; Liu et al., 2025; Hu et al., 2025; Tu et al., 2025). To facilitate reliable long-term interaction, LLM agents require robust memory systems to efficiently manage and utilize historical experience (Dev & Taranjeet, 2024; Fang et al., 2025; Wang & Chen, 2025; Tang et al., 2025; Yang et al., 2025; Ouyang et al., 2025).

While recent research has extensively explored the design of memory modules for LLM agents, current systems still suffer from suboptimal retrieval efficiency and low token utilization (Fang et al., 2025; Hu et al., 2025). On one hand, many existing systems maintain complete interaction histories through full-context extension (Li et al., 2025; Zhong et al., 2024). However, this approach introduce substantial redundant information (Hu et al., 2025). Specifically, during long-horizon interactions, user inputs and model responses accumulate substantial low-entropy noise (e.g., repetitive logs, non-task-oriented dialogue), which degrades the effective information density of the memory buffer. This redundancy adversely affects memory retrieval and downstream reasoning, often leading to middle-context degradation phenomena (Liu et al., 2023), while also incurring significant computational overhead during retrieval and secondary inference. On the other hand, some agentic frameworks mitigate noise through online filtering based on iterative reasoning procedures (Yan et al., 2025; Packer et al., 2023). Although such approaches improve retrieval relevance, they rely on repeated inference cycles, resulting in substantial computational cost, including increased latency and token usage. As a result, neither paradigm achieves efficient allocation of memory and computation resources.

To address these limitations, we introduce SimpleMem, an efficient memory framework inspired by the Complementary Learning Systems (CLS) theory (Kumaran et al., 2016) and built around structured semantic compression. The objective of SimpleMem is to improve information efficiency under fixed context and token budgets. We develop a threestage pipeline that supports dynamic memory compression, organization, and adaptive retrieval: (1) Semantic Structured Compression: we apply a semantic density gating mechanism via LLM-based qualitative assessment. The system uses the foundation model as a semantic judge to

![](images/367bb74e3bb73359151e9634c285ca2b2e1377c51eeabfe01682f040ef010e0e.jpg)  
Figure 1. Performance vs. Efficiency Trade-off. Comparison of F1 against Token Cost on the LoCoMo benchmark. SimpleMem achieves high accuracy with minimal token consumption.

estimate information gain relative to history, preserving only content with high downstream utility. Retained information is reformulated into compact memory units and indexed jointly using dense semantic embeddings, sparse lexical features, and symbolic metadata. (2) Online Semantic Synthesis: inspired by biological consolidation and optimized for real-time interaction, we introduce an intra-session process that reorganizes memory on-the-fly. Related memory units are synthesized into higher-level abstract representations during the write phase, allowing repetitive or structurally similar experiences to be denoised and compressed immediately. (3) Intent-Aware Retrieval Planning: we employ a planning-based retrieval strategy that infers latent search intent to determine retrieval scope dynamically. The system constructs a precise context by querying multiple indexes (symbolic, semantic, lexical) and unifying results through ID-based deduplication, balancing structural constraints and semantic relevance without complex linear weighting.

Our primary contribution is SimpleMem, an efficient memory framework grounded in structured semantic compression, which improves information efficiency through principled memory organization, online synthesis, and intentaware planning. As shown in Figure 1, our empirical experiments demonstrate that SimpleMem establishes a new state-of-the-art with an F1 score, outperforming strong baselines like Mem0 by $2 6 . 4 \%$ , while reducing inference token consumption by $3 0 \times$ compared to full-context models.

# 2. The SimpleMem Architecture

In this section, we present SimpleMem, which operates through a three-stage pipeline (see Figure 2 for the detailed architecture). Specifically, we first describe the Semantic Structured Compression, which utilizes implicit semantic gating to filter redundant interaction content and reformulate raw dialogue streams into compact memory units. Next, we describe Online Semantic Synthesis, an on-the-fly mechanism that instantly synthesizes related memory units into higher-level abstract representations, ensuring a compact and noise-free memory topology. Finally, we present Intent-Aware Retrieval Planning, which infers latent search intent

to dynamically adjust retrieval scope, constructing precise and token-efficient contexts for downstream reasoning.

# 2.1. Semantic Structured Compression

A primary bottleneck in long-term interaction is context inflation, the accumulation of raw, low-entropy dialogue. For example, a large portion of interaction segments in the real-world consists of phatic chit-chat or redundant confirmations, which contribute little to downstream reasoning but consume substantial context capacity. To address this, we introduce a mechanism to actively filter and restructure information at the source.

Specifically, first, incoming dialogue is segmented into overlapping sliding windows $W$ of fixed length, where each window represents a short contiguous span of recent interaction. These windows serve as the basic units for processing.

Unlike traditional approaches that rely on rigid heuristic filters or separate classification models, we employ an implicit semantic density gating mechanism integrated directly into the generation process. We model the information assessment as an instruction-following task performed by the foundation model itself. The system leverages the attention mechanism of the LLM $f$ to identify high-entropy spans within the window W relative to the immediate history $H$ .

Formally, we define the gating function $\Phi _ { \mathrm { g a t e } }$ not as a binary classifier, but as a generative filter resulting from the model’s extraction capability:

$$
\Phi_ {\text {g a t e}} (W) \rightarrow \{m _ {k} \} \quad \text {s . t .} \quad | \{m _ {k} \} | \geq 0 \tag {1}
$$

Here, the generation of an empty set (∅) inherently signifies a low-density window (e.g., pure phatic chitchat), effectively discarding it without explicit threshold tuning. This instruction-driven gating allows the system to capture subtle semantic nuances while naturally filtering redundancy through the model’s semantic compression objectives.

For windows containing valid semantic content, the system performs a unified De-linearization Transformation $\mathcal { F } _ { \theta }$ . Instead of sequential independent modules, we optimize the extraction, coreference resolution, and temporal anchoring as a joint generation task. The transformation projects the raw dialogue window $W$ directly into a set of context-independent memory units $\{ m _ { k } \}$ :

$$
\left\{m _ {k} \right\} = \mathcal {F} _ {\theta} (W; H) \approx \left(g _ {\text {t i m e}} \circ g _ {\text {c o r e f}} \circ g _ {\text {e x t}}\right) (W). \tag {2}
$$

In this unified pass, the model follows strict instructional constraints to: (1) resolve ambiguous pronouns to specific entity names $( g _ { \mathrm { c o r e f } } )$ , (2) convert relative temporal expressions into absolute ISO-8601 timestamps $( g _ { \mathrm { t i m e } } )$ , and (3) atomize complex dialogue flows into self-contained factual statements. By aggregating all resulting units $m _ { k }$ across sliding windows, we obtain the complete memory set $\mathcal { M }$ .

![](images/2a5caa299090b92dd6e1685338c2800630643dd6c4d08b4d8fbb01d7c2249565.jpg)  
Figure 2. The SimpleMem Architecture. SimpleMem follows a three-stage pipeline: (1) Semantic Structured Compression filters low-utility dialogue and converts informative windows into compact, context-independent memory units. (2) Online Semantic Synthesis consolidates related fragments during writing, maintaining a compact and coherent memory topology. (3) Intent-Aware Retrieval Planning infers search intent to adapt retrieval scope and query forms, enabling parallel multi-view retrieval and token-efficient context construction.

Following compression, the system organizes the memory units to support storage and retrieval. This stage consists of two synergistic processes: (i) structured multi-view indexing for precise access, and (ii) online semantic synthesis for minimizing redundancy at the point of creation.

To support flexible and high-fidelity retrieval, each memory unit is indexed through three complementary representations. First, at the Semantic Layer, we map the entry to a dense vector space $s _ { k }$ using embedding models, capturing abstract meaning to enable fuzzy matching (e.g., retrieving "latte" when querying "hot drink"). Second, the Lexical Layer utilizes an inverted index-based sparse representation. This acts as a high-dimensional sparse vector $l _ { k }$ focusing on exact keyword matches and rare proper nouns, ensuring that specific entities are not diluted in dense vector space. Third, the Symbolic Layer extracts structured metadata, such as timestamps and entity types, to enable deterministic filtering logic. Formally, for a given memory unit $m _ { k }$ , these projections form the comprehensive indexing $\mathcal { T }$ :

$$
\mathcal {I} \left(m _ {t, k}\right) = \left\{ \begin{array}{l l} s _ {k} = E _ {\text {d e n s e}} \left(m _ {k}\right) & (\text {S e m a n t i c L a y e r}) \\ l _ {k} = E _ {\text {s p a r s e}} \left(m _ {k}\right) & (\text {L e x i c a l L a y e r}) \\ r _ {k} = E _ {\text {s y m}} \left(m _ {k}\right) & (\text {S y m b o l i c L a y e r}) \end{array} \right. \tag {3}
$$

This architecture allows the system to flexibly query information based on conceptual similarity, exact keyword matches, or structured metadata constraints.

# 2.2. Online Semantic Synthesis

While this multi-view indexing strategy facilitates access, naively accumulating raw extractions leads to fragmentation, causing the memory structure to grow in a purely additive and unregulated manner that fails to adapt in real time to the evolving semantic context of an ongoing interaction. To address this, we introduce Online Semantic Synthesis, an intra-session consolidation mechanism. Unlike traditional systems that rely on asynchronous background maintenance, SimpleMem performs synthesis on-the-fly during the write phase. The model analyzes the stream of extracted facts

within the current session scope and synthesizes related fragments into unified, high-density entries before they are committed to the database.

Formally, we define this synthesis as a transformation function $\mathcal { F } _ { \mathrm { s y n } }$ that maps a set of new observations $\boldsymbol { O } _ { \mathrm { s e s s i o n } }$ to a consolidated memory entry $\mathcal { F } _ { \mathrm { s y n } } ( O _ { \mathrm { s e s s i o n } } , \mathcal { C } _ { \mathrm { c o n t e x t } } ; f )$ , where $\mathcal { C } _ { \mathrm { { c o n t e x t } } }$ represents the current conversational context. This operation denoises the input by merging scattered details into a coherent whole. For instance, rather than storing three separate fragments like "User wants coffee", "User prefers oat milk", and "User likes it hot", the synthesis layer consolidates them into a single, comprehensive entry: "User prefers hot coffee with oat milk". This proactive synthesis ensures that the memory topology remains compact and free of redundant fragmentation, significantly reducing the burden on the retrieval system during future interactions.

# 2.3. Intent-Aware Retrieval Planning

After memory entries are organized, the final challenge is to retrieve relevant information under constrained context budgets. Standard retrieval approaches typically fetch a fixed number of entries, which often results in recall failure for complex queries or token wastage for simple ones. To address this, we introduce Intent-Aware Retrieval Planning, a mechanism that dynamically determines the retrieval scope and depth by inferring the user’s latent search intent.

Unlike systems that rely on scalar complexity classifiers, SimpleMem leverages the reasoning capabilities of the LLM to generate a comprehensive retrieval plan. Given a query $q$ and history $H$ , the planning module $\mathcal { P }$ acts as a reasoner to decompose the information needs and estimate the necessary search depth $d$ :

$$
\left\{q _ {\text {s e m}}, q _ {\text {l e x}}, q _ {\text {s y m}}, d \right\} \sim \mathcal {P} (q, H) \tag {4}
$$

where qsem, qlex, and $q _ { \mathrm { s y m } }$ are optimized queries for semantic, lexical, and symbolic retrieval respectively. The parameter $d$ represents the adaptive retrieval depth, which reflects the

estimated complexity of the query. Based on $d$ , the system utilizes a candidate limit $n$ (where $n \propto d$ ) to balance recall coverage against context window constraints.

Guided by this plan, the system executes a parallel multiview retrieval. We simultaneously query all three index layers defined in Section 2.1, imposing the quantity limit $n$ on each path:

$$
\mathcal {R} _ {\mathrm {s e m}} = \operatorname {T o p -} n (\cos (E (q _ {\mathrm {s e m}}), E (m _ {i})) \mid m _ {i} \in \mathcal {M})
$$

$$
\mathcal {R} _ {\text {l e x}} = \operatorname {T o p -} n (\mathrm {B M 2 5} \left(q _ {\text {l e x}}, m _ {i}\right) \mid m _ {i} \in \mathcal {M}) \tag {5}
$$

$$
\mathcal {R} _ {\mathrm {s y m}} = \operatorname {T o p -} n (\left\{m _ {i} \in \mathcal {M} \mid \operatorname {M e t a} (m _ {i}) \vDash q _ {\mathrm {s y m}} \right\})
$$

Here, each view captures distinct relevance signals: ${ \mathcal { R } } _ { \mathrm { s e m } }$ retrieves based on dense embedding similarity; $\mathcal { R } _ { \mathrm { l e x } }$ matches exact keywords or proper nouns; and $\mathcal { R } _ { \mathrm { s y m } }$ filters entries based on structured metadata constraints.

Finally, we construct the context $\mathcal { C } _ { q }$ by merging the results from these three views using a set union operation. This step naturally deduplicates overlapping entries, ensuring a comprehensive yet compact context:

$$
\mathcal {C} _ {q} = \mathcal {R} _ {\text {s e m}} \cup \mathcal {R} _ {\text {l e x}} \cup \mathcal {R} _ {\text {s y m}} \tag {6}
$$

This hybrid approach ensures that strong signals from any view are preserved, allowing the system to adaptively scale its retrieval volume $n$ based on the inferred depth $d$ .

# 3. Experiments

In this section, we evaluate SimpleMem on the benchmark to answer the following research questions: (1) Does SimpleMem outperform other memory systems in complex longterm reasoning understanding tasks? (2) Can SimpleMem achieve a superior trade-off between retrieval accuracy and token consumption? (3) How effective are the proposed components? (4) What factors account for the observed performance and efficiency gains?

# 3.1. Experimental Setup

Benchmark Dataset. We evaluate performance on the LoCoMo (Maharana et al., 2024) and LongMemEval-S (Wu et al., 2024) benchmarks. Brief descriptions are provided below, with additional details in Appendix B.

LoCoMo (Maharana et al., 2024) is specifically designed to test the limits of LLMs in processing long-term conversational dependencies. The dataset comprises conversation samples ranging from 200 to 400 turns, containing complex temporal shifts and interleaved topics. The evaluation set consists of 1,986 questions categorized into four distinct reasoning types: (1) multi-hop reasoning; (2) temporal reasoning; (3) open Domain; (4) single hop.

LongMemEval-S (Wu et al., 2024) features extreme context lengths that pose severe challenges for memory systems.

Unlike standard benchmarks, it requires precise answer localization across multiple sub-categories (e.g., temporal events, user preferences) within exceptionally long interaction histories. We use gpt-4.1-mini to evaluate answer correctness against ground-truth references, labeling responses as CORRECT or WRONG based on semantic and temporal alignment. The full evaluation prompt is provided in Appendix A.4.

Baselines. We compare SimpleMem with representative memory-augmented systems: LOCOMO (Maharana et al., 2024), READAGENT (Lee et al., 2024), MEMORY-BANK (Zhong et al., 2024), MEMGPT (Packer et al., 2023), A-MEM (Xu et al., 2025), LIGHTMEM (Fang et al., 2025), and Mem0 (Dev & Taranjeet, 2024).

Backbone Models. To test robustness across capability scales, we instantiate each baseline and SimpleMem on multiple LLM backends: GPT-4o, GPT-4.1-mini, Qwen-Plus, Qwen2.5 (1.5B/3B), and Qwen3 (1.7B/8B).

Implementation Details. For semantic structured compression, we use a sliding window of size $W = 2 0$ . Memory indexing is implemented using LanceDB with a multi-view design: Qwen3-embedding-0.6b (1024 dimensions) for dense semantic embeddings, BM25 for sparse lexical indexing, and SQL-based metadata storage for symbolic attributes. During retrieval, we employ adaptive queryaware retrieval, where the retrieval depth is dynamically adjusted based on estimated query complexity, ranging from $k _ { \operatorname* { m i n } { } } = 3$ for simple lookups to $k _ { \operatorname* { m a x } } = 2 0$ for complex reasoning queries.

Evaluation Metrics. For LoCoMo, we report F1 and BLEU-1 (accuracy), Adversarial Success Rate (robustness to distractors), and Token Cost (retrieval efficiency). For LongMemEval-S, we use its standard accuracy-style metric.

# 3.2. Results and Analysis

Tables 1 and 3 present detailed performance comparisons on the LoCoMo benchmark across different model scales, while Table 2 reports results on LongMemEval-S.

Performance on High-Capability Models. Across Lo-CoMo and LongMemEval-S, SimpleMem consistently outperforms existing memory systems across model scales, achieving strong and robust gains in accuracy.

On LoCoMo (Table 1), SimpleMem leads all baselines. Using GPT-4.1-mini, it achieves an Average F1 of 43.24, substantially exceeding Mem0 (34.20) and the full-context baseline (18.70). The largest gains are observed in Temporal Reasoning, where SimpleMem reaches 58.62 F1 compared to 48.91 for Mem0, underscoring the effectiveness of semantic structured compression in resolving complex temporal dependencies. These improvements persist at larger scales:

Table 1. Performance on the LoCoMo benchmark with High-Capability Models (GPT-4.1 series and Qwen3-Plus). SimpleMem achieves superior efficiency-performance balance.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">MultiHop</td><td colspan="2">Temporal</td><td colspan="2">OpenDomain</td><td colspan="2">SingleHop</td><td colspan="2">Average</td><td rowspan="2">Token Cost</td></tr><tr><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td></tr><tr><td rowspan="7">GPT-4.1-mini</td><td>LoCoMo</td><td>25.02</td><td>21.62</td><td>12.04</td><td>10.63</td><td>19.05</td><td>17.07</td><td>18.68</td><td>15.87</td><td>18.70</td><td>16.30</td><td>16,910</td></tr><tr><td>ReadAgent</td><td>6.48</td><td>5.6</td><td>5.31</td><td>4.23</td><td>7.66</td><td>6.62</td><td>9.18</td><td>7.91</td><td>7.16</td><td>6.09</td><td>643</td></tr><tr><td>MemoryBank</td><td>5.00</td><td>4.68</td><td>5.94</td><td>4.78</td><td>5.16</td><td>4.52</td><td>5.72</td><td>4.86</td><td>5.46</td><td>4.71</td><td>432</td></tr><tr><td>MemGPT</td><td>17.72</td><td>16.02</td><td>19.44</td><td>16.54</td><td>11.29</td><td>10.18</td><td>25.59</td><td>24.25</td><td>18.51</td><td>16.75</td><td>16,977</td></tr><tr><td>A-Mem</td><td>25.06</td><td>17.32</td><td>51.01</td><td>44.75</td><td>13.22</td><td>14.75</td><td>41.02</td><td>36.99</td><td>32.58</td><td>28.45</td><td>2,520</td></tr><tr><td>LightMem</td><td>24.96</td><td>21.66</td><td>20.55</td><td>18.39</td><td>19.21</td><td>17.68</td><td>33.79</td><td>29.66</td><td>24.63</td><td>21.85</td><td>612</td></tr><tr><td>Mem0</td><td>30.14</td><td>27.62</td><td>48.91</td><td>44.82</td><td>16.43</td><td>14.94</td><td>41.3</td><td>36.17</td><td>34.20</td><td>30.89</td><td>973</td></tr><tr><td></td><td>SimpleMem</td><td>43.46</td><td>38.82</td><td>58.62</td><td>50.10</td><td>19.76</td><td>18.04</td><td>51.12</td><td>43.53</td><td>43.24</td><td>37.62</td><td>531</td></tr><tr><td rowspan="7">GPT-4o</td><td>LoCoMo</td><td>28.00</td><td>18.47</td><td>9.09</td><td>5.78</td><td>16.47</td><td>14.80</td><td>61.56</td><td>54.19</td><td>28.78</td><td>23.31</td><td>16,910</td></tr><tr><td>ReadAgent</td><td>14.61</td><td>9.95</td><td>4.16</td><td>3.19</td><td>8.84</td><td>8.37</td><td>12.46</td><td>10.29</td><td>10.02</td><td>7.95</td><td>805</td></tr><tr><td>MemoryBank</td><td>6.49</td><td>4.69</td><td>2.47</td><td>2.43</td><td>6.43</td><td>5.30</td><td>8.28</td><td>7.10</td><td>5.92</td><td>4.88</td><td>569</td></tr><tr><td>MemGPT</td><td>30.36</td><td>22.83</td><td>17.29</td><td>13.18</td><td>12.24</td><td>11.87</td><td>40.16</td><td>36.35</td><td>25.01</td><td>21.06</td><td>16,987</td></tr><tr><td>A-Mem</td><td>32.86</td><td>23.76</td><td>39.41</td><td>31.23</td><td>17.10</td><td>15.84</td><td>44.43</td><td>38.97</td><td>33.45</td><td>27.45</td><td>1,216</td></tr><tr><td>LightMem</td><td>28.15</td><td>21.83</td><td>36.53</td><td>29.12</td><td>13.38</td><td>11.54</td><td>33.76</td><td>28.02</td><td>27.96</td><td>22.63</td><td>645</td></tr><tr><td>Mem0</td><td>35.13</td><td>27.56</td><td>52.38</td><td>44.15</td><td>17.73</td><td>15.92</td><td>39.12</td><td>35.43</td><td>36.09</td><td>30.77</td><td>985</td></tr><tr><td></td><td>SimpleMem</td><td>35.89</td><td>32.83</td><td>56.71</td><td>20.57</td><td>18.23</td><td>16.34</td><td>45.41</td><td>39.25</td><td>39.06</td><td>27.25</td><td>550</td></tr><tr><td rowspan="7">Qwen3-Plus</td><td>LoCoMo</td><td>24.15</td><td>18.94</td><td>16.57</td><td>13.28</td><td>11.81</td><td>10.58</td><td>38.58</td><td>28.16</td><td>22.78</td><td>17.74</td><td>16,910</td></tr><tr><td>ReadAgent</td><td>9.52</td><td>6.83</td><td>11.22</td><td>8.15</td><td>5.41</td><td>5.23</td><td>9.85</td><td>7.96</td><td>9.00</td><td>7.04</td><td>742</td></tr><tr><td>MemoryBank</td><td>5.25</td><td>4.94</td><td>1.77</td><td>6.26</td><td>5.88</td><td>6.00</td><td>6.90</td><td>5.57</td><td>4.95</td><td>5.69</td><td>302</td></tr><tr><td>MemGPT</td><td>25.80</td><td>17.50</td><td>24.10</td><td>18.50</td><td>9.50</td><td>7.80</td><td>40.20</td><td>42.10</td><td>24.90</td><td>21.48</td><td>16,958</td></tr><tr><td>A-Mem</td><td>26.50</td><td>19.80</td><td>46.10</td><td>35.10</td><td>11.90</td><td>11.50</td><td>43.80</td><td>36.50</td><td>32.08</td><td>25.73</td><td>1,427</td></tr><tr><td>LightMem</td><td>28.95</td><td>24.13</td><td>42.58</td><td>38.52</td><td>16.54</td><td>13.23</td><td>40.78</td><td>36.52</td><td>32.21</td><td>28.10</td><td>606</td></tr><tr><td>Mem0</td><td>32.42</td><td>21.24</td><td>47.53</td><td>39.82</td><td>17.18</td><td>14.53</td><td>46.25</td><td>37.52</td><td>35.85</td><td>28.28</td><td>1,020</td></tr><tr><td></td><td>SimpleMem</td><td>33.74</td><td>29.04</td><td>50.87</td><td>43.31</td><td>18.41</td><td>16.24</td><td>46.94</td><td>38.16</td><td>37.49</td><td>31.69</td><td>583</td></tr></table>

Table 2. Performance comparison on the LongMemEval benchmark. The evaluation uses gpt-4.1-mini as the judge. SimpleMem achieves the best overall performance while maintaining balanced capabilities across different sub-tasks.   

<table><tr><td>Model</td><td>Method</td><td>Temporal</td><td>Multi-Session</td><td>Knowledge-Update</td><td>Single-Session-User</td><td>Single-Session-Assistant</td><td>Single-Session-Preference</td><td>Average</td></tr><tr><td rowspan="4">GPT-4.1-mini</td><td>Full-context</td><td>27.06%</td><td>30.08%</td><td>41.03%</td><td>47.14%</td><td>32.14%</td><td>60.00%</td><td>39.57%</td></tr><tr><td>Mem0</td><td>40.60%</td><td>50.37%</td><td>69.23%</td><td>87.14%</td><td>48.21%</td><td>63.33%</td><td>59.81%</td></tr><tr><td>LightMem</td><td>85.71%</td><td>47.37%</td><td>92.30%</td><td>88.57%</td><td>21.43%</td><td>76.67%</td><td>68.67%</td></tr><tr><td>SimpleMem</td><td>83.46%</td><td>60.92%</td><td>79.48%</td><td>85.71%</td><td>75.00%</td><td>76.67%</td><td>76.87%</td></tr><tr><td rowspan="4">GPT-4.1</td><td>Full-context</td><td>51.88%</td><td>39.10%</td><td>70.51%</td><td>65.71%</td><td>96.43%</td><td>16.67%</td><td>56.72%</td></tr><tr><td>Mem0</td><td>43.61%</td><td>54.89%</td><td>75.64%</td><td>54.29%</td><td>39.29%</td><td>83.33%</td><td>58.51%</td></tr><tr><td>LightMem</td><td>84.96%</td><td>57.89%</td><td>89.74%</td><td>87.14%</td><td>71.43%</td><td>70.00%</td><td>76.86%</td></tr><tr><td>SimpleMem</td><td>86.47%</td><td>81.20%</td><td>80.76%</td><td>98.57%</td><td>76.79%</td><td>80.00%</td><td>83.97%</td></tr></table>

on GPT-4o, SimpleMem attains the highest Average F1 (39.06), outperforming Mem0 (36.09) and A-Mem (33.45).

A task-level breakdown on LoCoMo further highlights the balanced capabilities of SimpleMem. In SingleHop QA, SimpleMem consistently achieves the best performance (e.g., 51.12 F1 on GPT-4.1-mini), demonstrating precise factual retrieval. In more challenging MultiHop settings, SimpleMem significantly outperforms Mem0 and Light-Mem on GPT-4.1-mini, indicating its ability to bridge disconnected facts and support deep reasoning without relying on expensive iterative retrieval loops.

Results on LongMemEval-S (Table 2) further demonstrate the robustness of SimpleMem under extreme context lengths. Using the gpt-4.1-mini backbone, Simple-Mem achieves the highest average accuracy of $7 6 . 8 7 \%$ , outperforming LightMem $( 6 8 . 6 7 \% )$ and substantially exceeding Mem0 $( 5 9 . 8 1 \% )$ and the full-context baseline $( 3 9 . 5 7 \% )$ .

Gains are most pronounced in the challenging Multi-Session category, where SimpleMem attains $6 0 . 9 2 \%$ accuracy, compared to $4 7 . 3 7 \%$ for LightMem and $3 0 . 0 8 \%$ for full-context, highlighting its effectiveness in cross-session information integration under severe context constraints.

When scaled to the more capable gpt-4.1 backbone, SimpleMem maintains its state-of-the-art performance with an average accuracy of $8 3 . 9 7 \%$ . It is particularly strong in the Single-Session User task $( 9 8 . 5 7 \% )$ , demonstrating nearperfect recall of immediate user-provided details. While LightMem exhibits strong performance on temporally specific queries, SimpleMem offers a more balanced profile: it avoids catastrophic failures in assistant-focused recall (where LightMem drop significantly on the mini model) while maintaining high accuracy in complex multi-session retrieval. This balance suggests that SimpleMem’s structured indexing strategy effectively disentangles episodic

Table 3. Performance on the LoCoMo benchmark with Efficient Models (Small parameters). SimpleMem demonstrates robust performance even on 1.5B/3B models, often surpassing larger models using baseline memory systems.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">MultiHop</td><td colspan="2">Temporal</td><td colspan="2">OpenDomain</td><td colspan="2">SingleHop</td><td colspan="2">Average</td><td rowspan="2">Token Cost</td></tr><tr><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td><td>F1</td><td>BLEU</td></tr><tr><td rowspan="7">Qwen2.5-1.5b</td><td>LoCoMo</td><td>9.05</td><td>6.55</td><td>4.25</td><td>4.04</td><td>9.91</td><td>8.50</td><td>11.15</td><td>8.67</td><td>8.59</td><td>6.94</td><td>16,910</td></tr><tr><td>ReadAgent</td><td>6.61</td><td>4.93</td><td>2.55</td><td>2.51</td><td>5.31</td><td>12.24</td><td>10.13</td><td>7.54</td><td>6.15</td><td>6.81</td><td>752</td></tr><tr><td>MemoryBank</td><td>11.14</td><td>8.25</td><td>4.46</td><td>2.87</td><td>8.05</td><td>6.21</td><td>13.42</td><td>11.01</td><td>9.27</td><td>7.09</td><td>284</td></tr><tr><td>MemGPT</td><td>10.44</td><td>7.61</td><td>4.21</td><td>3.89</td><td>13.42</td><td>11.64</td><td>9.56</td><td>7.34</td><td>9.41</td><td>7.62</td><td>16,953</td></tr><tr><td>A-Mem</td><td>18.23</td><td>11.94</td><td>24.32</td><td>19.74</td><td>16.48</td><td>14.31</td><td>23.63</td><td>19.23</td><td>20.67</td><td>16.31</td><td>1,300</td></tr><tr><td>LightMem</td><td>16.43</td><td>11.39</td><td>22.92</td><td>18.56</td><td>15.06</td><td>11.23</td><td>23.28</td><td>19.24</td><td>19.42</td><td>15.11</td><td>605</td></tr><tr><td>Mem0</td><td>20.18</td><td>14.53</td><td>27.42</td><td>22.14</td><td>19.83</td><td>15.68</td><td>27.63</td><td>23.42</td><td>23.77</td><td>18.94</td><td>942</td></tr><tr><td></td><td>SimpleMem</td><td>21.85</td><td>16.10</td><td>29.12</td><td>23.50</td><td>21.05</td><td>16.80</td><td>28.90</td><td>24.50</td><td>25.23</td><td>20.23</td><td>678</td></tr><tr><td rowspan="7">Qwen2.5-3b</td><td>LoCoMo</td><td>4.61</td><td>4.29</td><td>3.11</td><td>2.71</td><td>4.55</td><td>5.97</td><td>7.03</td><td>5.69</td><td>4.83</td><td>4.67</td><td>16,910</td></tr><tr><td>ReadAgent</td><td>2.47</td><td>1.78</td><td>3.01</td><td>3.01</td><td>5.57</td><td>5.22</td><td>3.25</td><td>2.51</td><td>3.58</td><td>3.13</td><td>776</td></tr><tr><td>MemoryBank</td><td>3.60</td><td>3.39</td><td>1.72</td><td>1.97</td><td>6.63</td><td>6.58</td><td>4.11</td><td>3.32</td><td>4.02</td><td>3.82</td><td>298</td></tr><tr><td>MemGPT</td><td>5.07</td><td>4.31</td><td>2.94</td><td>2.95</td><td>7.04</td><td>7.10</td><td>7.26</td><td>5.52</td><td>5.58</td><td>4.97</td><td>16,961</td></tr><tr><td>A-Mem</td><td>12.57</td><td>9.01</td><td>27.59</td><td>25.07</td><td>7.12</td><td>7.28</td><td>17.23</td><td>13.12</td><td>16.13</td><td>13.62</td><td>1,137</td></tr><tr><td>LightMem</td><td>16.43</td><td>11.39</td><td>6.92</td><td>4.56</td><td>8.06</td><td>7.23</td><td>18.28</td><td>15.24</td><td>12.42</td><td>9.61</td><td>605</td></tr><tr><td>Mem0</td><td>16.89</td><td>11.54</td><td>8.52</td><td>6.23</td><td>10.24</td><td>8.82</td><td>16.47</td><td>12.43</td><td>13.03</td><td>9.76</td><td>965</td></tr><tr><td></td><td>SimpleMem</td><td>17.03</td><td>11.87</td><td>21.47</td><td>19.50</td><td>12.52</td><td>10.19</td><td>20.90</td><td>18.01</td><td>17.98</td><td>14.89</td><td>572</td></tr><tr><td rowspan="7">Qwen3-1.7b</td><td>LoCoMo</td><td>10.28</td><td>8.82</td><td>6.45</td><td>5.78</td><td>10.42</td><td>9.02</td><td>11.16</td><td>10.35</td><td>9.58</td><td>8.49</td><td>16,910</td></tr><tr><td>ReadAgent</td><td>7.50</td><td>5.60</td><td>3.15</td><td>2.95</td><td>6.10</td><td>12.45</td><td>10.80</td><td>8.15</td><td>6.89</td><td>7.29</td><td>784</td></tr><tr><td>MemoryBank</td><td>11.50</td><td>8.65</td><td>4.95</td><td>3.20</td><td>8.55</td><td>6.80</td><td>13.90</td><td>11.50</td><td>9.73</td><td>7.54</td><td>290</td></tr><tr><td>MemGPT</td><td>11.50</td><td>8.20</td><td>4.65</td><td>4.10</td><td>13.85</td><td>11.90</td><td>10.25</td><td>7.85</td><td>10.06</td><td>8.01</td><td>16,954</td></tr><tr><td>A-Mem</td><td>18.45</td><td>11.80</td><td>25.82</td><td>18.45</td><td>10.90</td><td>9.95</td><td>21.58</td><td>16.72</td><td>19.19</td><td>14.23</td><td>1,258</td></tr><tr><td>LightMem</td><td>14.84</td><td>11.56</td><td>9.35</td><td>7.85</td><td>13.76</td><td>10.59</td><td>28.14</td><td>22.89</td><td>16.52</td><td>13.22</td><td>679</td></tr><tr><td>Mem0</td><td>18.23</td><td>13.44</td><td>18.54</td><td>14.22</td><td>16.82</td><td>13.54</td><td>31.15</td><td>26.42</td><td>21.19</td><td>16.91</td><td>988</td></tr><tr><td></td><td>SimpleMem</td><td>20.85</td><td>15.42</td><td>26.75</td><td>18.63</td><td>17.92</td><td>14.15</td><td>32.85</td><td>26.46</td><td>24.59</td><td>18.67</td><td>730</td></tr><tr><td rowspan="7">Qwen3-8b</td><td>LoCoMo</td><td>13.50</td><td>9.20</td><td>6.80</td><td>5.50</td><td>10.10</td><td>8.80</td><td>14.50</td><td>11.20</td><td>11.23</td><td>8.68</td><td>16,910</td></tr><tr><td>ReadAgent</td><td>7.20</td><td>5.10</td><td>3.50</td><td>3.10</td><td>5.50</td><td>5.40</td><td>8.10</td><td>6.20</td><td>6.08</td><td>4.95</td><td>721</td></tr><tr><td>MemoryBank</td><td>9.50</td><td>7.10</td><td>3.80</td><td>2.50</td><td>7.50</td><td>6.50</td><td>9.20</td><td>7.50</td><td>7.50</td><td>5.90</td><td>287</td></tr><tr><td>MemGPT</td><td>14.20</td><td>9.80</td><td>5.50</td><td>4.20</td><td>12.50</td><td>10.80</td><td>11.50</td><td>9.10</td><td>10.93</td><td>8.48</td><td>16,943</td></tr><tr><td>A-Mem</td><td>20.50</td><td>13.80</td><td>22.50</td><td>18.20</td><td>13.20</td><td>10.50</td><td>26.80</td><td>21.50</td><td>20.75</td><td>16.00</td><td>1,087</td></tr><tr><td>LightMem</td><td>18.53</td><td>14.23</td><td>26.78</td><td>21.52</td><td>14.12</td><td>11.24</td><td>29.48</td><td>23.83</td><td>22.23</td><td>17.71</td><td>744</td></tr><tr><td>Mem0</td><td>22.42</td><td>16.83</td><td>32.48</td><td>26.13</td><td>15.23</td><td>12.54</td><td>33.05</td><td>27.24</td><td>25.80</td><td>20.69</td><td>1,015</td></tr><tr><td></td><td>SimpleMem</td><td>28.97</td><td>24.93</td><td>42.85</td><td>36.49</td><td>15.35</td><td>13.90</td><td>46.62</td><td>40.69</td><td>33.45</td><td>29.00</td><td>621</td></tr></table>

noise from salient facts, providing a reliable memory substrate for long-term interaction.

Token Efficiency. A key strength of SimpleMem lies in its inference-time efficiency. As reported in the rightmost columns of Tables 1 and 3, full-context approaches such as LOCOMO and MEMGPT consume approximately 16,900 tokens per query. In contrast, SimpleMem reduces token usage by roughly $3 0 \times$ , averaging 530-580 tokens per query. Furthermore, compared to optimized retrieval baselines like Mem0 ( $\mathord { \sim } 9 8 0$ tokens) and A-Mem $( \sim 1 , 2 0 0 +$ tokens), SimpleMem reduces token usage by $40 { - } 5 0 \%$ while delivering superior accuracy. For instance, on GPT-4.1-mini, Simple-Mem uses only 531 tokens to achieve state-of-the-art performance, whereas ReadAgent consumes more (643 tokens) but achieves far lower accuracy (7.16 F1). This validates the effectiveness of SimpleMem in strictly controlling context bandwidth without sacrificing information density.

Performance on Smaller Models. Table 3 highlights the ability of SimpleMem to empower smaller parameter models. On Qwen3-8b, SimpleMem achieves an impressive

Average F1 of 33.45, significantly surpassing Mem0 (25.80) and LightMem (22.23). Crucially, a 3B-parameter model (Qwen2.5-3b) paired with SimpleMem achieves 17.98 F1, outperforming the same model with Mem0 (13.03) by nearly 5 points. Even on the extremely lightweight Qwen2.5-1.5b, SimpleMem maintains robust performance (25.23 F1), beating larger models using inferior memory strategies (e.g., Qwen3-1.7b with Mem0 scores 21.19).

# 3.3. Efficiency Analysis

We conduct a comprehensive evaluation of computational efficiency, examining both end-to-end system latency and the scalability of memory indexing and retrieval. To assess practical deployment viability, we measured the full lifecycle costs on the LoCoMo-10 dataset using GPT-4.1-mini.

As illustrated in Table 4, SimpleMem exhibits superior efficiency across all operational phases. In terms of memory construction, our system achieves the fastest processing speed at 92.6 seconds per sample. This represents a dramatic improvement over existing baselines, outperforming

Mem0 by approximately $1 4 \times$ (1350.9s) and A-Mem by over $5 0 \times$ (5140.5s). This massive speedup is directly attributable to our semantic structured compression, which processes data in a streamlined single pass, thereby avoiding the complex graph updates required by Mem0 or the iterative summarization overheads inherent to A-Mem.

Beyond construction, SimpleMem also maintains the lowest retrieval latency at 388.3 seconds per sample, which is approximately $33 \%$ faster than LightMem and Mem0. This gain arises from the adaptive retrieval mechanism, which dynamically limits retrieval scope and prioritizes high-level abstract representations before accessing finegrained details. By restricting retrieval to only the most relevant memory entries, the system avoids the expensive neighbor traversal and expansion operations that commonly dominate the latency of graph-based memory systems.

When considering the total time-to-insight, SimpleMem achieves a $4 \times$ speedup over Mem0 and a $1 2 \times$ speedup over A-Mem. Crucially, this efficiency does not come at the expense of performance. On the contrary, SimpleMem achieves the highest Average F1 among all compared methods. These results support our central claim that structured semantic compression and adaptive retrieval produce a more compact and effective reasoning substrate than raw context retention or graph-centric memory designs, enabling a superior balance between accuracy and computational efficiency.

Table 4. Comparison of construction time, retrieval time, total experiment time, and average F1 score across different memory systems (tested on LoCoMo-10 with GPT-4.1-mini; time values are reported as per-sample averages on LoCoMo-10).   

<table><tr><td>Model</td><td>Construction Time</td><td>Retrieval Time</td><td>Total Time</td><td>Average F1</td></tr><tr><td>A-mem</td><td>5140.5s</td><td>796.7s</td><td>5937.2s</td><td>32.58</td></tr><tr><td>Lightmem</td><td>97.8s</td><td>577.1s</td><td>675.9s</td><td>24.63</td></tr><tr><td>Mem0</td><td>1350.9s</td><td>583.4s</td><td>1934.3s</td><td>34.20</td></tr><tr><td>SimpleMem</td><td>92.6s</td><td>388.3s</td><td>480.9s</td><td>43.24</td></tr></table>

# 3.4. Ablation Study

In addition, we conduct an ablation study using the GPT-4.1-mini backend. We investigate the contribution of three key components. The results are summarized in Table 5.

Impact of Semantic Structured Compression. Replacing the proposed compression pipeline with standard chunkbased storage leads to a substantial degradation in temporal reasoning performance. Specifically, removing semantic structured compression reduces the Temporal F1 by $56 . 7 \%$ , from 58.62 to 25.40. This drop indicates that without context normalization steps such as resolving coreferences and converting relative temporal expressions into absolute timestamps, the retriever struggles to disambiguate events along the timeline. As a result, performance regresses to levels comparable to conventional retrieval-augmented generation systems that rely on raw or weakly structured context.

Impact of Online Semantic Synthesis. Disabling online semantic synthesis results in a $3 1 . 3 \%$ decrease in multi-hop reasoning performance. Without on-the-fly consolidation during the write phase, semantically related facts accumulate as fragmented entries, forcing the retriever to assemble dispersed evidence at query time. This fragmentation inflates contextual redundancy and rapidly exhausts the available context window in complex queries. The observed degradation demonstrates that proactive, intra-session synthesis is essential for maintaining a compact and semantically coherent memory topology, and for transforming local observations into reusable, high-density abstractions.

Intent-Aware Retrieval Planning. Removing intent-aware retrieval planning and reverting to a fixed-depth retrieval strategy primarily degrades performance on open-domain and single-hop tasks, with drops of $2 6 . 6 \%$ and $1 9 . 4 \%$ , respectively. In the absence of query-aware adjustment, the system either retrieves insufficient context for entity-specific queries or introduces excessive irrelevant information for simple queries. These results highlight the importance of dynamically modulating retrieval scope to balance relevance and efficiency during inference.

# 3.5. Case Study: Long-Term Temporal Grounding

To illustrate how SimpleMem handles long-horizon conversational history, Figure 3 presents a representative multisession example spanning two weeks and approximately 24,000 raw tokens. SimpleMem filters low-information dialogue during ingestion and retains only high-utility memory entries, reducing the stored memory to about 800 tokens without losing task-relevant content.

Temporal Normalization. Relative temporal expressions such as last week” and yesterday” refer to different absolute times across sessions. SimpleMem resolves it into absolute timestamps at memory construction time, ensuring consistent temporal grounding over long interaction gaps.

Precise Retrieval. When queried about Sarah’s past artworks, the intent-aware retrieval planner infers both the semantic focus (art-related activities) and the temporal constraints implied by the query. The system then performs parallel multi-view retrieval, combining semantic similarity with symbolic filtering to exclude unrelated activities and return only temporally valid entries. This example demonstrates how structured compression, temporal normalization, and adaptive retrieval jointly enable reliable long-term reasoning under extended interaction histories.

# 4. Related Work

Memory Systems for LLM Agents. Recent approaches manage memory through virtual context or structured representations. Virtual context methods, including MEMGPT (Packer et al., 2023), MEMORYOS (Kang et al., 2025), and

Table 5. Full ablation analysis with GPT-4.1-mini backend. The "Diff" columns indicate the percentage drop relative to the full SimpleMem model. The results confirm that each stage contributes significantly to specific reasoning capabilities.   

<table><tr><td rowspan="2">Configuration</td><td colspan="2">Multi-hop</td><td colspan="2">Temporal</td><td colspan="2">Open Domain</td><td colspan="2">Single Hop</td><td colspan="2">Average</td></tr><tr><td>F1</td><td>Diff</td><td>F1</td><td>Diff</td><td>F1</td><td>Diff</td><td>F1</td><td>Diff</td><td>F1</td><td>Diff</td></tr><tr><td>Full SimpleMem</td><td>43.46</td><td>-</td><td>58.62</td><td>-</td><td>19.76</td><td>-</td><td>51.12</td><td>-</td><td>43.24</td><td>-</td></tr><tr><td>w/o Semantic Compression</td><td>34.20</td><td>(↓21.3%)</td><td>25.40</td><td>(↓56.7%)</td><td>17.50</td><td>(↓11.4%)</td><td>48.05</td><td>(↓6.0%)</td><td>31.29</td><td>(↓27.6%)</td></tr><tr><td>w/o Online Synthesis</td><td>29.85</td><td>(↓31.3%)</td><td>55.10</td><td>(↓6.0%)</td><td>18.20</td><td>(↓7.9%)</td><td>49.80</td><td>(↓2.6%)</td><td>38.24</td><td>(↓11.6%)</td></tr><tr><td>w/o Intent-Aware Retrieval</td><td>38.60</td><td>(↓11.2%)</td><td>56.80</td><td>(↓3.1%)</td><td>14.50</td><td>(↓26.6%)</td><td>41.20</td><td>(↓19.4%)</td><td>37.78</td><td>(↓12.6%)</td></tr></table>

![](images/c68e3aa457557386ade79908e3bcc1ec87d25a6ea9c978e150de99e484b2cdec.jpg)  
Figure 3. A Case of SimpleMem for Long-Term Multi-Session Dialogues. SimpleMem processes multi-session dialogues by filtering redundant content, normalizing temporal references, and organizing memories into compact representations. During retrieval, it adaptively combines semantic, lexical, and symbolic signals to select relevant entries.

SCM (Wang et al., 2023), extend interaction length via paging or stream-based controllers (Wang et al., 2024b) but typically store raw conversation logs, leading to redundancy and increasing processing costs. In parallel, structured and graph-based systems, such as MEMORYBANK (Zhong et al., 2024), MEM0 (Dev & Taranjeet, 2024), ZEP (Rasmussen et al., 2025), A-MEM (Xu et al., 2025), and O-MEM (Wang et al., 2025), impose structural priors to improve coherence but still rely on raw or minimally processed text, preserving referential and temporal ambiguities that degrade long-term retrieval. In contrast, SimpleMem adopts a semantic compression mechanism that converts dialogue into independent, self-contained facts, explicitly resolving referential and temporal ambiguities prior to storage.

Context Management and Retrieval Efficiency. Beyond memory storage, efficient access to historical information remains a core challenge. Existing approaches primarily rely on either long-context models or retrieval-augmented generation (RAG). Although recent LLMs support extended context windows (OpenAI, 2025; Deepmind, 2025; Anthropic, 2025), and prompt compression methods aim to reduce costs (Jiang et al., 2023a; Liskavetsky et al., 2025), empirical studies reveal the “Lost-in-the-Middle” effect (Liu et al., 2023; Kuratov et al., 2024), where reasoning performance degrades as context length increases, along-

side prohibitive computational overhead for lifelong agents. RAG-based methods (Lewis et al., 2020; Asai et al., 2023; Jiang et al., 2023b), including structurally enhanced variants such as GRAPHRAG (Edge et al., 2024; Zhao et al., 2025) and LIGHTRAG (Guo et al., 2024), decouple memory from inference but are largely optimized for static knowledge bases, limiting their effectiveness for dynamic, timesensitive episodic memory. In contrast, SimpleMem improves retrieval efficiency through Intent-Aware Retrieval Planning, jointly leveraging semantic, lexical, and symbolic signals to construct query-specific retrieval plans and dynamically adapt the retrieval budget, achieving token-efficient reasoning under constrained context budgets.

# 5. Conclusion

We introduce SimpleMem, an efficient agent memory architecture grounded in the principle of semantic lossless compression. By treating memory as an active process rather than passive storage, SimpleMem integrates Semantic Structured Compression to filter noise at the source, Online Semantic Synthesis to consolidate fragmented observations during writing, and Intent-Aware Retrieval Planning to dynamically adapt retrieval scope. Empirical evaluation on the LoCoMo and LongMemEval-S benchmark demonstrates the effectiveness and efficiency of our method.

# Acknowledgement

This work is partially supported by Amazon Research Award, Cisco Faculty Research Award, and Coefficient Giving. This work is also partly supported by the National Center for Transportation Cybersecurity and Resiliency (TraCR) (a U.S. Department of Transportation National University Transportation Center) headquartered at Clemson University, Clemson, South Carolina, USA (USDOT Grant #69A3552344812). Any opinions, findings, conclusions, and recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of TraCR, and the U.S. Government assumes no liability for the contents or use thereof.

# References

Anthropic. Claude 3.7 sonnet and claude code. https://www.anthropic.com/news/ claude-3-7-sonnet, 2025.   
Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. Selfrag: Learning to retrieve, generate, and critique through self-reflection. arXiv preprint arXiv:2310.11511, 2023.   
Deepmind, G. Gemini 2.5: Our most intelligent AI model — blog.google. https://blog. google/technology/google-deepmind/ I gemini-model-thinking-updates-march-202 #gemini-2-5-thinking, 2025. Accessed: 2025- 03-25.   
Dev, K. and Taranjeet, S. mem0: The memory layer for ai agents. https://github.com/mem0ai/mem0, 2024.   
Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., and Larson, J. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.   
Fang, J., Deng, X., Xu, H., Jiang, Z., Tang, Y., Xu, Z., Deng, S., Yao, Y., Wang, M., Qiao, S., et al. Lightmem: Lightweight and efficient memory-augmented generation. arXiv preprint arXiv:2510.18866, 2025.   
Guo, Z., Xia, L., Yu, Y., Ao, T., and Huang, C. Lightrag: Simple and fast retrieval-augmented generation. arXiv preprint arXiv:2410.05779, 2024.   
Hu, Y., Liu, S., Yue, Y., Zhang, G., Liu, B., Zhu, F., Lin, J., Guo, H., Dou, S., Xi, Z., et al. Memory in the age of ai agents. arXiv preprint arXiv:2512.13564, 2025.   
Jiang, H., Wu, Q., Lin, C.-Y., Yang, Y., and Qiu, L. Llmlingua: Compressing prompts for accelerated inference of large language models. arXiv preprint arXiv:2310.05736, 2023a.

Jiang, Z., Xu, F. F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan, J., and Neubig, G. Active retrieval augmented generation. arXiv preprint arXiv:2305.06983, 2023b.   
Kang, J., Ji, M., Zhao, Z., and Bai, T. Memory os of ai agent. arXiv preprint arXiv:2506.06326, 2025.   
Kumaran, D., Hassabis, D., and McClelland, J. L. What learning systems do intelligent agents need? complementary learning systems theory updated. Trends in cognitive sciences, 20(7):512–534, 2016.   
Kuratov, Y. et al. In case of context: Investigating the effects of long context on language model performance. arXiv preprint, 2024.   
Lee, K.-H., Chen, X., Furuta, H., Canny, J., and Fischer, I. A human-inspired reading agent with gist memory of very long contexts. arXiv preprint arXiv:2402.09727, 2024.   
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-t., Rocktäschel, T., et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.   
Li, Z., Song, S., Wang, H., Niu, S., Chen, D., Yang, J., 5/ Xi, C., Lai, H., Zhao, J., Wang, Y., Ren, J., Lin, Z., Huo, J., Chen, T., Chen, K., Li, K.-R., Yin, Z., Yu, Q., Tang, B., Yang, H., Xu, Z., and Xiong, F. Memos: An operating system for memory-augmented generation (mag) in large language models. ArXiv, abs/2505.22101, 2025. URL https://api.semanticscholar. org/CorpusID:278960153.   
Liskavetsky, A. et al. Compressor: Context-aware prompt compression for enhanced llm inference. arXiv preprint, 2025.   
Liu, J., Xiong, K., Xia, P., Zhou, Y., Ji, H., Feng, L., Han, S., Ding, M., and Yao, H. Agent0-vl: Exploring selfevolving agent for tool-integrated vision-language reasoning. arXiv preprint arXiv:2511.19900, 2025.   
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172, 2023.   
Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., and Fang, Y. Evaluating very long-term conversational memory of llm agents, 2024. URL https: //arxiv.org/abs/2402.17753.   
OpenAI. Introducing gpt-5. https://openai.com/ index/introducing-gpt-5/, 2025.

Ouyang, S., Yan, J., Hsu, I., Chen, Y., Jiang, K., Wang, Z., Han, R., Le, L. T., Daruki, S., Tang, X., et al. Reasoningbank: Scaling agent self-evolving with reasoning memory. arXiv preprint arXiv:2509.25140, 2025.   
Packer, C., Fang, V., Patil, S. G., Lin, K., Wooders, S., and Gonzalez, J. Memgpt: Towards llms as operating systems. ArXiv, abs/2310.08560, 2023. URL https://api.semanticscholar. org/CorpusID:263909014.   
Qiu, J., Qi, X., Zhang, T., Juan, X., Guo, J., Lu, Y., Wang, Y., Yao, Z., Ren, Q., Jiang, X., et al. Alita: Generalist agent enabling scalable agentic reasoning with minimal predefinition and maximal self-evolution. arXiv preprint arXiv:2505.20286, 2025.   
Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., and Chalef, D. Zep: a temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025.   
Tang, X., Qin, T., Peng, T., Zhou, Z., Shao, D., Du, T., Wei, X., Xia, P., Wu, F., Zhu, H., et al. Agent kb: Leveraging cross-domain experience for agentic problem solving. arXiv preprint arXiv:2507.06229, 2025.   
Team, T. D., Li, B., Zhang, B., Zhang, D., Huang, F., Li, G., Chen, G., Yin, H., Wu, J., Zhou, J., et al. Tongyi deepresearch technical report. arXiv preprint arXiv:2510.24701, 2025.   
Tu, A., Xuan, W., Qi, H., Huang, X., Zeng, Q., Talaei, S., Xiao, Y., Xia, P., Tang, X., Zhuang, Y., et al. Position: The hidden costs and measurement gaps of reinforcement learning with verifiable rewards. arXiv preprint arXiv:2509.21882, 2025.   
Wang, B., Liang, X., Yang, J., Huang, H., Wu, S., Wu, P., Lu, L., Ma, Z., and Li, Z. Enhancing large language model with self-controlled memory framework. arXiv preprint arXiv:2304.13343, 2023.   
Wang, P., Tian, M., Li, J., Liang, Y., Wang, Y., Chen, Q., Wang, T., Lu, Z., Ma, J., Jiang, Y. E., et al. O-mem: Omni memory system for personalized, long horizon, self-evolving agents. arXiv e-prints, pp. arXiv–2511, 2025.   
Wang, T., Tao, M., Fang, R., Wang, H., Wang, S., Jiang, Y. E., and Zhou, W. Ai persona: Towards life-long personalization of llms. arXiv preprint arXiv:2412.13103, 2024a.   
Wang, Y. and Chen, X. Mirix: Multi-agent memory system for llm-based agents. arXiv preprint arXiv:2507.07957, 2025.

Wang, Z. Z., Mao, J., Fried, D., and Neubig, G. Agent workflow memory. arXiv preprint arXiv:2409.07429, 2024b.   
Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., and Yu, D. Longmemeval: Benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813, 2024.   
Xia, P., Zeng, K., Liu, J., Qin, C., Wu, F., Zhou, Y., Xiong, C., and Yao, H. Agent0: Unleashing self-evolving agents from zero data via tool-integrated reasoning. arXiv preprint arXiv:2511.16043, 2025.   
Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang, Y. A-mem: Agentic memory for llm agents. ArXiv, abs/2502.12110, 2025. URL https: //api.semanticscholar.org/CorpusID: 276421617.   
Yan, B., Li, C., Qian, H., Lu, S., and Liu, Z. General agentic memory via deep research. arXiv preprint arXiv:2511.18423, 2025.   
Yang, B., Xu, L., Zeng, L., Liu, K., Jiang, S., Lu, W., Chen, H., Jiang, X., Xing, G., and Yan, Z. Contextagent: Context-aware proactive llm agents with open-world sensory perceptions. arXiv preprint arXiv:2505.14668, 2025.   
Zhao, Y., Zhu, J., Guo, Y., He, K., and Li, X. Eˆ 2graphrag: Streamlining graph-based rag for high efficiency and effectiveness. arXiv preprint arXiv:2505.24226, 2025.   
Zhong, W., Guo, L., Gao, Q., Ye, H., and Wang, Y. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19724–19731, 2024.

# A. Detailed System Prompts

To ensure full reproducibility of the SimpleMem pipeline, we provide the exact system prompts used in the key processing stages. All prompts are designed to be model-agnostic but were optimized for GPT-4o-mini in our experiments to ensure cognitive economy.

# A.1. Stage 1: Semantic Structured Compression Prompt

This prompt performs entropy-aware filtering and context normalization. Its goal is to transform raw dialogue windows into compact, context-independent memory units while excluding low-information interaction content.

Listing 1. Prompt for Semantic Structured Compression and Normalization.   
```yaml
You are a memory encoder in a long-term memory system. Your task is to transform raw conversational input into compact, self-contained memory units.   
INPUT METADATA: Window Start Time: {window_start_time} (ISO 8601) Participants: {speakers_list}   
INSTRUCTIONS:   
1. Information Filtering: -Discard social filler, acknowledgements, and conversational routines that introduce no new factual or semantic information. -Discard redundant confirmations unless they modify or finalize a decision. -If no informative content is present, output an empty list.   
2. Context Normalization: -Resolve all pronouns and implicit references into explicit entity names. -Ensure each memory unit is interpretable without access to prior dialogue.   
3.Temporal Normalization: - Convert relative temporal expressions (e.g., "tomorrow", "last week") into absolute ISO 8601 timestamps using the window start time..   
4. Memory Unit Extraction: - Decompose complex utterances into minimal, indivisible factual statements.   
INPUT DIALOGUE: {dialogue_window}   
OUTPUT FORMAT (JSON): { "memory.units": [ { "content": "Alice agreed to meet Bob at the Starbucks on 5th Avenue on 2025-11-20T14 :00:00.", "entities": ["Alice", "Bob", "Starbucks", "5th Avenue"], "topic": "Meeting Planning", "timestamp": "2025-11-20T14:00:00", "salience": "high" } ] 
```

# A.2. Stage 2: Adaptive Retrieval Planning Prompt

This prompt analyzes the user query prior to retrieval. Its purpose is to estimate query complexity and generate a structured retrieval plan that adapts retrieval scope accordingly.

Listing 2. Prompt for Query Analysis and Adaptive Retrieval Planning.   
```txt
Analyze the following user query and generate a retrieval plan. Your objective is to retrieve sufficient information while minimizing unnecessary context usage. 
```

```txt
USER QUERY: {user_query}   
INSTRUCTIONS:   
1. Query Complexity Estimation: Assign "LOW" if the query can be answered via direct fact lookup or a single memory unit. Assign "HIGH" if the query requires aggregation across multiple events, temporal comparison, or synthesis of patterns.   
2. Retrieval Signals: - Lexical layer: extract exact keywords or entity names. Temporal layer: infer absolute time ranges if relevant. Semantic layer: rewrite the query into a declarative form suitable for semantic matching.   
OUTPUT FORMAT (JSON): {"complexity": "HIGH", "retrieval_rationale": "The query requires reasoning over multiple temporally separated events.", "lexical_keys": ["Starbucks", "Bob"], "temporalConstraints": { "start": "2025-11-01T00:00:00", "end": "2025-11-30T23:59:59" }, "semantic_query": "The user is asking about the scheduled meeting with Bob, including location and time." } 
```

# A.3. Stage 3: Reconstructive Synthesis Prompt

This prompt guides the final answer generation using retrieved memory. It combines high-level abstract representations with fine-grained factual details to produce a grounded response.

Listing 3. Prompt for Reconstructive Synthesis (Answer Generation).   
```txt
You are an assistant with access to a structured long-term memory.   
USER QUERY: {user_query}   
RETRIEVED MEMORY (Ordered by Relevance): [ABSTRACT REPRESENTATIONS]: {retrieved_ABstracts} [DETAILED MEMORY UNITS]: {retrieved.units}   
INSTRUCTIONS:   
1. Hierarchical Reasoning: - Use abstract representations to capture recurring patterns or general user preferences. - Use detailed memory units to ground the response with specific facts..   
2. Conflict Handling: - If inconsistencies arise, prioritize the most recent memory unit. - Optionally reference abstract patterns when relevant.   
3. Temporal Consistency: - Ensure all statements respect the timestamps provided in memory. 
```

```txt
4. Faithfulness:  
- Base the answer strictly on the retrieved memory.  
- If required information is missing, respond with: "I do not have enough information in my memory."  
FINAL ANSWER: 
```

# A.4. LongMemEval Evaluation Prompt

For the LongMemEval benchmark, we employed gpt-4.1-mini as the judge to evaluate the correctness of the agent’s responses. The prompt strictly instructs the judge to focus on semantic and temporal consistency rather than exact string matching. The specific prompt template used is provided below:

Listing 4. LLM-as-a-Judge Evaluation Prompt.   
```txt
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.   
You will be given the following data: (1) a question (posed by one user to another user), (2) a gold' (ground truth) answer, (3) a generated answer   
which you will score as CORRECT/WRONG.   
The point of the question is to ask about something one user should know about the other user based on their prior conversations.   
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:   
Question: Do you remember what I got the last time I went to Hawaii?   
Gold answer: A shell necklace   
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.   
For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.   
Now it's time for the real question: Question: {question}   
Gold answer: {gold_answer}   
Generated answer: {generated_answer}   
First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.   
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.   
Just return the label CORRECT or WRONG in a json format with the key as "label". 
```

# B. Extended Implementation Details and Experiments

# B.1. Dataset Description

LoCoMo (Maharana et al., 2024) is specifically designed to test the limits of LLMs in processing long-term conversational dependencies. The dataset comprises conversation samples ranging from 200 to 400 turns, containing complex temporal shifts and interleaved topics. The evaluation set consists of 1,986 questions categorized into four distinct reasoning types: (1) Multi-Hop Reasoning: Questions requiring the synthesis of information from multiple disjoint turns (e.g., “Based on what X said last week and Y said today...”); (2) Temporal Reasoning: Questions testing the model’s

ability to understand event sequencing and absolute timelines (e.g., “Did X happen before Y?”); (3) Open Domain: General knowledge questions grounded in the conversation context; (4) Single Hop: Direct retrieval tasks requiring exact matching of specific facts.

LongMemEval-S benchmark. The defining characteristic of this dataset is its extreme context length, which poses a unique and severe challenge for memory systems. Unlike standard benchmarks, LongMemEval-S requires the system to precisely locate specific answers across various sub-categories (e.g., temporal events, user preferences) within an exceptionally long interaction history. This massive search space significantly escalates the difficulty of retrieval and localization, serving as a rigorous stress test for the system’s precision. We utilized an LLM-as-a-judge protocol (using gpt-4.1-mini) to score the correctness of generated answers against ground-truth references, categorizing responses as either CORRECT or WRONG based on semantic and temporal alignment. The full evaluation prompt is provided in Appendix A.4.

# B.2. Hyperparameter Configuration

Table 7 summarizes the hyperparameters used to obtain the results reported in Section 3. These values were selected to balance memory compactness and retrieval recall, with particular attention to the thresholds governing semantic structured compression and recursive consolidation.

# B.3. Hyperparameter Sensitivity Analysis

To assess the effectiveness of semantic structured compression and to motivate the design of adaptive retrieval, we analyze system sensitivity to the number of retrieved memory entries $( k )$ . We vary $k$ from 1 to 20 and report the average F1 score on the LoCoMo benchmark using the GPT-4.1-mini backend.

Table 6 provides two key observations. First, rapid performance saturation is observed at low retrieval depth. SimpleMem achieves strong performance with a single retrieved entry (35.20 F1) and reaches approximately $9 9 \%$ of its peak performance at $k = 3$ . This behavior indicates that semantic structured compression produces memory units with high information content, often sufficient to answer a query without aggregating many fragments.

Second, robustness to increased retrieval depth distinguishes SimpleMem from baseline methods. While approaches such as MemGPT experience performance degradation at larger $k$ , SimpleMem maintains stable accuracy even when retrieving up to 20 entries. This robustness enables adaptive retrieval to safely

Table 6. Performance sensitivity to retrieval count (k). Simple-Mem demonstrates "Rapid Saturation," reaching near-optimal performance at $k = 3$ (42.85) compared to its peak at $k = 1 0$ (43.45). This validates the high information density of Atomic Entries, proving that huge context windows are often unnecessary for accuracy.

<table><tr><td rowspan="2">Method</td><td colspan="5">Top-k</td></tr><tr><td>k=1</td><td>k=3</td><td>k=5</td><td>k=10</td><td>k=20</td></tr><tr><td>ReadAgent</td><td>6.12</td><td>8.45</td><td>9.18</td><td>8.92</td><td>8.50</td></tr><tr><td>MemGPT</td><td>18.40</td><td>22.15</td><td>25.59</td><td>24.80</td><td>23.10</td></tr><tr><td>SimpleMem</td><td>35.20</td><td>42.85</td><td>43.24</td><td>43.45</td><td>43.40</td></tr></table>

expand context for complex reasoning tasks without introducing excessive irrelevant information.

Table 7. Detailed hyperparameter configuration for SimpleMem. The system employs adaptive thresholds to balance memory compactness and retrieval effectiveness.   

<table><tr><td>Module</td><td>Parameter</td><td>Value / Description</td></tr><tr><td rowspan="4">Stage 1: Semantic Structured Compression</td><td>Window Size (W)</td><td>20 turns</td></tr><tr><td>Sliding Stride</td><td>5 turns (25% overlap)</td></tr><tr><td>Model Backend</td><td>gpt-4.1-mini (temperature = 0.0)</td></tr><tr><td>Output Constraint</td><td>Strict JSON schema enforced</td></tr><tr><td rowspan="3">Stage 2: Online Semantic Synthesis</td><td>Embedding Model</td><td>Qwen3-embedding-0.6b (1024 dimensions)</td></tr><tr><td>Vector Database</td><td>LanceDB (v0.4.5) with IVF-PQ indexing</td></tr><tr><td>Stored Metadata</td><td>timestamp, entities,topic,salience</td></tr><tr><td rowspan="5">Stage 3: Intent-Aware Retrieval Planning</td><td>Query Complexity Estimator</td><td>gpt-4.1-mini</td></tr><tr><td>Retrieval Range</td><td>[3, 20]</td></tr><tr><td>Minimum Depth</td><td>1</td></tr><tr><td>Maximum Depth</td><td>20</td></tr><tr><td>Re-ranking</td><td>Disabled (multi-view score fusion applied directly)</td></tr></table>
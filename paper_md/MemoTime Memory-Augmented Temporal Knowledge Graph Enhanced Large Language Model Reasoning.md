# MemoTime: Memory-Augmented Temporal Knowledge Graph Enhanced Large Language Model Reasoning

Xingyu Tan UNSW

Data61, CSIRO

Sydney, Australia

xingyu.tan@unsw.edu.au

Xiaoyang Wang∗

UNSW

Sydney, Australia

xiaoyang.wang1@unsw.edu.au

Qing Liu

Data61, CSIRO

Hobart, Australia

q.liu@data61.csiro.au

Xiwei Xu

Data61, CSIRO

Sydney, Australia

xiwei.xu@data61.csiro.au

Xin Yuan

Data61, CSIRO

UNSW

Sydney, Australia

xin.yuan@data61.csiro.au

Liming Zhu

Data61, CSIRO

Sydney, Australia

liming.zhu@data61.csiro.au

Wenjie Zhang

UNSW

Sydney, Australia

wenjie.zhang@unsw.edu.au

# Abstract

Large Language Models (LLMs) have achieved impressive reasoning abilities, but struggle with temporal understanding, especially when questions involve multiple entities, compound operators, and evolving event sequences. Temporal Knowledge Graphs (TKGs), which capture vast amounts of temporal facts in a structured format, offer a reliable source for temporal reasoning. However, existing TKG-based LLM reasoning methods still struggle with four major challenges: maintaining temporal faithfulness in multi-hop reasoning, achieving multi-entity temporal synchronization, adapting retrieval to diverse temporal operators, and reusing prior reasoning experience for stability and efficiency. To address these issues, we propose MemoTime, a memory-augmented temporal knowledge graph framework that enhances LLM reasoning through structured grounding, recursive reasoning, and continual experience learning. MemoTime decomposes complex temporal questions into a hierarchical Tree of Time, enabling operator-aware reasoning that enforces monotonic timestamps and co-constrains multiple entities under unified temporal bounds. A dynamic evidence retrieval layer adaptively selects operator-specific retrieval strategies, while a self-evolving experience memory stores verified reasoning traces, toolkit decisions, and sub-question embeddings for cross-type reuse. Comprehensive experiments on multiple temporal QA benchmarks show that MemoTime achieves overall state-of-the-art results, outperforming the strong baseline by up to $2 4 . 0 \%$ . Furthermore, Memo-Time enables smaller models (e.g., Qwen3-4B) to achieve reasoning performance comparable to that of GPT-4-Turbo.

# CCS Concepts

• Information systems Question answering.

∗Corresponding author.

![](images/b9df47bd59e39d30b1384072b725f29d8a7b3e537d3c653888620064f35f112f.jpg)

This work is licensed under a Creative Commons Attribution 4.0 International License.

WWW ’26, Dubai, United Arab Emirates

© 2026 Copyright held by the owner/author(s).

ACM ISBN 979-8-4007-2307-0/2026/04

https://doi.org/10.1145/3774904.3792581

# Keywords

Large Language Models; Retrieval-Augmented Generation; Temporal Knowledge Graph; Temporal Knowledge Graph Question Answering; Memory-Augmented Retrieval-Augmented Generation

# ACM Reference Format:

Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, Liming Zhu, and Wenjie Zhang. 2026. MemoTime: Memory-Augmented Temporal Knowledge Graph Enhanced Large Language Model Reasoning. In Proceedings of the ACM Web Conference 2026 (WWW ’26), April 13–17, 2026, Dubai, United Arab Emirates. ACM, New York, NY, USA, 21 pages. https://doi.org/10.1145/3774904.3792581

# 1 Introduction

Large Language Models (LLMs) have demonstrated strong performance across a wide range of tasks by scaling to billions of parameters and pre-training on massive and diverse text corpora [2]. However, due to the prohibitive cost of retraining, these models are inherently static. This limitation results in factual gaps, temporal inconsistencies, and reasoning hallucinations when LLMs are required to process time-sensitive or evolving information [20, 33, 45].

Retrieval-Augmented Generation (RAG) has emerged as an effective paradigm to mitigate these limitations by allowing LLMs to access external information sources during inference [14, 15, 69]. Typical RAG systems embed both questions and documents into a shared vector space and retrieve semantically similar passages [27, 48]. Although effective in many cases, such retrieval often prioritizes semantic similarity while neglecting structural and temporal dependencies across entities and events [17]. For instance, two temporally distinct statements such as “Barack Obama is the President of the United States (2009–2017)” and “Joe Biden is the President of the United States (2021–)” may appear semantically related, yet they cannot simultaneously hold true. Plain-text retrieval often fails to distinguish between such temporally exclusive facts, leading to incorrect or contradictory reasoning chains.

Temporal reasoning further amplifies these challenges. Realworld entities and relationships evolve continuously, i.e., people change roles, organizations merge, and events unfold in sequence over time [4, 24]. Temporal-aware questions therefore often involve implicit time constraints, multiple dependent conditions, and mixed temporal granularities such as “Who chaired the committee

![](images/404670319f8ff575cfce1bd8799c24da29ba04aae9ef844d1a3dac9b1f92b1f7.jpg)  
Figure 1: Representative workflow of four LLM reasoning paradigms.

before the 2010 reform?” or “Which country hosted the first summit after 2015?”. These questions require reasoning that integrates both temporal and semantic alignment, which remains difficult for text-based RAG methods that lack structured temporal awareness.

To address this issue, integrating Temporal Knowledge Graphs (TKGs) with LLM reasoning provides a promising direction [4, 13, 17, 20, 35]. TKGs encode factual knowledge as quadruples (subject, relation, object, timestamp), offering explicit temporal grounding and relational structure. By leveraging TKGs, models can reason over evolving entities while maintaining factual and temporal consistency. Temporal Knowledge Graph Question Answering (TKGQA) serves as a representative evaluation for this task, requiring systems to answer natural language questions by retrieving temporally relevant facts from TKGs.

Challenges in existing methods. Most existing TKG-based reasoning frameworks follow a “plan-retrieve-answer” pipeline. In this paradigm, LLMs decompose complex temporal questions into a series of sub-tasks, retrieve related facts from a TKG, and generate an answer based on the retrieved context. While this improves interpretability and modularity, several challenges face.

Challenge 1: Temporal faithfulness in multi-hop reasoning. Most temporal QA pipelines[4, 17, 35], as shown in Figure 1(a), prioritize semantic similarity over chronological accuracy. Expanding from one-hop neighbors and ranking by semantics often retrieves paths that appear contextually relevant but violate time constraints. For example, answering “Which country did X last visit before 2020?” may simply return (X, visit, Y, 2019) because of lexical overlap, despite contradicting the before-last relation. In multi-hop queries such as “Who did X hire after Y resigned during Q3 2017?”, locally valid hops can combine into a globally inconsistent reasoning chain.

Challenge 2: Multi-entity temporal synchronization. When a question contains multiple entities, most systems [3, 35, 43] explore each entity independently and attempt to merge partial evidence later. This disjoint process often produces candidates that never align within a single, time-consistent reasoning path. For instance, as shown in Figure 1(b), independent exploration can yield valid facts for each entity, but fails to synchronize their temporal windows.

Challenge 3: Operator diversity and adaptive retrieval. Temporal questions encompass diverse operators, each requiring a distinct reasoning policy. Previous methods [35, 36] focus on single-round rewriting and can expose explicit timestamps but struggle with combined constraints, as shown in Figure 1(c). It prevents the model from adapting to such heterogeneity, resulting in either undercoverage (missing valid evidence) or over-retrieval (noisy results).

Challenge 4: Lack of reasoning experience management. Most existing pipelines [13, 27, 35, 47] remain memoryless, discarding successful reasoning traces after each run. They often rely on manually crafted or static exemplars, which are costly to construct and insufficiently generalizable across diverse temporal question types, leading models repeatedly re-solve similar sub-questions from scratch, failing to transfer prior knowledge across operators or tasks.

Contribution. In this paper, we introduce MemoTime, a Memory-Augmented Temporal Knowledge Graph framework designed to enhance LLM temporal reasoning, as shown in Figure 1(d). Unlike existing methods that rely on static retrieval or pre-defined templates, MemoTime integrates structured temporal grounding, hierarchical reasoning, dynamic toolkit invocation, and continual memory updating into a unified framework. It enables LLMs to reason faithfully over time-aware facts, adapt retrieval strategies to temporal operators, and progressively improve performance through experience reuse.

To address multi-hop temporal reasoning problem, MemoTime introduces a hierarchical reasoning framework that decomposes complex temporal questions under a unified global plan. All subindicators are evolved from the root indicator of the main question, ensuring that each branch inherits consistent temporal constraints and preserves monotonic timestamp progression. Guided by this global supervision, MemoTime performs controlled branch expansion, retrieving multi-hop reasoning paths remaining faithful to both semantic relevance and chronological order.

To handle multi-entity temporal synchronization, MemoTime ensures that the final reasoning path jointly incorporates all topic entities under a unified temporal framework and synchronized timeline. Instead of exploring entities independently, the system retrieves co-constrained evidence paths that preserve shared temporal bounds and consistent granularity, ensuring global coherence and containing the factual basis for the correct answer.

To adapt to diverse temporal operators, MemoTime introduces a library of temporal reasoning toolkits that support heterogeneous operators. Instead of fixed templates, MemoTime adaptively selects the most suitable toolkit through experience-guided prompts. This adaptive retrieval policy ensures that each temporal operator triggers an appropriate reasoning strategy, thereby improving precision and reducing over-retrieval noise.

To manage and reuse reasoning experience, MemoTime maintains a continuously evolving experience memory that records successful records. Each experience entry is stored alongside embeddings of both the question and its temporal indicator, enabling efficient similarity-based retrieval under operator and type constraints. During inference, MemoTime retrieves relevant exemplars to guide new reasoning, while after execution, verified trajectories are written back. The advantage of MemoTime can be abbreviated as follows:

• Memory-augmented temporal reasoning. MemoTime introduces a unified framework that integrates dynamic memory retrieval and update into temporal-aware question decomposition, allowing the model to recall, reuse, and refine reasoning trajectories for long-term temporal understanding.   
• Hierarchical and interpretable control. A hierarchical controller Tree of Time executes, verifies, and refines sub-questions, ensuring faithful and interpretable temporal reasoning.   
• Hybrid retrieval and pruning. An operator-aware retrieval layer combines symbolic graph expansion with embedding search, applying temporal-first pruning and semantic re-ranking for both chronological validity and precise evidence grounding.   
• Self-evolving experience memory. Verified reasoning responses are continuously organized in an adaptive experience pool, forming a closed feedback loop for continual improvement.   
• Efficiency and adaptability: a) MemoTime is a plug-and-play framework that can be seamlessly applied to various LLMs and TKGs. b) MemoTime is auto-refresh. New information is incorporated instantly via TKG retrieval instead of costly LLM finetuning. c) MemoTime achieves state-of-the-art results on all the tested datasets, surpasses the strong baseline by up to $2 4 . 0 \%$ , and enables smaller models (e.g., Qwen3-4B) to achieve reasoning performance comparable to GPT-4-Turbo.

# 2 Related Work

LLM-based knowledge graphs reasoning. Graphs are a natural representation for modeling relational structure among entities [22, 23, 46, 51–55, 63, 64]. Knowledge graphs (KGs) provide structured, verifiable knowledge that complements the implicit world knowledge in LLMs [12, 32, 57–59]. Early works embedded KG facts into neural networks during pre-training or finetuning [26, 39, 40, 65, 68], but such approaches hinder efficient updates and reduce interpretability. Recent studies explore LLM–KG integration by prompting LLMs to iteratively traverse graphs, as seen in [19, 43, 49, 62]. These systems guide the LLM to expand reasoning paths from a seed entity and refine answers through repeated retrieval–generation cycles. However, starting from a single vertex overlooks multi-entity connections and temporal dependencies, often yielding semantically plausible yet chronologically inconsistent paths. [47] alleviates this by modeling multi-hop reasoning paths, but relies on static KGs and lacks the dynamic temporal alignment.

Temporal knowledge graphs question answering. Temporal Knowledge graphs extend static graphs with timestamped facts, enabling reasoning over evolving entities and relations[60, 66, 67]. Earlier approaches fall into two main categories: semantic parsingbased and embedding-based methods. Parsing-based methods translate natural-language questions into logical forms executable on TKGs [7, 16, 31]. These achieve precise execution but fail on long

or ambiguous queries due to brittle symbolic parsing. Embeddingbased methods [28, 28, 36] learn vectorized temporal reasoning by aligning question embeddings with fact embeddings, but they are limited to short reasoning chains and simple time expressions. Recent LLM-based systems [11, 17, 35] improve interpretability by generating reasoning steps in natural language.

Hierarchical and memory-augmented reasoning. Recent advances in LLM-based QA frameworks have emphasized question decomposition as a way to improve reasoning depth and interpretability. Multi-hop reasoning systems such as [61] decompose complex queries into simpler sub-questions that are independently solved and aggregated. This decomposition enables finer-grained retrieval and targeted reasoning across heterogeneous evidence sources [10]. However, these methods decompose linearly, leading to accumulated errors. Moreover, most existing decomposition modules are either manually designed [30] or fine-tuned [56] for specific datasets, limiting their adaptability across temporal and structural question types. In addition to the decomposition, recent studies explore long-term memory systems for LLM agents [29]. Approaches such as [70] enable retrieval of prior interaction histories, while SCM [50] maintains selective access through controller mechanisms. But these frameworks are typically task-agnostic and lack structured representations of reasoning processes.

# 3 Preliminaries

Consider a Temporal Knowledge Graph (TKG) $\mathcal { G } ~ = ~ ( \mathcal { E } , \mathcal { R } , \mathcal { T } )$ , where $\varepsilon , \mathcal { R }$ , and $\mathcal { T }$ represent the set of entities, relations, and timestamps, respectively. $\mathcal { G } = ( \mathcal { E } , \mathcal { R } , \mathcal { T } )$ contains abundant temporal factual knowledge in the form of quadruples, i.e., $\mathcal { G } = \{ ( e _ { h } , r , e _ { t } , t ) \ |$ $e _ { h } , e _ { t } \in \mathcal { E } , r \in \mathcal { R } , t \in \mathcal { T } \}$ . Temporal constraint defines a condition related to a specific time point or interval that must be satisfied by both the answer and its supporting evidence. Following Allen’s interval algebra [1], this includes the 13 classical temporal relations, as well as extended operators such as temporal set relations, duration comparisons, and sorting mechanisms (e.g., first, last, nth) [44]. Temporal constraints thus enable fine-grained alignment between natural language queries and temporally consistent evidence paths.

Definition 1 (Temporal Path). Given a TKG $\mathcal { G }$ , a temporal path is a connected sequence of temporal facts, represented as: $\begin{array} { r c l } { { p a t h _ { \mathcal { G } } ( e _ { 1 } , e _ { l + 1 } ) } } & { { = } } & { { \{ ( e _ { 1 } , r _ { 1 } , e _ { 2 } , t _ { 1 } ) , ( e _ { 2 } , r _ { 2 } , e _ { 3 } , t _ { 2 } ) , \dots , ( e _ { l } , r _ { l } , e _ { l + 1 } , t _ { l } ) \} , } } \end{array}$ where ?? denotes the length of the path, i.e., length $( \mathrm { p a t h } _ { \mathcal { G } } ( e _ { 1 } , e _ { l + 1 } ) ) =$ ?? . The path must satisfy two constraints: (1) connectivity: the tail entity of each fact is identical to the head entity of the next; (2) temporal monotonicity: timestamps are non-decreasing, i.e., $t _ { 1 } \leq t _ { 2 } \leq \dots \leq t _ { l }$ .

Example 1 (Temporal Path). Consider a temporal path between “Merkel” and “EU” with a length of $3 : p a t h _ { \mathcal { G } } ( M e r k e l , E U ) =$ {(Merkel, visit, Paris, 2012), (Paris, host, Conference, 2013), (Conference, attended_by, EU, 2014)}, and can be visualized as:

$$
\text {M e r k e l} \xrightarrow [ 2 0 1 2 ]{\text {v i s i t}} \text {P a r i s} \xrightarrow [ 2 0 1 3 ]{\text {h o s t}} \text {C o n f e r e n c e} \xrightarrow [ 2 0 1 4 ]{\text {a t t e n d e d} _ {\text {b y}}} E U.
$$

Definition 2 (Temporal Reasoning Path). Given a $T K G \mathcal { G }$ , and an entity list $\left[ e _ { 1 } , e _ { 2 } , \ldots , e _ { n } \right]$ , a temporal reasoning path is a sequence of temporal segments $T R P _ { \mathcal { G } } ( [ e _ { 1 } , \dots , e _ { n } ] ) = \{ P _ { 1 } , P _ { 2 } , \dots , P _ { n } \} ,$ where each $P _ { i } = p a t h _ { \mathcal { G } } ( e _ { i } , n _ { i } )$ for some $n _ { i } \in \mathcal { E }$ , and the global monotonicity condition holds: $t ^ { e n d } ( P _ { i } ) \leq t ^ { s t a r t } ( P _ { i + 1 } )$ , $\forall \ : 1 \leq i < n$ .

![](images/1503aabe75b572182d593b070e682310349cd645df9f4411e5423294f9e6f28d.jpg)  
Figure 2: Overview of the MemoTime framework. Temporal Grounding: Topic entities and temporal operators are extracted from the input question to construct a question-specific subgraph. Tree of Time (Hierarchical Reasoning): Recursively decomposes the question into sub-questions guided by temporal dependencies, adaptively reusing experience or invoking toolkits for new evidence. Temporal Evidence Retrieval and Pruning: Performs operator-aware retrieval under monotonic time constraints, followed by semantic-temporal re-ranking and LLM-based sufficiency verification. Experience Memory: Verified reasoning traces are stored, updated, and retrieved for cross-type reuse, enabling continual self-improvement across reasoning cycles.

Example 2 (Temporal Reasoning Path). Consider a temporal reasoning path $T R P _ { \mathcal { G } }$ ( [??????????, ?????? ????????, ??????????]), by two temporally aligned segments: $P _ { 1 } =$ {(Obama, meet, UN, 2009)}, $P _ { 2 } =$ {(Beijing, linked_via, EU, 2011), (EU, event_in, Paris, 2012)}, visualized as:

$$
O b a m a \xrightarrow [ 2 0 0 9 ]{m e e t} U N \sim \text {B e i j i n g} \xrightarrow [ 2 0 1 1 ]{\text {l i n k e d} _ {\text {v i a}}} E U \xrightarrow [ 2 0 1 2 ]{\text {e v e n t} _ {\text {i n}}} P a r i s.
$$

Temporal Knowledge Graph Question Answering (TKGQA) is a fundamental reasoning task based on TKGs. Given a natural language question $Q$ and a TKG $\mathcal { G }$ , the objective is to devise a function $f$ that identify answer entities or timestamps ?? ∈ ???????????? (??) utilizing knowledge in $\mathcal { G }$ , i.e., $a = f ( q , { \mathcal { G } } )$ . Consistent with previous research [26, 27, 42, 43], we assume topic entities ???????????? are mentioned in $Q$ and answer entities or timestamps ???????????? (??) in ground truth are linked to $\mathcal { G }$ , i.e., ?? ?????????? $\subseteq \mathcal { E }$ and ???????????? $( Q ) \subseteq \{ \mathcal { E } \cup \mathcal { T } \}$ .

# 4 Method

Overview. MemoTime implements “TKG-grounded LLM reasoning” by grounding each question in temporal facts, recursively decomposing it into executable sub-queries, retrieving and pruning temporally valid evidence, and continually refining its experience memory. Unlike prior TKGQA or RAG systems that depend on static retrievers or fixed prompts, MemoTime integrates temporal alignment, hierarchical control, and self-evolving memory into a unified reasoning framework. The overall architecture of MemoTime is detailed in Figure 2, consisting of four key components:

• Temporal Grounding. The model begins by linking topic entities and constructing a $D _ { \mathrm { m a x } }$ -hop temporal subgraph $\mathit { G } _ { \mathit { Q } }$ from the TKG. It then classifies the temporal type of the question using exemplars retrieved from memory, producing structured temporal constraints for downstream reasoning (Section 4.1).

• Tree of Time Reasoning. Given the grounded question and its temporal type, MemoTime constructs a hierarchical decomposition tree and executes sub-questions in a top-down manner. For each node, it dynamically decides whether to recall prior experiences, invoke temporal toolkits, or refine unresolved branches, ensuring temporal consistency and interpretability (Section 4.2).   
• Temporal Evidence Retrieval and Pruning. Guided by toolkit configurations, MemoTime performs hybrid retrieval, combining time-monotone, graph exploration, and embedding-based search. Retrieved candidates are filtered by temporal constraints, reranked by semantic and temporal proximity, and finally verified through an LLM-aware selection step (Section 4.3).   
• Experience Memory. Verified reasoning traces, toolkit selections, and sub-question embeddings are stored in a dynamic memory pool. At inference, similar experiences are retrieved by type and relevance; after reasoning, new traces are written back to continuously refine retrieval and decision-making across future questions (Section 4.4).

# 4.1 Temporal Grounding

The temporal grounding stage transforms a natural-language temporal question into a structured reasoning representation, establishing a factual and temporal foundation for subsequent retrieval and inference. It consists of two core phases: knowledge fact grounding and question analysis. The pseudo-code of the temporal grounding is detailed in Algorithm 1 of Appendix A.

Knowledge fact grounding. Given an input question ??, MemoTime first identifies the relevant temporal subgraph $\mathcal { G } _ { Q }$ within the underlying TKG. This subgraph captures entities, relations, and time-stamped triples that are semantically or temporally associated

with $Q$ within a bounded $D _ { \mathrm { m a x } }$ -hop neighborhood, serving as the factual evidence base for downstream reasoning.

Topic entity recognition. To locate question-relevant entities, MemoTime employs LLMs to extract potential entity mentions and associated timestamps from ??. Following extraction, a Dense Retrieval Model (DRM) aligns these mentions with KG entities via embeddingbased similarity matching. Specifically, both question keywords and KG entities are encoded into dense embeddings, and an entity index is constructed using FAISS [9]. Cosine similarity is then computed between the two embedding spaces, and the top-ranked entities are selected to form the topic entity set ????????????. These topic entities act as the initial anchors for constructing a localized subgraph.

Subgraph construction. Once the topic entities are identified, MemoTime constructs a $\overline { { D _ { \mathrm { m a x } } } }$ -hop subgraph $\mathcal { G } _ { Q }$ that aggregates all triples connected to each topic entity within the defined temporal window. This subgraph provides a compact yet semantically rich neighborhood for temporal reasoning. To reduce redundancy and computational overhead, we apply graph reduction and relation clustering techniques following [47], ensuring $\mathit { G _ { Q } }$ remains concise while preserving the most relevant temporal relations.

Question analysis. After constructing $\mathcal { G } _ { Q }$ , the next step is to analyze the temporal intent of the question. Unlike conventional TKGQA approaches that treat all questions uniformly, or use the label from the dataset, leveraging the flexibility reduces. Temporal questions exhibit diverse syntactic forms and intricate time constraints. To capture these nuances, MemoTime first performs temporal type classification, identifying the temporal operator that governs the question’s reasoning logic.

Temporal type classification. Temporal type classification determines the intrinsic temporal relations implied by a question. Following Allen’s interval algebra [1], we consider thirteen fundamental temporal relations, as well as extended operators such as set relations, duration comparisons, and ordering mechanisms [44], as introduced in Section 3. Existing methods often rely on static prompts or manually designed exemplars, limiting their adaptability to complex or unseen temporal structures. In contrast, MemoTime leverages a continuously updated experience pool $\mathcal { E } _ { p o o l }$ that stores successful reasoning trajectories, toolkit configurations, and classification results. This memory-based design enables dynamic retrieval of contextually aligned exemplars, improving temporal reasoning consistency across question types. Formally, for a given question ??, type-specific exemplars are retrieved as: $\mathcal { E } _ { \mathrm { t y p e } } = \mathrm { G e t T y p e E x p } ( \mathcal { E } _ { p o o l } , Q , W _ { \mathrm { e x p } } ) ,$ , where $W _ { \mathrm { e x p } }$ denotes the exemplar retrieval limit. These exemplars are then incorporated into the LLM classification prompt: $\mathsf { T y p e } ( Q ) = \mathsf { P r o m p t } _ { \mathrm { T y p e S e l e c t } } ( Q , \mathcal { E } _ { \mathrm { t y p e } } )$ , allowing the model to identify the most appropriate temporal operator for $Q$ . This exemplar-guided strategy enables the MemoTime to adapt to new question patterns, enhances type prediction accuracy, and lays the foundation for the subsequent decomposition stage.

# 4.2 Tree of Time — Hierarchical Temporal Reasoning

After temporal grounding, MemoTime enters the Tree of Time (ToT) stage, which transforms grounded temporal structures into a hierarchical reasoning tree. Serving as the framework’s central

controller, it coordinates decomposition, toolkit execution, sufficiency testing, and answer synthesis. Guided by the global plan, the model adaptively alternates between recalling experience, invoking operator-specific toolkits, and refining sub-questions to maintain temporal and semantic consistency. Due to the space limitation, the pseudo-code of the ToT is detailed in Algorithm 2 of Appendix A.

Question tree construction. Given a temporal question $Q$ and its classified temporal type Type $( Q )$ , MemoTime begins by constructing a hierarchical decomposition tree $\mathcal { T } _ { Q }$ . Each node represents a sub-question with a specific temporal relation or constraint. To enhance robustness, decomposition exemplars are retrieved from the experience pool $\mathcal { E } _ { p o o l }$ and integrated into the LLM prompt:

$$
\mathcal {E} _ {\text {d e c o m p}} = \operatorname {G e t D e c o m p E x p} \left(\mathcal {E} _ {\text {p o o l}}, Q, W _ {\text {e x p}}, \text {T y p e} (Q)\right),
$$

$$
\mathcal {T} _ {Q} = \operatorname {P r o m p t} _ {\text {D e c o m p o s e}} (Q, \mathcal {E} _ {\text {d e c o m p}}, \text {T y p e} (Q)).
$$

This experience-guided process enables the system to build a typealigned reasoning tree that reflects temporal dependencies among sub-questions. For any parent–child pair $( q _ { i } , q _ { j } )$ , the temporal order satisfies $t ( q _ { i } ) \leq t ( q _ { j } )$ , ensuring temporal monotonicity.

Indicator extraction. From $\mathcal { T } _ { Q }$ , the system extracts a set of subquestion indicators $\left\{ \mathrm { I } _ { i } \right\}$ , i.e., $\mathrm { I } _ { i } = \langle x ? , R , y ? , C _ { t i m e } \rangle$ , each encoding the entities, relations, and temporal constraints required for reasoning. Based on this, a predicted depth $D _ { \mathrm { p r e d } } ( q _ { i } )$ is calculated, defined as the maximum distance between the predicted answer and each topic entity. For instance, in Figure 1(d), the predicted depth of the overall indicator is 2.

Hierarchical execution control. The reasoning tree is traversed in a top-down manner. At each node, MemoTime first checks whether a similar reasoning trace exists in the experience pool for direct reuse. If no match or evidence insufficient, the system proceeds to dynamic toolkit selection to discover new evidence.

Experience-guided toolkit selection. When memory lookup fails, MemoTime retrieves prior examples of successful toolkit usage to enhance selection accuracy:

$$
\mathcal {E} _ {\text {t o o l}} = \operatorname {G e t T o o l k i t E x p} \left(\mathcal {E} _ {\text {p o o l}}, q _ {i}, I _ {i}, \text {T y p e} (Q), W _ {\exp}\right),
$$

and constructs a context-enriched prompt: T?? = PromptToolkitSelect (T?? , I??, ????, Etool). The LLM may recommend multiple toolkits simultaneously, each representing a distinct retrieval or reasoning strategy (e.g., event ordering, interval comparison, or timeline construction). All selected tools are executed in parallel, and their candidate results are passed to the sufficiency evaluation stage.

Question answering. After each sub-question $q _ { i }$ obtains its candidate temporal results, MemoTime evaluates whether these results sufficiently answer the local query and contribute to the overall question. This stage integrates evidence evaluation, global reasoning synthesis, and hierarchical update for unresolved nodes.

Sub-question evaluation. For each node in the reasoning tree, the LLM assesses retrieved temporal paths for both semantic relevance and temporal validity. A sub-question is considered sufficient when one or more candidate paths provide consistent evidence aligned with its indicator. If multiple valid results exist, a lightweight debate–vote strategy aggregates them into a single coherent answer. The solved node and its reasoning trace are recorded in the experience pool $\mathcal { E } _ { p o o l }$ for future reuse.

Global sufficiency and answer synthesis. After all sub-questions are processed, MemoTime checks the temporal and logical coherence of the entire reasoning tree. When global sufficiency is achieved, verified sub-paths are merged into a unified temporal reasoning chain that summarizes the full evidence flow. This concise reasoning trace is then used to prompt the final LLM for answer generation, ensuring interpretability and temporal consistency.

Tree update and termination. If a sub-question fails sufficiency verification, MemoTime adaptively decides whether to retry retrieval, refine the query, or perform further decomposition based on its current reasoning. Newly generated nodes are appended to the tree and processed recursively until all branches are solved or the maximum depth $D _ { \mathrm { m a x } }$ is reached. Finally, verified reasoning paths are consolidated into the final evidence chain for answer generation.

# 4.3 Temporal Evidence Retrieval and Pruning

As discussed in Section 1, identifying reasoning paths that connect all topic entities is crucial for deriving accurate answers. These paths function as interpretable chains of thought. In this section, we operationalize the selected toolkits to perform hybrid retrieval and pruning, balancing efficiency and accuracy while ensuring both high temporal alignment and broad semantic coverage. The pseudo-code of this section is detailed in Algorithm 3 (AppendixA). Retrieval initialization. Before path exploration begins, MemoTime performs seed selection to identify relevant entities that serve as starting points for reasoning. If prior successful seeds exist in $\mathcal { E } _ { p o o l }$ , they are reused directly; otherwise, new seeds are chosen via the experience-augmented prompt: $\begin{array} { r l } { S _ { i } } & { { } = } \end{array}$ $\mathsf { P r o m p t } _ { \mathrm { S e e d S e l e c t } } ( T o p i c s , \mathrm { I } _ { i } , q _ { i } , \mathcal { E } _ { \mathrm { s e e d } } )$ , where $\mathcal { E } _ { \mathrm { s e e d } }$ denotes the retrieved relevant examples from memory.

Toolkit-driven temporal path retrieval. Each toolkit $T _ { \theta } \in { \mathcal { T } } _ { i }$ represents a specialized retrieval operator for a temporal reasoning pattern (e.g., one-hop expansion, interval comparison, event ordering, or range detection). Guided by the indicator $\mathrm { I } _ { i }$ , each toolkit constrains both the semantic relation and temporal boundary of the retrieval. Multiple toolkits are executed in parallel to form a diverse set of candidate paths $\mathcal { P } _ { i }$ .

Hybrid temporal retrieval. For graph-based retrieval, the system performs time-monotonic path expansion. Instead of starting from the maximum depth $D _ { \mathrm { m a x } }$ , MemoTime begins exploration at the predicted depth $D _ { \mathrm { p r e d } } ( \mathrm { I } _ { i } )$ . Given the subgraph $\mathit { G _ { Q } }$ , the ordered seed set $S _ { i }$ , the indicator $\mathrm { I _ { i } }$ , and the depth $D = \operatorname* { m i n } ( D _ { \mathrm { p r e d } } ( \mathrm { I } _ { i } ) , D _ { \mathrm { m a x } } )$ , we identify candidate temporal paths that include all seeds in order. To avoid exhaustive search, we apply a tree-structured bidirectional breadth-first search (BiBFS) from each entity to extract all potential paths, defined as: $C = \{ p \mid | S _ { i } | \cdot ( D { - } 1 ) < \mathrm { l e n g t h } ( p ) \leq | S _ { i } | \cdot D \}$ . In parallel, an embedding retriever retrieves semantically relevant facts from document-indexed KG segments, ensuring that implicit or cross-hop relations are also captured. The two result streams are concatenated and unified into a hybrid candidate path pool.

Temporal and semantic pruning. Candidate paths are then filtered through a multi-stage pruning pipeline by temporal consistency, semantics, and LLM selection. Together, these steps drastically reduce noise while retaining only candidates that satisfy both chronological and semantic alignment.

Temporal-first pruning. Traditional relevance-first pruning may retain semantically similar but temporally inconsistent paths. To mitigate this, MemoTime adopts a temporal-first pruning policy. Given the candidate pool $C = \{ p _ { i } \} _ { i = 1 } ^ { N }$ returned by the hybrid retriever, MemoTime first applies a strict temporal filter that removes paths violating time constriction $C _ { t i m e }$ or non-monotonic progressions, yielding a valid subset ${ \widetilde { C } } \subseteq C$ that satisfies all temporal constraints. Semantic–temporal re-ranking. From $\widetilde { C }$ , candidates are re-ranked by jointly considering (i) semantic compatibility with the indicator and (ii) proximity of their timestamps to the expected reference time. Let $\mathsf { D R M } ( \mathrm { I } _ { i } , \boldsymbol { p } )$ denote the semantic similarity between the indicator and path, and $t ( \cdot )$ denote the representative timestamp. Each candidate receives a composite score:

$$
\operatorname {S c o r e} (p) = \lambda_ {\text {s e m}} \cdot \operatorname {D R M} \left(\mathrm {I} _ {i}, p\right) + \lambda_ {\text {p r o x}} \cdot \exp \left(- | t (p) - t \left(\mathrm {I} _ {i}\right) | / \sigma\right),
$$

where $\lambda _ { \mathrm { s e m } }$ and $\lambda _ { \mathrm { p r o x } }$ balance semantic alignment and temporal proximity. The top- $W _ { 1 }$ candidates are retained as a compact, highquality pool for LLM-based selection.

LLM-aware selection. Following the re-ranking procedure, the candidate paths are reduced to $W _ { 1 }$ . We then prompt LLM to score and select the top- $W _ { \mathrm { m a x } }$ reasoning paths most likely to satisfy question’s temporal and semantic constraints. This final step emphasizes faithfulness and interpretability, producing a minimal but high-precision evidence set for the sufficiency verification in the reasoning loop.

# 4.4 Experience Memory

After each reasoning episode, verified temporal paths and toolkit decisions are stored for future reuse, forming the foundation of the experience memory layer. This layer serves as the long-term knowledge base of MemoTime, recording reasoning trajectories, toolkit selection patterns, and verified temporal facts. It bridges different reasoning cycles by allowing the model to retrieve relevant exemplars and continually refine its decision-making through accumulated experience. Unlike static prompt libraries, the memory operates as a dynamic, self-evolving component supporting both retrieval and continual adaptation across diverse question types. The pseudo-code of the section is shown in Algorithm 4 of Appendix A.

Experience retrieval. During the Tree of Time reasoning process, each sub-task may query the experience pool $\mathcal { E } _ { p o o l }$ to obtain relevant exemplars for decomposition, seed selection, or toolkit usage. All retrieval operations (e.g., GetTypeExp, GetDecompExp, GetToolkitExp) are implemented under a unified interface: $\varepsilon =$ RetrieveExperience $( \mathcal { E } _ { p o o l } , q _ { i } , \mathrm { I } _ { i } , \tau , W _ { \mathrm { e x p } } )$ , where $\tau = \mathrm { T y p e } ( Q )$ denotes the temporal type of the current reasoning context. Each query is restricted to the memory subset that shares the same type label $\tau$ , ensuring contextual consistency and preventing cross-type noise. Every experience record stores both textual metadata and its dense embeddings, one for the question text and another for its indicator, allowing semantic–temporal similarity search through a FAISS-based index [9]. The incorrect samples are used as warnings.

To accelerate lookup, a high-frequency buffer caches recently accessed exemplars. Entries in this buffer are ranked by a hybrid metric combining embedding similarity and hit frequency: Sc $\begin{array} { r } { \mathsf { o r e } ( E _ { j } ) = \lambda _ { \mathrm { s i m } } \cos ( e _ { q _ { i } } , e _ { E _ { j } } ) + \lambda _ { \mathrm { h i t } } \mathsf { C o u n t } ( E _ { j } ) } \end{array}$ , so that frequently used exemplars gain higher reuse priority. This dual-layer design

Table 1: Performance comparison on MultiTQ (Hits $( \overline { { \omega } } 1 , \% )$ . Best in bold, second-best underlined.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">LLM</td><td rowspan="2">Overall</td><td colspan="2">Question Type</td><td colspan="2">Answer Type</td></tr><tr><td>Multiple</td><td>Single</td><td>Entity</td><td>Time</td></tr><tr><td>BERT[8]</td><td rowspan="2">-</td><td>8.3</td><td>6.1</td><td>9.2</td><td>10.1</td><td>4.0</td></tr><tr><td>ALBERT[21]</td><td>10.8</td><td>8.6</td><td>11.6</td><td>13.9</td><td>3.2</td></tr><tr><td>EmbedKGQA[37]</td><td rowspan="3">-</td><td>20.6</td><td>13.4</td><td>23.5</td><td>29.0</td><td>0.1</td></tr><tr><td>CronKGQA[36]</td><td>27.9</td><td>13.4</td><td>33.7</td><td>32.8</td><td>15.6</td></tr><tr><td>MultiQA[5]</td><td>29.3</td><td>15.9</td><td>34.7</td><td>34.9</td><td>15.7</td></tr><tr><td>ChatGPT</td><td rowspan="5">GPT-3.5-Turbo</td><td>10.2</td><td>7.7</td><td>14.7</td><td>13.7</td><td>2.0</td></tr><tr><td>KG-RAG[4]</td><td>18.5</td><td>16.0</td><td>20.0</td><td>23.0</td><td>7.0</td></tr><tr><td>ReAct KB[4]</td><td>21.0</td><td>13.6</td><td>63.5</td><td>31.3</td><td>30.0</td></tr><tr><td>ARI[4]</td><td>38.0</td><td>68.0</td><td>21.0</td><td>39.4</td><td>34.4</td></tr><tr><td>TempAgent[13]</td><td>53.9</td><td>16.8</td><td>68.4</td><td>47.8</td><td>66.1</td></tr><tr><td rowspan="4">MemoTime</td><td>GPT-4o-mini</td><td>64.2</td><td>40.3</td><td>73.0</td><td>61.5</td><td>70.8</td></tr><tr><td>Qwen3-32B</td><td>68.2</td><td>42.5</td><td>77.6</td><td>62.1</td><td>81.7</td></tr><tr><td>DeepSeek-V3</td><td>73.0</td><td>45.9</td><td>82.9</td><td>67.7</td><td>84.6</td></tr><tr><td>GPT-4-Turbo</td><td>77.9</td><td>53.8</td><td>86.8</td><td>74.5</td><td>85.3</td></tr></table>

allows MemoTime to balance long-term generalization with shortterm adaptability during ongoing reasoning.

Experience update. After a reasoning cycle concludes, all verified sub-questions, selected toolkits, and reasoning paths are recorded back into $\mathcal { E } _ { p o o l }$ . Each record includes textual content, structured indicators, execution parameters, temporal constraints, and corresponding embeddings for both the question and the indicator. This dual representation supports precise semantic alignment and fast vector retrieval. New entries are inserted into both the global memory and the buffer; obsolete or low-value traces are periodically pruned to prevent redundancy. Through continuous recording and pruning, the experience pool evolves alongside the system’s reasoning behaviour, serving as a compact yet expressive representation of accumulated knowledge.

Cross-type augmentation. While retrieval is type-restricted, the memory layer supports cross-type augmentation during updates. When a sub-question from one temporal type exhibits high structural similarity with exemplars in another, the new record is annotated with multiple secondary type labels. This mechanism enables MemoTime to share transferable reasoning patterns across related temporal operators (e.g., BeforeLast and AfterFirst), forming an interconnected experience graph that improves recall and diversity without sacrificing retrieval precision.

Continual adaptation. The memory layer continuously reweights and reorganizes stored exemplars based on retrieval frequency, temporal freshness, and reasoning success. Highconfidence exemplars are prioritized in future retrievals, while outdated or inconsistent examples gradually decay in influence. The embedding index is periodically rebalanced to maintain uniform coverage and retrieval efficiency. This self-adaptive mechanism ensures that MemoTime focuses on temporally relevant reasoning strategies and generalizes effectively to unseen question types.

Integration with reasoning. The experience memory interacts closely with the Tree of Time reasoning loop described in Section 4.2. At runtime, it acts as a shared repository for experienceguided decomposition, toolkit selection, and answer synthesis. After each reasoning session, verified results, including embeddings, type labels, sufficiency status, and hit statistics, are propagated back into memory, updating both the long-term index and the

Table 2: Performance comparison on TimeQuestions $( \mathbf { H i t s } @ 1 , \% )$ . Best in bold, second-best underlined.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">LLM</td><td rowspan="2">Overall</td><td colspan="2">Question Type</td><td colspan="2">Answer Type</td></tr><tr><td>Explicit</td><td>Implicit</td><td>Temporal</td><td>Ordinal</td></tr><tr><td>PullNet[42]</td><td></td><td>10.5</td><td>2.2</td><td>8.1</td><td>23.4</td><td>2.9</td></tr><tr><td>Uniqorn[34]</td><td>-</td><td>33.1</td><td>31.8</td><td>31.6</td><td>39.2</td><td>20.2</td></tr><tr><td>GRAFT-Net[41]</td><td></td><td>45.2</td><td>44.5</td><td>42.8</td><td>51.5</td><td>32.2</td></tr><tr><td>CronKGQA[28]</td><td></td><td>46.2</td><td>46.6</td><td>44.5</td><td>51.1</td><td>36.9</td></tr><tr><td>TempoQR[28]</td><td></td><td>41.6</td><td>46.5</td><td>3.6</td><td>40.0</td><td>34.9</td></tr><tr><td>EXAQT[18]</td><td>-</td><td>57.2</td><td>56.8</td><td>51.2</td><td>64.2</td><td>42.0</td></tr><tr><td>TwiRGCN[38]</td><td></td><td>60.5</td><td>60.2</td><td>58.6</td><td>64.1</td><td>51.8</td></tr><tr><td>LGQA[25]</td><td></td><td>52.9</td><td>53.2</td><td>50.6</td><td>60.5</td><td>40.2</td></tr><tr><td>GenTKGQA[11]</td><td>Fine-tune</td><td>58.4</td><td>59.6</td><td>61.1</td><td>56.3</td><td>57.8</td></tr><tr><td>TimeR4[35]</td><td>Fine-tune</td><td>64.8</td><td>66.0</td><td>52.9</td><td>77.6</td><td>45.5</td></tr><tr><td>ChatGPT</td><td>GPT-3.5-Turbo</td><td>45.9</td><td>43.3</td><td>51.1</td><td>46.5</td><td>48.1</td></tr><tr><td rowspan="4">MemoTime</td><td>GPT-4o-mini</td><td>60.9</td><td>60.0</td><td>50.0</td><td>69.4</td><td>52.8</td></tr><tr><td>Qwen3-32B</td><td>60.6</td><td>58.3</td><td>51.1</td><td>69.8</td><td>53.3</td></tr><tr><td>DeepSeek-V3</td><td>67.8</td><td>63.1</td><td>53.3</td><td>71.3</td><td>54.3</td></tr><tr><td>GPT-4-Turbo</td><td>71.4</td><td>67.0</td><td>67.5</td><td>74.5</td><td>58.7</td></tr></table>

fast-access buffer. This design closes the reasoning–learning loop, enabling MemoTime to continually refine its temporal reasoning ability through iterative retrieval, execution, and self-improvement.

# 5 Experiments

In this section, we evaluate MemoTime on two challenge TKGQA datasets. The detailed experimental settings, including datasets, baselines, and implementations, can be found in Appendix D.

# 5.1 Main results

We evaluate MemoTime against a comprehensive set of baselines on MultiTQ and TimeQuestions, as shown in Table 1 and Table 2. On the MultiTQ dataset, traditional pre-trained language models (BERT, ALBERT) and embedding-based methods (EmbedKGQA, CronKGQA, MultiQA) show limited temporal reasoning ability, with overall accuracies below $3 0 \%$ . Their static embeddings fail to capture multi-hop or cross-entity temporal dependencies. LLMbased methods, e.g, KG-RAG, ReAct KB, ARI, TempAgent, leverage prompt reasoning but remain sensitive to implicit time expressions and struggle with multi-constraint temporal alignment. In contrast, MemoTime consistently outperforms all competitors across both question and answer types. MemoTime with GPT-4-Turbo achieves an overall $7 7 . 9 \%$ Hit@1, surpassing TempAgent by $2 4 . 0 \%$ and outperforming all GPT-3.5 baselines by a large margin. Even smaller backbones (e.g., Qwen3-32B) reach $6 8 . 2 \%$ , exceeding the best GPT-3.5-based models. These results confirm that MemoTime effectively integrates temporal grounding, hierarchical reasoning, and operator-aware retrieval to ensure temporal faithfulness and reasoning stability. On the TimeQuestions dataset, which emphasizes explicit versus implicit temporal expressions and temporal versus ordinal answer types, MemoTime again achieves the best overall results. As shown in Table 2, MemoTime with GPT-4-Turbo achieves $7 1 . 4 \%$ overall accuracy and $7 4 . 5 \%$ on temporal-type questions, outperforming the fine-tuned GenTKGQA and TimeR4 models, despite being fully training-free. This demonstrates MemoTime’s ability to handle implicit temporal relations and operator combinations through memory-guided decomposition and dynamic temporal retrieval. Overall, across both datasets, MemoTime achieves new stateof-the-art results under all backbone LLMs, validating its design as a memory-augmented temporal reasoning framework capable

Table 3: Performance comparison between the IO baseline and MemoTime across two temporal QA datasets using six backbone LLMs. The highest improvement is highlighted in bold, and the second-best is underlined for each data.   

<table><tr><td rowspan="2">Dataset</td><td colspan="3">Qwen3-4B</td><td colspan="3">Qwen3-8B</td><td colspan="3">Qwen3-32B</td><td colspan="3">Qwen3-80B</td><td colspan="3">DeepSeek-V3</td><td colspan="3">GPT-4-Turbo</td></tr><tr><td>IO</td><td>MemoTime</td><td>↑</td><td>IO</td><td>MemoTime</td><td>↑</td><td>IO</td><td>MemoTime</td><td>↑</td><td>IO</td><td>MemoTime</td><td>↑</td><td>IO</td><td>MemoTime</td><td>↑</td><td>IO</td><td>MemoTime</td><td>↑</td></tr><tr><td>MultiTQ</td><td>3.5</td><td>55.3</td><td>14.8 ×</td><td>2.5</td><td>57.0</td><td>21.8 ×</td><td>1.3</td><td>61.4</td><td>46.2 ×</td><td>9.5</td><td>67.5</td><td>6.1 ×</td><td>7.5</td><td>70.9</td><td>8.5 ×</td><td>12.0</td><td>76.3</td><td>5.4 ×</td></tr><tr><td>TimeQuestion</td><td>18.0</td><td>45.3</td><td>1.5 ×</td><td>28.5</td><td>49.4</td><td>0.7 ×</td><td>30.0</td><td>60.6</td><td>1.0 ×</td><td>41.0</td><td>62.7</td><td>0.5 ×</td><td>43.5</td><td>64.6</td><td>0.5 ×</td><td>44.0</td><td>68.1</td><td>0.5 ×</td></tr></table>

Table 4: Ablation study on MultiTQ (Hits@1, %).   

<table><tr><td rowspan="2">Model Variant</td><td rowspan="2">Overall</td><td colspan="2">Question Type</td><td colspan="2">Answer Type</td></tr><tr><td>Multiple</td><td>Single</td><td>Entity</td><td>Time</td></tr><tr><td>MemoTime (Full w/GPT-4o-mini)</td><td>64.2</td><td>40.3</td><td>73.0</td><td>61.5</td><td>70.8</td></tr><tr><td>w/o Graph-based Retrieval</td><td>52.9</td><td>23.5</td><td>65.0</td><td>48.2</td><td>64.2</td></tr><tr><td>w/o Embedding Retrieval</td><td>60.1</td><td>38.6</td><td>68.0</td><td>57.4</td><td>66.8</td></tr><tr><td>w/o Temporal Evidence Retrieval</td><td>11.2</td><td>8.0</td><td>12.4</td><td>11.5</td><td>12.0</td></tr><tr><td>w/o Question Tree</td><td>58.3</td><td>19.3</td><td>72.6</td><td>54.6</td><td>70.1</td></tr><tr><td>w/o Experience Memory</td><td>59.8</td><td>26.1</td><td>72.2</td><td>55.8</td><td>69.6</td></tr></table>

of learning from experience, synchronizing multiple entities, and maintaining temporal consistency across diverse question types.

# 5.2 Ablation Study

How does the performance of MemoTime vary across different LLM backbones? To evaluate the generality of our approach, we tested MemoTime on six LLM backbones, including Qwen3-4B, Qwen3-8B, Qwen3-32B, Qwen3-80B, DeepSeek-V3, and GPT-4-Turbo, across two temporal QA benchmarks (MultiTQ and TimeQuestions). As shown in Table 3, MemoTime consistently improves performance over the IO baseline across all settings. On the more complex MultiTQ, which requires multi-hop and crossentity temporal reasoning, MemoTime boosts accuracy from as low as $1 . 3 \%$ to $6 1 . 4 \%$ on Qwen3-32B, achieving a 46.2 times relative gain. Even the smallest backbones (e.g., Qwen3-4B) experience a 14.8 times improvement, demonstrating that our recursive, memory-augmented reasoning compensates for weaker temporal understanding in smaller models. For TimeQuestions, which involves simpler, single-hop temporal relations, MemoTime still improves performance modestly by up to 1.5 times, indicating consistent stability across reasoning difficulty levels. Overall, MemoTime enables smaller models to perform competitively with large-scale LLMs such as GPT-4-Turbo, while stronger models continue to benefit from enhanced temporal faithfulness and reduced reasoning variance. These results confirm that MemoTime generalizes well across architectures and scales, serving as a plug-in temporal reasoning module rather than relying on model-specific fine-tuning.

How does temporal evidence retrieval affect performance? To assess the contribution of the temporal evidence retrieval module, we disable different retrieval components. As shown in Table 4, removing graph-based retrieval reduces the overall accuracy from $6 4 . 2 \%$ to $5 2 . 9 \%$ , with a substantial drop on multiple-entity questions (from $4 0 . 3 \%$ to $2 3 . 5 \%$ ), indicating that explicit structural traversal is essential for maintaining temporally aligned entity paths. When embedding-based retrieval is removed, the performance decreases moderately from $6 4 . 2 \%$ to $6 0 . 1 \%$ , suggesting that semantic retrieval complements the graph search by capturing lexically diverse or contextually implicit temporal cues. In contrast, when all external retrieval modules are disabled and the model relies solely on its internal knowledge, the accuracy falls drastically to $1 1 . 2 \%$ , reflecting

a near-complete loss of temporal grounding. These findings confirm that temporal evidence retrieval is not only indispensable but also synergistic: graph-based retrieval ensures factual precision through topological consistency, while embedding-based retrieval enhances coverage through semantic generalization.

How does hierarchical decomposition affect performance? To examine the role of hierarchical decomposition, we disable the Tree-of-Time module, forcing the model to reason over the full question without recursive planning. As shown in Table 4, the overall accuracy increases from $6 4 . 2 \%$ to $5 8 . 3 \%$ , while performance on multiple-entity questions drops sharply from $4 0 . 3 \%$ to $1 9 . 3 \%$ . This substantial decline demonstrates that recursive decomposition is critical for handling complex temporal dependencies and multi-hop constraints. Even with retrieval retained, the absence of decomposition limits the model’s ability to enforce monotonic timestamp ordering and to synchronize temporal operators across entities, leading to degraded reasoning coherence.

How does experience memory affect performance? To analyse the role of continual learning, we remove the experience memory module, preventing MemoTime from accessing previously verified reasoning traces. The result in Table 4 shows it causes a moderate performance drop from $6 4 . 2 \%$ to $5 9 . 8 \%$ , with larger reductions observed on multi-hop questions, from $4 0 . 3 \%$ to $2 6 . 1 \%$ . The results indicate that experience memory enhances reasoning stability and efficiency. Although the model remains competitive without it, the memory mechanism contributes to more consistent reasoning trajectories and progressive self-improvement across inference cycles.

To further evaluate the effectiveness and efficiency of Memo-Time, we conduct additional experiments, including multi-granular temporal QA analysis, efficiency analysis, and case studies of temporal interpretability and faithful reasoning in Appendix B.

# 6 Conclusion

In this paper, we present MemoTime, a memory-augmented temporal knowledge graph framework that enhances LLM temporal reasoning through structured grounding, hierarchical decomposition, and continual experience learning. MemoTime enables temporally faithful, multi-entity reasoning by constructing operator-aware temporal paths and dynamically reusing verified reasoning traces. Its adaptive retrieval and memory integration allow it to balance efficiency, accuracy, and temporal consistency across diverse reasoning scenarios. Extensive experiments on multiple temporal QA datasets demonstrate that MemoTime outperforms existing baselines, showcasing its superior reasoning capabilities and interoperability.

# Acknowledgments

Xiaoyang Wang is supported by the Australian Research Council DP230101445 and DP240101322. Wenjie Zhang is supported by the Australian Research Council DP230101445 and FT210100303.

# References

[1] James F Allen. 1984. Towards a general theory of action and time. Artificial intelligence 23, 2 (1984), 123–154.   
[2] Tom B Brown. 2020. Language models are few-shot learners. arXiv preprint arXiv:2005.14165 (2020).   
[3] Liyi Chen, Panrong Tong, et al. 2024. Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs. In NeurIPS.   
[4] Ziyang Chen, Dongfang Li, Xiang Zhao, et al. 2024. Temporal knowledge question answering via abstract reasoning induction. In ACL.   
[5] Ziyang Chen, Jinzhi Liao, and Xiang Zhao. 2023. Multi-granularity temporal question answering over knowledge graphs. In ACL. 11378–11392.   
[6] Ziyang Chen, Jinzhi Liao, and Xiang Zhao. 2023. Multi-granularity Temporal Question Answering over Knowledge Graphs. In ACL. 11378–11392.   
[7] Zhuo Chen, Zhao Zhang, et al. 2024. Self-improvement programming for temporal knowledge graph question answering. arXiv preprint arXiv:2404.01720 (2024).   
[8] Jacob Devlin, Ming-Wei Chang, et al. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In NAACL.   
[9] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, et al. 2024. The faiss library. arXiv preprint arXiv:2401.08281 (2024).   
[10] Yair Feldman and Ran El-Yaniv. 2019. Multi-hop paragraph retrieval for opendomain question answering. arXiv preprint arXiv:1906.06606 (2019).   
[11] Yifu Gao, Linbo Qiao, et al. 2024. Two-stage generative question answering on temporal knowledge graph using large language models. In ACL.   
[12] Xinyan Guan, Yanjiang Liu, et al. 2024. Mitigating large language model hallucinations via autonomous knowledge graph-based retrofitting. In AAAI.   
[13] Qianyi Hu, Xinhui Tu, et al. 2025. Time-aware ReAct Agent for Temporal Knowledge Graph Question Answering. In Findings of NAACL.   
[14] Chengkai Huang, Hongtao Huang, Tong Yu, et al. 2025. A Survey of Foundation Model-Powered Recommender Systems: From Feature-Based, Generative to Agentic Paradigms. arXiv preprint arXiv:2504.16420 (2025).   
[15] Chengkai Huang, Yu Xia, Rui Wang, et al. 2025. Embedding-informed adaptive retrieval-augmented generation of large language models. In COLING.   
[16] Zhen Jia, Abdalghani Abujabal, et al. 2018. TEQUILA: Temporal Question Answering over Knowledge Bases. In CIKM.   
[17] Zhen Jia, Philipp Christmann, and Gerhard Weikum. 2024. Faithful temporal question answering over heterogeneous sources. In WWW. 2052–2063.   
[18] Zhen Jia, Soumajit Pramanik, Rishiraj Saha Roy, and Gerhard Weikum. 2021. Complex temporal question answering on knowledge graphs. In CIKM.   
[19] Jinhao Jiang et al. 2023. Structgpt: A general framework for large language model to reason over structured data. arXiv preprint arXiv:2305.09645 (2023).   
[20] Tushar Khot et al. 2022. Decomposed prompting: A modular approach for solving complex tasks. arXiv preprint arXiv:2210.02406 (2022).   
[21] Zhenzhong Lan, Mingda Chen, et al. 2019. Albert: A lite bert for self-supervised learning of language representations. arXiv preprint arXiv:1909.11942 (2019).   
[22] Fan Li, Xiaoyang Wang, Dawei Cheng, et al. 2025. Efficient dynamic attributed graph generation. In ICDE.   
[23] Fan Li, Zhiyu Xu, et al. 2024. AdaRisk: risk-adaptive deep reinforcement learning for vulnerable nodes detection. IEEE TKDE (2024).   
[24] Ke Liang, Lingyuan Meng, et al. 2023. Learn from relational correlations and periodic events for temporal knowledge graph reasoning. In SIGIR.   
[25] Yonghao Liu, Di Liang, Mengyu Li, et al. 2023. Local and Global: Temporal Question Answering via Information Fusion.. In IJCAI. 5141–5149.   
[26] Linhao Luo et al. 2023. Reasoning on graphs: Faithful and interpretable large language model reasoning. arXiv preprint arXiv:2310.01061 (2023).   
[27] Shengjie Ma, Chengjin Xu, et al. 2024. Think-on-Graph 2.0: Deep and Interpretable Large Language Model Reasoning with Knowledge Graph-guided Retrieval. arXiv preprint arXiv:2407.10805 (2024).   
[28] Costas Mavromatis, Prasanna Lakkur Subramanyam, et al. 2022. Tempoqr: temporal question reasoning over knowledge graphs. In AAAI.   
[29] Kai Mei, Zelong Li, Shuyuan Xu, Ruosong Ye, Yingqiang Ge, and Yongfeng Zhang. 2024. AIOS: LLM agent operating system. arXiv e-prints, pp. arXiv–2403 (2024).   
[30] Sewon Min, Victor Zhong, et al. 2019. Multi-hop reading comprehension through question decomposition and rescoring. arXiv preprint arXiv:1906.02916 (2019).   
[31] Sumit Neelam and Udit others Sharma. 2022. SYGMA: A System for Generalizable and Modular Question Answering Over Knowledge Bases. In Findings of EMNLP.   
[32] Shirui Pan, Linhao Luo, et al. 2024. Unifying large language models and knowledge graphs: A roadmap. IEEE TKDE (2024).   
[33] Fabio Petroni et al. 2020. KILT: a benchmark for knowledge intensive language tasks. arXiv preprint arXiv:2009.02252 (2020).   
[34] Soumajit Pramanik et al. 2024. Uniqorn: unified question answering over rdf knowledge graphs and natural language text. Journal of Web Semantics (2024).   
[35] Xinying Qian et al. 2024. TimeR4 : Time-aware Retrieval-Augmented Large Language Models for Temporal Knowledge Graph Question Answering. In EMNLP.   
[36] Apoorv Saxena et al. 2021. Question answering over temporal knowledge graphs. In ACL.   
[37] Apoorv Saxena, Aditay Tripathi, et al. 2020. Improving multi-hop question answering over knowledge graphs using knowledge base embeddings. In ACL.

[38] Aditya Sharma et al. 2022. Twirgcn: Temporally weighted graph convolution for question answering over temporal knowledge graphs. arXiv preprint arXiv:2210.06281 (2022).   
[39] Qing Sima, Xiaoyang Wang, et al. 2026. Beyond Homophily: Community Search on Heterophilic Graphs. arXiv preprint arXiv:2601.01703 (2026).   
[40] Qing Sima, Jianke Yu, et al. 2025. Deep overlapping community search via subspace embedding. SIGMOD (2025).   
[41] Haitian Sun et al. 2018. Open domain question answering using early fusion of knowledge bases and text. arXiv preprint arXiv:1809.00782 (2018).   
[42] Haitian Sun et al. 2019. Pullnet: Open domain question answering with iterative retrieval on knowledge bases and text. arXiv preprint arXiv:1904.09537 (2019).   
[43] Jiashuo Sun, Chengjin Xu, et al. 2024. Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph. In ICLR.   
[44] Qiang Sun, Sirui Li, et al. 2025. TimelineKGQA: A Comprehensive Question-Answer Pair Generator for Temporal Knowledge Graphs. In WWW.   
[45] Alon Talmor et al. 2018. Commonsenseqa: A question answering challenge targeting commonsense knowledge. arXiv preprint arXiv:1811.00937 (2018).   
[46] Xingyu Tan, Jingya Qian, Chen Chen, Sima Qing, Yanping Wu, Xiaoyang Wang, and Wenjie Zhang. 2023. Higher-order peak decomposition. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management.   
[47] Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, and Wenjie Zhang. 2025. Paths-over-graph: Knowledge graph empowered large language model reasoning. In Proceedings of the ACM on Web Conference 2025. 3505–3522.   
[48] Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, Liming Zhu, and Wenjie Zhang. 2025. Hydrarag: Structured cross-source enhanced large language model reasoning. In EMNLP. 14442–14470.   
[49] Xingyu Tan, Xiaoyang Wang, Qing Liu, Xiwei Xu, Xin Yuan, Liming Zhu, and Wenjie Zhang. 2026. PrivGemo: Privacy-Preserving Dual-Tower Graph Retrieval for Empowering LLM Reasoning with Memory Augmentation. arXiv preprint arXiv:2601.08739 (2026).   
[50] Bing Wang et al. 2023. Enhancing large language model with self-controlled memory framework. arXiv preprint arXiv:2304.13343 (2023).   
[51] Jinghao Wang, Yanping Wu, Xiaoyang Wang, et al. 2025. Effective Influence Maximization with Priority. In WWW.   
[52] Jinghao Wang, Yanping Wu, Xiaoyang Wang, et al. 2025. Time-Critical Influence Minimization via Node Blocking. SIGMOD (2025).   
[53] Jinghao Wang, Yanping Wu, Xiaoyang Wang, Ying Zhang, et al. 2024. Efficient influence minimization via node blocking. arXiv preprint arXiv:2405.12871 (2024).   
[54] Yiqi Wang, Long Yuan, et al. 2023. Towards efficient shortest path counting on billion-scale graphs. In ICDE.   
[55] Yiqi Wang, Long Yuan, et al. 2024. Simpler is more: Efficient top-k nearest neighbors search on large road networks. arXiv preprint arXiv:2408.05432 (2024).   
[56] Jian Wu et al. 2024. Gendec: A robust generative question-decomposition method for multi-hop reasoning. arXiv preprint arXiv:2402.11166 (2024).   
[57] Yanping Wu, Renjie Sun, Xiaoyang Wang, et al. 2024. Efficient Maximal Frequent Group Enumeration in Temporal Bipartite Graphs. Proc. VLDB Endow. (2024).   
[58] Yuhan Wu, Yuanyuan Xu, Xuemin Lin, and Wenjie Zhang. 2023. A holistic approach for answering logical queries on knowledge graphs. In ICDE.   
[59] Peiting Xie, Xiangjun Zai, Yanping Wu, et al. 2025. HL-index: Fast Reachability Query in Hypergraphs. arXiv preprint arXiv:2512.23345 (2025).   
[60] Zhengyi Yang, Wenjie Zhang, Xuemin Lin, et al. 2023. Hgmatch: A match-byhyperedge approach for subgraph matching on hypergraphs. In ICDE.   
[61] Shunyu Yao et al. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629 (2022).   
[62] Xiangjun Zai, Xingyu Tan, Xiaoyang Wang, Qing Liu, et al. 2025. PRoH: Dynamic Planning and Reasoning over Knowledge Hypergraphs for Retrieval-Augmented Generation. arXiv preprint arXiv:2510.12434 (2025).   
[63] Zian Zhai, Fan Li, Xingyu Tan, Xiaoyang Wang, and Wenjie Zhang. 2025. Graph is a natural regularization: Revisiting vector quantization for graph representation learning. arXiv preprint arXiv:2508.06588 (2025).   
[64] Zian Zhai, Qing Sima, Xiaoyang Wang, and Wenjie Zhang. 2025. SGPT: Few-Shot Prompt Tuning for Signed Graphs. In CIKM.   
[65] Hang Zhang et al. 2021. Poolingformer: Long document modeling with pooling attention. In ICML. PMLR.   
[66] Tianming Zhang, Xinwei Cai, et al. 2024. Towards efficient simulation-based constrained temporal graph pattern matching. World Wide Web (2024).   
[67] Tianming Zhang, Junkai Fang, et al. 2024. Tatkc: A temporal graph neural network for fast approximate temporal Katz centrality ranking. In WWW.   
[68] Liangwei Nathan Zheng, Chang George Dong, et al. 2024. Understanding Why Large Language Models Can Be Ineffective in Time Series Analysis: The Impact of Modality Alignment. arXiv preprint arXiv:2410.12326 (2024).   
[69] Liangwei Nathan Zheng, Wenhao Liang, Wei Emma Zhang, et al. 2025. Lifting Manifolds to Mitigate Pseudo-Alignment in LLM4TS. arXiv preprint arXiv:2510.12847 (2025).   
[70] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. Memorybank: Enhancing large language models with long-term memory. In AAAI.

# A Algorithm

# A.1 Temporal Grounding

We present the comprehensive algorithmic procedure for temporal grounding (Section 4.1) in Algorithm 1.

# A.2 Tree of Time Reasoning

We present the comprehensive algorithmic procedure for Tree of Time hierarchical reasoning (Section 4.2) in Algorithm 2.

# A.3 Temporal Evidence Retrieval and Pruning

We present the comprehensive algorithmic procedure for temporal evidence retrieval and pruning (Section 4.3) in Algorithm 3.

# A.4 Experience Memory

We present the comprehensive algorithmic procedure for experience memory (Section 4.4) in Algorithm 4.

Algorithm 1: TemporalGrounding   
Input:Question Q,TKG G, experience pool $\mathcal{E}_{\mathrm{pool}}$ limits $D_{\mathrm{max}},W_{\mathrm{exp}}$ Output: Subquestion tree $\mathcal{T}_Q$ ,indicators $\{\mathrm{I}_i\}$ ,topic entities Topics, type Type(Q),subgraph $\mathcal{G}_Q$ /\* LLM extraction $^+$ DRM alignment \*/   
1Keywords $\leftarrow$ NER(Q);Cand $\leftarrow$ LinkToKG(Keyboard, $\mathcal{G}$ 1   
2 Topics $\leftarrow$ Disambiguate(Q,Cand); /\* bounded $D_{\mathrm{max}}$ -hop temporal neighborhood \*/   
3 $\mathcal{G}_Q\gets$ BuildTemporalSubgraph(Topics,G,Dmax); /\* Temporal type classification \*/   
4 $\mathcal{E}_{type}\gets$ GetTypeExp(Epool,Q,Wexp);   
5 Type(Q) $\leftarrow$ PromptTypeSelect(Q, $\mathcal{E}_{type}$ );   
6 Return $\mathcal{T}_Q$ {I},Topics,Type(Q), $\mathcal{G}_Q$

Algorithm 2: TreeofTimeReasoning   
Input : indicators $\{\mathrm{I}_i\}$ , subgraph $\mathcal{G}_Q$ topics Topics, experience pool $\mathcal{E}_{pool}$ limits $D_{\mathrm{max}}$ $B_{\mathrm{max}}$ $W_{\mathrm{max}}$ $W_{\mathrm{exp}}$ Output: Local answers/paths per node; updated tree   
1 $\mathcal{E}_{decomp}\gets$ GetDecompExp( $\mathcal{E}_{pool},Q,W_{\mathrm{exp}},$ Type(Q));   
2 if $\exists$ compatible plan in $\mathcal{E}_{decomp}$ then   
3 $\mathcal{T}_Q\gets$ ReusePlan( $\mathcal{E}_{decomp},Q,$ Topics);   
4 else $\mathcal{T}_Q\gets$ PromptDecompose $(Q,\mathcal{E}_{decomp},$ Type(Q));   
5 $\{\mathrm{I}_i\} \leftarrow$ ExtractIndicators $(\mathcal{T}_Q)$ .   
6 for each $(q_i,I_i)\in$ TraverseRootToLeaf $(\mathcal{T}_Q)$ do   
7 $(\hat{a},\hat{P},$ sufficient) $\leftarrow$ MemoryLookupAndTest(qi, $\mathrm{I}_i$ , $\mathcal{E}_{pool}$ ); if sufficient then continue;   
9 $\mathcal{E}_{tool}\gets$ GetToolkitExp( $\mathcal{E}_{pool},q_i,I_i$ ,Type(Q),Wexp);   
10 $\mathcal{T}_i\gets$ PromptToolkitSelect $(\mathcal{T}_{\theta},\mathrm{I}_i,q_i,\mathcal{E}_{tool})$ .. $(C_i,S_i)\gets$ TemporalRetrieveAndPrune $(\mathcal{G}_Q,q_i,I_i,T_\text{opics},\mathcal{T}_i,\mathcal{E}_{pool},W_\mathrm{max})$ . $(a_{i},P_{i}$ ,sufficient) $\leftarrow$ DebateVoteAndTest(qi, $\mathrm{I}_i,C_i$ );   
13 if -sufficient then   
14 if DepthOK A BranchOK then   
15 {qnew}, $\{\mathrm{I}_{\mathrm{new}}\} \leftarrow$ RefineOrDecompose(qi,Ii,Ci);   
16 UpdateTree $(\mathcal{T}_Q,\{q_{new}\} ,\{\mathrm{I}_{new}\})$ .   
17 else WriteBackIfSuccess( $\mathcal{E}_{pool},q_i,I_i,a_i,P_i,\mathcal{T}_Q)$ .   
18 Return $\{a_i\} ,\{p_i\} ,\mathcal{T}_Q;$

Algorithm 3: TemporalRetrieveAndPrune   
Input : Subquestion $(q_{i},I_{i})$ , topics Topics, selected toolkits $\mathcal{T}_i$ experience pool $\mathcal{E}_{\mathrm{pool}}$ , subgraph $\mathcal{G}_Q$ , limits $W_{\mathrm{max}}$ $W_{\mathrm{exp}}$ $D_{\mathrm{max}}$ Output: Pruned candidates $C_i$ , selected seeds $S_{i}$ 1 $\mathcal{E}_{\mathrm{seed}}\gets$ GetSeedExp( $\mathcal{E}_{\mathrm{pool}},q_i,I_i,W_{\mathrm{exp}})$ .   
2 $S_{i}\gets$ PromptSeedsSelect(Topics, $I_{i}$ $q_{i}$ $\mathcal{E}_{\mathrm{seed}}$ ); $C\gets 0$ .   
3 for each $T_{\theta}\in \mathcal{T}_{i}$ in parallel do   
4 C C U TreeBasedTemporalPathRetrieval $(G_Q,T_\theta ,I_i,S_i,D_{\mathrm{max}})$ .   
5 C C U DenseEmbedRetrieve $(I_I,T_\theta ,Docu(G_Q))$ .   
6 $\widetilde{C}\gets \{p\in C|p\in C_{time}(I_i)\wedge Monotone(p)\} ;$ 7 for each $p\in \widetilde{C}$ do   
8 ssem $(p)\gets$ SemanticDRM(Ii,p);   
9 sprox $(p)\gets \exp (-|t(p) - t(I_i)| / \sigma)$ .   
10 Score $(p)\gets \lambda_{\mathrm{sem}}s_{\mathrm{sem}}(p) + \lambda_{\mathrm{prox}}s_{\mathrm{prox}}(p)$ .   
11 $\widetilde{C}\gets$ SelectTopPathandSort(C,W1);   
12 $C_i\gets$ PromptSelectPath $(\widetilde{C},q_i,I_i,W_{\mathrm{max}})$ .   
13 Return $C_i,S_i$ .   
14 Procedure TreeBasedTemporalPathRetrieval $(G_Q,T_\theta ,I_i,S_i,D_{\mathrm{max}})$ 15 StartList, $C_{time},D_{pred}\gets$ ExtractInfo $(T_{\theta},I_i,S_i)$ .   
16 D 1; Paths 0; Frontier $\leftarrow \{(s,t_0,[])\mid s\in \text{StartList},t_0 = \text{InitTime} (C_{time})\right\}$ 17 while $D\leq D_{\mathrm{max}}$ do   
18 NewFrontier 0;   
for each $(v,t_{last},P)\in$ Frontier do   
20 for each $(v\xrightarrow{r,t_e}u)\in$ ExpandOneHop $(G_Q,v,T_\theta)$ do if-Monotone(tlast,te) then continue   
22 P' P' (v r,te u); Paths Paths U {P}; NewFrontier $\leftarrow$ NewFrontier U {(u,te,P')};   
23 if |NewFrontier> Wmax then   
26 NewFrontier $\leftarrow$ RelevantPruning(NewFrontier,Ii,Wmax);   
27 Frontier 1 NewFrontier; $D\gets D + 1$ 28 Return Paths;

Algorithm 4: Experience Memory   
Input: Experience pool $\mathcal{E}_{pool}$ , query object (question or sub-question), indicator I, type label $\tau$ , buffer size $K$ Output: Matched exemplars $\mathcal{E}_{match}$ (optional), updated pool  
1 $e_{q} \gets \text{Encode}(q)$ ; $e_{l} \gets \text{Encode}(l)$ ;  
2 $\mathcal{E}_{match} \gets \text{ANN\_Search}(\mathcal{E}_{pool}, [e_{q}, e_{l}], \text{filter: type} = \tau, \text{top-} K)$ ;  
3 RankBy( $\lambda_{\text{sim}} \cdot \text{sim} + \lambda_{\text{hit}} \cdot \text{hit\_count}$ );  
4 Return $\mathcal{E}_{match}$ ;  
5 Procedure WriteBackIfSuccess( $\mathcal{E}_{pool}, q, I, a, P$ )  
6 Store(q, I, a, P, $\tau$ , embeddings = $[e_{q}, e_{l}]$ , sufficient = true);  
7 UpdateBufferStats(q, I);  
8 Procedure MemoryLookupAndTest(q, I, $\mathcal{E}_{pool}$ )  
9 Hist $\leftarrow$ Retrieve(q, I, filter: type = $\tau$ );  
10 for each ( $\hat{a}, \hat{P}$ ) in Hist do  
11 $\lfloor$ if Sufficient(q, I, $\hat{a}, \hat{P}$ ) then Return $\hat{a}, \hat{P}$ , sufficient = true;  
12 Return $\lfloor, \lfloor, \rfloor, \rfloor$ sufficient = false;

Table 5: Multi-granularity temporal reasoning on Hits@1. Best in each column is in bold.   
Table 6: Efficiency analysis of MemoTime on MultiTQ.   

<table><tr><td rowspan="2">Model</td><td colspan="3">Equal</td><td colspan="3">Before/After</td><td colspan="3">Equal-Multi</td></tr><tr><td>Day</td><td>Month</td><td>Year</td><td>Day</td><td>Month</td><td>Year</td><td>Day</td><td>Month</td><td>Year</td></tr><tr><td>BERT</td><td>4.9</td><td>10.3</td><td>13.6</td><td>15.0</td><td>16.4</td><td>17.5</td><td>6.4</td><td>10.2</td><td>9.0</td></tr><tr><td>DistillBERT</td><td>4.1</td><td>8.7</td><td>11.3</td><td>16.0</td><td>15.0</td><td>18.6</td><td>9.6</td><td>12.7</td><td>8.9</td></tr><tr><td>ALBERT</td><td>6.9</td><td>8.2</td><td>13.2</td><td>22.1</td><td>27.7</td><td>30.8</td><td>10.3</td><td>14.4</td><td>14.4</td></tr><tr><td>EmbedKGQA</td><td>20.0</td><td>33.6</td><td>21.8</td><td>39.2</td><td>51.8</td><td>51.1</td><td>14.5</td><td>32.1</td><td>26.3</td></tr><tr><td>CronKGQA</td><td>42.5</td><td>38.9</td><td>33.1</td><td>37.5</td><td>47.4</td><td>45.0</td><td>29.5</td><td>33.3</td><td>25.1</td></tr><tr><td>MultiQA</td><td>44.5</td><td>39.3</td><td>35.0</td><td>37.9</td><td>54.8</td><td>52.5</td><td>30.8</td><td>32.1</td><td>28.3</td></tr><tr><td>MemoTime w/ DeepSeek-V3</td><td>85.6</td><td>85.1</td><td>92.3</td><td>81.4</td><td>77.4</td><td>82.4</td><td>60.0</td><td>55.3</td><td>53.2</td></tr><tr><td>MemoTime w/ GPT-4-Turbo</td><td>85.8</td><td>85.0</td><td>93.8</td><td>92.5</td><td>86.2</td><td>94.1</td><td>68.4</td><td>70.4</td><td>64.4</td></tr></table>

<table><tr><td></td><td>Overall</td><td>After-First</td><td>Before-Last</td><td>Equal-Multi</td><td>First/Last</td><td>Before/After</td><td>Equal</td></tr><tr><td>Average Depth</td><td>1.37</td><td>2.12</td><td>2.00</td><td>1.57</td><td>1.00</td><td>1.54</td><td>1.04</td></tr><tr><td>Average Branch</td><td>2.64</td><td>3.91</td><td>3.94</td><td>3.05</td><td>2.00</td><td>2.78</td><td>2.01</td></tr><tr><td>Average LLM Calls</td><td>8.75</td><td>11.55</td><td>11.49</td><td>9.88</td><td>7.42</td><td>9.10</td><td>7.32</td></tr><tr><td>Running Time (s)</td><td>43.11</td><td>58.11</td><td>57.89</td><td>48.86</td><td>35.70</td><td>45.34</td><td>35.42</td></tr></table>

# B Additional Experiments

# B.1 Multi-Granular Temporal QA Analysis

We further evaluate MemoTime on temporal reasoning tasks that require distinguishing and aligning facts across different temporal granularities, including day, month, and year levels. This setting examines whether models can adapt to diverse temporal resolutions and maintain consistent reasoning performance. As shown in Table 5, traditional pre-trained and embedding-based baselines exhibit significant degradation as the temporal resolution becomes finer or when multiple granularities are combined. For example, while MultiQA achieves $4 4 . 5 \%$ accuracy on the Equal–Day level, its performance drops sharply to $2 8 . 3 \%$ on Equal–Multi–Year, indicating that these models struggle to maintain chronological consistency when reasoning across mixed time scales.

In contrast, MemoTime consistently achieves strong and stable results across all granularities and operators, reaching up to $9 4 . 1 \%$ Hits@1 on Before/After–Year and maintaining $7 0 . 4 \%$ under the most complex Equal–Multi–Month setting. The consistent improvement across both fine-grained and compound temporal tasks demonstrates that MemoTime effectively adapts to the diversity of temporal reasoning requirements. This robustness is mainly attributed to two key components: (1) operator-aware decomposition, which dynamically adjusts reasoning depth and retrieval scope according to the temporal operator, and (2) temporal path construction, which enforces monotonic timestamp progression and co-aligns evidence across multiple resolutions. Together, these mechanisms enable unified and faithful temporal reasoning while preserving interpretability across heterogeneous granularities.

# B.2 Efficiency Analysis

To comprehensively evaluate the efficiency of MemoTime, we conduct three analyses on the MultiTQ dataset: LLM calls cost analysis, running time analysis, and computational cost analysis. Table 6 presents the averaged statistics across different question types, providing insights into reasoning efficiency and structural scalability.

LLM calls cost analysis. To measure the efficiency of utilizing LLMs, we analyze the average number of LLM calls required to complete a reasoning cycle across different question types. As shown in Table 6, MemoTime performs an average of 8.75 calls per question, demonstrating its lightweight decomposition and reasoning pipeline. For simple operator types such as Equal and First/Last, the number of calls remains low (7–7.5), corresponding to a single reasoning round without recursive decomposition. In contrast, complex composite operators including After–First and Before–Last require deeper temporal validation, resulting in approximately 11.5 LLM calls. These results demonstrate that MemoTime maintains efficient reasoning, even for multi-hop and temporally constrained queries, by avoiding redundant invocations through structured decomposition and memory reuse.

Running time analysis. To further assess inference efficiency, we examine the average running time per question for different temporal types. The overall average runtime of MemoTime is 43.11 seconds per question, indicating an effective balance between retrieval and reasoning processes. Operator-specific analysis shows that single-hop or non-nested operators such as Equal and First/Last complete within approximately 35 seconds, whereas multi-hop operators like After–First and Before–Last require additional retrieval and temporal alignment, increasing the runtime to around 58 seconds. Despite these additional steps, MemoTime preserves

Table 7: MemoTime toolkit library with purposes, key parameters, and operator mappings.   

<table><tr><td>Toolkit</td><td>Purpose</td><td>Key parameters</td><td colspan="2">Operator mapping</td></tr><tr><td>OneHop</td><td>One-hop neighbors with temporal filters</td><td>entity, direction, after/before, limit</td><td>seed</td><td>local</td></tr><tr><td>AfterFirst</td><td>First Nth event after a cutoff</td><td>entity, after, relation_filter, limit=N</td><td>after_first</td><td></td></tr><tr><td>BeforeLast</td><td>Last Nth event before a cutoff</td><td>entity, before, relation_filter, limit=N</td><td>before_last</td><td></td></tr><tr><td>BetweenRange</td><td>Events within a time window</td><td>entity, between=(start,end), granularity</td><td>during</td><td>between</td></tr><tr><td>DayEvents</td><td>Global events on a specific date</td><td>date, relation_filter, limit</td><td>same-day</td><td>snapshot</td></tr><tr><td>Month/YearEvents</td><td>Global events in a month/year</td><td>month or year, relation_filter, limit</td><td>same-month</td><td>same-year</td></tr><tr><td>DirectConnection</td><td>Direct edges between two entities</td><td>entity1, entity2, direction, time filters</td><td>pairwise</td><td>validate</td></tr><tr><td>Timeline</td><td>Chronological sequence for an entity</td><td>entity, direction, after/before, limit</td><td>ordering</td><td>stitch</td></tr></table>

scalability by dynamically pruning irrelevant temporal paths and parallelizing evidence collection within each hierarchical layer.

Computational cost analysis. To evaluate the structural efficiency of temporal reasoning, we analyze the average recursion depth and branching factor, which jointly characterize the complexity of decomposition and search. MemoTime exhibits a compact reasoning structure, with an average recursion depth of 1.37 and branching factor of 2.64, indicating efficient but expressive reasoning behavior. As shown in Table 6, composite operators such as After–First and Before–Last require deeper reasoning hierarchies (depth around 2) and broader exploration (branch factor close to 4), whereas simpler operators such as Equal and First/Last remain shallow and nearly linear (depth near 1, branch factor around 2). These results demonstrate that MemoTime dynamically adjusts its computational footprint according to the temporal operator’s semantic complexity, achieving consistent temporal alignment while maintaining efficient reasoning performance.

# B.3 Case Study: Temporal Interpretability and Faithful Reasoning

To demonstrate how MemoTime performs interpretable and temporally faithful reasoning, in this section, we present case studies across different temporal question types in Tables 9–13. Each case study illustrates how complex temporal questions are decomposed into sub-questions with structured indicators, toolkit selections, and retrieved factual quadruples. Through examples involving diverse operators such as afterNfirst, beforeNlast, and between, we show how MemoTime incrementally constructs temporal reasoning chains that maintain monotonic time ordering, respect operatorspecific constraints, and yield consistent answers. These examples highlight how the proposed framework ensures transparency and temporal faithfulness throughout the reasoning process, producing clear, explainable evidence trails that align with humanunderstandable logical steps.

# C Toolkit Library

We implement eight specialized temporal retrieval toolkits exposing a unified interface with normalized parameters. Table 7 outlines their core purposes, key parameters, and corresponding temporal

operators. The tools are composable and operator-aware under temporal constraints. They cover the majority of temporal reasoning patterns required for multi-hop question answering.

# D Experiment Details

Experiment Datasets. Previous studies have revealed that the widely used CronQuestions dataset [30] contains spurious correlations that models can exploit to achieve inflated accuracy [38]. Following the analysis in [35], we therefore evaluate MemoTime on two more reliable and temporally challenging benchmarks: MultiTQ [6] and TimeQuestions [18]. MultiTQ is the largest publicly available Temporal Knowledge Graph Question Answering (TKGQA) dataset, constructed from the ICEWS05–15 event database, which records politically and socially significant events with precise timestamps. It consists of more than 500K question–answer pairs, covering 15 years of temporal scope. Each question is automatically generated from event triples $( h , r , t , \tau )$ and further refined to ensure diversity in natural language phrasing. The dataset supports multiple temporal granularities (year, month, and day), with questions distributed across over 3,600 calendar days. MultiTQ encompasses a wide range of temporal reasoning operators, including before, after, equal, first, and last, and contains both single-hop and multi-hop reasoning cases that require combining multiple temporal relations to derive the final answer. In contrast, TimeQuestions is a smaller but linguistically richer benchmark designed to test compositional and relational temporal reasoning. It contains around 16K manually curated questions spanning four reasoning categories: Explicit, Implicit, Temporal, and Ordinal. Unlike MultiTQ, it uses only yearly granularity and focuses on questions that involve reasoning over alignment and temporal order rather than precise timestamps. Each question is grounded in Wikidata events and entities, with linguistic templates textasizing contextual reasoning, event sequencing, and temporal comparisons across multiple entities. The detailed statistics of both datasets, including splits by reasoning type and dataset size, are summarized in Table 8.

Baselines. We compare MemoTime against a comprehensive set of baselines covering three major methodological categories: pretrained language models, knowledge graph embedding methods, and LLM-based reasoning methods.

On MultiTQ, we include: (1) Pre-trained language models (PLMs) such as BERT [8] and ALBERT [21], which are fine-tuned for QA over temporal contexts; (2) Embedding-based temporal KGQA methods, including EmbedKGQA [37], CronKGQA [36], and MultiQA [5], which leverage temporal embeddings to align question representations with time-stamped entity triples; (3) LLM-based in-context learning (ICL) approaches, such as direct instruction prompting (IO), KG-RAG [4], ReAct-KB [4, 61], ARI [4], and TempAgent [13], which combine TKG and reasoning via ChatGPT or similar models.

On TimeQuestions, we evaluate: (1) Static KGQA baselines, including PullNet [42], Uniqorn [34], and GRAFT-Net [41], which operate on non-temporal knowledge graphs; (2) Temporal KGQA baselines, including CronKGQA [36], TempoQR [28], EXAQT [18], and LGQA [25]; (3) LLM-based methods, including instruction prompting (IO) and two fine-tuned reasoning models: GenTKGQA [11], which fine-tunes TKG with LLaMA2-7B, and TimeR4 [35], which couples a jointly fine-tuned retriever with ChatGPT reasoning.

For each baseline, we adopt the reported performance values from their original publications to ensure comparability under consistent evaluation settings. Following [27, 43, 47], we use exact match accuracy (Hits@1) as the principal evaluation metric. Recall and F1 scores are not used since knowledge sources are not limited to document databases [27, 43].

Experiment Implementation. All experiments are conducted using GPT-4-Turbo as the primary reasoning backbone for MemoTime. To validate the generality and plug-and-play adaptability of our framework, we further instantiate MemoTime with several alternative LLMs, including DeepSeek-V3, Qwen3-Next-80B-A3B-Instruct (Qwen3-80B), Qwen3-32B, Qwen3-8B, and Qwen3-4B. These models represent a diverse spectrum of capacities and architectural scales, allowing us to evaluate how MemoTime performs under different model sizes and decoding behaviors. Following prior work [43, 47], the temperature is set to 0.4 during the evidence exploration phase to encourage reasoning diversity, and to 0 during the final answer generation stage to ensure deterministic output.The maximum generation length is 256 tokens. Sentence-BERT [8] is utilized as dense retrieval module (DRM). For temporal path construction, we set $W _ { \mathrm { m a x } } = 3$ , $D _ { \mathrm { m a x } } = 3$ , $W _ { 1 } = 8 0$ , and $\lambda _ { \mathrm { s e m } } = 0 . 6$ . During the experience memory phase, we use $\lambda _ { \mathrm { s i m } } = 0 . 6$ , $\lambda _ { \mathrm { h i t } } = 0 . 4$ , and $W _ { \mathrm { e x p } } = 1 0$ . All experiences are dynamically learned from verified reasoning traces, with five predefined exemplars used for cold-start initialization. The database buffer size is set to 200 for efficiency. The complete code implementation of MemoTime is publicly available.1

Table 8: Statistics of MultiTQ and TimeQuestions.   

<table><tr><td colspan="2">Category</td><td>Train</td><td>Dev</td><td>Test</td></tr><tr><td colspan="5">MultiTQ</td></tr><tr><td rowspan="3">Single</td><td>Equal</td><td>135,890</td><td>18,983</td><td>17,311</td></tr><tr><td>Before/After</td><td>75,340</td><td>11,655</td><td>11,073</td></tr><tr><td>First/Last</td><td>72,252</td><td>11,097</td><td>10,480</td></tr><tr><td rowspan="3">Multiple</td><td>Equal-Multi</td><td>16,893</td><td>3,213</td><td>3,207</td></tr><tr><td>After-First</td><td>43,305</td><td>6,499</td><td>6,266</td></tr><tr><td>Before-Last</td><td>43,107</td><td>6,532</td><td>6,247</td></tr><tr><td colspan="2">Total (MultiTQ)</td><td>386,787</td><td>57,979</td><td>54,584</td></tr><tr><td colspan="5">TimeQuestions</td></tr><tr><td colspan="2">Explicit</td><td>2,724</td><td>1,302</td><td>1,311</td></tr><tr><td colspan="2">Implicit</td><td>651</td><td>291</td><td>292</td></tr><tr><td colspan="2">Temporal</td><td>2,657</td><td>1,073</td><td>1,067</td></tr><tr><td colspan="2">Ordinal</td><td>938</td><td>570</td><td>567</td></tr><tr><td colspan="2">Total (TimeQuestions)</td><td>6,970</td><td>3,236</td><td>3,237</td></tr></table>

Table 9: Case study of interpretability and temporal faithfulness reasoning for “After the 2008 Olympics, which country was the first to sign an environmental treaty with China?”   

<table><tr><td>Question</td><td>After the 2008 Olympics, which country was the first to sign an environmental treaty with China?</td></tr><tr><td>Temporal Type</td><td>afterNfirst (requires t2 &gt; t1 and min(t2))</td></tr><tr><td>Topic Entities</td><td>[ Olympics 2008, China]</td></tr><tr><td>Overall Indicator</td><td>Edges: (Athletics 2008, opening, ?x, t1), (?y, sign environmental treaty, China, t2)Constraints: t2 &gt; t1, after_first(t2, t1)Time vars: [t1, t2]</td></tr><tr><td>Q1</td><td>When was the 2008 Olympics held (opening anchor)?</td></tr><tr><td>Selected Seed</td><td>[ Olympics 2008]</td></tr><tr><td>Indicator (quadruple)</td><td>(Olympics 2008, opening, ?x, t1)</td></tr><tr><td>Time vars</td><td>[same_year(t1, 2008)]</td></tr><tr><td>Toolkit &amp; Params</td><td>OneHop(entity = Olympics 2008, after = 2008-01-01, before = 2009-01-01)</td></tr><tr><td>Retrieved Facts</td><td>(Olympics 2008, opening, Beijing, 2008-08-08);(Athletics 2008, closing, Beijing, 2008-08-24)</td></tr><tr><td>Sub-answer</td><td>t1 = 2008-08-08 (use opening as temporal anchor; ?x = Beijing)</td></tr><tr><td>Q2</td><td>After 2008-08-08, which country first signed an environmental treaty with China?</td></tr><tr><td>Selected Seed</td><td>[China]</td></tr><tr><td>Indicator (quadruple)</td><td>(?y, sign environmental treaty, China, t2)</td></tr><tr><td>Time vars</td><td>[after(t2, 2008-08-08), after_first(t2, 2008-08-08)]</td></tr><tr><td>Toolkit &amp; Params</td><td>AfterFirst-entity = China, after = 2008-08-08, relation_filter = sign environmental treaty)</td></tr><tr><td>Retrieved Facts</td><td>(Japan, sign_treaty_env, China, 2009-02-10);(Korea, sign_treaty_env, China, 2009-07-18);(Germany, sign_treaty_env, China, 2010-03-02)</td></tr><tr><td>Sub-answer</td><td>Japan, t2 = 2009-02-10 (earliest valid time after 2008-08-08)</td></tr><tr><td>Temporal Reasoning Chain</td><td>Olympics 2008 \(\frac{\text{opening}}{2008-08-08}\)Beijing \(\twoheadrightarrow\) Japan \(\frac{\text{sign_treaty_env}}{2009-02-10}\)China (after_first 2008-08-08).</td></tr><tr><td>Temporal Faithfulness</td><td>All facts satisfy t2 &gt; t1; min(t2) = 2009-02-10; monotonic ordering verified.</td></tr><tr><td>Final Answer</td><td>Japan.</td></tr><tr><td>Response</td><td>&quot;Anchoring t1 at 2008-08-08 (opening of the Olympics), the earliest post-anchor treaty signing with China is by Japan on 2009-02-10, earlier than Korea (2009-07-18) and Germany (2010-03-02). Hence, the after first constraint selects Japan.&quot;</td></tr></table>

Table 10: Case study of interpretability and temporal faithfulness reasoning for “Who was the last leader to visit Beijing before 2010?”   

<table><tr><td>Question</td><td>Who was the last leader to visit Beijing before 2010?</td></tr><tr><td>Temporal Type</td><td>beforeNlast (requires t1&lt; 2010 and max(t1))</td></tr><tr><td>Topic Entities</td><td>[Beijing, leader]</td></tr><tr><td>Overall Indicator</td><td>Edges: (?x, visit, Beijing, t1)Constraints: t1&lt; 2010, before_last(t1, 2010)Time vars: [t1]</td></tr><tr><td>Q1</td><td>Which leaders visited Beijing before 2010?</td></tr><tr><td>Selected Seed</td><td>[Beijing]</td></tr><tr><td>Indicator (quadruple)</td><td>(?x, visit, Beijing, t1)</td></tr><tr><td>Time vars</td><td>[t1&lt; 2010]</td></tr><tr><td>Toolkit &amp; Params</td><td>Before(entity = Beijing, before = 2010-01-01, relation_filter = visit)</td></tr><tr><td>Retrieved Facts</td><td>(Barack Obama, visit, Beijing, 2009-11-15);(Gordon Brown, visit, Beijing, 2009-08-10);(Angela Merkel, visit, Beijing, 2008-12-05)</td></tr><tr><td>Sub-answer</td><td>Candidate set = [Obama 2009-11-15, Brown 2009-08-10, Merkel 2008-12-05].</td></tr><tr><td>Q2</td><td>Among them, who visited most recently before 2010?</td></tr><tr><td>Selected Seed</td><td>[Candidate leaders]</td></tr><tr><td>Indicator (quadruple)</td><td>(?x, visit, Beijing, t1)</td></tr><tr><td>Time vars</td><td>[before(t1, 2010), last(t1)]</td></tr><tr><td>Toolkit &amp; Params</td><td>FirstLast(mode = last, relation_filter = visit, before = 2010-01-01, sort = desc, limit = 1)</td></tr><tr><td>Retrieved Facts</td><td>(Barack Obama, visit, Beijing, 2009-11-15)</td></tr><tr><td>Sub-answer</td><td>Barack Obama, t1= 2009-11-15 (latest valid before 2010).</td></tr><tr><td>Temporal Reasoning Chain</td><td>Barack Obama visit2009-11-15Beijing (before_last 2010).</td></tr><tr><td>Temporal Faithfulness</td><td>All events satisfy t1&lt; 2010; max(t1) = 2009-11-15; temporal order monotonic.</td></tr><tr><td>Final Answer</td><td>Barack Obama.</td></tr><tr><td>Response</td><td>“Filtering visits to Beijing with t1&lt; 2010 yields three leaders; the most recent timestamp (2009-11-15) corresponds to Barack Obama, fulfilling the before last constraint.”</td></tr></table>

Table 11: Case study of interpretability and temporal faithfulness reasoning for “Before the 2010 Summit, which leader visited Beijing last?”   

<table><tr><td>Question</td><td>Before the 2010 Summit, which leader visited Beijing last?</td></tr><tr><td>Temporal Type</td><td>beforeNlast (requires t2 &lt; t1 and max(t2))</td></tr><tr><td>Topic Entities</td><td>[2010 Summit, Beijing]</td></tr><tr><td>Overall Indicator</td><td>Edges: (Leader, visit, Beijing, t2), (2010 Summit, held_in, ?x, t1)Constraints: t2 &lt; t1, before_last(t2, t1)Time vars: [t1, t2]</td></tr><tr><td>Q1</td><td>When was the 2010 Summit held?</td></tr><tr><td>Selected Seed</td><td>[2010 Summit]</td></tr><tr><td>Indicator (quadruple)</td><td>(2010 Summit, held_in, ?x, t1)</td></tr><tr><td>Time vars</td><td>[specific_year(t1, 2010)]</td></tr><tr><td>Toolkit &amp; Params</td><td>OneHop(entity = 2010 Summit, after = 2010-01-01, before = 2011-01-01)</td></tr><tr><td>Retrieved Facts</td><td>(2010 Summit, held_in, Toronto, 2010-06-26)</td></tr><tr><td>Sub-answer</td><td>t1 = 2010-06-26 (used as upper temporal boundary)</td></tr><tr><td>Q2</td><td>Before 2010-06-26, which leader visited Beijing last?</td></tr><tr><td>Selected Seed</td><td>[Beijing]</td></tr><tr><td>Indicator (quadruple)</td><td>(?y, visit, Beijing, t2)</td></tr><tr><td>Time vars</td><td>[before(t2, 2010-06-26), before_last(t2, 2010-06-26)]</td></tr><tr><td>Toolkit &amp; Params</td><td>BeforeLast entity = Beijing, before = 2010-06-26, relation_filter = visit)</td></tr><tr><td>Retrieved Facts</td><td>(Barack Obama, visit, Beijing, 2009-11-15);(Gordon Brown, visit, Beijing, 2009-08-10);(Angela Merkel, visit, Beijing, 2008-12-05)</td></tr><tr><td>Sub-answer</td><td>Barack Obama, t2 = 2009-11-15 (latest valid time before 2010-06-26)</td></tr><tr><td>Temporal Reasoning Chain</td><td>Barack Obama visit 2009-11-15 → Beijing ∼ 2010 Summit held_in 2010-06-26 → Toronto (before_last 2010-06-26).</td></tr><tr><td>Temporal Faithfulness</td><td>All facts satisfy t2 &lt; t1; max(t2) = 2009 - 11 - 15; monotonic order verified.</td></tr><tr><td>Final Answer</td><td>Barack Obama.</td></tr><tr><td>Response</td><td>“Anchoring t1 at 2010-06-26 (2010 Summit), the last recorded visit to Beijing before this date was by Barack Obama on 2009-11-15. Earlier visits (Gordon Brown 2009-08-10, Angela Merkel 2008-12-05) confirm Obama as the before_last case.”</td></tr></table>

Table 12: Case study of interpretability and temporal faithfulness reasoning for “How many times did the UN hold a climate summit before 2020?”   

<table><tr><td>Question</td><td>How many times did the UN hold a climate summit before 2020?</td></tr><tr><td>Temporal Type</td><td>count + before (requires t1&lt; 2020 and aggregation over events)</td></tr><tr><td>Topic Entities</td><td>[UN, climate summit]</td></tr><tr><td>Overall Indicator</td><td>Edges: (UN, hold, Summit, t1)Constraints: t1&lt; 2020, topic(Summit) = climateTime vars: [t1]</td></tr><tr><td>Q1</td><td>Which climate summits were held by the UN before 2020?</td></tr><tr><td>Selected Seed</td><td>[UN]</td></tr><tr><td>Indicator (quadruple)</td><td>(UN, hold, Summit, t1)</td></tr><tr><td>Time vars</td><td>[t1&lt; 2020]</td></tr><tr><td>Toolkit &amp; Params</td><td>Before( entity = UN, before = 2020-01-01, relation_filter = hold, keyword = climate)</td></tr><tr><td>Retrieved Facts</td><td>(UN, hold, ClimateSummit09, 2009-12-07);(UN, hold, ClimateSummit13, 2013-11-11);(UN, hold, ClimateSummit15, 2015-12-05);(UN, hold, ClimateSummit19, 2019-09-21);(UN, hold, ClimateSummit21, 2021-11-01)</td></tr><tr><td>Sub-answer</td><td>Valid events before 2020 = 4 (2009, 2013, 2015, 2019).</td></tr><tr><td>Q2</td><td>Count all valid events that occurred before 2020.</td></tr><tr><td>Selected Seed</td><td>[Filtered summits from Q1]</td></tr><tr><td>Indicator (quadruple)</td><td>(UN, hold, ClimateSummit_i, t1)</td></tr><tr><td>Time vars</td><td>[count(t1&lt; 2020)]</td></tr><tr><td>Toolkit &amp; Params</td><td>Count.filter= t1&lt; 2020)</td></tr><tr><td>Retrieved Facts</td><td>Total = 4 events before 2020.</td></tr><tr><td>Sub-answer</td><td>4 climate summits held before 2020.</td></tr><tr><td>Temporal Reasoning Chain</td><td>UN hold 2009-2019 ClimateSummit_i (4 instances before 2020).</td></tr><tr><td>Temporal Faithfulness</td><td>All timestamps t1&lt; 2020; aggregation respects strict temporal filter.</td></tr><tr><td>Final Answer</td><td>4 climate summits.</td></tr><tr><td>Response</td><td>“Filtering UN-held climate summits before 2020 yields four valid temporal events (2009, 2013, 2015, 2019). All satisfy t1&lt; 2020, ensuring temporally faithful counting.”</td></tr></table>

Table 13: Case study of interpretability and temporal faithfulness reasoning for “Between the 2015 Conference and the 2018 Summit, which company collaborated with Microsoft?”   

<table><tr><td>Question</td><td>Between the 2015 Conference and the 2018 Summit, which company collaborated with Microsoft?</td></tr><tr><td>Temporal Type</td><td>between (requires t1 &lt; t3 &lt; t2 and bounded interval)</td></tr><tr><td>Topic Entities</td><td>[2015 Conference, 2018 Summit, Microsoft]</td></tr><tr><td>Overall Indicator</td><td>Edges: (2015 Conference, held in, ?x, t1), (2018 Summit, held in, ?y, t2), (?z, collaborate with, Microsoft, t3)Constraints: t1 &lt; t3 &lt; t2, between(t3, [t1, t2])Time vars: [t1, t2, t3]</td></tr><tr><td>Q1</td><td>When was the 2015 Conference held?</td></tr><tr><td>Selected Seed</td><td>[2015 Conference]</td></tr><tr><td>Indicator</td><td>(2015 Conference, held in, ?x, t1)</td></tr><tr><td>Toolkit &amp; Params</td><td>OneHop(enterprise = 2015 Conference, after = 2015-01-01, before = 2016-01-01)</td></tr><tr><td>Retrieved Facts</td><td>(2015 Conference, held_in, New York, 2015-09-10)</td></tr><tr><td>Sub-answer</td><td>t1 = 2015-09-10</td></tr><tr><td>Q2</td><td>When was the 2018 Summit held?</td></tr><tr><td>Selected Seed</td><td>[2018 Summit]</td></tr><tr><td>Indicator</td><td>(2018 Summit, held in, ?y, t2)</td></tr><tr><td>Toolkit &amp; Params</td><td>OneHop(enterprise = 2018 Summit, after = 2018-01-01, before = 2019-01-01)</td></tr><tr><td>Retrieved Facts</td><td>(2018 Summit, held_in, Singapore, 2018-11-22)</td></tr><tr><td>Sub-answer</td><td>t2 = 2018-11-22</td></tr><tr><td>Q3</td><td>Between t1 and t2, which company collaborated with Microsoft?</td></tr><tr><td>Selected Seed</td><td>[Microsoft]</td></tr><tr><td>Indicator</td><td>(?z, collaborate with, Microsoft, t3)</td></tr><tr><td>Time vars</td><td>[between(t3, [t1, t2])]</td></tr><tr><td>Toolkit &amp; Params</td><td>BetweenRange(enterprise = Microsoft, between = (t1, t2), relation_filter = collaborate with)</td></tr><tr><td>Retrieved Facts</td><td>(NVIDIA, collaborate_with, Microsoft, 2016-05-20); (OpenAI, collaborate_with, Microsoft, 2018-03-14); (Apple, collaborate_with, Microsoft, 2019-02-10)</td></tr><tr><td>Sub-answer</td><td>NVIDIA (2016-05-20) and OpenAI (2018-03-14), both satisfying t1 &lt; t3 &lt; t2</td></tr><tr><td>Temporal Reasoning Chain</td><td>2015 Conference → NVIDIA → Microsoft → OpenAI → collaborate_with → Microsoft → OpenAI → collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsemit → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → Collaborate_with → Microsoft → COLLAB</td></tr><tr><td>Temporal Faithfulness</td><td>All facts satisfy t1 &lt; t3 &lt; t2; valid interval reasoning confirmed.</td></tr><tr><td>Final Answer</td><td>NVIDIA, OpenAI.</td></tr><tr><td>Response</td><td>“Anchoring the 2015 Conference at 2015-09-10 and the 2018 Summit at 2018-11-22, the companies collaborating with Microsoft within this interval are NVIDIA (2016) and OpenAI (2018). Both satisfy the temporal between constraint.”</td></tr></table>

# E Prompts

In this section, we detail the prompts required for our main experimental procedures.

# Temporal Type Classification Prompt Template

Given a natural-language question that may include explicit dates, relative temporal expressions, or comparative phrases, your task is to classify it into exactly one supported temporal type that best represents its temporal intent and logical operator. If multiple categories appear possible, choose the most specific operator-sensitive type.

Supported temporal types (single label only): equal, before, after, during, between, first, last, beforeNlast, afterNfirst, count, comparison.

Experience Examples: {In-Context Few-shot}

Q: {Question} A:

The {In-Context Few-shot} examples are retrieved from experience memory, representing previously successful classification traces. They guide the model in identifying the temporal operator patterns (e.g., “first Y after $Z ^ { \ast }$ ) and selecting the most appropriate category.

# Question Tree Construction Prompt Template

Given a temporal question and its classified type {Question_Type}, your task is to decompose it into a structured reasoning tree that contains complete subquestions, indicators, explicit temporal constraints, and time variables.

Map the question to its most relevant decomposition pattern according to its temporal operator and structural characteristics. Generate subquestions that reflect key reasoning steps, construct indicators for entity–relation–time tuples, and specify explicit temporal constraints (e.g., $\ t 2 > \ t 1$ , before(t2, t1)). Each subquestion must be a complete, grammatically correct sentence, and all time variables should preserve monotonic order $( t _ { 1 } \leq t _ { 2 } \leq \ldots \leq t _ { n } )$ ). Use ?x, ?y for unknown entities and t1, t2 for time variables. Represent indicators as Entity1 –[relation]– $\scriptscriptstyle - >$ Entity2 (t). Return the output strictly in the following format:

Subquestions: [sub1, sub2, ...]

Indicators: [edge1, edge2, ...]

Constraints: [constraint1, constraint2, ...]

Time_vars: [t1, t2, ...]

Experience Examples: {In-Context Few-shot}

Q: {Question} A:

The {Question_Type} is obtained from the temporal classification stage. Each retrieved example in {In-Context Few-shot} illustrates how similar questions were decomposed into substeps, indicators, and constraints, guiding the model to align with proven reasoning patterns.

# Seed Selection Prompt Template

Given a subquestion, its reasoning indicator, the topic entities, and optional contextual or temporal hints, your task is to select a concise set of seed entities to initialize retrieval. Prefer specific, high-yield entities that anchor reasoning paths; avoid overly broad categories. If multiple options exist, choose the minimal set that best covers the intended reasoning target.

Experience Examples: {In-Context Few-shot}

Q: {Subquestion}

Think Indicator: {Think_Indicator}

Available Topic Entities: {Available_Entities}

Context Info (optional): {Context_Info}

Time Hints (optional): {Time_Hints}

A:

Seed selection prompts guide the retrieval initialization phase in Section 4.3. The examples retrieved from memory illustrate successful entity anchors for similar subquestions, enabling the model to leverage prior reasoning experience when determining where to begin exploration.

# Experience-Guided Toolkit Selection Prompt Template

```txt
Given a subquestion, its reasoning indicator, temporal type, and contextual information, your task is to select the most suitable temporal toolkit(s) for solving it. Multiple toolkits may be selected if the reasoning requires combined temporal operations.   
You should identify which toolkit(s) align with the temporal operator and reasoning goal of the question. Recommend the most   
appropriate toolkit configuration(s), specifying both parameters and reasoning rationale.   
Available toolkits with descriptions: {Available_toolkits}   
Expected Output (JSON):   
{ "selected_toolkits": [ {"original_name": "ToolkitName", "reasoning": "Reason statement", "priority": 1 "parameters": {"entity1": "...", "entity2": "...", "limit": "Number", ...}}] }   
Experience Examples: {In-Context Few-shot}   
Q: {Subquestion}   
Think Indicator: {Think_Indicator}   
Temporal Type: {Question_Type}   
Seed Info: {Seed_Info}   
Time Hints: {Time_Hints}   
A: 
```

When a similar trace exists in the experience pool, MemoTime retrieves exemplar toolkit configurations as few-shot examples. Otherwise, the model enters a cold-start mode and infers a configuration from the question structure and temporal hints. Each selected toolkit represents a specific reasoning pattern (e.g., event ordering, interval comparison, timeline construction).

# Multiple Toolkits Debate–Vote Prompt Template

Given a subquestion and the outputs generated by multiple temporal toolkits, your task is to evaluate and compare their results to determine which toolkit provides the most accurate, temporally faithful, and semantically consistent answer. Each toolkit represents a distinct reasoning or retrieval strategy, and their outputs may vary in completeness, reliability, and explanatory depth.

The evaluation process compares all toolkit outputs across several complementary dimensions. Each candidate answer is assessed for its relevance to the subquestion, temporal faithfulness to the given time constraints, and path validity within the temporal knowledge graph. The model further considers evidence completeness, semantic consistency, and the adequacy of each toolkit’s configuration and parameters. When multiple valid but conflicting answers arise, preference is given to the one exhibiting stronger temporal grounding, clearer provenance, and greater explanatory transparency. Through this comparative reasoning, the model identifies the toolkit whose output demonstrates the most coherent, temporally consistent, and well-supported reasoning chain.

Expected Output (JSON):

```txt
{
    "winning_toolkit": <integer>,
    "winning_answer": {
        "entity": "<string>", "time": "<YYYY[-MM[-DD]] or Unknown>", "path": ["head", "relation", "tail"], "score": <float>, "reason": "<concise selection rationale>"},
    "evaluation": {
        "criteria Scores": {
            "toolkit_1": {"relevance": <0-1>, "accuracy": <0-1>, "completeness": <0-1}]
            "toolkit_2": {"relevance": <0-1>, "accuracy": <0-1>, "completeness": <0-1}]
        }
    "overall_winner": "<Toolkit number>", "reasoning": "<short comparative analysis across toolkits>"}
}  
Successful Examples: {In-Context Few-shot}  
Q: {Subquestion}  
Toolkit Results: {Collected_Results}  
A: 
```

This prompt is used at the debate–vote stage, where MemoTime aggregates and evaluates the outputs from all executed temporal toolkits. The model receives structured evidence from each toolkit—including reasoning paths, timestamps, parameters, and explanatory notes—and performs comparative reasoning according to the defined evaluation process. The decision prioritizes temporal faithfulness,

evidence completeness, and semantic consistency, ensuring that the final selected answer is both contextually grounded and logically coherent across all candidate toolkits.

# LLM-aware Selection Prompt Template

Given the main question, a chain of thought generated by the LLM that considers all entities, a set of split subquestions, and their retrieved knowledge graph paths, your task is to score the candidate paths and identify the top three that are most likely to contain the correct evidence for the question.

Successful Examples: {In-Context Few-shot}

Q: {Query}

Think Indicator: {Think_Indicator}

Candidate Paths: {Candidate_Paths}

This prompt lets the LLM act as a reasoning critic, ranking candidate paths using patterns learned from previous reasoning examples. The {In-Context Few-shot} samples come from successful path selections in prior tasks, demonstrating how to evaluate relevance, temporal ordering, and completeness.

# Sufficiency Evaluation Prompt Template

Given a question (or subquestion), its candidate answer(s), retrieved evidence paths, and reasoning context, your task is to determine whether the information provided is sufficient and consistent to answer the target question.

You should evaluate in two modes:

• Local (Subquestion) Sufficiency: Assess whether the retrieved evidence and reasoning path for this subquestion are adequate to support the proposed answer. If sufficient, respond {True} and restate the answer; otherwise respond {False}, provide a short explanation, and specify one corrective action from {Decompose, Refine, Retrieval Again}.   
• Global (Full Question) Sufficiency: Assess whether the entire reasoning trajectory, including all subquestions, intermediate answers, and aggregated evidence, is coherent and complete enough to answer the main question. If sufficient, respond {True} and provide the final answer; otherwise respond {False} with a reason and one corrective action from the same set.

Successful Examples: {In-Context Few-shot}

Q: {Question / Subquestion}

Candidate Answer(s): {Answer_Entity / Final_Answer}

Evidence Paths: {Evidence_Paths}

Reasoning Context / Trajectory: {Subquestions_and_Answers}

A:

This unified prompt template supports both local and global sufficiency evaluation. At the local level, it determines whether each reasoning step is self-contained and evidence-supported. At the global level, it verifies that the overall reasoning chain forms a temporally consistent, logically complete path leading to the correct final answer.

# Question Answering Generation Prompt Template

Given the main question, a reasoning chain of thought think indicator, and the complete reasoning global trajectory (all solved split questions, step answers, aggregated reasoning paths, and used toolkits for each split question), your task is to generate the final answer using the provided knowledge paths and your own reasoning. You should ensure that the final answer is logically consistent with the reasoning trajectory and fully supported by the retrieved paths. If multiple candidate entities are equally valid (e.g., simultaneous events or co-occurring facts), list all of them.

Successful Examples: {In-Context Few-shot}

Q: {Main_Question}

Think Indicator: {Think_Indicator}

Global Trajectory Summary: {SolvedQuestions,_Answers,_and_Proofs}

A:

This final prompt synthesizes all previous reasoning and evidence into a natural-language answer. The exemplars demonstrate how previous successful reasoning trajectories were concluded, ensuring the generated answer remains temporally faithful, complete, and well-grounded.
# A Simple Yet Strong Baseline for Long-Term Conversational Memory of LLM Agents

Sizhe Zhou 1 Jiawei Han 1

# Abstract

LLM-based conversational agents still struggle to maintain coherent, personalized interaction over many sessions: fixed context windows limit how much history can be kept in view, and most external memory approaches trade off between coarse retrieval over large chunks and fine-grained but fragmented views of the dialogue. Motivated by neo-Davidsonian event semantics, we propose an event-centric alternative that represents conversational history as short, event-like propositions which bundle together participants, temporal cues, and minimal local context, rather than as independent relation triples or opaque summaries. In contrast to work that aggressively compresses or forgets past content, our design aims to preserve information in a non-compressive form and make it more accessible, rather than more lossy. Concretely, we instruct an LLM to decompose each session into enriched elementary discourse units (EDUs)—self-contained statements with normalized entities and source turn attributions—and organize sessions, EDUs, and their arguments in a heterogeneous graph that supports associative recall. On top of this representation we build two simple retrieval-based variants that use dense similarity search and LLM filtering, with an optional graph-based propagation step to connect and aggregate evidence across related EDUs. Experiments on the LoCoMo and LongMemEvalS benchmarks show that these event-centric memories match or surpass strong baselines, while operating with much shorter QA contexts. Our results suggest that structurally simple, eventlevel memory provides a principled and practical foundation for long-horizon conversational agents. Our code and data will be released at https://github.com/KevinSRR/EMem.

# 1. Introduction

Large Language Models (LLMs) have demonstrated impressive conversational abilities, but their fixed context windows severely limit long-term coherence and personalization in extended interactions. Even in so–called long-context variants, performance can degrade sharply (Liu et al., 2024) and the LLMs can struggle to faithfully recall information that is many sessions old (Maharana et al., 2024; Wu et al., 2025a). Meanwhile, naively concatenating entire multi-session histories into the prompt is computationally expensive and still bounded by a finite context window.

A natural solution is to maintain an external store of dialogue history and retrieve relevant content. However, retrieval at the granularity of entire sessions or whole rounds often fails to recall fine-grained details while retrieval of turns fails to recover the larger context. Other approaches compress conversation histories into summaries or distilled “facts” before indexing (Chhikara et al., 2025; Kim et al., 2025; Huang et al., 2025; Fang et al., 2025), which improves efficiency but inevitably discards information. For longterm conversational memory—where queries may refer back to seemingly minor details many sessions ago—such lossy compression is risky.

To address these issues, recent research has drawn on cognitive science and structured knowledge representations. Cognitively-inspired architectures (Nan et al., 2025; Xu et al., 2025; Fang et al., 2025; Li et al., 2025) organize memory into episodic and semantic stores, employ multistage consolidation, or treat memory as a managed system resource. In parallel, graph-based systems (Gutierrez et al.´ , 2024; Gutierrez et al.´ , 2025; Rasmussen et al., 2025; Wu et al., 2025b; Wang et al., 2025; Chhikara et al., 2025) represent memories as networks of entities, relations, and text chunks, and perform graph search to support associative recall. These approaches highlight the importance of structure, yet most use entity–relation triples or coarse chunks as basic units, which fragments or coarsens the original discourse: a single utterance may be split into disconnected triples, while large chunks mix unrelated information.

Our starting point is the observation from neo-Davidsonian event semantics that the meaning of a sentence is often best

represented as an event with multiple arguments, rather than as a collection of independent binary relations (Davidson, 1967; Parsons, 1990). Inspired by this view, we model long-term conversational memory at the level of event-like elementary discourse units (EDUs). Instead of taking raw clauses, we instruct an LLM to rewrite each session into a set of enriched EDUs. Unlike traditional EDUs from discourse parsing, our EDUs are short event-style statements that may span multiple utterances, enrich the text with normalized entities and minimal context information, and sometimes include lightly inferred information so that each EDU is as self-contained and precise as possible.1 Together, these EDUs recover the information conveyed by the original session while making each atomic event explicit and maximally self-complete. We also attach turn-level source attributions to support downstream agentic behaviors to connect to larger and nuanced conversation contexts.

We then organize all sessions, EDUs, and extracted arguments from EDUs into an event-centric memory graph to support associative recall and dense-sparse integration that are difficult to realize with flat retrieval over independent chunks. At query time, conversational questions often refer to unnamed or generic entities (“my pet”, “that conference”), so we perform entity and concept mention detection on the query and retrieve both EDUs and argument nodes in embedding space, using mentions as anchors into the graph rather than relying solely on exact entity strings. Because similarity scores and fixed thresholds alone are brittle in this setting, we employ lightweight LLM-based relevance filters over both EDUs and arguments, designed to favor recall, and use the resulting scores to define query-specific seed weights on the graph. The full model, EMem-G, applies a Personalized PageRank step from these seeds to propagate relevance over EDU and argument nodes and select a small set of graph-consistent EDUs for augmenting the QA model, thereby capturing indirect associations across sessions. The lightweight variant, EMem, omits the graph and argument components and instead uses dense retrieval over EDUs followed by the same recall-oriented LLM filter, providing an efficient, conceptually simple baseline that still benefits from the event-centric representation. In summary, our contributions are:

1. We introduce an event-centric conversational memory representation based on EDUs and a heterogeneous graph linking sessions, EDUs, and argument nodes,

1For example, from a dialogue about a trip to Tokyo we derive EDUs such as “Bob traveled to Tokyo for five days to attend the Global AI Innovation Symposium 2024 in March $2 0 2 4 ^ { \circ }$ and “Bob presented his team’s multimodal learning work at the Global AI Innovation Symposium 2024.”. Here the entity mentions are normalized, and the timestamp is inferred based on the session timestamp. Note that the “event” concept also covers the expression of triple-style facts or knowledge.

grounded in neo-Davidsonian event semantics.

2. We propose two retrieval variants: EMem-G, which combines dense retrieval, LLM-based relevance filtering, and graph propagation via PPR; and EMem, a lightweight dense-retrieval+filter baseline that avoids graph computation while retaining strong performance.   
3. We empirically demonstrate that these simple designs form competitive baselines on LoCoMo and LongMemEvalS.

# 2. Related Work

# 2.1. Memory Architectures for LLM Agents

Cognitively-inspired memory systems such as Nemori (Nan et al., 2025), LightMem (Fang et al., 2025), LiCoMemory (Huang et al., 2025), and A-Mem (Xu et al., 2025) seek to emulate human memory processes by organizing interactions into episodes, applying multi-stage consolidation, and dynamically restructuring memories. Many of these frameworks perform explicit compression or abstraction— through clustering, summarization, or note-taking—to maintain a compact store, which improves efficiency but can drop fine-grained details. PREMem (Kim et al., 2025) moves part of the reasoning to the write phase, storing enriched memory fragments that encode inferred relations across sessions. System-level frameworks like MemOS (Li et al., 2025) and Mem0 (Chhikara et al., 2025) treat memory as an operating system primitive or services layer, focusing on scalable storage and retrieval and often combining summarization with fact extraction. In contrast, our approach intentionally avoids lossy compression: we aim at memory representations that preserve all the original information and we rely on event-centric structure plus inference-time efforts to manage relevance and satisfy information seeking requests.

# 2.2. Structured and Graph-Based Memory

Graph-based approaches have emerged as a powerful way to endow LLMs with associative recall. HippoRAG and HippoRAG 2 (Gutierrez et al. ´ , 2024; Gutierrez et al. ´ , 2025) build graphs over entities and passages and use personalized PageRank to retrieve multi-hop evidence. Zep (Rasmussen et al., 2025) and Mem0’s graph extension (Chhikara et al., 2025) maintain temporal knowledge graphs as a memory layer for agents, while ComoRAG (Wang et al., 2025) organizes long narratives into cognitively-inspired memory structures for iterative RAG. SGMem (Wu et al., 2025b) represents dialogue as sentence-level graphs, connecting sentences within and across sessions. Our work is closest in spirit to HippoRAG 2 and SGMem but differs in two key aspects: (i) we adopt an event-centric representation rooted in neo-Davidsonian semantics (Davidson, 1967; Parsons, 1990), where enriched EDUs—rather than triples or raw

![](images/24ddec5253f8aa0845f3f04920b67252d85a6e22e0e9b14b92bdbeaa4e93fe1f.jpg)  
Figure 1. Overview of our memory framework. Offline, conversations are decomposed into enriched EDUs, arguments, and an event graph; online, EMem-G retrieves EDUs/arguments and propagates relevance over the graph, while EMem uses only EDU retrieval and filtering to efficiently supply a compact memory context for QA.

sentences—are the primary memory units; and (ii) we couple graph-based retrieval with LLM-based recall-oriented filtering over both EDUs and argument nodes, which is particularly important for the implicit, unnamed references prevalent in less knowledge-intensive conversations.

# 3. Methodology

Figure 1 illustrates the overall offline indexing and online retrieval pipelines.

# 3.1. Problem Setting

A conversation consists of sessions $\boldsymbol { S } = \{ s _ { 1 } , \ldots , s _ { T } \}$ ordered by timestamps $\tau ( s )$ . Each session $s$ is a sequence of turns $s = \left( \left( \mathrm { S P K } _ { 1 } , u _ { 1 } \right) , \dots , \left( \mathrm { S P K } _ { L _ { s } } , u _ { L _ { s } } \right) \right)$ , where $u _ { \ell }$ is the utterance from speaker $\operatorname { S P K } _ { \ell }$ . At query time, the agent receives a natural language question $q$ and answer it conditioned on the entire conversation history. We assume access to an embedding encoder $h ( \cdot )$ that maps any text $x$ to an embedding $h ( x ) \in  { \mathbb { R } } ^ { d }$ , and a powerful QA model $f _ { \mathrm { Q A } }$ that takes (q, memory) as input and generates an answer.

# 3.2. Event-Centric Memory Graph

EDUs vs. relation triples. To make the notion of EDUs concrete, consider a short multi-turn exchange where the user describes a trip: from several utterances the extractor may produce an EDU such as “Bob spent five days in Tokyo in March 2024 to attend the Global AI Innovation Sympo-

sium 2024 at Tokyo University.” This EDU is mildly abstractive but self-contained: it unifies details mentioned across different turns and normalizes entities into their canonical forms. Here, “March 2024” is inferred from the session timestamp and Bob’s utterance; and the mentions “Global AI Innovation Symposium $2 0 2 4 ^ { \circ }$ and “Tokyo University” are expanded to full forms and normalized based on session context. As a memory item, this single EDU captures a complete event—who did what, where, when, and for what purpose.

In a triple-based knowledge graph, the same content would typically be decomposed into multiple relation triples, e.g., (Bob, attend, Global AI Innovation Symposium 2024), (Bob, stay in, Tokyo), (Bob, stay duration, five days), (Symposium 2024, held at, Tokyo University),

(Symposium 2024, time, March 2024), which are stored as separate edges that may be scattered across the graph. While such triples are useful for schema-driven reasoning, they fragment the original discourse: a retrieval step must find and recombine several triples to reconstruct the event, and fine-grained temporal or participant constraints can be lost or inconsistently represented. In contrast, our enriched EDUs follow the neo-Davidsonian intuition of treating the event as a single unit with multiple arguments; each EDU node in the memory graph is therefore a self-complete, human-readable memory cell that preserves the local coherence of the original conversation while still being small enough to enable fine-grained retrieval.

EDU extraction. For each session $s$ , we invoke an LLMbased extractor $g _ { \mathrm { E D U } }$ with a single in-context exemplar that describes the session (including the timestamp and all Es = {e(s)1 , . speakeroutputs $E _ { s } = \{ e _ { 1 } ^ { ( s ) } , \ldots , e _ { N _ { s } } ^ { ( s ) } \}$ a list of EDUs. Th, where each EDU $e$ extractoris a short natural language description plus metadata:

$$
e = \left(\operatorname {t e x t} (e), \operatorname {s r c} (e), \tau (e)\right).
$$

Here, $\operatorname { s r c } ( e )$ is the set of turn indices in $s$ that support $e$ , and $\tau ( e )$ is a timestamp derived from the session date (if available). For long, structured assistant responses (e.g., enumerated suggestions) from LongMemEvalS, we allow the extractor to output structured chunks: multi-sentence blocks, each accompanied by a 2–3 sentence summary that states the user request addressed, the information categories covered, and salient entities. We treat the summary as text(e) for indexing and retrieval, and reserve the full chunk for the QA stage. For completeness, our dataset-specific treatment of long assistant responses in LongMemEvalS is detailed in Appendix A. Across all sessions, the global EDU set is $\textstyle { \mathcal { E } } = \bigcup _ { s } E _ { s }$ .

Event-argument extraction. For each EDU $e \in { \mathcal { E } }$ , we invoke a second LLM $g _ { \mathrm { A R G } }$ that treats $e$ as a single event and returns an event type $t ( e )$ and a set of role–argument pairs $\{ ( r _ { k } , a _ { k } ) \} _ { k = 1 } ^ { K _ { e } }$ }Kek=1 We collect all unique argument strings into a . global argument set $\mathcal { A }$ . Each argument $a \in { \mathcal { A } }$ is associated with a node-level embedding $h _ { \mathrm { a r g } } ( a )$ . We do not enforce a fixed ontology of roles; the usage of $r _ { k }$ is not explored in our framework, while arguments themselves become nodes in the memory graph.

Graph construction. We construct a heterogeneous graph $G = ( V , E )$ with three node types:

• Session nodes $v _ { s }$ for $s \in S$ .   
• EDU nodes $v _ { e }$ for $e \in { \mathcal { E } }$ .   
• Argument nodes $v _ { a }$ for $a \in { \mathcal { A } }$ .

Edges are defined as:

$$
E _ {\text {s e s s - e d u}} = \left\{\left(v _ {s}, v _ {e}\right) \mid e \in E _ {s} \right\},
$$

$E _ { \mathrm { e d u - a r g } } = \{ ( v _ { e } , v _ { a } ) \mid a { \mathrm { ~ i s ~ a n ~ a r g u m e n t ~ o f ~ } } e \} ,$ $e \}$

$$
E _ {\text {s y n}} = \left\{\left(v _ {a}, v _ {a ^ {\prime}}\right) \mid \operatorname {s i m} \left(a, a ^ {\prime}\right) \geq \delta \right\}.
$$

Here, $\sin ( a , a ^ { \prime } )$ is cosine similarity between $h _ { \mathrm { a r g } } ( a )$ and $h _ { \mathrm { a r g } } ( a ^ { \prime } )$ , and we cap the number of synonym neighbors per $a$ (e.g., at 100). The final node set is $V = \{ v _ { s } \} \cup \{ v _ { e } \} \cup \{ v _ { a } \}$ and edge set $E = E _ { \mathrm { s e s s - e d u } } \cup E _ { \mathrm { e d u - a r g } } \cup E _ { \mathrm { s y n } }$ .

For later retrieval we cache embeddings for EDU texts, $h _ { \mathrm { e d u } } ( e ) ~ = ~ h ( \mathrm { t e x t } ( e ) )$ . Graph construction is performed offline as new sessions arrive.

# 3.3. Graph-Based Retrieval and QA (EMem-G)

Given a query $q$ , EMem-G performs the following steps (corresponding to the middle row in Figure 1).

Dense retrieval of EDUs and arguments. We encode the query with the same encoder: $z _ { q } = h ( q )$ . We first retrieve the top- $K _ { e }$ EDUs by cosine similarity between $z _ { q }$ and $h _ { \mathrm { e d u } } ( e )$ , obtaining a candidate set $C ^ { \mathrm { e d u } } ( q )$ . In parallel, we run an LLM-based mention detector on $q$ to extract a set of surface mentions $M ( q ) = \{ m _ { 1 } , . . . , m _ { M } \}$ corresponding to entities, noun phrases, and salient concepts. Each mention is embedded as $h ( m )$ , and we retrieve top- $K _ { a }$ argument nodes for each mention using similarity between $h ( m )$ and $h _ { \mathrm { a r g } } ( a )$ , forming a candidate argument set $C ^ { \mathrm { a r g } } ( q )$ .

Recall-oriented LLM filtering. Embedding similarity alone is brittle when mentions are generic (“pet”, “that trip”) or when arguments are highly specific. We therefore apply an LLM-based relevance filter to both candidate sets. For EDUs, we prompt a LLM once with the query $q$ and the list $\{ \mathrm { t e x t } ( e ) \mid e \in C ^ { \mathrm { e d u } } ( q ) \}$ , asking it to select the EDUs that are relevant to answering $q$ . Its discrete selection induces a binary indicator $f _ { \mathrm { E D U } } ( q , e ) \in \{ 0 , 1 \}$ , and we keep EDUs with $f _ { \mathrm { E D U } } ( q , e ) = 1$ . Similarly, for arguments we prompt the LLM with $q$ and the list $\{ a \mid a \in C ^ { \arg } ( q ) \}$ in minimal context, obtaining a binary indicator $f _ { \mathrm { A R G } } ( q , a ) \in$ $\{ 0 , 1 \}$ . The filtered sets are

$$
\tilde {C} ^ {\mathrm {e d u}} (q) = \left\{e \in C ^ {\mathrm {e d u}} (q) \mid f _ {\mathrm {E D U}} (q, e) = 1 \right\},
$$

$$
\tilde {C} ^ {\arg} (q) = \left\{a \in C ^ {\arg} (q) \mid f _ {\mathrm {A R G}} (q, a) = 1 \right\}.
$$

This design differs from precision-oriented filters (e.g., HippoRAG 2’s filter (Gutierrez et al. ´ , 2025)): we intentionally keep borderline candidates by biasing the LLM toward recall, and rely on subsequent graph propagation and final QA to down-weight spurious ones.

Seed initialization. We initialize a nonnegative weight function $s : V \to { \mathbb { R } } _ { \geq 0 }$ over graph nodes using embedding similarities. For EDU nodes we set

$$
s (v _ {e}) = \mathrm {s i m} (z _ {q}, h _ {\mathrm {e d u}} (e)), \qquad v _ {e} \in \tilde {C} ^ {\mathrm {e d u}} (q),
$$

and for argument nodes

$$
s (v _ {a}) = \operatorname {s i m} (h (m), h _ {\arg} (a)), \quad v _ {a} \in \tilde {C} ^ {\arg} (q),
$$

with $s ( v ) = 0$ for all remaining nodes and $m \in M ( q )$ is the corresponding mention that retrieves this candidate argument $v _ { a }$ . If more than $K$ argument nodes receive nonzero scores, we keep only the $K$ highest-scoring ones and set the rest to zero, so that propagation remains focused and the initial mass is not diluted over many weakly related arguments. The resulting vector s is used as the personalization (seed) vector for personalized PageRank.

Personalized PageRank. Let $T$ be the column-stochastic transition matrix derived from $G$ (e.g., uniform over neighbors). We compute a Personalized PageRank vector

$$
\pi = \operatorname {P P R} (G, \mathbf {s}) \triangleq (1 - \alpha) \mathbf {s} + \alpha T ^ {\top} \pi ,
$$

with a fixed damping factor $\alpha \in ( 0 , 1 )$ using a small number of power iterations. The resulting $\pi ( v )$ scores reflect how strongly each node is connected to the query seeds under random walks that repeatedly return to s.

Selecting EDUs and QA. We restrict $\pi$ to EDU nodes and select the top- $K$ EDUs, $R ( q ) = \mathrm { T o p K } \{ ( e , \pi ( v _ { e } ) ) :$ $e \in \mathcal { E } \}$ . For EDUs corresponding to structured chunks, we replace text(e) by the full chunk content before QA. We assemble a memory context by concatenating the selected EDUs along with their source session timestamps and source-turns’ speaker names. Finally, the QA model produces the answer:

$$
\hat {y} = f _ {\mathrm {Q A}} \left(q, \left\{\left(\operatorname {t e x t} (e), \operatorname {s r c} (e), \tau (e)\right): e \in R (q) \right\}\right),
$$

using a prompt that asks the model to first reason using the retrieved memories and then output a concise answer in zero-shot manner (Wei et al., 2022).

# 3.4. Lightweight Retrieval (EMem)

EMem shares the same event graph and EDU extraction as EMem-G but removes graph propagation and argumentlevel retrieval. Given a query $q$ , we compute $z _ { q } = h ( q )$ and retrieve the top- $K _ { e }$ EDUs by similarity to $h _ { \mathrm { e d u } } ( e )$ , obtaining $C ^ { \mathrm { e d u } } ( q )$ . We then apply the same recall-oriented LLM filter $f _ { \mathrm { E D U } }$ as in Section 3.3: the LLM is prompted once with $q$ and the list $\{ \mathrm { t e x t } ( e ) : e \in C ^ { \mathrm { e d u } } ( q ) \}$ and selects a subset of relevant EDUs, inducing binary decisions $f _ { \mathrm { E D U } } ( q , e ) \in$ $\{ 0 , 1 \}$ . The retained set is

$$
R _ {\text {l i t e}} (q) = \left\{e \in C ^ {\mathrm {e d u}} (q) \mid f _ {\mathrm {E D U}} (q, e) = 1 \right\},
$$

so the number of EDUs passed to QA is determined adaptively by the filter rather than fixed a priori. The QA model then answers as

$$
\hat {y} = f _ {\mathrm {Q A}} \big (q, \{(\text {t e x t} (e), \operatorname {s r c} (e), \tau (e)): e \in R _ {\text {l i t e}} (q) \} \big).
$$

Because EDUs are short, self-contained, and enriched with canonical entities and time information, dense retrieval over them already yields high-quality candidates. Retrieving a relatively large pool and then applying a recall-biased LLM filter captures most relevant memories without requiring personalized PageRank, while adaptively reducing the final context length. Empirically, this lightweight EMem variant is competitive with EMem-G and sometimes outperforms more sophisticated designs, making it a practical reference point for future work.

# 4. Experiment

# 4.1. Experimental Setup

Table 1. Category distributions of the LoCoMo (Maharana et al., 2024) and LongMemEvalS (Wu et al., 2025a) datasets.   

<table><tr><td colspan="2">LoCoMo</td><td colspan="2">LongMemEvals</td></tr><tr><td>Category</td><td>Count</td><td>Category</td><td>Count</td></tr><tr><td>Multi-Hop</td><td>278</td><td>knowledge-update</td><td>72</td></tr><tr><td>Temporal</td><td>320</td><td>multi-session</td><td>121</td></tr><tr><td>Open Domain</td><td>93</td><td>single-session-assistant</td><td>56</td></tr><tr><td>Single-Hop</td><td>829</td><td>single-session-preference</td><td>30</td></tr><tr><td>-</td><td>-</td><td>single-session-user</td><td>64</td></tr><tr><td>-</td><td>-</td><td>temporal-reasoning</td><td>127</td></tr><tr><td>Total</td><td>1,520</td><td>Total</td><td>470</td></tr></table>

We evaluated EMem and EMem-G on two widely used longterm conversational QA benchmark dataset, LoCoMo2 (Maharana et al., 2024) and LongMemEvalS (Wu et al., 2025a). LoCoMo contains 10 multi-session dialogues between two speakers with around 24K tokens on average while LongMemEval consists of 500 multi-session dialogues between user and assistant with around 105K average tokens. We exclude the adversarial type of questions following the prior research. The distribution of the evaluated questions are shown in Table 13.

We follow Nemori (Nan et al., 2025) to set up the evaluation framework4, leveraging LLM-judge5 to score the final QA accuracy on two benchmark datasets, and additionally report the F1 and BLEU-1 scores for QA on the LoCoMo dataset. The LLM-judge runs three times with mean and standard deviation reported.

We also follow Nemori (Nan et al., 2025) baseline setup to have: a full-context LLM that receives the entire dialogue history; a retrieval-augmented model (RAG-4096) that splits the dialogue into 4096-token chunks and uses dense retrieval to select context; and three memory-augmented systems, namely LangMem (LangChain AI, 2025) with a hierarchical memory organization, Zep (Rasmussen et al., 2025) based on temporal knowledge graphs, and Mem0 (Chhikara et al.,

Table 2. Performance on LoCoMo dataset (Maharana et al., 2024) categorized by question type. Bold indicates the best performance. Underline indicates the second best performance. The baseline performance comes from Nan et al., 2025.   

<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="3">Temporal Reasoning</td><td colspan="3">Open Domain</td><td colspan="3">Multi-Hop</td><td colspan="3">Single-Hop</td><td colspan="3">Overall</td></tr><tr><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td></tr><tr><td rowspan="8">gpt-4o-mini</td><td>FullContext</td><td>0.562 ± 0.004</td><td>0.441</td><td>0.361</td><td>0.486 ± 0.005</td><td>0.245</td><td>0.172</td><td>0.668 ± 0.003</td><td>0.354</td><td>0.261</td><td>0.830 ± 0.001</td><td>0.531</td><td>0.447</td><td>0.723 ± 0.000</td><td>0.462</td><td>0.378</td></tr><tr><td>LangMem</td><td>0.249 ± 0.003</td><td>0.319</td><td>0.262</td><td>0.476 ± 0.005</td><td>0.294</td><td>0.235</td><td>0.524 ± 0.003</td><td>0.335</td><td>0.239</td><td>0.614 ± 0.002</td><td>0.388</td><td>0.331</td><td>0.513 ± 0.003</td><td>0.358</td><td>0.294</td></tr><tr><td>Mem0</td><td>0.504 ± 0.001</td><td>0.444</td><td>0.376</td><td>0.406 ± 0.000</td><td>0.271</td><td>0.194</td><td>0.603 ± 0.000</td><td>0.343</td><td>0.252</td><td>0.681 ± 0.000</td><td>0.444</td><td>0.377</td><td>0.613 ± 0.000</td><td>0.415</td><td>0.342</td></tr><tr><td>RAG</td><td>0.237 ± 0.000</td><td>0.195</td><td>0.157</td><td>0.326 ± 0.005</td><td>0.190</td><td>0.135</td><td>0.313 ± 0.003</td><td>0.186</td><td>0.117</td><td>0.320 ± 0.001</td><td>0.222</td><td>0.186</td><td>0.302 ± 0.000</td><td>0.208</td><td>0.164</td></tr><tr><td>Zep</td><td>0.589 ± 0.003</td><td>0.448</td><td>0.381</td><td>0.396 ± 0.000</td><td>0.229</td><td>0.157</td><td>0.505 ± 0.007</td><td>0.275</td><td>0.193</td><td>0.632 ± 0.001</td><td>0.397</td><td>0.337</td><td>0.585 ± 0.001</td><td>0.375</td><td>0.309</td></tr><tr><td>Nemori</td><td>0.710 ± 0.000</td><td>0.567</td><td>0.466</td><td>0.448 ± 0.005</td><td>0.208</td><td>0.151</td><td>0.653 ± 0.002</td><td>0.365</td><td>0.256</td><td>0.821 ± 0.002</td><td>0.544</td><td>0.432</td><td>0.744 ± 0.001</td><td>0.495</td><td>0.385</td></tr><tr><td>EMem-G</td><td>0.760 ± 0.003</td><td>0.581</td><td>0.468</td><td>0.573 ± 0.013</td><td>0.242</td><td>0.199</td><td>0.747 ± 0.006</td><td>0.406</td><td>0.305</td><td>0.823 ± 0.001</td><td>0.504</td><td>0.422</td><td>0.780 ± 0.000</td><td>0.487</td><td>0.397</td></tr><tr><td>EMem</td><td>0.771 ± 0.004</td><td>0.574</td><td>0.461</td><td>0.602 ± 0.009</td><td>0.285</td><td>0.237</td><td>0.702 ± 0.004</td><td>0.406</td><td>0.307</td><td>0.830 ± 0.002</td><td>0.497</td><td>0.414</td><td>0.780 ± 0.001</td><td>0.483</td><td>0.393</td></tr><tr><td rowspan="8">gpt-4,1-mini</td><td>FullContext</td><td>0.742 ± 0.004</td><td>0.475</td><td>0.400</td><td>0.566 ± 0.010</td><td>0.284</td><td>0.222</td><td>0.772 ± 0.003</td><td>0.442</td><td>0.337</td><td>0.869 ± 0.002</td><td>0.614</td><td>0.534</td><td>0.806 ± 0.001</td><td>0.533</td><td>0.450</td></tr><tr><td>LangMem</td><td>0.508 ± 0.003</td><td>0.485</td><td>0.409</td><td>0.590 ± 0.005</td><td>0.328</td><td>0.264</td><td>0.710 ± 0.002</td><td>0.415</td><td>0.325</td><td>0.845 ± 0.001</td><td>0.510</td><td>0.436</td><td>0.734 ± 0.001</td><td>0.476</td><td>0.400</td></tr><tr><td>Mem0</td><td>0.569 ± 0.001</td><td>0.392</td><td>0.332</td><td>0.479 ± 0.000</td><td>0.237</td><td>0.177</td><td>0.682 ± 0.003</td><td>0.401</td><td>0.303</td><td>0.714 ± 0.001</td><td>0.486</td><td>0.420</td><td>0.663 ± 0.000</td><td>0.435</td><td>0.365</td></tr><tr><td>RAG</td><td>0.274 ± 0.000</td><td>0.223</td><td>0.191</td><td>0.288 ± 0.005</td><td>0.179</td><td>0.139</td><td>0.317 ± 0.003</td><td>0.201</td><td>0.128</td><td>0.359 ± 0.002</td><td>0.258</td><td>0.220</td><td>0.329 ± 0.002</td><td>0.235</td><td>0.192</td></tr><tr><td>Zep</td><td>0.602 ± 0.001</td><td>0.239</td><td>0.200</td><td>0.438 ± 0.000</td><td>0.242</td><td>0.193</td><td>0.537 ± 0.003</td><td>0.305</td><td>0.204</td><td>0.669 ± 0.001</td><td>0.455</td><td>0.400</td><td>0.616 ± 0.000</td><td>0.369</td><td>0.309</td></tr><tr><td>Nemori</td><td>0.776 ± 0.003</td><td>0.577</td><td>0.502</td><td>0.510 ± 0.009</td><td>0.258</td><td>0.193</td><td>0.751 ± 0.002</td><td>0.417</td><td>0.319</td><td>0.849 ± 0.002</td><td>0.588</td><td>0.515</td><td>0.794 ± 0.001</td><td>0.534</td><td>0.456</td></tr><tr><td>EMem-G</td><td>0.808 ± 0.001</td><td>0.513</td><td>0.406</td><td>0.717 ± 0.005</td><td>0.291</td><td>0.253</td><td>0.796 ± 0.002</td><td>0.376</td><td>0.308</td><td>0.905 ± 0.001</td><td>0.510</td><td>0.432</td><td>0.853 ± 0.000</td><td>0.473</td><td>0.393</td></tr><tr><td>EMem</td><td>0.800 ± 0.003</td><td>0.491</td><td>0.389</td><td>0.652 ± 0.010</td><td>0.270</td><td>0.237</td><td>0.790 ± 0.002</td><td>0.350</td><td>0.273</td><td>0.897 ± 0.001</td><td>0.509</td><td>0.430</td><td>0.842 ± 0.001</td><td>0.461</td><td>0.381</td></tr></table>

Table 3. Performance on LongMemEvalS dataset (Wu et al., 2025a) across different question types. LLM-judged QA accuracy is reported. Bold indicates the best performance.   

<table><tr><td></td><td>Question Type</td><td>Full-context (101K tokens)</td><td>Nemori (3.7-4.8K tokens)</td><td>EMem-G (1.0K-3.6K tokens)</td><td>EMem (0.6K-2.5K tokens)</td></tr><tr><td rowspan="7">gpt-40-mini</td><td>single-session-preference</td><td>6.7%</td><td>46.7%</td><td>32.2%</td><td>32.2%</td></tr><tr><td>single-session-assistant</td><td>89.3%</td><td>83.9%</td><td>87.5%</td><td>82.1%</td></tr><tr><td>temporal-reasoning</td><td>42.1%</td><td>61.7%</td><td>74.8%</td><td>69.8%</td></tr><tr><td>multi-session</td><td>38.3%</td><td>51.1%</td><td>73.6%</td><td>78.0%</td></tr><tr><td>knowledge-update</td><td>78.2%</td><td>61.5%</td><td>94.4%</td><td>87.5%</td></tr><tr><td>single-session-user</td><td>78.6%</td><td>88.6%</td><td>87.0%</td><td>86.5%</td></tr><tr><td>Average</td><td>55.0%</td><td>64.2%</td><td>77.9%</td><td>76.0%</td></tr><tr><td rowspan="7">gpt-4.1-mini</td><td>single-session-preference</td><td>16.7%</td><td>86.7%</td><td>50%</td><td>46.7%</td></tr><tr><td>single-session-assistant</td><td>98.2%</td><td>92.9%</td><td>87.5%</td><td>82.1%</td></tr><tr><td>temporal-reasoning</td><td>60.2%</td><td>72.2%</td><td>83.7%</td><td>80.6%</td></tr><tr><td>multi-session</td><td>51.1%</td><td>55.6%</td><td>82.6%</td><td>82.1%</td></tr><tr><td>knowledge-update</td><td>76.9%</td><td>79.5%</td><td>94.4%</td><td>95.4%</td></tr><tr><td>single-session-user</td><td>85.7%</td><td>90.0%</td><td>94.8%</td><td>93.8%</td></tr><tr><td>Average</td><td>65.6%</td><td>74.6%</td><td>84.9%</td><td>83.0%</td></tr></table>

2025) which maintains a store of extracted personalized memories.

For EMem and EMem-G, we utilize the OpenAI text-embedding-3-small embedding model across all of our experiments, and use OpenAI gpt-4o-mini and gpt-4.1-mini as our backbone LLM respectively. In event graph construction, we set the similarity threshold $\delta$ for synonym edges to be 0.9. The number of initially retrieved EDUs top- $K _ { e }$ is set to be 30 and the number of initially retrieved arguments top- $K _ { a }$ is set to 10. The upper bound of the initialized argument nodes is 30. The upper bound of the final retrieved EDUs top- $K$ is 10. For PPR, we followed HippoRAG 2 (Gutierrez et al. ´ , 2025)’s default parameters.

# 4.2. Main Results

Across both datasets and backbones, our method substantially improves over the memory baselines, while using

comparable or fewer tokens (Tables 2 and 3). On Lo-CoMo, EMem and EMem-G consistently outperform in LLM-judged accuracy. With gpt-4o-mini, the overall LLM score improves from 0.744 for Nemori to 0.780 for both EMem and EMem-G. The gains concentrate on the categories that truly require long-term and structured reasoning: temporal reasoning, open-domain, multi-hop questions. Single-hop questions are already easy for strong baselines, and all three top systems are effectively saturated there. The same pattern holds for gpt-4.1-mini: EMem-G reaches 0.853 overall vs. 0.806 for full-context and 0.794 for Nemori, indicating that memory retrieval can efficiently surpass naive full-context prompting even when the backbone LLM is strong. On LoCoMo, this accuracy is achieved with compact QA contexts: EMem passes only 509–1039 tokens to the backbone (average 738.2), and EMem-G 924–1062 tokens (average 987.8), which is substantially below Nemori’s reported 2,745 tokens and far below the 23,653-token full-context baseline (Nan et al., 2025).

Table 4. Statistics of graphs constructed by EMem-G. All metrics except the first row are averaged per conversation. For LongMemEvalS, “Avg Speaker EDU or Chunk Dist/Conv” and “Avg Speaker EDU or Chunk Len (words) Dist” show the averaged count and averaged word length respectively in the format of “User EDUs:Assistant EDUs:Assistant Chunks”. For LoCoMo, the two speakers of each conversation are ordered by their number of EDUs. The results are shown in the format of “max-EDUs speaker:min-EDUs speaker”.   

<table><tr><td></td><td>LongMemEvals (GPT-4o-mini)</td><td>LongMemEvals (GPT-4.1-mini)</td><td>LoCoMo (GPT-4o-mini)</td><td>LoCoMo (GPT-4.1-mini)</td></tr><tr><td>Number of Conversations</td><td>470</td><td>470</td><td>10</td><td>10</td></tr><tr><td>Avg Sessions/Conv</td><td>47.7</td><td>47.7</td><td>27.2</td><td>27.2</td></tr><tr><td>Avg Session Length (words)</td><td>1,644.4</td><td>1,644.4</td><td>536.9</td><td>536.9</td></tr><tr><td>Avg Turns/Session</td><td>10.3</td><td>10.3</td><td>21.6</td><td>21.6</td></tr><tr><td>Avg EDU Nodes/Conv</td><td>861.1</td><td>1391.5</td><td>552.8</td><td>570.2</td></tr><tr><td>Avg Arg Nodes/Conv</td><td>2,780.9</td><td>3,786.2</td><td>1,144.7</td><td>1,184.9</td></tr><tr><td>Avg Total Nodes/Conv</td><td>3,689.7</td><td>5,225.4</td><td>1,724.7</td><td>1,782.3</td></tr><tr><td>Avg Session Node Degree</td><td>18.1</td><td>29.2</td><td>20.3</td><td>21.0</td></tr><tr><td>Avg EDU Node Degree</td><td>5.5</td><td>4.7</td><td>4.9</td><td>4.7</td></tr><tr><td>Avg Arg Node Degree</td><td>1.5</td><td>1.4</td><td>1.9</td><td>1.8</td></tr><tr><td>Avg Session-EDU Edges/Conv</td><td>861.3</td><td>1391.8</td><td>552.9</td><td>570.3</td></tr><tr><td>Avg EDU-Arg Edges/Conv</td><td>3,908.3</td><td>5,106.4</td><td>2,149.9</td><td>2,100.3</td></tr><tr><td>Avg Synonym Edges/Conv</td><td>105.8</td><td>189.0</td><td>24.5</td><td>38.7</td></tr><tr><td>Avg Total Edges/Conv</td><td>4,875.4</td><td>6,687.1</td><td>2,727.3</td><td>2,709.3</td></tr><tr><td>Avg User EDUs/Conv</td><td>314.6</td><td>469.2</td><td>-</td><td>-</td></tr><tr><td>Avg Asst EDUs/Conv</td><td>421.6</td><td>725.6</td><td>-</td><td>-</td></tr><tr><td>Avg AsstChunks/Conv</td><td>125.9</td><td>197.7</td><td>-</td><td>-</td></tr><tr><td>Avg Speaker EDU or Chunk Dist/Conv</td><td>314.6:421.6:125.9</td><td>469.2:725.6:197.7</td><td>98.1:16.7</td><td>105.9:18.5</td></tr><tr><td>Avg Speaker EDU or Chunk Len (words) Dist</td><td>23.2:21.3:193.8</td><td>20.3:21.1:121.3</td><td>15.0:15.4</td><td>16.9:17.9</td></tr><tr><td>Avg Asst Chunk Summary Len (words)</td><td>43.7</td><td>40.1</td><td>-</td><td>-</td></tr></table>

While Nemori remains slightly ahead in F1 on LoCoMo, EMem/EMem-G match or exceed Nemori on BLEU-1 and, more importantly, achieve higher LLM-judged correctness, suggesting that our answers are semantically more often correct even if word overlap is not always maximal.

The LongMemEvalS results highlight the benefit of EMem/EMem-G in truly long conversations. With gpt-4o-mini, the average LLM-judged accuracy improves to $7 7 . 9 \%$ for EMem-G and $7 6 . 0 \%$ for EMem, while reducing the effective context from 101K tokens to 1.0K–3.6K and 0.6K–2.5K, respectively. With the stronger gpt-4.1-mini, we see the same trend. The largest gains come from temporal-reasoning, multi-session, and knowledge-update questions. These categories are exactly where the design of EDUs and event arguments matters: the EDU abstraction keeps who–did–what–when–where bundled into self-contained units, and recall-oriented LLM filtering plus event-aware retrieval makes it easier to locate and recombine the right long-range evidence than either full-context prompting or heuristic memory stores.

At the same time, Nemori remains competitive on singlesession questions, especially preference and assistantrelated ones. These questions depend more on local stylistic cues and summarizing user habits within a short window than on integrating long-range event structure. Nemori’s dual episodic–semantic memory, which first generates rich narrative episodes and then distills them into semantic knowledge (including user habits and inclinations), is therefore often sufficient and sometimes better aligned with the judge on these tasks. By contrast, the EDU extractor is deliberately tuned toward factual, event-like content; as

a result, purely attitudinal or stylistic information can be over-compressed or dropped, which limits performance on single-session-preference questions.

Comparing EMem and EMem-G directly, we see that the graph-based retrieval is helpful but not universally necessary. On LoCoMo, EMem and EMem-G attain essentially identical overall LLM scores with gpt-4o-mini, with EMem slightly stronger on open-domain questions and EMem-G slightly stronger on multi-hop questions. With gpt-4.1-mini, EMem-G leads by about one percentage point overall. On LongMemEvalS, EMem-G has a modest edge in average accuracy, and clearly helps on tasks that require stitching together scattered information (temporal reasoning, multi-session). In contrast, EMem is often on par or slightly better on knowledge-update questions, where a small number of highly relevant, recent EDUs dominate and graph propagation adds limited additional signal.

Overall, these patterns suggest that (i) the event-semanticscentric EDU representation plus recall-oriented LLM filtering is the primary source of gains over baselines; (ii) graph-based propagation over arguments provides additional benefit mainly for queries that require relational and temporal integration across distant parts of the dialogue; and (iii) the lightweight EMem variant is already a strong, practical default, with EMem-G offering extra headroom on the most structurally demanding long-term memory tasks at a modest additional complexity cost.

# 4.3. Graph Statistics

Table 4 summarizes the graphs constructed by EMem-G and confirms that both benchmarks are structurally challenging.

Table 5. Ablation performance on LoCoMo dataset (Maharana et al., 2024) categorized by question type. gpt-4o-mini is adopted as base LLM and OpenAI text-embedding-3-small is adopted as base embedding model.   

<table><tr><td rowspan="2"></td><td colspan="3">Temporal Reasoning</td><td colspan="3">Open Domain</td><td colspan="3">Multi-Hop</td><td colspan="3">Single-Hop</td><td colspan="3">Overall</td></tr><tr><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td></tr><tr><td>EMem-G</td><td>0.760 ± 0.003</td><td>0.581</td><td>0.468</td><td>0.573 ± 0.013</td><td>0.242</td><td>0.199</td><td>0.747 ± 0.006</td><td>0.406</td><td>0.305</td><td>0.823 ± 0.001</td><td>0.504</td><td>0.422</td><td>0.780 ± 0.000</td><td>0.487</td><td>0.397</td></tr><tr><td>w/o Query MD to node</td><td>0.754 ± 0.006</td><td>0.581</td><td>0.468</td><td>0.559 ± 0.009</td><td>0.276</td><td>0.236</td><td>0.675 ± 0.007</td><td>0.405</td><td>0.305</td><td>0.823 ± 0.002</td><td>0.507</td><td>0.419</td><td>0.766 ± 0.001</td><td>0.490</td><td>0.397</td></tr><tr><td>w/ Query NED to node</td><td>0.760 ± 0.001</td><td>0.580</td><td>0.462</td><td>0.559 ± 0.009</td><td>0.275</td><td>0.226</td><td>0.695 ± 0.002</td><td>0.407</td><td>0.302</td><td>0.820 ± 0.000</td><td>0.505</td><td>0.420</td><td>0.769 ± 0.001</td><td>0.489</td><td>0.395</td></tr><tr><td>w/o EDU Filter</td><td>0.747 ± 0.003</td><td>0.565</td><td>0.458</td><td>0.516 ± 0.000</td><td>0.252</td><td>0.201</td><td>0.602 ± 0.007</td><td>0.357</td><td>0.244</td><td>0.796 ± 0.001</td><td>0.502</td><td>0.420</td><td>0.733 ± 0.002</td><td>0.473</td><td>0.382</td></tr><tr><td>w/o QA zero-shot CoT</td><td>0.765 ± 0.004</td><td>0.595</td><td>0.485</td><td>0.595 ± 0.005</td><td>0.258</td><td>0.208</td><td>0.819 ± 0.001</td><td>0.413</td><td>0.308</td><td>0.819 ± 0.001</td><td>0.529</td><td>0.443</td><td>0.775 ± 0.003</td><td>0.505</td><td>0.413</td></tr><tr><td>w/o Graph &amp; PPR (EMem)</td><td>0.771 ± 0.004</td><td>0.574</td><td>0.461</td><td>0.602 ± 0.009</td><td>0.285</td><td>0.237</td><td>0.702 ± 0.004</td><td>0.406</td><td>0.307</td><td>0.830 ± 0.002</td><td>0.497</td><td>0.414</td><td>0.780 ± 0.001</td><td>0.483</td><td>0.393</td></tr><tr><td>w/o EDU Filter</td><td>0.748 ± 0.005</td><td>0.567</td><td>0.462</td><td>0.588 ± 0.005</td><td>0.268</td><td>0.221</td><td>0.633 ± 0.010</td><td>0.363</td><td>0.252</td><td>0.805 ± 0.002</td><td>0.502</td><td>0.417</td><td>0.748 ± 0.003</td><td>0.476</td><td>0.384</td></tr><tr><td>w/o QA zero-shot CoT</td><td>0.768 ± 0.004</td><td>0.596</td><td>0.476</td><td>0.595 ± 0.005</td><td>0.291</td><td>0.244</td><td>0.701 ± 0.003</td><td>0.415</td><td>0.307</td><td>0.819 ± 0.001</td><td>0.520</td><td>0.431</td><td>0.773 ± 0.001</td><td>0.503</td><td>0.407</td></tr></table>

Table 6. Ablation performance on LongMemEvalS dataset (Wu et al., 2025a) across different question type. gpt-4o-mini is adopted as base LLM and OpenAI text-embedding-3-small is adopted as base embedding model. LLM-judged QA accuracy is reported.   

<table><tr><td></td><td>single-session-preference</td><td>single-session-assistant</td><td>temporal-reasoning</td><td>multi-session</td><td>knowledge-update</td><td>single-session-user</td><td>Average</td></tr><tr><td>EMem-G</td><td>32.2%</td><td>87.5%</td><td>74.8%</td><td>73.6%</td><td>94.4%</td><td>87.0%</td><td>77.9%</td></tr><tr><td>w/o Query MD to node</td><td>26.7%</td><td>83.3%</td><td>72.7%</td><td>73.3%</td><td>88.9%</td><td>88.5%</td><td>75.8%</td></tr><tr><td>w/ Query NED to node</td><td>26.7%</td><td>86.3%</td><td>76.1%</td><td>77.7%</td><td>91.7%</td><td>89.6%</td><td>78.8%</td></tr><tr><td>w/o QA zero-shot CoT</td><td>3.3%</td><td>87.5%</td><td>72.4%</td><td>68.3%</td><td>90.3%</td><td>86.5%</td><td>73.4%</td></tr><tr><td>w/o EDU Filter</td><td>10.0%</td><td>85.1%</td><td>74.5%</td><td>62.0%</td><td>90.3%</td><td>91.1%</td><td>73.1%</td></tr><tr><td>w/o Graph &amp; PPR (EMem)</td><td>32.2%</td><td>82.1%</td><td>69.8%</td><td>78.0%</td><td>87.5%</td><td>86.5%</td><td>76.0%</td></tr><tr><td>w/o EDU Filter</td><td>26.7%</td><td>87.5%</td><td>70.1%</td><td>64.7%</td><td>84.7%</td><td>85.9%</td><td>72.4%</td></tr><tr><td>w/o QA zero-shot CoT</td><td>15.5%</td><td>85.7%</td><td>56.7%</td><td>66.1%</td><td>94.4%</td><td>92.2%</td><td>70.6%</td></tr></table>

Conversations span dozens of sessions with long sessions in terms of word count, especially on LongMemEvalS, where assistant turns are very long. After EDU abstraction, however, each conversation is reduced to a manageable graph with on the order of 500–1,400 EDUs and 1,000–4,000 argument nodes, depending on the dataset and backbone. Moving from gpt-4o-mini to gpt-4.1-mini produces a large increase in EDUs and arguments on LongMemEvalS but only a modest change on LoCoMo, indicating that stronger extractors matter most when turns are long and heterogeneous.

Despite the scale, the graphs are sparse: EDU nodes connect to only a handful of arguments, argument nodes have degree around one or two, and synonym edges are comparatively rare. This means that events form small, welllocalized neighborhoods, which is favorable for personalized PageRank—random walks remain concentrated around a few relevant clusters rather than diffusing through a dense graph. This could also indicate that a better event argument extraction method producing more normalized, atomic arguments could form a more dense graph for better graph-based retrieval performance. The speaker-wise statistics further show that the number of memory units per role is both manageable and highly imbalanced: in LongMemEvalS the assistant contributes the majority of EDUs and chunks, and in LoCoMo one interlocutor dominates the EDU count, reflecting where most factual content actually resides. At the same time, individual EDUs are short (roughly one to two sentences) and long assistant responses are represented by compact summaries, so most of the information that retrieval and QA operate on is concentrated into concise, event-centric nodes rather than raw turns. These properties

together explain why our event-centric graph scales to very long conversations while still enabling efficient, focused retrieval.

# 4.4. Ablation Study

We ablate the main components of EMem and EMem-G on LoCoMo and LongMemEvalS (Tables 5 and 6). On Lo-CoMo, the LLM-based EDU filter is the most critical piece: removing it reduces the overall LLM score from 0.780 to 0.733 for EMem-G and to 0.748 for EMem, with multihop performance dropping by 7–15 points, confirming that recall-oriented filtering is key to suppressing noisy EDUs while preserving relevant events. Removing query mention– argument node weight initialization in EMem-G leads to a smaller overall drop $( 0 . 7 8 0  0 . 7 6 6 )$ but hurts multi-hop reasoning $( 0 . 7 4 7  0 . 6 7 5 )$ ; replacing the LLM-based mention detector with a named entity mention variant largely closes this gap, indicating that graph propagation benefits from some form of query–argument anchoring but is fairly robust to the choice of detector. Dropping QA chain-ofthought has only a minor effect on the LoCoMo average (less than one point) for both models, mainly trading a small decrease in LLM score for slightly higher F1/BLEU.

On LongMemEvalS, the same components remain important but their impact is larger. For EMem-G, removing the EDU filter or QA CoT reduces the average accuracy from $7 7 . 9 \%$ to $7 3 . 1 \%$ and $7 3 . 4 \%$ , respectively, with multi-session questions dropping by up to 11.6 points, showing that pruning and explicit reasoning both matter when information is spread over many sessions. This could be related to our EDU extraction which leads to lots of topic-wise similar

![](images/c43ebfd6cb9f9b07144b5c309d707c0cc887480b73affd7cb5eb7cc7878781f6.jpg)

![](images/5065bbf891a358bf0f90350899fc27cbe404101242d1a89ed169472d099d8634.jpg)

![](images/4bfa415a9e2524c939be6c280b24c3703e0cfa4118c3f54baa7e5ddfc9bd78b1.jpg)  
Figure 2. LLM-judged QA accuracy of EMem with different linking Top- $k$ setup. gpt-4o-mini is adopted as base LLM and OpenAI text-embedding-3-small is adopted as base embedding model.

![](images/edbd99036ac580ead0c6eae75c9482644e1ef13a2b49f5440e2a33a8d8686f44.jpg)  
Figure 3. LLM-judged QA accuracy of EMem-G with different linking Top- $k$ setup. gpt-4o-mini is adopted as base LLM and OpenAI text-embedding-3-small is adopted as base embedding model.

EDUs from same sessions, hence requiring a relatively more powerful embedding model, reranker, or filter to remove noisy information. Removing the graph and PPR (EMem) lowers the average to $7 6 . 0 \%$ : the graph particularly helps temporal-reasoning and knowledge-update questions, while EMem slightly improves multi-session performance, consistent with the trade-offs in the main results. For EMem, ablations of the EDU filter and QA CoT again mainly harm multi-session and temporal-reasoning questions. Overall, these results support our design choice that event-semantics based EDUs plus recall-oriented filtering carry most of the gains, with graph propagation and QA CoT providing additional, dataset-dependent improvements on the hardest long-term reasoning cases.

# 4.5. Retrieval Hyperparameters Anaysis

We first vary the number of retrieved candidate EDUs before filtering (linking Top- $K _ { e }$ ) in EMem (Figure 2). Across both LoCoMo and LongMemEvalS, performance is relatively flat once $K _ { e }$ is in the range of 20–40, with a mild optimum around 20–30 depending on the dataset. This indicates that the combination of dense retrieval and recall-oriented EDU filtering is robust: as long as we retrieve a moderately sized candidate pool, the LLM filter is able to discard noisy EDUs and preserve most relevant information, and there is no need to aggressively tune $K _ { e }$ .

For EMem-G, we similarly vary linking Top- $K _ { e }$ , which also determines the upper bound on non-zero argument seeds (see Figure 3). Accuracy improves as $K _ { e }$ grows from small values and peaks around 30 on both datasets, with only marginal degradation beyond that point, suggesting that graph propagation benefits from a richer set of seeds but becomes slightly more susceptible to noise when too many low-relevance EDUs and arguments are injected. Finally, varying the QA Top- $K$ (the number of EDUs passed to the QA model) for EMem-G with fixed linking $K _ { e } = 3 0$ (Figure 4) shows a steep gain when moving from very small contexts (1–3 EDUs) to moderate ones (5–10 EDUs), after which performance quickly saturates around 10–15 EDUs. Overall, these trends indicate that our memory system is stable across a broad range of retrieval and augmentation hyperparameters, and that strong performance can be obtained with small, fixed context budgets that remain practical for real-world deployment.

# 5. Conclusion

We have argued for an event-centric view of conversational memory, grounded in neo-Davidsonian semantics, where long-term dialogue is reconstructed as a graph of enriched EDUs rather than as raw turns, coarse summaries, or fragmented triples. By instructing an LLM to produce self-

![](images/9f7ce1a042390d0630ed925f11f59ca195f5077b053587718e49a277e29381dd.jpg)

![](images/69da984b1b9913ab33c94c1586c2142f9727aec42033851a622173e7f1f08b65.jpg)  
Figure 4. LLM-judged QA accuracy of EMem with different memory augmentation Top- $k$ setup. gpt-4o-mini is adopted as base LLM and OpenAI text-embedding-3-small is adopted as base embedding model.

contained, normalized event units and organizing them into a heterogeneous graph over sessions, EDUs, and arguments, our framework enables associative recall and dense–sparse integration that are difficult to realize with flat retrieval over text chunks. The two retrieval variants, EMem and EMem-G, differ only in whether they invoke graph propagation, but share the same design philosophy: use simple, high-recall dense retrieval, apply a lightweight LLM filter to remove noisy candidates, and then reason over a compact set of event memories.

Experiments on LoCoMo and LongMemEvalS show that this design yields strong performance on temporal, multihop, and knowledge-update questions while operating with modest context budgets, and that the induced memory graphs are sparse, interpretable, and robust to retrieval hyperparameters. At the same time, weaker performance on single-session preference questions highlights a limitation of a purely event-centric representation for capturing finegrained user attitudes and styles. An important direction for future work is to pair event-level memory with complementary models of user profiles and interaction patterns, and to extend this framework beyond dialogue to other longhorizon settings such as tool-augmented agents and multidocument reasoning.

# Acknowledgments

Research was supported in part by the AI Institute for Molecular Discovery, Synthetic Strategy, and Manufacturing: Molecule Maker Lab Institute (MMLI), funded by U.S. National Science Foundation under Awards No. 2019897 and 2505932, NSF IIS 25-37827, and the Institute for Geospatial Understanding through an Integrative Discovery Environment (I-GUIDE) by NSF under Award No. 2118329. The research has used the Delta/DeltaAI advanced computing and data resource, supported in part by the University of Illinois Urbana-Champaign and through allocation #250851 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS)

program, which is supported by National Science Foundation grants OAC 2320345, #2138259, #2138286, #2138307, #2137603, and #2138296. Any opinions, findings, and conclusions or recommendations expressed herein are those of the authors and do not necessarily represent the views, either expressed or implied, of DARPA or the U.S. Government. The views and conclusions contained in this paper are those of the authors and should not be interpreted as representing any funding agencies.

# References

Chhikara, P., Khant, D., Aryan, S., Singh, T., and Yadav, D. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.   
Davidson, D. The logical form of action sentences. In Rescher, N. (ed.), The Logic of Decision and Action, pp. 81–95. University of Pittsburgh Press, Pittsburgh, 1967.   
Fang, J., Deng, X., Xu, H., Jiang, Z., Tang, Y., Xu, Z., Deng, S., Yao, Y., Wang, M., Qiao, S., et al. Lightmem: Lightweight and efficient memory-augmented generation. arXiv preprint arXiv:2510.18866, 2025.   
Gutierrez, B. J., Shu, Y., Qi, W., Zhou, S., and Su, Y. From ´ RAG to memory: Non-parametric continual learning for large language models. In Forty-second International Conference on Machine Learning, 2025. URL https: //openreview.net/forum?id=LWH8yn4HS2.   
Gutierrez, B. J., Shu, Y., Gu, Y., Yasunaga, M., and Su, Y. ´ Hipporag: Neurobiologically inspired long-term memory for large language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=hkujvAPVsg.   
Huang, Z., Tian, Z., Guo, Q., Zhang, F., Zhou, Y., Jiang, D., and Zhou, X. Licomemory: Lightweight and cognitive

agentic memory for efficient long-term reasoning. arXiv preprint arXiv:2511.01448, 2025.   
Kim, S., Lee, Y., Kim, S., Kim, H., and Cho, S. Pre-storage reasoning for episodic memory: Shifting inference burden to memory for personalized dialogue. arXiv preprint arXiv:2509.10852, 2025.   
LangChain AI. Langmem: Long-term memory sdk for llm agents. https://langchain-ai.github.io/ langmem/, 2025. Accessed: 2025-11-20.   
Li, Z., Song, S., Wang, H., Niu, S., Chen, D., Yang, J., Xi, C., Lai, H., Zhao, J., Wang, Y., et al. Memos: An operating system for memory-augmented generation (mag) in large language models. arXiv preprint arXiv:2505.22101, 2025.   
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157– 173, 2024. doi: 10.1162/tacl a 00638. URL https: //aclanthology.org/2024.tacl-1.9/.   
Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., and Fang, Y. Evaluating very long-term conversational memory of LLM agents. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13851–13870, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long. 747. URL https://aclanthology.org/2024. acl-long.747/.   
Nan, J., Ma, W., Wu, W., and Chen, Y. Nemori: Selforganizing agent memory inspired by cognitive science. arXiv preprint arXiv:2508.03341, 2025.   
Parsons, T. Events in the Semantics of English. MIT Press, Cambridge, MA, 1990.   
Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., and Chalef, D. Zep: a temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025.   
Wang, J., Zhao, R., Wei, W., Wang, Y., Yu, M., Zhou, J., Xu, J., and Xu, L. Comorag: A cognitive-inspired memoryorganized rag for stateful long narrative reasoning. arXiv preprint arXiv:2508.10419, 2025.   
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Wu, D., Wang, H., Yu, W., Zhang, Y., Chang, K.-W., and Yu, D. Longmemeval: Benchmarking chat assistants on long-term interactive memory. In The Thirteenth International Conference on Learning Representations, 2025a. URL https://openreview.net/forum? id=pZiyCaVuti.   
Wu, Y., Zhang, Y., Liang, S., and Liu, Y. Sgmem: Sentence graph memory for long-term conversational agents. arXiv preprint arXiv:2509.21212, 2025b.   
Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang, Y. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.

# A. Dataset-Specific EDU Extraction for LongMemEval

LongMemEval consists of multi-session dialogues between a user and an assistant (Wu et al., 2025a). We observed a systematic asymmetry in utterance style: user turns are typically short and focused, whereas assistant turns are often long, highly structured responses (e.g., enumerated lists, step-by-step plans, comparative analyses). Leveraging a relatively small LLMs (e.g., gpt-4o-mini) for EDU extraction on such long sessions, we frequently encounter the issue of missing EDUs, which is likely considered as unimportant information. To adapt our EDU extraction pipeline to this setting without overfitting the core method, we apply a slightly different treatment to the two speakers.

User utterances. For the user side, we directly apply the generic EDU extraction procedure from Section 3.2. Each session is passed to the extractor $g _ { \mathrm { E D U } }$ along with timestamps and speaker tags, and the model emits extracted EDUs with source turn index attributions. No dataset-specific modification is required.

Assistant utterances: atomic EDUs and structured chunks. To mitigate the above discussed issues, each assistant turn is processed in two parallel views:

1. Atomic EDUs. The extractor produces a set of fine-grained EDUs, analogous to the user side, each with its own source turn index. These capture localized facts (e.g., an atomic fact or an event).   
2. Structured chunks. In the same call, the extractor is asked to identify cohesive information blocks—“structured chunks”—that group related details presented in an organized way (comparisons, detailed overviews, comprehensive recommendations, step-by-step procedures, lists of related items, etc.). For each chunk $c$ , the model also generates a short summary $s ( c )$ of 2–3 sentences that (i) states which user request or question the chunk addresses, (ii) describes the main information categories covered, and (iii) naturally includes key entities and terms.

We treat each chunk as an additional EDU node in the memory graph, but use the summary $s ( c )$ as the textual content text(e) for argument extraction, indexing, and retrieval. Arguments for chunk nodes are extracted from $s ( c )$ rather than from the full chunk text. The original chunk content is stored separately and is only revealed to the QA model at answer time when the corresponding EDU node is retrieved. This design preserves the organizational structure of long assistant responses while keeping the indexable text short and information-dense.

In practice, both atomic EDUs and structured chunks share the same metadata schema, including source turn indices and session timestamps, and are handled uniformly by the retrieval and graph components. The only difference is that chunk nodes have a hidden “expanded” text used solely in the final QA prompt and we need to initiate two LLM calls on this dataset. This dataset-specific adaptation improves recall on LongMemEvalS without changing the core EMem/EMem-G algorithms.
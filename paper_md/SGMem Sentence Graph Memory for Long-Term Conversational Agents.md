# SGMEM: SENTENCE GRAPH MEMORY FOR LONG-TERM CONVERSATIONAL AGENTS

Yaxiong Wu, Yongyue Zhang, Sheng Liang, Yong Liu

Huawei Technologies Co., Ltd

wu.yaxiong@huawei.com

# ABSTRACT

Long-term conversational agents require effective memory management to handle dialogue histories that exceed the context window of large language models (LLMs). Existing methods based on fact extraction or summarization reduce redundancy but struggle to organize and retrieve relevant information across different granularities of dialogue and generated memory. We introduce SGMem (Sentence Graph Memory), which represents dialogue as sentence-level graphs within chunked units, capturing associations across turn-, round-, and session-level contexts. By combining retrieved raw dialogue with generated memory such as summaries, facts and insights, SGMem supplies LLMs with coherent and relevant context for response generation. Experiments on LongMemEval and LoCoMo show that SGMem consistently improves accuracy and outperforms strong baselines in longterm conversational question answering.

# 1 INTRODUCTION

Memory is a fundamental component of long-term conversational agents (Maharana et al., 2024; Wu et al., 2024), allowing them to augment dialogue context beyond the limited window of large language models (LLMs) (Zhang et al., 2025c; Wu et al., 2025; Sapkota et al., 2025). By acquiring, storing, managing, and retrieving information from prior interactions, memory supports accurate and personalized responses in multi-turn conversations. However, as interactions accumulate, agents inevitably face memory overload (Klingberg, 2009; Yun et al., 2010), where the scale, complexity, or redundancy of stored content exceeds their ability to manage and retrieve it effectively. This condition undermines dialogue understanding and constrains the agent’s capacity to deliver coherent and user-tailored responses.

Memory management (Xiong et al., 2025; Kang et al., 2025) seeks to organize, compress, and filter stored content to improve an agent’s ability to exploit large-scale memory and alleviate memory overload. In long-term conversational agents, memory typically consists of both the raw dialogue history—spanning turns, rounds, and sessions—and generated memory such as summaries, extracted facts, and reflective insights. While techniques like summarization, extraction, and reflection reduce redundancy, they often lead to the so-called memory fragmentation (Bedard-Gilligan & Zoellner, 2012; Kindt & Van Den Hout, 2003), where relevant information is dispersed across raw dialogues and derived snippets, hindering coherent retrieval. Figure 1 provides an overview of memory in long-term conversational agents for question answering (QA), which is typically formulated within a retrieval-augmented generation (RAG) to retrieve relevant memory segments and reduce hallucination (Siriwardhana et al., 2023; Fan et al., 2024).

Despite efforts from existing chunk-based and graph-based approaches (Wu et al., 2025; Pan et al., 2025; Zhang et al., 2025b)—such as employing memory composition and updating strategies inspired by Zettelkasten note-taking (Kadavy, 2021; Ahrens, 2022), or modeling entity–relation associations via event-centric memory graphs (Zhang et al., 2025b)—the problem of memory fragmentation remains largely unresolved. On the one hand, it is still challenging to determine the appropriate granularity at which raw dialogue history should be retrieved and to effectively integrate generated memory with raw history during retrieval. On the other hand, extracting entity-relation triples with LLMs incurs substantial computational costs and further exacerbates memory fragmentation.

![](images/81473a9d4bec40d28efbd113026e8859c17b0b4b28546511b3320804255b9811.jpg)  
Figure 1: Illustration of memory in long-term conversational agents.

To address these challenges, our central design choice is to structure dialogue memory at the sentence level. Sentences serve as the fundamental units of conversational exchange, each encapsulating a semantically coherent statement while remaining fine-grained enough to capture contextual dependencies. Compared with coarser units (e.g., turns, rounds, sessions), sentence-level representations enable more precise alignment between raw dialogue history and generated memory. Moreover, structuring sentences as nodes in a graph allows the agent to explicitly model associations—both within and across dialogue segments—thereby mitigating memory fragmentation and supporting coherent retrieval.

In this paper, we propose SGMem (Sentence Graph Memory), a memory management framework that improves the organization and retrieval of long-term conversational memory. SGMem models dialogue as sentence-level graphs within chunked units, enabling associations across turns, rounds, and sessions. By jointly retrieving raw dialogue history and generated memory (e.g., summaries, facts, and insights), SGMem supplies LLMs with coherent and contextually relevant evidence for response generation. Notably, SGMem requires no additional LLM-based extraction; it relies only on standard sentence segmentation tools (e.g., NLTK (Bird, 2006)) to construct sentence graphs, making it lightweight and readily deployable in long multi-turn conversational settings.

The contributions of this paper are threefold:

• Sentence-Graph Memory Construction & Management: We introduce SGMem, a hierarchical memory framework that mitigates memory fragmentation by organizing dialogue history into sentence-level graphs.   
• Sentence-Graph Memory Usage: We design a multi-hop retrieval mechanism over the sentence graph memory that integrates raw dialogue history with generated memory to support coherent and accurate long-term question answering.   
• Comprehensive Evaluation: We conduct extensive experiments on LongMemEval and LoCoMo, showing that SGMem achieves consistent accuracy gains and outperforms strong baselines, demonstrating both effectiveness and practical value.

# 2 RELATED WORK

Long-Term Conversational Agents. Long-term conversational agents are designed to sustain multisession interactions, but face challenges such as memory retention and update, temporal reasoning, context granularity, and coherent retrieval over fragmented histories. To benchmark these abilities, recent work has introduced dedicated datasets. LongMemEval (Wu et al., 2024) evaluates personal assistants on five memory skills—information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention—revealing significant performance drops in multi-session and

temporally dynamic settings. LoCoMo (Maharana et al., 2024) provides very long persona-grounded, event-driven conversations spanning up to 35 sessions, with tasks such as question answering, event summarization, and multimodal dialogue generation. Results on both benchmarks show that existing LLMs and RAG pipelines struggle with temporal consistency, knowledge updates, and coherent retrieval, underscoring the need for more structured and fine-grained memory management.

Memory Management. Memory management is a central challenge for long-term conversational agents (Wu et al., 2025; Xu et al., 2025a). Existing methods include MemoryBank (Zhong et al., 2024), which hierarchically summarizes events and aggregates personality insights; LD-Agent (Li et al., 2024), which separates long- and short-term memory banks for event summaries and contextual dialogue while updating user personas; RMM (Tan et al., 2025), which reflects on dialogue to form topic-based summaries; MemoryScope (Yu et al., 2024), which consolidates user observations into higher-level insights; and A-MEM (Xu et al., 2025b), which draws on Zettelkasten (Kadavy, 2021; Ahrens, 2022) to link memories as structured notes. Despite these advances, memory fragmentation—where information is scattered across raw dialogue and generated memory—remains unresolved. This motivates our approach, SGMem, which represents dialogue at the sentence level as a graph to align fine-grained semantics and support coherent retrieval.

Retrieval Augmented Generation. Retrieval-augmented generation (RAG) is a dominant paradigm for grounding LLMs with external knowledge in long-term conversational agents (Jin et al., 2024). Chunk-based RAG retrieves dialogue segments at the turn, round, or session level, offering simplicity and scalability but often limited by coarse granularity and fragmented retrieval. In contrast, graphbased RAG methods (Zhang et al., 2025a; Han et al., 2024), such as GraphRAG (Edge et al., 2024), LightRAG (Guo et al., 2024), and HippoRAG (Jimenez Gutierrez et al., 2024; Gutierrez et al., ´ 2025), construct structured indexes over entities, relations, or hierarchical clusters to capture richer cross-document associations. In line with this perspective, graph-based RAG has been employed to construct memory graphs centered on entity relationships, exemplified by the event-centric memory graph (Zhang et al., 2025b). A key limitation of entity-level memory graphs is their reliance on costly LLM computations for entity and relation extraction, which also discards rich contextual information. To address this issue, we introduce SGMem, a lightweight sentence-level graph memory that eliminates the need for LLM-based extraction while retaining the semantic content of sentences.

# 3 METHODOLOGY

# 3.1 PRELIMINARIES

We consider the task of long-term conversational question answering (QA), where the input consists of a sequence of sessions denoted as $\mathcal { S } = \{ s _ { 1 } , s _ { 2 } , \ldots , s _ { U } \}$ . Each session $s _ { u }$ contains multiple turns $\mathcal { T } = \{ \dot { t } _ { w } \} _ { w = 1 } ^ { W }$ , which can be further grouped into user-assistant rounds $\bar { \mathcal { R } } = \{ r _ { v } \} _ { v = 1 } ^ { V }$ to reflect higher-level conversational exchanges. In addition to raw sessions, long-term conversational agents often maintain various forms of generated memory, including summaries $\mathcal { M } = \{ m _ { x } \} _ { x = 1 } ^ { X }$ , facts $\mathcal { F } = \{ f _ { y } \} _ { y = 1 } ^ { Y }$ , and insights $\mathcal { T } = \{ i _ { z } \} _ { z = 1 } ^ { Z }$ x=1, produced by large language models (LLMs) through summarization, extraction, or reflection. At a finer granularity, each turn $t _ { w }$ can be segmented into a set of sentences $\boldsymbol { \mathcal { C } } = \{ c _ { j } \} _ { j = 1 } ^ { J }$ using standard NLP tools such as NLTK (Bird, 2006). These hierarchical units—sessions, rounds, turns, generated memory, and sentences—form the basis of our Sentence Graph Memory (SGMem) management and retrieval framework.

# 3.2 FRAMEWORK OVERVIEW

Long-term conversational agents often suffer from coarse memory segmentation, where both raw dialogue history (turns, rounds, sessions) and generated memories (summaries, facts, insights) are stored and retrieved at coarse granularity, leading to fragmented and incoherent context. To address this limitation, we propose Sentence Graph Memory (SGMem), which organizes dialogue at the sentence level and explicitly models semantic associations through graph structures. Figure 2 presents an overview of SGMem, which consists of two main components: (1) SGMem Construction & Management, and (2) SGMem Usage.

SGMem Construction & Management. The construction of Sentence Graph Memory (SGMem) consists of four steps: (1) Processing Conversations: Sessions are hierarchically decomposed into

![](images/11437f83bb01d1563328f7949ed7b7fea628a68fac3617c767b95e1284323419.jpg)

![](images/fc1671d0c0e77f0499ea981500289589ff615725fc12e4236eb8256be3e7dc26.jpg)  
Figure 2: Overview of the proposed Sentence Graph Memory (SGMem) framework, consisting of (a) SGMem Construction & Management for building sentence-level memory graphs and (b) SGMem Usage for retrieving relevant memory in long-term conversational QA.

rounds, turns, and sentences, while LLMs generate summaries, facts, and insights. (2) Indexing: All memory units are embedded into vector spaces to build seven searchable index tables. (3) Constructing Sentence Graph Memory: Chunk nodes (sessions, rounds, or turns) are linked to their constituent sentences and further connected by sentence–sentence similarity edges. (4) Storage: Index tables are stored in a vector database for efficient search, and the sentence graph memory is maintained in a graph database for reasoning and traversal. This design yields a structured and queryable memory foundation for SGMem Usage.

SGMem Usage. The usage of Sentence Graph Memory (SGMem) consists of four steps. (1) Retrieve Memory and Sentences: The query retrieves candidate summaries, facts, insights, and sentences from the vector database. (2) Rank Chunks with SGMem: Retrieved sentences are expanded via $n$ -hop graph traversal and mapped back to their parent chunks, which are ranked and truncated for relevance. (3) Collect Relevant Context: Selected chunks together with generated memories are aggregated into a unified relevant context. (4) Personalized Generation: The aggregated context is fed into the LLM to produce accurate and personalized responses. Overall, this dual design of vector retrieval and graph expansion ensures coherent context selection for long-term conversational QA.

# 3.3 SGMEM CONSTRUCTION & MANAGEMENT

Sentence Graph Memory (SGMem) is constructed and managed through four steps, forming a structured and queryable foundation for its usage.

Processing Conversations. Given a set of long conversations $\mathcal { S } = \{ s _ { u } \} _ { u = 1 } ^ { U }$ , we decompose each session $s _ { u }$ into rounds $\mathcal { R } = \{ r _ { v } \} _ { v = 1 } ^ { V }$ and turns $\mathbf { \bar { \mathcal { T } } } = \{ t _ { w } \} _ { w = 1 } ^ { W }$ . Each turn $t _ { w }$ is further segmented into a set of sentences $\mathcal { C } = \{ c _ { j } \} _ { j = 1 } ^ { J }$ using standard NLP tools (e.g., NLTK (Bird, 2006)). In parallel, we employ an LLM to derive three types of generated memory:

$$
\mathcal {M} = \left\{m _ {x} \right\} _ {x = 1} ^ {X}, \quad \mathcal {F} = \left\{f _ {y} \right\} _ {y = 1} ^ {Y}, \quad \mathcal {I} = \left\{i _ {z} \right\} _ {z = 1} ^ {Z},
$$

where $\mathcal { M } , \mathcal { F } , \mathcal { Z }$ denote summaries, facts, and insights, respectively.

Indexing. Each memory unit $u \in \{ s , r , t , c , m , f , i \}$ is encoded into a vector representation $\mathbf { e } _ { u } \in$ $\mathbb { R } ^ { d }$ using a pretrained embedding model $E ( \cdot )$ , such as Sentence-BERT (Reimers & Gurevych, 2019):

$\mathbf { e } _ { u } = E ( u )$ . This produces seven index tables corresponding to sessions, rounds, turns, sentences, summaries, facts, and insights.

Constructing Sentence Graph Memory. Raw dialogue units (session, round, or turn) are treated as chunk nodes $\mathcal { K } = \{ k _ { p } \}$ , and each chunk $k _ { p }$ is linked to its constituent sentences $c _ { j }$ through membership edges:

$$
\left(k _ {p}, c _ {j}\right) \in \mathcal {E} _ {\text {c h u n k - s e n t}} \quad \text {i f} c _ {j} \in k _ {p}.
$$

In addition, we compute pairwise sentence similarity: $\mathrm { s i m } ( c _ { j } , c _ { j ^ { \prime } } ) = \cos ( { \mathbf { e } } _ { c _ { j } } , { \mathbf { e } } _ { c _ { j ^ { \prime } } } )$ , and construct a $k$ -nearest-neighbor (KNN) graph:

$$
\left(c _ {j}, c _ {j ^ {\prime}}\right) \in \mathcal {E} _ {\text {s e n t - s e n t}} \quad \text {i f} c _ {j ^ {\prime}} \in \operatorname {T o p} - k \text {n e i g h b o r s} c _ {j}.
$$

The overall sentence graph memory is thus defined as:

$$
\mathcal {G} = \left(\mathcal {V}, \mathcal {E} _ {\text {c h u n k - s e n t}} \cup \mathcal {E} _ {\text {s e n t - s e n t}}\right), \quad \text {w h e r e} \quad \mathcal {V} = \mathcal {K} \cup \mathcal {C}.
$$

Storage. The seven index tables $\left\{ \mathbf { e } _ { u } \right\}$ are stored in a vector database (e.g., ElasticSearch1) for efficient similarity search, while the graph $\mathcal { G }$ is stored in a graph database (e.g., $\mathrm { N e o 4 j } ^ { 2 }$ ) to support reasoning and traversal-based retrieval.

# 3.4 SGMEM USAGE

The usage of Sentence Graph Memory (SGMem) consists of four steps designed to maximize retrieval coherence and personalization.

Retrieve Memory and Sentences. Given a query $q$ , we retrieve candidate summaries $\mathcal { M }$ , facts $\mathcal { F }$ , insights $\mathcal { T }$ , and sentences $\mathcal { C }$ from their respective index tables in the vector database. Retrieval is based on cosine similarity:

$$
\operatorname {s i m} (q, u) = \cos \left(\mathbf {e} _ {q}, \mathbf {e} _ {u}\right) + \epsilon , \quad u \in \left\{m _ {x}, f _ {y}, i _ {z}, c _ {j} \right\},
$$

where ${ \bf e } _ { q }$ and $\mathbf { e } _ { u }$ are embeddings of query $q$ and memory unit $u$ , and $\epsilon = 1$ . The top- $K$ memory units are retained as $\mathcal { M } ^ { * }$ , ${ \mathcal { F } } ^ { * }$ , and $\mathcal { T } ^ { * }$ , respectively. A similarity threshold $\gamma \in [ 0 , 2 ]$ and a maximum number of sentence nodes $n$ are employed to constrain the retrieval process of sentence nodes.

Rank Chunks with SGMem. Retrieved sentences $\mathcal { C } _ { q }$ are expanded via $h$ -hop traversal on the sentence graph $\mathcal { G }$ to gather neighbors ${ \mathcal { N } } _ { h } ( { \mathcal { C } } _ { q } )$ :

$$
\mathcal {C} ^ {*} = \mathcal {C} _ {q} \cup \mathcal {N} _ {h} (\mathcal {C} _ {q}).
$$

Each sentence $c _ { j } \in \mathcal { C } ^ { * }$ is mapped back to its parent chunk $k _ { p }$ (session, round, or turn), and chunks are ranked by an aggregate score:

$$
s c o r e (k _ {p}) = \frac {1}{| \mathcal {C} _ {k _ {p}} |} \sum_ {c _ {j} \in \mathcal {C} _ {k _ {p}}} \operatorname {s i m} (q, c _ {j}),
$$

where $\mathcal { C } _ { k _ { p } }$ is the set of retrieved and neighboring sentences belonging to chunk $k _ { p }$ . Top- $K$ chunks are retained as $\kappa ^ { * }$ .

Collect Relevant Context. The final relevant context is the union of retrieved chunks, summaries, facts, and insights:

$$
\mathcal {C} _ {\text {r e l e v a n t}} = \mathcal {K} ^ {*} \cup \mathcal {M} ^ {*} \cup \mathcal {F} ^ {*} \cup \mathcal {I} ^ {*}.
$$

Personalized Generation. The LLM generates an output response $\hat { y }$ conditioned on the query $q$ and the relevant context $\mathcal { C } _ { \mathrm { r e l e v a n t } }$ :

$$
\hat {y} = \operatorname {L L M} (q \mid \mathcal {C} _ {\text {r e l e v a n t}}).
$$

Overall, SGMem Usage benefits from the dual design of vector-based retrieval and graph-based expansion: vector indexes provide efficient access to heterogeneous memory types, while sentence graph traversal ensures contextual coherence, leading to improved response accuracy and personalization in long-term conversational QA.

# 4 EXPERIMENTAL SETTINGS

To comprehensively evaluate SGMem, we aim to answer the following research questions: • RQ1: Does SGMem outperform existing memory management and RAG methods in long-term conversational QA? • RQ2: How do different types of context (e.g., raw dialogue units vs. generated memory) influence QA effectiveness? • RQ3: How does SGMem perform across different query types? • RQ4: How do the hyperparameters (e.g., $k , h , n , \gamma )$ affect the performance of SGMem?

# 4.1 DATASETS

We conduct experiments on two long-term conversational benchmarks. LongMemEval (Wu et al., 2024) comprises 500 curated questions spanning six types—single-session-user (70), single-sessionassistant (56), single-session-preference (30), multi-session (133), knowledge-update (78), and temporal-reasoning (133)—embedded in user–assistant dialogues of varying length. LoCoMo (Maharana et al., 2024) provides very long multi-session dialogues (300 turns, 9K tokens, up to 35 sessions) grounded in personas and temporal event graphs; we randomly sample 500 questions, covering single-hop (156), multi-hop (133), temporal reasoning (133), and open-domain knowledge (78), to ensure computational feasibility and enable extensive ablation studies. Together, these datasets jointly evaluate fine-grained memory abilities and scalability to very long, multi-session interactions. The dataset details are provided in Appendix A.

# 4.2 EVALUATION METRIC

We evaluate long-term conversational question answering using Accuracy, where correctness of a model response is determined by the LLM-as-a-Judge paradigm (Gu et al., 2024). Specifically, a strong LLM is prompted to compare the generated response against the reference answer and decide whether it is correct. This design avoids the brittleness of exact string matching, allowing the metric to account for paraphrases and semantically equivalent answers while still providing a clear accuracy score. We report accuracy across different question types in both datasets. To ensure reproducibility, we adopt a fixed evaluation prompt and provide the full prompt template in the Appendix B.

# 4.3 BASELINES

Simple Baselines. No History: Answers questions without considering dialogue history, using only the query itself as input. Long Context: Directly feeds the LLM with dialogue history, either the most recent sessions $( L C _ { L a t e s t } )$ or all sessions $( L C _ { F u l l } )$ .

Memory Management Baselines. MemoryBank (Zhong et al., 2024): Maintains chronological memory with hierarchical summaries and uses rounds $^ +$ summaries as context. LD-Agent (Li et al., 2024): Disentangles long- and short-term memory banks and uses summaries $^ +$ facts as context. LongMemEval (Wu et al., 2024): Builds indexes over sessions augmented with corresponding facts, retrieving sessions as context. MemoryScope (Yu et al., 2024): Performs consolidation and reflection, using rounds + facts $^ +$ insights as context. RMM (Tan et al., 2025): Applies prospective reflection over history and uses facts as context.

Graph-based RAG Baselines. LightRAG (Guo et al., 2024): Constructs lightweight relational graphs and uses entities $^ +$ relations as context. MiniRAG (Fan et al., 2025): Compresses conversational memory into smaller graph structures and uses sessions $^ +$ entities as context. KG-Retriever (Chen et al., 2024): Builds hierarchical knowledge graphs and uses relations as context.

Chunk-based RAG Variants. We also implement chunk-based RAG variants by varying the memory unit used as retrieval context. Specifically, we evaluate RAG with turns (RAG-T, RAG-TF, RAG-TMFI), rounds (RAG-R, RAG-RF, RAG-RMFI), and sessions (RAG-S, RAG-SF, RAG-SMFI), where “TF” denotes turns with facts, and “TMFI” denotes turns with summaries, facts, and insights. For each variant, we retrieve the top- $K$ items from the specified memory types (e.g., turns, summaries, facts, insights) and concatenate them as context. SGMem is evaluated under the same variants for fair comparison.

# 4.4 SETUP

Retriever. We use Sentence-BERT (Reimers & Gurevych, 2019) for dense retrieval, specifically the all-MiniLM-L6-v2 model for embedding sentences and memory units. It is also employed

Table 1: Performance comparison on LongMemEval and LoCoMo using Accuracy (Top-5 / Top-10). KG denotes knowledge graph, HIG denotes hierarchical index graph, SG denotes sentence graph. T, R, S, M, F, I, E, L denote turn, round, session, summary, fact, insight, entity, and relation respectively. The overall best results are marked with *, the best RAG method is shown in bold, and the second-best RAG method is underlined.   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Mode</td><td rowspan="2">Graph</td><td colspan="8">Context Type</td><td colspan="2">LongMemEval</td><td colspan="2">LoCoMo</td></tr><tr><td>T</td><td>R</td><td>S</td><td>M</td><td>F</td><td>I</td><td>E</td><td>L</td><td>Top-5</td><td>Top-10</td><td>Top-5</td><td>Top-10</td></tr><tr><td>No History</td><td>LLM</td><td>-</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.000</td><td>0.000</td><td>0.050</td><td>0.050</td></tr><tr><td>\(LC_{Latest}\)</td><td>LLM</td><td>-</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.144</td><td>0.196</td><td>0.196</td><td>0.292</td></tr><tr><td>\(LC_{Full}\)</td><td>LLM</td><td>-</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.478</td><td>0.478</td><td>0.558*</td><td>0.558*</td></tr><tr><td>MemoryBank</td><td>RAG</td><td>-</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.498</td><td>0.558</td><td>0.388</td><td>0.422</td></tr><tr><td>LD-Agent</td><td>RAG</td><td>-</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>0.502</td><td>0.574</td><td>0.418</td><td>0.434</td></tr><tr><td>LongMemEval</td><td>RAG</td><td>-</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.552</td><td>0.556</td><td>0.346</td><td>0.410</td></tr><tr><td>MemoryScope</td><td>RAG</td><td>-</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>0.642</td><td>0.678</td><td>0.430</td><td>0.468</td></tr><tr><td>RMM</td><td>RAG</td><td>-</td><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>0.612</td><td>0.668</td><td>-</td><td>-</td></tr><tr><td>LightRAG</td><td>RAG</td><td>KG</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>0.420</td><td>0.428</td><td>0.360</td><td>0.406</td></tr><tr><td>MiniRAG</td><td>RAG</td><td>KG</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>0.422</td><td>0.468</td><td>0.268</td><td>0.336</td></tr><tr><td>KG-Retriever</td><td>RAG</td><td>HIG</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td><td>0.112</td><td>0.104</td><td>0.138</td><td>0.124</td></tr><tr><td>RAG-T</td><td>RAG</td><td>-</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.456</td><td>0.544</td><td>0.286</td><td>0.330</td></tr><tr><td>RAG-R</td><td>RAG</td><td>-</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.478</td><td>0.564</td><td>0.284</td><td>0.352</td></tr><tr><td>RAG-S</td><td>RAG</td><td>-</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.574</td><td>0.576</td><td>0.340</td><td>0.408</td></tr><tr><td>RAG-SF</td><td>RAG</td><td>-</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>0.656</td><td>0.684</td><td>0.478</td><td>0.502</td></tr><tr><td>RAG-SMFI</td><td>RAG</td><td>-</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>0.676</td><td>0.680</td><td>0.510</td><td>0.528</td></tr><tr><td>SGMem-S</td><td>RAG</td><td>SG</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>0.644</td><td>0.614</td><td>0.392</td><td>0.476</td></tr><tr><td>SGMem-SF</td><td>RAG</td><td>SG</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>0.690</td><td>0.730*</td><td>0.522</td><td>0.542</td></tr><tr><td>SGMem-SMFI</td><td>RAG</td><td>SG</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>0.700*</td><td>0.730*</td><td>0.526</td><td>0.532</td></tr></table>

to compute sentence similarities when constructing the $k$ -nearest-neighbor (KNN) graph in SGMem, where BM25 (Robertson et al., 2009) is adopted for a fair comparison.

LLM. For both question answering and evaluation tasks, we employ a state-of-the-art instructiontuned model with 32B parameters, Qwen2.5-32B-Instruct (Qwen et al., 2025). LLM was accessed via the BaiLian3 API platform with default generation parameters: temperatur $e = 0 . 7$ , top $p = 0 . 8$ , top $k = 2 0$ , max input tokens = 129,024, and max tokens = 8,192.

Hyperparameters. By default, for LongMemEval, we set $k = 3$ , $h = 1$ , $n = 1 5$ , $\gamma = 1 . 0$ , and $K = 5$ , with ablation studies conducted on the SGMem-TF variant $( \mathrm { T u r n } + \mathrm { F a c t } )$ ). For LoCoMo, we use $k = 1$ , $h = 1$ , $n = 1 5$ , $\gamma = 1 . 2$ , and $K = 5$ , with ablations conducted on the SGMem-SF variant (Session + Fact). The ranges explored in hyperparameter sensitivity analysis are: $k \in \{ 1 , 2 , 3 , 4 , 5 \}$ , $h \in \{ 0 , 1 , 2 \}$ , $n \in \{ 5 , 1 0 , 1 5 , 2 0 \}$ , $\gamma \in \{ 1 . 0 , 1 . 2 , 1 . 5 \}$ , $K \in \{ 5 , 1 0 \}$ .

# 5 EXPERIMENTAL RESULTS

Extensive experiments are performed to evaluate SGMem against strong baselines (Section 5.1), analyze the effect of different context types (Section 5.2), investigate its performance across different query types (Section 5.3), and study the sensitivity of hyperparameters (Section 5.4). This section addresses four research questions (RQ1–RQ4 in Section 4).

# 5.1 SGMEM VS. BASELINES

To address RQ1, we compare SGMem against a broad set of representative baselines (Section 4.3). Table 1 reports results on LongMemEval and LoCoMo using Accuracy (Top-5 / Top-10). Among the simple baselines, directly using no history yields near-zero performance, while $L C _ { L a t e s t }$ and $L C _ { F u l l }$ show limited improvement on LongMemEval, highlighting the insufficiency of naive longcontext usage. For memory management baselines, methods such as MemoryBank, LD-Agent, and LongMemEval achieve moderate gains, whereas more advanced strategies like MemoryScope and RMM substantially improve accuracy by introducing structured summaries, facts, and insights. Graph-based approaches, including LightRAG, MiniRAG, and KG-Retriever, underperform on

![](images/e1cb5e900523539c2084c6ee4303a0046e245c2b5e6ca9c3a70c7edec657316e.jpg)  
(a) LongMemEval (Top-5)

![](images/9266c242987f77a7a2b053376391c0832fa4fe19f588c4d8af53fd414665c215.jpg)  
(b) LongMemEval (Top-10)

![](images/593244cb80585d1399571f7a17ce636477c4bad73f09aece99954fc8fd5bd8d4.jpg)  
(c) LoCoMo (Top-5)

![](images/7d18a83c754c448bf834501c0a2b75be85300c8145974863b63c401d3b419afc.jpg)  
(d) LoCoMo (Top-10)

![](images/eb42bfd13b56cd7d3b206f5a64ea71670c933bc10eafb6f7505102a19a270d4b.jpg)  
Figure 3: Performance comparison of RAG and SGMem variants on LongMemEval and LoCoMo under Top-5 and Top-10 settings. Turn-, round-, and session-level denote raw dialogue units; M, F, I denote summary, fact, and insight, respectively.   
(a) LongMemEval

![](images/817a2872ea808f2f13acfcc0eb7b50ba39715ea37f9c3dbb24c87d2d44276149.jpg)  
(b) LoCoMo   
Figure 4: QA performance across various Query subsets on both datasets.

both benchmarks, due to their reliance on entity- or relation-level graphs that lack alignment with conversational granularity. RAG variants that retrieve turns, rounds, or sessions provide stronger results, with RAG-SMFI (sessions $^ +$ summaries, facts, insights) achieving the best performance among RAG-based baselines (0.676/0.680 on LongMemEval and 0.510/0.528 on LoCoMo).

In contrast, SGMem consistently outperforms all RAG-based baselines. SGMem-SF achieves 0.690/0.730 on LongMemEval and 0.522/0.542 on LoCoMo, while SGMem-SMFI further improves to 0.700 and 0.526 at Top-5, respectively—the best RAG-based results across both datasets. These improvements demonstrate the advantage of representing dialogue at the sentence level and explicitly modeling associations via graph structures, which mitigates memory fragmentation and enables more coherent retrieval. Overall, in response to RQ1, SGMem consistently outperforms existing memory management and RAG baselines on both LongMemEval and LoCoMo, establishing a new paradigm for long-term conversational QA.

# 5.2 IMPACT OF CONTEXT TYPE

To addresss RQ2, we analyze variants of SGMem and RAG that use turns, rounds, sessions, summaries, facts, and insights as context, and investigate the trade-off between fidelity (raw history) and conciseness (generated memory). Figure 3 compares RAG and SGMem variants under different context settings. We observe that using only raw dialogue units (turns, rounds, sessions) provides a faithful but fragmented context, often yielding limited gains. For instance, turn-level RAG (RAG-T) performs the weakest, while session-level RAG (RAG-S) achieves stronger accuracy, showing the importance of larger dialogue spans. In contrast, incorporating generated memory—summaries, facts, and insights—substantially improves performance across all granularities. Variants such as RAG-SF and RAG-SMFI outperform their raw-only counterparts, confirming that generated memory enhances relevance and reduces noise. SGMem further amplifies these benefits by leveraging sentence-level graphs: SGMem-SF and SGMem-SMFI consistently surpass corresponding RAG variants, indicating that sentence-level associations help align raw dialogue with generated memory. Overall, in response to RQ2, the results highlight that combining raw dialogue with generated memory is essential for effective retrieval, and that SGMem provides a principled way to integrate both.

# 5.3 PERFORMANCE ACROSS QUERY TYPES

Figure 4 presents accuracy results across query subsets for both LongMemEval and LoCoMo. SGMem consistently outperforms RAG variants across all query types, confirming its robustness

![](images/7271c33d2b0be1543a675740d5b83d3c4a7255adc0b23d34bc333c2cdbe42304.jpg)  
(a) Hop h

![](images/8e40990c69c02a498035f10fdb1cf25614e9f2d2c5f85ed730bfa7f9aa6f040e.jpg)  
(b) KNN k

![](images/6038d2597c13428d4a5d12d4df2de404c1766bfe5b892e292b456eaf70acd50f.jpg)  
(c) Max Nodes n

![](images/8e5f87b4d553290c7a9958df4abc610d2d3f976e04d14a64deca52318fb643ff.jpg)  
Figure 5: Ablation studies on LongMemEval.   
(a) Hop h

![](images/6007508db072018522433ffc376f8cd36342dc50c5058cbc5af626235bc83e85.jpg)  
(b) KNN k

![](images/fde6ca4ccee8d1e93e9ca67f42b1e399a39a748167d693878ab5d1f8b5dce815.jpg)  
(c) Max Nodes n   
Figure 6: Ablation studies on LoCoMo.

under diverse conversational scenarios. On LongMemEval, we find that the largest improvements appear in multi-session, knowledge-update, and temporal reasoning queries, where effective reasoning requires bridging fragmented or evolving user information. Single-session queries (user and assistant) are relatively easier, where all methods achieve higher performance, but SGMem still maintains a margin over RAG baselines. On LoCoMo, SGMem also shows clear advantages across single-hop, temporal, and open-domain queries, highlighting its ability to capture both fine-grained details and long-range dependencies. Overall, in response to RQ3, these results demonstrate that sentence-graph memory provides consistent benefits across query types, particularly in settings that demand temporal tracking, multi-session integration, and adaptation to evolving knowledge.

# 5.4 IMPACT OF HYPERPARAMETERS

Figures 5 and 6 report ablation studies on LongMemEval and LoCoMo by varying hop $h$ , KNN size $k$ , maximum nodes $n$ , and weighting factor $\gamma$ . The results highlight that SGMem’s performance depends on careful tuning, and its optimal configurations differ across datasets. On LongMemEval, moderate values of $h$ and $k$ (e.g., $h = 1$ , $k = 3$ ) yield small gains, while extreme settings bring diminishing returns. Accuracy peaks around $n = 1 0$ , and $\gamma = 1 . 0$ provides slightly stronger results than larger scaling factors. Between retrievers for KNN, BM25 remains more stable across various $k$ , although the dense retriever (all-MiniLM-L6-v2) achieves stronger peak accuracy when tuned properly. On LoCoMo, increasing $h$ consistently degrades accuracy, and larger $k$ or $n$ often introduce noise. Compared with LongMemEval, LoCoMo benefits more from careful tuning, with $\gamma = 1 . 2$ producing relatively stable results. BM25 again demonstrates robustness under variation, though at the cost of slightly lower best-case accuracy compared to the dense retriever. Overall, in response to RQ4, these findings suggest that SGMem requires hyperparameter calibration to achieve optimal performance, with LongMemEval being more tolerant to variation, while LoCoMo demands more careful tuning due to its longer and noisier conversational histories.

# 6 CONCLUSION

Long-term conversational agents demand robust memory management to overcome the limitations of LLM context windows and support accurate, personalized responses. In this paper, we introduced SGMem, a sentence graph memory framework that organizes dialogue into sentence-level graphs, bridging raw dialogue histories and generated memory through explicit associations. By integrating turns, rounds, and sessions with summaries, facts, and insights, SGMem provides coherent and contextually grounded evidence for response generation. Experiments on LongMemEval and LoCoMo demonstrate that SGMem consistently outperforms strong baselines, yielding improvements across diverse query types and showing robustness to hyperparameter variations. These results highlight the effectiveness of sentence-level graph memory in mitigating fragmentation and redundancy, paving the way toward more scalable and reliable long-term conversational agents.

# LIMITATIONS

While SGMem demonstrates consistent improvements over strong baselines, several limitations remain. First, although SGMem effectively integrates raw dialogue and generated memory, it does not yet address hallucinations or factual inconsistencies that may arise from LLM-generated summaries, facts, or insights. Second, our evaluation is conducted on two benchmarks (LongMemEval and LoCoMo), which, despite their coverage of diverse query types, may not fully capture the breadth of real-world conversational dynamics such as multimodal contexts, streaming updates, or highly personalized long-term memory. Finally, SGMem has not been optimized for efficiency at scale; constructing and maintaining sentence-level graphs over very large histories may incur additional computational and storage overhead. Future work could explore fact-verification mechanisms, multimodal extensions, and scalable graph maintenance to further enhance the reliability and applicability of SGMem.

# ETHICS STATEMENT

This work introduces SGMem, a sentence-level graph memory framework for long-term conversational agents. Our experiments are conducted exclusively on publicly available datasets that do not contain personally identifiable information or sensitive content. The proposed SGMem aims to improve accuracy in memory management without altering or fabricating raw dialogue content. While our approach does not involve direct human subjects, we note that summaries, facts, and insights are generated using large language models (LLMs), which may introduce biases or inaccuracies inherent to the models. We therefore encourage practitioners to apply SGMem responsibly, ensuring that both the raw dialogue histories and LLM-generated content are handled with appropriate safeguards to protect user privacy, mitigate bias amplification, and prevent misuse in deployment.

# REPRODUCIBILITY STATEMENT

We have made careful efforts to ensure the reproducibility of SGMem. All datasets used in our experiments are publicly available and described in detail in Section 4 and Appendix A. The construction of sentence-level graph memory, retrieval configurations, evaluation metrics, and prompting strategies are fully specified in the paper and appendix. Hyperparameters for all SGMem variants are reported. Collectively, these details should enable independent researchers to replicate our results without reliance on proprietary resources.

# REFERENCES

Sonke Ahrens. ¨ How to take smart notes: One simple technique to boost writing, learning and thinking. Sonke Ahrens, 2022.¨   
Michele Bedard-Gilligan and Lori A Zoellner. Dissociation and memory fragmentation in posttraumatic stress disorder: An evaluation of the dissociative encoding hypothesis. Memory, 20(3): 277–299, 2012.   
Steven Bird. Nltk: the natural language toolkit. In Proceedings of the COLING/ACL 2006 interactive presentation sessions, pp. 69–72, 2006.   
Weijie Chen, Ting Bai, Jinbo Su, Jian Luan, Wei Liu, and Chuan Shi. Kg-retriever: Efficient knowledge indexing for retrieval-augmented large language models. arXiv preprint arXiv:2412.05547, 2024.   
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.   
Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao Huang. Minirag: Towards extremely simple retrieval-augmented generation. arXiv preprint arXiv:2501.06713, 2025.

Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. A survey on rag meeting llms: Towards retrieval-augmented large language models. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pp. 6491–6501, 2024.   
Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024.   
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrievalaugmented generation. arXiv preprint arXiv:2410.05779, 2024.   
Bernal Jimenez Guti ´ errez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory: ´ Non-parametric continual learning for large language models. arXiv preprint arXiv:2502.14802, 2025.   
Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al. Retrieval-augmented generation with graphs (graphrag). arXiv preprint arXiv:2501.00309, 2024.   
Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically inspired long-term memory for large language models. Advances in Neural Information Processing Systems, 37:59532–59569, 2024.   
Bowen Jin, Jinsung Yoon, Jiawei Han, and Sercan O Arik. Long-context llms meet rag: Overcoming challenges for long inputs in rag. arXiv preprint arXiv:2410.05983, 2024.   
David Kadavy. Digital Zettelkasten: Principles, Methods, & Examples. Kadavy, Inc., 2021.   
Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. Memory os of ai agent. arXiv preprint arXiv:2506.06326, 2025.   
Merel Kindt and Marcel Van Den Hout. Dissociation and memory fragmentation: Experimental effects on meta-memory but not on actual memory performance. Behaviour Research and Therapy, 41(2):167–178, 2003.   
Torkel Klingberg. The overflowing brain: Information overload and the limits of working memory. Oxford University Press, 2009.   
Hao Li, Chenghao Yang, An Zhang, Yang Deng, Xiang Wang, and Tat-Seng Chua. Hello again! llm-powered personalized agent for long-term dialogue. arXiv preprint arXiv:2406.05925, 2024.   
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. arXiv preprint arXiv:2402.17753, 2024.   
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Xufang Luo, Hao Cheng, Dongsheng Li, Yuqing Yang, Chin-Yew Lin, H Vicky Zhao, Lili Qiu, et al. Secom: On memory construction and retrieval for personalized conversational agents. In The Thirteenth International Conference on Learning Representations, 2025.   
Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.   
Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084, 2019.   
Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389, 2009.

Ranjan Sapkota, Konstantinos I Roumeliotis, and Manoj Karkee. Ai agents vs. agentic ai: A conceptual taxonomy, applications and challenges. arXiv preprint arXiv:2505.10468, 2025.   
Shamane Siriwardhana, Rivindu Weerasekera, Elliott Wen, Tharindu Kaluarachchi, Rajib Rana, and Suranga Nanayakkara. Improving the domain adaptation of retrieval augmented generation (rag) models for open domain question answering. Transactions of the Association for Computational Linguistics, 11:1–17, 2023.   
Zhen Tan, Jun Yan, I Hsu, Rujun Han, Zifeng Wang, Long T Le, Yiwen Song, Yanfei Chen, Hamid Palangi, George Lee, et al. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. arXiv preprint arXiv:2503.08026, 2025.   
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval: Benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813, 2024.   
Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, and Yong Liu. From human memory to ai memory: A survey on memory mechanisms in the era of llms. arXiv preprint arXiv:2504.15965, 2025.   
Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Jiliang Tang, Himabindu Lakkaraju, and Zhen Xiang. How memory management impacts llm agents: An empirical study of experience-following behavior. arXiv preprint arXiv:2505.16067, 2025.   
Derong Xu, Yi Wen, Pengyue Jia, Yingyi Zhang, Yichao Wang, Huifeng Guo, Ruiming Tang, Xiangyu Zhao, Enhong Chen, Tong Xu, et al. Towards multi-granularity memory association and selection for long-term conversational agents. arXiv preprint arXiv:2505.19549, 2025a.   
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025b.   
Li Yu, Tiancheng Qin, Qingxu Fu, Sen Huang, Xianzhe Xu, Zhaoyang Liu, and Boyin Liu. MemoryScope, 09 2024. URL https://github.com/modelscope/MemoryScope.   
Richard J Yun, John H Krystal, and Daniel H Mathalon. Working memory overload: fronto-limbic interactions and effects on subsequent working memory function. Brain imaging and behavior, 4 (1):96–108, 2010.   
Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Hao Chen, Yilin Xiao, Chuang Zhou, Yi Chang, et al. A survey of graph retrieval-augmented generation for customized large language models. arXiv preprint arXiv:2501.13958, 2025a.   
Yujie Zhang, Weikang Yuan, and Zhuoren Jiang. Bridging intuitive associations and deliberate recall: Empowering llm personal assistant with graph-structured long-term memory. In Findings of the Association for Computational Linguistics: ACL 2025, pp. 17533–17547, 2025b.   
Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model-based agents. ACM Transactions on Information Systems, 43(6):1–47, 2025c.   
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19724–19731, 2024.

# A DATASET STATISTICS

LongMemEval. LongMemEval (Wu et al., 2024) is a large-scale benchmark designed to evaluate five core memory abilities of LLM-based chat assistants: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. It contains 500 curated questions embedded in user–assistant dialogues of varying length and complexity. Each question is annotated with its type and aligned with corresponding sessions that provide supporting evidence. The distribution of question types includes: 70 single-session-user, 56 single-session-assistant, 30 single-session-preference, 133 multi-session, 78 knowledge-update, and 133 temporal-reasoning. For example, a single-session-user question is:

```txt
Question: What degree did I graduate with?  
Answer: Business Administration  
Question.Type: single-session-user  
Question.Date: 2023/05/30 (Tue) 23:40  
Evidence: Session ID [answer_280352e9] 
```

This benchmark presents a challenging setting where existing long-context LLMs and commercial chat assistants show significant accuracy degradation when information must be recalled across extended interactions.

LoCoMo. LoCoMo (Maharana et al., 2024) is a very long-term conversational benchmark generated via a machine–human pipeline that grounds multi-session dialogues on personas and temporal event graphs. Each conversation averages 300 turns and 9K tokens across up to 35 sessions, and some dialogues incorporate multimodal interactions (e.g., image sharing and reactions). To ensure computational feasibility, we randomly sample 500 questions from the full set of 1,986 annotated questions. These questions are distributed across four categories: 156 single-hop, 133 multi-hop, 133 temporal reasoning, and 78 open-domain knowledge. For example, a temporal reasoning question is:

```txt
Question: Which country was Jolene located in during the last week of August 2023?  
Answer: Brazil  
Question.Type: temporal reasoning  
Evidence: Dialogue ID [D23:1] 
```

Experiments on LoCoMo highlight the difficulty of modeling long-range temporal and causal dynamics, where long-context LLMs and RAG systems still lag behind human performance.

# B PROMPTS

To facilitate reproducibility, we provide the full set of prompts used in our experiments. These include the RESPONSE PROMPT (Appendix B.1), EVALUATION PROMPT (Appendix B.2), SUMMARY PROMPT (Appendix B.3), FACT PROMPT (Appendix B.4), and INSIGHT PROMPT (Appendix B.5), which are designed for response generation, model evaluation, dialogue summarization, fact extraction, and insight generation, respectively.

# B.1 RESPONSE PROMPT

# Prompt B.1: Response Prompt

```txt
---Role---
```

```txt
You are a helpful assistant responding to questions about data provided.   
---Goal---   
Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data appropriate for the response length and format, and incorporating any relevant general knowledge.   
If you don't know the answer, just say so. Do not make anything up.   
Do not include information where the supporting evidence for it is not provided.   
---Target response length and format---   
Multiple Paragraphs   
Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown. 
```

# B.2 EVALUATION PROMPT

Prompt B.2: Evaluation Prompt   
---Role---   
You are a helpful evaluation assistant.   
You will be given a question, a gold-standard answer, and a candidate answer generated via retrieval-augmented generation (RAG).   
---Goal---   
Evaluate the candidate answer against the gold-standard answer based on factual accuracy and completeness in answering the question.   
Scoring Criteria:   
- score $= 1$ (Correct): The candidate answer is factually accurate and fully or reasonably paraphrases the gold-standard answer.   
- score $= 0$ (Incorrect): The candidate answer is factually incorrect, irrelevant, incomplete, or does not answer the question.   
---Output Format---   
Provide your evaluation in the following JSON format:   
``json { "score": X } ...   
where X is either 1 or 0.

# B.3 SUMMARY PROMPT

# Prompt B.3: Summary Prompt

---Role---

You are a helpful summarization assistant.

---Goal---

Please summarize the following dialogue as concisely as possible, extracting the main themes and key information. If there are multiple key events, you may summarize them separately.

# B.4 FACT PROMPT

# Prompt B.4: Fact Prompt

--- Role ---

You are a precise and helpful fact extraction assistant.

You will be given a list of conversation messages between a human user and an AI assistant.

--- Goal ---

Extract **all explicit personal facts** about the human user, including but not limited to:

- Life events (past, present, or planned)

- Personal experiences

- Preferences and interests

- Relationships and interactions with people

- Numbers, dates, locations, organizations, and other concrete details

Each extracted fact must:

1. Be a **standalone, self-contained sentence**.

2. Avoid pronouns (replace "I", "my", "she", "they" with explicit entities, e.g., "The user", "Maya", "Jake Watson").

3. Preserve all available details (time, place, quantity, frequency, etc.).

4. Remain strictly factual (do not infer, summarize, or speculate beyond the given text).

If no personal facts are found, output an empty list.

--- Output Format ---

Return the facts as a JSON list of strings, where each string is one fact:

‘‘‘json ["fact 1", "fact 2", "fact 3"] 1 1 ‘

# B.5 INSIGHT PROMPT

# Prompt B.5: Insight Prompt

```lua
--- Role ---   
You are a precise and helpful fact reflection assistant.   
You will be given a list of factual records about a human user.   
--- Goal ---   
Your task is to analyze the provided user memories and generate higher -level, insightful reflections.   
--- Analysis and Reflection Rules ---   
- Carefully read the memory entries and identify recurring themes, behaviors, or connections.   
- Multiple related memory entries may be \*\*merged into a single insight\*\* if they collectively represent a pattern or habit (e.g., repeated actions --> user habit).   
- The 'timestamp' of the generated insight should be the \*\*latest timestamp\*\* among the merged memory entries.   
- Reflect and summarize to generate higher-level insights such as user preferences, habits, routines, opinions, goals, or current status.   
- Each insight should be concise, self-contained, and written as a standalone statement.   
- Do not copy the input facts directly; instead, abstract them into meaningful patterns or insights.   
- Do not include any explanations, metadata, or comments outside of the JSON output..   
--- Input Format ---   
A list of user memories, each containing a timestamp and content: [ {'timestamp':'', 'content':}}, {'timestamp':'', 'content':}}, {'timestamp':'', 'content':}}, ... ]   
--- Output Format ---   
Return the insights as a \*\*JSON list of objects\*. Each object must have: - 'timestamp': the latest timestamp among the related memory entries - 'content': the generated insight. ['json [{'timestamp':'', 'content':}}, {'timestamp':'', 'content':}}, ... ] 
```

# C CASE STUDIES

We provide case studies on both LongMemEval (Appendix C.1) and LoCoMo (Appendix C.2) to compare RAG-SMFI with SGMem-SMFI. The examples demonstrate that RAG-SMFI often fails

to answer questions correctly based on retrieved memory alone, whereas SGMem-SMFI leverages multi-hop traversal within SGMem to locate relevant chunks (i.e., session-level information) and complement them with generated memory, leading to correct answers.

# C.1 A CASE STUDY ON LONGMEMEVAL

RAG-SMFI on LongMemEval. We present an illustrative case of long-term conversational QA evaluation with RAG-SMFI on LongMemEval, as shown in Use Case C.1. In this example, RAG-SMFI fails to provide the correct answer.

# Use Case C.1: RAG-SMFI on LongMemEval

```jsonl
{   
"question_id": "0bc8ad92",   
"question_type": "temporal-reasoning",   
"question_date": "2023/03/25 (Sat) 17:18",   
"question": "How many months have passed since I last visited a museum with a friend?",   
"answer": 5,   
"answer_session_ids": ["answer_f4ea84fb_3", "answer_f4ea84fb_2", " answer_f4ea84fb_1"],   
"results": {   
"response": "Based on the information provided, your last visit to a museum with a friend occurred on a recent date, specifically a behind-the-scenes tour of the Science Museum with a chemistry professor friend. This visit was referenced in the context of the user's recent museum visits, which included a guided tour at the Natural History Museum on February 18, 2023.\nSince the timestamp for the Natural History Museum visit is February 18, 2023, and the current timestamp is March 25, 2023, approximately one month has passed since your last mentioned museum visit.\nHowever, if you specifically visited the Science Museum with your chemistry professor friend after the Natural History Museum visit, please clarify the exact date of that visit. If no specific date was given for the Science Museum visit, we can assume it was within the same timeframe, meaning about one month has passed since your last museum visit.",   
"score": 0   
} 
```

SGMem-SMFI on LongMemEval. Use Case C.2 illustrates a long-term conversational QA evaluation with our proposed SGMem-SMFI on LongMemEval, where the model provides the correct answer.

# Use Case C.2: SGMem-SMFI on LongMemEval

```txt
{   
"question_id": "0bc8ad92",   
"question_type": "temporal-reasoning",   
"question_date": "2023/03/25 (Sat) 17:18",   
"question": "How many months have passed since I last visited a museum with a friend?",   
"answer": 5,   
"answer_session_ids": ["answer_f4ea84fb_3", "answer_f4ea84fb_2", " answer_f4ea84fb_1"],   
"results": {   
"topksentence_ids": ["answer_f4ea84fb_1_9_12", "answer_f4ea84fb_3_5_0 ", "answer_f4ea84fb_3_5_14", "answer_f4ea84fb_1_11_10", " 
```

```jsonl
answer_f4ea84fb_3_1_4", "answer_f4ea84fb_3_5_15", "answer_f4ea84fb_1_8_0", "answer_f4ea84fb_3_1_3", "answer_f4ea84fb_3_4_0", "answer_f4ea84fb_3_7_4", "answer_f4ea84fb_3_0_0", "answer_f4ea84fb_1_9_0", "answer_f4ea84fb_1_10_1", "answer_f4ea84fb_3_7_2", "answer_f4ea84fb_3_6_3"], "chunk_to_speeches": {"answer_f4ea84fb_3": ["answer_f4ea84fb_3_1_5"], "answer_f4ea84fb_3_8_0", "answer_f4ea84fb_3_1_4", "answer_f4ea84fb_3_7_2", "answer_f4ea84fb_3_7_4", "answer_f4ea84fb_3_1_3", "answer_f4ea84fb_3_5_0", "answer_f4ea84fb_3_7_10", "answer_f4ea84fb_3_5_15", "answer_f4ea84fb_3_0", "answer_f4ea84fb_3_6_3", "answer_f4ea84fb_3_1_10", "answer_f4ea84fb_3_7_7", "answer_f4ea84fb_3_5_14", "answer_f4ea84fb_3_4_0", "answer_f4ea84fb_3_7_5"], "answer_f4ea84fb_1": ["answer_f4ea84fb_1_10_1"], "answer_f4ea84fb_1_9_13", "answer_f4ea84fb_1_9_12", "answer_f4ea84fb_1_10_2", "answer_f4ea84fb_1_11_10"], }
"topk_chunk_ids": ["answer_f4ea84fb_1", "answer_f4ea84fb_3"], "response": "Based on the information provided, your last visit to a museum with a friend occurred on October 22, 2022. Given the current timestamp of March 25, 2023, approximately five months have passed since your last museum visit with your friend. \n\nCurrent date: March 25, 2023\n\nMonths since your last museum visit with your chemistry professor friend. If you're considering another visit soon, you might want to check the museums' schedules and any new exhibits that may have opened in the meantime.", 
"score": 1
} 
```

# C.2 A CASE STUDY ON LOCOMO

RAG-SMFI on LoCoMo. Use Case C.3 illustrates a long-term conversational QA evaluation with RAG-SMFI on LoCoMo, where the model fails to provide the correct answer.

# Use Case C.3: RAG-SMFI on LoCoMo

```txt
{   
"question_id": "conv-47",   
"question_type": "temporal reasoning",   
"question": "When did James depart for his trip to Canada?",   
"answer": "July 11, 2022",   
"answer Dialogue_ids": ["D16:9"],   
"answer_session_ids": ["conv-47_s15"],   
"results": {   
"response": "Based on the information provided, James bought air tickets to Toronto and planned to visit Vancouver as well. However, the exact departure date for his trip to Canada is not specified in the timestamps provided. The only dates mentioned are related to his plans to return from the trip on July 20 and a discussion about his travel plans taking place at 5:13 pm on July 9, 2022. Therefore, we can infer that James likely departed sometime after July 9, 2022, but before July 20, 2022, the date he intended to return. Without additional information, the precise departure date cannot be determined."}, 
```

```txt
"score": 0} 
```

SGMem-SMFI on LoCoMo. As shown in Use Case C.4, our proposed SGMem-SMFI successfully answers the long-term conversational QA task on LoCoMo.

# Use Case C.4: SGMem-SMFI on LoCoMo

{
    "question_id": "conv-47",
    "question_type": "temporal reasoning",
    "question": "When did James depart for his trip to Canada?", 
    "answer": "July 11, 2022",
    "answer Dialogue_ids": ["D16:9"]
    "answer_session_ids": ["conv-47_s15"]
    "results": {
        "topksentence_ids": ["conv-47_s15_10_0", "conv-47_s6_9_0", "conv-47_s5_13_0", "conv-47_s29_2_0", "conv-47_s5_11_0", "conv-47_s30_24_0",
        "conv-47_s15_6_0", "conv-47_s27_34_0", "conv-47_s19_21_0", "conv-47_s16_35_0", "conv-47_s20_10_0", "conv-47_s5_15_0", "conv-47_s27_20_0", "conv-47_s16_15_0", "conv-47_s12_9_0"]
        "chunk_to_sentences": {"conv-47_s13": ["conv-47_s13_0_0"], "conv-47_s22": ["conv-47_s22_20_0"], "conv-47_s10": ["conv-47_s10_17_0"], "conv-47_s19": ["conv-47_s19_21_0"], "conv-47_s27": ["conv-47_s27_24_0"], "conv-47_s27_34_0", "conv-47_s27_20_0"], "conv-47_s20": ["conv-47_s20_10_0"], "conv-47_s16": ["conv-47_s16_15_0", "conv-47_s16_35_0", "conv-47_s16_23_0"], "conv-47_s15": ["conv-47_s15_6_0",
        "conv-47_s15_10_0"], "conv-47_s29": ["conv-47_s29_2_0"], "conv-47_s12": ["conv-47_s12_9_0"], "conv-47_s6": ["conv-47_s6_9_0"], "conv-47_s8": ["conv-47_s8_21_0"], "conv-47_s5": ["conv-47_s5_15_0", 
        "conv-47_s5_11_0", "conv-47_s5_17_0", "conv-47_s5_13_0"], "conv-47_s17": ["conv-47_s17_15_0"], "conv-47_s30": ["conv-47_s30_24_0"], 
        "conv-47_s0": ["conv-47_s0_11_0"], "conv-47_s26": ["conv-47_s26_13_0",
        ], "conv-47_s11": ["conv-47_s11_6_0"]
    }, 
    "topk_chunk_ids": ["conv-47_s6", "conv-47_s15", "conv-47_s29", "conv
            -47_s5", "conv-47_s30'],
    "response": "Based on the information provided, James departed for his
    trip to Canada the day after the conversation he had with John on
    July 9, 2022. Since their conversation took place at 5:13 pm on
    July 9, 2022, and James mentioned that he was leaving \("the day
    after tomorrow evening,"\) his departure would have been on July 11,
    2022, in the evening. His planned return date from the trip was
    July 20, 2022.", 
    "score": 1
}
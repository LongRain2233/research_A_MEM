![](images/c2405ed39f8040abeb03d1117b60faf581fedbb665f58784a3e229bd32ea9889.jpg)

# Youtu-GraphRAG: Vertically Unified Agents for Graph Retrieval-Augmented Complex Reasoning

Junnan Dong1†, Siyu $\pmb { R } \pmb { n } ^ { 1 \dag \ddag }$ , Yifei $y _ { \pmb { U } } 1$ , Qian-Wen Zhang1, Linhao Luo2, Xiao Huang3, Yunsheng Wu1, Di Yin1,Xing Sun1

1Tencent Youtu Lab

2Monash University

3The Hong Kong Polytechnic University

Graph retrieval-augmented generation (GraphRAG) has effectively enhanced large language models in complex reasoning by organizing fragmented knowledge into explicitly structured graphs. Prior efforts have been made to improve either graph construction or graph retrieval in isolation, yielding suboptimal performance, especially when domain shifts occur. In this paper, we propose a vertically unified agentic paradigm, Youtu-GraphRAG, to jointly connect the entire framework as an intricate integration. Specifically, (i) a seed graph schema is introduced to bound the automatic extraction agent with targeted entity types, relations and attribute types, also continuously expanded for scalability over unseen domains; (ii) To obtain higher-level knowledge upon the schema, we develop novel dually-perceived community detection, fusing structural topology with subgraph semantics for comprehensive knowledge organization. This naturally yields a hierarchical knowledge tree that supports both top-down filtering and bottom-up reasoning with community summaries; (iii) An agentic retriever is designed to interpret the same graph schema to transform complex queries into tractable and parallel sub-queries. It iteratively performs reflection for more advanced reasoning; (iv) To alleviate the knowledge leaking problem in pre-trained LLM, we propose a tailored anonymous dataset and a novel ‘Anonymity Reversion’ task that deeply measures the real performance of the GraphRAG frameworks. Extensive experiments across six challenging benchmarks demonstrate the robustness of Youtu-GraphRAG, remarkably moving the Pareto frontier with up to $9 0 . 7 1 \%$ saving of token costs and $1 6 . 6 2 \%$ higher accuracy over state-of-the-art baselines. The results indicate our adaptability, allowing seamless domain transfer with minimal intervention on schema.

![](images/e8cbbde18f039c7cd5a16a1797b2cdedff8f63efe0c76ca8c000945bcc3ba427.jpg)

Code: https://github.com/TencentCloudADP/Youtu-GraphRAG

![](images/dc4740c1fe7b898e33e530336524bb0f0da9650fa1a807268844d6512e3b83a5.jpg)

Data: https://huggingface.co/datasets/Youtu-Graph/AnonyRAG

# 1 Introduction

Graph retrieval-augmented generation (GraphRAG) has emerged as a promising paradigm to enhance large language models (LLMs) with structured knowledge [Xiao et al., 2025, Pan et al., 2024], particularly for complex multi-hop reasoning tasks across multiple documents[Wang et al., 2024, Zhang et al., 2024]. By representing fragmented documents as connected graphs with underlying relations [He et al., 2024, Dong et al., 2023], GraphRAG enables LLMs to traverse explicit paths among documents and entities, performing complex reasoning that is otherwise infeasible within flat retrieval [Peng et al., 2024, Han et al., 2024]. The structured approach effectively addresses critical limitations in conventional RAG ([Dong et al., 2024]), which often struggles with the coherent relations between discrete pieces of information and multi-hop reasoning.

The evolution of GraphRAG brings two distinct but equally important trajectories since the foundational work of [Edge et al., 2024]. First, from the retrieval front, LightRAG [Guo et al., 2024] pioneered vector sparsification to improve efficiency. While GNN-RAG and GFM-RAG ([Mavromatis and Karypis, 2024, Luo et al., 2025]) advanced this direction further by incorporating graph neural networks for fine-grained node matching, more recent HippoRAG 1&2 [Jimenez Gutierrez et al., 2024, Gutiérrez et al., 2025] introduced

![](images/6a7d6c95a5a189fb644ddf723b668776d5963ad4599d813f0f784bd70df18b10.jpg)  
Figure 1. A sketched comparison among existing pipelines and Youtu-GraphRAG. 自 represents a non-tailored component, indicating current methods focus on either graph construction (a) or retrieval (b) in isolation, while Youtu-GraphRAG proposes a unified paradigm (c) for superior complex reasoning.

memory and personalized PageRank algorithms for contextaware retrieval. Second, in terms of graph construction, existing methods can be broadly categorized into flat and hierarchical approaches. Early methods, such as KGP [Wang et al., 2024], rely on existing hyperlinks or KNN-based graphs, resulting in coarse-grained relations that fail to capture nuanced hierarchical semantics. More recent advancements, such as GraphRAG [Edge et al., 2024], combine knowledge graphs with community detection and summarization for multi-level information. Followed by hierarchical methods like RAPTOR [Sarthi et al., 2024] and $\mathrm { E } ^ { \breve { 2 } }$ GraphRAG [Zhao et al., 2025], they further refine the graph using tree-like clustering and recursive summarization to enrich structural representation. However, they remain constrained by their isolated optimizations, concentrating on either construction or retrieval while neglecting their interdependencies. This potentially limits complex reasoning performance where cohesive knowledge organization and retrieval are equally important.

To bridge this gap, we aim to answer a critical question: How can we effectively unify graph construction and retrieval for more robust complex reasoning?

This task is challenging for two reasons. First, construction and retrieval are not readily aligned as two distinct components. It remains difficult to organically establish synergy between them, where the constructed graph could effectively benefit retrieval with both structures and semantics. Second, how to properly evaluate the performance remains a tough problem. With the rapid scaling of LLMs, almost all the existing datasets have already been ‘seen’ before. This fails to reflect the real performance of the entire GraphRAG.

In this paper, we propose a vertically unified agentic paradigm, Youtu-GraphRAG, to jointly consider both graph construction and retrieval as an intricate integration based on graph schema. To be specific, (i) a graph schema is introduced to bound the extraction agent that ensures the quality and conciseness with targeted entity types, relations and attribute types; The seed schema is continuously and automatically expanded based on the feedback. (ii) To obtain higher-level knowledge upon the schema, we develop duallyperceived community detection, fusing structural topology with subgraph semantics for comprehensive knowledge clustering. This naturally yields a hierarchical knowledge tree that supports both top-down filtering and bottom-up reasoning with community summaries; (iii) An agentic retriever is designed to interpret the same graph schema to transform complex queries into parallel sub-queries and perform iterative reflection. The agent iteratively performs both reasoning and reflection for more advanced performance; (iv) To alleviate the knowledge leaking problem in pre-trained LLM, we first propose a tailored anonymous dataset with an ‘Anonymity Reversion’ task. Extensive experiments across six challenging benchmarks demonstrate the robustness of Youtu-GraphRAG, remarkably moving the Pareto frontier with up to $9 0 . 7 1 \%$ saving of token consumption and $1 6 . 6 2 \%$ higher accuracy over SOTA baselines. The results also indicate our remarkable adaptability which allows seamless domain transfer with minimal intervention on the graph schema, providing insights of the next evolutionary GraphRAG paradigm for real-world applications.

Contributions. In general, our primary contributions are summarized hereunder:

• We first propose a vertically unified Agentic GraphRAG framework to integrate graph construction and retrieval for more robust and advanced reasoning, where both construction and retrieval agents are bounded by graph schema for effective extraction and query decomposition, respectively;   
• A novel theoretically-grounded community detection algorithm is employed to inject high-level sum-

![](images/979222b88a5b10862bc97005569738b1920ca02a002a91dae9317282cd9b1eeb.jpg)  
Figure 2. A toy overview of Youtu-GraphRAG that unifies graph construction and retrieval through a schema-guided agentic paradigm. (i) An extraction agent automatically processes documents into structured knowledge via targeted entity/relation extraction; (ii) A four-level knowledge tree is constructed upon the schema with a community detection that fuses topological structures and graph semantics, enabling hierarchical reasoning; (iii) A retrieval agent decomposes user queries into parallel sub-queries aligned with the schema, iteratively driving multi-route retrieval.

marization upon graph schema, simultaneously preserving structural and semantic graph properties;

• We present a tailored anonymous dataset and ‘Anonymous Revertion’ task is proposed to prevent LLM knowledge leaking for fair evaluation of the GraphRAG performance;   
• Extensive empirical experiments are conducted over five challenging benchmarks, showing state-of-the-art performance across diverse reasoning tasks and domains that moves the Pareto frontier with up to $9 0 . 7 1 \%$ saving of token costs and $1 6 . 6 2 \%$ higher accuracy.

# 2 Task Definition

In this section, we formally define the general GraphRAG pipeline with standardized notations from scratch, including both graph construction and graph retrieval. We denote scalars as lowercase alphabets (e.g., a), vectors as boldface lowercase alphabets (e.g., a), matrices as boldface uppercase alphabets (e.g., A) and copperplate for a set of elements (e.g., $\mathcal { A }$ ). We refer to GraphRAG as the task of answering a natural language question by first retrieving structured knowledge from a corpus and then generating a response.

Given a set of documents $\mathcal { D }$ , GraphRAG first leverages a frozen LLM fLLM(·) to extract important knowledge, connected by a structured graph $\mathcal { G }$ as output. To enrich the understanding of ${ \mathcal { G } } ,$ a community detection algorithm $f _ { \mathrm { c o m m } } ( \mathcal { G } )$ is employed to partition $\mathcal { G }$ into communities $\mathcal { C } = \{ \mathcal { C } _ { 1 } , \mathcal { \bar { C } } _ { 2 } \ldots \mathcal { C } _ { m } \}$ to obtain higher-level summarizations. Based on the constructed graph $\mathcal { G }$ , given a complex query $q \in \mathcal { Q } ,$ a retrieval model fretrieve(q, G) = arg max $\mathcal { P } ( \mathcal { G } _ { \mathrm { s u b } } \mid \mathbf { q } )$ traverses the graph and retrieves top- $\mathbf { \nabla } \cdot k$ question-specific subgraphs $\mathcal { G } _ { s u b } \subseteq \mathcal { G }$ that maximize the similarity with given query $q$ . The final performance is evaluated from multiple aspects: (i) graph construction costs including time efficiency and token consumptions; $( i i )$ retrieval accuracy and efficiency; and $( i i i )$ final answer accuracy comparing $a _ { \mathrm { p r e d } }$ and ground-truths $a _ { \mathrm { g o l d } }$ .

# 2.1 Construction Stage

Beginning with the documents $\mathcal { D }$ as corpus, contemporary GraphRAG research includes two synergistic knowledge organizations that form the graph $\mathcal { G }$ at different granularities. First, the fine-grained graph $\mathcal { G } _ { \mathrm { t r i p l e } } = \left( \mathcal { E } , \mathcal { R } , \mathcal { D } \right)$ is constructed by using $f _ { \mathrm { L L M } } ( \mathcal { D } )$ to extract atomic units in the form of triples $\left( h , r , t \right)$ from each document $d \in \mathcal { D }$ , where entities $\{ h , t \} \in { \mathcal { E } }$ and relations $r \in \mathcal { R }$ are explicitly linked to represent the abundant relational information among them. The extraction is performed by the frozen LLM $f _ { \mathrm { L L M } } ( d )$ , which processes raw text to populate $\mathcal { G } _ { \mathrm { t r i p l e } }$ with schema-compliant triples. Concurrently in another pipeline of research, a coarse-grained document graph $\mathcal { G } _ { \mathrm { d o c } } = ( \mathcal { D } , \mathcal { C } )$ is built by directly clustering documents to maximally preserve the raw context where the atomic units are documents instead of triples. To obtain higher-level knowledge, a complementary community detection algorithm $f _ { \mathrm { c o m m } } ( \mathcal { G } )$ is employed. Typically, $f _ { \mathrm { c o m m } } ( \mathcal { G } ) .$ , including Louvain, Leiden, GMM [Traag et al., 2019, Sarthi et al., 2024], etc., operates over $\mathcal { G }$ with

sufficient summaries and abstracts generated by $f _ { \mathrm { L L M } } ( d ) ,$ and results in communities $\mathcal { C } = \{ \mathcal { C } _ { 1 } , \mathcal { C } _ { 2 } \ldots \mathcal { C } _ { m } \}$ $\mathcal { C } _ { i } \subseteq \mathcal { G }$ is further summarized into a high-level meta-node $\hat { e } _ { i } = f _ { \mathrm { L L M } } ( \mathcal { C } _ { i } )$ by $f _ { \mathrm { L L M } } ( \mathcal { C } )$ where $\hat { e } _ { i } \in \mathcal G$ . The performance is evaluated by the time used tconstruct and the token consumptions $\$ 9$ during construction.

# 2.2 Retrieval Stage

During inference, given a query $q \in \mathcal { Q } ,$ the typical retrieval model $f _ { \mathrm { r e t r i e v e } } ( q , \mathcal { G } ) = \arg \operatorname* { m a x } _ { \mathbf { \mathcal { P } } } \mathcal { P } ( d \mid \mathbf { q } )$ ${ \mathcal { P } } ( d \mid { \mathfrak { q } } )$ directly returns the top- $k$ similar documents $\hat { \mathcal { D } } = \{ \hat { d } _ { 1 } , d _ { 2 } \cdot \cdot \cdot d _ { k } \}$ as the final answer, while graph-based methods provide a more explainable subgraph $\hat { \mathcal { G } }$ for multi-hop path traversal, i.e., $f _ { \mathrm { r e t r i e v e } } ( q , \dot { \mathcal { G } } ) \overset { \cdot } { = } \arg \operatorname* { m a x } _ { { \mathbf { \theta } } } \mathcal { P } ( \mathcal { G } \mid { \mathbf { q } } )$ ${ \mathcal { P } } ( { \hat { \mathcal { G } } } \mid { \mathbf { q } } )$ where $\hat { \mathcal { G } } = \{ e _ { 0 } \overset { r _ { 1 } } { \longrightarrow } e _ { 1 } \overset { r _ { 2 } } { \longrightarrow } \cdots \overset { r _ { k } } { \longrightarrow } e _ { k } \} \in \mathcal { G }$ . Based on the retrieved subgraph, $f _ { \mathrm { L L M } } ( \boldsymbol { q } , \hat { \mathcal { G } } )$ is employed to generate the final answer. The final performance is evaluated holistically by the retrieval recall comparing $\stackrel { \sim } { \mathcal { G } } _ { \mathrm { d o c } }$ and ground truth documents $\hat { \mathcal { A } } _ { \mathrm { g o l d } } ^ { \mathrm { d o c } }$ and answer accuracy by comparing between $a _ { \mathrm { p r e d } }$ and $a _ { \mathrm { g o l d } }$ .

# 3 Approach: Youtu-GraphRAG

In this section, we elaborate on the core methodology of Youtu-GraphRAG, designed to answer two fundamental research questions: (i) How to achieve unified optimization of graph construction and retrieval for higher robustness and generalizability? (ii) How could we enable effective reasoning across different knowledge granularities? Correspondingly, our framework integrates three designs in a vertically unified manner based on graph schema. First, a graph schema-bounded agent is designed to ensure construction quality while eliminating noise through automatic expansion. Second, beyond the schema, we present a dual-perception community detection that jointly analyzes both topological and semantic similarity to create multi-scale knowledge clusters which forms a four-level knowledge tree. Finally, an agentic retriever is designed to effectively decompose questions into schema-aligned atomic sub-queries with parallel retrieval routes and iterative reflection.

# 3.1 Schema-Bounded Agentic Extraction

Existing GraphRAG methods leverage either pure LLMs or OpenIE ([Jimenez Gutierrez et al., 2024, Gutiérrez et al., 2025, Luo et al., 2025, Edge et al., 2024]) for named entity recognition and triple extraction. However, this open-ended approach would inevitably introduce noise and irrelevant trivia, thereby reducing the usability of the graph. Instead, we treat graph extraction as constrained generation based on a high-quality seed graph schema for domain-specific tasks and define a compact schema as

$$
\mathcal {S} \triangleq \left\langle \mathcal {S} _ {e}, \mathcal {S} _ {r}, \mathcal {S} _ {\text {a t t r}} \right\rangle , \tag {1}
$$

where $ { \boldsymbol { S } } _ { e }$ indicates the targeted entity types (e.g., Person, Disease), $S _ { r }$ guides the extraction with condensed relations (e.g., treats, causes), and $\boldsymbol { S } _ { a t t r }$ lists attribute types that could be attached and used to describe any corresponding entities (e.g., occupation, gender). A frozen LLM-based agent $f _ { \mathrm { L L M } } ( S , { \mathcal { D } } )$ is bounded to identify matched information that appear in $\mathcal { S } _ { \varepsilon }$ , effectively reducing the search space to the Cartesian product $S _ { e } \times S _ { r } \times S _ { a }$ . Formally, for each document $d$ , we obtain a set of triples hereunder

$$
\mathcal {T} (d) = \left\{\left(h, r, t\right), \left(e, r _ {\text {a t t r}}, e _ {\text {a t t r}}\right) \mid \left\{f (h), f (t), f (e) \right\} \in \mathcal {S} _ {e}, \left\{r, r _ {\text {a t t r}} \right\} \in \mathcal {S} _ {r}, e _ {\text {a t t r}} \in \mathcal {S} _ {\text {a t t r}} \right\}. \tag {2}
$$

Therefore, in our paper, we define $\mathcal { G } _ { \mathrm { t r i p l e } } = ( \mathcal { E } , \mathcal { R } , \mathcal { D } ) _ { . }$ , where the entire entity set $\mathcal { E } = \{ \mathcal { E } _ { r } , \mathcal { E } _ { \mathrm { a t t r } } \}$ contains not only named entities $e$ but also the corresponding attributes $e _ { \mathrm { a t t r } }$ and the relation set $\mathcal { R }$ similarly contains both entity-entity relations $r$ and $r _ { \mathrm { a t t r } } ,$ , i.e., has_attribute relations to connect entities and attributes. However, a seed schema could be general and require manual efforts for predefinitions, which limits the scalability and adaptability of GraphRAG to unseen domains. We thereby equip the agent with an adding tool

![](images/e8486e14717903380aa1903acd5176eb8a2a3badc30ab43dc097640b31120347.jpg)  
Figure 3. An overview of our dually-perceived community detection. (a) Input graph partitioning into initial communities via triple embeddings; (b) community center identification through joint consideration of topology connectivity and subgraph semantic similarity; and (c) iterative pairwise community merging to form the final hierarchy. Distinct colors represent functionally coherent communities.

and incorporate an adaptive design that dynamically refines the initial schema $s$ through continuous interaction with the document content. The agent automatically proposes schema expansions by analyzing the underlying relational patterns in each document $d \in \mathcal { D }$ through the update function:

$$
\Delta \mathcal {S} = \left\langle \Delta \mathcal {S} _ {e}, \Delta \mathcal {S} _ {r}, \Delta \mathcal {S} _ {\text {a t t r}} \right\rangle = \mathbb {I} \left[ f _ {\mathrm {L L M}} (d, \mathcal {S}) \odot \mathcal {S} \right] \geq \mu , \tag {3}
$$

where ${ \mathbf { } } S ^ { ( t ) }$ represents the schema at iteration t, µ serves as a confidence threshold to control the acceptance of new schema elements. $\Delta \boldsymbol { S }$ contains candidate expansions for entity types, relations, and attributes, respectively. This dynamic adaptation enables the schema to evolve beyond its initial definitions while maintaining controlled growth, as the agent selectively incorporates only high-confidence patterns that demonstrate sufficient frequency and contextual consistency across documents in the new domain. Therefore, we expect the resulting schema to maintain its compact representation while gaining document-specific expressiveness, effectively balancing between strict schema guidance and flexible knowledge acquisition. Through this mechanism, our framework achieves more comprehensive knowledge coverage compared to static schema approaches, particularly when processing domains with emerging relational patterns.

# 3.2 Upon Schema: Graph Indexing with Knowledge Tree

The fine-grained raw graphs could quickly become extremely dense and noisy. Typically, a complementary community detection algorithm $f _ { \mathrm { c o m m } } ( \mathcal { G } )$ is employed to summarize the knowledge so as to reorganize the graph in communities $\mathcal { C } = \{ \mathcal { C } _ { 1 } , \mathcal { C } _ { 2 } \ldots \mathcal { C } _ { m } \}$ . Contemporary methods apply Louvain, Leiden, Gaussian Mixture Models (GMM) ([Traag et al., 2019, Sarthi et al., 2024]), etc., operates over $\mathcal { G }$ with sufficient summaries and abstracts generated by $f _ { \mathrm { L L M } } ( d )$ . $\mathcal { C } _ { i } \subseteq \mathcal { G }$ is further summarized into a high-level meta-node $\hat { e } _ { i } = f _ { \mathrm { L L M } } ( \mathcal { C } _ { i } )$ by $f _ { \mathrm { L L M } } ( \mathcal { C } )$ where $\hat { e } _ { i } \in \mathcal G$ .

However, the performance of existing community detection methods can hardly satisfy the real-world demands. They primarily rely on structural connectivity while largely neglecting the rich semantic information embedded in the relational context. As a result, they often produce suboptimal partitions in real-world knowledge graphs since both topological and semantic coherence are crucial for meaningful community detection. To address this limitation, we are motivated to propose a novel and revolutionary

dual-perception community detection framework that simultaneously optimizes for topological connectivity and semantic similarity through a three-stage optimization process. An illustration is shown in Figure 3. The final output is compressed into a Knowledge Tree $\kappa$ of depth $L$ that preserves fine-grained facts at the leaves and coarse summaries at internal nodes. In our paper, we define $L = 4$ , including {Community, Keywords, Entity-relation Triples, Attributes}.

Entity Representation. Given a graph $\mathcal { G } = ( \mathcal { E } , \mathcal { R } )$ , we first encode each entity $e _ { i } \in \mathcal { E }$ by harvesting its contextualized embedding $\mathbf { e } _ { i } \in \mathbb { R } ^ { 3 d }$ , aggregating the frozen LLM embeddings of all triples within its one-hop neighborhood $\mathcal { N } _ { i }$ . Specifically, for each triple $( e _ { i } , r , e _ { j } ) \in \mathcal { N } _ { i } ,$ we concatenate the embeddings of the head entity $\mathbf { e } _ { i } ,$ relation $\mathbf { r } _ { i j } ,$ and tail entity $\mathbf { e } _ { j } ,$ then average across all neighboring triples:

$$
\mathbf {e} _ {i} = \frac {1}{| \mathcal {N} _ {i} |} \sum_ {\left(e _ {i}, r, e _ {j}\right) \in \mathcal {N} _ {i}} \left[ \mathbf {e} _ {i} \| \mathbf {r} _ {i j} \| \mathbf {e} _ {j} \right]. \tag {4}
$$

To this end, the entity representation could effectively preserve both local structural patterns and semantic relations, enabling downstream clustering to leverage both signals.

Cluster Initialization. Due to the huge size of real-world graph $\mathcal { G }$ , we first reduce the search space by initializing the communities by applying K-means clustering on the entity embeddings $\{ \mathbf { e } _ { i } \} _ { i = 1 } ^ { N } ,$ producing an initial partition candidates $\{ \mathcal { C } _ { 1 } ^ { ( 0 ) } , . . . , \mathcal { C } _ { k } ^ { ( 0 ) } \} ,$ , where the superscript denotes the iteration count. While this step provides a coarse grouping, it does not yet account for the interplay between structural and semantic similarity. The cluster number is limited as $\begin{array} { r } { k = \operatorname* { m i n } \Big ( \operatorname* { m a x } \big ( 2 , \lfloor \frac { | \mathcal { E } | } { \beta } \rfloor \big ) , \eta \Big ) . } \end{array}$ , where $\beta { = } 1 0$ controls the granularity that ensures minimum 10 entities per cluster, $\scriptstyle \eta = 2 0 0$ prevents excessive fragmentation. We implement this with optimized KMeans (n_init=5, random_state $\scriptstyle 1 = 4 2$ ) to ensure reproducibility.

Iterative Community Fusion via Dual-Perception Scoring. First, to refine the initial clusters, we introduce a dual-perception scoring function $\phi ( e _ { i } , \mathcal { C } _ { m } ^ { ( t ) } )$ that quantifies the affinity between a node $e _ { i }$ and a community $\mathcal { C } _ { m } ^ { ( t ) }$ at iteration t. This score combines two considerations. $( i )$ topological connectivity overlap $( \mathbb { S } _ { r } )$ that measures the Jaccard similarity between the relation incident to $e _ { i }$ and those in $\mathcal { C } _ { m } ^ { ( t ) }$ ; (ii) subgraph semantic similarity $( \mathbb { S } _ { s } )$ , which computes the cosine similarity between the entities’s embedding $\mathcal { F } _ { \Theta } ( \mathbf { T } _ { i } )$ and the community centroid $\mathbb { E } _ { \mathcal { C } _ { m } ^ { ( t ) } } [ \bar { \mathcal { F } } _ { \Theta } ( \mathbf { T } _ { j k } ) ] .$ , where $\mathcal { F } _ { \Theta }$ is a matrix for embedding transformation. C m

$$
\phi \left(e _ {i}, \mathcal {C} _ {m}\right) = \underbrace {\mathbb {S} _ {r} \left(e _ {i} , \mathcal {C} _ {m}\right)} _ {\text {r e l a t i o n a l}} \oplus \lambda \underbrace {\mathbb {S} _ {s} \left(e _ {i} , \mathcal {C} _ {m}\right)} _ {\text {s e m a n t i c}}, \tag {5}
$$

with

$$
\mathsf {S} _ {r} (e _ {i}, \mathcal {C} _ {m}) = \frac {\| \Psi (e _ {i}) \cap \Psi (\mathcal {C} _ {m}) \| _ {2}}{\| \Psi (e _ {i}) \cup \Psi (\mathcal {C} _ {m}) \| _ {2}},
$$

$$
\mathbb {S} _ {s} \left(e _ {i}, \mathcal {C} _ {m}\right) = \phi \left(\mathcal {F} _ {\Theta} \left(\mathbf {T} _ {i}\right), \sum_ {j \in \mathcal {C} _ {m}} \left(\mathcal {F} _ {\Theta} \left(\mathbf {T} _ {j}\right)\right)\right), \tag {6}
$$

where $\mathbb { S } _ { s }$ denotes the Jaccard similarity matrix computed over the multiset of incident relation types $\Psi ( \cdot )$ $\mathbb { S } _ { s } ( i , j )$ measures the overlap of relation-specific neighborhoods between nodes $i$ and $j$ .

Leveraging the dual-perception score, at each iteration $t$ , we first locate the most representative centroid entities for each community, which maximizes its dual-perception affinity score $\phi ( \bar { e } _ { i } , \mathcal { C } _ { m } )$ with respect to the entire community subgraph. We define the center nodes as: i.e., $e _ { \mathrm { c e n t e r } } ^ { * } = \arg \operatorname* { m a x } \phi ( e _ { i } , \mathcal { C } _ { m } ) ,$ , where $e _ { i } \in \mathcal { C } _ { m } , \phi \mathopen { } \mathclose \bgroup \left( e _ { i } , \mathcal { C } _ { m } \aftergroup \egroup \right)$ is the dual-perception score as aforementioned, combining both topological relation overlap $\mathbb { S } _ { \boldsymbol { r } } \bigl ( e _ { i } , \mathcal { C } _ { m } \bigr )$ and semantic similarity $\mathbb { S } _ { s } ( e _ { i } , \mathcal { C } _ { m } )$ . This selection criterion ensures that the center node not only exhibits strong structural connectivity within the community, i.e., high $\mathbb { S } _ { r }$ but also encapsulates the dominant semantic characteristics of the subgraph, i.e., high $\mathbb { S } _ { s }$ . The resulting center nodes are then employed to serve as high-quality representatives for their respective communities, facilitating efficient pair-wise community

![](images/c8eae2f3411e13ce870cfa008fbe96adda40e91583c8068c8291d90479cf4cac.jpg)  
Figure 4. The figure contrasts three query-resolution strategies for a multi-hop question. While embedding matching retrieves disjointed facts (left) and traditional agents use repetitive templates (right), our agentic decomposer (center) leverages domain schema to plan efficient sub-queries: (1) compare record label revenues, (2) locate the larger group’s headquarters, and (3) trace the explorer’s visit—achieving precise, with parallel reasoning and outperforming unstructured retrieval and template-based agents.

fusion. We then facilitate the pairwise matching between all clusters using their centroid dual-perception score. Clusters $( \mathcal { C } _ { a } ^ { ( t ) } , \mathcal { C } _ { b } ^ { ( t ) } )$ are merged if their dual-perception divergence falls below a threshold ϵ:

$$
\mathbb {E} \left[ \phi \left(e _ {i}, \mathcal {C} _ {a} ^ {(t)}\right) \right] - \mathbb {E} \left[ \phi \left(e _ {i}, \mathcal {C} _ {b} ^ {(t)}\right) \right] <   \epsilon . \tag {7}
$$

This design further shrinks the search space from node-community comparison to node-node comparison, yielding a boosted efficient community detection.

# 3.2.1 Knowledge Tree

To this end, building upon our schema-bounded extraction framework, we develop a hierarchical knowledge organization pipeline that transforms raw graphs into a structured Knowledge Tree K. First, the process begins with our novel dual-perception community detection algorithm, which computes entity-community affinity through the combined metric, blending topological connectivity overlap with semantic subgraph similarity. Second, $f _ { \mathrm { L L M } } ( { \mathcal { C } } _ { m } )$ is then applied to generate a brief name and description for the entire community based on the member names. These community names are treated as community nodes and inserted into the original graph, connecting with each member entity with the relation member_of. Third, within each detected community ${ \mathcal { C } } _ { m } ,$ we identify pivotal keywords by selecting entities maximizing the structural-semantic score arg $\mathrm { m a x } _ { e _ { i } \in \mathcal { C } _ { m } } \phi ( e _ { i } , \mathcal { C } _ { m } )$ .

The resulting hierarchy, together with the schema, collectively informs the construction of our four-layer knowledge tree $\kappa$ . The tree maximizes bottom-up semantic coherence at each level, simultaneously preserving fine-grained reasoning through granular entity-relation/entity-attribute retrieval $( \mathcal { L } _ { 1 } )$ and enhancing high-level community-based filtering $( \mathcal { L } _ { 4 } )$ . We formally define it as $\textstyle { \mathcal { K } } = \bigcup _ { \ell = 1 } ^ { 4 } L _ { \ell }$

$$
L _ {\ell} = \left\{ \begin{array}{l l} \left\{\mathcal {C} _ {m} \right\} & \ell = 4 (\text {C o m m u n i t y}) \\ \left\{\arg \max  \phi \left(v _ {i}, \mathcal {C} _ {m}\right) \right\} & \ell = 3 (\text {K e y w o r d s}) \\ \{(h, r, t) \mid h, t \in \mathcal {E}, r \in \mathcal {R} \} & \ell = 2 (\text {E n t i t y - R e l a t i o n T r i p l e s}) \\ \left\{\left(e, \text {h a s} _ {-} \text {a t t r}, \left\{e _ {\text {a t t r}} ^ {\text {t y p e}}: e _ {\text {a t t r}} ^ {\text {v a l u e}} \right\}\right) \right\} & \ell = 1 (\text {A t t r i b u t e s}) \end{array} \right. \tag {8}
$$

# 3.3 Agentic Retriever

Schema-enhanced Query Decomposer. The complexity of multi-hop queries in large-scale knowledge graphs necessitates an intelligent decomposition mechanism that respects both the explicit schema constraints and implicit semantic relationships. Our schema-guided decomposition approach provides several key advantages over traditional methods. First, by leveraging the graph schema $\bar { \cal S } = \bar { \bf \Phi } ( S _ { e } , S _ { r } , S _ { \mathrm { a t t r } } ) .$ , where $ { \boldsymbol { S } } _ { e }$ denotes entity types, $S _ { r }$ represents relation types, and $\boldsymbol { S } _ { a t t r }$ contains attribute definitions, we ensure that each generated atomic sub-query strictly adheres to valid patterns in the knowledge graph. This schema-awareness prevents the generation of ill-formed queries that would either fail to return results or retrieve irrelevant information. For instance, when processing a query like "Which pharmaceutical companies manufacture diabetes drugs?", the schema guarantees that the "manufacture" relation only connects companies to drugs, not to other entity types. Second, the schema serves as a semantic framework that maintains coherence throughout the decomposition process. Consider the query "Where did Turing Award winners study?" Our method automatically maps "Turing Award winner" to the appropriate entity type $ { \boldsymbol { S } } _ { e }$ : Person with the specific award attribute, while correctly interpreting "study" as an $S _ { r }$ : educated_at. This semantic precision prevents the common problem of interpretation drift that often occurs in naive decomposition approaches. Therefore, the final $\mathcal { Q } = f _ { \mathrm { L L M } } ( q , S ) \overset { - } { = } \{ q _ { 1 } , q _ { 2 } \ldots q _ { i } \} ,$ , where $i$ is a pre-defined maximum number for total atomic sub-queries and each $q _ { i }$ explicitly targets either: (i) node-level retrieval (e, has_attr, a), (ii) triple-level matching $\hat { ( } h , r , t )$ , or $( i i i )$ community-level verification ${ \mathcal { C } } _ { m } ,$ as determined by schema elements $s _ { e } , s _ { r } ,$ and $\boldsymbol { S } _ { a t t r }$ .

Iterative Reasoning and Reflection. Since reasoning and reflection are two core cognitive capabilities for the agent, following the standard agent framework of perception-reasoning-action cycles, we formalize our agent as a tuple $\mathcal { A } \overset { \cdot } { = } \langle \mathcal { S } , \mathcal { H } , f _ { \mathrm { L L M } } \rangle .$ , where $\mathcal { H }$ denotes the agent’s historical memory containing both reasoning steps and the retrieval results, and the functions $f _ { \mathrm { L L M } }$ is employed to implement both key operations.

$$
\mathcal {A} ^ {(t)} = \underbrace {f _ {\text {L L M}} \left(q ^ {t} , \mathcal {H} ^ {(t - 1)}\right)} _ {\text {R e a s o n i n g R e f l e c t i o n}}, \tag {9}
$$

This process addresses the compositional generalization challenge in complex QA by (i) maintaining explicit symbolic grounding through $s$ during reasoning steps, and (ii) performing continuous self-monitoring via reflection to detect and correct reasoning paths. The agent’s operational flow alternates between forward reasoning with schema-guided query decomposition and retrieval and backward reflection for complex scenarios, creating a closed-loop framework that progressively converges to optimal solutions.

Multi-Route Retrieval. To handle diverse sub-query types, we implement four parallel retrieval strategies with distinct optimization objectives:

$$
\text {E n t i t y M a t c h i n g}: \quad \underset {e \in \mathcal {E}} {\arg \max } \cos (e, q _ {i})
$$

$$
\text {T r i p l e} \quad \text {M a t c h i n g}: \quad \arg \max  _ {(h, r, t) \in \mathcal {G}} \cos \left(\left(\mathbf {e} _ {h}, \mathbf {r}, \mathbf {e} _ {t}\right), \mathbf {q} _ {\mathbf {i}}\right)) \tag {10}
$$

$$
\text {C o m m u n t y F i l t e r i n g}: \quad \underset {\mathcal {C} _ {m} \in \mathcal {K}} {\arg \max } \cos \left(\mathrm {e} _ {\mathcal {C} _ {m}}, \mathrm {q} _ {\mathrm {i}}\right)
$$

$$
\text {D F S P a t h T r a v e r s a l}: \quad \mathcal {P} \left(q _ {i}\right) = e _ {0} \xrightarrow {r _ {1}} e _ {1} \xrightarrow {r _ {2}} \dots \xrightarrow {r _ {n}} e _ {n} \quad \text {s . t .} \quad \forall r _ {i} \in \mathcal {R}, n \leq d
$$

In general, the four retrieval paths exhibit distinct specialization patterns: (i) Entity Matching optimally handles single-hop simple queries requiring precise node identification, e.g., atomic fact check problem;

(ii) Triple Matching dominates few-hop reasoning tasks by modeling $( h , r , t )$ compositional semantics, particularly effective for relationship inference; (iii) Community Filtering aims to address global queries, e.g., summarization and cross-domain problems through top-down filtering in the cluster; (4) DFS Path Traversal scales to complex multi-constraint problems, we define the maximum depth $d = 5$ . This specialization aligns with the cognitive spectrum from atomic facts to complex reasoning scenarios.

# 4 Experiments

# 4.1 Evaluation Metrics

Following the workflow of RAG, the evaluation is typically divided into two stages: (i) assess the accuracy of retrieved evidence and (ii) examine the end-to-end performance by evaluating the quality of LLMs responses generated from the retrieved evidence. In practical deployment scenarios, where multiple valid retrieval references may exist for identical answers, the latter evaluation paradigm has emerged as the prevailing standard in practical applications.

Regarding the assessment of LLMs responses, several character-based matching protocols, e.g., recall, EM and F1 score were established. To account for semantic deviations caused by minor character variations, where slight textual differences may lead to substantially divergent meanings, we employ DeepSeek-V3-0324 to assess response similarity against ground truth references.

During the reproduction of various GraphRAG frameworks, we observed experimental results exhibit significant variations depending on the prompts in the LLMs generation stage. Specifically, some frameworks(Zhao et al. [2025]) instruct to explicitly reject to answer when retrieved evidence is insufficient, while others(Xiao et al. [2025], Sarthi et al. [2024]) allow LLMs to leverage its parametric knowledge or ambiguates the instruction in such cases. Given that most LLMs have been exposed to extensive corpora during pretraining, we identify answering questions based on LLMs’ knowledge rather than retrieval mechanism as a critical factor for fairly evaluation - we term knowledge leaking.

To separately assess two critical capabilities: (1) recognizing knowledge limitations, and (2) leveraging LLMs’ parametric knowledge, we therefore implement a dual-mode evaluation on three widely-used datasets:

• Reject mode. Under this mode, LLMs must reject to answer the question when retrieval fails to provide sufficient evidence. This strictly evaluates the retrieval effectiveness and prevent hallucination.   
• Open mode. LLMs are allowed to answer using either retrieved content or its inherently parametric knowledge. This maximally measures the overall capability in real-world practical deployment.

We have reproduced representative baselines and conducted comprehensive evaluations based on the metrics in this work. The corresponding prompts are provided in Appendix A. Moreover, the observations further underscore the importance of our proposed AnonyRAG dataset to ensure fair and comprehensive assessment of GraphRAG methods.

# 4.2 Datasets

We firstly evaluate Youtu-GraphRAG in dual-mode on three widely used multi-hop QA datasets: HotpotQA (Yang et al. [2018]), MuSiQue (Trivedi et al. [2022]) and 2WikiMultiHopQA (abbreviated as 2Wiki Ho et al. [2020]), following the setting in (Jimenez Gutierrez et al. [2024], Gutiérrez et al. [2025]) for fair comparison.

To evaluate the framework’s performance across diverse domains, we also employ GraphRAG-Bench(Xiao et al. [2025]), shorted as G-Bench, a benchmark dataset constructed from textbook corpora. Additionally, to

prevent knowledge leaking, we propose two novel bilingual anonymous datasets, i.e., AnonyRAG-CHS and AnonyRAG-ENG and propose a challenging ‘Anonymous Reversion’ task.

We anonymize specific entity types (e.g., people, locations) in the dataset to break the model’s memory shortcuts and prevent it from relying on pretrained knowledge rather than retrieved evidence. Moreover, we preserve semantic coherence through entity linking, enabling LLMs to maintain discourse comprehension despite anonymized mentions. The construction details of the dataset are documented in Appendix B.

# 4.3 Baselines

We include three pipelines of research as baselines. (i) Naive RAG, as the standard RAG approach that retrieves top- $k$ document chunks using vector similarity search without any explicit knowledge structuring; (ii) Pure GraphRAG, which builds flat knowledge graphs for retrieval but lacks hierarchical organization, focusing primarily on relational reasoning through graph traversal algorithms, including GraphRAG (Edge et al. [2024]), LightRAG (Guo et al. [2024]), G-Retriever (He et al. [2024]) and HippoRAG 1&2 (Jimenez Gutierrez et al. [2024], Gutiérrez et al. [2025]); (iii) Tree-based GraphRAG, represents hierarchical methods that employ recursive clustering and summarization to construct multi-level knowledge trees including RAPTOR (Sarthi et al. [2024]) and $\mathrm { E } ^ { \tilde { 2 } }$ GraphRAG (Zhao et al. [2025]).

To ensure a fair performance comparison, we reproduce all the baselines and Youtu-GraphRAG with the same setting and evaluate with consistent metrics. In terms of base models, we maintain DeepSeek-V3-0324 and Qwen3-32B as the base LLMs and a lightweight embedding model all-MiniLM-L6-v2.

# 4.4 Overall Evaluation

# 4.4.1 Comparison of Time and Token Consumption

For baselines involving graph construction and community detection stages, this section compares their token and time consumption. Unless otherwise specified, all LLM APIs invoked here are based on the DeepSeek-V3-0324 and deployed on identical hardware. All procedures are executed using 32-thread concurrent inference to ensure both the efficiency of graph construction and the fairness of comparisons.

Figure 5a presents the time and token consumption during the graph construction stage for Youtu-GraphRAG and five baselines. Our method consistently achieves the lowest token consumption across all six datasets and maintains relatively efficient time performance on five of the datasets. In the community detection stage, as shown in Figure 5b, Youtu-GraphRAG achieves the lowest token consumption compared with the other three baselines, consuming no more than 10,000 tokens on any dataset. Meanwhile, our method also demonstrates consistently efficient time performance across all datasets.

![](images/dbd12380a915c8fe0887f49f665dbe292dae44678b6dea49edaa45cca16414cf.jpg)  
(a) Consumption comparison of graph construction

![](images/9d2608dcb2a5122cb97896cbd5e1d0e680f54da20d2df35926f0f273446933c0.jpg)  
(b) Consumption comparison of community detection

Table 1. Overall performance comparisons over benchmark datasets in terms of top-20 Accuracy.   

<table><tr><td rowspan="2">Method</td><td colspan="2">HotpotQA</td><td colspan="2">2Wiki</td><td colspan="2">MuSiQue</td><td>G-Bench</td><td>Annoy-CHS</td><td>Annoy-ENG</td></tr><tr><td>Open</td><td>Reject</td><td>Open</td><td>Reject</td><td>Open</td><td>Reject</td><td>Open</td><td>Open</td><td>Open</td></tr><tr><td colspan="10">Deepseek-V3-0324</td></tr><tr><td>Zero-shot LLM</td><td>53.70</td><td>-</td><td>41.6</td><td>-</td><td>25.7</td><td>-</td><td>70.92</td><td>9.62</td><td>8.18</td></tr><tr><td>Naive RAG</td><td>79.90</td><td>72.40</td><td>70.3</td><td>38.9</td><td>47.49</td><td>30.63</td><td>71.81</td><td>12.5</td><td>43.02</td></tr><tr><td>E2GraphRAG</td><td>68.70</td><td>48.80</td><td>43.20</td><td>20.00</td><td>28.36</td><td>8.01</td><td>68.66</td><td>16.01</td><td>35.97</td></tr><tr><td>RAPTOR</td><td>80.90</td><td>73.60</td><td>70.10</td><td>38.40</td><td>48.50</td><td>31.10</td><td>73.08</td><td>12.08</td><td>40.2</td></tr><tr><td>LightRAG</td><td>71.90</td><td>56.00</td><td>58.00</td><td>29.20</td><td>38.98</td><td>24.57</td><td>70.83</td><td>9.16</td><td>22.14</td></tr><tr><td>GraphRAG</td><td>56.10</td><td>26.40</td><td>41.80</td><td>10.00</td><td>32.20</td><td>16.50</td><td>75.54</td><td>21.66</td><td>38.85</td></tr><tr><td>G-Retriever</td><td>49.00</td><td>6.70</td><td>35.80</td><td>5.00</td><td>23.50</td><td>1.70</td><td>70.63</td><td>4.07</td><td>5.08</td></tr><tr><td>HippoRAG</td><td>81.70</td><td>73.10</td><td>77.90</td><td>64.00</td><td>48.30</td><td>36.20</td><td>72.89</td><td>36.77</td><td>40.68</td></tr><tr><td>HippoRAG-IRCOT</td><td>81.00</td><td>74.60</td><td>78.40</td><td>66.00</td><td>46.70</td><td>35.50</td><td>73.38</td><td>36.05</td><td>42.17</td></tr><tr><td>HippoRAG2</td><td>81.80</td><td>74.90</td><td>77.30</td><td>48.30</td><td>50.80</td><td>37.80</td><td>79.37</td><td>12.92</td><td>43.16</td></tr><tr><td>Ours w/o Agent</td><td>83.70</td><td>75.30</td><td>72.80</td><td>57.80</td><td>51.40</td><td>40.00</td><td>81.53</td><td>37.06</td><td>40.05</td></tr><tr><td>Youtu-GraphRAG</td><td>86.50</td><td>81.20</td><td>85.50</td><td>77.60</td><td>53.60</td><td>47.50</td><td>86.54</td><td>42.88</td><td>43.26</td></tr><tr><td colspan="10">Qwen3-32B</td></tr><tr><td>Zero-shot LLM</td><td>36.40</td><td>-</td><td>33.30</td><td>-</td><td>13.40</td><td>-</td><td>70.04</td><td>5.11</td><td>6.49</td></tr><tr><td>Naive RAG</td><td>75.00</td><td>69.00</td><td>58.50</td><td>39.60</td><td>40.64</td><td>33.03</td><td>72.69</td><td>7.56</td><td>26.84</td></tr><tr><td>RAPTOR</td><td>79.20</td><td>72.90</td><td>61.20</td><td>40.10</td><td>38.99</td><td>32.86</td><td>72.20</td><td>13.37</td><td>22.14</td></tr><tr><td>HippoRAG</td><td>77.00</td><td>71.80</td><td>72.80</td><td>62.50</td><td>40.60</td><td>32.10</td><td>75.64</td><td>8.58</td><td>32.30</td></tr><tr><td>HippoRAG-IRCOT</td><td>80.30</td><td>76.60</td><td>74.80</td><td>65.40</td><td>44.70</td><td>37.40</td><td>77.11</td><td>9.16</td><td>33.15</td></tr><tr><td>HippoRAG2</td><td>81.80</td><td>71.30</td><td>65.20</td><td>39.90</td><td>51.40</td><td>37.70</td><td>80.35</td><td>12.65</td><td>38.36</td></tr><tr><td>Ours w/o Agent</td><td>83.80</td><td>73.90</td><td>74.90</td><td>55.30</td><td>52.90</td><td>40.10</td><td>80.74</td><td>34.88</td><td>35.13</td></tr><tr><td>Youtu-GraphRAG</td><td>85.90</td><td>78.60</td><td>85.70</td><td>74.20</td><td>54.60</td><td>45.30</td><td>84.48</td><td>39.24</td><td>40.05</td></tr></table>

# 4.4.2 Main Performance Comparison

In Table 1, we report the top-20 accuracy across six challenging benchmarks under both open and reject modes, based on two strong LLM backbones, i.e., DeepSeek-V3-0324 and Qwen3-32b. Across virtually all datasets and settings, Youtu-GraphRAG attains the highest performance, reflecting its ability to combine precise retrieval with robust reasoning. Besides, we also include an variant with no agent for iterative reasoning and reflection as a lightweight version, i.e., Ours w/o Agent, fulfilling real-world applications requiring real-time interactive feedback.

The distinction between the two evaluation modes provides complementary perspectives on system capability. Open mode unlocks the full reasoning potential of the LLM to synthesize an answer regardless of retrieval gaps. This mirrors high-coverage real-world deployments where maximizing end-task accuracy outweighs caution. Youtu-GraphRAG consistently outperforms existing baselines, achieving improvements from 2 to 8 points over the strongest competitor across datasets. When augmented with our agent framework, Youtu-GraphRAG further pushes the performance frontier, reaching top-20 accuracies of $8 6 . 5 \%$ , $8 5 . 5 \%$ , and $5 3 . 6 \%$ on HotpotQA, 2Wiki, and MuSiQue respectively under Deepseek-V3-0324, and $8 5 . 9 \%$ , $8 5 . 7 \% ,$ and $5 4 . 6 \%$ under Qwen3-32B, demonstrating a clear advantage in multi-hop reasoning and cross-document synthesis. Reject mode, by contrast, imposes a stringent criterion if the retrieved context is insufficient, the model must abstain. Youtu-GraphRAG attains $8 1 . 2 \%$ , $7 7 . 6 \% ,$ , and $4 7 . 5 \%$ on HotpotQA, 2Wiki, and MuSiQue, outperforming the strongest baseline by 7–14 points. Across all datasets, our method achieves consistently higher top-20 accuracy, confirming its ability to synergize graph-based retrieval with agent-driven reasoning for both high-coverage and high-precision scenarios. We value this metric since it directly probes retrieval quality, as speculative answers are penalized and the acceptance rate becomes a direct function of retrieval completeness and precision. Our superiority on two anonymous datasets also validates the generalizability of Youtu-GraphRAG beyond standard benchmarks. Specifically, under the open mode, it achieves $4 2 . 8 8 \%$

and $4 3 . 2 6 \%$ top-20 accuracy on Annoy-CHS and Annoy-ENG, respectively, surpassing all baselines by a clear margin. These results also reflect our robust reasoning and retrieval integration across diverse languages and domains, demonstrating that our approach could be easily transferred to previously unseen data distributions while maintaining high accuracy.

A key objective of Youtu-GraphRAG is to jointly optimize performance and efficiency by unifying graph construction and retrieval. Figure 6 illustrates the trade-off between token consumption during the construction and overall QA performance across six benchmarks. Our approach consistently achieves optimal performance with the least token consumption, effectively shifting the Pareto frontier compared to all baselines.

![](images/45201518340eda0e27fc0b0f50037f96f8ce74c2e2fc8ee6c78c46c3f6049969.jpg)  
Figure 6. Youtu-GraphRAG effectively moves the Pareto frontier with lower token costs and higher performance.

tains the best performance on all benchmarks while consuming up to an order of magnitude fewer tokens during graph construction. This demonstrates that careful integration of structured schema alignment, hierarchical knowledge tree, and adaptive retrieval can fundamentally improve the cost-effectiveness of GraphRAG systems in real-world applications.

While existing GraphRAG methods face a dilemma to balance the token consumption during construction and the accuracy for final generation, Youtu-GraphRAG leverages a vertically unified novel framework, i.e., schema-guided extraction, duallyperceived community detection and the schemaenhanced agentic retrieval to build concise yet semantically rich graphs and allow the agent to maximize reasoning effectiveness. As a result, our method effectively moves the Pareto frontier and at-

Table 2. Overall performance comparisons over benchmark datasets based on DeepSeek in terms of top-10 Accuracy.   

<table><tr><td rowspan="2">Method</td><td colspan="2">HotpotQA</td><td colspan="2">2Wiki</td><td colspan="2">MuSiQue</td><td>G-Bench</td><td>Annoy-CHS</td><td>Annoy-ENG</td></tr><tr><td>Open</td><td>Reject</td><td>Open</td><td>Reject</td><td>Open</td><td>Reject</td><td>Open</td><td>Open</td><td>Open</td></tr><tr><td>Naive RAG</td><td>79.40</td><td>68.00</td><td>67.60</td><td>33.70</td><td>45.58</td><td>26.73</td><td>71.22</td><td>12.08</td><td>38.93</td></tr><tr><td>RAPTOR</td><td>78.20</td><td>67.10</td><td>67.40</td><td>36.40</td><td>45.88</td><td>30.03</td><td>72.79</td><td>11.77</td><td>33.99</td></tr><tr><td>G-R retriever</td><td>49.90</td><td>5.90</td><td>38.00</td><td>3.80</td><td>23.50</td><td>1.70</td><td>70.24</td><td>5.38</td><td>5.50</td></tr><tr><td>LightRAG</td><td>71.98</td><td>58.10</td><td>65.70</td><td>38.10</td><td>39.40</td><td>22.90</td><td>69.74</td><td>8.58</td><td>18.90</td></tr><tr><td>GraphRAG</td><td>54.30</td><td>23.70</td><td>40.00</td><td>9.80</td><td>30.20</td><td>16.00</td><td>61.39</td><td>21.37</td><td>38.36</td></tr><tr><td>HippoRAG</td><td>78.20</td><td>69.40</td><td>77.10</td><td>61.10</td><td>45.20</td><td>30.90</td><td>70.14</td><td>34.01</td><td>40.12</td></tr><tr><td>HippoRAG-IRCOT</td><td>78.10</td><td>70.20</td><td>77.70</td><td>60.70</td><td>44.40</td><td>31.60</td><td>72.89</td><td>36.19</td><td>41.42</td></tr><tr><td>HippoRAG2</td><td>79.40</td><td>70.40</td><td>74.60</td><td>45.80</td><td>49.10</td><td>34.00</td><td>77.21</td><td>13.52</td><td>37.24</td></tr><tr><td>Ours w/o Agent</td><td>80.50</td><td>72.10</td><td>72.10</td><td>54.40</td><td>49.80</td><td>38.30</td><td>80.55</td><td>35.17</td><td>40.54</td></tr><tr><td>Youtu-GraphRAG</td><td>83.40</td><td>78.90</td><td>82.30</td><td>72.60</td><td>52.10</td><td>46.90</td><td>83.50</td><td>38.08</td><td>42.57</td></tr></table>

# 4.5 Analysis of Generalizability

To examine the domain-transfer capability of Youtu-GraphRAG, we evaluate it across six heterogeneous benchmarks without any task-specific fine-tuning. As shown in Figure 7, Youtu-GraphRAG achieves the best performance in both Open Accuracy and Reject Accuracy on all datasets, surpassing state-of-the-art GraphRAG baselines by a clear margin.

We attribute this strong generalizability to the intrinsic integration of graph construction and retrieval within our framework. (i) The schema-guided extraction agent produces consistent, domain-adaptive graphs; the dually-perceived community detection yields hierarchical knowledge structures that remain robust across domains; (ii) The agentic query decomposer dynamically adapts retrieval strategies to different question types without manual tuning. Notably, our model demonstrates particularly large gains on multi-

Table 3. Ablation studies of our method over six datasets. We evaluate three variants: without Community detection (w/o Comm.), without Agent coordination (w/o Agent), and without Schema guidance (w/o Schema).   

<table><tr><td>Variants</td><td>HotptQA</td><td>2Wiki</td><td>MuSiQue</td><td>G-Bench</td><td>AnonyRAG-CHS</td><td>AnonyRAG-ENG</td></tr><tr><td>w/o Comm.</td><td>79.50</td><td>75.10</td><td>44.00</td><td>85.02</td><td>39.97</td><td>39.92</td></tr><tr><td>w/o Agent</td><td>75.30</td><td>57.80</td><td>40.00</td><td>81.53</td><td>37.60</td><td>40.05</td></tr><tr><td>w/o Schema</td><td>77.10</td><td>73.40</td><td>45.60</td><td>83.50</td><td>35.61</td><td>40.32</td></tr><tr><td>Youtu-GraphRAG</td><td>81.20</td><td>77.60</td><td>47.50</td><td>86.54</td><td>42.88</td><td>43.26</td></tr></table>

hop reasoning datasets such as HotpotQA and 2Wiki in open settings, and shows superior abstention capability on MuSiQue and 2Wiki in reject settings, indicating robustness in both complex reasoning and uncertainty calibration. These results confirm that Youtu-GraphRAG can seamlessly transfer to unseen domains while preserving structural fidelity and reasoning depth, fulfilling the vision of a foundational GraphRAG paradigm.

![](images/19cdb4ec83f0e9fca52db8ecd814744a324ab84d6cce4718f539942d82e40088.jpg)  
Figure 7. We showcase the generalizability over six benchmark datasets in terms of both open and reject accuracy. attains $3 8 . 0 8 \%$ and $4 2 . 5 7 \%$ , reinforcing its consistent s confirm that our approach not only excels in high-co selection under stricter evaluation criteria, further val retrieval with agent-guided reasoning.

Furthermore, we summarize the top-10 results in Table 2, which provides a more stringent evaluation of retrieval. Our methods consistently outperform all baselines across both open and reject modes. In the open mode, Youtu-GraphRAG achieves top-10 accuracies of $8 3 . 4 \%$ , $8 2 . 3 \%$ , and $5 2 . 1 \%$ on HotpotQA, 2Wiki, and MuSiQue respectively, surpassing the strongest competitor by 48 points. Under the reject ˜ mode, the gains are even more pronounced, with improvements of 812 points, indicating robust re- ˜ trieval fidelity and reduced speculative answering. Notably, on the two anonymous datasets, Annoy-CHS and Annoy-ENG, our agent-enhanced model uperiority in diverse scenarios. These top- $k$ results verage settings but also maintains precise answer idating the effectiveness of integrating graph-based

# 4.6 Ablation Studies

To quantify the contribution of each component, we perform ablations by removing community detection (w/o Comm.), agent reasoning and reflection (w/o Agent), and schema guidance (w/o Schema). Results on six benchmarks are summarized in Table 3.

Specifically, removing community detection leads to a consistent drop across all datasets, particularly on multi-hop QA tasks such as HotpotQA and 2Wiki around $1 . 7 \%$ and $2 . 5 \% ,$ indicating that structuring knowledge into coherent communities facilitates more accurate retrieval and reasoning for global questions. The absence of agent reasoning and reflection causes the most severe degradation on complex reasoning datasets, especially on 2Wiki and MuSiQue with remarkable $1 9 . 8 \%$ and- $7 . 5 \%$ differences, supporting our motivation that the iterative reasoning-feedback loop plays an essential role for resolving ambiguous intermediate steps. Eliminating schema guidance results in noticeable performance drops on knowledgeintensive settings, especially on AnonyRAG-CHS with $7 . 2 7 \%$ decreases, highlighting the importance of a high-quality initialization of seed schema for new domains. This further demonstrates our advantage since Youtu-GraphRAG only requires minimum manual intervention to handle with domain shifts. In conclusion, our model consistently outperforms all ablated variants, demonstrating that the three components are complementary: community detection improves retrieval quality, agent reasoning enhances multi-step

inference, and schema guidance enforces structural fidelity. These findings suggest that removing any single component disrupts the synergy between retrieval and reasoning, with agent reasoning being most critical for multi-hop inference, while schema plays a vito role in low-resource and domain-specific scenarios.

# 5 Related Work

While large language models (LLMs) demonstrate remarkable capabilities in language understanding and reasoning, they are known to be prone to hallucinations—generating confident yet factually incorrect outputs—especially when reasoning over complex or multi-hop queries [Zhang et al., 2025, Qin et al., 2024, Kuang et al., 2025, Dong et al., 2024, Qin et al., 2024]. Integrating LLMs with graph-structured knowledge, therefore, combines the generative flexibility of LLMs with the factual rigor of structured data, enabling more accurate and trustworthy reasoning over complex domains [Luo et al., 2023, Dong et al., 2023, Bei et al., 2025, Yasunaga et al., 2021, Luo et al., 2024]. Evolving development of GraphRAG has progressed along two complementary research trajectories since the seminal work of [Edge et al., 2024]. The first following approaches have evolved from LightRAG’s [Guo et al., 2024] vector sparsification techniques to more sophisticated graph-aware methods. Subsequent innovations include GNN-RAG and GFM-RAG [Mavromatis and Karypis, 2024, Luo et al., 2025], which employ graph neural networks for enhanced node matching, and HippoRAG 1&2 [Jimenez Gutierrez et al., 2024, Gutiérrez et al., 2025] that introduced memory mechanisms and personalized PageRank algorithms for context-aware retrieval. While another Group of methods have focused on improving the quality of knowledge organization, hierarchical approaches like RAPTOR [Sarthi et al., 2024] and $\mathrm { \check { E } } ^ { 2 }$ GraphRAG [Zhao et al., 2025] employ tree-like clustering and recursive summarization to enhance semantic organization. However, current research remain constrained by their specialized optimizations, either focusing on retrieval or construction in isolation, and lack a unified design. This fragmentation limits their performance on complex reasoning tasks requiring tight integration of knowledge organization and retrieval capabilities, which makes it even harder to adjust the entire framework for generalizability especially when domain shifts occur. Our work bridges this gap by developing a holistic framework that jointly optimizes both aspects while maintaining graph foundation model properties.

# 6 Conclusions

In this paper, we propose Youtu-GraphRAG, a vertically unified agentic paradigm that jointly optimizes both aspects through a graph schema. Our framework introduces (i) a schema-guided agent for continuous knowledge extraction with predefined entity types, relations, and attributes; (ii) dually-perceived community and keyword detection, fusing structural topology with subgraph semantics to construct a hierarchical knowledge tree that supports top-down filtering and bottom-up reasoning; (iii) an agentic retriever interprets the schema to break complex queries into tractable sub-queries, paired with an iterative reasoning and reflection; and (iv) Anonymity Reversion, a novel task to mitigate knowledge leakage in LLMs, deeply measuring the real performance of GraphRAG frameworks supported by a carefully curated anonymous dataset. Extensive experiments across six challenging benchmarks demonstrate Youtu-GraphRAG’s robustness, advancing the Pareto frontier with up to $9 0 . 7 1 \%$ reduction in token costs and $1 6 . 6 2 \%$ higher accuracy than state-of-the-art baselines. Notably, our framework exhibits strong adaptability, enabling seamless domain transfer with minimal schema adjustments. These results underscore the importance of unified graph construction and retrieval, paving the way for more efficient and generalizable GraphRAG.

# References

[1] Yilin Xiao, Junnan Dong, Chuang Zhou, Su Dong, Qianwen Zhang, Di Yin, Xing Sun, and Xiao Huang. Graphrag-bench: Challenging domain-specific reasoning for evaluating graph retrieval-augmented generation. arXiv preprint arXiv:2506.02404, 2025.   
[2] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data Engineering, 36(7): 3580–3599, 2024.   
[3] Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr. Knowledge graph prompting for multi-document question answering. In AAAI, volume 38, pages 19206–19214, 2024.   
[4] Qinggang Zhang, Junnan Dong, Hao Chen, Daochen Zha, Zailiang Yu, and Xiao Huang. Knowgpt: Knowledge graph based prompting for large language models. NeurIPS, 37:6052–6080, 2024.   
[5] Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. NeurIPS, 37:132876–132907, 2024.   
[6] Junnan Dong, Qinggang Zhang, Xiao Huang, Keyu Duan, Qiaoyu Tan, and Zhimeng Jiang. Hierarchyaware multi-hop question answering over knowledge graphs. In The Web Conf, 2023.   
[7] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. Graph retrieval-augmented generation: A survey. arXiv preprint arXiv:2408.08921, 2024.   
[8] Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A Rossi, Subhabrata Mukherjee, Xianfeng Tang, et al. Retrieval-augmented generation with graphs (graphrag). arXiv preprint arXiv:2501.00309, 2024.   
[9] Junnan Dong, Qinggang Zhang, Huachi Zhou, Daochen Zha, Pai Zheng, and Xiao Huang. Modalityaware integration with large language models for knowledge-based visual question answering. In ACL, pages 2417–2429. ACL, 2024.   
[10] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.   
[11] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrievalaugmented generation. arXiv preprint arXiv:2410.05779, 2024.   
[12] Costas Mavromatis and George Karypis. Gnn-rag: Graph neural retrieval for large language model reasoning. arXiv preprint arXiv:2405.20139, 2024.   
[13] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, and Shirui Pan. Gfm-rag: graph foundation model for retrieval augmented generation. arXiv preprint arXiv:2502.01113, 2025.   
[14] Bernal Jimenez Gutierrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically inspired long-term memory for large language models. NeurIPS, 37:59532–59569, 2024.   
[15] Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory: Non-parametric continual learning for large language models. ICML, 2025.   
[16] Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. In The Twelfth International Conference on Learning Representations, 2024.   
[17] Yibo Zhao, Jiapeng Zhu, Ye Guo, Kangkang He, and Xiang Li. Eˆ 2graphrag: Streamlining graph-based rag for high efficiency and effectiveness. arXiv preprint arXiv:2505.24226, 2025.

[18] Vincent A Traag, Ludo Waltman, and Nees Jan Van Eck. From louvain to leiden: guaranteeing wellconnected communities. Scientific reports, 9(1):1–12, 2019.   
[19] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.   
[20] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.   
[21] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060, 2020.   
[22] Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan Dong, Hao Chen, Yi Chang, and Xiao Huang. A survey of graph retrieval-augmented generation for customized large language models. arXiv preprint arXiv:2501.13958, 2025.   
[23] Libo Qin, Qiguang Chen, Yuhang Zhou, Zhi Chen, Yinghui Li, Lizi Liao, Min Li, Wanxiang Che, and Philip S. Yu. Multilingual large language model: A survey of resources, taxonomy and frontiers. CoRR, abs/2404.04925, 2024. doi: 10.48550/ARXIV.2404.04925. URL https://doi.org/10.48550/arXiv. 2404.04925.   
[24] Jiayi Kuang, Ying Shen, Jingyou Xie, Haohao Luo, Zhe Xu, Ronghao Li, Yinghui Li, Xianfeng Cheng, Xika Lin, and Yu Han. Natural language understanding and inference with MLLM in visual question answering: A survey. ACM Comput. Surv., 57(8):190:1–190:36, 2025. doi: 10.1145/3711680. URL https://doi.org/10.1145/3711680.   
[25] Junnan Dong, Zijin Hong, Yuanchen Bei, Feiran Huang, Xinrun Wang, and Xiao Huang. Clr-bench: Evaluating large language models in college-level reasoning. arXiv preprint arXiv:2410.17558, 2024.   
[26] Libo Qin, Qiguang Chen, Xiachong Feng, Yang Wu, Yongheng Zhang, Yinghui Li, Min Li, Wanxiang Che, and Philip S. Yu. Large language models meet NLP: A survey. CoRR, abs/2405.12819, 2024. doi: 10.48550/ARXIV.2405.12819. URL https://doi.org/10.48550/arXiv.2405.12819.   
[27] Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. Reasoning on graphs: Faithful and interpretable large language model reasoning. arXiv preprint arXiv:2310.01061, 2023.   
[28] Yuanchen Bei, Weizhi Zhang, Siwen Wang, Weizhi Chen, Sheng Zhou, Hao Chen, Yong Li, Jiajun Bu, Shirui Pan, Yizhou Yu, et al. Graphs meet ai agents: Taxonomy, progress, and future opportunities. arXiv preprint arXiv:2506.18019, 2025.   
[29] Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec. Qa-gnn: Reasoning with language models and knowledge graphs for question answering. arXiv preprint arXiv:2104.06378, 2021.   
[30] Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Yuan-Fang Li, Chen Gong, and Shirui Pan. Graphconstrained reasoning: Faithful reasoning on knowledge graphs with large language models. arXiv preprint arXiv:2410.13080, 2024.

# A Prompt templates in LLMs generation

We present the prompt templates in A.1 and A.1, which designed to evaluate whether permitting LLMs to utilize its parametric knowledge within the RAG system affects performance. To minimize confounding factors, we employed minimalistic prompts that solely differentiate between the two modes.

# A.1 Reject mode

Given the question and the extracted knowledge from different retrieval paths, please answer the question below. If the extracted knowledge is not enough to answer, please reject to answer.

Question: {query}

Extracted Knowledge: {context}

Answer:

# A.2 Open mode

Given the question and the extracted knowledge from different retrieval paths, please answer the question below. If the extracted knowledge is not enough to answer, please answer it based on your own knowledge.

Question: {query}

Extracted Knowledge: {context}

Answer:

# B Data Collection and Processing

All raw data in this study are sourced from the original texts of four classic novels: Water Margin, Dream of the Red Chamber, Moby-Dick, and Middlemarch. The copyrights of all these works have entered the public domain, thus presenting no copyright issues. In selecting data sources, we pursued two key objectives: (1) Ensuring comprehensive multilingual evaluation coverage, while (2) Maintaining sufficient complexity in entity representations (e.g., persons, locations) to rigorously assess model capabilities. The basic statistical information of the dataset is in Table 4.

In our data processing methodology, we employed DeepSeek for entity extraction from the corpus, then the data chunks are anonymized with the extracted entities. Query-answer pairs were constructed by DeepSeek using queries from 2Wiki and MuSiQue as seed templates. Upon acquiring the question-answer pairs, we performed entity anonymization using the same anonymization dictionary as applied to the corpus. This procedure ensures that LLMs cannot effectively leverage parametric memorized patterns from

questions. A representative example of anonymized question-answer pairs is presented in Table 5. As clearly demonstrated, while LLMs could handle questions according to common sense knowledge, their performance significantly degrades when confronted with anonymized versions of these questions. This phenomenon forces LLMs to rely on retrieved contextual information rather than depending solely on their parametric knowledge.

To avoid the variance in evaluating subjective questions, we finally converted the questions into two formats:

Anonymity Reversion. We provide LLMs with anonymized question-answer pairs as context, requiring to infer and reconstruct the original entities that were anonymized. This task specifically assesses the model’s ability to leverage contextual clues for entity recovery.

Multiple Choice. To diversify question types and ensure objective evaluation, a subset of questions was converted into multiple-choice format.

We then performed zero-shot filtering to verify model performance on these transformed questions. This design preserves the original assessment objectives of testing the LLM’s contextual reasoning capabilities while guaranteeing answer objectivity and uniqueness. Crucially, it mitigates potential unreliability introduced by LLM-as-judge evaluation paradigms. Table 6 presents representative cases of these two question formats.

Table 4. Question Type and Difficulty Distribution Statistics   

<table><tr><td rowspan="2">Question Type</td><td rowspan="2">Difficulty Level</td><td colspan="2">Chinese Literature</td><td colspan="2">English Literature</td><td rowspan="2">Total</td></tr><tr><td>Water Margin</td><td>Dream of Red Chamber</td><td>Moby-Dick</td><td>Middlemarch</td></tr><tr><td rowspan="2">Objective</td><td>Simple (≤4 hops)</td><td>29</td><td>26</td><td>54</td><td>54</td><td>163</td></tr><tr><td>Complex (&gt;4 hops)</td><td>24</td><td>34</td><td>51</td><td>22</td><td>131</td></tr><tr><td rowspan="2">Subjective 
(Anonymous Reversion)</td><td>Simple (≤4 hops)</td><td>65</td><td>50</td><td>51</td><td>116</td><td>282</td></tr><tr><td>Complex (&gt;4 hops)</td><td>146</td><td>314</td><td>142</td><td>219</td><td>821</td></tr><tr><td colspan="2">Total</td><td>264</td><td>424</td><td>298</td><td>411</td><td>1397</td></tr></table>

Table 5. Comparations of anonymized and Non-anonymized questions   

<table><tr><td>Lang</td><td>Type</td><td>Question</td><td>Zero-shot LLM Response</td><td>Val</td></tr><tr><td rowspan="2">CHS</td><td>Raw Query</td><td>从史进拜师到最终离开少华山,期间共经历了哪些身份转变?</td><td>从富家子弟→王进徒弟→庄主→强盗同伙→落草为寇,共经历四次身份转变。</td><td>Correct</td></tr><tr><td>Annoy Query</td><td>从[PERSON#277]拜师到最终离开[LOCATION#759],期间共经历了哪些身份转变?</td><td>从拜师到离开[LOCATION#759],[PERSON#277]的关键身份转变可梳理如下:1.入门弟子阶段,初始以普通学徒身份拜入师门,主要承担基础杂役工作...</td><td>Wrong</td></tr><tr><td>ENG</td><td>Raw Query</td><td>What does the narrator&#x27;s reac-tion to Quequeg&#x27;s smoking in bed reveal about their relation-ship?</td><td>The narrator&#x27;s acceptance of Quequeg&#x27;s smoking re-veals their deepening friendship and shared comfort.</td><td>Correct</td></tr><tr><td></td><td>Annoy Query</td><td>What does the [PERSON#1999]&#x27;s reaction to [PERSON#200]&#x27;smoking in bed reveal about their relationship?</td><td>To analyze what [PERSON#1999]&#x27;s reaction to [PER-SON#200]&#x27;smoking in bed reveals about their rel-ationship, we would need more context about the specific reaction and...</td><td>Wrong</td></tr></table>

Table 6. Final Question-Answer Formats   

<table><tr><td>Lang</td><td>Question</td><td>Ground Truth</td></tr><tr><td colspan="3">Anonymity Reversion</td></tr><tr><td rowspan="2">CHS</td><td>请根据上下文对下面这段问答
Q:在[PERSON#532]离开[LOCATION#526]后，他在哪个村庄的酒店中与[PERSON#277]重逢？这个村庄附近的山上盘踞着哪两位头领？
A: [PERSON#532]在[LOCATION#110]附近的酒店与[PERSON#277]重逢，该村庄附近的[LOCATION#535]上盘踞[PERSON#503]和[PERSON#4]两位头领。
……</td><td rowspan="2">PERSON#532——鲁智深
PERSON#277——史进
PERSON#4——周通
PERSON#503——李忠
LOCATION#526——五台山
LOCATION#110——桃花村
LOCATION#535——桃花山</td></tr><tr><td>中已经被匿名化处理的所有人名和地名等进行推理，判断出被匿名的原本内容是哪些。</td></tr><tr><td rowspan="2">ENG</td><td>Please read the following QA pairs
Q: What does [PERSON#200]’s story about the wedding feast reveal about cultural misunderstandings?
A: The story reveals how cultural misunderstandings, such as [PERSON#588] mistaking the punchbowl for a finger-glass, can arise from ignorance of local customs.
……</td><td rowspan="2">PERSON#200——Queequeg
PERSON#588——captai</td></tr><tr><td>then for all anonymized Persons and Locations, perform inference to determine the original content that was anonymized.</td></tr><tr><td colspan="3">Multiple Choice</td></tr><tr><td>CHS</td><td>海棠诗社成立时，[PERSON#315]给自己取的别号是什么？这个别号与她居住的哪个场所相关？
A. [LOCATION#340]; [LOCATION#625]老农
B. [LOCATION#340]; [LOCATION#340]隐士
C. [LOCATION#625]老农；[LOCATION#340]
D. [LOCATION#340]居士；[LOCATION#625]老农</td><td>C. (李纨, 稻香老农, 稻香村)</td></tr><tr><td rowspan="2">ENG</td><td>Which two physical traits do [PERSON#1035] and her daughter [PERSON#445] share in common?</td><td rowspan="2">B. (Mrs. Garth, daughter Mary)</td></tr><tr><td>A. Straight hair and round faces
B. Curly hair and square faces
C. Wavy hair and oval faces
D. Short hair and triangular faces</td></tr></table>
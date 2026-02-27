# Pre-Storage Reasoning for Episodic Memory: Shifting Inference Burden to Memory for Personalized Dialogue

Sangyeop $\mathbf { K i m ^ { 1 , 2 * } }$ , Yohan Lee3*, Sanghwa $\mathbf { K i m ^ { 4 } }$ , Hyunjong $\mathbf { K i m ^ { 1 } }$ , Sungzoon Cho1†

1Seoul National University, 2Coxwave, 3Independent Researcher, 4KAIST

sy917kim@bdai.snu.ac.kr, yhlee.nlp@gmail.com, zoon@snu.ac.kr

# Abstract

Effective long-term memory in conversational AI requires synthesizing information across multiple sessions. However, current systems place excessive reasoning burden on response generation, making performance significantly dependent on model sizes. We introduce PRE-Mem (Pre-storage Reasoning for Episodic Memory), a novel approach that shifts complex reasoning processes from inference to memory construction. PREMem extracts finegrained memory fragments categorized into factual, experiential, and subjective information; it then establishes explicit relationships between memory items across sessions, capturing evolution patterns like extensions, transformations, and implications. By performing this reasoning during pre-storage rather than when generating a response, PREMem creates enriched representations while reducing computational demands during interactions. Experiments show significant performance improvements across all model sizes, with smaller models achieving results comparable to much larger baselines while maintaining effectiveness even with constrained token budgets. Code and dataset are available at https://github. com/sangyeop-kim/PREMem.

# 1 Introduction

Human cognition seamlessly synthesizes past experiences into coherent episodic memories that support personalized interactions (Piaget et al., 1952; Carey, 1985; Laird, 2012). When engaging with familiar people, individuals effortlessly perform relevant interactions, track evolving preferences, and maintain consistent mental models without explicitly reviewing conversation histories. This natural memory process enables meaningful relationships through contextualized understanding.

In conversational AI, well-designed memory structures are essential for maintaining personalized interactions across multiple sessions (Martins et al., 2022; Bae et al., 2022; Gutiérrez et al., 2024). Effective memory mechanisms allow AI assistants to track user preferences, recall shared experiences, and sustain consistent understanding over time—capabilities that form the foundation of truly personalized dialogue systems (Wu et al., 2025b; Fountas et al., 2025).

Current memory approaches in conversational AI systems rely on three core mechanisms (Wang et al., 2024b; Du et al., 2025): indexing and storing, retrieval, and memory-based generation. Recent advances have explored various structural granularities—from turn-level and session-level segmentation to compressed summaries (Pan et al., 2025) and knowledge graphs (Edge et al., 2025; Zhu et al., 2025). These approaches primarily investigate how different memory structures affect retrieval efficiency and accuracy, yet struggle with cross-session challenges that require understanding continuity, causality, and state changes.

Recent works (Xu et al., 2025; Gutiérrez et al., 2025) have attempted to address multi-session reasoning through metadata annotations and conceptlinking knowledge graphs. However, these methods typically define cross-session relationships as simple clusters without modeling the nature of relationships or temporal evolution.

Beyond these limitations of retrieval-focused approaches, a more critical challenge emerges even when retrieval succeeds. Even with optimal retrieval systems that can provide relevant context, models frequently struggle with complex reasoning tasks that require synthesis and inference—particularly temporal relationships and cross-session information integration (Mao et al., 2022; Yuan et al., 2025). Unlike simple information retrieval, these tasks demand sophisticated cognitive processes including pattern recognition, causal

reasoning, and contextual synthesis. This computational burden during response generation creates significant inefficiency and amplifies performance disparities between large and small models.

To address these challenges, we present PREMem (Pre-storage Reasoning for Episodic Memory), a cognitive science-grounded approach that shifts complex reasoning processes from response generation to memory construction. Our approach draws inspiration from human cognitive processes: rather than exhaustively reviewing conversation histories during interactions, humans rely on pre-consolidated memories that have undergone sophisticated synthesis during offline periods (Squire, 1987; Schacter and Addis, 2007). Based on schema theory (Rumelhart et al., 1976; Bartlett, 1995), human memory actively transforms information during storage through assimilation and accommodation processes, enabling efficient retrieval and coherent understanding across temporal contexts.

PREMem implements this cognitive principle by extracting memory fragments into three theoretically-grounded categories—factual, experiential, and subjective information—and establishing explicit cross-session relationships through five evolution patterns derived from schema modification mechanisms, as shown in Figure 1. By performing complex reasoning during pre-storage rather than at response time, our approach creates enriched memory representations while reducing computational demands during interactions—offering both performance gains and practical deployment advantages.

Experimental results on LongMemEval (Wu et al., 2025a) and LoCoMo (Maharana et al., 2024) benchmarks demonstrate significant improvements across all model sizes. PREMem shows particularly strong results on cross-session reasoning tasks, with even small language models $\scriptstyle ( \leq 4 \mathbf { B } )$ achieving competitive performance compared to much larger baseline models. Additional experiments confirm its practical applicability in resource-constrained environments through efficient token utilization.

Our contributions include: (1) A cognitive science-grounded memory framework based on established schema theory that extracts structured episodic fragments and models information evolution through five theoretically-validated patterns; (2) A pre-storage reasoning method that shifts complex cross-session synthesis from response time to memory construction, mirroring human cognitive consolidation processes; (3) Comprehensive experi-

mental validation across two benchmarks, multiple model families and question types, demonstrating robust generalization; (4) Practical advantages for resource-constrained applications through reduced inference-time computational requirements.

# 2 Related Works

# 2.1 Memory in Conversational AI Systems

Long-term memory in conversational AI systems requires integrating and updating experiences across multi-turn dialogues (Wang et al., 2024b; Du et al., 2025). Existing approaches employ unstructured formats such as summarization (Zhong et al., 2024; Wang et al., 2025) or compression (Pan et al., 2025; Chen et al., 2025), but struggle with temporal modeling and content overlap, leading to information loss and fragmented representations.

Knowledge graph-based methods (Edge et al., 2025; Guo et al., 2025; Zhu et al., 2025) enhance semantic connectivity through structured representations, but their partial graph construction prevents establishing relationships between temporally distant nodes across conversation sessions.

Recent efforts such as Li et al. (2025) and Ong et al. (2025) introduce modular memory architectures and timeline-based linking to better reflect dialogue dynamics. However, these approaches still perform memory relationship reasoning during response generation, making them heavily dependent on the capabilities of the underlying model.

Recent systems (Lee et al., 2024; Xu et al., 2025; Yuan et al., 2025) support dynamic memory evolution and attempt to establish connections between memories. However, they rely on implicit, unstructured associations rather than explicit schemas for modeling information evolution across sessions. This approach can lead to arbitrary links and inconsistent interpretations that are difficult to analyze.

We address these limitations with PREMem, a novel structured memory approach. Our method provides clear temporal relationships, well-defined semantic connections between related information, and systematically organized memory representations that enhance consistency, interpretability, and reasoning efficiency.

# 2.2 Cognitive Perspectives on Memory

Memory in AI-based conversational systems shares structural and functional characteristics with human memory, prompting researchers to incorporate cognitive science principles into memory system

![](images/8a807750e3b53dc3e49c59906acbf7c1773f2b0c86acc0309cff2772b5cc39b6.jpg)  
Memory Construction   
Figure 1: PREMem architecture divided into Memory Construction phase (comprising Step 1: Episodic Memory Extraction and Step 2: Pre-Storage Memory Reasoning) and Inference phase.

design (Wang et al., 2024a; Shan et al., 2025). This enables systems to maintain consistent user representations across multiple conversations.

Inspired by these cognitive principles, researchers have developed various methods for transforming conversation data into structured episodic memories (Hou et al., 2024; Fountas et al., 2025; Ge et al., 2025). Hou et al. (2024) models human memory consolidation by weighting information based on contextual relevance and recall frequency, while Fountas et al. (2025) applies event cognition principles to segment conversations using prediction errors and graph-theoretical clustering.

However, these approaches face limitations in cross-session reasoning, as they focus more on storage organization than on modeling information evolution across conversations (Qiu et al., 2024; Chu et al., 2024). Although systems like Xu et al. (2025) and Gutiérrez et al. (2025) attempt to address this through linked structures, they still struggle with tracking changing preferences and resolving contradictions (Huet et al., 2025; Wu et al., 2025a).

To overcome these limitations, we examine how humans reason about and synthesize memories. Cognitive science offers guidance through schema theory—detailed in Appendix B. This theory views memory as a structured interpretive system (Piaget et al., 1952; Rumelhart et al., 1976; Bartlett, 1995; Rumelhart, 2017). In this framework, new

information actively integrates with existing knowledge through generalization and exception handling (Fauconnier and Turner, 2008; Chi, 2009).

Based on these insights, our study not only structures conversations into temporal episodic units but also models the semantic relationships between them. This approach captures continuity, causality, and change patterns across conversations, enabling more consistent and personalized responses even as user preferences evolve over time.

# 3 Methodology

We present PREMem, a novel approach that shifts complex memory synthesis and analysis from response generation to the memory construction phase. By performing pre-storage reasoning across conversations, our approach reduces the computational burden during dialogue while creating more cognitive-inspired memory representations. Figure 1 illustrates the overall architecture of our approach, which consists of a Memory Construction phase (with two steps detailed in the following sections) and an Inference phase. This method improves personalized conversation performance across all model sizes, with smaller models $\scriptstyle ( \leq 4 \mathbf { B } )$ achieving results comparable to baselines using much larger models. All prompts and pseudo code can be found in Appendix A and F, respectively.

# 3.1 Step 1: Episodic Memory Extraction

We extract episodic memory from conversation history, classifying it into three categories that reflect human memory components (Squire, 1987; Schacter and Tulving, 1994):

• Factual Information: Objective facts about personal states, attributes, possessions, and relationships (“what I am/have/know”)   
• Experiential Information: Events, actions, and interactions experienced over time (“what I did/experienced”)   
• Subjective Information: Internal states including preferences, opinions, beliefs, goals, and plans (“what I like/think/want”)

Beyond comprehensive categorization, effective memory structure needs to solve the challenge of temporal reasoning—determining accurate time relationships. Previous research (Xu et al., 2025) shows language models struggle with relative time expressions such as “yesterday” and “last week”. We address this through a structured temporal representation with four patterns: (1) ongoing facts use message dates directly; (2) specific past events convert relative expressions to absolute dates; (3) unclear past events use “Before [message-date]”; and (4) future plans use “After [message-date].”

We formalize memory extraction through $L L M _ { e x t r a c t }$ which operates on conversation sessions $S _ { 1 } , S _ { 2 } , \cdots , S _ { N }$ :

$$
L L M _ {e x t r a c t} (S _ {i}) \to \{m _ {i} ^ {1}, m _ {i} ^ {2}, \dots , m _ {i} ^ {n _ {i}} \},
$$

where $n _ { i }$ is the number of memory fragments in session $S _ { i }$ . Each memory fragment $\dot { m } _ { i } ^ { j }$ includes source identification, key phrase, memory content, and temporal context:

$$
m _ {i} ^ {j} = \left(\operatorname {i d} _ {i} ^ {j}, \operatorname {k e y} _ {i} ^ {j}, \operatorname {c o n t e n t} _ {i} ^ {j}, \operatorname {t i m e} _ {i} ^ {j}\right).
$$

# 3.2 Step 2: Pre-Storage Memory Reasoning

From memory fragments, we analyze relationships between information across conversation sessions using cognitive schema theory (Rumelhart et al., 1976; Anderson, 2013; Meylani, 2024). This approach shifts complex cognitive tasks—including pattern recognition, information synthesis, and contextual reasoning—to the storage phase, reducing computational demands during dialogue while creating enriched memory representations with inferred relationships and implications.

# 3.2.1 Clustering and Temporal Linking

We organize memory fragments into semantic clusters using embeddings generated from combined key phrases and memory content. For each session $S _ { i }$ , we embed the memory fragments $\{ m _ { i } ^ { j } \} _ { j = 1 } ^ { n _ { i } }$ into vectors $\{ e _ { i } ^ { j } \} _ { j = 1 } ^ { n _ { i } }$ using an embedding model $f _ { e m b }$ that is, $e _ { i } ^ { j } = f _ { e m b } ( m _ { i } ^ { j } )$ . Using silhouette scores to determine optimal groupings, we form clusters $C _ { i } = \{ c _ { i } ^ { 1 } , c _ { i } ^ { 2 } , . . . , c _ { i } ^ { k _ { i } } \}$ with each cluster containing embedding of semantically related memory items. This clustering step serves two critical purposes: it reduces redundancy in memory representations to minimize noise during reasoning (Pan et al., 2025), and it prevents combinatorial explosion by limiting the number of cross-session comparisons required during relationship analysis.

For a cluster $c$ , the centroid is calculated as $\bar { c } =$ $\frac { 1 } { | c | } \sum _ { e \in c } e$ and the collection of memory fragments corresponding to the cluster $c$ is denoted as $M _ { c }$ .

We maintain a persistent memory pool $P _ { i }$ of clusters that have not yet found a semantic match with a cluster that comes after themselves up to the $i$ -th session, initialized as $P _ { 0 } = \{ \}$ . For each new session $S _ { i }$ , we measure the similarity between existing persistent cluster $p \in P _ { i - 1 }$ and new cluster $c \in C _ { i }$ using the cosine similarity of centroids:

$$
\operatorname {s i m} (p, c) = \frac {\bar {p} \cdot \bar {c}}{| | \bar {p} | | \cdot | | \bar {c} | |}.
$$

We define a pair $( p , c )$ as connected if $s i m ( p , c ) > \theta$ . We define a set $C P _ { i }$ that contains connected pairs $( p , c )$ , that is,

$$
C P _ {i} := \{(p, c): \operatorname {s i m} (p, c) > \theta \}
$$

$$
w h e r e p \in P _ {i - 1}, c \in C _ {i}
$$

The set $C P _ { i }$ consists of semantically related cluster pairs across sessions.

# 3.2.2 Cross-Session Reasoning Patterns

For each identified connection, we perform crosssession reasoning based on five information evolution patterns derived from schema modification mechanisms (Rumelhart et al., 1976; Anderson, 2013). These patterns synthesize findings from extensive cognitive science literature (Bransford and Johnson, 1972; Chi et al., 1981; Murphy, 2004; Chi, 2009) to capture fundamental ways humans integrate new information with existing knowledge structures. The detailed theoretical foundations are provided in Appendix B.

• Extension/Generalization: Expanding scope of existing information (e.g., inferring broader food preferences from restaurant choices)   
• Accumulation: Reinforcing knowledge through repeated similar information (e.g., recognizing consistent exercise habits)   
• Specification/Refinement: Developing more detailed understanding (e.g., clarifying music preferences from general to specific)   
• Transformation: Capturing changes in states or preferences (e.g., identifying shifts in product satisfaction)   
• Connection/Implication: Discovering relationships between separate information (e.g., linking language study with travel plans)

The model $L L M _ { r e a s o n }$ generates reasoning memory fragments by analyzing memory fragments in $M _ { p }$ and $M _ { c }$ for $( p , c ) \in C P _ { i }$ individually, extracting insights about the evolution patterns:

$$
L L M _ {r e a s o n} (M _ {p}, M _ {c}) \rightarrow \{r _ {p, c} ^ {j} \} _ {j = 1} ^ {d _ {p, c}},
$$

where $r _ { p , c } ^ { j }$ is the reasoning memory fragment that follows the same structure as memory fragments. We define a reasoning memory pof reasoning memory fragments $\{ r _ { p , c } ^ { j } \} _ { j = 1 } ^ { d _ { p , c } }$ $R _ { i }$ he unionover all connected pairs $( p , c ) \in C P _ { i }$ and denote embedding of $R _ { i }$ using embedding model $f _ { e m b }$ as $E _ { i } ^ { \prime }$ .

After reasoning on the pair $( p , c ) \in C P _ { i }$ , we remove $p$ from the persistent memory pool since it finds a semantic match with later-coming cluster $c$ . On the other hand, we put all latest clusters $c \in C _ { i }$ into the pool, then we get the updated persistent memory pool $P _ { i }$ , which is formally defined as:

$$
P _ {i} = P _ {i - 1} \setminus \{p: \exists c. s. t. (p, c) \in C P _ {i} \} \cup C _ {i}.
$$

This process serves two important purposes: first, it prevents computational explosion as sessions increase by eliminating already-processed information; second, it enables efficient long-term topic tracking across temporally distant conversations.

After this whole process is performed on the last conversation session $S _ { N }$ , we prepare memory storage $\mathcal { M }$ and reasoning memory storage $\mathcal { R }$ used in inference as $\mathcal { M } : = \bar { \cup } _ { i = 1 } ^ { N } \{ m _ { i } ^ { j } \} _ { j = 1 } ^ { n _ { i } }$ and $\mathcal { R } : =$ $\cup _ { i = 1 } ^ { N } R _ { i }$ ; and denote their embeddings using $f _ { e m b }$ as $E$ and $E ^ { \prime }$ , respectively.

# 3.3 Inference Phase

For a user query $( q )$ , we retrieve the most relevant items from our total memory store $\mathcal { M } \cup \mathcal { R }$ and select the top- $\mathbf { \nabla } \cdot \mathbf { k }$ items based on the similarity between embedded vectors $e \in ( E \cup E ^ { \prime } )$ and $f _ { e m b } ( q )$ . These retrieved memory items denoted by $m _ { * } ^ { 1 } , \ldots , m _ { * } ^ { k }$ are arranged chronologically and composed to form the context, with each item including its complete information (key, content, time). We then generate a response using this organized context:

$$
L L M _ {r e s p o n s e} (\text {c o n t e x t}, q) \rightarrow \text {r e s p o n s e}.
$$

# 4 Experiments

# 4.1 Experimental Setup

Table 1: Statistics of dataset category.   

<table><tr><td>Dataset</td><td>Category</td><td># Questions</td></tr><tr><td rowspan="4">LoCoMo</td><td>single-hop</td><td>1,123 (56.5%)</td></tr><tr><td>multi-hop</td><td>321 (16.1%)</td></tr><tr><td>temporal-reasoning</td><td>96 (4.8%)</td></tr><tr><td>adversarial</td><td>446 (22.4%)</td></tr><tr><td rowspan="5">LongMemEval</td><td>single-hop</td><td>150 (30.0%)</td></tr><tr><td>multi-hop</td><td>121 (24.2%)</td></tr><tr><td>temporal-reasoning</td><td>127 (25.4%)</td></tr><tr><td>adversarial</td><td>30 (6.0%)</td></tr><tr><td>knowledge-update</td><td>72 (14.4%)</td></tr></table>

Datasets We utilize two long-term memory QA datasets: LoCoMo (Maharana et al., 2024) and LongMemEval (Wu et al., 2025a). LoCoMo contains 1,986 QA instances from conversation history sets, averaging 27.2 dialogues per set with 21.6 turns per dialogue. LongMemEval has 500 QA pairs. We adopt the LongMemEval subset, which reflects more realistic constraints. LongMemEval averages 115K tokens per question.

We unify the question types across both datasets into five categories: single-hop, multi-hop, temporal reasoning, adversarial, and knowledge update (only in LongMemEval). Detailed dataset statistics for each category are provided in Table 1, and comprehensive information about the datasets, including unification criteria, is described in Appendix C.

To ensure a fair comparison across models and settings, we standardize the answer generation prompt for all experiments. The specific prompts used for each dataset are shown in Appendix A.

Evaluation Metrics We evaluate using BLEU-1, ROUGE-1, ROUGE-L, METEOR, BERTScore, and LLM-as-a-judge score. BLEU-1 measures n-gram precision while ROUGE metrics assess lexical overlap through n-grams. METEOR and

Table 2: Performance comparison across different model sizes and memory frameworks. Results show LLMjudge scores (LLM), ROUGE-1 (R1), and adversarial accuracy (Acc). Highest scores in bold and second highest underlined. Additional metrics (BLEU-1, ROUGE-L, METEOR, BERTScore) available in Appendix D.   

<table><tr><td rowspan="3" colspan="2">Model</td><td rowspan="3">Method</td><td colspan="11">LongMemEval</td><td colspan="8">LoCoMo</td><td></td></tr><tr><td colspan="2">Total</td><td colspan="2">Single-hop</td><td colspan="2">Multi-hop</td><td colspan="2">Temporal</td><td colspan="2">Knowledge</td><td>Adv</td><td colspan="2">Total</td><td colspan="2">Single-hop</td><td colspan="2">Multi-hop</td><td colspan="2">Temporal</td><td>Adv</td></tr><tr><td>LLM</td><td>R1</td><td>LLM</td><td>R1</td><td>LLM</td><td>R1</td><td>LLM</td><td>R1</td><td>LLM</td><td>R1</td><td>Acc</td><td>LLM</td><td>R1</td><td>LLM</td><td>R1</td><td>LLM</td><td>R1</td><td>LLM</td><td>R1</td><td>Acc</td></tr><tr><td rowspan="12">Qwen2.5</td><td rowspan="6">14B</td><td>Turn</td><td>39.7</td><td>25.3</td><td>59.4</td><td>42.6</td><td>28.3</td><td>9.0</td><td>19.5</td><td>23.5</td><td>42.5</td><td>23.5</td><td>73.3</td><td>61.6</td><td>28.9</td><td>74.6</td><td>42.7</td><td>49.5</td><td>15.9</td><td>47.9</td><td>14.3</td><td>46.9</td></tr><tr><td>Session</td><td>29.0</td><td>19.3</td><td>53.1</td><td>37.2</td><td>16.7</td><td>5.7</td><td>13.2</td><td>17.1</td><td>15.8</td><td>9.5</td><td>66.7</td><td>54.5</td><td>25.7</td><td>57.7</td><td>36.5</td><td>38.4</td><td>12.9</td><td>42.3</td><td>12.8</td><td>67.0</td></tr><tr><td>SeCom</td><td>37.6</td><td>24.5</td><td>60.1</td><td>42.9</td><td>26.0</td><td>9.8</td><td>19.0</td><td>23.5</td><td>35.3</td><td>17.3</td><td>73.3</td><td>60.4</td><td>31.0</td><td>72.9</td><td>45.8</td><td>36.4</td><td>15.1</td><td>50.2</td><td>16.3</td><td>57.4</td></tr><tr><td>HippoRAG-2</td><td>44.7</td><td>29.2</td><td>68.9</td><td>48.9</td><td>26.1</td><td>9.6</td><td>20.8</td><td>25.5</td><td>59.3</td><td>31.5</td><td>75.9</td><td>61.7</td><td>30.4</td><td>69.8</td><td>44.3</td><td>45.6</td><td>15.6</td><td>54.7</td><td>14.0</td><td>64.4</td></tr><tr><td>A-Mem</td><td>50.3</td><td>33.0</td><td>72.4</td><td>53.0</td><td>34.0</td><td>14.0</td><td>30.0</td><td>26.5</td><td>63.9</td><td>38.2</td><td>66.7</td><td>43.6</td><td>30.2</td><td>52.4</td><td>34.5</td><td>44.8</td><td>29.2</td><td>38.2</td><td>14.9</td><td>23.7</td></tr><tr><td>PREMem</td><td>64.7</td><td>40.4</td><td>59.5</td><td>43.3</td><td>75.7</td><td>35.0</td><td>48.6</td><td>38.8</td><td>88.3</td><td>56.9</td><td>70.0</td><td>68.0</td><td>29.4</td><td>69.2</td><td>38.0</td><td>74.1</td><td>30.1</td><td>55.5</td><td>18.0</td><td>69.1</td></tr><tr><td rowspan="6">72B</td><td>Turn</td><td>40.6</td><td>26.2</td><td>60.8</td><td>43.4</td><td>30.2</td><td>13.8</td><td>20.6</td><td>22.8</td><td>46.5</td><td>21.8</td><td>60.0</td><td>63.7</td><td>26.3</td><td>77.4</td><td>38.0</td><td>51.4</td><td>15.3</td><td>60.2</td><td>17.7</td><td>47.3</td></tr><tr><td>Session</td><td>30.9</td><td>20.8</td><td>54.8</td><td>38.4</td><td>20.0</td><td>10.8</td><td>13.0</td><td>17.1</td><td>17.7</td><td>9.1</td><td>73.3</td><td>54.2</td><td>23.9</td><td>60.4</td><td>33.6</td><td>40.3</td><td>12.4</td><td>54.4</td><td>15.8</td><td>53.4</td></tr><tr><td>SeCom</td><td>39.4</td><td>25.3</td><td>56.2</td><td>40.9</td><td>31.4</td><td>14.9</td><td>22.4</td><td>21.4</td><td>43.9</td><td>22.2</td><td>56.7</td><td>58.5</td><td>28.0</td><td>72.7</td><td>41.4</td><td>34.0</td><td>12.6</td><td>54.4</td><td>17.7</td><td>47.7</td></tr><tr><td>HippoRAG-2</td><td>45.9</td><td>29.8</td><td>70.6</td><td>49.9</td><td>29.6</td><td>11.8</td><td>21.8</td><td>24.6</td><td>57.1</td><td>32.7</td><td>69.0</td><td>61.6</td><td>30.4</td><td>72.1</td><td>44.4</td><td>44.6</td><td>16.0</td><td>62.3</td><td>16.6</td><td>57.0</td></tr><tr><td>A-Mem</td><td>53.6</td><td>36.2</td><td>73.0</td><td>55.4</td><td>38.1</td><td>16.5</td><td>39.1</td><td>31.6</td><td>66.2</td><td>41.2</td><td>60.0</td><td>45.6</td><td>31.7</td><td>54.6</td><td>36.2</td><td>49.8</td><td>32.7</td><td>42.6</td><td>16.2</td><td>21.5</td></tr><tr><td>PREMem</td><td>67.5</td><td>45.4</td><td>66.6</td><td>47.1</td><td>79.6</td><td>45.1</td><td>51.6</td><td>43.1</td><td>85.9</td><td>58.8</td><td>56.7</td><td>71.0</td><td>27.0</td><td>73.0</td><td>33.7</td><td>76.7</td><td>30.0</td><td>61.8</td><td>17.1</td><td>68.8</td></tr><tr><td rowspan="12">genna-3</td><td rowspan="6">12B</td><td>Turn</td><td>36.6</td><td>22.4</td><td>55.6</td><td>40.2</td><td>24.2</td><td>9.1</td><td>22.2</td><td>17.4</td><td>46.4</td><td>22.3</td><td>40.0</td><td>45.5</td><td>27.0</td><td>65.3</td><td>42.0</td><td>40.1</td><td>13.6</td><td>28.8</td><td>6.7</td><td>3.8</td></tr><tr><td>Session</td><td>28.8</td><td>16.9</td><td>51.0</td><td>34.8</td><td>16.7</td><td>7.0</td><td>15.8</td><td>11.1</td><td>15.1</td><td>8.4</td><td>70.0</td><td>38.0</td><td>24.2</td><td>51.6</td><td>37.1</td><td>33.1</td><td>12.6</td><td>22.0</td><td>6.9</td><td>11.7</td></tr><tr><td>SeCom</td><td>37.4</td><td>23.5</td><td>55.6</td><td>39.8</td><td>26.3</td><td>10.2</td><td>23.1</td><td>19.5</td><td>43.6</td><td>24.9</td><td>50.0</td><td>44.8</td><td>30.1</td><td>65.4</td><td>47.5</td><td>35.7</td><td>13.0</td><td>28.1</td><td>8.0</td><td>4.0</td></tr><tr><td>HippoRAG-2</td><td>43.9</td><td>27.1</td><td>66.2</td><td>48.2</td><td>30.0</td><td>8.3</td><td>24.1</td><td>21.2</td><td>58.1</td><td>31.0</td><td>48.3</td><td>44.3</td><td>29.7</td><td>61.8</td><td>45.9</td><td>38.9</td><td>15.6</td><td>29.1</td><td>6.4</td><td>8.9</td></tr><tr><td>A-Mem</td><td>39.0</td><td>28.1</td><td>65.5</td><td>51.1</td><td>22.2</td><td>10.2</td><td>22.8</td><td>24.3</td><td>49.0</td><td>23.8</td><td>33.3</td><td>43.9</td><td>31.2</td><td>47.2</td><td>31.4</td><td>40.7</td><td>20.2</td><td>22.4</td><td>6.3</td><td>43.8</td></tr><tr><td>PREMem</td><td>57.7</td><td>34.4</td><td>54.3</td><td>38.4</td><td>63.3</td><td>27.1</td><td>47.3</td><td>31.9</td><td>86.3</td><td>52.2</td><td>46.7</td><td>50.0</td><td>30.1</td><td>61.7</td><td>43.1</td><td>63.9</td><td>27.4</td><td>36.6</td><td>11.2</td><td>15.5</td></tr><tr><td rowspan="6">27B</td><td>Turn</td><td>38.0</td><td>23.6</td><td>56.4</td><td>41.5</td><td>25.1</td><td>8.9</td><td>21.8</td><td>20.0</td><td>44.3</td><td>22.7</td><td>66.7</td><td>49.7</td><td>27.5</td><td>67.7</td><td>42.6</td><td>39.5</td><td>14.1</td><td>30.6</td><td>7.7</td><td>17.3</td></tr><tr><td>Session</td><td>27.6</td><td>16.8</td><td>50.6</td><td>36.2</td><td>14.3</td><td>5.1</td><td>13.5</td><td>10.0</td><td>12.5</td><td>8.3</td><td>73.3</td><td>43.3</td><td>24.6</td><td>53.1</td><td>37.0</td><td>32.0</td><td>11.2</td><td>22.6</td><td>8.3</td><td>32.7</td></tr><tr><td>SeCom</td><td>38.9</td><td>23.2</td><td>55.4</td><td>39.2</td><td>30.1</td><td>10.4</td><td>24.0</td><td>19.6</td><td>40.8</td><td>22.8</td><td>63.3</td><td>49.1</td><td>30.2</td><td>67.5</td><td>48.0</td><td>33.2</td><td>10.9</td><td>30.0</td><td>10.6</td><td>19.7</td></tr><tr><td>HippoRAG-2</td><td>43.1</td><td>27.3</td><td>65.0</td><td>47.4</td><td>28.4</td><td>12.4</td><td>22.8</td><td>21.0</td><td>56.5</td><td>28.6</td><td>63.3</td><td>49.5</td><td>30.6</td><td>64.7</td><td>46.8</td><td>37.7</td><td>16.4</td><td>28.8</td><td>7.1</td><td>26.2</td></tr><tr><td>A-Mem</td><td>45.3</td><td>31.9</td><td>66.2</td><td>54.5</td><td>30.5</td><td>10.6</td><td>28.9</td><td>26.1</td><td>61.7</td><td>39.2</td><td>43.3</td><td>44.5</td><td>32.8</td><td>48.7</td><td>32.9</td><td>43.2</td><td>28.7</td><td>24.2</td><td>7.2</td><td>40.0</td></tr><tr><td>PREMem</td><td>61.9</td><td>39.2</td><td>52.7</td><td>39.8</td><td>69.6</td><td>33.4</td><td>51.2</td><td>38.5</td><td>91.3</td><td>60.1</td><td>66.7</td><td>54.6</td><td>30.6</td><td>62.5</td><td>40.3</td><td>57.0</td><td>25.2</td><td>34.8</td><td>8.8</td><td>38.3</td></tr><tr><td rowspan="12">gpt-4.1</td><td rowspan="6">mini</td><td>Turn</td><td>39.5</td><td>25.4</td><td>62.8</td><td>43.8</td><td>27.6</td><td>11.8</td><td>20.4</td><td>21.3</td><td>45.5</td><td>23.7</td><td>46.7</td><td>54.7</td><td>30.3</td><td>74.3</td><td>45.8</td><td>50.3</td><td>17.7</td><td>49.8</td><td>17.5</td><td>10.5</td></tr><tr><td>Session</td><td>29.7</td><td>18.0</td><td>54.4</td><td>37.3</td><td>18.4</td><td>6.2</td><td>13.2</td><td>12.8</td><td>17.6</td><td>9.0</td><td>66.7</td><td>48.1</td><td>27.5</td><td>58.5</td><td>41.3</td><td>41.4</td><td>15.4</td><td>38.2</td><td>16.9</td><td>30.3</td></tr><tr><td>SeCom</td><td>42.3</td><td>26.8</td><td>64.3</td><td>45.2</td><td>33.5</td><td>13.6</td><td>24.8</td><td>24.2</td><td>41.6</td><td>21.2</td><td>60.0</td><td>53.4</td><td>33.2</td><td>74.2</td><td>51.6</td><td>40.1</td><td>17.0</td><td>42.0</td><td>11.7</td><td>14.1</td></tr><tr><td>HippoRAG-2</td><td>44.8</td><td>29.1</td><td>69.5</td><td>48.3</td><td>30.4</td><td>12.2</td><td>19.5</td><td>23.2</td><td>56.3</td><td>33.1</td><td>72.0</td><td>54.6</td><td>34.1</td><td>70.0</td><td>50.2</td><td>52.9</td><td>25.0</td><td>42.6</td><td>17.0</td><td>23.3</td></tr><tr><td>A-Mem</td><td>53.9</td><td>35.5</td><td>75.1</td><td>57.4</td><td>41.6</td><td>16.5</td><td>34.1</td><td>27.9</td><td>67.4</td><td>39.0</td><td>66.7</td><td>52.7</td><td>37.0</td><td>56.1</td><td>36.8</td><td>61.1</td><td>38.0</td><td>38.5</td><td>11.0</td><td>42.5</td></tr><tr><td>PREMem</td><td>67.6</td><td>43.2</td><td>56.4</td><td>40.5</td><td>76.5</td><td>41.7</td><td>62.4</td><td>44.3</td><td>88.6</td><td>60.4</td><td>63.3</td><td>64.9</td><td>34.5</td><td>69.4</td><td>46.7</td><td>77.9</td><td>36.9</td><td>50.3</td><td>18.2</td><td>48.9</td></tr><tr><td rowspan="6">base</td><td>Turn</td><td>40.7</td><td>25.2</td><td>61.8</td><td>43.6</td><td>25.8</td><td>9.6</td><td>24.1</td><td>22.5</td><td>47.5</td><td>23.9</td><td>56.7</td><td>57.1</td><td>31.3</td><td>76.3</td><td>45.9</td><td>54.7</td><td>21.7</td><td>53.4</td><td>20.5</td><td>12.8</td></tr><tr><td>Session</td><td>30.3</td><td>18.3</td><td>54.9</td><td>37.6</td><td>20.6</td><td>9.0</td><td>10.3</td><td>11.0</td><td>14.8</td><td>9.2</td><td>76.7</td><td>50.1</td><td>27.9</td><td>59.6</td><td>40.6</td><td>42.2</td><td>17.0</td><td>49.1</td><td>20.7</td><td>34.1</td></tr><tr><td>SeCom</td><td>42.0</td><td>26.2</td><td>63.3</td><td>44.2</td><td>32.9</td><td>13.2</td><td>20.3</td><td>21.4</td><td>49.0</td><td>24.2</td><td>60.0</td><td>56.7</td><td>35.0</td><td>76.2</td><td>52.9</td><td>42.5</td><td>19.0</td><td>50.6</td><td>19.7</td><td>20.9</td></tr><tr><td>HippoRAG-2</td><td>45.2</td><td>29.2</td><td>70.3</td><td>50.5</td><td>28.2</td><td>11.3</td><td>19.1</td><td>21.6</td><td>58.3</td><td>34.6</td><td>76.7</td><td>57.3</td><td>34.0</td><td>71.6</td><td>49.8</td><td>49.4</td><td>22.4</td><td>54.9</td><td>22.6</td><td>30.9</td></tr><tr><td>A-Mem</td><td>55.9</td><td>37.5</td><td>78.0</td><td>61.3</td><td>41.4</td><td>17.0</td><td>37.8</td><td>30.9</td><td>64.7</td><td>39.2</td><td>66.7</td><td>49.5</td><td>34.7</td><td>55.6</td><td>36.6</td><td>58.6</td><td>39.8</td><td>39.6</td><td>11.5</td><td>30.9</td></tr><tr><td>PREMem</td><td>71.4</td><td>44.6</td><td>58.5</td><td>40.9</td><td>83.5</td><td>44.0</td><td>64.4</td><td>44.8</td><td>93.7</td><td>64.8</td><td>73.3</td><td>67.7</td><td>35.9</td><td>71.5</td><td>48.5</td><td>76.0</td><td>36.4</td><td>50.2</td><td>19.4</td><td>57.4</td></tr></table>

BERTScore capture semantic similarity beyond exact matches. LLM-as-a-judge score assesses overall response quality including coherence and informativeness, critical for LongMemEval and LoCoMo tasks that require recalling information from past interactions. For adversarial QA categories, we report accuracy based on the proportion of safe responses that identify unanswerable queries.

Baselines We compare our approach against baselines with varying memory granularity and state-of-the-art models. For granularity, we implement turn-level and session-level memory structures. For advanced approaches, we evaluate SeCom (Pan et al., 2025), which partitions dialogue into topic-based segments with compressionbased denoising; HippoRAG-2 (Gutiérrez et al., 2025), which encodes memory as an open knowledge graph with concept-context structures; and A-Mem (Xu et al., 2025), which organizes interconnected, evolving notes with semantic metadata.

Implementation Details In PREMem, we use identical LLMs for extraction and reasoning,

using the largest variant per family: Qwen2.5- 72B, Gemma3-27B, or gpt-4.1-base (“base” distinguishes from smaller variants). For LLMresponse, we evaluate across three LLM families—gpt-4.1 (OpenAI, 2025) (nano, mini, base), Qwen2.5 (Yang et al., 2024) (3B, 14B, 72B), and Gemma3 (Team et al., 2025) (4B, 12B, 27B)—to assess generalizability across different model capacities. During response generation, all models operate with a temperature of 0.7. LLM-as-a-judge score uses a deterministic decoding (temperature 0.0). We use Stella_en_400M_v5 (Zhang et al., 2025) as the embedding model to encode memory items and queries during retrieval. Additional implementation details are provided in Appendix E.

# 4.2 Main Results

Table 2 shows comprehensive results across LongMemEval and LoCoMo benchmarks using LLM-as-a-judge scores and ROUGE-1. PREMem achieves superior performance across most categories and model sizes, especially in complex reasoning tasks. For overall performance, PREMem

consistently outperforms all baselines by substantial margins across both benchmarks.

Results highlight two key findings. First, PRE-Mem demonstrates exceptionally strong performance on challenging cross-session reasoning tasks—multi-hop questions, temporal reasoning, and knowledge update categories. Second, while some baselines excel in specific subcategories (e.g., A-Mem on single-hop questions), PREMem delivers more consistent performance enhancement across all question types, maintaining robust results regardless of question complexity.

# 4.3 Small Language Models

Table 3: Small models with PREMem vs. larger models with baselines (LLM-as-a-judge scores).   

<table><tr><td>Method</td><td>Model</td><td>LongMemEval</td><td>LoCoMo</td></tr><tr><td>Turn</td><td></td><td>40.6</td><td>63.7</td></tr><tr><td>Session</td><td></td><td>30.9</td><td>54.2</td></tr><tr><td>SeCom</td><td>Qwen2.5 72B</td><td>39.4</td><td>58.5</td></tr><tr><td>HippoRAG-2</td><td></td><td>45.9</td><td>61.6</td></tr><tr><td>A-Mem</td><td></td><td>53.6</td><td>45.6</td></tr><tr><td>PREMem</td><td>Qwen2.5 3B</td><td>50.8</td><td>53.8</td></tr><tr><td>Turn</td><td></td><td>38.0</td><td>49.7</td></tr><tr><td>Session</td><td></td><td>27.6</td><td>33.3</td></tr><tr><td>SeCom</td><td>gemma-3 27B</td><td>38.9</td><td>49.1</td></tr><tr><td>HippoRAG-2</td><td></td><td>43.1</td><td>49.5</td></tr><tr><td>A-Mem</td><td></td><td>45.3</td><td>44.5</td></tr><tr><td>PREMem</td><td>gemma-3 4B</td><td>53.4</td><td>50.1</td></tr><tr><td>Turn</td><td></td><td>40.7</td><td>57.1</td></tr><tr><td>Session</td><td></td><td>30.3</td><td>50.1</td></tr><tr><td>SeCom</td><td>gpt-4.1</td><td>42.0</td><td>56.7</td></tr><tr><td>HippoRAG-2</td><td></td><td>45.2</td><td>57.3</td></tr><tr><td>A-Mem</td><td></td><td>55.9</td><td>49.5</td></tr><tr><td>PREMem</td><td>gpt-4.1 nano</td><td>58.7</td><td>58.8</td></tr></table>

Table 3 shows LLM-as-a-judge scores comparing PREMem with small models against baseline methods using larger models. The results demon-

strate that PREMem enables competitive performance even under limited model capacity.

In Gemma and gpt families, PREMem with smaller models outperforms baselines using larger counterparts across both benchmarks. For the Qwen family, all memory methods using Qwen2.5- 3B achieve scores below 50 on both benchmarks, except for PREMem which reaches 50.8 on Long-MemEval. With Qwen2.5-14B (Table 2), PREMem performance surpasses all baseline methods that use the much larger 72B model on both benchmarks. By offloading complex reasoning to the storage phase, PREMem enhances lightweight models with rich memory representations, reducing reliance on large-scale inference models.

# 4.4 Ablation Study

Table 4 shows ablation studies of PREMem. Step 1 (memory extraction) is vital, as its removal drops scores by $3 2 . 7 – 6 9 . 0 \%$ ; similarly, Step 2 (prestorage reasoning) proves valuable through crosssession pattern analysis. Our episodic memory categorization and temporal reasoning also contribute meaningfully, with their removal decreasing scores by up to $8 . 9 \%$ and $1 6 . 4 \%$ respectively.

These results confirm two key insights. First, our structured approach for memory extraction effectively organizes user information into meaningful categories. Second, performing cross-session reasoning before retrieval time significantly enhances performance across all model sizes. By shifting complex cognitive processes to the memory construction phase, models can focus on response generation during inference, leading to more effective handling of temporal relationships and multisession information synthesis.

Table 4: Ablation study of PREMem Components. Bold: ${ > } 1 0 \%$ drop, underlined: $5 . 1 0 \%$ drop from PREMem.   

<table><tr><td rowspan="3">Method</td><td colspan="6">LLM</td><td colspan="6">R1</td></tr><tr><td colspan="2">Qwen2.5</td><td colspan="2">gemma-3</td><td colspan="2">gpt-4.1</td><td colspan="2">Qwen2.5</td><td colspan="2">gemma-3</td><td colspan="2">gpt-4.1</td></tr><tr><td>14B</td><td>72B</td><td>12B</td><td>27B</td><td>mini</td><td>base</td><td>14B</td><td>72B</td><td>12B</td><td>27B</td><td>mini</td><td>base</td></tr><tr><td colspan="13">LongMemEval</td></tr><tr><td>PREMem</td><td>64.7</td><td>67.5</td><td>57.7</td><td>61.9</td><td>67.6</td><td>71.4</td><td>40.4</td><td>45.4</td><td>34.4</td><td>39.2</td><td>43.2</td><td>44.6</td></tr><tr><td>w/o Step 2</td><td>65.0(+0.5%)</td><td>69.8(+3.5%)</td><td>57.4(-0.6%)</td><td>59.9(-3.2%)</td><td>67.9(+0.4%)</td><td>69.8(-2.2%)</td><td>39.6(-2.1%)</td><td>45.2(-0.3%)</td><td>32.5(-5.5%)</td><td>38.1(-2.7%)</td><td>43.3(+0.3%)</td><td>43.4(-2.7%)</td></tr><tr><td>w/o Step 1</td><td>31.2(-51.8%)</td><td>35.9(-46.8%)</td><td>23.5(-59.3%)</td><td>24.1(-61.0%)</td><td>31.9(-52.8%)</td><td>34.2(-52.1%)</td><td>17.2(-57.4%)</td><td>18.4(-59.5%)</td><td>11.1(-67.8%)</td><td>12.2(-69.0%)</td><td>15.8(-63.5%)</td><td>17.2(-61.5%)</td></tr><tr><td>w/o Step 1 Categories</td><td>64.3(-0.7%)</td><td>68.9(+2.0%)</td><td>56.3(-2.4%)</td><td>59.9(-3.2%)</td><td>66.7(-1.3%)</td><td>69.6(-2.4%)</td><td>40.3(-0.3%)</td><td>45.7(+0.8%)</td><td>33.0(-4.1%)</td><td>38.1(-2.7%)</td><td>41.9(-2.9%)</td><td>42.9(-3.7%)</td></tr><tr><td>w/o Temporal Reasoning</td><td>63.7(-1.6%)</td><td>68.4(+1.3%)</td><td>56.0(-3.0%)</td><td>58.5(-3.4%)</td><td>66.2(-2.1%)</td><td>69.0(-3.4%)</td><td>39.1(-3.2%)</td><td>44.8(-1.2%)</td><td>30.8(-10.6%)</td><td>36.5(-6.8%)</td><td>42.9(-0.6%)</td><td>43.9(-1.6%)</td></tr><tr><td colspan="13">LoCoMo</td></tr><tr><td>PREMem</td><td>68.0</td><td>71.0</td><td>50.0</td><td>54.6</td><td>64.9</td><td>67.7</td><td>29.4</td><td>27.0</td><td>30.1</td><td>30.6</td><td>34.5</td><td>35.9</td></tr><tr><td>w/o Step 2</td><td>64.4(-5.3%)</td><td>68.2(-3.8%)</td><td>47.3(-5.4%)</td><td>52.8(-3.2%)</td><td>61.4(-5.4%)</td><td>64.7(-4.5%)</td><td>29.6(+0.6%)</td><td>28.6(+5.6%)</td><td>28.3(-6.0%)</td><td>28.5(-7.0%)</td><td>33.2(-3.6%)</td><td>34.2(-4.8%)</td></tr><tr><td>w/o Step 1</td><td>44.5(-34.6%)</td><td>47.8(-32.7%)</td><td>32.3(-35.4%)</td><td>33.9(-37.8%)</td><td>41.9(-35.4%)</td><td>44.1(-34.9%)</td><td>14.7(-49.9%)</td><td>14.6(-45.9%)</td><td>13.1(-56.4%)</td><td>13.2(-56.7%)</td><td>16.5(-52.1%)</td><td>17.9(-50.1%)</td></tr><tr><td>w/o Step 1 Categories</td><td>65.7(-3.4%)</td><td>68.1(-4.1%)</td><td>49.1(-1.8%)</td><td>52.4(-4.0%)</td><td>60.8(-6.2%)</td><td>63.5(-6.3%)</td><td>27.9(-5.3%)</td><td>26.3(-2.9%)</td><td>27.5(-8.7%)</td><td>27.9(-8.9%)</td><td>32.0(-7.2%)</td><td>33.9(-5.6%)</td></tr><tr><td>w/o Temporal Reasoning</td><td>64.2(-5.7%)</td><td>65.8(-7.3%)</td><td>47.8(-4.3%)</td><td>52.8(-3.2%)</td><td>60.9(-6.1%)</td><td>62.6(-7.6%)</td><td>27.4(-6.9%)</td><td>26.4(-2.5%)</td><td>25.1(-16.4%)</td><td>26.8(-12.6%)</td><td>31.1(-9.9%)</td><td>32.7(-9.1%)</td></tr></table>

# 5 Practical Applications

Memory systems are foundational for personalized conversational agents, with resource efficiency critical for real-world deployment. To demonstrate the value of PREMem under resource constraints, we evaluate three key dimensions: (1) storage efficiency through alternative retrieval methods (Section 5.1), (2) computational cost reduction using smaller reasoning models (Section 5.2), and (3) token budget for context efficiency (Section 5.3).

# 5.1 BM25 as Embedding Alternative

![](images/f70b24013c718068f8ba400240fb6bcd43e663582e160f3d9dd6ccc88f0b908a.jpg)  
Figure 2: Performance comparison (LLM-as-a-judge score) across retrieval mechanisms (left: BM25 vs. embedding) and memory reasoning models (right: lowspec vs. high-spec).

Vector embeddings for semantic search demand substantial storage for personalized assistants that must maintain separate indexes for each user. While keyword-based retrieval methods like BM25 typically underperform semantic search methods (Thakur et al., 2021), experimental results shown in Figure 2 (left) demonstrate BM25 remain surprisingly competitive with PREMem. This provides an efficient option for resource-constrained deployments with minimal performance tradeoffs.

# 5.2 Low-Spec Reasoning Models

Memory construction typically requires powerful LLMs. To explore more efficient alternatives, we investigate whether smaller models can effectively perform our pre-storage reasoning. We introduce PREMemS, which uses smaller variants from each model family (Qwen2.5-14B, Gemma3-12B, or gpt-4.1-nano) for memory construction.

Figure 2 (right) shows that reasoning-focused prompts in LLMextract and LLMreason help smaller models create high-quality memory representations. This approach effectively provides an

alternative for real-world applications by reducing computational costs during memory construction.

# 5.3 Token Budget Efficiency

Allocating thousands of tokens from limited context windows solely for memory retrieval represents a substantial opportunity cost for multipurpose assistants. We evaluate PREMem performance across varying token budgets.

Table 5: Performance across token budgets. Bold indicates the highest score in each range.   

<table><tr><td rowspan="3">Method</td><td rowspan="3">Token budget</td><td colspan="6">Model</td></tr><tr><td colspan="2">Qwen2.5</td><td colspan="2">gemma-3</td><td colspan="2">gpt-4.1</td></tr><tr><td>14B</td><td>72B</td><td>12B</td><td>27B</td><td>mini</td><td>base</td></tr><tr><td colspan="8">LongMemEval</td></tr><tr><td rowspan="3">SeCom</td><td>1024</td><td>35.8</td><td>37.2</td><td>33.4</td><td>33.0</td><td>37.0</td><td>35.5</td></tr><tr><td>2048</td><td>37.6</td><td>39.4</td><td>37.4</td><td>38.9</td><td>42.3</td><td>42.0</td></tr><tr><td>4096</td><td>42.8</td><td>44.4</td><td>38.8</td><td>40.4</td><td>44.3</td><td>44.4</td></tr><tr><td rowspan="3">A-Mem</td><td>1024</td><td>44.4</td><td>48.6</td><td>36.4</td><td>41.0</td><td>49.2</td><td>49.4</td></tr><tr><td>2048</td><td>50.3</td><td>53.6</td><td>39.0</td><td>45.3</td><td>53.9</td><td>55.9</td></tr><tr><td>4096</td><td>54.8</td><td>58.5</td><td>44.9</td><td>50.6</td><td>61.3</td><td>62.0</td></tr><tr><td rowspan="3">HippoRAG-2</td><td>1024</td><td>41.5</td><td>41.5</td><td>38.5</td><td>38.3</td><td>40.8</td><td>40.2</td></tr><tr><td>2048</td><td>44.7</td><td>45.9</td><td>43.9</td><td>43.1</td><td>44.8</td><td>45.2</td></tr><tr><td>4096</td><td>57.5</td><td>57.4</td><td>51.3</td><td>53.5</td><td>61.0</td><td>61.7</td></tr><tr><td rowspan="3">PREMem</td><td>1024</td><td>66.4</td><td>67.6</td><td>58.9</td><td>63.0</td><td>68.7</td><td>70.2</td></tr><tr><td>2048</td><td>64.7</td><td>67.5</td><td>57.7</td><td>61.9</td><td>67.6</td><td>71.4</td></tr><tr><td>4096</td><td>62.2</td><td>66.9</td><td>55.7</td><td>60.5</td><td>67.2</td><td>71.8</td></tr><tr><td colspan="8">LoCoMo</td></tr><tr><td rowspan="3">SeCom</td><td>1024</td><td>57.0</td><td>60.5</td><td>42.3</td><td>47.9</td><td>51.5</td><td>54.2</td></tr><tr><td>2048</td><td>60.4</td><td>58.5</td><td>44.8</td><td>49.1</td><td>53.4</td><td>56.7</td></tr><tr><td>4096</td><td>63.4</td><td>63.9</td><td>46.0</td><td>50.1</td><td>54.6</td><td>57.3</td></tr><tr><td rowspan="3">A-Mem</td><td>1024</td><td>42.9</td><td>45.9</td><td>43.8</td><td>44.5</td><td>52.9</td><td>51.5</td></tr><tr><td>2048</td><td>43.6</td><td>45.6</td><td>43.9</td><td>44.5</td><td>52.7</td><td>49.5</td></tr><tr><td>4096</td><td>43.5</td><td>45.1</td><td>43.8</td><td>44.3</td><td>53.3</td><td>50.2</td></tr><tr><td rowspan="3">HippoRAG-2</td><td>1024</td><td>56.0</td><td>55.7</td><td>40.8</td><td>46.5</td><td>49.7</td><td>53.1</td></tr><tr><td>2048</td><td>61.7</td><td>61.6</td><td>44.3</td><td>49.5</td><td>54.6</td><td>57.3</td></tr><tr><td>4096</td><td>64.1</td><td>65.3</td><td>47.0</td><td>51.0</td><td>56.2</td><td>59.5</td></tr><tr><td rowspan="3">PREMem</td><td>1024</td><td>63.7</td><td>65.6</td><td>48.1</td><td>52.7</td><td>64.6</td><td>67.3</td></tr><tr><td>2048</td><td>68.0</td><td>71.0</td><td>50.0</td><td>54.6</td><td>64.9</td><td>67.7</td></tr><tr><td>4096</td><td>67.0</td><td>68.7</td><td>47.2</td><td>53.3</td><td>64.8</td><td>67.1</td></tr></table>

While other methods degrade significantly with reduced context lengths, PREMem maintains robust performance even with minimal token, as shown in Table 5. This stability stems from memory fragments that capture pre-reasoned cognitive relationships rather than simply storing raw conversation turns or graph connections. The efficiency allows developers to allocate smaller portions of context windows to memory while preserving personalization quality in real-world applications.

# 6 Conclusion

We present PREMem, a novel episodic memory system that shifts complex reasoning processes from response generation to the memory construction phase. This method transforms conversations

into structured memories with categorized information types and cross-session reasoning patterns. Our approach significantly improves performance on LongMemEval and LoCoMo benchmarks, with particularly strong results for temporal reasoning and multi-session tasks. Notably, even modest-sized models using PREMem achieve competitive results compared to larger state-of-the-art systems. Additionally, our focus on token budget, retrieval efficiency, and streamlined memory construction makes PREMem effective for real-world conversational AI systems that require long-term personalization under resource constraints.

# Limitations

Our work has several limitations that present opportunities for future research:

Reduced efficiency in single-hop reasoning Our pre-reasoning structure shows lower performance for single-hop questions compared to direct retrieval methods. This could be due to additional processing that may not benefit straightforward queries. To address this, future work could consider utilizing original messages directly for single-hop reasoning tasks.

Lack of original conversation context Our implementation focuses on extracted and synthesized memory items rather than original conversation messages to reduce storage requirements. This approach sacrifices access to certain linguistic nuances, including users’ conversational styles and terminology preferences. A potential solution might involve query-dependent hybrid retrieval that combines structured memories with original conversation segments based on the nature of the user’s question.

Absence of memory decay mechanisms Our approach does not incorporate forgetting mechanisms found in human memory. While our similarity threshold helps filter retrieved items, managing truly long-term conversations would require additional constraints. For extended conversation histories, implementing explicit memory size limitations or importance-based decay functions would help control the persistent memory pool.

Limited theoretical contribution Our approach demonstrates practical improvements by applying cognitive science concepts to conversational systems. However, it remains primarily an empirical

contribution rather than advancing new theoretical insights about memory or cognition. Future work could explore deeper theoretical implications for human-AI interaction.

# Ethical Considerations

Research on episodic memory systems for conversational AI merits thoughtful consideration of privacy aspects, as these systems retain and process user information across multiple sessions. PRE-Mem’s structured approach to memory representation offers opportunities for enhanced transparency, potentially enabling clearer user controls over what information is stored. In real-world applications, implementing appropriate data management options would allow users to understand and guide their personalized experience.

The cross-session reasoning capabilities in our approach warrant attention to potential biases and inference accuracy. Our categorization helps distinguish between what users explicitly stated and what the system infers, but misinterpretations can still occur. Future research should develop confidence indicators for memory-based responses and create mechanisms for users to correct the system when it makes inappropriate connections between separate conversations, helping prevent potential misunderstandings from persisting across interactions.

# References

John R Anderson. 2013. The architecture of cognition. Psychology Press.   
Sanghwan Bae, Donghyun Kwak, Soyoung Kang, Min Young Lee, Sungdong Kim, Yuin Jeong, Hyeri Kim, Sang-Woo Lee, Woomyoung Park, and Nako Sung. 2022. Keep me updated! memory management in long-term conversations. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 3769–3787, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.   
Frederic Charles Bartlett. 1995. Remembering: A study in experimental and social psychology. Cambridge university press.   
John D Bransford and Marcia K Johnson. 1972. Contextual prerequisites for understanding: Some investigations of comprehension and recall. Journal of verbal learning and verbal behavior, 11(6):717–726.   
Susan Carey. 1985. Conceptual change in childhood. (No Title).

Nuo Chen, Hongguang Li, Jianhui Chang, Juhua Huang, Baoyuan Wang, and Jia Li. 2025. Compress to impress: Unleashing the potential of compressive memory in real-world long-term conversations. In Proceedings of the 31st International Conference on Computational Linguistics, pages 755–773, Abu Dhabi, UAE. Association for Computational Linguistics.   
Michelene TH Chi. 2009. Three types of conceptual change: Belief revision, mental model transformation, and categorical shift. In International handbook of research on conceptual change, pages 89–110. Routledge.   
Michelene TH Chi and 1 others. 1981. Expertise in problem solving.   
Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Haotian Wang, Ming Liu, and Bing Qin. 2024. TimeBench: A comprehensive evaluation of temporal reasoning abilities in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1204–1228, Bangkok, Thailand. Association for Computational Linguistics.   
Yiming Du, Wenyu Huang, Danna Zheng, Zhaowei Wang, Sebastien Montella, Mirella Lapata, Kam-Fai Wong, and Jeff Z. Pan. 2025. Rethinking memory in ai: Taxonomy, operations, topics, and future directions. Preprint, arXiv:2505.00675.   
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. 2025. From local to global: A graph rag approach to query-focused summarization. Preprint, arXiv:2404.16130.   
Gilles Fauconnier and Mark Turner. 2008. The way we think: Conceptual blending and the mind’s hidden complexities. Basic books.   
Zafeirios Fountas, Martin Benfeghoul, Adnan Oomerjee, Fenia Christopoulou, Gerasimos Lampouras, Haitham Bou Ammar, and Jun Wang. 2025. Humaninspired episodic memory for infinite context LLMs. In The Thirteenth International Conference on Learning Representations.   
Yubin Ge, Salvatore Romeo, Jason Cai, Raphael Shu, Monica Sunkara, Yassine Benajiba, and Yi Zhang. 2025. Tremu: Towards neuro-symbolic temporal reasoning for llm-agents with memory in multi-session dialogues. Preprint, arXiv:2502.01630.   
Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2025. Lightrag: Simple and fast retrievalaugmented generation. Preprint, arXiv:2410.05779.   
Bernal Jiménez Gutiérrez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. 2024. Hipporag: Neurobiologically inspired long-term memory for large language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. 2025. From rag to memory: Non-parametric continual learning for large language models. Preprint, arXiv:2502.14802.   
Yuki Hou, Haruki Tamoto, and Homei Miyashita. 2024. "my agent understands me better": Integrating dynamic human-like memory recall and consolidation in llm-based agents. In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, CHI EA ’24, New York, NY, USA. Association for Computing Machinery.   
Alexis Huet, Zied Ben Houidi, and Dario Rossi. 2025. Episodic memories generation and evaluation benchmark for large language models. In The Thirteenth International Conference on Learning Representations.   
Frank C Keil. 1979. Semantic and conceptual development: An ontological perspective. Harvard University Press.   
John E. Laird. 2012. The Soar Cognitive Architecture. The MIT Press.   
Kuang-Huei Lee, Xinyun Chen, Hiroki Furuta, John Canny, and Ian Fischer. 2024. A human-inspired reading agent with gist memory of very long contexts. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 26396–26415. PMLR.   
Hao Li, Chenghao Yang, An Zhang, Yang Deng, Xiang Wang, and Tat-Seng Chua. 2025. Hello again! LLMpowered personalized agent for long-term dialogue. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5259–5276, Albuquerque, New Mexico. Association for Computational Linguistics.   
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851– 13870, Bangkok, Thailand. Association for Computational Linguistics.   
Jean Matter Mandler. 2014. Stories, scripts, and scenes: Aspects of schema theory. Psychology Press.   
Kelong Mao, Zhicheng Dou, and Hongjin Qian. 2022. Curriculum contrastive context denoising for fewshot conversational dense retrieval. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’22, page 176–186, New York, NY, USA. Association for Computing Machinery.   
Pedro Henrique Martins, Zita Marinho, and Andre Martins. 2022. $\infty$ -former: Infinite memory transformer.

In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5468–5485, Dublin, Ireland. Association for Computational Linguistics.   
Rusen Meylani. 2024. Innovations with schema theory: Modern implications for learning, memory, and academic achievement. International Journal for Multidisciplinary Research, 6(1):2582–2160.   
Gregory Murphy. 2004. The big book of concepts. MIT press.   
Kai Tzu-iunn Ong, Namyoung Kim, Minju Gwak, Hyungjoo Chae, Taeyoon Kwon, Yohan Jo, Seungwon Hwang, Dongha Lee, and Jinyoung Yeo. 2025. Towards lifelong dialogue agents via timeline-based memory management. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8631–8661, Albuquerque, New Mexico. Association for Computational Linguistics.   
OpenAI. 2025. GPT-4.1. https://openai.com/ index/gpt-4-1/. Accessed: 2025-05-17.   
Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Xufang Luo, Hao Cheng, Dongsheng Li, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, and Jianfeng Gao. 2025. Secom: On memory construction and retrieval for personalized conversational agents. In The Thirteenth International Conference on Learning Representations.   
Jean Piaget, Margaret Cook, and 1 others. 1952. The origins of intelligence in children, volume 8. International universities press New York.   
Yifu Qiu, Zheng Zhao, Yftah Ziser, Anna Korhonen, Edoardo Ponti, and Shay Cohen. 2024. Are large language model temporally grounded? In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7064–7083, Mexico City, Mexico. Association for Computational Linguistics.   
David E Rumelhart. 2017. Schemata: The building blocks of cognition. In Theoretical issues in reading comprehension, pages 33–58. Routledge.   
David E Rumelhart, Donald A Norman, and 1 others. 1976. Accretion, tuning and restructuring: Three modes of learning. Citeseer.   
Daniel L Schacter and Donna Rose Addis. 2007. On the constructive episodic simulation of past and future events. Behavioral and Brain Sciences, 30(3):331– 332.   
Daniel L Schacter and Endel Tulving. 1994. Memory systems 1994. Memory Systems, 199.   
Roger C Schank and Robert P Abelson. 2013. Scripts, plans, goals, and understanding: An inquiry into human knowledge structures. Psychology press.

Lianlei Shan, Shixian Luo, Zezhou Zhu, Yu Yuan, and Yong Wu. 2025. Cognitive memory in large language models. Preprint, arXiv:2504.02441.   
L Squire. 1987. Memory and brain oxford university press: New york.   
Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and 197 others. 2025. Gemma 3 technical report. Preprint, arXiv:2503.19786.   
Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).   
Lei Wang, Jingsen Zhang, Hao Yang, Zhiyuan Chen, Jiakai Tang, Zeyu Zhang, Xu Chen, Yankai Lin, Ruihua Song, Wayne Xin Zhao, Jun Xu, Zhicheng Dou, Jun Wang, and Ji-Rong Wen. 2024a. User behavior simulation with large language model based agents. Preprint, arXiv:2306.02552.   
Qingyue Wang, Yanhe Fu, Yanan Cao, Shuai Wang, Zhiliang Tian, and Liang Ding. 2025. Recursively summarizing enables long-term dialogue memory in large language models. Neurocomputing, 639:130193.   
Yu Wang, Chi Han, Tongtong Wu, Xiaoxin He, Wangchunshu Zhou, Nafis Sadeq, Xiusi Chen, Zexue He, Wei Wang, Gholamreza Haffari, and 1 others. 2024b. Towards lifespan cognitive systems. arXiv preprint arXiv:2409.13265.   
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025a. Longmemeval: Benchmarking chat assistants on long-term interactive memory. In The Thirteenth International Conference on Learning Representations.   
Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, and Yong Liu. 2025b. From human memory to ai memory: A survey on memory mechanisms in the era of llms. Preprint, arXiv:2504.15965.   
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110.   
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 22 others. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Ruifeng Yuan, Shichao Sun, Yongqi Li, Zili Wang, Ziqiang Cao, and Wenjie Li. 2025. Personalized large language model assistant with evolving conditional memory. In Proceedings of the 31st International Conference on Computational Linguistics, pages 3764–3777, Abu Dhabi, UAE. Association for Computational Linguistics.

Dun Zhang, Jiacheng Li, Ziyang Zeng, and Fulong Wang. 2025. Jasper and stella: distillation of sota embedding models. Preprint, arXiv:2412.19048.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. Memorybank: Enhancing large language models with long-term memory. Proceedings of the AAAI Conference on Artificial Intelligence, 38(17):19724–19731.

Xiangrong Zhu, Yuexiang Xie, Yi Liu, Yaliang Li, and Wei Hu. 2025. Knowledge graph-guided retrieval augmented generation. Preprint, arXiv:2502.06864.

# A LLM Prompt

We include all prompts required to run and evaluate PREMem. In these figures, placeholders (denoted by {{$variable}}) indicate positions where specific content is dynamically inserted during execution. For Step 1, refer to Figure 3, for Step 2, refer to Figure 4. The response generation prompts for Long-MemEval and LoCoMo are provided in Figure 6 and Figure 5, respectively. The LLM-as-a-judge evaluation prompt is shown in Figure 7.

# B Cognitive Scientific Foundation for Memory Evolution Patterns

In cognitive science, schema is a framework (structure) that organizes an individual’s experiences, knowledge, and information, as well as the way they are stored in his memory. The schema stores one’s experiences, knowledge and information as its memory, and it is developed by assimilating and accommodating the information (Piaget et al., 1952). When new information aligns with an existing schema, the schema assimilation occurs. Conversely, misaligned information requires schema updates to incorporate new data.

A seminal work in schema theory (Rumelhart et al., 1976) introduced three modes of learning: accretion, tuning, and restructuring. This work has become the foundation of understanding how existing knowledge structures—known as schemata—are transformed whenever new information is encountered. In particular, accretion means adding new information to an existing schema without altering its structure. Tuning refines the existing schema, making it more efficient or accurate. Restructuring,

on the other hand, involves a more fundamental change in the schema’s structure. Thus, this work has become the foundation for further investigations on how schemata are modified and reorganized in response to new informaion.

Referring to further schema theory literature (Chi et al., 1981; Bartlett, 1995; Mandler, 2014; Rumelhart, 2017), we identify six major mechanisms of how a schema modified: (1) Schema expansion refers to adding a new attribute or feature to an existing schema; (2) Schema integration occurs when separate, related schemata become connected to form a more cohesive structure; (3) Schema refinement points to the process of a schema being refined or made more specific based on accumulated details; (4) Schema reinforcement happens when similar information is repeatedly acquired, strengthening the existing schema; (5) Schema restructuring completely reorganizes schema structure; and (6) Schema creation occurs when existing schema structure does not align with a new information, leading to the creation of an entirely new schema.

During a conversation, the individual acquires additional information, and integrates new information into established memory. When integrating, it is crucial to consider how the new information is related to prior memory. Referring to cognitive science (Anderson, 2013; Bransford and Johnson, 1972), which studies how people perceive and learn from information, and conceptual development (Carey, 1985; Murphy, 2004; Chi, 2009), which studies how infants learn concepts, we identify five information types– extension, accumulation, specification, transformation, and connection– each of which causes a different type of modification in the underlying schema.

Extension (Elaboration) A new information broadens the scope of existing knowledge. (Anderson, 2013; Carey, 1985) describe that exposure to information and experience extend the existing knowledge structure, paralleling the process of schema expansion.

Accumulation The similar type of information accumulates. Repeated exposure to similar information solidifies an existing framework. (Chi et al., 1981; Schank and Abelson, 2013) demonstrate that repeated encounters with similar information and experience solidify a schema.

# GOAL

Analyze the entire provided ⟨Conversation⟩ to identify all statements revealing personal information about the user. Categorize each piece of information as Factual, Experiential, or Subjective, and output the results as a single structured JSON object according to the ⟨Final Output JSON Format⟩.

# INFORMATION CATEGORY DEFINITIONS

1. Factual Information: Objective, verifiable facts about the user’s state, attributes, possessions, knowledge, skills, and relationships with others (family, friends, pets, etc.). (’What I am / What I have / Who I know’)   
* Keywords: is, am, have, own, live in, work as, know (skill/fact/person), my name/age/job/sister/friend is, etc.   
* Examples: "My name is Alex.", "I live in New York.", "I have a Bachelor’s degree in CS.", "I own two bikes.", "Emily is my sister.", "Luna is my cat.", "I know Python."

2. Experiential Information: Specific events, actions, activities, or interactions experienced by the user over time, often situated in a context. (’What I did / What happened to me’)

* Keywords: went, did, saw, met, visited, learned (an experience), attended, bought (as an event), have been, have visited, have tried, have experienced, last year, yesterday, when I was..., etc.   
* Examples: "I travelled to LA last weekend.", "I’ve assembled the IKEA bookshelf.", "I’ve been to Japan twice.", "I have met with the CEO.", "I attended the Imagine Dragons concert."

3. Subjective Information: The user’s internal states, including preferences, habits, opinions, beliefs, goals, plans, feelings, etc. (’What I like / think / want / feel / usually do’)

* Keywords: like, love, hate, prefer, think, believe, feel, want, plan to, hope to, usually, often, my goal is, etc.   
* Examples: "I love spicy food.", "I usually wake up at 7 AM.", "I thought that movie was great.", "My goal is to learn Spanish.", "I want to visit Europe next year."

# INSTRUCTIONS

1. Carefully read and analyze the entire ⟨Conversation⟩. ⟨Conversation⟩ consists of messages, each containing a [message_id] followed by its content.   
2. Identify all specific pieces of information about the user that fall into the Factual, Experiential, or Subjective categories based on the definitions above.   
3. Format each value as a phrase that starts with a verb in present tense, regardless of the original tense in the conversation.   
4. For the "date" field:   
* For ongoing facts or current states, use the date of the message   
* For past events with a specific timeframe mentioned (e.g., "yesterday", "three days ago"), calculate and use the actual date based on the message date   
* For past events mentioned in the conversation, mark as "Before [message-date]"   
* For future plans or intentions, mark as "After [message-date]"   
5. Format the output as a single JSON object with three categories: "Factual_Information", "Experiential_Information", and "Subjective_Information".   
Use empty lists ([]) for categories with no information.   
6. Use the exact same [message_id] as in the original message. Do not include pronouns in the value.

# Example

⟨Conversation⟩

[msg-301] (2024-05-17 Friday) I’m living in Rome now with my girlfriend, Hana. We moved here last summer because she started grad school at Stanford.

[msg-302] (2024-05-17 Friday) I quit my job at Coupang in March. I just didn’t see myself growing there anymore.

[msg-303] (2024-05-17 Friday) I’m thinking about switching into UX design. I’ve always liked the idea of making tech more human-friendly.

[msg-304] (2024-05-17 Friday) My brother Junho lives in Seattle. He’s an engineer and always sends me photos of the mountains.

[msg-305] (2024-05-17 Friday) I ate chicken with my friends yesterday.

Answer:

```jsonl
{ "Factual\_Information": [ { "key": "current residence", "value": "Lives in Rome with girlfriend Hana", "message_id": "msg-301", "date": "2024-05-17" },... ], "Experiential\_Information": [ { "key": "job resignation", "value": "Quit job at Coupang in March", "message_id": "msg-302", "date": "Before 2024-05-17" },... ], "Subjective_Intermation": [ { "key": "career dissatisfaction", "value": "Be Felt no growth potential at Coupang", "message\_id": "msg-302", "date": "Before 2024-05-17" },... ] }   
{Conversation} {{\$conversation}} 
```

Figure 3: Step 1: Personal information extraction and categorization prompt.

You are an AI assistant analyzing memory fragments to generate insights. Your task is to identify patterns and connections from the data provided.

Analyze these fragments and generate insights based on five inference types:

- Extension/Generalization: The process of expanding information from specific cases or situations to broader categories, domains, or patterns. This type of inference derives more general characteristics or tendencies from concrete information.   
- Accumulation: The process of identifying behaviors, experiences, or patterns that repeat or persist over time. This type of inference focuses on frequency, consistency, and persistence to infer habitual patterns or significant trends.   
- Specification/Refinement: The process of breaking down general information into more detailed and specific aspects. This type of inference decomposes broad concepts or experiences into concrete elements or details.   
- Transformation: The process of identifying changes in states, perspectives, emotions, behaviors, etc. over time. This type of inference discerns transitions or developments between previous and current/new states.   
- Connection/Implication: The process of identifying relationships, causality, or meaning between seemingly disparate pieces of information. This type of inference discerns connections or conclusions not explicitly mentioned.

Your output should be formatted as a JSON object with an "extended_insight" key containing an array of inference objects. Each inference object should have the following structure:

```json
{
    "inference_type": "one of the five inference types",
    "key": "brief description of the insight",
    "date": "relevant date or date range",
    "value": "detailed description of the insight (12 words or less)"
} 
```

Important instructions:

- You do NOT need to use all five inference types. Select only the inference types that clearly apply to the data.   
- Include multiple different inference types when appropriate, but don’t force all five types.   
- You may use the same inference type multiple times for different insights if appropriate.   
- Focus on quality over quantity - provide meaningful insights based on the data.   
- Avoid trivial or insignificant insights - focus only on substantive patterns and connections.

⟨example⟩

Example:

Below are the memory fragments to analyze:

[tech purchase, 2023-03-05]: Jordan buy new drawing tablet

[software usage, 2023-03-07]: Jordan spend three hours learning Procreate app

[online activity, 2023-03-15]: Jordan create account on digital art community DeviantArt

[social media, 2023-03-22]: Jordan share first digital artwork on Instagram

Output:

```jsonl
"extended_insight": [  
{  
    "inference_type": "extension/generalization",  
    "key": "skill development approach",  
    "date": "2023-03-05 to 2023-03-22",  
    "value": "Jordan follows a methodical learning approach with appropriate tools"  
},  
{  
    "inference_type": "accumulation",  
    "key": "digital art activities",  
    "date": "2023-03-05 to 2023-03-22",  
    "value": "Jordan engaged in 4 digital art activities within 17 days"  
},  
{  
    "inference_type": "specification/refinement",  
    "key": "artistic medium",  
    "date": "2023-03-22",  
    "value": "Jordan uses tablet-based digital illustration with Procreate"  
},  
{  
    "inference_type": "transformation",  
    "key": "identity shift",  
    "date": "Before 2023-03-05 to 2023-03-22",  
    "value": "Jordan evolved from art appreciator to digital artist"  
},  
{  
    "inference_type": "connection/implication",  
    "key": "artistic background",  
    "date": "Before 2023-03-05",  
    "value": "Jordan likely has previous art experience"  
}  
} 
```

Below are the memory fragments to analyze:

{{$memory_fragments}}

Figure 4: Step 2: Memory pattern analysis and inference prompt.

Based on the context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Context: {{$context}}

Question: {{$question}}

Short Answer:

Figure 5: LoCoMo answer generation prompt.

Specification The existing information becomes more detailed and developed more precisely. New information refines existing knowledge by adding more detailed or precise features, causing schema refinement. (Murphy, 2004; Keil, 1979) both claim that knowledge is refined and differentiated as precise and specific information is encountered.

Transformation The previous information is replaced by new information or fundamentally modified. New information drives schema restructuring. According to (Chi, 2009; Rumelhart et al., 1976), schema is reconstructed when the new information does not fit to the existing knowledge significantly.

Connection The relationship between the information and the causality are revealed. The connected information promotes existing schemata to be integrated. (Bransford and Johnson, 1972; Fauconnier and Turner, 2008) show that connection between information develops individual’s reasoning and understanding.

These five types of new information—extension, accumulation, specification, transformation, and connection—are consistent with the schema modification mechanisms: schema expansion, reinforcement, refinement, restructuring, and integration.

# C Dataset Description and Category Unification

For consistency, we unify the question types into five categories: single-hop, multi-hop, temporal reasoning, adversarial, and knowledge update (LongMemEval only). For LoCoMo, we treat all questions originally labeled as open-domain-knowledge as single-hop. The other labels—multi-hop, temporal reasoning, and adversarial—are retained as-is. For Long-MemEval, we apply these mappings: Any type containing the word single is mapped to single-hop.

You are an intelligent assistant designed to provide concise, accurate answers based on given context. Your task is to analyze the provided information and respond to a specific question with a few words or a short phrase.

Here is the context you should use to inform your answer:

{{$context}}

Now, please consider the following question:

{{$question}}

# Instructions:

1. Carefully read and analyze the provided context.   
2. Consider the question in relation to the context.   
3. Formulate a concise answer based solely on the information given in the context.   
4. Respond with a short phrase only. Do not use a full sentence.   
5. Do not include any explanations, reasoning, or greetings in your response.   
6. Ensure your answer is directly relevant to the question asked.

Your response should provide only the essential information in a brief phrase.

Answer:

Figure 6: LongMemEval answer generation prompt.

All other types are converted by replacing session with hop, aligning them with the multi-hop or temporal reasoning categories. If the question ID ends with _abs, it is classified as adversarial based on its original designation as an abstention question. Questions related to knowledge revision are assigned to the knowledge update category.

This unified labeling scheme supports direct comparison across datasets and is used for all category-level evaluations in this work. Datasets are available under CC-BY-NC-4.0 (LoCoMo) and MIT License (LongMemEval).

# D Complementary Results

We present the complete scores including metrics that were omitted from the main paper in Table 6.

Table 6: Complete experimental results.   

<table><tr><td rowspan="2">Inference LLM</td><td rowspan="2">Model</td><td colspan="9">LongMemEval</td><td colspan="6">LoCoMo</td></tr><tr><td>LLM</td><td>ROUGE-1</td><td>ROUGE-L</td><td>BLEU-1</td><td>METEOR</td><td>BERTScore</td><td>token length</td><td>LLM</td><td>ROUGE-1</td><td>ROUGE-L</td><td>BLEU-1</td><td>METEOR</td><td>BERTScore</td><td>token length</td><td></td></tr><tr><td rowspan="11">3B</td><td>Zero</td><td>15.93</td><td>7.77</td><td>7.20</td><td>5.16</td><td>4.30</td><td>85.44</td><td>0.00</td><td>22.24</td><td>7.12</td><td>6.10</td><td>6.12</td><td>4.85</td><td>83.99</td><td>0.00</td><td></td></tr><tr><td>Full</td><td>32.55</td><td>19.25</td><td>18.97</td><td>11.60</td><td>89.34</td><td>18710.50</td><td>42.56</td><td>20.33</td><td>19.50</td><td>15.87</td><td>16.37</td><td>86.21</td><td>19643.80</td><td></td><td></td></tr><tr><td>Turn</td><td>38.43</td><td>23.71</td><td>23.23</td><td>20.24</td><td>13.80</td><td>89.58</td><td>1854.30</td><td>44.90</td><td>22.06</td><td>21.17</td><td>16.31</td><td>16.57</td><td>86.49</td><td>2009.30</td><td></td></tr><tr><td>Session</td><td>31.02</td><td>19.38</td><td>18.88</td><td>16.10</td><td>10.76</td><td>88.42</td><td>1919.80</td><td>41.40</td><td>20.77</td><td>19.96</td><td>15.75</td><td>15.44</td><td>86.15</td><td>1989.30</td><td></td></tr><tr><td>Segment</td><td>36.59</td><td>21.86</td><td>21.32</td><td>19.53</td><td>12.90</td><td>89.35</td><td>1770.79</td><td>46.28</td><td>25.29</td><td>24.22</td><td>18.94</td><td>19.44</td><td>86.43</td><td>1881.45</td><td></td></tr><tr><td>SeCom</td><td>34.52</td><td>23.15</td><td>22.70</td><td>19.97</td><td>13.51</td><td>89.06</td><td>1775.10</td><td>42.93</td><td>23.85</td><td>22.85</td><td>17.75</td><td>17.60</td><td>86.23</td><td>1884.30</td><td></td></tr><tr><td>HippoRAG-2</td><td>40.45</td><td>27.31</td><td>26.72</td><td>24.26</td><td>16.27</td><td>89.46</td><td>3811.61</td><td>46.74</td><td>26.82</td><td>25.93</td><td>21.68</td><td>19.99</td><td>87.26</td><td>3811.61</td><td></td></tr><tr><td>A-Mem</td><td>44.42</td><td>31.15</td><td>30.27</td><td>27.64</td><td>20.31</td><td>90.06</td><td>6199.24</td><td>42.07</td><td>29.02</td><td>28.44</td><td>23.70</td><td>21.85</td><td>88.31</td><td>6199.24</td><td></td></tr><tr><td>PREMem</td><td>50.80</td><td>33.81</td><td>33.45</td><td>30.49</td><td>21.76</td><td>90.84</td><td>2032.40</td><td>53.79</td><td>24.64</td><td>23.16</td><td>15.86</td><td>16.46</td><td>86.07</td><td>2033.90</td><td></td></tr><tr><td>with bm25</td><td>48.25</td><td>33.26</td><td>32.82</td><td>29.78</td><td>22.00</td><td>90.64</td><td>2032.20</td><td>51.20</td><td>22.69</td><td>21.18</td><td>14.13</td><td>14.71</td><td>85.85</td><td>2034.50</td><td></td></tr><tr><td>PREMem_small</td><td>46.13</td><td>29.14</td><td>28.51</td><td>25.92</td><td>18.83</td><td>90.20</td><td>2032.30</td><td>51.26</td><td>23.15</td><td>21.55</td><td>14.49</td><td>15.01</td><td>85.96</td><td>2034.00</td><td></td></tr><tr><td rowspan="11">Qwen2.5</td><td>Zero</td><td>16.08</td><td>10.10</td><td>9.90</td><td>7.49</td><td>5.55</td><td>85.05</td><td>0.00</td><td>25.19</td><td>7.12</td><td>5.45</td><td>6.10</td><td>6.22</td><td>82.84</td><td>0.00</td><td></td></tr><tr><td>Full</td><td>37.43</td><td>25.37</td><td>24.73</td><td>19.92</td><td>15.84</td><td>88.13</td><td>18710.50</td><td>60.57</td><td>24.08</td><td>22.81</td><td>18.16</td><td>21.85</td><td>86.62</td><td>19643.80</td><td></td></tr><tr><td>Turn</td><td>39.70</td><td>25.27</td><td>24.51</td><td>20.48</td><td>15.68</td><td>87.88</td><td>1854.30</td><td>61.63</td><td>28.90</td><td>27.49</td><td>22.87</td><td>22.34</td><td>87.77</td><td>2009.30</td><td></td></tr><tr><td>Session</td><td>29.04</td><td>19.27</td><td>18.79</td><td>14.82</td><td>11.41</td><td>86.71</td><td>1919.80</td><td>54.46</td><td>25.68</td><td>24.34</td><td>20.27</td><td>20.49</td><td>86.48</td><td>1989.30</td><td></td></tr><tr><td>Segment</td><td>39.02</td><td>24.62</td><td>24.02</td><td>20.15</td><td>14.92</td><td>87.80</td><td>1770.79</td><td>63.54</td><td>33.22</td><td>31.90</td><td>26.12</td><td>26.12</td><td>87.66</td><td>1881.45</td><td></td></tr><tr><td>SeCom</td><td>37.63</td><td>24.51</td><td>23.90</td><td>19.33</td><td>15.56</td><td>87.87</td><td>1775.10</td><td>60.47</td><td>31.04</td><td>29.64</td><td>24.59</td><td>23.95</td><td>87.32</td><td>1884.30</td><td></td></tr><tr><td>HippoRAG-2</td><td>44.49</td><td>29.44</td><td>28.44</td><td>23.33</td><td>15.56</td><td>88.84</td><td>3811.61</td><td>61.91</td><td>29.72</td><td>28.30</td><td>23.40</td><td>24.59</td><td>87.38</td><td>3811.61</td><td></td></tr><tr><td>A-Mem</td><td>50.32</td><td>32.99</td><td>31.92</td><td>27.17</td><td>23.41</td><td>88.54</td><td>6199.24</td><td>43.62</td><td>30.42</td><td>29.66</td><td>25.38</td><td>22.51</td><td>88.27</td><td>6199.24</td><td></td></tr><tr><td>PREMem</td><td>64.73</td><td>40.41</td><td>39.86</td><td>32.16</td><td>28.75</td><td>89.54</td><td>2032.40</td><td>68.03</td><td>29.42</td><td>27.14</td><td>21.50</td><td>21.59</td><td>86.87</td><td>2033.90</td><td></td></tr><tr><td>with bm25</td><td>59.37</td><td>38.66</td><td>38.19</td><td>30.20</td><td>26.88</td><td>89.22</td><td>2032.20</td><td>63.97</td><td>26.67</td><td>24.43</td><td>18.89</td><td>19.58</td><td>86.41</td><td>2033.90</td><td></td></tr><tr><td>PREMem_small</td><td>51.86</td><td>31.06</td><td>30.64</td><td>24.60</td><td>21.33</td><td>88.42</td><td>2032.30</td><td>64.52</td><td>27.21</td><td>25.03</td><td>19.54</td><td>19.12</td><td>86.57</td><td>2034.00</td><td></td></tr><tr><td rowspan="11">72B</td><td>Zero</td><td>20.61</td><td>12.59</td><td>12.27</td><td>10.32</td><td>6.83</td><td>86.29</td><td>0.00</td><td>25.15</td><td>8.43</td><td>6.72</td><td>6.94</td><td>8.01</td><td>83.07</td><td>0.00</td><td></td></tr><tr><td>Full</td><td>36.95</td><td>24.53</td><td>23.94</td><td>20.20</td><td>14.93</td><td>88.12</td><td>18710.50</td><td>63.97</td><td>20.83</td><td>19.50</td><td>14.92</td><td>20.77</td><td>85.94</td><td>19643.80</td><td></td></tr><tr><td>Turn</td><td>40.63</td><td>26.22</td><td>25.57</td><td>20.98</td><td>15.30</td><td>88.33</td><td>1854.30</td><td>63.71</td><td>26.28</td><td>24.85</td><td>20.26</td><td>21.20</td><td>87.27</td><td>2009.30</td><td></td></tr><tr><td>Session</td><td>30.89</td><td>20.76</td><td>20.31</td><td>16.54</td><td>9.59</td><td>87.38</td><td>1919.80</td><td>54.21</td><td>23.85</td><td>22.64</td><td>18.75</td><td>19.47</td><td>86.27</td><td>1989.30</td><td></td></tr><tr><td>Segment</td><td>38.61</td><td>25.60</td><td>25.13</td><td>21.11</td><td>15.09</td><td>88.48</td><td>1770.79</td><td>61.91</td><td>29.72</td><td>28.30</td><td>23.40</td><td>23.58</td><td>87.20</td><td>1881.45</td><td></td></tr><tr><td>SeCom</td><td>39.40</td><td>25.25</td><td>24.82</td><td>20.78</td><td>14.59</td><td>88.24</td><td>1775.10</td><td>58.51</td><td>28.02</td><td>26.54</td><td>22.06</td><td>22.22</td><td>86.90</td><td>1884.30</td><td></td></tr><tr><td>HippoRAG-2</td><td>45.95</td><td>29.79</td><td>29.27</td><td>23.35</td><td>18.84</td><td>88.04</td><td>3811.61</td><td>61.62</td><td>30.40</td><td>29.07</td><td>24.43</td><td>24.12</td><td>87.43</td><td>3811.61</td><td></td></tr><tr><td>A-Mem</td><td>53.58</td><td>36.25</td><td>35.42</td><td>29.46</td><td>25.43</td><td>89.34</td><td>45.59</td><td>31.68</td><td>30.94</td><td>26.35</td><td>23.08</td><td>28.56</td><td>88.56</td><td>6199.24</td><td></td></tr><tr><td>PREMem</td><td>67.50</td><td>45.36</td><td>44.89</td><td>36.12</td><td>30.90</td><td>90.19</td><td>2032.40</td><td>70.96</td><td>27.05</td><td>24.71</td><td>19.05</td><td>20.48</td><td>86.47</td><td>2033.90</td><td></td></tr><tr><td>with bm25</td><td>62.03</td><td>42.10</td><td>41.67</td><td>34.21</td><td>28.76</td><td>89.98</td><td>2032.20</td><td>66.96</td><td>25.05</td><td>22.85</td><td>17.56</td><td>18.52</td><td>86.23</td><td>2034.50</td><td></td></tr><tr><td>PREMem_small</td><td>56.89</td><td>35.44</td><td>34.97</td><td>27.95</td><td>24.07</td><td>89.13</td><td>2032.30</td><td>66.68</td><td>25.59</td><td>23.39</td><td>17.87</td><td>18.73</td><td>86.39</td><td>2034.00</td><td></td></tr><tr><td rowspan="11">gamma-3</td><td>Zero</td><td>18.32</td><td>11.35</td><td>10.99</td><td>10.20</td><td>5.50</td><td>86.25</td><td>1.00</td><td>20.91</td><td>11.33</td><td>10.89</td><td>7.54</td><td>4.29</td><td>86.43</td><td>1.00</td><td></td></tr><tr><td>Full</td><td>32.45</td><td>20.81</td><td>20.48</td><td>17.15</td><td>11.93</td><td>88.14</td><td>18669.20</td><td>42.49</td><td>26.32</td><td>25.26</td><td>19.46</td><td>15.77</td><td>87.58</td><td>19591.00</td><td></td></tr><tr><td>Turn</td><td>34.96</td><td>22.19</td><td>21.80</td><td>17.37</td><td>13.01</td><td>88.42</td><td>1838.90</td><td>46.53</td><td>26.78</td><td>25.96</td><td>19.05</td><td>14.94</td><td>87.92</td><td>1984.00</td><td></td></tr><tr><td>Session</td><td>28.31</td><td>18.28</td><td>17.94</td><td>14.37</td><td>9.59</td><td>87.90</td><td>1929.00</td><td>38.22</td><td>23.54</td><td>22.95</td><td>16.90</td><td>13.53</td><td>87.09</td><td>1967.40</td><td></td></tr><tr><td>Segment</td><td>36.49</td><td>21.00</td><td>20.65</td><td>16.90</td><td>12.74</td><td>88.23</td><td>1843.93</td><td>47.79</td><td>31.16</td><td>30.28</td><td>21.55</td><td>17.67</td><td>87.79</td><td>1969.70</td><td></td></tr><tr><td>SeCom</td><td>35.24</td><td>21.16</td><td>20.86</td><td>16.86</td><td>12.78</td><td>88.06</td><td>1844.72</td><td>45.15</td><td>29.68</td><td>28.91</td><td>21.66</td><td>16.58</td><td>87.72</td><td>1971.58</td><td></td></tr><tr><td>HippoRAG-2</td><td>41.00</td><td>24.47</td><td>24.04</td><td>19.22</td><td>15.68</td><td>88.71</td><td>4000.39</td><td>47.48</td><td>31.14</td><td>30.21</td><td>23.81</td><td>18.86</td><td>88.27</td><td>4000.39</td><td></td></tr><tr><td>A-Mem</td><td>42.30</td><td>28.86</td><td>28.49</td><td>22.23</td><td>19.85</td><td>88.42</td><td>6250.81</td><td>45.70</td><td>33.11</td><td>32.66</td><td>28.11</td><td>25.70</td><td>88.84</td><td>6250.81</td><td></td></tr><tr><td>PREMem</td><td>53.39</td><td>31.63</td><td>31.37</td><td>25.00</td><td>23.84</td><td>89.03</td><td>50.14</td><td>29.00</td><td>27.29</td><td>19.60</td><td>17.01</td><td>86.98</td><td>1957.80</td><td></td><td></td></tr><tr><td>with bm25</td><td>49.71</td><td>30.81</td><td>30.68</td><td>25.13</td><td>22.76</td><td>89.09</td><td>1964.60</td><td>48.40</td><td>26.41</td><td>24.78</td><td>17.32</td><td>14.65</td><td>86.86</td><td>1961.50</td><td></td></tr><tr><td>PREMem_small</td><td>48.50</td><td>30.08</td><td>29.65</td><td>24.94</td><td>21.70</td><td>89.36</td><td>1963.70</td><td>47.83</td><td>27.24</td><td>25.60</td><td>18.10</td><td>16.11</td><td>86.78</td><td>1956.30</td><td></td></tr><tr><td rowspan="11">gamma-3</td><td>Zero</td><td>27.72</td><td>18.05</td><td>17.91</td><td>14.22</td><td>7.38</td><td>88.57</td><td>1.00</td><td>21.28</td><td>11.19</td><td>10.84</td><td>7.36</td><td>4.26</td><td>85.66</td><td>1.00</td><td></td></tr><tr><td>Full</td><td>34.80</td><td>21.40</td><td>21.25</td><td>17.25</td><td>12.50</td><td>88.23</td><td>18669.20</td><td>43.68</td><td>27.62</td><td>27.13</td><td>21.06</td><td>18.73</td><td>88.00</td><td>19591.00</td><td></td></tr><tr><td>Turn</td><td>36.60</td><td>22.45</td><td>22.21</td><td>17.50</td><td>13.15</td><td>88.09</td><td>1838.90</td><td>45.46</td><td>27.05</td><td>26.54</td><td>20.59</td><td>18.89</td><td>87.91</td><td>1984.00</td><td></td></tr><tr><td>Session</td><td>28.76</td><td>16.93</td><td>16.68</td><td>13.23</td><td>9.64</td><td>87.74</td><td>1929.00</td><td>37.95</td><td>24.19</td><td>23.80</td><td>17.85</td><td>16.16</td><td>86.96</td><td>1967.40</td><td></td></tr><tr><td>Segment</td><td>34.15</td><td>21.09</td><td>20.81</td><td>17.35</td><td>12.38</td><td>87.92</td><td>1843.93</td><td>46.72</td><td>31.31</td><td>30.79</td><td>23.73</td><td>22.12</td><td>87.92</td><td>1971.58</td><td></td></tr><tr><td>SeCom</td><td>37.43</td><td>23.52</td><td>23.37</td><td>17.34</td><td>13.83</td><td>88.31</td><td>1844.72</td><td>44.48</td><td>30.09</td><td>29.67</td><td>23.22</td><td>20.17</td><td>87.61</td><td>1971.58</td><td></td></tr><tr><td>HippoRAG-2</td><td>43.90</td><td>27.44</td><td>26.77</td><td>21.35</td><td>17.44</td><td>88.71</td><td>4000.39</td><td>44.28</td><td>29.76</td><td>23.21</td><td>20.77</td><td>17.79</td><td>87.79</td><td>4000.39</td><td></td></tr><tr><td>A-Mem</td><td>50.38</td><td>28.06</td><td>27.63</td><td>22.45</td><td>19.73</td><td>87.88</td><td>6250.81</td><td>43.94</td><td>31.18</td><td>30.81</td><td>25.95</td><td>23.63</td><td>88.47</td><td>6250.81</td><td></td></tr><tr><td>PREMem</td><td>57.70</td><td>34.44</td><td>33.93</td><td>27.73</td><td>25.58</td><td>89.20</td><td>1962.70</td><td>54.55</td><td>30.61</td><td>28.85</td><td>21.83</td><td>19.21</td><td>87.54</td><td>1957.80</td><td></td></tr><tr><td>with bm25</td><td>53.70</td><td>32.08</td><td>31.76</td><td>26.47</td><td>24.60</td><td>89.97</td><td>1964.60</td><td>48.16</td><td>26.67</td><td>25.62</td><td>19.12</td><td>16.98</td><td>87.17</td><td>1961.50</td><td></td></tr><tr><td>PREMem_small</td><td>55.01</td><td>31.56</td><td>31.14</td><td>25.46</td><td>22.72</td><td>89.74</td><td>1963.70</td><td>47.56</td><td>29.79</td><td>26.79</td><td>19.62</td><td>17.83</td><td>87.24</td><td>1956.30</td><td></td></tr><tr><td rowspan="65">gamma-3</td><td>Zero</td><td>30.15</td><td>17.58</td><td>17.43</td><td>12.88</td><td>5.95</td><td>88.62</td><td>1.00</td><td>22.16</td><td>10.19</td><td>9.85</td><td>6.68</td><td>5.20</td><td>85.63</td><td>1.00</td><td></td></tr><tr><td>Full</td><td>35.90</td><td>22.55</td><td>22.18</td><td>18.51</td><td>12.11</td><td>88.72</td><td>18669.20</td><td>47.37</td><td>29.40</td><td>28.49</td><td>24.02</td><td>22.15</td><td>88.16</td><td>19591.00</td><td></td></tr><tr><td>Turn</td><td>37.98</td><td>23.62</td><td>23.25</td><td>17.95</td><td>13.07</td><td>88.69</td><td>1838.90</td><td>47.92</td><td>27.53</td><td>26.76</td><td>21.01</td><td>17.87</td><td>87.99</td><td>1984.00</td><td></td></tr><tr><td>Session</td><td>27.58</td><td>17.67</td><td>16.50</td><td>13.07</td><td>7.77</td><td>88.31</td><td>1929.00</td><td>43.32</td><td>24.57</td><td>23.87</td><td>18.32</td><td>15.95</td><td>86.97</td><td>1967.40</td><td></td></tr><tr><td>Segment</td><td>36.83</td><td>21.54</td><td>21.11</td><td>16.92</td><td>10.17</td><td>88.64</td><td>1943.93</td><td>50.49</td><td>31.47</td><td>30.74</td><td>23.97</td><td>20.94</td><td>87.96</td><td>1969.70</td><td></td></tr><tr><td>SeCom</td><td>38.93</td><td>23.16</td><td>22.91</td><td>18.22</td><td>11.28</td><td>88.65</td><td>1449.49</td><td>51.25</td><td>34.67</td><td>33.75</td><td>27.61</td><td>25.22</td><td>88.43</td><td>1701.81</td><td></td></tr><tr><td>HippoRAG-2</td><td>43.15</td><td>27.34</td><td>27.00</td><td>24.42</td><td>18.68</td><td>89.46</td><td>4000.39</td><td>49.49</td><td>30.59</td><td>35.13</td><td>30.14</td><td>26.16</td><td>87.89</td><td>4000.39</td><td></td></tr><tr><td>A-Mem</td><td>45.32</td><td>31.90</td><td>31.33</td><td>26.27</td><td>21.79</td><td>88.60</td><td>6250.81</td><td>44.47</td><td>32.76</td><td>32.23</td><td>27.08</td><td>23.75</td><td>88.73</td><td>5205.81</td><td></td></tr><tr><td>PREMem</td><td>61.86</td><td>39.20</td><td>38.81</td><td>33.85</td><td>25.50</td><td>90.88</td><td>59.85</td><td>54.55</td><td>30.61</td><td>31.41</td><td>24.04</td><td>19.89</td><td>87.59</td><td>2035.80</td><td></td></tr><tr><td>with bm25</td><td>56.01</td><td>36.70</td><td>35.31</td><td>33.28</td><td>26.04</td><td>90.85</td><td>59.75</td><td>52.89</td><td>30.74</td><td>31.41</td><td>23.40</td><td>19.42</td><td>87.31</td><td>2035.80</td><td></td></tr><tr><td>PREMem_small</td><td>57.15</td><td>36.41</td><td>35.03</td><td>32.26</td><td>25.61</td><td>91.00</td><td>2034.10</td><td>54.81</td><td>32.06</td><td>30.78</td><td>23.66</td><td>20.40</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="3"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="13"></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="10"></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td colspan="12"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

# E Implementation Details

For our implementation, we set the threshold parameter $\theta$ to 0.6 for memory fragment selection. In Steps 1 and 2 of our methodology, we utilized few-shot examples to enhance performance, with the complete prompt templates available in Appendix A. To ensure consistent evaluation across experiments, we conducted preference testing to determine which answers were more favorable. Our analysis revealed no statistically significant difference between using GPT-4o and GPT-4.1-mini as judges, leading us to select GPT-4.1-mini as our LLM-as-a-judge for all evaluations.

For embedding generation, we employed NovaSearch/stella_en_400M_v5 (MIT license) from Huggingface. Our experiments were conducted across two model families with varying parameter sizes: Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-14B-Instruct, and Qwen/Qwen2.5-72B-Instruct from the Qwen family, and google/gemma-3-4b-it, google/gemma-3-12b-it, and google/gemma-3-27b-it from the Gemma family.

In compliance with licensing requirements, we adhered to both the Qwen and Gemma license agreements. Qwen requires attribution by displaying "Built with Qwen" or "Improved using Qwen" when distributing AI models and special authorization for services with over 100 million monthly active users. Gemma requires adherence to their use restrictions policy and proper attribution with copies of their license agreement to recipients. Our academic research complies with these requirements, including appropriate model attribution and usage within permitted applications.

Our hardware configuration consisted of an Intel(R) Xeon(R) Gold 6448Y CPU and four NVIDIA H100 80GB HBM3 GPUs for accelerated model inference and training.

You are an AI evaluator tasked with assessing the accuracy of predicted answers to questions. Your goal is to determine how well the predicted answer aligns with the expected (gold) answer and provide a numerical score.

You will be given the following information:

<question>

{{$question}}

</question>

<gold_answer>

{{$gold_answer}}

</gold_answer>

<predicted_answer>

{{$predicted_answer}}

</predicted_answer>

Instructions:

1. Carefully read the question, gold answer, and predicted answer.   
2. Analyze the relationship between the gold answer and the predicted answer.   
3. Consider the following criteria:   
- Does the predicted answer address the same topic as the gold answer?   
- For time-related questions, does the predicted answer refer to the same time period, even if the format differs?   
- Is the core information in the predicted answer consistent with the gold answer, even if expressed differently?   
4. Assign a score from 0 to 100, where:   
- 0 means the predicted answer is completely unrelated or incorrect   
- 100 means the predicted answer perfectly matches the gold answer   
- Scores in between reflect partial correctness or relevance   
5. Output your result as a single integer only. Do not use JSON or any other format.

Important:

- Do not include any examples in your analysis or output.   
- Provide only the integer score as your output, with no explanation or formatting.

Score:

Figure 7: LLM-as-a-judge prompt used to evaluate response quality.

Algorithm 1 Memory-Enhanced Conversational Learning with Dynamic Clustering and Reasoning   
1: Input: $LLM_{extract}$ , $LLM_{reason}$ , $LLM_{response}$ , $f_{emb}$ 2: Initialization: $P_0 = \{\}$ , $\mathcal{M} = \{\}$ , $\mathcal{R} = \{\}$ 3: for $i = 1, \dots, N$ do  
4: Step 1: Episodic Memory Extraction  
5: Observe conversation session $S_i$ 6: Extract memory fragments from $S_i$ :

$$
\{m _ {i} ^ {1}, \dots m _ {i} ^ {n _ {i}} \} \gets L L M _ {e x t r a c t} (S _ {i})
$$

7: Embed memory fragments: $\{ e _ { i } ^ { j } \} _ { j = 1 } ^ { n _ { i } }$ where $e _ { i } ^ { j } = f _ { e m b } ( m _ { i } ^ { j } )$

8: Cluster fragments into $C _ { i } = \{ c _ { 1 } , \ldots , c _ { k _ { i } } \}$ ▷ using silhouette scores 9: Construct a set $C P _ { i }$ : $\triangleright$ using cosine similarity

$$
C P _ {i} = \{(p, c): \operatorname {s i m} (p, c) > \theta , p \in P _ {i - 1}, c \in C _ {i} \}
$$

10: Step 2: Pre-Storage Memory Reasoning   
11: for $( p , c ) \in C P _ { i }$ do   
12: $M _ { p } \gets$ memory fragments in cluster $p$   
13: $M _ { c } \gets$ memory fragments in cluster $c$   
14: Generate reasoning

$$
\{r _ {p, c} ^ {j} \} _ {j = 1} ^ {d _ {p, c}} \gets L L M _ {r e a s o n} (M _ {p}, M _ {c})
$$

15: Store reasoning memory fragments: $\mathcal { R }  \mathcal { R } \cup \{ r _ { p , c } ^ { 1 } , \cdot \cdot \cdot , r _ { p , c } ^ { d _ { p , c } } \}$   
16: Update $P _ { i }$

$$
P _ {i} = P _ {i - 1} \backslash \{p: \exists c. s. t. (p, c) \in C P _ {i} \} \cup C _ {i}
$$

17: end for

18: Store raw memory fragments: $\mathcal { M } \gets \mathcal { M } \cup \{ m _ { i } ^ { 1 } , . . . , m _ { i } ^ { n _ { i } } \}$   
19: end for   
20: Inference Phase   
21: Get user query $q$ , compute $e _ { q } \gets f _ { \mathrm { e m b } } ( q )$   
22: Retrieve top- $k$ by similarity over $\mathcal { M } \cup \mathcal { R }$

$$
\operatorname {c o n t e x t} \leftarrow \operatorname {T o p K} _ {k} \left(\mathcal {M} \cup \mathcal {R}; \operatorname {s i m} \left(f _ {\mathrm {e m b}} (\cdot), e _ {q}\right)\right)
$$

23: Generate answer: $r e s p o n s e \gets L L M _ { r e s p o n s e } ( c o n t e x t , q )$   
24: Output: response
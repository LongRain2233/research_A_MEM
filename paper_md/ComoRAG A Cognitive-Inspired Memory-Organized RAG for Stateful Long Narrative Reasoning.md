# ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning

Juyuan Wang1* Rongchen Zhao1* Wei Wei2 Yufeng Wang1 Mo $\mathbf { V } \mathbf { u } ^ { 4 }$ Jie Zhou4 Jin $\mathbf { X } \mathbf { u } ^ { 1 , 3 }$ Liyan $\mathbf { X } \mathbf { u } ^ { 4 \dagger }$

1School of Future Technology, South China University of Technology

2Independent Researcher 3Pazhou Lab, Guangzhou

4WeChat AI, Tencent

# Abstract

Narrative comprehension on long stories and novels has been a challenging domain attributed to their intricate plotlines and entangled, often evolving relations among characters and entities. Given the LLM’s diminished reasoning over extended context and its high computational cost, retrieval-based approaches remain a pivotal role in practice. However, traditional RAG methods could fall short due to their stateless, single-step retrieval process, which often overlooks the dynamic nature of capturing interconnected relations within long-range context. In this work, we propose ComoRAG, holding the principle that narrative reasoning is not a oneshot process, but a dynamic, evolving interplay between new evidence acquisition and past knowledge consolidation, analogous to human cognition on reasoning with memory-related signals in the brain. Specifically, when encountering a reasoning impasse, ComoRAG undergoes iterative reasoning cycles while interacting with a dynamic memory workspace. In each cycle, it generates probing queries to devise new exploratory paths, then integrates the retrieved evidence of new aspects into a global memory pool, thereby supporting the emergence of a coherent context for the query resolution. Across four challenging long-context narrative benchmarks $2 0 0 \mathrm { K } +$ tokens), ComoRAG outperforms strong RAG baselines with consistent relative gains up to $11 \%$ compared to the strongest baseline. Further analysis reveals that ComoRAG is particularly advantageous for complex queries requiring global context comprehension, offering a principled, cognitively motivated paradigm towards retrieval-based stateful reasoning. Our framework is made publicly available at https://github.com/EternityJune25/ComoRAG.

# 1 Introduction

The core challenge of long narrative comprehension lies not merely in connecting discrete pieces of evidence, a task more naturally defined as multi-hop Question Answering (QA), but in performing a dynamic cognitive synthesis to grasp necessary background and content progression (Xu et al. 2024a). Unlike multi-hop QA (Yang et al. 2018), which seeks a static path through fixed facts, narrative comprehension requires emulating a human reader: continuously build-

![](images/b4b84dcbd695a2872f2be8d30b1b7720c6db91f24647c8db79e6bd2cf8d6acd5.jpg)  
Figure 1: Comparison of RAG reasoning paradigms.

ing and revising a global mental model of the plot, characters, and their evolving motivations (Johnson-Laird 1983). The complexity of this process is well exemplified by a classic question “Why did Snape kill Dumbledore?” from the Harry Potter series. Answering this requires weaving a complete web of evidence from disparate clues spanning multiple books—Dumbledore’s terminal illness, the Unbreakable Vow, and Snape’s deeply concealed loyalty. The true significance of these clues is only fully reconciled in hindsight. This capability is what we term stateful reasoning: it demands more than linking static evidence; it requires maintaining a dynamic memory of the narrative, one that is constantly updated as new revelations emerge. Long-context LLMs have demonstrated promising performance on benchmarks such as the “Needle in a Haystack” (Eisenschlos, Yogatama, and Al-Rfou 2023). However, their capacity to process long narratives $2 0 0 \mathrm { k } +$ tokens) remains limited by finite context windows. Furthermore, as the input length increases, these models are prone to the “lost in the middle” problem (Liu et al. 2024), which raises perplexity and impairs gen-

eration quality. This limitation is particularly pronounced in narrative tasks which require stateful reasoning. As a result, retrieval-augmented generation (RAG) (Lewis et al. 2020) has emerged as an important direction for tackling long context comprehension with LLMs, leveraging text embeddings or more advanced retrieval paradigms such as embeddings situated on global context (Wu et al. 2025).

However, existing RAG methods still struggle to effectively address this challenge. Advanced single-step retrieval remains limited by its static index. This includes methods such as RAPTOR (Sarthi et al. 2024), which clusters and summarizes text chunks to retrieve at different levels of details; HippoRAGv2 (Gutierrez et al. ´ 2025) and GraphRAG (Edge et al. 2025), which build knowledge graphs to achieve multi-hop reasoning in a single retrieval step. Nonetheless, one-shot static retrieval inevitably leads to shallow comprehension. For example, the evidence about Snape in Fig. 1(a) can mislead the model into making a false inference.

As a remedy, multi-step retrieval methods offer a more promising direction, such as IRCoT (Trivedi et al. 2023), which interleaves the retrieval process with Chain-of-Thought reasoning (Wei et al. 2022); Self-RAG (Asai et al. 2024), which trains a model to adaptively retrieve and reflect on evidence; and MemoRAG (Qian et al. 2025), which uses a dual-system architecture to generate clues from compressed global context. These methods all target to obtain richer context through iterative retrieval. However, their retrieval steps are typically independent, which lack coherent reasoning throughout explicit narrative progression, featuring fragmented evidence with a stateless comprehension. As illustrated in Figure 1(b), due to a lack of dynamic memory, multi-step retrieval fails to integrate contradictory evidence such as “Snape protects/bullies Harry” and cannot understand the evolution of his actions, ultimately unable to yield the correct answer.

In this work, we seek inspiration from the function of Prefrontal Cortex (PFC) in human brains, which employs a sophisticated reasoning process called Metacognitive Regulation (Fernandez-Duque, Baird, and Posner 2000). This process is not a single action but a dynamic interplay between new evidence acquisition, driven by goal-directed memory probes (Dobbins and Han 2006; Miller and Constantinidis 2024), and subsequent knowledge consolidation. During consolidation, new findings are integrated with past information to construct an evolving, coherent narrative. This iterative cycle allows the PFC to continuously assess its understanding and revise its strategy, providing a direct cognitive blueprint for our framework’s stateful reasoning approach.

We introduce ComoRAG, a cognitive-inspired, memoryorganized RAG framework, imitating the human Prefrontal Cortex (PFC) for achieving stateful reasoning. At its core is a dynamic cognitive loop operating on a memory workspace, which actively probes and integrates new evidence to build a coherent narrative comprehension.

This process, as illustrated in Figure 1(c), is a closed loop of evolving reasoning states. Faced with a complex query like “Why did Snape kill Dumbledore?”, the system’s memory state evolves from an initial “causally incomplete event” (Snape kills Albus), to an “apparent contradiction” upon

finding contradictory information (Snape protects Harry), and ultimately to a logically consistent coherent context through deeper exploration and evidence fusion. Only in this final, complete cognitive state can ComoRAG perform the correct stateful reasoning, deriving the true insight that it was “an act of loyalty, not betrayal”.

This cognitively-inspired design yields substantial improvements across four challenging long-context narrative benchmarks. ComoRAG is shown to consistently outperform all categories of strong baselines across each dataset. Our analysis reveals several key findings. First, these gains stem directly from the cognitive loop, which transforms a static knowledge base into a dynamic reasoning engine; for instance, accuracy on EN.MC jumps from a static-retrieval baseline of $6 4 . 6 \%$ to $7 2 . 9 \%$ , with performance efficiently converging in around 2-3 cycles. Second, our framework excels on narrative queries that require global understanding of plot progression, achieving up to a $19 \%$ relative F1 improvement on these challenging question types where others falter. Finally, our framework demonstrates remarkable modularity and generalizability. Its core loop can be flexibly integrated to existing RAG methods such as RAPTOR, which directly yields a $21 \%$ relative accuracy gain). Also, switching to a stronger model as the backbone LLM agents can upgrade reasoning in the entire cognitive loop, attaining accuracy from $7 2 . 9 3 \%$ to $7 8 . 1 7 \%$ . These results collectively validate that ComoRAG provides a principled, cognitivelyinspired new paradigm for retrieval-based long narrative comprehension towards stateful reasoning.

# 2 Methodology

We introduce ComoRAG, an autonomous cognitive architecture designed to formalize and implement the process of Metacognitive Regulation outlined in the Introduction. The architecture’s design is directly inspired by the functional mechanisms of the Prefrontal Cortex (PFC) and is founded on three conceptual pillars: (1) a Hierarchical Knowledge Source for deep contextual understanding; (2) a Dynamic Memory Workspace for tracking and integrating the multiturn reasoning; and (3) a Metacognitive Control Loop that drives the entire resolving procedure.

# 2.1 Problem Formulation: Towards Principled Narrative Reasoning

Our objective is to design a framework for stateful reasoning in RAG scenarios. Especially, it aims to resolve those queries that require global context comprehension in the first place, commonly seen in narratives, where conventional RAG may fail to recognize relevant context based on the surface form of queries. Formally, denote the initial query as $q _ { i n i t }$ , and a knowledge source $\mathcal { X }$ derived upon the original context, our framework $F$ leverages a series of adaptive operations to yield the final answer, $A _ { f i n a l }$ , through discrete time steps $t = 1 , \dots , T$ with underlying memory control.

At the beginning of each step t, $F$ determines its focus of reasoning—a set of new probing queries $\mathcal { P } ^ { ( t ) }$ , representing new information to seek that may logically deepen the query comprehension and ultimately complement the an-

![](images/387d796b42b1b937202f3d73ce8b3c737495fe03fd40d1612d90385dbb90fee8.jpg)  
Figure 2: An illustration of ComoRAG. Triggered by a reasoning impasse (Failure), the Metacognitive Regulation loop consists of five core operations described in Section 2.3: 1) Self-Probe to devise new exploratory probing queries based on past memory units; 2) Tri-Retrieve to retrieve evidence from three knowledge sources; 3) $\scriptstyle \overbrace { \left[ \mathrm { M e m - E n c o d e } \right] } ^ { \mathrm { - } }$ to form new memory units on how the latest evidence of new aspects could complement the final query resolution; 4) Mem-Fuse to generate cues integrating new and past memory units; 5) Try-Answer to perform query answering using new memory information produced in this cycle.

swer resolution. With newly retrieved information by $\mathcal { P } ^ { ( t ) }$ at each step, the framework utilizes the global memory pool maintained till the prior step M(t−1)pool , $\mathcal { M } _ { p o o l } ^ { ( t - 1 ) }$ and produces either the final answer, or a Failure Signal, indicating a reasoning impasse—and updates the memory pool to $\bar { \mathcal { M } } _ { p o o l } ^ { ( t ) }$ , acknowledge source, memory space and retrieval operations.

# 2.2 The Hierarchical Knowledge Source

To overcome the limitations of a monolithic representation of the given context, our framework first builds a hierarchical knowledge index $\mathcal { X }$ for retrieval that models the raw text from three complementary cognitive dimensions, analogous to how the PFC integrates different memory types from various brain regions, particularly supporting cross-layer reasoning from raw evidence to abstract relationships.

Veridical Layer: Grounding in Factual Evidence. To ensure all reasoning is traceable to source evidence, a veridical layer $\mathcal { X } ^ { v e r }$ is firstly established, constituted by raw text chunks directly, analogous to the precise recall of factual details in human memory. For more accurate retrieval on text chunks, we instruct a LLM to generate knowledge triples (subject-predicate-object) for each text chunk. These triples participate in each retrieval, and strengthen the matching between an incoming query and the corresponding text chunk, which is proven effective by HippoRAG (Jimenez Gutierrez et al. 2024). Further details are described in Appendix B.

Semantic Layer: Abstracting Thematic Structure. To capture thematic and conceptual connections that transcend

across long-range contextual dependencies, a semantic layer $\mathcal { X } ^ { s e m }$ is built, inspired by the prior work RAPTOR that employs a GMM-driven clustering algorithm to recursively summarize semantically similar text chunks into a hierarchical summary tree. We reckon such semantic abstraction is necessary for deeper comprehension and follow the same formulism. These summary nodes enable the framework to retrieve conceptual information beyond the surface level.

Episodic Layer: Reconstructing Narrative Flow. The previous two layers equip views of both factual details and high-level concepts. However, they lack temporal development or plot progression that can be especially crucial for narratives. To enable such view with long-range causal chains, we introduce the episodic layer, $\chi ^ { e p i }$ , which aims to reconstruct the plotline and story arc by capturing the sequential narrative development. The process features a sliding window summarization across text chunks; each resulting node is then a summary that aggregates the narrative development of continuous or causally related events according to the timeline. Optionally, the sliding window process can be applied recursively to form higher-level views of content progression, extracting different levels of narrative flow as part of the knowledge source.

# 2.3 The Architecture of Metacognitive Regulation

The core of ComoRAG is a control loop that fully realizes the concept of metacognitive regulation. It is composed of a Regulatory Process for reflection and planning at each step, and a Metacognitive Process for executing reasoning and memory management with the Memory Workspace.

Dynamic Memory Workspace. The memory workspace contains memory units that serve as the bridge for a cohesive multi-step exploration and reasoning by metacognitive regulation. Each memory unit $m$ functionally concludes one retrieval operation, denoted as a tuple of three elements: $m ~ = ~ ( p , \mathcal { E } _ { p } ^ { \bar { t } y p e } , \mathcal { C } _ { p } ^ { t y p e } )$ , where $p$ is the probing query that triggers this retrieval; $\mathcal { E } _ { p } ^ { t y p e }$ is the homogeneous set of evidence retrieved from a single knowledge layer $( t y p e \in \{ v e r , s e m , e p i \} )$ ; and $\mathcal { C } _ { p } ^ { t y p e }$ is a synthesized cue that reflects how these retrieved evidence by the probe $p$ could complement the comprehension and resolution of the original query LLM in the rol $q _ { i n i t }$ . Concretely, omprehensio $\mathcal { C } _ { p } ^ { t y p e }$ isnt, erated by a, denoted as $\mathcal { C } _ { p } ^ { t y p e } = \pi _ { c u e } ( q _ { i n i t } , p , \mathcal { E } _ { p } ^ { \bar { t } y p e } )$ . $\pi _ { c u e }$

The formation of a memory unit $( p , \mathcal { E } _ { p } ^ { t y p e } , \mathcal { C } _ { p } ^ { t y p e } )$ by each retrieval is defined as a $\scriptstyle \left[ { \overline { { \mathbf { M e m - E n c o d e } } } } \right]$ operation. The memory workspace/pool will be utilized and updated throughout the reasoning cycle described below.

The Regulatory Process. The regulatory process is invoked at the beginning of a reasoning cycle/step $t$ if the preceding cycle $t - 1$ is concluded in failure. The core operation, Self-Probe , plans new probing queries of which retrieved information may contribute to the final answer, thereby devising new exploratory paths to break the impasse. It is orchestrated by a Regulation Agent, $\pi _ { p r o b e }$ , whose decisions are informed by the reflection on the prior failure, exploring for more necessary background or relevant information towards a full context comprehension to resolve the original query. Self-Probe takes three inputs: (1) the ultimate goal qinit; (2) the complete exploration probing history P(t−1)hist $q _ { i n i t }$ $\mathcal { P } _ { h i s t } ^ { ( t - 1 ) }$ hist up to the end of the last step; and (3) the immediate knowledge gaps that caused the failure, concretized by all synthesized cues of memory units generated in the prior step, denoted as $\{ \mathcal { C } \} ^ { ( t - 1 ) }$ . Its output $\mathcal { P } ^ { ( t ) }$ is a new, strategic set of retrieving probes for the current cycle $t$ :

$$
\mathcal {P} ^ {(t)} = \pi_ {\text {p r o b e}} \left(q _ {\text {i n i t}}, \mathcal {P} _ {\text {h i s t}} ^ {(t - 1)}, \{\mathcal {C} \} ^ {(t - 1)}\right) \tag {1}
$$

The Metacognitive Process. The metacognitive process takes the new probes for this cycle $\mathcal { P } ^ { ( t ) }$ , and performs reasoning towards resolving the original query while keeping track of the progress with the memory space. It comprises a series of operations, described in details as follows.

Tri-Retrieve : for each probing query $p \in \mathcal { P } ^ { ( t ) }$ , a retrieval is conducted on each knowledge layer $\mathcal { X } ^ { t y p e }$ where type $\in$ $\{ v e r , s e m , e p i \}$ , such that evidence of high embedding similarity to $p$ per layer is retrieved in a standard Dense Passage Retrieval paradigm, with each evidence being either the raw text chunk, a semantically clustered summary, or a narrative flow summary.

$\scriptstyle \left[ { \mathrm { M e m - E n c o d e } } \right] :$ for each $p$ and type, the retrieved evidence is immediately processed by the aforementioned Mem-Encode to generate a new memory unit that keeps track of how this specific probing could complement to the final answer. The number of all generated memory units at this step can be denoted as $| \mathcal { M } _ { e n c o d e } ^ { ( t ) } | = 3 \times | \mathcal { P } ^ { ( t ) } |$ .

Mem-Fu se : new memory units in the above step M(t)enc $\mathcal { M } _ { e n } ^ { ( t ) }$ ode mainly emphasize aspects probed in the current cycle. To

fully utilize the past experience and historical knowledge, the framework further identifies relevant synthesized cues from past units in the existing memory pool $\boldsymbol { \mathcal { M } _ { p o o l } ^ { t - 1 } }$ , then generates a new synthesized cue for fusing past relevant evidence. Let $\boldsymbol { \mathcal { M } } _ { p o o l } ^ { t - 1 } \circ \boldsymbol { q } _ { i n i t }$ represent past memory units whose cues are of high embedding similarity with $q _ { i n i t }$ , and denote a LLM as Integration Agent $\pi _ { f u s e }$ that synthesizes these relevant past evidence into a high-level background summary, the new cue fusing past memory $\mathcal { C } _ { f u s e } ^ { ( t ) }$ is then:

$$
\mathcal {C} _ {f u s e} ^ {(t)} = \pi_ {f u s e} \left(q _ {i n i t}, \mathcal {M} _ {p o o l} ^ {t - 1} \circ q _ {i n i t}\right) \tag {2}
$$

Try-Answe r : with the new probing evidence in M(t)enc $\mathcal { M } _ { e n c o d e } ^ { ( t ) }$ and the past-fusing cue $\mathcal { C } _ { f u s e } ^ { ( t ) }$ , a QA Agent, $\pi _ { Q A }$ , is applied to these contexts to produce the cycle’s final output $O ^ { ( t ) }$ :

$$
O ^ {(t)} = \pi_ {Q A} \left(q _ {\text {i n i t}}, \mathcal {M} _ {\text {e n c o d e}} ^ {(t)}, \mathcal {C} _ {\text {f u s e}} ^ {(t)}\right) \tag {3}
$$

Specifically, a LLM is instructed to take these latest evidence and the past background as the context, and determine whether the original query can be resolved. It either yields the final answer and terminates the entire reasoning loop, or signals Failure and continues to the next step.

Mem-Update : this last step in a cycle simply incorporates the newly generated memory units into the global pool, with their embedding encoded, for future retrieval and reasoning:

$$
\mathcal {M} _ {\text {p o o l}} ^ {(t)} \leftarrow \mathcal {M} _ {\text {p o o l}} ^ {(t - 1)} \cup \mathcal {M} _ {\text {e n c o d e}} ^ {(t)} \tag {4}
$$

ComoRAG With the above six steps from Tri-Retrieve to Mem-Update , one cycle of the cognitive loop is realized. For the initial step as in $t = 0$ , ComoRAG starts with one round of Tri-Retrieve followed by Try-Answer . If Failure is signaled, it initiates the Metacognitive loop of stateful reasoning on exploratory paths, characterized by the interlocking operations with the memory workspace, which enables to tackle complex narrative comprehension.

In essence, our framework grasps on the principle that for long context comprehension, especially in narratives where the entire context is cohesively interconnected through the underlying plot progression (Xu et al. 2024a), the query resolution is not a linear pipeline; rather, it is a dynamic, evolving interplay between new evidence acquisition and past knowledge consolidation, analogous to the human cognitive process. The overall process is further depicted in the algorithm of Appendix A; detailed prompts used by each LLM agent are provided in Appendix D.

# 3 Experimental Settings

Datasets Our experiments cover four long-context narrative understanding datasets for comprehensive evaluation, featuring both question answering through free generation (QA), and multi-choice questions by selecting the best option (MC).

• NarrativeQA (Kocisky et al. ´ 2017): a QA dataset consisting of books and movie scripts. For ease of computation, we follow prior works and randomly sample 500 questions from the test set, with average context length 58k tokens.

Table 1: Evaluation results on four long narrative comprehension datasets. For fair comparison, all methods use GPT-4o-mini as the LLM backbone, and all non-naive RAG methods use BGE-M3 for retrieval (details in Section 3). We highlight the best and second-best results. ComoRAG is shown consistently outperform all baselines across all datasets.   

<table><tr><td rowspan="2">Category</td><td rowspan="2">Method</td><td colspan="2">NarrativeQA</td><td colspan="2">EN.QA</td><td>EN.MC</td><td>DetectiveQA</td><td colspan="2">QA Avg.</td><td>MC Avg.</td></tr><tr><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>ACC</td><td>ACC</td><td>F1</td><td>EM</td><td>ACC</td></tr><tr><td>LLM</td><td>GPT-4o-mini</td><td>27.29</td><td>7.00</td><td>29.83</td><td>12.82</td><td>30.57</td><td>30.68</td><td>28.56</td><td>9.91</td><td>30.63</td></tr><tr><td rowspan="3">Naive RAG</td><td>BGE-M3(0.3B)</td><td>23.16</td><td>15.10</td><td>23.71</td><td>16.24</td><td>59.82</td><td>54.54</td><td>23.44</td><td>15.67</td><td>57.18</td></tr><tr><td>NV-Embed-v2 (7B)</td><td>27.18</td><td>17.80</td><td>34.34</td><td>24.57</td><td>61.13</td><td>62.50</td><td>30.76</td><td>21.19</td><td>61.82</td></tr><tr><td>Qwen3-Embed-8B</td><td>24.19</td><td>15.60</td><td>25.79</td><td>17.95</td><td>65.50</td><td>61.36</td><td>24.99</td><td>16.78</td><td>63.43</td></tr><tr><td rowspan="2">Enhanced RAG</td><td>RAPTOR</td><td>27.84</td><td>17.80</td><td>26.33</td><td>19.65</td><td>57.21</td><td>57.95</td><td>27.09</td><td>18.73</td><td>57.58</td></tr><tr><td>HippoRAGv2</td><td>23.12</td><td>15.20</td><td>24.45</td><td>17.09</td><td>60.26</td><td>56.81</td><td>23.79</td><td>16.15</td><td>58.54</td></tr><tr><td rowspan="4">Multi-step RAG</td><td>Self-RAG</td><td>19.60</td><td>6.40</td><td>12.84</td><td>4.27</td><td>59.83</td><td>52.27</td><td>16.22</td><td>5.34</td><td>56.05</td></tr><tr><td>MemoRAG</td><td>23.29</td><td>15.20</td><td>19.40</td><td>11.64</td><td>55.89</td><td>51.13</td><td>21.35</td><td>13.42</td><td>53.51</td></tr><tr><td>RAPTOR+IRCoT</td><td>31.35</td><td>16.00</td><td>32.09</td><td>19.36</td><td>63.76</td><td>64.77</td><td>31.72</td><td>17.68</td><td>64.27</td></tr><tr><td>HippoRAGv2+IRCoT</td><td>28.98</td><td>13.00</td><td>29.27</td><td>18.24</td><td>64.19</td><td>62.50</td><td>29.13</td><td>15.62</td><td>63.35</td></tr><tr><td></td><td>ComoRAG (Ours)</td><td>31.43</td><td>18.60</td><td>34.52</td><td>25.07</td><td>72.93</td><td>68.18</td><td>32.98</td><td>21.84</td><td>70.56</td></tr></table>

• EN.QA from ∞BENCH (Zhang et al. 2024): a QA dataset with 351 questions on classic novels, with average context length over 200k tokens.   
• EN.MC from ∞BENCH: a MC dataset with 229 questions on classic novels of similar length as EN.QA.   
• DetectiveQA (Xu et al. 2024b): a MC dataset consisting of detective fiction with average length over 100k tokens. We randomly sample $20 \%$ of all stories to reduce the computational cost.

For evaluation metrics, we report both F1 and Exact Match (EM) scores for QA datasets, and report Accuracy (ACC) for MC datasets. To ensure fairness in resolving multiple-choice questions, we only expose the options during Try-Answer such that no retrieval-related actions can utilize potential hints present in the options.

Baselines We employ four types of baselines as follows, covering different paradigms for long context QA.

• LLM: the non-RAG setting, where the entire context (capped by length 128k) is provided to the LLM directly.   
• Naive RAG: the standard RAG setting that splits the raw context by chunks for retrieval. We set the max chunk length as 512 tokens in all experiments.   
• Enhanced RAG: RAG methods with augmented retrieval index, including RAPTOR (Sarthi et al. 2024) that constructs a semantic summary tree over text chunks, and HippoRAGv2 (Gutierrez et al. ´ 2025) that builds the knowledge base for entities in text chunks. We also experimented with GraphRAG (Edge et al. 2025); however, it requires exponential computational cost for building the retrieval index, being less practical for full evaluation. We separately report GraphRAG on a subset in Appendix B.   
• Multi-step RAG: RAG methods with multi-step or iterative retrieval strategies. IRCoT (Trivedi et al. 2023) leverages Chain-of-Thought (CoT) as intermediate queries that iteratively retrieve evidence. Self-RAG (Asai et al. 2024) trains a dedicated critic model to control when to stop retrieval. MemoRAG (Qian et al. 2025) trains a model that

compresses the global context, which generates clues as intermediate queries.

Implementation Details For the Hierarchical Knowledge Source, we follow the procedures of HippoRAGv2 and RAPTOR respectively to build the Veridical and Semantic layers; the Episodic layer employs an adaptive sliding window for narrative summaries described in Appendix B.

For LLMs, our main experiments adopt GPT-4o-mini in all approaches to ensure fair comparison. We additionally tested GPT-4.1 and Qwen3-32B (Yang et al. 2025) for generalization analysis in Section 4.3. For all RAG methods, we adopt the popular model BGE-M3 (Chen et al. 2024) for retrieval. Additionally, for naive RAG, we also experiment with larger but less practical embedding models, including NV-Embed-v2 (Lee et al. 2025) and Qwen3-Embed-8B (Zhang et al. 2025). The LLM context length for all RAG methods, including ComoRAG, is capped at 6k tokens.

For the Metacognitive Regulation loop, we set the framework to iterate for a maximum of 5 rounds. More regarding implementation details are provided in Appendix B.

# 4 Experimental Results

# 4.1 Main Results

Evaluation results of our main experiments are shown in Table 1. Remarkably, ComoRAG achieves the best performance upon all baselines across all datasets. Despite using the lightweight 0.3B BGE-M3 for retrieval, it significantly outperforms RAG with much larger 8B embedding models. Overall, ComoRAG demonstrates consistent improvement for tackling long narrative comprehension, surpassing strong prior RAG methods of various paradigms.

Upon closer examination, ComoRAG exhibits distinct advantages on the two ∞BENCH datasets featuring ultra-long contexts. More broadly, Figure 3 illustrates that ComoRAG is more robust and insensitive to longer contexts, sustaining its efficacy over HippoRAGv2, with the accuracy gap peaking at $+ 2 4 . 6 \%$ for documents exceeding 150k tokens, which

Table 2: Ablation studies of ComoRAG.   

<table><tr><td rowspan="2">Method</td><td>EN.MC</td><td colspan="2">EN.QA</td></tr><tr><td>ACC</td><td>F1</td><td>EM</td></tr><tr><td>ComoRAG</td><td>72.93</td><td>34.52</td><td>25.07</td></tr><tr><td>Baselines</td><td></td><td></td><td></td></tr><tr><td>HippoRAGv2</td><td>60.26</td><td>24.45</td><td>17.09</td></tr><tr><td>RAPTOR</td><td>57.21</td><td>26.33</td><td>19.65</td></tr><tr><td>Index</td><td></td><td></td><td></td></tr><tr><td>w/o Veridical</td><td>51.97</td><td>22.24</td><td>15.88</td></tr><tr><td>w/o Semantic</td><td>64.63</td><td>30.82</td><td>22.65</td></tr><tr><td>w/o Episodic</td><td>64.63</td><td>31.48</td><td>21.47</td></tr><tr><td>Retrieval</td><td></td><td></td><td></td></tr><tr><td>w/o Metacognition</td><td>62.01</td><td>26.95</td><td>18.53</td></tr><tr><td>w/o Regulation</td><td>55.02</td><td>27.95</td><td>20.59</td></tr><tr><td>w/o Both</td><td>54.15</td><td>25.64</td><td>17.35</td></tr></table>

highlights the importance of stateful multi-step reasoning for query resolution over long and coherent contexts.

# 4.2 Ablation Studies

We perform ablation studies on EN.MC and EN.QA datasets by systematically removing key modules in ComoRAG. The results are shown in Table 2.

Hierarchical Knowledge Source All three knowledge layers contribute supplementary enhancements to the final performance, with the Veridical Layer being the most significant retrieval index. It provides the basis for factualgrounded reasoning, as confirmed by the $30 \%$ relative performance drop upon its removal.

Metacognition Removing the Metacognition process essentially disables the memory workspace, where all agents operate on retrieved evidence directly, without knowledge consolidation by synthesized cues. Disabling this module leads to a significant performance drop, as seen by the $22 \%$ relative decrease in F1 score on EN.QA, and an approximate $15 \%$ decrease in accuracy on EN.MC, underscoring the critical role of dynamic memory organization.

Regulation Removing the Regulation process cuts off the goal-oriented guidance, such that each cycle uses the same initial query for new evidence retrieval (duplicated evidence is removed), without generating probing queries that are crucial to new evidence acquisition. Disabling this module severely impacts retrieval efficiency, causing a $24 \%$ drop in accuracy on EN.MC and a $19 \%$ drop in F1 score on EN.QA.

Notably, removing both Metacognition and Regulation further degrades performance, effectively reducing the system to a one-shot resolver without multi-step reasoning. Overall, the ablation study results corroborate that the enhancement offered by ComoRAG stems from the synergy between its memory consolidation and dynamic evidence exploration, facilitated by the hierarchical knowledge index to provide enriched semantic information. Removing any of the core components would significantly weaken its narrative reasoning capabilities.

![](images/bbd0e0db804f63853daf6e039b35f90a79c661220d25e5c658e0671efdd22081.jpg)  
Figure 3: Averaged accuracy across different document lengths on Multi-Choice datasets. ComoRAG is shown more robust to long contexts over the baseline.

![](images/c8b9ad8fb2d1888f7615ac1ed0e448c42c5969ca687dee06d273821e7246a7e9.jpg)  
Figure 4: Performance gains from iterative probing. GPT-4.1 marks the evaluation by using the stronger GPT-4.1 as LLM agents in ComoRAG (as opposed to GPT-4o-mini).

# 4.3 In-Depth Analysis of Iterative Retrieval

To further investigate the source of ComoRAG’s effectiveness, this section presents a quantitative analysis of its core iterative retrieval process.

Source of Gains: From Static Bottleneck to Dynamic Reasoning Our analysis suggests that the stateful multistep reasoning enabled by the Metacognitive loop is the key factor driving the observed improvement.

We first identify a “static bottleneck”: after the initial retrieval using the original query at step 0, the single-step evaluation score shows no significant advantage over strong baselines, with less than $1 \%$ compared to the best baseline HippoRAGv2+IRCoT. However, upon activating the cognitive loop, there presents a sustained and significant improvement, raising the accuracy to $7 2 . 9 3 \%$ on EN.MC, as shown in Figure 4. This further supports the findings from the ablation studies, which demonstrate a significant performance drop upon removing the entire loop. Additionally, Figure 4 illustrates that the majority of the improvement occurs within 2-3 cycles, confirming the efficiency of the process. The few remaining unresolved queries are tied to the inherent reasoning limitation of the base LLM, where our next analysis shows that the ceiling performance of ComoRAG can be lifted by switching to more capable LLMs.

Model-agnostic Generalization ComoRAG demonstrates generalization with different LLM backbones, with stronger LLMs further enhancing the reasoning process and final query resolution. To validate this, we replace GPT-4omini with GPT-4.1 and Qwen3-32B in the Metacognitive loop, using the same knowledge source for retrieval. The results, presented in Figure 4 and the upper section of Table 3, show a notable improvement particularly with GPT-4.1, boosting the F1 score on EN.QA from 34.52 to

Table 3: Efficacy of ComoRAG on model-agnostic generalization and Plug-and-Play flexibility.   

<table><tr><td>Method</td><td>NarQA</td><td>EN.QA</td><td>EN.MC</td><td>DetQA</td></tr><tr><td></td><td>F1</td><td>F1</td><td>ACC</td><td>ACC</td></tr><tr><td>ComoRAG</td><td>31.43</td><td>34.52</td><td>72.93</td><td>68.18</td></tr><tr><td>w/ Qwen3-32B</td><td>32.17</td><td>35.29</td><td>74.24</td><td>69.32</td></tr><tr><td>w/ GPT-4.1</td><td>35.43</td><td>38.82</td><td>78.17</td><td>76.14</td></tr><tr><td>HippoRAGv2</td><td>23.12</td><td>24.45</td><td>60.26</td><td>56.81</td></tr><tr><td>+ Our Loop</td><td>29.12</td><td>31.76</td><td>68.56</td><td>63.64</td></tr><tr><td>RAPTOR</td><td>27.84</td><td>26.33</td><td>57.21</td><td>57.95</td></tr><tr><td>+ Our Loop</td><td>30.55</td><td>34.31</td><td>69.00</td><td>62.50</td></tr></table>

38.82, and increases the accuracy on EN.MC from 72.93 to 78.17. These results demonstrate that ComoRAG effectively leverages and unleashes the model’s capabilities during its stateful iterative reasoning process.

Plug-and-Play: Flexibility To examine the modularity of our framework, we conduct further experiments by applying the Metacognitive loop of ComoRAG on existing RAG methods. As shown in the bottom section of Table 3, the cognitive loop can be seamlessly integrated with different RAG index including HippoRAGv2 and RAPTOR. This integration consistently results in significant performance improvements across all benchmarks, with accuracy on EN.MC increasing by over $8 \%$ for HippoRAGv2 and nearly $12 \%$ for RAPTOR (a similar trend is observed on EN.QA). These results demonstrate that ComoRAG could serve as a robust and flexible plug-and-play solution to enhance query resolution of existing RAG methods.

# 4.4 In-Depth Analysis of Query Resolution

To deepen the understanding of narrative query resolution, we roughly categorize all questions in our experimented datasets into three query types: factoid, narrative, and inferential, described as follows (details in Appendix C).

• Factoid Queries: queries answerable by a single, specific piece of information, often knowledge-seeking, e.g., “What religion is Octavio Amber?”   
• Narrative Queries: queries that require an understanding of plot progression as a coherent background context, e.g., “Where does Trace choose to live at the end of the novel?”   
• Inferential Queries: queries demanding reasoning beyond the literal text to understand implicit motivations, e.g., “What is the main reason that Nils first visits Aiden in his apartment?”

To systematically investigate the dynamics of ComoRAG reasoning, we first pose the question: what is the bottleneck in long-narrative reasoning for existing RAG methods? Figure 5 pictures a clear diagnosis. While one-shot retrieval suffices for factoid queries, which account for over $60 \%$ of initial solution, our iterative cognitive loop is essential for resolving complex narrative queries involving global context comprehension and deeper reasoning. These constitute nearly $50 \%$ of the problems that are solved exclusively through the Metacognitive loop.

![](images/566e0cd34084a40ca45b9f4257eed07f1b588d3ac06ef23cc3aa739ce8517e76.jpg)  
Figure 5: Distribution of solved query types.

![](images/d5d70d7d587e561291c0e6827af35e462fe31734bd89fbaebb60e0c637e557ea.jpg)  
Figure 6: Benchmarking RAG methods across query types.

This leads to the second question: how does our framework’s performance on this specific bottleneck compared to strong baselines? Figure 6 demonstrates that our method’s advantage is the most pronounced precisely in this area. On narrative queries, ComoRAG substantially outperforms the strongest baselines, achieving a $19 \%$ relative F1 improvement on EN.QA and a $16 \%$ accuracy gain on EN.MC. By addressing these queries, ComoRAG is demonstrated a targeted and effective solution for narrative queries that have posed challenges to the conventional RAG methodology.

Qualitatively, Figure 2 illustrates the dynamic reasoning mechanism with the query $q _ { i n i t }$ : “Mrs. MacIntyre never writes letters, so what is the sudden purpose of buying ink?” A standard, single-step retrieval would fail on this query, as it would only find a vague clue about a “cut out newspaper”, which is insufficient to form an answer. In contrast, ComoRAG initiates an iterative reasoning process by dynamically probing new queries and corresponding evidence for full resolution, constructing a complete evidence chain to deduce the final answer: Mrs. McGinty recognized a photo, wanted to sell the story, and intended to write to the newspaper. The full reasoning details of ComoRAG on this query are further provided in Appendix E.

# 5 Conclusion

In this work, we propose ComoRAG for long narrative reasoning, aiming to address the “stateless” limitation of conventional RAG. ComoRAG is especially inspired by the human brain’s Prefrontal Cortex: through a dynamic memory workspace and iterative probes, it fuses fragmented evidence into a coherent context to achieve stateful reasoning over narrative progression. Experiments validate that ComoRAG overcomes the bottleneck of existing methods by excelling at complex narrative and inferential queries, marking a paradigm shift from information retrieval to cognitive reasoning towards deeper long context comprehension.

# References

Asai, A.; Wu, Z.; Wang, Y.; Sil, A.; and Hajishirzi, H. 2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In The Twelfth International Conference on Learning Representations.   
Chen, J.; Xiao, S.; Zhang, P.; Luo, K.; Lian, D.; and Liu, Z. 2024. M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. In Ku, L.-W.; Martins, A.; and Srikumar, V., eds., Findings of the Association for Computational Linguistics: ACL 2024, 2318–2335. Bangkok, Thailand: Association for Computational Linguistics.   
Dobbins, I. G.; and Han, S. 2006. Cue- versus Probedependent Prefrontal Cortex Activity during Contextual Remembering. Journal of Cognitive Neuroscience, 18(9): 1439–1452.   
Edge, D.; Trinh, H.; Cheng, N.; Bradley, J.; Chao, A.; Mody, A.; Truitt, S.; Metropolitansky, D.; Ness, R. O.; and Larson, J. 2025. From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130.   
Eisenschlos, J. M.; Yogatama, D.; and Al-Rfou, R. 2023. Needle In A Haystack: Where Is It? Finding Factual Associations in Long Texts. arXiv preprint arXiv:2307.09288.   
Fernandez-Duque, D.; Baird, J. A.; and Posner, M. I. 2000. Executive Attention and Metacognitive Regulation. Consciousness and Cognition, 9(2): 288–307.   
Gutierrez, B. J.; Shu, Y.; Qi, W.; Zhou, S.; and Su, Y. 2025. ´ From RAG to Memory: Non-Parametric Continual Learning for Large Language Models. In Forty-second International Conference on Machine Learning.   
Jimenez Gutierrez, B.; Shu, Y.; Gu, Y.; Yasunaga, M.; and Su, Y. 2024. Hipporag: Neurobiologically inspired longterm memory for large language models. Advances in Neural Information Processing Systems, 37: 59532–59569.   
Johnson-Laird, P. N. 1983. Mental Models: Towards a Cognitive Science of Language, Inference, and Consciousness. Cambridge, MA: Harvard University Press.   
Kocisky, T.; Schwarz, J.; Blunsom, P.; Dyer, C.; Hermann, ´ K. M.; Melis, G.; and Grefenstette, E. 2017. The NarrativeQA Reading Comprehension Challenge. Transactions of the Association for Computational Linguistics, 6: 317–328.   
Lee, C.; Roy, R.; Xu, M.; Raiman, J.; Shoeybi, M.; Catanzaro, B.; and Ping, W. 2025. NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models. In The Thirteenth International Conference on Learning Representations.   
Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.; Goyal, N.; Kuttler, H.; Lewis, M.; Yih, W.-t.; Rockt ¨ aschel, ¨ T.; Riedel, S.; and Kiela, D. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Larochelle, H.; Ranzato, M.; Hadsell, R.; Balcan, M.; and Lin, H., eds., Advances in Neural Information Processing Systems, volume 33, 9459–9474. Curran Associates, Inc.   
Liu, N. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua, M.; Petroni, F.; and Liang, P. 2024. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics, 12: 157–173.

Miller, J. A.; and Constantinidis, C. 2024. Timescales of learning in prefrontal cortex. Nature Reviews Neuroscience, 25(9): 597–610.   
Qian, H.; Liu, Z.; Zhang, P.; Mao, K.; Lian, D.; Dou, Z.; and Huang, T. 2025. Memorag: Boosting long context processing with global memory-enhanced retrieval augmentation. In Proceedings of the ACM on Web Conference 2025, 2366– 2377.   
Sarthi, P.; Abdullah, S.; Tuli, A.; Khanna, S.; Goldie, A.; and Manning, C. 2024. RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval. In International Conference on Learning Representations (ICLR).   
Trivedi, H.; Balasubramanian, N.; Khot, T.; and Sabharwal, A. 2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In Rogers, A.; Boyd-Graber, J.; and Okazaki, N., eds., Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 10014–10037. Toronto, Canada: Association for Computational Linguistics.   
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; brian ichter; Xia, F.; Chi, E. H.; Le, Q. V.; and Zhou, D. 2022. Chain of Thought Prompting Elicits Reasoning in Large Language Models. In Oh, A. H.; Agarwal, A.; Belgrave, D.; and Cho, K., eds., Advances in Neural Information Processing Systems.   
Wu, J.; Li, J.; Li, Y.; Liu, L.; Xu, L.; Li, J.; Yeung, D.-Y.; Zhou, J.; and Yu, M. 2025. SitEmb-v1.5: Improved Context-Aware Dense Retrieval for Semantic Association and Long Story Comprehension. arXiv:2508.01959.   
Xu, L.; Li, J.; Yu, M.; and Zhou, J. 2024a. Fine-Grained Modeling of Narrative Context: A Coherence Perspective via Retrospective Questions. In Ku, L.-W.; Martins, A.; and Srikumar, V., eds., Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 5822–5838. Bangkok, Thailand: Association for Computational Linguistics.   
Xu, Z.; Ye, J.; Liu, X.; Sun, T.; Liu, X.; Guo, Q.; Li, L.; Liu, Q.; Huang, X.; and Qiu, X. 2024b. DetectiveQA: Evaluating Long-Context Reasoning on Detective Novels. ArXiv, abs/2409.02465.   
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.   
Yang, Z.; Qi, P.; Zhang, S.; Bengio, Y.; Cohen, W.; Salakhutdinov, R.; and Manning, C. D. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Riloff, E.; Chiang, D.; Hockenmaier, J.; and Tsujii, J., eds., Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2369–2380. Brussels, Belgium: Association for Computational Linguistics.   
Zhang, X.; Chen, Y.; Hu, S.; Xu, Z.; Chen, J.; Hao, M.; Han, X.; Thai, Z.; Wang, S.; Liu, Z.; et al. 2024. $\infty$ Bench: Extending long context evaluation beyond 100k tokens. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 15262–15277.

Zhang, Y.; Li, M.; Long, D.; Zhang, X.; Lin, H.; Yang, B.; Xie, P.; Yang, A.; Liu, D.; Lin, J.; Huang, F.; and Zhou, J. 2025. Qwen3 Embedding: Advancing Text Embedding and Reranking Through Foundation Models. arXiv:2506.05176.

# A ComoRAG Algorithm

Algorithm 1: ComoRAG (Described in Section 2)   
Require: Initial Query qinit, Knowledge Source X, Max Iterations T  
Ensure: The final answer O or a failure signal  
1: function COMORAG(qinit, X, T)  
2: $\mathcal{M}_{pool}^{(0)}, \mathcal{P}_{hist}^{(0)}, \{\mathcal{C}\}^{(0)} \leftarrow \emptyset, \emptyset, \emptyset \triangleright$ Initialize Memory Pool, Probing History, and Synthesized Cues  
3: $\mathcal{E}^{(0)} \leftarrow \boxed{\text{Tri-Retrieve}} (\{q_{init}, \mathcal{X}\})$ 4: $O^{(0)} \leftarrow \boxed{\text{Try-Answer}} (q_{init}, \mathcal{E}^{(0)})$ 5: if $O^{(0)} \neq$ FailureSignal then  
6: return $O^{(0)} \triangleright$ Return immediately if successful  
7: end if $\triangleright$ Triggered only if initial attempt fails  
8: $\mathcal{M}_{encode}^{(0)} \leftarrow \boxed{\text{Mem-Encode}} (q_{init}, \mathcal{P}_{hist}^{(0)}, \mathcal{E}^{(0)})$ 9: $\mathcal{M}_{pool}^{(0)} \leftarrow \boxed{\text{Mem-Update}} (\mathcal{M}_{pool}^{(0)}, \mathcal{M}_{encode}^{(0)})$ 10: $\mathcal{P}_{hist}^{(0)} \leftarrow q_{init}$ 11: $\{\mathcal{C}\}^{(0)} \leftarrow \mathcal{M}_{pool}^{(0)}$ 12: for $t = 1, \ldots, T$ do  
13: $\mathcal{P}^{(t)} \leftarrow \boxed{\text{Self-Probe}} (q_{init}, \mathcal{P}_{hist}^{(t-1)}, \{\mathcal{C}\}^{(t-1)})$ 14: $\mathcal{E}^{(t)} \leftarrow \boxed{\text{Tri-Retrieve}} (\mathcal{P}^{(t)}, \mathcal{X})$ 15: $\mathcal{M}_{encode}^{(t)} \leftarrow \boxed{\text{Mem-Encode}} (q_{init}, \mathcal{P}^{(t)}, \mathcal{E}^{(t)})$ 16: $\mathcal{C}_{fuse}^{(t)} \leftarrow \boxed{\text{Mem-Fuse}} (q_{init}, \mathcal{M}_{pool}^{(t-1)} \circ q_{init})$ 17: $O^{(t)} \leftarrow \boxed{\text{Try-Answer}} (q_{init}, \mathcal{M}_{encode}^{(t)}, \mathcal{C}_{fuse}^{(t)})$ 18: if $O^{(t)} \neq$ FailureSignal then return $O^{(t)}$ 19: end if  
20: $\mathcal{M}_{pool}^{(t)} \leftarrow \boxed{\text{Mem-Update}} (\mathcal{M}_{pool}^{(t-1)}, \mathcal{M}_{encode}^{(t)})$ 21: $\mathcal{P}_{hist}^{(t)} \leftarrow \mathcal{P}_{hist}^{(t-1)} \cup \mathcal{P}^{(t)}$ 22: $\{\mathcal{C}\}^{(t)} \leftarrow \mathcal{M}_{pool}^{(t)}$ 23: end for  
24: return FailureSignal  
25: end function

# B Implementation Details

# B.1 Veridical Layer

As described in Section 2.2, ComoRAG empowers Large Language Models by constructing a hierarchical knowledge source, whereby the Veridical Layer is a foundational component comprising text chunks of the original context. We largely follow the construction process of HippoRAGv2 (Gutierrez et al. ´ 2025) to add a mapping between knowledge graphs (KGs) and text chunks to facilitate retrieval. To construct the KG, a Large Language Model (LLM) is leveraged to extract (subject-predicate-object) knowledge triples. These triples from a document are then aggregated to form a

unified knowledge graph. Finally, a retrieval-optimized encoder adds supplementary edges to this graph by identifying and linking semantically similar entities (synonyms). The retrieval of the Veridical Layer thus follows HippoRAGv2 to utilize KGs towards more accurate retrieval. Statistics for this layer are detailed in Table 4.

Table 4: Statistics of the Veridical Layer across Datasets.   

<table><tr><td>Layer</td><td>Count</td><td>NarQA</td><td>EN.QA</td><td>EN.MC</td><td>DetQA</td></tr><tr><td rowspan="3">Veridical</td><td># ofChunks</td><td>4446</td><td>26465</td><td>47074</td><td>2406</td></tr><tr><td># of_Entities</td><td>33810</td><td>292170</td><td>401040</td><td>30969</td></tr><tr><td># of_Triplets</td><td>51012</td><td>372339</td><td>576595</td><td>33696</td></tr></table>

# B.2 Episodic Layer

To construct the Episodic Layer, a sequence of text chunks is summarized. Since the context lengths can vary significantly, the choice of a sliding window size for this summarization presents a trade-off: a large window can be too coarse for short narratives, while a small window may be inefficient and fail to capture long-range dependencies in longform content. Therefore, we dynamically adjust the window size W according to the total number of text chunks, $N$ , in the document. The specific heuristic is as follows.

• For short to medium-length narratives $( N \mathrm { ~  ~ { ~ \leq ~ } ~ } 2 0 0$ chunks): stepped window sizes (3, 5, 8, and 10) are used for documents up to 20, 50, 100, and 200 chunks respectively, aiming to preserve details for shorter contexts.   
• For long narratives $ { N }  { \mathrm { ~ > ~ } } 2 0 0 )$ : A logarithmic scaling function is applied to prevent the window from becoming excessively large. This sub-linear growth is intended to increase the summary scope for massive texts more slowly. The window size is calculated as follows to keep the window size between 10 to 20:

$$
W = \min  (2 0, \max  (1 0, \lfloor \log_ {2} (N) \times 2 \rfloor))
$$

For each window, the contained text chunks are concatenated and provided to an LLM agent (GPT-4o-mini in our experiments). The agent is instructed to generate a concise summary that maintains chronological order and identifies key events and causal relationships. The resulting summaries are then collected and sorted by their original window order to form the nodes of the Episodic Layer.

# B.3 GraphRAG Experiments

GraphRAG is a structured-augmented RAG method similar to HippoRAGv2, which involves the construction of a comprehensive knowledge graph from source documents, which is then used to identify interconnected information for retrieval. However, its formulation requires heavy computation for building the retrieval index that includes multi-level node relations and summaries.

We conducted preliminary experiments on a data subset to evaluate its viability. The results, detailed in Table 5, demonstrated that GraphRAG not only had significantly higher token consumption, but also attained lower scores compared to

other baselines adopted in our experiments. Considering the trade-offs between its computational cost and performance, we ultimately did not include GraphRAG as a primary baseline for a full-scale evaluation.

Table 5: Comparison of Performance, Token Usage, and Average Time for ComoRAG and GraphRAG.   

<table><tr><td></td><td>ComoRAG</td><td>GraphRAG</td></tr><tr><td colspan="3">Performance Metrics</td></tr><tr><td>F1 Score</td><td>33.61 (100.0%)</td><td>14.20 (42.3%)</td></tr><tr><td>EM Score</td><td>21.43 (100.0%)</td><td>8.00 (37.3%)</td></tr><tr><td colspan="3">Token Usage</td></tr><tr><td>Tokens</td><td>5.90M (100.0%)</td><td>27.12M (459.7%)</td></tr><tr><td colspan="3">Average Time Taken (sec)</td></tr><tr><td>Index</td><td>291 (100.0%)</td><td>1936 (665.3%)</td></tr><tr><td>Retrieve</td><td>25 (100.0%)</td><td>29 (116.0%)</td></tr></table>

# B.4 Hyperparameters for ComoRAG

The key hyperparameters for our ComoRAG framework are detailed in Table 6. All cognitive agents employ GPT-4omini, with retrieval powered by the widely-used BGE-M3 embedding model. For retrieval settings, The dynamic cognitive loop is configured to run for a maximum of 5 iterations, generating up to 3 new probing queries per cycle. The context for QA is capped at 6k tokens, in consistent with all RAG baselines in our experiments. This context is assembled via a proportional 8:2:2:1 allocation of evidence from the Veridical, Semantic, Episodic, and fused Historical memory, respectively. The “Mem-Fuse Threshold” is set to 0.5, indicating the proportion of evidences retrieved from the memory pool that are forwarded to the Integration Agent for memory fusion and summary generation.

Table 6: Hyperparameter settings for ComoRAG in our experiments. V, S, E, H refer to Veridical, Semantic, Episodic, and Historical evidence.   

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>LLM Agents (πprobe, etc.)</td><td>GPT-4o-mini</td></tr><tr><td>Retrieval Model</td><td>BGE-M3</td></tr><tr><td>Chunk Size</td><td>512 tokens</td></tr><tr><td>Context Length</td><td>6,000 tokens</td></tr><tr><td>Random Seed</td><td>0</td></tr><tr><td>Max Iterations</td><td>5</td></tr><tr><td>Max Probing Queries</td><td>3</td></tr><tr><td>Context Construction</td><td>Proportional Allocation (8:2:2:1 ratio for V:S:E:H)</td></tr><tr><td>Mem-Fuse Threshold</td><td>0.5</td></tr></table>

# C Query Types for Narratives

To facilitate a fine-grained analysis of our model’s performance, we (authors of this work) manually annotated the types of all questions in the EN.QA and EN.MC datasets.

Table 7: Distribution of query types across the two datasets.   

<table><tr><td>Dataset</td><td>Factoid</td><td>Narrative</td><td>Inferential</td><td>Total</td></tr><tr><td>EN.QA</td><td>224</td><td>84</td><td>43</td><td>351</td></tr><tr><td>EN.MC</td><td>132</td><td>46</td><td>51</td><td>229</td></tr></table>

Each question is classified into one of the three categories based on the cognitive processes required to answer it, described in Section 4.4:

• Factoid: questions answerable by locating a single, specific piece of information from the text.   
• Narrative: questions that demand an understanding of plot progression, requiring the aggregation of information from multiple text parts.   
• Inferential: questions that necessitate reasoning beyond the literal text to understand implicit motivations or causal links.

The final distribution of the annotated query types is presented in Table 7.

# Self-Probe Instruction Template for Probing Query Generation in Regulation Agent

# Role:

You are an expert in multi-turn retrieval-oriented probe generation. Your job is to extract diverse and complementary retrieval probes from queries to broaden and enrich subsequent corpus search results.

# Input Materials:

• Original Query: A question or information need that requires comprehensive information retrieval.   
• Context: Available background information, partial content, or relevant summaries.   
• Previous Probes: Previously generated probes from earlier iterations (if any).

# Task:

Based on the query and context, generate up to 3 non-overlapping retrieval probes that explore the query from distinct angles.

# Critical Requirements:

• Semantic Differentiation: Ensure new probes are semantically distinct from any previous probes provided.   
• Complementary Coverage: New probes should cover different information dimensions not addressed by previous probes.   
• Relevance Maintenance: All probes must remain directly relevant to answering the original query.

# Each probe should:

• Target different information dimensions relevant to the query type:

– Character-related: actions, motivations, relationships, timeline, consequences   
– Event-related: participants, causes, sequence, location, outcomes   
– Object-related: description, origin, usage, significance, connections   
– Location-related: events occurred, people involved, time periods, significance

• Expand search scope beyond obvious keywords to capture related content.   
• Avoid semantic overlap with previous probes while maintaining query relevance.   
• Be formulated as effective search terms or phrases.

# Probe Generation Strategy:

• When previous probes exist:

1. Analyze Previous Coverage: Identify what semantic domains/angles have been covered.   
2. Gap Identification: Find unexplored but relevant information dimensions.   
3. Alternative Angles: Generate probes from different conceptual perspectives.   
4. Semantic Distance: Ensure sufficient semantic distance from previous probes.

• When no previous probes exist:

– Probe 1: Direct elements explicitly mentioned in the query.   
– Probe 2: Contextual elements that might contain the answer.   
– Probe 3: Related concepts or alternative formulations.

# Output Format:

```snap
```
json
{
    "probel": "Content of probe 1",
    ...
} 
```

# Notes:

• For simple queries, you may generate only 1–2 probes.   
• If previous probes have covered most relevant angles, generate fewer new probes to avoid redundancy.   
• Prioritize quality and semantic distinctiveness over quantity.

# Mem-Encode Instruction Template for Synthesized Cue Generation in Comprehension Agent

# Role

You are an expert narrative analyst capable of identifying, extracting, and analyzing key information from narrative texts to provide accurate and targeted answers to specific questions.

# Material

You are given the following:

1. A final objective to be resolved   
2. A specific question that needs to be answered   
3. Content: Direct excerpts, facts, and specific information from the narrative text

# Task

1. Carefully analyze the question to identify:

• What type of information is being asked (character actions, locations, objects, events, motivations, etc.)   
• Which narrative elements are relevant to answering it   
• The specific details that need to be extracted

2. Systematically scan the content for:

• Direct mentions of relevant elements (names, places, objects, events)   
• Contextual probes that help answer the question   
• Temporal and spatial relationships   
• Cause-and-effect connections

3. Analyze the identified information considering:

• Explicit statements (directly stated facts)   
• Implicit information (suggested through context, dialogue, or narrative)   
• Logical connections between different narrative elements   
• Chronological sequence of events if relevant

4. Synthesize findings to construct a precise answer to the question.

# Response Format

Provide a structured analysis with up to 5 key findings:

Key Finding: <Most directly relevant information answering the question>

Key Finding: <Supporting evidence or context>

Key Finding: <Additional relevant details>

Key Finding: $<$ <Clarifying information if needed>

Key Finding: <Resolution of any ambiguities>

# Mem-Fuse Instruction Template for Cue Generation in Integration Agent

# Role:

You are an expert narrative synthesis specialist who excels at integrating and analyzing information from multiple narrative sources to create coherent and comprehensive insights.

# Input Material:

• Previous Analysis: Results from earlier memory fusion operations that contain analyzed narrative information.   
• Current Query: A question or information request that needs to be addressed.

# Task:

1. Review and understand the previous memory fusion outputs:

• Identify key narrative elements and their relationships.

• Note any established facts, character developments, or plot points.   
• Recognize patterns and connections across different analyses.

# 2. Analyze the current query in context:

• Determine how it relates to previously established information.   
• Identify any new aspects or angles that need to be addressed.   
• Consider how previous insights can inform the current response.

# 3. Synthesize the information:

• Integrate relevant previous findings with new analysis.   
• Create a coherent narrative that addresses the current query.   
• Ensure continuity and consistency with previous analyses.   
• Highlight any new insights or developments.

# 4. Provide a comprehensive response that:

• Directly answers the current query.   
• Incorporates relevant previous context.   
• Maintains narrative coherence.   
• Offers clear and insightful analysis.

# Response Format:

Provide a cohesive narrative response that integrates previous insights with new analysis to address the current query. Focus on creating a flowing, well-structured response.

Try-Answer

Prompt Template for Query Resolution in QA Agent

# Role:

You are an expert on reading and understanding books and articles.

# Task:

Given the following detailed article, semantic summary, Episodic summary from a book, and a related question with different options, you need to analyze which option is the best answer for the question.

# Inputs:

• Detail Article: {context}   
• Summary by Semantic: {semantic summary}   
• Summary by Episodic: {Episodic summary}   
• History Info: {history info}   
• Question: {question}

# Limits:

• Do not infer. Respond only based on the provided content strictly.   
• Pick the choice only if you find at least 2 places that support the answer.

# Response Format:

1. Content Understanding:

Start with a brief summary of the content in no more than three sentences. Begin this section with ### Content Understanding.

2. Question Analysis:

Based on the question, analyze and list all relevant items using a markdown list. Begin this section with ### Question Analyse.

3. Options Analysis:

Extract the key points related to 4 options, also using a markdown list. Begin this section with ### Options Analyse.

Note: Only analyze based on the provided materials, do not make guesses.

4. Final Answer:

Provide your final answer with a heading. Begin this section with ### Final Answer, followed by the best option in the format of [A] or [B] or [C] or [D]. If you cannot answer, give a failure signal: $\star$ .

# E Case Study on Narrative Reasoning

Table 8: Case Study on Narrative Reasoning. We present a case to demonstrate our model’s performance in long-context understanding, showing the final round of the Metacognitive Control Loop. Different colors are used to highlight the nature of the processed information: Blue is used for the key evidence that contributes to the correct answer, while Orange is used for the key cues.   

<table><tr><td>Input Data (No Options)
Query: Mrs. MacIntyre never writes letters, so what is the sudden purpose of buying ink?
Options: [A] Response to the government&#x27;s letter [B] Sending a birthday card to my niece [C] Write a letter to the Sunday Comet newspaper. [D] Blurring Photos</td></tr><tr><td>ComoRAG&#x27;s Choice Result</td></tr><tr><td>Memory Pool M(0) pool:
- A page of the newspaper had a portion cut out...</td></tr><tr><td>Step1
Probes P(1):
- What did Mrs. McGinty recognize in the clipping, and how did she try to profit from it?
...
Retrieved Passages:
...The narrative offers insight into Miss Pamela Hosford&#x27;s role at the Sunday Comet, as she casually recalls receiving a vague letter from a woman about a photograph but fails to retrieve it...
Cues C(1):
- Key Finding:Mrs. McGinty usually had Joe help her reply to letters.;
- Key Finding:Mrs. McGinty may have sent a letter to the newspaper;...</td></tr><tr><td>Memory Pool M(1) pool:
- A page of the newspaper had a portion cut out...
- Mrs. MacIntyre sent a letter to the Sunday Comet...</td></tr><tr><td>Step2
Probes P(2):
- What did Miss Pamela Hosford recall about a letter sent to the Sunday Comet, and what might it imply about Mrs. McGinty?
...
Retrieved assages:
...Miss Pamela Hosford&#x27;s role at the Sunday Comet, as she casually recalls receiving a vague letter from a woman about a photograph but fails to retrieve it...She liked eavesdropping, knew who was in the photo, and asked what we&#x27;d pay...
Cues C(2):
- Key Finding:Mrs. McGinty wanted to make some money from the photo;...</td></tr><tr><td>Chosen: C.(Correct)
(C) Write a letter to the Sunday Comet newspaper: Strong textual probes support this option. Mrs. McGinty cut out a part of the newspaper, recognized someone in a photo, asked about payment, and unusually bought ink—suggesting she intended to write to the paper. Final Answer: [C]</td></tr></table>
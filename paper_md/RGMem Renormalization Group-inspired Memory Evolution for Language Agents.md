# RGMem: Renormalization Group–inspired Memory Evolution for Language Agents

Ao Tian * 1 Yunfeng Lu * 1 Xinxin Fan 2 Changhao Wang 1 Lanzhi Zhou 1 Yeyao Zhang 3 Yanfang Liu 4

# Abstract

Personalized and continuous interactions are critical for LLM-based conversational agents, yet finite context windows and static parametric memory hinder the modeling of long-term, crosssession user states. Existing approaches, including retrieval-augmented generation and explicit memory systems, primarily operate at the fact level, making it difficult to distill stable preferences and deep user traits from evolving and potentially conflicting dialogues.To address this challenge, we propose RGMem, a self-evolving memory framework inspired by the renormalization group (RG) perspective on multi-scale organization and emergence. RGMem models long-term conversational memory as a multi-scale evolutionary process: episodic interactions are transformed into semantic facts and user insights, which are then progressively integrated through hierarchical coarse-graining, thresholded updates, and rescaling into a dynamically evolving user profile.By explicitly separating fast-changing evidence from slow-varying traits and enabling nonlinear, phase-transition-like dynamics, RGMem enables robust personalization beyond flat retrieval or static summarization. Extensive experiments on the LOCOMO and PersonaMem benchmarks demonstrate that RGMem consistently outperforms SOTA memory systems, achieving stronger cross-session continuity and improved adaptation to evolving user preferences. Code is available at https://github.com/fenhg297/RGMem

1Beihang University, China 2Institute of Computing Technology, Chinese Academy of Science, China 3LightSail, China 4State Key Laboratory of Complex & Critical Software Environment, Beihang University, China. Correspondence to: Ao Tian <tianao@buaa.edu.cn>, Yunfeng Lu <lyf@buaa.edu.cn>, Xinxin Fan <fanxinxin@ict.ac.cn>, Changhao Wang <wch@buaa.edu.cn>, Lanzhi Zhou <piki@buaa.edu.cn>, Yeyao Zhang <zhangyeyao@guangfan.ai>.

Preprint. February 3, 2026.

# 1. Introduction

Modern dialogue agents built on large language models are increasingly expected to sustain personalized interactions over extended periods, spanning multiple sessions and evolving user states (Li et al., 2025b; Chhikara et al., 2025). However, long-term personalization exposes a fundamental mismatch: interaction histories grow without bound, while the model’s reasoning at any moment is constrained by a finite context window (Collins et al., 2015; Xiao et al., 2024). This mismatch gives rise to a core challenge in long-term dialog personalization—how to maintain stable user representations across sessions while remaining responsive to new and potentially contradictory evidence. Existing memory and retrieval mechanisms struggle to achieve this balance. Limited context windows and the lost-in-the-middle effect (Liu et al., 2023; Zhong et al., 2024) make long-dialog reasoning unreliable, while static parametric memory is difficult to update incrementally (Wang & Chen, 2025). Although retrieval-augmented generation (RAG) (Edge et al., 2024; Guo et al., 2024) and explicit memory systems attempt to address this problem at the fact level, they are often dominated by lexical overlap and recency bias, making it difficult to abstract stable, higher-level traits across sessions. More fundamentally, dialog personalization is inherently multi-scale: it involves concrete events (micro), cross-situational regularities (meso), and long-term abstractions (macro), giving rise to the classic stability–plasticity dilemma.

From this perspective, abstraction is a selective, scaledependent transformation that is triggered by sufficient evidence and applied to preserve both stability and adaptability.

Yet despite recent progress, existing memory systems still lack a principled account of how user profiles should evolve under continuous and often conflicting evidence.To reason about constrained memory dynamics, we draw inspiration from the RG perspective (Ma, 1973; Parisen Toldin, 2022; Tu, 2023), which studies how stable macroscopic structure emerges through iterative coarse-graining and rescaling. Rather than treating RG as a physical model, we adopt it as an engineering lens for organizing and evolving long-term conversational memory across abstraction scales.

Motivation: Resorting to the theory of renormalization

![](images/88e577b829ddb7505109a680835c96e5009524bd89bffcd702eff526939cf7b7.jpg)

![](images/18acdd86aa3810b30559399c7577366cc1e049c694b03ec4d6fb4fe35e37c82b.jpg)

![](images/6e5febfa43e7fdf0d953e30ffdc3b7f4afc1cb90a1a2ff43f4c7acedb3955cdc.jpg)  
Figure 1. A comparative illustration of memory models encountering the multi-scale Challenge.

An illustration of how different memory models respond to the same multi-scale dialogue scenario. Left: Flat Memory. Standard memory systems retrieve fragmented information driven by keyword matching and recency, often missing deeper, long-term user intent. Right: RGMem. RGMem progressively integrates fine-grained interactions into stable, higher-level user representations while preserving important contextual conflicts, enabling balanced and proactive responses under complex, cross-scale user needs.

group, we reveals four key insights towards the predicament of long-term conversational memory for language agents:

Insight 1. Effective Information Density is Maximized via Hierarchical Coarse-Graining.

Insight 2. User Profile Updates Exhibit Phase-Transition-Like Dynamics.

Insight 3. Separating Slow and Fast Variables Resolves the Stability–Plasticity Dilemma.

Insight 4. Long-Term User Profiles Exhibit Macroscopic Invariance Beyond Fact-Level Stability.

To convince these insights, we also provide formal analytics from theoretical viewpoint in Appendix B. Motivated by these insights, we propose a novel self-evolving memory framework RGMem, it operationalizes hierarchical coarsegraining and thresholded profile evolution, multi-facet experiments demonstrate our approach can significantly improve the performance by 7.08 points on LOCOMO and 8.98 points on PersonaMem than the best baseline.

2. Related Work

2.1. Dialog Personalization and Implicit Memory

Implicit memory approaches encode user information directly into model parameters through fine-tuning or parameter-efficient adaptation (Wang et al., 2024; Tan et al.,

2025b; Wei et al., 2025; Hu et al., 2022; Yu et al., 2025). These methods excel at capturing stylistic patterns and soft preferences without explicit retrieval, but require parameter updates, limiting interoperability with closed-source models and cross-agent profile sharing. Moreover, latent or KVcache-based memories lack fine-grained auditability, controlled updates, and rollback mechanisms, making them unsuitable for traceable, evolving user profiles. Consequently, implicit memory systems struggle to support long-term personalization under continuous and conflicting evidence.

2.2. Explicit Memory and Retrieval-Augmented Generation

Explicit memory systems externalize user information for persistent storage and retrieval during generation (Rasmussen et al., 2025; Chhikara et al., 2025). These approaches improve traceability and online updates, but typically operate at the fact or paragraph level, emphasizing retrieval quality rather than memory organization or evolution. Without principled mechanisms for abstraction, contradiction management, or cross-scale integration, such systems often accumulate noise and inconsistencies over time (Pan et al., 2025; Tan et al., 2025a). In contrast to flat retrieval paradigms, RGMem models memory as a multiscale evolving system, explicitly separating factual evidence from higher-level, slowly-varying user traits.(see Fig. 1)

# 2.3. Hierarchical Memory and Profile Evolution

To manage long interaction horizons, recent research has increasingly adopted hierarchical memory architectures that decouple fine-grained observations from high-level abstractions. Representative approaches include recursive summarization trees (Rezazadeh et al., 2024) and pyramidal indices (Hu et al., 2025) that organize context at varying granularities. More complex formulations utilize layered knowledge graphs to segregate episodic traces from semantic concepts, as seen in GraphRAG (Edge et al., 2024), HippoRAG (Jimenez Gutierrez et al., 2024; Gutierrez et al.´ , 2025), and AriGraph (Anokhin et al., 2024). While these structures enhance retrieval density, their evolution mechanisms remain static or linear: abstraction typically occurs via uniform bottom-up propagation (Li et al., 2025a; Wu et al., 2025) or fixed-interval consolidation. Lacking explicit scale-dependent control over when structural reorganization should occur, these systems struggle to resolve the stability– plasticity dilemma, often over-fitting to transient noise or failing to consolidate genuine profile shifts. In contrast, RGMem models memory as a multi-scale dynamical system, where abstraction is governed by thresholded, non-linear phase transitions rather than uniform aggregation policies.

# 3. Methodology

# 3.1. Preliminaries and Motivation

The RG is a theoretical framework originally developed in statistical physics to explain how stable macroscopic structure emerges from microscopic interactions across scales (Wilson, 1983; Pelissetto & Vicari, 2002; Litim, 2001; Zinn-Justin, 2007; Yakhot & Orszag, 1986). Abstracted from its physical origins, RG can be viewed as a general principle for organizing multi-scale information: high-frequency, task-irrelevant details are progressively integrated out through coarse-graining, while scale-invariant, task-relevant structure is preserved along a renormalization flow. Similar RG-inspired ideas have appeared in machine learning, including connections to hierarchical abstraction in deep networks (Chen et al., 2018; Di Sante et al., 2022; Beny ´ , 2013; Wang & Liu, 2025), the Information Bottleneck principle (Tishby & Zaslavsky, 2015; Mehta & Schwab, 2014), and hierarchical pooling or graph coarsening methods (Ying et al., 2018).

However, existing RG-inspired approaches mainly address static representations, whereas long-term conversational memory requires modeling how representations evolve across abstraction scales. This gives rise to a fundamental stability–plasticity dilemma: a memory system must integrate transient, fine-grained interactions while preserving stable macroscopic user traits over time. Concretely, this entails balancing internal consistency, fidelity to new

evidence, and representational simplicity.

To make this intuition explicit, we introduce an abstract design objective, which we refer to as an effective Hamiltonian, to qualitatively characterize the desirability of a given user-profiling theory $\tau$ :

$$
\mathcal {H} (\mathcal {T}) = \alpha E _ {\text {c o n}} (\mathcal {T}) + \beta E _ {\text {f i d}} (\mathcal {T} \mid \mathcal {D} _ {L 0}) + \gamma E _ {\text {c o m}} (\mathcal {T}) \tag {1}
$$

where $E _ { \mathrm { { c o n } } }$ penalizes internal contradictions within the profile, $E _ { \mathrm { f i d } }$ measures deviations from the accumulated microscopic evidence $\mathcal { D } _ { L 0 }$ , and $E _ { \mathrm { { c o m } } }$ discourages overly redundant or fragmented representations.

Importantly, this Hamiltonian is not meant to be explicitly optimized. Instead, it serves as a conceptual objective guiding the design of memory evolution mechanisms, since direct optimization over high-dimensional textual memory states is intractable.

Accordingly, RGMem adopts a renormalization-inspired perspective in which user profiles evolve through coarsegrained, scale-specific transformations, heuristically approximating the minimization of $\mathcal { H } ( \mathcal { T } )$ across abstraction levels. Guided by this perspective, RGMem is designed as a multiscale memory system that (i) performs early coarse-graining over raw dialogue to suppress noise, (ii) updates user profiles incrementally through scale-aware transformations rather than repeated re-summarization, and (iii) allows stable representations to reorganize only when accumulated evidence crosses critical thresholds. Based on these principles, we formalize RGMem as a structured user-profiling model and present its algorithmic instantiation below.(see Fig. 2)

# 3.2. User-Profiling Modeling

Inspired by RG intuitions, we model long-term user profiling as a multi-scale representation that evolves from finegrained conversational evidence to stable abstract traits, with explicit separation of abstraction levels and scale-dependent update mechanisms. Concretely, RGMem consists of a memory state space, a scale-dependent profile representation, and a set of transformation operators that drive profile evolution.

Memory State Space. The complete memory state is defined as

$$
\mathcal {M} = \mathcal {D} _ {L 0} \times \mathcal {G}, \tag {2}
$$

where $\mathcal { D } _ { L 0 } = \{ d _ { 1 } , d _ { 2 } , . . . \}$ denotes discrete episodic memory units extracted from raw dialogues, and $\mathcal { G } = ( \nu , \mathcal { E } )$ is a dynamic knowledge graph with hierarchical structure . This separation distinguishes rapidly changing factual evidence from more slowly evolving relational and conceptual structures, providing the foundation for balancing stability and plasticity.

Multi-Scale Effective Theory. We define an effective the-

![](images/b21b9cbbdd34847118b15643295b9e22a6534879bfeb47ee1e6f01c28bc41bab.jpg)  
Figure 2. Overview of the RGMem framework. RGMem consists of three stages: (1) constructing a multi-scale memory state from raw dialogue, (2) evolving user profiles through scale-aware renormalization operators, and (3) performing multi-scale retrieval to generate coherent responses.

ory $\mathcal { T } ( \mathcal { M } , s )$ as a compact, scale-dependent representation of the user profile, parameterized by descriptors $\{ \lambda _ { i } ( s ) \}$ . Lower scales $( s \ : = \ : 0$ ) correspond to individual episodic evidence, while higher scales $s \geq 1 ,$ ) capture aggregated patterns and abstract user traits derived from structured knowledge.

RG Transformation. Profile evolution across scales is modeled by a transformation operator $\mathcal { R }$ :

$$
\mathcal {T} ^ {(s + 1)} = \mathcal {R} \left(\mathcal {T} ^ {(s)}\right). \tag {3}
$$

This operator jointly performs coarse-graining and rescaling, mapping lower-level representations into abstract profile summaries. $\mathcal { R }$ is an algorithmic update rule rather than an explicit optimization procedure, governing how profiles evolve under incoming evidence.

# 3.3. Renormalization Flow: Operators and Instantiation

RGMem instantiates the multi-scale formulation in Section 3.1 through a three-layer architecture (L0–L2). L0 constructs microscopic evidence, L1 drives multi-scale memory evolution, and L2 enables scale-aware retrieval. By separating fast-changing signals from slowly evolving user traits, this design supports stable yet adaptive memory evolution. The following subsections describe each layer in detail.

# 3.3.1. CONSTRUCTION OF MEMORY STATE SPACE

RGMem constructs its memory state space via coordinated processing in the L0 and L1 layers.

Microscopic Evidence Space $\mathcal { D } _ { L 0 }$ . Raw dialogue streams $D _ { \mathrm { r a w } }$ are transformed into a set of microscopic memory units $\mathcal { D } _ { L 0 }$ through an initial coarse-graining pipeline $f _ { c g } =$ $f _ { s y n t h } \circ f _ { s e g }$ . Each episodic unit is mapped to a structured state

$$
d = \left(\lambda_ {\text {f a c t}}, \Lambda_ {\text {c o n c}}\right), \tag {4}
$$

where $\lambda _ { f a c t }$ denotes an objective event-level fact and $\Lambda _ { c o n c }$ is a set of user-related conclusions. The conclusion set is partitioned as $\Lambda _ { c o n c } = \Lambda _ { b a s e } \cup \Lambda _ { r e l } ,$ , separating directly grounded interpretations from high-salience signals used for higher-level abstraction.

Structured Knowledge Space $\mathcal { G }$ . A hierarchical extraction function

$$
f _ {\text {e x t r a c t}}: \mathcal {D} _ {L 0} \rightarrow \mathcal {G} \tag {5}
$$

organizes microscopic evidence into a dynamic knowledge graph $\mathcal { G } = ( \nu , \mathcal { E } )$ . The node set $\nu$ is divided into three levels: abstract concepts $\gamma _ { a b s }$ , representing high-level and slowly evolving user concepts; general events $\nu _ { g e n }$ , capturing recurring activities or patterns; and instance events $\nu _ { i n s t }$ , corresponding to concrete, context-specific interactions. Edges $\mathcal { E }$ encode relationships between nodes and

are partitioned into static classification relations ${ \mathcal { E } } _ { c l s }$ and dynamic event relations $\mathcal { E } _ { e v t }$ .

This separation provides a stable topological backbone for long-term abstractions while allowing event-driven relations to evolve incrementally. As a result, RGMem can propagate stable representations upward through the hierarchy while selectively updating event-level information in response to new evidence.

# 3.3.2. INSTANTIATION OF RG OPERATORS

The effective user profile in RGMem evolves through a set of explicit, scale-aware memory operators that incrementally integrate new evidence, refine abstractions, and propagate higher-level structure across the knowledge graph, enabling stable and interpretable profile formation under continuous and conflicting inputs.

Relation Inference Operator $\mathcal { R } _ { K 1 }$ . This operator performs incremental integration at the relation level, aggregating repeated or reinforcing evidence associated with the same semantic relation over time. Its primary role is to prevent uncontrolled accumulation of redundant facts while enabling stable, interpretable summaries of recurring user behaviors or states. Let T (1,t)e d ${ \mathcal T } _ { e } ^ { ( 1 , t ) }$ enote the mesoscopic theory associated with relation $e$ at time step $t$ . When sufficient new evidence has accumulated for $e$ , the operator updates the relation-level representation according to:

$$
\mathcal {T} _ {e} ^ {(1, t + 1)} \leftarrow \mathcal {T} _ {e} ^ {(1, t)} + \beta \left(\mathcal {T} _ {e} ^ {(1, t)}, D _ {e} ^ {\text {n e w}}\right), \tag {6}
$$

where $D _ { e } ^ { n e w }$ is the set of newly observed microscopic evidence linked to relation $e$ . The non-linear update function $\beta ( \cdot )$ , instantiated by a language model, integrates prior summaries with new evidence to produce an updated relation theory T (1,t+1)e . $\mathcal { T } _ { e } ^ { ( 1 , t + 1 ) }$

To avoid premature abstraction from sparse or noisy signals, $\mathcal { R } _ { K 1 }$ is triggered only when the accumulated evidence for a relation exceeds a threshold $\theta _ { \mathrm { i n f } }$ . This thresholded execution ensures that relation-level summaries emerge from consistent patterns rather than isolated observations, providing a stable intermediate representation for higher-level abstraction.

Node-Level Abstraction Operator $\mathcal { R } _ { K 2 }$ . While $\mathcal { R } _ { K 1 }$ consolidates repeated evidence at the relation level, long-term user profiling requires abstraction across multiple related behaviors and events. Operator $\mathcal { R } _ { K 2 }$ performs this higherorder integration by aggregating information associated with an abstract concept node, producing a compact representation of user tendencies within a semantic domain.

For an abstract node $v \in \mathcal { V } _ { a b s }$ , the operator consumes a mixed-scale input set

$$
\mathcal {I} _ {v} ^ {\text {n e w}} = \left\{\mathcal {T} _ {e _ {i}} ^ {(1), \text {n e w}} \right\} _ {e _ {i} \in N (v)} \cup \left\{d _ {j} ^ {\text {n e w}} \right\} _ {j \in D (v)}, \tag {7}
$$

The input includes recently updated relation-level summaries and newly observed microscopic evidence associated with $v$ . Rather than treating all inputs equally, $\mathcal { R } _ { K 2 }$ prioritizes aggregated representations to guide abstraction toward stable intermediate structures. Accordingly, $\mathcal { R } _ { K 2 }$ performs an RG-inspired coarse-graining step that integrates heterogeneous evidence into a compact concept-level representation, and is decomposed into two sequential sub-operations:

$$
\mathcal {R} _ {K 2} = \mathbb {S} \circ \mathbb {P}, \tag {8}
$$

corresponding to projection-selection and synthesisrescaling, respectively.

Projection–Selection $( \mathbb { P } )$ . Given the mixed-scale input set $\mathcal { T } _ { v } ^ { n e w }$ , the projection-selection step filters and prioritizes evidence according to its level of abstraction and information density. Relation-level theories that have already undergone aggregation are favored over raw microscopic observations, reflecting their higher semantic stability.

Formally, the filtered evidence set is defined as:

$$
D _ {v} ^ {\prime} = \mathbb {P} \left(\mathcal {I} _ {v} ^ {\text {n e w}}\right), \tag {9}
$$

where $D _ { v } ^ { \prime }$ contains a reduced subset of evidence that best represents collective behavioral signals associated with node $v$ .

Synthesis–Rescaling (S). The synthesis step integrates the filtered evidence $D _ { v } ^ { \prime }$ with the previous node-level representation to construct an updated concept-level theory:

$$
\left(\Sigma_ {v} ^ {(2, t + 1)}, \Delta_ {v} ^ {(2, t + 1)}\right) = \mathbb {S} \left(D _ {v} ^ {\prime}, \Sigma_ {v} ^ {(2, t)}, \Delta_ {v} ^ {(2, t)}\right). \tag {10}
$$

Order Parameter $\Sigma$ . The order parameter $\Sigma$ captures dominant, recurring patterns that persist across multiple situations. It represents the most stable abstraction of user behavior within a concept, prioritizing internal consistency and low representational complexity.

Correction Term $\Delta$ . The correction term $\Delta$ preserves salient but non-universal signals that cannot be absorbed into the dominant abstraction without loss of fidelity. It explicitly represents internal tension within the profile, allowing the model to acknowledge conflicting or transitional behaviors.

These components are computed via non-linear aggregation functions:

$$
\Sigma_ {v} ^ {(2, t + 1)} = \operatorname {A g g} _ {\text {c o m m o n}} \left(D _ {v} ^ {\prime}, \Sigma_ {v} ^ {(2, t)}\right), \tag {11}
$$

$$
\Delta_ {v} ^ {(2, t + 1)} = \operatorname {E x t r a c t} _ {\text {s a l i e n t}} \left(D _ {v} ^ {\prime}, \Delta_ {v} ^ {(2, t)}\right), \tag {12}
$$

where both functions are instantiated by language models.

To prevent premature abstraction, $\mathcal { R } _ { K 2 }$ is executed only when sufficient new evidence accumulates for a node, controlled by a threshold $\theta _ { \mathrm { s u m } }$ .

Hierarchical Flow Operator $\mathcal { R } _ { K 3 }$ . This operator captures the iterative nature of renormalization by propagating information upward along the static conceptual hierarchy $\mathcal { E } _ { c l s }$ . It operates on macroscopic representations associated with abstract nodes, progressively integrating child-level summaries into higher-level profiles. At each level, the operator aggregates two complementary components $( \Sigma , \Delta )$ .

Formally, for each parent node $v _ { p }$ , the transformation integrates representations from its child nodes $\{ v _ { c _ { i } } \}$ as:

$$
\left(\Sigma_ {v _ {p}} ^ {(s + 1)}, \Delta_ {v _ {p}} ^ {(s + 1)}\right) = \mathcal {R} _ {K 3} \left(\left\{\left(\Sigma_ {v _ {c _ {i}}} ^ {(s)}, \Delta_ {v _ {c _ {i}}} ^ {(s)}\right) \right\} _ {i}\right), \tag {13}
$$

where $\mathcal { R } _ { K 3 }$ performs a structured synergy–tension analysis to distill common patterns while retaining critical residual signals. The execution of this operator is scheduled via a dirty-flag propagation mechanism to ensure efficient and incremental updates.

# 3.4. Dynamics and Multi-Scale Observations

Emergent Dynamics: Stability and Structural Shifts When applied continually over long interaction streams, RGMem exhibits two characteristic dynamical behaviors that govern the evolution of user profiles.

First, under consistent and reinforcing evidence, higherlevel profile representations gradually stabilize. Updates induced by new interactions diminish over time, leading to a regime where macroscopic profile states remain effectively invariant. In practice, this corresponds to a stable long-term user profile that is robust to routine or redundant inputs.

Second, when accumulated evidence becomes sufficiently strong and inconsistent with the current profile, the system undergoes a qualitative reorganization. Instead of incremental adjustment, higher-level representations are restructured in a coordinated manner, resulting in a rapid shift of the user profile. These structural shifts are triggered by thresholded update mechanisms and reflect genuine changes in user preferences rather than transient noise.

Together, these two regimes enable RGMem to balance long-term stability with responsiveness to change. As we show empirically in Section 4, this behavior manifests as non-monotonic performance under varying update thresholds and allows RGMem to simultaneously achieve strong stability and plasticity.

Multi-Scale Observations RGMem exposes its internal memory state through a multi-scale retrieval mechanism implemented in the L2 layer. Given a query $q$ , the retrieval function $f _ { \mathrm { r e t r } } ( \boldsymbol { q } , \mathcal { M } )$ constructs a query-specific context by selectively accessing memory representations at different abstraction levels.

At the microscopic level, the retriever returns relevant episodic evidence to support fact-level queries. At higher

![](images/878258ef2eaea3fb29d8515be1feff755ab340ce047cad1084d4b60bd5001a1c.jpg)  
Figure 3. Accuracy peaks at an effective context length

levels, it retrieves aggregated relational or profile-level representations that summarize longer-term patterns. These components are combined into a unified context $C ( q )$ and provided to the language model for response generation.

This design allows RGMem to adapt the granularity of retrieved information to the query intent, enabling both precise factual recall and abstract reasoning over long-term user traits within a bounded context.

# 4. Experiment Evaluation

# 4.1. Experimental Setup

Benchmarks and Baselines. We evaluate RGMem on two long-term conversational memory benchmarks: LO-COMO (Maharana et al., 2024) and PersonaMem under the 128k-token setting (Jiang et al., 2025). LOCOMO focuses on long-context reasoning and temporal consistency, while PersonaMem evaluates dynamic persona evolution under conflicting evidence. Detailed benchmark descriptions, evaluation protocols, and implementation details are provided in Appendix C.1 and C.7.

# 4.2. Memory Efficiency and Information Density

We examine whether hierarchical coarse-graining improves the effective information density of long-term conversational memory under limited context budgets. Using the LOCOMO benchmark, we analyze how reasoning performance varies with the amount of retrieved context. Fig 3 shows a clear non-monotonic relationship between context length and accuracy. Performance improves as context increases from approximately 3k to $3 . 8 \mathrm { k }$ tokens, but saturates and then degrades as more context is added. This indicates that indiscriminate accumulation of historical information does not yield monotonic gains in reasoning quality. Instead, the results reveal an optimal context scale at which information is maximally useful. Below this scale, evidence is insufficient; beyond it, redundant or weakly relevant content

Table 1. Performance on the PersonaMem benchmark $( \% )$ . We report the improvement of our method compared to the second-best baseline in parentheses.   

<table><tr><td rowspan="2">Backbone Method</td><td colspan="3">MEMORY RECALL</td><td colspan="3">REASONING &amp; ADAPTATION</td><td colspan="2">TEMPORAL</td></tr><tr><td>Recall Facts</td><td>Suggest Ideas</td><td>Revisit Reasons</td><td>Track Evol.</td><td>Aligned Rec.</td><td>General. Scen.</td><td>Latest Pref.</td><td>Avg.</td></tr><tr><td colspan="9">Backbone: GPT-4o-mini</td></tr><tr><td>LLM (Vanilla)</td><td>55.34</td><td>12.33</td><td>70.02</td><td>58.21</td><td>42.33</td><td>33.05</td><td>34.31</td><td>39.21</td></tr><tr><td>LangMem</td><td>64.76</td><td>23.93</td><td>81.82</td><td>53.30</td><td>45.45</td><td>42.26</td><td>58.12</td><td>52.36</td></tr><tr><td>Mem0</td><td>69.57</td><td>19.53</td><td>78.75</td><td>56.72</td><td>61.82</td><td>52.19</td><td>68.25</td><td>56.79</td></tr><tr><td>A-Mem</td><td>59.78</td><td>9.22</td><td>80.19</td><td>53.30</td><td>44.32</td><td>45.42</td><td>53.25</td><td>49.17</td></tr><tr><td>Memory OS</td><td>72.59</td><td>22.62</td><td>84.42</td><td>69.11</td><td>67.82</td><td>35.72</td><td>64.71</td><td>54.23</td></tr><tr><td>RGMem</td><td>77.06 (+4.47)</td><td>26.47 (+2.54)</td><td>85.29 (+0.87)</td><td>67.82</td><td>73.66 (+5.84)</td><td>56.62 (+4.43)</td><td>75.47 (+7.22)</td><td>63.87 (+7.08)</td></tr><tr><td colspan="9">Backbone: GPT-4.1</td></tr><tr><td>LLM (Vanilla)</td><td>64.76</td><td>19.53</td><td>81.82</td><td>67.21</td><td>53.14</td><td>53.91</td><td>51.62</td><td>51.86</td></tr><tr><td>LangMem</td><td>77.82</td><td>24.27</td><td>82.16</td><td>54.33</td><td>42.33</td><td>63.22</td><td>70.59</td><td>58.23</td></tr><tr><td>Mem0</td><td>81.02</td><td>16.28</td><td>81.82</td><td>54.33</td><td>54.35</td><td>65.13</td><td>81.22</td><td>60.44</td></tr><tr><td>A-Mem</td><td>82.57</td><td>28.31</td><td>86.35</td><td>56.72</td><td>70.12</td><td>65.13</td><td>77.81</td><td>63.95</td></tr><tr><td>Memory OS</td><td>79.72</td><td>19.33</td><td>82.16</td><td>60.34</td><td>73.66</td><td>74.18</td><td>77.81</td><td>65.03</td></tr><tr><td>RGMem</td><td>88.64 (+6.07)</td><td>35.04 (+6.73)</td><td>87.03 (+0.68)</td><td>70.89 (+3.68)</td><td>83.02 (+9.36)</td><td>72.55</td><td>85.07 (+3.85)</td><td>74.01 (+8.98)</td></tr></table>

Table 2. LOCOMO benchmark results $( \% )$ . LLM-as-a-judge evaluation.   

<table><tr><td rowspan="2">Method</td><td colspan="4">QUESTION TYPE</td><td rowspan="2">Avg.</td></tr><tr><td>S-Hop</td><td>M-Hop</td><td>Open</td><td>Temp.</td></tr><tr><td colspan="6">gpt-4o-mini</td></tr><tr><td>RAG</td><td>35.05</td><td>30.31</td><td>43.52</td><td>27.58</td><td>38.10</td></tr><tr><td>LangMem</td><td>62.23</td><td>47.92</td><td>71.12</td><td>23.43</td><td>58.10</td></tr><tr><td>Mem0</td><td>67.13</td><td>51.15</td><td>72.93</td><td>55.51</td><td>66.88</td></tr><tr><td>Zep</td><td>74.11</td><td>66.04</td><td>67.71</td><td>79.76</td><td>75.14</td></tr><tr><td>RGMem</td><td>80.15</td><td>69.16</td><td>67.71</td><td>81.27</td><td>78.92</td></tr><tr><td>Full-Context</td><td>83.01</td><td>66.79</td><td>49.53</td><td>58.22</td><td>71.41</td></tr><tr><td colspan="6">gpt-4.1-mini</td></tr><tr><td>RAG</td><td>37.94</td><td>37.69</td><td>48.96</td><td>61.83</td><td>51.62</td></tr><tr><td>LangMem</td><td>74.47</td><td>61.06</td><td>67.71</td><td>86.92</td><td>78.05</td></tr><tr><td>Mem0</td><td>62.41</td><td>57.32</td><td>44.79</td><td>66.47</td><td>62.47</td></tr><tr><td>Zep</td><td>79.43</td><td>69.16</td><td>73.96</td><td>83.33</td><td>79.09</td></tr><tr><td>RGMem</td><td>89.58</td><td>78.03</td><td>72.86</td><td>88.91</td><td>86.17</td></tr><tr><td>Full-Context</td><td>88.53</td><td>77.70</td><td>71.88</td><td>92.70</td><td>87.52</td></tr></table>

obscures salient signals. RGMem operates near this regime by hierarchically coarse-graining dialogue history into compact, high-density representations rather than relying on flat retrieval or full-context accumulation. From a renormalization perspective, this behavior reflects systematic scale selection, where irrelevant microscopic fluctuations are integrated out through coarse-graining and rescaling, yielding an effective description that preserves task-relevant invariants. These results support Insight 1: effective information density is maximized through hierarchical abstraction, not

![](images/56523ee403c76ed661159290a0ea22925d67c69ab12a00bde7586392f1810d48.jpg)

![](images/6aa26ffe1afe189a58b497ae692fe92aefadeb176d7c19b5b0ac56ed63f89466.jpg)  
(b)   
Figure 4. Non-linear dependence on evolution threshold.

brute-force context expansion.

# 4.3. Evidence of Phase-Transition Dynamics

Standard memory systems typically exhibit monotonic behavior, where updates lead to either gradual improvement or instability. In contrast, the RG perspective predicts nonlinear dynamics governed by control parameters. We analyze RGMem’s behavior as a function of the evolution threshold $\theta _ { \mathrm { i n f } }$ , which acts as a control parameter, while task-level performance serves as an observable order parameter reflecting the macroscopic state of the user profile.

Fig 4 reveals a sharp, non-monotonic performance peak at $\theta _ { \mathrm { i n f } } = 3$ across both LOCOMO (Fig 4a) and PersonaMem (Fig 4b). Rather than a smooth optimum, this behavior indicates the presence of a critical point in the memory dynamics.

Two Regimes and a Critical Point. The system exhibits

two qualitatively distinct regimes: Subcritical Regime $( \theta _ { \mathbf { i n f } } < 3 )$ : The system is overly sensitive, where even transient noise triggers frequent updates, leading to Supercritical Regime $\ell _ { \mathrm { i n f } } > 3 )$ ): Updates are excessively suppressed, causing the profile to become rigid and unresponsive to genuine preference changes.

At the critical threshold $\theta _ { \mathrm { i n f } } = 3 )$ ), RGMem achieves an optimal balance. Below this point, new observations are absorbed as minor perturbations; once accumulated evidence crosses the threshold, the system undergoes a rapid and coordinated reorganization of its macroscopic state. This phase-transition-like behavior enables robustness to noise while remaining responsive to meaningful shifts.

Universality and Implications. Universality and Implications. Despite differences in tasks and metrics, both benchmarks exhibit the same critical threshold, indicating that this behavior reflects an intrinsic property of multi-scale memory dynamics rather than dataset-specific artifacts. Operating near this critical point enables RGMem to suppress fastvarying noise while preserving slow-varying, task-relevant structure, providing a principled resolution to the stability– plasticity dilemma. From a renormalization perspective, this corresponds to integrating out irrelevant fluctuations while retaining the macroscopic components that define long-term user state. These results support Insight 2: user profile evolution follows a non-linear, threshold-driven dynamic.

![](images/bfa774a91d6733bf4c2d26c8c91c32e483ccb76cbd479ec344d7ddce08a7f82b.jpg)  
Figure 5. Stability–plasticity trade-off on PersonaMem.

# 4.4. Stability–Plasticity Trade-off and Emergent Macroscopic Invariance

Overall accuracy alone does not reveal how memory systems balance long-term consistency with rapid adaptation. To explicitly examine this tension, we analyze the stability– plasticity trade-off by jointly visualizing performance on Recall Facts and Latest Preference in Fig 5.

As shown in the figure, most baseline methods lie on a clear Pareto frontier. Systems that aggressively update memory improve adaptability but sacrifice factual retention, while more conservative designs preserve historical information at the cost of responsiveness. In contrast, RGMem lies strictly

beyond this frontier, achieving superior performance on both dimensions simultaneously. This indicates that RGMem does not merely shift the trade-off curve, but effectively breaks the conventional stability–plasticity constraint.

This behavior arises from RGMem’s separation of memory across scales. Factual evidence is retained at lower levels to ensure stability, while higher-level abstractions evolve selectively based on aggregated evidence, enabling plasticity without overreacting to noise. By decoupling slow-varying traits from fast-changing observations, RGMem avoids forcing information to evolve at a single scale.

More broadly, these results suggest that robustness in longterm user modeling stems from the emergence of macroscopic invariants, rather than fact-level stability. Tasks involving multi-hop reasoning or cross-scenario generalization benefit from abstract behavioral regularities that remain stable across contexts. In RGMem, such regularities function as order parameters,while detailed facts, acting as correction terms, preserve fdelity without dominating inference. These fndings well-support Insight 3: separating slow and fast variables resolves the stability–plasticity dilemma and Insight 4: long-term iser profiles exhibit macroscopic invariance beyond fact-level stability.

# 4.5. Additional Analyses

Appendix experiments include ablations (Appendix C.4), parameter sensitivity analysis (Appendix C.6), and an analysis of macroscopic profile evolution (Appendix C.5). Ablation results show that removing any core component of the multiscale memory design consistently degrades performance, even when more context is retrieved. Parameter sensitivity analysis indicates that RGMem remains stable across a broad range of retrieval budgets and evolution thresholds, with a clear optimal regime. Finally, the macroscopic profile evolution analysis reveals that under consistent long-term evidence, profile representations rapidly converge and stabilize, exhibiting attractor-like behavior.

# 5. Conclusion

We have studies long-term conversational memory as a multi-scale dynamical system rather than a static retrieval problem. Our multi-facet experiments on the representative datasets LOCOMO and PersonaMem reveal that effective long-term user-profile modeling depends on hierarchical coarse-graining, thresholded non-linear updates, separation of slow and fast variables, and macroscopic invariance beyond fact-level stability. Motivated by these fundamental findings, we propose a novel self-evolving memory framework RGMem, enabling to instantiate scale-aware abstraction and control evolution through explicit memory structures. This work unveils that a robust long-term personal-

ization in language agents requires principled multi-scale organization, rather than larger context windows or fat memory accumulation.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Anokhin, P., Semenov, N., Sorokin, A., Evseev, D., Kravchenko, A., Burtsev, M., and Burnaev, E. Arigraph: Learning knowledge graph world models with episodic memory for llm agents. arXiv preprint arXiv:2407.04363, 2024.   
Beny, C. Deep learning and the renormalization group. ´ arXiv preprint arXiv:1301.3124, 2013.   
Chen, R. T., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. Neural ordinary differential equations. Advances in neural information processing systems, 31, 2018.   
Chhikara, P., Khant, D., Aryan, S., Singh, T., and Yadav, D. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.   
Collins, M., Lee, L., Roy, S., Vieira, T., and Roth, D. Transactions of the association for computational linguistics, volume 3. Transactions of the Association for Computational Linguistics, 3, 2015.   
Di Sante, D., Medvidovic, M., Toschi, A., Sangiovanni, G., ´ Franchini, C., Sengupta, A. M., and Millis, A. J. Deep learning the functional renormalization group. Physical review letters, 129(13):136402, 2022.   
Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R. O., and Larson, J. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.   
Guo, Z., Xia, L., Yu, Y., Ao, T., and Huang, C. Lightrag: Simple and fast retrieval-augmented generation. arXiv preprint arXiv:2410.05779, 2024.   
Gutierrez, B. J., Shu, Y., Qi, W., Zhou, S., and Su, Y. From ´ rag to memory: Non-parametric continual learning for large language models. arXiv preprint arXiv:2502.14802, 2025.   
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.   
Hu, M., Chen, T., Chen, Q., Mu, Y., Shao, W., and Luo, P. Hiagent: Hierarchical working memory management

for solving long-horizon agent tasks with large language model. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 32779–32798, 2025.   
Jiang, B., Hao, Z., Cho, Y.-M., Li, B., Yuan, Y., Chen, S., Ungar, L., Taylor, C. J., and Roth, D. Know me, respond to me: Benchmarking llms for dynamic user profiling and personalized responses at scale. arXiv preprint arXiv:2504.14225, 2025.   
Jimenez Gutierrez, B., Shu, Y., Gu, Y., Yasunaga, M., and Su, Y. Hipporag: Neurobiologically inspired long-term memory for large language models. Advances in Neural Information Processing Systems, 37:59532–59569, 2024.   
Li, R., Zhang, Z., Bo, X., Tian, Z., Chen, X., Dai, Q., Dong, Z., and Tang, R. Cam: A constructivist view of agentic memory for llm-based reading comprehension. arXiv preprint arXiv:2510.05520, 2025a.   
Li, Z., Song, S., Xi, C., Wang, H., Tang, C., Niu, S., Chen, D., Yang, J., Li, C., Yu, Q., et al. Memos: A memory os for ai system. arXiv preprint arXiv:2507.03724, 2025b.   
Litim, D. F. Optimized renormalization group flows. Physical Review D, 64(10):105007, 2001.   
Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172, 2023.   
Ma, S.-k. Introduction to the renormalization group. Reviews of Modern Physics, 45(4):589, 1973.   
Maharana, A., Lee, D.-H., Tulyakov, S., Bansal, M., Barbieri, F., and Fang, Y. Evaluating very long-term conversational memory of llm agents. arXiv preprint arXiv:2402.17753, 2024.   
Mehta, P. and Schwab, D. J. An exact mapping between the variational renormalization group and deep learning. arXiv preprint arXiv:1410.3831, 2014.   
Pan, Z., Wu, Q., Jiang, H., Luo, X., Cheng, H., Li, D., Yang, Y., Lin, C.-Y., Zhao, H. V., Qiu, L., et al. On memory construction and retrieval for personalized conversational agents. arXiv preprint arXiv:2502.05589, 2025.   
Parisen Toldin, F. Finite-size scaling at fixed renormalization-group invariant. Physical Review E, 105 (3):034137, 2022.   
Pelissetto, A. and Vicari, E. Critical phenomena and renormalization-group theory. Physics Reports, 368(6): 549–727, 2002.

Rasmussen, P., Paliychuk, P., Beauvais, T., Ryan, J., and Chalef, D. Zep: a temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025.   
Rezazadeh, A., Li, Z., Wei, W., and Bao, Y. From isolated conversations to hierarchical schemas: Dynamic tree memory representation for llms. arXiv preprint arXiv:2410.14052, 2024.   
Tan, H., Zhang, Z., Ma, C., Chen, X., Dai, Q., and Dong, Z. Membench: Towards more comprehensive evaluation on the memory of llm-based agents. arXiv preprint arXiv:2506.21605, 2025a.   
Tan, Z., Yan, J., Hsu, I., Han, R., Wang, Z., Le, L. T., Song, Y., Chen, Y., Palangi, H., Lee, G., et al. In prospect and retrospect: Reflective memory management for long-term personalized dialogue agents. arXiv preprint arXiv:2503.08026, 2025b.   
Tishby, N. and Zaslavsky, N. Deep learning and the information bottleneck principle. In 2015 ieee information theory workshop (itw), pp. 1–5. Ieee, 2015.   
Tu, Y. The renormalization group for non-equilibrium systems. nature physics, 19(11):1536–1538, 2023.   
Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.   
Wang, S. and Liu, Z. Enhancing the efficiency of variational autoregressive networks through renormalization group. Physical Review E, 112(3):035310, 2025.   
Wang, Y. and Chen, X. Mirix: Multi-agent memory system for llm-based agents. arXiv preprint arXiv:2507.07957, 2025.   
Wei, J., Ying, X., Gao, T., Bao, F., Tao, F., and Shang, J. Ai-native memory 2.0: Second me. arXiv preprint arXiv:2503.08102, 2025.   
Wilson, K. G. The renormalization group and critical phenomena. Reviews of Modern Physics, 55(3):583, 1983.   
Wu, Y., Zhang, Y., Liang, S., and Liu, Y. Sgmem: Sentence graph memory for long-term conversational agents. arXiv preprint arXiv:2509.21212, 2025.   
Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks, 2024. URL https://arxiv. org/abs/2309.17453, 1, 2024.   
Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang, Y. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.

Yakhot, V. and Orszag, S. A. Renormalization group analysis of turbulence. i. basic theory. Journal of scientific computing, 1(1):3–51, 1986.   
Ying, Z., You, J., Morris, C., Ren, X., Hamilton, W., and Leskovec, J. Hierarchical graph representation learning with differentiable pooling. Advances in neural information processing systems, 31, 2018.   
Yu, H., Chen, T., Feng, J., Chen, J., Dai, W., Yu, Q., Zhang, Y.-Q., Ma, W.-Y., Liu, J., Wang, M., et al. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259, 2025.   
Zhong, W., Guo, L., Gao, Q., Ye, H., and Wang, Y. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19724–19731, 2024.   
Zinn-Justin, J. Phase transitions and renormalization group. Oxford University Press, 2007.

# Appendices

Within this supplementary material, we elaborate on the following aspects:

• Appendix A: Algorithm   
• Appendix B: Theoretical Analysis   
• Appendix C: Additional Experimental Results   
• Appendix D: Prompt Templates

# A. Algorithm

Algorithm 1 RGMem: Continual Memory Evolution and Multi-Scale Retrieval   
1: Input: dialogue stream $D_{\mathrm{raw}} = \{u_1,\dots ,u_T\}$ query set $Q$ , thresholds $(\theta_{inf},\theta_{sum})$ 2: Output: final memory state $\mathcal{M} = (\mathcal{D}_{L0},\mathcal{G})$ answers $\{a_q\}_{q\in Q}$ 3: Initialize $\mathcal{D}_{L0}\gets \emptyset$ 4: Initialize $\mathcal{G}\gets (\mathcal{V}_{abs},\mathcal{V}_{gen},\mathcal{V}_{inst},\mathcal{E}_{cls},\mathcal{E}_{evt})$ with base abstract nodes   
5: // ContinuaRg evolution along the dialogue   
6: for $t = 1$ to $T$ do   
7: $S_{t}\gets f_{seg}(u_{t})$ 8: for all $s\in S_t$ do   
9: $d\gets f_{synth}(s)$ // $d = (\lambda_{fact},\Lambda_{conc})$ 10: $\mathcal{D}_{L0}\gets \mathcal{D}_{L0}\cup \{d\}$ 11: paths $\leftarrow$ EXTRACTENTITIES(d)   
12: triples $\leftarrow$ EXTRACT_RELATIONS(d,paths)   
13: $(\mathcal{V}_{abs},\mathcal{V}_{gen},\mathcal{V}_{inst},\mathcal{E}_{cls},\mathcal{E}_{evt})\leftarrow$ UPDATEGRAPH(Vabs,Vgen,Vinst,Ecls,Eevt,paths,triples)   
14: update local buffers $D_e^{new},I_v^{new}$ , and counters mentions(e), score(v)   
15: end for   
16: // Symmetric scheduling of RG operators on new evidence   
17: for all $e\in \mathcal{E}_{evt}$ with mentions(e) $\geq \theta_{inf}$ do   
18: $\mathcal{T}_e\gets \mathcal{R}_{K1}(\mathcal{T}_e,D_e^{new})$ 19: $D_e^{new}\gets \emptyset$ , mentions(e) $\leftarrow 0$ 20: end for   
21: for all $v\in \mathcal{V}_{abs}$ with score(v) $\geq \theta_{sum}$ do   
22: $(\Sigma_v,\Delta_v)\gets \mathcal{R}_{K2}((\Sigma_v,\Delta_v),\mathcal{T}_v^{new})$ 23: $\mathcal{T}_v^{new}\gets \emptyset$ , score(v) $\leftarrow 0$ 24: mark parent(v) as dirty   
25: end for   
26: for all dirty $v_p$ in topological order do   
27: $(\Sigma_{v_p},\Delta_{v_p})\gets \mathcal{R}_{K3}(\{\Sigma_{v_c},\Delta_{v_c}\} _{v_c\in child(v_p)})$ 28: mark $v_p$ as clean   
29: end for   
30: end for   
31: // Multi-scale retrieval for queries   
32: for all $q\in Q$ do   
33: entityQueries $\leftarrow$ QUERYSTRUCTURING(q)   
34: L0ctx $\leftarrow$ BM25_RETRIEVE(DL0,entityQueries)   
35: L1ctx $\leftarrow$ KG_RETRIEVE(G,entityQueries)   
36: C(q) $\leftarrow$ FORMAT(L0ctx,L1ctx)   
37: $a_q\gets \mathrm{LLM}(q,C(q))$ 38: end for   
39: return $\mathcal{M} = (\mathcal{D}_{L0},\mathcal{G}),\{a_q\}_{q\in Q}$

# B. Theoretical Analysis

Proposition B.1 (Coarse-grained majority summary increases information). Let $Z \in \{ - 1 , + 1 \}$ be a binary latent variable with $\mathbb { P } ( Z = + 1 ) = \mathbb { P } ( Z = - 1 ) = \frac { 1 } { 2 }$ . Conditioned on $Z$ , let $( X _ { i } ) _ { i = 1 } ^ { n }$ be i.i.d. observations in $\{ - 1 , + 1 \}$ such that, for some fixed noise level $\varepsilon \in ( 0 , \frac { 1 } { 2 } )$ ,

$$
\mathbb {P} (X _ {i} = Z \mid Z) = 1 - \varepsilon , \qquad \mathbb {P} (X _ {i} = - Z \mid Z) = \varepsilon .
$$

Assume n is odd and define the majority-vote summary

$$
S = \operatorname {s i g n} \left(\sum_ {i = 1} ^ {n} X _ {i}\right) \in \{- 1, + 1 \}.
$$

Then there exists an integer $n _ { 0 } = n _ { 0 } ( \varepsilon )$ such that for all odd $n \geq n _ { 0 }$ ,

$$
I (Z; S) > I (Z; X _ {1}),
$$

where $I ( \cdot ; \cdot )$ denotes mutual information. In particular, if we count $S$ and $X _ { 1 }$ as occupying the same token budget (one symbol each), the coarse-grained summary $S$ carries strictly more information about the latent trait $Z$ per token than any single microscopic observation $X _ { i }$ .

Proof. Step 1: The single-observation channel is a binary symmetric channel. By construction, the conditional distribution of $X _ { 1 }$ given $Z$ is

$$
\mathbb {P} (X _ {1} = Z \mid Z) = 1 - \varepsilon , \qquad \mathbb {P} (X _ {1} = - Z \mid Z) = \varepsilon ,
$$

with $\varepsilon \in ( 0 , \frac { 1 } { 2 } )$ . Since $Z$ is symmetric and the channel is symmetric, the pair $( Z , X _ { 1 } )$ forms a binary symmetric channel (BSC) with crossover probability $\varepsilon$ . It is standard (see, e.g., information theory textbooks) that the mutual information of a BSC with crossover probability $\delta \in [ 0 , \frac { 1 } { 2 } )$ under a uniform prior is

$$
I _ {\mathrm {B S C}} (\delta) = 1 - h _ {2} (\delta),
$$

where $h _ { 2 } ( \delta ) = - \delta \log _ { 2 } \delta - ( 1 - \delta ) \log _ { 2 } ( 1 - \delta )$ is the binary entropy function. Thus

$$
I (Z; X _ {1}) = 1 - h _ {2} (\varepsilon). \tag {14}
$$

Step 2: The majority-vote summary is also a binary symmetric channel. Fix $n$ odd. For each $i$ , define

$$
Y _ {i} = Z X _ {i} \in \{- 1, + 1 \}.
$$

Conditioned on $Z$ , the random variables $( Y _ { i } ) _ { i = 1 } ^ { n }$ are i.i.d., and

$$
\mathbb {P} (Y _ {i} = + 1 \mid Z) = \mathbb {P} (X _ {i} = Z \mid Z) = 1 - \varepsilon , \qquad \mathbb {P} (Y _ {i} = - 1 \mid Z) = \varepsilon .
$$

In particular, the distribution of $( Y _ { i } )$ does not depend on the sign of $Z$ , so $( Y _ { i } )$ is independent of $Z$ after marginalizing over $Z$ .

By definition of $S$ ,

$$
S = \mathrm {s i g n} \left(\sum_ {i = 1} ^ {n} X _ {i}\right) = \mathrm {s i g n} \left(Z \sum_ {i = 1} ^ {n} Y _ {i}\right) = Z \cdot \mathrm {s i g n} \left(\sum_ {i = 1} ^ {n} Y _ {i}\right),
$$

where we used that $Z ^ { 2 } = 1$ and $n$ is odd, so $\textstyle \sum _ { i = 1 } ^ { n } Y _ { i } \neq 0$ almost surely (i.e., ties occur with probability zero; a deterministic tie-breaking rule could be adopted without affecting the argument).

Define the deterministic function

$$
g (Y _ {1}, \ldots , Y _ {n}) = \mathrm {s i g n} \left(\sum_ {i = 1} ^ {n} Y _ {i}\right) \in \{- 1, + 1 \}.
$$

Then we can write

$$
S = Z \cdot g (Y _ {1}, \dots , Y _ {n}).
$$

Because $( Y _ { i } )$ is independent of $Z$ , the conditional distribution of $S$ given $Z$ depends only on whether $g ( Y _ { 1 } , \ldots , Y _ { n } )$ agrees with $+ 1$ or $- 1$ . In particular, we have

$$
\mathbb {P} (S = Z \mid Z) = \mathbb {P} \big (g (Y _ {1}, \ldots , Y _ {n}) = + 1 \big), \qquad \mathbb {P} (S = - Z \mid Z) = \mathbb {P} \big (g (Y _ {1}, \ldots , Y _ {n}) = - 1 \big).
$$

Define the effective crossover probability

$$
\varepsilon_ {n} := \mathbb {P} (S \neq Z) = \mathbb {P} \left(\sum_ {i = 1} ^ {n} Y _ {i} <   0\right).
$$

Again, by symmetry of $Z$ and the form $S = Z \cdot g ( Y _ { 1 } , \dots , Y _ { n } )$ , the pair $( Z , S )$ also forms a BSC, now with crossover probability $\varepsilon _ { n }$ . Hence

$$
I (Z; S) = 1 - h _ {2} \left(\varepsilon_ {n}\right). \tag {15}
$$

Step 3: Majority voting strictly reduces the effective noise for large $n$ . We now show that, for sufficiently large odd $n$ we have $\varepsilon _ { n } < \varepsilon$ . Recall that $Y _ { i } \in \{ - 1 , + 1 \}$ with

$$
\mathbb {P} (Y _ {i} = + 1) = 1 - \varepsilon , \qquad \mathbb {P} (Y _ {i} = - 1) = \varepsilon ,
$$

so

$$
\mathbb {E} [ Y _ {i} ] = (1 - \varepsilon) \cdot (+ 1) + \varepsilon \cdot (- 1) = 1 - 2 \varepsilon > 0,
$$

because $\varepsilon < \textstyle { \frac { 1 } { 2 } }$ . Let

$$
\bar {Y} _ {n} = \frac {1}{n} \sum_ {i = 1} ^ {n} Y _ {i}.
$$

Then $\begin{array} { r } { \varepsilon _ { n } = \mathbb { P } \big ( \sum _ { i = 1 } ^ { n } Y _ { i } < 0 \big ) = \mathbb { P } ( \bar { Y } _ { n } < 0 ) , } \end{array}$

Applying Hoeffding’s inequality for bounded i.i.d. random variables $Y _ { i } \in [ - 1 , 1 ]$ with mean $\mu = \mathbb { E } [ Y _ { i } ] = 1 - 2 \varepsilon$ , we obtain for any $t > 0$ ,

$$
\mathbb {P} (\bar {Y} _ {n} \leq \mu - t) \leq \exp (- 2 n t ^ {2}).
$$

Choosing $t = \mu = 1 - 2 \varepsilon$ yields

$$
\mathbb {P} (\bar {Y} _ {n} \leq 0) = \mathbb {P} (\bar {Y} _ {n} \leq \mu - (\mu - 0)) \leq \exp \big (- 2 n (1 - 2 \varepsilon) ^ {2} \big).
$$

Therefore

$$
\varepsilon_ {n} = \mathbb {P} \left(\bar {Y} _ {n} <   0\right) \leq \exp \left(- 2 n (1 - 2 \varepsilon) ^ {2}\right). \tag {16}
$$

The right-hand side tends to 0 exponentially fast as $n \to \infty$ , while $\varepsilon$ is fixed in $( 0 , { \frac { 1 } { 2 } } )$ . Hence there exists an integer $n _ { 0 } ( \varepsilon )$ such that for all $n \geq n _ { 0 } ( \varepsilon )$ ,

$$
\exp \left(- 2 n (1 - 2 \varepsilon) ^ {2}\right) <   \varepsilon ,
$$

and thus, by (16),

$$
\varepsilon_ {n} <   \varepsilon . \tag {17}
$$

Step 4: Mutual information comparison. The binary entropy function $h _ { 2 } ( \delta )$ is strictly increasing on $\delta \in [ 0 , \frac { 1 } { 2 } ]$ . From (17) we have $\begin{array} { l } { \varepsilon _ { n } < \varepsilon < \frac { 1 } { 2 } } \end{array}$ , so

$$
h _ {2} (\varepsilon_ {n}) <   h _ {2} (\varepsilon).
$$

Combining this inequality with (14) and (15), we obtain

$$
I (Z; S) = 1 - h _ {2} (\varepsilon_ {n}) > 1 - h _ {2} (\varepsilon) = I (Z; X _ {1}),
$$

for all odd $n \geq n _ { 0 } ( \varepsilon )$ , as claimed.

Finally, if we count $S$ and $X _ { 1 }$ each as occupying a single token in a context window, then the information-per-token about $Z$ carried by $S$ is strictly larger than that carried by any individual microscopic observation $X _ { i }$ . This formalizes, in a simplified probabilistic model, the intuition that hierarchical coarse-graining can increase the effective information density of memory under a fixed context budget. □

Proposition B.2 (Thresholded profile dynamics induce phase-transition–like updates). Let $( e _ { t } ) _ { t \geq 1 }$ be a sequence of i.i.d. random variables representing incoming evidence, with

$$
\mathbb {E} [ e _ {t} ] = \mu \neq 0, \qquad | e _ {t} | \leq E _ {\max } <   \infty \quad a. s.
$$

Fix a threshold $\theta > 0$ and an initial profile state $m _ { 0 } \in \{ - 1 , + 1 \}$ . Define the cumulative evidence process $( S _ { t } ) _ { t \geq 0 }$ and the profile process $( m _ { t } ) _ { t \ge 0 }$ by

$$
S _ {0} = 0, \quad S _ {t} = \sum_ {k = 1} ^ {t} e _ {k}, \quad t \geq 1, \tag {18}
$$

$$
m _ {t} = \left\{ \begin{array}{l l} m _ {t - 1}, & \text {i f} | S _ {t} | <   \theta , \\ \operatorname {s i g n} \left(S _ {t}\right), & \text {i f} | S _ {t} | \geq \theta , \end{array} \quad t \geq 1. \right. \tag {19}
$$

Then the following properties hold:

(i) (Piecewise-constant dynamics) There exists a (possibly finite) increasing sequence of random time indices

$$
0 = \tau_ {0} <   \tau_ {1} <   \tau_ {2} <   \dots
$$

such that $m _ { t }$ is constant on each half-open interval $[ \tau _ { k } , \tau _ { k + 1 } )$ , and $m _ { \tau _ { k } } \neq m _ { \tau _ { k } - 1 }$ for all $k \geq 1$ . In particular, every change in $m _ { t }$ occurs only at time steps where $| S _ { t } |$ crosses $\theta$ for the first time after the previous change.

(ii) (Finite number of phase transitions) With probability one, the number of profile changes is finite, i.e.,

$$
\mathbb {P} (\# \{t \geq 1: m _ {t} \neq m _ {t - 1} \} <   \infty) = 1.
$$

(iii) (Asymptotic alignment with the dominant evidence) If $\mu > 0$ , then with probability one, there exists a finite (random) time $T$ such that

$$
m _ {t} = + 1 \quad f o r a l l t \geq T.
$$

Similarly, if $\mu < 0$ , then with probability one, there exists a finite $T$ such that $m _ { t } = - 1$ for all $t \geq T$ .

In other words, the thresholded update rule induces a trajectory that is piecewise constant with abrupt jumps (“phase transitions”) triggered by critical crossings of the cumulative evidence, and the long-term profile state converges to the sign of the average evidence.

Proof. We prove each part in turn.

(i) Piecewise-constant dynamics. By construction, $m _ { t } \in \{ - 1 , + 1 \}$ for all $t \geq 0$ . Define the random set of change times

$$
\mathcal {C} := \{t \geq 1: m _ {t} \neq m _ {t - 1} \}.
$$

If $\mathcal { C }$ is empty, then $m _ { t } \equiv m _ { 0 }$ and the statement holds trivially with no phase transitions.

Otherwise, enumerate the elements of $\mathcal { C }$ in increasing order:

$$
\tau_ {1} <   \tau_ {2} <   \dots ,
$$

and set $\tau _ { 0 } : = 0$ . By definition of the update rule, for each $t \not \in { \mathcal { C } }$ we have $m _ { t } = m _ { t - 1 }$ , so on each interval $[ \tau _ { k } , \tau _ { k + 1 } )$ the process $m _ { t }$ is constant and equal to $m _ { \tau _ { k } }$ .

Moreover, if $t \in { \mathcal { C } }$ , then by the update rule we must have $| S _ { t } | \geq \theta$ , because a profile change occurs only under this condition. Between $\tau _ { k - 1 }$ and $\tau _ { k } - 1$ , no change occurs, so for all $t \in ( \tau _ { k - 1 } , \tau _ { k } )$ we have $| S _ { t } | < \theta$ . Thus each $\tau _ { k }$ is exactly the first time after $\tau _ { k - 1 }$ when $| S _ { t } |$ crosses the threshold $\theta$ and triggers a profile jump, as claimed.

(ii) Finite number of phase transitions. We first observe that a change in $m _ { t }$ can only occur when $| S _ { t } | \geq \theta$ , and that $m _ { t }$ always takes the value $\mathrm { s i g n } ( S _ { t } )$ at such times. Thus, after any change at time $\tau _ { k }$ , the new profile $m _ { \tau _ { k } }$ equals $\mathrm { s i g n } ( S _ { \tau _ { k } } )$ .

We now argue that almost surely $\lvert S _ { t } \rvert$ can cross the threshold $\theta$ only finitely many times while also alternating sign infinitely often in a way that causes infinitely many changes in $m _ { t }$ .

By the strong law of large numbers (SLLN),

$$
\frac {S _ {t}}{t} = \frac {1}{t} \sum_ {k = 1} ^ {t} e _ {k} \xrightarrow [ t \to \infty ]{\mathrm {a . s .}} \mu .
$$

Hence, if $\mu > 0$ , then almost surely there exists a (random) time $T _ { 1 }$ such that for all $t \geq T _ { 1 }$ we have

$$
\frac {S _ {t}}{t} > \frac {\mu}{2} > 0, \quad \text {a n d t h u s} S _ {t} > \frac {\mu}{2} t.
$$

This implies that beyond $T _ { 1 }$ , $S _ { t }$ is eventually strictly positive and grows at least linearly in $t$ . In particular, there exists a (possibly larger) random time $T _ { 2 }$ such that $| S _ { t } | = S _ { t } \ge \theta$ for all $t \geq T _ { 2 }$ , and $S _ { t }$ never returns to $[ - \theta , \theta ]$ for $t \geq T _ { 2 }$ .

From that time onward, the profile $m _ { t }$ is updated (if needed) to $+ 1$ when $| S _ { t } |$ first exceeds $\theta$ , and because $S _ { t }$ never re-enters the region $| S _ { t } | < \theta$ , no further changes are triggered afterwards. Thus, the number of change times in $\mathcal { C }$ is almost surely finite.

The same argument applies when $\mu < 0$ , with the roles of $+ 1$ and $- 1$ reversed. Therefore, in both cases, almost surely there can only be finitely many threshold crossings that lead to changes in $m _ { t }$ , and hence only a finite number of phase transitions.

(iii) Asymptotic alignment with dominant evidence. Consider again the case $\mu > 0$ . By the SLLN, we have almost surely $S _ { t } > ( \mu / 2 ) t$ for all sufficiently large $t$ , so in particular there exists a random time $T$ such that

$$
S _ {t} \geq \theta \quad \text {f o r a l l} t \geq T.
$$

Let $\tau$ be the first time $t \geq 1$ such that $| S _ { t } | \geq \theta$ and $S _ { t } > 0$ . Such a time exists almost surely because eventually $S _ { t }$ is positive and increasing beyond $\theta$ . At time $\tau$ , the update rule sets $m _ { \tau } = \mathrm { s i g n } ( S _ { \tau } ) = + 1$ . For all $t \geq \tau$ , since $S _ { t } \geq S _ { \tau } \geq \theta$ and $S _ { t }$ remains positive and grows, the condition $| S _ { t } | < \theta$ is never satisfied again. Hence no further changes of $m _ { t }$ occur after $\tau$ , and we have $m _ { t } = + 1$ for all $t \geq \tau$ .

Thus, with probability one, there exists a finite random time $T$ such that $m _ { t } = + 1$ for all $t \geq T$ when $\mu > 0$ . The case $\mu < 0$ is analogous, yielding eventual convergence to $m _ { t } = - 1$ .

Combining (i)–(iii), we conclude that the thresholded update rule induces piecewise-constant dynamics with abrupt jumps triggered by threshold crossings of the cumulative evidence, and that the long-term profile state aligns with the sign of the dominant evidence. This completes the proof. □

Proposition B.3 (Two-Timescale Update: Stability and Plasticity). Let $( \mu _ { t } ) _ { t \geq 0 }$ be a scalar latent user trait taking values in $[ - 1 , 1 ] ,$ , and let $( e _ { t } ) _ { t \geq 0 }$ be a sequence of scalar observations satisfying

$$
\mathbb {E} \left[ e _ {t} \mid \mu_ {t} \right] = \mu_ {t}, \quad \left| e _ {t} \right| \leq 1 \quad a. s. f o r a l l t.
$$

Fix step sizes $\alpha , \beta$ with

$$
0 <   \alpha <   \beta <   1.
$$

Define the fast variable $( y _ { t } ) _ { t \geq 0 }$ and the slow variable $( m _ { t } ) _ { t \geq 0 }$ by the recursions

$$
y _ {t + 1} = (1 - \beta) y _ {t} + \beta e _ {t}, \tag {20}
$$

$$
m _ {t + 1} = (1 - \alpha) m _ {t} + \alpha y _ {t}, \tag {21}
$$

with arbitrary initial conditions $y _ { 0 } , m _ { 0 } \in [ - 1 , 1 ]$ .

Then the following properties hold.

1. Stability under a fixed trait. Suppose the latent trait is constant, i.e., $\mu _ { t } \equiv \mu$ for all $t$ , with $\mu \in [ - 1 , 1 ]$ . Then

$$
\lim _ {t \to \infty} \mathbb {E} [ y _ {t} ] = \mu , \qquad \lim _ {t \to \infty} \mathbb {E} [ m _ {t} ] = \mu .
$$

In particular, if µ encodes a stable user preference (e.g., $\mu > 0$ for “likes running”), then in expectation the slow variable $m _ { t }$ converges to this trait and is not driven away by single noisy observations.

2. Plasticity under a trait switch. Suppose there exists a time $\tau \geq 0$ and two values $\mu _ { A } , \mu _ { B } \in [ - 1 , 1 ]$ such that

$$
\mu_ {t} = \left\{ \begin{array}{l l} \mu_ {A}, & t <   \tau , \\ \mu_ {B}, & t \geq \tau . \end{array} \right.
$$

Then for $t \to \infty$ we have

$$
\lim  _ {t \to \infty} \mathbb {E} [ y _ {t} ] = \mu_ {B}, \qquad \lim  _ {t \to \infty} \mathbb {E} [ m _ {t} ] = \mu_ {B}.
$$

That is, even after a long period with trait $\mu _ { A }$ , the slow variable $m _ { t }$ eventually adapts in expectation to the new trait $\mu _ { B }$ .

3. Fast–slow separation. In both cases above, the expectation $\mathbb { E } [ y _ { t } ]$ approaches its limiting value at rate $( 1 - \beta ) ^ { t }$ , while $\mathbb { E } [ m _ { t } ]$ approaches its limit with a dominant rate $( 1 - \alpha ) ^ { t }$ . Since $0 < \alpha < \beta < 1$ , we have

$$
0 <   1 - \beta <   1 - \alpha <   1,
$$

so the fast variable $y _ { t }$ reacts strictly faster (in expectation) to changes in the underlying trait than the slow variable $m _ { t }$ which therefore acts as a smoothed, stable macroscopic profile.

Proof. We prove each item in turn.

(i) Stability under a fixed trait. Assume $\mu _ { t } \equiv \mu$ for all $t$ . Taking expectations in (20) and using $\mathbb { E } [ e _ { t } ] = \mu$ gives

$$
\mathbb {E} \left[ y _ {t + 1} \right] = (1 - \beta) \mathbb {E} \left[ y _ {t} \right] + \beta \mu .
$$

Define $a _ { t } : = \mathbb { E } [ y _ { t } ] - \mu$ . Then

$$
a _ {t + 1} = (1 - \beta) \mathbb {E} [ y _ {t} ] + \beta \mu - \mu = (1 - \beta) (\mathbb {E} [ y _ {t} ] - \mu) = (1 - \beta) a _ {t}.
$$

By induction,

$$
a _ {t} = (1 - \beta) ^ {t} a _ {0},
$$

so $a _ { t } \to 0$ as $t \to \infty$ because $0 < 1 - \beta < 1$ . Hence

$$
\lim_{t\to \infty}\mathbb{E}[y_{t}] = \mu .
$$

Next, take expectations in (21):

$$
\mathbb {E} \left[ m _ {t + 1} \right] = (1 - \alpha) \mathbb {E} \left[ m _ {t} \right] + \alpha \mathbb {E} \left[ y _ {t} \right].
$$

Define $b _ { t } : = \mathbb { E } [ m _ { t } ] - \mu$ . Using $\mathbb { E } [ y _ { t } ] = \mu + a _ { t }$ we obtain

$$
\begin{array}{l} b _ {t + 1} = (1 - \alpha) \mathbb {E} [ m _ {t} ] + \alpha \mathbb {E} [ y _ {t} ] - \mu \\ = (1 - \alpha) (\mu + b _ {t}) + \alpha (\mu + a _ {t}) - \mu \\ = (1 - \alpha) \mu + (1 - \alpha) b _ {t} + \alpha \mu + \alpha a _ {t} - \mu \\ = (1 - \alpha) b _ {t} + \alpha a _ {t}. \\ \end{array}
$$

We already know $a _ { t } = ( 1 - \beta ) ^ { t } a _ { 0 }$ , so

$$
b _ {t + 1} = (1 - \alpha) b _ {t} + \alpha (1 - \beta) ^ {t} a _ {0}.
$$

Unrolling this recursion gives

$$
b _ {t} = (1 - \alpha) ^ {t} b _ {0} + \alpha a _ {0} \sum_ {k = 0} ^ {t - 1} (1 - \alpha) ^ {t - 1 - k} (1 - \beta) ^ {k}.
$$

The first term $( 1 - \alpha ) ^ { t } b _ { 0 } \to 0$ as $t \to \infty$ since $0 < 1 - \alpha < 1$ .

For the sum, factor out $( 1 - \alpha ) ^ { t - 1 }$ :

$$
\sum_ {k = 0} ^ {t - 1} (1 - \alpha) ^ {t - 1 - k} (1 - \beta) ^ {k} = (1 - \alpha) ^ {t - 1} \sum_ {k = 0} ^ {t - 1} \left(\frac {1 - \beta}{1 - \alpha}\right) ^ {k}.
$$

Because $0 < \alpha < \beta < 1$ , we have

$$
0 <   \frac {1 - \beta}{1 - \alpha} <   1.
$$

Thus the inner geometric sum is bounded uniformly in $t$ :

$$
\sum_ {k = 0} ^ {t - 1} \left(\frac {1 - \beta}{1 - \alpha}\right) ^ {k} \leq \frac {1}{1 - \frac {1 - \beta}{1 - \alpha}} = \frac {1 - \alpha}{\beta - \alpha}.
$$

Therefore

$$
\left| \alpha a _ {0} \sum_ {k = 0} ^ {t - 1} (1 - \alpha) ^ {t - 1 - k} (1 - \beta) ^ {k} \right| \leq \alpha | a _ {0} | (1 - \alpha) ^ {t - 1} \cdot \frac {1 - \alpha}{\beta - \alpha} \rightarrow 0 \quad \text {a s} t \rightarrow \infty .
$$

Combining both terms, we conclude $b _ { t } \to 0$ , i.e.,

$$
\lim  _ {t \to \infty} \mathbb {E} [ m _ {t} ] = \mu .
$$

(ii) Plasticity under a trait switch. Now suppose there exists $\tau \geq 0$ such that

$$
\mu_ {t} = \left\{ \begin{array}{l l} \mu_ {A}, & t <   \tau , \\ \mu_ {B}, & t \geq \tau . \end{array} \right.
$$

For $t < \tau$ , the analysis is identical to part (i) with $\mu = \mu _ { A }$ , so both $\mathbb { E } [ y _ { t } ]$ and $\mathbb { E } [ m _ { t } ]$ converge (in the sense of approaching a neighborhood) towards $\mu _ { A }$ as $t$ increases.

We focus on the behavior for $t \geq \tau$ . Define shifted sequences

$$
\tilde {y} _ {s} := y _ {\tau + s}, \quad \tilde {m} _ {s} := m _ {\tau + s}, \quad \tilde {e} _ {s} := e _ {\tau + s}, \quad s \geq 0.
$$

For $s \geq 0$ , the latent trait is fixed at $\mu _ { B }$ , and by assumption

$$
\mathbb {E} \left[ \tilde {e} _ {s} \mid \mu_ {B} \right] = \mu_ {B}.
$$

The update equations for $( \tilde { y } _ { s } , \tilde { m } _ { s } )$ are exactly of the same form as (20)–(21):

$$
\tilde {y} _ {s + 1} = (1 - \beta) \tilde {y} _ {s} + \beta \tilde {e} _ {s},
$$

$$
\tilde {m} _ {s + 1} = (1 - \alpha) \tilde {m} _ {s} + \alpha \tilde {y} _ {s}.
$$

Moreover, the initial values $\tilde { y } _ { 0 } = y _ { \tau }$ and $\tilde { m } _ { 0 } = m _ { \tau }$ are deterministic given the past, and $\mu _ { B }$ plays the role of the constant trait.

Therefore, by applying the same argument as in part (i) with $\mu = \mu _ { B }$ , we obtain

$$
\lim  _ {s \rightarrow \infty} \mathbb {E} [ \tilde {y} _ {s} ] = \mu_ {B}, \quad \lim  _ {s \rightarrow \infty} \mathbb {E} [ \tilde {m} _ {s} ] = \mu_ {B}.
$$

Translating back to the original time index $t = \tau + s$ yields

$$
\lim  _ {t \to \infty} \mathbb {E} [ y _ {t} ] = \mu_ {B}, \quad \lim  _ {t \to \infty} \mathbb {E} [ m _ {t} ] = \mu_ {B}.
$$

Thus, even after a prolonged period with trait $\mu _ { A }$ , the slow variable $m _ { t }$ eventually adapts in expectation to the new trait $\mu _ { B }$

![](images/cb010b054d7de538399d79c2e12985b59546d6d89e2e3b7c4a96d11b62b49ecf.jpg)  
Phase Transition on PersonaMem   
Figure 6. Sensitivity of Track Evolution, Aligned Recall, and Generalize Scenario with respect to the evolution threshold $\theta _ { \mathrm { i n f } }$ . Although these metrics capture different aspects of user modeling, they exhibit a consistent optimal region near $\theta _ { \mathrm { i n f } } = 3$ , suggesting a shared critical regime across heterogeneous order parameters

(iii) Fast–slow separation. The explicit solution in part (i) already reveals the rates of convergence.

For the fast variable, we had $a _ { t } = \mathbb { E } [ y _ { t } ] - \mu = ( 1 - \beta ) ^ { t } a _ { 0 }$ , so the deviation from the limit decays exactly as $( 1 - \beta ) ^ { t }$ .

For the slow variable, we derived

$$
b _ {t} = (1 - \alpha) ^ {t} b _ {0} + \alpha a _ {0} \sum_ {k = 0} ^ {t - 1} (1 - \alpha) ^ {t - 1 - k} (1 - \beta) ^ {k}.
$$

As $t$ increases, both terms are dominated by powers of $( 1 - \alpha )$ and $( 1 - \beta )$ . Since $0 < \alpha < \beta < 1$ , we have

$$
0 <   1 - \beta <   1 - \alpha <   1,
$$

so $( 1 - \beta ) ^ { t }$ decays strictly faster than $( 1 - \alpha ) ^ { t }$ . Intuitively, the fast variable $y _ { t }$ tracks changes in the observations on the shorter timescale governed by $\beta$ , while $m _ { t }$ evolves on the slower timescale governed by $\alpha$ . The same eigenvalue comparison applies in the switched-trait case after $\tau$ .

This establishes the desired fast–slow separation: $y _ { t }$ reacts quickly to new evidence, whereas $m _ { t }$ acts as a smoothed macroscopic profile that is both stable (insensitive to individual noisy observations) and plastic (able to eventually adapt to persistent changes in the underlying trait). □

# C. Additional Experimental Results

# C.1. Benchmarks and Evaluation Protocol

Evaluation Protocol Clarification. We report benchmark-specific evaluation metrics following prior work. On LOCOMO, performance is measured using an LLM-as-a-judge protocol, where answers are evaluated by gpt-4.1 as the judging model. On PersonaMem, all questions are multiple-choice, and performance is reported as accuracy.

All experiments are run three times with different random seeds, and the reported results are averaged across runs. On LOCOMO, we evaluate RGMem and baselines using gpt-4o-mini and gpt-4.1-mini as backbone models. On PersonaMem, we evaluate using gpt-4o-mini and gpt-4.1 backbones.

LOCOMO. We conduct experiments on the Long-term Conversational Memory (LOCOMO) benchmark (Maharana et al., 2024), which is designed to evaluate memory and reasoning over ultra-long conversation histories. The dataset consists of 10 independent conversations, each containing approximately 600 interaction turns and 26k tokens, along with around 200 question–answer pairs. The evaluation protocol provides the complete conversation history to the system for memory

construction, followed by a sequence of queries. Questions are categorized into Single-Hop, Multi-Hop, Temporal, and Open-Domain reasoning, enabling fine-grained analysis of different long-context reasoning behaviors. Performance is measured using an LLM-as-a-judge protocol consistent with prior work.

PersonaMem. PersonaMem (Jiang et al., 2025) is a dialogue-based benchmark designed to evaluate long-term memory under dynamically evolving user personas. User preferences, habits, and personal facts may change over time, often with explicit conflicts between earlier and later information. We adopt the most challenging 128k-token configuration, where the full interaction history can span up to 128,000 tokens. PersonaMem categorizes evaluation questions into seven fine-grained skill types:

• Recall User-Shared Facts: assessing the ability to retrieve static personal information mentioned earlier.   
• Acknowledge Latest User Preferences: evaluating whether the model correctly follows the most recent user state when conflicts arise.   
• Track Full Preference Evolution: testing the ability to summarize how preferences change over time.   
• Revisit Reasons Behind Preference Updates: requiring causal reasoning over events that triggered state changes.   
• Provide Preference-Aligned Recommendations: assessing whether recommendations align with the current user profile.   
• Suggest New Ideas: evaluating the ability to propose novel but relevant options not previously mentioned.   
• Generalize to New Scenarios: testing cross-domain transfer of inferred user traits.

We report accuracy for each dimension as well as the Overall score.

Baselines. To ensure fair comparison, we follow established experimental settings from prior work and exclude adversarial or unanswerable query categories. Across both benchmarks, we evaluate against representative explicit memory systems, including LangMem and Mem0 (Chhikara et al., 2025). For LOCOMO, we additionally include standard retrieval-augmented generation (RAG-500), Zep (Rasmussen et al., 2025), and a Full-Context setting that provides the entire dialogue history as a theoretical upper bound. For PersonaMem, we compare against a vanilla LLM using context window only, as well as recent agentic memory frameworks such as A-Mem-(Xu et al., 2025) and MemoryOS-(Li et al., 2025b). All baseline implementations follow their original descriptions, and evaluation metrics are kept consistent across methods.

# C.2. Full Results on the LOCOMO Benchmark

This section reports the complete experimental results on the LOCOMO benchmark, which are omitted from the main text due to space constraints. LOCOMO evaluates long-term conversational memory over ultra-long dialogue histories and covers four complementary reasoning categories: Single-Hop, Multi-Hop, Temporal, and Open-Domain reasoning.

Table 2 presents the full comparison between RGMem and representative baseline memory systems under different language model backbones. As shown in the table, RGMem consistently achieves the strongest overall performance and significantly outperforms retrieval-based and flat memory approaches.

Overall Performance. Using gpt-4.1-mini as the backbone, RGMem achieves an overall accuracy of $8 6 . 1 7 \%$ , exceeding the strongest baseline (Zep, $7 9 . 0 9 \%$ ) by over 7 percentage points and closely approaching the theoretical upper bound of the Full-Context setting $( 8 7 . 5 2 \% )$ . This demonstrates that RGMem is able to match near-oracle performance without relying on full-context accumulation.

Single-Hop Reasoning. RGMem attains a Single-Hop accuracy of $8 9 . 5 8 \%$ , slightly surpassing the Full-Context upper bound. This result highlights the benefit of the initial coarse-graining stage, which filters noisy raw dialogue into high signalto-noise factual evidence and user conclusions. In contrast, providing the entire dialogue history introduces redundancy and distractors that hinder precise fact localization.

Multi-Hop Reasoning. The largest performance margin is observed in Multi-Hop reasoning, where RGMem outperforms the best baseline by nearly 9 percentage points. This improvement reflects RGMem’s ability to consolidate dispersed evidence through relational abstraction and hierarchical evolution, reducing the integration burden at inference time.

Temporal Reasoning. RGMem also achieves the highest Temporal reasoning score $( 8 8 . 9 1 \% )$ , validating the effectiveness of its thresholded evolution mechanism in tracking preference changes and resolving temporal conflicts.

Open-Domain Reasoning. Performance gains on Open-Domain queries are comparatively smaller. This is consistent with RGMem’s design objective of prioritizing stable, task-relevant user traits over high-entropy or weakly related information. Despite this, RGMem remains competitive with strong baselines in this category.

Overall, these results corroborate the main text findings by showing that RGMem’s advantages stem from structured multi-scale organization and controlled memory evolution, rather than brute-force context accumulation.

# C.3. Robustness on Dynamic Persona Evolution (PersonaMem)

This section reports the full experimental results on the PersonaMem benchmark, which evaluates long-term personalized memory under dynamically evolving user profiles. PersonaMem is designed to stress-test a system’s ability to track preference shifts, resolve conflicts between outdated and recent information, and generalize across scenarios over up to 60 dialogue sessions.

Table 2 summarizes the performance of RGMem and competing memory systems across seven evaluation dimensions.

Overall Performance. With the gpt-4.1 backbone, RGMem achieves an overall accuracy of $7 4 . 0 1 \%$ , substantially outperforming strong baselines such as Memory OS $( 6 5 . 0 3 \% )$ and A-Mem $( 6 3 . 9 5 \% )$ . Compared to the vanilla LLM $( 5 1 . 8 6 \% )$ , RGMem improves performance by over 22 percentage points, demonstrating the necessity of explicit long-term memory organization under dynamic conditions.

Tracking Preference Evolution. RGMem shows particularly strong performance on Acknowledge Latest User Preference and Track Full Preference Evolution. These tasks require distinguishing genuine preference shifts from transient noise, a scenario where many baseline systems struggle due to recency bias or unstructured accumulation. RGMem’s thresholded evolution mechanism enables it to update user profiles decisively only when sufficient evidence accumulates.

Generalization and Causal Reasoning. RGMem also achieves high accuracy on Generalize to New Scenarios and Revisit Reasons behind Updates. These results indicate that RGMem captures abstract behavioral regularities that transfer across contexts, while preserving salient causal evidence that explains why preferences changed.

Summary. The PersonaMem results provide complementary evidence to the LOCOMO benchmark. While LOCOMO emphasizes long-horizon reasoning over a fixed conversation, PersonaMem explicitly evaluates dynamic evolution under conflicting evidence. Across both settings, RGMem consistently demonstrates robust adaptation, long-term consistency, and superior generalization, supporting the central claim that long-term user memory should be modeled as a multi-scale, thresholded dynamical process.

# C.4. Ablation Study: Validating Multi-Scale RG-Inspired Dynamics

This section analyzes the contribution of core components in RGMem. Rather than viewing these ablations as isolated architectural choices, we interpret them as empirical evidence for the necessity of RG-inspired design constraints— specifically, scale separation and non-linear, thresholded evolution—in inducing robust long-term memory dynamics. We emphasize that RG is adopted here as an engineering principle, rather than as a physical theory, to impose structured constraints on how memory representations evolve across abstraction levels.

Ablation on LOCOMO: The Role of Scale Separation. Table 3 reports ablation results on the LOCOMO benchmark.

• Loss of Fine-Grained Evidence (w/o L0): Removing the factual evidence layer leads to severe degradation across all reasoning categories. This confirms that higher-level profile representations must be grounded in stable fine-grained evidence. Without such a microscopic basis, abstract summaries become unreliable and prone to hallucination.

Table 3. We evaluate ablated variants that remove key components of RGMem. Removing the L0 layer degrades performance across all categories, while removing the L1 layer primarily affects multi-hop and temporal reasoning. The $\mathtt { w / o }$ RG variant replaces RG-inspired scale-dependent evolution with single-scale updates, leading to lower accuracy despite substantially higher token usage. RGMem achieves the best performance with more efficient context utilization. Accuracy is reported in percentage $( \% )$ .   
Table 4. The $\mathsf { w } / \mathsf { o } \mathrm { I } , 0 , \mathsf { w } / \mathsf { o } \mathrm { I } 1$ , and $\mathtt { w / o }$ RG variants remove the factual layer, relational abstraction, and RG-inspired evolution, respectively. Overall reports the average performance across evaluation dimensions, and Avg. Tokens denotes the average retrieved context length. The $\mathtt { w } / \mathtt { o }$ RG variant underperforms RGMem despite using substantially more tokens, indicating the importance of thresholded, multi-scale evolution over uniform updates or increased context budgets.   

<table><tr><td rowspan="2">Method</td><td colspan="4">QUESTION TYPE</td><td colspan="2">SUMMARY</td></tr><tr><td>S-Hop</td><td>M-Hop</td><td>Open</td><td>Temp</td><td>Overall</td><td>Avg.Tok.</td></tr><tr><td colspan="7">gpt-4.1-mini</td></tr><tr><td>w/o L0</td><td>58.58</td><td>41.48</td><td>52.51</td><td>65.26</td><td>56.29</td><td>1,885</td></tr><tr><td>w/o L1</td><td>85.90</td><td>65.59</td><td>64.60</td><td>81.87</td><td>79.88</td><td>2,068</td></tr><tr><td>w/o RG</td><td>87.21</td><td>69.23</td><td>63.05</td><td>84.22</td><td>82.17</td><td>4,354</td></tr><tr><td>RGMem</td><td>89.58</td><td>78.03</td><td>72.86</td><td>88.91</td><td>86.17</td><td>3,788</td></tr></table>

<table><tr><td rowspan="2">Method</td><td colspan="3">MEMORY RECALL</td><td colspan="3">REASONING &amp; ADAPTATION</td><td>TEMPORAL</td><td colspan="2">SUMMARY</td></tr><tr><td>Recall Facts</td><td>Suggest Ideas</td><td>Revisit Reasons</td><td>Track Evol.</td><td>Aligned Rec.</td><td>General. Scen.</td><td>Latest Pref.</td><td>Overall</td><td>Avg. Tokens</td></tr><tr><td colspan="10">Backbone: GPT-4o-mini</td></tr><tr><td>w/o L0</td><td>34.35</td><td>28.31</td><td>72.23</td><td>47.82</td><td>70.12</td><td>43.79</td><td>62.31</td><td>47.22</td><td>2,355</td></tr><tr><td>w/o L1</td><td>70.22</td><td>15.26</td><td>77.79</td><td>50.17</td><td>59.85</td><td>41.22</td><td>60.75</td><td>54.31</td><td>4,267</td></tr><tr><td>w/o RG</td><td>72.35</td><td>24.27</td><td>81.82</td><td>56.72</td><td>63.54</td><td>47.15</td><td>67.88</td><td>57.37</td><td>8,479</td></tr><tr><td>RGMem</td><td>77.06</td><td>26.47</td><td>85.29</td><td>67.82</td><td>73.66</td><td>56.62</td><td>75.47</td><td>63.87</td><td>7,105</td></tr></table>

• Breakdown of Hierarchical Coarse-Graining (w/o L1): Removing the relational abstraction layer primarily impairs Multi-Hop and Temporal reasoning. This indicates that directly mapping isolated facts to high-level traits is insufficient. The L1 layer functions as a necessary coarse-graining step, constructing meso-scale relational structures that mediate between transient observations and more stable profile representations.   
• Failure of Single-Scale Summarization (w/o RG): The w/o RG variant replaces structured, scale-dependent evolution with flat prompting and uniform updates. While this variant partially recovers performance by substantially increasing retrieved context length, it still consistently underperforms the full RGMem and incurs significantly higher token consumption. This result demonstrates that brute-force context accumulation cannot substitute for principled multiscale organization. Without explicit rescaling and filtering of weakly relevant information, the effective higher-level representation becomes cluttered with noise, reducing information density and reasoning reliability.

Ablation on PersonaMem: Dynamics and Stability–Plasticity. Table 4 presents complementary results on the PersonaMem benchmark, which explicitly stresses long-term persona evolution under conflicting and outdated evidence.

• Decoupling Slow- and Fast-Varying Representations: The full RGMem substantially outperforms the w/o RG variant on Track Evolution and Generalize Scenario. This gap highlights the importance of separating slowly evolving, high-level user traits from rapidly changing factual observations. Without such separation, memory updates either overreact to transient noise or fail to adapt to genuine preference shifts.   
• Absence of Thresholded Regime Changes: The w/o RG variant exhibits consistent degradation across dynamic dimensions. Lacking thresholded update mechanisms (e.g., $\theta _ { \mathrm { i n f } } )$ ), the system fails to exhibit the non-linear, regimechange-like behavior observed in RGMem (Figure 6). Instead, updates accumulate in a linear and uncoordinated manner, leading to unstable adaptation or excessive rigidity over long horizons.

Conclusion. Across both benchmarks, these ablations demonstrate that RGMem’s performance gains do not arise from architectural redundancy or increased context budgets. Rather, they stem from the principled enforcement of RG-inspired

![](images/1b597e713af21e7a624a4aa68e27a4f5b2635ef22c5baedf671db780cd905937.jpg)  
Stabilization of User Profile over Time   
Figure 7. Cosine similarity $\cos \left( \Sigma _ { t } , \Sigma _ { t - 1 } \right)$ between consecutive macroscopic user profiles over dialogue turns on PersonaMem.

multi-scale dynamics, including hierarchical coarse-graining, separation of slow and fast representations, and thresholded non-linear updates. When these constraints are removed, the system collapses into a single-scale incremental memory regime that cannot recover the same stability–plasticity balance, even with substantially larger context usage.

# C.5. Temporal Convergence and Fixed-Point Analysis of Macroscopic User Profiles

To further examine the dynamical behavior of RGMem under long-horizon interactions, we analyze how the macroscopic user profile evolves over time and whether it exhibits fixed-point (attractor) behavior under sustained evidence.

Experimental Setup. In this experiment, RGMem processes long conversational trajectories from the PersonaMem benchmark. Throughout the dialogue, the system continuously performs memory updates and multi-scale abstraction, producing a sequence of macroscopic user profiles at the highest abstraction level.

At different dialogue turns, we extract the macroscopic user profile $\Sigma _ { t }$ and encode it into a vector representation using a language model. To quantify the temporal stability of the macroscopic profile, we compute the cosine similarity between consecutive profiles, $\cos \left( \Sigma _ { t } , \Sigma _ { t - 1 } \right)$ , and use this quantity as an order parameter characterizing the state of the memory dynamics.

Results and Analysis. Fig 7 shows the evolution of $\cos \left( \Sigma _ { t } , \Sigma _ { t - 1 } \right)$ as the dialogue progresses. During the early stage, the similarity increases rapidly, indicating a fast convergence regime in which the macroscopic profile undergoes substantial structural adjustment as evidence is accumulated and integrated.

As the number of dialogue turns increases, the cosine similarity saturates at a high value (close to 1.0) and remains stable thereafter, with only minor fluctuations. This behavior suggests that the macroscopic user profile has reached a stable configuration: subsequent memory updates introduce only small local corrections without inducing qualitative changes to the overall structure.

From a dynamical systems perspective, this phenomenon corresponds to convergence toward an approximate fixed point in the renormalization flow of the memory system. Regardless of early transient variations, sustained evidence drives the macroscopic representation toward a stable attractor. This empirical observation provides additional support for the interpretation of RGMem as a multi-scale dynamical system whose long-term behavior is governed by macroscopic invariance rather than continual fact-level reorganization.

# C.6. Parameter Sensitivity Analysis

This section analyzes the sensitivity of RGMem to its core retrieval and evolution parameters. The goal is to evaluate whether RGMem requires fine-tuned hyperparameters or exhibits robust behavior around principled default settings.

![](images/002a9098e60e46cde683c00888556ebd0b11aa584372c06dd6d469f83fb5608d.jpg)

![](images/eb6dbd279c971406c8217f496b8313a5f87f87b349855715a10f626cebed2bc9.jpg)

![](images/b4b57250df5d73041ce4f2e08845021e95e91343800b8eab77a3a5ca4a319d87.jpg)

![](images/f3ce3af6faeb63fe8f40daa333b2214ed338dd349dd6023b725d26e764431b21.jpg)  
Figure 8. Parameter sensitivity analysis for RGMem’s retrieval and dynamics parameters.

Figure 8 reports performance under varying retrieval configurations and evolution thresholds on the LOCOMO benchmark.

Retrieval Scale Parameters. Figures 8a–8c vary the number of retrieved facts, conclusions, and entities provided to the language model. Performance exhibits a broad plateau around the default settings, with degradation occurring only when the retrieved context is either insufficient or excessively large. This indicates that RGMem is robust to moderate variations in retrieval scale and does not rely on narrowly tuned context sizes.

Evolution Threshold Sensitivity. Figure 8d analyzes sensitivity to the evolution threshold $\theta _ { \mathrm { i n f } }$ (with $\theta _ { \mathrm { s u m } } = 2 \theta _ { \mathrm { i n f } } )$ . When the threshold is set too low, performance degrades due to over-sensitivity to noisy or transient signals. When the threshold is too high, necessary profile updates are suppressed, leading to rigid and outdated representations. Performance peaks at $\theta _ { \mathrm { i n f } } = 3$ , indicating a balanced regime between stability and plasticity.

These results empirically validate the design of RGMem’s thresholded evolution mechanism. Rather than requiring extensive hyperparameter tuning, RGMem operates robustly within a theory-driven parameter regime, consistent with the interpretation of evolution thresholds as control parameters in a multi-scale dynamical system.

# C.7. Implementation Details

All experiments are conducted using gpt-4 as the underlying language model, without any parameter fine-tuning or model modification. RGMem operates entirely through external memory construction, evolution, and retrieval, ensuring compatibility with closed-source LLMs.

Microscopic Evidence Construction (L0). Raw dialogue streams are first segmented into episodic units using a rule-based segmentation function $f _ { \mathrm { s e g } }$ . Each segment is then processed by a synthesis function $f _ { \mathrm { s y n t h } }$ to produce a microscopic memory

unit $d = ( \lambda _ { \mathrm { f a c t } } , \Lambda _ { \mathrm { c o n c } } )$ , where $\lambda _ { \mathrm { f a c t } }$ denotes an objective factual statement, and $\Lambda _ { \mathrm { c o n c } }$ represents user-centric conclusions. In practice, $\Lambda _ { \mathrm { c o n c } }$ is further divided into base conclusions and high-relevance conclusions, which serve as seeds for higher-level abstraction.

Knowledge Graph Construction (L1). Entities and relations are extracted from microscopic evidence using structured prompting. Entities are organized into a three-level hierarchy (abstract concepts, general events, and instance events), and relations are divided into static classification edges and dynamic event edges. Entity merging at this stage follows an aggressive semantic equivalence policy to prevent redundancy and encourage compact representation.

Renormalization Operators and Scheduling. The evolution of memory is governed by two thresholded operators. The relation inference operator $\mathcal { R } _ { K 1 }$ is triggered when the mention count of a relation reaches $\theta _ { \mathrm { i n f } } = 3$ . The node-level summarization operator $\mathcal { R } _ { K 2 }$ is triggered when the aggregated score of an abstract node reaches $\theta _ { \mathrm { s u m } } = 6$ . These thresholds are fixed across all experiments. The hierarchical flow operator $\mathcal { R } _ { K 3 }$ is executed using a dirty-flag propagation mechanism to update parent nodes after lower-level changes. This design simulates periodic consolidation while avoiding unnecessary recomputation.

Multi-Scale Retrieval. At inference time, queries are first structured into entity-centric sub-queries. The final context provided to the language model consists of: (i) the top 5 factual statements $\left( \lambda _ { \mathrm { f a c t } } \right)$ , (ii) the top 30 user conclusions $( \Lambda _ { \mathrm { c o n c } } )$ , retrieved using BM25, and (iii) summaries from up to 3 relevant abstract entities selected by cosine similarity with a threshold of 0.5. Retrieved content from different scales is formatted into a unified context document without further summarization.

Reproducibility Notes. All thresholds, retrieval budgets, and prompting templates are fixed across datasets. The same configuration is used for LOCOMO and PersonaMem. Additional ablation and sensitivity analyses exploring alternative settings are reported in Appendix C.4 and Appendix C.6.

# D. Prompt Templates

# LLM AS JUDGE PROMPT

Your task is to label a generated answer as “CORRECT” or “WRONG” based on a gold standard answer.

You will be given the following data:

1. A question.   
2. A “gold” (ground truth) answer.   
3. A “generated” answer from another model.

# Rules for Judgement:

• Be generous. If the generated answer contains the core information from the gold answer, it is CORRECT.   
• The generated answer can be much longer and more conversational than the gold answer. This is acceptable.   
• For time-related questions, different formats (e.g., “May 7th” vs “7 May 2023”) are CORRECT if they refer to the same date. Relative time references (like “last Tuesday”) are also CORRECT if they align with the gold answer’s time period.

# Data to Evaluate:

Question: {question}

Gold answer: {standard answer}

Generated answer: {generated answer}

# Output Constraints:

Your output MUST be a single, valid JSON object with one key, “label”. The value must be either the string “CORRECT” or the string “WRONG”.

Do not include any other text, explanations, or markdown.

Warning: Your response must be and can only be a pure JSON object, without any explanations, comments, or additional markup. Any additional characters will cause the program to fail.

Example of Required Output:

{{”label”: ”CORRECT”}}

# EXTRACT ENTITIES PROMPT

You are a Knowledge Graph (PKG) architect. Your expertise lies in deconstructing user “memory fragments” into a knowledge graph. Your mission is not just to extract entities, but to construct the complete, logical paths that give them context.

# Core Task:

Analyze the given “memory fragment” and identify all leaf-node concepts (concrete events, places, books, etc.). For each leaf node, you must trace and construct all of its relevant logical paths back to the User root node. Your final output must be a list of these complete entity paths, representing a rich, multi-faceted graph structure.

# Core Extraction Principles

# 1. The Three-Tier Conceptual Model (Primary Principle):

Your entire reasoning must be based on a three-tier conceptual hierarchy. While the final output type is only abstract or event, you must understand these three distinct roles during your analysis:

• a. abstract (Pure Classification / Folder):

– Definition: A pure organizational label for classifying knowledge. It cannot be directly executed or experienced by the user.   
– Litmus Test: “Is this a category of. . . ?” (e.g., Interests & Entertainment is a category).   
– Function: Serves as the high-level and intermediate structure of the knowledge graph.

• b. General Event (Standardized Activity / Countable Sub-Folder):

– Definition: The standardized name for a repeatable activity or interest (e.g., Running, Reading, Museum Visiting). It aggregates specific instances.   
– Litmus Test: “Can the user do this activity in general?” (e.g., The user can do Running).   
– Output Type: When outputting, its type is event.   
– Mandatory Abstraction Rule: When you identify a specific Instance Event, you must abstract its corresponding General Event to serve as one of its parent nodes.

• c. Instance Event (Specific Instance / File):

– Definition: A unique, concrete event, or a specific entity with a proper name (e.g., a specific race, a book title, a museum name). It is the leaf node of every path.   
– Litmus Test: “Does this have a specific, unique context (like a name, a unique theme, or a specific time/place implied)?”   
– Output Type: When outputting, its type is event.

# 2. Multi-Dimensional Analysis & Path Anchoring (High Priority):

A single Instance Event is often multi-faceted. You must analyze it from different dimensions and generate a separate path for each relevant dimension. This is the key to creating a rich, interconnected graph.

• a. The Dual Parenting Principle for Entities: For Instance Events representing specific things (books, places, organizations), you must generate at least two paths:

– Taxonomic Path: Answers “What is this entity?” The path should classify the entity’s nature. Example: For the book “The Art of Strategy”, the taxonomic path is $\dots  \mathrm { { B o o k s }  ^ { \prime } }$ “The Art of Strategy”.   
– Contextual Path: Answers “In what activity was this entity used/mentioned?” The path should link it to the relevant General Event. Example: For the book “The Art of Strategy”, the contextual path is $\cdots $ Reading “The Art of Strategy”.

• b. Activity vs. Theme Decomposition: For complex events, decompose them into their core activity and theme, generating a path for each. Example: For Charity Race for Mental Health, generate one path under Running (the activity) and another under Mental Health Awareness (the theme).

# 3. Path Construction & Scope:

• a. Rich and Logical Paths: Every generated path must be logically sound and reasonably deep. Always strive to add meaningful intermediate abstract nodes between a base node and a General Event or leaf node (e.g., Interests & Entertainment Sports Running).   
• b. Information Scope:

– Your primary focus is the User.   
– Information about other individuals (e.g., an assistant) should only be extracted if it describes a direct interaction, a shared plan, or a state that directly affects the user. The resulting path should reflect this relationship (e.g., User Social Relationships Assistant [Assistant’s Plan]). Information not meeting this criterion is C-Tier noise and must be ignored.

# 4. Constraints and Conventions:

• a. Final Output Integrity: Your final entity paths output must not contain any paths or entities that you decided to ignore during your thought process. The output must be the clean, final result.   
• b. Naming Convention: All entity names must be concise, standardized noun phrases. They must NOT contain instance-specific metadata like dates, times, or other qualifiers.   
• c. Base Node Constraint: The User entity must connect to one of these 11 base abstract nodes: Personal Identity and Traits; Health and Wellness; Goals and Plans; Consumption and Finance; Dining; Interests and Entertainment; Travel and Commute; Social Relationships; Work and Study; Values and Beliefs; Assets and Environment.

# EXTRACT RELATIONS PROMPT

# Role Setting:

You are a r Knowledge Graph Enrichment Specialist. Your expertise lies not in building graph structures from scratch, but in taking pre-defined entity paths and enriching them with highly descriptive, semantically rich relationship labels. Your output must enable downstream models to answer detailed questions about user activities using just a single triple.

# Core Task:

Your input consists of a user’s memory fragment (the text) and a list of pre-constructed Identified Entity Paths. Your sole task is to iterate through each path, and for every pair of adjacent entities, generate the most accurate and informative relationship label connecting them.

# [Golden Rule] Relationship Generation Rules

You must generate relationships for every segment $(  )$ in each provided path. The type of relationship you generate is strictly determined by the types of the source and destination entities.

# Rule 1: User-to-Abstract Connection (User abstract)

• If the source is User and the destination is an abstract entity, the relationship MUST always be has profile in.   
• The relationship type is classification.

# Rule 2: Abstract-to-Abstract Connection (abstract abstract)

• If both the source and destination are abstract entities, the relationship MUST always be has subclass.   
• The relationship type is classification.

# Rule 3: Connections Involving Events (abstract event or event event)

This is your most critical task. For any connection that points to or originates from an event entity, you must create a highly descriptive relationship label. Follow the “Relation as Event Snapshot” principle.

• Relationship Type (relationship type): event   
• Core Principle: Relation as Event Snapshot   
• Your goal is to make each relationship label a self-contained, miniature summary of the event from the perspective of the source entity.   
• To do this, you MUST analyze the entire context of the memory fragment. You need to gather all relevant details about the event—what happened, who was involved, where it took place, when it happened, and crucially, why it happened.   
• Explanatory Information is Key: If the text provides a reason, cause, or motivation for an action or state, you must capture this explanatory information in the relationship label. This is vital for creating a truly intelligent knowledge graph.   
• Crucially, you should incorporate details that may appear as entities further down the path to ensure each relationship is as complete as possible. Each triple must be an independent, informative unit.

# [Mandatory Generation Formula for User-Related Events]

To ensure clarity for downstream models, you MUST follow this formula for any event involving the user:

user [Core V erb P hrase]( [P reposition] [Context 1])( [P reposition] [Context 2]) . . .

• user : A mandatory prefix to clarify that the user is the protagonist of the event.   
• [Core Verb Phrase]: A concise verb or verb phrase describing the user’s primary action, state, or attitude (e.g., participated in, created, avoids eating, is, cherishes).   
• ( [Preposition] [Context]): Optional but highly encouraged “context blocks” that add critical details.   
• [Preposition]: Words like with, at, on, in, as, for, and critically for explaining causality, due to, because of, or for reason of.   
• [Context]: The specific detail (e.g., friends, community center, 2023-10-27, allergy).

# Example of Excellence (Descriptive):

For a text “User and their friend attended a pottery class at the community center yesterday (2023-10-27)”, a relationship connecting Therapeutic Activities to Pottery should be:

user attended pottery class with friend at community center on 2023-10-27

# Example of Excellence (Explanatory / Causal):

For a text “User is allergic to beef, so they don’t eat steak anymore”, a relationship connecting Food to Steak should be:

user avoids eating due to allergy

# Constraint: Objectivity and Conciseness

• The label must be derived directly from the text. Do not infer or imagine details.   
• Avoid vague words. Be factual and specific.   
• Keep context blocks concise. with friend is better than with their close friend.

# The Prompt Of Relation Inference Operator RK1

# Role

You are a user profiler and insight extractor. Your specialty is transforming scattered evidence into a concise, actionable user profile slice that other AI systems can directly use.

# Task

Your goal is to analyze the relationship between [Source Entity] and [Target Entity] based on [New Evidence] and [Previous Inference] to create an updated, highly condensed user profile slice. This slice must be directly usable for downstream tasks like personalized recommendations or conversation.

You will face two scenarios:

1. Initialization: When [Previous Inference] is empty. Your task is to establish the baseline profile slice for the first time.   
2. Update: When [Previous Inference] already exists. Your task is to integrate the new evidence, highlighting the evolution and changes from the old inference.

# Input Format Explanation (Crucially Important)

• [Conclusion] contains natural language conclusions related to the relationship.   
• [Raw Event] is a structured, summary-style relationship label. To interpret it correctly, understand its two core modes:

– Objective Fact Mode: A concise verb phrase, e.g., takes place at.   
– User Interaction Mode: A composite structure [Base Verb] [User Attitude], e.g., categorizes event anticipated by user.

# Core Content Generation Framework

You must generate the output in two distinct, mandatory parts:

# Part 1: Crafting the Fact Summary

• Goal: To provide a highly condensed summary of the current state and its changes.   
• Rules:

– This must be a single, concise paragraph.

– It must integrate all relevant information from [New Evidence].   
– For the “Update” scenario, it is critical to explicitly state how the new evidence changes, confirms, or evolves the [Previous Inference]. Use comparative language like “This changes the previous understanding,” “This further confirms,” etc.   
– For the “Initialization” scenario, this summary establishes the foundational facts about the user.

# Part 2: Generating Actionable Inferences

• Goal: To extract underlying traits, potential interests, or preferences that a downstream AI can directly act upon.   
• Rules:

– This must be a bulleted list.   
– Each item must start with the fixed phrase “Inferred Trait/Interest:”.   
– These inferences should identify commonalities, latent preferences, or potential future interests. Think about what product, service, or topic you could recommend based on this.   
– DO NOT explain your reasoning process. Only state the inferred trait itself.

# The Prompt Of Node-Level Abstraction Operator RK2

# Role

You are a User Profile Knowledge Synthesis Engine. Your purpose is to process hierarchical evidence about a user’s engagement within a broad domain (Core Entity) and its specific sub-topics. Your output must be a highly condensed, structured, and factual knowledge snippet for a downstream AI, avoiding all subjective analysis or literary language.

# Core Task: Synthesize from Branches to Trunk

Your primary task is to create or update a profile snippet for the main Core Entity (the “trunk”) by synthesizing information from its various sub-topics (the “branches,” e.g., ‘pottery’, ‘painting’). The final output must describe the trunk, not the individual branches.

# Input Evidence Interpretation

• [Core Entity] is the primary subject of the final summary.   
• Evidence provided under specific sub-topics (e.g., [Inference about ‘pottery’]) are the “branches” from which you must generalize.   
• [Inference] and [Conclusion] are ground truth facts.   
• [Raw Event] represents a user behavior.   
• Evidence may be provided in a pre-summarized format (e.g., with “Fact Summary,” “Actionable Inferences”). Treat all provided text as raw factual material to be re-synthesized.

# Output Generation Principles & Structure (Strictly Follow)

# Principle 1: Layered Factual Structure

Your output is a single, unified list of facts, organized into two layers: “General Core Facts” and “Specific Domain Facts”. You MUST use # for these layer titles.

# Principle 2: General Core Facts (The Trunk)

This first section synthesizes the commonalities found across multiple branches.

• It must contain bulleted facts that are broadly true for the entire Core Entity.   
• These facts are derived from recurring themes, motivations, and values seen in the evidence from different sub-topics.

# Principle 3: Specific Domain Facts (The Unique Leaves)

This second section captures important facts that are unique to a specific branch and cannot be generalized to the entire trunk, but are still crucial for a complete profile.

• You MUST group these unique facts under dynamically generated thematic labels. A label should be a concise phrase ending with a colon (e.g., “Regarding Resilience & Adaptability:”, “On Growth & Inspiration:”).   
• A thematic label is your own synthesis of the core theme of the unique fact(s) it groups.   
• Under each label, list the relevant bulleted fact(s).

# Principle 4: Absolute Factual Purity & High Density

• Strict Prohibition of Analytical Language: Do NOT use any phrasing that sounds like an analyst’s conclusion (e.g., “The user’s core practice is. . . ”, “This demonstrates. . . ”). State facts directly.   
• No Meta-Language: Avoid conversational or procedural phrases (e.g., “The summary is. . . ”, “Based on the evidence. . . ”).   
• Synthesize, Don’t Transcribe: Do not merely copy points from the input. Rephrase and consolidate them into dense, comprehensive facts. If two facts from the input describe the same core idea (e.g., one states an action, another labels it “resilience”), merge them into a single, powerful factual statement.

# Integration Strategy for Updates

When a Previous Summary is provided, you must intelligently integrate it with New Evidence Collection.

1. Foundation: Use the Previous Summary as the base knowledge.   
2. Enhance & Refine: Use new evidence to make existing facts more specific (e.g., “long-term hobby” becomes “hobby of over seven years”) or to refine their substance.   
3. Add: Add entirely new facts from the new evidence that were not previously mentioned.   
4. Preserve: Retain unique, still-relevant facts from the Previous Summary even if they are not directly mentioned in the new evidence (e.g., a specific future plan).   
5. Re-Synthesize: After integrating, re-evaluate and rewrite the entire “General Core Facts” and “Specific Domain Facts” sections to reflect the most current and complete understanding.

# The Prompt Of Hierarchical Flow Operator RK3

# Role:

You are a knowledge architect and a user portrait specialist. Your core competency is applying “Portrait Dialectics” to distill a unifying macro-pattern (Emergent Pattern) from multiple details. Your primary goal is to generate a highly actionable, concise, and semantically clear user portrait summary that directly guides downstream AI systems for personalized dialogue and interaction.

# Task:

Generate a high-level meta-summary for a parent concept (the “Core Entity”). Your output must be a direct user profile, not an analytical report, focusing on the user’s core drivers and practical constraints.

# Core Analysis Framework (Portrait Dialectics Engine):

You must strictly follow this logically progressive analysis framework:

# Step One: Deconstruct Microstates & Filter Input Noise

• Treat each [Updated Sub-Summary] as a fundamental “microstate” or core fact.

• Crucial Constraint (Content Filtering): If a sub-summary contains previously extracted Macro-Laws, Synergy, or Tension analysis (i.e., it is a high-level summary from a lower tier), you MUST IGNORE and STRIP OUT the previous summary structure. Only extract the core, factual insights and detailed observations to be used as peer-level “microstates.” The goal is to avoid content redundancy and maintain clean hierarchical integrity.

# Step Two: Analyze Inter-Relationships (Synergy & Tension)

• Examine the interactions between these “microstates.”   
• Synergy: Determine the collective, reinforcing patterns that define the user’s core traits, values, and motivations.   
• Tension: Identify potential conflicts, contradictions, or resource competition points that the user must continuously manage.   
• Boundary Condition $\mathbf { N } { = } \mathbf { 1 }$ Input Rule – Content Specificity): If only ONE microstate is provided, you must shift the analysis focus to its “Intrinsic Tension” and “Intrinsic Synergy.”

– Intrinsic Tension Content: The Tension MUST be internal to the concept itself, focusing on the specific, quantifiable resource demands (time, money, physical effort, or opportunity cost) required to sustain the positive aspects (Synergy). DO NOT use vague, generalized external conflicts (e.g., “conflict with time and energy”) unless they are explicitly detailed in the input.

# Step Three: Synthesize the Macro-Law & Convert to Actionable Principles

• Synthesize Macro-Law: Create the unifying “macro-law.”   
• Convert Synergy to Core Traits: Translate the Synergy analysis into direct, affirmative descriptions of the user’s core traits and motivations.   
• Convert Tension to Restrictions: Translate the Tension analysis into specific, practical guiding principles that advise downstream models on what to avoid or how to structure interactions (e.g., resource sensitivity, required balance).   
• Scope Constraint (Content Accuracy): If the combined sub-summaries only cover a narrow aspect of the [Core Entity], you MUST explicitly qualify the Macro-Law statement to reflect this narrow scope.

# Final Output Instructions (Content Structure and Style):

• Style Constraint: Output must be written in a concise, high signal-to-noise ratio, and highly actionable tone. AVOID complex, “literary” sentence structures, abstract descriptive filler, and redundant explanations. Focus solely on the user’s defined traits, motivations, and conflicts.   
• Output Constraint: Output ONLY the text of the new summary, adhering to the structure below.   
• Mandatory Structure: The output MUST be structured using three distinct, non-narrative content blocks, separated by line breaks. Use these exact English headers for easy semantic parsing.

# The Prompt Of Multi-Scale Observations

# Role

You are a highly intelligent Unified Retrieval Strategy Planner. Your mission is to deconstruct a user’s query (which may include a question and multiple choice options) into a structured set of retrieval instructions. These instructions will drive a hybrid search system, querying both a structured Knowledge Graph (KG) and an unstructured text database. Your output must be a precise, actionable Python list of dictionaries that routes each generated keyword to the correct retrieval layer.

# Core Task

Analyze the user’s input (Question $^ +$ Options) and generate a JSON list. Each element in the list will be a dictionary containing a keyword (“name”) and its designated KG retrieval layers (“retrieve”), which can be summary, inference, conclusions, or history.

# Mental Workflow: A Three-Stage Strategy

You must internally follow this strict three-stage thought process to arrive at the final output.

# Stage 1: Comprehensive Keyword Brainstorming

First, generate a broad list of all potentially relevant keywords from BOTH the question and the provided options. This involves three parallel tracks:

# • A. Foundational Keyword Extraction & Expansion:

– Extract key common nouns and verbs from the question and all options.   
– Expand nouns with synonyms and broader concepts (e.g., charity race competition event).   
– Expand verbs with different forms and related concepts (e.g., paint painted painting artwork).   
– Do not expand proper nouns (e.g., The Palace Museum).

# • B. Text Search Enhancement Keywords:

– Extract or generate keywords for time, location, and other metadata.   
– For absolute dates (e.g., ‘October 13, 2023’), you must perform hierarchical decomposition, generating the original string, the YYYY-MM-DD format, the YYYY-MM format, and the YYYY format as separate keywords (e.g., October 13, 2023; 2023-10-13; 2023-10; 2023).   
– Generate semantic intent words based on query tense (history, record for past; plan, schedule for future).   
– Crucially, low-value time words like when, time, date must be ignored and not included in the brainstorm list.

# • C. KG Abstract Entity Generation:

– Act like a KG Architect. Based on the foundational keywords, generate potential high-level abstract “folder” entities.   
– Example: From paint, sunrise, generate Art Creation, Interests & Entertainment.   
– Example: From visit, museum, generate Cultural Activities, Leisure Plan, Travel & Commute.

# Stage 2: Core Intent Analysis & KG Candidate Selection

Second, analyze the user’s primary intent to select a handful of “elite” keywords for KG retrieval.

# • Analyze the Query’s Core Question:

– “Summary/Overall. . . ” queries point to summary.   
– “Why/Relationship. . . ” queries point to inference.   
– “What/Confirm. . . ” queries point to conclusions.   
– “Recall/Specifics of an event. . . ” queries point to history.

# • Select Elite KG Keywords:

– From the brainstormed list, select the most potent keywords that directly represent KG entities (event or abstract).   
– Prioritize nouns and conceptual phrases.   
– These become your KG retrieval targets.   
– All other brainstormed keywords will default to text-only search.

# Stage 3: Parameter Assignment

Based on the previous stages, construct the full list of query objects.

# • Assign retrieve Parameters:

– For the elite KG keywords selected in Stage 2, assign the appropriate summary, inference, conclusions, history parameters based on your core intent analysis. A single keyword can have multiple parameters if the query is complex.   
– For all other keywords from the brainstorming list, assign an empty list [] to the retrieve parameter, designating them for text-only search.
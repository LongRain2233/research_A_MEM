# D-SMART: Enhancing LLM Dialogue Consistency via Dynamic Structured Memory And Reasoning Tree

Xiang Lei 1, Qin Li1*, Min Zhang1, Min Zhang1

1 Shanghai Key Laboratory of Trustworthy Computing, East China Normal University, Shanghai, China

# Abstract

Large Language Models (LLMs) often exhibit factual inconsistencies and logical decay in extended, multi-turn dialogues, a challenge stemming from their reliance on static, pre-trained knowledge and an inability to reason adaptively over the dialogue history. Prevailing mitigation strategies, such as Retrieval-Augmented Generation (RAG) and agentic working memories, improve information recall but still engage with fundamentally static knowledge sources and follow pre-defined single reasoning path. This hinders their ability to preserve factual and logical consistency of their responses in multi-turn dialogues while the context evolves over time. To address this issue, we propose D-SMART, a model-agnostic framework designed to maintain multi-turn dialogue consistency by enabling LLMs to build and reason over a dynamic, structured representation of the conversational context. This is achieved via two synergistic components: (1) a Dynamic Structured Memory (DSM), which incrementally constructs and maintains an authoritative, OWL-compliant knowledge graph of the conversation; and (2) a Reasoning Tree (RT), which executes inferences as an explicit and traceable multistep search over the graph. As the popular-used quality score (judged by GPT-4) can overlook logical flaws, we introduce new NLI-based metrics to better measure multi-turn dialogue consistency. Comprehensive experiments on the MT-Bench-101 benchmark show that D-SMART significantly outperforms state-of-the-art baselines, elevating the dialogue consistency score by over $48 \%$ for both proprietary and opensource models, and notably improves the quality score of the latter by up to $1 0 . 1 \%$ .

# Introduction

While Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks (Li et al. 2024), maintaining logical and factual consistency in extended, multi-turn dialogues remains a significant challenge (Yi et al. 2024). Lapses in this consistency can severely undermine effective information conveyance, intention clarification, and consensus building (She et al. 2024). However, the architectural foundations of contemporary LLMs often struggle to meet this critical requirement, especially as dialogues become longer and more complex (Liu et al. 2023; Huang et al. 2025).

![](images/06a056b1ab1e0a17cc258e1a1853f2bc1dafceb21bfcbab38faea77a49927168.jpg)  
Figure 1: A case of LLM exhibiting logical inconsistency in a multi-turn dialogue.

This challenge stems from a fundamental reliance on static, pre-trained knowledge and the processing of an unstructured dialogue history, where attention to prior turns can decay (Hu et al. 2024; Zhao et al. 2024). This can lead to difficulties in resolving context ambiguity, managing conflicts between the model’s internal knowledge and newly introduced dialogue-specific facts (Xu et al. 2024a), and preventing the propagation of errors from early misunderstandings (Laban et al. 2025).

Such vulnerabilities can lead to critical failures, as illustrated in Figure 1, where the model contradicts prior user input. This exposes the limitation of prevailing mitigation strategies like RAG (Gao et al. 2023) or agentic working memories. While such methods augment LLMs with external knowledge (She et al. 2024) or textual history, they do not guarantee that the model will correctly reason over the emergent facts presented (Xue et al. 2023). This gap hinders their ability to maintain long-term logical coherence and adapt to evolving factual contexts (Liu et al. 2024). Notably, an eloquent but inconsistent response, as shown in Figure 1, may still receive a high score from holistic quality evaluators like GPT-4, which can overlook severe logical flaws. This highlights the pressing need for more discerning eval-

uation methods that can directly assess the logical integrity of dialogue responses.

To address these limitations in both reasoning and evaluation, we propose D-SMART, a framework comprised of two synergistic components: (1) a Dynamic Structured Memory (DSM), which incrementally constructs an OWLcompliant knowledge graph to serve as a dynamic, logicallyconsistent knowledge base tailored for the dialogue; and (2) a Reasoning Tree (RT), which guides the LLM to perform explicit, traceable, multi-step reasoning over the DSM. In essence, the DSM provides a structured memory of “what to think about”, while the RT provides a deliberate process for “how to think” based on that memory. Complementing the framework, we also introduce new Natural Language Inference (NLI)-based metrics to better assess multi-turn dialogue consistency, moving beyond holistic quality scores.

The main contributions of this paper are as follows:

• We propose D-SMART, a model-agnostic framework that drives traceable and faithful reasoning over a dynamic, dialogue-specific structured memory, enhancing dialogue consistency.   
• We introduce two NLI-based metrics, the Consistency Score (CS) and the Dialogue Entailment Rate (DER), providing a more precise evaluation of multi-turn dialogue consistency by measuring logical entailment.   
• Comprehensive experiments on MT-Bench-101 show D-SMART significantly outperforms baselines, boosting consistency by over $48 \%$ and demonstrating stability against logical decay in extended dialogues turns.

![](images/47d36af6c53b0eab24d0ead00bd3325c921298318ecc76122d0c729e2b18b312.jpg)  
Figure 2: An overview of the D-SMART framework.

# The D-SMART Framework

Inspired by cognitive architectures like Soar (Laird, Newell, and Rosenbloom 1987), we propose D-SMART, a framework to enhance the logical consistency and factual grounding of LLMs in multi-turn dialogues. Formally, instead of the traditional paradigm: $\boldsymbol { r } _ { t } \sim p _ { \boldsymbol { \theta } } ( \boldsymbol { r } _ { t } \mid \boldsymbol { q } _ { t } , \mathcal { H } _ { t - 1 } )$ , D-SMART reformulates response generation as

$$
r _ {t} \sim p _ {\theta} \left(r _ {t} \mid q _ {t}, \mathcal {H} _ {t - 1}, \mathcal {G} _ {t - 1}, \mathcal {T} _ {t} ^ {*}\right) \tag {1}
$$

thereby coupling a dialogue-specific memory graph $\mathcal { G } _ { t }$ with a reasoning tree $\tau _ { t } ^ { * }$ . As illustrated in Figure 2, in the Response Phase, upon receiving a user query $( q _ { t } )$ , the framework forwards it to the Reasoning Engine. Guided by a set of actions, the engine’s RT actively explores the DSM $( \mathcal { G } _ { t - 1 } )$ to construct a grounded context for response generation $\left( r _ { t } \right)$ . Subsequently, the completed turn $( q _ { t } , r _ { t } )$ triggers the Memory Maintenance Phase, where the framework extracts new knowledge and resolves any emerging conflicts to dynamically update the memory graph. This updated DSM $( \mathcal { G } _ { t } )$ is prepared for the subsequent Response Phase, ensuring that the model’s responses are continuously anchored in an evolving representation of the conversational common ground.

# Dynamic Structured Memory Construction and Maintenance

The DSM mitigates context decay by incrementally transforming the unstructured dialogue history into an evolving, OWL-compliant knowledge graph. Unlike standard dialogue systems that represent history as a flat text sequence $\mathcal { H } _ { t - 1 }$ (Logan et al. 2019), which can be susceptible to information loss and contradiction (Liu, Yogatama, and Blunsom 2022), the DSM provides a persistent and computationally tractable knowledge base.

Formally, this graph at turn $t$ , which is a set of factual triples $\mathcal { G } _ { t } \overset { \cdot } { = } \{ ( e _ { h } , \overset { \cdot } { r } , \overset { \cdot } { e } _ { t } ) \} \subseteq \mathcal { E } _ { t } \times \mathcal { R } _ { t } \times \mathcal { E } _ { t }$ , yields a structured representation of the conversational common ground (Cheng et al. 2023), built exclusively from the dialogue itself.

The construction and maintenance of the DSM follows a turn-by-turn, multi-step workflow designed to ensure semantic fidelity and logical coherence (Hixon, Clark, and Hajishirzi 2015). For each completed dialogue turn $h _ { t } \ =$ $\left( q _ { t } , r _ { t } \right)$ , the following steps are executed:

1. Structured Statement Generation: We first employ an LLM to distill the core semantic content of the turn $h _ { t }$ into a concise, assertion-like natural language statement (Luo et al. 2023), $s _ { t }$ , normalizing colloquial or fragmented utterances into a format amenable to knowledge extraction, which can be formalized as: $s _ { t } \sim p _ { \theta } ^ { s } ( s _ { t } \mid h _ { t } )$ , where $p _ { \theta } ^ { s }$ is the LLM prompted to perform this distillation (prompt details are provided in Appendix E).   
2. Knowledge Fragment Extraction: From $s _ { t }$ , we extract a OWL-compliant knowledge graph fragment, $\mathcal { G } _ { t } ^ { \prime }$ . This is achieved by a neuro-symbolic pipeline (Gangemi et al. 2025), denoted as KGE: $\mathcal { G } _ { t } ^ { \prime } = \mathrm { K G E } ( s _ { t } )$ , achieving approximately $9 5 \%$ overall accuracy (Algorithm 1 in Appendix A). OWL’s formal semantics provide a foundation for consistency checking and logical inference, thereby

safeguarding against the propagation of logical errors (De Giorgis, Gangemi, and Russo 2025).

3. Dynamic Update and Conflict Resolution: To account for new, clarified, or corrected information, the newly extracted fragment $\mathcal { G } _ { t } ^ { \prime }$ is merged with the existing graph $\mathcal { G } _ { t - 1 }$ . This is a two-stage conflict resolution process:

• Conflict Detection: An LLM is prompted to perform a semantic comparison between the serializations of $\mathcal { G } _ { t - 1 }$ and $\mathcal { G } _ { t } ^ { \prime }$ . It identifies any triples in $\mathcal { G } _ { t - 1 }$ that are contradicted or superseded by the information in $\mathcal { G } _ { t } ^ { \prime }$ .   
• Graph Pruning and Merging: Conflicting triples identified in the previous step are pruned from $\mathcal { G } _ { t - 1 }$ . The $\mathcal { G } _ { t } ^ { \prime }$ is then merged into $\mathcal { G } _ { t - 1 }$ . An OWL reasoner can then be used to verify the logical consistency of the resulting graph $\mathcal { G } _ { t }$ against predefined axioms.

This update mechanism, denoted by the operator $\oplus$ , can be formalized as:

$$
\mathcal {G} _ {t} = \mathcal {G} _ {t - 1} \oplus \mathcal {G} _ {t} ^ {\prime} = \left(\mathcal {G} _ {t - 1} \backslash p _ {\theta} ^ {c} \left(\mathcal {G} _ {t - 1}, \mathcal {G} _ {t} ^ {\prime}\right)\right) \cup \mathcal {G} _ {t} ^ {\prime} \tag {2}
$$

where $p _ { \theta } ^ { c }$ represents the LLM-based conflict detection function that returns the set of triples from $\mathcal { G } _ { t - 1 }$ to be removed (see Appendix E for prompt details).

Through this workflow, the DSM evolves as a faithful and logically consistent chronicle of the dialogue. This structured memory provides a stable foundation for the system’s reasoning capabilities.

# Reasoning Tree for Memory Traversal

Simply linearizing a graph for the LLM’s context can be suboptimal, as LLMs may struggle to interpret raw structured data, which can lead to opaque and less reliable reasoning (Agrawal et al. 2024; Jiang et al. 2023). Effectively leveraging this structure for response generation requires a dedicated mechanism (Modarressi et al. 2023), the RT guides the LLM to explore multiple, alternative reasoning paths over the DSM. Inspired by the Tree-of-Thought (ToT) paradigm (Yao et al. 2023), the RT framework moves beyond a single, linear reasoning chain that may fail. This structure allows the model to deliberately explore diverse reasoning trajectories, backtrack from unpromising avenues, and select the most coherent path grounded in the DSM.

The core idea is to model the reasoning process for a given query $q _ { t }$ as a search for an optimal path within a reasoning tree, $\mathcal { T } _ { t }$ . The construction of this tree is an iterative process where the LLM acts as a high-level planner. Starting from a root node representing the initial query, the framework progressively expands the tree by generating potential next steps. A search algorithm navigates this expanding tree, prioritizing more promising paths based on a value function, also estimated by the LLM. This mitigates the risk of error accumulation often seen in linear reasoning chains.

Formally, each node $\tau _ { i }$ in the tree $\mathcal { T } _ { t }$ represents a reasoning state $\boldsymbol { S } _ { i }$ . We define this state as a four-tuple: $\begin{array} { r } { \mathcal { S } _ { i } ~ = } \end{array}$ $( \tilde { \mathcal { G } } _ { i } , \bar { \mathcal { Z } } _ { i } , v _ { i } , d _ { i } )$ .

• $\tilde { \mathcal { G } } _ { i } \subseteq \mathcal G _ { t }$ is the relevant subgraph accumulated at node $\tau _ { i }$ . It serves as the grounded, local context for the current reasoning step.

• $\mathcal { Z } _ { i } ~ = ~ \langle z _ { 1 } , z _ { 2 } , . . . , z _ { j } \rangle$ is the reasoning trajectory, a sequence of actions that led to the current state.   
• $v _ { i } \in [ 0 , 1 ]$ is a value score assigned by the LLM, evaluating the potential of state $s _ { i }$ to lead to a correct answer for $q _ { t }$ . This is estimated via $v _ { i } \sim p _ { \theta } ^ { v } ( v \mid q _ { t } , S _ { i } )$ .   
• $d _ { i }$ is the depth of the node in the tree.

To ground this reasoning process in the DSM, we define a set of discrete actions $\mathcal { A }$ that the LLM can propose to transition between states. These actions are designed for explicit graph traversal and knowledge manipulation, extending standard thought generation with symbolic operations:

• Expand Entity $\left( a _ { e e } \right)$ : Given an entity in $\tilde { \mathcal { G } } _ { i }$ , retrieve its n-hop neighborhood from the full DSM $\mathcal { G } _ { t }$ to enrich the local context.   
• Find Path $( a _ { f p } )$ : Given two entities, search for connecting paths within $\mathcal { G } _ { t }$ and add them to $\tilde { \mathcal { G } } _ { i }$ , facilitating multihop reasoning.   
• Think $( a _ { t k } )$ : Synthesize the information in $S _ { i }$ to generate an intermediate thought or plan the subsequent action.   
• Answer $( a _ { a s } )$ : A terminal action to generate the final response based on the information gathered in $s _ { i }$ .

At each expansion step from a node $\tau _ { i }$ , the LLM-based policy proposes an action $a _ { i j } \sim p _ { \theta } ^ { \pi } ( a \mid q _ { t } , S _ { i } )$ , which results in a new state $S _ { j }$ . The objective is to construct a tree that maximizes the joint probability of the sequence of actions forming the most plausible reasoning path. This can be formalized as:

$$
\begin{array}{l} \mathcal {T} _ {t} ^ {*} = \arg \max  _ {\mathcal {T} _ {t}} p _ {\theta} \left(\mathcal {T} _ {t} \mid q _ {t}, \mathcal {G} _ {t}\right) \tag {3} \\ \propto \prod_ {\left(\tau_ {i}, a _ {i j}, \tau_ {j}\right) \in \mathcal {T} _ {t}} p _ {\theta} ^ {\pi} \left(a _ {i j} \mid q _ {t}, \mathcal {S} _ {i}\right) \\ \end{array}
$$

where the probability of a tree is defined by the product of policy probabilities for the sequence of actions taken.

The search is not a blind exploration but a traceable traversal over a structured memory, enabling a form of deliberate, “slow” reasoning (Kahneman 2012) to address complex queries while ensuring conclusions remain faithful to the established dialogue history.

# From Structured Memory to Grounded Response

The core design principle of D-SMART is to leverage the LLM not as a passive generator operating on unstructured text, but as a high-level semantic orchestrator. The LLM guides the structured traversal of the DSM via the RT framework to find an optimal reasoning path from the tree $\tau _ { t } ^ { * }$ that produces a coherent and grounded answer to the query $q _ { t }$ , transforming the opaque, internal reasoning of a standard LLM into an explicit and traceable process.

To solve this search problem tractably, we operationalize the tree traversal using a beam search algorithm, which navigates the state space defined by the RT, balancing traversal of diverse reasoning paths with computational feasibility. At each step, the LLM-as-policy proposes actions from $\mathcal { A }$ to expand the most promising nodes on the beam. Graph-related

![](images/a2ba8b26c0233043e16bca8da73ba134a78c284ac8325e56a201e16eec43382e.jpg)  
Figure 3: The reasoning process at turn t. The tree expands by executing actions, enriching its local knowledge from the DSM. The LLM evaluates each step, and the final path leads to a grounded answer, which will update the DSM to $\mathcal { G } _ { t }$ for the next turn.

actions are executed deterministically against the DSM, ensuring that each reasoning step is grounded in the structured memory. The LLM-as-value-function then evaluates the newly generated states, and the beam is updated to retain the top- $k$ most promising candidates for the next iteration. The search terminates upon reaching a maximum depth or generating a final answer from the trajectory of the highestscoring state, as illustrated in Figure 3. The detailed procedure is presented in Algorithm 2 in Appendix A.

This approach, focusing on maintaining consistency with dialogue-specific knowledge, is complementary to existing paradigms like RAG, which primarily address the integration of external knowledge. The structured nature of D-SMART’s reasoning process ensures that the final response is not only contextually relevant but also demonstrably consistent with the facts established throughout the dialogue.

# Experimental Setup

To rigorously evaluate the efficacy of our D-SMART framework, we designed a comprehensive experimental procedure. Our experiments are structured to answer the following research questions (RQs):

• RQ1: How effective and stable is D-SMART’s performance in multi-turn dialogues compared to state-of-theart baselines?   
• RQ2: Can multi-turn consistency be enhanced without compromising response quality?   
• RQ3: What is the significance of introducing the two core components, DSM and RT, and do they exhibit a synergistic effect?

# Datasets and Tasks

We conduct our experiments on MT-Bench-101 (Bai et al. 2024), a benchmark designed for evaluating multi-turn dialogue capabilities. Its structure, featuring 13 task cate-

gories that require long-term memory and complex reasoning (e.g., Mathematical Reasoning, Instruction Clarification), provides an ideal testbed for assessing D-SMART’s dynamic memory and reasoning abilities. We adopt an autoregressive evaluation setting where models condition on their own previously generated responses. While this may result in lower absolute scores, it stringently tests a model’s ability to maintain coherence and recover from its own errors (see Appendix D for task information).

# Evaluation Metrics

Our protocol combines automated quality scoring with novel metrics designed to quantify logical consistency.

GPT Score. The holistic quality score (1–10), as judged by GPT-4, is used in MT-Bench-101 to assess overall coherence and task fulfillment. However, while powerful for macro-level assessment, its holistic nature means it can be susceptible to overlooking subtle logical contradictions, particularly when they are masked by a stylistically sound and persuasive response (Mendonca, Trancoso, and Lavie 2024).

NLI-based Consistency Metrics. To overcome the limitation of GPT score, we introduce two metrics that use a DeBERTa-v3-large model, fine-tuned for Natural Language Inference (NLI), to quantitatively audit for logical integrity and consistency. For each turn, the model computes the probabilities of Entailment $( P _ { E _ { i } } )$ and Contradiction $( P _ { C _ { i } } )$ against the dialogue history, providing a targeted measure of a statement’s logical grounding.

• Consistency Score (CS): Quantifies consistency by rewarding entailment and penalizing contradiction. A score near 1.0 indicates entailment, while a score near 0.0 signals contradiction. The CS of the $i$ -th turn of a dialogue can be calculated as:

$$
\mathrm {C S} _ {\mathrm {i}} = \frac {\left(P _ {E _ {i}} - P _ {C _ {i}}\right) + 1}{2} \tag {4}
$$

• Dialogue Entailment Rate (DER): Measures the proportion of turns classified as “ENTAILMENT”, reflecting the tendency to produce logically grounded responses. It is calculated for a dialogue of T turns as:

$$
\mathrm {D E R} = \frac {1}{\mathbb {T}} \sum_ {i = 1} ^ {\mathbb {T}} \mathbb {I} (\text {l a b e l} _ {i} = \text {E N T A I L M E N T}) \tag {5}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function. We note that CS is sensitive to the NLI model’s confidence (e.g., the margin between $P _ { E _ { i } }$ and $P _ { C _ { i } }$ ), while DER is not.

Lexical Metrics. For completeness, we report Word F1 and BLEU scores, while noting they are not primary indicators of the logical or semantic correctness that our work focuses on.

# Baselines and Implementation.

We compare D-SMART against a range of strong models and frameworks. For fair comparison, memory-augmented baselines and D-SMART are built on the same base LLMs.

# Proprietary Models.

• GPT-4o: Represents a powerful baseline for models relying on a large (128k) static context window, based on OpenAI’s state-of-the-art model.   
• Mem0 (w/ GPT-4o): A memory framework that uses an LLM to dynamically manage a set of unstructured textual facts extracted from the dialogue.   
• MemoryBank (w/ GPT-4o): A system utilizes dense retrieval over an expanding long-term memory store, which incorporates a mechanism for memory decay.

Both Mem0(Chhikara et al. 2025) and MemoryBank(Zhong et al. 2024) are representative of memory systems that operate on unstructured text, providing a contrast to D-SMART’s structured, formal knowledge graph approach.

# Open-Source Models.

• Qwen-8B: A baseline for standard LLM performance without explicit memory modules. Note that its internal “thinking mode” was disabled in our experiments, which may result in some performance degradation.   
• COMEDY-13B: An end-to-end framework representing an alternative memory management strategy based on compression of the dialogue history(Chen et al. 2025).

Detailed implementation specifics, hyper-parameters, and a computational cost analysis are provided in Appendix B.

# Results and Analysis

This section presents a comprehensive empirical evaluation structured around our three research questions. Results are averaged over three runs. Detailed case studies are provided in Appendix D.

# RQ1: Overall Performance against Baselines

Primary Finding. D-SMART significantly enhances both response quality and logical consistency over stateof-the-art baselines across both proprietary and opensource models. As detailed in Table 1, D-SMART applied to GPT-4o achieves a score of 8.63, surpassing the native GPT-4o (8.20) and other memory-augmented methods like Mem0 (8.31). More importantly, it demonstrates superior logical consistency, elevating the DER to $3 8 . 5 1 \%$ from $2 3 . 8 8 \%$ for the next-best baseline (MemoryBank), corresponding to an $84 \%$ improvement over the base model. This is further corroborated by the Consistency Score (CS), which rises to 0.692 from the baseline’s 0.594. Notably, this leap in logical integrity does not come at the cost of fluency; the lexical metrics (BLEU, Word F1) remain on par with the strongest baselines, demonstrating that the structured reasoning process does not render the output robotic or unnatural.

Amplifying Open-Source Models. The benefits of D-SMART are particularly pronounced on open-source models, where it acts as a strong capability enhancer. When integrated with Qwen-8B, D-SMART improves the score from 7.79 to 8.58 $( + 1 0 . 1 \% )$ , a score that approaches the D-SMART-GPT-4o level and far surpasses the much larger COMEDY-13B model (5.75). The consistency metrics show a similar uplift, with DER increasing from $2 6 . 2 3 \%$ to $3 8 . 7 3 \%$ (a $48 \%$ lift over the base model). This result suggests that D-SMART’s explicit, structured reasoning can effectively scaffold smaller models, enabling them to approach the performance of larger, state-of-the-art systems.

Stability in Extended Dialogues. D-SMART demonstrates remarkable stability against performance decay in multi-turn long dialogues, a common failure point for existing models. Figure 4 illustrates that while most baselines, including the powerful GPT-4o and memory-augmented variants, exhibit a sharp decline in performance and consistency in later turns (e.g., after turn 5), both D-SMART-GPT-4o and D-SMART-Qwen-8B maintain high and stable scores throughout the interaction.

![](images/65b3f3184f3c5740fb159011230eafe6c60d69c6fc34b5de061df01a4abb72e6.jpg)  
Figure 4: GPT Score (lines) and Consistency Score (bars) across dialogue turns. Error bands represent the standard error of the mean.

The performance collapse in baseline models is intrinsically linked to a decay in logical coherence. As shown by the bars in Figure 4, the Consistency Score (CS) of baselines drops in tandem with their GPT score, often falling

Table 1: Overall performance comparison on MT-Bench-101.   

<table><tr><td></td><td>Model</td><td>GPT Score</td><td>CS</td><td>DER</td><td>F1</td><td>BLEU</td></tr><tr><td rowspan="4">GPT-4o</td><td>Cloud-based</td><td>8.20</td><td>0.594</td><td>20.94%</td><td>0.424</td><td>0.522</td></tr><tr><td>Mem0</td><td>8.22</td><td>0.602</td><td>21.94%</td><td>0.426</td><td>0.524</td></tr><tr><td>MemoryBank</td><td>8.30</td><td>0.621</td><td>23.88%</td><td>0.393</td><td>0.554</td></tr><tr><td>D-SMOTE (ours)</td><td>8.63</td><td>0.692</td><td>38.51%</td><td>0.414</td><td>0.549</td></tr><tr><td>COMEDY-13B</td><td></td><td>5.75</td><td>0.522</td><td>6.34%</td><td>0.140</td><td>0.185</td></tr><tr><td rowspan="2">Qwen3-8B</td><td>Local Inference</td><td>7.79</td><td>0.627</td><td>26.23%</td><td>0.286</td><td>0.481</td></tr><tr><td>D-SMOTE (ours)</td><td>8.58</td><td>0.689</td><td>38.73%</td><td>0.388</td><td>0.548</td></tr></table>

below the 0.5 threshold, which signifies the generation of self-contradictory content. Conversely, D-SMART sustains the highest Consistency Score (CS) across all turns, demonstrating that its structured memory and reasoning process effectively tracks informational shifts to mitigate the logical decay induced by evolving conversational contexts.

# RQ2: Balancing Quality and Consistency.

A key challenge in dialogue generation is maintaining factual consistency without sacrificing response quality. Figure 5 visualizes this trade-off, plotting response quality (GPT Score) against consistency (CS). The results show that baseline models often operate in a “Risk” quadrant, producing fluent but logically flawed responses. In contrast, D-SMART models are consistently situated in the ideal upperright quadrant, demonstrating an ability to simultaneously achieve high quality and high consistency. The average position of D-SMART systems (marked by “X”) is clearly superior on both axes. Furthermore, the larger bubble sizes for D-SMART variants also indicate a higher DER, confirming their outputs are more frequently a direct logical entailment of the dialogue context.

![](images/67b7e603bf93133f5a631cd9cc6a8fd6db4897d5cd72658bbe549d8724d16913.jpg)  
Figure 5: Quality (GPT Score) is plotted against Consistency (CS), with bubble size representing DER.

# RQ3: Ablation Study on Core Components

To dissect the interplay between the DSM and the RT, we conducted an ablation study on a curated, more challenging subset of MT-Bench-101. The rationale for this methodological choice is detailed in Appendix C. The results, presented in Table 2 and Figure 6, reveal a nuanced, model-dependent relationship between the components.

For the highly capable GPT-4o, the DSM-only (“w/o RT”) configuration yields the highest GPT Score (9.17), demonstrating the model’s intrinsic ability to leverage the factual anchor provided by the DSM. However, its lower consistency scores (CS/DER) compared to the full framework suggest this unguided reasoning lacks discipline. The RT’s primary role here is not empowerment but regulation, enforcing a traceable, multi-path reasoning process that remains faithful to the DSM. Conversely, removing the DSM (“w/o DSM”) leads to a significant performance drop, as the RT’s structured reasoning becomes ungrounded and speculative.

Table 2: Ablation Analysis of the Impact of D-SMART’s Core Components.   

<table><tr><td></td><td>Model</td><td>GPT Score</td><td>CS</td><td>DER</td></tr><tr><td rowspan="4">GPT-4o</td><td>Base</td><td>9.02</td><td>0.57</td><td>16.66%</td></tr><tr><td>w/o RT</td><td>9.17</td><td>0.73</td><td>46.02%</td></tr><tr><td>w/o DSM</td><td>8.71</td><td>0.66</td><td>33.56%</td></tr><tr><td>Full (ours)</td><td>9.11</td><td>0.76</td><td>52.22%</td></tr><tr><td rowspan="4">Qwen3-8B</td><td>Base</td><td>7.80</td><td>0.66</td><td>32.10%</td></tr><tr><td>w/o RT</td><td>5.69</td><td>0.70</td><td>41.30%</td></tr><tr><td>w/o DSM</td><td>7.97</td><td>0.61</td><td>23.03%</td></tr><tr><td>Full (ours)</td><td>8.86</td><td>0.74</td><td>50.62%</td></tr></table>

In contrast, for a smaller model like Qwen-8B, the components are symbiotically indispensable. The DSM-only (“w/o RT”) setup results in a severe performance collapse (GPT Score drops to 5.69), indicating the model suffers from ”information overload” without the RT acting as a crucial navigator for the knowledge graph. While the RT-only (“w/o DSM”) variant provides a minor benefit from structured prompting, the sharp decline in DER (from $3 2 . 1 0 \%$ to $2 3 . 0 3 \%$ ) confirms that this reasoning, detached from the DSM’s factual basis, is logically inconsistent.

In essence, these findings highlight a clear synergy. The DSM provides a dialogue-consistent knowledge base (“what to think about”), while the RT ensures a faithful

![](images/e8092abaf6ce486029db28d4d66af7aea59745455c2f7b789ad3f6d2448b3e1c.jpg)

![](images/5c25f2e7be742dfd97e4044b422e05417fe4c147ea72311232693a464ee5a7e1.jpg)  
Figure 6: Ablation study on 13 tasks from MT-bench-101 (details in Appendix D) for GPT-4o and Qwen-8B.

and deliberate traversal of that base (“how to think”). For smaller models, this framework provides an essential cognitive scaffold; for larger models, it refines their powerful capabilities into reliable and verifiable performance.

# Related Work

This section grounds our approach by reviewing three interrelated areas: Conversational Memory for information recall, Knowledge Graph Grounding for factual accuracy, and Deliberate Reasoning Frameworks for structured inference.

# Conversational Memory in LLMs

A key research direction involves external memory systems to overcome the limitations of fixed context windows, such as the “lost in the middle” problem (Liu et al. 2024). Current approaches include memory compression (e.g., COMEDY (Chen et al. 2025)) and retrieval over unstructured text memories (e.g., MemoryBank (Zhong et al. 2024)) to improve information retention. D-SMART is architecturally distinct, employing a formal, OWL-compliant DSM whose semantics enable verifiable consistency checks and deterministic reasoning, capabilities not inherent to text-based memories.

# Knowledge Graph Grounding for Dialogue

Another line of work grounds dialogue in Knowledge Graphs (KGs) to improve factual accuracy (Chaudhuri, Rony, and Lehmann 2021). Yet, most approaches leverage large, static KGs and thus cannot adapt to new facts emerging within a conversation (Das et al. 2018). While some research explores dynamic KG construction for specific tasks (Lu and Wang 2025; Bosselut, Le Bras, and Choi 2021), the focus is often on integrating external world knowledge (Hixon, Clark, and Hajishirzi 2015). In contrast, D-SMART’s DSM is dialogue-specific and dynamic, designed to model and maintain the evolving conversational common ground. This focus necessitates a streamlined mechanism for real-time conflict detection and resolution as dialogue facts are introduced, clarified, or corrected.

# Deliberate Reasoning Frameworks for LLMs

To enhance the depth and reliability of LLM inference, various structured reasoning frameworks have been developed. While Chain-of-Thought prompting (Wei et al. 2022) initiated this trend, its strictly linear format restricts exploration and self-correction. The Tree-of-Thoughts (ToT) paradigm (Yao et al. 2023) mitigates these issues by branching multiple reasoning trajectories. For structured inputs, methods such as framing LLMs as KG-traversal agents (Sun et al. 2023) or merging LLMs with KGs for graph refinement (Pan et al. 2024; Xu et al. 2024b) have emerged. Our RT adopts ToT’s explicit architecture but functions on a symbolic DSM, executing deterministic actions. This shift from unstructured natural-language “thoughts” to symbolic traversal enhances control, verifiability, and reliability.

# Discussion

The efficacy of D-SMART is built upon its architectural shift from passive knowledge recall to active, structured reasoning. The framework’s dual components, the DSM and RT, synergistically overcome the limitations of relying on static, unstructured dialogue history. The DSM provides a consistently updated, dialogue-specific factual foundation to mitigate context logical decay, while the RT imposes a deliberate, traceable reasoning process upon this memory to guide the model toward sound conclusions. However, the framework’s performance is intrinsically linked to the capabilities of the underlying LLM that orchestrates it. The DSM’s integrity is contingent on the LLM’s proficiency in semantic distillation and conflict adjudication. Similarly, the RT’s effectiveness depends on its ability to generate plausible actions and evaluate intermediate states. The advancements demonstrated by D-SMART are thus predicated on the base model’s semantic and logical competence. Understanding and enhancing the LLM’s reliability in these fine-grained control tasks is therefore a crucial direction for maximizing the framework’s potential and for future research.

A primary limitation of this approach is the increased computational overhead and response latency as RT extends LLM inference time by branching multiple reasoning paths, our experiments show that the average inference time per turn for a local open-source model increases from approximately 0.3s to 1.3s. The subsequent memory maintenance requires approximately 6s per turn (detailed computational cost analysis are provided in Appendix B); note that it can be performed asynchronously, allowing it to overlap with the user’s input time, thereby mitigating its impact on the interactive flow. We contend this is a deliberate and necessary architectural trade-off. While methods relying on everlarger context windows are faster, they remain susceptible to factual inconsistency and the “lost in the middle” problem in extended dialogues. D-SMART, in contrast, exchanges a measure of speed for a significant enhancement in logical coherence and stability, achieving a near $48 \%$ increase in dialogue consistency (DER) for open-source models. This positions the framework as a viable option for applications where reliability and trustworthiness are paramount.

# Conclusion and Future Work

In this paper, we proposed D-SMART, a model-agnostic framework that enhances dialogue consistency by coupling a Dynamic Structured Memory with a Reasoning Tree. Our experiments show that by grounding the LLM’s generation process in an explicit, dynamically updated knowledge base, D-SMART significantly mitigates logical decay in multiturn dialogues and improves response quality, particularly in extended dialogues where baseline models falter.

Future work will proceed along three main directions: enhancing the DSM with more sophisticated memory management, optimizing the RT’s search efficiency to reduce latency through techniques such as heuristic pruning and batched parallel generation, and expanding the framework’s scope to integrate external knowledge.

# References

Agrawal, G.; Kumarage, T.; Alghamdi, Z.; and Liu, H. 2024. Can Knowledge Graphs Reduce Hallucinations in LLMs? : A Survey. In Duh, K.; Gomez, H.; and Bethard, S., eds., Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 3947–3960. Mexico City, Mexico: Association for Computational Linguistics.   
Bai, G.; Liu, J.; Bu, X.; He, Y.; Liu, J.; Zhou, Z.; Lin, Z.; Su, W.; Ge, T.; Zheng, B.; and Ouyang, W. 2024. MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues. In Ku, L.-W.; Martins, A.; and Srikumar, V., eds., Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 7421–7454. Bangkok, Thailand: Association for Computational Linguistics.   
Bosselut, A.; Le Bras, R.; and Choi, Y. 2021. Dynamic neuro-symbolic knowledge graph construction for zeroshot commonsense question answering. In Proceedings of the AAAI conference on Artificial Intelligence, volume 35, 4923–4931.   
Chaudhuri, D.; Rony, M. R. A. H.; and Lehmann, J. 2021. Grounding dialogue systems via knowledge graph aware decoding with pre-trained transformers. In European Semantic Web Conference, 323–339. Springer.   
Chen, N.; Li, H.; Chang, J.; Huang, J.; Wang, B.; and Li, J. 2025. Compress to Impress: Unleashing the Potential of Compressive Memory in Real-World Long-Term Conversations. In Rambow, O.; Wanner, L.; Apidianaki, M.; Al-Khalifa, H.; Eugenio, B. D.; and Schockaert, S., eds., Proceedings of the 31st International Conference on Computational Linguistics, 755–773. Abu Dhabi, UAE: Association for Computational Linguistics.   
Cheng, X.; Luo, D.; Chen, X.; Liu, L.; Zhao, D.; and Yan, R. 2023. Lift yourself up: Retrieval-augmented text generation with self-memory. Advances in Neural Information Processing Systems, 36: 43780–43799.   
Chhikara, P.; Khant, D.; Aryan, S.; Singh, T.; and Yadav, D. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Das, R.; Munkhdalai, T.; Yuan, X.; Trischler, A.; and Mc-Callum, A. 2018. Building dynamic knowledge graphs from text using machine reading comprehension. arXiv preprint arXiv:1810.05682.   
De Giorgis, S.; Gangemi, A.; and Russo, A. 2025. Neurosymbolic graph enrichment for Grounded World Models. Information Processing & Management, 62(4): 104127.   
Gangemi, A.; Graciotti, A.; Meloni, A.; Nuzzolese, A. G.; Presutti, V.; Reforgiato Recupero, D.; and Russo, A. 2025. py-amr2fred: A Python Library for Converting Text into OWL-Compliant RDF KGs. In Curry, E.; Acosta, M.; Poveda-Villalon, M.; van Erp, M.; Ojo, A.; Hose, K.; ´ Shimizu, C.; and Lisena, P., eds., The Semantic Web, 65– 83. Cham: Springer Nature Switzerland. ISBN 978-3-031- 94578-6.   
Gao, Y.; Xiong, Y.; Gao, X.; Jia, K.; Pan, J.; Bi, Y.; Dai, Y.; Sun, J.; Wang, H.; and Wang, H. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2(1).   
Hixon, B.; Clark, P.; and Hajishirzi, H. 2015. Learning Knowledge Graphs for Question Answering through Conversational Dialog. In Mihalcea, R.; Chai, J.; and Sarkar, A., eds., Proceedings of the 2015 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 851–861. Denver, Colorado: Association for Computational Linguistics.   
Hu, X.; Chen, J.; Li, X.; Guo, Y.; Wen, L.; Yu, P. S.; and Guo, Z. 2024. Towards understanding factual knowledge of large language models. In The Twelfth International Conference on Learning Representations.   
Huang, L.; Yu, W.; Ma, W.; Zhong, W.; Feng, Z.; Wang, H.; Chen, Q.; Peng, W.; Feng, X.; Qin, B.; and Liu, T. 2025. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. ACM Trans. Inf. Syst., 43(2).   
Jiang, J.; Zhou, K.; Dong, Z.; Ye, K.; Zhao, X.; and Wen, J.- R. 2023. StructGPT: A General Framework for Large Language Model to Reason over Structured Data. In Bouamor, H.; Pino, J.; and Bali, K., eds., Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 9237–9251. Singapore: Association for Computational Linguistics.   
Kahneman, D. 2012. Thinking, fast and slow. London: Penguin. ISBN 9780141033570 0141033576.   
Laban, P.; Hayashi, H.; Zhou, Y.; and Neville, J. 2025. Llms get lost in multi-turn conversation. arXiv preprint arXiv:2505.06120.   
Laird, J. E.; Newell, A.; and Rosenbloom, P. S. 1987. SOAR: An architecture for general intelligence. Artificial Intelligence, 33(1): 1–64.   
Li, J.; Yang, Y.; Bai, Y.; Zhou, X.; Li, Y.; Sun, H.; Liu, Y.; Si, X.; Ye, Y.; Wu, Y.; et al. 2024. Fundamental capabilities of large language models and their applications in domain scenarios: A survey. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 11116–11141.

Liu, B.; Ash, J.; Goel, S.; Krishnamurthy, A.; and Zhang, C. 2023. Exposing attention glitches with flip-flop language modeling. Advances in Neural Information Processing Systems, 36: 25549–25583.   
Liu, N. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua, M.; Petroni, F.; and Liang, P. 2024. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics, 12: 157–173.   
Liu, Q.; Yogatama, D.; and Blunsom, P. 2022. Relational Memory-Augmented Language Models. Transactions of the Association for Computational Linguistics, 10: 555–572.   
Logan, R.; Liu, N. F.; Peters, M. E.; Gardner, M.; and Singh, S. 2019. Barack’s Wife Hillary: Using Knowledge Graphs for Fact-Aware Language Modeling. In Korhonen, A.; Traum, D.; and Marquez, L., eds., ` Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 5962–5971. Florence, Italy: Association for Computational Linguistics.   
Lu, Y.; and Wang, J. 2025. KARMA: Leveraging Multi-Agent LLMs for Automated Knowledge Graph Enrichment. arXiv preprint arXiv:2502.06472.   
Luo, L.; Li, Y.-F.; Haffari, G.; and Pan, S. 2023. Reasoning on graphs: Faithful and interpretable large language model reasoning. arXiv preprint arXiv:2310.01061.   
Mendonca, J.; Trancoso, I.; and Lavie, A. 2024. ECoh: Turnlevel Coherence Evaluation for Multilingual Dialogues. In Kawahara, T.; Demberg, V.; Ultes, S.; Inoue, K.; Mehri, S.; Howcroft, D.; and Komatani, K., eds., Proceedings of the 25th Annual Meeting of the Special Interest Group on Discourse and Dialogue, 516–532. Kyoto, Japan: Association for Computational Linguistics.   
Modarressi, A.; Imani, A.; Fayyaz, M.; and Schutze, H. ¨ 2023. Ret-llm: Towards a general read-write memory for large language models. arXiv preprint arXiv:2305.14322.   
Pan, S.; Luo, L.; Wang, Y.; Chen, C.; Wang, J.; and Wu, X. 2024. Unifying large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data Engineering, 36(7): 3580–3599.   
Perez-Rosas, V.; Wu, X.; Resnicow, K.; and Mihalcea, R. ´ 2019. What Makes a Good Counselor? Learning to Distinguish between High-quality and Low-quality Counseling Conversations. In Korhonen, A.; Traum, D.; and Marquez, ` L., eds., Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 926–935. Florence, Italy: Association for Computational Linguistics.   
She, S.; Huang, S.; Wang, X.; Zhou, Y.; and Chen, J. 2024. Exploring the Factual Consistency in Dialogue Comprehension of Large Language Models. In Duh, K.; Gomez, H.; and Bethard, S., eds., Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 6087–6100. Mexico City, Mexico: Association for Computational Linguistics.   
Sun, J.; Xu, C.; Tang, L.; Wang, S.; Lin, C.; Gong, Y.; Ni, L. M.; Shum, H.-Y.; and Guo, J. 2023. Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph. arXiv preprint arXiv:2307.07697.

Wang, A.; Pruksachatkun, Y.; Nangia, N.; Singh, A.; Michael, J.; Hill, F.; Levy, O.; and Bowman, S. R. 2019. SuperGLUE: a stickier benchmark for general-purpose language understanding systems. Red Hook, NY, USA: Curran Associates Inc.   
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Xia, F.; Chi, E.; Le, Q. V.; Zhou, D.; et al. 2022. Chain-ofthought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35: 24824–24837.   
Xu, R.; Qi, Z.; Guo, Z.; Wang, C.; Wang, H.; Zhang, Y.; and Xu, W. 2024a. Knowledge Conflicts for LLMs: A Survey. In Al-Onaizan, Y.; Bansal, M.; and Chen, Y.-N., eds., Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 8541–8565. Miami, Florida, USA: Association for Computational Linguistics.   
Xu, Y.; He, S.; Chen, J.; Wang, Z.; Song, Y.; Tong, H.; Liu, G.; Zhao, J.; and Liu, K. 2024b. Generate-on-Graph: Treat LLM as both Agent and KG for Incomplete Knowledge Graph Question Answering. In Al-Onaizan, Y.; Bansal, M.; and Chen, Y.-N., eds., Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 18410–18430. Miami, Florida, USA: Association for Computational Linguistics.   
Xue, T.; Wang, Z.; Wang, Z.; Han, C.; Yu, P.; and Ji, H. 2023. Rcot: Detecting and rectifying factual inconsistency in reasoning by reversing chain-of-thought. arXiv preprint arXiv:2305.11499.   
Yao, S.; Yu, D.; Zhao, J.; Shafran, I.; Griffiths, T.; Cao, Y.; and Narasimhan, K. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems, 36: 11809–11822. Yi, Z.; Ouyang, J.; Liu, Y.; Liao, T.; Xu, Z.; and Shen, Y. 2024. A survey on recent advances in llm-based multi-turn dialogue systems. arXiv preprint arXiv:2402.18013.   
Zhao, Z.; Monti, E.; Lehmann, J.; and Assem, H. 2024. Enhancing Contextual Understanding in Large Language Models through Contrastive Decoding. In Duh, K.; Gomez, H.; and Bethard, S., eds., Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 4225–4237. Mexico City, Mexico: Association for Computational Linguistics.   
Zhong, W.; Guo, L.; Gao, Q.; Ye, H.; and Wang, Y. 2024. MemoryBank: enhancing large language models with longterm memory. In Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence, AAAI’24/IAAI’24/EAAI’24. AAAI Press. ISBN 978-1-57735-887-9.

# Appendix A: Algorithm Descriptions and Pseudocode

# KGE Pipeline for Knowledge Graph Generation

The KGE pipeline is a neuro-symbolic process that transforms a natural language statement $s _ { t }$ from a dialogue turn into a formal, OWL-compliant knowledge graph fragment $\mathcal { G } _ { t } ^ { \prime }$ . This is a critical step in the Memory Maintenance Phase. The pipeline first parses the text into an Abstract Meaning Representation (AMR) to capture its core semantic structure. This AMR is then converted into a preliminary OWL graph. Finally, a semantic enrichment stage enhances the graph’s logical rigor by linking ambiguous entities to formal lexical resources and high-level ontologies using Word Sense Disambiguation (WSD). The resulting grounded fragment is then ready for integration into the main DSM.

Algorithm 1: KGE Pipeline for Knowledge Graph Generation

Input: Text T

Output: RDF/OWL Knowledge Graph G

1: {Module 1: Text to Abstract Meaning Representation Parsing}   
2: {Parse text into AMR form}   
3: GAMR ← SPRING(T)   
4: {Module 2: AMR to OWL KG Translation}  
5: {Transform AMR to an initial OWL-compliant RDF KG }   
6: $\mathcal { G } \gets \mathrm { A M R 2 F R E D } ( \mathcal { G } _ { A M R } )$   
7: {Semantic Enrichment}   
8: for each node $n$ in $\mathcal { G } _ { A M R }$ that lacks a link to a lexical resource do   
9: {Perform Word Sense Disambiguation (WSD)}   
10: {Get word synset from Framester via EWISER }   
11: Ssynset ← EWISER(T, lemma(n))   
12: G ← G ∪ {(n, owl:equivalentClass, Ssynset)}   
13: {Align to top-level ontologies}   
14: {Get word “supersense” }   
15: Ssupersense ← Framester.getSupersense(Ssynset)   
16: G ← G ∪ {(n, rdfs:subClassOf, Ssupersense)}   
17: {Get DOLCE+DnS Ultra Lite (DUL) class }   
18: CDUL ← Framester.getDULClass(Ssynset)   
19: G ← G ∪ {(n, rdfs:subClassOf, CDUL)}   
20: end for   
21: return G

# Beam Search for Reasoning Tree Traversal

Algorithm 2 operationalizes the Reasoning Tree (RT) traversal, employing a beam search to find a grounded response $r _ { t }$ for a query $q _ { t }$ by reasoning over the memory graph $\mathcal { G } _ { t - 1 }$ . The search iteratively expands a tree of reasoning paths. At each step, for the most promising states on the beam, an LLM acting as a policy $( p _ { \theta } ^ { \pi } )$ proposes actions (e.g., EXPAND ENTITY, Think). Actions involving graph traversal are executed deterministically against the DSM to ensure factual grounding. An LLM acting as a value function $( p _ { \theta } ^ { v } )$ then scores the newly generated states. The beam

is subsequently pruned to retain only the top- $k$ candidates, which balances exploration of multiple reasoning paths with computational tractability. The process terminates upon generating a final answer or reaching a maximum depth, with the response extracted from the highest-scoring trajectory.

Algorithm 2: Beam Search for Reasoning Tree Traversal

Input: Query $q _ { t }$ , Dynamic Structured Memory $\mathcal { G } _ { t - 1 }$ , beam width $k$ , max depth $D _ { m a x }$

Output: Final response $r _ { t }$

1: Initialize root state $S _ { 0 } \gets ( \tilde { \mathcal { G } } _ { 0 } = \emptyset , \mathcal { Z } _ { 0 } = \langle \rangle , v _ { 0 } =$ $1 . 0 , d _ { 0 } = 0 \big /$ )   
2: Initialize beam $B  \{ S _ { 0 } \}$   
3: for $d = 0$ to $D _ { m a x } - 1$ do   
4: Initialize candidate pool $C \gets \emptyset$   
5: for each state $s _ { i }$ in $B$ do   
6: if $S _ { i }$ is not a terminal state then   
7: Propose actions $A _ { c a n d } \sim p _ { \theta } ^ { \pi } ( a \mid q _ { t } , S _ { i } )$   
8: for each action $a _ { i j }$ in $A _ { c a n d }$ do   
9: Execute $a _ { i j }$ on $s _ { i }$ to get a new state $S _ { j }$   
10: {e.g., query $\mathcal { G } _ { t - 1 }$ for $a _ { e e } , a _ { f p } \}$   
11: Evaluate new state: $v _ { j } \sim p _ { \theta } ^ { v } ( v \mid q _ { t } , S _ { j } )$   
12: Add $S _ { j }$ to $C$   
13: end for   
else   
15: Add $s _ { i }$ to $C$ {Keep terminal states}   
end if   
17: end for   
18: Sort $C$ by value scores $v _ { j }$ in descending order   
19: Update beam $B \gets$ top- $k$ states from $C$   
20: if all states in $B$ are terminal then   
break   
22: end if   
23: end for   
24: Select final state $S ^ { * } = \operatorname { a r g m a x } v$ $\quad - \sp { \mathrm { ~ a ~ . ~ } } \bar { \Sigma } \sp { \mathrm { ~ a ~ . ~ } } \scriptscriptstyle { S }$   
25: Extract response $r _ { t }$ from trajectory $\mathcal { Z } ^ { \ast }$ of $S ^ { * }$   
26: return $r _ { t }$

# Appendix B: Implementation and Computational Cost Analysis

To ensure the transparency and reproducibility of our research, this section provides a detailed account of our experimental setup, including the hardware and software environment, hyperparameter configurations for the D-SMART framework, implementation details for baseline models, and a comprehensive analysis of computational costs.

# Environment and Dependencies

All experiments were conducted on the Ubuntu 22.04 operating system. Our framework is implemented using PyTorch and the HuggingFace Transformers library for model handling. The DSM is managed and maintained using a suite of specialized libraries for knowledge graph operations, including RDFlib, NetworkX, owlrl, and owlready2. For proprietary models, we utilized the official OpenAI API. The specific versions of all libraries are detailed in the provided code repository.

The hardware infrastructure was distributed as follows to optimize our experimental workflow:

• Local Model Execution: All open-source models (e.g., Qwen-8B, COMEDY-13B) were run on a server equipped with an NVIDIA H800 GPU, a 20-core Intel(R) Xeon(R) Platinum 8458P CPU, and 100GB of RAM.   
• API-based Model Execution: Experiments involving API calls (e.g., GPT-4o, Mem0, MemoryBank) were conducted on a machine with an NVIDIA GeForce RTX 4090 GPU, a 32-core 13th Gen Intel(R) Core(TM) i9- 13900K CPU, and 128GB of RAM.   
• Evaluation: All evaluation processes, including the execution of NLI-based metrics, were performed on a workstation with an NVIDIA GeForce RTX 3080 GPU, a 20- core Intel(R) Core(TM) i9-10900K CPU, and 96GB of RAM.

To ensure full reproducibility, we will release our full code and experimental configurations at Github.

# Baseline and Framework Implementation Details

D-SMART Hyperparameter Configuration. The performance of the D-SMART framework is influenced by the parameters governing the RT search process. For all main experiments reported in this paper, we adopted the following configuration, which was determined through a sensitivity analysis detailed below:

• Beam Search Width $( B )$ : 3   
• Action Samples per Node (S): 3   
• Maximum Search Depth (Dmax): 5   
• Maximum Expansion Steps (Mexp): 12

Parameter Sensitivity Analysis. To understand the impact of key hyperparameters on performance, we conducted a sensitivity analysis on a randomly sampled subset of 100 dialogues from MT-Bench-101. We designed seven distinct configurations to systematically evaluate how reasoning depth $( D _ { m a x } )$ , beam search width $( B )$ , and the number

of action samples $( S )$ affect both dialogue consistency (Consistency Score) and overall quality (GPT Score).

The results, as illustrated in Figure 7, shows that increasing the reasoning depth $D _ { m a x }$ from 1 to 5, cf. Conf-1, 2, 3) consistently improves both quality and consistency. A deeper search allows the framework to form more complex reasoning chains. We observed an interesting pattern where the framework often uses a THINK action to generate a preliminary answer draft before the final ANSWER action. This two-step process—first establishing a logically sound core, then refining its linguistic expression—appears to foster responses that are both coherent and articulate.

![](images/9132a188aafa42d51c105f32938b56310f946b4532bdf91be157aafc17d343a3.jpg)  
Figure 7: GPT Score and Consistency Score under different hyperparameters.

Similarly, widening the beam search $B$ from 1 to 3, cf. Conf-3 vs. Conf-5) yields significant gains. A wider beam enables the RT to explore multiple reasoning paths in parallel. Some paths might prioritize linguistic fluency, while others may ensure factual grounding through more frequent interactions with the DSM (e.g., via EXPAND ENTITY). The beam search mechanism allows the system to systematically compare these trajectories and select the one that is most logically consistent and grounded in the established dialogue facts, mitigating the risk of contradiction inherent in singlethreaded reasoning.

Based on this analysis, we selected Conf-5 $( D _ { m a x } =$ $5 , B \ = \ 3 , S \ = \ 3 )$ as our final configuration for all main experiments. This setting achieves a strong balance between high performance and manageable computational overhead.

Large Language Model (LLM) API Parameters. To ensure deterministic and reproducible results where necessary, we carefully controlled the LLM API parameters.

• Temperature: For the state evaluation step within the RT, where deterministic scoring is crucial, the temperature was set to 0.0. For all other generation tasks, a temperature of 0.8 was used to encourage more diverse and natural-sounding responses.   
• Max Tokens: The maximum number of generated tokens was set to 8192.   
• Model Versions: We used the specific model versions gpt-4o-2024-05-13 and qwen3-8b to ensure that

our results can be precisely replicated in future studies. All other parameters were left at their default settings.

Baseline Implementation. We used the official code implementations for all baseline models and frameworks to ensure a fair and accurate comparison.

• Mem0 & MemoryBank: We adapted the official code by replacing the original backbone LLM with $\mathsf { G P T - 4 o }$ , aligning it with our D-SMART setup. Key internal parameters were kept at the default values recommended in their respective papers. Our observations indicate that Mem0’s reliance on the LLM for single-step decisions over unstructured text limits its ability to maintain longterm consistency with complex or contradictory information. MemoryBank’s memory update mechanism, inspired by the Ebbinghaus forgetting curve, is driven by temporal relevance rather than logical consistency, which can lead to the decay of crucial facts over extended dialogues.   
• COMEDY-13B: We utilized the officially released model weights and inference scripts. As an end-to-end framework, its compressive memory mechanism, while efficient, was observed to occasionally discard critical numerical or entity details during the compression process, which subsequently impaired its logical consistency on tasks requiring high precision.

# Computational Cost Analysis

Algorithmic Complexity Analysis. The computational overhead of D-SMART stems primarily from two phases: the Memory Maintenance Phase and the Response Phase. Let the key variables be defined as follows: $N$ is the total number of triples in the DSM, $N ^ { \prime }$ is the number of triples in a newly extracted graph fragment, $V$ and $E$ are the number of nodes and edges in the DSM, $T _ { l l m }$ is the average latency of a single LLM API call, $M _ { e x p }$ is the maximum expansion steps in the RT, $B$ is the beam width, and $S$ is the number of action samples.

• Memory Maintenance Phase: This phase involves knowledge extraction and graph updates.   
– Knowledge Fragment Extraction: The complexity is dominated by the KGE pipeline call, which we denote as $O ( T _ { k g e } )$ .   
– Conflict Detection: This relies on an LLM call to compare graph serializations. The complexity is approximately $\bar { O } ( N + N ^ { \prime } + T _ { l l m } )$ , where the LLM call is the main bottleneck.   
– Graph Merging: This involves set operations on the graph triples, with a complexity of roughly $O ( N +$ $N ^ { \prime }$ ).   
• Response Phase (Reasoning Tree Search): The complexity is governed by the beam search algorithm. The total complexity can be approximated as:

$$
\begin{array}{l} \text {C o m p l e x i t y} \approx M _ {e x p} \times \left[ O \left(N _ {\text {t r e e}} \times \log N _ {\text {t r e e}}\right) \right. \\ + B \times S \times \left(2 T _ {l l m} \right. \tag {6} \\ \left. + O (V + E)) \right] \\ \end{array}
$$

where $N _ { t r e e }$ is the number of nodes in the search tree. The term $O ( V + E )$ arises from the most expensive graph traversal action, FIND PATH, which invokes a shortestpath search on the entire graph. The $2 T _ { l l m }$ term accounts for the two LLM calls per expansion: one for action sampling and one for state evaluation. The parameters $M _ { e x p }$ , B, and $S$ are the primary levers for balancing performance and computational cost. We also note that the number of effective actions generated by the LLM is often less than or equal to $S$ , as duplicate actions may be proposed, slightly reducing the effective branching factor.

Empirical Latency Measurement. We measured the average end-to-end inference latency per dialogue turn on an H800 GPU and an 4090 GPU to provide a practical assessment of the overhead, as shown in Table 3.

Table 3: Average per-turn inference latency comparison. API-based models include network latency.   

<table><tr><td>Model</td><td>Baseline (s)</td><td>D-SMOTE (s)</td></tr><tr><td>Qwen3-8B</td><td>0.27</td><td>1.27</td></tr><tr><td>GPT-4o API</td><td>3.50</td><td>9.80</td></tr></table>

The overhead introduced by D-SMART is significant but represents a deliberate architectural trade-off for enhanced logical consistency and reliability. The latency increase in the Response Phase is primarily due to the multiple LLM calls for action sampling and state evaluation in the RT.

The Memory Maintenance Phase, while computationally intensive, can be optimized. Its average latency is approximately 6 seconds, broken down as follows:

• Structured Statement Generation: ${ \sim } 0 . 1 3 s$   
• Knowledge Fragment Extraction (KGE API): ${ \sim } 5 \mathrm { s }$   
• Conflict Detection & Merging: ${ \sim } 0 . 7 5 s$

Note that this entire maintenance phase can be executed asynchronously. It can process the completed turn $\left( q _ { t } , r _ { t } \right)$ while the user is preparing their next query $q _ { t + 1 }$ , thereby hiding most of this latency from the user and minimizing its impact on the interactive flow of the dialogue. Future work can further mitigate costs by developing more efficient search heuristics for the RT and employing smaller, specialized models for internal tasks like state evaluation.

# Appendix C: Details on the Ablation Study Rationale for Using a Refined Subset for Ablation Analysis

To rigorously dissect the individual contributions and the synergistic interplay of our core components, the DSM and the RT, we conducted the ablation study on a strategically curated subset of the MT-Bench-101 benchmark. A key motivation for this approach is to mitigate the confounding effects of performance saturation observed when evaluating on the full dataset.

Our preliminary analysis revealed that a significant portion of the MT-Bench-101 dialogues falls into two categories that limit the conclusiveness of ablation results:

• Ceiling Effect: Many tasks are simple enough that even baseline models can achieve high scores (Wang et al. 2019). In these scenarios, the performance is already saturated, leaving little to no room to measure the true uplift provided by components like DSM and RT.   
• Floor Effect: Conversely, some dialogues are exceedingly difficult or ill-posed, causing even the most advanced models to fail. In these cases, the models operate in a near-random-guess regime, and the impact of our framework’s components is masked by the fundamental difficulty of the task.

These effects can lead to an “Average Effect Masking,” where the true, significant impact of our components on moderately to highly complex tasks is diluted by the large number of non-informative cases. Therefore, to ensure a sensitive and meaningful evaluation, we designed a principled methodology to isolate a subset of dialogues that are both challenging and solvable, thus providing a clearer window into the components’ efficacy.

# Selection Methodology and Criteria

We curated this refined subset, which constitutes $2 5 \%$ of the original MT-Bench-101 dataset, through a two-stage, criteria-driven filtering process designed to maximize task complexity and discriminative power.

Stage 1: Performance-based Filtering. To eliminate dialogues susceptible to ceiling or floor effects, we first filtered the dataset based on the performance of the $\mathtt { Q w e n - 8 B }$ baseline model. We retained only those dialogues where the baseline model achieved a GPT score in a moderate range (e.g., between 4.0 and 8.0). This step effectively removes tasks that are either too trivial or prohibitively difficult for a standard model, ensuring our analysis focuses on dialogues where improvements are both possible and measurable.

Stage 2: Complexity-based Curation. From the pool of dialogues selected in Stage 1, we then ranked them based on a suite of linguistic and structural complexity metrics, similar to how Perez-Rosas et al. do. The goal was to identify ´ dialogues that place high demands on memory, reasoning, and context management. The primary criteria for this ranking included:

• Dialogue Scale and Information Density: A high Average Turns per Dialogue and Average Words per Turn to

test the model’s long-term memory and its ability to process information-dense utterances, which are core challenges for the DSM.

• Semantic Richness and Entity Management: A high Vocabulary Diversity (TTR) and Average Unique Entities per dialogue. These metrics identify conversations with broad topics and numerous entities, directly challenging the DSM’s capacity for knowledge construction and maintenance.   
• Logical and Structural Complexity: A high Conditional Statement Frequency and Negative Constraint Frequency. These metrics were chosen to specifically probe the model’s logical reasoning capabilities and its adherence to complex constraints, functions directly supported by the RT framework.   
• Context Dynamics: A high New Entity Introduction Rate, which signals a rapidly evolving conversational context. This tests the DSM’s dynamic update mechanism, including its ability to integrate new facts and resolve conflicts in real-time.

We selected the top-ranked dialogues from this process, finally we get a subset reached $2 5 \%$ of the original dataset’s size, we ensured the final set for the ablation study was composed of the most challenging and informative examples.

# Validation of the Refined Subset

To empirically validate the outcome of our selection methodology, we performed a statistical comparison between the curated subset and the original MT-Bench-101 dataset. As detailed in Table 4, the refined subset demonstrates significantly higher complexity across all measured dimensions. For instance, the average number of unique entities per dialogue nearly doubled (from 5.04 to 9.91), and the introduction rate of new entities per turn increased by over $74 \%$ (from 1.66 to 2.89). This confirms that our principled approach successfully isolated a more challenging testbed, making it ideal for a nuanced and insightful ablation analysis.

Table 4: Statistical comparison between the original MT-Bench-101 dataset and the refined, more challenging subset used for the ablation study. The refined subset shows consistently higher complexity across all metrics.   

<table><tr><td>Metric</td><td>Original</td><td>Refined</td></tr><tr><td>Avg. Turns per Dialogue</td><td>3.03</td><td>3.42</td></tr><tr><td>Avg. Words per Turn</td><td>66.3</td><td>82.99</td></tr><tr><td>Vocabulary Diversity (TTR)</td><td>6.12%</td><td>10.89%</td></tr><tr><td>Avg. Unique Entities</td><td>5.04</td><td>9.91</td></tr><tr><td>Conditional Freq.</td><td>16.96%</td><td>22.46%</td></tr><tr><td>Negative Constraint Freq.</td><td>12.36%</td><td>18.19%</td></tr><tr><td>New Entity Intro. Rate</td><td>1.66</td><td>2.89</td></tr></table>

# Appendix D: Benchmark Task Details and Case Studies

The MT-Bench-101 benchmark is structured around 13 distinct task categories, each formulated to rigorously evaluate specific facets of a model’s multi-turn dialogue capabilities, such as logical consistency and the faithful recall of long-term memory. To offer both a formal definition and a concrete understanding of the challenges posed by this benchmark, we first present a table detailing the full name, abbreviation, and description for each of the 13 tasks. Following this, we present dialogue examples (case studies) from key categories to illustrate how these tasks probe for specific capabilities in practice.

Note that in each case, light blue indicates information the model will recall in the future, light green indicates the model’s correct responses, light red indicates the model’s incorrect responses, and light orange indicates the contradictory or inconsistent content in the context. Each model or framework has $\ " \langle \bullet \mathrm { ‰ } { }$ on the left indicating that the NLI relation for that turn with the context was judged as ENTAILMENT, $\mathbf { \hat { \psi } } ( \mathbf { N } ) ^ { * }$ indicates NEUTRAL, and ${ \mathcal { \bullet } } ( { \mathcal { C } } ) ^ { \bullet }$ indicates CONTRADICTION.

# MT-Bench-101 Task Descriptions

Detailed task description of MT-Bench-101 are shown in Table 5.

Table 5: Detailed descriptions of the 13 tasks within MT-Bench-101.   

<table><tr><td>Task</td><td>Abbr.</td><td>Task Description</td></tr><tr><td>Context Memory</td><td>CM</td><td>Evaluates the model&#x27;s ability to recall details from earlier in the dialogue to address a user&#x27;s current question, assessing the consistency of its long-term memory and contextual continuity.</td></tr><tr><td>Anaphora Resolu-tion</td><td>AR</td><td>Assesses the model&#x27;s capacity to correctly resolve anaphoric references (e.g., pronouns like &quot;it&quot;, &quot;these&quot;) throughout a conversation, which is crucial for maintaining semantic consistency.</td></tr><tr><td>Separate Input</td><td>SI</td><td>Measures the model&#x27;s ability to maintain a consistent link between task instructions from an initial turn and the specific inputs provided in subsequent turns, a key test of operational consistency.</td></tr><tr><td>Topic Shift</td><td>TS</td><td>Tests the model&#x27;s agility in recognizing an abrupt topic switch from the user and maintaining conversational coherence by focusing on the new topic without being unduly influenced by prior context.</td></tr><tr><td>Content Confusion</td><td>CC</td><td>Probes the model&#x27;s skill in distinguishing between textually similar queries that have different meanings, testing its ability to avoid interference and maintain logical consistency in its responses.</td></tr><tr><td>Content Rephras-ing</td><td>CR</td><td>Requires the model to rephrase its previous response based on a new user requirement (e.g., simplification). This tests its ability to maintain informational consistency while altering the presentation.</td></tr><tr><td>Format Rephrasing</td><td>FR</td><td>Involves transforming the structure of a previous response (e.g., from paragraph to list) while preserving the original information, assessing its ability to maintain content fidelity across formats.</td></tr><tr><td>Self-correction</td><td>SC</td><td>Assesses the model&#x27;s ability to revise an erroneous response after receiving valid user feedback, thereby restoring consistency with factual accuracy.</td></tr><tr><td>Self-affirmation</td><td>SA</td><td>Evaluates the model&#x27;s capacity to adhere to its correct response when faced with an incorrect user challenge, a test of its confidence and consistency of knowledge.</td></tr><tr><td>Mathematical Rea-sonig</td><td>MR</td><td>Tests collaborative problem-solving where the model must maintain logical consistency while incrementally incorporating new conditions across turns to solve a complex mathematical problem.</td></tr><tr><td>General Reasoning</td><td>GR</td><td>Assesses the ability to work with users over successive turns to solve complex reasoning puzzles, which requires maintaining a consistent line of logical deduction or induction.</td></tr><tr><td>Instruction Clarifi-cation</td><td>IC</td><td>Targets the model&#x27;s ability to handle ambiguous queries by asking clarifying questions, a strategy to ensure the subsequent dialogue is consistent with the user&#x27;s true intent.</td></tr><tr><td>Proactive Interac-tion</td><td>PI</td><td>Measures the ability to pose relevant questions in response to user statements, aiming to maintain a coherent and engaging dialogue rather than letting the conversation terminate prematurely.</td></tr></table>

# Case Studies

Case.1: This case study illustrates the difference in factural consistency between D-SMART and a representative baseline framework (MemoryBank) when tasked with rephrasing and reformatting information across multiple turns. While both frameworks produce high-quality, fluent responses, MemoryBank introduces a subtle but significant logical conflict, a failure that D-SMART’s architecture is designed to prevent. Initially, both frameworks identify several risks of AI. D-SMART grounds its subsequent responses in the seven points it first established, maintaining informational integrity, as reflected by its high Consistency Scores (CS). In contrast, MemoryBank demonstrates a contextual inconsistency. In its first response, “developers” are framed as part of a collective group responsible for addressing AI risks. However, in its final turn, it introduces new, ungrounded information by stating that AI systems can reflect “the biases of their developers”, positioning developers as a source of the problem. This shift creates a conflict with the previously established information. This inconsistency is captured by its low CS of 0.57, even as it receives a high GPT Score (GS) of 10 for its fluency and surface-level correctness. This example highlights how D-SMART ensures that responses remain factural consistent. Baselines lacking such a mechanism may fail to preserve consistency, especially in nuanced aspects of the dialogue history, a flaw that can be overlooked by holistic quality metrics like the GPT Score.

# Case .1: Case.1: Maintaining Factual Consistency Across Dialogue Turns

# Turn 1

User: What are the potential risks of artificial intelligence?

Assistant:

![](images/d458acccb16d90d6fe5496924b67d8c98e2ee4f8236160f6667bceab6b5534be.jpg)

# D-SMART $( \mathbf { C S } _ { 1 } = 0 . 6 9 )$

Artificial intelligence (AI) has several potential risks, including:

1. Job Loss: Automation through AI could replace human workers in various industries, leading to unemployment and economic challenges.   
2. Bias and Discrimination: AI systems can inherit biases from the data they are trained on, leading to unfair or discriminatory decisions.   
3. Privacy Concerns: AI-powered technologies can collect and analyze vast amounts of personal data, raising concerns about surveillance and misuse of information.   
4. Security Risks: AI can be exploited for cyberattacks, creating vulnerabilities in systems and potentially endangering critical infrastructure.   
5. Unintended Consequences: Poorly designed AI systems may behave unpredictably, causing harm or making irresponsible decisions.   
6. Ethical and Moral Issues: AI raises questions about accountability for actions, especially in areas like autonomous weapons or life-altering decisions.   
7. Concentration of Power: The development of AI could lead to increased power for a few companies or nations, potentially creating inequality and dependency.

# Turn 2

User: Can you rephrase your answer in simpler terms, focusing on the main ideas?

Assistant:

![](images/71eecd7d3bf7a950524eee4b417570a0768d92d461d0e985069a9c17375cb58b.jpg)

# D-SMART $\mathbf { G S } _ { 2 } = 1 0$ , CS2 = 0.98)

AI can cause problems like replacing jobs, making unfair decisions due to biased training data, invading privacy by collecting too much personal information, being used in cyberattacks, acting in unpredictable ways, and raising ethical questions about accountability. It can also create inequalities if only a few powerful companies or countries control and benefit from it.

![](images/d341a1875ab78cf8bbe7b1d8e38443ee81b113bda9aff310deadb3472b9c3a44.jpg)

# MemoryBank $( \mathbf { C S } _ { 1 } = 0 . 5 1 $ )

Artificial intelligence has the potential to bring about many benefits, such as improving efficiency and advancing technological developments. However, there are also some risks associated with AI. Some concerns include the possibility of job displacement due to automation, the ethical implications of AI decision-making, privacy issues related to data collection and surveillance, and the potential for bias in AI algorithms.

It is important for researchers, developers, policymakers, and society as a whole

to work together to address these risks and ensure that AI is developed and deployed responsibly.

# Turn 3

User: Now, would you mind expressing those concerns in bullet points, each followed by a brief explanation?

Assistant:

![](images/128c5fe244361642cf4efaf3a3736a83d2fdbf8c1a1503f93d53405d5afc3e95.jpg)

# D-SMART $( \mathbf { G } \mathbf { S } _ { 3 } = 1 0 , \mathbf { C } \mathbf { S } _ { 3 } = 0 . 7 5 , \mathbf { D } \mathbf { E } \mathbf { R } = 6 6 . 7 \% )$ $\mathbf { G S } _ { 3 } = 1 0$ $\mathbf { C S } _ { 3 } = 0 . 7 5$

Here are the main concerns about artificial intelligence (AI, expressed simply):

- Job Loss: AI can automate tasks, which might replace people’s jobs and leave workers without employment opportunities.

- Unfair Decisions: AI systems can make biased choices if trained with unfair or discriminatory data, treating some groups unequally.

- Loss of Privacy: AI technology can collect too much personal information, potentially invading people’s privacy.

- Security Threats: AI can be misused to attack computer systems or infrastructure, causing damage or risking safety.

- Unexpected Behavior: Poorly designed AI might behave in ways that are unpredictable or harmful to people or systems.

- Ethical Problems: There are tricky issues about who is responsible when AI makes big mistakes or moral choices.

- Power Imbalances: Powerful AI might be controlled by only a few companies or countries, making others dependent or vulnerable.

These risks highlight the importance of careful development and oversight in AI technology.

![](images/6dff0ecd126831442f50fb02e053b7dd479c1c9ed0de6707cbd7a91d237eb419.jpg)

# MemoryBank (GS2 = 8, CS2 = 0.72)

Sure! Artificial intelligence can bring good things, like making things easier and creating new technology. But it can also have some problems. For example, it might take away people’s jobs, make decisions that aren’t fair, invade privacy by collecting too much data, or be biased. It’s important for everyone to work together to make sure AI is used in a good and fair way.

![](images/de401a15aa31a5613f00e716f48b1f36e5cb14d98a9ac3f860f50b49b5b35f9f.jpg)

# MemoryBank $( \mathbf { G S } _ { 3 } = 1 0 \small { < } ! >$ , $\mathrm { C S } _ { 3 } = 0 . 5 7$ , DER = 33.3%)

- Job displacement: AI could automate tasks that are currently done by people, leading to job losses in certain industries.

- Ethical implications: AI systems may make decisions that raise moral questions, such as who is responsible if something goes wrong.   
- Privacy concerns: AI’s ability to collect and analyze large amounts of data raises concerns about how this information is used and protected.   
- Bias in algorithms: AI systems can reflect the biases of their developers or the data they are trained on, potentially leading to discriminatory outcomes.

Case.2: This case illustrates D-SMART’s ability to maintain factual consistency and traceability throughout a multi-turn mathematical reasoning dialogue. The critical divergence appears in the second turn. The query requires recalling the initial dataset to perform a new calculation. D-SMART successfully accesses its structured memory of the dialogue history to reason through the steps and arrive at the correct, verifiable answer. In contrast, GPT-4o provides an incorrect and unsubstantiated answer, a classic example where the model fails to faithfully track the established facts from previous turns. In the third turn, while both models provide the correct final answer, D-SMART’s response is explicitly grounded in the results generated in the first turn, making its reasoning process transparent and trustworthy. GPT-4o’s correct response, however, is opaque. Following its error in the previous turn, it is impossible to determine if this correct answer stems from a reliable retrieval of prior information or an independent (and in this case, fortunate) recalculation. This example demonstrates how D-SMART ensures that responses are not only accurate but are also derived from a consistent and traceable line of reasoning based on the evolving dialogue context.

# Case .2: Case.2: Traceability and Consistency in Multi-Turn Mathematical Reasoning

Turn 1 User: In a physical education class, sixth-grade girls are doing a sit-up test. The standard is set at 28 sit-ups. Exceeding the standard is indicated by a positive number, and falling short is indicated by a negative number. The results of 10 girls are recorded as follows: $- 2 , + 5 , - 1 , 0 , + 1 , + 3 , 0 , + 8 , + 1 , + 6 .$ What are the actual number of sit-ups completed by each of the 10 girls?

Assistant:

$\textcircled{6}$ D-SMART (GS1 = 10, CS1 = 0.95)

To find the actual number of sit-ups completed by each girl, we need to add the recorded number (which indicates how much they exceeded or fell short of the standard) to the standard of 28 sit-ups.

Here is the calculation for each girl:

1. −2 + 28 = 26

2. +5 + 28 = 33

3. −1 + 28 = 27

4. 0 + 28 = 28

5. +1 + 28 = 29

6. +3 + 28 = 31

7. 0 + 28 = 28

8. +8 + 28 = 36

9. +1 + 28 = 29

10. +6 + 28 = 34

Final Answer:The actual number of sit-ups completed by each girl is:

26, 33, 27, 28, 29, 31, 28, 36, 29, 34.

$\mathbf { \textcircled { E } } \mathbf { G P T 4 4 0 } \left( \mathbf { G S } _ { 1 } = 1 0 , \mathbf { C S } _ { 1 } = 0 . 8 7 \right)$

To find the actual number of sit-ups completed by each of the 10 girls, we need to add the results recorded to the standard of 28 sit-ups.

Here are the actual sit-up counts for each girl based on their recorded results:

1. -2 sit-ups: $2 8 - 2 = 2 6$

2. +5 sit-ups: $2 8 + 5 = 3 3$

3. -1 sit-up: 28 - 1 = 27

4. 0 sit-ups: $2 8 + 0 = 2 8$

5. +1 sit-up: $2 8 + 1 = 2 9$

6. +3 sit-ups: $2 8 + 3 = 3 1$

7. 0 sit-ups: $2 8 + 0 = 2 8$

8. +8 sit-ups $2 8 + 8 = 3 6$

9. +1 sit-up: $2 8 + 1 = 2 9$

10. +6 sit-ups: $2 8 + 6 = 3 4$

So, the actual number of sit-ups completed by each of the 10 girls is:

26, 33, 27, 28, 29, 31, 28, 36, 29, 34.

Turn 2 User: How many girls met or exceeded the standard, and what percentage of the total does this represent?

Assistant:

$\textcircled{6}$ D-SMART (GS2 = 10, CS2 = 0.96)

To determine how many girls met or exceeded the standard, we look for recorded numbers that are 0 or positive (since exceeding the standard is a positive number, and meeting the standard is 0).

From the list of results: -2, +5, -1, 0, +1, +3, 0, +8, +1, +6

The non-negative values are: +5, 0, +1, +3, 0, +8, +1, +6

That is 8 girls who met or exceeded the standard.

Now, to find the percentage of the total:

$$
\mathrm{Percentage} = \left(\frac{8}{10}\right)\times 100 = 80\%
$$

Final Answer:

8 girls met or exceeded the standard, which represents $80 \%$ of the total.

Turn 3 User: Among the 10 girls, who did the most and the least number of sit-ups?

Assistant:

$\textcircled{6}$ D-SMART $( \mathbf { G } \mathbf { S } _ { 3 } = 1 0 , \mathbf { C } \mathbf { S } _ { 3 } = 0 . 8 4 , \mathbf { D } \mathbf { E } \mathbf { R } = 1 0 0 \% )$ $\mathbf { G S _ { 3 } } = 1 0$

To determine who did the most and the least number of sit-ups, we refer to the actual number of sit-ups completed by each girl, which we previously calculated as:

26, 33, 27, 28, 29, 31, 28, 36, 29, 34

- The most sit-ups were done by the girl who completed 36 sit-ups.

- The least sit-ups were done by the girl who completed 26 sit-ups. Final Answer: The girl who did the most sit-ups completed 36, and the girl who did the least completed 26.

$\mathrm { \textcircled { C } G P T 4 0 \ ( G S _ { 2 } = 2 , C S _ { 2 } = 0 . 3 4 ) }$

Out of the 10 girls, 6 girls met or exceeded the standard of 28 sit-ups. This represents $6 0 \%$ of the total.

$\textcircled{1}$ GPT-4o (GS3 = 9, CS3 = 0.72, DER = 66.7%)

The girl who did the most sit-ups completed 36 sit-ups, while the girl who did the least completed 26 sit-ups.

Case.3: This case study illustrates the performance contrast between D-SMART and another baseline framework (Mem0) in a multi-turn logical reasoning scenario, particularly focusing on consistency as new constraints are introduced. Initially, the Mem0 provides the correct answers for the first two turns but does so without presenting any explicit reasoning steps. However, its performance degrades in the final turn upon the introduction of a new global rule. The Mem0’s response exhibits a clear self-contradiction, as it generates a conclusion that directly violates a core premise established in the first turn. This phenomenon of “logical decay” shows the framework’s inability to maintain consistency with the conversational context when faced with evolving information. In contrast, D-SMART consistently builds its responses upon the established dialogue history. It successfully recalls prior deductions and correctly integrates the new rule in the final turn, methodically verifying that the new constraint does not conflict with the already-derived conclusions. This case highlights D-SMART’s ability to maintain a coherent and robust line of reasoning across a multi-turn interaction, a critical challenge where the baseline approach falters.

# Case .3: Case.3: Logical Reasoning with Evolving Constraints Pt.1

# Turn 1

User: There are three rooms. Room 1 is either occupied or not Room 2 is either occupied or not If Room 1 is occupied, then Room 2 is not If Room 2 is occupied, then Room 1 is not checked Room 1 and it was occupied . What can you infer about Room 2?

Assistant:

$\textcircled{6}$ D-SMART (GS1 = 10, CS1 = 0.99)

Given the information:

1. Room 1 is occupied   
2. If Room 1 is occupied, then Room 2 is not occupied   
3. If Room 2 is occupied, then Room 1 is not occupied

From the first statement, we know Room 1 is occupied

Now, applying the second condition:

“ If Room 1 is occupied, then Room 2 is not occupied. ”

Since Room 1 is occupied, it follows that Room 2 must not be occupied.

Therefore, the inference is: Room 2 is not occupied.

$\mathrm { \textcircled { E } M e m 0 \ : ( G S _ { 1 } = 8 , C S _ { 1 } = 0 . 6 5 ) }$

Room 2 is not occupied.

# Turn 2

User: I forgot to mention, there’s also a Room 3. Room 3 is occupied only if Room 2 is not . Now, is Room 3 occupied or not?

Assistant:

$\textcircled{6}$ D-SMART (GS2 = 9, CS2 = 0.97)

Given the updated information:

Room 1 is occupied (from the previous question).

If Room 1 is occupied, then Room 2 is not occupied Therefore, Room 2

is not occupied.

Room 3 is occupied only if Room 2 is not This means:

- If Room 2 is not occupied, then Room 3 may be occupied (but it’s not guaranteed).

- If Room 2 is occupied, then Room 3 is not occupied.

Since we have already concluded that Room 2 is not occupied , the condition for

Room 3 being occupied is satisfied.

So, the answer is: Room 3 is occupied.

(Continued on the next page.)

$\mathrm { \textcircled { E } M e m 0 \ : ( G S _ { 2 } = 8 , C S _ { 2 } = 0 . 6 9 ) }$

Room 3 is occupied.

# Case .4: Case.3: Logical Reasoning with Evolving Constraints Pt.2

# (Continued from the previous page.)

#

User: What if I told you that there’s a rule that not all rooms can be occupied . Does this change your conclusion about the rooms?

Assistant:

$\textcircled{6}$ D-SMART $\mathbf { G S } _ { 3 } = 1 0$ , CS3 = 0.99, DER = 100%)

Let’s analyze the situation with the new rule:

Given:

Room 1 is occupied

If Room 1 is occupied, then Room 2 is not occupied.   
If Room 2 is occupied, then Room 1 is not occupied.   
Room 3 is occupied only if Room 2 is not occupied.

- New rule: Not all rooms can be occupied.

Step-by-step deduction:

1. Room 1 is occupied.   
2. From the first conditional:

If Room 1 is occupied, then Room 2 is not occupied.

→ So, Room 2 is not occupied.

3. From the second conditional:

If Room 2 is occupied, then Room 1 is not occupied.

This is already satisfied because Room 2 is not occupied.

4. From the third rule:

Room 3 is occupied only if Room 2 is not occupied.

Since Room 2 is not occupied, Room 3 can be occupied (this is allowed).

5. So far, we have:

Room 1: occupied   
Room 2: not occupied   
Room 3: possibly occupied (but not guaranteed)

Now, the new rule says:

> Not all rooms can be occupied.

This means at least one room must be unoccupied.

But Room 1 is already occupied, and Room 2 is not, so at least one room is unoccupied (Room 2). That already satisfies the new rule.

So, the new rule does not change our conclusion about the occupancy of the rooms, as the condition is already met.

Final Answer: Room 3 is occupied.

And the new rule (not all rooms can be occupied) is already satisfied, because Room

2 is unoccupied.

$\textcircled { \mathrm { C } } \mathbf { M e m 0 } \left( \mathbf { G S } _ { 3 } = 4 , \mathbf { C S } _ { 3 } = 0 . 6 3 , \mathbf { D E R } = 6 6 . 7 \% \right)$

Yes, this changes the conclusion

If not all rooms can be occupied and Room 1 is occupied, then

Room 2 is occupied , and therefore Room 3 cannot be occupied either , as

it is only occupied if Room 2 is not . Thus, Room 3 is also not occupied

# Appendix E: LLM Prompt Templates

# Structured Statement Generation

### Task: Combine the user’s question and the AI’s answer into a single factual statement that preserves all key information while maintaining grammatical correctness. Follow these guidelines:

1. Begin with the answer’s subject as the statement subject   
2. Remove interrogative phrasing while retaining the core inquiry   
3. Use causal connectors (“because”, “since”, “as”) for explanatory relationships   
4. Maintain technical terms and precise quantities from the original answer   
5. Use ONLY information from the Question and the Answer unless complex concepts require more   
6. STRICTLY follow the example format.   
7. Output ONLY the final statement without explanations/notes

### Format:

Question: [User’s question]

Answer: [LLM’s response]

Statement: [Your synthesized statement]

### Examples:

Question: “What causes auroras?”

Answer: “Auroras occur when charged particles from the sun collide with Earth’s atmosphere.”

Statement: Auroras are caused by solar charged particles colliding with Earth’s atmosphere.

Question: “How much caffeine is in espresso?”

Answer: “A typical 1-ounce espresso shot contains about $6 3 \mathrm { m g }$ of caffeine.”

Statement: A 1-ounce espresso shot contains approximately 63 milligrams of caffeine.

### Now process:

Question: “{last question}”

Answer: “{last answer}”

Statement:

Figure 8: The prompt for the structured statement generation.

You are an expert in RDF (Resource Description Framework) and semantic web technologies. Your task is to detect semantic conflicts when merging a new RDF dataset into an original RDF dataset.

### Understanding RDF Entity Identity

#### Entity References in RDF

In RDF, the same real-world entity can be referred to by multiple URIs or identifiers:

1. Different URIs can refer to the same entity

- Domain-specific identifiers (e.g., domain.owl#person 1)   
- DBpedia URIs (e.g., dbpedia.org/resource/Person)   
- Wikidata URIs (e.g., wikidata.org/entity/Q123)

2. These different references do not create conflicts

- They are different ways to identify the same underlying entity   
- Properties linking these references (like owl:sameAs) help establish identity

3. The identity relationship is transitive

- If A refers to the same entity as B, and B refers to the same entity as C   
- Then A, B, and C all refer to the same entity

### Definitions of Semantic Conflicts in RDF

#### Semantic Conflicts

Conflicts require logical incompatibility between actual facts about the entity, not between different references to the same entity. A conflict exists ONLY when:

1. Property Cardinality Conflict (Primary Focus):

- The same real-world entity has multiple incompatible values for a functional property

- Examples:

- The same person (regardless of how they are identified) has two different birth dates   
- The same book (across different identifiers) has two different ISBN numbers

- Examples of functional properties (but not limited to):

- hasSpouse (a person can only have one legal spouse at a time)   
- hasBirthDate (a person can only have one birth date)   
- hasPassportNumber (one passport can only have one unique number)

2. Logical Contradiction:

- Contradictory facts about the same entity (considering all its identifiers):   
- Mutual exclusion (e.g., ‘status‘ cannot be both “alive” and “deceased”)   
- Numerical contradiction (e.g., ex:age 30 vs 40)   
- Temporal paradox (e.g., ex:deathDate before ex:birthDate)

3. Ontology Incompatibility:

- Different ontologies define the same URI with incompatible class/property constraints

4. Namespace Collision:

- Same URI maps to different resources due to namespace prefix redefinition   
- This applies to subjects, predicates, and objects with prefixes

Figure 9: The prompt for the conflict resolution pt.1

# #### Non-Conflicts

The following cases are not considered conflicts:

1. Facts present in origin rdf but missing in new rdf   
2. New facts in new rdf that don’t contradict origin rdf   
3. Multiple values for non-functional properties, such as:

- likes (a person can like multiple things)   
- hasChild (a person can have multiple children)   
- hasHobby (a person can have multiple hobbies)   
- worksFor (a person can work for multiple organizations)

4. Additional descriptions that enrich rather than contradict existing knowledge   
5. Multiple identifiers or references to the same entity   
6. Different ways of expressing the same fact about an entity

# ### Task

You are given two RDF datasets:

- origin rdf: The original RDF dataset.   
- new rdf: The new RDF dataset to be merged into the original.

Your task is to:

1. Analyze both datasets to determine if there are any semantic conflicts between new rdf and origin rdf based on the definitions provided.

- Consider all references to the same entity as a single logical entity   
- Look for true semantic conflicts in the facts about entities   
- Identify conflicts only when the actual information about an entity is incompatible

2. Identify the conflicting relationships in the origin rdf that cause these conflicts with new rdf.   
3. Output only the conflicting triples from origin rdf in Turtle syntax, using the format below.

# ### Output Format

- If there are conflicts, output the conflicting triples from origin rdf:

```markdown
```
turtle
    <subject1> <predicate1> <object1>
    <subject2> <predicate2> <object2>
    ...
</ turtle 
```

- If there are no conflicts, output an empty code block:

```txt
turtle 
```

# ### Input

- origin rdf: {origin rdf}   
- new rdf: {new rdf}

Please proceed with the analysis and provide the output as specified.

# ### Important Instructions

1. Always check and output the full URIs for all prefixed terms, if the subject/predicate/object of a triples has a prefix, add the full URI corresponding to this prefix at the very beginning, and the subject/predicate/object must be enclosed in $\langle \rangle$ .   
2. Focus on conflicts in actual facts about entities, not their identifiers   
3. Consider all references to the same entity as equivalent when checking for conflicts   
4. Do not consider different identifiers for the same entity as conflicts   
5. Do not provide any analysis or explanations outside the specified output format   
6. Only output conflicting triples from origin rdf

Figure 10: The prompt for the conflict resolution pt.2

You are a reasoning agent designed to answer questions by navigating a knowledge graph. Your task is to analyze the provided information and choose the single best next action to find the answer.

Question: {query}

Known Information (Facts from Knowledge Graph): {triples}

Reasoning Path (Actions already taken): {trajectory}

Your Task: Based on the information above, select the most appropriate next action from the list below.

# Available Actions:

1. EXPAND ENTITY [Entity]   
- Purpose: To gather more facts connected to a single entity.   
- When to use: Choose this if the ‘Known Information’ is insufficient to answer the question and you need more details about one specific entity.   
2. FIND PATH [Entity1] [Entity2]   
- Purpose: To find a connection or relationship between two entities.   
When to use: Choose this if answering the question requires understanding the relationship between two entities and that path is currently missing.   
3. THINK   
- Purpose: To pause, analyze, and synthesize the collected information to form a hypothesis or a plan.   
- When to use: Choose this when you have gathered some information but need to structure your thoughts or decide what to do next before taking another action like EXPAND ENTITY or ANSWER.   
- Output Format: Answer in form like “THINK: ...”, “...” means the contents after thinking.   
4. ANSWER   
- Purpose: To provide the final answer to the question.   
- When to use: Choose this **only** when you are confident that the ‘Known Information’ is sufficient to definitively answer the question.

Instruction: Respond with a single line containing only the action and its required parameters (e.g., ‘EXPAND ENTITY [John]‘). Do not provide any explanations.

Next Action:

Figure 11: The prompt for the RT action sampling

Evaluate the quality of the following answer and assign a single decimal score between 0.0 (worst) and 1.0 (best) according to the three weighted criteria below. After your reasoning, only output the score without any additional explanations or comments.

Question: {query} Answer: {answer}

Known Facts: {triples}

# Evaluation Criteria & Weights:

1. Accuracy $( 3 3 \% )$ : Does the answer correctly reflect the known facts and logical relationships?

- $0 . 0 =$ completely false or unsupported   
- $0 . 5 = { }$ partially correct but with errors   
- $1 . 0 =$ fully correct and evidence-based

2. Completeness $( 3 3 \% )$ : Does it fully address every aspect of the question?

$0 . 0 =$ misses most sub-questions   
$0 . 5 = { }$ addresses some parts, omits others   
$1 . 0 =$ covers all parts thoroughly

3. Clarity $( 3 3 \% )$ : Is the explanation clear, well-structured, and free of ambiguity?

- $0 . 0 =$ confusing or unreadable   
- $0 . 5 = { }$ understandable but needs improvement   
- $1 . 0 =$ crystal-clear, concise, and well organized

# Scoring instructions:

- For each criterion, choose a sub-score between 0.0 and 1.0, round to two decimals.   
- Compute the weighted sum:

‘final score $= 0 . 3 3 ~ ^ { * }$ Accuracy $+ 0 . 3 3 \ ^ { * }$ Completeness $+ 0 . 3 3 \ ^ { * }$ Clarity‘

- Round the final score to two decimals.

Please provide a score between 0.0 and 1.0 (1.0 being a perfect answer):

Score:

Figure 12: The prompt for the RT state evaluation (node of action “ANSWER”)

Assess the value of the current reasoning state toward solving the question, and output a single decimal score between 0.0 (worst) and 1.0 (best). Use the three weighted dimensions below. After your reasoning, only output the score without any additional explanations or comments.

Question: {query}

Known Facts: {triples}

Reasoning Process: {trajectory}

Evaluation Criteria & Weights:

1. Relevance $( 3 3 \% )$ : How directly the collected information pertains to answering the question.

- $0 . 0 =$ mostly irrelevant details   
- $0 . 5 = { }$ some relevant, some off-topic   
- $1 . 0 =$ entirely on-point

2. Progress $( 3 3 \% )$ : How much closer this state brings us toward a solution.

- $0 . 0 =$ no forward movement   
- $0 . 5 = { }$ partial progress, dead-ends remain   
- $1 . 0 =$ clear steps toward the answer

3. Potential $( 3 3 \% )$ : The likelihood that further development from this state will yield the correct answer.

- $0 . 0 =$ unlikely to help   
- $0 . 5 = { }$ uncertain but possible   
$1 . 0 =$ strong foundation for final solution

Scoring instructions:

- Assign each dimension a sub-score (0.0–1.0), rounded to two decimals.   
- Compute weighted sum:

‘final score $= 0 . 3 3 ~ ^ { * }$ Relevance + 0.33 * Progress $+ 0 . 3 3 \ ^ { * }$ Potential‘

- Round the final score to two decimals.

Please provide a score between 0.0 and 1.0 (1.0 indicates that the current state is highly valuable):

Score:

Figure 13: The prompt for the RT state evaluation (node of other actions)

Please answer the following question based on the provided information:

Question: {query}

Known Facts: {triples}

Reasoning Process: {trajectory}

# Instructions:

- Provide a direct, concise, and natural language answer relevant to the question.   
- Avoid technical terms or RDF-specific labels (e.g., “instance of”, “resource”, “synset”, etc.).   
- Express relationships and entities in simple terms, focusing on the core meaning.

Answer:

Figure 14: The prompt for the RT response generation
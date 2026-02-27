# NEMORI: SELF-ORGANIZING AGENT MEMORY INSPIRED BY COGNITIVE SCIENCE

Jiayan Nan*,† Wenquan Ma*,‡ Wenlong Wu** Yize Chen***

†School of Computer Science and Technology, Tongji University, Shanghai, China

‡School of Statistics and Data Science, Shanghai University of Finance and Economics, Shanghai, China

**School of Instrumentation and Optoelectronic Engineering, Beihang University, Beijing, China

***Tanka AI

{njy@tongji.edu.cn, wenquan.ma@stu.sufe.edu.cn, wlw@buaa.edu.cn, chenyize@tanka.ai}

# ABSTRACT

Large Language Models (LLMs) demonstrate remarkable capabilities, yet their inability to maintain persistent memory in long contexts limits their effectiveness as autonomous agents in long-term interactions. While existing memory systems have made progress, their reliance on arbitrary granularity for defining the basic memory unit and passive, rule-based mechanisms for knowledge extraction limits their capacity for genuine learning and evolution. To address these foundational limitations, we present Nemori, a novel self-organizing memory architecture inspired by human cognitive principles. Nemori’s core innovation is twofold: First, its Two-Step Alignment Principle, inspired by Event Segmentation Theory, provides a principled, top-down method for autonomously organizing the raw conversational stream into semantically coherent episodes, solving the critical issue of memory granularity. Second, its Predict-Calibrate Principle, inspired by the Free-energy Principle, enables the agent to proactively learn from prediction gaps, moving beyond pre-defined heuristics to achieve adaptive knowledge evolution. This offers a viable path toward handling the long-term, dynamic workflows of autonomous agents. Extensive experiments on the LoCoMo and LongMemEvalS benchmarks demonstrate that Nemori significantly outperforms prior state-of-the-art systems, with its advantage being particularly pronounced in longer contexts.

Code: The MVP implementation of Nemori is available as open-source software at https://github.com/ nemori-ai/nemori.

# 1 Introduction

The amnesia of Large Language Models (LLMs) stands as the core bottleneck to the grander vision of autonomous agents capable of genuine learning and intelligent self-evolution. For instance, LLMs exhibit a striking, seemingly personalized contextual ability within a single interaction; however, this illusion shatters with new sessions, as the agent, devoid of prior history, greets the user as a stranger. This inability to maintain long-term memory primarily stems from two core technical constraints: their limited context window, rooted in quadratic ${ \cal { O } } \bar { ( } n ^ { 2 } )$ attention complexity [1], and the Lost in the Middle phenomenon, hindering effective information utilization in long contexts [2]. Consequently, unless long-term memory of user interactions can be effectively addressed, the grander vision of human-like self-evolution for agents will remain unattainable [3, 4, 5].

Fortunately, the solution to this amnesia lies in selective information provision via In-Context Learning (ICL, 6), a principle most prominently realized at scale by the Retrieval-Augmented Generation (RAG, 7) framework. Analogous to human selective recall, the key lies in storing vast information in an organized manner for efficient, contextual retrieval [8]. RAG effectively grounds LLMs in external factual documents, mitigating hallucination and providing

domain-specific knowledge [9, 10]. This paradigm’s success is built upon a powerful, retrieval-centric philosophy [11]. This naturally raises a compelling question: could this powerful philosophy also be repurposed to augment an agent’s memory of its own past, solving its amnesia?

However, RAG’s core characteristics, designed for static knowledge bases, fundamentally misalign with the demands of dynamic conversation, giving rise to Memory-Augmented Generation (MAG, 12), a new paradigm focused on the self-organization of an agent’s own lived experience. This misalignment manifests on three critical levels: its stateless information patch approach prevents stateful learning [13, 14]; its reliance on offline indexing is antithetical to online conversational stream processing [15]; and its focus on fact retrieval proves insufficient for the complex, local-global reasoning inherent in dialogue [16]. This profound paradigm mismatch highlights MAG’s necessity, transforming traditional retrieval from static libraries into an autonomous process of organizing an agent’s lived, temporal experiences into an optimized representation [5]. MAG is essential not only for coherent conversations but also as a foundational component for achieving the long-term goal of agent self-evolution.

While the MAG paradigm represents a significant step beyond traditional RAG, existing methods have yet to unlock the full potential of human-like self-organization. We argue that this stems from a fundamental neglect of the self aspect of self-organization. The quality of a final memory unit, $y$ , is critically limited by a lack of self, which manifests as two sequential challenges, $y = f ( x )$ : the input chunks $( x )$ and the organizing mechanism itself $( f )$ . The first challenge concerns the input chunks $( x )$ . Existing MAG systems often adopt arbitrary or unspecified segmentation, inherently leading to a loss of contextual information. The root cause of this is the failure to leverage the agent’s own self capabilities to autonomously convert the raw stream into semantically coherent chunks. The second, more advanced problem is the organizing function itself $( f )$ . Existing methods struggle to balance retaining details with forming abstractions, resulting in redundant or incomplete memory representations. This stems from a failure to recognize that memory naturally follows a dual cognitive structure, which separates episodic details from semantic knowledge. Crucially, $( f )$ also lacks a proactive self -learning mechanism to bridge the gap between memory and the raw conversation. Existing systems within a pre-defined prompt are limited to what fits the preset schema, highlighting the need for an end-to-end, proactive process that intrinsically focuses on non-redundant content generation.

To address these fundamental challenges, we introduce Nemori, a novel self-organizing memory architecture built upon a dual-pillar cognitive framework. This framework offers a principled solution to the sequential challenges of defining the input chunks $( x )$ and designing the organizing function $( f )$ . First, for the input chunking problem $( x )$ , the framework’s Two-Step Alignment Principle offers a principled, top-down solution. It begins with the Boundary Alignment step, which, inspired by Event Segmentation Theory [17], autonomously organizes the raw conversational stream into semantically coherent experience chunks. Computationally, we achieve this by adapting and simplifying techniques from dialogue topic segmentation, allowing Nemori to move beyond arbitrary segmentation. Second, the challenge of the organizing function $( f )$ is addressed through a two-pronged approach. The initial step is Representation Alignment (a sub-principle of the Two-Step Alignment Principle), which simulates the natural human narration of Episodic Memory [18] to transform raw chunks into rich, narrative memory. This is complemented by our Predict-Calibrate Principle, a proactive learning mechanism inspired by the Free-energy Principle [19]. This principle

![](images/6730f03908b090bbacf264f21fb1858880756528ff4dcf51823e5c79d9e43ef5.jpg)  
Figure 1: The conceptual framework of Nemori, illustrating the mapping from problem to principle to computation. The framework addresses two core challenges: defining appropriate input chunks $( x )$ and designing an effective organizing function $( f )$ . The Two-Step Alignment Principle (comprising Boundary Alignment and Representation Alignment) solves the input chunking and initial representation problem. Concurrently, the Predict-Calibrate Principle provides a proactive mechanism for the organizing function, which operationalizes them via three core modules: Topic Segmentation, Episodic Memory Generation, and Semantic Memory Generation, as illustrated here.

posits that genuine learning stems from actively distilling prediction gaps, analogous to the effective human strategy of attempting a task before reflecting on discrepancies against a standard, which fosters deeper understanding than passively reviewing solutions. Together, these principles form a synergistic, complementary learning system [20] that underpins Nemori’s architecture, which operationalizes them via three core modules: Topic Segmentation, Episodic Memory Generation, and Semantic Memory Generation, illustrated in Figure 1. Our main contributions are as follows: (1) We propose a novel, dual-pillar framework for dynamic conversational memory, inspired by cognitive science: the Two-Step Alignment Principle for faithful experience representation and the Predict-Calibrate Principle for proactive knowledge distillation. (2) We design and implement Nemori, a complete memory architecture that operationalizes this framework, incorporating technical innovations like a top-down intelligent boundary detector and an asynchronous predict-calibrate pipeline. (3) We demonstrate Nemori’s effectiveness and robustness through extensive experiments, confirming it significantly outperforms prior state-of-the-art systems on the LoCoMo and LongMemEvalS benchmarks, with its advantage being particularly pronounced in longer contexts.

# 2 Related Works

# 2.1 Beyond Static RAG: The Frontier of Streaming Memory

Our work is situated within the broad paradigm of non-parametric memory, which enhances LLMs with an external memory store, distinct from parametric memory [21] or hybrid approaches [22]. Within this domain, Retrieval-Augmented Generation (RAG, 7) is dominant, designed for retrieving from static knowledge bases to ground LLMs in external facts and provide domain-specific knowledge [9, 10]. In contrast, our work contributes to Memory-Augmented Generation (MAG), a distinct current focusing on constructing and retrieving from a dynamic memory of an agent’s own lived, temporal experiences [5].

# 2.2 The Input Chunk Challenge (x): Solving the Granularity Gap in Agent Memory

A foundational, yet often overlooked, challenge within the MAG paradigm is defining the basic unit of experience, the input chunk $( x )$ . Cognitive science suggests an ideal memory unit should correspond to a coherent event[17], yet prior work reveals a spectrum of heuristic-based and incomplete approaches that neglect the agent’s self capability to autonomously define these units.

Prevailing methods often adopt arbitrary or heuristic segmentation. The most primitive approaches use a Single Message (an independent user input or system output, 23, 24) or an Interaction Pair (a bundled “user input $^ +$ system response”, 25, 26), resulting in fragmented memories that lack broader semantic context. To improve coherence, other systems employ external structures like Pre-defined Sessions (external structures, 27, 28) or outsource the task via User-defined Chunks (outsourcing the definition to human users, 29). While producing higher-quality chunks, these methods compromise scalability due to significant operational overhead and limited automation.

Concurrently, a significant body of work focuses on memory storage and retrieval, treating the memory unit as an Unspecified Unit or a black box [30, 31, 32, 12]. These systems advance memory management but do not address the foundational issue of how meaningful units are formed in the first place. Figure 2 visually illustrates the limitations of these approaches compared to our proposed episodic method.

This brings us to the frontier: the Self-organized Episode. Pioneering work like EM-LLM operationalized this via a bottom-up, token-level mechanism based on predictive surprise [33], which contrasts with the top-down reasoning required for holistic social interactions. While techniques from dialogue topic segmentation exist [34], their purpose is general topic analysis, not creating memory units for agents. This calls for a fully automated, top-down, cognitivegrounded approach to model underlying events, a challenge our Two-Step Alignment Principle is designed to solve, beginning with its Boundary Alignment step.

# 2.3 The Organizing Function Challenge (f): From Passive Storage to Proactive Learning

Beyond the granularity of input chunks $( x )$ , the second key challenge is the organizing function $( f )$ , which governs how memory is structured and evolved. Dual-memory systems, which typically maintain raw episodic memories and abstracted semantic knowledge, form a prominent approach, theoretically rooted in the Complementary Learning Systems theory (CLS, 20) which posits that the brain uses complementary fast-learning (episodic) and slow-learning (semantic) systems.

However, pioneering systems applying this concept, such as HEMA [25] and Mem0 [26], address the organizing function $( f )$ with significant limitations. Early work like HEMA applied CLS theory by using passive summarization to create global summaries from raw dialogue. Mem0 advanced this by extracting consolidated factual entries, which

enhances semantic queryability but often comes at the cost of compromising the original episodic context. Consequently, for memory representation, both approaches rely on simplified transformations rather than a principled, cognitivelyinspired method for narrative generation, a challenge our Representation Alignment principle addresses. More critically, their mechanism for knowledge evolution remains a passive, extraction-based process. This reveals an unaddressed gap: the absence of a proactive learning mechanism for the agent to autonomously evolve its knowledge base. Our work fills this gap with the Predict-Calibrate Principle. Inspired by the Free-energy Principle [19], it moves beyond passive extraction(e.g. pre-defined extraction rules) by actively distilling prediction gaps, enabling the system to learn from its own errors for a truly synergistic, complementary learning process.

# 3 Methodology

The Nemori methodology provides a concrete computational implementation of our dual-pillar cognitive framework: the Two-Step Alignment Principle and the Predict-Calibrate Principle. As illustrated in Figure 3, the system is composed of three core modules: Topic Segmentation, Episodic Memory Generation, and Semantic Memory Generation, all supported by a unified retrieval system. The first two modules, Topic Segmentation and Episodic Memory Generation, work in concert to operationalize the Two-Step Alignment Principle for faithful experience representation. The third module, Semantic Memory Generation, is designed to realize the Predict-Calibrate Principle for proactive knowledge evolution. In the following sections, we will detail the mechanisms of each component in accordance with this principle-driven structure.

# 3.1 The Two-Step Alignment Principle in Practice

The first principle is operationalized through two sequential modules: a boundary detector that realizes semantic Boundary Alignment, and an episode generator that realizes narrative Representation Alignment.

# 3.1.1 Boundary Alignment via Intelligent Detection.

The process of identifying episodic boundaries begins with a message buffer, denoted as $B _ { u }$ for each user $u$ , which accumulates incoming conversational messages. The sequence of messages in the buffer at a given time $t$ is represented as $M = \{ m _ { 1 } , m _ { 2 } , \ldots , m _ { t } \}$ , where each message $m _ { i }$ is a tuple $( \rho _ { i } , c _ { i } , \tau _ { i } )$ containing the role ρi ∈ {user, assistant}, the message content $c _ { i }$ , and the timestamp $\tau _ { i }$ .

For each new incoming message $m _ { t + 1 }$ , an LLM-based boundary detector, $f _ { \theta }$ , is invoked to determine if a meaningful semantic boundary has been crossed. Rather than a simple probability, the detector’s output is a structured response containing both a boolean decision and a confidence score:

$$
\left(b _ {\text {b o u n d a r y}}, c _ {\text {b o u n d a r y}}\right) = f _ {\theta} \left(m _ {t + 1}, M\right) \tag {1}
$$

where $b _ { \mathrm { b o u n d a r y } } \in \{ \mathrm { T r u e , F a l s e } \}$ and $c _ { \mathrm { b o u n d a r y } } \in [ 0 , 1 ]$ . The function $f _ { \theta }$ makes this determination by evaluating several factors, including contextual coherence (i.e., the semantic similarity between messages), temporal markers (e.g., by the way), shifts in user intent(e.g., from asking for information to making a decision), and other structural signals.

![](images/29ca8faad5e4ce82f6f2c021bbfc0d225f984ec2abeda863576f77d0e42bd2c7.jpg)  
Figure 2: An illustration of different conversation segmentation methods. Standard RAG (left) often relies on arbitrary, fixed-size chunking, which can break the semantic integrity of a dialogue (as shown by the split in the apple discussion). The Interaction Pair model (middle) groups user-assistant turns but can still separate related user messages. In contrast, our proposed Episodic segmentation (right), guided by semantic boundary detection, correctly groups the entire conversation about the apple into a single, coherent episode, preserving the interaction’s logical flow.

Topic segmentation is triggered when either of two conditions is met: a high-confidence semantic shift is detected, or the buffer reaches its capacity. This is formally expressed as:

$$
\mathrm {T} = \left(b _ {\text {b o u n d a r y}} \wedge c _ {\text {b o u n d a r y}} > \sigma_ {\text {b o u n d a r y}}\right) \vee \left(| M | \geq \beta_ {\max }\right) \tag {2}
$$

where σboundary is a configurable confidence threshold, $| M |$ is the number of messages in buffer $M$ , and $\beta _ { \mathrm { m a x } }$ is a predefined maximum buffer size. Upon triggering (i.e., when $\mathbf { T } = \mathbf { T r u e } ,$ ), the message sequence $M$ is passed to the next module for episodic memory generation, leaving the new message $m _ { t + 1 }$ to initialize the subsequent buffer.

# 3.1.2 Representation Alignment via Narrative Generation.

The Episodic Memory Generation module receives the Segmented Conversation, denoted as $M$ , upon the detection of a boundary. Its purpose is to transform this raw segment into a structured episodic memory, e. This transformation is performed by an LLM-based Episode Generator, $g _ { \phi }$ , which reframes the segmented dialogue into a coherent, narrative representation. The output of this process is a structured tuple:

$$
e = (\xi , \zeta) = g _ {\phi} (M) \tag {3}
$$

where $\xi$ represents a concise title that encapsulates the episode’s core theme, and $\zeta$ is a detailed third-person narrative that preserves the salient information and context of the interaction. This structured format of combining a title with a rich narrative aligns with our Representation Alignment principle. Subsequently, the complete episodic memory $e$ is stored in the Episodic Memory Database, while its title $\xi$ is passed to the Semantic Memory Generation module to initiate the learning cycle.

# 3.2 The Predict-Calibrate Principle in Practice

The second, proactive learning principle, Predict-Calibrate, is operationalized by the Semantic Memory Generation module. This component serves as the core of agent learning and evolution, implementing a novel mechanism for incremental knowledge acquisition inspired by the Free Energy Principle from cognitive science [19]. As depicted in Figure 3, this learning process operates in a three-stage cycle.

# 3.2.1 Stage 1: Prediction.

The cycle begins when the module receives the title $\xi$ of a newly generated episode $e _ { \mathrm { n e w } } = ( \xi , \zeta )$ . The first stage of the cycle is to forecast the episode’s content based on existing knowledge. This process unfolds in two main parts: first retrieving relevant memories, and then making the prediction.

Memory Retrieval. To identify relevant knowledge from the Semantic Memory Database $K$ , the system retrieves a set of relevant memories, $K _ { \mathrm { r e l e v a n t } }$ , for the new episode’s content. This retrieval is performed by our unified retrieval mechanism, which takes the embedding of the new episode’s concatenated title and content as a query, along with the

![](images/82ef59ef18b64171d41ce7de36b1fc33d69764ffb4041b50050b615252af3b11.jpg)  
Figure 3: The Nemori system features three modules: Topic Segmentation, Episodic Memory Generation, and Semantic Memory Generation. It segments conversations into Episodic Memory, then uses a Predict-Calibrate cycle to distill new Semantic Memory from prediction gaps against original conversations.

semantic memory database $K$ , a maximum number of results $m$ , and a configurable similarity threshold $\sigma _ { s }$ . The result is the definitive set of relevant memories:

$$
K _ {\text {r e l e v a n t}} = \operatorname {R e t r i e v e} \left(\operatorname {e m b e d} \left(\xi \oplus \zeta\right), K, m, \sigma_ {s}\right) \tag {4}
$$

This ensures high-quality contextual information for the subsequent prediction stage.

Episode Prediction. With the relevant knowledge retrieved, an LLM-based Episode Predictor, $h _ { \psi }$ , then forecasts the episode’s content, $\hat { e }$ , based on the episode’s title $\xi$ and the final set of relevant knowledge $K _ { \mathrm { r e l e v a n t } }$ :

$$
\hat {e} = h _ {\psi} (\xi , K _ {\text {r e l e v a n t}}) \tag {5}
$$

# 3.2.2 Stage 2: Calibration.

In the calibration stage, the predicted content $\hat { e }$ is compared against the ground truth of the interaction. Crucially, this ground truth is not the generated episodic narrative $\zeta$ , but the original, unprocessed Segmented Conversation block, $M$ . An LLM-based Semantic Knowledge Distiller, $r _ { \omega }$ , processes this comparison to identify the prediction gap—the novel or surprising information that the existing knowledge base failed to predict. From this gap, a new set of semantic knowledge statements, $K _ { \mathrm { n e w } }$ , is distilled:

$$
K _ {\text {n e w}} = r _ {\omega} (\hat {e}, M) \tag {6}
$$

# 3.2.3 Stage 3: Integration.

Finally, the newly generated and validated knowledge statements, $K _ { \mathrm { n e w } }$ , are integrated into the main Semantic Memory Database $K$ . This completes the learning cycle, enriching the agent’s knowledge base and refining its internal model of the world.

# 3.3 Unified Memory Retrieval

The system employs a unified vector-based retrieval approach, denoted as $\mathrm { R e t r i e v e } ( q , D , m , \sigma _ { s } )$ , optimized for accessing both episodic and semantic memories. This function takes a query $q$ , a memory database $D$ , a maximum number of results $m$ , and an optional similarity threshold $\sigma _ { s }$ to return a set of relevant memories. The retrieval mechanism uses dense vector search with cosine similarity to identify semantically relevant memories through a three-stage process: similarity computation, candidate selection, and threshold-based filtering.

# 4 Experiment

In this section, we conduct a series of experiments on two benchmark datasets to investigate the effectiveness of Nemori. Our research is designed to address the following key research questions (RQs):

RQ1: How does Nemori perform in long-term conversational memory tasks compared to state-of-the-art methods?   
RQ2: What are the contributions of Nemori’s key components to its overall performance?   
RQ3: How does the model’s performance change with adjustments to the number of retrieved episodic memories?   
RQ4: How well does Nemori scale to significantly longer and more challenging conversational contexts?

# 4.1 Experimental Setup

# 4.1.1 Datasets.

We evaluate Nemori on two distinct benchmarks to ensure a comprehensive validation of our approach.

• LoCoMo [35]: 10 dialogues with 24K average tokens, featuring 1,540 questions across four reasoning categories.   
• LongMemEvalS [36]: 500 conversations with 105K average tokens. While structurally similar to LoCoMo, it presents significantly greater challenges through longer, more realistic conversational contexts, allowing us to assess scalability under demanding conditions.

# 4.1.2 Baselines.

We benchmark Nemori against five powerful and representative baselines, categorized as follows:

• Standard Method: Full Context, which provides the entire dialogue history to the LLM, representing the theoretical upper bound of information availability.   
• Retrieval-Augmented Method: RAG-4096, a standard retrieval-augmented generation approach that chunks dialogues into 4096-token segments for dense retrieval.   
• Memory-Augmented Systems: We compare against three state-of-the-art memory systems: LangMem [37], which uses a hierarchical memory structure; Zep [38], a commercial solution based on temporal knowledge graphs; and Mem0 [26], a system that extracts and maintains personalized memories.

# 4.1.3 Evaluation Metrics.

On the LoCoMo dataset, our primary evaluation metric is the LLM-judge score, where we employ gpt-4o-mini as the judge. We supplement this with F1 and BLEU-1 scores for a more complete picture. For the LongMemEvalS dataset, we also use the LLM-judge score, but with prompts adapted to its specific question-answering format.

# 4.1.4 Reproducibility.

To ensure fair comparison, Mem0 and Zep utilize their commercial APIs to retrieve memory contexts, which are then fed to gpt-4o-mini and gpt-4.1-mini for answer generation. All other methods, including Nemori, employ gpt-4o-mini and gpt-4.1-mini as both internal backbone models and answer generation models. For Nemori specifically, embeddings are generated with text-embedding-3-small. Key hyperparameters were set as follows: similarity threshold $\sigma _ { s } = 0 . 0$ , boundary detection confidence $\sigma _ { \mathrm { b o u n d a r y } } = 0 . 7$ , and max buffer size $\beta _ { \mathrm { m a x } } = 2 5$ . For retrieval settings across all experiments, we maintain a fixed ratio between episodic and semantic memory retrieval: we retrieve top- $k$ episodic memories and top- $m = 2 k$ semantic memories. In the main experiments, $k = 1 0$ (thus $m = 2 0$ ), while in RQ3’s hyperparameter analysis, $k$ varies from 2 to 20. To balance informativeness and efficiency, only the top-2 episodic memories include their original conversation text, as higher-similarity episodes tend to be more useful.

# 4.2 Main Results (RQ1)

To answer RQ1, we report the performance comparison on the LoCoMo dataset in Table 1. Our observations are as follows: Superior Performance Across the Board. Nemori consistently outperforms all baseline methods across both

Table 1: Detailed performance comparison on LoCoMo dataset by question type. Bold indicates the best performance for each metric.   

<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="3">Temporal Reasoning</td><td colspan="3">Open Domain</td><td colspan="3">Multi-Hop</td><td colspan="3">Single-Hop</td><td colspan="3">Overall</td></tr><tr><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td><td>LLM Score</td><td>F1</td><td>BLEU-1</td></tr><tr><td rowspan="6">gpt-40-mini</td><td>FullContext</td><td>0.562 ± 0.004</td><td>0.441</td><td>0.361</td><td>0.486 ± 0.005</td><td>0.245</td><td>0.172</td><td>0.668 ± 0.003</td><td>0.354</td><td>0.261</td><td>0.830 ± 0.001</td><td>0.531</td><td>0.447</td><td>0.723 ± 0.000</td><td>0.462</td><td>0.378</td></tr><tr><td>LangMem</td><td>0.249 ± 0.003</td><td>0.319</td><td>0.262</td><td>0.476 ± 0.005</td><td>0.294</td><td>0.235</td><td>0.524 ± 0.003</td><td>0.335</td><td>0.239</td><td>0.614 ± 0.002</td><td>0.388</td><td>0.331</td><td>0.513 ± 0.003</td><td>0.358</td><td>0.294</td></tr><tr><td>Mem0</td><td>0.504 ± 0.001</td><td>0.444</td><td>0.376</td><td>0.406 ± 0.000</td><td>0.271</td><td>0.194</td><td>0.603 ± 0.000</td><td>0.343</td><td>0.252</td><td>0.681 ± 0.000</td><td>0.444</td><td>0.377</td><td>0.613 ± 0.000</td><td>0.415</td><td>0.342</td></tr><tr><td>RAG</td><td>0.237 ± 0.000</td><td>0.195</td><td>0.157</td><td>0.326 ± 0.005</td><td>0.190</td><td>0.135</td><td>0.313 ± 0.003</td><td>0.186</td><td>0.117</td><td>0.320 ± 0.001</td><td>0.222</td><td>0.186</td><td>0.302 ± 0.000</td><td>0.208</td><td>0.164</td></tr><tr><td>Zep</td><td>0.589 ± 0.003</td><td>0.448</td><td>0.381</td><td>0.396 ± 0.000</td><td>0.229</td><td>0.157</td><td>0.505 ± 0.007</td><td>0.275</td><td>0.193</td><td>0.632 ± 0.001</td><td>0.397</td><td>0.337</td><td>0.585 ± 0.001</td><td>0.375</td><td>0.309</td></tr><tr><td>Nemori (Ours)</td><td>0.710 ± 0.000</td><td>0.567</td><td>0.466</td><td>0.448 ± 0.005</td><td>0.208</td><td>0.151</td><td>0.653 ± 0.002</td><td>0.365</td><td>0.256</td><td>0.821 ± 0.002</td><td>0.544</td><td>0.432</td><td>0.744 ± 0.001</td><td>0.495</td><td>0.385</td></tr><tr><td rowspan="6">gpt-41-mini</td><td>FullContext</td><td>0.742 ± 0.004</td><td>0.475</td><td>0.400</td><td>0.566 ± 0.010</td><td>0.284</td><td>0.222</td><td>0.772 ± 0.003</td><td>0.442</td><td>0.337</td><td>0.869 ± 0.002</td><td>0.614</td><td>0.534</td><td>0.806 ± 0.001</td><td>0.533</td><td>0.450</td></tr><tr><td>LangMem</td><td>0.508 ± 0.003</td><td>0.485</td><td>0.409</td><td>0.590 ± 0.005</td><td>0.328</td><td>0.264</td><td>0.710 ± 0.002</td><td>0.415</td><td>0.325</td><td>0.845 ± 0.001</td><td>0.510</td><td>0.436</td><td>0.734 ± 0.001</td><td>0.476</td><td>0.400</td></tr><tr><td>Mem0</td><td>0.569 ± 0.001</td><td>0.392</td><td>0.332</td><td>0.479 ± 0.000</td><td>0.237</td><td>0.177</td><td>0.682 ± 0.003</td><td>0.401</td><td>0.303</td><td>0.714 ± 0.001</td><td>0.486</td><td>0.420</td><td>0.663 ± 0.000</td><td>0.435</td><td>0.365</td></tr><tr><td>RAG</td><td>0.274 ± 0.000</td><td>0.223</td><td>0.191</td><td>0.288 ± 0.005</td><td>0.179</td><td>0.139</td><td>0.317 ± 0.003</td><td>0.201</td><td>0.128</td><td>0.359 ± 0.002</td><td>0.258</td><td>0.220</td><td>0.329 ± 0.002</td><td>0.235</td><td>0.192</td></tr><tr><td>Zep</td><td>0.602 ± 0.001</td><td>0.239</td><td>0.200</td><td>0.438 ± 0.000</td><td>0.242</td><td>0.193</td><td>0.537 ± 0.003</td><td>0.305</td><td>0.204</td><td>0.669 ± 0.001</td><td>0.455</td><td>0.400</td><td>0.616 ± 0.000</td><td>0.369</td><td>0.309</td></tr><tr><td>Nemori (Ours)</td><td>0.776 ± 0.003</td><td>0.577</td><td>0.502</td><td>0.510 ± 0.009</td><td>0.258</td><td>0.193</td><td>0.751 ± 0.002</td><td>0.417</td><td>0.319</td><td>0.849 ± 0.002</td><td>0.588</td><td>0.515</td><td>0.794 ± 0.001</td><td>0.534</td><td>0.456</td></tr></table>

backbone models. With gpt-4o-mini, Nemori achieves an overall LLM score of 0.744, which already surpasses the Full Context baseline’s score of 0.723. With gpt-4.1-mini, Nemori further improves to 0.794. This demonstrates the powerful capability of Nemori’s self-organizing memory system. Moreover, the system scales up effectively as the underlying model capabilities strengthen.

Exceptional Temporal Reasoning. The advantage of our method is especially pronounced in the Temporal Reasoning category, where Nemori achieves scores of 0.710 and 0.776. This validates the effectiveness of our episode-based memory structure, which naturally preserves the chronological flow. A key reason for this superiority is Nemori’s ability to perform “reasoning during memory formation.” For instance, when faced with the question “When did Jon receive mentorship?”, the Full-Context baseline, confused by the term “yesterday” in the original text, incorrectly answered with the conversation date (June 16). In contrast, Nemori’s dual memory system retrieved both the relevant episodic memory and a semantic memory that had already processed the temporal information into a clear fact: “Jon

was mentored on June 15, 2023.” By combining episodic context with pre-reasoned semantic facts, Nemori transforms complex reasoning tasks into simple information retrieval, significantly boosting accuracy.

Table 2: Performance and efficiency comparison on LoCoMo dataset with gpt-4o-mini.   

<table><tr><td>Method</td><td>LLM Score</td><td>Tokens</td><td>Search (ms)</td><td>Total (ms)</td></tr><tr><td>FullContext</td><td>0.723</td><td>23,653</td><td>-</td><td>5,806</td></tr><tr><td>LangMem</td><td>0.513</td><td>125</td><td>19,829</td><td>22,082</td></tr><tr><td>Mem0</td><td>0.613</td><td>1,027</td><td>784</td><td>3,539</td></tr><tr><td>RAG-4096</td><td>0.302</td><td>3,430</td><td>544</td><td>2,884</td></tr><tr><td>Zep</td><td>0.585</td><td>2,247</td><td>522</td><td>3,255</td></tr><tr><td>Nemori</td><td>0.744</td><td>2,745</td><td>787</td><td>3,053</td></tr></table>

Efficiency Advantages. Table 2 highlights Nemori’s efficiency. While delivering superior performance, Nemori uses only 2,745 tokens on average, an $88 \%$ reduction compared to the 23,653 tokens required by the Full Context baseline. This demonstrates that Nemori not only improves accuracy but does so with remarkable computational efficiency.

# 4.3 Ablation Study (RQ2)

To answer RQ2, we conducted an ablation study to quantify the contribution of each key component in Nemori. The results, summarized in Table 3, lead to several key insights:

Table 3: Ablation study on Nemori components. Nemori-s uses direct semantic extraction.   

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td colspan="3">Overall Performance</td></tr><tr><td>LLM Score</td><td>F1</td><td>BLEU-1</td></tr><tr><td rowspan="5">gpt-4o-mini</td><td>w/o Nemori</td><td>0.006</td><td>0.005</td><td>0.009</td></tr><tr><td>Nemori-s</td><td>0.518</td><td>0.346</td><td>0.272</td></tr><tr><td>w/o e</td><td>0.615</td><td>0.434</td><td>0.340</td></tr><tr><td>w/o s</td><td>0.705</td><td>0.470</td><td>0.370</td></tr><tr><td>Nemori</td><td>0.744</td><td>0.495</td><td>0.385</td></tr><tr><td rowspan="5">gpt-4.1-mini</td><td>w/o Nemori</td><td>0.012</td><td>0.016</td><td>0.015</td></tr><tr><td>Nemori-s</td><td>0.623</td><td>0.391</td><td>0.322</td></tr><tr><td>w/o e</td><td>0.696</td><td>0.461</td><td>0.396</td></tr><tr><td>w/o s</td><td>0.756</td><td>0.501</td><td>0.435</td></tr><tr><td>Nemori</td><td>0.794</td><td>0.534</td><td>0.456</td></tr></table>

Legend: w/o Nemori $=$ without Nemori framework; w/o ${ \bf e } =$ without episodic retrieval; w/o ${ \bf s } =$ without semantic retrieval; Nemori $=$ full framework

Core Framework Necessity. Removing the entire Nemori framework (w/o Nemori) causes performance to collapse to near-zero. This confirms the fundamental necessity of a structured memory architecture for performing these tasks.

Validation of the Predict-Calibrate Principle. An important finding comes from comparing Nemori (w/o e) with Nemori-s. Both configurations rely solely on semantic memory, but differ critically in how that memory is generated. Nemori (w/o e) uses our proposed Predict-Calibrate Principle to proactively distill knowledge, while Nemori-s relies on naive, direct extraction from raw conversation logs. The performance gap between them is substantial (e.g., a score of 0.615 for Nemori (w/o e) vs. 0.518 for nemori-s on gpt-4o-mini). This result provides a direct empirical validation of our principle, demonstrating that proactively learning from prediction gaps produces a significantly more effective knowledge base than simple, reactive extraction.

Complementary Roles of Memory Types. Removing either episodic memory (w/o e) or semantic memory (w/o s) from the full Nemori model leads to performance degradation. The larger drop from removing episodic memory (from 0.744 to 0.615) compared to semantic memory (from 0.744 to 0.705) highlights the complementary and essential roles of both memory systems in our dual-memory architecture.

![](images/121d82c320abb2b2361fec347d0fd6648d39ffe3e03bfb9d9d591cea96940372.jpg)

![](images/9c67a8be6445a2e6b08a9caac10b89ed1b8cb19110af43ae1e0f2e174c417f1b.jpg)  
Figure 4: Impact of top-k episodes on LLM score across different models. Both models show performance rises sharply until $_ { \mathrm { k = 1 0 } }$ and then plateaus. The red dashed lines represent Full Context baseline performance for comparison.

# 4.4 Hyperparameter Analysis (RQ3)

To answer RQ3, we conducted a sensitivity analysis on the number of retrieved episodic memories, $k$ , to understand its impact on model performance. Throughout this analysis, we maintained the semantic memory retrieval count at $m = 2 k$ to preserve the relative balance between memory types. The results, shown in Figure 4, reveal a clear and consistent pattern: performance rises sharply as $k$ increases from 2 to 10 (with $m$ correspondingly increasing from 4 to 20), and then largely plateaus, with minimal marginal gains for $k > 1 0$ . This observation of diminishing returns is insightful. It demonstrates that the model’s performance is not contingent on retrieving an ever-larger number of memories, but rather can achieve near-optimal performance within a relatively small, targeted retrieval window.

Model-Dependent Performance Ceiling Analysis. An intriguing observation from Figure 4 is the differential relationship between Nemori and the Full Context baseline across different model capabilities. With gpt-4o-mini, Nemori achieves a clear performance advantage over Full Context (0.744 vs 0.723), while with gpt-4.1-mini, Nemori approaches but does not substantially exceed the baseline (0.794 vs 0.806). This pattern suggests an interesting interaction between model capacity and memory system effectiveness. For more capable models, the LoCoMo dataset may represent a relatively straightforward task where raw processing power can effectively utilize extensive context without sophisticated memory organization. However, as we demonstrate in RQ4 with the more challenging LongMemEvalS benchmark, both models benefit significantly from Nemori’s structured memory approach when facing truly complex, long-context scenarios. This finding highlights a crucial design principle: the value of intelligent memory systems becomes more pronounced as task complexity increases, particularly in resource-constrained environments where computational efficiency is paramount.

Note: Nemori achieves higher accuracy while using 95-96% less context than Full-context baseline. Table 4: Performance comparison on LongMemEvalS dataset across different question types.   

<table><tr><td></td><td>Question Type</td><td>Full-context (101K tokens)</td><td>Nemori (3.7-4.8K tokens)</td></tr><tr><td rowspan="7">gpt-40-mini</td><td>single-session-preference</td><td>6.7%</td><td>46.7%</td></tr><tr><td>single-session-assistant</td><td>89.3%</td><td>83.9%</td></tr><tr><td>temporal-reasoning</td><td>42.1%</td><td>61.7%</td></tr><tr><td>multi-session</td><td>38.3%</td><td>51.1%</td></tr><tr><td>knowledge-update</td><td>78.2%</td><td>61.5%</td></tr><tr><td>single-session-user</td><td>78.6%</td><td>88.6%</td></tr><tr><td>Average</td><td>55.0%</td><td>64.2%</td></tr><tr><td rowspan="7">gpt-4.1-mini</td><td>single-session-preference</td><td>16.7%</td><td>86.7%</td></tr><tr><td>single-session-assistant</td><td>98.2%</td><td>92.9%</td></tr><tr><td>temporal-reasoning</td><td>60.2%</td><td>72.2%</td></tr><tr><td>multi-session</td><td>51.1%</td><td>55.6%</td></tr><tr><td>knowledge-update</td><td>76.9%</td><td>79.5%</td></tr><tr><td>single-session-user</td><td>85.7%</td><td>90.0%</td></tr><tr><td>Average</td><td>65.6%</td><td>74.6%</td></tr></table>

# 4.5 Generalization Study (RQ4)

To answer RQ4, we evaluated Nemori on the LongMemEvalS dataset [36]. While structurally similar to LoCoMo in its conversational nature, LongMemEvalS presents a significantly greater challenge in terms of scale, with an average context length of 105K tokens. This serves as a crucial stress test for long-term memory retention and generalization. The results in Table 4 demonstrate Nemori’s strong performance under these demanding conditions. A closer analysis reveals two key findings. First, Nemori shows superior performance on user preference tasks. This is because its concise, high-quality structured memory enables the model to focus more effectively on user habits and inclinations, which are often diluted within the baseline’s extensive context. Second, the baseline’s better performance on singlesession-assistant tasks suggests that Nemori can lose some fine-grained details, a potential limitation to be addressed in future work.

# 5 Conclusion

In this work, we introduced Nemori, a cognitively-inspired memory architecture that offers a principled solution to agent amnesia. By integrating the Two-Step Alignment Principle for coherent experience segmentation and the novel Predict-Calibrate Principle for proactive knowledge distillation, Nemori reframes memory construction as an active learning process. Extensive experiments demonstrate its effectiveness: Nemori not only significantly outperforms stateof-the-art systems on the LoCoMo and LongMemEvals benchmarks but also surpasses the Full Context baseline with $8 8 \%$ fewer tokens, while showing strong generalization in contexts up to 105K tokens. By shifting the paradigm from passive storage to active knowledge evolution, Nemori provides a foundational component for developing autonomous agents capable of genuine, human-like learning.

# References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
[2] Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172, 2023.   
[3] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.   
[4] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. Sparks of artificial general intelligence: Early experiments with gpt-4. arXiv preprint arXiv:2303.12712, 2023.   
[5] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.   
[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.   
[7] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.   
[8] Zihong He, Weizhe Lin, Hao Zheng, Fan Zhang, Matt W Jones, Laurence Aitchison, Xuhai Xu, Miao Liu, Per Ola Kristensson, and Junxiao Shen. Human-inspired perspectives: A survey on ai long-term memory. arXiv preprint arXiv:2411.00489, 2024.   
[9] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM computing surveys, 55(12):1–38, 2023.   
[10] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2(1), 2023.   
[11] Samuel J Gershman, Ila Fiete, and Kazuki Irie. Key-value memory in the brain. Neuron, 113(11):1694–1707, 2025.

[12] Zhiyu Li, Shichao Song, Chenyang Xi, Hanyu Wang, Chen Tang, Simin Niu, Ding Chen, Jiawei Yang, Chunyu Li, Qingchen Yu, et al. Memos: A memory os for ai system. arXiv preprint arXiv:2507.03724, 2025.   
[13] Yuanzhe Hu, Yu Wang, and Julian McAuley. Evaluating memory in llm agents via incremental multi-turn interactions. arXiv preprint arXiv:2507.05257, 2025.   
[14] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. arXiv preprint arXiv:2310.11511, 2023.   
[15] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.   
[16] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.   
[17] Jeffrey M Zacks and Barbara Tversky. Event structure in perception and conception. Psychological bulletin, 127(1):3, 2001.   
[18] Endel Tulving et al. Episodic and semantic memory. Organization of memory, 1(381-403):1, 1972.   
[19] Karl Friston. The free-energy principle: a unified brain theory? Nature reviews neuroscience, 11(2):127–138, 2010.   
[20] James L McClelland, Bruce L McNaughton, and Randall C O’Reilly. Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. Psychological review, 102(3):419, 1995.   
[21] Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model based agents. ACM Transactions on Information Systems, 2024.   
[22] Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, et al. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259, 2025.   
[23] Kai Mei, Xi Zhu, Wujiang Xu, Wenyue Hua, Mingyu Jin, Zelong Li, Shuyuan Xu, Ruosong Ye, Yingqiang Ge, and Yongfeng Zhang. Aios: Llm agent operating system. arXiv preprint arXiv:2403.16971, 2024.   
[24] Charles Packer, Vivian Fang, Shishir_G Patil, Kevin Lin, Sarah Wooders, and Joseph_E Gonzalez. Memgpt: Towards llms as operating systems. 2023.   
[25] Kwangseob Ahn. Hema: A hippocampus-inspired extended memory architecture for long-context ai conversations. arXiv preprint arXiv:2504.16754, 2025.   
[26] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building productionready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.   
[27] Derong Xu, Yi Wen, Pengyue Jia, Yingyi Zhang, Yichao Wang, Huifeng Guo, Ruiming Tang, Xiangyu Zhao, Enhong Chen, Tong Xu, et al. Towards multi-granularity memory association and selection for long-term conversational agents. arXiv preprint arXiv:2505.19549, 2025.   
[28] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731, 2024.   
[29] Ryan Yen and Jian Zhao. Memolet: Reifying the reuse of user-ai conversational memories. In Proceedings of the 37th Annual ACM Symposium on User Interface Software and Technology, pages 1–22, 2024.   
[30] Yu Wang and Xi Chen. Mirix: Multi-agent memory system for llm-based agents. arXiv preprint arXiv:2507.07957, 2025.   
[31] Guibin Zhang, Muxin Fu, Guancheng Wan, Miao Yu, Kun Wang, and Shuicheng Yan. G-memory: Tracing hierarchical memory for multi-agent systems. arXiv preprint arXiv:2506.07398, 2025.   
[32] Akash Vishwakarma, Hojin Lee, Mohith Suresh, Priyam Shankar Sharma, Rahul Vishwakarma, Sparsh Gupta, and Yuvraj Anupam Chauhan. Cognitive weave: Synthesizing abstracted knowledge with a spatio-temporal resonance graph. arXiv preprint arXiv:2506.08098, 2025.   
[33] Zafeirios Fountas, Martin A Benfeghoul, Adnan Oomerjee, Fenia Christopoulou, Gerasimos Lampouras, Haitham Bou-Ammar, and Jun Wang. Human-like episodic memory for infinite context llms. arXiv preprint arXiv:2407.09450, 2024.

[34] Haoyu Gao, Rui Wang, Ting-En Lin, Yuchuan Wu, Min Yang, Fei Huang, and Yongbin Li. Unsupervised dialogue topic segmentation with topic-aware contrastive learning. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’23, page 2481–2485, New York, NY, USA, 2023. Association for Computing Machinery.   
[35] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851–13870, Bangkok, Thailand, August 2024. Association for Computational Linguistics.   
[36] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval: Benchmarking chat assistants on long-term interactive memory. In The Thirteenth International Conference on Learning Representations, 2025.   
[37] Harrison Chase. Langchain. https://github.com/langchain-ai/langchain, 2022. Accessed: 2025-07-20.   
[38] Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: a temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025.
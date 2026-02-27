# FROM EXPERIENCE TO STRATEGY: EMPOWERING LLM AGENTS WITH TRAINABLE GRAPH MEMORY

Siyu $\mathbf { X _ { i a } ^ { \bullet } } ^ { 1 , 2 }$ ∗, Zekun $\mathbf { X } \mathbf { u } ^ { 3 }$ ∗, Jiajun Chai3, Wentian Fan4, Yan Song5, Xiaohan Wang3 Guojun ${ \bf Y i n } ^ { 3 }$ , Wei $\mathbf { L i n ^ { 3 } }$ , Haifeng Zhang1,2†, Jun Wang5†

1Institute of Automation, Chinese Academy of Sciences, Beijing, China   
2School of Artificial Intelligence, University of Chinese Academy of Sciences, China   
3Meituan 4Nanjing University of Posts and Telecommunications   
5AI Centre, Department of Computer Science, University College London, London, UK

# ABSTRACT

Large Language Models (LLMs) based agents have demonstrated remarkable potential in autonomous task-solving across complex, open-ended environments. A promising approach for improving the reasoning capabilities of LLM agents is to better utilize prior experiences in guiding current decisions. However, LLMs acquire experience either through implicit memory via training, which suffers from catastrophic forgetting and limited interpretability, or explicit memory via prompting, which lacks adaptability. In this paper, we introduce a novel agentcentric, trainable, multi-layered graph memory framework and evaluate how context memory enhances the ability of LLMs to utilize parametric information. The graph abstracts raw agent trajectories into structured decision paths in a state machine and further distills them into high-level, human-interpretable strategic meta-cognition. In order to make memory adaptable, we propose a reinforcementbased weight optimization procedure that estimates the empirical utility of each meta-cognition based on reward feedback from downstream tasks. These optimized strategies are then dynamically integrated into the LLM agent’s training loop through meta-cognitive prompting. Empirically, the learnable graph memory delivers robust generalization, improves LLM agents’ strategic reasoning performance, and provides consistent benefits during Reinforcement Learning (RL) training.

# 1 INTRODUCTION

LLM-based agents are rapidly advancing the frontier of automated task execution, particularly in open-ended environments that demand long-horizon reasoning, strategic tool use, and adaptation from experience (Yao et al., 2022; Gao et al., 2023; Chai et al., 2025). While these agents demonstrate strong capabilities in decomposing and tackling complex tasks, their decision-making processes remain unstable, often resulting in inefficient action sequences, repeated mistakes, or even complete task failure (Singh et al., 2023). A central challenge lies in empowering agents not only to act, but to continuously learn and adapt by extracting insights from past successes and errors.

Methods for enabling LLMs to better leverage prior experience can be broadly categorized into two paradigms. The first is implicit memory, is typically formed through training procedures like RL, which denotes LLMs encode syntactic structures and semantic relations into parameter space (Li et al., 2025b; Bai et al., 2022). A more intuitive alternative is explicit memory via contextual prompting, which improves performance by injecting guidance directly into the input without modifying model weights. (Xu et al., 2025; Chhikara et al., 2025; Zhao et al., 2024).

However, both paradigms suffer from fundamental yet contrasting limitations. Explicit memory facilitates transparency by making reasoning steps externally visible through prompts; however, it lacks adaptability and struggles to generalize beyond specific tasks or contexts. Conversely, implicit

![](images/8e8e33013056a98ce25ac7b0be72ff1b494cba6a7bed09e0476ddb7b2a4866f6.jpg)  
Figure 1: Our method and existing approach Expel (Zhao et al., 2024).

memory enables generalization via training, but its black-box nature makes the contribution of specific past experiences inaccessible and difficult to interpret, while encoding knowledge directly into parameter space often incurs information loss and is vulnerable to catastrophic forgetting. This unresolved challenge motivates our central research question: Can we develop an agentic framework by leveraging dynamic, structured explicit memory to actively guide and enhance implicit policy learning?

This paper introduces a novel agent-centric, trainable, multi-layered graph memory framework and explores its integration with RL. First, we abstract episodic agent trajectories into canonical paths over a finite state machine, from which we derive high-level, generalizable meta-cognition. Second, we design a trainable graph architecture equipped with a reinforcement-driven weight optimization mechanism that calibrates the utility of stored strategies based on downstream task performance. Finally, the dynamic graph is operationalized as an explicit policy prior, selectively injecting high-quality strategies into the agent’s context during training. Empirical results across seven diverse question-answering benchmarks demonstrate that our framework delivers strong gains in both cross-task generalization and final task performance.

Our main contributions are threefold:

• We propose a novel agent-centric memory framework that abstracts low-level agent trajectories into canonical paths on a finite state machine, enabling the distillation of highlevel, generalizable meta-cognitive strategies.   
• We develop a reinforcement-driven weight optimization mechanism that dynamically calibrates the utility of memory connections, allowing the graph to selectively emphasize strategies with proven empirical effectiveness.   
• We demonstrate that incorporating this graph memory as an explicit policy prior within RL substantially enhances policy learning and final task performance.

Ultimately, this work presents a unified framework for creating more adaptive, efficient, and strategically-aware agents that not only act, but learn and reason from a continually evolving repository of their own experiences.

# 2 RELATED WORK

# 2.1 LLM AGENTS AND PLANNING WITH EXTERNAL TOOLS

LLM agents increasingly incorporate external tools to overcome reasoning limitations and expand their problem-solving capabilities. Early prompt-based approaches, including ReAct (Yao et al., 2022) and WebGPT (Nakano et al., 2021), demonstrate how agents can interleave reasoning and acting, embedding tool calls directly in the generation trace. Building on these foundations, Search-o1 introduces agentic RAG that dynamically retrieves knowledge during reasoning. Building on these foundations, Search-o1 (Li et al., 2025a) advances tool-augmented reasoning by enabling agents to autonomously decide when to invoke search tools during multi-step problem solving. Recent research has proposed more sophisticated coordination mechanisms using RL-based training (Sun et al., 2025; Zheng et al., 2025; Song et al., 2025). Search-R1 (Jin et al., 2025) represents a breakthrough RL framework that trains LLMs for alternating reasoning and search, enabling autonomous query generation and real-time information retrieval during step-by-step reasoning. Other recent approaches include optimized reward designs (Wang et al., 2025; Qian et al., 2025) and strategic tool

integration (Feng et al., 2025), with frameworks like RL-Factory (Chai et al., 2025) accelerating research in this domain.Despite these advances, the lack of explicit long-term memory for reusable tool-use patterns leaves deciding when and which tools to invoke as a key bottleneck. To address this limitation, we propose a differentiable graph-based memory system that encodes past decision paths into reusable strategic priors, enabling agents to systematically learn and generalize planning strategies across domains.

# 2.2 MEMORY ARCHITECTURES AND STRATEGIC LEARNING

Recent research has increasingly explored how to extract strategic knowledge and meta-cognition from agent experience. Reflexion (Zhang et al., 2023) equips agents with self-verbalized feedback to refine future behavior, while Expel (Zhao et al., 2024) identifies reusable reasoning trajectories to guide subsequent decisions. MEM1 (Zhou et al., 2025) and MemAgent (Yu et al., 2025) adapt memory usage over long-horizon tasks. A-MEM (Xu et al., 2025) builds dynamic memory notes that evolve with new inputs, Zep (Rasmussen et al., 2025)and HopRAG (Liu et al., 2025) construct logic-aware graphs to facilitate retrieval.

However, these methods typically apply graph structure in a static manner and lack mechanisms to assess or refine the utility of memory components. G-Memory (Zhang et al., 2025) demonstrates how hierarchical graph-based memory can evolve by assimilating new collaborative trajectories, enabling systems to leverage cross-trial knowledge and learn from prior experiences progressively. Pan & Zhao (2025)focus on whether different forms of memory can enhance reasoning. Xiong et al. (2025) investigate long-term memory evolution.While prior memory methods often rely on static storage or task-specific designs, they lack mechanisms for evaluating and refining strategies. In contrast, we propose a trainable graph-based memory that supports utility-aware strategy selection and reinforcement learning–driven updates, enabling generalizable and adaptive decision-making.

# 3 PRELIMINARIES

# 3.1 HETEROGENEOUS GRAPH STRUCTURE

Graphs provide a natural formalism for modeling structured dependencies among diverse entities. A heterogeneous graph (Zhang et al., 2019)can be defined as

$$
\mathcal {G} = (V, E, \mathcal {O} _ {V}, \mathcal {R} _ {E}, C),
$$

where $V$ denotes the set of nodes, $E \subseteq V \times V$ denotes the set of directed edges, ${ \mathcal { O } } _ { V }$ denotes the set of node types, $\mathcal { R } _ { E }$ denotes the set of relation types, and $C$ is the collection of node contents. Each edge $e = ( u , v , r ) \in E$ specifies a relation of type $r$ from node $u$ to node $v$ .

Connectivity in $\mathcal { G }$ is represented by node-type adjacency matrices

$$
A ^ {x y} \in \{0, 1 \} ^ {| V _ {x} | \times | V _ {y} |}, \quad (x, y) \in \mathcal {O} _ {V} \times \mathcal {O} _ {V},
$$

where $\nu _ { x }$ and $\mathcal { V } _ { y }$ denote the sets of nodes of type $x$ and $y$ , respectively. An entry $( A ^ { x y } ) _ { i j } = 1$ indicates that node $i$ of type $x$ is connected to node $j$ of type $y$ . This formulation emphasizes the structural dependencies across different node types.

To enable learning, each $A ^ { x y }$ is coupled with a weight matrix $W ^ { x y }$ , so that propagation is governed by the weighted operator $A ^ { x y } \odot W ^ { x y }$ . Thus, structure defines feasible paths, while weights determine effective information flow. Formally,

$$
\mathbf {H} _ {y} = \sigma \left(\left(A ^ {x y} \odot W ^ {x y}\right) ^ {\top} \mathbf {H} _ {x}\right),
$$

where $\mathbf { H } _ { x }$ are input values and $\sigma ( \cdot )$ denotes an activation function.

# 3.2 LLM AGENTS WITH TOOL-AUGMENTED REASONING

The interaction between a LLM and external tools can be formalized as a structured multi-turn decision process (Chai et al., 2025). At each time step $t$ , the agent observes

$$
s _ {t} = (q, h _ {1: t - 1}), \quad a _ {t} \sim \pi_ {\theta} (a _ {t} \mid s _ {t}).
$$

where $q$ is the user query and $h _ { 1 : t - 1 }$ is the dialogue or reasoning history, then generates an action $a$ which may correspond to internal reasoning, a tool invocation, or answer generation, using a protocol with tags such as <think>, <tool call>, and <answer>.

The process continues until either the tag <answer></answer> is generated or the agent has issued up to a maximum of $K$ tool invocations. A trajectory $\tau = ( s _ { 1 } , a _ { 1 } , o _ { 1 } , \dots , s _ { T } , a _ { T } , o _ { T } )$ yields reward $R ( \tau )$ , where $o _ { t }$ denotes the environment observation, i.e., tool outputs if $a _ { t }$ is a tool call, and the policy is optimized via

$$
J (\theta) = \mathbb {E} _ {\tau \sim \pi_ {\theta}} [ R (\tau) ], \quad \nabla_ {\theta} J (\theta) \approx \mathbb {E} _ {\tau} \Bigl [ \sum_ {t = 1} ^ {T} \nabla_ {\theta} \log \pi_ {\theta} (a _ {t} \mid s _ {t}) \hat {A} _ {t} \Bigr ].
$$

# 4 METHOD

In this section, we detail our proposed method in three stages. First, we describe how to construct a memory graph that encodes decision trajectories and strategic principles. Second, we present the learning framework for optimizing the weights within this memory graph. Finally, we explain how this structured memory is integrated into the RL training process to guide agent behavior and improve learning efficiency.The overall process of our method is shown in the Figure 2.

# 4.1 STAGE 1: HIERARCHICAL MEMORY GRAPH CONSTRUCTION

Memory Graph Structure. We instantiate the memory as a heterogeneous graph with node set $V = \mathcal { Q } \cup \mathcal { T } \cup \mathcal { M }$ and directed edges $E \subseteq ( \mathcal { Q } \times \mathcal { T } ) \cup ( \mathcal { T } \times \mathcal { M } )$ . Each node type forms a distinct layer in the hierarchy, consistent with the structural depiction in figure 2 (stage 1):

• Query Layer $( \mathcal { Q } )$ : Formed by query nodes $q _ { i }$ , each representing a task instance (e.g., a user query), including input, execution trajectory, and outcome labels. Since a single query may yield multiple responses, each $q _ { i }$ can connect to one or more transition paths $\{ t _ { j } \}$ via edges $( q _ { i } \to t _ { j }$ ).   
• Transition Path Layer $( \mathcal { T } )$ : Formed by path nodes $t _ { j }$ , each denoting a canonical decision pathway derived from a finite state machine $s$ that abstracts raw execution traces into standardized behavioral patterns. These nodes are linked to the meta-cognition layer through edges $( t _ { j } \to m _ { k } )$ ).   
• Meta-Cognition Layer $( \mathcal { M } )$ : Formed by meta-cognition nodes $m _ { k }$ , each encoding a highlevel strategic principle distilled from both successful and failed paths, serving as generalized heuristics for problem-solving.

Connectivity is encoded by bipartite adjacency matrices

$$
A ^ {q \rightarrow t} \in \{0, 1 \} ^ {| \mathcal {Q} | \times | \mathcal {T} |}, \quad A ^ {t \rightarrow m} \in \{0, 1 \} ^ {| \mathcal {T} | \times | \mathcal {M} |},
$$

augmented with learnable weights $w _ { q t }$ and $w _ { t m }$ respectively. Information flows as a weighted aggregation process from queries to meta-cognition, forming a directed acyclic topology.

Finite State Machine. To obtain a standardized and comparable representation of agent behaviors across tasks, we define a Finite State Machine (FSM) $ { \boldsymbol { S } } \ = \ (  { \boldsymbol { S } } ,  { \boldsymbol { A } } ,  { \boldsymbol { T } } )$ . Here, $S$ denotes abstract cognitive states (e.g., StrategyPlanning, InformationAnalysis), $A$ is the action space, and $T : S \times A \to S$ is the transition function. Each raw execution trajectory comprising tool invocations or reasoning steps is mapped onto a canonical path $t _ { j }$ within this FSM. This grounding enables structured comparison across queries while filtering execution-level noise, ensuring that the memory graph preserves only semantically meaningful decision points. The detailed specification of $s$ is provided in the Appendix B.1.

Meta-Cognition Induction. Meta-cognitions are induced by analyzing the canonical decision pathways. For each query $q _ { i }$ , the agent samples trajectories $\{ \bar { \tau _ { 1 } } ^ { ( i ) } , \ldots , \bar { \tau _ { N } ^ { ( i ) } } \}$ from its policy $\pi$ . If both successful $( \tau _ { s } )$ and failed $( \tau _ { f } )$ trajectories exist, contrasting their FSM paths yields a highconfidence meta-cognition $m _ { k }$ that explains the outcome divergence. If only failures occur, the

![](images/e6ce761b34bb7a10441901a37e0933c38e25767f65770491258d89c5567464ae.jpg)  
Figure 2: The framework of the proposed trainable memory. Stage 1 builds a graph from LLM trajectories, encoding queries, decision paths, and meta-cognition. Stage 2 estimates strategy utility via counterfactual rewards and updates graph weights. Stage 3 injects top-k strategies into RL training for policy optimization.

agent retrieves top- $K$ semantically similar queries $\mathrm { S i m } ( q _ { i } , q _ { j } ) = \cos ( { \bf e } _ { q _ { i } } , { \bf e } _ { q _ { j } } )$ , and derives speculative meta-cognitions from successful paths of these neighbors:

$$
\mathcal {M} ^ {\mathrm {s p e c}} (q _ {i}) = \bigcup_ {q _ {j} \in \operatorname {T o p K} (q _ {i})} \left\{m _ {k} \mid t _ {j} \in \operatorname {S u c c e s s P a t h s} (q _ {j}), m _ {k} \in \mathcal {M} (t _ {j}) \right\}.
$$

Concrete examples and the corresponding prompts are provided in Appendix E.3.

Meta-Cognition Update. The memory graph is dynamically updated to preserve relevance and utility. When a new decision path is generated, the agent evaluates its strategic value: reinforcing existing principles updates their confidence, novel patterns lead to new meta-cognition nodes, and redundant or low-confidence paths are discarded. This selective process curates a concise set of strategic principles that evolves with experience.

The hierarchical structure thus abstracts low-level trajectories into reusable strategies. At inference, the memory graph $\mathcal { G }$ functions as a structured policy prior guiding decision-making, while during training it provides supervision signals for reward-driven consolidation of meta-cognitive knowledge.

# 4.2 STAGE 2: TRAINABLE GRAPH WEIGHT OPTIMIZATION

The memory graph provides structural priors, but not all meta-cognitions contribute equally. To adaptively capture their utility, we introduce a reinforcement-driven weight optimization procedure.

Parameterizing the Graph for Utility Learning. We parameterize the memory graph $\mathcal { G }$ as a sparsely connected weighted network, where each edge is associated with a trainable coefficient reflecting its utility. Given query featuresaggregation: $\mathbf { H } _ { \mathcal { Q } } ^ { ( 0 ) }$ , information propagates through the graph via weighted

$$
\mathbf {H} _ {\mathcal {T}} ^ {(1)} = \sigma \Big ((A _ {q t} \odot W _ {q t}) ^ {\top} \mathbf {H} _ {\mathcal {Q}} ^ {(0)} \Big), \quad \mathbf {H} _ {\mathcal {M}} ^ {(2)} = \sigma \Big ((A _ {t m} \odot W _ {t m}) ^ {\top} \mathbf {H} _ {\mathcal {T}} ^ {(1)} \Big),
$$

which corresponds to the flow from the query layer, through the transition layer, and finally to the meta-cognition layer in Figure 2.

In our formulation, a new query is represented by its similarity to historical queries in the graph, and the top- $k$ most relevant neighbors are selected to activate a task-specific subgraph $\mathcal { G } ( q _ { \mathrm { n e w } } ) =$ $( \mathcal { Q } ^ { \prime } , \mathcal { T } ^ { \prime } , \bar { \mathcal { M } } ^ { \prime } )$ .Within this subgraph, a candidate meta-cognition $m _ { k } \in \mathcal { M } ^ { \prime }$ is sampled according to a relevance score $\rho ( m _ { k } )$ , derived from the learned graph weights.

To estimate its empirical utility, we contrast two trajectories: one guided by $m _ { k }$ , which yields reward $R _ { \mathrm { w i t h } } ( m _ { k } )$ , and another without such guidance, yielding reward $R _ { \mathrm { w / o } }$ . The resulting reward gap $\Delta R _ { k } = R _ { \mathrm { w i t h } } ( m _ { k } ) - R _ { \mathrm { w / o } }$ is employed as a utility signal, quantifying the marginal contribution of $m _ { k }$ to overall task performance.

Policy Gradient-Based Weight Optimization. The relevance score $\rho ( m _ { k } \mid q _ { \mathrm { n e w } } )$ is computed by aggregating path strengths from historical queries and transitions leading to $m _ { k }$ :

$$
\rho (m _ {k} \mid q _ {\mathrm {n e w}}) = \sum_ {q _ {i}, t _ {j}: q _ {i} \rightarrow t _ {j} \rightarrow m _ {k}} \operatorname {S i m} (q _ {\mathrm {n e w}}, q _ {i}) \cdot w _ {q t} ^ {(i, j)} \cdot w _ {t m} ^ {(j, k)}.
$$

Using a softmax over these scores, the selection probability $p ( m _ { k } \mid q _ { \mathrm { n e w } } ) \propto \exp ( \rho ( m _ { k } \mid q _ { \mathrm { n e w } } ) ) .$ .

We apply the REINFORCE algorithm to optimize the weights:

$$
\mathcal {L} _ {\mathrm {R L}} = - \mathbb {E} _ {m _ {k} \sim p} \left[ \Delta R _ {k} \cdot \log p \left(m _ {k} \mid q _ {\text {n e w}}\right) \right].
$$

A positive $\Delta R _ { k }$ increases the relevance score and strengthens the supporting paths, while a negative $\Delta R _ { k }$ decreases them, enabling the memory graph to refine itself over time.

# 4.3 STAGE 3: MEMORY-GUIDED POLICY OPTIMIZATION

Departing from prior works that leverage memory solely during inference, our framework explicitly integrates the structured memory into the training loop. Meta-cognitive strategies are dynamically retrieved from the optimized memory graph and incorporated into the agent’s context, serving as high-level strategic priors that guide the reinforcement learning process.

Strategic Context Retrieval. For each training instance $q _ { \mathrm { t r a i n } }$ , we compute a relevance score for every meta-cognition node $m \in \mathcal { M }$ . This score is derived from the aggregated weights of all paths connecting the corresponding query node to the meta-cognition node within the memory graph $\mathcal { G }$ (as formulated in Section 4.2). We then select the top- $k$ meta-cognitions $\{ m _ { 1 } , \ldots , m _ { k } \}$ with the highest scores. This mechanism ensures that the guidance is not only relevant but also grounded in empirically successful past trajectories, as encoded by the learned edge weights.

The retrieved strategies are verbalized and prepended to the original query to form an augmented prompt, $\tilde { q } _ { \mathrm { t r a i n } } = \left[ \stackrel { \smile } { m _ { 1 } } , \qquad m _ { 2 } , \dots , m _ { k } \ ; \ q _ { \mathrm { t r a i n } } \right]$ , this augmented prompt serves as the input to the policy network.

Optimization Objective. The agent’s policy, $\pi _ { \theta }$ , is optimized to maximize the expected cumulative reward conditioned on the augmented context. We employ a policy gradient method, where the parameters $\theta$ are updated by minimizing the following loss function:

$$
\mathcal {L} _ {\mathrm {R L} + \mathrm {M e m}} = - \mathbb {E} _ {a \sim \pi_ {\theta} (\cdot | \tilde {q} _ {\text {t r a i n}})} [ R (a) ].
$$

Table 1: Performance comparison across seven QA datasets in inference. † indicates in-domain datasets, while ⋆ denotes out-of-domain datasets. Percentages in Avg. column denote relative improvement over ITR.   

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Avg. (↑/↓ vs. ITR)</td><td colspan="3">General QA</td><td colspan="4">Multi-Hop QA</td></tr><tr><td>NQ*</td><td>TriviaQA*</td><td>PopQA*</td><td>HotpotQA†</td><td>2wiki*</td><td>Musique*</td><td>Bamboogle*</td></tr><tr><td>Qwen3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ITR</td><td>0.334 (-)</td><td>0.275</td><td>0.593</td><td>0.358</td><td>0.325</td><td>0.324</td><td>0.094</td><td>0.365</td></tr><tr><td>Direct Inference</td><td>0.269 (↓19.5%)</td><td>0.200</td><td>0.519</td><td>0.191</td><td>0.230</td><td>0.275</td><td>0.058</td><td>0.410</td></tr><tr><td>CoT</td><td>0.252 (↓24.6%)</td><td>0.209</td><td>0.512</td><td>0.182</td><td>0.223</td><td>0.271</td><td>0.055</td><td>0.308</td></tr><tr><td>Direct Trajectory</td><td>0.352 (↑5.4%)</td><td>0.317</td><td>0.604</td><td>0.380</td><td>0.329</td><td>0.363</td><td>0.105</td><td>0.364</td></tr><tr><td>A-MEM</td><td>0.334 (0.0%)</td><td>0.286</td><td>0.590</td><td>0.366</td><td>0.339</td><td>0.332</td><td>0.112</td><td>0.313</td></tr><tr><td>EXPEL</td><td>0.329 (↓1.5%)</td><td>0.306</td><td>0.594</td><td>0.379</td><td>0.317</td><td>0.327</td><td>0.092</td><td>0.287</td></tr><tr><td>Ours</td><td>0.365(↑9.3%)</td><td>0.316</td><td>0.622</td><td>0.382</td><td>0.358</td><td>0.354</td><td>0.128</td><td>0.392</td></tr><tr><td>Qwen3-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ITR</td><td>0.279 (-)</td><td>0.298</td><td>0.581</td><td>0.157</td><td>0.268</td><td>0.281</td><td>0.077</td><td>0.290</td></tr><tr><td>Direct Inference</td><td>0.211 (↓24.4%)</td><td>0.158</td><td>0.413</td><td>0.157</td><td>0.183</td><td>0.240</td><td>0.033</td><td>0.290</td></tr><tr><td>CoT</td><td>0.181 (↓35.1%)</td><td>0.149</td><td>0.375</td><td>0.146</td><td>0.156</td><td>0.190</td><td>0.022</td><td>0.228</td></tr><tr><td>Direct Trajectory</td><td>0.325 (↑16.5%)</td><td>0.310</td><td>0.558</td><td>0.379</td><td>0.282</td><td>0.344</td><td>0.076</td><td>0.327</td></tr><tr><td>A-MEM</td><td>0.319 (↑14.3%)</td><td>0.310</td><td>0.586</td><td>0.381</td><td>0.272</td><td>0.269</td><td>0.091</td><td>0.325</td></tr><tr><td>EXPEL</td><td>0.321 (↑15.1%)</td><td>0.312</td><td>0.570</td><td>0.388</td><td>0.294</td><td>0.347</td><td>0.075</td><td>0.263</td></tr><tr><td>Ours</td><td>0.351 (↑25.8%)</td><td>0.335</td><td>0.596</td><td>0.393</td><td>0.299</td><td>0.347</td><td>0.099</td><td>0.391</td></tr></table>

This tight integration ensures that the policy does not learn in isolation but is continually guided by a dynamically evolving corpus of strategic knowledge. This allows the agent to effectively bootstrap its learning process from a distilled representation of past successes.

In practice, we adopt the Generalized Reinforcement Policy Optimization (GRPO) algorithm to optimize the memory-augmented policy. The GRPO loss can be written as:

$$
\mathcal {L} _ {\mathrm {G R P O}} = - \mathbb {E} _ {t} \left[ \min  \left(\frac {\pi_ {\theta} (a _ {t} \mid \tilde {q} _ {\text {t r a i n}})}{\pi_ {\theta_ {\text {o l d}}} (a _ {t} \mid \tilde {q} _ {\text {t r a i n}})} \hat {A} _ {t}, \operatorname {c l i p} \left(\frac {\pi_ {\theta} (a _ {t} \mid \tilde {q} _ {\text {t r a i n}})}{\pi_ {\theta_ {\text {o l d}}} (a _ {t} \mid \tilde {q} _ {\text {t r a i n}})}, 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {t}\right) \right],
$$

where $\hat { A } _ { t }$ is the advantage estimator and $\epsilon$ the clipping parameter.

# 5 EXPERIMENT

# 5.1 DATASETS

To evaluate the effectiveness and generalizability of our approach, we conduct experiments on seven widely-used Question-Answering(QA) datasets, covering both single-turn and multi-hop reasoning tasks. (1) General QA Datasets: We include Natural Questions (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), and PopQA (Mallen et al., 2022), which consist of open-domain factoid questions requiring retrieval and basic reasoning capabilities. (2) Multi-hop QA Datasets: For more complex reasoning scenarios, we adopt HotpotQA (Yang et al., 2018), 2WikiMultiHopQA (Ho et al., 2020), Musique (Trivedi et al., 2022), and Bamboogle (Press et al., 2022), which require integrating information across multiple documents.

# 5.2 BASELINE EVALUATION

To comprehensively evaluate the effectiveness of our proposed method, we design experiments from two complementary perspectives: (1) Direct Inference Impact: We assess how the integration of our memory workflow influences model performance in zero-training settings, i.e., during direct inference. (2) Training Impact: We investigate how the memory architecture affects RL training dynamics, focusing on convergence speed and the final performance achieved. Detailed baseline configurations are provided in Appendix A.2.

# 5.3 MAIN RESULTS

Experimental Analysis: Memory-Guided Inference. The detailed inference results are summarized in Table 1. On the 8B-scale model, our method demonstrates strong competitiveness, achiev-

ing an average score of 0.365, which represents a notable $+ 9 . 3 \%$ relative improvement over the ITR baseline and ranks first among all contenders. The advantages of our method become even more dramatic on the smaller Qwen3-4B model. It achieves a staggering $+ 2 5 . 8 \%$ relative improvement in average performance over the ITR baseline, this significant performance improvement on a model with limited capacity suggests that our method effectively addresses its inherent deficiencies by providing a robust and structured reasoning framework.

A particularly noteworthy finding is that the memory component of our method was constructed exclusively using data from HotpotQA, the single in-domain dataset. Despite this, our method not only excels on HotpotQA but also achieves state-of-the-art or highly competitive performance across all out-of-domain datasets, including NQ, TriviaQA, PopQA, and 2wiki. This outcome is a strong testament to the remarkable generalization capability of our approach. It demonstrates that the reasoning structures learned from HotpotQA are not merely overfitted patterns.

Table 2: Performance comparison across seven QA datasets in training. Avg. column also reports relative improvement $( \% )$ compared to Search-R1 as the base. † indicates in-domain datasets, while ⋆ denotes out-ofdomain datasets.   

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Avg. (↑/↓ vs. Search-R1)</td><td colspan="3">General QA</td><td colspan="4">Multi-Hop QA</td></tr><tr><td>NQ*</td><td>TriviaQA*</td><td>PopQA*</td><td>HotpotQA†</td><td>2wiki*</td><td>Musique*</td><td>Bamboogle*</td></tr><tr><td colspan="9">Qwen3-8B</td></tr><tr><td>Search-R1</td><td>0.395 (-)</td><td>0.384</td><td>0.651</td><td>0.429</td><td>0.391</td><td>0.386</td><td>0.143</td><td>0.380</td></tr><tr><td>Direct Trajectory</td><td>0.400 (↑1.27%)</td><td>0.406</td><td>0.657</td><td>0.433</td><td>0.376</td><td>0.367</td><td>0.139</td><td>0.423</td></tr><tr><td>A-MEM</td><td>0.403(↑2.03%)</td><td>0.398</td><td>0.656</td><td>0.436</td><td>0.389</td><td>0.409</td><td>0.138</td><td>0.398</td></tr><tr><td>EXPEL</td><td>0.371 (↓6.08%)</td><td>0.362</td><td>0.621</td><td>0.407</td><td>0.354</td><td>0.375</td><td>0.121</td><td>0.357</td></tr><tr><td>Ours</td><td>0.408 (↑3.29%)</td><td>0.386</td><td>0.662</td><td>0.434</td><td>0.387</td><td>0.403</td><td>0.152</td><td>0.435</td></tr><tr><td colspan="9">Qwen3-4B</td></tr><tr><td>Search-R1</td><td>0.375 (-)</td><td>0.357</td><td>0.625</td><td>0.426</td><td>0.354</td><td>0.402</td><td>0.115</td><td>0.348</td></tr><tr><td>Direct Trajectory</td><td>0.415 (↑10.67%)</td><td>0.403</td><td>0.624</td><td>0.434</td><td>0.420</td><td>0.428</td><td>0.186</td><td>0.412</td></tr><tr><td>A-MEM</td><td>0.388 (↑3.47%)</td><td>0.393</td><td>0.603</td><td>0.439</td><td>0.385</td><td>0.322</td><td>0.157</td><td>0.418</td></tr><tr><td>EXPEL</td><td>0.337 (↓10.13%)</td><td>0.322</td><td>0.577</td><td>0.399</td><td>0.311</td><td>0.363</td><td>0.081</td><td>0.305</td></tr><tr><td>Ours</td><td>0.426 (↑13.60%)</td><td>0.408</td><td>0.646</td><td>0.462</td><td>0.410</td><td>0.407</td><td>0.189</td><td>0.463</td></tr></table>

Experimental Analysis: Memory-Guided Reinforcement Learning. We further evaluated our method by integrating it into RL training process. The detailed training results are in Table 2.

On the Qwen3-8B model, our method achieves the best average performance (0.408), improving upon Search-R1 baseline by $3 . 2 9 \%$ . This shows that our method provides additional benefits even after the model is already optimized with RL. The gains are most notable on challenging outof-domain datasets like TriviaQA and Bamboogle, suggesting our memory helps the RL agent learn more general reasoning strategies that transfer well to new tasks.

On the smaller Qwen3-4B model, the results are even more impressive. Our method achieves a remarkable $1 \bar { 3 } . 6 0 \%$ relative improvement over Search-R1. As seen in our inference experiments, the benefit of our method is especially pronounced on smaller models. Remarkably, our trained Qwen3-4B model (0.426) outperforms the baseline Qwen3-8B model (0.395), demonstrating a significant gain in efficiency.

![](images/1c7020004adcfbc63a3178b0b49f0cb83016c7bcaf6e8167d47ce6121e367456.jpg)  
Figure 3: (a) Training curve of 4B models.   
(b) Training curve of 8B models.

In summary, adding our method to inference or

RL training framework significantly boosts QA performance, especially for smaller models. Our structured memory helps the model learn general reasoning skills from the in-domain HotpotQA data and apply them successfully to other datasets. This allows smaller models to match or even exceed the performance of larger ones, offering a path to more efficient and capable models.

# 5.4 ABLATION STUDIES

We conduct ablation studies across three dimensions: (1) disabling memory weight updates (2) varying the number of meta-cognitions used as context and (3) altering the granularity of memory composition (i.e., API call structure).

![](images/269bd4ca31056594ed2d47f71141ab645c4b032fa6ae2db3697d0c278c3e989b.jpg)  
(a)

![](images/4c7cfa84779743430a32ba915909ed375fbd875a8b410de26d83665e596d7106.jpg)  
(b)

![](images/d15504db7c31da8d0f25814ed11b8da38e665096efeaea1a64674c76e3c25bae.jpg)  
(c)

![](images/a50d5e7360855c1210126c9f71562ca9f5d3b47258930fcf462ee8e57fb80749.jpg)  
(d)   
Figure 4: Ablation studies of the structured memory framework. (a) and (b) show the effect of disabling weight optimization. (c) varying the number of meta-cognition $k$ . (d) generalization across LLM backends.

Effect of Disabling Weight Optimization. We first examine the impact of freezing the memory graph weights (i.e., no learning of edge confidence). In this setup, we keep all memory edges at uniform weight and retrieve strategies purely based on structural presence. As shown in figure 4(a)(b), performance drops significantly, particularly on 2WikiMultiHopQA, indicating that learning to prioritize high-utility memory connections is crucial for effective strategy reuse. This validates our reinforcement-based update mechanism, which helps distinguish broadly useful meta-cognitions from less effective or overly specific ones.

Varying the Number of Meta-Cognitions. We further evaluate how the number of retrieved metacognitive strategies $( k )$ affects model performance. Figure 4(c) presents the average accuracy of the 4B model on seven benchmarks as a function of the number of meta-cognitions. Increasing $k$ from 0 (no memory) to 3 leads to steady improvement, as more strategic signals are injected into the prompt. However, further increasing $k$ yields diminishing returns and can even introduce noise due to overlapping or irrelevant strategies. This highlights a trade-off between strategy diversity and clarity, and suggests that a moderate value of $k = 3$ offers the best balance between guidance and prompt efficiency. The detailed results are shown in Table 3.

Generalization across LLMs backends. To evaluate whether our memory construction is tied to a specific LLM API, we replace the original OpenAI gpt-4o model with Gemini-2.5-pro and rerun the downstream evaluation using the same memory graph. As shown in Table 4, our memory-augmented approach consistently outperforms its non-memory counterpart even under a different LLM backend, though the absolute numbers differ slightly due to model capability gaps. This demonstrates that our structured memory graph and retrieval-guided prompting strategy are largely model-agnostic, enabling plug-and-play use across modern foundation models.

# 6 CONCLUSION

In this paper, we address the dual challenges of inefficient decision-making and poor experience reuse in LLM-based agents. We introduce a trainable, multi-level graph memory framework that structurally encodes historical queries, policy trajectories, and high-level metacognitive strategies. This design facilitates explicit strategy recall and integrates memory into the RL loop to guide and accelerate policy optimization.

Unlike prior works that rely on either implicit optimization or static prompting, our approach unifies explicit memory with dynamic learning. By updating memory weights via RL signals, the framework selectively reinforces high-utility strategies and re-injects them into the agent’s training process through prompt augmentation. This mechanism promotes strategic transfer and generalization from past experiences. Our experiments demonstrate that this method not only improves reasoning accuracy at inference time but also accelerates convergence during RL training, ultimately yielding superior final performance and strong generalization across diverse tasks.

# REFERENCES

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom Brown, Jack Clark, Sam McCandlish, Chris Olah, Ben Mann, and Jared Kaplan. Training a helpful and harmless assistant with reinforcement learning from human feedback, 2022. URL https://arxiv.org/abs/2204.05862.   
Jiajun Chai, Guojun Yin, Zekun Xu, Chuhuai Yue, Yi Jia, Siyu Xia, Xiaohan Wang, Jiwen Jiang, Xiaoguang Li, Chengqi Dong, Hang He, and Wei Lin. Rlfactory: A plug-and-play reinforcement learning post-training framework for llm multi-turn tool-use, 2025. URL https://arxiv. org/abs/2509.06980.   
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory, 2025. URL https://arxiv. org/abs/2504.19413.   
Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms, 2025. URL https://arxiv.org/abs/2504.11536.   
Leo Gao, Angela Beese, Max Fischer, Lariah Hou, Julia Kreutzer, Xi Victoria Lin, Jason Phang Wang, Luke Zettlemoyer, and Emiel van Miltenburg. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.   
Chia-Hsuan Ho, Shu-Hung Yeh, and Yun-Nung Chen. Constructing a multi-hop qa dataset via graph-based node ranking. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6151–6161, 2020.   
Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning, 2025. URL https://arxiv.org/abs/2503.09516.   
Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.   
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi ˘ Chen, and Wen tau Yih. Dense passage retrieval for open-domain question answering, 2020. URL https://arxiv.org/abs/2004.04906.   
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 2019.   
Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. Search-o1: Agentic search-enhanced large reasoning models, 2025a. URL https://arxiv.org/abs/2501.05366.   
Zhiyu Li, Shichao Song, Chenyang Xi, Hanyu Wang, Chen Tang, Simin Niu, Ding Chen, Jiawei Yang, Chunyu Li, Qingchen Yu, Jihao Zhao, Yezhaohui Wang, Peng Liu, Zehao Lin, Pengyuan Wang, Jiahao Huo, Tianyi Chen, Kai Chen, Kehang Li, Zhen Tao, Huayi Lai, Hao Wu, Bo Tang, Zhenren Wang, Zhaoxin Fan, Ningyu Zhang, Linfeng Zhang, Junchi Yan, Mingchuan Yang, Tong Xu, Wei Xu, Huajun Chen, Haofen Wang, Hongkang Yang, Wentao Zhang, Zhi-Qin John Xu, Siheng Chen, and Feiyu Xiong. Memos: A memory os for ai system, 2025b. URL https: //arxiv.org/abs/2507.03724.   
Hao Liu, Zhengren Wang, Xi Chen, Zhiyu Li, Feiyu Xiong, Qinhan Yu, and Wentao Zhang. Hoprag: Multi-hop reasoning for logic-aware retrieval-augmented generation, 2025. URL https:// arxiv.org/abs/2502.12442.

Eugene Mallen, Patrick Lewis, Semih Yavuz, Raymond Ng, Sebastian Riedel, Dhruvesh Sridhar, and Fabio Petroni. Popqa: Open-domain qa with popular questions. In Findings of the Association for Computational Linguistics: EMNLP 2022, 2022.   
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, and et al. Webgpt: Browser-assisted question answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.   
Bo Pan and Liang Zhao. Can past experience accelerate llm reasoning?, 2025. URL https: //arxiv.org/abs/2505.20643.   
Ofir Press, Daniel Khashabi, Ashish Sabharwal, and Tushar Khot. Measuring and narrowing the compositionality gap in language models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2022.   
Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tur, Gokhan ¨ Tur, and Heng Ji. Toolrl: Reward is all tool learning needs, 2025. URL https://arxiv. org/abs/2504.13958.   
Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A temporal knowledge graph architecture for agent memory, 2025. URL https://arxiv.org/ abs/2501.13956.   
Arjun Singh, Zhe Xiang, Dongheon Kim, Michael Ahn, Jesse Thomason, and Julie Shah. Progprompt: Generating situated robot task plans using large language models. arXiv preprint arXiv:2305.15343, 2023.   
Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen. R1-searcher: Incentivizing the search capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2503.05592.   
Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, and Jingren Zhou. Zerosearch: Incentivize the search capability of llms without searching, 2025. URL https://arxiv.org/abs/2505.04588.   
Harsh Trivedi, Daniel Khashabi, Tushar Khot, Ashish Sabharwal, and Dan Roth. Musique: Multihop questions via single-hop question composition. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics, 2022.   
Hongru Wang, Cheng Qian, Wanjun Zhong, Xiusi Chen, Jiahao Qiu, Shijue Huang, Bowen Jin, Mengdi Wang, Kam-Fai Wong, and Heng Ji. Acting less is reasoning more! teaching model to act efficiently, 2025. URL https://arxiv.org/abs/2504.14870.   
Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training, 2024. URL https://arxiv.org/abs/2212.03533.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Jiliang Tang, Himabindu Lakkaraju, and Zhen Xiang. How memory management impacts llm agents: An empirical study of experience-following behavior, 2025. URL https://arxiv.org/abs/2505.16067.   
Wujiang Xu, Kai Mei, Hang Gao, Juntao Tan, Zujie Liang, and Yongfeng Zhang. A-mem: Agentic memory for llm agents, 2025. URL https://arxiv.org/abs/2502.12110.   
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 2369–2380, 2018.

Shinn Yao, Jeffrey Zhao, Dian Yu, Hyung Won Chung, Jaybos Chen, Karthik Narasimhan, and et al. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.   
Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, and Hao Zhou. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent, 2025. URL https://arxiv.org/abs/2507. 02259.   
Chuxu Zhang, Dongjin Song, Chao Huang, Ananthram Swami, and Nitesh V Chawla. Heterogeneous graph neural network. In Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, pp. 793–803, 2019.   
Guibin Zhang, Muxin Fu, Guancheng Wan, Miao Yu, Kun Wang, and Shuicheng Yan. G-memory: Tracing hierarchical memory for multi-agent systems, 2025. URL https://arxiv.org/ abs/2506.07398.   
Shinn Zhang, Eric Wu, Frank Xu, Stuart Russell, and Dragomir Radev. Reflexion: Language agents with verbal reinforcement learning. arXiv preprint arXiv:2303.11366, 2023.   
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners, 2024. URL https://arxiv.org/abs/2308.10144.   
Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments, 2025. URL https://arxiv.org/abs/2504.03160.   
Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents, 2025. URL https://arxiv.org/abs/2506.15841.

# A EXPERIMENTAL SETUP DETAILS

# A.1 MULTI-TURN TOOL-INTERGRATED QA

When tackling QA benchmarks, we observe that incorporating external knowledge retrieval plays a crucial role in enhancing answer accuracy. To this end, we utilize the 2018 Wikipedia dump as our external knowledge base and adopt the E5 retriever for efficient document retrieval. Within our framework, the entire retrieval process is encapsulated as a modular MCP tool, which the LLM can invoke autonomously based on its reasoning needs.

Specifically, the Qwen3 model follows a structured interaction protocol in which reasoning and planning are carried out between <think> and </think> tags,tool invocation commands are embedded within <tool call> and </tool call>, and the corresponding outputs are captured between <tool response> and </tool response>. The model engages in multiple rounds of tool interaction until it either generates a final answer enclosed in <answer></answer> or reaches the maximum limit of tool invocations, which is set to six in our experiments. A concrete case study is presented in the following table to illustrate how this structured retrieval mechanism enables the model to access external knowledge and perform informed reasoning.

The table E.1presents a representative prompt designed to encourage tool usage, along with a concrete case study A.1demonstrating how the model leverages structured tool interactions to retrieve external knowledge and perform reasoning.

# A.2 EXPERIMENTAL SETTINGS

Direct Inference Baselines. We compare our method against the following direct inference baselines:Direct Inference $^ +$ Chain-of-Thought (CoT) (Wei et al., 2022) performs reasoning solely through its internal chain of thought. TIR (Tool-Integrated Reasoning) (Chai et al., 2025) enhances inference by retrieving documents using integrated search tools. For Reasoning with Memory, we compare with three variants: (1) Direct Trajectory, which uses past trajectories directly

# Case 1: Example Response with Tool-Integrated Reasoning halign title

response:<think> Okay, let’s tackle this question. The user is asking which U.S. state was historically dominated by the Republican party from 1860 to 1932 and is represented by State Senator George D. Maziarz.

First, I need to figure out the state in question. The key points here are the time period (1860-1932) and the Republican dominance. I remember that during the 19th century, the Republican Party was a major political force, especially after the Civil War. States like Ohio, New York, and Pennsylvania were known for their Republican strongholds. However, I’m not entirely sure which one fits the specific time frame mentioned.

Next, the mention of State Senator George D. Maziarz. I need to check if this senator is from a state that fits the historical Republican dominance. But I don’t recall specific information about George D. Maziarz. Maybe I should look up his state of origin. However, since I can’t access external resources right now, I need to rely on my existing knowledge.

```html
...</think>   
<tool_call> {name": "search-query_rag", arguments": {"query": "George D. Maziarz state", "topk": 3}} </tool_call>}   
<tool_response> Execute the tool search-query_rag succeeded - The args are: 'query': 'George D. Maziarz state', 'topk': 3 - The result is:.... </tool_response>   
<answer>New York</answer> 
```

as memory; (2) A-MEM (Xu et al., 2025), which maintains a dynamic memory graph; and (3) Expel (Zhao et al., 2024), which extracts high-level strategic insights from past experiences.

Reinforcement Learning Baselines. For RL training, we evaluate two groups of baselines: Search-R1: (Jin et al., 2025) A reinforcement learning agent that relies solely on multi-turn tool invocation without any memory support. RL with Memory Variants: We examine the performance of agents equipped with the three memory types described above—Direct Trajectory, A-MEM, and Expel Memory—to assess how different memory designs impact training efficiency and overall performance.

We conduct experiments with two model scales, Qwen-3-4B and Qwen-3-8B. For the retrieval component, we adopt the 2018 Wikipedia dump (Karpukhin et al., 2020) as the knowledge source and employ the E5 (Wang et al., 2024) retriever.

We exclusively use the HotpotQA dataset, both for model optimization and for constructing memory during the memory formation process. Evaluation is then carried out on the test or validation sets of seven diverse datasets, enabling assessment of performance both within the training domain and in out-of-domain settings. We report Exact Match (EM) as the primary evaluation metric. And for memory construction in A-Mem (Xu et al., 2025), Expel (Zhao et al., 2024), and our proposed method, where a high-capability large language model is required, we utilized GPT-4o.

We conduct experiments on seven datasets, where HotpotQA is selected as the in-domain test set, while the remaining six datasets are used for out-of-domain evaluation. From the HotpotQA training set, we sample 1,000 examples to construct the memory and an additional 5,000 examples for weight training.

During the RL training phase, we use the rest of the HotpotQA training set as the training corpus. We adopt a batch size of 512 with a micro batch size of 64, and the rollout sampling is performed with a temperature of 1.0. To accelerate the rollout process of the LLM, we deploy vLLM v1 with a tensor parallel size of 1.

Specifically for the GRPO algorithm, the number of rollout samples (n) is set to 8. All experiments are conducted on a cluster of 8 NVIDIA A100 GPUs.

# B MEMORY GRAPH CONSTRUCTION

The pseudo-code for the overall process of the graph, which is composed of the specific paths of LLM models, is as shown in the algorithm 1;

# B.1 THE CONSTRUCTION OF PATH IN FINITE STATE MACHINE

First, to formalize the agent’s decision-making process during tool invocation, we construct a Finite State Machine , the overall architecture of which is depicted in Figure 5. The states within this FSM are designed to encapsulate the critical cognitive junctures an LLM agent encounters, representing a synthesis of its internal knowledge and available external information. This design serves as a generalized abstraction of the agent’s decision pathway, ensuring high generalizability across diverse tasks.

We first have a Qwen3 series model produce a concrete answer to the query. Subsequently, to map an agent’s raw execution trajectory—the specific sequence of reasoning and tool calls—onto a canonical path within the FSM, we leverage a powerful large language model, see Table E.2 for the full prompt and the table B.1presents a specific case of Finite State Machine.

# Case 2: A example of Finite State Machine

Illustrative Decision Path. The following sequence illustrates a canonical decision path encoded within our framework:

This path represents a chain of cognitive states traversed by the agent. It begins with goal establishment, proceeds through planning and execution, encounters a knowledge gap leading to flawed reasoning, and concludes with self-diagnosis. By encoding such trajectories as nodes in the transition path layer, the graph provides a structured and abstract representation of a complex reasoning process, which can be analyzed, compared, and learned from.

# B.2 META-COGNITION CONSTRUCTION

The detailed descriptions of each type of node in the memory are as follows:

• Query Layer $\mathcal { Q }$ : Each node $q _ { i } \in \mathcal { Q }$ represents a specific task instance, such as a userissued query. It encapsulates the entirety of an interaction, including the initial input, the agent’s generated output, the complete execution trajectory, and a resultant outcome label (e.g., success or failure).   
• Transition Path Layer $\tau$ : Each node $t _ { j } \in \tau$ corresponds to a standardized decisionmaking pathway. These pathways are grounded in a predefined finite state machine (FSM) $s$ , representing a canonical sequence of the agent’s states and actions. This layer abstracts away instance-specific details to reveal underlying behavioral patterns.   
• Meta-Cognition Layer $\mathcal { M }$ : Each node $m _ { k } \in \mathcal { M }$ encodes a high-level, human-readable strategic principle. These principles are distilled from a comparative analysis of successful and failed transition paths, representing generalized heuristics for effective problemsolving.

The induction of meta-cognitions is accomplished through three primary analytical scenarios, each facilitated by a dedicated prompt:

Algorithm 1 Hierarchical Memory Graph Construction and Update   
1: Input:Memory Graph $\mathcal{G}$ ,new query $q_{i}$ ,policy $\pi$ ,FSM $S$ ,sample count $N$ ,similarity threshold $K$ 2: Ensure:Updated Memory Graph $\mathcal{G}^{\prime}$ 3: procedure UPDATEMEMORYGRAPH( $\mathcal{G},q_i,\pi ,\mathcal{S},N,K)$ 4: $T_{s}\gets \emptyset ,T_{f}\gets \emptyset$ ▷ Initialize sets for successful and failed paths   
5: $\mathcal{G}\leftarrow$ AddNode $(\mathcal{G},q_{i})$ ▷Add current query to the graph   
6: for $n = 1$ to $N$ do ▷Sample N trajectories from the policy   
7: $\tau_{n}\gets$ SampleRollout $(\pi ,q_{i})$ 8: $t_n\gets$ GroundTrajectoryToPath $(\tau_{n},S)$ ▷Map trajectory to a canonical FSM path   
9: $\mathcal{G}\gets$ AddNode $(\mathcal{G},t_n)$ 10: $\mathcal{G}\gets$ AddEdge $(\mathcal{G},q_i,t_n)$ 11: if IsSuccess $(\tau_{n})$ then   
12: $T_{s}\gets T_{s}\cup \{t_{n}\}$ 13: else   
14: $T_{f}\gets T_{f}\cup \{t_{n}\}$ 15: end if   
16: end for   
17: $M_{new}\gets$ InduceMetaCognition $(q_i,T_s,T_f,\mathcal{G},K)$ ▷ Derive new strategic principles   
18: for each new meta-cognition m in Mnew do   
19: $m_{exist}\gets$ FindMatchingMetaCognition(m, $\mathcal{G}$ 20: if $m_{exist}$ is null then   
21: $m_{final}\gets$ CreateNewMetaCognitionNode(m)   
22: $\mathcal{G}\gets$ AddNode $(\mathcal{G},m_{final})$ 23: else   
24: UpdateConfidence(mexist)   
25: $m_{final}\gets m_{exist}$ 26: end if   
27: for each path t that generated m do ▷Link paths to the principles they support   
28: $\mathcal{G}\gets$ AddEdge $(\mathcal{G},t,m_{final})$ 29: end for   
30: end for   
31: return G   
32: end procedure   
33: procedure INDUCEMETACOGNITION(qi,Ts,Tf,G,K)   
34: if $T_{s}\neq \emptyset$ and $T_{f}\neq \emptyset$ then ▷Case 1: High-confidence induction   
35: $t_s\gets$ SelectOne(Ts), $t_f\gets$ SelectOne(Tf)   
36: m- ContrastPaths(ts,tf) ▷e.g., find first diverging decision   
37: return {m}   
38: else if $T_{s} = \emptyset$ and $T_{f}\neq \emptyset$ then ▷Case 2: Speculative induction   
39: $M_{spec}\gets \emptyset$ 40: $Q_{sim}\gets$ FindSimilarQueries(qi,G,K) ▷Based on embedding similarity   
41: for each similar query qj in Qsim do   
42: for each successful path tj of qj do   
43: $M_{spec}\gets M_{spec}\cup GetMetaCognitionsFromPath(t_j,\mathcal{G})$ 44: end for   
45: end for   
46: return $M_{spec}$ 47: else   
48: return 0 ▷No new insights if only successes or no rollouts   
49: end if   
50: end procedure

![](images/91e6b7c15062117617be8f950135c280ff167386cb09cb8f9d3ccd3e9fdbb5d5.jpg)  
Figure 5: Finite State Machine

Intra-Query Analysis: This involves comparing successful and failed trajectories that originate from the identical query to distill a high-confidence causal principle. The prompt for this process is presented in E.3.

Inter-Query Analysis: This contrasts a failed trajectory with successful ones from semantically similar but distinct queries to generate speculative heuristics.

Positive Example Distillation: This process extracts generalizable strategies exclusively from a collection of successful execution paths.

# C EXPERIMENT ANALYSIS

# C.1 THE NUMBER OF THE META-COGNITION

To better understand how the quantity of retrieved meta-cognitive strategies affects agent performance, we evaluate four configurations: using 0 (no memory, denoted as ITR), 1, 3, and 5 strategies as contextual input. Results across seven QA benchmarks are presented in Table 3.

We observe that introducing even a single meta-cognitive strategy leads to a notable improvement over the baseline (ITR), especially on multi-hop tasks such as Bamboogle $( + 1 1 . 6 \% )$ and HotpotQA $( + 2 . 9 \% )$ . This suggests that explicit strategic signals can substantially aid reasoning even in limited quantities. As the number of strategies increases, performance generally improves, but the marginal gains become smaller—likely due to redundancy or prompt saturation. The best overall result is achieved at $\tt t o p \_ k = 5$ , which balances diversity and relevance.

These findings imply that a moderate number of well-curated strategies can enhance generalization and decision quality, without incurring the risks of prompt overload or noise from irrelevant memories.

Table 3: Performance of different numbers of meta-cognition.   

<table><tr><td rowspan="2">Methods</td><td colspan="3">General QA</td><td colspan="4">Multi-Hop QA</td><td rowspan="2">Avg.</td></tr><tr><td>NQ*</td><td>TriviaQA*</td><td>PopQA*</td><td>HotpotQA†</td><td>2wiki*</td><td>Musique*</td><td>Bamboogle*</td></tr><tr><td colspan="9">Qwen3-4B</td></tr><tr><td>ITR</td><td>0.298</td><td>0.581</td><td>0.157</td><td>0.268</td><td>0.281</td><td>0.077</td><td>0.290</td><td>0.279</td></tr><tr><td>topk=1</td><td>0.326</td><td>0.583</td><td>0.382</td><td>0.290</td><td>0.327</td><td>0.096</td><td>0.406</td><td>0.344</td></tr><tr><td>topk=3</td><td>0.335</td><td>0.596</td><td>0.393</td><td>0.299</td><td>0.347</td><td>0.099</td><td>0.391</td><td>0.351</td></tr><tr><td>topk=5</td><td>0.333</td><td>0.594</td><td>0.392</td><td>0.299</td><td>0.349</td><td>0.094</td><td>0.418</td><td>0.355</td></tr></table>

# C.2 CROSS-API MEMORY ROBUSTNESS

To further validate the portability and reliability of our structured memory graph, we construct the memory using two distinct LLM APIs: $9 9 ^ { \mathrm { t } - 4 0 }$ and Gemini-2.5-pro. These memory graphs are then integrated into the same downstream agent architecture (Qwen3-4B and Qwen3-8B), and evaluated across seven QA datasets. As shown in Table 4, the resulting performance differences are minor, with Gemini-based memory slightly outperforming its 4o counterpart in most cases.

Specifically, on the multi-hop benchmark Bamboogle, the Gemini-constructed memory shows a notable increase (e.g., $+ 0 . 0 4 3$ on Qwen3-8B), while maintaining parity or marginal gains in general QA datasets like TriviaQA and PopQA. These results indicate that while different APIs may introduce slight variations in strategy abstraction, our framework is robust to such differences and maintains high effectiveness regardless of the underlying model used to generate the memory.

# D CASE STUDIES

This case Dillustrates how the integration of meta-cognitive strategies enhances factual precision. Without meta-cognition, the agent returns a partially correct but under-specified answer (”National Security Law”). With meta-cognition, the agent engages in structured validation and corrects the response to the fully grounded and jurisdiction-specific ”Macau National Security Law”, aligning

Algorithm 2 Trainable Graph Weight Optimization via Policy Gradient   
1: Input: Memory Graph $\mathcal{G}$ with initial weights w, Training Queries $\mathcal{D}$ , Agent model, Reward function $\mathcal{R}$ , learning rate $\alpha$ .  
2: Output: Optimized Memory Graph $\mathcal{G}$ with updated weights $\mathbf{w}^*$ .  
3: procedure OPTIMIZEGRAPHWEIGHTS( $\mathcal{G}$ , $\mathcal{D}$ , $\alpha$ )  
4: for each query $q_{\mathrm{new}}$ in $\mathcal{D}$ do  
5: ▷— Step 1: Stochastic Guidance Selection —  
6: $m_k, p(m_k \mid q_{\mathrm{new}}) \leftarrow$ SelectGuidingMetaCognition( $\mathcal{G}$ , $q_{\mathrm{new}}$ )  
7: if $m_k$ is null then ▷ No relevant guidance found  
8: continue  
9: end if  
10: ▷— Step 2: Counterfactual Evaluation —  
11: Responsewith $\leftarrow$ Agent_generate $(q_{\mathrm{new}}, \mathrm{guidance} = m_k)$ 12: $R_{\mathrm{with}} \leftarrow \mathcal{R}(\text{Response}_{\mathrm{with}}, q_{\mathrm{new}})$ 13: Responsew/o $\leftarrow$ Agent_generate $(q_{\mathrm{new}}, \mathrm{guidance} = \mathrm{null})$ 14: $R_{\mathrm{w/o}} \leftarrow \mathcal{R}(\text{Response}_{\mathrm{w/o}}, q_{\mathrm{new}})$ 15: $\Delta R_k \leftarrow R_{\mathrm{with}} - R_{\mathrm{w/o}}$ ▷ Calculate reward gap (utility signal)  
16: ▷— Step 3: Policy Gradient Update —  
17: $\nabla_{\mathbf{w}} \mathcal{L} \leftarrow -\Delta R_k \cdot \nabla_{\mathbf{w}} \log p(m_k \mid q_{\mathrm{new}})$ ▷ Compute gradient for REINFORCE  
18: $\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \nabla_{\mathbf{w}} \mathcal{L}$ ▷ Update all contributing weights  
19: end for  
20: return $\mathcal{G}$ 21: end procedure  
22: procedure SELECTGUIDINGMETACOGNITION( $\mathcal{G}$ , $q_{\mathrm{new}}$ )  
23: ▷ Activate relevant subgraph based on semantic similarity  
24: $\mathcal{M}_{\mathrm{act}} \leftarrow$ ActivateSubgraph $(q_{\mathrm{new}}, \mathcal{G})$ 25: if $\mathcal{M}_{\mathrm{act}}$ is empty then  
26: return null, 0  
27: end if  
28: ▷ Compute relevance scores for all activated meta-cognitions  
29: for all $m \in \mathcal{M}_{\mathrm{act}}$ do  
30: $S(m \mid q_{\mathrm{new}}) \leftarrow 0$ 31: for all path $q_i \rightarrow t_j \rightarrow m$ in $\mathcal{G}$ do  
32: if $q_i$ is in activated subgraph then  
33: $S(m \mid q_{\mathrm{new}}) \leftarrow S(m \mid q_{\mathrm{new}}) + \operatorname{Sim}(q_{\mathrm{new}}, q_i) \cdot w_{qt}^{(i,j)} \cdot w_{tm}^{(j,m)}$ 34: end if  
35: end for  
36: end for  
37: ▷ Compute selection probabilities using softmax  
38: $Z \leftarrow \sum_{m' \in \mathcal{M}_{\mathrm{act}}} \exp(S(m' \mid q_{\mathrm{new}}))$ 39: for all $m \in \mathcal{M}_{\mathrm{act}}$ do  
40: $p(m \mid q_{\mathrm{new}}) \leftarrow \exp(S(m \mid q_{\mathrm{new}})) / Z$ 41: end for  
42: ▷ Stochastically sample a meta-cognition based on probabilities  
43: $m_k \leftarrow$ Sample $(\mathcal{M}_{\mathrm{act}}, \text{probabilities} = \{p(m \mid q_{\mathrm{new}})\})$ 44: return $m_k, p(m_k \mid q_{\mathrm{new}})$ 45: end procedure

Table 4: Performance comparison across LLM .   

<table><tr><td rowspan="2">Methods</td><td colspan="3">General QA</td><td colspan="4">Multi-Hop QA</td><td rowspan="2">Avg.</td></tr><tr><td>NQ*</td><td>TriviaQA*</td><td>PopQA*</td><td>HotpotQA†</td><td>2wiki*</td><td>Musique*</td><td>Bamboogle*</td></tr><tr><td colspan="9">Qwen3-8B</td></tr><tr><td>Ours(4o)</td><td>0.316</td><td>0.622</td><td>0.382</td><td>0.358</td><td>0.354</td><td>0.128</td><td>0.392</td><td>0.365</td></tr><tr><td>Ours(gemini)</td><td>0.318</td><td>0.621</td><td>0.385</td><td>0.362</td><td>0.336</td><td>0.123</td><td>0.434</td><td>0.369</td></tr><tr><td colspan="9">Qwen3-4B</td></tr><tr><td>Ours(4o)</td><td>0.335</td><td>0.596</td><td>0.393</td><td>0.299</td><td>0.347</td><td>0.099</td><td>0.391</td><td>0.351</td></tr><tr><td>Ours(gemini)</td><td>0.337</td><td>0.598</td><td>0.396</td><td>0.314</td><td>0.360</td><td>0.093</td><td>0.397</td><td>0.357</td></tr></table>

exactly with the ground truth. This aligns with our design goal: equipping agents with self-checking mechanisms that reduce synthesis ambiguity, particularly when internal confidence is high but error risk remains.

# Case 3: Correcting Answer via Meta-Cognition

Query: Hong Kong Macau cultural exchange was a trip that tested which law whose purpose was to fulfil Article 23 of the Macau Basic Law?

Ground Truth: Macau National Security Law

Meta-Cognition: Early recognition and affirmation of the ’KnowledgeSufficient’ state can help prevent synthesis inaccuracies in scenarios where internal confidence is prone to challenges. Structured validation strategies ensure precise answer formulation when internal knowledge seems sufficient but risks synthesis errors.

Response (w/o meta-cognition):

... Therefore, the answer should state that the law tested was the National Security Law, aimed at fulfilling Article 23. Final Answer: National Security Law

Response (with meta-cognition):

... That law is likely the Macau National Security Law, which was enacted in 2009... The answer should be the Macau National Security Law. Final Answer: Macau National Security Law

# E PROMPT TEMPLATES

# E.1 MULTI-TURN TOOL-INTERGRATED QA PROMPT

When LLMs needs to interact with tools multiple times to answer a question, it is necessary to guide the LLM on which tools to use and how to use them. The specific prompt is as follows.

# E.2 CONSTRUCTING FSM PATH

Given the specific response path of the LLM and the complete structure of the state machine, we employ an LLM (e.g., GPT-4o) to map the generated answer onto one of the predefined paths in the state machine. The following shows the exact prompt used.

# E.3 META-COGNITION CONSTRUCTING

With both successful and failed state-machine paths available, we derive high-level meta-cognitions by contrasting the two. The following prompt illustrates how a pair of successful and failed paths under the same query is used to induce meta-cognition.

Prompt A: System and User Prompt   
SYSTEM PROMPT:   
```markdown
# Tools   
You may call one or more functions to assist with the user query.   
You are provided with function signatures within <tools></tools>XML tags:   
<tools>   
{ "name": "search-query_rag", "description": "MCP RAG Query Tool (Synchronous Version) Arguments: query: query text topk: The default number of documents returned is 3 Returns: str: The formatted query result", "parameters": { "type": "object", "properties": { "query": {"title": "Query", "type": "string"}, "topk": {"default": 3, "title": "Topk", "type": "integer"} }, "required": ["query"] }   
}   
</tools>   
# Tool call format   
For each function call, return a JSON object with function name and arguments within <tool_call></tool_call> XML tags: <tool_call> { "name": <function-name>, "arguments": <args-jscript-object>   
}   
</tool_call> 
```

# USER PROMPT:

```txt
Answer the given question. After reasoning, if you find you lack some knowledge, you can call the search tool.   
You may search as many times as you want.   
If you find no further external knowledge is needed, you can directly provide the answer inside <answer> and</answer>, without detailed illustrations.   
For example: <answer> Beijing </answer>. 
```

```txt
Question: Which US State, historically dominated by the Republican party from 1860 to 1932, is represented by State Senator George D. Maziarz? 
```

# Prompt B: Prompt for constructing FSM path

Instruction: You are a metacognition analysis expert specialized in extracting generalized decision principles and guidance strategies from state machine execution paths.

State machine transition rules: {transitions info}

# Core Requirements:

1. Generalizability Focus: Output strategies and principles must be general, applicable to similar problems, without specific query details.   
2. Direct Usability: Generated content should be directly usable as guidance principles for new problems.   
3. Principled Expression: Use cautious guidance terms like “consider”, “may help”, “tends to” rather than definitive statements.   
4. Concise Effectiveness: Output only the most core insights, avoid redundancy and complexity.   
5. Quality Control: Strictly evaluate whether there is sufficient evidence to support new metacognition.   
6. Knowledge Confidence Awareness: Recognize that LLM’s internal knowledge confidence varies across queries — success patterns may be domain-specific.   
7. Uncertainty Acknowledgment: Express appropriate uncertainty in guidance principles, avoiding overly definitive conclusions.   
8. Quantity Management: When metacognition count exceeds 30, prioritize updating low-confidence existing metacognitions.

# Output Format (Quantity-Aware):

Your output must be a JSON object with the following structure:

```txt
{ "decision": "update" or "create" or "skip", "target_meta_id": <ID of metacognition to update (only when decision is "update")>, "reasoning": "Brief explanation including quantity management when count > 30.", "meta_cognition": { "summary": "Concise general guidance summary (use cautious language).", "strategy_principles": [ {"principle": "...", "confidence": "high" | "medium" | "low", "confidence_score": 30 - 85 }, ], "overall_confidence": "high" | "medium" | "low", "evidence_paths": <int>, "uncertainty-note": "Brief acknowledgment of limitations or knowledge-dependency concerns." } } 
```

# Prompt C: Metacognition Prompt Specification

# State machine transition rules: {transitions info}

# Core Requirements:

1. Generalizability Focus: Output strategies and principles must be general, applicable to similar problems, without specific query details.   
2. Direct Usability: Generated content should be directly usable as guidance principles for new problems.   
3. Principled Expression: Use cautious guidance terms like “consider”, “may help”, “tends to” rather than definitive statements.   
4. Concise Effectiveness: Output only the most core insights, avoid redundancy and complexity.   
5. Quality Control: Strictly evaluate whether there is sufficient evidence to support new metacognition.   
6. Knowledge Confidence Awareness: Recognize that LLM’s internal knowledge confidence varies across queries—success patterns may be domain-specific.   
7. Uncertainty Acknowledgment: Express appropriate uncertainty in guidance principles, avoiding overly definitive conclusions.   
8. Quantity Management: When metacognition count exceeds 30, prioritize updating low-confidence existing metacognitions.

# Critical Self-Reflection Requirements:

• Pattern Validity: Question whether identified patterns truly represent generalizable principles.   
• Knowledge Dependency: Consider if success stems from strategy effectiveness or the LLM’s domain familiarity.   
• Evidence Sufficiency: Demand higher evidence standards for strategies that could mislead future queries.   
• Simplicity Over Complexity: Favor simple, robust principles over complex, brittle ones.

# Metacognition Quantity Control Strategy:

When metacognition count $\leq 3 0$ : • Normal decision making: create, update, or skip based on evidence quality.

• Prefer creating new metacognition when patterns are sufficiently distinct.   
• Express appropriate uncertainty in new metacognitions.

When metacognition count $> 3 0$ : • Strongly prefer UPDATE over CREATE: Prioritize improving existing low-confidence metacognitions.

• Only create new metacognition if the pattern is exceptionally valuable and completely distinct.   
• Target metacognitions with confidence levels “low” or “medium” for updates.

# Analysis Focus:

1. Success Pattern Identification: Abstract reusable decision patterns from successful paths.   
2. Failure Cause Summary: Identify generalizable errors to avoid from failed paths.   
3. State Transition Optimization: Extract best practice principles for state machine execution.   
4. Knowledge Dependency Assessment: Evaluate whether patterns might be specific to certain knowledge domains.

5. Existing Knowledge Enhancement: When quantity is high, focus on strengthening weak metacognitions.

# Decision Options:

• create: Create new metacognition (when discovering valuable and distinct patterns, or when quantity $\leq 3 0$ ).   
• update: Update existing metacognition (preferred when quantity $> 3 0$ , especially targeting low-confidence ones).   
• skip: Skip metacognition operation (when evidence is insufficient or has no new value).

# Skip Metacognition Situations:

• Path data quality is poor, patterns are unclear.   
• Existing metacognition already covers the pattern, new evidence shows no significant improvement.   
• Success/failure path differences are not obvious, difficult to extract effective strategies.   
• Cannot distinguish whether success stems from strategy effectiveness or knowledge domain familiarity.   
• When quantity $> 3 0$ and no suitable low-confidence metacognition found for update.

# Output Format (Quantity-Aware): Your output must be a JSON object containing:

```json
{
    "decision": "update" or "create" or "skip",
    "target_meta_id": (when decision is update) ID of metacognition to update,
    "reasoning": "Brief decision analysis, must include quantity management when count > 30.,
    "meta_cognition": {
        "summary": "...",
        "strategy_principles": [
            {"principle": "...", "confidence": "high", "confidence_score": 80},
            {"principle": "...", "confidence": "medium", "confidence_score": 60}
        ],
        "overall_confidence": "medium",
        "evidence_paths": 7,
        "uncertainty-note": "..."
    }
} 
```
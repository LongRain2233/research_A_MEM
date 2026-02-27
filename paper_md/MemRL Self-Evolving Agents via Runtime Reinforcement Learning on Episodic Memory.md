# MemRL: Self-Evolving Agents via Runtime Reinforcement Learning on Episodic Memory

Shengtao Zhang 1 * Jiaqian Wang 2 * Ruiwen Zhou 3 Junwei Liao 1 4 Yuchen Feng 5 Zhuo Li 1 Yujie Zheng 1 Weinan Zhang 1 4 Ying Wen 1 4 Zhiyu Li 5 Feiyu Xiong 5 Yutao Qi 2 Bo Tang 6 5 Muning Wen 1

# Abstract

The hallmark of human intelligence is the selfevolving ability to master new skills by learning from past experiences. However, current AI agents struggle to emulate this self-evolution: finetuning is computationally expensive and prone to catastrophic forgetting, while existing memorybased methods rely on passive semantic matching that often retrieves noise. To address these challenges, we propose MEMRL, a non-parametric approach that evolves via reinforcement learning on episodic memory. By decoupling stable reasoning from plastic memory, MEMRL employs a Two-Phase Retrieval mechanism to filter noise and identify high-utility strategies through environmental feedback. Extensive experiments on HLE, BigCodeBench, ALFWorld, and Lifelong Agent Bench demonstrate that MEMRL significantly outperforms state-of-the-art baselines, confirming that MEMRL effectively reconciles the stability-plasticity dilemma, enabling continuous runtime improvement without weight updates. Code is available at https://github.com/ MemTensor/MemRL.

# 1. Introduction

Human intelligence balances cognitive stability and episodic plasticity (Grossberg, 2013; McClelland et al., 1995; Kumaran et al., 2016) via Constructive Episodic Simulation, enabling adaptation without rewiring neural circuitry (Schacter & Addis, 2007; Hassabis & Maguire, 2007; Schacter et al., 2012; Gick & Holyoak, 1980). Despite their rea-

*Equal contribution 1Shanghai Jiao Tong University, Shanghai, China 2Xidian University, Xi’an, China 3National University of Singapore, Singapore 4Shanghai Innovation Institute, Shanghai, China 5MemTensor (Shanghai) Technology Co., Ltd., Shanghai, China 6University of Science and Technology of China, Hefei, China. Correspondence to: Bo Tang <tangb@memtensor.cn>, Muning Wen <muningwen@sjtu.edu.cn>.

Preprint. February 13, 2026.

soning capabilities, current AI agents struggle to emulate this decoupled self-evolution (Wei et al., 2022; Yao et al., 2023; Schick et al., 2023; Wang et al., 2023). Specifically, fine-tuning internalizes experience by modifying weights (Ouyang et al., 2022; Stiennon et al., 2020; Rafailov et al., 2023; Ethayarajh et al., 2024) but suffers from computational costs and catastrophic forgetting (Kirkpatrick et al., 2017; Li et al., 2024; Wu et al., 2024). Conversely, Retrieval-Augmented Generation (RAG) (Lewis et al., 2020) provides a non-parametric alternative but remains passive, retrieving by semantic similarity rather than utility (Karpukhin et al., 2020; Gao et al., 2024); this prevents agents from effectively leveraging runtime feedback to distinguish high-value strategies from noise.

![](images/5247094b3afda8f657f921f4ee5392b5e30a67355ac8c3f4df8a588d30ba127b.jpg)  
Figure 1. The conceptual framework of MemRL.

This limitation underscores a critical research question: How can we enable an agent to continuously improve its performance after deployment, without compromising the stability of its pre-trained backbone? Our objective is to achieve an agent that evolves with continued usage and rapidly adapts to new tasks after deployment, referred to as Runtime Continuous Learning (Javed et al., 2023; Silver & Sutton, 2025; Parisi et al., 2019; Wu et al., 2024), all while keeping the backbone model frozen to prevent catastrophic forgetting (Finn et al., 2017; Wei et al., 2025). To address this challenge, inspired by the human cognitive mechanism of constructive simulation, we propose MEMRL, an approach that facilitates self-evolving agents by explicitly decoupling the model’s stable cognitive reason-

ing from dynamic episodic memory. Figure 1 illustrates the conceptual framework of our proposed MEMRL. Drawing on tries-and-errors manner in Reinforcement Learning (RL) to estimate expected experience utilities (Sutton & Barto, 2018), we formalize the interaction between the frozen LLM and external memory as a Markov Decision Process (MDP) (Puterman, 2014). Unlike traditional methods that optimize the backbone model, MEMRL optimizes the policy of memory usage without tuning model weights.

MEMRL organizes memory into a structured Intent-Experience-Utility triplet. This structure transforms retrieval from a passive semantic match task into an active decision-making process: Two-Phase Retrieval selects experiences based on their learned Q-values, reflecting expected utility, rather than semantic similarity alone (Watkins & Dayan, 1992); Utility-Driven Update refines these Qvalues through environmental feedback, applying Monte Carlo style updates (Metropolis & Ulam, 1949). This closedloop cycle enables the agent to distinguish high-value memories from similar noise, effectively learning from both success and failure without high computational cost or catastrophic forgetting risks associated with weight updates. As for experiments, we validate MEMRL on four diverse benchmarks, including HLE, BigCodeBench, ALFWorld, and Lifelong Agent Bench. Our results demonstrate consistent superiority over baselines, achieving relative improvement in exploration-heavy environments. Our in-depth analysis reveals a strong correlation between learned utility and task success, further confirming MEMRL’s effectiveness.

In summary, our contributions are threefold:

• We propose a runtime learning framework using Model-Memory decoupling and Intent-Experience-Utility triplet to reconcile the stability-plasticity dilemma, enabling tuning-free agent learning.   
• We introduce MEMRL, a non-parametric approach enabling agent self-evolution via Two-Phase Retrieval and Utility-Driven Update.   
• We conduct extensive evaluations and provide a rigorous analysis for MEMRL’s stability, showing how it ensures task integrity and minimizes forgetting.

# 2. Related Works

Runtime Learning Runtime Learning focuses on the post-deployment improvement of agents through interaction streams rather than offline data, marking a shift toward the “era of experience” (Silver & Sutton, 2025). Unlike Continual Learning (Parisi et al., 2019; Wu et al., 2024) or Test-Time Adaptation (Sun et al., 2020; Wang et al., 2020; Liang et al., 2025), which typically update parameters to handle forgetting or distribution shifts, our setting constrains

the backbone to remain frozen to ensure stability and efficiency. While recent memory-augmented agents (Zheng et al., 2025; Wei et al., 2025; Zhou et al., 2025b) emphasize memory organization, the selection problem—identifying which experiences to reuse under feedback—remains a critical challenge. Drawing from value-aware episodic control (Tulving et al., 1972; McClelland et al., 1995), we frame runtime learning as identifying valuable episodes. By using interaction feedback to assign utility, our approach guides retrieval and reuse without weight modification, thereby ensuring sustained improvement (Pritzel et al., 2017).

Reinforcement Learning Reinforcement learning has been widely adopted for LLMs enhancement. A representative paradigm is to construct reward signals from human feedback and optimize the model policy accordingly to align with human preference (Stiennon et al., 2020; Ouyang et al., 2022). Other recent approaches leverage rule-based verifiers to improve LLMs’ reasoning capabilities (Guo et al., 2025; Yu et al., 2025). In parallel, agent-oriented research explores how interaction signals can improve tool use and action decision-making, and investigates mechanisms by which language models execute composite actions in environments (Schick et al., 2023). Despite the demonstrated effectiveness of reward-driven optimization, these methods generally place learning in the model parameters or additional parametric modules, and thus do not avoid the cost of online updates or the risk of forgetting. In contrast, our method frames memory usage as a learnable decision problem and applies non-parametric reinforcement learning on memory to bypass the risk.

Agentic Memory To avoid the costs of fine-tuning, external memory systems have evolved from a static RAG paradigm to dynamic, governable memory structures (Lewis et al., 2020; Karpukhin et al., 2020). Early agentic memory introduced reflection mechanisms and hierarchical management to handle long context experiences (Shinn et al., 2023; Packer et al., 2024). More recent frameworks have systematized the memory lifecycle, focusing on unified storage and structured indexing for complex tasks (Li et al., 2025b; Xu et al., 2025; Huang et al., 2025; Ye, 2025). Furthermore, adaptive approaches now explore improving retrieval via feedback-driven updates or automated augmentation (Salama et al., 2025; Zhang et al., 2025; Li et al., 2025a; Zhou et al., 2025a). However, except for training additional learnable modules, most existing methods still rely predominantly on semantic similarity or heuristic rules, lacking a rigorous metric to evaluate the actual utility of a memory in maximizing returns. Inspired by cognitive theories of memory reconsolidation (Schacter & Addis, 2007; Gick & Holyoak, 1980; Nader et al., 2000), MEMRL bridges this gap by formulating retrieval as a value-based decision process, learning robust utility estimates (Q-values) from en-

![](images/164bf8b49132b39614461139d6690767465f53d3901e2baddee381b3adacf61f.jpg)  
Figure 2. An illustrative example of memory-augmented decision making under a Markov Decision Process. At time step $t$ , the agent starts with an initial memory set $\mathcal { M } _ { t }$ . At time step $t { + } 1$ , an intent (Intent A) retrieves relevant past experiences, but initially leads to a failed generation. In contrast, another intent (Intent B) succeeds, and its associated experience is added to memory. At time step $t + 2$ , Intent A retrieves the newly stored successful experience from Intent $B$ , resulting in a successful outcome. This example shows how memory retrieval enables knowledge reuse across intents, implicitly supporting self-evolution through shared experiences.

vironmental rewards to distinguish high-value experiences.

# 3. Problem Formulation

In this section, we formally define the problem of memoryaugmented generation and establish the theoretical link between agent policy and memory retrieval. We adopt the formulation of Memory-Based Markov Decision Process (M-MDP) (Zhou et al., 2025a), and apply our non-parametric reinforcement learning approach to it. Figure 2 provides an illustrative example of this memory-augmented decision process, showing how retrieval outcomes and memory evolution unfold over multiple time steps.

# 3.1. Memory-Augmented Agent Policy

To enable the agent to self-evolution, we adopt the M-MDP framework (Zhou et al., 2025a), defined by the tuple $( \mathcal { S } , \mathcal { A } , P , \mathcal { R } , \gamma , \mathcal { M } )$ . Here, $s$ and $\mathcal { A }$ represent state and action spaces, $P$ is the transition dynamics, $\mathcal { R }$ is the reward function of state and action, $\gamma \in [ 0 , 1 )$ denotes the discount factor, and $\mathcal { M } = ( \mathcal { S } \times \mathcal { A } \times \mathbb { R } ) ^ { * }$ constitutes the evolving memory bank of past experiences (Zhou et al., 2025a). At each step $t$ , the agent receives state $s _ { t }$ and leverages $\mathcal { M } _ { t }$ to generate a response $a _ { t }$ maximizing the expected reward. The joint policy $\pi ( a _ { t } | s _ { t } , \mathcal { M } _ { t } )$ is formulated as the marginal probability over all possible retrieved items $m$ (Zhou et al., 2025a):

$$
\pi \left(a _ {t} \mid s _ {t}, \mathcal {M} _ {t}\right) = \sum_ {m \in \mathcal {M} _ {t}} \mu \left(m \mid s _ {t}, \mathcal {M} _ {t}\right) p _ {L L M} \left(a _ {t} \mid s _ {t}, m\right). \tag {1}
$$

where $\mu ( m | s _ { t } , \mathcal { M } _ { t } )$ is the Retrieval Policy for selecting memory contexts, and $p _ { L L M } ( a _ { t } | s _ { t } , m )$ is the Inference Policy parameterized by a frozen LLM. This approach transforms retrieval from a passive match into an active decision process, effectively accounting for the functional utility of $m$ in generating successful outcomes $a _ { t }$ .

In previous RAG or memory-based agentic paradigms, the retrieval policy $\mu$ is usually determined by a fixed vector similarity metric, e.g., cosine similarity of embeddings. While effective for semantic matching, such policies fail to account for the utility of a memory, i.e., whether retrieving $m$ actually leads to a successful outcome $a _ { t }$ .

# 3.2. Non-Parametric Reinforcement Learning

To overcome static similarity limitations, we operationalize the M-MDP framework by formulating memory retrieval as a value-based decision-making process (Zhou et al., 2025a). Unlike parametric methods optimizing $\pi _ { L L M }$ via weight updates, we optimize the retrieval policy $\mu ( m | s , \mathcal { M } )$ directly within the memory space by mapping M-MDP components to a structured Intent-Experience-Utility triplet:

From Semantic Matching to Decision Making. We instantiate state s as the User Intent, encapsulated by the current query embedding (Lewis et al., 2020). Consequently, the action space $\boldsymbol { A } _ { t }$ becomes dynamic and discrete, corresponding to selecting a specific $m$ from the memory bank $\mathcal { M } _ { t }$ (Zhou et al., 2025a). In this formulation, retrieval is not a passive matching task but a strategic decision step taken to augment the generator’s action $a$ (Zhou et al., 2025a).

Defining Utility via Q-Values. While the agent’s executable action $a$ is generated by the policy $\pi$ (as formalized in Sec. 3.1), the quality of this generation is strictly conditioned on the retrieved context. Therefore, we adapt the traditional value function $Q ( s , a )$ to the retrieval phase, defining $Q ( s , m )$ as the expected utility of the subsequent action $a$ augmented by memory $m$ . MEMRL learns an optimal retrieval policy $\mu ^ { * }$ that maximizes this utility:

$$
\mu^ {*} (m | s, \mathcal {M}) = \arg \max  _ {m \in \mathcal {M}} Q (s, m). \tag {2}
$$

In this view, the Q-value acts as a critic for the retrieval mechanism, distinguishing memories that strategically aid

![](images/a5d24d691502a295b976ee6633679b9552f06c9d16433217bf68c0061d70596d.jpg)

![](images/7c63da619a5633d9edb45b94fd520d89d4eed3c3df900cb2e435ab02698f974c.jpg)

![](images/50ffd823d6e2ae56cfac0ace5cbaa1eacf12e8fbb9318157b7a9a08c4ccaaea9.jpg)  
Figure 3. Overview of MEMRL. The end-to-end learning loop(Top): given a query s, the agent retrieves context $\mathbf { m } _ { c t x }$ from memory M, generates output y, and updates memory value $Q$ based on reward $R$ . Two-Phase Retrieval(Bottom Left): Candidates are recalled via similarity, then re-ranked using learned Q-values. Utility Update(Bottom Right): Values $( Q )$ are updated using environmental rewards to distinguish functional utility from semantic similarity.

the generator from irrelevant noise that merely shares high semantic similarity.

Non-Parametric Learning. Since the retrieval action space is decoupled from LLM generation, we perform learning without modifying model weights. Upon receiving environmental feedback r, we update the Q-value via a Temporal-Difference (TD) error (Sutton, 1988):

$$
Q (s, m) \leftarrow Q (s, m) + \alpha [ r + \gamma \max  Q \left(s ^ {\prime}, m ^ {\prime}\right) - Q (s, m) ], \tag {3}
$$

or Monte Carlo style rule (Metropolis & Ulam, 1949):

$$
Q _ {\text {n e w}} \leftarrow Q _ {\text {o l d}} + \alpha (r - Q _ {\text {o l d}}), \tag {4}
$$

where $\alpha$ is the learning rate. Equation 4 performs as a naturally simplified version of Equation 3 by setting $s ^ { \prime }$ as a terminal state to balance complexity and performance, sharing a similar one-step MDP formulation with Guo et al. (2025). These manners allow utility estimates to converge to true expected returns over time (Bellman, 1966). By explicitly updating Q-values within the memory structure, MEMRL provides a non-parametric learning manner with a theoretical guarantee, enabling agents to self-evolve through interaction.

# 4. MEMRL

Building upon the M-MDP formulation defined in Section 3, we propose MEMRL, a framework that enables frozen LLMs to self-evolve via non-parametric reinforcement learning. Instead of modifying the model weights $\theta$ , MEMRL optimizes the retrieval policy $\mu ( m | s , \mathcal { M } )$ within an evolving memory space. As illustrated in Figure 3, the framework consists of three core components: (i) a structured Intent-Experience-Utility memory bank, (ii) a Two-Phase Retrieval mechanism that decouples semantic recall from value-aware selection, and (iii) a Runtime Utility Update rule that stabilizes Q-value estimation.

# 4.1. The Intent-Experience-Utility Triplet

To support value-based decision-making, we structure the external memory $\mathcal { M }$ not merely as key-value pairs, but as a set of triplets:

$$
\mathcal {M} = \left\{\left(z _ {i}, e _ {i}, Q _ {i}\right) \right\} _ {i = 1} ^ {| \mathcal {M} |}, \tag {5}
$$

where $z _ { i }$ represents the Intent, $e _ { i }$ stores the raw Experience (e.g., a successful solution trace or trajectory), and $Q _ { i }$ denotes the learned Utility. $Q _ { i }$ approximates the expected return of applying experience $e _ { i }$ to intents similar to $z _ { i }$ ,

serving as the critic in RL.

# 4.2. From Semantic Recall to Value-Aware Selection

Standard RAG systems assume “similar implies useful,” but agentic tasks often involve environment-specific routines that generalize poorly (Singh et al., 2025; Cuconasu et al., 2024; Gan et al., 2024; Zhou et al., 2024). Therefore, MEMRL implements a Two-Phase Retrieval strategy.

Phase A: Similarity-Based Recall. Given query $s$ , we isolate a candidate pool $\mathcal C ( s )$ of semantically consistent experiences by filtering memory bank $\mathcal { M }$ via cosine similarity and a sparsity threshold $\delta$ :

$$
\mathcal {C} (s) = \operatorname {T o p} \mathrm {K} _ {k _ {1}} \left(\left\{i \mid \operatorname {s i m} \left(E m b (s), E m b \left(z _ {i}\right)\right) > \delta \right\}\right), \tag {6}
$$

where Emb represents the Embedding Model to transfer the raw text to a vector. If ${ \mathcal { C } } ( s ) = \emptyset$ , MEMRL relies solely on the frozen LLM for exploration. Phase B: Value-Aware Selection. To determine the final context $\mathcal { M } _ { c t x } ( s )$ , we select top- $k _ { 2 }$ items from $\mathcal C ( s )$ using a composite score balancing exploration (similarity) and exploitation (utility $Q$ ):

$$
\operatorname {s c o r e} (s, z _ {i}, e _ {i}) = (1 - \lambda) \cdot \hat {s i m} (E m b (s), E m b (z _ {i})) + \lambda \cdot \hat {Q} _ {i}. \tag {7}
$$

where ˆ· denotes z-score normalization and $\lambda \in [ 0 , 1 ]$ modulates the trade-off. This mechanism filters out “distractor” memories—those semantically similar but with low historical utility. As detailed in Section 5.4.2, normalization and strict similarity thresholds are essential for noise filtering and maintaining stability during self-evolution.

# 4.3. Non-Parametric RL on Memory

The core of MEMRL is the continuous refinement of Qvalues based on environmental feedback, enabling the agent to “remember” what works. During runtime, MEMRL performs learning entirely in memory space. With the retrieved context $m$ , the agent samples an action $a$ according to the policy $\pi$ defined in Eq. 1. Executing $a$ then yields an environmental reward $r$ (e.g., execution success or scalar score). For the memories actually injected into the input context $\mathcal { M } _ { \mathrm { c t x } } ( s )$ , we update their utilities in triplets with a Monte Carlo style rule, i.e., the Eq. 4, following the runtime learning loop shown in Figure 3. This update drives $Q _ { \mathrm { n e w } }$ toward the empirical expected return of using experience $e _ { i }$ under similar intents. Meanwhile, for each sampled trajectory, we use an LLM to summarize the experience (Fang et al., 2025), and write it back into the memory bank as a new triplet $( z , e _ { \mathrm { n e w } } , Q _ { \mathrm { i n i t } } )$ , enabling continual expansion of experience while keeping the LLM parameters unchanged.

# 4.4. Theoretical Stability Analysis

We analyze the stability of MEMRL from a reinforcement learning perspective, with full analysis provided in Ap-

pendix A. We posit two standard assumptions: a frozen inference policy $p _ { \mathrm { L L M } }$ and a stationary task distribution. Under these conditions, the learning target $\beta ( s , m ) = \mathbb { E } [ r _ { t } | s , m ]$ is well-defined, where the expectation is taken over the stochastic rewards resulting from the Inference Policy $p _ { L L M }$ distribution. We prove that utility estimates updated via Eq. 4 are unbiased and variance-bounded (Sutton & Barto, 2018). Specifically, as $t \to \infty$ :

$$
\lim  _ {t \to \infty} \mathbb {E} [ Q _ {t} ] = \beta (s, m),
$$

$$
\lim  _ {t \rightarrow \infty} \operatorname {V a r} \left(Q _ {t}\right) \leq \frac {\alpha}{2 - \alpha} \operatorname {V a r} \left(r _ {t} \mid s, m\right). \tag {8}
$$

Furthermore, we address the challenge of the latent retrieval distribution $\operatorname* { P r } ( s | m )$ shifting during training by framing MEMRL as a Generalized Expectation-Maximization (GEM) (Dempster et al., 1977) process. The system performs coordinate ascent on a global objective: the retrieval ranking acts as the Policy Improvement (E-step), while the utility update acts as the Value Update (M-step). By the monotonic improvement theorem (Neal & Hinton, 1998), the system converges to a stationary point where the global memory utility stabilizes:

$$
\lim  _ {t \rightarrow \infty} \mathbb {E} [ Q _ {t} (m) ] = \sum_ {s \in \mathcal {S} (m)} \mathbb {E} [ r | s, m ] \Pr (s | m). \tag {9}
$$

where $s ( m )$ is the effective support set of the memory. This formulation guarantees global stability and prevents catastrophic forgetting. Details can be found in Appendix A.4.

# 5. Experiments

# 5.1. Experimental Setup

Baselines & Benchmarks. We compare MemRL against RAG-based (RAG, Self-RAG), Agentic Memory (Mem0, MemP), and Test-Time Scaling $\left( \operatorname* { P a s s } \ @ k \right)$ baselines under a frozen-backbone setting (see Appendix D.1). Evaluations span four domains: BigCodeBench (coding), ALFWorld (navigation), LifelongAgent Bench (OS/DB), and Humanity’s Last Exam(HLE). Details are in Appendix D.2. Backbones are selected per benchmark to avoid no-signal or ceiling problems, ensuring valid learning signals, while Appendix E.4 provides a unified comparison to demonstrate cross-task consistency under identical capacity.

Metrics. We employ two metrics: (1) Success Rate (SR), the ratio of tasks completed in an epoch; (2) Cumulative Success Rate (CSR), the proportion of tasks solved at least once across epochs.

We evaluate our MEMRL and baselines under two distinct settings: Runtime Learning, which assesses the ability to learn and adapt within a training session, and Transferring, which evaluates the generalization capability of the learned

![](images/80be569c6dc4078f6c15a16af8b973dc24a96f7140f5cfb0c6401a2e12bddb78.jpg)  
Figure 4. OS Interaction Performance. Performance comparison of MEMRL vs. baselines (MemP, RAG). Solid lines represent the Epoch Success Rate, while dashed lines indicate the Cumulative Success Rate (CSR). MEMRL demonstrates superior stability and a widening performance gap in CSR.

memory on unseen tasks. Implementation and reproducibility details, including all prompts used in our experiments, can be found in Appendix E and Appendix I.

# 5.2. Main Results

Runtime Learning Results. As detailed in Table 1, MEMRL demonstrates robust superiority across all domains, surpassing the strongest baseline (MemP) by an average of $+ 3 . 8 \%$ in Cumulative Success Rate (CSR). The gains are most significant in exploration-intensive environments like ALFWorld and OS tasks (both $+ 6 . 2 \%$ ), while maintaining a steady lead on the challenging HLE benchmark $( + 3 . 6 \% )$ . This confirms that our value-based mechanism, unlike MemP’s heuristic retrieval, effectively filters noise to retain high-utility procedural patterns.

Transferring Results. We evaluate memory transferability by freezing the memory bank after training and testing on held-out sets. As shown in Table 2, MEMRL exhibits superior transferability, outperforming the strongest baseline (MemP) by an average of $+ 2 . 8 \%$ in Success Rate. The advantage is particularly pronounced in complex environments like ALFWorld $( + 5 . 8 \% )$ and OS tasks $( + 2 . 6 \% )$ . These margins validate that our Two-Phase Retrieval effectively filters low-value noise, retaining high-utility procedural patterns that generalize robustly to unseen scenarios.

# 5.3. Ablations

# 5.3.1. EFFECTIVENESS OF RUNTIME RL

To isolate the efficacy of runtime RL, we compare MEMRL and its RAG-based variant against their non-RL counterparts (MemP and standard RAG) in the OS interaction environment. As shown in Figure 4, while initial performance is comparable, a clear divergence emerges as training progresses: MEMRL achieves a smoother learning curve and superior stability. Crucially, this advantage is most pronounced in the Cumulative Success Rate (dashed lines),

![](images/c5127d1c87a10af1ea54e677ba3e8b25ee0dca4489f1c42faaa932d7ddcd3038.jpg)  
Figure 5. Ablation on Q-value weighting factor $\lambda$ . Performance comparison across $\lambda \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ . Solid lines denote Epoch Success Rate, while dashed lines indicate Cumulative Success Rate (CSR). The balanced setting $\lambda = 0 . 5$ achieves the optimal trade-off between relevance and helpfulness.

where the monotonic widening gap indicates that the RLdriven value function effectively filters noisy memories and consolidates successful experiences.

# 5.3.2. IMPACT OF Q-VALUE WEIGHTING

To determine the optimal equilibrium between semantic grounding and value-based exploitation, we evaluate the Q-weighting factor $\lambda \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ . As shown in Figure 5, performance exhibits a clear concave trend peaking at $\lambda = 0 . 5$ . Deviating toward extremes degrades results: pure semantic retrieval $\lambda = 0$ ) plateaus due to an inability to filter functional distractors, while excessive RL weight ( $\lambda  1$ ) induces volatility and context detachment. This confirms that $\lambda = 0 . 5$ represents an effective balance, where semantic similarity guarantees content relevance and Q-value ensures its helpfulness.

# 5.3.3. ABLATION ANALYSIS: CROSS-TASK VS. SINGLE-TASK OPTIMIZATION

To investigate the source of our performance gains, we conduct an ablation study in Table 3 by restricting the memory retrieval scope. We compare the full MEMRL (Cross-Task Retrieval) against an ablated setting that only utilizes feedback from the single task instance, conceptually equivalent to Reflexion (Shinn et al., 2023). MEMRL demonstrates superior performance in structured environments, particularly on OS-Agent $( + 9 . 0 \% )$ and ALFWorld $( + 5 . 1 \% )$ . These benchmarks exhibit high intra-dataset similarity, allowing MEMRL to effectively perform horizontal transfer—retrieving and adapting successful policies from semantically similar historical tasks. While on the HLE benchmark, the single-task baseline (0.610) is tied with MEMRL (0.606). We attribute this to the HLE dataset’s low internal semantic similarity (0.186), as detailed in Appendix B.3. This prevents effective cross-task generalization, forcing the agent to rely solely on single feedback.

Table 1. Runtime Learning results. We compare MEMRL against various baselines over 10 epochs. The results are reported as Last Epoch Success Rate / Cumulative Success Rate (CSR). The Average column indicates the mean performance across all benchmarks.   

<table><tr><td rowspan="2">Method</td><td colspan="2">BigIntCodeBench</td><td colspan="2">Lifelong Agent Bench</td><td>ALFWorld</td><td>HLE</td><td>Average</td></tr><tr><td>Code Gen (Last / CSR)</td><td>OS Task (Last / CSR)</td><td>DB Task (Last / CSR)</td><td>Exploration (Last / CSR)</td><td>Knowledge Frontier (Last / CSR)</td><td>(Last / CSR)</td><td></td></tr><tr><td>Model</td><td>GPT-4o</td><td>GPT-4o-mini</td><td>GPT-4o-mini</td><td>GPT-5-mini</td><td>Gemini-3-pro</td><td>-</td><td></td></tr><tr><td>No Memory</td><td>0.485</td><td>0.674</td><td>0.860</td><td>0.777</td><td>0.357</td><td>0.631</td><td></td></tr><tr><td>Pass@10</td><td>-/0.577</td><td>-/0.756</td><td>-/0.928</td><td>-/0.928</td><td>-/0.524</td><td>-/0.743</td><td></td></tr><tr><td>RAG</td><td>0.475 / 0.483</td><td>0.690 / 0.700</td><td>0.914 / 0.916</td><td>0.887 / 0.930</td><td>0.430 / 0.475</td><td>0.679 / 0.699</td><td></td></tr><tr><td>Self-RAG</td><td>0.497 / 0.561</td><td>0.646 / 0.732</td><td>0.891 / 0.898</td><td>0.907 / 0.962</td><td>0.423 / 0.475</td><td>0.673 / 0.726</td><td></td></tr><tr><td>Mem0</td><td>0.487 / 0.495</td><td>0.670 / 0.702</td><td>0.920 / 0.926</td><td>0.894 / 0.969</td><td>0.436 / 0.470</td><td>0.681 / 0.712</td><td></td></tr><tr><td>MemP</td><td>0.578 / 0.602</td><td>0.736 / 0.742</td><td>0.960 / 0.966</td><td>0.885 / 0.919</td><td>0.522 / 0.570</td><td>0.736 / 0.760</td><td></td></tr><tr><td>MemRL (ours)</td><td>0.595 / 0.627</td><td>0.788 / 0.804</td><td>0.960 / 0.972</td><td>0.949 / 0.981</td><td>0.570 / 0.606</td><td>0.772 / 0.798</td><td></td></tr></table>

Table 2. Transfer Learning results on BigCodeBench, Lifelong Agent Bench and ALFWorld. We compare MEMRL against various baselines using the best validation results. The Average column represents the mean Success Rate across all benchmarks.   

<table><tr><td rowspan="2">Method</td><td>BigIntCodeBench</td><td colspan="2">Lifelong Agent Bench</td><td>ALFWorld</td><td>Average</td></tr><tr><td>Code Generation (Success Rate)</td><td>OS Task (Success Rate)</td><td>DB Task (Success Rate)</td><td>Exploration (Success Rate)</td><td>(Success Rate)</td></tr><tr><td>Model</td><td>GPT-4o</td><td>GPT-4o-mini</td><td>GPT-4o-mini</td><td>GPT-5-mini</td><td>-</td></tr><tr><td>No Memory</td><td>0.485</td><td>0.673</td><td>0.841</td><td>0.836</td><td>0.709</td></tr><tr><td>RAG</td><td>0.479</td><td>0.713</td><td>0.920</td><td>0.950</td><td>0.765</td></tr><tr><td>Self-RAG</td><td>0.500</td><td>0.653</td><td>0.881</td><td>0.950</td><td>0.746</td></tr><tr><td>Mem0</td><td>0.485</td><td>0.686</td><td>0.935</td><td>0.950</td><td>0.764</td></tr><tr><td>MemP</td><td>0.494</td><td>0.720</td><td>0.928</td><td>0.921</td><td>0.766</td></tr><tr><td>MemRL (ours)</td><td>0.508</td><td>0.746</td><td>0.942</td><td>0.979</td><td>0.794</td></tr></table>

Table 3. Ablation on Retrieval Scope. We compare MEMRL against an ablated version restricted to Single-Task Reflection.   

<table><tr><td>Setting</td><td>BCB</td><td>OS</td><td>DB</td><td>ALFWorld</td><td>HLE</td><td>Avg.</td></tr><tr><td>Single-Task Reflection</td><td>0.614</td><td>0.714</td><td>0.938</td><td>0.930</td><td>0.610</td><td>0.761</td></tr><tr><td>MEMRL (Cross-Task)</td><td>0.627</td><td>0.804</td><td>0.972</td><td>0.981</td><td>0.606</td><td>0.798</td></tr></table>

# 5.3.4. SENSITIVITY TO RETRIEVAL SIZE ( $k _ { 1 }$ AND k2).

To investigate the impact of retrieval bandwidth, we compare three memory density configurations on the HLE (CS/AI) subset benchmark: sparse $( k _ { 1 } = 3 , k _ { 2 } = 1 )$ ), moderate $( k _ { 1 } ~ = ~ 5 , k _ { 2 } ~ = ~ 3 )$ , and dense $( k _ { 1 } ~ = ~ 1 0 , k _ { 2 } ~ = ~ 5 )$ . As shown in Figure 6, performance follows an inverted-U trajectory, illustrating the trade-off between information sufficiency and context noise. The sparse setting limits performance due to insufficient guidance, whereas the dense setting degrades success rate by introducing distractions into the reasoning context. Consequently, the moderate configuration $( k _ { 1 } = 5 , k _ { 2 } = 3$ ) achieves the best result, effectively maximizing the signal-to-noise ratio of the retrieved context.

# 5.4. Discussion

In this section, we delve deeper into the mechanisms driving MEMRL’s performance, connecting empirical results to the challenge of balancing knowledge retention and adaptation.

![](images/9a5451064358285b3ccc476ee3853592aee479c5f00a126b7550a289eba3b187.jpg)  
Figure 6. Ablation on Retrieval Size $( k _ { 1 } , k _ { 2 } )$ . Performance comparison on the HLE (CS/AI) subset across sparse $( 3 / 1 )$ , moderate $( 5 / 3 )$ , and dense $( 1 0 / 5 )$ retrieval settings. The moderate setting achieves the optimal trade-off.

# 5.4.1. PREDICTIVE POWER OF THE Q CRITIC

As shown in Figure 7a, the learned Q-values exhibit a strong positive correlation (Pearson $r = 0 . 8 6 1 $ ) with empirical task success rates, rising from $2 1 . 5 \%$ in the lowest-confidence bin to $8 8 . 1 \%$ in the highest, which confirms the Critic’s ability to effectively rank memories by success likelihood. Beyond simple ranking, memory composition analysis (Figure 7b) reveals that the agent retains a small fraction of “failure” memories $( \sim 1 2 \%$ ) even in high-Q bins (0.9 − 1.0), suggesting that Q-values capture utility beyond binary outcomes by recognizing strategically useful near-misses. We

![](images/df7c9733f215c0277b35b091a434b86c3eb71af876ba7d90eea1aa90af0b5903.jpg)  
(a) Success Rate vs. Q-Range

![](images/1164a2e3327b23143c0ec2fd471a4db525e5862b2e71c0fc848f1df37ff7a047.jpg)  
(b) Memory Composition   
Figure 7. Q-Value Analysis. (a) Pearson $r = 0 . 8 6 1$ confirms Critic’s predictive power. (b) Failure memories $( \sim 1 2 \%$ ) in high Q-bins indicate latent strategic utility.

further substantiate this with concrete case studies in Appendix H. This indicates that the Critic prioritizes reusable guidance—including transferable procedural lessons from high-utility failures—rather than merely separating success from failure, thereby offering greater robustness than simple success-replay mechanisms.

# 5.4.2. STABILITY OF MEMRL

We analyze MEMRL through the lens of the stabilityplasticity dilemma. The superior CSR (Table 1) confirms that MEMRL effectively expands the solution space, enabling the agent to break through local optima. Furthermore, long-term dynamics (Figure 8) reveal a critical stability advantage: while heuristic methods like MemP suffer from catastrophic forgetting—evidenced by a widening gap between CSR and current Success Rate—MEMRL maintains synchronized growth. This is theoretically guaranteed by our stability analysis in Section 4.4, which constrains the policy to improve monotonically without drift.

We quantitatively validate these insights using the Forgetting Rate, defined as $F R = N _ { \mathrm { l o s t } } / N _ { \mathrm { f a i l } }$ , where $N _ { \mathrm { l o s t } }$ denotes tasks transitioning from previous success to current failure and $N _ { \mathrm { f a i l } }$ is the total number of failures in the current epoch (see Figure 9 in Appendix B.1 for the full trajectory). MEMRL achieves the lowest mean forgetting rate (0.041),

![](images/d334e4116762bc1585f75e2804fc9d9927b9720a17e7c42fd9bba1cc2205b67e.jpg)  
Figure 8. Epoch Success Rate and Cumulative Success Rate (CSR) of MEMRL and MemP in HLE.

outperforming MEMP (0.051). Additionally, ablation results demonstrate that removing z-score normalization and similarity gating causes the rate to spike to 0.073. This confirms that strict filtering is essential to manage utility variance and ensure that self-evolution remains stable.

# 5.4.3. EXTENDED ANALYSIS.

We conduct further investigations to characterize the underlying mechanisms and generalization of MEMRL. Specifically, Appendix B analyzes MEMRL’s role as a structural trajectory verifier and the correlation between task similarity and performance gains. Additionally, evaluations of advanced capabilities—including cross-model memory transferability and modular multi-task merging—are detailed in Appendix C, demonstrating MEMRL’s versatility and its capacity for modular capability expansion. We also analyze the cost and efficiency of MEMRL in Appendix F.

# 6. Limitations and Conclusion

Limitations and Future Work. While MEMRL establishes a foundation for non-parametric evolution, its runtime dynamics reveal several promising avenues. (i) The current step-wise update, though fast, may introduce high-variance noise in long-horizon trajectories, inspiring us to explore multi-step updates or periodic memory consolidation. (ii) Credit-assignment ambiguity during utility updates, especially with multiple referenced experiences, raises the need for more precise attribution methods like Shapley methods (Shapley et al., 1953) or value decomposition in multiagent reinforcement learning (Rashid et al., 2020; Sunehag et al., 2017). (iii) While MEMRL improves with increasing task exposure, performance may drift toward reflectionlike behavior when task similarity is low, highlighting the need for a sufficiently diverse yet relevant experience base; for industrial deployment, ensuring high task density and hierarchical abstraction may be crucial. Further detailed discussions on these and other challenges, including memory security, dedicated domains, and multi-agent memory sharing, are provided in Appendix G.

Conclusion. We proposed MEMRL, a non-parametric approach reconciling the stability-plasticity dilemma by treating memory retrieval as a value-based decision process. Through the Intent-Experience-Utility triplet structure and Monte Carlo style updates, MEMRL enables agents to selfevolve and differentiate high-utility strategies from semantic noise without weight updates. Extensive evaluations confirm MEMRL significantly outperforms baselines in both runtime adaptation and generalization. In a future where static training data becomes scarce, the interactive experiences generated by agents throughout their life cycle will become a new, vital source of knowledge. We hope this paradigm paves the way for building stable, continuously learning agents that efficiently adapt from interaction.

# Impact Statement

This paper presents MEMRL, a value-reinforced memory retrieval mechanism designed to enhance the long-term reasoning capabilities of LLM Agents. From a broader perspective, our work contributes to the development of more efficient and reliable autonomous systems. By optimizing the memory retrieval process, MEMRL reduces the computational overhead of large-scale agent deployments, potentially lowering the environmental impact of AI infrastructure. Furthermore, as LLM agents become more integrated into daily workflows, research into robust memory mechanisms helps ensure these systems remain grounded and consistent in their actions. We do not foresee any immediate negative social consequences specific to this algorithmic advancement, though we acknowledge that all autonomous systems should be deployed with appropriate human oversight to mitigate broader risks associated with AI decision-making.

# References

Asai, A., Wu, Z., Wang, Y., Sil, A., and Hajishirzi, H. Selfrag: Learning to retrieve, generate, and critique through self-reflection, 2023. URL https://arxiv.org/ abs/2310.11511.   
Bellman, R. Dynamic programming. science, 153(3731): 34–37, 1966.   
Chhikara, P., Khant, D., Aryan, S., Singh, T., and Yadav, D. Mem0: Building production-ready ai agents with scalable long-term memory, 2025. URL https: //arxiv.org/abs/2504.19413.   
Cuconasu, F., Trappolini, G., Siciliano, F., Filice, S., Campagnano, C., Maarek, Y., Tonellotto, N., and Silvestri, F. The power of noise: Redefining retrieval for rag systems. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR 2024, pp. 719–729. ACM,

July 2024. doi: 10.1145/3626772.3657834. URL http: //dx.doi.org/10.1145/3626772.3657834.   
Dempster, A. P., Laird, N. M., and Rubin, D. B. Maximum likelihood from incomplete data via the em algorithm. Journal of the Royal Statistical Society. Series B (Methodological), 39(1):1–38, 1977. ISSN 00359246. URL http://www.jstor.org/stable/2984875.   
Ethayarajh, K., Xu, W., Muennighoff, N., Jurafsky, D., and Kiela, D. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306, 2024.   
Fang, R., Liang, Y., Wang, X., Wu, J., Qiao, S., Xie, P., Huang, F., Chen, H., and Zhang, N. Memp: Exploring agent procedural memory. arXiv preprint arXiv:2508.06433, 2025. URL https://arxiv. org/abs/2508.06433.   
Finn, C., Abbeel, P., and Levine, S. Model-agnostic metalearning for fast adaptation of deep networks. In International conference on machine learning, pp. 1126–1135. PMLR, 2017.   
Gan, C., Yang, D., Hu, B., Zhang, H., Li, S., Liu, Z., Shen, Y., Ju, L., Zhang, Z., Gu, J., Liang, L., and Zhou, J. Similarity is not all you need: Endowing retrieval augmented generation with multi layered thoughts. arXiv preprint arXiv:2405.19893, 2024. doi: 10.48550/arXiv.2405.19893. URL https://arxiv. org/abs/2405.19893.   
Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., and Wang, H. Retrieval-augmented generation for large language models: A survey, 2024. URL https://arxiv.org/abs/2312.10997.   
Gick, M. L. and Holyoak, K. J. Analogical problem solving. Cognitive psychology, 12(3):306–355, 1980.   
Grossberg, S. Adaptive resonance theory: How a brain learns to consciously attend, learn, and recognize a changing world. Neural networks, 37:1–47, 2013.   
Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
Hassabis, D. and Maguire, E. A. Deconstructing episodic memory with construction. Trends in cognitive sciences, 11(7):299–306, 2007.   
Huang, Z., Tian, Z., Guo, Q., Zhang, F., Zhou, Y., Jiang, D., and Zhou, X. Licomemory: Lightweight and cognitive agentic memory for efficient long-term reasoning. arXiv preprint arXiv:2511.01448, 2025.

Javed, K., Shah, H., Sutton, R., and White, M. Online realtime recurrent learning using sparse connections and selective learning. Journal of Machine Learning Research, 2023.   
Karpukhin, V., Oguz, B., Min, S., Lewis, P. S., Wu, L., Edunov, S., Chen, D., and Yih, W.-t. Dense passage retrieval for open-domain question answering. In EMNLP (1), pp. 6769–6781, 2020.   
Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526, 2017.   
Kumaran, D., Hassabis, D., and McClelland, J. L. What learning systems do intelligent agents need? complementary learning systems theory updated. Trends in cognitive sciences, 20(7):512–534, 2016.   
Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., Yih, W.-t., Rockt ¨ aschel, ¨ T., et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.   
Li, H., Ding, L., Fang, M., and Tao, D. Revisiting catastrophic forgetting in large language model tuning. arXiv preprint arXiv:2406.04836, 2024.   
Li, L., Shi, D., Zhou, J., Wei, X., Yang, M., Jin, S., and Yang, S. Retrieval feedback memory enhancement large model retrieval generation method. arXiv preprint arXiv:2508.17862, 2025a.   
Li, Z., Song, S., Wang, H., Niu, S., Chen, D., Yang, J., Xi, C., Lai, H., Zhao, J., Wang, Y., et al. Memos: An operating system for memory-augmented generation (mag) in large language models. arXiv preprint arXiv:2505.22101, 2025b.   
Liang, J., He, R., and Tan, T. A comprehensive survey on test-time adaptation under distribution shifts. International Journal of Computer Vision, 133(1):31–64, 2025.   
McClelland, J. L., McNaughton, B. L., and O’Reilly, R. C. Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. Psychological review, 102(3):419–457, 1995.   
Metropolis, N. and Ulam, S. The monte carlo method. Journal of the American statistical association, 44(247): 335–341, 1949.   
Nader, K., Schafe, G. E., and Le Doux, J. E. Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. Nature, 406(6797):722–726, 2000.

Neal, R. M. and Hinton, G. E. A View of the Em Algorithm that Justifies Incremental, Sparse, and other Variants, pp. 355–368. Springer Netherlands, Dordrecht, 1998. ISBN 978-94-011-5014-9. doi: 10.1007/ 978-94-011-5014-9 12. URL https://doi.org/ 10.1007/978-94-011-5014-9_12.   
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.   
Packer, C., Wooders, S., Lin, K., Fang, V., Patil, S. G., Stoica, I., and Gonzalez, J. E. Memgpt: Towards llms as operating systems, 2024. URL https://arxiv. org/abs/2310.08560.   
Parisi, G. I., Kemker, R., Part, J. L., Kanan, C., and Wermter, S. Continual lifelong learning with neural networks: A review. Neural networks, 113:54–71, 2019.   
Phan, L., Gatti, A., Han, Z., Li, N., Hu, J., Zhang, H., Zhang, C. B. C., Shaaban, M., Ling, J., Shi, S., et al. Humanity’s last exam, 2025. URL https://arxiv.org/abs/ 2501.14249.   
Pritzel, A., Uria, B., Srinivasan, S., Badia, A. P., Vinyals, O., Hassabis, D., Wierstra, D., and Blundell, C. Neural episodic control. In International conference on machine learning, pp. 2827–2836. PMLR, 2017.   
Puterman, M. L. Markov decision processes: discrete stochastic dynamic programming. John Wiley & Sons, 2014.   
Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., and Finn, C. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36: 53728–53741, 2023.   
Rashid, T., Samvelyan, M., De Witt, C. S., Farquhar, G., Foerster, J., and Whiteson, S. Monotonic value function factorisation for deep multi-agent reinforcement learning. Journal of Machine Learning Research, 21(178):1–51, 2020.   
Salama, R., Cai, J., Yuan, M., Currey, A., Sunkara, M., Zhang, Y., and Benajiba, Y. Meminsight: Autonomous memory augmentation for llm agents. arXiv preprint arXiv:2503.21760, 2025.   
Schacter, D. L. and Addis, D. R. On the constructive episodic simulation of past and future events. Behavioral and Brain Sciences, 30(3):331–332, 2007.

Schacter, D. L., Addis, D. R., Hassabis, D., Martin, V. C., Spreng, R. N., and Szpunar, K. K. The future of memory: remembering, imagining, and the brain. Neuron, 76(4): 677–694, 2012.   
Schick, T., Dwivedi-Yu, J., Dess`ı, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., and Scialom, T. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551, 2023.   
Shapley, L. S. et al. A value for n-person games. 1953.   
Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., and Yao, S. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.   
Shridhar, M., Yuan, X., Cotˆ e, M.-A., Bisk, Y., Trischler, ´ A., and Hausknecht, M. Alfworld: Aligning text and embodied environments for interactive learning, 2021. URL https://arxiv.org/abs/2010.03768.   
Silver, D. and Sutton, R. S. Welcome to the era of experience. Google AI, 1, 2025.   
Singh, J., Magazine, R., Pandya, Y., and Nambi, A. Agentic reasoning and tool integration for llms via reinforcement learning. arXiv preprint arXiv:2505.01441, 2025. URL https://arxiv.org/abs/2505.01441.   
Stiennon, N., Ouyang, L., Wu, J., Ziegler, D., Lowe, R., Voss, C., Radford, A., Amodei, D., and Christiano, P. F. Learning to summarize with human feedback. Advances in neural information processing systems, 33:3008–3021, 2020.   
Sun, Y., Wang, X., Liu, Z., Miller, J., Efros, A., and Hardt, M. Test-time training with self-supervision for generalization under distribution shifts. In International conference on machine learning, pp. 9229–9248. PMLR, 2020.   
Sunehag, P., Lever, G., Gruslys, A., Czarnecki, W. M., Zambaldi, V., Jaderberg, M., Lanctot, M., Sonnerat, N., Leibo, J. Z., Tuyls, K., et al. Value-decomposition networks for cooperative multi-agent learning. arXiv preprint arXiv:1706.05296, 2017.   
Sutton, R. S. Learning to predict by the methods of temporal differences. Machine learning, 3(1):9–44, 1988.   
Sutton, R. S. and Barto, A. G. Reinforcement Learning: An Introduction. MIT Press, 2 edition, 2018. URL http://incompleteideas.net/ book/the-book-2nd.html.   
Tulving, E. et al. Episodic and semantic memory. Organization of memory, 1(381-403):1, 1972.

Wang, D., Shelhamer, E., Liu, S., Olshausen, B., and Darrell, T. Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726, 2020.   
Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An openended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.   
Watkins, C. J. and Dayan, P. Q-learning. Machine learning, 8(3):279–292, 1992.   
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
Wei, T., Sachdeva, N., Coleman, B., He, Z., Bei, Y., Ning, X., Ai, M., Li, Y., He, J., Chi, E. H., et al. Evo-memory: Benchmarking llm agent test-time learning with selfevolving memory. arXiv preprint arXiv:2511.20857, 2025.   
Wu, T., Luo, L., Li, Y.-F., Pan, S., Vu, T.-T., and Haffari, G. Continual learning for large language models: A survey, 2024. URL https://arxiv.org/abs/2402. 01364.   
Xu, W., Liang, Z., Mei, K., Gao, H., Tan, J., and Zhang, Y. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.   
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum? id=WE_vluYUL-X.   
Ye, Y. Task memory engine: Spatial memory for robust multi-step llm agents. arXiv preprint arXiv:2505.19436, 2025.   
Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.   
Zhang, Z., Dai, Q., Li, R., Bo, X., Chen, X., and Dong, Z. Learn to memorize: Optimizing llm-based agents with adaptive memory framework. arXiv preprint arXiv:2508.16629, 2025.   
Zheng, J., Cai, X., Li, Q., Zhang, D., Li, Z., Zhang, Y., Song, L., and Ma, Q. Lifelongagentbench: Evaluating llm agents as lifelong learners, 2025. URL https:// arxiv.org/abs/2505.11942.

Zhou, H., Chen, Y., Guo, S., Yan, X., Lee, K. H., Wang, Z., Lee, K. Y., Zhang, G., Shao, K., Yang, L., et al. Memento: Fine-tuning llm agents without fine-tuning llms. arXiv preprint arXiv:2508.16153, 2025a.   
Zhou, R., Yang, Y., Wen, M., Wen, Y., Wang, W., Xi, C., Xu, G., Yu, Y., and Zhang, W. TRAD: Enhancing llm agents with step-wise thought retrieval and aligned decision. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), pp. 3–13. ACM, 2024. URL https://arxiv.org/abs/2403.06221.   
Zhou, R., Hua, W., Pan, L., Cheng, S., Wu, X., Yu, E., and Wang, W. Y. Rulearena: A benchmark for rule-guided reasoning with llms in real-world scenarios. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025b.   
Zhuo, T. Y., Vu, M. C., Chim, J., Hu, H., Yu, W., Widyasari, R., Yusuf, I. N. B., Zhan, H., He, J., Paul, I., Brunner, S., Gong, C., Hoang, T., Zebaze, A. R., Hong, X., Li, W.-D., Kaddour, J., Xu, M., Zhang, Z., Yadav, P., Jain, N., Gu, A., Cheng, Z., Liu, J., Liu, Q., Wang, Z., Hui, B., Muennighoff, N., Lo, D., Fried, D., Du, X., de Vries, H., and Werra, L. V. Bigcodebench: Benchmarking code generation with diverse function calls and complex instructions, 2025. URL https: //arxiv.org/abs/2406.15877.

# A. Theoretical Analysis and Proofs

# A.1. Stability Analysis

We analyze the stability of MEMRL from a reinforcement learning perspective, focusing on the convergence behavior of the utility estimates stored in memory. MEMRL performs non-parametric runtime learning using a constant-step-size update. We show that under mild and realistic assumptions, the learned utility values converge in expectation to stable estimates of memory effectiveness, with bounded variance.

Setup. At each time step $t$ , the agent observes an intent state $s _ { t }$ , retrieves a memory item $m _ { t } \in \mathcal { M } _ { t }$ , generates an output $a _ { t }$ and receives a scalar reward $r _ { t } \in [ - 1 , 1 ]$ indicating task success or failure. The generation policy follows the decomposition defined in Eq. 1, where $\mu$ denotes the retrieval policy and $p _ { \mathrm { L L M } }$ is a frozen inference policy.

For each retrieved memory, MEMRL updates its utility using the exponential moving average rule as formulated in Eq. 4, with learning rate $\alpha \in ( 0 , 1 ]$ . For clarity in this analysis, we consider a fixed state–memory pair $( s , m )$ and write $Q _ { t } \equiv Q _ { t } ( s , m )$ .

Stationary Reward Assumption. We analyze the learning process on a fixed dataset and posit two key conditions that ensure the stability of the environment:

1. Frozen Inference Policy. The parameters of $p _ { \mathrm { L L M } } ( y | s , m )$ and the evaluator’s criteria are fixed.   
2. Fixed Task Distribution. Tasks $s$ are drawn from a stationary distribution over a fixed dataset.

These assumptions guarantee that the learning target is well-defined: the expected reward for any specific task-memory pair is time-invariant. Thus, we have:

$$
\mathbb {E} \left[ r _ {t} \mid s _ {t} = s, m _ {t} = m \right] = \beta (s, m). \tag {10}
$$

where the expectation is taken over the stochastic rewards resulting from the Inference Policy $p _ { L L M }$ distribution.

Expected Convergence of Utility Estimates. We now state the main stability result.

Theorem A.1. Let $\{ Q _ { t } \}$ be updated according to the rule in Eq. 4 with constant step size $\alpha \in ( 0 , 1 ]$ . If Eq. 10 holds and the pair $( s , m )$ is updated infinitely often, then:

$$
\lim  _ {t \rightarrow \infty} \mathbb {E} [ Q _ {t} ] = \mathbb {E} [ r _ {t} | s _ {t} = s, m _ {t} = m ] = \beta (s, m). \tag {11}
$$

Moreover, the convergence rate is exponential (Sutton & Barto, 2018):

$$
\mathbb {E} \left[ Q _ {t} \right] - \beta (s, m) = \left(1 - \alpha\right) ^ {t} \left(Q _ {0} - \beta (s, m)\right). \tag {12}
$$

Proof. Define the estimation error $\boldsymbol { e } _ { t } \triangleq \boldsymbol { Q } _ { t } - \beta ( s , m )$ . Based on the update rule in Eq. 4, the error recurrence relation is:

$$
e _ {t + 1} = (1 - \alpha) e _ {t} + \alpha \left(r _ {t} - \beta (s, m)\right).
$$

Taking conditional expectation given the history $\mathcal { F } _ { t }$ and using Eq. 10, we obtain:

$$
\mathbb {E} [ e _ {t + 1} | \mathcal {F} _ {t} ] = (1 - \alpha) e _ {t}.
$$

Taking full expectation yields:

$$
\mathbb {E} [ e _ {t + 1} ] = (1 - \alpha) \mathbb {E} [ e _ {t} ].
$$

Iterating the recursion gives $\mathbb { E } [ e _ { t } ] = ( 1 - \alpha ) ^ { t } e _ { 0 }$ , which converges to zero as $t \to \infty$ .

We provide the detailed derivation of the convergence proof in Appendix A.2.

Bounded Variance and Stability. If the reward variance $\mathrm { V a r } ( r _ { t } | s , m ) < \infty$ , then the variance of $Q _ { t }$ remains bounded:

$$
\lim  _ {t \rightarrow \infty} \operatorname {V a r} \left(Q _ {t}\right) \leq \frac {\alpha}{2 - \alpha} \operatorname {V a r} \left(r _ {t} | s, m\right). \tag {13}
$$

Thus, constant-step-size updates do not induce unbounded oscillations; instead, they yield stable utility estimates that track expected memory effectiveness while filtering high-frequency noise. We explicitly derive the variance bounds to demonstrate the global stability of the estimator under task clustering in Appendix A.3.

Global Stability via EM Convergence. The stability of the local estimate (Theorem A.1) extends to the global memory utility $Q ( m )$ . By the linearity of expectation, $Q ( m )$ acts as a Monte Carlo integrator striving to converge to:

$$
\lim  _ {t \rightarrow \infty} \mathbb {E} [ Q _ {t} (m) ] = \mathbb {E} [ r | m ] = \sum_ {s \in S (m)} \underbrace {\mathbb {E} [ r | s , m ]} _ {\text {S t a t i o n a r y}} \underbrace {\Pr (s | m)} _ {\text {R e t r i e v e - D e p e n d e n t}}. \tag {14}
$$

where $\begin{array} { r } { S ( m ) \triangleq \{ s \in S | \sin ( s , z _ { m } ) \geq \tau _ { A } \} } \end{array}$ denotes the effective support set for memory $m$ , comprising all task intents $s$ sufficiently similar to the memory’s intent embedding $z _ { m }$ to satisfy the Phase-A retrieval criterion.

A theoretical challenge arises here: the weighting term $\operatorname* { P r } ( s | m )$ is a latent variable governed by the retrieval policy $\mu ( \boldsymbol { m } | \boldsymbol { s } ; \mathcal { M } )$ , which itself shifts as $Q$ -values evolve. To prove convergence despite this dependency, we analyze MEMRL as a Generalized Expectation-Maximization (GEM) process (Dempster et al., 1977; Neal & Hinton, 1998). From a variational perspective, the system performs coordinate ascent on a global objective function ${ \mathcal { I } } ( Q , \mu )$ (the variational lower bound of expected reward): (i) E-Step (Policy Improvement): The Phase-B ranking updates the retrieval policy $\mu$ to align with current estimates, monotonically increasing $\mathcal { I }$ with respect to $\mu$ ; (ii) M-Step (Value Update): The utility update (Eq. 4) increases $\mathcal { I }$ with respect to $Q$ . By the Monotonic Improvement Theorem (Neal & Hinton, 1998), this alternating optimization guarantees that the system converges to a stationary point where the retrieve policy stabilizes $\left( \mu _ { t + 1 } \approx \mu _ { t } \right)$ . Consequently, the induced distribution $\operatorname* { P r } ( s | m )$ becomes time-invariant, ensuring that Eq. 14 holds and effectively preventing catastrophic forgetting by anchoring updates to a stable policy. More details can be found in Appendix A.4.

# A.2. Proof of Theorem A.1: Convergence of EMA Estimation

We aim to prove that for a fixed task-memory pair $( s , m )$ with a stationary reward distribution, the Q-value estimate $Q _ { t } ( s , m )$ converges in expectation to the true mean reward $\beta ( s , m )$ .

# Assumptions.

1. Stationary Reward. The reward $r _ { t }$ at step $t$ is drawn from a distribution induced by the stochastic action generation $a \sim p _ { L L M } ( a _ { t } | s _ { t } , m )$ , with a constant mean $\beta ( s , m ) = \mathbb { E } [ r _ { t } | s , m ]$ and finite variance $\sigma ^ { 2 }$ .   
2. Update Rule. The utility is updated via the linear EMA rule with learning rate $\alpha \in ( 0 , 1 )$ :

$$
Q _ {t + 1} = (1 - \alpha) Q _ {t} + \alpha r _ {t}.
$$

Derivation of Error Dynamics. Let ${ e _ { t } } \triangleq { Q _ { t } } - \beta ( s , m )$ be the estimation error at time step $t$ . Substituting $Q _ { t } \ =$ $e _ { t } + \beta ( s , m )$ into the update rule:

$$
\begin{array}{l} e _ {t + 1} + \beta (s, m) = (1 - \alpha) \left(e _ {t} + \beta (s, m)\right) + \alpha r _ {t} \\ e _ {t + 1} = (1 - \alpha) e _ {t} + (1 - \alpha) \beta (s, m) + \alpha r _ {t} - \beta (s, m) \\ e _ {t + 1} = (1 - \alpha) e _ {t} + \beta (s, m) - \alpha \beta (s, m) - \beta (s, m) + \alpha r _ {t} \\ e _ {t + 1} = (1 - \alpha) e _ {t} + \alpha \left(r _ {t} - \beta (s, m)\right). \tag {15} \\ \end{array}
$$

Convergence Analysis. We define $\mathcal { F } _ { t }$ as the filtration (history) up to time $t$ . Since the reward $r _ { t }$ depends on the action $a _ { t }$ sampled subsequently from the frozen LLM, taking the conditional expectation of Eq. 15 given $\mathcal { F } _ { t }$ :

$$
\mathbb {E} [ e _ {t + 1} | \mathcal {F} _ {t} ] = (1 - \alpha) e _ {t} + \alpha (\underbrace {\mathbb {E} [ r _ {t} | \mathcal {F} _ {t} ]} _ {\beta (s, m)} - \beta (s, m)) = (1 - \alpha) e _ {t}.
$$

By the Law of Iterated Expectations, taking the full expectation yields:

$$
\mathbb {E} \left[ e _ {t + 1} \right] = \mathbb {E} \left[ \mathbb {E} \left[ e _ {t + 1} \mid \mathcal {F} _ {t} \right] \right] = (1 - \alpha) \mathbb {E} \left[ e _ {t} \right].
$$

Iterating this recurrence relation from $t = 0$ :

$$
\mathbb {E} [ e _ {t} ] = (1 - \alpha) ^ {t} \mathbb {E} [ e _ {0} ].
$$

Since $0 < \alpha < 1$ , we have $| 1 - \alpha | < 1$ . Consequently:

$$
\lim  _ {t \rightarrow \infty} \mathbb {E} [ e _ {t} ] = \mathbb {E} [ e _ {0} ] \cdot \lim  _ {t \rightarrow \infty} (1 - \alpha) ^ {t} = 0. \tag {16}
$$

This proves that the estimator is unbiased in the limit, i.e., $\begin{array} { r } { \operatorname* { l i m } _ { t  \infty } \mathbb { E } [ Q _ { t } ] = \beta ( s , m ) } \end{array}$ .

# A.3. Bounded Variance and Global Stability

In this section, we provide the formal derivation for the variance bound of the estimator $Q _ { t }$ . We explicitly derive the finite-time variance formula via recursive unrolling and prove its asymptotic convergence, demonstrating how Phase-A clustering contributes to global stability.

Derivation of the Variance Bound. Let $\sigma ^ { 2 } \triangleq \operatorname { V a r } ( r _ { t } | s , m )$ be the variance of the reward signal, assumed to be finite. The EMA update rule is given by:

$$
Q _ {t + 1} = (1 - \alpha) Q _ {t} + \alpha r _ {t}.
$$

Since the reward $r _ { t }$ (current noise) is statistically independent of the current estimate $Q _ { t }$ (which is determined by history $\mathcal { F } _ { t - 1 } )$ , the variance of the sum is the sum of the variances:

$$
\begin{array}{l} \operatorname {V a r} \left(Q _ {t + 1}\right) = \operatorname {V a r} \left(\left(1 - \alpha\right) Q _ {t}\right) + \operatorname {V a r} \left(\alpha r _ {t}\right) \\ = (1 - \alpha) ^ {2} \operatorname {V a r} \left(Q _ {t}\right) + \alpha^ {2} \sigma^ {2}. \\ \end{array}
$$

Let $v _ { t } \triangleq \operatorname { V a r } ( Q _ { t } )$ . We obtain a linear recurrence relation $v _ { t + 1 } = ( 1 - \alpha ) ^ { 2 } v _ { t } + \alpha ^ { 2 } \sigma ^ { 2 }$ .

Recursive Unrolling. To solve for $v _ { t }$ , we expand the recurrence relation backward from step $t$ :

$$
\begin{array}{l} v _ {t} = (1 - \alpha) ^ {2} v _ {t - 1} + \alpha^ {2} \sigma^ {2} \\ = (1 - \alpha) ^ {2} \left[ (1 - \alpha) ^ {2} v _ {t - 2} + \alpha^ {2} \sigma^ {2} \right] + \alpha^ {2} \sigma^ {2} \\ = (1 - \alpha) ^ {4} v _ {t - 2} + \alpha^ {2} \sigma^ {2} [ 1 + (1 - \alpha) ^ {2} ] \\ \end{array}
$$

$$
= (1 - \alpha) ^ {2 t} v _ {0} + \alpha^ {2} \sigma^ {2} \sum_ {k = 0} ^ {t - 1} \left((1 - \alpha) ^ {2}\right) ^ {k}. \tag {17}
$$

Eq. 17 explicitly shows that the variance at time $t$ consists of two components: the decayed initial variance (first term) and the accumulated noise variance (second term).

Asymptotic Convergence. As $t \to \infty$ , since the learning rate $\alpha \in ( 0 , 1 )$ , the term $( 1 - \alpha ) ^ { 2 t }$ vanishes. The summation term is a geometric series $\textstyle \sum _ { k = 0 } ^ { \infty } r ^ { k } = { \frac { 1 } { 1 - r } }$ with ratio $r = ( 1 - \alpha ) ^ { 2 }$ . Thus:

$$
\lim _ {t \to \infty} v _ {t} = \alpha^ {2} \sigma^ {2} \cdot \frac {1}{1 - (1 - \alpha) ^ {2}}.
$$

Evaluating the denominator:

$$
1 - (1 - \alpha) ^ {2} = 1 - (1 - 2 \alpha + \alpha^ {2}) = 2 \alpha - \alpha^ {2} = \alpha (2 - \alpha).
$$

Substituting this back yields the tight variance bound:

$$
\lim  _ {t \rightarrow \infty} \operatorname {V a r} \left(Q _ {t}\right) = \frac {\alpha^ {2} \sigma^ {2}}{\alpha (2 - \alpha)} = \frac {\alpha}{2 - \alpha} \sigma^ {2}. \tag {18}
$$

Connection to Phase-A Clustering. This result provides the theoretical justification for the stability of MEMRL. While tasks within a memory cluster $\begin{array} { r } { S ( m ) \triangleq \{ s | \sin ( s , z _ { m } ) > \tau _ { A } \} } \end{array}$ may vary, the Smoothness Assumption implies their rewards are drawn from a distribution with bounded variance $\sigma _ { S ( m ) } ^ { 2 }$ . The derived bound $\frac { \alpha } { 2 - \alpha } \sigma _ { S ( m ) } ^ { 2 }$ guarantees that the memory utility $Q ( m )$ will not diverge but will instead oscillate within a controlled range around the true expected utility. This mechanism effectively filters out high-frequency noise from diverse task instances while retaining the stable generalized value.

# A.4. Convergence via Variational Inference

In this section, we provide a theoretical foundation for MEMRL, demonstrating that our retrieval strategy and update rules guarantee the convergence of value estimation.

# A.4.1. THE CONVERGENCE OBJECTIVE

Our ultimate goal is to ensure that the estimated utility $Q ( m )$ converges to the true expected return of memory $m$ . This target value is defined as:

$$
\lim  _ {t \rightarrow \infty} \mathbb {E} [ Q _ {t} (m) ] = \mathbb {E} [ r | m ] = \sum_ {s \in S (m)} \underbrace {\mathbb {E} [ r | s , m ]} _ {\text {S t a t i o n a r y}} \underbrace {\Pr (s | m)} _ {\text {R e t r i e v e - D e p e n d e n t}}. \tag {19}
$$

The challenge lies in the term $\operatorname* { P r } ( s | m )$ —the probability that a specific state $s$ triggers the retrieval of $m$ . This distribution depends on the retrieval policy $\mu _ { t } ( m | s )$ , which itself evolves during training, creating a circular dependency that threatens stability.

# A.4.2. VARIATIONAL OBJECTIVE WITH TRUST REGION

To resolve this, we formulate the problem as maximizing a global variational objective ${ \mathcal { I } } ( \mu , Q )$ . This objective serves as a tractable lower bound for the global expected return defined in Eq. 19, balanced by a semantic trust region:

$$
\mathcal {J} (\mu , Q) = \mathbb {E} _ {s \sim \mathcal {D}} \left[ \underbrace {\sum_ {m \in \mathcal {S} (s)} \mu (m | s) Q (s , m)} _ {\text {E x p e c t e d U t i l i t y} \approx \mathbb {E} [ Q _ {t} (m) ]} - \frac {1}{\beta} \underbrace {D _ {\mathrm {K L}} \left(\mu (\cdot | s) \| \pi_ {\mathrm {s i m}} (\cdot | s)\right)} _ {\text {S e m a n t i c T r u s t R e g i o n}} \right] \tag {20}
$$

Here, the expectation is taken over the state distribution $\mathcal { D }$ . The first term directly corresponds to the expected utility $\mathbb { E } [ Q _ { t } ( m ) ]$ we aim to converge, while $\pi _ { \mathrm { s i m } }$ represents the fixed semantic prior (derived from Phase-A). The KL-divergence term acts as a regularizer crucial for two reasons:

1. Trust Region: It constrains the policy to the support set $s$ , preventing the agent from retrieving high-Q but semantically irrelevant memories (out-of-distribution errors).   
2. Regularization: It stabilizes the learning dynamics during the “cold start” phase when Q-estimates are noisy.

# A.4.3. OPTIMIZATION VIA GENERALIZED EXPECTATION-MAXIMIZATION (GEM)

We treat the optimization of $\mathcal { I }$ as a GEM process, alternating between policy improvement and value evaluation:

E-Step (Policy Optimization). We assume the utility estimates $Q ( s , m )$ are fixed and seek the optimal retrieval policy $\mu ^ { * }$ that maximizes the global variational objective ${ \mathcal { I } } ( \mu , Q )$ . Since the expectation is taken over the state distribution $\mathcal { D }$ , we can maximize the objective for each state $s$ pointwise. The optimization problem for a specific state $s$ is:

$$
\max  _ {\mu (\cdot | s)} \left[ \sum_ {m \in \mathcal {S} (s)} \mu (m | s) Q (s, m) - \frac {1}{\beta} D _ {\mathrm {K L}} \left(\mu (\cdot | s) \| \pi_ {\text {s i m}} (\cdot | s)\right) \right] \tag {21}
$$

subject to the probability simplex constraint $\begin{array} { r } { \sum _ { m \in \mathcal { S } ( s ) } \mu ( m | s ) = 1 } \end{array}$

Expanding the KL-divergence term, the objective function becomes:

$$
\mathcal {L} (\mu) = \sum_ {m \in \mathcal {S} (s)} \mu (m | s) \left(Q (s, m) + \frac {1}{\beta} \log \frac {\pi_ {\operatorname* {s i m}} (m | s)}{\mu (m | s)}\right). \tag {22}
$$

To enforce the normalization constraint, we introduce the Lagrange multiplier $\lambda$ and construct the Lagrangian:

$$
L (\mu , \lambda) = \sum_ {m \in \mathcal {S} (s)} \mu (m | s) \left(Q (s, m) + \frac {1}{\beta} \log \frac {\pi_ {\mathrm {s i m}} (m | s)}{\mu (m | s)}\right) + \lambda \left(1 - \sum_ {m \in \mathcal {S} (s)} \mu (m | s)\right).
$$

Taking the derivative with respect to $\mu ( m | s )$ and setting it to zero:

$$
\begin{array}{l} \frac {\partial L}{\partial \mu (m | s)} = Q (s, m) + \frac {1}{\beta} \left(\log \frac {\pi_ {\operatorname {s i m}} (m | s)}{\mu (m | s)} + \mu (m | s) \cdot \frac {\mu (m | s)}{\pi_ {\operatorname {s i m}} (m | s)} \cdot \frac {- \pi_ {\operatorname {s i m}} (m | s)}{\mu (m | s) ^ {2}}\right) - \lambda \\ = Q (s, m) + \frac {1}{\beta} \left(\log \frac {\pi_ {\mathrm {s i m}} (m | s)}{\mu (m | s)} - 1\right) - \lambda = 0. \\ \end{array}
$$

Rearranging terms to solve for $\mu ( m | s )$ :

$$
\begin{array}{l} \log \frac {\pi_ {\mathrm {s i m}} (m | s)}{\mu (m | s)} = \beta \lambda - \beta Q (s, m) + 1 \\ \frac {\mu (m \mid s)}{\pi_ {\mathrm {s i m}} (m \mid s)} = \exp (\beta Q (s, m) - (\beta \lambda + 1)) \\ \mu (m | s) = \pi_ {\text {s i m}} (m | s) \exp (\beta Q (s, m)) \cdot \exp (- (\beta \lambda + 1)). \\ \end{array}
$$

Since $\exp ( - ( \beta \lambda + 1 ) )$ is independent of $m$ , it acts as a normalization constant $1 / Z ( s )$ . Thus, we recover the closed-form Boltzmann distribution used in our Phase-B retrieval:

$$
\mu^ {*} (m | s) = \frac {\pi_ {\operatorname {s i m}} (m | s) \exp (\beta Q (s , m))}{Z (s)} \propto \pi_ {\operatorname {s i m}} (m | s) \exp (\beta Q (s, m)).
$$

This derivation theoretically justifies our heuristic scoring function: the optimal retrieval policy naturally balances the semantic prior $\pi _ { \mathrm { s i m } }$ and the learned utility $Q$ . By taking the logarithm, we recover the specific scoring function used in our Phase-B Retrieval (Eq. 7):

$$
\log \mu^ {*} (m | s) \propto \underbrace {\log \pi_ {\mathrm {s i m}} (m | s)} _ {\approx \mathrm {s i m} (s, m)} + \beta Q _ {t} (s, m)
$$

This proves that our heuristic combination of similarity and Q-value is mathematically equivalent to the optimal policy under the variational objective.

M-Step (Policy Evaluation via Error Minimization). While the E-step improves the policy based on current estimates, the M-step ensures these estimates are grounded in reality. Fixing the policy $\mu _ { t + 1 }$ , our goal is to align the variational parameter $Q$ with the true environmental returns. We formulate this as minimizing the Mean Squared Error (MSE) between the estimated utility and the observed reward target $y = r$ (in our Monte Carlo style modeling):

$$
\min  _ {Q} \mathcal {L} (Q) = \mathbb {E} _ {\tau \sim \mu_ {t + 1}} \left[ \frac {1}{2} (y - Q (s, m)) ^ {2} \right] \tag {23}
$$

Minimizing this error is critical because it tightens the variational bound: it ensures that the expectation term $\mathbb { E } [ Q ]$ in the global objective $\mathcal { I }$ (Eq. 20) converges to the true expected return $\mathbb { E } [ r ]$ . The update rule used in our approach (Eq. 4) corresponds exactly to a Stochastic Gradient Descent (SGD) step on this objective:

$$
Q _ {t + 1} (s, m) \leftarrow Q _ {t} (s, m) - \alpha \nabla_ {Q} \mathcal {L} (Q) = Q _ {t} (s, m) + \alpha (y - Q _ {t} (s, m))
$$

By iteratively minimizing $\mathcal { L } ( Q )$ , the M-step propagates the environmental feedback into the utility estimates, ensuring that the subsequent E-step optimization occurs on a reliable value landscape.

# A.4.4. PROOF OF CONVERGENCE

By the Monotonic Improvement Theorem of GEM (Neal & Hinton, 1998), the sequence $( \mu _ { t } , Q _ { t } )$ is guaranteed to converge to a stationary point $( \mu ^ { * } , Q ^ { * } )$ . At stationarity, the policy stabilizes $( \mu _ { t + 1 } \approx \mu _ { t } )$ ), which implies that the inverse retrieval probability $\operatorname* { P r } ( s | m )$ becomes time-invariant:

$$
\operatorname * {P r} (s | m) = \frac {\mu^ {*} (m | s) \operatorname * {P r} (s)}{\sum_ {s ^ {\prime}} \mu^ {*} (m | s ^ {\prime}) \operatorname * {P r} (s ^ {\prime})}
$$

Consequently, the “Retrieve-Dependent” term in Eq. 19 is anchored. With a fixed data distribution, the $Q _ { t } ( m )$ converges to the unique fixed point:

$$
\lim  _ {t \rightarrow \infty} Q _ {t} (m) \rightarrow \mathbb {E} _ {\mu^ {*}} [ r | m ] \tag {24}
$$

Thus, our approach theoretically guarantees that the memory values converge to the true expected returns under the optimal retrieval policy.

# B. Extended Analysis and Insights

# B.1. Detailed Analysis of Forgetting Dynamics

In the main text, we reported the mean forgetting rate to quantify the stability of our method. Here, we provide a detailed visual analysis of how the forgetting rate evolves throughout the learning process on the HLE benchmark.

![](images/85bed06fb9681b1eb91556fc341e3086343aa300df98024ae09eb80b49fd547b.jpg)  
Figure 9. Forgetting Rate Trajectory on HLE. We compare the cumulative forgetting rate of the MEMRL against the strong baseline MemP and an ablated version of MEMRL (w/o Normalization & Gating). A lower curve indicates better stability (less catastrophic forgetting).

As illustrated in Figure 9, MEMRL demonstrates superior stability compared to MemP. Specifically:

• Stability vs. Plasticity: While MemP shows a gradual upward trend in forgetting rate as the number of episodes increases, MEMRL maintains a consistently low rate. This indicates that our dual-retrieval mechanism effectively balances the acquisition of new strategies without overwriting stable, high-utility memories.   
• Impact of Filtering: The ablation curve (denoted as w/o Norm & SimGate) exhibits significant higher overall forgetting rate (0.073 mean). The visible spikes in the curve suggest that without z-score normalization and similarity gating, the agent frequently retrieves and reinforces “noisy” strategies that work for specific instances but degrade general performance on previously mastered tasks.

# B.2. MEMRL as a Trajectory Verifier.

Table 4 reveals a correlation between task structural complexity and performance gain. The gains are most profound in multi-step sequential tasks (e.g., ALFWorld $+ 6 . 2 \%$ Points(pp)) compared to single-turn tasks (e.g., BigCodeBench $+ 2 . 5 \%$

Table 4. Impact of Task Structure. Comparison of Cumulative Success Rate (CSR) gains. Multi-step tasks benefit significantly more from MEMRL.   

<table><tr><td>Benchmark</td><td>Interaction</td><td>MemP (%)</td><td>MEMRL (%)</td><td>Gain (pp)</td></tr><tr><td>ALFWorld</td><td>Multi-step</td><td>91.9</td><td>98.1</td><td>+6.2</td></tr><tr><td>OS Task</td><td>Multi-step</td><td>74.2</td><td>80.4</td><td>+6.2</td></tr><tr><td>HLE</td><td>Single-step</td><td>57.0</td><td>60.6</td><td>+3.6</td></tr><tr><td>BigIntBench</td><td>Single-step</td><td>60.2</td><td>62.7</td><td>+2.5</td></tr></table>

pp). In sequential tasks, a retrieved memory must be valid for the entire trajectory. Standard semantic retrieval often fetches memories that match the initial instruction but fail in later steps. By propagating the final reward backward to the memory utility $Q$ , MEMRL effectively learns to verify the whole trajectory, filtering out brittle policies that look correct only on the surface.

This analysis indicates that MEMRL transcends the role of a simple retrieval enhancer to function as a Trajectory Verifier. Its value is maximized in tasks with complex temporal dependencies, where it learns to select memories that ensure the structural integrity of the entire interaction process.

# B.3. Impact of Task Similarity on Memory Efficacy

To understand the underlying conditions where MEMRL thrives, we analyze the correlation between the intra-dataset semantic similarity $( \mathrm { S i m } _ { i n t r a } )$ and the absolute performance gain provided by our method ( $\Delta =$ Success RateMemRL − Success RateNoMem).

![](images/8c04cba0c8697abcc44e6f434b526c0c314eab70562137ee23ee82c692134bfd.jpg)  
Figure 10. Similarity-based Generalization.

As illustrated in Figure 10, we analyze the correlation between intra-dataset semantic similarity and the absolute performance gain $( \Delta )$ provided by MEMRL. The linear regression trend reveals a general positive correlation: environments with higher structural repetition allow the agent to retrieve and reuse optimal policies more effectively. At the upper extreme, ALFWorld (similarity 0.518) acts as a strong anchor point for this trend, exhibiting the highest repetition and a corresponding maximum performance boost ( $\Delta = + 0 . 1 7 2$ ). This confirms that for highly repetitive procedural tasks, memory serves as an effective shortcut to optimal trajectories. Following the regression line, benchmarks with moderate similarity—such as Lifelong-OS (0.390) and BigCodeBench (0.308)—cluster in the middle region, showing steady improvements $( \Delta \approx + 0 . 1 0 \sim + 0 . 1 1 )$ where the agent successfully generalizes coding patterns or OS commands across related instructions.

# The HLE Anomaly: Generalization vs. Memorization.

HLE presents a unique outlier. Despite having the lowest similarity (0.186) due to its diverse, multi-disciplinary nature, it exhibits a surprisingly high runtime gain $( 0 . 3 5 7  0 . 5 7 0 , \Delta = + 0 . 2 1 3 )$ . This gain operates on a different mechanism than ALFWorld. In high-similarity benchmarks, MEMRL succeeds via Positive Transfer—generalizing shared patterns to new instances. In contrast, the gain in HLE stems from Runtime Memorization. Since HLE questions are distinct and domain-specific, the agent relies on the Runtime Learning phase to “memorize” specific solutions to difficult problems through repeated exposure. This distinction highlights MEMRL’s versatility: it supports both pattern generalization in

structured domains and specific knowledge acquisition in diverse domains.

# C. Advanced Capabilities: Transfer and Merging

# C.1. Cross-Model Memory Transferability

We investigate whether the procedural knowledge captured by MEMRL is specific to the training policy or if it generalizes across different architectures. To test this, we take a memory bank fully trained for 10 epochs on the HLE benchmark using our strongest agent, Gemini-3-pro, and directly transfer it—without any fine-tuning—to three distinct inference models: Qwen3-235B, GPT-5.2(High), and Gemini-3-flash.

As summarized in Table 5, the transferred memory yields substantial zero-shot performance gains across all models. Notably, smaller or distilled models experience the largest relative improvements; for instance, Qwen3-235B improves by over $3 \times$ $( 0 . 1 5 0  0 . 5 3 1 )$ ) and Gemini-3-flash nearly doubles its success rate $( 0 . 3 4 7  0 . 5 8 3 )$ ). Even GPT-5.2(High), a highly capable reasoning model, sees a significant boost $\mathrm { 0 . 3 5 4 \to 0 . 5 7 1 }$ ).

These results suggest that MEMRL captures model-agnostic problem-solving patterns—such as efficient code skeletons and reasoning templates—rather than model-specific artifacts. This effectively allows the memory bank to function as a portable knowledge base, enabling weaker models to “inherit” the capabilities of a stronger teacher model through simple retrieval.

Table 5. Cross-Model Transfer Results on HLE. Performance of various agents using a frozen memory bank originally trained by Gemini-3-pro. $\Delta$ denotes the absolute improvement over the base model.   

<table><tr><td>Inference Model</td><td>Base Score</td><td>Transfer Score</td><td>Gain (Δ)</td></tr><tr><td>Qwen3-235B</td><td>0.150</td><td>0.531</td><td>+0.381</td></tr><tr><td>Gemini-3-flash</td><td>0.347</td><td>0.583</td><td>+0.236</td></tr><tr><td>GPT-5.2 (High)</td><td>0.354</td><td>0.571</td><td>+0.217</td></tr></table>

# C.2. Multi-Task Memory Merging and Interference Analysis

To evaluate the composability and robustness of MEMRL in multi-task scenarios, we conducted a memory merging experiment. Specifically, we aggregated the finalized memory banks learned from the last epoch of two distinct domains within the Lifelong Agent Bench: Operating System Control $( M _ { O S } )$ and Database Management $( M _ { D B } )$ . The agent was then evaluated on each respective benchmark using this unified, heterogeneous memory bank $( M _ { U n i f i e d } = M _ { O S } \cup M _ { D B } )$ , without any further training or fine-tuning.

As presented in Table 6, the results demonstrate that merging memories introduces negligible interference. The performance on the DB Task remains identical (0.960), while the OS Task exhibits only a marginal fluctuation $( 0 . 7 8 8 \to 0 . 7 8 4 )$ . This robustness is intrinsic to our Two-Phase Retrieval mechanism. Since the semantic spaces of OS commands and SQL queries are largely orthogonal, the Phase-A similarity filter effectively acts as a semantic gate, automatically excluding irrelevant cross-task memories before they enter the value-based ranking stage. This confirms that MEMRL supports modular memory composition, allowing agents to scale capabilities by simply merging memory modules without suffering from negative transfer or catastrophic interference.

Table 6. Memory Merging Results. Performance comparison between using task-specific individual memory banks versus a merged unified memory bank. The minimal delta confirms strong resistance to cross-task interference.   

<table><tr><td>Benchmark Task</td><td>Individual Memory (Original)</td><td>Merged Memory (MOS ∪ MDB)</td><td>Performance Delta (Δ)</td></tr><tr><td>Lifelong-OS (OS Task)</td><td>0.788</td><td>0.784</td><td>-0.004</td></tr><tr><td>Lifelong-DB (DB Task)</td><td>0.960</td><td>0.960</td><td>0.000</td></tr></table>

# D. Baseline and Benchmark Details

To ensure a rigorous evaluation, we compare MEMRL against a diverse set of baselines ranging from simple sampling strategies to advanced agentic memory systems. All baselines are evaluated under a unified frozen-backbone setting to

isolate the contribution of the memory and retrieval mechanisms.

# D.1. Baselines

We categorize the baselines into three groups based on their interaction with memory and environment:

# I. Test-Time Scaling Strategy

• Pass@k: This is a standard sampling-based baseline. It generates $k$ independent candidate solutions for a given query and selects the best one based on the benchmark’s verifier (if available) or reports the success rate if at least one candidate passes. This baseline serves as a measure of the inherent capability of the frozen LLM without any memory persistence.

# II. Retrieval-Augmented Generation (RAG) Approaches

• RAG (Lewis et al., 2020): Represents the standard semantic retrieval paradigm. It utilizes an embedding model to encode the current query and retrieves the top- $k$ most semantically similar past experiences (or documents) from the external memory. These retrieved contexts are then prepended to the prompt to guide the LLM’s generation.   
• Self-RAG (Asai et al., 2023): An advanced RAG variant that incorporates a self-critique mechanism. Unlike standard RAG, Self-RAG performs selective retrieval and uses a critique model (or self-prompting) to verify the relevance and factual correctness of the retrieved content before integrating it into the generation process.

# III. Agentic Memory Systems

• Mem0 (Chhikara et al., 2025): A recently proposed memory layer for LLMs that manages memory through structured operations. It employs specific APIs for adding, retrieving, and updating memory, aiming to maintain a personalized and persistent context across interactions.   
• MemP (Fang et al., 2025): A framework focused on procedural memory. It distills past successful trajectories into reusable, procedure-like memory entries. MemP maintains a memory repository using a build-retrieve-update cycle, allowing the agent to recall high-level plans rather than raw trajectory data.

# D.2. Benchmark Datasets

We evaluate performance across four benchmarks selected to cover diverse domains: code generation, OS interactions, embodied decision-making, and multidisciplinary reasoning.

• BigCodeBench (Zhuo et al., 2025): A challenging benchmark for library-oriented code generation that requires agents to implement complex functionalities using diverse third-party libraries. Unlike traditional benchmarks focused on algorithmic snippets, BigCodeBench emphasizes practical software engineering capabilities. We evaluate on the BigCodeBench-Instruct (Full) split, which tasks the agent with synthesizing complete functional code from natural language instructions across the full range of difficulty levels.   
• Lifelong Agent Bench (Zheng et al., 2025): Designed to evaluate agents in a continuous learning setting involving Operating System (OS) and Database (DB) interactions. It tests the agent’s capacity to adapt to new tools and commands over a long horizon without forgetting previous skills.   
• ALFWorld (Shridhar et al., 2021): An embodied navigation and manipulation benchmark. It requires the agent to solve textual logic puzzles within a simulated household environment (e.g., “put a clean apple in the fridge”). This tests the agent’s ability to learn and retrieve multi-step plans.   
• Humanity’s Last Exam (HLE) (Phan et al., 2025): A rigorous multidisciplinary reasoning benchmark featuring hard problems from mathematics, humanities, and sciences. It serves as a stress test for the agent’s general reasoning capability and its ability to retrieve relevant knowledge for disparate tasks.

# E. Implementation and Reproducibility Details

To facilitate reproducibility, we provide the exact model versions, hyperparameter settings, and environmental configurations used in our experiments.

# E.1. Model Specifications

All LLM reasoning and generation tasks were performed using the models listed in Table 7. We used the official APIs with a fixed temperature to ensure deterministic evaluation where possible.

Table 7. Model and API Configurations.   

<table><tr><td>Component</td><td>Configuration / Version</td><td>Notes</td></tr><tr><td rowspan="4">Backbone LLM</td><td>GPT-4o</td><td>Used for BigCodeBench</td></tr><tr><td>GPT-4o-mini</td><td>Used for Lifelong Bench</td></tr><tr><td>GPT-5-mini</td><td>Used for ALFWorld</td></tr><tr><td>Gemini-3-pro</td><td>Used for HLE</td></tr><tr><td>Embedding Model</td><td>Text-Embedding-3-Large</td><td>Used for Intent and Query encoding</td></tr><tr><td rowspan="3">Generation Params</td><td>Temperature = 0.0</td><td>General (Greedy decoding)</td></tr><tr><td>Temperature = 0.6</td><td>Specific for HLE (gemini-3-pro)</td></tr><tr><td>Top-p = 1.0</td><td>Default</td></tr></table>

# E.2. Hyperparameter Settings

Table 8 details the specific hyperparameters used for MEMRL and the baselines. The similarity threshold $\delta$ is adaptive to the dataset density; specifically, we determine $\delta$ by calculating the pairwise cosine similarity distribution of task descriptions within each benchmark and selecting the threshold at the top $20 \%$ quantile. This ensures that only the most relevant historical experiences are considered for retrieval.

Table 8. Hyperparameter Settings across Benchmarks.   

<table><tr><td rowspan="2">Parameter</td><td rowspan="2">Description</td><td colspan="5">Benchmark Setting</td></tr><tr><td>BigIntBench</td><td>Lifelong Bench (OS)</td><td>Lifelong Bench (DB)</td><td>ALFWorld</td><td>HLE</td></tr><tr><td colspan="7">MEMRL (Ours)</td></tr><tr><td>α</td><td>Learning Rate (Eq. 4)</td><td>0.3</td><td>0.3</td><td>0.3</td><td>0.3</td><td>0.3</td></tr><tr><td>λ</td><td>Q-Weight Balance (Eq. 7)</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>δ</td><td>Similarity Threshold</td><td>0.38</td><td>0.50</td><td>0.37</td><td>0.62</td><td>0.25</td></tr><tr><td>k1</td><td>Phase-A Recall Size</td><td>10</td><td>10</td><td>10</td><td>5</td><td>5</td></tr><tr><td>k2</td><td>Phase-B Select Size</td><td>5</td><td>5</td><td>5</td><td>3</td><td>3</td></tr><tr><td>Qinit</td><td>Initial Q-value</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td colspan="7">Baselines</td></tr><tr><td>kRAG</td><td>Retrieval Top-k</td><td>5</td><td>5</td><td>5</td><td>3</td><td>3</td></tr><tr><td>kSelfRAG</td><td>Retrieval Top-k</td><td>5</td><td>5</td><td>5</td><td>3</td><td>3</td></tr><tr><td>kMem0</td><td>Retrieval Top-k</td><td>5</td><td>5</td><td>5</td><td>3</td><td>3</td></tr><tr><td>kMemP</td><td>Retrieval Top-k</td><td>5</td><td>5</td><td>5</td><td>3</td><td>3</td></tr></table>

# E.3. Data Partitioning

To evaluate the effectiveness of MEMRL, we categorize our experiments into Runtime Learning and Transfer Learning settings. Table 9 summarizes the dataset sizes and partitioning strategies used for each benchmark.

Table 9. Summary of Data Partitioning and Dataset Sizes.   

<table><tr><td>Benchmark</td><td>Runtime Learning</td><td>Transfer Learning</td><td>Split / Note</td></tr><tr><td>HLE</td><td>2,500 tasks</td><td>-</td><td>Full set (Runtime only)</td></tr><tr><td>Lifelong Agent (OS)</td><td>500 tasks</td><td>500 tasks</td><td>7:3 Split (Seed 42)</td></tr><tr><td>Lifelong Agent (DB)</td><td>500 tasks</td><td>500 tasks</td><td>7:3 Split (Seed 42)</td></tr><tr><td>BigIntCodeBench-I (Full)</td><td>1,140 tasks</td><td>1,140 tasks</td><td>7:3 Split (Seed 42)</td></tr><tr><td>ALFWorld</td><td>3,553 tasks</td><td>140 tasks</td><td>valid Seen (Novel instances)†</td></tr></table>

† For ALFWorld Transfer Learning, the agent uses the same 3,553 tasks as memory context but is evaluated on 140 novel instances of seen task types.

For benchmarks utilizing random splits (OS, DB, and BCB), we use a fixed random seed of 42 to ensure reproducibility. In ALFWorld, the Transfer Learning phase specifically tests generalization to new instances within known categories, ensuring the agent learns procedural patterns rather than specific trajectories.

# E.4. Model Selection and Performance Analysis

We select the backbone model for each benchmark to ensure a valid learning signal relative to task complexity. Since MemRL relies on high-value trajectories to perform utility update over retrieved memories, extremely low initial competence can make the feedback effectively unusable. For instance, on challenging benchmarks such as HLE, weaker models may start at $\approx 4 \%$ success rate (e.g., GPT-4o-mini), yielding too few successful trajectories for stable utility estimation; in this scenario, environmental feedback is dominated by noise, which can prevent meaningful learning and hinder convergence.

At the other extreme, using the strongest available models on simpler benchmarks can cause performance saturation (ceiling effects), leaving little headroom to quantify the marginal gains attributable to the memory mechanism. Therefore, we match model capability to each task’s difficulty to avoid both the no-signal situation (insufficient successes) and the ceiling effect (insufficient headroom). This design choice enables a more faithful evaluation of improvements that are intrinsic to MemRL rather than artifacts of an ill-posed performance condition.

Importantly, our evaluation spans backbones of different scales—from “mini” to “pro” tiers (Table 7)—to test whether MemRL remains effective across capability levels. The resulting performance trends support the scale-invariant robustness of MemRL. Moreover, the cross-model transfer results (Table 5) further indicate that the procedural knowledge captured in memory is portable across backbones, suggesting that the learned utility over memories generalizes beyond any single model’s inherent strength.

To further explore the generality of MemRL across different model capabilities and task complexities, we also deployed the GPT-4o-mini model in the ALFWorld benchmark. This model is identical to the backbone used in the Lifelong Agent Bench, allowing for a direct comparison of its performance across environments with varying exploration intensities. As an exploration-intensive environment, ALFWorld poses significant demands on the base model’s reasoning and planning capabilities. The Table 10 and Table 11 present the runtime learning and knowledge transfer abilities of GPT-4o-mini.

Table 10. Runtime Learning Performance of GPT-4o-mini Across Benchmarks   

<table><tr><td rowspan="2">Method</td><td colspan="2">Lifelong Agent (OS)</td><td colspan="2">Lifelong Agent (DB)</td><td colspan="2">ALFWorld</td><td colspan="2">Average</td></tr><tr><td>Last Epoch SR</td><td>CSR</td><td>Last Epoch SR</td><td>CSR</td><td>Last Epoch SR</td><td>CSR</td><td>Last Epoch SR</td><td>CSR</td></tr><tr><td>No Memory</td><td>0.674</td><td>N/A</td><td>0.860</td><td>N/A</td><td>0.278</td><td>N/A</td><td>0.604</td><td>N/A</td></tr><tr><td>Pass@10</td><td>N/A</td><td>0.756</td><td>N/A</td><td>0.928</td><td>N/A</td><td>0.462</td><td>N/A</td><td>0.715</td></tr><tr><td>MemP</td><td>0.736</td><td>0.742</td><td>0.960</td><td>0.966</td><td>0.299</td><td>0.413</td><td>0.665</td><td>0.707</td></tr><tr><td>MEMRL</td><td>0.788</td><td>0.804</td><td>0.960</td><td>0.972</td><td>0.440</td><td>0.680</td><td>0.730</td><td>0.819</td></tr></table>

These results, using GPT-4o-mini as a consistent backbone, unequivocally demonstrate MEMRL’s superior and generalized effectiveness across all benchmarks. In runtime learning (Table 10), MEMRL achieves the highest average Last Epoch Success Rate and Cumulative Success Rate, showing significant gains: its average Last Epoch SR is $1 2 . 6 \%$ points higher than “No Memory”, and its average CSR surpasses MemP by $1 1 . 2 \%$ points. This advantage is particularly pronounced in the challenging ALFWorld environment. Similarly, in transfer learning (Table 11), MEMRL secures the highest average Success Rate, $1 0 . 6 \%$ points higher than “No Memory” and $6 . 1 \%$ points higher than MemP. These consistent improvements across

Table 11. Transfer Learning Performance of GPT-4o-mini Across Benchmarks   

<table><tr><td>Method</td><td>Lifelong Agent (OS)</td><td>Lifelong Agent (DB)</td><td>ALFWorld</td><td>Average</td></tr><tr><td>No Memory</td><td>0.673</td><td>0.841</td><td>0.314</td><td>0.609</td></tr><tr><td>MemP</td><td>0.720</td><td>0.928</td><td>0.314</td><td>0.654</td></tr><tr><td>MEMRL</td><td>0.746</td><td>0.942</td><td>0.457</td><td>0.715</td></tr></table>

tasks and learning paradigms confirm that MEMRL effectively enhances the GPT-4o-mini backbone’s decision-making capabilities, leveraging learned utility over memories for more robust performance.

# F. Cost and Efficiency Analysis

Real-world deployment of autonomous agents requires balancing performance gains with computational costs. In this section, we analyze the token consumption and runtime latency of MEMRL compared to the strong baseline MemP on the compute-intensive Humanity’s Last Exam (HLE) benchmark.

# F.1. Token Consumption

Since MEMRL operates as a non-parametric approach without gradient-based fine-tuning, the primary cost arises from LLM API calls. We compare the average token usage per question (Q) across the entire learning trajectory (10 epochs).

MEMRL’s token consumption is comparable to that of MemP, as both methods utilize identical interaction loops (reasoning $^ +$ summarization). On the HLE benchmark, the average total token consumption per question for MEMRL is approximately 32K. This similarity in token usage is because the complexity of MEMRL lies in how memories are retrieved and updated (via Q-values), not in how much context is fed to the LLM. Therefore, the significant performance gains of MEMRL reported in the main text are achieved without increasing the inference budget.

# F.2. Runtime Latency and Stability

A common concern with two-stage retrieval and reinforcement learning components is the potential for increased latency. We empirically validate the wall-clock time required to complete each epoch (2,500 questions) in Figure 11.

![](images/7fb55288404884a86a1dd512da3a64e2fdece6acf3feef8e36a9d15afbebe589.jpg)  
Figure 11. Wall-Clock Time per Epoch on HLE. The x-axis represents learning epochs (2,500 questions per epoch), and the y-axis represents the total time in hours. MEMRL exhibits a stable runtime profile, confirming that the algorithmic overhead is negligible compared to system variances (e.g., network latency).

Negligible Algorithmic Overhead. Figure 11 demonstrates that the runtime of MEMRL is commensurate with, and often more stable than, that of MemP. The fluctuations observed (e.g., the spikes in the MemP curve at Epochs 4 and 6) are attributed to external factors such as API network latency and throughput variability, rather than algorithmic complexity.

Specifically, the additional components in MEMRL introduce minimal computational cost:

• Dual-Stage Retrieval: This involves basic vector dot-products and scalar score weighting, operating in milliseconds.   
• RL Update: The Q-value update in a Monte Carlo style on scalars, which is an $O ( 1 )$ operation.

Compared to the hours required for LLM generation over 2,500 questions, these millisecond-level operations are virtually imperceptible. The stability of the MEMRL curve further suggests that our method does not introduce complex blocking operations that would exacerbate network-induced delays.

# G. Extended Limitations and Future Work

This appendix provides a more detailed discussion on the limitations of MEMRL and outlines promising avenues for future research, building upon the foundations established in the main paper.

# G.1. Update Protocols and Memory Consolidation

The current step-wise update protocol of MEMRL, while enabling rapid adaptation, can introduce high-variance noise in long-horizon trajectories. This presents a challenge for stabilizing value estimation over extended sequences. A valuable future direction involves exploring more robust multi-step update mechanisms, which could offer slower but more stable learning. Furthermore, combining these with periodic consolidation of similar intentions and experiences within the memory bank could significantly improve the spatial efficiency of the memory, reducing redundancy and enhancing retrieval quality.

# G.2. Precise Credit Assignment in Multi-Memory Updates

A significant challenge arises from the credit-assignment ambiguity encountered when multiple experiences from the memory are referenced and updated simultaneously. Determining the precise contribution of each referenced memory to the final outcome is complex and affects the efficiency of learning. Future research could investigate methods for more precise experience attribution, drawing inspiration from techniques like Shapley values (Shapley et al., 1953), commonly used in cooperative game theory, or value decomposition methods prevalent in multi-agent reinforcement learning (Rashid et al., 2020; Sunehag et al., 2017). Such approaches could lead to more accurate updates and faster convergence.

# G.3. Task Similarity and Generalization

MEMRL’s effectiveness is observed to improve with the number of encountered tasks, aligning well with the characteristics of runtime learning. This is because the algorithm leverages past experiences, and a richer, more diverse set of similar experiences directly enhances its ability to retrieve relevant knowledge. However, when task similarity in the experience base is low, the method may inherently degrade into a less efficient reflection-like behavior, as direct experience transfer is limited. This is not a deficiency of the method but rather a characteristic of its reliance on learned utility from memory. For industrial deployment and real-world applications, this implies that maintaining a sufficiently high “task similarity density” within the agent’s operating environment or its collected memory is crucial. Strategies to achieve this could include active curriculum learning, environment design that promotes diverse yet related tasks, or techniques that facilitate hierarchical abstraction to enable generalization beyond direct, low-level experience matches. Enhancing retrieval mechanisms to proactively identify and adapt to novel task structures, or integrating hierarchical abstraction, are key for improving cross-task generalization.

# G.4. Memory Security and Robustness to Attack

MEMRL is sensitive to the quality of feedback, particularly vulnerable to “reward hacking” if the verifier produces false positives. Incorrectly learned high Q-values from spurious feedback can quickly solidify and propagate erroneous behavioral patterns, leading to systemic failures. This highlights a critical challenge: memory security. A maliciously injected sample into the memory bank could rapidly diffuse pollution, potentially causing an intelligent agent to collapse. Future research must address the safety and trustworthiness of agent memories. Fortunately, the mutable nature of experience offers a silver lining: once contamination or attack is identified, polluted experiences can be swiftly pruned, allowing for recovery without disrupting the utility of previously learned valid experiences.

# G.5. Multi-Agent Collaboration and Shared Memory

As many enterprises transition from single agents to multi-agent clusters (swarms), the question of “shared memory” becomes important. Does the MEMRL approach support knowledge sharing, allowing lessons learned and updated Q-values from one agent to immediately benefit others? We’ve explored scenarios where the memory, treated as a persistent resource, can indeed be shared among different models or agents (Appendix C.1). Furthermore, adopting a multi-agent paradigm, akin to distributed parameter updates, could be highly beneficial. Each agent could accumulate experience and save it

as memory during its operation, contributing to a large, collective memory pool. This would enable different agents to implicitly leverage each other’s learned experiences, fostering a more collaborative and efficient learning ecosystem.

Beyond this collective pooling, a significant and promising frontier involves investigating the selective nature of such knowledge diffusion through the lens of transfer learning. As the number of agents increases, particularly beyond simple pairings, the challenge of ”what to share” and ”with whom to share” becomes a critical research direction. Future work could explore mechanisms that differentiate between universally applicable procedural insights and task-specific noise, ensuring that memory transfer remains contextually relevant and avoids the risks of negative transfer. Navigating these trade-offs between global collective intelligence and targeted knowledge distribution will be essential for building scalable, self-evolving swarms.

# G.6. Dedicated Domains and Hybrid Architectures

While this work strongly advocates for freezing large language models (LLMs) to prevent catastrophic forgetting, a question arises regarding highly specialized domains where the base model may not comprehend foundational terminology. Is MEMRL alone sufficient in such cases? We anticipate a future hybrid model where companies periodically fine-tune foundational models (e.g., annually) to update their core vocabulary and understanding, while simultaneously leveraging MEMRL for daily behavioral adaptation and runtime learning. This approach requires the base model to possess a relatively high level of “intelligence” to initially grasp and subsequently learn new terms through interaction and feedback.

# H. Case Study: High-Utility Failure Analysis (Near-Misses)

This appendix section provides qualitative case studies of high-value near-miss memories mined from the 10-epoch OS-Interaction run. CS denotes Case Study. Each box below contains: origin task, retrieved reflection memory, a short explanation, and the target task where it was retrieved.

# Case Study 1 (CS1)

Utility. selected=19, success=19 (100.0%).

# Origin task.

```txt
Update the SSH daemon configuration using sed to set: Port=2222, PermitRootLogin=no, X11Forwarding=no, ClientAliveInterval=300, ClientAliveCountMax=2, MaxAuthTries=2, MaxSessions=2, PasswordAuthentication=no, AllowUsers='user1 user2', UseDNS=no, and Protocol=2. 
```

# Retrieved reflection memory.

- #1: [MEMORY TYPE] FAILURE_REFLECTION

[TASK]

```txt
Update the SSH daemon configuration using sed to set: Port=2222, PermitRootLogin=no, X11Forwarding=no, ClientAliveInterval=300, ClientAliveCountMax=2, MaxAuthTries=2, MaxSessions=2, PasswordAuthentication=no, AllowUsers='user1 user2', UseDNS=no, and Protocol=2. 
```

[REFLECTION]

ROOT CAUSE: The ‘sed‘ command did not modify the commented lines in the configuration file, which is why the new settings were not applied as expected.   
PATTERN TO AVOID: Avoid assuming that ‘sed‘ will replace commented lines without explicitly handling them.   
CORRECT APPROACH: Use ‘sed‘ to either uncomment the lines before replacing them or add the new configurations if they are commented out.

Explanation. This reflection captures a subtle config-edit pitfall: naive sed replacement may miss commented defaults. The transferable lesson is to explicitly handle commented lines (uncomment-or-append), which generalizes across configuration-editing tasks.

Target task (retrieval rank 1; run outcome: correct).

Modify the SSH daemon configuration file at /etc/ssh/sshd_config to change the port to 2222, disable root login, set max sessions to 2, and set client alive interval to 300. Create a backup file before making changes.

# Case Study 2 (CS2)

Utility. selected $= 2 0$ , success $\scriptstyle = 2 0$ (100.0%).

Origin task.

Replace ’DEBUG $\Longleftarrow$ True’ with ’DEBUG $\Longleftarrow$ False’ in /app/settings.cfg, append ’LOG_LEVEL= INFO’ after the modified line, and create a backup file.

# Retrieved reflection memory.

- #2: [MEMORY TYPE] FAILURE_REFLECTION

[TASK]

Replace ’DEBUG $\Longleftarrow$ True’ with ’DEBUG $\Longleftarrow$ False’ in /app/settings.cfg, append ’LOG_LEVEL= INFO’ after the modified line, and create a backup file.

[REFLECTION]

- ROOT CAUSE: The ‘sed‘ command added the line ‘LOG_LEVEL $=$ INFO‘ twice due to the way it was structured to append after the modified ‘DEBUG‘ line.   
- PATTERN TO AVOID: Avoid using multiple ‘-e‘ options in ‘sed‘ that can lead to unintended duplications when appending lines.   
- CORRECT APPROACH: Instead, use a single ‘sed‘ command that ensures the line is only added once, or check for the existence of the line before appending it.

Explanation. The originating attempt nearly completes the edit but fails due to a non-idempotent text transformation (duplicating an appended line). The memory teaches an idempotent editing pattern (single-pass edit or existence check), which transfers to many file-edit tasks.

Target task (retrieval rank 2; run outcome: correct).

Replace ’DEBUG $\Longleftarrow$ True’ with ’DEBUG $\Longleftarrow$ False’ in /app/settings.cfg, append ’LOG_LEVEL= INFO’ after the modified line, and create a backup file.

# Case Study 3 (CS3)

Utility. selected=48, success ${ = } 4 4$ (91.7%).

Origin task.

Create a user ’testuser’ with a home directory, set password expiration policy (max 60 days, min 7 days, warning 7 days), add to ’testgroup’, create ’/home/ testuser/private’ with 770 permissions, create ’/shared’ directory accessible only by ’testgroup’, add ’export PATH $= \hat { S }$ PATH:/custom’ to .bashrc, and create ’. hushlogin’ file with 644 permissions.

# Retrieved reflection memory.

- #4: [MEMORY TYPE] FAILURE_REFLECTION [TASK]

Create a user ’testuser’ with a home directory, set password expiration policy (max 60 days, min 7 days, warning 7 days), add to ’testgroup’, create ’/home/ testuser/private’ with 770 permissions, create ’/shared’ directory accessible only by ’testgroup’, add ’export PATH $= \hat { S }$ PATH:/custom’ to .bashrc, and create ’. hushlogin’ file with 644 permissions.

[REFLECTION]

- ROOT CAUSE: The output of the commands was empty, indicating successful execution without any visible confirmation or errors.   
- PATTERN TO AVOID: Relying solely on silent execution without feedback can lead to uncertainty about whether tasks were completed successfully.   
- CORRECT APPROACH: Always check the status of commands using conditional statements or output messages to confirm successful execution and provide clarity on the task’s completion.

Explanation. This memory highlights a common near-miss failure mode: assuming silent command output implies completion without verification. The transferable practice is to add explicit post-checks (exit codes, file/permission/ownership checks) after critical steps.

Target task (retrieval rank 4; run outcome: correct).

Create a user ’testuser’ with a home directory, set password expiration policy (max 60 days, min 7 days, warning 7 days), add to ’testgroup’, create ’/home/ testuser/private’ with 770 permissions, create ’/shared’ directory accessible only by ’testgroup’, add ’export PATH $= \hat { S }$ PATH:/custom’ to .bashrc, and create ’. hushlogin’ file with 644 permissions.

# Case Study 4 (CS4)

Utility. selected $= 2 6$ , success $^ { - 2 3 }$ $( 8 8 . 5 \% )$ ).

Origin task.

Run a sleep process in the background for 3 seconds, log its start and end times ( in epoch format) to ’/var/log/sleep.log’, store the process ID in ’/tmp/ sleep_pid’, and ensure the PID file is removed after the process completes.

Retrieved reflection memory.

- #1: [MEMORY TYPE] FAILURE_REFLECTION

[TASK]

Run a sleep process in the background for 3 seconds, log its start and end times ( in epoch format) to ’/var/log/sleep.log’, store the process ID in ’/tmp/ sleep_pid’, and ensure the PID file is removed after the process completes.

[REFLECTION]

- ROOT CAUSE: The PID file was not created because the command to capture the PID may not have executed correctly, leading to multiple executions of the sleep process without proper PID storage.   
- PATTERN TO AVOID: Avoid executing commands in a loop or without proper checks, which can lead to unintended repeated executions and failures to capture necessary outputs.   
- CORRECT APPROACH: Implement a structured approach with error handling and verification after each critical command to ensure that each step completes successfully before proceeding to the next.

Explanation. The failure comes from missing control/verification around process management (PID capture and cleanup), which leads to repeated execution and missing artifacts. The memory transfers as a general recipe: capture PID once, verify artifacts, and sequence commands with error handling.

Target task (retrieval rank 1; run outcome: correct).

Run a sleep process in the background for 3 seconds, log its start and end times ( in epoch format) to ’/var/log/sleep.log’, store the process ID in ’/tmp/ sleep_pid’, and ensure the PID file is removed after the process completes.

# Case Study 5 (CS5)

Utility. selected $_ { = 9 4 }$ , success ${ } _ { \cdot = 8 8 }$ $( 9 3 . 6 \% )$

Origin task.

Create a secure configuration file ’/etc/appconfig/settings.conf’ using vi, set ownership to root:configgroup, permissions to 660, and add ’appuser’ to ’ configgroup’. Ensure the file contains ’# Configuration file’ and ’key $=$ value’ lines.

Retrieved reflection memory.

- #2: [MEMORY TYPE] FAILURE_REFLECTION

[TASK]

Create a secure configuration file ’/etc/appconfig/settings.conf’ using vi, set ownership to root:configgroup, permissions to 660, and add ’appuser’ to ’ configgroup’. Ensure the file contains ’# Configuration file’ and ’key $=$ value’ lines.

[REFLECTION]

- ROOT CAUSE: The initial assumption that the group ‘configgroup‘ and the user ‘ appuser‘ already existed led to errors when attempting to set ownership and modify group membership.   
- PATTERN TO AVOID: Avoid making assumptions about the existence of users or groups in the system without verifying first.   
- CORRECT APPROACH: Always check for the existence of necessary users and groups before performing operations that depend on them, and create them if they do not exist.

Explanation. This reflection encodes a high-frequency near-miss: assuming required users/groups exist. The transferable heuristic is to preflight environment state (e.g., id/getent) and create prerequisites before applying ownership/permission changes.

Target task (retrieval rank 2; run outcome: correct).

Create a configuration file ’/etc/appconfig/settings.conf’ using ’vi’ containing the lines ’SERVER_IP $^ { = 1 }$ 92.168.1.100’ and ’DEBUG_MODE $=$ false’, set its group ownership to ’appgroup’, and permissions to 640.

# Case Study 6 (CS6)

Utility. selected $\scriptstyle 1 = 5 9$ , success ${ } _ { 1 = 5 5 }$ $( 9 3 . 2 \% )$ ).

Origin task.

Create a user ’testuser’, add them to group ’testgroup’, configure ’/data’ with group ownership ’testgroup’ and permissions 770, create ’/data/notes.txt’ using ’vi’ with content ’Hello from vi’, and create an executable script ’/check.sh’ using ’vi’ to verify group ownership of ’/data’.

Retrieved reflection memory.

```txt
- #5: [MEMORY TYPE] FAILURE_REFLECTION  
[TASK]  
Create a user 'testuser', add them to group 'testgroup', configure '/data' with group ownership 'testgroup' and permissions 770, create '/data/notes.txt' using 'vi' with content 'Hello from vi', and create an executable script '/check.sh' using 'vi' to verify group ownership of '/data'. 
```

[REFLECTION]   
```txt
- ROOT CAUSE: The output being empty does not indicate a failure, but rather that the commands executed successfully without any output. 
```

```txt
- PATTERN TO AVOID: Assuming that an empty output signifies an error can lead to misunderstandings about command execution results. 
```

```txt
- CORRECT APPROACH: Always verify the success of commands by checking their exit status or using logging to confirm that actions were completed as intended. 
```

Explanation. The lesson is that empty output usually signals success, not failure; therefore success should be validated via exit status and direct state checks. This reduces false negatives across shell automation tasks.

Target task (retrieval rank 5; run outcome: correct).

```txt
Create a file '/shared config/settings.conf' containing 'USER=user' and 'GROUP=user'  
then programmatically create a user and group using the values from the file. Finally, ensure '/shared/data' is owned by the group with 770 permissions. 
```

# I. Prompts

We provide the exact prompt strings and message templates used by our MEMRL implementation across all benchmarks. To minimize ambiguity, we separate prompts used to summarize experiences into memories from prompts used at task time for generation/inference.

# I.1. Experience Summarization Prompts

# HLE: Experience Summarization Prompts

Trajectory serialization (stored as the episode trajectory).

```txt
QUESTION {question} SOLUTION {model_output_stripped} 
```

High-level script generation prompt.

Analyze the following detailed task trajectory and create a concise, high-level script that captures the essential steps and decision points.

The script should be:

1. Generic enough to apply to similar tasks   
2. Specific enough to provide useful guidance   
3. 3-5 high-level steps maximum   
4. Focus on the strategy and key decisions, not detailed actions

Trajectory: {trajectory}

High-level script:

# Failure reflection prompt.

Task: {task_description}

Failed trajectory: {failed_trajectory}

This task failed. Analyze what went wrong and suggest improvements for future similar tasks.

Focus on:

1. Incorrect assumptions   
2. Steps to improve   
3. What to avoid next time

Provide a brief reflection:

# Stored memory content templates.

```txt
Successful memory  
Task: {task_description}  
SCRIPT:  
{script}  
TRAJECTORY:  
{trajectory}  
# Failure memory  
TASK REFLECTION:  
Task: {task_description}  
What went wrong:  
{reflection}  
Failed approach:  
{failed_trjectory} 
```

# ALFWorld: Experience Summarization Prompts

# Trajectory serialization (stored as the episode trajectory).

```txt
The full ALFWorld dialogue history is stored as the trajectory: List[{ "role": ... , "content": ...}, ... ] When interpolated into the summarization prompts, it is treated as a string representation. 
```

# High-level script generation prompt.

```txt
Analyze the following detailed task trajectory and create a concise, high-level script that captures the essential steps and decision points. 
```

The script should be:

1. Generic enough to apply to similar tasks   
2. Specific enough to provide useful guidance   
3. 3-5 high-level steps maximum   
4. Focus on the strategy and key decisions, not detailed actions

Trajectory: {trajectory}

High-level script:

# Failure reflection prompt.

Task: {task_description}

Failed trajectory: {failed_trajectory}

This task failed. Analyze what went wrong and suggest improvements for future similar tasks.

Focus on:

1. Incorrect assumptions   
2. Steps to improve   
3. What to avoid next time

Provide a brief reflection:

# Stored memory content templates.

```txt
Successful memory  
Task: {task_description}  
SCRIPT:  
{script}  
TRAJECTORY:  
{trajectory}  
# Failure memory  
TASK REFLECTION:  
Task: {task_description}  
What went wrong:  
{reflection}  
Failed approach:  
{failed_trjectory} 
```

# BCB (BigCodeBench): Experience Summarization Prompts

# Trajectory serialization (stored as the episode trajectory).

```txt
[BCB] epoch={epoch} phase={phase} task_id={task_id}  
[PROMPT]  
{bcb_task_prompt}  
[GENERATED CODE]  
``python  
{model_code}  
```  
[EVAL]  
{"status": "...", "error": "..."} 
```

# High-level script generation prompt.

Analyze the following detailed task trajectory and create a concise, high-level script that captures the essential steps and decision points.

The script should be:

1. Generic enough to apply to similar tasks   
2. Specific enough to provide useful guidance   
3. 3-5 high-level steps maximum   
4. Focus on the strategy and key decisions, not detailed actions

Trajectory: {trajectory}

High-level script:

# Failure reflection prompt.

```txt
Task: {task_description}  
Failed trajectory:  
{failed_trjectory}  
This task failed. Analyze what went wrong and suggest improvements for future similar tasks.  
Focus on:  
1. Incorrect assumptions  
2. Steps to improve  
3. What to avoid next time  
Provide a brief reflection: 
```

# Stored memory content templates.

```txt
Successful memory  
Task: {task_description}  
SCRIPT:  
{script}  
TRAJECTORY:  
{trajectory}  
# Failure memory  
TASK REFLECTION:  
Task: {task_description}  
What went wrong:  
{reflection}  
Failed approach:  
{failed_trjectory} 
```

# LLB (LifelongAgentBench): Experience Summarization Prompts

Trajectory serialization (stored as the episode trajectory).

```snap
{role_1}: {content_1}
{role_2}: {content_2}
...
```
```c 
```

High-level script generation prompt.

Analyze the following detailed task trajectory and create a concise, high-level script that captures the essential steps and decision points.

The script should be:

1. Generic enough to apply to similar tasks   
2. Specific enough to provide useful guidance   
3. 3-5 high-level steps maximum   
4. Focus on the strategy and key decisions, not detailed actions

Trajectory: {trajectory}

High-level script:

# Failure reflection prompt.

Task: {task_description}

Failed trajectory: {failed_trajectory}

This task failed. Analyze what went wrong and suggest improvements for future similar tasks.

Focus on:

1. Incorrect assumptions   
2. Steps to improve   
3. What to avoid next time

Provide a brief reflection:

# Stored memory content templates.

# Successful memory

Task: {task_description}

SCRIPT:

{script}

TRAJECTORY:

{trajectory}

# Failure memory

TASK REFLECTION:

Task: {task_description}

What went wrong:

{reflection}

Failed approach:

{failed_trajectory}

# I.2. Generation and Inference Prompts

# HLE: Generation and Inference Prompts

# System prompt (exact-match).

Your response should be in the following format:  
Explanation: {your explanation for your final answer}  
Exact Answer: {your succinct, final answer}  
Confidence: {your confidence score between $0\%$ and $100\%$ for your answer}

# System prompt (multiple-choice).

Your response should be in the following format:  
Explanation: {your explanation for your answer choice}  
Answer: {your chosen answer}  
Confidence: {your confidence score between $0\%$ and $100\%$ for your answer}

# Retrieved memory injection (system message).

$= = =$ Successful Memories $= = =$ {memory_full_content_1}   
{memory_full_content_2} $= = =$ Failed Memories (for caution) $= = =$ {memory_full_content_3}

# Optional repeat-attempt reflection note (system message).

```txt
You attempted this question before.  
Result: {CORRECT|INCORRECT}  
Question: {question}  
Previous attempt (solution only): {solution_only}  
Reflect on mistakes or gaps, then solve the problem again with a better solution. 
```

# User message text block (when images exist).

```txt
Now solve the following question:  
[ Image IDs: \{question_image_ids\} ]  
{question}  
Attached images:  
1. {{img_id_1}} ({source_1})  
2. {{img_id_2}} ({source_2})  
... 
```

# Message ordering.

1) system: exact-match OR multiple-choice format prompt   
2) system: optional reflection note (if enabled)   
3) system: optional retrieved memory context   
4) user: question content (text + optional images)

# ALFWorld: Generation and Inference Prompts

# Base system prompt (ReAct format $^ +$ action space).

Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.

For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.\nAction: your next action".

The available actions are:

1. go to {recep}

2. take {obj} from {recep}

3. move {obj} to {recep}

4. open {recep}

5. close {recep}

6. use {obj}

7. clean {obj} with {recep}

8. heat {obj} with {recep}

9. cool {obj} with {recep}

where {obj} and {recep} correspond to objects and receptacles.

After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened ", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>

Action: <your next action>

# Retrieved memory injection (system message).

In addition to the example, you have the following memories from your own past experiences. Use them to help you if they are relevant:

--- SUCCESSFUL MEMORIES (Examples to follow) ---

{formatted_success_memories_joined}

--- FAILED MEMORIES (Examples to avoid or learn from) ---

{formatted_failed_memories_joined}

# Current task prompt (user message).

```txt
Now, it's your turn to solve a new task.  
{task_description} 
```

Per-step observation prompt (user message).

```txt
Observation: {observation} 
```

Message ordering (high level).

1) system: base ALFWorld system prompt   
2) user/assistant: selected few-shot example dialogue (sequence of messages)   
3) system: optional retrieved memory context   
4) user: new task prompt   
5) loop: append user Observation: ..., model replies with Thought/Action

# BCB (BigCodeBench): Generation and Inference Prompts

Retrieved memory injection (system message).

```txt
[Retrieved Memory Context]   
##Memory1(id=mem_id_1，sim={similarity_1}） {memory_content_1}   
#Memory2id=mem_id_2，sim={similarity_2}） {memory_content_2}
```

Dataset-provided task prompt selection (user message).

if split $= =$ "instruct": return task["instruct_prompt"]   
if split $= =$ "complete": return task["complete_prompt"]

Message ordering.

```txt
1) system: optional [Retrieved Memory Context] ...  
2) user: {bcb_task_prompt} 
```

# LLB (LifelongAgentBench): Generation and Inference Prompts

Base system prompt.

You are an execution-focused AI agent solving database and operating-system tasks.

You may receive a [Retrieved Memory Context] block with past experiences from similar problems.

These are **references for learning**, not guaranteed solutions:

[MEMORY TYPE] SUCCESS_PROCEDURE: A successful approach from a similar task learn the pattern.   
[MEMORY TYPE] FAILURE_REFLECTION: A failed attempt with lessons avoid similar mistakes.

Use the memories as inspiration, but always analyze your current task independently and

adapt your approach based on its specific requirements.

# Strict output constraint (DB tasks).

STRICT OUTPUT FORMAT (LLB:DB, do not violate):

1) After your reasoning, include exactly ONE action line: - Action: Operation - Action: Answer   
2) If Action: Operation, put exactly ONE SQL statement in the FIRST fenced code block using ‘‘‘sql, on a single line. Do not add any extra text after that block.   
3) If Action: Answer, include ‘Final Answer: ...‘ on the next line and do not add extra text after that.

# Strict output constraint (OS tasks).

STRICT OUTPUT FORMAT (LLB:OS, do not violate):

1) After your reasoning, include exactly ONE action line: - Act: bash - Act: finish   
2) If Act: bash, the next lines MUST be a ‘‘‘bash fenced code block with your Bash commands. Do not include any other code blocks.   
3) If Act: finish, it must be the last line (no code blocks, no extra text).   
4) Do NOT use ‘Action:‘ in OS tasks (use ‘Act:‘ only).

# Retrieved memory injection block.

[Retrieved Memory Context] $= = =$ SUCCESSFUL EXPERIENCES (Learn from these) $= = =$ [SUCCEED 1] [TYPE: {mem_type}]   
{content} $= = =$ FAILED EXPERIENCES (Avoid these mistakes) $= = =$ [FAILURE 1] [TYPE: {mem_type}]   
{content}

Prompt assembly ordering (system prompt).

1) base system prompt   
2) optional [Retrieved Memory Context] ...   
3) strict output format block appended at the end (task-aligned)
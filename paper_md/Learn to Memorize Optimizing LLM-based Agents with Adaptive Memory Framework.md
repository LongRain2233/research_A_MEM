# Learn to Memorize: Optimizing LLM-based Agents with Adaptive Memory Framework

Zeyu Zhang1, Quanyu Dai2, Rui $\mathbf { L i } ^ { 1 }$ , Xiaohe $\mathbf { B o } ^ { 1 }$ , Xu Chen1, Zhenhua Dong2

1Gaoling School of Artificial Intelligence, Renmin University of China

2Huawei Noah’s Ark Lab

{zeyuzhang,xu.chen}@ruc.edu.cn, daiquanyu@huawei.com

# Abstract

LLM-based agents have been extensively applied across various domains, where memory stands out as one of their most essential capabilities. Previous memory mechanisms of LLM-based agents are manually predefined by human experts, leading to higher labor costs and suboptimal performance. In addition, these methods overlook the memory cycle effect in interactive scenarios, which is critical to optimizing LLM-based agents for specific environments. To address these challenges, in this paper, we propose to optimize LLM-based agents with an adaptive and datadriven memory framework by modeling memory cycles. Specifically, we design an MoE gate function to facilitate memory retrieval, propose a learnable aggregation process to improve memory utilization, and develop task-specific reflection to adapt memory storage. Our memory framework empowers LLM-based agents to learn how to memorize information effectively in specific environments, with both off-policy and on-policy optimization. In order to evaluate the effectiveness of our proposed methods, we conduct comprehensive experiments across multiple aspects. To benefit the research community in this area, we release our project at https://github.com/nuster1128/learn_to_memorize.

# 1 Introduction

Large language model (LLM) based agents have been widely applied in various fields [1–3], such as finance [4], recommender systems [5], and personal assistants [6]. During the interaction with environments, agents are supposed to perceive and memorize observations to support subsequent decision-making processes. These memories are crucial for maintaining the consistency of contextual interactions, and providing necessary information to facilitate reasoning under the current environment [7]. Previous studies have proposed various methods to construct memory of LLM-based agents. These methods primarily rely on retrieval-augmented generation (RAG) [8] to acquire relevant information about the current states for in-context learning [9–11]. Besides, recent approaches also explore transforming observations into modifications of model parameters to implement memories [12].

However, there are two significant limitations in previous studies. First of all, most of these methods are manually predefined by human experts, lacking a data-driven optimization process. For instance, Generative Agents [13] introduce a retrieval function by combining different aspects of memories (such as relevance and recency), but manually assign their weights. Similarly, MemoryBank [10] summarizes critical information from observations before storage, yet it relies on fixed and intuitive prompts. In such cases, human experts need to try numerous parameters for better performance, resulting in increased labor costs and suboptimal performance, as demonstrated in Figure 1(a).

Moreover, previous studies have largely overlooked the memory cycle effect, as illustrated in Figure 1(b), which highlights a significant difference between vanilla LLMs and LLM-based agents. For vanilla LLMs, due to the lack of interactive feedback from environments, they fail to establish a cycle

![](images/738b6a4bb101169545e4c11767d7e9c2132f1a62e6b91e83197167fffc6d2987.jpg)  
(a) Manual vs. Data-driven Memory Design

![](images/1217199a99f99cb8dd47c63f48b37ed4102ed014878ac5bb9a0d79bd6636b92b.jpg)  
(b) Memory Cycle Effect   
Figure 1: (a) In memory retrieval, the optimal weights for different aspects vary across different tasks. Similarly, in memory storage, the attention of information storage is task-dependent as well. However, manual model adaptation by human experts results in higher labor costs and suboptimal performance. (b) We demonstrate the memory cycle during interactions between agents and environments.

between memory storage and utilization. In contrast, LLM-based agents can store observations from environments as memories to support subsequent reasoning for actions. These actions will further influence the states of environments, resulting in new feedback as observations that will be stored in the next cycle. From this perspective, the policies of memory storage and utilization mutually influence each other during agent-environment interactions. Therefore, learning either of them in isolation may lead to suboptimal performance due to neglecting the memory cycle effect.

In this paper, we propose an adaptive memory framework that can be optimized with a data-driven approach. This framework formulates a memory cycle that consists of retrieval, utilization, and storage procedures. Specifically, we design a Mix-of-Expert (MoE) gate function across multiple aspects to implement adaptive combination for retrieval. We implement prompt optimization through taskspecific reflection to adjust the extraction attention for storage. We propose a learnable aggregation process to better utilize retrieved memories, which can be aligned by direct preference optimization (DPO) [14]. In addition, to optimize our framework based on the training data, we propose offpolicy and on-policy optimization strategies. Finally, we conduct extensive experiments to verify the effectiveness of our framework. To benefit the research community, we have released our project on the GitHub repository1. Our primary contributions can be summarized as follows:

• We propose an adaptive and data-driven memory framework that empowers LLM-based agents to learn to memorize, with optimizable memory retrieval, utilization, and storage procedures.   
• We formulate the memory cycle effect during agent-environment interactions, and propose offpolicy and on-policy optimization strategies for our memory framework.   
• We conduct comprehensive experiments to demonstrate the effectiveness of our framework in improving the performance of LLM-based agents when interacting with environments.

The rest of our paper is organized as follows. After introducing related works in Section 2, we propose our framework in Section 3 and optimization strategies in Section 4. We present extensive experiments with discussions in Section 5, and draw conclusions in Section 6.

# 2 Related Works

# 2.1 Reinforcement Learning Based Agents

Reinforcement learning (RL) primarily studies how agents can optimize their actions within an environment to maximize cumulative rewards [15]. Unlike supervised learning, RL emphasizes learning by interacting with environments rather than relying on labeled data. Specifically, an RLbased agent makes decisions, receives feedback, and adjusts its strategy based on the results of its actions. The target is to establish a mapping from states to actions to maximize their rewards. Previous research has extensively explored optimizing RL-based agents, using methods such as Policy Gradient [16], DQN [17], Actor-Critic [18], and DDPG [19]. These approaches typically construct cross-trial experiences through neural networks or tabular methods by exploration and exploitation.

![](images/53d024bbe8b1a1717702a11e89545a40d64babdfe95188d4f4c4bf64df3f5945.jpg)

![](images/8196de12d0fa3faeeb8c64351c3c374bf576a33e86a59d614cb67f4dea00d7ac.jpg)  
Figure 2: Overview of the memory cycle effect and adaptive memory framework.

# 2.2 Large Language Model Based Agents

With the rapid development of LLMs, building agents based on LLMs has emerged as a promising field of research [1]. These LLM-based agents have recently found extensive applications in various domains, including finance [4], social simulation [20], and personal assistants [6]. For instance, Generative Agents [13] aim to simulate human daily activities with LLM-based agents. Although they commonly have different architectures [2], most of them incorporate reasoning [21], memory [7], and action modules. Similar to RL-based agents, they also interact with environments by receiving observations, making decisions, and taking actions. However, due to their extensive pretraining, LLMbased agents have more prior world knowledge, which enhances their generalization capabilities.

# 2.3 Memory of Large Language Model Based Agents

For LLM-based agents, memory is one of the most critical capabilities for interacting with environments, as it maintains contextual consistency and supports inferences made by LLMs [7]. Previous studies primarily employ in-context learning to implement memories [13, 10, 22]. They commonly utilize RAG methods to retrieve relevant information about the current states and incorporate it into prompts [8]. For example, MemoryBank [10] implements a hierarchical memory structure with textual summarization and forgetting mechanisms. MemTree [22] proposes a tree-structured memory framework that dynamically updates memory storage. However, they still require human experts to manually design for specific applications, and they overlook the memory cycle effect during interactions. These limitations result in increased labor costs and suboptimal performance.

# 3 Methods

# 3.1 Preliminary: Memory Cycle Effect

Before proposing our adaptive memory framework, we explicitly formulate the memory cycle, as illustrated in Figure 2(a). We model the continuous interactions between agents and environments as a Markov Decision Process (MDP) [15]. Specifically, we denote the state transition distribution of an environment as $p _ { \mathrm { e n v } } ( \cdot | s ^ { t } , a ^ { t } )$ , where $s ^ { t }$ and $a ^ { t }$ represent the state and action at step $t$ . Besides, we employ the reward function $r ( s ^ { t } , a ^ { t } )$ to reflect the achievement of the task. We denote the agent’s policy as $\pi _ { \mathrm { a g e n t } } ( \cdot | s ^ { t } , \theta )$ , where the next action is determined by the current state $s ^ { t }$ with parameter $\theta$ . For LLM-based agents, their policies are typically implemented with memory contexts to construct prompts for LLMs. During the interaction process, at each step $t$ , the agent perceives the current state $s ^ { t }$ , and selects an action by $a ^ { t } \sim \pi _ { \mathrm { a g e n t } } ( \cdot | \bar { s } ^ { t } , \theta )$ . Then, the state is updated by $s ^ { t + 1 } \sim p _ { \mathrm { e n v } } ( \cdot | s ^ { t } , a ^ { t } )$ , obtaining the reward $r ( s ^ { t } , a ^ { t } )$ . The objective of agents is to maximize the cumulative reward.

The memory cycle effect in agent-environment interactions can be further formulated into a framework with three consecutive procedures as demonstrated in Figure 2(b), including memory storage $S ( \theta _ { s } ; \cdot )$ , retrieval $R ( \theta _ { r } ; \cdot )$ , and utilization $U ( \theta _ { u } ; \cdot )$ . Here, we use $\theta = \{ \theta _ { s } , \theta _ { r } , \theta _ { u } \}$ to emphasize their model parameters. First, the agent observes the current state $s ^ { t }$ and stores it into the storage $M ^ { t }$ , where $\mathbf { \dot { \boldsymbol { M } } } ^ { t } = S ( \theta _ { s } ; M ^ { t - 1 } , s ^ { t } )$ . Then, the agent retrieves a ranked subset of the current storage by $M _ { \mathrm { r a n k } } ^ { t } =$ $R ( \theta _ { r } ; s ^ { t } , M ^ { t } )$ based on the current state $s ^ { t }$ . After that, the agent integrates this memory subset into a prompt through the utilization procedure by $p ^ { t } = U ( \theta _ { u } ; M _ { \mathrm { r a n k } } ^ { t } , s ^ { t } )$ . Finally, the agent determines

the next action by LLM with $a ^ { t } = \operatorname { L L M } ( p ^ { t } )$ , and updates the state $s ^ { t + 1 } \sim p _ { \mathrm { e n v } } ( \cdot | s ^ { t } , a ^ { t } )$ for the next cycle. In these cycles, the memory storage, retrieval, and utilization procedures are not isolated, but influence each other. Therefore, the optimization of these three procedures should be performed jointly. Our adaptive memory framework is proposed based on this framework of the memory cycle effect, which can be optimized in a data-driven manner. An overview of the procedures of our framework is provided in Figure 2(c), and we present the details in the rest of this section.

# 3.2 Memory Retrieval Procedure

Due to the large collection of memories, agents should retrieve a subset of memories before integrating them into prompts. Previous studies typically calculate matching scores $f ( s ^ { t } , m _ { i } )$ between the current state $s ^ { t }$ and memories $m _ { i } \in M ^ { t }$ , and select the top- $k$ memories. These matching scores are often associated with metrics, such as semantic similarity, time recency, and so on. For instance, Generative Agents [13] calculate the matching scores by f (st, mi) = αrel · drel(st, mi) + αimp · dimp(st, mi) + $\alpha _ { \mathrm { r e c } } \cdot d _ { \mathrm { r e c } } ( s ^ { t } , m _ { i } )$ , where $d _ { \mathrm { r e l } } ( \cdot ) , d _ { \mathrm { i m p } } ( \cdot ) , d _ { \mathrm { r e c } } ( \cdot )$ are metric functions and $\alpha _ { \mathrm { r e l } }$ , $\alpha _ { \mathrm { i m p } }$ , αrec are weights on semantic relevance, memory importance, and time recency. However, these weights are fixed and manually determined by human experts, instead of learning from interactions with the environment. Besides, in different applications, the significance of metric functions can also be various, and the sensitivities of states and memories on different metrics are often not the same.

To solve these challenges, we propose an optimizable retrieval procedure. We define a vectorvalued function $\mathbf { d } ( s ^ { t } , m _ { i } ) \ = \ [ d _ { 1 } ( s ^ { t } , m _ { i } ) , d _ { 2 } ( s ^ { t } , m _ { i } ) , . . . , d _ { n } ( s ^ { t } , m _ { i } ) ]$ , where $d _ { 1 } ( \cdot ) , d _ { 2 } ( \cdot ) , . . . , d _ { n } ( \cdot )$ are metric functions. Then, we design a parameterized MoE gate function to activate different metrics as $\mathbf { g } ( \theta _ { r } ; s ^ { t } , m _ { i } ) = [ g _ { 1 } ( \theta _ { r } ; s ^ { t } , m _ { i } ) , g _ { 2 } ( \theta _ { r } ; s ^ { t } , m _ { i } ) , . . . , g _ { n } \bar { ( } \theta _ { r } ; s ^ { t } , m _ { i } ) ]$ , where $\theta _ { r }$ represents optimizable parameters. This gate function can adaptively adjust weights on metric functions for different states and memories based on the training data. After that, we calculate the matching scores by according $f ( \theta _ { r } ; s ^ { t } , m _ { i } ) = \mathbf { g } ( \theta _ { r } ; s ^ { t } , m _ { i } ) \cdot \mathbf { d } ( s ^ { t } , m _ { i } ) ^ { T }$ . Finally, all the ked memory list $m _ { i } \in M ^ { t }$ nked. $M _ { \mathrm { r a n k } } ^ { t } = [ \tilde { m } _ { 1 } ^ { t } , \tilde { m } _ { 2 } ^ { t } , . . . , \tilde { m } _ { t } ^ { t } ]$

In addition, we extend metric functions to cover more retrieval policies besides semantic relevance. We incorporate emotional relevance by pre-training a scoring function to extract emotions from memories. We also pre-train a judging model to assess the importance of memories. Moreover, we further extend linear time recency using Taylor’s Formula to get dprec(st, mi) = || ∆(s ,mi)t | $\begin{array} { r } { d _ { \mathrm { r e c } } ^ { p } ( s ^ { t } , m _ { i } ) = | | \frac { \Delta ( s ^ { t } , m _ { i } ) } { t } | | _ { p } } \end{array}$ on various $p$ -norms, where $\Delta ( s ^ { t } , m _ { i } )$ is the time gap between $s ^ { t }$ and $m _ { i }$ . Due to the page limitation, more details can be found in Appendix A. Besides, we implement $\mathbf { g } ( \theta _ { r } ; s ^ { t } , m _ { i } )$ with

$$
\mathbf {g} \left(\theta_ {r}; s ^ {t}, m _ {i}\right) = \operatorname {s o f t m a x} \left(W _ {2} \cdot \sigma \left(W _ {1} \cdot \left[ \mathbf {h} _ {s ^ {t}}; \mathbf {h} _ {m _ {i}} \right] ^ {T} + b _ {1}\right) + b _ {2}\right),
$$

where $\theta _ { r } = \{ W _ { 1 } , W _ { 2 } , b _ { 1 } , b _ { 2 } \}$ are optimizable parameters, and $\mathbf { h } _ { s ^ { t } } , \mathbf { h } _ { m _ { i } }$ are embeddings of $s ^ { t } , m _ { i }$

# 3.3 Memory Utilization Procedure

After obtaining the retrieval result $M _ { \mathrm { r a n k } } ^ { t }$ , it is necessary to transform it into a memory context to serve as part of the prompt $p ^ { t }$ . Most previous studies directly concatenate their top- $k$ memories. However, this approach solely focuses on state-memory matches but overlooks memory-memory relations. It leads to the recurrence of similar memories within the same context. To solve this problem, we design a learnable memory augmentation process that can be optimized using training datasets.

We iteratively integrate the memories from t t $M _ { \mathrm { r a n k } } ^ { t }$ into the memory context. Starting with the initialt t t memory context $\bar { p _ { 0 } ^ { t } }$ , we obtain $p _ { i } ^ { t } = \operatorname { L L M } ( \dot { \theta } _ { u } ; p _ { i - 1 } ^ { t } , \tilde { m } _ { i } ^ { t } , s ^ { t } )$ for $i \geq 1$ until the end of process, where $\theta _ { u }$ represents the optimizable parameters in LLMs. To prevent excessive merging steps, we calculate the word increase rate from $p _ { i - 1 } ^ { t }$ to $p _ { i } ^ { t }$ as $\Delta l _ { i } ^ { t }$ , and approximate the information gain as $\begin{array} { r } { c _ { i } = \mathrm { c l i p } ( \frac { \Delta l _ { i } } { \Delta l _ { i - 1 } } , 0 , 1 ) } \end{array}$ . Then, we sample the stop signal with $z _ { i } \sim B \left( 1 - \operatorname* { m a x } ( c _ { i } , c _ { i - 1 } ) \right)$ to allow one exemption, where $B ( \cdot )$ denotes a Bernoulli distribution. Finally, we incorporate the last memory context into the template to get the prompt $p ^ { t }$ . However, common LLMs may exhibit suboptimal performance for specific applications. To address this issue, we adjust the parameters $\theta _ { u }$ of LLMs to align with training datasets through SFT and DPO, as described in Section 4.

# 3.4 Memory Storage Procedure

When an agent perceives a new state during interactions, it typically extracts critical parts from complete observations before storing them. For instance, an agent designed for personal assistance should

# Algorithm 1: Algorithm of on-policy optimization.

Input: The number of total epochs $L$ , the trajectory size $n$ , the learning rates $\alpha _ { r } , \alpha _ { u } ^ { s } , \alpha _ { u } ^ { d }$ , and the initial parameters $\theta _ { s } ^ { 0 } , \dot { \theta } _ { r } ^ { 0 } , \theta _ { u } ^ { 0 }$ .

Output: The optimized parameters $\theta _ { s } ^ { * } , \theta _ { r } ^ { * } , \theta _ { u } ^ { * }$

# 1 for $l \gets 1$ to $L$

2 Sample trajectories $T _ { 1 } , T _ { 2 } , . . . , T _ { n }$ by interacting with the training environment.

$$
\theta_ {r} ^ {l} \leftarrow \theta_ {r} ^ {l - 1} - \alpha_ {r} \cdot \nabla \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {1}{t _ {i}} \sum_ {j = 1} ^ {t _ {i}} w _ {i, j} \ln \frac {\sigma \left[ f \left(\theta_ {r} ^ {l - 1} ; s _ {i} ^ {t _ {i}} , \tilde {m} _ {i , t _ {i} - j + 1} ^ {t _ {i}}\right) - f \left(\theta_ {r} ^ {l - 1} ; s _ {i} ^ {t _ {i}} , \tilde {m} _ {i , j} ^ {t _ {i}}\right) \right]}{\sigma \left[ f \left(\theta_ {r} ^ {l - 1} ; s _ {i} ^ {t _ {i}} , \tilde {m} _ {i , j} ^ {t _ {i}}\right) - f \left(\theta_ {r} ^ {l - 1} ; s _ {i} ^ {t _ {i}} , \tilde {m} _ {i , t _ {i} - j + 1} ^ {t _ {i}}\right) \right]}.
$$

$$
\left. \begin{array}{c c} 4 & \theta_ {u} ^ {l} \leftarrow \theta_ {u} ^ {l - 1} - \alpha_ {u} ^ {s} \cdot \nabla \frac {1}{n} \sum_ {i = 1} ^ {n} \mathrm {C E L o s s} \left(\theta_ {u} ^ {l - 1}; \tilde {p} _ {t _ {i}} ^ {t _ {i}} \mid p _ {t _ {i} - 1} ^ {t _ {i}}, \tilde {m} _ {t _ {i}} ^ {t _ {i}}, s _ {i} ^ {t _ {i}}\right). \end{array} \right.
$$

$$
\left.\begin{array}{c c}5&\theta_ {u} ^ {l} \leftarrow \theta_ {u} ^ {l} - \alpha_ {u} ^ {d} \cdot \nabla \frac {1}{n} \sum_ {i = 1} ^ {n} \ln \sigma \left[ \right. \beta \ln \frac {P \left(\theta_ {u} ^ {l} ; p _ {t _ {i}} ^ {t _ {i}} \mid p _ {t _ {i} - 1} ^ {t _ {i}} , \tilde {m} _ {i , t _ {i}} ^ {t _ {i}} , s _ {i} ^ {t _ {i}}\right)}{P \left(\theta_ {u} ^ {l - 1} ; p _ {t _ {i}} ^ {t _ {i}} \mid p _ {t _ {i} - 1} ^ {t _ {i}} , \tilde {m} _ {i , t _ {i}} ^ {t _ {i}} , s _ {i} ^ {t _ {i}}\right)} - \beta \ln \frac {P \left(\theta_ {u} ^ {l} ; p _ {t _ {i}} ^ {t _ {i}} \mid p _ {t _ {i} - 1} ^ {t _ {i}} , \tilde {m} _ {i , t _ {i}} ^ {t _ {i}} , s _ {i} ^ {t _ {i}}\right)}{P \big (\theta_ {u} ^ {l - 1} ; p _ {t _ {i}} ^ {t _ {i}} \mid p _ {t _ {i} - 1} ^ {t _ {i}} , \tilde {m} _ {i , t _ {i}} ^ {t _ {i}} , s _ {i} ^ {t _ {i}} \big)}.\end{array}\right]
$$

$$
\theta_ {u} ^ {l} \leftarrow \theta_ {u} ^ {l - 1} \bigcup_ {i = 1} ^ {n} \operatorname {L L M} \left(\left\{s _ {i} ^ {t _ {i}}, m _ {i} ^ {t _ {i}} \right\} \in T _ {i} ^ {\text {p o s}}\right) \cup \operatorname {L L M} \left(\left\{s _ {i} ^ {t _ {i}}, m _ {i} ^ {t _ {i}} \right\} \in T _ {i} ^ {\text {n e g}}\right).
$$

# 7 end

8 Obtain optimized parameters $\theta _ { s } ^ { * } = \theta _ { s } ^ { L } , \theta _ { s } ^ { r } * = \theta _ { r } ^ { L } , \theta _ { u } ^ { * } = \theta _ { u } ^ { L }$   
9 return $\theta _ { s } ^ { * } , \theta _ { r } ^ { * } , \theta _ { u } ^ { * }$

concentrate on factual daily information from observations [23], while an emotional companion agent should prioritize sentiments [10]. This extraction process can be implemented by LLM to highlight critical aspects in instructions. However, different applications inherently emphasize distinct aspects.

To solve this problem, we design an extraction process based on task-specific reflection [24]. Specifically, we structure an instruction with a general part $\scriptstyle p _ { \mathrm { g l o b } }$ and a task-specific part $p _ { \mathrm { t a s k } }$ . Then, we consider $p _ { \mathrm { t a s k } }$ as the learnable parameter $\theta _ { s }$ , and optimize it based on successful and unsuccessful trajectories from training datasets, as discussed in the next section. For each interaction, we transform an observation into a memory unit with $m _ { t } = \mathrm { L L M } ( p _ { \mathrm { g l o b } } , p _ { \mathrm { t a s k } } , s ^ { t } )$ , and update the storage with $M ^ { t } = M ^ { t - 1 } \cup \{ m _ { t } \}$ . To balance the extraction load, we set a cache to temporarily hold observations, and transfer them into storage whenever recalling memories or reaching the cache capacity.

# 4 Optimization Strategies

# 4.1 Overview: Optimization of LLM-based Agents

Unlike LLM optimization, which is based on a static corpus, optimizing LLM-based agents relies on interactions with dynamic environments. Therefore, we propose two strategies to optimize our memory framework. The first strategy is off-policy optimization, which samples trajectories $\mathcal { D }$ from training environments using the reference policy $\pi _ { \mathrm { a g e n t } } ( \cdot | s ^ { t } , \theta ^ { \mathrm { r e f } } )$ , and optimizes another policy $\pi _ { \mathrm { a g e n t } } ( \cdot | s ^ { t } , \theta )$ with the loss function $\mathcal { L } ( \cdot )$ . It supports offline training and the reuse of previous trajectories, making it more flexible and efficient. However, it encounters the issue of distribution shift between the sampling policy $\pi _ { \mathrm { a g e n t } } ( \cdot | s ^ { t } , \theta ^ { \mathrm { r e f } } )$ and the optimized policy $\pi _ { \mathrm { a g e n t } } ( \cdot | s ^ { t } , \theta )$ . Another strategy is on-policy optimization, which consistently employs the optimized policy $\dot { \pi } _ { \mathrm { a g e n t } } ( \cdot | s ^ { t } , \theta )$ to sample training trajectories for ongoing optimization. This approach requires online learning to keep alignment between the sampling and optimized policies, thereby alleviating distribution shifts.

# 4.2 Off-policy Optimization

Memory Retrieval Optimization. We propose a contrastive learning approach to optimize parameters $\theta _ { r }$ of the MoE gate function $\mathbf { g } ( \theta _ { r } ; s ^ { t } , m _ { i } )$ in the retrieval procedure. First of all, we filter out all the successful interactions $\mathcal { D } _ { s }$ whose final rewards exceed the threshold $\beta _ { r }$ (e.g., answer correctly for QAs). Then, we focus on the ranking result $M _ { \mathrm { r a n k } } ^ { t } = [ \tilde { m } _ { 1 } ^ { t } , \tilde { m } _ { 2 } ^ { t } , . . . , \tilde { m } _ { t } ^ { t } ]$ from their memory retrieval procedures. We pair all the elements $\tilde { m } _ { i } ^ { t } \in M _ { \mathrm { r a n k } } ^ { t }$ in reverse order as $x _ { i } = ( \tilde { m } _ { i } ^ { t } , \tilde { m } _ { t - i + 1 } ^ { t } )$ , where $1 \leq i \leq t$ . Then, we assign a weight $\begin{array} { r } { w _ { i } = \frac { - \mathrm { s i g n } \left( v _ { i } \right) } { \sum _ { j = 1 } ^ { t } \gamma _ { \it \mathrm { ~ i ~ } } ^ { v _ { i } } } \cdot \gamma ^ { v _ { i } } } \end{array}$ = P−sign(vi)tj=1 γvi · γvi with vi = t − 1 − |t − 2j + 1|, which $v _ { i } = t - 1 - \left| t - 2 j + 1 \right|$ allocates higher contrastive confidence to the pairs with larger ranking differences, thereby reducing the ranking noise. Finally, we define our loss function as

$$
\mathcal {L} (\theta_ {r}; \mathcal {D} _ {s}) = \frac {1}{| \mathcal {D} _ {s} |} \sum_ {s ^ {t}, M _ {\mathrm {r a n k}} ^ {t} \in \mathcal {D} _ {s}} \frac {1}{t} \sum_ {i = 1} ^ {t} w _ {i} \cdot \ln \frac {\sigma [ f (\theta_ {r} ; s ^ {t} , \tilde {m} _ {t - i + 1} ^ {t}) - f (\theta_ {r} ; s ^ {t} , \tilde {m} _ {t} ^ {t}) ]}{\sigma [ f (\theta_ {r} ; s ^ {t} , \tilde {m} _ {i} ^ {t}) - f (\theta_ {r} ; s ^ {t} , \tilde {m} _ {t - i + 1} ^ {t}) ]},
$$

and optimize the parameters with $\begin{array} { r } { \theta _ { r } ^ { * } = \arg \operatorname* { m i n } _ { \theta _ { r } } \mathcal { L } ( \theta _ { r } ; \mathcal { D } ) } \end{array}$ by gradient descent.

Memory Utilization Optimization. We employ SFT and DPO to optimize the parameters $\theta _ { u }$ of LLMs for the aggregation process in memory utilization. First of all, we choose the final interactions of training trajectories and denote them as $\mathcal { D } _ { l }$ . Then, we focus on their memory utilization procedures. Specifically, we optimize $\theta _ { u }$ in $p _ { t } ^ { t } = \mathrm { L L M } ( \theta _ { u } ; p _ { t - 1 } ^ { t } , \tilde { m } _ { t } ^ { t } , s ^ { t } )$ by collecting the outputs from expert models in $\tilde { p } _ { t } ^ { t } = E ( p _ { t - 1 } ^ { t } , \tilde { m } _ { t } , s ^ { t } )$ , where the expert model $E ( \cdot )$ can be implemented by domain-specific or more advanced LLMs. Finally, the SFT loss function can be expressed as

$$
\mathcal{L}^{\mathrm{SFT}}(\theta_{u};\mathcal{D}_{l}) = \frac{1}{|\mathcal{D}_{l}|}\sum_{\substack{p_{t - 1}^{t},\tilde{m}_{t}^{t},s^{t}\in \mathcal{D}_{l}}} \mathrm{CELoss}(\theta_{u};\tilde{p}_{t}^{t}|p_{t - 1}^{t},\tilde{m}_{t}^{t},s^{t}),
$$

where CELoss(·) is the cross-entropy loss function. Then, we have $\begin{array} { r } { \theta _ { u } ^ { \mathrm { S F T } } = \arg \operatorname* { m i n } _ { \theta _ { u } } \mathcal { L } ^ { \mathrm { S F T } } ( \theta _ { u } ; \mathcal { D } _ { l } ) } \end{array}$ . We further refine the expert model using DPO to better align with the expert model. Specifically, we consider $\mathrm { L L M } ( \theta _ { u } ^ { \mathrm { S F T } } ; \cdot )$ as the reference model, and re-generate utilization results with $\hat { p } _ { t } ^ { t } =$ $\mathrm { L L M } ( \theta _ { u } ; \hat { p } _ { t - 1 } ^ { t } , \tilde { m } _ { t } ^ { t } , s ^ { \bar { t } } )$ . Then, we establish the DPO loss function

$$
\mathcal {L} ^ {\mathrm {D P O}} (\theta_ {u}; \mathcal {D} _ {l}) = \frac {1}{| \mathcal {D} _ {l} |} \sum_ {{p _ {t - 1} ^ {t}, \tilde {m} _ {t} ^ {t}, s ^ {t} \in \mathcal {D} _ {l}}} \ln \sigma \left[ \beta \ln \frac {P (\theta_ {u} ; \hat {p} _ {t} ^ {t} | p _ {t - 1} ^ {t} , \tilde {m} _ {t} ^ {t} , s ^ {t})}{P (\theta_ {u} ^ {S F T} ; \hat {p} _ {t} ^ {t} | p _ {t - 1} ^ {t} , \tilde {m} _ {t} ^ {t} , s ^ {t})} - \beta \ln \frac {P (\theta_ {u} ; p _ {t} ^ {t} | p _ {t - 1} ^ {t} , \tilde {m} _ {t} ^ {t} , s ^ {t})}{P (\theta_ {u} ^ {S F T} ; p _ {t} ^ {t} | p _ {t - 1} ^ {t} , \tilde {m} _ {t} ^ {t} , s ^ {t})} \right],
$$

where $\beta$ is a parameter to control the deviation from the original parameter $\theta _ { u } ^ { \mathrm { S F T } }$ , and $P ( \cdot )$ is the output probability distribution of the LLMs given certain parameters. Finally, we obtain the optimal parameters by $\begin{array} { r } { \theta _ { u } ^ { * } = \arg \operatorname* { m i n } _ { \theta _ { u } } \mathcal { L } ^ { \mathrm { D P O } } ( \theta _ { u } ; \mathcal { D } _ { l } ) } \end{array}$ , where $\theta _ { u }$ is initialized as $\theta _ { u } ^ { \mathrm { S F T } }$ .

Memory Storage Optimization. To optimize the task-specific instruction for memory extraction, we optimize $\theta _ { s }$ by self-reflection. Specifically, we divide all the interactions into two groups based on their rewards. The interactions with rewards above the threshold $\beta _ { s }$ are placed in the positive group ${ \mathcal { D } } _ { \mathrm { p o s } }$ , while interactions with rewards below the threshold are assigned to the negative group $\mathcal { D } _ { \mathrm { n e g } }$ . For interactions in the positive group, we utilize LLMs to reflect and summarize their successful experiences. Similarly, the failure experiences can also be reflected and summarized by LLMs automatically. After that, we iteratively update the task-specific prompt with $p _ { \mathrm { t a s k } } $ $\dot { p _ { \mathrm { t a s k } } } \cup \mathbf { L } \mathbf { L } \mathbf { M } ( \{ s ^ { t } , m ^ { t } \} \stackrel { . } { \in } \mathcal { D } _ { \mathrm { p o s } } ) \cup \mathbf { L } \mathbf { L } \mathbf { M } ( \{ s ^ { t } , m ^ { t } \} \in \mathcal { D } _ { \mathrm { n e g } } )$ , where we have $\theta _ { s } ^ { * } = p _ { \mathrm { t a s k } } ^ { * }$ .

# 4.3 On-policy Optimization

Although off-policy optimization supports offline training, it is often hindered by distribution shifts between the sampling policy and the optimized policy, leading to suboptimal performance in memory cycles. To alleviate this problem, we extend our optimization strategy to on-policy optimization, as described in Algorithm 1. Building upon the model parameters after off-policy optimization, we further conduct the on-policy optimization with online learning. Specifically, during each epoch, we sample $n$ trajectories by interacting with the training environment based on the current model parameters. Then, we utilize the training procedures above with single-step optimization to update model parameters. Finally, we obtain the optimal model parameters from the last epoch.

# 5 Experiments

# 5.1 Experimental Settings

Our experiments are conducted on three datasets with various difficulty levels, including HotpotQAhard, HotpotQA-medium, and HotpotQA-easy [25]. Additionally, we also carry out experiments on MemDaily [23], and put the details in Appendix B due to the page limitation. To fulfill interactive scenarios between the agent and the environment, we adopt fullwiki mode in HotpotQA by implementing a simulator to create a dynamic environment, which presents greater challenges than distractor mode with static references. For the LLM-based agents, we employ the ReAct [26] reasoning structure along with textual memory contexts [7]. Our memory framework is compared against several baselines of memory models implemented by MemEngine [27] as follows:

• FUMemory (Full Memory): Directly concatenate all the observations into a memory context.   
• LTMemory (Long-term Memory): Retrieve the most relevant observations by semantic similarities.   
• STMemory (Short-term Memory): Keep the latest observations to combine as a memory context.   
• GAMemory (Generative Agents [13]): Memory with self-reflection and weighted retrieval.   
• MBMemory (MBMemory [10]): Hierarchical memory with summarization and forgetting.

Table 1: Overall performance across different baselines and inference models on various datasets. Bold values represent the best results, while underlined values represent the second-best results.   

<table><tr><td colspan="7">HotpotQA-Hard</td></tr><tr><td>Inference</td><td>ActOnly</td><td>CoTOnly</td><td>FUMemory</td><td>LTMemory</td><td>STMemory</td><td>GAMemory</td></tr><tr><td>GPT-4o-mini</td><td>0.2832</td><td>0.3274</td><td>0.3451</td><td>0.3274</td><td>0.3540</td><td>0.3186</td></tr><tr><td>Qwen-2.5</td><td>0.1504</td><td>0.2389</td><td>0.2920</td><td>0.2212</td><td>0.1504</td><td>0.2124</td></tr><tr><td>Llama-3.1</td><td>0.1770</td><td>0.2566</td><td>0.1239</td><td>0.0619</td><td>0.0177</td><td>0.0354</td></tr><tr><td>Inference</td><td>MBMemory</td><td>SCMemory</td><td>MTMemory</td><td>Ours-def</td><td>Ours-off</td><td>Ours-on</td></tr><tr><td>GPT-4o-mini</td><td>0.3009</td><td>0.3363</td><td>0.3628</td><td>0.3274</td><td>0.3186</td><td>0.3274</td></tr><tr><td>Qwen-2.5</td><td>0.2301</td><td>0.1416</td><td>0.2566</td><td>0.2832</td><td>0.2832</td><td>0.3186</td></tr><tr><td>Llama-3.1</td><td>0.1062</td><td>0.0619</td><td>0.1504</td><td>0.2478</td><td>0.1416</td><td>0.2920</td></tr><tr><td colspan="7">HotpotQA-Medium</td></tr><tr><td>Inference</td><td>ActOnly</td><td>CoTOnly</td><td>FUMemory</td><td>LTMemory</td><td>STMemory</td><td>GAMemory</td></tr><tr><td>GPT-4o-mini</td><td>0.3303</td><td>0.4220</td><td>0.4862</td><td>0.4037</td><td>0.3945</td><td>0.3853</td></tr><tr><td>Qwen-2.5</td><td>0.2202</td><td>0.2844</td><td>0.2844</td><td>0.2385</td><td>0.1651</td><td>0.1468</td></tr><tr><td>Llama-3.1</td><td>0.1560</td><td>0.2294</td><td>0.1284</td><td>0.0642</td><td>0.0275</td><td>0.0642</td></tr><tr><td>Inference</td><td>MBMemory</td><td>SCMemory</td><td>MTMemory</td><td>Ours-def</td><td>Ours-off</td><td>Ours-on</td></tr><tr><td>GPT-4o-mini</td><td>0.3853</td><td>0.3486</td><td>0.3853</td><td>0.4220</td><td>0.4037</td><td>0.4404</td></tr><tr><td>Qwen-2.5</td><td>0.2385</td><td>0.1009</td><td>0.2752</td><td>0.3119</td><td>0.3486</td><td>0.4037</td></tr><tr><td>Llama-3.1</td><td>0.0642</td><td>0.0826</td><td>0.1743</td><td>0.2752</td><td>0.1468</td><td>0.3119</td></tr><tr><td colspan="7">HotpotQA-Easy</td></tr><tr><td>Inference</td><td>ActOnly</td><td>CoTOnly</td><td>FUMemory</td><td>LTMemory</td><td>STMemory</td><td>GAMemory</td></tr><tr><td>GPT-4o-mini</td><td>0.3738</td><td>0.4019</td><td>0.3645</td><td>0.3832</td><td>0.3832</td><td>0.3738</td></tr><tr><td>Qwen-2.5</td><td>0.2991</td><td>0.3364</td><td>0.2710</td><td>0.2523</td><td>0.2056</td><td>0.1776</td></tr><tr><td>Llama-3.1</td><td>0.2991</td><td>0.3271</td><td>0.1589</td><td>0.0654</td><td>0.0374</td><td>0.0935</td></tr><tr><td>Inference</td><td>MBMemory</td><td>SCMemory</td><td>MTMemory</td><td>Ours-def</td><td>Ours-off</td><td>Ours-on</td></tr><tr><td>GPT-4o-mini</td><td>0.3364</td><td>0.3645</td><td>0.3271</td><td>0.3832</td><td>0.3738</td><td>0.3738</td></tr><tr><td>Qwen-2.5</td><td>0.2523</td><td>0.2056</td><td>0.3364</td><td>0.3925</td><td>0.3364</td><td>0.4112</td></tr><tr><td>Llama-3.1</td><td>0.1028</td><td>0.0748</td><td>0.1495</td><td>0.2523</td><td>0.1682</td><td>0.3271</td></tr></table>

• SCMemory (SCM [28]): Self-controlled memory with adaptive length of memory context.   
• MTMemory (MemTree [22]): Structured-based memory with node representation and update.

Besides, we implement two one-step baselines without memory for comparison as follows:

• ActOnly: Take actions based on current observations without memory or reasoning structure.   
• CoTOnly: Reason on current observations by Chain-of-Thought [29] to take actions.

We represent our models as Ours-def, Ours-off, and Ours-on, corresponding to the non-optimized model, the off-policy optimized model, and the on-policy optimized model, respectively. Following the previous work [25], we calculate the accuracy of Exact Match (EM) between the ground-truth and predicted answer, serving as the final reward of each trajectory. Specifically, agents are required to answer a question in each independent trajectory. Within a maximum number of steps, they can either search for a keyword on Wikipedia in each step to get its full document, or submit a predicted answer to finish this trajectory. Due to the page limitation, we provide additional details regarding experimental settings in Appendix D to facilitate the reproduction of our experiments.

# 5.2 Overall Performance

The results of major performances are present in Table 1. From the results, we find that our model with on-policy optimization outperforms other baselines in most cases, showing the effectiveness of our proposed framework. The results also reveal that our framework can still work with default parameters, showing a certain degree of multitask generalization, but its performance declines after off-policy optimization due to the distribution mismatch. Moreover, MTMemory and MBMemory also present relatively great performance, whereas the one-step baselines demonstrate weaknesses in more challenging tasks. Besides, it appears that some LLMs exhibit limited dependence on memory for easy-level questions, possibly because they have encountered the necessary references to these questions in their pre-training corpus. Finally, the performance of memory methods varies across different inference models, potentially due to disparities in their in-context learning abilities to

Table 2: Results of ablation studies across different baselines and inference models on various datasets. Bold values represent the best results, while underlined values represent the second-best results.   

<table><tr><td colspan="8">HotpotQA-Hard</td></tr><tr><td>Inference</td><td>Ours-def</td><td>Ours-R</td><td>Ours-U/sft</td><td>Ours-U/dpo</td><td>Ours-S</td><td>Ours-off</td><td>Ours-on</td></tr><tr><td>GPT-4o-mini</td><td>0.3274</td><td>0.3451</td><td>0.3097</td><td>0.3186</td><td>0.2920</td><td>0.3186</td><td>0.3274</td></tr><tr><td>Qwen-2.5</td><td>0.2832</td><td>0.3186</td><td>0.3009</td><td>0.3186</td><td>0.2832</td><td>0.2832</td><td>0.3186</td></tr><tr><td>Llama-3.1</td><td>0.2478</td><td>0.2301</td><td>0.2478</td><td>0.1593</td><td>0.2566</td><td>0.1416</td><td>0.2920</td></tr><tr><td colspan="8">HotpotQA-Medium</td></tr><tr><td>Inference</td><td>Ours-def</td><td>Ours-R</td><td>Ours-U/sft</td><td>Ours-U/dpo</td><td>Ours-S</td><td>Ours-off</td><td>Ours-on</td></tr><tr><td>GPT-4o-mini</td><td>0.4220</td><td>0.4587</td><td>0.4220</td><td>0.4312</td><td>0.4495</td><td>0.4037</td><td>0.4404</td></tr><tr><td>Qwen-2.5</td><td>0.3119</td><td>0.3303</td><td>0.3211</td><td>0.2661</td><td>0.3853</td><td>0.3486</td><td>0.4037</td></tr><tr><td>Llama-3.1</td><td>0.2752</td><td>0.2385</td><td>0.2385</td><td>0.0917</td><td>0.2844</td><td>0.1468</td><td>0.3119</td></tr><tr><td colspan="8">HotpotQA-Easy</td></tr><tr><td>Inference</td><td>Ours-def</td><td>Ours-R</td><td>Ours-U/sft</td><td>Ours-U/dpo</td><td>Ours-S</td><td>Ours-off</td><td>Ours-on</td></tr><tr><td>GPT-4o-mini</td><td>0.3832</td><td>0.3925</td><td>0.3645</td><td>0.3551</td><td>0.3645</td><td>0.3738</td><td>0.3738</td></tr><tr><td>Qwen-2.5</td><td>0.3925</td><td>0.4112</td><td>0.3271</td><td>0.3458</td><td>0.3645</td><td>0.3364</td><td>0.4112</td></tr><tr><td>Llama-3.1</td><td>0.2523</td><td>0.2710</td><td>0.2243</td><td>0.1589</td><td>0.2617</td><td>0.1682</td><td>0.3271</td></tr></table>

leverage memory contexts. Some methods show weak performance on open-source inference models, which may be attributed to the failure to organize effective memory contexts.

# 5.3 Ablation Studies

In order to further study each procedure and optimization strategy within our framework, we conduct ablation experiments by independently optimizing retrieval, utilization (SFT/DPO), and storage procedures with off-policy optimization. We denote these ablation models as Ours-R, Ours-U/sft, Ours-U/dpo, and Ours-S, respectively. The results, presented in Table 2, indicate that on-policy optimization is crucial for improving the performance of our framework. Besides, optimizing individual memory procedures can also take effect, especially for the retrieval procedure. However, directly combining the parameters of memory procedures results in reduced performance. Intuitively, the memory procedures have mutual influence due to the memory cycle effect, but the off-policy samples fail to trace the optimized memory outcomes. Therefore, the optimal parameters of a certain procedure are based on the initial parameters of other procedures, leading to a policy mismatch.

# 5.4 Extensive Studies on Reasoning Steps

To further study the effectiveness of memory methods inside trajectories, we calculate the average reasoning steps across different baselines under HotpotQA-hard and Qwen2.5, and we present the results in Figure 3. The results indicate that our approach significantly reduces the average reasoning steps within trajectories. Specifically, the proportion of five-step reasoning in our model decreases, while the occurrence of two-step reasoning trajectories increases. Under identical conditions, achieving tasks with fewer reasoning steps suggests that agents can make more informed decisions utilizing memory, thereby finding answers more quickly and confidently. Meanwhile, we observe that models with lower overall performance generally require more inference steps. This might be due to their inability to solve the problem even after reaching the maximum number of steps.

# 5.5 Extensive Studies on Pre-trained Metric Functions

We further conduct experiments to verify the effectiveness of pre-trained metric functions in the memory retrieval procedure. Specifically, we calculate NDCG@5 for the ranked messages based on importance scoring and use MSE to evaluate emotion scoring across different dimensions. Additionally, we record the instruction failure rate (IFR) for prompting methods. Due to the page limitation, the detailed results are presented in Appendix A.3. The results indicate that our pre-trained metric functions show certain improvements over zero-shot and few-shot prompting methods in predicting importance scores and emotion decomposition. Moreover, we observe that some open-source models exhibit instruction failures during the scoring process, leading to instability in retrieval metrics.

Table 3: Results of time costs (seconds) across different baselines.   

<table><tr><td>Efficiency</td><td>ActOnly</td><td>CoTOnly</td><td>FUMemory</td><td>LTMemory</td><td>STMemory</td><td>GAMemory</td></tr><tr><td>Time/Step</td><td>0.08</td><td>2.80</td><td>11.33</td><td>9.68</td><td>11.64</td><td>8.83</td></tr><tr><td>Time/Trajectory</td><td>0.08</td><td>2.80</td><td>43.05</td><td>32.91</td><td>54.73</td><td>39.72</td></tr><tr><td>Efficiency</td><td>MBMemory</td><td>SCMemory</td><td>MTMemory</td><td>Ours-def</td><td>Ours-off</td><td>Ours-on</td></tr><tr><td>Time/Step</td><td>8.38</td><td>7.14</td><td>107.34</td><td>14.98</td><td>13.03</td><td>11.74</td></tr><tr><td>Time/Trajectory</td><td>35.21</td><td>29.99</td><td>472.31</td><td>40.45</td><td>33.88</td><td>25.83</td></tr></table>

![](images/2b7a41495b36132156ff01e13b02070f1571701dd7dedac07635bb7f8d5b4757.jpg)  
Figure 3: Results of average reasoning steps across different baselines.

# 5.6 Influence of Hyper-parameters

We further explore the influence of some significant hyper-parameters in our framework, including SFT batch size, DPO batch size, and reflection batch size. Due to the page limitation, we include more details and the results in Appendix C. According to the results, we find that the best choice of SFT batch size is around 16, while the best DPO batch size is roughly 32. Additionally, we find that the accuracy is more sensitive to variations in DPO batch size, significantly diminishing when the values are particularly low. In contrast, the reflection batch size has a relatively minor impact on performance, with accuracy remaining similar when it ranges from 20 to 50 samples per reflection.

# 5.7 Analysis of Efficiency

In addition to evaluating the effectiveness of memory mechanisms, we conduct experiments to assess their efficiency. Specifically, we calculate the average time cost of different baselines for each step and each trajectory. Our experiments are performed on a computing server with 8 NVIDIA A800-SXM-80G GPUs, and the results are presented in Table 3. According to the results, while our method shows a slight increase in time per step due to additional operations, the time per trajectory is significantly reduced because the total number of reasoning steps decreases. Additionally, we observe that FUMemory, LTMemory, and STMemory exhibit higher time consumption per step, possibly due to increased inference costs associated with longer memory contexts in prompts.

# 6 Conclusion

In conclusion, we propose an adaptive and data-driven memory framework to optimize LLM-based agents. We formulate the memory cycle with retrieval, utilization, and storage procedures. We develop an MoE gate function to enhance the memory retrieval procedure, a task-specific reflection process to refine the memory extraction, and a post-training stage to improve the memory utilization procedure. Additionally, we design both off-policy and on-policy optimization strategies based on the memory cycle effect. The extensive experiments have verified the effectiveness and efficiency of our methods. In future works, we will focus on the optimization of parametric memory.

# Limitations and Ethical Statements

Despite the advancements achieved in our study, there are still some limitations in our work. First, our method focuses on explicit memory using RAG pipelines, and we primarily utilize CoT as the reasoning structure of agents. We will study implicit memory and other reasoning structures in future

work. Additionally, the questions in HotpotQA might pose a leakage risk in the pre-training corpus for LLMs. Our method can improve the memory capability of LLM-based agents, thereby better serving humans in social life. However, we should recognize the risks of memory injection during optimization. In addition, it is important to distinguish memory hallucination for usage.

# References

[1] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.   
[2] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. The rise and potential of large language model based agents: A survey. Science China Information Sciences, 68(2):121101, 2025.   
[3] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680, 2024.   
[4] Han Ding, Yinheng Li, Junhao Wang, and Hang Chen. Large language model agent in financial trading: A survey. arXiv preprint arXiv:2408.06361, 2024.   
[5] Yu Zhang, Shutong Qiao, Jiaqi Zhang, Tzu-Heng Lin, Chen Gao, and Yong Li. A survey of large language model empowered agents for recommendation and search: Towards next-generation information retrieval. arXiv preprint arXiv:2503.05659, 2025.   
[6] Yuanchun Li, Hao Wen, Weijun Wang, Xiangyu Li, Yizhen Yuan, Guohong Liu, Jiacheng Liu, Wenxing Xu, Xiang Wang, Yi Sun, et al. Personal llm agents: Insights and survey about the capability, efficiency and security. arXiv preprint arXiv:2401.05459, 2024.   
[7] Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model based agents. arXiv preprint arXiv:2404.13501, 2024.   
[8] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2, 2023.   
[9] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022.   
[10] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19724–19731, 2024.   
[11] Charles Packer, Vivian Fang, Shishir_G Patil, Kevin Lin, Sarah Wooders, and Joseph_E Gonzalez. Memgpt: Towards llms as operating systems. 2023.   
[12] Hongkang Yang, Zehao Lin, Wenjin Wang, Hao Wu, Zhiyu Li, Bo Tang, Wenqiang Wei, Jinbo Wang, Zeyun Tang, Shichao Song, et al. Memory3: Language modeling with explicit memory. arXiv preprint arXiv:2407.01178, 2024.   
[13] Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.   
[14] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.   
[15] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning: An introduction, volume 1. MIT press Cambridge, 1998.   
[16] Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.

[17] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602, 2013.   
[18] Vijay Konda and John Tsitsiklis. Actor-critic algorithms. Advances in neural information processing systems, 12, 1999.   
[19] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.   
[20] Chen Gao, Xiaochong Lan, Nian Li, Yuan Yuan, Jingtao Ding, Zhilun Zhou, Fengli Xu, and Yong Li. Large language models empowered agent-based modeling and simulation: A survey and perspectives. Humanities and Social Sciences Communications, 11(1):1–24, 2024.   
[21] Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, and Enhong Chen. Understanding the planning of llm agents: A survey. arXiv preprint arXiv:2402.02716, 2024.   
[22] Alireza Rezazadeh, Zichao Li, Wei Wei, and Yujia Bao. From isolated conversations to hierarchical schemas: Dynamic tree memory representation for llms. arXiv preprint arXiv:2410.14052, 2024.   
[23] Zeyu Zhang, Quanyu Dai, Luyu Chen, Zeren Jiang, Rui Li, Jieming Zhu, Xu Chen, Yi Xie, Zhenhua Dong, and Ji-Rong Wen. Memsim: A bayesian simulator for evaluating memory of llm-based personal assistants. arXiv preprint arXiv:2409.20163, 2024.   
[24] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.   
[25] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.   
[26] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.   
[27] Zeyu Zhang, Quanyu Dai, Xu Chen, Rui Li, Zhongyang Li, and Zhenhua Dong. Memengine: A unified and modular library for developing advanced memory of llm-based agents. arXiv preprint arXiv:2505.02099, 2025.   
[28] Bing Wang, Xinnian Liang, Jian Yang, Hui Huang, Shuangzhi Wu, Peihao Wu, Lu Lu, Zejun Ma, and Zhoujun Li. Enhancing large language model with self-controlled memory framework. arXiv preprint arXiv:2304.13343, 2023.   
[29] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
[30] Le Huang, Hengzhi Lan, Zijun Sun, Chuan Shi, and Ting Bai. Emotional rag: Enhancing role-playing agents through emotional retrieval. arXiv preprint arXiv:2410.23041, 2024.

# A Pre-trained Metric Functions

# A.1 Emotion Scoring Function

In addition to considering the semantic similarity between current states and memories, we propose incorporating emotional similarity as a factor for calculating matching scores. For each message, we represent its emotional content with eight dimensions: joy, acceptance, fear, surprise, sadness, disgust, anger, and anticipation [30]. This allows us to extract the emotion $\mathbf { h } _ { e } ( \phi _ { e } ; x )$ from message $x$ . While most previous studies use LLMs for emotion scoring, this approach is hampered by randomness and a lack of comparative analysis of emotions across different messages. It also suffers increased computational expense due to frequent LLM inferences. To mitigate these challenges, we propose a pre-trained emotion scoring function leveraging contrastive learning. Specifically, we have

$$
\mathbf {h} _ {e} (\phi_ {e}; x) = W _ {2} ^ {e} \cdot \tanh (W _ {1} ^ {e} \cdot \mathbf {h} _ {x} ^ {T} + b _ {1} ^ {e}) + b _ {2} ^ {e},
$$

where $\mathbf { h } _ { x }$ is the text embedding of $x$ and $\phi _ { e } = \{ W _ { 1 } ^ { e } , W _ { 2 } ^ { e } , b _ { 1 } ^ { e } , b _ { 2 } ^ { e } \}$ are trainable parameters. Then, the emotional similarity can be calculated as

$$
d _ {\mathrm {e m o}} (s ^ {t}, m _ {i}) = \frac {\mathbf {h} _ {e} (\phi_ {e} ; s ^ {t}) \cdot \mathbf {h} _ {e} (\phi_ {e} ; m _ {i}) ^ {T}}{| | \mathbf {h} _ {e} (\phi_ {e} ; s ^ {t}) | | \cdot | | \mathbf {h} _ {e} (\phi_ {e} ; m _ {i}) | |}.
$$

To optimize the emotion scoring function, we construct datasets for pre-training. The core assumption here is that the ability of LLMs to generate sentences with a specific sentiment is superior to their ability to discern the sentiment of given sentences. First, we generate a seed sentence $s _ { 0 }$ without emotion. Next, we randomly select combinations $\{ c _ { i } \} _ { i = 1 } ^ { n }$ of up to three emotions from the eight emotional dimensions. We then instruct the LLMs to generate sentences $s _ { i } = \mathbf { L L M } ( s _ { 0 } , c _ { i } )$ containing the specified emotions based on this seed sentence and each emotional combination. Finally, we compile a dataset ${ \mathcal { D } } ^ { \mathrm { e m o } } = \{ ( s _ { i } , c _ { i } ) \} _ { i = 1 } ^ { n }$ consisting of sentences with different emotions along with their corresponding emotion labels. Finally, we compile a dataset ${ \mathcal { D } } ^ { \mathrm { e m o } } = \{ ( s _ { i } , c _ { i } ) \} _ { i = 1 } ^ { n }$ consisting of sentences with varying emotions and their corresponding emotion labels. Subsequently, we optimize our emotion scoring function with

$$
\phi_ {e} ^ {*} = \arg \min  _ {\phi_ {e}} \frac {1}{| \mathcal {D} ^ {\mathrm {e m o}} |} \sum_ {(s _ {i}, c _ {i}) \in \mathcal {D} ^ {\mathrm {e m o}}} [ \mathbf {h} _ {e} (\phi_ {e}; x); c _ {i} ] ^ {2}.
$$

To verify the effectiveness of our proposed method, we conduct extensive experiments across different baselines, and we present more details and in Appendix A.3.

# A.2 Importance Scoring Function

In a similar approach, we pre-train an importance scoring function to evaluate various messages. It is crucial to differentiate between importance and relevance. Relevance pertains to the degree of semantic similarity between messages and is symmetrical in nature. Conversely, importance refers to the significawe propose $\mathbf { h } _ { p } ( \phi _ { p } ; \bar { s } ^ { t } ) = W _ { 1 } ^ { p } \cdot \mathbf { h } _ { s ^ { t } } ^ { T } + b _ { 1 } ^ { p }$ n relaand $\mathbf { h } _ { p } ( \phi _ { p } ; m _ { i } ) = W _ { 2 } ^ { p } \cdot \mathbf { h } _ { m _ { i } } ^ { T } + b _ { 2 } ^ { p }$ asymme, where $\mathbf { h } _ { s ^ { t } } , \mathbf { h } _ { m _ { i } }$ erefore,are the text embedding of $s ^ { t } , m _ { i }$ , and $\phi _ { p } = \{ W _ { 1 } ^ { p } , W _ { 2 } ^ { p } , b _ { 1 } ^ { p } , b _ { 2 } ^ { p } \}$ . Then, we calculate the importance score with

$$
d _ {\mathrm {i m p}} (s ^ {t}, m _ {i}) = \frac {\mathbf {h} _ {p} (\phi_ {p} ; s ^ {t}) \cdot \mathbf {h} _ {p} (\phi_ {p} ; m _ {i}) ^ {T}}{| | \mathbf {h} _ {p} (\phi_ {p} ; s ^ {t}) | | \cdot | | \mathbf {h} _ {p} (\phi_ {p} ; m _ {i}) | |}.
$$

To optimize the importance scoring function, we construct datasets for pre-training. Initially, we select a query $q$ and a seed sentence $s _ { 0 }$ containing minimal information. Subsequently, we incrementally enrich this seed sentence to generate new sentences $\{ s _ { i } \} _ { i = 1 } ^ { n }$ , thereby forming a partially ordered set. After that, we sample sentences from the same partially ordered set, forming positive and negative examples $( q , s ^ { + } , s ^ { - } )$ . Finally, we obtain the dataset $\mathcal { D } ^ { \mathrm { i m } \bar {  p } } = \{ ( q , s _ { j } ^ { + } , s _ { j } ^ { - } ) \} _ { j = 1 } ^ { m }$ for contrastive learning. We optimize our importance scoring function with

$$
\phi_ {p} ^ {*} = \arg \min  _ {\phi_ {p}} \frac {1}{| \mathcal {D} ^ {\mathrm {i m p}} |} \sum_ {(q, s ^ {+}, s ^ {-}) \in \mathcal {D} ^ {\mathrm {i m p}}} \log \sigma \left[ d _ {\mathrm {i m p}} (q, s ^ {+}) - d _ {\mathrm {i m p}} (q, s ^ {-}) \right] - \log \sigma \left[ d _ {\mathrm {i m p}} (q, s ^ {-}) - d _ {\mathrm {i m p}} (q, s ^ {+}) \right].
$$

# A.3 Comparison Experiments Among Different Scoring Methods

We conducted experiments to assess the effectiveness of pre-trained metric functions within the memory retrieval process. Specifically, we compute NDCG $\textcircled { a } 5$ for ranked messages based on importance scoring and MSE for emotional analysis across various dimensions. We also record the instruction failure rate (IFR) of prompting methods. For zero-shot and few-shot prompting techniques,

Table 4: Results of pre-trained metric functions on testing datasets. Bold values represent the best results, while underlined values represent the second-best results.   

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Base Models</td><td colspan="2">Importance Scorer</td><td colspan="2">Emotion Scorer</td><td rowspan="2">Cost</td></tr><tr><td>nDCG@5 ↑</td><td>IFR</td><td>MSE ↓</td><td>IFR</td></tr><tr><td>Random</td><td>Random</td><td>0.498</td><td>N/A</td><td>2.685</td><td>N/A</td><td>Low</td></tr><tr><td rowspan="4">Zero-shot Prompt</td><td>GPT-4o</td><td>0.648</td><td>0.000</td><td>0.999</td><td>0.000</td><td>High</td></tr><tr><td>GPT-4o-mini</td><td>0.826</td><td>0.000</td><td>0.983</td><td>0.000</td><td>High</td></tr><tr><td>Qwen-2.5</td><td>0.774</td><td>0.000</td><td>0.902</td><td>0.000</td><td>High</td></tr><tr><td>Llama-3.1</td><td>0.629</td><td>0.026</td><td>0.713</td><td>0.001</td><td>High</td></tr><tr><td rowspan="4">Few-shot Prompt</td><td>GPT-4o</td><td>0.672</td><td>0.000</td><td>0.904</td><td>0.000</td><td>High</td></tr><tr><td>GPT-4o-mini</td><td>0.814</td><td>0.000</td><td>0.993</td><td>0.000</td><td>High</td></tr><tr><td>Qwen-2.5</td><td>0.578</td><td>0.000</td><td>0.814</td><td>0.006</td><td>High</td></tr><tr><td>Llama-3.1</td><td>0.439</td><td>0.661</td><td>0.942</td><td>0.001</td><td>High</td></tr><tr><td>Ours (Learning)</td><td>E5-base-v2</td><td>0.978</td><td>N/A</td><td>0.491</td><td>N/A</td><td>Medium</td></tr></table>

![](images/854916a68adb74b6ed5d2c31a41ff262f3a677ecbad5d1be659590228867dde9.jpg)

![](images/fe9668fb4fb95ef823e38b3138218e9f7b64b0a9a32e7124d1d96c8019126df7.jpg)

![](images/9c98afcf05d917b7e7d7848c8d3fe631f1de8f30a7eeb5d3de9bdbaea7c55cd9.jpg)  
Figure 4: Results of different hyper-parameters.

we craft instructions for LLMs to generate scores within the range of [0.0, 1.0] concerning importance and various emotional aspects of specific messages. Additionally, we employ GPT-4o, GPT-4o-mini, Qwen-2.5, and Llama-3.1 as the base models. For the random method, scores were independently generated from a uniform distribution between 0.0 and 1.0. We utilize E5-base-v2 as the base model for sentence embedding within our pre-trained metric functions. The results demonstrate that our pre-trained metric functions achieve notable improvements over zero-shot and few-shot prompting methods in predicting importance scores and emotional analysis. However, some open-source models exhibited instruction failures during scoring, contributing to instability in retrieval metrics.

# B More Experiment Details on MemDaily

We conduct further experiments on the MemDaily dataset, focusing specifically on aggregative question-answering (QA) tasks. These tasks are the most challenging type, as they necessitate extended reasoning by recalling previous user messages. In line with our experiments on HotpotQA, we employ ReAct as the reasoning framework for our agents. To emulate an interactive scenario between users and agents, we divide all user messages in a specific trajectory into $k$ sequential blocks. These blocks serve as $k$ observations provided by the environment (i.e., the user) to the agent. The interactions between the agent and the environment consist of $k + 1$ steps. During the first $k$ steps, we treat each $t$ -th information block as the observation from the environment at step $t$ , where the agent is not required to return any action. In the final step, we present a question to the agent as the observation and require it to return a predicted answer. We then compare the agent’s final answer with the ground truth to calculate the Success Rate (SR). We set $k = 5$ in our experiments.

It should be noted that since the agent does not provide actions during the first $k$ steps, our proposed memory framework performs only one recall per trajectory. Consequently, there is only a single memory entity under this setting, and we optimize memory storage exclusively in off-policy and on-policy optimization. We employed Qwen-2.5 as the inference model for these tasks. The results are presented in Table 5. The experimental results indicate that our proposed method outperforms other baselines. Moreover, both GAMemory and LTMemory also demonstrate strong performance.

Table 5: Overall performance across different baselines MemDaily.   

<table><tr><td>Inference</td><td>FUMemory</td><td>LTMemory</td><td>STMemory</td><td>GAMemory</td><td>MBMemory</td><td>SCMemory</td><td>MTMemory</td><td>Ours-on</td></tr><tr><td>Accuracy</td><td>0.439</td><td>0.5366</td><td>0.5122</td><td>0.5366</td><td>0.4878</td><td>0.3171</td><td>0.2927</td><td>0.561</td></tr></table>

# C More Experiment Details of Hyper-parameter Influence

We further explore the influence of some significant hyper-parameters in our framework, including SFT batch size, DPO batch size, and reflection batch size. According to the results, we find that the best choice of SFT batch size is around 16, while the best DPO batch size is roughly 32. Additionally, we find that the accuracy is more sensitive to variations in DPO batch size, significantly diminishing when the values are particularly low. In contrast, the reflection batch size has a minor impact on performance, with accuracy remaining similar when it ranges from 20 to 50 samples per reflection.

# D Reproduction Details

# D.1 Environment Settings

We construct our environment based on HotpotQA, which includes hard, medium, and easy levels of difficulty. For each trajectory, the agent is required to give the correct answer to a question within a certain steps. For each step, the agent can observe feedback from the environment, and take the next action after that. There are two valid actions from agents: (1) Search[entity] means search the provided entity in Wikipedia. (2) Finish[answer] means give the final answer to the question.

We adopt the fullwiki mode in HotpotQA to make sure an interactive environment. In order to make the experiments more reproducible, we download the dumps file of Wikipedia. We obtain wikipedia_en_all_nopic_2024-06.zim (53.2GB) from Wikimedia Downloads 1, and implement a Wikipedia searcher with libzim package based on previous works. If the environment receives Search[entity], it will search the given entity and return the full document of its Wikipedia page. If the environment receives Finish[answer], it will compare answer with the ground truth and terminate this trajectory. If the environment receives other actions, it will return that the action is invalid. In our experiments, the maximum step is set as 5. The numbers of questions of hard, medium, and easy levels are 113, 109, and 107, respectively.

# D.2 Agent Settings

To better evaluate the memory capability of LLM-based agents, we standardize their reasoning structures as ReAct. For each step, the agent first receives the current state and executes the memory storage procedure. Then, it will execute the memory recall procedure to obtain a memory context. After that, it will think explicitly via LLM inference, and make the decision of actions based on the thought. Finally, it stores the thought and actions into memory and responds to the actions. The prompt of thinking and making action decisions is shown as follows.

# The prompt of thinking of LLM-based agents.

You are a knowledgeable expert, and you are answering a question. You are allowed to search in Wikipedia to get information.

The question is: {question}. Now, you can choose to answer the question or search an entity on Wikipedia. Please think step by step to analyze how to choose the next action, and output it into one paragraph in concise. In previous steps, you have already accumulated some knowledge in your memory as follows: {memory_context}.

# The prompt of making the action decision of LLM-based agents.

You are a knowledgeable expert, and you are answering a question. You are allowed to search in Wikipedia to get information.

The question is: {question}. You have thought step by step to analyze how to choose the next action as follows: {thought}.

Now, you can choose to answer the question or search an entry on Wikipedia: (1)

Search[entity], which searches the entity on Wikipedia and returns the paragraphs if they exist.

(2) Finish[answer], which returns the answer and finishes the task. Your answer should be in concise with several words, NOT a sentence. Please generate the next action accordingly.

Your output must follow one of the following two formats:

Search[entity]

Finish[answer]

Here are some examples:

Search[Alan Turing]

Finish[no]

Finish[Shanghai]

In previous steps, you have already accumulated some knowledge in your memory as follows:

{memory_context}

For the inference LLMs in our experiments, we utilize Qwen2.5-7B-Instruct, Llama3.1-8B-Instruct, and GPT-4o-mini.

# D.3 Baseline Settings

We implement baselines of memory methods based on MemEngine. For all the LLM inference inside memory methods, we utilize Qwen2.5-7B-Instruct as the backbone. For all the text embedding process, we adopt e5-v2-base model with 768 dimensions. For all the top- $k$ retrieval process, we set $k$ as 10 with cosine similarity. We set the maximum length of memory context as 8096 words. For all the summarization process, the prompt is as follows.

# The prompt of summarization process.

Content: {content}

Summarize the above content concisely, extracting the main themes and key information.

Please output your summary directly in a single line, and do not output any other messages.

For GAMemory, we set the question number as 2, the insight number as 2, the reflection threshold as 0.3, the reflection top- $k$ as 2, and the prompt of reflection as follows.

# The prompt of summarization process (generate question).

Information: {information} Given only the information above, what are {question_number} most salient highlevel questions we can answer about the subjects in the statements? Please output each question in a single line, and do not output any other messages.

# The prompt of summarization process (generate insight).

Statements: {statements}

What {insight_number} high-level insights can you infer from the above statements? Please output each insight in a single line (without index), and do not output any other messages.

For MBMemory, we set the forgetting coefficient as 5.0. For our method, the prompts of storage and utilization are shown as follows.

# The prompt of storage process.

Observation: {observation}

Hint: {hint}

From the above observation and according to the hint, please extract critical informative points and summarize them into a concise paragraph. You should just output the result of summarization, without any other messages.

# The prompt of utilization process.

Observation: {observation}

Existing Memory: {memory_context}

New Memory: {new_memory}

Please merge the above new memory into the existing memory, which is useful to response the observation. You should remove the duplicated information to make it concise, but do not lose any information. You should just output the final memory after merge, without any other information.

For the hyper-parameters of off-policy training, we set the SFT learning rate as 0.0001, the SFT batch size as 16, the DPO learning rate as 0.0001, the DPO batch size as 16, and the reflection size as 40. For the hyper-parameters of on-policy training, we set the SFT learning rate as 0.0005, the SFT batch size as 16, the DPO learning rate as 0.0001, the DPO batch size as 16, the reflection size as 15, the sample batch size as 30, and the training epoch as 5.
# NATURAL LANGUAGE REINFORCEMENT LEARNING

Xidong Feng1∗, Ziyu Wan2∗, Mengyue Yang1, Ziyan Wang3, Girish A. Koushik4, Yali $\mathbf { D } \bar { \mathbf { u } } ^ { 3 }$ , Ying Wen2, Jun Wang1

1University College London, 2Shanghai Jiao Tong University, 3King’s Collge London,

4University of Surrey

# ABSTRACT

Reinforcement Learning (RL) has shown remarkable abilities in learning policies for decision-making tasks. However, RL is often hindered by issues such as low sample efficiency, lack of interpretability, and sparse supervision signals. To tackle these limitations, we take inspiration from the human learning process and introduce Natural Language Reinforcement Learning (NLRL), which innovatively combines RL principles with natural language representation. Specifically, NLRL redefines RL concepts like task objectives, policy, value function, Bellman equation, and policy iteration in natural language space. We present how NLRL can be practically implemented with the latest advancements in large language models (LLMs) like GPT-4. Initial experiments over tabular MDPs demonstrate the effectiveness, efficiency, and also interpretability of the NLRL framework.

# 1 INTRODUCTION

Reinforcement Learning (RL) constructs a mathematical framework that encapsulates key decisionmaking elements. It quantifies the objectives of tasks through the concept of cumulative rewards, formulates policies with probability distributions, expresses value functions via mathematical expectations, and models environment dynamics through state transition and reward functions. This framework effectively converts the policy learning problem into an optimization problem.

Despite the remarkable achievements of RL in recent years, significant challenges still underscore the framework’s limitations. For example, RL suffers from the sample efficiency problem–RL algorithms are task-agnostic and do not leverage any prior knowledge, requiring large-scale and extensive sampling to develop an understanding of the environment. RL also lacks interpretability. Despite the superhuman performance of models like AlphaZero (Silver et al., 2017) in mastering complex games such as Go, the underlying strategic logic of their decision-making processes remains elusive, even to professional players. In addition, the supervision signal of RL is a one-dimensional scalar value, which is much more sparse compared with traditional supervised learning over informationrich datasets such as texts and images. This is also one of the reasons for the instability of RL training (Zheng et al., 2023; Andrychowicz et al., 2020).

These limitations drive us to a new framework inspired by the human learning process. Instead of mathematically modeling decision-making components like RL algorithms, humans tend to conduct relatively vague operations by natural language. First, natural language enables humans with textbased prior knowledge, which largely increases the sample efficiency when learning new tasks. Second, humans possess the unique ability to articulate their explicit strategic reasoning and thoughts in natural language before deciding on their actions, making their process fully interpretable by others, even if it’s not always the most effective approach for task completion. Third, natural language data contains information about thinking, analysis, evaluation, and future planning. It can provide signals with high information density, far surpassing that found in the reward signals of traditional RL.

Inspired by the human learning process, we propose Natural Language Reinforcement Learning (NLRL), a new RL paradigm that innovatively combines the traditional RL concepts and natural language representation. By transforming key RL components—such as task objectives, policies, value functions, the Bellman equation, and generalized policy iteration (GPI) (Sutton & Barto,

2018)—into their natural language equivalents, we harness the intuitive power of language to encapsulate complex decision-making processes. This transformation is made possible by leveraging recent breakthroughs in large language models (LLMs), which possess human-like ability to understand, process, and generate language-based information. Our initial experiments over tabular MDPs validate the effectiveness, efficiency, and interpretability of our NLRL framework.

# 2 PRELIMINARY OF REINFORCEMENT LEARNING

Reinforcement Learning models the decision-making problem as a Markov Decision Process (MDP), defined by the state space $s$ , action space $\mathcal { A }$ , probabilistic transition function $P : \mathcal { S } \times$ $\mathcal { A } \times \mathcal { S }  [ 0 , 1 ]$ , discount factor $\gamma \in [ 0 , 1 )$ and reward function $r : \mathcal { S } \times \mathcal { A }  [ - R _ { \mathrm { m a x } } , R _ { \mathrm { m a x } } ]$ . The goal of RL aims to learn a policy $\pi : \mathcal { S \times A } \ :  \ : [ 0 , 1 ]$ , which measures the action $a$ ’s probability given the state s: $\pi ( a | s ) = \operatorname* { P r } \left( A _ { t } = a \mid S _ { t } = s \right)$ . In decision-making task, the optimal policy tends to maximize the expected discounted cumulative reward: $\begin{array} { r } { \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r \left( \mathrm { s } _ { t } , \mathrm { a } _ { t } \right) \right] } \end{array}$ . The state-action and state value functions are two key concepts that evaluate states and actions by RL objective’s proxy: $Q _ { \pi } ( s _ { t } , a _ { t } ) \ = \ \mathbb { E } _ { ( \mathrm { s } , \mathrm { a } ) _ { t + 1 : \infty } \sim P _ { \pi } } \left[ \sum _ { i = t } ^ { \infty } \gamma ^ { i - t } r \left( s _ { i } , a _ { i } \right) \mid s _ { t } , a _ { t } \right]$ ∞ , $V _ { \pi } ( s _ { t } ) =$ $\begin{array} { r } { \mathbb { E } _ { ( \mathrm { s } , \mathrm { a } ) _ { t + 1 : \infty } \sim P _ { \pi } } \left[ \sum _ { i = t } ^ { \infty } \gamma ^ { i - t } r \left( s _ { i } , a _ { i } \right) \mid s _ { t } \right] } \end{array}$ P t, where $P _ { \pi }$ ∞  is the trajectory distribution given the policy $\pi$ and dynamic transition $P$ .

Given the definition of $V _ { \pi } ( s _ { t } )$ , the relationship between temporally adjacent state’s value can be derived as the Bellman expectation equation. Here is a one-step Bellman expectation equation:

$$
V _ {\pi} \left(s _ {t}\right) = E _ {a _ {t} \sim \pi_ {\theta}} \left[ r \left(s _ {t}, a _ {t}\right) + \gamma E _ {s _ {t + 1} \sim p \left(s _ {t + 1} \mid s _ {t}, a _ {t}\right)} \left[ V _ {\pi} \left(s _ {t + 1}\right) \right] \right], \forall s _ {t} \in S \tag {1}
$$

A similar equation can also be derived for $Q _ { \pi } ( s , a )$ . Given these basic RL definitions and equations, we can illustrate how policy evaluation and policy improvement are conducted in GPI.

Policy Evaluation. The target of the policy evaluation process is to estimate state value function $V _ { \pi } ( s )$ or state-action value function $Q _ { \pi } ( s , a )$ . For simplicity, we only utilize $V _ { \pi } ( s )$ in the following illustration. Two common value function estimation methods are the Monte Carlo (MC) estimate and the Temporal-Difference (TD) estimate (Sutton, 1988). MC estimate leverages Monte-Carlo sampling over trajectories to construct unbiased estimation: $\begin{array} { r } { V _ { \pi } ( s _ { t } ) \approx \frac { 1 } { k } \sum _ { n = 1 } ^ { K } [ \sum _ { i = t } ^ { \infty } \gamma ^ { i - t } r ( s _ { i } ^ { n } , a _ { i } ^ { n } ) ] } \end{array}$ . TD estimate relies on the temporal relationship shown in Equ.1 to construct an estimation: $V _ { \pi } ( s _ { t } ) \approx$ $\begin{array} { r } { \frac { 1 } { k } \sum _ { n = 1 } ^ { K } [ r ( s _ { t } , a _ { t } ^ { n } ) + \gamma V _ { \pi } ( s _ { t + 1 } ^ { n } ) ] } \end{array}$ , which can be seen as a bootstrap over next-state value function.

Policy Improvement. The policy improvement process aims to update and improve policy according to the result of policy evaluation. Specifically, it replaces the old policy $\pi _ { \mathrm { o l d } }$ with the new one $\pi _ { \mathrm { n e w } }$ to make the expected return increases: $V _ { \pi _ { \mathrm { n e w } } } \left( s _ { 0 } \right) \geq V _ { \pi _ { \mathrm { o l d } } } \left( s _ { 0 } \right)$ . In the environment with small, discrete action spaces, such improvements can be achieved by greedily choosing the action that maximizes $Q _ { \pi _ { \mathrm { o l d } } } ( s , a )$ at each state:

$$
\pi_ {\text {n e w}} (\cdot \mid s) = \underset {\bar {\pi} (\cdot | s) \in \mathcal {P} (\mathcal {A})} {\arg \max } \mathbb {E} _ {\mathrm {a} \sim \bar {\pi}} \left[ Q _ {\pi_ {\text {o l d}}} (s, \mathrm {a}) \right], \forall s \in \mathcal {S} \tag {2}
$$

Another improvement method involves applying policy gradient ascent (Sutton et al., 1999). It parameterizes the policy $\pi _ { \theta }$ with $\theta$ . The analytical policy gradient can be derived as follows:

$$
\left. \right.\left. \nabla_ {\theta} V \left(\pi_ {\theta}\right)\right| _ {\theta = \theta_ {\mathrm {o l d}}} = \mathbb {E} _ {(\mathrm {s}, \mathrm {a}) \sim P _ {\pi_ {\theta_ {\mathrm {o l d}}}}} \left[\left. \nabla_ {\theta} \log \pi_ {\theta} (a | s) Q _ {\pi_ {\theta_ {\mathrm {o l d}}}} (\mathrm {s}, \mathrm {a}) \right| _ {\theta = \theta_ {\mathrm {o l d}}} \right]. \tag {3}
$$

By choosing a relatively small step-size $\alpha \ > \ 0$ to conduct gradient ascent: $\begin{array} { r l r } { \theta _ { \mathrm { n e w } } } & { { } = } & { \theta + } \end{array}$ $\alpha \nabla _ { \boldsymbol { \theta } } V _ { \pi _ { \boldsymbol { \theta } } } ( s _ { 0 } ) | _ { \boldsymbol { \theta } = \theta _ { \mathrm { o l d } } }$ , we can guarantee the policy improvement: $V _ { \pi _ { \mathrm { n e w } } } \left( s _ { 0 } \right) \geq V _ { \pi _ { \mathrm { o l d } } } \left( s _ { 0 } \right)$ .

# 3 NATURAL LANGUAGE REINFORCEMENT LEARNING

In contrast to the precise statistical models used in traditional RL, humans typically frame all elements—including task objectives, value evaluations, and strategic policies—within the form of natural language. This section aims to mimic how humans navigate decision-making tasks using natural language, aligning it with the concepts, definitions, and equations in traditional RL. Due to the inherent ambiguity of natural language, the equations presented here are not strictly derived from mathematical definitions. They are instead analogical and based on empirical insights of original RL concepts. We leave rigorous theory for future work.

![](images/5cff3a285be61b9b66a3bb8abda80adc4ced4ef3ac2df4653e1387137158309e.jpg)  
Figure 1: We present an illustrative example of grid-world MDP to show how NLRL and traditional RL differ for task objective, value function, Bellman equation, and generalized policy iteration. In this grid-world, the robot needs to reach the crown and avoid all dangers. We assume the robot policy takes optimal action at each non-terminal state, except a uniformly random policy at state b.

# 3.1 DEFINITIONS

We start with definitions and equations in natural language RL to model human’s behaviors. We provide Fig. 1 with illustrative examples covering most concepts we will discuss in this section.

Text-based MDP: To conduct RL in natural language space, we first need to convert traditional MDP to text-based MDP, which leverages text and language descriptions to represent MDP’s basic concepts, including state, action, and environment feedback (state transitions and reward).

Language Task instruction: Humans usually define a natural language task instruction $T _ { L }$ , like “reaching the goal” or “opening the door”. Then, we denote a human metric by $F$ that measures the completeness of the task instruction given the trajectory description $D _ { L } ( \tau _ { \pi } )$ , a language descriptor $D _ { L }$ that can transform the trajectory distribution $\tau _ { \pi }$ into its corresponding language description $D _ { L } ( \tau _ { \pi } )$ . Based on the notation, the objective of NLRL is reformulated as

$$
\max  _ {\pi} F \left(D _ {L} \left(\tau_ {\pi}\right), T _ {L}\right), \tag {4}
$$

We are trying to optimize the policy so that the language description of the trajectory distribution $\tau _ { \pi }$ can show high completeness of the task instruction.

Language Policy: Instead of directly modeling action probability, humans determine the action by strategic thoughts, logical reasoning, and planning. Thus, we represent the policy on language as $\pi _ { L } ( \boldsymbol { a } , \boldsymbol { c } | \boldsymbol { s } )$ , which will first generate such thought process $c$ , then the final action probability $\pi ( \boldsymbol { a } | \boldsymbol { s } )$ .

Language Value Function: Similar to the definition of $Q$ and $V$ in traditional RL, humans leverage language value function, relying on natural language evaluation to assess the policy effectiveness. The language state value function $V _ { \pi } ^ { L }$ and language state-action value function $\mathrm { ~ \ i ~ } Q _ { \pi } ^ { L }$ are defined as:

$$
Q _ {\pi} ^ {L} \left(s _ {t}, a _ {t}\right) = D \left(\left(\mathrm {s}, \mathrm {a}\right) _ {t + 1: \infty} \sim P _ {\pi} \mid s _ {t}, a _ {t}, T _ {L}\right), V _ {\pi} ^ {L} \left(s _ {t}\right) = D \left(\mathrm {a} _ {\mathrm {t}}, (\mathrm {s}, \mathrm {a}) _ {t + 1: \infty} \sim P _ {\pi} \mid s _ {t}, T _ {L}\right), \tag {5}
$$

Given the current state $s _ { t }$ or state-action $\left( { { s _ { t } } , { a _ { t } } } \right)$ , $Q _ { \pi } ^ { L }$ and $V _ { \pi } ^ { L }$ leverage language descriptions instead of scalar value to demonstrate the effectiveness of policy for achieving the task objective $T _ { L }$ . The language value functions are intuitively rich in the information of values and enhance interpretability rather than the traditional scalar-based value. It can represent the evaluation results from different perspectives, consisting of the underlying logic/thoughts, prediction/analysis of future outcomes, comparison among different actions, etc.

Language Bellman Equation: In the Bellman equation, the state evaluation value $V ( s _ { t } )$ , can be decomposed into two parts. Firstly the intermediate changes, which include immediate action $a _ { t }$ , reward $r _ { t }$ , and next state $s _ { t + 1 }$ . Secondly, the state evaluation $V ( s _ { t + 1 } )$ over the next state. Based on such intuition of decomposition, we introduce the language Bellman equation Equ.6 following the decomposition principle.

$$
V _ {\pi} ^ {L} \left(s _ {t}\right) = G _ {1} ^ {a _ {t}, s _ {t + 1} \sim P _ {\pi}} \left(G _ {2} \left(d \left(a _ {t}, r \left(s _ {t}, a _ {t}\right), s _ {t + 1}\right), V _ {\pi} ^ {L} \left(s _ {t + 1}\right)\right)\right), \forall s _ {t} \in \mathcal {S} \tag {6}
$$

where $d ( a _ { t } , r ( s _ { t } , a _ { t } ) , s _ { t + 1 } ) )$ depicts the language description of intermediate changes, $G _ { 1 }$ and $G _ { 2 }$ serves as two information aggregation function. Specifically, $G _ { 2 }$ mimics the add $\cdot _ { + } ,$ operation in the original Bellman equation, aggregating information from intermediate changes’ descriptions and future evaluation given $a _ { t }$ and $s _ { t + 1 }$ . $G _ { 1 }$ takes the responsibility of the expectation operation $\mathbb { E }$ , aggregating information from different $\left( a _ { t } , s _ { t + 1 } \right)$ pairs by sampling from trajectory distribution $P _ { \pi }$ .

# 3.2 LANGUAGE GENERALIZED POLICY ITERATION

Given definitions and equations, in this part, we introduce how language GPI is conducted. Refer to Fig. 1 with illustrative examples of language GPI.

# 3.2.1 LANGUAGE POLICY EVALUATION

Language policy evaluation aims to estimate language value function $V _ { \pi } ^ { L }$ and $Q _ { \pi } ^ { L }$ for each state. We present how two classical estimations: MC and TD estimate work in language policy evaluation.

Language Monte-Carlo Estimate. Starting from the state $s _ { t }$ , MC estimate is conducted over text rollouts (i.e. $K$ full trajectories $\{ a _ { t } , ( s , a ) _ { t + 1 : \infty } \} )$ given the policy $\pi$ . Since we cannot take the average operation in language space, we instead leverage language aggregator/descriptor $G _ { 1 }$ to aggregate information over finite trajectories, approximating the expected evaluation.

$$
V _ {\pi} ^ {L} \left(s _ {t}\right) \approx G _ {1} \left(\left\{\mathrm {a} _ {\mathrm {t}} ^ {\mathrm {n}}, (\mathrm {s}, \mathrm {a}) _ {t + 1: \infty} ^ {n} \right\} _ {n = 1} ^ {k}\right) \tag {7}
$$

Language Temporal-Difference Estimate. TD estimate mainly relies on the one-step language Bellman equation illustrated in Equ. 6. Similar to MC estimate, we aggregate $K$ one-step samples to approximate the expected evaluation:

$$
V _ {\pi} ^ {L} \left(s _ {t}\right) \approx G _ {1} \left(\left\{G _ {2} \left(d \left(s _ {t}, a _ {t} ^ {n}, r \left(s _ {t}, a _ {t} ^ {n}\right), s _ {t + 1} ^ {n}\right), V _ {\pi} ^ {L} \left(s _ {t + 1} ^ {n}\right)\right) \right\} _ {n = 1} ^ {K}\right), \forall s _ {t} \in \mathcal {S} \tag {8}
$$

Language MC estimate is free from estimation “bias” as it directly utilizes samples from complete trajectories. However, the MC method is prone to high “variance” considering the significant variations in the long-term future steps. Such variability poses a challenge for the language aggregator $D$ in Equ. 7 to efficiently extract crucial information from diverse trajectories. On the contrary, while the inaccuracy of the next state evaluation $V _ { \pi } ^ { L } ( s _ { t + 1 } )$ can bring estimation “bias” to TD estimate, they effectively reduce “variance” by discarding future variations. $G _ { 1 }$ and $G _ { 2 }$ are only required to conduct simple one-step information aggregation with limited variations.1

# 3.2.2 LANGUAGE POLICY IMPROVEMENT

Similar to traditional policy improvement, the motivation of language policy improvement also aims to select actions that maximize the human task completeness function $F$ :

$$
\pi_ {\text {n e w}} (\cdot \mid s) = \underset {\bar {\pi} (\cdot | s) \in \mathcal {P} (\mathcal {A})} {\arg \max } F \left(Q _ {\pi_ {\text {o l d}}} ^ {L} (s, a), T _ {L}\right), \forall s \in \mathcal {S} \tag {9}
$$

As we mentioned, $F$ is typically a human measurement of task completeness, which is hard to quantify and take the argmax operation. Considering that $F$ largely depends on human prior knowledge, instead of mathematically optimizing it, we leverage the language analysis process $I$ to conduct policy optimization and select actions:

$$
\pi_ {\text {n e w}} (\cdot \mid s), c = I \left(Q _ {\pi_ {\text {o l d}}} ^ {L} (s, a), T _ {L}\right), \bar {\pi} (\cdot \mid s) \in \mathcal {P} (\mathcal {A}), \forall s \in \mathcal {S} \tag {10}
$$

Language policy improvement conducts strategic analysis $C$ to generate the thought process $c$ and determine the most promising action as the new policy $\pi _ { \mathrm { n e w } } ( \cdot \mid s )$ . This analysis is mainly based on human’s correlation judgment between the language evaluation $\dot { Q } _ { \pi _ { 0 } | \mathrm { d } } ^ { L } ( s , a )$ and task objective $T _ { L }$ .

# 3.3 PRACTICAL IMPLEMENTATION WITH LARGE LANGUAGE MODELS

Section 3 demonstrates the philosophy of NLRL: transfer RL key concepts into its human natural language correspondence. To practically implement these key concepts, we require a model that can understand, process, and generate language information. The Large language model, trained with large-scale human language and knowledge corpus, can be a natural choice to help mimic human behaviors and implement these language RL components.

LLMs as policy $( \pi _ { L } )$ . Many works adopted LLMs as the decision-making agent (Wang et al., $2 0 2 3 \mathrm { a }$ ; Feng et al., 2023a; Christianos et al., 2023; Yao et al., 2022) with Chain-of-thought process (Wei et al., 2022b). By setting proper instructions, LLMs can leverage natural language to describe their underlying thought for determining the action, akin to the human strategic thinking.

LLMs as information extractor and aggregator $( G _ { 1 } , G _ { 2 } )$ for concepts. LLMs can be powerful information summarizers (Zhang et al., 2023), extractors (Xu et al., 2023), and aggregators to help us fuse intermediate changes and future language evaluations for language value function estimates.

One core issue is to determine which kind of information we hope our LLMs to extract and aggregate. Inspired by works Das et al. (2023); Sreedharan et al. (2020); Schut et al. (2023); Hayes & Shah (2017) in the field of interpretable RL, we believe Concept can be the core. We adopt the illustration in Das et al. (2023) that concept is a general, task-objective oriented, and high-level abstraction grounded in human domain knowledge. For example, in the shortest path-finding problem, the path distance and available path sets are two concepts that are (1) high-level abstraction of the trajectories and are predefined in human prior knowledge, (2) generally applicable over different states, (3) directly relevant to the final task objective. Given these motivations, LLMs will try to aggregate and extract domain-specific concepts to form the value target information. Such concepts can be predefined by human’s prior knowledge, or proposed by LLMs themselves.

LLMs as value function approximator $( D _ { L } , Q ^ { L } , V ^ { L } )$ . The key idea of value function approximation (Sutton et al., 1999) is to represent the value function with a parameterized function instead of a table representation. Nowadays deep RL typically chooses neural networks that take the state as input and output one-dimension scalar value. For NLRL, the language value function approximation of $D _ { L } , Q ^ { L } , \dot { V } ^ { L }$ can be naturally handled by (multi-modal) LLMs. LLMs can take in the features from the task’s state, such as low-dimension statistics, text, and images, and output the corresponding language value judgment and descriptions. For the training of LLMs, we adopt the concept extractor/aggregator mentioned above to form MC or TD estimate (in Sec 3.2.1), which can be used to finetune LLMs for better language critics.

LLMs as policy improvement operator $( I )$ . With the chain-of-thought process and human prior knowledge, LLMs are better to determine the most promising action $\bar { \pi _ { \mathrm { n e w } } } ( \cdot \mid s )$ by taking language analysis $c$ over the correlation of language evaluation $Q _ { \pi _ { 0 } \mathrm { l d } } ^ { L } ( s , a )$ and task objective $T _ { L }$ . The underlying idea also aligns with some recent works (Kwon et al., 2023; Rocamonde et al., 2023) that leverage LLMs or Vision-language models as the reward–they can accurately model the correlation.

# 3.4 DISCUSSIONS OVER OTHER RL CONCEPTS

To illustrate the versatility of the framework, we show several examples of how other fundamental RL concepts can be framed into NLRL.

TD- $\lambda$ (Sutton, 1988). Equ. 6 considers the one-step decomposition of the value function, or in the context of traditional RL, the TD(1) situation. A natural extension is to conduct an $n$ -step

![](images/1294900f0714888ce45015c9d121607e82f7764078668a3e3692f54e815aece5.jpg)  
Figure 2: How the language evaluation over state $g$ at (0,3) evolves across iterations. Iter 0: initial descriptions. Iter 1: Intermediate changes exclude action go up and move right. However, determining the optimal move is not possible without next-state evaluations. Iter 3: identify two optimal actions by information transmission from the goal. Iter 4: Evaluation over the state $g$ converges.

![](images/74f4f549c6e3134896fc199b948f0ba90005133845ff68dcc3300af29bfc1306.jpg)

![](images/8cc03aa79d26686e3e17c66efc40484486c2f53f49a62e20c709e84645e7693f.jpg)

![](images/f74631608209f78715ce9507bdfa92d4496a2db5ecace459832f32e50fea01a2.jpg)

![](images/0c2154a706b84b5901bd5ab192a479920cd36ecda25f5992412783868e60650b.jpg)  
Figure 3: The information flows from the goal to every state across iterations. The yellow grid represents the grids where the goal information is transmitted. The blue array denotes the direction of information transmission by applying the one-step language Bellman update.

expansion of the language value function and form the $\mathrm { T D } ( n )$ estimate. By further aggregating these different estimations (1 to $n$ ), we obtain the NLRL’s TD- $\lambda$ estimation. TD- $\lambda$ under the language value function involves aggregating different language evaluations into the final language estimation.

Exploration. Exploration stands as a fundamental challenge in RL. Traditional RL approaches often utilize strategies that introduce randomnesses, such as the $\epsilon$ -greedy method or entropy regularization (Sutton & Barto, 2018), to promote exploration. In contrast, NLRL offers a pathway for exploration that is guided by uncertainty. For instance, the language value function in NLRL is adept at incorporating models of state uncertainty. This allows for a deeper analytical process during the language policy improvement, where the strategy can be refined to reduce such uncertainties.

Episodic and Retrieval-based RL. The concept of Episodic and Retrieval-based RL Goyal et al. (2022); Blundell et al. (2016) is to keep explicit records of past trajectories, which will be used for reference to augment the decision-making ability in the new state. This aligns with the idea of Retrieval-augmented generation (RAG) (Lewis et al., 2020) in LLMs. By leveraging RAG over past trajectories, NLRL can also be extended to Retrieval-based RL.

# 4 EXPERIMENTS

We conduct initial experiments over tabular MDP as proof-of-concept verifications. Refer to Appendix A for more experimental results, settings and prompts.

# 4.1 WARM-UP: LANGUAGE POLICY EVALUATION IN TEXT GRIDWORLD

We start with the shortest path-finding problem, which is commonly adopted as an illustration for dynamic programming-based policy iteration algorithms. As shown in Fig. 3 (Iteration 0), this is a $4 \times 4$ grid world and the goal is to reach two grey grids at (0,0) and (3,3) with the shortest path. We convert this environment to a text game, where we leverage natural language to represent the state, the transition, and the environment feedback such as path penalty and goal-reaching termination. We set the initial policy as a random policy that samples action from a uniform distribution over all available actions: $\begin{array} { r } { \pi ( \tilde { a | s } ) = \frac { 1 } { | \tilde { A | } } , \forall a \stackrel { \bullet } { \in } \tilde { A , s } \in \mathcal { S } } \end{array}$ . We conduct language policy evaluation over this policy, similar to the policy evaluation process in the original policy iteration algorithm.

Table 1: Predefined concept names and corresponding definitions.   

<table><tr><td>Concept name</td><td>Concept definition</td></tr><tr><td>Important state</td><td>Record the important state positions such as goal position, and hole positions.</td></tr><tr><td>Immediate risk</td><td>Identify immediate one-step failure.</td></tr><tr><td>Future risk</td><td>Identify potential future failure paths (more than one step) or future bad states.</td></tr><tr><td>Safest path</td><td>Generate the safest path starting from the given state and action.</td></tr><tr><td>Final evaluation</td><td>The final evaluation of the policy in the given state.</td></tr></table>

We initialize the language evaluation descriptions for all states within the language state value table as ”No evaluation information”, except terminal states, which are initialized with a predefined description specific to their terminal nature. Similar to the traditional policy evaluation process, each language policy evaluation iterates over all non-terminal states. For each state $s$ and action $a$ , it leverages the one-step Bellman equation shown in Equ. 6 to calculate the new language state-action value function $Q ^ { L } ( s , \bar { a } )$ and ${ \cal V } ^ { L } ( s )$ . This evaluation is performed over four iterations to observe the evolution of the language state-action value function across iterations. We employ the GPT-4- preview-1106 model as the information aggregator for the language Bellman updates. Moreover, by converting coordinates to letters, we ensure that GPT-4 relies exclusively on environmental feedback for information gathering, thereby eliminating dependence on its inherent knowledge base.

In Fig. 2, we showcase the results of language evaluation for state $g$ at (0,3) across various iterations, serving as a representative example. For comprehensive results pertaining to different states, please see Appendix A. Three key insights emerge from our analysis. First, despite that we prompt the GPT-4 model for policy evaluation, it inherently undertakes a policy improvement process, selecting the most advantageous action options. This demonstrates GPT-4’s capacity for language policy enhancement, as it interprets language evaluations and task goals. Second, the evaluation outcomes in the final iteration correctly identify the optimal action for every state. This confirms the language policy evaluation and iteration process’s effectiveness and efficiency, where merely four iterations of policy evaluation are sufficient to yield precise language evaluation and the respective optimal actions, significantly outperforming the conventional policy iteration method. Finally, the language evaluation outcomes elucidate the gradual transmission of goal-related information, with the example in Fig. 2 illustrating how information from the goal flows to state $g$ over three evaluation iterations. Furthermore, Fig. 3 is provided to depict the flow of information across iterations visually.

# Language Value Function

Important states: Goal: (3, 3), Hole: (1, 1), (2, 3), (3, 0),

Immediate Risk: (2,0) $_ - >$ Move down $- >$ (3,0) (hole), distance is 1; (2,0) $- >$ Move right -> (3,0) (hole), distance is 1

Future Risk: (2,0) $- >$ Move left -> (3,0) (hole), distance is 2; Moving from (2,1) involves risks of slipping into holes with most directions except moving right, which leads to the goal., Safest path: (2,0) -> Move down $- >$ (2,1) -> Move right -> (3,1) -> Move right -> (3,2) -> Move down $- >$ (3,3),

Final evaluation: The player's policy in the current state (2,0) presents a mix of risks and opportunities. The immediate risks involve potentially falling into a hole when moving down or right, with a one-step distance to failure. The action of moving left does not result in immediate failure but has a future risk of leading into a hole at (3,0). The safest path to the goal involves moving down to (2,1) and then navigating right and down towards the goal, avoiding immediate risks and leading directly to the goal. This path, however, assumes successful navigation to (2,1) without slipping…

# Language Policy Improvement

Analysis: Considering the evaluations, the Move right action is the most promising as it avoids immediate risks and offers a direct path to the goal despite the stochastic nature of the environment. While it introduces variability with potential outcomes that could lead to less favorable positions or directly into a hole, it's generally safe and provides a pathway to the goal. The other actions, Move left, Move down, and Move up, all present immediate risks of falling into holes or lead to positions with high future risks, making them less advisable,

Action: Move right

Figure 4: Frozen-Lake example of the language value function and language policy improvement. The language value function addresses the 5 predefined concepts, while the language policy improvements conduct chain-of-thought reasoning to determine the final action.

Table 2: Average state value over all states during the language policy iteration process, compared with the optimal policy’s value on the right.

<table><tr><td></td><td>Iter 0</td><td>Iter 1</td><td>Iter 2</td><td>Iter 3</td><td>Iter 4</td><td>Optimal</td></tr><tr><td>Average Value</td><td>0.062</td><td>0.217</td><td>0.326</td><td>0.261</td><td>0.327</td><td>0.555</td></tr></table>

# 4.2 LANGUAGE POLICY EVALUATION AND IMPROVEMENT IN STOCHASTIC ENVIRONMENT

In our second experiment, we explore the Frozen-Lake environment from Brockman et al. (2016), a grid world designed to navigate an agent to a goal while avoiding lake holes, challenging due to the randomness in state transitions. Following the framework of Sheng et al. (2023), we convert this environment into a text-based format. Our approach involves four iterations of language policy, each comprising an evaluation and an improvement phase.

Our initial experiments showed that the direct prompting method (used in our first experiment) easily failed the policy iteration process in the frozen-lake task. This is mainly because of the large number of trajectory variations, which can easily distract the LLM and disrupt the aggregation process. To tackle this problem, we leverage a more explicit method for aggregation: the conceptbased language value function and information extraction discussed in Section 3.3. We predefined several key concepts listed in Table 1 and prompted the GPT-4-0125-preview model to generate evaluations and aggregate information based on these concepts.

Fig. 4 presents an example of the language value function and the language policy improvement process. The value function example clearly explains why the definition of task-relevant concepts can facilitate information aggregation. The GPT-4 can easily conduct explicit concept aggregation: it can update the path information by appending the newest move to the leftmost, or extract environment information by transmitting the coordinates of goals and holes.

In Table. 2 we provide the policy value for each iteration by leveraging frozen-lake’s original reward function. The table presents an overall increasing trend, verifying the effectiveness of our language policy iteration process. Nevertheless, the system only attains $6 \hat { 0 } \%$ efficiency compared to the optimal policy. Further analysis of the language value function pinpointed the issue: even the most advanced GPT-4 model is not immune to hallucination problems, occasionally generating inaccurate information. These inaccuracies can accumulate through successive Bellman updates, leading to incorrect evaluations and decisions. Efforts are underway to mitigate this instability through improved prompting strategies.

# 5 RELATED WORK

Language Model Based Autonomous Agent. Inspired by the strong emergent capabilities of LLMs in complex reasoning and planning scenarios (Brown et al., 2020; Wei et al., 2022a; Anil et al., 2023; OpenAI, 2023; Feng et al., 2023b). The field of language model-based autonomous agents have witnessed a growing trend of leveraging LLMs for high-level planning purposes. For text agents, ReAct(Yao et al., 2022) leverages chain-of-thought planning about the intermediate goals and taskspecific generation via few-shot prompting. Reflexion (Shinn et al., 2023) is built upon ReAct (Yao et al., 2022) with self-reflexion, named as verbal reinforcement learning, to generate tips given the online feedback. This somehow aligns with the setting of language Monte-Carlo estimate where we extract core information/concepts by sampling multiple trajectories. Another line of work like Zhang et al. (2023); Xu et al. (2023) leverages large language models to summarize and extract information to help fuse intermediate changes. In embodied agent research (Duan et al., 2022), for instance, research for robot learning demonstrates large language models’ potential in hierarchical planning and subgoal generation (Huang et al., 2022; Liang et al., 2023).

Interpretable Reinforcement Learning. One of the major purposes of interpretable RL is to automatically seek explanations for non-AI experts. For instance, methods of concept-based explanations for sequential decision-making tasks. Ji et al. (2023) provide a concept-based explanation for 3D action recognition CovNets by clustering learned human interpretable features. Sreedharan et al. (2020) formulates concept-based explanations upon state preconditions and action costs, representing any factual statement a user associates with a given state. Similarly, Hayes & Shah (2017) uses logical formulas to summarize policies. Additionally, Das et al. (2023) trains a joint embedding model for state-action pairs and concept-based explanations.

Learning from Language Feedback. Our work is also related to learning from language feedback (Cheng et al., 2023). Cheng et al. (2023) focuses on how to benchmark algorithms while we aim to propose a new algorithm framework. It has been investigated how to provide feedback to an RL agent’s activity as part of the learning signal in addition to task rewards in the field of robotics and autonomous agents while in which most of the efforts were devoted to eliciting binary preference feedback(Biyik & Sadigh, 2018) or ranking-based labels from human(Sumers et al., 2021). Our algorihtms can also benefit from different benchmarks (Liu et al., 2023; Abdulhai et al., 2023; Sheng et al., 2023) that try to incorporate language feedback to develop RL policies. We are actively testing our algorihtms on these benchmarks.

LLM as evaluation function. Our language value function is also related to some work in the NLP field that leverages LLMs as evaluators or verifiers, including Wang et al. (2023b); Li et al. (2023);

Jiang et al. (2023); Gao et al. (2024). The main differences between our language value function with these works are two-fold. First, these works mainly focus on natural language tasks while we aim to tackle decision-making problems. Second, most of these works train their LLM evaluator by distilling strong models such as GPT- $4 ^ { 2 }$ , while we mainly rely on environmental feedback to generate synthetic evaluation data.

# 6 CONCLUSION AND LIMITATIONS

In this work, we propose a comprehensive framework NLRL, which reformulates the RL process with natural language representation. We believe NLRL opens a new door for understanding and implementing RL algorithms in natural language space. In addition, NLRL also provides a potential way for generating high-quality language synthetic data, which could be crucial in training more advanced language models.

We are working on addressing the following limitations: (1) Currently, the LLM’s hallucination issue severely influences the performance of NLRL. We will focus on addressing this problem to stabilize the NLRL’s policy iteration process, (2) Our experiment is only limited to tabular MDP. We are working on scaling up our experiments beyond tabular settings, and (3) Currently, the performance of the language value function and the process of language policy iteration is measured purely by the policy’s performance. We are working on a new metric to comprehensively evaluate the performance.

# REFERENCES

Marwa Abdulhai, Isadora White, Charlie Snell, Charles Sun, Joey Hong, Yuexiang Zhai, Kelvin Xu, and Sergey Levine. Lmrl gym: Benchmarks for multi-turn reinforcement learning with language models. arXiv preprint arXiv:2311.18232, 2023.   
Marcin Andrychowicz, Anton Raichuk, Piotr Stanczyk, Manu Orsini, Sertan Girgin, Raphael ´ Marinier, Leonard Hussenot, Matthieu Geist, Olivier Pietquin, Marcin Michalski, et al. What ´ matters in on-policy reinforcement learning? a large-scale empirical study. arXiv preprint arXiv:2006.05990, 2020.   
Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.   
Erdem Biyik and Dorsa Sadigh. Batch active preference-based learning of reward functions. In Conference on robot learning, pp. 519–528. PMLR, 2018.   
Charles Blundell, Benigno Uria, Alexander Pritzel, Yazhe Li, Avraham Ruderman, Joel Z Leibo, Jack Rae, Daan Wierstra, and Demis Hassabis. Model-free episodic control. arXiv preprint arXiv:1606.04460, 2016.   
Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.   
Ching-An Cheng, Andrey Kolobov, Dipendra Misra, Allen Nie, and Adith Swaminathan. Llf-bench: Benchmark for interactive learning from language feedback. arXiv preprint arXiv:2312.06853, 2023.   
Filippos Christianos, Georgios Papoudakis, Matthieu Zimmer, Thomas Coste, Zhihao Wu, Jingxuan Chen, Khyati Khandelwal, James Doran, Xidong Feng, Jiacheng Liu, et al. Pangu-agent: A fine-tunable generalist agent with structured reasoning. arXiv preprint arXiv:2312.14878, 2023.

Devleena Das, Sonia Chernova, and Been Kim. State2explanation: Concept-based explanations to benefit agent learning and user understanding. arXiv preprint arXiv:2309.12482, 2023.   
Jiafei Duan, Samson Yu, Hui Li Tan, Hongyuan Zhu, and Cheston Tan. A survey of embodied ai: From simulators to research tasks. IEEE Transactions on Emerging Topics in Computational Intelligence, 6(2):230–244, 2022.   
Xidong Feng, Yicheng Luo, Ziyan Wang, Hongrui Tang, Mengyue Yang, Kun Shao, David Mguni, Yali Du, and Jun Wang. Chessgpt: Bridging policy learning and language modeling. arXiv preprint arXiv:2306.09200, 2023a.   
Xidong Feng, Ziyu Wan, Muning Wen, Ying Wen, Weinan Zhang, and Jun Wang. Alphazerolike tree-search can guide large language model decoding and training. arXiv preprint arXiv:2309.17179, 2023b.   
Mingqi Gao, Xinyu Hu, Jie Ruan, Xiao Pu, and Xiaojun Wan. Llm-based nlg evaluation: Current status and challenges. arXiv preprint arXiv:2402.01383, 2024.   
Anirudh Goyal, Abram Friesen, Andrea Banino, Theophane Weber, Nan Rosemary Ke, Adria Puigdomenech Badia, Arthur Guez, Mehdi Mirza, Peter C Humphreys, Ksenia Konyushova, et al. Retrieval-augmented reinforcement learning. In International Conference on Machine Learning, pp. 7740–7765. PMLR, 2022.   
Bradley Hayes and Julie A Shah. Improving robot controller transparency through autonomous policy explanation. In Proceedings of the 2017 ACM/IEEE international conference on humanrobot interaction, pp. 303–312, 2017.   
Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022.   
Ying Ji, Yu Wang, and Jien Kato. Spatial-temporal concept based explanation of 3d convnets. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15444–15453, 2023.   
Dongfu Jiang, Yishan Li, Ge Zhang, Wenhao Huang, Bill Yuchen Lin, and Wenhu Chen. Tigerscore: Towards building explainable metric for all text generation tasks. arXiv preprint arXiv:2310.00752, 2023.   
Minae Kwon, Sang Michael Xie, Kalesha Bullard, and Dorsa Sadigh. Reward design with language models. arXiv preprint arXiv:2303.00001, 2023.   
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨ aschel, et al. Retrieval-augmented genera- ¨ tion for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33: 9459–9474, 2020.   
Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Hai Zhao, and Pengfei Liu. Generative judge for evaluating alignment. arXiv preprint arXiv:2310.05470, 2023.   
Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 9493–9500. IEEE, 2023.   
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, et al. Agentbench: Evaluating llms as agents. arXiv preprint arXiv:2308.03688, 2023.   
OpenAI. Gpt-4 technical report. ArXiv, abs/2303.08774, 2023.   
Juan Rocamonde, Victoriano Montesinos, Elvis Nava, Ethan Perez, and David Lindner. Visionlanguage models are zero-shot reward models for reinforcement learning. arXiv preprint arXiv:2310.12921, 2023.

Lisa Schut, Nenad Tomasev, Tom McGrath, Demis Hassabis, Ulrich Paquet, and Been Kim. Bridging the human-ai knowledge gap: Concept discovery and transfer in alphazero. arXiv preprint arXiv:2310.16410, 2023.   
Junjie Sheng, Zixiao Huang, Chuyun Shen, Wenhao Li, Yun Hua, Bo Jin, Hongyuan Zha, and Xiangfeng Wang. Can language agents be alternatives to ppo? a preliminary empirical study on openai gym. arXiv preprint arXiv:2312.03290, 2023.   
Noah Shinn, Beck Labash, and Ashwin Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. arXiv preprint arXiv:2303.11366, 2023.   
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676):354–359, 2017.   
Sarath Sreedharan, Utkarsh Soni, Mudit Verma, Siddharth Srivastava, and Subbarao Kambhampati. Bridging the gap: Providing post-hoc symbolic explanations for sequential decision-making problems with inscrutable representations. arXiv preprint arXiv:2002.01080, 2020.   
Theodore R Sumers, Mark K Ho, Robert D Hawkins, Karthik Narasimhan, and Thomas L Griffiths. Learning rewards from linguistic feedback. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 6002–6010, 2021.   
Richard S Sutton. Learning to predict by the methods of temporal differences. Machine learning, 3:9–44, 1988.   
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.   
Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.   
Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023a.   
Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, et al. Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization. arXiv preprint arXiv:2306.05087, 2023b.   
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682, 2022a.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022b.   
Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, and Enhong Chen. Large language models for generative information extraction: A survey. arXiv preprint arXiv:2312.17617, 2023.   
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.   
Tianyi Zhang, Faisal Ladhak, Esin Durmus, Percy Liang, Kathleen McKeown, and Tatsunori B Hashimoto. Benchmarking large language models for news summarization. arXiv preprint arXiv:2301.13848, 2023.   
Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, et al. Secrets of rlhf in large language models part i: Ppo. arXiv preprint arXiv:2307.04964, 2023.

# A EXPERIMENTAL DETAILS

# A.1 PROMPTS

We present the prompts used in our shortest path-finding grid-world and Frozen-Lake environment.

# The shortest path-finding prompts

terminal_state_description $=$ "The state is the terminal state. Your move ends here so you will not receive any negative path cost anymore in the future."

system_prompt $=$ "You are a helpful assistant that strictly follows the user’s instructions.\n" \

"We are finding the shortest path towards the terminal state in gridworld. We use (letter) to represent different grids.\n" \

"To help us evaluate a given grid position in the shortest path finding task, we have an agent that can conduct actions to move to other grids. The agent has 4 actions: go up, move left, go down and move right. The agent conducts several one-step rollouts to explore such grid world.\n"

"For each one-step rollout starting from a given grid, it contains two sources of information, one about the descrption of intermediate change of action, state and reward, while the other one is about our evaluation of the new grid state.\n" \

"I need you to help me aggregate/summarize on these rollout information to concisely evaluate the given grid position."

rollout_prompt $=$ "The current grid state is {current_state}. And here are several one-step rollout results starting from the current state {current_state}:\n\n"

trigger_prompt $=$ "By aggregating the above rollout results, please concisely generate your evaluation of the current state {current_state}. Start your answer with the sentence ’Final evaluation of the current state:’."

action_description $=$ f"You choose the {action_dict[action]} action. You receive a negative reward -1 as the path penalty. Now you are at a new state ({state_list[next_i][next_j]})."

# The prompts used in Frozen-Lake

### prompts for getting language state-action value function Q:

```python
AGG_TD_STATE_INPUT_FORMAT_str = ""State: {state}  
Action: {action}  
Possible outcomes:  
{agg_str}"''" 
```

AGG_TD_STATE_SYSTEM_PROMPT_FC $=$ """You are a highly skilled assistant, committed to following user instructions precisely. Our task involves a player navigating a stochastic environment known as ’Frozen Lake,’ with the ultimate goal of reaching the goal position while avoiding all holes. Your role is to assist in evaluating the player’s action on the given state.

In this environment, the player makes decisions in each state, leading to potentially varied subsequent states due to the stochastic nature of the environment. We will provide you with a set of possible outcomes resulting from one action.

For each outcome, you will receive two key pieces of information:

1. A detailed description of the immediate changes in state and the accompanying reward. This narrative will offer insights into the direct consequences of the player’s actions.   
2. An assessment of the player’s policy at the newly reached state. This evaluation will include a future analysis of potential outcomes stemming from the new state.

# ### Your task

Your task is to synthesize these elements - the future state evaluations and the intermediate state descriptions - to construct a comprehensive evaluation of the player’s action given current state.

# ### Json keys

You need to generate the analysis in the json format, here are the keys:

1. Important states: Record the important state positions such as goal position, hole positions or other important states.   
2. Immediate Risk: Identify one-step failure (the action can directly result in failure). Generate None if no paths found. Here is an example if you are evaluating over state (4,1): (4,1) -> Move left -> (4,0) (hole), distance is 1.   
3. Future Risk: Identify potential future failure paths (more than one step) or future bad states. Generate None if no paths found.   
4. Safest: Generate the safest path starting from the given state and action. Output None if there are no safe paths. Here is an example if you are evaluating over state (4,1): (4,1) -> Move right -> (4,2) -> Move down -> (6,2), this is the goal.   
5. Final evaluation: Based on the above infomation, generate your final evaluation of the policy in the given state.

Make your analysis and evaluation concise and short.

One criteria is that a good action should not result in immediate risk.

### prompts for getting language state value function V:

AGG_ACTION_INPUT_FORMAT_STR $=$ """State: {state}

Possible actions and evaluations:

{agg_str}"""

AGG_ACTION_SYSTEM_PROMPT_FC $=$ """Our task involves a player navigating a stochastic environment known as ’Frozen Lake,’ with the ultimate goal of reaching an unknown destination. Your role is to assist in evaluating the player’s action.

In this environment, the player makes decisions in each state, leading to potentially varied subsequent states due to the stochastic nature of the environment.

You are tasked with evaluating a player’s stochastic policy in ’Frozen Lake’. The player has multiple actions to choose from in each state.

You will be provided with current state infromation and action choices with their corresponding action evaluations (in json format).

# ### Your task

Your task is to synthesize the effectiveness of different actions and combine these insights to assess the overall effectiveness of the player’s policy in the current state.

# ### Json keys

You need to generate the analysis in the json format, here are the keys:

1. Important states: Record the important state positions such as goal position, hole positions or other important states.   
2. Immediate Risk: Identify one-step failure (the action can directly result in failure). Here is an example if you are evaluating over state (4,1): (4,1) $- >$ Move left -> (4,0) (hole), distance is 1.

3. Future Risk: Identify potential future failure paths (more than one step) or future bad states.

4. Safest path: Generate the safest path starting from the given state. Output None if there are no safe paths. Here is an example: (4,1) -> Move right -> (4,2) -> Move down -> (6,2), this is the goal.   
5. Final evaluation: Based on the above infomation, generate your final evaluation of the policy in the given state.

Make your analysis and evaluation concise and short.

One criteria is that a good action should not result in immediate failure and hole. """

# A.2 EXPERIMENTAL SETTINGS AND RESULTS

For the shortest path-finding problem, we leverage GPT-4-preview-1106, with temperature ${ \mathrm { = } } 1 . 0$ . Here we present the full language evaluation results for the whole 4 iterations. Each iteration has one $4 \times 4$ matrix for language state value function.

# Language evaluation results in grid-world across iterations

### Iteration 1

[[’The state is the terminal state. Your move ends here so you will not receive any negative path cost anymore in the future.’,

’The best action to take from state (b) is the Move Left action leading to state (f), which is the terminal state. This outcome is based on the provided rollouts, as it immediately leads to the terminal state without incurring any additional negative path costs.’,

’State (k) has four possible actions, each resulting in a -1 penalty. Going up keeps the agent in the same state (k), going left takes the agent to state (b), going down takes the agent to state (m), and moving right takes the agent to state (g). Without additional evaluation information for states (b), (m), and (g), it is not possible to determine the most beneficial action from state (k) towards the terminal state.’,

’The rollouts from state (g) indicate that going up and moving right results in the agent staying at the same state, which suggests these actions lead to a wall or boundary. Therefore, these directions do not provide a path towards the terminal state. Moving left and going down transitions the agent to new states (k and w, respectively) with a path penalty of -1 for each. Without additional information regarding the value of states (k) and (w), it is not possible to determine which one is closer or leads to the terminal state. However, since these are the only actions that lead to new states, further exploration from state (k) or state (w) would be necessary to find the shortest path toward the terminal state.’],

[’The current state (z) has four possible actions. The Go Up action leads directly to the terminal state (f) with a path cost of -1, which is the optimal move since it results in reaching the terminal state immediately. The Move Left action results in no change, staying at (z) with a path cost of -1. Both Go Down and Move Right actions lead to new states (r) and (t) respectively with a path cost of -1; however, their values or positions in relation to the terminal state are unknown from the given rollouts. Therefore, given the available information, the best action from state (z) is to Go Up to reach the terminal state (f) immediately.’,

’With the information provided, it is not possible to determine the optimal direction for the shortest path from state (t) as there are no evaluations for the resulting states (b), (z), (y), and (m) after performing the actions Go Up, Move Left, Go Down, and Move Right respectively. All actions from state (t) incur a path penalty of -1, but without further information about the desirability or utility of subsequent states, we cannot conclude which action would be part of the shortest path to the terminal state.’,

’Since all one-step rollouts from the current state (m) result in a negative reward of -1 and there is no further evaluation information available for the new states (k), (t), (q), and (w), we cannot determine which direction leads to the shortest path towards the terminal state. Further information about the states following (k), (t), (q), and (w) is required to make an informed evaluation.’,

’The one-step rollouts from state (w) lead to states (g), (m), (n), and back to (w) with a uniform negative reward of -1 for the path penalty. However, there is no further evaluation information provided for the new states (g), (m), or (n). Therefore, we cannot determine which of these one-step transitions, if any, leads closer to the terminal state. Additional evaluation information would be needed to determine the most promising action from state (w).’],

["Considering the one-step rollouts from state (r) and without any further evaluation information for states (z), (j), and (y), all actions from state (r) result in a negative reward of -1 due to the path penalty. There is currently no information to distinguish the actions based on the follow-up states’ values. More information or further exploration would be needed to determine the optimal action from state (r).",

’Final evaluation of the current state (y): With the available information from the one-step rollouts, it is not possible to determine which action leads to the shortest path towards the terminal state. Each action taken from the current state (y) incurs a path penalty of -1, and there is no further evaluation information given for the resulting new states (t), (r), (d), and (q). Therefore, additional information or exploration is needed to ascertain the most beneficial action from state (y).’,

’Since all actions from state (q) lead to different new states (m, y, e, n) with a uniform negative reward of -1 and no further evaluation information provided, the current state (q) cannot be distinctly evaluated over the other states in terms of guiding to a terminal state or providing a better reward path. Without additional information or context regarding the final goal state or additional state rewards, all actions from the current state appear to be of equal value based on the given one-step rollouts.’,

’The best action to take from state (n) is to Go Down, moving to state (x), as it leads to the terminal state, thereby ending the path and avoiding any further negative path cost.’],

[’Considering the outcomes of the rollouts from state (j), we have four attempted actions with associated penalties but no further evaluative information about the resulting states. Rollouts 1 and 2 return the agent to the same state (j), so these actions do not progress towards the terminal state. Rollouts 0 and 3 move the agent to new states (r) and (d) respectively, both with a path penalty of -1. Without additional information regarding states (r) and (d), we cannot determine which, if either, leads closer to the terminal state. Based on current data, state (j) is a decision point with yet unspecified potential, requiring further evaluation of subsequent states to ascertain the optimal action.’,

’Without any evaluation information on the resulting states (y), (j), (d), and (e) after the one-step rollouts from state (d), it is not possible to determine a preferred action or the value of state (d). Each action incurred a negative reward of -1, and further

information is required to assess which action leads us closer to the terminal state.’,

’Based on the provided one-step rollouts, the most favorable action from state (e) is Move Right, leading directly to the terminal state (x) with a path cost of -1 and no further negative costs incurred. Therefore, the path through state (x) is the shortest among the explored options.’,

’The state is the terminal state. Your move ends here so you will not receive any negative path cost anymore in the future.’]]

# ### Iteration 2

[[’The state is the terminal state. Your move ends here so you will not receive any negative path cost anymore in the future.’,

’The optimal action from state (b) is the Move Left action as it immediately leads to the terminal state (f), ensuring the least cost path with no further negative path penalties.’,

’The most promising action from state (k) appears to be the Move Left action, which takes the agent to state (b). From state (b), the agent can then take the Move Left action again to reach state (f), which is the terminal state. This sequence of actions leads to the terminal state with a total penalty of -2, based on the given rollout information. Other actions from state (k) either keep the agent in the same state or take the agent to states without a clear path to the terminal state, requiring further exploration for a definitive evaluation.’,

’Actions going up and moving right from state (g) do not provide progress towards the terminal state as the agent remains in state (g), indicating these directions likely lead to walls or boundaries. Actions moving left and going down lead to new states (k and w respectively) with a path penalty of -1 for each. Since there are no further details provided to compare the potential benefits of states (k) and (w), it is not possible to determine which direction is more favorable from the current information. Both state (k) and state (w) would need further exploration to ascertain the most effective route towards the terminal state.’],

[’The best action from state (z) is to Go Up to reach the terminal state (f) immediately with a path cost of -1, making it the optimal move from the current state given the rollout information provided.’,

’Final evaluation of the current state (t): The best action to take from state (t), based on the provided one-step rollouts, is the Move Left action, leading directly to state (z). From state (z), the optimal follow-up action is the Go Up action, which takes the agent to the terminal state (f) with the lowest cumulative path cost from the current state. This path incurs a total negative reward of -2 (one for moving left from (t) to (z), and one for going up from (z) to (f)), and leads to the terminal state with fewer steps compared to other available actions from state (t).’,

"Without additional evaluative information or a known terminal state, it is not possible to determine a preferred action from state (m), as all actions leading to states (k), (t), (q), and (w) result in a negative reward of -1 and each of these states has subsequent actions that also produce a negative reward of -1 without further direction. Additional information about the desirability of each state or knowledge of the terminal state’s location is required to make a valuable assessment for the shortest path strategy.",

’Based on the available one-step rollouts from state (w), the best action appears to be Go Down, leading to state (n). This state offers a pathway to state (x) without any further negative path costs, as state (x) is the terminal state. Actions such as Go Up, Move Left, and Move Right either lead to states with uncertain paths to the terminal or back to the same state (w), resulting in additional negative rewards without progression towards the goal. Therefore,

the optimal move from state (w) is to Go Down to state (n), followed by another Go Down to reach the terminal state (x).’],

[’Final evaluation of the current state (r): The best action to take from state (r) based on the given rollouts is to Go Up, which leads to state (z). From state (z), the agent can take the Go Up action to immediately reach the terminal state (f) with an overall path cost of -2, which is the minimum known cost for reaching the terminal state from (r).’,

"Final evaluation of the current state (y): Based on the one-step rollouts provided for state (y), all adjacent states (t, r, d, q) are reachable with a uniform negative reward of -1, which suggests that merely reaching these states does not provide sufficient information to distinguish the optimal move for the shortest path. The evaluation of the current grid position (y) requires additional information about the terminal state’s location, the grid’s wider layout, and potential rewards or penalties beyond the immediate adjacent states. As such, without further data or a defined terminal state, it is not possible to conclusively determine the value of state (y) or recommend the best action to progress towards the terminal state.",

"Final evaluation of the current state (q): The most beneficial action to take from state (q) is to conduct the action ’Go Down’ as per Rollout 2, since this action leads to state (e) and from there the agent can move right to the terminal state (x) with an overall path cost of -2, which appears to be the shortest path based on the given information.",

’The most promising action from state (n) is to Go Down, as it directly leads to the terminal state (x), which stops accruing any further negative path cost. Other actions such as Go Up, Move Left, and Move Right lead to states that either have no additional evaluative information to suggest they are closer to the terminal state or in the case of Move Right, lead back to the starting state (n). Hence, the optimal move from (n) is the Go Down action leading to the terminal state (x).’],

[’Final evaluation of the current state (j): Given the available data from the one-step rollouts, it appears the agent remains at state (j) when selecting either the Move Left action or the Go Down action, both resulting in a negative reward of -1 without progress toward the terminal state. The Go Up action leads to state (r), and the Move Right action leads to state (d), with both actions also incurring a negative reward of -1. However, there is insufficient data to determine whether state (r) or state (d) brings the agent closer to the terminal state. Consequently, more information and exploration are required to discern the optimal action from state (j).’,

’Based on the one-step rollout information provided, state (d) presents as a decision node in the grid world with uncertain outcomes. However, given that one of the one-step rollouts from state (d) leads to state (e), which in turn has a known path to the terminal state (x), we can deduce that moving right to state (e) and then once more to the terminal state (x) is currently the shortest path identifiable from state (d). Thus, the best current action from state (d) is to move right based on the limited information available. Further exploration from states (y), (j), and from state (d) itself may need to be conducted to conclusively determine the overall shortest path to the terminal state within the entire grid world.’,

’Based on one-step rollouts from state (e), the Move Right action leading directly to the terminal state (x) is the most favorable action, with a path penality of -1. Moving up to state (q), left to state (d), or down to remain in state (e) incurs a negative reward of -1 without leading to the terminal state, making these actions less desirable in the context of reaching the terminal state as quickly as possible.’,

’The state is the terminal state. Your move ends here so you will not receive any negative path cost anymore in the future.’]]

# ### Iteration 3

[[’The state is the terminal state. Your move ends here so you will not receive any negative path cost anymore in the future.’,

’Final evaluation of the current state (b): The optimal action from state (b) is the Move Left action as it leads directly to the terminal state (f) with a single path penalty of -1, which is the least cost path with no further negative path penalties according to the provided one-step rollouts.’,

’The most promising action from state (k) is the Move Left action, leading to state (b), which offers a clear path to the terminal state (f) with a total penalty of -2. Other actions such as Go Up, Go Down, and Move Right lead to states (k), (m), and (g) respectively, and do not present a clear or direct path to the terminal state based on the given rollout information.’,

’Starting from state (g), the agent should avoid the actions Go Up and Move Right as they result in no progress, likely due to walls or boundaries. The available actions leading to progress are Move Left to state (k) and Go Down to state (w), each incurring a path penalty of -1. From state (k), the agent can proceed with Move Left actions to reach the terminal state (f) with an additional -1 penalty, totaling a penalty of -2 from the current state (g). From state (w), the agent can reach the terminal state (x) by performing the Go Down action twice without any further penalties, also totaling a penalty of -2 from (g). Without additional information to differentiate the two terminal states (f and x), it is not possible to determine a single best path from state (g). Both paths to (f) and (x) seem equally efficient based on the given rollouts. Further rollouts or information about the specific terminal state goal would be necessary to make a conclusive decision on the optimal path.’],

[’Final evaluation of the current state (z): The best action to take from state (z) is the Go Up action, which leads directly to the terminal state (f) with an immediate reward of -1 for the path cost. This is the optimal move as it minimizes the path cost and reaches the terminal state in the fewest number of steps according to the given rollouts.’,

’Final evaluation of the current state (t): Given the results of the one-step rollouts, the Go Up action leading to state (b) appears to be the most promising, as the subsequent action from there is Move Left, which directly leads to the terminal state (f) with no further penalties. This suggests that the optimal path from the current state (t) involves the Go Up action followed by Move Left, minimizing the path cost to reach the terminal state efficiently. Rollouts from other actions did not provide a clear or immediate path to the terminal state.’,

’Final evaluation of the current state (m): Considering the one-step rollouts, the Go Up action that leads to state (k) appears to be the most promising initial action. From state (k), the agent can then Move Left to state (b) and again Move Left to reach the terminal state (f). This sequence of moves results in a total path penalty of -3, offering a clear and direct path to the terminal state, which based on the given rollout information is the shortest path with the fewest steps from the current state (m).’,

’The optimal move from state (w) is to Go Down, transitioning the agent to state (n), which allows for a subsequent Go Down action to reach the terminal state (x) without incurring further negative path penalties. This is the shortest path to the terminal state based on the provided rollout information.’],

[’Final evaluation of the current state (r): Based on the available one-step rollouts from the current state (r), it is determined that

the optimal action is to Go Up, leading to state (z). From state (z), the agent can Go Up again to reach the terminal state (f) with a total path cost of -2, which is the shortest known path to the terminal state from (r). Other actions lead to states (r), (j), and (y), which have not been determined to provide a more efficient path to the terminal state within the scope of the information provided. Therefore, the recommended action from state (r) is to Go Up.’,

’Final evaluation of the current state (y): Based on the one-step rollouts conducted from state (y), the optimal action to take is the Move Left action to reach state (r). From state (r), the best subsequent action is to Go Up, arriving at state (z). Finally, from state (z), the Go Up action will lead to the terminal state (f) with the lowest cumulative path cost of -2, implying that this route is the shortest path to the terminal state from the current state (y).’,

’Final evaluation of the current state (q): Based on the provided one-step rollouts, it appears that the shortest path toward the terminal state can be achieved by first selecting the Go Down action to transition to state (e), and subsequently the Move Right action to reach the terminal state (x). The other actions evaluated during the rollouts did not provide a path to the terminal state and consistently resulted in a negative reward of -1 without additional strategic information to justify their selection. Therefore, the path from state (q) to state (e), and then to the terminal state (x) is the most advantageous sequence of actions based on the available data.’,

’Final evaluation of the current state (n): The best action for the agent to take from the current state (n) is to Go Down, according to Rollout 2. This action leads directly to the terminal state (x) with a path penalty of -1, which is the shortest path with the least negative reward based on the available rollouts.’],

[’Final evaluation of the current state (j): Based on the available one-step rollout data, the agent remains at state (j) when selecting either the Move Left action or the Go Down action, both incurring a path penalty of -1 without advancement toward the terminal state. The choices of Go Up or Move Right from state (j) also result in a path penalty of -1, but lead to states (r) and (d) respectively. Rollout data suggests that state (r) is two steps away from the terminal state (f) with a known path cost of -2. State (d) is connected to a path that leads to the terminal state (x), but the exact path cost from state (d) to the terminal state (x) remains unclear. With the data at hand, and considering the lower known cost to reach the terminal state from state (r), the Go Up action from state (j) to state (r), and subsequently to state (z) and then to the terminal state (f), appears to be the most promising path. Additional exploration is needed to confirm the optimal route fully, as well as to ascertain the cost and route details for reaching the terminal state (x) from state (d).’,

’Final evaluation of the current state (d): Based on the gathered data from the one-step rollouts, the shortest confirmed path from state (d) towards the terminal state involves performing the Move Right action to transition to state (e), and then conducting another Move Right action to reach the terminal state (x), with each action incurring a negative reward of -1 as a path penalty. Any further actions from state (d) have not been concretely linked to a path towards the terminal state, necessitating additional exploration for a comprehensive shortest path analysis. However, with the current information, moving right twice from state (d) to reach state (x) appears to be the most optimal known path to the terminal state.’,

’Final evaluation of the current state (e): The most beneficial action to take from state (e) is to conduct the Move Right action which leads directly to the terminal state (x) with a path penalty of -1. This is the shortest path to the terminal state based on the one-step rollout results provided.’,

’The state is the terminal state. Your move ends here so you will not receive any negative path cost anymore in the future.’]]

# ### Iteration 4

[[’The state is the terminal state. Your move ends here so you will not receive any negative path cost anymore in the future.’,

’The optimal action from state (b) is the Move Left action as it leads directly to the terminal state (f) with a single path penalty of -1, which is the best option according to the current one-step rollouts provided.’,

’Final evaluation of the current state (k): The optimal action from state (k) is the Move Left action, resulting in a move to state (b) and then following another Move Left to reach the terminal state (f). This path incurs a total penalty of -2, which is the least negative penalty according to the provided one-step rollouts, offering the clearest and most direct path to the terminal state from the current state (k).’,

’From state (g), the agent has two equally efficient paths toward two different terminal states. One path is to Move Left to state (k), followed by another Move Left to the terminal state (f), totaling a path penalty of -2. The other path is to Go Down to state (w), followed by two Go Down actions reaching the terminal state (x), also with a total path penalty of -2. Actions such as Go Up and Move Right result in no progress and should be avoided. Without specific information on the desired terminal state, both paths are optimal based on the given rollouts.’],

[’Final evaluation of the current state (z): The best action to take from state (z) is the Go Up action, which directly leads to the terminal state (f) with an immediate reward of -1 for the path cost. This is the optimal move as it reaches the terminal state in the fewest number of steps with the least negative reward based on the given rollouts.’,

’Final evaluation of the current state (t): Given the outcomes of the one-step rollouts, the optimal action to take from state (t) is the Move Left action, resulting in the agent moving to state (z). From state (z), the agent can then Go Up to directly reach the terminal state (f). This sequence of actions incurs a total path penalty of -2, which is the lowest cumulative cost among all explored options, thus providing the shortest path to the terminal state from the current state (t).’,

’Final evaluation of the current state (m): After examining the one-step rollout outcomes, the most promising action from state (m) seems to be the Go Up action, leading to state (k). From state (k), the subsequent Move Left action would take the agent to state (b), which presents a direct and clear path to the terminal state (f), incurring a total path penalty of -2. This path is suggested due to the strategic information given in Rollout 0 that indicates a clear path to the terminal state from state (b). Other actions such as Move Left, Go Down, and Move Right lead to states (t), (q), and (w) respectively, but none of these rollouts suggest a better path to the terminal state than the one starting with the Go Up action.’,

’Final evaluation of the current state (w): The optimal action for an agent at state (w) appears to be the Go Down action, resulting in reaching state (n) with a -1 path penalty. Then, from state (n), the agent can perform another Go Down action, leading directly to the terminal state (x) with no additional path penalties. This sequence of actions results in a total path penalty of -2 and offers the shortest path to a terminal state from the current state (w) based on the given rollouts.’],

[’Final evaluation of the current state (r): Based on the one-step rollouts conducted, the most efficient action from state (r) is to Go Up to state (z). From state (z), the Go Up action again will lead

directly to the terminal state (f), with a total path cost of -2. This route offers the optimal path to the terminal state considering the rollout information provided.’,

’Final evaluation of the current state (y): Considering the information available from the one-step rollouts, the state with the most promising subsequent actions leading directly to the terminal state is state (t), which is reached by the Go Up action. From state (t), the agent can Move Left to directly reach the terminal state (f). Therefore, the recommended action from state (y) is to Go Up to state (t) and then Move Left towards the terminal state (f) for the shortest known path given the current data.’,

’Final evaluation of the current state (q): According to the rollout results, the most promising move from (q) appears to be the Go Down action leading to state (e), as it results in reaching the terminal state (x) with the lowest cumulative negative reward of -1, indicating the shortest path from the current state. Therefore, the next action should be Go Down.’,

’Final evaluation of the current state (n): The optimal action for the agent from state (n) is to Go Down, as demonstrated by Rollout 2. This action transitions the agent directly to the terminal state (x) with a path penalty of -1, representing the shortest path with the least negative reward. None of the other actions evaluated in the rollouts provide a more advantageous route to the terminal state from the current state (n).’],

[’Final evaluation of the current state (j): The optimal move from state (j) based on the available data is to Go Up to state (r). From state (r), the agent can Go Up again to reach state (z) and subsequently move up to the terminal state (f) with a total known path cost of -3. Other actions such as Move Left and Go Down lead back to state (j) with a path penalty of -1, and do not progress towards the terminal state. The Move Right action from state (j) leads to state (d), which connects to terminal state (x) via two Move Right actions with a total path cost of -2. However, since the total known path cost to reach terminal state (f) from state (j) is lower, the route from state (j) to state (r) to state (z) and finally to terminal state (f) is currently the most promising path for the shortest route based on the given data.’,

’Based on the one-step rollouts provided, the optimal action from state (d) is to Move Right, transitioning to state (e), and then to perform another Move Right action to reach the terminal state (x). This path incurs a total path penalty of -2, representing the shortest confirmed route to a terminal state from the current state (d).’,

’Final evaluation of the current state (e): Based on the one-step rollout results, the Move Right action from state (e) immediately leads to the terminal state (x) with a path penalty of -1. This is the optimal action to take from state (e) as it directly achieves the goal with the minimal penalty, rendering it the shortest path to the terminal state according to the data from the rollouts.’,

’The state is the terminal state. Your move ends here so you will not receive any negative path cost anymore in the future.’]]

![](images/eee7647e05bf5c0c812a23cdb5b55ed3e3ed09031213673d216d611b9be22439.jpg)  
Figure 5: The policy value at each state in Frozen-Lake.

For the frozen-lake experiment, we leverage GPT-4-preview-0125 with temperature as 0. In Fig. 5 we present the policy value for each state across different iterations. For the language evaluation data, we will open-source them on Huggingface.
# EFFICIENT EPISODIC MEMORY UTILIZATION OF COOP-ERATIVE MULTI-AGENT REINFORCEMENT LEARNING

Hyungho $\mathbf { N a } ^ { 1 }$ , Yunkyeong Seo1 & Il-Chul Moon1,2

1Korea Advanced Institute of Science and Technology (KAIST), 2summary.ai {gudgh723}@gmail.com,{tjdbsrud,icmoon}@kaist.ac.kr

# ABSTRACT

In cooperative multi-agent reinforcement learning (MARL), agents aim to achieve a common goal, such as defeating enemies or scoring a goal. Existing MARL algorithms are effective but still require significant learning time and often get trapped in local optima by complex tasks, subsequently failing to discover a goal-reaching policy. To address this, we introduce Efficient episodic Memory Utilization (EMU) for MARL, with two primary objectives: (a) accelerating reinforcement learning by leveraging semantically coherent memory from an episodic buffer and (b) selectively promoting desirable transitions to prevent local convergence. To achieve (a), EMU incorporates a trainable encoder/decoder structure alongside MARL, creating coherent memory embeddings that facilitate exploratory memory recall. To achieve (b), EMU introduces a novel reward structure called episodic incentive based on the desirability of states. This reward improves the TD target in Q-learning and acts as an additional incentive for desirable transitions. We provide theoretical support for the proposed incentive and demonstrate the effectiveness of EMU compared to conventional episodic control. The proposed method is evaluated in StarCraft II and Google Research Football, and empirical results indicate further performance improvement over state-of-the-art methods. Our code is available at: https://github.com/HyunghoNa/EMU.

# 1 INTRODUCTION

Recently, cooperative MARL has been adopted to many applications, including traffic control (Wiering et al., 2000), resource allocation (Dandanov et al., 2017), robot path planning (Wang et al., 2020a), and production systems (Dittrich & Fohlmeister, 2020), etc. In spite of these successful applications, cooperative MARL still faces challenges in learning proper coordination among multiple agents because of the partial observability and the interaction between agents during training.

To address these challenges, the framework of centralized training and decentralized execution (CTDE) (Oliehoek et al., 2008; Oliehoek & Amato, 2016; Gupta et al., 2017) has been proposed. CTDE enables a decentralized execution while fully utilizing global information during centralized training, so CTDE improves policy learning by accessing to global states at the training phase. Especially, value factorization approaches (Sunehag et al., 2017; Rashid et al., 2018; Son et al., 2019; Yang et al., 2020; Rashid et al., 2020; Wang et al., 2020b) maintain the consistency between individual and joint action selection, achieving the state-of-the-art performance on difficult multi-agent tasks, such as StarCraft II Multi-agent Challenge (SMAC) (Samvelyan et al., 2019). However, learning optimal policy in MARL still requires a long convergence time due to the interaction between agents, and the trained models often fall into local optima, particularly when agents perform complex tasks (Mahajan et al., 2019). Hence, researchers present a committed exploration mechanism under this CTDE training practice (Mahajan et al., 2019; Yang et al., 2019; Wang et al., 2019; Liu et al., 2021) with the expectation to find episodes escaping from the local optima.

Despite the required exploration in MARL with CTDE, recent works on episodic control emphasize the exploitation of episodic memory to expedite reinforcement learning. Episodic control (Lengyel & Dayan, 2007; Blundell et al., 2016; Lin et al., 2018; Pritzel et al., 2017) memorizes explored states and their best returns from experience in the episodic memory, to converge on the best policy. Recently, this episodic control has been adopted to MARL (Zheng et al., 2021), and this episodic

control case shows faster convergence than the learning without such memory. Whereas there are merits from episodic memory and control from its utilization, there exists a problem of determining which memories to recall and how to use them, to efficiently explore from the memory. According to Blundell et al. (2016); Lin et al. (2018); Zheng et al. (2021), the previous episodic control generally utilizes a random projection to embed global states, but this random projection hardly makes the semantically similar states close to one another in the embedding space. In this case, exploration will be limited to a narrow distance threshold. However, this small threshold leads to inefficient memory utilization because the recall of episodic memory under such small thresholds retrieves only the same state without consideration of semantic similarity from the perspective of goal achievement. Additionally, the naive utilization of episodic control on complex tasks involves the risk of converging to local optima by repeatedly revisiting previously explored states, favoring exploitation over exploration.

Contribution. This paper presents an Efficient episodic Memory Utilization for multi-agent reinforcement learning (EMU), a framework to selectively encourage desirable transitions with semantic memory embeddings.

• Efficient memory embedding: When generating features of a global state for episodic memory (Figure 1(b)), we adopt an encoder/decoder structure where 1) an encoder embeds a global state conditioned on timestep into a low-dimensional feature and 2) a decoder takes this feature as an input conditioned on the timestep to predict the return of the global state. In addition, to ensure smoother embedding space, we also consider the reconstruction of the global state when training the decoder to predict its return. To this end, we develop deterministic Conditional AutoEncoder (dCAE) (Figure 1(c)). With this structure, important features for overall return can be captured in the embedding space. The proposed embedding contains semantic meaning and thus guarantees a gradual change of feature space, making the further exploration on memory space near the given state, i.e., efficient memory utilization.   
• Episodic incentive generation: While the semantic embedding provides a space to explore, we still need to identify promising state transitions to explore. Therefore, we define a desirable trajectory representing the highest return path, such as destroying all enemies in SMAC or scoring a goal in Google Research Football (GRF) (Kurach et al., 2020). States on this trajectory are marked as desirable in episodic memory, so we could incentivize the exploration on such states according to their desirability. We name this incentive structure as an episodic incentive (Figure 1(d)), encouraging desirable transitions and preventing convergence to unsatisfactory local optima. We provide theoretical analyses demonstrating that this episodic incentive yields a better gradient signal compared to conventional episodic control.

We evaluate EMU on SMAC and GRF, and empirical results demonstrate that the proposed method achieves further performance improvement compared to the state-of-art baseline methods. Ablation studies and qualitative analyses validate the propositions made by this paper.

# 2 PRELIMINARY

# 2.1 DECENTRALIZED POMDP

A fully cooperative multi-agent task can be formalized by following the Decentralized Partially Observable Markov Decision Process (Dec-POMDP) (Oliehoek & Amato, 2016), $G =$ $\langle I , \dot { S } , A , P , R , \Omega , O , n , \gamma \rangle$ , where $I$ is the finite set of $n$ agents; $s \in S$ is the true state of the environment; $a _ { i } \in A$ is the $i$ -th agent’s action forming the joint action $\mathbf { \pmb { a } } \in A ^ { n }$ ; $P ( s ^ { \prime } | s , \pmb { a } )$ is the state transition function; $R$ is a reward function $r = R ( s , \pmb { a } , s ^ { \prime } ) \in \mathbb { R }$ ; $\Omega$ is the observation space; $O$ is the observation function generating an observation for each agent $o _ { i } \in \Omega$ ; and finally, $\gamma \in [ 0 , 1 )$ is a discount factor. At each timestep, an agent has its own local observation $o _ { i }$ , and the agent selects an action $a _ { i } \in A$ . The current state $s$ and the joint action of all agents $\textbf { \em a }$ lead to a next state $s ^ { \prime }$ according to $P ( s ^ { \prime } | s , \mathbf { a } )$ . The joint variable of s, a, and $s ^ { \prime }$ will determine the identical reward $r$ across the multi-agent group. In addition, similar to Hausknecht & Stone (2015); Rashid et al. (2018), each agent utilizes a local action-observation history $\tau _ { i } \in T \equiv ( \Omega \times A )$ for its policy $\pi _ { i } ( a | \tau _ { i } )$ , where $\stackrel { \_ } { \pi } : T \times A  [ 0 , 1 ]$ .

# 2.2 DESIRABILITY AND DESIRABLE TRAJECTORY

Definition 1. (Desirability and Desirable Trajectory) For a given threshold return $R _ { t h r }$ and a trajectory $\mathcal { T } : = \{ s _ { 0 } , \pmb { a _ { 0 } } , r _ { 0 } , s _ { 1 } , \pmb { a _ { 1 } } , r _ { 1 } , . . . , s _ { T } \}$ , $\tau$ is considered as a desirable trajectory, denoted as $\mathcal { T } _ { \xi }$ , when an episodic return is $R _ { t = 0 } = \Sigma _ { t ^ { \prime } = t } ^ { T - 1 } r _ { t ^ { \prime } } \geq R _ { \mathrm { t h r } }$ . A binary indicator $\xi ( \cdot )$ denotes the desirability of state $s _ { t }$ as $\xi ( s _ { t } ) = 1$ when $s _ { t } \in \forall \mathcal T _ { \xi }$ .

In cooperative MARL tasks, such as SMAC and GRF, the total amount of rewards from the environment within an episode is often limited as $R _ { \mathrm { m a x } }$ , which is only given when cooperative agents achieve a common goal. In such a case, we can set $R _ { t h r } = R _ { \operatorname* { m a x } }$ . For further description of cooperative MARL, please see Appendix A.

# 2.3 EPISODIC CONTROL IN MARL

Episodic control was introduced from the analogy of a brain’s hippocampus for memory utilization (Lengyel & Dayan, 2007). After the introduction of deep Q-network, Blundell et al. (2016) adopted this idea of episodic control to the model-free setting by storing the highest return of a given state, to efficiently estimate the Q-values of the state. This recalling of the high-reward experiences helps to increase sample efficiency and thus expedites the overall learning process (Blundell et al., 2016; Pritzel et al., 2017; Lin et al., 2018). Please see Appendix A for related works and further discussions.

At timestep $t$ , let us define a global state as $s _ { t }$ . When utilizing episodic control, instead of directly using $s _ { t }$ , researchers adopt a state embedding function $f _ { \phi } ( \bar { s } ) : \bar { S } \to \mathbb { R } ^ { k }$ to project states toward a $k$ -dimensional vector space. With this projection, a representation of global state $s _ { t }$ becomes $x _ { t } = f _ { \phi } ( s _ { t } )$ . The episodic control memorizes $H ( f _ { \phi } ( s _ { t } ) )$ , i.e., the highest return of a given global state $s _ { t }$ , in episodic buffer $\mathcal { D } _ { E }$ (Pritzel et al., 2017; Lin et al., 2018; Zheng et al., 2021). Here, $x _ { t }$ is used as a key to the highest return, $H ( x _ { t } )$ ; as a key-value pair in $\mathcal { D } _ { E }$ . The episodic control in Lin et al. (2018) updates $H ( x _ { t } )$ with the following rules.

$$
H \left(x _ {t}\right) = \left\{ \begin{array}{c c} \max  \left\{H \left(\hat {x} _ {t}\right), R _ {t} \left(s _ {t}, \boldsymbol {a} _ {t}\right) \right\}, & \text {i f} \left| \left| \hat {x} _ {t} - x _ {t} \right| \right| _ {2} <   \delta \\ R _ {t} \left(s _ {t}, \boldsymbol {a} _ {t}\right), & \text {o t h e r w i s e ,} \end{array} \right. \tag {1}
$$

where $R _ { t } ( s _ { t } , { \pmb a } _ { t } )$ is the return of a given $\left( { { s _ { t } } , { a _ { t } } } \right)$ ; $\delta$ is a threshold value of state-embedding difference; and $\hat { x } _ { t } = f _ { \phi } ( \hat { s } _ { t } )$ is $x _ { t } = f _ { \phi } ( s _ { t } )$ ’s nearest neighbor in $\mathcal { D } _ { E }$ . If there is no similar projected state $\hat { x } _ { t }$ such that $| | \hat { x } _ { t } - x _ { t } | | _ { 2 } < \delta$ in the memory, then $H ( x _ { t } )$ keeps the current $R _ { t } ( s _ { t } , \pmb { a } _ { t } )$ . Leveraging the episodic memory, EMC (Zheng et al., 2021) presents the one-step TD memory target $Q _ { E C } ( \bar { f } _ { \phi } ( \bar { s } _ { t } ) , \bar { a _ { t } } )$ as

$$
Q _ {E C} \left(f _ {\phi} \left(s _ {t}\right), \boldsymbol {a} _ {t}\right) = r _ {t} \left(s _ {t}, \boldsymbol {a} _ {t}\right) + \gamma H \left(f _ {\phi} \left(s _ {t + 1}\right)\right). \tag {2}
$$

Then, the loss function $L _ { \theta } ^ { E C }$ for training can be expressed as the weighted sum of one-step TD error and one-step TD memory error, i.e., Monte Carlo (MC) inference error, based on $Q _ { E C } ( \bar { f _ { \phi } } ( s _ { t } ) , \mathbf { a } _ { t } )$ .

$$
L _ {\theta} ^ {E C} = \left(y (s, \boldsymbol {a}) - Q _ {t o t} (s, \boldsymbol {a}; \theta)\right) ^ {2} + \lambda \left(Q _ {E C} \left(f _ {\phi} (s), \boldsymbol {a}\right) - Q _ {t o t} (s, \boldsymbol {a}; \theta)\right) ^ {2}, \tag {3}
$$

where $y ( s , \pmb { a } )$ is one-step TD target; $Q _ { t o t }$ is the joint Q-value function parameterized by $\theta$ ; and $\lambda$ is a scale factor.

Problem of the conventional episodic control with random projection Random projection is useful for dimensionality reduction as it preserves distance relationships, as demonstrated by the Johnson-Lindenstrauss lemma (Dasgupta & Gupta, 2003). However, a random projection adopted for $f _ { \phi } ( s )$ hardly has a semantic meaning in its embedding $x _ { t }$ , as it puts random weights on the state features without considering the patterns of determining the state returns. Additionally, when recalling the memory from $\mathcal { D } _ { E }$ , the projected state $x _ { t }$ can abruptly change even with a small change of $s _ { t }$ because the embedding is not being regulated by the return. This results in a sparse selection of semantically similar memories, i.e. similar states with similar or better rewards. As a result, conventional episodic control using random projection only recalls identical states and relies on its own Monte-Carlo (MC) return to regulate the one-step TD target inference, limiting exploration of nearby states on the embedding space.

The problem intensifies when the high-return states in the early training phase are indeed local optima. In such cases, the naive utilization of episodic control is prone to converge on local minima. As a result, for the super hard tasks of SMAC, EMC (Zheng et al., 2021) had to decrease the magnitude of this regularization to almost zero, i.e., not considering episodic memories anymore.

# 3 METHODOLOGY

This section introduces Efficient episodic Memory Utilization (EMU) (Figure 1). We begin by explaining how to construct (1) semantic memory embeddings to better utilize the episodic memory, which enables memory recall of similar, more promising states. To further improve memory utilization, as an alternative to the conventional episodic control, we propose (2) episodic incentive that selectively encourages desirable transitions while preventing local convergence towards undesirable trajectories.

![](images/9870dc8121ad3ae3c92e4d104e8a44b63265538c816e5f7c80c01b2d8aafa967.jpg)  
Figure 1: Overview of EMU framework.

# 3.1 SEMANTIC MEMORY EMBEDDING

Episodic Memory Construction To address the problems of a random projection adopted in episodic control, we propose a trainable embedding function $f _ { \phi } ( s )$ to learn the state embedding patterns affected by the highest return. The problem of a learnable embedding network $f _ { \phi }$ is that the match between $H ( f _ { \phi } ( s _ { t } ) )$ and $s _ { t }$ breaks whenever $f _ { \phi }$ is updated. Hence, we save the global state $s _ { t }$ as well as a pair of $H _ { t }$ and $x _ { t }$ in $\mathcal { D } _ { E }$ , so that we can update $x = f _ { \phi } ( s )$ whenever $f _ { \phi }$ is updated. In addition, we store the desirability $\xi$ of $s _ { t }$ according to Definition 1. Appendix E.1 illustrates the details of memory construction proposed by this paper.

Learning framework for State Embedding When training $f _ { \phi } ( s _ { t } )$ , it is critical to extract important features of a global state that affect its value, i.e., the highest return. Thus, we additionally adopt a decoder structure ${ \bar { H } } _ { t } = f _ { \psi } ( x _ { t } )$ to predict the highest return $H _ { t }$ of $s _ { t }$ . We call this embedding function as EmbNet, and its learning objective of $f _ { \phi }$ and $f _ { \psi }$ can be written as

$$
\mathcal {L} (\phi , \psi) = \left(H _ {t} - f _ {\psi} \left(f _ {\phi} \left(s _ {t}\right)\right)\right) ^ {2}. \tag {4}
$$

When constructing the embedding space, we found that an additional consideration of reconstruction of state s conditioned on timestep $t$ improves the quality of feature extraction and constitutes a smoother embedding space. To this end, we develop the deterministic conditional autoencoder (dCAE), and the corresponding loss function can be expressed as

$$
\mathcal {L} (\phi , \psi) = \left(H _ {t} - f _ {\psi} ^ {H} \left(f _ {\phi} (s _ {t} | t) | t)\right) ^ {2} + \lambda_ {r c o n} \| s _ {t} - f _ {\psi} ^ {s} \left(f _ {\phi} (s _ {t} | t) | t\right) \| _ {2} ^ {2}, \right. \tag {5}
$$

where $f _ { \psi } ^ { H }$ predicts the highest return; $f _ { \psi } ^ { s }$ reconstructs $s _ { t }$ ; $\lambda _ { \mathit { r c o n } }$ is a scale factor. Here, $f _ { \psi _ { . } } ^ { H }$ and $f _ { \psi _ { - } } ^ { s }$ share the lower part of networks as illustrated in Figure 1(c). Appendix C.1 presents the details of network structure of $f _ { \phi }$ and $f _ { \psi }$ , and Algorithm 1 in Appendix C.1 presents the learning framework for $f _ { \phi }$ and $f _ { \psi }$ . This training is conducted periodically in parallel to the RL policy learning on $Q _ { t o t } ( \cdot ; \theta )$ .

Figure 2 illustrates the result of t-SNE (Van der Maaten & Hinton, 2008) of 50K samples of $\boldsymbol { x } \in \mathcal { D } _ { E }$ out of 1M memory data in training for $3 s \_ { \mathrm { v s } } \_ { \mathrm { 5 z } }$ task of SMAC. Unlike supervised learning with label data, there is no label for each $x _ { t }$ . Thus, we mark $x _ { t }$ with its pair of the highest return $H _ { t }$ . Compared to a random projection in Figure 2(a), $x _ { t }$ via $f _ { \phi }$ is well-clustered, according to the similarity of the embedded state and its return. This clustering of $x _ { t }$ enables us to safely select

![](images/4c1ed93f4377368073726b60283f6cb608a2015ce628aac5e46a4f3bf27e141e.jpg)  
(a) Random Projection

![](images/2992e73928697aa7c02f394b24bd0030865e2a0c5953bef8f3b4b6b6b5b3c6be.jpg)  
(b) EmbNet

![](images/87e17cc53e6af4e31d4b5dd2714f9149504399c8ff7d4c9629564a8e55243bbe.jpg)  
(c) dCAE   
Figure 2: t-SNE of sampled embedding $\boldsymbol { x } \in \mathcal { D } _ { E }$ . Colors from red to purple (rainbow) represent from low return to high return.

episodic memories around the key state $s _ { t }$ , which constitutes efficient memory utilization. This memory utilization expedites learning speed as well as encourages exploration to a more promising state $\hat { s } _ { t }$ near $s _ { t }$ . Appendix F illustrates how to determine $\delta$ of Eq. 1 in a memory-efficient way.

# 3.2 EPISODIC INCENTIVE

With the learnable memory embedding for an efficient memory recall, how to use the selected memories still remains a challenge because a naive utilization of episodic memory is prone to converge on local minima. To solve this issue, we propose a new reward structure called episodic incentive $r ^ { p }$ by leveraging the desirability $\xi$ of states in $\mathcal { D } _ { E }$ . Before deriving the episodic incentive $r ^ { p }$ , we first need to understand the characteristics of episodic control. In this section, we denote the joint Q-function $Q _ { t o t } ( \cdot ; \theta )$ simply as $Q _ { \theta }$ for conciseness.

Theorem 1. Given a transition $( s , \pmb { a } , r , s ^ { \prime } )$ and $H ( x ^ { \prime } )$ , let $L _ { \theta }$ be the $Q$ -learning loss with additional transition reward, i.e., ${ \cal L } _ { \theta } : = ( y ( s , \pmb { a } ) + r ^ { E C } ( s , \pmb { a } , s ^ { \prime } ) - Q _ { t o t } ( s , \pmb { a } ; \theta ) ) ^ { 2 }$ where $r ^ { E C } ( s , { \pmb a } , s ^ { \prime } ) : =$ $\lambda ( r ( s , \pmb { a } ) + \gamma H ( x ^ { \prime } ) - Q _ { \theta } ( s , \pmb { a } ) )$ , then $\nabla _ { \theta } L _ { \theta } = \nabla _ { \theta } L _ { \theta } ^ { E C }$ . (Proof in Appendix B.1)

As Theorem 1 suggests, we can generate the same gradient signal as the episodic control by leveraging the additional transition reward $r ^ { E C } ( s , \pmb { a } , s ^ { \prime } )$ . However, $r ^ { \widecheck { E } C } ( s , \mathbf { { a } } , s ^ { \prime } )$ accompanies a risk of local convergence as discussed in Section 2.3. Therefore, instead of applying $r ^ { E C } ( s , \pmb { a } , s ^ { \prime } )$ , we propose the episodic incentive $r ^ { p } : = \gamma \hat { \eta } ( s ^ { \prime } )$ that provides an additional reward for the desirable transition $( s , \pmb { a } , r , s ^ { \prime } )$ , such that $\xi ( s ^ { \prime } ) = 1$ . Here, $\hat { \eta } ( s ^ { \prime } )$ estimates $\eta ^ { * } ( s ^ { \prime } )$ , which represents the difference between the true value $V ^ { * } ( s ^ { \prime } )$ of $s ^ { \prime }$ and the predicted value via target network $\operatorname* { m a x } _ { a ^ { \prime } } Q _ { \theta ^ { - } } ( s ^ { \prime } , { \pmb a } ^ { \prime } )$ , defined as

$$
\eta^ {*} \left(s ^ {\prime}\right) := V ^ {*} \left(s ^ {\prime}\right) - \max  _ {\boldsymbol {a} ^ {\prime}} Q _ {\theta -} \left(s ^ {\prime}, \boldsymbol {a} ^ {\prime}\right). \tag {6}
$$

Note that we do not know $V ^ { * } ( s ^ { \prime } )$ and subsequently $\eta ^ { * } ( s ^ { \prime } )$ . To accurately estimate $\eta ^ { * } ( s ^ { \prime } )$ with $\hat { \eta } ( s ^ { \prime } )$ , we use the expected value considering the current policy $\pi _ { \theta }$ as $\hat { \eta } ( s ^ { \prime } ) : = \mathbb { E } _ { \pi _ { \theta } } [ \eta ( s ^ { \prime } ) ]$ where $\dot { \eta } \in \left[ 0 , \eta _ { \mathrm { m a x } } ( s ^ { \prime } ) \right]$ for $s ^ { \prime } \sim P ( s ^ { \prime } | s , \pmb { a } \sim \pi _ { \theta } )$ . Here, $\eta _ { \mathrm { m a x } } ( s ^ { \prime } )$ can be reasonably approximated by using $H ( f _ { \phi } ( s ^ { \prime } ) )$ in $\mathcal { D } _ { E }$ . Then, with the count-based estimation $\hat { \eta } ( s ^ { \prime } )$ , episodic incentive $r ^ { p }$ can be expressed as

$$
r ^ {p} = \gamma \hat {\eta} \left(s ^ {\prime}\right) = \gamma \mathbb {E} _ {\pi_ {\theta}} \left[ \eta \left(s ^ {\prime}\right) \right] \simeq \gamma \frac {N _ {\xi} \left(s ^ {\prime}\right)}{N _ {\text {c a l l}} \left(s ^ {\prime}\right)} \eta_ {\max } \left(s ^ {\prime}\right) = \gamma \frac {N _ {\xi} \left(s ^ {\prime}\right)}{N _ {\text {c a l l}} \left(s ^ {\prime}\right)} \left(H \left(f _ {\phi} \left(s ^ {\prime}\right)\right) - \max  _ {a ^ {\prime}} Q _ {\theta -} \left(s ^ {\prime}, a ^ {\prime}\right)\right), \tag {7}
$$

where $N _ { c a l l } ( s ^ { \prime } )$ is the number of visits on $\hat { x } ^ { \prime } = \mathrm { N N } ( f _ { \phi } ( s ^ { \prime } ) ) \in \mathcal { D } _ { E }$ ; and $N _ { \xi }$ is the number of desirable transition from $\hat { x } ^ { \prime }$ . Here, $\operatorname { N N } ( \cdot )$ represents a function for selecting the nearest neighbor. From Theorem 1, the loss function adopting episodic control with an alternative transition reward $r ^ { p }$ instead of $r ^ { E C }$ can be expressed as

$$
L _ {\theta} ^ {p} = \left(r (s, \boldsymbol {a}) + r ^ {p} + \gamma \max  _ {\boldsymbol {a} ^ {\prime}} Q _ {\theta^ {-}} \left(s ^ {\prime}, \boldsymbol {a} ^ {\prime}\right) - Q _ {\theta} (s, \boldsymbol {a})\right) ^ {2}. \tag {8}
$$

Then, the gradient signal of the one-step TD inference loss $\nabla _ { \boldsymbol { \theta } } L _ { \boldsymbol { \theta } } ^ { p }$ with the episodic reward $r ^ { p } =$ $\gamma \hat { \eta } ( s ^ { \prime } )$ can be written as

$$
\nabla_ {\theta} L _ {\theta} ^ {p} = - 2 \nabla_ {\theta} Q _ {\theta} (s, a) (\Delta \varepsilon_ {T D} + r ^ {p}) = - 2 \nabla_ {\theta} Q _ {\theta} (s, a) (\Delta \varepsilon_ {T D} + \gamma \frac {N _ {\xi} \left(s ^ {\prime}\right)}{N _ {c a l l} \left(s ^ {\prime}\right)} \eta_ {\max } \left(s ^ {\prime}\right)), \tag {9}
$$

where $\begin{array} { r } { \Delta \varepsilon _ { T D } = r ( s , a ) + \gamma \mathrm { { m a x } } _ { a ^ { \prime } } Q _ { \theta ^ { - } } ( s ^ { \prime } , a ^ { \prime } ) - Q _ { \theta } ( s , a ) } \end{array}$ is one-step inference TD error. Here, the gradient signal $\nabla _ { \boldsymbol { \theta } } L _ { \boldsymbol { \theta } } ^ { p }$ with the proposed episodic reward $r ^ { p }$ can accurately estimate the optimal gradient signal as follows.

Theorem 2. Let $\nabla _ { \theta } L _ { \theta } ^ { * } = - 2 \nabla _ { \theta } Q _ { \theta } ( s , a ) ( \Delta \varepsilon _ { T D } ^ { * } )$ be the optimal gradient signal with the true one step $T D$ error $\Delta \varepsilon _ { T D } ^ { * } = r ( s , a ) + \gamma V ^ { * } ( s ^ { \prime } ) - Q _ { \theta } ( s , a )$ . Then, the gradient signal $\nabla _ { \boldsymbol { \theta } } L _ { \boldsymbol { \theta } } ^ { p }$ with the episodic incentive $r ^ { p }$ converges to the optimal gradient signal as the policy converges to the optimal policy $\pi _ { \theta } ^ { * }$ , i.e., $\nabla _ { \theta } L _ { \theta } ^ { p } \to \nabla _ { \theta } L _ { \theta } ^ { * }$ as $\pi _ { \boldsymbol { \theta } }  \pi _ { \boldsymbol { \theta } } ^ { * }$ . (Proof in Appendix B.2)

Theorem 2 also implies that there exists a certain bias in $\nabla _ { \boldsymbol { \theta } } L _ { \boldsymbol { \theta } } ^ { E C }$ as described in Appendix B.2. Besides the property of convergence to the optimal gradient signal presented in Theorem 2, the episodic incentive has the following additional characteristics. (1) The episodic incentive is only applied to the desirable transition. We can simply see that $r ^ { p } =$ $\gamma \hat { \eta } = \gamma \mathbb { E } _ { \pi _ { \theta } } [ \eta ] \simeq \dot { \gamma } \eta _ { m a x } N _ { \xi } / N _ { c a l l }$ and if $\xi ( s ^ { \prime } ) ~ = ~ 0$ then $N _ { \xi } ~ = ~ 0$ , yielding $r ^ { p } \to 0$ . Subsequently, (2) there is no need to adjust a scale factor by the task complexity. (3) The episodic incentive can reduce the risk of overestimation by considering the

![](images/fb3f0679a739b306fc6b69947e112e1eee3f833517bcb7d2b8ea82b15b8d759e.jpg)  
(a) 3s5z

![](images/0081420a8c26b388a90c2c7f97ee938001ccdec253dd89d5341ecd0aa0f46bd6.jpg)  
(b) MMM2   
Figure 3: Episodic incentive. Test trajectories are plotted on the embedded space with sampled memories in $\mathcal { D } _ { E }$ , denoted with dotted markers. Star markers and numbers represent the desirability of state and timestep in the episode, respectively. Color represents the same semantics as Figure 2.

expected value of $\dot { \mathbb { E } } _ { \pi _ { \theta } } [ \eta ]$ . Instead of considering the optimistic $\eta _ { m a x }$ , the count-based estimation $r ^ { p } = \gamma \hat { \eta } = \gamma { \mathbb E } _ { \pi _ { \theta } } [ \eta ]$ can consider the randomness of the policy $\pi _ { \theta }$ . Figure 3 illustrates how the episodic incentive works with the desirability stored in $\mathcal { D } _ { E }$ constructed by Algorithm 2 presented in Appendix E.1. In Figure 3 as we intended, high-value states (at small timesteps) are clustered close to the purple zone, while low-value states (at large timesteps) are located in the red zone.

# 3.3 OVERALL LEARNING OBJECTIVE

To construct the joint Q-function $Q _ { t o t }$ from individual $Q _ { i }$ of the agent $i$ , any form of mixer can be used. In this paper, we mainly adopt the mixer presented in QPLEX (Wang et al., 2020b) similar to Zheng et al. (2021), which guarantees the complete Individual-Global-Max (IGM) condition (Son et al., 2019; Wang et al., 2020b). Considering any intrinsic reward $r ^ { c }$ encouraging an exploration (Zheng et al., 2021) or diversity (Chenghao et al., 2021), the final loss function for the action policy learning from Eq. 8 can be extended as

$$
\mathcal {L} _ {\theta} ^ {p} = \left(r (s, \boldsymbol {a}) + r ^ {p} + \beta_ {c} r ^ {c} + \gamma \max  _ {\boldsymbol {a} ^ {\prime}} Q _ {t o t} \left(s ^ {\prime}, \boldsymbol {a} ^ {\prime}; \theta^ {-}\right) - Q _ {t o t} (s, \boldsymbol {a}; \theta)\right) ^ {2}, \tag {10}
$$

where $\beta _ { c }$ is a scale factor. Note that the episodic incentive $r ^ { p }$ can be used in conjunction with any form of intrinsic reward $r ^ { c }$ being properly annealed throughout the training. Again, $\theta$ denotes the parameters of networks related to action policy $Q _ { i }$ and the corresponding mixer network to generate $Q _ { t o t }$ . For the action selection via $Q$ , we adopt a GRU to encode a local action-observation history $\tau$ presented in 2.1 similar to Sunehag et al. (2017); Rashid et al. (2018); Wang et al. (2020b); but in Eq. 10, we denote equations with $s$ instead of $\tau$ for the coherence with derivation in the previous section. Appendix E.2 presents the overall training algorithm.

# 4 EXPERIMENTS

In this part, we have formulated our experiments with the intention of addressing the following inquiries denoted as Q1-3.

• Q1. How does EMU compare to the state-of-the-art MARL frameworks?   
• Q2. How does the proposed state embedding change the embedding space and improve the performance?   
• Q3. How does the episodic incentive improve performance?

We conduct experiments on complex multi-agent tasks such as SMAC (Samvelyan et al., 2019) and GRF (Kurach et al., 2020). The experiments compare EMU against EMC adopting episodic control (Zheng et al., 2021). Also, we include notable baselines, such as value-based MARL methods QMIX (Rashid et al., 2018), QPLEX (Wang et al., 2020b), CDS encouraging individual diversity (Chenghao et al., 2021). Particularly, we emphasize that EMU can be combined with any MARL framework, so we present two versions of EMU implemented on original QPLEX and CDS, denoted as EMU (QPLEX) and EMU (CDS), respectively. Appendix C provides further details of experiment settings and implementations, and Appendix D.12 provides the applicability of EMU to single-agent tasks, including pixel-based high-dimensional tasks.

# 4.1 Q1. COMPARATIVE EVALUATION ON STARCRAFT II (SMAC)

![](images/e50e99e26778c45c9aab5d15cfa2aa89716101b9216e4018867980fbe6bc5ff5.jpg)  
Figure 4: Performance comparison of EMU against baseline algorithms on three easy and hard SMAC maps: 1c3s5z, 3s_vs_5z, and 5m_vs_6m, and three super hard SMAC maps: MMM2, $6 \mathrm { { h } \_ { \nabla } \mathrm { { s } \_ { - } \mathrm { { 8 } } \mathrm { { z } } } }$ , and $3 \mathrm { s } 5 \mathrm { z \_ v s \_ } 3 \mathrm { s } 6 \mathrm { z }$ .

Figure 4 illustrates the overall performance of EMU on various SMAC maps. The map categorization regarding the level of difficulty follows the practice of Samvelyan et al. (2019). Thanks to the efficient memory utilization and episodic incentive, both EMU (QPLEX) and EMU (CDS) show significant performance improvement compared to their original methodologies. Especially, in super hard SMAC maps, the proposed method markedly expedites convergence on optimal policy.

# 4.2 Q1. COMPARATIVE EVALUATION ON GOOGLE RESEARCH FOOTBALL (GRF)

Here, we conduct experiments on GRF to further compare the performance of EMU with other baseline algorithms. In our GRF task, CDS and EMU (CDS) do not utilize the agent’s index on observation as they contain the prediction networks while other baselines (QMIX, EMC, QPLEX) use information of the agent’s identity in observations. In addition, we do not utilize any additional algorithm, such as prioritized experience replay (Schaul et al., 2015), for all baselines and our method, to expedite learning efficiency. From the experiments, adopting EMU achieves significant performance improvement, and EMU quickly finds the winning or scoring policy at the early learning phase by utilizing semantically similar memory.

![](images/41fee71f9b312baee3ac673af5d84fc1dd0366d172af052066b8ab50af1dc77a.jpg)  
Figure 5: Performance comparison of EMU against baseline algorithms on Google Research Football.

# 4.3 Q2. PARAMETRIC AND ABLATION STUDY

In this section, we examine how the key hyperparameter $\delta$ and the choice of design for $f _ { \phi }$ affect the performance. To compare the learning quality and performance more quantitatively, we propose a new performance index called overall win-rate, $\bar { \mu } _ { w }$ . The purpose of $\bar { \mu } _ { w }$ is to consider both training efficiency (speed) and quality (win-rate) for different seed cases (see Appendix D.1 for details). We conduct experiments on selected SMAC maps to measure $\bar { \mu } _ { w }$ according to $\delta$ and design choice for $f _ { \phi }$ such as (1) random projection, (2) EmbNet with Eq. 4 and (3) dCAE with Eq. 5.

![](images/ac54fc100a79f981267ad9604a14357e28e03f1875b34a5b39a43ec852d309cc.jpg)  
(a) 3s_vs_5z

![](images/c83b949faaad86e36e022b743e4a8c5ca9537194af5178b568e2408f4c63eb1c.jpg)  
(b) 5m_vs_6m

![](images/f31ffb770d4a9149b5218245ec818899fd4b5e71b61484a8845fe7740e34c312.jpg)  
(a) 3s_vs_5z

![](images/400f1b5e596b6e4c46285314ef579082d2f8e6064995cfab1c08ffb7064e38ef.jpg)  
(b) 5m_vs_6m   
Figure 6: $\bar { \mu } _ { w }$ according to $\delta$ and various design choices for $f _ { \phi }$ on SMAC maps.   
Figure 7: Final win-rate according to $\delta$ and various design choices for $f _ { \phi }$ on SMAC maps.

Figure 6 and Figure 7 show $\bar { \mu } _ { w }$ values and test win-rate at the end of training time according to different $\delta$ , presented in log-scale. To see the effect of design choice for $f _ { \phi }$ distinctly, we conduct experiments with the conventional episodic control. More data of $\bar { \mu } _ { w }$ is presented in Tables 4 and 5 in Appendix D.2. Figure 6 illustrates that dCAE structure shows the best training efficiency throughout various $\delta$ while achieving the optimal policy as other design choices as presented in Figure 7.

Interestingly, dCAE structure works well with a wider range of $\delta$ than EmbNet. We conjecture that EmbNet can select very different states as exploration if those states have similar return $H$ during training. This excessive memory recall adversely affects learning and fails to find an optimal policy as a result. See Appendix D.2 for detailed analysis and Appendix D.8 for an ablation study on the loss function of dCAE.

Even though a wide range of $\delta$ works well as in Figures 6 and 7, choosing a proper value of $\delta$ in

more difficult MARL tasks significantly improves the overall learning performance. Figure 8 shows the learning curve of EMU according to $\delta _ { 1 } \overset { \cdot } { = } 1 . 3 e ^ { - 7 }$ , $\delta _ { 2 } = 1 . 3 e ^ { - 5 }$ , $\bar { \delta } _ { 3 } ^ { ^ { \scriptsize - } } = 1 . 3 e ^ { - 3 }$ , and $\bar { \delta } _ { 4 } = 1 . 3 e ^ { - 2 }$ In super hard MARL tasks such as $6 \mathrm { h } _ { - } \mathrm { v } s _ { - } 8 z$ in SMAC and CA_hard in GRF, $\delta _ { 3 }$ shows the best performance compared to other $\delta$ values. This is consistent with the value suggested in Appendix F, where $\delta$ is determined in a memory-efficient way. Further parametric study on $\delta$ and $\lambda _ { \mathit { r c o n } }$ are presented in Appendix D.5 and D.6, respectively.

![](images/4c135490b74fde27740624a1422cb2cb35de397dc2729c8ae006f4d1d4a28c3a.jpg)  
(a) CA_hard (GRF)

![](images/b50e2d380ffaee6dc166c951d9f0b8f7514b3f3675a16551262d0d7960b10359.jpg)  
(b) 6h_vs_8z (SMAC)   
Figure 8: Effect of varying $\delta$ on complex MARL tasks.

# 4.4 Q3. FURTHER ABLATION STUDY

In this section, we carry out further ablation studies to see the effect of episodic incentive $r ^ { p }$ presented in Section 3.2. From EMU (QPLEX) and EMU (CDS), we ablate the episodic incentive and denote them with (No-EI). We additionally ablate embedding network $f _ { \phi }$ from EMU and denote them with (No-SE). In addition, we ablate both parts, yielding EMC (QPLEX-original) and CDS (QPLEXoriginal). We evaluate the performance of each model on super hard SMAC maps. Additional ablation studies on GRF maps are presented in Appendix D.7. Note that EMC (QPLEX-original) utilizes the conventional episodic control presented in Zheng et al. (2021).

Figure 9 illustrates that the episodic incentive largely affects learning performance. Especially, EMU (QPLEX-No-EI) and EMU (CDS-No-EI) utilizing the conventional episodic control show a large performance variation according to different seeds. This demonstrates that a naive utilization of episodic control could be detrimental to learning an optimal policy. On the other hand, the episodic incentive selectively encourages transition considering desirability and thus prevents such a local convergence. Appendix D.9 and D.10 present an additional ablation study on semantic embedding

![](images/e5fa57a752dacd0cf49e5c148170726fd0ee4f03917f8b6e5895654c7be783f2.jpg)  
(a) 6h_vs_8z SMAC

![](images/955bed5c892ea6d8236bd5f8fbd319820ba505ed36e5bfacee4390be05394a21.jpg)  
(b) 3s5z_vs_3s6z SMAC

![](images/aad74254809adf10e2acd936d433fd099637a61e89c16aecf1a93d296e1b206a.jpg)  
(c) 3s5z_vs_3s6z SMAC   
Figure 9: Ablation studies on episodic incentive via complex MARL tasks.

and $r ^ { c }$ , respectively. In addition, Appendix D.11 presents a comparison with an alternative incentive (Henaff et al., 2022) presented in a single-agent setting.

# 4.5 QUALITATIVE ANALYSIS AND VISUALIZATION

In this section, we conduct analysis with visualization to check how the desirability $\xi$ is memorized in $\mathcal { D } _ { E }$ and whether it conveys correct information. Figure 10 illustrates two test scenarios with different seeds, and each snapshot is denoted with a corresponding timestep. In Figure 11, the trajectory of each episode is projected onto the embedded space of $\mathcal { D } _ { E }$ .

In Figure 10, case (a) successfully defeated all enemies, whereas case (b) lost the engagement. Both cases went through a similar, desirable trajectory at the beginning. For example, until $t = 1 0$ agents in both cases focused on killing one enemy and kept all ally agents alive at the same time. However, at $t = 1 2$ , case (b) lost one agent, and two trajectories of case (a) and (b) in embedded space began to bifurcate. Case (b) still had a chance to win around $t = 1 4 \sim 1 6$ . However,

![](images/bbd2bb63b6a425378864eaa27229777d83d8159c99e5e4d388d992768497ef6a.jpg)

![](images/8f89b21ca7f9abb26fbeaa4444b5af5ec608369f6167a82a1cfe665e8b20ee09.jpg)  
(a) Desirable trajectory on 5m_vs_6m SMAC map   
(b) Undesirable trajectory on 5m_vs_6m SMAC map   
Figure 10: Visualization of test episodes.

the states became undesirable (denoted without star marker) after losing three ally agents around $t = 2 0$ , and case (b) lost the battle as a result. These sequences and characteristics of trajectories are well captured by desirability $\xi$ in $\mathcal { D } _ { E }$ as illustrated in Figure 11.

Furthermore, the desirable state denoted with $\xi = 1$ encourages exploration around it though it is not directly retrieved during batch sampling. This occurs through the propagation of its desirability to states currently distinguished as undesirable during memory construction, using Algorithm 2 in Appendix E.1. Consequently, when the state’s desirability is precisely memorized in $\mathcal { D } _ { E }$ , it can encourage desirable transitions through the episodic incentive $r ^ { p }$ .

![](images/f813354b7c765ef40e3fd680f513e717d92e988851d6a0f5253136bf43c0e27f.jpg)  
(a) Desirable trajectory

![](images/bd07901e36b9ba2a8389c056e161ad5841b26f06167d527150e107f31a986746.jpg)  
(b) Undesirable trajectory   
Figure 11: Test trajectories on embedded space of $\mathcal { D } _ { E }$ .

# 5 CONCLUSION

This paper presents EMU, a new framework to efficiently utilize episodic memory for cooperative MARL. EMU introduces two major components: 1) a trainable semantic embedding and 2) an episodic incentive utilizing desirability of state. Semantic memory embedding allows us to safely utilize similar memory in a wide area, expediting learning via exploratory memory recall. The proposed episodic incentive selectively encourages desirable transitions and reduces the risk of local convergence by leveraging the desirability of the state. As a result, there is no need for manual hyperparameter tuning according to the complexity of tasks, unlike conventional episodic control. Experiments and ablation studies validate the effectiveness of each component of EMU.

# ACKNOWLEDGEMENTS

This research was supported by AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data(IITP) funded by the Ministry of Science and ICT(2022-0-00077).

# REFERENCES

Marc Bellemare, Sriram Srinivasan, Georg Ostrovski, Tom Schaul, David Saxton, and Remi Munos. Unifying count-based exploration and intrinsic motivation. Advances in neural information processing systems, 29, 2016.   
Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of Artificial Intelligence Research, 47: 253–279, 2013.   
Charles Blundell, Benigno Uria, Alexander Pritzel, Yazhe Li, Avraham Ruderman, Joel Z Leibo, Jack Rae, Daan Wierstra, and Demis Hassabis. Model-free episodic control. arXiv preprint arXiv:1606.04460, 2016.   
Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018.   
Li Chenghao, Tonghan Wang, Chengjie Wu, Qianchuan Zhao, Jun Yang, and Chongjie Zhang. Celebrating diversity in shared multi-agent reinforcement learning. Advances in Neural Information Processing Systems, 34:3991–4002, 2021.   
Nikolay Dandanov, Hussein Al-Shatri, Anja Klein, and Vladimir Poulkov. Dynamic self-optimization of the antenna tilt for best trade-off between coverage and capacity in mobile networks. Wireless Personal Communications, 92(1):251–278, 2017.   
Sanjoy Dasgupta and Anupam Gupta. An elementary proof of a theorem of johnson and lindenstrauss. Random Structures & Algorithms, 22(1):60–65, 2003.   
Marc-André Dittrich and Silas Fohlmeister. Cooperative multi-agent system for production control using reinforcement learning. CIRP Annals, 69(1):389–392, 2020.   
Yali Du, Lei Han, Meng Fang, Ji Liu, Tianhong Dai, and Dacheng Tao. Liir: Learning individual intrinsic reward in multi-agent reinforcement learning. Advances in Neural Information Processing Systems, 32, 2019.   
Scott Fujimoto, Herke Hoof, and David Meger. Addressing function approximation error in actorcritic methods. In International conference on machine learning, pp. 1587–1596. PMLR, 2018.   
Jayesh K Gupta, Maxim Egorov, and Mykel Kochenderfer. Cooperative multi-agent control using deep reinforcement learning. In International conference on autonomous agents and multiagent systems, pp. 66–83. Springer, 2017.   
Matthew Hausknecht and Peter Stone. Deep recurrent q-learning for partially observable mdps. In 2015 aaai fall symposium series, 2015.   
Mikael Henaff, Roberta Raileanu, Minqi Jiang, and Tim Rocktäschel. Exploration via elliptical episodic bonuses. Advances in Neural Information Processing Systems, 35:37631–37646, 2022.   
Rein Houthooft, Xi Chen, Yan Duan, John Schulman, Filip De Turck, and Pieter Abbeel. Vime: Variational information maximizing exploration. Advances in neural information processing systems, 29, 2016.   
Hao Hu, Jianing Ye, Guangxiang Zhu, Zhizhou Ren, and Chongjie Zhang. Generalizable episodic memory for deep reinforcement learning. International conference on machine learning, 2021.

Natasha Jaques, Angeliki Lazaridou, Edward Hughes, Caglar Gulcehre, Pedro Ortega, DJ Strouse, Joel Z Leibo, and Nando De Freitas. Social influence as intrinsic motivation for multi-agent deep reinforcement learning. In International conference on machine learning, pp. 3040–3049. PMLR, 2019.   
Hyoungseok Kim, Jaekyeom Kim, Yeonwoo Jeong, Sergey Levine, and Hyun Oh Song. Emi: Exploration with mutual information. arXiv preprint arXiv:1810.01176, 2018.   
Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.   
Karol Kurach, Anton Raichuk, Piotr Stanczyk, Michal Zajkc, Olivier Bachem, Lasse Espeholt, Carlos Riquelme, Damien Vincent, Marcin Michalski, Olivier Bousquet, et al. Google research football: A novel reinforcement learning environment. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pp. 4501–4510, 2020.   
Lei Le, Andrew Patterson, and Martha White. Supervised autoencoders: Improving generalization performance with unsupervised regularizers. Advances in neural information processing systems, 31, 2018.   
Máté Lengyel and Peter Dayan. Hippocampal contributions to control: the third way. Advances in neural information processing systems, 20, 2007.   
Zichuan Lin, Tianqi Zhao, Guangwen Yang, and Lintao Zhang. Episodic memory deep q-networks. arXiv preprint arXiv:1805.07603, 2018.   
Iou-Jen Liu, Unnat Jain, Raymond A Yeh, and Alexander Schwing. Cooperative exploration for multi-agent deep reinforcement learning. In International Conference on Machine Learning, pp. 6826–6836. PMLR, 2021.   
Anuj Mahajan, Tabish Rashid, Mikayel Samvelyan, and Shimon Whiteson. Maven: Multi-agent variational exploration. Advances in Neural Information Processing Systems, 32, 2019.   
David Henry Mguni, Taher Jafferjee, Jianhong Wang, Nicolas Perez-Nieves, Oliver Slumbers, Feifei Tong, Yang Li, Jiangcheng Zhu, Yaodong Yang, and Jun Wang. Ligs: Learnable intrinsic-reward generation selection for multi-agent learning. arXiv preprint arXiv:2112.02618, 2021.   
Shakir Mohamed and Danilo Jimenez Rezende. Variational information maximisation for intrinsically motivated reinforcement learning. Advances in neural information processing systems, 28, 2015.   
Frans A Oliehoek and Christopher Amato. A concise introduction to decentralized POMDPs. Springer, 2016.   
Frans A Oliehoek, Matthijs TJ Spaan, and Nikos Vlassis. Optimal and approximate q-value functions for decentralized pomdps. Journal of Artificial Intelligence Research, 32:289–353, 2008.   
Georg Ostrovski, Marc G Bellemare, Aäron Oord, and Rémi Munos. Count-based exploration with neural density models. In International conference on machine learning, pp. 2721–2730. PMLR, 2017.   
Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning, pp. 2778–2787. PMLR, 2017.   
Alexander Pritzel, Benigno Uria, Sriram Srinivasan, Adria Puigdomenech Badia, Oriol Vinyals, Demis Hassabis, Daan Wierstra, and Charles Blundell. Neural episodic control. In International Conference on Machine Learning, pp. 2827–2836. PMLR, 2017.   
Santhosh K Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alex Clegg, John Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X Chang, et al. Habitatmatterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai. arXiv preprint arXiv:2109.08238, 2021.

Tabish Rashid, Mikayel Samvelyan, Christian Schroeder, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning. In International conference on machine learning, pp. 4295–4304. PMLR, 2018.   
Tabish Rashid, Gregory Farquhar, Bei Peng, and Shimon Whiteson. Weighted qmix: Expanding monotonic value function factorisation for deep multi-agent reinforcement learning. Advances in neural information processing systems, 33:10199–10210, 2020.   
Mikayel Samvelyan, Tabish Rashid, Christian Schroeder De Witt, Gregory Farquhar, Nantas Nardelli, Tim GJ Rudner, Chia-Man Hung, Philip HS Torr, Jakob Foerster, and Shimon Whiteson. The starcraft multi-agent challenge. arXiv preprint arXiv:1902.04043, 2019.   
Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver. Prioritized experience replay. arXiv preprint arXiv:1511.05952, 2015.   
Kihyuk Sohn, Honglak Lee, and Xinchen Yan. Learning structured output representation using deep conditional generative models. Advances in neural information processing systems, 28, 2015.   
Kyunghwan Son, Daewoo Kim, Wan Ju Kang, David Earl Hostallero, and Yung Yi. Qtran: Learning to factorize with transformation for cooperative multi-agent reinforcement learning. In International conference on machine learning, pp. 5887–5896. PMLR, 2019.   
Bradly C Stadie, Sergey Levine, and Pieter Abbeel. Incentivizing exploration in reinforcement learning with deep predictive models. arXiv preprint arXiv:1507.00814, 2015.   
Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinicius Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z Leibo, Karl Tuyls, et al. Value-decomposition networks for cooperative multi-agent learning. arXiv preprint arXiv:1706.05296, 2017.   
Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2018.   
Haoran Tang, Rein Houthooft, Davis Foote, Adam Stooke, OpenAI Xi Chen, Yan Duan, John Schulman, Filip DeTurck, and Pieter Abbeel. # exploration: A study of count-based exploration for deep reinforcement learning. Advances in neural information processing systems, 30, 2017.   
Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026–5033. IEEE, 2012. doi: 10.1109/IROS.2012.6386109.   
Laurens Van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. Journal of machine learning research, 9(11), 2008.   
Binyu Wang, Zhe Liu, Qingbiao Li, and Amanda Prorok. Mobile robot path planning in dynamic environments through globally guided reinforcement learning. IEEE Robotics and Automation Letters, 5(4):6932–6939, 2020a.   
Jianhao Wang, Zhizhou Ren, Terry Liu, Yang Yu, and Chongjie Zhang. Qplex: Duplex dueling multi-agent q-learning. arXiv preprint arXiv:2008.01062, 2020b.   
Tonghan Wang, Jianhao Wang, Yi Wu, and Chongjie Zhang. Influence-based multi-agent exploration. arXiv preprint arXiv:1910.05512, 2019.   
Tonghan Wang, Tarun Gupta, Anuj Mahajan, Bei Peng, Shimon Whiteson, and Chongjie Zhang. Rode: Learning roles to decompose multi-agent tasks. In Proceedings of the International Conference on Learning Representations (ICLR), 2021.   
Marco A Wiering et al. Multi-agent reinforcement learning for traffic light control. In Machine Learning: Proceedings of the Seventeenth International Conference (ICML’2000), pp. 1151–1158, 2000.   
Jiachen Yang, Igor Borovikov, and Hongyuan Zha. Hierarchical cooperative multi-agent reinforcement learning with skill discovery. arXiv preprint arXiv:1912.03558, 2019.

Yaodong Yang, Jianye Hao, Ben Liao, Kun Shao, Guangyong Chen, Wulong Liu, and Hongyao Tang. Qatten: A general framework for cooperative multiagent reinforcement learning. arXiv preprint arXiv:2002.03939, 2020.   
Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. The surprising effectiveness of ppo in cooperative multi-agent games. Advances in Neural Information Processing Systems, 35:24611–24624, 2022.   
Lulu Zheng, Jiarui Chen, Jianhao Wang, Jiamin He, Yujing Hu, Yingfeng Chen, Changjie Fan, Yang Gao, and Chongjie Zhang. Episodic multi-agent reinforcement learning with curiosity-driven exploration. Advances in Neural Information Processing Systems, 34:3757–3769, 2021.   
Guangxiang Zhu, Zichuan Lin, Guangwen Yang, and Chongjie Zhang. Episodic reinforcement learning with associative memory. International conference on learning representations, 2020.

# A RELATED WORKS

This section presents the related works regarding incentive generation for exploration, episodic control, and the characteristics of cooperative MARL.

# A.1 INCENTIVE FOR MULTI-AGENT EXPLORATION

Balancing between exploration and exploitation in policy learning is a paramount issue in reinforcement learning. To encourage exploration, modified count-based methods (Bellemare et al., 2016; Ostrovski et al., 2017; Tang et al., 2017), prediction error-based methods (Stadie et al., 2015; Pathak et al., 2017; Burda et al., 2018; Kim et al., 2018), and information gain-based methods (Mohamed & Jimenez Rezende, 2015; Houthooft et al., 2016) have been proposed for a single agent reinforcement learning. In most cases, an incentive for exploration is introduced as an additional reward to a TD target in Q-learning; or such an incentive is added as a regularizer for overall loss functions. Recently, various aforementioned methods to encourage exploration have been adopted to the multi-agent setting (Mahajan et al., 2019; Wang et al., 2019; Jaques et al., 2019; Mguni et al., 2021) and have shown their effectiveness. MAVEN (Mahajan et al., 2019) introduces a regularizer maximizing the mutual information between trajectories and latent variables to learn a diverse set of behaviors. LIIR (Du et al., 2019) learns a parameterized individual intrinsic reward function by maximizing a centralized critic. CDS (Chenghao et al., 2021) proposes a novel information-theoretical objective to maximize the mutual information between agents’ identities and trajectories to encourage diverse individualized behaviors. EMC (Zheng et al., 2021) proposes a curiosity-driven exploration by predicting individual Q-values. This individual-based Q-value prediction can capture the influence among agents as well as the novelty of states.

# A.2 EPISODIC CONTROL

Episodic control (Lengyel & Dayan, 2007) was well adopted on model-free setting (Blundell et al., 2016) by storing the highest return of a given state, to efficiently estimate its values or Q-values. Given that the sample generation is often limited by simulation executions or real-world observations, its sample efficiency helps to find an accurate estimation of Q-value (Blundell et al., 2016; Pritzel et al., 2017; Lin et al., 2018). NEC (Pritzel et al., 2017) uses a differentiable neural dictionary as an episodic memory to estimate the action value by the weighted sum of the values in the memory. EMDQN (Lin et al., 2018) utilizes a fixed random matrix to generate a state representation, which is used as a key to link between the state representation and the highest return of the state in the episodic memory. ERLAM (Zhu et al., 2020) learns associative memories by building a graphical representation of states in memory, and GEM (Hu et al., 2021) develops state-action values of episodic memory in a generalizable manner. Recently, EMC (Zheng et al., 2021) extended the approach of EMDQN to a deep MARL with curiosity-driven exploration incentives. EMC utilizes episodic memory to regularize policy learning and shows performance improvement in cooperative MARL tasks. However, EMC requires a hyperparameter tuning to determine the level of importance of the one-step TD memory-based target during training, according to the difficulties of tasks. In this paper, we interpret this regularization as an additional transition reward. Then, we present a novel form of reward, called episodic incentive, to selectively encourage the transition toward desired states, i.e., states toward a common goal in cooperative multi-agent tasks.

# A.3 COOPERATIVE MULTI-AGENT REINFORCEMENT LEARNING (MARL) TASK

In general, there is a common goal in cooperative MARL tasks, which guarantees the maximum return that can be obtained from the environment. Thus, there could be many local optima with high returns but not the maximum, which means the agents failed to achieve the common goal in the end. In other words, there is a distinct difference between the objective of cooperative MARL tasks and that of a single-agent task, which aims to maximize the return as much as possible without any boundary determining success or failure. Our desirability definition presented in Definition 1 in MARL setting becomes well justified from this view. Under this characteristic of MARL tasks, learning optimal policy often takes a long time and even fails, yielding a local convergence. EMU was designed to alleviate these issues in MARL.

# B MATHEMATICAL PROOF

In this section, we present the omitted proofs of Theorem 1 and Theorem 2 as follows.

# B.1 PROOF OF THEOREM 1

Proof. The loss function of a conventional episodic control, $L _ { \theta } ^ { E C }$ , can be expressed as the weighted sum of one-step inference TD error $\begin{array} { r } { \Delta \varepsilon _ { T D } = r ( s , a ) + \gamma \mathrm { m a x } _ { a ^ { \prime } } Q _ { \theta ^ { - } } ( s ^ { \prime } , \bar { a ^ { \prime } } ) - Q _ { \theta } ( s , a ) } \end{array}$ and MC inference error $\Delta \varepsilon _ { E C } = Q _ { E C } ( s , a ) - Q _ { \theta } ( s , a )$ .

$$
L _ {\theta} ^ {E C} = \left(r (s, \boldsymbol {a}) + \gamma \max  _ {\boldsymbol {a} ^ {\prime}} Q _ {\theta^ {-}} \left(s ^ {\prime}, \boldsymbol {a} ^ {\prime}\right) - Q _ {\theta} (s, \boldsymbol {a})\right) ^ {2} + \lambda \left(Q _ {E C} (s, \boldsymbol {a}) - Q _ {\theta} (s, \boldsymbol {a})\right) ^ {2}, \tag {11}
$$

where $Q _ { E C } ( s , \pmb { a } ) = r ( s , \pmb { a } ) + \gamma H ( s ^ { \prime } )$ and $Q _ { \theta ^ { - } }$ is the target network parameterized by $\theta ^ { - }$ . Then, the gradient of $L _ { \theta } ^ { E C }$ can be derived as

$$
\begin{array}{l} \nabla_ {\theta} L _ {\theta} ^ {E C} = - 2 \nabla_ {\theta} Q _ {\theta} (s, \pmb {a}) [ (r (s, \pmb {a}) + \gamma \max _ {a ^ {\prime}} Q _ {\theta^ {-}} (s ^ {\prime}, \pmb {a} ^ {\prime}) - Q _ {\theta} (s, \pmb {a})) + \lambda (Q _ {E C} (s, \pmb {a}) - Q _ {\theta} (s, \pmb {a})) ] \\ = - 2 \nabla_ {\theta} Q _ {\theta} (s, \boldsymbol {a}) \left(\Delta \varepsilon_ {T D} + \lambda \Delta \varepsilon_ {E C}\right). \tag {12} \\ \end{array}
$$

Now, we consider an additional reward $r ^ { E C }$ for the transition to a conventional Q-learning objective, the modified loss function $L _ { \theta }$ can be expressed as

$$
L _ {\theta} = \left(r (s, \boldsymbol {a}) + r ^ {E C} \left(s, \boldsymbol {a}, s ^ {\prime}\right) + \gamma \max  _ {\boldsymbol {a} ^ {\prime}} Q _ {\theta^ {-}} \left(s ^ {\prime}, \boldsymbol {a} ^ {\prime}\right) - Q _ {\theta} \left(s, \boldsymbol {a}\right)\right) ^ {2}. \tag {13}
$$

Then, the gradient of $L _ { \theta }$ presented in Eq. 13 is computed as

$$
\nabla_ {\theta} L _ {\theta} = - 2 \nabla_ {\theta} Q _ {\theta} (s, \boldsymbol {a}) \left(\Delta \varepsilon_ {T D} + r ^ {E C}\right). \tag {14}
$$

Comparing Eq. 12 and Eq. 14, if we set the additional transition reward as $r ^ { E C } ( s , { \pmb a } , s ^ { \prime } ) =$ $\lambda ( r ( \bar { s } , \pmb { a } ) \bar { + } \gamma \bar { H } ( s ^ { \prime } ) - Q _ { \theta } ( \bar { s , } \pmb { a } ) )$ , then $\nabla _ { \theta } L _ { \theta } = \nabla _ { \theta } L _ { \theta } ^ { E C }$ holds.

# B.2 PROOF OF THEOREM 2

Proof. From Eq. 7, the value of $\hat { \eta } ( s ^ { \prime } )$ can be expressed as

$$
\hat {\eta} \left(s ^ {\prime}\right) = \mathbb {E} _ {\pi_ {\theta}} \left[ \eta \left(s ^ {\prime}\right) \right] \simeq \frac {N _ {\xi} \left(s ^ {\prime}\right)}{N _ {c a l l} \left(s ^ {\prime}\right)} \left(H \left(f _ {\phi} \left(s ^ {\prime}\right)\right) - \max  _ {a ^ {\prime}} Q _ {\theta^ {-}} \left(s ^ {\prime}, a ^ {\prime}\right)\right) ]. \tag {15}
$$

When the joint actions from the current time follow the optimal policy, ${ \mathbf { } } a \sim \pi _ { \theta } ^ { * }$ , the cumulative reward from $s ^ { \prime }$ converges to $V ^ { * } ( s ^ { \prime } )$ , i.e., $H ( f _ { \phi } ( s ^ { \prime } ) )  V ^ { * } ( s ^ { \prime } )$ . Then, every recall of $\hat { x } ^ { \prime } = \mathrm { N N } ( f _ { \phi } ( s ^ { \prime } ) ) \in$ $\mathcal { D } _ { E }$ guarantees the desirable transition, i.e., $\xi ( s ^ { \prime } ) = 1$ , where $\operatorname { N N } ( \cdot )$ represents a function for selecting the nearest neighbor. As a result, as Ncall(s′) → ∞, Nξ(s′)Ncall(s′) $N _ { c a l l } ( s ^ { \prime } )  \infty$ $\frac { N _ { \xi } ( s ^ { \prime } ) } { N _ { c a l l } ( s ^ { \prime } ) }  1$ , yielding $\hat { \eta } ( s ^ { \prime } ) \simeq$ $\begin{array} { r } { \frac { N _ { \xi } ( s ^ { \prime } ) } { N _ { c a l l } ( s ^ { \prime } ) } \big ( H ( f _ { \phi } ( s ^ { \prime } ) ) - \operatorname* { m a x } _ { a ^ { \prime } } Q _ { \theta ^ { - } } ( s ^ { \prime } , a ^ { \prime } ) \big ) \to V ^ { * } ( s ^ { \prime } ) - \operatorname* { m a x } _ { a ^ { \prime } } Q _ { \theta ^ { - } } ( s ^ { \prime } , a ^ { \prime } ) } \end{array}$ . Then, the gradient signal with the episodic incentive $\nabla _ { \boldsymbol { \theta } } L _ { \boldsymbol { \theta } } ^ { p }$ becomes

$$
\begin{array}{l} \nabla_ {\theta} L _ {\theta} ^ {p} = - 2 \nabla_ {\theta} Q _ {\theta} (s, \boldsymbol {a}) [ \Delta \varepsilon_ {T D} + r ^ {p} ] \\ = - 2 \nabla_ {\theta} Q _ {\theta} (s, \boldsymbol {a}) \left[ \Delta \varepsilon_ {T D} + \gamma \hat {\eta} \left(s ^ {\prime}\right) \right] \\ \simeq - 2 \nabla_ {\theta} Q _ {\theta} (s, \pmb {a}) [ \Delta \varepsilon_ {T D} + \gamma (V ^ {*} (s ^ {\prime}) - \max _ {\pmb {a} ^ {\prime}} Q _ {\theta^ {-}} (s ^ {\prime}, \pmb {a} ^ {\prime})) ] \\ = - 2 \nabla_ {\theta} Q _ {\theta} (s, \boldsymbol {a}) \left[ r (s, \boldsymbol {a}) + \gamma \max  _ {\boldsymbol {a} ^ {\prime}} Q _ {\theta^ {-}} \left(s ^ {\prime}, \boldsymbol {a} ^ {\prime}\right) - Q _ {\theta} (s, \boldsymbol {a}) + \gamma \left(V ^ {*} \left(s ^ {\prime}\right) - \max  _ {\boldsymbol {a} ^ {\prime}} Q _ {\theta^ {-}} \left(s ^ {\prime}, \boldsymbol {a} ^ {\prime}\right)\right) \right] \\ = - 2 \nabla_ {\theta} Q _ {\theta} (s, \boldsymbol {a}) [ r (s, \boldsymbol {a}) + \gamma V ^ {*} \left(s ^ {\prime}\right) - Q _ {\theta} (s, \boldsymbol {a}) ] \\ = \nabla_ {\theta} L _ {\theta} ^ {*}, \tag {16} \\ \end{array}
$$

which completes the proof.

In addition, when $\operatorname* { m a x } _ { a ^ { \prime } } Q _ { \theta ^ { - } } \mathopen { } \mathclose \bgroup \left( s ^ { \prime } , { } a ^ { \prime } \aftergroup \egroup \right)$ accurately estimates $V ^ { * } ( s ^ { \prime } )$ , the original TD-target is preserved as the episodic incentive becomes zero, i.e., $r ^ { p } \to 0$ . Then with the properly annealed intrinsic reward $r ^ { c }$ , the learning objective presented in Eq. 10 degenerates to the original Bellman optimality equation (Sutton & Barto, 2018). On the other hand, even though the assumption of $H ( s ^ { \prime } ) \to V ^ { * } ( s ^ { \prime } )$ yields $\Delta \varepsilon _ { E C }  \Delta \varepsilon _ { T D } ^ { * }$ , $\nabla _ { \boldsymbol { \theta } } L _ { \boldsymbol { \theta } } ^ { E C }$ has an additional bias $\Delta \varepsilon _ { T D }$ due to weighted sum structure presented in Eq. 3. Thus, $\nabla _ { \boldsymbol { \theta } } L _ { \boldsymbol { \theta } } ^ { E C }$ can converge to $\nabla _ { \boldsymbol { \theta } } L _ { \boldsymbol { \theta } } ^ { * }$ only when $\begin{array} { r } { \operatorname* { m a x } _ { \pmb { a } ^ { \prime } } Q _ { \theta ^ { - } } ( s ^ { \prime } , \pmb { a } ^ { \prime } )  V ^ { * } ( s ^ { \prime } ) } \end{array}$ and $\lambda  0$ at the same time.

# C IMPLEMENTATION AND EXPERIMENT DETAILS

# C.1 DETAILS OF IMPLEMENTATION

# Encoder and Decoder Structure

As illustrated in Figure 1(c), we have an encoder and decoder structure. For an encoder $f _ { \phi }$ , we evaluate two types of structure, EmbNet and dCAE. For EmbNet with the learning objective presented in Eq. 4, two fully connected layers with 64-dimensional hidden state are used with ReLU activation function between them, followed by layer normalization block at the head. On the other hand, for dCAE with the learning objective presented in Eq. 5, we utilize a deeper encoder structure which contains three fully connected layers with ReLU activation function. In addition, dCAE takes a timestep $t$ as an input as well as a global state $s _ { t }$ . We set episodic latent dimension $\mathrm { d i m } ( x ) = 4$ as Zheng et al. (2021).

![](images/154d7b45ada91bac2318699776728eb9f50ffa1702fce57c222797f329465fbb.jpg)  
(a) EmbNet

![](images/dbf790fbbb3bdf0c8182d2e2b3c6de172fe212070cbb5cc2d1bbbef33c17f90f.jpg)  
(b) dCAE   
Figure 12: Illustration of network structures.

For a decoder $f _ { \psi }$ , both EmbNet and dCAE utilize a three-fully connected layer with ReLU activation functions. Differences are that EmbNet takes only $x _ { t }$ as input and utilizes the 128-dimensional hidden state while dCAE takes $x _ { t }$ and $t$ as inputs and adopts the 64-dimensional hidden state. As illustrated in Figure 1(c), to reconstruct global state $s _ { t }$ , dCAE has two separate heads while sharing lower parts of networks; $f _ { \phi } ^ { s }$ to reconstruct $s _ { t }$ and $f _ { \phi } ^ { H }$ to predict the return of $s _ { t }$ , denoted as $H _ { t }$ . Figure 12 illustrates network structures of EmbNet and dCAE. The concept of supervised VAE similar to EMU can be found in (Le et al., 2018).

The reason behind avoiding probabilistic autoencoders such as variational autoencoder (VAE) (Kingma & Welling, 2013; Sohn et al., 2015) is that the stochastic embedding and the prior distribution could have an adverse impact on preserving a pair of $x _ { t }$ and $H _ { t }$ for given a $s _ { t }$ . In particular, with stochastic embedding, a fixed $s _ { t }$ can generate diverse $x _ { t }$ . As a result, it breaks the pair of $x _ { t }$ and $H _ { t }$ for given $s _ { t }$ , which makes it difficult to select a valid memory from $\mathcal { D } _ { E }$ .

For training, we periodically update $f _ { \phi }$ and $f _ { \psi }$ with an update interval of $t _ { e m b }$ in parallel to MARL training. At each training phase, we use $M _ { e m b }$ samples out of the current capacity of $\mathcal { D } _ { E }$ , whose maximum capacity is 1 million (1M), and batch size of $m _ { e m b }$ is used for each training step. After updating $f _ { \phi }$ , every $x \in \mathcal { D } _ { E }$ needs to be updated with updated $f _ { \phi }$ . Algorithm 1 shows the details of learning framework for $f _ { \phi }$ and $f _ { \psi }$ . Details of the training procedure for $f _ { \phi }$ and $f _ { \psi }$ along with MARL training are presented in Appendix E.2.

# Other Network Structure and Hyperparameters

For a mixer structure, we adopt QPLEX (Wang et al., 2020b) in both EMU (QPLEX) and EMU (CDS) and follow the same hyperparameter settings used in their source codes. Common hyperparameters related to individual Q-network and MARL training are adopted by the default settings of PyMARL (Samvelyan et al., 2019).

Algorithm 1 Training Algorithm for State Embedding   
1: Parameter: learning rate $\alpha$ , number of training dataset $N$ , batch size $B$ 2: Sample Training dataset $(s^{(i)}, H^{(i)}, t^{(i)})_{i=1}^{N} \sim \mathcal{D}_E$ ,  
3: Initialize weights $\phi, \psi \gets 0$ 4: for $i = 1$ to $\lfloor N / B \rfloor$ do  
5: Compute $(x^{(j)} = f_{\phi}(s^{(j)}|t^{(j)})_{j=(i-1)B+1}^{iB}$ 6: Predict return $(\bar{H}^{(j)} = f_{\psi}^H(x^{(j)}|t^{(i)}))_{j=(i-1)B+1}^{iB}$ 7: Reconstruct state $(\bar{s}^{(j)} = f_{\psi}^s(x^{(j)}|t^{(i)}))_{j=(i-1)B+1}^{iB}$ 8: Compute Loss $\mathcal{L}(\phi, \psi)$ via Eq. 5  
9: Update $\phi \gets \phi - \alpha \frac{\partial \mathcal{L}}{\partial \phi}, \psi \gets \psi - \alpha \frac{\partial \mathcal{L}}{\partial \psi}$ 10: end for

# C.2 EXPERIMENT DETAILS

We utilize PyMARL (Samvelyan et al., 2019) to execute all of the baseline algorithms with their open-source codes, and the same hyperparameters are used for experiments if they are presented either in uploaded codes or in their manuscripts.

For a general performance evaluation, we test our methods on various maps, which require a different level of coordination according to the map’s difficulties. Win-rate is computed with 160 samples: 32 episodes for each training random seed, and 5 different random seeds unless denoted otherwise.

Both the mean and the variance of the performance are presented for all the figures to show their overall performance according to different seeds. Especially for a fair comparison, we set $n _ { \mathrm { c i r c l e } }$ , the number of training per a sampled batch of 32 episodes during training, as 1 for all baselines since some of the baselines increase $n _ { \mathrm { c i r c l e } } = 2$ as a default setting in their codes.

For performance comparison with baseline methods, we use their codes with fine-tuned algorithm configuration for hyperparameter settings according to their codes and original paper if available. For experiments on SMAC, we use the version of starcraft.py presented in RODE (Wang et al., 2021) adopting some modification for compatibility with QPLEX (Wang et al., 2020b). All SMAC experiments were conducted on StarCraft II version 4.10.0 in a Linux environment.

For Google research football task, we use the environmental code provided by (Kurach et al., 2020). In the experiments, we consider three official scenarios such as academy_3_vs_1_with_keeper (3_vs_1WK), academy_counterattack_easy (CA_easy), and academy_counterattack_hard (CA_hard).

In addition, for controlling $r ^ { c }$ in Eq. 10, the same hyperparameters related to curiosity-based (Zheng et al., 2021) or diversity-based exploration Chenghao et al. (2021) are adopted for EMU (QPLEX) and EMU (CDS) as well as for baselines EMC and CDS. After further experiments, we found that the curiosity-based $r ^ { c }$ from (Zheng et al., 2021) adversely influenced super hard SMAC task, with the exception of corridor scenario. Furthermore, the diversity-based exploration from Chenghao et al. (2021) led to a decrease in performance on both easy and hard SMAC maps. Thus, we decided to exclude the use of $r ^ { c }$ for EMU (QPLEX) on the super hard SMAC task and for EMU (CDS) on the easy/hard SMAC maps. EMU set task-dependent $\delta$ values as presented in Table 1. For other hyperparameters introduced by EMU, the same values presented in Table 8 are used throughout all tasks. For EMU (QPLEX) in corridor, $\delta = 1 . 3 e - 5$ is used instead of $\delta = 1 . 3 e - 3$ .

Table 1: Task-dependent hyperparameter of EMU.   

<table><tr><td>Category</td><td>δ</td></tr><tr><td>easy/hard SMAC maps</td><td>1.3e-5</td></tr><tr><td>super hard SMAC maps</td><td>1.3e-3</td></tr><tr><td>GRF</td><td>1.3e-3</td></tr></table>

# C.3 DETAILS OF MARL TASKS

In this section, we specify the dimension of the state space, the action space, the episodic length, and the reward of SMAC (Samvelyan et al., 2019) and GRF (Kurach et al., 2020).

In SMAC, the global state of each task of SMAC includes the information of the coordinates of all agents, and features of both allied and enemy units. The action space of each agent consists of the moving actions and attacking enemies, and thus it increases according to the number of enemies. The dimensions of the state and action space and the episodic length vary according to the tasks as shown in Table 2. For reward structure, we used the shaped reward, i.e., the default reward settings of SMAC, for all scenarios. The reward is given when dealing damage to the enemies and get bonuses for winning the scenario. The reward is scaled so that the maximum cumulative reward, $R _ { m a x }$ , that can be obtained from the episode, becomes around 20 (Samvelyan et al., 2019).

Table 2: Dimension of the state space and the action space and the episodic length of SMAC   

<table><tr><td>Task</td><td>Dimension of state space</td><td>Dimension of action space</td><td>Episodic length</td></tr><tr><td>1c3s5z</td><td>270</td><td>15</td><td>180</td></tr><tr><td>3s5z</td><td>216</td><td>14</td><td>150</td></tr><tr><td>3s_vs_5z</td><td>68</td><td>11</td><td>250</td></tr><tr><td>5m_vs_6m</td><td>98</td><td>12</td><td>70</td></tr><tr><td>MMM2</td><td>322</td><td>18</td><td>180</td></tr><tr><td>6h_vs_8z</td><td>140</td><td>14</td><td>150</td></tr><tr><td>3s5z_vs_3s6z</td><td>230</td><td>15</td><td>170</td></tr><tr><td>corridor</td><td>282</td><td>30</td><td>400</td></tr></table>

In GRF, the state of each task includes information on coordinates, ball possession, and the direction of players, etc. The dimension of the state space differs among the tasks as in Table 3. The action of each agent consists of moving directions, different ways to kick the ball, sprinting, intercepting the ball and dribbling. The dimensions of the action spaces for each task are the same. Table 3 summarizes the dimension of the action space and the episodic length. In GRF, there can be two reward modes: one is "sparse reward" and the other is "dense reward." In sparse reward mode, agents get a positive reward $+ 1$ when scoring a goal and get -1 when conceding one to the opponents. In dense reward mode, agents can get positive rewards when they approach to opponent’s goal, but the maximum cumulative reward is up to $+ 1$ . In our experiments, we adopt sparse reward mode, and thus the maximum reward, $R _ { m a x }$ is $+ 1$ for GRF.

Table 3: Dimension of the state space and the action space and the episodic length of GRF   

<table><tr><td>Task</td><td>Dimension of state space</td><td>Dimension of action space</td><td>Episodic length</td></tr><tr><td>3_vs_1WK</td><td>26</td><td>19</td><td>150</td></tr><tr><td>CA_easy</td><td>30</td><td>19</td><td>150</td></tr><tr><td>CA-hard</td><td>34</td><td>19</td><td>150</td></tr></table>

# C.4 INFRASTRUCTURE

Experiments for SMAC (Samvelyan et al., 2019) are mainly carried out on NVIDIA GeForce RTX 3090 GPU, and training for the longest experiment such as corridor via EMU (CDS) took less than 18 hours. Note that when training is conducted with $n _ { \mathrm { c i r c l e } } = 2$ , it takes more than one and a half times longer. Training encoder/decoder structure and updating $\mathcal { D } _ { E }$ with updated $f _ { \phi }$ together only took less than 2 seconds at most in corridor task. As we update $f _ { \phi }$ and $f _ { \psi }$ periodically with $t _ { e m b }$ , the additional time required for a trainable embedder is certainly negligible compared to MARL training.

# D FURTHER EXPERIMENT RESULTS

# D.1 NEW PERFORMANCE INDEX

In this section, we present the details of a new performance index called overall win-rate, $\bar { \mu } _ { w }$ . For example, let $f _ { w } ^ { i } ( s )$ be the test win-rate at training time s of ith seed run and $\mu _ { w } ^ { i } ( t )$ represents the time integration of $f _ { w } ^ { i } ( s )$ until $t$ . Then, a normalized overall win-rate, $\bar { \mu } _ { w }$ , can be expressed as

$$
\bar {\mu} _ {w} (t) = \frac {1}{\mu_ {\operatorname* {m a x}}} \frac {1}{n} \sum_ {i = 1} ^ {n} \mu_ {w} ^ {i} (t) = \frac {1}{\mu_ {\operatorname* {m a x}}} \frac {1}{n} \sum_ {i = 1} ^ {n} \int_ {0} ^ {t} f _ {w} ^ {i} (s) d s, \tag {17}
$$

where $\mu _ { \mathrm { m a x } } = t$ and $\bar { \mu } _ { w } \in [ 0 , 1 ]$ .

![](images/9d0b1e79823ff06a23d330786f45efa810f7a6f4361952afa6275c56471379b8.jpg)  
Figure 13: Illustration of $\mu _ { w } ^ { i } ( t )$ .

Figure 13 illustrates the concept of time integration of win-rate, i.e., $\mu _ { w } ^ { i } ( t )$ , to construct the overall win-rate, $\bar { \mu } _ { w }$ . By considering the integration of win-rate of each seed case, the performance variance can be considered, and thus $\bar { \mu } _ { w }$ shows the training efficiency (speed) as well as the training quality (win-rate).

# D.2 ADDITIONAL EXPERIMENT RESULTS

In Section 4.3, we present the summary of parametric studies on $\delta$ with respect to various choices of $f _ { \phi }$ . To see the training efficiency and performance at the same time, Table 4 and 5 present the overall win-rate $\bar { \mu } _ { w }$ according to training time. We conduct the experiments for 5 different seed cases and at each test phase 32 samples were used to evaluate the win-rate $[ \% ]$ . Note that we discard the component of episodic incentive $r ^ { p }$ to see the performance variations according to $\delta$ and types of $f _ { \phi }$ more clearly.

Table 4: $\bar { \mu } _ { w }$ according to $\delta$ and design choice of embedding function on hard SMAC map, 3s_vs_5z.   

<table><tr><td>Training time [mil]</td><td colspan="3">0.69</td><td colspan="3">1.37</td><td colspan="3">2.00</td></tr><tr><td>δ</td><td>random</td><td>EmbNet</td><td>dCAE</td><td>random</td><td>EmbNet</td><td>dCAE</td><td>random</td><td>EmbNet</td><td>dCAE</td></tr><tr><td>1.3e-7</td><td>0.033</td><td>0.051</td><td>0.075</td><td>0.245</td><td>0.279</td><td>0.343</td><td>0.413</td><td>0.443</td><td>0.514</td></tr><tr><td>1.3e-5</td><td>0.010</td><td>0.044</td><td>0.063</td><td>0.171</td><td>0.270</td><td>0.325</td><td>0.320</td><td>0.441</td><td>0.491</td></tr><tr><td>1.3e-3</td><td>0.034</td><td>0.043</td><td>0.078</td><td>0.226</td><td>0.270</td><td>0.357</td><td>0.381</td><td>0.439</td><td>0.525</td></tr><tr><td>1.3e-2</td><td>0.019</td><td>0.005</td><td>0.079</td><td>0.205</td><td>0.059</td><td>0.346</td><td>0.348</td><td>0.101</td><td>0.518</td></tr></table>

Table 5: $\bar { \mu } _ { w }$ according to $\delta$ and design choice of embedding function on hard SMAC map, 5m_vs_6m.   

<table><tr><td>Training time [mil]</td><td colspan="3">0.69</td><td colspan="3">1.37</td><td colspan="3">2.00</td></tr><tr><td>δ</td><td>random</td><td>EmbNet</td><td>dCAE</td><td>random</td><td>EmbNet</td><td>dCAE</td><td>random</td><td>EmbNet</td><td>dCAE</td></tr><tr><td>1.3e-7</td><td>0.040</td><td>0.117</td><td>0.110</td><td>0.287</td><td>0.397</td><td>0.397</td><td>0.577</td><td>0.690</td><td>0.701</td></tr><tr><td>1.3e-5</td><td>0.064</td><td>0.107</td><td>0.131</td><td>0.334</td><td>0.402</td><td>0.436</td><td>0.634</td><td>0.714</td><td>0.749</td></tr><tr><td>1.3e-3</td><td>0.040</td><td>0.080</td><td>0.064</td><td>0.333</td><td>0.377</td><td>0.363</td><td>0.646</td><td>0.687</td><td>0.677</td></tr><tr><td>1.3e-2</td><td>0.038</td><td>0.000</td><td>0.048</td><td>0.288</td><td>0.001</td><td>0.332</td><td>0.584</td><td>0.001</td><td>0.643</td></tr></table>

As Table 4 and 5 illustrate that dCAE structure for $f _ { \phi }$ , which considers the reconstruction loss of global state $s$ as in Eq. 5, shows the best training efficiency in most cases. For $5 \mathrm { m \_ v s \_ 6 m }$ task with $\bar { \delta } = 1 . 3 e ^ { - 3 }$ , EmbNet achieves the highest value among $f _ { \phi }$ choices in terms of $\bar { \mu } _ { w }$ but fails to find optimal policy at $\delta = 1 . 3 e ^ { - 2 }$ unlike other design choices. This implies that the reconstruction loss of dCAE facilitates the construction of a smoother embedding space for $\mathcal { D } _ { E }$ , enabling the retrieval of memories within a broader range of $\delta$ values from the key state. Figure 15 and 16 show the corresponding learning curves of each encoder structure for different $\delta$ values. A large $\delta$ value results in a higher performance variance than the cases with smaller $\delta$ , according to different seed cases.

This is because a high value of $\delta$ encourages exploratory memory recall. In other words, by adjusting $\delta$ , we can control the level of exploration since it controls whether to recall its own MC return or the highest value of other similar states within δ. Thus, without constructing smoother embedding space as in dCAE, learning with exploratory memory recall within large $\delta$ can converge to sub-optimality as illustrated by the case of EmbNet in Figure 16(d).

In Figure 14 which shows the averaged number of memory recall $( \bar { N } _ { c a l l } )$ of all memories in $\mathcal { D } _ { E }$ ,

$\bar { N } _ { c a l l }$ of EmbNet significantly increases as training proceeds. On the other hand, dCAE was able to prevent this problem and recalled the proper memories in the early learning phase, achieving good training efficiency. Thus, embedding with dCAE can leverage a wide area of memory in $\mathcal { D } _ { E }$ and becomes robust to hyperparameter $\delta$ .

![](images/25b772f8ea45eb55e95e63cf608fac2f6a15623dcac6dc428ab537e15cd79db9.jpg)  
Figure 14: $\bar { N } _ { c a l l }$ of all memories in $\mathcal { D } _ { E }$ when $\delta = 0 . 0 1 3$ according to design choice for $f _ { \phi }$ .

![](images/a2da06c362f9db7bcc4b075a591c6b9186cc34ff1858b7e64ad90471bd5dd587.jpg)  
(a) δ = 1.3e−7

![](images/e0ae2db30f84e979d237b2e67a20cd899cc1941d687b9462ef508e4c94709095.jpg)  
(b) δ = 1.3e−5

![](images/7fbffaa6cf2a69755b9306a3142e21d1dea4024efd5863f7fc9561c1e4555abb.jpg)  
(c) δ = 1.3e−3

![](images/fa617a6dd5d3e38939c4d1c05a1a63a39410b4f0eb9b322031c0ed00fa808b73.jpg)  
(d) δ = 1.3e−2

![](images/fa722510b9525c8e7e9fbfa61bffff42fee4e8f3be38c0343d976c0aa9b38ddd.jpg)  
Figure 15: Parametric studies for $\delta$ on $3 s \_ { \nabla S \_ } 5 z$ SMAC map.   
(a) δ = 1.3e−7

![](images/e2f99bc5f67e3eda6c96b8d39fbc9d039d92d2c0effaa28e1b099cdcd6aeec74.jpg)  
(b) δ = 1.3e−5

![](images/cc6204f4b4eacf6604381fa73d11d7f091ead207f66a1d0b759eb390f2b12922.jpg)  
(c) δ = 1.3e−3

![](images/918488169306d9bc5ff6b265c672f3fa9dfafca9f2439cb0dc84c07281f112a8.jpg)  
(d) δ = 1.3e−2   
Figure 16: Parametric studies for $\delta$ on 5m_vs_6m SMAC map.

# D.3 COMPARATIVE EVALUATION ON ADDITIONAL STARCRAFT II MAPS

![](images/f19d83648c33b137b32a523d3676ad424a9959931d04ca650d4b4442d706e3ed.jpg)  
Figure 17 presents a comparative evaluation of EMU with baseline algorithms on additional SMAC maps. Adopting EMU shows performance gain in various tasks.   
Figure 17: Performance comparison of EMU against baseline algorithms on additional SMAC maps.

# D.4 COMPARISON OF EMU WITH MAPPO ON SMAC

In this subsection, we compare the EMU with MAPPO (Yu et al., 2022) on selected SMAC maps. Figure 18 shows the performance in six SMAC maps: 1c3s5z, 3s_vs_5z, 5m_vs_6m, MMM2, $6 \mathrm { { h } \_ { \nabla } \mathrm { { s } \_ { - } \mathrm { { 8 } } \mathrm { { z } } } }$ and $3 \mathrm { s } 5 \mathrm { z \_ v s \_ } 3 \mathrm { s } 6 \mathrm { z }$ . Similar to the previous performance evaluation in Figure 4, Win-rate is computed with 160 samples: 32 episodes for each training random seed and 5 different random seeds. Also, for MAPPO, scenario-dependent hyperparameters are adopted from their original settings in the uploaded source code.

From Figure 18, we can see that EMU performs better than MAPPO with an evident gap. Although after extensive training MAPPO showed a comparable performance against off-policy algorithm in its original paper (Yu et al., 2022), within the same training timestep used for our experiments, we found that MAPPO suffers from local convergence in super hard SMAC tasks such as MMM2 and $3 \mathrm { s } 5 \mathrm { z \_ v s \_ } 3 \mathrm { s } 6 \mathrm { z }$ as shown in Figure 18. Only in $6 \mathrm { h } _ { - } \mathrm { v } s _ { - } 8 z$ , MAPPO shows comparable performance to EMU (QPLEX) with higher performance variance across different seeds.

![](images/74185e35f256f234e1f563b2455407aafafae68d1918123473be026bda0e700c.jpg)

![](images/5e1a93410843d912259008a816bebc287453564e2c414990172d07f99bce5dae.jpg)  
Figure 18: Performance comparison with MAPPO on selected SMAC maps.

# D.5 ADDITIONAL PARAMETRIC STUDY

In this subsection, we conduct an additional parametric study to see the effect of key hyperparameter $\delta$ . Unlike the previous parametric study on Appendix D.2, we adopt both dCAE embedding network for $f _ { \phi }$ and episodic reward. For evaluation, we consider three GRF tasks such as academy_3_vs_1_with_keeper (3_vs_1WK), academy_counterattack_easy (CA-easy), and academy_counterattack_hard (CA-hard); and one super hard SMAC map such as 6h_vs_8z. For each task to evaluate EMU, four $\delta$ values, such as $\delta _ { 1 } = 1 . 3 e ^ { - 7 }$ , $\delta _ { 2 } \stackrel { - } { = } 1 . 3 e ^ { - 5 }$ , $\delta _ { 3 } = 1 . 3 e ^ { - 3 }$ , and $\delta _ { 4 } = 1 . 3 e ^ { - 2 }$ , are considred. Here, to compute the win-rate, 160 samples (32 episodes for each training random seed and 5 different random seeds) are used for 3_vs_1WK and $6 \mathrm { { h } \_ { \nabla } \mathrm { { v s } \_ { \nabla } 8 \mathrm { { z } } } }$ while 100 samples (20 episodes for each training random seed and 5 different random seeds) are used for CA-easy and CA-hard. Note that CDS and EMU (CDS) utilize the same hyperparameters, and EMC and EMU (QPLEX) use the same hyperparameters without a curiosity incentive presented in Zheng et al. (2021) as the model without it showed the better performance when utilizing episodic control.

![](images/6dce44bf7d009d317019edb80116e36bd574f622cbf1f954c91d6d7ad6f3eab2.jpg)  
(a) 3_vs_1WK (GRF)

![](images/d4e074d28439188202764506b760632509b9cb104d930b7cfdaf70ed7fb4a10c.jpg)  
(b) CA-easy (GRF)

![](images/8bcbf7fb77b7495c8c4d89527108494cef292622198bbe72cd2fb1bb2dcb4dc9.jpg)  
(c) CA-hard (GRF)

![](images/a5bd59232ca5c201ec647fa92b9bf6e479d6965ccb4818761b61ffafe1cb1742.jpg)  
(d) 6h_vs_8z (SMAC)   
Figure 19: Parametric studies for $\delta$ on various GRF maps and super hard SMAC map.

In all cases, EMU with $\delta _ { 3 } = 1 . 3 e ^ { - 3 }$ shows the best performance. The tasks considered here are all complex multi-agent tasks, and thus adopting a proper value of $\delta$ benefits the overall performance and achieves the balance between exploration and exploitation by recalling the semantically similar memories from episodic memory. The optimal value of $\delta _ { 3 }$ is consistent with the determination logic on $\delta$ in a memory efficient way presented in Appendix F.

# D.6 ADDITIONAL PARAMETRIC STUDY ON $\lambda _ { \mathit { r c o n } }$

Additionally, we conduct a parametric study for $\lambda _ { \mathit { r c o n } }$ in Eq. 5. For each task, EMU with five $\lambda _ { \mathit { r c o n } }$ values, such as $\lambda _ { r c o n , 0 } = 0 . 0 1$ , $\lambda _ { r c o n , 1 } = 0 . 1$ , $\lambda _ { r c o n , 2 } = 0 . 5$ , $\lambda _ { r c o n , 3 } = 1 . 0$ and $\lambda _ { r c o n , 4 } = 1 0$ , are evaluated. Here, to compute the win-rate of each case, 160 samples (32 episodes for each training random seed and 5 different random seeds) are used. From Figure 20, we can see that broad range of

![](images/a8a629b4a9f62f1c15a6239883421b32a62984394f4bbe8746fd3288091cb609.jpg)  
(a) 3s5z (SMAC)

![](images/c2be22a113c85d52ccfef5646035014a378550cdcdc8983000354a0a51148c60.jpg)  
(b) 3s_vs_5z (SMAC)   
Figure 20: Parametric study for $\lambda _ { \mathit { r c o n } }$

$\lambda _ { r c o n } \in \{ 0 . 1 , 0 . 5 , 1 . 0 \}$ work well in general. However, with large $\lambda _ { \mathit { r c o n } }$ as $\lambda _ { r c o n , 4 } = 1 0$ , we can observe that some performance degradation at the early learning phase in $3 5 5 z$ task. This result is in line with the learning trends of Case 1 and Case 2 of $3 \le 5 z$ in Figure 23, which do not consider prediction loss and only take into account the reconstruction loss. Thus, considering both prediction loss and reconstruction loss as Case 4 in Eq. 5 with proper $\lambda _ { \mathit { r c o n } }$ is essential to optimize the overall learning performance.

# D.7 ADDITIONAL ABLATION STUDY IN GRF

![](images/65973ac91ca1c8e069ff72706a40a412d1cca329f75787c0b8fe09fbf8a77954.jpg)  
(a) 3_vs_1WK (GRF)

![](images/af69c75058976a1b01ba6673cd6cbcb5753590ccf9efa4d1f2d87eea931c9d96.jpg)  
(b) CA-easy (GRF)   
Figure 21: Ablation studies on episodic incentive on GRF tasks.

In this subsection, we conduct additional ablation studies via GRF tasks to see the effect of episodic incentive. Again, EMU (CDS-No-EI) ablates episodic incentive from EMU (CDS) and utilizes the conventional episodic control presented in Eq. 3 instead. Again, EMU (CDS-No-SE) ablates semantic embedding by dCAE and adopts random projection with episodic incentive $r ^ { p }$ . In both tasks, utilizing episodic memory with the proposed embedding function improves the overall performance compared to the original CDS algorithm. By adopting episodic incentives instead of conventional episodic control, EMU (CDS) achieves better learning efficiency and rapidly converges to optimal policies compared to EMU (CDS-No-EI).

# D.8 ADDITIONAL ABLATION STUDY ON EMBEDDING LOSS

In our case, the autoencoder uses the reconstruction loss to enforce the embedded representation $x$ to contain the full information of the original feature, s. We are adding $( H _ { t } - f _ { \psi } ^ { H } ( \dot { f _ { \phi } } ( s _ { t } | t ) | t ) ) ^ { 2 }$ to guide the embedded representation to be consistent to $H _ { t }$ , as well, which works as a regularizer to the autoencoder. Therefore, $f _ { \psi } ^ { H }$ is used in Eq. 5 to predict the observed $H _ { t }$ from $D _ { E }$ as a part of the semantic regularization effort.

Because $H _ { t }$ is different from $f _ { \psi } ^ { H } ( x _ { t } )$ , the effort of minimizing their difference becomes the regularizer creating a gradient signal to learn $\psi$ and $\phi$ . The update of $\phi$ results in the updated $x$ influenced by the regularization. Note that we update $\phi$ through the backpropagation of $\psi$ .

The case of $L ( \phi , \psi ) = | | s _ { t } - f _ { \psi } ^ { s } ( f _ { \phi } ( s _ { t } | t ) | t ) | | _ { 2 } ^ { 2 }$ occurs when $\lambda _ { \mathit { r c o n } }$ becomes relatively much higher than 1, which makes $( H _ { t } - f _ { \psi } ^ { H } ( f _ { \phi } ( s _ { t } | t ) | t ) ) ^ { 2 }$ becomes ineffective. In other words, when $\lambda _ { \mathit { r c o n } }$ in Eq. 5 becomes relatively much higher than 1, $( H _ { t } - f _ { \psi } ^ { H } ( f _ { \phi } ( s _ { t } | t ) | t ) ) ^ { 2 }$ becomes ineffective.

The case of $L ( \phi , \psi ) = ( H _ { t } - f _ { \psi } ^ { H } ( f _ { \phi } ( s _ { t } | t ) | t ) ) ^ { 2 }$ occurs when the scale factor $\lambda _ { \mathit { r c o n } }$ becomes relatively much smaller than 1, which makes $( H _ { t } - f _ { \psi } ^ { H } ( f _ { \phi } ( s _ { t } | t ) | t ) ) ^ { 2 }$ become a dominant factor. We conduct ablation studies considering four cases as follows:

• Case 1: $L ( \phi , \psi ) = | | s _ { t } - f _ { \psi } ^ { s } ( f _ { \phi } ( s _ { t } ) ) | | _ { 2 } ^ { 2 }$ , presented in Figure 22(a)   
• Case 2: $L ( \phi , \psi ) = | | s _ { t } - f _ { \psi } ^ { s } ( f _ { \phi } ( s _ { t } | t ) | t ) | | _ { 2 } ^ { 2 }$ , presented in Figure 22(b)   
• Case 3: $L ( \phi , \psi ) = ( H _ { t } - f _ { \psi } ^ { H } ( f _ { \phi } ( s _ { t } | t ) | t ) ) ^ { 2 }$ , presented in Figure 22(c)   
• Case 4: $L ( \phi , \psi ) = ( H _ { t } - f _ { \psi } ^ { H } ( f _ { \phi } ( s _ { t } | t ) | t ) ) ^ { 2 } + \lambda _ { r c o n } | | s _ { t } - f _ { \psi } ^ { s } ( f _ { \phi } ( s _ { t } | t ) | t ) | | _ { 2 } ^ { 2 }$ , i.e., Eq. 5, presented in Figure 22(d)

We visualize the result of t-SNE of 50K samples $x \in D _ { E }$ out of 1M memory data trained by various loss functions: The task was 3s_vs_5z of SMAC as in Figure 2 and the training for all models proceeds for $1 . 5 \mathrm { { m i l } }$ training steps. Case 1 and Case 2 showed irregular return distribution across the embedding space. In those two cases, there was no consistent pattern in the reward distribution. Case 3 with only return prediction in the loss showed better patterns compared to Case 1 and 2 but some features are not clustered well. We suspect that the consistent state representation also contributes to the return prediction. Case 4 of our suggested loss showed the most regular pattern in the return distribution arranging the low-return states as a cluster and the states with desirable returns as another

![](images/c0594be7130fb30db0583b56f3186f2fe3dd61954708a521752de5184c77e5d6.jpg)  
(a) Loss (case 1)

![](images/56870683e66cea282ffe7ca976eff4775c42f091cbd8dd78f7c7425f0f165dcf.jpg)  
(b) Loss (case 2)

![](images/8b70b36984a054e09e4ad253a406c77a5c66580b9c92627f1931966876b20236.jpg)  
(c) Loss (case 3)

![](images/bdbed3c150015d076f0cd95b36823df45543096d95c30f2f705e230be737e701.jpg)  
(d) Loss (case 4)   
Figure 22: t-SNE of sampled embedding $x \in D _ { E }$ trained by dCAE with various loss functions in 3s_vs_5z SMAC map. Colors from red to purple represent from low return to high return.

cluster. In Figure 23, Case 4 shows the best performance in terms of both learning efficiency and terminal win-rate.

![](images/3474e6c90a31b326aad035a07bce8784ff3fcb21dd66cfcdbdef53be03a99ba7.jpg)  
(a) 3s5z (SMAC)

![](images/edfdc5266d4a6fd68f67b31a3ca25d96f63694ae163819902d6a071e54bee354.jpg)  
(b) 3s_vs_5z (SMAC)   
Figure 23: Performance comparison of various loss functions for dCAE.

# D.9 ADDITIONAL ABLATION STUDY ON SEMANTIC EMBEDDING

To further understand the role of semantic embedding, we conduct additional ablation studies and present them with the general performance of other baseline methods. Again, EMU (CDS-No-SE) ablates semantic embedding by dCAE and adopts random projection instead, along with episodic incentive $r ^ { p }$ .

![](images/1dfb495525ae2b4198108211cd4b7c2c00181ee946036dffee7919693f89443b.jpg)  
Figure 24: Performance comparison of EMU against baseline algorithms on three easy and hard SMAC maps: $1 \mathrm { c } 3 \mathrm { s } 5 \mathrm { z }$ , 3s_vs_5z, and 5m_vs_6m, and three super hard SMAC maps: MMM2, 6h_vs_8z, and 3s5z_vs_3s6z.

![](images/fcac2cb604b411736de655a5194be9d6bd9eee52fa1b9293fdac90cb20c5e7ff.jpg)  
Figure 25: Performance comparison of EMU against baseline algorithms on Google Research Football.

For relatively easy tasks, EMU (QPLEX-No-SE) and EMU (CDS-No-SE) show comparable performance at first but they converge on sub-optimal policy in most tasks. Especially, this characteristic is well observed in the case of EMU (CDS-No-SE). As large size of memories are stored in an episodic buffer as training goes on, the probability of recalling similar memories increases. However, with random projection, semantically incoherent memories can be recalled and thus it can adversely affect the value estimation. We deem this is the reason for the convergence on suboptimal policy in the case of EMU (No-SE). Thus we can conclude that recalling semantically coherent memory is an essential component of EMU.

# D.10 ADDITIONAL ABLATION ON $r ^ { c }$

In Eq.10, we introduce $r ^ { c }$ as an additional reward which may encourage exploratory behavior or coordination. The reason we introduce $r ^ { c }$ is to show that EMU can be used in conjunction with any form of incentive encouraging further exploration. Our method may not be strongly effective until some desired states are found, although it has exploratory behavior via the proposed semantic embeddings, controlled by $\delta$ . Until then, such incentives could be beneficial to find desired or goal states. Figures 26-27 show the ablation study of with and without $r ^ { c }$ , and the contribution of $r ^ { c }$ is limited compared to $r ^ { p }$ .

![](images/80b98cb7c835a58e3deb2206ef89f79798d431bbfdbc9389e6cacd2f45fe2187.jpg)  
(a) 3s_vs_5z

![](images/224ee5619d59d21e7b8cd840c5a1ebc0cda971faa8842b60191241a919e13187.jpg)  
(b) 5m_vs_6m

![](images/72a9dc5d9e6bcaec2c6e26f8196ef6273d57fa1cb4b99b3b7caf9073557766f7.jpg)  
(c) 3s5z_vs_3s6z

![](images/9feaff53f636f1781e38baf7e61b036d779d18e4ecb2add8819a6fb9fe3a32d9.jpg)  
Figure 26: Ablation studies on $r ^ { c }$ in SMAC tasks.   
(a) 3s_vs_5z

![](images/efe2f1bff44b5efa6e6bc353faf0f25e4b28a2afb1d93c388126e4d1c79fe867.jpg)  
(b) 5m_vs_6m

![](images/e539e8ebf65cd20dfa7acde008255f9bb7337275f775f73845b81821f43b0749.jpg)  
(c) 3s5z_vs_3s6z   
Figure 27: Ablation studies on $r ^ { c }$ in SMAC tasks.

# D.11 COMPARISON OF EPISODIC INCENTIVE WITH EXPLORATORY INCENTIVE

In this subsection, we replace the episodic incentive with another exploratory incentive, introduced by (Henaff et al., 2022). In (Henaff et al., 2022), the authors extend the count-based episodic bonuses to continuous spaces by introducing episodic elliptical bonuses for exploration. In this concept, a high reward is given when the state projected in the embedding space is different from the previous states within the same episode. In detail, with a given feature encoder $\phi$ , the elliptical bonus $b _ { t }$ at timestep $t$ is computed as follows:

$$
b _ {t} = \phi \left(s _ {t}\right) ^ {T} C _ {t} ^ {- 1} \phi \left(s _ {t}\right) \tag {18}
$$

where $C _ { t } ^ { - 1 }$ is an inverse covariance matrix with an initial value of $C _ { t = 0 } ^ { - 1 } = 1 / \lambda _ { e 3 b } I$ . Here, $\lambda _ { e 3 b }$ is a covariance regularizer. For update inverse covariance, the authors suggested a computationally efficient update as

$$
C _ {t + 1} ^ {- 1} = C _ {t} ^ {- 1} - \frac {1}{1 + b _ {t + 1}} u u ^ {T} \tag {19}
$$

where $u = C _ { t } ^ { - 1 } \phi ( s _ { t + 1 } )$ . Then, the final reward $\bar { r } _ { t }$ with episodic elliptical bonuses $b _ { t }$ is expressed as

$$
\bar {r} _ {t} = r _ {t} + \beta_ {e 3 b} b _ {t} \tag {20}
$$

where $\beta _ { e 3 b }$ and $r _ { t }$ are a corresponding scale factor and external reward given by the environment, respectively.

For this comparison, we utilize the dCAE structure as a state embedding function $\phi$ . For a mixer, QPLEX (Wang et al., 2020b) is adopted for all cases, and we denote the case with an elliptical incentive instead of the proposed episodic incentive as QPLEX $\mathrm { S E + E 3 B }$ ). Figure 28 illustrates

![](images/38a49dc533f9255f4db5ced652cb48b083b19192871a4a975e4e3a159ed67f0e.jpg)  
Figure 28: Performance comparison with elliptical incentive on selected SMAC maps.

the performance of adopting an elliptical incentive for exploration instead of the proposed episodic incentive. QPLEX $\mathrm { ( S E + E 3 B ) }$ ) uses the same hyperparameters with EMU $\mathrm { \ S E { + E I } ) }$ and we set $\lambda _ { e 3 b } = 0 . 1$ according to Henaff et al. (2022).

As illustrated by Figure 28, adopting an elliptical incentive presented by (Henaff et al., 2022) instead of an episodic incentive does not give any performance gain and even adversely influences the performance compared to QPLEX. It seems that adding excessive surprise-based incentives can be a disturbance in MARL tasks since finding a new state itself does not guarantee better coordination among agents. In MARL, agents need to find the proper combination of joint action in a given similar observations when finding an optimal policy. On the other hand, in high-dimensional pixelbased single-agent tasks such as Habitat (Ramakrishnan et al., 2021), finding a new state itself can be beneficial in policy optimization. From this, we can note that adopting a certain algorithm from a single-agent RL case to MARL case may require a modification or adjustment with domain knowledge.

As a simple tuning, we conduct parametric study for $\beta _ { e 3 b } = \{ 0 . 0 1 , 0 . 1 \}$ to adjust magnitude of incentive of E3B. Figure 29 illustrates the results. In Figure 29, QPLEX $\mathrm { S E + E } 3 \mathrm { B } )$ with $\beta _ { e 3 b } = 0 . 0 1$ shows a better performance than the case with $\beta _ { e 3 b } = 0 . 1$ and comparable performance to EMC in 5m_vs_6m. However, EMU with the proposed episodic incentive shows the best performance. From this comparison, we can see that incentives proposed by previous work need to be adjusted

![](images/e647ace12b3c5b11b5d7a1da225466c126892158d47329d96e570c844a6d9a1d.jpg)  
Figure 29: Performance comparison with an elliptical incentive on selected SMAC maps.

according to the type of tasks, as it was done in EMC (Zheng et al., 2021). On the other hand, with the proposed episodic incentive we do not need such hyperparameter-scaling, allowing much more flexible application across various tasks.

# D.12 ADDITIONAL TOY EXPERIMENT AND APPLICABILITY TESTS

In this section, we conduct additional experiments on the didactic example presented by (Zheng et al., 2021) to see how the proposed method would behave in a simple but complex coordination task. Additionally, by defining $R _ { t h r }$ to define the desirability presented in Definition 1, we can extend EMU to a single-agent RL task, where a strict goal is not defined in general.

Didactic experiment on Gridworld We adopt the didactic example such as gridworld environment from (Zheng et al., 2021) to demonstrate the motivation and how the proposed method can overcome the existing limitations of the conventional episodic control. In this task, two agents in gridworld (see Figure 30(a)) need to reach their goal states at the same time to get a reward $r = 1 0$ and if only one arrives first, they get a penalty with the amount of $- p$ . Please refer to (Zheng et al., 2021) for further details.

![](images/46d878944344ab9db45970357412b16c2d69d45cbeaa2d4f2b991e18e383678a.jpg)  
(a) Gridworld

![](images/e692087ccc2886af83ab13879330804b71dd98f3e308cb811575c089afb681a6.jpg)  
(b) Performance evaluation $( p = 2$ )   
Figure 30: Didactic experiments on gridworld.

To see the sole effect of the episodic control, we discard the curiosity incentive part of EMC, and for a fair comparison, we set the same exploration rate of $\epsilon$ -greedy with $T _ { \epsilon } = 2 0 0 K$ for all algorithms. We evaluate the win-rate with 180 samples (30 episodes for each training random seed and 6 different random seeds) at each training time. Notably, adopting episodic control with a naive utilization suffers from local convergence (see QPLEX and EMC (QPLEX) in Figure 30(b)), even though it expedites learning efficiency at the early training phase. On the other hand, EMU shows more robust performance under different seed cases and achieves the best performance by an efficient and discreet utilization of episodic memories.

Applicability test to single agent RL task We first need to define $R _ { t h r }$ value to effectively apply EMU to a single-agent task where a goal of an episode is generally not strictly defined, unlike cooperative multi-agent tasks with a shared common goal.

In a single-agent task where the action space is continuous such as MuJoCo (Todorov et al., 2012), the actor-critic method is often adopted. Efficient memory utilization of EMU can be used to train the critic network and thus indirectly influence policy learning, unlike general cooperative MARL tasks where value-based RL is often considered.

We implement EMU on top of TD3 and use the open-source code presented in (Fujimoto et al., 2018). We begin to train the model after sufficient data is stored in the replay buffer and conduct 6 times of training per episode with 256 mini-batches. Note that this is different from the default settings of RL training, which conducts training at each timestep. Our modified setting aims to see the effect on the sample efficiency of the proposed model. The performance of the trained model is evaluated at every 50k timesteps.

We use the same hyperparameter settings as in MARL task presented in Table 8 except for the update interval, $t _ { e m b } = 1 0 0 K$ according to large episodic timestep in single-RL compared to MARL tasks. It is worth mentioning that additional customized parameter settings for single-agent tasks may further improve the performance. In our evaluation, three single-agent tasks such as Hopper-v4, Walker2D-v4 and Humanoid-v4 are considered, and Figure 32 illustrates each task. Here, $\delta _ { 2 } = 1 . 3 e \mathrm { ~ - ~ } 5$ is used for Hopper-v4 and Walker2D-v4, and $\delta _ { 3 } = 1 . 3 e \mathrm { ~ - ~ } 3$ is used for Humanoid-v4 as Humanoid-v4 task contains much higher state dimension space as 376-dimension. Please refer to Todorov et al. (2012) for a detailed description of tasks.

![](images/1e13d8ee6d1a87a5a0e1b95a120eedabd1a34fcaa52b7f58fe23ecf469b0dc46.jpg)  
(a) Hopper-v4

![](images/852eb086170ea674273a8c9a14af01b70545833e0da1dbc4bab5bbbef7ce4b4b.jpg)  
(b) Walker2D-v4

![](images/cd07db6059126baa050a3da0fe38c5a74099a79b715873a38535642a1eb9659d.jpg)  
(c) Humanoid-v4

![](images/a7408ed7e847cd48680b21f51c4e46bd71409083b3c9ca52f007ba2b488a8fee.jpg)  
Figure 31: Illustration of MuJoCo scenarios.   
(a) Performance (Hopper)

![](images/3899c4bc8a3ae9496094b8f7f67dd51d58b4ad38864d17bb738ca4f2bcb2c8f1.jpg)  
(b) Performance (Walker2D)

![](images/06680961bd0989e736bd197f6ef4f0e51662de34da48d2e0e7914c5ef6528569.jpg)  
(c) Performance (Humanoid)   
Figure 32: Applicability test to single agent task $( R _ { t h r } = 5 0 0 )$ ).

In Figure 32, EMU (TD3) shows the performance improvement compared to the original TD3. Thanks to semantically similar memory recall and episodic incentive, states deemed desirable could have high values, and trained policy is encouraged to visit them more frequently. As a result, EMU (TD3) shows the better performance. Interestingly, under state dimension as Humanoid-v4 task, TD3 and EMU (TD3) show a distinct performance gap in the early training phase. This is because, in a task with a high-dimensional state space, it is hard for a critic network to capture important features determining the value of a given state. Thus, it takes longer to estimate state value accurately. However, with the help of semantically similar memory recall and error compensation through episodic incentive, a critic network in EMU (TD3) can accurately estimate the value of the state much faster than the original TD3, leading to faster policy optimization.

Unlike cooperative MARL tasks, single-RL tasks normally do not have a desirability threshold. Thus, one may need to determine $R _ { t h r }$ based on domain knowledge or a preference for the level of return to be deemed successful. Figure 33 presents a performance variation according to $R _ { t h r }$ .

![](images/c58be76119119fd17684f0e9943a007eb566b9c882b90c7e6e099b9e6d40ea99.jpg)  
(a) Hopper-v4

![](images/55df4ca696dbeb227a3c8d0abfcdfdf235c62dff3422a10c7f6f64d6cf17526c.jpg)  
(b) Walker2d-v4   
Figure 33: Parametric study on $R _ { t h r }$ .

When we set $R _ { t h r } = 1 0 0 0$ in Walker2d task, desirability signal is rarely obtained compared to the case with $R _ { t h r } = 5 0 0$ in the early training phase. Thus, EMU with $R _ { t h r } = 5 0 0$ shows the better performance. However, both cases of EMU show better performance compared to the original TD3. In Hopper task, both cases of $R _ { t h r } = 5 0 0$ and $R _ { t h r } = 1 0 0 0$ show the similar performance. Thus,

when determining $R _ { t h r }$ , it can be beneficial to set a small value rather than a large one that can be hardly obtained.

Although setting a small $R _ { t h r }$ does not require much domain knowledge, a possible option to detour this is a periodic update of desirability based on the average return value $H ( s )$ in all $s \in \mathcal { D } _ { E }$ . In this way, a certain state with low return which was originally deemed as desirable can be reevaluated as undesirable as training proceeds. The episodic incentive is not further given to those undesirable states.

Scalability to image-based single-agent RL task Although MARL tasks already contain highdimension state space such as 322-dimension in MMM2 and 282-dimension in corridor, imagebased single RL tasks, such as Atari Bellemare et al. (2013) game, often accompany higher state spaces such as [210x160x3] for "RGB" and [210x160] for "grayscale". We use the "grayscale" type for the following experiments. For the details of the state space in MARL task, please see Appendix C.3.

In an image-based task, storing all state values to update all the key values in $\mathcal { D } _ { E }$ as $f _ { \phi }$ updates can be memory-inefficient, and a semantic embedding from original states may become overhead compared to the case without it. In such case, one may resort to a pre-trained feature extraction model such as ResNet model provided by torch-vision in a certain amount for dimension reduction only, before passing through the proposed semantic embedding. The feature extraction model above is not an object of training.

As an example, we implement EMU on the top of DQN model and compare it with the original DQN on Atari task. For the EMU (DQN), we adopt some part of pre-trained ResNet18 presented by torch-vision for dimensionality reduction, before passing an input image to semantic embedding. At each epoch, 320 random samples are used for training in Breakout task, and 640 random samples are used in Alien task. The same mini-batch size of 32 is used for both cases. For $f _ { \phi }$ training, the same parameters presented in Table 8 are adopted except for the $t _ { e m b } = 1 0 K$ considering the timestep of single RL task. We also use the same $\delta _ { 2 } = 1 . 3 e - 5$ and set $R _ { t h r } = 5 0$ for Breakout and $R _ { t h r } = 4 0$ for Alien, respectively. Please refer to Bellemare et al. (2013) and https://gymnasium.farama.org/environments/atari for task details. As in Figure 34, we found a performance gain by adopting EMU on high-dimensional image-based tasks.

![](images/bffd5b720dec0b2724c84c209f98130063d3bc51bfb92d1224cacc01848ea43e.jpg)  
(a) Breakout

![](images/4a62e72902299f41fe6f457b2b5f5212f697295e2c5248c6ba2c1a91495a963c.jpg)  
(b) Performance (Breakout)

![](images/e0b3f07c612d1793ef46c7decbe044e1d6e0b1dca2cd415ef145fc8f311348a8.jpg)  
(c) Alien

![](images/90f6e3f9797177d21088de80a506dc85a3dc1e87d005c4506ccf7da6e43bbfe5.jpg)  
(d) Performance (Alien)   
Figure 34: Image-based single-RL task example.

# E TRAINING ALGORITHM

# E.1 MEMORY CONSTRUCTION

During the centralized training, we can access the information on whether the episodic return reaches the highest return $R _ { \mathrm { m a x } }$ or threshold $R _ { t h r }$ , i.e., defeating all enemies in SMAC or scoring a goal in GRF. When storing information to $\mathcal { D } _ { E }$ , by the definition presented Definition. 1, we set $\xi ( s ) = 1$ for $\forall s \in \mathcal { T } _ { \xi }$ .

For efficient memory construction, we propagate the desirability of the state to a similar state within the threshold $\delta$ . With this desirability propagation, similar states have an incentive for a visit. In addition, once a memory is saved in $\mathcal { D } _ { E }$ , the memory is preserved until it becomes obsolete (the oldest memory to be recalled). When a desirable state is found near the existing suboptimal memory within $\delta$ , we replace the suboptimal memory with the desirable one, which gives the effect of a memory shift to the desirable state. Algorithm 2 presents the memory construction with the desirability propagation and memory shift.

Algorithm 2 Episodic memory construction   
1: $\xi_{\mathcal{T}}$ : Optimality of trajectory   
2: $\mathcal{T} = \{s_0,a_0,r_0,s_1,\dots,s_T\}$ : Episodic trajectory   
3: Initialize $R_{t} = 0$ 4: for $t = T$ to 0 do   
5: Compute $x_{t} = f_{\phi}(s_{t})$ and $y_{t} = (x_{t} - \hat{\mu}_{x}) / \hat{\sigma}_{x}$ 6: pick the nearest neighbor $\hat{x}_t\in \mathcal{D}_E$ and get $\hat{y}_t$ 7: if $||\hat{y}_t - y_t||_2 <   \delta$ then   
8: $N_{call}(\hat{x}_t)\gets N_{call}(\hat{x}_t) + 1$ 9: if $\xi_{\mathcal{T}} = = 1$ then   
10: $N_{\xi}(\hat{x}_t)\gets N_{\xi}(\hat{x}_t) + 1$ 11: end if   
12: if $\xi_{t} = = 0$ and $\xi_{\mathcal{T}} = = 1$ then   
13: $\xi_t\gets \xi_\tau$ >desirability propagation   
14: $\hat{x}_t\gets x_t,\hat{y}_t\gets y_t,\hat{s}_t\gets s_t$ >memory shift   
15: $\hat{H}_t\gets R_t$ 16: else   
17: if $\hat{H}_t <   R_t$ then $\hat{H}_t\gets R_t$ 18: end if   
19: end if   
20: else   
21: Add memory $\mathcal{D}_E\gets (x_t,y_t,R_t,s_t,\xi_t)$ 22: end if   
23: end for

For memory capacity and latent dimension, we used the same values as Zheng et al. (2021), and Table 6 shows the summary of hyperparameter related to episodic memory.

Table 6: Configuration of Episodic Memory.   

<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>episodic latent dimension, dim(x)</td><td>4</td></tr><tr><td>episodic memory capacity</td><td>1M</td></tr><tr><td>a scale factor, λ</td><td>0.1</td></tr><tr><td>(for conventional episodic control only)</td><td></td></tr></table>

The memory construction for EMU seems to require a significantly large memory space, especially for saving global states s. However, $\mathcal { D } _ { E }$ uses CPU memory instead of GPU memory, and the memory required for the proposed embedder structure is minimal compared to the memory usage of original

Table 7: Additional CPU memory usage to save global states.   

<table><tr><td>SMAC task</td><td>CPU memory usage (1M data) 
(GiB)</td></tr><tr><td>5m_vs_6m</td><td>0.4</td></tr><tr><td>3s5z_vs_3s6z</td><td>0.9</td></tr><tr><td>MMM2</td><td>1.2</td></tr></table>

RL training $( < 1 \% )$ . Thus, a memory burden due to a trainable embedding structure is negligible. Table 7 presents examples of CPU memory usage to save global states $s \in \mathcal { D } _ { E }$ .

# E.2 OVERALL TRAINING ALGORITHM

In this section, we present details of the overall MARL training algorithm including training of $f _ { \phi }$ . Additional hyperparameters related to Algorithm 1 to update encoder $f _ { \phi }$ and decoder $f _ { \psi }$ are presented in Table 8. Note that variables $N$ and $B$ are consistent with Algorithm 1.

Table 8: EMU Hyperparameters for $f _ { \phi }$ and $f _ { \psi }$ training.   

<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>a scale factor of reconstruction loss, λrcon</td><td>0.1</td></tr><tr><td>update interval, t_emb</td><td>1K</td></tr><tr><td>training samples, N</td><td>102.4K</td></tr><tr><td>batch size of training, B</td><td>1024</td></tr></table>

Algorithm 3 presents the pseudo-code of overall training for EMU. In Algorithm 3, network parameters related to a mixer and individual Q-network are denoted as $\theta$ , and double Q-learning with target network is adopted as other baseline methods (Rashid et al., 2018; 2020; Wang et al., 2020b; Zheng et al., 2021; Chenghao et al., 2021).

Algorithm 3 EMU: Efficient episodic Memory Utilization for MARL   
1: $\mathcal{D}$ : Replay buffer  
2: $\mathcal{D}_E$ : Episodic buffer  
3: $Q_{\theta}^{i}$ : Individual Q-network of $n$ agents  
4: $M$ : Batch size of RL training  
5: Initialize network parameters $\theta, \phi, \psi$ 6: while $t_{env} \leq t_{max}$ do  
7: Interact with the environment via $\epsilon$ -greedy policy based on $[Q_{\theta}^{i}]_{i=1}^{n}$ and get a trajectory $\mathcal{T}$ .  
8: Run Algorithm 2 to update $\mathcal{D}_E$ with $\mathcal{T}$ 9: Append $\mathcal{T}$ to $\mathcal{D}$ 10: for $k = 1$ to $n_{circle}$ do  
11: Get $M$ sample trajectories $[\mathcal{T}]_{i=1}^{M} \sim \mathcal{D}$ 12: Run MARL training algorithm using $[\mathcal{T}]_{i=1}^{M}$ and $\mathcal{D}_E$ , to update $\theta$ with Eq.10  
13: end for  
14: if $t_{env}$ mod $t_{emb} == 0$ then  
15: Run Algorithm 1 to update $\phi, \psi$ 16: Update all $x \in \mathcal{D}_E$ with updated $f_{\phi}$ 17: end if  
18: end while

Here, any CTDE training algorithm can be adopted for MARL training algorithm in line 12 in Algorithm 3. As we mentioned in Section C.4, training of $f _ { \phi }$ and $f _ { \psi }$ and updating all $\boldsymbol { x } \in \mathcal { D } _ { E }$ only

takes less than two seconds at most under the task with largest state dimension such as corridor. Thus, the computation burden for trainable embedder is negligible compared to the original MARL training.

# F MEMORY UTILIZATION

A remaining issue in utilizing episodic memory is how to determine a proper threshold value $\delta$ in Eq. 1. Note that this $\delta$ is used for both updating the memory and recalling the memory. One simple option is determining $\delta$ based on prior knowledge or experience, such as hyperparameter tuning. Instead, in this section, we present a more memory-efficient way for $\delta$ selection. When computing $| | \hat { x } - x | | _ { 2 } < \delta$ , the similarity is compared elementwisely. However, this similarity measure puts a different weight on each dimension of $x$ since each dimension of $x$ could have a different range of distribution. Thus, instead of $x$ , we utilize the normalized value. Let us define a normalized embedding $y$ with the statistical mean $( \mu _ { x } )$ and variance $( \sigma _ { x } )$ of $x$ as

$$
y = \left(x - \mu_ {x}\right) / \sigma_ {x}. \tag {21}
$$

Here, the normalization is conducted for each dimension of $x$ . Then, the similarity measure via $| | \hat { y } - y | | _ { 2 } < \delta$ with Eq. 21 puts an equal weight to each dimension, as $y$ has a similar range of distribution in each dimension. In addition, an affine projection of Eq. 21 maintains the closeness of original $x$ -distribution, and thus we can safely utilize $y$ -distribution instead of $x$ -distribution to measure the similarity.

In addition, $y$ defined in Eq. 21 nearly follows the normal distribution, although it does not strictly follow it. This is due to the fact that the memorized samples $x$ in $\mathcal { D } _ { E }$ do not originate from the same distribution, nor are they uncorrelated, as they can stem from the same episode. However, we can achieve an approximate coverage of the majority of the distribution, specifically $3 \sigma _ { y }$ in both positive and negative directions of $y$ , by setting $\delta$ as

$$
\delta \leq \frac {\left(2 \times 3 \sigma_ {y}\right) ^ {\dim (y)}}{M}. \tag {22}
$$

For example, when $M = 1 e ^ { 6 }$ and $\mathrm { d i m } ( y ) = 4$ , if $\sigma _ { y } \approx 1$ then $\delta \leq 0 . 0 0 1 3$ . This is the reason we select $\delta = 0 . 0 0 1 3$ for the exploratory memory recall.
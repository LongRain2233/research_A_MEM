# MULTI-AGENT IN-CONTEXT COORDINATION VIA DECENTRALIZED MEMORY RETRIEVAL

A PREPRINT

Tao Jiang1,2, Zichuan $\mathbf { L i n ^ { 3 } }$ ∗, Lihe $\mathbf { L i } ^ { 1 , 2 }$ , Yi-Chen $\mathbf { L i } ^ { 1 , 2 }$ , Cong Guan1,2,

Lei Yuan1,2, Zongzhang Zhang1,2∗, Yang $\mathbf { Y } \mathbf { u } ^ { 1 , 2 }$ , Deheng Ye3

1 National Key Laboratory of Novel Software Technology, Nanjing University, Nanjing, China

2 School of Artificial Intelligence, Nanjing University, Nanjing, China

3 Tencent, Shenzhen, China

{jiangt,lilh,liyc,guanc,yuanl}@lamda.nju.edu.cn, {zzzhang, yuy}@nju.edu.cn,

{zichuanlin, dericye}@tencent.com

# ABSTRACT

Large transformer models, trained on diverse datasets, have demonstrated impressive few-shot performance on previously unseen tasks without requiring parameter updates. This capability has also been explored in Reinforcement Learning (RL), where agents interact with the environment to retrieve context and maximize cumulative rewards, showcasing strong adaptability in complex settings. However, in cooperative Multi-Agent Reinforcement Learning (MARL), where agents must coordinate toward a shared goal, decentralized policy deployment can lead to mismatches in task alignment and reward assignment, limiting the efficiency of policy adaptation. To address this challenge, we introduce Multi-Agent In-Context Coordination via Decentralized Memory Retrieval (MAICC), a novel approach designed to enhance coordination by fast adaptation. Our method involves training a centralized embedding model to capture fine-grained trajectory representations, followed by decentralized models that approximate the centralized one to obtain team-level task information. Based on the learned embeddings, relevant trajectories are retrieved as context, which, combined with the agents’ current sub-trajectories, inform decision-making. During decentralized execution, we introduce a novel memory mechanism that effectively balances test-time online data with offline memory. Based on the constructed memory, we propose a hybrid utility score that incorporates both individual- and team-level returns, ensuring credit assignment across agents. Extensive experiments on cooperative MARL benchmarks, including Level-Based Foraging (LBF) and SMAC (v1/v2), show that MAICC enables faster adaptation to unseen tasks compared to existing methods. Code is available at https://github.com/LAMDA-RL/MAICC.

# 1 Introduction

In-Context Learning (ICL) has emerged as a compelling paradigm for few-shot generalization, enabling models to tackle novel tasks by interpreting contextual cues without explicit retraining [Brown et al.(2020)]. This approach is epitomized by large language models, whose remarkable in-context abilities, unlocked through pretraining on vast web-scale corpora, have set new standards in natural language processing [Dong et al.(2024)]. The success of this paradigm has catalyzed a parallel pursuit within Reinforcement Learning (RL) to instill agents with similar onthe-fly policy adaptation capabilities [Moeini et al.(2025)]. To this end, the prevailing strategy reformulates RL as a sequence modeling problem [Chen et al.(2021)]: agents are trained on diverse trajectory datasets to internalize learning algorithms, allowing them to adapt to novel downstream tasks by conditioning on a few contextual examples [Laskin et al.(2023), Lee et al.(2023)]. This burgeoning field of In-Context Reinforcement Learning (ICRL) has shown notable promise, but its success has mainly been demonstrated in structured environments such as single-agent grid worlds and game-based tasks.

ICRL has demonstrated strong capabilities for fast adaptation in single-agent environments. Typically, these methods condition on in-context trajectories and maintain a memory that is continuously updated with new online experiences to inform decision-making. Despite its notable success, extending this paradigm to cooperative Multi-Agent Reinforcement Learning (MARL) scenarios presents significant challenges. Unlike the single-agent setting, where an agent aims to maximize its individual cumulative rewards, cooperative MARL requires multiple agents to collaborate towards a shared objective [Yuan et al.(2023)]. This collaborative nature introduces distinct challenges, particularly when deployed in a decentralized manner [Kraemer and Banerjee(2016)]. Firstly, decentralized execution confines each agent to its local observations, often leading to a biased or incomplete understanding of the overall task characteristics. Secondly, agents typically receive only a shared team-level reward, making it difficult to assess individual contributions. This ambiguity in credit assignment can lead to the “lazy agent" problem [Sunehag et al.(2018)], where certain agents fail to learn effective policies and contribute meaningfully to the team’s success. These twin challenges of partial observability and credit assignment critically undermine the efficacy of conventional ICRL approaches in the MARL setting. Therefore, given the proven adaptive capabilities of ICL, a method that enables efficient adaptation to unseen cooperative tasks in decentralized multi-agent settings is urgently needed.

To address the above objective, we propose Multi-Agent In-Context Coordination via Decentralized Memory Retrieval (MAICC), a framework designed for rapid team coordination under Decentralized Partially Observable Markov Decision Processes (Dec-POMDPs) [Oliehoek and Amato(2016)]. Specifically, we train a single centralized embedding model to extract fine-grained trajectory representations, and multiple decentralized embedding models for decentralized execution that approximate the centralized model to obtain team-level task information. With these pretrained models, we retrieve relevant multi-agent trajectories to serve as in-context examples. These retrieved trajectories, combined with agents’ current sub-trajectories, are used to guide and improve the decision-making process. During test time, we introduce a novel memory mechanism that efficiently balances an online replay buffer with a multi-task offline dataset for trajectory retrieval. Building upon this memory, we design a hybrid utility score that integrates both individual- and team-level returns, thereby enabling more accurate credit assignment across agents.

We evaluate our method on several standard cooperative MARL benchmarks, including Level-Based Foraging (LBF) [Papoudakis et al.(2021)] and the StarCraft Multi-Agent Challenge (SMAC) v1 and v2 [Samvelyan et al.(2019), Ellis et al.(2023)]. Experimental results show that MAICC, equipped with efficient trajectory retrieval for ICL, enables significantly faster adaptation to unseen tasks compared to existing ICRL and multi-task MARL baselines. Additionally, visualizations of the learned trajectory embeddings verify the effectiveness of our embedding model design, capturing both individual- and team-level behavior patterns. Ablation studies further isolate and confirm the contribution of each key component in our framework. Together, these findings demonstrate MAICC’s strong empirical performance, its ability to address the limitations of prior ICRL approaches, and its potential for broader deployment in complex multi-agent scenarios.

# 2 Related Work

In-context RL. By framing RL as a sequence modeling problem, Decision Transformer (DT) [Chen et al.(2021)] can make decisions based on provided prompts [Xu et al.(2022)]. Subsequent studies scaled up model size and training data, enabling agents to exhibit ICL capabilities [Lee et al.(2022a), Reed et al.(2022)]. Algorithm Distillation [Laskin et al.(2023)] takes a significant step towards ICRL by utilizing historical trajectories. This enables agents to automatically improve their performance through trial and error, without updating their parameters. Agentic Transformer (AT) [Liu and Abbeel(2023)] further demonstrates that cross-episodic contexts can help agents leverage hindsight, thus enabling performance improvement at test time [Huang et al.(2024a)]. Decision-Pretrained Transformer (DPT) [Lee et al.(2023)] explores an alternative approach by predicting the optimal action given random historical trajectories and the current state. Subsequently, Retrieval-Augmented Decision Transformer (RADT) [Schmied et al.(2024)] introduces retrieval augmentation into ICRL, utilizing a DT-based embedding model to select relevant historical trajectories and thereby further aid action prediction. However, these methods have only demonstrated effectiveness on single-agent tasks with simple interactions [Sridhar et al.(2025)] and perform poorly on more complex decentralized cooperative tasks. In contrast, our approach achieves efficient trajectory retrieval tailored to the characteristics of Multi-Agent Systems (MASs), thereby facilitating collaborative adaptation to unseen tasks. To the best of our knowledge, our method is the first ICRL approach for Dec-POMDPs [Oliehoek and Amato(2016)].

Cooperative multi-agent RL. Many real-world problems are large-scale and complex, rendering single-agent modeling inefficient and often impractical [Feng et al.(2025)]. These challenges are more effectively addressed within the MAS setting [Dorri et al.(2018)], where MARL provides a robust framework for solution [Yuan et al.(2023)]. In cooperative MARL, where agents pursue shared objectives, significant progress has been made in domains such as video games [Li et al.(2025)], domain calibration [Jiang et al.(2024)], and financial trading [Huang et al.(2024b)]. A central challenge in cooperative MARL is partial observability due to decentralized execution. The Centralized

Training with Decentralized Execution (CTDE) framework [Lowe et al.(2017)] addresses this by propagating teamlevel information to individual agents during training, thereby enhancing coordination at execution. Another key issue is the absence of individual rewards, which leads to the “lazy agent" problem [Sunehag et al.(2018)], where agents fail to improve their policies due to an inability to assess their own contributions. Actor-critic methods such as COMA [Foerster et al.(2018)] mitigate this by introducing counterfactual baselines for policy updates, while valuebased approaches like QMIX [Rashid et al.(2020)] achieve implicit credit assignment by enforcing monotonicity in the value function [Son et al.(2019), Wang et al.(2020)]. In this work, we address both challenges within the ICRL framework by incorporating corresponding modules, thereby enabling rapid adaptation to unseen cooperative tasks.

# 3 Background

# 3.1 Multi-Agent Reinforcement Learning

A cooperative multi-agent task is typically modeled as a Dec-POMDP [Oliehoek and Amato(2016)], defined by the tuple $\mathcal { M } = \langle \mathcal { S } , \mathcal { A } , \mathcal { T } , R , \Omega , O , \mathcal { N } , H , \rho \rangle$ . Here, $s$ and $\mathcal { A }$ denote the state and action spaces, respectively; $\rho \in \Delta ( \mathcal { S } )$ is the initial state distribution where $\Delta ( S )$ represents the set of probability distributions over the state space $s$ ; $H \in \mathbb { N }$ is the episode horizon where $\mathbb { N }$ denotes the set of natural numbers. Each episode begins with an initial state $s ^ { 0 }$ sampled from $\rho$ . At each time step $h$ , given the global state $s ^ { h } \in \mathcal { S }$ , each agent $j \in \mathcal { N } = \{ 1 , 2 , \cdots , n \}$ receives a local observation $o _ { j } ^ { h } \in \Omega$ generated by the observation function $O ( s ^ { h } , j )$ , and selects an action $a _ { j } ^ { h } \in { \mathcal { A } }$ according to its individual learnable policy $\pi _ { j } ( a _ { j } ^ { h } | \tau _ { j } ^ { h } )$ . Here, $\tau _ { j } ^ { h }$ denotes the trajectory $( o _ { j } ^ { 0 } , a _ { j } ^ { 0 } , \cdots , o _ { j } ^ { h } )$ . The joint action is denoted as $\mathbf { a } ^ { h } = ( a _ { 1 } ^ { h } , a _ { 2 } ^ { h } , \cdot \cdot \cdot , a _ { n } ^ { h } )$ . The environment then transitions to the next state $s ^ { \bar { h } + 1 } \sim \bar { \mathcal { T } } ( \cdot | s ^ { h } , \mathbf { a } ^ { h } )$ and provides a global reward $r ^ { h } = R ( s ^ { h } , \mathbf { a } ^ { h } )$ . The episode terminates when a predefined condition is met or after $H$ steps. The objective is to optimize the joint policy $\pmb { \pi } = ( \pi _ { 1 } , \pi _ { 2 } , \cdots , \pi _ { n } )$ to maximize the value function $\begin{array} { r } { V ^ { \mathcal { M } } ( \pmb { \pi } ) = \mathbb { E } _ { \pmb { \pi } } \left[ \sum _ { h = 0 } ^ { H - 1 } r ^ { h } \right] } \end{array}$

# 3.2 Decision Transformer

Transformers, originally developed for sequence modeling in language tasks [Vaswani et al.(2017)], have been applied to RL by Decision Transformer (DT) [Chen et al.(2021)], which frames decision-making as sequence modeling [Janner et al.(2021)]. Instead of learning value functions as in traditional RL methods, DT derives policies from sequences of input tokens within a single trajectory, represented as $( \hat { R } ^ { 0 } , o ^ { 0 } , a ^ { 0 } , \hat { R } ^ { 1 } , o ^ { 1 } , a ^ { 1 } , \cdot \cdot \cdot )$ , where each token corresponds to the Return-To-Go (RTG), obis defined as the sum of future rewards: tion, respectively. The RTG at time step . DT is trained in a supervised manner, s $h$ , denoted as milar to beha $\hat { R } ^ { h }$ $\begin{array} { r } { \hat { R } ^ { h } = \sum _ { t = h } ^ { H - 1 } r ^ { t } } \end{array}$ cloning (BC) [Atkeson and Schaal(1997)]. During testing, by conditioning on a high RTG, DT can autoregressively generate actions aimed at achieving high cumulative rewards (return).

# 3.3 Problem Setting

In this paper we study ICRL [Moeini et al.(2025)], a practical form of meta-RL [Beck et al.(2023)], where agents learn new cooperative Dec-POMDP tasks sampled from $P ( \mathcal { M } )$ via limited online trials without updating model parameters. During training agents only access datasets $\mathcal { D } = \{ \mathcal { D } _ { i } \}$ of trajectories collected under unknown cooperative policies $\pmb { \mu }$ in tasks $\mathcal { M } _ { i }$ . After pretraining, the model parameters are fixed. At test time, the agent team interacts with a new, unseen environment randomly sampled from $P ( \mathcal { M } )$ for only $T$ episodes, with the goal of achieving fast coordination without parameter updating. This objective can be formulated as maximizing the expected return in the final adaptation episode under the task distributions: max $\mathbb { E } _ { \mathcal { M } \sim P ( \cdot ) } V ^ { \mathcal { M } } ( \pi )$ .

# 4 Method

In this section, we present the MAICC (Multi-Agent In-Context Coordination) framework, which exploits the ICL capabilities of Transformer-based models for rapid adaptation to unseen cooperative tasks. The overall architecture is shown in Fig. 1. During training, we first learn embedding models to capture the characteristics of multi-agent trajectories for efficient context retrieval (Sec. 4.1). Specifically, a centralized embedding model (CEM) extracts team-level information via autoregressive prediction, which guides decentralized embedding models (DEMs) for decentralized execution. The DEMs are then used to retrieve trajectories with similar embeddings for a given input, enabling the decision model to generate appropriate actions (Sec. 4.2). By leveraging these in-context trajectories, the pretrained decision model can infer task characteristics and generalize across diverse tasks. In the testing phase, we introduce a novel memory mechanism that combines an online replay buffer with offline datasets to enhance retrieval efficiency.

![](images/dd5f17aa6e3872193447193cbd0bd3c36aceb6b9ebb0aa81641d5ba24114b9ba.jpg)  
Figure 1: The conceptual workflow of MAICC. Dashed lines show data flow during centralized training, where CEM samples offline trajectories for training and distills team information to DEMs. Solid lines show data flow during decentralized execution, where sub-trajectories retrieve trajectories from the constructed memory based on similarity and hybrid utility score. Blue $\circ$ and purple $\bigtriangledown$ denote different embeddings output by CEM and DEMs, respectively. $\oplus$ denotes concatenation of retrieved trajectories with the current sequence, which helps decision models adapt quickly.

We further propose a hybrid utility score that integrates individual- and team-level information to select high-quality in-context trajectories, promoting effective coordination (Sec. 4.3). Finally, we provide a theoretical analysis of the online cumulative regret of our approach (Sec. 4.4).

# 4.1 Multi-Agent Trajectory Embedding Models

Efficient multi-agent trajectory retrieval relies on learning high-quality trajectory embeddings. To achieve this, we adopt the CTDE paradigm [Kraemer and Banerjee(2016), Lowe et al.(2017)] and design both centralized and decentralized embedding models. During training, agents have access to global team observations and actions, allowing the CEM to capture fine-grained team-level information. In contrast, during execution, each agent is limited to its own local observations and actions, resulting in less informative embeddings from the DEMs. To address this disparity, we employ the CEM to distill team-level knowledge into the DEMs during training, thereby enhancing the DEMs’ representational capacity for decentralized execution.

Formally speaking, we denote the number of agents as $n$ . The multi-task offline dataset $\mathcal { D }$ consists of trajectories $\tau = ( \stackrel { \star } { o ^ { 0 } } , \stackrel { \star } { a ^ { 0 } } , r ^ { 0 } , \stackrel {  } { \dots } , \stackrel { \quad } { o ^ { H - 1 } } , { a ^ { H - 1 } } , r ^ { H - 1 } ) .$ , where $\pmb { o } = \left( o _ { 1 } , \ldots , o _ { n } \right)$ and $\pmb { a } = ( a _ { 1 } , \ldots , a _ { n } )$ . Our trajectory embedding models employ three types of tokens: observation $o$ , action $a$ , and post-step information $\hat { P }$ . Following prior work [Liu and Abbeel(2023), Huang et al.(2024a)], the token $\hat { P }$ comprises the global reward, done signal, and task completion flag, which are essential for modeling long-horizon trajectories. We omit the RTG token, as it can cause retrieval of trajectories from irrelevant tasks that happen to have similar RTG values, thereby reducing the informativeness of in-context examples and harming action prediction.

As illustrated in Fig. 2, the CEM receives the agents’ local observations $\{ o _ { j } ^ { h } \} _ { j = 1 } ^ { n }$ , actions $\{ a _ { j } ^ { h } \} _ { j = 1 } ^ { n }$ , and post-step information ${ \hat { P } } ^ { h }$ at each time step $h$ , and outputs the corresponding embeddings: $\{ Z _ { o , j } ^ { h } \} _ { j = 1 } ^ { n } , \{ Z _ { a , j } ^ { h } \} _ { j = 1 } ^ { n } , Z _ { p } ^ { h } \ =$ $\mathrm { C E M } ( \{ o _ { j } ^ { h } \} _ { j = 1 } ^ { n } , \{ a _ { j } ^ { h } \} _ { j = 1 } ^ { n } , \hat { P } ^ { h } )$ . To be compatible with centralized training, we adapt the causal transformer by introducing intra-team visibility, enabling observation and action tokens within the same team and time step to attend to each other. We further design three loss functions to model the behavior policy $( \mathcal { L } _ { \mu } )$ , reward function $( \mathcal { L } _ { R } )$ , and observation transition dynamics $( \mathcal { L } _ { T } )$ of the trajectory:

$$
\mathcal {L} _ {\mathrm {C E M}} = \mathcal {L} _ {\mu} + \mathcal {L} _ {R} + \mathcal {L} _ {\mathcal {T}}, \tag {1}
$$

$$
\mathcal {L} _ {\mu} = - \mathbb {E} _ {\tau \sim \mathcal {D}} \sum_ {h = 0} ^ {H - 1} \sum_ {j = 1} ^ {n} \log \mathrm {M L P} _ {o \rightarrow a} \left(a _ {j} ^ {h} \mid Z _ {o, j} ^ {h}\right), \tag {2}
$$

$$
\mathcal {L} _ {R} = \mathbb {E} _ {\tau \sim \mathcal {D}} \sum_ {h = 0} ^ {H - 1} \left(r ^ {h} - \sum_ {j = 1} ^ {n} \mathrm {M L P} _ {a \rightarrow r} \left(Z _ {a, j} ^ {h}\right)\right) ^ {2}, \tag {3}
$$

$$
\mathcal {L} _ {\mathcal {T}} = - \mathbb {E} _ {\tau \sim \mathcal {D}} \sum_ {h = 0} ^ {H - 2} \sum_ {j = 1} ^ {n} \log \mathrm {M L P} _ {p \rightarrow o} \left(o _ {j} ^ {h + 1} \mid Z _ {p} ^ {h}, o _ {j} ^ {h}\right), \tag {4}
$$

![](images/099d3f47eb485a253166341b260e0114ff349afe4a71e632ae1b6f4dcd6e45aa.jpg)  
Figure 2: The illustration of CEM. Intra-team visibility enables observation and action tokens within the same team to attend to each other at each time step. The causal transformer predicts individual actions and rewards, while the poststep information token, concatenated with the previous individual observation, is used to predict the next observation.

where MLPs with different subscripts fit different functions. Eq. 3 can be regarded as performing implicit credit assignment [Sunehag et al.(2018), Rashid et al.(2020)], which benefits subsequent decentralized adaptation.

During decentralized execution, the DEMs capture embeddings using only local information, i.e., $z _ { o , j } ^ { h } , z _ { a , j } ^ { h } , z _ { p } ^ { h } =$ $\mathrm { D E M } ( o _ { j } ^ { h } , a _ { j } ^ { h } , \hat { P } ^ { h } )$ . To enhance coordination, we introduce auxiliary objectives that distill team-level information by minimizing the KL divergence between the embeddings generated by the CEM and those produced by the DEMs:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm {D E M}} = \mathbb {E} _ {\tau \sim \mathcal {D}} \sum_ {h = 0} ^ {H - 1} \sum_ {j = 1} ^ {n} \left[ \mathrm {K L} \left(Z _ {o, j} ^ {h}, z _ {o, j} ^ {h}\right) + \mathrm {K L} \left(Z _ {a, j} ^ {h}, z _ {a, j} ^ {h}\right) \right] \\ + \mathbb {E} _ {\tau \sim \mathcal {D}} \sum_ {h = 0} ^ {H - 1} \mathrm {K L} \left(Z _ {p} ^ {h}, z _ {p} ^ {h}\right), \tag {5} \\ \end{array}
$$

where $\mathrm { K L } ( p , q )$ measures the divergence from the target distribution $p$ to the approximate distribution $q$

# 4.2 Retrieval-Based In-Context Decision Training

To address diverse cooperative tasks with a single decision model, we use the trained DEMs to retrieve trajectories that inform action generation. Given an individual query sub-trajectory $\tau _ { j } ^ { q } = ( o _ { j } ^ { 0 } , a _ { j } ^ { 0 } , r ^ { 0 } , \cdot \cdot \cdot , a _ { j } ^ { q - 1 } , r ^ { q - 1 } , o _ { j } ^ { q } ) \sim \mathcal { D }$ up to a certain time step, we first input it into the DEM and extract the embeddings at the final step, which, due to the transformer’s long-range dependency modeling, summarizes the entire sub-trajectory. We textracted embeddings over different tokens to obtain the final query embedding, i.e., $z _ { j } ^ { q } = \overset { \cdot } { \mathrm { M E A N } } ( z _ { o , j } ^ { q } , z _ { a , j } ^ { q - 1 } , z _ { p } ^ { q - 1 } )$ Using Maxitrajectories: cosine simil ouze et, where g compu .(2024)], we retrieve the top-is the number of in-context td in the same manner as the $k$ most relevant in-contextjectories, cossim denotesery. $\mathcal { C } ( \tau _ { j } ^ { q } ) = \arg \operatorname* { m a x } _ { \tau ^ { c } \in \mathcal { D } } ^ { k } \cos \sin ( z ^ { c } , z _ { j } ^ { q } )$ $k$ $z ^ { c }$

The retrieved in-context trajectories provide additional task-specific information to the current sub-trajectory. We concatenate these trajectories with the query and train the decision model $\pi _ { \theta }$ (a causal transformer with parameter $\theta$ sharing across agents) using the following loss function:

$$
\mathcal {L} _ {\pi} = - \mathbb {E} _ {\tau_ {j} ^ {q}} \sim_ {\mathcal {D}} \log \pi_ {\theta} \left(a _ {j} ^ {q} | \text {C O N C A T} \left(\mathcal {C} \left(\tau_ {j} ^ {q}\right), \tau_ {j} ^ {q}\right)\right), \tag {6}
$$

where CONCAT denotes the concatenate function. It is worth noting that, in addition to the three types of tokens—observation, action, and post-step information—the decision model also receives a RTG token. Unlike the embedding models, the decision model leverages the RTG from the retrieved trajectory to guide action selection towards achieving the desired return.

# 4.3 Decentralized In-Context Fast Coordination

After pretraining the embedding and decision models, a new task is randomly sampled from the task distribution $P ( \mathcal { M } )$ . The agent team must then rapidly adapt and coordinate on this task without further parameter updates. During $T$ episodes of interaction, data are stored in an online replay buffer. Agents can retrieve trajectories from both the multi-task offline dataset $\mathcal { D }$ , which may exhibit distribution shift, and the online buffer $\boldsymbol { B }$ , which is aligned with the current task but initially contains limited experience. To address this, we propose a selective memory mechanism with exponential time decay: early episodes prioritize offline data to encourage exploration, while later episodes increasingly leverage high-value online trajectories to enhance exploitation. Specifically, we introduce a coefficient $\begin{array} { r } { \beta _ { t } = \exp \left( - \lambda \frac { t } { T } \right) } \end{array}$ for episode $t$ , where the hyper-parameter $\lambda$ controls the decay rate [Ross et al.(2011)]. We construct a new buffer $B ^ { \prime }$ by sampling from $\mathcal { D }$ with probability $\beta _ { t }$ and from $\boldsymbol { B }$ with probability $1 - \beta _ { t }$ . This method is simple, effective, and theoretically grounded.

Based on the constructed memory $B ^ { \prime }$ , we further enhance the exploitation of high-value trajectories by introducing a hybrid utility score during inference, defined as $S _ { \mathrm { u t i l } } ( \tau ) = \alpha \mathrm { n o r m } ( \mathcal { R } ) + ( 1 - \alpha ) \mathrm { n o r m } ( \tilde { \mathcal { R } } )$ . Here, $\begin{array} { r } { \dot { \mathcal { R } } = \sum _ { h = 0 } ^ { H - 1 } \dot { r } ^ { h } } \end{array}$ is the global return, $\begin{array} { r } { \tilde { \mathcal { R } } = \sum _ { h = 0 } ^ { H - 1 } \tilde { r } _ { j } ^ { h } } \end{array}$ is the predicted individual return for agent $j$ , $\mathrm { n o r m } ( \cdot )$ denotes normalization to [0, 1], and $\alpha \in [ 0 , 1 ]$ is a hyper-parameter. In Dec-POMDPs, where individual rewards are unavailable, we leverage the pretrained DEMs to predict individual rewards from action embeddings, i.e., $\tilde { r } _ { j } ^ { h } = \mathrm { M L P } _ { a  r } ( z _ { a , j } ^ { h } )$ . This hybrid utility score enables agents to retrieve trajectories that are beneficial at both the individual and team levels, thereby mitigating the “lazy agent" problem in multi-agent systems. Incorporating the similarity score used during training, the retrieval process is formulated as $\begin{array} { r } { \mathscr { C } ( \tau _ { j } ^ { q } ) = \arg \operatorname* { m a x } _ { \tau ^ { c } } \boldsymbol { S } ( \tau ^ { c } , \tau _ { j } ^ { q } ) } \end{array}$ , where $\tau ^ { c } \in \mathcal { B } ^ { \prime }$ and $S ( \tau ^ { c } , \tau _ { j } ^ { q } ) = \mathrm { c o s s i m } ( z ^ { c } , z _ { j } ^ { \bar { q } } ) + S _ { \mathrm { u t i l } } ( \tau ^ { c } )$ . The decision model then outputs actions conditioned on the concatenation of the retrieved in-context trajectories and the input trajectory: $a \sim \pi _ { \boldsymbol { \theta } } ( \cdot | \mathrm { C O N C A T } ( \mathcal { C } ( \tau _ { j } ^ { q } ) , \tau _ { j } ^ { q } ) )$ . The overall pseudo code of MAICC is provided in Alg. 1.

Algorithm 1 Multi-agent In-context Coordination via Decentralized Memory Retrieval   
Input: Initialized two trajectory embedding models CEM, DEM, decision model $\pi_{\theta}$ , multi-task offline dataset $\mathcal{D}$ , empty online replay buffer $\mathcal{B}$ 1: // Multi-Agent Trajectory Embedding Models  
2: while not converged do  
3: Update CEM and DEM by Eq. 1 and Eq. 5 on $\mathcal{D}$ 4: end while  
5: // Retrieval-Based In-Context Decision Training  
6: while not converged do  
7: Retrieve in-context trajectories $\mathcal{C}$ with DEM  
8: Update $\pi_{\theta}$ with $\mathcal{C}$ by Eq. 6  
9: end while  
10: // Decentralized In-Context Fast Adaptation  
11: for $t = 1, 2, \dots, T$ do  
12: Construct the memory $\mathcal{B}'$ 13: while episode not ended do  
14: Retrieve in-context trajectories $\mathcal{C}$ with $\mathcal{S}$ and $\mathcal{B}'$ 15: Decentralized execution with $\pi_{\theta}$ conditioned on $\mathcal{C}$ 16: end while  
17: Store episode trajectory in $\mathcal{B}$ 18: end for

# 4.4 Theoretical Analysis

In this section, we provide a bound on the online cumulative regret of MAICC. For a given task $\mathcal { M }$ with $| \Omega | = \omega$ , $| { \mathcal { A } } | = A$ $H$ $\pi ^ { * }$ ote the, where trieves t tive regret ovewith subscript m the offline da $T$ sodes ie MAI, while efined as policy inretrieves $\begin{array} { r } { \mathbf { R e g } _ { \mathcal { M } } = \sum _ { t = 1 } ^ { T } V ^ { \mathcal { M } } ( \pi ^ { * } ) - V ^ { \mathcal { M } } ( \hat { \pi } _ { t } ) } \end{array}$ $\hat { \pi } _ { t } = \beta _ { t } \pi ^ { \mathcal { D } } + ( 1 - \beta _ { t } ) \pi _ { t } ^ { \mathcal { B } }$ $t$ $t$ $\pi ^ { \mathcal { D } }$ $\mathcal { D }$ $\pi _ { t } ^ { B }$ them from the online buffer $\boldsymbol { B }$ accumulated up to episode $t$ , described in Sec. 4.3.

Assumption 1. (Sufficiency of Retrieval) Let $\pi _ { t } ^ { B * }$ denote the policy that, for each query $\tau ^ { q }$ , directly uses the entire online buffer accumulated over t episodes as the in-context input (i.e., without retrieval). For all $( \tau ^ { q } , \boldsymbol { B } , t )$ , we have $\pi _ { t } ^ { B } ( a | \tau ^ { q } ) { \overset { \sim } { = } } \pi _ { t } ^ { B * } ( a | \tau ^ { q } )$ for all $a \in { \mathcal { A } }$ .

![](images/34afd93020e7835181b16c73ec0d268e59839a940ac48734d829984b36f091e0.jpg)  
Figure 3: In-context adaptation performance across different scenarios. Each scenario is evaluated over 50 test runs on randomly sampled tasks, with results reported as the mean return and $9 5 \%$ confidence interval.

This assumption is trivially satisfied if the number of retrieved trajectories $k$ equals $t$ . Even when $k < t$ , a carefully selected $k$ trajectories can still capture most of the relevant information. Since Transformer inference time scales quadratically with context length, using a representative subset rather than the entire buffer is both efficient and practical.

Theorem 1. Suppose $\operatorname* { s u p } _ { \mathcal { M } } P ( \mathcal { M } ) / P _ { \mathcal { D } } ( \mathcal { M } ) \leq C$ for some $C > 0 ;$ , where $P _ { \mathcal { D } } ( \mathcal { M } )$ denotes the training task distribution. Then the expected online cumulative regret of MAICC satisfies $\mathbb { E } _ { P ( \mathcal { M } ) } [ \mathbf { R e g } _ { \mathcal { M } } ] \leq \tilde { \mathcal { O } } ( C H ^ { 3 / 2 } \omega \sqrt { A T } )$ .

MAICC offers a theoretical guarantee similar to prior ICRL methods [Schmied et al.(2024), Lee et al.(2023), Jing et al.(2024)] as $\tilde { \mathcal { O } }$ is Big-O ignoring poly-logarithmic factors in complexity. In practice, however, the initial online replay buffer may lack sufficiently informative trajectories, leading to inefficient exploration. By leveraging our selective memory mechanism, MAICC adapts to new tasks more efficiently. Experimental results further support this advantage, and detailed derivations are provided in Appendix B.

# 5 Experiments

In this section, we evaluate the proposed MAICC framework empirically. We begin by describing the experimental environments and baseline methods in Sec. 5.1. We then conduct a series of experiments to address the following questions: (1) How does MAICC compare to various baselines in terms of fast coordination (Sec. 5.2)? (2) How effectively do the DEMs capture representations of multi-agent trajectories (Sec. 5.3)? (3) What is the contribution of each component of MAICC to overall performance (Sec. 5.4)?

# 5.1 Experiment Setup

We evaluate MAICC and baseline methods on several cooperative benchmarks. The first is the Level-Based Foraging (LBF) [Papoudakis et al.(2021)], a grid-world environment where agents must coordinate to collect food items simultaneously. Each agent observes only its local field of view and must collect food at different locations for each task, within a limited number of time steps. We consider two scenarios: LBF: $7 { x } 7 { - } I 5 s$ and LBF: $9 x 9 – 2 0 s$ , which differ in grid size and time limits. We further assess MAICC on the StarCraft Multi-Agent Challenge (SMAC) [Samvelyan et al.(2019)], using three sets of tasks where Protoss, Terran, and Zerg units cooperate to defeat enemy units controlled by the built-in AI of the same race. Each task features varying agent types and numbers,

![](images/aa1fe779a212bc7282c1de4053622363a3ac65970f9b1511712a3af70995901a.jpg)  
(a)

![](images/aec7b1255b7927363497ff63ada3c786f5f84a192441373a208ff19808d846e0.jpg)  
(b)

![](images/f893bc72f068c26f0842de31a1bd786fc455993cb3f1b4a8f2deacbc03445791.jpg)  
(c)

![](images/2adf9c5cb3a5a331762f4060b2164bebe800d23c5500ee53e726bfea0dc48b4b.jpg)  
(d)   
Figure 4: Visualization results illustrating the effects of different embedding model training settings. Each point in the figure represents the embedding of a trajectory from the dataset, with points of the same color corresponding to trajectories from the same task.

with corresponding enemy configurations. Additionally, we evaluate on the StarCraft Multi-Agent Challenge-v2 (SMACv2) [Ellis et al.(2023)], an extension of SMAC with increased randomness. For this benchmark, we further challenge MAICC by pretraining a single model to handle all three task types. For each scenario, we use QMIX [Rashid et al.(2020)] to train on multiple tasks, forming the multi-task offline dataset $\mathcal { D }$ . Further details on these benchmarks and datasets are provided in Appendix C.

MAICC is pretrained on a multi-task offline dataset and learns rapid coordination through online decentralized adaptation. For comparison, we select several baselines with similar settings. MADT [Meng et al.(2023)] extends DT to the multi-agent domain and achieves strong performance in single-task scenarios. AT [Liu and Abbeel(2023)] and RADT [Schmied et al.(2024)] are state-of-the-art in-context RL algorithms trained on offline data for online adaptation; while effective in single-agent settings, they lack designs specific to multi-agent coordination. HiSSD [Liu et al.(2025)] is a recent multi-task MARL algorithm that learns generalizable skills from a multi-task offline dataset, but does not support online adaptation. MAICC-S is an ablated version of our method, where only the DEM is trained for trajectory modeling during pretraining, without the CEM; all other components remain unchanged. Except for HiSSD, all methods are Transformer-based, and we use the same-size GPT-2 model [Radford et al.(2019)] for fair comparison.

Experimental results are obtained by training each model with 5 different random seeds. For each seed, performance is evaluated on 10 random tasks, yielding a total of 50 test runs. We report the mean and $9 5 \%$ confidence intervals. Additional implementation details are provided in Appendix D.

# 5.2 Main Results

We first evaluate the in-context adaptation performance of our method and the baselines across various scenarios. As shown in Fig. 3, agent teams are required to improve their average return over the task distribution within a limited number of episodes in each scenario. Our method consistently outperforms all baselines across six test scenarios, achieving faster adaptation to unseen cooperative tasks without requiring model parameter updates.

Since MADT and HiSSD lack online adaptation capabilities, their performance is shown as fixed horizontal lines. On SMAC-type tasks, their results are comparable to in-context RL baselines; however, on LBF tasks—where agent observability is more limited—their performance drops significantly, underscoring the importance of in-context adaptation. AT predicts actions based on trajectories from previous episodes, yielding good results only on the small LBF: $7 { x } 7 { - } I 5 s$ map. Although RA-DT also utilizes trajectory retrieval, its coarse-grained encoding and lack of adaptation for cooperative scenarios limit its effectiveness. The performance gap between MAICC-S and our method further demonstrates the necessity of explicitly modeling multi-agent characteristics in trajectory embeddings. Notably, in more complex SMAC scenarios, only our method exhibits clear in-context adaptation. The performance gap is most pronounced in the SMACv2: all scenario, which features the greatest task diversity, highlighting the strong potential of our approach in large-scale data settings.

# 5.3 Visualization of Learned Embeddings

To assess the effectiveness of the trajectory embedding model, we conduct visualization experiments. As shown in Fig. 4, for the SMACv2: all scenario, all trajectories are encoded and projected onto a two-dimensional plane using t-SNE [Maaten and Hinton(2008)]. Points with the same color represent trajectories from the same task.

We evaluate four different embedding configurations. In our proposed setting (Fig. 4(a)), the embedding models are trained without the RTG token and utilizes three loss functions, resulting in fine-grained embeddings where trajectories

Table 1: Ablation Study on MAICC. Unless otherwise noted, all settings follow the default configuration. “Ret.” indicates the average return over 50 test runs (with $9 5 \%$ confidence interval), evaluated in the final adaptation episode.   

<table><tr><td>Variants</td><td>EM With RTG</td><td>Coefficient β</td><td>CEM loss</td><td>Hyper-parameter α</td><td>SMACv2: all Ret.</td></tr><tr><td>Default</td><td>False</td><td>βt = exp(-λt/T)</td><td>Lμ + LR + LT</td><td>α = 0.8</td><td>14.51±0.46</td></tr><tr><td>(A)</td><td>True</td><td></td><td></td><td></td><td>13.52±0.62</td></tr><tr><td rowspan="2">(B)</td><td></td><td>βt = 0</td><td></td><td></td><td>12.16±0.72</td></tr><tr><td></td><td>βt = 1</td><td></td><td></td><td>11.17±0.64</td></tr><tr><td rowspan="3">(C)</td><td></td><td></td><td>Lμ + LR</td><td></td><td>13.43±0.51</td></tr><tr><td></td><td></td><td>Lμ + LT</td><td>α = 1</td><td>12.32±0.48</td></tr><tr><td></td><td></td><td>Lμ</td><td>α = 1</td><td>10.55±0.39</td></tr><tr><td rowspan="2">(D)</td><td></td><td></td><td></td><td>α = 1</td><td>13.61±0.40</td></tr><tr><td></td><td></td><td></td><td>α = 0</td><td>13.26±0.66</td></tr></table>

from the same task are grouped. In contrast, incorporating the RTG token (Fig. 4(b)) causes embeddings from the same task to form several small, overlapping clusters with those from other tasks, increasing the risk of retrieving irrelevant trajectories. In Fig. 4(c) and (d), only a subset of the loss functions is used to model trajectories [Schmied et al.(2024)]. While the model still encodes trajectories from the same task into nearby regions, overfitting occurs, resulting in overly compact clusters due to coarse-grained modeling. When tested on unknown tasks, such trajectory representations lack the generalization capability for extrapolated estimation. These findings highlight the importance of carefully designing both the embedding models and its associated loss functions for effective trajectory retrieval.

# 5.4 Ablation Study

We evaluated the importance of different MAICC components by systematically modifying the default model and measuring performance changes on the most challenging scenario, SMACv2: all, as shown in Tab. 1.

In row (A), we examine the effect of incorporating the RTG token during embedding model training. The results show degraded performance, likely due to an increased likelihood of retrieving irrelevant trajectories.

Row (B) explores different values of $\beta$ for memory construction. When the memory consists solely of either the offline dataset or the online buffer—instead of combining both sources using exponential time decay as coefficient—performance drops significantly. This indicates that each data source has limitations, and their weighted combination is crucial for effective adaptation to unseen tasks.

In row (C), we analyze the impact of different CEM loss functions on in-context adaptation. The results indicate that all three loss functions are necessary; fine-grained trajectory modeling enhances action prediction. Notably, omitting $\mathcal { L } _ { R }$ prevents individual return prediction during testing, further reducing overall performance.

Row (D) evaluates the role of the hybrid utility score. Using only the global return $( \alpha = 1 )$ ) leads to insufficient credit assignment, while relying solely on the predicted individual return $( \alpha = 0$ ) may suffer from prediction inaccuracies. Therefore, the hybrid approach, which combines both, yields improved adaptation performance.

# 6 Conclusion and Discussion

In this paper, we address rapid cooperative adaptation in Dec-POMDP settings by proposing the MAICC framework, which enables agent teams to quickly coordinate on unseen tasks without requiring parameter updates. During training, MAICC leverages the CEM to extract fine-grained representations of multi-agent trajectories and guides the DEM to optimize these representations for decentralized execution. Given a current sub-trajectory, agents use the DEM to retrieve and concatenate relevant trajectories for decision model training. During testing, each agent retrieves trajectories from a constructed memory that integrates both online buffer and offline data. Credit assignment is achieved by combining team- and individual-level returns. Experiments on cooperative MARL benchmarks demonstrate that MAICC enables rapid adaptation to previously unseen tasks. A potential constraint of MAICC is that relying solely on exponential time decay for memory construction may limit applicability in certain scenarios; incorporating uncertainty-based metrics [Lockwood and Si(2022)] could further enhance generalization and facilitate real-world deployment.

# References

[Atkeson and Schaal(1997)] Christopher G. Atkeson and Stefan Schaal. 1997. Robot learning from demonstration. In International Conference on Machine Learning. 12–20. 3.2   
[Ball et al.(2023)] Philip J Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. 2023. Efficient online reinforcement learning with offline data. In International Conference on Machine Learning. 1577–1594. A   
[Beck et al.(2023)] Jacob Beck, Risto Vuorio, Evan Zheran Liu, Zheng Xiong, Luisa Zintgraf, Chelsea Finn, and Shimon Whiteson. 2023. A survey of meta-reinforcement learning. arXiv preprint arXiv:2301.08028 (2023). 3.3   
[Brown et al.(2020)] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems. 1877–1901. 1   
[Chen et al.(2021)] Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Misha Laskin, Pieter Abbeel, Aravind Srinivas, and Igor Mordatch. 2021. Decision transformer: Reinforcement learning via sequence modeling. In Advances in Neural Information Processing Systems. 15084–15097. 1, 2, 3.2, D.1   
[Dong et al.(2024)] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. 2024. A survey on in-context learning. In Empirical Methods in Natural Language Processing. 1107–1128. 1   
[Dorri et al.(2018)] Ali Dorri, Salil S Kanhere, and Raja Jurdak. 2018. Multi-agent systems: A survey. IEEE Access 6 (2018), 28573–28593. 2   
[Douze et al.(2024)] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2024. The faiss library. arXiv preprint arXiv:2401.08281 (2024). 4.2   
[Ellis et al.(2023)] Benjamin Ellis, Jonathan Cook, Skander Moalla, Mikayel Samvelyan, Mingfei Sun, Anuj Mahajan, Jakob Nicolaus Foerster, and Shimon Whiteson. 2023. SMACv2: An improved benchmark for cooperative multi-agent reinforcement learning. In NeurIPS Datasets and Benchmarks Track. 1, 5.1, C.2   
[Feng et al.(2025)] Zhaohan Feng, Ruiqi Xue, Lei Yuan, Yang Yu, Ning Ding, Meiqin Liu, Bingzhao Gao, Jian Sun, Xinhu Zheng, and Gang Wang. 2025. Multi-agent embodied ai: Advances and future directions. arXiv preprint arXiv:2505.05108 (2025). 2   
[Foerster et al.(2018)] Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson. 2018. Counterfactual multi-agent policy gradients. In Proceedings of the AAAI conference on artificial intelligence. 2974–2982. 2   
[Hester et al.(2018)] Todd Hester, Matej Vecerík, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, Gabriel Dulac-Arnold, John P. Agapiou, Joel Z. Leibo, and Audrunas Gruslys. 2018. Deep q-learning from demonstrations. In Proceedings of the AAAI Conference on Artificial Intelligence. 3223–3230. A   
[Huang et al.(2024a)] Sili Huang, Jifeng Hu, Hechang Chen, Lichao Sun, and Bo Yang. 2024a. In-context decision transformer: reinforcement learning via hierarchical chain-of-thought. In International Conference on Machine Learning. 19871–19885. 2, 4.1   
[Huang et al.(2024b)] Yuling Huang, Chujin Zhou, Kai Cui, and Xiaoping Lu. 2024b. A multi-agent reinforcement learning framework for optimizing financial trading strategies based on TimesNet. Expert Systems with Applications 237 (2024), 121502. 2   
[Janner et al.(2021)] Michael Janner, Qiyang Li, and Sergey Levine. 2021. Offline reinforcement learning as one big sequence modeling problem. In Advances in Neural Information Processing Systems. 1273–1286. 3.2   
[Jiang et al.(2024)] Tao Jiang, Lei Yuan, Lihe Li, Cong Guan, Zongzhang Zhang, and Yang Yu. 2024. Multi-agent domain calibration with a handful of offline data. In Advances in Neural Information Processing Systems. 69607– 69636. 2   
[Jing et al.(2024)] Yuheng Jing, Kai Li, Bingyun Liu, Yifan Zang, Haobo Fu, Qiang Fu, Junliang Xing, and Jian Cheng. 2024. Towards offline opponent modeling with in-context learning. In International Conference on Learning Representations. 4.4

[Kiran et al.(2021)] B Ravi Kiran, Ibrahim Sobh, Victor Talpaert, Patrick Mannion, Ahmad A Al Sallab, Senthil Yogamani, and Patrick Pérez. 2021. Deep reinforcement learning for autonomous driving: A survey. IEEE Transactions on Intelligent Transportation Systems 23, 6 (2021), 4909–4926. A   
[Kraemer and Banerjee(2016)] Landon Kraemer and Bikramjit Banerjee. 2016. Multi-agent reinforcement learning as a rehearsal for decentralized planning. Neurocomputing 190 (2016), 82–94. 1, 4.1   
[Laskin et al.(2023)] Michael Laskin, Luyu Wang, Junhyuk Oh, Emilio Parisotto, Stephen Spencer, Richie Steigerwald, DJ Strouse, Steven Stenberg Hansen, Angelos Filos, Ethan Brooks, Maxime Gazeau, Himanshu Sahni, Satinder Singh, and Volodymyr Mnih. 2023. In-context reinforcement learning with algorithm distillation. In International Conference on Learning Representations. 1, 2   
[Lee et al.(2023)] Jonathan Lee, Annie Xie, Aldo Pacchiano, Yash Chandak, Chelsea Finn, Ofir Nachum, and Emma Brunskill. 2023. Supervised pretraining can learn in-context reinforcement learning. In Advances in Neural Information Processing Systems. 43057–43083. 1, 2, 4.4, B, B   
[Lee et al.(2022a)] Kuang-Huei Lee, Ofir Nachum, Mengjiao Yang, Lisa Lee, Daniel Freeman, Sergio Guadarrama, Ian Fischer, Winnie Xu, Eric Jang, Henryk Michalewski, and Igor Mordatch. 2022a. Multi-game decision transformers. In Advances in Neural Information Processing Systems. 27921–27936. 2   
[Lee et al.(2022b)] Seunghyun Lee, Younggyo Seo, Kimin Lee, Pieter Abbeel, and Jinwoo Shin. 2022b. Offline-toonline reinforcement learning via balanced replay and pessimistic q-ensemble. In Conference on Robot Learning. 1702–1712. A   
[Levine and Koltun(2013)] Sergey Levine and Vladlen Koltun. 2013. Guided policy search. In International Conference on Machine Learning. 1–9. A   
[Levine et al.(2020)] Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. 2020. Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. arXiv preprint arXiv:2005.01643 (2020). A   
[Li et al.(2021)] Lanqing Li, Rui Yang, and Dijun Luo. 2021. FOCAL: Efficient fully-offline meta-reinforcement learning via distance metric learning and behavior regularization. In International Conference on Learning Representations. A   
[Li et al.(2025)] Zhengyang Li, Qijin Ji, Xinghong Ling, and Quan Liu. 2025. A comprehensive review of multi-agent reinforcement learning in video games. Authorea Preprints (2025). 2   
[Lin et al.(2022)] Sen Lin, Jialin Wan, Tengyu Xu, Yingbin Liang, and Junshan Zhang. 2022. Model-based offline meta-reinforcement learning with regularization. In International Conference on Learning Representations. A   
[Liu and Abbeel(2023)] Hao Liu and Pieter Abbeel. 2023. Emergent agentic transformer from chain of hindsight experience. In International Conference on Machine Learning. 21362–21374. 2, 4.1, 5.1, D.1   
[Liu et al.(2025)] Sicong Liu, Yang Shu, Chenjuan Guo, and Bin Yang. 2025. Learning generalizable skills from offline multi-task data for multi-agent cooperation. In International Conference on Learning Representations. 5.1, D.1   
[Lockwood and Si(2022)] Owen Lockwood and Mei Si. 2022. A review of uncertainty for deep reinforcement learning. In Proceedings of the AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment. 155–162. 6   
[Lowe et al.(2017)] Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. 2017. Multi-agent actor-critic for mixed cooperative-competitive environments. In Advances in Neural Information Processing Systems. 6379–6390. 2, 4.1   
[Maaten and Hinton(2008)] Laurens van der Maaten and Geoffrey Hinton. 2008. Visualizing data using t-SNE. Journal of machine learning research 9, Nov (2008), 2579–2605. 5.3   
[Meng et al.(2023)] Linghui Meng, Muning Wen, Chenyang Le, Xiyun Li, Dengpeng Xing, Weinan Zhang, Ying Wen, Haifeng Zhang, Jun Wang, Yaodong Yang, and Bo Xu. 2023. Offline pre-trained multi-agent decision transformer. Machine Intelligence Research 20, 2 (2023), 233–248. 5.1, D.1   
[Moeini et al.(2025)] Amir Moeini, Jiuqi Wang, Jacob Beck, Ethan Blaser, Shimon Whiteson, Rohan Chandra, and Shangtong Zhang. 2025. A survey of in-context reinforcement learning. arXiv preprint arXiv:2502.07978 (2025). 1, 3.3, A   
[Nair et al.(2020)] Ashvin Nair, Abhishek Gupta, Murtaza Dalal, and Sergey Levine. 2020. Awac: Accelerating online reinforcement learning with offline datasets. arXiv preprint arXiv:2006.09359 (2020). A

[Nair et al.(2018)] Ashvin Nair, Bob McGrew, Marcin Andrychowicz, Wojciech Zaremba, and Pieter Abbeel. 2018. Overcoming exploration in reinforcement learning with demonstrations. In IEEE International Conference on Robotics and Automation. 6292–6299. A   
[Oliehoek and Amato(2016)] Frans A Oliehoek and Christopher Amato. 2016. A concise introduction to decentralized POMDPs. Springer. 1, 2, 3.1   
[Osband et al.(2013)] Ian Osband, Daniel Russo, and Benjamin Van Roy. 2013. (More) efficient reinforcement learning via posterior sampling. In Advances in Neural Information Processing Systems. 3003–3011. B   
[Papoudakis et al.(2021)] Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, and Stefano V Albrecht. 2021. Benchmarking multi-agent deep reinforcement learning algorithms in cooperative tasks. In NeurIPS Datasets and Benchmarks Track. 1, 5.1, C.1   
[Radford et al.(2019)] Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. OpenAI blog 1, 8 (2019), 9. 5.1   
[Rashid et al.(2020)] Tabish Rashid, Mikayel Samvelyan, Christian Schroeder De Witt, Gregory Farquhar, Jakob Foerster, and Shimon Whiteson. 2020. Monotonic value function factorisation for deep multi-agent reinforcement learning. Journal of Machine Learning Research 21, 178 (2020), 1–51. 2, 4.1, 5.1, C.1   
[Reed et al.(2022)] Scott E. Reed, Konrad Zolna, Emilio Parisotto, Sergio Gómez Colmenarejo, Alexander Novikov, Gabriel Barth-Maron, Mai Gimenez, Yury Sulsky, Jackie Kay, Jost Tobias Springenberg, Tom Eccles, Jake Bruce, Ali Razavi, Ashley Edwards, Nicolas Heess, Yutian Chen, Raia Hadsell, Oriol Vinyals, Mahyar Bordbar, and Nando de Freitas. 2022. A generalist agent. arXiv preprint arXiv:2205.06175 (2022). 2   
[Ross et al.(2011)] Stéphane Ross, Geoffrey Gordon, and Drew Bagnell. 2011. A reduction of imitation learning and structured prediction to no-regret online learning. In International Conference on Artificial Intelligence and Statistics. 627–635. 4.3, B   
[Samvelyan et al.(2019)] Mikayel Samvelyan, Tabish Rashid, Christian Schroeder de Witt, Gregory Farquhar, Nantas Nardelli, Tim GJ Rudner, Chia-Man Hung, Philip HS Torr, Jakob Foerster, and Shimon Whiteson. 2019. The StarCraft multi-agent challenge. In International Conference on Autonomous Agents and MultiAgent Systems. 2186–2188. 1, 5.1, C.2   
[Schmied et al.(2024)] Thomas Schmied, Fabian Paischer, Vihang Patil, Markus Hofmarcher, Razvan Pascanu, and Sepp Hochreiter. 2024. Retrieval-augmented decision transformer: External memory for in-context rl. arXiv preprint arXiv:2410.07071 (2024). 2, 4.4, 5.1, 5.3, D.1   
[Son et al.(2019)] Kyunghwan Son, Daewoo Kim, Wan Ju Kang, David Earl Hostallero, and Yung Yi. 2019. Qtran: Learning to factorize with transformation for cooperative multi-agent reinforcement learning. In International Conference on Machine Learning. 5887–5896. 2   
[Song et al.(2023)] Yuda Song, Yifei Zhou, Ayush Sekhari, Drew Bagnell, Akshay Krishnamurthy, and Wen Sun. 2023. Hybrid RL: Using both offline and online data can make RL efficient. In International Conference on Learning Representations. A   
[Sridhar et al.(2025)] Kaustubh Sridhar, Souradeep Dutta, Dinesh Jayaraman, and Insup Lee. 2025. REGENT: A retrieval-augmented generalist agent that can act in-context in new environments. In International Conference on Learning Representations. 2   
[Sunehag et al.(2018)] Peter Sunehag, Guy Lever, Audrunas Gruslys, Wojciech Marian Czarnecki, Vinícius Flores Zambaldi, Max Jaderberg, Marc Lanctot, Nicolas Sonnerat, Joel Z. Leibo, Karl Tuyls, and Thore Graepel. 2018. Value-decomposition networks for cooperative multi-agent learning based on team reward. In International Conference on Autonomous Agents and MultiAgent Systems. 2085–2087. 1, 2, 4.1   
[Todorov et al.(2012)] Emanuel Todorov, Tom Erez, and Yuval Tassa. 2012. MuJoCo: A physics engine for modelbased control. In International Conference on Intelligent Robots and Systems. 5026–5033. D.1   
[Vaswani et al.(2017)] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems. 5998–6008. 3.2   
[Wang et al.(2020)] Jianhao Wang, Zhizhou Ren, Terry Liu, Yang Yu, and Chongjie Zhang. 2020. QPLEX: Duplex dueling multi-agent q-learning. In International Conference on Learning Representations. 2   
[Wang et al.(2024)] Zhi Wang, Li Zhang, Wenhao Wu, Yuanheng Zhu, Dongbin Zhao, and Chunlin Chen. 2024. Meta-DT: Offline meta-RL as conditional sequence modeling with world model disentanglement. Advances in Neural Information Processing Systems (2024), 44845–44870. A

[Wolf et al.(2020)] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. "Transformers: State-of-the-art natural language processing. In Empirical Methods in Natural Language Processing. 38–45. D.2   
[Xu et al.(2022)] Mengdi Xu, Yikang Shen, Shun Zhang, Yuchen Lu, Ding Zhao, Joshua Tenenbaum, and Chuang Gan. 2022. Prompting decision transformer for few-shot policy generalization. In International Conference on Machine Learning. 24631–24645. 2   
[Yuan and Lu(2022)] Haoqi Yuan and Zongqing Lu. 2022. Robust task representations for offline meta-reinforcement learning via contrastive learning. In International Conference on Machine Learning. 25747–25759. A   
[Yuan et al.(2023)] Lei Yuan, Ziqian Zhang, Lihe Li, Cong Guan, and Yang Yu. 2023. A survey of progress on cooperative multi-agent reinforcement learning in open environment. arXiv preprint arXiv:2312.01058 (2023). 1, 2   
[Zhang et al.(2024)] Xinyu Zhang, Wenjie Qiu, Yi-Chen Li, Lei Yuan, Chengxing Jia, Zongzhang Zhang, and Yang Yu. 2024. Debiased offline representation learning for fast online adaptation in non-stationary dynamics. In International Conference on Machine Learning. 59741–59758. A   
[Zhou et al.(2024)] Renzhe Zhou, Chen-Xiao Gao, Zongzhang Zhang, and Yang Yu. 2024. Generalizable task representation learning for offline meta-reinforcement learning with data limitations. In Proceedings of the AAAI Conference on Artificial Intelligence. 17132–17140. A

# A Extended Related Work

Offline meta-RL. Offline Reinforcement Learning (RL) aims to learn policies from static datasets collected by a behavior policy, without further interaction with the environment [Levine et al.(2020)]. Offline Meta-RL (OMRL) extends this paradigm by training on a distribution of such offline tasks, thereby enabling generalization to novel tasks. Among existing approaches, context-based meta-learning methods employ a context encoder to perform approximate inference over task representations, conditioning the meta-policy on the inferred belief to improve generalization. For example, FOCAL [Li et al.(2021)] introduces a distance metric learning loss to distinguish between tasks, while subsequent works [Yuan and Lu(2022), Zhou et al.(2024), Zhang et al.(2024)] focus on accurately identifying environment dynamics and debiasing representations from the behavior policy. However, this paradigm can be unstable due to function approximation and bootstrapping, particularly in the absence of further online adaptation [Wang et al.(2024)]. In contrast, gradient-based OMRL methods adapt to unseen tasks with a few interactions [Lin et al.(2022)], but the required online gradient updates can be prohibitively expensive in real-world applications. Recently, in-context reinforcement learning (ICRL) [Moeini et al.(2025)] has emerged as an alternative, enabling rapid adaptation through trial-and-error without parameter updates. This paradigm demonstrates significant potential for practical applications.

Online RL with offline datasets. Online RL is often impractical in high-risk domains such as autonomous driving [Kiran et al.(2021)], due to the high cost and safety concerns associated with trial-and-error learning. In contrast, offline RL leverages large amounts of pre-collected data to safely update policies, but suffers from the distribution shift problem [Levine et al.(2020)]. To address the limitations of both approaches, online RL with offline datasets has emerged as a promising solution [Ball et al.(2023)]. One line of work pre-trains policies using offline RL and subsequently performs online fine-tuning [Hester et al.(2018), Lee et al.(2022b)]. However, the mismatch between offline data and online samples often results in inefficient adaptation during fine-tuning. Another line of research encourages the online agent to mimic behaviors found in the offline data [Levine and Koltun(2013), Nair et al.(2020)]. While this can stabilize policy updates, it still requires a large number of online samples to accomplish tasks due to the inherent limitations of the offline data. A third line of work initializes the replay buffer with offline data, which has been theoretically shown to provide strong guarantees and achieve excellent performance in both theory and practice [Nair et al.(2018), Song et al.(2023)]. Our method is similar to this third category: we design a retrieval mechanism based on exponential time decay over the replay buffer and offline data, enabling the agent to quickly adapt to deployment environments.

# B Detailed Theoretical Analysis

We first show that, under Assumption 1, $\pi _ { t } ^ { B }$ is equivalent to the online test setting in DPT [Lee et al.(2023)]. In DPT, the model predicts optimal actions during both training and testing by leveraging an in-context dataset sampled from the same task, together with the current context. When Assumption 1 holds, the in-context trajectories retrieved by MAICC’s embedding model can serve the same purpose as the in-context dataset in DPT. Therefore, under these conditions, the two approaches are equivalent. To facilitate the derivation of the online cumulative regret for MAICC, by we first present a lemma that bounds the total variation distance between the distributions of observations encountered $\hat { \pi } _ { t }$ and $\pi _ { t } ^ { B }$ [Ross et al.(2011)], where $\hat { \pi } _ { t } = \beta _ { t } \pi ^ { \mathcal { D } } + ( 1 - \beta _ { t } ) \pi _ { t } ^ { \mathcal { B } }$ .

Lemma 1. $| | d _ { \hat { \pi } _ { t } } - d _ { { \pi } _ { t } ^ { B } } | | \leq 2 H \beta _ { t }$ .

Proof. Let $d$ denote the distribution of observations over $H$ steps conditioned on $\hat { \pi } _ { t }$ selecting $\pi ^ { \mathcal { D } }$ at least once during these $H$ steps. Since $\hat { \pi } _ { t }$ executes $\pi _ { t } ^ { B }$ exclusively over $H$ steps with probability $\dot { ( 1 - \beta _ { t } ) } ^ { \tilde { H } }$ , we have $d _ { \hat { \pi } _ { t } } ~ = ~ ( 1 ~ \bar { - }$ $\beta _ { t } ) ^ { H } d _ { \pi _ { t } ^ { B } } + \bar { ( 1 - ( 1 - \beta _ { t } ) ^ { H } ) } d$ . Thus,

$$
\begin{array}{l} \left| \left| d _ {\hat {\pi} _ {t}} - d _ {\pi_ {t} ^ {R}} \right| \right| _ {1} = \left(1 - \left(1 - \beta_ {t}\right) ^ {H}\right) \left| \left| d - d _ {\pi_ {t} ^ {R}} \right| \right| _ {1} \\ \leq 2 (1 - (1 - \beta_ {t}) ^ {H}) \\ \leq 2 H \beta_ {t}, \tag {7} \\ \end{array}
$$

where the last inequality follows from the fact that $( 1 - \beta ) ^ { H } \geq 1 - \beta H$ for any $\beta \in [ 0 , 1 ]$ .

Next, if we set $\beta _ { t }$ as an exponential decay over time $t$ , we can obtain:

Theorem 2. Suppose $\operatorname* { s u p } _ { \mathcal { M } } P ( \mathcal { M } ) / P _ { \mathcal { D } } ( \mathcal { M } ) \leq C$ for some $C > 0$ , where $P _ { \mathcal { D } } ( \mathcal { M } )$ denotes the training task distribution. Then the expected online cumulative regret of MAICC satisfies $\mathbb { E } _ { P ( \mathcal { M } ) } [ \mathbf { R e g } _ { \mathcal { M } } ] \leq \tilde { \mathcal { O } } ( C H ^ { 3 / 2 } \omega \sqrt { A T } )$ .

Proof. For a given task $\mathcal { M } \in \mathop { P _ { D } } ( \cdot )$ , when the reward is bounded within [0, 1], the cumulative performance gap between $\hat { \pi } _ { t }$ and $\pi _ { t } ^ { B }$ is bounded as follows:

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} V ^ {\mathcal {M}} \left(\pi_ {t} ^ {\mathcal {B}}\right) - V ^ {\mathcal {M}} \left(\hat {\pi} _ {t}\right) \leq \sum_ {t = 1} ^ {T} \sum_ {h = 1} ^ {H} 2 h \beta_ {t} \\ \leq \sum_ {t = 1} ^ {T} 2 H ^ {2} \beta_ {t}. \\ \end{array}
$$

Although this appears to be $\mathcal { O } ( H ^ { 2 } )$ , if we choose $\beta _ { t } = \gamma ^ { t }$ with $\gamma \in [ 0 , 1 ]$ , then there exists an $n _ { \beta }$ , defined as the largest $n \leq T$ such that $\beta _ { n } > 1 / H$ . In this case, the bound can be further refined as follows:

$$
\begin{array}{l} \sum_ {t = 1} ^ {T} V ^ {\mathcal {M}} \left(\pi_ {t} ^ {\mathcal {B}}\right) - V ^ {\mathcal {M}} \left(\hat {\pi} _ {t}\right) \leq 2 H \sum_ {t = 1} ^ {T} \min  \left(1, H \beta_ {t}\right) \\ = 2 H (n _ {\beta} + H \sum_ {t = n _ {\beta} + 1} ^ {T} \beta_ {t}) \\ \leq 2 H \frac {\log H + 1}{1 - \gamma}, \\ \end{array}
$$

which becomes $\mathcal { O } ( H \log H )$ . In addition, $\beta _ { t } = \gamma ^ { t }$ is equivalent to $\begin{array} { r } { \beta _ { t } = \exp ( - \lambda \frac { t } { T } ) } \end{array}$ as used in the main paper, if we set $\lambda = - T \log \gamma$ . Following the results of previous work [Lee et al.(2023), Osband et al.(2013)], we can obtain that the cumulative regret between $\pi _ { t } ^ { B }$ and the expert policy $\pi ^ { * }$ is given by:

$$
\sum_ {t = 1} ^ {T} V ^ {\mathcal {M}} (\pi^ {*}) - V ^ {\mathcal {M}} (\pi_ {t} ^ {\mathcal {B}}) \leq \tilde {\mathcal {O}} (H ^ {3 / 2} \omega \sqrt {A T}),
$$

therefore the cumulative regret between $\hat { \pi } _ { t }$ and the expert policy $\pi ^ { * }$ is bounded as:

$$
\begin{array}{l} \mathbf {R e g} _ {\mathcal {M}} = \sum_ {t = 1} ^ {T} V ^ {\mathcal {M}} \left(\pi^ {*}\right) - V ^ {\mathcal {M}} \left(\hat {\pi} _ {t}\right) \\ \leq \tilde {\mathcal {O}} \left(H ^ {3 / 2} \omega \sqrt {A T}\right) + \mathcal {O} (H \log H) \\ = \tilde {\mathcal {O}} (H ^ {3 / 2} \omega \sqrt {A T}), \\ \end{array}
$$

when $T$ is huge enough. Since $\operatorname* { s u p } _ { \mathcal { M } } P ( \mathcal { M } ) / P _ { \mathcal { D } } ( \mathcal { M } ) \leq C$ for some $C > 0$ , then

$$
\begin{array}{l} \mathbb {E} _ {P (\mathcal {M})} [ \mathbf {R e g} _ {\mathcal {M}} ] = \int P (\mathcal {M}) \mathbf {R e g} _ {\mathcal {M}} d \mathcal {M} \\ \leq C \int P _ {\mathcal {D}} (\mathcal {M}) \mathbf {R e g} _ {\mathcal {M}} d \mathcal {M} \\ \le \tilde {\mathcal {O}} (C H ^ {3 / 2} \omega \sqrt {A T}). \\ \end{array}
$$

In practice, since the online replay buffer may lack sufficiently informative trajectories in the early stages, our constructed memory performs better, as demonstrated by the experimental results.

# C Extended Benchmark Descriptions

# C.1 Level-Based Foraging

Level-Based Foraging (LBF) [Papoudakis et al.(2021)] is a widely used cooperative grid world environment in which agents must coordinate to collect food simultaneously. To accommodate the setting of rapid decentralized cooperative adaptation, we introduce the following modifications. Taking LBF: $9 x 9 – 2 0 s$ in Fig. 5 as an example, three agents are initialized at fixed positions (as shown in the figure), each with a limited field of view covering only the adjacent grid cells (highlighted in blue). The task can only be successfully completed if all three agents simultaneously execute the

![](images/7cfcc1d86ad677db52b71b534491458c4ec3b6e05f40b3e543340d79559a02e2.jpg)  
Figure 5: Illustration of LBF: $9 x 9 – 2 0 s$ . The agents are required to cooperate within a limited number of time steps to concurrently collect the food based on their local observations. The blue areas indicate the agents’ local fields of view, the yellow areas represent possible spawn locations for the food (each corresponding to a specific task), and the red apples denote the food positions included in the training tasks.

![](images/f0952a405aedee5210ae227bf44d3b7be03027c8d2ee1f18493e0cc22a6959f8.jpg)  
(a) Protoss

![](images/2ac7bf7c6330180e384f349bdc5bf39e45d6b8924c9d1f00f46aea3d308345c2.jpg)  
(b) Terran

![](images/74c1385f798d8d039e857445bba7cfc28289f1d07cedf6d2154066b209b9a853.jpg)  
(c) Zerg   
Figure 6: Illustration of SMAC. Agents are divided into three races: (a) Protoss, (b) Terran, and (c) Zerg. In each scenario, a random number of agents with randomly selected unit types from the chosen race fight against an equal number of enemy units controlled by the built-in AI.

foraging action around the sole food item on the map. Each agent’s discrete action space consists of six actions: staying still, moving in four directions, and foraging. The team receives a reward of 0.33 each time an agent first reaches the food, and an additional reward of 1 upon successful foraging. An episode terminates either when the agents complete the task or when the time step limit (20 steps in this scenario) is reached. In the figure, yellow areas indicate the possible spawn locations of the food, while red apples denote the nine fixed tasks used during training. We employ the QMIX [Rashid et al.(2020)] algorithm, implemented using the EPyMARL codebase, to collect 200 trajectories for each of the nine tasks at $0 \%$ , $2 5 \%$ , $50 \%$ , $7 5 \%$ , and $100 \%$ of the maximum return, resulting in a multi-task dataset comprising 9,000 trajectories in total. For the alternative scenario, LBF: $7 x 7 – I 5 s$ , we reduce the grid world size to 7x7 and correspondingly shorten the maximum time steps to 15.

<table><tr><td>Race</td><td colspan="2">Basic Unit Types</td><td>Exceptional Unit Type</td></tr><tr><td>Protoss</td><td>Stalker</td><td>Zealot</td><td>Colossus</td></tr><tr><td>Terran</td><td>Marine</td><td>Marauder</td><td>Medivac</td></tr><tr><td>Zerg</td><td>Zergling</td><td>Hydralisk</td><td>Baneling</td></tr></table>

Table 2: The unit types of different races in SMAC.

# C.2 StarCraft Multi-Agent Challenge

StarCraft Multi-Agent Challenge (SMAC) is built upon the popular real-time strategy game StarCraft II and comprises a diverse set of cooperative multi-agent tasks. As illustrated in Fig. 6, the game features three distinct races—Protoss, Terran, and Zerg—each characterized by unique attributes. Every race consists of two basic unit types and one exceptional unit type, each fulfilling different roles and capable of executing various tasks. Notably, the exceptional unit type can exert a substantial influence on the outcome of a battle, as listed in Tab. 2. In each scenario, allied agents engage in combat against enemy agents controlled by the built-in AI. Each agent may choose from a discrete set of actions, including remaining stationary, moving in one of four directions, or attacking an enemy within its field of view. Team rewards are assigned based on the progression of the battle, with a complete victory yielding a reward of 20.

Based on the above settings, we designed two types of experiments. The first is SMACv1 [Samvelyan et al.(2019)], which exhibits relatively low randomness and comprises three scenarios: protoss, terran, and zerg. In each scenario, tasks differ in both the number and types of agents (restricted to basic unit types), with fixed spawn positions for all agents. For each scenario, we consider three agent configurations: 5v5, 7v7, and 10v11. Using the QMIX algorithm, we collected 500 trajectories for each configuration at both $50 \%$ and $100 \%$ of the maximum return, resulting in a total of 3,000 trajectories per scenario. During testing, the number of allied agents is randomly set between 3 and 12, with the number of enemy agents adjusted accordingly.

In addition, under the more complex SMACv2 [Ellis et al.(2023)] setting, we designed the most challenging scenario, denoted as all. In this scenario, the differences between tasks are not only reflected in the number and types of agents (including both basic and exceptional unit types), but also in their spawn positions, which vary across tasks. This requires agent teams to thoroughly explore in order to understand the current task. Furthermore, within a single task, any of the three aforementioned races may be present, posing an additional challenge for general decision-making models. In this scenario, the multi-task offline dataset contains 9,000 trajectories, collected in a manner similar to that used for the three scenarios described above.

# D Implementation Details

# D.1 Baselines

Here, we introduce the baselines used in our experiments, including the Multi-Agent Decision Transformer (MADT), ICRL methods, a multi-task MARL method, and an ablated version of MAICC.

MADT [Meng et al.(2023)] extends Decision Transformer (DT) [Chen et al.(2021)] to multi-agent systems by performing offline training through sequential modeling. Although it introduces a transformer-based model into MARL, it does not consider collaborative adaptation to unseen tasks. While the method proposes optional online fine-tuning, the expensive and time-consuming gradient updates hinder its practical application. Therefore, in our experimental results, the performance curve of MADT appears as a horizontal line, since its performance does not improve with an increasing number of episodes. By comparing with MADT, we demonstrate the advantage of ICRL-based methods in incorporating cross-trajectory information into decision-making. In contrast, MADT can only make decisions based on the contextual information of the current episode, which prevents the model from understanding the environment and thus hinders multi-task generalization.

AT [Liu and Abbeel(2023)] feeds cross-trajectory contexts into transformer-based models as prompts, enabling agents to adapt more quickly during the testing phase. During training, multiple historical trajectories are sorted in ascending order of return to form an implicit chain-of-experience, aiming to leverage hindsight for performance improvement. This trial-and-error update method, which does not require gradient updates, has demonstrated impressive results on MuJoCo tasks [Todorov et al.(2012)]. However, this approach does not perform retrieval over in-context trajectories;

in complex scenarios, simply selecting the most recent trajectories with higher returns is insufficient for agents to understand the characteristics of the environment, thereby hindering rapid adaptation.

RADT [Schmied et al.(2024)] is a recent ICRL work that introduces retrieval augmentation for the first time. It first utilizes a pre-trained DT model as an embedding model, and then performs trajectory retrieval based on the embeddings of action tokens. The retrieved similar trajectories are combined with the current input trajectory via cross-attention to assist decision-making, achieving superior adaptation in single-agent tasks compared to previous methods. However, in complex multi-agent scenarios, its coarse-grained encoder design faces challenges, leading to the retrieval of irrelevant trajectories. In addition, it does not include specific module designs to address the challenges of decentralized execution, which causes it to struggle in complex environments.

HiSSD [Liu et al.(2025)] is a recent work that achieves generalizable offline multi-task MARL through learning both common and task-specific skills. Although it performs well on multi-agent task suites with small task discrepancies, it lacks an online adaptation module. As a result, when the differences between tasks are large, agents are unable to adapt to new tasks using the learned skills, which leads to a collapse in cooperation.

MAICC-S is an ablated version of our method, with the only difference being that it does not utilize the centralized training property. Instead, it optimizes the DEM using only the following loss function:

$$
\mathcal {L} _ {\mathrm {D E M}} = \mathcal {L} _ {\mu} + \mathcal {L} _ {R} + \mathcal {L} _ {\mathcal {T}}, \tag {8}
$$

$$
\mathcal {L} _ {\mu} = - \mathbb {E} _ {\tau \sim \mathcal {D}} \sum_ {h = 0} ^ {H - 1} \sum_ {j = 1} ^ {n} \log \mathrm {M L P} _ {o \rightarrow a} \left(a _ {j} ^ {h} \mid z _ {o, j} ^ {h}\right), \tag {9}
$$

$$
\mathcal {L} _ {R} = \mathbb {E} _ {\tau \sim \mathcal {D}} \sum_ {h = 0} ^ {H - 1} \sum_ {j = 1} ^ {n} \left(r ^ {h} - \mathrm {M L P} _ {a \rightarrow r} \left(z _ {a, j} ^ {h}\right)\right) ^ {2}, \tag {10}
$$

$$
\mathcal {L} _ {\mathcal {T}} = - \mathbb {E} _ {\tau \sim \mathcal {D}} \sum_ {h = 0} ^ {H - 2} \sum_ {j = 1} ^ {n} \log \mathrm {M L P} _ {p \rightarrow o} \left(o _ {j} ^ {h + 1} \mid z _ {p} ^ {h}, o _ {j} ^ {h}\right). \tag {11}
$$

Here, the agent performs autoregressive prediction solely based on the embeddings generated from local information, and Eq. 10 cannot provide credit assignment to support subsequent decentralized execution. These factors together result in its collaborative adaptation ability on unseen tasks being inferior to that of MAICC, especially in SMAC scenarios.

# D.2 Model Architecture

Both the embedding models and the decision model in MAICC are implemented based on the GPT-2 model from the transformer codebase [Wolf et al.(2020)]. For CEM, we introduce intra-team visibility, which is achieved by modifying the causal mask in the model. This allows each agent’s observation and action tokens at the same timestep to attend to information from teammates, enabling team-level trajectory modeling. In addition, for all baselines and ablation studies in our experiments (except for HiSSD, as it is not transformer-based), we use models of the same scale to ensure the fairness of the experimental results.

# D.3 Hyper-Parameter Settings

In this paper, multiple hyperparameters are involved. Here, we list the key hyperparameter settings used in our experiments, as shown in Tab. 3.

# D.4 Computing Infrastructure

Most experiments were conducted on a server outfitted with an AMD EPYC 9654 96-Core Processor CPU, a NVIDIA GeForce RTX 4090 GPU, and 125 GB of RAM, running Ubuntu 20.04. MAICC was trained in a Python environment, and all relevant software libraries, along with their names and versions, are specified in the requirements.txt file included with the code.

Table 3: The key hyper-parameters in MAICC. The values before the / correspond to the LBF tasks, while the values after the / correspond to the SMAC tasks.   

<table><tr><td>Attribute</td><td>Value</td></tr><tr><td>Embedding size for GPT-2</td><td>64</td></tr><tr><td>Number of Layers for GPT-2</td><td>8</td></tr><tr><td>Number of attention heads for GPT-2</td><td>8</td></tr><tr><td>The dropout rate for GPT-2</td><td>0.1</td></tr><tr><td>Hidden layers of the MLPs</td><td>[64, 64]</td></tr><tr><td>Learning rate</td><td>5e-4</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Number of in-context trajectories k</td><td>3/2</td></tr><tr><td>Number of online adaptation episodes T</td><td>200/100</td></tr><tr><td>Coefficient α</td><td>0.8</td></tr><tr><td>Exponential decay rate λ</td><td>-log 0.01</td></tr><tr><td>Optimizer</td><td>Adam</td></tr></table>

![](images/c9b388be04312e05a9e488abd6578bf1d3c245d3f142031b7dc75b7bd69d5a37.jpg)  
(a)

![](images/27d86c16096c1e2fbd7cb83f28cda2ae2eeb96d8c47dbf02e6c832f7127ddb8e.jpg)  
(b)   
Figure 7: Visualization results illustrating the effects of different embedding model training settings. Each point in the figure represents the embedding of a trajectory from the dataset, with points of the same color corresponding to trajectories from the same task.

# E Additional Experiment Results

# E.1 Visualization of Learned Embeddings

Here, similar to Sec. 5.3 in the main paper, we further conduct a visualization analysis of the learned trajectory embeddings under different settings, as shown in Fig. 7. In (a), we show the results when the team information distilled by CEM is not used during training. It can be observed that the blue trajectories form two distinct clusters, which will inevitably affect the effectiveness of trajectory retrieval. In (b), for the training results using only ${ \mathcal { L } } _ { \mu }$ and $\mathcal { L } _ { T }$ , we also observe an overfitting phenomenon in the trajectory representations. These results, together with the experimental results in the main paper, demonstrate the effectiveness of our embedding model design.

# E.2 Ablation Study

Similar to Sec. 5.4 in the main paper, we conducted the same ablation study on LBF: 9x9-20s. The results are presented in Tab. 4. It can be observed that, in the grid-world environment, our default setting still achieves the best performance, demonstrating the effectiveness of each module. Among them, the design of embedding models has a relatively minor impact on the results, whereas the constructed memory plays a crucial role in experimental performance. Relying solely on either the offline dataset or the online buffer for retrieval leads to severe adaptation failure.

Table 4: Ablation Study on MAICC. Unless otherwise noted, all settings follow the default configuration. “Ret.” indicates the average return over 50 test runs (with $9 5 \%$ confidence interval), evaluated in the final adaptation episode.   

<table><tr><td>Variants</td><td>EM With RTG</td><td>Coefficient β</td><td>CEM loss</td><td>Hyper-parameter α</td><td>LBF: 9x9-20s Ret.</td></tr><tr><td>Default</td><td>False</td><td>βt = exp(-λt/T)</td><td>Lμ + LR + LT</td><td>α = 0.8</td><td>1.71±0.08</td></tr><tr><td>(A)</td><td>True</td><td></td><td></td><td></td><td>1.69±0.09</td></tr><tr><td rowspan="2">(B)</td><td></td><td>βt = 0</td><td></td><td></td><td>0.68±0.07</td></tr><tr><td></td><td>βt = 1</td><td></td><td></td><td>0.84±0.09</td></tr><tr><td rowspan="3">(C)</td><td></td><td></td><td>Lμ + LRT</td><td></td><td>1.58±0.11</td></tr><tr><td></td><td></td><td>Lμ + LT</td><td>α = 1</td><td>1.60±0.12</td></tr><tr><td></td><td></td><td>Lμ</td><td>α = 1</td><td>1.44±0.09</td></tr><tr><td rowspan="2">(D)</td><td></td><td></td><td></td><td>α = 1</td><td>1.64±0.08</td></tr><tr><td></td><td></td><td></td><td>α = 0</td><td>1.66±0.07</td></tr></table>

![](images/a6de175637605e1ecc757382c0cc9178dd73cec29f4a6d0b05bf6181edbc6688.jpg)  
(a) Sensitivity of ?

![](images/060fb247d03efe0b5304155f6193eac6495e7e2bf04fabfa91a5444fff1b1e81.jpg)  
(b) Sensitivity of ?   
Figure 8: Sensitivity of hyper-parameters. The red bars indicate the parameter values selected in our main experiments.

# E.3 Sensitivity of Hyper-Parameters

In MAICC, the selection of certain hyper-parameters affects the experimental results. Therefore, in this subsection, we conduct a sensitivity analysis on two key hyper-parameters in the LBF: $9 x 9 – 2 0 s$ scenario. As shown in Fig. 8(a), we first conduct a sensitivity analysis on the number of in-context trajectories $k$ . A larger $k$ allows the context to provide more information, enabling the agent to better understand the task requirements. However, since the inference speed of the transformer grows quadratically with the context length, we ultimately set $k = 3$ for LBF environments and $k = 2$ for SMAC tasks to balance performance and inference speed. In addition, Fig. 8(b) presents our study on the exponential decay rate $\lambda$ . We observe that if the decay is too fast or too slow, the quality of trajectory retrieval deteriorates, which in turn reduces the final adaptation performance. Therefore, we set $\lambda = - \log 0 . 0 1$ in all experiments.
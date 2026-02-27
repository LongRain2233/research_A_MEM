# SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience

Zeyi Sun1,2 Ziyu Liu1,2 Yuhang Zang2 Yuhang Cao2 Xiaoyi Dong2,3 Tong Wu3 Dahua Lin2,3 Jiaqi Wang2

1Shanghai Jiao Tong University 2Shanghai Artificial Intelligence Laboratory 3The Chinese University of Hong Kong szy2023@sjtu.edu.cn, wangjiaqi@pjlab.org.cn https://github.com/SunzeY/SEAgent

# Abstract

Repurposing large vision-language models (LVLMs) as computer use agents (CUAs) has led to substantial breakthroughs, primarily driven by human-labeled data. However, these models often struggle with novel and specialized software, particularly in scenarios lacking human annotations. To address this challenge, we propose SEAgent, an agentic self-evolving framework enabling CUAs to autonomously evolve through interactions with unfamiliar software. Specifically, SEAgent empowers computer-use agents to autonomously master novel software environments via experiential learning, where agents explore new software, learn through iterative trial-and-error, and progressively tackle auto-generated tasks organized from simple to complex. To achieve this goal, we design a World State Model for step-wise trajectory assessment, along with a Curriculum Generator that generates increasingly diverse and challenging tasks. The agent’s policy is updated through experiential learning, comprised of adversarial imitation of failure actions and Group Relative Policy Optimization (GRPO) on successful ones. Furthermore, we introduce a specialist-to-generalist training strategy that integrates individual experiential insights from specialist agents, facilitating the development of a stronger generalist CUA capable of continuous autonomous evolution. This unified agent ultimately achieves performance surpassing ensembles of individual specialist agents on their specialized software. We validate the effectiveness of SEAgent across five novel software environments within OS-World. Our approach achieves a significant improvement of $2 3 . 2 \%$ in success rate, from $1 1 . 3 \%$ to $3 4 . 5 \%$ , over a competitive open-source CUA, i.e., UI-TARS.

# 1 Introduction

“A new generation of agents will acquire superhuman capabilities by learning predominantly from experience.” [55]

— David Silver, Richard S. Sutton

With the rapid development of large vision-language models (LVLMs) [61, 16, 7, 64, 42, 5, 60], computer use agents (CUAs) [3, 43, 48, 29, 67] have not only emerged but also demonstrated increasing practical utility. By leveraging the powerful perception and reasoning capabilities of LVLMs, these agents can interpret screenshots as visual inputs and operate computers via keyboard and mouse actions. Despite their promising capabilities, current CUAs [47, 46, 12, 19, 6, 34] primarily depend on costly human-curated datasets [12, 9, 67, 24, 28], which are typically derived

![](images/20fa174f9d7eff5fc4404c0cb29b4b66c63ddf190137bcb7a05b262ccec0957d.jpg)  
Figure 1: SEAgent enables computer use agents self-evolving in novel environments by autonomously exploring and learning from their own experiences without human intervention. The specialist-to-generalist training strategy further enhances the development of a strong generalist agent.

from demonstrations [34, 78, 18, 51, 75] or video tutorials in the wild [70]. However, new software continuously emerges and existing software may regularly be updated, often in the absence of annotated human data. It is both necessary and timely to enter an era that emphasizes learning from experience [55] in CUA domain. In this paper, we aim to enable CUAs to autonomously explore unfamiliar software environments and evolve into experts without relying on human supervision.

To address this challenge, we propose SEAgent, an agentic self-evolving framework in which Computer Use Agents (CUAs) are exposed to previously unfamiliar software environments and engage in autonomous exploration and experiential learning, as illustrated in Fig. 1. Enabling such selfevolution requires addressing two key challenges: (1) generating executable tasks within unfamiliar software environments, and (2) accurately assessing task success and precisely identifying the step at which failure occurs. To this end, we introduce a World State Model for environmental state captioning and step-wise trajectory assessment, together with a Curriculum Generator powered by a continuously updated software guidebook memory to generate increasingly diverse and challenging tasks, thereby establishing a curriculum learning paradigm. The agent’s policy is optimized through experiential learning from both failures and successes, combining adversarial imitation of failure actions and Group Relative Policy Optimization (GRPO) on successful ones.

Given the critical role of reward accuracy, we conduct extensive evaluations and observe that existing reward models of computer use tasks fall short in terms of judgment precision and reward density. Leveraging the enhanced long-context processing capabilities of advanced LVLMs, we input the agent’s full trajectory of states into the reward model and fine-tune a reward model, World State Model, using Qwen2.5-VL [7], substantially narrowing the gap with commercial models such as GPT-4o [42] with $+ 7 . 5 \%$ improvement in precision compared to baseline model in evaluating CUAs’ trajectories on AgentRewardBench [35], enable World State Model to provide high quality step level reward signals in self-evolving agentic system.

Moreover, SEAgent enables agents to evolve into either single-software specialists or multi-software generalists. To overcome the limitation that directly training a generalist underperforms compared to specialists, inspired by [77], we introduce a novel specialist-to-generalist training strategy, which even surpasses the performance of individual specialists on their respective software applications.

We perform extensive experiments of SEAgent built on UI-TARS [48] and evaluated on five professional software applications from OSWorld [68]. SEAgent with the specialist-to-generalist strategy significantly improves the UI-TARS [48] from $1 1 . 3 \%$ to $3 4 . 5 \%$ . Furthermore, SEAgent with the

specialist-to-generalist strategy $( 3 4 . 5 \% )$ outperforms both specialist RL $( 3 2 . 2 \% )$ and generalist RL $( 3 0 . 6 \% )$ by a substantial margin, demonstrating the effectiveness of the specialist-to-generalist paradigm. In general, SEAgent offers a promising approach for developing more powerful and versatile computer-use agents without human involvement.

# 2 Related Work

Agent for Computer Use. With the recent advances in LLMs and LVLMs [61, 16, 30, 7, 64], which enable human-level perception and reasoning capabilities, the development of agents for computer use has garnered significant attention [22, 20, 11, 41, 29]. These agents either rely solely on structured text inputs [47, 40, 46, 26, 36] or, in a more human-like manner, use multi-modal inputs such as screenshots combined with textual conditions [21, 29, 67, 43]. Although they have been extensively studied and show strong performance on in-domain benchmarks [34, 79, 31, 27, 11], computer use agents still lag significantly behind human-level performance in simulated environments [68, 50, 25, 80]. This gap highlights the challenges posed by the multi-dimensional demands on LVLMs, including grounding, decision-making, and reasoning. Some approaches address this by decomposing tasks into specialized expert models [15, 62] and employing agent collaboration [1, 2, 32, 74] through prompt engineering [71, 19, 76, 63, 66]. However, improvements from these training-free methods remain limited without fine-tuning. In this work, we explore the next phase of computer use agents, where a pretrained agent is fine-tuned to learn from its own experience, enabling self-evolution on novel, specialized software without human annotations.

Reinforcement Learning for LLMs/LVLMs. Previous scalable training efforts for LLMs and LVLMs [61, 16, 30, 7, 64, 69, 59, 58, 13] have primarily relied on supervised fine-tuning (SFT) [30, 65]. Analogous to imitation learning or behavior cloning in reinforcement learning (RL), SFT trains models to produce desired outputs based on labeled data, making it heavily dependent on high-quality human-curated procedures. Recently, DeepSeek-R1 [17] demonstrated strong reasoning capabilities via Group Relative Policy Optimization (GRPO) [53] using verifiable rewards. Earlier works [44, 82, 49] have also employed RL for single-turn optimization from human feedback. However, in agentic scenarios such as computer usage—where feedback is sparse with reward signals often results from multi-step interactions—it becomes crucial to introduce stable, step-level reward signals. Prior RL approaches for agents [6, 47, 81, 73, 8] have fine-tuned their own critic models for advantage estimation [52], either using output reward models (ORMs) trained on labeled data or adopting Direct Preference Optimization (DPO) [49] based on interaction data [46, 48]. In this work, we investigate various strategies for constructing high-performing reward models for CUAs and find that full-process-based analysis yields more accurate evaluations with fine-grained reward signals compared to training dedicated critic models for advantage estimation as done in [6, 47] or with filtered behavior cloning [45, 10].

# 3 Methods

Problem Formulation. The objective of SEAgent is to establish a training pipeline enabling the Computer Use Agent (CUA) to autonomously explore its environment (Sec. 3.1) and progressively self-evolve on novel software applications via reinforcement learning from experience (Sec. 3.2). Specifically, the SEAgent pipeline comprises three primary components: an Actor Model $\pi$ performing exploratory actions to accomplish these tasks, and a World State Model $\mathcal { M } _ { s t a t e }$ describing the current environment state and evaluating the success or failure of executed actions, and a Curriculum Generator $\mathcal { M } _ { t a s k }$ that continuously proposes more diverse and challenging exploration tasks:

(1) Actor Model π: The policy $\pi ( \boldsymbol { a } | \boldsymbol { s } _ { t } , I )$ defines the probability of taking action $a$ at time step $t$ conditioned on the current state $s _ { t }$ and the overall task instruction $I$ .   
(2) World State Model $\mathcal { M } _ { s t a t e }$ : This component is a fine-tuned Large Vision-Language Model (LVLM) responsible for providing detailed descriptions of environment states. It also evaluates each step of the trajectory executed by the Actor Model $\pi$ , producing trajectory judgement $\mathcal { I }$ which indicates whether the task has been successfully completed. Joint training with state change captioning $\mathcal { C }$ of the software GUI has been shown to enhance judgment accuracy, as shown in Table 1.

![](images/ca3e06af63a7c39ebb327496569f43c6806753327488794b8efe94c5ee10510a.jpg)

![](images/ae9c8805d7d2bbcda80582079971886ed5cfbc6149e078b04e7cdea4f20282ca.jpg)  
Figure 2: SEAgent autonomous exploration and experiential learning pipeline. Guided by tasks generated by the Curriculum Generator, the Actor Model is updated according to step-level rewards from the World State Model through verifiable reward functions tailored for different action types.

(3) Curriculum Generator $\mathcal { M } _ { t a s k }$ : This component utilizes a powerful Large Language Model (LLM) to automatically generate novel exploration tasks. It also maintains and updates a software guidebook $U$ based on the state change captioning $\mathcal { C }$ and the trajectory judgement $\mathcal { I }$ provided by $\mathcal { M } _ { s t a t e }$ during interactions. The gradually enriched guidebook $U$ enables $\mathcal { M } _ { t a s k }$ to progressively generate increasingly diverse and challenging tasks in a curriculum learning fashion.

SEAgent can be applied to enable the self-evolution of a computer-use agent, either as a specialist for a single software or as a generalist across multiple software. However, we observe that direct training for generalist agents is suboptimal. We introduce a specialist-to-generalist training strategy, which achieves even better overall performance than training multiple generalist agents, as discussed in Sec. 3.3.

# 3.1 Autonomous Exploration with Self-evolving Curriculum

Autonomous exploration is essential for enabling the Computer Use Agent (CUA) to develop proficiency in novel software applications that are previously unseen or poorly understood. This process involves addressing two key challenges: (1) generating executable tasks within unfamiliar software environments, and (2) evaluating task completion success and pinpointing the specific step at which failure occurs. To tackle these challenges, we introduce two novel components: the World State Model $\mathcal { M } _ { \mathrm { s t a t e } }$ and the Curriculum Generator $\mathcal { M } _ { \mathrm { t a s k } }$ . These components jointly support a self-evolving curriculum paradigm, which facilitates the autonomous generation of increasingly diverse and challenging tasks.

The self-evolving curriculum paradigm pipeline is structured into $P$ sequential phases. Before the first phase, a set of initial tasks targeting basic GUI operations is generated (details provided in Sup. C.1). In each phase, these tasks are executed and step-wise evaluated. The resulting judgments and descriptions of the exploration trajectories are fed into the Curriculum Generator $\mathcal { M } _ { \mathrm { t a s k } }$ , which updates a self-maintained software guidebook $U$ . Leveraging this updated guidebook and the current capabilities of the CUA, the generator then produces more diverse and challenging tasks for subsequent phases. The following outlines each step of the process in detail:

(1) Task initiation: The initial state of the unfamiliar software is provided, typically in the form of screenshots of its basic GUI interface. The World State Model $\mathcal { M } _ { \mathrm { s t a t e } }$ performs dense captioning of the GUI elements, including button detection and OCR-based recognition. These detailed captions are passed to the Curriculum Generator $\mathcal { M } _ { \mathrm { t a s k } }$ , which generates an initial set of task instructions $\mathcal { T } _ { 0 } = \{ I _ { 0 } ^ { ( 1 ) } , I _ { 0 } ^ { ( 2 ) } , \cdot \cdot \cdot \}$ along with an initial software guidebook $U _ { 0 }$ for the software.   
(2) World state judgment: In the $p$ -th phase of Auto Exploration, the Actor Model $\pi _ { p }$ executes tasks based on the instructions in $\mathcal { T } _ { p }$ . Each execution is evaluated by the World State Model $\mathcal { M } _ { \mathrm { s t a t e } }$ , which provides feedback generates a detaile $\mathcal { T } _ { p } = \{ J _ { p } ^ { ( 1 ) } , J _ { p } ^ { ( 2 ) } , \cdot \cdot \cdot \}$ for each step within the operation trajectory. In additiotate changes based on captured screenshots, denoted as $\mathcal { C } _ { p }$   
(3) Task self-evolving: Based on the outcomes $\mathcal { I } _ { p }$ and $\zeta _ { p }$ , the Curriculum Generator $\mathcal { M } _ { \mathrm { t a s k } }$ produces a more challenging task set ${ \mathcal { T } } _ { p + 1 }$ and expands the agent’s knowledge boundary by updating the software guidebook to $U _ { p + 1 }$ . The detailed prompting process is illustrated in Fig. 8. This iterative

update can be formalized as:

$$
U _ {p + 1}, \mathcal {I} _ {p + 1} = \mathcal {M} _ {\text {t a s k}} \left(U _ {p}, \mathcal {I} _ {p}, \mathcal {J} _ {p}, \mathcal {C} _ {p}\right) \tag {1}
$$

Here, $U _ { p + 1 }$ serves as a more comprehensive software guidebook memory, while ${ \mathcal { T } } _ { p + 1 }$ represents a more challenging task set tailored to the current capabilities of the Actor Model $\pi _ { p }$ . Examples of $\mathcal { T } _ { p }$ are provided in Fig. 4, where the Actor Model $\pi$ demonstrates curriculum learning by handling increasingly complex tasks across different phases $p$ . Illustrations of $U _ { p }$ across various software applications are provided in Sup. J. Comparison with previous methods [39, 38, 56] on task generation are detailed in Sup.C.2

(4) Autonomous RL Training: Through iterative reinforcement learning, the Actor Model $\pi _ { p }$ is updated based on its execution of the instruction set $\mathcal { T } _ { p }$ , guided by evaluation feedback $\mathcal { I } _ { p }$ and a set of action-specific verifiable functions $\mathcal { R } _ { \mathrm { v e r i f e r } }$ . The resulting policy $\pi _ { p + 1 }$ is then used as the actor in the subsequent phase. Further details are provided in Sec. 3.2.

# 3.2 Reinforcement Learning from Experience

The World State Model $\mathcal { M } _ { s t a t e }$ provides step-level reward signals for reinforcement learning. Unlike previous reward models for CUA [47, 6, 46, 45, 35], our $\mathcal { M } _ { s t a t e }$ model takes the entire trajectory of states and actions, ${ \mathcal { H } } = \left\{ { \big ( } s _ { 0 } , a _ { 0 } { \big ) } , { \big ( } s _ { 1 } , a _ { 1 } { \big ) } , \ldots \right\}$ , as input. It classifies each action $a$ as either $a _ { F }$ or $a _ { T }$ , where $a _ { F }$ indicates an incorrect action leading to failure or redundant loops, and $a _ { T }$ represents a correct action that contributes to successful progression without redundancy. The curated prompt used for judgment is depicted in Fig. 7. For historical states that result in $a _ { T }$ , we encourage CUA to reinforce these actions through verifiable rewards defined by a set of functions $\mathcal { R } _ { \mathrm { v e r i f e r } } = \{ r _ { d i s t } \}$ . Conversely, for states leading to $a _ { F }$ , we penalize them using negative KL divergence with adversarial imitation.

Adversarial Imitation for Failure Action Punishment. To explicitly encourage the policy to diverge from failure-inducing behaviors, we employ a contrastive log-ratio loss based on a reference failure action $a _ { F }$ . This objective encourages the policy to sample actions $a$ that minimize alignment with the failure action $a _ { F }$ :

$$
\mathcal {L} _ {\mathrm {A I}} (\pi_ {\theta}) = \mathbb {E} _ {\nu} \left[ - \log \frac {\pi_ {\theta} (a \mid s , I)}{\pi_ {\mathrm {r e f}} \left(a _ {F} \mid s , I\right)} \right] \tag {2}
$$

This formulation serves as an adversarial imitation signal. By maximizing divergence from this distribution, the agent is trained to explore alternative action distributions that deviate from those leading to failure, particularly in complex GUI interaction scenarios. Notably, this loss shares a similar form with DPO [49] but only the negative part.

Verifiable Rewards for Correct Action Encouragement. To more effectively guide the policy toward correct actions $a _ { T }$ , we adopt Reinforcement Learning with Verifiable Rewards (RLVR) [17, 53], which has recently shown success in enhancing language models on tasks with objectively verifiable answers, such as mathematics [53], and more recently, counting and grounding in the vision-language domain [33, 54, 37]. After labeling the correct step $( s , a _ { T } )$ using the World State Model, we apply Group Relative Policy Optimization (GRPO), computing the relative advantage of each response based on its reward:

$$
A ^ {(i)} = \frac {r ^ {(i)} - \operatorname {m e a n} \left(\left\{r ^ {(j)} \right\} _ {j = 1} ^ {G}\right)}{\operatorname {s t d} \left(\left\{r ^ {(j)} \right\} _ {j = 1} ^ {G}\right)}, \quad i = 1, \dots , G. \tag {3}
$$

As we design distinct reward signals for different action types, we define the reward function between a predicted action $a$ and the ground-truth action $a _ { T }$ as:

$$
r ^ {(i)} = r \left(a ^ {(i)}, a _ {T}\right) = \mathbb {I} \left(\operatorname {t y p e} \left(a ^ {(i)}\right) = \operatorname {t y p e} \left(a _ {T}\right)\right) + r _ {\text {d i s t}} \left(a ^ {(i)}, a _ {T}\right), \tag {4}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function that returns 1 if the predicted action and ground-truth action are of the same type, and 0 otherwise. The distance-based reward term $r _ { \mathrm { d i s t } } ( a ^ { ( i ) } , a _ { T } )$ is defined according to the specific action type: for click actions, it is computed based on the normalized L1 distance between the clicked coordinates; for drag and select actions, it is computed using the Intersection over Union (IoU) between the predicted and ground-truth bounding boxes; and for type actions, it

is determined by the character-level BLEU score between the predicted and ground-truth text. All $r _ { \mathrm { d i s t } }$ rewards are normalized to the range [0, 1] to ensure consistency across different action types. A comprehensive list of $r _ { \mathrm { d i s t } } ( a ^ { ( i ) } , a _ { T } )$ definitions for various action types is provided in Tab. 8. The final loss of GRPO is directly adopted from [53]:

$$
\mathcal {L} _ {\mathrm {G R P O}} (\pi_ {\theta}) = - \mathbb {E} _ {(s, I) \sim \mathcal {D}, \{a ^ {(i)} \} _ {i = 1} ^ {G} \sim \pi_ {\mathrm {r e f}} (\cdot | s, I)} \tag {5}
$$

$$
\left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| a ^ {(i)} |} \sum_ {t = 1} ^ {| a ^ {(i)} |} \left\{\min  \left(r _ {t} ^ {(i)} (\theta) A ^ {(i)}, \operatorname {c l i p} \left(r _ {t} ^ {(i)} (\theta), 1 - \epsilon , 1 + \epsilon\right) A ^ {(i)}\right) - \beta D _ {\mathrm {K L}} ^ {(i, t)} \left(\pi_ {\theta} \| \pi_ {\mathrm {r e f}}\right) \right\} \right],
$$

where ri,t(θ) = πθ(a(i)|s, I ) $\begin{array} { r } { r ^ { i , t } ( \theta ) = \frac { \pi _ { \theta } \left( a ^ { ( i ) } | s , I \right) } { \pi _ { \theta _ { \mathrm { r e f } } } \left( a ^ { ( i ) } | s , I \right) } \mathrm { a n d } D _ { \mathrm { K L } } ^ { i , t } ( \pi _ { \theta } , \pi _ { \mathrm { r e f } } ) = \frac { \pi _ { \mathrm { r e f } } \left( a ^ { ( i ) } | s , I \right) } { \pi _ { \theta } \left( a ^ { ( i ) } | s , I \right) } - 1 - \log \frac { \pi _ { \mathrm { r e f } } \left( a ^ { ( i ) } | s , I \right) } { \pi _ { \theta } \left( a ^ { ( i ) } | s , I \right) } . } \end{array}$ $r ^ { i , t } ( \theta ) = \frac { \pi _ { \theta } ( a ^ { ( i ) } | s , I ) } { \pi _ { \theta _ { \mathrm { r e f } } } ( a ^ { ( i ) } | s , I ) }$ $D _ { \mathrm { K L } } ^ { i , t } ( \pi _ { \theta } , \pi _ { \mathrm { r e f } } ) = \frac { \pi _ { \mathrm { r e f } } ( a ^ { ( i ) } | s , I ) } { \pi _ { \theta } ( a ^ { ( i ) } | s , I ) } - 1 - \log \frac { \pi _ { \mathrm { r e f } } ( a ^ { ( i ) } | s , I ) } { \pi _ { \theta } ( a ^ { ( i ) } | s , I ) } .$ − 1 − log

Similar to [53, 17], advantage $A$ is weighted on the whole reasoning token logits to encourage free form thinking for performing action and planning.

The final training loss is defined as a weighted combination of positive and negative action samples, i.e., correct actions $a _ { T }$ and incorrect actions $a _ { F }$ : $\mathcal { L } ( \pi ( \theta ) ) = \mathcal { L } _ { \mathrm { G R P O } } + \gamma \mathcal { L } _ { \mathrm { A I } }$ . We set $\gamma = 0 . 2$ during training, and the rationale for this choice is discussed in the ablation study presented in Sup. F.

This strategy is shown to be more effective in Sec. 4.2 compared to Generalized Advantage Estimation (GAE) [52]-based RL methods [47, 6], as the more powerful reward model $\mathcal { M } _ { s t a t e }$ provides accurate step-level reward signals by leveraging the entire episode trajectory $\mathcal { H }$ from a global perspective.

# 3.3 From Specialist to Generalist.

Achieving a generalist agent capable of operating across multiple software platforms is an ambitious and valuable goal. We first attempted to train such a generalist directly using the proposed SEAgent framework across all software environments. However, this approach led to suboptimal performance compared to specialized agents, as the actor struggled to learn effectively in the multi-software environment.

We thus introduce a specialist to generalist strategy, as illustrated in Fig. 1. Specifically, we first train software-specialized agents via SEAgent on individual environments, allowing each to master a specific application. These specialists are then distilled into a single generalist model through supervised fine-tuning (SFT) on synthesized successful trajectories. Finally, the generalist is refined via SEAgent on multiple software. This generalist, now equipped with better reasoning, planning abilities, and software-specific commonsense, achieves significantly improved performance, outperforming both the SEAgent via direct general RL and the performance combination of multi-specialists as in Table 2.

# 4 Experiments

# 4.1 Benchmark of Reward Model for computer use agent.

Providing CUA agents with reliable reward signals is crucial for enabling self-evolution in agentic systems, consisting of an actor (CUA) and a judge model, especially when interacting with unfamiliar software environments. Recent work, AgentRewardBench [35], proposes to evaluate the precision of reward models by assessing the accuracy of judge predictions on web-based tasks using trajectories from diverse agents. Building upon AgentRewardBench [35], we further extend the evaluation beyond web tasks to a broader set of PC software environments. Specifically, we evaluate on all 339 feasible tasks from OSWorld [68], using rule-based criteria for determining success or failure. Trajectories are sampled from UI-TARS [48] and Gemini-2.5-Pro [14], and rule-based evaluation is used as ground-truth supervision. We then compute the confusion matrix by comparing the predictions of different reward models against these labels.

The judge strategy in AgentRewardBench [35] relies solely on the final state and the associated action history. However, it is more natural and reliable for a judge model to consider the entire trajectory when assessing task success, rather than focusing only on the final state. For example, consider

Table 1: Precision and Negative Predictive Value (NPV) on AgentReardBench [35] and OS-World [68] with last screenshot only (LS) or entire process screenshots (ES) as input. World State Model closes the gap with commercial model supporting full process high resolution screenshots as input. The co-training with screenshot change description (CD) improves judgment precision.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Input</td><td colspan="2">AgentRewardBench</td><td colspan="2">OS-World-Full</td><td colspan="2">Prof/Office</td></tr><tr><td>Precision</td><td>NPV</td><td>Precision</td><td>NPV</td><td>Precision</td><td>NPV</td></tr><tr><td rowspan="2">GPT-4o [23]</td><td>LS</td><td>68.1</td><td>92.3</td><td>46.3</td><td>88.2</td><td>40.5</td><td>81.0</td></tr><tr><td>ES</td><td>72.1</td><td>92.2</td><td>74.6</td><td>95.2</td><td>70.4</td><td>85.3</td></tr><tr><td rowspan="2">Qwen2.5-VL-72B [7]</td><td>LS</td><td>64.5</td><td>94.2</td><td>41.5</td><td>86.9</td><td>31.7</td><td>78.7</td></tr><tr><td>ES</td><td>26.2</td><td>83.0</td><td>26.8</td><td>83.0</td><td>25.6</td><td>76.6</td></tr><tr><td rowspan="2">Qwen2.5-VL-7B [7]</td><td>LS</td><td>64.1</td><td>90.3</td><td>37.3</td><td>85.2</td><td>31.8</td><td>79.0</td></tr><tr><td>ES</td><td>25.4</td><td>83.8</td><td>20.0</td><td>81.7</td><td>23.5</td><td>76.0</td></tr><tr><td>World State Model (w/o CD)</td><td>ES</td><td>69.1</td><td>88.5</td><td>71.1</td><td>88.4</td><td>65.0</td><td>81.1</td></tr><tr><td>World State Model (w/ CD)</td><td>ES</td><td>71.6</td><td>91.2</td><td>73.9</td><td>90.5</td><td>69.3</td><td>82.0</td></tr></table>

the task of booking a flight to London. A final state message such as "Your flight ticket has been successfully booked." does not confirm whether the correct date and time were selected, which can lead to compromised judgment accuracy.

However, we observe that current open-sourced LVLMs do not perform well under this more holistic evaluation strategy. As shown in Fig. 3, feeding additional historical screenshots into Qwen2.5-VL [7] significantly degrades its Average Precision (AP), diverging notably from GPT-4o [23] on the same curated prompt detailed in Fig.6. We attribute this performance drop to the insufficient pretraining of Qwen2.5-VL on long sequences of high-resolution screenshots, which likely pushes it toward the limits of its 32K context length.

To address this issue, we propose a distilled model based on Qwen2.5-VL-7B, referred to as World State Model, which conducts step-by-step screenshot analysis to produce final judgments. The training process for World State Model is detailed in Sup. A.2, using a dataset of 0.86K GPT-4o [23] generated evaluations on trajectories with dense GUI change descriptions, exclusively from Chrome within the OSWorld [68] environment. Despite being trained solely on Chrome data, World State Model exhibits strong generalization to both other professional software in OSWorld and the external AgentRewardBench [35] benchmark. This demonstrates that the model learns transferable judgment patterns rather than overfitting to the specifics of a single application, thanks to the diversity and quality of step-level annotations in the training data.

We evaluate World State Model and our full-process screenshot-conditioned strategy on AgentRewardBench [35], as well as on agent trajectories from OSWorld [68]. As shown in Tab. 1 and further analyzed in Fig. 3, World State Model achieves state-of-the-art performance among open-sourced models, significantly narrowing the gap with GPT-4o [23]. More importantly, it exhibits a similar performance trend to GPT-4o when conditioned on historical screenshots. Despite being trained on a relatively small dataset, World State Model is explicitly encouraged to capture the sequential dependencies among historical screenshots and to perform step-by-step reasoning for final judgment. Serving as our foundation reward model, World State Model provides reliable, step-level reward sig-

![](images/4047dd4aec8c88f891f62ea5139a244a47a4c65b1854f08ebd2b38b69633498a.jpg)  
Figure 3: The Average Precision on AgentRewardBench [35], where GUI-Judge exhibits an improvement in AP as the number of input middle states increases, showing a similar trend to that of the closed sourced GPT-4o [23] when compared with its base model.

In line with our agentic system design—which full open-sourced models—we intentionally avoid uring training and inference (also due to inefficiency). Sup.A.

Table 2: Success Rate (SR) on OSWorld [68]. SEAgent demonstrates strong performance after reinforcement learning from experience. In addition to evolving on separate software, a new General Model achieves better performance after another iteration of SEAgent. *Indicates specialist agents trained separately for each software with ensembled results. All results are averaged over three runs.   

<table><tr><td>Model</td><td>VScode</td><td>GIMP</td><td>Impress</td><td>VLC</td><td>Writer</td><td>Overall</td></tr><tr><td>Human Performance</td><td>73.9</td><td>73.1</td><td>80.9</td><td>70.6</td><td>73.9</td><td>74.5</td></tr><tr><td>GPT-4o [23]</td><td>4.35</td><td>3.85</td><td>6.77</td><td>16.1</td><td>4.35</td><td>7.08</td></tr><tr><td>GPT-4V [42]</td><td>0.00</td><td>7.69</td><td>2.52</td><td>18.3</td><td>4.35</td><td>6.59</td></tr><tr><td>Gemini-Pro-1.5 [60]</td><td>0.00</td><td>11.5</td><td>13.2</td><td>6.53</td><td>8.71</td><td>7.99</td></tr><tr><td>Claude3.7 Sonnet [4]</td><td>18.8</td><td>24.4</td><td>10.6</td><td>27.5</td><td>17.4</td><td>19.7</td></tr><tr><td>Gemini-Pro-2.5 [14]</td><td>21.7</td><td>26.9</td><td>9.92</td><td>25.5</td><td>24.6</td><td>21.7</td></tr><tr><td>UI-TARS-7B-DPO [34]</td><td>13.0</td><td>23.1</td><td>4.26</td><td>11.8</td><td>4.35</td><td>11.3</td></tr><tr><td>UI-TARS-72B-DPO [34]</td><td>18.8</td><td>25.6</td><td>6.38</td><td>15.7</td><td>8.70</td><td>15.0</td></tr><tr><td>DigiRL [6] (Specialized RL)*</td><td>21.7</td><td>32.1</td><td>12.8</td><td>23.5</td><td>18.8</td><td>21.8</td></tr><tr><td>WebRL [47] (Specialized RL)*</td><td>27.5</td><td>29.5</td><td>10.6</td><td>25.5</td><td>15.9</td><td>21.8</td></tr><tr><td>SEAgent (Specialized RL)*</td><td>37.7</td><td>38.5</td><td>22.0</td><td>33.3</td><td>29.0</td><td>32.2</td></tr><tr><td>DigiRL [6] (General RL)</td><td>21.7</td><td>35.9</td><td>12.1</td><td>19.6</td><td>15.9</td><td>21.0</td></tr><tr><td>WebRL [47] (General RL)</td><td>20.3</td><td>32.5</td><td>9.93</td><td>21.6</td><td>14.5</td><td>19.6</td></tr><tr><td>SEAgent (General RL)</td><td>36.2</td><td>39.7</td><td>19.9</td><td>31.4</td><td>26.1</td><td>30.6</td></tr><tr><td>SEAgent (General SFT)</td><td>30.4</td><td>37.2</td><td>18.4</td><td>31.9</td><td>20.3</td><td>27.9</td></tr><tr><td>SEAgent (Specialist-to-Generalist)</td><td>40.5</td><td>42.3</td><td>22.7</td><td>35.3</td><td>31.8</td><td>34.5</td></tr></table>

# 4.2 Self evolution of GUI Agents

![](images/0e3fb1d6421e43f9bf52f5c7ff74d1601f2741f8ac161653f6a8f3175ac5c9bf.jpg)  
Figure 4: Self-evolved task instructions and success rate (SR) curves across different software. Tasks are progressively upgraded by the Curriculum Generator without human intervention, based on the evolving capabilities of the Actor Model at different training phases.

Models Before Self-Evolution. Our self-evolving system is initialized with three locally deployed models: UI-TARS-7B-DPO [48] as the Actor Model, World State Model as the step-level reward model, and Qwen2.5-72B [72] as the Curriculum Generator for task evolution with software guidebook memory. We conduct experiments on five professional and office-related software applications from OSWorld [68]. As shown in Tab. 2, the initial actor agent demonstrates limited performance on these software environments, achieving an average success rate of $1 1 . 3 \%$ only.

Evolution Process Details. At beginning, we provide World State Model with the initial GUI state of the novel software. The Curriculum Generator then generates the first software guidebook and a set of basic tasks (illustrated in Fig.5). This yields an initial instruction set $\mathcal { T } _ { 0 }$ , averaging 150.2 instructions, which are executed by the Actor Model. The resulting trajectories are evaluated by World State Model and parsed into an average of 1361.5 multi-turn conversation pairs (detailed statistics are in Sup.H). We then perform reinforcement fine-tuning (RFT) following the methodology described in Sec. 3.2. Training is conducted for 1k iterations on 8 NVIDIA A100 80GB GPUs, with $G = 8$ , a batch size of 16, and a learning rate of $2 \times 1 0 ^ { - 5 }$ , scheduled via cosine decay. This evolution process is repeated iteratively for three phases using the same training configuration.

Specialist Evaluation. For a fair comparison with previous reinforcement learning methods [6, 47], we adapt their training strategies to the UI-TARS [48] model. Specifically, we initialize the actor

agent from UI-TARS-7B-DPO and, instead of providing step-level reward signals, We evaluate its executed trajectories with binary success or failure outcomes using World State Model. A separate critic model is also initialized from UI-TARS-7B-DPO, with additional random initialized MLP layers taking the LLM’s hidden states as input to regress value predictions. This critic is trained to perform advantage estimation based on Generalized Advantage Estimation (GAE) [52]. The loss functions follow the same configurations as in [6, 47]. Both the critic and the actor agent are trained iteratively using the same phased reinforcement fine-tuning (RFT) process, where the Curriculum Generator continually generates new curriculum-style tasks.

As shown in Fig. 4 and Tab. 2, we train separate actor agents for five different software applications. Our approach, denoted as SEAgent (Specialist), achieves strong performance compared to previous reinforcement learning methods such as DigiRL [6] and WebRL [47]. We attribute this improvement to the use of World State Model, which provides fine-grained, step-level reward signals derived from a comprehensive understanding of the full history of states and actions. This contrasts with previous approaches that rely on separate critic models—typically initialized from the actor itself—to estimate advantages from sparse, final success/failure signals. Furthermore, the curriculum of task instructions generated by the Curriculum Generator, as illustrated in Fig. 4, validates the effectiveness of our autonomous learning framework. These tasks progress from simple to complex based on the actor’s evolving capabilities, enabling it to gradually specialize in each target software environment. Based on the observed evolution curves, we set the number of training phases to three, as performance gains saturate beyond that point.

From Specialist to Generalist. After training five strong software specialists, we pursue generalization using the methodology described in Sec. 3.3. Specifically, we collect task instructions generated during each specialist’s training phase and use them to prompt the respective specialists for execution. A total of 3.5K successful trajectories, along with their corresponding reasoning traces, are distilled into a new base model (UI-TARS-7B [48]) via supervised fine-tuning (SFT). This distilled model is then further optimized through reinforcement learning (RL) across all five software environments.

As shown in Tab. 2, the resulting generalist model surpasses the performance of the individual specialist ensemble, demonstrating the effectiveness of a specialization-first strategy for achieving generalization. By learning from a broad range of software tasks, the generalist improves its reasoning and decision-making capabilities, acquiring transferable commonsense knowledge across domains.

Ablation Study of Specialist Training. In Tab. 3, we present an ablation study on the effectiveness of various components in our training pipeline, using the success rate on VS-Code from OSWorld [68] as the evaluation metric. First, we ablate the use of the World State Model for reward signal generation. Its high precision in judging the success or failure of the actor agent’s actions—compared to using a base model—is shown to be essential for effective self-evolution.

Table 3: Ablation of different configurations and their corresponding VScode success rates on OSWorld [68]. Using World State Model as the reward model yields significant performance gains. We further compare different training strategies including supervised fine-tuning (behavior cloning), GRPO, and Adversarial Imitation (AI).

<table><tr><td>Qwen2.5VL-72B</td><td>World State Model</td><td>SFT (BC)</td><td>GRPO</td><td>AI</td><td>VScode SR</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>13.0</td></tr><tr><td>✓</td><td></td><td>✓</td><td></td><td></td><td>10.1</td></tr><tr><td>✓</td><td></td><td></td><td>✓</td><td></td><td>11.6</td></tr><tr><td></td><td>✓</td><td>✓</td><td></td><td></td><td>23.2</td></tr><tr><td></td><td>✓</td><td>✓</td><td></td><td>✓</td><td>30.4</td></tr><tr><td></td><td>✓</td><td></td><td>✓</td><td></td><td>34.8</td></tr><tr><td></td><td>✓</td><td></td><td>✓</td><td>✓</td><td>37.7</td></tr></table>

In addition to reward quality, reinforcement fine-tuning (RFT) also proves critical. Compared to direct supervised fine-tuning (behavior cloning), RFT encourages more diverse and exploratory reasoning patterns under verifiable rewards, enabling more generalized task planning. Finally, incorporating adversarial imitation to penalize critical failure-inducing actions allows the CUA to learn from its mistakes, yielding additional performance gains. This highlights the importance of learning not only from successful behaviors but also from failure signals.

# 5 Conclusion

In this work, we introduce SEAgent, an autonomous Computer Use Agent (CUA) exploration system that learns from its own experience on specific software. Powered by a robust World State Model that provides step-level reward signals, and a carefully designed reinforcement learning framework that

encourages free-form reasoning through trial and error, the CUA is able to evolve into a specialist for individual software platforms. Furthermore, a specialist-to-generalist training strategy enables the development of a strong generalist agent capable of operating across multiple software environments. Given that computer software constitutes a highly regularized virtual world, we believe this work can inspire future research on agentic systems in both gaming and real world embodied environments.

Limitations and future work. While promising, our work still has several unresolved limitations. Firstly, our self evolving agent system is bounded by GUI-Judge to provide reliable reward signal instead of real signal from environment. As its still challenging to learning from sparse reward signal in complex environment. Secondly, though we tested on relatively complex and novel software like libreoffice-tools and GIMP. The task is still relatively simple as it only takes a human expert less than 20 step to accomplish. How to adapt the system to achieve hours-long workflow in even more challenging software used by real human expert are thus interesting future directions.

# References

[1] Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s: An open agentic framework that uses computers like a human. arXiv preprint arXiv:2410.08164, 2024.   
[2] Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s2: A compositional generalist-specialist framework for computer use agents. arXiv preprint arXiv:2504.00906, 2025.   
[3] Anthropic. Claude computer use. 2024.   
[4] Anthropic. Claude 3.7 sonnet system card. https://assets.anthropic.com/m/ 785e231869ea8b3b/original/claude-3-7-sonnet-system-card.pdf, 2025.   
[5] Anthropic. Claude’s extended thinking. 2025.   
[6] Hao Bai, Yifei Zhou, Jiayi Pan, Mert Cemri, Alane Suhr, Sergey Levine, and Aviral Kumar. Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning. Advances in Neural Information Processing Systems, 37:12461–12495, 2024.   
[7] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.   
[8] Thomas Carta, Clément Romac, Thomas Wolf, Sylvain Lamprier, Olivier Sigaud, and Pierre-Yves Oudeyer. Grounding large language models in interactive environments with online reinforcement learning. In International Conference on Machine Learning, pages 3676–3713. PMLR, 2023.   
[9] Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. Guicourse: From general vision language models to versatile gui agents. arXiv preprint arXiv:2406.11317, 2024.   
[10] Xinyue Chen, Zijian Zhou, Zheng Wang, Che Wang, Yanqiu Wu, and Keith Ross. Bail: Bestaction imitation learning for batch deep reinforcement learning. Advances in Neural Information Processing Systems, 33:18353–18363, 2020.   
[11] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935, 2024.   
[12] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. Advances in Neural Information Processing Systems, 36:28091–28114, 2023.   
[13] Shengyuan Ding, Shenxi Wu, Xiangyu Zhao, Yuhang Zang, Haodong Duan, Xiaoyi Dong, Pan Zhang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Mm-ifengine: Towards multimodal instruction following. arXiv preprint arXiv:2504.07957, 2025.   
[14] Google DeepMind. Gemini 2.5 Pro Preview (03-25). https://deepmind.google/ technologies/gemini, 2025.   
[15] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024.   
[16] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.   
[17] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[18] Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. A real-world webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856, 2023.   
[19] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919, 2024.   
[20] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281–14290, 2024.   
[21] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxuan Zhang, Juanzi Li, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Cogagent: A visual language model for GUI agents. CoRR, abs/2312.08914, 2023.   
[22] Xueyu Hu, Tao Xiong, Biao Yi, Zishu Wei, Ruixuan Xiao, Yurun Chen, Jiasheng Ye, Meiling Tao, Xiangxin Zhou, Ziyu Zhao, et al. Os agents: A survey on mllm-based agents for general computing devices use, 2024.   
[23] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.   
[24] Raghav Kapoor, Yash Parag Butala, Melisa Russak, Jing Yu Koh, Kiran Kamble, Waseem AlShikh, and Ruslan Salakhutdinov. Omniact: A dataset and benchmark for enabling multimodal generalist autonomous agents for desktop and web. In European Conference on Computer Vision, pages 161–178. Springer, 2024.   
[25] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. arXiv preprint arXiv:2401.13649, 2024.   
[26] Hanyu Lai, Xiao Liu, Iat Long Iong, Shuntian Yao, Yuxuan Chen, Pengbo Shen, Hao Yu, Hanchen Zhang, Xiaohan Zhang, Yuxiao Dong, et al. Autowebglm: A large language modelbased web navigating agent. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 5295–5306, 2024.   
[27] Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. Screenspot-pro: Gui grounding for professional high-resolution computer use. arXiv preprint arXiv:2504.07981, 2025.   
[28] Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. On the effects of data scale on computer control agents. arXiv e-prints, pages arXiv–2406, 2024.   
[29] Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Weixian Lei, Lijuan Wang, and Mike Zheng Shou. Showui: One vision-language-action model for gui visual agent. arXiv preprint arXiv:2411.17465, 2024.   
[30] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.   
[31] Junpeng Liu, Yifan Song, Bill Yuchen Lin, Wai Lam, Graham Neubig, Yuanzhi Li, and Xiang Yue. Visualwebbench: How far have multimodal llms evolved in web page understanding and grounding? arXiv preprint arXiv:2404.05955, 2024.   
[32] Zhiwei Liu, Weiran Yao, Jianguo Zhang, Le Xue, Shelby Heinecke, Rithesh Murthy, Yihao Feng, Zeyuan Chen, Juan Carlos Niebles, Devansh Arpit, et al. Bolaa: Benchmarking and orchestrating llm-augmented autonomous agents. arXiv preprint arXiv:2308.05960, 2023.

[33] Ziyu Liu, Zeyi Sun, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Haodong Duan, Dahua Lin, and Jiaqi Wang. Visual-rft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785, 2025.   
[34] Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. Gui odyssey: A comprehensive dataset for cross-app gui navigation on mobile devices. arXiv preprint arXiv:2406.08451, 2024.   
[35] Xing Han Lù, Amirhossein Kazemnejad, Nicholas Meade, Arkil Patel, Dongchan Shin, Alejandra Zambrano, Karolina Stanczak, Peter Shaw, Christopher J Pal, and Siva Reddy. Agen- ´ trewardbench: Evaluating automatic evaluations of web agent trajectories. arXiv preprint arXiv:2504.08942, 2025.   
[36] Kaixin Ma, Hongming Zhang, Hongwei Wang, Xiaoman Pan, Wenhao Yu, and Dong Yu. Laser: Llm agent with state-space exploration for web navigation. arXiv preprint arXiv:2309.08172, 2023.   
[37] Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Botian Shi, Wenhai Wang, Junjun He, Kaipeng Zhang, et al. Mm-eureka: Exploring visual aha moment with rule-based large-scale reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.   
[38] Shikhar Murty, Christopher Manning, Peter Shaw, Mandar Joshi, and Kenton Lee. Bagel: Bootstrapping agents by guiding exploration with language, 2024.   
[39] Shikhar Murty, Hao Zhu, Dzmitry Bahdanau, and Christopher D. Manning. Nnetnav: Unsupervised learning of browser agents through environment interaction in the wild, 2025.   
[40] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.   
[41] Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, et al. Gui agents: A survey. arXiv preprint arXiv:2412.13501, 2024.   
[42] OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023.   
[43] OpenAI. Operator. 2025.   
[44] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.   
[45] Jiayi Pan, Yichi Zhang, Nicholas Tomlin, Yifei Zhou, Sergey Levine, and Alane Suhr. Autonomous evaluation and refinement of digital agents. arXiv preprint arXiv:2404.06474, 2024.   
[46] Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, and Rafael Rafailov. Agent q: Advanced reasoning and learning for autonomous ai agents. arXiv preprint arXiv:2408.07199, 2024.   
[47] Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Wenyi Zhao, Yu Yang, Xinyue Yang, Jiadai Sun, Shuntian Yao, et al. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. arXiv preprint arXiv:2411.02337, 2024.   
[48] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, Wanjun Zhong, Kuanye Li, Jiale Yang, Yu Miao, Woyu Lin, Longxiang Liu, Xu Jiang, Qianli Ma, Jingyu Li, Xiaojun Xiao, Kai Cai, Chuang Li, Yaowei Zheng, Chaolin Jin, Chen Li, Xiao Zhou, Minchao Wang, Haoli Chen, Zhaojian Li, Haihua Yang, Haifeng Liu, Feng Lin, Tao Peng, Xin Liu, and Guang Shi. UI-TARS: pioneering automated GUI interaction with native agents. CoRR, abs/2501.12326, 2025.   
[49] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36:53728–53741, 2023.

[50] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573, 2024.   
[51] Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. Androidinthewild: A large-scale dataset for android device control. Advances in Neural Information Processing Systems, 36:59708–59728, 2023.   
[52] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. Highdimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.   
[53] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
[54] Haozhan Shen, Peng Liu, Jingcheng Li, Chunxin Fang, Yibo Ma, Jiajia Liao, Qiaoli Shen, Zilun Zhang, Kangjia Zhao, Qianqian Zhang, et al. Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615, 2025.   
[55] David Silver and Richard S Sutton. Welcome to the era of experience. Preprint of a chapter to appear in Designing an Intelligence, edited by George Konidaris, MIT Press (forthcoming), 2025.   
[56] Qiushi Sun, Kanzhi Cheng, Zichen Ding, Chuanyang Jin, Yian Wang, Fangzhi Xu, Zhenyu Wu, Chengyou Jia, Liheng Chen, Zhoumianze Liu, et al. Os-genesis: Automating gui agent trajectory construction via reverse task synthesis. arXiv preprint arXiv:2412.19723, 2024.   
[57] Qiushi Sun, Zhoumianze Liu, Chang Ma, Zichen Ding, Fangzhi Xu, Zhangyue Yin, Haiteng Zhao, Zhenyu Wu, Kanzhi Cheng, Zhaoyang Liu, et al. Scienceboard: Evaluating multimodal autonomous agents in realistic scientific workflows. arXiv preprint arXiv:2505.19897, 2025.   
[58] Zeyi Sun, Ziyang Chu, Pan Zhang, Tong Wu, Xiaoyi Dong, Yuhang Zang, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. X-prompt: Towards universal in-context image generation in auto-regressive vision language foundation models, 2024.   
[59] Zeyi Sun, Tong Wu, Pan Zhang, Yuhang Zang, Xiaoyi Dong, Yuanjun Xiong, Dahua Lin, and Jiaqi Wang. Bootstrap3d: Improving 3d content creation with synthetic data. arXiv e-prints, pages arXiv–2406, 2024.   
[60] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.   
[61] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
[62] Jianqiang Wan, Sibo Song, Wenwen Yu, Yuliang Liu, Wenqing Cheng, Fei Huang, Xiang Bai, Cong Yao, and Zhibo Yang. Omniparser: A unified framework for text spotting key information extraction and table recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15641–15653, 2024.   
[63] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.   
[64] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

[65] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
[66] Zhiyong Wu, Chengcheng Han, Zichen Ding, Zhenmin Weng, Zhoumianze Liu, Shunyu Yao, Tao Yu, and Lingpeng Kong. Os-copilot: Towards generalist computer agents with self-improvement. arXiv preprint arXiv:2402.07456, 2024.   
[67] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218, 2024.   
[68] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh J Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.   
[69] Long Xing, Qidong Huang, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Jinsong Li, Shuangrui Ding, Weiming Zhang, Nenghai Yu, et al. Scalecap: Inference-time scalable image captioning via dual-modality debiasing. arXiv preprint arXiv:2506.19848, 2025.   
[70] Yiheng Xu, Dunjie Lu, Zhennan Shen, Junli Wang, Zekun Wang, Yuchen Mao, Caiming Xiong, and Tao Yu. Agenttrek: Agent trajectory synthesis via guiding replay with web tutorials. arXiv preprint arXiv:2412.09605, 2024.   
[71] An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Lin, Linjie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian McAuley, Jianfeng Gao, et al. Gpt-4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation. arXiv preprint arXiv:2311.07562, 2023.   
[72] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.   
[73] Simon Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Peter Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, et al. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. Advances in neural information processing systems, 37:110935– 110971, 2024.   
[74] Chi Zhang, Zhao Yang, Jiaxuan Liu, Yanda Li, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pages 1–20, 2025.   
[75] Jianguo Zhang, Tian Lan, Rithesh Murthy, Zhiwei Liu, Weiran Yao, Ming Zhu, Juntao Tan, Thai Hoang, Zuxin Liu, Liangwei Yang, et al. Agentohana: Design unified data and training pipeline for effective agent learning. arXiv preprint arXiv:2402.15506, 2024.   
[76] Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. Android in the zoo: Chain-of-action-thought for gui agents. arXiv preprint arXiv:2403.02713, 2024.   
[77] Kaiyan Zhang, Biqing Qi, and Bowen Zhou. Towards building specialized generalist ai with system 1 and system 2 fusion, 2024.   
[78] Zhuosheng Zhang and Aston Zhang. You only look at screens: Multimodal chain-of-action agents. arXiv preprint arXiv:2309.11436, 2023.   
[79] Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614, 2024.   
[80] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. arXiv preprint arXiv:2307.13854, 2023.

[81] Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. Archer: Training language model agents via hierarchical multi-turn rl. arXiv preprint arXiv:2402.19446, 2024.   
[82] Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

# A World State Model

The World State Model (WSM) is a central component of SEAgent, responsible for understanding visual state changes and evaluating the effectiveness of the agent’s actions.

# A.1 Model Architecture and Operation

The WSM is built upon the Qwen2.5-VL-7B vision-language model. It operates in two distinct modes, each with a specific input-output structure to perform different tasks:

# 1. Trajectory Judgment:

Input: A sequence of screenshot images captured during an episode.

Output: Short captions for each screenshot, the reasoning process for the judgment, and a structured judgment dictionary (containing fields such as Correctness, Redundant, and First Error Step, as detailed in Figure 7 of the supplementary material).

# 2. State Change Description:

Input: Two screenshot images, one from before and one after a single action was executed.

Output: A detailed description of the visual differences between the two images.

# A.2 Fine-Tuning Dataset and Process

To equip the WSM with these capabilities, a specialized dataset was constructed for fine-tuning.

Data Construction The data construction process is as follows:

1. Trajectory Sampling: A Computer Using Agent (CUA), powered by UI-TARS and Gemini-2.5-Pro, was used to sample trajectories from 43 feasible tasks in Google Chrome within the OSWorld benchmark. These trajectories were saved as screenshot sequences.   
2. GPT-4o Annotation: Using the prompts detailed in Figures 6 and 7 of the supplementary material, GPT-4o was employed to annotate the sampled trajectories, generating judgments and screenshot captions. Only samples where the judgment matched the ground truth from OSWorld evaluation protocols were retained, resulting in 860 high-quality annotated trajectories.   
3. Change Description Data: An additional 1,000 pairs of (before action, after action) screenshots were sampled. GPT-4o was used to generate detailed descriptions of the differences, creating a 1,000-sample Change Description (CD) dataset.

Fine-Tuning Process The fine-tuning was performed using the Llama-Factory framework on 8 NVIDIA A100 (80G) GPUs for 2,000 iterations. A learning rate of $2 \times 1 0 ^ { - 5 }$ was used, and LoRA (rank ${ } = 1 2 8$ ) was employed for parameter-efficient fine-tuning. The 860 annotated trajectories serve as the core training data for teaching the model trajectory judgment, captioning, and reasoning. The 1,000-sample CD dataset acts as auxiliary data, specifically to encourage the model to focus on fine-grained visual differences, which enhances its overall state understanding. As shown in Table 1 of the main paper, incorporating CD data significantly boosts judgment performance. The two datasets were combined for training without any special re-weighting.

# A.3 Reward Generation from Trajectory Analysis

The trajectory judgment capability of the WSM is the core source of the reward signal for reinforcement learning. After an agent executes a full trajectory $\mathcal { H } = \{ s _ { 0 } , a _ { 0 } , s _ { 1 } , a _ { 1 } , . . . , s _ { \mathrm { f i n a l } } \}$ , the WSM analyzes it and outputs a structured judgment. Based on this output, actions within the trajectory are dynamically labeled as either positive actions $( a _ { T } )$ or failure actions $( a _ { F } )$ :

• Fully Successful Trajectory: If Correctness is ‘True‘ and there are no Redundant steps, all actions $a$ in the trajectory are labeled as $a _ { T }$ .   
• Successful but Inefficient Trajectory: If Correctness is ‘True‘ but Redundant steps begin at step $k$ , all actions prior to step $k$ are labeled as $a _ { T }$ .

• Failed Trajectory: If Correctness is ‘False‘ and the First Error Step is $e$ , all actions prior to step $e$ are labeled as $a _ { T }$ , while the erroneous action $a _ { e }$ is labeled as $a _ { F }$ .

These dynamically labeled $a _ { T }$ and $a _ { F }$ actions constitute the reward signals for the RL pipeline. During training, the actor predicts an action $a _ { t }$ based on the history $\{ a _ { 0 } , s _ { 0 } , \ldots , s _ { t } \}$ and uses these labels to calculate rewards.

# B Curriculum Generator

The Curriculum Generator is designed to dynamically produce tasks of increasing difficulty and diversity, guiding the agent through a systematic exploration of the software’s capabilities.

# B.1 Task Generation Mechanism

The workflow of the Curriculum Generator is detailed in the pseudocode in our supplementary material. Its core idea is to leverage the WSM’s analysis of completed tasks to generate new ones. The process, illustrated by the "add a rectangle" example from Figure 5, involves three main steps:

1. Analysis and Feedback: The agent successfully completes an initial task, "add a rectangle." The WSM analyzes the execution trajectory and extracts two key pieces of information: a task evaluation (Exam) and a list of observed state changes (CD_list).

CD_list: {"add a rectangle": ["The Edit bar is expanded...", "The cursor has changed into a cross...", "A blue box appears on the screen with side bars showing properties such as fill, line, color, width, transparency, and corner style..."], ...}

Exam: [{"task": "add a rectangle", "status": "success"}, ...]

2. Knowledge Integration and Task Generation: The CD_list and Exam are fed into the Curriculum Generator. It distills new knowledge, such as "properties of a rectangle," and integrates it into its internal Software guidebook. Based on this new knowledge, it generates more challenging tasks like "Add a green rectangle" or "Add a red rectangle with $50 \%$ transparency," which are then added to the task buffer.   
3. Iterative Learning: In the next RL phase, the agent samples from this updated, more challenging task buffer. The continuously enriched Software guidebook acts as the system’s long-term memory, driving the Curriculum Generator to propose increasingly sophisticated and unexplored tasks in subsequent rounds, thereby guiding the agent toward mastery.

# C Details of Curriculum Generator.

# C.1 Exemplar Case during Task Evolution.

We provide an exemplar case of our task evolution pipeline in Fig. 5, demonstrated using LibreOffice Impress. Initially, the World State Model parses a screenshot of the Impress interface into detailed captions describing the layout and individual buttons. The Task Generator then produces an initial task set, I0 = {I (1)0 , I (2)0 , . $\mathcal { T } _ { 0 } = \{ I _ { 0 } ^ { \bar { ( } 1 ) } , I _ { 0 } ^ { ( 2 ) } , . . . \}$ , and summarizes the initial software guidance memory $U _ { 0 }$ . The initial agent executes tasks in $\mathcal { T } _ { 0 }$ , such as “Add a Rectangle,” while the World State Model evaluates these actions, providing judgments and detailed descriptions of resulting changes. As shown in the Auto-Exploration stage, this includes generating captions for newly appeared property panels and assessing execution success. The Task Generator incorporates feedback on execution success and newly revealed properties (e.g., transparency) to evolve new tasks, such as “Draw a green rectangle with $50 \%$ transparency.” This process iteratively improves through reinforcement learning, enabling continuous task evolution and agent self-improvement.

# C.2 Comparative Analysis of Instruction Generation Strategies.

To validate the effectiveness of our Curriculum Generator, we conducted a comparative analysis against state-of-the-art instruction generation methods, namely those from NNetNav [39] and WebRL [47].

![](images/4b26d78ab2d23edfdef654054579bd88cd8044d98bcee5378fe3f1b0df15d57e.jpg)  
Figure 5: SEAgent autonomous exploration pipeline. The agent (policy model) and World State Model iteratively generate new task and perform RL to become a specialist in novel software.

Experimental Setup We adapted the official code and prompts from these prior works from web environments to general software applications. To ensure a fair comparison of the curriculum quality, for each strategy, we employed two leading LLMs: the open-source Qwen2.5-72B [7] and the proprietary Gemini-2.5-Pro [14]. The tasks generated by each strategy were used to train an RL agent (using GRPO only), with reward signals uniformly provided by our fine-tuned WSM. The evaluation was performed on two applications: VSCode from OSWorld (a standard software) and Celestia from ScienceBoard [? ] (a more challenging, out-of-domain scientific application). The primary metric was the task success rate.

Results and Discussion The results are presented in Table 4.

Table 4: Success rate $( \% )$ comparison of different task generation strategies on two software applications.   

<table><tr><td>Task Generation Strategy</td><td>LLM</td><td>VSCode</td><td>Celestia</td></tr><tr><td>WebRL</td><td>Qwen2.5-72B</td><td>27.5</td><td>0.00</td></tr><tr><td>WebRL</td><td>Gemini2.5-Pro-thinking</td><td>36.2</td><td>3.03</td></tr><tr><td>NNetNav</td><td>Qwen2.5-72B</td><td>34.6</td><td>0.00</td></tr><tr><td>NNetNav</td><td>Gemini2.5-Pro-thinking</td><td>43.6</td><td>5.05</td></tr><tr><td>Curriculum Generator (Ours)</td><td>Qwen2.5-72B</td><td>37.7</td><td>9.09</td></tr><tr><td>Curriculum Generator (Ours)</td><td>Gemini2.5-Pro-thinking</td><td>42.3</td><td>12.12</td></tr></table>

As shown, the reverse instruction generation strategy from NNetNav [39] is highly effective on the in-domain application (VSCode), demonstrating high data generation efficiency by producing successful trajectories. However, a critical trade-off was observed: this approach tends to generate many similar tasks, limiting its ability to explore the full breadth of the software’s functionalities. This limitation becomes more pronounced when the task generator is unfamiliar with the target software, as seen in the OOD Celestia environment.

In contrast, our guidebook-based method, while having a lower initial data generation efficiency, excels at systematic exploration. It builds structured knowledge of the software from scratch, making it more robust for tackling novel applications. This is evidenced by its superior performance on the more challenging Celestia software.

We conclude that these two strategies are complementary. Reverse instruction generation can efficiently exploit known functionalities, while our guidebook-based method can systematically explore new ones and help the task generator build a more comprehensive understanding of the target software. A hybrid approach combining both strategies is a promising direction for future work.

# D Test on TARS-1.5

Our work focuses on enabling agents to adapt to out-of-domain (OOD) and novel software where human-labeled data is not available. To test this, we applied our SEAgent pipeline to the UI-TARS-1.5 [48] model on two distinct benchmarks. On OSWorld [68], we observed moderate performance gains. We hypothesize this is because UI-TARS-1.5’s training data already targeted OSWorld, making it a familiar, in-domain environment for the base model. However, on the ScienceBoard [57] benchmark—a suite of scientific applications that are truly novel to UI-TARS-1.5—our pipeline delivers significant and substantial improvements. This strongly validates our core claim: SEAgent is most impactful when performing self-evolution learning on truly OOD software. We excluded two of the six ScienceBoard applications—Lean and TeX—as they are primarily text- and code-based software for mathematics and typesetting, which are not suitable for evaluating a GUI-centric agent like UI-TARS.

Table 5: Performance comparison on OSWorld and ScienceBoard benchmarks. Scores represent success rates $( \% )$ .   

<table><tr><td rowspan="2">Model</td><td colspan="3">OSWorld</td><td colspan="4">ScienceBoard</td></tr><tr><td>LibreOffice Impress</td><td>LibreOffice Writer</td><td>GIMP</td><td>ChamerX</td><td>GrassGIS</td><td>KAlgebra</td><td>Celestia</td></tr><tr><td>UI-TARS-1.5-7B-DPO</td><td>19.15</td><td>33.04</td><td>51.54</td><td>12.41</td><td>0.00</td><td>11.61</td><td>4.85</td></tr><tr><td>UI-TARS-1.5-7B-DPO+SEAgent</td><td>23.83</td><td>35.65</td><td>56.92</td><td>23.45</td><td>10.59</td><td>21.29</td><td>11.52</td></tr></table>

# E Sensitivity Analysis on Key Hyperparameters

We conducted a sensitivity analysis on key hyperparameters to evaluate their impact on the SEAgent pipeline. For model sampling, we set the temperature $t = 0$ for better reproducibility. We analyze two specific parameters: the number of generated tasks and the number of change descriptions. The results are presented in Table 6 and discussed below.

Table 6: Sensitivity analysis for key hyperparameters in the SEAgent pipeline, evaluated on VSCode. The metric is Success Rate $( \% )$ .   

<table><tr><td># Tasks Generated</td><td>VScode SR</td><td># Change Descriptions</td><td>VScode SR</td></tr><tr><td>30</td><td>31.88</td><td>30</td><td>33.33</td></tr><tr><td>50</td><td>36.23</td><td>50</td><td>37.68</td></tr><tr><td>100</td><td>37.68</td><td>100</td><td>37.68</td></tr><tr><td>200</td><td>37.68</td><td>200</td><td>34.78</td></tr></table>

Number of Generated Tasks This parameter controls the breadth of exploration in each learning cycle. As shown in our analysis, performance improves as more diverse tasks are generated, eventually plateauing around 100 tasks.

Number of Change Descriptions This parameter controls how much new information the generator receives to update its "software guidebook." We found a clear trade-off: A sufficient number of descriptions (50–100) is essential for the generator to learn about new UI functionalities and create meaningful, unexplored tasks. However, providing too many descriptions (e.g., 200) creates an overly long context for the LLM, which degrades the quality of task generation and hurts final performance.

# F Ablation on the Loss Balance Factor.

In Sec.3.2, we use $\gamma$ to balance the ratio of two loss item: adversarial imitation that learn from error and GRPO that learn to achieve success. We ablate the choice of $\gamma$ in Tab.7, according to which we set $\gamma = 0 . 2$ in main experiments.

Table 7: VScode Success Rate on OSWorld [68] under different loss balance factor $\gamma$ values.   

<table><tr><td>γ</td><td>0.0</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.5</td><td>0.8</td></tr><tr><td>Success Rate (%)</td><td>34.8</td><td>36.2</td><td>37.7</td><td>31.9</td><td>26.1</td><td>23.1</td></tr></table>

# G Reward Function for Different Actions.

Table 8: Reward computation for each action type in GUI agent   

<table><tr><td>Action Type</td><td>Description</td><td>Distance-based Reward</td></tr><tr><td>click, left_single, right_single, hover</td><td>Click or hover on a location</td><td>Normalized L1 distance between predicted and ground-truth coordinates</td></tr><tr><td>left-double, double click</td><td>Double click on a region</td><td>Normalized L1 distance between clicked coordinates</td></tr><tr><td>drag, select</td><td>Drag from start box to end box</td><td>Intersection over Union (IoU) between predicted and ground-truth boxes</td></tr><tr><td>type</td><td>Type textual input</td><td>Character-level BLEU score between predicted and ground-truth text</td></tr><tr><td>hotkey</td><td>Press multiple keys at once</td><td>Character-level BLEU score between predicted and ground-truth key combinations</td></tr><tr><td>press</td><td>Press a single key</td><td>Character-level BLEU score between predicted and ground-truth key</td></tr><tr><td>scroll</td><td>Scroll in a certain direction</td><td>Character-level BLEU score between predicted and ground-truth direction</td></tr><tr><td>move Mouse</td><td>Move mouse to a specific location</td><td>Normalized L1 distance between predicted and ground-truth coordinates</td></tr><tr><td>highlight</td><td>Highlight a rectangular UI region</td><td>IoU between predicted and ground-truth region</td></tr><tr><td>copy, paste</td><td>Clipboard operations</td><td>BLEU score between copied/pasted content</td></tr><tr><td>wait</td><td>Explicit wait command</td><td>Fixed reward + 1</td></tr><tr><td>finished, finish_task</td><td>Finish current task/trajectory</td><td>Fixed reward + 1</td></tr></table>

# H Data Statistics during Iterative Reinforcement Learning.

Table 9: Number of episode (Success/Failure) across four phases for different software tools during self-evolution. Each episode contains 8.8 multi-turn conversions in average.   

<table><tr><td></td><td>Phase0</td><td>Phase1</td><td>Phase2</td><td>Phase3</td></tr><tr><td>VSCode</td><td>112/39</td><td>282/83</td><td>161/34</td><td>98/55</td></tr><tr><td>GIMP</td><td>104/51</td><td>309/90</td><td>183/50</td><td>95/52</td></tr><tr><td>Impress</td><td>102/44</td><td>290/92</td><td>185/61</td><td>87/51</td></tr><tr><td>VLC</td><td>85/29</td><td>114/41</td><td>160/48</td><td>53/27</td></tr><tr><td>Writer</td><td>123/62</td><td>278/101</td><td>201/69</td><td>101/43</td></tr></table>

# I Detailed Prompt Templates.

For evaluation on AgentRewardBench [35], we use their official template for final state screenshot only testing and modified prompt in Fig.6 for entire process (or sampled middle screenshots) testing.

For evaluation on OSWorld Sampled trajectories, we use prompt in Fig.7 to prompt GPT-4o to provide step level judges, the sampled judges on Chrome in OSWorld [68] serves as training data of GUI-Judge. This template is also used in training GUI-Judge and at inference time in autonomous exploration stage.

For navigator, we use prompt template in Fig.8, which takes previous software usage manual and the performance of actor agent evaluated by judge (Empty if in initial phase.) as well as detailed exploration caption as input and output the updated usage manual as well as new task for agent to execute.

# J Self documented usage manual on different software during exploration.

In Fig.9 Fig.11, Fig.10, Fig.12, we demonstrate the self-documented usage manuals of the navigator (Qwen2.5-72B [72]) in the exploration and learning system introduced in Sec.3.1.

# Web Step Level Judge Prompt Template

You are a Language Model specialized in judging the performance of web agents in web-navigation tasks. For a certain website, you are given the goal of a navigation task, the current URL of the webpage, the actions taken by the agent, and the thought process of the agent. Additionally, you will have access to the sequence of key frame screenshots Your task is to answer several questions about the agent's performance in the task.

You should carefully look at the sequencial screenshot images in order to decide whether its sucessfully finish the task or failed halfway.

Question 1: Was the sequence of actions successful in achieving the goal?

Choices: <success>Successful</success>, <success>Unsuccessful</success>

Question 2: Did the agent perform unnecessary actions that could lead to unintended side effects?

Choices: <side>Yes</side>, <side>No</side>

Question 3: Did the agent perform the task optimally, by only performing necessary actions and avoiding unnecessary ones?

Choices:

<optimal>1. Complete Failure</optimal>   
<optima $^ { > 2 }$ . Suboptimal</optimal>   
<optima ${ > } 3$ . Somewhat Optimal</optimal>   
<optima ${ > } 4 .$ . Completely Optimal</optimal>

Question 4: Did the agent loop through a sequence of actions that did not make progress towards the goal?

Choices: <loop>Yes</loop>, <loop>No</loop>

Provide your reasoning for each question.

Your answer **must** follow this exact format:

<reasoning $>$ your reasoning here</reasoning>   
<success>answer</success>   
<side>answer</side>   
<optimal>answer</optimal>   
<loop>answer</loop>

Figure 6: Prompt Template of GUI-Judge for web agent trajectories evaluations with history screenshots as input, its difference with default prompt of AgentRewardBench [35] is highlighted in bold.

# K Broader Impacts

Potential positive societal impacts: SEAgent introduces a self-evolving paradigm for Computer Use Agents (CUAs), enabling them to autonomously learn and adapt to previously unseen software without human supervision. This significantly reduces the need for extensive manual data annotation and domain-specific customization, allowing intelligent agents to assist users across a wide range of applications—including productivity tools, multimedia editing, and educational software. By automating repetitive tasks and providing guidance in complex software environments, SEAgent holds promise for improving accessibility, enhancing digital literacy, and reducing cognitive workload in both professional and everyday settings.

Potential negative societal impacts: The capability of SEAgent to autonomously explore and operate complex software also introduces risks of misuse. Malicious actors might repurpose SEAgent for unauthorized software automation, such as automating account creation, spamming interfaces, or conducting surveillance via GUI interactions. In addition, as the agent learns from its own experience, there exists a risk that the agent may inadvertently inherit or amplify software-specific biases, potentially leading to unfair or inappropriate behaviors in sensitive applications (e.g., finance, legal automation). Mitigation strategies include controlled release of models, behavior filters during deployment, and incorporating safeguards in the World State Model to detect and prevent unintended or adversarial behavior.

# OSWorld Step Level Judge Prompt Template

I am evaluating the performance of a UI agent. The images provided are sequential keyframes that represent the full execution trajectory of the agent when attempting to follow a command. These keyframes correspond to the instruction: [INSTRUCTION].

Please thoroughly analyze the sequence to assess the following aspects:

1. Correctness — Did the agent successfully complete the task as instructed?   
2. Redundant Steps — Identify any unnecessary or repeated actions that do not contribute to the goal.   
3. Optimization — Did the agent follow an efficient plan with a minimal number of steps?   
5. First Error Step — If the execution is incorrect or sub-optimal, determine the index of the first 5. keyframe where a mistake occurred.   
6. Error Analysis — Provide a brief explanation of the mistake at that step.   
7. Correct Action Suggestion — Explain what the agent should have done instead at the point of error.

Important Instructions:

The agent may have made progress toward the goal, but unless the task is fully and correctly completed, you must set 'Correctness' to False.

Be cautious in determining success. Missing confirmation screens, skipped inputs, or wrong UI elements clicked all count as errors.

Carefully examine all UI changes, button interactions, text entries, and any visual feedback in the screenshots.

Clearly indicate which exact steps are redundant (starting from 1).

Once you finish the analysis, return your evaluation in the following dictionary format. Include your step-by-step reasoning above the result.

<thinkires_dict $>$ step by step reasoning.</thinking> $= \left\{ \begin{array} { r l r l } \end{array} \right.$

"Correctness": True or False,   
"Redundant": [step numbers],   
"Optimized": True or False,   
"First_Error_Step": step number or None,   
"Error_Type": "brief description of the mistake",   
"Correct_Action": "what should have been done instead"

Figure 7: Prompt Template of GUI-Judge for OSWorld [68] trajectories, which prompts judge model to provide step level reward signal.

# L SEAgent Self-Evolution Algorithm

Algorithm 1 presents the core self-evolution training loop of SEAgent in a specialized software environment. The procedure is divided into four major stages:

(1) Task Initialization. Given the initial GUI state of a target software application, the World State Model performs dense captioning to extract structural semantics (e.g., menu bar, buttons), which is used by the Curriculum Generator to create an initial set of executable tasks and an editable software guidebook.   
(2) Autonomous Exploration and Effect Evaluation. The agent explores each task via its current policy. The World State Model then performs step-level trajectory analysis, assigning each action a feedback label—either correct $( a _ { T } )$ or incorrect $( a _ { F } ) .$ —and generating GUI state change captions. This produces rich supervision signals for both policy learning and downstream task generation.   
(3) Policy Update via Reinforcement Fine-Tuning. Based on the labeled execution data, positive and negative action steps are separated. We apply Group Relative Policy Optimization (GRPO) to reinforce correct actions, and Adversarial Imitation (AI) to suppress failure-prone behaviors. The updated policy is used for the next exploration round.

# Task Buffer Update Prompt Template

You are now a teacher training a Computer Use Agent (CUA). This CUA is exposed to a new software environment and undergoes multiple rounds of iterative training. Your task is to issue new tasks for the agent to explore and train on, based on the feedback from the agent's actions. You are also responsible for summarizing a software usage manual to help the agent remember knowledge about the software.

The agent has provided the following feedback on its operations within the software:

{json.dumps(action_decription_list)}

Here is the software usage document you summarized in the previous round: {document}

Here is the agent's performance on the task you provided in the previous round:

{json.dumps(exam)}

Your are also access to the previous given tasks with the screenshot caption after agent's execution. You can also use these captions and results to evaluate the agent's capability and generate new task and update document accordingly given the caption of the new screen and the corresponding intruction with judged evaluation: {json.dumps(prev_states)}

Please:

- Analyze the agent's performance.

- Integrate new knowledge from the feedback.

- Update the usage manual accordingly.

- Design a new set of tasks (with increased difficulty) (30 or more) that reinforce the concepts the agent struggled with in the last round.

- Each task **must be concise and specific** , targeting a concrete atomic action, based on the document and agent's observations, such as:

- "Create a file named main.py."

- "Open Terminal card."

- Each task must be executable from software initial state with no file open, e.g. you should not generate task like save xxx.txt if xxx.txt doesn't exist or created.

- if task is in sequencial order with reliance, you should output a seq list like [subtask1, subtask2, ...], if there is no reliance, output [task].

- Decompose and target previous errors in a more focused way.

Output your reasoning and analysis process first. Then output the updated usage document and task list in the following JSON format within a SINGLE JSON DICT easier for me to parse:

```json
json
{
    "software_document_new": "...",
    "exam_new": [[subtask1, subtask2, ..., [task]...]]
} 
```

Figure 8: Prompt Template for task buffer update, which generates new tasks in a curriculum manner and update software documents. The new tasks are used for actor to perform next phase of RL.

(4) Task Update. The Curriculum Generator leverages feedback signals $( \mathcal { T } )$ and GUI state transitions $( \mathcal { C } )$ to propose more diverse and challenging tasks, thereby expanding the task frontier in a curriculum fashion.

This process repeats over multiple curriculum phases, ultimately yielding a specialized agent policy capable of mastering complex operations in the given software environment.

# Visual Studio Code Usage Manual (v2)

# 1.Overview

# 2. Menu Bar

# 2.1. File Menu

•New Text File (Ctrl+N)t Creates a new, untitled, unsaved text file (e.g, °Untitled-1").

• Save As., (Ctri+Shift+S): O soecific name and locatio

•Preferences:

wrspace setings,

•Close Editor (Ctrl+W) (

# 2.2. View Menu

•CombandPalete,

r (Ctrt

Search (Ct

•Extensions (Ctr

• Word Wrap (Alt+Z): T

B

#

G to File,(Ctri+P) worikspace bv name

#

#

#

• New Tern al (Ctrl+Shift+*); O)

tw0

#

# 3. Activity Bar

•Source Contrel Icon (Branchingt Opens the Source Control panel for Git operations. If no folder is

# 4. Integrated Terminal Panel

# 5.Command Palette

# 6. Settings UI

#

# 8.Dialogs & Pop-ups

Figure 9: Automatically generated usage manual during self exploration on VScode.

# GIMP (GNU Image Manipulation Program) Usage Manual

# 1. GIMP Interface Overview

# 1. Main Menu Bar (Top of Application Window)

# 2, Toolbox (Typically on the Left Side)

# Tool Icons Grid

#

# 3. Tool Options Panel (Tvpicallv Below the Toolbox)

#

# Tool C

# 4.Jmag a(Center)

# 5. Dockable Dialogs (Typically on the Right Side)

Common Tabs:

# II. Common Operations

4. Manage Colors

6. Use Di

# III. Important Distinctions

'Colors’ 1 r Swatches

Figure 10: Automatically generated usage manual during self exploration on GIMP.

# LibreOffice Impress (Version 4.x) Usage Manual-v1

# 1. Overview

# 2.Main Application Window Components

# 2.1. Menu Bar (Top)

# 2.2. Standard Toolbar (First Row of lcons)

# 2.3. Drawing / Formatting Toolbar (Second Row)

#

# 2.4. Notification Area (Below Toolbars)

Exampl

# 2.5. Main Interface Panes

# 2.5.1. Slides Pane (Left)

# 2.5.2. Workspace / Slide Editor (Center)

# 2.5.3. Sidebar (Right)

Vertical Tab Bar Icons (far right):

Properties Tab Example Content

# 2.6. Status Bar (Bottom of the Window)

Center

Figure 11: Automatically generated usage manual during self exploration on LibreOffice_Impress.

```txt
LibreOffice Writer License
LibreOffice Writer is a word processor. The main interface consists of a Menu Bar, Toolbars, a Document Area, a Status Bar, and a Sidebar. An initial launch typically opens a new blank document (e.g., "Untitled 1").  
2. Main Window Components  
2.1. Title Bar  
- Display the current menu and application name (e.g., selected 1).<Lcneffice:  
    - written or document name; clicks <Lcneffice: written>  
2.2. Menu Bar  
- Located at the top of the application window, below the Title Bar.  
- Contains drop-down menus for various commands.  
- Items with <...>, <...>open dialog; arrows indicate submenus.  
Menu Include:  
- File > New, Open, Save, Print, Export.  
- Save: Unio Redo, Cut, Copy, Paste, Find and Replace.  
- View: Toolbars, Zoom, Layout.  
- Insert: Images, Tables, Charts, Page Breaks, Hyperlinks, Comments.  
- Format: Bold, Paragraph, Page Style.  
- Style: Manage and apply styles.  
- Table: Table operations and properties.  
- File: Form controls like Text Box, Design Mode.  
- Yes! : Spelling, Macro Options.  
- Write: Manage open windows.  
- Help: help contents and about page.  
2.3. Toolbars  
Two rows: Standard and formatting toolbars.  
2.3.1. Standard Toolbar  
Common icons include:  
- New, Save, Save, Expert as .RDF, Print, Preview;  
- Cut, Copy, Paste.  
- Close: Formattinging  
- SaveAs, SaveTo 
```

- field and replace, Spelt1ng.
- Toggle formatting works (t)
- Insert: Table, Page, Chart, Text Box, Page Break, Special Character, Hyperlink,
		Equimode, Table of Contents, Comment:
- Track changes.
- Line splicing.
- View arrow functions.  
2.3.2. Formatting Toolbar
Text formatting controls include:
- Paragraph style, droppendown
- Font Name, Font Size
- WHRD, WHTC,字体, stroke/length
- superscript, Subscript
- Clear, Direct, HotSpelling
- metacolor, lightHgItting
- Alignment: Left, Center, Right, Justify
- Lists to/Isolated, Numbered
- decrease/increase Indents
- More options ()  
2.4. Rulers
- Horizontal Radius: for margins, tabc, Indents.
- Vertical Ruler: For vertical placement, page margins.  
2.5. Document Area
- Central workspace for typing and editing.
2.6. Vertical Scrollbar
- Located on the right, scrolls the document vertically.  
2.7. Status Bar
Display:
- Page number, word/character count.
- Page style
- Document language
- View layout: Icon, single page, multi-page, book mode
- Zoom controls:
    - $\times  \leftarrow  / +  \times$ buttons

- Sizzle handle
- Percentage display (e.g., $30\%$ )
2.8. Right Sidebar
Vertical drop with fringes for panels:
- Properties (sidebars/width score)
- Page (page icon)
- styles (A + brush)
- IsThere (picture frame)
- NavInspector (components)
2.9. Information Banners
- Appear under Indicators with messages like "Help us make Library better"
- Can be closed with an $\times$ .
2.10. Dialog Boxes
- Actions like $\#16\times$ .open... or $\#16\times$ .close... open dialogs.
- Usually include (dc, cancel), close.
2.11. Toolkits
- Hovering over UI elements shows toolkits (e.g., "Bold")
2.13. File Format Support
Writer supports a wide range of document formats:
- Default Format: .clifts (.Open Document Text)
- Common Import/Export Formats:
  - .clift/. .clift (.Text/Word)
  - .clift (.Rich Text Format)
  - .clift (.Plain Text)
  - .clift/. .clift
  - .clift (.Export only)
Note: Some formatting may be lost when opening or saving in non-native formats.

Figure 12: Automatically generated usage manual during self exploration on LibreOffice_Writer.

Algorithm 1 SEAgent Specialized Self-Evolution Training Loop  
1: Input: Initial policy $\pi_0$ , World State Model $\mathcal{M}_{\mathrm{state}}$ , Curriculum Generator $\mathcal{M}_{\mathrm{task}}$ , Initial GUI state $S_0$ 2: 1. Task Initialization  
3: $\mathcal{C}_0 \leftarrow$ CaptionGUI( $S_0$ )  
4: $\mathcal{I}_0, U_0 \leftarrow \mathcal{M}_{\mathrm{task}}(\emptyset, \emptyset, \emptyset, \mathcal{C}_0)$ 5: for $p = 0$ to $P - 1$ do  
6: 2.1 Autonomous Exploration  
7: $\mathcal{D}_{\mathrm{traj}} \leftarrow \emptyset$ 8: for all $I \in \mathcal{I}_p$ do  
9: $\tau \leftarrow$ ExecuteInstruction( $\pi_p, I$ )  
10: 2.2 Effect Evaluation  
11: $\mathcal{J}_I, \mathcal{C}_I \leftarrow \mathcal{M}_{\mathrm{state}}(\tau)$ 12: $\mathcal{D}_{\mathrm{traj}} \leftarrow \mathcal{D}_{\mathrm{traj}} \cup \{(\tau, \mathcal{J}_I, \mathcal{C}_I)\}$ 13: end for  
14: 2.3 Policy Update (RFT)  
15: Split $\mathcal{D}_{\mathrm{traj}}$ into:  
16: $\mathcal{D}_{\mathrm{pos}}$ : steps labeled as positive $a_T$ 17: $\mathcal{D}_{\mathrm{neg}}$ : steps labeled as negative $a_F$ 18: Compute GRPO loss on $\mathcal{D}_{\mathrm{pos}}$ :  
19: $r(a, a_T) = \mathbb{I}[type(a) = type(a_T)] + r_{\mathrm{dist}}(a, a_T)$ 20: Compute Adversarial Imitation loss on $\mathcal{D}_{\mathrm{neg}}$ :  
21: $\mathcal{L}_{\mathrm{AI}} = -\log \frac{\pi_\theta(a|s,I)}{\pi_{\mathrm{ref}}(a_F|s,I)}$ 22: Total loss: $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{GRPO}} + \gamma \mathcal{L}_{\mathrm{AI}}$ 23: $\pi_{p+1} \leftarrow$ Update( $\pi_p, \mathcal{L}_{\mathrm{total}}$ )  
24: 2.4 Task Update  
25: $\mathcal{I}_{p+1}, U_{p+1} \leftarrow \mathcal{M}_{\mathrm{task}}(U_p, \mathcal{I}_p, \{\mathcal{J}_I\}, \{\mathcal{C}_I\})$ software knowledge and performance feedback  
end for  
Output: Specialized agent policy $\pi_P$ after $P$ stages of self-evolution
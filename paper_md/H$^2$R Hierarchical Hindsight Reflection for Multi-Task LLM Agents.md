# $H ^ { 2 } R$ : Hierarchical Hindsight Reflection for Multi-Task LLM Agents

1st Shicheng Ye

Sun Yat-sen University

yeschch@mail2.sysu.edu.cn

$2 ^ { \mathrm { n d } }$ Chao Yu

Sun Yat-sen University

yuchao3 $@$ mail.sysu.edu.cn

$3 ^ { \mathrm { r d } }$ Kaiqiang Ke

Sun Yat-sen University

kekq@mail2.sysu.edu.cn

$4 ^ { \mathrm { t h } }$ Chengdong Xu

Sun Yat-sen University

xuchd6@mail2.sysu.edu.cn

$5 ^ { \mathrm { t h } }$ Yinqi Wei

The University of Sydney

weiy0041@e.ntu.edu.sg

Abstract—Large language model (LLM)-based agents have shown strong potential in multi-task scenarios, owing to their ability to transfer knowledge across diverse tasks. However, existing approaches often treat prior experiences and knowledge as monolithic units, leading to inefficient and coarse-grained knowledge transfer. In this work, we propose a novel hierarchical memory architecture that enables fine-grained knowledge transfer by decoupling high-level planning memory from low-level execution memory. To construct and refine these hierarchical memories, we introduce Hierarchical Hindsight Reflection $( H ^ { 2 } R )$ , a mechanism that distills reusable and hierarchical knowledge from past agent–environment interactions. At test time, $H ^ { 2 } R$ performs retrievals of high-level and low-level memories separately, allowing LLM-based agents to efficiently access and utilize task-relevant knowledge for new tasks. Experimental results across two benchmarks demonstrate that ${ \hat { H } } ^ { 2 } R$ can improve generalization and decision-making performance, outperforming prior baselines such as Expel.

Index Terms—large language model, agent, memory, multi-task

# I. INTRODUCTION

Multi-task learning [1] is a key step toward general artificial intelligence, as it requires agents to handle a variety of tasks with distinct goals and requirements. Due to the strong generalization and in-context learning abilities of large language models (LLMs), LLM-based agents are particularly well suited to multi-task settings by adapting knowledge and skills acquired from previous tasks to novel situations [2].

A common paradigm for LLM-based agents to solve multitasks is to construct a memory repository storing task-solving insights extracted from past interactions, such as comprehension of environmental dynamics [3], improvement of task execution plans [4]–[6], and correction of errors [7]. When dealing with new tasks, the agents can selectively retrieve the most relevant memories from this repository to inform decision making, thereby improving performance in multi-task scenarios. For example, an agent in a household environment that has learned to “clean a pan and place it on the countertop” could reuse the knowledge from prior experiences to execute a new task like “cooling lettuce and placing it on the countertop” more effectively.

However, existing approaches often treat episodic experiences and insights as coarse-grained units representing wholetask knowledge. As a result, knowledge from previous tasks may include irrelevant subgoals, which can distract reasoning and degrade performance on new tasks. Building on the earlier example, suppose the LLM-based agent has previously learned the task “cleaning a pan and placing it on the countertop” and now faces a new task “cooling lettuce and placing it on the countertop”. When storing the whole-task knowledge in the memory, the agent may retrieve the entire memory unit of the previous task. This coarse-grained memory unit may include knowledge associated to the irrelevant subgoal “cleaning a pan”, which can divert attention from the reusable placement subgoal “placing it on the countertop”, thus increasing cognitive overhead and hindering performance [8]. This example highlights the importance of transferring only minimally taskrelevant fragments of knowledge in multi-task settings.

To address this issue, we propose a hierarchical memory framework that is structured into a high-level planning memory and a low-level execution memory. To construct and refine this memory, we introduce a Hierarchical Hindsight Reflection $( H ^ { 2 } R )$ mechanism that distills agent–environment interactions from past tasks into structured, semantically meaningful memory representations. At test time, when dealing with new tasks, the agent can selectively retrieve high-level memories for subgoal planning and low-level memories for action execution, which allows for more targeted and efficient knowledge reuse. This architecture leads to improved robustness and efficiency in generalization across diverse multi-task scenarios. To validate the efficacy of the framework, we conduct experiments across two benchmarks, including AlfWorld and PDDLGame. Experimental results demonstrate that our framework enables more efficient knowledge transfer and outperforms existing baselines like Expel.

The remainder of this paper is organized as follows: Section II reviews related works, Section III formalizes the problem setting, Section IV presents the proposed $H ^ { 2 } R$ mechanism, Section V reports experimental results and analysis, and Section VI concludes with future directions.

# II. RELATED WORKS

# A. LLMs in Decision Making

When an LLM acts as a decision-making agent, it conditions its textual output on the input prompt’s contextual state representation, which is a composite encoding of environmental observations, objectives, and relevant information [9]. This foundational capability is further significantly enhanced by structured reasoning frameworks such as Chain-of-Thought (CoT) [10], Reasoning-Acting (Re-Act) [11], Retrieval-Augmented Generation (RAG) [12], etc. Recent advances have expanded the decision-making and reasoning abilities of LLM-based agents across diverse domains, including embodied robotics [13], clinical diagnostics [14], quantitative finance [15], and others [16], [17].

# B. Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) enhances LLMs by integrating external evidence such as knowledge bases or the web [12]. However, its reliance on such resources hinders deployment in scenarios where they are inaccessible or costly, a common situation in agent–environment interactions. Inspired by RAG, we shift the retrieval of external existing knowledge inward to the retrieval of internal generated memories, i.e., deriving knowledge directly from agent trajectories and reflections to enable context-sensitive reuse of task-relevant experience and improve decision making in new tasks.

# C. Experiential Learning for LLM Agents

Experiential learning encodes agent–environment interactions into compact representations and reflective summaries, which can be selectively retrieved to guide actions in new tasks. Recent approaches have instantiated this paradigm in diverse ways: Reflexion [7] encodes feedback into textual directives stored in episodic memory for self-corrective learning; Voyager [18] builds a skill library by synthesizing reusable code primitives from trajectories; and methods such as ExpeL [4], RAP [19], and Skill Set Optimization [20] extract insights or trajectories from prior tasks to enhance decision making. Further work incorporates causal analysis to ground responses and reduce hallucination [3], [21]. Nonetheless, most existing works treat episodic memories as coarse-grained units, risking irrelevant knowledge transfer that can impair performance, whereas recent findings highlight the need for selective and modular memory reuse [8].

# III. PROBLEM STATEMENT

We consider an autonomous LLM agent embedded within a discrete interactive environment to sequentially execute predefined tasks. Similar to Reinforcement Learning (RL) [22], such interaction process could be modeled as a Partially Observable Markov Decision Process (POMDP), which provides a mathematical framework for sequential decision making under uncertainty, formally defined by the tuple $\langle S , A , \mathcal { O } , T , \Omega , R , \gamma \rangle$ . Under this formulation, an agent receives only ambiguous observations $o _ { t } \in \mathcal { O }$ generated through the observation function $\Omega ( s , a ) = \operatorname* { P r } ( o | s , a )$ . The historical interaction trajectory at

timestep $t$ is formalized as $h _ { t } = ( o _ { 0 } , \ldots , a _ { t } , o _ { t } )$ , where $\left( { { a } _ { t } } , { { o } _ { t } } \right)$ represents action-observation pairs at step $t$ . Particularly, at the final timestep $T , h _ { T }$ captures the complete sequence of interactions, denoted as the full trajectory $\tau$ . The structured input prompt for the agent at timestep $t$ is then formed from the current trajectory $h _ { t }$ , the task goal $g$ , and relevant knowledge $\kappa$ , i.e., $p _ { t } = ( h _ { t } , g , \mathcal { K } )$ . The policy $\pi _ { \theta }$ , instantiated by an LLM with parameters $\theta$ , generates the subsequent action $a _ { t + 1 }$ via policy mapping $\pi _ { \theta } : p _ { t } \mapsto a _ { t + 1 }$ .

In this work, we investigate efficient knowledge utilization strategies to empower future task decision-making performance. Focusing on multi-task settings, we consider an agent operating in a multi-task environment with dataset $\mathcal { D } = \{ \mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { n } \}$ , where tasks are split into a training set $\mathcal { D } _ { \mathrm { t r a i n } }$ and a testing set $\mathcal { D } _ { \mathrm { t e s t } }$ . The agent acquires experiential knowledge through interactions with tasks in $\mathcal { D } _ { \mathrm { t r a i n } }$ and is then required to leverage this knowledge when solving unseen tasks in $\mathcal { D } _ { \mathrm { t e s t } }$ . The core challenge lies in enabling efficient knowledge transfer when encountering new tasks $\mathcal { D } _ { \mathrm { n e w } } \in \mathcal { D } _ { \mathrm { t e s t } }$ .

# IV. HIERARCHICAL HINDSIGHT REFLECTION

To tackle the challenge of efficient knowledge transfer in multi-task scenarios, we begin by proposing a hierarchical memory architecture that organizes knowledge into highlevel and low-level components. To construct and refine these memories, we introduce $H ^ { 2 } R$ , which distills task experiences into reusable memory units at different levels of granularity. Leveraging these memories, the agent can then plan and act in a hierarchical manner, where a high-level Planner decomposes tasks into subgoals and a low-level Executor carries out these subgoals through atomic actions. This decoupling enables the agent to retrieve and reuse task-relevant knowledge at the appropriate granularity, thereby improving decision making in multi-task settings. Fig. 1 provides an overview of the core reflection workflow underlying the proposed architecture.

# A. The Hierarchical Memory Architecture

Our hierarchical memory architecture consists of a highlevel memory component $\mathcal { M } _ { \mathrm { h i g h } }$ that captures abstract task structures and a low-level memory component $\mathcal { M } _ { \mathrm { l o w } }$ that encodes detailed interaction mechanics for executing subgoals. This architectural decoupling enables each memory unit in the low-level memory component $\mathcal { M } _ { \mathrm { l o w } }$ to specialize in its own atomic subgoal, thereby mitigating interference from irrelevant knowledge inherent to naive memory designs.

Specifically, both memory components consist of structured memory units that support efficient retrieval of relevant knowledge. As shown in Fig. 1, each high-level memory unit in $\mathcal { M } _ { \mathrm { h i g h } }$ consists of the task description $\mathcal { X }$ , the sequence of realized subgoals during execution $\mathcal { G }$ , and the planning insights $\mathcal { T } _ { \mathrm { p l a n } }$ . Similarly, each low-level memory unit in $\mathcal { M } _ { \mathrm { l o w } }$ contains a specific subgoal $g$ , the corresponding detailed interaction trajectory $\tau$ , and the fine-grained execution insights $\boldsymbol { \mathcal { T } } _ { \mathrm { l o w } }$ . Unlike RAG systems, these memory units are derived from agent–environment interactions and hindsight reflection

![](images/a72d020a5dbbbb3b6c5bbc9a15269156cac469f4eb2cc34bddc3c11d2eda7291.jpg)  
Fig. 1. Overview of Hierarchical Hindsight Reflection $( H ^ { 2 } R )$ framework, which consists of four key processes: (1) Subgoal Inference, which decomposes tasks into achieved subgoals given tasks and corresponding task trajectories; (2) Subtrajectory Inference, which segments trajectories by subgoals to extract subtrajectory sequences; (3) Insight Extraction, performed at both high-level (from tasks, subgoals, and trajectories) and low-level (from individual subgoals and their trajectories) to derive reusable and beneficial rules; and (4) Memory Organization, where relevant insights are attached to corresponding memory units. This architecture enables efficient knowledge transfer through level-specific retrieval mechanisms that effectively decouple high-level planning from low-level execution in multi-task scenarios.

processes, rather than being provided a priori. When confronting tasks with analogous inner structure, relevant memory units, which encode experiential patterns along with reflective insights, will be activated to facilitate cross-task knowledge transfer and enhance decision-making performance.

# B. The Construction of Memory Units

To gather experiences for knowledge extraction and further memorial retrieval, we first perform tasks from the training set $\mathcal { D } _ { \mathrm { t r a i n } }$ . During task execution, $H ^ { 2 } R$ employs the Planner to generate subgoals and the Executor to carry out these subgoals through atomic actions. However, the agent, particularly its high-level Planner, has limited knowledge of the underlying task inner structure and may therefore generate inappropriate subgoals that contribute little to or even hinder task progress. In such cases, even if the Executor correctly accomplishes the given subgoal, the overall task may still fail to proceed. To address this problem, during training (or collecting experiences), the high-level Planner is constrained to output the current task directly instead of generating any subgoals. To improve the likelihood of success in the next attempt of the same task, Reflexion [7] is employed to analyze the failed trajectories, following the experience-gathering strategy of Expel [4].

Based on the collected trajectories, $H ^ { 2 } R$ analyzes task executions to extract and refine structured memory units in the high- and low-level components, capturing abstract strategies for task planning and detailed patterns for subgoal execution. Specifically, the process performs high-level reflection to construct high-level memory units and low-level reflection to create low-level memory units, with each reflection capturing knowledge at its respective level of granularity.

1) High-level Reflection: This process is designed to extract high-level memory units through reflective processing. Given a set of trajectories, we generate high-level memory units, each consisting of three components: the task description $\mathcal { X }$ , the sequence of realized subgoals during execution $\mathcal { G }$ , and the planning insights $\mathcal { T } _ { \mathrm { p l a n } }$ . The high-level reflection process consists of two steps: subgoal inference and insight extraction.

Subgoal Inference. Subgoal inference is introduced to extract the sequence of subgoals realized from a given task and the interaction trajectory. The intuition behind this process is that, in order to generate useful knowledge about subgoal planning, we must first assess the quality of the subgoals proposed by the Planner. However, a single interaction trajectory does not directly reveal the corresponding subgoal sequence, nor does it guarantee that the Executor has accurately executed these subgoals. Therefore, we infer the subgoal sequence through a hindsight reflection process and assume that the Executor successfully executes them, given that the inferred subgoals are grounded in the actions the agent actually performs. Formally, for a given task $\mathcal { X } ^ { i }$ and the corresponding interaction trajectory $\tau ^ { i }$ , reflection is conducted to infer subgoal sequence by prompting an LLM:

$$
\mathcal {G} ^ {i} \leftarrow \mathcal {F} _ {\text {s u b g o a l}} \left(\mathcal {X} ^ {i}, \tau^ {i}\right), \tag {1}
$$

with $\mathcal { G } ^ { i } \ : = \ : \{ g _ { 1 } ^ { i } , . . . , g _ { k } ^ { i } \}$ denoting the inferred sequence of subgoals.

Insight Extraction. Based on the inferred subgoals, highlevel insights are extracted and maintained in a fixed-size set $\mathcal { T } _ { h i g h }$ , using the mechanism proposed in ExpeL [4]. To update the set, contrastive reflection is applied to analyze strategies that lead to success while identifying potential causes of failure. This reflection process is implemented by prompt-

![](images/17d31c2b3228ab3751b4da08e5e191238a711687bacacaab3b6d61ceda5b1954.jpg)  
Fig. 2. Overview of utilization of memory components. The system comprises three core components: (1) Memory Module featuring two specialized components: (a) the High-Level Memory Component containing memory units $( m _ { i } ^ { \mathrm { { h i g h } } } )$ that store task description, subgoal sequence, and planning insights and (b) the Low-Level Memory Component containing memory units $( m _ { i } ^ { \mathrm { l o w } } )$ that store subgoal description, execution trajectory, and execution insights. For any given task, relevant memory units from both components are retrieved to inform decision making. (2) Planner that decomposes tasks into subgoals using task descriptions, planning history, current trajectories, and retrieved high-level memory, outputting structured subgoals like “shot1 contains cocktail2”. (3) Executor that translates subgoals into actionable steps using task context, current subgoals, ongoing trajectories, and retrieved low-level memory, generating action (e.g., “left grasp shot1”) or termination signals.

ing an LLM to perform four types of operations on $\mathcal { T } _ { h i g h }$ : add (introduce a new insight), modify (refine an existing insight), upvote (increase the importance of an insight), and downvote (decrease the importance of an insight). Formally, this contrastive reflection and insight updating process can be represented as a function $\mathcal { F } _ { h i g h }$ , which takes the current task $\mathcal { X } ^ { i }$ , its successful trajectory $\tau _ { + } ^ { i }$ and failed trajectory $\boldsymbol { \tau } _ { - } ^ { i }$ with subgoal sequences $\mathcal { G } _ { + } ^ { i }$ and $\mathcal { G } _ { - } ^ { i }$ , and the existing set of insights $\mathcal { T } _ { h i g h }$ as input, and outputs the updated set of highlevel insights:

$$
\mathcal {I} _ {h i g h} \leftarrow \mathcal {F} _ {h i g h} \left(\mathcal {X} ^ {i}, \tau_ {+} ^ {i}, \tau_ {-} ^ {i}, \mathcal {G} _ {+} ^ {i}, \mathcal {G} _ {-} ^ {i}, \mathcal {I} _ {h i g h}\right). \tag {2}
$$

Once the high-level reflection process is finished and all insights have been generated, a memory unit is constructed for each task, which is formed by combining three key components: the task description itself $\mathcal { X } ^ { i }$ , its corresponding successful subgoal sequence $\mathcal { G } _ { + } ^ { i }$ , and the relevant insights $\mathcal { T } _ { h i g h } ^ { i }$ . An LLM-based grounding function $F _ { g r o u n d }$ selects highthe task-relevant high-level insights $\mathcal { T } _ { h i g h } ^ { i }$ from the full set $\mathcal { T } _ { h i g h }$ by evaluating the relevance of each insight in $\mathcal { T } _ { h i g h }$ to the current task $\mathcal { X } ^ { i }$ . The resulting high-level memory unit $\{ \mathcal { X } ^ { i } , \mathcal { G } _ { + } ^ { i } , \mathcal { T } _ { h i g h } ^ { i } \}$ thus encapsulates the essential abstractions and high-level insights derived from raw trajectories. These abstractions form the foundation for subsequent low-level reflection.

2) Low-level Reflection: This process is responsible for extracting low-level memory units, which capture fine-grained execution details and insights about action-level patterns. To guarantee that each low-level memory unit captures information specific to a single subgoal, the sub-trajectory corresponding to each subgoal is first extracted and subsequently analyzed to derive low-level insights.

To ensure the reliability of the low-level memory units, only subgoals inferred from successful trajectories are utilized.

Formally, given a task $\mathcal { X } ^ { i }$ , a successful trajectory $\tau _ { + } ^ { i }$ and its corresponding sequence of implemented subgoals $\mathcal { G } _ { + } ^ { i }$ , an LLM extracts the associated sub-trajectories as follows:

$$
\mathcal {T} \leftarrow \mathcal {F} _ {\text {t r a j e c t o r y}} \left(\mathcal {X} ^ {i}, \tau_ {+} ^ {i}, \mathcal {G} _ {+} ^ {i}\right), \tag {3}
$$

with $\begin{array} { r c l } { { \mathcal T } } & { { = } } & { { \{ \tau _ { + , 1 } ^ { i } , . . . , \tau _ { + , k } ^ { i } \} } } \end{array}$ denoting the extracted subtrajectories corresponding to the given subgoals. Reflection is then performed based on each subgoal $g ^ { i }$ , its corresponding trajectory $\tau _ { + } ^ { i }$ , and a failed trajectory $\boldsymbol { \tau } _ { - } ^ { i }$ of the same task. As in high-level reflection, the insight extraction mechanism is also employed and formulated for low-level reflection as follows:

$$
\mathcal {I} _ {l o w} \leftarrow \mathcal {F} _ {l o w} \left(g ^ {i}, \tau_ {+} ^ {i}, \tau_ {-} ^ {i}, \mathcal {I} _ {l o w}\right). \tag {4}
$$

Then, in a manner analogous to the construction of highlevel memory units, we retrieve relevant insights for a specific subgoal $g ^ { i }$ using an LLM and formulate them into a low-level memory unit $\{ g ^ { i } , \tau _ { + } ^ { i } , \mathcal { T } _ { l o w } ^ { i } \}$ . Since the content of low-level reflection is grounded in that of high-level reflection, we refer to our approach as Hierarchical Hindsight Reflection $( H ^ { 2 } R )$ . The overall workflow of $H ^ { 2 } R$ is illustrated in Algorithm 1.

# C. The Utilization of Memory Units

After completing the overall reflection process, the extracted high-level and low-level memory units can be applied to new tasks. Specifically, as shown in Fig. 2, the Planner relies on high-level memories, while the Executor draws exclusively on low-level memories. Rooted in the principle of hierarchical task decomposition, the Planner formally maps natural language task specifications into structured intermediate subgoals and enhances its subgoal planning for the current task $\mathcal { X }$ by utilizing relevant memories. To retrieve such memories, vector embeddings of the current task description and stored task descriptions in the high-level memory component are computed using a pretrained sentence encoder e. By measuring

# Algorithm 1 Hierarchical Hindsight Reflection

Input: Collected trajectories $\tau$ , subgoal inference module $\mathcal { F } _ { s u b g o a l }$ , high-level reflection module $\mathcal { F } _ { h i g h }$ , sub-trajectory partition module $\mathcal { F } _ { t r a j e c t o r y }$ , low-level reflection module $\mathcal { F } _ { l o w }$   
1: Initialize the high-level memory component $\mathcal { M } _ { \mathrm { h i g h } }$ , the low-level memory component $\mathcal { M } _ { \mathrm { l o w } }$ , the high-level insight set $\mathcal { T } _ { h i g h }$ and the low-level insight set $\mathcal { T } _ { l o w }$ .   
2: for Each pair $\mathcal { X } ^ { i } , \tau _ { + } ^ { i } , \tau _ { - } ^ { i } \in \mathcal { T }$ do   
3: # extract subgoal sequences of $\tau _ { + } ^ { i } , \tau _ { - } ^ { i }$   
4: $\mathcal { G } _ { + } ^ { i }  \mathcal { F } _ { s u b g o a l } ( \mathcal { X } ^ { i } , \tau _ { + } ^ { i } )$   
5: $\mathcal { G } _ { - } ^ { i }  \mathcal { F } _ { s u b g o a l } ( \mathcal { X } ^ { i } , \tau _ { - } ^ { i } )$   
6: # update high-level insights about planning   
7: $\mathcal { F } _ { h i g h } ( \mathcal { X } ^ { i } , \tau _ { + } ^ { i } , \tau _ { - } ^ { i } , \mathcal { G } _ { + } ^ { i } , \mathcal { G } _ { - } ^ { i } , \mathcal { T } _ { h i g h } )$   
8: add $\{ \mathcal { X } ^ { i } , \mathcal { G } _ { + } ^ { i } , \emptyset \}$ to $\mathcal { M } _ { \mathrm { h i g h } }$   
9: # partition the positive trajectory into sub-trajectories   
10: $\mathcal { T } _ { s u b } ^ { i }  \mathcal { F } _ { t r a j e c t o r y } ( \tau _ { + } ^ { i } , \mathcal { G } _ { + } ^ { i } )$   
11: for Each subgoal $g _ { j } ^ { i } \in \mathcal { G } _ { + } ^ { i }$ do   
12: # update low-level insights about execution   
13: $\mathcal { F } _ { l o w } ( g _ { j } ^ { i } , \tau _ { + } , \tau _ { - } , \mathcal { T } _ { l o w } )$   
14: add $\{ g _ { j } ^ { i } , \tau _ { + } ^ { i } , \emptyset \}$ to $\mathcal { M } _ { \mathrm { l o w } }$   
15: end for   
16: end for   
17: # attach relevant insights to memory units   
18: for Each high-level memory unit $m _ { \mathrm { h i g h } } ^ { i }$ in $\mathcal { M } _ { \mathrm { h i g h } }$ do   
19: $\mathcal { T } _ { h i g h } ^ { i }  F _ { g r o u n d } ( m _ { \mathrm { h i g h } } ^ { i } , \mathcal { T } _ { h i g h } )$   
20: replace $m _ { \mathrm { h i g h } } ^ { i }$ by $\{ \mathcal { X } ^ { i } , \mathcal { G } _ { + } ^ { i } , \mathcal { T } _ { h i g h } ^ { i } \}$   
21: end for   
22: for Each high-level memory unit $m _ { \mathrm { l o w } } ^ { i }$ in $\mathcal { M } _ { \mathrm { l o w } }$ do   
23: $\mathcal { T } _ { l o w } ^ { i }  F _ { g r o u n d } ( m _ { \mathrm { l o w } } ^ { i } , \mathcal { T } _ { l o w } )$   
24: replace $m _ { \mathrm { l o w } } ^ { i }$ by $\{ g _ { i } , \tau _ { + } ^ { i } , \mathcal { T } _ { l o w } ^ { i } \}$   
25: end for   
26: return $\mathcal { M } _ { \mathrm { h i g h } } , \mathcal { M } _ { \mathrm { l o w } }$

cosine similarity between both embeddings, we retrieve the top- $k$ most relevant memory units:

$$
\mathcal {M} _ {\text {h i g h}} ^ {\text {r e l e v a n t}} = \underset {m _ {h i g h} ^ {i} \in \mathcal {M} _ {h i g h}} {\operatorname {t o p -} k} [ \operatorname {s i m} (\mathcal {X}, \mathcal {X} ^ {i}) ]. \tag {5}
$$

Functioning as the action grounding module, the Executor grounds the textual subgoal $g$ from the Planner into executable atomic action selected from a predefined set ${ \mathcal { A } } =$ $\{ a _ { 1 } , \dotsc , a _ { K } \} \cup \{ a _ { + } , a _ { - } \}$ , where $a _ { 1 }$ to $a _ { K }$ are domain-specific atomic actions, $a _ { + }$ is a primitive indicating subgoal completion and $a _ { - }$ signals that the subgoal is invalid. When the Executor determines that the current subgoal $g$ has been completed or is unachievable, it outputs $a _ { + }$ or $a _ { - }$ to trigger the Planner to replan a new subgoal. Similarly, the top- $k$ most relevant lowlevel memory units are retrieved by computing the semantic similarity between the current subgoal description and the subgoal descriptions stored within these units:

$$
\mathcal {M} _ {\text {l o w}} ^ {\text {r e l e v a n t}} = \underset {m _ {l o w} ^ {i} \in \mathcal {M} _ {l o w}} {\operatorname {t o p} - k} [ \operatorname {s i m} (g, g ^ {i}) ]. \tag {6}
$$

In practice, FAISS [23] can be employed for efficient highdimensional similarity search, and retrieval performance can be further enhanced through existing techniques such as reranking [24] and rewriting [25].

By organizing memory hierarchically, $H ^ { 2 } R$ allows LLMbased agents to selectively access only the knowledge necessary for the current context, reducing interference from irrelevant experiences and enabling more robust and efficient decision making in multi-task scenarios.

# V. EXPERIMENTS

We conduct experiments to demonstrate the effectiveness of $H ^ { 2 } R$ across diverse multi-task environments. The experimental design addresses two key research questions: (1) How does hierarchical memory organization compare to existing memory architectures in multi-task scenarios? and (2) What are the individual contributions of different memory components in our framework?

# A. Experimental Setup

Experiments are conducted on AlfWorld (a textbased household environment) [26] and PDDLGame (a strategic game environment) [27], with comparisons against two representative baselines: (1) ReAct [11], the foundational reasoning-acting paradigm without memory mechanisms; and (2) ExpeL [4], which extracts insights from successful and failed trajectories. The datasets of both benchmarks are split in half for training and evaluation, with a maximum of 30 steps per episode in AlfWorld and 40 steps in PDDLGame. Six task types from ALFWorld (pick and place, pick clean then place, pick heat then place, pick cool then place, look at obj, pick two obj) and three types from PDDLGame (barman, gripper, tyreworld) are included in the datasets. All agent components, including reflection, planning, and execution, are implemented using Qwen3-235B-A22B-Instruct-2507, and semantic similarity for memory retrieval is computed using Qwen3-Embedding-0.6B. Methods are evaluated on three held-out test episodes per environment, with results averaged across three independent runs to ensure statistical reliability.

TABLE I   
COMPARISON RESULTS   

<table><tr><td rowspan="2">Algorithms</td><td colspan="2">Success Rate on Benchmarks(%)</td></tr><tr><td>AlfWorld</td><td>PDDLGame</td></tr><tr><td>ReAct</td><td>46.3</td><td>66.7</td></tr><tr><td>Expel</td><td>72.4</td><td>72.2</td></tr><tr><td>H2R</td><td>75.9</td><td>80.5</td></tr></table>

# B. Comparison Results

Table 1 summarizes the performance comparison across both benchmark environments. As can be seen, $H ^ { 2 } R$ outperforms all baselines, with success rates of $7 5 . 9 \%$ in AlfWorld and $8 0 . 5 \%$ in PDDLGame, representing relative improvements of $3 . 5 \%$ and $8 . 3 \%$ over the baseline ExpeL. The performance

TABLE II ABLATION STUDIES   

<table><tr><td>Algorithms</td><td>Success Rate (%)</td></tr><tr><td>H2R</td><td>80.5</td></tr><tr><td>H2R w/o high-level memories</td><td>52.8</td></tr><tr><td>H2R w/o low-level memories</td><td>61.1</td></tr></table>

gains validate our core hypothesis that hierarchical memory organization enables more effective knowledge transfer in multitask settings. Notably, the improvements are most pronounced in PDDLGame, which involves more complex hierarchical planning requirements, demonstrating the particular strength of our approach in complex decision-making scenarios. These observations highlight the fundamental advantage of decoupling high-level planning knowledge from low-level execution patterns, enabling more precise and interference-free knowledge transfer across diverse multi-task environments.

# C. Ablation Studies

To evaluate the individual contribution of each component in our hierarchical framework, we conduct ablation experiments by examining the contribution of different levels in our $H ^ { 2 } R$ mechanism by selectively removing high-level or low-level memory units. When eliminating high-level memory units, the system cannot extract task-level insights and subgoal sequences, forcing it to operate without strategic planning knowledge. This results in performance degradation of $2 7 . 7 \%$ in PDDLGame. Conversely, removing low-level memories prevents the utilization of execution insights and subgoalspecific patterns, leading to $1 9 . 4 \%$ performance drops. These results demonstrate that both reflection levels are essential for the comprehensive knowledge extraction.

# VI. CONCLUSION

In this work, we propose a novel hierarchical memory architecture that decouples high-level planning memory from low-level execution memory, enabling fine-grained knowledge transfer in multi-task scenarios. By distilling reusable and hierarchical knowledge from past agent-environment interactions and performing retrieval separately at each memory level, our framework selectively reuses specialized knowledge relevant to the current context. Experimental results across AlfWorld and PDDLGame demonstrate that our framework improves generalization and decision-making performance, outperforming strong baselines such as ReAct and ExpeL. Future work will extend $H ^ { 2 } R$ to more complex and dynamic environments, while supporting multi-agent scenarios to facilitate collaborative decision making and knowledge sharing.

# REFERENCES

[3] Anirudh Chari, Suraj Reddy, Aditya Tiwari, Richard Lian, and Brian Zhou. Mindstores: Memory-informed neural decision synthesis for task-oriented reinforcement in embodied systems. arXiv preprint arXiv:2501.19318, 2025.

[4] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642, 2024.

[5] Yao Fu, Dong-Ki Kim, Jaekyeom Kim, Sungryull Sohn, Lajanugen Logeswaran, Kyunghoon Bae, and Honglak Lee. Autoguide: Automated generation and selection of context-aware guidelines for large language model agents. Advances in Neural Information Processing Systems, 37:119919–119948, 2024.

[6] Minghao Chen, Yihang Li, Yanting Yang, Shiyu Yu, Binbin Lin, and Xiaofei He. Automanual: Constructing instruction manuals by llm agents via interactive environmental learning. Advances in Neural Information Processing Systems, 37:589–631, 2024.

[7] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634– 8652, 2023.

[8] Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Jiliang Tang, Himabindu Lakkaraju, and Zhen Xiang. How memory management impacts llm agents: An empirical study of experience-following behavior. arXiv preprint arXiv:2505.16067, 2025.

[9] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.

[10] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[11] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023. URL https://arxiv. org/abs/2210.03629, 2023.

[12] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yixin Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. Retrievalaugmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2(1), 2023.

[13] Yeseung Kim, Dohyun Kim, Jieun Choi, Jisang Park, Nayoung Oh, and Daehyung Park. A survey on integration of large language models with intelligent robots. Intelligent Service Robotics, 17(5):1091–1107, 2024.

[14] Wenxuan Wang, Zizhan Ma, Zheng Wang, Chenghan Wu, Wenting Chen, Xiang Li, and Yixuan Yuan. A survey of llm-based agents in medicine: How far are we from baymax? arXiv preprint arXiv:2502.11211, 2025.

[15] Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M Mulvey, H Vincent Poor, Qingsong Wen, and Stefan Zohren. A survey of large language models for financial applications: Progress, prospects and challenges. arXiv preprint arXiv:2406.11903, 2024.

[16] Sihao Hu, Tiansheng Huang, Gaowen Liu, Ramana Rao Kompella, Fatih Ilhan, Selim Furkan Tekin, Yichang Xu, Zachary Yahn, and Ling Liu. A survey on large language model-based game agents. arXiv preprint arXiv:2404.02039, 2024.

[17] Nitin Liladhar Rane, Abhijeet Tawde, Saurabh P Choudhary, and Jayesh Rane. Contribution and performance of chatgpt and other large language models (llm) for scientific and research advancements: a double-edged sword. International Research Journal of Modernization in Engineering Technology and Science, 5(10):875–899, 2023.

[18] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An openended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

[19] Tomoyuki Kagaya, Thong Jing Yuan, Yuxuan Lou, Jayashree Karlekar, Sugiri Pranata, Akira Kinose, Koki Oguri, Felix Wick, and Yang You. Rap: Retrieval-augmented planning with contextual memory for multimodal llm agents. arXiv preprint arXiv:2402.03610, 2024.

[20] Kolby Nottingham, Bodhisattwa Prasad Majumder, Bhavana Dalvi Mishra, Sameer Singh, Peter Clark, and Roy Fox. Skill set optimization: Reinforcing language model behavior via transferable skills. arXiv preprint arXiv:2402.03244, 2024.

[1] Rich Caruana. Multitask learning. Machine learning, 28(1):41–75, 1997. [2] Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, and Enhong Chen. Understanding the planning of llm agents: A survey. arXiv preprint arXiv:2402.02716, 2024.

[21] Zhiyuan Sun, Haochen Shi, Marc-Alexandre Cotˆ e, Glen Berseth, Xingdi ´ Yuan, and Bang Liu. Enhancing agent learning through world dynamics modeling. arXiv preprint arXiv:2407.17695, 2024.   
[22] Richard S Sutton, Andrew G Barto, et al. Reinforcement learning. Journal of Cognitive Neuroscience, 11(1):126–134, 1999.   
[23] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazare, Maria Lomeli, Lucas Hosseini, and ´ Herve J ´ egou. The faiss library. ´ arXiv preprint arXiv:2401.08281, 2024.   
[24] Michael Glass, Gaetano Rossiello, Md Faisal Mahbub Chowdhury, Ankita Rajaram Naik, Pengshan Cai, and Alfio Gliozzo. Re2g: Retrieve, rerank, generate. arXiv preprint arXiv:2207.06300, 2022.   
[25] Jie Liu and Barzan Mozafari. Query rewriting via large language models. arXiv preprint arXiv:2403.09060, 2024.   
[26] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk, ´ Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768, 2020.   
[27] Ma Chang, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. Agentboard: An analytical evaluation board of multi-turn llm agents. Advances in neural information processing systems, 37:74325–74362, 2024.
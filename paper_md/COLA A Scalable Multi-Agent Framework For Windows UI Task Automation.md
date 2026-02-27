# COLA: A SCALABLE MULTI-AGENT FRAMEWORK FORWINDOWS UI TASK AUTOMATION

A PREPRINT

Di Zhao1, Longhui $\mathbf { M a } ^ { 1 }$ , Siwei Wang2, Miao Wang2, Zhao Lv2 1National University of Defense Technology, 2Academy of Military Sciences zhaodi@nudt.edu.cn

November 2, 2024

# ABSTRACT

With the rapid advancements in Large Language Models (LLMs), an increasing number of studies have leveraged LLMs as the cognitive core of agents to address complex task decision-making challenges. Specially, recent research has demonstrated the potential of LLM-based agents on automating Windows GUI operations. However, existing methodologies exhibit two critical challenges: (1) static agent architectures fail to dynamically adapt to the heterogeneous requirements of OS-level tasks, leading to inadequate scenario generalization;(2) the agent workflows lack fault tolerance mechanism, necessitating complete process re-execution for UI agent decision error. To address these limitations, we introduce $C O L A$ , a collaborative multi-agent framework for automating Windows UI operations. In this framework, a scenario-aware agent Task Scheduler decomposes task requirements into atomic capability units, dynamically selects the optimal agent from a decision agent pool, effectively responds to the capability requirements of diverse scenarios. The decision agent pool supports plug-and-play expansion for enhanced flexibility. In addition, we design a memory unit equipped to all agents for their self-evolution. Furthermore, we develop an interactive backtracking mechanism that enables human to intervene to trigger state rollbacks for non-destructive process repair. Our experimental results on the GAIA benchmark demonstrates that the COLA framework achieves state-of-the-art performance with an average score of $3 1 . 8 9 \%$ , significantly outperforming baseline approaches without web API integration. Ablation studies further validate the individual contributions of our dynamic scheduling. The code is available at https://github.com/Alokia/COLA-demo.

# 1 Introduction

In recent years, technologies based on Large Language Models (LLMs) have advanced rapidly, demonstrating significant potential in language dialogue and problem-solving [Wang et al., 2024a, Guo et al., 2024]. More complex multi-modal models (MLLMs), such as GPT-4v [Achiam et al., 2023], GPT-4o, and Gemini [Team et al., 2023], introduce a visual dimension, expanding the capabilities of LLMs and demonstrating outstanding capabilities across a broader range of fields [Guo et al., 2024]. High-capacity LLMs, MLLMs often serve as the main backbone of agents tested in specialized fields, such as software development [Hong et al., 2024, Chan et al., 2024, Li et al., 2023], social simulation [Park et al., 2023, Gao et al., 2023], and gaming [Akata et al., 2024, Wang et al., 2023, Tan et al., 2024].

As a practical multi-modal application scenario, automated tasks on personal computers are emerging as a key area of technological advancement in AI system assistants [Niu et al., 2024, Wu et al., 2024a, Zhang et al., 2024a]. Users interact with computers and access information primarily through the User Interface (UI) or Graphical User Interface (GUI) of software applications. However, due to limited screen recognition, operation, and location capabilities, existing MLLMs face challenges in this scenario [Zhang et al., 2024b, Wang et al., 2024b]. To address this, existing work leverages MLLM-based agent architecture to endow MLLMs with various capabilities for perceiving and operating computer device [Song et al., 2024, Wang et al., 2024c, Nguyen et al., 2024, Roucher, 2024]. UFO [Zhang et al., 2024a] introduces a dual-agent system, utilizing an AppAgent to manage application operations and decision-making across various scenarios. Nevertheless, this approach struggles to handle more complex GAIA datasets [Mialon et al., 2024].

![](images/bbbbc6cb75d45aa5424dbb88f5a6ae11bed36cdfb73b6e0d5cd68abed8502369.jpg)  
Figure 1: An illustration of the COLA multi-agent framework. In the first step, Planner takes request $q$ from user and decomposes it into a sequence of coarse-grained subtasks $( \mathcal { T } _ { c g } )$ . Task Scheduler then dynamically selects optimal decision agents through scenario-aware matching. Selected Decision Agents subsequently perform hierarchical task refinement, utilizing their domain-specific expertise to decompose assigned subtasks into fine-grained subtasks $( \mathcal { T } _ { f g } )$ giving an atomic action $O$ and an intention $I$ to execute that action. Executor executes it and obtains the environmental feedback result $R$ . Finally, the Reviewer evaluates the success of the action based on the environment $E _ { t }$ , $E _ { t + 1 }$ before and after execution, the intention $I$ and the result $R$ . The judgment $J$ is then sent back to the selected Decision Agent. This cyclic refinement continues until all subtask requirements are satisfied, with the Task Scheduler orchestrating inter-subtask transitions. Throughout the process, humans can intervene in the workflow at any time, providing guidance to correct the agent’s response.

MMAC [Song et al., 2024] develops agents for four distinct tasks: programming, screen semantic recognition, video analysis, and general knowledge. However, the system design suffers from limited scalability and lacks flexibility. Any error in the execution process necessitates a complete restart, which can significantly hinder efficiency and adaptability in practical applications.

Computer task scenarios are inherently complex, involving a range of specialized skills such as coding, information retrieval, file management, and system configuration, often necessitating the integration of information across multiple applications. Current research [Zhang et al., 2024a, Song et al., 2024, Wu et al., 2024a] typically utilizes orchestrated static agent systems to manage these tasks, but such approaches are inefficient when applied to more complex scenarios. This limitation underscores a significant gap in adaptability and scalability, which are essential for addressing increasingly sophisticated computer task environments.

In this work, we introduce COLA, a scalable and flexible multi-agent framework for Windows operating system assistants. COLA incorporates five specialized agent roles: planner, task scheduler, decision agent pool, executor, and reviewer. The decision agent pool consists of agents with domain-specific expertise, each tailored to handle specific tasks such as web browsing, file manipulation, programming, and others. We utilize the task scheduler to select the most appropriate decision agent from the pool when faced with different scenarios. The decision agent pool supports plug-and-play extensions, allowing users to design specialized agents for a particular scenario, thus eliminating the need to rewrite the framework and completing the extension of capabilities. Furthermore, to avoid the framework requiring to execute from scratch when agent responds anomalously, inspired by Swarm 1, we develop an interactive

backtracking mechanism. This mechanism allows users to revert to any previous response state of an agent, provide corrective guidance, and resume workflow execution from that state. The entire process is illustrated in Figure 1.

Our summarized contributions are as follows:

• We propose COLA, a scalable and flexible collaborative multi-agent framework that implements hierarchical task resolution through five specialized agent roles for complex computer task automation. Each agent is equipped with two types of memory unit: a short-term memory unit that updates as tasks progress, aiding the agent in understanding task developments, and a long-term memory unit that logs completed work, continuously enhancing the agent’s capabilities.   
• We formalize the decision agents as a pool of various agents with domain-specific expertise, and use a special Task Scheduler to perceive the task scenario and dynamically select the optimal decision agent from the pool. Furthermore, we develop an interactive backtracking mechanism that allows human users to intervene to trigger state rollbacks for non-destructive process repair.   
• Experimental results on the GAIA benchmark demonstrate state-of-the-art performance, validating COLA’s effectiveness in handling complex computer task automation scenarios.

# 2 Related Work

# 2.1 LLM based Agents

In recent years, LLM-based agents have been considered as a promising approach to achieving artificial general intelligence (AGI) [Wang et al., 2024a]. It significantly expands the capabilities of LLMs, empowering them to engage in planning, memorization and executing actions [Guo et al., 2024]. This enhancement allows LLMs to accomplish more complex tasks by mimicking human-like thinking processes and the ability to interact with the environment. LLM-based agents have been designed and applied in a variety of domains, including social simulation [Park et al., 2023, Gao et al., 2023], gaming [Akata et al., 2024, Tan et al., 2024, Wang et al., 2023], code generation [Hong et al., 2024, Chan et al., 2024], etc.

Inspired by human-team collaboration, multi-agent systems are receiving increasing attention. [Qian et al., 2024] presents an end-to-end framework for software development that utilizes multiple agents to collaborate on software development tasks. [Tao et al., 2025, Chan et al., 2024, Subramaniam et al., 2024] explores the potential of enhancing the quality of generated content through the use of multiple agents participating in debates. [Hong et al., 2024] presents a groundbreaking framework for encoding Standardized Operating Procedures (SOPs) into prompt sequences to enhance collaboration.

# 2.2 LLM-based UI Operation Agent

The utilization of LLM-based agent systems for navigating and controlling graphical user interfaces (GUIs) has emerged as a novel and rapidly expanding research area. [Yan et al., 2023] proposes a multi-modal agent based on GPT-4V for navigating mobile applications by directly inputting the screenshot. Mobile Agent [Wang et al., 2024d] integrates Optical Character Recognition (OCR) technology to enhance the agent’s visual understanding. Furthermore, Mobile Agent v2 [Wang et al., 2024e] improves single-agent to multi-agent for better performance on multiple tasks. It defines three types of agents: Planning Agent, Decision Agent and Reflection Agent, and all actions are given by a Decision Agent. UFO [Zhang et al., 2024a] utilizes the Python package pywinauto to inspect the UI controls and implement actions. It defines two types of agents: HostAgent and AppAgent, where all decisions are made by the AppAgent. However, these approaches rely on a single agent to make decisions for all tasks, which limits their scalability. For complex tasks, it is challenging for one agent to handle decisions across all scenarios. To address this limitation, in this paper, we propose the COLA framework, which treats Decision Agent as a scalable pool of specialized agents, each designed to handle specific tasks. The framework assigns the tasks to the most appropriate agent, enabling decision-making based on the current scenario.

# 3 The COLA Framework

In this section, we will provide a detailed overview of the COLA architecture. The operation of COLA is sequential and iterative, and its process is depicted in Figure 1. We design the memory unit to enhance the agent’s comprehension of task progress and its capacity to self-evolution based on prior experience. Additionally, we develop an interactive backtracking mechanism that enables non-destructive process repair and avoids workflow execution from scratch. The prompts for each agent are described in the Appendix B.

![](images/22fd369322c85ebff78238dc3168c60cd37db3cc73c73073a4a4357780cba5d5.jpg)  
Interactive Controls Information   
Figure 2: A visual perception example for Microsoft Edge with information provided by pywinauto. The raw screenshot, annotated screenshot and interactive controls information make up the visual perception component $P _ { t }$ .

# 3.1 Visual Perception and Interaction

Screen recognition remains challenging even for state-of-the art MLLMs. Making accurate decisions from MLLMs based solely on screenshots is particularly difficult. Therefore, we adopt the same methodology employed by UFO [Zhang et al., 2024a], utilizing the Python package pywinauto [Bim and Min-shuai, 2014] to inspect interactive controls within applications. We define the process of visual perception as $\mathcal { F }$ , which is formally represented by the following equation:

$$
P _ {t} = \mathcal {F} \left(E _ {t}\right) \tag {1}
$$

where $E _ { t }$ represents the screen state at step t. $P _ { t }$ denotes the visual perception component as illustrated in Figure 2.

In subsequent developments, agents requiring screen information for decision-making will incorporate the visual perception component as part of their prompt. This enhancement significantly improve their perception and comprehension of the current desktop environment, fostering more accurate and effective decisions across various applications.

To interact with the computer environment, we developed eight actions, as detailed in Appendix A. We design a domain mechanism for each action so that only agents registered in the domain can use the action. This design paradigm effectively manages the agent’s capabilities while minimizing the complexity associated with expanding the action space. By employing these predefined actions, the agent is able to interact with the computer system. Users can custom actions to meet their specific requirements and configure the domain in which agent can recognize and apply them.

# 3.2 Memory Unit For Self-Evolution

Due to the fact that the computer operation task involves numerous scenarios and requires multi-step sequential operations, the LLM-based agent needs to have the capacity to learn from past experiences and be able to articulate progress on current tasks. Inspired by how humans become increasing effective and efficient in operating computer, we maintain a long-term memory and a short-term memory.

Long-Term Memory The long-term memory (denoted as $L T$ ) preserves a complete record of prior task executions, facilitating the agent’s ability to learn from its past experiences. To enable the agent to access records in $L T$ , we introduce a retrieval function L: $\mathcal { Q } \times \mathbb { N }  2 ^ { L \hat { T } }$ , where $\mathcal { Q }$ represents the space of queries and N denotes the set of

positive integers. For each record, a summary is generated. These summaries are subsequently embedded to create a set of indices corresponding to the records. Given a query $q \in \mathcal { Q }$ and an integer $n \in \mathbb { N }$ , the function $\mathcal { L } ( q , n )$ embeds the query using the same embedding applied to the summaries, then computes the cosine similarity between the query’s embedding and the embedding of each record’s summary. The top- $\mathbf { \nabla } \cdot n$ records in $L T$ with the highest similarity scores are returned to the agent as part of its prompt. For convenience, we denote the top- $^ n$ records associated with query $q$ at step $t$ as $L T _ { t } ^ { n }$ .

Short-Term Memory The short-term memory (denoted as $S T$ ) retains the historical responses generated at each step of the current task, forming a sequence of operations: $S T _ { t } = \{ s t _ { 1 } , s t _ { 2 } , . . . , s t _ { t } \}$ , where $s t _ { t }$ represents the response produced by the agent at step t. However, including the entire $S T _ { t }$ in the prompt may lead to increased computational costs. To mitigate this, only the most recent $m$ responses are utilized: $S T _ { t } ^ { m } = \{ s t _ { t - m + 1 }$ , $s t _ { t - m + 2 }$ , ..., $s t _ { t } \big \}$ . Each agent is equipped with both types of memory, which are not shared among agents.

Each agent possesses an independent memory storage space. The short-term memory records responses at each step of task execution, facilitating the agent’s understanding of task progress. Upon task completion, decisions are stored in long-term memory, enabling the agent to recall past decision-making processes when encountering similar tasks in the future, thereby expanding its strategic perspective. When a query is received, the relevant long-term memory $( L T _ { t } ^ { n } )$ and short-term memory $( S T _ { t } ^ { m } )$ are integrated into the prompt to improve the agent’s decision-making capacity.

# 3.3 Hierarchical Multi-Agent Framework

Numerous studies have demonstrated that collaboration among multiple agents possessing diverse skills can enhance task performance [Hong et al., 2024, Chan et al., 2024, Wang et al., 2024e, Wu et al., 2024b]. In COLA framework, we established five types of agents: Planner, Task Scheduler, Decision Agent Pool, Executor, and Reviewer. The Decision Agent Pool, in particular, comprise a series of scalable agents, each with specialized skills, including the Application Manager, File Manager, Programmer, and Searcher, as depicted in Figure 1. The inputs and outputs of each agent are detailed as follows.

# 3.3.1 Planner

The planner plays a crucial role in managing the workflow process by breaking down user requests into subtasks, establishing an organized and methodical foundation for task execution. The planner initially generates coarse-grained subtasks, which are then further refined into fine-grained subtasks by subsequent decision agent. This hierarchical planning approach is particularly effective for handling complex and variable tasks.

We define the coarse-grained subtasks generated by planner as $\mathcal { T } _ { c g }$ . Given a user request $q$ , this process is represented by the following formula:

$$
\mathcal {T} _ {c g} = \left\{s _ {1}, s _ {2}, \dots , s _ {k} \right\} = P L \left(q, L T _ {t} ^ {n}, S T _ {t} ^ {m}\right) \tag {2}
$$

where $P L$ represents the LLM of planner, each $s _ { k } \in \mathcal { T } _ { c g }$ represents a coarse-grained subtask.

The planner’s high degree of functional decoupling from other agents affords significant flexibility, enabling it to effectively employ a variety of reasoning strategies, such as COT [Wei et al., 2022], TOT [Yao et al., 2023], or multi-agent debate [Liang et al., 2024], to improve response performance.

# 3.3.2 Task Scheduler

The task scheduler is designed to identify the capabilities required for each coarse-grained subtask generated by the planner. It then assigns these subtasks to the appropriate agents based on their specialized descriptions in the decision agent pool. This process is represented by the following formula:

$$
\begin{array}{l} \mathcal {D} = \left\{\left(\text {r o l e} _ {1}, r t _ {1}\right), \left(\text {r o l e} _ {2}, r t _ {2}\right), \dots , \left(\text {r o l e} _ {k}, r t _ {k}\right) \right\} \\ = T S \left(\mathcal {T} _ {c g}, D A _ {d e s c}, L T _ {t} ^ {n}, S T _ {t} ^ {m}\right) \tag {3} \\ \end{array}
$$

where $T S$ represents the LLM of task scheduler, $r o l e _ { k }$ refers to an agent in the decision agent pool, ${ r t } _ { k }$ denotes the coarse-grained subtasks assigned to $r o l e _ { k }$ , and $D A _ { d e s c }$ represents the description of the specialties of all agents in the decision agent pool.

After the assignment $\mathcal { D }$ is generated, each agent $r o l e _ { k }$ is sequentially tasked with performing its assigned subtask ${ r t } _ { k }$

# 3.3.3 Decision Agent Pool

Previous approaches relied on single or multiple fixed agents to make decisions in specific scenarios [Zhang et al., 2024a, Song et al., 2024, Wang et al., 2024e], but this static agent architecture fails to dynamically adapt to the heterogeneous

demands of operating system tasks, struggling to manage the complexity and variety of computer tasks. In contrast, drawing inspiration from the Mixture of Experts (MoE) model [Jacobs et al., 1991], we formalize the decision agent as a scalable pool comprising agents with specialized capabilities, each tailored to distinct scenarios. Each agent’s expertise is represented by a natural language description, denoted as $D A _ { d e s c }$ . When the task scheduler assigns subtasks, the selected agent $r o l e _ { k }$ from the pool completes the assigned subtasks $r t _ { k }$ sequentially, based on the visual perception component $P _ { t }$ discussed in Section 3.1. This process is represented by the following formula:

$$
(I, O, \mathcal {T} _ {f g}) = D A _ {\text {r o l e} _ {k}} \left(q, r t _ {k}, P _ {t}, J, L T _ {t} ^ {n}, S T _ {t} ^ {m}\right) \tag {4}
$$

where the $D A _ { r o l e _ { k } }$ represents the selected agent in decision agent pool; $O$ represents the action to be performed, and $I$ is the intention to perform the action; $\mathcal { T } _ { f g }$ is a fine-grained list of subtasks that is regenerated for each execution to adjust the planning immediately; $J$ is the judgment given by reviewer.

This design paradigm, which dynamically assigns tasks through the task scheduler, enables plug-and-play scaling of the decision agent pool. It allows users to customize both specialized agents and their associated actions using the domain mechanism described in Section 3.1.

Figure 3 illustrates the skill requirements of the GAIA benchmark. In this study, we implemented four ad-hoc agents to meet these requirements, each described in natural language as follows:

• Application Manager: Can open applications such as browsers, explorers, chat software, etc.   
• File Manager: Can open, create, and delete files, such as txt, xlsx, pdf, png, mp4 and other documents.   
• Searcher: Can use an opened browser to search for information, open web pages, etc. Can also do everything related to web pages, such as playing videos in web pages, opening files, reading documents in web pages, and so on.   
• Programmer: Possesses logical reasoning and analytical skills. Can reason to arrive at an answer to a question or write Python code to get the result.

# 3.3.4 Executor

The executor is responsible for directly interacting with the computer environment. Since its role is limited to executing actions, there is no need for a memory unit. This process is represented by the following formula:

$$
\left(E _ {t + 1}, R\right) = \operatorname {E x e c} (O, E _ {t}) \tag {5}
$$

where the Exec denotes the function by which the executor performs the action, $E _ { t + 1 }$ represents the environment state after executing action $O$ , and $R$ is the result of the action, which may be null, as in the case of actions like a mouse click.

Due to the potential dangers of operations within the computer environment, such as file deletion, we have decoupled the direct interaction functionalities of decision agents from the environment into a separate executor component. This decoupling facilitates subsequent research on restricting sensitive operations without necessitating modifications to other system components.

# 3.3.5 Reviewer

Due to the hallucination problem associated with LLMs [Liu et al., 2023, Gunjal et al., 2023, Cui et al., 2023], agents may generate unintended actions in certain scenarios. To address this, we design the reviewer to assess the validity of an action based on changes in the operating environment before $( E _ { t } )$ and after $( E _ { t + 1 } )$ the action is performed, as well as the intent $( I )$ behind the action, generated by the decision agent. This process is formalized as follows:

$$
J = \operatorname {R e} \left(E _ {t}, E _ {t + 1}, I, O, R\right) \tag {6}
$$

where Re represents the LLM of reviewer.

The reviewer evaluates the actions of decision agents in task-oriented systems, assessing their suitability based on outcomes and intentions. If an action is deemed unsuitable, the reviewer provides reasons and feedback to guide improvements. Numerous studies [Song et al., 2024, Wang et al., 2024e] have substantiated the efficacy of the evaluation-modification approach in improving the accuracy of task execution.

# 3.4 Interactive Backtracking Mechanism

To enable non-destructive process repair, we propose an interactive backtracking mechanism comprising two functions: role switching and dialog backtracking. Role switching allows users to dynamically change the current dialog agent

during the interaction, while dialog backtracking enables users to revert the agent to a previous response and re-execute the workflow from that point.

As illustrated in Figure 4, the traditional agent framework encounters improper responses during execution that requires re-executing the task from the scratch, which not only consumes time but also increases token overhead. In contrast, the interaction backtracking mechanism allows the user to roll back the workflow to the state where the improper response appeared at the beginning, and provides guidance suggestions to fix the subsequent process, reducing time and token overhead.

Additionally, we provides three modes of interaction: automatic, passive, and active, each of which alters the manner in which humans engage with the workflow:

• Automatic: In this mode, the workflow runs autonomously and human is not required. If an issue arises during execution, the entire workflow halts.   
• Passive: In this mode, the workflow operates autonomously, but if the agent encounters a problem, it requests human assistance. The human can then provide guidance to ensure proper execution.   
• Active: In this mode, the workflow pauses at each step, awaiting human input. The human can choose to skip the guidance or correct the agent’s response as needed.

# 4 Experiment

Table 1: Performance comparison between our model and multiple baseline models on the GAIA benchmark. “No Pipeline” refers to the raw GPT-4o, with no agent pipeline applied. Web APIs represents the way to browse the web, “ ” indicates that the web is accessed through an API, such as AutoGen web browser tool [Wu et al., 2024b], “ ” means navigating web pages by simulating human interaction with the browser. Each value is reported on the GAIA Leaderboard2 and represents the average exact match percentage between the predicted result and the ground truth.   

<table><tr><td>Agent Pipeline</td><td>Level 1</td><td>Level 2</td><td>Level 3</td><td>Avg.</td><td>Web APIs</td></tr><tr><td>Magenta-1 [Fourney et al., 2024]</td><td>46.24</td><td>28.30</td><td>18.37</td><td>32.23</td><td>✓</td></tr><tr><td>HF Agents [Roucher, 2024]</td><td>49.46</td><td>28.30</td><td>18.37</td><td>33.22</td><td>✓</td></tr><tr><td>Sibyl [Wang et al., 2024c]</td><td>47.31</td><td>32.70</td><td>16.33</td><td>34.55</td><td>✓</td></tr><tr><td>DynaSaur [Nguyen et al., 2024]</td><td>51.61</td><td>36.48</td><td>18.37</td><td>38.21</td><td>✓</td></tr><tr><td>No Pipeline [Nguyen et al., 2024]</td><td>13.98</td><td>8.81</td><td>2.04</td><td>9.30</td><td>X</td></tr><tr><td>FRIDAY [Wu et al., 2024a]</td><td>40.86</td><td>20.13</td><td>6.12</td><td>24.25</td><td>-</td></tr><tr><td>MMAC [Song et al., 2024]</td><td>45.16</td><td>20.75</td><td>6.12</td><td>25.91</td><td>X</td></tr><tr><td>COLA*</td><td>49.46</td><td>27.67</td><td>12.24</td><td>31.89</td><td>X</td></tr></table>

![](images/2001906b6ba943a521297f994a0a767361b8e1186be5b568fa32433a5b221e37.jpg)  
Figure 3: Number of questions covered for each skill. Each value is reported in GAIA [Mialon et al., 2024].

Benchmark We evaluate COLA using the GAIA dataset [Mialon et al., 2024], a benchmark dedicated to evaluating general AI assistants. The GAIA dataset contains 466 human-designed and annotated questions, covering basic competencies such as reasoning, multimodal comprehension, coding, and tool usage. To answer these questions, agent needs skill to write code, browse the web, manipulate files, and process video and audio data, among others. Figure 3 illustrates the skill requirements for solving GAIA tasks. The graph indicates that tasks requiring web browsing skills comprised $76 \%$ of the total tasks, suggesting that the method of web page navigation significantly influences the outcomes.

![](images/337a9b35ecc74c29cba19b05733d093c05a3b07ed77ac0ffb34843bb1242d414.jpg)  
Figure 4: A comparison between the traditional agent framework and COLA reveals key differences.

Table 2: Ablation study performance comparison results on the GAIA test set.   

<table><tr><td>Configuration</td><td>Level 1</td><td>Level 2</td><td>Level 3</td><td>Avg.</td></tr><tr><td>COLA</td><td>49.46</td><td>27.67</td><td>12.24</td><td>31.89</td></tr><tr><td>w/o decision agent pool</td><td>43.01</td><td>18.24</td><td>2.04</td><td>23.26</td></tr></table>

Baselines We compared several agent systems featured in the GAIA leaderboard, with a particular focus on those involving web browsing, as it constitutes the majority of tasks. Given the prominence of web browsing, we categorized these approaches based on their browsing methods: those that use APIs to retrieve web content include Magentic-1 [Fourney et al., 2024], Hugging Face Agents (HF Agents) [Roucher, 2024], Sibyl System v0.2 (Sibyl) [Wang et al., 2024c], and DynaSaur [Nguyen et al., 2024]; those that simulate human manipulation of the browser include MMAC v1.1 (MMAC) [Song et al., 2024]; and approaches not described in the paper include FRIDAY [Wu et al., 2024a]. Additionally, we use the raw GPT-4o performance, without any agentic framework, as a lower bound for comparison.

Settings We utilize OpenAI’s text-embedding-3-large as the embedding model for our memory unit, tasked with encoding queries and memory summaries into vectors for similarity matching. For the decision agent, since it has to make decisions directly based on the environment, the visual perception component will bring huge content to the prompt, so we set their long-term memory parameter $n$ to 2 and short-term memory parameter $m$ to 6 to mitigate token overhead. For other agents, given their fewer token requirements, we set the long-term memory parameter $n$ to 3 and the short-term memory parameter $m$ to 10 to better enhance their capabilities. We utilize GPT-4o (gpt-4o-2024-08-06) as LLM backbone for all agentic pipelines, with the maximum number of reasoning steps limited to 20. For further analysis, to save costs, we only evaluate using GPT-4o.

Implementation Details In the preliminary phase of the study, the experiment was initiated by activating the interactive mode to Active. During this stage, the agent underwent operation on the validation set, receiving continual guidance from a human supervisor to ensure accurate task completion. Subsequently, after amassing sufficient experiential data in the form of long-term memory, the mode was transitioned to Automatic for the critical assessment phase conducted on the test set, allowing assessment of the agent’s autonomous performance under test conditions.

# 4.1 Main Result

In this analysis, our proposed method, COLA, is evaluated through comparative studies with several established baseline approaches, as presented in Table 1. The data clearly demonstrates that COLA outperforms other methods in simulating human web browsing behaviors within the GAIA private test set, particularly in the more challenging Level 2 and Level 3 tasks. Significant improvements in accuracy metrics are observed, with COLA showing a notable increase over the No Pipeline approach: from $1 3 . 9 8 \%$ to $4 9 . 4 6 \%$ in Level 1 tasks, from $8 . 8 1 \%$ to $2 7 . 6 7 \%$ in Level 2 tasks, and from $2 . 0 4 \%$ to $1 2 . 2 4 \%$ in Level 3 tasks. These advancements highlight the effectiveness of the COLA method.

Furthermore, as depicted in Figure 3, the GAIA dataset reveals that a significant majority of tasks, approximately $7 6 . 1 8 \%$ , necessitate web browsing skills, thus emphasizing the importance of developing robust simulation techniques. Unlike methods that rely solely on Web APIs, COLA utilizes mouse and keyboard interactions for webpage manipulation, broadening its application scope. However, this approach necessitates multi-modal large language models (MLLMs) with advanced image understanding, in contrast to Web API-based methods. COLA performs well at Level 1, where tasks are relatively simple and involve basic web browsing. As task complexity increases to Level 2 and Level 3, which require more complex web manipulation, a more pronounced gap begins to emerge between methods of simulating human manipulation of the browser and methods of accessing web pages using APIs. This difference underscores the limitations of current MLLMs in handling continuous webpage image comprehension over extended steps, thereby indicating areas for potential future enhancements.

# 4.2 Ablation Study

We conduct ablation studies more deeply on the GAIA test set in order to investigate the contribution of the decision agent pool in the COLA framework. For comparison purposes, we design a single agent equipped with all the actions responsible for handling all task scenarios. We use the same testing approach - first providing guidance on the validation set, gaining experience, and then running it on the test set - we obtained the results shown in Table 2. The decrease in the overall average score from $3 1 . 8 9 \%$ to $2 3 . 2 6 \%$ highlights the importance of the Decision Agents pool. While the difference in Level 1 scores $4 9 . 4 6 \%$ vs. $4 3 . 0 1 \%$ ) is minimal, there is a significant gap in Level 2 $( 2 7 . 6 7 \%$ vs. $1 8 . 2 4 \%$ ) and Level 3 scores $1 2 . 2 4 \%$ vs. $2 . 0 4 \%$ ), indicating that task specialization by scenario is effective.

We compared the traditional agent framework with COLA, as shown in Figure 4. In the traditional model, when task execution deviates from expectations, the process must be restarted from the beginning. In contrast, COLA’s interactive backtracking mechanism allows for flexible state backtracking, enabling non-destructive repairs without restarting the entire process.

# 4.3 Case Study

We present real case studies to illustrate the COLA workflow process, as detailed in Appendix C. Figure 5 provides a simplified view of the workflow. Upon receiving a user request, the planner decomposes it into a coarse-grained list of subtasks and identifies the questions that need to be addressed. The task scheduler recognizes that the first subtask requires operations on the application, thus it assigns this task to the application manager. It also determines that the subsequent tasks will require the use of an already-opened browser, and therefore assigns the next three tasks to the searcher. The decision agent then executes these tasks sequentially. First, the application manager opens the Edge browser and completes its task. Next, the searcher manipulates the browser, breaking down the assigned coarse-grained subtasks into fine-grained tasks and progressively accomplishing them. Finally, the information gathered is sent back to the planner to obtain the final answer.

Figure 6 illustrates a scenario in which the interactive backtracking mechanism is employed. Initially, the planner provides an inadequate subtask plan, leading the task scheduler to misidentify the capacity requirements of the subtasks, causing the workflow to deviate from the intended path. Upon noticing the issue, a human identifies the problem with the subtask planning and switches roles to the planner. After pointing out the issue and offering guidance, the human helps steer the workflow back on track, ensuring proper execution.

# 5 Conclusion

We introduce COLA, an extensible multi-agent framework developed as an AI assistant for Windows operating systems. By decomposing complex tasks into scenario-specific subtasks and assigning a customized agent to each, COLA forms a scalable pool of Decision Agents. And a task scheduler was designed to identify the capabilities required to complete the task and assign them to the appropriate decision agent. These agents collaborate to complete intricate tasks, resulting in significant results in the GAIA data set. In addition, the interaction backtracking mechanism allows the user to

intervene in the workflow at any time and backtrack the workflow to any state, correcting the execution process of the workflow and getting more accurate results. We anticipate that this approach to scalable task decomposition and agent assignment can be extended to more complex task scenarios.

# 6 Limitations

We acknowledge that the current COLA has some limitations. Firstly, the allocation of tasks solely based on the skill descriptions of decision agents is insufficient. In scenarios where there is an overlap in skills among decision agents, it may not be possible to assign tasks to the desired agent. To address this limitation, future research could involve tracking the performance of agents in completing tasks and dynamically updating the capability descriptions of decision agents.

Secondly, the operation system’s environment is complex, and manually designing decision agents for various scenarios is labor-intensive. We hope that future studies will support the automation of constructing scenario-specific agents, such as creating expert agents automatically based on software user guides, enabling COLA to handle an expanded range of tasks.

# 7 Ethical Considerations

When agents are permitted to operate autonomously on computer systems, it is crucial to consider the security implications for system integrity. While no harmful actions were observed during our experiments, we strongly recommend conducting such tests within a controlled virtual environment. Future research should focus on developing methods to restrict the privileges of autonomous agents and block sensitive operations, thereby safeguarding overall system security.

# References

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6): 186345, 2024a. URL https://link.springer.com/article/10.1007/s11704-024-40231-1.   
Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. Large language model based multi-agents: a survey of progress and challenges. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI ’24, 2024. ISBN 978-1-956792-04-1. doi:10.24963/ijcai.2024/890. URL https://doi.org/10.24963/ijcai.2024/890.   
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023. URL https://arxiv.org/abs/2303.08774.   
Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. URL https://arxiv.org/abs/2312.11805.   
Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=VtmBAGCN7o.   
Chi-Min Chan, Weize Chen, Yusheng Su, Jianxuan Yu, Wei Xue, Shanghang Zhang, Jie Fu, and Zhiyuan Liu. Chateval: Towards better LLM-based evaluators through multi-agent debate. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=FQepisCUWu.   
Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: communicative agents for "mind" exploration of large language model society. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc. URL https://ghli.org/publication/neurips2023camel/.   
Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology, pages 1–22, 2023. URL https://dl.acm.org/doi/10.1145/3586183. 3606763.

Chen Gao, Xiaochong Lan, Zhi jie Lu, Jinzhu Mao, Jing Piao, Huandong Wang, Depeng Jin, and Yong Li. S3: Social-network simulation system with large language model-empowered agents. ArXiv, abs/2307.14984, 2023. URL https://api.semanticscholar.org/CorpusID:260202947.   
Elif Akata, Lion Schulz, Julian Coda-Forno, Seong Joon Oh, Matthias Bethge, and Eric Schulz. Playing repeated games with large language models, 2024. URL https://openreview.net/forum?id=CSpWgKo0ID.   
Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi (Jim) Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. Trans. Mach. Learn. Res., 2024, 2023. URL https://api.semanticscholar.org/CorpusID:258887849.   
Weihao Tan, Ziluo Ding, Wentao Zhang, Boyu Li, Bohan Zhou, Junpeng Yue, Haochong Xia, Jiechuan Jiang, Longtao Zheng, Xinrun Xu, Yifei Bi, Pengjie Gu, Xinrun Wang, Börje F. Karlsson, Bo An, and Zongqing Lu. Towards general computer control: A multimodal agent for red dead redemption II as a case study. In ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024. URL https://openreview.net/forum?id=pmcFzuUxsP.   
Runliang Niu, Jindong Li, Shiqi Wang, Yali Fu, Xiyu Hu, Xueyuan Leng, He Kong, Yi Chang, and Qi Wang. Screenagent: A vision language model-driven computer control agent. In Kate Larson, editor, Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pages 6433–6441. International Joint Conferences on Artificial Intelligence Organization, 8 2024. doi:10.24963/ijcai.2024/711. URL https: //doi.org/10.24963/ijcai.2024/711. Main Track.   
Zhiyong Wu, Chengcheng Han, Zichen Ding, Zhenmin Weng, Zhoumianze Liu, Shunyu Yao, Tao Yu, and Lingpeng Kong. Os-copilot: Towards generalist computer agents with self-improvement. ArXiv, abs/2402.07456, 2024a. URL https://api.semanticscholar.org/CorpusID:267626905.   
Chaoyun Zhang, Liqun Li, Shilin He, Xu Zhang, Bo Qiao, Si Qin, Minghua Ma, Yu Kang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, and Qi Zhang. Ufo: A ui-focused agent for windows os interaction. CoRR, abs/2402.07939, 2024a. URL https://doi.org/10.48550/arXiv.2402.07939.   
Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Ming-Jie Ma, Qingwei Lin, S. Rajmohan, Dongmei Zhang, and Qi Zhang. Large language model-brained gui agents: A survey. ArXiv, abs/2411.18279, 2024b. URL https://api.semanticscholar.org/CorpusID:274306375.   
Shuai Wang, Weiwen Liu, Jingxuan Chen, Weinan Gan, Xingshan Zeng, Shuai Yu, Xinlong Hao, Kun Shao, Yasheng Wang, and Ruiming Tang. Gui agents with foundation models: A comprehensive survey. ArXiv, abs/2411.04890, 2024b. URL https://api.semanticscholar.org/CorpusID:273877415.   
Zirui Song, Yaohang Li, Meng Fang, Zhenhao Chen, Zecheng Shi, Yuan Huang, and Ling Chen. Mmac-copilot: Multi-modal agent collaboration operating system copilot. ArXiv, abs/2404.18074, 2024. URL https://api. semanticscholar.org/CorpusID:269448905.   
Yulong Wang, Tianhao Shen, Lifeng Liu, and Jian Xie. Sibyl: Simple yet effective agent framework for complex real-world reasoning. ArXiv, abs/2407.10718, 2024c. URL https://api.semanticscholar.org/CorpusID: 271213801.   
Dang Nguyen, Viet Dac Lai, Seunghyun Yoon, Ryan Rossi, Handong Zhao, Ruiyi Zhang, Puneet Mathur, Nedim Lipka, Yu Wang, Trung Bui, Franck Dernoncourt, and Tianyi Zhou. Dynasaur: Large language agents beyond predefined actions. ArXiv, abs/2411.01747, 2024. URL https://api.semanticscholar.org/CorpusID:273811411.   
Aymeric Roucher. Huggingface agent. 2024. URL https://huggingface.co/learn/cookbook/agents.   
Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. GAIA: a benchmark for general AI assistants. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=fibxvahvs3.   
Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. ChatDev: Communicative agents for software development. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15174–15186, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi:10.18653/v1/2024.acl-long.810. URL https://aclanthology.org/2024.acl-long.810/.   
Mingxu Tao, Dongyan Zhao, and Yansong Feng. Chain-of-discussion: A multi-model framework for complex evidencebased question answering. In Owen Rambow, Leo Wanner, Marianna Apidianaki, Hend Al-Khalifa, Barbara Di Eugenio, and Steven Schockaert, editors, Proceedings of the 31st International Conference on Computational Linguistics, pages 11070–11085, Abu Dhabi, UAE, January 2025. Association for Computational Linguistics. URL https://aclanthology.org/2025.coling-main.734/.

Vighnesh Subramaniam, Antonio Torralba, and Shuang Li. DebateGPT: Fine-tuning large language models with multi-agent debate supervision, 2024. URL https://openreview.net/forum?id=ChNy95ovpF.   
An Yan, Zhengyuan Yang, Wanrong Zhu, Kevin Qinghong Lin, Linjie Li, Jianfeng Wang, Jianwei Yang, Yiwu Zhong, Julian J. McAuley, Jianfeng Gao, Zicheng Liu, and Lijuan Wang. Gpt-4v in wonderland: Large multimodal models for zero-shot smartphone gui navigation. ArXiv, abs/2311.07562, 2023. URL https://api.semanticscholar. org/CorpusID:265149992.   
Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. In ICLR 2024 Workshop on Large Language Model (LLM) Agents, 2024d. URL https://openreview.net/forum?id=jE6pDYCnVF.   
Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. Mobile-agent-v2: Mobile device operation assistant with effective navigation via multi-agent collaboration. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024e. URL https://openreview. net/forum?id=O0nBMRlkc8.   
HE Bim and WANG Min-shuai. Application of pywinauto in software performance test. Computer and Modernization, (8):135, 2014. URL http://www.c-a-m.org.cn/EN/abstract/abstract3451.shtml.   
Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. Autogen: Enabling next-gen LLM applications via multi-agent conversation, 2024b. URL https://openreview.net/forum?id=tEAF9LBdgu.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Proceedings of the 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA, 2022. Curran Associates Inc. ISBN 9781713871088. URL https://dl.acm.org/doi/10.5555/3600270.3602070.   
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: deliberate problem solving with large language models. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates Inc. URL https://dl.acm.org/doi/10.5555/3666122.3666639.   
Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi-agent debate. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 17889–17904, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi:10.18653/v1/2024.emnlp-main.992. URL https://aclanthology.org/2024.emnlp-main.992/.   
Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, and Geoffrey E. Hinton. Adaptive mixtures of local experts. Neural Computation, 3(1):79–87, 1991. doi:10.1162/neco.1991.3.1.79. URL https://ieeexplore.ieee.org/ abstract/document/6797059.   
Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang. Aligning large multi-modal model with robust instruction tuning. ArXiv, abs/2306.14565, 2023. URL https://api.semanticscholar.org/ CorpusID:263860779.   
Anish Gunjal, Jihan Yin, and Erhan Bas. Detecting and preventing hallucinations in large vision language models. In AAAI Conference on Artificial Intelligence, 2023. URL https://api.semanticscholar.org/CorpusID: 260887222.   
Chenhang Cui, Yiyang Zhou, Xinyu Yang, Shirley Wu, Linjun Zhang, James Zou, and Huaxiu Yao. Holistic analysis of hallucination in gpt-4v(ision): Bias and interference challenges. ArXiv, abs/2311.03287, 2023. URL https: //api.semanticscholar.org/CorpusID:265033982.   
Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Erkang Zhu, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, Peter Chang, Ricky Loynd, Robert West, Victor Dibia, Ahmed M. Awadallah, Ece Kamar, Rafah Hosn, and Saleema Amershi. Magentic-one: A generalist multi-agent system for solving complex tasks. ArXiv, abs/2411.04468, 2024. URL https://api.semanticscholar.org/ CorpusID:273877854.

# A Details Of The Actions

In this section, we introduce an exhaustive compilation of actions implemented in our framework, along with comprehensive descriptions and the specific domains to which they are allocated. Our approach has been meticulously structured to minimize redundancy and to distinctly delineate the unique functionalities of each agent within the system. To achieve this, we have systematically designed a domain for each individual action, which ensures that only agents operating within the designated domain are authorized to employ the respective action, as detailed in Table 3. This stratification not only enhances system efficiency but also facilitates seamless coordination among agents by precisely defining their operational scope.

During the operational phase, every action undergoes a transformation into a string description, coupled with its relevant parameters. This converted string is subsequently incorporated into the agent’s operational prompt, thereby enabling the agent to effectively access and implement the action through its parameters. This methodical process ensures that each agent possesses the necessary directives to execute actions with precision, tailored to the specific requirements of their domain.

Users possess the capability to tailor operations to meet their specific requirements. The essential condition involves effectively implementing the desired functionalities and establishing a domain that delineates which agents are authorized to access and employ these customized operations. This framework ensures that only authorized agents can perform the tailored actions, thereby maintaining a controlled operational environment that aligns with the users’ objectives.

Table 3: List of defined actions. Only agents in the Domain can use this action.   

<table><tr><td>Action</td><td>Description</td><td>Domain</td></tr><tr><td>click_input</td><td>Click the control with the given button and double-click if needed.</td><td>Searcher, File Manager</td></tr><tr><td>keyboard_input</td><td>Use to simulate the keyboard input.</td><td>Searcher, File Manager</td></tr><tr><td>hotkey</td><td>Use this API to simulate the keyboard shortcut keys or press a single key. It can be used to copy text, find information existing on a web page, and so on.</td><td>Searcher, File Manager, Application Manager</td></tr><tr><td>scroll</td><td>Use to scroll the control item. It typical apply to a ScrollBar type of control item when user request is to scroll the control item, or the targeted control item is not visible nor available in the control item list, but you know the control item is in the application window and you need to scroll to find it.</td><td>Searcher, File Manager</td></tr><tr><td>wait_for_loadings</td><td>Waiting for functions to load.</td><td>Searcher, File Manager, Application Manager</td></tr><tr><td>open.application</td><td>Open the application with the given name.</td><td>Application Manager</td></tr><tr><td>run_PYthon_code</td><td>Run the given Python code.</td><td>Programmer</td></tr><tr><td>read_file</td><td>Read the contents of file.</td><td>File Manager</td></tr></table>

# B Prompts

The system prompts used for agents in COLA are shown in Tables 4 to 10. Integrating the skill descriptions of all decision agents into the system prompt is crucial for planners and task schedulers. This integration facilitates the formulation of sub-tasks and their allocation by aligning with the capabilities of expert agents, leading to more contextually appropriate responses. For decision agents, actions are categorized by domain and incorporated into each agent’s system prompt. This approach strengthens the connection between agents and their respective actions, allowing each agent to function more efficiently and effectively within its designated domain. Additionally, for reviewers, combining the descriptive functionalities and parameter lists of all actions within the system prompt is essential. This comprehensive prompt enables reviewers to make more informed assessments of the actions being considered, improving the accuracy of their evaluations.

# C Case Study

![](images/498801968935c6d712d6853cb50e8789189294dc5a4fee6f8cc533636d6cb944.jpg)  
Figures 5 and 6 presents a real-world case study from the GAIA benchmark. For clarity, certain elements, such as the executor and reviewer, have been omitted from the figure.   
Figure 5: An abbreviated description of the workflow when COLA performs task "The article ‘Technology in the Dystopian Novel’ by Gorman Beauchamp begins with a block quote attributed to a novelist from the Victorian era. In what year did the borough in which this novelist was born attain city status?"

![](images/0eb865c24f26f8e02e91d7170e79179c328250f594302503a8d98000a35d66d8.jpg)  
Figure 6: An example of using role switching. The task is: “What is the dimension of the boundary of the tame twindragon rounded to two decimal places?”. While executing a workflow, Planner gives inappropriate coarse-grained subtasks, resulting in the task being assigned to an inappropriate Programmer. Human discovers this, talks to the Programmer, switches the agent to Planner, and gives guidance to change the trajectory of the workflow.

# Planner

# <Objective>

You are an AI Planner designed to efficiently operate Windows computers and proficiently handle high-level task planning and mission summaries.

<Capabilities and Skills>

1. You know how to use a computer for given tasks, such as searching using a browser, browsing for documents, etc. So you can break down a complex goal into manageable coarse-grained subtasks.   
2. You can generate a plan for a given task, including the steps to be taken, the order in which they should be executed, and the expected outcome.   
3. You know what the downstream agent is capable of, and you can always split the task into separate functions when you make a list of subtasks so that each subtask is given to a separate agent to accomplish.

{role_capabilities}

4. If you come across a request that requires logical reasoning, think of it as a whole and put that entire task on the decomposition list.

<Output Format>

You need to output a response of type json. json contains parameters and its interpretation as follows:

{

"branch": "typing.Optional[cola.fundamental.base_response_format.BranchType]. The following are the values that can be set for this parameter and their explanations: Set to ‘Continue‘ when normal response processing of the task is underway, so that the next action can be performed. set to ‘Interrupt‘ when you really don’t know what to do with a task. This is a dangerous operation, unless you have a good reason to refuse to continue the mission.",

"problem": "<class ’str’>. The problems you encountered. When the task is executed normally, this parameter is set to an empty string ‘”‘.",

"message": "<class ’str’>. The information you want to tell the next agent. If there is no information that needs to be specified, it is set to empty string ‘”‘.",

"summary": "<class ’str’>. Summarize the conversation. Include: Did the answers you gave in the previous step meet the requirements of the task? What have you done now? Why are you doing this?",

"sub_tasks": "typing.List[str]. A list of subtasks generated by the yourself. Each subtask is a string When you can not complete the task, set ‘sub_tasks‘ to empty list []",

"question": "<class ’str’>. The questions the task is expected to answer and the format of the answers. If the task does not need to return a reply, this parameter is set to an empty string ”. For example: Task: ’Open the browser and search for the book <Pride and Prejudice>, tell me the author of the book.’ Question: ’What is the author of the book? Another example: Task: ’Open the browser and search for the book <Pride and Prejudice>’ Question: ”"

} “‘

<Notice>

1. When splitting a complex task into subtask steps, please consider the ability of the downstream Agents and keep the granularity of the subtasks at a level that can be accomplished by a single Agent.   
For example, if a subtask requires two Agents to complete, it needs to be split into two finer-grained subtasks.   
2. You can’t generate an empty task breakdown list, if you can’t do it, just put the whole task in the list.   
3. You only need to give rough steps, not specific implementation arrangements. For example:   
Give Task: "Tell me the weather today"   
Your should give a rough plan: "1. Open the browser. 2. Search for the weather today."

Table 4: The system prompt for the planner. role_capabilities denotes the skill descriptions of all agents in the decision agent pool.

# Task Scheduler

<Objective>

You are a Task Scheduler specializing in assigning a set of tasks to the appropriate Agent.

You are very good at high-level task scheduling and can assign different types of tasks to the right Agent based on the downstream Agent’s capabilities.

<Capabilities and Skills>

1. You know all the roles that specialize in different scenarios and tasks. The following are descriptions of the capabilities of these roles:

{role_capabilities}

2. You have the ability to choose an optimal role for the task at hand.

3. When you find that a current task cannot be assigned to the right Agent, you can report this so that the task can be re-planned.

<Output Format>

You need to output a response of type json. json contains parameters and its interpretation as follows:

“‘json {

"branch": "typing.Optional[cola.fundamental.base_response_format.BranchType]. The following are the values that can be set for this parameter and their explanations: Set to ‘Continue‘ when normal response processing of the task is underway, so that the next action can be performed. set to ‘RemakeSubtasks‘ when the list of subtasks not suit the downstream role. set to ‘Interrupt‘ when you really don’t know what to do with a task. This is a dangerous operation, unless you have a good reason to refuse to continue the mission.",

"problem": "<class ’str’>. The problems you encountered. When the task is executed normally, this parameter is set to an empty string ‘”‘.",

"message": "<class ’str’>. The information you want to tell the next agent. If there is no information that needs to be specified, it is set to empty string ‘”‘. ",

"summary": "<class ’str’>. Summarize the conversation. Include: Did the answers you gave in the previous step meet the requirements of the task? What have you done now? Why are you doing this?",

"distribution": "typing.List[__main__.DistributionFormat]. A list of subtasks that need to be processed by different roles. If the role is not assigned subtasks, it does not need to be listed on the list. Type <class _main__.DistributionFormat’> is defined as follows: { ¨role¨: <class ’str’>. The role to process the subtasks ¨ ¨, ¨role_tasks¨: ¨typing.List[str]. A list of subtasks that the specified role needs to process}" } ¨

<Notice>

When assigning a task, think deeply about the capabilities required for the task at hand in the context of a human operating a computer, and select an Agent from among the downstream Agents that is capable of accomplishing that task.

Table 5: The system prompt for the task scheduler. role_capabilities denotes the skill descriptions of all agents in the decision agent pool.

# Reviewer

<Objective>

You are a Reviewer and are particularly good at determining whether an action has been successfully executed based on how the target and the Windows computer desktop have changed.

<Capabilities and Skills>

1. You can determine whether an action has successfully met expectations based on the intent, the screen state before the action is executed, and the screen state after the action is executed.   
2. You know the functions of all operations as described below:

{all_action_description}

3. You are able to give feedback when you think the action did not work, analyzing whether the action was not helpful in achieving the intent or whether the action was not performed correctly.   
4. You are able to anticipate the results of each function execution. You need to be able to tell when a function execution won’t change the desktop, and not make a wrong judgment because there is no difference between two desktop screenshots.

<Output Format>

You need to output a response of type json. json contains parameters and its interpretation as follows:

"branch": "typing.Optional[cola.fundamental.base_response_format.BranchType]. The following are the values that can be set for this parameter and their explanations: Set to ‘Continue‘ when normal response processing of the task is underway, so that the next action can be performed. set to ‘Interrupt‘ when you really don’t know what to do with a task. This is a dangerous operation, unless you have a good reason to refuse to continue the mission.",

"problem": "<class ’str’>. The problems you encountered. When the task is executed normally, this parameter is set to an empty string ‘”‘.",

"message": "<class ’str’>. The information you want to tell the next agent. If there is no information that needs to be specified, it is set to empty string ‘”‘.",

"summary": "<class ’str’>. Summarize the conversation. Include: Did the answers you gave in the previous step meet the requirements of the task? What have you done now? Why are you doing this?",

"analyze": "<class ’str’>. Give your process for analyzing the scenario.",

"judgement": "<class ’str’>. Give your judgment as to whether the action accomplishes the intent." }

<Notice>

Make sure you are familiar with the scenarios in which computers operate, as well as the scenarios in which humans operate computers to accomplish tasks.

Be sure to analyze the screenshots of your desktop before and after the action, including the smallest changes, and think deeply about whether the action meets your expectations and is consistent with your requirements.

You only need to determine whether the action was successfully executed, not solely based on the intent to determine the effect of the action, as long as the action was successfully executed.

Table 6: The system prompt for the reviewer. all_action_description denotes the description of all actions, excluding parameter descriptions.

# Searcher

# <Objective>

You are a Searcher, especially good at using browser to search for information.

Very good at manipulating browsers to navigate information, open websites, etc. Not very good at anything but browser-related tasks.

<Capabilities and Skills>

1. You can manipulate the browser, e.g. Edge, Chrome, etc.   
2. You can use the browser to search for information. You can navigate web pages, browse information for answering tasks, or download and upload files, etc.   
3. You can’t do anything other than operate the browser.   
4. When you search the web, locate the page number, you need to add the ENTER key at the end to perform the action.

# <Output Format>

You need to output a response of type json. json contains parameters and its interpretation as follows:

1 "thought_process": "typing.List[str]. Give your thought process on the question, please step by step. Give a complete thought process.",

"local_plan": "typing.List[str]. Give more detailed execution steps based on your historical experience and current scenarios and subtasks.",

"intention": "<class ’str’>. What is your intention of this step, that is, the purpose of choosing this ‘operation‘.", "operation": "typing.Optional[cola.tools.op.OpType]. You choose to perform the operation and its parameters. If you don’t need to perform the operation, set it to empty.",

"branch": "typing.Optional[cola.fundamental.base_response_format.BranchType]. The following are the values that can be set for this parameter and their explanations: Set to ‘Continue‘ when normal response processing of the task is underway, so that the next action can be performed. Set to ‘RoleTaskFinish‘ when all the assigned subtasks are complete, so that the other subtasks can be executed. set to ‘TaskMismatch‘ when you have been assigned a subtask that exceeds your capacity, so that you can reassign the subtask. set to ‘Interrupt‘ when you really don’t know what to do with a task. This is a dangerous operation, unless you have a good reason to refuse to continue the mission.",

"problem": "<class ’str’>. The problems you encountered. When the task is executed normally, this parameter is set to an empty string ‘.",

"message": "<class ’str’>. The information you want to tell the next agent. If there is no information that needs to be specified, it is set to empty string ‘”‘.",

"summary": "<class ’str’>. Summarize the conversation. Include: Did the answers you gave in the previous step meet the requirements of the task? What have you done now? Why are you doing this?",

"observation": "<class ’str’>. Give a detailed description of the current scene based on the current screenshot and the task to be accomplished.",

"information": "<class ’str’>. If the current scenario is relevant to the question to be answered, extract useful information from it that will be used as a basis for answering the question. This parameter is set to an empty string if the current task does not require a response.",

"selected_control": "typing.Optional[str]. The label of the chosen control for the operation. If you don’t need to manipulate the control this time, you don’t need this parameter."

# <Available operations>

The following is a description of the operational functions you can use and their functions and parameters:

“

# {action_description}

# <Notice>

You need to carefully judge the current scenario based on the current desktop screenshot and the screenshot labeled by the controls, as well as the current task, and give a plan for the next step in the execution to complete the task.

Based on all the available controls in the current screenshot, select the one that will be helpful in accomplishing the task and give its method of operation.

Table 7: The system prompt for the decision agent searcher. action_description is a description of all the actions of this role in the domain.

# Programmer

<Objective>

You’re a Programmer, you’re good at thinking through problems and dealing with logical reasoning, and you’re skilled at using Python code to perform calculations.

<Capabilities and Skills>

1. You can analyze complex tasks in depth and gain insight into the variables, correlations, and rules that govern them.   
2. You can use insights into factors, conditions, and rules to analyze the connections, think step by step, and give solutions and end results to problems.   
3. You can write Python code to perform some steps that require computation or some operations that you want to do.   
4. You are very proficient in the Python programming language and have the ability to write code in Python to accomplish the required tasks and give the results of execution.   
5. If you really don’t know how to accomplish the task at hand, you can ask a human for help!

<Output Format>

You need to output a response of type json. json contains parameters and its interpretation as follows:

{

"thought_process": "typing.List[str]. Give your thought process on the question, please step by step. Give a complete thought process.",

"local_plan": "typing.List[str]. Give more detailed execution steps based on your historical experience and current scenarios and subtasks.",

"intention": "<class ’str’>. What is your intention of this step, that is, the purpose of choosing this ‘operation‘.",

"operation": "typing.Optional[cola.tools.op.OpType]. You choose to perform the operation and its parameters. If you don’t need to perform the operation, set it to empty.",

"branch": "typing.Optional[cola.fundamental.base_response_format.BranchType]. The following are the values that can be set for this parameter and their explanations: Set to ‘Continue‘ when normal response processing of the task is underway, so that the next action can be performed. Set to ‘RoleTaskFinish‘ when all the assigned subtasks are complete, so that the other subtasks can be executed. set to ‘TaskMismatch‘ when you have been assigned a subtask that exceeds your capacity, so that you can reassign the subtask. set to ‘Interrupt‘ when you really don’t know what to do with a task. This is a dangerous operation, unless you have a good reason to refuse to continue the mission.",

"problem": "<class ’str’>. The problems you encountered. When the task is executed normally, this parameter is set to an empty string ‘”‘.",

"message": "<class ’str’>. The information you want to tell the next agent. If there is no information that needs to be specified, it is set to empty string ‘”‘.",

"summary": "<class ’str’>. Summarize the conversation. Include: Did the answers you gave in the previous step meet the requirements of the task? What have you done now? Why are you doing this?",

"analyze": "<class ’str’>. Give your process for analyzing the scenario.",

"answer": "<class ’str’>. If the task requires an answer, give a thoughtful answer. If you need to write code to get the result, give the answer based on the execution result. If answer is not empty, the task is completed and the branch is set to ‘RoleTaskFinish‘."

<Available operations>

The following is a description of the operational functions you can use and their functions and parameters:

{action_description}

<Notice>

Please answer the questions based on the above.

Note that if you need to write code to get the results, use the Python programming language. and use a function to return the result, such as:

"""# Your code

def get_result():

return result

111

Table 8: The system prompt for the decision agent programmer. action_description is a description of all the actions of this role in the domain.

# File Manager

<Objective>

You are a FileManager, specialized in operating Windows systems. You are responsible for the management of files in the operating system. You can open, create, and delete files.

<Capabilities and Skills>

1. You can operate Explorer to find, create, delete, and open files.   
2. In Explorer, right-clicking on an empty area brings up a menu that allows you to accomplish the task of creating a file.   
3. In Explorer, right-clicking on a file brings up a menu that can be used to perform tasks such as deleting, renaming, copying, and so on.   
4. In Explorer, double-click the left mouse button on the file can be used to open the file, such as txt, xlsx, pdf, png, mp4 and other documents.   
5. For text files, you can read the contents directly without having to open them with the Task Manager.   
6. If you really don’t know how to accomplish the task at hand, you can ask a human for help!

<Output Format>

You need to output a response of type json. json contains parameters and its interpretation as follows:

“‘json {

"thought_process": "typing.List[str]. Give your thought process on the question, please step by step. Give a complete thought process.",

"local_plan": "typing.List[str]. Give more detailed execution steps based on your historical experience and current scenarios and subtasks.",

"intention": "<class ’str’>. What is your intention of this step, that is, the purpose of choosing this ‘operation‘.",

"operation": "typing.Optional[cola.tools.op.OpType]. You choose to perform the operation and its parameters. If you don’t need to perform the operation, set it to empty.",

"branch": "typing.Optional[cola.fundamental.base_response_format.BranchType]. The following are the values that can be set for this parameter and their explanations: Set to ‘Continue‘ when normal response processing of the task is underway, so that the next action can be performed. Set to ‘RoleTaskFinish‘ when all the assigned subtasks are complete, so that the other subtasks can be executed. set to ‘TaskMismatch‘ when you have been assigned a subtask that exceeds your capacity, so that you can reassign the subtask. set to ‘Interrupt‘ when you really don’t know what to do with a task. This is a dangerous operation, unless you have a good reason to refuse to continue the mission.", "problem": "<class ’str’>. The problems you encountered. When the task is executed normally, this parameter is set to an empty string ”‘.",

"message": "<class ’str’>. The information you want to tell the next agent. If there is no information that needs to be specified, it is set to empty string ‘”‘.",

"summary": "<class ’str’>. Summarize the conversation. Include: Did the answers you gave in the previous step meet the requirements of the task? What have you done now? Why are you doing this?"

<Available operations>

The following is a description of the operational functions you can use and their functions and parameters:

{action_description}

<Notice>

Please carefully analyze the current task requirements and develop reasonable steps to complete the task and give the correct response.

Table 9: The system prompt for the decision agent file manager. action_description is a description of all the actions of this role in the domain.

Application Manager   
<Objective>   
You are a ApplicationManager, specialized in operating Windows systems. You can open applications.   
<Capabilities and Skills>   
1. You can select the desired application from those already present in the background.   
2. If you don't need any of the applications you have opened, you can open the application you need directly based on the application name.   
3. If you really don't know how to open the apps you need, or don't know what apps you need, you can ask a human for help!   
<Some Applications>   
The following are just a few examples of applications you can work with, if you need other applications you can identify them yourself.   
There's more to apps than you know. Here are some examples:   
"json   
{ "Microsoft Edge": "This is a browser that can be used to browse the web and search for information.", "Explorer": "This is Explorer, which can be used to manage your computer's files.", "wechat": "It's a chat program."   
}   
<Output Format>   
You need to output a response of type json. json contains parameters and its interpretation as follows:   
"json   
{ "thought_process": "typing.List[str]. Give your thought process on the question, please step by step. Give a complete thought process.", "local_plan": "typing.List[str]. Give more detailed execution steps based on your historical experience and current scenarios and subtasks.", "intention": "<class 'str'>. What is your intention of this step, that is, the purpose of choosing this 'operation'.", "operation": "typing Optional[ola.tools.op.OpType]. You choose to perform the operation and its parameters. If you don't need to perform the operation, set it to empty.", "branch": "typing Optional[ola.fundamental.base_response_format/BranchType]. The following are the values that can be set for this parameter and their explanations: Set to Continue when normal response processing of the task is underway, so that the next action can be performed. RoleTaskFinish' can only be set when a result is obtained. set to TaskMismatch' when you have been assigned a subtask that exceeds your capacity, so that you can reassign the subtask. set to Interrupt' when you really don't know what to do with a task. This is a dangerous operation, unless you have a good reason to refuse to continue the mission.", "problem": "<class 'str'>. The problems you encountered. When the task is executed normally, this parameter is set to an empty string ".". "message": "<class 'str'>. The information you want to tell the next agent. If there is no information that needs to be specified, it is set to empty string ".". "summary": "<class 'str'>. Summarize the conversation. Include: Did the answers you gave in the previous step meet the requirements of the task? What have you done now? Why are you doing this?", "analyze": "<class 'str'>. Give your process for analyzing the scenario."   
}   
<Available operations>   
The following is a description of the operational functions you can use and their functions and parameters:   
{action_description}   
<Notice>   
Please fully analyze the applications needed for the task, first look for them from the applications already open in the background, and if there are none needed, then you can open them by application name. You should not set branch to RoleTaskFinish when you do not get the application until you get the result.

Table 10: The system prompt for the decision agent application manager. action_description is a description of all the actions of this role in the domain.
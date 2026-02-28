# Towards Long-Horizon Vision-Language Navigation: Platform, Benchmark and Method

Xinshuai Song $^{1*}$ Weixing Chen $^{1*}$ Yang Liu $^{1,3\dagger}$ Weikai Chen $\ddagger$ Guanbin Li $^{1,2,3}$ Liang Lin $^{1,2,3}$ Sun Yat-sen University, China $^{2}$ Peng Cheng Laboratory Guangdong Key Laboratory of Big Data Analysis and Processing

{songxsh,chenwx228}@mail2.sysu.edu.cn,liuy856@mail.sysu.edu.cn,chenwk891@gmail.com

liguanbin@mail.sysu.edu.cn,linliang@ieee.org

hcplab-sysu.github.io/LH-VLN

![](images/262163921cbb1f1e83de29e8b45c86efd0b7344abfcdf5e9117402d0c0bd40f2.jpg)  
Figure 1. Framework Overview. Different from existing vision language navigation, object loco-navigation, and demand-driven navigation benchmarks, LH-VLN divides navigation into multiple subtasks, requiring the agent to complete these tasks sequentially within the scene. Our data generation framework provides a general LH-VLN task generation pipeline, and the newly built LHPR-VLN benchmark for multi-stage navigation tasks. Our navigation model, based on the chain-of-thought (CoT) feedback and adaptive memory design, achieves efficient navigation by utilizing CoT prompts and dynamic long-term and short-term memories.

# Abstract

Existing Vision-Language Navigation (VLN) methods primarily focus on single-stage navigation, limiting their effectiveness in multi-stage and long-horizon tasks within complex and dynamic environments. To address these limitations, we propose a novel VLN task, named Long-Horizon Vision-Language Navigation (LH-VLN), which emphasizes long-term planning and decision consistency across consecutive subtasks. Furthermore, to support LH-VLN, we develop an automated data generation platform NavGen, which constructs datasets with complex task structures and improves data utility through a bidirectional, multi-granularity generation approach. To accurately evaluate

complex tasks, we construct the Long-Horizon Planning and Reasoning in VLN (LHPR-VLN) benchmark consisting of 3,260 tasks with an average of 150 task steps, serving as the first dataset specifically designed for the long-horizon vision-language navigation task. Furthermore, we propose Independent Success Rate (ISR), Conditional Success Rate (CSR), and CSR weight by Ground Truth (CGT) metrics, to provide fine-grained assessments of task completion. To improve model adaptability in complex tasks, we propose a novel Multi-Granularity Dynamic Memory (MGDM) module that integrates short-term memory blurring with long-term memory retrieval to enable flexible navigation in dynamic environments. Our platform, benchmark and method supply LH-VLN with a robust data generation pipeline, comprehensive model evaluation dataset, reasonable metrics, and a novel VLN model, establishing a foundational framework for advancing LH-VLN.

# 1. Introduction

Current Vision-Language Navigation (VLN) benchmarks and methods primarily focus on single-stage or short-term tasks, which involve simple objectives and limited action sequences, making them suitable for controlled settings but insufficient for real-world applications [23] (see Figure 1). In practical scenarios, agents must handle complex, long-horizon instructions that span multiple sub-tasks, requiring ongoing decision-making, dynamic re-planning, and sustained reasoning across extended periods [8, 27, 30]. These capabilities are crucial for applications like autonomous assistants or service robots where coherent navigation over a long temporal horizon is essential. To address this gap, we propose, for the first time, a new task-coded Long-Horizon VLN (LH-VLN), to evaluate and enhance agents' abilities to manage multi-stage, context-rich navigation tasks that more accurately reflect real-world complexity.

The LH-VLN task is dedicated to push agents beyond simple, short-term navigation by requiring them to deeply comprehend complex task instructions, maintain continuous navigation, and handle sequential sub-tasks seamlessly across a dynamic environment. Achieving this goal involves three critical components: 1) an automated data generation platform to construct benchmarks with complex task structures and improves data utility, 2) a high-quality benchmark capturing the complexity of long-horizon, multi-stage tasks and accurately assess the agent's task execution and detailed sub-task performance with reasonable metrics, and 3) a specialized method to equip agents with adaptive memory for complex navigation. In this work, we provide a comprehensive solution that addresses these three aspects, laying the foundation for robust LH-VLN in real-world scenarios.

Platform-wise, previous platforms [16, 29, 35, 36, 41] for VLN data generation lack sufficient versatility and depend on a specific simulation platform and assets, resulting in relatively limited generated data [6]. To overcome these limitations, we introduce NavGen, a novel data generation platform that automates the construction of complex, multistage datasets. NavGen generates data through a bidirectional, multi-granularity approach, producing forward and backward sub-tasks to enrich task diversity and improve data utility. This automated platform allows for the scalable creation of richly varied navigation tasks that support advanced model training and long-horizon VLN evaluation.

Benchmark-wise, existing VLN benchmarks [17, 20, 48] are limited by their simple task structures, low data diversity, and constrained instructional flexibility, which restrict model generalization and hinder support for complex, long-horizon tasks. These benchmarks often rely on manual annotation, making them labor-intensive to create and less scalable for handling multi-stage tasks [21, 46]. To overcome these challenges, we build Long-Horizon Planning and Reasoning in VLN (LHPR-VLN) based on the NavGen

platform. $LHPR-VLN$ is the first LH-VLN benchmark that consists of 3,260 tasks with an average of 150 task steps. This large-scale benchmark captures the depth and variety required for evaluating long-horizon VLN, encompassing a wide range of sub-task structures and navigation complexities. Additionally, traditional coarse-grained success rates (SR) are inadequate for complex tasks, as task complexity makes it difficult for overall success rates to accurately reflect model capabilities. Therefore, we propose three new metrics for more thorough evaluation: Conditional Success Rate (CSR), CSR weighted by Ground Truth (CGT), and Independent Success Rate (ISR), to assess success for each subtasks, capturing the model's performance at each step and offering a more detailed evaluation of execution across the full scope of LH-VLN challenges.

Existing VLN methods typically rely on discretizing the environment into static points for path prediction, limiting adaptability in complex, dynamic settings [2, 24, 39, 47]. To bridge this gap and enhance real-world applicability in LH-VLN tasks, we introduce a Multi-Granularity Dynamic Memory (MGDM) module to enhance the model's adaptability and memory handling. The MGDM module operates by integrating both short-term and long-term memory mechanisms. While short-term memory blurring and forgetting functions help the model focus on recent, relevant information, long-term memory retrieval pulls in key historical data from previous navigation steps [31]. This combination allows the model to adjust to environmental changes and retain context over extended sequences, addressing the challenges of sustained reasoning and adaptive re-planning in dynamic environments. With MGDM, we achieve state-of-the-art performance on the LH-VLN task, demonstrating its effectiveness in maintaining coherent decision-making and robust navigation over long, multi-stage tasks. Our contributions can be summarized as follows:

- We propose the LH-VLN task, a new task designed to evaluate agents in complex, multi-stage navigation tasks requiring sustained reasoning and adaptability.   
- We develop NavGen, an automated data generation platform that produces a high-quality, long-horizon dataset, enabling scalable task diversity and improved data utility.   
- We introduce the LHPR-VLN benchmark with 3,260 tasks, each averaging 150 steps, and propose three new metrics for detailed, sub-task-level evaluation.   
- We present the MGDM model, designed to enhance model adaptability in dynamic settings through combined short-term and long-term memory mechanisms.

# 2. Related Work

# 2.1. Vision-Language Navigation

Embodied Vision-Language Navigation (VLN) aims to enable agents to perform navigation tasks in complex environ-

![](images/692c2517f18d990b1559393d78900beea695672df8c2396686d52637e2f55417.jpg)  
Figure 2. The NavGen data generation platform. The forward generation generates LH-VLN complex tasks and corresponding subtasks by prompting GPT-4 with sampling asserts. The sampled assets are deployed on the simulator. Based on the navigation model or expert decisions, corresponding trajectory data is generated. In the backward generation, the trajectory of each subtask is split into action-label pairs by trajectory splitting algorithm according to the trajectory type, these pairs are then input into GPT-4 to generate step-by-step tasks.

ronments based on language instructions. Current methods advance in three main directions: map-based strategies, waypoint prediction, graph-based approaches, and large-model predictions. Map-based strategies, such as VLNVER [22] and HNR-VLN [37], employ volumetric representations or neural radiance fields to facilitate spatial understanding and exploration by the agent. Modular designs like those in FILM [26] integrate language instructions with environmental perception, enhancing task efficiency. The second category, waypoint prediction-based methods, includes models such as ETPNav [1] and MultiPLY [11], which optimize navigation through key-point prediction and environmental graph learning, thereby supporting improved generalization across discrete and continuous environments [10]. Additionally, large language model-based approaches, including NaviLLM [45] and NaViD [43], excel at interpreting complex instructions by tightly integrating language reasoning with visual tasks. However, existing methods often remain limited to single-stage tasks and lack consistent planning for long-horizon, multi-stage tasks.

# 2.2. Benchmark for Vision-Language Navigation

The progression of VLN tasks has been propelled by a range of datasets, each introducing unique challenges and enhancing evaluation benchmarks for embodied agents performing tasks in complex environments. Early datasets, such as Room-to-Room (R2R) [3] and its extension Room-for-Room (R4R) [12], focus on step-by-step navigation through predefined paths with fine-grained instructions based on static images, while later datasets like VLN-CE [17] shift towards continuous navigation in dynamic spaces, requiring more flexible decision-making. More recent datasets, including CVDN [33], REVERIE [28], and SOON [48], further broaden the scope of VLN by integrating dialogue history, object localization, and complex instruction comprehension, pushing agents to understand high-level natural language commands and locate specific targets. Meanwhile, OVMM [42] and Behavior-1K [20] add layers of

complexity by incorporating navigation, manipulation, and object interaction, simulating extended real-world tasks that involve multiple sub-tasks. IVLN [18] and Goat-Bench [15] allow the agent to continuously complete multiple independent single-target navigation tasks while maintaining memory. Despite these progresses, there is still a notable gap in benchmarks that support LH-VLN with multi-stage sub-tasks in highly complex environments.

# 3. Platform, Benchmark, and Metrics

We developed a data generation platform named NavGen, specifically designed to support the data needs of the LHVLN task. Based on this platform, we created the LHPR-VLN benchmark to evaluate model performance in terms of long-term planning capabilities within this task.

# 3.1. NavGen

The NavGen platform integrates automated data generation with a bi-directional generation mechanism to produce task instructions and associated trajectory data. The two-pronged approach includes forward data generation, which focuses on complex LH-VLN task creation, and backward data generation, which decomposes multi-stage navigation sub-tasks into granular, actionable steps, shown in Fig. 2.

# 3.1.1 Forward Data Generation

In the forward data generation phase, we utilize GPT-4 to create task instructions by synthesizing scene assets and robot configurations, as shown in Fig. 2. Specifically, our scene assets come from the HM3D dataset [40], which offers a rich collection of 3D panoramic scenes annotated semantically across 216 settings, providing an extensive foundation for task creation. Additionally, robot configurations are carefully tailored to different robotic platforms, such as Boston Dynamics' Spot and Hello Robot's Stretch, each with unique camera heights, task spaces, and sensor parameters to accommodate a variety of tasks. With these assets

Table 1. Comparison to VLN benchmarks.   

<table><tr><td>Benchmark</td><td>Avg. Instruction Length</td><td>Avg. Task Steps</td><td>Simulator</td><td>Task Type</td><td>Scenes</td><td>Task Num</td></tr><tr><td>R2R [3]</td><td>29</td><td>&lt;8</td><td>Matterport3D</td><td>Step-by-step Nav</td><td>90</td><td>21567</td></tr><tr><td>REVERIE [28]</td><td>18</td><td>&lt;8</td><td>Matterport3D</td><td>Obj Loco-nav</td><td>90</td><td>21702</td></tr><tr><td>VLN-CE [17]</td><td>30</td><td>55.88</td><td>Habitat</td><td>Step-by-step Nav</td><td>90</td><td>4475</td></tr><tr><td>FAO [48]</td><td>39</td><td>10</td><td>Matterport3D</td><td>Obj Loco-nav</td><td>90</td><td>3848</td></tr><tr><td>Behavior-1k [20]</td><td>3.27</td><td>-</td><td>OmniGibson</td><td>Complex Housework</td><td>50</td><td>1000</td></tr><tr><td>IVLN [18]</td><td>-</td><td>-</td><td>M3D&amp;Habitat</td><td>Iterative VLN</td><td>72</td><td>789</td></tr><tr><td>Goat-Bench [15]</td><td>-</td><td>-</td><td>Habitat</td><td>Iterative VLN</td><td>181</td><td>725360</td></tr><tr><td>LHPR-VLN (Ours)</td><td>18.17</td><td>150.95</td><td>Habitat</td><td>Multi-stage VLN</td><td>216</td><td>3260</td></tr></table>

and configurations as the initial resource pool, a custom-designed prompt serves as the input for GPT-4, which combines scene details $S$ and robot configurations $R$ . Then GPT-4 outputs an instruction list $D_{ins} = \mathcal{G}(S,R,\mathrm{prompt}_1)$ , including the sub-task and multi-stage instructions, and $\mathcal{G}$ is denoted the GPT-4. This list is imported into the Habitat3 simulator Sim, where an expert model or a well-trained navigation model guides the agent $A$ through the task, which the expert model is a navmesh model and greedy pathfinder algorithm built from Habitat [9, 19]. The simulator autonomously generates trajectories $D_{traj}$ , the foundational data for subsequent splitting into task segments:

$$
D _ {t r a j} = \operatorname {S i m} \left(D _ {i n s}, S, A, \mathbf {O R} (M, E)\right) \tag {1}
$$

where OR represents that either $M$ or $E$ can be used.

# 3.1.2 Backward Data Generation

After obtaining the trajectory through forward task generation, we decompose the trajectory of complex tasks and create step-by-step VLN tasks for each trajectory segment. The trajectory decomposition algorithm (more detail can be found in the supplementary material) splits complex task trajectories into multiple single-stage navigation task trajectories. Within a single-stage navigation goal trajectory, the algorithm divides the trajectory into segments representing "move forward," "turn left," "turn right," and "bypass forward." By using a dynamic sliding window, the algorithm continuously searches for all the longest continuous action segments within the trajectory. These continuous action segments serve as the basic units of action instructions in step-by-step navigation tasks. For each segment, the RAM image annotation model [44] provides high-confidence visual annotations. These annotations, coupled with action instructions, are input as prompts into GPT-4 to generate VLN tasks for step-by-step guidance, thereby creating a refined set of decomposed single-stage navigation tasks.

# 3.2. The LHPR-VLN Benchmark

Our LHPR-VLN benchmark defines a complex task that includes multiple single-stage subtasks. For an LHPR-VLN task, the basic format is: "Find something somewhere, and take it to something somewhere, then..." Each complex task involves locating an object at a specified initial location

![](images/90c67d6e2455d83006b90348d955f30b6b12a68e7c12e27a6a9e1567684f1525.jpg)  
Figure 3. Overview of the LHPR-VLN benchmark statistics. In our statistics, Spot and Stretch robot-type tasks account for $50.5\%$ and $49.5\%$ , respectively. LH-VLN tasks containing 2, 3, and 4 subtasks account for $39.0\%$ , $52.4\%$ , and $8.6\%$ , respectively.

and transporting it to a designated target location, potentially encompassing two to four sequential navigation subtasks. The embodied agent needs to sequentially complete these single-stage navigation tasks to ultimately fulfill the instruction. For each single-stage navigation task, the agent must approach within a 1-meter geodesic distance of the target object, ensuring the object is positioned within a 60-degree horizontal field of view to maintain task fidelity.

Throughout navigation, the agent acquires observational data from three perspectives $(+60^{\circ}, 0^{\circ}, -60^{\circ})$ and is permitted to execute fundamental actions: turn left, move forward, turn right, and stop. When the agent selects the "stop" action, the sub-task is deemed complete, and task success is evaluated based on the agent's final positional state relative to the target. Table 1 presents a comparison between representative VLN benchmarks, our LHPR-VLN is the first LHVLN benchmark, containing 3,260 multi-stage and step-by-step VLN tasks from 216 complex scenes, with an average of 150 task action steps and 18.17 instruction length.

# 3.3. Reasonable Metrics

To rigorously assess model performance in the LH-VLN task, we introduced specialized metrics, complementing the standard evaluation metrics (Success Rate (SR), Oracle Success Rate (OSR), Success weighted by Path Length (SPL), and Navigation Error (NE)). These new metrics include Independent Success Rate (ISR), Conditional Success Rate (CSR), and CSR weighted by Ground Truth (CGT). ISR quantifies the success rate of each sub-task individually, providing insight into independent sub-task comple

![](images/5507bb6bb12bcaf808679e6a1245b1d6c7adf2f26621051a9aefecfb07dfea90.jpg)  
Figure 4. The framework of the Multi-Granularity Dynamic Memory (MGDM) model. The CoT feedback module receives task instructions and, based on historical observation of corresponding memory, generates a chain of thought and constructs language prompts. The short-term memory module aims to minimize the entropy of the confidence vector, using pooling operations to forget and blur the memory sequence. The long-term memory module selects and matches data from the dataset to weight the decisions of the LLM, ultimately determining the action to be executed by the agent.

tion rates. CSR evaluates the success of the overall complex task, as the outcome of each sub-task impacts the subsequent ones, thus encapsulating interdependencies in the task sequence.

$$
I S R = \sum_ {j = 0} ^ {M} \sum_ {i = 0} ^ {N} \frac {s _ {j , i}}{M \cdot N} \tag {2}
$$

where $M$ is the numble of tasks, and $N$ is the number of sub-tasks in $\mathrm{Task}_j$ . The CSR metric is calculated as follows:

$$
C S R = \sum_ {j = 0} ^ {M} \sum_ {i = 0} ^ {N} \frac {s _ {j , i} (1 + (N - 1) s _ {j , i - 1})}{M \cdot N ^ {2}} \tag {3}
$$

where $s_{j,i}$ denotes the success of the $i$ -th sub-task in $\mathrm{Task}_j$ .

CGT further refines CSR by incorporating ground truth of sub-task path length $P_{i}$ and full task path length $P$ weighting, to account for deviations in path difficulty. CGT is calculated as:

$$
C G T = \sum_ {j = 0} ^ {M} \sum_ {i = 0} ^ {N} \frac {P _ {i}}{P} \cdot \frac {s _ {j , i} (1 + (N - 1) s _ {j , i - 1})}{M \cdot N} \tag {4}
$$

We also designed a metric Target Approach Rate (TAR) based on NE to reflect the model's performance in cases where the navigation success rate is relatively low. The relevant settings can be found in the supplementary materials.

Furthermore, the multi-granularity task instructions generated by the NavGen platform allow us to test the model's responsiveness to various instruction types within the same trajectory. This testing approach not only facilitates an analysis of the agent's focus during navigation but also enables a robust evaluation of task comprehension and execution across complex scenarios through these novel metrics. Thus, these new metrics provide a comprehensive evaluation of model performance in LH-VLN tasks.

# 4. Multi-Granularity Dynamic Memory Model

To achieve robust LH-VLN, our Multi-Granularity Dynamic Memory (MGDM) model follows the general VLN pipeline and comprises three essential components: the base model, the Chain-of-Thought (CoT) Feedback module, and Adaptive Memory Integration and Update (AMIU), as shown in Fig. 4. These components enable robust performance in LH-VLN, addressing challenges related to spatial awareness [25], instruction comprehension [4], and task continuity [13] across long-horizon sequences.

# 4.1. Base Model

The base model aligns with the standard structure of VLN models. For scene observation, the model encodes multidirectional visual information using a pre-trained visual encoder vit. Each observed image $I_{i}$ is processed into visual features $v_{i}$ . To integrate scene information across multiple directions, a Transformer encoder is used for multi-view feature fusion. The directional image features $\{v_{i}\}_{i = 1}^{n}$ are processed through the Transformer encoder, resulting in contextually enriched representations $\{o_i\}_{i = 1}^n$ that capture inter-relational information across views.

Each directional view is distinguished by embedding directional tokens ('left', 'front', 'right') to construct a comprehensive scene representation $S$ :

$$
S = \left[ \mathcal {E} \left(^ {\prime} \text {l e f t} ^ {\prime}\right), o _ {\text {l e f t}}, \dots , \mathcal {E} \left(^ {\prime} \text {r i g h t} ^ {\prime}\right), o _ {\text {r i g h t}} \right] \tag {5}
$$

where $\mathcal{E}$ denotes the embedding layer. For historical observations $H_{i}$ , each previous scene is encoded similarly, with stepwise embeddings added to capture temporal relations, establishing sequential order within the observation history:

$$
H _ {n + 1} = \left[ \mathcal {E} (1), h _ {1}, \dots , \mathcal {E} (n), h _ {n} \right] \tag {6}
$$

The scene and historical representations are then combined into a unified prompt, which is fed into the large language

model (LLM) $\mathcal{G}$ to select the next action:

$$
a _ {n + 1} = \mathcal {G} (\mathcal {E} (\text {p r o m p t} _ {3}), S, H _ {n}) \tag {7}
$$

# 4.2. Navigate with CoT and Memory

To address the limited interpretability and susceptibility to "hallucinations" [34] in LLM-based VLN models (wherein the agent completes tasks without true comprehension), we introduce a Chain-of-Thought (CoT) [38] Feedback module that receives task instructions and, based on historical observation of corresponding memory, generates a chain of thought and constructs language prompts. This module aims to enhance the agent's reasoning capability by iteratively refining its task understanding and action planning.

CoT Feedback. At the beginning of each sub-task and periodically during navigation, the CoT Feedback module receives task instructions, current observation, and history visual observations in memory, along with the prompt, are input into GPT-4 to generate the chain of thought CoT = GPT-4(Obs, Hist, Instruction, Prompt). GPT-4 uses past observations and task instructions to establish the current task context, which implies comprehensive task understanding. The task is then decomposed based on this understanding, guiding the agent's immediate actions. This reflective process enables the agent to adjust and refine its interpretations, improving task comprehension and execution.

Adaptive Memory Integration and Update. Previous VLN works often used visual encoding from past observations as memory, which is typically effective. However, in LH-VLN tasks, the lengthy task duration causes an excessive accumulation of memory, making this approach impractical. Moreover, existing methods often discard the oldest memories to maintain a fixed-length memory sequence or just discard some memories that the model thinks inappropriate [2], which inadvertently removes critical information. To mitigate these limitations, we design an Adaptive Memory Integration and Update (AMIU) module incorporating short-term memory, long-term memory, and a memory blurring and forgetting process.

Short-term memory $M_{st}$ is structured from historical observation encoding, capturing temporally ordered observations as the agent moves through the environment:

$$
M _ {s t} = \left\{h _ {i} \right\} _ {i = 0} ^ {n} \tag {8}
$$

When the memory length $n$ reaches a set maximum $N$ , dynamic forgetting is triggered. Each memory element $h_i$ has an associated confidence score $c_i = \mathcal{G}(\cdot)_i$ , representing the model's confidence in corresponding action. The memory sequence $M_{st}$ thus has an associated confidence vector $C = \{c_i\}_{i = o}^n$ .

The forgetting module employs a "pooling function" that we define it as $\mathcal{P}$ . $\mathcal{P}(C)_i$ represents the pooling operation with a window size of 2 applied to the $i_{th}$ element and its

neighboring elements in $C$ , which reduces its length by one:

$$
\mathcal {P} (C) _ {i} = \left\{c _ {1}, \dots , \operatorname {A v g P o o l} \left(c _ {i - 1}, c _ {i}, c _ {i + 1}\right), \dots , c _ {n} \right\} = C _ {i} \tag {9}
$$

where $C_i \in \mathbb{R}^{n-1}$ . We apply the pooling operation to each element in $C$ separately, obtaining $\{C_i\}_{i=0}^n = \{\mathcal{P}(C)_i\}_{i=0}^n$ . We then calculate the entropy of each $C_i$ and identify the pooling index with the smallest entropy:

$$
\arg \min  _ {i} \left(- \sum_ {j = 1} ^ {n - 1} s _ {j} \log s _ {j}\right), s _ {j} = \frac {C _ {i , j}}{\sum_ {j = 0} ^ {n - 1} C _ {i , j}} \tag {10}
$$

The same pooling operation is applied to the $M_{st}$ elements corresponding to the pooling index and add new short-term memory to maintain the memory sequence.

$$
M _ {s t} = \mathcal {P} \left(M _ {s t}\right) _ {i} + h _ {n} ^ {*} \tag {11}
$$

Long-term memory $M_{lt}$ serves as a reinforcement mechanism. As the agent navigates, long-term memory retrieves relevant observations and actions based on target $T$ from the dataset, matching them with the agent's current observation to provide guidance. The retrieval process selects the top $k$ matching observation-action pairs, which are weighted to inform the current decision vector. This memory is sourced from the LHPR-VLN dataset, reinforcing prior learning:

$$
M _ {l t} = \operatorname {D a t a s e t} (T) = \left\{o b s _ {j}, a c t _ {j} \right\} _ {j = 1} ^ {m} \tag {12}
$$

Thus, the indices of the selected $M_{lt}$ can be formulated as:

$$
I _ {k} = \operatorname {a r g s o r t} _ {t = 0} ^ {k} \left(\left\{\frac {\operatorname {o b s} _ {j} \cdot v}{\sqrt {\sum_ {i = 1} ^ {n _ {v}} \operatorname {o b s} _ {j , i} ^ {2}} \cdot \sqrt {\sum_ {i = 1} ^ {n _ {v}} v _ {i} ^ {2}}} \right\} _ {j = 1} ^ {m}\right) \tag {13}
$$

The action decision $a$ is weighted by averaging the retrieved actions:

$$
a = a \cdot \operatorname {a v g} \left(\left\{a c t _ {t} \right\} _ {t = 0} ^ {k}\right) \tag {14}
$$

where $a$ is the current decision vector. The final cross-entropy loss is computed between the model's decision $a$ and the expert's decision $e$ at current action:

$$
\arg \min  _ {\Theta} \mathcal {L} (a, e) = \arg \min  _ {\Theta} \left(- \sum_ {i = 0} ^ {n} a _ {i} \log \left(e _ {i}\right)\right) \tag {15}
$$

# 5. Experiment

# 5.1. Experimental Settings

Simulator: We conduct experiments in Habitat3 [9, 19], which provides a continuous 3D scene platform for VLN. Additionally, we perform experiments in Isaac Sim, which has high-quality scene rendering and physical interactions.

Sensors: For each action step, the agent receives RGB observations from there directions of front, left $(+60^{\circ})$ , and

Table 2. Performance comparison in LH-VLN Task with different task length.   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Type</td><td colspan="5">2-3 Subtasks</td><td colspan="5">3-4 Subtasks</td></tr><tr><td>SR↑</td><td>NE↓</td><td>ISR↑</td><td>CSR↑</td><td>CGT↑</td><td>SR↑</td><td>NE↓</td><td>ISR↑</td><td>CSR↑</td><td>CGT↑</td></tr><tr><td>Random</td><td>-</td><td>0.</td><td>14.09</td><td>0.</td><td>0.</td><td>0.</td><td>0.</td><td>10.91</td><td>0.</td><td>0.</td><td>0.</td></tr><tr><td>GLM-4v prompt [7]</td><td>Zero-shot</td><td>0.</td><td>15.63</td><td>0.</td><td>0.</td><td>0.</td><td>0.</td><td>10.97</td><td>0.</td><td>0.</td><td>0.</td></tr><tr><td>NaviLLM [45]</td><td>Pretrain</td><td>0.</td><td>12.11</td><td>0.</td><td>0.</td><td>0.</td><td>0.</td><td>10.04</td><td>0.</td><td>0.</td><td>0.</td></tr><tr><td>NaviLLM [45]</td><td>Finetuned</td><td>0.</td><td>12.24</td><td>0.</td><td>0.</td><td>0.</td><td>0.</td><td>9.79</td><td>3.54</td><td>2.53</td><td>5.24</td></tr><tr><td>GPT-4 + NaviLLM</td><td>Pretrain</td><td>0.</td><td>12.23</td><td>0.</td><td>0.</td><td>0.</td><td>0.</td><td>10.00</td><td>4.37</td><td>2.91</td><td>5.23</td></tr><tr><td>MGDM (Ours)</td><td>Finetuned</td><td>0.</td><td>3.54</td><td>0.</td><td>0.</td><td>0.</td><td>0.</td><td>1.23</td><td>4.69</td><td>3.30</td><td>5.83</td></tr></table>

right $(-60^{\circ})$ . Depth images for these three directions can also be customized.

Actions: We provide atomic actions for the agent, including 'move forward' $(+0.25)$ , 'turn left' $(+30^{\circ})$ , 'turn right' $(-30^{\circ})$ , and 'stop'. When the agent performs the stop action, the current task (or sub-task) is considered complete. We also provide a coordinate-based movement option.

Scene Assets: Our scene assets are primarily from HM3D [40], which includes 216 large-scale indoor 3D reconstructed scenes with semantic annotations. Besides, we use HSSD [14], which includes 211 high-quality indoor scenes, to test the data generation with NavGen.

Robot Configurations: The robots include the Stretch robot from Hello Robot and the Spot robot from Boston Dynamics. Stretch has a wheeled base and a manipulator with a structural frame, while Spot is a quadruped robot dog capable of mounting a mechanical arm on its back.

Training Settings: We alternately use imitation learning and trajectory-based supervised learning. The LLM is Vi-cuna 7B v0 [5], and the visual encoder is the ViT model from EVA-CLIP-02-Large [32]. The visual encoder remains frozen during training. In the training phase, we utilize the Adam optimizer with a learning rate of 3e-5.

Metrics: Besides our metrics ISR, CSR, and CGT, we also used traditional metrics [3], including SR (Success Rate), SPL (Success weighted by Path Length), OSR (Oracle Success Rate), and NE (Navigation Error). For SR, OSR and SPL, the task is considered successful only when all sub-tasks in a LH-VLN task are completed correctly in the logical sequence of instructions. For NE, only when the agent takes the action of 'stop', the NE counts.

# 5.2. Baseline Models

- ETPNav [2]: ETPNav is a graph-based navigation model where the agent's current and historical observations are modeled as graph nodes.   
- GLM-4v prompt [7]: GLM-4v is a state-of-the-art vision-language model. To evaluate the performance of vision-language models in LH-VLN tasks, we use prompt engineering to guide GLM-4v to produce reasonable outputs and test its actual performance.   
- NaviLLM [45]: NaviLLM is the state-of-the-art model for navigation in discrete environments. We adapted this approach to continuous environments and fine-tuned it on the dataset to evaluate its performance in LH-VLN.

- GPT-4 + NaviLLM: To evaluate the performance of traditional single-stage models in LH-VLN with the assistance of a LLM to decompose complex tasks, we combined GPT-4 with NaviLLM. GPT-4 first decomposes the complex task into several sub-task, and NaviLLM then executes each sub-task sequentially.

# 5.3. Result Analysis

We test baseline models on LH-VLN task and its corresponding step-by-step trajectories with LHPR-VLN benchmark. Through these tests, we aim to answer the following questions: Q1: Can existing models understand and complete multi-stage complex tasks with limited information?

Q2: How to understand the relations between multi-stage complex tasks and single-stage simple tasks? Q3: What is the significance of memory in multi-stage complex tasks?

RQ1: For ETPNav, due to the inherent limitations of its waypoint predictor, even with only three viewpoint RGBD settings, the model still fails to effectively predict navigable points, despite being designed to handle invalid navigation points and deadlock states. The performance of each model in LH-VLN task is shown in Table 2. As seen, all models perform poorly. In the relatively short LH-VLN tasks with 2-3 subtasks, the SR, ISR, CSR, and CGT of all models are 0. This indicates that these models are unable to complete even a single subtask. In the longer LH-VLN tasks with 3-4 subtasks, only fine-tuned NaviLLM, GPT-4+NaviLLM, and our MGDM can complete some subtasks. This suggests that existing models cannot effectively understand and complete multi-stage complex tasks with limited information.

RQ2: To explore the relation between multi-stage complex tasks and single-stage simple tasks, we test the combination of the single-stage navigation model NaviLLM with GPT-4 task decomposition. By using GPT-4 to decompose complex tasks, NaviLLM can sequentially perform several single-object navigation tasks. In Table 2, it can be seen that the performance of GPT-4+NaviLLM shows some improvement compared to the pre-trained NaviLLM and fine-tuned NaviLLM, especially in ISR, where it improves by $23\%$ compared to the fine-tuned NaviLLM. This indicates a significant performance improvement on individual subtasks, highlighting its single-stage navigation ability.

However, the performance of the GPT-4+NaviLLM method is still slightly lower than that of our MGDM, which has been specifically designed for complex tasks, especially

![](images/8ff355a02fc277bcb84718205b7db9e7570fabf0922aed4031aade4717d45287.jpg)

![](images/cf154f504c9c7f249affb02683b6d2a712d0cb56af50692bc73135f6848e69a1.jpg)  
Instruction: Take the towel from the bathroom and place it in the box in the living room, then retrieve the book from the living room.   
Complex, Long-horizon VLN Scene   
Figure 5. Visualization of a partially successful long-horizon navigation of our MGDM. We highlight aligned landmarks by colored bounding boxes in images and words in the instruction using the same color. In the first navigation segment, the agent looks for a towel in the bathroom. It successfully finds both the bathroom and the towel but does not enter the bathroom or gets close enough to the towel for the task to be marked as successful. In the next phase, the agent successfully finds the box in the living room.

Table 3. Performance comparison in step-by-step LH-VLN task.   

<table><tr><td>Method</td><td>SR↑</td><td>OSR↑</td><td>SPL↑</td><td>NE↓</td></tr><tr><td>Random</td><td>0.</td><td>0</td><td>0.</td><td>8.59</td></tr><tr><td>GLM-4v prompt [7]</td><td>0.</td><td>11.1</td><td>0.</td><td>6.50</td></tr><tr><td>NaviLLM [45]</td><td>6.67</td><td>6.67</td><td>2.86</td><td>10.17</td></tr><tr><td>MGDM (Ours)</td><td>0.</td><td>26.92</td><td>0.</td><td>1.70</td></tr></table>

in CGT. In fact, the CGT metric for GPT-4+NaviLLM is even lower than that of fine-tuned NaviLLM. Since CGT is weighted based on the length of the ground truth, this result suggests that our MGDM is better at completing longer and more difficult subtasks. The reason may be that our MGDM directly executes complex tasks can maintain more coherent and complete memories, which help it accomplish more complex tasks. Additionally, the advantage in CSR further indicates that MGDM has a better comprehensive understanding of multi-stage LH-VLN tasks.

Actually, combining task decomposition for complex tasks with single-stage navigation models can improve the performance of single-stage models on complex tasks to some extent. However, this approach also leads to a lack of holistic understanding of complex tasks, as well as incomplete and fragmented memory.

RQ3: Furthermore, all models perform better in ISR, CSR, and CGT on LH-VLN tasks with 3-4 subtasks than on those with 2-3 subtasks. This may be due to the fact that while longer and multi-stage tasks may be more difficult, the memory accumulated from previous stages can help the VLN model complete subtasks in subsequent stages. This may suggest the significance of developing VLN models for multi-stage complex tasks. The tendency of navigation target distribution in the LH-VLN task with different numbers of subtasks and task settings may also influence this result. Relevant details can be found in the supplementary materials. It is worth noting that our MGDM has a relatively low NE when tasks are so difficult that the model performs poorly, NE reflects the gap between the model performance and success. This suggests that our MGDM may have greater potential for LH-VLN. Additionally, in the step-by-step tasks shown in Table 3, although our MGDM has higher OSR and lower NE, its SR and SPL metrics are

Table 4. Ablation results.   

<table><tr><td>Method</td><td>NE↓</td><td>ISR↑</td><td>CSR↑</td><td>CGT↑</td></tr><tr><td>MGDM w/o Adap Mem</td><td>4.44</td><td>0.</td><td>0.</td><td>0.</td></tr><tr><td>MGDM w/o LT Mem</td><td>11.13</td><td>2.20</td><td>1.27</td><td>2.08</td></tr><tr><td>MGDM w/o CoT</td><td>2.45</td><td>0.</td><td>0.</td><td>0.</td></tr><tr><td>MGDM</td><td>1.23</td><td>4.69</td><td>3.30</td><td>5.83</td></tr></table>

both 0. This indicates that our MGDM faces an issue in effectively determining whether the goal has been achieved.

# 5.4. Ablation Studies

We performed ablation studies on multi-granularity dynamic memory module, long term memory module and the chain-of-thought (CoT) feedback module, with results shown in Table 4. As observed, the model's performance is significantly affected whether the CoT feedback module, long term memory module or the multi-granularity dynamic memory module is ablated. This indicates the crucial role of chain-of-thought generation and memory in the model's ability to solve LH-VLN tasks. From the perspective of NE, especially the multi-granularity dynamic memory module, it has significant impact on model's performance. This is also reflected in the visualization analysis of a successful long-horizon navigation example (see Figure 5). The agent's actions are very chaotic at the beginning (1-3 steps). It only acts effectively once the memory sequence reaches a certain length. This further underscores the importance of memory module design for LH-VLN tasks.

# 6. Conclusion

We address the challenges of long-horizon vision-language navigation (LH-VLN) from three aspects: platform, benchmark, and method. Specifically, we develop an automated data generation platform NavGen, which constructs datasets with complex task structures and improves data utility. We also construct the LHPR-VLN benchmark, which provides three new metrics for detailed, sub-task-level evaluation. Additionally, we present the MGDM model, designed to enhance model adaptability in dynamic settings through combined short-term and long-term memory mechanisms, achieving outstanding performance on the LH-VLN task.

# Acknowledgement

This work is supported in part by the National Key R&D Program of China under Grant No.2021ZD0111601, in part by the National Natural Science Foundation of China under Grant No. 62436009, No. 62002395 and No. 62322608, in part by the Guangdong Basic and Applied Basic Research Foundation under Grant NO. 2025A1515011874 and No.2023A1515011530, and in part by the Guangzhou Science and Technology Planning Project under Grant No. 2023A04J2030. We also thank the National Supercomputer Center in Guangzhou for computational support.

# References

[1] Dong An, Hanqing Wang, Wenguan Wang, Zun Wang, Yan Huang, Keji He, and Liang Wang. Etpnav: Evolving topological planning for vision-language navigation in continuous environments. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 3   
[2] Dong An, Hanqing Wang, Wenguan Wang, Zun Wang, Yan Huang, Keji He, and Liang Wang. Etpnav: Evolving topological planning for vision-language navigation in continuous environments. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 2, 6, 7   
[3] Peter Anderson, Qi Wu, Damien Teney, Jake Bruce, Mark Johnson, Niko Sunderhauf, Ian Reid, Stephen Gould, and Anton van den Hengel. Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2018. 3, 4, 7   
[4] Weixing Chen, Yang Liu, Binglin Chen, Jiandong Su, Yongsen Zheng, and Liang Lin. Cross-modal causal relation alignment for video question grounding. arXiv preprint arXiv:2503.07635, 2025. 5   
[5] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhang-hao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yong-hao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with $90\%$ * chatgpt quality, 2023. 7   
[6] Matt Deitke, Dustin Schwenk, Jordi Salvador, Luca Weihs, Oscar Michel, Eli VanderBilt, Ludwig Schmidt, Kiana Ehsani, Aniruddha Kembhavi, and Ali Farhadi. Objverse: A universe of annotated 3d objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13142-13153, 2023. 2   
[7] Team GLM, Aohan Zeng, Bin Xu, and Zihan Wang. Chatglm: A family of large language models from glm-130b to glm-4 all tools, 2024. 7, 8   
[8] Jing Gu, Eliana Stefani, Qi Wu, Jesse Thomason, and Xin Wang. Vision-and-language navigation: A survey of tasks, methods, and future directions. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7606-7623, 2022. 2   
[9] Saurabh Gupta, James Davidson, Sergey Levine, Rahul Sukthankar, and Jitendra Malik. Cognitive mapping and plan

ning for visual navigation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. 4,6   
[10] Yicong Hong, Zun Wang, Qi Wu, and Stephen Gould. Bridging the gap between learning in discrete and continuous environments for vision-and-language navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15439-15449, 2022. 3   
[11] Yining Hong, Zishuo Zheng, Peihao Chen, Yian Wang, Junyan Li, and Chuang Gan. Multiply: A multisensory object-centric embodied large language model in 3d world. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26406-26416, 2024. 3   
[12] Vihan Jain, Gabriel Magalhaes, Alexander Ku, Ashish Vaswani, Eugene Ie, and Jason Baldridge. Stay on the path: Instruction fidelity in vision-and-language navigation. arXiv preprint arXiv:1905.12255, 2019. 3   
[13] Kaixuan Jiang, Yang Liu, Weixing Chen, Jingzhou Luo, Ziliang Chen, Ling Pan, Guanbin Li, and Liang Lin. Beyond the destination: A novel benchmark for exploration-aware embodied question answering. arXiv preprint arXiv:2503.11117, 2025. 5   
[14] Mukul Khanna, Yongsen Mao, Hanxiao Jiang, Sanjay Haresh, Brennan Shacklett, Dhruv Batra, Alexander Clegg, Eric Undersander, Angel X. Chang, and Manolis Savva. Habitat synthetic scenes dataset (hssd-200): An analysis of 3d scene scale and realism tradeoffs for objectgoal navigation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16384–16393, 2024. 7   
[15] Mukul Khanna*, Ram Ramrakhya*, Gunjan Chhablani, Sriram Yenamandra, Theophile Gervet, Matthew Chang, Zsolt Kira, Devendra Singh Chaplot, Dhruv Batra, and Roozbeh Mottaghi. Goat-bench: A benchmark for multi-modal lifelong navigation. In CVPR, 2024. 3, 4   
[16] Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Matt Deitke, Kiana Ehsani, Daniel Gordon, Yuke Zhu, et al. Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474, 2017. 2   
[17] Jacob Krantz, Erik Wijmans, Arjun Majumdar, Dhruv Batra, and Stefan Lee. Beyond the nav-graph: Vision-and-language navigation in continuous environments. In Computer Vision-ECCV 2020,Lecture Notes in Computer Science, page 104-120, 2020. 2, 3, 4   
[18] Jacob Krantz, Shurjo Banerjee, Wang Zhu, Jason Corso, Peter Anderson, Stefan Lee, and Jesse Thomason. Iterative vision-and-language navigation. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14921-14930, 2023. 3, 4   
[19] Ashish Kumar, Saurabh Gupta, David Fouhey, Sergey Levine, and Jitendra Malik. Visual memory for robust path following. In Advances in Neural Information Processing Systems, 2018. 4, 6   
[20] Chengshu Li, Ruohan Zhan, Josiah Wong, and Li Fei-Fei. Behavior-1k: A human-centered, embodied ai benchmark with 1,000 everyday activities and realistic simulation. In Conference on Robot Learning (CoRL) 2022, 2022. 2, 3, 4

[21] Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, et al. Mybench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195-22206, 2024. 2   
[22] Rui Liu, Wenguan Wang, and Yi Yang. Volumetric environment representation for vision-language navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16317-16328, 2024. 3   
[23] Yang Liu, Weixing Chen, Yongjie Bai, Xiaodan Liang, Guanbin Li, Wen Gao, and Liang Lin. Aligning cyber space with physical world: A comprehensive survey on embodied ai. arXiv preprint arXiv:2407.06886, 2024. 2   
[24] Yuxing Long, Wenzhe Cai, Hongcheng Wang, Guanqi Zhan, and Hao Dong. Instructnav: Zero-shot system for generic instruction navigation in unexplored environment. arXiv preprint arXiv:2406.04882, 2024. 2   
[25] Jingzhou Luo, Yang Liu, Weixing Chen, Zhen Li, Yaowei Wang, Guanbin Li, and Liang Lin. Dspnet: Dual-vision scene perception for robust 3d question answering. arXiv preprint arXiv:2503.03190, 2025. 5   
[26] So Yeon Min, Devendra Singh Chaplot, Pradeep Kumar Ravikumar, Yonatan Bisk, and Ruslan Salakhutdinov. Film: Following instructions in language with modular methods. In International Conference on Learning Representations. 3   
[27] Utkarsh Aashu Mishra, Shangjie Xue, Yongxin Chen, and Danfei Xu. Generative skill chaining: Long-horizon skill planning with diffusion models. In Conference on Robot Learning, pages 2905-2925. PMLR, 2023. 2   
[28] Yuankai Qi, Qi Wu, Peter Anderson, Xin Wang, William Yang Wang, Chunhua Shen, and Anton van den Hengel. Reverie: Remote embodied visual referring expression in real indoor environments. In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020. 3, 4   
[29] Manolis Savva, Abhishek Kadian, Oleksandr Maksymets, Yili Zhao, Erik Wijmans, Bhavana Jain, Julian Straub, Jia Liu, Vladlen Koltun, Jitendra Malik, et al. Habitat: A platform for embodied ai research. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9339-9347, 2019. 2   
[30] Pierre Sermanet, Tianli Ding, Jeffrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, Sharath Maddineni, Nikhil J Joshi, et al. Robovqa: Multimodal long-horizon reasoning for robotics. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 645-652. IEEE, 2024. 2   
[31] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2998-3009, 2023. 2   
[32] Quan Sun, Yuxin Fang, Ledell Wu, Xinlong Wang, and Yue Cao. Eva-clip: Improved training techniques for clip at scale. arXiv preprint arXiv:2303.15389, 2023. 7

[33] Jesse Thomason, Michael Murray, Maya Cakmak, and Luke Zettlemoyer. Vision-and-dialog navigation. In Conference on Robot Learning, pages 394-406. PMLR, 2020. 3   
[34] Maya Varma, Jean-Benoit Delbrouck, Zhihong Chen, Akshay Chaudhari, and Curtis Langlotz. Ravl: Discovering and mitigating spurious correlations in fine-tuned vision-language models. arXiv preprint arXiv:2411.04097, 2024. 6   
[35] Xiangyu Wang, Donglin Yang, Ziqin Wang, Hohin Kwan, Jinyu Chen, Wenjun Wu, Hongsheng Li, Yue Liao, and Si Liu. Towards realistic uav vision-language navigation: Platform, benchmark, and methodology. arXiv preprint arXiv:2410.07087, 2024. 2   
[36] Yufei Wang, Zhou Xian, Feng Chen, Tsun-Hsuan Wang, Yian Wang, Katerina Fragkiadaki, Zackory Erickson, David Held, and Chuang Gan. Robogen: Towards unleashing infinite data for automated robot learning via generative simulation. arXiv preprint arXiv:2311.01455, 2023. 2   
[37] Zihan Wang, Xiangyang Li, Jiahao Yang, Yeqi Liu, Junjie Hu, Ming Jiang, and Shuqiang Jiang. Lookahead exploration with neural radiance representation for continuous vision-language navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13753-13762, 2024. 3   
[38] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824-24837, 2022. 6   
[39] Zhenyu Wu, Ziwei Wang, Xiwei Xu, Jiwen Lu, and Haibin Yan. Embodied instruction following in unknown environments. arXiv preprint arXiv:2406.11818, 2024. 2   
[40] Karmesh Yadav, Ram Ramrakhya, Santhosh Kumar Ramakrishnan, Theo Gervet, John Turner, Aaron Gokaslan, Noah Maestre, Angel Xuan Chang, Dhruv Batra, Manolis Savva, et al. Habitat-matterport 3d semantics dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4927-4936, 2023. 3, 7   
[41] Yue Yang, Fan-Yun Sun, Luca Weihs, Eli VanderBilt, Alvaro Herrasti, Winson Han, Jiajun Wu, Nick Haber, Ranjay Krishna, Lingjie Liu, et al. Holodeck: Language guided generation of 3d embodied ai environments. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16227-16237, 2024. 2   
[42] Sriram Yenamandra, Arun Ramachandran, Karmesh Yadav, Austin Wang, Mukul Khanna, Theophile Gervet, Tsung-Yen Yang, Vidhi Jain, AlexanderWilliam Clegg, John Turner, Zsolt Kira, Manolis Savva, Angel Chang, DevendraSingh Chaplot, Dhruv Batra, Roozbeh Mottaghi, Yonatan Bisk, and Chris Paxton. Homerobot: Open-vocabulary mobile manipulation. arXiv:2306.11565, 2023. 3   
[43] Jiazhao Zhang, Kunyu Wang, Rongtao Xu, Gengze Zhou, Yicong Hong, Xiaomeng Fang, Qi Wu, Zhizheng Zhang, and He Wang. Nvid: Video-based vlm plans the next step for vision-and-language navigation. arXiv preprint arXiv:2402.15852, 2024. 3   
[44] Youcai Zhang, Xinyu Huang, Jinyu Ma, Zhaoyang Li, Zhaochuan Luo, Yanchun Xie, Yuzhuo Qin, Tong Luo,

Yaqian Li, Shilong Liu, et al. Recognize anything: A strong image tagging model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1724-1732, 2024. 4   
[45] Duo Zheng, Shijia Huang, Lin Zhao, Yiwu Zhong, and Liwei Wang. Towards learning a generalist model for embodied navigation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13624-13634, 2024. 3, 7, 8   
[46] Ge Zheng, Bin Yang, Jiajin Tang, Hong-Yu Zhou, and Sibei Yang. Ddcot: Duty-distinct chain-of-thought prompting for multimodal reasoning in language models. Advances in Neural Information Processing Systems, 36:5168-5191, 2023. 2   
[47] Gengze Zhou, Yicong Hong, Zun Wang, Xin Eric Wang, and Qi Wu. Navigpt-2: Unleashing navigational reasoning capability for large vision-language models. In European Conference on Computer Vision, pages 260–278. Springer, 2025. 2   
[48] Fengda Zhu, Xiwen Liang, Yi Zhu, Qizhi Yu, Xiaojun Chang, and Xiaodan Liang. Soon: Scenario oriented object navigation with graph-based exploration. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021. 2, 3, 4
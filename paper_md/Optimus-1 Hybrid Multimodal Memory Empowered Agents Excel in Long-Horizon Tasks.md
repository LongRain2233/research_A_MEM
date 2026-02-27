# Optimus-1 : Hybrid Multimodal Memory

Zaijing Li1 2, Yuquan Xie1, Rui Shao1∗, Gongwei Chen1, Dongmei Jiang2, Liqiang Nie1∗

1Harbin Institute of Technology, Shenzhen

2Peng Cheng Laboratory

{lzj14011,xieyuquan20016,rshaojimmy,nieliqiang}@gmail.com

![](images/da42fd43e5ba7d833aba1f8e14ed4032ce6089dcf798586a2fe9f8ef61977190.jpg)  
Figure 1: An illustration of Optimus-1 performing long-horizon tasks in Minecraft. Given the task “Craft stone sword”, Knowledge-Guided Planner incorporates knowledge from Hierarchical Directed Knowledge Graph into planning, then Action Controller executes these planning sequences step-bystep. During the execution of the task, the Experience-Driven Reflector is periodically activated and retrieve experience from Abstracted Multimodal Experience Pool to make reflection.

# Abstract

Building a general-purpose agent is a long-standing vision in the field of artificial intelligence. Existing agents have made remarkable progress in many domains, yet they still struggle to complete long-horizon tasks in an open world. We attribute this to the lack of necessary world knowledge and multimodal experience that can guide agents through a variety of long-horizon tasks. In this paper, we propose a Hybrid Multimodal Memory module to address the above challenges. It 1) transforms knowledge into Hierarchical Directed Knowledge Graph that allows agents to explicitly represent and learn world knowledge, and 2) summarises historical information into Abstracted Multimodal Experience Pool that provide agents

with rich references for in-context learning. On top of the Hybrid Multimodal Memory module, a multimodal agent, Optimus-1, is constructed with dedicated Knowledge-guided Planner and Experience-Driven Reflector, contributing to a better planning and reflection in the face of long-horizon tasks in Minecraft. Extensive experimental results show that Optimus-1 significantly outperforms all existing agents on challenging long-horizon task benchmarks, and exhibits near humanlevel performance on many tasks. In addition, we introduce various Multimodal Large Language Models (MLLMs) as the backbone of Optimus-1. Experimental results show that Optimus-1 exhibits strong generalization with the help of the Hybrid Multimodal Memory module, outperforming the GPT-4V baseline on various tasks. Please see the project page at https://cybertronagent.github.io/Optimus-1.github.io/.

# 1 Introduction

Optimus Prime faces complex tasks alongside humans in Transformers to protect the peace of the planet. Creating an agent [43, 13] like Optimus that can perceive, plan, reflect, and complete long-horizon tasks in an open world has been a longstanding aspiration in the field of artificial intelligence [22, 35, 36]. Early research developed simple policy through reinforcement learning [7] or imitation learning [1, 25]. A lot of work [46, 49] have utilized Large Language Models (LLMs) as action planners for agents, generating executable sub-goal sequences for low-level action controllers. Further, recent studies [51, 32] employed Multimodal Large Language Models (MLLMs) [4, 38, 55] as planner and reflector. Leveraging the powerful instruction-following and logical reasoning capabilities of (Multimodal) LLMs [24], LLM-based agents have achieved remarkable success across multiple domains [14, 9, 10, 54]. Nevertheless, the ability of these agents to complete long-horizon tasks still falls significantly short of human-level performance.

According to relevant studies [27, 41, 45], the human ability to complete long-horizon tasks in an open world relies on long-term memory storage, which is divided into knowledge and experience. The storage and utilization of knowledge and experience play a crucial role in guiding human behavior and enabling humans to adapt flexibly to their environments in order to accomplish long-horizon tasks. Inspired by this theory, we summarize the challenges faced by current agents as follows:

Insufficient Exploration of Structured Knowledge: Structured knowledge, encompassing open world rules, object relationships, and interaction methods with the environment, is essential for agents to complete complex tasks [33, 43]. However, MLLMs such as GPT-4V * lack sufficient knowledge in Minecraft. Existing agents [1, 25, 7] only learn dispersed knowledge from video data and are unable to efficiently represent and learn this structured knowledge, rendering them incapable of performing complex tasks.

Lack of Multimodal Experience: Humans derive successful strategies and lessons from information on historical experience [8, 31], which assists them in tackling current complex tasks. In a similar manner, agents can benefit from in-context learning with experience demonstrations [42, 53]. However, existing agents [46, 50, 32] only consider unimodal information, which prevents them from learning from multimodal experience as humans do.

To address the aforementioned challenges, we propose Hybrid Multimodal Memory module that consists of Hierarchical Directed Knowledge Graph (HDKG) and Abstracted Multimodal Experience Pool (AMEP). For HDKG, we map the logical relationships between objects into a directed graph structure, thereby transforming knowledge into high-level semantic representations. HDKG efficiently provides the agent with the necessary knowledge for task execution, without requiring any parameter updates. For AMEP, we dynamically summarize and store the multimodal information (e.g., environment, agent state, task plan, video frames, etc.) from the agent’s task execution process, ensuring that historical information contains both a global overview and local details. Different from the method of directly storing successful cases as experience [51], AMEP considers both successful and failed cases as references. This innovative approach of incorporating failure cases into in-context learning significantly enhances the performance of the agent.

On top of the Hybrid Multimodal Memory module, we construct a multimodal composable agent, Optimus-1. As shown in Figure 1, Optimus-1 consists of Knowledge-Guided Planner, Experience-Driven Reflector, and Action Controller. To enhance the ability of agents to cope with complex environments and long-horizon tasks, Knowledge-Guided Planner incorporates visual observation into the planning phase, leveraging HDKG to capture the knowledge needed. This allows the agent to efficiently transform tasks into executable sub-goals. Action Controller takes the sub-goal and the current observation as inputs and generates low-level actions, interacting with the game environment to update the agent’s state. In open-world complex environments, agents are prone to be erroneous when performing long-horizon tasks. To address this, we propose Experience-Driven Reflector, which is periodically activated to retrieve relevant multimodal experiences from AMEP. This encourages the agent to reflect on its current actions and refine the plan.

We validate the performance of Optimus-1 in Minecraft, a popular open-world game environment. Experimental results show that Optimus-1 exhibits remarkable performance on long-horizon tasks, representing up to $3 0 \%$ improvement over existing agents. Moreover, we introduce various Multimodal Large Language Models (MLLMs) as the backbone of Optimus-1. Experimental results show that Optimus-1 has a 2 to 6 times performance improvement with the help of Hybrid Multimodal Memory, outperforming powerful GPT-4V baseline on lots of long-horizon tasks. Additionally, we verified that the plug-and-play Hybrid Multimodal Memory can drive Optimus-1 to incrementally improve its performance in a self-evolution manner. The extensive experimental results show that Optimus-1 makes a major step toward a general agent with a human-like level of performance. Main contributions of our paper:

• We propose Hybrid Multimodal Memory module which is composed of HDKG and AMEP. HDKG helps the agent make the planning of long-horizon tasks efficiently. AMEP provides refined historical experience and guides the agent to reason about the current situation state effectively.   
• On top of the Hybrid Multimodal Memory module, we construct Optimus-1, which consists of Knowledge-Guided Planner, Experience-Driven Reflector, and Action Controller. Optimus-1 outperforms all baseline agents on long-horizon task benchmarks, and exhibits capabilities close to the level of human players.   
• Driven by Hybrid Multimodal Memory, various MLLM-based Optimus-1 have demonstrated 2 to 6 times performance improvement, demonstrating the generalization of Hybrid Multimodal Memory.

# 2 Optimus-1

In this section, we first elaborate on how to implement the Hybrid Multimodal Memory in Sec 2.1. As a core innovation, it plays a crucial role in enabling Optimus-1 to execute long-horizon tasks. Next, we give an overview of Optimus-1 framework (Sec 2.2), which consists of Hybrid Multimodal Memory, Knowledge-Guided Planner, Experience-Driven Reflector, and Action Controller. Finally, we introduce a non-parametric learning approach to expand the hybrid multimodal memory (Sec 2.3), thereby enhancing the success rate of task execution for Optimus-1.

# 2.1 Hybrid Multimodal Memory

In order to endow agent with a long-term memory storage mechanism [27, 45], we propose the Hybrid Multimodal Memory module, which consists of Abstracted Multimodal Experience Pool (AMEP) and Hierarchical Directed Knowledge Graph (HDKG).

# 2.1.1 Abstracted Multimodal Experience Pool

Relevant studies [23, 28, 17, 15] highlight the importance of historical information for agents completing long-horizon tasks. Minedojo [7] and Voyager [46] employed unimodal storage of historical information. Jarvis-1 [51] used a multimodal experience mechanism that stores task planning and visual information without summarization, posing challenges to storage capacity and retrieval speed. To address this issue, we propose AMEP, which aims to dynamically summarize all multimodal information during task execution. It preserves the integrity of long-horizon data while enhancing storage and retrieval efficiency.

![](images/17250fd19b1386dbe86b5de524de215090242690a9f0009a0c40475fa4c34f4f.jpg)  
(a)

![](images/4cfdb8e05d8ebf55be049c644f70a0fa00cfb29c85ba2db8f4b3c9ea8be94d2e.jpg)  
(b)   
Figure 2: (a) Extraction process of multimodal experience. The frames are filtered through video buffer and image buffer, then MineCLIP [7] is employed to compute the visual and sub-goal similarities and finally they are stored in Abstracted Multimodal Experience Pool. (b) Overview of Hierarchical Directed Knowledge Graph. Knowledge is stored as a directed graph, where its nodes represent objects, and directed edges point to materials that can be crafted by this object.

Specifically, as depicted in Figure 2, to conduct the static visual information abstraction, the video stream captured by Optimus-1 during task execution is first input to a video buffer, filtering the stream at a fixed frequency of 1 frame per second. Based on the filtered video frames, to further perform a dynamic visual information abstraction, these frames are then fed into an image buffer with a window size of 16, where the image similarity is dynamically computed and final abstracted frames are adaptively updated. To align such abstracted visual information with the corresponding textual sub-goal, we then utilize MineCLIP [7], a pre-trained video-text alignment model, to calculate their multimodal correlation. When this correlation exceeds a threshold, the corresponding image buffer and textual sub-goal are saved as multimodal experience into a pool. Finally, we further incorporate environment information, agent initial state, and plan generated by Knowledge-Guided Planner, into such a pool, which forms the AMEP. In this way, we consider the multimodal information of each sub-goal, and summarise it to finally compose the multimodal experience of the given task.

# 2.1.2 Hierarchical Directed Knowledge Graph

In Minecraft, mining and crafting represent a complex knowledge network crucial for effective task planning. For instance, crafting a diamond sword $\checkmark$ requires two diamonds $\circledcirc$ and one wooden stick $\diagup$ , while mining diamonds requires an iron pickaxe $\mathcal { \bar { A } }$ , which involving further materials and steps. Such knowledge is essential for an agent’s ability to perform long-horizon complex tasks. Instead of implicit learning through fine-tuning [32, 58], we propose HDKG, which transforms knowledge into a graph representation. It enables the agent to perform explicit learning by retrieving information from the knowledge graph.

As shown in the Figure 2, we transform knowledge into a graph $\mathcal { D } ( \boldsymbol { \nu } , \boldsymbol { \mathcal { E } } )$ , where nodes set $\nu$ represent objects, and directed edges set $\mathcal { E }$ point to nodes that can be crafted by this object. An edge $e \in { \mathcal { E } }$ in the $\mathcal { D }$ can be represented as $\boldsymbol { e } = \left( u , v \right)$ , where $u , v \in \mathcal { V }$ . The directed graph efficiently stores and updates knowledge. For a given object $x$ , retrieving the corresponding node allows extraction of a sub-graph $\mathcal { D } _ { j } ( \mathcal { V } _ { j } , \mathcal { E } _ { j } ) \in \mathcal { D }$ , where nodes set $\nu _ { j }$ and edges set $\mathcal { E } _ { j }$ can be formulated as:

$$
\mathcal {V} _ {j} = \{v \in \mathcal {V} \mid x \}, \quad \mathcal {E} _ {j} = \left\{e = (u, v) \in \mathcal {V} \mid u \in \mathcal {V} _ {j} \cup v \in \mathcal {V} _ {j} \right\}, \tag {1}
$$

Then by topological sorting, we can get all the materials and their relationships needed to complete the task. This knowledge is provided to the Knowledge-Guided Planner as a way to generate a more

![](images/3f4df0e7a178eca4a8129a956b1a7c2ed13f463fc48e3c9218137453aba9fcdc.jpg)  
Figure 3: Overview framework of our Optimus-1. Optimus-1 consists of Knowledge-Guided Planner, Experience-Driven Reflector, Action Controller, and Hybrid Multimodal Memory architecture. Given the task “craft stone sword”, Optimus-1 incorporates the knowledge from HDKG into Knowledge-Guided Planning, then Action Controller generates low-level actions. Experience-Driven Reflector is periodically activated to introduce multimodal experience from AMEP to determine if the current task can be executed successfully. If not, it will ask the Knowledge-Guided Planner to refine the plan.

reasonable sequence of sub-goals. With HDKG, we can significantly enhance the world knowledge of the agent in a train-free manner.

# 2.2 Optimus-1: Framework

Relevant studies indicate that the human brain is essential for planning and reflection, while the cerebellum controls low-level actions, both crucial for complex tasks [39, 40]. Inspired by this, we divide the structure of Optimus-1 into Knowledge-Guided Planner, Experience-Driven Reflector, and Action Controller. In a given game environment with a long-horizon task, the Knowledge-Guided Planner senses the environment, retrieves knowledge from HDKG, and decomposes the task into executable sub-goals. The action controller then sequentially executes these sub-goals. During execution, the Experience-Driven Reflector is activated periodically, leveraging historical experience from AMEP to assess whether Optimus-1 can complete the current sub-goal. If not, it instructs the Knowledge-Guided Planner to revise its plan. Through iterative interaction with the environment, Optimus-1 ultimately completes the task.

Knowledge-Guided Planner. Open-world environments vary greatly, affecting task execution. Previous approaches [50] using LLMs for task planning failed to consider the environment, leading to the failure of tasks. For example, an agent in a cave aims to catch fish. It lacks visual information to plan conditions on the current situation, such as “leave the cave and find a river”. Therefore, we integrate environmental information into the planning stage. Unlike Jarvis-1 [51] and MP5 [32], which convert observation to textual descriptions, Optimus-1 directly employs observation as visual conditions to generate environment-related plans, i.e., sub-goal sequences. This results in more comprehensive and reasonable planning. More importantly, Knowledge-Guided Planner retrieves the knowledge needed to complete the task from HDKG, allowing task planning to be done once, rather than generating the next step in each iteration. Given the task $t$ , observation $o$ , the sub-goals

sequence $g _ { 1 } , g _ { 2 } , g _ { 3 } , . . . , g _ { n }$ can be formulated as:

$$
g _ {1}, g _ {2}, g _ {3}, \dots , g _ {n} = p _ {\theta} (o, t, p _ {\eta} (t)), \tag {2}
$$

where $n$ is the number of sub-goals, $p _ { \eta }$ denotes sub-graph retrieved from HDKG, $p _ { \theta }$ denotes MLLM. In this paper, we employ OpenAI’s GPT-4V as Knowledge-Guided Planner and Experience-Driven Reflector. We also evaluate other alternatives of GPT-4V, such as open-source models like Deepseek-VL [26] and InternLM-XComposer2-VL [6] in Section 3.4.

Action Controller. It takes the sub-goal and the current observation as inputs and then generates low-level actions, which are control signals for the mouse and keyboard. Thus, it can interact with the game environment to update the agent’s state and the observation. The formulation is as follows:

$$
a _ {k} = p _ {\pi} (o, g _ {i}), \tag {3}
$$

where $a _ { k }$ denotes low-level action at time $k$ , $p _ { \pi }$ denotes action controller. Unlike generating code [46, 32, 49], generating control actions for the mouse and keyboard [1, 25, 51, 3] more closely resembles human behavior. In this paper, we employ STEVE-1 [25] as our Action Controller.

Experience-Driven Reflector. The sub-goals generated by Knowledge-Guided Planner are interdependent. The failure of any sub-goal halts the execution of subsequent ones, leading to overall task failure. Therefore, a reflection module is essential to identify and rectify errors promptly. During task execution, the Experience-Driven Reflector activates at regular intervals, retrieving historical experience from AMEP, and then analyzing the current state of Optimus-1. The reflection results of Optimus-1 are categorized as COMPLETE, CONTINUE, or REPLAN. COMPLETE indicates successful execution, prompting the action controller to proceed to the next sub-goal. CONTINUE signifies ongoing execution without additional feedback. REPLAN denotes failure, requiring the Knowledge-Guided Planner to revise the plan. The reflection $r$ generated by Experience-Driven Reflector can be formulated as:

$$
r = p _ {\theta} \left(o, g _ {i}, p _ {\epsilon} (t)\right), \tag {4}
$$

where $p _ { \epsilon }$ denotes multimodal experience retrieved from AMEP. Experimental results in Section 3.3 demonstrate that the Experience-Driven Reflector significantly enhances the success rate of long-horizon tasks.

During task execution, even in cases where task failure necessitates REPLAN, multimodal experiences are stored in AMEP. Thus, during the reflection phase, Optimus-1 can retrieve the most relevant cases from each of the three scenarios COMPLETE, CONTINUE, and REPLAN from AMEP as references. Experimental Results in Section 3.3 demonstrate the effectiveness of this innovative method of incorporating failure cases into in-context learning.

# 2.3 Non-parametric Learning of Hybrid Multimodal Memory

To implement the Hybrid Multimodal Memory and enhance Optimus-1’s capacity, we propose a nonparametric learning method named “free exploration-teacher guidance”. In the free exploration phase, Optimus-1’s equipment and tasks are randomly initialized, and it explores random environments, acquiring world knowledge through environmental feedback. For example, it learns that “a stone sword $\mathcal { A }$ can be crafted with a wooden stick $\diagup$ and two cobblestones ”, storing this in the HDKG. Additionally, successful and failed cases are stored in the AMEP, providing reference experience for the reflection phase. We initialize multiple Optimus-1, and they share the same HDKG and AMEP. Thus the memory is filled up efficiently. After free exploration, Optimus-1 has basic world knowledge and multimodal experience. In the teacher guidance phase, Optimus-1 needs to learn a small number of long-horizon tasks based on extra knowledge. For example, it learns “a diamond sword $\pmb { x }$ is obtained by a stick $\diagup$ and two diamonds ” from the teacher, then perform the task “craft diamond sword”. During the teacher guidance phase, Optimus-1’s memory is further expanded and it gains the experience of executing complete long-horizon tasks.

Unlike fine-tuning, this method enhances Optimus-1 incrementally without updating parameters, in a self-evolution manner. Starting with an empty Hybrid Multimodal Memory, Optimus-1 iterates between “free exploration-teacher guidance” learning and unseen task inference. With each iteration, its memory capacity grows, enabling mastery of tasks from easy to hard.

Table 1: Main Result of Optimus-1 on long-horizon tasks benchmark. We report the average success rate (SR), average number of steps (AS), and average time (AT) on each task group, the results of each task can be found in the Appendix F. Lower AS and AT metrics mean that the agent is more efficient at completing the task, while $+ \infty$ indicates that the agent is unable to complete the task. Overall represents the average result on the five groups of Iron, Gold, Diamond, Redstone, and Armor.   

<table><tr><td>Group</td><td>Metric</td><td>GPT-3.5</td><td>GPT-4V</td><td>DEPS</td><td>Jarvis-1</td><td>Optimus-1</td><td>Human-level</td></tr><tr><td rowspan="3">Wood</td><td>SR ↑</td><td>40.16</td><td>41.42</td><td>77.01</td><td>93.76</td><td>98.60</td><td>100.00</td></tr><tr><td>AT ↓</td><td>56.39</td><td>55.15</td><td>85.53</td><td>67.76</td><td>47.09</td><td>31.08</td></tr><tr><td>AS ↓</td><td>1127.78</td><td>1103.04</td><td>1710.61</td><td>1355.25</td><td>841.94</td><td>621.59</td></tr><tr><td rowspan="3">Stone</td><td>SR ↑</td><td>20.40</td><td>20.89</td><td>48.52</td><td>89.20</td><td>92.35</td><td>100.00</td></tr><tr><td>AT ↓</td><td>135.71</td><td>132.77</td><td>138.71</td><td>141.50</td><td>129.94</td><td>80.85</td></tr><tr><td>AS ↓</td><td>2714.21</td><td>2655.47</td><td>2574.30</td><td>2830.05</td><td>2518.88</td><td>1617.00</td></tr><tr><td rowspan="3">Iron</td><td>SR ↑</td><td>0.00</td><td>0.00</td><td>16.37</td><td>36.15</td><td>46.69</td><td>86.00</td></tr><tr><td>AT ↓</td><td>+∞</td><td>+∞</td><td>944.61</td><td>722.78</td><td>651.33</td><td>434.38</td></tr><tr><td>AS ↓</td><td>+∞</td><td>+∞</td><td>8892.24</td><td>8455.51</td><td>6017.85</td><td>5687.60</td></tr><tr><td rowspan="3">Gold</td><td>SR ↑</td><td>0.00</td><td>0.00</td><td>0.00</td><td>7.20</td><td>8.51</td><td>17.31</td></tr><tr><td>AT ↓</td><td>+∞</td><td>+∞</td><td>+∞</td><td>787.37</td><td>726.35</td><td>557.08</td></tr><tr><td>AS ↓</td><td>+∞</td><td>+∞</td><td>+∞</td><td>15747.13</td><td>15527.07</td><td>13141.60</td></tr><tr><td rowspan="3">Diamond</td><td>SR ↑</td><td>0.00</td><td>0.00</td><td>0.60</td><td>8.98</td><td>11.61</td><td>16.98</td></tr><tr><td>AT ↓</td><td>+∞</td><td>+∞</td><td>1296.96</td><td>1255.06</td><td>1150.98</td><td>744.82</td></tr><tr><td>AS ↓</td><td>+∞</td><td>+∞</td><td>23939.30</td><td>25101.25</td><td>23019.64</td><td>16237.54</td></tr><tr><td rowspan="3">Redstone</td><td>SR ↑</td><td>0.00</td><td>0.00</td><td>0.00</td><td>16.31</td><td>25.02</td><td>33.27</td></tr><tr><td>AT ↓</td><td>+∞</td><td>+∞</td><td>+∞</td><td>1070.42</td><td>932.50</td><td>617.89</td></tr><tr><td>AS ↓</td><td>+∞</td><td>+∞</td><td>+∞</td><td>17408.40</td><td>12709.99</td><td>12357.00</td></tr><tr><td rowspan="3">Armor</td><td>SR ↑</td><td>0.00</td><td>0.00</td><td>9.98</td><td>15.82</td><td>19.47</td><td>28.48</td></tr><tr><td>AT ↓</td><td>+∞</td><td>+∞</td><td>997.59</td><td>924.60</td><td>824.53</td><td>551.30</td></tr><tr><td>AS ↓</td><td>+∞</td><td>+∞</td><td>17951.95</td><td>16492.96</td><td>16350.56</td><td>11026.00</td></tr><tr><td>Overall</td><td>SR ↑</td><td>0.00</td><td>0.00</td><td>5.39</td><td>16.89</td><td>22.26</td><td>36.41</td></tr></table>

# 3 Experiments

# 3.1 Experiments Setting

Environment. To ensure realistic gameplay like human players, we employ MineRL [11] with Minecraft 1.16.5 as our simulation environment. The agent operates at a fixed speed of 20 frames per second and only interacts with the environment via low-level action control signals of the mouse and keyboard. For more information about the detailed descriptions of the observation and action spaces, please refer to the Appendix B.

Benchmark. We constructed a benchmark of 67 tasks to evaluate the Optimus-1’s ability to complete long-horizon tasks. As illustrated in Table 5, we divide the 67 Minecraft tasks into 7 groups according to recommended categories in Minecraft. Please refer to Appendix D for more details.

Baseline. We compare Optimus-1 with various agents, including GPT-3.5 2, GPT-4V, DEPS [50], and Jarvis-1 [51] on the challenging long-horizon tasks benchmark. In addition, we employed 10 volunteers to perform the same task on the benchmark, and their average performance served as a human-level baseline. Please refer to Appendix D.2 for more details about human-level baseline. For a more comprehensive comparison, we also report Optimus-1’s performances on the benchmark used by Voyager [46], MP5 [32], and DEPS [50] in the Appendix F.2. Note that we initialize Optimus-1 with an empty inventory, while DEPS [50] and Jarvis-1 [51] have tools in their initial state. This makes it more challenging for Optimus-1 to perform the same tasks.

Table 2: Ablation study results. We report average Table 3: Ablation study on AMEP. We report success rate (SR) on each task group. P., R., K.,the average success rate (SR) on each task group. E. represent Planning, Reflection, Knowledge, and Zero, Suc., and Fail. represent retrieving from Experience, respectively. AMEP without getting the case, getting the success   

<table><tr><td colspan="4">Ablation Setting</td><td colspan="5">Task Group</td></tr><tr><td>P.</td><td>R.</td><td>K.</td><td>E.</td><td>Wood</td><td>Stone</td><td>Iron</td><td>Gold</td><td>Diamond</td></tr><tr><td></td><td></td><td></td><td></td><td>14.29</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>✓</td><td></td><td></td><td></td><td>42.95</td><td>25.67</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>✓</td><td>✓</td><td></td><td></td><td>55.00</td><td>47.37</td><td>18.11</td><td>2.08</td><td>1.11</td></tr><tr><td>✓</td><td>✓</td><td></td><td>✓</td><td>73.53</td><td>64.20</td><td>24.19</td><td>3.08</td><td>1.86</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td></td><td>92.37</td><td>69.63</td><td>38.33</td><td>3.49</td><td>2.42</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>97.49</td><td>94.26</td><td>53.33</td><td>11.54</td><td>9.59</td></tr></table>

<table><tr><td colspan="3">Ablation Setting</td><td colspan="5">Task Group</td></tr><tr><td>Zero</td><td>Suc.</td><td>Fai.</td><td>Wood</td><td>Stone</td><td>Iron</td><td>Gold</td><td>Diamond</td></tr><tr><td>✓</td><td></td><td></td><td>92.00</td><td>79.26</td><td>36.32</td><td>4.25</td><td>3.25</td></tr><tr><td></td><td>✓</td><td></td><td>95.00</td><td>84.29</td><td>46.98</td><td>9.36</td><td>7.89</td></tr><tr><td></td><td></td><td>✓</td><td>95.00</td><td>81.10</td><td>45.47</td><td>7.50</td><td>6.39</td></tr><tr><td></td><td>✓</td><td>✓</td><td>97.49</td><td>94.26</td><td>53.33</td><td>11.54</td><td>9.59</td></tr></table>

![](images/37546038134de57c63e4a89b2a648d290adc68f5a1789cb16a0539d6973046a5.jpg)  
Figure 4: Illustration of the role of reflection mechanism. Without the help of reflective mechanisms, STEVE-1 [25] often gets into trouble and fails to complete the task. While Optimus-1, with the help of the Experience-Driven Reflector, leverages the AMEP to retrieve relevant experience, reflect current situation and correct errors. This improves Optimus-1’s success rate on long-horizon tasks.

Evaluation Metrics. The agent always starts in survival mode, with an empty inventory. We conducted at least 30 times for each task using different world seeds and reported the average success rate to ensure fair and thorough evaluation. Additionally, we add the average steps and average time of completing the task as evaluation metrics.

# 3.2 Experimental Results

The overall experimental results on benchmark are shown in Table 1, see the accuracy for each task in Appendix F. Optimus-1 has a success rate near $1 0 0 \%$ on the Wood Group . Compared with Jarvis-1, Optimus-1 has $\bar { 2 9 . 2 8 \% }$ and $5 3 . 4 0 \%$ improvement on the Diamond Group $\circledcirc$ and Redstone Group , respectively. Optimus-1 achieves the best performance and the shortest elapsed time among all task groups. It reveals the effectiveness and efficiency of our proposed Optimus-1 framework. Moreover, compared with all baselines, Optimus-1 performance was closer (average $5 . 3 7 \%$ improvement) to human levels on long-horizon task groups.

# 3.3 Ablation Study

We conduct extensive ablation experiments on 18 tasks, experiment setting can be found in Table 6. As shown in Table 2, we first remove Knowledge-Driven Planner and Experience-Driven Reflector, the performance of Optimus-1 on all task groups drops dramatically. It demonstrates the necessity of Knowledge-Guided Planner and Experience-Driven Reflector modules for performing long-horizon tasks. As for Hybrid Multimodal Memory, we remove HDKG from Optimus-1. Without the help of world knowledge, the performance of Optimus-1 decreased by an average of $2 0 \%$ across all task groups. We then removed AMEP, this resulted in the performance of Optimus-1 decreased by an

![](images/92d1bd8f045b0d083332f9d98fa0af58c1689c41f51b961db1aa3abd5cb93228.jpg)  
Figure 5: (a) With the help of Hybrid Multimodal Memory, various MLLM-based Optimus-1 have demonstrated 2 to 6 times performance improvement. (b) Illustration of the change in Optimus-1 success rate on the unseen task over 4 epochs.

average of $1 2 \%$ . Finally, we performed ablation experiments on the way of retrieving cases from AMEP. As shown in Table 3, without retrieving cases from AMEP, the success rate shows an average of $1 0 \%$ decrease across all groups. It reveals that this reflection mechanism, which considers both success and failure cases, has a significant impact on the performance of Optimus-1. To illustrate the role of the reflection mechanism, we have shown some cases in Figure 4.

# 3.4 Generalization Ability

In this section, we explore an interesting issue: whether generic MLLMs can effectively perform various long-horizon complex tasks in Minecraft using Hybrid Multimodal Memory. As shown in Figure 5, We employ Deepseek-VL [26] and InternLM-XComposer2-VL [6] as Knowledge-Guided Planner and Experience-Driven Reflector. The experimental results show that the original MLLM has low performance on long-horizon tasks due to the lack of knowledge and experience of Minecraft. With the assistance of Hybrid Multimodal Memory, the performance of MLLMs has improved by 2 to 6 times across various task groups, outperforming the GPT-4V baseline. This encouraging result demonstrates the generalization of the proposed Hybrid Multimodal Memory.

# 3.5 Self-Evolution via Hybrid Multimodal Memory

As shown in Section 2.3, we randomly initialize the Hybrid Multimodal Memory of Optimus-1, then update it multiple times by using the “free exploration-teacher guidance” learning method. We set the epoch to 4, and the number of learning tasks to 160. At each period, Optimus-1 performs free exploration on 150 tasks and teacher guidance learning on the remaining 10 tasks, we then evaluate Optimus-1’s learning ability on the task groups same as ablation study. Experimental results are shown in Figure 5. It reveals that Optimus-1 keeps getting stronger through the continuous expansion of memory during the learning process of multiple periods. Moreover, it demonstrates that MLLM with Hybrid Multimodal Memory can incarnate an expert agent in a self-evolution manner [44].

# 4 Related Work

# 4.1 Agents in Minecraft

We summarise the differences of existing Minecraft agents in the Appendix D.3. Earlier work [29, 56, 2, 3] introduced policy models for agents to perform simple tasks in Minecraft. MineCLIP [7] used text-video data to train a contrastive video-language model as a reward model for policy, while VPT [1] pre-trained on unlabelled videos but lacked instruction as input. Building on VPT and MineCLIP, STEVE-1 [25] added text input to generate low-level action sequences from human instructions and images. However, these agents struggle with complex tasks due to limitations in instruction comprehension and planning. Recent work [49, 46, 59] incorporated LLMs as planning and reflection modules, but lacked visual information integration for adaptive planning. MP5

[32], MineDreamer [58], and Jarvis-1 [51] enhanced situation-aware planning by obtaining textual descriptions of visual information, yet lacked detailed visual data. Optimus-1 addresses these issues by directly using observation as situation-aware conditions in the planning phase, enabling more rational, visually informed planning. Additionally, unlike other agents requiring multiple queries for task refinement, Optimus-1 generates a complete and effective plan in one step with the help of HDKG. This makes Optimus-1 planning more efficient.

# 4.2 Memory in Agents

In the agent-environment interaction process, memory is key to achieving experience accumulation [21], environment exploration [16], and knowledge abstraction [57]. There are two forms to represent memory content in LLM-based agents: textual form [17, 15, 30] and parametric form [5, 28, 47, 20]. In textual form, the information is explicitly retained and recalled by natural languages. In parametric form, the memory information [37] is encoded into parameters and implicitly influences the agent’s actions. Recent work [48, 52, 12] has explored the long-term visual information storage [18, 19] and summarisation in MLLM. Our proposed hybrid multimodal memory module is plug-and-play and can provide world knowledge and multimodal experience for Optimus-1 efficiently.

# 5 Conclusion

In this paper, we propose Hybrid Multimodal Memory module, which consists of two parts: HDKG and AMEP. HDKG provides the necessary world knowledge for the planning phase of the agent, and AMEP provides the refined historical experience for the reflection phase of the agent. On top of the Hybrid Multimodal Memory, we construct the multimodal composable agent, Optimus-1, in Minecraft. Extensive experimental results show that Optimus-1 outperforms all existing agents on long-horizon tasks. Moreover, we validate that general-purpose MLLMs, based on Hybrid Multimodal Memory and without additional parameter updates, can exceed the powerful GPT-4V baseline. The extensive experimental results show that Optimus-1 makes a major step toward a general agent with a human-like level of performance.

# 6 Limitation and Future Work

In the framework of Optimus-1, we are dedicated to leverage proposed Hierarchical Directed Knowledge Graph and Abstracted Multimodal Experience Pool can be used to enhance the agent’s ability to plan and reflect. For Action Controller, we directly introduce STEVE-1 [25] as a generator of low-level actions. However, limited by STEVE-1’s ability to follow instructions and execute complex actions, Optimus-1 is weak in completing challenging tasks such as “beat ender dragon” and “build a house”. Therefore, a potential future research direction is to enhance the instruction following and action generation capabilities of action controller.

In addition, most of the work, including Optimus-1, utilize a multimodal large language model for planning and reflection, which then drives an action controller to perform the task. Building an end-to-end vision-language-action agent will be future work.

# 7 Acknowledgement

This study is supported by National Natural Science Foundation of China (Grant No. 62236003 and 62306090), Shenzhen College Stability Support Plan (Grant No. GXWD20220817144428005), Natural Science Foundation of Guangdong Province of China (Grant No. 2024A1515010147), and Major Key Project of Peng Cheng Laboratory (Grant No. PCL2023A08).

# References

[1] Bowen Baker, Ilge Akkaya, Peter Zhokov, Joost Huizinga, Jie Tang, Adrien Ecoffet, Brandon Houghton, Raul Sampedro, and Jeff Clune. Video pretraining (vpt): Learning to act by watching unlabeled online videos. Advances in Neural Information Processing Systems, 35:24639–24654, 2022.   
[2] Shaofei Cai, Zihao Wang, Xiaojian Ma, Anji Liu, and Yitao Liang. Open-world multi-task control through goal-aware representation learning and adaptive horizon prediction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13734–13744, 2023.   
[3] Shaofei Cai, Bowei Zhang, Zihao Wang, Xiaojian Ma, Anji Liu, and Yitao Liang. Groot: Learning to follow instructions by watching gameplay videos. In The Twelfth International Conference on Learning Representations, 2023.   
[4] Gongwei Chen, Leyang Shen, Rui Shao, Xiang Deng, and Liqiang Nie. Lion: Empowering multimodal large language model with dual-level visual knowledge. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.   
[5] Nicola De Cao, Wilker Aziz, and Ivan Titov. Editing factual knowledge in language models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6491–6506, 2021.   
[6] Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Bin Wang, Linke Ouyang, Xilin Wei, Songyang Zhang, Haodong Duan, Maosong Cao, et al. Internlm-xcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model. arXiv preprint arXiv:2401.16420, 2024.   
[7] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. Advances in Neural Information Processing Systems, 35:18343–18362, 2022.   
[8] Mariel K Goddu and Alison Gopnik. The development of human causal learning and reasoning. Nature Reviews Psychology, pages 1–21, 2024.   
[9] Maitrey Gramopadhye and Daniel Szafir. Generating executable action plans with environmentally-aware language models. In 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3568–3575. IEEE, 2023.   
[10] Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. A real-world webagent with planning, long context understanding, and program synthesis. arXiv preprint arXiv:2307.12856, 2023.   
[11] William H Guss, Brandon Houghton, Nicholay Topin, Phillip Wang, Cayden Codel, Manuela Veloso, and Ruslan Salakhutdinov. Minerl: A large-scale dataset of minecraft demonstrations. arXiv preprint arXiv:1907.13440, 2019.   
[12] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. arXiv preprint arXiv:2404.05726, 2024.   
[13] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. arXiv preprint arXiv:2311.12871, 2023.   
[14] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In International Conference on Machine Learning, pages 9118–9147. PMLR, 2022.   
[15] Ziheng Huang, Sebastian Gutierrez, Hemanth Kamana, and Stephen MacNeil. Memory sandbox: Transparent and interactive memory management for conversational agents. In Adjunct Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology, pages 1–3, 2023.

[16] Yunfan Jiang, Agrim Gupta, Zichen Zhang, Guanzhi Wang, Yongqiang Dou, Yanjun Chen, Li Fei-Fei, Anima Anandkumar, Yuke Zhu, and Linxi Fan. Vima: General robot manipulation with multimodal prompts. In NeurIPS 2022 Foundation Models for Decision Making Workshop, 2022.   
[17] Dacheng Li, Rulin Shao, Anze Xie, Ying Sheng, Lianmin Zheng, Joseph Gonzalez, Ion Stoica, Xuezhe Ma, and Hao Zhang. How long can context length of open-source llms truly promise? In NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following, 2023.   
[18] Xiaojie Li, Shaowei He, Jianlong Wu, Yue Yu, Liqiang Nie, and Min Zhang. Mask again: Masked knowledge distillation for masked video modeling. In Proceedings of the ACM International Conference on Multimedia, page 2221–2232. ACM, 2023.   
[19] Xiaojie Li, Jianlong Wu, Shaowei He, Shuo Kang, Yue Yu, Liqiang Nie, and Min Zhang. Finegrained key-value memory enhanced predictor for video representation learning. In Proceedings of the ACM International Conference on Multimedia, page 2264–2274. ACM, 2023.   
[20] Xiaojie Li, Yibo Yang, Xiangtai Li, Jianlong Wu, Yue Yu, Bernard Ghanem, and Min Zhang. Genview: Enhancing view quality with pretrained generative model for self-supervised learning. In Proceedings of the European Conference on Computer Vision. Springer, 2024.   
[21] Xiaojie Li, Yibo Yang, Jianlong Wu, Bernard Ghanem, Liqiang Nie, and Min Zhang. Mambafscil: Dynamic adaptation with selective state space model for few-shot class-incremental learning. arXiv preprint arXiv:2407.06136, 2024.   
[22] Zaijing Li, Fengxiao Tang, Ming Zhao, and Yusen Zhu. Emocaps: Emotion capsule based model for conversational emotion recognition. In Findings of the Association for Computational Linguistics: ACL 2022, pages 1610–1618, 2022.   
[23] Zaijing Li, Ting-En Lin, Yuchuan Wu, Meng Liu, Fengxiao Tang, Ming Zhao, and Yongbin Li. Unisa: Unified generative framework for sentiment analysis. In Proceedings of the 31st ACM International Conference on Multimedia, pages 6132–6142, 2023.   
[24] Zaijing Li, Gongwei Chen, Rui Shao, Dongmei Jiang, and Liqiang Nie. Enhancing the emotional generation capability of large language models via emotional chain-of-thought. arXiv preprint arXiv:2401.06836, 2024.   
[25] Shalev Lifshitz, Keiran Paster, Harris Chan, Jimmy Ba, and Sheila McIlraith. Steve-1: A generative model for text-to-behavior in minecraft. Advances in Neural Information Processing Systems, 2023.   
[26] Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Yaofeng Sun, et al. Deepseek-vl: towards real-world vision-language understanding. arXiv preprint arXiv:2403.05525, 2024.   
[27] Simon Makin. The amyloid hypothesis on trial. Nature, 559(7715):S4–S4, 2018.   
[28] Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D Manning. Fast model editing at scale. In International Conference on Learning Representations, 2021.   
[29] Junhyuk Oh, Satinder Singh, Honglak Lee, and Pushmeet Kohli. Zero-shot task generalization with multi-task deep reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning, pages 2661–2670. PMLR, 2017.   
[30] Arka Pal, Deep Karkhanis, Manley Roberts, Samuel Dooley, Arvind Sundararajan, and Siddartha Naidu. Giraffe: Adventures in expanding context lengths in llms. arXiv preprint arXiv:2308.10882, 2023.   
[31] Eileen Parkes. Scientific progress is built on failure. Nature, 10, 2019.   
[32] Yiran Qin, Enshen Zhou, Qichang Liu, Zhenfei Yin, Lu Sheng, Ruimao Zhang, Yu Qiao, and Jing Shao. Mp5: A multi-modal open-ended embodied system in minecraft via active perception. arXiv preprint arXiv:2312.07472, 2023.

[33] Maria Abi Raad, Arun Ahuja, Catarina Barros, Frederic Besse, Andrew Bolt, Adrian Bolton, Bethanie Brownfield, Gavin Buttimore, Max Cant, Sarah Chakera, et al. Scaling instructable agents across many simulated worlds. arXiv preprint arXiv:2404.10179, 2024.   
[34] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.   
[35] Rui Shao, Xiangyuan Lan, Jiawei Li, and Pong C Yuen. Multi-adversarial discriminative deep domain generalization for face presentation attack detection. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10023–10031, 2019.   
[36] Rui Shao, Tianxing Wu, and Ziwei Liu. Detecting and grounding multi-modal media manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6904–6913, 2023.   
[37] Rui Shao, Tianxing Wu, Jianlong Wu, Liqiang Nie, and Ziwei Liu. Detecting and grounding multi-modal media manipulation and beyond. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.   
[38] Leyang Shen, Gongwei Chen, Rui Shao, Weili Guan, and Liqiang Nie. Mome: Mixture of multimodal experts for generalist multimodal large language models. In NeurIPS, 2024.   
[39] Shan H Siddiqi, Konrad P Kording, Josef Parvizi, and Michael D Fox. Causal mapping of human brain function. Nature reviews neuroscience, pages 361–375, 2022.   
[40] JF Stein. Role of the cerebellum in the visual guidance of movement. Nature, pages 217–221, 1986.   
[41] Tim Stuart, Andrew Butler, Paul Hoffman, Christoph Hafemeister, Efthymia Papalexi, William M Mauck, Yuhan Hao, Marlon Stoeckius, Peter Smibert, and Rahul Satija. Comprehensive integration of single-cell data. cell, 177(7):1888–1902, 2019.   
[42] Quan Sun, Yufeng Cui, Xiaosong Zhang, Fan Zhang, Qiying Yu, Zhengxiong Luo, Yueze Wang, Yongming Rao, Jingjing Liu, Tiejun Huang, et al. Generative multimodal models are in-context learners. arXiv preprint arXiv:2312.13286, 2023.   
[43] Weihao Tan, Ziluo Ding, Wentao Zhang, Boyu Li, Bohan Zhou, Junpeng Yue, Haochong Xia, Jiechuan Jiang, Longtao Zheng, Xinrun Xu, et al. Towards general computer control: A multimodal agent for red dead redemption ii as a case study. arXiv preprint arXiv:2403.03186, 2024.   
[44] Zhengwei Tao, Ting-En Lin, Xiancai Chen, Hangyu Li, Yuchuan Wu, Yongbin Li, Zhi Jin, Fei Huang, Dacheng Tao, and Jingren Zhou. A survey on self-evolution of large language models. arXiv preprint arXiv:2404.14387, 2024.   
[45] Deniz Vatansever, Jonathan Smallwood, and Elizabeth Jefferies. Varying demands for cognitive control reveals shared neural processes supporting semantic and episodic memory retrieval. Nature communications, 12(1):2134, 2021.   
[46] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.   
[47] Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, and Huajun Chen. Detoxifying large language models via knowledge editing. arXiv preprint arXiv:2403.14472, 2024.   
[48] Shihao Wang, Zhiding Yu, Xiaohui Jiang, Shiyi Lan, Min Shi, Nadine Chang, Jan Kautz, Ying Li, and Jose M Alvarez. Omnidrive: A holistic llm-agent framework for autonomous driving with 3d perception, reasoning and planning. arXiv preprint arXiv:2405.01533, 2024.

[49] Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, and Yitao Liang. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. arXiv preprint arXiv:2302.01560, 2023.   
[50] Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, and Yitao Liang. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. arXiv preprint arXiv:2302.01560, 2023.   
[51] Zihao Wang, Shaofei Cai, Anji Liu, Yonggang Jin, Jinbing Hou, Bowei Zhang, Haowei Lin, Zhaofeng He, Zilong Zheng, Yaodong Yang, et al. Jarvis-1: Open-world multi-task agents with memory-augmented multimodal language models. arXiv preprint arXiv:2311.05997, 2023.   
[52] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. arXiv preprint arXiv:2404.03384, 2024.   
[53] Xu Yang, Yongliang Wu, Mingzhuo Yang, Haokun Chen, and Xin Geng. Exploring diverse in-context configurations for image captioning. Advances in Neural Information Processing Systems, 36, 2024.   
[54] Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. Appagent: Multimodal agents as smartphone users. arXiv preprint arXiv:2312.13771, 2023.   
[55] Qilang Ye, Zitong Yu, Rui Shao, Xinyu Xie, Philip Torr, and Xiaochun Cao. Cat: Enhancing multimodal large language model to answer questions in dynamic audio-visual scenarios. In ECCV, 2024.   
[56] Haoqi Yuan, Chi Zhang, Hongcheng Wang, Feiyang Xie, Penglin Cai, Hao Dong, and Zongqing Lu. Skill reinforcement learning and planning for open-world long-horizon tasks. arXiv preprint arXiv:2303.16563, 2023.   
[57] Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model based agents. arXiv preprint arXiv:2404.13501, 2024.   
[58] Enshen Zhou, Yiran Qin, Zhenfei Yin, Yuzhou Huang, Ruimao Zhang, Lu Sheng, Yu Qiao, and Jing Shao. Minedreamer: Learning to follow instructions via chain-of-imagination for simulated-world control. arXiv preprint arXiv:2403.12037, 2024.   
[59] Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, et al. Ghost in the minecraft: Generally capable agents for open-world enviroments via large language models with text-based knowledge and memory. arXiv preprint arXiv:2305.17144, 2023.

# A Broader Impact

With the increasing capability level of Multimodal Large Language Models (MLLM) comes many potential benefits and also risks. On the positive side, we anticipate that the techniques that used to create Optimus-1 could be applied to the creation of helpful agents in robotics, video games, and the web. This plug-and-play architecture that we have created can be quickly adapted to different MLLMs, and the proposed methods also provide a viable solution for other application areas in the agent domain. However, on the negative side, it is imperative to acknowledge the inherent stochastic nature of MLLMs in text generation. If not addressed carefully, this could lead to devastating consequences for society. Prior to deploying MLLMs in conjunction with the Hybrid Multimodal Memory methodology, a comprehensive assessment of their potential risks must be undertaken. We hope that while the stakes are low, works such as ours can improve access to safety research on instruction-following models in multimodal agents domains.

# B Minecraft

Minecraft is an extremely popular sandbox video game developed by Mojang Studios 3. It allows players to explore a blockly, procedurally generated 3D world with infinite terrain, discover and extract raw materials, craft tools and items, and build structures or earthworks (shown in Figure 6). Minecraft is a valuable and representative environment for evaluating long-horizon tasks, offering greater diversity and complexity compared to other environments. Unlike web/app navigation [54] and embodied manipulation [16], Minecraft is an open world with a complex and dynamic environment (79 biomes, including ocean, plains, forest, desert, etc.). To complete long-horizon tasks, agents must achieve multiple sub-goals (e.g., 15 sub-goals to craft a diamond sword), making the construction of a Minecraft agent quite challenging. Many studies [46, 32, 51] have chosen Minecraft as the environment for validating performance on long-horizon tasks. Extensive experimental results in the paper show that Optimus-1 outperforms all baselines. Therefore, we chose Minecraft as open-world environment to evaluate the ability of agents to perform long-horizon tasks.

# B.1 Basic Rules

Biomes. The Minecraft world is divided into different areas called “biomes”. Different biomes contain different blocks and plants and change how the land is shaped. There are 79 biomes in Minecraft 1.16.5, including ocean, plains, forest, desert, etc. Diverse environments have high requirements for the generalization of agents.

Time. Time passes within this world, and a game day lasts for 20 real-world minutes. Nighttime is much more dangerous than daytime: the game starts at dawn, and agents have 10 minutes of game time before nightfall. Hostile or neutral mobs spawn when night falls, and most of these mobs are dangerous, trying to attack agents. How to survive in such a dangerous world is an open problem for Minecraft agents research.

Item. In Minecraft 1.16.5, there are 975 items can be obtained, such as wooden pickaxe , iron sword $\mathcal { A }$ . Item can be obtained by crafting or destroying blocks or attacking entities. For example, agent can attack cows $☉$ to obtain leather and beef . Agent also can use 1 stick and 2 diamonds $\circledcirc$ to craft diamond sword $\times$ .

Gameplay progress. Progression primarily involves discovering and utilizing various materials and resources, each of which unlocks new capabilities and options. For instance, crafting a wooden pickaxe enables the player to mine stone , which can be used to create a stone pickaxe $\mathcal { \bar { A } }$ and a furnace ; these, in turn, allow for the mining and smelting of iron ore . Subsequently, an iron pickaxe $\mathcal { \bar { A } }$ permits the extraction of diamonds $\circledcirc$ , and a diamond pickaxe can mine virtually any block in the game. Similarly, cultivating different crops allows for the breeding of various animals, each providing distinct resources beyond mere sustenance. Enemy drops also have specific applications, with some being more beneficial than others. By integrating resources from mining, farming, and breeding, players can enchant their equipment. The collection and crafting of materials also facilitate construction, enabling players to build diverse structures. Beyond practical

![](images/3c0509249d8eba133adffa1156d130420bf587a73406d7bb7a70951e8a519f6f.jpg)  
(a)

![](images/d8a431679eebca8bf368f030937c4b033f3455b71d69e09b4a5b5b6a1a69f189.jpg)  
(b)

![](images/623f7da7faf53d468028c2c227955ede5d175b411de487aec290f89206ba9061.jpg)  
(c)

![](images/0538c6b56b1991968824ff408af6a0e80d2b7aef58601508fa97bb87b434e1ef.jpg)  
(d)   
Figure 6: The screenshots in Minecraft. (a). The world has different complex terrains, including plains, river, forest and mine. (b). The agent can use crafting table to craft tools and items with recipes. (c). The agent can use the furnace to smelt ore to obtain precious ingot. (d). The agent can grow wheat near the river.

considerations such as secure bases and farms, the creative aspect of building personalized structures constitutes a significant part of the Minecraft experience.

Freedom. In Minecraft, player can do anything they can imagine. Player can craft tools, smelt ore, brew potions, trade with villagers and wandering traders, attack mobs, grow crops, raise animals in captivity, etc. Player even can use redstone $\Bumpeq$ to build a computer. This is a world of freedom and infinite possibilities.

More Challenge than Diamond . Progression beyond the Overworld is fairly limited: Eventually, you can build a nether portal to reach the Nether, where you can get materials for more complex crafting, the resources to brew potions, and the top tier of tools and armor. The Nether materials also let you reach the End dimension, where you must defeat the Ender Dragon to unlock the outer End Islands, where you can get an elytra that lets you fly, and shulker boxes for more storage.

# B.2 Observation and Action Spaces

Observation. Our observation space is completely consistent with human players. The agent only receives an RGB image with dimensions of $6 4 0 \times 3 6 0$ during the gameplay process, including the hotbar, health indicators, food saturation, and animations of the player’s hands. It is worth helping the agent see more clearly in extremely dark environments, we have added a night vision effect for the agent, which increases the brightness of the environment during the night.

Action Spaces. Our action space is almost similar to human players, except for craft and smelt actions. It consists of two parts: the mouse and the keyboard. The keypresses are responsible for controlling the movement of agents, such as jumping, forward, back, etc. The mouse movements are responsible for controlling the perspective of agents and the cursor movements when the GUI is opened. The left and right buttons of the mouse are responsible for attacking and using or placing items. In Minecraft, precise mouse movements are important when completing complex tasks that need open inventory or crafting table. In order to achieve both the same action space with MineDojo

Table 4: Our action space.   

<table><tr><td>Index</td><td>Action</td><td>Human Action</td><td>Description</td></tr><tr><td>1</td><td>Forward</td><td>key W</td><td>Move forward.</td></tr><tr><td>2</td><td>Back</td><td>key S</td><td>Move back.</td></tr><tr><td>3</td><td>Left</td><td>key A</td><td>Strafe left.</td></tr><tr><td>4</td><td>Right</td><td>key D</td><td>Strafe right.</td></tr><tr><td>5</td><td>Jump</td><td>key Space</td><td>Jump. When swimming, keeps the player afloat.</td></tr><tr><td>6</td><td>Sneak</td><td>key left Shift</td><td>Slowly move in the current direction of movement.</td></tr><tr><td>7</td><td>Sprint</td><td>key left Ctrl</td><td>Move quickly in the direction of current movement.</td></tr><tr><td>8</td><td>Attack</td><td>left Button</td><td>Destroy blocks (hold down); Attack entity (click once).</td></tr><tr><td>9</td><td>Use</td><td>right Button</td><td>Place blocks, entity, open items or other interact actions defined by game.</td></tr><tr><td>10</td><td>hotbar [1-9]</td><td>keys 1-9</td><td>Selects the appropriate hotbar item.</td></tr><tr><td>11</td><td>Open/Close Inventory</td><td>key E</td><td>Opens the Inventory. Close any open GUI.</td></tr><tr><td>12</td><td>Yaw</td><td>move Mouse X</td><td>Turning; aiming; camera movement.Ranging from -180 to +180.</td></tr><tr><td>13</td><td>Pitch</td><td>move Mouse Y</td><td>Turning; aiming; camera movement.Ranging from -180 to +180.</td></tr><tr><td>14</td><td>Craft</td><td>-</td><td>Execute a crafting recipe to obtain new item</td></tr><tr><td>15</td><td>Smelt</td><td>-</td><td>Execute a smelting recipe to obtain new item.</td></tr></table>

[7], we abstract the craft and the smelt action into action space. The detailed action space is described in Table 4.

# B.3 Long-horizon Tasks

Long-horizon Tasks are complex tasks that require world knowledge to solve and consist of multiple indispensable subtask sequences. In Minecraft, technology has six levels, including wood , stone $\oplus$ , iron , golden $\Leftrightarrow$ , diamond $\textcircled { - }$ , and netherite . Wooden tools can mine stone-level blocks, but can’t mine iron-level and upper-level blocks. Stone tools can mine iron-level blocks, but can’t mine diamond-level and upper-level blocks. Iron-level tools can mine diamond-level blocks, but can’t mine netherite-level blocks. Diamond-level tools can mine any level blocks.

For example, the agent now wants to complete the task “Craft iron sword $\mathbf { \nabla } _ { \mathbf { x } } , \mathbf { \nabla } _ { \mathbf { x } } ,$ . The agent needs to craft wood-level tools to mine stone , and craft stone-level tools to mine iron ore . In order to craft tools, the agent needs a crafting table $\textcircled { 1 }$ . To smelt iron ore $\$ 1$ into iron ingot $\Leftrightarrow .$ , the agent needs a furnace . Moreover, craft crafting table needs 4 planks, and craft furnace needs 8 cobblestone. In summary, the agent needs to obtain many raw materials, wood-level and stone-level tools, 1 crafting table, 1 furnace, and most importantly, 2 iron ingots. The process of this task is shown in Figure 7.

# C Theory

In this section, we briefly introduce the relevant theory of cognitive science. For more details, please refer to the original articles.

Our ability to understand and predict the world around us depends on our long-term memory stores, which have historically been divided into two distinct systems [27, 41, 45]. The semantic memory system provides a conceptual framework for describing the similar meanings of words and objects as they are encountered in different contexts (e.g., a bee is a flying insect with yellow and black stripes that produces honey), whereas the episodic memory system records our personal experiences characterized by the co-occurrence of words and objects at different times and places (e.g., being stung by a bee while eating honey at a picnic last weekend). These information stores and the interactions between them play a crucial role in guiding our behaviour and giving us the flexibility to adapt to the various demands of our environment.

In this paper, inspired by the above theory, we divide the agent memory module into two parts: knowledge and experience. Based on this, we propose Hierarchical Directed Knowledge Graph and Abstracted Multimodal Experience Pool to enable the agent to acquire, store, and utilize knowledge and experience during the execution of tasks. Extensive experimental results demonstrate the effectiveness of the proposed methodology

![](images/1cf1d408b156d0dd30f3bdd151fe9cd8725b80aaa5804b6cc422fc9c1722d632.jpg)  
(a) Chop 7 logs

![](images/51491e5eb0baf827aee2eeb3e1e284833225ffe72642a11e1428ca9d55f1ab8c.jpg)  
(b) Craft 21 planks

![](images/ea2f3d4c0e49897e38bfa8f175c32ffd321fb8b88a4b952f7de5f4529c00f100.jpg)  
(c) Craft 5 sticks

![](images/d7a5454c39931f49eb1437deb7d78251142905852d70f7256bba674f089bd849.jpg)  
(d) Craft 1 crafting table

![](images/b48067c61f777a86b683a24a3a4f382e2963f03abdd8588951824a8f2e1fceeb.jpg)  
(e) Craft 1 wooden pickaxe

![](images/f48ca0b627d623b9750ae59693d34457f31f9b424f4a4b7ed96b0de3cfaa04fa.jpg)  
(f) Mine 11 cobblestone

![](images/ce20e6ef2fa961b51cffe49d944c451f69e395eb2ecbae41f23ca6026c1f0efa.jpg)  
(g) Craft 1 furnace

![](images/52260cdf1c046b1fbe05ddac4cdf7b2f895d4f4e4794205d98f01ae896745b99.jpg)  
(h) Craft 1 stone pickaxe

![](images/6e95f51b4fb5d399b3635674cd23e8df2582fab78e54740b33894a2c7a295e9d.jpg)  
(i) Dig down more deeper to find iron ore

![](images/a78a4ca282e4b26253bb9bb0610ea200bb69f3987e83209db977b3537291b0ee.jpg)  
(j) Mine 2 iron ores

![](images/476c92d58918418b3da50a5a5ab3efe8e3068204cb183691c794796e2be51904.jpg)  
(k) Smelt 2 iron ingots

![](images/275ca002d29df7401289c236313b3240c64938f9c1d5eefe31b90a1e8166d017.jpg)  
(l) Craft 1 iron sword   
Figure 7: Processing of task "Craft 1 iron sword". Optimus-1 needs thousands of steps to complete this task. To craft and smelt precisely, the mouse movements action can’t have any error.

# D Benchmark Suite

# D.1 Benchmark

We constructed a benchmark of 67 tasks to evaluate Optimus-1’s ability to complete long-horizon tasks in Minecraft. According to recommended categories in Minecraft, we have classified these tasks into 7 groups: Wood , Stone , Iron $\Leftrightarrow$ , Gold $\odot ,$ , Diamond $\circledcirc$ , Redstone , Armor . The statistics for benchmark are shown in Table 5. Due to the varying complexity of these tasks, we adopt different maximum gameplay steps (Max. Steps) for each task. The maximum steps are determined by the average steps that human players need to complete the task. Due to the randomness of Minecraft, the world and initial spawn point of the agent could vary a lot. In our benchmark setting, We initialize the agent with an empty inventory, which makes it necessary for the agent to complete a series of sub-goals (mining materials, crafting tools) in order to perform any tasks. This makes every task challenging, even for human players.

Note that Diamonds are a very rare item that only spawns in levels 2 to 16 and have a $0 . 0 8 4 6 \%$ chance of spawning in Minecraft 1.16.5. Diamonds are usually found near level 9, or in man-made or natural mines no higher than level 16. In order to reduce the huge impact that diamond generation

Table 5: Setting of 7 groups encompassing 67 Minecraft long-horizon tasks.   

<table><tr><td>Group</td><td>Task Num.</td><td>Example Task</td><td>Max. Steps</td><td>Initial Inventory</td><td>Avg. Sub-goal Num.</td></tr><tr><td>Wooden</td><td>10</td><td>Craft a wooden axe</td><td>3600</td><td>Empty</td><td>4.60</td></tr><tr><td>Stone</td><td>9</td><td>Craft one stone pickaxe</td><td>7200</td><td>Empty</td><td>8.78</td></tr><tr><td>Iron</td><td>16</td><td>Craft a iron pickaxe</td><td>12000</td><td>Empty</td><td>12.75</td></tr><tr><td>Golden</td><td>6</td><td>Mine gold and smelt into golden ingot</td><td>36000</td><td>Empty</td><td>15.83</td></tr><tr><td>Redstone</td><td>6</td><td>Craft a piston</td><td>36000</td><td>Empty</td><td>17.50</td></tr><tr><td>Diamond</td><td>7</td><td>Dig down and mine a diamond</td><td>36000</td><td>Empty</td><td>15.14</td></tr><tr><td>Armor</td><td>13</td><td>Craft one iron helmet</td><td>36000</td><td>Empty</td><td>15.85</td></tr></table>

Table 6: We evaluate Optimus-1 on these tasks in ablation study which are the subset of our benchmark.   

<table><tr><td>Group</td><td>Task</td><td>Sub-Goal Num.</td><td>Max. Step</td><td>Initial Inventory</td></tr><tr><td rowspan="2">Wooden</td><td>Craft a wooden axe</td><td>5</td><td>3600</td><td>Empty</td></tr><tr><td>Craft a crafting table</td><td>3</td><td>3600</td><td>Empty</td></tr><tr><td rowspan="3">Stone</td><td>Craft a stone pickaxe</td><td>10</td><td>7200</td><td>Empty</td></tr><tr><td>Craft a stone axe</td><td>10</td><td>7200</td><td>Empty</td></tr><tr><td>Craft a furnace</td><td>9</td><td>7200</td><td>Empty</td></tr><tr><td rowspan="5">Iron</td><td>Craft a iron pickaxe</td><td>13</td><td>12000</td><td>Empty</td></tr><tr><td>Craft a bucket</td><td>13</td><td>12000</td><td>Empty</td></tr><tr><td>Craft a rail</td><td>13</td><td>12000</td><td>Empty</td></tr><tr><td>Craft a iron sword</td><td>12</td><td>12000</td><td>Empty</td></tr><tr><td>Craft a shears</td><td>12</td><td>12000</td><td>Empty</td></tr><tr><td rowspan="3">Golden</td><td>Craft a golden pickaxe</td><td>16</td><td>36000</td><td>Empty</td></tr><tr><td>Craft a golden axe</td><td>16</td><td>36000</td><td>Empty</td></tr><tr><td>Smelt a golden ingot</td><td>15</td><td>36000</td><td>Empty</td></tr><tr><td rowspan="5">Diamond</td><td>Craft a diamond pickaxe</td><td>15</td><td>36000</td><td>Empty</td></tr><tr><td>Craft a diamond axe</td><td>16</td><td>36000</td><td>Empty</td></tr><tr><td>Craft a diamond hoe</td><td>15</td><td>36000</td><td>Empty</td></tr><tr><td>Craft a diamond sword</td><td>15</td><td>36000</td><td>Empty</td></tr><tr><td>Dig down and mine a diamond</td><td>15</td><td>36000</td><td>Empty</td></tr></table>

probability has on agent’s likelihood of completing a task, we have adjusted the diamond generation probability to $2 0 \%$ , spawns in levels 2 to 16. This setting applies to human players as well.

In the ablation study, we select the subset of our benchmark as the test set (shown in Table 6). The environment setting is the same as the benchmark.

# D.2 Baselines

Existing Baseline. On the one hand, we employ GPT-3.5 and GPT-4V as baseline, which are evaluated without integrating hybrid multimodal memory modules. During the planning phase, they generate a plan for the action controller based on task prompt (and observation). During the reflection phase, they generate reflection results in a zero-shot manner. On the other hand, we compare existing SOTA Agents [50, 51] in Minecraft.

Human-level Baseline. To better demonstrate agent’s performance level in Minecraft, we hired 10 volunteers to play the game as a human-level baseline. The volunteers played the game with the same environment and settings, and every volunteer asked to perform the each task on the benchmark 10 times. Ultimately, we used the average scores of 10 volunteers as the human-level baseline. The results of the human-level baseline are shown in Table 1. To ensure the validity of the experiment, we ensured that each volunteer had at least 20 hours of Minecraft gameplay before conducting the experiment. For each volunteer, we pay $\$ 25$ as reward.

Table 7: Statistics for various Minecraft agents.   

<table><tr><td>Agent</td><td>Pub.</td><td>Env.</td><td>Input</td><td>Output</td><td>Planning</td><td>Reflection</td><td>Knowledge</td><td>Experience</td></tr><tr><td>VPT [1]</td><td>NeurIPS&#x27; 22</td><td>MineRL</td><td>V</td><td>low-level action</td><td></td><td></td><td></td><td></td></tr><tr><td>MineDOJO [7]</td><td>NeurIPS&#x27; 22</td><td>MineDOJO</td><td>T+V</td><td>low-level action</td><td></td><td></td><td></td><td></td></tr><tr><td>STEVE-1 [25]</td><td>NeurIPS&#x27; 23</td><td>MineRL</td><td>T+V</td><td>low-level action</td><td></td><td></td><td></td><td></td></tr><tr><td>Voyager [46]</td><td>NeurIPS&#x27; 23</td><td>Mineflayer</td><td>T+V</td><td>code</td><td>✓</td><td>✓</td><td></td><td>✓</td></tr><tr><td>DEPS [50]</td><td>NeurIPS&#x27; 23</td><td>MineDOJO</td><td>T+V</td><td>code</td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>GROOT [3]</td><td>ICLR&#x27; 24</td><td>MineRL</td><td>T+V</td><td>low-level action</td><td></td><td></td><td></td><td></td></tr><tr><td>MP5 [32]</td><td>CVPR&#x27; 24</td><td>MineDOJO</td><td>T+V</td><td>code</td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>Jarvis-1 [51]</td><td>-</td><td>MineRL</td><td>T+V</td><td>low-level action</td><td>✓</td><td>✓</td><td></td><td>✓</td></tr><tr><td>Optimus-1</td><td>-</td><td>MineRL</td><td>T+V</td><td>low-level action</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

# D.3 Minecraft Agents

In this section, we summarise the differences between existing Minecraft agents. As shown in the Table 7, earlier work [1, 7, 25, 3] constructed Transformer-based policy network as agent. Recent work [46, 50, 51, 32] introduces the Multimodal Large Language Model, which empowers the agent to complete long-horizon tasks by exploiting the powerful language comprehension and planning capabilities of LLM.

In the Mineflayer and Minedojo environments, agents [7, 46, 50, 32] can accomplish sub-goals by calling APIs (in the form of codes), which is a different behavioral pattern from humans. In MineRL [11], agents [1, 25, 3, 51] must generate low-level actions to perform tasks, which is more challenging to accomplish long-horizon tasks.

Moreover, existing agents lack knowledge and experience, and their performance in Minecraft is still vastly gapped from the human level. In this paper, we introduce Hybrid Multimodal Memory, which empowers Optimus-1 with hierarchical knowledge and multimodal experience. This makes Optimus-1 significantly outperform all existing agents on challenging long-horizon tasks benchmark, and exhibits near human-level performance on many tasks.

# E Implementation Details

# E.1 Hybrid Multimodal Memory

# E.1.1 Abstracted Multimodal Experience Pool

Relevant studies [5, 28, 17, 15] have demonstrated the importance of memory for agents to complete long-horizon tasks. To implement the memory mechanism, Minedojo [7] and Voyager [46] only considered unimodal storage of historical information. Jarvis-1 [51] considered a multimodal memory mechanism to store task planning and visual information as experience, but it stores all historical information without summarisation. This approach stores all visual images, which poses a huge challenge in storage size and retrieval efficiency. To solve the problem, we propose the Abstracted Multimodal Experience Pool structure, which summarizes all historical information during the agent’s execution of the task, which maintains the integrity of long sequential information and greatly improves the storage and retrieval efficiency of the experience.

As shown in Figure 2, we first input the visual image stream to the video buffer, which filters the image stream at a fixed frequency. It makes the length of the image stream substantially shorter. Empirically, we set the frequency of filtering to 1 second/frame, meaning that the video buffer takes one frame per second from the original image stream to compose the filtered image stream. We found that above this frequency makes the visual information redundant (too much similarity between images), and below this frequency does not preserve enough complete visual information.

Then, we feed the filtered frames into an image buffer with a window size of 16. We dynamically compute the similarity between images in the image buffer, when a new image comes in, we compute the similarity between the new image and the most recent image, and then we remove the image with the highest similarity in order to keep the image buffer’s window size to 16.

Subsequently, we introduce MineCLIP [7], a pre-trained model of video-text alignment with a structure similar to CLIP [34], as our visual summariser. For a given sub-goal, it calculates the

correlation between the visual content within the current memory bank and the sub-goal, and when this correlation exceeds a pre-set threshold, the frames within the memory bank are saved as the visual memories corresponding to that sub-goal. Finally, we store the visual memories with the sub goal’s textual description into the Abstracted Multimodal Experience Pool. In addition, we incorporate the environment information, agent initial state, plan from Knowledge-Guided Planner, etc. into the experience memory of the given task. In this way, we consider the history information of each sub-goal and summaries and summarise it to finally compose the multimodal experience of the given task.

Note that we also store these visual memories as failure cases when the feedback from the reflection phase is REPLAN. Therefore, when Optimus-1 executes a long-horizon task, it can retrieve past successes and failures as references and update memory after the task is finished. In the reflection phase, Optimus-1 retrieve the most relevant cases from Abstracted Multimodal Experience Pool, which contains the three scenarios COMPLETE, CONTINUE, and REPLAN, to help the agent better assess which state the current situation belongs to. This approach of considering both successful and failed cases for in-context learning is inspired by related research [8, 31], and its effectiveness is validated in Section 3.3.

# E.1.2 Hierarchical Directed Knowledge Graph

As shown in the Figure 2, crafting a diamond sword $\times$ requires two diamonds $\circledcirc$ and a wooden stick $\diagup$ , while mining diamonds requires an iron pickaxe $\widehat { \lambda }$ , which in turn requires additional raw materials and crafting steps. We transform this mine and craft knowledge into a graph structure, where the nodes of the graph are objects, and the nodes point to objects that can be crafted or completed by that object. With directed graph, we show that connections between objects are established, and that this knowledge can be stored and updated efficiently. For a given object, we only need to retrieve the corresponding node to extract the corresponding subgraph from the knowledge graph. Then by topological sorting, we can get the antecedents and required materials for the object, and this information is provided to the Knowledge-Guided Planner as a way to generate a more reasonable sequence of sub-goals. With Hierarchical Directed Knowledge Graph, we can significantly enhance the world knowledge of the agent in a train-free manner, as shown in the experimental results in Section 3.3.

Our HDKG can be efficiently updated and expanded. When adding new nodes, the HDKG can be updated by simply merging the nodes and relationships into the graph. This method involves local linear modifications to the graph rather than altering the entire graph, making the process efficient and time-saving. For example, when M new nodes and N edges are added, the HDKG can be updated with $\mathbf { M } { + } \mathbf { N }$ times of operations. Moreover, an HDKG containing 851 objects (nodes) requires less than 1 MB of memory. Thus, the HDKG can be efficiently updated and maintained.

# E.2 Hybrid Multimodal Memory Driven Optimus-1

In order to implement the proposed Hybrid Multimodal Memory and to progressively increase the capacity of Optimus-1 in a self-evolution manner, we propose a non-parametric learning method named “free exploration-teacher guidance”.

In the free exploration phase, we randomly initialize the environment, materials, and tasks. For the task “craft a wooden pickaxe”, we provide initial materials (three planks, two sticks), and then Optimus-1 (only the action controller activated) attempts to complete the task. If the environment feedback indicates the task is successful, the knowledge {3 planks, 2 sticks wooden pickaxe} is added to the HDKG. Note that we randomly initialize materials and their quantities, which means that the task may not always succeed. As a result, each free exploration may not acquire the corresponding knowledge, but it can record the relevant experience (whether successful or fail). In the free exploration phase, Optimus-1 learns simple atomic operations, such as crafting sticks in the Wooden Group and mining diamonds in the Diamond Group.

In the teacher guidance phase, Optimus-1 need to learn a small number of long-horizon tasks based on extra knowledge. For example, during the free exploration phase, Optimus-1 mastered crafting stick $\diagup$ and mining diamond $\circledcirc$ , but did not know that “a diamond sword $\pmb { x }$ is obtained by a stick $\mathcal { f }$ and two diamonds $\oplus ^ { , , }$ . So we provide some task plans, which will serve as extra knowledge to guide

Optimus-1 to complete the task of “craft diamond sword”. We built the following automated process to get the task plan needed for “free exploration”:

• We randomly select 5 tasks for each Group (7 groups in total) that are not included in the benchmark.   
• For each selected task, we use a script to automatically obtain the crafting relationships from the Minecraft Wiki 4. Taking the task “craft a wooden sword” as an example, we use the script to automatically obtain the crafting relationships: 1 wooden stick, 2 planks, 1 crafting table $ 1$ wooden sword, $1 \log  4$ planks, 2 planks $ 4$ sticks, 4 planks $ 1$ crafting table.   
• These relationships are converted into a directed acyclic graph through an automated script. By performing a topological sort, the graph can be converted into tuples of materials and their quantities: (wooden sword, 1), (crafting table, 1), (wooden stick, 1) (planks, 8), (log, 2).   
• We prompt GPT-4 to construct a plan in order from basic materials to advanced materials.   
• Finally, we get the plan: 1. Get two logs 2. Craft eight planks 3. Craft a crafting table 4. Craft a wooden stick 5. Craft a wooden sword

During the teacher guidance phase, Optimus-1’s memory is further expanded and it gains the experience of executing complete long-horizon tasks. Teacher guidance phase allows Optimus-1 to acquire advanced knowledge and learn multimodal experiences through complete long-horizon tasks.

# E.3 Backbone of Optimus-1

Optimus-1 consists of Knowledge-Guided Planner, Experience-Driven Reflector, and Action Controller. In this paper, we employ OpenAI’s GPT-4V (gpt-4-turbo) 5 as Knowledge-Guided Planner and Experience-Driven Reflector, and STEVE-1 [25] as Action Controller. We also employ opensource models like Deepseek-VL [26] and InternLM-XComposer2-VL [6] as Knowledge-Guided Planner and Experience-Driven Reflector.

All experiments were implemented on $4 \mathbf { x }$ NVIDIA A100 GPUs. We employ multiple Optimus-1 to perform different tasks at the same time, and this parallelized inference greatly improves our experimental efficiency. In the free exploration and teacher guidance phases, there is no need to access OpenAI’s API, and the learning process takes approximately 16 hours on 4x A100 80G GPUs. During the inference phase, it takes about 20 hours on 4x A100 80G GPUs.

Throughout the experiment, we spent about $5, 000 to access the GPT-4V API. However, we also offer more cost-effective solutions. As shown in Figure 5, if we employ Deepseek-VL [26] or InternLM-XComposer2-VL [6] as Optimus-1’s backbone, we can get comparable performance with low-cost!

# E.4 Prompt for Optimus-1

We show the prompt templates for Experience-Driven Reflector and Action Controller as follows.

System: You are a MineCraft game expert and you can guide agents to complete complex tasks.

User: For a given game screen and task, you need to complete <goal inference> and <visual inference>.

<goal inference>: According to the task, you need to infer the weapons, equipment, or materials required to complete the task.

<visual inference>: According to the game screen, you need to infer the following aspects: health bar, food bar, hotbar, environment.

I will give you an example as follow: [Example]

<task>: craft a stone sword.

<goal inference>: stone sword

<visual inference>

health bar: full

food bar: full

4https://minecraft.wiki/

5https://openai.com/index/gpt-4v-system-card/

Listing 1: Prompt for Knowledge-Guided Planner.   
```txt
hotbar: empty   
environment: forest   
Here is a game screen and task, you MUST output in example format. <task>: {task}.   
<game screen>: {image}   
Assistant:   
User: Now you need to make a plan with the help of <visual info> and <craft graph>.   
<visual info>: Consists of the following aspects: health bar, food bar, hotbar, environment. Based on the current visual information, you need to consider whether prequel steps needed to ensure that agent can complete the task.   
<craft graph>: a top-down list of all the tools and materials needed to complete the task. I will give you an example of planning under specific visual conditions as follow: [Example] {example}   
Here is a game screen and task, you MUST output in example format. Remember <task planning> MUST output in example format.   
<task>: {task}   
<game screen>: {image}   
<craft graph>: {graph}   
Assistant: 
```

Listing 2: Prompt for Experience-Driven Reflector.   
```txt
System: You are a MineCraft game expert and you can guide agents to complete complex tasks. Agent is executing the task: {task}.   
Given two images about agent's state before executing the task and its current state, you should first detection the environment (forest, cave, ocean, etc.) in which the agent is located, then determine whether the agent's current situation is done, continue, or replan.   
<done>: Comparing the image before the task was performed, the current image reveals that the task is complete.   
<continue>: Current image reveals that the task is NOT complete, but agent is in good state ( good health, not hungry) with high likelihood to complete task..   
<replan>: Current image reveals that the task is NOT complete, and agent is in bad state (bad health, or hungry) or situation (in danger, or in trouble), need for replanning. For replan, you need to further determine whether the agent's predicament is "drop_down" or "in_water". "drop_down" means that the agent has fallen into a cave or is trapped in a mountain or river, while "in_water" means that the agent is in the ocean and needs to return to land immediately.   
User: I'll give you some examples to illustrate the different situations. Each example consists of two images, where the first image is the state of the agent before performing the task and the second image is the current state of the agent.   
[Examples]   
<done>: {image1},{image2}   
<continue>: {image1},{image2}   
<replan>: {image1},{image2}   
Now given two images about agent's state before executing the task and its current state, you MUST and ONLY output in following format:   
Enviroment: <environment>   
Situation: <situation>   
(if situation is replan) Predicament: <predicament> 
```

# F Additional Experimental Results

# F.1 Full Results on Our Benchmark

We list the results of each task on the benchmark below, with details including task name, sub-goal numbers, success rate (SR), average number of steps (AS), average time (AT), and eval times. All tasks are evaluated in Minecraft 1.16.5 Survival Mode. Note that each time Optimus-1 performs a task, we initial it with an empty initial inventory and a random start point. This makes it challenging for Optimus-1 to perform each task.

Table 8: The results of Optimus-1 on various tasks in the Wood group. SR, AS, AT denote success rate, average number of steps, and average time (seconds), respectively.   

<table><tr><td>Task</td><td>Sub-Goal Num.</td><td>SR</td><td>AS</td><td>AT(s)</td><td>Eval Times</td></tr><tr><td>Craft a wooden shovel</td><td>6</td><td>95.00</td><td>995.58</td><td>49.78</td><td>40</td></tr><tr><td>Craft a wooden pickaxe</td><td>5</td><td>100.00</td><td>1153.91</td><td>57.70</td><td>30</td></tr><tr><td>Craft a wooden axe</td><td>5</td><td>96.67</td><td>1010.28</td><td>50.51</td><td>30</td></tr><tr><td>Craft a wooden hoe</td><td>5</td><td>100.00</td><td>1042.80</td><td>52.14</td><td>30</td></tr><tr><td>Craft a stick</td><td>4</td><td>97.14</td><td>372.97</td><td>18.65</td><td>70</td></tr><tr><td>Craft a crafting table</td><td>3</td><td>98.55</td><td>448.63</td><td>22.43</td><td>69</td></tr><tr><td>Craft a wooden sword</td><td>5</td><td>100.00</td><td>1214.90</td><td>60.74</td><td>30</td></tr><tr><td>Craft a chest</td><td>4</td><td>100.00</td><td>573.80</td><td>28.69</td><td>30</td></tr><tr><td>Craft a bowl</td><td>4</td><td>100.00</td><td>744.30</td><td>37.21</td><td>30</td></tr></table>

Table 9: The results of Optimus-1 on various tasks in the Stone group. SR, AS, AT denote success rate, average number of steps, and average time (seconds), respectively.   

<table><tr><td>Task</td><td>Sub-Goal Num.</td><td>SR</td><td>AS</td><td>AT(s)</td><td>Eval Times</td></tr><tr><td>Craft a stone shovel</td><td>8</td><td>90.32</td><td>2221.00</td><td>111.05</td><td>31</td></tr><tr><td>Craft a stone pickaxe</td><td>10</td><td>96.77</td><td>2310.09</td><td>115.50</td><td>31</td></tr><tr><td>Craft a stone axe</td><td>10</td><td>96.88</td><td>2112.59</td><td>105.63</td><td>32</td></tr><tr><td>Craft a stone hoe</td><td>8</td><td>94.64</td><td>2684.60</td><td>134.23</td><td>56</td></tr><tr><td>Craft a charcoal</td><td>9</td><td>88.57</td><td>3083.35</td><td>154.17</td><td>35</td></tr><tr><td>Craft a smoker</td><td>9</td><td>90.24</td><td>3118.89</td><td>155.94</td><td>41</td></tr><tr><td>Craft a stone sword</td><td>8</td><td>94.29</td><td>2067.92</td><td>103.40</td><td>35</td></tr><tr><td>Craft a furnace</td><td>9</td><td>93.75</td><td>2842.71</td><td>142.14</td><td>32</td></tr><tr><td>Craft a torch</td><td>8</td><td>85.71</td><td>2109.00</td><td>105.45</td><td>95</td></tr></table>

Moreover, in MineRL [11] environment, ’steps’ refers to the number of interactions between the agent and the environment, occurring at a frequency of 20 times per second. For example, if an agent takes 2 seconds to complete the task “chop a tree”, it interacts with the environment 40 times, resulting in a recorded steps number of 40. Experimental results show that Optimus-1’s average task completion step (AS) is significantly lower than other baselines.

# F.2 Results on Other Benchmark

For a more comprehensive comparison with current Minecraft Agents, we also report Optimus-1’s performances on the benchmark used by Voyager [46], MP5 [32], and DEPS [50] below. Due to the different environments and settings, agents perform tasks with varying degrees of difficulty. For example, Optimus-1 requires low-level action to perform any task in MineRL [11], and we initialize its inventory to be empty. While Voyager [46] performs tasks in Mineflayer 6 environment only through encapsulated code, MP5 [32] performs tasks in MineDOJO [7] environment only needs a specific control signal to craft tools, no low-level actions (mouse movement and click) are needed.

Optimus-1’s success rate in completing tasks with these baselines is shown in the Table 15 and Table 16, and Optimus-1’s efficiency in unlocking the tech tree in Minecraft is shown in the Figure 8. These results reveal that Optimus-1 outperforms a variety of powerful baseline agents, even in challenging environmental settings!

Table 10: The results of Optimus-1 on various tasks in the Iron group. SR, AS, AT denote success rate, average number of steps, and average time (seconds), respectively.   

<table><tr><td>Task</td><td>Sub Goal Num.</td><td>SR</td><td>AS</td><td>AT(s)</td><td>Eval Times</td></tr><tr><td>Craft an iron shovel</td><td>13</td><td>54.79</td><td>5677.35</td><td>637.81</td><td>73</td></tr><tr><td>Craft an iron pickaxe</td><td>13</td><td>59.42</td><td>6157.39</td><td>591.81</td><td>69</td></tr><tr><td>Craft an iron axe</td><td>13</td><td>54.29</td><td>6026.26</td><td>676.97</td><td>70</td></tr><tr><td>Craft a stone_hoe</td><td>13</td><td>52.70</td><td>6650.97</td><td>743.82</td><td>74</td></tr><tr><td>Craft a bucket</td><td>13</td><td>54.29</td><td>6124.61</td><td>591.35</td><td>70</td></tr><tr><td>Craft a hopper</td><td>14</td><td>46.67</td><td>7242.14</td><td>710.17</td><td>60</td></tr><tr><td>Craft a rail</td><td>13</td><td>42.19</td><td>6713.07</td><td>754.48</td><td>64</td></tr><tr><td>Craft an iron sword</td><td>12</td><td>57.14</td><td>5625.49</td><td>633.91</td><td>70</td></tr><tr><td>Craft a shears</td><td>12</td><td>53.62</td><td>5058.00</td><td>570.35</td><td>69</td></tr><tr><td>Craft a smithing table</td><td>12</td><td>44.93</td><td>5317.39</td><td>594.81</td><td>69</td></tr><tr><td>Craft a tripwire hook</td><td>13</td><td>48.57</td><td>4968.74</td><td>562.66</td><td>70</td></tr><tr><td>Craft a chain</td><td>13</td><td>44.93</td><td>5764.42</td><td>645.33</td><td>69</td></tr><tr><td>Craft an iron bars</td><td>12</td><td>42.00</td><td>6508.43</td><td>723.13</td><td>50</td></tr><tr><td>Craft an iron nugget</td><td>12</td><td>30.99</td><td>4697.23</td><td>525.29</td><td>71</td></tr><tr><td>Craft a blast furnace</td><td>14</td><td>25.71</td><td>7760.67</td><td>711.05</td><td>35</td></tr><tr><td>Craft a stonecutter</td><td>13</td><td>34.78</td><td>5993.38</td><td>675.52</td><td>46</td></tr></table>

Table 11: The results of Optimus-1 on various tasks in the Gold group. SR, AS, AT denote success rate, average number of steps, and average time (seconds), respectively.   

<table><tr><td>Task</td><td>Sub Goal Num.</td><td>SR</td><td>AS</td><td>AT(s)</td><td>Eval Times</td></tr><tr><td>Craft a golden shovel</td><td>16</td><td>9.80</td><td>13734.75</td><td>686.74</td><td>51</td></tr><tr><td>Craft a golden pickaxe</td><td>16</td><td>13.75</td><td>9672.00</td><td>783.60</td><td>80</td></tr><tr><td>Craft a golden axe</td><td>16</td><td>4.44</td><td>10158.75</td><td>707.94</td><td>45</td></tr><tr><td>Craft a golden hoe</td><td>16</td><td>3.33</td><td>13120.50</td><td>756.03</td><td>27</td></tr><tr><td>Craft a golden sword</td><td>16</td><td>3.33</td><td>9792.00</td><td>789.60</td><td>26</td></tr><tr><td>Smelt and craft a gold_ ingot</td><td>15</td><td>16.42</td><td>9630.27</td><td>681.51</td><td>67</td></tr></table>

Table 12: The results of Optimus-1 on various tasks in the Diamond group. SR, AS, AT denote success rate, average number of steps, and average time (seconds), respectively.   

<table><tr><td>Task</td><td>Sub Goal Num.</td><td>SR</td><td>AS</td><td>AT(s)</td><td>Eval Times</td></tr><tr><td>Craft a diamond shovel</td><td>15</td><td>18.75</td><td>23696.75</td><td>1184.84</td><td>64</td></tr><tr><td>Craft a diamond pickaxe</td><td>15</td><td>15.71</td><td>32189.50</td><td>1609.46</td><td>70</td></tr><tr><td>Craft a diamond axe</td><td>16</td><td>4.00</td><td>21920.50</td><td>1096.03</td><td>75</td></tr><tr><td>Craft a diamond hoe</td><td>15</td><td>4.61</td><td>24031.00</td><td>1201.55</td><td>65</td></tr><tr><td>Craft a diamond sword</td><td>15</td><td>14.52</td><td>27555.50</td><td>1377.78</td><td>62</td></tr><tr><td>Dig down and mine a diamond</td><td>15</td><td>9.09</td><td>20782.13</td><td>1039.11</td><td>64</td></tr><tr><td>Craft a jukebox</td><td>15</td><td>14.58</td><td>25056.00</td><td>1252.80</td><td>48</td></tr></table>

Table 13: The results of Optimus-1 on various tasks in the Redstone group. SR, AS, AT denote success rate, average number of steps, and average time (seconds), respectively.   

<table><tr><td>Language Instruction</td><td>Sub-Goal Num.</td><td>SR</td><td>AS</td><td>AT(s)</td><td>Eval Times</td></tr><tr><td>Craft a piston</td><td>16</td><td>28.57</td><td>6457.10</td><td>822.85</td><td>35</td></tr><tr><td>Craft a redstone torch</td><td>16</td><td>29.63</td><td>6787.87</td><td>939.39</td><td>27</td></tr><tr><td>Craft an activator rail</td><td>18</td><td>15.68</td><td>8685.62</td><td>934.28</td><td>51</td></tr><tr><td>Craft a compass</td><td>23</td><td>15.00</td><td>14908.67</td><td>845.43</td><td>40</td></tr><tr><td>Craft a dropper</td><td>16</td><td>37.50</td><td>7272.80</td><td>1063.64</td><td>40</td></tr><tr><td>Craft a note block</td><td>16</td><td>24.32</td><td>6727.89</td><td>936.39</td><td>37</td></tr></table>

Table 14: The results of Optimus-1 on various tasks in the Armor group. SR, AS, AT denote success rate, average number of steps, and average time (seconds), respectively.   

<table><tr><td>Task</td><td>Sub Goal Num.</td><td>SR</td><td>AS</td><td>AT(s)</td><td>Eval Times</td></tr><tr><td>Craft shield</td><td>14</td><td>43.33</td><td>7229.00</td><td>861.45</td><td>30</td></tr><tr><td>Craft iron chestplate</td><td>14</td><td>47.22</td><td>7230.24</td><td>851.51</td><td>36</td></tr><tr><td>Craft iron boots</td><td>14</td><td>23.81</td><td>6597.33</td><td>729.87</td><td>42</td></tr><tr><td>Craft iron leggings</td><td>14</td><td>6.67</td><td>9279.00</td><td>763.95</td><td>30</td></tr><tr><td>Craft iron helmet</td><td>14</td><td>58.14</td><td>6287.11</td><td>814.36</td><td>43</td></tr><tr><td>Craft diamond helmet</td><td>17</td><td>2.08</td><td>7342.00</td><td>867.10</td><td>48</td></tr><tr><td>Craft diamond chestplate</td><td>17</td><td>2.70</td><td>7552.00</td><td>777.60</td><td>37</td></tr><tr><td>Craft diamond leggings</td><td>17</td><td>9.68</td><td>7664.67</td><td>883.23</td><td>31</td></tr><tr><td>Craft diamond boots</td><td>17</td><td>16.67</td><td>10065.60</td><td>803.28</td><td>30</td></tr><tr><td>Craft golden helmet</td><td>17</td><td>12.50</td><td>11563.25</td><td>778.16</td><td>32</td></tr><tr><td>Craft golden leggings</td><td>17</td><td>14.60</td><td>10107.33</td><td>805.37</td><td>41</td></tr><tr><td>Craft golden boots</td><td>17</td><td>6.06</td><td>10311.00</td><td>915.55</td><td>33</td></tr><tr><td>Craft golden chestplate</td><td>17</td><td>9.67</td><td>10407.58</td><td>820.38</td><td>31</td></tr></table>

![](images/0c7e5e560cf06e25a23cbe081e13bd31e620caa84dcf06425c13b238e9ffa738.jpg)  
Figure 8: An illustration of Optimus-1 unlocking the tech tree in Minecraft.

Table 15: Result on Process-Dependent Tasks compared with MP5 [32]. SR, AS, AT denote success rate, average number of steps, and average time (seconds), respectively.   

<table><tr><td rowspan="2" colspan="2">Task Level</td><td rowspan="2">MP5 [32]SR</td><td colspan="3">Optimus-1</td></tr><tr><td>SR</td><td>AS</td><td>AT(s)</td></tr><tr><td rowspan="6">Basic Level</td><td>log</td><td>96.67</td><td>100.00</td><td>586.58</td><td>29.33</td></tr><tr><td>sand</td><td>96.67</td><td>94.32</td><td>1540.33</td><td>77.02</td></tr><tr><td>planks</td><td>96.67</td><td>100.00</td><td>571.06</td><td>28.55</td></tr><tr><td>stick</td><td>96.67</td><td>97.14</td><td>372.97</td><td>18.65</td></tr><tr><td>crafting table</td><td>93.33</td><td>98.55</td><td>448.63</td><td>22.43</td></tr><tr><td>Average</td><td>96.00</td><td>98.00</td><td>703.91</td><td>35.20</td></tr><tr><td rowspan="6">Wooden Level</td><td>bowl</td><td>93.33</td><td>100.00</td><td>744.30</td><td>37.21</td></tr><tr><td>boat</td><td>93.33</td><td>92.86</td><td>1170.00</td><td>58.50</td></tr><tr><td>chest</td><td>90.00</td><td>100.00</td><td>573.80</td><td>28.69</td></tr><tr><td>wooden sword</td><td>86.67</td><td>100.00</td><td>1214.90</td><td>60.74</td></tr><tr><td>wooden pickaxe</td><td>80.00</td><td>100.00</td><td>1153.91</td><td>57.70</td></tr><tr><td>Average</td><td>88.67</td><td>98.57</td><td>971.38</td><td>48.56</td></tr><tr><td rowspan="6">Stone Level</td><td>cobblestone</td><td>80.00</td><td>95.29</td><td>1492.00</td><td>74.60</td></tr><tr><td>furnace</td><td>80.00</td><td>93.75</td><td>2842.71</td><td>142.14</td></tr><tr><td>stone pickaxe</td><td>80.00</td><td>96.77</td><td>2310.09</td><td>115.50</td></tr><tr><td>iron ore</td><td>60.00</td><td>50.00</td><td>3017.00</td><td>150.85</td></tr><tr><td>glass</td><td>80.00</td><td>81.11</td><td>3870.75</td><td>193.54</td></tr><tr><td>Average</td><td>76.00</td><td>83.38</td><td>2706.51</td><td>135.32</td></tr><tr><td rowspan="6">Iron Level</td><td>iron ingot</td><td>56.67</td><td>59.42</td><td>4697.23</td><td>634.86</td></tr><tr><td>shield *</td><td>56.67</td><td>43.33</td><td>7229.00</td><td>661.45</td></tr><tr><td>bucket</td><td>53.33</td><td>54.29</td><td>6124.61</td><td>606.23</td></tr><tr><td>iron pickaxe</td><td>50.00</td><td>59.42</td><td>6157.39</td><td>607.87</td></tr><tr><td>iron door</td><td>43.33</td><td>48.28</td><td>5528.00</td><td>676.40</td></tr><tr><td>Average</td><td>52.00</td><td>52.94</td><td>5947.25</td><td>637.36</td></tr><tr><td rowspan="6">Diamond Level</td><td>diamond ore *</td><td>30.00</td><td>9.09</td><td>20782.13</td><td>1039.10</td></tr><tr><td>mind redstone</td><td>20.00</td><td>25.12</td><td>6787.87</td><td>739.39</td></tr><tr><td>compass</td><td>16.67</td><td>15.00</td><td>14908.67</td><td>745.43</td></tr><tr><td>diamond pickaxe</td><td>23.33</td><td>15.71</td><td>32189.50</td><td>1609.48</td></tr><tr><td>piston</td><td>20.00</td><td>28.57</td><td>6457.10</td><td>622.85</td></tr><tr><td>Average</td><td>22.00</td><td>18.70</td><td>14963.83</td><td>948.19</td></tr></table>

Table 16: Results (success rate) on 8 META TASK groups compared with DEPS [49].   

<table><tr><td>Meta-Task</td><td>Task Object</td><td>InnerMonologue</td><td>Code-as-Policy</td><td>DEPS [49]</td><td>Ours</td></tr><tr><td rowspan="7">Basic MT1</td><td>planks</td><td>83.3</td><td>83.3</td><td>83.3</td><td>100.0</td></tr><tr><td>stick</td><td>83.3</td><td>83.3</td><td>86.7</td><td>97.1</td></tr><tr><td>chest</td><td>0.0</td><td>50.0</td><td>76.7</td><td>100.0</td></tr><tr><td>sign</td><td>0.0</td><td>43.3</td><td>86.7</td><td>94.3</td></tr><tr><td>boat</td><td>26.7</td><td>56.7</td><td>73.3</td><td>92.9</td></tr><tr><td>trapdoor</td><td>56.7</td><td>56.7</td><td>76.7</td><td>96.2</td></tr><tr><td>bowl</td><td>23.3</td><td>46.7</td><td>80.0</td><td>100.0</td></tr><tr><td rowspan="10">Tool(Simple) MT2</td><td>crafting_table</td><td>70.0</td><td>70.0</td><td>90.0</td><td>98.5</td></tr><tr><td>wooden_pickaxe</td><td>80.0</td><td>80.0</td><td>80.0</td><td>100.0</td></tr><tr><td>wooden_sword</td><td>83.3</td><td>83.3</td><td>86.7</td><td>100.0</td></tr><tr><td>wooden_shovel</td><td>76.7</td><td>76.7</td><td>90.0</td><td>95.0</td></tr><tr><td>furnace</td><td>40.0</td><td>40.0</td><td>66.7</td><td>93.7</td></tr><tr><td>stone_pickaxe</td><td>36.7</td><td>53.3</td><td>73.3</td><td>96.7</td></tr><tr><td>stone_axe</td><td>30.0</td><td>30.0</td><td>70.0</td><td>96.8</td></tr><tr><td>stone_hoe</td><td>36.7</td><td>56.7</td><td>66.7</td><td>94.6</td></tr><tr><td>stone_shovel</td><td>36.7</td><td>36.7</td><td>66.7</td><td>90.3</td></tr><tr><td>stone_sword</td><td>53.3</td><td>36.7</td><td>80.0</td><td>94.2</td></tr><tr><td rowspan="6">Hunt and Food MT3</td><td>bed</td><td>6.7</td><td>6.7</td><td>43.3</td><td>90.0</td></tr><tr><td>painting</td><td>16.7</td><td>16.7</td><td>86.7</td><td>92.2</td></tr><tr><td>carpet</td><td>0.0</td><td>13.3</td><td>43.3</td><td>91.3</td></tr><tr><td>cooked_porkchop</td><td>0.0</td><td>0.0</td><td>50.0</td><td>90.0</td></tr><tr><td>cooked_beef</td><td>0.0</td><td>0.0</td><td>63.3</td><td>90.0</td></tr><tr><td>cooked_mutton</td><td>0.0</td><td>0.0</td><td>66.7</td><td>90.0</td></tr><tr><td rowspan="5">Dig-down MT4</td><td>stone_stairs</td><td>36.7</td><td>16.7</td><td>66.7</td><td>90.3</td></tr><tr><td>stone_slab</td><td>16.7</td><td>33.3</td><td>73.3</td><td>91.2</td></tr><tr><td>lever</td><td>46.7</td><td>46.7</td><td>83.3</td><td>91.0</td></tr><tr><td>coal</td><td>6.7</td><td>0.0</td><td>20.0</td><td>86.5</td></tr><tr><td>torch</td><td>6.7</td><td>0.0</td><td>13.3</td><td>85.7</td></tr><tr><td rowspan="9">Equipment MT5</td><td>leather_boots</td><td>13.3</td><td>13.3</td><td>60.0</td><td>68.2</td></tr><tr><td>leather_chestplate</td><td>0.0</td><td>6.7</td><td>36.7</td><td>64.2</td></tr><tr><td>leather_helmet</td><td>6.7</td><td>0.0</td><td>70.0</td><td>65.9</td></tr><tr><td>leather-leggedgings</td><td>20.0</td><td>0.0</td><td>56.7</td><td>65.5</td></tr><tr><td>iron_chestplate</td><td>0.0</td><td>0.0</td><td>0.0</td><td>47.2</td></tr><tr><td>iron-leggedgings</td><td>0.0</td><td>0.0</td><td>3.3</td><td>6.6</td></tr><tr><td>iron_helmet</td><td>0.0</td><td>0.0</td><td>3.3</td><td>58.1</td></tr><tr><td>iron_boots</td><td>0.0</td><td>0.0</td><td>20.0</td><td>23.8</td></tr><tr><td>shield</td><td>0.0</td><td>6.7</td><td>13.3</td><td>43.3</td></tr><tr><td rowspan="7">Tool Complex MT6</td><td>bucket</td><td>0.0</td><td>3.3</td><td>6.7</td><td>54.3</td></tr><tr><td>shears</td><td>0.0</td><td>0.0</td><td>30.0</td><td>53.6</td></tr><tr><td>iron_pickaxe</td><td>6.7</td><td>0.0</td><td>10.0</td><td>59.4</td></tr><tr><td>iron_axe</td><td>0.0</td><td>0.0</td><td>16.7</td><td>54.3</td></tr><tr><td>iron_hoe</td><td>0.0</td><td>0.0</td><td>13.3</td><td>52.7</td></tr><tr><td>iron_shovel</td><td>0.0</td><td>0.0</td><td>13.3</td><td>57.8</td></tr><tr><td>iron_sword</td><td>0.0</td><td>3.3</td><td>6.7</td><td>54.7</td></tr><tr><td rowspan="5">Iron-Stage MT7</td><td>iron-bars</td><td>0.0</td><td>0.0</td><td>6.7</td><td>42.0</td></tr><tr><td>hopper</td><td>0.0</td><td>0.0</td><td>6.7</td><td>46.7</td></tr><tr><td>ironDoor</td><td>0.0</td><td>0.0</td><td>3.3</td><td>48.3</td></tr><tr><td>tripwire hook</td><td>6.7</td><td>0.0</td><td>30.0</td><td>48.6</td></tr><tr><td>rail</td><td>0.0</td><td>0.0</td><td>6.7</td><td>42.2</td></tr><tr><td>Challenge MT8</td><td>diamond</td><td>0.0</td><td>0.0</td><td>0.6</td><td>9.1</td></tr></table>

# G Case Study

This section introduces several cases to comprehensively demonstrate Optimus-1’s capabilities.

Figures 9, 10, and 11 demonstrate the superiority of our reflection mechanism, which dynamically adjusts the plan based on the current game progress.

• Figure 9 illustrates Optimus-1’s replanning ability. When Optimus-1 realizes it cannot complete a task (such as a craft failure shown in the figure), it will replan the current task and continue execution.   
• Figures 10 and 11 showcase Optimus-1’s ability to make judgments based on visual signals. When Optimus-1 determines that it has completed a task (such as “kill a cow ” in Figure 10), it will finish the current task and move on to the next one. If Optimus-1 discovers that it has not yet completed the task and the task has not failed(as shown in Figure 11), it will continue executing the task.

Figures 12 and 13 illustrate the advantages of planning with knowledge. With the Hierarchical Directed Knowledge Graph, we can generate a high-quality plan in one step and dynamically adjust the plan based on current visual signals.

• Figure 12 demonstrates the importance of knowledge. For a long-horizon task such as “Mine 1 diamond $\circledcirc$ ,” Optimus-1 first generates a plan based on the Hierarchical Directed Knowledge Graph. However, this plan needs to be adjusted based on the current visual signals. For example, in this figure, Optimus-1 appears in a cave, so the primary task is not to “chop a tree” but to “leave the cave” first. Only after exiting the cave can Optimus-1 proceed with the initial plan.   
• Figure 13 demonstrates the high efficiency of our method. Agents like MP5 [32] and Voyager [46] use an iterative planning approach, which is very time-consuming, generating the final plan step by step. During this process, agent does not take any action. As shown in Figure 13, a zombie is gradually approaching the agent, but the agent is still iterating on its plan. Optimus-1, however, generates the plan in one step based on the Hierarchical Directed Knowledge Graph and makes reasonable plans based on the current visual signals.

![](images/10d3b3730f371d73b277b2de1d33e47377eb515c99ca4ee5bc11bf6edbf7e17e.jpg)

![](images/8cc5c8baac02c11be664dd8c4d359273d6a4894af9ee3df8d075615da9b07a84.jpg)

![](images/f15efcdc5ba54158a5981abdc5f6da96dd214e51b06f32115884338960759dfa.jpg)

![](images/a9c4c8e08fd0ec8751532c31ceb5d5a062e917f73d08f2daefdd1aa22982d576.jpg)

![](images/8fefc64b3d848f5015e3e977ceffc84bc1c2a90363cf6ac19466a83753cda7d9.jpg)

![](images/e81c3738db614479e0589f7fef219c7bf8756e2168e25831e122f900dd9ff7db.jpg)

![](images/15f8c786ffcdeda11217b2cd3b0cb908acb3aa725f9196040661838183073bf3.jpg)

![](images/b8b36b9c67e7c6e61ac05ddb08aa21e0717b2d40a355f20c2a1be48ab3af43c1.jpg)  
4.Craft 1 Crafting table   
5.Craft 1 wooden pickaxe (failed)   
6.Replan: Mine 1 log, Craft 4 planks, Craft 1 wooden pickaxe   
Figure 9: The process of completing the task "Craft 1 wooden pickaxe". Optimus-1 gives wrong planning. When Optimus-1 realizes it cannot complete the task, it will replan the current task.

![](images/816b09e1fa94eb4fdced7768289ff9f0bbc6bbb5192103ab8faf40bcacceff0c.jpg)  
1 .Mine 3 logs

![](images/f509682c93355b5d5e73488a4f3eb3eb4b610753390c33e0311585fd1ad44b0e.jpg)  
2.Craft 1 2 planks

![](images/9dfdeec0ec39f8cc755a1039bc07007f1d9acdba71f78d2079d7c8acfc60574d.jpg)  
3.Craft 4 sticks   
Task: Find a cow and kill it

![](images/e58a7fc93928b7bf0a5ee6bae6d18d08ccbec75722645502c62ecd7daa1bb1a5.jpg)  
Hierarchical Directed Knowledge Graph

![](images/12888c2f508a421c75472c95875d5b12f5e508a29412126863aa534e7a9154bd.jpg)

![](images/699e5169c294edd90b44c93f071129c3d43edcd95dbb369cb8175889db00d0f7.jpg)  
4.Craft 1 Crafting table

![](images/e7c9cbeb89cb143acef43405c5d5eebfc8b095c220c603c31ae9a01da5073318.jpg)  
5.Craft 1 wooden sword

![](images/64afb4a1e811640eb8ff979b4ebc3da65e0c4ba9d374fb8815a9367d88094a2c.jpg)  
6.Done: Find and kill cow   
Figure 10: The process of completing the task "Find a cow and kill it". Hierarchical Directed Knowledge Graph indicates that having a wooden sword will make the task easier to complete. Therefore, Optimus-1 first crafts a wooden sword and then proceeds to find and kill a cow.

![](images/8e9eff3495ba45f67d508030fdbb4aeb016445a20d72a90bf3eb4b67fa1bcebe.jpg)  
1 .Mine 1 0 logs

![](images/a3bfa4369860005446ee6dd0027a7779c1ba418d5cb0e23f11c84ee5a329c155.jpg)  
2.Mine 1 0 logs

![](images/ce3eefb96e7f2f3f813b01db69c1d198af9d5d926c33e5539fc02a48978ae439.jpg)  
3.Mine 1 0 logs   
Task: Chop tree to obtain 10 logs

![](images/98e1f9c900fc591c6423065fec1f8f10d4e54388e0c69c4d4027c9b0dcad2c68.jpg)  
Hierarchical Directed Knowledge Graph

![](images/761274b4c8def0ebbbef70c850b3e2828965ead0a9dd06ef43fd98bacfbde3eb.jpg)  
4.Continue: Mine 1 0 logs

![](images/5909974323e63b1b8b57fd2a9290609dcbb3f75ad015ac813ad90062a565aa27.jpg)  
5.Mine 1 0 logs

![](images/44d9a721ac801134d592cdd982a3e18b4cdd01bec685ff7765b22fdb01dae2be.jpg)  
6.Done: Mine 1 0 logs   
Figure 11: The process of completing the task "Chop tree to obtain 10 logs". Hierarchical Directed Knowledge graph indicates that no tools are needed to complete this goal. After finding a tree, Optimus-1 starts chopping it down. The task requires a substantial amount of wood, so midway through, Optimus-1 performs a reflection. The task is not yet complete but is progressing smoothly, and the result of the reflection is to continue.

![](images/50add57d7cc33356fe97fd4e7e2d94b85123c857e43a5ac768a437ea251eca67.jpg)

![](images/d7f3eb3ea2747f69b2f51db2be778f41211c75c23fe2a9831fc5d92e4762d88d.jpg)

![](images/497c1a504da455142cada297c0b85345dc9ca525c71b7e58951de242e89dcd97.jpg)

![](images/fdf9435228253fcb79dc7c422ac9155e222dc0fd7adf1fb91046ed078abcba5a.jpg)

![](images/97f6fa9f5473d98179874d8094d9f6708f1a615cc08302705d681e82558f9466.jpg)

![](images/0759b9cad447aeae10a1e22fdb2172c8f9d5e0a4c8e1391e8d5e98c5b350397d.jpg)  
3.Craft 1 stone pickaxe   
4.Craft 1 iron pickaxe   
5.Mine 1 diamond

![](images/894cee98aac1c0603505b48c3ce204875e8e7719478ff90bf330b600781c192d.jpg)  
Task: Mine 1 diamond Hierarchical Directed Knowledge Graph

![](images/4a3d6411b717480abda0d9242469628581d82a5c126311885863915dbdefa8e8.jpg)  
Plan Leave the cave

![](images/9b852dd25100060ad5f2430eda7debb08ac5e0c70bb38a8f106d341a12126ac8.jpg)  
Figure 12: The process of completing the task "Mine 1 diamond $\textcircled { \scriptsize { \div } } 1$ ". Mining diamonds $\textcircled { - }$ is a highly complex task. Diamonds can only be mined with an iron pickaxe $\mathcal { \bar { A } }$ , so an iron pickaxe $\mathcal { \bar { A } }$ must be crafted first. Crafting an iron pickaxe $\mathcal { \bar { A } }$ requires iron ingots $\circledast$ , which are smelted from iron ore $\oplus$ Mining iron ore requires a stone pickaxe $\mathcal { \bar { n } }$ . Crafting a stone pickaxe $\mathcal { \bar { A } }$ requires stone , which in turn must be mined with a wooden pickaxe . Crafting a wooden pickaxe $\hat { \lambda }$ requires wooden planks and sticks $\diagup$ . All these crafting processes require a crafting table $\textcircled { 1 }$ , and smelting requires a furnace $\circledast$ . In this case, the agent spawns at a cave, so Optimus-1 must leave the cave to chop logs.

![](images/f9cac4e313ff2f5f77867d867bdb5298a9a3e3d392145cca46d0893b7ea13846.jpg)  
Task: Craft 1 iron sword

![](images/f57e629db19fd4f7d410783aae5354d5e1d1cfc03ab3337ff2d88e5d8b259c6a.jpg)

![](images/85093e8ae485b04e4325768e3b3161d6847ed6e903345bf7d5b5128b5aed783a.jpg)

![](images/d3ef56fb5f3c74529ff505979a4de66892912ed1f8fc338917ebff82a9d46eb7.jpg)  
Figure 13: In this example, a zombie is slowly approaching the agent. Agents like MP5 [32] and Voyager [46] uses an iterative planning strategy to generate the plan, which consumes a great deal of time and puts the agent in danger. While Optimus-1 directly generates a plan in one step based on the knowledge graph. Using the current visual information, it makes a plan to "run to a sunny place," allowing the agent to avoid danger then begin to achieve sub-goals.
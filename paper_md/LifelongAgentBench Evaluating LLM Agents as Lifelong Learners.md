# R LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners

Junhao Zheng1∗, Xidi Cai1∗, Qiuke Li1, Duzhen Zhang2, Zhong-Zhi $\mathbf { L i ^ { 3 } }$ , Yingying Zhang4, Le Song2, Qianli Ma1†

1 South China University of Technology, 2 MBZUAI, 3 Chinese Academy of Sciences, 4 East China Normal University

Project Page Datasets Source Code

# Abstract

Lifelong learning is essential for intelligent agents operating in dynamic environments. Current large language model (LLM)-based agents, however, remain stateless and unable to accumulate or transfer knowledge over time. Existing benchmarks treat agents as static systems and fail to evaluate lifelong learning capabilities. We present LifelongAgentBench, the first unified benchmark designed to systematically assess the lifelong learning ability of LLM agents. It provides skill-grounded, interdependent tasks across three interactive environments—Database, Operating System, and Knowledge Graph—with automatic label verification, reproducibility, and modular extensibility. Extensive experiments reveal that conventional experience replay has limited effectiveness for LLM agents due to irrelevant information and context length constraints. We further introduce a group self-consistency mechanism that significantly improves lifelong learning performance. We hope LifelongAgentBench will advance the development of adaptive, memory-capable LLM agents.

# 1 Introduction

The rapid development of large language models (LLMs) has revolutionized language-based artificial intelligence, achieving state-of-the-art performance across a wide range of natural language processing tasks. Recently, research has shifted from static models to LLM-based agents, designed to interact with dynamic environments, perform complex decision-making, and continuously improve through experience. These agents combine the language understanding and generation capabilities of pretrained LLMs with autonomous action selection and interaction policies.

However, a critical limitation remains: today’s LLM-based agents fundamentally lack memory and the ability to incrementally accumulate knowledge over time. They operate in a stateless manner, treating each task independently without the capacity to remember, adapt, or transfer past experiences. Achieving general artificial intelligence demands agents that can continuously acquire, retain, and reuse knowledge across diverse environments and long time horizons. This lifelong learning capability is widely regarded as a cornerstone of human-level intelligence but remains largely unaddressed in current agent research [20, 21, 11].

Existing LLM agent benchmarks [22, 10, 12] have been designed under the static agent paradigm. They focus on isolated tasks, ignoring inter-task dependencies, skill reuse, and the realistic challenges of knowledge retention and catastrophic forgetting. More critically, there is currently no standardized

Table 1: Comparison between LifelongAgentBench and existing benchmarks. †: [19] highlights label error issues in WebArena.   

<table><tr><td>Benchmark</td><td>WebArena [22] | VisualWebArena [10]</td><td>AgentBench [12]</td><td>VisualAgentBench [13]</td><td>LifelongAgentBench (Ours)</td></tr><tr><td>Task Execution Scheme</td><td colspan="3">Parallel execution with undetermined execution order</td><td>Serial execution with fixed order and historical dependency retention</td></tr><tr><td>Task Dependency</td><td></td><td>X</td><td></td><td>✓</td></tr><tr><td>Knowledge Transfer</td><td></td><td>X</td><td></td><td>✓</td></tr><tr><td>Modular Extension</td><td></td><td>X</td><td></td><td>✓</td></tr><tr><td>Supported Environment</td><td>Web Only</td><td colspan="2">Environment with serializable action and observation</td><td>Environment with serializable action and observation</td></tr><tr><td>Label Verification Mechanism</td><td>Human Annotation †</td><td colspan="2">Automatic Label Verification</td><td>Automatic label verification via human review, LLM judgement, and execution results</td></tr><tr><td>Deployment</td><td>Single Machine Deployment</td><td colspan="2">Distributed Deployment</td><td>Both Single Machine and Distributed Deployment</td></tr><tr><td>Transparent RPC</td><td>/</td><td>X</td><td></td><td>✓</td></tr><tr><td>Code Quality Control</td><td>Not declared</td><td colspan="2">Code are partially check by black, mpy, beatty.</td><td>All code are checked by black and mpy (strict).</td></tr><tr><td># Instance</td><td>812</td><td>910</td><td>1091</td><td>746</td></tr><tr><td></td><td></td><td></td><td></td><td>1396</td></tr></table>

benchmark for systematically evaluating lifelong learning in LLM agents. This absence has severely limited progress toward developing agents capable of lifelong adaptation and memory. Additionally, practical adoption is hindered by label inaccuracies [19], lack of verifiability, and poor reproducibility in prior benchmarks.

To address these critical gaps, we propose LifelongAgentBench, the first unified benchmark specifically designed to evaluate the lifelong learning capabilities of LLM-based agents across realistic and diverse interactive environments (Figure 1). LifelongAgentBench systematically tests agents’ abilities to acquire atomic skills, transfer them across tasks, and maintain stable performance over long sequences of dependent tasks. It includes three task-rich environments—Database (DB), Operating System (OS), and Knowledge Graph (KG)—to simulate complex, evolving scenarios requiring continuous learning.

LifelongAgentBench provides four key innovations that set it apart from existing LLM agent benchmarks: Task Dependency: Tasks are skill-grounded and explicitly designed to quantify intertask relatedness, enabling rigorous analysis of knowledge transfer and catastrophic forgetting. Label Verifiability: Each environment includes automatic label verification (e.g., SQL query validation, OS state hashing, SPARQL output verification) to ensure objective and reproducible evaluation. Reproducibility: The benchmark provides a fully containerized infrastructure and modular design, making it straightforward for researchers to reproduce experiments and extend the framework. Modularity: The platform offers extensible callback functions and a pluggable LLM agent interface, supporting both open-source and commercial models such as LLaMA [5], DeepSeek [7], Qwen [18], and GPT-4 [1]. A detailed comparison between LifelongAgentBench and prior benchmarks is presented in Table 1.

We conduct extensive experiments using LifelongAgentBench, yielding several key insights: (1) While experience replay is effective in traditional continual learning, its impact in agent settings varies significantly depending on model size, architecture, and task complexity. (2) Increasing the volume of past experience does not always improve performance and can even degrade it due to irrelevant information and context length limitations. (3) To mitigate this, we propose a novel group self-consistency mechanism, which partitions historical experience into groups and applies voting strategies to improve decision quality. We show that group self-consistency significantly enhances the effectiveness of experience replay across multiple model backbones.

Our contributions are threefold: (1) We introduce LifelongAgentBench, the first unified benchmark specifically designed to evaluate the lifelong learning capabilities of LLM-based agents across diverse, realistic interactive environments. (2) We provide the first systematic analysis of lifelong learning in LLM agents, revealing key limitations of conventional experience replay due to irrelevant information and context length constraints. (3) We propose a novel group self-consistency mechanism that partitions historical experiences and applies voting strategies, significantly enhancing the effectiveness of lifelong learning across multiple LLM backbones.

# 2 Related Work

Lifelong Learning. Lifelong learning, or continual learning, aims to enable AI systems to acquire and retain knowledge across sequential tasks while mitigating catastrophic forgetting [4]. Prior research has primarily focused on static, non-interactive settings such as image classification or continual instruction tuning, where models are fine-tuned on sequential datasets without interacting with an external environment [21, 20]. In these tasks, both inputs and outputs are fixed, and the model is not required to actively take actions or adapt based on environmental feedback. In contrast, lifelong

![](images/2dd1f061e2977420a74fe3ea4a0b4ea870a0d2f952a92b4ca3e9829541fd3475.jpg)  
Figure 1: Overview of LifelongAgentBench: a unified dataset and evaluation framework with skillgrounded tasks, modular components, and four key properties: task dependency, label verifiability, reproducibility, and modularity.

learning for LLM-based agents interacting with complex environments over long horizons remains largely unexplored.

LLM Agent Benchmarks. Several benchmarks have been proposed to evaluate the capabilities of LLM-based agents. WebArena [22], AgentBench [12], and VisualWebArena [10] offer valuable evaluation settings but focus on single-episode performance in static environments. These platforms lack mechanisms to model sequential decision making, cumulative learning, or skill transfer across tasks. While recent works have explored LLM agents in interactive scenarios such as game playing [3] and tool use [14], they do not provide standardized lifelong learning protocols.

LifelongAgentBench addresses this critical gap by providing the first benchmark specifically designed to evaluate LLM agents under lifelong learning constraints. It introduces reproducible lifelong evaluation with persistent environment states, explicit task dependencies via skill taxonomies, and scalable experience replay, establishing a foundation for systematic study of generalization, skill transfer, and long-term retention in LLM agents.

# 3 Problem Formulation of Lifelong Learning for LLM Agents

We model lifelong learning for LLM-based agents as sequential decision making over a series of tasks, each framed as a goal-conditioned partially observable Markov decision process (POMDP) [21]. Environment: An environment is $\mathcal { E } = ( S , A , \mathcal { G } , T , R , \Omega , O )$ , where $s$ is the state space; $\mathcal { A }$ , natural language actions; $\mathcal { G }$ , task goals; $T$ , state transitions; $R$ , rewards; $\Omega$ , observations; and $O$ , the observation function. LifelongAgentBench provides DB, OS, and KG environments. Agent A task is and Task: An LLM agent follows a policy $\mathcal { T } ^ { ( i ) } = \langle \mathcal { E } ^ { ( i ) } , o _ { 0 } ^ { ( i ) } , g ^ { ( i ) } \rangle$ , with $o _ { 0 } ^ { ( i ) }$ as the initial observation and $\pi : \Omega  A$ mapping observation $g ^ { ( i ) }$ as the goal. The $o _ { t }$ to action $a _ { t }$ . agent generates a trajectory $\xi ^ { ( i ) } = ( o _ { 0 } , a _ { 0 } , r _ { 0 } , \dots , o _ { T } , a _ { T } , r _ { T } )$ , receiving a single reward upon submitting a final answer (success $= 1$ , failure ${ } = 0$ ). Objective: Given tasks $\mathcal { U } = \{ \mathcal { T } ^ { ( 1 ) } , \dots , \mathcal { T } ^ { ( n ) } \} ,$ the goal is to maximize cumulative expected reward: $\begin{array} { r } { \operatorname* { m a x } _ { \pi } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { \xi ^ { ( i ) } \sim \pi } \left[ \sum _ { t = 0 } ^ { T } R ( o _ { t } , a _ { t } , g ^ { ( i ) } ) \right] } \end{array}$ LifelongAgentBench evaluates agents on their ability to leverage past experience to improve current task performance.

# 4 Data Construction

To rigorously evaluate LLM agents in lifelong learning scenarios, we introduce a novel and meticulously constructed benchmark dataset composed of three distinct and challenging environments:

Database, Operating System, and Knowledge Graph. Unlike conventional benchmarks that often rely on isolated, simplistic tasks with loosely defined inter-task relationships, our dataset is innovatively designed to reflect complex, realistic lifelong learning contexts. The key contributions of this dataset include the systematic generation of tasks explicitly tied to clearly defined atomic skills, sophisticated methodologies for controlling skill distribution and task complexity, and rigorous noise management to simulate real-world variability. The dataset’s construction required extensive validation and curation efforts, underscoring the intricacy and robustness of our approach. Detailed descriptions of the construction procedures and sample data are provided in Appendix A and Appendix C, respectively.

# 4.1 Design Principles

The data construction process follows three core principles. First, we adopt a skill-centric task generation approach. Each environment $\mathcal { E } ^ { ( i ) }$ is characterized by a set of atomic skills $\mathcal { S } \mathcal { K } _ { \mathcal { E } ^ { ( i ) } }$ , where the number of skills $N _ { \mathcal { E } ( i ) }$ varies with the environment’s complexity. Each task $\mathcal { T } _ { j } ^ { ( i ) }$ is associated with a subset of these skills SK(j) , $S \mathcal { K } _ { \mathcal { E } ^ { ( i ) } } ^ { ( j ) }$ E(i) ensuring consistent competency representation across tasks. The relationship between tasks $m$ and $n$ is quantified by the harmonic mean of shared skill proportions: $a s _ { \mathcal { E } ^ { ( i ) } } ^ { ( \bar { m } , n ) } = 2 a s _ { \mathcal { E } ^ { ( i ) } } ^ { ( m ) } a s _ { \mathcal { E } ^ { ( i ) } } ^ { ( n ) } / ( a s _ { \mathcal { E } ^ { ( i ) } } ^ { ( m ) } + a s _ { \mathcal { E } ^ { ( i ) } } ^ { ( n ) } )$ 2as where $a s ^ { ( m ) }$ and $a s ^ { ( n ) }$ denote the proportion of shared skills relative to each task’s total skills. This formulation captures both commonality and uniqueness across tasks.

![](images/53f7c5e6191c8b16f5fdb1ec806c1e3dbd023096dc1d32521dbaaae821aaf92c.jpg)  
Figure 2: Skill concurrency in the Database environment.

To mitigate skill isolation, we employ a probabilistic sampling strategy where infrequent skills have higher sampling probabilities, ensuring balanced representation across the dataset. Noise levels are controlled by regulating the proportion of tasks containing rare skills,

facilitating robustness analysis. Tasks span simple, intermediate, and complex configurations to mimic real-world variability and allow evaluation across progressive difficulty levels. As shown in Figure 2, extensive connections exist between skills across tasks. A summarization of the skill set in each enviroment is in Table 7.

# 4.2 Environment Implementations

Database Environment: We implement this environment using Docker-containerized MySQL instances to ensure reproducibility. A fresh MySQL container is created for each experimental run to maintain task isolation. Tasks are initialized by generating a database table with predefined attributes, which is deleted upon task completion. We identify 22 SQL-related skills, including column aliasing, complex filtering with WHERE and HAVING clauses, multi-column grouping, data manipulation (INSERT, UPDATE, DELETE), and nested subqueries. Detailed description of each skill is provided in Appendix A.1.1.

Task construction begins by sampling skills, with infrequent skills prioritized. SQL queries corresponding to sampled skills are generated using the DeepSeek-R1 model. Each query is executed on a synthetic database instance, and invalid or inconsistent tasks are discarded. To prevent skill imbalance, we require a minimum of 20 occurrences per skill across 500 selected tasks. Task correctness is verified both automatically (e.g., result matching, MD5 hashing of database states) and manually by inspecting $10 \%$ of randomly sampled tasks for syntax and logical coherence.

Operating System Environment: This environment leverages disposable Docker containers running Ubuntu to isolate tasks. Containers are destroyed and re-instantiated after each task. We define 29 Bash command skills, including file manipulation (cp, mv, rm), user management (useradd, groupadd), text processing (awk, grep, sed), and system monitoring (ps, top). Tasks are grouped by complexity: simple (1–4 commands), intermediate (5–8), and complex (9–12). Detailed description of each skill is provided in Appendix A.2.1.

![](images/5b54a69fad6a0a36c2cabd5cb7fc707742c18311579b459ee578fc485032afe9.jpg)  
(a) Database

![](images/348f6e75ed71d4804df59c9192b86acd30911e958f8607e7f8fc127b693f66e1.jpg)  
(b) Operating System   
Figure 3: Skill distributions across tasks. Diverse and balanced skill coverage is maintained.

Command sequences are generated using DeepSeek-R1, ensuring logical consistency across multiple steps. Validation scripts automatically compare command outputs against expected results, with file changes verified via checksums. Preliminary experiments revealed that simpler tasks provided limited lifelong learning value; therefore, the final dataset focuses primarily on complex tasks to capture inter-skill dependencies.

Knowledge Graph Environment: This environment is based on a SPARQL query system. Tasks involve querying structured data through operations such as relation extraction and intersection. Tasks were curated from the GrailQA dataset [6] by mapping S-expressions to logical action sequences. These sequences range from 2 to 9 steps to ensure uniform distribution across task lengths. Each query is validated on a synthetic knowledge graph to confirm result correctness. Complex queries (7–9 steps) received additional manual validation to ensure semantic accuracy. Detailed description of each skill is provided in Appendix A.3.1.

# 4.3 Quality Control

Label Validation: We employ automated validation mechanisms, including result comparison for SQL queries (Appendix A.1.2, Figure 5), exit code checking for Bash commands (Appendix A.2.2, Figure 6), and output verification for SPARQL queries (Appendix A.3.2). Additionally, $10 \%$ of tasks from each environment were manually reviewed for logical consistency and practical relevance. Pilot testing informed the final configuration to optimize task complexity and skill coverage. This multi-stage validation ensures that datasets are both challenging and representative of real-world scenarios.

Balanced Skills: In the Database environment, we generated 1,306 tasks with DeepSeek-R1 and selected 500 high-quality samples. Tasks cover 22 SQL skills with balanced distributions ensured by stratified sampling (Appendix A.1.2, Figure 3a). In the Operating System environment, 500 complex tasks were curated, with command sequences ranging from 9 to 12 steps to maximize inter-task skill overlap. Lower complexity tasks (1–8 steps) were excluded after preliminary tests showed minimal replay benefits (Figure 3b). In the Knowledge Graph environment, 396 tasks were extracted from GrailQA, mapped to atomic action sequences ranging from 2 to 9 steps. Replay effects were observed to diminish in sequences exceeding six steps.

# 5 Evaluation Framework

LifelongAgentBench is designed as a unified evaluation framework that integrates datasets and APIs to benchmark LLM-based agents under lifelong learning settings. In contrast to prior benchmarks that focus on static or single-task evaluation, our framework emphasizes sequential task execution and experience accumulation, offering a realistic simulation of continual learning scenarios. The detailed description is provided in Section B.

# 5.1 System Architecture

The framework comprises six loosely coupled components: model pool (Appendix B.1.1), agent (B.1.2), environment (B.1.3), chat history factory (B.1.4), controller (B.1.5), and callbacks (B.1.6). Each component can be deployed independently across different servers and communicates via a

custom remote procedure call (RPC) toolkit (B.2.1), enabling flexible distributed or local deployment (B.2.2).

The model pool maintains mappings between model names and instances, supporting both opensource and proprietary LLM backends. The agent module translates environment observations and dialogue history into formatted inputs, queries the LLM, and parses outputs into executable actions. The environment component executes these actions and returns updated observations to the controller. It also implements standardized methods such as reset, interact, complete, calculate_metric, and release to ensure consistency across environments.

The controller manages the interaction loop, oversees task scheduling, and relays agent actions to the environment. The callback system provides extensible hooks for monitoring internal events, facilitating reproducibility and experimental customization.

# 5.2 Reproducibility and Modularity

Two core design principles of LifelongAgentBench are reproducibility and modularity. The framework guarantees deterministic behavior under fixed random seeds and uses containerized environment snapshots to ensure identical task conditions across experimental runs. Additionally, it exposes modular APIs for integrating new environments, task generators, custom agent architectures, or evaluation metrics with minimal engineering overhead. This flexibility allows researchers to experiment with various lifelong learning strategies while maintaining consistency and comparability.

# 5.3 Differences from Prior Benchmarks

Existing LLM agent benchmarks such as WebArena and AgentBench either rely on parallel task execution to reduce evaluation time or use process pools to manage multiple task sequences concurrently. These designs are incompatible with lifelong learning evaluation, where the strict order of task execution directly impacts the agent’s accumulated knowledge and performance.

In contrast, LifelongAgentBench enforces strict sequential execution to preserve the integrity of experience accumulation and transfer learning assessments. Moreover, while prior frameworks tightly coupled agents, controllers, and environments into complex multi-process architectures, our design promotes developer-friendly single-process debugging with optional distributed scalability. This architecture substantially lowers the barrier to conducting lifelong learning research with LLM agents. The difference is summarized in Table 1.

# 6 Evaluation of LifelongAgentBench

We comprehensively evaluate the lifelong learning abilities of LLM-based agents using LifelongAgentBench across three environments: Database (DB), Operating System (OS), and Knowledge Graph (KG). Our experiments systematically investigate experience replay and group self-consistency under unified protocols.

# 6.1 Experimental Setup

Models: We evaluate four LLM-based agents: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, DeepSeek-R1-Distill-Llama-8B, and DeepSeek-R1-Distill-Qwen-7B. All agents share a unified API with reproducible initialization, dynamic experience replay, and optional group self-consistency. Baselines and Metric: We evaluate agents under baseline (no replay), experience replay (1, 2, 4, 8, 16 prior trajectories), and experience replay with group self-consistency. Prior experiences are retrieved from recent successful trajectories. The evaluation metric is task success rate, defined as correct action sequences completing the task. Environments: Experiments run on Linux servers NVIDIA A800 (80GB). Code is based on Huggingface Transformers and PyTorch, with a distributed RPC-based framework for modular deployment. The system supports automatic checkpointing for recovery from interruptions.

Table 2: Main Result of LifelongAgentBench . The backbone model is Llama-3.1-8B-Instruct. "Exp" represents that the number of recent successful trajectories that are provided to agent. The best result for each environment is bold. "OOM" represents out of memory.   

<table><tr><td></td><td>Exp=0</td><td>Exp=1</td><td>Exp=4</td><td>Exp=16</td><td>Exp=32</td><td>Exp=64</td></tr><tr><td>DB</td><td>0.19</td><td>0.41</td><td>0.73</td><td>0.75</td><td>0.77</td><td>0.78</td></tr><tr><td>OS</td><td>0.43</td><td>0.46</td><td>0.50</td><td>0.50</td><td>0.42</td><td>0.44</td></tr><tr><td>KG</td><td>0.28</td><td>0.35</td><td>0.33</td><td>OOM</td><td>OOM</td><td>OOM</td></tr></table>

Table 3: The result when using different backbone LLM. The environment is DB. The best result for each backbone LLM is bold. "OOM" represents out of memory.   

<table><tr><td>Model</td><td>Exp=0</td><td>Exp=1</td><td>Exp=4</td><td>Exp=16</td><td>Exp=32</td><td>Exp=64</td></tr><tr><td>DeepSeek-R1-Distill-Llama-8B</td><td>0.07</td><td>0.13</td><td>0.35</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-7B</td><td>0.10</td><td>0.12</td><td>0.18</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>QwQ-32B</td><td>0.29</td><td>0.23</td><td>0.21</td><td>0.25</td><td>0.31</td><td>OOM</td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>0.74</td><td>0.71</td><td>0.76</td><td>0.74</td><td>OOM</td><td>OOM</td></tr><tr><td>Qwen2.5-32B-Instruct</td><td>0.82</td><td>0.77</td><td>0.71</td><td>0.72</td><td>0.74</td><td>OOM</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>0.19</td><td>0.41</td><td>0.73</td><td>0.75</td><td>0.77</td><td>0.78</td></tr><tr><td>Llama-3.1-70B-Instruct</td><td>0.81</td><td>0.83</td><td>0.86</td><td>0.88</td><td>0.88</td><td>0.90</td></tr></table>

# 6.2 Main Results

Strong performance of open-source models. The results in Table 2 demonstrate that the open-source Llama-3.1-8B-Instruct model achieves reasonable and stable performance across all environments. This contrasts with prior benchmarks such as AgentBench, where open-source LLMs often underperform, limiting academic reproducibility and accessibility.

Experience replay consistently improves performance. Incorporating past successful trajectories consistently boosts agent performance compared to the baseline $\operatorname { ( E x p = 0 }$ ). For DB, replay increases accuracy from $19 \%$ to $78 \%$ at 64 examples. For OS environment, accuracy improves from $43 \%$ to $50 \%$ at 4–16 examples. For KG, accuracy improves from $28 \%$ to $3 5 \%$ with just 1 example.

Trade-off between replay benefits and memory limitations. Increasing replay beyond optimal values leads to diminishing or negative returns due to excessive input length, increased reasoning complexity, and out-of-memory (OOM) failures. DB tasks, which involve shorter trajectories, benefit from large replay buffers. In contrast, OS and KG tasks, with multi-turn and long-form interactions, show peak performance at lower replay sizes before degradation or OOM occurs.

Memory-efficient replay remains an open challenge. These findings suggest that while experience replay is a valuable mechanism for improving LLM agent performance, it introduces significant memory and inference costs. Designing more efficient retrieval and summarization strategies for lifelong learning remains an important avenue for future research.

# 6.3 Effect of Model Backbone and Task Difficulty

Backbone LLMs. We evaluate a range of open-source LLMs across different architectures and model scales. The results in Table 3 reveal notable differences in how backbone choice influences the effectiveness of experience replay. For strong base models such as Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct, the performance gain from adding prior experience is small or even negative. In particular, Qwen2.5-32B-Instruct achieves 0.82 accuracy without replay, with subsequent replay settings showing no clear improvement. In contrast, the Llama-3.1 series shows consistent and stable gains as more experience is provided, with Llama-3.1-70B-Instruct reaching 0.90 accuracy with 64 examples. This indicates that model architecture may substantially impact the utility of experience replay: some models may inherently learn well from single episodes, while others benefit more from historical trajectories.

Reasoning LLMs vs. non-Reasoning LLMs. Models designed for complex reasoning, such as DeepSeek-R1-Distill-Llama-8B and DeepSeek-R1-Distill-Qwen-7B, perform significantly worse and are prone to OOM failures at large replay sizes. These reasoning-optimized models tend to generate verbose thought chains and redundant intermediate outputs, which increase input length and can confuse the execution environment [2]. This highlights a key difference between LifelongAgentBench and prior benchmarks such as LiveCodeBench [8] and GPQA [15], where complex multi-hop reasoning is the primary objective. In contrast, LifelongAgentBench emphasizes efficient skill acquisition and knowledge reuse from past interactions.

Model Size. We observe a scaling trend where larger backbones consistently outperform their smaller counterparts across most replay settings. Llama-3.1-70B-Instruct demonstrates superior robustness, achieving the highest accuracy (0.90) without encountering OOM even at 64 replay examples. Interestingly, medium-scale models such as Llama-3.1-8B-Instruct achieve comparable performance (0.78 at 64 examples), suggesting that with careful experience replay and model tuning, smaller models can approach the performance of much larger ones while offering substantially lower computational cost.

# 6.4 Effect of Task Difficulty

Experience replay helps most on complex tasks. We first analyze the DB environment, where task difficulty is manually categorized as Easy, Medium, or Hard based on required SQL skill combinations. As shown in Table 4, experience replay provides marginal gains on Easy tasks $70 \%$ to $76 \%$ ) but leads to substantial improvements on Hard tasks ( $49 \%$ to $62 \%$ ). This suggests that replay is particularly valuable when agents face complex, multi-skill reasoning where prior examples offer useful reference points.

Task length strongly correlates with replay benefit in KG. In the KG environment, task difficulty is naturally reflected by the length of the ground-truth action sequence. Table 5 shows that short tasks (length 2–4) benefit significantly from replay (e.g., length 2 improves from $48 \%$ to $84 \%$ ), while longer tasks (length 7–9) see minimal or no improvement. As trajectory length increases, the added experience creates longer input sequences, which reduces the effective signal-to-noise ratio and increases the risk of context overflow or degraded performance.

Table 4: Performance on different difficulty levels in DB.   

<table><tr><td>Difficulty</td><td>Exp=0</td><td>Exp=1</td><td>Exp=2</td><td>Exp=8</td></tr><tr><td>Easy</td><td>0.70</td><td>0.76</td><td>0.71</td><td>0.75</td></tr><tr><td>Medium</td><td>0.56</td><td>0.53</td><td>0.59</td><td>0.54</td></tr><tr><td>Hard</td><td>0.49</td><td>0.59</td><td>0.57</td><td>0.62</td></tr></table>

LifelongAgentBench sensitively captures replay–difficulty interactions. Overall, these findings demonstrate that LifelongAgentBench provides a fine-grained benchmark to study how prior experience impacts learning under varying task difficulty. While experience replay is highly beneficial for short, well-bounded tasks, it poses scalability challenges for long-horizon tasks. Designing more effective memory compression, filtering, or retrieval strategies to handle these cases remains an important direction for future research.

Table 5: Performance by action sequence length in KG under different replay sizes.   

<table><tr><td># Ground-Truth Actions</td><td># Task</td><td>Exp=0</td><td>Exp=1</td><td>Exp=4</td><td>Exp=16</td></tr><tr><td>2</td><td>19114</td><td>0.48</td><td>0.72</td><td>0.78</td><td>0.84</td></tr><tr><td>3</td><td>1481</td><td>0.52</td><td>0.70</td><td>0.72</td><td>0.72</td></tr><tr><td>4</td><td>7067</td><td>0.56</td><td>0.56</td><td>0.72</td><td>0.78</td></tr><tr><td>5</td><td>1740</td><td>0.16</td><td>0.30</td><td>0.30</td><td>0.44</td></tr><tr><td>6</td><td>626</td><td>0.14</td><td>0.30</td><td>0.10</td><td>0.14</td></tr><tr><td>7</td><td>592</td><td>0.04</td><td>0.04</td><td>0.08</td><td>0.08</td></tr><tr><td>8</td><td>63</td><td>0.08</td><td>0.12</td><td>0.08</td><td>0.08</td></tr><tr><td>9</td><td>46</td><td>0.11</td><td>0.13</td><td>0.13</td><td>0.17</td></tr></table>

# 6.5 Scaling Experience with Group Self-Consistency

Group self-consistency reduces memory and stabilizes performance. To mitigate the memory and inference overhead of large-scale experience replay, we propose group self-consistency, which splits retrieved experiences into smaller groups and aggregates their predictions using self-consistency voting [17]. An illustration is provided in Figure 4. The experimental results are summarized in Tables 6.

Significant accuracy gains in DB. In DB, group self-consistency provides clear performance improvements as replay size increases. Llama-3.1-8B-Instruct achieves 0.75 accuracy with 16 groups

![](images/6ea28116f701bb2821541922e1e61f1d3622280e64fac3f286c4e56518d9c0f3.jpg)  
Figure 4: Illustration of group self-consistency.

Table 6: Comparison of the accuracy (average input tokens) under different group self-consistency settings.   

<table><tr><td rowspan="2">Environment</td><td colspan="2"># Experience</td><td>0</td><td>1</td><td colspan="3">4</td><td colspan="3">16</td></tr><tr><td># Groups</td><td>/</td><td>1</td><td>1</td><td>2</td><td>4</td><td>1</td><td>4</td><td>16</td><td></td></tr><tr><td rowspan="4">DB</td><td>Llama-3.1-BB-Instruct</td><td>0.16 (1128)</td><td>0.51 (1773)</td><td>0.63 (4189)</td><td>0.57 (2750)</td><td>0.59 (2512)</td><td>0.61 (17874)</td><td>0.70 (6008)</td><td>0.75 (2888)</td><td></td></tr><tr><td>DeepSeek-R1-Distill-Llama-8B</td><td>0.06 (1213)</td><td>0.11 (2706)</td><td>0.13 (6542)</td><td>0.12 (4645)</td><td>0.13 (3469)</td><td>0.12 (27737)</td><td>0.13 (10365)</td><td>0.18 (4932)</td><td></td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>0.76 (951)</td><td>0.69 (1567)</td><td>0.68 (3952)</td><td>0.73 (2445)</td><td>0.73 (2292)</td><td>0.72 (16383)</td><td>0.71 (5443)</td><td>0.77 (2685)</td><td></td></tr><tr><td>DeepSeek-R1-Distill-Qwen-7B</td><td>0.11 (1232)</td><td>0.09 (2353)</td><td>0.16 (5830)</td><td>0.26 (3861)</td><td>0.12 (2292)</td><td>0.15 (18845)</td><td>0.23 (6113)</td><td>0.15 (2744)</td><td></td></tr><tr><td rowspan="2">KG</td><td>Llama-3.1-BB-Instruct</td><td>0.27 (2978)</td><td>0.32 (9390)</td><td>0.26 (16420)</td><td>0.33 (12175)</td><td>0.36 (8842)</td><td>0.32 (56409)</td><td>0.34 (22191)</td><td>0.34 (11002)</td><td></td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>0.22 (2858)</td><td>0.27 (7150)</td><td>0.19 (18436)</td><td>0.29 (10998)</td><td>0.30 (7522)</td><td>0.19 (59950)</td><td>0.19 (18539)</td><td>0.36 (11339)</td><td></td></tr></table>

(16 examples), compared to only 0.61 without grouping. Qwen2.5-7B-Instruct similarly benefits, improving from 0.72 to 0.77 accuracy. Smaller models such as DeepSeek-R1-Distill variants show limited gains, likely constrained by reduced model capacity.

Drastic memory savings in KG tasks. In KG, where experience trajectories are much longer, group self-consistency dramatically reduces input token lengths. For Llama-3.1-8B-Instruct, token usage at 16 examples drops from 56,409 tokens (no grouping) to 11,002 tokens (16 groups), while maintaining stable accuracy. Qwen2.5-7B-Instruct shows similar trends, decreasing from 59,950 to 11,339 tokens with minimal accuracy loss.

Group self-consistency offers a scalable replay strategy. Overall, group self-consistency consistently stabilizes performance and alleviates memory bottlenecks across environments and models. Its simplicity and effectiveness make it a promising technique to enhance lifelong learning scalability for LLM agents under heavy replay loads. Future work may explore dynamic or adaptive grouping strategies that optimize the trade-off between experience diversity, inference cost, and available context window.

# 6.6 Failure Mode Analysis

To understand the failure patterns of LLM agents in LifelongAgentBench, we classify all task outcomes based on agent behavior and system status. We observe four common failure modes: (1) Incorrect final submission: the agent outputs an answer in the correct format but with wrong content (completed, Appendix D.1). (2) Failure to commit: the agent completes multiple operations but never explicitly submits the final answer (task_limit_reached, Appendix D.2). (3) Format violation: the agent violates the required output format or instruction pattern (agent_validation_failed, Appendix D.4). (4) Context overflow: the agent exceeds the LLM context window due to excessive interactions or large intermediate outputs (agent_context_limit, Appendix D.5). These results reveal key limitations of current LLM agents in multi-step interactive tasks: unstable reasoning, poor instruction adherence, and context management issues. Detailed examples of typical cases are provided in Appendix D.

# 7 Conclusion, Limitation, and Future Works

We present LifelongAgentBench, the first unified benchmark specifically designed to evaluate the lifelong learning capabilities of LLM-based agents. Unlike prior benchmarks that treat agents as static systems, LifelongAgentBench systematically measures agents’ ability to accumulate, retain, and transfer knowledge across diverse interactive environments. Our experiments demonstrate the potential of experience replay and group self-consistency to improve agent performance, while also revealing critical challenges.

Despite these advances, limitations remain. Experience replay introduces substantial memory and context length overhead, especially on long-horizon tasks. Performance also varies across model architectures, with smaller or reasoning-optimized models benefiting less from replay.

LifelongAgentBench establishes a standardized platform for studying continual adaptation in agents, providing clear baselines and diagnostic tools to facilitate further research. We hope this work will inspire the development of more scalable, robust, and memory-efficient lifelong learning agents. Promising directions include more efficient memory retrieval strategies, dynamic experience selection, and extending the benchmark to multi-modal and real-world agent tasks.

# References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. ArXiv preprint, abs/2303.08774, 2023.   
[2] Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, et al. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. ArXiv preprint, abs/2412.21187, 2024.   
[3] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De-An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open-ended embodied agents with internet-scale knowledge. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022.   
[4] Robert M French. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128–135, 1999.   
[5] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. ArXiv preprint, abs/2407.21783, 2024.   
[6] Yu Gu, Sue Kase, Michelle Vanni, Brian M. Sadler, Percy Liang, Xifeng Yan, and Yu Su. Beyond I.I.D.: three levels of generalization for question answering on knowledge bases. In Jure Leskovec, Marko Grobelnik, Marc Najork, Jie Tang, and Leila Zia, editors, WWW ’21: The Web Conference 2021, Virtual Event / Ljubljana, Slovenia, April 19-23, 2021, pages 3477–3488. ACM / IW3C2, 2021.   
[7] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. ArXiv preprint, abs/2501.12948, 2025.   
[8] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. ArXiv preprint, abs/2403.07974, 2024.   
[9] Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. Tree search for language model agents. ArXiv preprint, abs/2407.01476, 2024.   
[10] Jingxiang Koh et al. Visualwebarena: Benchmarking vision-language agents for web interaction. ArXiv preprint, abs/2402.09556, 2024.   
[11] Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, et al. From system 1 to system 2: A survey of reasoning large language models. arXiv preprint arXiv:2502.17419, 2025.

[12] Han Liu et al. Agentbench: Benchmarking llms as agents. ArXiv preprint, abs/2308.04035, 2023.   
[13] Xiao Liu, Tianjie Zhang, Yu Gu, Iat Long Iong, Yifan Xu, Xixuan Song, Shudan Zhang, Hanyu Lai, Xinyi Liu, Hanlin Zhao, et al. Visualagentbench: Towards large multimodal models as visual foundation agents. ArXiv preprint, abs/2408.06327, 2024.   
[14] Libo Qin et al. Toolllm: Facilitating language model reasoning with tool understanding. ArXiv preprint, abs/2309.03409, 2023.   
[15] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024.   
[16] Runchu Tian, Yining Ye, Yujia Qin, Xin Cong, Yankai Lin, Yinxu Pan, Yesai Wu, Haotian Hui, Weichuan Liu, Zhiyuan Liu, et al. Debugbench: Evaluating debugging capability of large language models. ArXiv preprint, abs/2401.04621, 2024.   
[17] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023.   
[18] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. ArXiv preprint, abs/2412.15115, 2024.   
[19] Ke Yang, Yao Liu, Sapana Chaudhary, Rasool Fakoor, Pratik Chaudhari, George Karypis, and Huzefa Rangwala. Agentoccam: A simple yet strong baseline for llm-based web agents. In The Thirteenth International Conference on Learning Representations, 2025.   
[20] Junhao Zheng, Shengjie Qiu, Chengming Shi, and Qianli Ma. Towards lifelong learning of large language models: A survey. ACM Computing Surveys, 57(8):1–35, 2025.   
[21] Junhao Zheng, Chengming Shi, Xidi Cai, Qiuke Li, Duzhen Zhang, Chenxing Li, Dong Yu, and Qianli Ma. Lifelong learning of large language model based agents: A roadmap. ArXiv preprint, abs/2501.07278, 2025.   
[22] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations, 2024.

# Appendix

# A Lifelong Learning Environment Design 12

A.1 Database . 12   
A.2 Operating System . . 14   
A.3 Knowledge Graph . 15

# B Evaluation Framework Design 16

B.1 Framework Architecture 17   
B.2 Differences Between Distributed Deployment and Single-Machine Deployment . . 20   
B.3 The Execution Process of a Task 24

# C Datasets Samples 24

C.1 Database Samples . 24   
C.2 Operating System Samples . . 27   
C.3 Knowledge Graph Samples . . 28

# D Error Case Analysis 30

D.1 Task Completed 31   
D.2 Task Limited Reached . 35   
D.3 Error Case 4 . 37   
D.4 Agent Validation Failed . 40   
D.5 Agent Context Limit 43

# A Lifelong Learning Environment Design

Table 7: Skill set in each environment.   

<table><tr><td>Environment</td><td>Skills</td></tr><tr><td>DB</td><td>column alias, delete, group_by_multiple.columns, group_by_single_column, having_aggregate Calculation, having_multiple_conditions with_aggregate, having_single_condition with_aggregate, insert, limit_and_offset, limit_only, order_by_multiple.columns_different Directions, order_by_multiple.columns_sameDirection, order_by_single_column, select, subquery_multiple, subquery_nested, ubquery_single, tablealias, update, where_multiple_conditions, where_nested_conditions, where_single_condition</td></tr><tr><td>OS</td><td>addgroup, awk, cat, cd, chage, chgrp, chmod, chown, chsh, cp, echo, exit, find, gppasswd, grep, groupadd, ln, ls, mkdir, mv, rm, sed, sleep, tee, touch, useradd, usermod, vi, wc</td></tr><tr><td>KG</td><td>get Relations, get_neighbors, intersection, get_attributes, argmax, argmax, count</td></tr></table>

In this section, we introduce the design principle and data construction process of each lifelong learning environment.

# A.1 Database

# A.1.1 Environment Introduction

To ensure experiment reproducibility and reduce deployment complexity in the database environment, we encapsulate the MySQL database using Docker. At the start of each evaluation, the framework initializes a new Docker container from a predefined image. For each task, new tables are created based on task-specific information prior to execution and deleted upon task completion. This guarantees that modifications made during a task do not interfere with subsequent tasks. To optimize efficiency, the Docker container is launched only once at the beginning of the evaluation, rather than

being recreated for each task, as the initialization of a MySQL container is time-consuming and frequent restarts would significantly increase overall evaluation time.

During each task, the agent interacts with the database based on the given goal $g$ , performing operations such as querying, inserting, deleting, or updating records. For query tasks, the agent must submit the result at the end of the task, and the evaluation framework determines task success by comparing the submitted result with the ground truth. For other task types, the framework calculates a hash of the database state at the end of the task and compares it with a ground truth hash stored in the database to assess correctness.

In total, we define 22 skills within the database environment. The names and descriptions of these skills are as follows:

• column_alias: Assigning an alias to a column in a SELECT statement, e.g., SELECT name AS employee_name.   
• delete: Deleting data using the DELETE statement.   
• group_by_multiple_columns: Using the GROUP BY clause to group results by multiple columns.   
• group_by_single_column: Using the GROUP BY clause to group results by a single column.   
• having_aggregate_calculation: Including calculations on aggregate function results in the HAVING clause, e.g., MAX(salary) - MIN(salary).   
• having_multiple_conditions_with_aggregate: Including multiple conditions involving aggregate functions in the HAVING clause.   
• having_single_condition_with_aggregate: Including a single condition based on an aggregate function in the HAVING clause.   
• insert: Inserting data using the INSERT statement.   
• limit_and_offset: Using both LIMIT and OFFSET clauses together.   
• limit_only: Using only the LIMIT clause to restrict the number of returned rows, without using OFFSET.   
• order_by_multiple_columns_different_directions: Sorting results by multiple columns with different sorting directions in the ORDER BY clause.   
• order_by_multiple_columns_same_direction: Sorting results by multiple columns with the same sorting direction in the ORDER BY clause.   
• order_by_single_column: Sorting results by a single column in the ORDER BY clause.   
• select: Querying data using the SELECT statement.   
• subquery_multiple: Having multiple subqueries in the main query.   
• subquery_nested: Nesting one subquery inside another subquery.   
• subquery_single: Including a single subquery in the SELECT, WHERE, or HAVING clause that involves only one table.   
• table_alias: Assigning an alias to a table in the FROM clause, e.g., FROM employees AS e.   
• update: Updating data using the UPDATE statement.   
• where_multiple_conditions: Including multiple conditions connected by AND or OR in the WHERE clause.   
• where_nested_conditions: Including nested logical conditions in the WHERE clause, e.g., WHERE (A AND B) OR C.   
• where_single_condition: Including a single condition in the WHERE clause, e.g., WHERE id $= ~ 5$ .

# A.1.2 Data Construction Process

The task construction process in the database environment is illustrated in Figure 5. To prevent data leakage, we adopt the methodology proposed in [16], leveraging a large language model to generate the benchmark data.

![](images/1e673dae4c66dbc1d882093fcfe7d06be24400af9bd4682a9b6df38d0a4f8890.jpg)  
Figure 5: The data construction process in database environment.

During data construction, we begin by specifying a list of skills that each task should incorporate. To ensure balanced coverage, skills with lower occurrence frequencies are assigned higher sampling probabilities. Additionally, we enforce a minimum occurrence threshold for each skill across the dataset to avoid the presence of isolated tasks.

# A.2 Operating System

# A.2.1 Environment Introduction

In the operating system environment, we also use Docker containers to encapsulate the Ubuntu system with which the agent interacts. Due to the wide variety and broad impact of operations that the agent can perform in this environment, it is difficult to fully roll back the system state to its initial condition after each task. To address this, we destroy the current container instance after each task and instantiate a new one from the original image before the next task begins. This approach ensures both experimental reproducibility and isolation between tasks.

In the operating system environment, the agent is required to generate shell scripts composed of Bash commands based on the given goal $g$ , in order to interact with the system. After each operation, the agent can choose either to continue issuing shell commands or to terminate the interaction by outputting a string that matches a predefined pattern.

Similarly, if the number of interactions between the agent and the environment reaches a predefined limit, the evaluation framework automatically terminates the interaction. Regardless of the reason for termination, the framework then determines whether the agent has successfully completed the task. Each task is associated with an evaluation script consisting of Bash commands. If the agent completes the task successfully, the script returns 0; otherwise, it returns a non-zero value. The success of the task is thus determined based on the return value of the evaluation script.

In the operating system environment, each task is associated with a specific skill, determined based on the ground truth Bash command required to solve the task. To ensure controlled and consistent evaluation, we restrict the agent to a predefined set of 29 Bash commands during task execution, with each command corresponding to a distinct skill. The complete list of these 29 skills is presented in Table 7.

# A.2.2 Data Construction Process

The task construction process in the operating system environment is illustrated in Figure 6. As in the database environment, we use a large language model to generate the tasks in the dataset.

To reduce token costs, however, we minimize the number of calls to the language model during data construction in the operating system environment compared to the database environment.

![](images/89bb45251270173bf7a2ac178e8eb7c80c1587419b11a19214b1b9971b8ba0b2.jpg)  
Figure 6: The data construction process in operating system environment.

# A.3 Knowledge Graph

# A.3.1 Environment Introduction

In the knowledge graph environment, the agent is tasked with achieving the goal $g$ by interacting with the knowledge graph, given both the goal and the entities involved. While humans typically query a knowledge graph using SPARQL statements, generating a complete SPARQL query in a single step is relatively challenging for an agent. This difficulty arises from the agent’s lack of basic knowledge about the graph structure—such as entity types, outgoing relations, and incoming relations. Directly including all this information in the prompt results in excessively long inputs, which increases the risk of the agent overlooking critical details [19]. Moreover, such a direct approach to knowledge graph access is impractical for real-world applications.

Following the approach proposed in [12], we use a constrained set of actions that the agent can execute. This design enables the agent to progressively and autonomously explore relevant information within the knowledge graph and achieve the goal $g$ through a sequence of actions.

During the interaction between the agent and the knowledge graph environment, the environment returns sets of relations and entities based on the actions executed by the agent. Because the agent must reference previously retrieved relations to construct new actions in subsequent steps, all queried relations are returned in full. For entity sets, we use the concept of variables [12] to represent the results. Each variable is associated with the SPARQL statement that produced it, and instead of returning the set of queried entities directly, the environment returns the corresponding variable name to the agent. This mechanism allows the agent to conveniently reference previously queried entity sets in future actions, thereby enabling multi-hop reasoning over the knowledge graph.

In the knowledge graph environment, the set of available actions and their corresponding parameters is defined as follows. To complete a given task, the agent must execute a sequence of actions to achieve the goal $g$ . Accordingly, we treat the actions appearing in the ground truth action sequence of each task as the task’s skills.

• get_relations: Given an entity or variable, returns all outgoing relations connected to it.   
• get_neighbors: Given an entity or variable and an outgoing relation, returns the set of entities reachable via that relation, represented as a new variable.   
• intersection: Given two variables, returns a new variable representing the intersection of entities contained in both.   
• get_attributes: Given a variable, returns all attributes associated with the entities in that variable.   
• argmax: Given a variable and an attribute, returns a new variable containing the entity (or entities) with the maximum value for the specified attribute.   
• argmin: Given a variable and an attribute, returns a new variable containing the entity (or entities) with the minimum value for the specified attribute.   
• count: Given a variable, returns the number of entities it contains.

# A.3.2 Data Construction Process

In the knowledge graph environment, tasks generated solely by large language models often lack practical relevance. To address this, we construct tasks based on existing knowledge graph question answering datasets. Specifically, we adopt the dataset introduced in [6] and convert the S-expressions representing graph queries into ground truth action sequences for each task.

# B Evaluation Framework Design

After constructing the environment and tasks, it is essential to develop a robust evaluation framework to facilitate the interaction between LLM agents and the environment. In designing our framework, we drew inspiration from widely-used web-based agent benchmarks such as WebArena[22], VisualWebArena[10], as well as non-lifelong learning evaluation frameworks like AgentBench[12] and VisualAgentBench[13]. However, these frameworks commonly suffer from several limitations, including long evaluation times, incompatibility with lifelong learning scenarios, and the lack of a unified development interface.

First, the evaluation time for a single sample is often prolonged. Take WebArena as an example. WebArena is a benchmark designed to assess the performance of LLM agents in real-world web environments, encapsulating each deployed website within a Docker container. However, due to the complexity of the webpages used, instantiating each Docker container takes approximately one minute. To mitigate this issue, WebArena claims to employ a carefully orchestrated task sequence such that the execution of a preceding task does not affect subsequent ones. It therefore recommends initializing Docker containers only once before evaluation begins, rather than prior to each task. However, given that agent behaviors during evaluation are inherently unpredictable, determining whether tasks interfere with each other solely based on their goals $g$ is insufficient. Additionally, WebArena uses the accessibility tree of webpages as the environment observation. During the conversion to the accessibility tree, the evaluation framework traverses every HTML element in the DOM tree and maps it to a corresponding node. Even when using proprietary LLMs, the interaction time per task can reach up to ten minutes; for locally deployed open-source models, this duration increases further due to hardware constraints.

Second, these frameworks typically suggest evaluating agent performance via parallel execution of tasks, which is unsuitable for lifelong learning scenarios. For example, WebArena recommends partitioning the task sequence and running different sequences concurrently on separate servers. AgentBench manages all tasks using a central list and evaluates agent performance in parallel by distributing tasks across multiple processes in a process pool. When one process completes its assigned task, the framework automatically pulls the next one from the list. While these strategies reduce evaluation time, they fail to preserve task execution order—an essential factor in lifelong learning, where task sequencing determines the availability of transferable experience. Although both frameworks technically allow sequential task execution, doing so significantly increases overall evaluation time.

Third, these frameworks are primarily designed to assess the capabilities of different LLM agents, but do not provide a standardized interface for integrating or comparing various capability enhancement methods. This hampers fair comparisons. For instance, in WebArena, repeatedly resets webpage states within a single task using the built-in reset function to facilitate tree search. This practice is unrealistic, as real-world environments generally lack such reset mechanisms, and agent actions are often irreversible. Consequently, the comparison between the method[9] and those proposed in other studies is inherently unfair. This stems from the framework’s failure to provide a unified interface for the development of capability enhancement methods. Similarly, AgentBench separates the environment, controller, and agent into three independent processes that communicate via HTTP. Even if researchers have sufficient computational resources to deploy all components on a single machine, multi-process debugging introduces substantial development overhead. This complexity may explain why, compared to WebArena, fewer studies leverage AgentBench for investigating agent enhancement techniques.

To address these issues, we propose a new evaluation framework tailored to lifelong learning scenarios, aiming to support and accelerate the development of LLM-based agents and methods for continual capability improvement.

![](images/a38499f148286a7f09b3e08874a1dba2abcece39e21fcb10fa20b3f619860e31.jpg)  
Figure 7: Evaluation Framework Architecture

# B.1 Framework Architecture

The evaluation framework consists of six key components: the language model pool, the agent, the environment, the chat history item factory, the controller, and the callbacks. The callback module is further composed of a callback handler and a list of callback functions.

# B.1.1 Language Model Pool

In the framework, the language model pool is implemented as a one-to-one mapping from string identifiers to language model objects. These model objects may belong to different language model classes, allowing for flexible integration of diverse models.

# B.1.2 Agent

The agent component is responsible for converting the historical actions taken by the agent, as well as past observations received from the environment, into a textual input sequence. This sequence is then fed into the language model to generate the agent’s next action through inference.

# B.1.3 Environment

The environment component is responsible for executing the agent’s actions and returning the corresponding responses to the controller. It provides five core methods: reset, interact, complete, calculate_metric, and release. Specifically:

• reset: Initializes the environment to the initial state $s _ { 0 }$ at the beginning of each task and returns the task goal $g$ . In environments that utilize Docker containers (e.g., the operating system environment), this method also creates a new container for agent interaction.   
• interact: Executes the agent’s action and returns the resulting observation from the environment to the controller.   
• complete: Evaluates whether the agent has successfully completed the task based on the goal-task reward function $R$ . In environments where a new Docker container is created for each task (e.g., the operating system environment), this method also handles the destruction of the container. In environments that reuse a single container across tasks (e.g., the database environment), it restores the container to its initial state.   
• calculate_metric: Assesses the agent’s overall performance within a lifelong learning scenario, based on the cumulative rewards obtained across all tasks.   
• release: Used in environments that do not require per-task container creation. It performs a unified cleanup by destroying the container to free system resources.

All test environments introduced in Section 4.2 share a common abstract parent class. Each environment must implement all abstract methods defined by this parent class and may only expose the public methods specified therein to other components. This design not only facilitates the extension of the evaluation framework to support new environments, but also encourages the development of methods for enhancing the capabilities of large language model-based agents that are broadly applicable across various lifelong learning environments.

# B.1.4 Interaction History Item Factory

The interaction history item factory is responsible for initializing the interaction history based on the environment description and task goal $g$ . During the environment’s reset process, the construct method of the factory is called to generate the initial interaction history by combining the task goal $g$ with the initial observation $O _ { 0 }$

Recent research has shown that incorporating historical experience into prompts or dynamically selecting prompt strategies based on environmental observations can significantly enhance the performance of LLM-based agents. To promote modular system design and decouple the logic of interaction history construction from environment-specific logic, we abstract this functionality into an independent interaction history item factory component. This design improves both the clarity and extensibility of the framework’s codebase.

# B.1.5 Controller

The controller component is primarily responsible for creating sessions, executing tasks, and storing the execution history of previously completed tasks.

A session is a structured object that encapsulates a task. In addition to recording key information such as the task index, interaction history, and the reward obtained by the agent, each session also maintains the session state that indicates its current status and the reason for task termination. As the task progresses, the interaction history grows incrementally. Both the agent and environment components can update the interaction history to exchange newly generated actions and observations, and they can also set the session state to notify the controller of any exceptional conditions encountered during task execution. Upon completion, each session is saved to a history list for future reference.

LLM-based experiments often span several hours or even days. To address this, our framework incorporates a snapshot mechanism that periodically persists critical data to disk at predefined intervals. This allows for efficient recovery and resumption in the event of unexpected interruptions or failures. Upon experiment startup, the controller attempts to restore any previously interrupted experiment. If the experiment is recoverable, the controller reads the necessary data from disk and skips already completed tasks, continuing with the remaining ones. After each task is completed and its corresponding session is added to the history list, the controller immediately saves the session data to disk. With this mechanism, an unexpected interruption results in the loss of, at most, a single task’s interaction data.

The controller is also responsible for calling callbacks at pre-defined events during task execution. We detail the callback mechanism in section B.1.6.

# B.1.6 Callback

Inspired by the Trainer module in the transformers library, we designed a callback component for the evaluation framework. This component can manage multiple callbacks, each of which can modularly implement a specific function.

We manage all callbacks through a centralized list. Key points in task execution are defined as events, and when an event occurs, all callbacks in the list are invoked sequentially. Each callback inherits from a common parent class, which provides event handler functions named after each event. By default, these handler functions do not affect task execution. However, researchers can inherit from the base callback class and override the event functions of interest. The customized callbacks can then be added to the callback list to influence task behavior and implement methods for enhancing agent capabilities.

This design ensures that all lifelong learning methods are developed and evaluated under consistent assumptions, thereby promoting comparability and fairness across experiments. Furthermore, since

callbacks in the list are isolated and unaware of each other, researchers can combine multiple callbacks from different methods within a single experiment to explore the synergistic effects of various approaches on agent performance.

Each event handler function receives five arguments: the agent, the environment, the current session, the session history list, and the session controller. The agent and environment refer to the components described in Appendix B.1.2 and Appendix B.1.3. The current session encapsulates a task, as detailed in Appendix B.1.5. The history session list is a key parameter for lifelong learning. Event handler functions are expected to extract useful experience from this list and enhance the large language model-based agent’s performance in lifelong learning scenarios by modifying the interaction history of the current task.

The session controller, a data member accessible to callbacks, consists of a set of boolean flags that can be used to skip specific steps in the interaction between the agent and the environment. For example, if an event handler detects that the agent’s response does not contain a valid action that the environment can interpret, it can use the session controller to skip the step of sending the action to the environment and instead prompt the agent to regenerate its response.

Notably, our framework allows each callback to optionally hold a reference to the language model during initialization and call it when needed. In contrast, most existing LLM-agent evaluation frameworks (e.g., AgentBench) do not decouple the LLM from the agent; instead, they treat the LLM-based agent as a single monolithic component. This design choice hinders the development of methods aimed at enhancing agent capabilities. For example, if a single LLM $L _ { 1 }$ is used by an agent and two different methods, each requiring LLM $L _ { 2 }$ access, are applied simultaneously via separate callbacks, and no model pool exists, both methods would redundantly load identical model arguments into GPU memory, as the callbacks are unaware of each other.

Additionally, if a researcher wants to perform multiple forward passes within a callback to improve response quality, calling the agent’s inference API directly may be inappropriate, since it typically only supports inference for a single dialogue history, not for batched parallel queries.

With the introduction of a shared language model pool, identical language model parameters used by both callbacks and the agent are loaded into GPU memory only once. This significantly reduces hardware requirements for developing agent capability enhancement methods and supports better modularity in the framework design.

We define a total of seven events for the callback mechanism. These events are as follows. During a single run of the evaluation framework, the restore_state event occurs at most once. Within the execution of a single task, on_session_create, on_environment_reset, on_environment_interact, and on_state_save each occur once, In contrast, on_agent_inference and on_environment_interact may be triggered multiple times throughout the task.

• restore_state: During the execution of an experiment, not only the controller’s state but also the internal states of the callbacks (i.e., the values of their data members) may change. Callbacks should save their internal state to disk when the on_state_save event occurs and reload it when the restore_state event is triggered.   
• on_session_create: This event is triggered after a session is initialized. At this point, the session only contains the index of the assigned task.   
• on_environment_reset: This event is triggered after the reset method of the environment is called.   
• on_agent_inference: This event is triggered after the agent’s inference method is called.   
• on_environment_interact: This event is triggered after the environment’s interact method is called.   
• on_environment_complete: This event is triggered after the environment’s complete method is called.   
• on_state_save: This event occurs when the controller writes its state to disk.

# B.2 Differences Between Distributed Deployment and Single-Machine Deployment

Nearly all LLM-based agent evaluation frameworks, including the one we propose, encapsulate the interactive environments for agents using Docker containers. However, due to the large number of parameters in LLM agents, they typically need to be deployed on high-performance computing (HPC) nodes. Most HPC nodes do not allow users to run Docker applications. As a result, most existing evaluation frameworks for LLM-based agents allow researchers to deploy the environment and the LLM on separate servers. WebArena [22] and VisualWebArena [10] , which only consider environments in the form of webpages, deploy Docker containers containing the webpages on external servers, and then use headless browsers running on the LLM-hosting server to access those webpages. However, this approach clearly cannot be generalized to other types of environments. AgentBench [12] and VisualAgentBench [13] support distributed deployment via custom-built Remote Procedure Call (RPC) toolkit. Although their RPC toolkit can handle diverse environments such as web pages and databases, the introduction of RPC brings additional cognitive load to developers. Even when researchers use servers capable of running Docker and opt for small-parameter LLMs to develop agent enhancement methods, the use of RPC forces them to run different components of the evaluation framework in separate processes, significantly complicating debugging.

To address these issues, our evaluation framework is designed to support both single-machine and distributed deployment modes. During development, researchers can deploy all components on the same server and within a single process, enabling rapid prototyping and easier debugging. During evaluation, however, the framework also allows different components to be deployed across multiple servers and processes, making it possible to evaluate agents powered by large-scale language models in lifelong learning scenarios.

Crucially, the use of RPC should be transparent to researchers during development. The transition from local to distributed deployment should require only the execution of a small amount of additional, highly reusable code, allowing the framework to switch deployment modes seamlessly without introducing unnecessary complexity.

To achieve these goals, we designed and implemented a plug-and-play RPC toolkit based on a client-server architecture. During development, researchers can choose not to use this toolkit and debug all components of the framework within the same process. They can also use the toolkit to split components that need to be deployed on other servers into a client and a server, and convert component calls into remote procedure calls to enable distributed deployment of the evaluation framework. Throughout the development process, researchers can operate under the assumption that all components run within the same process, without needing to understand the internal details of the toolkit or RPC mechanisms. The plug-and-play nature of the toolkit allows effortless switching between single-machine and distributed deployments. The plug-and-play RPC toolkit helps reduce the cognitive load on researchers during development, shortens debugging time, and promotes the proposal of new agent enhancement methods in lifelong learning scenarios.

We first introduce the RPC toolkit we implemented, and then introduce the structure of the framework under distributed deployment.

# B.2.1 Remote Procedure Call Toolkit

The implementation of RPC toolkit is shown in the Figure 8. In the figure, "client," "server," and "target" all refer to instantiated objects, but for brevity, we omit the word "object." We refer to the class that needs to be remotely called as the "target class." The target class can provide methods for other components of the evaluation framework to call. For each object of the target class (i.e., the target object), we wrap it with an object of the target server class (i.e., the target server object). Callers within the evaluation framework can access the target through the corresponding target client class object (i.e., the target client object). All target server classes and client classes are subclasses of the server and client base classes provided by our RPC toolkit. To implement RPC functionality, each target class must define its corresponding target server subclass and client subclass.

The target server continuously listens on a specific port on the server. When a caller calls a method of the target via the target client, the client first serializes the arguments and sends them via HTTP to the port that the target server is listening on. Upon receiving the HTTP request, the server deserializes the arguments into a format that the target can directly use and then performs the method call. After the call is completed, the result is returned to the target server, which then serializes the result and

The process of calling the target before using the remote call toolkit

![](images/2847dd9bbab33f23d9990e66a072dadb6fdd3e42dfea2b86b7241dbd3cf92b1f.jpg)

The process of calling the target after using the remote call toolkit

![](images/cad2e779e51af3b5a22e7395983c27035f3e6ef089a897bf84f8ac35c8fa0f95.jpg)  
Figure 8: The Implementation of our RPC toolkit

returns it via HTTP to the target client. The client then deserializes the result to obtain the correctly formatted output. The serialization and deserialization processes are mainly handled by the server and client base classes. The toolkit also supports reading and modifying data members in the same manner.

When implementing the server and client subclasses for a target, researchers only need to ensure that the target client provides the same interface as the target and defines the parameter types and return types of each interface using the BaseModel class from the Pydantic library. This makes it possible to implement remote invocation of the target without needing to understand the specifics of the serialization and deserialization processes. Through this design, we decouple the process of data transfer between processes during remote invocation from the target’s data processing logic.

The RPC toolkit supports serialization and deserialization of various object types, including immutable built-in Python types, enumeration types, and any subclass of the client base class. We implement the client base class as a subclass of the BaseModel. Consequently, all concrete client classes are also subclasses of BaseModel. The BaseModel class requires developers to explicitly specify the types of data members via type annotations and automatically performs type validation during object construction. This ensures that objects maintain data integrity and type consistency after serialization and deserialization.

By supporting serialization and deserialization of client subclass objects, our toolkit allow client subclass objects to be used as return values in remote procedure calls and passed back to the caller. This enables researchers to conveniently implement chained remote procedure calls, where the result obtained from one remote call can be directly used to initiate another remote procedure call, without the need to explicitly instantiate the client subclass object during development. The instantiation process is handled automatically by the toolkit, further simplifying the implementation.

The process of chained remote calls is illustrated in the Figure 9. In the following example, there are two target objects, $P _ { a }$ and $P _ { b }$ . $P _ { a }$ contains $P _ { b }$ as a data member. $P _ { a }$ and $P _ { b }$ can be two objects of the same target class or of different target classes. We assume the caller caller wants to call the method of $P _ { b }$ via $P _ { a }$ , and caller, $P _ { a }$ , and $P _ { b }$ are located in three different processes. Through the toolkit, what caller actually accesses is the client object $C _ { a }$ corresponding to $P _ { a }$ , and what $P _ { a }$ actually stores is the client object $C _ { b }$ corresponding to $P _ { b }$ .

Note that during development, researchers can still think of caller as accessing $P _ { a }$ rather than $C _ { a }$ and $P _ { a }$ as containing $P _ { b }$ as a data member rather than $C _ { b }$ . When using the dot operator and calling $P _ { b }$ ’s method, a remote procedure call is triggered. The dot operator takes a name string as input and returns the corresponding data member. In Figure 9, the server objects corresponding to $P _ { a }$ and $P _ { b }$ are denoted as $S _ { a }$ and $S _ { b }$ respectively, and $C _ { b } ^ { \prime }$ is the internally instantiated client of $P _ { b }$ created by the toolkit, which is transparent to the researcher and used to implement the chained remote call.

![](images/62b7c3a96b6fc05e8c45dfb27eb8b59cbf671033d2729afa56289fee1cfffd88.jpg)  
Figure 9: Example of Using our RPC Toolkit

![](images/e7e9538a7577c0e5be448cfec3bd35d1603624bda41d88abf304721ae6c28f4b.jpg)  
Figure 10: The Architecture of the Evaluation Framework in a Distributed Deployment

In the above remote procedure call process, all steps except for Step 3, Step 7, and Step 10 are implemented by the toolkit and are transparent to the researchers.

# B.2.2 Distributed Deployment Framework Structure

The architecture of the evaluation framework under distributed deployment is illustrated in the Figure 10. For clarity, some internal details of the components and the RPC toolkit are omitted. During experiments, the client-side controller process typically runs on a high-performance computing node, while the controller server, interaction history item factory server, and environment server processes usually run on standard servers with Docker support. Communication between these distributed processes is handled by the RPC toolkit described in Appendix B.2.1.

Before the experiment begins, researchers first need to launch the server-side controller on a standard server, followed by launching the client-side controller on the high-performance computing node. The server-side controller continuously listens on a designated port. When the client-side controller starts, it sends an HTTP request to the server-side controller to start the environment server and the interaction history item factory server. Once the client-side controller receives confirmation of

![](images/34d98147c5c60fb9fc4fc07d18ba7f80c410e3cf24d1d93cff94f5eb98a5da3e.jpg)  
Figure 11: The execution process of an assessment task

successful server initialization, it begins task scheduling. After all tasks are scheduled, the client-side controller sends another HTTP request to the server-side controller to shut down the interaction history item factory server and the environment server.

The end of an experiment corresponds to the termination of the client-side controller, interaction history item factory server, and environment server processes. In contrast, the server-side controller process continues to run. This design allows the server-side controller to continuously listen on a designated port and, upon the next launch of the client-side controller, reinitialize the environment server and interaction history item factory server based on the configuration file provided. In the distributed deployment, the client-side controller can be viewed as a wrapper around the single-

machine controller, with additional functionality for starting and shutting down the necessary servers at the beginning and end of each experiment.

During the implementation of the evaluation framework, we found that if the interaction history item factory component is treated as a regular data member of the environment component and not wrapped in a server subclass from our RPC toolkit, modifications made to the interaction history item factory on the client side will not be synchronized with the environment server. Specifically, without running the interaction history item factory component in a separate process, the client-side controller’s process can easily obtain a copy of the interaction history item factory that is identical to the one in the environment component. However, this copy is merely a local mutable variable on the frame of the client-side controller process, and any modifications made to its data members will not be synchronized with the interaction history item factory in the environment server. To resolve this issue, we run the interaction history item factory component in a separate process. Callers in the process where the environment server resides can access and modify the interaction history item factory through remote procedure calls. Meanwhile, callers in the process where the client-side controller resides can access and modify the interaction history item factory via chained remote procedure calls.

# B.3 The Execution Process of a Task

To clearly demonstrate the execution process of the evaluation framework and show the specific timing of events during task execution, we present the execution process of a task in the evaluation process in Figure 11. We assume that the evaluation framework is deployed on a single machine, so the components interact directly with each other without the need for the server or client in the RPC toolkit. In distributed deployment, for the researchers, the interaction methods and arguments passed between components remain the same as in a single-machine deployment.

In the evaluation framework, the parameters of the callback component are passed by reference, so the callback does not explicitly return results to the controller. In the diagram, should_environment_reset, should_agent_inference, should_environment_interact, and should_environment_complete are the interaction flow control conditions determined by the callback component. All four interaction flow control conditions default to true. Callbacks can modify these control flags to influence the interaction process between the agent and the environment.

# C Datasets Samples

# C.1 Database Samples

# Environment: DB || Case 1

Instruction: Insert a new payment record for member ID 102 with a payment date of "2023- 10-15", amount 75, and payment method "Credit Card" into the membership payments table. \nThe name of this table is membership_payments, and the headers of this table are payment_id, member_id, payment_date, amount, payment_method.

Ground Truth Action: INSERT INTO membership_payments (member_id, pay ment_date, amount, payment_method) VALUES (102, "2023-10-15", 75, "Cr edit Card")

Skill List: ["insert"]

# Environment: DB || Case 2

Instruction: Delete all entries where maintenance_status is equal to 0.\nThe name of this table is equipment_service, and the headers of this table are equipment_id, equipment_type, installation_date, maintenance_status, service_interval.

Ground Truth Action: DELETE FROM equipment_service WHERE maintenance_ status $\mathit { \Theta } = \mathit { \Theta } 0$ ;

Skill List: ["delete", "where_single_condition"]

# Environment: DB || Case 3

Instruction: Update the result status to "completed" and hours spent to 48 for the experiment with ID equal to 105 in the experiment_data table.\nThe name of this table is experiment_data, and the headers of this table are experiment_id, researcher_name, experiment_date, result_status, hours_spent, samples_used.

Ground Truth Action: UPDATE experiment_data SET result_status $=$ "com pleted", hours_spent = 48 WHERE experiment_id = 105;

Skill List: ["update", "where_single_condition"]

# Environment: DB || Case 4

Instruction: What are the models, years, and prices of vehicles that are currently available? Return the model, year, and price, ordered by price from highest to lowest.\nThe name of this table is vehicle_inventory, and the headers of this table are vehicle_id, model, year, price, status.

Ground Truth Action: SELECT model, year, price FROM vehicle_inventory WHERE status $=$ "available" ORDER BY price DESC;

Skill List: ["order_by_single_column", "select", "where_single_condition"]

# Environment: DB || Case 5

Instruction: What are the property listings" id (listing ID), type (property type), price, and square feet? Return the results ordered by price from highest to lowest, then square feet from highest to lowest, and limit the output to 10 entries.\nThe name of this table is property_listings, and the headers of this table are listing_id, property_type, price, square_feet, listed_date, status, agent_id.

Ground Truth Action: SELECT listing_id AS id, property_type AS typ e, price, square_feet FROM property_listings ORDER BY price DESC, square_feet DESC LIMIT 10;

Skill List: ["column_alias", "limit_only", "order_by_multiple_columns_same_direction", "select"]

# Environment: DB || Case 6

Instruction: Which routes have a distance greater than the average distance and less than the maximum distance? Return the route ID and driver name.\nThe name of this table is delivery_routes, and the headers of this table are route_id, driver_name, vehicle_type, distance_km, delivery_status.

Ground Truth Action: SELECT route_id, driver_name FROM delivery_rou tes WHERE distance_km $>$ (SELECT AVG(distance_km) FROM delivery_routes) AND distance_km $<$ (SELECT MAX(distance_km) FROM delivery_routes);

Skill List: ["select", "subquery_multiple", "where_multiple_conditions"]

# Environment: DB || Case 7

Instruction: What are the owner IDs and the total number of vaccinations administered to dogs since January 1, 2023? Return the owner ID and corresponding total vaccinations,

only including owners with more than 3 vaccinations.\nThe name of this table is vaccination_records, and the headers of this table are record_id, owner_id, pet_type, vaccination_type, vaccination_date, veterinarian, dose_number.

Ground Truth Action: SELECT owner_id, COUNT() AS total_vaccinations FROM vaccination_records WHERE pet_type $=$ "Dog" AND vaccination_date >= "2023-01-01" GROUP BY owner_id HAVING COUNT() > 3;

Skill List: ["column_alias", "group_by_single_column", "where_multiple_conditions", "having_single_condition_with_aggregate", "select"]

# Environment: DB || Case 8

Instruction: What are the item IDs and names for items that are either: 1) in the "Electronics" category with stock quantity below the average stock quantity of all items, or 2) in the same category as the item with ID 100 (determined dynamically via its category)? Return the item ID and name.\nThe name of this table is inventory_management, and the headers of this table are item_id, item_name, category, stock_quantity, last_restock_date.

Ground Truth Action: SELECT i.item_id AS id, i.item_name AS name FROM inventory_management AS i WHERE (i.category $=$ "Electronics" AND i.stoc k_quantity $<$ (SELECT AVG(stock_quantity) FROM inventory_management)) OR i.category $=$ (SELECT category FROM inventory_management WHERE item_ id $\mathit { \Theta } = \ 1 0 0 ^ { \cdot }$ );

Skill List: ["column_alias", "subquery_multiple", "table_alias", "where_nested_conditions", "select"]

# Environment: DB || Case 9

Instruction: Which seat numbers have a price difference (maximum price minus minimum price) greater than 50? Return the seat number and price difference, ordered by seat number in descending order, and limit the results to 5 entries.\nThe name of this table is ticket_sales, and the headers of this table are ticket_id, seat_number, price, sale_date, customer_name, status.

Ground Truth Action: SELECT seat_number, MAX(price)-MIN(price) AS pri ce_diff FROM ticket_sales GROUP BY seat_number HAVING MAX(price)-MIN(p rice) $\mathtt { > 5 0 }$ ORDER BY seat_number DESC LIMIT 5;

Skill List: ["column_alias", "group_by_single_column", "having_aggregate_calculation", "having_single_condition_with_aggregate", "limit_only", "order_by_single_column", "select"]

# Environment: DB || Case 10

Instruction: What are the guest IDs, their average ratings, and total reviews for guests who have either given a 5-star rating or submitted a review before January 1, 2023, and have more than 5 total reviews with an average rating below 4.5? Return the results ordered by average rating in descending order and total reviews in ascending order.\nThe name of this table is hotel_reviews, and the headers of this table are review_id, guest_id, review_text, rating, review_date.

Ground Truth Action: SELECT guest_id AS gid, AVG(rating) AS avg_ratin g, COUNT(review_id) AS total_reviews FROM hotel_reviews AS hr WHERE guest_id IN (SELECT guest_id FROM hotel_reviews WHERE rating $\mathit { \Theta } = \mathit { \Theta } 5 \mathit { \Theta }$ ) OR guest_id IN (SELECT guest_id FROM hotel_reviews WHERE review_date < "2

023-01-01") GROUP BY guest_id HAVING COUNT(review_id) > 5 AND AVG(rati $\mathtt { n g } ) ~ < ~ 4 . 5$ ORDER BY avg_rating DESC, total_reviews ASC;

Skill List: ["where_multiple_conditions" "group_by_single_column", "select", "table_alias", "subquery_multiple", "column_alias", "having_multiple_conditions_with_aggregate", "order_by_multiple_columns_different_directions"]

# C.2 Operating System Samples

# Environment: OS || Case 11

Instruction: Create three files "/tmp/file1", "/tmp/file2", and "/tmp/file3", setting each file"s permissions to 600 with a 1-second delay after creating each file. Use the "sleep" command between file operations.

Ground Truth Action: "rm -f /tmp/file1 /tmp/file2 /tmp/file3"

Skill List: ["chmod", "sleep", "touch"]

# Environment: OS || Case 12

Instruction: Recursively change ownership of "/var/webapp" to user "webadmin", set directory permissions to 750, and file permissions to 640. Ensure hidden files in subdirectories are included.

Ground Truth Action: "mkdir -p /var/webapp/{public,private/.cache} && touch /var/webapp/public/index.html /var/webapp/private/.cache/temp.dat && echo "config" $>$ /var/webapp/private/.env"

Skill List: ["chown", "find", "useradd"]

# Environment: OS || Case 13

Instruction: Modify the Nginx configuration file at /etc/nginx/sites-available/default with the following changes using sed: 1) Replace all "http://" with "https://", 2) Change the listen port from 80 to 8080, 3) Set server_name to "myapp.com", 4) Disable server_tokens by setting it to "off", 5) Add "client_max_body_size 20M;" after the server block opening, 6) Insert "keepalive_timeout 65;" before the location block, 7) Update the proxy_pass directive to use HTTPS, 8) Add a comment "# Security update" before server_name, and 9) Ensure all changes are made in-place. Preserve the original configuration structure.

Ground Truth Action: "mkdir -p /etc/nginx/sites-available && printf "server {\\\\n listen 80;\\\\n server_name example.com;\\\\n server_ tokens on;\\\\n location / {\\\\n proxy_pass http://backend;\\\\n }\ \\\n}\\\\n" > /etc/nginx/sites-available/default"

Skill List: ["cd", "cp", "echo", "sed"]

# Environment: OS || Case 14

Instruction: 1. Create a group "testgroup", add "testuser" to it, set group ownership recursively on "/var/www/project" to "testgroup", set directory permissions to 775, file permissions to 660, and enable setgid on the project directory.

Ground Truth Action: "useradd -m testuser && mkdir -p /var/www/proje ct/assets,logs && touch /var/www/project/assets/image1.jpg /var/www/p roject/logs/debug.log && chmod 755 /var/www/project"

Skill List: ["addgroup", "chgrp", "chmod", "find", "usermod"]

# Environment: OS || Case 15

Instruction: Remove /tmp/cleanup/file1.tmp and /tmp/cleanup/dir1/file2.tmp, delete all empty directories within /tmp/cleanup, and ensure the /tmp/cleanup directory exists.

Ground Truth Action: "mkdir -p /tmp/cleanup/dir1 /tmp/cleanup/dir2 && touch /tmp/cleanup/file1.tmp /tmp/cleanup/dir1/file2.tmp /tmp/cleanup /dir2/file3.log"

Skill List: ["echo", "exit", "find", "ls", "mkdir", "rm"]

# Environment: OS || Case 16

Instruction: Create a group "devteam", add users "user1", "user2", and "user3" to it, create a directory "/var/devteam_projects" accessible only to the group, and generate a log file with group details.

Ground Truth Action: "useradd -m user1 && useradd -m user2 && useradd -m user3"

Skill List: ["addgroup", "chgrp", "chmod", "echo", "mkdir", "tee", "touch", "usermod"]

# Environment: OS || Case 17

Instruction: Create a backup directory "/backup", copy "/source/test.txt" to both "/backup" and "/destination", append "Backup completed" to "/logs/backup.log", set permissions of "/backup/test.txt" to 644, and ensure "/destination/test.txt" is owned by root.

Ground Truth Action: "mkdir -p /source /destination /logs && echo "Sa mple data" > /source/test.txt && touch /logs/backup.log"

Skill List: ["chmod", "chown", "cp", "echo", "find", "grep", "ls", "mkdir", "tee"]

# Environment: OS || Case 18

Instruction: 1. Add "/bin/zsh" to /etc/shells. 2. Change "testuser" login shell to /bin/zsh. 3. Create group "devteam". 4. Add "testuser" to "devteam". 5. Create /shared directory with permissions 770 owned by "devteam" group.

Ground Truth Action: "useradd -m -s /bin/bash testuser && sed -i "/\\ \\/bin\\\\/zsh/d" /etc/shells"

Skill List: ["addgroup", "chage", "chgrp", "chmod", "chown", "chsh", "echo", "grep", "mkdir", "tee", "touch", "usermod"]

# C.3 Knowledge Graph Samples

# Environment: KG || Case 19

Instruction: which institution has national wine centre of australia?, Entities: ["National Wine Centre of Australia"

Ground Truth Action: ["get_relations(m.03hd1z)", "get_neighbors(m.03 hd1z,education.educational_institution.parent_institution)"]

Skill List: ["get_neighbors"]

# Environment: KG || Case 20

Instruction: how many characters appear on the cover of tintin in the land of the soviets?

Ground Truth Action: ["get_relations(m.02ll5h)", "get_neighbors(m.02 ll5h,comic_books.comic_book_issue.characters_on_cover)", "count(#0)"]

Skill List: ["count", "get_neighbors"]

# Environment: KG || Case 21

Instruction: Question: which short story of the sacred band of stepsons universe universe is know to have the earliest copyright date?, Entities: ["The Sacred Band of Stepsons universe"]

Ground Truth Action: ["get_relations(m.0ch8hcq)", "get_neighbors(m.0c h8hcq,fictional_universe.fictional_universe.works_set_here)", "get_att ributes(#0)", "argmin(#0,book.written_work.copyright_date)"]

Skill List: ["argmin", "get_neighbors"]

# Environment: KG || Case 22

Instruction: which fictional character produced by marv wolfman did trevor von eeden create?, Entities: ["Trevor Von Eeden", "Marv Wolfman"]

Ground Truth Action: ["get_relations(m.0279q8n)", "get_neighbors(m.02 79q8n,fictional_universe.fictional_character_creator.fictional_charact ers_created)", "get_relations(m.02gn9g)", "get_neighbors(m.02gn9g,fict ional_universe.fictional_character_creator.fictional_characters_create d)", "intersection(#0,#1)"]

Skill List: ["get_neighbors", "intersection"]

# Environment: KG || Case 23

Instruction: what was the most recently formed cyclone in the same category as hurricane dolly?, Entities: ["Hurricane Dolly"]

Ground Truth Action: ["get_relations(m.04dn799)", "get_neighbors(m.04 dn799,meteorology.tropical_cyclone.category)", "get_relations(#0)","ge t_neighbors(#0,meteorology.tropical_cyclone_category. tropical_cyclon es)", "get_attributes(#1)", "argmax(#1,meteorology.tropical_cyclone.fo rmed)"]

Skill List: ["argmax", "get_neighbors"]

# Environment: KG || Case 24

Instruction: how much content about talk radio is produced by the person that produces weekend edition sunday?, Entities: ["Talk radio", "Weekend Edition Sunday"]

Ground Truth Action: ["get_relations(m.07dn1)", "get_neighbors(m.07dn 1,broadcast.genre.content)", "get_relations(m.0t4t10s)", "get_neighbor s(m.0t4t10s,broadcast.content.producer)", "get_relations(#1)", "get_ne ighbors(#1,broadcast.producer.produces)", "intersection(#0,#2)", "count(#3)"]

Skill List: ["count", "get_neighbors", "intersection"]

# Environment: KG || Case 25

Instruction: what semi-firm textured cheese is made from the products of lamb and goat?, Entities: ["lamb", "Goat", "semi-firm"]

Ground Truth Action: ["get_relations(m.07bgp)", "get_neighbors(m.07bg p,food.cheese_milk_source.cheeses)", "get_relations(m.03fwl)", "get_ne ighbors(m.03fwl,food.cheese_milk_source.cheeses)", "get_relations(m.02 h82t0)", "get_neighbors(m.02h82t0,food.cheese_texture.cheeses)", "inte rsection(#1,#2)", "intersection(#0,#3)"]

Skill List: ["get_neighbors", "intersection"]

# Environment: KG || Case 26

Instruction: is a model of opel super 6 related to a eagle talon?, Entities: ["Opel Super 6", "Eagle Talon"]

Ground Truth Action: ["get_relations(m.0gdk70)", "get_neighbors(m.0gd k70,automotive.model.automotive_class)", "get_relations(#0)", "get_nei ghbors(#0,automotive.automotive_class.examples)", "get_relations(m.02p 04r)","get_neighbors(m.02p04r,automotive.model.related_models)", "get_ relations(#2)", "get_neighbors(#2,automotive.similar_automobile_models .related_model)", "intersection(#1,#3)"]

Skill List: ["get_neighbors", "intersection"]

# D Error Case Analysis

To better understand the failure modes of LLM agents in LifelongAgentBench , we analyze all task outcomes using two orthogonal attributes: evaluation_outcome (correct vs. incorrect) and sample_status (detailed agent or system status). This provides a fine-grained categorization of agent behavior and helps identify major bottlenecks.

Successful Cases (evaluation_outcome $= =$ "correct") We observe two typical success patterns:

• sample_status $= =$ "completed": the agent explicitly submits a valid final answer following the required format (e.g., correctly querying customer feedback with SQL and submitting results).   
• sample_status $= =$ "task_limit_reached": the agent stops without explicitly committing, but the final database state satisfies the evaluation criteria (e.g., correctly updating records within the step limit).

Error Cases (evaluation_outcome $= =$ "incorrect") We identify four common failure modes:

• sample_status $= =$ "completed": the agent explicitly commits an incorrect answer despite following the required format (e.g., submitting an incomplete result in a multi-step SQL query).   
• sample_status $= =$ "task_limit_reached": the agent fails to converge within the interaction limit, often due to redundant or looping operations (e.g., repeatedly adjusting query logic without submitting).   
• sample_status $= =$ "agent_validation_failed": the agent violates output constraints, such as missing required keywords or formatting (e.g., omitting Action: Answer tag).   
• sample_status $= =$ "agent_context_limit": the agent exceeds the LLM context window, especially in environments requiring long reasoning chains or large intermediate outputs.

System-Level Failures We also observe rare non-agent errors:

• task_environment_error: task-side infrastructure failures (e.g., Docker crashes).   
• task_unknown_error: unknown exceptions in task execution.   
• agent_out_of_memory: GPU memory exhaustion on large models.   
• agent_unknown_error: unexpected internal failures within the agent framework.

This taxonomy highlights that even when equipped with powerful LLMs, agents remain vulnerable to subtle evaluation failures, unstable reasoning loops, and external infrastructure issues. Our benchmark thus provides valuable diagnostic insights for developing more robust lifelong learning agents.

# D.1 Task Completed

# D.1.1 Error Case 1

Task Description: The agent was instructed to query the database to retrieve insurance policy details, specifically the policy IDs, holder names, and policy types from the insurance_policies table. The results needed to be ordered by effective_date from the most recent to the oldest, skipping the first five entries and returning the subsequent ten entries.

Agent Behavior: Initially, the agent executed the following SQL query:

```sql
SELECT policy_id, holder_name, policy_type  
FROM insurance_policies  
ORDER BY effective_date DESC  
LIMIT 4 OFFSET 5; 
```

This query was incorrect due to mistakenly using LIMIT 4 instead of the required LIMIT 10. Consequently, the query returned only four entries instead of the intended ten.

Recognizing the mistake after receiving the results, the agent correctly identified the error in the limit clause. It then adjusted and executed the corrected query:

```sql
SELECT policy_id, holder_name, policy_type  
FROM insurance_policies  
ORDER BY effective_date DESC  
LIMIT 10 OFFSET 5; 
```

The updated query returned the correct number of rows (ten entries), satisfying the task requirements.

Reason for Failure: Despite eventually correcting the SQL query, the task outcome was marked as incorrect (status completed but evaluation incorrect). The failure arose from the initial misunderstanding and execution of an inaccurate query, causing unnecessary computational resources (an extra interaction) and illustrating unstable reasoning. Such errors highlight difficulties in accurately

translating natural language constraints (e.g., “starting from the 6th entry and limited to 10 entries”) into correct SQL clauses (OFFSET and LIMIT).

# Implications and Insights:

• Unstable Reasoning: The agent correctly explained the logic but initially failed to accurately apply it, demonstrating inconsistency between stated understanding and execution.   
• Inefficiency: The error required additional agent-system interactions, revealing suboptimal reasoning efficiency.   
• Potential Solutions: Future improvements could involve explicit verification of each SQL query component against the original natural language instruction or introducing intermediate verification steps before query execution.

This case exemplifies a common issue where an agent’s conceptual interpretation of instructions is correct but its initial execution is flawed, underscoring the necessity for enhanced accuracy in natural language-to-code translation.

# Error Case 1 || Environment: DB || Error Type: completed

User: I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.\nYou have to explain the problem and your solution to me and write down your thoughts.\nAfter thinking and explaining thoroughly, every round you can choose one of the two actions: Operation or Answer.\n\nTo do operation, the format should be like this:\nAction: Operation\n"sql\nSELECT FROM table WHERE condition; $\mathrm { \backslash n " } \mathrm { \backslash n Y o u }$ MUST put SQL in markdown format without any other comments. Your SQL should be in one line.\nI will use Äction: Operation¨literally to match your SQL.\nEvery time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.\nIf the SQL is not executed successfully, the response will be the error message.\nOtherwise, the response will be the raw MySQL response.\nFor SELECT queries, the response will be the result of the query, such as [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...], where each tuple represents a row and the elements are the values of the columns in the row.\nFor SQL such as INSERT, UPDATE, and DELETE, the response will be an empty list [] indicating that the SQL was executed successfully. $\mathrm { \backslash n u n H }$ you have obtain the answer by interacting with the database, and you MUST commit your final answer using the format like this:\nAction: Answer\nFinal Answer: [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...]\nDO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.\nYour answer should be accurate. Your answer must be exactly the same as the correct answer.\nIf the question is about modifying the database, then after done operation, your answer field can be anything.\nIf the question is about querying the database, then after done operation, your answer field should be the result of the query.\nWe note that the column names will not be displayed in the result, and you need to ensure both the orders of the columns and rows are correct.\nIf your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.\nOnce you commit your answer or the number of rounds reaches 3, the task will be finished and the system will judge whether you pass the task or not.\n\nNow, I will give you the question that you need to solve.

Agent: OK.

User: What are the policy IDs, holder names, and policy types of the insurance policies? Return the results ordered by effective date from most recent to oldest, starting from the 6th entry and limited to 10 entries.\nThe name of this table is insurance_policies, and the headers of this table are policy_id, holder_name, policy_type, effective_date, premium_amount, coverage_limit.

Agent: To solve this problem, we need to first understand the requirements. We need to return the policy IDs, holder names, and policy types of the insurance policies. We also need to order the results by effective date from most recent to oldest, starting from the 6th entry and limited to 10 entries.\n\nHere’s my thought process: $\mathrm { \backslash n u n 1 }$ . We need to select the policy ID, holder name, and policy type from the insurance_policies table. $\mathrm { i n } 2$ . We need to order the results by effective

date in descending order (most recent to oldest).\n3. We need to start the result set from the 6th entry.\n4. We need to limit the result set to 10 entries.\n\nBased on this, I will choose the Öperationäction to execute the SQL query.\n\nAction: Operation\n"sql\nSELECT policy_id, holder_name, policy_type \nFROM insurance_policies \nORDER BY effective_date DESC \nLIMIT 4 OFFSET 5;\n"\nPlease execute this SQL query.

User: [(6, "Alice Brown", "Health"), (7, "Bob Green", "Auto"), (8, "Charlie White", "Life"), (9, "Diana Black", "Home")]

Agent: The query I executed returned the first 4 rows from the 6th row onwards, ordered by effective date in descending order. However, this is not what we want. We want to return the first 10 rows from the 6th row onwards, ordered by effective date in descending order.\n\nTo achieve this, we need to modify the query to return the first 10 rows from the 6th row onwards, ordered by effective date in descending order.\n\nHere’s my thought process: $\mathrm { \backslash n u n 1 }$ . We need to select the policy ID, holder name, and policy type from the insurance_policies table. $\mathrm { l n } 2$ . We need to order the results by effective date in descending order (most recent to oldest).\n3. We need to start the result set from the 6th entry. $\backslash \mathrm { { m 4 } }$ . We need to limit the result set to 10 entries.\n\nBased on this, I will choose the Öperationäction to execute the SQL query.\n\nAction: Operation\n"sql\nSELECT policy_id, holder_name, policy_type \nFROM insurance_policies \nORDER BY effective_date DESC \nLIMIT 10 OFFSET 5;\n"

User: [(6, "Alice Brown", "Health"), (7, "Bob Green", "Auto"), (8, "Charlie White", "Life"), (9, "Diana Black", "Home"), (10, "Eve Gray", "Travel"), (11, "Alice Brown", "Health"), (12, "Bob Green", "Auto"), (13, "Charlie White", "Life"), (14, "Diana Black", "Home"), (15, "Eve Gray", "Travel")]

Agent: The query I executed returned the first 10 rows from the 6th row onwards, ordered by effective date in descending order. This is what we want.\n\nNow, we can commit the final answer.\n\nAction: Answer\nFinal Answer: [(6, "Alice Brown", "Health"), (7, "Bob Green", "Auto"), (8, "Charlie White", "Life"), (9, "Diana Black", "Home"), (10, "Eve Gray", "Travel"), (11, "Alice Brown", "Health"), (12, "Bob Green", "Auto"), (13, "Charlie White", "Life"), (14, "Diana Black", "Home"), (15, "Eve Gray", "Travel")]

# D.1.2 Error Case 2

Task Description: The agent was tasked with querying the database to identify regions from the customer_usage table that have a total data usage exceeding 1000 and an average call duration under 300. The results needed to include the region, total data usage, and average call duration, grouped by region, and limited to 5 entries starting from the third entry.

Agent Behavior: Initially, the agent executed the following SQL query:

```sql
SELECT region, SUM(datausage) AS total_datausage,  
AVG(call_duration) AS average_call_duration  
FROM customerusage  
GROUP BY region  
HAVING SUM(datausage) > 1000 AND AVG(call_duration) < 300  
ORDER BY total_datausage DESC  
LIMIT 2, 5; 
```

However, the agent misunderstood the usage of LIMIT and OFFSET syntax in SQL, using LIMIT 2, 5 instead of the correct LIMIT 5 OFFSET 2 syntax. After executing the initial query, the agent attempted to correct its misunderstanding by issuing a second query:

```sql
SELECT region, SUM(datausage) AS total_datausage,  
AVG(call_duration) AS average_call_duration  
FROM customerusage  
GROUP BY region  
HAVING SUM(datausage) > 1000 AND AVG(call_duration) < 300 
```

ORDER BY total_data_usage DESC LIMIT 3, 5;

Yet, this second query also incorrectly interpreted the desired offset and limit, resulting in incorrect indexing.

Reason for Failure: The agent’s failure stemmed from confusion regarding the SQL syntax for specifying offsets and limits. Instead of correctly applying OFFSET 2 LIMIT 5 to start from the third entry, the agent incorrectly applied LIMIT 2,5 and subsequently LIMIT 3,5, leading to incorrect or shifted result sets.

# Implications and Insights:

• Syntax Misinterpretation: The agent demonstrated difficulty accurately applying SQL syntax for offset and limit clauses, highlighting issues in translating natural language queries to precise SQL statements.   
• Persistent Mistake: Despite attempting to correct its mistake, the agent continued to misuse the syntax, illustrating repeated misunderstanding rather than one-off oversight.   
• Potential Solutions: Improved SQL syntax awareness or explicit syntactic verification mechanisms during intermediate reasoning steps could significantly enhance accuracy.

This case highlights the importance of reinforcing SQL syntax knowledge and accuracy during the reasoning and execution phases for LLM agents.

# Error Case 2 || Environment: DB || Error Type: completed

User: I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.\nYou have to explain the problem and your solution to me and write down your thoughts.\nAfter thinking and explaining thoroughly, every round you can choose one of the two actions: Operation or Answer.\n\nTo do operation, the format should be like this:\nAction: Operation\n"sql\nSELECT FROM table WHERE condition; $\mathrm { \backslash n " } \mathrm { \backslash n Y o u }$ MUST put SQL in markdown format without any other comments. Your SQL should be in one line.\nI will use "Action: Operation" literally to match your SQL.\nEvery time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.\nIf the SQL is not executed successfully, the response will be the error message.\nOtherwise, the response will be the raw MySQL response.\nFor SELECT queries, the response will be the result of the query, such as [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...], where each tuple represents a row and the elements are the values of the columns in the row.\nFor SQL such as INSERT, UPDATE, and DELETE, the response will be an empty list [] indicating that the SQL was executed successfully. $\mathrm { \backslash n u n H }$ you have obtain the answer by interacting with the database, and you MUST commit your final answer using the format like this:\nAction: Answer\nFinal Answer: [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...]\nDO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.\nYour answer should be accurate. Your answer must be exactly the same as the correct answer.\nIf the question is about modifying the database, then after done operation, your answer field can be anything.\nIf the question is about querying the database, then after done operation, your answer field should be the result of the query.\nWe note that the column names will not be displayed in the result, and you need to ensure both the orders of the columns and rows are correct.\nIf your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.\nOnce you commit your answer or the number of rounds reaches 3, the task will be finished and the system will judge whether you pass the task or not.\n\nNow, I will give you the question that you need to solve.

# Agent: OK.

User: Which regions have a total data usage exceeding 1000 and an average call duration under 300? Return the region, total data usage, and average call duration, grouped by region. Limit the results to 5 entries starting from the third entry.\nThe name of this table is customer_usage, and the headers of this table are customer_id, region, data_usage, call_duration, month.

Agent: To solve this problem, we need to first calculate the total data usage and average call duration for each region. Then, we can filter the results to only include regions with a total data usage exceeding 1000 and an average call duration under 300. Finally, we can group the results by region and limit them to 5 entries starting from the third entry.\n\nAction: Operation\n"sql\nSELECT region, SUM(data_usage) AS total_data_usage, AVG(call_duration) AS average_call_duration \nFROM customer_usage \nGROUP BY region \nHAVING SUM(data_usage) $> 1 0 0 0$ AND AVG(call_duration) $< 3 0 0$ \nORDER BY total_data_usage DESC \nLIMIT 2, 5;\n"

User: [("South", Decimal("1200"), Decimal("250.0000")), ("Central", Decimal("1200"), Decimal("280.0000")), ("EastCoast", Decimal("1100"), Decimal("290.0000")), ("WestCoast", Decimal("1100"), Decimal("200.0000")), ("Southwest", Decimal("1050"), Decimal("290.0000"))]

Agent: The results show the top 5 regions with a total data usage exceeding 1000 and an average call duration under 300. However, we need to exclude the first two entries since we want to start from the third entry.\n\nAction: Operation\n"sql\nSELECT region, SUM(data_usage) AS total_data_usage, AVG(call_duration) AS average_call_duration \nFROM customer_usage \nGROUP BY region \nHAVING SUM(data_usage) $> 1 0 0 0$ AND AVG(call_duration) $< 3 0 0$ \nORDER BY total_data_usage DESC \nLIMIT 3, 5;\n"

User: [("Central", Decimal("1200"), Decimal("280.0000")), ("EastCoast", Decimal("1100"), Decimal("290.0000")), ("WestCoast", Decimal("1100"), Decimal("200.0000")), ("Southwest", Decimal("1050"), Decimal("290.0000"))]

Agent: The results now show the top 5 regions with a total data usage exceeding 1000 and an average call duration under 300, starting from the third entry.\n\nAction: Answer\nFinal Answer: [("Central", Decimal("1200"), Decimal("280.0000")), ("EastCoast", Decimal("1100"), Decimal("290.0000")), ("WestCoast", Decimal("1100"), Decimal("200.0000")), ("Southwest", Decimal("1050"), Decimal("290.0000"))]

# D.2 Task Limited Reached

# D.2.1 Error Case 3

Task Description: The agent was tasked with updating the hourly rates for staff members in Security roles from the staff_shifts table. Specifically, it was required to increase the hourly rate by $15 \%$ for Security roles where the hours worked exceeded 30, and the current hourly rate was below the average hourly rate of all Security roles.

Agent Behavior: Initially, the agent correctly calculated the average hourly rate for Security roles using the following SQL query:

```sql
SELECT AVG(hourly_rate) AS avg_rate  
FROM staff-shifts  
WHERE role = "Security"; 
```

Upon identifying the average hourly rate (20.5), the agent correctly queried the eligible entries to be updated:

```sql
SELECT * FROM staff-shifts WHERE role = "Security" AND hours_worked > 30 AND hourly_rate < 20.5; 
```

Having successfully retrieved the targeted entries, the agent executed the update query:

```sql
UPDATE staff-shifts  
SET hourly_rate = hourly_rate * 1.15  
WHERE role = "Security" AND hours_worked > 30  
AND hourly_rate < 20.5; 
```

Finally, the agent attempted to verify the successful execution of the update by recalculating the average hourly rate. However, at this point, the agent reached the maximum allowed interaction steps, failing to explicitly commit a final answer.

Reason for Failure: The agent failed due to exceeding the maximum interaction limit without committing a final answer explicitly. Although the operations performed were logically sound and correct, the agent did not conclude the task by confirming the success explicitly within the allowed steps.

# Implications and Insights:

• Lack of Task Commitment: The agent demonstrated proper initial reasoning and executed correct SQL statements. However, it failed to finalize the task promptly, reflecting inadequate task closure behavior.   
• Interaction Efficiency: This case highlights the necessity for the agent to strategically plan and optimize interactions, especially when task constraints (such as interaction limits) are strict.   
• Potential Solutions: Introducing intermediate commitment strategies or improved planning to manage interaction limits effectively could enhance overall task performance and success rates.

This case emphasizes the importance of explicit task completion strategies and efficient interaction management in multi-step tasks for LLM agents.

# Error Case 3 || Environment: DB || Error Type: task limit reached

User: I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.\nYou have to explain the problem and your solution to me and write down your thoughts.\nAfter thinking and explaining thoroughly, every round you can choose one of the two actions: Operation or Answer.\n\nTo do operation, the format should be like this:\nAction: Operation\n"sql\nSELECT FROM table WHERE condition; $\mathrm { \backslash n " } \mathrm { \backslash n Y o u }$ MUST put SQL in markdown format without any other comments. Your SQL should be in one line.\nI will use "Action: Operation" literally to match your SQL.\nEvery time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.\nIf the SQL is not executed successfully, the response will be the error message.\nOtherwise, the response will be the raw MySQL response.\nFor SELECT queries, the response will be the result of the query, such as [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...], where each tuple represents a row and the elements are the values of the columns in the row.\nFor SQL such as INSERT, UPDATE, and DELETE, the response will be an empty list [] indicating that the SQL was executed successfully. $\mathrm { \backslash n u n H }$ you have obtain the answer by interacting with the database, and you MUST commit your final answer using the format like this:\nAction: Answer\nFinal Answer: [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...]\nDO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.\nYour answer should be accurate. Your answer must be exactly the same as the correct answer.\nIf the question is about modifying the database, then after done operation, your answer field can be anything.\nIf the question is about querying the database, then after done operation, your answer field should be the result of the query.\nWe note that the column names will not be displayed in the result, and you need to ensure both the orders of the columns and rows are correct.\nIf your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.\nOnce you commit your answer or the number of rounds reaches 3, the task will be finished and the system will judge whether you pass the task or not.\n\nNow, I will give you the question that you need to solve.

# Agent: OK.

User: Update the hourly rate by increasing it by $15 \%$ for Security roles where hours worked exceed 30 and their current hourly rate is below the average hourly rate of all Security roles.\nThe name of this table is staff_shifts, and the headers of this table are shift_id, staff_name, role, shift_date, hours_worked, hourly_rate, total_pay.

Agent: To solve this problem, we need to first calculate the average hourly rate of all Security roles. Then, we need to find the Security roles where hours worked exceed 30 and their current hourly rate is below the average hourly rate of all Security roles. Finally, we can update the hourly rate of these roles by increasing it by 15%.\n\nAction: Operation\n"sql\nSELECT AVG(hourly_rate) AS avg_rate FROM staff_shifts WHERE role $=$ "Security";\n"

User: [(Decimal("20.5000"),)]

Agent: The average hourly rate of all Security roles is 20.5. Now, we need to find the Security roles where hours worked exceed 30 and their current hourly rate is below 20.5.\n\nAction: Operation\n"sql\nSELECT FROM staff_shifts WHERE role $=$ "Security" AND hours_worked $> 3 0$ AND hourly_rate $< 2 0 . 5 ; \mathrm { l n } "$

User: [(1, "Mike Johnson", "Security", "2023-01-01", 35, 18, 630), (2, "Anna Lee", "Security", "2023-01-02", 32, 20, 640), (5, "Evan Smith", "Security", "2023-01-05", 31, 19, 589), (7, "George Adams", "Security", "2023-01-07", 33, 17, 561), (10, "Jenna Hall", "Security", "2023-01-10", 34, 16, 544)]

Agent: We have found the Security roles that meet the conditions. Now, we can update the hourly rate of these roles by increasing it by $15 \%$ .\n\nAction: Operation\n"sql\nUPDATE staff_shifts SET hourly_rate $=$ hourly_rate 1.15 WHERE role $=$ "Security" AND hours_worked $> 3 0$ AND hourly_rate $< 2 0 . 5 ; \mathrm { i n }$ "

User: []

Agent: The update operation was successful. Now, we need to recalculate the average hourly rate of all Security roles to confirm that the update operation was correct.\n\nAction: Operation\n"sql\nSELECT AVG(hourly_rate) AS avg_rate FROM staff_shifts WHERE role $=$ "Security";\n"

# D.3 Error Case 4

Task Description: The agent was tasked with querying the tax_records table to identify tax statuses and their total amounts owed for the tax year 2023, specifically focusing on statuses labeled "unpaid" or "pending". The results needed to be grouped by status, ordered by status in descending order, and within each status, by total owed in ascending order.

Agent Behavior: Initially, the agent executed the following correct SQL query:

```sql
SELECT status, SUM(amount_owed) as total_owed  
FROM taxRecords  
WHERE tax_year = 2023 AND status IN ("unpaid", "pending")  
GROUP BY status  
ORDER BY status DESC, total_owed ASC; 
```

The query returned the desired result set correctly:

[("unpaid", Decimal("6800")), ("pending", Decimal("6100"))]

However, the agent incorrectly assumed that additional sorting logic was needed to handle cases with identical total owed amounts, even though no such cases were present. This misunderstanding led to multiple redundant attempts using SQL functions like FIELD() and conditional expressions (IF()) that did not alter the result. The agent unnecessarily repeated query attempts without reaching a conclusion.

Reason for Failure: The agent failed due to repeatedly performing redundant and unnecessary operations that did not alter the query outcome, exhausting the maximum allowed interaction steps without explicitly committing the final correct answer.

Implications and Insights:

• Redundant Reasoning: The agent’s repeated attempts to address an imagined problem illustrate ineffective recognition of successfully completed tasks, leading to redundant queries.   
• Misinterpretation of Results: Despite correct initial execution, the agent incorrectly inferred a need for additional complex sorting logic, indicating a misunderstanding of task completion criteria.   
• Potential Solutions: Encouraging explicit checks or verification steps to recognize already successful outcomes could help avoid redundant actions. Improved monitoring of agent understanding could also mitigate unnecessary complexity.

This case emphasizes the necessity of accurate result interpretation and effective termination strategies to prevent redundant interactions in multi-step tasks for LLM agents.

# Error Case 4 || Environment: DB || Error Type: task_limit_reached

User: I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.\nYou have to explain the problem and your solution to me and write down your thoughts.\nAfter thinking and explaining thoroughly, every round you can choose one of the two actions: Operation or Answer.\n\nTo do operation, the format should be like this:\nAction: Operation\n"sql\nSELECT FROM table WHERE condition; $\mathrm { \backslash n " } \mathrm { \backslash n Y o u }$ MUST put SQL in markdown format without any other comments. Your SQL should be in one line.\nI will use "Action: Operation" literally to match your SQL.\nEvery time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.\nIf the SQL is not executed successfully, the response will be the error message.\nOtherwise, the response will be the raw MySQL response.\nFor SELECT queries, the response will be the result of the query, such as [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...], where each tuple represents a row and the elements are the values of the columns in the row.\nFor SQL such as INSERT, UPDATE, and DELETE, the response will be an empty list [] indicating that the SQL was executed successfully.\n\nIf you have obtain the answer by interacting with the database, and you MUST commit your final answer using the format like this:\nAction: Answer\nFinal Answer: [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...]\nDO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.\nYour answer should be accurate. Your answer must be exactly the same as the correct answer.\nIf the question is about modifying the database, then after done operation, your answer field can be anything.\nIf the question is about querying the database, then after done operation, your answer field should be the result of the query.\nWe note that the column names will not be displayed in the result, and you need to ensure both the orders of the columns and rows are correct.\nIf your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.\nOnce you commit your answer or the number of rounds reaches 3, the task will be finished and the system will judge whether you pass the task or not.\n\nNow, I will give you the question that you need to solve.

# Agent: OK.

User: What are the statuses and their total amounts owed for the tax year 2023 where the status is either "unpaid" or "pending"? Return each status and the corresponding total amount owed, grouped by status, ordered by status in descending order and total owed in ascending order.\nThe name of this table is tax_records, and the headers of this table are taxpayer_id, name, tax_year, amount_owed, status, filing_date.

Agent: To solve this problem, we need to first identify the tax records for the tax year 2023 where the status is either "unpaid" or "pending". Then, we can group these records by status and calculate the total amount owed for each status. Finally, we can order the results by status in descending order and total owed in ascending order.\n\nHere’s my thought process: $\mathrm { \backslash n u n 1 }$ . We need to select the tax records for the tax year 2023 where the status is either "unpaid" or "pending". $\mathrm { l n } 2$ . We need to group these records by status and calculate the total amount owed for each status. $\mathrm { \backslash n } 3$ . We need to order the results by status in descending order and total owed in ascending order. $\scriptstyle \mathrm { \backslash n backslash n B }$ ased on these steps, I will choose the "Operation" action to execute the

SQL query.\n\nAction: Operation\n"sql\nSELECT status, SUM(amount_owed) as total_owed FROM tax_records WHERE tax_year = 2023 AND status IN ("unpaid", "pending") GROUP BY status ORDER BY status DESC, total_owed ASC;\n"

User: [("unpaid", Decimal("6800")), ("pending", Decimal("6100"))]

Agent: The query returned the total amount owed for each status. However, we also need to consider the case where the total amount owed is the same for two or more statuses. In this case, we want to order the statuses in descending order, and then order the statuses with the same total amount owed in ascending order of their status names.\n\nTo achieve this, we can modify the query to use the FIELD function, which returns the index of the first occurrence of a value in a list. We can use this function to order the statuses in descending order, and then order the statuses with the same total amount owed in ascending order of their status names.\n\nHere’s my thought process: $\mathrm { \backslash n u n 1 }$ . We need to select the tax records for the tax year 2023 where the status is either "unpaid" or "pending". $\mathrm { . } \mathrm { m } 2$ . We need to group these records by status and calculate the total amount owed for each status. $\cdot \mathrm { m } 3$ . We need to order the results by status in descending order, and then order the statuses with the same total amount owed in ascending order of their status names.\n\nBased on these steps, I will choose the "Operation" action to execute the SQL query.\n\nAction: Operation\n"sql\nSELECT status, SUM(amount_owed) as total_owed FROM tax_records WHERE tax_year $= 2 0 2 3$ AND status IN ("unpaid", "pending") GROUP BY status ORDER BY FIELD(status, "unpaid", "pending"), total_owed ASC;\n"

User: [("unpaid", Decimal("6800")), ("pending", Decimal("6100"))]

Agent: The query still didn’t order the statuses with the same total amount owed in ascending order of their status names. To achieve this, we can use a combination of the FIELD function and the IF function to create a custom ordering.\n\nHere’s my thought process: $\mathrm { \backslash n u n 1 }$ . We need to select the tax records for the tax year 2023 where the status is either "unpaid" or "pending".\n2. We need to group these records by status and calculate the total amount owed for each status. $\mathrm { \backslash n } 3$ . We need to order the results by status in descending order, and then order the statuses with the same total amount owed in ascending order of their status names.\n\nBased on these steps, I will choose the "Operation" action to execute the SQL query.\n\nAction: Operation\n"sql\nSELECT status, SUM(amount_owed) as total_owed FROM tax_records WHERE tax_year $= 2 0 2 3$ AND status IN ("unpaid", "pending") GROUP BY status ORDER BY FIELD(status, "unpaid", "pending"), IF(status $=$ "unpaid", 0, 1), total_owed ASC;\n"

User: [("unpaid", Decimal("6800")), ("pending", Decimal("6100"))]

Agent: The query still didn’t order the statuses with the same total amount owed in ascending order of their status names. To achieve this, we can use a combination of the FIELD function and the IF function to create a custom ordering.\n\nHowever, I realized that the previous queries were close, but the FIELD function was not necessary. We can simply use the status names in the ORDER BY clause to achieve the desired ordering.\n\nHere’s my thought process: $\mathrm { \backslash n u n 1 }$ . We need to select the tax records for the tax year 2023 where the status is either "unpaid" or "pending". $\mathrm { . } \mathrm { m } 2$ . We need to group these records by status and calculate the total amount owed for each status. $\mathrm { \backslash n } 3$ . We need to order the results by status in descending order, and then order the statuses with the same total amount owed in ascending order of their status names.\n\nBased on these steps, I will choose the "Operation" action to execute the SQL query.\n\nAction: Operation\n"sql\nSELECT status, SUM(amount_owed) as total_owed FROM tax_records WHERE tax_year $= 2 0 2 3$ AND status IN ("unpaid", "pending") GROUP BY status ORDER BY status DESC, total_owed ASC;\n"

# D.4 Agent Validation Failed

# D.4.1 Error Case 5

Task Description: The agent was tasked with deleting up to 100 records from the package_tracking table, specifically targeting packages marked as "lost" with a last update before January 1, 2023, or packages weighing less than 5 kilograms.

Agent Behavior: Initially, the agent executed an incorrect SQL query attempting to combine LIMIT with a subquery:

```sql
DELETE FROM package_tracking  
WHERE (status = "lost" AND last_update < "2023-01-01")  
OR package_id IN (SELECT package_id FROM package_tracking WHERE weight_kg < 5 LIMIT 100); 
```

This resulted in a MySQL syntax error, as LIMIT is not supported within subqueries in MySQL.

Recognizing this error, the agent attempted a valid alternative by using a temporary table to bypass the limitation:

```sql
CREATE TEMPORARY TABLE temp AS
SELECT package_id FROM package_tracking WHERE weight_kg < 5 LIMIT 100; 
```

DELETE FROM package_tracking WHERE (status = "lost" AND last_update $<$ "2023-01-01") OR package_id IN (SELECT package_id FROM temp);

Although logically correct, the agent mistakenly executed two SQL statements simultaneously, violating the instructions that explicitly require executing only one SQL statement per interaction.

Reason for Failure: The task failed due to an agent_validation_failed error, specifically because the agent violated the given operational constraints by submitting multiple SQL statements in one operation instead of executing them sequentially.

# Implications and Insights:

• Instruction Adherence: The agent clearly understood the logical solution but overlooked explicit instructions requiring single-statement execution per interaction.   
• Validation Awareness: This highlights the need for the agent to better adhere to clearly defined operational constraints, even when the logical reasoning is correct.   
• Potential Solutions: Implementing explicit intermediate validation checks or enhancing agent instruction comprehension could significantly reduce such format violations.

This case illustrates the importance of strict adherence to operational instructions and suggests reinforcing agent compliance through validation checks to prevent procedural errors.

Error Case 5 || Environment: DB || Error Type: agent_validation_failed   
```jsonl
User: I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.\nYou have to explain the problem and your solution to me and write down your thoughts.\nAfter thinking and explaining thoroughly, every round you can choose one of the two actions: Operation or Answer.\nTo do operation, the format should be like this:\nAction: Operation\n"sql\nSELECT FROM table WHERE condition;\n"You MUST put SQL in markdown format without any other comments. Your SQL should be in one line.\nI will use "Action: Operation" literally to match your SQL.\nEvery time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.\nIf the SQL is not executed successfully, the response will be the error message.\nOtherwise, the response will be the raw MySQL response.\nFor SELECT queries, the response will be the result of the query, such as [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...], where each tuple 
```

represents a row and the elements are the values of the columns in the row.\nFor SQL such as INSERT, UPDATE, and DELETE, the response will be an empty list [] indicating that the SQL was executed successfully. $\mathrm { \backslash n u n H }$ you have obtain the answer by interacting with the database, and you MUST commit your final answer using the format like this:\nAction: Answer\nFinal Answer: [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...]\nDO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.\nYour answer should be accurate. Your answer must be exactly the same as the correct answer.\nIf the question is about modifying the database, then after done operation, your answer field can be anything.\nIf the question is about querying the database, then after done operation, your answer field should be the result of the query.\nWe note that the column names will not be displayed in the result, and you need to ensure both the orders of the columns and rows are correct.\nIf your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.\nOnce you commit your answer or the number of rounds reaches 3, the task will be finished and the system will judge whether you pass the task or not.\n\n\nQuestion What are the pet IDs, checkup dates, and weights in kilograms for checkups with a discharge status of "Completed"? Return the results ordered by checkup date in ascending order and then by weight in ascending order.\nThe name of this table is animal_checkups, and the headers of this table are checkup_id, pet_id, checkup_date, weight_kg, veterinarian_id, discharge_status.:\nassistant: To solve this problem, we need to filter the checkups with a discharge status of "Completed" and then order the results by checkup date in ascending order and then by weight in ascending order.\n\nAction: Operation\n"sql\nSELECT pet_id, checkup_date, weight_kg FROM animal_checkups WHERE discharge_status $=$ "Completed" ORDER BY checkup_date ASC, weight_kg ASC;\n"\nuser: [(2, "2020-03-15", 4), (1, "2020- 03-15", 5), (1, "2020-07-22", 6), (4, "2021-05-20", 7), (3, "2021-05-20", 8), (6, "2022-01-10", 9), (5, "2022-01-10", 10), (7, "2022-06-18", 12), (10, "2023-04-05", 13), (9, "2023-04-05", 14), (8, "2023-04-05", 15), (5, "2023-10-30", 20)]\nassistant: Action: Answer\nFinal Answer: [(2, "2020-03-15", 4), (1, "2020-03-15", 5), (1, "2020-07-22", 6), (4, "2021-05-20", 7), (3, "2021-05-20", 8), (6, "2022-01-10", 9), (5, "2022-01-10", 10), (7, "2022-06-18", 12), (10, "2023-04-05", 13), (9, "2023-04-05", 14), (8, "2023-04-05", 15), (5, "2023-10-30", 20)]\n\nNow, I will give you the question that you need to solve.

Agent: OK.

User: Delete up to 100 packages that are either marked as "lost" with a last update before January 1, 2023, or have an ID that belongs to packages weighing less than 5 kilograms.\nThe name of this table is package_tracking, and the headers of this table are package_id, status, last_updated, weight_kg, destination_zip, carrier_code, estimated_delivery.

Agent: To solve this problem, we need to filter the packages that are either marked as "lost" with a last update before January 1, 2023, or have an ID that belongs to packages weighing less than 5 kilograms, and then delete up to 100 of these packages.\n\nAction: Operation\n"sql\nDELETE FROM package_tracking WHERE (status $=$ "lost" AND last_updated < "2023-01-01") OR package_id IN (SELECT package_id FROM package_tracking WHERE weight_kg $< 5$ LIMIT 100);\n"

User: 1235 (42000): This version of MySQL doesn’t yet support "LIMIT & IN/ALL/ANY/- SOME subquery"

Agent: This error is due to the fact that the LIMIT clause cannot be used in the subquery. We can use a temporary table to solve this problem.\n\nAction: Operation\n"sql\nCREATE TEMPORARY TABLE temp AS SELECT package_id FROM package_tracking WHERE weight $\mathrm { k g } < 5$ LIMIT 100;\nDELETE FROM package_tracking WHERE (status $=$ "lost" AND last_updated < "2023-01-01") OR package_id IN (SELECT package_id FROM temp);\n"

User: []

Agent: Action: Answer\nThe operation was successful, and up to 100 packages were deleted.

# D.4.2 Error Case 6

Task Description: The agent was instructed to insert a new payment record into the membership_payments table for member ID 102, specifying the payment date ("2023-10-15"), amount (75), and payment method ("Credit Card").

Agent Behavior: The agent attempted the following SQL operation:

INSERT INTO membership_payments (member_id, payment_date, amount, payment_method) VALUES (102, "2023-10-15", 75, "Credit Card");

However, the agent explicitly violated the instructions by failing to wrap the SQL command within the required markdown-formatted code block. According to the task instructions, the SQL statement must strictly follow the specified markdown format and execution structure.

Reason for Failure: The task failed due to an agent_validation_failed error resulting from a violation of the explicit formatting instructions. The agent executed the correct logical SQL command but neglected to encapsulate it correctly within markdown syntax.

# Implications and Insights:

• Instruction Compliance: The agent’s oversight highlights a gap in strict adherence to specified formatting and structural guidelines, even when logical accuracy is maintained.   
• Attention to Detail: This example underscores the necessity for careful attention to formatspecific constraints, which are critical for system validation purposes.   
• Potential Solutions: Enhancing the agent’s ability to rigorously follow explicit formatting rules or implementing automated format checks before submission could prevent such failures.

This case emphasizes the critical role of meticulous compliance with explicitly stated operational instructions, highlighting the need for enhanced format verification mechanisms within agent interactions.

# Error Case 6 || Environment: DB || Error Type: agent_validation_failed

User: I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.\nYou have to explain the problem and your solution to me and write down your thoughts.\nAfter thinking and explaining thoroughly, every round you can choose one of the two actions: Operation or Answer.\n\nTo do operation, the format should be like this:\nAction: Operation\n"sql\nSELECT FROM table WHERE condition;\n"\nYou MUST put SQL in markdown format without any other comments. Your SQL should be in one line.\nI will use "Action: Operation" literally to match your SQL.\nEvery time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.\nIf the SQL is not executed successfully, the response will be the error message.\nOtherwise, the response will be the raw MySQL response.\nFor SELECT queries, the response will be the result of the query, such as [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...], where each tuple represents a row and the elements are the values of the columns in the row.\nFor SQL such as INSERT, UPDATE, and DELETE, the response will be an empty list [] indicating that the SQL was executed successfully. $\mathrm { \backslash n u n H }$ you have obtain the answer by interacting with the database, and you MUST commit your final answer using the format like this:\nAction: Answer\nFinal Answer: [(1, "John Doe", "HR"), (2, "Jane Smith", "IT"), ...]\nDO NOT write this pattern unless you are sure about your answer. I expect an accurate and correct answer.\nYour answer should be accurate. Your answer must be exactly the same as the correct answer.\nIf the question is about modifying the database, then after done operation, your answer field can be anything.\nIf the question is about querying the database, then after done operation, your answer field should be the result of the query.\nWe note that the column names will not be displayed in the result, and you need to ensure both the orders of the columns and rows are correct.\nIf your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.\nOnce you commit your answer or the number of rounds reaches 3, the

task will be finished and the system will judge whether you pass the task or not.\n\nNow, I will give you the question that you need to solve.

Agent: OK.

User: Insert a new payment record for member ID 102 with a payment date of "2023-10-15", amount 75, and payment method "Credit Card" into the membership payments table.\nThe name of this table is membership_payments, and the headers of this table are payment_id, member_id, payment_date, amount, payment_method.

Agent: Action: Operation\nINSERT INTO membership_payments (member_id, payment_date, amount, payment_method) VALUES (102, "2023-10-15", 75, "Credit Card");

# D.5 Agent Context Limit

# D.5.1 Error Case 7

Task Description: The agent was given a system administration task on an Ubuntu system: to append several outputs and messages (including static strings, directory contents, hostname, and extracted file data) to the file /var/log/tee_test.log using the tee command, and then set the file’s permissions to 644 and ownership to root:adm. The task had a strict interaction cap of five rounds.

Agent Behavior: The agent provided a correct breakdown of the required steps and formulated a Bash script attempting to execute all operations in one go:

```shell
tee -a /var/log/tee_test.log << "Line 1"  
tee -a /var/log/tee_test.log << "Line 2"  
find / -print | tee -a /var/log/tee_test.log  
hostname | tee -a /var/log/tee_test.log  
grep root /etc/passwd | tee -a /var/log/tee_test.log  
wc -l /etc/passwd | tee -a /var/log/tee_test.log  
tee -a /var/log/tee_test.log << "Line 3"  
tee -a /var/log/tee_test.log << "Done"  
chmod 644 /var/log/tee_test.log  
chown root:adm /var/log/tee_test.log 
```

Although all command steps were logically appropriate, the inclusion of the find / command caused the output to explode in size, producing over 4 million characters of output and ultimately exceeding the LLM context window.

Reason for Failure: The task failed due to the agent_context_limit error. The excessive length of the command output (especially from find /) overwhelmed the language model’s context window, causing the agent to become unable to continue interaction or process the remainder of the task.

# Implications and Insights:

• Unbounded Output Risks: The use of high-volume commands like find / without filtering or scoping demonstrates a lack of foresight about output size constraints.   
• Context Management Deficiency: This case highlights the need for LLM agents to proactively assess output volume when planning shell operations in multi-step tasks.   
• Potential Solutions: Agents should be trained or prompted to use output-limiting constructs (e.g., find / -maxdepth 1, head, du -sh) or to chunk operations across interactions to preserve context capacity.

This case underscores a crucial limitation of LLM agents when interacting with open-ended system environments: the inability to manage large outputs effectively. It motivates future work on contextaware execution planning and resource-aware decision making for long-context tasks.

User: I will provide you with a task to perform on a Linux (Ubuntu) system. Your objective is to complete the task by executing the appropriate Bash commands.\n\n### Interaction Rules:\n1. Thorough Analysis and Reasoning: $\mathrm { \backslash n }$ - Before performing any action, carefully analyze the task and explain your thought process.\n - Include a detailed explanation of the logic behind your choice of commands and approach. $\mathrm { \ u n m } 2$ . Action Choices:\n - At the end of your reasoning, select one and only one action for each turn.\n - "bash": When you need to execute a command or perform an operation, provide the corresponding Bash code. Structure your response as: $\mathrm { \backslash n }$ Act: bash\n "bash\n # Your Bash command(s) here\n "\n - "finish": When the task is complete and no further action is required, conclude with: $\backslash \mathrm { n }$ Act: finish\n\n3. Other Guidelines: $\backslash \boldsymbol { \mathrm { n } } \boldsymbol { \cdot } \boldsymbol { \mathrm { I } }$ will use "Act: bash" and "Act: finish" literally to determine whether your action is to execute commands or conclude the task.\n - Use the provided format accurately and consistently. $\backslash \mathrm { n }$ - Ensure all Bash commands are compatible with Linux (Ubuntu) systems. $\mathrm { \backslash n }$ - Avoid interactive operations (e.g., read, readline) in your Bash commands. $\mathrm { . } \mathrm { \backslash n u 4 }$ . Task Completion: $\backslash \mathrm { n }$ - The task will conclude either when you select the "finish" action or when the number of rounds reaches $5 . \mathrm { { i n } }$ - The system will evaluate your performance to determine if the task was successfully completed.\n\nNow, I will give you the question that you need to solve.

Agent: OK.

User: Append the following command outputs to "/var/log/tee_test.log" using tee: "Line 1", "Line $2 "$ , the contents of "/" and "/home" directories, the system’s hostname, the root user entry from "/etc/passwd", the line count of "/etc/passwd", "Line $3 "$ , and "Done". Ensure the file is owned by root:adm with permissions 644.

Agent: To accomplish this task, we need to carefully analyze the requirements and break it down into manageable steps. $\mathrm { \backslash n u n 1 }$ . We need to append the specified text and directory contents to a log file named "/var/log/tee_test.log". $\mathrm { l n } 2$ . The log file should be owned by the "root:adm" group and have permissions $6 4 4 . \mathrm { \backslash n } 3$ . We need to use the "tee" command to append the specified text and directory contents to the log file.\n\nHere’s the step-by-step thought process: $\mathrm { \backslash n u n } .$ - First, we’ll use the "tee" command to append the specified text to the log file.\n-Then, we’ll use the "find" command to get the contents of the "/" and "/home" directories.\n-Next, we’ll use the "hostname" command to get the system’s hostname.\n- After that, we’ll use the "grep" command to get the root user entry from "/etc/passwd".\n- We’ll also use the "wc" command to get the line count of "/etc/passwd".\n- Finally, we’ll use the "chmod" and "chown" commands to set the permissions and ownership of the log file.\n\nNow, let’s select the action for this task.\n\nAct: bash\n"bash\ntee -a /var/log/tee_test.log $\ll <$ "Line 1"\ntee -a /var/log/tee_test.log $\ll <$ "Line 2"\nfind / -print | tee -a /var/log/tee_test.log\nhostname | tee -a /var/log/tee_test.log\ngrep root /etc/passwd | tee -a /var/log/tee_test.log\nwc -l /etc/passwd | tee -a /var/log/tee_test.log\ntee -a /var/log/tee_test.log $\ll <$ "Line 3"\ntee -a /var/log/tee_test.log «< "Done"\nchmod 644 /var/log/tee_test.log\nchown root:adm /var/log/tee_test.log\n"\nThis will append the specified text and directory contents to the log file, set the permissions to 644, and change the ownership to "root:adm".

User: The output of the OS:\n\nLine 1\nLine 2\n/\n/opt\n/boot\n/home\n/home/ubuntu\n/home /ubuntu/.bash_log...4250311 characters is omitted...nssl.cnf\n/.dockerenv\ndd3bf76b6367\nro ot:x:0:0:root:/root:/bin/bash\n19 /etc/passwd\nLine 3\nDone\n

Agent:
# LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark

Guangyi Liu† Zhejiang University Hangzhou, China

Zhiming Chen vivo AI Lab Hangzhou, China

Hao Wang vivo AI Lab ShenZhen, China

Pengxiang Zhao† Zhejiang University Hangzhou, China

Yuxiang Chai vivo AI Lab Hangzhou, China

Shibo He Zhejiang University Hangzhou, China

Liang Liu‡ vivo AI Lab Hangzhou, China

Shuai Ren vivo AI Lab ShenZhen, China

Wenchao MengB Zhejiang University Hangzhou, China wmengzju@zju.edu.c

#

![](images/e2d3f7a563d319179d3c813b390dc0c76e941188e9f49e5228211110cf23af6f.jpg)

![](images/4c4d35735eae281d1f1057ce0c33cc35b855caea648c2fcc7a9bcaee13ec35c2.jpg)

#

![](images/19e7318d340973f1fd6880e8a878c985450941c98d9179cce938dcfe4db91d27.jpg)

![](images/c9d22948372d67a96d2b79a761937ea2c9689aeef94bc8a9e87c0517130e485a.jpg)

#

![](images/defa093de6cc066acbf9f011d3d83fe5e0a11d7f05287a09adcf1f2893921b40.jpg)

![](images/d41dfcad36d590182e67e570e3b4c5588eed31d239e6d42eea68f8b95268ebcd.jpg)  
Same In Domain OOD

![](images/4d2155bb3c57fa17440aaec6dfb521d38495ad744b4bea5836fe71b91d127e4a.jpg)  
Figure 1: The LearnAct Framework and LearnGUI Benchmark focus on addressing the long-tail challenges in mobile GUI agent performance through demonstration-based learning. From rule-based automation to LLM-powered agents, mobile GUI automation has evolved significantly, yet still struggles with long-tail scenarios due to interface diversity. Our LearnAct framework introduces demonstrationbased learning to effectively handle these challenges, outperforming existing methods in both offline and online evaluations.

# ABSTRACT

Mobile GUI agents show promise in automating tasks but face generalization challenges in diverse real-world scenarios. Traditional approaches using pre-training or fine-tuning with massive datasets struggle with the diversity of mobile applications and user-specific tasks. We propose enhancing mobile GUI agent capabilities through human demonstrations, focusing on improving performance in unseen scenarios rather than pursuing universal generalization through larger datasets. To realize this paradigm, we introduce LearnGUI, the first comprehensive dataset specifically designed for studying demonstration-based learning in mobile GUI agents. It comprises 2,252 offline tasks and 101 online tasks with high-quality

human demonstrations. We further develop LearnAct, a sophisticated multi-agent framework that automatically extracts knowledge from demonstrations to enhance task completion. This framework integrates three specialized agents: DemoParser for knowledge extraction, KnowSeeker for relevant knowledge retrieval, and ActExecutor for demonstration-enhanced task execution. Our experimental results show significant performance gains in both offline and online evaluations. In offline assessments, a single demonstration improves model performance, increasing Gemini-1.5-Pro’s accuracy from $1 9 . 3 \%$ to $5 1 . 7 \%$ . In online evaluations, our framework enhances UI-TARS-7B-SFT’s task success rate from $1 8 . 1 \%$ to $3 2 . 8 \%$ . Learn-Act framework and LearnGUI benchmark establish demonstrationbased learning as a promising direction for more adaptable, personalized, and deployable mobile GUI agents. The project resources are available at https://lgy0404.github.io/LearnAct.

![](images/bbd027a0ed616b98f4cf375fbc6ad44e81e7bd58a5957b8d056d550bd8673c99.jpg)

![](images/49ed8b17d9575cf06b584be3dc22a7b7545722d1f1d67d7fb6cda059c5e411a2.jpg)

![](images/9a47fdd1a81d634067a634d627664a28269f017c4e7ac351d406876c5aba2b30.jpg)

![](images/0395ec59bf192f3064cdc3a25f48f1044ff41df3d82c40e9cab2201f89d84745.jpg)

![](images/79cb848a7fdd91b32e9cc1f1f2befb8e3701e3795f8ae0c5d1fe3ecd824fbf9b.jpg)

![](images/6e1a6a2922b29b74bf0a9bc15a740af52cb69f6775500fd884256ce3ff5e6370.jpg)  
Support Task: Please check the temperature in the living room for me and adjust the windows and air conditioner to a suitable state. Query Task: Please check the humidity in the bedroom for me and adjust the humidifier and windows to a suitable state.

![](images/8f7eaa40ba8c0439ee990224ed672a27fbff40202b408a011597c7f3d2d55c73.jpg)

![](images/1baa096830fdde7cba09c216cee5f75587e8f4777106da3a1ab1558b14f9a070.jpg)

![](images/c9c2f7dc66c454b2a7359e26f89c6bc1ee181d9b3949f28a6eda0f6760dce184.jpg)

![](images/9bb12ca054656e1cbd583e7bbe0c28c42e21f2878aa098d754b3c4de879f051a.jpg)  
Figure 2: A toy example for demonstration learning on mobile GUI Agent. We build a benchmark named LearnGUI for demonstration learning on Mobile GUI Agent, which provides different few-shot task combinations and offers multi-dimensional metrics including task similarity, UI similarity, and action similarity between support tasks and query tasks.

# 1 INTRODUCTION

Mobile device automation has evolved significantly over time, from simple rule-based scripts to sophisticated AI-powered agents [17, 32, 38, 43]. Traditional automation approaches like Robotic Process Automation (RPA) [1] and rule-based shortcuts [10, 13] relied on predefined scripts to execute repetitive tasks, but they struggled with dynamic interfaces, required frequent maintenance when apps updated, and lacked understanding of complex user intentions.

More recently, mobile Graphical User Interface (GUI) agents have emerged as a transformative technology with the potential to revolutionize how humans interact with mobile devices. These agents leverage Large Language Models (LLMs) to autonomously complete human tasks through environmental interaction [6, 18, 20, 28, 29, 33, 36, 37, 46]. They perceive phone states of mobile phone by observing screens (through screenshots or application UI trees) and generate actions (such as CLICK, TYPE, SWIPE, PRESS_BACK, PRESS_HOME, and PRESS_ENTER) that are executed via the phone user interface [17, 32, 38, 43]. By harnessing the powerful perception and reasoning capabilities of LLMs, mobile GUI agents have the potential to fundamentally change how people interact with their mobile devices, bringing to life the "J.A.R.V.I.S." effect seen in science fiction.

Despite these promising advances, mobile GUI agents continue to face significant challenges in real-world deployment scenarios. The immense diversity of mobile applications and user interfaces creates pervasive long-tail scenarios where current agents struggle

to perform effectively. The prevailing approaches to building modern mobile GUI agents rely on either the inherent capabilities of general-purpose LLMs [18, 20, 28, 29, 34, 36, 37, 46] or fine-tuning with large volumes of data [11, 16, 41, 48]. However, these methods face fundamental limitations when confronted with diverse realworld usage scenarios. As of 2025, billions of users interact with 1.68 million applications on Google Play alone [17], each with unique task requirements and UI layouts [32, 43]. Pre-training or finetuning datasets cannot feasibly cover this immense variety, leading to poor performance in unseen scenarios and hindering the widespread adoption of mobile GUI agents [14], as illustrated in Figure 1 (left side). Traditional approaches simply cannot cover the entire spectrum of possible interactions and user-specific requirements across this heterogeneous landscape.

To address these limitations, we propose a novel paradigm that enhances mobile GUI agent capabilities through few-shot demonstration learning. Unlike traditional approaches that either lack flexibility or require massive datasets, our demonstrationbased approach achieves both robustness and personalization by learning from a small number of user-provided examples. We recognize that mobile users have unique, repetitive tasks with inherent variability—such as smart home control with dynamic configurations, health monitoring with personalized parameters, or enterprise software with company-specific layouts. These scenarios combine stable patterns with variable elements, creating a "personalization gap" that pre-trained models cannot bridge. By leveraging user-specific demonstrations, our approach enables personalized assistants that learn both consistent patterns and adaptation strategies, acquiring task-specific knowledge impossible to cover in general training datasets. This personalization allows mobile GUI agents to overcome performance bottlenecks and provide truly helpful automation for the tasks users most want to delegate.

To fill the gap in high-quality demonstration data, we introduce LearnGUI, the first dataset specifically designed to research and evaluate mobile GUI agents’ ability to learn from few-shot demonstrations. Built upon AMEX [5] and Android-World [23], LearnGUI comprises 2,252 offline few-shot tasks and 101 online tasks with high-quality human demonstrations. This dataset enables systematic research into demonstration-based learning for mobile GUI agents. A toy example for LearnGUI is shown in Figure 2.

Furthermore, we present LearnAct, a multi-agent framework that automatically understands human demonstrations, generates instructional knowledge, and uses this knowledge to assist mobile GUI agents in reasoning about unseen scenarios. LearnAct consists of three specialized agents: (1) DemoParser, a knowledge generation agent that extracts usable knowledge from demonstration trajectories to form a knowledge base; (2) KnowSeeker, a knowledge retrieval agent that searches the knowledge base for demonstration knowledge relevant to the current task; and (3) ActExecutor, a task execution agent that combines user instructions, real-time GUI environment, and retrieved demonstration knowledge to perform tasks effectively.

Our experimental results decisively validate the effectiveness of demonstration-based learning for mobile GUI agents, as shown in Figure 1 (right side). In offline evaluations, a single demonstration dramatically improves model performance across diverse scenarios,

Table 1: Comparison of different datasets and environments for benchmarking Mobile GUI agents. Column definitions: # Inst. (number of instructions), # Apps (number of applications), # Step (average steps per task), Env. (supports environment interactions), HL (has high-level instructions), LL (has low-level instructions), GT (provides ground truth trajectories), FS (supports few-shot learning).   

<table><tr><td>Dataset</td><td># Inst.</td><td># Apps</td><td># Step</td><td>Env.</td><td>HL</td><td>LL</td><td>GT</td><td>FS</td></tr><tr><td>PixelHelp [15]</td><td>187</td><td>4</td><td>4.2</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>MoTIF [4]</td><td>276</td><td>125</td><td>4.5</td><td>X</td><td>✓</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>UIbert [3]</td><td>16,660</td><td>-</td><td>1</td><td>X</td><td>X</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>UGIF [27]</td><td>523</td><td>12</td><td>6.3</td><td>X</td><td>✓</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>AITW [24]</td><td>30,378</td><td>357</td><td>6.5</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>AITZ [45]</td><td>2,504</td><td>70</td><td>7.5</td><td>X</td><td>✓</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>AndroidControl [14]</td><td>15,283</td><td>833</td><td>4.8</td><td>X</td><td>✓</td><td>✓</td><td>✓</td><td>X</td></tr><tr><td>AMEX [5]</td><td>2,946</td><td>110</td><td>12.8</td><td>X</td><td>✓</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>MobileAgentBench [30]</td><td>100</td><td>10</td><td>-</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td></tr><tr><td>AppAgent [44]</td><td>50</td><td>10</td><td>-</td><td>X</td><td>✓</td><td>X</td><td>X</td><td>X</td></tr><tr><td>LlamaTouch [47]</td><td>496</td><td>57</td><td>7.01</td><td>✓</td><td>✓</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>AndroidWorld [23]</td><td>116</td><td>20</td><td>-</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td></tr><tr><td>AndroidLab [40]</td><td>138</td><td>9</td><td>8.5</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td></tr><tr><td>LearnGUI (Ours)</td><td>2,353</td><td>73</td><td>13.2</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

with the most striking results seen in Gemini-1.5-Pro [26], whose accuracy increases from $1 9 . 3 \%$ to $5 1 . 7 \%$ (a $1 9 8 . 9 \%$ relative improvement). Performance gains are particularly pronounced in complex applications, with accuracy in CityMapper increasing from $1 4 . 1 \%$ t o $6 9 . 4 \%$ and in To-Do apps from $1 7 . 4 \%$ to $6 9 . 2 \%$ . For real-world online evaluations, our framework demonstrates exceptional effectiveness, with Qwen2-VL-7B [31] with LearnAct achieving significant performance gains, while UI-TARS-7B-SFT [22]’s task success rate improves from $1 8 . 1 \%$ to $3 2 . 8 \% ( + 1 4 . 7 \% )$ . These findings offer a practical pathway to developing more adaptable and personalized mobile GUI agents.

In summary, our contributions are as follows:

• We develop LearnGUI, the first dataset designed for studying demonstration-based learning in mobile GUI agents, comprising 2,252 offline and 101 online tasks with high-quality human demonstrations.   
• We design and implement LearnAct, a sophisticated multiagent framework that systematically extracts, retrieves, and leverages knowledge from human demonstrations. This framework includes three specialized components: DemoParser (knowledge extraction), KnowSeeker (knowledge retrieval), and ActExecutor (task execution).   
• Our evaluations demonstrate unprecedented performance gains: a single demonstration improves Gemini-1.5-Pro [26]’s accuracy by $1 9 8 . 9 \%$ in offline tests, while enhancing UI-TARS-7B-SFT [22]’s online task success rate from $1 8 . 1 \%$ to $3 2 . 8 \%$ , advancing mobile GUI agents toward greater adaptability and practical deployability.

# 2 RELATED WORK

Mobile GUI Datasets and Environments. The development of mobile GUI agents relies heavily on high-quality datasets for training and evaluation. Table 1 compares LearnGUI and existing mobile

GUI datasets and benchmarks. These resources can be broadly categorized into static datasets and dynamic benchmarking environments. Static datasets [3–5, 14, 15, 24, 27, 30, 45] typically provide natural language task descriptions, UI states (screenshots and/or application UI trees), and corresponding user actions (CLICK, SWIPE, TYPE, and other standardized interactions). These datasets vary in scale, ranging from hundreds to tens of thousands of instructions across different applications. Recent work like AppAgent [44] has explored demonstration-based learning but without ground truth annotations or systematic analysis, providing only 50 tasks across 10 applications with high-level instructions. Notably, the average task length varies significantly across datasets, with AMEX [5] featuring substantially longer sequences (12.8 steps on average) compared to AndroidControl (4.8 steps) and AITW [24] (6.5 steps). Benchmarking environments, on the other hand, typically select a limited number of tasks and applications to provide dynamic testing environments [17]. These frameworks evaluate agent performance through metrics such as task completion rates, critical state achievements, and execution time. Examples include Llama-Touch [47], AndroidWorld [23], and AndroidLab [40], which offer interactive environments but lack few-shot demonstration capabilities. We present the first systematic study of demonstration-based learning for mobile GUI agents through LearnGUI, which distinguishes itself through three key innovations. First, it is designed to evaluate few-shot learning capabilities with a comprehensive collection of 2,252 offline tasks and 101 online tasks. Built upon AMEX [5] and AndroidWorld [23], which feature longer, more complex tasks ideal for out-of-distribution and demonstration-based learning scenarios, LearnGUI provides a unified framework for both offline and online evaluation. Second, while the original AMEX [5] dataset contains 2,946 independent tasks unsuitable for few-shot evaluation, we conducted detailed analyses to transform and enhance this resource. Specifically, we made three key modifications: (1) Action Space Standardization, refining the original action space by removing inconsistent TASK_IMPOSSIBLE actions, enhancing TASK_COMPLETE to support information retrieval tasks, and standardizing formats for consistency; (2) K-shot Task Combinations, constructing systematic task groupings by recovering application context, computing instruction similarity within applications, and creating k-shot combinations with similar tasks as support demonstrations; and (3) Similarity Measurement, computing UI and action similarity through descriptive representations, enabling analysis of how different similarity types affect learning efficacy. Third, regarding online evaluation, AndroidWorld [23] originally provides 116 dynamically constructed tasks without human demonstration trajectories. We collected 101 high-quality human demonstrations based on AndroidWorld’s environment and dynamic instructions, forming LearnGUI-Online for evaluating the few-shot capabilities of mobile GUI agents in real-time scenarios. By addressing the limitations of existing datasets, LearnGUI enables systematic research into few-shot learning for mobile GUI agents with varying k-shot configurations and controlled similarity conditions between support and query tasks.

Mobile GUI Agents. Mobile GUI agents are intelligent systems that leverage large language models to understand, plan, and execute tasks on mobile devices by integrating natural language

processing, multimodal perception, and action execution capabilities [32, 38]. Recent developments in this field have explored various approaches to enhance agent performance and generalizability. One prominent category of work focuses on designing effective prompting strategies to guide pre-trained LLMs without additional training [8, 35, 42]. By crafting prompts that incorporate task descriptions, interface states, and action histories, researchers can direct model behavior toward specific automation goals [25, 28, 29, 37]. These approaches leverage the inherent capabilities of generalpurpose LLMs but often struggle with complex tasks. A second category involves adapting LLMs specifically for mobile automation through fine-tuning techniques [7, 9, 11, 16, 19, 21, 41]. These methods train models on GUI-specific data to enhance their understanding of and interaction with graphical interfaces. While improving performance over pre-training approaches, these fine-tuned models require substantial training data and still face generalization challenges. Despite the progress made by both approaches, a fundamental limitation persists: the inability to generalize effectively to out-of-distribution scenarios. These methods both struggle with unseen applications, novel UI layouts, or unexpected task variations. These limitations stem from the impossibility of covering all potential real-world scenarios during training, creating significant bottlenecks in mobile GUI agent development. To address these critical challenges, we introduce LearnAct, a sophisticated multiagent framework that learns and reasons from screenshots without requiring UI tree information. The framework extracts, retrieves, and utilizes demonstration knowledge through three specialized components, enabling effective adaptation to new scenarios with minimal demonstrations.

# 3 LEARNGUI DATASET

# 3.1 Task Definition

Mobile GUI tasks require agents to interact with digital environments by executing actions to fulfill user instructions. These tasks can be formally described as a Partially Observable Markov Decision Process (POMDP), defined as $\boldsymbol { \mathcal { M } } = ( S , O , \mathcal { A } , \mathcal { T } , \mathcal { R } )$ , where $s$ is the state space (current state of the mobile device), $o$ is the observation space (instructions, screenshots, UI trees, etc.), $\mathcal { A }$ is the action space (e.g., click, type, swipe), $\mathcal { T } : \boldsymbol { S } \times \mathcal { A }  \boldsymbol { S }$ is the state transition function, and $\mathcal { R } : \mathcal { S } \times \mathcal { A }  [ 0 , 1 ]$ is the reward function. For example, a user might request the agent to "find the cheapest hotel in Paris for next weekend." The agent must perceive the current screen—either through an image or a UI tree—and execute a sequence of actions to complete the given task.

The key innovation in our approach is the integration of human demonstration knowledge into this POMDP framework. By incorporating demonstration knowledge $\mathcal { D }$ into the decision process, we enhance the agent’s ability to handle out-of-distribution scenarios. This knowledge influences the agent’s policy $\pi : O \times \mathcal { D }  \mathcal { A }$ which maps observations and relevant demonstration knowledge to actions, providing valuable examples of successful interaction patterns.

To study the impact of demonstration-based learning on mobile GUI agents, we need a dataset that provides various k-shot demonstrations with controlled similarity relationships between support and query tasks. This allows us to systematically investigate

how demonstration quantity and task similarity affect agent performance. While cross-application knowledge transfer remains an interesting research direction, we focus on within-application task learning, as this represents the most practical use case where users would provide demonstrations for applications they frequently use.

Our dataset design specifically enables research on three key dimensions:

(1) Unified comprehensive evaluation framework: Learn-GUI provides a standardized platform for studying few-shot demonstration learning in mobile GUI agents, featuring a unified action space and evaluation protocols that reflect real-world use cases   
(2) K-shot demonstration learning: The dataset systematically explores how varying quantities of demonstrations (k=1, 2, or 3) affect agent performance, enabling research on the optimal number of examples needed   
(3) Multi-dimensional similarity analysis: LearnGUI enables investigation of how different types of similarity between demonstration and query tasks influence learning efficacy and generalization capabilities

This comprehensive approach allows for a nuanced analysis of how mobile GUI agents can leverage human demonstrations to improve task performance, especially in scenarios not covered by their training data.

# 3.2 Data Collection

The LearnGUI dataset consists of two components: LearnGUI-Offline for systematic evaluation of few-shot learning capabilities across varying similarity conditions, and LearnGUI-Online for real-time assessment in an interactive environment. Both components share a unified action space to ensure consistent evaluation, as detailed in Table 2.

Table 2: LearnGUI Action Space   

<table><tr><td>Action</td><td>Definition</td></tr><tr><td>CLICK[x, y]</td><td>Click at coordinates (x, y).</td></tr><tr><td>TYPE{text]</td><td>Type the specified text.</td></tr><tr><td>SWIPE [direction]</td><td>Swipe in the specified direction.</td></tr><tr><td>PRESS_HOME</td><td>Go to the home screen.</td></tr><tr><td>PRESS_BACK</td><td>Go back to the previous app screen.</td></tr><tr><td>PRESS_ENTER</td><td>Press the enter button.</td></tr><tr><td>TASK_COMPLETE[answer]</td><td>Mark the task as complete. Provide answer inside brackets if required.</td></tr></table>

3.2.1 LearnGUI-Offline. We built LearnGUI-Offline by restructuring and enhancing the AMEX dataset [5], which contains 2,946 independent mobile tasks. To transform this resource for few-shot learning evaluation, we made several key modifications:

Action Space Standardization. We refined the original action space to better align with real-world scenarios. First, we removed

Table 3: Statistics of LearnGUI dataset splits. Each split is analyzed across multiple dimensions: Tasks (number of tasks), Apps (number of applications covered), Step actions (total action steps), similarity metrics (Avg Ins/UI/ActSim), and distribution across four similarity profiles categorized by high (SH) and low (SL) UI and action similarity.   

<table><tr><td>Split</td><td>K-shot</td><td>Tasks</td><td>Apps</td><td>Step actions</td><td>Avg Inssim</td><td>Avg UISim</td><td>Avg ActSim</td><td>UIShActSH</td><td>UIShActSL</td><td>UISLActSH</td><td>UISLActSL</td></tr><tr><td>Offline-Train</td><td>1-shot</td><td>2,001</td><td>44</td><td>26,184</td><td>0.845</td><td>0.901</td><td>0.858</td><td>364</td><td>400</td><td>403</td><td>834</td></tr><tr><td>Offline-Train</td><td>2-shot</td><td>2,001</td><td>44</td><td>26,184</td><td>0.818</td><td>0.898</td><td>0.845</td><td>216</td><td>360</td><td>358</td><td>1,067</td></tr><tr><td>Offline-Train</td><td>3-shot</td><td>2,001</td><td>44</td><td>26,184</td><td>0.798</td><td>0.895</td><td>0.836</td><td>152</td><td>346</td><td>310</td><td>1,193</td></tr><tr><td>Offline-Test</td><td>1-shot</td><td>251</td><td>9</td><td>3,469</td><td>0.798</td><td>0.868</td><td>0.867</td><td>37</td><td>49</td><td>56</td><td>109</td></tr><tr><td>Offline-Test</td><td>2-shot</td><td>251</td><td>9</td><td>3,469</td><td>0.767</td><td>0.855</td><td>0.853</td><td>15</td><td>42</td><td>55</td><td>139</td></tr><tr><td>Offline-Test</td><td>3-shot</td><td>251</td><td>9</td><td>3,469</td><td>0.745</td><td>0.847</td><td>0.847</td><td>10</td><td>36</td><td>49</td><td>156</td></tr><tr><td>Online-Test</td><td>1-shot</td><td>101</td><td>20</td><td>1,423</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

TASK_IMPOSSIBLE actions due to inconsistent labeling in the original dataset, which included errors such as tasks being incorrectly marked as impossible. Second, we enhanced TASK_COMPLETE to TASK_COMPLETE[answer] for information retrieval tasks. Many mobile tasks require returning specific information rather than just completion status. This aligns with both AMEX [5] and Android-World [23] paradigms.

K-shot Task Combinations. We constructed systematic k-shot task combinations through a multi-step process. We began by recovering the application context for each task through instruction and screenshot analysis, as the original dataset lacked explicit app labels. Next, we computed instruction similarity between tasks within the same application using the all-MiniLM-L6-v2 model. Finally, we created k-shot combinations $_ { ( \mathrm { k } = 1 , 2 , 3 ) }$ for each query task by selecting the k most similar tasks within the same application as support demonstrations, ensuring that the average similarity exceeded a minimum threshold of 0.6. This process yielded 2,252 tasks with valid k-shot combinations.

Similarity Measurement. To enable multi-dimensional similarity analysis, we computed metrics across three key dimensions. For Instruction Similarity, we utilized the scores calculated during the K-shot Task Combinations process. For UI Similarity, we merged the UI trees from all steps of each task and calculated similarity using TF-IDF vectorization and cosine similarity, capturing the visual and structural similarity of interfaces. For Action Similarity, following the DemoParser approach detailed in Section 4.1, we generated descriptive representations of each action and computed embedding-based cosine similarity between task pairs.

3.2.2 LearnGUI-Online. For evaluating mobile GUI agents in realtime interactive scenarios, we developed LearnGUI-Online based on the AndroidWorld environment [23]. While AndroidWorld provides 116 dynamically constructed task templates, it lacks human demonstration trajectories essential for few-shot learning evaluation.

We identified 101 tasks suitable for human completion, excluding 15 tasks that proved challenging for human users. We then collected high-quality human demonstrations for these tasks. For tasks with dynamic elements, we generated specific instances and recorded corresponding demonstrations.

The resulting LearnGUI-Online dataset provides a realistic testbed for evaluating few-shot learning capabilities in mobile GUI agents under authentic conditions.

# 3.3 Dataset Statistics

Table 1 presents the comprehensive statistics of the LearnGUI dataset in comparison with existing datasets. With 2,353 instructions across 73 applications and an average of 13.2 steps per task, LearnGUI offers rich data for studying demonstration-based learning in mobile GUI agents. The dataset provides various k-shot combinations $_ { ( \mathrm { k = 1 } , 2 , 3 ) }$ ) for each task, along with multi-dimensional similarity metrics across instruction, UI, and action dimensions. This design enables systematic analysis of how different types and quantities of demonstrations affect learning outcomes. The similarity distribution reflects the natural variation in mobile tasks within applications, with a meaningful spread across similarity levels that allows for a detailed investigation of knowledge transfer under different conditions. A detailed visualization of these similarity distributions is provided in Appendix A.

# 3.4 Dataset Splits

We divided LearnGUI-Offline into training and testing splits to enable systematic evaluation of few-shot learning capabilities. Table 3 presents the detailed statistics of these splits, including the distribution of tasks across different similarity profiles.

The training set contains 2,001 tasks for each k-shot configuration (1, 2, and 3), spanning 44 applications with an average of 13.1 steps per task. The test set includes 251 tasks per k-shot configuration across 9 applications. Both splits maintain the same action space and similarity measurement methodology.

Based on empirical analysis, we established threshold values of 0.9447 for UI similarity and 0.9015 for action similarity to classify tasks into high (SH) and low (SL) similarity categories, enabling systematic analysis of how different similarity types affect learning from demonstrations.

As shown in Figure 3, we classify tasks into four categories based on UI and action similarity:

• UISHActSH: High UI similarity and high action similarity. For example, in a smart home app, two tasks that both involve adjusting the brightness of different lights in the living room would navigate through similar UI screens.   
• UISHActSL: High UI similarity but low action similarity. For instance, in a smart home app, turning on all lights with a single button press versus adjusting each light’s color temperature.   
• UISLActSH: Low UI similarity but high action similarity. For example, setting a schedule for lights versus setting a

![](images/27e138dfddf04acb20a626ac3dd89c5735027ad04a8c4555f213ebe746d5a679.jpg)  
Figure 3: Joint distribution of UI similarity and action similarity in LearnGUI-Offline. The scatter plot shows the relationship between UI and action similarity measures across task pairs. The quadrant divisions represent our categorization of tasks into four profiles: $\mathrm { U I } _ { \mathrm { S H } } \mathrm { A c t } _ { \mathrm { S H } }$ , UISHActSL, $\mathrm { U I } _ { \mathrm { S L } } \mathrm { A c t } _ { \mathrm { S H } }$ , and $\mathrm { U I } _ { \mathrm { S L } } \mathrm { A c t } _ { \mathrm { S L } }$ , enabling analysis of how different similarity combinations affect learning transfer.

schedule for the thermostat—different UI screens but similar action patterns.

• $\mathbf { U I _ { S L } A c t _ { S L } }$ : Low UI similarity and low action similarity. For instance, checking security camera footage versus creating a scene that coordinates multiple devices.

This categorization enables a detailed analysis of how different types of similarity affect learning efficacy. For instance, we can investigate whether UI similarity or action similarity has a greater impact on successful knowledge transfer from demonstrations.

Additionally, the LearnGUI-Online test set contains 101 tasks across 20 applications. Unlike the offline dataset, these tasks are evaluated in real time through direct interaction with the mobile environment.

The comprehensive structure of LearnGUI, with its carefully designed splits and similarity classifications, provides a resource for studying how mobile GUI agents can learn from demonstrations under varying conditions of task similarity and demonstration quantity.

# 4 METHOD: LEARNACT

Building on the insights from our LearnGUI dataset, we introduce LearnAct, a novel framework designed to break through the limitations of traditional training approaches for mobile GUI agents. Rather than pursuing universal generalization through extensive training data, LearnAct establishes demonstration-based learning as a paradigm for developing more adaptable, personalized, and practically deployable mobile GUI agents. As illustrated in Figure 4,

LearnAct is a sophisticated multi-agent framework that automatically understands human demonstrations, generates instructional knowledge, and leverages this knowledge to assist mobile GUI agents in reasoning about unseen scenarios. The LearnAct framework consists of three specialized components, each addressing a critical aspect of demonstration-based learning: (1) DemoParser (Section 4.1), a knowledge generation agent that extracts usable knowledge from demonstration trajectories to form a knowledge base; (2) KnowSeeker (Section 4.2), a knowledge retrieval agent that searches the knowledge base for demonstration knowledge relevant to the current task; and (3) ActExecutor (Section 4.3), a task execution agent that combines user instructions, real-time GUI environment, and retrieved demonstration knowledge to perform tasks effectively.

# 4.1 DemoParser

The DemoParser transforms raw human demonstrations into structured demonstration knowledge. It takes as input a raw action sequence (consisting of coordinates-based clicks, swipes, and text inputs) along with corresponding screenshots and task instructions. It then utilizes a vision-language model to generate semantically descriptive action descriptions that capture the essence of each demonstration step (e.g., “On Search Page, click the search box, to enter keywords”). Building on these descriptions, it constructs a structured knowledge base that records both the high-level action semantics and the contexts in which they occur, as shown in Figure 5.

Formally, DemoParser implements a knowledge generation function $G : J \times S \times \mathcal { A }  \mathcal { K }$ , where $\boldsymbol { \mathcal { T } }$ represents the space of instructions, $s$ is the space of screenshot sequences, $\mathcal { A }$ is the space of action sequences, and $\mathcal { K }$ is the knowledge space. For each demonstration trajectory $( i , s , a ) \in \mathcal { I } \times S \times \mathcal { A }$ , DemoParser generates a knowledge entry $k \in \mathcal { K }$ that encapsulates the demonstration in a semantically descriptive format, converting raw coordinate-based actions (e.g., CLICK[123,456]) into meaningful operation descriptions (e.g., "click search box").

The knowledge generation process is decomposed into a sequence of description generation steps for each action in the demonstration trajectory. Let $d _ { j }$ represent the description for action $a _ { j }$ which is generated using a context-aware description function $\delta : \mathcal { T } \times \mathcal { A } _ { j } \times \mathcal { V } _ { j } \times \mathcal { H } _ { j - 1 } \to \mathcal { D }$ , where $\mathcal { V } _ { j }$ is the visual representation of action $a _ { j }$ execution and $\mathcal { H } _ { j - 1 } = \{ d _ { 1 } , d _ { 2 } , \dotsc , d _ { j - 1 } \}$ is the history of previous action descriptions.

Algorithm 1 in Appendix B.3 outlines the knowledge generation process. For each demonstration, DemoParser preserves the original task instruction and action sequence while generating semantically descriptive action descriptions. These descriptions provide crucial context about the purpose and significance of each action in the demonstration, enabling more effective knowledge transfer to new scenarios.

For intermediate actions, DemoParser analyzes a visual representation of the action execution, showing before-action and afteraction screenshots with the action visualized (e.g., click locations highlighted). The framework combines this visual input with the task instruction, action history, and current action to generate a description that follows a standardized format: " $\scriptstyle [ \mathrm { O n } / \mathrm { I n } ]$ [Screen

![](images/394d999372a1feec3833c4266a4484aed230cf9cc69a53e2cb5d976f1230b1cd.jpg)  
Figure 4: Illustration of the overall framework of LearnAct. Architecture diagram showing the three main components (DemoParser, KnowSeeker, ActExecutor) and their interconnections within the LearnAct system, including data flow from human demonstrations to execution.

Name], [Action Details], to [Purpose]". For example: "On Home Screen, tap ’Settings’ icon, to access device configuration." For terminal actions, DemoParser processes the final screenshot, task instruction, and complete action history to generate a conclusion in the format: " $\scriptstyle { \mathrm { [ O n / I n ] } }$ [Screen], complete task, [Reason/Answer]"

A distinctive feature of DemoParser is its memory mechanism, which captures critical information observed during task execution that may be necessary for future steps. The model identifies and annotates task-relevant information that is directly related to the user’s instruction, will likely be needed in subsequent steps, and has not been previously recorded. These memory annotations are included in the action descriptions when appropriate: " $\mathrm { ` [ O n / I n ] }$ [Screen], [Action], to [Purpose]. [Memory: important information for future steps]". For example, in a shopping task, a memory annotation might capture: "[Memory: iPhone 13 Pro costs $\$ 999$ with 128GB storage]". The detailed prompt for this memory mechanism is provided in Appendix B.1.

This memory mechanism is particularly valuable for complex tasks requiring information retention across multiple steps, such as comparing prices, remembering account details, or tracking status changes. By transforming raw demonstrations into structured, semantically descriptive knowledge with memory capabilities, DemoParser enables effective knowledge transfer from human demonstrations to automated task execution.

# 4.2 KnowSeeker

KnowSeeker is the retrieval component of the LearnAct framework that identifies demonstration knowledge most relevant to the current task context. As depicted in Figure 6, this agent serves as the bridge between the knowledge base generated by DemoParser and the execution environment of ActExecutor. While DemoParser focuses on transforming demonstrations into structured knowledge, KnowSeeker specializes in efficiently accessing and selecting the most applicable knowledge for a specific task, addressing the critical challenge of knowledge relevance in few-shot learning scenarios.

Formally, KnowSeeker implements a retrieval function $R : J \times$ $\mathcal { K } \to \mathcal { K } ^ { ( s ) }$ , where $\boldsymbol { \mathcal { T } }$ is the instruction space, $\mathcal { K }$ is the knowledge base, and $\mathcal { K } ^ { ( s ) } \subset \mathcal { K }$ is a subset of knowledge entries determined to be relevant for the given instruction. This retrieval process is crucial for effective knowledge utilization, as it filters the potentially vast knowledge base to focus exclusively on demonstrations that offer valuable insights for the current task.

The core of KnowSeeker’s retrieval mechanism relies on semantic similarity measurement between the current task instruction and the instructions associated with demonstrations in the knowledge base. This similarity-based retrieval can be formally defined as:

![](images/07d8f7b279dd3acace56c36cf16d13052aa3f5e9e0164000c67603477ca3ffb0.jpg)  
Figure 5: Pipeline of DemoParser Agent. Input instructions and corresponding actions and screenshots; output low-level action descriptions and create knowledge database. This process transforms high-level user instructions into precise operation sequences while building a reusable domain knowledge base to improve mobile interface interaction automation efficiency.

![](images/0834c419f7d3c4dc03500c9106e9f9401638ef03c58daf9aef624a444ab7d72b.jpg)  
Figure 6: Pipeline of KnowSeeker Agent. The KnowSeeker Agent converts demo trajectories from the knowledge base into a vector database. When executing user tasks, KnowSeeker retrieves the top-k relevant demos from the vector database for subsequent use. This approach enables efficient retrieval of similar demonstrations to assist with new task execution.

$$
R (i, K) = \left\{k _ {j} \in K \mid s i m (i, i _ {j}) \geq \tau_ {s} \right\} _ {j = 1} ^ {t o p - k} \tag {1}
$$

where ?? is the current instruction, $i _ { j }$ is the instruction associated with knowledge entry $k _ { j }$ , $s i m ( \cdot , \cdot )$ is a similarity function, $\tau _ { s }$ is a

similarity threshold, and ?????? − ?? indicates selection of the $k$ most similar entries.

To implement this similarity measurement efficiently, KnowSeeker employs a two-phase approach:

(1) Embedding Generation: Instructions are transformed into dense vector representations using a pre-trained sentence transformer model. Specifically, we utilize the all-MiniLM-L6-v2 model, which offers an optimal balance between computational efficiency and semantic representational power. This model has been fine-tuned on diverse natural language understanding tasks, making it particularly well-suited for capturing the semantic essence of mobile GUI task instructions.   
(2) Similarity Computation: The cosine similarity between embedding vectors is calculated to quantify the semantic relationship between instructions. For instructions ?? and $i _ { j }$ with corresponding embeddings $e _ { i }$ and $e _ { j }$ , the similarity is computed as:

$$
\operatorname {s i m} (i, i _ {j}) = \frac {e _ {i} \cdot e _ {j}}{\left| \left| e _ {i} \right| \right| \cdot \left| \left| e _ {j} \right| \right|} \tag {2}
$$

To optimize retrieval efficiency, KnowSeeker pre-computes embeddings for all instructions in the knowledge base during initialization. This approach transforms the potentially expensive operation of computing pairwise similarities during runtime into a more manageable vector comparison task. The pre-computation process is described as:

![](images/dfe85852fe161b886bac466eb9595c859eea561e544bdfc57ae16d29836b51bd.jpg)  
Figure 7: Pipeline of ActExecutor Agent. The ActExecutor Agent executes the low-level action descriptions generated by the Action Planner Agent. It uses the KnowSeeker Agent to retrieve relevant demonstrations from the knowledge base and execute the actions in the demonstrations. This approach enables efficient execution of low-level actions to assist with new task execution.

$$
E = \left\{e _ {j} = f _ {\text {e m b e d}} \left(i _ {j}\right) \mid k _ {j} \in K \right\} \tag {3}
$$

where $f _ { e m b e d }$ is the embedding function implemented by the sentence transformer model.

During task execution, when presented with a new instruction ??, KnowSeeker: 1. Computes the embedding $e _ { i } = f _ { e m b e d } ( i )$ 2. Calculates similarities $S = \{ s i m ( e _ { i } , e _ { j } ) \mid e _ { j } \in E \}$ 3. Selects the top- $\mathbf { \nabla } \cdot k$ knowledge entries based on similarity scores

This approach ensures that knowledge retrieval scales efficiently with the size of the knowledge base, enabling rapid identification of relevant demonstrations even as the framework’s experiential knowledge grows over time. By systematically identifying the most relevant demonstration knowledge, KnowSeeker enables ActExecutor to perform tasks more effectively, particularly in unfamiliar scenarios.

# 4.3 ActExecutor

ActExecutor is the execution component of the LearnAct framework that translates retrieved demonstration knowledge into effective actions in the target environment. As illustrated in Figure 7, this agent represents the culmination of the LearnAct pipeline, integrating user instructions, real-time GUI observations, and demonstration knowledge to navigate even unfamiliar mobile applications successfully. While DemoParser creates structured knowledge and KnowSeeker retrieves relevant demonstrations, ActExecutor applies this knowledge to solve practical tasks, addressing the critical challenge of knowledge utilization in few-shot learning scenarios.

ActExecutor implements the POMDP framework introduced earlier, with the critical enhancement of incorporating demonstration knowledge into the decision-making process. The execution process can be formally described as a sequential decision-making loop that iteratively selects actions $a _ { t } \in \mathcal { A }$ based on current observations $o _ { t } \in O$ and demonstration knowledge $\mathcal { D }$ , following policy $\pi : O \times \mathcal { D }  \mathcal { A }$ .

The ActExecutor policy $\pi$ is implemented through a large visionlanguage model that processes a carefully constructed prompt integrating all available information sources. This prompt-based policy can be expressed as:

$$
\pi \left(o _ {t}, \mathcal {D}\right) = f _ {L L M} \left(P \left(i, o _ {t}, h _ {t - 1}, \mathcal {D}\right)\right) \tag {4}
$$

where ?? is the user instruction, $o _ { t }$ is the current observation (screenshot), $h _ { t - 1 }$ is the action history up to time ?? − 1, $\mathcal { D }$ is the retrieved demonstration knowledge, $P$ is a prompt construction function, and $f _ { L L M }$ is the LLM-based decision function.

Algorithm 2 in Appendix B.3 outlines the execution process. For each task, ActExecutor processes the user instruction and screenshot observations through a sequence of perception, decision, and action phases until the task is completed or a maximum step limit is reached.

The execution process integrates three key phases:

(1) Perception Phase: ActExecutor perceives the current state of the mobile device through screenshot observations $o _ { t }$ . These observations provide the visual context essential for understanding the available interaction options and current application state.   
(2) Decision Phase: The agent constructs a comprehensive prompt that integrates the user instruction ??, current observation $o _ { t }$ , action history $h$ , and retrieved demonstrations $\mathcal { D }$ . This prompt is processed by a large vision-language model using templates detailed in Appendix B.2, resulting in a selected action from the predefined action space described in Table 2.   
(3) Action Phase: The selected action $a _ { t }$ is executed in the mobile environment, generating a state transition according to the transition function $\mathcal { T }$ of the POMDP. Additionally, the agent generates a description $d _ { t }$ of the executed action using a process similar to DemoParser’s description generation, which serves as part of the action history for subsequent steps.

The prompt construction function $P$ plays a critical role in ActExecutor’s effectiveness. It integrates the agent’s role definition, demonstration examples, task and observation context, action history, and the action space definition into a comprehensive prompt that guides the model’s decision-making.

This approach enables ActExecutor to leverage demonstrations as exemplars that guide its decision-making process. When faced with a novel UI state, the agent identifies analogous situations from demonstrations and adapts the demonstrated actions to the current context. This capability is particularly valuable for handling out-ofdistribution scenarios where the agent lacks direct experience.

By closing the loop between demonstration knowledge and task execution, ActExecutor completes the LearnAct framework’s endto-end pipeline for demonstration-based learning. The combination of knowledge generation (DemoParser), knowledge retrieval (KnowSeeker), and knowledge-guided execution (ActExecutor) enables effective few-shot learning for mobile GUI agents, addressing the fundamental challenge of generalization to unseen scenarios with minimal examples.

# 5 EXPERIMENTS

We conducted comprehensive evaluations of the LearnAct framework through both offline and online experiments. The offline experiments were performed on the LearnGUI-Offline dataset to evaluate step-by-step task execution capabilities, while the online experiments utilized the LearnGUI-Online platform to assess end-to-end task completion in real-world interactive scenarios. We evaluated a diverse set of models, including both commercial (e.g., Gemini-1.5-Pro [26]) and open-source models (e.g., UI-TARS-7B-SFT [22], Qwen2-VL-7B [31]), to demonstrate the broad applicability of our approach across different model architectures and capabilities.

# 5.1 Experiment Setup

The diverse similarity profiles in LearnGUI provide a unique opportunity to evaluate mobile GUI agents’ capabilities. Our experiments have two primary goals: (1) to evaluate the feasibility and effectiveness of enhancing mobile agents through few-shot demonstrations as a means to overcome the limitations of traditional pre-training or fine-tuning approaches; and (2) to investigate how different factors such as demonstration quantity $_ { ( \mathrm { k } = 1 , 2 , 3 ) }$ ) and various similarity aspects (instruction, UI, and action) influence the effectiveness of demonstration-based learning.

Implementation Details. We conducted experiments with three foundation models: Gemini-1.5-Pro [26], UI-TARS-7B-SFT [22], and Qwen2-VL-7B [31]. For all models, we set the temperature to zero to obtain deterministic responses. For Qwen2-VL-7B [31] and UI-TARS-7B-SFT [22], we employed parameter-efficient fine-tuning using LoRA with rank 64, alpha 128, and dropout probability 0.1. We targeted all modules while freezing the vision encoder to ensure computational efficiency. Training used a learning rate of 1e-5 with cosine scheduling, batch size of 1, gradient accumulation over 8 steps, a warmup ratio of 0.001, and was conducted for 1 epoch. All fine-tuning experiments were conducted on 8 NVIDIA L40S GPUs. For offline experiments, Gemini-1.5-Pro [26] was evaluated directly on the LearnGUI-Offline test set without additional training. UI-TARS-7B-SFT [22] and Qwen2-VL-7B [31] were fine-tuned on the LearnGUI-Offline training set before evaluation. For online experiments, we deployed all models except Gemini-1.5-Pro [26] (which showed limited task completion capabilities in preliminary tests despite accuracy improvements) to the LearnGUI-Online environment, using 1-shot demonstration retrieval for all LearnActenhanced models.

Baselines. To rigorously evaluate our approach, we compared LearnAct against several baselines. These include: (1) SPHINX-GUI Agent, the original agent developed for the AMEX dataset [5], providing a reference point for task execution on similar data; (2) Zero-shot inference versions of all models (Gemini-1.5-Pro [26], UI-TARS-7B-SFT [22], and Qwen2-VL-7B [31]) within the Learn-Act framework but without demonstration knowledge, maintaining identical execution environments for fair comparison; and (3) For online evaluation, we additionally compared against GPT-4o, Gemini-Pro-1.5, Claude Computer-Use, and Aguvis to benchmark against current advanced systems.

Evaluation Metrics. For offline evaluation, we adopted mainstream evaluation protocols widely used in recent mobile GUI agent research, such as UI-TARS [22] and OS-ATLAS [39]. Specifically, we

Table 4: Performance comparison of mobile GUI agents on LearnGUI-Offline dataset (action match accuracy $\%$ ). Results show absolute values and relative improvements [in brackets] compared to baselines. Performance is evaluated across different models and number of support examples (1/2/3-shot).   

<table><tr><td>Models</td><td>Method</td><td>Supports</td><td>Average</td><td>Gmail</td><td>Booking</td><td>Music</td><td>SHEIN</td><td>NBC</td><td>CityMapper</td><td>ToDo</td><td>Signal</td><td>Yelp</td></tr><tr><td>SPHINX-GUI Agent[5]</td><td>AMEX</td><td>0-shot</td><td>67.2</td><td>45.9</td><td>64.5</td><td>74.4</td><td>71.8</td><td>70.3</td><td>67.4</td><td>79.3</td><td>64.9</td><td>66.3</td></tr><tr><td rowspan="4">gemini-1.5-pro</td><td rowspan="2">Baseline</td><td>0-shot</td><td>19.3</td><td>20.1</td><td>16.4</td><td>24.5</td><td>10.2</td><td>35.6</td><td>14.1</td><td>17.4</td><td>27.9</td><td>15.2</td></tr><tr><td>1-shot</td><td>51.7 [+32.4]</td><td>55.5</td><td>47.1</td><td>60.0</td><td>35.7</td><td>56.4</td><td>54.7</td><td>60.6</td><td>63.1</td><td>54.6</td></tr><tr><td rowspan="2">LearnAct</td><td>2-shot</td><td>55.6 [+36.3]</td><td>57.5</td><td>53.2</td><td>55.3</td><td>39.6</td><td>56.1</td><td>58.2</td><td>68.1</td><td>69.7</td><td>60.0</td></tr><tr><td>3-shot</td><td>57.7 [+38.4]</td><td>58.4</td><td>56.6</td><td>54.6</td><td>43.9</td><td>53.9</td><td>69.4</td><td>69.2</td><td>70.5</td><td>57.6</td></tr><tr><td rowspan="4">UI-TARS-7B-SFT</td><td rowspan="2">Baseline</td><td>0-shot</td><td>77.5</td><td>68.1</td><td>81.0</td><td>81.1</td><td>72.9</td><td>80.9</td><td>70.6</td><td>66.0</td><td>92.6</td><td>82.4</td></tr><tr><td>1-shot</td><td>82.8 [+5.3]</td><td>79.9</td><td>82.9</td><td>86.6</td><td>75.7</td><td>86.3</td><td>79.4</td><td>84.0</td><td>89.3</td><td>83.0</td></tr><tr><td rowspan="2">LearnAct</td><td>2-shot</td><td>81.9 [+4.4]</td><td>80.1</td><td>80.7</td><td>86.2</td><td>76.1</td><td>87.2</td><td>80.0</td><td>83.7</td><td>84.4</td><td>84.2</td></tr><tr><td>3-shot</td><td>82.1 [+4.6]</td><td>79.9</td><td>80.9</td><td>86.2</td><td>75.7</td><td>86.9</td><td>81.2</td><td>85.8</td><td>84.4</td><td>84.2</td></tr><tr><td rowspan="4">Qwen2-VL-7B</td><td rowspan="2">Baseline</td><td>0-shot</td><td>71.8</td><td>60.8</td><td>73.9</td><td>76.0</td><td>65.5</td><td>75.5</td><td>62.9</td><td>78.7</td><td>82.8</td><td>69.1</td></tr><tr><td>1-shot</td><td>77.3 [+5.5]</td><td>75.0</td><td>77.5</td><td>77.8</td><td>69.8</td><td>83.5</td><td>72.9</td><td>78.0</td><td>83.6</td><td>78.8</td></tr><tr><td rowspan="2">LearnAct</td><td>2-shot</td><td>78.5 [+6.7]</td><td>75.0</td><td>78.0</td><td>77.8</td><td>73.3</td><td>86.0</td><td>73.5</td><td>81.9</td><td>87.7</td><td>77.6</td></tr><tr><td>3-shot</td><td>79.4 [+7.6]</td><td>75.0</td><td>78.8</td><td>78.6</td><td>72.6</td><td>87.8</td><td>77.1</td><td>82.6</td><td>87.7</td><td>80.6</td></tr></table>

measured step accuracy, which consists of two components: action type accuracy and action match accuracy. Action type accuracy measures the percentage of steps where the predicted action type (CLICK, TYPE, SWIPE, etc.) matches the ground truth. Action match accuracy measures the percentage of steps where both the action type and its parameters are correct, following standard evaluation criteria. For CLICK actions, coordinates are considered correct if they fall within $1 4 \%$ of the screen width from the ground truth. For TYPE actions, the content is correct if the F1 score between prediction and ground truth exceeds 0.5. For SWIPE actions, the direction must precisely match the ground truth. For other actions (e.g., PRESS_BACK), an exact match is required. For TASK_COMPLETE actions, we only verify the action type and ignore the answer field. For online evaluation, we measured the task success rate (SR), which represents the percentage of tasks completed successfully in the real-time interactive environment.

# 5.2 Main Results

5.2.1 Offline Agent Capability Evaluation. Table 4 presents the performance comparison of different models on the LearnGUI-Offline dataset. The results demonstrate the substantial improvements achieved by the LearnAct framework across all tested models. Gemini-1.5-Pro [26] shows the most dramatic improvement, with performance increasing from $1 9 . 3 \%$ to $5 1 . 7 \%$ $( + 3 2 . 4 \% )$ with just a single demonstration, and further improving to $5 7 . 7 \%$ $\left( + 3 8 . 4 \% \right)$ with three demonstrations. This represents a $1 9 8 . 9 \%$ relative improvement, highlighting the powerful potential of demonstrationbased learning even for advanced foundation models. UI-TARS-7B-SFT [22], despite already having strong zero-shot performance $( 7 7 . 5 \% )$ , still achieves significant gains with LearnAct, reaching $8 2 . 8 \%$ $( + 5 . 3 \% )$ with a single demonstration. This indicates that even models specifically fine-tuned for GUI tasks can benefit from demonstration knowledge. Qwen2-VL-7B [31] demonstrates consistent improvement from $7 1 . 8 \%$ to $7 7 . 3 \% \left( + 5 . 5 \% \right)$ with one demonstration, and to $7 9 . 4 \%$ $\left( + 7 . 6 \% \right)$ with three demonstrations, confirming that

the benefits of LearnAct generalize across models with different architectures and capabilities.

The results also reveal interesting patterns regarding the impact of demonstration quantity. For Gemini-1.5-Pro [26], performance scales monotonically with the number of demonstrations, suggesting that less specialized foundation models can benefit substantially from additional examples. In contrast, UI-TARS-7B-SFT [22] achieves its peak performance with just one demonstration, indicating that models already fine-tuned for GUI tasks may efficiently extract necessary information from minimal demonstrations.

Application-specific results highlight LearnAct’s consistent improvement across diverse scenarios, with particularly notable gains in complex applications like CityMapper (from $1 4 . 1 \%$ to $6 9 . 4 \%$ for Gemini-1.5-Pro [26]) and To-Do apps (from $1 7 . 4 \%$ to $6 9 . 2 \%$ ). This suggests that demonstration-based learning is especially valuable for navigating applications with complex interactions and nonstandard interfaces.

To further understand the factors influencing LearnAct’s effectiveness, we analyzed performance across different similarity profiles, as shown in Table 5. Several important insights emerge: Gemini-1.5-Pro [26] shows substantial improvements across all similarity combinations, with the largest gains in action match accuracy (ranging from $+ 2 9 . 3 \%$ to $+ 3 9 . 6 \%$ ). This indicates that demonstration knowledge significantly enhances the model’s ability to execute precise actions regardless of similarity conditions. UI-TARS-7B-SFT [22] exhibits the most pronounced improvements in UISHActSH scenarios $( + 1 3 . 9 \%$ with 3-shot), suggesting that the model can extract maximum value from demonstrations when both UI and action patterns are similar to the target task. Qwen2-VL-7B [31] shows notably large improvements in action type accuracy for 2-shot settings (e.g., $+ 6 7 . 4 \%$ for UISHActSH), potentially indicating a threshold effect where multiple demonstrations trigger significant pattern recognition improvements.

Interestingly, while UI similarity generally correlates with higher performance gains, we observe that action similarity also plays a

Table 5: Performance breakdown of LearnAct-Offline on different UI and action combinations. Performance metrics (type and match accuracy) across four similarity quadrants showing absolute values and relative improvements [in brackets] compared to baselines. Results are grouped by model and number of support examples (1/2/3-shot).   

<table><tr><td rowspan="2">Models</td><td rowspan="2">Supports</td><td colspan="2">\( \mathbf{U}\mathbf{I}_{\mathbf{S}\mathbf{H}}\mathbf{A}\mathbf{c}\mathbf{t}_{\mathbf{S}\mathbf{H}} \)</td><td colspan="2">\( \mathbf{U}\mathbf{I}_{\mathbf{S}\mathbf{H}}\mathbf{A}\mathbf{c}\mathbf{t}_{\mathbf{S}\mathbf{L}} \)</td><td colspan="2">\( \mathbf{U}\mathbf{I}_{\mathbf{S}\mathbf{L}}\mathbf{A}\mathbf{c}\mathbf{t}_{\mathbf{S}\mathbf{H}} \)</td><td colspan="2">\( \mathbf{U}\mathbf{I}_{\mathbf{S}\mathbf{L}}\mathbf{A}\mathbf{c}\mathbf{t}_{\mathbf{S}\mathbf{L}} \)</td></tr><tr><td>type</td><td>match</td><td>type</td><td>match</td><td>type</td><td>match</td><td>type</td><td>match</td></tr><tr><td rowspan="3">gemini-1.5-pro</td><td>1-shot</td><td>79.5 [+12.8]</td><td>50.2 [+35.6]</td><td>78.1 [+12.3]</td><td>47.8 [+33.2]</td><td>77.5 [+9.2]</td><td>52.3 [+30.5]</td><td>77.9 [+14.1]</td><td>44.2 [+29.3]</td></tr><tr><td>2-shot</td><td>77.7 [+13.0]</td><td>53.9 [+37.3]</td><td>73.2 [+10.8]</td><td>49.9 [+34.7]</td><td>80.0 [+9.0]</td><td>56.5 [+34.8]</td><td>77.2 [+12.9]</td><td>48.9 [+34.4]</td></tr><tr><td>3-shot</td><td>72.3 [+15.8]</td><td>53.5 [+39.6]</td><td>72.8 [+12.9]</td><td>49.5 [+34.6]</td><td>78.7 [+10.4]</td><td>60.0 [+38.4]</td><td>79.2 [+12.8]</td><td>51.6 [+36.3]</td></tr><tr><td rowspan="3">Qwen2-VL-7B</td><td>1-shot</td><td>86.0 [+5.3]</td><td>72.2 [+6.3]</td><td>85.4 [+4.9]</td><td>69.6 [+5.5]</td><td>86.0 [+2.0]</td><td>76.2 [+5.4]</td><td>82.9 [+1.3]</td><td>69.4 [+4.3]</td></tr><tr><td>2-shot</td><td>85.0 [+67.4]</td><td>75.6 [+9.3]</td><td>84.0 [+67.2]</td><td>71.2 [+5.7]</td><td>86.9 [+73.3]</td><td>76.8 [+6.3]</td><td>84.0 [+68.5]</td><td>70.5 [+5.5]</td></tr><tr><td>3-shot</td><td>80.2 [+5.0]</td><td>70.3 [+7.9]</td><td>82.9 [+4.7]</td><td>70.2 [+5.7]</td><td>85.6 [+1.9]</td><td>77.5 [+8.4]</td><td>85.6 [+3.4]</td><td>72.8 [+6.6]</td></tr><tr><td rowspan="3">UI-TARS-7B-SFT</td><td>1-shot</td><td>88.1 [+1.9]</td><td>77.8 [+6.6]</td><td>87.2 [+2.1]</td><td>75.3 [+6.4]</td><td>87.7 [+0.3]</td><td>80.1 [+5.9]</td><td>85.0 [-0.2]</td><td>75.0 [+2.8]</td></tr><tr><td>2-shot</td><td>85.5 [+2.1]</td><td>76.7 [+8.3]</td><td>85.7 [+1.6]</td><td>75.9 [+4.9]</td><td>87.3 [-0.4]</td><td>79.1 [+5.9]</td><td>84.9 [-0.8]</td><td>74.1 [+2.1]</td></tr><tr><td>3-shot</td><td>87.1 [+7.9]</td><td>78.2 [+13.9]</td><td>85.5 [+2.6]</td><td>75.4 [+4.9]</td><td>86.0 [-0.9]</td><td>78.9 [+6.8]</td><td>85.5 [-0.9]</td><td>75.2 [+2.7]</td></tr></table>

Table 6: Performance comparison of different models on the LearnGUI-Online benchmark. Comparison of models with different inputs (Image, Image $^ +$ AXTree) and parameters, measuring task success rate (LearnGUI-OnlineSR) with improvements shown in brackets for models with LearnAct enhancement.   

<table><tr><td>Input</td><td>Models</td><td># Params</td><td>LearnGUI-OnlinesR</td></tr><tr><td>Image + AXTree</td><td>GPT-40[12]</td><td>-</td><td>34.5</td></tr><tr><td>Image + AXTree</td><td>Gemini-Pro-1.5[26]</td><td>-</td><td>22.8</td></tr><tr><td>Image</td><td>Claude Computer-Use[2]</td><td>-</td><td>27.9</td></tr><tr><td>Image</td><td>Aguvis[41]</td><td>72B</td><td>26.1</td></tr><tr><td>Image</td><td>Qwen2-VL-7B + 0-shot</td><td>7B</td><td>9.9</td></tr><tr><td>Image</td><td>Qwen2-VL-7B + LearnAct</td><td>7B</td><td>21.1 [+11.2]</td></tr><tr><td>Image</td><td>UI-TARS-7B-SFT + 0-shot</td><td>7B</td><td>18.1</td></tr><tr><td>Image</td><td>UI-TARS-7B-SFT + LearnAct</td><td>7B</td><td>32.8 [+14.7]</td></tr></table>

crucial role. For instance, Gemini-1.5-Pro [26] achieves its highest match accuracy in UISLActSH scenarios $( + 3 8 . 4 \%$ with 3-shot), suggesting that action similarity can sometimes compensate for UI differences. This finding highlights the importance of considering both UI and action similarity when designing demonstration-based learning approaches for mobile GUI agents.

These results validate our hypothesized framework design, demonstrating that LearnAct successfully leverages demonstration similarity to enhance performance across varying conditions, with the most substantial benefits observed when demonstrations can provide both perceptual and procedural knowledge relevant to the target task.

5.2.2 Online Agent Capability Evaluation. While offline evaluations provide valuable insights into step-by-step execution capabilities, real-world deployment requires successful end-to-end task completion. Table 6 presents the results of our online evaluation on the LearnGUI-Online benchmark, which reveals several important findings. The LearnAct framework substantially improves performance for both evaluated models, with Qwen2-VL-7B [31] improving from $9 . 9 \%$ to $2 1 . 1 \%$ $( + 1 1 . 2 \% )$ and UI-TARS-7B-SFT [22] from $1 8 . 1 \%$ to 32.8% $\left( + 1 4 . 7 \% \right)$ . These significant gains demonstrate

that the benefits of demonstration-based learning translate effectively to real-world interactive scenarios. Qwen2-VL-7B [31] with LearnAct achieves $2 1 . 1 \%$ success rate, showing meaningful improvements over its baseline performance. This suggests that the quality and relevance of demonstrations are highly effective for enhancing model capabilities. UI-TARS-7B-SFT [22] with LearnAct achieves $3 2 . 8 \%$ success rate, approaching the performance of GPT-4o $( 3 4 . 5 \% )$ despite using a much smaller model. This indicates that demonstration-based learning can help bridge the gap between smaller specialized models and large foundation models. Detailed visualizations of these performance comparisons are provided in Appendix C.1.To provide concrete examples of how LearnAct performs in real-world scenarios, we present three detailed case studies in Appendix C.2.

The most striking finding is the effectiveness of our demonstrationbased learning approach. The LearnAct framework provides significant performance improvements through its demonstration mechanism, with gains of up to $1 4 . 7 \%$ in task success rate. This demonstrates the power of high-quality demonstrations for enhancing model performance, highlighting the importance of relevant examples over simply increasing model size.

These results confirm that the LearnAct framework provides a practical pathway to developing effective mobile GUI agents, making it particularly valuable for application-specific customization and personalization scenarios.

# 5.3 Ablation Study

To understand the contribution of each component in the LearnAct framework, we conducted ablation experiments on the LearnGUI-Offline dataset using Gemini-1.5-Pro [26]. As shown in Table 7, we systematically evaluated the impact of removing either the DemoParser or KnowSeeker component while keeping all other settings constant.

The results reveal several important insights. Both components are essential, as removing either component leads to substantial performance degradation compared to the full LearnAct framework. The complete framework achieves $5 1 . 7 \%$ accuracy, while removing DemoParser reduces performance to $4 0 . 6 \%$ (-11.1%) and removing

Table 7: Ablation study of LearnAct components. Performance comparison across four configurations: baseline (no components), DemoParser only, KnowSeeker only, and both components combined. Results are presented as overall average accuracy and per-application breakdown across nine applications.   

<table><tr><td colspan="2">Ablation Setting</td><td rowspan="2">Average</td><td rowspan="2">Gmail</td><td rowspan="2">Booking</td><td rowspan="2">Music</td><td rowspan="2">SHEIN</td><td rowspan="2">NBC</td><td rowspan="2">CityMapper</td><td rowspan="2">ToDo</td><td rowspan="2">Signal</td><td rowspan="2">Yelp</td></tr><tr><td>DemoParser</td><td>KnowSeeker</td></tr><tr><td colspan="2">Baseline</td><td>19.3</td><td>20.1</td><td>16.4</td><td>24.5</td><td>10.2</td><td>35.6</td><td>14.1</td><td>17.4</td><td>27.9</td><td>15.2</td></tr><tr><td></td><td>✓</td><td>40.6</td><td>47.7</td><td>31.3</td><td>55.4</td><td>29.1</td><td>47.0</td><td>43.0</td><td>58.2</td><td>48.8</td><td>50.7</td></tr><tr><td>✓</td><td></td><td>41.6</td><td>46.9</td><td>34.1</td><td>52.7</td><td>27.9</td><td>51.9</td><td>45.3</td><td>51.4</td><td>61.1</td><td>51.8</td></tr><tr><td>✓</td><td>✓</td><td>51.7</td><td>55.5</td><td>47.1</td><td>60.0</td><td>35.7</td><td>56.4</td><td>54.7</td><td>60.6</td><td>63.1</td><td>54.6</td></tr></table>

KnowSeeker reduces it to $4 1 . 6 \% \left( - 1 0 . 1 \% \right)$ . Regarding DemoParser’s contribution, comparing "KnowSeeker only" $( 4 0 . 6 \% )$ to the baseline $( 1 9 . 3 \% )$ , we observe that even without action descriptions, relevant demonstrations improve performance by $2 1 . 3 \%$ . However, the addition of DemoParser’s action descriptions further enhances performance by $1 1 . 1 \%$ , confirming the value of structured knowledge extraction. For KnowSeeker’s contribution, the "DemoParser only" configuration $( 4 1 . 6 \% )$ also substantially outperforms the baseline, indicating that detailed action descriptions are valuable even with randomly selected demonstrations. However, KnowSeeker’s retrieval of relevant demonstrations provides an additional $1 0 . 1 \%$ improvement, highlighting the importance of demonstration relevance.

The performance variations across applications are particularly informative. For instance, in the Signal application, DemoParser appears more important $6 1 . 1 \%$ vs. $4 8 . 8 \%$ for KnowSeeker only), suggesting that detailed action descriptions are crucial for applications with complex interaction patterns. Conversely, for the ToDo application, KnowSeeker seems more valuable $5 8 . 2 \%$ vs. $5 1 . 4 \%$ for DemoParser only), indicating that demonstration relevance may be more critical for applications with varied task types.

These findings validate our multi-agent framework design, confirming that both knowledge extraction (DemoParser) and relevant demonstration retrieval (KnowSeeker) play complementary and essential roles in enabling effective demonstration-based learning for mobile GUI agents.

# 6 DISCUSSION AND FUTURE WORK

Our experimental results demonstrate that demonstration-based learning significantly enhances mobile GUI agents’ capabilities. The substantial performance improvements across all evaluated models validate our core hypothesis that demonstration-based learning effectively addresses generalization challenges. Even advanced foundation models like Gemini-1.5-Pro [26] show dramatic improvements ( $1 9 8 . 9 \%$ relative improvement). Our multi-dimensional similarity analysis reveals that both UI similarity and action similarity influence learning efficacy, with action similarity sometimes compensating for UI differences.

Data Collection and Dataset Expansion. While our approach shows promising results, several limitations and future directions warrant consideration. First, regarding data collection, our current dataset, while comprehensive, could benefit from greater diversity and representativeness. The LearnGUI dataset, comprising 2,252

offline tasks and 101 online tasks, represents a significant step forward but remains limited in scale compared to the vast diversity of mobile applications and user interactions. Future work should expand the dataset to include a broader range of applications, particularly those with complex interaction patterns and specialized domains.

K-shot Learning Analysis. Second, our current investigation of k-shot learning is limited to $\mathrm { k } { = } 1$ , 2, and 3 demonstrations. While these configurations provide valuable insights, a more comprehensive analysis of how demonstration quantity affects performance would be beneficial. Future research could explore the relationship between the number of demonstrations and performance gains, potentially identifying optimal demonstration counts for different scenarios and model architectures.

Enhanced Learning and Execution Strategies. Third, our learning and execution strategies could be enhanced to better leverage the relationship between support tasks and query tasks. While our current approach effectively retrieves relevant demonstrations, more sophisticated methods could be developed to extract and transfer knowledge more efficiently. For instance, techniques for abstracting common patterns across demonstrations, identifying critical decision points, and adapting demonstrated strategies to novel scenarios could further improve performance.

Agent Self-Learning. A promising direction for future research is to enable agents to learn from their own successful executions. Currently, our framework relies exclusively on human demonstrations, but agents could potentially learn from their own successful task completions. By incorporating these successful agent executions into the knowledge base, we could enable a form of "selflearning" where agents continuously improve their capabilities through their own experiences.

By addressing these limitations and pursuing these research directions, demonstration-based learning can evolve into a robust paradigm for developing adaptable, personalized, and practically deployable mobile GUI agents that effectively address the diverse needs of real-world users. The insights gained from our multidimensional similarity analysis provide valuable guidance for future research in this domain, suggesting that both UI similarity and action similarity play crucial roles in successful knowledge transfer.

# 7 CONCLUSION

This paper introduces a novel demonstration-based learning paradigm that fundamentally addresses the generalization challenges

faced by mobile GUI agents. Rather than pursuing universal coverage through ever-larger datasets, our approach leverages human demonstrations to enhance agent performance in unseen scenarios. We developed LearnGUI, the first comprehensive dataset for studying demonstration-based learning in mobile GUI agents, comprising 2,252 offline tasks and 101 online tasks with high-quality human demonstrations. We further designed LearnAct, a sophisticated multi-agent framework with three specialized components: DemoParser for knowledge extraction, KnowSeeker for relevant knowledge retrieval, and ActExecutor for demonstration-enhanced task execution. Our experimental results demonstrate remarkable performance gains, with a single demonstration increasing Gemini-1.5-Pro [26]’s accuracy from $1 9 . 3 \%$ to $5 1 . 7 \%$ in offline tests and enhancing UI-TARS-7B-SFT [22]’s online task success rate from $1 8 . 1 \%$ to $3 2 . 8 \%$ . These findings establish demonstration-based learning as a promising direction for developing more adaptable, personalized, and practically deployable mobile GUI agents.

# REFERENCES

[1] Simone Agostinelli, Andrea Marrella, and Massimo Mecella. 2019. Research challenges for intelligent robotic process automation. In Business Process Management Workshops: BPM 2019 International Workshops, Vienna, Austria, September 1–6, 2019, Revised Selected Papers 17. Springer, 12–18.   
[2] Anthropic. 2024. Developing a computer use model. https://www.anthropic. com/news/developing-computer-use   
[3] Chongyang Bai, Xiaoxue Zang, Ying Xu, Srinivas Sunkara, Abhinav Rastogi, Jindong Chen, et al. 2021. Uibert: Learning generic multimodal representations for ui understanding. arXiv preprint arXiv:2107.13731 (2021).   
[4] Andrea Burns, Deniz Arsan, Sanjna Agrawal, Ranjitha Kumar, Kate Saenko, and Bryan A Plummer. 2021. Mobile app tasks with iterative feedback (motif): Addressing task feasibility in interactive visual environments. arXiv preprint arXiv:2104.08560 (2021).   
[5] Yuxiang Chai, Siyuan Huang, Yazhe Niu, Han Xiao, Liang Liu, Dingyu Zhang, Peng Gao, Shuai Ren, and Hongsheng Li. 2024. Amex: Android multi-annotation expo dataset for mobile gui agents. arXiv preprint arXiv:2407.17490 (2024).   
[6] Yuxiang Chai, Hanhao Li, Jiayu Zhang, Liang Liu, Guozhi Wang, Shuai Ren, Siyuan Huang, and Hongsheng Li. 2025. A3: Android Agent Arena for Mobile GUI Agents. arXiv preprint arXiv:2501.01149 (2025).   
[7] Wentong Chen, Junbo Cui, Jinyi Hu, Yujia Qin, Junjie Fang, Yue Zhao, Chongyi Wang, Jun Liu, Guirong Chen, Yupeng Huo, et al. 2024. GUICourse: From General Vision Language Models to Versatile GUI Agents. arXiv preprint arXiv:2406.11317 (2024).   
[8] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. 2022. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588 (2022).   
[9] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Yantao Li, Jianbing Zhang, and Zhiyong Wu. 2024. Seeclick: Harnessing gui grounding for advanced visual gui agents. arXiv preprint arXiv:2401.10935 (2024).   
[10] Tiago Guerreiro, Ricardo Gamboa, and Joaquim Jorge. 2008. Mnemonical body shortcuts: improving mobile interaction. In Proceedings of the 15th European conference on Cognitive ergonomics: the ergonomics of cool interaction. 1–8.   
[11] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. 2024. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14281–14290.   
[12] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).   
[13] Courtney Kennedy and Stephen E Everett. 2011. Use of cognitive shortcuts in landline and cell phone surveys. Public Opinion Quarterly 75, 2 (2011), 336–348.   
[14] Wei Li, William Bishop, Alice Li, Chris Rawles, Folawiyo Campbell-Ajala, Divya Tyamagundlu, and Oriana Riva. 2024. On the Effects of Data Scale on Computer Control Agents. arXiv preprint arXiv:2406.03679 (2024).   
[15] Yang Li, Jiacong He, Xin Zhou, Yuan Zhang, and Jason Baldridge. 2020. Mapping natural language instructions to mobile UI action sequences. arXiv preprint arXiv:2005.03776 (2020).   
[16] Kevin Qinghong Lin, Linjie Li, Difei Gao, Zhengyuan Yang, Shiwei Wu, Zechen Bai, Weixian Lei, Lijuan Wang, and Mike Zheng Shou. 2024. Showui: One vision language-action model for gui visual agent. arXiv preprint arXiv:2411.17465 (2024).

[17] William Liu, Liang Liu, Yaxuan Guo, Han Xiao, Weifeng Lin, Yuxiang Chai, Shuai Ren, Xiaoyu Liang, Linghao Li, Wenhao Wang, et al. 2025. Llm-powered gui agents in phone automation: Surveying progress and prospects. (2025).   
[18] Zhe Liu, Cheng Li, Chunyang Chen, Junjie Wang, Boyu Wu, Yawen Wang, Jun Hu, and Qing Wang. 2024. Vision-driven Automated Mobile GUI Testing via Multimodal Large Language Model. arXiv preprint arXiv:2407.03037 (2024).   
[19] Quanfeng Lu, Wenqi Shao, Zitao Liu, Fanqing Meng, Boxuan Li, Botong Chen, Siyuan Huang, Kaipeng Zhang, Yu Qiao, and Ping Luo. 2024. GUI Odyssey: A Comprehensive Dataset for Cross-App GUI Navigation on Mobile Devices. arXiv preprint arXiv:2406.08451 (2024).   
[20] Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. 2024. Omniparser for pure vision based gui agent. arXiv preprint arXiv:2408.00203 (2024).   
[21] Pawel Pawlowski, Krystian Zawistowski, Wojciech Lapacz, Marcin Skorupa, Adam Wiacek, Sebastien Postansque, and Jakub Hoscilowicz. 2024. TinyClick: Single-Turn Agent for Empowering GUI Automation. arXiv preprint arXiv:2410.11871 (2024).   
[22] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. 2025. UI-TARS: Pioneering Automated GUI Interaction with Native Agents. arXiv preprint arXiv:2501.12326 (2025).   
[23] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. 2024. AndroidWorld: A dynamic benchmarking environment for autonomous agents. arXiv preprint arXiv:2405.14573 (2024).   
[24] Christopher Rawles, Alice Li, Daniel Rodriguez, Oriana Riva, and Timothy Lillicrap. 2024. Androidinthewild: A large-scale dataset for android device control. Advances in Neural Information Processing Systems 36 (2024).   
[25] Yunpeng Song, Yiheng Bian, Yongtao Tang, and Zhongmin Cai. 2023. Navigating Interfaces with AI for Enhanced User Interaction. arXiv preprint arXiv:2312.11190 (2023).   
[26] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530 (2024).   
[27] Sagar Gubbi Venkatesh, Partha Talukdar, and Srini Narayanan. 2022. Ugif: Ui grounded instruction following. arXiv preprint arXiv:2211.07615 (2022).   
[28] Junyang Wang, Haiyang Xu, Haitao Jia, Xi Zhang, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. 2024. Mobile-Agent-v2: Mobile Device Operation Assistant with Effective Navigation via Multi-Agent Collaboration. arXiv preprint arXiv:2406.01014 (2024).   
[29] Junyang Wang, Haiyang Xu, Jiabo Ye, Ming Yan, Weizhou Shen, Ji Zhang, Fei Huang, and Jitao Sang. 2024. Mobile-agent: Autonomous multi-modal mobile device agent with visual perception. arXiv preprint arXiv:2401.16158 (2024).   
[30] Luyuan Wang, Yongyu Deng, Yiwei Zha, Guodong Mao, Qinmin Wang, Tianchen Min, Wei Chen, and Shoufa Chen. 2024. MobileAgentBench: An Efficient and User-Friendly Benchmark for Mobile LLM Agents. arXiv preprint arXiv:2406.08184 (2024).   
[31] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. 2024. Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution. arXiv preprint arXiv:2409.12191 (2024).   
[32] Shuai Wang, Weiwen Liu, Jingxuan Chen, Weinan Gan, Xingshan Zeng, Shuai Yu, Xinlong Hao, Kun Shao, Yasheng Wang, and Ruiming Tang. 2024. GUI Agents with Foundation Models: A Comprehensive Survey. arXiv preprint arXiv:2411.04890 (2024).   
[33] Wenhao Wang, Zijie Yu, William Liu, Rui Ye, Tian Jin, Siheng Chen, and Yanfeng Wang. 2025. FedMobileAgent: Training Mobile Agents Using Decentralized Self-Sourced Data from Diverse Users. arXiv preprint arXiv:2502.02982 (2025).   
[34] Zhenhailong Wang, Haiyang Xu, Junyang Wang, Xi Zhang, Ming Yan, Ji Zhang, Fei Huang, and Heng Ji. 2025. Mobile-Agent-E: Self-Evolving Mobile Assistant for Complex Tasks. arXiv preprint arXiv:2501.11733 (2025).   
[35] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35 (2022), 24824–24837.   
[36] Hao Wen, Yuanchun Li, Guohong Liu, Shanhui Zhao, Tao Yu, Toby Jia-Jun Li, Shiqi Jiang, Yunhao Liu, Yaqin Zhang, and Yunxin Liu. 2024. Autodroid: Llm-powered task automation in android. In Proceedings of the 30th Annual International Conference on Mobile Computing and Networking. 543–557.   
[37] Hao Wen, Hongming Wang, Jiaxuan Liu, and Yuanchun Li. 2023. Droidbot-gpt: Gpt-powered ui automation for android. arXiv preprint arXiv:2304.07061 (2023).   
[38] Biao Wu, Yanda Li, Meng Fang, Zirui Song, Zhiwei Zhang, Yunchao Wei, and Ling Chen. 2024. Foundations and recent trends in multimodal mobile agents: A survey. arXiv preprint arXiv:2411.02006 (2024).   
[39] Zhiyong Wu, Zhenyu Wu, Fangzhi Xu, Yian Wang, Qiushi Sun, Chengyou Jia, Kanzhi Cheng, Zichen Ding, Liheng Chen, Paul Pu Liang, et al. 2024. Os-atlas: A foundation action model for generalist gui agents. arXiv preprint arXiv:2410.23218

(2024).   
[40] Yifan Xu, Xiao Liu, Xueqiao Sun, Siyi Cheng, Hao Yu, Hanyu Lai, Shudan Zhang, Dan Zhang, Jie Tang, and Yuxiao Dong. 2024. AndroidLab: Training and Systematic Benchmarking of Android Autonomous Agents. arXiv preprint arXiv:2410.24024 (2024).   
[41] Yiheng Xu, Zekun Wang, Junli Wang, Dunjie Lu, Tianbao Xie, Amrita Saha, Doyen Sahoo, Tao Yu, and Caiming Xiong. 2024. Aguvis: Unified Pure Vision Agents for Autonomous GUI Interaction. arXiv preprint arXiv:2412.04454 (2024).   
[42] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2024. Tree of thoughts: Deliberate problem solving with large language models. Advances in Neural Information Processing Systems 36 (2024).   
[43] Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Qingwei Lin, Saravan Rajmohan, et al. 2024. Large Language Model-Brained GUI Agents: A Survey. arXiv preprint arXiv:2411.18279 (2024).   
[44] Chi Zhang, Zhao Yang, Jiaxuan Liu, Yucheng Han, Xin Chen, Zebiao Huang, Bin Fu, and Gang Yu. 2023. Appagent: Multimodal agents as smartphone users. arXiv preprint arXiv:2312.13771 (2023).   
[45] Jiwen Zhang, Jihao Wu, Yihua Teng, Minghui Liao, Nuo Xu, Xiao Xiao, Zhongyu Wei, and Duyu Tang. 2024. Android in the zoo: Chain-of-action-thought for gui agents. arXiv preprint arXiv:2403.02713 (2024).   
[46] Jiayi Zhang, Chuang Zhao, Yihan Zhao, Zhaoyang Yu, Ming He, and Jianping Fan. 2024. MobileExperts: A Dynamic Tool-Enabled Agent Team in Mobile Devices. arXiv preprint arXiv:2407.03913 (2024).   
[47] Li Zhang, Shihe Wang, Xianqing Jia, Zhihan Zheng, Yunhe Yan, Longxi Gao, Yuanchun Li, and Mengwei Xu. 2024. LlamaTouch: A Faithful and Scalable Testbed for Mobile UI Automation Task Evaluation. arXiv preprint arXiv:2404.16054 (2024).   
[48] Zhuosheng Zhang and Aston Zhang. 2023. You only look at screens: Multimodal chain-of-action agents. arXiv preprint arXiv:2309.11436 (2023).

# A ADDITIONAL LEARNGUI STATISTICS

Figure 8 illustrates the distribution of similarity scores across different dimensions in the LearnGUI-Offline dataset, enabling systematic analysis of how different types of similarity between demonstration and query tasks affect learning efficacy.

![](images/d98b0c93e6a5d0b10d82c5610c5e6985f4fd419e32d111f4b40a00cefea0c7ff.jpg)  
UI Similarity Distribution

![](images/1bb99a7da032ec726dd69e97f5877d685f089be7f7bcafcadae828afec67ec3d.jpg)  
Action Similarity Distribution

![](images/ec4cd9c027b83928c5d522985ed72291b41fa803a15f69087146d7b2d0d19f2a.jpg)  
Instruction Similarity Distribution   
Figure 8: Distribution of instruction, UI, and action similarity scores in LearnGUI-Offline. The histograms show the distribution of similarity scores across three dimensions: instruction similarity (top), UI similarity (middle), and action similarity (bottom). These distributions enable systematic analysis of how different types of similarity between demonstration and query tasks affect learning efficacy.

# B LEARNACT FRAMEWORK DETAILS

This section provides detailed descriptions of the components of our LearnAct framework, corresponding to the methods presented in Section 4 of the paper.

# B.1 DemoParser Prompts

We provide all of our prompt templates used in DemoParser for generating semantically descriptive action descriptions from demonstration data. These carefully designed prompts guide the vision-language model to produce structured knowledge that captures the essence of human demonstrations, as shown in Figures 9 and 10.

# Prompt 1: Intermediate Action Description

# System Prompt:

You are a mobile UI interaction analyst. Follow these rules: 1. Analyze the split-screen image (Before-action left, After-action right) 2. For click actions, a high-contrast red marker (white-bordered circle) shows the precise click location, with a green square surrounding it and a ’C’ label at the top-right corner of the square indicating the click. 3. Output JSON with ONLY ONE ’action_description’ field in this exact format: "[On/In] [Screen Name], [Action Details], to [Purpose]"

Action Types: - click [element] (e.g., ’Search button’) - swipe [up/down/left/right] - type [text] in [field] - press [back/home/enter]

Validation Rules: 1. Screen names should be 2-6 words 2. Keep purpose descriptions under 8 words 3.

Never mention coordinates/IDs

MEMORY RECORDING RULES: If the current screen contains information relevant to the user’s instruction that needs to be remembered for future steps, include a Memory part in your action description. The format should be: $" [ \mathrm { O n } / \mathrm { I n } ]$ [Screen Name], [Action Details], to [Purpose]. [Memory: important information for future steps]"

Memory should ONLY be added when: 1. The information is relevant to completing the user’s instruction 2. The information will likely be needed in future steps 3. This specific information has NOT been recorded in previous action history entries

Memory examples: 1. For a travel planning task: On Travel Blog, click ’Bali Beach Guide’, to read article. [Memory: Guide mentions Kuta Beach has surfing lessons for $\$ 25,$ /hour] 2. For a shopping task: In Product Details, click ’Add to Cart’, to select item. [Memory: iPhone 13 Pro costs $\$ 999$ with 128GB storage] 3. For a note-taking task: On Weather App, swipe down forecast, to view weekend. [Memory: Saturday will be rainy with $8 0 \%$ precipitation]

Avoid using Memory for: 1. Obvious UI changes that don’t contain task-relevant information 2. Information already captured in previous action steps 3. Generic observations not specific to the user’s task objective

Figure 9: Prompt template for intermediate action descriptions. The template guides DemoParser to generate standardized descriptions for intermediate actions, including detailed rules for memory annotations that capture important information observed during task execution.

# B.2 ActExecutor Prompts

We provide the prompt templates used by ActExecutor to make decisions based on current observations, action history, and demonstration knowledge. These prompts guide the vision-language model to select appropriate actions for task execution, as shown in Figure 11.

# B.3 Algorithm Details

We provide the detailed algorithms for the DemoParser and ActExecutor components of our LearnAct framework, which are the core computational processes enabling knowledge extraction and task execution.

# C ADDITIONAL EXPERIMENTAL RESULTS AND ANALYSES

This section provides additional experimental results and analyses that supplement the findings presented in Section 5 of the paper.

# C.1 Online Performance Comparisons

Figures 12 and 13 provide detailed comparisons of model performance with and without LearnAct enhancement in online evaluation scenarios.

# C.2 Case Studies of LearnAct Online Experiments

We present three detailed case studies from our online experiments to provide concrete examples of how LearnAct leverages demonstration knowledge to solve tasks in unseen mobile applications. These case studies highlight different scenarios where demonstration knowledge proves particularly beneficial for task execution.

# Prompt 2: Terminal Action Description - Standard Completion

# System Prompt for standard completion:

Determine the final task status. Output rules: 1. Use ONLY ONE ’action_description’ field 2. Format: "[On/In] [Screen], complete task, [Reason]"

Validation Rules: - Reason should be less than 10 words - Screen name must match previous context

Examples: 1. Basic completion: On Payment Screen, complete task, successfully submit order 2. Failure case: In Search Results, cannot complete task, no nearby Vivo mobile phone stores found

# Prompt 3: Terminal Action Description - With Answer

# System Prompt for completion with answer:

Determine the final task status with the given answer. Output rules: 1. Use ONLY ONE ’action_description’ field 2. Format: " $\mathrm { ' } [ \mathrm { O n } / \mathrm { I n } ]$ [Screen], complete task, the answer is [answer]"

Validation Rules: - Screen name must match previous context - Use the exact answer provided in the TASK_COMPLETE action

Examples: 1. Answer is a price: On Checkout Screen, complete task, the answer is " $" \$ 299.9 "$ . 2. Answer is a list: On Payment Options Screen, complete task, the answer is "google pay, check out with affirm, add credit/debit card".

Figure 10: Prompt templates for terminal action descriptions. The templates provide specific formats for both standard task completion and information retrieval tasks, ensuring consistent output structure across different task types.

Algorithm 1 DemoParser Knowledge Generation Process   
Require: Demonstration dataset $D = \{(i_k,s_k,a_k)^N\}$ where $i_k$ is instruction, $s_k$ is screenshot sequence, $a_k$ is action sequence  
Ensure: Knowledge base $K$ with semantically descriptive action descriptions  
1: $K \gets \emptyset$ ▷ Initialize empty knowledge base  
2: for each demonstration $(i,s,a)$ in $D$ do  
3: $d \gets \emptyset$ ▷ Initialize empty description sequence  
4: for $j = 1$ to $|a|$ do  
5: if $j < |a|$ then ▷ Intermediate action  
6: Create visualization of action $a_j$ with before-after screenshots from $s_j$ and $s_{j+1}$ 7: $h \gets$ Previous action descriptions $\{d_1,d_2,\ldots,d_{j-1}\}$ 8: $d_j \gets$ GenerateDescription $(i,a_j,\text{visualization},h)$ using prompt format detailed in Appendix B.1  
9: $d_j$ follows format: "[On/In] [Screen], [Action], to [Purpose]" with optional memory  
10: else ▷ Terminal action  
11: $h \gets$ Complete action history $\{d_1,d_2,\ldots,d_{|a|-1}\}$ 12: $d_{|a|} \gets$ GenerateFinalDescription $(i,s_{|a|},h,a_{|a|})$ using prompt detailed in Appendix B.1  
13: $d_{|a|}$ follows format: "[On/In] [Screen], complete task, [Reason/Answer]"  
14: end if  
15: Add $d_j$ to description sequence $d$ 16: end for  
17: Add $(i,a,d)$ to knowledge base $K$ 18: end for  
19: return $K$

# Prompt 4: Task Execution Prompt

# Role Definition:

You are a smartphone assistant to help users complete tasks by interacting with apps. I will give you a screenshot of the current phone screen.

Example Tasks: [Only when demonstrations are available]

Example 1: [Demonstration instruction] Steps taken in this example: Step-1: [Action] [Action Description] Step-2: [Action] [Action Description] ...

Background: This image is a phone screenshot. Its width is [width] pixels and its height is [height] pixels. The user’s instruction is: [instruction]

History operations: [Only when action history is available]

Before reaching this page, some operations have been completed. You need to refer to the completed operations to decide the next operation. These operations are as follow: Step-1: [Action] [Action Description] Step-2: [Action] [Action Description] ...

Response requirements: Now you need to combine all of the above to decide just one action on the current page. You must choose one of the actions below:

"SWIPE[UP]": Swipe the screen up. "SWIPE[DOWN]": Swipe the screen down. "SWIPE[LEFT]": Swipe the screen left. "SWIPE[RIGHT]": Swipe the screen right. "CLICK[x,y]": Click the screen at the coordinates (x, y). x is the pixel from left to right and y is the pixel from top to bottom "TYPE[text]": Type the given text in the current input field. "PRESS_BACK": Press the back button. "PRESS_HOME": Press the home button. "PRESS_ENTER": Press the enter button. "TASK_COMPLETE[answer]": Mark the task as complete. If the instruction requires answering a question, provide the answer inside the brackets. If no answer is needed, use empty brackets "TASK_COMPLETE[]".

Response Example: Your output should be a string and nothing else, containing only the action type you choose from the list above. For example: "SWIPE[UP]" "CLICK[156,2067]" "TYPE[Rome]" "PRESS_BACK" "PRESS_HOME" "PRESS_ENTER" "TASK_COMPLETE[1h30m]" "TASK_COMPLETE[]"

Figure 11: Task execution prompt template. This comprehensive prompt directs ActExecutor to generate actions based on current observations, action history, and retrieved demonstrations, with explicit formatting requirements to ensure consistent action outputs.

Algorithm 2 ActExecutor Task Execution Process   
Require: User instruction $i$ , Knowledge base $K$ , Maximum steps $T$ Ensure: Task execution trajectory  
1: $t \gets 0$ 2: $h \gets \emptyset$ 3: $\mathcal{D} \gets \text{KnowSeeker}(i, K)$ 4: while $t < T$ and not IsTaskComplete do  
5: $o_t \gets \text{GetObservation}$ 6: $P_t \gets \text{ConstructPrompt}(i, o_t, h, \mathcal{D})$ 7: $a_t \gets f_{LLM}(P_t)$ 8: $d_t \gets \text{GenerateDescription}(i, a_t, o_t, h)$ 9: $h \gets h \cup \{(a_t, d_t)\}$ 10: ExecuteAction $(a_t)$ 11: $t \gets t + 1$ 12: end while  
13: return $\{(a_0, d_0), (a_1, d_1), \ldots, (a_{t-1}, d_{t-1})\}$

![](images/303e532a165ff6754418576f4bb9c35762bb11add2360a0b51c4a4f3f462e24c.jpg)  
Qwen2-VL-7B Task Performance Comparison   
Figure 12: Detailed performance comparison of Qwen2-VL-7B with and without LearnAct on LearnGUI-Online. The figure shows the task success rates of Qwen2-VL-7B baseline versus Qwen2-VL-7B enhanced with LearnAct across different task dimensions in the LearnGUI-Online benchmark.

![](images/9b7082c39669263c5d89f40d332a507665123189a1ee402d42d0c9a2a5a2faef.jpg)  
Figure 13: Detailed performance comparison of UI-TARS-7B-SFT with and without LearnAct on LearnGUI-Online. The figure presents a comprehensive breakdown of task success rates for UI-TARS-7B-SFT baseline versus UI-TARS-7B-SFT enhanced with LearnAct across multiple task dimensions in the LearnGUI-Online benchmark.

![](images/952b1a369443b875f381e8472c19c5b23d3bda967e3631ced374d0b598bf07f4.jpg)  
Baseline Failed

<instruction> What quantity of acai berries do I need for the recipe 'Tacos' in the Joplin app? Express your answer in the format <amount> <unit> where both the amount and unit exactly match the format in the recipe.

![](images/576ebb20a2cc610cb83de5238283760849bafe8a96fa1d5d133678dd8a08c190.jpg)

![](images/8976c5419f0fc41a661cb81d83391ff6e5a6ad99bf906b5bc02295ec6d4e31c7.jpg)

![](images/e4ff414fa25d75e42a0ebad326aed95430d07abe09032ab1f7526e5b8c672ede.jpg)

![](images/34c28d69cf2fa113d9aa159c60baa08b2ce1c92d701aae3abc1cfecb2439af26.jpg)

![](images/3706b20873e53a4e942e6d6a289310ab2ac75d3d3e5f9a6fa65c9e5b5c4e8cf4.jpg)

![](images/971d8175b74d2f30636635b6eefa34e798ba1d36d991092b4fbaad806973dafa.jpg)  
Support Mandate Execution

<instruction> What quantity of almond flour do I need for the recipe 'Tacos' in the Joplin app? Express your answer in the format <amount> <unit> where both the amount and unit exactly match the format in the recipe.

![](images/96fe9d337232e6203f1f3cdf5e89e346d75a4327cc9fea2ed419bac3183eaef0.jpg)

![](images/d3e071609cd5ef0fa4856275c0a8e7ed05e9a91dce4bc565279bfe2677a1d94f.jpg)

![](images/7ee933775c8dda33450a6fe6b036c41ce89f18efa1a5a7b1cf39139ec13d3927.jpg)

![](images/6cc8e6146975d538b3b343232593035b301e5a9eff90f8238cae0499c029a4d5.jpg)

![](images/995eb41e7100fa0a3889553aaaafc82ec5f693dfd7566e5758a3ca91a826613f.jpg)  
LearnAct Successed

<instruction> What quantity of buckwheat groats do I need for the recipe 'Tacos' in the Joplin app? Express your answer in the format <amount> <unit> where both the amount and unit exactly match the format in the recipe.

![](images/34f0463b836fd9eb41c3ad2c05724d895a4006c9f9b27571cec0704e825af1da.jpg)

![](images/d51e0ddca6f8e4b334eae500dd8474ce706c9ebba68feb337fe0c9158ffa6e81.jpg)

![](images/6f19cae71d86273ec4b66ae850454a4a8b2cc008279caebbaa94806282faa9f7.jpg)

![](images/02ee3740afd3cc6bf4c52bcbd2ce469932cb6dfe3450dca26726a84cb1803a99.jpg)

![](images/f8281d306c06ae17359706ec29aa71f76f42ccc69b37dd9d869b0848c6c93bfa.jpg)

![](images/4520cc0be9fbf3cbff46e836e99670c10b1d450e2441a3383bc69f449ccd9290.jpg)  
Figure 14: UI-TARS-7B-SFT with LearnAct vs. Baseline in NotesRecipeIngredientCount Task. Task template: "What quantity of {ingredient} do I need for the recipe ’{title}’ in the Joplin app? Express your answer in the format <amount> <unit> without using abbreviations."

![](images/709c02ed07b03d0c50ee3094684868ff330b71e2cc87ecdfa5e655272ab627d1.jpg)

# Baseline Failed

<instruction> In Simple Calendar Pro, delete the calendar event on 2023-10-31 at 13h with the title 'Review session for Annual Report'

![](images/f3aa6f84b8d363c50ec6dde8aa9b2bf037fe32eed84a4bb2768ba71cb8ab2b42.jpg)

![](images/fad282af9c92a1c75910a3e99281dad7bc491dacfc41a0ec9da4fa66b1264618.jpg)

# Support Mandate Execution

<instruction> In Simple Calendar Pro, delete the calendar event on 2023-10-26 at 17h with the title 'Workshop on Annual Report'.

![](images/1c0a6b27224a2c51a56b3d4eeea2534f04977b26fc89b6a9d46c2c6299b74574.jpg)

![](images/efbf25593f1d8b2d5c5e0a29a00546e36cb561df64976e9dff2b6a39396845db.jpg)

# LearnAct Successed

<instruction> In Simple Calendar Pro, delete the calendar event on 2023-10-25 at 0h with the title 'Review session for Annual Report'.

![](images/b7774721a4167cb75c6c515d4079c6024dfb652337ae31ec191b3ca48aef3264.jpg)  
Figure 15: UI-TARS-7B-SFT with LearnAct vs. Baseline in SimpleCalendarDeleteOneEvent Task. Task template: "In Simple Calendar Pro, delete the calendar event on {year}-{month}-{day} at {hour}h with the title ’{event_title}’"

![](images/ab9a1bd3fa0556235ed183c2e537ef258ccc277454f4b657ecd5750ea089e8ec.jpg)

# Baseline Failed

<instruction> Delete the following expenses from pro expense: Undergarments, Event Tickets, Streaming Services.

![](images/6197bf66bf3ca2fe0996bf2fb306c0f0c4754fa421a38f6c2218f3e43d6a1164.jpg)

![](images/7df50172e0ff5689434e0b00a41f841ee898d676f151d4e25c4bfe3c6e83de16.jpg)

# Support Mandate Execution

<instruction> Delete the following expenses from pro expense: Mortgage, Concert Tickets, Home Insurance.

![](images/a0feff4a417c8e7238a00518a799bcb35eafdcc8e9844818aed0d3fe91e357bc.jpg)

![](images/eb88b8a74637df88b18ede51d86dfeee6bac79fc054a5e93bd668eb9c46b1dd8.jpg)

![](images/3e82b9e0f9857416bd3706786ef3015d1ebe869901329650690ef7386f7afbba.jpg)

# LearnAct Successed

<instruction> Delete the following expenses from pro expense: Museum Tickets, Rent Payment, Health Insurance.

![](images/99ac4ef47101b913c5a5e82af4ec0408e4c62aa47660dd8058ef90d11b6feabb.jpg)  
Figure 16: Qwen2-VL-7B with LearnAct vs. Baseline in ExpenseDeleteMultiple Task. Task template: "Delete the following expenses23 from arduia pro expense: {expenses}."
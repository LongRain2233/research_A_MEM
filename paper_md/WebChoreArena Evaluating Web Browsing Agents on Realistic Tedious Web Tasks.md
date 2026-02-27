# WebChoreArena: Evaluating Web Browsing Agents on Realistic Tedious Web Tasks

Atsuyuki Miyai Zaiying Zhao Kazuki Egashira Atsuki Sato Tatsumi Sunada Shota Onohara Hiromasa Yamanishi Mashiro Toyooka Kunato Nishina Ryoma Maeda Kiyoharu Aizawa Toshihiko Yamasaki

miyai@cvm.t.u-tokyo.ac.jp

The University of Tokyo

https://webchorearena.github.io/

![](images/566df2d91a3cca705ae2e489280f7db2a057d6187dfdaad299a07b1d81ff14a5.jpg)  
WebArena

![](images/609918a9b3bd7475fda4c3181e2de16d2d7000e8e680496d45c2161ba2293b1f.jpg)  
WebArena vs. WebChoreArena   
Figure 1: The WebChoreArena challenge. WebChoreArena extends WebArena by introducing more complex and labor-intensive tasks, pushing the boundaries of agent capabilities. This enhanced benchmark allows for a clearer evaluation of progress in advanced models and reveals that even powerful models such as Gemini 2.5 Pro still have significant room for improvement.

# Abstract

Powered by a large language model (LLM), a web browsing agent operates web browsers in a human-like manner and offers a highly transparent path toward automating a wide range of everyday tasks. As web agents become increasingly

Preprint.

capable and demonstrate proficiency in general browsing tasks, a critical question emerges: Can they go beyond general browsing to robustly handle tasks that are tedious and complex, or chores that humans often avoid doing themselves? In this paper, we introduce WebChoreArena, a new fully reproducible benchmark comprising 532 carefully curated tasks designed to extend the scope of WebArena beyond general browsing to more labor-intensive and tedious tasks. WebChore-Arena systematically integrates three key challenges: (i) Massive Memory tasks requiring accurate retrieval of large amounts of information in the observations, (ii) Calculation tasks demanding precise mathematical reasoning, and (iii) Long-Term Memory tasks necessitating long-term memory across multiple webpages. Built on top of the fully reproducible and widely adopted four WebArena simulation environments, WebChoreArena ensures strict reproducibility and enables fair, direct comparisons with the established WebArena benchmark, offering key insights into agent progress. Our experimental results demonstrate that as LLMs evolve, represented by GPT-4o, Claude 3.7 Sonnet, and Gemini 2.5 Pro, significant improvements in performance are observed on WebChoreArena. These findings suggest that WebChoreArena is well-suited to measure the advancement of state-ofthe-art LLMs with greater clarity. Nevertheless, the results also indicate that even with Gemini $2 . 5 \mathrm { P r o }$ , there remains substantial room for improvement compared to WebArena, highlighting the increased challenges posed by WebChoreArena.

# 1 Introduction

Graphical User Interfaces (GUIs) serve as the primary medium through which humans perform everyday tasks. In recent years, browsing agents have gained attention as a means of automating these tasks. These agents take inputs such as accessibility trees or screenshots and produce humanlike actions such as clicking and typing. While Application Programming Interfaces (APIs) and programming-based approaches enable programmatic interactions with software [26], browsing agents can directly manipulate UIs, making them applicable to a wide range of web pages where APIs are unavailable. Moreover, compared to other approaches, browsing agents offer greater transparency and are more amenable to human oversight [9]. As a result, various browsing agents have been developed, continuously pushing the limits of capabilities [21, 17, 12, 33, 19].

Among GUI agent benchmarks [36, 13, 9, 10, 32, 29], WebArena [36] has emerged as the de facto standard for evaluating web browsing agents due to its highly realistic tasks and reproducible environment. WebArena provides fully functional websites across four common domains: e-commerce platforms (OneStopShop), social forums for idea and opinion exchange (Reddit), collaborative software development (GitLab), and content management systems for online data creation and management (online store management). While several recent works have made efforts in exploration on real-world websites [10, 32, 34, 22], ensuring reproducibility remains important. WebArena addresses this need, with broad adoption from both academic and industrial communities [33, 27, 4, 7, 23, 16, 21, 17, 12].

However, WebArena has two notable limitations. First, most of its tasks focus on general web browsing. While such tasks were effective for evaluating agent performance in earlier stages, recent advances in large language models (LLMs) and web agents have made them insufficient for precisely evaluating the performance limits and capabilities of modern models. Second, we observed that some tasks in WebArena contain ambiguous instructions or annotation errors [16] (see Appendix A). Although these issues had little impact when agent performance was low, they have become more serious as agents improve and these noises limit the upper bound of performance that the benchmark can accurately capture. As LLMs and agents continue to advance, increasing task difficulty and eliminating errors in evaluation are essential for accurately assessing their capabilities for automating more tedious tasks.

To address these limitations, we introduce WebChoreArena, a substantial extension of the widely adopted WebArena benchmark. Fig. 1 shows the illustration of the tasks in WebChoreArena. WebChoreArena consists of 532 human-curated tasks across the four websites used in WebArena, designed to go beyond general browsing, targeting more tedious and complex scenarios. These tasks can be broadly categorized into four types: (i) Massive Memory: tasks that require accurate memorization of a large number of information in the observations, (ii) Calculation: tasks that involve performing mathematical reasoning based on memorized information, (iii) Long-Term Memory:

tasks that require long-term memory across multiple web pages, and (iv) Others: tasks that involve special operations specific to the structure or functionality of certain websites. This benchmark enables the systematic investigation of capabilities that have been relatively underexplored in prior work on browsing agents, such as memory utilization and memory-based calculation. Furthermore, by leveraging the identical simulation environment as WebArena, WebChoreArena enables rigorous comparisons of agent performance under increased task difficulty, yielding clearer insights into agent progress on more challenging tasks.

For our experiments, we evaluate three LLMs: GPT-4o [11], a representative LLM used in prior academic research [14, 7], and two recent high-capacity LLMs, Claude 3.7 Sonnet [2] and Gemini 2.5 Pro [1]. We tested these LLMs with two state-of-the-art web agents, AgentOccam [33] and BrowserGym [6]. Our key findings are summarized as follows:

• GPT-4o, a representative LLM used in prior academic research, achieved no more than $6 . 8 \%$ accuracy on WebChoreArena, despite reaching $4 2 . 8 \%$ on WebArena.   
• As LLMs become more advanced, their performance on WebChoreArena improves. However, even the latest model, Gemini 2.5 Pro, achieved only $4 4 . 9 \%$ , showing there is still significant room for improvement compared to its performance on WebArena.   
• Since WebChoreArena more clearly demonstrates performance differences between LLMs than WebArena, it serves as a more accurate benchmark for evaluating the performance of increasingly powerful agents based on the advanced LLMs.

# 2 Related Work

Benchmarks for Web Agent. Early benchmarks for web agents were primarily built on synthetic web environments [25, 15]. To move closer to real-world interactions, Mind2Web [8] introduced a dataset of 2,000 web interactions from 137 websites, but it remains a static collection and does not capture interactive environments [20]. Recently, interactive web-based benchmarks have gained attention [36, 13, 10, 9, 31, 24, 32]. Among them, WebArena stands out for its reproducible simulation of real-world websites. Its high fidelity to real-world interactions and strong reproducibility have attracted substantial community efforts. In this work, we extend WebArena with more complex and labor-intensive tasks, aiming to better measure agent progress in challenging tasks.

Memory-intensive Benchmarks for Web Agent. Memory-intensive benchmarks have gained increasing attention in recent years, leading to the development of various benchmarks [18, 37, 5, 30, 35, 3, 34]. Prominent examples for web browsing agents include GAIA [18], WebWalker [30], and MMInA [35]. These benchmarks focus on real web tasks to evaluate agents’ capabilities. While realwebsite benchmarks are crucial for evaluating practical performance, they often lack reproducibility, which can result in overlooking finer-grained progress. In terms of reproducibility, WorkArena [9] and WorkArena++ [3] are notable alternatives. These benchmarks are designed specifically for ServiceNow, an enterprise cloud platform for business applications. However, as they are tailored to a specific platform, they do not cover general website tasks, highlighting the continued importance of benchmarks that evaluate agents on general web environments. Furthermore, WebChoreArena is fully compatible with WebArena, allowing the community’s collective efforts in WebArena to be seamlessly transferred to WebChoreArena. This compatibility enables accurate measurement of agent progress in more complex tasks.

# 3 WebChoreArena Benchmark

# 3.1 Revisiting the WebArena Benchmark

WebArena [36] is a realistic and reproducible web environment designed to facilitate the development of autonomous web browsing agents. The environment of WebArena comprises four fully operational, self-hosted web applications: online shopping (Shopping), discussion forums (Reddit), collaborative development (GitLab), business content management (Shopping Admin), and tools such as map. WebArena consists of manually created 812 long-horizon web-based tasks over the above websites. It evaluates whether the execution result achieves the intended goal.

![](images/b42a8f82444e4db0203e2ac8aa1817e8752d538b140aa3658812cdd957118b98.jpg)  
(a) Distribution of websites in WebChoreArena

![](images/d163b4078609dc09c4d44d1a69a966ddca98666a8607debd6e0ce2b0f82ce4f3.jpg)  
(b) Distribution of task types in WebChoreArena   
Figure 2: Distribution of websites and task types in WebChoreArena.

WebArena has become a standard evaluation tool in the field due to its high reproducibility and its hosting of real-world-like websites, and it has been widely used to evaluate various browsing agents [21, 17, 12, 33, 19].

# 3.2 Overview of the WebChoreArena

Our WebChoreArena benchmark consists of 532 newly and carefully human-curated tasks. It follows the design principles of WebArena and includes four simulated websites, Shopping (e-commerce platforms), Shopping Admin (content management systems), Reddit (social forum platforms), and GitLab (collaborative development platforms), providing a fully reproducible evaluation environment. The distribution of tasks across websites is illustrated in Fig. 2a: 117 tasks for Shopping, 132 for Shopping Admin, 91 for Reddit, 127 for GitLab, and 65 Cross-site tasks that require navigation across multiple websites.

# 3.3 Statistics of the WebChoreArena

WebChoreArena further categorizes the task types into the following four types:

1. Massive Memory: Tasks that require the agent to store a large amount of observations in memory at once. For example, as shown in the top-left of Fig. 3, the agent must accurately collect review scores from a category page. These tasks evaluate the agent’s ability to extract and retain all necessary information from a webpage.   
2. Calculation: Tasks that require mathematical reasoning based on previously observed content. As illustrated in the bottom-left of Fig. 3, an agent must track and sum the number of comments across the top 40 posts. This category assesses whether the agent can perform arithmetic or logical operations over stored memories from earlier steps.   
3. Long-Term Memory: Tasks that necessitate long-term memory and reasoning across multiple web pages. For instance, in the top-right of Fig. 3, the agent must first retrieve pricing rules from one page and then apply them while interacting with an order page. These tasks evaluate the agent’s capacity to recall and correctly use earlier information after multiple navigational steps.   
4. Others: Tasks involving uncommon or specialized operations, such as assigning labels in GitLab, as shown in the bottom-right of Fig. 3. These problems test the agent’s ability to handle unusual UI elements or actions not commonly encountered in simpler browsing scenarios.

These questions can have multiple types. For each question, we defined up to two types: a main type (‘type_main’) and a sub-type (‘type_sub’). Fig. 2b shows the distribution of the number of task types, considering both ‘type_main’ and ‘type_sub’. More detailed information is provided in Appendix B.

Furthermore, depending on the task, some can be solved using either screenshots or accessibility tree inputs, while others are solvable exclusively with accessibility tree inputs or exclusively with screenshot inputs. WebChoreArena contains 451 tasks solvable with any observation, 69 tasks that require text (i.e., accessibility trees), and 12 tasks that require images (i.e., screenshots). Examples of each type are provided in Appendix B. We retained this diversity as it reflects realistic challenges commonly encountered on actual websites. Importantly, each task’s configuration file specifies the

required input modality, enabling evaluators to select tasks based on the input type necessary for their specific objectives.

# 3.4 Dataset Construction Pipeline

We assigned three annotators (selected from the authors) to each of the four simulated websites. To ensure consistency in task quality across different websites, one annotator was assigned to all four websites. In total, ten annotators were involved in the task creation process.

Following the creation process of WebArena, our annotators were guided to explore the websites to familiarize themselves with the websites’ content and functionalities. Next, we instructed the annotators to formulate intents based on the following criteria:

1. Emphasis on Memory-intensive Analytical Tasks. We deliberately focused on collecting tasks that require memory, that is, tasks in which information from past observations is essential to reach the correct answer. Such tasks are common in real-world scenarios but remain largely underrepresented in existing benchmarks such as WebArena. To avoid overly simplistic tasks, we first prototyped early task ideas and evaluated them using a Claude-based agent to identify model limitations and refine the task designs. This process ensured that our final tasks were both meaningful and appropriately challenging. As a result of this construction process, our tasks fall naturally into the four task categories in Sec. 3.3.

2. Reducing Ambiguity in Task Specification and Evaluation. We explicitly instructed annotators to eliminate ambiguity in both task descriptions and evaluation criteria. While handling ambiguous instructions is important for agents aiming to operate flexibly in real-world human interactions, we prioritize clear evaluability, since reliable evaluation is essential for measuring progress. In WebArena, vague instructions often lead to scenarios where agents produce reasonable answers that are incorrectly marked as failures. For example, consider the task: “Buy the highest rated product from the category within a budget under $\$ 20$ .” with the ground-truth answer: “The website does not support sorting by rating and there are too many products in this category.” Although the agent actually purchased a product with a $100 \%$ rating, its behavior is considered incorrect according to the ground truth. In addition, we observed that the evaluation protocol in WebArena can fail to reliably assess answers due to vague output format expectations. To mitigate ambiguity in answer evaluation, we standardized the required output formats, e.g., “Provide only the answer without any additional words.” when aiming for exact matching with the ground truth (refer to Sec. 3.5 for evaluation details).

3. Template-based Task Construction and Extension. Following WebArena, we instructed annotators to create task templates and extend them to several task instances. The annotators were also responsible for developing several instantiations for each variable. This templated design enables a more robust and systematic evaluation of agent performance across tasks that share semantic similarity but exhibit diverse execution traces [36, 13].

We created a total of 117 task templates: 25 for Shopping, 29 for Shopping Admin, 20 for Reddit, 28 for GitLab, and 15 for Cross-site tasks. On average, each template yielded approximately 4.5 task instances. Here, WebArena includes several tasks based on the map website (OpenStreetMap). Although we attempted to create tasks for the map website as well, we encountered two major issues: the website’s functionality was insufficient, and the internal server managed by the WebArena team became inactive after April 2025, preventing user access. Therefore, we decided to focus on the main four websites, aiming to build a more reliable and accessible benchmark. Further details are provided in Appendix B.

To ensure the quality and correctness of each task, we conducted cross-checking with three annotators per website. Since many ambiguities were only revealed during actual task execution, we iterated through multiple rounds of inference, error analysis, and revision. This annotation process was both meticulous and labor-intensive, totaling over 300 hours of careful refinement.

# 3.5 Evaluation Protocol

Following WebArena, we adopt three evaluation metrics: string_match for assessing textual outputs, url_match for verifying the final displayed URL against the ground truth, and program_html

![](images/825e93ba37ade9576ec371422cb2f21782ab4daccce584fc876023ef7d3ef15d.jpg)  
Massive Memory (sub: Long-Term Memory)

![](images/48cc7997f1f159f3fcf19911e3b7ada5d3c382480784b0ee9b61f03f14e56771.jpg)  
Long-Term Memory (sub: Calculation)

![](images/4a03a4efbd152a807e538f6c3e8ace9295edfc2fbf0f73085090ac8021ded8c3.jpg)  
Calculation (sub: Massive Memory)

![](images/fa7842ce11d587a0af99be91a2a5cc96f56956b7e6e68279e22f75d9960d0fe6.jpg)  
Others   
Figure 3: Examples in each task type in WebChoreArena. (i) Massive Memory tasks require accurately memorizing a large amount of information from the given page. (ii) Calculation tasks involve performing arithmetic operations. (iii) Long-Term Memory tasks require the agent to retain relevant information across many steps and interactions. (iv) Others involve tasks that require special or domain-specific operations.

for functional evaluation of web interactions. The descriptions of the string-based and functional interaction evaluations are provided below.

Evaluation of Textual Outputs (string_match). String evaluation can be divided into the following three categories. (i) exact_match: A success is recorded only if the output exactly matches the ground truth. (ii) must_include: A success is recorded if the ground truth is included anywhere within the output. (iii) fuzzy_match: This function leverages a language model (GPT-4o in our implementation) to assess whether the output is semantically equivalent to the ground truth.

Evaluation of Web Interactions (program_html). This verifies whether the expected state change has occurred on the webpage after the agent’s actions. Specifically, we extract information from designated elements on the post-action webpage using locators, and compare it against the ground truth, which determines correctness in a functional manner.

# 4 Web Browsing Agents

# 4.1 Problem Formulation

The environment and agent can be modeled as a partially observable Markov decision process (POMDP): $\mathcal { E } = ( S , A , \bar { \Omega _ { \star } } T , \mathcal { M } )$ , where $S$ represents the set of states, $A$ represents the set of actions, $\Omega$ represents the set of observations and $\mathcal { M }$ is the set of memory states. The transition function is defined as $T : S \times A \to S$ , with deterministic transitions between states conditioned on actions. At each time step $t$ , the environment is in some state $s _ { t }$ (e.g., a particular page), with a partial observation $o _ { t } \in \Omega$ along with a memory buffer $M _ { t } \in \mathcal { M }$ that stores important information from previous steps up to $t - 1$ . An agent then issues an action $a _ { t } \in A$ conditioned on both $o _ { t }$ and the stored memory $M _ { t }$ , which results in a new state $s _ { t + 1 } \in S$ and a new partial observation $o _ { t + 1 } \in \Omega$ of the resulting page. Simultaneously, relevant information from $o _ { t }$ is written to the memory, updating it to $M _ { t + 1 }$ . The action $a _ { t }$ may be an interaction executed on the webpage or simply a string output.

# 4.2 Baseline Agents

For our experiments, we referred to the WebArena leaderboard [28] and adopted two open-source agents: a BrowserGym-based agent [6] and AgentOccam [33], which currently achieves stateof-the-art performance among open-source agents on WebArena. BrowserGym [6] is a unified, extensible environment for developing and evaluating web agents across diverse benchmarks with standardized observation and action spaces. AgentOccam [33] is specifically designed for the WebArena benchmark, incorporating refined observation and action spaces to better align with the

Table 1: Overall and per-website accuracy $( \% )$ . For the overall scores in WebChoreArena, we include the gap in the score from WebArena. The results indicate a substantial performance drop in WebChoreArena, suggesting significant room for improvement.   

<table><tr><td>Agent</td><td>Model</td><td>Shopping</td><td>Admin</td><td>Reddit</td><td>GitLab</td><td>Cross</td><td>Overall</td></tr><tr><td colspan="8">WebArena</td></tr><tr><td rowspan="3">AgentOccam</td><td>GPT-4o</td><td>37.4</td><td>44.0</td><td>66.0</td><td>38.9</td><td>10.3</td><td>42.8</td></tr><tr><td>Claude 3.7 Sonnet</td><td>49.7</td><td>49.5</td><td>74.5</td><td>50.0</td><td>13.8</td><td>52.0</td></tr><tr><td>Gemini 2.5 Pro</td><td>54.5</td><td>53.3</td><td>75.5</td><td>51.7</td><td>10.3</td><td>54.8</td></tr><tr><td rowspan="3">BrowserGym</td><td>GPT-4o</td><td>31.6</td><td>33.5</td><td>59.4</td><td>36.7</td><td>0.0</td><td>36.4</td></tr><tr><td>Claude 3.7 Sonnet</td><td>44.9</td><td>51.1</td><td>70.8</td><td>54.4</td><td>6.9</td><td>51.5</td></tr><tr><td>Gemini 2.5 Pro</td><td>53.5</td><td>51.6</td><td>80.2</td><td>67.2</td><td>17.2</td><td>59.2</td></tr><tr><td colspan="8">WebChoreArena</td></tr><tr><td rowspan="3">AgentOccam</td><td>GPT-4o</td><td>10.3</td><td>4.5</td><td>9.9</td><td>7.1</td><td>0.0</td><td>6.8 (-36.0)</td></tr><tr><td>Claude 3.7 Sonnet</td><td>27.4</td><td>28.8</td><td>23.1</td><td>22.8</td><td>7.7</td><td>23.5 (-28.5)</td></tr><tr><td>Gemini 2.5 Pro</td><td>41.9</td><td>42.4</td><td>44.0</td><td>38.6</td><td>10.8</td><td>37.8 (-17.0)</td></tr><tr><td rowspan="3">BrowserGym</td><td>GPT-4o</td><td>0.9</td><td>2.3</td><td>5.5</td><td>3.9</td><td>0.0</td><td>2.6 (-33.8)</td></tr><tr><td>Claude 3.7 Sonnet</td><td>16.2</td><td>26.5</td><td>18.7</td><td>25.2</td><td>30.8</td><td>23.1 (-28.4)</td></tr><tr><td>Gemini 2.5 Pro</td><td>47.9</td><td>50.0</td><td>44.0</td><td>40.2</td><td>40.0</td><td>44.9 (-14.3)</td></tr></table>

![](images/cb6870b41df3df93b6de17b1b31e5840a7da7d5445705b7784053de660e7c58a.jpg)  
(a) GPT-4o

![](images/4476e43796c3511c068a5a7ae9b12a715dda52fb900ea524a2783aa5e973edc1.jpg)  
(b) Claude Sonnet 3.7

![](images/70f54e38965a3c83e2242d8acc30fa8babb7c350f29135ee15b49dd4e2eaed2e.jpg)  
(c) Gemini 2.5 Pro   
Figure 4: Comparison across different task types. This result reveals that the methodology of the agent itself has a substantial impact on its effectiveness across different task types.

pre-training data of LLMs. We also attempted to run experiments using closed-source agents such as OpenAI’s Operator [21], but were unable to do so because attempts to connect to the WebArena sandbox were blocked due to the failure to establish a secure connection. Therefore, we did not include them in our experiments. The more detailed information on these agents is included in Appendix C due to the space limitation.

# 5 Experiment

# 5.1 Main Results

For this experiment, we employed GPT-4o, Claude 3.7 Sonnet, and Gemini 2.5 Pro as our foundational LLMs. The rationale is as follows: GPT-4o represents the most frequently utilized LLM in prior academic research, thereby ensuring compatibility with prior studies [14, 7, 33]. Furthermore, the inclusion of Claude 3.7 Sonnet and Gemini 2.5 Pro aimed to evaluate the performance of recent state-of-the-art LLMs.

Table 1 presents the results of WebChoreArena, alongside the results of WebArena for comparison. The number of tasks in WebArena is as follows: Shopping: 187, Shopping Admin: 182, Reddit: 106, GitLab: 180, and Cross: 29 (excluding Map for fair comparison). Our main findings are as follows:

F1: GPT-4o Struggles Significantly on WebChoreArena. From Table 1, it is evident that GPT-4o struggles significantly on WebChoreArena. While it achieved around $4 2 . 8 \%$ accuracy on WebArena, its performance drops sharply to $6 . 8 \%$ and $2 . 6 \%$ on WebChoreArena. This indicates that WebChore-Arena is significantly more challenging than WebArena, emphasizing the need for more advanced LLMs to tackle these tasks.

F2: Latest LLMs Show Progress but Have Significant Room for Improvement. As LLMs have evolved with models such as Claude 3.7 Sonnet and Gemini 2.5 Pro, their performance in

Table 2: Performance with different modalities. We evaluated three LLMs using BrowserGym across different input modalities. The results indicate that incorporating image inputs does not necessarily lead to overall performance improvements.   

<table><tr><td></td><td>Input</td><td>Shopping (#25)</td><td>Admin (#29)</td><td>Reddit (#20)</td><td>GitLab (#28)</td><td>Overall (#102)</td></tr><tr><td rowspan="2">GPT-4o</td><td>Image + A11y Tree</td><td>0.0</td><td>3.4</td><td>5.0</td><td>3.6</td><td>2.9</td></tr><tr><td>A11y Tree</td><td>0.0</td><td>3.4</td><td>5.0</td><td>3.6</td><td>2.9</td></tr><tr><td rowspan="2">Claude</td><td>Image + A11y Tree</td><td>4.0</td><td>13.8</td><td>10.0</td><td>17.9</td><td>11.8</td></tr><tr><td>A11y Tree</td><td>16.0</td><td>34.5</td><td>5.0</td><td>35.7</td><td>24.5</td></tr><tr><td rowspan="2">Gemini</td><td>Image + A11y Tree</td><td>28.0</td><td>55.2</td><td>40.0</td><td>32.1</td><td>39.2</td></tr><tr><td>A11y Tree</td><td>48.0</td><td>48.3</td><td>45.0</td><td>42.9</td><td>46.1</td></tr></table>

WebChoreArena has also improved. However, even for Gemini, there is still an approximate $14 \%$ performance drop in overall scores, highlighting both the difficulty of tasks in WebChoreArena and the current limitations of state-of-the-art LLM-based agents. Although cross-site tasks in WebChoreArena perform slightly better than in WebArena, it is important to note that the presence of the small number of tasks in WebArena $\scriptstyle ( 1 = 2 9$ ) limit the reliability of such comparisons.

F3: WebChoreArena Enables a Clearer and Deeper Measurement of the Performance Differences among the Models. WebChoreArena serves as a more effective benchmark for distinguishing model performance. Unlike WebArena, which presents a narrower performance spectrum (GPT-4o: $3 6 . 4 \%$ vs. Gemini 2.5 Pro: $5 9 . 2 \%$ with BrowserGym), WebChoreArena exposes a substantial performance divergence (GPT-4o: $2 . 6 \%$ vs. Gemini 2.5 Pro: $4 4 . 9 \%$ ). Therefore, WebChoreArena provides model developers and evaluators with clear insights into the strengths and weaknesses of each model.

F4: WebChoreArena Enables Fine-grained Analysis of Task-specific Performance. Fig. 4 presents a detailed analysis of each agent’s performance across diverse task typologies. The results underscore the significant influence of agent architecture, beyond the type of LLMs, on type-wise performance. Notably, Gemini 2.5 Pro performs best on Massive Memory Tasks in BrowserGym, whereas AgentOccam shows the worst performance in this category. This divergence can be attributed to fundamental differences in their memory management strategies (Further elaboration in Appendix C). In this way, analyzing the performance in each task type allows model and agent developers to receive feedback on which mechanisms should be improved.

# 5.2 Analysis

Effect on Input Modality. We investigate the impact of input data modality on agent performance. The main experiments primarily utilized text-based inputs (i.e., accessibility trees) to mitigate visual hallucinations (with the exception of only three templates requiring image inputs) following previous work [33, 36, 7]. Nevertheless, analyzing how performance changes when image input (i.e., screenshots) is incorporated would provide significant insights. We selected one task from each task template across the four websites in WebChore-Arena, creating a small-set specifically for analysis. We adopt BrowserGym for this experiment. The results are summarized in Table 2. These results indicate an overall trend of decreased performance when incorporating image inputs. Notably, certain website categories, such as shopping, exhibit a significant performance shift with the inclusion of visual information.

Table 3: Analysis by required observation type.   

<table><tr><td rowspan="2"></td><td rowspan="2">Input</td><td colspan="2">Required Obs</td></tr><tr><td>Text (#15)</td><td>Any (#85)</td></tr><tr><td rowspan="2">GPT-4o</td><td>Image+Tree</td><td>6.7</td><td>2.4</td></tr><tr><td>Tree</td><td>6.7</td><td>2.4</td></tr><tr><td rowspan="2">Claude</td><td>Image+Tree</td><td>13.3</td><td>11.8</td></tr><tr><td>Tree</td><td>13.3</td><td>27.1</td></tr><tr><td rowspan="2">Gemini</td><td>Image+Tree</td><td>20.0</td><td>43.5</td></tr><tr><td>Tree</td><td>40.0</td><td>48.2</td></tr></table>

To further investigate this issue, Table 3 presents performance scores based on the required observation types annotated in the WebChoreArena (refer to Sec. 3.3). Here, a required observation of “Text” refers to cases where a gap exists between the visual and textual (tree) information, and the correct answer can only be obtained through the textual modality. In contrast, “Any” indicates tasks that are solvable using either modality. Representative examples are provided in Appendix B. While the number of samples

is limited, the results clearly reveal that tasks requiring text-only information (i.e., those in which hallucinations occur on the visual side) exhibit a notable performance drop, particularly for Gemini. Consequently, exploring methodologies to leverage visual information while mitigating hallucinations represents a crucial direction for future research.

Does Tool Use Improve Performance? We investigate whether the use of external tools, particularly calculators, enhances agent performance. For this experiment, we utilized a web-based calculator developed by the WebArena team, which provides a GUI-based interface that allows agents to perform arithmetic operations seamlessly. We explicitly give agents the following instruction: “If you need to do some calculations, you can use the calculator at <URL for Calculator>.”

We extracted 215 calculation-specific tasks from Web-ChoreArena to evaluate the effectiveness of tool use. The results are presented in Table 4. The results showed that the overall performance remained largely unchanged. The main reason is that the model rarely attempts to use tools. As shown in Table 4, out of 215 tasks, the number of tool-using tasks accounts for less than $28 \%$ of the total. Agents appear to prefer solving problems directly, as it is more efficient than using the tool when they perceive the

Table 4: Performance with Calculators.   

<table><tr><td></td><td colspan="2">Acc.</td><td rowspan="2">#Tool Usage</td></tr><tr><td></td><td>Normal</td><td>w. Tool</td></tr><tr><td>GPT-4o</td><td>3.7</td><td>2.8</td><td>35</td></tr><tr><td>Claude</td><td>19.5</td><td>18.6</td><td>59</td></tr><tr><td>Gemini</td><td>40.0</td><td>42.8</td><td>41</td></tr></table>

problem as solvable on their own. Therefore, it was found that simply using the calculator tool does not necessarily improve WebChoreArena’s performance.

# 6 Error Analysis

This section presents an analysis of the tendency of the errors of Gemini 2.5 Pro (BrowserGym). We carefully examined the failure cases and identified several distinct types of mistakes:

Counting Errors. In the Massive Memory task, while agents can accurately count items within a single webpage, they often encounter difficulties and commit counting errors when the task necessitates navigating and aggregating information across multiple pages.

Calculation Errors. We observed no errors in simple addition or multiplication tasks. However, Gemini 2.5 Pro started to make calculation mistakes noticeably more often when it had to add or multiply more than fifteen numbers.

Forgetting Instructions. We observed several instances where instructions were overlooked. For example, the agent occasionally disregarded the instruction to select only products with “more than 5 reviews” or failed to adhere to a specified output format.

Operational Errors. We also observed several operational errors. For example, the agent sometimes failed to remember its previous actions. In one case, it successfully reached the second page but mistakenly believed it was still on the first page, causing it to navigate to another page unnecessarily.

Other Errors. Other errors include listing products that do not exist, ending the search too soon without checking all the pages, and quitting complex searches in the middle to try a faster way but getting lost and unable to complete the task.

# 7 Conclusion and Limitations

This paper introduces WebChoreArena, a new fully reproducible benchmark comprising 532 carefully curated tasks designed to extend the scope of WebArena beyond general browsing to more laborintensive and tedious tasks. Our limitations are (i) Method Development: This work primarily contributes through the construction of the benchmark and does not focus on developing new methods. We consider that designing novel methods is a crucial next step based on the findings revealed in this study. We believe our results provide a strong foundation to facilitate future research in this direction. (ii) Simulation-based Websites: Our experiments are conducted in a simulated web environment that ensures full reproducibility while closely approximating real-world websites. Although some gap may remain, we believe this setup provides a valuable testbed for rigorous evaluation. Developing an online extension of WebChoreArena is a crucial next step to further align with real-world settings while preserving reproducibility.

# Appendix

In this Appendix, we provide reviews of WebArena in Sec. A, details of WebChoreArena in Sec. B and experimental details in Sec. C.

# A Review of WebArena

# A.1 Annotation Error Analysis

To investigate the upper bound of WebArena performance, we analyzed the annotation errors (including ambiguous task descriptions) in the WebArena benchmark. Here, we would like to emphasize that identifying annotation errors in web agent benchmarks is inherently difficult. In many cases, such errors only become apparent after running strong agents on the benchmark. Given that such powerful agents were not available during WebArena’s initial development, we believe that achieving perfect annotation at that time was extremely challenging.

We first extracted 229 tasks out of 684 (excluding map website) that were failed by all three Browser-Gym agents (GPT-4o, Claude 3.7 Sonnet, and Gemini $2 . 5 \mathrm { P r o } $ ). These tasks were reviewed by the authors. Our analysis revealed that approximately 134 out of 229 tasks $( 5 8 . 5 \%$ , $2 0 . 0 \%$ for all tasks) contained either annotation errors (75) or evaluation issues (59). As for common evaluation issues, one example is using exact_match (perfect matching with GT) without clearly instructing the agent to return only the answer string, leading to mismatches due to extra context in the output. Another example is using fuzzy_match (i.e., GPT-based evaluation) without explicit formatting instructions, leading to cases where even GPT marks the agent’s output incorrect due to superficial differences in format. Annotation errors were identified across all websites: 15 in Shopping, 21 in Shopping Admin, 19 in GitLab, 12 in Reddit, and 8 in Cross-site tasks.

As a result, performance would get stuck around $80 \%$ , and these issues can introduce noise that prevents the benchmark from accurately capturing the agent’s true performance. This highlights the need for new benchmarks that are more challenging and carefully designed to minimize errors for the recent advanced LLM-based agents.

# A.2 Rationale for Excluding the Map Domain

As noted in Sec. 3.4 of the main paper, the map website in WebArena has had issues since April 2025 (e.g., search results for locations no longer appear). We reported this issue to the WebArena team via a GitHub issue, and several followers raised similar concerns. Consequently, we decided not to include tasks on the map category in our WebChoreArena. However, we emphasize that inaccessibility is not the sole reason for its removal. We analyzed map websites/tasks when access was still available, and identified several critical issues that motivated the removal of the category. The main concerns are as follows:

Limited Interaction Diversity. The core functionality of the map website is fundamentally restricted to two actions: (i) searching for a location and (ii) finding a route between two locations. In particular, for case (i), the task of finding “B near A” is especially problematic, as the system only works when queries follow a specific format, such as specifying the amenity type followed by “near” and a location. For example, “cafe near NYU” (where cafe is a valid amenity type) works correctly, but queries like “Starbucks near NYU” (where Starbucks is not an amenity type), or “cafe close to NYU” (which does not follow the required format), do not. This significantly limits the diversity of tasks that can be constructed and makes it difficult to create challenging problems that recent LLMs-based agents struggle with. Also, many of the existing WebArena tasks in the map domain rely heavily on knowledge-based queries. For example, in the task “Tell me the full address of all international airports that are within a driving distance of 30 km to Carnegie Art Museum”, the key challenge should lie in retrieving locations within a $3 0 \mathrm { k m }$ radius through browsing. However, due to the above technical difficulty, agents resorts to using its own parametric knowledge to identify relevant airports and then answers the question correctly by generating their full names. We argue that such behavior does not reflect true browsing capabilities. Therefore, constructing high-quality tasks that genuinely test browsing ability in the map domain remains difficult.

Low Reproducibility. The map interface exhibits high sensitivity to minor input differences. First, we observed that using abbreviations versus full names for the same location often yields different

![](images/c43bf1bbff5461733d00f493113cdeaafaf0365842e99e9d8232ce1448d6595b.jpg)  
(a) Distribution of single type and multiple types

![](images/0cad95da1310b7d1b5b0bcc97357b5b7573777053d24b552871a3bccd1731be3.jpg)  
(b) Distribution of task type combinations   
Figure A: Task type distributions in WebChoreArena.

results. For example, searching for “CMU” and “Carnegie Mellon University” returns slightly different locations. Also, we observed that the search outcomes can be affected by the visible region of the map at the search time.

Due to the above reasons, we decided not to create tasks in the map website. We consider that even without the map domain, the remaining four main websites are sufficient to accurately evaluate agent performance.

# B Details and Failure Cases of WebChoreArena

# B.1 Details of Task Distribution

As described in Sec. 3.3, each task in WebChoreArena is associated with up to two task types. The distribution of these types is shown in Fig.A. Here, since there was no significant difference between ‘type_main’ and ‘type_sub’, we counted them equally without distinguishing between primary and secondary types. Fig.A (a) shows that $6 6 . 5 \%$ of the tasks belong to a single type, while the remaining tasks are associated with multiple types. Furthermore, Fig.A (b) illustrates the diverse combinations of task types. These results highlight the variety and richness of tasks in WebChoreArena.

# B.2 Examples of Tasks That Only be Solvable with Text or Require Visual Input

Fig. B presents illustrative examples of tasks that only be solvable with text (accessibility tree) or require visual input (screenshot) in addition to text. As reported in Sec. 3.3 (main), WebChoreArena contains 451 tasks solvable with any observation, 69 tasks that can only be solved using text input, and 12 tasks that require images. The details in Fig. B are as follows:

• Left (Shopping): A product review snippet shows a rating that appears to be “80 (4 stars)” in the screenshot, but the accessibility tree reveals that the true value is “78”. Due to this discrepancy, the task requires the textual accessibility tree. We label it as required_obs $=$ text.   
• Right (Shopping Admin): The number of review stars is shown only as graphical icons and does not appear in the accessibility tree. In this case, visual information is necessary to answer correctly, so we set required_obs $=$ image.

For each task, we provide only the accessibility tree as input when the required_obs is any or text, and both the accessibility tree and a screenshot when the required_obs is image, as described in Sec. 5.2.

# B.3 Failure Examples in WebChoreArena

We present two failure cases per website, taken from the results of Gemini 2.5 Pro using BrowserGym.

![](images/ab264ff5655544ff4df69f716036e6ac58da003b1f7318a252294666e8d72d9f.jpg)  
Figure B: Examples of tasks with discrepancies between visual and textual information.

# B.3.1 Shopping

Failed Task1. “List only the SKUs (without any additional text) of products in the Home Storage Hooks category that are priced within $60 \%$ of the original product NOZE Rustic Coat Rack Wall Mounted Shelf with 4 Hooks, Hanging Entryway Organizer for Mug Coffee Cup, Holding Solid Wooden Shelf with 2 Baskets for Kitchen Living Room, Bathroom and Bedroom from my Order 170 and have a higher rating than the original product. Format your answer with one SKU per line, sorted alphabetically.”

Reason for Failure1. The agent failed to detect some of the target items.

Failed Task2. “What is the combined price of the most affordable items between Latest iPhone in Cell phone category products and the cheapest Flip Case with Card Holder that is compatible with latest iPhone? Please only output the number.”

Reason for Failure2. The agent refers to a product, iPhone 13, which does not exist on the website.

# B.3.2 Shopping Admin

Failed Task1. “List all the customer names (bill-to) who placed the highest sales total order(s) in each month of the first half of 2022, where the sales total of completed orders exceeded the monthly average for that year. The final answer should be in the format ’[name1, name2, ...]’ without reasoning, where names are sorted based on the descending order of their corresponding month’s sales total. If multiple months have the same sales total, those months should be ordered chronologically. If multiple customers placed the highest sales total order in the same month, their names should be sorted alphabetically.”

Reason for Failure1. When identifying months that exceed the monthly average sales, the agent compares the monthly average to individual order sales rather than to the total sales per month.

Failed Task2. “I want to replenish 40 units of each Simple Product that has the Attribute Bottom. These products should be among the top 50 oldest last-updated products within that category. Here, if the same updated time appears beyond the top 50, include those as well. If a product with the same ID appears more than once, it is a bug, so please ignore it. If the purchase unit price for restocking is 60% of the selling price, how much will the total cost be? Do all calculations in decimals and round the purchase amount to the nearest integer just before answering.”

Reason for Failure2. Although instructed to take the top 50 items, the agent took more than 50 entries.

# B.3.3 Reddit

Failed Task1. “I want to analyze all the submissions from my account. Please count how many times I’ve posted in each forum, and return the result in the following dictionary format, sorted by the number of posts in descending order. If multiple forums have the same number of posts, sort them alphabetically by forum name: forum3: 51, forum1: 25, forum2: 4, forum6: 4,...”

Reason for Failure1. The agent first counted all of its submissions, but made a mistake and had to start over. It could not match the new results with the earlier memory information, and the task failed as a result.

Failed Task2. “ Calculate the total number of comments on the top 40 most commented posts of all time on forum worldnews. ”

Reason for Failure2. The agent made a calculation mistake.

# B.3.4 GitLab

Failed Task1. “ How many issues in total have the accessibility label or the redesign label? Include both Open and Closed issues (All). Consider all issues, not limited to yours. If an issue has both labels, count it only once. ”

Reason for Failure1. The agent incorrectly specified the label.

Failed Task2. “ Count the number of open issues that have at least 2 comments in TrafficMonitor. Output the count as a single integer with no extra text, spaces, or formatting.

Reason for Failure2. The agent prematurely terminated the task after interacting with only a limited number of pages.

# C Experimental Details

# C.1 Web Agent Configurations

# C.1.1 Overview of Agents

BrowserGym. BrowserGym [6] is a unified, extensible environment for developing and evaluating web agents across diverse benchmarks with standardized observation and action spaces. The action list is shown in Table A (a). Following the implementation of WebArena, we ran WebChoreArena within the BrowserGym framework using a set of 15 actions, which are composed of basic actions, tab operations, page operations, and messaging.

AgentOccam. AgentOccam [33] is specifically designed for the WebArena benchmark, incorporating refined observation and action spaces to better align with the pre-training data of LLMs. Moreover, it employs a planning strategy that supports branching, allowing the agent to generate alternative plans, and pruning, which eliminates suboptimal plans based on intermediate outcomes, thereby enabling more efficient and adaptive decision-making. The action list is shown in Table A (b). Following the original implementation, we ran WebChoreArena using a set of 8 actions which are composed of basic actions, page operations, workflow management, and planning actions.

# C.1.2 Execution Settings

BrowserGym. We utilize the implementation of the BrowserGym code in Agent Workflow Memory [27]. We gratefully acknowledge the authors for providing such easily reusable code. Following the existing implementation, we permit multiple actions per step. We set the maximum number of steps to 50 for all WebChoreArena tasks.

AgentOccam. We utilize the original implementation of the AgentOccam [33]. Following the default settings, we permit only a single action per step. We set the maximum number of steps to 50 for all WebChoreArena tasks.

# C.2 LLM Implementation Details

GPT-4o. We used the GPT-4o model provided by Azure, specifically the GPT-4o-2024-05-13 version. This version was chosen because newer versions of GPT-4o tend to make agents respond prematurely [7], and the authors of that study also recommend using GPT-4o-2024-05-13 for more stable agent behavior. For the hyperparameters, we followed the existing implementation when available. For BrowserGym, we set the temperature to 0.1 and the max new tokens to 2,000. For AgentOccam, we set the temperature to 0.5, the top-p value to 0.95, and the max tokens to 128,000.

Table A: Action Spaces   
(a) Action space of BrowserGym   

<table><tr><td>Action Type</td><td>Description</td></tr><tr><td>noop</td><td>Do nothing</td></tr><tr><td>scroll(dir)</td><td>Scroll up/down</td></tr><tr><td>press(key_comb)</td><td>Press a key combination</td></tr><tr><td>click(elem)</td><td>Click at an element</td></tr><tr><td>fill(elem, text)</td><td>Type to an element,</td></tr><tr><td>hover(elem)</td><td>Hover on an element</td></tr><tr><td>select_option(elem, option)</td><td>Select options</td></tr><tr><td>tab-focus(index)</td><td>Focus on i-th tab</td></tr><tr><td>new_tab</td><td>Open a new tab</td></tr><tr><td>tab_close</td><td>Close current tab</td></tr><tr><td>go_back</td><td>Visit the last URL</td></tr><tr><td>go_forward</td><td>Undo go_back</td></tr><tr><td>goto(URL)</td><td>Go to URL</td></tr><tr><td>send msg_to_user(message)</td><td>Send a message to the user</td></tr><tr><td>report_infeasible Reason)</td><td>Send special message and terminate</td></tr></table>

(b) Action space of AgentOccam   

<table><tr><td>Action Type</td><td>Description</td></tr><tr><td>click[id]</td><td>Click at an element</td></tr><tr><td>type[id][content]</td><td>Type into an element</td></tr><tr><td>go_back</td><td>Visit the last URL</td></tr><tr><td>go_home(URL)</td><td>Go to the home page</td></tr><tr><td>note[content]</td><td>Take notes</td></tr><tr><td>stop[answer]</td><td>Stop with an answer</td></tr><tr><td>branch[id][intent]</td><td>Generate a new plan</td></tr><tr><td>prune{id}[reason]</td><td>Restore to a previous plan</td></tr></table>

Claude 3.7 Sonnet. We used claude-3-7-sonnet-20250219. Claude 3.7 Sonnet is the hybrid reasoning model. For the hyperparameters, we followed the existing implementation when available. For BrowserGym, we set the temperature to 0.1, top-p to 0.95, and max new tokens to 2,000. For AgentOccam, we set the temperature to 0.95, the top-p to 0.95, and the max tokens to 4,096.

Gemini 2.5 Pro. We used gemini-2.5-pro-preview-03-25. Gemini 2.5 Pro is the most advanced reasoning Gemini model, capable of solving complex problems. For the hyperparameters, we followed the existing implementation if they exist. For BrowserGym, we set the temperature to 0.1, top-p to 0.95, and the maximum number of new tokens to 8,000. For AgentOccam, we set the default values (the specific settings are unofficial).

# C.3 Prompt Design

# C.3.1 Website-specific Tips

We refer to the WebArena implementation provided by OpenAI’s CUA [21] and incorporate websitespecific tips in our experiments. The tips we provide are identical in content to those used in [21]. Below, we present the actual tips applied to each website.

# Tips for Shopping:

Here are tips for using this website:

• 1. This website provides very detailed category of products. You can hover categories on the top menu to see subcategories.   
• 2. If you need to find information about your previous purchases, you can go My Account $>$ My Orders, and find order by date, order number, or any other available information   
• 3. An order is considered out of delivery if it is marked as "processing" in the order status   
• 4. When the task asks you to draft and email. DO NOT send the email. Just draft it and provide the content in the last message

# Tips for Shopping Admin:

Here are tips for using this website:

• 1. When you add a new product in the CATALOG $>$ Products tab, you can click the downwardarrow beside the "Add Product" button to select options like "Simple Product", "Configurable Product", etc.   
• 2. If you need to add new attribute values (e.g. size, color, etc) to a product, you can find the product at CATALOG $>$ Products, search for the product, edit product with "Configurable Product" type, and use "Edit Configurations" to add the product with new attribute values. If the value that you want does not exist, you may need to add new values to the attribute.   
• 3. If you need to add new values to product attributes (e.g. size, color, etc), you can visit STORES $>$ Attributes $>$ Product, find the attribute and click, and add value after clicking "Add Swatch" button.   
• 4. You canREPORTS $>$ enerate various reports by using menus in the REPORTS tab. Select "report type", select options, and click "Show Report" to view report.   
• 5. In this website, there is a UI that looks like a dropdown, but is just a 1-of-n selection menu. For example in REPORTS $>$ Orders, if you select "Specified" Order Status, you will choose one from many options (e.g. Canceled, Closed, ...), but it’s not dropdown, so your click will just highlight your selection (1-of-n select UI will not disappear).   
• 6. Configurable products have some options that you can mark as "on" of "off". For example, the options may include "new", "sale", "eco collection", etc.   
• 7. You can find all reviews and their counts in the store in MARKETING $>$ User Content $>$ All Reviews. If you see all reviews grouped by product, go REPORTS $>$ By Products and search by Product name.   
• 8. This website has been operating since 2022. So if you have to find a report for the entire history, you can select the date from Jan 1, 2022, to Today.   
• 9. Do not export or download files, or try to open files. It will not work.

# Tips for Reddit:

Here are tips for using this website:

• 1. when the task mentions subreddit, it is referring to ‘forum’.   
• 2. if you need find a relevant subreddit or forum, you can find the name after clicking "alphabetical" in the "Forum" tab.   
• 3. if you have to find submissions (posts) or comments by a particular user, visit reddit.site/user/<user name> to see the list

# Tips for GitLab:

Here are tips for using this website:

• 1. your user name is byteblaze   
• 2. To add new members to the project, you can visit project information $>$ members tab and click blue "invite members" button on top right   
• 3. To set your status, click profile button on top right corner of the page (it’s next to the question mark button) and click edit status   
• 4. To edit your profile, click profile button on top right corner of the page (it’s next to the question mark button) and click edit profile   
• 5. You can also access to your information e.g. access token, notifications, ssh keys and more from "edit profile" page   
• 6. Projects that you have contributed to are listed under Project / Yours / All tab of gitlab.site. You can sort repos using dropdown button on top right   
• 7. Projects’s repository tab has menus like Commits, Branches, Contributors, and more. Contributors tab shows contributors and their number of commits   
• 8. If you want to see all the issues for you, you can either click button on the right of $^ +$ icon on top right menu bar   
• 9. When the task mentions branch main, it often means master

# C.3.2 Full Prompt Examples

We present examples of the full input provided to the agent in Fig. C, D, E, and F. Fig. C and D illustrates input examples from BrowserGym, while Fig. E and F show input examples from AgentOccam.

When the task is a cross-site task, we add some hints following the original implementations. For BrowserGym, we add the following hint after the task description: “(Note: if you want to visit other websites, check out the homepage at <home_url>. It has a list of websites you can visit. <home_url>/password.html lists all the account name and password for the websites. You can use them to log in to the websites.)”. For AgentOccam, we add the additional action after the last action description: “- go_home: To return to the homepage where you can find other websites.”

Based on these inputs, we conduct a discussion on the memory mechanism in the following section.

# C.4 Agent Memory Mechanisms

We explain the details of the memory function for AgentOccam and BrowserGym. BrowserGym adopts an explicit memory mechanism. At each step, the agent outputs a reason for action, the action itself, and, when necessary, memory content to be stored. In the subsequent step, the input includes the past reasons for action, actions, and memory entries. This setup allows the agent to retain and refer back to essential information in memory. Therefore, for tasks that require past memory, it is sufficient for the agent to output the relevant memory information appropriately at each step.

In contrast, AgentOccam employs a different strategy. At each step, it outputs the interaction history summary, observation summary, reason for action, action, and an observation highlight. The next input includes the reason for action, action, and the observation highlight (or the observation summary if the highlight is too long). A key feature of AgentOccam is the note [content] action, which allows the agent to explicitly store important information. Once this action is issued, the content within [content] is included in subsequent inputs.

The key difference lies in memory handling: BrowserGym explicitly outputs memory at each step, while AgentOccam relies on summaries or must issue a note [content] action to retain important information. This explains why AgentOccam performs worse on Massive Memory tasks in Fig. 4 (the main paper).

You are an agent trying to solve a web task based on the content of the page and a user instructions. You can interact with the page and explore. Each time you submit an action it will be sent to the browser and you will receive a new page.

Here are tips for using this website:

1. This website provides very detailed category of products. You can hover categories on the top menu to see subcategories.   
2. If you need to find information about your previous purchases, you can go My Account $>$ My Orders, and find order by date, order number, or any other available information   
3. An order is considered out of delivery if it is marked as "processing" in the order status   
4. When the task asks you to draft and email. DO NOT send the email. Just draft it and provide the content in the last message

Figure C: A prompt example of system_message for BrowserGym.

You are a UI Assistant, your goal is to help the user perform tasks using a web browser. You can communicate with the user via a chat, in which the user gives you instructions and in which you can send back messages. You have access to a web browser that both you and the user can see, and with which only you can interact via specific commands.

Review the instructions from the user, the current state of the page and all other information to find the best possible next action to accomplish your goal. Your answer will be interpreted and executed by a program, make sure to follow the formatting instructions.

# ## Chat messages:

- [assistant] Hi! I am your UI assistant, I can perform web tasks for you. What can I help you with?

- [user] Please provide the distribution of reviews for Snakebyte Twin Charge X - Xbox One Controller Charger Dual Docking/Charging Station incl. 2 Rechargeable Battery Packs for XBOX One Controller / Elite / S Controller Gamepad, Black. Here, the review stars (15) correspond to a Rating that is 20 times their value ˜ $1 =$ Rating 20, $2 =$ Rating 40, $3 =$ Rating 60, $4 =$ Rating 80, $5 =$ Rating 100).

Follow the format below, using numerical values: 5: {number}, 4: {number}, 3: {number}, 2: {number}, 1: {number}

# Observation of current step:

## AXTree: (omitted)

# History of interaction with the task:

## step 0

### Action:

hover(’856’)

...(omitted some steps)

## step 7

### Action:

hover(’1068’)

click(’1068’)

### Memory:

The product is related to Xbox One and should be under the "Video Games" category.

## step 8

### Action:

hover(’1068’)

click(’1068’)

### Memory:

The product "Snakebyte Twin Charge X - Xbox One Controller Charger Dual Docking/Charging Station incl. 2 Rechargeable Battery Packs for XBOX One Controller / Elite / S Controller Gamepad, Black" is related to Xbox One and should be under the "Video Games" category.

# Action space:

15 different types of actions are available. (omitted)

# Abstract Example

Here is an abstract version of the answer with description of the content of each tag. Make sure you follow this structure, but replace the content with your answer: (omitted)

# Concrete Example

Here is a concrete example of how to format your answer. Make sure to follow the template with proper tags: (omitted)

Figure D: A prompt example of user_message for BrowserGym.

You are an AI assistant performing tasks on a web browser. You will be provided with task objective, current step, web page observations, previous plans, and interaction history. You need to issue an action for this step.

Generate the response in the following format:

INTERACTION HISTORY SUMMARY:

Emphasize all important details in the INTERACTION HISTORY section.

OBSERVATION DESCRIPTION:

Describe information in the CURRENT OBSERVATION section. Emphasize elements and features that are relevant or potentially helpful for fulfilling the objective in detail.

REASON:

Provide your rationale for proposing the subsequent action commands here.

ACTION:

Select your action here.

OBSERVATION HIGHLIGHT:

List the numerical ids of elements on the current webpage based on which you would issue your action. Also include elements on the current webpage you would attend to if you fail in the future and have to restore to this step. Don’t include elements from the previous pages. Select elements at a higher hierarchical level if most their children nodes are considered crucial. Sort by relevance and potential values from high to low, and separate the ids with commas. E.g., ‘1321, 52, 756, 838‘.

You are ONLY allowed to use the following action commands. Strictly adheres to the given format. Only issue one single action. If you think you should refine the plan, use the following actions:   
- branch [parent_plan_id] [new_subplan_intent]: To create a new subplan based on PREVIOUS PLANS. Ensure the new subplan is connected to the appropriate parent plan by using its ID. E.g., ‘branch [12] [Navigate to the "Issue" page to check all the issues.]‘   
- prune [resume_plan_id] [reason]: To return to a previous plan state when the current plan is deemed impractical. Enter the ID of the plan state you want to resume. E.g., ‘prune [5] [The current page lacks items "black speaker," prompting a return to the initial page to restart the item search.]‘ Otherwise, use the following actions:   
- click [id]: To click on an element with its numerical ID on the webpage. E.g., ‘click [7]‘ If clicking on a specific element doesn’t trigger the transition to your desired web state, this is due to the element’s lack of interactivity or GUI visibility. In such cases, move on to interact with OTHER similar or relevant elements INSTEAD.   
- type [id] [content] [press_enter_after=0|1]: To type content into a field with a specific ID. By default, the "Enter" key is pressed after typing unless ‘press_enter_after‘ is set to 0. E.g., ‘type [15] [Carnegie Mellon University] [1]‘ If you can’t find what you’re looking for on your first attempt, consider refining your search keywords by breaking them down or trying related terms.   
- stop [answer]: To stop interaction and return response. Present your answer within the brackets. If the task doesn’t require a textual answer or appears insurmountable, indicate "N/A" and additional reasons and all relevant information you gather as the answer. E.g., ‘stop [5h 47min]‘   
- note [content]: To take note of all important info w.r.t. completing the task to enable reviewing it later. E.g., ‘note [Spent $\$ 10$ on 4/1/2024]‘   
- go_back: To return to the previously viewed page.

# Here are tips for using this website:

1. This website provides very detailed category of products. You can hover categories on the top menu to see subcategories.   
2. If you need to find information about your previous purchases, you can go My Account $>$ My Orders, and find order by date, order number, or any other available information   
3. An order is considered out of delivery if it is marked as "processing" in the order status   
4. When the task asks you to draft and email. DO NOT send the email. Just draft it and provide the content in the last message

Figure E: A prompt example of system_message for AgentOccam.

OBJECTIVE: Please provide the distribution of reviews for Snakebyte Twin Charge X - Xbox One Controller Charger Dual Docking/Charging Station incl. 2 Rechargeable Battery Packs for XBOX One Controller / Elite / S Controller Gamepad, Black. Here, the review stars (1 5) correspond to a Rating that is 20 times their value $(1 =$ Rating 20, $2 =$ Rating 40, $3 =$ Rating 60, $4 =$ Rating 80, $5 =$ Rating 100). Follow the format below, using numerical values: 5: {number}, 4: {number}, 3: {number}, 2: {number}, 1: {number}  
PREVIOUS PLANS: [0] (Active Plan) Find the solution to "Please provide the distribution of reviews for Snakebyte Twin Charge X - Xbox One Controller Charger Dual Docking/Charging Station incl. 2 Rechargeable Battery Packs for XBOX One Controller / Elite / S Controller Gamepad, Black. Here, the review stars (1 5) correspond to a Rating that is 20 times their value $(1 =$ Rating 20, $2 =$ Rating 40, $3 =$ Rating 60, $4=$ Rating 80, $5 =$ Rating 100). Follow the format below, using numerical values: 5: {number}, 4: {number}, 3: {number}, 2: {number}, 1: {number}"  
INTERACTION HISTORY:  
<step_0_interaction>  
OBSERVATION:  
RootWebArea 'One Stop Market'  
link 'My Account'  
link 'My Wish List'  
link 'Sign Out'  
text 'Welcome, Emma Lopez!'  
link 'Skip to Content'  
link 'store logo'  
link 'My Cart'  
combobox 'Search' [required: False]  
link 'Advanced Search'  
button 'Search' [disabled: True]  
REASON FOR ACTION:  
To locate the specific product and its review distribution, I need to use the search functionality provided on the webpage. This will help narrow down the search to the exact product. ACTION:  
type [2139] [Snakebyte Twin Charge X - Xbox One Controller Charger Dual Docking/Charging Station incl. 2 Rechargeable Battery Packs for XBOX One Controller / Elite / S Controller Gamepad, Black] [1]  
</step_0_interaction>  
<step_1_interaction>  
OBSERVATION:  
The search results page for "Snakebyte Twin Charge X - Xbox One Controller Charger Dual Docking/Charging Station incl. 2 Rechargeable Battery Packs for XBOX" is displayed. The relevant product is listed with the following details: - Product link: 'Snakebyte Twin Charge X - Xbox One Controller Charger Dual Docking/Charging Station incl. 2 Rechargeable Battery Packs for XBOX One Controller / Elite / S Controller Gamepad, Black' - Rating: 52 - 32 Reviews There are also options to add the product to the cart, wish list, or compare it with other products. REASON FOR ACTION:  
To view the distribution of reviews for the specific product, I need to click on the product link to access its detailed review section. ACTION: click [6276]  
</step_1_interaction>  
CURRENT OBSERVATION: (omitted)

Figure F: A prompt example of user_message for AgentOccam.

# References

[1] Google AI. Gemini 2.5: Our most intelligent ai model, 2025. URL https://blog.google/ technology/google-deepmind/gemini-model-thinking-updates-march-2025/ #gemini-2-5-thinking. Accessed: 2025-04-01.   
[2] Anthropic. Claude 3.7 sonnet system card. Technical report, Anthropic, 2024. URL https://assets.anthropic.com/m/785e231869ea8b3b/original/ claude-3-7-sonnet-system-card.pdf. Accessed: 2025-03-14.   
[3] Léo Boisvert, Megh Thakkar, Maxime Gasse, Massimo Caccia, Thibault de Chezelles, Quentin Cappart, Nicolas Chapados, Alexandre Lacoste, and Alexandre Drouin. Workarena++: Towards compositional planning and reasoning-based common knowledge work tasks. In NeurIPS, 2024.   
[4] Hyungjoo Chae, Namyoung Kim, Kai Tzu-iunn Ong, Minju Gwak, Gwanwoo Song, Jihoon Kim, Sunghwan Kim, Dongha Lee, and Jinyoung Yeo. Web agents with world models: Learning and leveraging environment dynamics in web navigation. In ICLR, 2025.   
[5] Jingxuan Chen, Derek Yuen, Bin Xie, Yuhao Yang, Gongwei Chen, Zhihao Wu, Li Yixing, Xurui Zhou, Weiwen Liu, Shuai Wang, et al. Spa-bench: A comprehensive benchmark for smartphone agent evaluation. In ICLR, 2025.   
[6] De Chezelles, Thibault Le Sellier, Maxime Gasse, Alexandre Lacoste, Alexandre Drouin, Massimo Caccia, Léo Boisvert, Megh Thakkar, Tom Marty, Rim Assouel, et al. The browsergym ecosystem for web agent research. arXiv preprint arXiv:2412.05467, 2024.   
[7] Brandon Chiou, Mason Choey, Mingkai Deng, Jinyu Hou, Jackie Wang, Ariel Wu, Frank Xu, Zhiting Hu, Hongxia Jin, Li Erran Li, Graham Neubig, Yilin Shen, and Eric P. Xing. Reasoneragent: A fully open source, ready-to-run agent that does research in a web browser and answers your queries, February 2025. URL https://reasoner-agent.maitrix.org/.   
[8] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Sam Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. In NeurIPS, 2023.   
[9] Alexandre Drouin, Maxime Gasse, Massimo Caccia, Issam H Laradji, Manuel Del Verme, Tom Marty, Léo Boisvert, Megh Thakkar, Quentin Cappart, David Vazquez, et al. Workarena: How capable are web agents at solving common knowledge work tasks? In ICLR, 2024.   
[10] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. In ACL, 2024.   
[11] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.   
[12] Jace AI. Jace ai, 2025. URL https://jace.ai/. Accessed: 2025-03-14.   
[13] Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. In ACL, 2024.   
[14] Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. Tree search for language model agents. arXiv preprint arXiv:2407.01476, 2024.   
[15] Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. Reinforcement learning on web interfaces using workflow-guided exploration. In ICLR, 2018.   
[16] Xiao Liu, Tianjie Zhang, Yu Gu, Iat Long Iong, Yifan Xu, Xixuan Song, Shudan Zhang, Hanyu Lai, Xinyi Liu, Hanlin Zhao, et al. Visualagentbench: Towards large multimodal models as visual foundation agents. arXiv preprint arXiv:2408.06327, 2024.   
[17] Sami Marreed, Alon Oved, Avi Yaeli, Segev Shlomov, Ido Levy, Aviad Sela, Asaf Adi, and Nir Mashkif. Towards enterprise-ready computer using generalist agent. arXiv preprint arXiv:2503.01861, 2025.

[18] Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. In ICLR, 2024.   
[19] Magnus Müller and Gregor Žunic. Browser use: Enable ai to control your browser, 2024. URL ˇ https://github.com/browser-use/browser-use. Accessed: 2025-04-01.   
[20] Dang Nguyen, Jian Chen, Yu Wang, Gang Wu, Namyong Park, Zhengmian Hu, Hanjia Lyu, Junda Wu, Ryan Aponte, Yu Xia, et al. Gui agents: A survey. arXiv preprint arXiv:2412.13501, 2024.   
[21] OpenAI. Computer-using agent: Introducing a universal interface for ai to interact with the digital world. 2025. URL https://openai.com/index/computer-using-agent. Accessed: 2025-03-14.   
[22] Yichen Pan, Dehan Kong, Sida Zhou, Cheng Cui, Yifei Leng, Bing Jiang, Hangyu Liu, Yanyi Shang, Shuyan Zhou, Tongshuang Wu, et al. Webcanvas: Benchmarking web agents in online environments. arXiv preprint arXiv:2406.12373, 2024.   
[23] Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Wenyi Zhao, Yu Yang, Xinyue Yang, Jiadai Sun, Shuntian Yao, et al. Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning. In ICLR, 2025.   
[24] Christopher Rawles, Sarah Clinckemaillie, Yifan Chang, Jonathan Waltz, Gabrielle Lau, Marybeth Fair, Alice Li, William Bishop, Wei Li, Folawiyo Campbell-Ajala, et al. Androidworld: A dynamic benchmarking environment for autonomous agents. In ICLR, 2025.   
[25] Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. World of bits: An open-domain platform for web-based agents. In ICML, 2017.   
[26] Yueqi Song, Frank Xu, Shuyan Zhou, and Graham Neubig. Beyond browsing: Api-based web agents. arXiv preprint arXiv:2410.16464, 2024.   
[27] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. In ICML, 2025.   
[28] Webarena Team. Webarena Leaderboard. https://docs.google.com/spreadsheets/d/ 1M801lEpBbKSNwP-vDBkC_pF7LdyGU1f_ufZb_NWNBZQ. Accessed: 2025-04-01.   
[29] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. arXiv preprint arXiv:2504.12516, 2025.   
[30] Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Deyu Zhou, Pengjun Xie, and Fei Huang. Webwalker: Benchmarking llms in web traversal. arXiv preprint arXiv:2501.07572, 2025.   
[31] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Jing Hua Toh, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. In NeurIPS, 2024.   
[32] Tianci Xue, Weijian Qi, Tianneng Shi, Chan Hee Song, Boyu Gou, Dawn Song, Huan Sun, and Yu Su. An illusion of progress? assessing the current state of web agents. arXiv preprint arXiv:2504.01382, 2025.   
[33] Ke Yang, Yao Liu, Sapana Chaudhary, Rasool Fakoor, Pratik Chaudhari, George Karypis, and Huzefa Rangwala. Agentoccam: A simple yet strong baseline for llm-based web agents. In ICLR, 2025.   
[34] Ori Yoran, Samuel Joseph Amouyal, Chaitanya Malaviya, Ben Bogin, Ofir Press, and Jonathan Berant. Assistantbench: Can web agents solve realistic and time-consuming tasks? arXiv preprint arXiv:2407.15711, 2024.   
[35] Ziniu Zhang, Shulin Tian, Liangyu Chen, and Ziwei Liu. Mmina: Benchmarking multihop multimodal internet agents. arXiv preprint arXiv:2404.09992, 2024.

[36] Shuyan Zhou, Frank F Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, et al. Webarena: A realistic web environment for building autonomous agents. In ICLR, 2024.   
[37] Andrew Zhu, Alyssa Hwang, Liam Dugan, and Chris Callison-Burch. Fanoutqa: A multi-hop, multi-document question answering benchmark for large language models. In ACL, 2024.
# Tiannan Wang1*, Meiling $\mathbf { T a o } ^ { 2 * }$ , Ruoyu Fang3, Huilin Wang4, Shuai Wang5 Yuchen Eleanor Jiang1, Wangchunshu Zhou1,

1OPPO AI Center, 2Guangdong University of Technology

3University of Illinois at Urbana-Champaign, 4Beihang University, 5Tsinghua University

# Abstract

In this work, we introduce the task of lifelong personalization of large language models. While recent mainstream efforts in the LLM community mainly focus on scaling data and compute for improved capabilities of LLMs, we argue that it is also very important to enable LLM systems, or language agents, to continuously adapt to the diverse and ever-changing profiles of every distinct user and provide upto-date personalized assistance. We provide a clear task formulation and introduce a simple, general, effective, and scalable framework for life-long personalization of LLM systems and language agents. To facilitate future research on LLM personalization, we also introduce methods to synthesize realistic benchmarks and robust evaluation metrics. We will release all codes and data for building and benchmarking life-long personalized LLM systems.

# 1 Introduction

The advent of large language models (LLMs) that can effectively understand and generate human language (Radford et al., 2018, 2019; Brown et al., 2020; Ouyang et al., 2022; OpenAI, 2023; Touvron et al., 2023a,b) has marked a transformative era in artificial intelligence. LLMs such as ChatGPT, Gemini, and Claude have shown remarkable capabilities, not only as general chatbot providing useful responses but also as agents (Zhou et al., 2023a, 2024; Chen et al., 2023; Liang et al., 2024b,a) that assist hundreds of millions of users in diverse realworld tasks. Mainstream efforts in the AI/LLM community have been focusing on scaling data and compute on pre-training or post-training stages for building stronger LLMs that can better serve the needs of users. Millions of GPUs and billions of dollars are consumed every year for this purpose.

However, a crucial question is: does satisfactory user experience naturally emerges with improved capabilities of LLMs? While it is true that improved reasoning abilities help LLMs solve more complex tasks and improved instruction-following abilities enable LLMs to better understand user intents, current LLM systems are by design incapable of capturing diverse and ever-changing personal profiles of distinct users that are implicitly encoded in life-long human-AI interaction histories. For example, an AI assistant must have rich information of a user’s residential address, personal agenda, income and consumption habits, preferences for foods and restaurants, etc., to generate a satisfactory response for a query as simple as "help me reserve a restaurant for dinner." Moreover, many aspects, including personalities, intents, preferences, etc., in user profiles are dynamic and ever-changing in the real world, making it crucial for LLM systems to be capable of constantly adjusting to the changes of user profiles. Therefore, we argue that stronger capabilities of LLMs are not all we need for AGI and life-long personalization of LLMs is another important building block for AI systems that are helpful, satisfactory, and engaging for everyone.

While personalization has been carefully investigated in other domains such as recommendation systems (Barkan et al., 2020; Wo´zniak et al., 2024; Dai et al., 2024; Li et al., 2024; Yang et al., 2023b; Kang et al., 2023), research on LLM personalization is quite limited and suffers from three major limitations: First, most recent work on LLM personalization (Mysore et al., 2023; Zhiyuli et al., 2023; Yang et al., 2023a) are task-specific and the methodology for LLM personalization can not be generalized to other tasks; Second, most existing LLM personalization methods requires training either the entire LLM or a few blocks within the LLM, making them impossible or very costly to scale to real-world applications used by million

of users on a daily basis; Finally, the lack of diverse and realistic benchmarks makes it hard to evaluate new methods. To be specific, most recent work conducts experiments on the LaMP benchmark (Salemi et al., 2024b). However, the tasks in this benchmark such as citation identification and scholarly title generation are very different from queries asked by real-world users and therefore not representative enough. Moreover, recent study (Salemi et al., 2024a) has shown that a nonpersonalized LLMs can also achieve competitive performance on these tasks, suggesting that the benchmark is not suitable for evaluating advanced LLM personalization approaches. In addition, all previous studies consider LLM personalization as a one-time task. However, it is crucial for LLM systems to constantly adjusting to ever-changing user profiles in real-world applications by learning from human-AI interactions.

In this paper, we introduce the task of life-long personalization of LLMs and provide a detailed task formulation emphasizing the difference from previous static LLM personalization problems. We present AI PERSONA, a simple, general, effective, and scalable framework to build life-long personalized LLM systems or language agents. Specifically, we define user profiles as learnable dictionaries where the keys are fields representing various aspects of a real-world user, including demographics, personality, usage patterns, and preferences. The values are the users’ personal information in the corresponding fields. During inference, a user’s AI persona will be dynamically assembled into a part of the prompt/input to the LLM backbone so that it will generate personalized responses. Lifelong adaptation of user profiles is achieved by a carefully designed LLM-based persona optimizer, which constantly adjust the AI Persona of the user during the interaction between the user and the LLM system. To the best of our knowledge, the proposed framework is the first method in the literature that can constantly adjust the profiles of each users during the progress of human-AI interaction. Moreover, our approach does not involve model training and only requires to store a lightweight config file for each user, making it scalable for real-world applications with large amount of users.

To test the effectiveness of the proposed AI PERSONA framework and facilitate future research, we build PERSONABENCH, a benchmark for (lifelong) LLM personalization research. PERSON-ABENCH consists of diverse and realistic user pro-

files and user queries generated with a carefully designed LLM-based workflow. Our experiments on PERSONABENCH demonstrates that the proposed AI PERSONA framework2 can effectively learn and adapt to user profiles over time.

Our contributions can be summarized as follows:

1. We provide a clear definition of life-long personalization, emphasizing the necessity of continuous adaptation in understanding user needs. To our knowledge, this is the first work to address this task in the literature.   
2. We propose a novel pipeline for generating realistic persona chat data, encompassing diverse persona configurations and authentic user-agent interaction simulations.   
3. We introduce a life-long personalized agent framework, serving as a baseline solution for this task.   
4. We conduct experiments that demonstrate the effectiveness of our methods, showcasing significant improvements in agent performance and adaptability to user personas.

By addressing these aspects, our work aims to advance the field of personalized conversational agents, fostering more effective and meaningful user interactions.

# 2 Related Work

# 2.1 Personalized LLMs

The personalization of language models (LLMs) has garnered significant interest in industries such as recommendation systems and search, focusing on providing tailored responses that adapt to individual user preference (Barkan et al., 2020; Wo´zniak et al., 2024; Dai et al., 2024; Li et al., 2024; Yang et al., 2023b; Kang et al., 2023; Tan and Jiang, 2023; Tao et al., 2023). Recently, this focus has extended to other domains such as research assistants (Wang et al., 2024; Lin et al., 2024), travel planners (Xie et al., 2024), writing assistants (Mysore et al., 2023), book recommending (Zhiyuli et al., 2023), shopping counselors (Yang et al., 2023a), and programming agents (Gao et al., 2023).

A common approach involves fine-tuning personalized LLMs. Zhou et al. (2023b) integrate persona prediction with response generation, while Tan et al. (2024b) use LoRA (Hu et al., 2021) to fine-tune Llama (Touvron et al., 2023a) for individual users. To further enhance efficiency, Zhuang et al. (2024) and Tan et al. (2024a) group users and fine-tune LoRA at the group level.

Despite these advancements, fine-tuning approaches face limitations in maintaining adaptability over time due to the need for frequent retraining, which is impractical in real-world scenarios. To address this, RAG-based personalized LLMs, offer an alternative by leveraging user-specific historical data. For example, Salemi et al. (2024b) introduced a pseudo-RAG approach for incorporating user history, later enhanced by Salemi et al. (2024a) through retriever optimization. Similarly, Li et al. (2023) and Mysore et al. (2023) propose retrievalbased methods to integrate user-authored documents for prompt augmentation. However, the input length constraints still hinder effective personalization when user interactions are lengthy. To address this, some studies have utilized comprehensive user histories to generate summaries based on user interactions (Christakopoulou et al., 2023; Zhiyuli et al., 2023; Richardson et al., 2023). While significant progress has been made, existing approaches rarely consider realistic, life-long adaptation scenarios, leaving room for further exploration in dynamic, long-term personalization.

# 2.2 Benchmarking Personalized LLMs

While benchmarks for LLM agents have been extensively developed (Shen et al., 2023; Mialon et al., 2023; Liu et al., 2023), few focus on personalized agents. A significant challenge in this area is the scarcity of realistic data. Existing benchmarks, such as LaMP (Salemi et al., 2024b), utilize public datasets with user identifiers, offering limited user-data associations. LaMP comprises seven tasks, primarily binary classification and singleinstance generation, such as Personalized Citation Identification (choosing which paper is likely to be cited) and Personalized Scholarly Title Generation (generating a title from an abstract). However, its minimal reliance on historical user data limits its ability to evaluate true personalization. Notably, these task designs are insufficient since a large language model (LLM) could achieve competitive performance without access to any historical user data about $u$ (Salemi et al., 2024a).

To overcome these limitations, our proposed data synthesis pipeline generates a broad spectrum of persona configurations and user-agent interactions, simulating realistic conversational scenarios that evolve. This approach not only supports the development of more adaptable agents but also establishes a comprehensive benchmark for evaluating life-long personalization capabilities. By dynamically generating data that reflects diverse user behaviors and preferences, we provide a more robust framework for training and testing personalized LLMs in environments that closely resemble realworld usage.

# 3 AI Persona: Towards Life-long Personalized LLMs

# 3.1 Task Formulation

We will provide a clear definition and task formulation of Life-long personalized LLM in this section. Consider a model or an agent, noted $P$ , which takes the input query $x$ from a user $u$ . A personalized agent should conditioning on not only $x$ but also the profile of $u$ in order to generate satisfactory responses that is tailored to this specific $u$ . In previous setting, particularly in LaMP (Salemi et al., 2024b), the profile of $u$ , $P _ { u }$ , is defined as a set of user’s historical data, i.e., the past input and personalized outputs produced by or approved by the user. That is $P _ { u } =$ $\{ ( x _ { u 1 } , y _ { u 1 } ) , ( x _ { u 2 } , y _ { u 2 } ) , \ldots , ( x _ { u k } , y _ { u k } ) \}$ . Through out the evaluation, the $P _ { u }$ is fixed for each $u$ which makes the persona profile static. In this work, we re-define the user profile as learnable dictionaries where the keys are fields representing various aspects of a real-world user and the values are the users’ personal information or traits in the corresponding field.

$$
P _ {u} = \{(k _ {1}, v _ {u 1}), (k _ {2}, v _ {u 2}), \ldots , (k _ {n}, v _ {u n}) \},
$$

where $k _ { i }$ represents the $i$ -th field of user attributes such as demographics, personality, usage patterns, and preferences, and $v _ { u i }$ denotes the corresponding value reflecting user $u$ ’s information in that field.

The learnable aspect of $P _ { u }$ means that each value $v _ { u i }$ is dynamically updated based on the ongoing interactions between the user and the agent $P$ . Mathematically, this is modeled as:

$$
v _ {u i} ^ {(t)} = f _ {\theta} (v _ {u i} ^ {(t - 1)}, (x _ {t}, y _ {t})),
$$

![](images/60e797b0efce14aec6f2b190403ffcfc0e3e5aa9e75775c7b61577f69bfbe7e4.jpg)  
Figure 1: Data generation pipeline for PersonaBench. This pipeline consists of 5 stages: seed data collection, persona synthesis, scene generation, personalized query generation and data filtering and refinement.

where $f _ { \theta }$ is the persona optimizer parameterized by $\theta$ , and $( x _ { t } , y _ { t } )$ is the interaction data at time step t. Note that $f _ { \theta }$ can either be a learnable model with trainable parameters or a pre-trained LLM-based agent guided by well-designed prompts. In this work, we opt for the latter approach, keeping $\theta$ fixed and utilizing the LLM’s emergent abilities to adapt through prompting rather than parameter updates. This formulation allows $P _ { u }$ to continuously evolve, capturing the latest user behaviors and preferences in real-time.

In contrast to static representations, our approach ensures that the user profile remains up-to-date and context-aware, enabling the model to adapt its responses more accurately to the evolving characteristics of the user in a longer life-span.

# 3.2 Data Generation Pipeline

The most challenging aspect of personalization is the scarcity of realistic user data and the difficulty of accessing it. To tackle this problem, we propose a persona chat data generation pipeline, as shown in Figure1, to synthesize real persona profile and generate user-agent conversation data according to each aspect of the persona profile.

# 3.2.1 Persona Generation

First we pre-define the necessary fields that a comprehensive persona profile should contain, which are demographics, personality, usage patterns, and preferences.

• Demographics: This field captures key factual information about the user’s identity and background, including age, gender, nationality, language, and career information or education background. It provides a basic understanding of the user.

• Personality: This field defines the user’s psychological characteristics and values, reflecting how they typically think, feel, and behave. In our setting, we represent it using MBTI (Myers-Briggs Type Indicator) and interests. Personality traits implicitly influence how users express themselves, respond to different conversational styles, and engage with the agent.

• Patterns: This field represents the user’s habits and interactions with the personalized agents, such as behavior engagement patterns, usage patterns and purchase pattern. Understanding usage patterns enables the agent to anticipate user needs and provide proactive support, thereby enhancing user experience and engagement.

• Preferences: This field encompass the user’s preferred interaction styles, formats, and workflows. Capturing preferences helps the agent to personalize responses and recommendations, making interactions more relevant and satisfying for the user.

To create more comprehensive and realistic persona profiles, we first ask volunteers from diverse backgrounds (e.g., different professions and life stages) who has a habit to regularly use AI products to complete persona profiles. Using these real personas as seed configurations, we then prompt the LLM to summarize these profiles into brief descriptions as seed hints. With the seed hints and seed configurations in place, we instruct the LLM to generate diverse persona descriptions in a selfinstruct manner (Wang et al., 2023a) to generate a large amount of persona hints, then using the seed configurations as in-context exemplars to guide the model in producing diverse, comprehensive and realistic persona data. 3

# 3.2.2 Scene generation

In order to synthesize realistic user-agent chat data, a critical component is to generate realistic and personalized queries. Previous work only focus on chit-chat or role-play scenario (Jandaghi et al., 2023; Wang et al., 2023b). In contrast, our work

aims at more practical settings where personalized agent is involved in solving real-world problems.

To achieve realistic query generation, we believe it is important to provide the LLM comprehensive and contextual information besides the persona information. Therefore,we first identify and define several common scenes in which people might utilize AI as an assistant. By prompting LLM to generate more detailed scene descriptions according to different persona profiles, these common scenes are then adapted to various personas, resulting in personalized scene descriptions that better reflect the unique preferences and characteristics of different users. Next, we use these common scenes as in-context exemplars, prompting the LLM to generate persona-specific scene descriptions based on the given persona profile.

To achieve realistic query generation, we first identify and define several common scenes in which people might utilize AI as an assistant. These common scenes serve as a baseline and are then adapted to various personas, resulting in personalized scene descriptions that better reflect the unique preferences and characteristics of different users. Next, we use these common scenes as incontext exemplars, prompting the LLM to generate persona-specific scene descriptions based on the given persona profiles.

Contextual Scene Information We prompt the LLM to enrich the scene descriptions with additional contextual information that is tailored to both the persona and the specific scenario. For instance, in the context of a Job Seeking scene, the synthe sized description for Brandon, a virtual persona who is a master’s student specializing in computer vision and deep learning, might involve preparing for campus recruitment. The contextual complement for this scene would include specific topics relevant to Brandon’s field of study and career, such as mock interviews for CV engineer or hot questions in high-tech companies, etc. These detailed contextual elements help create a more realistic and personalized interaction, ensuring that the generated queries are both contextually relevant and aligned with the persona’s background and goals.

Function Call Generation One of the major limitations of current LLMs compared to fictional AI assistants like Jarvis in the Iron Man movies is their inability to interoperate in the real world autonomously. To address this gap and make our setting more practical, we incorporate function call

data into our benchmark. This is particularly important for scenarios where users expect the AI to execute specific tasks on their behalf, such as checking whether they should carry an umbrella today, planning the remaining budget for the month, or searching for job opportunities that match their profile. For each scene, we identify potential API functions that align with the user’s intended actions, providing the personalized LLM with a structured way to obtain relevant information or perform the desired tasks.

# 3.2.3 Personalized Query Generation

After establishing personalized scene descriptions and their contextual information, the next step is to generate realistic and contextually appropriate queries. To achieve this, we employ a user simulator capable of role-playing based on a given persona profile. The user simulator reads the persona profile, current scene description, along with its contextual information to produce relevant and nuanced queries. This approach moves beyond generic role-play or simple chit-chat, instead generating queries that are deeply rooted in realistic scenarios that users may encounter.

# 3.2.4 Data Filtering and Refinement

To avoid generating unanswerable or nonsensical queries, the model evaluates whether it can provide a reasonable response, filtering out any queries that fail this criterion. Additionally, to prevent the user simulator from directly revealing persona traits in the query, we conduct a data refinement procedure that neutralizes the query, retaining only the essential information and the intended purpose.

# 3.3 AI Persona Framework

Our proposed AI Persona framework is composed of three main components: a Historical Session Manager, a Tool Executor, and a Personalized Chatbot. Each component plays a critical role in enabling life-long personalized interactions with users.

Historical Session Manager The Historical Session Manager is responsible for managing and storing conversation histories across multiple sessions for different users. It provides a comprehensive record of user interactions, enabling the system to maintain context and continuity over time. Its core functionalities include initializing, loading, saving, and retrieving conversation sessions, ensuring that

![](images/6489a0c9042821d3a8176a59502a5e2012345773f35866cc9f7a1d5f31f424f6.jpg)  
Figure 2: AI Persona Framework.

the system can seamlessly recall past interactions to support coherent and context-aware responses.

Tool Executor The Tool Executor is a wellprompted LLM designed to simulate external API execution. It interprets function calls from the Personalized Chatbot and generates appropriate responses based on predefined API descriptions provided in the scene information. Its primary function is to generate realistic and contextually accurate function call responses, bridging the gap between the chatbot and external tools or databases.

Personalized Chatbot The Personalized Chatbot is a well-designed conversational agent that leverages user personas to deliver personalized and context-aware responses. It acts as a versatile agent, capable of dynamically adapting its behavior to align with the user’s evolving profile and current context. Its core functionalities include:

• Persona-based Interaction: The chatbot generates tailored responses based on the user’s persona and query, ensuring that each interaction is relevant and engaging.   
• Dynamic Persona Updates: The chatbot updates the user’s persona profile in real-time as the interaction progresses, reflecting any changes in user preferences, behaviors, or context.   
• Function Calling: When necessary, the chatbot initiates appropriate function call to external APIs associated with the current scene, enriching the response with precise and contextually appropriate information.

As shown in Figure 2, during inference, the framework operates in a sequential manner, integrating all three components to provide a coherent and personalized user experience. Whole process is illustrated in the Algo 1.

Step 1: Persona and Session Initialization When a new conversation begins, the Personalized Chatbot loads the user’s persona profile(if exists) to model the persona and context. Concurrently, the user simulator—acting as the user—interprets the current scene and formulates a query based on the persona attributes and the scene description. This step initiates the session’s conversation.

Step 2: Query and Response Generation Based on the integrated context and scene understanding, the Personalized Chatbot then generates a tailored response by considering the user’s persona and the query. It may also issue function calls to the Tool Executor if external data or actions are required to enhance the response and, if necessary, explicitly includes these function calls in the response. For example, in a job-seeking scenario, the chatbot might call an web_search API to look for the latest interview questions for a specific role, providing user with informative advices.

Step 3: Tool Execution and Information Integration When the chatbot issues a function call, the Tool Executor interprets the request and simulates the external API execution according to pre-defined API documentations. It then returns the generated response, which the chatbot incorporates into its final output to the user. This allows the chatbot to provide an informative and accurate response, seamlessly integrating external data into the conversation.

Step 4: Satisfaction Evaluation After receiving the response, the User Simulator conducts a satisfaction evaluation to determine whether the generated response meets the expectation. This evaluation is based on a pre-generated reference response, an abstract expectation that represents the ideal outcome for the given persona and scenario. The User Simulator reviews the current session’s conversation history and, by referencing both the persona configuration and the expected response, assesses whether the chatbot’s reply aligns with the user’s needs and objectives, as defined by the persona.

Step 5: Persona Update and Session Storage When user satisfaction is confirmed, this conversation session is deemed finished. The personalized chatbot will then updates the user’s persona profile if necessary. For example, if the user expresses new preferences or changes their goals, these updates are reflected in the persona profile. Practically, the

model updates the persona configuration after every $k$ sessions, allowing it to accumulate more interaction data before making adjustments. Finally, the Historical Session Manager saves the current session data, ensuring that all interactions are recorded for future reference.

This multi-step process enables the AI Persona Framework to provide nuanced and adaptive interactions that cater to the user’s individual needs, fostering a more engaging and personalized user experience.

Algorithm 1 AI Persona Framework Inference Process  
Input: Persona profile $P$ , Scene information $S$ , Personalized Chatbot $LM$ and User simulator $U$ # Initialization $H \leftarrow$ Conversation history [empty list] $T \leftarrow$ Tool Executor loads $S$ satisfied $\leftarrow$ False  
while not satisfied do  
# Query Generation $Q \leftarrow U.\text{get_query}(P, S, H)$ # Response Generation with tool executions $R \leftarrow LM.\text{get_response}(Q, P, H, T)$ # Append query-response pair to chat history $H \leftarrow H \cup \{(Q, R)\}$ # Satisfaction Check  
satisfied $\leftarrow$ U.satisfaction_check(P, R, H)  
end while  
# Persona Update and Chat History Storage  
Update $P$ and save $H$

# 4 Experiments

# 4.1 Experiment Setup

Benchmark Setting Our proposed benchmark, PERSONABENCH, is composed of 200 diverse persona profiles, each paired with 10 common scene settings and 10 persona-specific scene settings. For each persona, we randomly sample 3 to 5 different scenes and prompt the LLM to re-generate scene descriptions, contextual information, and potential function calls. This approach simulates a common scenario where users often ask similar questions about the same topics over time. By incorporating this data type, we aim to assess whether a personalized LLM can learn from previous sessions and improve performance when handling similar scenarios. Additionally, we pre-generate the initial

user queries and carefully hand-check each one to ensure a fair comparison. In total, we synthesized over 6,000 data points to construct the benchmark.

Baselines We compare our proposed AI Persona framework with 3 baselines:

• No Persona Access: The model generates responses without any access to the persona configuration, simulating a scenario where the model serves as an AI chatbot.   
• Golden Persona Access: The model is provided with access to the ground truth persona configuration during inference, enabling it to generate responses that are fully aligned with the user’s defined attributes.   
• Conversations RAG: The model could not maintain or learn the user’s persona but it could retrieve conversations in which there exist similar queries to generate responses according to historical interactions.

Evaluation Metrics We evaluate the models across several key dimensions to measure their ability to align with user personas and efficiently meet user needs:

• Persona Satisfaction: We use an LLM as a judge to score the first utterance of each session based on how well the generated responses solve the problem and how well they align with the user’s persona. This score reflects if a chatbot can instantly get users’ intentions.   
• Persona Profile Similarity: After the session ends, we evaluate the final saved persona profile by comparing it to the ground truth persona. This measure reflects how accurately the model has updated and maintained the persona throughout the interactions.   
• Utterance Efficiency: We measure how many utterances are required for the model to fully satisfy the user’s needs. Fewer utterances indicate better alignment and understanding of the user’s requirements, as the model can meet the user’s needs more efficiently with less backand-forth interaction.

Table 1: Performance of different persona settings.   

<table><tr><td rowspan="2">Setting</td><td colspan="2">Personalized Response</td><td rowspan="2">Persona Similarity</td><td rowspan="2">Utterance Efficiency</td></tr><tr><td>Helpfulness</td><td>Personalization</td></tr><tr><td>Conversations RAG</td><td>8.07</td><td>7.48</td><td>-</td><td>2.89</td></tr><tr><td>No Persona</td><td>7.96</td><td>7.35</td><td>-</td><td>2.24</td></tr><tr><td>Golden Persona</td><td>8.34</td><td>7.78</td><td>-</td><td>1.78</td></tr><tr><td>Persona Learning</td><td></td><td></td><td></td><td></td></tr><tr><td>-k=1</td><td>8.09</td><td>7.59</td><td>5.88</td><td>1.98</td></tr><tr><td>-k=3</td><td>8.29</td><td>7.63</td><td>6.07</td><td>1.81</td></tr><tr><td>-k=5</td><td>8.03</td><td>7.59</td><td>5.23</td><td>2.15</td></tr></table>

Table 2: Personalized response scores evaluated across base LLMs in three persona settings. The score in red denotes the improvement of Persona Learning over No Persona setting.   

<table><tr><td rowspan="2">Setting</td><td colspan="2">Golden</td><td colspan="2">No Persona</td><td colspan="2">Persona Learning</td></tr><tr><td>Helpful</td><td>Personal</td><td>Helpful</td><td>Personal</td><td>Helpful</td><td>Personal</td></tr><tr><td>GPT-4o (full bench)</td><td>8.34</td><td>7.78</td><td>7.96</td><td>7.35</td><td>8.29 △0.33</td><td>7.63 △0.28</td></tr><tr><td>GPT-4o-mini</td><td>8.14</td><td>7.61</td><td>8.06</td><td>7.38</td><td>8.26 △0.20</td><td>7.56 △0.18</td></tr><tr><td>Gemini-1.5-pro</td><td>8.16</td><td>7.93</td><td>8.17</td><td>7.37</td><td>8.27 △0.10</td><td>7.64 △0.27</td></tr><tr><td>Gemini-1.5-flash</td><td>8.03</td><td>7.65</td><td>7.58</td><td>7.24</td><td>8.07 △0.49</td><td>7.29 △0.05</td></tr><tr><td>Claude-1.5-sonnet</td><td>8.11</td><td>7.28</td><td>8.01</td><td>7.11</td><td>8.03 △0.02</td><td>7.20 △0.09</td></tr></table>

Model Setting In our experiments, we conduct persona learning experiments using PERSON-ABENCH under three distinct persona settings: No Persona, Golden Persona, and Persona Learning. The experiments were carried out using various proprietary LLMs, specifically the GPT series (gpt-4o, gpt-4o-mini), the Gemini series (gemini-1.5-prolatest, gemini-1.5-flash-latest), and Claude (claude-3.5-sonnet). To ensure a fair comparison, we use the same prompt template across all models.

To evaluate the effectiveness and robustness of our proposed method, we conducted an ablation study on a randomly selected subset of 10 personas from PERSONABENCH for all models except gpt-4o, which was evaluated on the entire benchmark.

# 4.2 Main Results

As shown in the top row of Table 1, the Golden Persona setting achieves the highest scores in both personalized response helpfulness (8.34) and personalization (7.78), representing the upper bound of performance for a personalized chatbot. In contrast, the No Persona setting serves as a bottom line, where the chatbot makes responses without any prior knowledge of the user’s persona. The Conversations RAG setting shows a slight improvement in both helpfulness and personalization but with significantly lower utterance efficiency. By examining the results, we observe that the retrieved

conversations occasionally cause the personalized chatbot to misinterpret the user’s current intent, leading to nuanced responses that may also confuse the user simulator.

Next, we present the results for the AI Persona framework (Persona Learning setting) in the bottom row of the table. Specifically, we experiment with different persona update frequencies $\mathit { k } = 1 , 3 , 5$ . Among these, we observe that updating the persona every 3 conversations yields the best results, with a personalized response helpfulness of 8.29 and a personalization score of 7.63, which are very close to the Golden Persona scores. The other two update frequency settings also show slight improvement compared to the No Persona setting and Conversation RAG setting, demonstrating the effectiveness of our AI Persona framework.

In terms of the Utterance Efficiency, we can see that in each update frequency, Persona Learning shows a remarkable improvement over the No Persona baseline. Specifically, the persona update frequency of every 3 conversations $( k = 3$ ) results in the best utterance efficiency, closely approaching the performance of the Golden Persona setting, indicating the ability of our AI Persona framework to generate relevant and succinct responses that tailored to the user’s intent in fewer turns.

In terms of persona similarity, the $k = 3$ setting exhibits the highest persona similarity score of

6.07, while the $k = 1$ and $k = 5$ settings achieve scores of 5.88 and 5.23, respectively. Notably, this comparison highlights an interesting finding: more frequent updates (as in the $k = 1$ setting) and more information in each update (as in the $k = 5$ setting) do not necessarily result in better learning outcomes. This finding emphasizes the importance of carefully selecting the learning frequency $k$ in the life-long personalization of LLMs.

![](images/493b54ab1270d733062237bb8a5ade41f3850400d43c1b2fb9865f5b3f26da69.jpg)  
Figure 3: Average number of utterances required per scene. The blueline represents Persona Learning, the orangeline represents Golden Persona, and the greenline represents No Persona. Lower average utterance counts indicate better performance, as it means the dialogue is more efficient and the model requires fewer turns to satisfy the user.

![](images/293d0707b9de0869c1f4f5852e34a7631e555139179f90788fe0810f5b7d0358.jpg)  
Figure 4: Average winning rate of the pair-wise comparison of Golden Persona and Persona Learning as the scene number increases.

![](images/43f0138920ea0d6ade5634730c3052bdef08403b553fa59dd2beb125da24d831.jpg)  
Figure 5: Average number of utterances for different model bases and persona settings.

# 4.3 Procedural Learning of Personalized LLMs

In this subsection, we illustrate how the performance evolves over time as user-agent interactions progress under different settings: No Persona Access, Ground Truth Persona Access, and Persona Update. The primary goal is to evaluate whether the personalized LLM can improve its responses as it engages in more conversations and accumulates more personal information of the user.

As depicted in Figure 3, the x-axis represents the number of sessions, while the y-axis denotes the number of utterances for the user-simulator to be satisfied. We plot three lines, each corresponding to one of the settings. The comparison demonstrates how the availability of persona information and updating mechanisms influences the model’s ability to generate more tailored responses over time.

The blue learning curve is the Persona Learning setting which showcases a notable improvement over No Persona. The line in the figure reflects a steady decrease in the number of utterances needed for satisfaction and the standard deviation over time as well, demonstrating the effectiveness of persona learning. Remarkably, the performance in the final few sessions under the Persona Learning setting approaches that of the Golden Persona setting, indicating the effectiveness of our proposed method. It shows that with just over ten updates, the model can learn and adapt to the user’s persona efficiently.

To further evaluate the effectiveness of our persona learning framework, we conducted a pairwise comparison of responses generated by the Persona Learning setting and the Golden Persona setting. Figure 4 presents the results, categorized by session groups (1-10, 11-20, 21-32). The win rate of the "Persona Learning" responses steadily increases as the session progresses, demonstrating that our AI persona framework effectively learns and adapts to the user’s persona over time.

This procedural learning analysis highlights the importance of dynamic persona modeling in personalized LLM systems, emphasizing the advantages of updating user profiles based on cumulative interactions.

# 4.4 Performance across various Base LLMs

The primary focus of Table 2 is to observe the performance of each model under different persona settings. We observe that our proposed Persona Learning method improves the performance across

all LLMs. Among them gpt-4o shows the best adaptability of personalization.

Figure 5 summarize the results, demonstrating the average number of utterances and the personalized response scores across different configurations.

The performance differences across base models illustrate that while all LLMs can benefit from persona learning, GPT-4o and Gemini-1.5-pro are better equipped with the ability to adapt and perform in personalized scenarios.

# 5 Conclusion

This paper introduces the task of life-long personalization for large language models (LLMs) and proposes the AI PERSONA framework, which enables scalable and dynamic adaptation to evolving user’s persona without requiring model retraining. We present PERSONABENCH, a synthesized but realistic and diverse benchmark for evaluating personalized LLMs. Experimental results demonstrate the effectiveness of our framework in improving personalized responses and maintaining updated user profiles. Our work provides a novel, generalizable, and efficient solution for continuous LLM personalization, addressing key limitations in existing approaches.

# 6 Limitations

Although our proposed AI PERSONA framework are designed to be language-agnostic, the seed data collection and annotation processes in this study were conducted by Chinese native speakers. As a result, the PERSONABENCH is more representative of scenarios and linguistic nuances specific to Chinese users. While our approach can theoretically generalize to other languages and cultural contexts, its current implementation and evaluation are better suited for Chinese language applications. Future work should involve expanding the data collection and annotation processes to include diverse linguistic and cultural backgrounds to fully validate the framework’s adaptability across different languages and user demographics.

# References

Oren Barkan, Yonatan Fuchs, Avi Caciularu, and Noam Koenigstein. 2020. Explainable recommendations via attentive multi-persona collaborative filtering. In Proceedings of the 14th ACM Conference on Recommender Systems, pages 468–473.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam Mc-Candlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. Preprint, arXiv:2005.14165.   
Weize Chen, Yusheng Su, Jingwei Zuo, Cheng Yang, Chenfei Yuan, Chi-Min Chan, Heyang Yu, Yaxi Lu, Yi-Hsin Hung, Chen Qian, Yujia Qin, Xin Cong, Ruobing Xie, Zhiyuan Liu, Maosong Sun, and Jie Zhou. 2023. Agentverse: Facilitating multiagent collaboration and exploring emergent behaviors. Preprint, arXiv:2308.10848.   
Konstantina Christakopoulou, Alberto Lalama, Cj Adams, Iris Qu, Yifat Amir, Samer Chucri, Pierce Vollucci, Fabio Soldo, Dina Bseiso, Sarah Scodel, et al. 2023. Large language models for user interest journeys. arXiv preprint arXiv:2305.15498.   
Yijia Dai, Joyce Zhou, and Thorsten Joachims. 2024. Language-based user profiles for recommendation. Preprint, arXiv:2402.15623.   
Difei Gao, Lei Ji, Zechen Bai, Mingyu Ouyang, Peiran Li, Dongxing Mao, Qinchen Wu, Weichen Zhang, Peiyi Wang, Xiangwu Guo, et al. 2023. Assistgui: Task-oriented desktop graphical user interface automation. arXiv preprint arXiv:2312.13108.   
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685.   
Pegah Jandaghi, XiangHai Sheng, Xinyi Bai, Jay Pujara, and Hakim Sidahmed. 2023. Faithful persona-based conversational dataset generation with large language models. arXiv preprint arXiv:2312.10007.   
Wang-Cheng Kang, Jianmo Ni, Nikhil Mehta, Maheswaran Sathiamoorthy, Lichan Hong, Ed Chi, and Derek Zhiyuan Cheng. 2023. Do llms understand user preferences? evaluating llms on user rating prediction. Preprint, arXiv:2305.06474.   
Cheng Li, Mingyang Zhang, Qiaozhu Mei, Yaqing Wang, Spurthi Amba Hombaiah, Yi Liang, and Michael Bendersky. 2023. Teach llms to personalize– an approach inspired by writing education. arXiv preprint arXiv:2308.07968.   
Yuanchun Li, Hao Wen, Weijun Wang, Xiangyu Li, Yizhen Yuan, Guohong Liu, Jiacheng Liu, Wenxing Xu, Xiang Wang, Yi Sun, Rui Kong, Yile Wang, Hanfei Geng, Jian Luan, Xuefeng Jin, Zilong Ye, Guanjing Xiong, Fan Zhang, Xiang Li, Mengwei Xu, Zhijun Li, Peng Li, Yang Liu, Ya-Qin Zhang, and

Yunxin Liu. 2024. Personal llm agents: Insights and survey about the capability, efficiency and security. Preprint, arXiv:2401.05459.   
Xuechen Liang, Meiling Tao, Yinghui Xia, Tianyu Shi, Jun Wang, and JingSong Yang. 2024a. Cmat: A multi-agent collaboration tuning framework for enhancing small language models. arXiv preprint arXiv:2404.01663.   
Xuechen Liang, Meiling Tao, Yinghui Xia, Tianyu Shi, Jun Wang, and JingSong Yang. 2024b. Self-evolving agents with reflective and memory-augmented abilities. arXiv preprint arXiv:2409.00872.   
Guanyu Lin, Tao Feng, Pengrui Han, Ge Liu, and Jiaxuan You. 2024. Paper copilot: A self-evolving and efficient llm system for personalized academic assistance. arXiv preprint arXiv:2409.04593.   
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Yuxian Gu, Hangliang Ding, Kai Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Shengqi Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. 2023. Agentbench: Evaluating llms as agents. ArXiv, abs/2308.03688.   
Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2023. Gaia: a benchmark for general ai assistants. arXiv preprint arXiv:2311.12983.   
Sheshera Mysore, Zhuoran Lu, Mengting Wan, Longqi Yang, Steve Menezes, Tina Baghaee, Emmanuel Barajas Gonzalez, Jennifer Neville, and Tara Safavi. 2023. Pearl: Personalizing large language model writing assistants with generation-calibrated retrievers. arXiv preprint arXiv:2311.09180v1.   
OpenAI. 2023. GPT4 technical report. arXiv preprint arXiv:2303.08774.   
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. Preprint, arXiv:2203.02155.   
Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training.   
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. 2019. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9.   
Chris Richardson, Yao Zhang, Kellen Gillespie, Sudipta Kar, Arshdeep Singh, Zeynab Raeesy, Omar Zia

Khan, and Abhinav Sethy. 2023. Integrating summarization and retrieval for enhanced personalization via large language models. arXiv preprint arXiv:2310.20081.   
Alireza Salemi, Surya Kallumadi, and Hamed Zamani. 2024a. Optimization methods for personalizing large language models through retrieval augmentation. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 752–762.   
Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. 2024b. Lamp: When large language models meet personalization. Preprint, arXiv:2304.11406.   
Yongliang Shen, Kaitao Song, Xu Tan, Wenqi Zhang, Kan Ren, Siyu Yuan, Weiming Lu, Dongsheng Li, and Yueting Zhuang. 2023. Taskbench: Benchmarking large language models for task automation. arXiv preprint arXiv:2311.18760.   
Zhaoxuan Tan and Meng Jiang. 2023. User modeling in the era of large language models: Current research and future directions. arXiv preprint arXiv:2312.11518.   
Zhaoxuan Tan, Zheyuan Liu, and Meng Jiang. 2024a. Personalized pieces: Efficient personalized large language models through collaborative efforts. arXiv preprint arXiv:2406.10471.   
Zhaoxuan Tan, Qingkai Zeng, Yijun Tian, Zheyuan Liu, Bing Yin, and Meng Jiang. 2024b. Democratizing large language models via personalized parameter-efficient fine-tuning. arXiv preprint arXiv:2402.04401.   
Meiling Tao, Xuechen Liang, Tianyu Shi, Lei Yu, and Yiting Xie. 2023. Rolecraft-glm: Advancing personalized role-playing in large language models. arXiv preprint arXiv:2401.09432.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. 2023a. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten,

Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288.   
Xintao Wang, Jiangjie Chen, Nianqi Li, Lida Chen, Xinfeng Yuan, Wei Shi, Xuyang Ge, Rui Xu, and Yanghua Xiao. 2024. Surveyagent: A conversational system for personalized and efficient research survey. Preprint, arXiv:2404.06364.   
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2023a. Self-instruct: Aligning language models with self-generated instructions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13484–13508, Toronto, Canada. Association for Computational Linguistics.   
Zekun Moore Wang, Zhongyuan Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou, Yuhan Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Man Zhang, et al. 2023b. Rolellm: Benchmarking, eliciting, and enhancing role-playing abilities of large language models. arXiv preprint arXiv:2310.00746.   
Stanisław Wo´zniak, Bartłomiej Koptyra, Arkadiusz Janz, Przemysław Kazienko, and Jan Kocon. 2024.´ Personalized large language models. Preprint, arXiv:2402.09269.   
Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and Yu Su. 2024. Travelplanner: A benchmark for real-world planning with language agents.   
Dongjie Yang, Ruifeng Yuan, Yuantao Fan, Yifei Yang, Zili Wang, Shusen Wang, and Hai Zhao. 2023a. Refgpt: Dialogue generation of gpt, by gpt, and for gpt. arXiv preprint arXiv:2305.14994.   
Fan Yang, Zheng Chen, Ziyan Jiang, Eunah Cho, Xiaojiang Huang, and Yanbin Lu. 2023b. Palr: Personalization aware llms for recommendation. Preprint, arXiv:2305.07622.   
Aakas Zhiyuli, Yanfang Chen, Xuan Zhang, and Xun Liang. 2023. Bookgpt: A general framework for book recommendation empowered by large language model. arXiv preprint arXiv:2305.15673.   
Wangchunshu Zhou, Yuchen Eleanor Jiang, Long Li, Jialong Wu, Tiannan Wang, Shi Qiu, Jintian Zhang, Jing Chen, Ruipu Wu, Shuai Wang, Shiding Zhu, Jiyu Chen, Wentao Zhang, Xiangru Tang, Ningyu Zhang, Huajun Chen, Peng Cui, and Mrinmaya Sachan. 2023a. Agents: An open-source framework for autonomous language agents.   
Wangchunshu Zhou, Qifei Li, and Chenle Li. 2023b. Learning to predict persona information for dialogue

personalization without explicit persona description. In Findings of the Association for Computational Linguistics: ACL 2023, pages 2979–2991, Toronto, Canada. Association for Computational Linguistics.   
Wangchunshu Zhou, Yixin Ou, Shengwei Ding, Long Li, Jialong Wu, Tiannan Wang, Jiamin Chen, Shuai Wang, Xiaohua Xu, Ningyu Zhang, Huajun Chen, and Yuchen Eleanor Jiang. 2024. Symbolic learning enables self-evolving agents.   
Yuchen Zhuang, Haotian Sun, Yue Yu, Qifan Wang, Chao Zhang, and Bo Dai. 2024. Hydra: Model factorization framework for black-box llm personalization. arXiv preprint arXiv:2406.02888.

# Api call prompt template (Chinese).

# System Prompt:

需要你 一个在终 上部署的AI助 ，你可以访问、 作终 上的应 ，以及进行我 模拟 端 手 操 端 用联网 索 。 给你输入一个API 时， 需要你去 一下真实的API来返回搜 等等 当用户 调用 我 模拟结果（也 是，假设这个API 被 行了，会 到什么样的结果）。

就 调用 执 拿下面是一些常见的API 的描述，希望你可以理解这些API的功能，方便你更 地这个API的结果。

# User Prompt:

{api_call}

# Api call prompt template (English).

# System Prompt:

I need you to simulate an AI assistant deployed on a terminal. You can access and operate terminal applications, as well as perform online searches. When the user provides you with an API call, I need you to simulate the actual API and return a result (i.e., assume the API call was executed and provide the kind of result it would return).

Below are descriptions of some common API calls. Please understand these API functionalities to better simulate the results of these APIs.

# User Prompt:

{api_call}

# Personalized chatbot API call prompt template(Chinese).

# System Instruction:

你是一个在终 上部署的AI助 ，除了像ChatGPT一样的 轮对话之 ，你还可以访端 手问、 作终 上的应 ，以及进行联网 索。

操场景描述：

{scene}

在这个场景中你可能需要访问 的API有：

{api_docs}

的人设信息和使 习惯 下：

用户{persona}

根据你对 前场景和 提问的理解，你可以 行决定是否需要访问 API。

当 用户 自 或调用果你认为要给出一个 的回 ，你 要 API，则请你显 地在回 中使 下如 好 复面的api 的例子来进行回 ：

调用{API_Example}

并且 这个回 放在最 。 果有部分内容是可以不访问/ API 可以回 的，将 复 开头 如 调用 就 答你可以先 可能回 问 ，再在最后补充类似于关于{具体需求}， 需要访问/ {具尽 答 题 我 调用体API名称} 能提供完整的 案。然后再参照api call的例子中的格 输出你的API call即可。

# User Prompt:

{query}

# Personalized chatbot API call prompt template (English).

# System Instruction:

You are an AI assistant deployed on a terminal. In addition to multi-turn conversations like ChatGPT, you can also access and operate applications on the terminal and perform online searches.

Scenario Description:

{scene}

In this scenario, you may need to access or call the following APIs:

{api_docs}

User persona information and usage habits:

{persona}

Based on your understanding of the current scenario and the user’s query, you can decide whether you need to access or call an API.

If you believe that providing a good response requires calling an API, you must explicitly include the API call in your response using the example format below:

{API_Example}

Make sure this response is placed at the very beginning. If part of the query can be answered without accessing/calling an API, try to answer that first. Then, at the end, add a statement such as regarding specific need, I need to access/call specific API name to provide a complete answer. Finally, include the API call in the example format as a supplement.

User Prompt:

{query}

# Persona Update prompt template(Chinese).

# System Instruction:

你是一个 于 像 取的AI机器人，你的任务是基于 的人设config和 最近使用 用户画 抽 用户AI助 的对话历史来判断人设的哪些字 （field）需要被更新。

用 手前的 像（人设config） 下:

当 用{persona}

注意， 需要你仔细斟 在上面的对话历史中，思考 的人设有没有哪 发 了改我 酌 用户 里变。你可以先理解人设、分析近期的对话，再来判断有哪些字 是需要被更新的。

最终的 更新的字 下的 输出：

<fields>

这 是具体要更新的字 和字 更新后的内容

里</fields>

下面是一个输出的例子:

{Fields_Update_Example}

User Prompt:

{chat_history}

# Persona Update prompt template (English).

# System Instruction:

You are an AI bot designed for extracting user personas. Your task is to determine, based on the user’s persona config and their recent conversation history with the AI assistant, which fields (if any) in the persona need to be updated.

Current user persona config:

```txt
{persona} 
```

Note: Please carefully consider the conversation history above and reflect on whether any changes have occurred in the user’s persona. You can start by understanding the persona and analyzing recent conversations before determining which fields need to be updated.

You can first understand the persona and analyze the recent conversation before determining which fields need to be updated.

The fields to be updated should be output in the following format:

```xml
<fields> 
```

Here are the specific fields that need updating and their updated content.

```twig
</fields> 
```

Below is an example of the output:

```txt
{Fields_Update_Example} 
```

# User Prompt:

```txt
{chat_history} 
```

# User Simulation prompt template (Chinese).

# System Prompt:

接下来 需要你 演一个真实的人类来和 人常 的AI助 进行 轮对话。 会给你一我 扮 此 用 手 多 我个persona setting，请你先理解这个persona的内容再 己完全代入到这个角色 中。

给你的 像 下：

用户画名：{name}

姓年龄：{age}

性别：{gender}

国籍：{nationality}

语 ：{language}

言职业信息：{career}

MBTI: {MBTI}

价值观与 ：{values}

爱好行为 像：{pattern}

画使 偏 ：{preference}

用 好前场景描述：

当{scene}。

还有些更有语境的场景信息可以供你参考，（注意！只是参考，不要全部使 ！）：

{scene_context}

下面是参考的例子:

{EXAMPLE}

注意事 ：1. 这 的AI不是一个 型AI（钢铁侠中的Jarvis那种什么都能做的 是项 里 概念 就 概念型AI），而是一个终 上的AI助 ， 们假设AI的功能 只有终 上的应 级 作、联端 手 我 就 端 用 操网 索和 常的 轮对话。2. 请你 记你的AI助 是非常了解你的， 以你无需在提问搜 正 多 牢 手 所时 你的人设 背景情况， 且仅 你发现AI助 给你的回 并不 合你的人设 场重申 或 当 当 手 复 符 或景设定时，你 能在后续的聊 中补充你的需求。3. 在 对话的时候 需要你能 真才 天 模拟 我 够实带入到AI助 的使 者的这个视角，语气不要 客气，同时不要主动的询问你的AI助手 用 太“你还需要哪些信息” ，要让AI助 来询问你，你再给出相应的信息。

手 等等 手一个可以供你参考的流 为：先 单描述一下问 ，并说出需要AI助 帮你做的事，程 简 题 手然后 AI给出 确的理解（ 果发现AI没有 确理解你的意图，则需要再 纠 ）之等待 正后，再 合AI给出相关信息。

配相信你已经get到了，现在，无 说任何 余的废话， 即进入你的角色！

我User Prompt:

{chat_history}

# User Simulation prompt template (English).

# System Prompt:

You will now play the role of a real human engaging in multi-turn conversations with their commonly used AI assistant. I will provide you with a persona setting. Please first understand the persona details and fully immerse yourself into this role.

User Persona Setting:

Name: {name}

Age: {age}

Gender: {gender}

Nationality: {nationality}

Language: {language}

Career Info: {career}

MBTI: {MBTI}

Values and Hobbies: {values}

Behavioral Traits: {pattern}

Usage Preferences: {preference}

Current Scenario Description: {scene}.

Additional contextual scene information is provided for reference (Note: Use it as reference only, do not fully adopt it!):

{scene_context}

The following is an example for reference: {EXAMPLE}

Important Notes: 1. This AI is not a conceptual AI (like Jarvis in Iron Man, which can handle everything), but rather a terminal-based assistant. Assume its functionalities are limited to terminal operations, online searches, and normal multi-turn dialogues. 2. Keep in mind that the AI assistant knows you very well. Thus, you don’t need to reiterate your persona or background when asking questions. Only if the AI’s response does not align with your persona or scenario should you provide additional clarifications. 3. When simulating the dialogue, fully immerse yourself in the perspective of the AI assistant user. Avoid being overly polite, and do not proactively ask the AI questions like, "What other information do you need?" Let the AI prompt you for additional details instead.

A typical flow to follow: First, briefly describe your issue and what you need the AI to do. Wait for the AI’s understanding (and correct its interpretation if needed). Then provide the necessary details to proceed.

I believe you’ve got it. Now, without saying anything unnecessary, immediately step into your role!

User Prompt:

{chat_history}

# Satisfaction check prompt template (Chinese).

# System Prompt:

接下来 需要你 演一个真实的人类， 会给你一个人设的persona，请你先理解这我 扮 我个persona的内容再 己完全代入到这个角色 中。

将给你的 像 下：

用户画名：{name}

姓年龄：{age}

性别：{gender}

国籍：{nationality}

语 ：{language}

言职业信息：{career}

MBTI: {MBTI}

价值观与 ：{values}

爱好行为 像：{pattern}

画使 偏 ：{preference}

用 好请你先理解上面给出的人设和场景，再根据你的理解去检查以下对话是否达到了“你”（即这个人设） 期的目标。

对话记 ：

录{chat_history}

下面是另一个Personalized Agent 对 前人设和场景给出的 期应 ，你可以 其作为参当考，但是最终对于是否 合要求还是 你来定 。

{expected_results}

注意：

你的输出 遵 以下原则： 果认为AI 助 回 的可以了， 输出且只输出：<满必须 循 如 手 答 就意>； 果你认为回 不 合要求的话，请你输出且只输出：<继续>。

如User Prompt:

{chat_history}

# User Simulation prompt template (English).

# System Prompt:

Next, I need you to play the role of a real human. I will provide you with a persona setting. Please first understand the persona’s content and fully immerse yourself in this role.

# User Persona Setting:

Name: {name}

Age: {age}

Gender: {gender}

Nationality: {nationality}

Language: {language}

Career Info: {career}

MBTI: {MBTI}

Values and Hobbies: {values}

Behavioral Traits: {pattern}

Usage Preferences: {preference}

Please first understand the above persona and scenario. Then, based on your understanding, evaluate whether the following conversation meets the expectations of "you" (i.e., this persona).

Conversation History:

{chat_history}

Below is the expected response given by another Personalized Agent for the current persona and scenario. You can use it as a reference, but ultimately, it is up to you to decide whether it meets the requirements.

{expected_results}

Note:

Your output must follow the principle below: If you think the AI assistant’s response is acceptable, output and only output: <Satisfied>. If you think the response does not meet the requirements, output and only output: <Continue>.

User Prompt:

{chat_history}

# Prompt Template (GPT Evaluation for Personalized Responses, Chinese).

# System Prompt:

接下来 会给你一个 和AI助 的对话以及这个 的人设Persona信息。 需要你我 用户 手 用户 我先理解 的Persona，再基于 这一轮的Query和AI助 给出的回 来 分，以评用户 用户 手估AI助 的回 在 度上解决了 问 中表达的 和需求。

手评分维度 下：

如1. 问 解决 度

题 程10分：AI助 完全满足了 需求，并提供了清晰、有效且有帮助的解决方案。

手 用户7分：AI助 部分满足了 需求，但可能存在一些不足，例 缺 方案的具体信手 用户息、API接口的 流 的 作 。

调用或 程 操 步骤5分：AI助 提供了一些与 需求相关的知识，但没有真 解决问 。

手 用户 正 题3分：AI助 提供的知识与 需求没有直接关系， 信息缺乏准确性。

手 用户 或0分：AI助 没有提供任何与 需求相关的解决方案 信息。

手2. 个性化 度

程10分：AI助 完全理解 的Persona，并根据 的 点和喜 ，提供了完整的个性化手的回 和建议。

复7分：AI助 在回 中体现了对 Persona的部分理解，但仍有进一 提升的 间。

手 复 用户 步5分：AI助 没有明显体现对 个性的理解，只是单纯地进行了对话型问 。

手 用户 答0分：AI助 对 Persona的理解错误，误解了 的意图，进行了错误的回 。

输出格 ：

式<analysis>

这 针对solution_score 和persona_score 可以做出分析。

里</analysis>

<rating>

solution_score; persona_score

</rating>

User Prompt:

以下是 像， 的Query和AI助 的回 ：

用户画 用像：{persona}

用户画Query：{query}

用户AI助 回 ：{answer}

# Prompt Template (GPT Evaluation for Personalized Responses, English).

# System Prompt:

I will now provide you with a conversation between a user and an AI assistant, along with the persona information of the user. You are required to first understand the user’s persona and then evaluate the AI assistant’s response based on the user’s query and the assistant’s response.

Evaluation Dimensions:

1. Solution Score   
10: The AI assistant fully meets the user’s needs and provides clear, effective, and helpful solutions.   
7: The AI assistant partially meets the user’s needs but has certain shortcomings, such as missing detailed information, API calls, or operational steps.   
5: The AI assistant provides some knowledge related to the user’s needs but does not solve the problem.   
3: The AI assistant provides knowledge unrelated to the user’s needs or lacks accuracy.   
0: The AI assistant provides no relevant solutions or information related to the user’s needs.   
2. Personalization Score   
10: The AI assistant fully understands the user’s persona and provides completely personalized responses and suggestions based on the user’s characteristics and preferences.   
7: The AI assistant shows partial understanding of the user’s persona in its response but has room for improvement.   
5: The AI assistant shows no clear understanding of the user’s persona and merely performs generic Q&A.   
0: The AI assistant misunderstands the user’s persona and provides incorrect responses that do not align with the user’s intent.

Output Format:

<analysis>

An analysis can be conducted on solution_score and persona_score.

</analysis>

<rating>

solution_score; persona_score

</rating>

# User Prompt:

Below are the user’s persona, query, and the AI assistant’s response:

User Persona: {persona}

User Query: {query}

AI Assistant Response: {answer}

# Prompt Template (GPT Evaluation for Persona Similarity, Chinese).

# System Prompt:

接下来 会给你一个 的两个Persona人设信息，一个是<ground_truth>人设，另一个我 用户是<learned_persona>人设。<learned_persona>是通过一 部的Persona Learning Pipeline基套外于 和AI助 在不同场景中的对话来建 的 。 需要你先阅读<ground_truth>用户 手 模 特征 我 里的Persona，并理解这个 的真实人设。再去给<learned_persona>的人设整体的一 性用户和人设的细节还原 度 分。

评分维度 下：

如1. 人设整体的一 性 分：  
致 打来评估 的人设是否在整体语义上与<ground_truth>保持一 ，包 语义的相关性性用 生成和不同字 内容的连贯性。  
段10分：整体语义完全一 ，行为连贯，无 义 偏差。  
致 歧 或7分：整体语义基本一 ，但存在 不 响理解的 差 。  
致 少量 影 小 异5分：整体语义部分一 ，但有明显差 ， 响整体理解。  
致 异 影3分：整体语义偏离较 ，信息传达困难 有较 矛盾。  
大 或0分：完全不一 ，语义严 背离目标人设。  
致2. 细节还原度 分：  
打来评估 型对于各个字 建 出来的是否准确、细 地还原了目标人设中的各个字用 模 段，包 关键 和相关细节。  
段 括 特征10分： 有字 的内容 度准确、具体，完整覆盖目标人设的 和细节。  
所 段 高 特征7分： 部分字 还原准确，细节基本到位，但有 字 显 糊 缺 。  
大 段 少量 段略 模 或5分：部分字 能体现 ，但整体还原不 深入，细节缺 较 。  
段 用户特征 够3分： 数字 还原度较低，内容 统， 字 描述存在错误。  
多 段 笼 或 段0分：字 内容与目标人设完全不 ， 细节缺 错误。

段输出格 ：

式<analysis>

这 针对<learned_persona>和<ground_truth>整体内容和个别字 内容可以做出分析。

里</analysis>

<rating>

consistency_score ; detail_restoration_score </rating>

# User Prompt:

下面是真实 人设：

用<ground_truth>

{persona_gt}

</ground_truth>

下面是通过人机交互的对话历史建 到的人设：

<learned_truth>

{persona_learned}

</learned_truth>

# Prompt Template (GPT Evaluation for Persona Similarity, English).

# System Prompt:

I will provide you with two persona profiles for a user: one is the <ground_truth>persona, and the other is the <learned_persona>persona. The <learned_persona>is modeled based on user-AI interactions in different scenarios through an external Persona Learning Pipeline. You need to first read the persona in <ground_truth>and understand the user’s true characteristics. Then, score the <learned_persona> for overall consistency and detail restoration.

Scoring Dimensions:

1. Overall Consistency Score:   
Used to evaluate whether the generated persona is semantically consistent with the <ground_truth> persona, including semantic relevance and coherence across different fields.   
10: Fully consistent in semantics, coherent behavior, no ambiguity or deviation.   
7: Mostly consistent in semantics, with minor differences that do not affect understanding.   
5: Partially consistent, with noticeable differences that affect overall understanding.   
3: Significant deviation in semantics, difficult to understand or conflicting information.   
0: Completely inconsistent, with severe semantic deviation from the target persona.   
2. Detail Restoration Score:

Used to evaluate whether the fields in the model accurately and thoroughly restore the target persona, including key characteristics and relevant details.

10: All fields are highly accurate, specific, and fully cover the target persona’s features and details.   
7: Most fields are accurately restored with basic details in place, but some fields are slightly vague or missing.   
5: Some fields reflect user characteristics, but overall restoration lacks depth, with many missing details.   
3: Most fields have low restoration accuracy, with vague content or incorrect descriptions.   
0: Field content does not match the target persona, with substantial missing or incorrect details.

Output Format:

<analysis>

Provide analysis on the overall content and specific fields of <learned_persona>compared to <ground_truth>.

</analysis>

<rating>

consistency_score ; detail_restoration_score

</rating>

# User Prompt:

Below is the ground-truth persona:

<ground_truth>

{persona_gt}

</ground_truth>

Below is the persona modeled through user-AI interaction:

<learned_truth>

{persona_learned}

</learned_truth>
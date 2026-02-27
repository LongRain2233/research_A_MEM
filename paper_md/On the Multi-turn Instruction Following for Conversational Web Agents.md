# On the Multi-turn Instruction Following for Conversational Web Agents

Yang Deng1∗, Xuan Zhang1∗, Wenxuan Zhang2, Yifei Yuan3, See-Kiong $\mathbf { N g } ^ { 1 }$ , Tat-Seng Chua1

1National University of Singapore, 2DAMO Academy, Alibaba Group, 3University of Copenhagen ydeng@nus.edu.sg xuanzhang@u.nus.edu

# Abstract

Web agents powered by Large Language Models (LLMs) have demonstrated remarkable abilities in planning and executing multi-step interactions within complex web-based environments, fulfilling a wide range of web navigation tasks. Despite these advancements, the potential for LLM-powered agents to effectively engage with sequential user instructions in real-world scenarios has not been fully explored. In this work, we introduce a new task of Conversational Web Navigation, which necessitates sophisticated interactions that span multiple turns with both the users and the environment, supported by a specially developed dataset named Multi-Turn Mind2Web (MT-Mind2Web). To tackle the limited context length of LLMs and the context-dependency issue of the conversational tasks, we further propose a novel framework, named self-reflective memory-augmented planning (Self-MAP), which employs memory utilization and self-reflection techniques. Extensive experiments are conducted to benchmark the MT-Mind2Web dataset, and validate the effectiveness of the proposed method.1

# 1 Introduction

A longstanding objective in artificial intelligence is to develop AI agents (Wooldridge and Jennings, 1995) that can execute complex tasks, thereby minimizing human effort in routine activities. With the advent of Large Language Models (LLMs), LLMpowered agents (Wang et al., 2023; Xi et al., 2023) showcase exceptional planning capabilities in performing multi-turn interactions with diverse environments, which contribute to various real-world problem-solving. As shown in Figure 1(a), the web agent (Deng et al., 2023; Zhou et al., 2024; Yao et al., 2022) is designed to interpret the states of a

![](images/ccf9e3df117ef5e714338f9e82ad27c6ea5a024e449cc4c1505fc076093dda84.jpg)

Instruction: Book a WWE ticket for price range between $50 to $100.

![](images/1d5134b9d17339713cfb52e8748287c60a44014e949fefe5c25777468bf177a0.jpg)

![](images/eb0cb1f97a49d5f84768fbcb130582b731813f1d940d232a33678a8630bb7f38.jpg)

![](images/aa7b2dd3a120b3f78ba7e7b2048218a74a9231a73149de4a564d9876a31d4712.jpg)

Action 1: [link] WWE Tickets $^ { - > }$ CLICK 5 Action 2: [link] TICKETS -> CLICK …

![](images/e90426db56da0916dde49e74445bee396b17a78e1de8c9c597262763075950f7.jpg)

# (a) Web Navigation

![](images/123374b26f8b163cf03d09eae0543a1b8f15fa25434460b91123984eca77d10f.jpg)

Instruction 1: Provide insights into the correlation between economic indicators such as GDP, … Instruction 2: Now, explain them again like I’m ve.

![](images/f1c4e39cacf9ec6b901d1718525352c2d92a18a9d5bfce2978ef51c1bf53d5af.jpg)

![](images/48213b10894ee422b670b1a5a390316d4b07a44b23be62dffcb39c23b3689a22.jpg)

# (b) Conversational Information Seeking

![](images/19a3acadd70cb0e7c9b7352e82983341f2442cc1e4307831a01a4b5ef937f9a3.jpg)

Instruction 1: Check the WWE tickets. Instruction 2: Book one by price range between $50 to $100. Instruction 3: Also Adele's concert ? tickets in Las Vegas on the last weekend of June.

![](images/b96fef1d73a4deb29bfe07076581ea460efc035784283a7dd0b2df204c38dbc3.jpg)

![](images/dfd5fd3df0b7ec0db229895e568a2ad445dadff79de9a7fe3f91da89c232dd87.jpg)

![](images/3bf4fd925dfebfc3f065ec2fbb39805fe9561b4136d85d3f392aed3d539b877e.jpg)  
Figure 1: Illustrations of different problems.

# (c) Conversational Web Navigation

webpage and execute a series of actions using keyboard and mouse inputs. Its purpose is to accomplish the tasks defined in natural language, such as booking tickets, through multi-turn interactions with the web-grounded environment.

Despite the proficiency in executing each individual instruction, the capability of interacting with multi-turn user instructions remains underexplored, which is crucial for applying LLMpowered agents onto real-world applications. As the example shown in Figure 1(c), during a conversational web navigation session, users tend to request follow-up or co-referencing instructions without repeating previous information. They may also provide a succinct or brief instruction, which is similar to other conversation problems. Motivated by recent efforts (Zheng et al., 2023a; Pan et al., 2023; Deng et al., 2024) on the investigation of conversational capabilities in the interactions with human users for LLMs, we propose a novel task, named Conversational Web Navigation. It requires the multi-turn interaction capabil-

ities with both users and environment. In particular, we introduce a new dataset, named Multi-Turn Mind2Web (MT-Mind2Web). MT-Mind2Web is constructed by using the single-turn interactions from Mind2Web (Deng et al., 2023), an expertannotated web navigation dataset, as the guidance to construct conversation sessions.

In other conversational tasks, LLMs can answer conversational questions (Zheng et al., 2023a) by utilizing their inherent knowledge from pretrained data or retrieval techniques to assess external databases (Figure 1(b)). Compared with these tasks, the conversation history in conversational web navigation contains both the previous useragent and agent-environment interactions, as the instruction completion relies on the dynamic environment status. Therefore, the history context can be much longer and noisier than that in the traditional conversation problems.

In light of these challenges, we propose a novel framework, named self-reflective memoryaugmented planning (Self-MAP). This framework is designed to maximize the utility of the limited memory space (i.e., input length limitation) of LLM-powered agents addressing the conversational web navigation problem. Specifically, we first construct a memory bank using the conversational interaction history, where each memory snippet stores each interaction step at each conversation turn. To reduce the noise from previous interactions, we propose a multifaceted matching approach to retrieve memory snippets that are semantically relevant and have similar trajectories. Furthermore, we design a reflection module to simplify the retrieved memory snippets by filtering out irrelevant information from the environment state. We then refine the retrieved memory snippets by generating reasoning rationales to enrich the memory information. Finally, we plan the next action by utilizing the self-reflective memory.

To sum up, our contributions are as follows:

• To study the multi-turn instruction-following capability of web agents, we define the problem of conversational web navigation and introduce a novel dataset, namely MT-Mind2Web.   
• We propose a self-reflective memory-augmented planning method (Self-MAP) that combines memory utilization and self-reflection for tackling the underlying challenges in the conversational web navigation task.

• We benchmark the MT-Mind2Web dataset with extensive baselines and provide comprehensive evaluations on different settings. Experimental results also validate the effectiveness of the proposed method.

# 2 Related Works

Web Agents Evolving from web agents with simplified environment simulation (Shi et al., 2017; Liu et al., 2018; Mazumder and Riva, 2021; Yao et al., 2022), recent studies investigate web navigation problems under more practical and complex settings, including multi-domain (Deng et al., 2023), real-time interactions (Zhou et al., 2024), and visual UI understanding (Zheng et al., 2024a). To handle these advanced web navigation problems, there has been increasing attention on building autonomous web agents powered by LLMs (Wang et al., 2023; Xi et al., 2023). Various promptbased methods have been proposed to enhance the LLM-powered web agents, such as recursive self-correction prompting (Kim et al., 2023), codebased prompting (Sun et al., 2023), and trajectoryaugmented prompting (Zheng et al., 2024b). However, prompt-based methods typically fail to compete with fine-tuned methods (Gur et al., 2024; Deng et al., 2023) in advanced settings, such as Mind2Web. In this work, we propose a new task, namely conversational web navigation, which requires multi-turn interaction capabilities with both users and the environment.

Multi-turn Interactions with Environment Interacting with the external environment enables LLM-powered agents to handle challenging tasks (Liu et al., 2024; Ma et al., 2024). For example, agents can interact with a code-grounded environment to access databases or perform programming (Xu et al., 2024; Hong et al., 2024), game-grounded environment to foster entertainment (Shridhar et al., 2021), web-grounded environment to navigate webpages (Deng et al., 2023) or perform online shopping (Yao et al., 2022). These works mainly focus on completing a standalone user instruction by planning a sequence of actions to interact with the environment. Some latest studies (Wang et al., 2024; Xie et al., 2023) investigate the utilization of multi-turn user feedback for solving a given task. In real-world applications, users may not always ask for the assistance for only a single task, while follow-up instructions and multi-turn requests are common during a conversation session.

![](images/0069be71a76f49c3f2d4a76a1bae57f89c0c4a63f092e1645e037dc3f4f60005.jpg)  
Figure 2: Overall pipeline for MT-Mind2Web creation with examples.

Multi-turn Interactions with Users Extensive studies demonstrate the exceptional capabilities of LLMs in seamless multi-turn interactions (Zheng et al., 2023a) with human users for completing various conversational tasks during a conversation session, such as recommendation (He et al., 2023; Huang et al., 2023), tutoring (Dan et al., 2023; Deng et al., 2024), counseling (Zheng et al., 2023b). For instance, MT-Bench (Zheng et al., 2023a) is one of the most popular benchmarks for evaluating the multi-turn instruction-following ability of LLMs. It consists of 80 high-quality multiturn questions ranging from 8 common instructionfollowing abilities, such as writing, roleplay, reasoning, etc. However, these conversational tasks mainly rely on the inherent knowledge of LLMs or just perform a one-time request from the external environment for each turn, such as conversational information seeking (Pan et al., 2023), without the need to access the dynamic environment for multiple times.

# 3 MT-Mind2Web Dataset

# 3.1 Annotation & Quality Control

Inspired by the typical construction process of existing conversation datasets, such as HybriDialogue (Nakamura et al., 2022) from OTT-QA (Chen et al., 2021), MMCoQA (Li et al., 2022) from MMQA (Talmor et al., 2021), and PACIFIC (Deng et al., 2022) from TAT-QA (Zhu et al., 2021),

we build the MT-Mind2Web dataset from the Mind2Web dataset (Deng et al., 2023) by using its single-turn interaction data as guidance for constructing conversation sessions. In order to reuse the expert-annotated action sequences in Mind2Web for ensuring the system response quality, the conversation construction process mainly focuses on the user instructions. In specific, the construction process contains three main steps:

1) Organize Conversation Sessions Given the same context, i.e., the same domain and website in Mind2Web, set up a conversation session with consecutive topics from multiple individual task instructions. Two instructions that share the same entities or intents are regarded as talking about the same topic. As the example in Figure 2, both the original Instruction 1 and Instruction 2 from Mind2Web are concerning about a ticket booking task upon the same Event domain and the same TicketCenter website, which can be naturally combined into a natural conversation session.   
2) Decompose Complex Instructions Some instructions in Mind2Web exhibit complex action sequences, which are not common in daily conversations. On the other hand, complex instructions can serve as a good starting point for constructing follow-up instructions in multi-turn interactions. To facilitate the decomposition of complex instructions, we employ human-AI collaborative annotation, since AI is more proficient in determining how

long action sequences can be divided into multiple executable sub-sequences while humans can decompose the instruction into multi-turn instructions in a more natural way. Specifically, we first employ ChatGPT for dividing the original instruction with complex action sequences into $N$ subtasks with corresponding action sub-sequences. Note that we set the target number of subtasks as $N = \lceil N ^ { \prime } / 4 \rceil$ , where $N ^ { \prime }$ is the number of actions in the original instruction. The prompt for instructing ChatGPT to decompose action sequences is as follows:

Analyze the instruction and corresponding actions provided for <domain> website, organize these actions into <N> distinct steps.

### Requirements

1. Review the instruction and related actions for completing a task on the specified website.   
2. Divide actions into logical, sequential steps.   
3. Format your response as a JSON array, with each object labeled as "step $i "$ and containing an array of the sequential numbers of the actions that belong to each step.

### Example

{ "step 1": [1, 2, 3], "step 2": [...], ... }

### Instruction

<original instruction>

### Actions

<original action sequences>

As the example in Figure 2, the Action Sequence 1 is sequentially decomposed into two action sub-sequences, including Action Subsequence 1-1 and Action Sub-sequence 1-2. Then human annotators are asked to verify whether these sub-tasks are reasonable and executable. If not, they can re-arrange the decomposition based on their experiences from navigating the webpages. Overall, the pass rate of ChatGPT in decomposing action sequences is $9 8 . 5 \%$ .

3) Rewrite Conversational Instructions We refine the original standalone instructions into conversational ones by using anaphora and ellipsis, especially when consecutive instructions within a conversation session involve the same entities or the same actions. For example, T2 uses one to refer to the WWE ticket mentioned in T1. While T3 shifts to another task with the same action of booking tickets, the verb book is omitted. Similarly, the repeated content in T3 is also omitted in T4.

Quality Verification To ensure the quality of annotation in MT-Mind2Web, we conduct quality ver-

Table 1: Statistics of the MT-Mind2Web dataset.   

<table><tr><td rowspan="2"></td><td rowspan="2">Train</td><td colspan="3">Test (Cross-X)</td></tr><tr><td>Task</td><td>Website</td><td>Subdomain</td></tr><tr><td># Conversations</td><td>600</td><td>34</td><td>42</td><td>44</td></tr><tr><td># Turns</td><td>2,896</td><td>191</td><td>218</td><td>216</td></tr><tr><td>Avg. # Turn/Conv.</td><td>4.83</td><td>5.62</td><td>5.19</td><td>4.91</td></tr><tr><td>Avg. # Action/Turn</td><td>2.95</td><td>3.16</td><td>3.01</td><td>3.07</td></tr><tr><td>Avg. # Element/Turn</td><td>573.8</td><td>626.3</td><td>620.6</td><td>759.4</td></tr><tr><td>Avg. Inst. Length</td><td>36.3</td><td>37.4</td><td>39.8</td><td>36.2</td></tr><tr><td>Avg. HTML Length</td><td>169K</td><td>195K</td><td>138K</td><td>397K</td></tr></table>

ification to validate the constructed conversations. If any mistake or problem is found, e.g., the constructed conversation is incoherent, the annotator will be asked to fix it until the annotation passes the verification.

# 3.2 Dataset Statistics

After the dataset creation, we obtain a total of 720 web navigation conversation sessions, which contain 3,525 corresponding instruction and action sequence pairs in total and an average of 5 turns of user-agent interactions in each conversation session. Following the evaluation settings in Mind2Web (Deng et al., 2023), we also select and divide the test set into three subsets, including cross-task, cross-website, and cross-subdomain, for evaluating how well an agent can generalize across tasks, websites, and domains. In specific, we select 44 samples for cross-subdomain evaluation from "Digital" and "Hotel", 42 samples for cross-website evaluation from "redbox", "viator", "nfl", "exploretock", "rentalcars", "cabelas", "bookdepository”, and 34 samples for cross-task evaluation. Then the remaining 600 samples are adopted as the training set. We present the train/test split in Table 1. Compared to traditional web navigation and conversational tasks, the conversational history can be extremely longer, including both the multi-turn user-agent conversation history and the multi-turn agent-environment interaction history within each conversation turn.

# 3.3 Problem Definition

We introduce the task of Conversational Web Navigation, where the agent engages in not only multi-turn interactions with the environment, but also conversational interactions with the user. Given the conversational interaction history $C _ { t } =$ $\{ q _ { 1 } , A _ { 1 } , . . . , A _ { t - 1 } , q _ { t } \}$ where $A _ { i } = \{ a _ { i } ^ { 1 } , a _ { i } ^ { 2 } , . . . , a _ { i } ^ { k } \}$ denotes the environment interaction history at each conversation turn, and the current environment state $E _ { t }$ (e.g., HTML of the current webpage),

![](images/26cc0df584b394d1fa6951bfb05c363ef6e059211992f77c0834e3c24a207ea6.jpg)  
Figure 3: Overview of Self-MAP.

the objective is to accurately predict the action sequence $A _ { t }$ to accomplish the current user instruction $q _ { t }$ , which encompasses the target element for interaction and the operation.

# 4 Method

We introduce the Self-MAP framework, which combines memory-augmented planning with selfreflection. The overview of Self-MAP is presented in Figure 3, consisting of three main components: Memory, Reflection, and Planning Modules.

# 4.1 Memory Module

The memory bank for conversational web agents is constructed by the conversational interaction history $C _ { t }$ , where each memory snippet can be represented by $M _ { t } ^ { k } = \{ q _ { t } , A _ { t } ^ { k - 1 } , E _ { t } ^ { k } , \bar { a _ { t } ^ { k } } \}$ . It requires a significant number of tokens to inject each memory snippet into the current running memory of the agent, which will be limited by the maximum input length of language models. Meanwhile, some memory snippets, due to the irrelevance and inconsistency of their instructions and actions to the current environment setting, fail to provide useful guidance for the agent to predict the subsequent action. As such, we introduce a multifaceted matching approach to retrieve the top- $K$ relevant snippets within the memory bank at the action level.

Formally, given an on-going conversational interaction trajectory $C _ { t } ^ { k } = \{ q _ { 1 } , A _ { 1 } , \ldots , q _ { t } , A _ { t } ^ { k - 1 } \} ,$ , where $A _ { t } ^ { k - 1 } = \{ a _ { t } ^ { 1 } , \tilde { a _ { t } ^ { 2 } } , \dots , \tilde { a _ { t } ^ { k - 1 } } \}$ ak−1t } represents the a trajectory of agent-environment interactions at the current conversation turn, multifaceted matching constructs the query using both the user instruction and the present agent action sequence $( q _ { t } , A _ { t } ^ { k - 1 } )$ to retrieve relevant memory snippets from the memory bank. In this manner, the query encodes not only the semantic relevance of the current instruction to the conversation context but also

the similarity of the action trajectory to the historical interactions. Specifically, we adopt OpenAI’s text-embedding-ada-002 as the embedding method to transform the query and the memory snippets into vector representations. Then we compute the cosine similarity in the embedding space for retrieving top- $K$ memory snippets.

# 4.2 Reflection Module

Due to the limitation of on-going memory space (i.e., input length limitation) for LLM-powered agents, we design a reflection module to maximize the utility of the limited memory space, which involves two steps: 1) Memory Simplification, and 2) Memory Refinement.

Memory Simplification In the candidate generation process in the MINDACT framework (Deng et al., 2023), a small pre-trained LM (e.g., De-BERTa (He et al., 2021)) is adopted for ranking the top- $. N$ candidate DOM elements from the environment state (i.e., HTML) that are related to the instruction and the current step for improving the final action prediction. To simplify each memory snippet, we apply the same process to remove taskirrelevant and noisy elements from the environment state, thereby freeing up memory space for more extensive conversation history retention. Afterwards, we denote the simplified environmental state $E _ { t } ^ { k }$ in the memory snippet $M _ { t } ^ { k }$ as $e _ { t } ^ { k }$ .

Memory Refinement Drawing inspiration from self-reflection techniques (Shinn et al., 2023; Asai et al., 2024), we design a specialized Memory Refinement approach for the domain of conversational web navigation. This module diverges from traditional self-reflection methods, as it does not collect incorrect trajectories for the model to analyze. This is primarily due to the constraints of a static evaluation setting and the limited context length to present the full webpage. Instead, we leverage the exceptional reasoning capability of LLMs to generate intermediate reasoning rationale as a supervised signal to enrich the memory information. For each retrieved memory snippet $( q _ { t } , A _ { t } ^ { k - 1 } , a _ { t } ^ { k } )$ , we prompt the LLM to generate an in-depth rationale $r _ { t } ^ { k }$ explaining the reason for the decision-making process of the next action.

Self-reflective Memory After the previous two steps, we obtain the self-reflective memory snippet, which not only filters out the irrelevant and noisy information from the environmental state but also

Table 2: Experimental results on MT-Mind2Web. TSR can be regarded as the main metric.   

<table><tr><td rowspan="2"></td><td colspan="4">Cross-Task</td><td colspan="4">Cross-Website</td><td colspan="4">Cross-Subdomain</td></tr><tr><td>Ele. Acc</td><td>Op. F1</td><td>SSR</td><td>TSR</td><td>Ele. Acc</td><td>Op. F1</td><td>SSR</td><td>TSR</td><td>Ele. Acc</td><td>Op. F1</td><td>SSR</td><td>TSR</td></tr><tr><td>DeBERTa (He et al., 2021)</td><td>36.8</td><td>-</td><td>-</td><td>-</td><td>31.7</td><td>-</td><td>-</td><td>-</td><td>27.7</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MINDACT (GPT-3.5) (Deng et al., 2023)</td><td>4.3</td><td>27.6</td><td>1.9</td><td>1.0</td><td>6.7</td><td>22.2</td><td>2.1</td><td>1.7</td><td>4.0</td><td>22.9</td><td>1.5</td><td>1.1</td></tr><tr><td>MINDACT (Flan-T5base) (Deng et al., 2023)</td><td>43.2</td><td>79.1</td><td>36.6</td><td>14.2</td><td>38.8</td><td>69.4</td><td>29.2</td><td>15.2</td><td>41.9</td><td>77.2</td><td>35.5</td><td>15.7</td></tr><tr><td>MINDACT + CAR (Anand et al., 2023)</td><td>47.8</td><td>78.8</td><td>41.4</td><td>16.1</td><td>37.0</td><td>67.5</td><td>32.2</td><td>9.6</td><td>41.2</td><td>75.3</td><td>35.4</td><td>13.2</td></tr><tr><td>MINDACT + Fixed (Huq et al., 2023)</td><td>51.0</td><td>80.8</td><td>42.6</td><td>18.4</td><td>42.4</td><td>70.0</td><td>35.4</td><td>15.3</td><td>43.1</td><td>77.6</td><td>37.5</td><td>17.7</td></tr><tr><td>Synapse (Zheng et al., 2024b)</td><td>49.6</td><td>79.9</td><td>41.9</td><td>18.4</td><td>43.1</td><td>70.6</td><td>33.1</td><td>13.7</td><td>41.7</td><td>77.8</td><td>35.9</td><td>16.0</td></tr><tr><td>Self-MAP</td><td>56.2</td><td>82.5</td><td>47.1</td><td>24.7</td><td>48.3</td><td>71.8</td><td>40.6</td><td>18.2</td><td>46.4</td><td>79.1</td><td>38.3</td><td>20.8</td></tr><tr><td>MINDACT (Flan-T5large) (Deng et al., 2023)</td><td>59.0</td><td>80.6</td><td>53.2</td><td>26.0</td><td>43.6</td><td>67.6</td><td>36.5</td><td>12.4</td><td>46.8</td><td>74.0</td><td>38.9</td><td>21.8</td></tr><tr><td>MINDACT + CAR (Anand et al., 2023)</td><td>54.5</td><td>79.5</td><td>47.8</td><td>19.8</td><td>43.2</td><td>69.2</td><td>36.1</td><td>12.2</td><td>44.5</td><td>75.0</td><td>40.2</td><td>15.6</td></tr><tr><td>MINDACT + Fixed (Huq et al., 2023)</td><td>58.0</td><td>79.7</td><td>51.3</td><td>26.4</td><td>46.2</td><td>69.7</td><td>37.6</td><td>15.2</td><td>47.4</td><td>74.9</td><td>38.8</td><td>21.4</td></tr><tr><td>Synapse (Zheng et al., 2024b)</td><td>57.5</td><td>82.0</td><td>50.0</td><td>23.2</td><td>45.1</td><td>69.0</td><td>37.1</td><td>13.0</td><td>47.4</td><td>74.1</td><td>39.3</td><td>19.4</td></tr><tr><td>Self-MAP</td><td>58.1</td><td>80.5</td><td>51.7</td><td>26.6</td><td>44.8</td><td>68.8</td><td>36.8</td><td>15.7</td><td>52.0</td><td>77.1</td><td>43.6</td><td>25.4</td></tr></table>

integrates the additional informative rationale. We denote the self-reflective memory snippet as $\hat { M } _ { t } ^ { k } =$ $\{ q _ { t } , A _ { t } ^ { k - 1 } , e _ { t } ^ { k } , a _ { t } ^ { k } , r _ { t } ^ { k } \}$ .

# 4.3 Planning with Self-reflective Memory

For each interaction step $k$ at the current conversation turn $t$ , given the current user instruction $q _ { t }$ and previous action sequences $A _ { t } ^ { k - 1 }$ , we first obtain the top- $K$ retrieved memory snippets with self-reflection $\mathcal { M } _ { t } ^ { k } = \{ \hat { M } \} ^ { K }$ from the reflection module, and the top- $. N$ candidate elements $e _ { t } ^ { k }$ simplified from the current environment state $E _ { t } ^ { k }$ using the same ranker as memory simplification. Then we fine-tune the LLM to plan the next action $a _ { t } ^ { k }$ including the target element and the operation, based on the input consisting of $( q _ { t } , A _ { t } ^ { k - 1 } , e _ { t } ^ { k } , \mathcal { M } _ { t } ^ { k } )$ . Similar to the settings in Deng et al. (2023), there are two types of planning paradigms: 1) Multi-choice Question Answering, and 2) Direct Generation.

# 5 Experiment

# 5.1 Experimental Setups

Baselines As conversational web navigation is a new task, we first adapt several state-of-the-art traditional web navigation methods as baselines, including DeBERTa (He et al., 2021), MINDACT (Deng et al., 2023), MINDACT $^ +$ Fixed (Huq et al., 2023), and Synapse (Zheng et al., 2024b). We further include a classic baseline for conversational tasks, i.e., MINDACT $^ +$ Context-Aware Rewriting (CAR) (Anand et al., 2023). Details of these baselines are presented in Appendix A.1.

Evaluation Metrics Following the single-turn setting (Deng et al., 2023), we adopt the following metrics for evaluation: 1) Element Accuracy (Ele. Acc) matches the selected element with all required

elements. 2) Operation F1 (Op. F1) stands for the token-level F1 score for the predicted operation. 3) Step Success Rate (SSR). An interaction step is regarded as successful only if both the selected element and the predicted operation are correct. 4) Turn Success Rate (TSR). A conversation turn is regarded as successful only if all steps at this turn have succeeded. We report macro average metrics, which first calculate the average per task, and then average over all tasks.

Implementation Details The overall Self-MAP framework basically follows the same configuration as MINDACT for a fair comparison. Specifically, we use the base version of DeBERTa-v3 (He et al., 2021) as the candidate HTML element ranker. We adopt the base and large versions of Flan-T5 (Chung et al., 2022) as the generation model to plan the next action. All the usage of ChatGPT in the experiments is based on gpt-3.5-turbo-1106. Flan- ${ \cdot } \mathrm { T } 5 _ { \mathrm { b a s e } }$ and Flan-T5large are trained on servers with 4 A5000 24GB GPUs. DeBERTa is trained with single A100 40GB GPU. More implementation details are presented in Appendix A.2.

# 5.2 Overall Evaluation

Experimental results on MT-Mind2Web are summarized in Table 2. Among the baselines, similar to the findings in Deng et al. (2023), DeBERTa, which only performs element selection, and MINDACT (GPT-3.5), which relies on the in-context learning capabilities of LLMs without fine-tuning, fall short of tackling the web navigation problem. For MINDACT+CAR, we observe that its performance is worse than the vanilla MINDACT (except for Cross-Task with Flan- $. T 5 _ { \mathrm { b a s e . } }$ ), where GPT-3.5 fails to effectively rewrite the current conversational instruction, which further obfuscates the original in-

Table 3: Ablation study. "w/o Generation-based Planning" denotes that we use MCQ-based Planning, while "w/o Multifaceted Matching" denotes that we prepend the chronological conversation context without retrieval.   

<table><tr><td rowspan="2"></td><td colspan="4">Cross-Task</td><td colspan="4">Cross-Website</td><td colspan="4">Cross-Subdomain</td></tr><tr><td>Ele. Acc</td><td>Op. F1</td><td>SSR</td><td>TSR</td><td>Ele. Acc</td><td>Op. F1</td><td>SSR</td><td>TSR</td><td>Ele. Acc</td><td>Op. F1</td><td>SSR</td><td>TSR</td></tr><tr><td>Self-MAP</td><td>56.2</td><td>82.5</td><td>47.1</td><td>24.7</td><td>48.3</td><td>71.8</td><td>40.6</td><td>18.2</td><td>46.4</td><td>79.1</td><td>38.3</td><td>20.8</td></tr><tr><td>w/o Generation-based Planning</td><td>51.7</td><td>79.4</td><td>43.5</td><td>22.2</td><td>43.1</td><td>69.5</td><td>34.9</td><td>15.5</td><td>44.8</td><td>77.2</td><td>37.3</td><td>17.7</td></tr><tr><td>w/o Memory Simplification</td><td>50.5</td><td>80.7</td><td>41.0</td><td>20.7</td><td>44.9</td><td>69.6</td><td>36.9</td><td>16.6</td><td>42.3</td><td>79.2</td><td>36.4</td><td>15.9</td></tr><tr><td>w/o Memory Refinement</td><td>52.1</td><td>81.3</td><td>43.0</td><td>23.2</td><td>48.9</td><td>70.8</td><td>39.1</td><td>18.1</td><td>46.3</td><td>78.7</td><td>37.2</td><td>17.8</td></tr><tr><td>w/o Multifaceted Matching</td><td>52.6</td><td>80.6</td><td>44.3</td><td>21.6</td><td>46.9</td><td>71.2</td><td>37.9</td><td>17.2</td><td>44.8</td><td>78.6</td><td>35.8</td><td>17.8</td></tr></table>

struction. In contrast, both MINDACT+Fixed and Synapse generally outperform MINDACT, which also validates our motivation of retrieving memory from the conversational interaction history. Surprisingly, Synapse (SOTA method in Mind2Web) performs even worse than MINDACT+Fixed which employs the fixed memory selection. This indicates the coarse-grained $k \mathbf { N N }$ matching in Synapse fails to effectively measure the relevance between the current conversation status and the candidate memory snippets in our conversational setting. In general, using a stronger base model (e.g., Flan-$\mathrm { T } 5 _ { \mathrm { l a r g e } ) }$ ) improves the final performance. Overall, Self-MAP consistently and substantially outperforms these baselines with a noticeable margin (e.g., $+ 6 . 3 / + 2 . 9 / + 3 . 1$ TSR scores with Flan-T5base over the strongest baselines). This showcases the effectiveness of utilizing our proposed memoryaugmented planning framework as well as the selfreflection strategy for enhancing memory utilization.

# 5.3 Ablation Study

To validate the specific designs of the Self-MAP framework, we present the ablation study in Table 3. First, we observe that Generation-based Planning substantially surpasses MCQ-based Planning in performance. This superiority is attributed not only to the advanced generative capabilities of large language models (LLMs) but also to their efficiency in conserving context space for memory utilization. Second, the process of Memory Simplification emerges as the most critical factor in enhancing overall performance. This finding underscores the importance of optimizing the use of limited context space, highlighting the necessity of efficient memory management. Third, the contribution of Memory Refinement is notably more pronounced in cross-task scenarios compared to other settings. This indicates its relatively lower generalizability in modeling decision-making processes, compared to the other components of our framework. Lastly,

![](images/a88f57925163c260233ae892c96842e121a8a1dc5db3ac53a75fba9722389347.jpg)

![](images/e392c325c1fd4a6fef995bfdd567511ecd7fadee390006596193aa38d195aa3e.jpg)

![](images/64a874d0cbe380bea43b06b70eb943d9e047fcbbedef86f523905afd74603dbe.jpg)

![](images/6ca904e13857dec943fc256c2a85457af6be62a3387bacee1eb1b5d276d96bea.jpg)  
Figure 4: Performance in terms of different number of retrieved memory snippets.

Multifaceted Matching for memory retrieval significantly outperforms vanilla memory prepending, which suggests the necessity of filtering out noisy conversational interaction history to focus on the relevant part.

# 5.4 Detailed Analysis

Effect of the Number of Retrieved Memory We first analyze the effect of the number of retrieved memory snippets by varying $K$ from 1 to 5. The results are presented in Figure 4. We observe that the performance increases along with the growth of the number of retrieved memory snippets at the beginning $( K \leq 3 )$ ), indicating the value of refining the memory utility for exploiting more relevant information. However, the continued increase on $K$ fails to contribute to the performance improvement, even making worse performance in some subsets, (e.g., cross-task and cross-website). As shown in the dataset statistics (Table 1), the average number of conversational turns is about 5 turns. Therefore, it may introduce noisy information from those irrelevant turns when increasing the number of retrieved memory snippets.

![](images/29888d7262f2da5aaaa50aa5c0ba913a0dc3649c75164c5873e5c3287e250b33.jpg)  
Figure 5: Step success rate regarding each website grouped by the three test splits.

![](images/442955994764b3be5cf7e2710d48569c9a12af3a862f7ba0cea48150a2179b65.jpg)  
Figure 6: Performance in terms of different number of retrieved memory snippets.

Analysis of Generalizability Compared with the analysis of generalizability conducted in Mind2Web (Deng et al., 2023), we draw some similar observations from Figure 5 in the conversational setting: 1) All models perform better on the Cross-Task setting than the other two settings. 2) There is no significant difference between Cross-Website and Cross-Subdomain settings, indicating that the challenges primarily come from the diversity in website designs and interaction logic rather than domain specifics. Differently, we observe that the performance gap between the Cross-Task setting and the other two settings is more substantial than that in Mind2Web $1 0 \%  2 0 \%$ ), which suggests that the interaction logic becomes more complicated when introducing multi-turn user-agent interactions.

Analysis of Conversation Prompt Designs Unlike traditional web navigation tasks, more types of information are supposed to be included in the prompt for action planning in the MT-Mind2Web task. We first examine the impact of memory snippet order in conversation prompts, including the adopted relevance-based order and the typical

chronological (sequential) order, in the Synapse and Self-MAP methods. As shown in Figure 6, both methods generally perform much better with relevance-based order compared to chronological order. In addition, we introduce state-based information into the proposed multifaceted matching approach. In Self-MAP, we omit $A _ { t } ^ { k - 1 }$ in the $M _ { t } ^ { k }$ , as in actual conversational contexts, explicitly identifying the state within a sequence-ordered trajectory is unnecessary. However, in the context of action-level matching, which lacks a sequential framework, state-based information cannot be inferred from the trajectory. Our results suggest that multifaceted matching typically achieves better performance without state-based information in the retrieved memory. Based on these analyses, we finalize our prompt designs, which are presented in Appendix B.2.

# 6 Conclusions

To investigate the capability of web agents to follow instructions over multiple turns, we introduce the MT-Mind2Web dataset for conversational web navigation, which requires complex, multi-turn interactions with both users and the web environment. To overcome the underlying challenges, such as the restricted context length of LLMs and their dependency on conversational context, we present a novel framework named Self-Reflective Memory-Augmented Planning (Self-MAP), which utilizes memory augmentation and self-reflection techniques. We rigorously evaluate the MT-Mind2Web dataset against extensive baselines, conducting thorough analyses across various domains. Our experimental findings demonstrate the effectiveness of our proposed approach.

# Limitation

Multimodal Environment With the advent of multimodal LLMs, recent studies demonstrate the

effectiveness of applying multimodal web agents (Zheng et al., 2024a; He et al., 2024) onto the web navigation problem. Without loss of generality, the constructed MT-Mind2Web dataset can also be adapted to the multimodal environment as the original Mind2Web dataset. In this work, we mainly focus on benchmarking general HTML-grounded methods, while we believe that it will also be a promising research direction on studying the conversational web navigation problem under the multimodal setting.

Online Evaluation As a pioneer study of conversational web agents, we follow the typical offline evaluation settings of both conversational tasks (Zheng et al., 2023a) and single-turn web navigation tasks (Deng et al., 2023), which allows researchers and practitioners to efficiently and conveniently evaluate the web gents using snapshots of complex real-world websites. However, it also inherits the drawback of the offline evaluation setting, e.g., evaluating dynamic interactions.

# References

Abhijit Anand, Venktesh V, Vinay Setty, and Avishek Anand. 2023. Context aware query rewriting for text rankers using LLM. CoRR, abs/2308.16753.   
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In ICLR 2024.   
Wenhu Chen, Ming-Wei Chang, Eva Schlinger, William Yang Wang, and William W. Cohen. 2021. Open question answering over tables and text. In ICLR 2021.   
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai, Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. 2022. Scaling instruction-finetuned language models. CoRR, abs/2210.11416.   
Yuhao Dan, Zhikai Lei, Yiyang Gu, Yong Li, Jianghao Yin, Jiaju Lin, Linhao Ye, Zhiyan Tie, Yougen Zhou, Yilei Wang, Aimin Zhou, Ze Zhou, Qin Chen, Jie Zhou, Liang He, and Xipeng Qiu. 2023. Educhat: A large-scale language model-based chatbot system for intelligent education. CoRR, abs/2308.02773.   
Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samuel Stevens, Boshi Wang, Huan Sun, and Yu Su.

2023. Mind2web: Towards a generalist agent for the web. In NeurIPS 2023.   
Yang Deng, Wenqiang Lei, Wenxuan Zhang, Wai Lam, and Tat-Seng Chua. 2022. PACIFIC: towards proactive conversational question answering over tabular and textual data in finance. In EMNLP 2022, pages 6970–6984.   
Yang Deng, Wenxuan Zhang, Wai Lam, See-Kiong Ng, and Tat-Seng Chua. 2024. Plug-and-play policy planner for large language model powered dialogue agents. In ICLR 2024.   
Izzeddin Gur, Hiroki Furuta, Austin Huang, Mustafa Safdari, Yutaka Matsuo, Douglas Eck, and Aleksandra Faust. 2024. A real-world webagent with planning, long context understanding, and program synthesis. In ICLR 2024.   
Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. 2024. Webvoyager: Building an end-toend web agent with large multimodal models. CoRR, abs/2401.13919.   
Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2021. Deberta: decoding-enhanced bert with disentangled attention. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.   
Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bodhisattwa Prasad Majumder, Nathan Kallus, and Julian J. McAuley. 2023. Large language models as zero-shot conversational recommenders. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, CIKM 2023, Birmingham, United Kingdom, October 21-25, 2023, pages 720–730. ACM.   
Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, and Chenglin Wu. 2024. Metagpt: Meta programming for multi-agent collaborative framework. In ICLR 2024.   
Xu Huang, Jianxun Lian, Yuxuan Lei, Jing Yao, Defu Lian, and Xing Xie. 2023. Recommender AI agent: Integrating large language models for interactive recommendations. CoRR, abs/2308.16505.   
Faria Huq, Jeffrey P. Bigham, and Nikolas Martelaro. 2023. "what’s important here?": Opportunities and challenges of using llms in retrieving information from web interfaces. CoRR, abs/2312.06147.   
Geunwoo Kim, Pierre Baldi, and Stephen McAleer. 2023. Language models can solve computer tasks. In NeurIPS 2023.   
Yongqi Li, Wenjie Li, and Liqiang Nie. 2022. MM-CoQA: Conversational question answering over text, tables, and images. In ACL 2022, pages 4220–4231.

Evan Zheran Liu, Kelvin Guu, Panupong Pasupat, Tianlin Shi, and Percy Liang. 2018. Reinforcement learning on web interfaces using workflow-guided exploration. In 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net.   
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. 2024. Agentbench: Evaluating llms as agents. In ICLR 2024.   
Chang Ma, Junlei Zhang, Zhihao Zhu, Cheng Yang, Yujiu Yang, Yaohui Jin, Zhenzhong Lan, Lingpeng Kong, and Junxian He. 2024. Agentboard: An analytical evaluation board of multi-turn llm agents.   
Sahisnu Mazumder and Oriana Riva. 2021. FLIN: A flexible natural language interface for web navigation. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, pages 2777–2788. Association for Computational Linguistics.   
Kai Nakamura, Sharon Levy, Yi-Lin Tuan, Wenhu Chen, and William Yang Wang. 2022. HybriDialogue: An information-seeking dialogue dataset grounded on tabular and textual data. In Findings of ACL: ACL 2022, pages 481–492.   
Haojie Pan, Zepeng Zhai, Hao Yuan, Yaojia Lv, Ruiji Fu, Ming Liu, Zhongyuan Wang, and Bing Qin. 2023. Kwaiagents: Generalized information-seeking agent system with large language models. CoRR, abs/2312.04889.   
Tianlin Shi, Andrej Karpathy, Linxi Fan, Jonathan Hernandez, and Percy Liang. 2017. World of bits: An open-domain platform for web-based agents. In Proceedings of the 34th International Conference on Machine Learning, ICML 2017, Sydney, NSW, Australia, 6-11 August 2017, volume 70 of Proceedings of Machine Learning Research, pages 3135–3144. PMLR.   
Noah Shinn, Beck Labash, and Ashwin Gopinath. 2023. Reflexion: an autonomous agent with dynamic memory and self-reflection. In NeurIPS 2023.   
Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew J. Hausknecht. 2021. Alfworld: Aligning text and embodied environments for interactive learning. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net.   
Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, and Chao Zhang. 2023. Adaplanner: Adaptive planning from feedback with language models. In NeurIPS 2023.

Alon Talmor, Ori Yoran, Amnon Catav, Dan Lahav, Yizhong Wang, Akari Asai, Gabriel Ilharco, Hannaneh Hajishirzi, and Jonathan Berant. 2021. Multimodal $\{ { \tt q a } \}$ : complex question answering over text, tables and images. In ICLR 2021.   
Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Ji-Rong Wen. 2023. A survey on large language model based autonomous agents. CoRR, abs/2308.11432.   
Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, and Heng Ji. 2024. MINT: evaluating llms in multi-turn interaction with tools and language feedback. In ICLR 2024.   
Michael Wooldridge and Nicholas R Jennings. 1995. Intelligent agents: Theory and practice. The knowledge engineering review, 10(2):115–152.   
Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wensen Cheng, Qi Zhang, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huan, and Tao Gui. 2023. The rise and potential of large language model based agents: A survey. CoRR, abs/2309.07864.   
Tianbao Xie, Fan Zhou, Zhoujun Cheng, Peng Shi, Luoxuan Weng, Yitao Liu, Toh Jing Hua, Junning Zhao, Qian Liu, Che Liu, Leo Z. Liu, Yiheng Xu, Hongjin Su, Dongchan Shin, Caiming Xiong, and Tao Yu. 2023. Openagents: An open platform for language agents in the wild. CoRR, abs/2310.10634.   
Yiheng Xu, Hongjin Su, Chen Xing, Boyu Mi, Qian Liu, Weijia Shi, Binyuan Hui, Fan Zhou, Yitao Liu, Tianbao Xie, Zhoujun Cheng, Siheng Zhao, Lingpeng Kong, Bailin Wang, Caiming Xiong, and Tao Yu. 2024. Lemur: Harmonizing natural language and code for language agents. In ICLR 2024.   
Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop: Towards scalable realworld web interaction with grounded language agents. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.   
Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024a. Gpt-4v(ision) is a generalist web agent, if grounded. CoRR, abs/2401.01614.   
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023a. Judging llm-as-a-judge with mt-bench and chatbot arena. In NeurIPS 2023.

Longtao Zheng, Rundong Wang, and Bo An. 2024b. Synapse: Leveraging few-shot exemplars for humanlevel computer control. In ICLR 2024.   
Zhonghua Zheng, Lizi Liao, Yang Deng, and Liqiang Nie. 2023b. Building emotional support chatbots in the era of llms. CoRR, abs/2308.11584.   
Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. 2024. Webarena: A realistic web environment for building autonomous agents.   
Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv, Fuli Feng, and Tat-Seng Chua. 2021. TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. In ACL/IJCNLP 2021, pages 3277–3287.

# Appendix

# A Details of Experimental Setups

# A.1 Details of Baselines

• DeBERTa (He et al., 2021). Following Deng et al. (2023), we also fine-tune DeBERTa as the ranker for selecting target elements.   
• MINDACT (Deng et al., 2023) performs multichoice question answering to select the target element from a list of options. Under the conversational setting, the input includes the whole conversational interaction history.   
• MINDACT $^ +$ CAR (Anand et al., 2023). We first employ context-aware rewriting (CAR) using ChatGPT to reconstruct the self-contained instructions from the conversational instructions and the conversation context. Then the selfcontained instructions are directly used as the input instructions for Mind2Act. The prompting details are presented in Appendix B.1.   
• MINDACT $^ +$ Fixed (Huq et al., 2023). Huq et al. (2023) empirically observe that using fixed examples outperforms relevance-based example selection for demonstration-based learning in the web navigation task. We fix the first 3 turns in the conversation history in chronological order as the memory.   
• Synapse (Zheng et al., 2024b). Synapse employs metadata, including website, domain, subdomain, and task as keywords to conduct kNN-based exemplar retrieval. Given that each conversation turn in our task shares the same website, domain, and subdomain information, we only keep the

task in the metadata and perform the turn-level kNN.

# A.2 More Details on Implementation

• Memory Simplification. We use Sentence-Transformers2 and fine-tune DeBERTa-v3-base (He et al., 2021) for our multi-turn task. Following Deng et al. (2023), we choose 5 random elements, including the positive candidate for training, and select the top-50 elements compared in groups of 5 for evaluation. During the training, we set the batch size as 32, the learning rate as 3e-5, and trained for 5 epochs.   
• Action Planning. We use Flan- ${ \cdot } \mathrm { T } 5 _ { \mathrm { b a s e } }$ and Flan-$\mathrm { T } 5 _ { \mathrm { l a r g e } }$ (Chung et al., 2022) for MCQ-based and generation-based action planning. We set the maximum sequence length at 2,048. Since the max context length for the tokenizer is 512, we tokenize the system message, HTML, user input, and assistant response separately. During the training, we set the batch size as 8 and 4 for Flan-$\mathrm { T 5 _ { b a s e } }$ and Flan- $. \mathrm { T } 5 _ { \mathrm { l a r g e } }$ respectively, the learning rate as 5e-5, and trained for 5 epochs.   
• Multifaceted Matching. We use the OpenAI embedding model text-embedding-ada-002 for matching, and choose cosine similarity for calculating embedding. We set the number of retrieved memories $K$ to 3. The prompting details of two paradigms of action planning are introduced in Appendix B.2.   
• Memory Refinement We use ChatGPT with the version of gpt-3.5-turbo-1106, maximum new tokens as 100, and temperature as 0. We only extract HTML snippets of the positive element for ChatGPT to generate rationales. If there is no positive element in the HTML snippet, we use "The assistant’s answer is derived from the absence of a specific option in the provided HTML content, leading to the conclusion that none of the options provided are suitable for the user’s task." as the default rationale. The prompting details for memory refinement are presented in Appendix B.3.

# B Prompting Details

# B.1 Prompts for Context-aware Rewriting

The prompts for implementing Context-Aware Rewriting (CAR) (Anand et al., 2023) are presented

Table 4: Prompts for context-aware rewriting.   

<table><tr><td>Role</td><td>Content</td></tr><tr><td>system</td><td>You are a helpful assistant adept at understanding and rewriting user queries. Your task is to evaluate the relevance of previous queries, add any relevant missing details from the previous queries, and rewrite the current query.</td></tr><tr><td>user</td><td>Rewrite: Help me check the popularity in 2015.
Previous queries:
Find the baby girl&#x27;s name.
Show me the most popular one.</td></tr><tr><td>assistant</td><td>Show me the popularity in 2015 of the current most popular baby girl name.</td></tr><tr><td>user</td><td>Rewrite: List the best rated for me.
Previous queries:
Search for pizza recipes for me.
I want the ones that take 30 minutes or less.
Show me the vegan option.
Find Halloween dishes.
Help me sort by rating.
Find pie recipes.
Show me all the content.</td></tr><tr><td>assistant</td><td>Find pie recipes and show the best rated ones.</td></tr><tr><td>user</td><td>Rewrite: How about a list of CDB product reviews.
Previous queries:
Find me a gluten-free diet to lose weight for a pregnant woman.</td></tr><tr><td>assistant</td><td>Browse a list of CDB product reviews.</td></tr></table>

in Table 4.

# B.2 Prompts for Planning

The prompt templates for Flan-T5 planning are presented in Table 5 for MCQ-based planning and in Table 6 for generation-based planning. Note that $\{ \}$ represents predefined placeholders in the MT-Mind2Web task. The system message for GPT-3.5 planning is listed in Table 7.

# B.3 Prompts for Memory Refinement

The prompts for implementing memory refinement using GPT-3.5 are shown in Table 8.

# C Case Study

# C.1 Conversation Description

This section examines a specific case from the MT-Mind2Web task, focusing on the responses from MINDACT $^ +$ CAR, Synapse, and Self-MAP. The conversation history, comprising seven trajectories, is detailed in Table 9. The instruction of the current conversation is Search 'xbox series x console'., with no previous actions noted.

# C.2 Analysis and Comparison

The analysis of different final evaluation steps for the current conversation is shown in Table 11. For

MINDACT $^ +$ CAR, we highlight the rephrased query in blue. This model, however, integrates irrelevant turns from the conversation history, aligning with our observations in Subsection 5.2.

Synapse employs a coarse-grained kNN matching method, retaining all historical conversation turns. Compared with Synapse, Table 10 displays the augmented memory and self-reflection from Self-MAP. Notably, Self-MAP selects Trajectories 2, 1, and 7 due to their relevance to the current instruction. These selections are along with reasoning rationales generated by GPT-3.5 and highlighted in blue. Both Synapse and Self-MAP select identical HTML elements in their final evaluation step, as indicated in Table 11. Synapse does not process the sequence of the search operation correctly. This oversight makes it trigger a submit action without entering the search keyword, as a result of the noisy information in its retrieved memory. Conversely, Self-MAP’s success in the same scenario can be attributed to its understanding and combination of relevant conversation history from multifaceted matching and self-reflection, highlighting the efficacy of its approach.

# System Message

You are a helpful assistant that is great at website design, navigation, and executing tasks for the user.

# Conversation History

### Human:

{HTML snippets including 5 elements}

Based on the HTML webpage above, try to complete the following task:

Task: {instruction}

What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. ’None of the above’):

A. None of the above

B. {element 1}

C. {element 2}

D. {element 3}

E. {element 4}

F. {element 5}

### Assistant: {response}

{Optional: Reflection}

# Current Conversation

### Human: ```

{HTML snippets including 5 elements}

Based on the HTML webpage above, try to complete the following task:

Task: {instruction}

Previous actions:

{last 5 action representations}

What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. ’None of the above’):

A. None of the above

B. {element 1}

C. {element 2}

D. {element 3}

E. {element 4}

F. {element 5}

### Assistant: {response}

Table 5: Prompt Templates for MCQ-based Flan-T5 Planning

# System Message

You are a helpful assistant that is great at website design, navigation, and executing tasks for the user.

# Conversation History

### Human:

{HTML snippets including 5 elements}

Based on the HTML webpage above, try to complete the following task:

Task: {instruction}

What should be the next action? Please select the element to interact with, and the action to perform along with the value to type in or select. If the task cannot be completed, output None:

### Assistant: {response}

{Optional: Reflection}

# Current Conversation

### Human: ```

{HTML snippets including 5 elements}

Based on the HTML webpage above, try to complete the following task:

Task: {instruction}

Previous actions:

{last 5 action representations}

What should be the next action? Please select the element to interact with, and the action to perform along with the value to type in or select. If the task cannot be completed, output None:

### Assistant: {response}

Table 6: Prompt Templates for Generation-based Flan-T5 Planning

You are a helpful assistant that is great at website design, navigation, and executing tasks for the user.

### Example

<html> <div> <div> <a tock home page $/ >$ <button id=0 book a reservation. toggle open> <span> Book a reservation </span> </button> <button book a reservation. toggle open> </button> </div> <div> <select id=1 type> <option reservations true> Dine in </option> <option pickup> Pickup </option> <option delivery> Delivery </option> <option events> Events </option> <option wineries> Wineries </option> <option all> Everything </option> </select> <div $\mathrm { i d } { = } 2 >$ <p> Celebrating and supporting leading women shaking up the industry. ${ < } I { \mathsf { p } } { > }$ <span> Explore now </span> </div> </div> </div> </html>

Based on the HTML webpage above, try to complete the following task:

Task: Check for pickup restaurant available in Boston, NY on March 18, 5pm with just one guest

Previous actions:

None

What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. ’None of the above’):

A. None of the above

B. <button id=0 book a reservation. toggle open> <span> Book a

C. <select id=1 type> <option reservations true> Dine in </option> <option

D. <div id=2> <p> Celebrating and supporting leading women shaking up

Assistant:

Answer: C.

Action: SELECT

Value: Pickup

Table 7: System Message for GPT-3.5 Planning

Table 8: Prompts for Memory Refinement   

<table><tr><td>Role</td><td>Content</td></tr><tr><td>system</td><td>You are an advanced reasoning agent who specializes in analyzing conversational web navigation. You will be presented with a conversation between users and assistants involving a webpage&#x27;s HTML content, a user&#x27;s query, and an AI assistant&#x27;s response. Your objective is to provide a concise and clear one-sentence rationale that explains how the assistant&#x27;s response is derived from the HTML content in relation to the user&#x27;s specific query.</td></tr><tr><td>user</td><td>Conversation
Webpage: &lt;select id=1 type&gt;&lt;option reservations true&gt;Dine in &lt;/option&gt;&lt;option pickup&gt; Pickup &lt;/option&gt;
&lt;option delivery&gt; Delivery &lt;/option&gt;&lt;option events&gt; Events &lt;/option&gt;&lt;option wineries&gt; Wineries &lt;/option&gt;
&lt;option all&gt; Everything &lt;/option&gt;&lt;/select&gt;User: Based on the HTML webpage above, try to complete the following task:
Task: Check for pickup restaurant available in Boston, NY on March 18, 5pm with just one guest.
Previous actions:
None
What should be the next action?assistant
Assistant: Action: SELECT
Value: Pickup
Rationale
The assistant chose to select &quot;Pickup&quot; directly corresponds to the user&#x27;s request to check for a pickup restaurant. This action is informed by the HTML structure of the webpage, which contains a dropdown menu (&lt;select id=1 type&gt;) with various options including &quot;Pickup&quot; (&lt;option pickup&gt;).</td></tr><tr><td>user</td><td>Conversation
Webpage: &lt;button id=2 selected pick-up date 03/19/2023&gt;&lt;span&gt; 19 &lt;/span&gt;&lt;div&gt; &lt;span&gt; Mar &lt;/span&gt; 2023 &lt;/div&gt;&lt;/span&gt;&lt;/button&gt;
User: Based on the HTML webpage above, try to complete the following task:
Task: Find a mini van at Brooklyn City from April 5th to April 8th for a 22 year old renter.
Previous actions:
[searchbox] Pick-up &amp; Return Location (ZIP, City or Airport) (... -&gt; TYPE: Brooklyn [option] Brooklyn, NY, US Select -&gt; CLICK
What should be the next action?assistant
Assistant: Action: CLICK
Rationale
The assistant&#x27;s response to perform a &quot;CLICK&quot; action is given the user&#x27;s progress in the task. The user has already selected a location (Brooklyn) for picking up a minivan. The next logical step in the process would be to click the button to select the pick-up date.</td></tr><tr><td>assistant</td><td></td></tr></table>

Table 9: Conversation History   

<table><tr><td>Previous Instruction</td><td>Action Representations</td></tr><tr><td>Search for new laptops.</td><td>[combobox] Search for anything -&gt; TYPE: laptop
[button] Search -&gt; CLICK</td></tr><tr><td>Set price from $400 to $500.</td><td>[input] -&gt; CLICK
[ textbox] Minimum Value in $ -&gt; TYPE: 400
[ textbox] Maximum Value in $ -&gt; TYPE: 500
[button] Submit price range -&gt; CLICK</td></tr><tr><td>Search for free shipping.</td><td>[input] -&gt; CLICK</td></tr></table>

# History Trajectory 2

### Human: 、

(html (body (header banner (input id=0 submit search ) ) (div main (div group - (ul id=1 (a (img nintendo switch red / blue ) (span $\$ 166.00$ ) ) (a (img nintendo switch hac-001 neon yellow! ) (span $\$ 99.99$ ) ) [. . .]

Based on the HTML webpage above, try to complete the following task:

Task: Search for new laptops.

What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. ’None of the above’):

A. None of the above   
B. (input $\mathrm { i d } { = } 0$ submit search )   
C. (ul id=1 (a (img nintendo switch red / blue )   
D. (div id=2 (button button go to previous slide - )   
E. (a id=3 eBay Community )   
F. (a id=4 Accessibility )

### Assistant: B.

Action: CLICK

Rationale: The assistant’s response to perform a "CLICK" action is derived from the user’s task to search for new laptops, and the previous action of typing "laptop" into the search box. Clicking the search button is the next logical step to initiate the search for new laptops based on the HTML structure provided.

# History Trajectory 1

### Human: ```

(html (body (header banner (td (input $\mathrm { i d } { = } 0$ combobox text search for anything _nkw ) (ul listbox (li id=1 (a option vintage clothing recent searches (b vintage clothing ) ) ) ) ) ) (div main (div group - (a (img id=2 diamond stripe comforter set by ) (span $\$ 34.99$ ) ) ) [. . .]

Based on the HTML webpage above, try to complete the following task:

Task: Search for new laptops.

What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. ’None of the above’):

A. None of the above

B. (input $\mathrm { i d } { = } 0$ combobox text search for anything _nkw )

C. (li id=1 (a option vintage clothing recent searches (b vintage

D. (img id=2 diamond stripe comforter set by )

E. (a id=3 (img down alternative forget me not ) (span

F. (li id=4 (a Contact Us ) )

### Assistant: B.

Action: TYPE

Value: laptop

Rationale: The assistant’s response to type "laptop" is derived from the user’s task to search for new laptops, and the HTML structure of the webpage, which contains an input element with the id $" 0 "$ and the type "combobox text" for searching. Therefore, the next action is to type "laptop" into the search input field.

# History Trajectory 7

### Human: ```

[. . .] (li lh_fs (div (input id=1 checkbox free shipping on ) (span Free Shipping ) ) ) ) (div main (div id=2 (div (img shop on ebay ) (div (a (span heading Shop on eBay ) (span Opens in a new window or tab ) ) (span Brand New ) [. . .]

Based on the HTML webpage above, try to complete the following task:

Task: Search for free shipping.

What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. ’None of the above’):

A. None of the above

B. (li id=0 (div button (h3 Buying Format ) ) (ul

C. (input $\mathrm { i d } { = } 1$ checkbox free shipping on )

D. (div id=2 (div (img shop on ebay ) (div (a

E. (a id=3 (div 64 GB (span - apply Shop by

F. (a id=4 Your Privacy Choices )

### Assistant: C.

Action: CLICK

Rationale: The assistant’s response to click is derived from the user’s task to search for free shipping, which is represented by the checkbox with the label "Free Shipping" in the HTML content. Clicking on this checkbox would allow the user to search for products with free shipping.

# MindAct + CAR

(html (body (button id=0 button Open (svg id=1 (desc Expand Svg ) ) ) (header banner (tr (input id=2 combobox text search for anything _nkw ) (input $\mathrm { i d } { = } 3$ submit search ) ) ) (div main (div (a Video Game Consoles ) (a id=4 see all video game consoles See all ) ) ) ) )

Based on the HTML webpage above, try to complete the following task:

Task: Search for the 'xbox series x console' with free shipping and a price range from $\$ 400$ to $\$ 500$

Previous actions:

What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. ’None of the above’):

A. None of the above   
B. (button id=0 button Open (svg id=1 (desc Expand Svg )   
C. (svg $\mathrm { i d } { = } 1$ (desc Expand Svg ) )   
D. (input id=2 combobox text search for anything _nkw )   
E. (input id=3 submit search )   
F. (a $\mathrm { i d } { = } 4$ see all video game consoles See all )

### Assistant: E.

Action: CLICK

# Synapse

### Human: ```

(html (body (header banner (tr (input id=0 combobox text search for anything _nkw ) (input id=1 submit search ) ) ) (div main (li (a $\mathrm { i d } { = } 2$ Electronics ) (button Expand: Electronics ) ) ) ) )

Based on the HTML webpage above, try to complete the following task:

Task: Search 'xbox series x console'.

Previous actions:

None

What should be the next action? Please select from the following choices (If the correct action is not in the page above, please select A. ’None of the above’):

A. None of the above   
B. (input id=0 combobox text search for anything _nkw )   
C. (input id=1 submit search )   
D. (a id=2 Electronics )

### Assistant: C.

Action: CLICK

# Self-MAP

### Human: ```

(html (body (header banner (tr (input id=0 combobox text search for anything _nkw ) (input $\mathrm { i d } { = } 1$ submit search ) ) ) (div main (li (a $\mathrm { i d } { = } 2$ Electronics ) (button Expand: Electronics ) ) ) ) )

Based on the HTML webpage above, try to complete the following task:

Task: Search 'xbox series x console'.

Previous actions:

None

What should be the next action? Please select the element to interact with, and the action to perform along with the value to type in or select. If the task cannot be completed, output None.

### Assistant: Element: (input id=0 combobox text search for anything _nkw )

Action: TYPE

Value: xbox series x console

Table 11: Final Evaluation Steps
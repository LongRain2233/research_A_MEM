# AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents

Petr Anokhin1 , Nikita Semenov2 , Artyom Sorokin1 , Dmitry Evseev2 , Andrey Kravchenko4 , Mikhail Burtsev3 and Evgeny Burnaev2,1

1AIRI, Moscow, Russia

2Skoltech, Moscow, Russia

3London Institute for Mathematical Sciences, London, UK

4University of Oxford, UK

anokhin@airi.net

# Abstract

Advancements in the capabilities of Large Language Models (LLMs) have created a promising foundation for developing autonomous agents. With the right tools, these agents could learn to solve tasks in new environments by accumulating and updating their knowledge. Current LLM-based agents process past experiences using a full history of observations, summarization, retrieval augmentation. However, these unstructured memory representations do not facilitate the reasoning and planning essential for complex decision-making. In our study, we introduce AriGraph, a novel method wherein the agent constructs and updates a memory graph that integrates semantic and episodic memories while exploring the environment. We demonstrate that our Ariadne LLM agent, consisting of the proposed memory architecture augmented with planning and decision-making, effectively handles complex tasks within interactive text game environments difficult even for human players. Results show that our approach markedly outperforms other established memory methods and strong RL baselines in a range of problems of varying complexity. Additionally, AriGraph demonstrates competitive performance compared to dedicated knowledge graph-based methods in static multi-hop question-answering.

# 1 Introduction

Impressive language generation capabilities of large language models (LLMs) has sparked substantial interest in their application as core components for creating autonomous agents capable of interacting with dynamic environments and executing complex tasks. Over the past year, the research community has explored general architectures and core modules for such LLM agents [Wang et al., 2024; Sumers et al., 2024; Cheng et al., 2024]. A crucial property of a general cognitive agent is its ability to accumulate and use knowledge. A long-term memory allows an agent to store and recall past experiences and knowledge, enabling it to learn from previous

encounters and make informed decisions. However, the question of the best way to equip an agent with these capabilities remains open. Despite the constraints inherent in transformer architectures, contemporary methods enable LLMs to manage contexts encompassing millions of tokens [Ding et al., 2024b]. However, this approach proves inefficient for agents required to maintain continuous interaction with their environment. Such agents must hold an entire historical context in memory to perform actions, which is not only costly but also limited in handling complex logic hidden in vast amounts of information. Research into alternative frameworks like Recurrent Memory Transformer [Bulatov et al., 2022; Bulatov et al., 2024] and MAMBA [Gu and Dao, 2023] seeks to provide long-term memory solutions, though these models are still in their infancy.

Currently, the most popular solution for incorporating memory to LLM agents is the Retrieval-Augmented Generation (RAG) approach. RAG in a form of vector retrieval leverages an external database to enhance the model’s prompt with relevant information. This technique is commonly used in memory architectures for LLM agents, often to recall specific observations or learned skills. However, it suffers from unstructured nature, greatly reducing the ability to retrieve related information, which may be scattered throughout the agent’s memory. These limitations can be overcome by using knowledge graphs as database. This approach has also experienced a resurgence with the advent of LLMs [Pan et al., 2024]. However, for a robust memory architecture, integrating both structured and unstructured data is essential. In cognitive science, this integration parallels the concepts of semantic and episodic memories. Semantic memory encompasses factual knowledge about the world, whereas episodic memory pertains to personal experiences, which often contain richer and more detailed information. Though traditionally considered separate due to their distinct neurological representations, recent studies suggest these memory types are interconnected [Wong Gonzalez, 2018]. Semantic knowledge is built upon the foundation of episodic memory and subsequently provides a structured base for associative memory. This allows for the integration of various memory aspects, including episodic memories themselves.

In our research, we have developed a memory architecture

![](images/aa8f2971648fd22508e02d076bb2be6d0197734bd4f4ff4e7d57e8d9e9ffc5a1.jpg)

![](images/b6fa2fdcb8e0e0319cf4e765bc48ed69a8d04c9b443c70649f4e0ccbc988c9f0.jpg)  
Figure 1: (A) The architecture of our Ariadne agent, equipped with AriGraph memory. AriGraph integrates both semantic knowledge graph and past experiences. Memory in the form of a semantic knowledge graph extended with episodic vertices and edges significantly enhances the performance of LLM-agent in text-based games. (B) The average performance of our agent on text games, compared to various baselines including human players and other LLM memory implementations. The LLM-agents differ only in the memory module, while the decisionmaking component remains identical across all versions. The results for the agents are displayed for the top three out of five runs. For human players, the results are presented as both the top three and the average across all participants.

called Ariadne’s Graph (AriGraph), that integrates semantic and episodic memories within a memory graph framework. A knowledge graph represents a network of interconnected semantic knowledge, while episodic memories are depicted as episodic edges that can connect multiple relations within the graph. As an agent interacts with environment, it learns joint semantic and episodic world model by updating and extending knowledge graph based memory. This architecture not only serves as a foundational memory framework but also aids in environmental modeling, improving spatial orientation and exploration capabilities. For the general framework of our LLM agent called Ariadne, we employed pipeline of memory retrieval, planing and decision making. For evaluation of proposed methods we set up experiments to study two research questions.

RQ1. Can LLM based agents learn useful structured world model from scratch via interaction with an environment?

RQ2. Does structured knowledge representation improve retrieval of relevant facts from memory and enable effective exploration?

We evaluated our agent in complex interactive tasks in Textworld and NetHack environments [Cotˆ e´ et al., 2018; Kuttler¨ et al., 2020]. Experimental results demonstrate that our agent Ariadne can effectively learn through interactions with environment and significantly outperforms other memory approaches for LLMs such as full history, summarization, RAG, Simulacra [Park et al., 2023] and Reflexion [Shinn et al., 2023]. We also show that our method outperforms existing reinforcement learning (RL) baselines. We also evaluated our approach on the classical roguelike game NetHack, where our agent with local observations achieved scores comparable

to an agent with ground-truth knowledge. Although AriGraph was originally designed for an agent interacting with the environment, it also demonstrates competitive performance on multi-hop question answering tasks.

# 2 AriGraph World Model

Memory graph structure. AriGraph world model $G \ =$ $( V _ { s } , E _ { s } , V _ { e } , E _ { e } )$ consists of semantic $( V _ { s } , E _ { s } )$ and episodic memory $( V _ { e } , E _ { e } )$ vertices and edges (see Figure 2). At each step $t$ agent receives observation $o _ { t }$ and sends action $a _ { t }$ back to the environment. The environment also returns rewards $r _ { t }$ that are not visible to the LLM agent but are used to evaluate its performance. The agent continuously learns world model $G$ by extracting semantic triplets $( o b j e c t _ { 1 } , r e l a t i o n , o b j e c t _ { 2 } )$ from textual observations $o _ { t }$ .

• $V _ { s }$ is a set of semantic vertices. Semantic vertices correspond to objects extracted from triplets.   
• $E _ { s }$ is a set of semantic edges. Semantic edge is a tuple $( v , r e l , u )$ , where u, v are semantic vertices and rel is a relationship between them. Semantic edges essentially represent triplets integrated in the semantic memory.   
• $V _ { e }$ is a set of episodic vertices. Each episodic vertex corresponds to an observation received from the environment at the respective step $v _ { e } ^ { t } = o _ { t }$ .   
• $E _ { e }$ is a set of episodic edges. Each episodic edge $e _ { e } ^ { t } = ( v _ { e } ^ { t } , E _ { s } ^ { t } )$ e te connects all semantic triplets $\dot { \boldsymbol E } _ { s } ^ { t }$ extracted from $o _ { t }$ with each other and corresponding episodic vertex $v _ { e } ^ { t }$ . In other words episodic edges represent temporal relationship “happened at the same time”.1

Algorithm 1: Memory Graph Search   
Input: set of queries $Q$ , $V_{s}$ , $E_{s}$ , $V_{e}$ , $E_{e}$ , number of episodic vertices $k$ , semantic search depth $d$ and width $w$ Result: retrieved episodic vertices $V_{e}^{Q}$ , retrieved semantic triplets $E_{s}^{Q}$ $E_{s}^{Q} \gets \emptyset$ ,  
foreach $q$ in $Q$ do $\begin{array}{c} E_{s}^{\prime} \leftarrow \text{SemanticSearch}(q, V_{s}, E_{s}, d, w) \\ E_{s}^{Q} \leftarrow E_{s}^{Q} \cup E_{s}^{\prime} \end{array}$ end $V_{e}^{Q} \gets \text{EpisodicSearch}(E_{s}^{Q}, V_{e}, E_{e}, k)$ return $E_{s}^{Q}, V_{e}^{Q}$

Constructing AriGraph. Interaction with the environment can provide the agent with an information about the world to create new or update previously acquired knowledge. Given new observation $o _ { t }$ , LLM agent extracts new triplets as semantic vertices $V _ { s } ^ { t }$ and edges $\bar { E _ { s } ^ { t } }$ . To find already existing knowledge about the objects mentioned in $o _ { t }$ a set of all semantic edges $E _ { s } ^ { r e l }$ incident to vertices $V _ { s } ^ { t }$ is filtered out. Then outdated edges in $E _ { s } ^ { r e l }$ are detected by comparing them with $E _ { s } ^ { t }$ and removed from the graph. After clearing outdated knowledge we expand semantic memory with $V _ { s } ^ { \overline { { t } } }$ and $E _ { s } ^ { t }$ . Episodic memory is updated by simply adding new episodic vertex $v _ { e } ^ { t }$ containing $o _ { t }$ and new episodic edge that connect all edges in $E _ { s } ^ { t }$ with $v _ { e } ^ { t }$ . Episodic nodes store agent’s past history and episodic edges connects all knowledge received at the same step. See Appendix E for prompts used to extract new triplets and detect outdated knowledge.

Retrieval from AriGraph. For successful decisionmaking in a partially observable environment, the agent needs to be able to retrieve relevant knowledge. Retrieval from the AriGraph memory consists of two procedures: (1) a semantic search returns the most relevant triplets (semantic edges) and (2) an episodic search that, given extracted triplets, returns the most relevant episodic vertices $V _ { e }$ . The pseudo-code for the search is presented in the Algorithm 1.

Semantic search relies on semantic similarity and semantic graph structure to recall the most relevant triplets. Given a query, the retriever (pre-trained Contriever model [Izacard et al., 2022]) selects the most relevant semantic triplets. Then, the set of vertices incident to the found triplets is used to recursively retrieve new edges from the graph. Depth and breadth of the search can be controlled by respective hyperparameters $d$ and $w$ . For details see Appendix A.

Episodic search starts with the results of the semantic search as an input. Episodic edges link the input triplets with past episodic vertices representing observations. The number of input triplets associated with a particular episodic vertex is used to calculate their relevance:

$$
\operatorname {r e l} \left(v _ {e} ^ {i}\right) = \frac {n _ {i}}{\max  \left(N _ {i} , 1\right)} \log \left(\max  \left(N _ {i}, 1\right)\right), \tag {1}
$$

where $n _ { i }$ is a number of input triplets incident to episodic

edge $e ^ { i }$ , $N _ { i }$ is a total number triplets (semantic edges) incident to $e ^ { i }$ and log $( m a x ( N _ { i } , 1 ) )$ is a weighting factor to prevent high scores for low information observations. We apply a $l o g _ { 2 } ( N _ { i } )$ scaling to give more weight to observations with more extracted triplets. Additionally, observations containing exactly one triplet are assigned zero weight, as they are unlikely to provide information beyond the triplet itself. $k$ most relevant episodic vertices (containing respective observations) are returned as a result of the episodic search.

# 3 Ariadne cognitive architecture

To test utility of AriGraph world modelling method we propose an agentic architecture called Ariadne. Ariadne agent interacts with an unknown environment to accomplish a goal set by a user. Throughout this process, at each time step, the agent learns a world model, plans and executes actions. Ariadne has long-term memory stored as AriGraph and working memory containing information for current planning and decision making.

Given an observation the agent updates world model and retrieves semantic and episodic knowledge from AriGraph to working memory. Working memory is also populated with a final goal description, current observation, history of recent observation and actions. At the planning stage, Ariadne agent uses content of working memory to create new or update existing plan as a series of task-relevant sub-goals, each accompanied by a concise description. The planning module also evaluates the outcomes of actions based on feedback from the environment after each action at step $t - 1$ , adjusting the plan accordingly.

The revised plan is added to the working memory which is accessed by the decision-making module, tasked with selecting the most suitable action aligned with the current plan’s objectives. This module adheres to the ReAct [Yao et al., 2023] framework, requiring the agent to articulate the rationale behind an action before execution. Separation of planning from decision-making enables LLMs to focus on distinct cognitive processes. In text-based environments an agent selects an action from the list of valid actions. Our agent can also use graph specific function for navigation utilizing its memory module. It extends its action space with “go to location” type commands and infers an optimal route to a target location using spatial relations stored in a semantic graph.

# 4 Experimental Setup

# 4.1 TextWorld interactive environments

We compared Ariadne agent with alternative methods in a series of text based games involving spatial navigation, object collection and tool manipulation. Detailed descriptions of each game, including their difficulty levels and environmental maps, can be found in the Appendix F. All these games can be considered Partially Observable MDPs (POMDPs). Such games have long been benchmarks for researching agents capable of effectively remembering information and establishing long-term dependencies [Parisotto et al., 2020; Pleines et al., 2022; Sorokin et al., 2022].

Treasure Hunting. The primary objective is to retrieve the hidden treasure, with a series of rooms providing keys

![](images/fac0cc04661fc2228a7f5588ae1d7d15aaebb3c67e911dd5139a092ba306ef79.jpg)

![](images/9487b9ecbdebee8ad20972d2b1325be82ced5604dfdcd5fc8f6506f10eac8d70.jpg)  
Figure 2: AriGraph world model and Ariadne cognitive architecture. (A) AriGraph learns episodic and semantic knowledge during interaction with unknown environment. At each time step $t$ new episodic vertex (containing full textual observation $o _ { t }$ ) is added to the episodic memory. Then LLM model parses observation $o _ { t }$ to extract relevant relationships in a form of triplets (object1, relation, object2). These triplets are used to update semantic memory graph. The connection between episodic and semantic memory occurs through episodic edges that link each episodic vertex with all triplets extracted from respective observation. (B) Ariadne agent explores the environment and accomplishes tasks with AriGraph. User sets goal to the agent. Working memory is populated with recent history of observations and actions, relevant semantic and episodic knowledge retrieved from the AirGraph world model. Planing LLM module uses content of working memory to generate new or update existing plan. Results of planning are stored back in working memory. Finally, a ReAct-based module reads memory content and selects one of possible actions to be executed in the environment. Every observation triggers learning that updates agent’s world model.

and clues leading to the final goal. The basic variation has 12 rooms and 4 keys, hard one has 16 rooms and 5 keys and hardest contains 36 rooms, 7 keys and additional distract items in every room.

Cleaning. The goal is to clean a house by identifying and returning misplaced items to their correct locations. Environment consists of 9 rooms (kitchen, pool, etc.) and contains 11 misplaced items (among many other items). To solve the problem, the agent needs to memorize the location of rooms and objects, as well as reason about objects placement.

Cooking. The goal is to prepare and consume a meal by following a recipe, selecting the correct ingredients, and using appropriate tools, while navigating in multi-room house. The task is testing agents ability to remember relevant information and plan according to it. Basic difficulty task features 9 locations and 3 ingredients and hard task features 12 locations and 4 ingredients, while hardest task also features closed doors and inventory management.

For baselines we used Ariadne’s planning and decision making module with one of the following types of memory instead of AriGraph model: full history of observations and actions, iterative summarization, RAG, RAG with Reflexion [Shinn et al., 2023], and Simulacra - memory implementation from [Park et al., 2023].

Full history involves retaining a complete record of all observations and actions to inform decision-making at every step. Summarization, as an alternative to storing the full history, focuses on retaining only the necessary information while discarding the rest. The standard RAG baseline retrieves top-k memories based on their similarity score to the current observation and plan. Simulacra features a scoring mechanism that integrates recency, importance, and relevance, alongside reflections on the extracted memories. The Reflexion baseline differs from other methods in its approach,

as it operates over multiple trials. After failing a trial, the agent reflects on its trajectories to document information that may assist in solving the task in subsequent trials. We used the gpt-4-0125-preview as LLM backbone for AriGraph and other LLM-based baselines.

Additionally, we tested our architecture on a variation of the cooking test from [Adhikari et al., 2021] to compare it with RL baselines. These tasks have 4 levels of difficulty, however, they are significantly simpler than our main tasks, having fewer locations, ingredients, and required actions (Appendix F).

For RL baselines, we collect the best results reported by [Adhikari et al., 2021; Tuli et al., 2022; Basu et al., 2024] for the GATA, LTL-GATA, and EXPLORER architectures on the Cooking task with four difficulties levels from [Adhikari et al., 2021].

To estimate human performance in the same games, we developed a graphical user interface , allowing volunteers to play basic versions of the Treasure Hunt, The Cleaning, and the Cooking. After collecting the data, we excluded sessions where the game was not completed.

# 4.2 NetHack environment

NetHack [Kuttler ¨ et al., 2020] is a classic roguelike adventure game featuring procedurally generated multi-level dungeon (see Figure 13 in Appendix for a dungeon level example). It poses significant challenges for both LLM-based and RLbased approaches, requiring complex exploration, resource management, and strategic planning.

We based our experiments on NetPlay [Jeurissen et al., 2024] agent, which demonstrates state-of-the-art performance among LLM agents that do not rely on finetuning or RL. In NetPlay agent receives textual observations containing all information about current explored dungeon level. These

![](images/6c88dbfad72e63e4b0a09417cded074ecbd8091e65e980063672db636c374ac6.jpg)  
Figure 3: AriGraph world model enables Ariadne agent to successfully solve variety of text games. (A) Ariadne outperform baseline agents with alternative types of memory. (B) Ariadne with episodic and semantic memory scales to harder environments without losing performance. (C) Ariadne shows performance comparable to the best human players. The Y-axis shows the normalized score, which is calculated relative to the maximum possible points that can be obtained in each environment. Error bars show standard deviation. The number of max steps is set to 60 in the Cooking and to 150 in other games.

observations (Level obs) effectively function as handcrafted memory oracle for the agent.

To evaluate our Ariadne agent, we restricted textual observations to agent’s current room or corridor (Room Obs), testing whether AriGraph world model could compensate for this restriction by remembering all relevant level information.

We compare three agents. The first is NetPlay [Room obs] with restricted textual observations, the second is our Ariadne [Room obs] agent that receives Room Obs and updates Ari-Graph, and the last is NetPlay [Level obs] with access to inforation about explored level.

# 4.3 Multi-hop Q&A

Although our memory architecture was originally designed for an agent interacting with the environment, we evaluated its performance on standard multi-hop Q&A benchmarks — Musique [Trivedi et al., 2022] and HotpotQA [Yang et al., 2018] to show its robustness and efficiency in more standard retrieval tasks. We made slight adjustments to the promts and replaced Contriever model with BGE-M3[Chen et al., 2024], as it is a better fit for general text encoding. We used 200 random samples from both datasets similar to [Li et al., 2024a]. We compared the performance of our approach against Graphreader [Li et al., 2024a], ReadAgent [Lee et al., 2024], HOLMES [Panda et al., 2024], GraphRAG [Edge et al., 2024] and RAG baselines provided in [Li et al., 2024a].

# 5 Results

# 5.1 TextWorld

Every LLM based agent had five attempts to solve each game. The normalized score of one means that an agent completed

the game, and score less than one represents intermediate progress. Results on text-based games are shown on the Figure 3 (for dynamics see Appendix G). We estimate performance as average of three best runs. Ariadne successfully remembers and uses information about state of the world for all three tasks. Baseline agents are unable to solve the Treasure Hunt, and fail to find even second key in the Treasure Hunt Hardest. On the other hand, Ariadne successfully solves the Treasure Hunt in about fifty steps, maintains robust performance in the Treasure Hunt Hard, and is able to complete the Treasure Hunt Hardest with more then double amount of rooms compared to Hard version, additional keys and distractors (see Appendix G).

Compared to the Treasure Hunt, the Cleaning game possesses a slightly different challenge as it is more important to properly filter outdated information about object locations, than not to lose any information. This is evident from the reduced usefulness of Episodic Memory in Ariadne agent and Full history baseline, since both memory modules focus on retaining long-term information. Overall Ariadne notably outperforms alternatives in this game. Moreover, Ariadne also outperforms Reflexion, which has additional information between episodes [Shinn et al., 2023]. This baseline shows markable performance growth (in comparison wuth RAG) at the second try, but degrades with following tries.

The Cooking game has the highest difficulty, because any error at intermediate step prevents completion of the whole game. All baseline agents (except Reflexion 2-shot with obvious advantage over other methods) fail to complete cooking tasks due to insufficient or misused information. In this game, episodic memory is particularly important, allowing the agent

![](images/afcef03c7e1fc79bfc23bfb1b5ca90e6e1ce0476c14f7281b00a5a8f0a90f4dd.jpg)  
Figure 4: Ariadne LLM agent shows top performance compared to RL alternatives. Comparison of Ariadne and Full History baseline (GPT-4) with RL baselines in the cooking benchmark. Ariadne demonstrates superior performance across all 4 difficulty levels

to recall useful observations such as the content of the recipe or cooking instructions. For token usage of every method see Table 3, Appendix D.

Comparison with RL baselines on variation of the Cooking task is shown in Figure 4. We run Ariadne and GPT-4 with Full history on 4 difficulty levels from the cooking benchmark [Adhikari et al., 2021]. Ariadne shows superior performance to RL-agents on all 4 levels, especially harder ones. GPT-4 agent with Full history solves only first two levels which is consistent with previous result as the Cooking from Figure 3.A is harder than level 4.

Human evaluation. Comparison with respect to human players is shown in Figure 3.C. All Humans is the average score of all valid (completed) human trials. Human Top-3 is the average score of three best plays for each task. Ariadne outperforms average human player from our sample on all tasks, and scores similarly to the best human plays in the Cooking and the Treasure Hunt, but underperforms in the Cleaning.

Graph quality. We measured AriGraph’s growth rate and update rate during gameplay (see Figure 5). The graph actively grows during the exploration phase and flattens once the agent becomes familiar with the environment. We argue that this indicates that agent can generalize to long interactions with the environment despite constant updates to the semantic graph. Additional results in Appendix C demonstrate that the growth rate of the graph decreases with the increase in quality of LLM backbone.

Overal results demonstrate clear advantage of Ariadne agent over LLM based and RL baselines. Semantic memory enables the Ariadne Agent to build and update knowledge about the current state of the POMDP environment, which is crucial for navigation, exploration and capturing relevant details in interactive environments. On the other hand, episodic memory assists the agent in retrieving detailed long-term information that may not be captured in semantic memory, as demonstrated by the results in the Cooking task.

# 5.2 NetHack

The results are presented in Table 1. Scores column shows average game score across 3 runs, Levels column shows average number of dungeon levels completed by an agent. GPT-4o was used for all agents. Underscoring the importance

of memory in this task, NetPlay [Level obs] with access to memory oracle achieved the highest scores, while NetPlay [Room obs] with only current room observations performed the worst. Ariadne [Room obs] successfully utilized Ari-Graph word model, achieving performance comparable to the baseline with memory oracle.

Table 1: Ariadne with obscured partial observations performs comparable to NetPlay agent full level information.   

<table><tr><td>Method</td><td>Score</td><td>Levels</td></tr><tr><td>Ariadne (Room obs)</td><td>593.00 ± 202.62</td><td>6.33 ± 2.31</td></tr><tr><td>NetPlay (Room obs)</td><td>341.67 ± 109.14</td><td>3.67 ± 1.15</td></tr><tr><td>NetPlay (Level obs)</td><td>675.33 ± 130.27</td><td>7.33 ± 1.15</td></tr></table>

# 5.3 Multi-hop Q&A

We compared AriGraph with the latest LLM-based approaches that employ knowledge graph construction and retrieval techniques for question answering over documents (Table 2). Our memory architecture, adapted from the Ariadne TextWorld agent, utilizing both GPT-4 and GPT-4omini outperformed baseline methods like ReadAgent (GPT-4), GPT-4 RAG, GPT-4 full context and GraphReader (GPT-4). GraphRAG served as a strong GPT-4o-mini baseline, due to its extremely hight costs. ArigGraph (GPT-4o-mini) showed weaker performance on Musique, but outperformed GraphRAG on HotpotQA. Notably, our approach is more then $1 0 \mathrm { x }$ cheaper in comparison to GraphRAG (Table 3, Appendix D).

The best performance using GPT-4 was achieved by HOLMES, but AriGraph (GPT-4) exhibited comparable results. Notably, all baseline methods were specifically designed for Q&A tasks, incorporating task-specific prompt tuning and additional architectural enhancements. Both GraphRAG and HOLMES employ hyper-relations in their graphs to connect source data with extracted entities, similar to our method. However, these approaches lack mechanisms for updates in dynamic environments, a key advantage of AriGraph.

# 6 Related work

Voyager [Wang et al., 2023a], Ghost in the Minecraft [Zhu et al., 2023] and Jarvis-1 [Wang et al., 2023b] are advanced, open-ended LLM agents that show significantly better performance in Minecraft compared to earlier techniques. These agents feature memory capabilities through a library of learned skills, summaries of successful actions, and episodic memory with plans for successful task execution. However, they fall short in representing knowledge with semantic structure and depend heavily on the LLM’s extensive Minecraft knowledge, or even access to the Minecraft wiki. Generative agents [Park et al., 2023] mimic human behavior in multiagent environments and were among the pioneers in introducing an advanced memory system for LLM agents, which we use as our baseline. Reflexion [Shinn et al., 2023] and CLIN [Majumder et al., 2023] enables agents to reflect on past trajectories, allowing them to store relevant insights about com-

![](images/cde916d75f83ef795d98ae82c4b1618f1f495c4be19167212b35bd4ef640bd2a.jpg)  
Figure 5: AriGraph demonstrate good scaling during learning and with environment size. A size of the knowledge graph quickly saturates during exploration and learning phase. KG grows moderately when the Treasure Hunt and the Cooking games include more rooms and objects in their hard versions.

Table 2: AriGraph memory demonstrates competitive performance on Multi-Hop Q&A datasets. Even in non interactive tasks AriGraph is comparable to strong QA baseline agents. The best results with the base GPT-4o and GPT-4o-mini are shown in bold and underline respectively.   

<table><tr><td rowspan="2">Method</td><td colspan="2">MuSiQue</td><td colspan="2">HotpotQA</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>BM25(top-3)</td><td>25.0</td><td>31.1</td><td>45.7</td><td>58.5</td></tr><tr><td>Ada-002(top-3)</td><td>24.5</td><td>32.1</td><td>45.0</td><td>58.1</td></tr><tr><td>GPT-4 full context</td><td>33.5</td><td>42.7</td><td>53.0</td><td>68.4</td></tr><tr><td>GPT-4 + supporting facts</td><td>45.0</td><td>56.0</td><td>57.0</td><td>73.8</td></tr><tr><td>ReadAgent(GPT-4)</td><td>35.0</td><td>45.1</td><td>48.0</td><td>62.0</td></tr><tr><td>GraphReader(GPT-4)</td><td>38.0</td><td>47.4</td><td>55.0</td><td>70.0</td></tr><tr><td>HOLMES(GPT-4)</td><td>48.0</td><td>58.0</td><td>66.0</td><td>78.0</td></tr><tr><td>AriGraph(GPT-4)</td><td>45.0</td><td>57.0</td><td>68.0</td><td>74.7</td></tr><tr><td>GraphRAG(GPT-4o-mini)</td><td>40.0</td><td>53.5</td><td>58.7</td><td>63.3</td></tr><tr><td>AriGraph(GPT-4o-mini)</td><td>36.5</td><td>47.9</td><td>60.0</td><td>68.6</td></tr></table>

pleted actions in a long-term memory module, but has no structural representation of knowledge and episodic memories. LARP [Yan et al., 2023] utilizes the concepts of episodic and semantic memories but treats them as separate instances and lacks a structural representation of knowledge.

Considerable research is dedicated to leveraging established knowledge graphs for enhancing Q&A [Baek et al., 2023; Li et al., 2024b] systems to address the factual knowledge deficiency observed in LLMs. The latest research demonstrating best performance in Q&A tasks includes Graphreader [Li et al., 2024a], HOLMES [Panda et al., 2024], HippoRAG [Gutierrez ´ et al., 2024], GraphRAG [Edge et al., 2024] which all employ the technique of building knowledge graphs from texts. However, these studies do not address the context of functioning within an interactive environment, nor do they take into account the updates to

knowledge graphs prompted by new experiences.

Text-based environments [Cotˆ e´ et al., 2018; Hausknecht et al., 2019; Shridhar et al., 2021; Wang et al., 2022] were originally designed to evaluate reinforcement learning (RL) agents [Guo et al., 2020; Yao et al., 2020; Ammanabrolu et al., 2020; Ammanabrolu and Hausknecht, 2020; Tuli et al., 2022; Adhikari et al., 2021]. Multiple experiments have already explored the potential of LLMs in these complex scenarios [Tsai et al., 2023; Tan et al., 2023; Momennejad et al., 2023; Ding et al., 2024a]. However raw LLMs show poor results in these games without proper agentic architecture and memory.

# 7 Conclusions

In this paper, we introduced AriGraph, a novel knowledge graph world model tailored for LLM agents. AriGraph uniquely integrates semantic and episodic memories from textual observations, providing a structured and dynamic representation of knowledge. We evaluated this approach across a range of interactive text-based games and multi-hop Q&A benchmarks, comparing it against existing memory architectures. To test its capabilities comprehensively, we developed a cognitive architecture called Ariadne, which combines Ari-Graph with planning and decision-making components.

Our results demonstrate that AriGraph significantly outperforms other memory systems in tasks requiring long-term memory, such as decision-making, planning, and exploration in partially observable environments. The structured knowledge representation provided by AriGraph enables efficient retrieval and reasoning, accelerating learning and task completion. Additionally, AriGraph’s scalability was evident as it maintained high performance even when the complexity of tasks increased, involving more objects and locations. In multi-hop Q&A benchmarks, AriGraph exhibited competitive performance, underscoring its robustness and adaptability beyond interactive environments.

While promising, our approach can be further enhanced by incorporating multi-modal observations, procedural memories, and more sophisticated graph search methods.

# References

[Adhikari et al., 2021] Ashutosh Adhikari, Xingdi Yuan, Marc-Alexandre Cotˆ e, Mikul ´ a´s Zelinka, Marc-Antoine ˇ Rondeau, Romain Laroche, Pascal Poupart, Jian Tang, Adam Trischler, and William L. Hamilton. Learning dynamic belief graphs to generalize on text-based games, 2021.   
[Ammanabrolu and Hausknecht, 2020] Prithviraj Ammanabrolu and Matthew Hausknecht. Graph constrained reinforcement learning for natural language action spaces, 2020.   
[Ammanabrolu et al., 2020] Prithviraj Ammanabrolu, Ethan Tien, Matthew Hausknecht, and Mark O. Riedl. How to avoid being eaten by a grue: Structured exploration strategies for textual worlds, 2020.   
[Baek et al., 2023] Jinheon Baek, Alham Fikri Aji, and Amir Saffari. Knowledge-augmented language model prompting for zero-shot knowledge graph question answering, 2023.   
[Basu et al., 2024] Kinjal Basu, Keerthiram Murugesan, Subhajit Chaudhury, Murray Campbell, Kartik Talamadupula, and Tim Klinger. Explorer: Exploration-guided reasoning for textual reinforcement learning, 2024.   
[Bulatov et al., 2022] Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. Recurrent memory transformer, 2022.   
[Bulatov et al., 2024] Aydar Bulatov, Yuri Kuratov, Yermek Kapushev, and Mikhail S. Burtsev. Scaling transformer to 1m tokens and beyond with rmt, 2024.   
[Chen et al., 2024] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation, 2024.   
[Cheng et al., 2024] Yuheng Cheng, Ceyao Zhang, Zhengwen Zhang, Xiangrui Meng, Sirui Hong, Wenhao Li, Zihao Wang, Zekai Wang, Feng Yin, Junhua Zhao, and Xiuqiang He. Exploring large language model based intelligent agents: Definitions, methods, and prospects, 2024.   
[Cotˆ e´ et al., 2018] Marc-Alexandre Cotˆ e,´ Akos K ´ ad´ ar, ´ Xingdi Yuan, Ben Kybartas, Tavian Barnes, Emery Fine, James Moore, Ruo Yu Tao, Matthew Hausknecht, Layla El Asri, Mahmoud Adada, Wendy Tay, and Adam Trischler. Textworld: A learning environment for text-based games. CoRR, abs/1806.11532, 2018.   
[Ding et al., 2024a] Peng Ding, Jiading Fang, Peng Li, Kangrui Wang, Xiaochen Zhou, Mo Yu, Jing Li, Matthew Walter, and Hongyuan Mei. MANGO: A benchmark for evaluating mapping and navigation abilities of large language models, 2024.   
[Ding et al., 2024b] Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, and Mao Yang. Longrope: Extending llm context window beyond 2 million tokens, 2024.   
[Edge et al., 2024] Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt,

and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization, 2024.   
[Gu and Dao, 2023] Albert Gu and Tri Dao. Mamba: Lineartime sequence modeling with selective state spaces, 2023.   
[Guo et al., 2020] Xiaoxiao Guo, Mo Yu, Yupeng Gao, Chuang Gan, Murray Campbell, and Shiyu Chang. Interactive fiction game playing as multi-paragraph reading comprehension with reinforcement learning, 2020.   
[Gutierrez ´ et al., 2024] Bernal Jimenez Guti ´ errez, Yiheng ´ Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hipporag: Neurobiologically inspired long-term memory for large language models, 2024.   
[Hausknecht et al., 2019] Matthew Hausknecht, Prithviraj Ammanabrolu, Cotˆ e Marc-Alexandre, and Yuan Xingdi. ´ Interactive fiction games: A colossal adventure. CoRR, abs/1909.05398, 2019.   
[Izacard et al., 2022] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning, 2022.   
[Jeurissen et al., 2024] Dominik Jeurissen, Diego Perez-Liebana, Jeremy Gow, Duygu Cakmak, and James Kwan. Playing nethack with llms: Potential and limitations as zero-shot agents, 2024.   
[Kuttler ¨ et al., 2020] Heinrich Kuttler, Nantas Nardelli, ¨ Alexander Miller, Roberta Raileanu, Marco Selvatici, Edward Grefenstette, and Tim Rocktaschel. The nethack ¨ learning environment. Advances in Neural Information Processing Systems, 33:7671–7684, 2020.   
[Lee et al., 2024] Kuang-Huei Lee, Xinyun Chen, Hiroki Furuta, John Canny, and Ian Fischer. A human-inspired reading agent with gist memory of very long contexts, 2024.   
[Li et al., 2024a] Shilong Li, Yancheng He, Hangyu Guo, Xingyuan Bu, Ge Bai, Jie Liu, Jiaheng Liu, Xingwei Qu, Yangguang Li, Wanli Ouyang, Wenbo Su, and Bo Zheng. Graphreader: Building graph-based agent to enhance longcontext abilities of large language models, 2024.   
[Li et al., 2024b] Yihao Li, Ru Zhang, Jianyi Liu, and Gongshen Liu. An enhanced prompt-based llm reasoning scheme via knowledge graph-integrated collaboration, 2024.   
[Majumder et al., 2023] Bodhisattwa Prasad Majumder, Bhavana Dalvi Mishra, Peter Jansen, Oyvind Tafjord, Niket Tandon, Li Zhang, Chris Callison-Burch, and Peter Clark. Clin: A continually learning language agent for rapid task adaptation and generalization, 2023.   
[Momennejad et al., 2023] Ida Momennejad, Hosein Hasanbeig, Felipe Vieira Frujeri, Hiteshi Sharma, Nebojsa Jojic, Hamid Palangi, Robert Ness, and Jonathan Larson. Evaluating cognitive maps and planning in large language models with cogeval. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.   
[Murugesan et al., 2020] Keerthiram Murugesan, Mattia Atzeni, Pavan Kapanipathi, Pushkar Shukla, Sadhana

Kumaravel, Gerald Tesauro, Kartik Talamadupula, Mrinmaya Sachan, and Murray Campbell. Text-based rl agents with commonsense knowledge: New challenges, environments and baselines, 2020.   
[Pan et al., 2024] Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. Unifying large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data Engineering, page 1–20, 2024.   
[Panda et al., 2024] Pranoy Panda, Ankush Agarwal, Chaitanya Devaguptapu, Manohar Kaul, and Prathosh A P. Holmes: Hyper-relational knowledge graphs for multi-hop question answering using llms, 2024.   
[Parisotto et al., 2020] Emilio Parisotto, Francis Song, Jack Rae, Razvan Pascanu, Caglar Gulcehre, Siddhant Jayakumar, Max Jaderberg, Raphael Lopez Kaufman, Aidan Clark, Seb Noury, et al. Stabilizing transformers for reinforcement learning. In International conference on machine learning, pages 7487–7498. PMLR, 2020.   
[Park et al., 2023] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. Generative agents: Interactive simulacra of human behavior, 2023.   
[Pleines et al., 2022] Marco Pleines, Matthias Pallasch, Frank Zimmer, and Mike Preuss. Memory gym: Partially observable challenges to memory-based agents. In The eleventh international conference on learning representations, 2022.   
[Shinn et al., 2023] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023.   
[Shridhar et al., 2021] Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk, Adam Trischler, and´ Matthew Hausknecht. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In Proceedings of the International Conference on Learning Representations (ICLR), 2021.   
[Sorokin et al., 2022] Artyom Sorokin, Nazar Buzun, Leonid Pugachev, and Mikhail Burtsev. Explain my surprise: Learning efficient long-term memory by predicting uncertain outcomes. Advances in Neural Information Processing Systems, 35:36875–36888, 2022.   
[Sumers et al., 2024] Theodore R. Sumers, Shunyu Yao, Karthik Narasimhan, and Thomas L. Griffiths. Cognitive architectures for language agents, 2024.   
[Tan et al., 2023] Qinyue Tan, Ashkan Kazemi, and Rada Mihalcea. Text-based games as a challenging benchmark for large language models, 2023.   
[Trivedi et al., 2022] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 2022.

[Tsai et al., 2023] Chen Feng Tsai, Xiaochen Zhou, Sierra S. Liu, Jing Li, Mo Yu, and Hongyuan Mei. Can large language models play text games well? current state-of-theart and open questions, 2023.   
[Tuli et al., 2022] Mathieu Tuli, Andrew C. Li, Pashootan Vaezipoor, Toryn Q. Klassen, Scott Sanner, and Sheila A. McIlraith. Learning to follow instructions in text-based games, 2022.   
[Wang et al., 2022] Ruoyao Wang, Peter Jansen, Marc-Alexandre Cotˆ e, and Prithviraj Ammanabrolu. Science-´ world: Is your agent smarter than a 5th grader?, 2022.   
[Wang et al., 2023a] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models, 2023.   
[Wang et al., 2023b] Zihao Wang, Shaofei Cai, Anji Liu, Yonggang Jin, Jinbing Hou, Bowei Zhang, Haowei Lin, Zhaofeng He, Zilong Zheng, Yaodong Yang, Xiaojian Ma, and Yitao Liang. Jarvis-1: Open-world multi-task agents with memory-augmented multimodal language models, 2023.   
[Wang et al., 2024] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6), March 2024.   
[Wong Gonzalez, 2018] Daniela Wong Gonzalez. The Relationship Between Semantic and Episodic Memory: Exploring the Effect of Semantic Neighbourhood Density on Episodic Memory. PhD thesis, Electronic Theses and Dissertations, 2018. Paper 7585.   
[Yan et al., 2023] Ming Yan, Ruihao Li, Hao Zhang, Hao Wang, Zhilan Yang, and Ji Yan. Larp: Language-agent role play for open-world games, 2023.   
[Yang et al., 2018] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering, 2018.   
[Yao et al., 2020] Shunyu Yao, Rohan Rao, Matthew Hausknecht, and Karthik Narasimhan. Keep calm and explore: Language models for action generation in text-based games, 2020.   
[Yao et al., 2023] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023.   
[Zhu et al., 2023] Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, Yu Qiao, Zhaoxiang Zhang, and Jifeng Dai. Ghost in the minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory, 2023.

# A Memory Graph Search Details

Pseudo-code for SemanticSearch function in AriGraph is listed in Algorithm 2. This algorithm is close to BFS search. The main difference is the use of retrieval mechanism in function EmbedAndRetrieve. Function EmbedAndRetrieve $( E , q , w )$ uses pretrained Contriever [Izacard et al., 2022] model to compute embeddings for edges $E$ and query $q$ and then returns top $w$ edges with a highest similarity score. Similarity score between edge $e$ and query $q$ is a dot product between their embeddings. Most of the times, when query for EmbedAndRetrieve is a semantic vertex, it simply returns edges incident to this vertex, but also has ability to retrieve edges connected to vertices that are semantically close to the query vertex. For example, semantic graph can contain separate vertices for ”grill” and ”grilling” that are not directly connected, so searching for ”grill” can potentially return edge (”bbq”, ”used for”, ”grilling”).

Algorithm 2: Semantic Search   
Input: search query $q$ , $E_{s}$ , search depth $d$ , search width $w$

Result: relevant vertices semantic $V_{s}^{q}$ and edges $E_{s}^{q}$ $E_{s}^{q}\gets \emptyset$

```txt
[ L \gets \emptyset ] // init empty queue of queries 
```

Enqueue $(L,q)$ //pushqinto queue $L$

while $L$ is not empty do   
$D[q]\gets 0$ // set search distance for $q$ to zero

```latex
\[
\begin{array}{c}
q' \leftarrow \text{Dequeue}(L) // \text{remove} \\
\text{first element from } L
\end{array}
\] 
```

end   
if $D[q^{\prime}]\geq d$ then continue

foreach $e _ { i }$ in $E _ { s } ^ { \prime }$ do   
// use Contriever model to find top $w$ triplets closest to $q^{\prime}$ $E_{s}^{\prime}\gets \mathrm{EmbedAndRetrieve}(E_{s},q^{\prime},w)$

$V_{s}^{\prime}\gets$ IncidentVertices $(e_i)$ //returns two incident semantic vertices

foreach $v$ in $V _ { s } ^ { \prime }$ do   
if $v$ not in $L$ then Enqueue $(L,v)$ $D[v]\gets D[q^{\prime}] + 1$ end

1 end   
end $E_{s}^{q}\gets E_{s}^{q}\cup E_{s}^{\prime}$

end   
return $E_{s}^{q}$

# B Exploration

Before any planning or decision-making occurs, an auxiliary agent assesses the need for exploration based on a pre-

established plan. Depending on this assessment, it either activates or deactivates exploration mode. Moreover, the graph agent extracts triplets containing information about exits, such as ”kitchen, has an unexplored exit, south,” and triplets detailing spatial connections between locations like ”hall, east of, kitchen.” Subsequently, simple graph-based methods can be employed to identify all exits from each room that the agent has detected but not yet explored. This information is then added to the working memory of the agent. Function for finding triplets corresponding to probable unexplored actions is listed in Algorithm 3. Current implementation uses expert knowledge about what elements of the semantic graph can represent locations and exits between them.

Algorithm 3: Unexplored exit detection   
Input: $V_{s}, E_{s}, V_{e}, E_{e}$ , current location $v_{l}$

Result: triplets $E_{s}^{exp}$ with information about unexplored exits from $v_{l}$

$E^{exp}\gets \emptyset$

$E^{out}\gets \mathrm{GetOutgoing}(E_s,v_l)$ // get semantic edges outgoing from $v_{l}$

foreach $e$ in $E ^ { o u t }$ do   
if RepresentExit(e) then $\begin{array}{r}\big{\lfloor}E^{exp}\gets E^{exp}\cup \{e\} \end{array}$

foreach $e$ in $E ^ { i n }$ do   
$E^{in}\gets \mathrm{GetIncomng}(E_s,v_l)$ // get semantic edges incoming in $v_{l}$

if RepresentExit(e) then // remove exits leading to explored locations $E^{exp}\gets$ $\boldsymbol{E}^{exp}\backslash \{\mathrm{FindRelatedExit}(E^{exp},e)\}$

return $E^{exp}$

# C Graph statistics

When working with a graph, it would be advantageous to assess the quality of its construction. Unfortunately, due to the variability in its construction (e.g., the use of synonyms, equivalent restructuring of connections, etc.), direct measurement of its correspondence to the true graph is extremely challenging. Moreover, the inclusion of information in the constructed graph that is not present in the true graph is not always detrimental and can sometimes yield positive effects. For these reasons, we did not measure the quality of the graph but instead measured its growth rate and update rate during gameplay (Fig.5). Furthermore, we conducted a separate experiment in which the graph was constructed according to our pipeline, but the agent’s actions were chosen randomly (Fig.6). The Cleaning environment was chosen for this setting because, unlike the Treasure Hunt, it contains a large variety of objects with which the agent can interact, and unlike the Cooking, the agent cannot fail. In this setting, we also measured the graph’s growth rate and update rate.

From the obtained results, one can conclude that the graph grows most actively during the exploration phase but

![](images/d887921b61512bb127442d1225aa2ef85f618e53fd19574876fbead9d5816a09.jpg)

![](images/445cddbafedbe175087156957c55c1cc870663b7ddbdcecb8c26746d1f33720f.jpg)

![](images/1558986faf78970da8370513b8c113718edae1a1e3220eff783089824289683a.jpg)

![](images/b07a7722998a6fc80e12fe52f17d310d260be1f088da186878ec37bbd099b020.jpg)

![](images/45fa03ad108c32492f758bb1950ad553abd53b2af000a297e8a4bc56b7bcc725.jpg)

Figure 6: Statistics of graph construction and updating during the random walk.   
![](images/15f7d61ce3f55ef1de3e5f4966a8a5a1bd6b1d8c37752a3014fd78bd8567a6c9.jpg)  
Number of triplets Number of items New triplets Replacements

maintains a slower growth rate when the agent operates within the familiar parts of the environment. Additionally, it is evident that the growth rate of the graph decreases as the quality of the LLM used in its construction increases.

# D Token usage

Table 3: Token Usage Analysis for Text Games and QA Tasks   

<table><tr><td>Method</td><td>Prompt Tokens</td><td>Completion Tokens</td></tr><tr><td>Text Games (per step)</td><td></td><td></td></tr><tr><td>Ariadne agent</td><td>6,000</td><td>500</td></tr><tr><td>RAG memory agent</td><td>4,000</td><td>350</td></tr><tr><td>Summary memory agent</td><td>3,800</td><td>350</td></tr><tr><td>Full history memory agent (step 150)</td><td>14,000</td><td>350</td></tr><tr><td>Simulacra memory agent</td><td>7,500</td><td>400</td></tr><tr><td>Reflexion memory agent</td><td>5,800</td><td>350</td></tr><tr><td>QA Benchmarks (per task)</td><td></td><td></td></tr><tr><td>AriGraph</td><td>11,000</td><td>2,500</td></tr><tr><td>GraphRAG</td><td>115,000</td><td>20,000</td></tr></table>

# Triplet Extraction Prompt

Guidelines for Building the Knowledge Graph:

Creating Nodes and Triplets: Nodes should depict entities or concepts, similar to Wikipedia nodes. Use a structured triplet format to capture data, as follows : "subject, relation, object". For example, from Albert Einstein, born in Germany, is known for developing the theory of relativity," extract "Albert Einstein, country of birth, Germany; Albert Einstein , developed, Theory of Relativity."

Remember that you should break complex triplets like "John, position, engineer in Google" into simple triplets like "John, position, engineer", "John, work at, Google".

Length of your triplet should not be more than 7 words. You should extract only concrete knowledges, any assumptions must be described as hypothesis.

For example, from phrase "John have scored many points and potentiallyy will be winner" you should extract "John , scored many, points; John, could be, winner" and should not extract "John, will be, winner".

Remember that object and subject must be an atomary units while relation can be more complex and long.

If observation states that you take item, the triplet shoud be: ’item, is in, inventory’ and nothing else.

Do not miss important information. If observation is ’book involves story about knight, who needs to kill a dragon’, triplets should be ’book, involves, knight’, ’knight, needs to kill, dragon’. If observation involves some type of notes, do not forget to include triplets about entities this note includes.

There could be connections between distinct parts of observations. For example if there is information in the beginning of the observation that you are in location, and in the end it states that there is an exit to the east, you should extract triplet: ’ location, has exit, east’.

Several triplets can be extracted, that contain information about the same node. For example ’kitchen, contains, apple’, ’kitchen, contains, table’, ’apple, is on, table’. Do not miss this type of connections.

Other examples of triplets: ’room z, contains, black locker ’; ’room x, has exit, east’, ’apple, is on, table’, ’ key, is in, locker’, ’apple, to be, grilled’, ’potato , to be, sliced’, ’stove, used for, frying’, ’recipe, requires, green apple’, ’recipe, requires, potato’.

Do not include triplets that state the current location of an agent like ’you, are in, location’.

Do not use ’none’ as one of the entities.

If there is information that you read something, do not forget to incluse triplets that state that entitie that you read contains information that you extract.

Example of triplets you have extracted before: {example}

Observation: {observation}

Remember that triplets must be extracted in format: " subject_1, relation_1, object_1; subject_2, relation_2, object_2; ...

Extracted triplets:’’’

# Outdated Triplet Selection Prompt

The triplets denote facts about the environment where the player moves. The player takes actions and the environment changes, so some triplets from the list of existing triplets can be replaced with one of the new triplets. For example, the player took the item from the locker and the existing triplet "item, is in , locker" should be replaced with the new triplet " item, is in, inventory".

Sometimes there are no triplets to replace:

Example of existing triplets: "Golden locker, state, open"; "Room K, is west of, Room I"; "Room K, has exit, east".

Example of new triplets: "Room T, is north of, Room N"; "

Room T, has exit, south".

Example of replacing: []. Nothisg to replace here

Sometimes several triplets can be replaced with one:

Example of existing triplets: "kitchen, contains, broom"; " broom, is on, floor".

Example of new triplets: "broom, is in, inventory".

Example of replacing: [["kitchen, contains, broom" -> " broom, is in, inventory"], ["broom, is on, floor" -> "broom, is in, inventory"]]. Because broom changed location from the floor in the kitchen to players inventory.

Ensure that triplets are only replaced if they contain redundant or conflicting information about the same aspect of an entity. Triplets should not be replaced if they provide distinct or complementary information about entities compared to the new triplets. Specifically, consider the relationships, properties, or contexts described by each triplet and verify that they align before replacement. If there is uncertainty about whether a triplet should be replaced, prioritize retaining the existing triplet over replacing it. When comparing existing and new triplets, if they refer to different aspects or attributes of entities, do not replace them. Replacements should only occur when there is semantic duplication between an existing triplet and a new triplet.

Example of existing triplets: "apple, to be, cooked", ’ knife, used for, cutting’, ’apple, has been, sliced’

Example of new triplets: "apple, is on, table", ’kitchen, contsins, knife’, ’apple, has beed, grilled’.

Example of replacing: []. Nothing to replace here. These triplets describe different properties of items, so they should not be replaced.

Another example of when not to replase existung triplets: Example of existing triplets: "brush, used for, painting". Example of new triplets: "brush, is in, art class".

Example of replacing: []. Nothing to replace here. These triplets describe different properties of brush, so they should not be replaced.

I repeat, do not replace triplets, if they carry differend type of information about entities!!! It is better to leave a tripplet, than to replace the one that has important information. Do not state that triplet needs to be replaced if you are not sure!!!

If you find triplet in Existing triplets which semantically duplicate some triplet in New triplets, replace such triplet from Existing triplets. However do not replace triplets if they refer to different things.

####

Generate only replacing, no descriptions are needed.

Existing triplets: {ex_triplets}.

New triplets: {new_triplets}.

####

Warning! Replacing must be generated strictly in following format: [[outdated_triplet_1 $- >$ actual_triplet_1], [ outdated_triplet_2 $- >$ actual_triplet_2], ...], you MUST NOT include any descriptions in answer.

Replacing:

####

INSTRUCTION:

You will be provided with sub-goals and reasons for it from plan of an agent. Your task is to state if this sub goals require exploration of the environment, finding or locating something.

Answer with just True or False.

####

Plan:

{plan0}

# Planning prompt

# ####

INSTRUCTION:

You are a planner within the agent system tasked with navigating the environment in a text-based game.

Your role is to create a concise plan to achieve your main goal or modify your current plan based on new information received.

Make sure your sub-goals will benefit the achivment of your main goal. If your main goal is an ongoing complex process, also put sub-goals that can immediately benifit achiving something from your main goal.

If you need to find something, put it into sub-goal.

If you wish to alter or delete a sub-goal within the current plan, confirm that this sub-goal has been achieved according to the current observation or is no longer relevant to achieving your main goal. Untill then do not change wording in "sub_goal" elements of your plan and their position in the plan. Only change wording in "reason" part to track the progress of completion of sub-goals.

If sub-goal was completed or confirmed to be no more relevant, delete it, replase it with new one or with lower priority sub-goals from the plan. Untill then keep the structure of sub-goals as it is. Create new sub-goals only if they will benifit your main goal and do not prioritize them over current sub-goals.

If your task is to obtain something, make shure that the item is in your inventory before changing your subgoal.

Your plan contains important information and goals you need to complete. Do not alter sub-goals or move them in hierarchy if they were not completed!

Pay attention to your inventory, what items you are carring , when setting the sub-goals. These items might be important.

Pay attention to information from your memory module, it is important.

There should always be at least one sub-goal.

State the progress of completing your sub-goals in "reason" for each sub-goal.

```txt
Write your answer exactly in this json format:  
{ "main_goal": "...",  
"plan_steps": {[  
    "sub_goal_1": "...",  
    "reason": "...}  
},  
{  
    "sub_goal_2": "...",  
    "reason": "...}  
},  
{  
    "sub_goal_...": "...",  
    "reason": "...}  
}],  
"your(emotion):  
{ "your_current(emotion": "emotion",  
"reason Behind emotion": "..."} 
```

Do not write anything else.

# ####

1. Main goal: {main_goal}   
2. History of {n_prev} last observations and actions: { observations}   
3. Your current observation: {observation}   
4. Information from the memory module that can be relevant to current situation: {subgraph}   
5. Your {topk_episodic} most relevant episodic memories from the past for the current situation: { top_episodic}.   
6. Your previous plan: {plan0}   
*if is explore* 7. Yet unexplored exits in the environment: {all_unexpl_exits}

# ReAct decision making prompt

# ####

INSTRUCTION:

You are an action selector within an agent system designed to navigate an environment in a text-based game. Your

role involves receiving information about an agent and the state of the environment alongside a list of possible actions.

Your primary objective is to choose an action from the list of possible actions that aligns with the goals outlined in the plan, giving precedence to main goal or sub-goals in the order they appear (main goal is highest priority, then sub_goal_1, sub_goal_2, etc.). However, prioritize sub-goals that can be solved by perfroming single action in current situation, like ’ take something’, over long term sub-goals.

Actions like "go to ’location’" will move an agent directly to stated location, use them instead of "go_west’ type of actions, if the destination you want to move to is further than 1 step away.

In tasks centered around exploration or locating something, prioritize actions that guide the agent to previously unexplored areas. You can deduce which locations have been visited based on the history of observations and information from your memory module.

Performing same action typically will not provide different results, so if you are stuck, try to perform other actions or prioritize goals to explore the environment.

You may choose actions only from the list of possible actions. You must choose strictly one action.

Write your answer exactly in this json format:

```txt
"reason_for_action": "reason"  
"action_to.take": "selected action" 
```

}

Do not write anything else.

#

1. Main goal: {main_goal}   
2. History of {n_prev} last observations and actions: { observations}   
3. Your current observation: {observation}   
4. Information from the memory module that can be relevant to current situation: {subgraph}   
5. Your {topk_episodic} most relevant episodic memories from the past for the current situation: { top_episodic}.   
6. Your current plan: {plan0}   
7. Yet unexplored exits in the environment: { all_unexpl_exits}

Possible actions in current situation: {valid_actions}

# Summarization prompt

# ####

INSTRUCTION:

You are a guide within a team of agents engaging in a textbased game. Your role is to concisely yet thoroughly detail all the essential aspects of the current situation. Ensure that your summary aids in information extraction and facilitates the decisionmaking process by focusing on pertinent details and excluding extraneous information. Incorporate a strategic outlook in your narrative, emphasizing information integral to forming a tactical plan.

Accurately relay the outcomes of previously attempted actions, as this is pivotal for shaping subsequent choices. Your account will form the sole basis on which the decision-making agents operate; thus, clarity and avoidance of potential confusion are paramount.

Be judicious with your inferences, presenting only wellsubstantiated information that is likely to be of practical benefit. Your account should be succinct, encapsulated within a maximum of three paragraphs.

# ####

1. Main goal: {main_goal}   
2. History of {n_prev} last observations and actions: { observations}   
3. Your current observation: {observation}   
4. Your previous summary: {summary}   
Your summary:

# F Text-based Games

Treasure Hunting. The primary objective is to unlock the golden locker and retrieve the treasure hidden within. The game consists of rooms each featuring a locker of a different color. Initially, the player finds a key in the first room, along with instructions detailing which specific locker this key can unlock. Each subsequent locker contains another key and a note directing the player to the next key’s location, creating a chain of clues and discoveries leading to the golden locker. The easy variation has 12 rooms and 4 keys and hard one has 16 rooms and 5 keys, however second key is significantly harder to find. The hardest variation consist of 36 total rooms, 7 keys and also additional items in each room as noise. Agent receives 1 point for picking a key and 1 point for completing the game. Examples of the Treasure Hunt environment are presented in figures 7, 8 9

Cleaning. The goal is to clean the house that consists of 9 distinct rooms, each designated for a specific purpose, such as a pool, kitchen, etc. Each room contains items, some of which are misplaced, for example, a toothbrush found in the dining room. There are total of 11 misplaced items. The agent’s objective is to tidy up the house by identifying items that are out of place and returning them to their appropriate locations. This task requires the agent to utilize reasoning skills to determine the proper locations for items and to effectively recall from memory where each item belongs, while simultaneously managing multiple tasks. Agent receives 1 point for picking up every displaced items, 1 point for placing an item in the right location and -1 for manipulation with correctly placed item. Our task is conceptually similar to the TextWorld Commonsense (TWC) benchmark[Murugesan et al., 2020]. However, while TWC primarily centers on logical reasoning within one or at most two locations, our emphasis is on environmental exploration and the testing of memory based on past observations. Consequently, we have substantially expanded the number of rooms and items in our setup. Example of the Cleaning environment can be found in figure 10.

Cooking. The goal is to prepare and consume a meal. First an agent needs to find a recipe that provides detailed instructions, including required ingredients and specific preparation methods such as dicing, slicing, chopping, grilling, frying, and roasting. Medium difficulty task features 9 locations and 3 ingredients and hard task features 12 locations and 4 ingredients. The agent receives points for correctly selecting ingredients and executing the proper procedures. The game is considered lost if the agent performs any incorrect action or uses an inappropriate tool for a given ingredient. We enhanced the initial observation with instructions and provided explanations for specific actions to tailor the task for LLM models, particularly concerning the appropriate use of household kitchen appliances for different actions with ingredients. For instance, a BBQ should be used for grilling, while a stove is appropriate for frying (see Appendix E for prompt). This allows to test the agent’s ability to remember and adhere to instructions. Examples of the Cooking environment are presented in figures 11, 12.

RL comparison benchmark. We tested our architecture

on a variation of the Cooking test from [Adhikari et al., 2021] to compare it with RL baselines. These tasks have 4 levels of difficulty, however, they are significantly simpler than our main tasks, having fewer locations, ingredients, and required actions.

• Level 1: 1 ingredient, 1 room, cutting.   
• Level 2: 1 ingredient, 1 room, cutting $^ +$ cooking.   
• Level 3: 1 ingredient, 9 rooms, no processing.   
• Level 4: 3 ingredients, 6 rooms, cutting $^ +$ cooking.

We tested or agent and raw GPT-4 with full history on 3 randomly generated environments for each task difficulty. We slightly adapted the task for the LLM models. We gave the instruction to grill, roast or fry the ingredient using appropriate kitchen tools such as BBQ, oven and stove in the first observation of the game. RL agent can learn this rules by interacting with the environment, however, this is not a commonsense knowledge and without this instruction the game becomes unbeatable for LLM-agents and human players. While this adaptation should not impact the difficulty level, it’s important to note that the environments for comparison between LLM and RL models were not $100 \%$ identical.

# G Additional graphics and tables

This section presents the step-by-step dynamics of performance of different agents and human players (figure.14) . Additionally, this section includes a table summarizing the results of all experiments across three versions of the Treasure Hunt (Medium, Hard, Hardest), three versions of the Cooking (Medium, Hard, Hardest), and the Cleaning (table 4).

![](images/62705c65bcb268deb6ce281e7abbb581044bd81abb72d0386599454c16d7e304.jpg)  
Figure 7: Treasure Hunt environment

Table 4: All normalised scores across all tasks in TextWorld environment. Based on the results, it is evident that the agent with AriGraph significantly outperforms all baselines and scales well to larger and more complex environments. An important outcome is that our agent demonstrated near-human performance in text-based games, which has not been previously achieved using LLM.   

<table><tr><td></td><td>Treasure Hunt</td><td>Treasure Hunt Hard</td><td>Treasure Hunt Hardest</td><td>Cooking</td><td>Cooking Hard</td><td>Cooking Hardest</td><td>Cleaning</td></tr><tr><td>Full History</td><td>0.47</td><td>-</td><td>-</td><td>0.18</td><td>-</td><td>-</td><td>0.05</td></tr><tr><td>Summary</td><td>0.33</td><td>0.17</td><td>-</td><td>0.52</td><td>0.21</td><td>-</td><td>0.35</td></tr><tr><td>RAG</td><td>0.33</td><td>0.17</td><td>-</td><td>0.36</td><td>0.17</td><td>-</td><td>0.39</td></tr><tr><td>Reflexion</td><td>0.93</td><td>-</td><td>-</td><td>1.0</td><td>-</td><td>-</td><td>0.27</td></tr><tr><td>Simulacra</td><td>0.4</td><td>-</td><td>-</td><td>0.3</td><td>-</td><td>-</td><td>0.7</td></tr><tr><td>AriGraph</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.65</td><td>0.79</td></tr><tr><td>AriGraph w/o exploration</td><td>0.87</td><td>-</td><td>-</td><td>0.87</td><td>-</td><td>-</td><td>0.76</td></tr><tr><td>AriGraph w/o episodic</td><td>1.0</td><td>0.67</td><td>-</td><td>0.64</td><td>0.45</td><td>-</td><td>0.92</td></tr><tr><td>AriGraph LLaMA-3-70B</td><td>0.47</td><td>-</td><td>-</td><td>0.67</td><td>-</td><td>-</td><td>0.5</td></tr><tr><td>Human Top-3</td><td>1.0</td><td>-</td><td>-</td><td>1.0</td><td>-</td><td>-</td><td>1.0</td></tr><tr><td>Human All</td><td>0.96</td><td>-</td><td>-</td><td>0.32</td><td>-</td><td>-</td><td>0.59</td></tr></table>

![](images/414a41ff6ad42dfaf5781cbb6ea053cc65cbd05f8345c2a2bd0891bec46cbbdc.jpg)  
Figure 8: Treasure Hunt hard environment

![](images/5ac5c05ed7a9f9ec7cfce8ecde9dba71a9a05c721343471e386087a02f6662dc.jpg)  
Figure 9: Treasure Hunt hardest environment

![](images/29a8f859e0730a1c23e71bfa33fa5a34686241f2698df22773958b322096428e.jpg)  
Figure 10: Cleaning environment

![](images/d7c0b421f6275885e7108d8b162a9c8d4efcdc11aa3651123290d6416204156f.jpg)  
Figure 11: Cooking environment

![](images/fb6925f7591d4c2ec131c934bd257fe27d04934c1ed61fbd2c655c2838eaa1fa.jpg)  
Figure 12: Cooking hard environment

![](images/c6d30eeb0f495e5da3db58becb82d8427dbab9e26e34832b13996990c565da82.jpg)  
Figure 13: Example of NetHack level

![](images/431a77ba92d772eac51042d43d05f7bd1816e56bab99165f62b6604cfb560b9d.jpg)

![](images/292ec7669b10dc53fcd7621234286d67e600d9eca0e89d9b12c0459a381e4489.jpg)

![](images/3249f92b57f682735a480aeff9002dcc0cb0ac9b006a9cbb04bbde981741f76e.jpg)

![](images/813294448edc24fa842f94de8642ca2dabf18deed1a1d202c07be378f2bdd62c.jpg)

Figure 14: Performance Dynamics in Test Games. In the Treasure Hunt, the Ariadne agent delivers performance comparable to top players; in the Cleaning task, it falls slightly behind, but in the Cooking, it surpasses top human players in speed. The hard variants of the tasks demonstrate that the quality of Ariadne’s performance does not decrease with increasing task difficulty, and also highlight the importance of episodic memory.   
![](images/c6fa1b3b128583047cf7e2752e0b7f813c2820baa3b10271ddb8ceda07265aaf.jpg)  
Ariadne   
- Ariadne w/o exp   
Ariadne w/o episodic   
Human Top-3   
_ -- Human All   
Fullhist   
---RAG   
.... Summary
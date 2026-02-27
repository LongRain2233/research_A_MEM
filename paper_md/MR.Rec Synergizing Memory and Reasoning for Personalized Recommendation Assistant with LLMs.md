# MR.Rec: Synergizing Memory and Reasoning for Personalized Recommendation Assistant with LLMs

Jiani Huang*

jianihuang01@gmail.com

The Hong Kong Polytechnic University

Hong Kong SAR, China

Lianghao Xia†

aka_xia@foxmail.com

The University of Hong Kong

Hong Kong SAR, China

# Abstract

The application of Large Language Models (LLMs) in recommender systems faces key challenges in delivering deep personalization and intelligent reasoning, especially for interactive scenarios. Current methods are often constrained by limited context windows and single-turn reasoning, hindering their ability to capture dynamic user preferences and proactively reason over recommendation contexts. To address these limitations, we propose MR.Rec, a novel framework that synergizes memory and reasoning for LLM-based recommendations. To achieve personalization, we develop a comprehensive Retrieval-Augmented Generation (RAG) system that efficiently indexes and retrieves relevant external memory to enhance LLM personalization capabilities. Furthermore, to enable the synergy between memory and reasoning, our RAG system goes beyond conventional query-based retrieval by integrating reasoningenhanced memory retrieval. Finally, we design a reinforcement learning framework that trains the LLM to autonomously learn effective strategies for both memory utilization and reasoning refinement. By combining dynamic memory retrieval with adaptive reasoning, this approach ensures more accurate, context-aware, and highly personalized recommendations. Extensive experiments demonstrate that MR.Rec significantly outperforms state-of-the-art baselines across multiple metrics, validating its efficacy in delivering intelligent and personalized recommendations. We will release code and data upon paper notification.

# 1 Introduction

As digital content continues to grow, recommender systems (RSs) have become widely adopted across web platforms to filter massive information spaces and provide users with personalized content [34]. With the rapid advancement of artificial intelligence, user expectations have evolved toward more intelligent and personalized recommendation assistants. Specifically, users now expect these systems to understand natural language queries—similar to voice assistants like Siri—and provide immediate, personalized recommendations [7, 10, 38]. Traditional RSs, such as matrix factorization (MF) [19] or graph neural network (GNN)–based methods [29], primarily rely on implicit signals from interaction histories (e.g.,

Xingchen Zou*

xzou428@connect.hkust-gz.edu.cn

The Hong Kong University of Science and Technology

(Guangzhou)

Guangzhou, China

Qing Li†

qing-prof.li@polyu.edu.hk

The Hong Kong Polytechnic University

Hong Kong SAR, China

Recommend a T-shirt for my sport clas

Memory

Global Memory: for cloth scenario, users typically care about color, prize, style… Local Memory: the user is a girl who prefer…, she might dislike…, she cares

Reasoning

Based on user’s preference, query and my reasoning, I would recommend: a white T-shirt with simple design, woman style, cozy size for sport without sport icons, made with high-quality cotton and breathable fabric.

![](images/758f75c5124edf338b2df06e6ff8a49b9623a59d1a3ac82360e130dfe36fc40c.jpg)  
Figure 1: Illustration of Personalized and Intelligent LLMbased Recommendation Assistant.

clicks, purchases). While effective in many contexts, they cannot interpret natural language inputs, limiting their ability to function as intelligent, conversational recommendation assistants.

The advent of large language models (LLMs) has opened new possibilities for developing intelligent, interactive recommendation assistants in various domains [26, 41]. With their advanced language understanding capability, extensive general knowledge and instruction-following ability [42], LLMs have the potential to understand user natural language queries and serve as intelligent recommendation assistants. Therefore, existing research has explored the integration of LLMs into recommendation tasks [4, 8]. Despite recent advancements, challenges remain in achieving truly personalized and intelligent recommendation assistants, primarily due to the limited ability of LLMs to memorize user preferences and reason effectively on recommendation tasks.

One of the primary obstacles lies in personalization, as the assistant must be able to maintain a comprehensive user memory that captures user profiles, evolving preferences, and contextual histories [18, 33]. This persistent memory allows the assistant to anticipate user needs and deliver relevant recommendations seamlessly. For example, as illustrated in Figure 1, when a user asks for T-shirt recommendations, the pre-modeled user memory enables the assistant to understand the user’s past preferences (e.g., liking white color, simple design, and high quality) and inferred user profile (e.g., potential gender), thereby facilitating highly personalized and contextual recommendations that go beyond simple query matching. Existing LLM-based RSs typically inject user personalization information into LLMs either by prompting recent interaction histories [17, 25] or by providing pre-generated, static user profiles [24, 43]. However, these methods are often constrained by

the limited context window of LLMs, which prevents the assistant from fully capturing a user’s profile and evolving preferences. In addition, they may introduce irrelevant or noisy information that is not aligned with the current query, thereby distracting the model and reducing its ability to generate accurate recommendations.

While personalization is essential, it does not fully address the challenge of building a capable recommendation assistant. A complementary requirement is intelligence, namely the ability to actively explore the factors behind effective recommendations and perform in-depth multi-step reasoning [5, 35]. In the recommendation domain, such reasoning is essential for interpreting user queries, uncovering underlying intents, and generating accurate suggestions. As illustrated in Figure 1, when a user requests a T-shirt for sports, the recommendation assistant should reason that breathable fabric is essential for athletic activities, and then integrate this reasoning with the user’s memory to generate a more intelligent and contextually appropriate recommendation. Recent works have begun to activate LLMs’ reasoning in recommendations [1, 27]; for instance, Tang et al. [22] enhance user representations through implicit multi-step reasoning in sequential recommendation. Nevertheless, these efforts generally constrain the LLM to reason only over the limited information provided in the pre-defined prompt template, without enabling it to actively explore which additional user memories are helpful for resolving the current recommendation problem. This decoupled paradigm between reasoning and memory restricts the assistant to shallow reasoning, limiting its capacity to effectively leverage stored memory and perform coherent, multi-step reasoning essential for generating personalized and contextually grounded recommendations.

To address these challenges, we propose MR.Rec, a model that synergizes Memory and Reasoning for LLM-based Recommendation assistants. First, we develop a comprehensive Retrieval-Augmented Generation (RAG) system that efficiently indexes and retrieves relevant external memory to enhance LLM personalization capabilities. Our RAG system employs a hierarchical memory indexing mechanism that consists of User-specific Local Memory and Cross-user Global Memory. User-specific Local Memory captures user-specific information at multiple granularities, progressively organizing interaction histories, preference patterns, and user profiles to comprehensively represent personalized preferences, while Cross-user Global Memory aggregates cross-user knowledge to provide broader experiential understanding for diverse recommendation scenarios. Second, to enable the synergy between memory and reasoning, our RAG system goes beyond conventional query-based retrieval by integrating reasoning-enhanced memory retrieval. Rather than directly retrieving based on surface-level query similarity, our approach first leverages reasoning to identify implicit user preferences and relevant preference dimensions, then selectively retrieves memory segments accordingly. Furthermore, we extend LLM reasoning beyond simple chain-of-thought to encompass multi-step memory retrieval stages, creating an adaptive framework where reasoning and information gathering are interleaved to enable dynamic memory exploration and evidencegrounded inference that significantly enhances reasoning depth and context awareness. Finally, we design a reinforcement learning framework that trains the LLM to autonomously learn effective

strategies for both memory utilization and reasoning refinement. Through diverse reward mechanisms that guide memory retrieval, reasoning quality, and recommendation accuracy, our end-to-end optimization enables the model to develop adaptive behaviors that optimize long-term recommendation performance while efficiently leveraging the hierarchical memory structure.

In summary, our contributions can be summarized as follows:

• We propose a comprehensive RAG system with hierarchical memory indexing that efficiently captures and retrieves personalized and cross-user knowledge, significantly extending external memory capabilities for LLM-based recommendations.   
• We develop a reinforcement learning paradigm with novel rewards to harmonize memory access and recommendation reasoning, enabling the LLM to autonomously learn effective memory utilization strategies for final recommendation.   
• Extensive experiments demonstrate that our MR.Rec outperforms state-of-the-art baselines, validating its efficacy in delivering intelligent, personalized recommendations.

# 2 Preliminary

Memory-Enhanced Recommendation. Memory enables recommenders to retain user interaction histories and contextual information for personalized recommendations [30]. Given users $\mathcal { U } = \{ u _ { 1 } , u _ { 2 } , . . . , u _ { | U | } \}$ and items $J = \{ i _ { 1 } , i _ { 2 } , . . . , i _ { | I | } \}$ , each user $u$ has interaction history $\mathcal { H } _ { u } = \{ ( i _ { t } , r _ { t } ) \} _ { t = 1 } ^ { T _ { u } }$ , where $i _ { t }$ is the interacted item and $r _ { t }$ is auxiliary signals (ratings, reviews). A memory function $M ( \cdot )$ transforms this history into structured memory:

$$
\mathcal {M} _ {u} = \mathcal {M} \left(\mathcal {H} _ {u}\right), \tag {1}
$$

where $\mathcal { M } _ { u }$ encodes preferences as embeddings, textual summaries, or hybrid structures for LLM retrieval during recommendation.

Reasoning in LLM-based Recommendation. Effective LLMbased recommendation requires explicit reasoning to decompose complex queries and adaptively utilize relevant memory. Rather than direct prediction, LLMs should generate interpretable reasoning trajectories $\mathcal { T }$ that systematically trace from query understanding through memory analysis to final recommendation justification. This reasoning-driven approach enables dynamic memory exploration and provides transparent recommendation decisions.

Problem Statement. We develop a memory-enhanced recommendation system that generates personalized recommendations through explicit reasoning. Given user $u$ with textual query $q _ { u }$ and memory $M _ { u }$ , our system generates an ideal item profile $i ^ { * }$ via reasoning-enhanced LLM:

$$
i ^ {*} = \operatorname {L L M} \left(\mathcal {P} \left(q _ {u}, \mathcal {M} _ {u}\right); \Theta\right), \tag {2}
$$

where $\mathcal { P } ( \cdot )$ integrates query and memory into structured prompts, and $\Theta$ are learnable parameters optimized for reasoning quality. We then retrieve top- $k$ items from item pool $\boldsymbol { \mathcal { T } }$ based on similarity:

$$
R _ {u} = \arg \operatorname {t o p} - k _ {i \in I} \sin \left(i ^ {*}, \text {m e t a d a t e} (i)\right). \tag {3}
$$

Our objective is to jointly optimize memory utilization and reasoning parameters $\Theta$ to improve recommendation accuracy.

# 3 Methodology

This section elaborates technical details of our MR.Rec framework. The overall framework is depicted in Figure 2.

![](images/0e07273d376ea3a924190d56a9946f6b2327009c19339384be3bb1c9658450dd.jpg)  
(c) Reinforcement Learning for Memory-synergized Reasoning   
Figure 2: The overall model architecture of the proposed MR.Rec framework.

# 3.1 Memory Indexing for RAG Recommender

To enable LLMs to effectively and efficiently obtain external information from vast user interaction records, we propose a retrievalaugmented generation (RAG)-based External Memory Mechanism for recommendation. This RAG module adopts a Hierarchical Memory Indexing method. Specifically, effective memory indexing for recommendation faces two key challenges. First, using raw interaction histories can exceed context limits and cause overfitting to recent behavior while introducing irrelevant noise. Second, recommender systems need both fine-grained user-specific memories and comprehensive knowledge aggregated across users to provide robust recommendations. Therefore, we compress interaction data into the following two complementary memory structures:

3.1.1 User-specific Local Memory. User-specific memory involves users’ purchase histories, ratings, and reviews [20]. This data can easily exceed LLMs’ context window, but using users’ recent interactions may overlook their long-term preferences and overfit their short-term behaviors. This makes it infeasible to directly use the raw interaction data. More essentially, raw interaction histories are inherently noisy: not all past behaviors are equally relevant to the current recommendation query. For instance, past grocery purchases are irrelevant when recommending home appliances, while electronics-related behaviors provide more reliable signals.

To address these challenges, we organize user preferences at multiple granularities, enabling selective retrieval of relevant behavioral patterns while filtering noise, specifically including:

• Behavior Records: It preserves the original interaction data of a user, such as purchased items, ratings, and reviews.   
• Preference Patterns: Moving beyond raw behavioral records, this part indexes preference patterns within item categories. We partition interaction history into different domains (e.g., electronics, clothing) and apply LLM to summarize compact preference

descriptions. This filters out irrelevant cross-category signals while preserving salient information. Formally, given user interaction history $H _ { u }$ , we partition it into subsets $H _ { u } ^ { c }$ by category ?? and use LLM to summarize category-specific user preference $P _ { u } ^ { c }$

$$
P _ {u} ^ {c} = f _ {\text {L L M}} \left(H _ {u} ^ {c}\right). \tag {4}
$$

• User Profile: To model coherent preference structures across domains, we construct a higher-level abstraction integrating fragmented category-specific patterns into consistent cross-category preference structures. Formally, the user profile is expressed as:

$$
U _ {u} = f _ {\mathrm {L L M}} \left(\left\{P _ {u} ^ {c}: c \in C _ {u} \right\}\right), \tag {5}
$$

where $C _ { u }$ denotes the set of categories associated with user $u$ The resulting profile $U _ { u }$ serves as a compact, high-level summary that complements fine-grained preference patterns.

Since preference patterns and user profiles are textual summaries, we apply chunking to decompose them into smaller, semantically coherent segments, enabling efficient storage and selective retrieval of the most pertinent information during recommendation.

3.1.2 Cross-User Global Memory. To capture shared behavioral patterns from the entire dataset, MR.Rec constructs a global memory that captures common decision-making dimensions across users within similar recommendation scenarios, serving as a complementary knowledge source. For example, in baby product recommendations, such memory reveals critical domain-specific factors like safety certifications, ergonomic design, and age-appropriate features that users typically consider but rarely explicitly mention.

Concretely, for each recommendation scenario ?? (e.g., a category in e-commerce), we sample a set of queries $Q _ { s } = \{ q _ { 1 } , q _ { 2 } , . . . , q _ { m } \}$ from the training corpus, each paired with its ground-truth item $i ^ { + } \in \mathcal { I } _ { s }$ . To uncover latent dimensions of decision-making, we incorporate negative items ${ \cal T } _ { s } ^ { - } \subset { \cal T } _ { s }$ from the same scenario that were not chosen. These alternatives offer valuable contrasts, enabling the model to identify why the selected item was favored and to derive more granular, adaptable decision criteria. Formally, we define the

global memory construction process as follows:

$$
M _ {\text {g l o b a l}} = f _ {\mathrm {L L M}} \left(\left\{\left(q, i ^ {+}, I _ {s} ^ {-}\right) \mid q \in Q _ {s} \right\}\right), \tag {6}
$$

where $M _ { s }$ denotes the global memory for scenario ??, and $f _ { \mathrm { L L M } } ( \cdot )$ extracts organized aspects and rationales.

# 3.2 Unifying Memory and Reasoning for Recommendation

3.2.1 Reasoning-enhanced Memory Retrieval. With the indexed recommendation memory, our RAG-based recommender retrieves user interaction patterns relevant to the query to enhance recommendation accuracy. A critical challenge in memory retrieval for recommendations is implicit user preferences that remain unstated in queries. Different recommendation scenarios engage diverse preference dimensions that users often omit unintentionally. For instance, when requesting clothing recommendations, users may have unstated considerations about material, style, or brand reputation that are crucial for personalized suggestions. This challenge causes existing static retrieval methods to fail, as they rely solely on surface-level query similarity without understanding the deeper preference dimensions. Consequently, such approaches can incorporate irrelevant data or miss critical information, limiting the assistant’s ability to provide personalized recommendations.

To address these limitations, we propose reasoning-enhanced memory retrieval, which integrates reasoning and memory access in a dynamic, iterative process. Given a user query, the LLM first analyzes the recommendation scenario to identify relevant preference aspects that should guide personalization, leveraging global memory to understand typical evaluation criteria. The LLM then uses these reasoning-inferred aspects to selectively retrieve segments from user-specific local memory, enabling progressive refinement of understanding. Formally, we identify preference aspects by:

$$
\mathcal {A} _ {q} = f _ {\text {L L M}} (q, M _ {\text {g l o b a l}}), \tag {7}
$$

where $f _ { \mathrm { L L M } } ( \cdot )$ represents the reasoning process over the query and global memory, and $\mathcal { A } _ { q }$ denotes the identified preference aspects.

We then use the identified aspects to selectively retrieve relevant segments from the user-specific local memory, reducing noise and enhancing personalization:

$$
\hat {M} _ {u} (q) = g _ {\text {r e t r i e v a l}} \left(\mathcal {A} _ {q}, M _ {\text {l o c a l}}\right), \tag {8}
$$

where $g _ { \mathrm { r e t r i e v a l } } ( \cdot )$ denotes the LLM-based retrieval function conditioned on the reasoning-inferred aspects, and $\hat { M } _ { u } ( q )$ represents the retrieved local memory segments.

3.2.2 Retrieval-Augmented Item Generation. Finally, the LLM integrates the retrieved memories to generate an ideal item profile that aligns with both the user’s query and personalized preferences:

$$
\mathcal {I} _ {u} (q) = f _ {\text {L L M}} \left(q, \mathcal {A} _ {q}, \hat {M} _ {u} (q)\right), \tag {9}
$$

where $\mathcal { T } _ { u } ( q )$ represents the ideal item profile that captures the optimal item characteristics for the current query.

Through this reasoning-enhanced retrieval mechanism, the assistant dynamically leverages both global and local memory to produce contextually aware and highly personalized recommendations, treating memory as an active resource that evolves with the reasoning process rather than static input.

3.2.3 Enhancing Reasoning with Memory Retrieval. Except that our memory retrieval method gains enhancement from LLM

reasoning, we highlight that this method also represents an important enhancement over existing reasoning methods for recommendation tasks. Effective reasoning is crucial for understanding complex user preferences and making contextually appropriate connections between user needs and available options. Current recommendation LLMs typically employ chain-of-thought reasoning or prompt-based inference on static input, but these approaches suffer from limited reasoning depth as they cannot dynamically explore relevant information during the inference process, leading to suboptimal personalization and reduced context awareness. Our method enhances LLM reasoning by extending it to encompass memory retrieval stages, creating an adaptive framework where reasoning and information gathering are interleaved.

Specifically, the LLM iteratively analyzes the recommendation scenario, identifies relevant preference dimensions, and retrieves corresponding memory entries to refine its understanding. This enables the LLM to treat memory as a dynamic resource aligned with the evolving reasoning process, significantly enhancing reasoning capabilities through multi-step, evidence-grounded inference that adapts to task complexity. Importantly, this extended reasoning mechanism can be jointly optimized through our reinforcement learning framework, allowing the model to learn optimal strategies for both memory exploration and reasoning refinement.

# 3.3 Reinforcement Learning for Memory-synergized Reasoning

To enhance our memory-synergized reasoning, we fine-tune the base LLM through reinforcement learning. Our MR.Rec requires robust reasoning and effective memory exploration through multiturn interactions and iterative retrieval. However, comprehensive annotation for such complex processes is prohibitively resourceintensive, making supervised fine-tuning impractical.

To overcome this, we adopt a multi-turn reinforcement learning framework. This approach allows the LLM to actively explore the recommendation and memory environment, refining its policy through iterative interactions. By leveraging feedback signals instead of static annotations, the model develops adaptive behaviors that optimize long-term recommendation accuracy and efficient memory utilization. Specifically, inspired by Group Relative Policy Optimization (GRPO) [6], we adopt a reinforcement learning with verifiable rewards framework that incorporates multi-turn feedback signals into policy optimization. For a given query $q$ , the assistant is prompted multiple times to analyze the query, retrieve potentially useful memory entries, and reason through these before generating ?? final candidate responses $\left\{ o _ { 1 } , o _ { 2 } , \ldots , o _ { G } \right\}$ . A reward model then assigns reward scores $\{ r _ { 1 } , r _ { 2 } , \dots , r _ { G } \}$ to each candidate response. Based on these scores, relative advantages $A ( o _ { i } )$ are computed, which indicate the comparative quality of different outputs and guide subsequent policy optimization.

To guide the model toward the desired behavior, we design a set of multi-faceted reward functions:

• Format Reward $\left( R _ { \mathrm { f o r m a t } } \right)$ . To extract the desired answers from the reasoning process, we require the LLM to output its final answer in a prescribed format. The reward is assigned as 1 if the format is correct, and 0 otherwise.   
• Recommendation Reward $( R _ { \mathrm { r e c } } )$ . To quantify recommendation quality, we define a reward $R _ { \mathrm { r e c } } = \mathrm { n D C G @ 1 0 0 0 + n D C G } \ @ 1 0 0$

where nDCG@1000 mitigates reward sparsity, and nDCG@100 emphasizes fine-grained ranking among top candidates.

• Memory Utilization Reward $\left( R _ { \mathrm { m e m } } \right)$ . To encourage reasoning grounded in memory, the LLM is required to perform a memory retrieval step. The reward is defined as a binary indicator, where the reward is 1 if the model successfully calls memory retrieval during generation, and 0 otherwise.

The overall reward model is a weighted combination of the three reward components, with weights ??1, ??2, ??3:

$$
r = w _ {1} R _ {\text {f o r m a t}} + w _ {2} R _ {\text {r e c}} + w _ {3} R _ {\text {m e m}}. \tag {10}
$$

Finally, the optimization objective is formulated as a clipped policy gradient similar to PPO:

$$
\begin{array}{l} \mathcal {J} (\theta) = \mathbb {E} _ (q, g t) \sim \mathcal {D}, \{o _ {i} \} _ {i = 1} ^ {G} \sim^ {\pi_ {\theta_ {\mathrm {o l d}}} (\cdot | q)} = \\ \left. \right.\left[ \frac {1}{N} \sum_ {i = 1} ^ {G} \sum_ {t = 1} ^ {| o _ {i} |} \min  \left(\frac {\pi_ {\theta} (o | q)}{\pi_ {\theta_ {\text {o l d}}} (o | q)} A _ {i, t}, \operatorname {c l i p} \left(\frac {\pi_ {\theta} (o | q)}{\pi_ {\theta_ {\text {o l d}}} (o | q)}, 1 - \epsilon , 1 + \epsilon\right) A _ {i, t}\right)\right], \tag {11} \\ \end{array}
$$

where $A _ { i , t }$ denotes the advantage of each token derived from relative reward signals, which equals $\begin{array} { r } { A ( o _ { i } ) = \frac { r _ { i } - \mathrm { m e a n ( \mathbf { r } ) } } { \mathrm { s t d ( \mathbf { r } ) } } } \end{array}$ . Importantly, during optimization, we mask the retrieved memory tokens to ensure that the advantage estimation depends only on the assistant’s reasoning and recommendation outputs.

# 4 Experiments

In this section, we evaluate the proposed MR.Rec framework to answer the following research questions (RQs):

• RQ1: How does MR.Rec perform in providing personalized recommendation compared to baseline methods?   
• RQ2: What is the contribution of each main component and different types of memory to the overall performance of MR.Rec?   
• RQ3: Does our RAG-based memory mechanism retrieve genuinely beneficial information to address recommendation queries?   
• RQ4: How is the efficiency of indexing and retrieval of our MR.Rec?   
• RQ5: How do different model settings impact the performance of MR.Rec, including hyperparameter settings, LLM backbones, and different retriever configurations?

# 4.1 Experimental Setup

4.1.1 Datasets. We construct our dataset based on the Amazon-C4 [9] dataset, which provides user queries generated by ChatGPT from product reviews. These review-based queries are often highly detailed, implicitly covering nearly all aspects of a user’s preferences. However, such exhaustive queries are not representative of typical user behavior in real-world recommendation scenarios. To better evaluate our method and compare it with different baselines in leveraging user memory for personalized recommendations, we simplify these queries using GPT-o3-mini to remove some of the detailed preference information. More dataset statistics and the prompt used for query simplification can be found in Appendix A.

4.1.2 Baselines. We evaluate our method against a set of representative models, which are divided into two groups. The first group consists of general-purpose LLM backbones: GPT-4o, DeepSeek-R1, and Qwen-2.5-3B-Instruct. The second group comprises models specifically fine-tuned for recommendation tasks: BLAIR [9], a sentence embedding model pretrained on user review and item metadata pairs using a contrastive objective, and Rec-R1 [14], which

directly optimizes LLM generation via feedback from a fixed, blackbox recommendation model. To investigate both the impact of memory and the models’ ability to leverage user memory for recommendation, we consider three memory settings for these models:

• w/o Memory (Query-only): Models receive only the current user query, without access to any historical interactions.   
• w/ Naive Memory (Query $^ +$ user interaction history): Except queries, models are provided with the latest interaction history of the user, representing straightforward memory integration.   
• w/ Static Memory (Query $^ +$ pre-generated user summary): Models are provided with a pre-constructed summary of the user’s preferences, representing static summarized memory.

4.1.3 Evaluation settings. For each model under the corresponding memory setting, given a user query, we require the model to generate either the ideal item profile or its embedding, which is then used to retrieve items from the candidate pool. Recommendation performance is evaluated using standard metrics, including nDCG@100, nDCG@1000, Recall $@ 1 0 0$ and Recall $@ 1 0 0 0$ .

4.1.4 Implementation details. We select Qwen-2.5-3B-Instruct as the backbone LLM in our method. During training, we use a learning rate of 1e-6, a group size of 5, and set the maximum response length to 768. Training is conducted for up to 5 epochs with early stopping (patience $= 1$ ). In our method, the hyperparameters for weighted reward are set as $w _ { 1 } = 0 . 1$ , $w _ { 2 } = 5$ , $w _ { 3 } = 0 . 1$ . More implementation details can be found in Appendix B.

# 4.2 Overall Performance (RQ1)

We compare MR.Rec’s recommendation accuracy with baselines, with results shown in Table 1. We make the following discussions:

• Performance Superiority of MR.Rec. MR.Rec shows consistent advantages across different baseline categories. Compared to baselines without memories, our RAG-based memory mechanism improves performance by retrieving collaborative patterns. This advantage persists against baselines with naive memory and static summaries, demonstrating the effectiveness of synergizing memory with reasoning to enable dynamic preference exploration and contextual personalization. Compared to recommendation reasoning methods (i.e. Rec-R1 and BLAIR), MR.Rec extends reasoning to include iterative memory retrieval, enabling adaptive information gathering that evolves with the reasoning process rather than relying on static input, significantly enhancing reasoning depth and context awareness.   
• Incorporating Interaction Records for Baselines. For all baselines, additionally incorporating naive memory (i.e. using raw interaction records) generally brings performance gains, validating the information gain of involving users’ past behaviors in better understanding user queries. However, this improvement is marginal for baselines fine-tuned for recommendation (i.e. Rec-R1). This suggests that while LLM fine-tuning has unlocked models’ recommendation potential, noise in raw interactions and limited reasoning depth prevent further improvements.   
• Static Summarization is Sometimes Harmful. To handle noisy interaction records, fixed memory generates user summaries as external memories for the baseline methods. However, results show less improvement than naive memory and even cause performance degradation. This demonstrates the difficulty

Table 1: The overall performance of baselines and MR.Rec. The bold/underline values represent the best/the second-best result, respectively. The test dataset comprises 28 categories, and we report the averages for all while highlighting the top three.   

<table><tr><td rowspan="2">Memory Setting</td><td rowspan="2">Model</td><td colspan="4">All</td><td colspan="4">Home</td><td colspan="4">Clothing</td><td colspan="4">Tools</td></tr><tr><td>R@100</td><td>R@10</td><td>N@100</td><td>N@10</td><td>R@100</td><td>R@10</td><td>N@100</td><td>N@10</td><td>R@100</td><td>R@10</td><td>N@100</td><td>N@10</td><td>R@100</td><td>R@10</td><td>N@100</td><td>N@10</td></tr><tr><td rowspan="6">w/o Memory</td><td>GPT-4o</td><td>0.226</td><td>0.092</td><td>0.086</td><td>0.059</td><td>0.227</td><td>0.057</td><td>0.090</td><td>0.033</td><td>0.100</td><td>0.030</td><td>0.036</td><td>0.022</td><td>0.247</td><td>0.097</td><td>0.096</td><td>0.065</td></tr><tr><td>DeepSeek-R1</td><td>0.252</td><td>0.099</td><td>0.090</td><td>0.060</td><td>0.256</td><td>0.085</td><td>0.082</td><td>0.049</td><td>0.130</td><td>0.035</td><td>0.040</td><td>0.022</td><td>0.290</td><td>0.118</td><td>0.093</td><td>0.059</td></tr><tr><td>Qwen2.5-3B</td><td>0.255</td><td>0.096</td><td>0.097</td><td>0.065</td><td>0.261</td><td>0.081</td><td>0.081</td><td>0.043</td><td>0.120</td><td>0.035</td><td>0.036</td><td>0.020</td><td>0.290</td><td>0.108</td><td>0.114</td><td>0.078</td></tr><tr><td>BLAIR-BASE</td><td>0.227</td><td>0.072</td><td>0.070</td><td>0.040</td><td>0.213</td><td>0.062</td><td>0.056</td><td>0.025</td><td>0.135</td><td>0.035</td><td>0.037</td><td>0.018</td><td>0.280</td><td>0.086</td><td>0.080</td><td>0.042</td></tr><tr><td>BLAIR-LARGE</td><td>0.215</td><td>0.065</td><td>0.069</td><td>0.040</td><td>0.232</td><td>0.052</td><td>0.065</td><td>0.030</td><td>0.090</td><td>0.025</td><td>0.025</td><td>0.012</td><td>0.312</td><td>0.097</td><td>0.104</td><td>0.061</td></tr><tr><td>Rec-R1</td><td>0.258</td><td>0.111</td><td>0.099</td><td>0.071</td><td>0.265</td><td>0.085</td><td>0.086</td><td>0.047</td><td>0.126</td><td>0.037</td><td>0.04</td><td>0.022</td><td>0.297</td><td>0.114</td><td>0.117</td><td>0.08</td></tr><tr><td rowspan="6">w/ Naive Memory</td><td>GPT-4o</td><td>0.258</td><td>0.109</td><td>0.104</td><td>0.072</td><td>0.278</td><td>0.081</td><td>0.091</td><td>0.047</td><td>0.125</td><td>0.035</td><td>0.041</td><td>0.025</td><td>0.301</td><td>0.119</td><td>0.110</td><td>0.079</td></tr><tr><td>DeepSeek-R1</td><td>0.260</td><td>0.106</td><td>0.100</td><td>0.067</td><td>0.275</td><td>0.085</td><td>0.090</td><td>0.051</td><td>0.127</td><td>0.033</td><td>0.043</td><td>0.026</td><td>0.301</td><td>0.118</td><td>0.109</td><td>0.074</td></tr><tr><td>Qwen2.5-3B</td><td>0.246</td><td>0.107</td><td>0.095</td><td>0.068</td><td>0.280</td><td>0.088</td><td>0.084</td><td>0.049</td><td>0.105</td><td>0.035</td><td>0.037</td><td>0.026</td><td>0.280</td><td>0.108</td><td>0.107</td><td>0.073</td></tr><tr><td>BLAIR-BASE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BLAIR-LARGE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Rec-R1</td><td>0.260</td><td>0.108</td><td>0.097</td><td>0.075</td><td>0.269</td><td>0.086</td><td>0.085</td><td>0.050</td><td>0.128</td><td>0.036</td><td>0.040</td><td>0.027</td><td>0.299</td><td>0.112</td><td>0.119</td><td>0.085</td></tr><tr><td rowspan="6">w/ Static Memory</td><td>GPT-4o</td><td>0.252</td><td>0.098</td><td>0.095</td><td>0.065</td><td>0.237</td><td>0.076</td><td>0.071</td><td>0.041</td><td>0.105</td><td>0.030</td><td>0.037</td><td>0.022</td><td>0.311</td><td>0.107</td><td>0.110</td><td>0.069</td></tr><tr><td>DeepSeek-R1</td><td>0.249</td><td>0.089</td><td>0.087</td><td>0.057</td><td>0.232</td><td>0.076</td><td>0.072</td><td>0.044</td><td>0.125</td><td>0.025</td><td>0.038</td><td>0.020</td><td>0.301</td><td>0.086</td><td>0.092</td><td>0.050</td></tr><tr><td>Qwen2.5-3B</td><td>0.246</td><td>0.106</td><td>0.098</td><td>0.069</td><td>0.265</td><td>0.081</td><td>0.081</td><td>0.044</td><td>0.110</td><td>0.035</td><td>0.034</td><td>0.020</td><td>0.280</td><td>0.115</td><td>0.116</td><td>0.086</td></tr><tr><td>BLAIR-BASE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>BLAIR-LARGE</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Rec-R1</td><td>0.259</td><td>0.105</td><td>0.095</td><td>0.069</td><td>0.264</td><td>0.082</td><td>0.083</td><td>0.046</td><td>0.122</td><td>0.033</td><td>0.038</td><td>0.024</td><td>0.286</td><td>0.110</td><td>0.115</td><td>0.071</td></tr><tr><td colspan="2">Ours</td><td>0.270</td><td>0.122</td><td>0.113</td><td>0.084</td><td>0.284</td><td>0.090</td><td>0.092</td><td>0.054</td><td>0.130</td><td>0.040</td><td>0.045</td><td>0.027</td><td>0.333</td><td>0.129</td><td>0.132</td><td>0.091</td></tr><tr><td colspan="2">Improvement</td><td>+3.84%</td><td>+9.91%</td><td>+8.65%</td><td>+12.00%</td><td>+2.16%</td><td>+2.27%</td><td>+1.10%</td><td>+5.88%</td><td>-</td><td>+8.18%</td><td>+4.65%</td><td>+0.00%</td><td>+7.07%</td><td>+8.40%</td><td>+10.92%</td><td>+5.81%</td></tr></table>

![](images/35caabfb6dc53035e3bbbc59bc2bf0b136698481de8c6532e9be03678f56f36d.jpg)  
Figure 3: Ablation of different components of MR.Rec.

of building a global user summarization template, as the optimal solution may vary greatly for different users and item categories. Our dynamic memory retrieval mechanism with reasoning enhancement effectively addresses this concern and consistently delivers substantial performance gains across all scenarios.

# 4.3 Ablation Study (RQ2)

4.3.1 Effect of Key Components. We first study the impact of different technical components of MR.Rec. The evaluation results are presented in Figure 3. We study the following components:

• w/o Local/Global Memory. The results demonstrate that removing either local or global memory mechanisms causes significant performance degradation, with varying impact across different datasets. This validates the effectiveness of our memory indexing method in capturing beneficial interaction patterns and our retrieval method in effectively accessing this information.   
• w/o RL Tuning. Compared to removing memory mechanisms, directly using the base LLM without RL tuning causes larger performance damage. This is because RL tuning not only improves the base model’s performance, but also significantly impacts the retrieval loop effectiveness. Without our tuning, small-parameter LLMs such as the 3B model struggle to determine when and how to leverage memory and perform reasoning. This highlights the critical role of RL tuning for MR.Rec.

Table 2: Ablation of user-specific local memory components.   

<table><tr><td>Variant</td><td>R@100</td><td>R@10</td><td>N@100</td><td>N@10</td></tr><tr><td>w/o Local Memory</td><td>0.258</td><td>0.113</td><td>0.098</td><td>0.069</td></tr><tr><td>w/ Behavior Records</td><td>0.268↑</td><td>0.117↑</td><td>0.109↑</td><td>0.079↑</td></tr><tr><td>w/ Preference Patterns</td><td>0.272↑</td><td>0.116↑</td><td>0.110↑</td><td>0.079↑</td></tr><tr><td>w/ User Profile</td><td>0.269↑</td><td>0.113</td><td>0.113↑</td><td>0.080↑</td></tr><tr><td>w/ B+P+U</td><td>0.270↑</td><td>0.122↑</td><td>0.113↑</td><td>0.084↑</td></tr></table>

4.3.2 Effect of Local Memory Components. As shown in Table 2, we further investigate the impact of different components of user-specific local memory on model performance. The results reveal that the absence of the entire local memory mechanism leads to the weakest performance, whereas incorporating any single component (Behavior Records, Preference Patterns, or User Profile) individually yields clear improvements. The best results are achieved when all three components are combined, suggesting a complementary effect that more comprehensively captures user preferences relevant to the current query.

# 4.4 Memory Effectiveness Study (RQ3)

To further investigate the effectiveness of memory in the recommendation process, we analyzed the retrieved memory and designed two metrics: Memory-to-Profile Contribution (MPC) and Memory-to-Recommendation Contribution (MRC). MPC measures whether the retrieved memory aids in generating the ideal item profile, while MRC evaluates whether it contributes to the final correct recommendation.

To ensure a comprehensive and objective evaluation, we employed two assessment approaches: a heuristic method, checking keyword overlaps between retrieved memory and profiles or ground-truth items, and an LLM-judged method, where an LLM determines whether the memory is helpful.

As shown in Table 3, both Global and Local Memory are valuable in the recommendation process. For MPC, they both achieve

Table 3: Contribution of retrieved memory.   

<table><tr><td rowspan="2">Memory Type</td><td colspan="2">MPC</td><td colspan="2">MRC</td></tr><tr><td>Heuristic</td><td>LLM-Judged</td><td>Heuristic</td><td>LLM-Judged</td></tr><tr><td>Global Memory</td><td>0.9605</td><td>0.9172</td><td>0.9012</td><td>0.5375</td></tr><tr><td>Local Memory</td><td>0.9368</td><td>0.9511</td><td>0.4743</td><td>0.7036</td></tr></table>

high scores, indicating that retrieved memory significantly aids the generation of ideal item profiles. This demonstrates that MR.Rec effectively leverages memory to provide final recommendation. For MRC, Global Memory scores relatively high under the heuristic method but somewhat lower under LLM judgment, reflecting that it often overlaps with ground-truth items at the surface level, while semantic contribution is more nuanced. By contrast, Local Memory shows modest heuristic scores but substantially higher LLM-judged scores. This indicates that, despite limited surface-level overlap, local memory provides more semantically meaningful support for recommendations. This effectiveness arises from our approach: instead of one-shot retrieval based solely on query similarity, the LLM is trained to reason about which memory entries are genuinely useful, enabling more precise and context-aware recommendations.

# 4.5 Efficiency Study (RQ4)

This experiment studies the costs and efficiency of our memory indexing and retrieval mechanisms. We built user-specific local memory for 3,000 users and sampled 157 users for cross-user global memory, creating 73,078 local and 1,970 global memory entries. The statistics and costs are shown in Table 4.

• Time and Cost Efficiency of Indexing. In our proposed MR.Rec, indexing time per entry is 0.07s for local and 0.06s for global memory, with API costs of $\$ 54$ for 3,000 users ($0.018/user) and $\$ 0.07$ for global memory. Global memory uses only 0.48M input and 0.06M output tokens, roughly 1/1000 of local memory, showing high indexing efficiency.   
• Token Efficiency of Retrieval. Table 4 shows our method achieves highest $\mathbb { R } \ @ \mathbb { 1 } 0 0$ with 95.43 tokens, yielding an efficiency of 0.299 (R@100/100 tokens), outperforming baselines that utilize recent interactions or static profiles as memory. By retrieving the most relevant memory entries for recommendation, our method introduces less noise into the context, leading to higher recall and more efficient reasoning.

Table 4: Cost and Efficiency for Memory Indexing and Retrieval in the MR.Rec Framework   

<table><tr><td colspan="4">Statistics and Cost of Memory Indexing</td></tr><tr><td></td><td>Local Memory</td><td colspan="2">Global Memory</td></tr><tr><td>Source Users</td><td>3,000</td><td colspan="2">157</td></tr><tr><td>Input Tokens (M)</td><td>385.21</td><td colspan="2">0.48</td></tr><tr><td>Output Tokens (M)</td><td>42.00</td><td colspan="2">0.06</td></tr><tr><td>Memory Entries</td><td>73,078</td><td colspan="2">1,970</td></tr><tr><td>Total Time (s)</td><td>2,435</td><td colspan="2">121</td></tr><tr><td>Per-entry Time (s)</td><td>0.07</td><td colspan="2">0.06</td></tr><tr><td>API Cost (GPT-4o-mini)</td><td>54.09</td><td colspan="2">0.07</td></tr><tr><td>Per-entry Cost</td><td>\( 7.4 \times 10^{-4} \)</td><td colspan="2">\( 3.6 \times 10^{-5} \)</td></tr><tr><td colspan="4">Token Efficiency of Memory Retrieval</td></tr><tr><td>Memory Setting</td><td>Avg. MemoryTokens</td><td>R@100(GPT-4o)</td><td>Efficiency(R@100/100 tokens)</td></tr><tr><td>Recent 10 interactions</td><td>283.51</td><td>0.261</td><td>0.092</td></tr><tr><td>Static user profile</td><td>492.7</td><td>0.261</td><td>0.053</td></tr><tr><td>MR.Rec</td><td>95.43</td><td>0.285</td><td>0.299</td></tr></table>

![](images/d4b7d3da16ce2fa3b27f4535f5fa14db69156f66f67b572c457ba57793516d46.jpg)  
Figure 4: Reward trajectories of MR.Rec using the base and instruct models of Qwen-2.5-3B, during our RL tuning.

![](images/8c71aa9481a0410858c076edef4b109fddae53eb17dd966e0d22d005a72dcfbf.jpg)  
Figure 5: Effect of using different retrievers for MR.Rec.

# 4.6 Sensitivity Study (RQ5)

This section studies the impact of different configurations of our MR.Rec, including LLM backbones and different retrieval settings.

4.6.1 Impact of LLM Backbones. To study the impact of different LLM backbones, we compare the default backbone of MR.Rec, Qwen-2.5-3B-Instruct, with its base model Qwen-2.5-3B-Base. The former has undergone prior instruction tuning and reinforcement learning from human feedback (RLHF) to strengthen its instructionfollowing capabilities, while the latter has not. Both MR.Rec variants using these backbone LLMs are tuned using our RL method, and their reward trajectories are recorded and presented in Figure 4.

The results demonstrate strong instruction-following ability of the Instruct model from the start, correctly invoking memory retrieval (Memory reward) and generating properly formatted outputs (Format reward). In contrast, the Base model initially struggles to follow instructions, with format reward remaining near zero. During training, the Base model gradually learns to produce properly formatted outputs and shows improvements in recommendation performance, as reflected by Recall@100. However, it consistently fails to learn memory retrieval, as indicated by near-zero memory reward, instead relying solely on direct recommendations.

4.6.2 Impact of Base Retrievers. We compared using three different base retrieval algorithms for our reasoning-enhanced retrieval process, including Qwen3-Embedding-0.6B [37], BGE-M3 [2], and BM25 [21]. Among these, Qwen3-Embedding-0.6B supports customized instructions during retrieval, BGE-M3 is a pre-trained deep learning model, while BM25 is a traditional lexical matching method that requires no pre-training.

The experimental results shown in Figure 5 demonstrate that Qwen3-Embedding-0.6B achieves the best performance, while BM25, which lacks specialized training for dense retrieval, shows a significant performance gap compared to the embedding-based models. The performance gaps between models become more pronounced as the value of k increases in NDCG and Recall@k metrics, indicating that the advantages of instruction-tuned embedding models are more evident when retrieving larger sets of relevant items.

4.6.3 Impact of Retrieved Memory Size. We further analyze the impact of retrieving different numbers of top-k relevant memory

![](images/515eacc32c2f4d9cc1ed02db8a5da70c547dbe302d187e3351907eb29b0a67e9.jpg)  
Figure 7: Case study demonstrating MR.Rec’s memorysynergized reasoning for personalized cap recommendation.

Figure 6: Effect of different top-k of memory retrieval.

# Case Study

User Query: Find me a cap suitable for post-winter wear.

GPT-4o: Cap for post-winter wear, suitable for mild to cool weather, lightweight yet warm, stylish and comfortable, ideal for transitional seasons after winter.

MR.Rec: [Reasoning] : For this query, I should consider whether user has preferences on "Design and Style Balance", "Material Quality and Feel" and "Fit and Comfort".

[Retrieved Memory] : - For "Material Quality and Feel": the user prefers premium, natural materials like real leather, rubber, and carbon fiber over synthetic options, valuing durability and a high-quality tactile experience.

- For "Fit and Comfort": The user likes true-to-size, well-fitting clothing that without feeling tight or restrictive, with good support and enough room.

[Reasoning] $:$ Given the user’s query and retrieved memory, the ideal profile includes:

- Material Quality: Durable, high-quality materials like real leather or carbon fiber.   
- Fit and Comfort: True-to-size and adjustable for a comfortable fit.   
- Seasonal Suitability: Suitable for post-winter, lightweight.

Ground Truth Item: DALLY Up Leather Patch Western Lifestyle Adjustable 6-Panel Snapback Hat. The Dally Cap line is for those who want to look just as Western on the outside as they are on the inside.

entries on model performance, with results shown in Figure 6. The results reveal a trade-off pattern: smaller k values may miss useful memory information, while larger k values may introduce noise and extend the context window, potentially degrading LLM performance. Based on these findings, we select $\mathrm { k } { = } 3$ as the optimal balance, which consistently achieves strong performance.

# 4.7 Case Study

We present a case study in Figure 7, illustrating how MR.Rec integrates memory and reasoning for recommendation. When a user queries for "a cap suitable for post-winter wear," the baseline GPT-4o generates a generic response focused on seasonal suitability. In contrast, MR.Rec first reasons through implicit preference dimensions like "Material Quality and Feel" and "Fit and Comfort." It then retrieves relevant memory segments reflecting the user’s preference for premium materials like real leather and adjustable items. The model generates an ideal item profile combining these preferences with seasonal needs. The final recommendation strongly aligns with the ground truth item, a leather patch adjustable snapback hat, with key attributes (leather, adjustable) appearing in both the generated profile and actual item, demonstrating how our reasoning-enhanced memory retrieval successfully captures user-specific preferences that would be missed by generic baselines.

# 4.8 Related Work

4.8.1 Memory Modeling in Recommender Systems. Memory modeling is essential for capturing dynamic user preferences in

RSs. Traditional sequential models, such as SASRec [12], implicitly encode user history but often struggle with long sequences and lack explicit interpretability. The emergence of LLMs has introduced explicit memory mechanisms, enabling more sophisticated and interpretable representations of user behavior. A prominent approach involves utilizing user interaction history as memory [10, 26, 39], with methods like AutoMR [23] employing learned retrievers to intelligently select relevant historical segments. In contrast, MARM [16] and LMN [15] enhance efficiency by caching computations and utilizing product quantization to handle long sequences and large memory banks. An alternative approach leverages LLM-generated user profiles, which synthesize extensive interaction histories into textual summaries [32]. MAP [3] constructs user profiles as tables of historical ratings and reviews, but this flat memory structure struggles with noisy and multi-granularity preferences. MemoCRS [31] introduces an entity-based user memory and a shared general memory for multi-turn dialogues. Nevertheless, it fails to effectively integrate memory with reasoning, resulting in a superficial reliance on similarity-based retrieval, which neglects deeper semantic relationships within the memory and limits the generation of contextually relevant recommendations. In comparison, MR.Rec develops hierarchical memory indexing and reasoning-enhanced retrieval, better filtering noisy information while achieving efficiency.

4.8.2 LLM Reasoning for Recommendation. LLMs have transformed RSs by introducing advanced reasoning beyond traditional collaborative filtering. Key developments include Chain-of-Thought (CoT) [28] reasoning, which allows LLMs to generate step-by-step logical recommendations. For instance, HetGCoT-Rec [11] integrates heterogeneous graphs with CoT for journal recommendations, and Li et al. [13] use CoT to enhance user intention inference and contextual awareness. However, prompting-based CoT methods remain limited in complex recommendation tasks. To address these limitations, recent studies have shifted toward training LLMs explicitly for reasoning rather than relying solely on prompting. A representative approach for enhancing LLM reasoning is GRPO [6]. Reason-to-Recommend [40] employs GRPO to strengthen "Interaction-of-Thought" reasoning, enabling emergent planning behaviors in challenging recommendation tasks. Similarly, LatentR [36] adopts a modified GRPO framework to optimize latent reasoning trajectories, avoiding explicit text generation while preserving reasoning capacity. Our work extends LLM finetuning beyond CoT reasoning to include memory-synergized reasoning.

# 5 Conclusion

In this paper, we introduced MR.Rec, a framework that unifies memory and reasoning to advance LLM-based recommendation assistants. By combining a RAG-based external memory mechanism with reinforcement learning, our approach enables the LLM to actively explore and select useful preferences and experiences while performing multi-step reasoning over retrieved information to generate accurate recommendations. Extensive experiments on multiple benchmark datasets demonstrate that MR.Rec consistently outperforms state-of-the-art baselines in both personalization and reasoning capability. We believe MR.Rec takes a meaningful step toward intelligent and user-centered recommendation assistants.

# References

[1] Millennium Bismay, Xiangjue Dong, and James Caverlee. Reasoningrec: Bridging personalized recommendations and human-interpretable explanations through llm reasoning. arXiv preprint arXiv:2410.23180, 2024.   
[2] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 2024.   
[3] Jiarui Chen. Memory assisted llm for personalized recommendation system. arXiv preprint arXiv:2505.03824, 2025.   
[4] Jiabao Fang, Shen Gao, Pengjie Ren, Xiuying Chen, Suzan Verberne, and Zhaochun Ren. A multi-agent conversational recommender system. arXiv preprint arXiv:2402.01135, 2024.   
[5] Mohamed Amine Ferrag, Norbert Tihanyi, and Merouane Debbah. From llm reasoning to autonomous ai agents: A comprehensive review. arXiv preprint arXiv:2504.19678, 2025.   
[6] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
[7] Donghee Han, Hwanjun Song, and Mun Yong Yi. Rethinking llm-based recommendations: A query generation-based, training-free approach. arXiv preprint arXiv:2504.11889, 2025.   
[8] Zhankui He, Zhouhang Xie, Harald Steck, Dawen Liang, Rahul Jha, Nathan Kallus, and Julian McAuley. Reindex-then-adapt: Improving large language models for conversational recommendation. In Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining, pages 866–875, 2025.   
[9] Yupeng Hou, Jiacheng Li, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. Bridging language and items for retrieval and recommendation. arXiv preprint arXiv:2403.03952, 2024.   
[10] Jiani Huang, Shijie Wang, Liang-bo Ning, Wenqi Fan, Shuaiqiang Wang, Dawei Yin, and Qing Li. Towards next-generation recommender systems: A benchmark for personalized recommendation assistant with llms. arXiv preprint arXiv:2503.09382, 2025.   
[11] Runsong Jia, Mengjia Wu, Ying Ding, Jie Lu, and Yi Zhang. Hetgcot-rec: Heterogeneous graph-enhanced chain-of-thought llm reasoning for journal recommendation. arXiv preprint arXiv:2501.01203, 2025.   
[12] Wang-Cheng Kang and Julian McAuley. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM), pages 197–206. IEEE, 2018.   
[13] Gangmin Li, Fan Yang, and Yong Yue. Identify user intention for recommendation using chain-of-thought prompting in llm. In International Conference for Emerging Technologies in Computing, pages 60–69. Springer, 2024.   
[14] Jiacheng Lin, Tian Wang, and Kun Qian. Rec-r1: Bridging generative large language models and user-centric recommendation systems via reinforcement learning. arXiv preprint arXiv:2503.24289, 2025.   
[15] Hui Lu, Zheng Chai, Yuchao Zheng, Zhe Chen, Deping Xie, Peng Xu, Xun Zhou, and Di Wu. Large memory network for recommendation. In Companion Proceedings of the ACM on Web Conference 2025, pages 1162–1166, 2025.   
[16] Xiao Lv, Jiangxia Cao, Shijie Guan, Xiaoyou Zhou, Zhiguang Qi, Yaqiang Zang, Ming Li, Ben Wang, Kun Gai, and Guorui Zhou. Marm: Unlocking the future of recommendation systems through memory augmentation and scalable complexity. arXiv preprint arXiv:2411.09425, 2024.   
[17] Hanjia Lyu, Song Jiang, Hanqing Zeng, Yinglong Xia, Qifan Wang, Si Zhang, Ren Chen, Christopher Leung, Jiajie Tang, and Jiebo Luo. Llm-rec: Personalized recommendation via prompting large language models. arXiv preprint arXiv:2307.15780, 2023.   
[18] Lucie Charlotte Magister, Katherine Metcalf, Yizhe Zhang, and Maartje ter Hoeve. On the way to llm personalization: Learning to remember user conversations. arXiv preprint arXiv:2411.13405, 2024.   
[19] Andriy Mnih and Russ R Salakhutdinov. Probabilistic matrix factorization. Advances in neural information processing systems, 20, 2007.   
[20] Bruno Pradel, Savaneary Sean, Julien Delporte, Sébastien Guérif, Céline Rouveirol, Nicolas Usunier, Françoise Fogelman-Soulié, and Frédéric Dufau-Joel. A case study in a recommender system based on purchase data. In Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 377–385, 2011.   
[21] Stephen Robertson, Hugo Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389, 2009.   
[22] Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Jian Wu, and Yuning Jiang. Think before recommend: Unleashing the latent reasoning power

for sequential recommendation. arXiv preprint arXiv:2503.22675, 2025.   
[23] Chengbing Wang, Yang Zhang, Fengbin Zhu, Jizhi Zhang, Tianhao Shi, and Fuli Feng. Leveraging memory retrieval to enhance llm-based generative recommendation. In Companion Proceedings of the ACM on Web Conference 2025, pages 1346–1350, 2025.   
[24] Lu Wang, Di Zhang, Fangkai Yang, Pu Zhao, Jianfeng Liu, Yuefeng Zhan, Hao Sun, Qingwei Lin, Weiwei Deng, Dongmei Zhang, et al. Lettingo: Explore user profile generation for recommendation system. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 2985–2995, 2025.   
[25] Mingze Wang, Chongming Gao, Wenjie Wang, Yangyang Li, and Fuli Feng. Tunable llm-based proactive recommendation agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19262–19276, 2025.   
[26] Shijie Wang, Wenqi Fan, Yue Feng, Shanru Lin, Xinyu Ma, Shuaiqiang Wang, and Dawei Yin. Knowledge graph retrieval-augmented generation for llm-based recommendation. arXiv preprint arXiv:2501.02226, 2025.   
[27] Yan Wang, Zhixuan Chu, Xin Ouyang, Simeng Wang, Hongyan Hao, Yue Shen, Jinjie Gu, Siqiao Xue, James Zhang, Qing Cui, et al. Llmrg: Improving recommendations through large language model reasoning graphs. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 19189–19196, 2024.   
[28] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
[29] Shiwen Wu, Fei Sun, Wentao Zhang, Xu Xie, and Bin Cui. Graph neural networks in recommender systems: a survey. ACM Computing Surveys, 55(5):1–37, 2022.   
[30] Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, and Yong Liu. From human memory to ai memory: A survey on memory mechanisms in the era of llms. arXiv preprint arXiv:2504.15965, 2025.   
[31] Yunjia Xi, Weiwen Liu, Jianghao Lin, Bo Chen, Ruiming Tang, Weinan Zhang, and Yong Yu. Memocrs: Memory-enhanced sequential conversational recommender systems with large language models. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pages 2585–2595, 2024.   
[32] Jiarui Zhang. Guided profile generation improves personalization with llms. arXiv preprint arXiv:2409.13093, 2024.   
[33] Qinyao Zhang, Bin Guo, Yao Jing, Yan Liu, and Zhiwen Yu. Mindmemory: Augmented llm with long-term memory and mental personality. In CCF Conference on Computer Supported Cooperative Work and Social Computing, pages 462–476. Springer, 2024.   
[34] Shuai Zhang, Lina Yao, Aixin Sun, and Yi Tay. Deep learning based recommender system: A survey and new perspectives. ACM computing surveys (CSUR), 52(1):1– 38, 2019.   
[35] Yadong Zhang, Shaoguang Mao, Tao Ge, Xun Wang, Adrian de Wynter, Yan Xia, Wenshan Wu, Ting Song, Man Lan, and Furu Wei. Llm as a mastermind: A survey of strategic reasoning with large language models. arXiv preprint arXiv:2404.01230, 2024.   
[36] Yang Zhang, Wenxin Xu, Xiaoyan Zhao, Wenjie Wang, Fuli Feng, Xiangnan He, and Tat-Seng Chua. Reinforced latent reasoning for llm-based recommendation. arXiv preprint arXiv:2505.19092, 2025.   
[37] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.   
[38] Yu Zhang, Shutong Qiao, Jiaqi Zhang, Tzu-Heng Lin, Chen Gao, and Yong Li. A survey of large language model empowered agents for recommendation and search: Towards next-generation information retrieval. arXiv preprint arXiv:2503.05659, 2025.   
[39] Zijian Zhang, Shuchang Liu, Ziru Liu, Rui Zhong, Qingpeng Cai, Xiangyu Zhao, Chunxu Zhang, Qidong Liu, and Peng Jiang. Llm-powered user simulator for recommender system. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 13339–13347, 2025.   
[40] Keyu Zhao, Fengli Xu, and Yong Li. Reason-to-recommend: Using interactionof-thought reasoning to enhance llm recommendation. arXiv preprint arXiv:2506.05069, 2025.   
[41] Zihuai Zhao, Wenqi Fan, Jiatong Li, Yunqing Liu, Xiaowei Mei, Yiqi Wang, Zhen Wen, Fei Wang, Xiangyu Zhao, Jiliang Tang, et al. Recommender systems in the era of large language models (llms). IEEE Transactions on Knowledge and Data Engineering, 36(11):6889–6907, 2024.   
[42] Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911, 2023.   
[43] Joyce Zhou, Yijia Dai, and Thorsten Joachims. Language-based user profiles for recommendation. arXiv preprint arXiv:2402.15623, 2024.

Table 5: Dataset Statistics   

<table><tr><td>Num. Queries</td><td>6096</td></tr><tr><td>Num. Users</td><td>6039</td></tr><tr><td>Num. Categories</td><td>28</td></tr><tr><td>Num. Candidate Items</td><td>1,058,417</td></tr><tr><td>Num. User-Item Interactions</td><td>577,376</td></tr></table>

# Prompt Template for Pre-processing Queries

System Prompt: Rewrite the "Original Query" into a single, casual, conversational question. Rules:

- Keep: 1 core item $+ \nobreakspace 1 \nobreakspace$ core use case (e.g., "a laptop for gaming").   
- Delete: All other details (including brand, specs, price, personal preferences, etc.).   
- Tone: Natural and conversational.   
- Format: Must be a single sentence.

Original Query: I’m in need of an external Blu-ray drive that actually works well with reading Blu-rays. The one I currently own has a poor ability to read discs, requiring multiple tries and hoping for a successful read. I want a Blu-ray drive that is amazing by comparison, where I can simply insert a disc and it starts playing right away. Skipping around on the disc should be smooth and there shouldn’t be any endless seeking from the drive. I came across one drive that shows up as a Pioneer BDR-TD03 in the properties, and based on its performance, I’m really satisfied with its quality. The name "Dainty" might be unusual for a disk drive, but it doesn’t affect its functionality, which is fantastic.

Rewritten Query: Hey, can you recommend an external Blu-ray drive that reads discs reliably and plays them smoothly right from the start?

# APPENDIX

# A Dataset

We derive our experimental data from the Amazon-C4 corpus, which contains user queries synthesized by ChatGPT from product review text. These synthetic queries often include fine-grained, product-level details (e.g., brand names, model identifiers, and exact technical specifications) that are uncommon in real-world user utterances. In practice, users tend to give concise or underspecified requests—such as “recommend a budget laptop for college” or “gift ideas for my father who likes hiking”—and expect the recommender to infer context from prior interactions or stored user memories. To better emulate these memory-driven recommendation scenarios, we transform Amazon-C4 so that queries resemble realistic, high-level requests. Concretely, we (i) remove or mask overly specific attributes (brand names, exact model identifiers, serial numbers, and verbatim review excerpts) and (ii) preserve the user’s core intent and constraints (product category, coarse price band, primary use case). To achieve this, we prompt GPT-o3-mini to produce concise, naturalistic rewrites that retain the original intent and constraints while omitting gratuitous detail. The prompt

used for simplification and representative before/after examples are provided below.

Due to the limited availability of API resources, we randomly sampled 6,096 queries from the Amazon-C4 dataset and applied the aforementioned preprocessing procedure to this subset. The resulting dataset retains the essential characteristics necessary for our experiments while remaining computationally manageable. A detailed summary of the processed dataset is provided in Table 5.

# B Implementation Details

# B.1 Environment

Our experiments were conducted using 4 x NVIDIA A800(80GB) GPUs. We employed PyTorch 2.6.0 as the deep learning framework, Transformers 4.48.0 for model implementation, and the GRPO implementation provided by VeRL. For inference acceleration, we utilized SgLang 0.4.6.

# B.2 Prompts for LLM-based Baselines

With the exception of BLAIR, all other baselines are LLM-based. Following the Rec-R1 framework, we prompt the LLM to generate an ideal item profile corresponding to a given user query. Specifically, we employ the following prompt template: "You are an expert in generating queries for dense retrieval. Given a customer query, your task is to retain the original query while expanding it with additional semantically relevant information, retrieve the most relevant products, ensuring they best meet customer needs. If no useful expansion is needed, return the original query as is. # Below is the product search query:{query}"

# B.3 Details for Our Method

For both memory retrieval and item retrieval, we adopt the Qwen3- Embedding-0.6B model as the retrieval encoder. For reinforcement learning with LLMs, the learning rate is set to 1e5 with a batch size of 256. During trajectory generation, we configure the group size $\left( \mathrm { G } = 5 \right)$ ) and use a rollout temperature of 1.0.

# C Prompts Used in Our Method

The prompts employed in our method fall into two primary categories: Memory Indexing and LLM-based Recommendation.

# C.1 Prompts for Memory Indexing

C.1.1 Cross-User Global Memory. To construct the Cross-User Global Memory, we sample interaction histories from multiple users across different recommendation scenarios (e.g., purchasing products from various categories). These histories include user queries, selected items, and their associated metadata. The LLM is then prompted to analyze each scenario and infer additional aspects that users might consider beyond what is explicitly mentioned in the query, thereby extracting richer recommendation experiences. To encourage the LLM to better reason about these additional aspects, we also sample items that are similar to the ground-truth purchases but were not selected by the user as negative items. The prompt used in this process is as above.

Using the prompt described above, the LLM analyzes each user’s interaction data to extract recommendation knowledge specific to the corresponding scenario. To capture generalizable cross-user

# Prompt for Cross-User Global Memory (A)

You are given the following information for a recommendation task:

-Recommendation scenario: {scenario}   
-User query: {query}   
-Ground truth item (the item finally purchased): {ground truth item metadata}   
-Negative items (similar items not chosen by the user): {negative items}

# Your task:

1. Analyze the user query, the purchased item, and the unchosen items.   
2. Identify additional aspects or dimensions that the user might have implicitly considered beyond what is explicitly stated in the query.

These aspects should support more personalized recommendations.

At the category level: What general personalization factors might users care about beyond the query, such as overall budget, preferred brands, aesthetics, or usage context? Do not mention specific product or features in this level.

At the subcategory level (e.g., the relevant subcategory of ground truth item): What specific personalization factors might be relevant, especially when comparing the ground truth item with the negative items (e.g., screen size, smart features, energy efficiency, style)?   
3. Focus on uncovering subtle or implicit preferences (e.g., color, price sensitivity, design style, brand affinity, feature trade-offs) that can be inferred by comparing the ground truth item with the negative items.

# Output format (structured):

```json
{ "category_level_personalization"Aspects": [ 
    "category": "category_name",
    "aspect": "aspect_name",
    "description": "description"
}, 
"subcategory_level_personalization"Aspects": [ 
    "subcategory": "subcategory_name",
    "aspect": "aspect_name",
    "description": "description "
], 
} 
```

global memory, we sample interactions from multiple users and prompt the LLM to aggregate the extracted knowledge, producing a concise and coherent summary that represents common recommendation considerations within the scenario. The prompt used for this process is presented below:

C.1.2 User-specific Local Memory. Our proposed User-specific Local Memory is organized into three hierarchical levels: Behavior Records, Preference Patterns, and User Profile. These levels capture user information at progressively finer granularity, ranging from abundant and potentially noisy interaction data to highly abstracted and fine-grained summaries. Specifically, Preference Patterns are derived from*Behavior Records, and the User Profile represents a further distillation and consolidation of these preference patterns, providing a succinct yet informative summary of the user’s interests and tendencies.

For a given user, we collect their interaction history within a specific recommendation scenario and prompt the LLM to analyze these interactions to identify Preference Patterns. This process

enables the extraction of recurring behaviors, interests, and implicit constraints that characterize the user’s preferences within the scenario. The prompt used for this analysis is as follows:

After generating Preference Patterns for a user across different scenarios, we prompt the LLM to reason over these patterns to infer the user’s potential User Profile. This step consolidates scenariospecific preferences into a coherent and high-level summary of the user’s interests and characteristics. The prompt used for this inference process is as follows:

# C.2 Prompts for LLM-based Recommendation

In this study, we develop an LLM-based recommendation assistant that integrates memory and reasoning. To enable the LLM to autonomously leverage user-specific local memory guided by crossuser global memory, we employ the following prompt, which instructs the model to generate keywords for global memory retrieval.

# Prompt for Cross-User Global Memory (B)

You are given some aspects that multiple users consider important for a category.

Category: {category}

Current aggregated aspects: {aspects}

Task: - Merge semantically duplicate or highly overlapping aspects into unified canonical names. - Normalize naming style (use consistent Title Case English or best canonical phrasing). - Consolidate and de-duplicate descriptions under each merged aspect; keep key insights, remove redundancy. - These are shared aspects of multiple users, so do not use any specific product/parameter/user in the description.

Output strictly as a compact JSON object mapping aspect_name to list of descriptions, e.g.:

```python
{""aspect_name": ["description", ...], ... }} 
```

# Prompt for LLM-based Recommendation

You are a recommendation expert. Your task is to generate an ideal item profile based on a user’s query and memories. When given a user query, you should analyze the user’s query and identify additional factors that might refine the recommendation. {Retrieved Cross-User Global Memory Based on Query}

You should retrieve current user’s memories that related to these aspects via ‘memory_retrieval_tool‘. After retrieving the memories, combine them with the user’s query, think step by step to reason about an ideal item profile for item retrieval. This profile should be as detailed as possible, do not miss any important keywords, otherwise relevant items may be missed. Your final output should be structured as follows:

\boxed{"ideal_item_profile": str,

"useful_memory_ids": List[str]}

Remember: First, retrieve the relevant memories, then reason about detailed ideal item profile based on the query and the memories you retrieved.

# Prompt for Extracting Preference Patterns

You are given the following information about a user’s purchase history in a specific category:

- Category: {category} - User’s purchased items in this category, each with its metadata and the user’s review: {reviews}

Your task: Analyze the items (including their metadata and the user’s reviews) and summarize the user’s preferences in this category. The summary should capture consistent patterns across items and reviews (e.g., favored brands, preferred price range, styles, features, quality expectations). Be as detailed as possible, but do not fabricate information that is not supported by the input. Directly output the preference summary (string) below:

# Prompt for Generating User Profile

You are given the following information about a user’s preferences across different categories or aspects: {preference patterns}

Your task: Based on these preferences, infer the user’s overall profile. The profile should summarize general traits, patterns, and tendencies that can be reasonably inferred from the given preferences (e.g., spending habits, brand inclination, style choices, feature priorities, quality expectations, lifestyle hints). Do not fabricate information that is not supported by the input. Directly output the profile (string) below:

The LLM then uses the retrieved memory to perform reasoning and produce personalized recommendations.
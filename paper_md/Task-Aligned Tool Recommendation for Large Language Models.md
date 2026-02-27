# Task-Aligned Tool Recommendation for Large Language Models

Hang Gao

Rutgers University

h.gao@rutgers.edu

Yongfeng Zhang

Rutgers University

yongfeng.zhang@rutgers.edu

# Abstract

By augmenting Large Language Models (LLMs) with external tools, their capacity to solve complex problems has been significantly enhanced. However, despite ongoing advancements in the parsing capabilities of LLMs, incorporating all available tools simultaneously in the prompt remains impractical due to the vast number of external tools. Consequently, it is essential to provide LLMs with a precise set of tools tailored to the specific task, considering both quantity and quality. Current tool retrieval methods primarily focus on refining the ranking list of tools and directly packaging a fixed number of top-ranked tools as the tool set. However, these approaches often fail to equip LLMs with the optimal set of tools prior to execution, since the optimal number of tools for different tasks could be different, resulting in inefficiencies such as redundant or unsuitable tools, which impede immediate access to the most relevant tools. This paper addresses the challenge of recommending precise toolsets for LLMs. We introduce the problem of tool recommendation, define its scope, and propose a novel Precision-driven Tool Recommendation (PTR) approach. PTR captures an initial, concise set of tools by leveraging historical tool bundle usage and dynamically adjusts the tool set by performing tool matching, culminating in a multi-view-based tool addition. Additionally, we present a new dataset, RecTools, and a metric, TRACC, designed to evaluate the effectiveness of tool recommendation for LLMs. We further validate our design choices through comprehensive experiments, demonstrating promising accuracy across two open benchmarks and our RecTools dataset. Our code and data are publicly available at https://github.com/GHupppp/PTR.

# 1 Introduction

Large Language Models (LLMs) have established themselves as powerful intermediaries, demonstrating remarkable impacts across a variety of down-

stream tasks, including text generation, code debugging, and personalized recommendations (Brown et al., 2020; Touvron et al., 2023; Nam et al., 2024; Chen et al., 2024a; Zhao et al., 2024). However, as these models continue to evolve, they still struggle to solve highly complex problems due to limitations arising from their pre-training data (Mialon et al., 2023; Mallen et al., 2022; Yuan et al., 2023). To expand the potential of LLMs in managing more complex tasks efficiently, recommendations at various levels have been increasingly applied to LLMs. Typically, memory recommendations (Borgeaud et al., 2022) and knowledge-based recommendations (Gao et al., 2023; Hu et al., 2023) enhance consistency and context awareness in ongoing tasks for LLMs, while data augmentation recommendations (Xu et al., 2020) facilitate the inclusion of additional data to augment training. Furthermore, architecture recommendations (Elsken et al., 2019; Fedus et al., 2022) and prompt recommendations (Shin et al., 2020; Pryzant et al., 2023; Liu et al., 2023) optimize efficiency and generate more relevant outputs. Simultaneously, to reduce the cognitive load on LLMs and enhance their complex problem-solving capabilities by enabling actions beyond natural language processing, it is crucial to augment LLMs with recommendations of optimal external tool sets, an aspect currently lacking in existing recommendation frameworks for LLMs. Furthermore, this approach will be helpful to address the challenge of input length limitations encountered when incorporating a large number of external tools into the prompt. Providing LLMs with a precise and dynamically adaptable recommended toolset can help to enhance the effectiveness of LLM’s task-solving ability.

Considering that the capability of LLMs to master and control external tools is instrumental in overcoming some of their fundamental weaknesses, the field of tool retrieval, which aims to identify the top-K most suitable tools for a given query from a vast

![](images/bc135e5a3acd7a393b375457b63781ceb98045067d814c5e4f42d4912a7f457d.jpg)  
Figure 1: Tool retrieval often provides a broad and variable number of tools with inconsistent quality, whereas tool recommendation delivers a precise, high-quality set of tools directly.

set of tools, has been increasingly explored. The advent of tool retrieval (Zhuang et al., 2023; Li et al., 2023; Tang et al., 2023; Yang et al., 2024) signifies a nuanced evolution, most directly employing termbased methods (Sparck Jones, 1972; Robertson et al., 2009) or semantic-based techniques (Kong et al., 2023; Yuan et al., 2024; Gao et al., 2024). Generally, the primary objective of these methods is to refine the ranked list of tools and subsequently select a fixed number of tools from the top (top-K) (Qu et al., 2024a; Zheng et al., 2024; Qu et al., 2024c). Although such approaches have demonstrated good performance when retrieving a single tool (Patil et al., 2023; Xu et al., 2023) or a small number of tools (generally fewer than three) (Qin et al., 2023; Huang et al., 2023), they remain susceptible to under-selection or over-selection, as illustrated in Figure.1. This limitation may prevent LLMs from addressing the current query or cause them to over-interpret the query, thereby reducing the effectiveness of LLMs in solving complex problems with external tools. Additionally, the validation of these methods often relies on datasets that use a fixed number of tools for each query, meaning that during testing, the number of tools to be used is known in advance—an unrealistic scenario in practical applications where the number of tools needed can vary dynamically. Therefore, recommending a precise and dynamically adjustable set of external tools to LLMs in a single step prior to query execution is increasingly important. This approach not only enhances the thoroughness of problem-solving but also improves efficiency by reducing the need to execute additional tools.

To address these limitations, we first provide a comprehensive explanation of tool recommendation and clearly define the problem (Appendix A), considering the lack of definition and the incompleteness of goals pursued by existing tool retrieval methods. Toward this objective, we propose PTR, a novel model-agnostic Precision-Driven

Tool Recommendation approach aimed at recommending a precise tool set for LLMs prior to query execution. By leveraging historical tool bundle usage data to uncover patterns of idiomatic use and dependencies between tools, this method is structured into three main stages: Tool Bundle Acquisition, Functional Coverage Mapping, and Multi-view-based Re-ranking. Initially, using traditional pre-trained language models, we acquire semantic matching information between queries and previously used tool bundles, thereby addressing potential performance issues of these models in zero-shot scenarios for tool recommendation tasks. Subsequently, to evaluate the effectiveness of the selected tool bundle in solving the query, LLMs are prompted to match tools with the specific subproblems they can address and to identify unresolved issues. Based on this, a multi-view-based re-ranking method is employed to select tools that can help resolve the identified issues and complement the existing tool sets. More specifically, to address the unresolved issues, we construct the final ranked list by aggregating three tool lists and ranking each tool based on their frequency of occurrence. The ranked tool list, constructed from multiple views, reduces the randomness associated with selecting tools from the entire available set.

Additionally, we construct a dataset, RecTools, tailored to specific queries with recommended tool sets. In contrast to previous tool datasets that standardize the number of tools used for each query (Huang et al., 2023) or employ a small number of tools (Qu et al., 2024a), our tool recommendation set incorporates varying numbers of tools for different queries, with up to ten tools used for a single query. This is achieved through an automated process in which LLMs are prompted to generate specific queries to be addressed by given tool bundles. These queries and tool bundles are subsequently evaluated by prompting LLMs to determine whether the selected tools adequately address the corresponding queries, ensuring that neither excess nor insufficient tools are utilized. Dedicated validation and deduplication steps are implemented to ensure the precision of tool usage, thereby enhancing the quality of the tool recommendation set.

Furthermore, traditional retrieval metrics such as Recall (Zhu, 2004) and Normalized Discounted Cumulative Gain (NDCG) (Järvelin and Kekäläinen, 2002), fail to capture the level of precision required for effective tool recommendation. The

absence of necessary tools can lead to the failure of LLMs in performing tasks, while the redundancy of tools may cause LLMs to generate unnecessary responses. This indicates that metrics focusing solely on completeness are inadequate for evaluating tool recommendation tasks. To bridge this gap, we introduce TRACC, a novel metric designed to assess tool recommendation performance, considering both the accuracy of the quantity and the quality of the recommended tools. TRACC serves as a reliable indicator of the effectiveness of tool recommendation processes.

To summarize, the main contributions of this work are as follows:

1. We introduce tool recommendation as a novel problem, necessitating the provision of precise tool sets to LLMs for a given query. We propose PTR, an effective tool recommendation approach that leverages historical tool bundle information between queries and tools, resulting in a more accurate and comprehensive final recommended tool list.   
2. We present a new dataset, RecTools, and an effective evaluation metric, TRACC, specifically designed to assess tool recommendation for LLMs. This not only addresses gaps in existing tool sets but also advances future research related to tool recommendation.   
3. Extensive experiments validate the effectiveness of RecTools and demonstrate the efficacy of PTR in recommending tools for LLMs. The recommended tool sets are both comprehensive and accurate, enhancing the overall performance of LLMs in processing tasks.

# 2 The Precision-driven Tool Recommendation

We introduce a novel approach, Precision-driven Tool Recommendation (PTR), to address the challenges faced by prior research through a three-stage recommendation process: (1) Tool Bundle Acquisition, which involves establishing a potentially useful tool bundle by leveraging past usage patterns across all tool combinations, as opposed to relying solely on instructions for individual tool usage; (2) Functional Coverage Mapping, which entails effectively mapping the tools from the acquired tool bundle to the functionalities of the original query, thereby identifying which tools should be retained and which should be discarded, resulting in any remaining unsolved sub-problems; and (3)

Multi-view Based Re-ranking, which involves the effective re-ranking of relevant tools from a large tool set, tailored to each unsolved sub-problem identified in the second stage, and selecting the top-ranked tool after re-ranking to complete the final recommended toolset. The overview of our approach is illustrated in Figure.2. Please note that all symbols are globally defined in Section 2. In the following sections, we present the details of these three PTR recommendation stages.

# 2.1 Tool Bundle Acquisition

To obtain a initiate set of tools, we employ an retriever to capture the relevance between historical tool combinations and the current query. Unlike existing methods that focus on retrieving single tools by analyzing the relationship between a query and individual tools, our approach introduces tool bundle retrieval. By leveraging historical tool combinations, we capture a richer contextual relationship between queries and sets of tools that have been used together effectively in the past. This facilitates a more holistic understanding of tool dependencies and synergies, thereby enhancing the relevance of retrieved tool sets for complex queries. Specifically, Let $T = \{ T _ { 1 } , T _ { 2 } , \dots , T _ { n } \}$ be the set of all available tools. Let $D = \{ ( Q _ { i } , B _ { i } ) \} _ { i = 1 } ^ { M }$ represent a set of past queries and their associated tool bundles, where $Q _ { i }$ is a past query, and $B _ { i }$ is the corresponding tool bundle used for $Q _ { i }$ , with $B _ { i } \subseteq T$ . The collection of unique tool bundles is $B = \{ B _ { 1 } , B _ { 2 } , . . . , B _ { N } \}$ . Given a new query $Q$ , we select a tool bundle $B _ { K } = \{ T _ { 1 } , \ldots , T _ { z } \}$ from $B$ that is most relevant to $Q$ through the retriever, which ideally contains tools potentially useful. The subsequent recommendations operate on this obtained tool bundle—either based on sparse representations or dense representations.

# 2.2 Functional Coverage Mapping

As illustrated in Figure.3, functional coverage mapping presents a structured approach to evaluate and optimize a set of tools in relation to a specific query. By systematically aligning required functionalities with the capabilities of available tools, this method ensures that the toolset comprehensively addresses the user’s needs while minimizing redundancies and identifying any gaps, as each tool may correspond to multiple functionalities. At its core, Functional Coverage Mapping aims to determine whether an initial set of tools $B _ { K } = \{ T _ { 1 } , T _ { 2 } , \ldots , T _ { z } \}$ adequately ful-

![](images/c69fe924c782523533ad0cae29a289426688645f9b163febae6a46d0e65c3b1c.jpg)  
Stage-1: Tool Bundle Acquisition   
Stage-2: Functional Coverage Mapping   
Figure 2: Architecture of the three-stage recommendation framework PTR for tool recommendation.

![](images/ec2781e9d38d6e49cf9e25e28101ec82460ea19b97215f7d893081782cd0ee93.jpg)  
Figure 3: The four stages of Functional Coverage Mapping in PTR.

fills a query $Q$ with its key functionalities $F =$ $\{ F _ { 1 } , F _ { 2 } , \ldots , F _ { m } \}$ . Specifically, Functional Coverage Mapping achieves this objective through four steps: Extraction of Key Requirements, Matching Tools to Functionalities, Assessment of Toolset Completeness, and Identification of Unsolved Problems, which are described as follows:

Extraction of Key Functionalities. The first step involves decomposing the user’s query $Q$ into a set of discrete and actionable functionalities $F$ . This extraction ensures a comprehensive understanding of the query that the toolset must address. This extraction is achieved by prompting the language model to identify and enumerate these functionalities directly from the query, ensuring that both explicit and implicit functionalities are captured. Matching Tools to Functionalities. Once the key

functionalities $F$ are established, the subsequent phase entails mapping each functionality $F _ { i }$ to the tools $T _ { j }$ within the obtained tool bundle $B _ { K }$ . This mapping process determines which tools are capable of fulfilling specific functionalities. To achieve this, targeted prompts are employed with the language model, directing it to associate each functionality with the most suitable tool based on tool descriptions.

Assessment of Toolset Completeness. With the mapping $M ( F , B _ { K } )$ established, the method evaluates whether the toolset $B _ { K }$ fully addresses all functionalities $F$ . This assessment categorizes the toolset into one of three scenarios: (1) Exact Solving: All functionalities are met by all tools without any redundancies; (2) Oversolving: The toolset includes tools that provide functionalities not required by the query; and (3) Partial Solving: Some functionalities remain unfulfilled and some tools remain unused. Based on the identified scenario, the tool bundle is optimized by retaining essential tools and discarding redundant ones. Tools that do not contribute to fulfilling any requirement are removed to streamline the toolset.

Identification of Unsolved Problems. In cases of partial solving, the method identifies the remaining unsolved problems directly from the original query $Q$ . These unsolved problems $U \ =$ $\{ U _ { 1 } , U _ { 2 } , \ldots , U _ { y } \}$ are presented in a format that can be directly utilized in the subsequent recommendation stage. To achieve this, the language model is prompted to extract the unmet functionalities without further functional decomposition. This

approach ensures that each unsolved problems retains the context of the original query $Q$ , thereby facilitating seamless integration with the following re-ranking method. Furthermore, this direct identification allows for straightforward utilization in the following re-ranking process, where each unsolved problem can be addressed individually.

# 2.3 Multi-view Based Re-ranking

Addressing the challenge of selecting pertinent tools from an extensive toolset to resolve unresolved problems requires comprehensive consideration. The proposed PTR employs a multifaceted similarity evaluation strategy that integrates three essential dimensions of the unresolved problem $U _ { j }$ : (1) Direct Semantic Alignment, wherein the system quantifies the semantic similarity between the user query and each available tool, ensuring the immediate identification of tools intrinsically aligned with the query’s intent; (2) Historical Query Correlation, which involves analyzing past queries that closely resemble the current one to extract tools previously utilized in similar contexts, thereby enriching the current toolset with empirically effective solutions while maintaining uniqueness through aggregation and deduplication; and (3) Contextual Tool Expansion, which leverages the most relevant tool identified through direct semantic alignment to retrieve additional tools exhibiting high similarity to this primary tool, thereby uncovering supplementary options that may offer complementary or alternative functionalities beneficial to the user’s query. The multi-view matching process involves obtaining the tool list $L$ through direct semantic alignment (DSA), historical query correlation (HQC), and contextual tool expansion (CTE), respectively. These three tool lists are then aggregated and ranked according to their frequency of occurrence, with the most frequent tools being selected. After performing the multi-view based reranking for each unsolved problem, the top-ranked tool in each list is selected and added to the final recommended toolset. In some cases, it is also possible that this tool already exists in the toolset acquired from the second-stage recommendation; in such instances, the tool will be ignored. The algorithm for multi-view-based re-ranking is summarized in Algorithm.1.

# Algorithm 1 Multi-view Based Re-ranking

Require: Unresolved problem $U _ { j }$ , Toolset $\begin{array} { r l r } { T } & { { } \ } & { = { } \ } & { \left\{ T _ { 1 } , T _ { 2 } , \ldots , T _ { n } \right\} } \end{array}$ , Historical queries $\begin{array} { r c l } { Q } & { = } & { \{ Q _ { 1 } , Q _ { 2 } , \dots , Q _ { m } \} } \end{array}$ , Select $\kappa$ represents the function that selects the top $K$ candidates with the highest similarity, $\sigma$ indicates the similarity measure.

Ensure: Recommended Tool $\tau$ .

1: Initialize lists: LDSA, LHQC, $L _ { \mathrm { C T E } }$

/ /Direct Semantic Alignment

2: $L _ { \mathrm { D S A } }  \mathrm { S e l e c t } _ { K }$ $( \{ T _ { i } \in T \mid \sigma ( U _ { j } , T _ { i } ) \} )$

/ /Historical Query Correlation

3: LHistoricalQuery ← SelectK $( \{ Q _ { i } \in Q \mid \sigma ( U _ { j } , Q _ { i } ) \} )$

4: for each query $Q _ { i }$ in LHistoricalQuery do

5: for each tool $T _ { l }$ used in $Q _ { i }$ do

Add $T _ { l }$ to $L _ { \mathrm { H Q C } }$

7: end for

8: end for

9: Remove duplicates from $L _ { \mathrm { H Q C } }$

/ /Contextual Tool Expansion

10: if LDSA is not empty then   
11: $T _ { \mathrm { p r i m a r y } }  L _ { \mathrm { D S A } } [ 0 ]$   
12: LCTE ← SelectK ({Ti ∈ T | σ(Tprimary, Ti)})   
13: end if   
14: Combine lists: $L _ { \mathrm { C o m b i n e d } }  L _ { \mathrm { D S A } } + L _ { \mathrm { H Q C } } + L _ { \mathrm { C T E } }$   
15: Count frequency of each tool in $L _ { \mathrm { C o m b i n e d } }$   
16: Rank tools by frequency in descending order.   
17: Select the top ranked tool as $\tau$

return $\tau$ .

# 3 Datasets and Metrics

Datasets. To verify the effectiveness of PTR, we utilize three datasets for tool recommendation: ToolLens (Qu et al., 2024a), MetaTool (Huang et al., 2023), and a newly constructed dataset, Rec-Tools. We randomly select $20 \%$ of each dataset to serve as the test data. Both ToolLens and MetaTool focus on multi-tool tasks, leading us to select them as the primary datasets for our experiments. While ToolLens uniquely emphasizes creating queries that are natural, concise, and intentionally multifaceted, MetaTool is a benchmark designed to evaluate whether LLMs possess tool usage awareness and can correctly choose appropriate tools.

Metrics. As evaluation metrics for tool recommendation, following previous work focusing on tool retrieval (Gao et al., 2024; Qu et al., 2024c), the widely used retrieval metrics are Recall and NDCG. However, they do not adequately address the requirements for accuracy in both the number of recommended tools and the specific tools recommended, disregarding the impact of differences in size between the tool sets. Therefore, to further tailor the assessment to the challenges of tool recommendation tasks, we introduce a new metric, named TRACC. This metric is designed to measure the extent to which the recommended toolset aligns with the ground-truth set in terms of both the

accuracy of the number of tools and the accuracy of the tools themselves:

$$
\mathrm {T R A C C} = \left(1 - \frac {1}{| A \cup B |} \cdot | n _ {2} - n _ {1} |\right) \cdot A C C
$$

where $A$ denotes the ground-truth tool set and $B$ represents the recommended tool set. The cardinalities of $A$ and $B$ are denoted by $n _ { 1 }$ and $n _ { 2 }$ , respectively. And $| A \cup B |$ signifies the cardinality of the union of $A$ and $B$ . ACC represents $\frac { | A \cap B | } { n _ { 1 } }$ , where $| A \cap B |$ indicates the size of the their intersection. More details can be found in Appendix C.

# 4 Experiments

# 4.1 Implementation Details

Baselines. We considered the following baselines: Random, which randomly select from historical tools; BM25 (Robertson et al., 2009), a classical sparse retrieval method that extends TF-IDF by leveraging term frequency and inverse document frequency of keywords; Contriever (Izacard et al., 2021), which utilizes inverse cloze tasks, cropping for positive pair generation, and momentum contrastive training to develop dense retrievers; SBERT (Reimers and Gurevych, 2019), a library providing BERT-based sentence embeddings. Specifically, we use all-mpnet-base-v2; TAS-B (Hofstätter et al., 2021), the retriever introduces an efficient topic-aware query and balanced margin sampling technique; And SimCSE (Gao et al., 2021), a simple contrastive learning framework that greatly advances state-of-the-art sentence embeddings.

Besides, we initially implement the PTR using the open source model open-mistral-7b, due to its cost-effectiveness. Subsequently, we evaluate PTR with the model GPT-3.5-turbo and GPT-4o, to determine its effectiveness when employing a more advanced model. For evaluation metrics, in addition to the specifically designed TRACC metric, we also calculate Recall $@ \mathbf { K }$ and ${ \mathrm { N D C G } } @ { \mathrm { K } }$ , reporting these metrics with K set to the size of the ground-truth tool set.

# 4.2 Experimental Results

Table 1 presents the main results of the PTR applied to ToolLens, MetaTool, and RecTools using various models and unsupervised retrievers. Based on these findings, we draw the following observations and conclusions.

We first observe that the MetaTool dataset yields notable performance, whereas other datasets exhibit comparatively standard. This discrepancy can be attributed to the presence of relatively straightforward patterns within the MetaTool dataset, which motivates us the construction of a structurally diversified and high-quality tool-query dataset. Furthermore, the Random baseline indicates that random sampling of tool bundles leads to relatively poor performance, whereas other unsupervised retrievers outperform the Random baseline, particularly in the ToolLens dataset. This suggests that, although the latter two phases of the PTR can supplement or refine the recommended tool set, employing a targeted bundle in the early stages can enhance PTR performance. Conversely, the Sim-CSE approach demonstrated a significant improvement over the Random baseline, especially when utilizing GPT-4o as the backbone. Absolute Recall $@ \mathbf { K }$ improvements of 0.141, 0.111, and 0.117 were observed on the ToolLens, MetaTool, and RecTools datasets, respectively, highlighting the SimCSE method’s capability to leverage tool bundle information for more effective tool recommendation. Despite this advantage, all the methods fall short in the TRACC metric, which is specifically designed for evaluating precision in tool recommendation. This suggests that, although effective for tool retrieval tasks, Recall $@ \mathrm { K }$ and ${ \mathrm { N D C G } } @ { \mathrm { K } }$ may not fully satisfy the unique requirements of tool recommendation. Additionally, the results demonstrate that PTR consistently achieves strong performance when utilizing GPT-4o, confirming that PTR remains beneficial for tool recommendation even when employing more capable backbone models.

Overall, PTR exhibits effectiveness across all metrics and datasets, attributable to its implementation of a three-stage recommendation framework. This framework comprises tool bundle acquisition, functional coverage mapping for the deletion or retention of tools, and multi-view-based re-ranking for the addition of tools. By employing this structured approach, PTR dynamically addresses the entirety of the query, thereby facilitating the recommendation of a precise and well-tailored tool set.

# 4.3 Further Analysis

In this section, we conduct an in-depth analysis of the effectiveness for PTR, using the same datasets and evaluation metrics.

Table 2 reports the ablation results of PTR,

Table 1: Performance comparisons of PTR under different methods within different backbones on ToolLens, MetaTool, and RecTools datasets. “N/A” indicates that this method works alone. The best results are bolded, the best results of each colunmn are denoted as “∗”.   

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Framework</td><td colspan="3">ToolLens</td><td colspan="3">MetaTool</td><td colspan="3">RecTools</td></tr><tr><td>Recall@K</td><td>NDCG@K</td><td>TRACC</td><td>Recall@K</td><td>NDCG@K</td><td>TRACC</td><td>Recall@K</td><td>NDCG@K</td><td>TRACC</td></tr><tr><td rowspan="4">Random</td><td>N/A</td><td>0.036</td><td>0.061</td><td>0.034</td><td>0.133</td><td>0.202</td><td>0.133</td><td>0.137</td><td>0.271</td><td>0.097</td></tr><tr><td>+PTR+open-mistral-7b</td><td>0.185</td><td>0.225</td><td>0.145</td><td>0.608</td><td>0.785</td><td>0.505</td><td>0.457</td><td>0.756</td><td>0.235</td></tr><tr><td>+PTR+GPT-3.5-turbo</td><td>0.213</td><td>0.282</td><td>0.172</td><td>0.645</td><td>0.823</td><td>0.543</td><td>0.475</td><td>0.784</td><td>0.288</td></tr><tr><td>+PTR+GPT-4o</td><td>0.227</td><td>0.303</td><td>0.187</td><td>0.663</td><td>0.843</td><td>0.562</td><td>0.492</td><td>0.802</td><td>0.305</td></tr><tr><td rowspan="4">BM25</td><td>N/A</td><td>0.131</td><td>0.194</td><td>0.125</td><td>0.429</td><td>0.603</td><td>0.429</td><td>0.486</td><td>0.596</td><td>0.382</td></tr><tr><td>+PTR+open-mistral-7b</td><td>0.206</td><td>0.254</td><td>0.162</td><td>0.659</td><td>0.834</td><td>0.554</td><td>0.524</td><td>0.795</td><td>0.355</td></tr><tr><td>+PTR+GPT-3.5-turbo</td><td>0.247</td><td>0.313</td><td>0.193</td><td>0.694</td><td>0.874</td><td>0.593</td><td>0.541</td><td>0.815</td><td>0.408</td></tr><tr><td>+PTR+GPT-4o</td><td>0.261</td><td>0.331</td><td>0.208</td><td>0.712</td><td>0.892</td><td>0.612</td><td>0.545</td><td>0.810</td><td>0.414</td></tr><tr><td rowspan="4">Contriever</td><td>N/A</td><td>0.130</td><td>0.190</td><td>0.121</td><td>0.439</td><td>0.672</td><td>0.439</td><td>0.367</td><td>0.786</td><td>0.304</td></tr><tr><td>+PTR+open-mistral-7b</td><td>0.208</td><td>0.256</td><td>0.164</td><td>0.662</td><td>0.837</td><td>0.557</td><td>0.512</td><td>0.773</td><td>0.342</td></tr><tr><td>+PTR+GPT-3.5-turbo</td><td>0.250</td><td>0.316</td><td>0.196</td><td>0.697</td><td>0.877</td><td>0.596</td><td>0.528</td><td>0.792</td><td>0.396</td></tr><tr><td>+PTR+GPT-4o</td><td>0.264</td><td>0.334</td><td>0.211</td><td>0.715</td><td>0.895</td><td>0.615</td><td>0.559</td><td>0.834</td><td>0.426</td></tr><tr><td rowspan="4">SBERT</td><td>N/A</td><td>0.251</td><td>0.349</td><td>0.209</td><td>0.495</td><td>0.725</td><td>0.495</td><td>0.496</td><td>0.772</td><td>0.434</td></tr><tr><td>+PTR+open-mistral-7b</td><td>0.272</td><td>0.362</td><td>0.226</td><td>0.682</td><td>0.862</td><td>0.582</td><td>0.538</td><td>0.821</td><td>0.452</td></tr><tr><td>+PTR+GPT-3.5-turbo</td><td>0.308</td><td>0.403</td><td>0.252</td><td>0.723</td><td>0.902</td><td>0.623</td><td>0.555</td><td>0.840</td><td>0.484</td></tr><tr><td>+PTR+GPT-4o</td><td>0.322</td><td>0.422</td><td>0.268</td><td>0.741</td><td>0.921</td><td>0.642</td><td>0.572</td><td>0.859</td><td>0.501</td></tr><tr><td rowspan="4">TAS-B</td><td>N/A</td><td>0.279</td><td>0.381</td><td>0.263</td><td>0.657</td><td>0.897</td><td>0.657</td><td>0.509</td><td>0.841</td><td>0.454</td></tr><tr><td>+PTR+open-mistral-7b</td><td>0.298</td><td>0.398</td><td>0.278</td><td>0.702</td><td>0.882</td><td>0.602</td><td>0.552</td><td>0.854</td><td>0.472</td></tr><tr><td>+PTR+GPT-3.5-turbo</td><td>0.335</td><td>0.438</td><td>0.305</td><td>0.741</td><td>0.922</td><td>0.642</td><td>0.567</td><td>0.872</td><td>0.505</td></tr><tr><td>+PTR+GPT-4o</td><td>0.352</td><td>0.456</td><td>0.321</td><td>0.759</td><td>0.941</td><td>0.661</td><td>0.583</td><td>0.890</td><td>0.522</td></tr><tr><td rowspan="4">SimCSE</td><td>N/A</td><td>0.293</td><td>0.386</td><td>0.279</td><td>0.675</td><td>0.849</td><td>0.675</td><td>0.563</td><td>0.808</td><td>0.523</td></tr><tr><td>+PTR+open-mistral-7b</td><td>0.312</td><td>0.407</td><td>0.291</td><td>0.716</td><td>0.897</td><td>0.631</td><td>0.578</td><td>0.861</td><td>0.542</td></tr><tr><td>+PTR+GPT-3.5-turbo</td><td>0.350</td><td>0.448</td><td>0.319</td><td>0.756</td><td>0.937</td><td>0.671</td><td>0.594</td><td>0.879</td><td>0.575</td></tr><tr><td>+PTR+GPT-4o</td><td>0.368*</td><td>0.467*</td><td>0.336*</td><td>0.774*</td><td>0.956*</td><td>0.690*</td><td>0.609*</td><td>0.896*</td><td>0.591*</td></tr></table>

Table 2: Ablation results on three datasets.   

<table><tr><td>Dataset</td><td>Variant</td><td>Recall@K</td><td>NDCG@K</td><td>TRACC</td></tr><tr><td rowspan="3">ToolLens</td><td>w/o Stage-1</td><td>0.283</td><td>0.391</td><td>0.235</td></tr><tr><td>w/o Stage-2</td><td>0.355</td><td>0.455</td><td>0.322</td></tr><tr><td>w/o Stage-3</td><td>0.351</td><td>0.448</td><td>0.318</td></tr><tr><td></td><td>PTR</td><td>0.368</td><td>0.467</td><td>0.336</td></tr><tr><td rowspan="3">MetaTool</td><td>w/o Stage-1</td><td>0.745</td><td>0.922</td><td>0.677</td></tr><tr><td>w/o Stage-2</td><td>0.762</td><td>0.945</td><td>0.678</td></tr><tr><td>w/o Stage-3</td><td>0.759</td><td>0.942</td><td>0.676</td></tr><tr><td></td><td>PTR</td><td>0.774</td><td>0.956</td><td>0.690</td></tr><tr><td rowspan="3">RecTools</td><td>w/o Stage-1</td><td>0.581</td><td>0.916</td><td>0.439</td></tr><tr><td>w/o Stage-2</td><td>0.598</td><td>0.888</td><td>0.552</td></tr><tr><td>w/o Stage-3</td><td>0.601</td><td>0.892</td><td>0.566</td></tr><tr><td></td><td>PTR</td><td>0.609</td><td>0.896</td><td>0.591</td></tr></table>

where we individually remove each of the three stages, Tool Bundle Acquisition, Functional Coverage Mapping, and Multi-view Re-ranking, to examine their respective contributions. For this analysis, we adopt the GPT-4o backbone and SimCSE retriever, which achieved the strongest overall performance in the main experiments (Table 1). This setting allows us to isolate the effect of each stage under the most competitive configuration.

w/o Stage-1 (Tool Bundle Acquisition). Removing the initial bundle prevents PTR from leveraging historical co-usage patterns, forcing the model to start from unresolved queries without prior tool context. This leads to a clear decline across all datasets, particularly in TRACC (e.g., $0 . 3 3 6 $

0.235 on ToolLens), confirming that historical bundle information provides a strong inductive bias for identifying effective tool combinations.

w/o Stage-2 (Functional Coverage Mapping). Without this stage, PTR cannot explicitly align the retrieved tools with query-specific functionalities, resulting in redundant or missing tools. Although the model still benefits from bundle retrieval, TRACC drops considerably (e.g., $0 . 5 9 1 $ 0.552 on RecTools), showing that this stage is critical for achieving a minimal sufficient tool set by pruning irrelevant tools and surfacing unresolved sub-problems.

w/o Stage-3 (Multi-view Based Re-ranking). Eliminating the re-ranking phase removes the complementary views (semantic, historical, and contextual), causing the model to rely solely on the initial coverage output. The performance gap between this variant and the full model (e.g., $0 . 5 6 6 \to 0 . 5 9 1$ on RecTools) demonstrates that aggregating multiple similarity perspectives refines the final tool set and improves precision.

Performance w.r.t to accuracy in quantity. Furthermore, to evaluate the performance of PTR in terms of tool number precision, we calculate the average length difference between the recommended tool set and the ground truth tool set for each method and backbone. Figure.4 demonstrates the

![](images/97190eb9c8a67ba7c2698fb07147d31466fefcf4fd5384637159ec23846e6998.jpg)

![](images/dee64344dd582ba731e20dc88892ee07bc9a87bf7b862a3da5ff71634bf6a3b4.jpg)

![](images/99c11804648b913cffa2f688b7fb21fc7f2f555015c9d3d0a5245622e330df16.jpg)  
Figure 4: The average length difference between the recommended tool set and the ground truth tool set for each method and backbone.

effectiveness of PTR in maintaining consistency in the number of tools. In the MetaTool and ToolLens dataset, which exhibits relatively simple and small patterns, PTR clearly shows its effectiveness. Regarding our RecTools dataset, which has a variable structure and involves a wide range of tools for each query, the average length difference is effectively controlled within a considerable range, especially when it comes to the Embedding method.

# 5 Related Work

Early tool retrieval methods relied on term-based similarity measures such as BM25 (Robertson et al., 2009) and TF-IDF (Sparck Jones, 1972), which matched queries and tool descriptions through lexical overlap. With the advent of dense retrievers (Karpukhin et al., 2020; Guu et al., 2020; Xiong et al., 2020), neural encoders began to capture semantic relationships between queries and tool documents more effectively. Recent approaches further improve retriever training: Confucius (Gao et al., 2024) introduces an easy-to-difficult curriculum to enhance LLMs’ understanding of tool usage, while iterative feedback methods refine selection based on tool execution outcomes (Wang et al., 2023; Xu et al., 2024). ToolkenGPT (Hao et al., 2024) represents each tool as a learnable token (“toolken”), enabling tool invocation as a token-generation process. Additionally, research on diversity-aware retrieval (Carbonell and Goldstein, 1998; Gao and Zhang, 2024b) improves the coverage and complementarity of retrieved tools. Our work differs in its focus on pre-execution tool-set recommendation: selecting a minimal sufficient subset of tools for a query before any tool is invoked. This regime emphasizes precision prior to execution and is evaluated using both TRACC (which combines

sufficiency and minimality) and rank-based metrics. A distinct line of studies addresses execution-time tool use and refinement (Schick et al., 2023; Cai et al., 2024), including frameworks such as Re-Invoke (Chen et al., 2024b), DRAFT (Qu et al., 2024b), and TECTON (Alazraki and Rei, 2025), which optimize invocation success via feedback from real tool executions or trajectories. However, these advantages inherently depend on runtime feedback; converting them into an offline, preexecution variant removes their core learning signal and yields non-diagnostic surrogates. In this view, PTR can serve as a front-end recommender that supplies a lean, capability-complete tool set to downstream execution planners, while the two paradigms target distinct optimization goals.

# 6 Conclusions

This study presents a novel challenge, tool recommendation, and offers a precise formalization of the problem. In response, we propose a new approach, PTR, designed to improve the accuracy of tool recommendations, considering both the quantity and the selection of tools. PTR operates through three key stages: tool bundle acquisition, functional coverage mapping, and multi-view-based re-ranking. By dynamically adjusting the tool bundle obtained in the first stage, through the addition or removal of tools, PTR progressively refines the recommended toolset. Extensive experiments and detailed analyses showcase PTR’s effectiveness in addressing diverse query structures requiring multiple tool recommendations. Furthermore, we introduce Rec-Tools, a new dataset, along with TRACC, a comprehensive evaluation metric. Both serve as valuable contributions to the future research in the field of tool recommendation.

# 7 Limitations

Although our proposed framework demonstrates precise and reliable toolset recommendation for LLM agents, several aspects remain open for further study. For example, the effectiveness of our approach is partially shaped by the comprehension abilities of the underlying language models, which may yield minor variations in toolset selection when different models are employed. Moreover, our current implementation is optimized for text-based scenarios, and its extension to incorporate additional modalities, such as visual or structured data, could broaden its practical reach.

# References

Lisa Alazraki and Marek Rei. 2025. Meta-reasoning improves tool use in large language models. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 7885–7897.   
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, and 1 others. 2022. Improving language models by retrieving from trillions of tokens. In International conference on machine learning, pages 2206–2240. PMLR.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.   
Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. 2024. Large language models as tool makers. In ICLR.   
Jaime Carbonell and Jade Goldstein. 1998. The use of mmr, diversity-based reranking for reordering documents and producing summaries. In Proceedings of the 21st annual international ACM SIGIR conference on Research and development in information retrieval, pages 335–336.   
Jin Chen, Zheng Liu, Xu Huang, Chenwang Wu, Qi Liu, Gangwei Jiang, Yuanhao Pu, Yuxuan Lei, Xiaolong Chen, Xingmei Wang, and 1 others. 2024a. When large language models meet personalization: Perspectives of challenges and opportunities. World Wide Web, 27(4):42.   
Yanfei Chen, Jinsung Yoon, Devendra Sachan, Qingze Wang, Vincent Cohen-Addad, Mohammadhossein

Bateni, Chen-Yu Lee, and Tomas Pfister. 2024b. Reinvoke: Tool invocation rewriting for zero-shot tool retrieval. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 4705–4726.   
Thomas Elsken, Jan Hendrik Metzen, and Frank Hutter. 2019. Neural architecture search: A survey. Journal of Machine Learning Research, 20(55):1–21.   
William Fedus, Barret Zoph, and Noam Shazeer. 2022. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39.   
Hang Gao and Yongfeng Zhang. 2024a. Memory sharing for large language model based agents. arXiv preprint arXiv:2404.09982.   
Hang Gao and Yongfeng Zhang. 2024b. Vrsd: Rethinking similarity and diversity for retrieval in large language models. arXiv preprint arXiv:2407.04573.   
Shen Gao, Zhengliang Shi, Minghang Zhu, Bowen Fang, Xin Xin, Pengjie Ren, Zhumin Chen, Jun Ma, and Zhaochun Ren. 2024. Confucius: Iterative tool learning from introspection feedback by easy-to-difficult curriculum. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 18030– 18038.   
T Gao, X Yao, and Danqi Chen. 2021. Simcse: Simple contrastive learning of sentence embeddings. In EMNLP 2021-2021 Conference on Empirical Methods in Natural Language Processing, Proceedings.   
Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, and Haofen Wang. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997.   
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. 2020. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929–3938. PMLR.   
Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. 2024. Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings. Advances in neural information processing systems, 36.   
Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently teaching an effective dense retriever with balanced topic aware sampling. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 113–122.   
Linmei Hu, Zeyi Liu, Ziwang Zhao, Lei Hou, Liqiang Nie, and Juanzi Li. 2023. A survey of knowledge enhanced pre-trained language models. IEEE Transactions on Knowledge and Data Engineering.

Yue Huang, Jiawen Shi, Yuan Li, Chenrui Fan, Siyuan Wu, Qihui Zhang, Yixin Liu, Pan Zhou, Yao Wan, Neil Zhenqiang Gong, and 1 others. 2023. Metatool benchmark for large language models: Deciding whether to use tools and which to use. arXiv preprint arXiv:2310.03128.   
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118.   
Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated gain-based evaluation of ir techniques. ACM Transactions on Information Systems (TOIS), 20(4):422–446.   
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick˘ Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. arXiv preprint arXiv:2004.04906.   
Yilun Kong, Jingqing Ruan, Yihong Chen, Bin Zhang, Tianpeng Bao, Shiwei Shi, Guoqing Du, Xiaoru Hu, Hangyu Mao, Ziyue Li, and 1 others. 2023. Tptuv2: Boosting task planning and tool usage of large language model-based agents in real-world systems. arXiv preprint arXiv:2311.11315.   
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, and 1 others. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474.   
Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. 2023. Api-bank: A comprehensive benchmark for tool-augmented llms. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3102–3116.   
Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190.   
Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. 2023. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Computing Surveys, 55(9):1–35.   
Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hannaneh Hajishirzi, and Daniel Khashabi. 2022. When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories. arXiv preprint arXiv:2212.10511, 7.   
Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu,

Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, and 1 others. 2023. Augmented language models: a survey. arXiv preprint arXiv:2302.07842.   
Daye Nam, Andrew Macvean, Vincent Hellendoorn, Bogdan Vasilescu, and Brad Myers. 2024. Using an llm to help with code understanding. In Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, pages 1–13.   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.   
Shishir G Patil, Tianjun Zhang, Xin Wang, and Joseph E Gonzalez. 2023. Gorilla: Large language model connected with massive apis. arXiv preprint arXiv:2305.15334.   
Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H Miller, and Sebastian Riedel. 2019. Language models as knowledge bases? arXiv preprint arXiv:1909.01066.   
Reid Pryzant, Dan Iter, Jerry Li, Yin Tat Lee, Chenguang Zhu, and Michael Zeng. 2023. Automatic prompt optimization with" gradient descent" and beam search. arXiv preprint arXiv:2305.03495.   
Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, and 1 others. 2023. Toolllm: Facilitating large language models to master $1 6 0 0 0 +$ real-world apis. arXiv preprint arXiv:2307.16789.   
Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. 2024a. Colt: Towards completeness-oriented tool retrieval for large language models. arXiv preprint arXiv:2405.16089.   
Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. 2024b. From exploration to mastery: Enabling llms to master tools via self-driven interactions. arXiv preprint arXiv:2410.08197.   
Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. 2024c. Tool learning with large language models: A survey. arXiv preprint arXiv:2405.17935.   
Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.   
Laria Reynolds and Kyle McDonell. 2021. Prompt programming for large language models: Beyond the few-shot paradigm. In Extended abstracts of the 2021 CHI conference on human factors in computing systems, pages 1–7.

Stephen Robertson, Hugo Zaragoza, and 1 others. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389.   
Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551.   
Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. 2020. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. arXiv preprint arXiv:2010.15980.   
Karen Sparck Jones. 1972. A statistical interpretation of term specificity and its application in retrieval. Journal of documentation, 28(1):11–21.   
Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, Boxi Cao, and Le Sun. 2023. Toolalpaca: Generalized tool learning for language models with 3000 simulated cases. arXiv preprint arXiv:2306.05301.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.   
Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, and Heng Ji. 2023. Mint: Evaluating llms in multi-turn interaction with tools and language feedback. arXiv preprint arXiv:2309.10691.   
Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. 2022. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 139–149.   
Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808.   
Benfeng Xu, Licheng Zhang, Zhendong Mao, Quan Wang, Hongtao Xie, and Yongdong Zhang. 2020. Curriculum learning for natural language understanding. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6095–6104.   
Qiancheng Xu, Yongqi Li, Heming Xia, and Wenjie Li. 2024. Enhancing tool retrieval with iterative feedback from large language models. arXiv preprint arXiv:2406.17465.

Qiantong Xu, Fenglu Hong, Bo Li, Changran Hu, Zhengyu Chen, and Jian Zhang. 2023. On the tool manipulation capability of open-source large language models. arXiv preprint arXiv:2305.16504.   
Wujiang Xu, Kai Mei, Hang Gao, Juntao Tan, Zujie Liang, and Yongfeng Zhang. 2025. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110.   
Rui Yang, Lin Song, Yanwei Li, Sijie Zhao, Yixiao Ge, Xiu Li, and Ying Shan. 2024. Gpt4tools: Teaching large language model to use tools via self-instruction. Advances in Neural Information Processing Systems, 36.   
Lifan Yuan, Yangyi Chen, Xingyao Wang, Yi R Fung, Hao Peng, and Heng Ji. 2023. Craft: Customizing llms by creating and retrieving from specialized toolsets. arXiv preprint arXiv:2309.17428.   
Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Ren Kan, Dongsheng Li, and Deqing Yang. 2024. Easytool: Enhancing llm-based agents with concise tool instruction. arXiv preprint arXiv:2401.06201.   
Yuyue Zhao, Jiancan Wu, Xiang Wang, Wei Tang, Dingxian Wang, and Maarten de Rijke. 2024. Let me do it for you: Towards llm empowered recommendation via tool learning. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1796–1806.   
Yuanhang Zheng, Peng Li, Wei Liu, Yang Liu, Jian Luan, and Bin Wang. 2024. Toolrerank: Adaptive and hierarchy-aware reranking for tool retrieval. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 16263–16273.   
Mu Zhu. 2004. Recall, precision and average precision. Department of Statistics and Actuarial Science, University of Waterloo, Waterloo, 2(30):6.   
Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang. 2023. Toolqa: A dataset for llm question answering with external tools. Advances in Neural Information Processing Systems, 36:50117– 50143.

# Appendix

# A Problem Definition

Tool retrieval, as discussed in previous, involves generating a comprehensive list of tools that are potentially relevant to a user’s query. This approach emphasizes breadth, aiming to maximize the inclusion of pertinent tools. While effective in ensuring extensive coverage, tool retrieval often prioritizes recall over precision, resulting in the

inclusion of extraneous tools that may not be essential for the task at hand. Addressing this limitation, we propose a new optimization direction–Tool Recommendation–for LLMs. It aims to ensure that the recommended set of tools aligns closely with the ground-truth set of tools for a task, both in quantity and quality. Specifically, given a user query with a ground-truth toolset {A, B, C}, tool recommendation aims to identify precisely {A, B, C}, avoiding omissions or the inclusion of redundant tools. Here is the definition of the tool recommendation task:

Definition 1 Tool Recommendation: Given a comprehensive set of tools $T = \{ T _ { 1 } , T _ { 2 } , \dots , T _ { n } \}$ and a query $Q$ , let $T _ { g r o u n d } \subseteq T$ denote the ground truth toolset that fully satisfies $Q$ . The objective is to recommend a toolset $T _ { r e c o m m e n d } = \{ T _ { 1 } , T _ { 2 } , . . . , T _ { k } \}$ from $T$ such that $T _ { r e c o m m e n d } = T _ { g r o u n d }$ and the cardinality constraint $| T _ { r e c o m m e n d } | = | T _ { g r o u n d } |$ holds.

Achieving precision in tool recommendation is pivotal for enhancing the performance and reliability of LLMs. By minimizing the inclusion of irrelevant tools, LLMs can reduce computational overhead, streamline task execution, and improve the overall quality of responses. Addressing precised tool recommendation not only mitigates the drawbacks associated with broad tool retrieval but also paves the way for more sophisticated and usercentric LLM applications. This advancement is essential for deploying LLMs in environments where efficiency, accuracy, and user satisfaction are crucial.

# B Extended Related Work

Recommendation for LLMs. Recent research has explored a variety of recommendation techniques to enhance Large Language Models (LLMs), integrating capabilities across multiple dimensions. Data recommendation (Xu et al., 2020; Ouyang et al., 2022) is crucial for selecting relevant datasets to fine-tune models for specific domains, ensuring ongoing performance improvements. Memory recommendation (Borgeaud et al., 2022; Gao and Zhang, 2024a; Xu et al., 2025) facilitates the retrieval of relevant past experiences or interactions, improving continuity, consistency, and long-term context in multi-turn conversations. Knowledge base recommendation (Gao et al., 2023; Hu et al., 2023; Petroni et al., 2019; Lewis et al., 2020) enhances factual grounding by retrieving the most pertinent information from external sources, ensuring

that model outputs are accurate and up to date. Architecture recommendation (Elsken et al., 2019; Fedus et al., 2022) optimizes model performance by dynamically selecting the most appropriate model components or layers to activate for different tasks, thereby improving efficiency. Lastly, prompt recommendation (Shin et al., 2020; Reynolds and Mc-Donell, 2021; Li and Liang, 2021; Wang et al., 2022; Liu et al., 2023) guides LLMs in utilizing the most effective input prompts, thereby enhancing the quality of generated responses through optimized input-output interactions. Together, these recommendation techniques form a comprehensive framework that enhances the adaptability, efficiency, and task-specific performance of LLMs. However, there remains a lack of research on tool recommendation. In this work, we motivate to seek to provide a clear definition of tool recommendation and proposes an effective recommendation method. Additionally, new datasets and metrics are created to advance research in this area.

# C Validation of TRACC

Definition and Intuition. The TRACC metric is designed to evaluate whether a predicted tool set $B$ achieves both content sufficiency and size fidelity relative to the ground-truth tool set $A$ . It is defined as:

$$
\operatorname {T R A C C} = \left(\frac {| A \cap B |}{| A |}\right) \times \left(1 - \frac {\left| | B | - | A | \right|}{| A \cup B |}\right).
$$

The first term measures how accurately the predicted tools overlap with the ground-truth set, while the second term penalizes deviations in the number of predicted tools. TRACC thus reflects the pre-execution objective of recommending a minimal sufficient tool set that is both complete and non-redundant.

Limitations of Conventional Metrics. Traditional retrieval metrics such as Recall $@ \mathbf { K }$ and ${ \mathrm { N D C G } } \ @ { \mathrm { K } }$ are not well suited for this objective because they implicitly assume that $K = | A |$ is known in advance, whereas in tool recommendation the model must determine the set size dynamically. These metrics also fail to penalize redundancy once all relevant tools appear within the top-$K$ . Consequently, oversolved predictions—those including extra tools—can still obtain deceptively high Recall $@ \mathbf { K }$ or NDCG@K scores. Moreover, such rank-based metrics can be inflated by over-

predicting large sets in which correct tools are ranked near the top, regardless of overall precision.

Illustrative Example. Consider $| A | \ = \ 3$ and $A = \{ a , b , c \}$ . Three representative cases illustrate the difference:

• Exact Solving: $B = \{ a , b , c \} \Rightarrow \mathrm { R e c a l l } @ 3 =$ 1.00, NDCG@3 = 1.00, TRACC = 1.00.   
• Oversolving: $\begin{array} { r c l } { B } & { = } & { \{ a , b , c , x , y \} } \end{array}$ (with $a , b , c$ ranked highest) $\Rightarrow$ Recall@3 = 1.00, $\mathrm { N D C G } @ 3 \approx 1 . 0 0$ , $\mathrm { T R A C C } = 0 . 6 0$ , reflecting a penalty for over-selection $( + 2 )$ .   
• Undersolving: $B = \{ a , b \} \Rightarrow \operatorname { R e c a l l } @ 3 =$ 0.67, NDCG $\textcircled { a } 3 < 1 . 0 0$ , $\mathrm { T R A C C } \approx 0 . 4 4$ , penalizing both missing capability and reduced size.

This comparison highlights that TRACC uniquely integrates correctness and compactness, whereas Recall $@ \mathbf { K }$ and NDCG@K evaluate only partial ranking quality.

Empirical Validation. The distinction is also evident in empirical results. On the RecTools dataset, the variant w/o Stage-1 in Table.2 achieves NDCG $= 0 . 9 1 6$ but only $\mathrm { T R A C C } = 0 . 4 3 9$ , while the full PTR model obtains comparable NDCG (0.896) yet markedly higher TRACC (0.591). This indicates that TRACC captures precision-related errors that rank-based metrics overlook. Furthermore, Figure.4 shows that PTR yields a smaller average length difference $\left| \left| B \right| - \left| A \right| \right|$ , which directly corresponds to the size penalty term in TRACC and aligns with its design objective.

Summary. TRACC serves as an appropriate primary metric for evaluating precision-driven toolset recommendation, as it simultaneously encodes sufficiency and minimality within a single interpretable score. While Recall $@ \mathbf { K }$ and NDCG@K remain informative for assessing rank quality, they cannot detect the over- or under-sizing behaviors that critically influence the reliability, efficiency, and cost of downstream LLM reasoning.

# D Resource Consumption Analysis

We report the average resource consumption per query under the GPT-4o pricing model (USD 2.50 per million input tokens; USD 10.00 per million output tokens). The per-query cost is computed as:

$$
\mathrm {C o s t} = 0. 0 0 2 5 \times \frac {\mathrm {I n p u t}}{1 0 0 0} + 0. 0 1 \times \frac {\mathrm {O u t p u t}}{1 0 0 0}.
$$

Table 3: Statistics of the experimental datasets.   

<table><tr><td>Feature</td><td>ToolLens</td><td>MetaTool</td><td>RecTools</td></tr><tr><td>Tools per Query</td><td>1-3</td><td>2</td><td>1-10</td></tr><tr><td>Unified used tool number</td><td>✓</td><td>×</td><td>✓</td></tr><tr><td>Exact Solving Test</td><td>6.34%</td><td>55.1%</td><td>61.3%</td></tr></table>

ToolLens: $1 , 6 3 8 \mathrm { i n } + 1 3 9 \mathrm { o u t } \Rightarrow 1 . 6 3 8 \times 0 . 0 0 2 5 +$ $0 . 1 3 9 \times 0 . 0 1 = 0 . 0 0 5 4 8 5 \approx \mathrm { U S D } 0 . 0 0 5 5$

MetaTool: $2 , 0 3 6 \ \mathrm { i n } + 1 9 4 \ \mathrm { o u t } \Rightarrow 2 . 0 3 6 \times 0 . 0 0 2 5 +$ $0 . 1 9 4 \times 0 . 0 1 = 0 . 0 0 7 0 3 0 \approx \mathrm { U S D } 0 . 0 0 7 0$

RecTools: $2 , 6 5 7 \mathrm { i n } + 2 0 7 \mathrm { o u t } \Rightarrow 2 . 6 5 7 \times 0 . 0 0 2 5 +$ $0 . 2 0 7 \times 0 . 0 1 = 0 . 0 0 8 7 1 3 \approx \mathrm { U S D } 0 . 0 0 8 7$

Across datasets, the per-query cost ranges from about USD 0.0055 to USD 0.0087, corresponding to less than one cent per query, confirming that the proposed pipeline remains cost-efficient and scalable for large-scale tool recommendation experiments.

# E Details of RecTools

Both existed datasets impose a low upper limit on the number of tools used per query. As the capabilities of LLMs continue to develop, more tools need to be recommended to solve increasingly complex problems, thereby limiting the applicability of these datasets. Additionally, all queries in these two datasets utilize a fixed number of tools, which not only fails to fully simulate the dynamic nature of tool usage in real-world scenarios but also introduces bias in the subsequent testing of the method. Most importantly, since tool recommendation focuses on the precision of the recommended toolset, the test datasets require that each query be exactly solvable by the provided tools (Exact Solving). Using one fewer tool leads to partial solving, while using one additional tool results in oversolving. To validate the effectiveness of the two datasets, we first employ GPT-4o as an evaluator to determine whether the provided toolset can achieve an “Exact Solving” outcome for each query. Subsequently, for each query, we randomly remove one tool from the corresponding toolset and prompt GPT-4o to assess whether the modified toolset can achieve a “Partial Solving” outcome. Queries and their respective toolsets that meet the criteria for both evaluations are considered qualified. The performance of these two datasets is not ideal. Based on these limitations, we constructed a new dataset, RecTools, where queries do not have a uniform number of tools and have a high upper limit on

the number of tools used. Additionally, RecTools significantly outperforms ToolLens and Metatool in the GPT-4o “Exact Solving” test. The statistics of the three datasets are summarized in Table.3. Specifically, for all (query, tools) pairs involving the use of two and three tools, the success rates of RecTools reached $76 \%$ and $89 \%$ , respectively.

# E.1 Dataset Construction

To construct our dataset, we utilized tools from the MetaTool (Huang et al., 2023) dataset, along with their corresponding descriptions. Since their objective of tools was to address the issue of overlapping—where a single query could be resolved by multiple tools—MetaTool consolidates groups of tools with similar functionalities into a single tool entity. Besides, those tools and their description come from OpenAI’s plugin list, making them more practical. In our dataset RecTools, there are 10 usage scenarios in total (from 1-10), where the usage scenarios mean the quantitative classification, like two tools be used together, ten tools be used together. Each scenario of tools usage contains 100 examples. In each scenario, there are 20 different tool combinations. In terms of each combination, we randomly select from all possible combinations(i.e., ${ \bigl ( } _ { n } ^ { 1 } { \bigr ) } , { \bigl ( } _ { n } ^ { 2 } { \bigr ) } , . . . , { \bigl ( } _ { n } ^ { 1 0 } { \bigr ) } )$ . And for each tool combinations, we generate 5 queries. The prompt is as follows:

```txt
You are an assistant tasked with generating user queries that can be exclusively solved by a specific set of tools. 
```

```txt
**Requirements for the query:**  
1. The query must **only** require the functionalities of the selected tools.  
2. All tools in the selected set must be **necessary** to solve the query.  
3. The query should **not** require any tools outside the selected set.  
4. The query should be **clear, specific, and realistic**.  
5. **Each query should address a different scenario or aspect** to ensure uniqueness. Avoid merely rephrasing similar ideas; focus on varied use cases. 
```

```txt
**Selected Tools:**  
XX, XXX  
**Tool Descriptions:**  
- **XX**: Search for podcasts and summarize their content.  
- **XXX**: Discover and support restaurants, shops & services near you. 
```

Generate one unique query that meets the above requirements .

# E.2 Dataset Evaluation

To ensure precision in tool recommendation, it is crucial that the query is addressed entirely by the provided tools. If any tool is missing, the query cannot be fully solved, and if an unnecessary tool is included, the solution becomes redundant or repetitive. We employ GPT-4 as an evaluator to determine whether the provided toolset can achieve an "Exact Solving" outcome for each query. Subsequently, for each query, we randomly remove one tool from the corresponding toolset and prompt GPT-4 to assess whether the modified toolset can achieve a "Partial Solving" outcome. Queries and their respective toolsets that meet the criteria for both evaluations are considered qualified. For the first evaluation, if it achieves "Exact Solving", we give it a score 1, else 0; For the second evaluation, if it achieves "Partial Solving", we give it a score 1, else 0; For the final score, if both of them are 1, then 1; else, 0. The prompt is as follows:

```txt
Prompt1(Before deletion)  
**Query:** "XXX" 
```

```txt
\*\*Tools:\*\*  
- \*\*XX\*\*: xxxxxx  
- \*\*XX\*\*: xxxxxx  
- \*\*XX\*\*: xxxxxx 
```

```javascript
\*\*Classification: \*\* (.Categorize the solving scenario into one of the following: 1. \*\*Exact Solving: \*\* All functionalities are met by all tools without any redundancies. 2. \*\*Oversolving: \*\* The toolset includes tools that provide functionalities not required by the query. 3. \*\*Partial Solving: \*\* Some functionalities remain unfulfilled and some tools remain unused.) 
```

```javascript
Prompt2(Afterdeletion)   
\*\*Query:\*\* "XXX"   
\*\*Toolsafterremovingone tool:\*   
- \*\*XX\*:xxxxxxxxx   
- \*\*XX\*:xxxxxxxxx 
```

```txt
**Classification:** (.Categorize the solving scenario into one of the following:  
1. **Exact Solving:** All functionalities are met by all tools without any redundancies.  
2. **Oversolving:** The toolset includes tools that provide functionalities not 
```

The final output of evaluation is like this:   
```txt
required by the query.  
3. **Partial Solving:** Some functionalities remain unfulfilled and some tools remain unused.) 
```

```txt
{ "query":"XXX", "tools_used":[ XX", XX"] , "first evaluation": "xxx", "second evaluation_after_deletion : "xxx", "score":X   
}, 
```

Listing 1: An full example for evaluation   
```txt
Few-Shot Examples: 
```

```python
**Query:** "I need the latest weather forecast for New York and a reminder to carry an umbrella." 
```

```txt
\*\*Tools:\*\* 
```

```txt
- \*\*\*WeatherTool\*\*: Provide you with the latest weather information. 
```

```txt
- \*\*ReminderTool\*: No description available. 
```

```txt
**Classification:** Exact Solving 
```

```txt
**Query:**** "Show me the top-rated restaurants nearby and provide a route to get there." 
```

```txt
\*\*Tools:\*\* 
```

```txt
- \*\*RestaurantFinder\*\*: No description available. 
```

```txt
- **RoutePlanner**: No description available. 
```

```txt
**Classification:** Exact Solving 
```

```txt
**Query:** "Find me a good book to read and suggest a nearby coffee shop." 
```

```txt
\*\*Tools:\*\* 
```

```txt
- \*\*BookRecommender\*: No description available. 
```

```txt
- **WeatherTool**: Provide you with the latest weather information. 
```

```txt
**Classification:** ** Partial Solving 
```

```txt
**Query:**** "Provide the current exchange rates and set a reminder to check them later." 
```

```txt
\*\*Tools:\*\* 
```

```txt
- **FinanceTool**: Stay informed with the latest financial updates, real-time insights, and analysis on a wide range of options, stocks, cryptocurrencies, and more. 
```

```txt
- \*\*ReminderTool\*: No description available. 
```

```txt
- **NewsTool**: Stay connected to global events with our up-to-date news around the world. 
```

```javascript
\*\*Classification:\*\*Oversolving 
```

```txt
**Query:** "I want to track my fitness goals and get news updates." 
```

```txt
\*\*Tools:\*\* 
```

```txt
- **FitnessTracker**: No description available. 
```

```txt
- **NewsTool**: Stay connected to global events with our up-to-date news around the world. 
```

```txt
**Classification:** Exact Solving 
```

```txt
**Query:** "Schedule a meeting and find the latest sports news." 
```

```txt
\*\*Tools:\*\* 
```

```txt
- **CalendarTool**: No description available. 
```

```txt
- **NewsTool**: Stay connected to global events with our up-to-date news around the world. 
```

```txt
- **FinanceTool**: Stay informed with the latest financial updates, real-time insights, and analysis on a wide range of options, stocks, cryptocurrencies, and more. 
```

```javascript
\*\*Classification:\*\*Oversolving 
```

```txt
**Query:** "Research and select appropriate investment options for setting up a trust fund, ensure compliance with relevant laws, and find suitable gifts for beneficiaries to commemorate the establishment of the trust." 
```

```txt
**Tools:** 
```

```txt
- **FinanceTool**: Stay informed with the latest financial updates, real-time insights, and analysis on a wide range of options, stocks, cryptocurrencies, and more. 
```

```txt
- \*\*LawTool\*: Enables quick search functionality for relevant laws. 
```

```txt
- **GiftTool**: Provide suggestions for gift selection. 
```

```javascript
**Classification:** (Respond with only one of the following exact phrases: "Exact Solving", "Oversolving", or "Partial Solving". Do not include any additional text or explanations.) 
```

```txt
First Evaluation: Exact Solving 
```

```txt
Few-Shot Examples: 
```

```python
**Query:** "I need the latest weather forecast for New York and a reminder to carry an umbrella." 
```

```txt
**Tools:**  
- **WeatherTool**: Provide you with the latest weather information.  
- **ReminderTool**: No description available. 
```

```txt
**Classification:** Exact Solving 
```

```txt
**Query:**** "Show me the top-rated restaurants nearby and provide a route to get there." 
```

```txt
\*\*Tools:\*\* 
```

```txt
- \*\*RestaurantFinder\*\*: No description available. 
```

```txt
- \*\*RoutePlanner\*: No description available. 
```

```txt
**Classification:** Exact Solving 
```

```txt
**Query:** "Find me a good book to read and suggest a nearby coffee shop." 
```

```txt
\*\*Tools:\*\* 
```

```txt
- **BookRecommender**: No description available. 
```

```txt
- \*\*\*WeatherTool\*\*: Provide you with the latest weather information. 
```

```txt
**Classification:**\* Partial Solving 
```

```txt
**Query:** "Provide the current exchange rates and set a reminder to check them later." 
```

```txt
\*\*Tools:\*\* 
```

```txt
- **FinanceTool**: Stay informed with the latest financial updates, real-time insights, and analysis on a wide range of options, stocks, cryptocurrencies, and more. 
```

```txt
- \*\*ReminderTool\*: No description available. 
```

```txt
- **NewsTool**: Stay connected to global events with our up-to-date news around the world. 
```

```txt
**Classification:** Oversolving 
```

```txt
**Query:** "I want to track my fitness goals and get news updates." 
```

```txt
\*\*Tools:\*\* 
```

```txt
- **FitnessTracker**: No description available. 
```

```txt
- **NewsTool**: Stay connected to global events with our up-to-date news around the world. 
```

```txt
**Classification:** Exact Solving 
```

```txt
**Query:** "Schedule a meeting and find the latest sports news." 
```

```txt
\*\*Tools:\*\* 
```

```txt
- **CalendarTool**: No description available. 
```

```txt
- **NewsTool**: Stay connected to global events with our up-to-date news around the world. 
```

```txt
- **FinanceTool**: Stay informed with the latest financial updates, real-time insights, and analysis on a wide range of options, stocks, cryptocurrencies, and more. 
```

```javascript
\*\*Classification:\*\*Oversolving 
```

```txt
**Query:** "Research and select appropriate investment options for setting up a trust fund, ensure compliance with relevant laws, and find suitable gifts for beneficiaries to commemorate the establishment of the trust." 
```

```txt
**Tools after removing one tool:** - **FinanceTool**: Stay informed with the latest financial updates, real-time insights, and analysis on a wide range of options, stocks, cryptocurrencies, and more. 
```

```txt
- \*\*LawTool\*: Enables quick search functionality for relevant laws. 
```

```javascript
**Classification:** (Respond with only one of the following exact phrases: "Exact Solving", "Oversolving", or "Partial Solving". Do not include any additional text or explanations.) 
```

```txt
Second Evaluation (After Deletion): Partial Solving Score for this query: 1 
```

```javascript
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* 
```

```javascript
\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\*\* 
```

```txt
{ 
```

```jsonl
"query": "Research and select appropriate investment options for setting up a trust fund, ensure compliance with relevant laws, and find suitable gifts for beneficiaries to commemorate the establishment of the trust.", "tools_used": [ "FinanceTool", "LawTool", "GiftTool"] , "first.evaluation": "Exact Solving ", "second.evaluation_after_deletion": "Partial Solving", "score": 1 } 
```

# F Functional Coverage Mapping

# F.1 Extraction of Key Functionalities

You are an assistant helping to extract key requirements from user queries .

```txt
Example 1:User Query 
```

" I want a website where users can create accounts , post messages , and follow other users ."

Key Requirements :

- Users can create accounts   
- Users can post messages   
- Users can follow other users

Example 2:

User Query :

" I need an e - commerce platform that supports product listings , shopping cart functionality , payment processing , and order tracking ."

Key Requirements :

- Supports product listings   
- Provides shopping cart functionality   
- Handles payment processing   
- Offers order tracking

Now , given the following user query , extract the key requirements .

User Query :

XXX

Key Requirements :

# F.2 Matching Tools to Functionalities

You are an assistant helping to match tools to requirements , as long as the tool description can roughly provid the needed information for requirments , it does not need to be very specific , ignore the proper nouns .

Available Tools : XX : xxxxx ; XX : xxxxxx .

Example 1:

Requirement :

" I want to know the latest news about Tesla "

Matched Tools :

- NewsTool : Stay connected to global events with our up - to - date news around the world .

Example 2:

Requirement :

" Please provide me with the current stock price of Apple "

Matched Tools :

- FinanceTool : Stay informed with the latest financial updates , real - time insights , and analysis on a wide range of options , stocks , cryptocurrencies , and more .

Now , for the following requirement , list the tools from the available tools that can fulfill it .

Requirement :

XXX

XXX

XXX

Matched Tools :

# F.3 Examples

# Listing 2: An example in ToolLens

You are an assistant helping to extract key requirements from user queries .

Example 1:

User Query :

" I want a website where users can create accounts , post messages , and follow other users ."

Key Requirements :

- Users can create accounts   
- Users can post messages   
- Users can follow other users

Example 2:

User Query :

" I need an e - commerce platform that supports product listings , shopping cart functionality , payment processing , and order tracking ."

Key Requirements :

- Supports product listings   
- Provides shopping cart functionality   
- Handles payment processing   
- Offers order tracking

Now , given the following user query , extract the key requirements .

User Query :

"I'm preparing for a marathon in Paris , France .

Key Requirements :

- Marathon preparation

- Location : Paris , France

****************************

****************************

You are an assistant helping to match tools to requirements , as long as the tool description can roughly provid the needed information for requirments , it does not need to be very specific , ignore the proper nouns .

Available Tools :

- ** Countries **: This gets geo data on a country . Use ISO2 for country_code .

- ** Skyscanner_v2 **: Search for a place to get the ** entityId ** needed in searching the hotel API .

- ** TimeTable Lookup **: Returns the nearest airports for a given latitude and longitude

Example 1:

Requirement :

" I want to know the latest news about Tesla "

Matched Tools :

- NewsTool : Stay connected to global events with our up - to - date news around the world .

Example 2:

Requirement :

" Please provide me with the current stock price of Apple "

Matched Tools :

- FinanceTool : Stay informed with the latest financial updates , real - time insights , and analysis on a wide range of options , stocks , cryptocurrencies , and more .

Now , for the following requirement , list the tools from the available tools that can fulfill it .

Requirement :

" Marathon preparation "

Matched Tools :

You are an assistant helping to match tools to requirements , as long as the tool description can roughly provid the needed information for requirments , it does not need to be very specific , ignore the proper nouns .

Available Tools :

- ** Countries **: This gets geo data on a country . Use ISO2 for country_code .

- ** Skyscanner_v2 **: Search for a place to get the ** entityId ** needed in searching the hotel API .

- ** TimeTable Lookup **: Returns the nearest airports for a given latitude and longitude

Example 1:

Requirement :

" I want to know the latest news about Tesla "

Matched Tools :

- NewsTool : Stay connected to global events with our up - to - date news around the world .

Example 2:

Requirement :

" Please provide me with the current stock price of Apple "

Matched Tools :

- FinanceTool : Stay informed with the latest financial updates , real - time insights , and analysis on a wide range of options , stocks , cryptocurrencies , and more .

Now , for the following requirement , list

the tools from the available tools that can fulfill it .

Requirement :

" Location : Paris , France "

Matched Tools :

Tool Matches :

- Requirement : 'Marathon preparation ' matched with Tools : None

- Requirement : 'Location : Paris , France ' matched with Tools : None

Does the toolset exactly solve the query ? No

Tools to Keep :

Unsolved Problems :

- Marathon preparation

- Location : Paris , France

# Listing 3: An example in MetaTool

You are an assistant helping to extract key requirements from user queries .

Example 1:

User Query :

" I want a website where users can create accounts , post messages , and follow other users ."

Key Requirements :

- Users can create accounts

- Users can post messages

- Users can follow other users

Example 2:

User Query :

" I need an e - commerce platform that supports product listings , shopping cart functionality , payment processing , and order tracking ."

Key Requirements :

- Supports product listings

- Provides shopping cart functionality

- Handles payment processing

- Offers order tracking

Now , given the following user query , extract the key requirements .

User Query :

"I'm looking for a family - friendly destination in Europe with good weather . Can you suggest some options and what the weather will be like during summer ?"

Key Requirements Extracted :

- Family - friendly destination in Europe

- Options about Europe

- Information on weather during summer

****************************

****************************

You are an assistant helping to match tools to requirements , as long as the

tool description can roughly provid the needed information for requirments , it does not need to be very specific , ignore the proper nouns .

# Available Tools :

- ** ResearchFinder **: Tool for searching academic papers .

- ** WeatherTool **: Provide you with the latest weather information .

# Example 1:

Requirement :

" I want to know the latest news about Tesla "

# Matched Tools :

- NewsTool : Stay connected to global events with our up - to - date news around the world .

# Example 2:

Requirement :

" Please provide me with the current stock price of Apple "

# Matched Tools :

- FinanceTool : Stay informed with the latest financial updates , real - time insights , and analysis on a wide range of options , stocks , cryptocurrencies , and more .

Now , for the following requirement , list the tools from the available tools that can fulfill it .

# Requirement :

" Family - friendly destination in Europe "

Matched Tools :

You are an AI assistant helping to match tools to requirements , as long as the tool description can roughly provid the needed information for requirments , it does not need to be very specific , ignore the proper nouns .

# Available Tools :

- ** ResearchFinder **: Tool for searching academic papers .

- ** WeatherTool **: Provide you with the latest weather information .

# Example 1:

Requirement :

" I want to know the latest news about Tesla "

# Matched Tools :

- NewsTool : Stay connected to global events with our up - to - date news around the world .

# Example 2:

Requirement :

" Please provide me with the current stock price of Apple "

# Matched Tools :

- FinanceTool : Stay informed with the latest financial updates , real - time insights , and analysis on a wide range of options , stocks , cryptocurrencies , and more .

Now , for the following requirement , list the tools from the available tools that can fulfill it .

# Requirement :

" Options about Europe "

Matched Tools :

You are an AI assistant helping to match tools to requirements , as long as the tool description can roughly provid the needed information for requirments , it does not need to be very specific , ignore the proper nouns .

# Available Tools :

- ** ResearchFinder **: Tool for searching academic papers .

- ** WeatherTool **: Provide you with the latest weather information .

# Example 1:

Requirement :

" I want to know the latest news about Tesla "

# Matched Tools :

- NewsTool : Stay connected to global events with our up - to - date news around the world .

# Example 2:

Requirement :

" Please provide me with the current stock price of Apple "

# Matched Tools :

- FinanceTool : Stay informed with the latest financial updates , real - time insights , and analysis on a wide range of options , stocks , cryptocurrencies , and more .

Now , for the following requirement , list the tools from the available tools that can fulfill it .

# Requirement :

" Information on weather during summer "

# Matched Tools :

WeatherTool : Provide you with the latest weather information .

# Tool Matches :

- Requirement : 'Family - friendly destination in Europe ' matched with Tools : None

- Requirement : 'Good weather ' matched with Tools : None

- Requirement : ' Information on weather during summer ' matched with Tools : WeatherTool

Does the toolset exactly solve the query ? N o Tools to Keep : WeatherTool

Unsolved Problems :

- Family - friendly destination in Europe   
- Options about Europe   
- Information on weather during summer

# Listing 4: An example in RecTools

You are an assistant helping to extract key requirements from user queries .

Example 1:

User Query :

" I want a website where users can create accounts , post messages , and follow other users ."

Key Requirements :

- Users can create accounts   
- Users can post messages   
- Users can follow other users

Example 2:

User Query :

" I need an e - commerce platform that supports product listings , shopping cart functionality , payment processing , and order tracking ."

Key Requirements :

- Supports product listings   
- Provides shopping cart functionality   
- Handles payment processing   
- Offers order tracking

Now , given the following user query , extract the key requirements .

User Query :

" I want to find a local restaurant with a menu that fits my diet plan , book a table , get astrology insights on the best date for my dinner , and select a thoughtful gift for my dining companion

Key Requirements Extracted :

- Find a local restaurant   
- Provide a menu that fits the user 's diet plan   
- Book a table   
- Offer astrology insights on the best date for dinner   
- Select a thoughtful gift for the dining companion

****************************

****************************

You are an assistant helping to match tools to requirements , as long as the tool description can roughly provid the

needed information for requirments , it does not need to be very specific , ignore the proper nouns .

Available Tools :

- ** DietTool **: A tool that simplifies calorie counting , tracks diet , and provides insights from many restaurants and grocery stores . Explore recipe , menus , and cooking tips from millions of users , and access recipe consultations and ingredient delivery services from thousands of stores .

- ** GiftTool **: Provide suggestions for gift selection .

- ** HousePurchasingTool **: Tool that provide all sorts of information about house purchasing

- ** HouseRentingTool **: Tool that provide all sorts of information about house renting

- ** MemoryTool **: A learning application with spaced repetition functionality that allows users to create flashcards and review them .

- ** RestaurantBookingTool $\star \star$ : Tool for booking restaurant

- ** ResumeTool **: Quickly create resumes and receive feedback on your resume .

- ** StrologyTool **: Povides strology services for you .

- ** local **: Discover and support

restaurants , shops & services near you .

Example 1:

Requirement :

" I want to know the latest news about Tesla "

Matched Tools :

- NewsTool : Stay connected to global events with our up - to - date news around the world .

Example 2:

Requirement :

" Please provide me with the current stock price of Apple "

Matched Tools :

- FinanceTool : Stay informed with the latest financial updates , real - time insights , and analysis on a wide range of options , stocks , cryptocurrencies , and more .

Now , for the following requirement , list the tools from the available tools that can fulfill it .

Requirement :

" Find a local restaurant "

Matched Tools :

You are an assistant helping to match tools to requirements , as long as the tool description can roughly provid the needed information for requirments , it does not need to be very specific , ignore

the proper nouns .

Available Tools :

- ** DietTool **: A tool that simplifies calorie counting , tracks diet , and provides insights from many restaurants and grocery stores . Explore recipe , menus , and cooking tips from millions of users , and access recipe consultations and ingredient delivery services from thousands of stores .

- ** GiftTool **: Provide suggestions for gift selection .

- ** HousePurchasingTool **: Tool that provide all sorts of information about house purchasing

- ** HouseRentingTool **: Tool that provide all sorts of information about house renting

- ** MemoryTool **: A learning application with spaced repetition functionality that allows users to create flashcards and review them .

- $\star \star$ RestaurantBookingTool $\star \star$ : Tool for booking restaurant

- ** ResumeTool $\star \star$ : Quickly create resumes and receive feedback on your resume . ** StrologyTool **: Povides strology services for you .

- ** local **: Discover and support restaurants , shops & services near you .

Example 1:

Requirement :

" I want to know the latest news about Tesla "

Matched Tools :

- NewsTool : Stay connected to global events with our up - to - date news around the world .

Example 2:

Requirement :

" Please provide me with the current stock price of Apple "

Matched Tools :

- FinanceTool : Stay informed with the latest financial updates , real - time insights , and analysis on a wide range of options , stocks , cryptocurrencies , and more .

Now , for the following requirement , list the tools from the available tools that can fulfill it .

Requirement :

" Provide a menu that fits the user 's diet plan "

Matched Tools :

DietTool : A tool that simplifies calorie counting , tracks diet , and provides insights from many restaurants and grocery stores . Explore recipe , menus , and cooking tips from millions of users , and access recipe consultations and ingredient delivery services from

thousands of stores .

You are an assistant helping to match tools to requirements , as long as the tool description can roughly provid the needed information for requirments , it does not need to be very specific , ignore the proper nouns .

Available Tools :

- ** DietTool **: A tool that simplifies calorie counting , tracks diet , and provides insights from many restaurants and grocery stores . Explore recipe , menus , and cooking tips from millions of users , and access recipe consultations and ingredient delivery services from thousands of stores .

- ** GiftTool **: Provide suggestions for gift selection .

- ** HousePurchasingTool **: Tool that provide all sorts of information about house purchasing

- ** HouseRentingTool **: Tool that provide all sorts of information about house renting

- ** MemoryTool **: A learning application with spaced repetition functionality that allows users to create flashcards and review them .

- ** RestaurantBookingTool $\star \star$ : Tool for booking restaurant

- ** ResumeTool **: Quickly create resumes and receive feedback on your resume .

- ** StrologyTool **: Povides strology services for you .

- ** local **: Discover and support restaurants , shops & services near you .

Example 1:

Requirement :

" I want to know the latest news about Tesla "

Matched Tools :

- NewsTool : Stay connected to global events with our up - to - date news around the world .

Example 2:

Requirement :

" Please provide me with the current stock price of Apple "

Matched Tools :

- FinanceTool : Stay informed with the latest financial updates , real - time insights , and analysis on a wide range of options , stocks , cryptocurrencies , and more .

Now , for the following requirement , list the tools from the available tools that can fulfill it .

Requirement :

" Book a table "

Matched Tools :

You are an AI assistant helping to match tools to requirements , as long as the tool description can roughly provid the needed information for requirments , it does not need to be very specific , ignore the proper nouns .

# Available Tools :

- ** DietTool **: A tool that simplifies calorie counting , tracks diet , and provides insights from many restaurants and grocery stores . Explore recipe , menus , and cooking tips from millions of users , and access recipe consultations and ingredient delivery services from thousands of stores .

- ** GiftTool **: Provide suggestions for gift selection .

- ** HousePurchasingTool **: Tool that provide all sorts of information about house purchasing

- ** HouseRentingTool **: Tool that provide all sorts of information about house renting

- ** MemoryTool $\star \star$ : A learning application with spaced repetition functionality that allows users to create flashcards and review them .

- ** RestaurantBookingTool **: Tool for booking restaurant

- ** ResumeTool **: Quickly create resumes and receive feedback on your resume .

- ** StrologyTool **: Povides strology services for you .

- ** local **: Discover and support restaurants , shops & services near you .

# Example 1:

# Requirement :

" I want to know the latest news about Tesla "

# Matched Tools :

- NewsTool : Stay connected to global events with our up - to - date news around the world .

# Example 2:

# Requirement :

" Please provide me with the current stock price of Apple "

# Matched Tools :

- FinanceTool : Stay informed with the latest financial updates , real - time insights , and analysis on a wide range of options , stocks , cryptocurrencies , and more .

Now , for the following requirement , list the tools from the available tools that can fulfill it .

# Requirement :

" Offer astrology insights on the best date for dinner "

# Matched Tools :

StrologyTool : Povides strology services

for you .

You are an AI assistant helping to match tools to requirements , as long as the tool description can roughly provid the needed information for requirments , it does not need to be very specific , ignore the proper nouns .

# Available Tools :

- ** DietTool **: A tool that simplifies calorie counting , tracks diet , and provides insights from many restaurants and grocery stores . Explore recipe , menus , and cooking tips from millions of users , and access recipe consultations and ingredient delivery services from thousands of stores .

- ** GiftTool **: Provide suggestions for gift selection .

- ** HousePurchasingTool **: Tool that provide all sorts of information about house purchasing

- ** HouseRentingTool **: Tool that provide all sorts of information about house renting

- ** MemoryTool **: A learning application with spaced repetition functionality that allows users to create flashcards and review them .

- ** RestaurantBookingTool $\star \star$ : Tool for booking restaurant

- ** ResumeTool **: Quickly create resumes and receive feedback on your resume .

- ** StrologyTool **: Povides strology services for you .

- ** local **: Discover and support restaurants , shops & services near you .

# Example 1:

# Requirement :

" I want to know the latest news about Tesla "

# Matched Tools :

- NewsTool : Stay connected to global events with our up - to - date news around the world .

# Example 2:

# Requirement :

" Please provide me with the current stock price of Apple "

# Matched Tools :

- FinanceTool : Stay informed with the latest financial updates , real - time insights , and analysis on a wide range of options , stocks , cryptocurrencies , and more .

Now , for the following requirement , list the tools from the available tools that can fulfill it .

# Requirement :

" Select a thoughtful gift for the dining companion "

# Matched Tools :

GiftTool : Provide suggestions for gift selection .

# Tool Matches :

- Requirement : 'Find a local restaurant ' matched with Tools : None

- Requirement : 'Provide a menu that fits the user 's diet plan ' matched with Tools : DietTool

- Requirement : 'Book a table ' matched with Tools : None

- Requirement : 'Offer astrology insights on the best date for dinner ' matched with Tools : StrologyTool

- Requirement : 'Select a thoughtful gift for the dining companion ' matched with Tools : GiftTool

Does the toolset exactly solve the query ? No

Tools to Keep : DietTool , StrologyTool , GiftTool

Unsolved Problems :

- Find a local restaurant

- Book a table
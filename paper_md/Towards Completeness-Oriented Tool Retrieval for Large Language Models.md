# Towards Completeness-Oriented Tool Retrieval for Large Language Models

Changle Qu

Sunhao Dai

Gaoling School of Artificial Intelligence

Renmin University of China

Beijing, China

{changlequ,sunhaodai}@ruc.edu.cn

Xiaochi Wei

Baidu Inc.

Beijing, China

weixiaochi@baidu.com

Hengyi Cai

Institute of Computing Technology

Chinese Academy of Sciences

Beijing, China

caihengyi@ict.ac.cn

Shuaiqiang Wang

Baidu Inc.

Beijing, China

shqiang.wang@gmail.com

Dawei Yin

Baidu Inc.

Beijing, China

yindawei@acm.org

Jun Xu∗

Ji-Rong Wen

Gaoling School of Artificial Intelligence

Renmin University of China

Beijing, China

{junxu,jrwen}@ruc.edu.cn

# ABSTRACT

Recently, integrating external tools with Large Language Models (LLMs) has gained significant attention as an effective strategy to mitigate the limitations inherent in their pre-training data. However, real-world systems often incorporate a wide array of tools, making it impractical to input all tools into LLMs due to length limitations and latency constraints. Therefore, to fully exploit the potential of tool-augmented LLMs, it is crucial to develop an effective tool retrieval system. Existing tool retrieval methods primarily focus on semantic matching between user queries and tool descriptions, frequently leading to the retrieval of redundant, similar tools. Consequently, these methods fail to provide a complete set of diverse tools necessary for addressing the multifaceted problems encountered by LLMs. In this paper, we propose a novel modelagnostic COllaborative Learning-based Tool Retrieval approach, COLT, which captures not only the semantic similarities between user queries and tool descriptions but also takes into account the collaborative information of tools. Specifically, we first fine-tune the PLM-based retrieval models to capture the semantic relationships between queries and tools in the semantic learning stage. Subsequently, we construct three bipartite graphs among queries, scenes, and tools and introduce a dual-view graph collaborative learning framework to capture the intricate collaborative relationships among tools during the collaborative learning stage. Extensive experiments on both the open benchmark and the newly introduced ToolLens dataset show that COLT achieves superior performance.

∗Jun Xu is the corresponding author. Work partially done at Engineering Research Center of Next-Generation Intelligent Search and Recommendation, Ministry of Education.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

CIKM ’24, October 21–25, 2024, Boise, ID, USA.

$\circledcirc$ 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 979-8-4007-0436-9/24/10

https://doi.org/10.1145/3627673.3679847

Notably, the performance of BERT-mini (11M) with our proposed model framework outperforms BERT-large (340M), which has 30 times more parameters. Furthermore, we will release ToolLens publicly to facilitate future research on tool retrieval.

# CCS CONCEPTS

• Information systems Information retrieval.

# KEYWORDS

Tool Retrieval, Retrieval Completeness, Large Language Model

# ACM Reference Format:

Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. 2024. Towards Completeness-Oriented Tool Retrieval for Large Language Models. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management (CIKM ’24), October 21–25, 2024, Boise, ID, USA. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3627673.3679847

# 1 INTRODUCTION

Recently, large language models (LLMs) have demonstrated remarkable progress across various natural language processing tasks [1, 2, 4, 41]. However, they often struggle with solving highly complex problems and providing up-to-date knowledge due to the constraints of their pre-training data [23, 42]. A promising approach to overcome these limitations is tool learning [18, 26, 29, 31, 48], which enables LLMs to dynamically interact with external tools, significantly facilitating access to real-time data and the execution of complex computations. By integrating tool learning, LLMs transcend the confines of their outdated or limited pre-trained knowledge [2], offering responses to user queries with significantly improved accuracy and relevance [14, 28]. However, real-world systems typically involve a large number of tools, making it impractical to take the descriptions of all tools as input for LLMs due to length limitations and latency constraints. Thus, as illustrated in Figure 1(a), developing an effective tool retrieval system becomes essential to fully exploit the potential of tool-augmented LLMs [8].

![](images/c43f76db68fef6a9ebb4603998379d4e72408dede1e3649314d595460f32dac8.jpg)  
(a) Pipeline of user interaction with tool-augmented LLMs.

![](images/acaf754333d06ac6e06fa260312babfe5c2aadbc7d821114618a423407c70e93.jpg)  
(b) Illustration of different responses with different tools.   
Figure 1: An illustration of tool retrieval for LLMs with tool learning and varied responses using different tools.

Typically, existing tool retrieval approaches directly employ dense retrieval techniques [8, 28, 49], solely focusing on matching semantic similarities between queries and tool descriptions. However, these approaches may fall short when addressing multifaceted queries that require a collaborative effort from multiple tools to formulate a comprehensive response. For instance, in Figure 1(b), consider a user’s request to calculate the value of 5 ounces of gold plus 1 million AMZN stocks in CNY. Such a query requires the simultaneous use of tools for gold prices, stock values, and currency exchange rates. The absence of any of these tools yields an incomplete answer. In this example, dense retrieval methods that rely solely on semantic matching may retrieve multiple tools related to stock prices while neglecting others. This highlights a significant limitation of dense retrieval methods that overlook the necessity for tools to interact collaboratively. Thus, ensuring the completeness of retrieved tools is an essential aspect of a tool retrieval system, which is often neglected by traditional retrieval approaches.

Toward this end, this paper proposes COLT, a novel modelagnostic COllaborative Learning-based Tool retrieval approach aimed at enhancing completeness-oriented tool retrieval. This method is structured into two main stages: semantic learning and collaborative learning. Initially, we fine-tune traditional pre-trained language models (PLMs) on tool retrieval datasets to acquire semantic matching information between queries and tools, thereby addressing the potential performance issues of these models in zero-shot scenarios for tool retrieval tasks. Subsequently, to capture the intricate collaborative relationship among tools, a concept of “scene” is proposed to indicate a group of collaborative tools. Based on this, COLT integrates three bipartite graphs among queries,

scenes, and tools. More specifically, given the initial semantic embedding from the semantic learning stage, the high-order collaborative relationship is better integrated via message propagation and cross-view graph contrastive learning among these graphs. The learning objective incorporates a list-wise multi-label loss to ensure the simultaneous acquisition of tools from the entire ground-truth set without favoring any specific tool.

Moreover, traditional retrieval metrics like Recall [52] and NDCG [16] fail to capture the completeness necessary for effective tool retrieval. As illustrated in Figure 1(b), the exclusion of any essential tool from the ground-truth tool set compromises the ability to fully address user queries, indicating that metrics focused solely on individual tool ranking performance are inadequate when multiple tools are required. To bridge this gap, we introduce COMP@??, a new metric designed to assess tool retrieval performance based on completeness, which can serve as a reliable indicator of how well a tool retrieval system performs for downstream tool learning applications. Additionally, we construct a new dataset called Tool-Lens, in which a query is typically solved with multiple relevant but diverse tools, reflecting the multifaceted nature of user requests in real-world scenarios.

In summary, our main contributions are as follows:

• The collaborative relationships among multiple tools in LLMs have been thoroughly studied, which reveals that incomplete tool retrieval hinders accurate answers, underscoring the integral role each tool plays in the collective functionality.   
• We introduce COLT, a novel tool retrieval approach that uses message propagation and cross-view graph contrastive learning among queries, scenes, and tools, incorporating better collaborative information among various tools.   
• Extensive experiments demonstrate the superior performance of COLT against state-of-the-art dense retrieval methods in both tool retrieval and downstream tool learning.   
• We introduce a new dataset and a novel evaluation metric specifically designed for assessing multi-tool usage in LLMs, which will facilitate future research on tool retrieval.

# 2 RELATED WORK

Tool Learning. Recent studies highlight the potential of LLMs to utilize tools in addressing complex problems [24, 27]. Existing tool learning approaches can be categorized into two types: tuningfree and tuning-based methods [8]. Tuning-free methods capitalize on the in context learning ability of LLMs through strategic prompting [32, 33, 43, 47]. For example, ART [25] constructs a task library, from which it retrieves demonstration examples as few-shot prompts when encountering real-world tasks. Conversely, tuningbased methods involve directly fine-tuning the parameters of LLMs on specific tool datasets to master tool usage. For example, ToolL-LaMA [28] employs the instruction-solution pairs derived from the DFSDT method to fine-tune the LLaMA model, thereby significantly enhancing its tool usage capabilities. Despite these advancements, most strategies either provide a manual tool set [31, 38, 46] or employ simple dense retrieval [8] for tool retrieval. However, LLMs must choose several useful tools from a vast array of tools in realworld applications, necessitating a robust tool retriever to address the length limitations and latency constraints of LLMs.

Tool Retrieval. Tool retrieval aims to find top- $K$ most suitable tools for a given query from a vast set of tools. Existing tool retrieval methods typically directly adopt traditional retrieval approaches, and state-of-the-art retrieval methods can be categorized into two types: term-based and semantic-based. Term-based methods, such as TF-IDF [35] and BM25 [30], prioritize term matching via sparse representations. Conversely, semantic-based methods, such as ANCE [45], TAS-B [12], coCondensor [7], and Contriever [15], utilize neural networks to learn the semantic relationship between queries and tool descriptions and then calculate the semantic similarity using methods such as cosine similarity. Despite these advancements, existing methods for tool retrieval overlook the importance of the collaborative relationship among multiple tools, thereby falling short of meeting the completeness criterion for tool retrieval. Our work seeks to mitigate these issues by collaborative learning that leverages graph neural networks and cross-view contrastive learning among graphs.

# 3 OUR APPROACH: COLT

In this section, we first introduce task formulation of tool retrieval. Then we describe the details of the proposed COLT approach.

# 3.1 Task Formulation

Formally, given a user query $q \in \mathcal { Q }$ , the goal of tool retrieval is to filter out the top- $K$ most suitable tools $\bar { \{ t ^ { ( 1 ) } , t ^ { ( 2 ) } , \ldots , t ^ { ( K ) } \} }$ from the entire tool set $\mathcal { T } = \{ ( t _ { 1 } , d _ { 1 } ) , ( t _ { 2 } , d _ { 2 } ) , \dots , ( t _ { N } , d _ { N } ) \}$ , where each element represents a specific tool $t _ { i }$ associated with its description $d _ { i }$ , and $N$ is the number of tools in the tool set.

Goal. As discussed in Section 1, the comprehensiveness of the tools retrieved is crucial for LLMs to enhance their ability to accurately address multifaceted and real-time questions. Therefore, it is necessary to ensure that the retrieved tools encompass all the tools required by the user question. Considering these factors, the goal of tool retrieval is to optimize both accuracy and completeness, ensuring the provision of desired tools for downstream tasks.

# 3.2 Overview of COLT

As illustrated in Figure 2, COLT employs a two-stage learning strategy, which includes semantic learning followed by collaborative learning. In the first stage, the semantic learning module processes both queries and tools to derive their semantic representations, aiming to align these representations closely within the semantic space. Subsequently, the collaborative learning module enhances these preliminary representations by introducing three bipartite graphs among queries, scenes, and tools. Through dual-view graph contrastive learning within these three bipartite graphs, COLT is able to capture the high-order collaborative information between tools. Furthermore, a list-wise multi-label loss is utilized in the learning objective to facilitate the balanced retrieval of diverse tools from the complete ground-truth set, avoiding undue emphasis on any specific tool.

In the following sections, we will present the details of these two key learning stages in COLT.

# 3.3 Semantic Learning

As shown in Figure 2 (a), in the first stage of COLT, we adopt the established dense retrieval (DR) framework [9, 50], leveraging pretrained language models (PLMs) such as BERT [17] to encode both the query $q$ and the tool ?? into low dimensional vectors. Specifically, we employ a bi-encoder architecture, with the cosine similarity between the encoded vectors serving as the initial relevance score:

$$
\widehat {y} _ {\mathrm {S L}} (q, t) = \operatorname {s i m} \left(\mathbf {e} _ {q}, \mathbf {e} _ {t}\right), \tag {1}
$$

where $\mathbf { e } _ { q }$ and $\mathbf { e } _ { t }$ are the mean pooling vectors from the final layer of the PLM, and $s \mathrm { i m } ( \cdot , \cdot )$ represents the cosine similarity function.

For training, we utilize the InfoNCE loss [10, 45], a standard contrastive learning technique used in training DR models, which contrasts positive pairs against negative ones. Specifically, given a query $q$ , its relevant tool $t ^ { + }$ and the set of irrelevant tools $\{ t _ { 1 } ^ { - } , \cdots , t _ { k } ^ { - } \}$ , we minimize the following loss:

$$
- \log \frac {e ^ {\operatorname {s i m} (q , t ^ {+})}}{e ^ {\operatorname {s i m} (q , t ^ {+})} + \sum_ {j = 1} ^ {k} e ^ {\operatorname {s i m} (q , t _ {j} ^ {-})}}. \tag {2}
$$

Through this loss function, we can increase the similarity score between the query and its relevant tool while decreasing the similarity scores between the query and irrelevant tools.

This semantic learning phase ensures good representations for each query and tool from the text description view. However, relying solely on semantic-based retrieval is insufficient for completeness-oriented tool retrieval, as it often falls short in addressing multifaceted queries effectively.

# 3.4 Collaborative Learning

3.4.1 Bipartite Graphs in Tool Retrieval. To capture the collaborative information between tools and achieve completeness-oriented tool retrieval, we first formulate the relationship between queries and tools with three bipartite graphs. Specifically, as illustrated in Figure 2 (b), we conceptualize the ground-truth tool set for each query as a “scene”, considering that a collaborative operation of multiple tools is essential to fully address multifaceted queries. For example, given the query $^ { \mathrm { { * } } } \mathrm { { I } }$ want to travel to Paris.”, it doesn’t merely seek a single piece of information but initiates a “scene” of travel planning, which involves using various tools for transportation, weather forecasts, accommodation choices, and details about attractions. This scenario underscores the need for scene matching beyond traditional semantic search or recommendation scenarios, where the focus is on selecting any relevant documents or items without considering their collaborative utility. As a result, traditional semantic-based retrieval systems may only retrieve tools related to Paris attractions, thus failing to provide a comprehensive and complete tool set for the LLMs. Conversely, we construct three bipartite graphs linking queries, scenes, and tools, i.e., Q-S (Query-Scene) graph, Q-T (Query-Tool) graph, and S-T (Scene-Tool) graph. By formulating these three graphs, we can further capture the high-order relationships among tools with graph learning, facilitating a scene-based understanding that aligns to achieve completeness-oriented tool retrieval.

3.4.2 Dual-view Graph Collaborative Learning. Leveraging the initial query and tool representations derived from the first-stage semantic learning, along with the three constructed bipartite graphs,

![](images/fa25a66a5c6ace3a514b6a7628bd9d680db9a9046bd195b1c3ce72a8b7e31b59.jpg)

![](images/4db00a026c9429feddab9e6f07494a17cd41770c08f78d76a28977f1939fee2c.jpg)

![](images/f17c806bb46b9e8fef63b2dd37f3e2c87f76b483ccbdb6b0c88ca9ebb6ead12c.jpg)  
Figure 2: The architecture of the proposed two-stage learning framework COLT for tool retrieval.

we introduce a dual-view graph collaborative learning framework. This framework is designed to capture the relationships between tools, as depicted in Figure 2 (c). It assesses the relevance between queries and tools from two views:

• Scene-centric View: Through the Q-S graph and S-T graph, this view captures the relevance between queries and tools mediated by a scene. This offers a nuanced view that considers the collaborative context in which tools work together to meet the requirements of a query.

• Tool-centric View: Utilizing the Q-T graph, this view establishes a direct relevance between each query and its corresponding tools, providing a straightforward measure of their relevance.

This dual-view framework allows for comprehensive access to query-tool relevance, integrating both direct relevance and the broader context of tool collaboration within scenes, thereby enhancing the completeness of tool retrieval.

For the scene-centric view, we adopt the simple but effective Graph Neural Network (GNN)-based LightGCN [11] model to delve into the complex relationships between queries and scenes. This is achieved through iterative aggregation of neighboring information across $I$ layers within the Q-S graph. The aggregation process for the ??-th layer, enhancing the representations of queries e?? (?? $i \cdot$ $\mathbf { e } _ { q } ^ { S ( i ) }$ and scenes $\mathbf { e } _ { s } ^ { S ( i ) }$ e?? (?? )?? , is defined as follows:

$$
\left\{ \begin{array}{l} \mathbf {e} _ {q} ^ {S (i)} = \sum_ {s \in \mathcal {N} _ {q} ^ {S}} \frac {1}{\sqrt {\left| \mathcal {N} _ {q} ^ {S} \right|} \sqrt {\left| \mathcal {N} _ {s} ^ {Q} \right|}} \mathbf {e} _ {s} ^ {S (i - 1)}, \\ \mathbf {e} _ {s} ^ {S (i)} = \sum_ {q \in \mathcal {N} _ {s} ^ {Q}} \frac {1}{\sqrt {\left| \mathcal {N} _ {q} ^ {S}\right)} \sqrt {\left| \mathcal {N} _ {s} ^ {Q}\right |}} \mathbf {e} _ {q} ^ {S (i - 1)}, \end{array} \right. \tag {3}
$$

where $N _ { q } ^ { S }$ , $\mathcal { N } _ { s } ^ { Q }$ represent the sets of neighbors of query $q$ and scene ?? in the Q-S graph, respectively. ${ \mathbf e } _ { q } ^ { S ( 0 ) }$ e?? originates from the representations acquired in the first semantic learning stage, while ${ \bf e } _ { s } ^ { { \hat { S ( 0 ) } } }$ e?? is derived from the mean pooling of the representations of ground-truth tools associated with each scene:

$$
\mathbf {e} _ {s} ^ {S (0)} = \frac {1}{| \mathcal {N} _ {s} ^ {T} |} \sum_ {t \in \mathcal {N} _ {s} ^ {T}} \mathbf {e} _ {t}, \tag {4}
$$

where ${ \cal N } _ { s } ^ { T }$ represents the set of first-order neighbors of scene ?? in the S-T graph.

Then we sum the representations from the 0-th layer to the $I -$ -th layer to get the final query representations $\mathbf { e } _ { q } ^ { S }$ and scene representation $\mathbf { e } _ { s } ^ { S }$ for the scene-centric view:

$$
\left\{ \begin{array}{l} \mathbf {e} _ {q} ^ {S} = \mathbf {e} _ {q} ^ {S (0)} + \dots + \mathbf {e} _ {q} ^ {S (I)}, \\ \mathbf {e} _ {s} ^ {S} = \mathbf {e} _ {s} ^ {S (0)} + \dots + \mathbf {e} _ {s} ^ {S (I)}. \end{array} \right. \tag {5}
$$

In parallel with the scene-centric view, the tool-centric view utilizes LightGCN on the Q-T graph to refine query and tool representations through iterative aggregation. For each layer ??, the enhanced representations, e ?? $\mathbf { e } _ { q } ^ { T ( i ) }$ for queries and $\mathbf { e } _ { t } ^ { T ( i ) }$ e?? for tools, are derived as follows:

$$
\left\{ \begin{array}{l} \mathbf {e} _ {q} ^ {T (i)} = \sum_ {t \in \mathcal {N} _ {q} ^ {T}} \frac {1}{\sqrt {\left| \mathcal {N} _ {q} ^ {T} \right|} \sqrt {\left| \mathcal {N} _ {t} ^ {Q} \right|}} \mathbf {e} _ {t} ^ {T (i - 1)}, \\ \mathbf {e} _ {t} ^ {T (i)} = \sum_ {q \in \mathcal {N} _ {t} ^ {Q}} \frac {1}{\sqrt {\left| \mathcal {N} _ {q} ^ {T} \right|} \sqrt {\left| \mathcal {N} _ {t} ^ {Q} \right|}} \mathbf {e} _ {q} ^ {T (i - 1)}, \end{array} \right. \tag {6}
$$

where $\mathcal { N } _ { q } ^ { T } , \mathcal { N } _ { t } ^ { Q }$ represent the first-order neighbors of query $q$ and tool ?? in the Q-T graph, respectively. ${ \mathbf e } _ { q } ^ { T ( 0 ) }$ and e ?? ${ \bf e } _ { t } ^ { T ( 0 ) }$ are obtained from the first semantic learning stage.

Then we sum the representations from the 0-th layer to the ?? -th layer to derive the final query representations ${ \bf e } _ { q } ^ { \dot { T } }$ and tool representation ${ \mathbf e } _ { t } ^ { T }$ for the tool-centric view:

$$
\left\{ \begin{array}{l} \mathbf {e} _ {q} ^ {T} = \mathbf {e} _ {q} ^ {T (0)} + \dots + \mathbf {e} _ {q} ^ {T (I)}, \\ \mathbf {e} _ {t} ^ {T} = \mathbf {e} _ {t} ^ {T (0)} + \dots + \mathbf {e} _ {t} ^ {T (I)}. \end{array} \right. \tag {7}
$$

Furthermore, leveraging the learned tool representations ${ \mathbf e } _ { t } ^ { T }$ and the S-T graph, the scene representation ${ \bf e } _ { s } ^ { T }$ within the tool-centric view can be obtained by pooling all related tool representations:

$$
\mathbf {e} _ {s} ^ {T} = \frac {1}{\left| \mathcal {N} _ {s} ^ {T} \right|} \sum_ {t \in \mathcal {N} _ {s} ^ {T}} \mathbf {e} _ {t} ^ {T}. \tag {8}
$$

Algorithm 1 The Learning Algorithm of COLT   
Input: PLM, semantic learning training epoch $E$ , Query-scene bipartite graph, query-tool bipartite graph, scene-tool bipartite graph, learning rate $lr$ , weight decay, layer number $I$ , contrastive loss weight $\lambda$ , temperature coefficient $\tau$ , list length $L$ .

Output: COLT Model with learnable parameters $\theta$ .

// Semantic Learning:

1: for $e = 1$ to $E$ do

2: Calculate the InfoNCE loss using Eq. (2)   
3: Update parameter of PLM using AdaW   
4: end for

// Collaborative Learning:

5: Calculate initial ${ \mathbf e } _ { q } ^ { S ( 0 ) }$ e?? , ${ \bf e } _ { s } ^ { S ( 0 ) }$ e ?? , $\mathbf { e } _ { q } ^ { T ( 0 ) }$ and e?? ?? ${ \bf e } _ { t } ^ { T ( 0 ) }$ using the embeddings obtained from the first-stage semantic learning and Eq. (4)   
6: while COLT not Convergence do   
7: for $i = 1$ to ?? do   
8: Conduct message propagation using Eq. (3) and Eq. (6)   
9: end for   
10: Calculate final $\mathbf { e } _ { q } ^ { S }$ , e ???? , e ???? , $\mathbf { e } _ { s } ^ { T }$ and ${ \mathbf e } _ { t } ^ { T }$ using Eq. (5), Eq. (7) and Eq. (8)   
11: Calculate contrastive loss $\mathcal { L } _ { Q } ^ { C }$ and $\mathcal { L } _ { s } ^ { C }$ using Eq. (10) and Eq. (11)   
12: Calculate multi-label loss $\mathcal { L } _ { \mathrm { l i s t } }$ using Eq. (14)   
13: Calculate total loss $\mathcal { L }$ using Eq. (15)   
14: Update model parameter using Adam   
15: end while   
16: return ??

In summary, our dual-view graph collaborative learning framework yields two sets of embeddings: $\mathbf { e } _ { q } ^ { S }$ and $\mathbf { e } _ { s } ^ { S }$ from the scenecentric view, and ${ \mathbf e } _ { q } ^ { T }$ and ${ \bf e } _ { s } ^ { T }$ from the tool-centric view for queries and scenes, respectively. Then, the final matching score of each query-tool pair $( q , t )$ is calculated using the following formula:

$$
\widehat {y} (q, t) = \operatorname {s i m} \left(\mathbf {e} _ {q} ^ {S}, \mathbf {e} _ {t} ^ {T}\right) + \operatorname {s i m} \left(\mathbf {e} _ {q} ^ {T}, \mathbf {e} _ {t} ^ {T}\right). \tag {9}
$$

3.4.3 Learning Objective. As shown in Figure 2 (d), we capture high-order collaborative relationships between tools and align the cooperative interactions across two views using a cross-view contrastive loss. Specifically, the representations of queries and scenes can be learned by optimizing the cross-view InfoNCE [10, 37] loss:

$$
\mathcal {L} _ {Q} ^ {C} = - \frac {1}{| Q |} \sum_ {q \in Q} \log \frac {e ^ {\sin \left(\mathbf {e} _ {q} ^ {S} , \mathbf {e} _ {q} ^ {T}\right) / \tau}}{\sum_ {q - \in Q} e ^ {\sin \left(\mathbf {e} _ {q} ^ {S} , \mathbf {e} _ {q -} ^ {T}\right) / \tau}}, \tag {10}
$$

$$
\mathcal {L} _ {\mathcal {S}} ^ {C} = - \frac {1}{| \mathcal {S} |} \sum_ {s \in \mathcal {S}} \log \frac {e ^ {\operatorname {s i m} \left(\mathbf {e} _ {s} ^ {S} , \mathbf {e} _ {s} ^ {T}\right) / \tau}}{\sum_ {s - \in \mathcal {S}} e ^ {\operatorname {s i m} \left(\mathbf {e} _ {s} ^ {S} , \mathbf {e} _ {s -} ^ {T}\right) / \tau}}, \tag {11}
$$

where $\tau$ is the temperature parameter.

To ensure the complete retrieval of diverse tools from the full set of ground-truth tools, without favoring any particular tool, we design a list-wise multi-label loss as the main learning objective loss. Given a query $q$ , the labeled training data is $\Gamma _ { q } = \{ \mathcal { T } _ { q } = \{ t _ { i } , d _ { i } \} , y =$ $\{ y ( q , t _ { i } ) \} | 1 \leq i \leq L \}$ , where $\mathcal { T } _ { q }$ denotes a tool list with length $L$ , comprising $N _ { q }$ ground-truth tools and $L - N _ { q }$ negative tools that are randomly sampled from the entire tool set. $y ( q , t _ { i } )$ is the binary relevance label, taking a value of either 0 or 1, and the ideal scoring function should meet the following criteria:

$$
p _ {q} ^ {t} = \frac {\gamma (y (q , t))}{\sum_ {t ^ {\prime} \in \mathcal {T} _ {q}} \gamma (y (q , t ^ {\prime}))}, \tag {12}
$$

Table 1: Statistics of the experimental datasets. Tools/Query denotes the number of ground-truth tools for each query.   

<table><tr><td rowspan="2">Dataset</td><td colspan="3"># Query</td><td rowspan="2"># Tool</td><td rowspan="2"># Tools/Query</td></tr><tr><td>Training</td><td>Testing</td><td>Total</td></tr><tr><td>ToolLens</td><td>16,893</td><td>1,877</td><td>18,770</td><td>464</td><td>1 ~ 3</td></tr><tr><td>ToolBench (I2)</td><td>74,257</td><td>8,250</td><td>82,507</td><td>11,473</td><td>2 ~ 4</td></tr><tr><td>ToolBench (I3)</td><td>21,361</td><td>2,373</td><td>23,734</td><td>1,419</td><td>2 ~ 4</td></tr></table>

where ${ p } _ { q } ^ { t }$ is the probability of selecting tool ?? . $\gamma ( y ( q , t ) ) \ : = \ : 1$ if $y ( q , t ) = \overset { \cdot } { 1 }$ and $\gamma ( y ( q , t ) ) = 0$ if $y ( q , t ) = 0$ .

Similarly, given the predicted scores $\{ \widehat { y } ( q , t _ { 1 } ) , \cdot \cdot \cdot , \widehat { y } ( q , t _ { L } ) \}$ , the probability of selecting tool $t$ bcan be derived:

$$
\widehat {p _ {q} ^ {t}} = \frac {\gamma (\widehat {y} (q , t))}{\sum_ {t ^ {\prime} \in \mathcal {T} _ {q}} \gamma (\widehat {y} (q , t ^ {\prime}))}. \tag {13}
$$

Therefore, the list-wise multi-label loss function minimizes the discrepancy between these two probability distributions:

$$
\mathcal {L} _ {\text {l i s t}} = - \sum_ {q \in Q} \sum_ {t \in \mathcal {T} _ {q}} p _ {q} ^ {t} \log \widehat {p _ {q} ^ {t}} + (1 - p _ {q} ^ {t}) \log (1 - \widehat {p _ {q} ^ {t}}), \tag {14}
$$

Based on t multi-label loss $\mathcal { L } _ { \mathrm { l i s t } }$ and the contrastive loss $\mathcal { L } _ { Q } ^ { C }$ $\mathcal { L }$

$$
\mathcal {L} = \mathcal {L} _ {\text {l i s t}} + \lambda \left(\mathcal {L} _ {Q} ^ {C} + \mathcal {L} _ {S} ^ {C}\right), \tag {15}
$$

where $\lambda$ is the co-efficient to balance the two losses.

The learning algorithm of COLT is summarized in Algorithm 1.

# 4 DATASETS

To verify the effectiveness of COLT, we utilize two datasets for multi-tool scenarios: ToolBench and a newly constructed dataset, ToolLens. We randomly select $1 0 \%$ of the entire dataset to serve as the test data. The statistics of the datasets after preprocessing are summarized in Table 1.

ToolBench. ToolBench [28] is a benchmark commonly used to evaluate the capability of LLMs in tool usage. In our experiments, we notice that its three subsets exhibit distinct characteristics. The first subset (I1) focuses on single-tool scenarios, which diverges from our emphasis on multi-tool tasks. However, both the second subset (I2) and the third subset (I3) align with our focus on multitool tasks. Therefore, we chose I2 and I3 as the primary datasets for our experiments.

ToolLens. While existing datasets like ToolBench [28] and TOOLE [14] provide multi-tool scenarios, they present limitations. TOOLE encompasses only 497 queries, and ToolBench’s dataset construction, which involves providing complete tool descriptions to ChatGPT, results in verbose and semantically direct queries. These do not accurately reflect the brief and often multifaceted nature of real-world user queries. To address these shortcomings, we introduce ToolLens, crafted specifically for multi-tool scenarios.

As shown in Figure 3, the creation of ToolLens involves a novel five-step methodology: 1) Tool Selection: To create a high-quality tool dataset, we rigorously filter ToolBench, focusing on 464 available and directly callable tools relevant to everyday user queries, excluding those for authentication or testing. 2) Scene Mining:

![](images/0b02d4f4c7485bd8e85e871de6dd3dc59009d106f9870dce65f76ced11f73d7d.jpg)  
Figure 3: An overview of the dataset construction pipeline of ToolLens. Human verification is included at each step.

Table 2: Quality verification of ToolLens.   

<table><tr><td>Evaluator</td><td colspan="3">ToolLens vs. ToolBench</td><td colspan="3">ToolLens vs. TOOLE</td></tr><tr><td colspan="7">Whether the query is natural?</td></tr><tr><td>GPT-4</td><td>ToolLens 68%</td><td>ToolBench 14%</td><td>Equal 18%</td><td>ToolLens 44%</td><td>TOOLE 36%</td><td>Equal 20%</td></tr><tr><td>Human</td><td>ToolLens 64%</td><td>ToolBench 10%</td><td>Equal 26%</td><td>ToolLens 54%</td><td>TOOLE 24%</td><td>Equal 22%</td></tr><tr><td colspan="7">Whether the user intent is multifaceted?</td></tr><tr><td>GPT-4</td><td>ToolLens 62%</td><td>ToolBench 14%</td><td>Equal 24%</td><td>ToolLens 50%</td><td>TOOLE 24%</td><td>Equal 26%</td></tr><tr><td>Human</td><td>ToolLens 60%</td><td>ToolBench 12%</td><td>Equal 28%</td><td>ToolLens 58%</td><td>TOOLE 18%</td><td>Equal 24%</td></tr></table>

We prompt GPT-4 to generate potential scenes that are relevant to the detailed descriptions of the selected tools, and ensure their validity through human verification. 3) Query Generation: We then employ GPT-4 to generate queries based on the provided scene and the parameters required for tool calling. Notably, we avoid providing the complete tool description to GPT-4 to avoid the generated query being closely aligned with the tool. 4) Tool Aggregation: The queries generated in the aforementioned way are only relevant to a single tool. To enhance the relevance of queries across multiple tools, we reprocess them through GPT-4 to identify categories of tools that could be relevant, which are then aligned with our tool set through dense retrieval and manual verification. 5) Query Rewriting: Finally, we utilize GPT-4 to revise queries to incorporate all necessary parameters by providing it with both the initial query and a list of required parameters, thereby yielding concise yet intentionally multifaceted queries that better mimic real-world user interactions. It is worth noting that we incorporate a human verification process at each step to ensure data quality.

This comprehensive construction pipeline ensures ToolLens accurately simulates real-world tool retrieval dynamics. The resulting ToolLens dataset includes 18,770 queries and 464 tools, with each query being associated with $1 \sim 3$ verified tools.

Discussion and Quality Verification. Unlike prior datasets, Tool-Lens uniquely focuses on creating natural, concise, and multifaceted queries to reflect real-world demands. To assess the quality of Tool-Lens, following previous works [8, 21, 34], we employ GPT-4 as an evaluator and human evaluation where three well-educated doctor students are invited to evaluate 50 randomly sampled cases from ToolLens, ToolBench and TOOLE in the following two aspects:(1) Natural-query: whether the query is natural. (2) Multifaceted intentions: whether the user intent is multifaceted. The results are

illustrated in Table 2. In most cases, ToolLens outperforms Tool-Bench and TOOLE. Furthermore, using GPT-4 as the evaluator shows a high degree of consistency with human evaluation trends, which underscores the validity of employing GPT-4 as an evaluator.

# 5 EXPERIMENTS

In this section, we first describe the experimental setups and then conduct an extensive evaluation and analysis of the proposed COLT. The source code and the proposed ToolLens dataset are publicly available at https://github.com/quchangle1/COLT.

# 5.1 Experimental Setups

5.1.1 Evaluation Metrics. Following the previous works [8, 28, 51], we utilize the widely used retrieval metrics Recall@?? and NDCG@?? and report the metrics for $K \in \{ 3 , 5 \}$ . However, as discussed in Section 1, Recall and NDCG do not adequately fulfill the requirements of completeness that are crucial for effective tool retrieval. To further tailor our assessment to the specific challenges of tool retrieval tasks, we also introduce a new metric, $\operatorname { C O M P } @ K$ This metric is designed to measure whether the top- $K$ retrieved tools form a complete set with respect to the ground-truth set:

$$
\mathrm {C O M P} @ K = \frac {1}{| Q |} \sum_ {q = 1} ^ {| Q |} \mathbb {I} (\Phi_ {q} \subseteq \Psi_ {q} ^ {K}),
$$

where $\Phi _ { q }$ is the set of ground-truth tools for query ??, $\Psi _ { q } ^ { K }$ represents the top- $K$ tools retrieved for query $q$ , and I(·) is an indicator function that returns 1 if the retrieval results include all ground-truth tools within the top- $K$ results for query $q$ , and 0 otherwise.

5.1.2 Baselines. As our proposed COLT is model-agnostic, we apply it to several representative PLM-based retrieval models (as backbone models) to validate the effectiveness:

ANCE[45] uses a dual-encoder architecture with an asynchronously updated ANN index for training, enabling global selection of hard negatives. TAS-B[12] is a bi-encoder that employs balanced margin sampling to ensure efficient query sampling from clusters per batch. co-Condenser[7] uses a query-agnostic contrastive loss to cluster related text segments and distinguish unrelated ones. Contriever[15] leverages inverse cloze tasks, cropping for positive pair generation, and momentum contrastive learning to achieve state-of-the-art zero-shot retrieval performance.

In addition to PLM-based dense retrieval methods, we also compare with the classical lexical retrieval model BM25, widely used for tool retrieval as documented in [8, 28]. BM25 [30] uses an inverted index to identify suitable tools based on exact term matching.

Table 3: Performance comparison of different tool retrieval methods on ToolLens and ToolBench datasets. “†” denotes the best results for each column. The term “Zero-shot” refers to the performance of dense retrieval models without any training. “+Fine-tune” indicates that retrieval models are fine-tuned on ToolLens and ToolBench datasets. “+COLT (Ours)” indicates that dense retrieval backbones are equipped with our proposed method. R@??, N@??, and $\mathbf { C } @ K$ are short for Recal $@ K$ , NDCG@?? and COMP@??, respectively.   

<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Framework</td><td colspan="6">ToolLens</td><td colspan="6">ToolBench (I2)</td><td colspan="6">ToolBench (I3)</td></tr><tr><td>R@3</td><td>R@5</td><td>N@3</td><td>N@5</td><td>C@3</td><td>C@5</td><td>R@3</td><td>R@5</td><td>N@3</td><td>N@5</td><td>C@3</td><td>C@5</td><td>R@3</td><td>R@5</td><td>N@3</td><td>N@5</td><td>C@3</td><td>C@5</td></tr><tr><td>BM25</td><td>-</td><td>21.58</td><td>26.88</td><td>23.19</td><td>26.09</td><td>3.89</td><td>6.13</td><td>17.06</td><td>21.38</td><td>17.83</td><td>19.88</td><td>2.39</td><td>4.37</td><td>29.33</td><td>35.88</td><td>32.20</td><td>35.08</td><td>5.52</td><td>9.78</td></tr><tr><td rowspan="3">ANCE</td><td>Zero-shot</td><td>20.82</td><td>26.56</td><td>21.45</td><td>24.57</td><td>5.06</td><td>7.46</td><td>20.82</td><td>26.56</td><td>21.45</td><td>24.57</td><td>5.06</td><td>7.46</td><td>21.55</td><td>26.38</td><td>23.44</td><td>25.60</td><td>2.44</td><td>4.59</td></tr><tr><td>+Fine-tune</td><td>80.62</td><td>94.17</td><td>82.35</td><td>90.15</td><td>54.23</td><td>85.83</td><td>58.58</td><td>67.20</td><td>58.58</td><td>63.75</td><td>26.46</td><td>42.80</td><td>65.11</td><td>76.63</td><td>69.27</td><td>74.14</td><td>34.68</td><td>53.64</td></tr><tr><td>+COLT (Ours)</td><td>92.15</td><td>\( 97.78^{\dagger} \)</td><td>92.78</td><td>96.10</td><td>80.50</td><td>94.40</td><td>70.76</td><td>80.59</td><td>73.64</td><td>77.98</td><td>45.10</td><td>62.93</td><td>73.37</td><td>83.97</td><td>77.95</td><td>82.14</td><td>46.01</td><td>66.41</td></tr><tr><td rowspan="3">TAS-B</td><td>Zero-shot</td><td>19.10</td><td>23.71</td><td>19.81</td><td>22.33</td><td>5.17</td><td>7.14</td><td>19.10</td><td>23.71</td><td>19.81</td><td>22.33</td><td>5.17</td><td>7.14</td><td>25.32</td><td>31.15</td><td>27.80</td><td>30.36</td><td>3.84</td><td>6.40</td></tr><tr><td>+Fine-tune</td><td>81.26</td><td>94.06</td><td>82.54</td><td>89.94</td><td>54.66</td><td>85.72</td><td>62.78</td><td>67.49</td><td>58.96</td><td>64.21</td><td>26.74</td><td>43.66</td><td>66.04</td><td>77.64</td><td>70.41</td><td>75.34</td><td>35.69</td><td>55.75</td></tr><tr><td>+COLT (Ours)</td><td>91.49</td><td>96.91</td><td>92.48</td><td>95.63</td><td>79.00</td><td>92.22</td><td>71.64</td><td>81.12</td><td>74.60</td><td>78.74</td><td>46.77</td><td>64.38</td><td>74.49</td><td>84.58</td><td>79.03</td><td>82.95</td><td>48.16</td><td>68.35</td></tr><tr><td rowspan="3">coCondensor</td><td>Zero-shot</td><td>15.33</td><td>19.37</td><td>16.15</td><td>18.32</td><td>3.02</td><td>5.33</td><td>15.33</td><td>19.37</td><td>16.15</td><td>18.32</td><td>3.02</td><td>5.30</td><td>20.80</td><td>25.24</td><td>23.21</td><td>25.10</td><td>2.07</td><td>3.75</td></tr><tr><td>+Fine-tune</td><td>82.37</td><td>94.69</td><td>83.90</td><td>91.06</td><td>56.37</td><td>86.73</td><td>57.70</td><td>69.46</td><td>60.80</td><td>66.07</td><td>28.78</td><td>46.06</td><td>66.97</td><td>79.30</td><td>71.20</td><td>76.50</td><td>37.08</td><td>58.66</td></tr><tr><td>+COLT (Ours)</td><td>92.65</td><td>\( 97.78^{\dagger} \)</td><td>93.16</td><td>96.17</td><td>82.25</td><td>\( 94.56^{\dagger} \)</td><td>73.91</td><td>83.47</td><td>76.75</td><td>80.87</td><td>49.15</td><td>67.75</td><td>75.48</td><td>84.97</td><td>80.00</td><td>83.55</td><td>49.17</td><td>\( 68.64^{\dagger} \)</td></tr><tr><td rowspan="3">Contriever</td><td>Zero-shot</td><td>25.67</td><td>31.15</td><td>26.96</td><td>29.95</td><td>7.46</td><td>9.80</td><td>25.67</td><td>31.15</td><td>26.96</td><td>29.95</td><td>7.46</td><td>9.80</td><td>31.37</td><td>38.60</td><td>34.13</td><td>37.37</td><td>6.03</td><td>11.42</td></tr><tr><td>+Fine-tune</td><td>83.58</td><td>95.17</td><td>84.98</td><td>91.69</td><td>59.46</td><td>88.65</td><td>58.89</td><td>70.75</td><td>62.11</td><td>67.42</td><td>29.77</td><td>48.31</td><td>68.58</td><td>80.05</td><td>72.86</td><td>77.69</td><td>39.70</td><td>60.89</td></tr><tr><td>+COLT (Ours)</td><td>\( 93.64^{\dagger} \)</td><td>97.75</td><td>\( 94.53^{\dagger} \)</td><td>\( 96.91^{\dagger} \)</td><td>\( 84.55^{\dagger} \)</td><td>94.08</td><td>\( 75.72^{\dagger} \)</td><td>\( 85.03^{\dagger} \)</td><td>\( 78.57^{\dagger} \)</td><td>\( 82.54^{\dagger} \)</td><td>\( 51.97^{\dagger} \)</td><td>\( 70.10^{\dagger} \)</td><td>\( 76.63^{\dagger} \)</td><td>\( 85.50^{\dagger} \)</td><td>\( 81.21^{\dagger} \)</td><td>\( 84.18^{\dagger} \)</td><td>\( 52.00^{\dagger} \)</td><td>68.47</td></tr></table>

5.1.3 Implementation Details. We utilize the BEIR [40] framework for dense retrieval baselines, setting the training epochs to 5 with the learning rate of $2 e { \mathrm { - } } 5$ , weight decay of 0.01, and using the AdamW optimizer. Our model-agnostic approach directly applies dense retrieval for semantic learning. During collaborative learning, we set the batch size as 2048 and carefully tune the learning rate among $\left\{ 1 e - 3 , 5 e - 3 , 1 e - 4 , 5 e - 4 , 1 e - 5 \right\}$ , the weight decay among $\left\{ 1 e - 5 , 1 e - 6 , 1 e - 7 \right\}$ , as well as the layer number ?? among $\{ 1 , 2 , 3 \}$ .

# 5.2 Experimental Results

5.2.1 Retrieval Performance. Table 3 presents the results of different tool retrieval methods on ToolLens, ToolBench (I2 and I3). From the results, we have the following observations and conclusions:

We can observe that traditional dense retrieval models perform poorly in zero-shot scenarios, even inferior to that of BM25. This indicates that these models may not be well-suited for tool retrieval tasks. Conversely, the BM25 model significantly lags behind finetuned PLM-based dense retrieval methods, underscoring the superior capability of the latter in leveraging contextual information for more effective tool retrieval. Despite this advantage, PLM-based methods fall short in the COMP metric, which is specifically designed for evaluating completeness in tool retrieval scenarios. This suggests that while effective for general retrieval tasks, PLM-based methods may not fully meet the unique demands of tool retrieval.

All base models equipped with COLT exhibit significant performance gains across all metrics on all three datasets, particularly in the COMP@3 metric. These improvements demonstrate the effectiveness of COLT, which can be attributed to the fact that COLT adopts a two-stage learning framework with semantic learning followed by collaborative learning. In this way, COLT can capture the intricate collaborative relationships between tools, resulting in effectively retrieving a complete tool set.

5.2.2 Downstream Tool Learning Performance. To verify that improvements of COLT in tool retrieval truly enhance downstream tool learning applications, we conduct a validation study using the pairwise comparison method [5, 19, 36]. We randomly select 100

Table 4: Elo ratings for different models w.r.t. “Coherence”, “Relevance”, “Comprehensiveness” and “Overall” evaluated by GPT-4 on ToolLens dataset.   

<table><tr><td rowspan="2"></td><td colspan="5">Evaluation Aspects</td></tr><tr><td>Coherence</td><td>Relevance</td><td>Comprehensiveness</td><td colspan="2">Overall</td></tr><tr><td>BM25</td><td>848</td><td>845</td><td>860</td><td>780</td><td></td></tr><tr><td>ANCE</td><td>934</td><td>936</td><td>946</td><td>1016</td><td></td></tr><tr><td>TAS-B</td><td>995</td><td>991</td><td>988</td><td>1028</td><td></td></tr><tr><td>coCondensor</td><td>1031</td><td>1036</td><td>1041</td><td>1035</td><td></td></tr><tr><td>Contriever</td><td>1076</td><td>1082</td><td>1044</td><td>1046</td><td></td></tr><tr><td>COLT (Ours)</td><td>1116</td><td>1110</td><td>1121</td><td>1096</td><td></td></tr></table>

queries from the test set of ToolLens and use various retrieval models to retrieve the top-3 tools for each query. Then we utilize GPT-4 as an evaluator, examining the responses generated with different retrieved tools across four dimensions: Coherence, Relevance, Comprehensiveness, and Overall. Specifically, the user query and a pair of responses are utilized as prompts to guide GPT-4 in determining the superior response. Additionally, we also consider that LLMs may respond differently to the order in which text is presented in the prompt [13, 20, 22, 39]. So each comparison is conducted twice with reversed response order to mitigate potential biases from text order, ensuring a more reliable assessment.

We establish a tournament-style competition using the Elo ratings system, which is widely employed in chess and other twoplayer games to measure the relative skill levels of the players [6, 44]. Following previous works [3], we start with a score of 1, 000 and set $K$ -factor to 32. Additionally, to minimize the impact of match sequences on Elo scores, we conduct these computations 10, 000 times using various random seeds to control for ordering effects.

The results in Table 4 show that superior tool retrieval models can significantly improve downstream tool learning performance. Moreover, responses generated with the tools retrieved from COLT notably outperform those from other methods, achieving the highest Elo ratings in all four assessed dimensions. These results highlight the pivotal role of effective tool retrieval in tool learning applications and further confirm the superiority of COLT.

Table 5: Ablation study of the proposed COLT.   

<table><tr><td rowspan="2">Methods</td><td colspan="2">ToolLens</td><td colspan="2">ToolBench</td></tr><tr><td>R@|N|</td><td>C@|N|</td><td>R@|N|</td><td>C@|N|</td></tr><tr><td>ANCE+COLT (Ours)</td><td>91.08</td><td>78.36</td><td>72.22</td><td>44.28</td></tr><tr><td>w/o semantic learning</td><td>36.49</td><td>6.84</td><td>21.92</td><td>1.60</td></tr><tr><td>w/o collaborative learning</td><td>77.36</td><td>49.01</td><td>62.39</td><td>30.12</td></tr><tr><td>w/o list-wise learning</td><td>79.94</td><td>52.68</td><td>66.02</td><td>35.82</td></tr><tr><td>w/o contrastive learning</td><td>85.63</td><td>63.87</td><td>66.57</td><td>34.55</td></tr><tr><td>TAS-B+COLT (Ours)</td><td>90.29</td><td>77.73</td><td>72.84</td><td>45.46</td></tr><tr><td>w/o semantic learning</td><td>38.49</td><td>9.16</td><td>32.16</td><td>5.47</td></tr><tr><td>w/o collaborative learning</td><td>76.86</td><td>47.83</td><td>63.61</td><td>31.73</td></tr><tr><td>w/o list-wise learning</td><td>79.89</td><td>52.25</td><td>66.91</td><td>37.27</td></tr><tr><td>w/o contrastive learning</td><td>84.86</td><td>62.65</td><td>67.66</td><td>36.36</td></tr><tr><td>coCondensor+COLT (Ours)</td><td>91.49</td><td>79.86</td><td>74.00</td><td>47.49</td></tr><tr><td>w/o semantic learning</td><td>30.38</td><td>5.54</td><td>25.07</td><td>2.27</td></tr><tr><td>w/o collaborative learning</td><td>78.83</td><td>50.61</td><td>64.38</td><td>33.08</td></tr><tr><td>w/o list-wise learning</td><td>81.42</td><td>54.16</td><td>69.18</td><td>40.67</td></tr><tr><td>w/o contrastive learning</td><td>86.78</td><td>67.07</td><td>68.92</td><td>37.80</td></tr><tr><td>Contriever+COLT (Ours)</td><td>92.76</td><td>82.95</td><td>75.40</td><td>49.81</td></tr><tr><td>w/o semantic learning</td><td>65.21</td><td>30.90</td><td>53.33</td><td>19.63</td></tr><tr><td>w/o collaborative learning</td><td>80.60</td><td>54.44</td><td>68.20</td><td>36.91</td></tr><tr><td>w/o list-wise learning</td><td>81.49</td><td>54.93</td><td>71.80</td><td>46.07</td></tr><tr><td>w/o contrastive learning</td><td>84.58</td><td>60.52</td><td>69.46</td><td>39.02</td></tr></table>

# 5.3 Further Analysis

Next, we delve into investigating the effectiveness of COLT. We report the experimental results on the ToolLens and ToolBench (I3) datasets, observing similar trends on ToolBench (I2). Recall@|N| and COMP@|N| are adopted as evaluation metrics, with |N| representing the count of ground-truth tools suitable for each query.

5.3.1 Ablation Study. We conduct ablation studies to assess the impact of various components within our COLT. The results presented in Table 5, highlight the significance of each element:

w/o semantic learning denotes an off-the-shelf PLM is directly employed to get the initial representation for the subsequent collaborative learning stage without semantic learning on the given dataset in Section 3.3. The absence of semantic learning significantly diminishes performance, confirming its essential role in aligning the representations of tools and queries as the basic for the following collaborative learning. Notably, the omission of semantic learning elements markedly reduces performance across other models more than with Contriever. This highlights the superior ability of Contriever in zero-shot learning scenarios compared to the other models.

w/o collaborative learning is a variant where the collaborative learning state is omitted (i.e., only semantic learning). The significant decline in performance in this variant further supports the effectiveness of COLT in capturing the high-order relationships between tools through graph collaborative learning, thereby achieving comprehensive tool retrieval.

w/o list-wise learning refers to a variant that optimizes using pair-wise loss in place of the list-wise loss defined in Eq. (14). This substitution results in a significant drop in performance, highlighting that compared to pairwise loss, list-wise loss optimizes the tools in the same scenario as a whole entity, proving more effective in focusing on completeness.

![](images/c35f2b29e1dbc79b5f666c5197763440d07c0e10db950a0faa7d6fa58d02c9ec.jpg)  
(a) ToolLens

![](images/8e24ff330b32b237434bb0c33e2c660d85b67415d14d77bf737782a683496aab.jpg)  
(b) ToolBench   
Figure 4: Comparison of different model sizes of PLM.

w/o contrastive learning refers to a variant that optimizes without the contrastive loss defined in Eq. (10) and (11); This omission also leads to a noticeable performance drop, emphasizing the benefits of introducing contrastive learning to achieve better representation for queries and tools within a dual-view learning framework. Additionally, our analysis reveals that contrastive learning is particularly crucial for Contriever, as its absence results in performance lagging behind the other models. This also indicates that the importance of contrastive learning varies across different backbones.

5.3.2 Performance w.r.t. Model Size of PLM. To verify the adaptability and effectiveness of COLT across varying sizes of PLMs, we explore its integration with a range of BERT models, from BERTmini to BERT-large. This analysis aims to determine whether COLT could generally enhance tool retrieval performance across different model sizes. Figure 4 shows that while the performance of the base model naturally improves with larger PLM sizes, the integration of COLT consistently boosts performance across all sizes. Remarkably, even BERT-mini equipped with COLT, significantly outperforms a much larger BERT-large model ( $3 0 \mathrm { x }$ larger) operating without our COLT. These results underscore the generalization and robustness of COLT, demonstrating its potential to significantly improve tool retrieval performance for PLMs of any scale.

5.3.3 Performance w.r.t. Different Tool Sizes. The ToolLens dataset encompasses queries that require $1 \sim 3$ tools, while ToolBench includes queries needing $2 \sim 4$ tools. To assess how well COLT adapts to queries with diverse tool requirements, we divide each dataset into three subsets based on the number of tools required by each query and conduct a focused analysis on these subsets. As shown in Figure 5, there is a discernible decline in performance as the number of ground-truth tools increases, reflecting the escalating difficulty of achieving complete retrieval. However, COLT demonstrates consistent performance improvement across all subsets and backbones. This improvement is especially significant in the most challenging cases, where queries may involve using three or four tools. These results consistently highlight the robustness of COLT and its potential to meet the complex demands of tool retrieval tasks across various scenarios.

5.3.4 Hyper-parameter Analysis. Figure 6 illustrates the sensitivity of COLT to the temperature parameter $\tau$ and the loss weight ??, but shows relative insensitivity to variations in the sampled list length ??. The influence of $\tau$ varies across two datasets, suggesting that its impact depends on the specific data distribution. Conversely, the pattern observed for $\lambda$ across both datasets is consistent, marked

![](images/0d64a0bbac89f36f29b79982755e6da6811a705cf207f0aa0c7fa613435acc7e.jpg)

![](images/b45cd41732b4928b6cd3e23f200096947c0022935e26fdd58a500e438905d766.jpg)

![](images/5a7bcf72ad20271c4a3f963d86513708a5d28c987a57bfdb6e8c190d903c81e8.jpg)

![](images/152b6fca3e2a8251803d3d2b3549ef212f98856feef17cefaec5f9b9605faefc.jpg)

![](images/0ea7c26f394ff7bafd04b7b1cdfaa39127cc1a35e358e972c46e3a3139ceb902.jpg)

![](images/c5faecfd5c2cbc9f1d4d134d8e7fe900758bc60a35a8f0cb60055fbd1f0100f7.jpg)

![](images/fe918cc42ae92cd25036d4be6fbb395bed61c229b0301793824eab23c7162761.jpg)

![](images/6d1ac1545c151b14f0bc532ad371d4c80c042aff8e03d952e1355afcdf2c7918.jpg)  
(b) ToolBench

![](images/c6ffe1deb45cbad1df3726ec270983aa9d8841bd65e74c1a95e9981d6e2c70ab.jpg)  
Figure 5: Performance comparison regarding different sizes of ground-truth tool sets.

![](images/3a2c94b7d89e9de24cb480479146268a524eee6ec3e83dea9b00df8f268e5a4c.jpg)

![](images/ccdae3416ba32783b03fa9de41b8e67ad119ed351beaf85bb9da7d7cc17f5b17.jpg)

![](images/289fe2b87b2e75cc20c6d1658cba60d9849615e046617e869a14e056c6f76b62.jpg)  
(a) Temperature ??.

![](images/2d8e4650a9de6dba6ad73f550c64981839b351b77df1959a46e542ea14fe9c6a.jpg)

![](images/427090fcc5e8d9c381b74e52117a1ae8ad07fa904646177259d1cee9b3b11894.jpg)

![](images/8e147c39fc977dc5334d07176cfb0f755261486206001f81b091dd1b8dbcc130.jpg)

![](images/83feccf93aa89e17dfd46fa89a649f2328afac4557744e1fb185d8e8b539172d.jpg)  
(b) Loss weight ??.

![](images/e7fbfb792c5f3c8e6ab3d93304cabfda9f2f10172f90288b55f803e221205f3b.jpg)

![](images/42a04149929ced76bbe60db3ef19d10a35db01b10c885693b4011e3e45b15ecd.jpg)

![](images/516c568e4437a5ae107121a1b247b14de1d23616b8752ba11c2c4acf63a72551.jpg)

![](images/f326b6bcf960a4c413783179232294d311387b6cbc758476acec69dc74223b7a.jpg)  
(c) List length ??.   
Figure 6: Sensitivity analysis of COLT performance to hyper-parameters. (a) shows the dependency of model performance on temperature ??. (b) illustrates the influence of loss weight ??. (c) examines the effect of list length ??.

by an initial performance improvement that eventually plateaus, underscoring the importance of carefully selecting ?? to maximize the effectiveness of COLT.

# 6 CONCLUSION

This study introduces COLT, a novel model-agnostic approach designed to enhance the completeness of tool retrieval tasks, comprising two stages: semantic learning and collaborative learning. We initially employ semantic learning to ensure semantic representation between queries and tools. Subsequently, by incorporating graph collaborative learning and cross-view contrastive learning, COLT captures the collaborative relationships among tools. Extensive experimental results and analysis demonstrate the effectiveness

of COLT, especially in handling multifaceted queries with multiple tool requirements. Furthermore, we release a new dataset ToolLens and introduce a novel evaluation metric COMP, both of which are valuable resources for facilitating future research on tool retrieval.

# ACKNOWLEDGMENTS

This work was funded by the National Key R&D Program of China (2023YFA1008704), the National Natural Science Foundation of China (No. 62377044), Beijing Key Laboratory of Big Data Management and Analysis Methods, Major Innovation & Planning Interdisciplinary Platform for the “Double-First Class” Initiative, funds for building world-class universities (disciplines) of Renmin University of China, and PCC@RUC.

# REFERENCES

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).   
[2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. Advances in neural information processing systems 33 (2020), 1877–1901.   
[3] Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An Open-Source Chatbot Impressing GPT-4 with $9 0 \% ^ { \star }$ ChatGPT Quality. https://vicuna.lmsys.org   
[4] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2023. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research 24, 240 (2023), 1–113.   
[5] Sunhao Dai, Ninglu Shao, Haiyuan Zhao, Weijie Yu, Zihua Si, Chen Xu, Zhongxiang Sun, Xiao Zhang, and Jun Xu. 2023. Uncovering ChatGPT’s Capabilities in Recommender Systems. In Proceedings of the 17th ACM Conference on Recommender Systems (RecSys ’23). ACM. https://doi.org/10.1145/3604915.3610646   
[6] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314 [cs.LG]   
[7] Luyu Gao and Jamie Callan. 2021. Unsupervised corpus aware language model pre-training for dense passage retrieval. arXiv preprint arXiv:2108.05540 (2021).   
[8] Shen Gao, Zhengliang Shi, Minghang Zhu, Bowen Fang, Xin Xin, Pengjie Ren, Zhumin Chen, Jun Ma, and Zhaochun Ren. 2024. Confucius: Iterative Tool Learning from Introspection Feedback by Easy-to-Difficult Curriculum. In AAAI.   
[9] Jiafeng Guo, Yinqiong Cai, Yixing Fan, Fei Sun, Ruqing Zhang, and Xueqi Cheng. 2022. Semantic models for the first-stage retrieval: A comprehensive review. ACM Transactions on Information Systems (TOIS) 40, 4 (2022), 1–42.   
[10] Michael Gutmann and Aapo Hyvärinen. 2010. Noise-contrastive estimation: A new estimation principle for unnormalized statistical models. In Proceedings of the thirteenth international conference on artificial intelligence and statistics. JMLR Workshop and Conference Proceedings, 297–304.   
[11] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. Lightgcn: Simplifying and powering graph convolution network for recommendation. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval. 639–648.   
[12] Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, and Allan Hanbury. 2021. Efficiently teaching an effective dense retriever with balanced topic aware sampling. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval. 113–122.   
[13] Yupeng Hou, Junjie Zhang, Zihan Lin, Hongyu Lu, Ruobing Xie, Julian McAuley, and Wayne Xin Zhao. 2024. Large Language Models are Zero-Shot Rankers for Recommender Systems. arXiv:2305.08845 [cs.IR]   
[14] Yue Huang, Jiawen Shi, Yuan Li, Chenrui Fan, Siyuan Wu, Qihui Zhang, Yixin Liu, Pan Zhou, Yao Wan, Neil Zhenqiang Gong, and Lichao Sun. 2023. MetaTool Benchmark: Deciding Whether to Use Tools and Which to Use. arXiv preprint arXiv: 2310.03128 (2023).   
[15] Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118 (2021).   
[16] Kalervo Järvelin and Jaana Kekäläinen. 2002. Cumulated gain-based evaluation of IR techniques. ACM Transactions on Information Systems (TOIS) 20, 4 (2002), 422–446.   
[17] Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of naacL-HLT, Vol. 1. 2.   
[18] Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. 2023. API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 3102–3116. https://doi.org/10.18653/v1/2023.emnlp-main.187   
[19] Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D. Manning, Christopher Ré, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu Ren, Huaxiu Yao, Jue Wang, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. 2023. Holistic Evaluation of Language Models. arXiv:2211.09110 [cs.CL]

[20] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the Middle: How Language Models Use Long Contexts. arXiv:2307.03172 [cs.CL]   
[21] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. arXiv:2303.16634 [cs.CL]   
[22] Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. 2022. Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity. arXiv:2104.08786 [cs.CL]   
[23] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. arXiv preprint arXiv:2212.10511 (2022).   
[24] Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, et al. 2023. Augmented language models: a survey. arXiv preprint arXiv:2302.07842 (2023).   
[25] Bhargavi Paranjape, Scott Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, and Marco Tulio Ribeiro. 2023. Art: Automatic multi-step reasoning and tool-use for large language models. arXiv preprint arXiv:2303.09014 (2023).   
[26] Aaron Parisi, Yao Zhao, and Noah Fiedel. 2022. Talm: Tool augmented language models. arXiv preprint arXiv:2205.12255 (2022).   
[27] Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Yufei Huang, Chaojun Xiao, Chi Han, et al. 2023. Tool learning with foundation models. arXiv preprint arXiv:2304.08354 (2023).   
[28] Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al. 2023. Toolllm: Facilitating large language models to master $1 6 0 0 0 +$ real-world apis. arXiv preprint arXiv:2307.16789 (2023).   
[29] Changle Qu, Sunhao Dai, Xiaochi Wei, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Jun Xu, and Ji-Rong Wen. 2024. Tool Learning with Large Language Models: A Survey. arXiv preprint arXiv:2405.17935 (2024).   
[30] Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends® in Information Retrieval 3, 4 (2009), 333–389.   
[31] Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761 (2023).   
[32] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. 2024. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. Advances in Neural Information Processing Systems 36 (2024).   
[33] Yifan Song, Weimin Xiong, Dawei Zhu, Cheng Li, Ke Wang, Ye Tian, and Sujian Li. 2023. Restgpt: Connecting large language models with real-world applications via restful apis. arXiv preprint arXiv:2306.06624 (2023).   
[34] Andrea Sottana, Bin Liang, Kai Zou, and Zheng Yuan. 2023. Evaluation Metrics in the Era of GPT-4: Reliably Evaluating Large Language Models on Sequence to Sequence Tasks. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 8776–8788. https: //doi.org/10.18653/v1/2023.emnlp-main.543   
[35] Karen Sparck Jones. 1972. A statistical interpretation of term specificity and its application in retrieval. Journal of documentation 28, 1 (1972), 11–21.   
[36] Weiwei Sun, Zheng Chen, Xinyu Ma, Lingyong Yan, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, and Zhaochun Ren. 2023. Instruction Distillation Makes Large Language Models Efficient Zero-shot Rankers. arXiv:2311.01555 [cs.IR]   
[37] Jiakai Tang, Sunhao Dai, Zexu Sun, Xu Chen, Jun Xu, Wenhui Yu, Lantao Hu, Peng Jiang, and Han Li. 2024. Towards Robust Recommendation via Decision Boundary-aware Graph Contrastive Learning. Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (2024).   
[38] Qiaoyu Tang, Ziliang Deng, Hongyu Lin, Xianpei Han, Qiao Liang, and Le Sun. 2023. ToolAlpaca: Generalized Tool Learning for Language Models with 3000 Simulated Cases. arXiv preprint arXiv:2306.05301 (2023).   
[39] Raphael Tang, Xinyu Zhang, Xueguang Ma, Jimmy Lin, and Ferhan Ture. 2023. Found in the Middle: Permutation Self-Consistency Improves Listwise Ranking in Large Language Models. arXiv:2310.07712 [cs.CL]   
[40] Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2). https://openreview. net/forum?id=wCu6T5xFjeJ   
[41] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288 (2023).   
[42] Tu Vu, Mohit Iyyer, Xuezhi Wang, Noah Constant, Jerry Wei, Jason Wei, Chris Tar, Yun-Hsuan Sung, Denny Zhou, Quoc Le, et al. 2023. Freshllms: Refreshing large language models with search engine augmentation. arXiv preprint arXiv:2310.03214 (2023).

[43] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35 (2022), 24824–24837.   
[44] Minghao Wu and Alham Fikri Aji. 2023. Style Over Substance: Evaluation Biases for Large Language Models. arXiv:2307.03025 [cs.CL]   
[45] Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. arXiv preprint arXiv:2007.00808 (2020).   
[46] Qiantong Xu, Fenglu Hong, Bo Li, Changran Hu, Zhengyu Chen, and Jian Zhang. 2023. On the Tool Manipulation Capability of Open-source Large Language Models. arXiv preprint arXiv:2305.16504 (2023).   
[47] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629 (2022).

[48] Junjie Ye, Guanyu Li, Songyang Gao, Caishuang Huang, Yilong Wu, Sixian Li, Xiaoran Fan, Shihan Dou, Qi Zhang, Tao Gui, et al. 2024. Tooleyes: Fine-grained evaluation for tool learning capabilities of large language models in real-world scenarios. arXiv preprint arXiv:2401.00741 (2024).   
[49] Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Ren Kan, Dongsheng Li, and Deqing Yang. 2024. EASYTOOL: Enhancing LLM-based Agents with Concise Tool Instruction. arXiv preprint arXiv:2401.06201 (2024).   
[50] Wayne Xin Zhao, Jing Liu, Ruiyang Ren, and Ji-Rong Wen. 2023. Dense Text Retrieval based on Pretrained Language Models: A Survey. ACM Trans. Inf. Syst. (dec 2023).   
[51] Yuanhang Zheng, Peng Li, Wei Liu, Yang Liu, Jian Luan, and Bin Wang. 2024. ToolRerank: Adaptive and Hierarchy-Aware Reranking for Tool Retrieval. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING) (2024).   
[52] Mu Zhu. 2004. Recall, precision and average precision. Department of Statistics and Actuarial Science, University of Waterloo, Waterloo 2, 30 (2004), 6.
# Enhancing Long-Term Memory using Hierarchical Aggregate Tree for Retrieval Augmented Generation

Aadharsh Aadhithya $\mathbf { A } ^ { * }$ , Sachin Kumar $\mathbf { S } ^ { \ast }$ and Soman K.P∗

Amrita School of Artificial Intelligence,Coimbatore

Amrita Vishwa Vidyapeetham, India

# Abstract

Large language models (LLMs) have limited context capacity, hindering reasoning over long conversations. We propose the Hierarchical Aggregate Tree (HAT) memory structure to recursively aggregate relevant dialogue context through conditional tree traversals. HAT encapsulates information from children nodes, enabling broad coverage with depth control. We formulate finding best context as optimal tree traversal. Experiments show HAT improves dialog coherence and summary quality over baseline contexts, demonstrating the technique’s effectiveness for multi-turn reasoning without exponential parameter growth. HAT balances information breadth and depth for long-form dialogues. This memory augmentation enables more consistent, grounded long-form conversations from LLMs.

# 1 Introduction

Large language models (LLMs) like ChatGPT are having an impact across various areas and applications. One of the most straightforward applications is using LLMs as personalized chat agents. There have been several efforts to develop chatbots for various applications, both generic and domain-specific, particularly after the advent of LLMs and associated Pythonic libraries, which have made it very easy for people to develop their own chatbots.

Customizing and aligning these LLMs is still an active area of research. One basic alignment we would want is to make chatbots behave according to our expectations, particularly when context is specific and requires some information to be highlighted that is not necessarily in the model’s pretraining corpus. While LLMs are considered snapshots of the internet, one limitation is that

they are closed systems and providing external information to LLMs is an active research area.

Two primary ways of providing external data to LLMs are through: a) finetuning and b) retrievalaugmented generation (RAG). Finetuning requires access to model weights and can only be done with open models. RAG relies on strategies to retrieve information from a datastore given a user query, without needing internal model information, allowing it to be used with more model types. However, RAG is limited by the model’s context length budget. Although very large context LLMs exist, the budget remains limited. Hence, how and what data is retrieved given a user query is an important research task.With the advent of "LLM agents", a separate memory management module is often required. Some solutions train models for this task while others take a retrieval-based approach. Still others use the LLM itself to accomplish it. However, current approaches tend to rely solely on providing a summary versus retrieving from a datastore, with little in between. We hence propose a method that combines both worlds using a new data structure called the "Hierarchical Aggregate Tree".

# 1.1 Recent works

There has been growing interest in developing techniques to enhance LLMs’ capabilities for long-term multi-session dialogues. (Xu et al., 2021a) collected a dataset of human conversations across multiple chat sessions and showed that existing models perform poorly in maintaining consistency, necessitating long-context models with summarization and retrieval abilities. (Xu et al., 2021b) proposed a model to extract and update conversational personas over long-term dialogues. (Bae et al.) presented a task and dataset for memory management in long chats, dealing with outdated information. (Wang et al.) proposed recursive summarization

for equipping LLMs with long-term memory. (Lee et al.) showed prompted LLMs can match finetuned models for consistent open-domain chats. (Zhang et al.) encoded conversation history hierarchically to generate informed responses. Developing more sophisticated strategies for information integration remains an open challenge.

# 2 Task Definition

The Task at hand is straightforward. Given a history of conversations between a user and an assistant(system), we are to predict the response of the system. In other words, given the history of conversation at time step t, $H _ { t } =$ $\left\{ u _ { 1 } , a _ { 1 } , u _ { 2 } , a _ { 2 } \cdot \cdot \cdot \cdot a _ { t - 1 } \right\}$ (where $u _ { i }$ represents a user response and $a _ { i }$ represents assistant response) and a user query $u _ { t }$ , our task is to find a relevant function $f$ such that

$$
a _ {t} \approx L L M (u _ {t}, f (H _ {t} | u _ {t}))
$$

Where $f$ can be thought of as some function, mapping the entire history of conversation to a condensed space conditioned on the user query at time step t. Note that, $f$ can be a selection function, a trained neural network as a memory agent, or even a simple summary function. In our experiments, the dataset is organized as sessions and episodes. An episode consists of multiple consecutive dialogue sessions with a specific user, oftentimes requiring information from previous sessions to respond back. Hence in addition to the history $H$ , at the end of every session, we also have a memory $M _ { s } ^ { e }$ for session $s$ and episode $e$ , constructed by combining $H _ { s } ^ { e }$ and $M _ { s - 1 } ^ { e }$ , where $H _ { s } ^ { e }$ represents the history of session s and episode e. Therefore, we also have this auxiliary task of finding $M _ { s } ^ { e }$ given $M _ { s - 1 } ^ { e }$ and $H _ { s } ^ { e }$ .

# 3 Methedology

An overview of our methodology is depicted in Figure 1. The memory module is tasked to retrieve a relevant context that has necessary information to respond to user query. The module does so by leveraging 2 primary components:

1. Hierarchical Aggregate Tree (HAT) : We propose the use of a novel datastructure called HAT to manage memory, particularly in long form texts like open domain conversations. It has some important property like resolution

retention as detailed in section 3.1. At a highlevel, the intended characteristic of HAT is that we have more resolution as we move Top-Down and we have more latest information as we move Left-Right.

2. Memory Agent : The memory agent is tasked to find the best traversal in HAT, conditioned on the user query such that at the end of traversal, we are in a node that contains relevant context to answer the user query.

# 3.1 Hierarchical Aggregate Tree (HAT)

Hierarchical Aggregate Tree (HAT) is defined as $\begin{array} { r l r } { H S T } & { { } = } & { \left( L , M , A , \Sigma \right) } \end{array}$ , where, $L ~ = ~ \{ l _ { 0 } , l _ { 1 } , \ldots , l _ { n } \}$ is a finite set of layers, $M$ is the memory length, a positive integer, $A$ is an aggregation function and $\Sigma$ is a set of nodes. The layers in L are hierarchical, with $l _ { 0 }$ being the root layer. Each layer $l _ { i } \in L$ is a set of nodes $\Sigma _ { i }$ , where $\Sigma _ { i } \subseteq \Sigma$ . A node $\sigma \in \Sigma$ can have children nodes and contains a text element. A node recalculates its text property, whenever there is a change in the node’s children. The text property of a node $\sigma ~ \in ~ \Sigma _ { i } i \neq | L |$ is given by $A ( C ( \sigma ) )$ , where $C ( \sigma ) = \{ \tau \in \Sigma \mid \tau$ is a child of $\sigma \}$

# Aggregate Function

The aggregate function $A$ maps the text of child nodes to produce the text stored at a parent node.

$$
A: \mathcal {P} (\Sigma) \to T e x t
$$

Where, ${ \mathcal { P } } ( \Sigma )$ represents the power set of nodes. The exact implementation of $A$ can vary depending on the usecase. Figure 2 for example, depicts an HAT with concatenation as aggregate function. For our implementation, we use GPT as aggregate function.

It is important that the aggregate function should be designed to produce concise summaries reflecting key information from child nodes for a meaningful HAT. It executes whenever new child nodes are inserted to maintain consistency up the tree. For our implementation, we use GPT as aggregate function. We say GPT to summarize persona’s from the children’s text. The exact prompt given can be found in the appendix.

# Node

A node $\sigma \in \Sigma$ represents a single element in the HAT structure. Whenever the set of children nodes for $\sigma$ changes, the update_text()

![](images/9c80cdaf282793b9c64250079f47a89768e91d04acc123ee729b060355023cd7.jpg)  
Figure 1: Overview of our approach. Given a user query, the memory module is responsible to give a relevant context by traversing the HAT. The LLM then generates response for the user query, given the context.

![](images/1b96174daa257feacdc64f1a3038947be6981969c16953cf0f8739cc393ee04d.jpg)  
Figure 2: Illustration of HAT, with example aggregation function as simple concatenation and memory length of 2.

method is called to update $\sigma$ ’s text. This text is given by applying the aggregator function to the set of texts from the current child nodes. The previous aggregated texts of given different combinations of children are cached in previous_complete_state to enable reuse instead of recomputing.

After updating, $\sigma$ triggers the parent node to also update, thereby propagating changes upwards in the HAT structure.

Each node $\sigma$ contains the following components:

• id: A unique identifier   
• text: The node’s aggregated text content   
• previous_complete_state: A dictionary mapping hashes of the node’s children to the previously aggregated text when the node had those children   
• parent: The parent node in the HAT (None for the root node)   
• aggregator: The aggregation function $A$   
• children: The set of child nodes $C ( \sigma )$

The HAT datastructure satisfies the invariant that if $\sigma _ { k , i } \in \Sigma _ { k }$ is a child of $\tau _ { k - 1 , j } \in \Sigma _ { k - 1 }  j =$ $\lfloor { \frac { i } { M } } \rfloor$ , where $\sigma _ { k , i }$ is ith node in kth layer. This connects child nodes to parent nodes between layers based on the memory length, $M$ . When inserting a new node $\phi \in \Sigma _ { y }$ , the aggregation function $A$ is recursively applied to update ancestor nodes. That is, For all $\sigma _ { y - 1 , z } ~ \in ~ P ( \phi )$ that are parents of $\phi$ , $\sigma _ { y , z } . t e x t \ = \ A ( C ( \sigma _ { y , z } ) )$ , Where $C ( \sigma _ { y , z } ) = \{ \tau \in \Sigma | \tau$ is a child of $\sigma _ { y , z } \}$

This maintains the invariant while propagating updated information through the tree. The number of layers $| L |$ and nodes $| \Sigma |$ changes dynamically based on $M$ and node insertions. $| L |$ represents the depth of the tree.

For Brevity, we restrict further detailing on the datastructure. However, All necessary details to replicate the datastructure shall be given in appendix.

# 3.2 Memory Agent

The memory agent is tasked with finding the optimal traversal in HAT conditioned on a user query $q$ . This can be mathematically formulated as:

$$
a_{0:T}^{*} = \operatorname *{arg  max}_{a_{0:T}}R(s_{0:T},a_{0:T}|q)
$$

where $s _ { 0 : T }$ is the state sequence from the root node to a leaf node, $a _ { 0 : T }$ is the action sequence, and $R$ is the total reward over the traversal dependent on $q$ . Reward, in our case is the quality of response the model is giving. This essentially can be posed as a Markov Decision Process (MDP). The agent starts at the root node $s _ { 0 }$ and takes an action $a _ { t } \in \mathcal A$ at each time step, transitioning to a new state $s _ { t + 1 } \sim \mathcal { P } ( \cdot | s _ { t } , a _ { t } )$ . For cases like this, It is difficult to design a reward function, and we will require annotated training data for training the agent.

Hence, we resort to GPT and will ask GPT to act as a traversal agent. GPT is well-suited for conditional text generation, which allows it to traverse HAT by generating an optimal sequence of actions based on the text representation at each node and the user query.

The exact prompt used for the memory agent can be found in the Appendix.

The MDP is defined by the tuple $( S , { \mathcal { A } } , { \mathcal { P } } , { \mathcal { R } } , \gamma )$ , where:

• $\boldsymbol { \mathcal { S } }$ - set of tree nodes   
• $\mathcal { A } = \{ U , D , L , R , S , O , U \}$ - set of actions

– U - move up the tree   
– D - move down   
– L - move left   
– R - move right   
– $S$ - reset to root node   
– O - sufficient context for query $q$   
– $U$ - insufficient context for query $q$

• $\mathcal { P }$ - state transition probabilities   
• $\mathcal { R } : \mathcal { S } \times \mathcal { A }  \mathbb { R }$ - reward function   
• $\gamma \in ( 0 , 1 )$ - discount factor

GPT is well-suited for conditional text generation, which allows it to traverse HAT by generating an optimal sequence of actions based on the text representation at each node and the user query. But, It is important to note that our proposed framework is open and generic. The memory agent can be anything from neural network or RL Agent to an GPT Aproximation.

# 4 Experiments

# 4.1 Dataset

Table 1: Number of episodes across sessions.   

<table><tr><td>Data Type</td><td>Train</td><td>Valid</td><td>Test</td></tr><tr><td>Session 1</td><td>8939</td><td>1000</td><td>1015</td></tr><tr><td>Session 2</td><td>4000</td><td>500</td><td>501</td></tr><tr><td>Session 3</td><td>4000</td><td>500</td><td>501</td></tr><tr><td>Session 4</td><td>1001</td><td>500</td><td>501</td></tr><tr><td>Session 5</td><td>-</td><td>500</td><td>501</td></tr></table>

We use the multi-session-chat dataset from (Xu et al., 2022). The dataset contains two speakers

chat online in a series of sessions as is for example common on messaging platforms. Each individual chat session is not especially long before it is “paused”. Then, after a certain amount of (simulated) time has transpired, typically hours or days, the speakers resume chatting, either continuing to talk about the previous subject, bringing up some other subject from their past shared history, or sparking up conversation on a new topic. Number of episodes per session is on Table 1.

We utilize 501 episodes whose session 5 is available from the Test set.

# 4.2 Evaluation Metrics

We evaluate the dialogue generation performance of our model using automatic metrics. We report BLEU-1/2 , F1 score compared to humanannotated responses, and DISTINCT-1/2.

BLEU (Bilingual Evaluation Understudy) measures overlap between machine generated text and human references, with values between 0 to 1 (higher is better). We use BLEU-1 and BLEU-2 which compare 1-grams and 2-grams respectively. F1 score measures overlap between generated responses and human references. We report F1 to assess relevance of content. DISTINCT-1/2 quantifies diversity and vocabulary usage based on the number of distinct unigrams and bigrams in the generated responses, normalized by the total number of generated tokens (higher is better).

# 4.3 Baselines

We benchmark against three trivial methods: 1) All Context: LLM generated dialogues with all dialogues in context. 2) Part Context: LLM generates dialogues with only current session’s context. 3) Gold Memory: The LLM generated dialogues with Gold memory from the dataset as context. Further, We also evaluate different Traversal methods including BFS,DFS,and GPTAgent. In BFS and DFS, we follow naive BFS or DFS traversal, and at every step as gpt if this information is enough to answer the user question. If it tells okay, we stop there and return the context.

# 5 Results

Table 2 compares GPTAgent to breadth-first search (BFS) and depth-first search (DFS) traversal methods. Across BLEU-1/2 and DISTINCT-1/2 metrics, GPTAgent significantly outperforms both

Table 2: Dialogue generation comparison between traversal methods   

<table><tr><td></td><td>BLEU-1/2</td><td>DISTINCT-1/2</td></tr><tr><td>BFS</td><td>0.652 / 0.532</td><td>0.072 / 0.064</td></tr><tr><td>DFS</td><td>0.624 / 0.501</td><td>0.064 / 0.058</td></tr><tr><td>GPTAgent</td><td>0.721 / 0.612</td><td>0.092 / 0.084</td></tr></table>

in quality and diversity of dialogues. This supports our approach of learning to traverse conditioned on query relevance over hand-designed heuristics.

Next, Table 3 benchmarks GPTAgent against contexts with complete, partial or gold dialogue history. GPTAgent achieves highest scores, demonstrating the benefit of our focused memory retrieval. Access to full history or gold references improves over just current context, but lacks efficiency of precisely identifying relevant information. Finally, Table 4 evaluates fidelity of memories generated by GPTAgent compared to dataset ground truth references. We again see strong results surpassing 0.8 on both word overlap and diversity measures.

Table 3: Dialogue generation comparison between baselines   

<table><tr><td></td><td>BLEU-1/2</td><td>DISTINCT-1/2</td></tr><tr><td>All Context</td><td>0.612 / 0.492</td><td>0.051 / 0.042</td></tr><tr><td>Part Context</td><td>0.592 / 0.473</td><td>0.043 / 0.038</td></tr><tr><td>Gold Memory</td><td>0.681 / 0.564</td><td>0.074 / 0.064</td></tr><tr><td>GPTAgent</td><td>0.721 / 0.612</td><td>0.092 / 0.084</td></tr></table>

Table 4: Memory generation scores   

<table><tr><td></td><td>BLEU-1/2</td><td>DISTINCT-1/2</td><td>F1</td></tr><tr><td>GPT</td><td>0.842 / 0.724</td><td>0.102 / 0.094</td><td>0.824</td></tr></table>

In summary, experiments validate effectiveness of our method in extracting salient dialogue context in long form conversations. Both conversations and summarized memories demonstrate quality and relevance gains over alternate approaches. The query conditioning provides efficiency over exhaustive history while retaining enough specificity for the current need.

# 6 Limitations

While the proposed method has a potential and could work with long-form texts, The current implementation takes longer than a usual time taken by a dialogue agent to respond. Also, Making

HTTP API calls to gpt, is causing an additional overhead on the time taken. These limitations could be overcome, by turning to heuristic-based tree searches or Monte-Carlo Tree search like methods in the future. Further, A Coupled-HAT : One HAT with textual information and another HAT with dense vector representation, would be more efficient. Combined with hybrid retrieval techniques, we could have a much more efficient way of doing conditional retrival. Further, Another limitation of this kind of Retrieval system is that, As the leaf nodes expands exponentially, the memory footprint might become larger than expected. Several optimizations on this front, also could be potential future work in this direction.

# 7 Conclusion

In this work, we have presented the Hierarchical Aggregate Tree (HAT) - a new data structure designed specifically for memory storage and retrieval for long form text based conversational agents. Rather than solely providing a summary or retrieving raw excerpts, our key innovation is recursive aggregation of salient points by traversing conditional paths in this tree.We formulate the tree traversal as an optimization problem using a GPT-based memory agent. In conclusion,Our Experiments demonstrate significant gains over alternate traversal schemes and baseline methods. HAT introduces a flexible memory structure for dialogue agents that balances extraction breadth versus depth through hierarchical aggregation. Our analysis confirms the viability and advantages of conditional traversal over existing limited budget solutions, opening up further avenues for augmented language model research.

# References

Sanghwan Bae, Donghyun Kwak, Soyoung Kang, Min Young Lee, Sungdong Kim, Yuin Jeong, Hyeri Kim, Sang-Woo Lee, Woomyoung Park, and Nako Sung. Keep me updated! memory management in long-term conversations.   
Gibbeum Lee, Volker Hartmann, Jongho Park, Dimitris Papailiopoulos, and Kangwook Lee. Prompted LLMs as chatbot modules for long open-domain conversation. In Findings of the Association for Computational Linguistics: ACL 2023, pages 4536–4554.   
Qingyue Wang, Liang Ding, Yanan Cao, Zhiliang Tian, Shi Wang, Dacheng Tao, and Li Guo. Recursively summarizing enables long-term dialogue memory in large language models.

Jing Xu, Arthur Szlam, and Jason Weston. 2021a. Beyond goldfish memory: Long-term open-domain conversation. arXiv preprint arXiv:2107.07567.   
Jing Xu, Arthur Szlam, and Jason Weston. 2022. Beyond goldfish memory: Long-term open-domain conversation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5180–5197, Dublin, Ireland. Association for Computational Linguistics.   
Xinchao Xu, Zhibin Gou, Wenquan Wu, Zheng-Yu Niu, Hua Wu, Haifeng Wang, and Shihang Wang. 2021b. Long time no see! open-domain conversation with long-term persona memory.   
Tong Zhang, Yong Liu, Boyang Li, Zhiwei Zeng, Pengwei Wang, Yuan You, Chunyan Miao, and Lizhen Cui. History-aware hierarchical transformer for multi-session open-domain dialogue system.
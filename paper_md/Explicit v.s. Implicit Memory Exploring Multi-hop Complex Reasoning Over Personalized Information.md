# Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over Personalized Information

Zeyu Zhang1, Yang Zhang2, Haoran Tan1, Rui Li1, Xu Chen1

1Gaoling School of Artificial Intelligence, Renmin University of China, China

2National University of Singapore, Singapore

{zeyuzhang,xu.chen}@ruc.edu.cn,zyang1580@gmail.com

# Abstract

In large language model-based agents, memory serves as a critical capability for achieving personalization by storing and utilizing users’ information. Although some previous studies have adopted memory to implement user personalization, they typically focus on preference alignment and simple question-answering. However, in the real world, complex tasks often require multi-hop reasoning on a large amount of user information, which poses significant challenges for current memory approaches. To address this limitation, we propose the multi-hop personalized reasoning task to explore how different memory mechanisms perform in multi-hop reasoning over personalized information. We explicitly define this task and construct a dataset along with a unified evaluation framework. Then, we implement various explicit and implicit memory methods and conduct comprehensive experiments. We evaluate their performance on this task from multiple perspectives and analyze their strengths and weaknesses. Besides, we explore hybrid approaches that combine both paradigms and propose the HybridMem method to address their limitations. We demonstrate the effectiveness of our proposed model through extensive experiments. To benefit the research community, we release this project at https://github.com/nuster1128/MPR.

# Keywords

Personalized Agent, Memory Mechanism, Multi-hop Reasoning, Large Language Model, Information Retrieval

# ACM Reference Format:

Zeyu Zhang1, Yang Zhang2, Haoran Tan1, Rui $\mathrm { L i } ^ { 1 }$ , Xu Chen1. 2025. Explicit v.s. Implicit Memory: Exploring Multi-hop Complex Reasoning Over Personalized Information. In . ACM, New York, NY, USA, 15 pages. https: //doi.org/10.1145/nnnnnnn.nnnnnnn

# 1 Introduction

With the rapid advancement of large language models (LLMs) [1, 18, 45], LLM-based agents have been widely applied for providing personalized services [9, 30, 36]. Specifically, memory serves as a key component for achieving personalized agents, responsible for storing and utilizing personalized information to meet

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

Conference’17, Washington, DC, USA

$\circledcirc$ 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 978-x-xxxx-xxxx-x/YYYY/MM

https://doi.org/10.1145/nnnnnnn.nnnnnnn

users’ requirements [42]. Although previous works have proposed various memory approaches to improve personalization, most of them focus on user preference alignment tasks [22, 27] and simple question-answering [7]. These tasks do not demand explicit reasoning processes by aggregating users’ factual information. However, in real-world personalized applications, complex tasks that require multi-hop reasoning over a large amount of personalized information are more practical and challenging for the memory of agents, which still remains unexplored. To fill this gap, we focus on multihop personalized reasoning (MPR) tasks in this study.

MPR tasks are characterized by two key features: (1) tasks must be completed based on given personalized information and cannot be accomplished solely through general knowledge; (2) tasks can not be directly accomplished with a single piece of personalized information, but require multi-hop reasoning on several pieces. We demonstrate an example of MPR tasks in Figure 1(a), and a formal definition will be provided in later sections.

We emphasize that MPR tasks are significantly different from previous personalization tasks, as illustrated in Figure 1(b). First, most previous tasks keep consistent distributions between training data (i.e., user histories) and testing data (i.e., predictions) that are drawn from the same distribution of user utterances [22]. However, each MPR task requires multiple pieces of user information, leading to composition gaps between training and testing. Besides, previous tasks do not emphasize reasoning processes, whereas MPR tasks necessitate multi-hop reasoning over massive personalized information to accomplish. Finally, user histories in MPR tasks consist of factual information for complex reasoning, rather than preference or stylistic information [13, 22].

Like other personalization tasks, users’ personalized information should be memorized by LLM-based agents to accomplish MPR tasks. According to previous works, agent memory can be categorized into explicit and implicit forms based on their representation [42]. Explicit memory refers to storing users’ personalized information in textual form and utilizing it through in-context learning (ICL) [6], typically implemented through Retrieval-Augmented Generation (RAG) methods [8]. Implicit memory refers to storing users’ personalized information within model parameters, typically implemented through supervised fine-tuning (SFT) methods [41]. However, considering the multi-hop reasoning feature of MPR tasks, explicit memory may suffer from retrieval mismatch problems between reasoning steps, while implicit memory may face difficulties in accurately storing large amounts of factual personalized information. Therefore, how explicit and implicit memory perform on MPR tasks represents a valuable but unexplored question.

In this paper, we focus on exploring explicit and implicit memory on MPR tasks. We formally define MPR tasks and construct datasets

# Question: Which city does Alice's husband work in?

# Reasoning Process

![](images/fbd1d149bbe282aeeab0d1f8e9e20454f010914ee7527a6fdc8e196b4d76764f.jpg)

Feature 1: Dependent on personalized information. Feature 2: Require multi-hop reasoning to accomplish.

Answer: Alice's husband works in New York.

# (a) Multi-hop Personalized Reasoning Tasks

![](images/e645880dc3647e0e518cef26952dd9753016f901fbc590cfab19b7d1d779e86c.jpg)

![](images/70c396f90fb07aa54c18a639d66cdd166587d783b2c8f22550b532c616f716b8.jpg)

# Personalized Information

- Alice works as a teacher in Boston.   
- Alice's husband is Bob. ①   
- Alice and Bob got married three years ago.

- Bob is David's department leader. ②

- Bob graduated from MIT in 2015.   
- Bob's favorite restaurant is in downtown Manhattan.

……   
- David's department is located in New York. ③   
- David's department was previously located in Chicago.

# (b) Previous Personalization Tasks

User History

$$
H ^ {u} = \left\{\left(x _ {1} ^ {u}, y _ {1} ^ {u}\right), \dots , \left(x _ {n} ^ {u}, y _ {n} ^ {u}\right) \right\}
$$

New Query ????????

Response $\hat { y } ^ { u } = f ( x ^ { u } , H ^ { u } )$

Limitations:

- Consistent distribution between $x ^ { u }$ and $H ^ { u }$ .   
- Do not require multi-hop reasoning over personalized information.

# Personalized Text Classification

Citation Identification

![](images/17f04ac4565576edcf94fc130338c2e4090e5d79813ebfdc24d99383b1f3e8c0.jpg)

????????: Academic Paper from ????

????????: Reference to Cite by ????

Product Rating

![](images/d74dc877ea02b2251cf097852a004f3dd51117aa50acabe651ed331349fab637.jpg)

$x ^ { u }$ : Product Review from ????

$y ^ { u }$ : Product Rating of ????

# Personalized Text Generati

News Headline Generation

![](images/da62e0436296df67cee7273244e2a9f34f2e761ac3620b3fe7600460efb87a40.jpg)

Single(Short)-hop QA

![](images/571ac76b02a7d21d163099548c3f714eccf3a884469f0b4641dfdb2e5a5e7cdb.jpg)

????????: Query from ????

????????: Answer by ????

Example:

Alice is 24 years old.

Q: How old is Alice?

A: 24 years old.

![](images/4accc6b508e4287d81a4f2ea6a59ea4e80d1ce8699bc22e358e75f04cf63075e.jpg)  
Figure 1: Demonstration of multi-hop personalized reasoning tasks and previous personalization tasks.

along with an evaluation framework for memory mechanisms in multi-hop reasoning. We comprehensively study explicit memory, implicit memory, and hybrid memory on these tasks from multiple perspectives, drawing several key findings. We find that explicit memory demonstrates clear advantages across various reasoning structures, and incorporating implicit memory can enhance explicit memory. Additionally, we discover that reasoning structure significantly affects performance across all memory types. Finally, we propose a simple yet effective hybrid memory approach to improve the performance of agent reasoning on long-hop tasks. To benefit the research community, we release our code at the GitHub repository https://github.com/nuster1128/MPR.

Our major contributions are summarized as follows:

• We identify the MPR tasks, highlighting their unique challenges for agent memory compared to previous works.   
• We formally define MPR tasks and construct a new dataset with a unified evaluation framework for systematically exploring different memory methods on MPR tasks.   
• We conduct comprehensive experiments on explicit, implicit, and hybrid memory approaches, presenting key findings and proposing a new hybrid memory method for long-hop reasoning tasks.

# 2 Related Work

# 2.1 Personalized LLM-based Agents

Recently, LLM-based agents have been extensively applied to user personalization tasks for enhancing user experiences. Many previous works focus on aligning LLMs with user preferences to achieve user personalization [14]. For instance, LaMP [22] constructs a personalized benchmark to evaluate models’ ability to infer subsequent user responses given user historical utterances. CFRAG [24] incorporates users’ previous documents in textual form through RAG along with collaborative information to improve personalization.

OPPU [27] fine-tunes a LoRA adapter for each user based on previous utterances to empower personalization. Besides, some studies achieve agent personalization by delivering personalized services. For example, PerLTQA [7] constructs personalized questions to evaluate agents’ long-term memory and proposes a retrieval-based memory framework. However, these tasks do not consider the multi-hop reasoning complexity in real-world tasks, and pay less attention to the distributional mismatch between personalized information and tasks. Therefore, in this paper, we intend to highlight the importance of exploring MPR tasks.

# 2.2 Memory of LLM-based Agents

Memory is a crucial capability of LLM-based agents, responsible for storing historical information and utilizing relevant content to support decision-making [42]. It is typically categorized into explicit and implicit forms. Explicit memory stores and utilizes information in textual form. For example, MemoryBank [46] stores past conversations as text and retrieves semantically relevant information using retrieval models. In contrast, implicit memory stores and utilizes information within model parameters. For instance, MEND [17] leverages meta-learning to train a model that transforms knowledge into parameter adjustments. In addition, some previous works also focus on agent memory evaluation [26, 34, 43] and development [4, 44], particularly in long-term scenarios. Agents can accomplish user modeling by designing memory mechanisms to capture users’ preferences [42]. This enables the personalized agents to avoid the "one-size-fits-all" phenomenon [2].

# 2.3 Multi-hop Reasoning in Agents

Reasoning is important for LLM-based agents to accomplish complex tasks, enabling them to decompose one task into multiple steps, thereby reducing the difficulty of inference [12]. Chain-of-thought (CoT) [33] is a representative work that enhances the reasoning capabilities of LLMs by adding instructions to make the model think

step-by-step. After that, many different structures have been proposed to implement reasoning processes, such as Tree-of-Thought (ToT) [39] and divide-and-conquer approaches [31]. Some previous works have also constructed multi-hop datasets to evaluate agents’ reasoning on general tasks [23, 29, 48]. We highlight that MPR tasks are different from previous open-ended reasoning tasks. First, the information in MPR tasks is user-specific and cannot be obtained in advance (e.g., from Wikipedia), avoiding reasoning shortcuts. Moreover, their difficulties depend not only on the problems themselves, but also on the available reasoning evidence, which is controllable.

# 3 Preliminaries

# 3.1 Problem Definition

First of all, we deliver a formal definition of MPR tasks as follows: Definition 1 (MPR Task). Given a collection of personalized statements $\boldsymbol { S } = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { n } \}$ , where $s _ { i } ( 1 \leq i \leq n )$ describes a single-hop factual information from the user, the model $f$ is required to predict an answer $\hat { a } = f ( q ; S )$ to a question $q$ based on $s$ , in order to match the correct answer ??. Meanwhile, the tasks should satisfy:

(1) $\exists S \subseteq S ( S \neq \emptyset )$ such that $S$ is necessary and sufficient to infer the correct answer $a$ to question $q$ .   
(2) $\nexists s _ { i } \in S$ such that the correct answer ?? to question $q$ can be obtained solely from $s _ { i }$ without other statements.

These two conditions correspond to the features of MPR tasks and serve as important criteria that distinguish them from other tasks. For instance, for the question "Which city does Alice’s husband work in?" in Figure 1(a), we can only obtain the answer from the user’s statement, rather than relying on general knowledge. Furthermore, the user’s statement does not contain a direct answer such as "Alice’s husband works in New York." Instead, the answer must be derived through reasoning across multiple statements.

# 3.2 Solution Paradigm

The most naive approach for MPR tasks is to integrate the entire statement collection $s$ and question $q$ into a prompt, and let the LLM directly infer the answer, that is,

$$
\hat {a} = f (q; \mathcal {S}) = g \left(s _ {1} \| s _ {2} \| \dots \| s _ {n} \| q; \theta\right),
$$

where the function $g ( x ; \theta )$ represents an LLM inference for prompt $x$ with parameter $\theta$ , and $\parallel$ denotes string concatenation. However, this naive approach has two obvious limitations. First, due to the large scale of users’ personalized information, it is almost impossible to integrate all these statements in the prompt. Therefore, one solution is to rely on RAG to select a subset ${ \hat { S } } \subseteq S$ as the explicit memory, and we have

$$
\hat {a} = f (q; \mathcal {S}) = g (\hat {s} _ {1} \| \hat {s} _ {2} \| \dots \| \hat {s} _ {k} \| q; \theta),
$$

$$
\hat {S} = \left\{\hat {s} _ {1}, \hat {s} _ {2}, \dots , \hat {s} _ {k} \right\} = \operatorname {R A G} (q, S),
$$

where $k$ is the number of retrieved statements. Another approach is to convert the information into parameter modifications as implicit memory through SFT, and we have

$$
\hat {a} = f (q; \mathcal {S}) = g (q; \hat {\theta}),
$$

$$
\hat {\theta} = \operatorname {S F T} (\theta ; S),
$$

where $\hat { \theta }$ is a personalized parameter under $s$ .

Another problem is that question $q$ requires multi-hop reasoning to solve, so single-step inference $g ( x ; \theta )$ of LLMs may not be able to predict the answer effectively. Therefore, it is necessary to

Table 1: The comparison of related datasets. KR: knowledgeintensive reasoning. PK: private knowledge. PE: personalized evidence of factual information. SR: supporting reference. EC: explicit chain of reasoning.   

<table><tr><td>Dataset</td><td># Hop</td><td>KR</td><td>PK</td><td>PE</td><td>SR</td><td>EC</td><td># QA</td></tr><tr><td>MoreHopQA [23]</td><td>≤ 5</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td><td>✓</td><td>3,620</td></tr><tr><td>HybridQA [3]</td><td>≤ 5</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td><td>✗</td><td>69,611</td></tr><tr><td>2WikiMultiHopQA [10]</td><td>≤ 5</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td><td>✓</td><td>192,606</td></tr><tr><td>MuSiQue [29]</td><td>≤ 5</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td><td>✓</td><td>24,814</td></tr><tr><td>FanOutQA [48]</td><td>~ 10</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td><td>✓</td><td>1,035</td></tr><tr><td>HotpotQA [38]</td><td>≤ 5</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td><td>✗</td><td>112,779</td></tr><tr><td>MultiHop-RAG [28]</td><td>≤ 5</td><td>✓</td><td>✗</td><td>✗</td><td>✓</td><td>✗</td><td>2,556</td></tr><tr><td>PerLTQA [7]</td><td>≤ 5</td><td>✓</td><td>✓</td><td>✗</td><td>✓</td><td>✗</td><td>8,593</td></tr><tr><td>LongMemEval [34]</td><td>≤ 5</td><td>✓</td><td>✓</td><td>✗</td><td>✓</td><td>✗</td><td>500</td></tr><tr><td>CoFCA [35]</td><td>≤ 5</td><td>✓</td><td>✓</td><td>✗</td><td>✓</td><td>✓</td><td>4,500</td></tr><tr><td>MQuAKe [47]</td><td>≤ 5</td><td>✓</td><td>✓</td><td>✗</td><td>✓</td><td>✓</td><td>9,218</td></tr><tr><td>MPR (Ours)</td><td>~ 10</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>108,000</td></tr></table>

implement question processing and answering with multi-hop reasoning structures. CoT is a commonly used approach for multi-hop reasoning. Formally, we have the following iterative processes:

$$
o _ {1} = g _ {1} \left(s _ {1} \| s _ {2} \| \dots \| s _ {n} \| q; \theta\right),
$$

$$
o _ {i} = g _ {i} \left(s _ {1} \| s _ {2} \| \dots \| s _ {n} \| q \| o _ {1} \| o _ {2} \| \dots \| o _ {i - 1}; \theta\right), 2 \leq i \leq l - 1,
$$

$$
\hat {a} = f (q; \mathcal {S}) = g \left(s _ {1} \| s _ {2} \| \dots \| s _ {n} \| q \| o _ {1} \| o _ {2} \| \dots \| o _ {l - 1}; \theta\right),
$$

and can be replaced by explicit memory and implicit memory:

$$
\hat {S} _ {i} = \{\hat {s} _ {1} ^ {i}, \hat {s} _ {2} ^ {i}, \dots , \hat {s} _ {k} ^ {i} \} = \operatorname {R A G} (o _ {1} \| o _ {2} \| \dots \| o _ {i - 1} \| q, S),
$$

$$
\hat {\theta} = \operatorname {S F T} (\theta ; S).
$$

Therefore, the memory problem and reasoning problem correspond to the two major challenges of MPR tasks, that is, how to store and retrieve knowledge, and how to use knowledge for inference. They are also the core focus of our subsequent experimental exploration.

# 3.3 Dataset Construction

3.3.1 Previous Datasets and Benchmarks. There is still a lack of comprehensive datasets to evaluate MPR tasks. Some multi-hop question-answering (QA) datasets provide reasoning problems, but most of them rely on public knowledge such as Wikipedia, rather than personalized information [38]. This poses risks of reasoning shortcuts and information leakage in pre-training corpora. Although some datasets construct personalized information for QAs, their questions typically contain 1 or 2 hops [7]. They do not provide explicit reasoning chains, and the number of QAs is limited. Therefore, we propose to construct a new MPR dataset to better evaluate MPR tasks, which contains 108,000 personalized reasoning tasks ranging from 2 to 10 hops, and provides explicit reasoning paths with reference evidence. We compare our MPR Dataset with previous datasets from multiple perspectives in Table 1.

3.3.2 MPR Dataset Generation. First, we construct a meta graph to represent the graph space before instantiating specific graphs. Then, we generate user statements from edges, and construct multihop questions through path sampling on the graph, with answers assigned by path endpoints. The process is presented in Figure 2(a).

Step 1: Generate Meta Graph. In order to improve user diversity, we define a meta graph as the sampling space for users. Formally, we denote the meta graph as $\mathcal { G } = ( \mathcal { V } , \mathcal { E } )$ , where $_ \mathrm { ~  ~ }$ and E represent the node set and edge set respectively. In meta graph $\mathcal { G }$ , each node $V _ { i } \in \mathcal { V } ( 1 \leq i \leq | \mathcal { V } | )$ represents an conceptual space

for a type of entity or attribute (e.g., persons), containing specific entities or attribute values (e.g., Alice) that can be sampled, that is, $V _ { i } = \{ v _ { 1 } ^ { i } , v _ { 2 } ^ { i } , . . . , v _ { | V _ { i } | } ^ { i } \}$ . Each edge $E _ { i } \in \mathcal { E } ( 1 \leq i \leq | \mathcal { E } | )$ represents a relation space (e.g., social relations) between two nodes, including specific relationships (e.g., be supervised by) that can be instantiated, that is, $E _ { i } = \bar { \{ { e _ { 1 } ^ { i } , e _ { 2 } ^ { i } , . . . , e _ { | E _ { i } | } ^ { i } \} } }$ . According to the type of two connected nodes (entity or attribute), we categorize all these edges into three categories. Entity-oriented edges connect two entities, representing the relationship between them. Attribute-oriented edges connect an entity and an attribute, indicating that the entity possesses a value of this attribute. Value-oriented edges connect two attributes, representing the comparative relationship between them. The first two types of edges involve reasoning with personalized information, while the last type relies on general knowledge, which is designed to enhance the diversity of reasoning.

Step 2: Instantiate Specific Graphs. Then, we sample attribute values of nodes using LLMs, and simultaneously instantiate edges. We organize all the sampled nodes $\tilde { V } = \{ \tilde { v } _ { i } | \tilde { v } _ { i } \in V _ { i } , 1 \leq i \leq | \bar { \mathcal { V } } | \}$ and edges $\tilde { E } = \{ \tilde { e } _ { i } | \tilde { e } _ { i } \in E _ { i } , 1 \le i \le | \mathcal { E } | \}$ as a specific graph $\tilde { G } =$ $( \tilde { V } , \tilde { E } )$ , which serves as the prior knowledge for constructing MPR tasks for a certain user. We can generate multiple specific graphs $\tilde { G }$ based on the meta graph, thereby constructing different users.

Step 3: Collect User Statements. After that, we rewrite each non-value-oriented edge in graph $\tilde { G }$ to obtain textual statements for users. Specifically, we instantiate the information from edges and their connected nodes into prompts, and leverage LLMs to generate user statements $\boldsymbol { S } = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { | S | } \}$ , which serve as single-hop personalized information for users.

Step 4: Sample Reasoning Paths. We sample a reasoning path $T = \left[ t _ { 1 } , t _ { 2 } , . . . , t _ { k } \right]$ on graph $\tilde { G }$ as the reference for constructing $k$ - hop $k \geq 2 )$ questions. Specifically, we randomly select an initial edge $t _ { 1 } = \langle p _ { 0 } , p _ { 1 } \rangle \in \tilde { E }$ as the first edge, and use its endpoint $\mathcal { P } 1$ as the starting node to sample the next edge $t _ { 2 } = \langle p _ { 1 } , p _ { 2 } \rangle \in \tilde { E }$ from its adjacent edges. To prevent ambiguity in attribute-oriented edges (i.e., multiple entities possessing the same attribute value), we implement the disambiguation mechanism. When such an ambiguous edge $t _ { i } = \langle p _ { i - 1 } , p _ { i } \rangle$ occurs, we iteratively sample other attributeoriented edges from the target entity $\mathbf { \nabla } \ p _ { i - 1 }$ until no ambiguity exists. We iterate for $k$ steps to construct a $k$ -hop reasoning path $T$ and ensure that personalized edges are major among the edges.

Step 5: Derive Questions and Answers. Based on the reasoning path $T = [ t _ { 1 } , t _ { 2 } , . . . , t _ { k } ]$ , we get the user statement $m _ { i }$ corresponding to the edge $t _ { i }$ , obtaining the references of user statements $M = [ m _ { 1 } , m _ { 2 } , . . . , m _ { k } ]$ $( m _ { i } \in S , 1 \leq i \leq k )$ ). Then, we instruct LLMs to rewrite them as a multi-hop reasoning question $q$ , whose answer is the endpoint of ?? . Finally, we construct the user statements $s$ as the training or retrieval corpus, and formulate quadruples $\mathcal { D } = \{ ( T _ { i } , M _ { i } , q _ { i } , a _ { i } ) \} _ { i = 1 } ^ { m }$ to represent $m$ testing tasks. To enhance the diversity of users for evaluation, we sample ?? training sets and testing sets (i.e., sub-datasets) as final dataset $\mathcal { H } = \{ ( S _ { i } , \mathcal { D } _ { i } ) \} _ { i = 1 } ^ { n }$ .

3.3.3 MPR Dataset Statistics. The MPR dataset contains a total of 10,800 multi-hop reasoning QA tasks, covering questions ranging from 2-hop to 10-hop to explicitly represent different reasoning difficulties. For each user, we construct over 13,000 user statements as personalized information, and build 1,000 QA tasks for each hop count ranging from 2 to 9. In addition, the MPR dataset contains

Table 2: The statistics of the MPR datsest.   

<table><tr><td>Statistic</td><td>Number</td><td>Statistic</td><td>Number</td></tr><tr><td>Statement (Total)</td><td>157,166</td><td>QA (Total)</td><td>108,000</td></tr><tr><td>Statement (Piece/User)</td><td>13,097±6.34</td><td>QA (Each User)</td><td>9,000</td></tr><tr><td>Statement (Word/User)</td><td>104,376±846</td><td>QA (Hop)</td><td>2~10</td></tr><tr><td>Graph Node (Each User)</td><td>5,902±43.1</td><td>Entity Type</td><td>108</td></tr><tr><td>Graph Edge (Each User)</td><td>89,586±59,259</td><td>Relation Type</td><td>34</td></tr></table>

12 different specific graphs $\tilde { G }$ . More detailed statistics of the MPR dataset are available in Table 2.

# 3.4 Experimental Settings

3.4.1 Overview. The evaluation pipeline of MPR tasks is demonstrated in Figure 2(b). First, for each sub-dataset $( S _ { i } , \mathcal { D } _ { i } )$ , $1 \leq i \leq n$ we provide the collection of user statements $S _ { i }$ to the memory model of agents, and allow it to perform preprocessing, such as building indices or conducting fine-tuning. Then, we conduct the evaluation on questions in $\mathcal { D } _ { i }$ . Specifically, for each testing point $\left( T , M , q , a \right) \in \mathcal { D } _ { i }$ , we only present the question $q$ to agents to obtain a predicted answer $\hat { a }$ by multi-hop reasoning according to $s _ { i }$ . At this point, the reasoning path ?? , references $M$ , and answer ?? are considered as ground-truths that are invisible to the agents. Finally, we compare the predicted answer $\hat { a }$ with the ground-truth answer ?? after normalization, and calculate the average accuracy (ACC) using Exact Match (EM) [20] as follows:

$$
\mathrm {A C C} = \frac {1}{| \mathcal {H} |} \sum_ {\left(\mathcal {S} _ {i}, \mathcal {D} _ {i}\right) \in \mathcal {H}} \frac {1}{| \mathcal {D} _ {i} |} \sum_ {\left(T _ {j}, M _ {j}, q _ {j}, a _ {j}\right) \in \mathcal {D} _ {i}} \mathrm {E M} \left[ f \left(q _ {j}; \mathcal {S} _ {j}\right), a _ {j} \right],
$$

where $f$ represents the evaluated model (i.e., LLM-based agent). In order to enable evaluating more aspects at a finer granularity, we select 2,700 questions (uniformly across 2-hop to 10-hop) from the entire dataset as a sub-dataset to reduce computational overhead. Our experiments are consistently conducted on a server equipped with 8 NVIDIA A800-SXM-80G GPUs. To control for confounding factors, we use Qwen2.5-7B [37] as our base model for all experiments except exploring different model sizes.

3.4.2 Memory Mechanisms. Our study primarily explores the performance of explicit memory and implicit memory in MPR tasks. Therefore, we respectively utilize explicit, implicit, and hybrid memory methods to store and utilize user statements, as illustrated in Figure 2(c). For explicit memory, we store user statements in textual form and build indices during the training phase, and retrieve based on the current query when testing. We commonly set the retrieval count to $k = 2 0$ (i.e., top-20 statements), adopt BM25s [16] for all sparse retrievals, and utilize e5-base-v2 [32] for dense retrievals. For implicit memory, we fine-tune the base model according to user statements during training, and perform inference using the fine-tuned model. Specifically, we adopt LoRA [11] to conduct finetuning and set the LoRA rank to 8 and alpha to 32. We tune the training epochs for all methods ranging from 1 to 10. For hybrid memory, we store and build indices while fine-tuning the model during the training phase, and utilize the trained model with retrieved information for testing. The detailed baselines will be described and discussed in the later sections.

3.4.3 Reasoning Structures. Since MPR tasks require multi-hop reasoning, we implement several different reasoning structures to

![](images/d56b4c58f763625fb2515741cfb546649cd46eaf2b6853990a9082c8d1c3e5a1.jpg)  
(a) Dataset Generation

![](images/dc39bbb682aa15bea7b44551275cc29000e6a7cdc360317d8fb7ac8b41adbe55.jpg)  
(b) Evaluation Pipeline

![](images/4ce099998ee4f8c810d8e2372477f8c6b7a449d3eaf23b989ca668ee120ba7f2.jpg)  
Explicit Memory

![](images/5a6c8cd6747364ad282de000d4c08157cf069443717504b867a9bba1be4ff4b1.jpg)

![](images/9933b78673141020fbde58ba22bb2c955802453590843dcfa7c38a6f5405d119.jpg)  
Implicit Memory

![](images/105a9cc7d85ff7915f1b0845502045466377372f3e2919f5542bb82f4ed4f2f8.jpg)  
AskSFT How old is Alice? 26 years old.

![](images/02bddb45f34ce2e8e461fb769286283091e54b586de14a53dee2b5bf42707d70.jpg)  
(c) Memory Mechanisms   
Naive Reasoning (NR)

![](images/48b2352b80b42b922f4bc56ccc87f66203e015f2ec9faa40de2cbed4ab06b5d4.jpg)  
Multi-path Reasoning (MR)

![](images/61e4fc81534f642a956024f9d4c5dd2f67df5743f3f4c97b5a6394f748977f64.jpg)

![](images/437f55dfcb3d0bdfe9836ddefd1253458eb1f19e4c78623cdec9866c25893a5a.jpg)  
Decomposition Reasoning (DR)   
(d) Reasoning Structures

![](images/2d1c09aa31f570a6bdf582bbbc27c15664264ff00ae206dd7149facad737d99a.jpg)  
Figure 2: Overview of exploration on multi-hop personalized reasoning tasks.   
Figure 3: Overall performances of explicit memory, with mean values (line) and standard deviation values (shading). better explore the performance of memory under different reasoning strategies, as shown in Figure 2(d) and described as follows:

• Naive Reasoning (NR): the vanilla reasoning method that obtains the predicted answer through only a single-step LLM inference, without using multi-hop reasoning structures.   
• Sequential Reasoning (SR): the chain-like reasoning where each inference is based on the results from the last step, forming a sequence of multiple LLM inferences, implemented by CoT [33].   
• Multi-path Reasoning (MR): the extension of sequential reasoning that maintains multiple chains and chooses one for extended reasoning at each inference step, implemented by ToT [39].   
• Decomposition Reasoning (DR): the divide-and-conquer structure [31] that decomposes a task into multiple sub-tasks, then processes them and combines their results to infer the final answer.

For different reasoning structures, we carefully design the reasoning prompts while ensuring consistent basic content. More details are available in Appendix B. We set the number of multi-hop reasoning steps to 5, except for experiments exploring the impact of reasoning steps. For the Multi-path Reasoning structure, we set the number of augmented branches to 2 at each step.

# 4 Explicit Memory on MPR Task

# 4.1 Overview

First of all, we explore the performance of explicit memory for LLMbased agents on MPR tasks. To ensure fair evaluation, we adopt

the general RAG pipeline [8], which retrieves relevant statements based on the current query (or reasoning state) and integrates them into the prompt to facilitate LLM inference. The baselines of RAG methods for explicit memory are as follows:

• SparseRAG: an unstructured RAG method that represents statements and queries as sparse vectors based on TF-IDF [25], leveraging exact keyword matching and statistical term importance.   
• DenseRAG: an unstructured RAG method that utilizes neural embedding models [32] to encode statements and queries into dense vectors, employing cosine similarity to get contextual relevance.   
• TreeRAG: a tree-structured RAG method that represents each statement as a leaf node and considers parent nodes as summaries of their child nodes hierarchically, implemented as MemTree [21].   
• GraphRAG: a graph-structured RAG method that extracts entities and relations from each statement and constructs a knowledge graph [19]. During retrieval, it calculates the semantic relevance between the query and entities/relations.

Besides the methods mentioned above, we establish two special baselines for more comprehensive studies as follows:

• Ignoramus: an ablation baseline for comparison, which conducts inference without using any user statements.   
• Oracle: an upper-bound baseline that utilizes additional golden references for inference, eliminating retrieval errors.

In the following sections, we evaluate all these explicit memory baselines across all reasoning structures. We analyze the experimental results and draw conclusions.

# 4.2 Overall Performances

The overall performances are presented in Figure 3. From the results, we find that reasoning structures significantly affect task performance in explicit memory. The performance of SR and MR is significantly higher than DR and NR, with an overall improvement of $1 0 \%$ to $2 0 \%$ . Besides, NR is weaker than other multi-hop structures,

![](images/e03592e53f013b628024bf3819270225595650e29997f1b7e90562ff91c0ad00.jpg)  
Figure 4: Performance of various retrieved statement counts. Darker colors indicate higher accuracy in MPR tasks.

![](images/ed1a83d91882886dc0bb74c1ac06bf079840f7d61f4d5cccf8b1dec7ebcb782f.jpg)  
Figure 5: Performance of various reasoning steps, with mean values (line) and standard deviation values (shading).

especially on long-hop questions. It indicates that the test-time scaling [40] is effective, and multi-hop reasoning structures are important for complex problems. Additionally, DR relies on the initial question for task decomposition, which may exhibit certain myopia that affects further reasoning.

From the perspective of memory baselines, we find DenseRAG performs best on short-hop questions, while SparseRAG performs best on long-hop questions. This may be because DenseRAG can better locate required information based on semantic granularity, while SparseRAG can retrieve broader information based on word granularity. Besides, TreeRAG performs comparably to the above two methods. To our surprise, GraphRAG shows poor performance in our settings, which may be affected by similar entities during graph construction, thereby introducing much noise.

In addition, as the question hop increases, the overall accuracy shows a declining trend as expected. For example, the accuracy of DenseRAG on SR and MR gradually decreases from over $6 0 \%$ o n 2-hop questions to around $2 0 \%$ on 10-hop questions. Besides, we find SR and MR can mitigate the decline on Oracle, but other RAG methods still exhibit rapid degradation. As the difficulty of questions increases, the requirement of multi-hop reasoning also boosts, so SR and MR greatly improve the performance of long-hop questions on Oracle. However, other baselines may lack a global perspective due to partial retrieval, thereby reducing planning abilities.

# 4.3 Impact of Retrieved Statement Counts

For explicit memory, the number of retrieved information is a critical factor. Too few retrievals may result in information loss, while too many can lead to noise and extra cost. Therefore, we further explore the impact of retrieved statement counts. Specifically, we construct heatmaps to analyze four RAG baselines, and use different colors to represent reasoning structures, as presented in Figure 4.

![](images/77443b2a1aafd7f5171c39341ec7fc1bb4ced42db9eb0a92e00ff8826754b934.jpg)  
Figure 6: Performance of various backbone sizes. In each bar, the cells indicate increasing hops from bottom to top.

According to the results, we find that in short-hop questions, the accuracy typically improves as $k$ increases. We speculate shorthop questions require recalling more comprehensive information to facilitate short-range reasoning, so a larger $k$ can provide more sufficient information (i.e., high recall). However, in long-hop questions, there exists a peak of accuracy at an intermediate $k$ value. This may be because long-range reasoning is more susceptible to noise, and too large $k$ values may not be applicable (i.e., low precision).

# 4.4 Impact of Reasoning Steps

For reasoning tasks, the number of reasoning steps can be a critical bottleneck. Increasing the number of reasoning steps typically leads to higher accuracy, but also results in linear growth of reasoning costs. To further study the impact on model performance, we set different numbers of reasoning steps and conduct experiments across various memory baselines and reasoning structures. The results are presented in Figure 5.

From the results, we find that model performances improve as the number of reasoning steps increases, but the gains beyond 3 steps are not significant. Furthermore, we observe that DR exhibits greater sensitivity compared to the other two reasoning structures. This may be because extending the reasoning chain can potentially mitigate the limitation of initial question decomposition. Additionally, long-hop problems are more affected compared to short-hop and medium-hop questions, which is intuitive as long-hop problems require more reasoning steps. Finally, Oracle is more affected than other RAG baselines, possibly because the model with golden references benefits more from increased reasoning steps.

# 4.5 Impact of Backbone Sizes

Explicit memory primarily relies on the ICL capability of LLMs, and different backbone sizes exhibit varying abilities for context understanding and reasoning. Therefore, we further study the performance of various baselines and reasoning structures under different sizes of backbones. We present the results in Figure 6.

We observe that the 3B model shows significant degradation compared to the 7B model across all scenarios. Among reasoning structures, the performance decline of NR and DR is significantly higher than the others, indicating that sequential reasoning demonstrates higher robustness. Additionally, different RAG baselines exhibit consistent performance degradation, as retrieval and generation are relatively independent processes under our setting.

![](images/c8290d9a8b4430b95c729fe24f288b343b39ce0820096b7c3cab1e52277ebfa1.jpg)  
(a) Explicit Memory

![](images/5c534efbdfc80b3c65d57fdadc0522aae69606abb6436256f02c67191600d0fe.jpg)  
(b) Implicit Memory

![](images/ecd2b632ad71724fe6dba7451e9886726dd20fbcad18c1ecab5e9c5e083a9fd3.jpg)  
(c) Hybrid Memory   
Figure 7: Efficiency of different memory mechanisms.

# 4.6 Model Efficiency

Efficiency is critical for LLM-based agents, affecting user satisfaction and computational overhead. Efficient memory approaches contribute to improving the response speed of LLM-based agents, especially in online environments. Therefore, we further evaluate the efficiency of various memory baselines and reasoning structures. Specifically, we calculate the average task completion time under different question hops. The results are shown in Figure 7(a).

The results indicate that multi-hop reasoning structures demonstrate a substantial increase in time consumption compared to single-step reasoning, because additional reasoning steps bring extra inference time. Moreover, the RAG baselines exhibit higher time overhead than both the Oracle and Ignoramus, and TreeRAG exhibits significantly higher overhead than the other methods. This is because TreeRAG constructs extensive summarization nodes for retrieval. Finally, the more hops a question has, the longer the time consumption, as a higher number of hops implies greater question complexity, leading to more reasoning tokens to reach the answer.

# 5 Implicit Memory on MPR Task

# 5.1 Overview

We further explore the performance of implicit memory in LLMbased agents on MPR tasks. During the training phase, we transform user statements into input-output tuples and conduct fine-tuning based on LoRA. During the testing phase, we do not employ retrieval strategies, but perform inference directly. According to the user statement transformation, we have the following baselines:

• MaskSFT: randomly mask entities or relations of user statements as input, and use the masked information as output to conduct instruction fine-tuning, inspired by Devlin et al. [5].   
• AskSFT: rewrite user statements into QA pairs, and use them as input and output, respectively for instruction fine-tuning.

To better compare the effects of implicit memory, we also add two special baselines, MaskSFT $^ +$ Oracle (MSO) and AskSFT $^ +$ Oracle (ASO), which incorporate golden references during inference.

# 5.2 Overall Performances

The results of overall performances are presented in Figure 8. We find that across all reasoning structures, using implicit memory alone achieves poor performance, indicating that SFT cannot effectively handle large-scale detailed information. Besides, there exists consistent degradation on SR and MR by adding implicit memory on Oracle baselines, possibly because SFT may decline the reasoning capability of LLMs. We also find that on DR, ASO shows certain improvement over Oracle, especially on long-hop problems, which may be due to the enhancement of task decomposition capability.

![](images/4ddf3fd1af54af6a776b6f3baae39857bb8fd58dbcb73f5e944d7ace1300df27.jpg)  
Figure 8: Overall performances of implicit memory.

# 5.3 Impact of Training Steps

The number of training epochs is significant for SFT, as too many epochs can cause overfitting, while too few epochs may lead to underfitting. Therefore, we further explore the model performance under different training epochs. Due to the page limitation, we put the results in Appendix A.1. The results indicate that implicit memory alone cannot achieve great performance even after training for more epochs. Additionally, the performance of MSO and ASO significantly declines as training steps increase, possibly because more training steps lead to greater reasoning capability degradation.

# 5.4 Impact of Reasoning Steps

Similar to explicit memory, we also explore the impact of reasoning steps on implicit memory, with results shown in Appendix A.2 due to the page limitation. We find that even with increased reasoning steps, implicit memory cannot achieve reasonable results under our setting. For MSO and ASO, their results exhibit similar patterns to explicit memory as analyzed in the previous section.

# 5.5 Impact of Base Model Sizes

We also study the impact of model size on implicit memory, with results presented in Appendix A.3 due to the page limitation. Across various reasoning structures and memory baselines, the 3B model shows significant degradation in accuracy. Among different reasoning structures, NR and DR exhibit the most pronounced performance decline, significantly higher than the other two reasoning structures, with DR declining by approximately $8 0 \%$ .

# 5.6 Model Efficiency

We conduct an efficiency analysis of implicit memory approaches, with results illustrated in Figure 7(b). Beyond the patterns consistent with explicit memory findings, we find that implicit memory does not require retrieval overhead and extracts tokens to contain retrieved content. It suggests that implicit memory approaches offer considerable potential for enhancing inference efficiency.

# 6 Hybrid Memory on MPR Task

# 6.1 Overview

Our experiments reveal that relying solely on implicit memory is insufficient for MPR tasks. However, implicit memory demonstrates potential for enhancing reasoning capabilities. Therefore, we further combine implicit memory with explicit memory to form hybrid memory for additional experiments. A straightforward approach involves using SFT models as backbones with retrieved statements for reasoning, resulting in 8 hybrid models.

# 6.2 BlockSFT Method

Although directly combining SFT and RAG is easy to implement, it has several limitations. First, SFT struggles to capture a large number of details and may encounter conflicts in the constructed

Table 3: Overall performances of different hybrid memory methods. The sign $\mathbf { + X }$ represents direct combination of various explicit memory methods, and the bold values indicate the best performances.   

<table><tr><td colspan="7">Naive Reasoning</td><td colspan="5">Sequential Reasoning</td></tr><tr><td>Hops</td><td>Methods</td><td>Oracle</td><td>DenseRAG</td><td>SparseRAG</td><td>TreeRAG</td><td>GraphRAG</td><td>Oracle</td><td>DenseRAG</td><td>SparseRAG</td><td>TreeRAG</td><td>GraphRAG</td></tr><tr><td rowspan="4">Short(2~6)</td><td>X</td><td>0.428</td><td>0.206</td><td>0.204</td><td>0.212</td><td>0.112</td><td>0.703</td><td>0.324</td><td>0.361</td><td>0.334</td><td>0.190</td></tr><tr><td>MS+X</td><td>0.437</td><td>0.204</td><td>0.207</td><td>0.202</td><td>0.106</td><td>0.658</td><td>0.289</td><td>0.319</td><td>0.307</td><td>0.174</td></tr><tr><td>AS+X</td><td>0.432</td><td>0.212</td><td>0.212</td><td>0.213</td><td>0.117</td><td>0.627</td><td>0.263</td><td>0.287</td><td>0.269</td><td>0.164</td></tr><tr><td>HybridMem</td><td>0.436</td><td>0.197</td><td>0.202</td><td>0.207</td><td>0.114</td><td>0.702</td><td>0.326</td><td>0.368</td><td>0.336</td><td>0.192</td></tr><tr><td rowspan="4">Long(7~10)</td><td>X</td><td>0.202</td><td>0.094</td><td>0.084</td><td>0.096</td><td>0.051</td><td>0.566</td><td>0.216</td><td>0.200</td><td>0.208</td><td>0.140</td></tr><tr><td>MaskSFT+X</td><td>0.216</td><td>0.086</td><td>0.081</td><td>0.084</td><td>0.050</td><td>0.479</td><td>0.190</td><td>0.190</td><td>0.174</td><td>0.121</td></tr><tr><td>AskSFT+X</td><td>0.219</td><td>0.097</td><td>0.089</td><td>0.096</td><td>0.058</td><td>0.446</td><td>0.168</td><td>0.170</td><td>0.134</td><td>0.112</td></tr><tr><td>HybridMem</td><td>0.208</td><td>0.094</td><td>0.090</td><td>0.093</td><td>0.058</td><td>0.563</td><td>0.223</td><td>0.232</td><td>0.209</td><td>0.139</td></tr></table>

<table><tr><td colspan="7">Multi-path Reasoning</td><td colspan="6">Decomposition Reasoning</td></tr><tr><td>Hops</td><td>Methods</td><td>Oracle</td><td>DenseRAG</td><td>SparseRAG</td><td>TreeRAG</td><td>GraphRAG</td><td>Oracle</td><td>DenseRAG</td><td>SparseRAG</td><td>TreeRAG</td><td>GraphRAG</td><td></td></tr><tr><td rowspan="4">Short(2~6)</td><td>X</td><td>0.661</td><td>0.309</td><td>0.332</td><td>0.306</td><td>0.171</td><td>0.518</td><td>0.237</td><td>0.238</td><td>0.214</td><td>0.125</td><td></td></tr><tr><td>MS+X</td><td>0.600</td><td>0.263</td><td>0.286</td><td>0.276</td><td>0.153</td><td>0.528</td><td>0.228</td><td>0.234</td><td>0.211</td><td>0.132</td><td></td></tr><tr><td>AS+X</td><td>0.529</td><td>0.202</td><td>0.195</td><td>0.208</td><td>0.121</td><td>0.463</td><td>0.222</td><td>0.219</td><td>0.201</td><td>0.124</td><td></td></tr><tr><td>HybridMem</td><td>0.672</td><td>0.304</td><td>0.326</td><td>0.310</td><td>0.166</td><td>0.533</td><td>0.234</td><td>0.239</td><td>0.210</td><td>0.122</td><td></td></tr><tr><td rowspan="4">Long(7~10)</td><td>X</td><td>0.510</td><td>0.217</td><td>0.188</td><td>0.193</td><td>0.123</td><td>0.316</td><td>0.136</td><td>0.111</td><td>0.093</td><td>0.068</td><td></td></tr><tr><td>MS+X</td><td>0.407</td><td>0.159</td><td>0.166</td><td>0.156</td><td>0.100</td><td>0.322</td><td>0.128</td><td>0.099</td><td>0.104</td><td>0.073</td><td></td></tr><tr><td>AS+X</td><td>0.328</td><td>0.151</td><td>0.123</td><td>0.116</td><td>0.086</td><td>0.277</td><td>0.119</td><td>0.103</td><td>0.099</td><td>0.068</td><td></td></tr><tr><td>HybridMem</td><td>0.518</td><td>0.209</td><td>0.194</td><td>0.206</td><td>0.123</td><td>0.321</td><td>0.144</td><td>0.113</td><td>0.089</td><td>0.069</td><td></td></tr></table>

![](images/bb6b097a1b371959f1baa4f48a26ab431c7f2cfdcc168b016c1dc269ba3baf5c.jpg)

![](images/02c67ad3e6043b0c59000a8556052094773dbaf86d44a36a6bf41d197b8238d1.jpg)

![](images/5a61bee640aaa8704025df1c76ac9280fcd9d3db32d4dc53b273860907a5b1dc.jpg)

![](images/fb2373ea94129a952a08c9be3aecbf0adf01e8439f2917aa5527039697acbb5e.jpg)  
Figure 9: Performance of various cluster counts. In each bar, the cells indicates increasing hops from bottom to top.

instructions. Besides, training on the entire corpus can degrade reasoning performance. Finally, the inference process is not queryrelated, resulting in limited enhancement of explicit memory.

To address these limitations, we propose a novel hybrid method called HybridMem. In MPR tasks, user statements typically exhibit contextual relationships, forming multiple local clusters. Therefore, we divide the entire collection of user statements into multiple clusters and conduct SFT independently on each cluster to obtain multiple LoRA adapters while constructing corresponding indices. During inference, we identify the most relevant LoRA adapter based on the current retrieved statements and conduct inference using the selected adapter. Specifically, we employ K-means [15] for clustering, adopt a voting aggregation strategy to determine the adapter most relevant to the retrieval results, and use the same dense retrieval method as explicit memory to construct indices.

# 6.3 Overall Performance

The overall performance results are presented in Table 3. Our method achieves the best overall performance on multi-hop reasoning structures, with particularly notable improvements on long-hop questions. This demonstrates the effectiveness of our proposed approach. Additionally, we observe that implicit memory can degrade model performance, especially for $_ { \mathrm { A S + X } }$ under multi-hop reasoning scenarios. This degradation likely results from a trade-off between the implicit memory’s command of overall knowledge and the reasoning degradation introduced during training. Finally, we find that HybridMem consistently improves performance across multiple RAG methods and also enhances the Oracle baseline.

![](images/0d97d1d5b94345beaf66f2c405317c2fd1ed436f7ea1cb7f176aa02de0480462.jpg)  
Figure 10: Performance of training epochs. Darker colors indicate higher accuracy in MPR tasks.

# 6.4 Impact of Cluster Counts

The number of clusters used to divide the entire statement collection is a crucial factor. Fewer clusters require each LoRA adapter to carry more information, while more clusters incur greater training overhead. Therefore, we evaluate the performance across different cluster numbers, with results shown in Figure 9. We find that the performance difference between 30 blocks and 50 blocks is minimal, though the 50-block configuration shows slight improvements. This indicates that our method exhibits good robustness to block counts.

# 6.5 Impact of Training Epochs

Additionally, we explore the impact of different training epochs on our method, with results shown in Figure 10. The results show that model performance initially improves before decreasing as the number of training epochs increases. The model achieves optimal performance at epoch 7, particularly for long-hop questions. Furthermore, for different reasoning structures, we notice that the number of training epochs significantly affects SR and MR performance, but shows minimal impact on NR and DR.

# 6.6 Model Efficiency

In addition, we evaluate the efficiency of various hybrid models, with results presented in Figure 7(c). The results show that Hybrid-Mem generally requires more time than other approaches, particularly on SR and MR structures. This is attributed to the additional

cost of statement clustering and adapter loading. Furthermore, HybridMem significantly increases the differences in time requirements across various RAG methods.

# 7 Conclusion

In this paper, we formally define the MPR task and construct a dataset to evaluate different memory approaches. We conduct experiments with various types of memory on MPR tasks, drawing conclusions and findings. We propose HybridMem to better solve long-hop tasks. In future research, we will explore the adaptive integration of implicit memory and explicit memory, as well as multimodal personalized memory with reasoning strategies.

# References

[1] Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. 2024. A survey on evaluation of large language models. ACM transactions on intelligent systems and technology 15, 3 (2024), 1–45.   
[2] Jin Chen, Zheng Liu, Xu Huang, Chenwang Wu, Qi Liu, Gangwei Jiang, Yuanhao Pu, Yuxuan Lei, Xiaolong Chen, Xingmei Wang, et al. 2024. When large language models meet personalization: Perspectives of challenges and opportunities. World Wide Web 27, 4 (2024), 42.   
[3] Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Wang. 2020. Hybridqa: A dataset of multi-hop question answering over tabular and textual data. arXiv preprint arXiv:2004.07347 (2020).   
[4] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413 (2025).   
[5] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). 4171–4186.   
[6] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al. 2022. A survey on in-context learning. arXiv preprint arXiv:2301.00234 (2022).   
[7] Yiming Du, Hongru Wang, Zhengyi Zhao, Bin Liang, Baojun Wang, Wanjun Zhong, Zezhong Wang, and Kam-Fai Wong. 2024. PerLTQA: A Personal Long-Term Memory Dataset for Memory Classification, Retrieval, and Synthesis in Question Answering. arXiv preprint arXiv:2402.16288 (2024).   
[8] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A survey on rag meeting llms: Towards retrieval-augmented large language models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 6491–6501.   
[9] Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. 2024. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680 (2024).   
[10] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060 (2020).   
[11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. ICLR 1, 2 (2022), 3.   
[12] Xu Huang, Weiwen Liu, Xiaolong Chen, Xingmei Wang, Hao Wang, Defu Lian, Yasheng Wang, Ruiming Tang, and Enhong Chen. 2024. Understanding the planning of LLM agents: A survey. arXiv preprint arXiv:2402.02716 (2024).   
[13] Ishita Kumar, Snigdha Viswanathan, Sushrita Yerra, Alireza Salemi, Ryan A Rossi, Franck Dernoncourt, Hanieh Deilamsalehy, Xiang Chen, Ruiyi Zhang, Shubham Agarwal, et al. 2024. Longlamp: A benchmark for personalized long-form text generation. arXiv preprint arXiv:2407.11016 (2024).   
[14] Yuanchun Li, Hao Wen, Weijun Wang, Xiangyu Li, Yizhen Yuan, Guohong Liu, Jiacheng Liu, Wenxing Xu, Xiang Wang, Yi Sun, et al. 2024. Personal llm agents: Insights and survey about the capability, efficiency and security. arXiv preprint arXiv:2401.05459 (2024).   
[15] Stuart Lloyd. 1982. Least squares quantization in PCM. IEEE transactions on information theory 28, 2 (1982), 129–137.   
[16] Xing Han Lù. 2024. BM25S: Orders of magnitude faster lexical search via eager sparse scoring. arXiv:2407.03618 [cs.IR] https://arxiv.org/abs/2407.03618   
[17] Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D Manning. 2021. Fast model editing at scale. arXiv preprint arXiv:2110.11309 (2021).   
[18] Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. 2023. A comprehensive overview of large language models. ACM Transactions on Intelligent Systems and Technology (2023).   
[19] Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. 2024. Graph retrieval-augmented generation: A survey. arXiv preprint arXiv:2408.08921 (2024).   
[20] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: $^ { 1 0 0 , 0 0 0 + }$ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250 (2016).   
[21] Alireza Rezazadeh, Zichao Li, Wei Wei, and Yujia Bao. 2024. From Isolated Conversations to Hierarchical Schemas: Dynamic Tree Memory Representation for LLMs. arXiv preprint arXiv:2410.14052 (2024).   
[22] Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. 2023. Lamp: When large language models meet personalization. arXiv preprint arXiv:2304.11406 (2023).   
[23] Julian Schnitzler, Xanh Ho, Jiahao Huang, Florian Boudin, Saku Sugawara, and Akiko Aizawa. 2024. Morehopqa: More than multi-hop reasoning. arXiv preprint

arXiv:2406.13397 (2024).   
[24] Teng Shi, Jun Xu, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Yang Song, and Han Li. 2025. Retrieval Augmented Generation with Collaborative Filtering for Personalized Text Generation. arXiv preprint arXiv:2504.05731 (2025).   
[25] Karen Sparck Jones. 1972. A statistical interpretation of term specificity and its application in retrieval. Journal of documentation 28, 1 (1972), 11–21.   
[26] Haoran Tan, Zeyu Zhang, Chen Ma, Xu Chen, Quanyu Dai, and Zhenhua Dong. 2025. MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents. arXiv:2506.21605 [cs.CL] https://arxiv.org/abs/2506.21605   
[27] Zhaoxuan Tan, Qingkai Zeng, Yijun Tian, Zheyuan Liu, Bing Yin, and Meng Jiang. 2024. Democratizing large language models via personalized parameter-efficient fine-tuning. arXiv preprint arXiv:2402.04401 (2024).   
[28] Yixuan Tang and Yi Yang. 2024. Multihop-rag: Benchmarking retrievalaugmented generation for multi-hop queries. arXiv preprint arXiv:2401.15391 (2024).   
[29] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions of the Association for Computational Linguistics 10 (2022), 539–554.   
[30] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2024. A survey on large language model based autonomous agents. Frontiers of Computer Science 18, 6 (2024), 186345.   
[31] Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023. Plan-and-solve prompting: Improving zero-shot chainof-thought reasoning by large language models. arXiv preprint arXiv:2305.04091 (2023).   
[32] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv preprint arXiv:2212.03533 (2022).   
[33] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35 (2022), 24824–24837.   
[34] Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2024. Longmemeval: Benchmarking chat assistants on long-term interactive memory. arXiv preprint arXiv:2410.10813 (2024).   
[35] Jian Wu, Linyi Yang, Zhen Wang, Manabu Okumura, and Yue Zhang. 2024. Cofca: A Step-Wise Counterfactual Multi-hop QA benchmark. arXiv preprint arXiv:2402.11924 (2024).   
[36] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. 2025. The rise and potential of large language model based agents: A survey. Science China Information Sciences 68, 2 (2025), 121101.   
[37] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115 (2024).   
[38] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600 (2018).   
[39] Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems 36 (2023), 11809–11822.   
[40] Qiyuan Zhang, Fuyuan Lyu, Zexu Sun, Lei Wang, Weixu Zhang, Wenyue Hua, Haolun Wu, Zhihan Guo, Yufei Wang, Niklas Muennighoff, et al. 2025. A Survey on Test-Time Scaling in Large Language Models: What, How, Where, and How Well? arXiv preprint arXiv:2503.24235 (2025).   
[41] Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Fei Wu, and Guoyin Wang. 2024. Instruction Tuning for Large Language Models: A Survey. arXiv:2308.10792 [cs.CL] https: //arxiv.org/abs/2308.10792   
[42] Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. 2024. A survey on the memory mechanism of large language model based agents. arXiv preprint arXiv:2404.13501 (2024).   
[43] Zeyu Zhang, Quanyu Dai, Luyu Chen, Zeren Jiang, Rui Li, Jieming Zhu, Xu Chen, Yi Xie, Zhenhua Dong, and Ji-Rong Wen. 2024. Memsim: A bayesian simulator for evaluating memory of llm-based personal assistants. arXiv preprint arXiv:2409.20163 (2024).   
[44] Zeyu Zhang, Quanyu Dai, Xu Chen, Rui Li, Zhongyang Li, and Zhenhua Dong. 2025. MemEngine: A Unified and Modular Library for Developing Advanced Memory of LLM-based Agents. In Companion Proceedings of the ACM on Web Conference 2025. 821–824.

[45] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. 2023. A survey of large language models. arXiv preprint arXiv:2303.18223 1, 2 (2023).   
[46] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2024. Memorybank: Enhancing large language models with long-term memory. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 19724–19731.   
[47] Zexuan Zhong, Zhengxuan Wu, Christopher D Manning, Christopher Potts, and Danqi Chen. 2023. Mquake: Assessing knowledge editing in language models via multi-hop questions. arXiv preprint arXiv:2305.14795 (2023).   
[48] Andrew Zhu, Alyssa Hwang, Liam Dugan, and Chris Callison-Burch. 2024. Fanoutqa: Multi-hop, multi-document question answering for large language models. arXiv preprint arXiv:2402.14116 (2024).

# A More Results of Experiments

# A.1 Results of Implicit Memory with Different Training Epochs

The results are presented as follows:

![](images/42de7edb28b0636ed4a5065ac822141f3b5421aa8a11cce7045dfec3a919cdd2.jpg)  
Figure 11: Performance of various training epochs.

The results indicate that implicit memory alone cannot achieve great performance even after training for more epochs. Additionally, the performance of MSO and ASO significantly declines as training steps increase, possibly because more training steps lead to greater reasoning capability degradation.

# A.2 Results of Implicit Memory with Different Reasoning Steps

The results are presented as follows:

![](images/04123f7cab81241bfc72011a4db97a4575094468b09a9b926ab9aa96730dd414.jpg)  
Figure 12: Performance of various reasoning steps.

We find that even with increased reasoning steps, implicit memory cannot achieve reasonable results under our setting. For MSO and ASO, their results exhibit similar patterns to explicit memory as analyzed in the previous section.

# A.3 Results of Implicit Memory with Different Backbone Size

The results are presented as follows:

![](images/454c314c0593b9008494b7f290efca4b5bbd21aebe8373ef46e276d1d25cb19e.jpg)  
Figure 13: Performance of various backbone size.

Across various reasoning structures and memory baselines, the 3B model shows significant degradation in accuracy. Among different reasoning structures, NR and DR exhibit the most pronounced performance decline, significantly higher than the other two reasoning structures, with DR declining by approximately $8 0 \%$ .

# B More Details of Reasoning Structures

# B.1 Prompts of Naive Reasoning

The prompts of naive reasoning are as follows:

# The prompt of NR with explicit memory.

Please help me answer the following question based on the given information.

Information: [Retrieved References]

Question: [Question]

Requirements:

1. your answer should be as concise as possible, commonly in a few words.   
2. if the answer is a date, please output it in YYYY-MM-DD format.   
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions.

# The prompt of NR with implicit memory.

Please help me answer the following question.

Question: [Question]

Requirements:

1. your answer should be as concise as possible, commonly in a few words.   
2. if the answer is a date, please output it in YYYY-MM-DD format.   
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions.

# B.2 Prompts of Sequential Reasoning

The process of sequential reasoning has three parts, including starting, thinking, and answering. The prompts of these steps are presented as follows:

# (Starting) The prompt of SR with explicit memory.

To better answer the following question, let’s think step by step.

The current step is 1, and you should provide the final answer at step [Max Steps].

Please generate your thoughts for the current step, and you may refer to the given information.

Information: [Retrieved References]

Question: [Question]

Requirements:

1. Your thoughts should be concise but informative sentences.   
2. You should only output the thoughts in one line (no code block).

# (Thinking) The prompt of SR with explicit memory.

To better answer the following questions, let’s think step by step.

The current step is [Current Step], and you should provide the final answer at step [Max Steps].

Please generate your thoughts for the current step, and you may refer to the given information and your previous thought.

Information: [Retrieved References]

Previous Thoughts: [Previous Thought]

Question: [Question]

Requirements:

1. Your thoughts should be concise but informative sentences.   
2. You should only output the thoughts in one line (no code block).

# (Answering) The prompt of SR with explicit memory.

To better answer the following question, let’s think step by step.

Please help me generate the answer to the question based on the given information and your previous thoughts.

Information: [Retrieved References]

Previous Thoughts: [Previous Thought]

Question: [Question]

Requirements:

1. your answer should be as concise as possible, commonly in a few words.   
2. if the answer is a date, please output it in YYYY-MM-DD format.   
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).   
You should only output the answer in one line (no code block), without any other descriptions.

# (Starting) The prompt of SR with implicit memory.

To better answer the following question, let’s think step by step.

The current step is 1, and you should provide the final answer at step [Max Steps].

Please generate your thoughts for the current step.

Question: [Question]

Requirements:

1. Your thoughts should be concise but informative sentences.   
2. You should only output the thoughts in one line (no code block).

# (Thinking) The prompt of SR with implicit memory.

To better answer the following questions, let’s think step by step.

The current step is [Current Step], and you should provide the final answer at step [Max Steps].

Please generate your thoughts for the current step, and you may refer to your previous thought.

Previous Thoughts: [Previous Thought]

Question: [Question]

Requirements:

1. Your thoughts should be concise but informative sentences.   
2. You should only output the thoughts in one line (no code block).

# (Answering) The prompt of SR with implicit memory.

To better answer the following question, let’s think step by step.

Please help me generate the answer to the question based on your previous thoughts.

Previous Thoughts: [Previous Thought]

Question: [Question]

Requirements:

1. your answer should be as concise as possible, commonly in a few words.   
2. if the answer is a date, please output it in YYYY-MM-DD format.   
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions.

# B.3 Prompts of Multi-path Reasoning

The process of multi-path reasoning has three parts, including starting, thinking, and answering. The prompts of these steps are presented as follows:

# (Starting) The prompt of MR with explicit memory.

To better answer the following question, let’s think step by step.

Please generate your thoughts for the current step, and you may refer to the given information.

Information: [Retrieved References]

Question: [Question]

Requirements:

1. Your thoughts should be concise but informative sentences.   
2. You should only output the thoughts in one line (no code block).

# (Thinking) The prompt of MR with explicit memory.

To better answer the following questions, let’s think step by step.

Please generate your thoughts for the current step, and you may refer to the given information and your previous thoughts.

Information: [Retrieved References]

Previous Thoughts: [Previous Thought]

Question: [Question]

Requirements:

1. Your thoughts should be concise but informative sentences.   
2. You should only output the thoughts in one line (no code block).

# (Thinking) The prompt of MR with implicit memory.

To better answer the following questions, let’s think step by step.

Please generate your thoughts for the current step, and you may refer to your previous thoughts.

Previous Thoughts: [Previous Thought]

Question: [Question]

Requirements:

1. Your thoughts should be concise but informative sentences.   
2. You should only output the thoughts in one line (no code block).

# (Answering) The prompt of MR with explicit memory.

To better answer the following question, let’s think step by step.

Please help me generate the answer to the question based on the given information and your previous thoughts.

Information: [Retrieved References]

Previous Thoughts: [Previous Thought]

Question: [Question]

Requirements:

1. your answer should be as concise as possible, commonly in a few words.   
2. if the answer is a date, please output it in YYYY-MM-DD format.   
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).   
You should only output the answer in one line (no code block), without any other descriptions.

# (Answering) The prompt of MR with implicit memory.

To better answer the following question, let’s think step by step.

Please help me generate the answer to the question based on your previous thoughts.

Information: [Retrieved References]

Previous Thoughts: [Previous Thought]

Question: [Question]

Requirements:

1. your answer should be as concise as possible, commonly in a few words.   
2. if the answer is a date, please output it in YYYY-MM-DD format.   
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).

You should only output the answer in one line (no code block), without any other descriptions.

# B.4 Prompts of Decomposition Reasoning

The process of decomposition reasoning has three parts, including dividing, solving, and merging. The prompts of these steps are presented as follows:

# (Starting) The prompt of MR with implicit memory.

To better answer the following question, let’s think step by step.

Please generate your thoughts for the current step.

Question: [Question]

Requirements:

1. Your thoughts should be concise but informative sentences.   
2. You should only output the thoughts in one line (no code block).

# (Dividing) The prompt of NR with explicit memory.

To better answer the following question, please break it down into several sub-questions.

Question: [Question]

Requirements:

1. The sub-questions should be concise.   
2. Each sub-questions is on a separate line, without any other descriptions.   
3. You can decompose the problem into 1 to [Max Subquestion] sub-questions, and any content beyond 5 lines will be ignored.

# (Solving) The prompt of NR with explicit memory.

In order to better answer the following question, we have decomposed them into several sub-questions.

Please help me generate the answer to the current subquestion based on the given information.

Question: [Question]

Sub-questions: [Current State]

Current Sub-question: [Current Sub-question]

Information: [Retrieved References]

Requirements:

1. The answer should be concise but informative.   
2. You should only output the answer in one line (no code block), without any other descriptions.

# (Merging) The prompt of NR with explicit memory.

In order to better answer the following question, we have decomposed them into several sub-questions and answered them separately.

Please help me generate the final answer to the question based on the sub-questions and the given information.

Question: [Question]

Sub-questions and Answers: [Current State]

Information: [Retrieved References]

Requirements:

1. your answer should be as concise as possible, commonly in a few words.   
2. if the answer is a date, please output it in YYYY-MM-DD format.   
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).   
You should only output the answer in one line (no code block), without any other descriptions.

# (Dividing) The prompt of NR with implicit memory.

To better answer the following question, please break it down into several sub-questions.

Question: [Question]

Requirements:

1. The sub-questions should be concise.   
2. Each sub-questions is on a separate line, without any other descriptions.   
3. You can decompose the problem into 1 to [Max Subquestion] sub-questions, and any content beyond 5 lines will be ignored.

# (Solving) The prompt of NR with implicit memory.

In order to better answer the following question, we have decomposed them into several sub-questions.

Please help me generate the answer to the current subquestion.

Question: [Question]

Sub-questions: [Current State]

Current Sub-question: [Current Sub-question]

Requirements:

1. The answer should be concise but informative.   
2. You should only output the answer in one line (no code block), without any other descriptions.

# (Merging) The prompt of NR with implicit memory.

In order to better answer the following question, we have decomposed them into several sub-questions and answered them separately.

Please help me generate the final answer to the question based on the sub-questions.

Question: [Question]

Sub-questions and Answers: [Current State]

Requirements:

1. your answer should be as concise as possible, commonly in a few words.   
2. if the answer is a date, please output it in YYYY-MM-DD format.   
3. if the answer is a number, please do not include commas. If the numerical answer has units, please indicate them in parentheses, such as 5 (USD).   
You should only output the answer in one line (no code block), without any other descriptions.
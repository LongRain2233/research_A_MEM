# MemInsight: Autonomous Memory Augmentation for LLM Agents

Rana Salama, Jason Cai, Michelle Yuan, Anna Currey, Monica Sunkara, Yi Zhang, Yassine Benajiba

AWS AI

{ranasal, cjinglun, miyuan, ancurrey, sunkaral, yizhngn, benajiy }@amazon.com

# Abstract

Large language model (LLM) agents have evolved to intelligently process information, make decisions, and interact with users or tools. A key capability is the integration of long-term memory capabilities, enabling these agents to draw upon historical interactions and knowledge. However, the growing memory size and need for semantic structuring pose significant challenges. In this work, we propose an autonomous memory augmentation approach, MemInsight, to enhance semantic data representation and retrieval mechanisms. By leveraging autonomous augmentation to historical interactions, LLM agents are shown to deliver more accurate and contextualized responses. We empirically validate the efficacy of our proposed approach in three task scenarios; conversational recommendation, question answering and event summarization. On the LLM-REDIAL dataset, MemInsight boosts persuasiveness of recommendations by up to $14 \%$ . Moreover, it outperforms a RAG baseline by $34 \%$ in recall for LoCoMo retrieval. Our empirical results show the potential of MemInsight to enhance the contextual performance of LLM agents across multiple tasks

# 1 Introduction

LLM agents have emerged as an advanced framework to extend the capabilities of LLMs to improve reasoning (Yao et al., 2023; Wang et al., 2024c), adaptability (Wang et al., 2024d), and selfevolution (Zhao et al., 2024a; Wang et al., 2024e; Tang et al., 2025). A key component of these agents is their memory module, which retains past interactions to allow more coherent, consistent, and personalized responses across various tasks. The memory of the LLM agent is designed to emulate human cognitive processes by simulating how knowledge is accumulated and historical experiences are leveraged to facilitate complex reasoning and the retrieval of relevant information to inform

actions (Zhang et al., 2024). However, the advantages of an LLM agent’s memory also introduce notable challenges (Wang et al., 2024b). As interactions accumulate over time, retrieving relevant information becomes increasingly difficult, especially in long-term or complex tasks. Raw historical data grows rapidly and, without effective memory management, can become noisy and imprecise, hindering retrieval and degrading agent performance. Moreover, unstructured memory limits the agent’s ability to integrate knowledge across tasks and contexts. Therefore, structured knowledge representation is essential for efficient retrieval, enhancing contextual understanding, and supporting scalable long-term memory in LLM agents. Improved memory management enables better retrieval and contextual awareness, making this a critical and evolving area of research.

Hence, in this paper we introduce an autonomous memory augmentation approach, MemInsight, which empowers LLM agents to identify critical information within the data and proactively propose effective attributes for memory enhancements. This is analogous to the human processes of attentional control and cognitive updating, which involve selectively prioritizing relevant information, filtering out distractions, and continuously refreshing the mental workspace with new and pertinent data (Hu et al., 2024; Hou et al., 2024).

MemInsight autonomously generates augmentations that encode both relevant semantic and contextual information for memory. These augmentations facilitate the identification of memory components pertinent to various tasks. Accordingly, MemInsight can improve memory retrieval by leveraging relevant attributes of memory, thereby supporting autonomous LLM agent adaptability and self-evolution.

Our contributions can be summarized as follows:

• We propose a structured autonomous approach that adapts LLM agents’ memory representations while preserving context across extended conversations for various tasks.   
• We design and apply memory retrieval methods that leverage the generated memory augmentations to filter out irrelevant memory while retaining key historical insights.   
• Our promising empirical findings demonstrate the effectiveness of MemInsight on several tasks: conversational recommendation, question answering, and event summarization.

# 2 Related Work

Well-organized and semantically rich memory structures enable efficient storage and retrieval of information, allowing LLM agents to maintain contextual coherence and provide relevant responses. Developing an effective memory module in LLM agents typically involves two critical components: structural memory generation and memory retrieval methods (Zhang et al., 2024; Wang et al., 2024a).

LLM Agents Memory Recent research in LLM agents memory focuses on storing and retrieving prior interactions to improve adaptability and generalization (Packer et al., 2024; Zhao et al., 2024a; Zhang et al., 2024; Zhu et al., 2023). Common approaches structure memory as summaries, temporal events, or reasoning chains to reduce redundancy and highlight key information (Maharana et al., 2024; Anokhin et al., 2024; Liu et al., 2023a). Some methods enrich raw dialogues with semantic annotations, such as event sequences (Zhong et al., 2023; Maharana et al., 2024) or reusable workflows (Wang et al., 2024f). Recent models like A-Mem(Xu et al., 2025) that uses manually defined task-specific notes to structure an agent’s memory, while Mem0(Chhikara et al., 2025) offers a scalable, real-time memory pipeline for production use. However, most existing methods rely on unstructured memory or manually defined schemas. In contrast, MemInsight autonomously discovers semantically meaningful attributes, enabling structured memory representation without human-crafted definitions.

LLM Agents Memory Retrieval Recent work has explored memory retrieval techniques to improve efficiency when handling large-scale historical context in LLM agents (Hu et al., 2023a; Zhao

et al., 2024b; Tack et al., 2024; Ge et al., 2025). Common approaches involves generative retrieval models, which encode memory entries as dense vectors and retrieve the top- $k$ most relevant documents using similarity search (Zhong et al., 2023; Penha et al., 2024). Similarity metrics such as cosine similarity (Packer et al., 2024) are widely used, often in combination with dual-tower dense retrievers, where memory entries are embedded independently and indexed via tools like FAISS (Johnson et al., 2017) for efficient retrieval (Zhong et al., 2023). Additionally, techniques such as Locality-Sensitive Hashing (LSH) are utilized to retrieve tuples containing related entries in memory (Hu et al., 2023b).

# 3 Autonomous Memory Augmentation

Our proposed model, MemInsight, encapsulates the agent’s memory $M$ , offering a unified framework for augmenting and retrieving user–agent interactions represented as memory instances m. As new interactions occur, they are autonomously augmented and incorporated into memory, forming an enriched set $M = \left\{ m _ { 1 < a u g m e n t e d > } , \ldots , m _ { n < a u g m e n t e d > } \right\}$ . As shown in Figure 1, MemInsight comprises three core modules: attribute mining, annotation, and memory retrieval.

# 3.1 Attribute Mining and Annotation

Attribute mining extracts structured and semantically meaningful attributes from input dialogues for memory augmentation. The process follows a principled framework guided by three key dimensions:

(1) Perspective, from which attributes are derived (e.g., entity- or conversation-centric annotations)   
(2) Granularity, indicating the level of annotation detail (e.g., turn-level or session-level)   
(3) Annotation, which ensures that extracted attributes are appropriately aligned with the corresponding memory instance. A backbone LLM is leveraged to autonomously identify and generate relevant attributes.

# 3.1.1 Attribute Perspective

An attribute perspective entails two main orientations: entity-centric and conversation-centric. The entity-centric focuses on annotating specific items referenced in memory, such as books or movies, using attributes that capture their key properties (e.g., director, author, release year). In contrast, the

![](images/5a6f3015db723bce7975ba107eeffdc3d47fd7971c03f4760aaec4d28300ec2e.jpg)  
Figure 2: An example for Turn level and Session level annotations for a sample dialogue conversation from the LoCoMo Dataset.

Figure 1: MemInsight framework comprising three core modules: Attribute Mining (including perspective and granularity), Annotation (with attribute prioritization), and Memory Retrieval (including refined and comprehensive retrieval). These components are triggered by various downstream tasks such as Question Answering, Event Summarization, and Conversational Recommendation.

Melanie: "Hey Caroline, since we last chatted, I've had a lot of things happening to me. I ran a charity race for mental health last Saturday it was really rewarding. Really made me think about taking care of our minds.“

Caroline: "That charity race sounds great, Mel! Making a difference & raising awareness for mental health is super rewarding - I'm really proud of you for taking part!“

Melanie: "Thanks, Caroline! The event was really thought-provoking. I'm starting to realize that self-care is really important. It's a journey for me, but when I look after myself, I'm able to better look after my family. “

Caroline: "I totally agree, Melanie. Taking care of ourselves is so important - even if it's not always easy. Great that you're prioritizing self-care.

# Turn Level Augmentation:

Tum 1: [eventl:<charity race for mental health> [time]: <"last saturday“> [emotion]:<"rewarding“> [topic]: <mental health>

# Session Level Augmentation:

Melanie: [eventl<ran charity race for mental health> [emotion]<rewarding>,[intent]<thinking about selfcare> Caroline:[event]<raising mental health awareness>, [emotion]<proud>

conversation-centric perspective captures attributes that reflect the overall user interaction with respect to users’ intent, preferences, sentiment, emotions, motivations, and choices, thereby improving response generation and memory retrieval. An illustrative example is provided in Figure 4.

# 3.1.2 Attribute Granularity

Conversation-centric augmentations introduce the notion of attribute granularity, which defines the level of details captured in the augmentation process. The augmentation attributes can be analyzed at varying levels of abstraction, either at the level of individual turns within a user conversation (turnlevel), or across the entire dialogue session (sessionlevel), each offering distinct insights into the con-

versational context. Turn-level focuses on the specific content of individual turns to generate more nuanced and contextual attributes, while sessionlevel augmentation captures broader patterns and user intent across the interaction. Figure 2 illustrates this distinction, showing how both levels offer complementary perspectives on a sample dialogue.

# 3.1.3 Annotation and Attribute Prioritization

Subsequently, the generated attributes and their corresponding values are used to annotate the agent’s memory. Annotation is done by aggregating attributes and values in the relevant memory.

Given an interaction $i$ , the module applies an LLMbased extraction function $\mathcal { F } _ { \mathrm { L L M } }$ to produce a set of attribute–value pairs:

$$
A = \mathcal {F} _ {\mathrm {L L M}} (D) = \{(a _ {j}, v _ {j}) \} _ {j = 1} ^ {k}
$$

where: $a _ { j } ~ \in ~ { \mathcal { A } }$ represents the attribute (e.g., emotion, entity, intent) and $\upsilon _ { j } ~ \in ~ \nu$ the value of this attribute These attributes are then used to annotate the corresponding memory instance $m _ { i }$ , resulting in an augmented memory $M _ { a }$ : $M _ { a } \ = \ \{ ( A _ { 1 } , \tilde { m } _ { 1 } ) , ( A _ { 2 } , \tilde { m } _ { 2 } ) , . . . , ( A _ { i } , \tilde { m } _ { i } ) \}$ , Attributes are typically aggregated using the Attribute Prioritization method, which can be classified into Basic and Priority. In Basic Augmentation, attributes are aggregated without a predefined order, resulting in an arbitrary sequence. In contrast, Priority Augmentation sorts attribute-value pairs

according to their relevance to the memory being augmented. This prioritization follows a structured order in which attribute $( A _ { 1 } , \tilde { m _ { 1 } } )$ holds the highest significance, ensuring that more relevant attributes are processed first.

# 3.2 Memory Retrieval

MemInsight augmentations are employed to both enrich memory representations and support the retrieval of contextually relevant memory. These augmentations are utilized in one of two ways.

(1) Comprehensive retrieval, retrieves all related memory instances along with their associated augmentations to support context-aware inference.   
(2) Refined retrieval, where the current context is augmented to extract task-specific attributes, which then guide the retrieval process through one of the following methods:

a- Attribute-based Retrieval: which uses the current attributes as filters to select memory instances with matching or related augmentations only. Given a query session $Q$ with attributes $A _ { Q }$ , retrieve relevant memories:

$$
\mathcal {R} _ {\mathrm {a t t r}} \left(A _ {Q}, \mathbb {M}\right) = \operatorname {T o p} - k \left\{\left(A _ {k}, M _ {k}\right) \mid \operatorname {m a t c h} \left(A _ {Q}, A _ {k}\right) \right\}
$$

b- Embedding-based Retrieval where memory augmentations are embedded as dense vectors. A query embedding is derived from the current context’s augmentations and used to retrieve the top- $k$ most similar memory entries via similarity search. Let $\phi : A _ { k } \to \mathbb { V } ^ { d }$ be the embedding function over attributes. Then:

$$
s i m (A _ {Q}, A _ {k}) = \frac {\phi (A _ {Q}) \cdot \phi (A _ {k})}{\| \phi (A _ {Q}) \| \cdot \| \phi (A _ {k}) \|}
$$

$$
\mathcal {R} _ {\mathrm {e m b e d}} \left(A _ {Q}, \mathbb {M}\right) = \operatorname {T o p} - k \left\{\left(A _ {k}, M _ {k}\right) \mid \operatorname {s i m} \left(A _ {Q}, A _ {k}\right) \right\}
$$

Finally, the retrieved memories are then integrated into the current context to inform the ongoing interaction. Further implementation details of embedding-based retrieval are provided in Appendix C.

# 4 Evaluation

# 4.1 Datasets

We evaluate MemInsight on two benchmarks: LLM-REDIAL (Liang et al., 2024) and Lo-CoMo (Maharana et al., 2024). LLM-REDIAL is a dataset for conversational movie recommendation, comprising 10K dialogues and 11K movie mentions. LoCoMo is a dataset for evaluating Question Answering and Event Summarization, with 30

multi-session dialogues between two speakers. It features five question types: Single-hop, Multi-hop, Temporal reasoning, Open-domain, and Adversarial, each annotated with the relevant dialogue turn required for answering. LoCoMo also provides event labels for each speaker in a session, which serve as ground truth for evaluating event summarization.

# 4.2 Experimental Setup

To evaluate our model, we begin by augmenting the datasets using zero-shot prompting to extract relevant attributes and their corresponding values. For attribute generation across tasks, we employ Claude Sonnet1, LLaMA $3 ^ { 2 }$ , and Mistral3. For the Event Summarization task, we additionally utilize Claude 3 Haiku4. In embedding-based retrieval, we use the Titan Text Embedding model5 to generate embeddings of augmented memory, which are indexed and searched using FAISS (Johnson et al., 2017). To ensure consistency across all experiments, we use the same base model for the primary tasks: recommendation, answer generation, and summarization, while varying the models used for memory augmentation. Claude Sonnet serves as the backbone LLM in all baseline evaluations.

# 4.3 Evaluation Metrics

We evaluate MemInsight using a combination of standard and LLM-based metrics. For Question Answering, we report F1-score for answer prediction and recall for accuracy; for Conversational Recommendation, we use Recall $@ \mathbf { K }$ , NDCG@K, along with LLM-based metrics for genre matching. We further incorporate subjective metrics, including Persuasiveness (Liang et al., 2024), which measures how persuasive a recommendation aligns with the ground truth. Additionally, we introduce a Relatedness metric where we prompt an LLM to measure how comparable are recommendation attributes to the ground truth, categorizing them as not comparable, comparable, or highly comparable. For Event Summarization, we adopt G-Eval (Liu et al., 2023b), an LLM-based metric that evaluates the relevance, consistency, and coherence of generated summaries against reference labels. Together, these metrics provide a comprehensive framework

1claude-3-sonnet-20240229-v1   
2llama3-70b-instruct-v1   
3mistral-7b-instruct-v0   
4claude-3-haiku-20240307-v1   
5titan-embed-text-v2:0

for evaluating both retrieval effectiveness and response quality.

# 5 Experiments

# 5.1 Questioning Answering

Question Answering experiments are conducted to evaluate the effectiveness of MemInsight in answer generation. We evaluate the overall accuracy to measure the system’s ability to retrieve and integrate relevant information using memory augmentations. The base model, which incorporates all historical dialogues without any augmentation, serves as a baseline. Additionally, we report results on the LoCoMo benchmark using the same backbone model (Mistral v1) to ensure a fair evaluation. We also compare with stronger GPT-based baselines, including MemoryBank(Zhong et al., 2023) and ReadAGent (Lee et al., 2024), which utilizes external memory modules to support long-term reasoning. We also consider Dense Passage Retrieval (DPR) (Karpukhin et al., 2020) as a representative baseline of RAG due to its scalability and retrieval efficiency.

Memory Augmentation In this task, memory is constructed from historical conversational dialogues, which requires the generation of conversation-centric attributes for augmentation. Given that the ground-truth labels consist of dialogue turns relevant to the question, the dialogues are annotated at the turn level. An LLM backbone is prompted to generate augmentation attributes for both conversation-centric and turn-level annotations.

Memory Retrieval To answer a given question, MemInsight first augments it to extract relevant attributes, which guide memory retrieval. In attributebased retrieval, dialogue turns with matching augmentation attributes are retrieved. In embeddingbased retrieval, the question and its attributes are embedded to perform a vector similarity search over indexed memory. The top- $k$ most similar dialogue turns are then integrated into the current context to generate an answer.

Experimental Results As shown in Table 1, MemInsight achieves significantly higher overall accuracy on the question answering task compared to all baselines, using both attribute-based and embedding-based memory retrieval. In the attribute-based setting, MemInsight with Claude-3- Sonnet demonstrates notable gains in single-hop,

temporal, and adversarial questions, which require more complex contextual reasoning. These results highlight the effectiveness of memory augmentation in enriching context and enhancing answer quality. MemInsight further outperforms all other benchmark models across most question types, with the exception of multi-hop and temporal questions in LoCoMo, where evaluation is based on a partial-match F1 metric (Maharana et al., 2024).

For embedding-based retrieval, we evaluate MemInsight using both basic and priority augmentation, alongside the DPR baseline. MemInsight consistently outperforms all baselines, except in temporal and adversarial questions, where DPR achieves slightly higher accuracy. Nevertheless, MemInsight maintains the highest overall accuracy. Priority augmentation also consistently outperforms basic augmentation across nearly all question types, validating its effectiveness in improving contextual relevance. Notably, MemInsight demonstrates substantial gains on multi-hop questions, which require reasoning over multiple pieces of supporting evidence, highlighting its ability to integrate dispersed information from historical dialogue. As shown in Table 2, recall metrics further support this trend, with priority augmentation yielding a $3 5 \%$ overall improvement and consistent gains across all categories.

# 5.2 Conversational Recommendation

We simulate conversational recommendation by preparing dialogues for evaluation under the same conditions proposed by Liang et al. (2024). This process involves masking the dialogue and randomly selecting $n = 2 0 0$ conversations for evaluation to ensure a fair comparison. Each conversational dialogue used is processed by masking the ground truth labels, followed by a turn cut-off, where all dialogue turns following the first masked turn are removed and retained as evaluation labels. Subsequently, the dialogues are augmented using a conversation-centric approach to identify relevant user interest attributes for retrieval. Finally, we prompt the LLM model to generate a movie recommendation that best aligns with the masked token, guided by the augmented movies retrieved based on the user’s historical interactions.

The baseline for this evaluation is the results presented in the LLM-REDIAL paper (Liang et al., 2024) which employs zero-shot prompting for rec-

<table><tr><td>Model</td><td>Single-hop</td><td>Multi-hop</td><td>Temporal</td><td>Open-domain</td><td>Adversarial</td><td>Overall</td></tr><tr><td>Baseline (Claude-3-Sonnet)</td><td>15.0</td><td>10.0</td><td>3.3</td><td>26.0</td><td>45.3</td><td>26.1</td></tr><tr><td>LoCoMo (Mistral v1)</td><td>10.2</td><td>12.8</td><td>16.1</td><td>19.5</td><td>17.0</td><td>13.9</td></tr><tr><td>ReadAgent (GPT-4o)</td><td>9.1</td><td>12.6</td><td>5.3</td><td>9.6</td><td>9.81</td><td>8.5</td></tr><tr><td>MemoryBank (GPT-4o)</td><td>5.0</td><td>9.6</td><td>5.5</td><td>6.6</td><td>7.3</td><td>6.2</td></tr><tr><td colspan="7">Attribute-based Retrieval</td></tr><tr><td>MemInsight (Claude-3-Sonnet)</td><td>18.0</td><td>10.3</td><td>7.5</td><td>27.0</td><td>58.3</td><td>29.1</td></tr><tr><td colspan="7">Embedding-Based Retrieval</td></tr><tr><td>RAG Baseline (DPR)</td><td>11.9</td><td>9.0</td><td>6.3</td><td>12.0</td><td>89.9</td><td>28.7</td></tr><tr><td>MemInsight (Llama v3Priority)</td><td>14.3</td><td>13.4</td><td>6.0</td><td>15.8</td><td>82.7</td><td>29.7</td></tr><tr><td>MemInsight (Mistral v1Priority)</td><td>16.1</td><td>14.1</td><td>6.1</td><td>16.7</td><td>81.2</td><td>30.0</td></tr><tr><td>MemInsight (Claude-3-SonnetBasic)</td><td>14.7</td><td>13.8</td><td>5.8</td><td>15.6</td><td>82.1</td><td>29.6</td></tr><tr><td>MemInsight (Claude-3-SonnetPriority)</td><td>15.8</td><td>15.8</td><td>6.7</td><td>19.7</td><td>75.3</td><td>30.1</td></tr></table>

Table 1: Results for F1 Score $( \% )$ for answer generation accuracy for attribute-based and embedding-based memory retrieval methods. Baseline is Claude-3-Sonnet model to generate answers using all memory without augmentation, for Attribute-based retrieval. In addition to the Dense Passage Retrieval(DPR) for Embedding-based retrieval. Evaluation is done with $k = 5$ . Best results per question category over all methods are in bold.   
Table 2: Results for the RECALL $\boldsymbol { @ } \mathrm { k } { = } 5$ accuracy for Embedding-based retrieval for answer generation using LoCoMo dataset. Dense Passage Retrieval(DPR) RAG model is the baseline. Best results are in bold.   

<table><tr><td>Model</td><td>Single-hop</td><td>Multi-hop</td><td>Temporal</td><td>Open-domain</td><td>Adversarial</td><td>Overall</td></tr><tr><td>RAG Baseline (DPR)</td><td>15.7</td><td>31.4</td><td>15.4</td><td>15.4</td><td>34.9</td><td>26.5</td></tr><tr><td>MemInsight (Llama v3Priority)</td><td>31.3</td><td>63.6</td><td>23.8</td><td>53.4</td><td>28.7</td><td>44.9</td></tr><tr><td>MemInsight (Mistral v1Priority)</td><td>31.4</td><td>63.9</td><td>26.9</td><td>58.1</td><td>36.7</td><td>48.9</td></tr><tr><td>MemInsight (Claude-3-SonnetBasic)</td><td>33.2</td><td>67.1</td><td>29.5</td><td>56.2</td><td>35.7</td><td>48.8</td></tr><tr><td>MemInsight (Claude-3-SonnetPriority)</td><td>39.7</td><td>75.1</td><td>32.6</td><td>70.9</td><td>49.7</td><td>60.5</td></tr></table>

Table 3: Statistics of attributes generated for the LLM-REDIAL Movie dataset, which include total number of movies, average number of attributes per item, number of failed attributes, and the counts for the most frequent five attributes.   

<table><tr><td colspan="2">Statistic</td><td>Count</td></tr><tr><td colspan="2">Total Movies</td><td>9687</td></tr><tr><td colspan="2">Avg. Attributes</td><td>7.39</td></tr><tr><td colspan="2">Failed Attributes</td><td>0.10%</td></tr><tr><td rowspan="5">Top-5 Attributes</td><td>Genre</td><td>9662</td></tr><tr><td>Release year</td><td>5998</td></tr><tr><td>Director</td><td>5917</td></tr><tr><td>Setting</td><td>4302</td></tr><tr><td>Characters</td><td>3603</td></tr></table>

ommendation using the ChatGPT model6. In addition to the baseline model that uses memory without augmentation.

Evaluation includes direct matches between recommended and ground truth movie titles using RE-CALL $@$ [1,5,10] and NDCG $@$ [1,5,10]. Furthermore, to address inconsistencies in movie titles generated by LLMs, we incorporate an LLM-based evaluation that assesses recommendations based on genre similarity. Specifically, a recommended movie is considered a valid match if it shares the same genre as the corresponding ground truth label.

Memory Augmentation We initially augment the dataset with relevant attributes, primarily employing entity-centric augmentations for memory annotation, as the memory consists of movies. In this context, we conduct a detailed evaluation of the generated attributes to provide an initial assessment

of the effectiveness and relevance of MemInsight augmentations. To evaluate the quality of the generated attributes, Table 3 presents statistical data on the generated attributes, including the five most frequently occurring attributes across the entire dataset. As shown in the table, the generated attributes are generally relevant, with "genre" being the most significant attribute based on its cumulative frequency across all movies (also shown in Figure 5). However, the relevance of attributes vary, emphasizing the need for prioritization in augmentation. Additionally, the table reveals that augmentation was unsuccessful for $0 . 1 \%$ of the movies, primarily due to the LLM’s inability to recognize certain movie titles or because the presence of some words in the movie titles conflicted with the LLM’s policy.

Memory Retrieval For this task we evaluate attribute-based retrieval using the Claude-3-Sonnet model with both filtered and comprehensive settings. Additionally, we examine embedding-based retrieval using all other models. For embeddingbased retrieval, we set $k = 1 0$ , meaning that 10 memory instances are retrieved (as opposed to 144 in the baseline).

Experimental Results Table 4 shows the results for conversational recommendation evaluating comprehensive setting, attribute-based retrieval and embedding-based retrieval. As shown in the table, comprehensive memory augmentation tends to outperform the baseline and LLM-REDIAL model for

Table 4: Results for Movie Conversational Recommendation using (1) Attribute-based retrieval with Claude-3-Sonnet model (2) Embedding-based retrieval across models (Llama v3, Mistral v1, Claude-3-Haiku, and Claude-3-Sonnet) (3) Comprehensive setting using Claude-3-Sonnet that includes ALL augmentations. Evaluation metrics include RECALL, NDCG, and an LLMbased genre matching metric, with $n = 2 0 0$ and $k = 1 0$ . Baseline is Claude-3-Sonnet without augmentation. Best results are in bold.   

<table><tr><td>Model</td><td>Avg. Items Retrieved</td><td colspan="3">Direct Match (↑)</td><td colspan="3">Genre Match (↑)</td><td colspan="3">NDCG(↑)</td></tr><tr><td></td><td></td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>N@1</td><td>N@5</td><td>N@10</td></tr><tr><td>Baseline (Claude-3-Sonnet)</td><td>144</td><td>0.000</td><td>0.010</td><td>0.015</td><td>0.320</td><td>0.57</td><td>0.660</td><td>0.005</td><td>0.007</td><td>0.008</td></tr><tr><td>LLM-REDIAL Model</td><td>144</td><td>-</td><td>0.000</td><td>0.005</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.000</td><td>0.001</td></tr><tr><td colspan="11">Attribute-Based Retrieval</td></tr><tr><td>MemInsight (Claude-3-Sonnet)</td><td>15</td><td>0.005</td><td>0.015</td><td>0.015</td><td>0.270</td><td>0.540</td><td>0.640</td><td>0.005</td><td>0.007</td><td>0.007</td></tr><tr><td colspan="11">Embedding-Based Retrieval</td></tr><tr><td>MemInsight (Llama v3)</td><td>10</td><td>0.000</td><td>0.005</td><td>0.028</td><td>0.380</td><td>0.580</td><td>0.670</td><td>0.000</td><td>0.002</td><td>0.001</td></tr><tr><td>MemInsight (Mistral v1)</td><td>10</td><td>0.005</td><td>0.010</td><td>0.010</td><td>0.380</td><td>0.550</td><td>0.630</td><td>0.005</td><td>0.007</td><td>0.007</td></tr><tr><td>MemInsight (Claude-3-Haiku)</td><td>10</td><td>0.005</td><td>0.010</td><td>0.010</td><td>0.360</td><td>0.610</td><td>0.650</td><td>0.005</td><td>0.007</td><td>0.007</td></tr><tr><td>MemInsight (Claude-3-Sonnet)</td><td>10</td><td>0.005</td><td>0.015</td><td>0.015</td><td>0.400</td><td>0.600</td><td>0.64</td><td>0.005</td><td>0.010</td><td>0.010</td></tr><tr><td colspan="11">Comprehensive</td></tr><tr><td>MemInsight (Claude-3-Sonnet)</td><td>144</td><td>0.010</td><td>0.020</td><td>0.025</td><td>0.300</td><td>0.590</td><td>0.690</td><td>0.010</td><td>0.015</td><td>0.017</td></tr></table>

recall and NDCG metrics. For genre match we find the results to be comparable when considering all attributes. However, attributed-based filtering retrieval still outperforms the LLM-REDIAL model and is comparable to the baseline with almost $90 \%$ less memory retrieved.

Table 5 presents the results of subjective LLMbased evaluation for Persuasiveness and Relatedness. The findings indicate that memory augmentation enhances partial persuasiveness by $10 \mathrm { - } 1 1 \%$ using both comprehensive and attribute-based retrieval, while also reducing unpersuasive recommendations and increasing highly persuasive ones by $4 \%$ in attribute-based retrieval. Furthermore, the results highlights the effectiveness of embeddingbased retrieval, which leads to a $12 \%$ increase in highly persuasive recommendations and enhances all relatedness metrics. This illustrates how MemInsight enriches the recommendation process by incorporating condensed, relevant knowledge, thereby producing more persuasive and related recommendations. However, these improvements were not reflected in recall and NDCG metrics.

# 5.3 Event Summarization

We evaluate the effectiveness of MemInsight in enriching raw dialogues with relevant insights for event summarization. We utilize the generated annotations to identify key events within conversations and hence use them for event summarization. We compare the generated summaries against Lo-CoMo’s event labels as the baseline. Figure 3 illustrates the experimental framework, where the baseline is the raw dialogues sent to the LLM model to generate an event summary, then both event summaries, from raw dialogues and augmentation based summaries, are compared to the ground truth

![](images/827a23027e7294aab84464ccbe6ca4a6d78148ec54570b1e3ed4bc01725f92c1.jpg)  
Figure 3: Evaluation framework for event summarization with MemInsight, exploring augmentation at Turn and Session levels, considering attributes alone or both attributes and dialogues for richer summaries.

summaries in the LoCoMo dataset.

Memory Augmentation In this experiment, we evaluate the effectiveness of augmentation granularity; turn-level dialogue augmentations as opposed to session-level dialogue annotations. We additionally, consider studying the effectiveness of using only the augmentations to generate the event summaries as opposed to using both the augmentations and their corresponding dialogue content.

Experimental Results As shown in Table 6, our MemInsight model achieves performance comparable to the baseline, despite relying only on dialogue turns or sessions containing the event label. Notably, turn-level augmentations provided more precise and detailed event information, leading to improved performance over both the baseline and session-level annotations.

For Claude-3-Sonnet, all metrics remain comparable, indicating that memory augmentations effectively capture the semantics within dialogues at both the turn and session levels. This proves that the augmentations sufficiently enhance context representation for generating event summaries. To further investigate how backbone LLMs impact aug-

<table><tr><td>Model</td><td>Avg. Items 
Retrieved</td><td colspan="3">LLM-Persuasiveness %</td><td colspan="3">LLM-Relatedness%</td></tr><tr><td></td><td></td><td>Unpers*</td><td>Partially Pers.</td><td>Highly Pers.</td><td>Not Comp*</td><td>Comp</td><td>Match</td></tr><tr><td>Baseline (Claude-3-Sonnet)</td><td>144</td><td>16.0</td><td>64.0</td><td>13.0</td><td>57.0</td><td>41.0</td><td>2.0</td></tr><tr><td>Attribute-Based Retrieval</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MemInsight (Claude-3-Sonnet)</td><td>15</td><td>2.0</td><td>75.0</td><td>17.0</td><td>40.5</td><td>54.0</td><td>2.0</td></tr><tr><td>Embedding-Based Retrieval</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MemInsight (Llama v3)</td><td>10</td><td>11.3</td><td>63.0</td><td>20.4</td><td>19.3</td><td>80.1</td><td>0.5</td></tr><tr><td>MemInsight (Mistral v1)</td><td>10</td><td>16.3</td><td>61.2</td><td>18.0</td><td>16.3</td><td>82.5</td><td>5.0</td></tr><tr><td>MemInsight (Claude-3-Haiku)</td><td>10</td><td>1.6</td><td>53.0</td><td>25.0</td><td>23.3</td><td>74.4</td><td>2.2</td></tr><tr><td>MemInsight (Claude-3-Sonnet)</td><td>10</td><td>2.0</td><td>59.5</td><td>20.0</td><td>29.5</td><td>68.0</td><td>2.5</td></tr><tr><td>Comprehensive</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MemInsight (Claude-3-Sonnet)</td><td>144</td><td>2.0</td><td>74.0</td><td>12.0</td><td>42.5</td><td>56.0</td><td>1.0</td></tr></table>

Table 5: Movie Recommendations results (with similar settings to Table 4) using LLM-based metrics; (1) Persuasiveness— $\%$ of Unpersuasive (lower is better), Partially, and Highly Persuasive cases. (2) Relatedness— $\%$ of Not Comparable (lower is better), Comparable, and Exactly Matching cases. Best results are in bold. Comprehensive setting includes ALL augmentations. Totals may NOT sum to $100 \%$ due to cases the LLM model could not evaluate.   
Table 6: Event Summarization results using G-Eval metrics (higher is better): Relevance, Coherence, and Consistency. Comparing summaries generated with augmentations only at Turn-Level (TL) and Session-Level (SL) and summaries generated using both augmentations and dialogues (MemInsight +Dialogues) at TL and SL. Best results are in bold.   

<table><tr><td>Model</td><td colspan="3">Claude-3-Sonnet</td><td colspan="3">Llama v3</td><td colspan="3">Mistral v1</td><td colspan="3">Claude-3-Haiku</td></tr><tr><td></td><td>Rel.</td><td>Coh.</td><td>Con.</td><td>Rel.</td><td>Coh.</td><td>Con.</td><td>Rel.</td><td>Coh.</td><td>Con.</td><td>Rel.</td><td>Coh.</td><td>Con.</td></tr><tr><td>Baseline Summary</td><td>3.27</td><td>3.52</td><td>2.86</td><td>2.03</td><td>2.64</td><td>2.68</td><td>3.39</td><td>3.71</td><td>4.10</td><td>4.00</td><td>4.4</td><td>3.83</td></tr><tr><td>MemInsight (TL)</td><td>3.08</td><td>3.33</td><td>2.76</td><td>1.57</td><td>2.17</td><td>1.95</td><td>2.54</td><td>2.53</td><td>2.49</td><td>3.93</td><td>4.3</td><td>3.59</td></tr><tr><td>MemInsight (SL)</td><td>3.08</td><td>3.39</td><td>2.68</td><td>2.0</td><td>2.62</td><td>3.67</td><td>4.13</td><td>4.41</td><td>4.29</td><td>3.96</td><td>4.30</td><td>3.77</td></tr><tr><td>MemInsight +Dialogues (TL)</td><td>3.29</td><td>3.46</td><td>2.92</td><td>2.45</td><td>2.19</td><td>2.87</td><td>4.30</td><td>4.53</td><td>4.60</td><td>4.23</td><td>4.52</td><td>4.16</td></tr><tr><td>MemInsight +Dialogues (SL)</td><td>3.05</td><td>3.41</td><td>2.69</td><td>2.24</td><td>2.80</td><td>3.86</td><td>4.04</td><td>4.48</td><td>4.33</td><td>3.93</td><td>4.33</td><td>3.73</td></tr></table>

Table 7: Results for Event Summarization using Llama v3, where the baseline is the model without augmentation as opposed to the augmentation model (turn-level) using Claude-3-Sonnet vs Llama v3.   

<table><tr><td>Model</td><td colspan="3">G-Eval % (↑)</td></tr><tr><td></td><td>Rel.</td><td>Coh.</td><td>Con.</td></tr><tr><td>Baseline(Llama v3)</td><td>2.03</td><td>2.64</td><td>2.68</td></tr><tr><td>Llama v3 + Llama v3</td><td>2.45</td><td>2.19</td><td>2.87</td></tr><tr><td>Claude-3-Sonnet + Llama v3</td><td>3.15</td><td>3.59</td><td>3.17</td></tr></table>

mentation quality, we employed Claude-3-Sonnet as opposed to Llama v3 for augmentation while still using Llama for event summarization. As presented in Table 7, Sonnet augmentations resulted in improved performance for all metrics, providing empirical evidence for the effectiveness and stability of Sonnet in augmentation. Additional experiments and detailed analysis are provided in Appendix E.4.

Qualitative Analysis To more rigorously assess the quality of the autonomously generated augmentations, we conduct a qualitative analysis of the annotations produced by Claude-3 Sonnet. Using the DeepEval hallucination metric (Yang et al., 2024), we find that $9 9 . 1 4 \%$ of the annotations are grounded in the dialogue, demonstrating a high level of factual consistency. The remaining $0 . 8 6 \%$ primarily consist of abstract or generic attributes, rather than explicit inaccuracies. Additional experimental details and examples are provided in

Appendix F.

# 6 Conclusion

This paper introduced MemInsight, an autonomous memory augmentation framework that enhances LLM agents’ memory through structured, attributebased augmentations. While maintaining competitive performance on standard metrics, MemInsight achieves substantial improvements in LLM-based evaluation scores, demonstrating its effectiveness in capturing semantic relevance and improving performance across tasks and datasets. Experimental results show that both attribute-based filtering and embedding-based retrieval methods effectively leverage the generated augmentations. Prioritybased augmentation, in particular, improves similarity search and retrieval accuracy. MemInsight also complements traditional RAG models by enabling customized, attribute-guided retrieval, enhancing the integration of memory with LLM reasoning. Moreover, in benchmark comparisons, MemInsight consistently outperforms baseline models in overall accuracy and delivers stronger performance in recommendation tasks, yielding more persuasive outputs. Qualitative analysis further confirms the high factual consistency of the generated annotations. These results highlight MemInsight’s potential as a scalable memory solution for LLM agents.

# 7 Limitations

While MemInsight demonstrates strong performance across multiple tasks and datasets, several limitations remain and highlight areas for future exploration. Although the model autonomously generates augmentations, it may occasionally produce abstract or overly generic annotations, especially in ambiguous dialogue contexts. While these are not factually incorrect, they may reduce retrieval specificity in tasks requiring fine-grained memory access. Additionally, MemInsight ’s performance is dependent on the capabilities of the underlying LLM used for attribute generation. Less capable or unaligned models may produce less consistent augmentations. We also acknowledge that our current implementation is limited to text-based interactions. Future work could extend MemInsight to support multimodal inputs, such as images or audio, enabling richer and more comprehensive contextual representations.

# References

Petr Anokhin, Nikita Semenov, Artyom Sorokin, Dmitry Evseev, Mikhail Burtsev, and Evgeny Burnaev. 2024. Arigraph: Learning knowledge graph world models with episodic memory for llm agents. Preprint, arXiv:2407.04363.   
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. Preprint, arXiv:2504.19413.   
Yubin Ge, Salvatore Romeo, Jason Cai, Raphael Shu, Monica Sunkara, Yassine Benajiba, and Yi Zhang. 2025. Tremu: Towards neuro-symbolic temporal reasoning for llm-agents with memory in multi-session dialogues. arXiv preprint arXiv:2502.01630.   
Yuki Hou, Haruki Tamoto, and Homei Miyashita. 2024. “my agent understands me better”: Integrating dynamic human-like memory recall and consolidation in llm-based agents. In Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, page 1–7. ACM.   
Chenxu Hu, Jie Fu, Chenzhuang Du, Simian Luo, Junbo Zhao, and Hang Zhao. 2023a. Chatdb: Augmenting llms with databases as their symbolic memory. arXiv preprint arXiv:2306.03901.   
Chenxu Hu, Jie Fu, Chenzhuang Du, Simian Luo, Junbo Zhao, and Hang Zhao. 2023b. Chatdb: Augmenting llms with databases as their symbolic memory. Preprint, arXiv:2306.03901.   
Mengkang Hu, Tianxing Chen, Qiguang Chen, Yao Mu, Wenqi Shao, and Ping Luo. 2024. Hiagent: Hierarchical working memory management for solving

long-horizon agent tasks with large language model. Preprint, arXiv:2408.09559.   
Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with gpus. Preprint, arXiv:1702.08734.   
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick˘ Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen tau Yih. 2020. Dense passage retrieval for open-domain question answering. Preprint, arXiv:2004.04906.   
Kuang-Huei Lee, Xinyun Chen, Hiroki Furuta, John Canny, and Ian Fischer. 2024. A human-inspired reading agent with gist memory of very long contexts. Preprint, arXiv:2402.09727.   
Tingting Liang, Chenxin Jin, Lingzhi Wang, Wenqi Fan, Congying Xia, Kai Chen, and Yuyu Yin. 2024. LLM-REDIAL: A large-scale dataset for conversational recommender systems created from user behaviors with LLMs. In Findings of the Association for Computational Linguistics: ACL 2024, pages 8926–8939, Bangkok, Thailand. Association for Computational Linguistics.   
Lei Liu, Xiaoyan Yang, Yue Shen, Binbin Hu, Zhiqiang Zhang, Jinjie Gu, and Guannan Zhang. 2023a. Think-in-memory: Recalling and post-thinking enable llms with long-term memory. Preprint, arXiv:2311.08719.   
Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023b. G-eval: Nlg evaluation using gpt-4 with better human alignment. Preprint, arXiv:2303.16634.   
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of llm agents. Preprint, arXiv:2402.17753.   
Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. Memgpt: Towards llms as operating systems. Preprint, arXiv:2310.08560.   
Gustavo Penha, Ali Vardasbi, Enrico Palumbo, Marco de Nadai, and Hugues Bouchard. 2024. Bridging search and recommendation in generative retrieval: Does one task help the other? Preprint, arXiv:2410.16823.   
Jihoon Tack, Jaehyung Kim, Eric Mitchell, Jinwoo Shin, Yee Whye Teh, and Jonathan Richard Schwarz. 2024. Online adaptation of language models with a memory of amortized contexts. arXiv preprint arXiv:2403.04317.   
Zhengyang Tang, Ziniu Li, Zhenyang Xiao, Tian Ding, Ruoyu Sun, Benyou Wang, Dayiheng Liu, Fei Huang, Tianyu Liu, Bowen Yu, and Junyang Lin. 2025. Enabling scalable oversight via self-evolving critic. Preprint, arXiv:2501.05727.

Junlin Wang, Jue Wang, Ben Athiwaratkun, Ce Zhang, and James Zou. 2024a. Mixture-of-agents enhances large language model capabilities. Preprint, arXiv:2406.04692.   
Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. 2024b. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6).   
Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong, and Yangqiu Song. 2024c. Rethinking the bounds of llm reasoning: Are multi-agent discussions the key? Preprint, arXiv:2402.18272.   
Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong, and Yangqiu Song. 2024d. Rethinking the bounds of llm reasoning: Are multi-agent discussions the key? Preprint, arXiv:2402.18272.   
Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong, and Yangqiu Song. 2024e. Rethinking the bounds of llm reasoning: Are multi-agent discussions the key? Preprint, arXiv:2402.18272.   
Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. 2024f. Agent workflow memory. Preprint, arXiv:2409.07429.   
Wujiang Xu, Kai Mei, Hang Gao, Juntao Tan, Zujie Liang, and Yongfeng Zhang. 2025. A-mem: Agentic memory for llm agents. Preprint, arXiv:2502.12110.   
Yixin Yang, Zheng Li, Qingxiu Dong, Heming Xia, and Zhifang Sui. 2024. Can large multimodal models uncover deep semantics behind images? Preprint, arXiv:2402.11281.   
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. Preprint, arXiv:2210.03629.   
Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. 2024. A survey on the memory mechanism of large language model based agents. Preprint, arXiv:2404.13501.   
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024a. Expel: Llm agents are experiential learners. Preprint, arXiv:2308.10144.   
Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024b. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 19632–19642.   
Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2023. Memorybank: Enhancing large language models with long-term memory. Preprint, arXiv:2305.10250.

Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, Yu Qiao, Zhaoxiang Zhang, and Jifeng Dai. 2023. Ghost in the minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory. Preprint, arXiv:2305.17144.

# A Ethical Consideration

We have thoroughly reviewed the licenses of all scientific artifacts, including datasets and models, ensuring they permit usage for research and publication purposes. To protect anonymity, all datasets used are de-identified. Our proposed method demonstrates considerable potential in significantly reducing both the financial and environmental costs typically associated with enhancing large language models. By lessening the need for extensive data collection and human labeling, our approach not only streamlines the process but also provides an effective safeguard for user and data privacy, reducing the risk of information leakage during training corpus construction. Additionally, throughout the paper-writing process, Generative AI was exclusively utilized for language checking, paraphrasing, and refinement.

# B Autonomous Memory Augmentation

# B.1 Attribute Mining

Figure 4 illustrates examples for the two types of attribute augmentation: entity-centric and conversation-centric. The entity-centric augmentation represents the main attributes generated for the book entitled ’Already Taken’, where attributes are derived based on entity-specific characteristics such as genre, author, and thematic elements. The conversation-centric example illustrates the augmentation generated for a sample two turns dialogue from the LLM-REDIAL dataset, highlighting attributes that capture contextual elements such as user intent, motivation, emotion, perception, and genre of interest.

Furthermore, Figure 5 presents an overview of the top five attributes across different domains in the LLM-REDIAL dataset. These attributes represent the predominant attributes specific to each domain, highlighting the significance of different attributes in augmentation generation. Consequently, the integration of priority-based embeddings has led to improved performance.

# C Embedding-based Retrieval

In the context of embedding-based memory retrieval, movies are augmented using MemInsight, and the generated attributes are embedded to retrieve relevant movies from memory. Two main embedding methods were considered:

(1) Averaging Over Independent Embeddings Each attribute and its corresponding value in the generated augmentations is embedded independently. The resulting attribute embeddings are then averaged across all attributes to generate the final embedding vector representation, as illustrated in Figure 6 which are subsequently used in similarity search to retrieve relevant movies.

(2) All Augmentations Embedding In this method, all generated augmentations, including all attributes and their corresponding values, are encoded into a single embedding vector and stored for retrieval as shown in Figure 6. Additionally, Figure 7 presents the cosine similarity results for both methods. As depicted in the figure, averaging over all augmentations produces a more consistent and reliable measure, as it comprehensively captures all attributes and effectively differentiates between similar and distinct characteristics. Consequently, this method was adopted in our experiments.

# D Question Answering

# D.1 Prompts

Table 8 outlines the prompts used in the Question Answering task for generating augmentations in both questions and conversations.

# E Conversational Recommendation

# E.1 Prompts

Table 9 presents the prompts used in Conversational Recommendation for movie recommendations, incorporating both basic and priority augmentations.

# E.2 Evaluation Framework

Figure 8 presents the evaluation framework for the Conversation Recommendation task. The process begins with (1) augmenting all movies in memory using entity-centric augmentations to enhance retrieval effectiveness. (2) Next, all dialogues in the dataset are prepared to simulate the recommendation process by masking the ground truth labels and prompting the LLM to find the masked labels based on augmentations from previous user interactions. (3) Recommendations are then generated using the retrieved memory, which may be attributebased—for instance, filtering movies by specific attributes such as genre or using embedding-based retrieval. (4) Finally, the recommended movies are evaluated against the ground truth labels to assess

![](images/ed7e772af0fa3338fd98ccfc0690dcd5a04af320bdcfc2e4634dc3d92bd4e0d7.jpg)  
Figure 4: An example of entity-centric augmentation for the book ’Already Taken’, and a conversation-centric augmentation for a sample dialogue from the LLM-REDIAL dataset.

![](images/41fa27939da56165b4bc0bbadb92b41239b7c76bad06829e1fbc123b85fe2309.jpg)

![](images/a65e5b5c451e31ed65f12025882feee8dc7f0c977adc5b3457b4a3833d20a2e3.jpg)

![](images/3d767c4cf1717345c453b1a6c3077a60cf905925dacee821c31e13a5dc5770ec.jpg)

![](images/14c5de0a99fddda0c0b5937a835ddebb4cdbee505adccf9e7db0fc83de5afde9.jpg)  
Attributes   
Figure 5: Top 10 attributes by frequency in the LLM-REDIAL dataset across domains (Movies, Sports Items, Electronics, and Books) using MemInsight Attribute Mining. Frequency indicates how often each attribute was generated to augment different movies.

the accuracy and effectiveness of the retrieval and recommendation approach.

# E.3 Event Summarization

# E.3.1 Prompts

Table 10 presents the prompt used in Event Summarization to augment dialogues by generating relevant attributes. In this process, only attributes related to events are considered to effectively summarize key events from dialogues, ensuring a focused and structured summarization approach.

# E.4 Additional Experiments

In this experiment, we include an additional baseline for event summarization: raw summaries generated directly by LLMs using zero-shot prompting, without any memory augmentation. This serves as a clear reference point to isolate the impact of MemInsight ’s augmentation strategy on summa-

rization quality. Table 11 shows the results of this experiment. As illustrated, MemInsight consistently improves event summarization quality across models, with the best performance achieved when augmentations are integreted with dialogue context highlighting the value of fine-grained annotations and contextual grounding. Overall, the findings confirm that MemInsight enhances the factual and semantic quality of generated summaries.

# F Qualitative Analysis

Figure 9 illustrates the augmentations generated using different LLM models, including Claude-Sonnet, Llama, and Mistral for a dialogue turn from the LoCoMo dataset. As depicted in the figure, augmentations produced by Llama include hallucinations, generating information that does not exist. In contrast, Figure 10 presents the augmentations for the subsequent dialogue turn using the

![](images/7905b844054f7d563fac4166f0bac613b690dc0ed04b5943c393c05689f26b64.jpg)  
Movie Augmentations   
[ttribute1]<value>|[Attribute2]<value>[Attrbute3]<value>[Attribute4]<value> Movie   
Embedding-based Retrieval

![](images/b0ca6a6350637def16df171712abf6bc15148f9143ac770a8e11c5d37d2a5c5f.jpg)  
Figure 6: Embedding methods for Embedding-based retrieval methods using generated Movie augmentations including (a) Averaging over Independent Embeddings and (b) All Augmentations Embedding.   
(a) Averaging over Independent Embeddings

![](images/8ec4352947b0103cb2902939bf4959ba7400c45f707a2d50c465705c56c2df5c.jpg)  
(b) All Augmentations Embedding   
Figure 7: An illustrative example of augmentation embedding methods for three movies: (1) The Departed, (2) Shutter Island, and (3) The Hobbit. Movies 1 and 2 share similar attributes, whereas movies 1 and 3 differ. Te top 5 attributes of every movie were selected for a simplified illustration.

same models. Notably, Claude-Sonnet maintains consistency across both turns, suggesting its stable performance throughout all experiments. While Mistral model tend to be less stable as it included attributes that are not in the dialogue. A hallucination evaluation conducted using DeepEval yielded a score of $9 9 . 1 4 \%$ , indicating strong factual consistency. Table 12 presents examples of annotations with lower scores. While these annotations are more generic or abstract, they remain semantically aligned with the original input.

![](images/5bb24b1fa2f66fa0c74f9e986b750a86ad36f5aac133296028b7aa69c8bae4e2.jpg)  
Figure 8: Evaluation Framework for Conversation Recommendation Task.

![](images/a7af147dc186e0a9f2e733276fff2cfbead7d90068a2607708a283a6c422ea2e.jpg)  
Figure 9: Augmentation generated on a Turn-level for a sample dialogue turn from the LoCoMo dataset using Claude-3-Sonnet, Llama v3 and Mistral v1 models.

![](images/d6d3dc41691ec821c471707df22cef134f552b1a3cf78ffed4dec93d28a7567e.jpg)  
Figure 10: Augmentations generated for the turn following the turn in Figure 9 using Claude-3-Sonnet, Llama v3 and Mistral v1 models. Hallucinations are presented in red.

<table><tr><td>Question Augmentation</td></tr><tr><td>Given the following question, determine what are the main inquiry attribute to look for and the person the question is for. Respond in the format: Person:[names]Attributes:[].</td></tr><tr><td>Basic Augmentation</td></tr><tr><td>You are an expert annotator who generates the most relevant attributes in a conversation. Given the conversation below, identify the key attributes and their values on a turn by turn level. 
Attributes should be specific with most relevant values only. Don’t include speaker name. Include value information that you find relevant and their names if mentioned. Each dialogue turn contains a dialogue id between [ ]. Make sure to include the dialogue the attributes and values are extracted form. Important: Respond only in the format [{{speaker name}:[Dialog id]:[attribute]&lt;value&gt;}] . Dialogue Turn:[}</td></tr><tr><td>Priority Augmentation</td></tr><tr><td>You are an expert dialogue annotator, given the following dialogue turn generate a list of attributes and values for relevant information in the text. 
Generate the annotations in the format: [attribute]&lt;value&gt;where attribute is the attribute name and value is its corresponding value from the text. 
and values for relevant information in this dialogue turn with respect to each person. Be concise and direct. 
Include person name as an attribute and value pair. 
Please make sure you read and understand these instructions carefully. 
1- Identify the key attributes in the dialogue turn and their corresponding values. 
2- Arrange attributes descendingly with respect to relevance from left to right. 
3- Generate the sorted annotations list in the format: [attribute]&lt;value&gt;where attribute is the attribute name and value is its corresponding value from the text. 
4- Skip all attributes with none vales 
Important: YOU MUST put attribute name is between [ ] and value between &lt;&gt; . Only return a list of [attribution]&lt;value&gt;nothing else. Dialogue Turn:[}</td></tr></table>

Table 8: Prompts used in Question Answering for generating augmentations for questions. Also, augmentations for conversations, utilizing both basic and priority augmentations.   

<table><tr><td>Basic Augmentation</td></tr><tr><td>For the following movie identify the most important attributes independently. Determine all attributes that describe the movie based on your knowledge of this movie. Choose attribute names that are common characteristics of movies in general. Respond in the following format: [attribute]. The Movie is: {}</td></tr><tr><td>Priority Augmentation</td></tr><tr><td>You are a movie annotation expert tasked with analyzing movies and generating key-attribute pairs. For the following movie identify the most important. Determine all attribute that describe the movie based on your knowledge of this movie. Choose attribute names that are common characteristics of movies in general. Respond in the following format: [attribute]. Sort attributes from left to right based on their relevance. The Movie is: {}</td></tr><tr><td>Dialogue Augmentation</td></tr><tr><td>Identify the key attributes that best describe the movie the user wants for recommendation in the dialogue. These attributes should encompass movie features that are relevant to the user sorted descendingly with respect to user interest. Respond in the format: [attribute].</td></tr></table>

Table 9: Prompts used in Conversational Recommendation for recommending Movies utilizing both basic and priority augmentations.   

<table><tr><td>Dialogue Augmentation</td></tr><tr><td>Given the following attributes and values that annotate a dialogue for every speaker in the format [attribute]&lt;value&gt;, generate a summary for the event attributes only to describe the main and important events represented in these annotations. Refrain from mentioning any minimal event. Include any event-related details and speaker. Format: a bullet paragraph for major life events for every speaker with no special characters. Don’t include anything else in your response or extra text or lines. Don’t include bullets. Input annotations: {}</td></tr></table>

Table 10: Prompt used in Event Summarization to augment dialogues   
Table 11: LLM-based evaluation scores for event summarization using relevance (Rel.), coherence (Coh.), and consistency (Con.) across different models and augmentation settings. Baseline summaries are generated using zero-shot prompting without memory augmentation. MemInsight is evaluated in both turn-level (TL) and sessionlevel (SL) configurations, with and without access to dialogue context.   

<table><tr><td>Model</td><td colspan="3">Llama v3</td><td colspan="3">Mistral v1</td><td colspan="3">Claude-3 Haiku</td><td colspan="3">Claude-3 Sonnet</td></tr><tr><td></td><td>Rel.</td><td>Coh.</td><td>Con.</td><td>Rel.</td><td>Coh.</td><td>Con.</td><td>Rel.</td><td>Coh.</td><td>Con.</td><td>Rel.</td><td>Coh.</td><td>Con.</td></tr><tr><td>Baseline LLM Summary</td><td>2.23</td><td>2.66</td><td>2.63</td><td>3.34</td><td>3.77</td><td>4.11</td><td>3.97</td><td>4.33</td><td>3.79</td><td>3.27</td><td>3.64</td><td>2.78</td></tr><tr><td>MemInsight (TL)</td><td>1.60</td><td>2.17</td><td>1.95</td><td>2.53</td><td>2.49</td><td>2.38</td><td>3.98</td><td>4.37</td><td>3.66</td><td>3.09</td><td>3.27</td><td>2.77</td></tr><tr><td>MemInsight (SL)</td><td>1.80</td><td>2.62</td><td>3.67</td><td>4.09</td><td>4.38</td><td>4.19</td><td>3.94</td><td>4.31</td><td>3.69</td><td>3.08</td><td>3.39</td><td>2.68</td></tr><tr><td>MemInsight + Dialogues (TL)</td><td>2.41</td><td>2.79</td><td>3.01</td><td>4.30</td><td>4.53</td><td>4.60</td><td>4.24</td><td>4.43</td><td>4.16</td><td>3.25</td><td>3.43</td><td>2.86</td></tr><tr><td>MemInsight + Dialogues (SL)</td><td>2.01</td><td>2.70</td><td>3.86</td><td>4.04</td><td>4.48</td><td>4.34</td><td>3.95</td><td>4.33</td><td>3.71</td><td>3.02</td><td>3.37</td><td>2.73</td></tr></table>

Table 12: MemInsight annotations that scored below $1 \%$ hallucination rate in the DeepEval hallucination evaluation.   

<table><tr><td>Input</td><td>Augmentations</td><td>Hall. Score</td></tr><tr><td>‘Evan’: [[“Evan&#x27;s son had an accident where he fell off his bike last Tuesday but is doing better now.&quot;, D20:3], [&quot;Evan is supportive and encouraging towards Sam, giving advice to believe in himself and take things one day at a time.&quot;, D20:9], [&quot;Evan is a painter who finished a contemporary figurative painting emphasizing emotion and introspection.&quot;, D20:15], [&quot;Evan had a painting published in an exhibition with the help of a close friend.&quot;, D20:17]], &#x27;Sam&#x27;: [[“Sam used to love hiking but hasn&#x27;t had the chance to do it recently.&quot;, D20:6], [&quot;Sam is struggling with weight and confidence issues, feeling like they lack motivation.&quot;, D20:8], [&quot;Sam acknowledges that trying new things can be difficult.&quot;, D20:12]]</td><td>&quot;evan&quot;: {&quot;[event]: &quot;&lt;son&#x27;s accident&gt;&quot;, &quot;[emotion&quot;:&quot;&lt;worry&gt;&quot;, &quot;[hobby&quot;:&quot;&lt;hiking&gt;&quot;, &quot;[activity&quot;:&quot;&lt;painting&gt;&quot;, &quot;sam&quot;: {&quot;[emotion&quot;:&quot;&lt;struggling&gt;&quot;, &quot;[issue&quot;:&quot;&lt;weight&gt;&quot;, &quot;[emotion&quot;:&quot;&lt;lack of confidence&gt;&quot;, &quot;[action&quot;:&quot;&lt;trying new things&gt;&quot;}}</td><td>0.66</td></tr><tr><td>{&#x27;James&#x27;: [[“James has a dog named Ned that he adopted and can&#x27;t imagine life without.&quot;, D21:3], [&quot;James is interested in creating a strategy game similar to Civilization.&quot;, D21:9], [&quot;James suggested meeting at Starbucks for coffee with John.&quot;, D21:13}], &#x27;John&#x27;: [[“John helps his younger siblings with programming and is proud of their progress&quot;, D21:2], [&quot;John is working on a coding project with his siblings involving a text-based adventure game.&quot;, D21:6], [&quot;John prefers light beers over dark beers when going out.&quot;, D21:16], [&quot;John agreed to meet James at McGee&#x27;s Pub after discussing different options.&quot;, D21:18]]}</td><td>{&quot;james&quot;: {&quot;[emotion&quot;:&quot;&lt;excited&gt;&quot;, &quot;[intent&quot;:&quot;&lt;socializing&gt;&quot;, &quot;[topic&quot;:&quot;&lt;dogs&gt;&quot;, &quot;[topic&quot;:&quot;&lt;gaming&gt;&quot;, &quot;[topic&quot;:&quot;&lt;starbucks&gt;&quot;, &quot;[topic&quot;:&quot;&lt;pubMeeting&gt;&quot;, &quot;[activity&quot;:&quot;&lt;coffee&gt;&quot;, &quot;[activity&quot;:&quot;&lt;beer&gt;&quot;, &quot;john&quot;: {&quot;[topic&quot;:&quot;&lt;siblings&gt;&quot;, &quot;[topic&quot;:&quot;&lt;programming&gt;&quot;, &quot;[activity&quot;:&quot;&lt;adventure game&gt;&quot;, &quot;[emotion&quot;:&quot;&lt;proud&gt;&quot;, &quot;[intent&quot;:&quot;&lt;socializing&gt;&quot;}}</td><td>0.50</td></tr></table>
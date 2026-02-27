# Memory3: Language Modeling with Explicit Memory

Hongkang Yang1, Zehao Lin1, Wenjin Wang1, Hao Wu1, Zhiyu $\mathrm { L i ^ { 1 } }$ , Bo Tang1, Wenqiang Wei1, Jinbo Wang1, Zeyun Tang1, Shichao $\mathrm { { S o n g ^ { 1 } } }$ , Chenyang $\mathrm { \ X i ^ { 1 } }$ , Yu $\mathrm { Y u ^ { 1 } }$ , Kai Chen1, Feiyu Xiong1, Linpeng Tang2, and Weinan $\mathrm { E ^ { * 3 , 1 } }$

1Center for LLM, Institute for Advanced Algorithms Research, Shanghai 2Moqi Inc 3Center for Machine Learning Research, Peking University

July 1, 2024

# Abstract

The training and inference of large language models (LLMs) are together a costly process that transports knowledge from raw data to meaningful computation. Inspired by the memory hierarchy of the human brain, we reduce this cost by equipping LLMs with explicit memory, a memory format cheaper than model parameters and text retrieval-augmented generation (RAG). Conceptually, with most of its knowledge externalized to explicit memories, the LLM can enjoy a smaller parameter size, training cost, and inference cost, all proportional to the amount of remaining “abstract knowledge”. As a preliminary proof of concept, we train from scratch a 2.4B LLM, which achieves better performance than much larger LLMs as well as RAG models, and maintains higher decoding speed than RAG. The model is named Memory3, since explicit memory is the third form of memory in LLMs after implicit memory (model parameters) and working memory (context key-values). We introduce a memory circuitry theory to support the externalization of knowledge, and present novel techniques including a memory sparsification mechanism that makes storage tractable and a two-stage pretraining scheme that facilitates memory formation.

![](images/c3480604f127efc6ff60247f3ae3e44e368147f4a9d780f4b3c1df544dca6001.jpg)  
Figure 1: The Memory3 model converts texts to explicit memories, and then recalls these memories during inference. The explicit memories can be seen as retrievable model parameters, externalized knowledge, or sparsely-activated neural circuits.

![](images/92113c05894deaf5b76a195f5796b42415400adfa9760e96c4da2b98a735a3b5.jpg)

![](images/5d8b2a64d28bab4e3d6ef7cd795a1c0c7a5f711e7e0c1a055a2da7e0821ad58d.jpg)  
Figure 2: Left: Performance on benchmarks, with respect to model size (top-left is better). Right: Retrieval-augmented performance on professional tasks, versus decoding speed with retrieval (top-right is better). The left plot is based on Table 16. The right plot is based on Tables 20 and 21. Memory3 uses high frequency retrieval of explicit memories, while the RAG models use a fixed amount of 5 references. This is a preliminary experiment and we have not optimized the quality of our pretraining data as well as the efficiency of our inference pipeline, so the results may not be comparable to those of the SOTA models.

# 1 | Introduction

Large language models (LLMs) have enjoyed unprecedented popularity in recent years thanks to their extraordinary performance [5, 9, 110, 11, 126, 4, 56, 54]. The prospect of scaling laws [60, 53, 99] and emergent abilities [119, 105] constantly drives for substantially larger models, resulting in the rapid increase in the cost of LLM training and inference. People have been trying to reduce this cost through optimizations in various aspects, including architecture [40, 6, 30, 75, 89, 109], data quality [104, 58, 48, 66], operator [32, 63], parallelization [95, 103, 62, 91], optimizer [71, 124, 117], scaling laws [53, 127], generalization theory [132, 55], hardware [33], etc.

We introduce the novel approach of optimizing knowledge storage. The combined cost of LLM training and inference can be seen as the cost of encoding the knowledge from text data into various memory formats, plus the cost of reading from these memories during inference:

$$
\sum_ {\text {k n o w l e d g e} k} \min  _ {\text {f o r m a t} m} \operatorname {c o s t} _ {\text {w r i t e}} (k, m) + n _ {k} \cdot \operatorname {c o s t} _ {\text {r e a d}} (k, m) \tag {1}
$$

where $\mathbf { c o s t } _ { \mathbf { w r i t e } }$ is the cost of encoding a piece of knowledge $k$ into memory format $_ m$ , $\mathbf { c o s t } _ { \mathbf { r e a d } }$ is the cost of integrating $k$ from format m into inference, and $n _ { k }$ is the expected usage count of this knowledge during the lifespan of this LLM (e.g. a few months for each version of ChatGPT [86, 102]). The definitions of knowledge and memory in the context of LLMs are provided in Section 2, and this paper uses knowledge as a countable noun. Typical memory formats include model parameters and plain text for retrievalaugmented generative models (RAG); their write functions and read functions are listed in Table 3, and their costwrite and $\mathrm { c o s t } _ { \mathrm { r e a d } }$ are provided in Figure 4.

We introduce a new memory format, explicit memory, characterized by moderately low write cost and read cost. As depicted in Figure 1, our model first converts a knowledge base (or any text dataset) into explicit memories, implemented as sparse attention key-values, and then during inference, recalls these memories and integrates them into the self-attention layers. Our design is simple so that most of the existing Transformer-based LLMs should be able to accommodate explicit memories with a little finetuning, and thus it is a general-purpose “model amplifier”. Eventually, it should reduce the cost of pretraining LLMs, since there will be much less knowledge that must be stored in parameters, and thus less training data and smaller model size.

The new memory format enables us to define a memory hierarchy for LLMs:

$$
\text {p l a i n} \quad \mathrm {(R A G)} \rightarrow \text {e x p l i c i t m e m o r y} \rightarrow \text {m o d e l p a r a m e t e r}
$$

such that by going up the hierarchy, costwrite increases while costread decreases. To minimize the cost (1), one should store each piece of knowledge that is very frequently/rarely used in the top/bottom of this hierarchy, and everything in between as explicit memory. As illustrated in Table 3, the memory hierarchy

of LLMs closely resembles that of humans. For humans, the explicit/implicit memories are the long-term memories that are acquired and used consciously/unconsciously [59].

Table 3: Analogy of the memory hierarchies of humans and LLMs.   

<table><tr><td>Memory format of humans</td><td>Example</td><td>Memory format of LLMs</td><td>Write</td><td>Read</td></tr><tr><td>Implicit memory</td><td>common expressions</td><td>model parameters</td><td>training</td><td>matrix multiplication</td></tr><tr><td>Explicit memory</td><td>books read</td><td>this work</td><td>memory encoding</td><td>self-attention</td></tr><tr><td>External information</td><td>open-book exam</td><td>plain text (RAG)</td><td>none</td><td>encode from scratch</td></tr></table>

As a remark, one can compare the plain LLMs to patients with impaired explicit memory, e.g. due to injury to the medial temporal lobe. These patients are largely unable to learn semantic knowledge (usually stored as explicit memory), but can acquire sensorimotor skills through repetitive priming (stored as implicit memories) [42, 26, 12]. Thus, one may hypothesize that due to the lack of explicit memory, the training of plain LLMs is as inefficient as repetitive priming, and thus has ample room for improvement. In analogy with humans, for instance, it is easy to recall and talk about a book we just read, but to recite it as unconsciously as tying shoe laces requires an enormous effort to force this knowledge into our muscle memory. From this perspective, it is not surprising that LLM training consumes so much data and energy [121, 77]. We want to rescue LLMs from this poor condition by equipping it with an explicit memory mechanism as efficient as that of humans.

![](images/484f2c7f26cb69b0fe4e09fbb07e2f6bd7f75adb3f338f5ce53446d79ecf1e5d.jpg)  
Figure 4: The total cost (TFlops) of writing and reading a piece of knowledge by our 2.4B model with respect to its expected usage count. The curves represent the cost of different memory formats, and the shaded area represents the minimum cost given the optimal format. The plot indicates that (0.494, 13400) is the advantage interval for explicit memory. The calculations are provided in Appendix A. (The blue curve is only a lower bound on the cost of model parameters.)

A quantitative illustration of the cost (1) is given by Figure 4, where we characterize costwrite and $\mathrm { c o s t } _ { \mathrm { r e a d } }$ by the amount of compute (TFlops). The plot indicates that if a piece of knowledge has an expected usage count $\in ( 0 . 4 9 4 , 1 3 4 0 0 )$ , then it is optimal to be stored as an explicit memory. Moreover, the introduction of explicit memory helps to externalize the knowledge stored in model parameters and thus allow us to use a lighter backbone, which ultimately reduces all the costs in Figure 4.

The second motivation for explicit memory is to alleviate the issue of knowledge traversal. Knowledge traversal happens when the LLM wastefully invokes all its parameters (and thus all its knowledge) each time it generates a token. As an analogy, it is unreasonable for humans to recall everything they learned whenever they write a word. Let us define the knowledge efficiency of an LLM as the ratio of the minimum amount of knowledge sufficient for one decoding step to the amount of knowledge actually used. An optimistic estimation of knowledge efficiency for a 10B LLM is $1 0 ^ { - 5 }$ : On one hand, it is unlikely that generating one token would require more than $1 0 ^ { 4 }$ bits of knowledge (roughly equivalent to a thousand-token long passage, sufficient for enumerating all necessary knowledge); on the other hand, each parameter is involved in the computation and each stores at least 0.1 bit of knowledge [7, Result 10] (this density could be much higher if the LLM is trained on cleaner data), thus using $1 0 ^ { 9 }$ bits in total.

A novel architecture is needed to boost the knowledge efficiency of LLMs from $1 0 ^ { - 5 }$ to 1, whereas current designs are far from this goal. Consider the mixture-of-experts architecture (MoE) for instance, which uses multiple MLP layers (experts) in each Transformer block and process each token with only a few MLPs. The boost of MoE, namely the ratio of the total amount of parameters to the amount of active parameters, is usually bounded by $4 \sim 3 2$ [40, 56, 98]. Similarly, neither the mixture-of-depth architecture [37, 94] nor sparsified MLP neurons and attention heads [75] can bring greater gains. RAG appears very sparse if we compare the amount of retrieved texts with the size of the text database; nevertheless, RAG is usually built upon a plain LLM as backbone, which provides most of the knowledge used in inference, and thus offers little assistance in addressing the knowledge traversal problem.

An ideal solution is to retrieve only the needed parameters for each token. This is naturally achieved by explicit memories if we compare memory recall to parameter retrieval.

The third motivation is that, as a human-like design, explicit memory enables LLMs to develop more human-like capabilities. To name a few,

■ Infinitely long context: LLMs have the difficulty of processing long texts since their working memory (context key-values) costs too much GPU memory and compute. Meanwhile, despite that humans have very limited working memory capacity [27, 28], they can manage to read and write long texts by converting working memories to explicit memories (thus saving space) and retrieving only the needed explicit memories for inference (thus saving compute). Similarly, by saving explicit memories on drives and doing frequent and constant-size retrieval, LLMs can handle arbitrarily long contexts with time complexity $O ( l \log { l } )$ instead of $\Theta ( l ^ { 2 } )$ , where $ { l }$ is the context length.   
■ Memory consolidation: Instead of writing a piece of knowledge directly into implicit memory, i.e. training model parameters, LLM can first convert it to explicit memory through plain encoding, and then convert this explicit memory to implicit memory through a low-cost step such as compression and finetuning, thus reducing the overall cost.   
■ Factuality and interpretability: Encoding texts as explicit memories is less susceptible to information loss compared to dissolving them in model parameters. With more factual details provided by explicit memories, the LLMs would have less tendency to hallucinate. Meanwhile, the correspondence of explicit memories to readable texts makes the inference more transparent to humans, and also allows the LLM to consciously examine its own thought process.

We demonstrate the improved factuality in the experiments section, and leave the rest to future work.

In this work, we introduce a novel architecture and training scheme for LLM based on explicit memory. The architecture is called Memory3, as explicit memory is the third form of memory in LLM after working memory (context key-values) and implicit memory (model parameters).

Memory3 utilizes explicit memories during inference, alleviating the burden of model parameters to memorize specific knowledge.   
■ The explicit memories are encoded from our knowledge base, and our sparse memory format maintains a realistic storage size.   
■ We trained from scratch a Memory3 model with 2.4B non-embedding parameters, and its performance surpasses SOTA models with greater sizes. It also enjoys better performance and faster inference than RAG.   
Furthermore, Memory3 boosts factuality and alleviates hallucination, and it enables fast adaptation to professional tasks.

This paper is structured as follows: Section 2 lays the theoretical foundation for Memory3, in particular our definitions of knowledge and memory. Section 3 discusses the basic design of Memory3, including its architecture and training scheme. Sections 4, 5, and 6 describes the training of Memory3. Section 7 evaluates the performance of Memory3 on general benchmarks and professional tasks. Finally, Section 8 concludes this paper and discusses future works.

# 1.1 | Related work

# 1.1.1 | Retrieval-augmented Training

Several language models have incorporated text retrieval from the pretraining stage. REALM [49] augments a BERT model with one retrieval step to solve QA tasks. Retro [16] enhances auto-regressive decoding with multiple rounds of retrieval, once per 64 tokens. The retrieved texts are injected through a two-layer encoder and then several cross-attention layers in the decoder. Retro $^ { + + }$ [113] explores the scalability of Retro by reproducing Retro up to 9.5B parameters.

Meanwhile, several models are adapted to retrieval in the finetuning stage. WebGPT [83] learns to use search engine through imitation learning in a text-based web-browsing environment. Toolformer [100] performs decoding with multiple tools including search engine, and the finetuning data is labeled by the LM iself.

The closest model to ours is Retro. Unlike explicit memory, Retro needs to encode the retrieved texts in real-time during inference. To alleviate the cost of encoding these references, it chooses to use a separate, shallow encoder and also retrieve few references. Intuitively, this compromise greatly reduces the amount of knowledge that can be extracted and supplied to inference.

Another line of research utilizes retrieval to aid long-context modeling. Memorizing Transformer [123] extends the context of language models by an approximate kNN lookup into a non-differentiable cache of past key-value pairs. LongLlama [112] enhances the discernability of context key-value pairs by a finetuning process inspired by contrastive learning. LONGMEM [118] designs a decoupled architecture to avoid the memory staleness issue when training the Memorizing Transformer. These methods are not directly applicable to large knowledge bases since the resulting key-value caches will occupy enormous space. Our method overcomes this difficulty through a more intense memory sparsification method.

# 1.1.2 | Sparse Computation

To combat the aforementioned knowledge traversal problem and improve knowledge efficiency, ongoing works seek novel architectures that process each token with a minimum and adaptive subset of model parameters. This adaptive sparsity is also known as contextual sparsity [75]. The Mixture-of-Experts (MoE) use sparse routing to assign Transformer submodules to tokens, scaling model capacity without large increases in training or inference costs. The most common MoE design [40] hosts multiple MLP layers in each Transformer block and routes each token to a few MLPs with the highest allocation score predicted by a linear classifier. Furthermore, variants based on compression such as QMoE [41] are introduced to alleviate the memory burden of MoE. Despite the sparse routing, the boost in parameter efficiency is usually bounded by $4 \sim 3 2$ . For instance, the Arctic model [98], one of the sparsest MoE LLM in recent years, has an active parameter ratio of about $3 . 5 \%$ . Similarly, the Mixture of Depth architecture processes each token with an adaptive subset of the model layers. The implementations can be based on early exit [37] or top- $k$ routing [94], reducing the amount of compute to $1 2 . 5 \sim 5 0 \%$ . More fine-grained approaches can perform sparsification at the level of individual MLP neurons and attention heads. The model Deja Vu [75] trains a low-cost network for each MLP/attention layer that predicts the relevance of each neuron/head at this layer to each token. Then, during inference, Deja Vu keeps the top $5 \sim 1 5 \%$ MLP neurons and $2 0 \sim 5 0 \%$ attention heads for each token.

# 1.1.3 | Parameter as memory

Several works have portrayed model parameters as implicit memory, in accordance with our philosophy. [46] demonstrates that the neurons in the MLP layers of GPTs behave like key-value pairs. Specifically, with the MLP layer written as $\sigma ( X K ^ { T } ) V$ , each row of the first layer weight $K _ { i }$ functions like a key vector, with the corresponding row in the second layer weight $V _ { i }$ being the value vector. [46] observes that for most of the MLP neurons, the $K _ { i }$ is activated by context texts that obey some human interpretable pattern, and the $V _ { i }$ activates the column of the output matrix that corresponds to the most probable next token of the pattern (e.g. $_ n$ -gram). Based on this observation, [108] designs a GPT variant that consists of only attention layers, with performance matching that of the usual GPTs. The MLP layers are incorporated into the attention layers in the form of key-value vector pairs, which are called persistent memories. Similarly, using sensitivity analysis, [29] discovers that factual knowledge learned by BERT is often localized at one or few MLP neurons. These neurons are called “knowledge neurons”, and by manipulating them, [29] manages to update single pieces of knowledge of BERT. Meanwhile, [38] studies an interesting phenomenon known as superposition or polysemanticity, that a neural network can store many unrelated concepts into a single neuron.

# 2 Memory Circuitry Theory

This section introduces our memory circuitry theory, which defines knowledge and memory in the context of LLM. We will see that this theory helps to determine which knowledge can be stored as explicit memory, and what kind of model architecture is suitable for reading and writing explicit memories. For readers interested primarily in the results, it may suffice to review Claim 1 and Remark 1 before proceeding to the subsequent sections. The concepts to be discussed are illustrated in Figure 5.

![](images/c02669f70bc37dccce51d0545a6cd2a6e63d00c53ba1080c96208e55038e48de.jpg)  
Figure 5: Categorization of knowledge and memory formats. The explicit memories, extracted from model activations, lie half-way between raw data and model parameters, so we use a dotted line to indicate that they may or may not be regarded as parameters.

# 2.1 | Preliminaries

The objective is to decompose the computations of a LLM into smaller, recurring parts, and analyze which parts can be separated from the LLM. These small parts will be defined as the “knowledge” of the LLM, and this characterization helps to identify what knowledge can be externalized as explicit memory, enabling both the memory hierarchy and a lightweight backbone.

One behaviorist approach is to define the smaller parts as input-output relations between small subsequences, such that if the input text contains a subsequence belonging to some pattern, then the output text of the LLM contains a subsequence that belongs to some corresponding pattern.

■ One specific input-output relation is that if the immediate context contains “China” and “capital”, then output the token “Beijing”.   
■ One abstract input-output relation is that if the immediate context is some arithmetic expression (e.g. ${ } ^ { * } 1 2 3 \times 4 5 6 = { } ^ { \mathfrak { N } } .$ ) then output the answer (e.g. “56088”).   
■ One abstract relation that will be mentioned frequently is the “search, copy and paste” [85], such that if the context has the form “. . . [a][b]. . . [a]” then output “[b]”, where [a] and [b] are arbitrary tokens.

A decomposition into these relations seems natural since autoregressive LLMs can be seen as upgraded versions of $_ n$ -grams, with the fixed input/output segments generalized to flexible patterns and with the plain lookup table generalized to multi-step computations.

Nevertheless, a behaviorist approach is insufficient since an input-output relation alone cannot uniquely pin down a piece of knowledge: a LLM may answer correctly to arithmetic questions based on either the actual knowledge of arithmetic or memorization (hosting a lookup table for all expressions such as

$\mathbf { ^ { \circ } 1 2 3 \times 4 5 6 } = 5 6 0 8 8 ^ { \circ }$ ). Therefore, we take a white-box approach that includes in the definition the internal computations of the LLM that convert these inputs to the related outputs.

Here are two preliminary examples of internal computations.

Example 1. Several works have studied the underlying mechanisms when LLMs answer to the prompt “The capital of China is” with “Beijing”, as well as other factual questions [29, 46, 79, 22]. At least two mechanisms are involved, and the LLM may use their superposition [79]. One mechanism is to use general-purpose attention heads (called “mover heads”) to move “capital” and “China” to the last token “is”, and then use the MLP layers to map the feature of the last token to “Beijing” [79]. Often, only one or a few MLP neurons are causally relevant, and they are called “knowledge neurons” [29]. This mechanism is illustrated in Figure 6 (left). Another mechanism involves attention heads $h$ whose value-to-output matrices $W _ { V } ^ { h } W _ { O } ^ { h }$ function like bigrams, e.g. mapping “captial” to {“Paris”, “Beijing”, . . . } and “China” to {“panda”, “Beijing”, . . . } , which sum up to produce “Beijing” [22, 46, 79]. This mechanism is illustrated in Figure 6 (middle).

Example 2. The ability of LLMs to perform “search, copy and paste”, namely answering to the context “. . . [a][b]. . . [a]” with “[b]”, is based on two attention heads, together called induction heads [85]. The first head copies the feature of the previous token, enabling [b] to “dress like” its previous token [a]. The second head searches for similar features, enabling the second [a] to attend to [b], which now has the appearance of [a]. Thereby, the last token [a] manages to retrieve the feature of [b] and to output [b]. This mechanism is illustrated in Figure 6 (right). A similar mechanism is found for in-context learning [116].

![](images/f033c1575a14776fe447330608f128b1923430b8743a1a3c9b9f5a609c63653e.jpg)  
Figure 6: Illustration of three subgraphs. Left: A subgraph that inputs “the capital of China is” and outputs “Beijing”. The knowledge neuron is marked in red and the mover heads in green. Middle: Another subgraph with similar function using task-specific heads. Right: The induction-heads subgraph that inputs “[a][b]...[a]” and outputs [b], where [a], [b] are arbitrary tokens. The notations are introduced in Section 2.2. The locations of these attention heads and MLP neurons may be variable.

We will address the internal mechanism for an input-output relation as a circuit, and will define a piece of knowledge as an input-output relation plus its circuit. By manipulating these circuits, one can separate many pieces of knowledge from a LLM while keeping its function intact.

Recent works on circuit discovery demonstrate that some knowledge and skills possessed by Transformer LLMs can be identified with patterns in their computation graphs [85, 116, 106, 45, 115, 24, 29, 46], but there has not been a universally accepted definition of circuit. Different from works on Boolean circuits [50, 80] and circuits with Transformer submodules as their nodes [24, 129], we characterize a circuit as a “spatial-temporal” phenomenon, whose causal structure is localized at the right places (MLP neurons and attention heads) and right times (tokens). Thus, we define a computation graph as a directed acyclic graph, whose nodes are the hidden features of all tokens at all all MLP and attention layers, and whose edges correspond to all activations inside these layers. In particular, the computation graph hosts one copy of the Transformer architecture at each time step. To transcend this phenomenological characterization, we define a circuit as an equivalence class of similar subgraphs across multiple computation graphs.

As a remark, it is conceptually feasible to identify a circuit with the minimal subset of Transformer parameters that causes this circuit. The benefit is that such definition of knowledge seems more intrinsic

to the LLM. Nevertheless, with the current definition, it is easier to perform surgery on the circuits and derive constructive proofs. Besides, it is known that Transformer submodules exhibit superposition or polysemanticity, such that one MLP neuron or attention head may serve multiple distinct functions [38, 79], making the identification of parameter subsets a challenge task.

# 2.2 | Knowledge

We begin with the definition of the knowledge of LLMs. For now, it suffices to adopt heuristic definitions instead of fully rigorous ones. Throughout this section, by LLM we mean autoregressive Transformer LLM that has at least been pretrained. Let $L$ be the number of Transformer blocks and $H$ be the number of attention heads at each attention layer, and the blocks and heads are numbered by $l = 0 , \ldots L - 1$ and $h = 0 , \ldots H - 1$ . There are in total $2 L$ layers (MLP layers and attention layers), and the input features to these layers are numbered by $0 , \ldots 2 L - 1$ .

Definition 1. Given an LLM and a text $\mathbf { t } = ( t _ { 0 } , \ldots t _ { n } )$ , the computation graph $G$ on input $\left( t _ { 0 } , \ldots t _ { n - 1 } \right)$ and target $\left( t _ { 1 } , \ldots t _ { n } \right)$ is a directed graph with weighted edges such that

■ Its nodes consist of the hidden vectors $\mathbf { x } _ { i } ^ { 2 l }$ before all attention layers, the hidden vectors $\mathbf { x } _ { i } ^ { 2 l + 1 }$ before all MLP layers, and the output vectors $\mathbf { x } _ { i } ^ { 2 L }$ , for all blocks $l = 0 , \ldots L - 1$ and positions $i = 0 , \ldots n - 1$ .   
■ Its directed edges consist of each attention edge el,hi,j $e _ { i , j } ^ { l , h }$ that goes from $\mathbf { x } _ { i } ^ { 2 l }$ to $\mathbf { x } _ { j } ^ { 2 l + 1 }$ at the $h$ -th head of the $\it l .$ -th attention layer for all $l , h$ and $i \leq j$ , as well as each MLP edge $e _ { i } ^ { l , m }$ that goes from $\mathbf { x } _ { i } ^ { 2 l + 1 }$ to $\mathbf { x } _ { i } ^ { 2 l + 2 }$ through the $_ m$ -th neuron of the $\it l .$ -th MLP layer for all $l , m , i$ .   
The weight of each attention edge el,hi,j , $e _ { i , j } ^ { l , h }$ which measures the influence of the attention score al,hi,j $a _ { i , j } ^ { l , h }$ on the LLM output, is defined by

$$
\mathcal{L} - \mathcal{L}\big|_{a^{l,h}_{i,j} = 0}\quad \text{or}\quad \frac{\partial\mathcal{L}}{\partial a^{l,h}_{i,j}}
$$

where $\mathcal { L }$ is the log-likelihood of the target $\left( t _ { 1 } , \ldots t _ { n } \right)$ , with $\scriptstyle { \mathcal { L } } | _ { a = 0 }$ obtained by setting $a = 0$ (i.e. causal intervention). Similarly, the weight of each MLP edge $e _ { i } ^ { l , m }$ , which measures the influence of the neuron activation $a _ { i } ^ { l , m }$ ion the LLM output, is defined likewise.

■ Given any subgraph $S \subseteq G$ , define the associated input of $S$ as a subsequence $\mathbf { t } _ { \mathrm { i n } } ( S ) \subseteq ( t _ { 0 } , \dots t _ { n - 1 } )$ such that a token $t _ { i }$ belongs to $\mathbf { t } _ { \mathrm { i n } } ( S )$ if and only if $\left\| \nabla _ { \mathbf { x } _ { i } ^ { 0 } } a \right\|$ is large for some attention edge (or MLP edge) in $S$ with attention score (or activation) $^ { a }$ .   
■ Similarly, define the associated output of the subgraph $S$ as a subsequence $\mathbf { t _ { \mathrm { o u t } } } ( S ) \subseteq ( t _ { 1 } , \dots t _ { n } )$ such that a token $t _ { i }$ belongs to $\mathbf { t _ { \mathrm { o u t } } } ( S )$ if and only if

$$
\left. \mathcal {L} _ {i} - \mathcal {L} _ {i} \right| _ {a = 0} \quad \text {o r} \quad \frac {\partial \mathcal {L} _ {i}}{\partial a}
$$

is large for some attention edge (or MLP edge) in $S$ with attention score (or activation) $^ { a }$ . Here $\mathcal { L } _ { i }$ is the log-likelihood of $t _ { i }$ with respect to the LLM output.

Definition 2. Given two computation graphs $G _ { 1 } , G _ { 2 }$ of an LLM and their subgraphs $S _ { 1 } , S _ { 2 }$ , a mapping $f$ from the nodes of $S _ { 1 }$ to the nodes of $S _ { 2 }$ (not necessarily injective) is a homomorphism if

■ every node at depth $l \in \{ 0 , \ldots 2 L \}$ is mapped to depth $ { l }$   
■ if two nodes are on the same position $_ i$ , then they are mapped onto the same position,   
if two nodes share an edge on attention head $h$ or MLP neuron $_ m$ , then their images also share an edge on head $h$ or neuron $_ m$ .

If such a homomorphism exists, then we say that $S _ { 1 }$ is homomorphic to $S _ { 2 }$

It may be more convenient to define the mapping to be between the input tokens of two sentences, but we adopt the current formulation as it is applicable to more general settings without an obvious correspondence between the tokens and the hidden features at each layer.

![](images/d657a276e5f5daa82098e0bdc858e4209ee62666933da3ed5ffd6528951197be.jpg)

![](images/2639887b9296914752b0c17ed900e0644059c76a8a52a8918c7d1b5610ad006a.jpg)  
Figure 7: Left: Illustration of the computation graph over one Transformer block, showing only three tokens, one attention head and three MLP neurons. The edge weights are not shown. Right: The subgraphs $S _ { 1 } , S _ { 2 }$ , namely the induced subgraphs of the attention edges (black arrows), belong to the circuit of the induction head. The red arrows denote a homomorphism from $S _ { 1 }$ to $S _ { 2 }$ , and the blue arrows denote a homomorphism from $S _ { 2 }$ to $S _ { 1 }$ .

Definition 3. Given an LLM and a distribution of texts, a circuit is an equivalence class $\kappa$ of subgraphs from computation graphs on random texts, such that

■ The computation graph on a random text contains some subgraph $S \in \kappa$ with positive probability.   
■ All subgraphs $S \in \kappa$ are homomorphic to each other.   
■ All edges of all $S \in \kappa$ have non-negligible weights.   
■ The pairs $( \mathbf { t } _ { \mathrm { i n } } ( S ) , \mathbf { t } _ { \mathrm { o u t } } ( S ) )$ share some interpretable meaning across all $S \in \kappa$

Definition 4. Given an LLM and a distribution of texts, we call each circuit a knowledge. Furthermore, a circuit $\kappa$ is called a

specific knowledge, if the associated inputs $t _ { \mathrm { i n } } ( S )$ for all subgraphs $S \in \kappa$ share some interpretable meaning, and the associated outputs $ { t _ { \mathrm { o u t } } } ( S )$ for all $S \in \kappa$ are the same or differ by at most a small fraction of tokens.   
■ abstract knowledge, else.

From now on, we use knowledge as a countable noun since the circuits are countable. Note that the criterion in Definition 4 is stronger than the last criterion in Definition 3, e.g. consider the circuit that always copy-and-pastes the previous token. We will see that the rigidity of specific knowledges makes them easier to externalize.

Here are some well-known examples of knowledge.

Example 3. Recall the knowledge neuron from Example 1 that helps to answer “The capital of China is Beijing”. Such neurons can be activated by a variety of contexts that involve the subject-relation pair (“China”, “capital”) [29]. Its circuit can be simply defined as the equivalence class of subgraphs induced by edges $e _ { i } ^ { l , m }$ , where $( l , m )$ is the fixed location of the knowledge neuron and $_ { i }$ is the variable position of the last token of the context. The associated inputs are “China” and “capital”, and the associated outputs are always “Beijing”. By definition, this circuit is a specific knowledge, since its associated output is fixed and its associated inputs share a clear pattern (fixed tokens with variable positions).

Similarly, by straightforward construction, one can show that each $_ n$ -gram can be expressed as a specific knowledge.

Example 4. Recall the induction heads [85] from Example 2 that complete “[a][b] . . . [a]” with “[b]”. Let $( l , h ) , ( l + 1 , h ^ { \prime } )$ be the locations of these two heads, and denote the variable positions of the two token [a]’s by $i , j$ . Its circuit is the equivalence class of subgraphs induced by the two edges $e _ { i , i + 1 } ^ { l , h } , e _ { i + 1 , j } ^ { l + 1 , h ^ { \prime } }$ , el+1,h′ Although . the associated input-output pairs “[a][b]. . . [a][b]” have a clear pattern, the associated outputs “[b]” alone can be arbitrary, so the induction head is an abstract knowledge.

More sophisticated abstract knowledges have been identified for in-context learning [116] and indirect object identification [115].

Definition 5. Given a LLM and a knowledge $\kappa$ , a text $\mathbf { t } = ( t _ { 0 } , \ldots t _ { n } )$ is called a realization of $\kappa$ , if the computation graph on t has a subgraph that belongs to $\kappa$ .

For instance, any text of the form [a][b]. . . [a][b] can be a realization of the abstract knowledge of induction head.

Our definition of knowledge is extrinsic, depending on a specific LLM, instead of intrinsic, depending only on texts. From this perspective, Problem (1) can be interpreted as relocating the knowledges from an all-encompassing LLM to more efficient models equipped with memory hierarchy. For concreteness, one can fix this reference LLM to be the latest version of ChatGPT or Claude [5, 9], or some infinitely large model from a properly defined limit that has learned from infinite data.

Assumption 1 (Completeness). Fix a reference LLM and a distribution of texts, let $G$ be the computation graph of a random text. Assume that there exists a set K of knowledges such that, with probability 1 over the random text, the subgraph of $G$ induced by edges with non-negligible weights can be expressed as a union of subgraphs $\{ S _ { i } \in \mathbb { K } _ { i } \}$ from $\{ { \mathcal { K } } _ { i } \} \subseteq \mathbb { A }$ .

Essentially, Assumption 1 posits that all computations in the LLM can be fully decomposed into circuits, so that the LLM is nothing more than a collection of specific and abstract knowledges. This viewpoint underscores that the efficiency of LLMs is ultimately about the effective organization of these knowledges, an objective partially addressed by Problem (1).

# 2.3 | Memory

Now the question is what knowledge can be separated from the model parameters and moved to the lower levels of the memory hierarchy.

Definition 6. A knowledge $\kappa$ of the reference LLM is separable if there exists another LLM $M$ such that

■ $M$ does not possess this knowledge, such that for any realization t of $\kappa$ , the model $M$ cannot generate each token of the associated output $\mathbf { t _ { \mathrm { o u t } } }$ with high probability, e.g. $\mathbb { P } _ { M } ( t _ { i } | t _ { 0 } \dots t _ { i - 1 } ) \le 1 / 2$ for some ti ∈ tout. $t _ { i } \in \mathbf { t _ { \mathrm { o u t } } }$   
■ There exists a text $\mathbf { t } _ { * }$ such that for any realization t of $\kappa$ , the model $M$ using $\mathbf { t } _ { * }$ as prefix can generate each token of the associated output $\mathbf { t _ { \mathrm { o u t } } }$ with high probability, e.g. $\mathbb { P } _ { M } ( t _ { i } | \mathbf { t } _ { * } t _ { 0 } \ldots t _ { i - 1 } ) \geq 0 . 9$ for every $t _ { i } \in \mathbf { t _ { \mathrm { o u t } } }$ .

If among the realizations of $\kappa$ , the same associated input $\mathbf { t _ { \mathrm { i n } } }$ can correspond to multiple associated outputs $\mathbf { t _ { \mathrm { o u t } } }$ , then the above probabilities are summed over all branches if position $_ { i }$ is a branching point.

Definition 7. A separable knowledge $\kappa$ of the reference LLM is imitable if any realization $\mathbf { t } ^ { \prime }$ of $\kappa$ can be used as the prefix $\mathbf { t } _ { * }$ in Definition 6, e.g. for any realizations $\mathbf { t } , \mathbf { t } ^ { \prime }$ of $\kappa$ , we have $\mathbb { P } _ { M } ( t _ { i } | \mathbf { t } ^ { \prime } t _ { 0 } \ldots t _ { i - 1 } ) \geq 0 . 9$ for every $t _ { i } \in \mathbf { t _ { \mathrm { o u t } } }$ .

Basically, imitability means that LLMs can achieve the same effect as possessing this knowledge by retrieving example texts that demonstrate this knowledge. Few-shot prompting can be seen as a special case of providing realizations.

Separability is a more general property than imitability. For instance, one can set the prefix $\mathbf { t } _ { * }$ to be an abstract description of $\kappa$ instead of its realization, and this is reminiscent of instruction prompting. Nevertheless, it is not obvious whether the set of separable knowledges is strictly larger than the set of imitable knowledges.

Claim 1. Every specific knowledge $\kappa$ is imitable and thus is separable.

Proof (informal). Without loss of generality, we can assume that for any realization t of $\kappa$ , all tokens of the associated input $\mathbf { t } _ { \mathrm { i n } }$ precede all tokens of the associated output $\mathbf { t _ { \mathrm { o u t } } }$ . Otherwise, we can split $\mathbf { t } _ { \mathrm { i n } }$ into two halves $\mathbf { t } _ { 1 } , \mathbf { t } _ { 2 }$ that precedes/does not precede $\mathbf { t _ { \mathrm { o u t } } }$ , and split the corresponding subgraph $S \in \kappa$ into two halves $S _ { 1 } , S _ { 2 }$ that have high weights with respect to $\mathbf { t } _ { 1 } , \mathbf { t } _ { 2 }$ . Using monotonicity arguments once Definition 3 is fully formalized, one can try to show that this splitting is invariant across $S \in \kappa$ and therefore the sets of $S _ { 1 } , S _ { 2 }$ are two specific knowledges.

Consider sequences of the form [a][b]. . . [a’][b’], where [a], [a’] (or [b], [b’]) could be the associated inputs (or outputs) of any subgraphs $S , S ^ { \prime } \in \mathcal { K }$ . By Definition 4, [a] and [a’] always share some interpretable meaning, while [b] and [b’] are approximately the same sequence. One can construct an abstract knowledge that completes [a][b]. . . [a’] with [b’]: the first part of this circuit detects the common feature of the [a]’s (possibly overlapping with the subgraphs of $\kappa$ ), the second part is an induction head (analogous

to Example 4, it provides [b] with the common feature of the [a]’s and lets $\left[ \mathbf { a } \right]$ to attend to [b]), and the third part generates [b’] based on [b] with possible slight modifications. This circuit is an abstract knowledge since it can be applied to other specific knowledges as long as their associated inputs share the same meaning with the [a]’s, no matter how their associated outputs could vary.

Meanwhile, construct the model $M$ by letting the reference model forget $\kappa$ (e.g. by finetuning on a modified data distribution such that the associated input of $\kappa$ is never followed by the associated output, while the rest of the distribution remains the same). Combining this circuit with $M$ completes the proof. □

Claim 1 indicates that a lot of knowledges can be externalized from the model parameters. The converse of Claim 1 may not hold, since it is imaginable that some abstract knowledges can also be substituted with their realizations.

Remark 1. There are three details in the proof of Claim 1 that will be useful later

1. The circuit we construct has only one attention head that attends to the reference text $\mathbf { t } ^ { \prime }$ from the present text t, while all other computations are confined within either t or $\mathbf { t } ^ { \prime }$ .   
2. Moreover, in this attention head, the circuit only needs the edges from [b] to [a’]. Thus, in general this head only needs to attend to very few tokens in the reference.   
3. It suffices for the reference $\mathbf { t } ^ { \prime }$ to attend only to itself.

These properties will guide our architecture design.

To finish the set-up of Problem (1), we define the memory formats. The definition should subsume the aforementioned formats of model parameters, explicit memories and plain texts for RAG, and also allow for new memory formats of future LLMs.

Definition 8. Let $\mathscr { \kappa }$ be the complete set of knowledges from Assumption 1 and consider the subset of separable knowledges. Let T be a set that contains one or several realizations t for each separable knowledge. Let $f _ { 1 } , \ldots f _ { m }$ be any functions over T. Abstractly speaking, a memory-augmented LLM M is some mapping from prefixes to token distributions with additional inputs

$$
M: \left(\left(t _ {0} \dots t _ {i - 1}\right), \left\{\mathcal {K} _ {1}, \dots \mathcal {K} _ {N} \right\}, X _ {1}, \dots X _ {m}\right) \mapsto \mathbb {P} \left(\cdot \mid t _ {0} \dots t _ {i - 1}\right) \tag {2}
$$

where the set $\{ \mathcal { K } _ { 1 } , . . . \mathcal { K } _ { N } \}$ consists of non-separable knowledges of $M$ that are invoked at this step, and the sets $X _ { j }$ consist of encoded texts

$$
X _ {j} = \left\{f _ {j} \left(\mathbf {t} _ {j, k}\right) \right\} \tag {3}
$$

for some $\mathbf { t } _ { j , k } \in \mathfrak { T }$

Each $j = 1 , \dotsc m$ represents a memory format and $f _ { j }$ is called the write function of this format. If some realization of a separable knowledge $\kappa$ participates in the mapping $M$ , then we say that $\kappa$ is written in format $j$ and read by $M$ .

Analogous to Assumption 1, we are decomposing each step of LLM inference into the invoked circuits, but the decomposition here also involves reference texts that are written in various memory formats.

Table 3 demonstrates that the write functions could be diverse, and the list is probably far from conclusive. Nevertheless, some heuristics still apply. The write function $f _ { j }$ and the read process in $M$ for each format $j$ should be non-trivial such that, for any separable knowledge $\kappa$ not contained in $M$ and any realization t of $\kappa$ , if $\kappa$ enters in $M$ through format $j$ , then $M$ should be able to generate each token of the associated output of $\kappa$ in t with higher probability as in Definition 6. Thus, informally speaking, the total cost of writing and reading $\kappa$ must be bounded from 0, since some minimum computation is necessary for reducing the uncertainty in generating the correct tokens. It follows that the write cost and read cost are complementary, i.e. cheaper writing must be accompanied by more expensive reading.

We define this inverse relationship between the write cost and read cost as the memory hierarchy. This relationship is in accordance with our experience regarding the three examples of human memories in Table 3, e.g. we can utter the common expressions almost immediately while it may take a few seconds to recall a book we read, but the former skill is acquired through years of language speaking. For the LLM memories in Table 3, the inverse relationship is illustrated Figure 4 and established by the calculations in Appendix A.

The imbalanced use of knowledges leads to a heterogeneous distribution of knowledges across the memory hierarchy. To minimize the total cost (1), the separable knowledges that are used more often

![](images/164c3a529c733e66b2f285283966feb77e229e3471c794f838a6cd32af1831e7.jpg)  
Figure 8: Different memory formats with different balances of write cost and read cost. The specific knowledges with high to low usage counts are exemplified by common expressions, expertise and trivia, and are assigned to implicit memory, explicit memory and external information.

should be assigned to memory formats with high write cost and low read cost, whereas the rarely used knowledges should be assigned to formats with low write cost and high read cost. Also, adding a new memory format $m + 1$ is always beneficial as it expands the search space and decreases the minimum cost whenever the usage count of some knowledge $\kappa$ lies in the interval

$$
[ n _ {m + 1} ^ {-}, n _ {m + 1} ^ {+} ] = \left\{n \in [ 0, \infty) \mid \operatorname {a r g m i n} _ {j} \operatorname {c o s t} _ {\mathrm {w r i t e}} (\mathcal {K}, j) + n \cdot \operatorname {c o s t} _ {\mathrm {r e a d}} (\mathcal {K}, j) = m + 1 \right\}
$$

Examples of these intervals are displayed in Figure 4. For concreteness, Figure 8 depicts a reasonable distribution of the specific knowledges for humans, and we expect a similar distribution to hold for LLMs equipped with explicit memory.

# 3 | Design

This section describes the architecture and training scheme of Memory3.

Regarding architecture, the goal is to design an explicit memory mechanism for Transformer LLMs with moderately low write cost and read cost. In addition, we want to limit the modification to the Transformer architecture to be as little as possible, adding no new trainable parameters, so that most of the existing Transformer LLMs can be converted to Memory3 models with little finetuning. Thus, we arrive at a simple design:

■ Write cost: Before inference, the LLM writes each reference to an explicit memory, saved on drives. The memory is selected from the key-value vectors of the self-attention layers, so the write process involves no training. Each reference is processed independently, avoiding the cost of long-context attention.   
■ Read cost: During inference, explicit memories are retrieved from drives and read by self-attention alongside the usual context key-values. Each memory consists of very few key-values from a small amount of attention heads, thus greatly reducing the extra compute, GPU storage, drive storage and loading time. It allows the LLM to retrieve many references frequently with limited influence on decoding speed.

Regarding training, the goal is to reduce the cost of pretraining with a more efficient distribution of knowledge. Based on the discussion in Section 2.3, we want to encourage the LLM to learn only abstract knowledges, with the specific knowledges mostly externalized to the explicit memory bank. Ideally, the pretraining cost should be reduced to be proportional to the small amount of knowledge stored in the model parameters, thereby taking a step closer to the learning efficiency of humans.

# 3.1 | Inference Process

From now on, we refer to the realizations of separable knowledges (Definitions 5 and 6) as references. Our knowledge base (or reference dataset) consists of $1 . 1 \times 1 0 ^ { 8 }$ text chunks with length bounded by 128 tokens. Its composition is described in Section 4.4.

Each reference can be converted to an explicit memory, which is a tensor with shape

(memory layers, 2, key-value heads, sparse tokens, head dimension) = (22, 2, 8, 8, 80)

The 2 stands for the key and value, while the other numbers are introduced later.

Before inference, the Memory3 model converts all references to explicit memories and save them on drives or non-volatile storage devices. Then, at inference time, whenever (the id of) a reference is retrieved, its explicit memory is loaded from drives and sent to GPU to be integrated into the computation of Memory3. By Remark 1, a reference during encoding does not need to attend to any other texts (e.g. other references or query texts), so it is fine to encode each reference independently prior to inference. Such isolation also helps to reduce the compute of attention.

One can also employ a “cold start” approach to bypass preparation time: each reference is converted to explicit memory upon its initial retrieval, rather than prior to inference. Subsequent retrievals will then access this stored memory. The aforementioned inference with precomputed explicit memories will be called “warm start”.

![](images/b6bd88ce68d6d3d0b1680df7ddc404eeba2b49132a0d6a696a59e0eb00296427.jpg)  
Figure 9: The decoding process of Memory3 with memory recall. Each chunk is a fixed-length interval of tokens, which may belong to either the prompt or generated text.

During inference, as illustrated in Figure 9, whenever the LLM generates 64 tokens, it discards the current memories, uses these 64 tokens as query text to retrieve 5 new memories, and continues decoding with these memories. Similarly, when processing the prompt, the LLM retrieves 5 memories for each chunk of 64 tokens. Each chunk attends to its own memories, and the memories could be different across chunks. We leave it to future work to optimize these hyperparameters.

The retrieval is performed with plain vector search with cosine similarity. The references as well as the query chunks are embedded by BGE-M3, a multilingual BERT model [17]. The query and key vectors for retrieval are both obtained from the output feature of the $\langle \mathbf { c l s } \rangle$ token. The vector index is built with FAISS [35].

To further save time, we maintain a fixed-size cache in RAM to store the most recently used explicit memories. It’s been observed that adjacent chunks often retrieve some of the same references. So the cache reduces the cost of loading explicit memories from drives.

Remark 2. It would be ideal to perform retrieval using the hidden features from the LLM itself, since conceptually the LLM should know its needs better than any external module, and such internalized retrieval appears more anthropomorphic. Moreover, retrieving with the hidden features from different layers, different heads and different keywords can help to obtain more diverse results. One simple implementation is to use the sparsified attention queries of the query text to directly search for the explicit memories. Since the explicit memories are the attention key-values, such retrieval can work without the need to finetune the LLM. Specifically, this multi-vector retrieval can follow the routine of [61] with the additional constraint that a query from attention head $h$ can only search for keys from $h$ , while the sparse attention queries can be obtain using the same selection mechanism for explicit memories described later.

Remark 3. One shortcoming of RAG is that the references are usually text chunks instead of whole documents, and thus during inference the references are encoded without their contexts, making them

less comprehensible. This shortcoming can be easily overcome for explicit memories. One solution is to encode each document as one sequence, then chunk the attention key-values into 128-token chunks and sparsify them into explicit memories. This procedure allows the key-values to attend to all their contexts.

# 3.2 | Writing and Reading Memory

Each explicit memory is a subset of the attention key-values from a subset of attention heads when encoding a reference. Thus, during inference, the LLM can directly read the retrieved explicit memories through its self-attention layers by concatenating them with the usual context key-values (Figure 9). Specially, for each attention head $h$ at layer l, if it is chosen as a memory head, then its output $Y ^ { l , h }$ changes from the usual

$$
Y _ {i} ^ {l, h} = \mathrm {s o f t m a x} \Big (\frac {X _ {i} ^ {l , h} W _ {Q} ^ {l , h} \left(X _ {[ : i ]} ^ {l , h} W _ {K} ^ {l , h}\right) ^ {T}}{\sqrt {d _ {h}}} \Big) X _ {[: i ]} ^ {l, h} W _ {V} ^ {l, h} W _ {O} ^ {l, h}
$$

where $X _ { [ : i ] }$ denotes all tokens before or at position $_ i$ and $d _ { h }$ denotes the head dimension, to

$$
Y _ {i} ^ {l, h} = \mathrm {s o f t m a x} \Big (\frac {X _ {i} ^ {l , h} W _ {Q} ^ {l , h} \cdot \mathrm {c o n c a t} \big (K _ {0} ^ {l , h} , \ldots K _ {4} ^ {l , h} , X _ {[ : i ]} ^ {l , h} W _ {K} ^ {l , h} \big) ^ {T}}{\sqrt {d _ {h}}} \Big) \mathrm {c o n c a t} \big (V _ {0} ^ {l, h}, \ldots V _ {4} ^ {l, h}, X _ {[ : i ]} ^ {l, h} W _ {V} ^ {l, h} \big) W _ {O} ^ {l, h} (4)
$$

where each $( K _ { j } , V _ { j } )$ denotes the keys and values of an explicit memory.

While the context BOS token is $\langle \mathbf { s } \rangle$ as usual, when encoding each reference we modify the BOS to “⟨s⟩Reference:” to help the LLM distinguish between encoding normal texts and encoding references. This modified BOS is also prepended to the context during inference, as illustrated in Figure 9, while the context BOS token now serves as a separator between the references and context. Unlike the explicit memories which only appear at a subset of attention heads, this modified BOS is placed at every head at every layer. The motivation is that since the context BOS can attend to the references, its feature is no longer constant, so the LLM needs the modified BOS to serve as the new constant for all attention heads.

Furthermore, we adopt parallel position encoding for all explicit memories, namely the positions of all their keys lie in the same interval of length 128, as depicted in Figure 9. We use the rotary position encoding (RoPE) [107]. The token sparsification is applied after RoPE processes the attention keys, so the selected tokens retain their relative positions in the references. Besides flexibility, one motivation for parallel position is to avoid the “lost in the middle” phenomenon [72], such that if the references are positioned serially, then the ones in the middle are likely to be ignored. Similarly, token sparsification also helps to alleviate this issue by making the attention more focused on the important tokens. We note that designs analogous to the parallel position have been used to improve in-context learning [96] and long-context modeling [15].

# 3.3 | Memory Sparsification and Storage

One of the greatest challenges for explicit memories is that the attention key-values occupy too much space. They not only demand more disk space, which could be costly, but also occupy GPU memory during inference, which could harm the batch size and thus the throughput of LLM generation. An intense compression is needed to save space. The full attention key tensor (or value tensor) for each reference has shape (layers, key-value heads, tokens, head dimension), so we compress all four dimensions.

Regard layers, we only set the first half of the attention layers to be memory layers, i.e. layers that produce and attend to explicit memories (4), while the second half remain as the usual attention layers. Note that Remark 1 suggests that it is usually the attention heads in the middle of the LLM that attend to the references. So it seems that appointing the middle attention layers (e.g. the ones within the $2 5 \%$ to $7 5 \%$ depth range) to be memory layers is a more sensible choice. This heuristic is supported by the observations in [122, 39] that the attention to the distant context usually takes place in the middle layers.

Regarding heads, we set all key-value heads at each memory layer to be memory heads. We reduce their amount by grouped query attention (GQA) [6], letting each key-value head be shared by multiple query heads, and obtain $2 0 \%$ sparsity (8 versus 40 heads). It is worth mentioning that, besides GQA and memory layers, another approach is to select a small subset of heads that are most helpful for reading memories, and this selection does not have to be uniform across layer. We describe several methods for selecting memory heads in Remark 4.

Regarding tokens, we select 8 tokens out of 128 for each key-value head. We choose a high level of sparsity, since Remark 1 indicates that the attention from the context to the references are expected to be

concentrated on very few tokens. Note that the selected tokens are in general different among heads, so in principle their union could cover a lot of tokens. For each head $h$ at layer $ { l }$ , the selection uses top-8 over the attention weight

$$
w _ {j} ^ {l, h} = \sum_ {i = 0} ^ {1 2 7} \tilde {a} _ {i, j} ^ {l, h}, \quad \tilde {a} _ {i, j} ^ {l, h} = \mathrm {s o f t m a x} _ {j} \Big (\frac {X _ {i} ^ {l , h} W _ {Q} ^ {l , h} (X _ {j} ^ {l , h} W _ {K} ^ {l , h}) ^ {T}}{\sqrt {d _ {h}}} \Big)
$$

which measures the importance of a token by the attention received from all tokens. The BOS tokens and paddings do not participate in the the computation of the weights. These attention weights $\tilde { a }$ are different from the usual ones, such that there is no causal mask or position encoding involved. The consideration is that since the explicit memories are prepared before any inference, the selection can only depend on the reference itself instead of any context texts. The removal of causal mask and position encoding ensures that tokens at any position has an equal chance to receive attention from others. To speed up computation, we adopt the following approximate weights in our implementation, although in retrospect this speedup is not necessary.

$$
w _ {j} ^ {l, h} = \sum_ {i = 0} ^ {1 2 7} \exp \left(\frac {X _ {i} ^ {l , h} W _ {Q} ^ {l , h} (X _ {j} ^ {l , h} W _ {K} ^ {l , h}) ^ {T}}{\sqrt {d _ {h}}}\right)
$$

Similar designs that sparsify tokens based on attention weights have been adopted in long-context modeling to save space [74, 131].

Regarding head dimension, we optionally use a vector quantizer to compress each of the key and value vectors using residual quantizations [18] built with FAISS [35]. The compression rate is $8 0 / 7 \approx 1 1 . 4$ During inference, the retrieved memories are first loaded from drives, and then decompressed by the vector quantizer before being sent to GPU. The evaluations in Section 7.1 indicate that this compression has negligible influence on the performance of Memory3. More details can be found in Appendix B.

Hence, the total sparsity is 160 or 1830 (without or with vector compression). Originally, the explicit memory bank would have an enormous size of 7.17PB or equivalently 7340TB (given the model shape described in Section 3.4 and saved in bfloat16). Our compression brings it down to 45.9TB or 4.02TB (without or with vector compression), both acceptable for the drive storage of a GPU cluster.

To deploy the Memory3 model on end-side devices such as smart phones and laptops, one can place the explicit memory bank and the vector index on a cloud server, while the devices only need to store the model parameters and the decoder of the vector quantizer. During inference, to perform retrieval, the model on the end-side device sends the query vector to the cloud server, which then searches the index and returns the compressed memories. The speed test of this deployment is recorded in Section 7.5.

Remark 4. If one wants to finetune a pretrained LLM into a Memory3 model, there are several ways to select a small but effective subset of attention heads (among all heads at all layers) for memory heads (4). Methods such as [122, 39] are proposed to identify the heads that contribute the most to long-context modeling by retrieving useful information from distant tokens, and usually these special heads account for only $< 1 0 \%$ of the total heads. Here we also propose a simple method for selecting memory heads: Given the validation subsets of a representative collection of evaluation tasks, one can measure the average performance $s _ { h }$ for a modified version of the LLM for each attention head $h$ . The modification masks the distant tokens for head $h$ so it can only see the preceding 100 tokens and the BOS token. Then, it is reasonable to expect that $s _ { h }$ would be markedly low for a small subset of heads $h$ , indicating that they are specialized for long-range attention.

Remark 5. Actually, Remark 1 suggests that each reference only needs to be attended to by just one attention head, although in general this special head may be different among the references. Thus, it seems a promising approach to apply adaptive sparsity not only to token selection, but also to the memory heads, namely each reference is routed to one or two heads (analogously to MoE), and its explicit memory is produced and read by these heads. Such design if feasible can further boost the sparsity of explicit memory and save much more space.

# 3.4 | Model Shape

As discussed in Section 2.3, the specific knowledges can be externalized to explicit memories, and thus to minimize the total cost (1), the model parameters (or implicit memory) only need to store abstract knowledges and the subset of specific knowledges that are frequently used. The shape of our model, i.e. (the number of Transformer blocks $L$ , heads $H$ , head dimension $d _ { h }$ , width of the MLP layers W ), is chosen

to accommodate this desired knowledge distribution. Informally speaking, given a fixed parameter size $P$ , the shape maximizes the following objective

$$
\max  _ {L, H, d _ {h}, W} \left\{\frac {\text {c a p a c i t y f o r a b s t r a c t k n o w l e d g e}}{\text {c a p a c i t y f o r s p e c i f i c k n o w l e d g e}} \mid \operatorname {s i z e} (L, H, d _ {h}, W) \approx P \right\} \tag {5}
$$

Here we set $P$ to be 2.4 billion.

Some recent works suggest that the capacities for learning specific knowledges and abstract knowledges are subject to different constraints. On one hand, [29] observes that the amount of bits of trivia information (such as a person’s name, date of birth and job title) that a LLM can store depends only on its parameter size. Regardless of $L$ and $H$ , the max capacity is always around 2 bits per parameter.

On the other hand, [120] trains Transformers to learn simple algorithms such as reversing a list and counting the occurrence of each letter. It is observed that for several such tasks, there exists a minimum $L _ { 0 }$ and $H _ { 0 }$ such that a Transformer with $L \geq L _ { 0 }$ and $H \geq H _ { 0 }$ can learn the task with perfect accuracy, whereas the accuracy drops significantly for Transformers with either $L = L _ { 0 } - 1$ or $H = H _ { 0 } - 1$ (given that either $L _ { 0 }$ or $H _ { 0 } \geq 2$ ). This sharp transition supports the view that the layers and heads of Transformer LLMs can be compared to algorithmic steps, and tasks with a certain level of complexity require at least a certain amount of steps. It is worth mentioning that the emergent phenomenon [119, 105] of LLMs can also be explained by this view and thus adds support to it, although it may not be the only explanation.

By Definition 4, the abstract knowledges are expected to be circuits with greater complexity than specific knowledges, since their associated inputs and outputs exhibit greater variability and thus express more complex patterns. It follows that, in the context of the aforementioned works, the separation of specific and abstract knowledges should be positively correlated with the distinction between trivia information and algorithmic procedures. Hence, it is reasonable to adopt the approximation that the capacity of an LLM for specific knowledges only depends on its parameter size, whereas the capacity for abstract knowledges depends only on $L$ and $H$ .

The informal problem (5) reduces to the maximization of $L$ and $H$ given a fixed parameter size. However, we are left with two ambiguities: first, this formulation does not specify the ratio between $L$ and $H$ , and second the head dimension $d _ { h }$ and MLP width W cannot be too small as the training may become unstable. Regarding the second point, our experiments indicate that pretraining becomes more unstable with increased spikes if $d _ { h } \leq 6 4$ , so we set $d _ { h } = 8 0$ (though it needs to be pointed out that the loss spikes may not be solely attributed to the choice of $d _ { h }$ , and high-quality data for instance may stabilize training and allow us to choose a smaller $d _ { h }$ ). Also, the MLP width W is set to be equal to the hidden dimension $d = H d _ { h }$ . Regarding the first point, controlled experiments (Figure 10) indicate that the loss decreases slightly more rapidly with $L : H \approx 1$ than with other ratios, so we adopt this ratio.

![](images/1a02c98a8663c5d918d7012f3563815ab07240cf1ac8b2f507e57b61438cc238.jpg)  
Figure 10: Comparison of the training losses of models with different shapes, whose parameter sizes range in $2 . 1 \sim 2 . 4 \mathrm { B }$ . The legend l44h40d80 denotes $L = 4 4 , H = 4 0 , d _ { h } = 8 0$ , and the $_ x$ -axis denotes the amount of training samples. Nevertheless, this comparison is not definite, since this is only the warmup stage of our training scheme (Section 3.6) and the ranking may change in the continual train stage when explicit memory is introduced.

In addition, as discussed in Section 3.3, our model uses grouped query attention (GQA), so the number of key-value heads $H _ { k v }$ is set to be 8, which is the usual choice for GQA. The MLP layers are gated

two-layer networks without bias, which are the default choice in recent years [110, 11, 21, 8].

Finally, the model shape is set to be $L = 4 4 , H = 4 0 , H _ { k v } = 8 , d _ { h } = 8 0 , W = 3 2 0 0$ , with the total non-embedding parameter size being 2.4B.

# 3.5 | Training Designs

Similar to our architecture design, the design of our training scheme focuses on learning abstract knowledges. The goal is to reduce the training compute, as the LLM no longer needs to memorize many of the specific knowledges. This shift in learning objective implies that all the default settings for pretraining LLMs may need to be redesigned, as they were optimized for the classical scenario when the LLMs learn both abstract and specific knowledges.

1. Data: Ideally, the pretraining data should have a high concentration of abstract knowledges and minimum amount of specific knowledges. It is known that LLM pretraining is very sensitive to the presence of specific knowledges. For instance, [55] observes that a small model can master arithmetic (e.g. addition of large numbers) if trained on clean data. However, if the training data is mixed with trivial information (e.g. random numbers), then the test accuracy stays at zero unless the model size is increased by a factor of 1500. It suggests that training on specific knowledges significantly inhibits the learning of abstract knowledges, and may explain why emergent abilities [119] are absent from small models. Notably, the Phi-3 model [4] is pretrained with a data composition that closely matches our desired composition. Although the technical details are not revealed, it is stated that they filter data based on two criteria: the data should encourage reasoning, and should not contain information that is too specific.   
2. Initialization: [132] observes that initializing Transformer parameters with a smaller standard deviation $d ^ { c }$ with $c < - 1 / 2$ instead of the usual $\Theta ( d ^ { - 1 / 2 } )$ [47, 52]) can encourage the model to learn compositional inference instead of memorization. Specially, an arithmetic dataset is designed with a train set and an out-of-distribution test set, which admits two possible answers. One answer relies on memorizing more rules during training, while the other requires an understanding of the compositional structure underlying these rules. The proposed mechanism is that training with smaller initialization belongs to the condensed regime that encourages sparse solutions, contrary to training with large initialization that belongs to the kernel regime or critical regime [78, 19].   
3. Weight decay: [90, 88] observe that using a larger weight decay coefficient (i.e. greater than the usual range of $0 . 0 0 1 \sim 0 . 1 $ ) can guide LLMs to favor generalization over memorization, and accelerate the learning of generalizable solutions. They consider settings that exhibit grokking [90] such that training would transit from perfect train accuracy and zero test accuracy to perfect test accuracy, and generalization ability is measured by how quickly this transition occurs. Moreover, theoretically speaking, it is expected that training generative models needs stronger regularization than training regression models, in order to prevent the generated distributions from collapsing onto the training data and become trivial [128].

In summary, it is recommendable to pretrain the Memory3 model with a data composition that emphasizes abstract knowledges and minimizes specific information, a smaller initialization for parameters, and a larger weight decay coefficient.

Since this work is only a preliminary version of Memory3, we decide to stick with the conventional setting for training and have not experimented with any of these ideas. We look forward to incorporating these designs in future versions of the Memory3 model.

# 3.6 | Two-stage Pretrain

The Memory3 model learns to write and read explicit memories during pretraining. The training data is prepended with retrieved references; the model encodes these references into explicit memories in real time, and integrates them into the self-attention computation of the training data.

Unexpectedly, our pretraining consists of two stages, which we name as warmup and continual train. Only the continual train stage involves explicit memories, while the warmup stage uses the same format as ordinary pretraining. Our motivation is depicted in Figure 11. We observe that pretraining with explicit memories from the beginning would render the memories useless, as there appears to be no gain in training loss compared to ordinary pretraining. Meanwhile, given a checkpoint from ordinary pretraining, continual training with explicit memory exhibits a visible decrease in training loss. This comparison

implies that a memory-less warmup stage might be necessary for pretraining a Memory3 model. One possible explanation for this phenomenon is that in the beginning of pretraining, the model is too weak to understand and leverage the explicit memories it generates. Then, to reduce distraction, the self-attention layers might learn to always ignore these memories, thus hindering indefinitely the development of explicit memory.

![](images/776b77cef5969d3c61dcc1eb15b7cea757450e3d40699bcfd2876e7c3a451607.jpg)

![](images/22cdcdc24bfb409c6e96829250157724c3d3c4874f809e5c032b2c43487bb1ea.jpg)  
Figure 11: Left: Comparison of the warmup stage (training from scratch) with and without explicit memory. The blue and green curves are trained without and with explicit memories, respectively. Right: Comparison of the continual train stage. The blue and green curves are continual trained from their warmup checkpoints, and the red curve is initialized with the warmup checkpoint of the blue curve and continual trained with explicit memory. These plots indicate that pretraining a Memory3 model requires a memory-less warmup stage. These experiments use a smaller model with 0.92B non-embedding parameters $( L = 4 0 , H = 3 2 , d _ { h } = 6 4 )$ $L = 4 0$ ). The warmup stage uses 60B data and the continual train stage uses 22B.

Another modification is to reduce the cost of continual train. Recall from Section 3.1 that during inference, each 64-token chunk attends to five explicit memories, or equivalently five 128-token references if using cold start, increasing the amount of input tokens by 10 times. The inference process avoids the cost of memory encoding by precomputation or warm start, but for the continual train, the references need to be encoded in real time. Our solution is to let the chunks share their references during training to reduce the total number of references in a batch. Specifically, each chunk of a training sequence retrieves only one reference, and in compensation, attends to the references of the previous four chunks, besides its own reference. Each train sequence has length 2048 and thus 32 chunks, so it is equipped with $3 2 \times 1 2 8 = 4 0 9 6$ reference tokens. The hidden features of these reference tokens are discarded once passing the last memory layer, since after that they no longer participate in the update of the hidden feature of the train tokens. Hence, each continual train step takes slightly more than twice the amount of time of a warmup step.

It is necessary to avoid information leakage when equipping the training data with references (i.e. the train sequence and its retrieved references could be the same text), for otherwise training becomes too easy and the model would not learn much. Previously, Retro [16] requires that no train sequence can retrieve a reference from the same document, but this criterion may be insufficient since near-identical paragraphs may appear in multiple documents. Thus, we require that no train sequence can be accompanied by a reference sequence that has $> 9 0 \%$ overlap with it. The overlap is measured by the length of their longest common subsequence divided by the length of the reference length. Specially, given any train sequence t and reference r, define their overlap by

$$
\operatorname {o v e r l a p} (\mathbf {t}, \mathbf {r}) := \frac {1}{| \mathbf {r} |} \max  \left\{N \mid \exists 1 \leq i _ {1} <   \dots <   i _ {N} \leq | \mathbf {t} | \text {a n d} \exists 1 \leq j _ {1} <   \dots <   j _ {N} \leq | \mathbf {r} | \right. \tag {6}
$$

$$
\left. \text {a n d} | i _ {N} - i _ {1} | \leq 2 | \mathbf {r} |, \text {s u c h t h a t} \mathbf {t} _ {i _ {k}} = \mathbf {r} _ {j _ {k}} \text {f o r} k = 1, \dots N \right\}
$$

The constraint $| i _ { N } - i _ { 1 } | \leq 2 | \mathbf { r } |$ ensures that the overlap is not over-estimated as $| \mathbf { t } | \to \infty$ .

# 4 Pretraining Data

This section describes the procedures for collecting and filtering our pretraining dataset and knowledge base (or reference dataset).

![](images/8d9c51262a94609ca373e8cc764033aa5e6c64a1350f1c5335b7823282f92d25.jpg)  
Figure 12: Composition of our pretraining dataset.

# 4.1 | Data Collection

The pretrain data is gathered from English and Chinese text datasets, mostly publicly available collections of webpages and books. We also include code, SFT data (supervised finetuning), and synthetic data.

Specially, the English data mainly consists of RedPajamaV2 [23], SlimPajama [104] and the Piles [43], in total 200TB prior to filtering. The Chinese data mainly comes from Wanjuan [51], Wenshu [2], and MNBVC [81], in total 500TB prior to filtering. The code data mainly comes from Github, and we take the subset with the highest repository stars. The SFT data is included since these samples generally have higher quality than the webpages. We use the same data as in SFT training (Section 6.1), except that these samples are treated as ordinary texts during pretraining, i.e. all tokens participate in the loss computation, not just the answer tokens.

# 4.2 | Filtering

The raw data is filtered with three steps: deduplication, rule-based filtering, and model-based filtering.

First, deduplication is performed with MinHash for most of the datasets. One exception is RedPajamaV2, which already comes with deduplication labels.

Second, we devise heuristic, rule-based filters analogous to the ones from [76, 92, 25]. The purpose is to eliminate texts that are ostensibly unsuitable for training, such as ones that only contain webpage source codes, random numbers, or incomprehensible shards. Our filters remove documents with less than 50 words, documents whose mean word lengths exceed 10 characters, documents with $7 0 \%$ of context being non-alphabetic characters, documents whose fractions of unique words are disproportionately high, documents whose entropy of unigrams is excessively low, and so on.

Finally, we select the subset of data with highest “quality”, a score produced by a finetuned BERT model. Specially, we sample ten thousand documents and grade them by the XinYu-70B model [65, 68] with prompt-guided generation. The prompt asks the model to determine whether the input text is informative and produce a score between 0 and 5. Then, these scores are used to finetune the Tiny-BERT model [57], which has only 14M parameters. The hyperparameters of this finetuning are optimized with respect to a held-out validation set. After that, we use this lightweight BERT to grade the entire dataset.

Remark 6. Recall from Section 3.5 that the pretraining data of Memory3 should emphasize abstract knowledges and minimize specific knowledges. The purpose is to not only obtain a lightweight LLM with an ideal distribution of knowledges in accordance with the memory hierarchy (Figure 8), but also prevent the specific knowledges from hindering the learning process of the model. The focus of our prompt on “informativeness” might be contradictory to this goal, since the selected texts that are rich in information content may contain too many specific knowledges. For future versions of Memory3, we will switch to a model-based filter favoring texts that exhibit more reasoning and less specifics.

The filtered dataset consists of around four trillion tokens, and its composition is illustrated in Figure 12.

# 4.3 | Tokenizer

Similar to our dataset, our tokenizer mainly consists of Chinese and English tokens. The English vocabulary comes from the 32000 tokens of the LLaMA2 tokenizer. We include roughly the same amount of Chinese tokens produced from byte-pair encoding (BPE). The BPE is trained on a 20GB Chinese corpus that consists of Chinese news and e-books. After deduplication, the final vocabulary has 60299 tokens.

# 4.4 | Knowledge Base

The knowledge base (or reference dataset) is used during training and inference as the source of explicit memories, as depicted in Figure 1. It consists of reference texts that are split into token sequences with length $\leq 1 2 8$ , as described in Section 3.1.

Heuristically, a larger knowledge base is always better, as long as it does not contain misinformation, so it is not surprising that the reference dataset of Retro contains its entire pretrain dataset [16]. Nevertheless, the storage of explicit memories is more costly than plain texts despite our sparsification (Section 3.3), and thus to save storage space, we select a small subset of our pretrain dataset as the knowledge base.

With a focus on high quality data, we include for references the English Wikipedia, WikiHow, the Chinese baike dataset, the subset of English and Chinese books whose titles appear academic, Chinese news, synthetic data and high quality codes. These texts are tokenized and split into chunks of 128 tokens, resulting in $1 . 1 \times 1 0 ^ { 8 }$ references in total.

One may be curious whether our knowledge base may contain some of the evaluation questions, rendering our evaluation results (Section 7.1) less credible. To prevent such leakage, we include in our evaluation code a filtering step, such that for each evaluation question, if a retrieved reference has an overlap with the question that exceeds a threshold, then it is discarded. This deduplication is analogous to the one used when preparing for continual train (Section 3.6), with the overlap measured by (6). The threshold 2/3 is chosen since we observe that typically a reference that contains a question would have an overlap $\geq 8 0 \%$ , while a relevant but distinct reference would have an overlap $\leq 4 0 \%$ .

Remark 7. Currently, the compilation of the knowledge base is based on human preference. For future versions of Memory3, we plan to take a model-oriented approach and measure the fitness of a candidate reference by its actual utility, e.g. the expected decrease in the validation loss of the LLM conditioned on this reference being retrieved by a random validation sample.

# 5 | Pretrain

This section describes the details of the pretraining process. The two-stage pretrain and memory-augmented data follow the designs introduced in Section 3.6. As an interpretation, the Memory3 model during the warmup stage develops its reading comprehension, which is necessary during the continual train stage for initiating memory formation.

# 5.1 | Set-up

Training is conducted with the Megatron-DeepSpeed package [3] and uses mixed-precision training with bfloat16 model parameters, bfloat16 activations, and float32 AdamW states. The batch size is around 4 million training tokens with sequence length 2048, not including the reference tokens. The weight decay is the common choice of 0.1.

We adopt the “warmup-stable-decay” learning rate schedule of MiniCPM [54], which is reportedly better than the usual cosine schedule in term of training loss reduction. The learning rate linearly increases to the maximum value, then stays there for the majority of training steps, and finally in the last $1 0 \%$ steps decays rapidly to near zero. Our short-term experiments confirm the better performance of this schedule. Nevertheless, frequent loss spikes and loss divergences are encountered during the official pretraining, so we have to deviate from this schedule and manually decrease the learning rate to stabilize training.

Originally, it is planned that both the warmup and continual train stages go through the entire 4T token pretrain dataset (Section 4). Due to the irremediable loss divergences, both stages have to be terminated earlier.

![](images/2cd6a6d828e9e77b9df008ac1c85199f8fc9e37456da036ca634b7708d010bbf.jpg)

![](images/a7dd7acb35aabbab1a5ea08e79a5e421838fea21030e3f52958d1aa2aaaee636.jpg)  
Figure 13: The warmup stage without explicit memory. Left: Training loss. Right: Learning rate schedule.

# 5.2 | Warmup Stage

The training loss and learning rate schedule are plotted in Figure 13. Whenever severe loss divergence occurs, we restart from the last checkpoint before the divergence with a smaller learning rate, and thus the divergences are not shown in the figure. Eventually, the training terminates at around 3.1T tokens, when reducing the learning rate can no longer avoid loss divergence.

# 5.3 | Continual Train Stage

![](images/75953e25b007e35c70d75cbedf09f49ca76935b5cc0666a5384fd1c502a81724.jpg)

![](images/9b1f8a1aa59d2212ac3c561b745657302d1e7f2941f183b4152eb94deb829169.jpg)  
Figure 14: The continual train stage with explicit memory. Left: Training loss. Right: Learning rate schedule.

The explicit memories enter into the Memory3 model at this stage. The training steps are slower since the model needs to encode the references retrieved for the pretrain data to explicit memories in real time, and each step takes a bit more than twice the time of a warmup step. The training loss and learning rate schedule are plotted in Figure 14.

The loss divergence soon becomes irremediable at around 120B training tokens, much shorter than the planned 4T tokens, and training has to stop there. One possible cause is that the continual train is initialized from the latest warmup checkpoint, which is located immediately before the break down of the warmup stage, and thus is already at the brink of divergence. The smaller learning rate of continual train delays the onset of divergence but not for long.

# 6 Fine-tuning and Alignment

This section describes our model finetuning, specifically supervised finetuning (SFT) and direct preference optimization (DPO).

# 6.1 | Supervised Finetuning

Analogous to the StableLM model [14], our Memory3 model is finetuned on a diverse collection of SFT datasets. We use the following datasets, which are publicly accessible on the Hugging Face Hub: UltraChat [34], WizardLM [125], SlimOrca [67], ShareGPT [114], Capybara [31], Deita [73], and MetaMathQA

[130]. We also include synthetic data with emphasis on multi-round chat, mathematics, commonsense and knowledge. Each training sample consists of one or more rounds of question and answer pairs. We remove any sample with more than eight rounds. The final composition is listed in Table 15.

Table 15: Composition of SFT dataset.   

<table><tr><td>Dataset</td><td>Source</td><td>Number of Samples</td></tr><tr><td>UltraChat</td><td>HuggingFaceH4/ultrachat_200k</td><td>194409</td></tr><tr><td>WizardLM</td><td>WizardLM/WizardLM_evol_instruct_V2_196k</td><td>80662</td></tr><tr><td>SlimOrca</td><td>Open-Orca/SlimOrca-Dedup</td><td>143789</td></tr><tr><td>ShareGPT</td><td>openchat/openchat_sharegpt4_dataset</td><td>3509</td></tr><tr><td>Capybara</td><td>LDJnr/Capybara</td><td>7291</td></tr><tr><td>Deita</td><td>hkust-nlp/deita-10k-v0</td><td>2860</td></tr><tr><td>MetaMathQA</td><td>meta-math/MetaMathQA</td><td>394418</td></tr><tr><td>Multi-round Chat</td><td>synthetic</td><td>20000</td></tr><tr><td>Mathematics</td><td>synthetic</td><td>20000</td></tr><tr><td>Commonsense</td><td>synthetic</td><td>150000</td></tr><tr><td>Knowledge</td><td>synthetic</td><td>270000</td></tr></table>

The training process uses the cosine learning rate schedule with a max learning rate of $5 \times 1 0 ^ { - 5 }$ and a $1 0 \%$ linear warmup phase. The weight decay is 0.1, batch size is 512, and max sequence length is 2048 tokens. Finetuning is performed for 3 epochs.

# 6.2 | Direct Preference Optimization

The Memory3 model is further finetuned by DPO [93], to align with human preference and improve its conversation skills. The DPO dataset consists of general conversations (UltraFeedback Binarized [111]), math questions (Distilabel Math [10]) and codes questions (Synth Code [36]). The training uses the cosine learning rate schedule with max lr $4 \times 1 0 ^ { - 6 }$ . The inverse temperature $\beta$ of the DPO loss is set to 0.01. The improvement from DPO is displayed in Section 7.2.

# 7 | Evaluation

We evaluate the general abilities (benchmark tasks), conversation skills, professional abilities (law and medicine), and facutality & hallucination of the Memory3 model. We also measure its decoding speed. Our model is compared with SOTA LLMs of similar and larger sizes, as well as RAG models.

# 7.1 | General Abilities

To evaluate the general abilities of Memory3, we adopt all tasks from the Huggingface leaderboard and also include two Chinese tasks. Most of the results are displayed in Table 16, while TruthfulQA is listed in Table 19. All results are obtained in bfloat16 format, using the lm-evaluation-harness package [44] and the configuration of HuggingFace Open LLM leaderboard [13], i.e. the number of few-shot examples and grading methods.

As described in Section 4.4, to prevent cheating, a filtering step is included in the retrieval process so that the model cannot copy from references that resemble the evaluation questions.

The results of our model without using explicit memory is included, which indicates that explicit memory boosts the average score by $2 . 5 1 \%$ . In comparison, the score difference between Llama2-7B and 13B is $4 . 9 1 \%$ while the latter has twice the amount of non-embedding parameters. Thus, it reasonable to say that explicit memory can increase the “effective model size” by $2 . 5 1 / 4 . 9 1 \approx 5 1 . 1 \%$ . (Also, the score difference between Qwen-1.8B and 4B is $8 . 4 8 \%$ while the latter has $1 6 7 \%$ more non-embedding parameters. With respect to this scale, explicit memory increases the “effective model size” by 1.2.51 $7 8 . 4 8 \times 1 . 6 7 \approx 4 9 . 4 \% .$

We also include the results of Memory3 with vector compression (Section 3.3). Even though the key-value vectors of the explicit memories are compressed to $8 . 7 5 \%$ of their original sizes, the performance of our model does not show any degradation.

Other supplementary evaluations can be found in Appendix C.

Next, we compare with a LLM that is pretrained with text retrieval. Specially, we consider the largest version of the Retro $^ { + + }$ model [113], Retro $^ { + + }$ XXL with 9.5B parameters. All tasks from Table 6 of [113]

Table 16: Few-shot evaluation of general abilities. The model sizes only include non-embedding parameters.   

<table><tr><td rowspan="2">LLM</td><td rowspan="2">Size</td><td rowspan="2">Avg.</td><td colspan="5">English</td><td colspan="2">Chinese</td></tr><tr><td>ARC-C</td><td>HellaSwag</td><td>MMLU</td><td>Winogrand</td><td>GSM8k</td><td>CEVAL</td><td>CMMLU</td></tr><tr><td>Falcon-40B</td><td>41B</td><td>55.75</td><td>61.86</td><td>85.28</td><td>56.89</td><td>81.29</td><td>21.46</td><td>41.38</td><td>42.07</td></tr><tr><td>Llama2-7B-Chat</td><td>6.5B</td><td>46.87</td><td>52.90</td><td>78.55</td><td>48.32</td><td>71.74</td><td>7.35</td><td>34.84</td><td>34.40</td></tr><tr><td>Llama2-13B-Chat</td><td>13B</td><td>51.78</td><td>59.04</td><td>81.94</td><td>54.64</td><td>74.51</td><td>15.24</td><td>38.63</td><td>38.43</td></tr><tr><td>Llama3-8B-it</td><td>7.0B</td><td>65.77</td><td>62.03</td><td>78.89</td><td>65.69</td><td>75.77</td><td>75.82</td><td>50.52</td><td>51.70</td></tr><tr><td>Vicuna-13B-v1.5</td><td>13B</td><td>52.02</td><td>57.08</td><td>81.24</td><td>56.67</td><td>74.66</td><td>11.30</td><td>41.68</td><td>41.53</td></tr><tr><td>Mistral-7B-v0.1</td><td>7.0B</td><td>59.15</td><td>59.98</td><td>83.31</td><td>64.16</td><td>78.37</td><td>37.83</td><td>45.91</td><td>44.49</td></tr><tr><td>Gemma-2B-it</td><td>2.0B</td><td>36.64</td><td>38.02</td><td>40.36</td><td>55.74</td><td>35.29</td><td>55.88</td><td>8.26</td><td>29.94</td></tr><tr><td>Gemma-7B-it</td><td>7.8B</td><td>47.23</td><td>51.45</td><td>71.96</td><td>53.52</td><td>67.96</td><td>32.22</td><td>27.93</td><td>25.70</td></tr><tr><td>MiniCPM-2B-SFT</td><td>2.4B</td><td>54.37</td><td>47.53</td><td>71.95</td><td>51.32</td><td>67.72</td><td>45.26</td><td>48.07</td><td>48.76</td></tr><tr><td>Phi-2</td><td>2.5B</td><td>55.70</td><td>61.09</td><td>75.11</td><td>58.11</td><td>74.35</td><td>54.81</td><td>34.40</td><td>32.04</td></tr><tr><td>ChatGLM3-6B</td><td>5.7B</td><td>54.62</td><td>41.38</td><td>66.98</td><td>50.54</td><td>64.25</td><td>51.25</td><td>54.01</td><td>53.91</td></tr><tr><td>Baichuan2-7B-Chat</td><td>6.5B</td><td>55.16</td><td>52.73</td><td>74.06</td><td>52.77</td><td>69.77</td><td>28.28</td><td>53.12</td><td>55.38</td></tr><tr><td>Qwen1.5-1.8B-Chat</td><td>1.2B</td><td>49.67</td><td>38.74</td><td>60.02</td><td>45.87</td><td>59.67</td><td>33.59</td><td>55.57</td><td>54.22</td></tr><tr><td>Qwen1.5-4B-Chat</td><td>3.2B</td><td>58.15</td><td>43.26</td><td>69.73</td><td>55.55</td><td>64.96</td><td>52.24</td><td>61.89</td><td>59.39</td></tr><tr><td>Qwen1.5-7B-Chat</td><td>6.5B</td><td>64.80</td><td>56.48</td><td>79.02</td><td>60.52</td><td>66.38</td><td>54.36</td><td>68.20</td><td>68.67</td></tr><tr><td>Memory3-SFT</td><td>2.4B</td><td>63.31</td><td>58.11</td><td>80.51</td><td>59.68</td><td>74.51</td><td>52.84</td><td>59.29</td><td>58.24</td></tr><tr><td>with vector compression</td><td>2.4B</td><td>63.33</td><td>57.94</td><td>80.65</td><td>59.66</td><td>75.14</td><td>52.24</td><td>59.66</td><td>58.05</td></tr><tr><td>without memory</td><td>2.4B</td><td>60.80</td><td>57.42</td><td>73.14</td><td>57.29</td><td>74.35</td><td>51.33</td><td>56.32</td><td>55.72</td></tr></table>

are taken, except for HANS, which is not available on lm-eval-harness, and all tasks are zero-shot. Similar to Table 16, Memory3 is tested with a filtering threshold of 2/3. The results are listed in Table 17, where Memory3 outperforms the model with much larger parameter size and reference dataset size.

Table 17: Zero-shot comparison of LLMs pretrained with retrieval. The scores of Retro++ are taken from [113]. The size of a reference dataset is its number of tokens. The non-embedding parameter size of Retro++ is inferred from its vocabulary size.   

<table><tr><td>LLM</td><td>Param size</td><td>Avg.</td><td>HellaSwag</td><td>BoolQ</td><td>Lambada</td><td>RACE</td></tr><tr><td>Retro++ XXL</td><td>9.1B</td><td>61.0</td><td>70.6</td><td>70.7</td><td>72.7</td><td>43.2</td></tr><tr><td>Memory3-SFT</td><td>2.4B</td><td>64.7</td><td>83.3</td><td>80.4</td><td>57.9</td><td>45.3</td></tr><tr><td colspan="2">Reference size</td><td></td><td>PiQA</td><td>Winogrand</td><td>ANLI-R2</td><td>WiC</td></tr><tr><td colspan="2">330B</td><td></td><td>77.4</td><td>65.8</td><td>35.5</td><td>52.4</td></tr><tr><td colspan="2">14.3B</td><td></td><td>76.6</td><td>75.8</td><td>41.6</td><td>56.9</td></tr></table>

# 7.2 | Conversation Skill

Next we evaluate the conversation skills of Memory3. We use MT-Bench (the Multi-turn Benchmark) [133] that consists of multi-round and open-ended questions. The results are listed in Table 18, including the Memory3 model finetuned by DPO introduced in Section 6.2.

We obtain all these scores using GPT-4-0613 as grader, following the single answer grading mode of MT-Bench. Our model outperforms Vicuna-7B, Falcon-40B-Instruct, and ChatGLM2-6B with fewer parameters.

# 7.3 | Hallucination and Factuality

Despite considerable progress, LLMs still face issues with hallucination, leading to outputs that often stray from factual accuracy [97]. Conceptually, Memory3 should be less vulnerable to hallucination, since its explicit memories directly correspond to reference texts, whereas compressing texts into the model parameters might incur information loss. To evaluate hallucination, we select two English datasets, TruthfulQA [70] and HaluEval, and one Chinese dataset [64], HalluQA [20]. TruthfulQA is implemented with lm-evaluation-harness [44], while HaluEval and HalluQA are implemented with UHGEval [69]. The results are shown in Table 19, with Memory3 achieving the highest scores on most tasks.

Table 18: MT-Bench scores. The model sizes only include non-embedding parameters.   

<table><tr><td>LLM</td><td>Size</td><td>MT-Bench Score</td></tr><tr><td>Phi-3</td><td>3.6B</td><td>8.38</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>7.0B</td><td>7.60</td></tr><tr><td>Qwen1.5-7B-Chat</td><td>6.5B</td><td>7.60</td></tr><tr><td>Zephyr-7B-beta</td><td>7.0B</td><td>7.34</td></tr><tr><td>MiniCPM-2B-DPO</td><td>2.4B</td><td>6.89</td></tr><tr><td>Llama-2-70B-Chat</td><td>68B</td><td>6.86</td></tr><tr><td>Mistral-7B-Instruct-v0.1</td><td>7.0B</td><td>6.84</td></tr><tr><td>Llama-2-13B-Chat</td><td>13B</td><td>6.65</td></tr><tr><td>Llama-2-7B-Chat</td><td>6.5B</td><td>6.57</td></tr><tr><td>MPT-30B-Chat</td><td>30B</td><td>6.39</td></tr><tr><td>ChatGLM2-6B</td><td>6.1B</td><td>4.96</td></tr><tr><td>Falcon-40B-Instruct</td><td>41B</td><td>4.07</td></tr><tr><td>Vicuna-7B</td><td>6.5B</td><td>3.26</td></tr><tr><td>Memory3-SFT</td><td>2.4B</td><td>5.31</td></tr><tr><td>Memory3-DPO</td><td>2.4B</td><td>5.80</td></tr></table>

Table 19: Evaluation of hallucination. HaluE and TruQA denote HaluEval and TruthfulQA, respectively. Bolded numbers are the best results. The model sizes only include non-embedding parameters. Vicuna-13B-v1.5 gets one N/A since that entry is near zero and seems abnormal.   

<table><tr><td rowspan="2">LLM</td><td rowspan="2">Size</td><td rowspan="2">Avg.</td><td colspan="4">English</td><td>Chinese</td></tr><tr><td>HaluE-QA</td><td>HaluE-Dialogue</td><td>TruQA-MC1</td><td>TruQA-MC2</td><td>HalluQA</td></tr><tr><td>Falcon-40B</td><td>41B</td><td>35.37</td><td>46.84</td><td>40.80</td><td>27.29</td><td>41.71</td><td>20.18</td></tr><tr><td>Llama2-13B</td><td>13B</td><td>28.01</td><td>23.34</td><td>31.05</td><td>25.95</td><td>36.89</td><td>22.81</td></tr><tr><td>Vicuna-13B-v1.5</td><td>13B</td><td>37.07</td><td>24.93</td><td>37.35</td><td>35.13</td><td>50.88</td><td>N/A</td></tr><tr><td>Baichuan2-13B</td><td>13B</td><td>37.64</td><td>46.02</td><td>45.45</td><td>26.81</td><td>39.79</td><td>30.12</td></tr><tr><td>Gemma-7B</td><td>7.8B</td><td>37.03</td><td>50.91</td><td>48.19</td><td>20.69</td><td>46.65</td><td>18.71</td></tr><tr><td>Mistral-7B-v0.1</td><td>7.0B</td><td>34.18</td><td>40.68</td><td>37.64</td><td>28.03</td><td>42.60</td><td>21.93</td></tr><tr><td>Llama2-7B</td><td>6.5B</td><td>36.80</td><td>52.46</td><td>51.93</td><td>25.09</td><td>38.94</td><td>15.59</td></tr><tr><td>Baichuan2-7B</td><td>6.5B</td><td>38.63</td><td>62.33</td><td>47.84</td><td>23.01</td><td>37.46</td><td>22.51</td></tr><tr><td>ChatGLM3-6B</td><td>5.7B</td><td>40.96</td><td>43.38</td><td>50.03</td><td>33.17</td><td>49.87</td><td>28.36</td></tr><tr><td>Qwen1.5-4B-Chat</td><td>3.2B</td><td>33.30</td><td>24.64</td><td>37.72</td><td>29.38</td><td>44.74</td><td>30.00</td></tr><tr><td>Phi-2</td><td>2.5B</td><td>38.31</td><td>50.71</td><td>39.55</td><td>31.09</td><td>44.32</td><td>25.89</td></tr><tr><td>MiniCPM-SFT</td><td>2.4B</td><td>36.47</td><td>49.24</td><td>47.80</td><td>24.11</td><td>37.51</td><td>23.71</td></tr><tr><td>Gemma-2B</td><td>2.0B</td><td>38.04</td><td>53.41</td><td>52.22</td><td>24.60</td><td>39.78</td><td>20.18</td></tr><tr><td>Qwen1.5-1.8B-Chat</td><td>1.2B</td><td>37.52</td><td>47.18</td><td>52.11</td><td>26.68</td><td>40.57</td><td>21.05</td></tr><tr><td>Memory3-SFT</td><td>2.4B</td><td>48.60</td><td>56.61</td><td>53.91</td><td>38.80</td><td>57.72</td><td>35.96</td></tr></table>

# 7.4 | Professional Tasks

One benefit of using explicit memory is that the LLM can easily adapt to new fields and tasks by updating its knowledge base. One can simply import task-related references into the knowledge base of Memory3, and optionally, convert them to explicit memories in the case of warm start. Then, the model can perform inference with this new knowledge, skipping the more costly and possibly lossy process of finetuning, and running faster than RAG. This cost reduction has been demonstrated in Figure 4 and Appendix A, and could facilitate the rapid deployment of LLMs across various industries.

Besides cost reduction, we need to demonstrate that Memory3 can perform no worse than RAG. We consider two professional tasks in law and medicine. The legal task consists of multiple-choice questions from the Chinese National Judicial Examination (JEC-QA) dataset [134]. The field-specific references are legal documents from the Chinese national laws and regulations database [1]. These references are merged with our general-purpose knowledge base (Section 4.4) for inference.

The medical task consists of the medicine-related questions of C-Eval, MMLU and CMMLU, specifically from the following subsets:

■ C-Eval: clinical medicine, basic medicine   
■ MMLU: clinical knowledge, anatomy, college medicine, college biology, nutrition, virology, medical genetics, professional medicine

■ CMMLU: anatomy, clinical knowledge, college medicine, genetics, nutrition, traditional Chinese medicine, virology

Our knowledge base is supplemented with medical texts from the open-source medical books dataset [101].

Table 20: Comparison with RAG on professional tasks.   

<table><tr><td rowspan="2">LLM</td><td colspan="3">JEC-QA</td><td colspan="3">MED</td></tr><tr><td>3 refs</td><td>5 refs</td><td>7 refs</td><td>3 refs</td><td>5 refs</td><td>7 refs</td></tr><tr><td>Memory3-2B-SFT</td><td></td><td>39.38</td><td></td><td></td><td>56.22</td><td></td></tr><tr><td>MiniCPM-2B-SFT</td><td>38.83</td><td>37.65</td><td>37.94</td><td>53.73</td><td>53.29</td><td>52.84</td></tr><tr><td>Gemma-2B</td><td>28.16</td><td>28.06</td><td>25.29</td><td>42.04</td><td>42.49</td><td>42.96</td></tr><tr><td>Gemma-2B-it</td><td>30.04</td><td>31.13</td><td>29.34</td><td>41.70</td><td>43.24</td><td>42.66</td></tr><tr><td>Llama-2-7B</td><td>28.06</td><td>24.70</td><td>24.90</td><td>45.14</td><td>44.43</td><td>37.96</td></tr><tr><td>Llama-2-7B-Chat</td><td>26.18</td><td>25.10</td><td>25.20</td><td>48.18</td><td>47.29</td><td>39.39</td></tr><tr><td>Phi-2</td><td>25.00</td><td>25.30</td><td>23.32</td><td>50.05</td><td>45.42</td><td>45.59</td></tr><tr><td>Qwen1.5-1.8B-Chat</td><td>42.98</td><td>43.87</td><td>41.50</td><td>52.16</td><td>52.50</td><td>52.16</td></tr><tr><td>Qwen1.5-4B-Chat</td><td>51.98</td><td>50.49</td><td>50.99</td><td>61.19</td><td>61.02</td><td>61.06</td></tr></table>

The results are shown in Table 20, and Memory3 achieves better performance than most of the models. All evaluations use 5-shot prompting. The RAG models retrieve from the same knowledge bases and FAISS indices, except that they receive text references instead of explicit memories. They only retrieve once for each question, using only the question text for query, so the 5-shot examples do not distract the retrieval. Since the optimal number of references is not known for these RAG models, we test them for 3, 5, and 7 references per question, and it seems that $3 \sim 5$ references are optimal. The usual formatting for RAG is used, i.e. header $^ { 1 + }$ reference $^ { 1 + }$ reference $^ { 2 + }$ reference $3 +$ header $^ { 2 + }$ few-shot examples $^ +$ question, all separated by line breaks.

The performance plotted in Figure 2 (right) is the average of the scores of the two tasks in Table 20 with five references.

# 7.5 | Inference Speed

Finally, we evaluate the decoding speed or throughput of Memory3, measured by generated tokens per second. The results are compared to those of RAG models, to quantify the speedup of explicit memory over text retrieval.

A direct comparison of speeds is uninformative: The memory hierarchy (Figure 8) implies that the Memory3 model is more reliant on retrieval to supply knowledge, and naturally Memory3 performs retrieval with higher frequency (5 references per 64 tokens, possibly higher in future versions). Therefore, it is necessary to jointly compare performance and speed. The speed measured in this section is plotted against the retrieval-augmented test accuracy from Section 7.4, resulting in Figure 2 (right).

We measure decoding speed on a A800 GPU, and run all models with Flash Attention [32]. All models receive an input of batch size 32 and length 128 tokens, and generate an output with length 128 tokens. The throughput is computed by $3 2 \times 1 2 8$ divided by the time spent. We test each model 9 times, remove the first record, and take the average of the rest. Memory3 performs $2 \times 1 2 8 / 6 4 - 1 = 3$ retrievals (the $^ { - 1 }$ means that the first decoded chunk inherits the explicit memories retrieved by the last input chunk). Each retrieval uses 32 queries to get $3 2 \times 5$ explicit memories. We consider the warm start scenario, with the explicit memories precomputed and saved to drives. We implement the worst case scenario, such that the reference ids are reset to be unique after vector search and the memory cache on RAM is disabled, forcing Memory3 to load $3 2 \times 5$ memories from drives. Meanwhile, each RAG model performs one retrieval with query length 64 tokens, receives 5 references for each sample, and inserts them at the beginning of the sample, similar to the setup for Table 20.

The results are listed in Table 21 (local server). The throughput of these models without retrieval is also provided.

In addition, we study the throughput of these models when they are hosted on an end-side device and retrieve from a knowledge base on a remote server. Specifically, we use Jetson AGX Orin, and the server uses the vector engine MyScale [82]. The models are run with plain attention, with batch size 1. To simulate real-world use cases, the input is a fixed text prompt, with approximately 128 tokens, while the exact length can vary among different tokenizers. The output length is fixed to be 128 tokens. The results are listed in Table 21 (end-side device), and the Memory3 model .

Table 21: Inference throughput, measured by tokens per second.   

<table><tr><td rowspan="2">LLM</td><td rowspan="2">Size</td><td colspan="2">Local server</td><td colspan="2">End-side device</td></tr><tr><td>with retrieval</td><td>w/o retrieval</td><td>with retrieval</td><td>w/o retrieval</td></tr><tr><td>Memory3-2B</td><td>2.4B</td><td>733.0</td><td>1131</td><td>27.6</td><td>44.36</td></tr><tr><td>MiniCPM-2B</td><td>2.4B</td><td>501.5</td><td>974.0</td><td>21.7</td><td>51.79</td></tr><tr><td>Gemma-2B-it</td><td>2.0B</td><td>1581</td><td>2056</td><td>22.0</td><td>29.23</td></tr><tr><td>Gemma-7B-it</td><td>7.8B</td><td>395.6</td><td>1008</td><td>9.5</td><td>18.61</td></tr><tr><td>Mistral-7B-Instruct-v0.1</td><td>7.0B</td><td>392.9</td><td>894.5</td><td>11.1</td><td>28.7</td></tr><tr><td>Llama-2-7B-Chat</td><td>6.5B</td><td>382.8</td><td>1005</td><td>10.0</td><td>23.19</td></tr><tr><td>Llama-2-13B-Chat</td><td>13B</td><td>241.1</td><td>632.5</td><td>2.5</td><td>5.44</td></tr><tr><td>Qwen1.5-1.8B-Chat</td><td>1.2B</td><td>908.2</td><td>1770</td><td>-</td><td>-</td></tr><tr><td>Qwen1.5-4B-Chat</td><td>3.2B</td><td>460.7</td><td>1002</td><td>22.3</td><td>53.39</td></tr><tr><td>Qwen1.5-7B-Chat</td><td>6.5B</td><td>365.8</td><td>894.5</td><td>-</td><td>-</td></tr><tr><td>Phi-2</td><td>2.5B</td><td>622.2</td><td>1544</td><td>-</td><td>-</td></tr></table>

Remark 8. Table 21 indicates that our Memory3-2B model is $1 - 7 3 3 / 1 1 3 1 \approx 3 5 . 2 \%$ slower than the same model without using memory. This is peculiar considering that reading explicit memories accounts for only a tiny fraction of the total compute:

$$
\frac{2.884\times 10^{-3}\mathrm{TFlops}}{1.264\mathrm{TFlops}}\approx 0.228\%
$$

(The calculations are based on Appendix A.) Controlled experiments indicate that the time consumption is mainly due to two sources:

Loading the memory key-values from drives to GPU: This overhead becomes prominent as Memory3 retrieves with higher frequency.   
■ Python implementation of chunkwise attention: When encoding a prompt, since each chunk attends to a different set of explicit memories, we use a for loop over the chunks to compute their attentions.

They dominate other sources such as computing query vectors by the embedding model and searching the vector index. We will try to optimize our code to reduce these overheads to be as close as possible to $0 . 2 2 8 \%$ of the total inference time, e.g. implement the chunkwise attention with a CUDA kernel.

# 8 | Conclusion

The goal of this work is to reduce the cost of LLM training and inference, or equivalently, to construct a more efficient LLM that matches the performance of larger and slower LLMs. We analyze LLMs from the new perspective of knowledge manipulation, characterizing the cost of LLMs as the transport cost of “knowledges” in and out of various memory formats. Two causes of inefficiency are identified, namely the suboptimal placement of knowledges and the knowledge traversal problem. We solve both problems with explicit memory, a novel memory format, along with a new training scheme and architecture. Our preliminary experiment, the Memory3-2B model, exhibits stronger abilities and higher speed than many SOTA models with greater sizes as well as RAG models.

For future work, we plan to explore the following directions:

1. Efficient training with abstract knowledges: Ideally, the training cost of Memory3 model should be proportional to the small amount of non-separable knowledges, approaching the learning efficiency of humans. One approach is to filter the training data to maximize abstract knowledges and minimize specific knowledges (cf. Section 3.5 and Remark 6), and preferably the LLM should assess the quality of its own training data and ignore the unhelpful tokens.   
2. Human-like capabilities: As described in the introduction, the explicit memory allows for interesting cognitive functions such as handling infinite contexts (conversion of working memory to explicit memory), memory consolidation (conversion of explicit memory to implicit memory), and conscious reasoning (reflection on the memory recall process). These designs may further improve the efficiency and reasoning ability of Memory3.   
3. Compact representation of explicit memory: The explicit memory of humans can be subdivided into episodic memory, which involve particular experiences, and semantic memory, which involve general

truths [59]. This classification is analogous to our definition of specific and abstract knowledges. Our current implementation of explicit memory is closer to the episodic memory of humans, as each memory directly corresponds to a reference text. To improve its reasoning ability, one can try to equip Memory3 with semantic memories, e.g. obtained from induction on the episodic memories.

Besides these broad topics, there are also plenty of engineering works that can be done. For instance, an internalized retrieval process that matches sparse attention queries with memory keys (Remark 2), sparser memory heads with routing (Remark 5), memory extraction that fully preserves contexts (Remark 3), compilation of the knowledge base based on machine preference (Remark 7), reduction of the time consumption of explicit memory to be proportional to its compute overhead (Remark 8), and so on.

# Acknowledgement

This work is supported by the NSFC Major Research Plan - Interpretable and General Purpose Nextgeneration Artificial Intelligence of China (No. 92270001). We thank Prof. Zhiqin Xu, Prof. Zhouhan Lin, Fangrui Liu, Liangkai Hang, Ziyang Tao, Xiaoxing Wang, Mingze Wang, Yongqi Jin, Haotian He, Guanhua Huang, Yirong Hu for helpful discussions.

# 9 References

[1] The Chinese National Laws and Regulations Database. https://flk.npc.gov.cn/. [Accessed 20-03- 2024].   
[2] Wenshu. https://wenshu.court.gov.cn/. [Accessed 20-03-2024].   
[3] Megatron-DeepSpeed. https://github.com/microsoft/Megatron-DeepSpeed, 2022.   
[4] Marah Abdin, Sam Ade Jacobs, Ammar Ahmad Awan, Jyoti Aneja, Ahmed Awadallah, Hany Awadalla, Nguyen Bach, Amit Bahree, Arash Bakhtiari, Harkirat Behl, Alon Benhaim, Misha Bilenko, Johan Bjorck, S´ebastien Bubeck, and et al. Phi-3 technical report: A highly capable language model locally on your phone, 2024.   
[5] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. GPT-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
[6] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebr´on, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints, 2023.   
[7] Zeyuan Allen-Zhu and Yuanzhi Li. Physics of language models: Part 3.3, knowledge capacity scaling laws, 2024.   
[8] Ebtesam Almazrouei, Hamza Alobeidli, Abdulaziz Alshamsi, Alessandro Cappelli, Ruxandra Cojocaru, M´erouane Debbah, Etienne Goffinet, Daniel Hesslow, Julien Launay, Quentin Malartic, ´ Daniele Mazzotta, Badreddine Noune, Baptiste Pannier, and Guilherme Penedo. The falcon series of open language models, 2023.   
[9] AI Anthropic. The Claude 3 model family: Opus, Sonnet, Haiku. Claude-3 Model Card, 2024.   
[10] Argilla. Distilabel Math Preference DPO. https://huggingface.co/datasets/argilla/ distilabel-math-preference-dpo, 2023.   
[11] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, et al. Qwen technical report, 2023.   
[12] Peter J Bayley and Larry R Squire. Failure to acquire new semantic knowledge in patients with large medial temporal lobe lesions. Hippocampus, 15(2):273–280, 2005.

[13] Edward Beeching, Cl´ementine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open LLM leaderboard. https: //huggingface.co/spaces/HuggingFaceH4/open llm leaderboard, 2023.   
[14] Marco Bellagente, Jonathan Tow, Dakota Mahan, Duy Phung, Maksym Zhuravinskyi, Reshinth Adithyan, James Baicoianu, Ben Brooks, Nathan Cooper, Ashish Datta, et al. Stable lm 2 1.6 b technical report. arXiv preprint arXiv:2402.17834, 2024.   
[15] Amanda Bertsch, Uri Alon, Graham Neubig, and Matthew Gormley. Unlimiformer: Long-range transformers with unlimited length input. Advances in Neural Information Processing Systems, 36, 2024.   
[16] Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. Improving language models by retrieving from trillions of tokens. In International conference on machine learning, pages 2206–2240. PMLR, 2022.   
[17] Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. BGE M3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 2024.   
[18] Yongjian Chen, Tao Guan, and Cheng Wang. Approximate nearest neighbor search by residual vector quantization. Sensors, 10(12):11259–11273, 2010.   
[19] Zheng Chen, Yuqing Li, Tao Luo, Zhaoguang Zhou, and Zhi-Qin John Xu. Phase diagram of initial condensation for two-layer neural networks. ArXiv, abs/2303.06561, 2023.   
[20] Qinyuan Cheng, Tianxiang Sun, Wenwei Zhang, Siyin Wang, Xiangyang Liu, Mozhi Zhang, Junliang He, Mianqiu Huang, Zhangyue Yin, Kai Chen, and Xipeng Qiu. Evaluating hallucinations in chinese large language models, 2023.   
[21] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, et al. Palm: Scaling language modeling with pathways, 2022.   
[22] Bilal Chughtai, Alan Cooney, and Neel Nanda. Summing up the facts: Additive mechanisms behind factual recall in llms. arXiv preprint arXiv:2402.07321, 2024.   
[23] Together Computer. Redpajama: an open dataset for training large language models, October 2023.   
[24] Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adri`a Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. Advances in Neural Information Processing Systems, 36:16318–16352, 2023.   
[25] Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzm´an, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised cross-lingual representation learning at scale. arXiv preprint arXiv:1911.02116, 2019.   
[26] Suzanne Corkin. What’s new with the amnesic patient H.M.? Nature reviews neuroscience, 3(2):153–160, 2002.   
[27] Nelson Cowan. The magical number 4 in short-term memory: A reconsideration of mental storage capacity. Behavioral and Brain Sciences, 24(1):87–114, 2001.   
[28] Nelson Cowan. Working memory capacity: Classic Edition. Routledge, 2016.   
[29] Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. Knowledge neurons in pretrained transformers. arXiv preprint arXiv:2104.08696, 2021.   
[30] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime Carbonell, Quoc V. Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context, 2019.   
[31] Luigi Daniele and Suphavadeeprasit. Amplify-instruct: Synthetically generated diverse multi-turn conversations for effecient llm training. arXiv preprint arXiv:(coming soon), 2023.

[32] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher R´e. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in Neural Information Processing Systems, 35:16344–16359, 2022.   
[33] Nolan Dey, Gurpreet Gosal, Zhiming, Chen, Hemant Khachane, William Marshall, Ribhu Pathria, Marvin Tom, and Joel Hestness. Cerebras-gpt: Open compute-optimal language models trained on the cerebras wafer-scale cluster, 2023.   
[34] Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations, 2023.   
[35] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazar´e, Maria Lomeli, Lucas Hosseini, and Herv´e J´egou. The faiss library, 2024.   
[36] Phung Van Duy. synth code preference 4k. https://huggingface.co/datasets/pvduy/synth code preference 4k, 2023.   
[37] Maha Elbayad, Jiatao Gu, Edouard Grave, and Michael Auli. Depth-adaptive transformer, 2020.   
[38] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition. Transformer Circuits Thread, 2022. https://transformer-circuits.pub/2022/toy model/index.html.   
[39] Junjie Fang, Likai Tang, Hongzhe Bi, Yujia Qin, Si Sun, Zhenyu Li, Haolun Li, Yongjian Li, Xin Cong, Yukun Yan, Xiaodong Shi, Sen Song, Yankai Lin, Zhiyuan Liu, and Maosong Sun. UniMem: Towards a unified view of long-context large language models, 2024.   
[40] William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. J. Mach. Learn. Res., 23:120:1–120:39, 2022.   
[41] Elias Frantar and Dan Alistarh. Qmoe: Practical sub-1-bit compression of trillion-parameter models. CoRR, abs/2310.16795, 2023.   
[42] John DE Gabrieli, Neal J Cohen, and Suzanne Corkin. The impaired learning of semantic knowledge following bilateral medial temporal-lobe resection. Brain and cognition, 7(2):157–177, 1988.   
[43] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, Shawn Presser, and Connor Leahy. The pile: An 800gb dataset of diverse text for language modeling. CoRR, abs/2101.00027, 2021.   
[44] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. A framework for few-shot language model evaluation, 12 2023.   
[45] Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. arXiv preprint arXiv:2304.14767, 2023.   
[46] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories, 2021.   
[47] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pages 249–256. JMLR Workshop and Conference Proceedings, 2010.   
[48] Suriya Gunasekar, Yi Zhang, Jyoti Aneja, Caio C´esar Teodoro Mendes, Allie Del Giorno, Sivakanth Gopi, Mojan Javaheripi, Piero Kauffmann, Gustavo de Rosa, Olli Saarikivi, Adil Salim, Shital Shah, Harkirat Singh Behl, Xin Wang, S´ebastien Bubeck, Ronen Eldan, Adam Tauman Kalai, Yin Tat Lee, and Yuanzhi Li. Textbooks are all you need, 2023.

[49] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929–3938. PMLR, 2020.   
[50] Yiding Hao, Dana Angluin, and Robert Frank. Formal language recognition by hard attention transformers: Perspectives from circuit complexity. Transactions of the Association for Computational Linguistics, 10:800–810, 2022.   
[51] Conghui He, Zhenjiang Jin, Chao Xu, Jiantao Qiu, Bin Wang, Wei Li, Hang Yan, Jiaqi Wang, and Dahua Lin. Wanjuan: A comprehensive multimodal dataset for advancing english and chinese large models. CoRR, abs/2308.10755, 2023.   
[52] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pages 1026–1034, 2015.   
[53] Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.   
[54] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, Xinrong Zhang, Zheng Leng Thai, Kaihuo Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm: Unveiling the potential of small language models with scalable training strategies, 2024.   
[55] Yufei Huang, Shengding Hu, Xu Han, Zhiyuan Liu, and Maosong Sun. Unified view of grokking, double descent and emergent abilities: A perspective from circuits competition. arXiv preprint arXiv:2402.15175, 2024.   
[56] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b, 2023.   
[57] Xiaoqi Jiao, Yichun Yin, Lifeng Shang, Xin Jiang, Xiao Chen, Linlin Li, Fang Wang, and Qun Liu. Tinybert: Distilling bert for natural language understanding, 2020.   
[58] Jean Kaddour. The minipile challenge for data-efficient language models, 2023.   
[59] E.R. Kandel, J.D. Koester, S.H. Mack, and S.A. Siegelbaum. Principles of Neural Science, Sixth Edition. McGraw Hill LLC, 2021.   
[60] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020.   
[61] Omar Khattab and Matei Zaharia. Colbert: Efficient and effective passage search via contextualized late interaction over bert, 2020.   
[62] Vijay Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models, 2022.   
[63] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention, 2023.   
[64] Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. Halueval: A large-scale hallucination evaluation benchmark for large language models. In The 2023 Conference on Empirical Methods in Natural Language Processing, 2023.   
[65] Miao Li, Ming-Bin Chen, Bo Tang, Shengbin Hou, Pengyu Wang, Haiying Deng, Zhiyu Li, Feiyu Xiong, Keming Mao, Peng Cheng, and Yi Luo. Newsbench: A systematic evaluation framework for assessing editorial capabilities of large language models in chinese journalism, 2024.

[66] Yuanzhi Li, S´ebastien Bubeck, Ronen Eldan, Allie Del Giorno, Suriya Gunasekar, and Yin Tat Lee. Textbooks are all you need ii: phi-1.5 technical report, 2023.   
[67] Wing Lian, Guan Wang, Bleys Goodson, Eugene Pentland, Austin Cook, Chanvichet Vong, and ”Teknium”. Slimorca: An open dataset of gpt-4 augmented flan reasoning traces, with verification, 2023.   
[68] Xun Liang, Shichao Song, Simin Niu, Zhiyu Li, Feiyu Xiong, Bo Tang, Yezhaohui Wang, Dawei He, Peng Cheng, Zhonghao Wang, and Haiying Deng. Uhgeval: Benchmarking the hallucination of chinese large language models via unconstrained generation, 2024.   
[69] Xun Liang, Shichao Song, Simin Niu, Zhiyu Li, Feiyu Xiong, Bo Tang, Zhaohui Wy, Dawei He, Peng Cheng, Zhonghao Wang, and Haiying Deng. UHGEval: Benchmarking the hallucination of chinese large language models via unconstrained generation. arXiv preprint arXiv:2311.15296, 2023.   
[70] Stephanie Lin, Jacob Hilton, and Owain Evans. TruthfulQA: Measuring how models mimic human falsehoods. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland, May 2022. Association for Computational Linguistics.   
[71] Hong Liu, Zhiyuan Li, David Hall, Percy Liang, and Tengyu Ma. Sophia: A scalable stochastic second-order optimizer for language model pre-training, 2024.   
[72] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts, 2023.   
[73] Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, and Junxian He. What makes good data for alignment? a comprehensive study of automatic data selection in instruction tuning, 2023.   
[74] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. Scissorhands: Exploiting the persistence of importance hypothesis for llm kv cache compression at test time, 2023.   
[75] Zichang Liu, Jue Wang, Tri Dao, Tianyi Zhou, Binhang Yuan, Zhao Song, Anshumali Shrivastava, Ce Zhang, Yuandong Tian, Christopher R´e, and Beidi Chen. Deja vu: Contextual sparsity for efficient llms at inference time. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pages 22137–22176. PMLR, 2023.   
[76] Shayne Longpre, Gregory Yauney, Emily Reif, Katherine Lee, Adam Roberts, Barret Zoph, Denny Zhou, Jason Wei, Kevin Robinson, David Mimno, et al. A pretrainer’s guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity. arXiv preprint arXiv:2305.13169, 2023.   
[77] Alexandra Sasha Luccioni, Sylvain Viguier, and Anne-Laure Ligozat. Estimating the carbon footprint of bloom, a 176b parameter language model, 2022.   
[78] Tao Luo, Zhi-Qin John Xu, Zheng Ma, and Yaoyu Zhang. Phase diagram for two-layer relu neural networks at infinite-width limit. Journal of Machine Learning Research, 22(71):1–47, 2021.   
[79] Ang Lv, Yuhan Chen, Kaiyi Zhang, Yulong Wang, Lifeng Liu, Ji-Rong Wen, Jian Xie, and Rui Yan. Interpreting key mechanisms of factual recall in transformer-based language models, 2024.   
[80] William Merrill and Ashish Sabharwal. A logic for expressing log-precision transformers, 2023.   
[81] MOP-LIWU Community and MNBVC Team. Mnbvc: Massive never-ending bt vast chinese corpus. https://github.com/esbatmop/MNBVC, 2023.   
[82] MyScale. MyScaleDB. https://github.com/myscale/MyScaleDB. [Accessed 20-03-2024].   
[83] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted questionanswering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

[84] Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Anand Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient large-scale language model training on gpu clusters using megatron-lm, 2021.   
[85] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads, 2022.   
[86] OpenAI. GPT-4 turbo and GPT-4. https://platform.openai.com/docs/models/ gpt-4-turbo-and-gpt-4, 2024. [Accessed 22-05-2024].   
[87] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730– 27744, 2022.   
[88] Adam Pearce, Asma Ghandeharioun, Nada Hussein, Nithum Thain, Martin Wattenberg, and Lucas Dixon. Do machine learning models memorize or generalize? People+ AI Research, 2023.   
[89] Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, et al. Rwkv: Reinventing rnns for the transformer era, 2023.   
[90] Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets, 2022.   
[91] Penghui Qi, Xinyi Wan, Guangxing Huang, and Min Lin. Zero bubble pipeline parallelism, 2023.   
[92] Jack W Rae, Sebastian Borgeaud, Trevor Cai, Katie Millican, Jordan Hoffmann, Francis Song, John Aslanides, Sarah Henderson, Roman Ring, Susannah Young, et al. Scaling language models: Methods, analysis & insights from training gopher. arXiv preprint arXiv:2112.11446, 2021.   
[93] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2024.   
[94] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. Mixture-of-depths: Dynamically allocating compute in transformer-based language models, 2024.   
[95] Jeff Rasley, Samyam Rajbhandari, Olatunji Ruwase, and Yuxiong He. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505–3506, 2020.   
[96] Nir Ratner, Yoav Levine, Yonatan Belinkov, Ori Ram, Inbal Magar, Omri Abend, Ehud Karpas, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. Parallel context windows for large language models. arXiv preprint arXiv:2212.10947, 2022.   
[97] Vipula Rawte, Swagata Chakraborty, Agnibh Pathak, Anubhav Sarkar, S.M Towhidul Islam Tonmoy, Aman Chadha, Amit Sheth, and Amitava Das. The troubling emergence of hallucination in large language models - an extensive definition, quantification, and prescriptive remediations. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2541–2573, Singapore, December 2023. Association for Computational Linguistics.   
[98] Snowflake AI Research. Snowflake arctic: The best LLM for enterprise AI — efficiently intelligent, truly open, Apr 2024. Accessed: 2024-05-15.   
[99] Yangjun Ruan, Chris J. Maddison, and Tatsunori Hashimoto. Observational scaling laws and the predictability of language model performance, 2024.

[100] Timo Schick, Jane Dwivedi-Yu, Roberto Dess`ı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36, 2024.   
[101] Scienceasdf. Medical books. https://github.com/scienceasdf/medical-books. [Accessed 20-03-2024].   
[102] Azure AI Services. GPT-4 and GPT-4 turbo models. https://learn.microsoft.com/en-us/azure/ ai-services/openai/concepts/models#gpt-4-and-gpt-4-turbo-models, 2024. [Accessed 22-05-2024].   
[103] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2020.   
[104] Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://www.cerebras. net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama, June 2023.   
[105] Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adri`a Garriga-Alonso, Agnieszka Kluska, Aitor Lewkowycz, Akshat Agarwal, Alethea Power, Alex Ray, Alex Warstadt, Alexander W. Kocurek, Ali Safaya, Ali Tazarv, Alice Xiang, and et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models, 2023.   
[106] Alessandro Stolfo, Yonatan Belinkov, and Mrinmaya Sachan. A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7035–7052, 2023.   
[107] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.   
[108] Sainbayar Sukhbaatar, Edouard Grave, Guillaume Lample, Herve Jegou, and Armand Joulin. Augmenting self-attention with persistent memory, 2019.   
[109] Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, and Furu Wei. You only cache once: Decoder-decoder architectures for language models. arXiv preprint arXiv:2405.05254, 2024.   
[110] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
[111] Lewis Tunstall, Edward Beeching, Nathan Lambert, Nazneen Rajani, Kashif Rasul, Younes Belkada, Shengyi Huang, Leandro von Werra, Cl´ementine Fourrier, Nathan Habib, Nathan Sarrazin, Omar Sanseviero, Alexander M. Rush, and Thomas Wolf. Zephyr: Direct distillation of lm alignment, 2023.   
[112] Szymon Tworkowski, Konrad Staniszewski, Miko laj Pacek, Yuhuai Wu, Henryk Michalewski, and Piotr Mi lo´s. Focused transformer: Contrastive training for context scaling. Advances in Neural Information Processing Systems, 36, 2024.   
[113] Boxin Wang, Wei Ping, Peng Xu, Lawrence McAfee, Zihan Liu, Mohammad Shoeybi, Yi Dong, Oleksii Kuchaiev, Bo Li, Chaowei Xiao, Anima Anandkumar, and Bryan Catanzaro. Shall we pretrain autoregressive language models with retrieval? A comprehensive study. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7763–7786, Singapore, December 2023. Association for Computational Linguistics.   
[114] Guan Wang, Sijie Cheng, Xianyuan Zhan, Xiangang Li, Sen Song, and Yang Liu. Openchat: Advancing open-source language models with mixed-quality data, 2023.   
[115] Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. arXiv preprint arXiv:2211.00593, 2022.

[116] Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. Label words are anchors: An information flow perspective for understanding in-context learning. arXiv preprint arXiv:2305.14160, 2023.   
[117] Mingze Wang, Haotian He, Jinbo Wang, Zilin Wang, Guanhua Huang, Feiyu Xiong, Zhiyu Li, Weinan E, and Lei Wu. Improving generalization and convergence by enhancing implicit regularization, 2024.   
[118] Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. Augmenting language models with long-term memory. Advances in Neural Information Processing Systems, 36, 2024.   
[119] Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models, 2022.   
[120] Gail Weiss, Yoav Goldberg, and Eran Yahav. Thinking like transformers, 2021.   
[121] Carole-Jean Wu, Ramya Raghavendra, Udit Gupta, Bilge Acun, Newsha Ardalani, Kiwan Maeng, Gloria Chang, Fiona Aga Behram, James Huang, Charles Bai, Michael Gschwind, Anurag Gupta, Myle Ott, Anastasia Melnikov, Salvatore Candido, David Brooks, Geeta Chauhan, Benjamin Lee, Hsien-Hsin S. Lee, Bugra Akyildiz, Maximilian Balandat, Joe Spisak, Ravi Jain, Mike Rabbat, and Kim Hazelwood. Sustainable ai: Environmental implications, challenges and opportunities, 2022.   
[122] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality, 2024.   
[123] Yuhuai Wu, Markus Norman Rabe, DeLesley Hutchins, and Christian Szegedy. Memorizing transformers. In International Conference on Learning Representations, 2021.   
[124] Xingyu Xie, Pan Zhou, Huan Li, Zhouchen Lin, and Shuicheng Yan. Adan: Adaptive nesterov momentum algorithm for faster optimizing deep models, 2023.   
[125] Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023.   
[126] Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, Fan Yang, Fei Deng, Feng Wang, Feng Liu, Guangwei Ai, Guosheng Dong, Haizhou Zhao, et al. Baichuan 2: Open large-scale language models, 2023.   
[127] Greg Yang, Edward J Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer. arXiv preprint arXiv:2203.03466, 2022.   
[128] Hongkang Yang. A mathematical framework for learning probability distributions. Journal of Machine Learning, 1(4):373–431, 2022.   
[129] Yunzhi Yao, Ningyu Zhang, Zekun Xi, Mengru Wang, Ziwen Xu, Shumin Deng, and Huajun Chen. Knowledge circuits in pretrained transformers, 2024.   
[130] Longhui Yu, Weisen Jiang, Han Shi, Jincheng Yu, Zhengying Liu, Yu Zhang, James T Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. arXiv preprint arXiv:2309.12284, 2023.   
[131] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher R´e, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36, 2024.   
[132] Zhongwang Zhang, Pengxiao Lin, Zhiwei Wang, Yaoyu Zhang, and Zhi-Qin John Xu. Initialization is critical to whether transformers fit composite functions by inference or memorizing, 2024.   
[133] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36, 2024.

[134] Haoxi Zhong, Chaojun Xiao, Cunchao Tu, Tianyang Zhang, Zhiyuan Liu, and Maosong Sun. Jec-qa: A legal-domain question answering dataset. Proceedings of the AAAI Conference on Artificial Intelligence, 34(05):9701–9708, Apr. 2020.

# A | Cost Estimation

This section provides the calculations for Figure 4, and we equate cost with the amount of compute measured in Tflops.

Our 2.4B Memory3 model is adopted as the backbone. Recall from Section 3.4 that this model has shape

■ Transformer blocks $L = 4 4$   
■ Query heads $H = 4 0$ and key-value heads $H _ { k v } = 8$   
■ Head dimension $d _ { h } = 8 0$ and hidden dimension $d = H d _ { h } = 3 2 0 0$   
■ MLP width $W = d$   
■ Vocabulary size as well as LM head size $n _ { \mathrm { v o c a b } } = 6 0 4 1 6$   
■ memory layers $L _ { \mathrm { m e m } } = 2 2$ , which is also the depth of the deepest memory layer.

Fix a separable knowledge $\kappa$ , and represent it by one of its realizations t (Definition 5), and assume that t has length $l _ { \mathrm { r e f } } = 1 2 8$ tokens, following the setup of our reference dataset (Section 4.4). Recall from Section 3.3 that each memory has $l _ { \mathrm { { m e m } } } = 8$ tokens per memory head, and it is read by a chunk of length $l _ { \mathrm { c h u n k } } = 6 4$ .

Since we want to show that explicit memory is cheaper than implicit memory and RAG, it suffices to use coarse lower bounds on their costs.

# A.1 | Implicit Memory

The write cost of implicit memory or model parameters is the training compute with t as input. Usually the training data of Transformer LLMs have length $2 0 4 8 \sim 8 1 9 2$ , so we assume that t is a subsequence of a train sample $\mathbf { t } _ { \mathrm { t r a i n } }$ with length $l _ { \mathrm { t r a i n } } = 2 0 4 8$ . By [84], the training compute of one step with one sample is approximately

$$
3 \cdot 2 \cdot \left[ L \left(l _ {\mathrm {t r a i n}} (2 d ^ {2} + 2 d d _ {h} H _ {k v} + 3 d W) + 2 \frac {l _ {\mathrm {t r a i n}} ^ {2}}{2} d\right) + l _ {\mathrm {t r a i n}} n _ {\mathrm {v o c a b}} d \right]
$$

where 3 means that the backward step costs twice as the forward step (and thus 3 times in total), the first 2 means that the compute of matrix multiplication involves same amount of additions and multiplications. The five terms in the bracket come from QO embedding, KV embedding, MLP, attention, and LM head, respectively. The lower order terms, such as layer normalizations, are omitted. The fraction of the compute attributable to t is given by

$$
3 \cdot 2 \cdot \left[ L \big (l _ {\mathrm {r e f}} (2 d ^ {2} + 2 d d _ {h} H _ {k v} + 3 d W) + 2 l _ {\mathrm {r e f}} \frac {l _ {\mathrm {t r a i n}}}{2} d \big) + l _ {\mathrm {r e f}} n _ {\mathrm {v o c a b}} d \right]
$$

Assume that one training step is sufficient for storing knowledge $\kappa$ into model parameters. Then, the write cost is equal to the above term, and we obtain

$$
\mathrm {c o s t} _ {\mathrm {w r i t e}} \approx 2. 2 4 \mathrm {T F l o p s}
$$

Meanwhile, we lower bound the read cost by zero.

$$
\mathrm {c o s t} _ {\mathrm {r e a d}} \geq 0 \mathrm {T F l o p s}
$$

This lower bound is obviously correct and suits our comparison, since it makes implicit memory appear more competitive. The difficulty in estimating the cost is that the correspondence between knowledges and parameters is not fully understood. Nevertheless, we describe a possible way to obtain a reasonable bound. Recall from Section 1 that the model parameters suffer from the issue of knowledge traversal such that each parameter (and thus each implicit memory) is invoked during each call of the LLM. So

the read cost of each implicit memory does not depend on its usage count $n _ { k }$ , but instead on the total amount of model calls during the lifespan of this LLM. Dividing the total amount of inference compute used by this LLM by the amount of knowledges it possesses gives an estimation of the average read cost of a knowledge. The amount of knowledges in the LLM can be upper bounded based on the knowledge capacities measured by [7].

# A.2 | Explicit Memory

The write cost of an each explicit memory mainly comes from $L _ { \mathrm { m e m } }$ self-attention layers, $L _ { \mathrm { m e m } } - 1$ MLP layers, and $L _ { \mathrm { m e m } }$ token sparsification operations (computing the full attention matrix):

$$
\begin{array}{l} \mathrm {c o s t} _ {\mathrm {w r i t e}} = 2 \cdot \left[ L _ {\mathrm {m e m}} \big (l _ {\mathrm {r e f}} (2 d ^ {2} + 2 d d _ {h} H _ {k v}) + 2 \frac {l _ {\mathrm {r e f}} ^ {2}}{2} d \big) + (L _ {\mathrm {m e m}} - 1) (l _ {\mathrm {r e f}} \cdot 3 d W) + L _ {\mathrm {m e m}} (l _ {\mathrm {r e f}} ^ {2} d) \right] \\ \approx 0. 3 0 8 \mathrm {T F l o p s} \\ \end{array}
$$

The read cost consists of the attention to the sparse tokens of an explicit memory from the chunk that retrieves this memory:

$$
\mathrm {c o s t} _ {\mathrm {r e a d}} = 2 L _ {\mathrm {m e m}} \cdot 2 l _ {\mathrm {c h u n k}} l _ {\mathrm {m e m}} d \approx 1. 4 4 \times 1 0 ^ {- 4} \mathrm {T F l o p s}
$$

# A.3 | External Information

The write cost of text retrieval-augmented generation (RAG) is set to be zero, since the reference is stored as plain text.

$$
\mathrm {c o s t} _ {\mathrm {w r i t e}} = 0 \mathrm {T F l o p s}
$$

The read cost is the additional compute brought by the retrieved references that are inserted in the prompt. To make RAG appear more competitive, we assume that only a chunk of the prompt or decoded text with length $l _ { \mathrm { c h u n k } }$ can attend to the references, and each reference can only attend to itself, which in general is not true. Then,

$$
\begin{array}{l} \operatorname {c o s t} _ {\text {w r i t e}} \geq 2 \cdot \left[ L \left(l _ {\text {r e f}} \left(2 d ^ {2} + 2 d d _ {h} H _ {k v}\right) + 2 l _ {\text {r e f}} \left(\frac {l _ {\text {r e f}}}{2} + l _ {\text {c h u n k}}\right) d\right) + (L - 1) \left(l _ {\text {r e f}} \cdot 3 d W\right) \right] \\ \approx 0. 6 2 4 \mathrm {T F l o p s} \\ \end{array}
$$

In summary, the total cost (TFlops) of writing and reading each separable knowledge in terms of its expected usage count $_ n$ is given by

$$
\left\{ \begin{array}{l} c _ {\text {i m p l i c i t}} (n) \geq 2. 2 4 \\ c _ {\text {e x p l i c i t}} (n) = 0. 3 0 8 + 0. 0 0 0 1 4 4 n \\ c _ {\text {e x t e r n a l}} (n) \geq 0. 6 2 4 n \end{array} \right.
$$

These curves are plotted in Figure 4. Hence, if $n \in ( 0 . 4 9 4 , 1 3 4 0 0 )$ , then it is optimal to store the knowledge as an explicit memory.

Remark 9 (Knowledge retention). One aspect not covered by Problem (1) is the retention of knowledges in the model if its parameters are updated, e.g. due to finetuning. Both implicit memory and explicit memory are vulnerable to parameter change. Usually, model finetuning would include some amount of pretrain data to prevent catastrophic forgetting [87]. Similarly, if some explicit memories have already been produced, then they need to be rebuilt in order to remain readable by the updated model. It is an interesting research direction to design a more efficient architecture such that the implicit and explicit memories are robust with respect to model updates.

# B | Vector Compression

Regarding the vector quantizer discussed in Sections 3.3 and 7.1, we use the composite index of FAISS with index type OPQ20x80-Residual2x14-PQ8x10. It can encode a 80-dimensional bfloat16 vector into a 14-dimensional uint8 vector, and thus its compression rate is $\frac { 8 0 \times 2 } { 1 4 \times 1 } \approx 1 1 . 4$ .

To train this quantizer, we sample references from our knowledge base, encode them into explicit memories by our Memory3-2B-SFT model, and feed these key-value vectors to the quantizer. The references are sampled uniformly and independently, so the training is not biased towards the references that are retrieved by any specific evaluation task.

# C | Supplementary Evaluation Results

First, Table 22 records the growth of the test scores (Table 16) over the three training stages: warmup, continual train, and SFT. We believe that for future versions of Memory3, fixing the loss divergence during the warmup stage can allow the continual train stage to proceed much further (cf. Section 5.3), and thus increase the performance boost of this stage.

Table 22: Performance of Memory3-2B at different stages of training. The setup of the evaluation tasks is the same as in Table 16.   

<table><tr><td rowspan="2">LLM</td><td rowspan="2">Avg.</td><td colspan="5">English</td><td colspan="2">Chinese</td></tr><tr><td>ARC-C</td><td>HellaSwag</td><td>MMLU</td><td>Winogrand</td><td>GSM8k</td><td>CEVAL</td><td>CMMLU</td></tr><tr><td>Warmup</td><td>42.13</td><td>40.27</td><td>64.57</td><td>41.62</td><td>61.96</td><td>5.23</td><td>40.12</td><td>41.17</td></tr><tr><td>Continual train</td><td>45.12</td><td>42.66</td><td>79.21</td><td>41.81</td><td>59.43</td><td>6.29</td><td>42.20</td><td>44.21</td></tr><tr><td>- without memory</td><td>42.89</td><td>42.15</td><td>66.98</td><td>39.79</td><td>61.80</td><td>6.44</td><td>39.97</td><td>43.13</td></tr><tr><td>SFT</td><td>63.31</td><td>58.11</td><td>80.51</td><td>59.68</td><td>74.51</td><td>52.84</td><td>59.29</td><td>58.24</td></tr><tr><td>- without memory</td><td>60.80</td><td>57.42</td><td>73.14</td><td>57.29</td><td>74.35</td><td>51.33</td><td>56.32</td><td>55.72</td></tr></table>

Next, recall that for the evaluations in Section 7.1, a filter is included in the retrieval process to prevent copying, which removes references that overlap too much with the evaluation question. The filtering threshold should lie between $1 0 0 \%$ and the usual level of overlap between two related but distinct texts, and we set it to 2/3 in Table 16. Table 23 records the impact of the filtering threshold on the test scores. The scores are stable for most tasks, indicating that their questions do not appear in our knowledge basis.

Table 23: Influence of the filtering threshold on the test scores in Table 16.   

<table><tr><td>Threshold</td><td>Avg.</td><td>ARC-C</td><td>HellaSwag</td><td>MMLU</td><td colspan="2">Winogrande GSM8k</td><td>CEVAL</td><td>CMMLU</td></tr><tr><td>no filter</td><td>63.71</td><td>58.11</td><td>83.37</td><td>59.65</td><td>74.51</td><td>52.84</td><td>59.29</td><td>58.22</td></tr><tr><td>80%</td><td>63.62</td><td>58.11</td><td>82.69</td><td>59.65</td><td>74.51</td><td>52.84</td><td>59.29</td><td>58.24</td></tr><tr><td>2/3</td><td>63.31</td><td>58.11</td><td>80.51</td><td>59.68</td><td>74.51</td><td>52.84</td><td>59.29</td><td>58.24</td></tr><tr><td>without memory</td><td>60.80</td><td>57.42</td><td>73.14</td><td>57.29</td><td>74.35</td><td>51.33</td><td>56.32</td><td>55.72</td></tr></table>

Finally, Table 24 studies the influence of the few-shot prompts on the benchmark tasks. Recall that the number of few-shot examples for each task is ARC-C (25), HellaSwag (10), MMLU (5), Winogrande (5), GSM8k (5) as in HuggingFace OpenLLM Leaderboard [13], and we also adopt CEVAL (5), CMMLU (5). Interestingly, the boost from explicit memory increases from $2 . 5 1 \%$ to $3 . 7 0 \%$ as we switch to 0-shot.

Table 24: Few-shot versus 0-shot for the benchmark tasks in Table 16.   

<table><tr><td>Mode</td><td>Avg.</td><td>ARC-C</td><td>HellaSwag</td><td>MMLU</td><td colspan="2">Winogrande GSM8k</td><td>CEVAL</td><td>CMMLU</td></tr><tr><td>Few-shot</td><td>63.31</td><td>58.11</td><td>80.51</td><td>59.68</td><td>74.51</td><td>52.84</td><td>59.29</td><td>58.24</td></tr><tr><td>- without memory</td><td>60.80</td><td>57.42</td><td>73.14</td><td>57.29</td><td>74.35</td><td>51.33</td><td>56.32</td><td>55.72</td></tr><tr><td>0-shot</td><td>58.23</td><td>58.79</td><td>83.29</td><td>60.53</td><td>75.85</td><td>13.50</td><td>57.95</td><td>57.74</td></tr><tr><td>- without memory</td><td>54.54</td><td>57.34</td><td>73.15</td><td>58.59</td><td>74.98</td><td>10.46</td><td>54.53</td><td>54.26</td></tr></table>
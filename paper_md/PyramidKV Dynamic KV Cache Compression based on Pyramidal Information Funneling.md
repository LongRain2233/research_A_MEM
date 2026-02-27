# PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling

Zefan Cai1, Yichi Zhang2, Bofei $\mathbf { G a o } ^ { 2 } .$ , Yuliang $\mathbf { L i u } ^ { 3 }$ , Yucheng $\mathbf { L i } ^ { 4 }$ , Tianyu Liu5, Keming $\mathbf { L } \mathbf { u } ^ { 5 }$ , Wayne Xiong7, Yue Dong6, Junjie $\mathbf { H } \mathbf { u } ^ { 1 }$ , Wen Xiao7,

1University of Wisconsin - Madison 2Peking University 3Nanjing University

3University of Surrey 5Qwen 6University of California - Riverside 7Microsoft zefncai@gmail.com

https://github.com/Zefan-Cai/PyramidKV

# Abstract

In this study, we investigate whether attention-based information flow inside large language models (LLMs) is aggregated through noticeable patterns for long context processing. Our observations reveal that LLMs aggregate information through Pyramidal Information Funneling where attention is scattering widely in lower layers, progressively consolidating within specific contexts, and ultimately focusing on critical tokens (a.k.a massive activation or attention sink) in higher layers. Motivated by these insights, we developed PyramidKV, a novel and effective KV cache compression method. This approach dynamically adjusts the KV cache size across different layers, allocating more cache in lower layers and less in higher ones, diverging from traditional methods that maintain a uniform KV cache size. Our experimental evaluations, utilizing the LongBench benchmark, show that PyramidKV matches the performance of models with a full KV cache while retaining only $1 2 \%$ of the KV cache, thus significantly reducing memory usage. In scenarios emphasizing memory efficiency, where only $0 . 7 \%$ of the KV cache is maintained, PyramidKV surpasses other KV cache compression techniques, achieving up to a 20.5 absolute accuracy improvement on TREC dataset. In the Needle-in-a-Haystack experiment, PyramidKV outperforms competing methods in maintaining long-context comprehension in LLMs; notably, retaining just 128 KV cache entries enables the LLAMA-3-70B model to achieve 100.0 Acc. performance.

# 1 Introduction

Large language models (LLMs) (Achiam et al., 2023; Touvron et al., 2023a;b; Jiang et al., 2023) are integral to various natural language processing applications, including dialogue systems (Chiang et al., 2023), document summarization (Fabbri et al., 2019a), and code completion (Roziere et al., 2023). These models have recently been scaled up to handle long contexts (Fu et al., 2024; Ding et al., 2024; Zhu et al., 2023; Chen et al., 2023), with GPT-4 processing up to 128K tokens and Gemini-pro-1.5 handling 1M tokens. However, scaling LLMs to extremely long contexts naturally leads to a significant delay due to the quadratic computation of attention over long contexts. A common solution to mitigate such inference delays involves caching the key and value states (KV) of previous tokens (Waddington et al., 2013), with the trade-off of requiring extensive GPU memory storage. For instance, maintaining a KV cache for 100K tokens in LLaMA-2 7B requires over 50GB of memory, while a 2K context requires less than 1GB of memory (Wu et al., 2024).

To tackle these memory constraints, recent studies have explored the optimization of KV caching, including approaches such as low-rank decomposition of the KV cache (Dong et al., 2024) or pruning non-essential KV cache (Zhang et al., 2024; Li et al., 2024; Ge et al., 2023). Notably, it has been shown that maintaining merely $2 0 \%$ of the KV cache can preserve a substantial level of performance (Zhang et al., 2024). Moreover, extreme compression of the

![](images/e38c7fb899489911f98589d9ca3e7fce2071b5cf01de71919894d89933a90d1e.jpg)

![](images/af108fecdf82d4ad3410ef29de952730190f3dc091bbf66fba5e6ca2e395c98e.jpg)

![](images/35fc565562119fcda1214095712025bcbce2767a8b6951f16e90d3d46862b3eb.jpg)

![](images/57cd026cf500d6732c6ac5ab45bc0bb9c3c7269fda6c381e84bab5e81293afd8.jpg)  
Figure 1: Illustration of PyramidKV compared with existing KV cache compression methods. (a) Full KV has all tokens stored in the KV cache in each layer; cache size increases as the input length increases. (b) StreamingLLM (Xiao et al., 2023) only keeps few initial tokens with a fixed cache size in each layer. (c) SnapKV (Li et al., 2024) and H2O (Zhang et al., 2024) keep a fixed cache size across Transformer layers, and their selection is based on the attention score. (d) PyramidKV maintains pyramid-like cache sizes, allocating more cache budget to lower layers and less to higher layers. This approach to KV cache selection better aligns with the increasing attention sparsity observed in multi-layer Transformers (§3).

KV cache for tasks of longer contexts (e.g., retrieval augmented generation or RAG for short) can drastically improve efficiency and further reduce resource use. However, questions about the universal applicability of these strategies across all layers of an LLM remain open. (1) Are these KV cache strategies applicable to all layers? (2) Is it computationally efficient to use the same KV cache size across layers as previous studies have done? These considerations suggest a need for an in-depth, more nuanced understanding of KV cache optimization in LLMs.

To examine these questions, we aim to systematically investigate the design principles of the KV cache compression across different layers, specifically tailored to the behaviors of the attention mechanism. We first investigate how information flow is aggregated via attention mechanisms across different layers in multi-document question answering (QA), a classic task involving long contexts. Our analysis identifies a notable transition of attention distribution from a broad coverage of global contexts to a narrow focus of local tokens over layers in LLMs. This pattern suggests an aggregated information flow where information is initially gathered broadly and subsequently narrowed down to key tokens, epitomizing the massive attention phenomenon. Our findings provide unique insights beyond the previously documented “massive activation” (Sun et al., 2024) that very few activations exhibit significantly larger values than others when calculating multi-head attention in LLMs and “attention sink” (Xiao et al., 2023) that keeping the KV of initial tokens will largely recover the performance of window attention.

Building on these insights on how information flows are aggregated through a pyramid pattern, we design a novel and effective KV cache pruning approach that mirrors the geometric shape, named PyramidKV. As shown in Figure 1, unlike the fixed-and-same length KV cache pruning common in prior works (Zhang et al., 2024; Ge et al., 2023; Li et al., 2024), PyramidKV allocates more KV cache to the lower layers where information is more dispersed and each KV holds less information while reducing the KV cache in higher layers where information becomes concentrated in fewer key tokens. To the best of our knowledge, PyramidKV is the first KV cache compression method with varied cache retention across layers, tailoring cache amounts to the informational needs of each layer.

We conducted comprehensive experiments on LongBench (Bai et al., 2023) using 17 datasets across various tasks and domains with three backbone models (LLaMa-3-8B-Instruct, LLaMa-3-70B-Instruct and Mistral-7B (Jiang et al., 2023)). The results show that PyramidKV preserves performance using just $1 2 . 0 \%$ of the KV cache (KV Cache size $= 2 0 \dot { 4 } 8 \dot { }$ ) on the LongBench benchmark and significantly outperforms other methods in extreme conditions, retaining only $0 . 7 \%$ of the KV cache. Moreover, PyramidKV outperforms baseline models (H2O (Zhang et al., 2024), SnapKV (Li et al., 2024), StreamingLLM (Xiao et al., 2023)) across all tested cache sizes (64, 96, 128, 256), with its advantages most pronounced at smaller

![](images/1d93b57a7944b6de88c4ea3329fea647f51c46ceab118de9c4ab17bde8469309.jpg)  
Figure 2: Attention patterns of retrieval-augmented generation across layers in LlaMa (Touvron et al., 2023a;b) reveal that in the lower layers, the model exhibits a broad-spectrum mode of attention, distributing attention scores uniformly across all content. In the middle layers, attention becomes more localized within each document, indicating refined information aggregation (dotted red triangular shapes in layers 6 and 10). This culminates in the upper layers, where “massive attention” focuses on a few key tokens (concentrated attention bars after layer 18), efficiently extracting essential information for answers.

cache sizes. In the Needle In A Haystack experiment, PyramidKV effectively maintains the long-context comprehension in LLMs, outperforming than competing methods. Remarkably, with PyramidKV, retaining only $1 2 8 \mathrm { K } \dot { \mathrm { V } }$ cache entries allows the LLaMa-3-70B-Instruct model to achieve 100.0 Acc. performance, matching the performance of a full KV cache.

# 2 Related Work

There has been a growing interest in addressing LLMs’ memory constraints on processing long context inputs. FastGen (Ge et al., 2023) introduces an adaptive KV cache management strategy that optimizes memory use by tailoring retention tactics to the specific nature of attention heads. SnapKV (Li et al., 2024) improves efficiency by compressing KV caches via selecting/clustering significant KV positions based on their attention scores. Heavy Hitter Oracle (H2O) (Zhang et al., 2024) implements a dynamic eviction policy that effectively balances the retention of recent and historically significant tokens, optimizing memory usage while preserving essential information. StreamingLLM (Xiao et al., 2023) enables LLMs trained on finite attention windows to handle infinite sequence lengths without fine-tuning, thus expanding the models’ applicability to broader contexts.

# 3 Pyramidal Information Funneling

To systematically understand the attention mechanism over layers in LLMs for long-context inputs, we conduct a fine-grained study focusing on the multi-document question answering (QA) task. The model is presented with multiple interrelated documents and prompted to generate an answer for the given query. The main target is to investigate how the model aggregates dispersed information within these retrieved documents for accurate responses.

In particular, we focus on our analysis of the LLaMa (Touvron et al., 2023a;b) and visualize the distribution and behavior of attention scores over layers. To assess the distinct behaviors of each multi-head self-attention layer, we compute the average attention from all heads within each layer. Figure 2 shows the attention patterns of one QA example over six different layers (i.e., 0, 6, 12, 18, 24, and 30).

We identify an approximately uniform distribution of attention scores from the lower layers (e.g., the 0th layer). This suggests that the model operates in a broad-spectrum mode at the lower layers, aggregating information globally from all available content without prioritizing its attention on specific input segments. Notably, a distinct transition to a more localized attention pattern within each document emerges, as the model progresses to encode information at the middle layers (6th to 18th layers). In this phase, attention is predominantly directed towards tokens within the same document, suggesting a more refined aggregation of information within individual contexts.

This trend continues and intensifies in the upper layers (from the 24th to the 30th layer), where we observed the emergence of ‘massive attention’ phenomena. In these layers, the attention mechanism concentrates overwhelmingly on a few key tokens. This pattern of attention allocation, where extremely high attention scores are registered, signifies that the model has aggregated the essential information into these focal tokens. Such behavior underscores a sophisticated mechanism by which LLMs manage and streamline complex and voluminous information, culminating in the efficient extraction of the most pertinent data points necessary for generating accurate answers.

# 4 PyramidKV

# 4.1 Preliminaries and Problem Formulation

In an autoregressive transformer LLM, the generation of the $i .$ -th token requires that the attention module computes the query, key, and value vectors for all previous $i - 1$ tokens. To speed up inference process and avoid duplicate computations, the key and value matrices are typically stored in the GPU memory. While the KV cache enhances inference speed and reduces redundant computations, it can consume significant memory when dealing with long input contexts. To optimize memory usage, a strategy called KV cache compression is proposed (Zhang et al., 2024; Xiao et al., 2023; Li et al., 2024), which involves retaining only a minimal amount of KV cache while preserving as much information as possible.

In a LLM with m transformer layers, we denote the key and value matrices in the l-th attention layer respectively as $K ^ { l }$ ${ K } ^ { l } , \dot { \boldsymbol { V } } ^ { l } \in \mathbb { R } ^ { n \times d } , \forall l \in [ 0 , m - \dot { 1 } ]$ when encoding a sequence of $n$ tokens. The goal of KV cache compression is to seek two sub-matrices $K _ { s } ^ { l } , V _ { s } ^ { l } \in \mathbb { R } ^ { k ^ { l } \times d }$ from the full matrices $K ^ { l }$ and $V ^ { l }$ , given a cache budget $k ^ { l } < n$ for each layer $l \in [ 0 , m - 1 ]$ while maximizing performance preservation. A LLM with KV cache compression only uses $K _ { s } ^ { l }$ and $V _ { s } ^ { l }$ in the GPU memory for inference on a dataset $\mathcal { D }$ , and obtains a similar result to a full model according to an evaluation scoring metric, i.e., score $( K ^ { l } , V ^ { l } , { \mathcal { D } } ) \approx \operatorname { s c o r e } ( K _ { s } ^ { l } , V _ { s } ^ { l } , { \mathcal { D } } )$ .

# 4.2 Proposed Method

In this section, we introduce our method, PyramidKV, based on the pyramidal information funneling observed across different layers in §3. PyramidKV consists of two steps: (1) Dynamically allocating different KV cache sizes/budgets across different layers $( \hat { \mathbb { S } } 4 . 2 . 1 )$ ; and (2) Selecting important KV vectors in each attention head for caching (§4.2.2).

# 4.2.1 KV Cache Size/Budget Allocation

Previous work on KV cache compression (Li et al., 2024; Zhang et al., 2024; Xiao et al., 2023) often allocates a fixed KV cache size across LLM layers. However, as our analysis in §3 demonstrates, attention patterns are not identical across different layers. Particularly dense attention is observed in the lower layers, and sparse attention in higher layers. Therefore,

using a fixed KV cache size across layers may lead to suboptimal performance. These approaches may retain many unimportant tokens in the higher layers of sparser attentions while potentially overlooking many crucial tokens in the lower layers of denser attentions.

Thus, we propose to increase compression efficiency by dynamically allocating the cache budgets across layers to reflect the aggregated information flow based on attention patterns. Specifically, PyramidKV allocates more KV cache to the lower layers where information is more dispersed and each KV state contains less information, while reducing the KV cache in higher layers where information becomes concentrated in a few key tokens.

Following the common practice in KV cache compression (Li et al., 2024; Xiao et al., 2023), we first retain the KV cache for the last α tokens of the input across all layers, as these tokens have been shown to contain the most immediate task-related information, where $\alpha$ is a hyperparameter, controlling the number of last few tokens being included in the KV cache. For simplicity, we call these tokens “instruction tokens”, which is also referred to as “local window” in previous literature (Zhang et al., 2024; Li et al., 2024; Xiao et al., 2023).

Subsequently, given the remaining total cache budget $\begin{array} { r } { k ^ { \mathrm { t o t a l } } = \sum _ { l \in [ 0 , m - 1 ] } k ^ { l } } \end{array}$ that can be used over all transformer layers (noted as $m$ ), we first determine the cache sizes for the top and bottom layers, and use an arithmetic sequence to compute the cache sizes for the intermediate layers to form the pyramidal shape. The key intuition is to follow the attention pattern in aggregated information flow, reflecting a monotonically decreasing pattern of important tokens for attention from lower layers to upper layers. We allocate $\bar { k } ^ { m - 1 } = k ^ { \mathrm { t o t a l } } \bar { / } ( \beta \cdot m )$ for the top layer and $k ^ { 0 } = ( 2 \cdot k ^ { \mathrm { t o t a l } } ) / m - k ^ { \hat { m } - 1 }$ for the bottom layer„ where $\beta$ is a hyperparameter to adjust the pyramid’s shape. The hyperparameter $\beta$ is still required to determine the top layer. Once the top layer is identified, the budget of the bottom layer can be calculated by summing the budgets across all layers and equating this sum to the total budget. Once the cache sizes of the bottom and top layers are determined, the cache sizes for all intermediate layers are set according to an arithmetic sequence, defined as

$$
k ^ {l} = k ^ {0} - \frac {k ^ {0} - k ^ {m - 1}}{m - 1} \times l. \tag {1}
$$

# 4.2.2 KV Cache Selection

Once the KV cache budget is determined for each layer, our method needs to select specific KV states for caching within each layer in LLMs. As described in the previous section, the KV cache of the last α tokens, referred to as instruction tokens, are retained across all layers. Following SnapKV (Li et al., 2024), the selection of the remaining tokens is then guided by the attention scores derived from these instruction tokens—tokens receiving higher attention scores are deemed more relevant to the generation process and are thus their KV states are prioritized for retention in the GPU cache.

In a typical LLM, the attention mechanism in each head $h$ is calculated using the formula:

$$
\boldsymbol {A} ^ {h} = \operatorname {s o f t m a x} \left(\boldsymbol {Q} ^ {h} \cdot \left(\boldsymbol {K} ^ {h}\right) ^ {\top} / \sqrt {d _ {k}}\right), \tag {2}
$$

where $d _ { k }$ denotes the dimension of the key vectors. Following (Li et al., 2024), we utilize a pooling layer at $A ^ { h }$ to avoid the risk of being misled by some massive activation scores.

To quantify the importance of each token during the generation process, we measure the level of attention each token receives from the instruction tokens, and use this measurement to select important tokens for KV caching. Specifically, we compute the score of selecting $i$ -th token for retention in the KV cache as $s _ { i } ^ { h }$ in each attention head $h$ by:

$$
s _ {i} ^ {h} = \sum_ {j \in [ n - \alpha , n ]} A _ {i j} ^ {h} \tag {3}
$$

where $\left[ n - \alpha , n \right]$ is the range of the instruction tokens. In each layer l and for each head $h ,$ the top $k ^ { l }$ tokens with the highest scores are selected, and their respective KV caches are retained. All other KV caches are discarded and will not be utilized in any subsequent computations throughout the generation process.

# 5 Experiment

We conduct comprehensive experiments to evaluate the effectiveness of PyramidKV on performance preserving and memory reduction.

# 5.1 Experiment Setup

We maintain a fixed constant KV cache size for each layer for the baseline methods. In contrast, PyramidKV employs varying KV cache sizes across different layers. To ensure a fair comparison, we adjusted the average KV cache size in PyramidKV to match that of the baseline models, to keep the total memory consumption of all methods the same. We set $\beta = 2 0$ and $\alpha = 8$ . We use the same prompt for each dataset in all experiments.

# 5.1.1 Backbone LLMs

We compare PyramidKV against baselines using state-of-the-art open-sourced LLMs, namely LLaMa-3-8B-Instruct, Mistral-7B-Instruct (Jiang et al., 2023) and LLaMa-3-70B-Instruct. Testing examples are evaluated in a generative format, with answers generated by greedy decoding across all tasks to ensure a fair comparison.

# 5.1.2 Datasets

We use LongBench (Bai et al., 2023) to assess the performance of PyramidKV on tasks involving long-context inputs. LongBench is a meticulously designed benchmark suite that tests the capabilities of language models in handling extended documents and complex information sequences. This benchmark was created for comprehensive multi-task evaluation of long context inputs. It includes 17 datasets covering tasks such as single-document QA (Koˇcisky et al., 2018; Dasigi et al., 2021), multi-document QA (Yang et al., 2018; Ho ` et al., 2020), summarization (Huang et al., 2021; Zhong et al., 2021; Fabbri et al., 2019b), few-shot learning (Li and Roth, 2002; Gliwa et al., 2019; Joshi et al., 2017), synthetic, and code generation (Guo et al., 2023; Liu et al., 2023b). The datasets feature an average input length ranging from 1,235 to 18,409 tokens (detailed average lengths can be found in Table 1), necessitating substantial memory for KV cache management. For all these tasks, we adhered to the standard metrics recommended by LongBench (Bai et al., 2023) (i.e., F1 for QA, Rouge-L for summarization, Acc. for synthetic and Edit Sim. for code generation.) We refer readers to more details at Appendix F.

# 5.1.3 Baselines

We compare PyramidKV with three baselines, all of which keep the same KV cache size across different layers, with different strategies for KV cache selection.

• StreamingLLM (SLM) (Xiao et al., 2023) is an efficient framework that enables LLMs to accept infinite input length.   
• Heavy Hitter Oracle (H2O) (Zhang et al., 2024) is a KV cache compression policy that dynamically retains a balance of recent and Heavy Hitter (H2) tokens.

SnapKV (SKV) (Li et al., 2024) automatically compresses KV caches by selecting clustered important tokens for each attention head.

FullKV (FKV) caches all keys and values for each input token in each layer. All methods are compared to the FullKV simultaneously.

# 5.2 Main Results

The evaluation results from LongBench (Bai et al., 2023) are shown in Table 1 and Figure 3. In Figure 3, we report the average score across datasets for 64, 96, 128, and 256 case sizes. In Table 1, we report the results for two different KV cache sizes with 64 and 2048. These two sizes represent two distinct operational scenarios—the memory-efficient scenario and the

![](images/1788826eee5d231e9ac55d035724a857c0a1388ffed4d7c92a5abc33aebc9253.jpg)

![](images/5117f4f929afce5340b047d3a9c0996b376447008b765033843e0991b589d51c.jpg)

![](images/625d019ebacda0f4e9c41116e686fcadcd72d7540356efdc3478eb791cff1075.jpg)  
Figure 3: The evaluation results from LongBench (Bai et al., 2023) across 64, 96, 128 and 256 cache sizes at LLaMa-3-8B-Instruct (Left), Mistral-7B-Instruct (Middle) and LLaMa-3- 70B-Instruct (Right). The evaluation metrics are the average score of LongBench across datasets. PyramidKV outperforms H2O (Zhang et al., 2024), SnapKV (Li et al., 2024) and StreamingLLM (Xiao et al., 2023), especially in small KV cache sizes.

Table 1: Performance comparison of PyramidKV (Ours) with SnapKV (SKV), H2O, StreamingLLM (SLM) and FullKV (FKV) on LongBench for LlaMa-3-8B-Instruct, Mistral-7B-Instruct and LlaMa-3-70B-Instruct. PyramidKV generally outperforms other KV Cache compression methods across various KV Cache sizes and LLMs. The performance strengths of PyramidKV are more evident in small KV Cache sizes (i.e. KV Size $= 6 4$ ).   

<table><tr><td rowspan="2">Method</td><td colspan="3">Single-Document QA</td><td colspan="3">Multi-Document QA</td><td colspan="3">Summarization</td><td colspan="3">Few-shot Learning</td><td colspan="2">Synthetic</td><td colspan="2">Code</td><td></td></tr><tr><td>NirrQA</td><td>Qasper</td><td>MF-en</td><td>HoPoiQA</td><td>2WikiMQA</td><td>Musique</td><td>GovReport</td><td>OMSum</td><td>MultiNews</td><td>TREC</td><td>TriviaQA</td><td>SAMGum</td><td>pCount</td><td>pRe</td><td>Lcc</td><td>RB-P</td><td>Avg.</td></tr><tr><td></td><td>18409</td><td>3619</td><td>4559</td><td>9151</td><td>4887</td><td>11214</td><td>8734</td><td>10614</td><td>2113</td><td>5177</td><td>8209</td><td>6258</td><td>11141</td><td>9289</td><td>1235</td><td>4206</td><td></td></tr><tr><td colspan="17">LlaMa-3-8B-Instruct, KV Size = Full</td><td></td></tr><tr><td>FKV</td><td>25.70</td><td>29.75</td><td>41.12</td><td>45.55</td><td>35.87</td><td>22.35</td><td>25.63</td><td>23.03</td><td>26.21</td><td>73.00</td><td>90.56</td><td>41.88</td><td>4.67</td><td>69.25</td><td>58.05</td><td>50.77</td><td>41.46</td></tr><tr><td colspan="17">LlaMa-3-8B-Instruct, KV Size = 64</td><td></td></tr><tr><td>SKV</td><td>19.86</td><td>9.09</td><td>27.89</td><td>37.34</td><td>28.35</td><td>18.17</td><td>15.86</td><td>20.80</td><td>16.41</td><td>38.50</td><td>85.92</td><td>36.32</td><td>5.22</td><td>69.00</td><td>51.78</td><td>48.38</td><td>33.05</td></tr><tr><td>H2O</td><td>20.80</td><td>11.34</td><td>27.03</td><td>37.25</td><td>30.01</td><td>17.94</td><td>18.29</td><td>21.49</td><td>19.13</td><td>38.00</td><td>84.70</td><td>37.76</td><td>5.63</td><td>69.33</td><td>53.44</td><td>50.15</td><td>33.89</td></tr><tr><td>SLM</td><td>17.44</td><td>8.68</td><td>22.25</td><td>35.37</td><td>31.51</td><td>15.97</td><td>15.46</td><td>20.06</td><td>14.64</td><td>38.00</td><td>72.33</td><td>29.10</td><td>5.42</td><td>69.50</td><td>46.14</td><td>45.09</td><td>30.43</td></tr><tr><td>Ours</td><td>21.13</td><td>14.18</td><td>30.26</td><td>35.12</td><td>23.76</td><td>16.17</td><td>18.33</td><td>21.65</td><td>19.23</td><td>58.00</td><td>88.31</td><td>37.07</td><td>5.23</td><td>69.50</td><td>52.61</td><td>45.74</td><td>34.76</td></tr><tr><td colspan="17">LlaMa-3-8B-Instruct, KV Size = 2048</td><td></td></tr><tr><td>SKV</td><td>25.86</td><td>29.55</td><td>41.10</td><td>44.99</td><td>35.80</td><td>21.81</td><td>25.98</td><td>23.40</td><td>26.46</td><td>73.50</td><td>90.56</td><td>41.66</td><td>5.17</td><td>69.25</td><td>56.65</td><td>49.94</td><td>41.35</td></tr><tr><td>SLM</td><td>21.71</td><td>25.78</td><td>38.13</td><td>40.12</td><td>32.01</td><td>16.86</td><td>23.14</td><td>22.64</td><td>26.48</td><td>70.00</td><td>83.22</td><td>31.75</td><td>5.74</td><td>68.50</td><td>53.50</td><td>45.58</td><td>37.82</td></tr><tr><td>H2O</td><td>25.56</td><td>26.85</td><td>39.54</td><td>44.30</td><td>32.92</td><td>21.09</td><td>24.68</td><td>23.01</td><td>26.16</td><td>53.00</td><td>90.56</td><td>41.84</td><td>4.91</td><td>69.25</td><td>56.40</td><td>49.68</td><td>39.35</td></tr><tr><td>Ours</td><td>25.40</td><td>29.71</td><td>40.25</td><td>44.76</td><td>35.32</td><td>21.98</td><td>26.83</td><td>23.30</td><td>26.19</td><td>73.00</td><td>90.56</td><td>42.14</td><td>5.22</td><td>69.25</td><td>58.76</td><td>51.18</td><td>41.49</td></tr><tr><td colspan="17">Mistral-7B-Instruct, KV Size = Full</td><td></td></tr><tr><td>FKV</td><td>26.90</td><td>33.07</td><td>49.20</td><td>43.02</td><td>27.33</td><td>18.78</td><td>32.91</td><td>24.21</td><td>26.99</td><td>71.00</td><td>86.23</td><td>42.65</td><td>2.75</td><td>86.98</td><td>56.96</td><td>54.52</td><td>42.71</td></tr><tr><td colspan="17">Mistral-7B-Instruct, KV Size = 64</td><td></td></tr><tr><td>SKV</td><td>16.94</td><td>17.17</td><td>39.51</td><td>36.87</td><td>22.26</td><td>15.18</td><td>14.75</td><td>20.35</td><td>21.45</td><td>37.50</td><td>84.16</td><td>37.28</td><td>4.50</td><td>61.13</td><td>42.40</td><td>38.44</td><td>30.72</td></tr><tr><td>SLM</td><td>15.01</td><td>13.84</td><td>28.74</td><td>30.97</td><td>24.50</td><td>13.42</td><td>13.25</td><td>19.46</td><td>19.17</td><td>35.50</td><td>76.91</td><td>29.61</td><td>4.67</td><td>27.33</td><td>38.71</td><td>35.29</td><td>25.60</td></tr><tr><td>H2O</td><td>18.19</td><td>19.04</td><td>37.40</td><td>30.18</td><td>22.22</td><td>13.77</td><td>16.60</td><td>21.52</td><td>21.98</td><td>37.00</td><td>81.02</td><td>38.62</td><td>5.00</td><td>66.03</td><td>43.54</td><td>40.46</td><td>30.88</td></tr><tr><td>Ours</td><td>20.91</td><td>20.21</td><td>39.94</td><td>33.57</td><td>22.87</td><td>15.70</td><td>17.31</td><td>21.23</td><td>21.41</td><td>54.00</td><td>81.98</td><td>36.96</td><td>3.58</td><td>60.83</td><td>44.52</td><td>37.99</td><td>32.19</td></tr><tr><td colspan="17">Mistral-7B-Instruct, KV Size = 2048</td><td></td></tr><tr><td>SKV</td><td>25.89</td><td>32.93</td><td>48.56</td><td>42.96</td><td>27.42</td><td>19.02</td><td>26.56</td><td>24.47</td><td>26.69</td><td>70.00</td><td>86.27</td><td>42.57</td><td>5.50</td><td>88.90</td><td>50.42</td><td>46.72</td><td>41.56</td></tr><tr><td>SLM</td><td>20.31</td><td>26.64</td><td>45.72</td><td>35.25</td><td>24.31</td><td>12.20</td><td>27.47</td><td>21.57</td><td>24.51</td><td>68.50</td><td>71.95</td><td>31.19</td><td>5.00</td><td>22.56</td><td>43.38</td><td>37.08</td><td>32.35</td></tr><tr><td>H2O</td><td>25.76</td><td>31.10</td><td>49.03</td><td>40.76</td><td>26.52</td><td>17.07</td><td>24.81</td><td>23.64</td><td>26.60</td><td>55.00</td><td>86.35</td><td>42.48</td><td>5.50</td><td>88.15</td><td>49.93</td><td>46.57</td><td>39.95</td></tr><tr><td>Ours</td><td>25.53</td><td>32.21</td><td>48.97</td><td>42.26</td><td>27.50</td><td>19.36</td><td>26.60</td><td>23.97</td><td>26.73</td><td>71.00</td><td>86.25</td><td>42.94</td><td>4.50</td><td>87.90</td><td>53.12</td><td>47.21</td><td>41.63</td></tr><tr><td colspan="17">LlaMa-3-70B-Instruct, KV Size = Full</td><td></td></tr><tr><td>FKV</td><td>27.75</td><td>46.48</td><td>49.45</td><td>52.04</td><td>54.90</td><td>30.42</td><td>32.37</td><td>22.27</td><td>27.58</td><td>73.50</td><td>92.46</td><td>45.73</td><td>12.50</td><td>72.50</td><td>40.96</td><td>63.91</td><td>46.55</td></tr><tr><td colspan="17">LlaMa-3-70B-Instruct, KV Size = 64</td><td></td></tr><tr><td>SKV</td><td>23.92</td><td>31.09</td><td>36.54</td><td>46.66</td><td>50.40</td><td>25.30</td><td>18.05</td><td>21.11</td><td>19.79</td><td>41.50</td><td>91.06</td><td>40.26</td><td>12.00</td><td>72.50</td><td>43.33</td><td>57.62</td><td>39.45</td></tr><tr><td>SLM</td><td>22.07</td><td>23.53</td><td>27.31</td><td>43.21</td><td>51.66</td><td>23.85</td><td>16.62</td><td>19.74</td><td>15.20</td><td>39.50</td><td>76.89</td><td>33.06</td><td>12.00</td><td>72.50</td><td>40.23</td><td>50.20</td><td>35.47</td></tr><tr><td>H2O</td><td>25.45</td><td>34.64</td><td>33.23</td><td>48.25</td><td>50.30</td><td>24.88</td><td>20.03</td><td>21.50</td><td>21.39</td><td>42.00</td><td>90.36</td><td>41.58</td><td>12.00</td><td>71.50</td><td>43.83</td><td>58.16</td><td>39.94</td></tr><tr><td>Ours</td><td>25.47</td><td>36.71</td><td>42.29</td><td>47.08</td><td>46.21</td><td>28.30</td><td>20.60</td><td>21.62</td><td>21.62</td><td>64.50</td><td>89.61</td><td>41.28</td><td>12.50</td><td>72.50</td><td>45.34</td><td>56.50</td><td>42.01</td></tr><tr><td colspan="17">LlaMa-3-70B-Instruct, KV Size = 2048</td><td></td></tr><tr><td>SKV</td><td>26.73</td><td>45.18</td><td>47.91</td><td>52.00</td><td>55.24</td><td>30.48</td><td>28.76</td><td>22.35</td><td>27.31</td><td>72.50</td><td>92.38</td><td>45.58</td><td>12.00</td><td>72.50</td><td>41.52</td><td>69.27</td><td>46.36</td></tr><tr><td>SLM</td><td>26.69</td><td>41.01</td><td>35.97</td><td>46.55</td><td>52.98</td><td>25.71</td><td>27.81</td><td>20.81</td><td>27.16</td><td>69.00</td><td>91.55</td><td>44.02</td><td>12.00</td><td>72.00</td><td>41.44</td><td>68.73</td><td>43.96</td></tr><tr><td>H2O</td><td>27.67</td><td>46.51</td><td>49.54</td><td>51.49</td><td>53.85</td><td>29.97</td><td>28.57</td><td>22.79</td><td>27.53</td><td>59.00</td><td>92.63</td><td>45.94</td><td>12.00</td><td>72.50</td><td>41.39</td><td>63.90</td><td>45.33</td></tr><tr><td>Ours</td><td>27.22</td><td>46.19</td><td>48.72</td><td>51.62</td><td>54.56</td><td>31.11</td><td>29.76</td><td>22.50</td><td>27.27</td><td>73.50</td><td>91.88</td><td>45.47</td><td>12.00</td><td>72.50</td><td>41.36</td><td>69.12</td><td>46.55</td></tr></table>

performance-preserving scenario, respectively for a trade-off between memory and model performance. In Appendix N, we report results of KV cache sizes with 64, 96, 128 and 2048.

Overall, PyramidKV preserves the performance with only $1 2 \%$ of the KV cache and it consistently surpasses other method across a range of KV cache sizes and different backbone models, with its performance advantages becoming particularly pronounced in memory-

![](images/84057c10c3af072b4dde8846929927c23372778bc1e009482b44eb7f24aeba69.jpg)  
LLaMA-3-70B - 8K Context Size

![](images/878055456515e0ad8d33008db1cc4fb9833674e137817c038b7b09b92f6962a1.jpg)  
(a) FullkV,KV ize $=$ Full, acc 100.0

![](images/fbe3bd05204018359a6912b25b0d58f790b03ba9f56ab2fd45ed4f7f2dbac0b9.jpg)  
(b) PyramidKV,KV Size=128,acc 100.0

![](images/2ebaef2395642f58c250997488af24a13670444d600ab1f9d270785d7ee312cc.jpg)  
(c) SnapKV,KV Size=128,acc 98.6   
(d) H2O,KV Size=128,acc 82.3   
Figure 4: Results of the Fact Retrieval Across Context Lengths (“Needle In A HayStack”) test in LlaMa-3-70B-Instruct with 8k context size in ${ \bf 1 2 8  K V }$ cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the length.

constrained environments where only about $0 . 8 \%$ of the KV cache from the prompt is retained. Upon examining specific tasks, PyramidKV demonstrates a notably superior performance on the TREC task, a few-shot question answering challenge. This suggests that the model effectively aggregates information from the few-shot examples, highlighting the potential for further investigation into in-context learning tasks.

Notably, we initially observe the pyramidal attention patterns from the visualization analysis on the multi-document QA task (Figure 2), but the pyramid heuristic has demonstrated its effectiveness on a range of other LongBench tasks (e.g., single-document QA, In-Context Learning), suggesting its promising generalizability beyond multi-document QA.

The performance advantage of PyramidKV increases as the KV cache memory decreases. By focusing on optimizing budget allocation across layers, PyramidKV accurately allocates resources in memory-constrained scenarios, ensuring that retained information is effectively preserved to maintain model performance. Moreover, as in long bench results shown in

Table 1, even in the performance-preserving scenario (i.e., KV cache size $= 2 0 4 8$ ), PyramidKV improves the performance over baseline methods and even outperforms FullKV.

Among the 16 datasets, the tasks where our proposed method performs slightly worse than the baseline are mostly saturated (e.g., HotpotQA, Musique, etc under the LlaMa-3- 8B-Instruct setting with KV Size = 64, as shown in Table 1). In these cases, our method is only marginally inferior to the baseline and remains competitive. Conversely, on tasks with greater potential for improvement (e.g., Qasper, MF-en, TREC, TriviaQA, etc under the same setting), our method significantly outperforms the baseline. Consequently, the overall average performance of our method surpasses that of the baselines. Notably, these tasks include several In-Context Learning tasks (i.e., TREC), our method enjoys best performance gain at In-Context Learning tasks.

# 5.3 Discussion and Insights

# 5.3.1 PyramidKV Preserves the Long-Context Understanding Ability

We conduct the "Fact Retrieval Across Context Lengths" (Needle In A Haystack) experiment (Liu et al., 2023a; Fu et al., 2024), which is a dataset designed to test whether a model can find key information in long input sequences, to evaluate the in-context retrieval capabilities of LLMs when utilizing various KV cache compression methods. For this purpose, we employ LlaMa-3-70B-Instruct as our base, with context lengths extending up to 8k. We compared several KV cache compression techniques (PyramidKV, SnapKV (Li et al., 2024), and H2O (Zhang et al., 2024)) at cache sizes of 128 and full cache. The results, presented in Figure 4 1. The results demonstrate that with only 128 KV cache retained, PyramidKV effectively maintains the model’s ability to understand short contexts, and shows only modest degradation for longer contexts. In contrast, other KV cache compression methods significantly hinder the performance of LLMs. Notably, for the larger model (LlaMa-3-70B-Instruct), PyramidKV achieves 100.0 Acc. performance, matching the results of FullKV, thereby demonstrating its ability to preserve long-context comprehension with a substantially reduced KV cache. We adopt the haystack setting of haystack formed from a long corpus for the Needle In A Haystack task as Wu et al. (2024).

# 5.3.2 PyramidKV Significantly Reduces Memory with Limited Performance Drop

In this section, we study how sensitive the methods are with different sizes of KV cache. We report the KV cache memory reduction in Table 2. We evaluate the memory consumption of LLaMa-3-8B-Instruct. Specifically, we evaluate the memory consumption of all methods with a fixed batch size of 1, a sequence length of 8192, and model weights in fp16 format. We observe that PyramidKV substantially reduces the KV cache memory across different numbers of cache sizes. We also present that the allocation strategy and score-based selection add minimal complexity in the inference phase as Appendix L.

Table 2: Memory reduction effect and benchmark result by using PyramidKV. We conducted a comparison of memory consumption between the Llama-3-8B-Instruct model utilizing the Full KV cache and the Llama-3-8B-Instruct model compressed with the PyramidKV.   

<table><tr><td>cache size</td><td>Memory</td><td>Compression Ratio</td><td>QMSum</td><td>TREC</td><td>TriviaQA</td><td>PCount</td><td>PRe</td><td>Lcc</td></tr><tr><td>512</td><td>428M</td><td>6.3%</td><td>22.80</td><td>71.50</td><td>90.61</td><td>5.91</td><td>69.50</td><td>58.16</td></tr><tr><td>1024</td><td>856M</td><td>12.5%</td><td>22.55</td><td>71.50</td><td>90.61</td><td>5.91</td><td>69.50</td><td>58.16</td></tr><tr><td>2048</td><td>1712M</td><td>25.0%</td><td>22.55</td><td>72.00</td><td>90.56</td><td>5.58</td><td>69.25</td><td>56.79</td></tr><tr><td>Full</td><td>6848M</td><td>100.0%</td><td>23.30</td><td>73.00</td><td>90.56</td><td>5.22</td><td>69.25</td><td>58.76</td></tr></table>

# 6 Conclusion

In this study, we investigate Pyramidal Information Funneling, the intrinsic attention patterns of Large Language Models (LLMs) when processing long context inputs. Motivated by this discovery, we design a novel KV cache compression approach PyramidKV that utilizes this information flow pattern. Our method excels in memory-constrained settings, preserves long-context understanding ability, and significantly reduces memory usage.

# References

•Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508, 2023.   
Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. arXiv preprint arXiv:2403.06764, 2024a.   
Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia. Longlora: Efficient fine-tuning of long-context large language models. arXiv preprint arXiv:2309.12307, 2023.   
Zhuoming Chen, Ranajoy Sadhukhan, Zihao Ye, Yang Zhou, Jianyu Zhang, Niklas Nolte, Yuandong Tian, Matthijs Douze, Leon Bottou, Zhihao Jia, and Beidi Chen. Magicpig: Lsh sampling for efficient llm generation, 2024b. URL https://arxiv.org/abs/2410.16179.   
Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with $9 0 \% ^ { * }$ chatgpt quality, March 2023. URL https://lmsys.org/blog/2023-03-30-vicuna/.   
Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4599–4610, 2021.   
Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, and Mao Yang. Longrope: Extending llm context window beyond 2 million tokens. arXiv preprint arXiv:2402.13753, 2024.   
Harry Dong, Xinyu Yang, Zhenyu Zhang, Zhangyang Wang, Yuejie Chi, and Beidi Chen. Get more with less: Synthesizing recurrence with kv cache compression for efficient llm inference. arXiv preprint arXiv:2402.09398, 2024.   
Alexander Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir Radev. Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. In Anna Korhonen, David Traum, and Lluís Màrquez, editors, Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1074–1084, Florence, Italy, July 2019a. Association for Computational Linguistics. doi: 10.18653/v1/P19-1102. URL https://aclanthology.org/P19-1102.   
Alexander Richard Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir Radev. Multi-news: A large-scale multi-document summarization dataset and abstractive hierarchical model. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 1074–1084, 2019b.   
Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, and Hao Peng. Data engineering for scaling language models to 128k context. arXiv preprint arXiv:2402.10171, 2024.

Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive kv cache compression for llms. arXiv preprint arXiv:2310.01801, 2023.   
Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. Samsum corpus: A human-annotated dialogue dataset for abstractive summarization. EMNLP-IJCNLP 2019, page 70, 2019.   
Daya Guo, Canwen Xu, Nan Duan, Jian Yin, and Julian McAuley. Longcoder: A long-range pre-trained language model for code completion. arXiv preprint arXiv:2306.14893, 2023.   
Chi Han, Qifan Wang, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. Lm-infinite: Simple on-the-fly length generalization for large language models. arXiv preprint arXiv:2308.16137, 2023.   
Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, 2020.   
Luyang Huang, Shuyang Cao, Nikolaus Parulian, Heng Ji, and Lu Wang. Efficient attentions for long document summarization. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1419–1436, 2021.   
Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.   
Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. arXiv preprint arXiv:2407.02490, 2024.   
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, 2017.   
Tomáš Koˇcisky, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gábor ` Melis, and Edward Grefenstette. The narrativeqa reading comprehension challenge. Transactions of the Association for Computational Linguistics, 6:317–328, 2018.   
Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. Infinigen: Efficient generative inference of large language models with dynamic kv cache management, 2024. URL https://arxiv.org/abs/2406.19707.   
Xin Li and Dan Roth. Learning question classifiers. In COLING 2002: The 19th International Conference on Computational Linguistics, 2002.   
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation. arXiv preprint arXiv:2404.14469, 2024.   
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts, 2023a.   
Tianyang Liu, Canwen Xu, and Julian McAuley. Repobench: Benchmarking repository-level code auto-completion systems, 2023b.   
Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.

Mingjie Sun, Xinlei Chen, J Zico Kolter, and Zhuang Liu. Massive activations in large language models. arXiv preprint arXiv:2402.17762, 2024.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.   
Daniel Waddington, Juan Colmenares, Jilong Kuang, and Fengguang Song. Kv-cache: A scalable high-performance web-object cache for manycore. In 2013 IEEE/ACM 6th International Conference on Utility and Cloud Computing, pages 123–130. IEEE, 2013.   
Lean Wang, Lei Li, Damai Dai, Deli Chen, Hao Zhou, Fandong Meng, Jie Zhou, and Xu Sun. Label words are anchors: An information flow perspective for understanding in-context learning. arXiv preprint arXiv:2305.14160, 2023.   
Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality. arXiv preprint arXiv:2404.15574, 2024.   
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.   
Dongjie Yang, XiaoDong Han, Yan Gao, Yao Hu, Shilin Zhang, and Hai Zhao. Pyramidinfer: Pyramid kv cache compression for high-throughput llm inference. arXiv preprint arXiv:2405.12532, 2024.   
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, 2018.   
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36, 2024.   
Ming Zhong, Da Yin, Tao Yu, Ahmad Zaidi, Mutethia Mutuma, Rahul Jha, Ahmed Hassan, Asli Celikyilmaz, Yang Liu, Xipeng Qiu, et al. Qmsum: A new benchmark for querybased multi-domain meeting summarization. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 5905–5921, 2021.   
Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li. Pose: Efficient context window extension of llms via positional skip-wise training. arXiv preprint arXiv:2309.10400, 2023.

# A Limitations

Our experiments were limited to three base models: LLAMA-3-8B-Instruct, LLAMA-3-70B-Instruct and Mistral-7B-Instruct. While these models demonstrated consistent trends, the robustness of our findings could be enhanced by testing a broader array of model families, should resources permit. Additionally, our research was conducted exclusively in English, with no investigations into how these findings might be transferred to other languages. Expanding the linguistic scope of our experiments could provide a more comprehensive understanding of the applicability of our results globally. Based on our results at LongBench and Needle-in-a-HayStack experiment, PyramidKV generally works decently in most of the language tasks (i.e., Single-Document QA, Multi-Document QA, Summerization, Few-Shot In-Context Learning, etc.). Although we observe that PyramidKV performs better in some tasks (i.e., Few-Shot In-Context Learning) compared with some other tasks (i.e., Summerization), we have not observed cases that the decoding result collapses at some tasks. This remains a new topic for future work to explore.

# B Future Work

Our investigation on PyramidKV highlights considerable opportunities for optimizing KV cache compression by adjusting the number of KV caches retained according to the distinct attention patterns of each layer (or even for each head). For instance, the retention of KV cache for each layer could be dynamically modified based on real-time analysis of the attention matrices, ensuring that the compression strategy is consistently aligned with the changing attention dynamics within LLMs. Furthermore, our experiments indicate that PyramidKV significantly surpasses other methods in few-shot learning tasks, suggesting promising applications of KV cache in in-context learning. This approach could potentially enable the use of more shots within constrained memory limits.

![](images/88b574b48d9d3698d9ab3eb404bcf0d3459519e88b1ef1e566e597b2850077d3.jpg)  
Figure 5: Attention patterns of retrieval-augmented generation across layers in Mistral-7B-Instruct model (Jiang et al., 2023)

![](images/cfad8878bcfa42e2bf3fc021d00c0d5459c3793a7592038d345a5c668439ed6e.jpg)  
Figure 6: Attention patterns of retrieval-augmented generation across layers in Mixtral-8x7B-Instruct Mixture-of-Experts model.

# C Related Work

Interpretation of LLMs Prior research has shown that attention matrices in LLMs are typically sparse (Chen et al., 2024a; Xiao et al., 2023; Zhang et al., 2024), focusing disproportionately on a few tokens. For instance, Xiao et al. (2023) identified an “attention sink” phenomenon, where maintaining the Key and Value (KV) states of the first few tokens can substantially restore the performance of windowed attention, despite these tokens not being semantically crucial. Similarly, Sun et al. (2024) identified a “massive activations” pattern, where a minority of activations show significantly larger values than others within LLMs. Interestingly, these values remain relatively constant across different inputs and act as critical bias terms in the model.

Further explorations in this field reveal distinct patterns across various attention heads and layers. Li et al. (2024) observed that certain attention heads consistently target specific prompt attention features during decoding. Additionally, Wang et al. (2023) discovered that in In-Context Learning scenarios, label words in demonstration examples serve as semantic anchors. In the lower layers of an LLM, shallow semantic information coalesces around these label words, which subsequently guide the LLMs’ final output predictions by serving as reference points. Recently, Wu et al. (2024) revealed that a special type of attention head, the so-called retrieval head, is largely responsible for retrieving information. Inspired by these findings that the attention mechanism exhibits varying behaviors across different layers, we discovered that “Massive Activation” does not consistently manifest across all layers in long context sequences; instead, it predominantly occurs in the upper layers. Additionally, we identified a novel trend of information aggregation specific to long-context inputs, which will be further explained in $\ S 3$ .

KV Cache Compression There has been a growing interest in addressing LLMs’ memory constraints on processing long context inputs. FastGen (Ge et al., 2023) introduces an adaptive KV cache management strategy that optimizes memory use by tailoring retention tactics to the specific nature of attention heads. This method involves evicting long-range contexts from heads that prioritize local interactions, discarding non-special tokens from heads focused on special tokens, and maintaining a standard KV cache for heads that engage

broadly across tokens. SnapKV (Li et al., 2024) improves efficiency by compressing KV caches via selecting/clustering significant KV positions based on their attention scores. Heavy Hitter Oracle (H2O) (Zhang et al., 2024) implements a dynamic eviction policy that effectively balances the retention of recent and historically significant tokens, optimizing memory usage while preserving essential information. StreamingLLM (Xiao et al., 2023) enables LLMs trained on finite attention windows to handle infinite sequence lengths without fine-tuning, thus expanding the models’ applicability to broader contexts. LM-Infinite (Han et al., 2023) allows LLMs pre-trained with 2K or 4K-long segments to generalize to up to 200M length inputs while retaining perplexity without parameter updates.

While these approaches have significantly advanced the efficient management of memory for LLMs, they generally apply a fixed KV cache size across all layers. In contrast, our investigations into the attention mechanisms across different layers of LLMs reveal that the attention patterns vary from layer to layer, making a one-size-fits-all approach to KV cache management suboptimal. In response to this inefficiency, we propose a novel KV cache compression method, called PyramidKV that allocates different KV cache budgets across different layers, tailored to the unique demands and operational logic of each layer’s attention mechanism. This layer-specific strategy takes a significant step toward balancing both memory efficiency and model performance, addressing a key limitation in existing methodologies.

# D Pyramidal Information Funneling

Figure 5 and Figure 6 shows the attention patterns of one QA example over six different layers (i.e., 0, 6, 12, 18, 24, and 30) for Mistral-7B-Instruct model and Mixtral-8x7B-Instruct Mixture-of-Experts model. Figure 5 and Figure 6 demonstrate that the Pyramidal Information Funneling phenomenon is also evident in both the Mistral model and Mixtral model . The results reveal that, akin to Llama-like models, Mistral exhibit a progressively narrowing attention focus across layers. This supports the universality of the Pyramidal Information Funneling phenomenon across diverse model families. We hope this addresses your concern and underscores the generalizability of our findings.

Our analysis uniquely examines attention metrics across all transformer layers, from 0 to 30, leading to the discovery of a key phenomenon we term Pyramidal Information Funneling.

Lee et al. (2024) conducted a limited investigation into attention patterns, focusing only on the lower layer (layer 0) and a single upper layer (layer 18). While Lee et al. (2024) noted that attention becomes more skewed in upper layers, it did not provide a fine-grained observation of attention patterns across all layers. In contrast, our study reveals several novel findings:

• Localized Attention: We observe that attention progressively narrows its focus, targeting specific components within the input sequence.   
• Massive Attention Mechanism: In the upper layers, attention heavily concentrates on a small set of critical tokens. Notably, these tokens are not limited to the leading positions, as observed in Lee et al. (2024), but also appear at regular intervals across the sequence. The discrepancy arises from differences in input settings, with Lee et al. (2024) identifying massive attention only at the initial tokens.

These insights motivated us to propose a token-selection method based on the highest attention scores in the upper layers, rather than solely relying on tokens from earlier positions.

To the best of our knowledge, Chen et al. (2024b) has not analyzed attention patterns across transformer layers.

Therefore, although Lee et al. (2024) and Chen et al. (2024b) are considered contemporaneous with our work, making a comparison unnecessary, the perspective of our observation is considered novel compared with Lee et al. (2024) and Chen et al. (2024b). Moreover, although Lee et al. (2024) also observed attention patterns, the method we proposed based

![](images/468a0ba1e5f1cda592e66f7b67aa343b3d9c0c21a5913d39c8f36053f77e5c00.jpg)  
Figure 7: Illustration of PyramidKV. At the lower level of the transformer, the PyramidKV selects more keys and values based on the exhibited average attention pattern. Fewer keys and values at the higher level are selected based on the massive activation pattern, where we observe that attention scores are concentrated over local regions.

on our observations is significantly different from Lee et al. (2024), further highlighting the novelty of our work.

# E Details of Proposed Method

Based on the pyramidal information funneling observed across different layers, PyramidKV consists of two steps: (1) Dynamically allocating different KV cache sizes/budgets across different layers; and (2) Selecting important KV vectors in each attention head for caching as Figure 7.

Our decision to use an arithmetic sequence is driven by three key factors:

• Alignment with Pyramidal Information Funneling Pattern: Empirical observations reveal a pyramidal information funneling pattern, where lower layers exhibit dispersed attention while higher layers concentrate on fewer tokens. Inspired by this, we adopt the arithmetic sequence design to align with this natural progression.   
• Superior Empirical Performance: Through extensive experimentation across diverse datasets, we compared various methods, including the arithmetic sequence and adaptive approaches. Results consistently showed that the arithmetic sequence method outperformed others.   
• Computational Efficiency: The arithmetic sequence method introduces minimal computational overhead compared to adaptive approaches, which require dynamically computing cache budgets across layers.

To perform KV cache eviction, we use torch.gather. Below, we outline the memory allocation and release process of torch.gather:

• Index Selection: Identify the positions of the elements to extract from the input tensor.   
• Memory Location Calculation: Compute the specific memory locations of the elements to be extracted using the strides of the input tensor across each dimension.

• Output Tensor Creation: Allocate memory to create a new output tensor and copy the selected elements to their corresponding positions in the output tensor.   
• Memory Management: Since torch.gather is not an in-place operation, it creates a new tensor to store the results, while the memory of the original input tensor is released.

The speed-up offered by PyramidKV is complementary to that achieved through tensor parallelism and pipeline parallelism, as these approaches are not mutually exclusive. PyramidKV can be seamlessly integrated with both tensor parallelism and pipeline parallelism.

# F Details of Evaluation

We use LongBench (Bai et al., 2023) to assess the performance of PyramidKV on tasks involving long-context inputs. LongBench is a meticulously designed benchmark suite that tests the capabilities of language models in handling extended documents and complex information sequences. This benchmark was created for multi-task evaluation of long context inputs.

We present the details of metrics, language and data for LongBench at Table 3.

We run all the experiments on NVIDIA A100.

Table 3: An overview of the dataset statistics in LongBench (Bai et al., 2023). ‘Source’ denotes the origin of the context. ‘Accuracy (CLS)’ refers to classification accuracy, while ‘Accuracy (EM)’ refers to exact match accuracy.   

<table><tr><td>Dataset</td><td>Source</td><td>Avg len</td><td>Metric</td><td>Language</td><td>#data</td></tr><tr><td>Single-Document QA</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NarrativeQA</td><td>Literature, Film</td><td>18,409</td><td>F1</td><td>English</td><td>200</td></tr><tr><td>Qasper</td><td>Science</td><td>3,619</td><td>F1</td><td>English</td><td>200</td></tr><tr><td>MultiFieldQA-en</td><td>Multi-field</td><td>4,559</td><td>F1</td><td>English</td><td>150</td></tr><tr><td>Multi-Document QA</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HotpotQA</td><td>Wikipedia</td><td>9,151</td><td>F1</td><td>English</td><td>200</td></tr><tr><td>2WikiMultihopQA</td><td>Wikipedia</td><td>4,887</td><td>F1</td><td>English</td><td>200</td></tr><tr><td>MuSiQue</td><td>Wikipedia</td><td>11,214</td><td>F1</td><td>English</td><td>200</td></tr><tr><td>Summarization</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GovReport</td><td>Government report</td><td>8,734</td><td>Rouge-L</td><td>English</td><td>200</td></tr><tr><td>QMSum</td><td>Meeting</td><td>10,614</td><td>Rouge-L</td><td>English</td><td>200</td></tr><tr><td>MultiNews</td><td>News</td><td>2,113</td><td>Rouge-L</td><td>English</td><td>200</td></tr><tr><td>Few-shot Learning</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TREC</td><td>Web question</td><td>5,177</td><td>Accuracy (CLS)</td><td>English</td><td>200</td></tr><tr><td>TriviaQA</td><td>Wikipedia, Web</td><td>8,209</td><td>F1</td><td>English</td><td>200</td></tr><tr><td>SAMSum</td><td>Dialogue</td><td>6,258</td><td>Rouge-L</td><td>English</td><td>200</td></tr><tr><td>Synthetic Task</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PassageCount</td><td>Wikipedia</td><td>11,141</td><td>Accuracy (EM)</td><td>English</td><td>200</td></tr><tr><td>PassageRetrieval-en</td><td>Wikipedia</td><td>9,289</td><td>Accuracy (EM)</td><td>English</td><td>200</td></tr><tr><td>Code Completion</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LCC</td><td>Github</td><td>1,235</td><td>Edit Sim</td><td>Python/C#/Java</td><td>500</td></tr><tr><td>RepoBench-P</td><td>Github repository</td><td>4,206</td><td>Edit Sim</td><td>Python/Java</td><td>500</td></tr></table>

# G License

LongBench: MIT

# H Handle Rotary Embedding after Tokens are Removed in PyramidKV

We keep the rotary embedding unchanged after tokens are removed, so that LLMs can still capture the exact position information even if the tokens are removed. StreamingLLM (Xiao et al., 2023) shows that rolling kv cache with the correct relative position is crucial for maintaining performance. This is because StreamingLLM is designed to mainly handle unlimited context sizes, where contexts exceed the LLM’s fixed context length. Without changing the rotary embedding after token removal, LLMs would receive rotary embedding of a non-monotonic position sequence. For example, after the first KV cache compression, LLMs might receive the input position embedding as $[ 0 , 1 , 2 , 3 , 3 0 9 6 , 3 0 9 7 , \cdot \cdot \cdot , 4 { \hat { 0 } } 9 6 ] ,$ , and the position embedding of the generated sequences could be $[ 1 0 0 5 , 1 0 0 6 , 1 0 0 7 , \cdot \cdot \cdot ]$ . The position sequence of [0, 1, 2, 3, 3096, . . . , 4096, 1005, 1006, 1007, · · · ] is a non-monotonic sequence, which may negatively hurts the performance. In contrast, our targeting settings will not process unlimited context size. For example, given a input sequence of 4012 length, after KV cache compression, the position sequence would be [0, 4, 6, 16, · · · , 3927, 3987, 4012], and the position sequence of the generated tokens would be $[ 4 0 1 3 , 4 0 1 4 , \cdot \cdot \cdot ]$ . By keeping the rotary embedding unchanged after the tokens are removed, the LLM avoids non-monotonic position sequences, and the LLM can capture the exact position information even if the tokens are shifted. Our preliminary results show that rolling KV cache with the correct relative position will slightly decrease the performance.

# I Ablation Study

In this section, we present an ablation study for hyperparameters and allocation strategies.

Based on our observations of the attention pattern, we find that a relatively stable, linear arithmetic decrease aligns more closely with the underlying structure of the pattern. We conduct experiments comparing various allocation strategies.

We conducted hyperparameter testing on the original development sets of 16 datasets in LongBench. The parameter $\beta$ demonstrated remarkable stability, showing minimal sensitivity to varying hyperparameter settings, which highlights its robustness. Conversely, α consistently produced superior results when set to 8 or 16. Consequently, these values were adopted for subsequent experiments. In Appendix H.2 and H.3, we further analyzed the impact of hyperparameter selection on KV cache budget allocation across different layers. The experiments reaffirmed that $\beta$ had negligible influence on the outcomes, underscoring its stability. Meanwhile, α continued to deliver optimal results at values of 8 and 16.

# I.1 Allocation Srategies

Based on our observations of the attention pattern, we find that a relatively stable, linear arithmetic decrease aligns more closely with the underlying structure of the pattern.

We conduct experiments comparing various pyramidal allocation strategies (i.e., linear decay strategy, geometric decay strategy and exponential decay strategy) with a cache size of 64 as Table 4 to confirm that a linear strategy is indeed optimal or preferable.

We also propose three adaptive allocation baselines, which are based on the entropy, Gini coefficient, and sparsity of the attention values at each layer. The weight of each layer is calculated based on its corresponding metric (entropy, Gini coefficient, or sparsity), and the budget is allocated accordingly. Specifically:

• Entropy-based allocation: Layers with higher entropy receive higher weights. Each layer’s entropy is calculated based on the the layer’s attention.   
• Gini coefficient-based allocation: Layers with higher Gini coefficients receive higher weights. Each layer’s Gini coefficient is calculated based on the the layer’s attention

The empirical results as Table 4 consistently showed that the linear strategy outperformed its counterparts, establishing it as the most effective approach for our use case. The experiment strengthens the rationale for choosing the specific allocation method.

Table 4: Ablation study of allocation strategies.   

<table><tr><td rowspan="2">Stra.</td><td colspan="3">Single-Document QA</td><td colspan="3">Multi-Document QA</td><td colspan="3">Summarization</td><td colspan="3">Few-shot Learning</td><td colspan="2">Synthetic</td><td>Code</td><td>Avg.</td><td></td></tr><tr><td>NrVQA</td><td>Qasper</td><td>MF-en</td><td>HotpotQA</td><td>2WikMQA</td><td>Musique</td><td>GovReport</td><td>OMSum</td><td>MultiNews</td><td>TREC</td><td>TriviaQA</td><td>SAMSum</td><td>PCount</td><td>PRe</td><td>Lcc</td><td>RB-P</td><td></td></tr><tr><td>Geo.</td><td>20.51</td><td>15.04</td><td>29.4</td><td>34.93</td><td>26.41</td><td>16.6</td><td>18.32</td><td>21.68</td><td>18.81</td><td>52</td><td>87.51</td><td>36.15</td><td>5.18</td><td>69.17</td><td>53.11</td><td>44.91</td><td>34.36</td></tr><tr><td>Exp.</td><td>20.58</td><td>14.82</td><td>28.74</td><td>34.34</td><td>26.24</td><td>16.11</td><td>18.41</td><td>21.63</td><td>18.75</td><td>52.00</td><td>87.94</td><td>36.26</td><td>5.19</td><td>69.17</td><td>54.34</td><td>43.21</td><td>34.23</td></tr><tr><td>Lin.</td><td>21.13</td><td>14.18</td><td>30.26</td><td>35.12</td><td>23.76</td><td>16.17</td><td>18.33</td><td>21.65</td><td>19.23</td><td>58.00</td><td>88.31</td><td>37.07</td><td>5.23</td><td>69.50</td><td>52.61</td><td>45.74</td><td>34.76</td></tr><tr><td>Entropy.</td><td>18.12</td><td>14.12</td><td>27.22</td><td>33.21</td><td>21.16</td><td>15.16</td><td>17.76</td><td>19.87</td><td>17.09</td><td>51</td><td>87.31</td><td>34.29</td><td>5.09</td><td>68.91</td><td>50.12</td><td>42.98</td><td>32.71</td></tr><tr><td>Gini.</td><td>17.92</td><td>14.61</td><td>28.21</td><td>32.67</td><td>19.98</td><td>15.98</td><td>16.20</td><td>19.29</td><td>18.21</td><td>51.00</td><td>86.21</td><td>34.97</td><td>5.11</td><td>65.51</td><td>51.98</td><td>43.37</td><td>32.58</td></tr></table>

# I.2 Hyper Parameter α

We present the study of $\alpha$ for LlaMa-3-8B-Instruct in 128 KV cache size budget at Table 5.We find that a small alpha value (i.e., 8, 16) leads to better performance than a larger alpha value (i.e., 24, 32, 40, 48).

Table 5: Ablation on α.   

<table><tr><td rowspan="2">α</td><td colspan="3">Single-Document QA</td><td colspan="3">Multi-Document QA</td><td colspan="3">Summarization</td><td colspan="3">Few-shot Learning</td><td colspan="2">Synthetic</td><td>Code</td><td>Avg.</td><td></td></tr><tr><td>NvQA</td><td>Qesper</td><td>MF-en</td><td>HoPoiQA</td><td>2WikiMQA</td><td>Museique</td><td>GovReport</td><td>OMsum</td><td>MultiNews</td><td>TREC</td><td>TriviaQA</td><td>SAMsum</td><td>pCount</td><td>pRe</td><td>Lcc</td><td>RB-p</td><td></td></tr><tr><td>8</td><td>21.40</td><td>16.92</td><td>31.62</td><td>38.45</td><td>28.72</td><td>18.59</td><td>19.96</td><td>22.49</td><td>20.96</td><td>66.50</td><td>89.35</td><td>38.43</td><td>5.92</td><td>69.00</td><td>57.86</td><td>51.80</td><td>37.37</td></tr><tr><td>16</td><td>23.37</td><td>16.21</td><td>33.93</td><td>38.24</td><td>27.28</td><td>20.57</td><td>19.71</td><td>21.93</td><td>20.86</td><td>60.00</td><td>88.75</td><td>38.34</td><td>5.48</td><td>69.12</td><td>57.84</td><td>53.42</td><td>37.19</td></tr><tr><td>24</td><td>22.85</td><td>14.51</td><td>32.26</td><td>38.38</td><td>28.36</td><td>20.33</td><td>19.55</td><td>21.72</td><td>20.72</td><td>54.50</td><td>88.71</td><td>38.46</td><td>5.48</td><td>69.50</td><td>56.83</td><td>53.65</td><td>36.61</td></tr><tr><td>32</td><td>23.01</td><td>14.54</td><td>31.68</td><td>38.86</td><td>29.90</td><td>19.16</td><td>19.20</td><td>21.83</td><td>20.52</td><td>49.50</td><td>87.01</td><td>38.01</td><td>5.75</td><td>69.50</td><td>57.02</td><td>54.54</td><td>36.25</td></tr><tr><td>40</td><td>21.70</td><td>13.06</td><td>30.14</td><td>36.78</td><td>27.34</td><td>18.88</td><td>18.72</td><td>21.37</td><td>19.79</td><td>44.00</td><td>87.74</td><td>38.43</td><td>6.08</td><td>69.25</td><td>56.11</td><td>53.89</td><td>35.21</td></tr><tr><td>48</td><td>21.51</td><td>12.30</td><td>29.77</td><td>39.04</td><td>26.76</td><td>17.97</td><td>18.65</td><td>21.20</td><td>20.29</td><td>44.50</td><td>87.73</td><td>38.44</td><td>5.51</td><td>69.25</td><td>56.73</td><td>53.88</td><td>35.22</td></tr></table>

# I.3 Hyper Parameter $\beta$

One topic we want to analyze for our ablation study is the selection of $\beta _ { \cdot }$ , which can determine the staircase. The smaller $\beta$ is, the gentler the staircase is; the larger $\beta$ is, the steeper the staircase is. We want to investigate the effect of $\beta$ step size on the final result. Results on $1 2 8 \mathrm { K V }$ cache size and LlaMa-3-8B-Instruct are shown in Table 6. The results at Table 6 show that using a relatively small value of $\beta$ yields better outcomes, and PyramidKV is generally robust to the selection of $\beta$ .

Table 6: Ablation on $\beta$ .   

<table><tr><td rowspan="2">β</td><td colspan="3">Single-Document QA</td><td colspan="3">Multi-Document QA</td><td colspan="3">Summarization</td><td colspan="3">Few-shot Learning</td><td colspan="2">Synthetic</td><td>Code</td><td>Avg.</td><td></td></tr><tr><td>NwQA</td><td>Qasper</td><td>ME-en</td><td>HotpotQA</td><td>2WikiMQA</td><td>MusicJaque</td><td>GovReport</td><td>OMsum</td><td>MultiNews</td><td>TREC</td><td>TriviaQA</td><td>SAMsum</td><td>pCount</td><td>pRe</td><td>Lcc</td><td>RB-p</td><td></td></tr><tr><td>20</td><td>21.40</td><td>16.92</td><td>33.79</td><td>39.73</td><td>28.72</td><td>18.59</td><td>19.86</td><td>22.48</td><td>20.95</td><td>66.50</td><td>89.35</td><td>38.39</td><td>5.92</td><td>69.00</td><td>56.49</td><td>47.95</td><td>37.25</td></tr><tr><td>18</td><td>21.71</td><td>16.24</td><td>33.59</td><td>39.89</td><td>27.94</td><td>18.38</td><td>19.76</td><td>22.32</td><td>21.20</td><td>66.50</td><td>88.98</td><td>38.93</td><td>5.46</td><td>69.50</td><td>56.47</td><td>49.23</td><td>37.25</td></tr><tr><td>16</td><td>21.74</td><td>14.86</td><td>33.64</td><td>39.18</td><td>28.17</td><td>18.77</td><td>19.57</td><td>22.25</td><td>21.48</td><td>66.50</td><td>89.69</td><td>38.87</td><td>5.82</td><td>69.50</td><td>57.02</td><td>50.11</td><td>37.32</td></tr><tr><td>14</td><td>22.53</td><td>16.31</td><td>33.50</td><td>40.50</td><td>28.15</td><td>19.26</td><td>19.66</td><td>22.39</td><td>21.38</td><td>65.50</td><td>90.02</td><td>38.56</td><td>5.75</td><td>69.50</td><td>57.51</td><td>49.71</td><td>37.51</td></tr></table>

# J Integation with MInference

We would like to clarify that PyramidKV and MInference Jiang et al. (2024) are complementary approaches addressing different aspects of KV cache optimization. Specifically:

• MInference focuses on accelerating the generation of KV caches during the prefilling stage of LLM inference.

• In contrast, PyramidKV targets efficient KV cache management during LLM decoding.

To evaluate their respective strengths, we compared PyramidKV and MInference on Longbench using a KV cache size of 128. The results demonstrated the superior performance of PyramidKV.

Furthermore, we demonstrate that MInference and PyramidKV can be seamlessly integrated to achieve highly efficient inference while maintaining performance comparable to full attention. The results of MInference combined with PyramidKV, evaluated on Longbench with a KV cache size of 128, as PyramidKV $^ +$ MInference hybrid approach.

Table 7: Comparison between PyramidKV, MInference and MInference-PyramidKV hybrid method.   

<table><tr><td rowspan="2">Stra.</td><td colspan="3">Single-Document QA</td><td colspan="3">Multi-Document QA</td><td colspan="3">Summarization</td><td colspan="3">Few-shot Learning</td><td colspan="2">Synthetic</td><td>Code</td><td>Avg.</td></tr><tr><td>NmtQA</td><td>Qasper</td><td>Mf-en</td><td>HotpOQA</td><td>2WikiMQA</td><td>Museque</td><td>GovReport</td><td>QMSum</td><td>MultiNews</td><td>TREC</td><td>TriviaQA</td><td>SAMSum</td><td>PCount</td><td>PRe</td><td>Lcc</td><td>RB-P</td></tr><tr><td>PyramidKV</td><td>23.99</td><td>20.61</td><td>38.28</td><td>43.23</td><td>31.62</td><td>20.94</td><td>21.27</td><td>22.69</td><td>22.83</td><td>71</td><td>90.48</td><td>39.86</td><td>5.83</td><td>69.25</td><td>56.94</td><td>39.31</td></tr><tr><td>MInference</td><td>19.74</td><td>30.63</td><td>40.41</td><td>44.28</td><td>35.22</td><td>20.65</td><td>28.43</td><td>23.35</td><td>26.75</td><td>72.00</td><td>87.90</td><td>42.78</td><td>6.30</td><td>64.00</td><td>58.76</td><td>5.06</td></tr><tr><td>M. + P</td><td>20.04</td><td>31.74</td><td>39.98</td><td>43.10</td><td>35.21</td><td>21.60</td><td>27.41</td><td>23.06</td><td>26.76</td><td>73.00</td><td>88.03</td><td>43.36</td><td>6.28</td><td>64.00</td><td>58.57</td><td>45.42</td></tr></table>

In summary, we demonstrate that PyramidKV outperforms MInference on Longbench. Furthermore, when integrated with MInference, PyramidKV enhances its performance even further.

# K Comparison with PyramidInfer

Our work differs from PyramidInfer in two key aspects:

• Decay Strategy: While PyramidInfer Yang et al. (2024) employs a geometric decay strategy, our method adopts an arithmetic decay strategy. We argue that the relatively stable and linear nature of arithmetic decay better aligns with the behavior of the attention mechanism. This strategy is derived from empirically observed attention patterns, aiming to closely match them. Notably, our approach also achieves superior results, as demonstrated in the experimental results presented in the table below.   
• Token Selection: PyramidInfer discards tokens in earlier layers, preventing them from being reconsidered in later layers. In contrast, our method allows previously discarded tokens to be re-evaluated in higher layers, recognizing that these tokens may still hold relevance at different stages of the model’s processing.   
• Pyramidal Information Funneling Pattern: A key contribution of our work lies in identifying and leveraging the pyramidal information funneling phenomenon within attention mechanisms. Through in-depth analysis, we observe that attention tends to disperse in earlier layers and progressively concentrates on crucial tokens in higher layers. This insight forms the foundation of our arithmetic decay strategy, ensuring that our method aligns more naturally with these intrinsic patterns.

Despite some similarities between the two approaches, these differences lead to significantly distinct outcomes. As shown in Table 8, our method consistently outperforms PyramidInfer, highlighting the effectiveness of our design choices.

Table 8: Comparison between PyramidKV and Pyramidinfer.   

<table><tr><td rowspan="2">Stra.</td><td colspan="3">Single-Document QA</td><td colspan="3">Multi-Document QA</td><td colspan="3">Summarization</td><td colspan="3">Few-shot Learning</td><td colspan="2">Synthetic</td><td colspan="2">Code</td><td>Avg.</td></tr><tr><td>NntvQA</td><td>Qapper</td><td>Mf-en</td><td>HotpotQA</td><td>2WikiMQA</td><td>Musique</td><td>GovReport</td><td>QMSum</td><td>MultiViews</td><td>TREC</td><td>TriviaQA</td><td>SAMSum</td><td>PCount</td><td>Ppre</td><td>Lcc</td><td>RB-P</td><td></td></tr><tr><td>Pyramidinfer</td><td>20.42</td><td>12.77</td><td>25.21</td><td>35.81</td><td>25.83</td><td>16.88</td><td>18.27</td><td>21.78</td><td>18.52</td><td>51.00</td><td>88.54</td><td>35.76</td><td>5.61</td><td>69.25</td><td>53.21</td><td>44.12</td><td>33.94</td></tr><tr><td>PyramidKV</td><td>21.13</td><td>14.18</td><td>30.26</td><td>35.12</td><td>23.76</td><td>16.17</td><td>18.33</td><td>21.65</td><td>19.23</td><td>58.00</td><td>88.31</td><td>37.07</td><td>5.23</td><td>69.50</td><td>52.61</td><td>45.74</td><td>34.76</td></tr></table>

# L PyramidKV will cause minimal extra inference overhead.

The allocation strategy and score-based selection add minimal complexity in the inference phase compared to the computation required for next-token predictions as Table 9. Each row shows the setting of using a specific “[Prompt length, Generation length]” combination. We show the inference speed comparison between total inference time, time for allocation strategy and time for score-based selection on LlaMa-3-8B-Instruct. Each cell is the latency measured in seconds. Furthermore, our budget allocation can be calculated before inference, requiring only a one-time computation. Thus, PyramidKV will cause minimal extra inference overhead.

Table 9: Extra inference overhead of PyramidKV   

<table><tr><td>Prompt Length</td><td>Generation Length</td><td>Inference Time</td><td>Allocation Time</td><td>Selection Time</td></tr><tr><td>512</td><td>512</td><td>18.26</td><td>0.0000003</td><td>0.0194</td></tr><tr><td>512</td><td>1024</td><td>34.69</td><td>0.000002</td><td>0.0133</td></tr><tr><td>512</td><td>2048</td><td>70.69</td><td>0.000003</td><td>0.013</td></tr><tr><td>512</td><td>4096</td><td>138.62</td><td>0.000005</td><td>0.013</td></tr><tr><td>1024</td><td>512</td><td>17.32</td><td>0.000002</td><td>0.0131</td></tr><tr><td>1024</td><td>1024</td><td>34.67</td><td>0.000002</td><td>0.01288</td></tr><tr><td>1024</td><td>2048</td><td>70.21</td><td>0.000005</td><td>0.01296</td></tr><tr><td>1024</td><td>4096</td><td>138.61</td><td>0.000003</td><td>0.01297</td></tr><tr><td>2048</td><td>512</td><td>17.48</td><td>0.000004</td><td>0.0128</td></tr><tr><td>2048</td><td>1024</td><td>34.78</td><td>0.000006</td><td>0.0129</td></tr><tr><td>2048</td><td>2048</td><td>69.50</td><td>0.000003</td><td>0.01297</td></tr><tr><td>2048</td><td>4096</td><td>138.59</td><td>0.000003</td><td>0.013</td></tr><tr><td>4096</td><td>512</td><td>17.58</td><td>0.000002</td><td>0.013</td></tr><tr><td>4096</td><td>1024</td><td>34.93</td><td>0.000004</td><td>0.0129</td></tr><tr><td>4096</td><td>2048</td><td>69.65</td><td>0.000002</td><td>0.013</td></tr><tr><td>4096</td><td>4096</td><td>138.87</td><td>0.000002</td><td>0.013</td></tr></table>

# M Inference Speed Comparison

PyramidKV does not require extra computation time for budget allocation at inference by design. We show the inference speed comparison between PyramidKV and baselines on LlaMa-3-8B-Instruct as Table 10. Each row shows the setting of using a specific “[Prompt length, Generation length]” combination. Each cell is the latency measured in seconds. PyramidKV does not sacrifice the speed. PyramidKV provides performance improvement and memory saving while runs at a comparable speed compared with baselines (i.e. SnapKV (Li et al., 2024), StreamingLLM (Xiao et al., 2023) and H2O (Zhang et al., 2024)). That’s because the allocation strategy requires very limited additional complexity in the inference/generation phase compared with computation required for generation as Appendix L.

# N PyramidKV Excels in all KV Cache Size Limitation

The evaluation results from LongBench(Bai et al., 2023) are shown in Table 11, Table 12, andTable 13. We report the results using LlaMa-3-8B-Instruct, LlaMa-3-70B-Instruct and Mistral-7B-Instruct(Jiang et al., 2023) for different KV cache sizes.

Overall, PyramidKV consistently surpasses other method across a range of KV cache sizes and different backbone models, with its performance advantages becoming particularly pronounced in memory-constrained environments. Upon examining specific tasks, PyramidKV demonstrates a notably superior performance on the TREC task, a few-shot question answering challenge. This suggests that the model effectively aggregates information from the few-shot examples, highlighting the potential for further investigation into in-context learning tasks.

Table 10: Performance comparison across different configurations and methods.   

<table><tr><td>Prompt Length</td><td>Generation Length</td><td>H2O</td><td>SnapKV</td><td>StreamingLLM</td><td>PyramidKV</td></tr><tr><td>512</td><td>512</td><td>18.47</td><td>18.25</td><td>18.96</td><td>18.26</td></tr><tr><td>512</td><td>1024</td><td>35.10</td><td>34.76</td><td>36.20</td><td>34.69</td></tr><tr><td>512</td><td>2048</td><td>70.21</td><td>69.60</td><td>72.35</td><td>70.69</td></tr><tr><td>512</td><td>4096</td><td>140.80</td><td>139.42</td><td>146.37</td><td>138.62</td></tr><tr><td>1024</td><td>512</td><td>17.63</td><td>17.34</td><td>18.12</td><td>17.32</td></tr><tr><td>1024</td><td>1024</td><td>35.16</td><td>34.61</td><td>36.17</td><td>34.67</td></tr><tr><td>1024</td><td>2048</td><td>71.02</td><td>69.17</td><td>72.37</td><td>70.21</td></tr><tr><td>1024</td><td>4096</td><td>140.51</td><td>138.83</td><td>146.09</td><td>138.61</td></tr><tr><td>2048</td><td>512</td><td>17.64</td><td>19.54</td><td>18.22</td><td>17.48</td></tr><tr><td>2048</td><td>1024</td><td>35.09</td><td>34.76</td><td>36.29</td><td>34.78</td></tr><tr><td>2048</td><td>2048</td><td>70.84</td><td>69.56</td><td>72.46</td><td>69.50</td></tr><tr><td>2048</td><td>4096</td><td>140.16</td><td>139.55</td><td>145.22</td><td>138.59</td></tr><tr><td>4096</td><td>512</td><td>17.75</td><td>17.67</td><td>18.40</td><td>17.58</td></tr><tr><td>4096</td><td>1024</td><td>35.20</td><td>35.08</td><td>36.46</td><td>34.93</td></tr><tr><td>4096</td><td>2048</td><td>70.02</td><td>69.26</td><td>72.58</td><td>69.65</td></tr><tr><td>4096</td><td>4096</td><td>139.87</td><td>138.57</td><td>144.98</td><td>138.87</td></tr></table>

Table 11: Performance comparison of PyramidKV (Ours) with SnapKV (SKV), H2O, StreamingLLM (SLM) and FullKV (FKV) on LongBench for LlaMa-3-8B-Instruct. PyramidKV generally outperforms other KV Cache compression methods across various KV Cache sizes and LLMs. The performance strengths of PyramidKV are more evident in small KV Cache sizes. Bold text represents the best performance.   

<table><tr><td rowspan="2">Method</td><td colspan="3">Single-Document QA</td><td colspan="3">Multi-Document QA</td><td colspan="3">Summarization</td><td colspan="3">Few-shot Learning</td><td colspan="2">Synthetic</td><td colspan="2">Code</td><td></td></tr><tr><td>NttyQA</td><td>Qasper</td><td>MF-en</td><td>HotspotQA</td><td>2WikiMQA</td><td>Musique</td><td>GovReport</td><td>OMSum</td><td>MultiNews</td><td>TREC</td><td>TriviaQA</td><td>SAMSum</td><td>PCount</td><td>pRe</td><td>Lcc</td><td>RB-P</td><td>Avg.</td></tr><tr><td></td><td>18409</td><td>3619</td><td>4559</td><td>9151</td><td>4887</td><td>11214</td><td>8734</td><td>10614</td><td>2113</td><td>5177</td><td>8209</td><td>6258</td><td>11141</td><td>9289</td><td>1235</td><td>4206</td><td></td></tr><tr><td colspan="17">LlaMa-3-8B-Instruct, KV Size = Full</td><td></td></tr><tr><td>FKV</td><td>25.70</td><td>29.75</td><td>41.12</td><td>45.55</td><td>35.87</td><td>22.35</td><td>25.63</td><td>23.03</td><td>26.21</td><td>73.00</td><td>90.56</td><td>41.88</td><td>04.67</td><td>69.25</td><td>58.05</td><td>50.77</td><td>41.46</td></tr><tr><td colspan="17">LlaMa-3-8B-Instruct, KV Size = 64</td><td></td></tr><tr><td>SKV</td><td>19.86</td><td>9.09</td><td>27.89</td><td>37.34</td><td>28.35</td><td>18.17</td><td>15.86</td><td>20.80</td><td>16.41</td><td>38.50</td><td>85.92</td><td>36.32</td><td>5.22</td><td>69.00</td><td>51.78</td><td>48.38</td><td>33.05</td></tr><tr><td>H2O</td><td>20.80</td><td>11.34</td><td>27.03</td><td>37.25</td><td>30.01</td><td>17.94</td><td>18.29</td><td>21.49</td><td>19.13</td><td>38.00</td><td>84.70</td><td>37.76</td><td>5.63</td><td>69.33</td><td>53.44</td><td>50.15</td><td>33.89</td></tr><tr><td>SLM</td><td>17.44</td><td>8.68</td><td>22.25</td><td>35.37</td><td>31.51</td><td>15.97</td><td>15.46</td><td>20.06</td><td>14.64</td><td>38.00</td><td>72.33</td><td>29.10</td><td>5.42</td><td>69.50</td><td>46.14</td><td>45.09</td><td>30.43</td></tr><tr><td>Ours</td><td>21.13</td><td>14.18</td><td>30.26</td><td>35.12</td><td>23.76</td><td>16.17</td><td>18.33</td><td>21.65</td><td>19.23</td><td>58.00</td><td>88.31</td><td>37.07</td><td>5.23</td><td>69.50</td><td>52.61</td><td>45.74</td><td>34.76</td></tr><tr><td colspan="17">LlaMa-3-8B-Instruct, KV Size = 96</td><td></td></tr><tr><td>SKV</td><td>20.45</td><td>10.34</td><td>31.84</td><td>37.85</td><td>28.65</td><td>18.52</td><td>17.90</td><td>21.26</td><td>19.07</td><td>41.50</td><td>86.95</td><td>37.82</td><td>5.08</td><td>69.12</td><td>54.69</td><td>51.31</td><td>34.51</td></tr><tr><td>H2O</td><td>21.55</td><td>11.21</td><td>28.73</td><td>37.66</td><td>30.12</td><td>18.47</td><td>19.57</td><td>21.57</td><td>20.44</td><td>38.50</td><td>87.63</td><td>38.47</td><td>5.60</td><td>69.00</td><td>54.51</td><td>50.16</td><td>34.57</td></tr><tr><td>SLM</td><td>18.67</td><td>8.43</td><td>24.98</td><td>38.35</td><td>30.59</td><td>16.37</td><td>17.33</td><td>19.84</td><td>18.41</td><td>41.00</td><td>73.92</td><td>29.38</td><td>5.80</td><td>69.50</td><td>47.15</td><td>45.61</td><td>31.58</td></tr><tr><td>Ours</td><td>21.67</td><td>15.10</td><td>33.50</td><td>39.73</td><td>26.48</td><td>17.47</td><td>19.64</td><td>22.28</td><td>20.49</td><td>61.50</td><td>87.38</td><td>38.18</td><td>6.00</td><td>69.25</td><td>55.30</td><td>46.78</td><td>36.29</td></tr><tr><td colspan="17">LlaMa-3-8B-Instruct, KV Size = 128</td><td></td></tr><tr><td>SKV</td><td>21.19</td><td>13.55</td><td>32.64</td><td>38.75</td><td>29.64</td><td>18.73</td><td>18.98</td><td>21.62</td><td>20.26</td><td>45.00</td><td>88.36</td><td>37.64</td><td>5.13</td><td>68.85</td><td>55.84</td><td>51.82</td><td>35.50</td></tr><tr><td>H2O</td><td>22.12</td><td>13.20</td><td>31.61</td><td>37.79</td><td>32.71</td><td>18.45</td><td>20.32</td><td>22.02</td><td>21.10</td><td>38.50</td><td>87.75</td><td>39.14</td><td>5.83</td><td>69.50</td><td>55.06</td><td>50.97</td><td>35.37</td></tr><tr><td>SLM</td><td>18.61</td><td>9.65</td><td>25.99</td><td>37.95</td><td>29.39</td><td>16.34</td><td>18.03</td><td>20.11</td><td>20.08</td><td>43.50</td><td>74.08</td><td>29.86</td><td>5.90</td><td>69.50</td><td>47.47</td><td>45.60</td><td>32.00</td></tr><tr><td>Ours</td><td>21.40</td><td>16.92</td><td>33.79</td><td>39.73</td><td>28.72</td><td>18.59</td><td>19.86</td><td>22.48</td><td>20.95</td><td>66.50</td><td>89.35</td><td>38.39</td><td>5.92</td><td>69.00</td><td>56.49</td><td>47.95</td><td>37.25</td></tr><tr><td colspan="17">LlaMa-3-8B-Instruct, KV Size = 2048</td><td></td></tr><tr><td>SKV</td><td>25.86</td><td>29.55</td><td>41.10</td><td>44.99</td><td>35.80</td><td>21.81</td><td>25.98</td><td>23.40</td><td>26.46</td><td>73.50</td><td>90.56</td><td>41.66</td><td>5.17</td><td>69.25</td><td>56.65</td><td>49.94</td><td>41.35</td></tr><tr><td>SLM</td><td>21.71</td><td>25.78</td><td>38.13</td><td>40.12</td><td>32.01</td><td>16.86</td><td>23.14</td><td>22.64</td><td>26.48</td><td>70.00</td><td>83.22</td><td>31.75</td><td>5.74</td><td>68.50</td><td>53.50</td><td>45.58</td><td>37.82</td></tr><tr><td>H2O</td><td>25.56</td><td>26.85</td><td>39.54</td><td>44.30</td><td>32.92</td><td>21.09</td><td>24.68</td><td>23.01</td><td>26.16</td><td>53.00</td><td>90.56</td><td>41.84</td><td>4.91</td><td>69.25</td><td>56.40</td><td>49.68</td><td>39.35</td></tr><tr><td>Ours</td><td>25.40</td><td>29.71</td><td>40.25</td><td>44.76</td><td>35.32</td><td>21.98</td><td>26.83</td><td>23.30</td><td>26.19</td><td>73.00</td><td>90.56</td><td>42.14</td><td>5.22</td><td>69.25</td><td>58.76</td><td>51.18</td><td>41.49</td></tr></table>

Table 12: Performance comparison of PyramidKV (Ours) with SnapKV (SKV), H2O, StreamingLLM (SLM) and FullKV (FKV) on LongBench for Mistral-7B-Instruct. PyramidKV generally outperforms other KV Cache compression methods across various KV Cache sizes and LLMs. The performance strengths of PyramidKV are more evident in small KV Cache sizes. Bold text represents the best performance.   

<table><tr><td rowspan="2">Method</td><td colspan="3">Single-Document QA</td><td colspan="3">Multi-Document QA</td><td colspan="3">Summarization</td><td colspan="3">Few-shot Learning</td><td colspan="2">Synthetic</td><td colspan="2">Code</td><td></td></tr><tr><td>NvQA</td><td>Qasper</td><td>MF-en</td><td>HotspotQA</td><td>2WikiMQA</td><td>Musicie</td><td>GovReport</td><td>OMSum</td><td>MultiNews</td><td>TREC</td><td>TriviaQA</td><td>SAMSum</td><td>PCount</td><td>PRe</td><td>Lcc</td><td>RB-P</td><td>Avg.</td></tr><tr><td></td><td>18409</td><td>3619</td><td>4559</td><td>9151</td><td>4887</td><td>11214</td><td>8734</td><td>10614</td><td>2113</td><td>5177</td><td>8209</td><td>6258</td><td>11141</td><td>9289</td><td>1235</td><td>4206</td><td></td></tr><tr><td colspan="17">Mistral-7B-Instruct, KV Size = Full</td><td></td></tr><tr><td>FKV</td><td>26.90</td><td>33.07</td><td>49.20</td><td>43.02</td><td>27.33</td><td>18.78</td><td>32.91</td><td>24.21</td><td>26.99</td><td>71.00</td><td>86.23</td><td>42.65</td><td>2.75</td><td>86.98</td><td>56.96</td><td>54.52</td><td>42.71</td></tr><tr><td colspan="17">Mistral-7B-Instruct, KV Size = 64</td><td></td></tr><tr><td>SKV</td><td>16.94</td><td>17.17</td><td>39.51</td><td>36.87</td><td>22.26</td><td>15.18</td><td>14.75</td><td>20.35</td><td>21.45</td><td>37.50</td><td>84.16</td><td>37.28</td><td>4.50</td><td>61.13</td><td>42.40</td><td>38.44</td><td>30.72</td></tr><tr><td>SLM</td><td>15.01</td><td>13.84</td><td>28.74</td><td>30.97</td><td>24.50</td><td>13.42</td><td>13.25</td><td>19.46</td><td>19.17</td><td>35.50</td><td>76.91</td><td>29.61</td><td>4.67</td><td>27.33</td><td>38.71</td><td>35.29</td><td>25.60</td></tr><tr><td>H2O</td><td>18.19</td><td>19.04</td><td>37.40</td><td>30.18</td><td>22.22</td><td>13.77</td><td>16.60</td><td>21.52</td><td>21.98</td><td>37.00</td><td>81.02</td><td>38.62</td><td>5.00</td><td>66.03</td><td>43.54</td><td>40.46</td><td>30.88</td></tr><tr><td>Ours</td><td>20.91</td><td>20.21</td><td>39.94</td><td>33.57</td><td>22.87</td><td>15.70</td><td>17.31</td><td>21.23</td><td>21.41</td><td>54.00</td><td>81.98</td><td>36.96</td><td>3.58</td><td>60.83</td><td>44.52</td><td>37.99</td><td>32.19</td></tr><tr><td colspan="17">Mistral-7B-Instruct, KV Size = 96</td><td></td></tr><tr><td>SKV</td><td>19.92</td><td>18.80</td><td>43.29</td><td>39.66</td><td>23.08</td><td>15.94</td><td>16.65</td><td>21.26</td><td>21.47</td><td>43.50</td><td>83.48</td><td>39.74</td><td>4.00</td><td>60.10</td><td>45.53</td><td>41.12</td><td>32.47</td></tr><tr><td>SLM</td><td>15.15</td><td>15.48</td><td>31.44</td><td>30.03</td><td>23.93</td><td>12.73</td><td>16.76</td><td>19.15</td><td>19.19</td><td>41.50</td><td>75.31</td><td>28.71</td><td>5.00</td><td>28.48</td><td>38.92</td><td>36.05</td><td>26.37</td></tr><tr><td>H2O</td><td>19.44</td><td>20.81</td><td>38.78</td><td>32.39</td><td>21.51</td><td>14.43</td><td>17.68</td><td>22.40</td><td>21.99</td><td>38.00</td><td>82.51</td><td>39.94</td><td>6.06</td><td>77.48</td><td>45.18</td><td>42.43</td><td>32.67</td></tr><tr><td>Ours</td><td>20.35</td><td>21.87</td><td>41.15</td><td>34.94</td><td>21.85</td><td>15.81</td><td>18.21</td><td>21.66</td><td>21.43</td><td>65.00</td><td>83.60</td><td>39.60</td><td>4.50</td><td>67.80</td><td>45.83</td><td>39.38</td><td>34.08</td></tr><tr><td colspan="17">Mistral-7B-Instruct, KV Size = 128</td><td></td></tr><tr><td>SKV</td><td>19.16</td><td>21.46</td><td>43.52</td><td>38.60</td><td>23.35</td><td>16.09</td><td>17.66</td><td>21.84</td><td>21.47</td><td>47.50</td><td>84.15</td><td>40.24</td><td>5.00</td><td>69.31</td><td>46.98</td><td>42.97</td><td>34.96</td></tr><tr><td>SLM</td><td>16.57</td><td>14.68</td><td>32.40</td><td>30.19</td><td>22.64</td><td>12.34</td><td>18.08</td><td>18.96</td><td>19.19</td><td>43.50</td><td>74.22</td><td>29.02</td><td>4.50</td><td>29.48</td><td>39.23</td><td>36.16</td><td>27.57</td></tr><tr><td>H2O</td><td>21.20</td><td>21.90</td><td>41.55</td><td>33.56</td><td>21.28</td><td>12.93</td><td>18.59</td><td>22.61</td><td>21.99</td><td>39.00</td><td>82.37</td><td>40.44</td><td>6.00</td><td>83.19</td><td>46.41</td><td>42.66</td><td>34.73</td></tr><tr><td>Ours</td><td>21.75</td><td>22.03</td><td>44.32</td><td>34.06</td><td>22.79</td><td>15.77</td><td>18.58</td><td>21.89</td><td>21.43</td><td>66.00</td><td>83.46</td><td>39.75</td><td>4.50</td><td>66.90</td><td>46.96</td><td>41.28</td><td>35.72</td></tr><tr><td colspan="17">Mistral-7B-Instruct, KV Size = 2048</td><td></td></tr><tr><td>SKV</td><td>25.89</td><td>32.93</td><td>48.56</td><td>42.96</td><td>27.42</td><td>19.02</td><td>26.56</td><td>24.47</td><td>26.69</td><td>70.00</td><td>86.27</td><td>42.57</td><td>5.50</td><td>88.90</td><td>50.42</td><td>46.72</td><td>41.56</td></tr><tr><td>SLM</td><td>20.31</td><td>26.64</td><td>45.72</td><td>35.25</td><td>24.31</td><td>12.20</td><td>27.47</td><td>21.57</td><td>24.51</td><td>68.50</td><td>71.95</td><td>31.19</td><td>5.00</td><td>22.56</td><td>43.38</td><td>37.08</td><td>32.35</td></tr><tr><td>H2O</td><td>25.76</td><td>31.10</td><td>49.03</td><td>40.76</td><td>26.52</td><td>17.07</td><td>24.81</td><td>23.64</td><td>26.60</td><td>55.00</td><td>86.35</td><td>42.48</td><td>5.50</td><td>88.15</td><td>49.93</td><td>46.57</td><td>39.95</td></tr><tr><td>Ours</td><td>25.53</td><td>32.21</td><td>48.97</td><td>42.26</td><td>27.50</td><td>19.36</td><td>26.60</td><td>23.97</td><td>26.73</td><td>71.00</td><td>86.25</td><td>42.94</td><td>4.50</td><td>87.90</td><td>53.12</td><td>47.21</td><td>41.63</td></tr></table>

Table 13: Performance comparison of PyramidKV (Ours) with SnapKV (SKV), H2O, StreamingLLM (SLM) and FullKV (FKV) on LongBench for LlaMa-3-70B-Instruct. PyramidKV generally outperforms other KV Cache compression methods across various KV Cache sizes and LLMs. The performance strengths of PyramidKV are more evident in small KV Cache sizes. Bold text represents the best performance.   

<table><tr><td rowspan="2">Method</td><td colspan="3">Single-Document QA</td><td colspan="3">Multi-Document QA</td><td colspan="3">Summarization</td><td colspan="3">Few-shot Learning</td><td colspan="2">Synthetic</td><td colspan="2">Code</td><td></td></tr><tr><td>NtrvQA</td><td>Qasper</td><td>MF-en</td><td>HotpotQA</td><td>2WikiMQA</td><td>Musique</td><td>GovReport</td><td>OMStat</td><td>MultiNews</td><td>TREC</td><td>TriviaQA</td><td>SAMSum</td><td>PCount</td><td>Pre</td><td>Lcc</td><td>RB-P</td><td>Avg.</td></tr><tr><td></td><td>18409</td><td>3619</td><td>4559</td><td>9151</td><td>4887</td><td>11214</td><td>8734</td><td>10614</td><td>2113</td><td>5177</td><td>8209</td><td>6258</td><td>11141</td><td>9289</td><td>1235</td><td>4206</td><td></td></tr><tr><td colspan="17">LlaMa-3-70B-Instruct, KV Size = Full</td><td></td></tr><tr><td>FKV</td><td>27.75</td><td>46.48</td><td>49.45</td><td>52.04</td><td>54.9</td><td>30.42</td><td>32.37</td><td>22.27</td><td>27.58</td><td>73.5</td><td>92.46</td><td>45.73</td><td>12.5</td><td>72.5</td><td>40.96</td><td>63.91</td><td>46.55</td></tr><tr><td colspan="17">LlaMa-3-70B-Instruct, KV Size = 64</td><td></td></tr><tr><td>SKV</td><td>23.92</td><td>31.09</td><td>36.54</td><td>46.66</td><td>50.40</td><td>25.30</td><td>18.05</td><td>21.11</td><td>19.79</td><td>41.50</td><td>91.06</td><td>40.26</td><td>12.00</td><td>72.50</td><td>43.33</td><td>57.62</td><td>39.45</td></tr><tr><td>SLM</td><td>22.07</td><td>23.53</td><td>27.31</td><td>43.21</td><td>51.66</td><td>23.85</td><td>16.62</td><td>19.74</td><td>15.20</td><td>39.50</td><td>76.89</td><td>33.06</td><td>12.00</td><td>72.50</td><td>40.23</td><td>50.20</td><td>35.47</td></tr><tr><td>H2O</td><td>25.45</td><td>34.64</td><td>33.23</td><td>48.25</td><td>50.30</td><td>24.88</td><td>20.03</td><td>21.50</td><td>21.39</td><td>42.00</td><td>90.36</td><td>41.58</td><td>12.00</td><td>71.50</td><td>43.83</td><td>58.16</td><td>39.94</td></tr><tr><td>Ours</td><td>25.47</td><td>36.71</td><td>42.29</td><td>47.08</td><td>46.21</td><td>28.30</td><td>20.60</td><td>21.62</td><td>21.62</td><td>64.50</td><td>89.61</td><td>41.28</td><td>12.50</td><td>72.50</td><td>45.34</td><td>56.50</td><td>42.01</td></tr><tr><td colspan="17">LlaMa-3-70B-Instruct, KV Size = 96</td><td></td></tr><tr><td>SKV</td><td>25.78</td><td>35.71</td><td>42.13</td><td>50.38</td><td>51.46</td><td>26.68</td><td>19.61</td><td>21.40</td><td>21.98</td><td>48.50</td><td>92.11</td><td>41.21</td><td>12.00</td><td>72.00</td><td>44.85</td><td>59.05</td><td>41.55</td></tr><tr><td>SLM</td><td>23.31</td><td>29.46</td><td>29.21</td><td>41.85</td><td>45.92</td><td>23.00</td><td>18.42</td><td>19.71</td><td>18.57</td><td>45.00</td><td>76.79</td><td>33.54</td><td>12.00</td><td>72.50</td><td>40.49</td><td>50.73</td><td>36.28</td></tr><tr><td>H2O</td><td>25.30</td><td>35.13</td><td>35.54</td><td>47.39</td><td>50.61</td><td>26.20</td><td>20.87</td><td>21.80</td><td>22.93</td><td>41.00</td><td>90.47</td><td>43.42</td><td>12.00</td><td>72.00</td><td>43.84</td><td>59.86</td><td>40.52</td></tr><tr><td>Ours</td><td>25.47</td><td>37.61</td><td>44.00</td><td>47.33</td><td>45.36</td><td>27.91</td><td>21.05</td><td>21.60</td><td>22.31</td><td>66.00</td><td>91.45</td><td>42.36</td><td>12.00</td><td>72.50</td><td>45.12</td><td>56.88</td><td>42.43</td></tr><tr><td colspan="17">LlaMa-3-70B-Instruct, KV Size = 128</td><td></td></tr><tr><td>SKV</td><td>26.22</td><td>37.49</td><td>45.70</td><td>50.86</td><td>52.82</td><td>28.50</td><td>20.38</td><td>21.72</td><td>22.56</td><td>53.00</td><td>91.61</td><td>41.43</td><td>12.00</td><td>71.50</td><td>45.06</td><td>60.50</td><td>42.58</td></tr><tr><td>SLM</td><td>24.25</td><td>29.12</td><td>29.24</td><td>40.20</td><td>46.28</td><td>21.80</td><td>19.55</td><td>19.42</td><td>20.61</td><td>48.00</td><td>76.60</td><td>33.21</td><td>12.00</td><td>72.50</td><td>40.65</td><td>51.03</td><td>36.53</td></tr><tr><td>H2O</td><td>25.61</td><td>35.02</td><td>37.74</td><td>47.77</td><td>51.16</td><td>26.87</td><td>20.57</td><td>20.78</td><td>23.33</td><td>42.00</td><td>91.65</td><td>43.85</td><td>12.00</td><td>72.50</td><td>43.50</td><td>59.67</td><td>40.88</td></tr><tr><td>Ours</td><td>26.06</td><td>40.35</td><td>45.67</td><td>50.20</td><td>52.78</td><td>29.36</td><td>22.31</td><td>22.02</td><td>23.69</td><td>71.00</td><td>92.27</td><td>44.33</td><td>12.00</td><td>72.50</td><td>45.90</td><td>59.55</td><td>44.37</td></tr><tr><td colspan="17">LlaMa-3-70B-Instruct, KV Size = 2048</td><td></td></tr><tr><td>SKV</td><td>26.73</td><td>45.18</td><td>47.91</td><td>52.00</td><td>55.24</td><td>30.48</td><td>28.76</td><td>22.35</td><td>27.31</td><td>72.50</td><td>92.38</td><td>45.58</td><td>12.00</td><td>72.50</td><td>41.52</td><td>69.27</td><td>46.36</td></tr><tr><td>SLM</td><td>26.69</td><td>41.01</td><td>35.97</td><td>46.55</td><td>52.98</td><td>25.71</td><td>27.81</td><td>20.81</td><td>27.16</td><td>69.00</td><td>91.55</td><td>44.02</td><td>12.00</td><td>72.00</td><td>41.44</td><td>68.73</td><td>43.96</td></tr><tr><td>H2O</td><td>27.67</td><td>46.51</td><td>49.54</td><td>51.49</td><td>53.85</td><td>29.97</td><td>28.57</td><td>22.79</td><td>27.53</td><td>59.00</td><td>92.63</td><td>45.94</td><td>12.00</td><td>72.50</td><td>41.39</td><td>63.90</td><td>45.33</td></tr><tr><td>Ours</td><td>27.22</td><td>46.19</td><td>48.72</td><td>51.62</td><td>54.56</td><td>31.11</td><td>29.76</td><td>22.50</td><td>27.27</td><td>73.50</td><td>91.88</td><td>45.47</td><td>12.00</td><td>72.50</td><td>41.36</td><td>69.12</td><td>46.55</td></tr></table>

With a small budget, our proposed method enables more effective allocation, better preserving useful attention information. Second, with a large budget, such allocation becomes less critical, as it is sufficient to cover the necessary information. To further illustrate this phenomenon, we have included an ablation study titled "Attention Recall Rate Experiment" as Figure 8. The results show that with a small budget, PyramidKV improves the attention recall rate (the percentage of attention computed using the keys retrieved by the method and the query, relative to the attention computed using all keys and the query.). However, with a larger budget (i.e., 2k KV Cache Size), the improvement decreases. For 64, 128, 256, 512, 1024 and 2048 KV Cache sizes, PyramidKV’s average attention recall rate improvements are $1 . 8 7 \%$ , $0 . 6 4 \%$ , $0 . 6 1 \%$ , $0 . 5 6 \%$ , $0 . 4 7 \%$ and $0 . 3 6 \%$ .

![](images/ae4c5dbc05084d9330d75d7c19287cd3a94c8ce86aedac23856fcf08d31418f8.jpg)

![](images/6be3393a53d37cea239e0f82b141bafdd1a2b16e90293df580693af0737656f4.jpg)

![](images/902efc6336764c9db9e1a635878109d5e089f3ddc1e898e527cda25cff584f18.jpg)

![](images/7828bd313ff6e648e71acd1185609069234a6517b0ae17c4264474441397cf2f.jpg)

![](images/35cdafe97c0f286e3a6c94482d5a70b3ed8d43ba557cac80fbc649d42b193e4a.jpg)

![](images/013feac63d26117ed7e46cbc96cac85e36f357cf04ccece48c4b0cf314fc67e5.jpg)  
Figure 8: Attention recall rate (the percentage of attention computed using the keys retrieved by the method and the query, relative to the attention computed using all keys and the query.) comparison of PyramidKV and SnapKV.

# O LongBench results for 128 context length

We conducted additional experiments using Llama-3-8B-Instruct-Gradient-1048k with a sequence length of 128k as Table 14. The results, summarized in the table below, showcase the model’s performance with extended context lengths. These findings provide further validation of the scalability and robustness of our approach.

Table 14: Comparison of PyramidKV with baselines at 128k context length.   

<table><tr><td rowspan="2">Method</td><td colspan="3">Single-Document QA</td><td colspan="3">Multi-Document QA</td><td colspan="3">Summarization</td><td colspan="3">Few-shot Learning</td><td colspan="2">Synthetic</td><td>Code</td><td>Avg.</td><td></td></tr><tr><td>NntvQA</td><td>Qasper</td><td>ME-en</td><td>HotspotQA</td><td>2WileLMQA</td><td>Musicie</td><td>Govtreport</td><td>OMSum</td><td>MultiNews</td><td>TREC</td><td>TriviaQA</td><td>SAMSum</td><td>PCount</td><td>PRe</td><td>Lcc</td><td>RBP</td><td></td></tr><tr><td>SnapKV</td><td>6.10</td><td>8.14</td><td>23.12</td><td>8.87</td><td>10.54</td><td>5.59</td><td>20.27</td><td>17.95</td><td>18.07</td><td>50.50</td><td>82.78</td><td>34.67</td><td>3.50</td><td>49.25</td><td>45.39</td><td>41.68</td><td>26.65</td></tr><tr><td>H2O</td><td>3.47</td><td>7.49</td><td>14.17</td><td>7.30</td><td>8.74</td><td>4.55</td><td>24.13</td><td>17.83</td><td>21.91</td><td>61.50</td><td>81.45</td><td>23.60</td><td>3.55</td><td>41.80</td><td>43.25</td><td>38.51</td><td>25.20</td></tr><tr><td>StreamingLLM</td><td>3.47</td><td>7.49</td><td>14.17</td><td>7.30</td><td>8.74</td><td>4.55</td><td>19.21</td><td>17.83</td><td>21.91</td><td>61.50</td><td>78.21</td><td>23.60</td><td>3.55</td><td>41.80</td><td>43.25</td><td>38.51</td><td>24.69</td></tr><tr><td>PyramidKV</td><td>5.41</td><td>8.42</td><td>22.61</td><td>9.71</td><td>10.73</td><td>5.82</td><td>20.37</td><td>18.24</td><td>18.32</td><td>54.00</td><td>85.33</td><td>34.60</td><td>3.50</td><td>52.75</td><td>47.23</td><td>42.58</td><td>27.48</td></tr></table>

# P PyramidKV Preserves the Long-Context Understanding Ability

We perform Fact Retrieval Across Context Lengths (“Needle In A HayStack”) (Liu et al., 2023a; Fu et al., 2024) to test the in-context retrieval ability of LLMs after leveraging different KV cache methods. We conducted the Needle-in-a-Haystack experiment using various LLMs

(i.e., Mistral-7B-Instruct-32k, LLaMA-3-8B-Instruct-8k, and LLaMA-3-70B-Instruct-8k), various KV cache sizes (i.e., 64, 96, and 128) and various methods (i.e., FullKV, PyramidKV, H2O and StreamingLLM). PyramidKV achieves Acc. performance closest to FullKV, while other methods show significant decreases. It is worth noting that PyramidKV with 128 KV cache size achieves the same 100.0 Acc. performance compared with FullKV with 8k context size for LLaMA-3-70B-Instruct.

Figure 9, Figure 10, Figure 11 show the results of Mistral-7B-Instruct (Jiang et al., 2023) with different cache size (64, 96 and 128, respectively).

Figure 12, Figure 13, Figure 14 show the results of LlaMa-3-8B-Instruct with different cache size (64, 96 and 128, respectively).

Figure 15, Figure 16, Figure 17 show the results of LlaMa-3-70B-Instruct with different cache size (64, 96 and 128, respectively).

Table 15: Recall Accuracy performance from Fact Retrieval Across Context Lengths (“Needle In A HayStack”)   

<table><tr><td>Model</td><td>Length</td><td>KV Cache</td><td>Full KV Acc.</td><td>PyramidKV Acc.</td><td>SnapKV Acc.</td><td>H2O Acc.</td></tr><tr><td>Mistral-7B</td><td>32k</td><td>64</td><td>100.00</td><td>80.50</td><td>43.90</td><td>48.40</td></tr><tr><td>Mistral-7B</td><td>32k</td><td>96</td><td>100.00</td><td>90.50</td><td>72.20</td><td>59.10</td></tr><tr><td>Mistral-7B</td><td>32k</td><td>128</td><td>100.00</td><td>91.60</td><td>80.10</td><td>64.90</td></tr><tr><td>LLaMa-3-8B</td><td>8k</td><td>64</td><td>100.00</td><td>92.90</td><td>62.00</td><td>31.90</td></tr><tr><td>LLaMa-3-8B</td><td>8k</td><td>96</td><td>100.00</td><td>95.80</td><td>80.70</td><td>44.20</td></tr><tr><td>LLaMa-3-8B</td><td>8k</td><td>128</td><td>100.00</td><td>97.40</td><td>87.40</td><td>49.10</td></tr><tr><td>LLaMa-3-70B</td><td>8k</td><td>64</td><td>100.00</td><td>99.60</td><td>76.20</td><td>47.30</td></tr><tr><td>LLaMa-3-70B</td><td>8k</td><td>96</td><td>100.00</td><td>98.60</td><td>94.40</td><td>69.90</td></tr><tr><td>LLaMa-3-70B</td><td>8k</td><td>128</td><td>100.00</td><td>100.00</td><td>98.60</td><td>82.30</td></tr></table>

![](images/818abcbeceb99124cac7ebb91c1d619f75cffecea4e4c0b8e6ca7314460c2752.jpg)  
Mistral-7B-v0.2 - 32K Context Size

![](images/3ad5cbad36ce65ca6cb47f06b8cadd45de4c31dbf6869b877683417c9e081627.jpg)  
(a) FullKV, KV Size $=$ Full, acc 100.0

![](images/a4c59e36c4c4d192501e1d4328ac48189810ec706313025640561d6dd8b79403.jpg)  
(b) PyramidKV,KV Size=64, acc 80.5   
(c) SnapKV,KV Size=64,acc 43.9

![](images/03e3672bf8d704d4263e850aa40c26ff633839bae9d79e3cc1ca27b02b023958.jpg)  
(d) H2O,KV Size=64, acc 48.4   
Figure 9: Results of the Fact Retrieval Across Context Lengths (“Needle In A HayStack”) test in Mistral-7B-Instruct with 32k context size in 64 KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](images/a7cf3ac9cf7a761562315700ae1b15508ce09ec5247e2f2e9ea0317fc7610b6c.jpg)  
Mistral-7B-v0.2 - 32K Context Size

![](images/e1ca543b15af25db731bb269050d9827a58eac1145c8607e33e06c7905483197.jpg)  
(a) FullKV,KVize $=$ Full, acc 100.0   
(b) PyramidKV,KV Size-96,acc 90.5

![](images/506f7dcf3ac73b7ef3cd3718786b4874c0dfdf770e1de2c6045160fb301c08de.jpg)  
(c) SnapKV,KV Size=96,acc 72.2

![](images/e6bb38241130a24c8b608ab64f6dadc001076579c22b3f6d586a491b1ebe8438.jpg)  
(d) H2O,KV Size=96,acc 59.1   
Figure 10: Results of the Fact Retrieval Across Context Lengths (“Needle In A HayStack”) test in Mistral-7B-Instruct with 32k context size in 96 KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](images/84629fe5ad71038a4ab5c710467ee1fd86f0b13097ffa2d4a3f3c0b9e2f7fb06.jpg)  
Mistral-7B-v0.2 - 32K Context Size

![](images/ff83144cc38dfe832e574e8eca32e32285056112571c3eada886385b9dba5f7b.jpg)  
(a)FullKV,KV ize $=$ Full, acc 100.0   
(b)PyramidKV,KV Size=128,acc91.6

![](images/5877e5097e220b0ef0db8ffb4ce1271e98fe3d2ce582802e8540bf93043adc8c.jpg)  
(c) SnapKV,KV Size=128,acc 80.1

![](images/7d0ac0ce2cca77992742d3ce547f5d5f5fe6ebf16d83920edadb2b9d0140b611.jpg)  
(d) H2O,KV Size=128,acc 64.9   
Figure 11: Results of the Fact Retrieval Across Context Lengths (“Needle In A HayStack”) test in Mistral-7B-Instruct with $\mathbf { 3 2 k }$ context size in 128 KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](images/5ecc03afb946dfb9905c1521f00da7f358618b261e2d36b536bfa5733eef28f8.jpg)  
LLaMA-3-8B - 8K Context Size

![](images/26a686d66f4a95cd1a10fa4ee3b48c170ea5853eeb0736dcd5c58d8f7bb03956.jpg)  
(a) FullKV, KV Size $=$ Full, acc 100.0   
(b) PyramidKV, KV Size=64,acc 92.9

![](images/b8450979a7ce0f26095d7ca2fa612f1b406f04cb3f5ed2b55edc3be280067197.jpg)

![](images/92e17ac64ab6dbe284d7be849b2ec59fbe88ad590ff3b29fecf500ca390417c0.jpg)  
(c) SnapKV, KV Size=64, acc 62.0   
(d) H2O, KV Size=64,acc 31.9   
Figure 12: Results of the Fact Retrieval Across Context Lengths (“Needle In A HayStack”) test in LlaMa-3-8B-Instruct with 8k context size in 64 KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](images/a5cc9e6d1e6925c998d9b35b10a8757054784a4bed0aa465ba18146043b3c156.jpg)  
Figure 13: Results of the Fact Retrieval Across Context Lengths (“Needle In A HayStack”) needle_llamatest in LlaMa-3-8B-Instruct with 8k context size in 96 KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](images/d10364d9a86d1ab0f2e447f6277679017a80ddda06243405a88364d695a27e22.jpg)  
LLaMA-3-8B - 8K Context Size

![](images/264178d4209693e9196f1fa909569ef8787a199797c3899896c26b054477b6eb.jpg)  
(a)FullKV,KVSize $=$ Full, acc 100.0   
(b) PyramidKV, KV Size=128, acc 97.4

![](images/1953ae9c7bcaa2876dfb7b3d5d424662117e2d645cb7f66ba9e51a21221c6038.jpg)  
(c) SnapKV,KV Size=128,acc 87.4

![](images/8b5b37a1bf79643f11cca228e816af557ca419c7398a5750da6b4f869c9fec1d.jpg)  
(d) H2O, KV Size=128,acc 49.1   
Figure 14: Results of the Fact Retrieval Across Context Lengths (“Needle In A HayStack”) test in LlaMa-3-8B-Instruct with 8k context size in 128 KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](images/3290d09ec1ab45554c044c789a71322d9b6ee5dd910fae0885ee3d1b67e79097.jpg)  
LLaMA-3-70B - 8K Context Size   
(a)FullKV,KV Size $=$ Full, acc 100.0

![](images/f90de33957be9cffad726b3dcd210d7f21bf424db63bdb02ce01046c467babf2.jpg)  
(b) PyramidKV,KV Size=64,acc 99.6

![](images/a86e62093fd35350af85dee216484e7679c06c20fe6b8eea151b93feb65b1a81.jpg)  
(c) SnapKV, KV Size=64,acc 76.2

![](images/4bf992aa370885dbf78818365520ff8cd4eaab4a26b34b103dd4f50e8f697ebf.jpg)  
(d) H2O,KV Size=64,acc 47.3   
Figure 15: Results of the Fact Retrieval Across Context Lengths (“Needle In A HayStack”) test in LlaMa-3-70B with 8k context size in 64 KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](images/8b955b393aa2345c946428f90c9bda11de60cca37a278a9db1cebd2685fff4fe.jpg)  
LLaMA-3-70B - 8K Context Size

![](images/6a298696bad02c48f758829e3ab62227af188c061496918b075a30f2b9c3be9f.jpg)  
(a) FullKV,KV Size $=$ Full, acc 100.0   
(b) PyramidKV, KV Size=96, acc 98.6

![](images/03622c39cd9f02e4457ecffa10d6b11632bec6b815c1df2d726ee74cd614cdcd.jpg)  
(c) SnapKV,KV Size=96,acc 94.4

![](images/5d4deb7fa342146868ef233c390a62cef024b0660b507d1c900e887002c27f2e.jpg)  
(d) H2O,KV Size=96,acc 69.9   
Figure 16: Results of the Fact Retrieval Across Context Lengths (“Needle In A HayStack”) test in LlaMa-3-70B with 8k context size in 96 KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

![](images/0e267101151b0c80e4c424d0d3c967b367537a72c8e20c25cd2eb86e5a4fa8ee.jpg)  
LLaMA-3-70B - 8K Context Size

![](images/de6a52ae5f68ddda5783b6102f41a9079af36a2b6f6926f4042568f4da0ed94a.jpg)  
(a) FullKV,KV Size $=$ Full acc 100.0

![](images/0a407d907511aa920c033936376390a92039dfc17e2ab8f94ee018c94398a94b.jpg)  
(b) PyramidKV, KV Size=128,acc 100.0

![](images/78c1d5c5897c5c2a1dfcacce22a2f9f815f55713830f46336bb6b7a1f2d5059d.jpg)  
(c) SnapKV,KV Size=128,acc 98.6   
(d) H2O,KV Size=128,acc 82.3   
Figure 17: Results of the Fact Retrieval Across Context Lengths (“Needle In A HayStack”) test in LlaMa-3-70B with 8k context size in 128 KV cache size. The vertical axis of the table represents the depth percentage, and the horizontal axis represents the token length. PyramidKV mitigates the negative impact of KV cache compression on the long-context understanding capability of LLMs.

# Q Attention Patterns across heads in the Bottom Layer

Retrieval heads are predominantly located in the higher layers. Notably, no retrieval heads are observed in bottom layers. To further investigate, we conducted additional experiments on the bottom layer to analyze the attention patterns of the heads as Figure 18. Our findings indicate the absence of "massive attention" in any individual head.

![](images/08ed1b3679994d970511528eb3e985751f03d2f9f24fcd527c69ef1a28226392.jpg)  
Figure 18: Attention patterns of retrieval-augmented generation across heads in the bottom layer in LlaMa.

# R PyramidKV Implementation at vLLM

To help compare the vLLM implementation with the vanilla dense attention backend in terms of throughput, we perform the experiment. We present the throughput comparison between the PyramidKV vLLM implementation and the vanilla dense attention backend in a setting where the inputs have varying context lengths without shared prefixes.

In Figure Figure 19, we plot the throughput of the LlaMa 8b model by varying length. We observe that relative throughput under compression decreases as the new input context length approaches the limit, causing new sequences to wait longer before being added to the decoding batch.

We find that allocating/releasing/moving/accessing very small chunks of memory may cause inefficiency and fragmentation in a naive implementation of PyramidKV at vLLM. As PyramidKV applies different allocation budgets for different layers. The top layers have less budget, while the bottom layers have more budget. The application of KV cache eviction with different budgets across layers at the standard paged attention frameworks (i.e., vLLM) is ineffective as it only reduces the cache size proportionally to the layer with the lowest compression rate, and all evictions beyond this rate merely increase cache fragmentation.

However, the problem could be solved by adapting paged attention to page out cache on a per-layer basis. We expand the block tables of each sequence to include block tables for each layer of the cache so that they can be retrieved for each layer’s KV cache during attention without the use of fixed memory offsets.

![](images/460692e2531cb949a6a6a96944971b3ff41bca9ed370ec9ea25a3987fe216ddc.jpg)  
Figure 19: Throughout performance of PyramidKV across different input context lengths using LlaMa-3-8b model.
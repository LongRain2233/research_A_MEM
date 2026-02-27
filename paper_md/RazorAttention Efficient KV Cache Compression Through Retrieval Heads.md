# RAZORATTENTION: EFFICIENT KV CACHE COMPRESSIONTHROUGH RETRIEVAL HEADS

A PREPRINT

Hanlin Tang*1, Yang Lin1, Jing Lin1, Qingsen Han1, Shikuan Hong1, Yiwu Yao1, and Gongyi Wang1

1Huawei Technologies Co., Ltd

# ABSTRACT

The memory and computational demands of Key-Value (KV) cache present significant challenges for deploying long-context language models. Previous approaches attempt to mitigate this issue by selectively dropping tokens, which irreversibly erases critical information that might be needed for future queries. In this paper, we propose a novel compression technique for KV cache that preserves all token information. Our investigation reveals that: i) Most attention heads primarily focus on the local context; ii) Only a few heads, denoted as retrieval heads, can essentially pay attention to all input tokens. These key observations motivate us to use separate caching strategy for attention heads. Therefore, we propose RazorAttention, a training-free KV cache compression algorithm, which maintains a full cache for these crucial retrieval heads and discards the remote tokens in non-retrieval heads. Furthermore, we introduce a novel mechanism involving a “compensation token” to further recover the information in the dropped tokens. Extensive evaluations across a diverse set of large language models (LLMs) demonstrate that RazorAttention achieves a reduction in KV cache size by over $70 \%$ without noticeable impacts on performance. Additionally, RazorAttention is compatible with FlashAttention, rendering it an efficient and plug-and-play solution that enhances LLM inference efficiency without overhead or retraining of the original model.

# 1 Introduction

Long-context large language models (LLMs) have significantly advanced capabilities in natural language processing across diverse tasks. However, the growth of the Key-Value (KV) cache under increasing input length has become the major bottleneck for deployment. There are been plenty of previous work designed to alleviate this problem by compressing the KV cache size, including quantization [1–3], token-dropping [4, 5], local attention [6, 7], etc.

![](images/046f1fb74555a28c5ffdc89b4ae935de2a2b41671f6b06f416587ae94f5ad7e1.jpg)

![](images/ee340b44fc277961e7c0f4ba54071d837afa2056f116a1a8eede9c76abc33d15.jpg)  
Figure 1: RazorAttention achieves comparable performance to the original model, even with $7 0 \%$ KV cache compressed. To demonstrate this, we tested Llama2-13B-64K [8] on the Needle in A Haystack benchmark [9].

<table><tr><td>Input context: &quot;DOD&#x27;s MILCON appropriations are used to fund the acquisition, construction, installation... 
Mary&#x27;s favorite number is 34251... Bob&#x27;s favorite number is 7690... reviewing project cost estimates.&quot;</td></tr><tr><td>Q1: &quot;What is Mary&#x27;s favourite number?&quot; 
Original model: &quot;Mary&#x27;s favorite number is 34251.&quot; √ 
H2O: &quot;Mary&#x27;s favorite number is not explicitly mentioned in the text provided.&quot; × 
SnapKV: &quot;Mary&#x27;s favorite number is 34251.&quot; √ 
RazorAttention: &quot;Mary&#x27;s favorite number is 34251.&quot; √</td></tr><tr><td>Q2: &quot;What is Bob&#x27;s favourite number?&quot; 
Original model: &quot;Bob&#x27;s favorite number is 7690.&quot; √ 
H2O: &quot;{X}SnapKV: &quot;{X} 
RazorAttention: &quot;Bob&#x27;s favorite number is 7690.&quot; √</td></tr></table>

Figure 2: Importance-based token-dropping methods cannot work when querying the less relevant information to the main theme. Here, we use an 8K document from LongBench [10] and add two sentences that are not relevant to the main theme. In this case, H2O discards tokens that are less relevant to the main theme, leading to failures in both Q1 and Q2. SnapKV discards tokens based on the first query, making it effective for Q1 but failing in subsequent queries like Q2. Only RazorAttention successfully outputs the exact information from the lengthy input even when we compress $7 0 \%$ of the KV cache.

One major direction for KV cache compression is to directly drop tokens deemed unimportant so far [4, 5, 11, 12]. These methods inherently assume that tokens considered unimportant will not be needed in future queries, which does not hold in practical scenarios. For instance, a user might request information that is not directly aligned with the main theme of the processed text, or engage in a multi-round conversation querying different segments from the context. In these cases, the importance-based token-dropping methods can lead to significant performance degradation since the actual information required by the query might be discarded if considered unimportant (see our example on Qwen1.5-7B-Chat [13] in Figure 2). This leads us to pose a critical question:

“Can we find a way to reduce the KV cache size without losing semantic information?”

In this work we address this problem from a novel perspective. Our investigation reveals that there exists a “retrieve and process” mechanism in LLMs when processing a long context. More specifically, LLMs can accurately recall the queried information from a lengthy input through certain group of attention heads, which we denote as “retrieval heads” (see Section 3.3 for definition). These heads are capable of concentrating most of their attention weights on the relevant information (w.r.t. the queries) and increasing the output probability for those words. Another important finding is that non-retrieval heads primarily focus on local context or the attention sink [5], which means these heads cannot effectively utilize all the semantic information from the input. Based on these important findings, we hypothesize that LLM runs the reasoning procedure on a “retrieve and process” basis. That says, the model first uses the retrieval heads to gather relevant information, and then non-retrieval heads to process the retrieved information and generate the final response. This motivates us to design separate caching strategies for different heads: For retrieval heads, we keep the KV cache unaltered; for the rest heads, we only cache recent tokens and attention sinks.

Beyond this, we notice that there still exists a certain accuracy gap when we directly discard all the remote tokens in the non-retrieval heads. Therefore for these non-retrieval heads, we designed a “compensation token” for compressing the dropped cache into one token, and proved that the accuracy degradation due to the truncated KV cache gets further improved with this compensation token. With retrieval heads and compensation tokens, we prove that our algorithm, namely RazorAttention, can successfully compress $7 0 \%$ of the KV cache without noticeable performance degradation as illustrated in Figure 1.

Last but not least, previous importance-based token-dropping methods cannot be combined with FlashAttention due to their reliance on the attention weights to compute the importance score, making them impracticable for implementation since FlashAttention is one of the most important components in long-context inference. RazorAttention addresses this problem since it does not use the attention map as the metric. The head-wise pruning criterion is totally compatible with FlashAttention, and the computation overhead of the compensation token is negligible. Therefore RazorAttention could achieve a substantial inference speedup when compared to previous methods.

To the best of our knowledge, RazorAttention is the first training-free token reduction algorithm that achieves a nearly lossless 3X KV cache reduction. We evaluated RazorAttention on models including Qwen [13], Llama-2 [14], Llama-3 [15] and Baichuan [16] on long-context tasks to prove its effectiveness. Our contribution can be summarized as follows:

• We systematically analyze the attention dynamic of Transformers under lengthy inputs. Our work reveals that only a few retrieval heads can essentially recall information from the whole input while the rest heads mainly focus on the local context.   
• We introduce a novel algorithm, namely RazorAttention, that is capable of reducing the KV cache size by $7 0 \%$ under minimal impact on performance for contexts ranging from 8K to 100K tokens. We designed an accurate and data-free metric for allocating all the retrieval heads, together with an error compensation strategy for compensating the information loss due to the truncated KV cache.   
• RazorAttention introduces negligible overhead in compression and is compatible with FlashAttention, rendering it an efficient and plug-and-play solution that enhances LLM inference efficiency without training or significant overhead. Extensive experiments demonstrate that RazorAttention can be effectively applied to various models and tasks.

# 2 Related Work

As the sequence length increases, the memory consumption of KV cache rapidly expands, potentially surpassing the size of the model parameters themselves. This leads to an urgent need for KV cache compression, particularly in scenarios with limited GPU memory. One direction is non-Transformer architecture design, such as Mamba [17], Mamba2 [18], Infini-Transformer [19], RWKV [20]and Griffin [21]. However, in this paper we focus on KV cache reduction for typical Transformers, which is the most widely used model structure. Below we introduce several approaches for KV cache compression.

Quantization Quantization is a classic yet effective approach to neural network compression. In the field of LLM Quantization, while the outlier challenge attracts great attention [22–24] to tackle, the application of which on KV cache is often seen as a by-product of activation quantization. Nevertheless, there are several noteworthy works demonstrating the value of KV cache quantization. FlexGen, Atom and QServe [1–3] carefully designed quantization pipelines that utilize KV cache compression to boost the overall inference throughput. KVQuant [25] integrates several techniques to minimize KV quantization error and KIVI [26] pushed the limit towards 2-bits. Besides the post-training methods, LLM-QAT [27] offers a data-free distillation process that further recovers the performance of the model.

Token-dropping Token-dropping methods assume that not all key-value pairs are essential in self-attention computations, so memory usage can be saved by identifying and removing unimportant KV cache. StreamingLLM [5] utilizes sliding window technology, preserving only the KV pairs of attention sink tokens and those within the sliding window, thereby reducing memory footprint and stabilizing model performance. H2O [4] is one of the pioneers that use the attention scores to evaluate the importance of each token, followed by an eviction strategy that greedily selects cache with higher scores. Scissorhands [11] and one of the latest work SnapKV [12] use similar ideas by narrowing the computation range to consider attention scores related to recent information. Built on that, PyramidKV and PyramidInfer [28, 29] analyze the attention concentration patterns and further reduce KV cache in later layers. Moreover, research efforts have been made to understand KV cache from different perspectives: FastGen [30] paid attention to special tokens and punctuation, SubGen [31] investigated the clusterability of key embedding and CORM [32] discovered strong correlation amongst tokens of near neighbors.

Non-MHA Attention Another category focuses on reducing KV cache by sharing cache across attention heads. MQA [33] aggressively uses a single KV head for all heads, whereas GQA [34] suggests an intermediate number of heads to balance the trade-off between inference speed and output quality. Furthermore, MLA [35] presents a novel caching method by low-ranking KV cache of all heads into single latent space.

Our algorithm is motivated by the idea from [36], where the authors noticed that there are certain groups of attention heads, denoted as the induction heads, that can effectively recall the queried information from the input. Recent study [37] also validated this property under extended inputs. This is the first work that proposes a head-wise pruning criterion for KV cache compression based on the interpretability of the attention mechanism.

# 3 Methodology

In this section, we introduce the key components of RazorAttention. We firstly apply RazorAttention to models using ALiBi [38] positional embedding (denoted as ALiBi models) to provide an intuitive understanding of the retrieval and non-retrieval heads. Afterwards, we demonstrate that models using RoPE [39] positional embedding (denoted as RoPE models) also exhibit this crucial characteristic, which reveal that KV cache within RoPE models can also be efficiently compressed under minimal loss of accuracy.

# 3.1 RazorAttention for ALiBi models

For ALiBi models, its $h$ -th attention head computes the attention score according to

$$
S _ {m \rightarrow n} (\boldsymbol {q}; \boldsymbol {k}) = \boldsymbol {q} _ {m} \boldsymbol {k} _ {n} ^ {\intercal} - l _ {h} (m - n), \tag {1}
$$

where $\mathbf { \delta } \mathbf { q } _ { m }$ is the query tensor at the $m$ -th position, $k _ { n }$ is the key tensor at the $n$ -th position, $l _ { h }$ is the head-specific slope, $S _ { m  n } ( q ; k )$ is the attention score. Notice that $( m \geq n )$ ) is guaranteed by the casualty of attention.

In the scenario where $l _ { h } ( m - n )$ significantly dominates $\mathbf { \Delta } q _ { m } k _ { n } ^ { \intercal }$ , the attention between $\mathbf { \nabla } \mathbf { q } _ { m }$ and $k _ { n }$ would decay to zero, meaning that the contribution of any tokens positioned further than $n$ becomes negligible for the output at position $m$ . The following theorem formalizes this observation.

Theorem 1. Given an attention head that calculates the attention score as per (1), for any $\epsilon \in ( 0 , 1 )$ , the attention weight from $\mathbf { \nabla } \mathbf { q } _ { m }$ to $k _ { n }$ can be upper bounded by:

$$
\begin{array}{l} A t t n _ {m \rightarrow n} (\boldsymbol {q}; \boldsymbol {k}) = \frac {\exp \left(S _ {m \rightarrow n} (\boldsymbol {q} ; \boldsymbol {k})\right)}{\sum_ {n = 0} ^ {m} \exp \left(S _ {m \rightarrow n} (\boldsymbol {q} ; \boldsymbol {k})\right)} \leq \epsilon , \quad \forall n <   m - C _ {0}, \\ L _ {h} := \frac {2 \| W _ {Q _ {h}} W _ {K _ {h}} \| _ {2} \left(\| \boldsymbol {\gamma} \| ^ {2} + \| \boldsymbol {b} \| ^ {2}\right) - \log (\epsilon)}{l _ {h}}. \tag {2} \\ \end{array}
$$

Here $W _ { Q _ { h } }$ and $W _ { K _ { h } }$ are the query and key matrices of the $h$ -th attention head, $\gamma$ and $^ { b }$ are the weight and bias for the LayerNorm layer before attention ( $\mathbf b = \mathbf 0$ for RMSNorm [40]), and $\| \cdot \| _ { 2 }$ denotes the $l _ { 2 }$ -norm of the matrix. $L _ { h }$ can be viewed as the vision scope of the head. The detailed proof can be found in Appendix A.

Theorem (1) indicates that when the distance between $\mathbf { \nabla } \mathbf { q } _ { m }$ and $k _ { n }$ exceeds $C _ { 0 }$ , the attention weight between these two tokens falls below $\epsilon$ . When $\epsilon$ is sufficiently small (e.g., $0 . 1 \%$ ), remote tokens impose minimal influence on the final output and can thus be discarded. Building on this principle, ALiBi models dynamically adjust the KV cache size for each head. We first compute the effective attention scope $L _ { h }$ , and keep only the recent $L _ { h }$ tokens in the KV cache, since any token further than $L _ { h }$ impose attention weight no more than $\epsilon$ , we can safely discard them for compression. Therefore, for ALiBi models, the retrieval heads are the ones with a larger $L _ { h }$ , while the non-retrieval heads has a smaller attention vision $L _ { h }$ .

# 3.2 RazorAttention for RoPE models

For RoPE models, each attention head computes the attention score according to

$$
S _ {m \rightarrow n} (\boldsymbol {q}; \boldsymbol {k}) = \boldsymbol {q} _ {m} \boldsymbol {k} _ {n} ^ {\intercal}, \quad \boldsymbol {q} _ {m} = \mathcal {R} _ {m} \boldsymbol {q}, \quad \boldsymbol {k} _ {n} = \mathcal {R} _ {n} \boldsymbol {k} \tag {3}
$$

where $\mathbf { \nabla } \mathbf { q } _ { m }$ and $k _ { n }$ are the query and key state after rotary transformation, $\mathcal { R } _ { m }$ and $\mathcal { R } _ { n }$ are the rotary matrices at position $m$ and $n$ (see [39] for details). Although RoPE embedding does not inherently suggest a long-range decaying attention, our empirical findings indicate that the majority of attention heads maintain a limited scope of attention. Notably, only about $15 \%$ of the heads, which we term as retrieval heads, are capable of effectively utilizing long-range information while the rest heads only focus on the local context. As shown in Table 1, a significant decrease in accuracy of $16 \%$ is observed when the KV cache size is reduced for these retrieval heads. Conversely, dropping remote tokens within non-retrieval heads results in a comparatively minor performance degradation of $1 . 5 \%$ .

Based on the findings above, we directly decrease the KV cache for all non-retrieval heads. The performance of the model is mostly retained as shown in Table 1.However, a notable accuracy gap remains, indicating that some information is still being lost. Moreover, the test result on Needle in a Haystack shows a clear performance degradation even when we protect the KV cache of retrieval heads (see our ablation result in Figure 6). To further improve performance, we designed a lightweight and effective way to compress the information in the dropped token into a “compensation token”.

The compensation token is defined as

$$
\hat {\boldsymbol {k}} = \frac {1}{N _ {d}} \sum_ {m \in \{\mathcal {D} \}} \boldsymbol {k} _ {m}, \quad \hat {\boldsymbol {v}} = \frac {1}{N _ {d}} \sum_ {m \in \{\mathcal {D} \}} \boldsymbol {v} _ {m}. \tag {4}
$$

<table><tr><td>Protection heads</td><td>All</td><td>Retrieval heads</td><td>Random heads</td><td>None</td></tr><tr><td>MultiFieldQA-en</td><td>46.94%</td><td>45.48%</td><td>40.7%</td><td>40.81%</td></tr></table>

Table 1: We protected the KV cache within different groups of attention heads while keeping only the recent 4K tokens in the rest. Protecting the KV cache in the retrieval heads can retain most of the LLM’s performance, while protecting randomly heads brings no performance gain. This clearly indicates that most of the attention heads only use local context and only retrieval heads can essentially utilize all context information.

Here $\hat { k }$ , $\hat { v }$ are the compensation tokens for the dropped KV cache, $\{ \mathcal { D } \}$ contains the indices of the dropped tokens and $N _ { d }$ is the number of the dropped tokens. Afterward, we discard the dropped tokens and augment the KV cache with the compensation token $\hat { k }$ and $\hat { v }$ , where $\{ K , V \}$ are the KV cache of the remaining token after rotary transformation. Denoting the compressed KV cache as $\{ K , { \hat { k } } \}$ and $\{ V , \hat { v } \}$ , the attention output of the current token follows

$$
\operatorname {A t t n} \left(\boldsymbol {q} _ {m}, \{K, \hat {\boldsymbol {k}} \}, \{V, \hat {\boldsymbol {v}} \} \}\right) = \frac {N _ {d} \exp \left(\boldsymbol {q} _ {m} \hat {\boldsymbol {k}} ^ {\intercal}\right) \hat {\boldsymbol {v}} + \sum_ {n \notin \{\mathcal {D} \}} \exp \left(\boldsymbol {q} _ {m} \boldsymbol {k} _ {n} ^ {\intercal}\right) \boldsymbol {v} _ {n}}{N _ {d} \exp \left(\boldsymbol {q} _ {m} \hat {\boldsymbol {k}} ^ {\intercal}\right) + \sum_ {n \notin \{\mathcal {D} \}} \exp \left(\boldsymbol {q} _ {m} \boldsymbol {k} _ {n} ^ {\intercal}\right)}. \tag {5}
$$

In Figure 3(a) we provide an illustrative example of RazorAttention for RoPE models. With compensation tokens, the accuracy is further improved, making RazorAttention almost lossless even dropping $7 0 \%$ of the KV cache in the non-retrieval heads. Below we introduce how we determine the retrieval heads group.

![](images/33aa99ba367c164b434e41dabdf032c4d7bb2ef04740926344b79aa1d690bc7b.jpg)  
(a)

![](images/54c6b7b3031743c3433bc0035231e393817fd83aef8e14fd2db3b4a73f003acb.jpg)  
(b)   
Figure 3: In Figure 3(a) we present the illustration of how RazorAttention compress the KV cache. For retrieval heads, we maintain a full cache for retaining all the tokens’ information. For non-retrieval heads, we directly discard remote tokens and compress the discarded tokens into a compensation token whose KV cache is denoted as $\{ \hat { k } , \hat { v } \}$ . In Figure 3(b) we provide an illustration example of the echo head and induction head. The current token is “B” and the generated token is “C”. In this case, the echo head would mainly attend to token “B” while the induction head mainly attend to token “C” in previous context.

# 3.3 Identification of Retrieval Heads

For ALiBi models, the attention scope can be directly determined via (2) and KV cache can be dropped accordingly. However, for RoPE models, the retrieval heads need to be identified in a more sophisticated way. Our investigation reveals that two groups of heads are essential in processing long context, so both of them should be included as retrieval heads as stated below.

• Echo head: The head tends to attends back to previous token (referred as echo token) identical to the current token.   
• Induction head: The head tends to attend to the previous token (namely induction token) that is immediately succeeded by the current token. Basically it attends to the coming token that also exists in previous context.

In Figure 3(b) we present an illustrative example explaining the echo heads and induction heads. In order to identify the retrieval heads, we generate $K$ (for example, $K = 2 5 0 0$ ) random tokens, repeat these tokens 4 times, and then use it as the input of the model. This design minimizes semantic dependencies among tokens, thereby allowing a clearer observation of the behavior of echo and induction heads.

Table 2: General hyper-parameter settings for experiments in the paper, which leads to 3.125x compression of KV cache under long context input.   

<table><tr><td>Hyper-parameter</td><td>Settings</td></tr><tr><td>Buffer length</td><td>max(4000, N/5)</td></tr><tr><td>Induction head protection</td><td>top 14%</td></tr><tr><td>Echo head protection</td><td>top 1%</td></tr><tr><td>Sink token num</td><td>4</td></tr></table>

Table 3: Performance comparison of RazorAttention and other compression algorithms across various LLMs on LongBench. Notice that the performance of Llama3-8B-Instruct on TREC and LSHT are not applicable (close to 0), hence we do not include their result on Llama3-8B.   

<table><tr><td>LLMs</td><td></td><td>NvivQA</td><td>Qespet</td><td>MF-en</td><td>MF-zh</td><td>HotpolQA</td><td>2WikimQA</td><td>Musique</td><td>GovReport</td><td>OMSum</td><td>MultiNews</td><td>VCSUM</td><td>TREC</td><td>TriviaQA</td><td>LSHT</td><td>Lcc</td><td>Average</td></tr><tr><td rowspan="4">Qwenl.5-7B-Chat</td><td>All KV</td><td>17.58</td><td>43.16</td><td>46.94</td><td>60.98</td><td>50.96</td><td>36.36</td><td>27.86</td><td>28.78</td><td>23.24</td><td>24.02</td><td>13.91</td><td>17.64</td><td>83.77</td><td>16.96</td><td>48.35</td><td>36.03</td></tr><tr><td>StreamingLLM</td><td>6.22</td><td>24.62</td><td>18.9</td><td>34.51</td><td>20.68</td><td>12.31</td><td>5.88</td><td>3.86</td><td>3.52</td><td>20.74</td><td>3.17</td><td>8.5</td><td>36.57</td><td>13</td><td>42.51</td><td>17.00</td></tr><tr><td>H2O</td><td>16.5</td><td>38.15</td><td>40.22</td><td>51.46</td><td>50.19</td><td>35.69</td><td>27.12</td><td>28.42</td><td>22.00</td><td>22.70</td><td>14.03</td><td>18.25</td><td>83.72</td><td>16.4</td><td>47.54</td><td>34.16</td></tr><tr><td>RA</td><td>16.63</td><td>43.1</td><td>46.66</td><td>61.08</td><td>50.49</td><td>36.1</td><td>28.79</td><td>26.68</td><td>22.59</td><td>23.96</td><td>13.83</td><td>20.87</td><td>83.83</td><td>15.66</td><td>47.85</td><td>35.87</td></tr><tr><td rowspan="4">Qwenl.5-72B-Chat</td><td>All KV</td><td>28.32</td><td>46.73</td><td>48.25</td><td>63.41</td><td>55.91</td><td>46.23</td><td>34.56</td><td>32.47</td><td>22.69</td><td>24.86</td><td>15.61</td><td>71.0</td><td>91.15</td><td>46</td><td>65.05</td><td>46.15</td></tr><tr><td>StreamingLLM</td><td>9.57</td><td>28.33</td><td>19.06</td><td>34.98</td><td>25.32</td><td>13.42</td><td>10.08</td><td>4.11</td><td>3.79</td><td>21.1</td><td>3.74</td><td>43.0</td><td>43.72</td><td>20.5</td><td>53.6</td><td>22.29</td></tr><tr><td>H2O</td><td>27.98</td><td>41.45</td><td>43.69</td><td>55.93</td><td>54.77</td><td>45.16</td><td>34.61</td><td>32.24</td><td>22.35</td><td>24.36</td><td>14.5</td><td>70.0</td><td>91.15</td><td>42</td><td>64.2</td><td>44.29</td></tr><tr><td>RA</td><td>27.97</td><td>46.44</td><td>47.36</td><td>63.04</td><td>55.92</td><td>46.15</td><td>34.36</td><td>32.35</td><td>22.75</td><td>24.91</td><td>15.17</td><td>71.0</td><td>91.49</td><td>46</td><td>64.68</td><td>45.97</td></tr><tr><td rowspan="4">Ljama3.8B-Instruct</td><td>All KV</td><td>21.84</td><td>37.04</td><td>45.07</td><td>52.34</td><td>44.63</td><td>27.28</td><td>23.04</td><td>28.18</td><td>24.54</td><td>26.26</td><td>14.41</td><td>0</td><td>85.90</td><td>3</td><td>30.17</td><td>35.44</td></tr><tr><td>StreamingLLM</td><td>0.61</td><td>16.29</td><td>13.41</td><td>20.05</td><td>2</td><td>5.84</td><td>0.37</td><td>5.22</td><td>4.63</td><td>18.89</td><td>2.52</td><td>-</td><td>11.54</td><td>-</td><td>26.83</td><td>9.86</td></tr><tr><td>H2O</td><td>21.14</td><td>34.1</td><td>40.84</td><td>47.13</td><td>43.47</td><td>27.13</td><td>21.31</td><td>22.85</td><td>16.36</td><td>22.3</td><td>14.52</td><td>-</td><td>86.17</td><td>-</td><td>30.26</td><td>32.89</td></tr><tr><td>RA</td><td>21.16</td><td>36.22</td><td>42.88</td><td>51.93</td><td>44.07</td><td>26.89</td><td>22.03</td><td>26.56</td><td>23.86</td><td>25.83</td><td>15.69</td><td>-</td><td>85.83</td><td>-</td><td>30.25</td><td>34.86</td></tr><tr><td rowspan="4">Baiqihan2-13B</td><td>All KV</td><td>18.63</td><td>30.16</td><td>44.1</td><td>50.36</td><td>37.93</td><td>32.62</td><td>13.90</td><td>26.16</td><td>20.14</td><td>24.58</td><td>15.66</td><td>62.5</td><td>86.61</td><td>27.5</td><td>55.36</td><td>36.41</td></tr><tr><td>StreamingLLM</td><td>5.12</td><td>12.44</td><td>23.53</td><td>32.52</td><td>16.93</td><td>16.08</td><td>6.15</td><td>5.53</td><td>1.03</td><td>5.6</td><td>3.94</td><td>42.22</td><td>30.15</td><td>7.32</td><td>35.42</td><td>16.27</td></tr><tr><td>H2O</td><td>17.81</td><td>29.89</td><td>43.74</td><td>49.54</td><td>37.02</td><td>31.71</td><td>13.54</td><td>25.8</td><td>18.96</td><td>23.31</td><td>15.11</td><td>62.41</td><td>85.25</td><td>26.86</td><td>54.45</td><td>35.69</td></tr><tr><td>RA</td><td>18.22</td><td>31.87</td><td>43.6</td><td>51.36</td><td>36.97</td><td>32.89</td><td>13.98</td><td>25.51</td><td>20.13</td><td>24.51</td><td>15.41</td><td>62.5</td><td>87.23</td><td>28</td><td>54.53</td><td>36.45</td></tr></table>

Subsequently, we calculated the echo score (attention weight to the echo token) and induction score (attention weight to the induction token) of all words across all heads. The selection of retrieval heads involves the top- $\cdot 1 4 \%$ attention heads with the highest induction score and top- $. 1 \%$ of attention heads with the highest echo score (see Table 2). Notice that although we only use much fewer echo heads than retrieval heads, our investigation indicates that both heads are crucial for the retrieving performance for LLMs (see Section 4.3 for ablation results).

With the retrieval heads being identified, we hereby introduce RazorAttention for RoPE Models in Algorithm 1.

# Algorithm 1 RazorAttention for RoPE Models

Input: Non-retrieval headset $\{ H \}$ , original KV cache (after rotary transformation) $\{ K , V \}$ , compression ratio $C$ , compression threshold $S _ { 0 }$ , sink token num $N _ { 0 }$ .

1: for non-retrieval head $h \in \{ H \}$ do   
2: Compute the buffer length $\begin{array} { r } { \dot { L _ { h } } = \operatorname* { m a x } \left( S _ { 0 } , \frac { N } { C } \right) } \end{array}$ , here $N$ is the number of tokens in the head.   
3: Keeping only the recent $L _ { h }$ tokens near output and first $N _ { 0 }$ sink tokens, discarding the remaining tokens and compress them into a compensation token according to (4).   
4: end for   
5: Non-retrieval heads compute attention according to (5), while retrieval heads follow the original attention.

Output: Generated output tokens.

# 4 Experiments

A variety of recent-released LLMs are selected to validate our proposals, including Qwen [13], Llama2 [14], Llama3 [15] and Baichuan [16]. The selected models are evaluated on Longbench [10] and Needle In A Haystack [9] to demonstrate their capabilities in long-context circumstances. The experiments are conducted on NVIDIA GeForce RTX 4090 (24GB). We will first validate the effectiveness of our proposal on various tasks, followed by the ablation study of each component in our algorithm design. Unless explicitly stated, we use RazorAttention with the hyper-parameters as

![](images/6c8ba7db48e6ffd7c4addc52ced97e86efdd3ecaaca0edba8e16b1a4a05aa505.jpg)

![](images/d78022782e11b498e120143e29c2c09bf8710c34221534b4f05054d84ad4d90f.jpg)

![](images/9c35d429e88462e9bec91901a6208dd92024754c816a00dcc6bd6e711f46c4f0.jpg)  
Figure 4: Performance comparison of RazorAttention and other compression algorithms on Llama2-7b-80K, Needle In A Haystack. Notice that H2O is incompatible with FlashAttention so we get OOM errors when tested on longer sequences, and its performance has already become unusable in this case.

in Table 2. We use H2O [4] and StreamingLLM [5] for comparison. Notice that we do not include SnapKV [12] as the baseline because it assumes that the query is known before compression, which does not hold in general cases or in a multi-round conversation where the user might query different information from the context (as discussed in Section 1).

# 4.1 LongBench Evaluation

In Table 3 we present the results of different algorithms on LongBench [10], which provides a comprehensive assessment to evaluate long-context related abilities of LLMs. We use Qwen1.5-7B and Qwen1.5-72B for testing since they are RoPE models with a context length of 32K. We also include Llama3-8B to validate the performance of RazorAttention on GQA models. We choose Baichuan2-13B to demonstrate the effectiveness of RazorAttention on ALiBi models. It can be seen that RazorAttention achieved a superior performance across all models compared to StreamingLLM and H2O. The compelling outcomes indicate that RazorAttention can achieve comparable performance as the uncompressed baseline, even under 3X compression ratio.

Moreover, we test Llama3-8B-Instruct as a GQA instance where every 4 attention heads share a single set of KV cache. Hence, we consider the attention heads in a group as all retrieval if one or more heads satisfy inductive or echoing property. The results in Table 3 clearly prove that RazorAttention still work for GQA models.

![](images/bf25bb791ae3da963838f3fa5222d58efdd8f5d7ff398e70d79af0ddf1132a91.jpg)

![](images/f1444e97a24fae75143ce57bfabfbf4ee9bf42816f3ec5bba34ec760747ad7ba.jpg)  
Figure 5: Adding $1 \%$ of the echo heads can significantly enhances the retrieving performance of RazorAttention on Llama2-7B-80k.

# 4.2 Needle In A Haystack Evaluation

In Figure 4 we present the results on Needle In A Haystack. We use Llama2-7B-80K from [8] since the context length of this model is 80K. Unlike H2O, whose performance is severely degraded under long inputs, RazorAttention can still accurately recall the queried information. This is a strong evidence proving that RazorAttention can retain all the semantic information within the original context, while importance-based methods inevitably discard information that might be useful in future queries.

# 4.3 Ablation Studies

Below we present the ablation results of RazorAttention, and prove that the algorithm design and configuration are optimally chosen to achieve a higher compression ration with acceptable performance degradation.

<table><tr><td>Protection scheme</td><td>Score</td></tr><tr><td>1% Echo + 5% Induction Head</td><td>69.54%</td></tr><tr><td>1% Echo + 8% Induction Head</td><td>78.40%</td></tr><tr><td>1% Echo + 11% Induction Head</td><td>84.55%</td></tr><tr><td>1% Echo + 14% Induction Head</td><td>86.59%</td></tr><tr><td>Baseline</td><td>87.05%</td></tr></table>

Table 4: Qwen1.5-7B-Chat using RazorAttention with different numbers of heads protected, tested on Needle in A Haystack.

# 4.3.1 Importance of Echo Heads

Although we only include $1 \%$ echo heads in RazorAttention, we notice that this group of heads is quite essential in retrieving information under long context as shown in Figure 5. One possible explanation is that the induction heads depend on the existence of echo heads as discussed in [36].

# 4.3.2 Number of Induction Heads

To determine the optimal number of induction heads to use in RazorAttention, in Table 4 we present the accuracy of RazorAttention under various numbers of induction heads. The results show that the accuracy improves continuously with an increasing number of induction heads. We decide to include $1 4 \%$ of the induction heads in order to achieve an optimal balance between the compression ratio and model performance.

![](images/634c8b7abe144f6a920da80b4ee3245cf411371d42c51ecbe1d420a792842b99.jpg)

![](images/919f9d0cbb8b822e7736c92cb4ec5494450c8a25148790e9387b34566eb7bb30.jpg)  
Figure 6: The compensation token is critical for recovering the information loss introduced by the truncated KV cache.

# 4.3.3 Importance of the Compensation Token

In Figure 6, it is clearly demonstrated that compensation tokens are critical for the performance of RazorAttention. The compensation tokens successfully compressed most of the information from the dropped tokens,thereby maintaining high accuracy even with significant KV cache reduction.

# 5 Conclusion

In this paper, we propose RazorAttention, a novel KV cache compression algorithm, which successfully achieves a 3X compression ratio for models use RoPE or ALiBi embeddings. Unlike previous importance-based token-dropping methods which inevitably discard semantic information, RazorAttention preserves all semantic information within retrieval heads. We demonstrate that remote tokens can be effectively compressed into compensation tokens within non-retrieval heads. Furthermore, our head-wise pruning criterion is fully compatible with FlashAttention, making RazorAttention a plug-and-play compression method that accelerates the inference of LLMs under extended context. Our experiments demonstrate that RazorAttention can achieve comparable performance with the original model and surpasses previous methods in both accuracy and efficiency.

# 6 Limitation

However, there are still certain limitations of our work. The first question is why attention heads in LLMs behave so differently and how retrieval heads operate under lengthy inputs. The second challenge lies in achieving a higher compression ratio. Although we have successfully reduced the KV cache by $7 0 \%$ , we believe this number can be further

improved. Moreover, although we have tested our algorithm on several models, the optimal configuration on other models might be different, meaning that we might need more or less retrieval heads under different cases. These topics are quite important and we will keep investigating them in the future work.

# References

[1] Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Re, Ion Stoica, and Ce Zhang. FlexGen: High-throughput generative inference of large language models with a single GPU. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 31094–31116. PMLR, 23–29 Jul 2023.   
[2] Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci. Atom: Low-bit quantization for efficient and accurate llm serving, 2024.   
[3] Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, and Song Han. Qserve: W4a8kv4 quantization and system co-design for efficient llm serving, 2024.   
[4] Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2o: Heavy-hitter oracle for efficient generative inference of large language models. In Thirty-seventh Conference on Neural Information Processing Systems, 2023.   
[5] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks, 2024.   
[6] Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023.   
[7] Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with sparse transformers, 2019.   
[8] Yao Fu, Rameswar Panda, Xinyao Niu, Xiang Yue, Hannaneh Hajishirzi, Yoon Kim, and Hao Peng. Data engineering for scaling language models to 128k context, 2024.   
[9] gkamradt. Needle In A Haystack - Pressure Testing LLMs, 2023.   
[10] Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. Longbench: A bilingual, multitask benchmark for long context understanding, 2023.   
[11] Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. Scissorhands: Exploiting the persistence of importance hypothesis for llm kv cache compression at test time, 2023.   
[12] Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. Snapkv: Llm knows what you are looking for before generation, 2024.   
[13] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report, 2023.   
[14] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023.

[15] AI@Meta. Llama 3 model card. 2024.   
[16] Baichuan. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023.   
[17] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2024.   
[18] Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality, 2024.   
[19] Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. Leave no context behind: Efficient infinite context transformers with infini-attention, 2024.   
[20] Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Stella Biderman, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Jiaju Lin, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Guangyu Song, Xiangru Tang, Bolun Wang, Johan S. Wind, Stanislaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Qinghua Zhou, Jian Zhu, and Rui-Jie Zhu. Rwkv: Reinventing rnns for the transformer era, 2023.   
[21] Soham De, Samuel L. Smith, Anushan Fernando, Aleksandar Botev, George Cristian-Muraru, Albert Gu, Ruba Haroun, Leonard Berrada, Yutian Chen, Srivatsan Srinivasan, Guillaume Desjardins, Arnaud Doucet, David Budden, Yee Whye Teh, Razvan Pascanu, Nando De Freitas, and Caglar Gulcehre. Griffin: Mixing gated linear recurrences with local attention for efficient language models, 2024.   
[22] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant: Accurate and efficient post-training quantization for large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 38087–38099. PMLR, 23–29 Jul 2023.   
[23] Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. Outlier suppression: Pushing the limit of low-bit transformer language models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 17402–17414. Curran Associates, Inc., 2022.   
[24] Xiuying Wei, Yunchen Zhang, Yuhang Li, Xiangguo Zhang, Ruihao Gong, Jinyang Guo, and Xianglong Liu. Outlier suppression+: Accurate quantization of large language models by equivalent and optimal shifting and scaling, 2023.   
[25] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization, 2024.   
[26] Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. Kivi : Plug-and-play 2bit kv cache quantization with streaming asymmetric quantization. 2023.   
[27] Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. Llm-qat: Data-free quantization aware training for large language models. arXiv preprint arXiv:2305.17888, 2023.   
[28] Zefan Cai., Yichi Zhang, Bofei Gao, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Baobao Chang, Junjie Hu, and Wen Xiao. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling, 2024.   
[29] Dongjie Yang, XiaoDong Han, Yan Gao, Yao Hu, Shilin Zhang, and Hai Zhao. Pyramidinfer: Pyramid kv cache compression for high-throughput llm inference, 2024.   
[30] Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. Model tells you what to discard: Adaptive kv cache compression for llms, 2024.   
[31] Amir Zandieh, Insu Han, Vahab Mirrokni, and Amin Karbasi. Subgen: Token generation in sublinear time and memory, 2024.   
[32] Jincheng Dai, Zhuowei Huang, Haiyun Jiang, Chen Chen, Deng Cai, Wei Bi, and Shuming Shi. Sequence can secretly tell you what to discard, 2024.   
[33] Noam Shazeer. Fast transformer decoding: One write-head is all you need, 2019.   
[34] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints, 2023.   
[35] DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin,

Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jin Chen, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Size Zheng, T. Wang, Tian Pei, Tian Yuan, Tianyu Sun, W. L. Xiao, Wangding Zeng, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, X. Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Liu, Xin Xie, Xingkai Yu, Xinnan Song, Xinyi Zhou, Xinyu Yang, Xuan Lu, Xuecheng Su, Y. Wu, Y. K. Li, Y. X. Wei, Y. X. Zhu, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Zheng, Yichao Zhang, Yiliang Xiong, Yilong Zhao, Ying He, Ying Tang, Yishi Piao, Yixin Dong, Yixuan Tan, Yiyuan Liu, Yongji Wang, Yongqiang Guo, Yuchen Zhu, Yuduan Wang, Yuheng Zou, Yukun Zha, Yunxian Ma, Yuting Yan, Yuxiang You, Yuxuan Liu, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhewen Hao, Zhihong Shao, Zhiniu Wen, Zhipeng Xu, Zhongyu Zhang, Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, and Ziwei Xie. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model, 2024.   
[36] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads, 2022.   
[37] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality, 2024.   
[38] Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation, 2022.   
[39] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023.   
[40] Biao Zhang and Rico Sennrich. Root mean square layer normalization, 2019.

# A Appendix: Proof of Theorem 1

Below we first give an upper bound for the product of the queries and keys, and then show that the attention weight would decay to zero when the positional bias is significantly larger than that upper bound.

Proof. Since we have $\pmb q = W _ { Q _ { h } } \pmb x$ and $\pmb { k } = W _ { K _ { h } } \pmb { x }$ where $_ { \textbf { \em x } }$ is the input of the Attention block, this leads to

$$
\boldsymbol {q} \boldsymbol {k} ^ {\intercal} = \boldsymbol {x} W _ {Q _ {h}} W _ {K _ {h}} \boldsymbol {x} ^ {\intercal} \leq \| W _ {Q _ {h}} W _ {K _ {h}} \| _ {2} \| \boldsymbol {x} \| ^ {2}. \tag {6}
$$

Since $_ { \textbf { \em x } }$ is attained after LayerNorm, which means

$$
\boldsymbol {x} = \gamma \odot \frac {\hat {x} - \mu}{\sigma} + \boldsymbol {b},
$$

$$
\mu = \frac {1}{d} \sum_ {i = 1} ^ {d} \hat {x} _ {i}, \quad \sigma = \frac {1}{d} \sum_ {i = 1} ^ {d} (\hat {x} _ {i} - \mu) ^ {2}.
$$

Here $\hat { x }$ is the input of LayerNorm, $d$ is its dimension and ${ \hat { x } } _ { i }$ is the $i$ -th dimension of $\hat { x }$ . The equation above leads to

$$
\begin{array}{l} \left\| \boldsymbol {x} \right\| ^ {2} = \left\| \boldsymbol {\gamma} \odot \frac {\hat {x} - \mu}{\sigma} + \boldsymbol {b} \right\| ^ {2} \\ \leq 2 \left\| \boldsymbol {\gamma} \odot \frac {\hat {x} - \mu}{\sigma} \right\| ^ {2} + 2 \| \boldsymbol {b} \| ^ {2} \\ \leq 2 \| \boldsymbol {\gamma} \| ^ {2} + 2 \| \boldsymbol {b} \| ^ {2}. \tag {7} \\ \end{array}
$$

Combining (6) and (7) we get

$$
\boldsymbol {q} \boldsymbol {k} ^ {\intercal} \leq \left\| W _ {Q _ {h}} W _ {K _ {h}} \right\| _ {2} \left(2 \| \boldsymbol {\gamma} \| ^ {2} + 2 \| \boldsymbol {b} \| ^ {2}\right) \tag {8}
$$

In order to give an upper bound for the attention weight, we have

$$
\begin{array}{l} \operatorname {A t t n} _ {m \rightarrow n} (\boldsymbol {q}; \boldsymbol {k}) = \frac {\exp \left(S _ {m \rightarrow n} (\boldsymbol {q} ; \boldsymbol {k})\right)}{\sum_ {n = 0} ^ {m} \exp \left(S _ {m \rightarrow n} (\boldsymbol {q} ; \boldsymbol {k})\right)} \\ = \frac {\exp \left(\boldsymbol {q} \boldsymbol {k} ^ {\intercal} - l _ {h} (m - n)\right)}{\sum_ {n = 0} ^ {m} \exp \left(S _ {m \rightarrow n} (\boldsymbol {q} ; \boldsymbol {k})\right)} \\ \leq \frac {\exp \left(\boldsymbol {q} \boldsymbol {k} ^ {\intercal} - l _ {h} (m - n)\right)}{\exp \left(S _ {n \rightarrow n} (\boldsymbol {q} ; \boldsymbol {k})\right)} \\ \leq \frac {\exp \left(\boldsymbol {q} \boldsymbol {k} ^ {\intercal} - l _ {h} (m - n)\right)}{\exp \left(\boldsymbol {q} \boldsymbol {q} ^ {\intercal}\right)} \\ \leq \exp \left(\boldsymbol {q} \boldsymbol {k} ^ {\intercal} - l _ {h} (m - n)\right) \\ = \frac {\exp \left(\boldsymbol {q} \boldsymbol {k} ^ {\intercal}\right)}{\exp \left(l _ {h} (m - n)\right)}. \\ \end{array}
$$

Therefore to ensure $\begin{array} { r } { \mathrm { A t t } \mathfrak { n } _ { m \to n } \left( \pmb q ; \pmb { k } \right) \leq \epsilon } \end{array}$ , which is equivalent as $\begin{array} { r } { \log \left( \operatorname { A t t n } _ { m \to n } \left( \pmb { q } ; \pmb { k } \right) \right) \leq \log ( \epsilon ) } \end{array}$ , we need

$$
\log \left(\operatorname {A t t n} _ {m \rightarrow n} (\boldsymbol {q}; \boldsymbol {k})\right) \leq \boldsymbol {q} \boldsymbol {k} ^ {\intercal} - l _ {h} (m - n) \leq \log (\epsilon)
$$

Taking (8) into the equation above, we get

$$
\left\| W _ {Q _ {h}} W _ {K _ {h}} \right\| _ {2} \left(2 \| \boldsymbol {\gamma} \| ^ {2} + 2 \| \boldsymbol {b} \| ^ {2}\right) - l _ {h} (m - n) \leq \log (\epsilon),
$$

which gives us

$$
m - n \geq \frac {2 \| W _ {Q _ {h}} W _ {K _ {h}} \| _ {2} \left(\| \boldsymbol {\gamma} \| ^ {2} + \| \boldsymbol {b} \| ^ {2}\right) - \log (\epsilon)}{l _ {h}}.
$$

In this case, we have

$$
\operatorname {A t t n} _ {m \rightarrow n} (\boldsymbol {q}; \boldsymbol {k}) \leq \epsilon .
$$

![](images/776a43674e5e2cc773ae6223b161ce86b2e6159ef8c713cfd7663dcdca47e134.jpg)
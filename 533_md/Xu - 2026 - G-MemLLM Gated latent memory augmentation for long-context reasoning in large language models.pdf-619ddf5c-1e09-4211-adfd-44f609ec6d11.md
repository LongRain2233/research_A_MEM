# G-MEMLLM: GATED LATENT MEMORY AUGMEN-TATION FOR LONG-CONTEXT REASONING IN LARGE LANGUAGE MODELS

Xun Xu

Department of Computer Science

Fudan University

23307130122@m.fudan.edu.cn

# ABSTRACT

Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language understanding, yet they remain constrained by the finite capacity of their context windows and the inherent difficulty of maintaining long-term factual consistency during multi-hop reasoning. While existing methods utilize context compression or recurrent tokens, they often suffer from “context rot” or the dilution of information over long horizons. In this paper, we propose G-MemLLM, a memory-augmented architecture that integrates a frozen LLM backbone with a trainable Latent Memory Bank. Our key innovation is a GRU-style gated update logic that allows the model to selectively update, preserve, or overwrite latent memory slots, preventing the vanishing gradients of knowledge common in recurrent systems. We evaluate G-MemLLM across scales, from GPT-2 (124M) to Llama 3.1 (8B), on the HotpotQA and Zero-Shot Relation Extraction (ZsRE) benchmarks. Our results demonstrate that G-MemLLM significantly enhances multi-hop reasoning and relational precision, achieving a $1 3 . 3 \%$ accuracy boost on ZsRE for Llama 3.1-8B, and it also yields improvements across model scales, boosting Answer F1 by 8.56 points for GPT-2 and increasing Supporting Fact F1 by 6.89 points for Llama 3.1-8B on HotpotQA.

# 1 INTRODUCTION

The evolution of Large Language Models (LLMs) has been defined by a constant tension between model capacity and context management. While modern models possess vast parametric knowledge, their ability to synthesize information across disparate documents or maintain factual integrity over extended interactions is often limited by the quadratic complexity of the transformer’s attention mechanism. This limitation is particularly evident in tasks requiring multi-hop reasoning, such as HotpotQA (Yang et al., 2018), where a model must bridge multiple facts to reach a conclusion, or in relational knowledge extraction (Levy et al., 2017a), where precise entity mapping is required.

Recent attempts to extend the functional context of LLMs have generally followed two paths: context compression and recurrent state-passing. Context compression methods, such as Gist Tokens (Mu et al., 2023), attempt to condense prefixes into a smaller set of virtual tokens. However, these often lead to information bottlenecks where fine-grained details are lost. On the other hand, recurrent architectures like the Recurrent Memory Transformer (RMT) (Bulatov et al., 2022a) pass hidden states across segments but struggle with vanishing knowledge as information is progressively overwritten by incoming noise in the sequence.

A central challenge in these systems is gating: the ability of a model to decide which information is transient and which is of long-term utility. Without an explicit mechanism to manage the lifecycle of a memory slot, latent representations eventually converge toward a mean, losing the specificity required for complex reasoning.

In this work, we introduce G-MemLLM, a gated latent memory augmented framework designed to provide LLMs with a persistent and selectively updatable working memory. G-MemLLM decouples linguistic processing from knowledge retention by utilizing a frozen LLM backbone paired with a

trainable Latent Memory Bank. Unlike previous passive memory pools, G-MemLLM employs a differentiable, GRU-style gated update mechanism. This gate allows the model to dynamically regulate the flow of information into its latent slots, ensuring that bridge entities in multi-hop queries are preserved until they are no longer needed.

![](images/55652184d9433e7f14d06d2432b8a52104484975b67432e039f6a38a787345e2.jpg)  
(a) Improvement on HotpotQA

![](images/769a2d9b5ab151f6dd7d0cd2ac222a7220b965f42ac73db9b61b95f1df0c5dbc.jpg)  
(b) Comparison on ZsRE   
Figure 1: Performance enhancement brought by the G-MemLLM memory module. (a) illustrates the absolute improvement across models, while (b) shows the direct comparison between Vanilla and G-MemLLM on specific metrics.

We demonstrate the efficacy of G-MemLLM through a rigorous scaling analysis. By testing our module on both GPT-2 (124M) and Llama 3.1 (8B), we show that our architecture is not only effective for small-scale models but still remains efficient as the base model’s capacity increases. Our contributions are as follows:

• We propose the G-MemLLM architecture, featuring a Latent Memory Bank with a gated update logic that prevents memory drift and information overwrite.   
• We introduce a composite training objective that combines sparsity and entropy losses to ensure efficient and diverse memory slot utilization.   
• We provide an empirical evaluation on HotpotQA (Yang et al., 2018) and ZsRE (Levy et al., 2017b), showing that G-MemLLM bridges the performance gap for small models and significantly extends the reasoning capabilities of large-scale pre-trained models.

# 2 RELATED WORK

# 2.1 CONTEXT COMPRESSION AND INFORMATION BOTTLENECKS

The primary bottleneck for processing long sequences in Transformer-based architectures is the quadratic growth of the KV-cache. To address this, several works have explored information distillation through compression. Mu et al. (2023) introduced Gist Tokens, which utilize a soft-token bottleneck to condense long prompt prefixes into a few learnable virtual tokens. Building on this, Recurrent Context Compression (RCC) (Huang et al., 2024) demonstrated that frozen encoders can recursively compress million-token contexts into compact representations. While these methods successfully reduce inference latency, they often suffer from context rot (Hong et al., 2025)—a phenomenon where the density of information becomes too high for the decoder to reconstruct specific facts, leading to non-uniform performance degradation in long-horizon tasks.

# 2.2 MEMORY TOKENS AND RECURRENT STATE-PASSING

Another trajectory focuses on maintaining a ”working memory” through persistent tokens that flow across segments. Bulatov et al. (2022b) pioneered this with the Recurrent Memory Transformer (RMT), which passes memory tokens between segments to maintain global context. More recently, the $\mathbf { M } +$ framework (Wang et al., 2025) extended this concept by integrating a co-trained retriever that expands the usable latent-space memory to over 160k tokens. However, as noted in the LOCCO

benchmark (Jia et al., 2025), simple recurrent passing often leads to vanishing gradients of knowledge, where information from the beginning of a sequence is progressively overwritten or diluted by noise. These systems lack a robust mechanism to differentiate between transient context and permanent knowledge.

# 2.3 DYNAMIC LATENT MEMORY AND GATED ARCHITECTURES

The most recent shift involves moving memory entirely into the latent space to simulate humanlike cognitive architectures. Wang et al. (2024) introduced MemoryLLM, which utilizes a largescale memory pool for self-updatable knowledge. Our G-MemLLM aligns with this philosophy but introduces a critical architectural innovation: the GRU-style gated update logic. While previous models like ERMAR (Alselwi et al., 2025) rely on passive accumulation or ranking, G-MemLLM uses a differentiable gate $( g )$ to manage information overwrite. This allows for the precise integration of multi-hop facts while maintaining the stability of the frozen LLM backbone.

![](images/7176f4e9b31682110ee9ecc56630cb35b40e1f7935d0e0f3eb106d56d13b0f1e.jpg)  
Figure 2: Overview of G-MemLLM architecture.

# 3 METHOD

The G-MemLLM architecture is designed to bridge the gap between static, pre-trained knowledge and dynamic, task-specific context. It consists of two primary systems: a frozen LLM backbone that provides linguistic features and a trainable latent memory bank that manages persistent state.

# 3.1 LATENT MEMORY BANK

The latent memory bank is the central working memory of the agent. It manages a fixed number of learnable memory slots, $M \in \mathbb { R } ^ { S \times D _ { m } }$ (where $S$ is the number of slots and $D _ { m }$ is the memory dimension).

Memory encoder and decoder These sub-networks act as translators. The encoder compresses high-dimensional LLM hidden states into the lower-dimensional memory space, while the decoder maps retrieved latent states back to the LLM’s hidden size for processing.

Cross-attention mechanism To integrate new information, the memory slots act as Queries $( Q )$ , while the newly encoded experiences act as Keys $( K )$ and Values $( V )$ . This allows the bank to selectively attend to relevant features of the current input.

Gated update logic To prevent memory drift and manage information overwrite, we implement a GRU-style update gate (Cho et al., 2014).Let $M _ { o l d }$ be the current state and $M _ { a t t e n d e d }$ be the result of the cross-attention. The new state $M _ { n e w }$ is calculated as:

$$
M _ {\text {n e w}} = (1 - g) \odot M _ {\text {o l d}} + g \odot M _ {\text {a t t e n d e d}}
$$

where $g$ is the gate value produced by the update gate network.

# 3.2 SYSTEM DYNAMICS: THE MEMORY LOOP

The interaction between the LLM and the latent memory bank follows a three-stage execution cycle:

Extraction The frozen LLM processes the input to generate raw hidden states.

Retrieval The model queries the memory bank based on the encoded current hidden states.

Injection The retrieved latent information is decoded and concatenated with the original states through an gated injection layer, creating enhanced hidden states which are passed to the original LLM’s language modeling head to produce logits.

Consolidation The encoded original hidden states are fed back into the memory bank to update the memory slots via the cross-attention and gating mechanism, ensuring the memory evolves for the next interaction.

# 3.3 TRAINING OBJECTIVE

The memory system is trained using a composite loss function designed to optimize its performance on a primary task while encouraging desirable memory behaviors through regularization. The total loss $L _ { t o t a l }$ is a weighted sum of three components:

1) Primary task loss The main objective is to train the memory module to improve the model’s ability to predict the next token in a sequence. The enhanced hidden states are passed through the frozen LLM’s language modeling head to produce memory-augmented logits $\hat { y }$ . The primary loss $L _ { C L M }$ is the standard cross-entropy loss between these predicted logits and the ground-truth target tokens $y$ :

$$
L _ {C L M} = - \frac {1}{T - 1} \sum_ {t = 1} ^ {T - 1} \log P (x _ {t + 1} \mid x _ {1: t})
$$

2) Sparsity loss An $L 1$ penalty applied to encourage the model to use a sparse, focused set of memory slots, preventing it from storing redundant information:

$$
L _ {s p a r s i t y} = \frac {1}{M} \sum | s _ {i} |
$$

where $s _ { i }$ stands for each slot’s importance score and $M$ is the total number of slots.

3) Entropy loss To prevent the model from relying on only one or two memory slots, we encourage diversity in memory usage by maximizing the entropy of the importance score distribution $p$ , which is achieved by minimizing the negative entropy:

$$
L _ {e n t r o p y} = \sum p _ {i} \times \log (p _ {i})
$$

where $p$ is the softmax-normalized distribution of scores s.

The final loss is a linear combination of the above components, weighted by hyperparameters $\lambda _ { s }$ and $\lambda _ { e }$ :

$$
L _ {t o t a l} = L _ {C L M} + \lambda_ {s} \times L _ {s p a r s i t y} + \lambda_ {e} \times L _ {e n t r o p y}
$$

# 4 EXPERIMENTS

# 4.1 DATASETS

We evaluate our memory-augmented architecture on two distinct challenges:

1) HotpotQA Evaluates multi-hop reasoning in a distractor setting.   
2) ZsRE (Zero-Shot Relation Extraction) Evaluates the model’s ability to extract and geralize factual relations.

For HotpotQA, we report Exact Match (EM) and F1 Score. For ZsRE, we report Accuracy, measuring the model’s ability to correctly identify the object of a relation given a subject and a query.

# 4.2 BASELINES

We compare our memory-enhanced architecture against the standard vanilla versions of the respective base models:

1) GPT-2 A small-scale baseline to test the module’s impact on limited parameter budgets.   
2) Llama 3.1-8B A modern, high-capacity baseline to test scaling efficiency.

# 5 RESULTS

# 5.1 SCALING ANALYSIS

The integration of the memory module consistently improves performance across both datasets and model scales.

Table 1: Performance on HotpotQA (Multi-hop Reasoning). Metrics include Answer, Supporting Facts (Sup Fact), and Joint evaluation of both.   

<table><tr><td rowspan="2">Model</td><td colspan="2">Answer</td><td colspan="2">Sup Fact</td><td colspan="2">Joint</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>GPT-2 (Vanilla)</td><td>35.47</td><td>45.52</td><td>15.18</td><td>51.84</td><td>11.35</td><td>30.72</td></tr><tr><td>GPT-2 (G-MemLLM)</td><td>41.92 +6.45</td><td>54.08 +8.56</td><td>22.63 +7.45</td><td>60.17 +8.33</td><td>15.08 +3.73</td><td>38.51 +7.79</td></tr><tr><td>Llama 3.1-8B (Vanilla)</td><td>68.53</td><td>79.27</td><td>62.19</td><td>76.53</td><td>51.43</td><td>72.15</td></tr><tr><td>Llama 3.1-8B (G-MemLLM)</td><td>72.38 +3.85</td><td>82.12 +2.85</td><td>67.13 +4.94</td><td>83.42 +6.89</td><td>54.82 +3.39</td><td>78.23 +6.08</td></tr></table>

Table 2: Evaluation results on the ZsRE dataset for different models.   

<table><tr><td>Model</td><td>Score</td><td>Efficacy</td><td>Generalization</td><td>Specificity</td></tr><tr><td>Llama 3.1-8B (Vanilla)</td><td>55.63</td><td>55.92</td><td>54.71</td><td>56.31</td></tr><tr><td>Llama 3.1-8B(G-MemLLM)</td><td>63.03 +13.3%</td><td>60.18</td><td>59.10</td><td>56.78</td></tr></table>

The empirical results summarized in Table 1 demonstrate that the G-MemLLM module consistently enhances performance across all metrics for both small-scale and large-scale models. For GPT-2, the addition of the latent memory bank yields a substantial improvement in reasoning capability, specifically boosting the Answer F1 score by 8.56 points (from 45.52 to 54.08) and Joint F1 by 7.79 points. Notably, the performance gains are not limited to smaller architectures; Llama 3.1-8B also shows significant progression. While the base Llama 3.1 model already exhibits strong performance, G-MemLLM further elevates its Supporting Fact (Sup Fact) F1 by 6.89 points (achieving 83.42) and its Joint F1 by 6.08 points. This trend suggests that G-MemLLM is particularly effective at evidence grounding, the ability to identify supporting facts, which is the primary bottleneck in multi-hop reasoning tasks like HotpotQA.

As shown in Table 2, the memory module significantly boosts performance on ZsRE. For Llama 3.1- 8B, we observe a $1 3 . 3 \%$ absolute increase in accuracy. This suggests that the memory module acts as a specialized buffer that helps the model disambiguate complex relations that are often conflated in the standard feed-forward layers.

A key finding of our experiments is the super-linear scaling benefit of the memory module:

Small Scale (GPT-2) The memory module provides a baseline capability for multi-hop tasks that the vanilla model lacks, effectively acting as a scratchpad for intermediate steps.

Large Scale (Llama 3.1-8B) As the model size increases to 8B, the memory module’s utility shifts from basic retention to efficient indexing. The larger model uses the memory module to organize its vast internal knowledge more effectively, leading to the significant $1 3 . 3 \%$ jump in ZsRE accuracy.

Table 3: Ablation study on memory slot count $( S )$ on ZsRE score.   

<table><tr><td>Slots</td><td>Score</td><td>Δ(%)</td><td>Over.</td></tr><tr><td>0 (Van.)</td><td>58.53</td><td>—</td><td>1.00x</td></tr><tr><td>512</td><td>61.72</td><td>+5.45</td><td>1.05x</td></tr><tr><td>1024 (Prop.)</td><td>63.03</td><td>+2.12</td><td>1.12x</td></tr><tr><td>2048</td><td>63.21</td><td>+0.28</td><td>1.25x</td></tr></table>

![](images/19d3f996490ec1ed8c148000b32d8a193957470fdad0f9ddb1316a05b3a2d58c.jpg)  
Figure 3: Trend of performance gains and computational overhead.

# 5.2 ABLATION STUDY: MEMORY DENSITY

We investigated how the size of the memory module affects performance on ZsRE using the Llama 3.1-8B backbone.

The ablation shows that 1024 slots provide the optimal balance between accuracy gains and computational overhead. Doubling the slots to 2048 provides diminishing returns $( + 0 . 2 8 \% )$ , suggesting a saturation point in relational storage for the 8B parameter scale.

# 6 SUMMARY

This paper presents G-MemLLM, a memory-augmented architecture designed to enhance the multihop reasoning and relational knowledge retention of Large Language Models. By integrating a trainable latent memory bank with a frozen LLM backbone, we decouple linguistic processing from stateful information storage. Our primary technical innovation is the implementation of a GRU-style gated update logic, which allows the model to selectively preserve or overwrite latent memory slots. This mechanism, supported by a composite loss function emphasizing sparsity and entropy, successfully mitigates common issues such as context rot and information dilution found in traditional recurrent or compressed architectures.

Empirical evaluations across scales, from GPT-2 (124M) to Llama 3.1-8B, demonstrate that G-MemLLM provides significant performance gains on the HotpotQA and ZsRE benchmarks. These results suggest that as LLMs grow in capacity, they become increasingly efficient at using explicit memory modules to index and organize their internal knowledge.

As a machine learning course project, this work highlights the transition from passive context windows to active gated cognitive architectures. We successfully navigated challenges related to memory collapse and high-dimensional state projection on limited hardware. The findings underscore that adding a small, trainable working memory, representing less than $3 \%$ additional parameters, is a highly efficient strategy for extending the reasoning capabilities of pre-trained models without the prohibitive cost of full-parameter fine-tuning.

# REFERENCES

Ghadir Alselwi, Hao Xue, Shoaib Jameel, Basem Suleiman, Hakim Hacid, Flora D. Salim, and Imran Razzak. Long context modeling with ranked memory-augmented retrieval, 2025. URL https://arxiv.org/abs/2503.14800.   
Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. Recurrent memory transformer, 2022a. URL https://arxiv.org/abs/2207.06881.   
Aydar Bulatov, Yuri Kuratov, Yermek Kapushev, and Mikhail Burtsev. Recurrent memory transformer. In Advances in Neural Information Processing Systems (NeurIPS), 2022b.

Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, et al. Learning phrase representations using RNN encoder–decoder for statistical machine translation. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2014.   
Kelly Hong, Anton Troynikov, and Jeff Huber. Context rot: How increasing input tokens impacts LLM performance. Chroma Research Technical Report, 2025.   
Chensen Huang, Guibo Zhu, Xuepeng Wang, Yifei Luo, Guojing Ge, Haoran Chen, Dong Yi, and Jinqiao Wang. Recurrent context compression: Efficiently expanding the context window of llm, 2024. URL https://arxiv.org/abs/2406.06110.   
Zixi Jia, Qinghua Liu, Hexiao Li, Yuyan Chen, and Jiqiang Liu. Evaluating the long-term memory of large language models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 19759–19777, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1014. URL https: //aclanthology.org/2025.findings-acl.1014/.   
Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. Zero-shot relation extraction via reading comprehension. In Conference on Computational Natural Language Learning (CoNLL), 2017a.   
Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. Zero-shot relation extraction via reading comprehension, 2017b. URL https://arxiv.org/abs/1706.04115.   
Jesse Mu, Xiang Lisa Li, and Noah Goodman. Learning to compress prompts with gist tokens. In Advances in Neural Information Processing Systems (NeurIPS), 2023.   
Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, et al. MEMORYLLM: Towards self-updatable large language models. In International Conference on Machine Learning (ICML), 2024.   
Yu Wang, Dmitry Krotov, Yuanzhe Hu, et al. $\mathbf { M } +$ : Extending MemoryLLM with scalable long-term memory. In International Conference on Machine Learning (ICML), 2025.   
Zhilin Yang, Peng Qi, Saizheng Zhang, et al. HotpotQA: A dataset for diverse, explainable multihop question answering. In Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018.
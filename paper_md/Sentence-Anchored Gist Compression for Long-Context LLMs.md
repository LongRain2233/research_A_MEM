# Sentence-Anchored Gist Compression for Long-Context LLMs

Dmitrii Tarasov

FusionBrainLab

HSE University

tarasov@fusionbrainlab.com

Elizaveta Goncharova

FusionBrainLab

HSE University

Kuznetsov Andrey

FusionBrainLab

Innopolis University

# Abstract

This work investigates context compression for Large Language Models (LLMs) using learned compression tokens to reduce the memory and computational demands of processing long sequences. We demonstrate that pretrained LLMs can be fine-tuned to compress their context by factors of $2 \mathbf { x }$ to ${ } ^ { 8 \mathrm { { X } } }$ without significant performance degradation, as evaluated on both short-context and long-context benchmarks. Furthermore, in experiments on a 3-billion-parameter LLaMA model, our method achieves results on par with alternative compression techniques while attaining higher compression ratios.

# 1 Introduction

The growing demand for processing long-context inputs in Large Language Models (LLMs) is often bottlenecked by memory and computational requirements of the self-attention mechanism. In a recent paper Kuratov et al. (2025) showed that LLMs possess a significant capacity for compression, with an 8B model capable of condensing up to 1568 tokens into a single vector in a prompt-tuning fashion (smaller models achieve lower rates, e.g., 128 tokens per vector).

We argue that this inherent ability can be extended beyond prompt-tuning. Instead of a single compression token, we propose learning a set of compression tokens that pool the most salient information from distinct text segments. These tokens then serve as the conditioning context for subsequent segments during language modeling, effectively creating an information bottleneck, where each segment primarily attends to the compressed representation of the previous one. This architecture can be implemented with minimal modifications to the standard transformer: by extending the model’s vocabulary to include the new compression tokens and adjusting the attention mask to enforce the segment-conditioning pattern. This stands in

contrast to methods like Zhang et al. (2024) proposed, which, while powerful, originally required Backpropagation Through Time (BPTT) — though a mask-based implementation is also feasible. A key advantage of our setup is that the model learns to compress the context end-to-end using only the standard Language Modeling (LM) objective, eliminating the need for an auxiliary reconstruction loss as proposed by Deng et al. (2024) to improve uniform gist-based compression.

Furthermore, while preceding works typically employ a uniform distribution of compression tokens (Zhang et al., 2024; Deng et al., 2024), we introduce a rule-based strategy for token insertion at the end of each sentence. This data-dependent positioning aims to align compression boundaries with natural semantic units, thereby facilitating more meaningful and coherent information aggregation. In summary, our contributions are as follows:

• We propose a novel, rule-based strategy for positioning compression tokens to enable datadependent and semantically-aware context compression.   
• We implement the proposed method via a simple attention mask modification, which enables efficient parallel processing during both training and prefilling, enhancing scalability.   
• We extensively evaluate the proposed framework and demonstrate that it maintains strong performance with no significant degradation on both short- and long-context benchmarks.   
• We achieve high KV-cache compression rates, ranging from $2 \mathbf { x }$ to 8x, across various evaluation benchmarks.

# 2 Related Work

Sparse Attention. Sparse attention methods aim to reduce computational costs by limiting the num-

ber of tokens to which each token can attend. In Native Sparse Attention Yuan et al. (2025) proposed a hardware-aligned hierarchical sparse attention that combines coarse-grained token compression with fine-grained token selection. The Forgetting Transformer (Lin et al., 2025) learns to limit local attention spans via precomputed forgetting scores.

Recurrence. Recurrent methods compress context into a fixed state for long-range dependency modeling. The Recurrent Memory Transformer (Bulatov et al., 2022) learns memory embeddings in an RNN-like fashion, significantly reducing memory requirements. AutoCompressors (Chevalier et al., 2023) extend this idea by learning to generate compression tokens that are used as soft prompts. More recently, the In-context AutoEncoder (Ge et al., 2024b) uses an autoregressive encoder to compress context into learned tokens for a decoder, achieving up to 4x compression without BPTT. While effective, many recurrent approaches require Backpropagation Through Time, which can slow training and hinder parallelization.

KV-Cache Compression. These methods reduce the memory footprint of the Key-Value cache during generation. Cai et al. (2025) uses redundancy estimation and importance scoring to retain approximately $10 \%$ of the cache. Li et al. (2024) employs clustering of attention patterns for an $8 . 2 \mathrm { x }$ memory reduction. Zhang et al. (2023) evicts tokens based on attention scores, reducing the cache by up to 5x. Wang et al. (2024) merges cache entries via clustering, achieving 3x compression. FastGen (Ge et al., 2024a) profiles and applies head-specific strategies (e.g., punctuation, locality) to halve the cache budget. StreamingLLM (Xiao et al., 2023) uses attention sinks and a sliding window ( 1000 tokens) for streaming applications. SepLLM (Chen et al., 2024) extends this by caching separator tokens (e.g., punctuation), showing it generalizes StreamingLLM and improves performance, though it requires task-specific window tuning.

Gist Token Compression. This line of work introduces learned “gist” or “beacon” tokens to summarize context. The most related to ours is Zhang et al. (2024), which processes context in chunks with interleaved beacon tokens to accumulate information. In contrast, our method requires only a single forward pass for the prefill stage and employs data-dependent compression at sentence boundaries. Mu et al. (2023) compresses task-specific

prompts into gist tokens via a modified attention mask, achieving up to 26x compression. Deng et al. (2024) explores different compression rates and token placement strategies, using an autoencoding loss to improve compression quality.

# 3 Sentence Transformer

The proposed Sentence transformer takes inspiration from gist or beacon token compression (Zhang et al., 2024; Deng et al., 2024) and data-dependent compression from SepLLM (Chen et al., 2024).

# 3.1 Learned Gist Tokens

We extend the LLM’s embedding vocabulary by adding $N _ { g }$ new gist tokens. These tokens are initialized by sampling from a multivariate normal distribution whose parameters $( \mu , \Sigma )$ are derived from the existing vocabulary embeddings, following the “mean-resizing” approach by Hewitt (2021). The language modeling head is resized correspondingly. When the LM head is tied to the input embeddings (as is common in smaller models (e.g., 1–3B parameters)) we resize the shared matrix jointly; otherwise, the two matrices are resized separately.

# 3.2 Gist Tokens Placement

The gist tokens in this work serve as aggregators for some text segments. This work employs a simple, rule-based strategy for positioning compression tokens. Specifically, gist tokens are inserted at the end of each sentence, anchored to standard sentence-ending punctuation marks such as ‘.’, ‘!’, and ‘?’. This approach aims to align compression boundaries with natural semantic units.

# Processing Example For $N _ { g } = 2$ gist tokens:

Original: The sun was shining brightly. Birds were singing in the forest.

Processed: The sun was shining brightly. $< g _ { 1 } > < g _ { 2 } >$ Birds were singing in the forest. $\cdot$

# 3.3 Attention Mask

Within the transformer architecture, the gist tokens for a given sentence are allowed to attend to all tokens within that same sentence, as well as to all gist tokens from preceding sentences. This enables the aggregation of both local sentence-level information and global context from the compressed history. Figure 1 illustrates the sentence attention pattern, which exhibits sparsity despite an increased

![](images/1b7f2ef5c348a05438aa5c4a332051d1f4fad6d4f023a675df0dcf4282585e3c.jpg)  
(a) Causal attention

![](images/686ea06c2c5c82917fab97e2945af84dd730f02ca313aae73ca2a624d6119695.jpg)  
(b) Sentence attention   
Figure 1: Comparison of attention mechanisms: (a) causal attention, and (b) sentence attention with $N _ { g } = 1$ gist token. Gist tokens ${ \bf \Xi } ( \mathrm { ~  ~ \theta ~ } )$ are inserted at sentence boundaries and pool information from their entire sentence. They are visible to all subsequent tokens, while regular tokens $( t _ { i } )$ only attend within their sentence.

sequence length. This expansion occurs because gist tokens are added to the initial sequence.

# 3.4 Objective

The training methodology relies exclusively on the conventional language modeling objective. Given an input sequence $X = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { T } \}$ and compressed context representation $C$ , we optimize:

$$
\mathcal {L} = - \mathbb {E} _ {x \sim \mathcal {D}} \left[ \sum_ {t = 1} ^ {T} \log P (x _ {t} | x _ {<   t}, C) \right] \tag {1}
$$

where $C = f _ { \theta } ( \boldsymbol { X } )$ is the compressed context produced by our model parameters $\theta$ . This streamlined approach removes the dependency on additional reconstruction losses used by Deng et al. (2024) to optimize uniform gist compression.

# 3.5 Training Stages

Our experimental results demonstrated that initializing training with gist token optimization alone yields superior benchmark performance compared to end-to-end training from the outset. This finding led us to adopt a structured three-phase training approach:

Stage 1: Gist Token Warm-up In this initial phase, we freeze all base model parameters and train only the newly introduced gist tokens. This allows the model to learn somewhat contextcompression patterns without interfering with the pre-trained representations.

Stage 2: Full Model Fine-tuning Next, we unfreeze all parameters and conduct standard finetuning across the entire architecture. This stage leverages the initialized compression behavior from Stage 1 while adapting the base model to effectively utilize the compressed representations.

Stage 3: Large-Batch Cold Down To further enhance model convergence and stability, we implement a final training stage with significantly increased batch sizes up to 1M tokens with linear decay learning rate schedule following Zhai et al. (2021).

For our primary experiments, we utilized Llama3.2-3B (Grattafiori et al., 2024) as the base architecture, applying this staged training protocol across multiple downstream tasks and evaluation benchmarks.

# 4 Experiments

For Llama3.2-3B (Grattafiori et al., 2024) model we trained 4 checkpoints with corresponding 1, 2, 4, 8 gist tokens compression.

For all stages we used randomly sampled subsets of FineWeb-Edu (Lozhkov et al., 2024) total tokens budget for all training stages is approximately 4B tokens. Other training details can be found in Appendix A.

# 4.1 Short-Context benchmarks

First, we evaluate the model on short-context benchmarks: HellaSwag (Zellers et al., 2019), ARC (Clark et al., 2018), MMLU (Hendrycks et al., 2021), and WinoGrande (Sakaguchi et al., 2019). These benchmarks probe general linguistic and academic abilities, allowing us to assess whether shortcontext compression affects performance. During Stage 2 of training (Section 3.5), accuracy on these tasks saturates quickly and closely matches that of the vanilla base model. Results are reported in Table 1.

On short benchmarks, compression rates range from 16.55 for HellaSwag to 82.96 for WinoGrande with $N _ { g } = 1$ . These rates decrease by approximately half with each doubling of $N _ { g }$ Detailed compression rates across benchmarks are provided in Appendix D.

# 4.2 Long-Context Benchmarks

For long-context evaluation, we employed the HEL-MET benchmark (Yen et al., 2025). To enable

Table 1: Evaluation results on short-context benchmarks for the Llama3.2-3B model. $N _ { g }$ denotes the number of gist tokens per sentence. HS - HellaSwag, WG - WinoGrande. MMLU cloze.   

<table><tr><td>Model</td><td>ARC</td><td>HS</td><td>MMLU</td><td>WG</td></tr><tr><td colspan="5">Llama-3.2-3B Base</td></tr><tr><td>Vanilla</td><td>59.21</td><td>70.90</td><td>42.50</td><td>65.11</td></tr><tr><td>Sentence</td><td>Llama-3.2-3B</td><td>Base</td><td></td><td></td></tr><tr><td>Ng=1</td><td>49.77</td><td>63.63</td><td>38.25</td><td>58.96</td></tr><tr><td>Ng=2</td><td>53.64</td><td>66.61</td><td>38.90</td><td>61.01</td></tr><tr><td>Ng=4</td><td>54.59</td><td>68.07</td><td>39.73</td><td>61.48</td></tr><tr><td>Ng=8</td><td>55.17</td><td>67.86</td><td>39.89</td><td>61.17</td></tr></table>

rapid evaluation during training, we created a reduced version, HELMET (Tiny), consisting of 100 samples per task. The performance results on this benchmark are presented in Table 2. For most tasks, performance on HELMET (Tiny) improved steadily as the number of gist tokens $( N _ { g } )$ increased. Also our model maintained performance comparable to the baseline. We compared our model with strong 7B models: SepLLM (Chen et al., 2024) (trainingfree method) and activation beacon (Zhang et al., 2024), the most close to ours work. Despite being half the size of the compared models, our method achieves performance comparable strong baselines while attaining higher compression rates. For Sentence Llama with $N _ { g } = 4$ , the average compression rate across all long-context benchmarks is approximately $6 \times$ (see Appendix D for task-specific rates). This compares favorably with the Activation Beacon model, which achieves only $2 \times \mathrm { K V }$ -cache compression. The complete set of evaluation metrics for HELMET (Tiny) is provided in Appendix B.

# 4.3 Punctuation Sensitivity

Our analysis revealed that the model’s performance is sensitive to punctuation in benchmark templates. The icl benchmark exhibited the most pronounced sensitivity, where the addition of a single period (‘.’) nearly doubled the measured performance. Without this final period, the model incorrectly compresses a question’s label into gist tokens associated with the subsequent question, thereby degrading task performance.

# HELMET ICL template modification:

# Original:

What is swap math ?

label: 4

When does the average teenager first

Table 2: Sentence Attention Comparison on HELMET (tiny) with SepCache (Chen et al., 2024) and Beacon Compression (Zhang et al., 2024). $N _ { g }$ denotes the number of gist tokens per sentence.   

<table><tr><td>Model</td><td>recall</td><td>icl</td><td>longqa</td><td>cite</td></tr><tr><td colspan="5">Llama-3.2-3B Base</td></tr><tr><td>Vanilla</td><td>100.0</td><td>68.2</td><td>37.3</td><td>30.0</td></tr><tr><td colspan="5">Sentence Llama-3.2-3B Base</td></tr><tr><td>Ng=1</td><td>1.0</td><td>28.2</td><td>36.5</td><td>17.6</td></tr><tr><td>Ng=2</td><td>30.0</td><td>67.4</td><td>31.8</td><td>17.7</td></tr><tr><td>Ng=4</td><td>90.0</td><td>69.6</td><td>32.0</td><td>20.4</td></tr><tr><td>Ng=8</td><td>95.0</td><td>63.0</td><td>34.5</td><td>22.4</td></tr><tr><td colspan="5">Llama-3.1-8B-Instruct</td></tr><tr><td>Vanilla</td><td>100.0</td><td>15.0</td><td>29.3</td><td>34.9</td></tr><tr><td>SepCache</td><td>10.0</td><td>15.8</td><td>26.7</td><td>23.5</td></tr><tr><td colspan="5">Qwen2-7B-Instruct</td></tr><tr><td>Vanilla</td><td>100.0</td><td>71.2</td><td>31.4</td><td>37.4</td></tr><tr><td>Beacon</td><td>88.0</td><td>64.2</td><td>27.4</td><td>32.1</td></tr></table>

have intercourse ?

label: 5

# With extra dot:

What is swap math ?

label: 4 4.

When does the average teenager first have intercourse ?

label: 5 5.

# 5 Limitations

Despite achieving significant compression rates on long-context benchmarks, our approach has several limitations that warrant discussion:

• Rule-based token placement: The current method relies on rule-based gist token positioning. Future work should explore learned compression token locations for more adaptive context aggregation.   
• Fixed compression budget: Using a fixed number of gist tokens per segment limits flexibility. Dynamic allocation of compression tokens based on content complexity would be more efficient.   
• Punctuation sensitivity: As discussed in Section 4.3, the rule-based placement strategy in-

troduces sensitivity to punctuation variations in benchmark templates.

• Performance gap: The compressed model does not fully recover the performance of the base model without compression.   
• Implementation inefficiency. The current implementation materializes the full attention mask, which becomes memory-prohibitive for very long contexts (e.g., 128K tokens).   
• Limited model size. All ablation studies and experiments were conducted on a 3Bparameter model; scaling to larger models is left to future work.

# References

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlícek,ˇ Agustín Piqueres Lajarín, Vaibhav Srivastav, and 1 others. 2025. Smollm2: When smol goes big–datacentric training of a small language model. arXiv preprint arXiv:2502.02737.   
Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. 2022. Recurrent memory transformer. Advances in Neural Information Processing Systems, 35:11079– 11091.   
Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng Li, Yeyang Zhou, Li-Wen Chang, Jiuxiang Gu, and 1 others. 2025. Rkv: Redundancy-aware kv cache compression for training-free reasoning models acceleration. arXiv preprint arXiv:2505.24133.   
Guoxuan Chen, Han Shi, Jiawei Li, Yihang Gao, Xiaozhe Ren, Yimeng Chen, Xin Jiang, Zhenguo Li, Weiyang Liu, and Chao Huang. 2024. Sepllm: Accelerate large language models by compressing one segment into one separator. arXiv preprint arXiv:2412.12094.   
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3829–3846, Singapore. Association for Computational Linguistics.   
Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. ArXiv, abs/1803.05457.

Chenlong Deng, Zhisong Zhang, Kelong Mao, Shuaiyi Li, Xinting Huang, Dong Yu, and Zhicheng Dou. 2024. A silver bullet or a compromise for full attention? a comprehensive study of gist token-based context compression. arXiv preprint arXiv:2412.17483.   
Suyu Ge, Yunan Zhang, Liyuan Liu, Minjia Zhang, Jiawei Han, and Jianfeng Gao. 2024a. Model tells you what to discard: Adaptive KV cache compression for LLMs. In The Twelfth International Conference on Learning Representations.   
Tao Ge, Hu Jing, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024b. In-context autoencoder for context compression in a large language model. In The Twelfth International Conference on Learning Representations.   
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.   
Nathan Habib, Clémentine Fourrier, Hynek Kydlícek,ˇ Thomas Wolf, and Lewis Tunstall. 2023. Lighteval: A lightweight framework for llm evaluation.   
Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. Preprint, arXiv:2009.03300.   
John Hewitt. 2021. Initializing new word embeddings for pretrained language models.   
Yuri Kuratov, Mikhail Arkhipov, Aydar Bulatov, and Mikhail Burtsev. 2025. Cramming 1568 tokens into a single vector and back again: Exploring the limits of embedding space capacity. Preprint, arXiv:2502.13063.   
Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. Snapkv: Llm knows what you are looking for before generation. Advances in Neural Information Processing Systems, 37:22947–22970.   
Zhixuan Lin, Johan Obando-Ceron, Xu Owen He, and Aaron Courville. 2025. Adaptive computation pruning for the forgetting transformer. arXiv preprint arXiv:2504.06949.   
Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In International Conference on Learning Representations.   
Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. 2024. Fineweb-edu: the finest collection of educational content.

Jesse Mu, Xiang Li, and Noah Goodman. 2023. Learning to compress prompts with gist tokens. Advances in Neural Information Processing Systems, 36:19327– 19352.   
Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. 2020. Compressive transformers for long-range sequence modelling. In International Conference on Learning Representations.   
Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2019. Winogrande: An adversarial winograd schema challenge at scale. Preprint, arXiv:1907.10641.   
Zheng Wang, Boxiao Jin, Zhongzhi Yu, and Minjia Zhang. 2024. Model tells you where to merge: Adaptive kv cache merging for llms on long-context tasks. arXiv preprint arXiv:2407.08454.   
Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2023. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453.   
Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and Danqi Chen. 2025. Helmet: How to evaluate longcontext language models effectively and thoroughly. In International Conference on Learning Representations (ICLR).   
Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Yuxing Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. 2025. Native sparse attention: Hardwarealigned and natively trainable sparse attention. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23078–23097, Vienna, Austria. Association for Computational Linguistics.   
Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4791–4800, Florence, Italy. Association for Computational Linguistics.   
Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. 2021. Scaling vision transformers. in 2022 ieee. In CVF Conference on Computer Vision and Pattern Recognition (CVPR), volume 4.   
Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. 2024. Long context compression with activation beacon. arXiv preprint arXiv:2401.03462.   
Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, and 1 others. 2023. H2o: Heavy-hitter oracle for efficient

Table 3: Training hyperparameters across different stages.   

<table><tr><td></td><td>Stage 1</td><td>Stage 2</td><td>Stage 3</td></tr><tr><td>Max Seq Len</td><td>4096</td><td>4096</td><td>4096</td></tr><tr><td>Tokens</td><td>0.1B</td><td>2B</td><td>2B</td></tr><tr><td>Batch Size</td><td>64</td><td>128</td><td>512</td></tr><tr><td>Training Time</td><td>3 hours</td><td>30 hours</td><td>30 hours</td></tr><tr><td>Optim. Steps</td><td>1000</td><td>9000</td><td>2000</td></tr><tr><td>Max LR</td><td>1e-4</td><td>1e-4</td><td>5e-5</td></tr><tr><td>Min LR</td><td>5e-5</td><td>5e-5</td><td>0</td></tr><tr><td>Warmup Steps</td><td>100</td><td>1000</td><td>100</td></tr><tr><td>LR Schedule</td><td>Cosine</td><td>Cosine</td><td>Linear</td></tr><tr><td>MaxGradNorm</td><td>1.0</td><td>2.0</td><td>2.0</td></tr></table>

generative inference of large language models. Advances in Neural Information Processing Systems, 36:34661–34710.

# A Training Details

All training stages employed the AdamW optimizer (Loshchilov and Hutter, 2019) with weight decay 0.1, betas $( \beta _ { 1 } ~ = ~ 0 . 9 , \beta _ { 2 } ~ = ~ 0 . 9 5 )$ ), and epsilon $1 \times 1 0 ^ { - 8 }$ . Training was conducted in bfloat16 precision on a single node with 8 NVIDIA A100 GPUs. Table 3 summarizes the hyperparameters used across different training stages, including processed tokens, batch size, optimization steps, learning rate, warmup steps, learning rate schedule type, gradient clipping threshold, and training duration.

# B Evaluation Across Training Stages

Table 4 presents the evaluation results on shortcontext benchmarks across all training stages. We observe rapid performance saturation, with only marginal improvements as the number of gist tokens $( N _ { g } )$ increases.

Table 5 presents the evaluation results on the HELMET (Tiny) benchmark. The initial training stage resulted in performance degradation across most tasks, while the second stage yielded substantial improvements. The third stage provided only marginal additional gains. Notably, on the rerank task, our Sentence Attention model with $N _ { g } = 8$ significantly outperformed the base model, achieving an NDCG $@ 1 0$ of 7.32 compared to 0.40 for the base Llama3.2 3B model.

Table 4: Short benchmarks evaluation across training stages. Evaluation results on short-context benchmarks for the Llama3.2-3B model. $N _ { g }$ denotes the number of gist tokens per sentence. HS - HellaSwag, WG - WinoGrande, MMLU cloze version was evaluated   

<table><tr><td>Ng</td><td>ARC</td><td>HS</td><td>MMLU</td><td>WG</td><td>PG19</td></tr><tr><td colspan="6">Base</td></tr><tr><td>-</td><td>59.21</td><td>70.90</td><td>42.50</td><td>65.11</td><td>9.64</td></tr><tr><td colspan="6">Stage 1. Gist Embeddings only</td></tr><tr><td>1</td><td>37.65</td><td>55.59</td><td>33.06</td><td>57.46</td><td>27.06</td></tr><tr><td>2</td><td>40.94</td><td>58.85</td><td>36.45</td><td>57.77</td><td>19.24</td></tr><tr><td>4</td><td>47.82</td><td>61.70</td><td>36.55</td><td>56.99</td><td>15.68</td></tr><tr><td>8</td><td>51.42</td><td>61.77</td><td>37.37</td><td>58.01</td><td>10.48</td></tr><tr><td colspan="6">Stage 2. Finetune</td></tr><tr><td>1</td><td>47.16</td><td>62.65</td><td>37.40</td><td>58.96</td><td>13.65</td></tr><tr><td>2</td><td>51.46</td><td>65.75</td><td>38.08</td><td>59.51</td><td>11.98</td></tr><tr><td>4</td><td>54.64</td><td>67.34</td><td>39.45</td><td>60.93</td><td>9.52</td></tr><tr><td>8</td><td>53.01</td><td>67.11</td><td>38.39</td><td>61.25</td><td>7.58</td></tr><tr><td colspan="6">Stage 3. Cold down</td></tr><tr><td>1</td><td>49.77</td><td>63.63</td><td>38.25</td><td>58.96</td><td>12.83</td></tr><tr><td>2</td><td>53.64</td><td>66.61</td><td>38.90</td><td>61.01</td><td>11.29</td></tr><tr><td>4</td><td>54.59</td><td>68.07</td><td>39.73</td><td>61.48</td><td>9.24</td></tr><tr><td>8</td><td>55.17</td><td>67.86</td><td>39.89</td><td>61.17</td><td>7.17</td></tr></table>

Table 5: HELMET Tiny evaluation results across training stages.   

<table><tr><td>Ng</td><td>recall</td><td>rerank</td><td>cite</td><td>longqa</td><td>icl</td></tr><tr><td colspan="6">Vanilla</td></tr><tr><td>-</td><td>100</td><td>0.40</td><td>30.02</td><td>37.34</td><td>68.20</td></tr><tr><td colspan="6">Stage 1. Gist Embeddings only</td></tr><tr><td>1</td><td>0.00</td><td>0.00</td><td>13.13</td><td>12.93</td><td>2.00</td></tr><tr><td>2</td><td>0.00</td><td>0.00</td><td>15.50</td><td>20.72</td><td>0.00</td></tr><tr><td>4</td><td>0.00</td><td>0.00</td><td>17.57</td><td>23.64</td><td>0.00</td></tr><tr><td>8</td><td>0.00</td><td>0.12</td><td>20.64</td><td>30.55</td><td>15.40</td></tr><tr><td colspan="6">Stage 2. Finetune</td></tr><tr><td>1</td><td>1.00</td><td>0.00</td><td>17.56</td><td>36.50</td><td>28.20</td></tr><tr><td>2</td><td>30.00</td><td>2.34</td><td>17.74</td><td>31.81</td><td>67.40</td></tr><tr><td>4</td><td>90.00</td><td>2.05</td><td>20.43</td><td>31.96</td><td>69.60</td></tr><tr><td>8</td><td>95.00</td><td>5.80</td><td>22.38</td><td>34.53</td><td>63.00</td></tr><tr><td colspan="6">Stage 3. Finetune</td></tr><tr><td>1</td><td>3.00</td><td>0.18</td><td>17.87</td><td>33.94</td><td>32.80</td></tr><tr><td>2</td><td>46.00</td><td>2.00</td><td>18.09</td><td>34.45</td><td>71.20</td></tr><tr><td>4</td><td>96.00</td><td>3.06</td><td>20.86</td><td>33.36</td><td>63.20</td></tr><tr><td>8</td><td>98.00</td><td>7.31</td><td>22.94</td><td>33.49</td><td>64.60</td></tr></table>

# C Evaluation Details

# C.1 Short-Context Benchmarks

For the short-context benchmarks evaluation we used lighteval (Habib et al., 2023) with the same configuration as in SmolLM2 (Allal et al., 2025). For MMLU we used cloze formatting and chose the option with minimal perplexity. All other benchmarks were also evaluated based on the perplexity of the correct answers. There were no generative benchmarks in this subset.

# C.2 HELMET Tiny

All long-context benchmarks in the HELMET evaluation suite are generative: the model produces a limited number of tokens, and the generated output is then compared against a ground-truth answer. For the HELMET Tiny subset, we selected a small number of samples per task, with sequence lengths ranging from 4K to 8K tokens. Table 6 provides detailed information for each task, including maximum sequence length and sample count.

Table 6: HELMET Tiny benchmark configuration: task specifications and dataset statistics.   

<table><tr><td>Task</td><td>Max Length</td><td>Samples</td></tr><tr><td>recall</td><td>4k</td><td>100</td></tr><tr><td>rerank</td><td>8k</td><td>100</td></tr><tr><td>cite</td><td>8k</td><td>100</td></tr><tr><td>longqa</td><td>8k</td><td>100</td></tr><tr><td>icl</td><td>8k</td><td>500</td></tr></table>

# C.3 PG19 Perplexity Evaluation

On PG19 (Rae et al., 2020), higher compression yields lower overall perplexity, but this is driven by gist tokens. When excluding all but the final gist token per segment, perplexity increases significantly. This occurs because gist tokens have predictable patterns, while the final gist token must predict subsequent sentence beginnings, resulting in higher perplexity.

# D Benchmark Compression rates

Table 7 and table 8 shows compression rates on short benchmarks and HELMET (tiny) benchmarks correspondingly. The highest compression rate showed WinoGrande while the lowest compression rates is for synthetic ICL benchmark. For $N _ { g } = 8$ model on ICL bench sequence length even gets

![](images/c9bf756d0246b94ba1659a57db20d03fbab6b7ecae803c82afa874c0d24fee63.jpg)  
Figure 2: PG19 perplexity for Sentence Llama3.2-3B $( N _ { g } \in \{ 4 , 8 \} )$ ) compared to the base model across different prefix lengths. The “no Gist Tokens” curves represent perplexity calculated while excluding all but the final gist token in each segment.

longer than t he original one. For this table comression rate $( R _ { c } )$ was computed by this formula:

$$
R _ {c} = n _ {\text {r e g u l a r}} / n _ {\text {g i s t}}, \tag {2}
$$

where $n _ { r e g u l a r }$ is a number of regular tokens, $n _ { g i s t }$ is a number of all gist tokens in processed sequence.

Table 7: Short benchmarks compression rates.   

<table><tr><td>Ng</td><td>ARC</td><td>HS</td><td>MMLU</td><td>WG</td></tr><tr><td>1</td><td>19.86</td><td>16.55</td><td>21.31</td><td>82.96</td></tr><tr><td>2</td><td>9.93</td><td>8.28</td><td>10.66</td><td>41.48</td></tr><tr><td>4</td><td>4.96</td><td>4.14</td><td>5.33</td><td>20.74</td></tr><tr><td>8</td><td>2.48</td><td>2.07</td><td>2.66</td><td>10.37</td></tr></table>

Table 8: Long benchmarks compression rates.   

<table><tr><td>Ng</td><td>cite</td><td>icl</td><td>longqa</td><td>recall</td><td>rerank</td></tr><tr><td>1</td><td>32.77</td><td>7.01</td><td>31.31</td><td>17.75</td><td>25.15</td></tr><tr><td>2</td><td>16.39</td><td>3.51</td><td>15.66</td><td>8.88</td><td>12.57</td></tr><tr><td>4</td><td>8.19</td><td>1.75</td><td>7.83</td><td>4.44</td><td>6.29</td></tr><tr><td>8</td><td>4.10</td><td>0.88</td><td>3.91</td><td>2.22</td><td>3.14</td></tr></table>
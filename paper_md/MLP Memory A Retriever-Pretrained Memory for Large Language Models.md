# MLP Memory: A Retriever-Pretrained Memory for Large Language Models

Rubin Wei1,∗, Jiaqi Cao1,∗, Jiarui Wang1, Jushi Kai1, Qipeng Guo2, Bowen Zhou2,3, Zhouhan Lin1,2‡

1LUMIA Lab, School of Artificial Intelligence, Shanghai Jiao Tong University   
2Shanghai AI Laboratory 3Department of Electronic Engineering, Tsinghua University   
# weirubinn@gmail.com, lin.zhouhan@gmail.com $^ *$ Equal Contribution ‡ Corresponding Author.   
§ https://github.com/Rubin-Wei/MLPMemory https://huggingface.co/Rubin-Wei/MLPMemory-Mistral-wikipedia

Abstract Modern approaches to enhancing Large Language Models’ factual accuracy and knowledge utilization face a fundamental trade-off: non-parametric retrieval-augmented generation (RAG) provides flexible access to external knowledge but suffers from high inference latency and shallow integration, while parametric fine-tuning methods like LoRA risk catastrophic forgetting and degraded general capabilities. In this work, we propose MLP Memory, a lightweight parametric module that learns to internalize retrieval patterns without explicit document access. By pretraining an MLP to imitate a ??NN retriever’s behavior on the entire pretraining dataset, we create a differentiable memory component that captures the benefits of retrieval-based knowledge access in a fully parametric form. Our architecture integrates this pretrained MLP Memory with Transformer decoders through simple probability interpolation, yielding $1 7 . 5 \%$ and $2 4 . 1 \%$ scaling gains on WikiText-103 and Web datasets, respectively. It further achieves $1 2 . 3 \%$ relative improvement on five question-answering benchmarks and 5.2 points absolute gain across nine general NLP tasks, while reducing hallucinations by up to 10 points on HaluEval. Moreover, MLP Memory delivers $2 . 5 \times$ faster inference than RAG with superior accuracy. Our findings show that learning retrieval patterns parametrically bridges the gap between efficient inference and effective knowledge access, offering a practical alternative to both RAG and fine-tuning approaches.

![](images/d34de53382910e37994d6c666e628e359a8981b1553bc16a7b8cb737ac1a3991.jpg)  
NQ

![](images/299d7eac8886c2c2fa4baabc256305bfafd50480239df30c789389a2eb904e06.jpg)  
WebQA

![](images/e853292301be55aa422198703a343810a1b3a91933d843d236033599c9d5c730.jpg)  
TriviaQA

![](images/a0d6d19edfbc211ee4bb25a7b6d726a500cb546ea853155248123013f8275dd2.jpg)  
Base LM +CPT +LoRA +RAG +MLP Memory

Time to First token( ) TTFT(ms)   
![](images/4a5691c152a12cba00842c28b5786ba699ea075600a9ea7a97a558b6e83bf03a.jpg)  
Base LM(7B) +RAG

Tokens per second( ) TPS(tokens/s)

Figure 1 | Performance and efficiency comparison. Left: accuracy across three QA benchmarks. MLP Memory consistently outperforms the base model, surpassing both parametric methods (CPT, LoRA) and non-parametric retrieval (RAG). Right: inference efficiency, measured by time to first token (TTFT, ↓ lower is better) and tokens per second (TPS, ↑ higher is better). RAG results are shown for top-5 retrieval. ??NN-LM is accelerated via dimension reduction $4 0 9 6 {  } 2 5 6$ ), and both RAG and ??NN-LM use the Wikipedia-2021 retrieval corpus. MLP Memory uses 1B parameters.   
![](images/a14dee4b2aefab1f09307af15c0355ac4a19fcac75baa65f5ba39ff9f8ebacdf.jpg)  
+kNN-LM +MLP Memory(1B)

![](images/c6765cbd8f696e4c6555a115a538b3660e9fedf4b985344979af29b5e2890d75.jpg)

![](images/8fee62723451b50c3504f56b01ad39ee3eeb94d81f43f3dbdde94f894b189fca.jpg)

![](images/cd5f6a0ce046bf9196ed13da105986d325de20a4c4a3f3e06f3f52055dd26516.jpg)

![](images/d4ad1dc6f06fdf666250f8f59c534581c54be9d1231079f575efa10a4986a484.jpg)  
Figure 2 | Approaches to enhance factual accuracy and knowledge utilization. Top left: Nonparametric RAG provides flexible knowledge access but suffers from high latency. Top right: Parametric fine-tuning risks catastrophic forgetting. Bottom: MLP Memory learns retrieval patterns during training (left) and enables efficient inference without explicit retrieval (right).

# 1. Introduction

Decoder-only architectures such as GPT [Brown et al., 2020], LLaMA [Grattafiori et al., 2024], Qwen [Qwen et al., 2025], and DeepSeek [Liu et al., 2024] have achieved remarkable success in various tasks, including open-ended text generation [OpenAI et al., 2024], code completion [Chen et al., 2021], image synthesis [Chen et al., 2020], and multimodal reasoning [Liu et al., 2023]. However, despite their impressive capabilities, these models often struggle with effective knowledge utilization, producing responses that may be fluent but fail to accurately leverage the factual information encoded in their parameters.

Current approaches to enhance knowledge utilization in LLMs face significant trade-offs. Retrievalaugmented generation (RAG) methods [Gao et al., 2022, Izacard et al., 2022, Lewis et al., 2021, Peng et al., 2023] dynamically fetch relevant documents to ground model outputs, providing flexible access to external knowledge sources. However, these non-parametric approaches introduce substantial inference latency through expensive nearest-neighbor searches and longer context from retrieved documents. They also suffer from shallow integration with the base model, as the retrieval component remains isolated from the LLM’s computational graph. Conversely, parametric adaptation methods such as continued pre-training (CPT) and LoRA [Hu et al., 2022] directly modify model weights to incorporate domain-specific knowledge. While computationally efficient at inference time, these approaches risk catastrophic forgetting of previously learned capabilities and often degrade performance on general tasks, requiring careful task-specific tuning that limits their broader applicability. Figure 2 illustrates how our approach differs fundamentally from both non-parametric retrieval methods and parametric adaptation approaches.

In contrast to decoder-only LLMs, neuroscience research reveals a lateralized human brain where language processing is dominated by the left hemisphere while memory formation occurs in the hippocampus [Douglas, 1967, Gazzaniga, 2005a,b]. This insight has inspired memory-augmented models in machine learning. Early approaches like Memory Networks [Weston et al., 2015] enabled read/write operations on external memory, while Sparse Access Memory introduced differentiable memory access schemes. However, these were task-specific with limited general applicability. In the

# RAG

Refer to the background document and answer the question:

Background: 1. The SS Security Service, known as the SS SD-Amt, became the official security organization of the Nazi Party in 1934. Consisting at first of paid agents and a few hundred unpaid informants scattered across Germany, the SD was quickly professionalized under Heydrich, who commissioned National Socialist academics and lawyers to ensure that the SS and its Security Service in particular, operated \"within the framework of National Socialist ideology.\" Heydrich was given the power to select men for the SS Security Service from among any SS subdivisions since Himmler considered the organization of the SD as important. In September 1939, the SD was divided into two departments, the interior department (Inland-SD) and the foreign department (Ausland-SD), and placed under the authority of the Reich Security Main Office (RSHA).

Question: In WWII, who was the head of the Nazi party's security service? The answer is:

![](images/4538f529e773aaa25f14857003786c816dfdee44c9391fad50fe2d5ecf078ac6.jpg)

Output: 1. Heinrich Himm

![](images/47701c590dbc3d5b02e92ad02284f5969476cd9a1ae7c75b8afe40706531df78.jpg)

# Base LM

Answer the question:

Question: In WWII, who was the head of the Nazi party's security service? The answer is:

# MLP Memory

Answer the question:

Output: Question: In WW

![](images/fcc708720b9f7eeeadf7f5cb95775a43acab4a62e10be84753e91bef350ab535.jpg)

Question: In WWII, who was the head of the Nazi party's security service? The answer is:

Output: Reinhard Heydrich.

![](images/79e7d2f962d6830d60493ce8ecef17c160abf0035cd78ab0569d4606de42c8fc.jpg)  
Figure 3 | Comparison of model outputs on a factual question. Despite retrieving relevant documents with correct information (highlighted in green), RAG is misled by contextual distractors and produces an incorrect answer. MLP Memory generates the correct answer without explicit retrieval.

LLM era, methods such as Memory Transformers [Burtsev et al., 2021] incorporate trainable memory tokens for global context, while AutoCompressors [Chevalier et al., 2023] compress long contexts into summary vectors. Nevertheless, these memory tokens primarily function as working memory supplements for context extension rather than long-term memory capable of retaining information from the entire training corpus.

In this work, we propose an external memory for LLM that is pretrained to mimic a retriever on the entire pretraining dataset. Specifically, following the RAG setting in ??NN-LM [Khandelwal et al., 2020], this memory learns to map the LLM hidden state at a certain step to a vocabulary distribution matching the output of the ??NN retriever. During inference, the LLM’s native output is interpolated with the retriever-pretrained output from the external memory. Our resulting architecture, illustrated in Figure 4, consists of a transformer decoder and an external MLP memory, each pretrained separately with different pretraining tasks. For our pretrained external memory, we aim to achieve the following features simultaneously:

1) End-to-end differentiability. Unlike the non-parametric nature of retrievers, our MLP memory is fully parameterized and allows gradieat flow during training. This enables end-to-end joint optimization of the entire model architecture.   
2) Highly compressible memory. The MLP memory compresses large datastores (e.g., 40TB for 5B tokens in ??NN-LM) into a compact parametric form (e.g., 4GB for 1B parameters storing 5B tokens), facilitating efficient deployment without performance degradation.   
3) Low inference-time latency. MLP memory eliminates costly retrieval operations, achieving $2 . 5 \times$ faster inference than RAG methods and $5 . 6 \times$ faster inference than kNN-LM when using a 5B-token retrieval corpus. Crucially, unlike retrieval-based approaches, our method’s inference speed remains constant regardless of the retrieval corpus size.

4) Scalable, general knowledge memorization, covering the whole training set. MLP memory is trained on the entire pretraining dataset, not limited to the context level. It supports general knowledge retention and demonstrates stronger scaling behavior than decoder-only models.   
5) Long-term memory. While existing memory tokens serve primarily as working memory by storing local context for immediate use, our MLP memory functions as a long-term repository of generalizable knowledge acquired during the pretraining phase.

Experimental results demonstrate that MLP Memory significantly outperforms existing approaches across multiple dimensions. Our architecture exhibits steeper power-law scaling with model size, achieving $1 7 . 5 \%$ and $2 4 . 1 \%$ improvement on WikiText-103 and Web datasets compared to decoderonly architectures, while continuing to benefit from additional training without overfitting. On downstream tasks, it achieves average relative improvements of $1 2 . 3 \%$ (Mistral-7B) and $7 . 8 \%$ (Llama2- 7B) on five QA benchmarks, with WebQA showing exceptional gains $( 3 7 . 4 5 \%$ vs. $2 9 . 2 8 \%$ baseline). On nine general NLP tasks, it delivers a 5.2 points absolute improvement. MLP Memory also substantially reduces hallucinations on HaluEval, with accuracy improvements of 9.68, 10.08, and 2.14 points on dialogue, QA, and summarization tasks respectively. Most notably, it achieves $2 . 5 \times$ faster time-to-firsttoken than RAG and $5 . 6 \times$ faster than ??NN-LM, while maintaining constant inference speed regardless of corpus size, unlike retrieval methods whose latency scales with data size. Figure 1 illustrates MLP Memory’s performance gains and inference efficiency over baselines and Figure 3 demonstrates a case where MLP Memory correctly answers factual questions while RAG fails despite retrieving correct information. These results confirm that parametric compression of retrieval patterns offers a more efficient and effective alternative to explicit retrieval.

# 2. Preliminary: $k$ -nearest neighbors language model

The ??NN-LM [Khandelwal et al., 2020] augments a pre-trained LM by interpolating its parametric distribution with a non-parametric distribution from nearest neighbor retrieval. Given context $c _ { t } =$ $( w _ { 1 } , . . . , w _ { t - 1 } )$ , and $w _ { t }$ denotes the next token. The next-token probability is:

$$
p \left(w _ {t} \mid c _ {t}\right) = \lambda p _ {k N N} \left(w _ {t} \mid c _ {t}\right) + (1 - \lambda) p _ {L M} \left(w _ {t} \mid c _ {t}\right), \tag {1}
$$

where $\lambda \in [ 0 , 1 ]$ is the interpolation parameter, $p _ { L M }$ is the LM’s distribution, and $p _ { k N N }$ is retrieval-based distribution.

Datastore Constructed via a forward pass over a corpus, the datastore consists of key-value pairs $\left( k _ { t } , \upsilon _ { t } \right)$ where $k _ { t } = f ( c _ { t } )$ encodes context $c _ { t }$ using LM representations, and $\nu _ { t }$ is the next token $w _ { t }$ :

$$
\left(\mathcal {K}, \mathcal {V}\right) = \left\{\left(f \left(c _ {t}\right), w _ {t}\right) \mid \left(c _ {t}, w _ {t}\right) \in \mathcal {D} \right\}. \tag {2}
$$

Inference The LM encodes context $c$ into query $f ( c )$ and retrieves $k$ -nearest neighbors $N$ from $( \mathcal { K } , \mathcal { V } )$ using distance metric $d ( \cdot , \cdot )$ (typically squared $L ^ { 2 }$ ). The non-parametric distribution is:

$$
p _ {k N N} (y \mid c) \propto \sum_ {\left(k _ {i}, v _ {i}\right) \in \mathcal {N}} \mathbb {I} _ {y = v _ {i}} \exp (- d \left(k _ {i}, f (c)\right)). \tag {3}
$$

While ??NN-LM improves predictions through explicit memory, it suffers from substantial storage requirements and high-latency retrieval. For instance, the Wikitext-103 datastore requires nearly 500 GB of storage even for the GPT2-small model [He et al., 2021]. These limitations motivate our MLP Memory, a compact parametric model pretrained to approximate the retrieval function: given a query embedding, it directly outputs a ??NN-like next token distribution, thereby eliminating both the substantial storage requirements and high-latency retrieval.

![](images/2279b5682bdd2ba5bda1f9653275813e9cc786dc77c3779f7ddba146379b3d5c.jpg)  
Figure 4 | Overview of MLP Memory architecture. (a) Inference: MLP Memory processes context representations from a specific LLM layer, generating token probabilities that are interpolated with LLM outputs for final predictions. (b) Training: MLP Memory learns to imitate retriever behavior using LLM representations as input and distributions generated by ??NN retrievers as targets, optimized through a hybrid objective.

# 3. MLP Memory

In this section, we present MLP Memory, a lightweight parametric module that learns to internalize retrieval patterns without explicit document access. Our approach consists of three key components: a stack of MLPs that processes hidden representations without token-mixing operations (Section 3.1), a specialized pre-training procedure that enables the MLP to mimic non-parametric retrieval distributions (Section 3.2), and an efficient inference mechanism for deployment (Section 3.3). As illustrated in Figure 4, MLP Memory first learns to mimic non-parametric retrieval distributions during pre-training (Figure 4(b)), then seamlessly integrates with the language model during inference (Figure 4(a)), eliminating both the storage requirements of large datastores and the computational cost of nearest neighbor search.

# 3.1. Architecture

Our MLP Memory learns to mimic non-parametric retrieval by mapping query embeddings to $k \mathbf { N N }$ distributions. Given query $q \ = \ f ( c )$ from context ??, the MLP directly predicts $p _ { k N N } ( y | c )$ without neighbor search, transforming discrete retrieval into a differentiable mapping $\mathcal { M } : \mathbb { R } ^ { d }  \mathbb { R } ^ { | V | }$ , where ?? is the embedding dimension and $| V |$ is the vocabulary size.

In designing the memory module, we observe from Section 2 that the retriever imitation task processes

a single-vector representation without requiring token-mixing operations. Recent studies [Geva et al., 2020] have identified that FFN layers function as key-value memories, suggesting that MLPs play a specialized role in knowledge memorization within LLMs. Based on these insights, we propose pretraining an all-MLP memory that effectively functions as a non-parametric retriever, as illustrated in Figure 4.

The MLP Memory takes hidden representations $f ( c )$ from the pretrained LM as input and is trained to predict the corresponding ??NN distribution $p _ { k N N } ( y | c )$ as its target. Once trained, the MLP’s output distribution is interpolated with the LM’s parametric distribution during inference, following the same interpolation scheme as ??NN-LM but without requiring datastore access or neighbor search.

# 3.2. Training

The training procedure for MLP Memory consists of two primary stages: constructing supervision signals from non-parametric retrieval distributions, and optimizing the MLP to mimic these distributions through a carefully designed loss function.

Data Construction To generate supervision for training MLP Memory, we leverage the datastore construction process described in Section 2. We build the datastore $( \mathcal { K } , \mathcal { V } )$ through a forward pass over the training corpus, storing context representations and their corresponding next tokens. For each training example $( c _ { t } , w _ { t } ) \in \mathcal { D }$ , we compute the non-parametric distribution $p _ { k N N } ( y | c _ { t } )$ by retrieving $k$ -nearest neighbors from the datastore. To prevent trivial self-retrieval that would contaminate the learning signal, we exclude the query itself from the neighbor set when constructing the target distribution. These embedding-distribution pairs $\{ ( f ( c _ { t } ) , p _ { k N N } ( \cdot | c _ { t } ) ) \}$ are precomputed offline and cached for efficient training.

Loss Function Unlike traditional language modeling with single-label targets, ??NN distributions capture the diversity of plausible continuations by encoding multiple valid next tokens weighted by their contextual similarity. Our ablation studies in Section 5.5 demonstrate that a hybrid objective combining two complementary losses yields optimal performance. Our approach centers on minimizing the Kullback-Leibler divergence [Van Erven and Harremos, 2014] between MLP Memory’s output distribution and the cached ??NN distributions:

$$
\mathcal {L} _ {K L} \left(c _ {t}\right) = \mathrm {K L} \left(p _ {k N N} \left(\cdot \mid c _ {t}\right) \| p _ {M L P} \left(\cdot \mid c _ {t}\right)\right) \tag {4}
$$

This encourages the memory module to match the full probability distribution rather than merely predicting the most likely token. To prevent excessive deviation from the underlying corpus distribution, we integrate a complementary Cross-Entropy loss [Zhang and Sabuncu, 2018]:

$$
\mathcal {L} _ {C E} \left(c _ {t}\right) = - \log p _ {M L P} \left(w _ {t} \mid c _ {t}\right) \tag {5}
$$

The final training objective balances these two components through a hyperparameter ??:

$$
\mathcal {L} \left(c _ {t}\right) = \alpha \cdot \mathcal {L} _ {K L} \left(c _ {t}\right) + (1 - \alpha) \cdot \mathcal {L} _ {C E} \left(c _ {t}\right) \tag {6}
$$

The KL term encourages learning distributional patterns while the CE term ensures accurate groundtruth prediction, preventing the overfitting that occurs with cross-entropy alone.

# 3.3. Inference

Once trained, MLP Memory integrates with the base language model through simple probability interpolation. During inference, MLP Memory processes hidden representations from the language

model $M _ { \mathrm { L M } }$ and produces a distribution that is interpolated with the LM’s output:

$$
p _ {\text {f i n a l}} \left(w _ {t} \mid c _ {t}\right) = \lambda \cdot p _ {M L P} \left(w _ {t} \mid c _ {t}\right) + (1 - \lambda) \cdot p _ {L M} \left(w _ {t} \mid c _ {t}\right) \tag {7}
$$

where $\lambda \in [ 0 , 1 ]$ controls the influence of retrieval-based knowledge.

Unlike retrieval-augmented approaches that require nearest neighbor search and extended context processing, MLP Memory requires only a single forward pass through a lightweight all-MLP architecture. As demonstrated in Figure 1, our method achieves $2 . 5 \times$ faster time-to-first-token than RAG (top-5) and $5 . 6 \times$ faster than ??NN-LM, despite ??NN-LM employing dimension reduction from 4096 to 256 for acceleration. For tokens per second, MLP Memory delivers $1 . 5 \times$ higher throughput than RAG and $6 \times$ higher than ??NN-LM, while introducing only $1 . 2 \times$ overhead relative to the base model. Crucially, this performance remains constant regardless of retrieval corpus size, unlike retrieval-based methods whose latency scales with datastore size.

# 4. Experimental Setup

Overview We conduct comprehensive experiments to evaluate MLP Memory across five critical dimensions. First, we analyze the scaling behavior (5.1) of models augmented with MLP memory through power-law fitting. Second, we assess performance on five question-answering benchmarks (5.2) to demonstrate that our approach represents a novel form of parametric memory that surpasses both traditional parametric methods (continued pretraining, LoRA) and non-parametric approaches (RAG). Third, we evaluate on fundamental NLP tasks (5.3) to verify that integrating MLP Memory preserves the base model’s general capabilities. Fourth, we examine hallucination reduction (5.4) on HaluEval to validate our method’s effectiveness in improving factual accuracy. Finally, we present an ablation study (5.5) to analyze design choices such as loss weighting and layer selection.

Implementation Details We conduct our experiments on 32×A800 80GB GPUs. To demonstrate the generalizability of our approach, we employ two distinct backbone models: Llama-2-7B [Touvron et al., 2023] and Mistral-7B-v0.3 [Jiang et al., 2023]. For scaling law, please refer to Section 5.1 for detailed settings. For question-answering benchmarks, we build key-value datastores and nonparametric distributions using both models on preprocessed Wikipedia-2021 [Izacard et al., 2022], and train separate 1B-parameter MLP Memory modules with learning rate 4e-4. For general NLP tasks, we build datastores using Mistral-7B-v0.3 on a heterogeneous corpus following Geng et al. [2024], and train the MLP Memory with learning rate 4e-4. For hallucination evaluation, we directly apply the MLP Memory trained from question-answering experiments. All experiments use a training budget equivalent to the computational cost of training a 7B parameter model for 1 epoch. The training hyperparameter $\alpha$ is set to 0.4 across all tasks. The interpolation hyperparameter ?? is tuned on the validation split of each task following Khandelwal et al. [2020], see more details in Appendix B.

Baselines We compare MLP Memory against established methods for improving factual accuracy and knowledge utilization: RAG, which employs BGE [Chen et al., 2024] as the retrieval model and retrieves top-5 documents to ensure comprehensive context coverage. ??NN-LM [Khandelwal et al., 2020], configured with interpolation parameter $\lambda = 0 . 1$ and temperature $\tau = 1 0 . 0$ following [Geng et al., 2024]. LoRA [Hu et al., 2022], applied to query, key, value, and MLP layers, with rank adjusted to match the parameter count of our MLP Memory modules. Continued Pretraining (CPT), which involves further training of all model parameters on the corresponding corpus.

# 5. Experimental Results

![](images/85f8b71eac3bd488f0df73c18e53789c8d0bb4c90a4f24a211d90853a01d6d69.jpg)  
(a) Scaling law on WikiText-103

![](images/5113c43fd6f570bf7718b367ea7307ba18bdee6c553102f7eb08e828606e4e51.jpg)  
(b) Scaling law on Web

![](images/da111aecebf5a35e020baf72e1c9955a11468e13e889fa43e92f9d2dc4d86fdb.jpg)  
(c) Training compute Scaling Law   
Figure 5 | Power-law scaling behavior with model size $N$ and training compute ??. (a) Scaling results compare the continued training of GPT2 (GPT2-ConTrain) with our overall model architecture (GPT2+MLP Mem) under fixed compute. Our fitted curve shows a $1 7 . 5 \%$ exponent improvement on WikiText-103. (b) On the larger Web dataset, our architecture exhibits stronger scaling gains from increased data size, with an exponent improvement of $2 4 . 1 \%$ . (c) At the GPT2-xl scale, our architecture continues to benefit from additional training on the Web dataset without overfitting.

# 5.1. Scaling law

Setup We conduct scaling law experiments using standard decoder-only models and our overall model architecture. As baselines, we use four GPT-2 Radford et al. [2019] variants with increasing parameter counts: GPT2-small (124M), GPT2-medium (345M), GPT2-large (774M), and GPT2-xl (1.5B). For MLP Memory, we define three configurations: small (124M), medium (335M), and large (774M) that align with the scaling trend of standard architectures. The MLP Memory module is externally integrated with a matching-sized GPT-2 variant, resulting in total parameter counts of approximately 248M, 710M, and 1.5B for the small, medium, and large configurations, respectively. All models are trained on two datasets: WikiText-103 Merity et al. [2016] (around 100M tokens) and a mixed Web dataset (around 600M tokens). Following Shi et al. [2022a], our Web dataset combines diverse knowledge sources relevant to common NLP tasks, including WikiText-103, Amazon Reviews He and McAuley [2016], CC-NEWS Hamborg et al. [2017], and IMDB Maas et al. [2011].

Scaling law with model parameters ?? Following Kaplan et al. [2020], we model perplexity scaling as $P P L = ( \boldsymbol { \beta } \cdot \boldsymbol { N } ) ^ { \gamma }$ . Under fixed compute, we compare our architecture to continued GPT-2 training on WikiText-103 and Web datasets in terms of test perplexity scaling with model size ??. Results in Figure 5 show our architecture demonstrates a steeper scaling curve than the decoder-only model, indicating improved scaling efficiency. The power-law scaling laws on WikiText-103 can be expressed as:

$$
P P L _ {d} = (9. 3 \cdot 1 0 ^ {- 8} N) ^ {- 0. 1 4 3} \quad a n d \quad P P L _ {m} = (1. 1 \cdot 1 0 ^ {- 6} N) ^ {- 0. 1 6 8} \tag {8}
$$

where $P P L _ { d }$ and $P P L _ { m }$ denote the test perplexity of the decoder model and our architecture, respectively. The corresponding power-law scaling laws on the Web dataset are as follows:

$$
P P L _ {d} = (7. 7 \cdot 1 0 ^ {- 5} N) ^ {- 0. 2 1 6} \quad a n d \quad P P L _ {m} = (5. 0 \cdot 1 0 ^ {- 4} N) ^ {- 0. 2 6 8} \tag {9}
$$

These results highlight the superior scaling efficiency of our overall model architecture compared to the standard decoder-only baseline, on both WikiText-103 and the Web dataset.

Scaling law with training compute ?? We further examine how model performance scales with training compute ?? while keeping model size fixed. At the GPT2-xl scale, we conduct experiments on the Web dataset, measuring test perplexity after varying amounts of training flops. As illustrated in

Table 1 | Question answering performance across five benchmarks. Positive gains are shown in green and negative changes in red. Percentage in parentheses denotes the relative improvement over the base model. All methods use the same Wikipedia-2021 corpus for training or retrieval.   

<table><tr><td rowspan="2">Methods</td><td colspan="3">Open-Domain QA</td><td>Long-form QA</td><td>Multihop QA</td><td rowspan="2">Average</td></tr><tr><td>NQ</td><td>WebQA</td><td>TriviaQA</td><td>TruthfulQA</td><td>HotpotQA</td></tr><tr><td>Llama2-7B</td><td>23.18</td><td>32.09</td><td>56.91</td><td>29.16</td><td>22.72</td><td>32.81</td></tr><tr><td colspan="7">Non-parametric methods</td></tr><tr><td>+RAG</td><td>14.60-8.58</td><td>36.71+4.62</td><td>62.20+5.29</td><td>31.59+2.43</td><td>19.60-3.12</td><td>32.94(+0.4%)</td></tr><tr><td colspan="7">Parametric methods</td></tr><tr><td>+CPT</td><td>12.90-10.28</td><td>31.55-0.54</td><td>58.81+1.90</td><td>29.56+0.40</td><td>15.49-7.23</td><td>29.66(-9.6%)</td></tr><tr><td>+LoRA</td><td>17.88-5.30</td><td>35.19+3.10</td><td>58.14+1.23</td><td>28.33-0.83</td><td>17.18-5.54</td><td>31.34(-4.5%)</td></tr><tr><td>+MLP Mem</td><td>27.04+3.86</td><td>36.61+4.52</td><td>57.50+0.59</td><td>30.04+0.88</td><td>25.69+2.97</td><td>35.38(+7.8%)</td></tr><tr><td>Mistral-7B-v0.3</td><td>20.63</td><td>29.28</td><td>57.65</td><td>32.09</td><td>20.96</td><td>32.12</td></tr><tr><td colspan="7">Non-parametric methods</td></tr><tr><td>+RAG</td><td>22.56+1.93</td><td>24.90-4.38</td><td>54.21-3.44</td><td>35.47+3.38</td><td>29.77+8.81</td><td>33.38(+3.9%)</td></tr><tr><td colspan="7">Parametric methods</td></tr><tr><td>+CPT</td><td>12.16-8.47</td><td>34.06+4.78</td><td>61.21+3.56</td><td>29.18-2.91</td><td>16.04-4.92</td><td>30.53(-5.0%)</td></tr><tr><td>+LoRA</td><td>18.17-2.46</td><td>34.50+5.22</td><td>61.60+3.95</td><td>30.91-1.18</td><td>16.23-4.73</td><td>32.28(+0.5%)</td></tr><tr><td>+MLP Mem</td><td>25.20+4.57</td><td>37.45+8.17</td><td>60.99+3.34</td><td>32.54+0.45</td><td>24.14+3.18</td><td>36.06(+12.3%)</td></tr></table>

Figure 5 (c), our overall model architecture achieves significantly lower perplexity with increasing training compute, with no signs of overfitting. This suggests that the retriever imitation pretraining task is more challenging and continues to benefit from additional compute.

# 5.2. Question Answering Performance

We evaluate MLP Memory on five diverse QA benchmarks: Natural Questions (NQ) [Kwiatkowski et al., 2019], WebQA [Berant et al., 2013], TriviaQA [Joshi et al., 2017], TruthfulQA [Lin et al., 2022], and HotpotQA [Yang et al., 2018], comparing against CPT, LoRA, and RAG. As shown in Table 1, Mistral-7B-v0.3 with MLP Memory achieves an average relative improvement of $1 2 . 3 \%$ over the baseline across five benchmarks, with particularly striking improvements on NQ $2 5 . 2 0 \%$ vs. baseline $2 0 . 6 3 \%$ ) and WebQA $( 3 7 . 4 5 \%$ vs. baseline $2 9 . 2 8 \%$ ). While CPT and LoRA suffer significant degradation across all tasks—likely due to catastrophic forgetting during domain-specific training—MLP Memory maintains or improves performance by learning to emulate retrieval behavior without modifying the base model’s parameters. Notably, our approach outperforms RAG despite RAG having access to the same external knowledge at inference time, suggesting that our parametric compression of retrieval patterns captures richer contextual relationships than explicit document retrieval. The consistent gains across both factoid QA (NQ, TriviaQA) and multi-hop reasoning (HotpotQA) demonstrate that MLP Memory effectively bridges the gap between parametric and non-parametric memory systems.

# 5.3. General NLP Task Performance

To ensure MLP Memory doesn’t compromise fundamental language understanding, we evaluate on nine standard NLP tasks spanning sentiment classification (SST-2 [Socher et al., 2013], MR [Pang and Lee, 2005a], CR [Hu and Liu, 2004], RT [Pang and Lee, 2005b], HYP [Kiesel et al., 2019]), textual entailment (CB [De Marneffe et al., 2019], RTE [Dagan et al., 2010]), and topic classification (AGNews [Zhang et al., 2015a], Yahoo [Zhang et al., 2015b]). Table 2 reveals that MLP Memory achieves comprehensive improvements across all general tasks, achieving the highest average score

Table 2 | Performance on nine general NLP tasks spanning sentiment classification, textual entailment, and topic classification. ↑ indicate improvement over the Mistral-7B-v0.3 baseline, while ↓ indicate decreased performance.   

<table><tr><td rowspan="2">Methods</td><td colspan="5">Sentiment Classification</td><td colspan="2">Textual.</td><td colspan="2">Topic.</td><td rowspan="2">Average</td></tr><tr><td>SST2</td><td>MR</td><td>CR</td><td>RT</td><td>HYP</td><td>CB</td><td>RTE</td><td>AGN</td><td>Yahoo</td></tr><tr><td>Mistral-7B-v0.3</td><td>81.21</td><td>75.35</td><td>62.30</td><td>74.95</td><td>55.42</td><td>69.64</td><td>59.57</td><td>75.95</td><td>56.36</td><td>67.86</td></tr><tr><td colspan="11">Non-parametric methods</td></tr><tr><td>+RAG</td><td>87.20↑</td><td>83.70↑</td><td>71.55↑</td><td>82.36↑</td><td>54.65↓</td><td>57.14↓</td><td>66.43↑</td><td>75.64↓</td><td>58.43↑</td><td>70.79↑</td></tr><tr><td>+kNN-LM</td><td>82.15↑</td><td>76.85↑</td><td>61.70↓</td><td>74.95</td><td>56.78↑</td><td>71.42↑</td><td>60.28↑</td><td>76.13↑</td><td>56.26↓</td><td>68.50↑</td></tr><tr><td colspan="11">Parametric methods</td></tr><tr><td>+CPT</td><td>87.09↑</td><td>82.85↑</td><td>82.60↑</td><td>77.48↑</td><td>60.65↑</td><td>57.14↓</td><td>52.71↓</td><td>83.10↑</td><td>51.56↓</td><td>70.58↑</td></tr><tr><td>+LoRA</td><td>86.54↑</td><td>83.20↑</td><td>75.10↑</td><td>79.83↑</td><td>55.42</td><td>51.78↓</td><td>56.31↓</td><td>65.46↓</td><td>57.30↑</td><td>67.88↑</td></tr><tr><td>+MLP Mem</td><td>83.19↑</td><td>79.90↑</td><td>75.95↑</td><td>75.42↑</td><td>64.15↑</td><td>76.79↑</td><td>64.62↑</td><td>80.28↑</td><td>57.33↑</td><td>73.07↑</td></tr></table>

Table 3 | Performance on HaluEval benchmark across question answering, dialogue, and summarization tasks. Results show accuracy $( \% )$ . RAG is not evaluated on summarization as this task requires only the source document.   

<table><tr><td></td><td>Dialogue</td><td>QA</td><td>Summarization</td></tr><tr><td>Mistral-7B-v0.3</td><td>57.18</td><td>53.99</td><td>50.27</td></tr><tr><td>+CPT</td><td>51.68–5.50</td><td>46.49–7.50</td><td>47.39–2.88</td></tr><tr><td>+LoRA</td><td>55.51–1.67</td><td>50.02–3.97</td><td>50.38+0.11</td></tr><tr><td>+RAG</td><td>59.06+1.88</td><td>65.09+11.10</td><td>-</td></tr><tr><td>+MLP Mem</td><td>66.86+9.68</td><td>64.07+10.08</td><td>52.41+2.14</td></tr></table>

compared to all baselines. The improvements are particularly pronounced on reasoning-intensive tasks like RTE $6 4 . 6 2 \%$ vs. baseline $5 9 . 5 7 \%$ ) and CB $7 6 . 7 9 \%$ vs. baseline $6 9 . 6 4 \%$ ), suggesting that the retrieval patterns learned by MLP Memory provide useful inductive biases even for tasks that don’t explicitly require factual knowledge retrieval. In contrast, CPT and LoRA show mixed results with improvements on some tasks but degradation on others. The robust performance across this diverse task suite demonstrates that MLP Memory’s external parametric memory complements rather than interferes with the base model’s learned representations.

# 5.4. Hallucination Reduction

We assess MLP Memory’s ability to reduce hallucinations using HaluEval [Li et al., 2023] across three generation tasks: dialogue, question answering, and summarization, where models must identify factual inconsistencies in generated content. As shown in Table 3, parametric methods (CPT and LoRA) severely degrade performance, confirming the risks of weight modification. MLP Memory consistently improves hallucination detection across all three domains, with gains of 9.68, 10.08, and 2.14 points respectively. These substantial improvements indicate that the retrieval patterns encoded in MLP Memory significantly help the model better distinguish factual from hallucinated content. The effectiveness across diverse generation contexts—from free-form dialogue to constrained summarization—suggests that MLP Memory’s learned retrieval behavior provides a general mechanism for grounding language generation in factual knowledge. This hallucination reduction, combined with strong QA performance and enhanced general capabilities, validates our core hypothesis that decoupling memorization from reasoning through retriever-pretrained external memory can enhance factual accuracy without the typical trade-offs of parametric or non-parametric approaches.

![](images/648688871f517c3d98fcfffbfe1b3ddf869adacc694e4b50964cb010568af5f5.jpg)  
(a)

![](images/ce24bfbf967c1453a4dfb6e04b20a9b82a3321e946c4f38585a53fa9aff7fe50.jpg)  
(b)   
Figure 6 | (a) Impact of KL loss weight $\alpha$ on retriever imitation. Lower PPL (min-max normalized for clarity) indicates better performance, with optimal balance at $\alpha = 0 . 4$ . (b) Impact of input layer depth on MLP Memory performance across model sizes. Layer percentage denotes depth in the decoder stack (e.g., $7 0 \%$ corresponds to layer 25 in GPT2-large).

# 5.5. Ablation study

Ablation Setup We conduct ablation experiments across three GPT2 [Radford et al., 2019] scales: small (12 layers), medium (24 layers), and large (36 layers), paired with corresponding MLP Memory modules of 117M, 345M, and 774M parameters respectively. All experiments are evaluated on WikiText-103 [Merity et al., 2016] to investigate loss weighting and optimal layer selection.

Impact of Loss Weighting We examine how balancing KL and CE losses affects retriever imitation by varying $\alpha$ from 0.0 to 1.0. As Figure 6(a) shows, extreme values produce suboptimal results—low values prevent the MLP memory from learning from the ??NN distribution, while high values cause overfitting to the language modeling objective. The optimal balance occurs at $\alpha = 0 . 4$ , indicating both objectives are necessary. KL divergence leverages the information-rich kNN distribution, enabling more effective generalization, while CE loss provides essential token-level prediction accuracy. This balanced approach prevents overfitting while maintaining predictive power.

Which Layer Provides the Best Representation for MLP Memory? While ??NN-LM performs best using the input to the final feedforward layer as the retrieval key, our MLP Memory consistently achieves optimal performance at around $7 0 \%$ of network depth, regardless of model scale. Our finding aligns with Memorizing Transformers [Burtsev et al., 2021], which also selected around $7 5 \%$ depth for optimal retrieval performance. We evaluate GPT2-small (12 layers), GPT2-medium (24 layers), and GPT2-large (36 layers), attaching the MLP Memory to various transformer blocks. As shown in Figure 6(b), the x-axis indicates relative depth $( 2 0 \% - 1 0 0 \% )$ ), and the y-axis shows min-max normalized perplexity $\mathrm { { 0 \% = } }$ best, $1 0 0 \% =$ worst). This consistent pattern across all model sizes contrasts with the ??NN-LM convention of using final-layer representations.

# 6. Related Work

Retrieval-Augmented Generation RAG [Gao et al., 2022, Lewis et al., 2021, Peng et al., 2023] mitigates hallucinations by grounding generation in external knowledge. Despite improving factual accuracy, RAG faces limitations: retrieval latency, coarse granularity, and limited LLM integration [Zhang et al., 2024]. Recent work [Su et al., 2025] explores enhanced retrieval with LLM priors. Our approach

proposes a parametric memory mimicking non-parametric retrieval, eliminating explicit document retrieval while preserving knowledge augmentation.

Memory-Augmented Language Models Various architectures explored memory augmentation, from Memory Networks [Weston et al., 2015] with explicit read-write components to Memory Transformers [Burtsev et al., 2021] with extended attention. LongMem [Wang et al., 2023] and MemoRAG [Qian et al., 2025] introduced decoupled architectures for long-term history storage. While these focus on context extension, our MLP memory expands across the entire pre-training corpus, enabling long-term generalizable knowledge storage.

MLP Architectures All-MLP architectures emerged as transformer alternatives, with gMLP [Liu et al., 2021] matching transformer performance and sparse MLPs [Yu et al., 2022] showing superior training efficiency. Studies [Geva et al., 2020] identified FFN layers as key-value memories in LLMs. Inspired by this, we propose pretraining an all-MLP memory as a non-parametric retriever, leveraging MLPs’ memorization capabilities for a compact, differentiable knowledge store.

# 7. Conclusion

In this paper, we introduced MLP Memory, a novel approach for enhancing language models by learning to internalize retrieval patterns. By pretraining a lightweight MLP module to imitate kNN retriever behavior on the entire pretraining corpus, MLP Memory captures the benefits of retrieval-augmented generation in a fully parametric form, without requiring explicit document access.

The key advantage of MLP Memory lies in its efficiency and effectiveness, as our architecture exhibits stronger scaling behavior than decoder-only models. On downstream tasks, our approach achieves $1 2 . 3 \%$ relative improvement on question-answering benchmarks, 5.2 points gain on general NLP tasks, and up to 10 points reduction in hallucinations—while delivering $2 . 5 \times$ faster inference than RAG and maintaining constant speed regardless of corpus size. Unlike parametric fine-tuning that risks catastrophic forgetting or non-parametric RAG that suffers from high latency, MLP Memory enhances model capabilities without these typical trade-offs.

MLP Memory introduces a new paradigm for language model enhancement that fundamentally reimagines how models access and utilize knowledge. By parametrically encoding retrieval behavior through a pretrained memory component, our approach creates a more efficient, accurate, and scalable framework that bridges the gap between parametric and non-parametric methods.

# References

Jonathan Berant, Andrew Chou, Roy Frostig, and Percy Liang. Semantic parsing on freebase from question-answer pairs. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1533–1544, 2013.   
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners, 2020. URL https://arxiv.org/abs/2005.1 4165.   
Mikhail S. Burtsev, Yuri Kuratov, Anton Peganov, and Grigory V. Sapunov. Memory transformer, 2021. URL https://arxiv.org/abs/2006.11527.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. arXiv preprint arXiv:2402.03216, 2024.   
Mark Chen, Alec Radford, Rewon Child, Jeff Wu, Heewoo Jun, David Luan, and Ilya Sutskever. Generative pretraining from pixels. In Proceedings of the 37th International Conference on Machine Learning, ICML’20. JMLR.org, 2020.   
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.   
Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan Zhao. xrag: Extreme context compression for retrieval-augmented generation with one token. Advances in Neural Information Processing Systems, 37:109487–109516, 2024.   
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to compress contexts, 2023. URL https://arxiv.org/abs/2305.14788.   
Ido Dagan, Bill Dolan, Bernardo Magnini, and Dan Roth. Recognizing textual entailment: Rational, evaluation and approaches–erratum. Natural Language Engineering, 16(1):105–105, 2010.   
Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser. The commitmentbank: Investigating projection in naturally occurring discourse. In proceedings of Sinn und Bedeutung, volume 23, pages 107–124, 2019.   
Robert J Douglas. The hippocampus and behavior. Psychological bulletin, 67(6):416, 1967.   
Luyu Gao, Xueguang Ma, Jimmy Lin, and Jamie Callan. Precise zero-shot dense retrieval without relevance labels, 2022. URL https://arxiv.org/abs/2212.10496.   
Michael S Gazzaniga. The ethical brain. Dana press, 2005a.   
Michael S Gazzaniga. Forty-five years of split-brain research and still going strong. Nature Reviews Neuroscience, 6(8):653–659, 2005b.   
Shangyi Geng, Wenting Zhao, and Alexander M Rush. Great memory, shallow reasoning: Limits of $k$ nn-lms. arXiv preprint arXiv:2408.11815, 2024.   
Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. arXiv preprint arXiv:2012.14913, 2020.   
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.   
Felix Hamborg, Norman Meuschke, Corinna Breitinger, and Bela Gipp. news-please: A generic news crawler and extractor, March 2017. URL https://doi.org/10.5281/zenodo.4120316.   
Junxian He, Graham Neubig, and Taylor Berg-Kirkpatrick. Efficient nearest neighbor language models, 2021. URL https://arxiv.org/abs/2109.04212.   
Ruining He and Julian McAuley. Ups and downs: Modeling the visual evolution of fashion trends with one-class collaborative filtering. In proceedings of the 25th international conference on world wide web, pages 507–517, 2016.

Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, and Luke Zettlemoyer. Surface form competition: Why the highest probability answer isn’t always right. arXiv preprint arXiv:2104.08315, 2021.   
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.   
Minqing Hu and Bing Liu. Mining and summarizing customer reviews. In Proceedings of the Tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’04, page 168–177, New York, NY, USA, 2004. Association for Computing Machinery. ISBN 1581138881. doi: 10.1145/1014052.1014073. URL https://doi.org/10.1145/1014052.1014073.   
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language models, 2022. URL https://arxiv.org/abs/2208.03299.   
Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. Mistral 7b, 2023. URL https://arxiv.org/abs/23 10.06825.   
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. arXiv preprint arXiv:1705.03551, 2017.   
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.   
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models, 2020. URL https://arxiv.org/ab s/1911.00172.   
Johannes Kiesel, Maria Mestre, Rishabh Shukla, Emmanuel Vincent, Payam Adineh, David Corney, Benno Stein, and Martin Potthast. SemEval-2019 task 4: Hyperpartisan news detection. In Jonathan May, Ekaterina Shutova, Aurelie Herbelot, Xiaodan Zhu, Marianna Apidianaki, and Saif M. Mohammad, editors, Proceedings of the 13th International Workshop on Semantic Evaluation, pages 829–839, Minneapolis, Minnesota, USA, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/S19-2145. URL https://aclanthology.org/S19-2145/.   
Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7: 453–466, 2019.   
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks, 2021. URL https://arxiv. org/abs/2005.11401.   
Junyi Li, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. Halueval: A large-scale hallucination evaluation benchmark for large language models, 2023. URL https://arxiv.or g/abs/2305.11747.

Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods, 2022. URL https://arxiv.org/abs/2109.07958.   
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.   
Hanxiao Liu, Zihang Dai, David R. So, and Quoc V. Le. Pay attention to mlps, 2021. URL https: //arxiv.org/abs/2105.08050.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.   
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Dekang Lin, Yuji Matsumoto, and Rada Mihalcea, editors, Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL https://aclanthology.org/P11-1015/.   
Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016. URL https://arxiv.org/abs/1609.07843.   
OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, and Shyamal Anadkat. Gpt-4 technical report, 2024. URL https://arxiv.org/abs/2303.08774.   
Bo Pang and Lillian Lee. Seeing stars: exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the 43rd Annual Meeting on Association for Computational Linguistics, ACL ’05, page 115–124, USA, 2005a. Association for Computational Linguistics. doi: 10.3115/1219840.1219855. URL https://doi.org/10.3115/1219840.1219855.   
Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the ACL, 2005b.   
Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, and Jianfeng Gao. Check your facts and try again: Improving large language models with external knowledge and automated feedback, 2023. URL https: //arxiv.org/abs/2302.12813.   
Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun Huang. Memorag: Boosting long context processing with global memory-enhanced retrieval augmentation. In Proceedings of the ACM on Web Conference 2025, pages 2366–2377, 2025.   
Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.   
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.

Weijia Shi, Julian Michael, Suchin Gururangan, and Luke Zettlemoyer. Nearest neighbor zero-shot inference. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3254–3265, Abu Dhabi, United Arab Emirates, December 2022a. Association for Computational Linguistics. doi: 10.18653 /v1/2022.emnlp-main.214. URL https://aclanthology.org/2022.emnlp-main.214/.   
Weijia Shi, Julian Michael, Suchin Gururangan, and Luke Zettlemoyer. Nearest neighbor zero-shot inference. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3254–3265, 2022b.   
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In David Yarowsky, Timothy Baldwin, Anna Korhonen, Karen Livescu, and Steven Bethard, editors, Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA, October 2013. Association for Computational Linguistics. URL https://aclanthology.org/D13-1170/.   
Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, and Yiqun Liu. Parametric retrieval augmented generation, 2025. URL https://arxiv. org/abs/2501.15915.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
Tim Van Erven and Peter Harremos. Rényi divergence and kullback-leibler divergence. IEEE Transactions on Information Theory, 60(7):3797–3820, 2014.   
Weizhi Wang, Li Dong, Hao Cheng, Xiaodong Liu, Xifeng Yan, Jianfeng Gao, and Furu Wei. Augmenting language models with long-term memory. Advances in Neural Information Processing Systems, 36: 74530–74543, 2023.   
Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks, 2015. URL https://arxiv. org/abs/1410.3916.   
Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. arXiv preprint arXiv:1809.09600, 2018.   
Ping Yu, Mikel Artetxe, Myle Ott, Sam Shleifer, Hongyu Gong, Ves Stoyanov, and Xian Li. Efficient language modeling with sparse all-mlp, 2022. URL https://arxiv.org/abs/2203.06850.   
Lingxi Zhang, Yue Yu, Kuan Wang, and Chao Zhang. Arl2: Aligning retrievers for black-box large language models via self-guided adaptive relevance labeling. arXiv preprint arXiv:2402.13542, 2024.   
Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28, 2015a.   
Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28, 2015b.   
Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. Advances in neural information processing systems, 31, 2018.

# A. Implementation Details

Datasets For the general NLP tasks in Section 5.3, we utilize a heterogeneous corpus constructed by aggregating several publicly available sources that cover diverse domains relevant to common NLP tasks. Following the methodology from [Geng et al., 2024], this corpus comprises WikiText-103 Merity et al. [2016] for encyclopedic content, Amazon Reviews He and McAuley [2016] for user-generated product feedback, CC-NEWS Hamborg et al. [2017] for journalistic content, and IMDB Maas et al. [2011] for movie reviews and discussions.

This diverse mixture captures both formal and informal language patterns, spans multiple domains from news articles to consumer opinions, and provides comprehensive coverage of linguistic phenomena encountered in real-world NLP applications. The complete dataset is publicly available at: https://huggingface.co/datasets/wentingzhao/knn-prompt-datastore.

Evaluation Metrics For question answering benchmarks, following Cheng et al. [2024], we evaluate three Open Domain Question Answering datasets and HotpotQA using the Exact Match (EM) metric. For long-form QA evaluation, we employ three complementary metrics: MC1, which measures whether the model assigns the highest likelihood to the most accurate answer; MC2, which sums the normalized probabilities over all correct answers; and MC3, which evaluates whether the model assigns a higher average likelihood to true answers than to false ones. We report the average of these three metrics as the final performance measure for long-form QA tasks. For general NLP tasks, following the methodology from Shi et al. [2022b], we report results using the domain-conditional PMI scoring rule Holtzman et al. [2021]. For hallucination reduction evaluation, we use accuracy as the primary metric to assess the model’s ability to generate factually correct responses.

Hyperparameters In Table 4, we list the hyperparameters used for training the 1B MLP Memory model (excluding embedding parameters).

Table 4 | Hyperparameters for training the 1B MLP Memory model.   

<table><tr><td>Hyperparameter</td><td>Assignment</td></tr><tr><td>optimizer</td><td>AdamW</td></tr><tr><td>learning rate</td><td>4e-4</td></tr><tr><td>lr scheduler type</td><td>linear</td></tr><tr><td>warmup ratio</td><td>0.03</td></tr><tr><td>weight decay</td><td>0.0</td></tr><tr><td>epochs</td><td>5</td></tr><tr><td>flash attention</td><td>False</td></tr><tr><td>batch size</td><td>4</td></tr><tr><td>gradient accumulation steps</td><td>4</td></tr><tr><td>num GPUs</td><td>32</td></tr><tr><td>max train samples</td><td>33,000,000</td></tr></table>

# B. Sensitivity to Interpolation Weight ??

We conducted a comprehensive analysis of our method’s sensitivity to the interpolation weight ?? on the HaluEval benchmark using Mistral-7B-v0.3. Table 5 presents the results across three tasks: Dialogue, QA, and Summarization, with ?? values ranging from 0.1 to 0.9.

Our findings demonstrate that the method exhibits robust performance across a wide range of ??

values, with optimal performance generally observed in the 0.35-0.55 range. Specifically, the Dialogue task achieves its best performance at $\lambda = 0 . 4 5$ $( 6 4 . 0 7 \% )$ , QA peaks at $\lambda = 0 . 5 5$ $( 6 6 . 8 6 \% )$ , and Summarization reaches its maximum at $\lambda = 0 . 3 5$ (52.41%). Notably, all three tasks show consistent improvements over the baseline Mistral-7B-v0.3 model across the optimal range, with QA showing the most substantial gains (up to 10.08 points improvement).

The performance remains relatively stable within the 0.3-0.6 range, with only gradual degradation outside this interval. When ?? approaches extreme values (e.g., 0.9), performance deteriorates significantly, particularly for Dialogue and Summarization tasks, though still maintaining improvements over the baseline in the QA task.

These results confirm that our method is not overly sensitive to the specific choice of ?? within a reasonable range, making it practical for deployment without extensive hyperparameter tuning. The consistent improvements across different ?? values and tasks validate the robustness of our approach.

Table 5 | Performance sensitivity analysis of interpolation weight ?? on HaluEval benchmark using Mistral-7B-v0.3. Results are reported as accuracy $( \% )$ across three tasks: Dialogue, QA, and Summarization. The first row shows baseline Mistral-7B-v0.3 performance without memory augmentation. Bold values indicate the best performance for each task.   

<table><tr><td>λ</td><td>Dialogue</td><td>QA</td><td>Summarization</td></tr><tr><td>Mistral-7B-v0.3</td><td>57.18</td><td>53.99</td><td>50.27</td></tr><tr><td>0.10</td><td>56.80</td><td>59.86</td><td>50.92</td></tr><tr><td>0.20</td><td>59.43</td><td>62.01</td><td>51.87</td></tr><tr><td>0.30</td><td>61.99</td><td>64.11</td><td>52.17</td></tr><tr><td>0.35</td><td>63.01</td><td>64.96</td><td>52.41</td></tr><tr><td>0.40</td><td>63.88</td><td>66.03</td><td>52.11</td></tr><tr><td>0.45</td><td>64.07</td><td>66.55</td><td>51.55</td></tr><tr><td>0.50</td><td>63.57</td><td>66.57</td><td>51.39</td></tr><tr><td>0.55</td><td>63.24</td><td>66.86</td><td>50.73</td></tr><tr><td>0.60</td><td>62.38</td><td>66.30</td><td>49.79</td></tr><tr><td>0.70</td><td>59.65</td><td>64.77</td><td>47.72</td></tr><tr><td>0.80</td><td>56.42</td><td>62.78</td><td>46.53</td></tr><tr><td>0.90</td><td>49.71</td><td>60.36</td><td>46.67</td></tr></table>

# C. Inference Efficiency Analysis

Table 6 presents the computational cost breakdown for both Transformer and MLP architectures in terms of FLOPs per token. As demonstrated, the primary difference in computational efficiency stems from the absence of attention mechanisms in pure MLP models.

By comparing these computational requirements, we derive the theoretical speed ratio between the Transformer (denoted as $F L O P s _ { t . }$ ) and the MLP models (denoted as $F L O P s _ { m . }$ ):

$$
\frac {F L O P s _ {t}}{F L O P s _ {m}} \approx \frac {4 n _ {\text {l a y e r}} d _ {\text {m o d e l}} \left(2 d _ {\text {a t t n}} + d _ {f f}\right) + 2 n _ {\text {l a y e r}} n _ {\text {c t x}} d _ {\text {a t t n}}}{6 n _ {\text {l a y e r}} d _ {\text {m o d e l}} d _ {f f}} = 1 + \frac {n _ {\text {c t x}}}{1 2 d _ {\text {m o d e l}}}, \tag {10}
$$

??????ℎ ??ℎ?? ???????????????? $d _ { a t t n } = d _ { f f } / 4 = d _ { m o d e l }$

This relationship in Equation 10 reveals that MLPs maintain a consistent computational advantage across all context lengths, with the efficiency gap widening as context length increases.

Table 6 | Flops per Token at inference time. Following Kaplan et al. [2020], we analyze computational requirements for Transformer and MLP architectures where $n _ { l a y e r }$ (number of layers), $d _ { m o d e l }$ (dimension of the residual stream), $d _ { f f }$ (dimension of the intermediate feed-forward layer), $d _ { a t t n }$ (dimension of the attention output) , $n _ { h e a d s }$ (number of attention heads per layer), $n _ { c t x }$ (the length of input context), $n _ { \nu o c a b }$ (vocabulary size). ?? ?? ???????????? denotes computational cost per token inference step.   

<table><tr><td>Openration</td><td>FLOPs per Token(Transformer)</td><td>FLOPs per Token(MLP)</td></tr><tr><td>Embed</td><td>4dmodel</td><td>-</td></tr><tr><td>Attention: QKV</td><td>2nlayerdmodel3dattn</td><td>-</td></tr><tr><td>Attention: Mask</td><td>2nlayernctxdattn</td><td>-</td></tr><tr><td>Attention: Project</td><td>2nlayerdattndmodel</td><td>-</td></tr><tr><td>Feedforward</td><td>2nlayer2dmodeldff</td><td>3nlayer2dmodeldff</td></tr><tr><td>De-embed</td><td>2dmodelnvocab</td><td>2dmodelnvocab</td></tr><tr><td>Total(Non-Embedding)</td><td>Cforward = 4nlayerdmodel(2dattn + dff) +2nlayernctxdattn</td><td>Cforward = 6nlayerdmodeldff</td></tr></table>

# D. Comparing Output Distribution Characteristics of LM, ??NN, and MLP Memory

As two samples illustrated in Figure 7, distributions produced by LM, ??NN search, and MLP Memory exhibit distinct characteristics. LM typically yields smooth and dense probability distributions, as it is trained to generalize across large corpora and capture broad contextual patterns.

In contrast, the kNN approach produces sparse and spiky distributions, concentrating most of the probability mass on only a few retrieved neighbors. For instance, when using a GPT-2 model (vocabulary size 50,257), even after retrieving the top- $k$ neighbors (e.g., $k = 1 0 2 4 ,$ ), only a small subset of these neighbors meaningfully influences the output distribution, while the majority receive near-zero probability.

The MLP Memory lies between LMs and kNN in terms of distribution characteristics. As a neural model, it is trained using a combination of KL loss and CE loss to approximate the sparse and spiky distributions produced by the kNN approach. While its outputs remain somewhat smoother due to the training objective, the resulting distributions are sharper than those of standard LMs, yet not as extreme as the highly concentrated outputs of kNN.

Table 7 | Number of tokens with non-zero probability mass at different thresholds. This table reports the number of tokens assigned non-zero probabilities by the LM, ??NN, and MLP Memory, across a range of probability thresholds. All values are averaged over 20,000 test samples.   
Table 7 compares the output sparsity of LM, ??NN, and MLP Memory by reporting the number of tokens assigned non-zero probabilities at various thresholds. The LM assigns non-zero mass to all 50,257   

<table><tr><td>Types</td><td>&gt;0.0</td><td>&gt;10-6</td><td>&gt;10-5</td><td>&gt;10-4</td><td>&gt;10-3</td><td>&gt;10-2</td><td>&gt;10-1</td></tr><tr><td>LM</td><td>50257</td><td>1760</td><td>562</td><td>148</td><td>34</td><td>7</td><td>2</td></tr><tr><td>kNN</td><td>251</td><td>217</td><td>197</td><td>146</td><td>43</td><td>9</td><td>2</td></tr><tr><td>MLP</td><td>50257</td><td>1151</td><td>388</td><td>115</td><td>30</td><td>7</td><td>2</td></tr></table>

![](images/c484b0b15b0a4c74c0a5261ee8caaa5eeb62e40c86ed056cd04d39c10c800d1d.jpg)

![](images/48cbc254f6968ec4b9210384a5b78910690eb95df7f59b8d4c8558ed4c052c7b.jpg)  
Figure 7 | Comparison of output probability distributions. Two samples show the top-16 probabilities from the LM and ??NN distributions using GPT2-large, along with the distribution generated by the MLP Memory based on the same large model size.

tokens, reflecting its dense distribution. However, this number drops sharply at higher thresholds, with only 2 tokens receiving probabilities above 0.1, indicating a rapid decay despite its broad support.

In contrast, the ??NN output is highly sparse, with only 251 tokens assigned any non-zero probability. Even at low thresholds (e.g., $1 0 ^ { - 6 }$ ), the number remains limited, confirming its concentrated nature shaped by a small set of retrieved neighbors.

MLP Memory exhibits intermediate behavior. Although it outputs over the full vocabulary like the LM, the number of tokens exceeding higher thresholds aligns more closely with ??NN. This suggests that MLP Memory learns to approximate the spiky distributions of ??NN while maintaining some smoothness from its parametric formulation.

Table 8 | Cumulative token count required to reach probability mass thresholds. This table indicates the number of top-ranked tokens needed to accumulate a total probability mass exceeding thresholds such as 0.8, 0.9, etc. All values are averaged over 20,000 test samples.   

<table><tr><td>Types</td><td>Top Prob Count (sum &gt; 0.8)</td><td>sum &gt; 0.9</td><td>sum &gt; 0.95</td><td>sum &gt; 0.99</td></tr><tr><td>LM</td><td>23</td><td>63</td><td>142</td><td>617</td></tr><tr><td>kNN</td><td>22</td><td>43</td><td>68</td><td>126</td></tr><tr><td>MLP</td><td>13</td><td>33</td><td>72</td><td>308</td></tr></table>

Table 8 further examines distribution sharpness by reporting the number of top-ranked tokens needed to accumulate a specified proportion of total probability mass. Here, we observe that the ??NN distribution reaches $9 9 \%$ cumulative probability with only 126 tokens, while the language model (LM) requires 617 tokens to achieve the same threshold. This suggests that the LM’s probability mass is more broadly spread across the vocabulary, in contrast to the highly concentrated outputs of ??NN.

Interestingly, the MLP Memory achieves $9 9 \%$ cumulative probability with 308 tokens, placing it between LM and ??NN. Notably, the MLP reaches $8 0 \%$ total probability with only 13 tokens—fewer than both LM and ??NN—indicating that it captures prominent signals more efficiently. These results support the observation that MLP Memory produces sharper distributions than LMs, yet avoids the

extreme sparsity of ??NN.

# E. Effect of Different ??NN Target Distributions

Figure 8 presents the test perplexity of our overall model architecture evaluated at various training steps. In all settings, both the base language model and the MLP Memory are of small size (GPT2- small), with the MLP Memory trained to mimic ??NN target distributions constructed from different base models: GPT2-small, GPT2-medium, GPT2-large, and GPT2-xl. As training progresses, test perplexity steadily declines across all variants, indicating stable optimization and effective learning. Among them, the model trained on the ??NN-XL distribution achieves the lowest final test perplexity (12.84), closely followed by the one trained on ??NN-large (12.85). In contrast, the models trained on ??NN-medium and ??NN-small converge to higher perplexities of approximately 12.87 and 12.91, respectively.

These results demonstrate that ??NN target distributions derived from larger base models lead to improved performance when used to train the MLP Memory. The richer and more informative supervision encoded in these distributions appears to enhance the parametric memory’s generalization ability.

![](images/5c7c6b146402cda0e16440505e54e0f2d8052293ee0c6702156626c36b0641ca.jpg)  
Figure 8 | Test perplexity of our overall model architecture, where both the base language model and the MLP Memory are of small size (GPT2-small). The MLP Memory is trained to mimic different $k \mathrm { N N }$ target distributions constructed from various base models: ??NN-small (GPT2-small), ??NN-med (GPT2-medium), ??NN-large (GPT2-large), and ??NN-XL (GPT2-xl).

# F. Sensitivity to $k$ in Target Distribution Generation

We used $k = 1 0 2 4$ for generating all target distributions. Table 9 shows the sensitivity analysis using GPT2-large on WikiText-103. While smaller $k$ values degrade performance, values beyond 1024

yield minimal gains while significantly increasing computational costs, making $k = 1 0 2 4$ optimal for practical deployment.

Table 9 | Test perplexity sensitivity to different values of $k$ in target distribution generation using GPT2-large on WikiText-103.   

<table><tr><td>Models</td><td>k</td><td>Perplexity</td></tr><tr><td>GPT2-large</td><td>-</td><td>10.43</td></tr><tr><td></td><td>1</td><td>10.30</td></tr><tr><td></td><td>2</td><td>10.11</td></tr><tr><td></td><td>4</td><td>9.95</td></tr><tr><td></td><td>8</td><td>9.83</td></tr><tr><td></td><td>16</td><td>9.71</td></tr><tr><td>+kNN-LM</td><td>32</td><td>9.63</td></tr><tr><td></td><td>64</td><td>9.57</td></tr><tr><td></td><td>128</td><td>9.52</td></tr><tr><td></td><td>256</td><td>9.48</td></tr><tr><td></td><td>512</td><td>9.46</td></tr><tr><td></td><td>1024</td><td>9.43</td></tr><tr><td></td><td>2048</td><td>9.42</td></tr></table>

# G. Case Study

As show in Figure 9, we observe that RAG often fails even with relevant retrieved documents due to contextual noise interference. When documents contain related but distracting information, RAG’s shallow integration cannot effectively filter these distractors, leading to incorrect answers. In contrast, MLP Memory learns intelligent corpus compression during training, capturing richer contextual relationships that enable robust disambiguation without explicit retrieval.

# RAG

Refer to the background document and answer the question:

Background: 1. The SS Security Service, known as the SS SD-Amt, became the official security organization of the Nazi Party in 1934. Consisting at first of paid agents and a few hundred unpaid informants scattered across Germany, the SD was quickly professionalized under Heydrich, who commissioned National Socialist academics and lawyers to ensure that the SS and its Security Service in particular, operated \"within the framework of National Socialist ideology.\" Heydrich was given the power to select men for the SS Security Service from among any SS subdivisions since Himmler considered the organization of the SD as important. In September 1939, the SD was divided into two departments, the interior department (Inland-SD) and the foreign department (Ausland-SD), and placed under the authority of the Reich Security Main Office (RSHA).

Question: In WWII, who was the head of the Nazi party's security service? The answer is:

![](images/00c1d7287257b097bf6999ddf3854b2f1f1240a74f8903b7a8b77fb84a32602f.jpg)

Output: 1. Heinrich Himm

![](images/e3ec6d8f9813c62d342eaee72438f9b76fc573aa771482c5ed63dd1c9dbe0f48.jpg)

# Base LM

Answer the question:

Question: In WWII, who was the head of the Nazi party's security service? The answer is:

# MLP Memory

Answer the question:

Question: In WWII, who was the head of the Nazi party's security service? The answer is:

Output: Question: In WW

![](images/2f5ba5b26f2791f6bccde9da29b232236858dae7ebb3e3f492dea44ed8eb6904.jpg)

Output: Reinhard Heydrich.

![](images/a75a2da940e064e5f3f2d14d1af0ee911f3a325a940b65fd3c88cac5ba287892.jpg)  
Figure 9 | Comparison of model outputs on a factual question. Despite retrieving relevant documents with correct information (highlighted in green), RAG is misled by contextual distractors and produces an incorrect answer. MLP Memory generates the correct answer without explicit retrieval.

# RAG

Refer to the background document and answer the question:

Background: 1. The Theory of Flight (1998) ; Mystics (2003) ; Harry Potter and the Order of the Phoenix (2007) ; Harry Potter and the Half-Blood Prince (2009) ; Harry Potter and the Deathly Hallows – Part 1 (2010) ; Harry Potter and the Deathly Hallows – Part 2 (2011) ; The Company You Keep (2012) ; About Time (2013) ; Ex Machina (2014) ; The Legend of Tarzan (2016) ; Fantastic Beasts and Where to Find Them (2016) ; Fantastic Beasts: The Crimes of Grindelwald (2018) ; Fantastic Beasts: The Secrets of Dumbledore (2022)

Question: what are the names of harry potter movies in order? The answer is:

![](images/005170bcae5bf593b66a7956f2e974737cc4fde845987efc3aa0a7baa13bccb7.jpg)

Output: 1. Harry Potter and the Philosopher’s Stone (

![](images/7f519c9e7a70de24c19ef44d557719d87b7fb28e5ba43fb22a41ea935704dc3c.jpg)

# MLP Memory

# Base LM

Answer the question:

Question: what are the names of harry potter movies in order? The answer is:

Answer the question:

Output: 1. Harry Potter and

![](images/90f254168069fb6954b3aa8dab3bf3874c7bfdf097ff2c96be8bf4546d2d0fa9.jpg)

Question: what are the names of harry potter movies in order? The answer is:

Output: Harry Potter and the Philosopher's Stone, Harry Potter and the Chamber of Secrets, Harry Potter and the Prisoner of Azkaban, Harry Potter and the Goblet of Fire, Harry Potter and the Order of the Phoenix, Harry Potter and the Half-Blood Prince, Harry Potter and the Deathly Hallows – Part 1, Harry Potter and the Deathly Hallows – Part 2.

![](images/0c39a54a96d4100780c4244d3903e58a7eb9d13615c649b53276693e368a3228.jpg)  
Figure 10 | RAG retrieves irrelevant documents that introduce interference, while MLP Memory demonstrates perfect performance.

# RAG

Refer to the background document and answer the question:

Background: 1. The Nordic countries are a geographical and cultural region in Northern Europe and the North Atlantic, which includes the countries of Denmark, Finland, Iceland, Norway and Sweden, and the autonomous territories of the Faroe Islands and Greenland. Anthropology has a diverse history in the Nordic countries, tracing all the way back to the early nineteenth century with the establishment of ethnographic museums.

Question: where are the nordic countries? The answer is:

![](images/6061ec85deee7ab984c87a4187bfe20b439eb4adacf5cbb5a8009acf1813b2b4.jpg)

Output: Nordic

![](images/3bdb5390f60211b88924019fe98aea6b36daafcb9f8af85db42f74304c85b04c.jpg)

# Base LM

Answer the question:

Question: where are the nordic countries? The answer is:

Output: Question:

![](images/e6507607d4b81766fd836b0d62c68ccc61ecc7630ccb3e54935a56bea51813c3.jpg)

# MLP Memory

Answer the question:

Question: where are the nordic countries? The answer is:

Output: $< \mathtt { S } >$ The Nordic countries are located in Northern Europe.<s> The Nordic countries are: Denmark,

![](images/6699b2eddb85f3226a770ab12707e820894f65e33dc213fe5d5c4071b6ad47c8.jpg)  
Figure 11 | RAG fails to extract answer despite retrieving relevant content, while MLP Memory provides accurate response

#

Refer to the background document and answer the question:

Background: 1. London

![](images/a2d3acc0847c55311a964446a7e756b38ee44ef68cac0182a043620347699c0d.jpg)

Question: what to visit in london city? The answer is:

Output: 1. The London

![](images/4d4a90ac4a067548ffbb5af13aeb92b80a172ca78f746f3247f5b4e816d5f695.jpg)

# Base LM

Answer the question:

Question: what to visit in london city? The answer is:

Output: London is a city that

![](images/d8a199f0a11c44e6cd0b5a7258ae3132f6c7d66d1bd35e9a5a22ad75edefdee2.jpg)

Answer the question:

Question: what to visit in london city? The answer is:

Output: 1. Big Ben 2. Buckingham Palace 3. Tower Bridge 4. Tower of London 5. London Eye 6. Westminster Abbey 7. St Paul's Cathedral 8. Tate Modern 9. National Gallery 10. British Museum 11. Natural History Museum 12. Science Museum 13. London Zoo 14. Madame

Tussauds 15. London Dungeon 16

![](images/6e52cc7e69c68c9df4702666cebeaf249f2e5b5331c1d3ac8c63e8a687526899.jpg)  
Figure 12 | RAG system distracted by retrieved content fails to answer the question, while MLP Memory provides comprehensive and accurate response

# MLP Memory
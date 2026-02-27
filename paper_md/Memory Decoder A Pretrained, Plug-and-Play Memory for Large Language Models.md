# Memory Decoder: A Pretrained, Plug-and-Play Memory for Large Language Models

Jiaqi $\mathbf { C a o ^ { 1 , 4 * } }$ , Jiarui Wang1∗, Rubin Wei1, Qipeng Guo2, Kai Chen2, Bowen Zhou2,3, Zhouhan Lin1,2‡

1LUMIA Lab, School of Artificial Intelligence, Shanghai Jiao Tong University   
2Shanghai AI Laboratory 3Tsinghua University   
4SJTU Paris Elite Institute of Technology

#maximus.cao@outlook.com #lin.zhouhan@gmail.com ∗ Equal Contribution. ‡ Corresponding Author.   
§ https://github.com/LUMIA-Group/MemoryDecoder   
https://huggingface.co/collections/Clover-Hill/memorydecoder

Abstract Large Language Models (LLMs) have shown strong abilities in general language tasks, yet adapting them to specific domains remains a challenge. Current method like Domain Adaptive Pretraining (DAPT) requires costly full-parameter training and suffers from catastrophic forgetting. Meanwhile, Retrieval-Augmented Generation (RAG) introduces substantial inference latency due to expensive nearest-neighbor searches and longer context. This paper introduces Memory Decoder, a plug-and-play pretrained memory that enables efficient domain adaptation without changing the original model’s parameters. Memory Decoder employs a small transformer decoder that learns to imitate the behavior of an external non-parametric retriever. Once trained, Memory Decoder can be seamlessly integrated with any pretrained language model that shares the same tokenizer, requiring no model-specific modifications. Experimental results demonstrate that Memory Decoder enables effective adaptation of various Qwen and Llama models to three distinct specialized domains: biomedicine, finance, and law, reducing perplexity by an average of 6.17 points. Overall, Memory Decoder introduces a novel paradigm centered on a specially pretrained memory component designed for domain-specific adaptation. This memory architecture can be integrated in a plug-and-play manner, consistently enhancing performance across multiple models within the target domain.

![](images/826f2bd367fcf3f724d68a960400c645bb52677c7f9abf82c8e128fdc94dc990.jpg)

![](images/c6132832b27810fd4494939ea1dee7369514c1b1b1e1eeb7611153304b055235.jpg)

![](images/18c0139502a2a7102fbe06ec3d3e75f14288e08a847b826ad9bd0b33af2d6bab.jpg)  
Figure 1 | Comparison of domain adaptation approaches. DAPT (left) requires separate pre-training for each model size, modifying original parameters. RAG (middle) maintains model parameters but requires expensive retrieval from external datastores during inference. Memory Decoder (right) offers a plug-and-play solution where a single pretrained memory component can be interpolated with models of different sizes, avoiding both parameter modification and retrieval overhead.

# 1. Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks [Grattafiori et al., 2024, Guo et al., 2025, Liu et al., 2024, Yang et al., 2024]. Pretrained on vast corpora of general text data, LLMs have revolutionized how we approach language understanding and generation tasks. However, despite their impressive general capabilities, adapting LLMs to perform optimally in specific domains remains a significant challenge. Domain-specific adaptation is crucial for applications in specialized fields such as biomedicine, finance, and law [Chen et al., 2023, Colombo et al., 2024, Liu et al., 2023b], where domain expertise and terminology are essential for accurate and reliable performance.

Domain adaptation for pretrained language models has traditionally followed several approaches, each with distinct advantages and limitations. Domain Adaptive Pre-Training (DAPT) involves continued pre-training of the LLM on domain-specific

![](images/4701e47537c70e60c802cf25d28bf383202363c78b181bbf9b0b2d4f4b6b9699.jpg)  
Figure 2 | Perplexity comparison of Qwen2.5 models augmented by Memory Decoder and LoRA adapter of the same param count on the finance domain.

corpora [Gururangan et al., 2020]. While effective, this approach suffers from substantial computational costs associated with full-parameter training, especially as model sizes continue to grow into billions of parameters. Furthermore, adapting multiple models to the same domain requires separate training runs for each model, leading to resource inefficiency. Even with successful DAPT implementation, these models often encounter catastrophic forgetting, where the adaptation process diminishes the model’s general capabilities [Kirkpatrick et al., 2017, van de Ven et al., 2024].

Retrieval-Augmented Generation (RAG) offers an alternative approach by enhancing model outputs with relevant retrieved information [Izacard et al., 2023, Lewis et al., 2020]. While this method preserves the original model parameters, it introduces substantial computation overhead during inference due to expensive nearest neighbor (kNN) searches across large datastores and extended context [He et al., 2021].

These two approaches present a fundamental dilemma in domain adaptation: DAPT requires costly training procedures and cannot efficiently adapt multiple models to the same domain, while RAG introduces significant computation and storage overhead during inference. This inherent trade-off between the plug-and-play nature of RAG and the inference efficiency of DAPT highlights the research gap for a solution offering both adaptability across models and computational efficiency during deployment. To address this challenge, we propose Memory Decoder (MemDec), a plug-and-play pretrained memory designed for efficient domain adaptation of large language models without modifying their parameters. Our approach draws inspiration from retrieval-based methods like kNN-LM [Khandelwal et al., 2019], but overcomes their limitations through a different paradigm. Rather than building and searching model-specific datastores during inference, Memory Decoder employs a small transformer decoder that is specially pretrained to imitate the behavior of non-parametric retrievers by aligning its output distribution with the ones of non-parametric retrievers. Figure 1 illustrates how our approach differs fundamentally from both DAPT and RAG.

The key innovation of our approach lies in its plug-and-play functionality: once trained, a single Memory Decoder can be seamlessly integrated with any large language model that shares the same tokenizer, without requiring model-specific adaptations or additional training. This architectural design enables immediate deployment across diverse model architectures, significantly reducing

the computational resources needed for domain adaptation pre-training. Furthermore, unlike RAG methods, Memory Decoder achieves domain-specific performance improvements with minimal impact on inference latency, combining versatility with computational efficiency.

Experimental results across three specialized domains (biomedicine, finance, and law) and multiple model architectures demonstrate the versatility of Memory Decoder. As shown in Figure 2. the same Memory Decoder with only 0.5B parameters consistently enhances performance across seven different models from the Qwen2.5 model family on the finance domain. Our comprehensive analysis confirms that Memory Decoder successfully preserves the advantages of non-parametric approaches while eliminating their computational overhead, establishing a new paradigm for efficient domain adaptation of LLMs.

Our contributions can be summarized as follows:

• We introduce Memory Decoder, a plug-and-play pretrained memory that enables efficient domain adaptation for large language models without modifying their original parameters.   
• We present the first approach that replaces traditional non-parametric retrievers with a compact parametric model, achieving superior performance while eliminating costly retrieval operations during inference.   
• We demonstrate Memory Decoder’s generalizability, where a single domain-specific pretrained memory can be seamlessly integrated across all models with the same tokenizer.

# 2. Background

# 2.1. Problem Formulation

Domain adaptation aims to enhance a pretrained language model’s performance on specialized text. Formally, given a pretrained model $M _ { \mathrm { P L M } }$ with parameters $\theta$ and a domain corpus $\mathcal { D } _ { \mathrm { d o m a i n } }$ , the goal is to optimize the next-token prediction distribution $p _ { \mathrm { P L M } } ( y _ { t } | x ; \theta )$ for the target domain. Here, $x = ( x _ { 1 } , x _ { 2 } , . . . , x _ { t - 1 } )$ represents the context sequence and $y _ { t }$ denotes the target token.

# 2.2. Nearest Neighbor Language Models

The $k$ -nearest neighbor language model (kNN-LM) [Khandelwal et al., 2019] enables non-parametric domain adaptation without modifying the pretrained model’s parameters.

For a domain corpus, kNN-LM first constructs a key-value datastore:

$$
(K, V) = \left\{\left(\phi \left(x _ {i}\right), y _ {i}\right) \mid \left(x _ {i}, y _ {i}\right) \in \mathcal {D} _ {\text {d o m a i n}} \right\} \tag {1}
$$

where $\phi ( \cdot )$ extracts hidden representations from the pretrained model.

During inference, for context $x$ , it computes $k _ { t } = \phi ( x )$ , retrieves $k$ -nearest neighbors, and constructs a probability distribution:

$$
p _ {\mathrm {k N N}} \left(y _ {t} \mid x\right) \propto \sum_ {\left(k _ {i}, v _ {i}\right) \in \mathcal {N} \left(k _ {t}, k\right)} \mathbb {1} _ {y _ {t} = v _ {i}} \exp \left(- d \left(k _ {t}, k _ {i}\right) / \tau\right) \tag {2}
$$

The final prediction interpolates between the pretrained model and kNN distributions:

$$
p _ {\mathrm {k N N - P L M}} \left(y _ {t} \mid x\right) = \lambda \cdot p _ {\mathrm {k N N}} \left(y _ {t} \mid x\right) + (1 - \lambda) \cdot p _ {\mathrm {P L M}} \left(y _ {t} \mid x\right) \tag {3}
$$

![](images/b731e0355330a5c38cfdf402b8a8285ebdf563f5dec529175dbd785ce32fd06d.jpg)

![](images/8a4182d7b3de5b6ec2c933a03d5eb1231d7bc63fd024b46cbfe8ba68d3eb33e8.jpg)  
Figure 3 | Overview of Memory Decoder architecture. Upper§ 3.1: During pre-training, Memory Decoder learns to align its output distributions with those generated by non-parametric retrievers through distribution alignment loss. Lower§ 3.2: During inference, Memory Decoder processes input in parallel with the base LLM, and their distributions are interpolated to produce domain-enhanced predictions without retrieval overhead.

While effective, kNN-LM introduces substantial computational and storage overhead during inference. For instance, the Wikitext-103 datastore requires nearly 500GB storage even for GPT2-small model [He et al., 2021]. These limitations motivate our Memory Decoder, a compact parametric model pretrained to mimic retrieval behavior while eliminating the need for large datastores.

# 3. Memory Decoder

In this section, we present Memory Decoder (MemDec), a plug-and-play pretrained memory designed for efficient domain adaptation of large language models. Our method consists of two primary components: a specialized pre-training procedure that aligns the output distribution of Memory Decoder with those of non-parametric retrievers (Section 3.1), and an efficient inference mechanism that enables plug-and-play domain adaptation (Section 3.2). As illustrated in Figure 3, Memory Decoder first learns to mimic non-parametric retrieval distributions during pre-training (upper part), then seamlessly integrates with any compatible language model during inference (lower part), eliminating the computational overhead associated with datastore maintenance and nearest neighbor search.

# 3.1. Pre-training

Our primary goal during pre-training is to enable Memory Decoder $M _ { \mathrm { M e m } }$ to produce probability distributions that closely resemble those generated by non-parametric retrievers when encountering the same context. This approach effectively encodes the domain knowledge captured in large key-value datastores into the parameters of our compact model.

Data Construction Since we require non-parametric distributions as supervision signals, we construct training pairs of $( x _ { i } , p _ { \mathrm { k N N } } ( \cdot | x _ { i } ) )$ in advance to enable efficient pre-training. Here, $x _ { i }$ represents the input context and $p _ { \mathrm { k N N } } ( \cdot | x _ { i } )$ denotes the probability distribution generated by the non-parametric retriever for that context. First, we build a key-value datastore $( K , V ) = \{ ( \phi ( x _ { i } ) , y _ { i } ) \mid ( x _ { i } , y _ { i } ) \in \mathcal { D } _ { \operatorname { t r a i n } } \}$ using our domain-specific corpus, where $\phi ( \cdot )$ extracts hidden representations from a specific layer of the pretrained model. For each context $x _ { i }$ in the corpus, we then perform $k$ -nearest neighbor search against this datastore to identify similar contexts. To avoid trivial self-retrieval that would contaminate the learning signal, we exclude the top-1 neighbor where its key exactly matches the query key. Finally, we compute the non-parametric distribution $p _ { \mathrm { k N N } } ( \cdot | x _ { i } )$ for each context using the retrieved neighbors and cache these context-distribution pairs for training.

Pre-training Objective Unlike traditional language modeling with single-label targets, kNN distributions offer richer supervision signals by capturing the diversity of plausible continuations in the domain [Xu et al., 2023](see Appendix C for detailed analysis on kNN distributions). Through extensive experimentation, we have identified that a hybrid objective yields optimal performance.

Our approach centers on a Distribution Alignment Loss that minimizes the KL divergence [Van Erven and Harremos, 2014] between Memory Decoder’s output distribution and the cached kNN distributions for each sample:

$$
\mathcal {L} _ {\mathrm {K L}} \left(x _ {i}\right) = \mathrm {K L} \left(p _ {\mathrm {k N N}} \left(\cdot \mid x _ {i}\right) \| p _ {\mathrm {M e m}} \left(\cdot \mid x _ {i}\right)\right) \tag {4}
$$

To prevent excessive deviation from the underlying corpus distribution, we integrate a complementary standard Language Modeling objective [Zhang and Sabuncu, 2018]:

$$
\mathcal {L} _ {\mathrm {L M}} \left(x _ {i}\right) = - \log p _ {\text {M e m}} \left(y _ {i} \mid x _ {i}\right) \tag {5}
$$

The final loss function balances these two objectives through a hyperparameter $\beta$

$$
\mathcal {L} \left(x _ {i}\right) = \beta \cdot \mathcal {L} _ {\mathrm {K L}} \left(x _ {i}\right) + (1 - \beta) \cdot \mathcal {L} _ {\mathrm {L M}} \left(x _ {i}\right) \tag {6}
$$

Previous failed attempts to learn kNN distributions and our conjecture on why vanilla KL divergence with cross-entropy regularization succeeds are detailed in Appendix D.

# 3.2. Inference

Once pretrained, Memory Decoder exhibits a key plug-and-play capability that allows it to adapt any language model with a compatible tokenizer to the target domain via simple interpolation. During inference, both the pretrained language model $M _ { \mathrm { P L M } }$ and Memory Decoder $M _ { \mathrm { M e m } }$ process the same input context in parallel, and their output distributions are interpolated:

$$
p _ {\text {M e m - P L M}} \left(y _ {t} \mid x\right) = \alpha \cdot p _ {\text {M e m}} \left(y _ {t} \mid x\right) + (1 - \alpha) \cdot p _ {\text {P L M}} \left(y _ {t} \mid x\right) \tag {7}
$$

where $\alpha \in [ 0 , 1 ]$ controls the influence of domain-specific knowledge.

Unlike traditional retrieval-augmented approaches that introduce substantial latency from nearest neighbor search and extended context processing, Memory Decoder requires only a single forward pass through a relatively small transformer decoder. As demonstrated in Figure 4, our method achieves significant improvements in inference efficiency compared to alternative domain adaptation techniques. With just $1 . 2 8 \times$ overhead relative to the base model, Memory Decoder substantially outperforms both In-Context RAG [Ram et al., 2023] $( 1 . 5 1 \times )$ and kNN-LM [Khandelwal et al., 2019] $( 2 . 1 7 \times )$ . This computational advantage, combined with Memory Decoder’s model-agnostic design, makes our approach particularly valuable for production environments where both performance and efficiency are critical considerations.

![](images/ebad101fd27df28e89e63038077fc599b1b88e7947dcc5521c3a27b257955a9c.jpg)  
Figure 4 | Inference latency comparison across domain adaptation methods. These measurements were conducted on Qwen2.5-1.5B [Yang et al., 2024] for biomedicine domain text, augmented by a 0.5B Memory Decoder.

# 4. Experimental Setup

Overview We evaluate Memory Decoder across four complementary settings: (1) Language modeling on WikiText-103 (§5.1) to demonstrate effectiveness across GPT-2 model scales; (2) Downstream tasks (§5.2) to verify preservation of general capabilities during domain adaptation; (3) Crossmodel adaptation (§5.3) showing a single Memory Decoder enhancing Qwen models from 0.5B to 72B parameters; (4) Cross-vocabulary adaptation (§5.4) demonstrating efficient transfer between tokenizer families; These experiments establish Memory Decoder as a versatile, plug-and-play solution for efficient domain adaptation across diverse architectures and applications.

Datasets For language modeling experiments, we use Wikitext-103 [Merity et al., 2016], a standard benchmark containing over 100M tokens from Wikipedia articles. For downstream evaluation, following the kNN-prompt framework, we assess performance across nine NLP tasks: sentiment analysis (SST2 [Socher et al., 2013], MR [Pang and Lee, 2005a], CR [Hu and Liu, 2004], RT [Pang and Lee, 2005b]), textual entailment (HYP [Kiesel et al., 2019], CB [De Marneffe et al., 2019], RTE [Dagan et al., 2010]), and text classification (AGN [Zhang et al., 2015a], Yahoo [Zhang et al., 2015b]). For domain-specific adaptation, we utilize three specialized corpora: (1) biomedical text from MIMIC-III [Johnson et al., 2016] clinical notes covering over 46,000 patients, (2) financial news [Liu et al., 2023a] from April 2024 to October 2024, and (3) legal text from the Asylex corpus [Barale et al., 2023] containing 59,112 documents of refugee status determination in Canada from 1996 to 2022.

Baselines We compare Memory Decoder against several established domain adaptation methods: In-Context RAG [Ram et al., 2023], which implements a BM25 retriever that processes 32 query tokens, with retrieval occurring every 4 tokens. kNN-LM [Khandelwal et al., 2019], configured with interpolation parameter $\lambda = 0 . 2 5$ and temperature settings of $\tau = 1$ for GPT-2 small and medium, and $\tau = 1 3$ for large and xl models. LoRA [Hu et al., 2022], applied to query, key, value and MLP layers, with rank adjusted for each model to achieve parameter counts comparable to Memory Decoder. Domain Adaptive Pretraining(DAPT) [Gururangan et al., 2020], which involves complete retraining of all model parameters on the domain-specific corpus.

Table 1 | Perplexity comparison of domain adaptation methods across GPT2 model sizes on Wikitext-103. The best performing results are highlighted in bold, while the second-best results are underlined. Notably, applying our Memory Decoder(124M) to GPT2-medium(345M) outperforms DAPT of GPT2- medium(345M), demonstrating the effectiveness of our approach in capturing domain knowledge without modifying original parameters.   

<table><tr><td></td><td>GPT2-small</td><td>GPT2-med</td><td>GPT2-large</td><td>GPT2-xl</td></tr><tr><td>base</td><td>24.89</td><td>18.29</td><td>15.80</td><td>14.39</td></tr><tr><td colspan="5">Non-parametric methods</td></tr><tr><td>+In-Context RAG</td><td>18.46</td><td>14.01</td><td>12.09</td><td>11.21</td></tr><tr><td>+kNN-LM</td><td>15.62</td><td>12.95</td><td>12.21</td><td>11.30</td></tr><tr><td colspan="5">Parametric methods</td></tr><tr><td>+DAPT</td><td>14.76</td><td>12.78</td><td>11.10</td><td>10.16</td></tr><tr><td>+LoRA</td><td>18.63</td><td>13.88</td><td>11.77</td><td>10.67</td></tr><tr><td>+MemDec</td><td>13.36</td><td>12.25</td><td>11.53</td><td>10.93</td></tr></table>

Training Details We conduct our experiments on an 8×A800 80GB GPU setup. For language modeling and downstream evaluations, we use a GPT2-xl model(finetuned on wikitext) to build the key-value datastore and non-parametric distributions for training, and continue training on a GPT2-small model(finetuned on wikitext) with learning rate 1e-3. For cross-model adaptation, we use Qwen2.5- 1.5B [Yang et al., 2024] to build the datastore, and continue training on Qwen2.5-0.5B with learning rate 1e-4. For cross-vocabulary adaptation, we use Llama3.2-1B [Grattafiori et al., 2024] to build the datastore, and continue training on the Memory Decoder trained from cross-model experiments, with its embedding layer and language model head re-initialized. All experiments use a training budget equivalent to the computational cost of training a 7B parameter model for 1 epoch, with DAPT and LoRA baselines using the same maximum training FLOPS but early stopped to prevent overfitting. The training hyperparameter $\beta$ is set to 0.5 across all tasks.

Evaluation Metrics For language modeling, cross-model, and cross-tokenizer experiments, we use sliding window perplexity. Following Baevski and Auli [2018], in each test example, the context length is set to 1024 where only the latter 512 tokens are scored. For downstream evaluation, following methodology from Shi et al. [2022], we report results using the domain-conditional PMI scoring rule [Holtzman et al., 2021]. The interpolation hyperparameter $\alpha$ is tuned on the validation split of each task following Khandelwal et al. [2019], see more details in Appendix A.

# 5. Results

# 5.1. Language Modeling on Wikitext-103

Table 1 demonstrates the exceptional effectiveness of Memory Decoder across all GPT2 model sizes. A single Memory Decoder with only 124M parameters consistently enhances the entire GPT2 family, showcasing its plug-and-play capability regardless of base model size. For smaller models, our approach delivers superior results compared to all adaptation methods—notably maintaining an advantage for GPT2-medium despite utilizing only one third of the parameters. Even when applied to larger models where DAPT has inherent advantages due to full model updates, Memory Decoder remains highly competitive while consistently outperforming all other parameter-efficient methods without modifying any original parameters. These results validate that a small parametric decoder can effectively capture the benefits of non-parametric retrieval while eliminating computational overhead.

Table 2 | Performance on nine diverse NLP tasks including sentiment analysis, textual entailment, and text classification.   

<table><tr><td></td><td>SST2</td><td>MR</td><td>CR</td><td>RT</td><td>HYP</td><td>CB</td><td>RTE</td><td>AGN</td><td>Yahoo</td><td>Avg</td></tr><tr><td>base</td><td>81.98</td><td>78.40</td><td>84.40</td><td>76.54</td><td>63.75</td><td>41.07</td><td>52.70</td><td>78.79</td><td>49.40</td><td>67.45</td></tr><tr><td colspan="11">Non-parametric methods</td></tr><tr><td>+kNN-LM</td><td>81.98</td><td>77.95</td><td>83.80</td><td>77.95</td><td>64.14</td><td>39.28</td><td>52.70</td><td>77.73</td><td>49.63</td><td>67.24</td></tr><tr><td colspan="11">Parametric methods</td></tr><tr><td>+DAPT</td><td>83.52</td><td>80.15</td><td>80.45</td><td>77.39</td><td>36.04</td><td>50.00</td><td>51.26</td><td>64.31</td><td>24.40</td><td>60.84</td></tr><tr><td>+LoRA</td><td>80.88</td><td>76.90</td><td>83.95</td><td>76.07</td><td>64.14</td><td>39.28</td><td>53.79</td><td>81.06</td><td>49.46</td><td>67.28</td></tr><tr><td>+MemDec</td><td>82.43</td><td>78.35</td><td>84.35</td><td>77.30</td><td>64.15</td><td>57.14</td><td>55.24</td><td>79.80</td><td>49.37</td><td>69.79</td></tr></table>

# 5.2. Downstream Performance

Table 2 reveals Memory Decoder’s ability to enhance domain adaptation while preserving general language capabilities in zero-shot evaluation settings. Unlike DAPT, which suffers catastrophic forgetting on several tasks (particularly HYP and Yahoo where performance drops by nearly half; see Appendix B for detailed analysis), Memory Decoder maintains or improves performance across all evaluated tasks. Our approach achieves the highest average score across all nine tasks, outperforming the base model, kNN-LM, and LoRA while demonstrating particular strength on textual entailment tasks like CB and RTE. These results validate a key advantage of our plug-and-play architecture: by keeping the original model parameters intact while augmenting them with domain knowledge, Memory Decoder achieves domain adaptation without sacrificing general capabilities. Importantly, all experiments are conducted in a zero-shot setting, and our method should be viewed as orthogonal to in-context learning approaches.

# 5.3. Cross-Model Adaptation

Table 3 demonstrates Memory Decoder’s exceptional plug-and-play capabilities across diverse model sizes and architectures. A single Memory Decoder (0.5B parameters) consistently enhances performance across all models in both the Qwen2 and Qwen2.5 families, spanning from 0.5B to 72B parameters. For smaller models like Qwen2-0.5B, our approach achieves dramatic perplexity reductions—transforming domain-specific performance to state-of-the-art results on both biomedical and financial text. Even for the largest models in the family, Memory Decoder provides substantial improvements, demonstrating that retrieval-augmented knowledge remains valuable regardless of model scale. These results validate Memory Decoder’s core strength: a single pretrained memory component can enhance multiple models sharing the same tokenizer, providing efficient domain adaptation that scales from the smallest to the largest models while consistently outperforming existing approaches.

# 5.4. Cross-Vocabulary Adaptation

Table 4 demonstrates Memory Decoder’s ability to generalize across different tokenizers and model architectures. By re-initializing only the embedding layer and language model head of our Qwen2.5- trained Memory Decoder, we successfully adapt it to the Llama model family with just $1 0 \%$ of the original training budget. This efficient transfer enables substantial performance improvements across all Llama variants. For Llama3-8B, Memory Decoder achieves roughly $5 0 \%$ perplexity reduction on both biomedical and financial domains. Similar improvements extend to the Llama3.1 and Llama3.2 families, with our method consistently outperforming LoRA on biomedical and financial domains, though showing room for improvement on legal text. These findings illustrate Memory Decoder’s

Table 3 | Cross-model adaptation results across three specialized domains. A single 0.5B Memory Decoder enhances models ranging from 0.5B to 72B parameters.   

<table><tr><td>Model</td><td>Bio</td><td>Fin</td><td>Law</td><td>Avg</td></tr><tr><td colspan="5">Qwen2 Family</td></tr><tr><td>Qwen2-0.5B</td><td>18.41</td><td>16.00</td><td>10.23</td><td>14.88</td></tr><tr><td>+LoRA</td><td>7.28</td><td>9.70</td><td>5.82</td><td>7.60</td></tr><tr><td>+MemDec</td><td>3.75</td><td>3.84</td><td>4.57</td><td>4.05</td></tr><tr><td>Qwen2-1.5B</td><td>12.42</td><td>10.96</td><td>7.69</td><td>10.36</td></tr><tr><td>+LoRA</td><td>5.73</td><td>7.37</td><td>4.84</td><td>5.98</td></tr><tr><td>+MemDec</td><td>3.68</td><td>3.61</td><td>4.32</td><td>3.87</td></tr><tr><td>Qwen2-7B</td><td>8.36</td><td>8.31</td><td>5.92</td><td>7.53</td></tr><tr><td>+LoRA</td><td>4.47</td><td>5.64</td><td>4.02</td><td>4.71</td></tr><tr><td>+MemDec</td><td>3.59</td><td>3.38</td><td>4.00</td><td>3.66</td></tr><tr><td>Qwen2-72B</td><td>6.15</td><td>6.62</td><td>4.84</td><td>5.87</td></tr><tr><td>+MemDec</td><td>3.45</td><td>3.20</td><td>3.69</td><td>3.45</td></tr><tr><td colspan="5">Qwen2.5 Family</td></tr><tr><td>Qwen2.5-0.5B</td><td>17.01</td><td>16.04</td><td>9.86</td><td>14.30</td></tr><tr><td>+LoRA</td><td>7.02</td><td>9.88</td><td>5.75</td><td>7.55</td></tr><tr><td>+MemDec</td><td>3.74</td><td>3.87</td><td>4.57</td><td>4.06</td></tr><tr><td>Qwen2.5-1.5B</td><td>11.33</td><td>11.20</td><td>7.42</td><td>9.98</td></tr><tr><td>+LoRA</td><td>5.59</td><td>7.50</td><td>4.82</td><td>5.97</td></tr><tr><td>+MemDec</td><td>3.67</td><td>3.61</td><td>4.29</td><td>3.86</td></tr><tr><td>Qwen2.5-3B</td><td>9.70</td><td>9.83</td><td>6.68</td><td>8.74</td></tr><tr><td>+LoRA</td><td>5.07</td><td>6.71</td><td>4.45</td><td>5.41</td></tr><tr><td>+MemDec</td><td>3.63</td><td>3.52</td><td>4.16</td><td>3.77</td></tr><tr><td>Qwen2.5-7B</td><td>8.19</td><td>8.61</td><td>5.94</td><td>7.58</td></tr><tr><td>+LoRA</td><td>4.03</td><td>5.31</td><td>3.81</td><td>4.38</td></tr><tr><td>+MemDec</td><td>3.57</td><td>3.42</td><td>4.01</td><td>3.67</td></tr><tr><td>Qwen2.5-14B</td><td>7.01</td><td>7.60</td><td>5.35</td><td>6.65</td></tr><tr><td>+MemDec</td><td>3.51</td><td>3.31</td><td>3.86</td><td>3.56</td></tr><tr><td>Qwen2.5-32B</td><td>6.65</td><td>7.38</td><td>5.18</td><td>6.40</td></tr><tr><td>+MemDec</td><td>3.48</td><td>3.29</td><td>3.81</td><td>3.53</td></tr><tr><td>Qwen2.5-72B</td><td>5.90</td><td>6.80</td><td>4.84</td><td>5.85</td></tr><tr><td>+MemDec</td><td>3.44</td><td>3.23</td><td>3.70</td><td>3.46</td></tr></table>

Table 4 | Cross-vocabulary adaptation results demonstrating efficient knowledge transfer between model families. Memory Decoder trained on Qwen2.5 can be adapted to Llama models with minimal additional training $1 0 \%$ of original budget), achieving substantial perplexity reductions across all Llama variants and consistently outperforming LoRA in biomedical and financial domains.   

<table><tr><td>Model</td><td>Bio</td><td>Fin</td><td>Law</td><td>Avg</td></tr><tr><td colspan="5">Llama3 Family</td></tr><tr><td>Llama3-8B</td><td>7.95</td><td>8.63</td><td>5.96</td><td>7.51</td></tr><tr><td>+LoRA</td><td>4.38</td><td>5.68</td><td>4.12</td><td>4.73</td></tr><tr><td>+MemDec</td><td>3.92</td><td>4.32</td><td>4.46</td><td>4.23</td></tr><tr><td>Llama3-70B</td><td>5.92</td><td>6.87</td><td>4.90</td><td>5.90</td></tr><tr><td>+MemDec</td><td>3.74</td><td>4.01</td><td>4.07</td><td>3.94</td></tr><tr><td colspan="5">Llama3.1 Family</td></tr><tr><td>Llama3.1-8B</td><td>7.82</td><td>8.46</td><td>5.88</td><td>7.39</td></tr><tr><td>+LoRA</td><td>4.38</td><td>5.72</td><td>4.10</td><td>4.73</td></tr><tr><td>+MemDec</td><td>3.91</td><td>4.30</td><td>4.42</td><td>4.21</td></tr><tr><td>Llama3.1-70B</td><td>5.85</td><td>6.68</td><td>4.89</td><td>5.81</td></tr><tr><td>+MemDec</td><td>3.73</td><td>3.97</td><td>4.06</td><td>3.92</td></tr><tr><td colspan="5">Llama3.2 Family</td></tr><tr><td>Llama3.2-1B</td><td>12.81</td><td>11.85</td><td>8.23</td><td>10.96</td></tr><tr><td>+LoRA</td><td>5.97</td><td>7.83</td><td>5.21</td><td>6.34</td></tr><tr><td>+MemDec</td><td>4.06</td><td>4.85</td><td>5.11</td><td>4.67</td></tr><tr><td>Llama3.2-3B</td><td>9.83</td><td>9.70</td><td>6.83</td><td>8.79</td></tr><tr><td>+LoRA</td><td>5.11</td><td>6.55</td><td>4.59</td><td>5.42</td></tr><tr><td>+MemDec</td><td>3.99</td><td>4.45</td><td>4.76</td><td>4.40</td></tr></table>

versatility beyond a single tokenizer family, demonstrating that domain knowledge learned from one architecture can be efficiently transferred to another with minimal additional training. This capability expands the practical utility of our approach, offering a streamlined path to domain adaptation across diverse model ecosystems.

# 6. Analysis

# 6.1. Case Study: Bridging Parametric and Non-Parametric Methods

Memory Decoder fundamentally learns to compress the knowledge stored in large non-parametric datastores into a compact parametric model, combining the memorization capabilities of retrieval methods with the efficiency and generalization of parametric approaches. To validate this hypothesis, we conducted case studies on WikiText-103 examining how different methods assign probabilities to specific tokens.

As shown in Table 5, Memory Decoder exhibits two crucial capabilities:

Long-tail Knowledge: For factual information like "Jacobi" and "1906", Memory Decoder assigns dramatically higher probabilities than the base model $6 8 . 9 4 \%$ vs. $0 . 1 2 \%$ and $9 8 . 6 5 \%$ vs. $1 . 5 7 \% )$ ), successfully capturing the memorization benefits of non-parametric methods.

Semantic Coherence: For function words and logical continuations like "on" and "C", Memory Decoder maintains probabilities closer to the base model rather than following kNN-LM’s lower probabilities, demonstrating its ability to preserve coherent language modeling capabilities that pure retrieval methods sacrifice.

<table><tr><td colspan="4">Long-tail Knowledge Learning</td></tr><tr><td>Context (target token underlined)</td><td>MemDec</td><td>kNN</td><td>Base LM</td></tr><tr><td>he starred alongside actors Mark Strong and Derek Jacobi</td><td>68.94%</td><td>9.39%</td><td>0.12%</td></tr><tr><td>The launch of HMS Dreadnought in 1906 by the Royal Navy raised the stakes</td><td>98.65%</td><td>40.62%</td><td>1.57%</td></tr><tr><td colspan="4">Semantic Coherence and Reasoning</td></tr><tr><td>Context (target token underlined)</td><td>MemDec</td><td>kNN</td><td>Base LM</td></tr><tr><td>In 2000 Boulter had a guest-starring role on the television series The Bill</td><td>40.11%</td><td>8.07%</td><td>45.51%</td></tr><tr><td>...three tank squadrons for special overseas operations, known as ‘A’, ‘B’ and ‘C’ Special Service Squadrons</td><td>50.10%</td><td>10.76%</td><td>63.04%</td></tr></table>

Table 5 | Probability assignments for specific tokens by different methods. Orange section: Memory Decoder excels at capturing long-tail factual knowledge, assigning dramatically higher probabilities than the base model. Cyan section: For semantic coherence, Memory Decoder intelligently balances between kNN-LM and base model probabilities, preserving linguistic fluency.   
Table 6 | Performance comparison with different Memory Decoder sizes. Even small Memory Decoders achieve competitive performance with full-parameter DAPT while maintaining plug-and-play capability.   

<table><tr><td></td><td>GPT2-small</td><td>GPT2-medium</td><td>GPT2-large</td><td>GPT2-xl</td><td>Avg</td></tr><tr><td>Base</td><td>24.89</td><td>18.29</td><td>15.80</td><td>14.39</td><td>18.34</td></tr><tr><td>DAPT</td><td>14.76</td><td>12.78</td><td>11.10</td><td>10.16</td><td>12.20</td></tr><tr><td>+ MemDec-small (124M)</td><td>13.36</td><td>12.25</td><td>11.53</td><td>10.93</td><td>12.01</td></tr><tr><td>+ MemDec-medium (345M)</td><td>12.08</td><td>11.59</td><td>10.92</td><td>10.43</td><td>11.26</td></tr><tr><td>+ MemDec-large (774M)</td><td>11.67</td><td>11.23</td><td>10.83</td><td>10.28</td><td>11.00</td></tr></table>

These observations confirm that Memory Decoder occupies a unique position: it enhances memorization of domain-specific and long-tail knowledge like non-parametric methods, while maintaining the generalization and reasoning capabilities inherent to parametric models.

# 6.2. Impact of Memory Decoder Size

Table 6 examines how Memory Decoder size affects performance across the GPT2 family. As Memory Decoder size increases, performance consistently improves across all base models, with the large variant achieving the best average perplexity. These results validate that Memory Decoder provides an efficient alternative to full model fine-tuning: practitioners can choose the decoder size based on their computational constraints while maintaining the crucial advantage of preserving the original model’s capabilities.

# 6.3. Ablation on the pre-training objective

We compare Memory Decoder against logit interpolation with a DAPT model. Table 7 shows Memory Decoder consistently outperforms DAPT interpolation across all GPT2 scales, with an average gain of

Table 7 | Memory Decoder vs. DAPT model interpolation on WikiText-103. Both use 124M parameter models with different training objectives.   

<table><tr><td>Base Model</td><td>Baseline PPL</td><td>+ DAPT-small</td><td>+ MemDec-small</td></tr><tr><td>GPT2-small</td><td>24.89</td><td>15.95</td><td>13.36 (-2.59)</td></tr><tr><td>GPT2-medium</td><td>18.29</td><td>14.26</td><td>12.25 (-2.01)</td></tr><tr><td>GPT2-large</td><td>15.80</td><td>13.13</td><td>11.53 (-1.60)</td></tr><tr><td>GPT2-xl</td><td>14.39</td><td>12.30</td><td>10.93 (-1.37)</td></tr><tr><td>Average</td><td>18.34</td><td>13.91</td><td>12.01 (-1.90)</td></tr></table>

1.90 perplexity points. Notably, the gains persist even at larger scales, confirming that our hybrid pre-training objective provides complementary value beyond what standard language modeling objectives can achieve, regardless of model size.

# 7. Related Work

Retrieval-Augmented Generation Retrieval-Augmented Generation (RAG) enhances language models by incorporating knowledge from external sources, with retrieval granularity ranging from documents [Chen et al., 2017] to passages [Guu et al., 2020, Izacard et al., 2023, Lewis et al., 2020] to tokens [He et al., 2021, Khandelwal et al., 2019, Min et al., 2022, Yogatama et al., 2021]. Tokenlevel retrieval achieves superior performance for rare patterns and out-of-domain scenarios but introduces substantial computation overhead during inference. While non-differentiable retrieval mechanisms prevent end-to-end optimization and memory token approaches [Chevalier et al., 2023] enable differentiable access but are limited to local contexts, Memory Decoder provides both differentiable optimization and full-dataset knowledge access without expensive retrieval operations or model-specific datastores.

Domain Adaptation Domain adaptation techniques have evolved from domain-specific pre-training (SciBERT [Beltagy et al., 2019], BioBERT [Lee et al., 2020], ClinicalBERT [Huang et al., 2019]) to parameter-efficient methods like LoRA [Hu et al., 2022] and adapters [Diao et al., 2021, 2023, Wang et al., 2020]. However, these approaches require model-specific modifications, preventing generalization across architectures. Memory Decoder addresses this limitation by providing a domain-specific memory module that enhances multiple frozen language models without parameter modifications, enabling cross-model adaptation within tokenizer families and efficient cross-tokenizer transfer with minimal additional training.

# 8. Conclusion

In this paper, we introduced Memory Decoder, a novel plug-and-play approach for domain adaptation of large language models. By pre-training a small transformer decoder to emulate the behavior of non-parametric retrievers, Memory Decoder effectively adapts any compatible language model to a specific domain without modifying its parameters. Our comprehensive experiments across multiple model families and specialized domains demonstrate that Memory Decoder consistently outperforms both parametric adaptation methods and traditional retrieval-augmented approaches.

The key innovation of Memory Decoder lies in its versatility and efficiency. A single pretrained Memory Decoder can seamlessly enhance any model that shares the same tokenizer, and with minimal additional training, can be adapted to models with different tokenizers and architectures. This capability enables efficient domain adaptation across model families, dramatically reducing the resources typically required for specialized model development. Our results confirm that Memory

Decoder preserves the performance benefits of retrieval-augmented methods while maintaining the general capabilities of the base models, avoiding the catastrophic forgetting often observed with fine-tuning approaches.

Memory Decoder introduces a new paradigm for domain adaptation that fundamentally reimagines how we specialize language models for particular domains. By decoupling domain expertise from model architecture through a pretrained memory component, our approach creates a more modular, efficient, and accessible framework for enhancing language model performance in specialized fields.

# 9. Limitations

While Memory Decoder demonstrates significant advantages for domain adaptation, we acknowledge several limitations in our current approach. The pre-training phase requires searching in key-value datastores to obtain kNN distributions as training signals, introducing computational overhead during the training process. Although this cost is incurred only once per domain and is amortized across all adapted models, it remains a bottleneck in the pipeline. Additionally, while cross-tokenizer adaptation requires minimal training compared to training from scratch, it still necessitates some parameter updates to align embedding spaces, preventing truly zero-shot cross-architecture transfer.

# Acknowledgement

This work is sponsored by the National Key Research and Development Program of China (No. 2023ZD0121402), Shanghai Fundamental Research Program for General AI Models (No. 2025SHZDZX0251101), and the National Natural Science Foundation of China (NSFC) grant (No.62576211).

# References

Alexei Baevski and Michael Auli. Adaptive input representations for neural language modeling. arXiv preprint arXiv:1809.10853, 2018.   
Claire Barale, Michael Rovatsos, and Nehal Bhuta. Automated refugee case analysis: An nlp pipeline for supporting legal practitioners. arXiv preprint arXiv:2305.15533, 2023.   
Iz Beltagy, Kyle Lo, and Arman Cohan. Scibert: A pretrained language model for scientific text. arXiv preprint arXiv:1903.10676, 2019.   
Danqi Chen, Adam Fisch, Jason Weston, and Antoine Bordes. Reading wikipedia to answer opendomain questions. arXiv preprint arXiv:1704.00051, 2017.   
Yirong Chen, Zhenyu Wang, Xiaofen Xing, Zhipei Xu, Kai Fang, Junhong Wang, Sihang Li, Jieling Wu, Qi Liu, Xiangmin Xu, et al. Bianque: Balancing the questioning and suggestion ability of health llms with multi-turn health conversations polished by chatgpt. arXiv preprint arXiv:2310.15896, 2023.   
Daixuan Cheng, Shaohan Huang, and Furu Wei. Adapting large language models via reading comprehension. In The Twelfth International Conference on Learning Representations, 2023.   
Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to compress contexts. arXiv preprint arXiv:2305.14788, 2023.   
Pierre Colombo, Telmo Pessoa Pires, Malik Boudiaf, Dominic Culver, Rui Melo, Caio Corro, Andre FT Martins, Fabrizio Esposito, Vera Lúcia Raposo, Sofia Morgado, et al. Saullm-7b: A pioneering large language model for law. arXiv preprint arXiv:2403.03883, 2024.   
Ido Dagan, Bill Dolan, Bernardo Magnini, and Dan Roth. Recognizing textual entailment: Rational, evaluation and approaches–erratum. Natural Language Engineering, 16(1):105–105, 2010.   
Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser. The commitmentbank: Investigating projection in naturally occurring discourse. In proceedings of Sinn und Bedeutung, volume 23, pages 107–124, 2019.   
Shizhe Diao, Ruijia Xu, Hongjin Su, Yilei Jiang, Yan Song, and Tong Zhang. Taming pre-trained language models with n-gram representations for low-resource domain adaptation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 3336–3349, 2021.   
Shizhe Diao, Tianyang Xu, Ruijia Xu, Jiawei Wang, and Tong Zhang. Mixture-of-domain-adapters: Decoupling and injecting domain knowledge to pre-trained language models memories. arXiv preprint arXiv:2306.05406, 2023.   
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
Suchin Gururangan, Ana Marasović, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A Smith. Don’t stop pretraining: Adapt language models to domains and tasks. arXiv preprint arXiv:2004.10964, 2020.   
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929–3938. PMLR, 2020.   
Junxian He, Graham Neubig, and Taylor Berg-Kirkpatrick. Efficient nearest neighbor language models. arXiv preprint arXiv:2109.04212, 2021.   
Ari Holtzman, Peter West, Vered Shwartz, Yejin Choi, and Luke Zettlemoyer. Surface form competition: Why the highest probability answer isn’t always right. arXiv preprint arXiv:2104.08315, 2021.   
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.   
Minqing Hu and Bing Liu. Mining and summarizing customer reviews. In Proceedings of the Tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’04, page 168–177, New York, NY, USA, 2004. Association for Computing Machinery. ISBN 1581138881. doi: 10.1145/1014052.1014073. URL https://doi.org/10.1145/1014052.1014073.   
Kexin Huang, Jaan Altosaar, and Rajesh Ranganath. Clinicalbert: Modeling clinical notes and predicting hospital readmission. arXiv preprint arXiv:1904.05342, 2019.   
Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research, 24(251):1–43, 2023.   
Alistair EW Johnson, Tom J Pollard, Lu Shen, Li-wei H Lehman, Mengling Feng, Mohammad Ghassemi, Benjamin Moody, Peter Szolovits, Leo Anthony Celi, and Roger G Mark. Mimic-iii, a freely accessible critical care database. Scientific data, 3(1):1–9, 2016.   
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. arXiv preprint arXiv:1911.00172, 2019.   
Johannes Kiesel, Maria Mestre, Rishabh Shukla, Emmanuel Vincent, Payam Adineh, David Corney, Benno Stein, and Martin Potthast. SemEval-2019 task 4: Hyperpartisan news detection. In Jonathan May, Ekaterina Shutova, Aurelie Herbelot, Xiaodan Zhu, Marianna Apidianaki, and Saif M. Mohammad, editors, Proceedings of the 13th International Workshop on Semantic Evaluation, pages 829–839, Minneapolis, Minnesota, USA, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/S19-2145. URL https://aclanthology.org/S19-2145/.   
James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13): 3521–3526, 2017.   
Jinhyuk Lee, Wonjin Yoon, Sungdong Kim, Donghyeon Kim, Sunkyu Kim, Chan Ho So, and Jaewoo Kang. Biobert: a pre-trained biomedical language representation model for biomedical text mining. Bioinformatics, 36(4):1234–1240, 2020.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.   
Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.   
Xiao-Yang Liu, Guoxuan Wang, Hongyang Yang, and Daochen Zha. Data-centric fingpt: Democratizing internet-scale data for financial large language models. NeurIPS Workshop on Instruction Tuning and Instruction Following, 2023a.   
Xiao-Yang Liu, Guoxuan Wang, Hongyang Yang, and Daochen Zha. Fingpt: Democratizing internetscale data for financial large language models. arXiv preprint arXiv:2307.10485, 2023b.   
Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models, 2016. URL https://arxiv.org/abs/1609.07843.   
Sewon Min, Weijia Shi, Mike Lewis, Xilun Chen, Wen-tau Yih, Hannaneh Hajishirzi, and Luke Zettlemoyer. Nonparametric masked language modeling. arXiv preprint arXiv:2212.01349, 2022.   
Bo Pang and Lillian Lee. Seeing stars: exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the 43rd Annual Meeting on Association for Computational Linguistics, ACL ’05, page 115–124, USA, 2005a. Association for Computational Linguistics. doi: 10.3115/1219840.1219855. URL https://doi.org/10.3115/1219840.1219855.   
Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales. In Proceedings of the ACL, 2005b.   
Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 11:1316–1331, 2023.   
Weijia Shi, Julian Michael, Suchin Gururangan, and Luke Zettlemoyer. Nearest neighbor zero-shot inference. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 3254–3265, 2022.   
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In David Yarowsky, Timothy Baldwin, Anna Korhonen, Karen Livescu, and Steven Bethard, editors, Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631–1642, Seattle, Washington, USA, October 2013. Association for Computational Linguistics. URL https://aclanthology.org/D13-1170/.   
Gido M van de Ven, Nicholas Soures, and Dhireesha Kudithipudi. Continual learning and catastrophic forgetting. arXiv preprint arXiv:2403.05175, 2024.   
Tim Van Erven and Peter Harremos. Rényi divergence and kullback-leibler divergence. IEEE Transactions on Information Theory, 60(7):3797–3820, 2014.   
Ruize Wang, Duyu Tang, Nan Duan, Zhongyu Wei, Xuanjing Huang, Guihong Cao, Daxin Jiang, Ming Zhou, et al. K-adapter: Infusing knowledge into pre-trained models with adapters. arXiv preprint arXiv:2002.01808, 2020.

Frank F Xu, Uri Alon, and Graham Neubig. Why do nearest neighbor language models work? In International Conference on Machine Learning, pages 38325–38341. PMLR, 2023.   
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2. 5 technical report. arXiv preprint arXiv:2412.15115, 2024.   
Dani Yogatama, Cyprien de Masson d’Autume, and Lingpeng Kong. Adaptive semiparametric language models. Transactions of the Association for Computational Linguistics, 9:362–373, 2021.   
Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28, 2015a.   
Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for text classification. Advances in neural information processing systems, 28, 2015b.   
Zhilu Zhang and Mert Sabuncu. Generalized cross entropy loss for training deep neural networks with noisy labels. Advances in neural information processing systems, 31, 2018.

# A. Interpolation hyperparameter $\alpha$ of all tasks

# A.1. Language Modeling on Wikitext-103

For language modeling on WikiText-103 (section 5.1), we use the following $\alpha$ values for different GPT-2 model sizes:

Table 8 | Interpolation hyperparameter $\alpha$ for GPT-2 models on WikiText-103.   

<table><tr><td>Model</td><td>α</td></tr><tr><td>GPT-2-small</td><td>0.80</td></tr><tr><td>GPT-2-medium</td><td>0.60</td></tr><tr><td>GPT-2-large</td><td>0.55</td></tr><tr><td>GPT-2-xl</td><td>0.55</td></tr></table>

The trend of smaller $\alpha$ for larger GPT models aligns with intuition—stronger base models require less augmentation from the memory component. The general pattern centers around $\alpha { = } 0 . 6$ , confirming it as a robust default choice.

# A.2. Downstream Performance

Table 9 presents the optimal $\alpha$ values for downstream tasks in section 5.2.

Table 9 | Optimal interpolation hyperparameter $\alpha$ for downstream tasks.   

<table><tr><td>Task</td><td>α</td></tr><tr><td>SST-2</td><td>0.30</td></tr><tr><td>MR</td><td>0.30</td></tr><tr><td>CR</td><td>0.05</td></tr><tr><td>RT</td><td>0.20</td></tr><tr><td>HYP</td><td>0.20</td></tr><tr><td>CB</td><td>0.30</td></tr><tr><td>RTE</td><td>0.60</td></tr><tr><td>AGN</td><td>0.20</td></tr><tr><td>Yahoo</td><td>0.20</td></tr></table>

The general pattern centers around $\alpha { = } 0 . 3$ , which is consistent with the findings in Shi et al. [2022].

# A.3. Cross-Model and Cross-Vocabulary Adaptation

For domain-specific language modeling tasks (section 5.3 and 5.4), we tune $\alpha$ on the validation set by searching over {0.4, 0.6, 0.8, 0.9}.

# B. Analysis of DAPT Performance on Downstream Tasks

Previous work has shown that domain-adaptive pretraining can adversely affect a model’s prompting ability [Cheng et al., 2023]. Our experiments reveal that this effect is particularly pronounced when using domain-conditional PMI (DCPMI) scoring for evaluation, especially on tasks where label verbalizers overlap with the pretraining domain vocabulary.

Table 10 | Comparison of standard language modeling (LM) scores versus domain-conditional PMI (DCPMI) scores for DAPT models on Yahoo and HYP tasks.   

<table><tr><td>Model</td><td>Yahoo (LM)</td><td>Yahoo (DCPMI)</td><td>HYP (LM)</td><td>HYP (DCPMI)</td><td>Avg</td></tr><tr><td>GPT-2-small</td><td>0.466</td><td>0.495</td><td>0.639</td><td>0.638</td><td>0.559</td></tr><tr><td>+DAPT</td><td>0.429</td><td>0.244</td><td>0.608</td><td>0.361</td><td>0.410</td></tr><tr><td>Δ</td><td>-0.037</td><td>-0.251</td><td>-0.031</td><td>-0.277</td><td>-0.149</td></tr><tr><td>GPT-2-xl</td><td>0.520</td><td>0.499</td><td>0.628</td><td>0.609</td><td>0.564</td></tr><tr><td>+DAPT</td><td>0.490</td><td>0.491</td><td>0.624</td><td>0.618</td><td>0.556</td></tr><tr><td>Δ</td><td>-0.030</td><td>-0.008</td><td>-0.004</td><td>+0.009</td><td>-0.008</td></tr></table>

As shown in Table 10, while direct language modeling evaluation reveals only modest performance drops with DAPT, the DCPMI scores show dramatic degradation for smaller models. This discrepancy arises because we employ fuzzy verbalizers following Shi et al. [2022], and the label spaces for Yahoo and AGN tasks contain terms (e.g., “politics,” “technology”) that appear frequently in WikiText-103. When DAPT increases the domain probability for these terms, it causes the conditional PMI scores to drop substantially, as the denominator in the DCPMI calculation becomes inflated.

The results for GPT-2-xl demonstrate that larger models exhibit greater robustness to this evaluation artifact, maintaining relatively stable DCPMI scores after domain adaptation. This suggests that the apparent failure of DAPT on certain downstream tasks is partly an artifact of the evaluation methodology rather than a fundamental limitation of the approach, though the phenomenon highlights an important interaction between domain adaptation and prompt-based evaluation methods.

# C. Characteristics of $k$ -NN Distributions

# C.1. Extreme Sparsity and Concentration

$k$ -NN distributions exhibit fundamentally different characteristics from standard language model outputs. While LM distributions maintain smooth probability mass across vocabulary with extensive long tails, $k$ -NN distributions demonstrate extreme sparsity—typically assigning non-zero probabilities to only 2–3 tokens from a 50,257-dimensional vocabulary.

This concentration emerges from two factors: (1) the hard constraint of selecting only $k$ nearest neighbors eliminates low-probability candidates, and (2) high-dimensional embedding spaces (e.g., 1280 dimensions for GPT-2-Large) amplify distance relationships through the curse of dimensionality, causing nearest neighbors to dominate disproportionately.

# C.2. Scale-Dependent Behavior

Model scale dramatically affects $k$ -NN distribution quality. GPT-2-small (124M) produces distributions marginally different from its LM outputs (top-1 probability $5 0 \%$ ), while GPT-2-Large (1.5B) generates radically sparse distributions with $9 3 . 4 8 \%$ average top-1 probability—a $6 7 \%$ relative increase over its baseline.

Larger models benefit from: (1) higher-dimensional spaces where distance concentration intensifies, and (2) superior contextual representations that better disambiguate polysemous tokens and preserve semantic distinctions, leading to more coherent nearest neighbor retrievals.

![](images/d91c3e1ed723896126b08c304d58a80c126824d8b2468aa0c9bbbba22b570afd.jpg)  
Figure 5 | Probability distributions from $k$ -NN retrieval, standard LM, and Memory Decoder for GPT-2-Large. The $k$ -NN distribution shows extreme sparsity with concentrated probability mass.

Table 11 | Perplexity with Memory Decoder (124M) trained using $k$ -NN distributions from different source models   

<table><tr><td rowspan="2">Base Model</td><td rowspan="2">Baseline PPL</td><td colspan="3">PPL with MemDec from:</td></tr><tr><td>Small</td><td>Medium</td><td>Large</td></tr><tr><td>GPT-2-Small</td><td>24.89</td><td>14.01</td><td>13.80</td><td>13.77</td></tr><tr><td>GPT-2-Medium</td><td>18.29</td><td>12.88</td><td>12.74</td><td>12.70</td></tr><tr><td>GPT-2-Large</td><td>15.80</td><td>12.05</td><td>11.95</td><td>11.93</td></tr></table>

# C.3. Domain Adaptation Effects

Fine-tuned models produce sharper $k$ -NN distributions than base models. Domain adaptation creates specialized embedding clusters with reduced intra-cluster variance and increased inter-cluster separation, leading to more decisive retrievals. Memory Decoders trained with fine-tuned distributions consistently achieve lower perplexity, validating that domain-adapted representations provide superior retrieval targets.

# D. Alternative Loss Functions for Imitating $k$ -NN Distributions

# D.1. Failed Approaches

While KL divergence (combined with cross-entropy regularization) successfully matches $k$ -NN distributions, we systematically evaluated several alternative loss functions that all demonstrated inferior performance:

![](images/be4d61170d9f6b98aee021266d25998301cfc31a25cafd54dcf1fab7accdb219.jpg)  
Figure 6 | ??-NN distribution sparsity across model scales. Despite identical retrieval parameters $( k = 1 0 2 4 )$ , larger models produce substantially sparser distributions.

# D.1.1. Focal Loss

Adapted from object detection to handle class imbalance through gradient rescaling:

$$
\mathcal {L} _ {\mathrm {F o c a l}} = - \sum_ {i} [ \alpha (1 - p _ {\theta} (i)) ^ {\gamma} p _ {\mathrm {k N N}} (i) \log p _ {\theta} (i) + (1 - \alpha) p _ {\theta} (i) ^ {\gamma} (1 - p _ {\mathrm {k N N}} (i)) \log (1 - p _ {\theta} (i)) ] \tag {8}
$$

With $\alpha = 0 . 5$ and $\gamma = 2$ , focal loss theoretically emphasizes hard-to-classify sparse regions but failed to achieve sufficient distribution concentration in practice.

# D.1.2. Jensen-Shannon Divergence

A symmetric alternative to KL divergence:

$$
\operatorname {J S D} (P \parallel Q) = \frac {1}{2} D _ {\mathrm {K L}} (P \parallel M) + \frac {1}{2} D _ {\mathrm {K L}} (Q \parallel M), \quad M = \frac {1}{2} (P + Q) \tag {9}
$$

Despite avoiding the directional bias of KL divergence, JSD provided no advantage for our extremely sparse target distributions.

# D.1.3. Bi-directional Logits Difference (BiLD)

BiLD focuses on relative rankings by computing pairwise differences among top- $k$ logits:

$$
\mathcal {L} _ {\mathrm {B i L D}} = D _ {\mathrm {K L}} \left[ p _ {\text {l e d}} ^ {\mathrm {k N N}} \| p _ {\text {c o r}} ^ {\theta} \right] + D _ {\mathrm {K L}} \left[ p _ {\text {c o r}} ^ {\mathrm {k N N}} \| p _ {\text {l e d}} ^ {\theta} \right] \tag {10}
$$

While theoretically suited for distributions where relative ordering matters more than exact probabilities, BiLD consistently underperformed standard KL divergence.

# D.1.4. Explicit Sparsity Penalty

Direct penalization of non-zero predictions in zero-probability regions:

$$
\mathcal {L} _ {\text {s p a r s e}} = \mathcal {L} _ {\mathrm {K L}} + \alpha \sum_ {i} \mathbb {I} _ {\{p _ {\mathrm {k N N}} (i) = 0 \}} \cdot p _ {\theta} (i), \quad \alpha = 0. 0 1 \tag {11}
$$

This approach created training instability without meaningfully improving output sparsity.

# D.2. Why KL Divergence Succeeds

The superior performance of KL divergence (with cross-entropy regularization) for matching $k$ -NN distributions likely stems from its unique mathematical properties that align with the retrieval-based nature of the target:

Asymmetric penalty structure: KL divergence probability mass where the target has none (wh $\begin{array} { r } { D _ { \mathrm { K L } } ( P | | Q ) = \sum _ { i } P ( i ) \log \frac { P ( i ) } { Q ( i ) } } \end{array}$ heavily penalizes placinghile being more forgiving $P ( i ) \approx 0$ $Q ( i ) > 0 )$ of missing mass where the target has some. This asymmetry naturally encourages sparsity—the model learns to aggressively zero out predictions outside the $k$ -NN support.

Mode-seeking behavior: The forward KL divergence $D _ { \mathrm { K L } } ( P | | Q )$ is inherently mode-seeking, preferring to capture a few high-probability modes rather than covering the entire distribution. For $k$ -NN distributions with 2-3 dominant modes, this bias perfectly matches the desired behavior, unlike symmetric losses (JSD) or mode-covering alternatives.

Information-theoretic optimality: KL divergence directly minimizes the expected encoding length difference between distributions. For $k$ -NN distributions that encode "retrieval-aware uncertainty," KL naturally preserves the information structure—maintaining both the sharp peaks (high retrieval confidence) and the specific ranking among top candidates that emerges from the datastore’s empirical distribution.

The cross-entropy regularization component anchors the model to linguistically valid outputs, preventing collapse to degenerate solutions while the KL term drives sparsity. This combination uniquely balances the competing demands of extreme concentration and semantic coherence.
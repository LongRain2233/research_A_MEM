# Memory Retrieval and Consolidation in Large Language Models through Function Tokens

Shaohua Zhang, Yuan Lin, Hang Li

ByteDance Seed

# Abstract

The remarkable success of large language models (LLMs) stems from their ability to consolidate vast amounts of knowledge into the memory during pre-training and to retrieve it from the memory during inference, enabling advanced capabilities such as knowledge memorization, instructionfollowing and reasoning. However, the mechanisms of memory retrieval and consolidation in LLMs remain poorly understood. In this paper, we propose the function token hypothesis to explain the workings of LLMs: During inference, function tokens activate the most predictive features from context and govern next token prediction (memory retrieval). During pre-training, predicting the next tokens (usually content tokens) that follow function tokens increases the number of learned features of LLMs and updates the model parameters (memory consolidation). Function tokens here roughly correspond to function words in linguistics, including punctuation marks, articles, prepositions, and conjunctions, in contrast to content tokens. We provide extensive experimental evidence supporting this hypothesis. Using bipartite graph analysis, we show that a small number of function tokens activate the majority of features. Case studies further reveal how function tokens activate the most predictive features from context to direct next token prediction. We also find that during pre-training, the training loss is dominated by predicting the next content tokens following function tokens, which forces the function tokens to select the most predictive features from context.

Date: October 10, 2025

Correspondence: {zhangshaohua.cola,linyuan.0,lihang.lh}@bytedance.com

# 1 Introduction

Large Language Models (LLMs) [2, 4, 25, 30, 31, 46] have demonstrated remarkable capabilities. They possess strong knowledge memorization abilities, ranging from remembering simple factual knowledge (e.g., The capital of the United States is Washington, $D . C .$ .) to the verbatim reproduction of lengthy passages (e.g., Recite Martin Luther King Jr’s “I Have a Dream" speech word by word). Beyond that, LLMs also exhibit strong general skills, such as instruction following [32, 54] (e.g., As a financial analyst: explain quantitative tightening, then list three stock market impacts.) and reasoning [22, 55] (e.g., The streets are wet and the sidewalks are slick. What is the most likely explanation? ).

In the human brain, long-term memory forms through synaptic consolidation, where the synapses between neurons are strengthened, ultimately creating neural circuits that store knowledge [19]. Inspired by this biological mechanism, artificial neural networks have been developed. These systems consist of neurons linked by weighted connections, and their weights (parameters) are obtained by training on data. The weights of a

![](images/e51660809dbc08c4802c6e2cb2a746de3db25eab74ee0ae8e9a3a652abc60d77.jpg)  
Figure 1 Function tokens can dynamically activate the most predictive features from the context to guide the next-token prediction. For example, the function token ‘in’ reactivates features ‘J.K. Rowling’ and ‘Location’ from context (while suppressing feature ’French’) and activates ‘England’ to predict ‘Britain’. In contrast, the content token ‘Harry’ activates feature ‘Harry Potter’.

neuron determines how it responds to its inputs to produce an activation [15]. A technique utilizing Sparse Autoencoders (SAEs) [9] has been developed recently to analyze Transformer-based LLMs [50]. It enables the decomposition of neuron activations into interpretable features, providing insights into how the circuits within the Transformer’s layers are composed of these interpretable features [7, 12, 17].

Despite significant progress in understanding LLM neuron activations, the memory mechanisms remain poorly understood. In particular, two fundamental questions are still not well addressed: (1) How is the memory retrieved during inference? and (2) How is the memory consolidated during pre-training? In this paper, we present our investigation into these questions. We find that analyzing from the perspective of function tokens and content tokens can help unravel the mystery of memory retrieval and memory consolidation.

In linguistics, function words are words that have little semantic meanings but play crucial grammatical and connective roles within and between sentences, such as articles, prepositions, and conjunctions [5]. In contrast, content words are words that convey semantically explicit and rich meanings. The distribution of words in natural language follows Zipf’s law [20]. In this distribution, function words occur with disproportionately high frequencies, occupying the head, while content words appear with much lower frequencies, forming the long tail. LLMs utilize tokens, which may represent words, sub-words, or punctuation marks. In our work, for ease of experimentation, we automatically classify tokens into ‘function tokens’ and ‘content tokens’ based on their frequencies in the pre-training corpus, using this as an approximation of the linguistic concepts.

To investigate the role of function tokens during inference, we construct a bipartite graph connecting tokens to features obtained via SAE decomposition. We show that, although few in number, function tokens activate a large proportion of the LLM’s features. Furthermore, our case studies show that the activation patterns for function tokens differ from those for content tokens. Function tokens dynamically reactivate predictive features from the context, whereas content tokens show little evidence of this effect. To understand why feature activations are centered on function tokens, we conduct pre-training experiments. We track next-token prediction loss across four categories based on whether the current token and the next token are function or content tokens. We find that LLMs first learn to predict function tokens before gradually learning to predict content tokens, a process accompanied by an increase in the number of features and the learning of the parameters. Furthermore, pre-training is dominated by the prediction of content tokens that follow function tokens. These observations reveal why function tokens can access a large portion of the LLM’s features. Based on these findings, we propose the Function Token Hypothesis (see an example in Figure 1).

In this paper, the LLMs are GPT-type models with a Transformer decoder architecture, obtained through pre-training and post-training (including SFT and RL) [28, 37]. Both pre-training and inference are conducted autoregressively via next-token prediction. At each layer of the Transformer, a vector of activations (after the add-norm operation of FFN) can be created, with each dimension representing a neuron. SAE can be performed on this activation vector to obtain a linear combination of features for each neuron. Here, knowledge refers to the LLM’s parameters as well as all possible features that can be derived from them. Memory is the virtual system that stores the knowledge. Memory retrieval means the activations of features and circuits [11, 27, 29, 51], while memory consolidation means the learning of the parameters to form and expand features and circuits.

Function Token Hypothesis. During inference, function tokens activate the most predictive features from the context to direct the next-token prediction (memory retrieval). During pre-training, predicting content tokens based on the function tokens drives the LLM to update its parameters to learn and expand features (memory consolidation).

The function token hypothesis is also supported by many phenomena observed in LLM research. For example, activations with unusually large magnitudes often occur at the initial tokens, periods, or newlines [45]. Meaningless separator tokens disproportionately affect attention compared to semantically rich tokens [6]. The use of ‘pivot tokens’ during post-training can significantly enhance performance in response [1]. Training that concentrates on high-entropy tokens also yields better performance [52]. We argue that these tokens are all function tokens that behave as the hypothesis predicts.

We believe that unraveling the important role of function tokens in LLM memory mechanisms not only enhances research on LLM interpretability but also provides insights for designing advanced learning algorithms, particularly those for enhancing alignment with human values.

The main contributions of this paper are summarized as follows:

• We demonstrate that during inference, function tokens are responsible for activating the most predictive features from the context to govern next-token prediction.   
• We show that feature growth during pre-training is driven by the prediction of content tokens that follow function tokens.   
• We propose the Function Token Hypothesis for explaining LLM memory mechanisms.

# 2 Preliminary

# 2.1 Model Memory and Superposition Phenomenon

Feed-Forward Network as Key-Value Memory Existing work views the Feed-Forward Network (FFN) layer in each block of a Transformer as a key-value memory or a neural memory [15]. Specifically, the FFN can be formulated as (bias terms are omitted, as in common practice):

$$
\mathbf {z} = \operatorname {R e L U} \left(\mathbf {x} \cdot \mathbf {W} _ {k} ^ {\top}\right) \tag {1}
$$

$$
\mathbf {y} = \mathbf {z} \cdot \mathbf {W} _ {v} \tag {2}
$$

Here, $\mathbf { x } \in \mathbb { R } ^ { d }$ is the input vector, $\pmb { \mathsf { y } } \in \mathbb { R } ^ { d }$ is the output vector, $\pmb { z } \in \mathbb { R } ^ { d _ { m } }$ is the weight vector, $\mathbf { W } _ { k } \in \mathbb { R } ^ { d _ { m } \times d }$ is the key matrix, $\mathbf { W } _ { v } \in \mathbb { R } ^ { d _ { m } \times d }$ is the value matrix, and $d _ { m }$ denotes the memory size. The output vector $\pmb { \ y }$ is in fact the activation of the FFN layer, where each dimension corresponds to a neuron.

In the key-value memory interpretation, there are $d _ { m }$ pairs of key vector and value vector. Each row of $\mathbf { W } _ { k } \in \mathbb { R } ^ { d _ { m } \times d }$ corresponds to a key vector $\mathbf { k } _ { i } \in \mathbb { R } ^ { d }$ and each row of $\mathbf { W } _ { v } \in \mathbb { R } ^ { d _ { m } \times d }$ corresponds to a value vector $\pmb { v } _ { i } \in \mathbb { R } ^ { d }$ . Given the input vector $\pmb { \times }$ , the similarity between $\mathbf { x }$ and each of the key vectors $\mathsf { \pmb { k } } _ { i }$ is first calculated as $z _ { i } = \mathrm { R e L U } ( \mathbf { x } \cdot \mathbf { k } _ { i } ^ { \top } ) \geq 0$ , where ReLU acts as an unnormalized weighting function; the weighted sum of the corresponding value vectors based on the similarities is then calculated and output as $\begin{array} { r } { \pmb { \mathsf { y } } = \sum _ { i = 1 } ^ { d _ { m } } z _ { i } \mathbf { v } _ { i } } \end{array}$ . The interpretation suggests that knowledge of the Transformer is represented in the parameters of the FFN layers. Note that Transformer attention layers also form key-value memories using softmax weighting.

Superposition Phenomenon Recent work on LLM interpretability shows the phenomenon of superposition [12], in which features can be extracted from the activations of neurons in a Transformer-based LLM. The number of extracted features usually far exceeds the number of neurons. There exist many polysemantic neurons, each of which represents multiple meanings.

Through sparse dictionary learning, the activations of polysemantic neurons can be decomposed into monosemantic features, each corresponding to a distinct, human-interpretable concept, such as the Golden Gate Bridge [49]. A widely used method for dictionary learning is the Sparse Autoencoder (SAE), which learns to

linearly decompose neuron activations through a reconstruction task. SAE decomposes an activation $\mathbf { y } \in \mathbb { R } ^ { d }$ , typically the output of a specific layer, into a linear combination of features:

$$
\mathbf {y} = \sum_ {i = 1} ^ {n} c _ {i} \mathbf {f} _ {i} = c _ {1} \mathbf {f} _ {1} + c _ {2} \mathbf {f} _ {2} + \dots + c _ {n} \mathbf {f} _ {n}. \tag {3}
$$

Here, $\mathbf { x } \in \mathbb { R } ^ { d }$ is the input vector, $\pmb { \mathsf { y } } \in \mathbb { R } ^ { d }$ is the output vector, $\pmb { z } \in \mathbb { R } ^ { d _ { m } }$ is the weight vector, $\mathbf { W } _ { k } \in \mathbb { R } ^ { d _ { m } \times d }$ is the key matrix, $\mathbf { W } _ { v } \in \mathbb { R } ^ { d _ { m } \times d }$ is the value matrix, and $d _ { m }$ is the memory size. The output vector $\pmb { \ y }$ is the activation of the FFN layer, where each dimension corresponds to a neuron. Furthermore, the behavior of the LLM during generation can be partially controlled by steering the activations of features. For example, steering can control both specific concepts (e.g., Golden Gate Bridge [49]) and behavioral patterns (e.g., sycophantic behavior [33]). Similar feature activation phenomena are observed in human memory recall, with empirical evidence supporting the existence of neurons representing either specific or general concepts.

# 2.2 Function Tokens and Content Tokens

![](images/2e0f10565162bdf309c35d62411641c08ff3636b3c39aef81a126480ee06ee4c.jpg)  
Figure 2 Token frequency statistics in SlimPajama-627B.

(a) Zip’f distribution of tokens on a log-log scale.

(b) The 15 most frequent tokens.   

<table><tr><td>Token rank</td><td>Token text</td><td>Token Fraction</td><td>Cumulative Fraction</td><td>Document Coverage</td></tr><tr><td>1</td><td>,</td><td>3.60%</td><td>3.60%</td><td>95.00%</td></tr><tr><td>2</td><td>the</td><td>3.19%</td><td>6.79%</td><td>90.92%</td></tr><tr><td>3</td><td>.</td><td>2.23%</td><td>9.10%</td><td>95.80%</td></tr><tr><td>4</td><td>and</td><td>1.81%</td><td>10.91%</td><td>89.69%</td></tr><tr><td>5</td><td>of</td><td>1.80%</td><td>12.71%</td><td>87.59%</td></tr><tr><td>6</td><td>to</td><td>1.68%</td><td>14.40%</td><td>88.71%</td></tr><tr><td>7</td><td>white space</td><td>1.59%</td><td>15.99%</td><td>81.35%</td></tr><tr><td>8</td><td>a</td><td>1.33%</td><td>17.32%</td><td>87.62%</td></tr><tr><td>9</td><td>in</td><td>1.16%</td><td>18.48%</td><td>86.04%</td></tr><tr><td>10</td><td>\n</td><td>0.90%</td><td>19.39%</td><td>84.58%</td></tr><tr><td>11</td><td>is</td><td>0.73%</td><td>20.13%</td><td>78.90%</td></tr><tr><td>12</td><td>\n</td><td>0.70%</td><td>20.84%</td><td>42.30%</td></tr><tr><td>13</td><td>for</td><td>0.64%</td><td>21.48%</td><td>79.82%</td></tr><tr><td>14</td><td>that</td><td>0.62%</td><td>22.09%</td><td>67.02%</td></tr><tr><td>15</td><td>&#x27;s</td><td>0.49%</td><td>22.58%</td><td>63.02%</td></tr></table>

We tokenized the SlimPajama-627B corpus [43], a widely used pre-training dataset, using the LLaMA-3.1 tokenizer and sampled 1 billion tokens for statistical analysis. We group the tokens into function tokens and content tokens based on their frequency. This leverages the linguistic fact that function words typically have higher frequency, while content words have lower frequency. Starting from the most frequent, we add tokens until the set covered $4 0 \%$ of all token occurrences, yielding 122 tokens labeled as function tokens; the rest are taken as content tokens. The resulting set of function tokens roughly corresponds to the function words defined in linguistics, with several exceptions like punctuation marks. The full list of function tokens appears in Appendix D.

Token Frequency and Zipf’s Law As shown in Figure 2a, token frequency follows the Zipf’s law [36]: $f ( r ) \propto r ^ { - \alpha }$ where $f ( r )$ is the frequency of the token ranked $r$ , revealing a fundamental property of natural language: a few tokens are used frequently, while most are used infrequently. For example, the 15 most frequent tokens account for $2 2 . 5 8 \%$ of the corpus (Figure 2b).

Document Coverage A pre-training corpus contains a vast number of documents. High-frequency tokens are distributed uniformly across documents, while low-frequency tokens appear frequently within a limited number of documents [42], showing bursty distributions. For instance, as shown in Figure 3, the function token ‘of’ appears with similar frequency across documents, whereas the content token ‘Tokyo’ occurs only in a few. Thus, high-frequency tokens are utilized in nearly all training examples, while low-frequency tokens are used only in a small fraction of them.

![](images/00640419ddb716f9c05ae1cf2bd875a7abf2a145dacbb19b88d6dd036cbc8d9a.jpg)  
(a) Distribution of the function token ‘of’ across documents, showing uniform and dense coverage.

![](images/08621c78f95a65adebaaecd2bc6828fde3f012a478f3263a4a88618c15d76148.jpg)  
(b) Distribution of the content token ‘Tokyo’ across documents, showing sparse coverage.

![](images/bbd3e8bb5c0adf4f2657706bc88c564e5666fa58f1a5b9d8d10eef819b4d44d9.jpg)  
(c) Document coverage versus token rank (ordered by frequency, log-log scale).

![](images/811f5935eae5f02b50f0ef3e05fd190136d7389c4225dc3cbaea4170532c8a68.jpg)  
Figure 3 Distribution of function and content tokens. Document bins represent equal partitions of corpus documents.   
Figure 4 Construction of the bipartite graph using token-feature activation pairs as edges. Nodes consist of tokens from the vocabulary and features from the SAE decomposition.

Figure 3c shows a strong correlation between token frequency and document coverage: high-frequency tokens typically appear across most documents. Figure 2b presents the 15 most frequent tokens, along with their corresponding document coverage values. Here the document coverage of a token $t$ is defined as $\frac { | \{ d \in D : t \in d \} | } { | D | }$ , where $D$ denotes the entire set of documents.

# 3 Memory Retrieval through Function Tokens

We study the relationships between tokens and model features during inference. The results show that a small set of function tokens can activate most features. Our case study reveals how the same function tokens create different activation patterns in different contexts, leading to different outputs.

# 3.1 A Few Function Tokens Activate Most Features

We use Gemma2-9B [48] for our analysis, as it provides both models of different sizes and open-source SAEs [24]. Gemma Scope has SAEs with varying dictionary widths. Among these, we select the SAE with the largest dictionary width, $2 ^ { 2 0 }$ , to facilitate a more comprehensive feature decomposition.

To study how features are activated during inference, as illustrated in Figure 4, we construct a token-feature bipartite graph through the following steps:

• Step 1:Extract activations. We feed 10,000 randomly sampled raw documents from the SlimPajama validation dataset into Gemma2-9B, with approximately 5 million tokens, and extract activations from the residual stream. We focus on three representative layers: layer 9 (shallow), layer 20 (middle), and layer 31 (deep).   
• Step 2: Feature decomposition.For each layer, we apply the corresponding SAE to decompose token activations into sparse features.   
• Step 3: Bipartite graph construction. A token is linked to a feature if it activates the feature in a context. Each token-feature pair is connected by at most one edge, regardless of how many times the activation occurs.

The bipartite graph comprises two node types: tokens and features. The number of token nodes equals the vocabulary size, while the number of feature nodes (each connected to at least one token) is 965,635, 947,341, and 919,220 for the three layers, respectively. With a dictionary width of $2 ^ { 2 0 }$ , this yields activation rates of $9 2 . 1 \%$ , 90.3% and 87.7%, confirming sufficient coverage for analysis.

![](images/d9657bf6b6e079dd32fd586ea846bc5794ee078a9e8bb50c559153a439fa049b.jpg)  
(a) Layer 9

![](images/82ac3bd572d56ca4f19bc310ec89f74c42c1d1e724a35a83e5783d090cd3234d.jpg)  
(b) Layer 20

![](images/211a043527c80afc99a5e5cc3b8d29388903f215a5de89b57836df0f100d30e0.jpg)  
(c) Layer 31   
Figure 5 Token degrees in the token-feature bipartite graph on a log-log scale. Tokens are ranked by frequency from the sampled data.

Table 1 Cumulative feature coverage by top-10 frequent tokens across different layers   

<table><tr><td colspan="2">Token</td><td colspan="3">Cumulative Feature Coverage</td></tr><tr><td>Rank</td><td>Text</td><td>Layer 9</td><td>Layer 20</td><td>Layer 31</td></tr><tr><td>1</td><td>.</td><td>23.19%</td><td>51.32%</td><td>37.21%</td></tr><tr><td>2</td><td>,</td><td>32.01%</td><td>62.45%</td><td>49.78%</td></tr><tr><td>3</td><td>the</td><td>36.88%</td><td>66.93%</td><td>55.15%</td></tr><tr><td>4</td><td>\n</td><td>39.68%</td><td>71.30%</td><td>59.86%</td></tr><tr><td>5</td><td>and</td><td>41.21%</td><td>71.97 %</td><td>61.48%</td></tr><tr><td>6</td><td>to</td><td>43.16%</td><td>73.07 %</td><td>63.30%</td></tr><tr><td>7</td><td>of</td><td>46.00%</td><td>74.43 %</td><td>65.16%</td></tr><tr><td>8</td><td>white space</td><td>47.44%</td><td>75.70 %</td><td>67.08%</td></tr><tr><td>9</td><td>a</td><td>47.96%</td><td>76.12%</td><td>67.74%</td></tr><tr><td>10</td><td>in</td><td>48.52%</td><td>76.46%</td><td>68.27%</td></tr></table>

Figure 5 presents the degree of each token in the token-feature bipartite graph. The results reveal that a small set of function tokens can activate most features. Table 1 shows that the top 10 frequent tokens alone account for a substantial proportion of feature activations. In particular, in the middle layer, known to be the most expressive and interpretable [7, 34, 44], these tokens can activate more than 70% of the features, demonstrating function tokens’s universal access to the feature space.

# 3.2 Feature Reactivation via Function Tokens

Why can a small number of function tokens activate most features? We hypothesize that function tokens can reactivate the most predictive features, based on preceding contexts.

We design an experiment to examine this hypothesis. First, we identify three interpretable features in Gemma2-9B-it [47]: Feature 15261 corresponds to ‘Speak Chinese’, Feature 9591 corresponds to ‘Russia’, and Feature 13751 corresponds to ‘UK’. The approach for identifying interpretable features is described in Appendix A. We then examine their activations during inference. We employ the following prompt template, wrapping the chat template used in Gemma2-9B-it.

# Prompt Template

```txt
<bos><start_of_turn>user  
{prompt}Directly answer the question<end_of_turn>  
<start_of_turn>model 
```

We evaluate the following two prompts and record each token’s feature activations, as shown in Figure 6.

• Prompt 1: Answer the question in Chinese: What is the capital of Russia?   
• Prompt 2: Answer the question in Chinese: What is the capital of UK?

As shown in Figure 6, for Prompt 1, the ‘Speak Chinese’ feature is first activated by the token ‘Chinese’, and the ‘Russia’ feature by the token ‘Russia’. Function tokens such as ‘:’, ‘the’ and ‘\n’ serve as conduits for propagating and re-creating these activations. A similar pattern is observed for Prompt 2. Notably, the only difference between the prompts is the replacement of ‘Russia’ with ‘UK’, yet the same function tokens orchestrate different feature combinations, resulting in distinct model outputs.

![](images/85fa9376810ee7caa91295bc637ab5e1cb15cf1f8dfe3a0c98d7f09f98897e11.jpg)  
Figure 6 Function tokens can dynamically reactivate predictive features based on different contexts.

Furthermore, we show that steering activations on function tokens can directly influence model outputs. The steering method is described in Appendix §A. We evaluate this effect using the following prompts:

• Prompt 3: Where is Mount Fuji?   
• Prompt 4: Tell me a university.   
• Prompt 5: Could you recommend a tourist attraction?

![](images/c4915592e1ba22ef368623d001ca287a1ec9f2418608e2efe3b69e0a58d4c652.jpg)  
Figure 7 Response of Gemma2-9B-it when editing the activation at the final function token ( $\mathbf { \ddot { \rho } } ( \mathbf { n } ^ { \prime } )$ in the prompt. The Chinese terms shown in the table and their corresponding English translations are: 日本 (Japan), 哈佛 学 (Harvard 大University), 故宫 (The Forbidden City), 英国 (UK), 津 学 (Oxford University), 伦敦眼 (London Eye), 俄罗斯 牛 大(Russia), 莫斯科国 学 (Moscow State University), and 叶卡捷琳娜宫 (Catherine Palace).

As shown in Figure 7, features activated by the function token are predictive, driving the subsequent token generation. For Prompt 3, the model normally answers in English (‘Japan’). Steering only the activations on the final function token in the prompt (‘\n’) changes the response: activating the ‘Speak Chinese’ feature switches the answer to ‘日本’ (Japan in Chinese), activating the ‘Russia’ feature changes the answer to ‘Russia’, and jointly activating ‘Speak Chinese’ and ‘UK’ features yields ‘英国’ (UK in Chinese). Prompts 4 and 5 exhibit the same behavior, demonstrating that function tokens activate predictive features. For more case studies, see Table 11 (§A).

In addition, steering features enable generalized control rather than merely triggering specific word outputs. For example, activating the ‘Russia’ feature can produce contextually appropriate responses, such as ‘Moscow State University’ and ‘Alexandrinsky Theatre’, instead of simply outputting the token ‘Russia’. This demonstrates that the features encode high-level semantic concepts.

# 4 Memory Consolidation through Function Tokens

We analyze how memory consolidation occurs during pre-training. We train two models and track their losses on function and content tokens, as well as their feature growth patterns across different training stages. Our key findings are:

• As the number of training steps increases, the number of the learned features increases.   
• Pre-training initially focuses on learning to predict function tokens.   
• Subsequently, the optimization process becomes dominated by learning to predict content tokens, especially predicting content tokens that follow function tokens.

# 4.1 Pre-Training Setup

We train two models from scratch using the LLaMA-3.1-8B [16] architecture: an 8B model with the originial 32 layers and a 1.5B models with only 2 layers, keeping other components unchanged. We use SlimPajama-627B [43] as our pre-training corpus, which is a diverse, high-quality collection of web data that has been carefully deduplicated and filtered. This dataset is well-suited for studying memory consolidation during pretraining. We train for one complete epoch over its 627 billion tokens. We replicate the training hyperparameters of LLaMA-3.1-8B for reproducibility: batch size 1024, max sequence length 4095, AdamW optimizer. The learning rate warm up linearly for 8,000 steps to $8 \times 1 0 ^ { - 5 }$ , then decays by cosine annealing to $8 \times 1 0 ^ { - 7 }$ . Training runs on 128 GPUs with 80GB memory each.

# 4.2 Memory Consolidation as Feature Expansion

Due to computational constraints, we perform feature decomposition only on the 1.5B model. To track the number of emergent features during pre-training, we train SAEs on second-layer activations at multiple checkpoints. We use JumpReLU-SAE [39] with a tanh penalty function [3], wich outperforms alternatives such as TopK-SAE [14] and Gated-SAE [38]. Training details are in Appendix C.

We select three representative checkpoints of the pre-training for SAE training: 3000 steps, 50,000 steps and 130,000 steps, corresponding to early, intermediate and late stages of pre-training. For each checkpoint, we sample text sequences from SlimPajama to obtain 500,000 activations, which are input to the SAE to count the total number of decomposed unique features. As shown in Figure 8a, the number of features grows substantially over the progress of pre-training, reflecting the model’s increasing representational capability and corresponding to memory consolidation.

![](images/e50c9e81b8e5e7e6ff606c0baf127e5eb10fc09343590ec5d95cb71e99a67277.jpg)  
(a) Number of learned features

![](images/8c4f10028c3d3554e6b3ede58a485101c0616c71bb78dcb1fdd7c48a8ea62b3b.jpg)  
(b) Token degree by checkpoints   
Figure 8 Tracking memory consolidation in relation to feature expansion during pre-training.

Using the bipartite graph analysis described in Section 3.2, we study how token-feature activation evolve. As shown in Figure 8b, the number of features grows during training, but function tokens consistently activate most features, in contrast with content tokens. This disparity widens over time, as evidenced by the gradually steepening slopes in the graph.

# 4.3 Loss on Function and Content Tokens

![](images/1169f4c2e1b14789bfe017a974e74e8bacccad6d9672d1a5a06d237b5956ecee.jpg)  
(a) Grouped token loss trajectories during 1.5B model pre-training

![](images/bd20f6e77cc98ced615235104bb43f8ef1ffbdf3c3eb87a6f84cea7d80148fdb.jpg)  
(b) Grouped token loss trajectories during 8B model pre-training

![](images/0914b2be9d66a81b1e35c8b091c406382b314112cdc8e83e105fba77e2c2becf.jpg)  
(c) Next-token loss for typical function tokens in 1.5B model pretrain   
Figure 9 Pre-training loss curves of different token groups.

To track loss changes for function and content tokens, we categorize next-token prediction of the form $p$ (next token|current token, context) into four groups based on whether the current and next tokens are

function tokens or content tokens. This yields four distinct categories. For example, $p$ (next token = function token | current token = function token, context) is denoted as function function. The other three categories are defined similarly: function content, content function, and content content. Figure 9 presents the pre-training loss curves of four groups for both the 1.5B and 8B models, along with the average loss curve across all tokens. We highlight several key observations.

Function Content drives the optimization and memory consolidation. Throughout pre-training, the function content group has the highest loss in both the 1.5B and 8B models, making it the hardest prediction task. As a result, optimization is dominated by this task, which in turn pushes function tokens to develop the capability to reactivate predictive features from context. Furthermore, the feature growth during pre-training likewise primarily driven by function content prediction.

Function token prediction is learned faster and more easily. For both 1.5B and 8B models, loss decreases more quickly and converge lower when predicting function tokens than content tokens. Function tokens reach very low loss early in training, showing that LLMs first learn to predict function tokens. Figure 9 plots loss curves of several representative function tokens (‘the’, ‘of’, and ‘,’) as next tokens to be predicted, alongside the average loss across all tokens, highlighting rapid convergence within the first 3,000 steps. This indicates that the model first learns to generate function tokens before learning to generate more complex token sequences.

Scaling enhances content token prediction. Scaling from 1.5B to 8B parameters yields small loss reductions for content function group (1.90 to 1.64, $\Delta = 0 . 2 6$ ) and function function group (2.12 to 1.87, $\Delta = 0 . 2 5$ ), but much larger loss reductions for function content group (4.88 to 4.27, $\Delta = 0 . 6 1$ ) and content→content group (3.69 to 3.08, $\Delta = 0 . 6 1$ ). These results indicate that scaling model size primarily enhances the content token prediction.

<table><tr><td>Model Size</td><td>Training Steps</td><td>When</td><td>young</td><td>children</td><td>are</td><td>learning</td><td>to</td><td>read</td><td>,</td><td>they</td><td>often</td><td>struggle</td><td>with</td><td>complicated</td><td>words</td></tr><tr><td rowspan="4">1.5b</td><td>100 steps (0.7%)</td><td>sharing</td><td>SUCCEED</td><td>GHz</td><td>liament</td><td>&quot;).</td><td>liament</td><td>プロ</td><td>Liter</td><td>follower</td><td>readOnly</td><td>readOnly</td><td>posites</td><td>↑</td><td>にnergり</td></tr><tr><td>3000 steps (2.3%)</td><td>,</td><td>a</td><td>,</td><td>a</td><td>to</td><td>the</td><td>the</td><td>and</td><td>are</td><td>,</td><td>,</td><td>the</td><td>and</td><td>,</td></tr><tr><td>50000 steps (38%)</td><td>the</td><td>young</td><td>are</td><td>in</td><td>to</td><td>the</td><td>the</td><td>and</td><td>are</td><td>find</td><td>to</td><td>the</td><td>ideas</td><td>and</td></tr><tr><td>130000 steps (94%)</td><td>the</td><td>young</td><td>are</td><td>in</td><td>to</td><td>read</td><td>the</td><td>they</td><td>are</td><td>have</td><td>to</td><td>the</td><td>language</td><td>and</td></tr><tr><td>8b</td><td>130000 steps (94%)</td><td>the</td><td>people</td><td>are</td><td>not</td><td>to</td><td>read</td><td>,</td><td>they</td><td>are</td><td>have</td><td>to</td><td>reading</td><td>words</td><td>and</td></tr></table>

Figure 10 Next token predictions at different training steps. The first row shows the prompt. Each subsequent row shows the next-token predictions conditioned on all preceding tokens. For example, the third column uses ‘When young’ as input, and the fourth uses ‘When young children’ as input.

At last, Figure 10 provides an example to show how LLM generation evolves during pre-training. At the earliest stage (step 100), generation are random. By step 3,000, the model predicts only function tokens (e.g., ‘the’, ‘a’). By step 50,000, it generates locally coherent phrases like ‘learning to’ and ‘to be’. More complex predictions requiring capturing long-range dependencies, emerge at later stages in the 8B model. For instance, correctly predicting the next token after ‘...struggle with’ requires recalling the earlier context ‘learning to read’.

# 5 Function Token Hypothesis

Our experiments suggest that function tokens are crucial for memory consolidation and memory retrieval in LLMs, leading to our Function Token Hypothesis. During inference, function tokens activate the most predictive features from the context to direct the prediction of the next token (memory retrieval). During training, predicting the content token after the function tokens drives parameter updates and feature learning (memory consolidation).

We postulate that the function token hypothesis is the compound result of four factors in LLM training: the training loss (cross entropy loss), learning algorithm (SGD [40] or backpropagation [41]), model architecture (Transformer), and nature of language data.

The training of an LLM is driven by next token prediction. Maximally reducing the loss for next token prediction means making the prediction as accurate as possible. (Minimizing the total loss for predicting all next tokens in the training data is equivalent to compressing the training data as compactly as possible. [10]) During training, the SGD algorithm always manages to reduce the training loss the most by computing and utilizing the steepest descent.

Each block of the Transformer (decoder-only) consists of a multi-head self-attention layer followed by an FFN layer. Both the self-attention layer and the FFN layer can be viewed as key-value memories, as explained. Their roles, however, are different. The self-attention layer is responsible for producing a new internal vector from all internal vectors in the context (note that compositionality is the key characteristic of language [8]). The FFN layer is responsible for producing an output vector from the new internal vector. Knowledge is represented as parameters in the FFN layer, and features can be extracted from the output vector.

A natural language text is always segmented by function tokens. From each function token to one of its preceding function tokens, a chunk exists, extending until the beginning of the text. These chunks can represent a phrase, a sentence, or a paragraph, and they are nested. When the LLM’s prediction reaches the token immediately following a function token, this implies the start of predicting the next chunk; the task is far more challenging, as it requires understanding the meaning of the entire context up to that point. This high-challenge prediction compels the LLM to activate the most predictive features in the context during training and reactivate the most predictive features during inference.

Overall, the memory mechanisms of LLMs are extremely complex, due to the complexities of the models and algorithm, as well as the scales of the models and data. Nonetheless, we think that our extensive investigations have convincingly validated the function token hypothesis.

# 6 Related Work

Research on neural memory dates back to the Hopfield network [18], also known as associative memory network, which consolidates memories by adjusting weights between neurons. Hopfield networks have evolved into restricted Boltzmann machines [13] and feed-forward networks that utilize key-value memories [15]. Recent research on superposition [12] has shown that it is possible to uncover the features of neural networks such as Transformer, where the number of features is much larger than that of neurons. Through dictionary learning, the superposed activations can be decomposed into monosemantic features. Existing work has demonstrated that such decomposed features can effectively steer model behaviors [7, 34], by maintaining specific feature activations, controlling access to memories, and directing the model through the generation process.

Existing research on LLMs has identified important patterns involving function tokens. For example, separator tokens produce large activations [45] and distinct attention weights [6], enabling efficient KV cache designs retaining only separator caches. The crucial role of “formatting" in post-training is also widely recognized [23, 26, 56, 57], a function primarily controlled by tokens such as ‘\n’. Furthermore, recent work on reinforcement learning for reasoning finds that training primarily on high-entropy tokens like ‘thus’ improves performance [53], while Phi-4 [1] identifies ‘pivot tokens’, often following function tokens, as critical for response accuracy. We argue these are all function tokens, marked by high frequency and diverse contextual usage. This view is supported by previous work demonstrating that the effective learning of function token representations is crucial for overall LLM performance. Building on this, we propose the Function Token Hypothesis and analyze how these tokens drive memory retrieval and consolidation in LLMs.

# 7 Conclusion and Open Questions

In this work, we propose the function token hypothesis: during inference, function tokens activate the most predictive features from context to guide next token prediction. During pre-training, the prediction of content tokens preceding function tokens drives the model to learn and expand its features. Our experiments provide strong evidence for this hypothesis.

In the meantime, our study raises several open questions:

• One important question is how function tokens acquire the ability to dynamically activate predictive features, in contrast to content tokens. This capability likely emerges from the interplay of model architecture, data nature, training loss, and learning algorithm during training. Investigating this interaction is essential for a better understanding of the phenomena.   
• Post-training typically requires only a small number of training steps to achieve substantial improvements in capabilities such as instruction following, chain-of-thought reasoning, and search-agent behavior. Remarkably, training only on function tokens through reinforcement learning can enhance reasoning performance, suggesting that post-training merely activates latent capabilities acquired during pretraining. However, how post-training modifies these activation patterns in function tokens remains an open question.   
• In our pre-training experiments, we observe that scaling up (more training data, increased computation, and larger model size) reduces loss, accompanied by an increase in the number of learned features. Notably, function tokens consistently activate most features, exhibiting a scale-free property (tokenfeature degree distribution follows a power law) throughout training. However, the dynamics of feature formation and the underlying reason of this scale-free property remain unclear, and whether these phenomena follow specific principles requires further investigation.   
• Our case studies confirm existing findings that middle layers offer superior interpretability and steerability However, the mechanistic explanation for why this steerability is concentrated in middle layers, rather than shallow or deep layers, remains elusive.

# References

[1] Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024.   
[2] AI Anthropic. The claude 3 model family: Opus, sonnet, haiku. Claude-3 Model Card, 1(1):4, 2024.   
[3] Joseph Bloom, Curt Tigges, Anthony Duong, and David Chanin. Saelens. https://github.com/jbloomAus/ SAELens, 2024.   
[4] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings. neurips.cc/paper_files/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf.   
[5] Rudolf Carnap. Logical syntax of language. Routledge, 2014.   
[6] Guoxuan Chen, Han Shi, Jiawei Li, Yihang Gao, Xiaozhe Ren, Yimeng Chen, Xin Jiang, Zhenguo Li, Weiyang Liu, and Chao Huang. Sepllm: Accelerate large language models by compressing one segment into one separator, 2025. URL https://arxiv.org/abs/2412.12094.   
[7] Runjin Chen, Andy Arditi, Henry Sleight, Owain Evans, and Jack Lindsey. Persona vectors: Monitoring and controlling character traits in language models, 2025. URL https://arxiv.org/abs/2507.21509.   
[8] Noam Chomsky. Syntactic structures. Mouton de Gruyter, 2002.   
[9] Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models, 2023. URL https://arxiv.org/abs/2309.08600.   
[10] Grégoire Delétang, Anian Ruoss, Paul-Ambroise Duquenne, Elliot Catt, Tim Genewein, Christopher Mattern, Jordi Grau-Moya, Li Kevin Wenliang, Matthew Aitchison, Laurent Orseau, Marcus Hutter, and Joel Veness. Language modeling is compression, 2024. URL https://arxiv.org/abs/2309.10668.   
[11] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. https://transformer-circuits.pub/2021/framework/index.html.   
[12] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition. Transformer Circuits Thread, 2022. https://transformer-circuits.pub/2022/toy_model/index.html.   
[13] Asja Fischer and Christian Igel. An introduction to restricted boltzmann machines. In Iberoamerican congress on pattern recognition, pages 14–36. Springer, 2012.   
[14] Leo Gao, Tom Dupre la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=tcsZt9ZNKD.   
[15] Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories, 2021. URL https://arxiv.org/abs/2012.14913.   
[16] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius,

Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, Zoe Papakipos, Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi, Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip

Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu, Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.   
[17] Roee Hendel, Mor Geva, and Amir Globerson. In-context learning creates task vectors. arXiv preprint arXiv:2310.15916, 2023.   
[18] John J Hopfield. Neural networks and physical systems with emergent collective computational abilities. Proceedings of the national academy of sciences, 79(8):2554–2558, 1982.   
[19] Sheena A Josselyn and Susumu Tonegawa. Memory engrams: Recalling the past and imagining the future. Science, 367(6473):eaaw4325, 2020.   
[20] Jasmeen Kanwal, Kenny Smith, Jennifer Culbertson, and Simon Kirby. Zipf’s law of abbreviation and the principle of least effort: Language users optimise a miniature lexicon for efficient communication. Cognition, 165:45–52, 2017.   
[21] Adam Karvonen, Can Rager, Johnny Lin, Curt Tigges, Joseph Bloom, David Chanin, Yeu-Tong Lau, Eoin Farrell, Callum McDougall, Kola Ayonrinde, Demian Till, Matthew Wearden, Arthur Conmy, Samuel Marks, and Neel Nanda. Saebench: A comprehensive benchmark for sparse autoencoders in language model interpretability, 2025. URL https://arxiv.org/abs/2503.09532.   
[22] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.   
[23] Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu, Xiangxi Mo, Eric Tang, Sumanth Hegde, Kourosh Hakhamaneshi, Shishir G. Patil, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Llms can easily learn to reason from demonstrations structure, not content, is what matters!, 2025. URL https://arxiv.org/abs/2502.07374.   
[24] Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2, 2024. URL https://arxiv.org/abs/2408.05147.   
[25] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.   
[26] Siddarth Mamidanna, Daking Rai, Ziyu Yao, and Yilun Zhou. All for one: Llms solve mental math at the last token with information transferred from other tokens, 2025. URL https://arxiv.org/abs/2509.09650.   
[27] Jack Merullo, Carsten Eickhoff, and Ellie Pavlick. Circuit component reuse across tasks in transformer language models. In The Twelfth International Conference on Learning Representations, 2024.   
[28] Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.   
[29] Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. Distill, 2020. doi: 10.23915/distill.00024.001. https://distill.pub/2020/circuits/zoom-in.   
[30] OpenAI. Chatgpt: Optimizing language models for dialogue, 2022. URL https://openai.com/blog/chatgpt.   
[31] R OpenAI. Gpt-4 technical report. arxiv 2303.08774. View in Article, 2(5):1, 2023.

[32] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.   
[33] Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering llama 2 via contrastive activation addition. arXiv preprint arXiv:2312.06681, 2023.   
[34] Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering llama 2 via contrastive activation addition, 2024. URL https://arxiv.org/abs/2312.06681.   
[35] Guilherme Penedo, Hynek Kydlíček, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=n6SCkn2QaG.   
[36] Steven T Piantadosi. Zipf’s word frequency law in natural language: A critical review and future directions. Psychonomic bulletin & review, 21(5):1112–1130, 2014.   
[37] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.   
[38] Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Tom Lieberum, Vikrant Varma, János Kramár, Rohin Shah, and Neel Nanda. Improving dictionary learning with gated sparse autoencoders, 2024. URL https://arxiv.org/abs/2404.16014.   
[39] Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, János Kramár, and Neel Nanda. Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders, 2024. URL https://arxiv.org/abs/2407.14435.   
[40] Sebastian Ruder. An overview of gradient descent optimization algorithms, 2017. URL https://arxiv.org/abs/ 1609.04747.   
[41] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by back-propagating errors. nature, 323(6088):533–536, 1986.   
[42] Pavel Rychl`y. Words’ burstiness in language models. In RASLAN, pages 131–137, 2011.   
[43] Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://www.cerebras.net/ blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama, 2023. URL https:// huggingface.co/datasets/cerebras/SlimPajama-627B.   
[44] Anna Soligo, Edward Turner, Senthooran Rajamanoharan, and Neel Nanda. Convergent linear representations of emergent misalignment, 2025. URL https://arxiv.org/abs/2506.11618.   
[45] Mingjie Sun, Xinlei Chen, J. Zico Kolter, and Zhuang Liu. Massive activations in large language models, 2024. URL https://arxiv.org/abs/2402.17762.   
[46] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.   
[47] Gemma Team. Gemma. 2024. doi: 10.34740/KAGGLE/M/3301. URL https://www.kaggle.com/m/3301.   
[48] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, Anton Tsitsulin, Nino Vieillard, Piotr Stanczyk, Sertan Girgin, Nikola Momchev, Matt Hoffman, Shantanu Thakoor, Jean-Bastien Grill, Behnam Neyshabur, Olivier Bachem, Alanna Walton, Aliaksei Severyn, Alicia Parrish, Aliya Ahmad, Allen Hutchison, Alvin Abdagic, Amanda Carl, Amy Shen, Andy Brock, Andy Coenen, Anthony Laforge, Antonia Paterson, Ben Bastian, Bilal Piot, Bo Wu, Brandon Royal, Charlie Chen, Chintu Kumar, Chris Perry, Chris Welty, Christopher A. Choquette-Choo, Danila Sinopalnikov, David Weinberger, Dimple Vijaykumar, Dominika Rogozińska, Dustin Herbison, Elisa Bandy, Emma Wang, Eric Noland, Erica Moreira, Evan Senter, Evgenii Eltyshev, Francesco Visin, Gabriel Rasskin, Gary Wei, Glenn Cameron, Gus Martins, Hadi Hashemi, Hanna Klimczak-Plucińska, Harleen Batra, Harsh Dhand, Ivan Nardini, Jacinda Mein, Jack Zhou, James Svensson, Jeff Stanway, Jetha Chan, Jin Peng Zhou, Joana Carrasqueira, Joana Iljazi, Jocelyn Becker, Joe Fernandez, Joost van Amersfoort, Josh

Gordon, Josh Lipschultz, Josh Newlan, Ju yeong Ji, Kareem Mohamed, Kartikeya Badola, Kat Black, Katie Millican, Keelin McDonell, Kelvin Nguyen, Kiranbir Sodhia, Kish Greene, Lars Lowe Sjoesund, Lauren Usui, Laurent Sifre, Lena Heuermann, Leticia Lago, Lilly McNealus, Livio Baldini Soares, Logan Kilpatrick, Lucas Dixon, Luciano Martins, Machel Reid, Manvinder Singh, Mark Iverson, Martin Görner, Mat Velloso, Mateo Wirth, Matt Davidow, Matt Miller, Matthew Rahtz, Matthew Watson, Meg Risdal, Mehran Kazemi, Michael Moynihan, Ming Zhang, Minsuk Kahng, Minwoo Park, Mofi Rahman, Mohit Khatwani, Natalie Dao, Nenshad Bardoliwalla, Nesh Devanathan, Neta Dumai, Nilay Chauhan, Oscar Wahltinez, Pankil Botarda, Parker Barnes, Paul Barham, Paul Michel, Pengchong Jin, Petko Georgiev, Phil Culliton, Pradeep Kuppala, Ramona Comanescu, Ramona Merhej, Reena Jana, Reza Ardeshir Rokni, Rishabh Agarwal, Ryan Mullins, Samaneh Saadat, Sara Mc Carthy, Sarah Cogan, Sarah Perrin, Sébastien M. R. Arnold, Sebastian Krause, Shengyang Dai, Shruti Garg, Shruti Sheth, Sue Ronstrom, Susan Chan, Timothy Jordan, Ting Yu, Tom Eccles, Tom Hennigan, Tomas Kocisky, Tulsee Doshi, Vihan Jain, Vikas Yadav, Vilobh Meshram, Vishal Dharmadhikari, Warren Barkley, Wei Wei, Wenming Ye, Woohyun Han, Woosuk Kwon, Xiang Xu, Zhe Shen, Zhitao Gong, Zichuan Wei, Victor Cotruta, Phoebe Kirk, Anand Rao, Minh Giang, Ludovic Peran, Tris Warkentin, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia Hadsell, D. Sculley, Jeanine Banks, Anca Dragan, Slav Petrov, Oriol Vinyals, Jeff Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Sebastian Borgeaud, Noah Fiedel, Armand Joulin, Kathleen Kenealy, Robert Dadashi, and Alek Andreev. Gemma 2: Improving open language models at a practical size, 2024. URL https://arxiv.org/abs/2408.00118.   
[49] Adly Templeton, Tom Conerly, Jonathan Marcus, Jack Lindsey, Trenton Bricken, Brian Chen, Adam Pearce, Craig Citro, Emmanuel Ameisen, Andy Jones, Hoagy Cunningham, Nicholas L Turner, Callum McDougall, Monte MacDiarmid, C. Daniel Freeman, Theodore R. Sumers, Edward Rees, Joshua Batson, Adam Jermyn, Shan Carter, Chris Olah, and Tom Henighan. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Transformer Circuits Thread, 2024. URL https://transformer-circuits.pub/2024/ scaling-monosemanticity/index.html.   
[50] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
[51] Kevin Ro Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. In The Eleventh International Conference on Learning Representations, 2023.   
[52] Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning, 2025. URL https://arxiv.org/abs/2506.01939.   
[53] Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. arXiv preprint arXiv:2506.01939, 2025.   
[54] Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652, 2021.   
[55] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
[56] Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. Limo: Less is more for reasoning, 2025. URL https://arxiv.org/abs/2502.03387.   
[57] Chunting Zhou, Pengfei Liu, Puxin Xu, Srini Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, LILI YU, Susan Zhang, Gargi Ghosh, Mike Lewis, Luke Zettlemoyer, and Omer Levy. LIMA: Less is more for alignment. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id=KBMOKmX2he.

# Appendix

# A Steering Method for Large Language Models

Given a target trait, our goal is to identify the corresponding feature in Geema2-9B from the SAE decomposition. Specifically, we extract the feature at the last function token in the prompt, which is an newline token. The process involves four steps:

Step 1. Collect contrastive prompts. We construct two sets of prompts: (i) a single prompt that enforces the target trait, and (ii) 20 prompts that do not. For example, to isolate the ‘Speak Chinese’ feature, we use Prompt 1 (‘Answer the question in Chinese: What is the capital of UK?’) as the trait-enforcing prompt. This explicitly instructs the model to respond in Chinese. In contrast, Prompt 3 (‘Where is Mount Fuji?’) is included in the non-trait set, as it contains no language specification and thus defaults to English. The trait-enforcing prompt is used to identify the relevant layer and feature. The non-trait prompts serve as a test set to evaluate steering effectiveness.

Step 2. Identify the most informative layer. For each layer $\it l$ , we take the activation of the final function token in the trait-enforcing prompt (e.g., Prompt 1 for ‘Speak Chinese’) and denote it as a steer vector [33], $v _ { l } \in \mathbb { R } ^ { d }$ . We then modify the last function token’s activation of each test prompt as $h _ { l }  h _ { l } + v _ { l }$ , generate responses, and measure the success rate of producing the trait (e.g., ‘Speak Chinese’). The layer with the highest success rate is chosen as the most informative. For the traits ‘Speak Chinese’, ‘Russia’, and ‘UK’, the most informative layer all correspond to layer 26.

Step 3. Identify the feature. At the chosen layer, we decompose $v _ { l }$ using SAE. By Equation 8, $v _ { l }$ can be expressed by

$$
v _ {l} = W _ {\mathrm {d e c}} \cdot \mathbf {z} + \mathbf {b} _ {\mathrm {d e c}}, \tag {4}
$$

where $\mathbf { z } = ( z _ { 1 } , z _ { 2 } , \cdots , z _ { n } ) ^ { \top }$ . To locate the trait-specific feature, we rank features by activation strength $z _ { i }$ in descending order. We then apply a binary search to find the smallest $k$ such that activating the top- $k$ features enables the trait, while the top- $\left( k - 1 \right)$ does not. The corresponding steering vector in hidden space is:

$$
v _ {l} ^ {S _ {k}} = \alpha \cdot W _ {\mathrm {d e c}} \sum_ {i \in S _ {k}} \mathbf {e} _ {i}, \tag {5}
$$

where $S _ { k }$ is the set of top- $k$ feature IDs, $\mathtt { e } _ { i }$ is the $_ i$ -th standard basis vector, and $\alpha$ is the steering strength. We apply $h _ { l } \gets h _ { l } + v _ { l } ^ { S _ { k } }$ on the test set and evaluate the success. For ‘Speak Chinese’, the identified feature is ID 15261 at layer 26; other examples include ‘Russia’ (feature ID 9591, layer 26) and ‘UK’ (feature ID 13751, layer 26).

Step 4. Steering the model. Once the feature $i$ is identified, the model can be steered with the feature-specific steering vector:

$$
v _ {l} ^ {i} = \alpha_ {i} \cdot W _ {\mathrm {d e c}} \mathbf {e} _ {i}. \tag {6}
$$

By applying $h _ { l }  h _ { l } + v _ { l } ^ { i }$ to the last function token of a prompt, we can induce traits such as ‘Speak Chinese’, ‘Russia’ or ‘UK’.

# B Additional Case Study

We present more interesting examples of steering activations on function tokens, as shown in Figure 11.

# C SAE Training Details

Given an activation $\mathbf { x } \in \mathbb { R } ^ { d }$ from the residual stream with $n$ dimensions, the JumpReLU-SAE comprises an encoder and decoder:

$$
\mathbf {z} = \operatorname {J u m p R e L U} _ {\theta} \left(W _ {\mathrm {e n c}} \mathbf {x} + \mathbf {b} _ {\mathrm {e n c}}\right) \tag {7}
$$

$$
\hat {\mathbf {x}} = W _ {\mathrm {d e c}} \mathbf {z} + \mathbf {b} _ {\mathrm {d e c}} \tag {8}
$$

<table><tr><td>Steer Method</td><td>Prompt</td><td>Please name my newborn baby daughter.</td><td>Recommend me a traditional alcoholic beverage.</td><td>Recommend me a traditional dish.</td></tr><tr><td>None</td><td></td><td>Eleanor</td><td>Sake.</td><td>Chicken Tikka Masala</td></tr><tr><td>Activate &#x27;Speak Chinese&#x27; feature</td><td></td><td>小雨 (Xiaо Yǔ)</td><td>啤酒 (Pijiǔ) - Chinese Beer</td><td>意大利面 Carbonara</td></tr><tr><td>Activate &#x27;UK&#x27; feature</td><td></td><td>Amelia</td><td>Scotch whisky.</td><td>Shepherd&#x27;s Pie</td></tr><tr><td>Activate &#x27;Russia&#x27; feature</td><td></td><td>Anastasia</td><td>Vodka Martini.</td><td>Borscht.</td></tr><tr><td>Activate &#x27;Speak Chinese&#x27; feature + Activate &quot;UK&quot; feature</td><td></td><td>艾莉亚 (Alia)</td><td>威士忌 (Whiskey)</td><td>英国的鱼薯条</td></tr><tr><td>Activate &#x27;Speak Chinese&#x27; feature + Activate &quot;Russia&quot; feature</td><td></td><td>安娜 (Anna)</td><td>伏特加 (Vodka)</td><td>俄式肉丸子 (Russian Meatballs)</td></tr></table>

Figure 11 Response of Gemma2-9B-it when editing the activation at the final function token (‘\n’) in the prompt. The Chinese terms shown in the table and their corresponding English translations are: 雨 (Xiaoyu, a common Chinese 小feminine nickname), 啤 (beer), 意 利面 (Carbonara), 艾丽娅 (Alia), 威士 (Whiskey), 英国的鱼薯条 (British fish 酒 大 忌and chips), 安娜 (Anna), 伏 加 (Vodka), and 俄 肉丸子 (Russian meatballs).

where $W _ { \mathrm { e n c } } \in \mathbb { R } ^ { n \times d }$ , ${ \sf b } _ { \mathrm { e n c } } \in \mathbb { R } ^ { n }$ , ${ \pmb { \mathsf { b } } } _ { \mathrm { d e c } } \ \in \ \mathbb { R } ^ { d }$ and $W _ { \mathrm { d e c } } ~ \in ~ \mathbb { R } ^ { d \times n }$ . The optimization objective combines reconstruction loss with a $L _ { 0 }$ sparsity penalty:

$$
\mathcal {L} (\mathbf {x}) = \underbrace {\| \mathbf {x} - \hat {\mathbf {x}} \| _ {2} ^ {2}} _ {\mathcal {L} _ {\text {r e c o n s t r u c t}}} + \underbrace {\lambda \| \mathbf {z} \| _ {0}} _ {\mathcal {L} _ {\text {s p a r s i t y}}} \tag {9}
$$

We train our JumpReLU-SAEs using the open-source library sae_lens1 [3]. Training data consists of one billion activations collected from pre-training dataset [35] with a context size of 1024 tokens. The dictionary width is set to 16 times the activation dimension, resulting in a dictionary size of 65,536. We use a constant learning rate of $1 \times 1 0 ^ { - 5 }$ .

We adopt default JumpReLU-SAE training setting: batch size 4096 and a dead-feature [49] detection window of 1000. For JumpReLU, we set the bandwidth to 0.02 and the initialization threshold to 0.01.

SAE training involves a tradeoff between reconstruction quality and sparsity. To quantify reconstruction quality, we use the cross-entropy reconstruction score [21], which is defined as $\frac { H _ { * } - H _ { 0 } } { H _ { o r i g } - H _ { 0 } }$ , where $H _ { o r i g }$ is the cross-entropy loss of the original model for next-token prediction, $H _ { * }$ is the cross-entropy loss after substituting the model activation $x$ with its SAE reconstruction during the forward pass, and $H _ { 0 }$ is the cross-entropy loss when zero-ablating $x$ . The metric ranges from 0 to 1, with higher values indicating more faithful reconstruction.

Using the default $L _ { 0 }$ penalty coefficient, $\lambda = 4$ , we find that reconstruction scores varied across early (3,000 steps), intermediate (50,000 steps), and late (130,000 steps) checkpoints, as shown in Figure 12. To make feature counts comparable across these stages, we tuned $\lambda$ to similar reconstruction scores. Specifically, we use $\lambda = 1 0$ for early checkpoint, $\lambda = 4$ for the intermediate checkpoint, and $\lambda = 2 . 5$ for the late checkpoint.

From Figure 12, we also observe that, as pre-training progresses, the model’s feature representations become increasingly complex and more difficult to decompose.

![](images/4d98175c467a30cbd56cb7ef6d60110bb1a294cc1a767d1aa47d2706c8d31d77.jpg)  
Figure 12 Cross-Entropy reconstruction scores under varying $\lambda$ Values

# D Function Token List

Table 2 presents all function tokens identified in our experiments, ranked by frequency in SlimPajama-627B in descending order. Tokens not appearing in this table are classified as content tokens.

Table 2 Token statistics with corresponding document coverage, token fractions, and cumulative fractions.   

<table><tr><td>Token Text</td><td>Document Coverage</td><td>Token Fraction</td><td>Cumulative Fraction</td></tr><tr><td>,</td><td>95.00%</td><td>3.60%</td><td>3.60%</td></tr><tr><td>_the</td><td>90.92%</td><td>3.19%</td><td>6.79%</td></tr><tr><td>_</td><td>95.80%</td><td>2.31%</td><td>9.10%</td></tr><tr><td>_and</td><td>89.69%</td><td>1.81%</td><td>10.91%</td></tr><tr><td>_of</td><td>87.59%</td><td>1.80%</td><td>12.71%</td></tr><tr><td>_to</td><td>88.71%</td><td>1.68%</td><td>14.40%</td></tr><tr><td>_--</td><td>81.35%</td><td>1.59%</td><td>15.99%</td></tr><tr><td>_a</td><td>87.62%</td><td>1.33%</td><td>17.32%</td></tr><tr><td>_in</td><td>86.04%</td><td>1.16%</td><td>18.48%</td></tr><tr><td>_n</td><td>84.58%</td><td>0.91%</td><td>19.39%</td></tr><tr><td>_is</td><td>78.90%</td><td>0.74%</td><td>20.13%</td></tr><tr><td>_n</td><td>42.30%</td><td>0.70%</td><td>20.84%</td></tr><tr><td>_for</td><td>79.82%</td><td>0.64%</td><td>21.48%</td></tr><tr><td>_that</td><td>67.02%</td><td>0.62%</td><td>22.09%</td></tr><tr><td>_s</td><td>63.02%</td><td>0.49%</td><td>22.58%</td></tr><tr><td>_on</td><td>72.40%</td><td>0.47%</td><td>23.05%</td></tr><tr><td>_with</td><td>73.68%</td><td>0.47%</td><td>23.52%</td></tr><tr><td>_ (</td><td>55.05%</td><td>0.47%</td><td>23.99%</td></tr><tr><td>:</td><td>52.73%</td><td>0.42%</td><td>24.41%</td></tr><tr><td>_it</td><td>57.50%</td><td>0.38%</td><td>24.79%</td></tr><tr><td>_I</td><td>37.43%</td><td>0.38%</td><td>25.17%</td></tr><tr><td>_as</td><td>61.49%</td><td>0.37%</td><td>25.54%</td></tr><tr><td>_you</td><td>47.06%</td><td>0.35%</td><td>25.90%</td></tr><tr><td>_be</td><td>60.03%</td><td>0.33%</td><td>26.23%</td></tr><tr><td>_are</td><td>60.45%</td><td>0.33%</td><td>26.56%</td></tr><tr><td>_was</td><td>45.51%</td><td>0.33%</td><td>26.89%</td></tr><tr><td colspan="4">Continued on next page</td></tr></table>

Table 2 – continued from previous page   

<table><tr><td>Token Text</td><td>Document Coverage</td><td>Token Fraction</td><td>Cumulative Fraction</td></tr><tr><td>1</td><td>40.84%</td><td>0.30%</td><td>27.18%</td></tr><tr><td>_at</td><td>59.38%</td><td>0.29%</td><td>27.48%</td></tr><tr><td>_by</td><td>58.44%</td><td>0.29%</td><td>27.77%</td></tr><tr><td>_“</td><td>43.01%</td><td>0.28%</td><td>28.05%</td></tr><tr><td>_The</td><td>55.12%</td><td>0.28%</td><td>28.34%</td></tr><tr><td>_from</td><td>61.23%</td><td>0.28%</td><td>28.62%</td></tr><tr><td>)</td><td>44.33%</td><td>0.28%</td><td>28.90%</td></tr><tr><td>this</td><td>56.27%</td><td>0.26%</td><td>29.16%</td></tr><tr><td>_have</td><td>55.12%</td><td>0.26%</td><td>29.41%</td></tr><tr><td>_or</td><td>50.42%</td><td>0.25%</td><td>29.66%</td></tr><tr><td>2</td><td>39.09%</td><td>0.25%</td><td>29.91%</td></tr><tr><td>-</td><td>38.67%</td><td>0.24%</td><td>30.15%</td></tr><tr><td>_an</td><td>56.55%</td><td>0.23%</td><td>30.38%</td></tr><tr><td>0</td><td>31.70%</td><td>0.22%</td><td>30.60%</td></tr><tr><td>.not</td><td>46.51%</td><td>0.21%</td><td>30.81%</td></tr><tr><td>_will</td><td>46.71%</td><td>0.19%</td><td>31.00%</td></tr><tr><td>_can</td><td>47.99%</td><td>0.19%</td><td>31.19%</td></tr><tr><td>_has</td><td>49.09%</td><td>0.19%</td><td>31.38%</td></tr><tr><td>201</td><td>33.71%</td><td>0.18%</td><td>31.56%</td></tr><tr><td>we</td><td>35.13%</td><td>0.18%</td><td>31.74%</td></tr><tr><td>\</td><td>1.30%</td><td>0.17%</td><td>31.91%</td></tr><tr><td>The</td><td>48.49%</td><td>0.17%</td><td>32.08%</td></tr><tr><td>_your</td><td>34.99%</td><td>0.17%</td><td>32.25%</td></tr><tr><td>3</td><td>35.29%</td><td>0.17%</td><td>32.41%</td></tr><tr><td>_but</td><td>41.84%</td><td>0.16%</td><td>32.57%</td></tr><tr><td>_his</td><td>25.09%</td><td>0.16%</td><td>32.73%</td></tr><tr><td>_</td><td>34.19%</td><td>0.16%</td><td>32.88%</td></tr><tr><td>_all</td><td>45.24%</td><td>0.15%</td><td>33.04%</td></tr><tr><td>_they</td><td>39.27%</td><td>0.15%</td><td>33.19%</td></tr><tr><td>—he</td><td>23.69%</td><td>0.15%</td><td>33.34%</td></tr><tr><td>{</td><td>1.18%</td><td>0.15%</td><td>33.49%</td></tr><tr><td>_they</td><td>35.37%</td><td>0.15%</td><td>33.64%</td></tr><tr><td>&#x27;t</td><td>33.12%</td><td>0.15%</td><td>33.78%</td></tr><tr><td>更多的</td><td>42.84%</td><td>0.14%</td><td>33.93%</td></tr><tr><td>-one</td><td>41.94%</td><td>0.14%</td><td>34.07%</td></tr><tr><td>_which</td><td>40.67%</td><td>0.14%</td><td>34.21%</td></tr><tr><td>4</td><td>31.49%</td><td>0.13%</td><td>34.34%</td></tr><tr><td>5</td><td>32.71%</td><td>0.13%</td><td>34.47%</td></tr><tr><td>$</td><td>12.48%</td><td>0.13%</td><td>34.61%</td></tr><tr><td>\_</td><td>0.90%</td><td>0.13%</td><td>34.73%</td></tr><tr><td>-about</td><td>37.54%</td><td>0.13%</td><td>34.86%</td></tr><tr><td>---</td><td>5.40%</td><td>0.11%</td><td>34.97%</td></tr><tr><td>;</td><td>21.62%</td><td>0.11%</td><td>35.09%</td></tr><tr><td>_who</td><td>33.50%</td><td>0.11%</td><td>35.20%</td></tr><tr><td>_also</td><td>40.22%</td><td>0.11%</td><td>35.31%</td></tr><tr><td>_our</td><td>30.62%</td><td>0.11%</td><td>35.42%</td></tr><tr><td>_were</td><td>27.00%</td><td>0.11%</td><td>35.53%</td></tr><tr><td>_out</td><td>36.49%</td><td>0.11%</td><td>35.64%</td></tr><tr><td>/</td><td>20.32%</td><td>0.11%</td><td>35.75%</td></tr><tr><td>6</td><td>28.01%</td><td>0.11%</td><td>35.86%</td></tr></table>

Continued on next page

Table 2 – continued from previous page   

<table><tr><td>Token Text</td><td>Document Coverage</td><td>Token Fraction</td><td>Cumulative Fraction</td></tr><tr><td>_up</td><td>36.43%</td><td>0.11%</td><td>35.97%</td></tr><tr><td>8</td><td>28.60%</td><td>0.11%</td><td>36.08%</td></tr><tr><td>Been</td><td>35.32%</td><td>0.11%</td><td>36.18%</td></tr><tr><td>_had</td><td>25.51%</td><td>0.11%</td><td>36.29%</td></tr><tr><td>_if</td><td>30.49%</td><td>0.10%</td><td>36.39%</td></tr><tr><td>7</td><td>27.31%</td><td>0.10%</td><td>36.50%</td></tr><tr><td>(so</td><td>33.25%</td><td>0.10%</td><td>36.60%</td></tr><tr><td>.my</td><td>20.96%</td><td>0.10%</td><td>36.70%</td></tr><tr><td>=</td><td>6.62%</td><td>0.10%</td><td>36.80%</td></tr><tr><td>_time</td><td>34.79%</td><td>0.10%</td><td>36.90%</td></tr><tr><td>_her</td><td>15.21%</td><td>0.10%</td><td>37.00%</td></tr><tr><td>9</td><td>26.28%</td><td>0.10%</td><td>37.10%</td></tr><tr><td>_,-</td><td>19.91%</td><td>0.10%</td><td>37.20%</td></tr><tr><td>_s</td><td>27.13%</td><td>0.10%</td><td>37.30%</td></tr><tr><td>_would</td><td>27.35%</td><td>0.09%</td><td>37.39%</td></tr><tr><td>_new</td><td>32.43%</td><td>0.09%</td><td>37.49%</td></tr><tr><td>_when</td><td>32.82%</td><td>0.09%</td><td>37.58%</td></tr><tr><td>_other</td><td>33.77%</td><td>0.09%</td><td>37.67%</td></tr><tr><td>_there</td><td>30.15%</td><td>0.09%</td><td>37.76%</td></tr><tr><td>_A</td><td>28.29%</td><td>0.09%</td><td>37.86%</td></tr><tr><td>_its</td><td>29.64%</td><td>0.09%</td><td>37.95%</td></tr><tr><td>_It</td><td>31.56%</td><td>0.09%</td><td>38.04%</td></tr><tr><td>_like</td><td>30.40%</td><td>0.09%</td><td>38.13%</td></tr><tr><td>_do</td><td>29.89%</td><td>0.09%</td><td>38.22%</td></tr><tr><td>what</td><td>28.23%</td><td>0.09%</td><td>38.39%</td></tr><tr><td>_,</td><td>3.87%</td><td>0.09%</td><td>38.48%</td></tr><tr><td>_,</td><td>18.94%</td><td>0.09%</td><td>38.57%</td></tr><tr><td>into</td><td>31.66%</td><td>0.09%</td><td>38.65%</td></tr><tr><td>200</td><td>19.03%</td><td>0.08%</td><td>38.74%</td></tr><tr><td>}</td><td>2.01%</td><td>0.08%</td><td>38.82%</td></tr><tr><td>_than</td><td>30.00%</td><td>0.08%</td><td>38.90%</td></tr><tr><td>_said</td><td>19.12%</td><td>0.08%</td><td>38.98%</td></tr><tr><td>_some</td><td>29.97%</td><td>0.08%</td><td>39.06%</td></tr><tr><td>_them</td><td>27.36%</td><td>0.08%</td><td>39.14%</td></tr><tr><td>_In</td><td>28.39%</td><td>0.08%</td><td>39.22%</td></tr><tr><td>_ &amp;</td><td>17.66%</td><td>0.08%</td><td>39.30%</td></tr><tr><td>_ -</td><td>18.50%</td><td>0.08%</td><td>39.38%</td></tr><tr><td>_people</td><td>24.05%</td><td>0.08%</td><td>39.46%</td></tr><tr><td>ing</td><td>29.18%</td><td>0.08%</td><td>39.53%</td></tr><tr><td>first</td><td>29.94%</td><td>0.08%</td><td>39.61%</td></tr><tr><td>)\\n</td><td>13.24%</td><td>0.08%</td><td>39.69%</td></tr><tr><td>I</td><td>23.86%</td><td>0.08%</td><td>39.76%</td></tr><tr><td>?</td><td>24.01%</td><td>0.08%</td><td>39.84%</td></tr><tr><td>A</td><td>27.74%</td><td>0.08%</td><td>39.92%</td></tr><tr><td>just</td><td>27.64%</td><td>0.07%</td><td>39.99%</td></tr></table>
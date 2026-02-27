# TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG

Pengqian Lu, Jie $\mathbf { L } \mathbf { u } ^ { * }$ , Anjin Liu, and Guangquan Zhang

Australian Artificial Intelligence Institute (AAII)

University of Technology Sydney

Ultimo, NSW 2007, Australia

{Pengqian.Lu@student., Jie.Lu@, Anjin.Liu@, Guangquan.Zhang@}uts.edu.au

# Abstract

Detecting hallucinations in Retrieval-Augmented Generation remains a challenge. Prior approaches attribute hallucinations to a binary conflict between internal knowledge stored in FFNs and the retrieved context. However, this perspective is incomplete, failing to account for the impact of other components of the LLM, such as the user query, previously generated tokens, the self token, and the Final LayerNorm adjustment. To comprehensively capture the impact of these components on hallucination detection, we propose TPA which mathematically attributes each token’s probability to seven distinct sources: Query, RAG Context, Past Token, Self Token, FFN, Final LayerNorm, and Initial Embedding. This attribution quantifies how each source contributes to the generation of the next token. Specifically, we aggregate these attribution scores by Part-of-Speech (POS) tags to quantify the contribution of each model component to the generation of specific linguistic categories within a response. By leveraging these patterns, such as detecting anomalies where Nouns rely heavily on LayerNorm, TPA effectively identifies hallucinated responses. Extensive experiments show that TPA achieves state-of-the-art performance.

# 1 Introduction

Large Language Models (LLMs), despite their impressive capabilities, are prone to hallucinations (Huang et al., 2025). Consequently, Retrieval-Augmented Generation (RAG) (Lewis et al., 2020) is widely used to alleviate hallucinations by grounding models in external knowledge. However, RAG systems are not perfect. They can still hallucinate by ignoring or misinterpreting the retrieved information (Sun et al., 2025). Detecting such failures is therefore a critical challenge.

Step 1: Input & Generated Response From RagTruth Dataset

Query

Briefly answer the followinq question: how to soak off gel polish..

Retrieved Context (RAG)

"Gelish Soak OffGel Nail Polish isaunique formula that lasts up to 21 days on natural nails. It applies ikepolishandcuresin anLEDlamp..

LLM Response (with POS-Tag Visualization)

![](images/1d71133cfc8a338061091fc9c288c386bfbcea571028e104fba7d97d40efd0a5.jpg)  
Legend:

![](images/c297baf35006c30b39306d582df49a7ed2c8979e494e85f3a3ca3000300b4518.jpg)

Step 2: Per-POS Tag Component Attribution

7-component average contribution per POS tag (acrossaltokens).

<table><tr><td>POS Tag</td><td>Query</td><td>RAG</td><td>Past Out</td><td>LN</td><td>Self</td><td>FFN</td><td>Initial</td></tr><tr><td>NOUN</td><td>0.1579</td><td>0.1563</td><td>0.0892</td><td>0.0067</td><td>0.0143</td><td>0.5362</td><td>0.0</td></tr><tr><td>NUM</td><td>0.0889</td><td>0.0199</td><td>0.0970</td><td>0.0425</td><td>0.0056</td><td>0.6424</td><td>0.0</td></tr><tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr></table>

(a) TPA attributes each next-token probability to seven sources, then aggregates source attributions by POS tags to form the features for detecting hallucination.

![](images/5dfac98a82cd382b486a88bc737f16e7b0f5bd481274bf3d051026fa28defe42.jpg)  
Figure 1: Applying the TPA framework to a Llama2-7b response from RAGTruth dataset (Niu et al., 2024).

(b) Feature-importance analysis (SHAP) shows that the detector leverages source contributions conditioned on POS tags. For example, responses are more likely to be hallucinated when RAG contributes little to NOUN tokens or when LN contributes too much to NUM tokens.

The prevailing paradigm for hallucination detection typically relies on hand-crafted proxy signals. For example, common approaches detect hallucination through consistency checks (Manakul et al., 2023) or scalar uncertainty metrics such as semantic entropy (Han et al., 2024). However, these

methods only measure the symptoms of hallucination, such as output variance or surface confidence, rather than the underlying architectural causes. Consequently, they often fail when a model is confidently incorrect (Simhi et al., 2025).

To address the root cause of hallucination, recent research has shifted focus to the model’s internal representations. Pioneering works such as ReDeEP (Sun et al., 2025) explicitly assume the RAG context is correct. They reveal that hallucinations in RAG typically stem from a disproportionate dominance of internal parametric knowledge (stored in FFNs) over the retrieved external context.

This insight inspires a fundamental question: Is the binary conflict between FFNs and RAG the only cause of hallucination? Critical components like LayerNorm and User Query are often overlooked. Do contributions from these sources also drive hallucinations? In this paper, we extend the analysis to cover all additive components along the transformer residual stream. This approach enables detection based on the model’s full internal mechanics instead of relying on partial proxy signals.

To achieve this, we also assume the RAG context contains relevant information and introduce TPA (Next Token Probability Attribution for Detecting Hallucinations in RAG). This framework mathematically attributes the final probability of each token to seven distinct sources: Query, RAG, Past, Self Token, FFN, Final LayerNorm, and Initial Embedding. The attribution scores of these seven parts sum to the token’s final probability, ensuring we capture the complete generation process.

To compute these attributions, we proposed a probe function similar with nostalgebraist (2020) that uses the model’s unembedding matrix to read out the next-token probability directly from an intermediate residual-stream state. Concretely, for each component on the residual stream, we define its contribution as the change in the probed next-token probability before versus after applying that component. In this way, we can compute the contribution from Initial Embedding, attention block, FFN, and Final LayerNorm. For the attention block, we further distribute its contribution to Query, RAG, Past and Self Token according to their attention weights.

However, these attention scores are insufficient for detection. A high reliance on internal parametric knowledge (FFNs) does not necessarily imply a hallucination. This pattern is expected for function words like "the" or "of". Yet, it becomes highly sus-

picious when found in named entities. Therefore, treating all tokens equally fails to capture these critical distinctions.

To capture this distinction, we aggregate the attribution scores using Part-of-Speech (POS) tags. We employ POS tags to capture comprehensive syntactic patterns. Unlike Named Entity Recognition (NER), which is limited to specific entity types, POS tagging covers all tokens (including critical categories like Numerals and Adpositions) and maintains high computational efficiency.

Figure 1 illustrates how TPA turns a single response into detection features: we first compute token-level source attributions, then aggregate them by POS tags. The second step is critical since hallucination signals vary across distinct parts of speech. For example, low RAG contribution on nouns or high LN contribution on numerals is often indicative of hallucination. These patterns are harder to capture if we only use raw token-level attribution scores without POS information.

Our main contributions are:

1. We propose TPA, a novel framework that mathematically attributes each token’s probability to seven distinct attribution sources. This provides a comprehensive mechanistic view of the token generation process.   
2. We introduce a syntax-aware aggregation mechanism. By quantifying how attribution sources drive distinct parts of speech, this approach enables the detector to pinpoint anomalies in specific entities while ignoring benign grammatical patterns.   
3. Extensive experiments demonstrate that TPA achieves state-of-the-art performance. Our framework also offers transparent interpretability, automatically uncovering novel mechanistic signatures, such as anomalous LayerNorm contributions, that extend beyond the traditional FFN-RAG binary conflict.

# 2 Related Work

Uncertainty and Proxy Metrics. Approaches in this category estimate hallucination via output inconsistency or proxy signals. Some methods quantify uncertainty using model ensembles (Malinin and Gales, 2021) or by measuring self-consistency across multiple sampled generations from a single model (Manakul et al., 2023). Others utilize

lightweight proxy scores computable from a single generation pass, such as energy-based OOD proxy scores (Liu et al., 2020), embedding-based distance scores for conditional LMs (Ren et al., 2023), and token-level uncertainty heuristics for hallucination detection (Lee et al., 2024; Zhang et al., 2023). While efficient, these scores provide indirect signals (e.g., confidence or distribution shift) and therefore may be imperfect indicators of factual correctness.

LLM-based Evaluation. External LLMs are also employed as verifiers. In RAG settings, outputs can be checked against retrieved evidence (Friel and Sanyal, 2023) or through claim extraction and reference-based verification (Hu et al., 2024), and LLM-as-a-judge baselines are often instantiated using curated prompts (Niu et al., 2024). Automated evaluation suites (Es et al., 2024; True-Lens, 2024) have also been developed. Other strategies include cross-examination to expose inconsistencies (Cohen et al., 2023; Yehuda et al., 2024) or fine-tuning detectors for span-level localization (Su et al., 2025). However, many of these approaches require extra LLM calls or multi-step verification.

Probing Internal Activations. Recent work extracts factuality signals from internal representations, e.g., linear truthful directions or inferencetime shifts (Burns et al., 2022; Li et al., 2023), and probe-based detectors trained on hidden states (Azaria and Mitchell, 2023; Han et al., 2024). Related studies show internal states remain predictive for hallucination detection (Chen et al., 2024). Beyond detection, mechanistic analyses conflicts between FFN and RAG context (Sun et al., 2025), and lightweight indicators use attention-head norms (Ho et al., 2025). Active approaches steer or edit activations (Park et al., 2025; Li et al., 2023), or adjust decoding probabilities for diagnosis (Chen et al., 2025). In contrast, we decompose the final token probability into fine-grained sources.

# 3 Methodology

As illustrated in Figure 2, TPA operates in three stages and can be implemented with a fully parallel teacher-forced pass. Given the generated response sequence y of length $T$ , we can feed the entire sequence into the model with standard causal masking to extract hidden states and attention maps for all $T$ tokens in a single teacher-forced pass. This avoids autoregressive resampling while enabling

efficient attribution computation.

We first derive a complete decomposition of token probabilities (Sec. 3.2), then attribute attention contributions to specific attribution sources (Sec. 3.3). Finally, we aggregate these scores to quantify how sources drive distinct parts of speech (Sec. 3.4). The pseudo-code and complexity analysis are provided in the Appendix. We report complexity instead of wall-clock time since the latter varies in different implementation hardware. To provide the theoretical basis for our method, we first formalize the transformer’s architecture.

# 3.1 Preliminaries: Transformer Architecture

# 3.1.1 Notations

We consider a standard decoder-only Transformer with $L$ layers. We denote the query tokens as $\mathbf { x } _ { \mathrm { q r y } }$ , the retrieved context tokens as $\mathbf { x } _ { \mathrm { r a g } }$ , and the generated response as $\mathbf { y } = ( y _ { 1 } , \dots , y _ { T } )$ , with prompt length $T _ { 0 } = | \mathbf { x } _ { \mathrm { q r y } } | + | \mathbf { x } _ { \mathrm { r a g } } |$ . We analyze generation at step $t \in \{ 1 , \ldots , T \}$ , where the model observes the prefix $\mathbf { s } _ { t } = [ \mathbf { x } _ { \mathrm { q r y } } , \mathbf { x } _ { \mathrm { r a g } } , y _ { 1 } , \dots , y _ { t - 1 } ]$ , and predicts the next token $y _ { t }$ . Let $n _ { t } = \left| \mathbf { s } _ { t } \right| = T _ { 0 } + t - 1$ denote the position index of the last token in $\mathbf { s } _ { t }$ (the token whose embedding is used to predict $y _ { t }$ ).

Unless stated otherwise, all hidden states and residual outputs (e.g., $\mathbf { h } ^ { ( l ) } , \dot { \mathbf { h } } _ { \mathrm { m i d } } ^ { ( l ) } )$ refer to the vector at the last position $n _ { t }$ , and we omit the explicit index $n _ { t }$ and the step index $t$ for clarity. We keep explicit indices only for attention weights (e.g., $\mathbf { A } _ { h } ^ { ( l ) } [ n _ { t } , k ] )$ . We use $d$ for the hidden dimension, $H$ for the number of attention heads, and $d _ { h } = d / H$ for the head dimension.

# 3.1.2 Residual Updates and Probing

The input tokens are mapped to continuous vectors via an embedding matrix ${ \mathbf W } _ { e } \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ and summed with positional embeddings. The initial state at the target position is $\mathbf { h } ^ { ( 0 ) } = \mathbf { W } _ { e } [ \mathbf { s } _ { t } [ n _ { t } ] ] +$ $\mathbf { p } _ { n _ { t } }$ , where $\mathbf { p } _ { n _ { t } }$ is the positional embedding at position $n _ { t }$ . We adopt the Pre-LN configuration. Crucially, each layer l updates the hidden state via additive residual connections:

$$
\mathbf {h} _ {\mathrm {m i d}} ^ {(l)} = \mathbf {h} ^ {(l - 1)} + \operatorname {A t t n} (\operatorname {L N} \left(\mathbf {h} ^ {(l - 1)}\right)) \tag {1}
$$

$$
\mathbf {h} ^ {(l)} = \mathbf {h} _ {\mathrm {m i d}} ^ {(l)} + \operatorname {F F N} \left(\operatorname {L N} \left(\mathbf {h} _ {\mathrm {m i d}} ^ {(l)}\right)\right) \tag {2}
$$

Here, $\mathrm { A t t n } ( \cdot )$ denotes the attention output vector at position $n _ { t }$ under causal masking. This structure implies that the final representation is the sum of the initial embedding and all subsequent layer updates. To quantify these updates, we define a

![](images/e60db9ef61b89e2793ab022c221386ef15d6d514a6262e3cc5126cb599ac5739.jpg)

![](images/c639fd1536c30128e86ed5fcd3332ff8c268b7768c54b8ed0758e1857afc24a8.jpg)  
Figure 2: Overview of the TPA framework. (1) Coarse-Grained Decomposition: Complete decomposition of token probability into four components (Section 3.2). (2) Fine-Grained Attribution: Mapping attention contributions to four input sources via head-specific weights (Section 3.3). (3) Syntax-Aware Feature Engineering: Aggregating these attributions by POS tags to construct the final detection features (Section 3.3.4).

Probe Function $\Phi ( \mathbf { h } , y )$ similar to the logit lens technique (nostalgebraist, 2020) that measures the hypothetical probability of the target token $y$ given any intermediate state vector h:

$$
\Phi (\mathbf {h}, y) = \left[ \operatorname {S o f t m a x} \left(\mathbf {h} \mathbf {W} _ {U}\right) \right] _ {y} \tag {3}
$$

where $\mathbf { W } _ { U }$ is the unembedding matrix.

Guiding Question: Since the model is a stack of residual updates, can we mathematically decompose the final probability exactly into the sum of component contributions?

# 3.2 Coarse-Grained Decomposition

We answer the preceding question affirmatively by leveraging the additive nature of the residual updates. Based on the probe function $\Phi ( \mathbf { h } , y )$ defined in Eq. (3), we isolate the probability contribution of each model component as the distinct change it induces in the probe output.

We define the baseline contribution from input static embeddings $( \Delta P _ { \mathrm { i n i t i a l } } )$ , the incremental gains from Attention and FFN blocks in layer l (∆ $( \Delta P _ { \mathrm { a t t } } ^ { ( l ) } , \Delta P _ { \mathrm { f f n } } ^ { ( l ) } )$ P (l), ∆P (l)), ffn and the adjustment from the final LayerNorm $( \Delta P _ { \mathrm { L N } } )$ as follows:

$$
\Delta P _ {\text {i n i t i a l}} (y) = \Phi \left(\mathbf {h} ^ {(0)}, y\right) \tag {4}
$$

$$
\Delta P _ {\mathrm {a t t}} ^ {(l)} = \Phi \left(\mathbf {h} _ {\mathrm {m i d}} ^ {(l)}, y\right) - \Phi \left(\mathbf {h} ^ {(l - 1)}, y\right) \tag {5}
$$

$$
\Delta P _ {\mathrm {f f n}} ^ {(l)} = \Phi (\mathbf {h} ^ {(l)}, y) - \Phi (\mathbf {h} _ {\mathrm {m i d}} ^ {(l)}, y) \tag {6}
$$

$$
\Delta P _ {\mathrm {L N}} = P _ {\text {f i n a l}} (y) - \Phi (\mathbf {h} ^ {(L)}, y) \tag {7}
$$

We define $P _ { \mathrm { f i n a l } } ( y )$ as the model output probability after applying the final LayerNorm at position $n _ { t }$ .

By summing these differences, we derive the complete decomposition of the model’s output.

Theorem 1 (Complete Probability Decomposition). The final probability for a target token y is exactly the sum of the contribution from the initial embedding, the cumulative contributions from Attention

and FFN blocks across all L layers, and the adjustment from the final LayerNorm:

$$
\begin{array}{l} P _ {f i n a l} (y) = \Delta P _ {i n i t i a l} (y) + \Delta P _ {L N} \\ + \sum_ {l = 1} ^ {L} \left(\Delta P _ {a t t} ^ {(l)} + \Delta P _ {f f n} ^ {(l)}\right) \tag {8} \\ \end{array}
$$

Proof. See Appendix.

![](images/b1dd05dc4d7d46214391abc7d94d694e1d91beb5e1503d87a5c82782465fb494.jpg)

The Guiding Question: While Eq. (8) quantifies how much the model components contribute to the prediction probability, it treats the term ∆P (l)att $\Delta P _ { \mathrm { a t t } } ^ { ( l ) }$ as a black box. To effectively detect hallucinations, we must identify where this attention is focused.

# 3.3 Fine-Grained Attribution

To identify the focus of attention, we must decompose the attention contribution ∆P (l)att $\Delta P _ { \mathrm { a t t } } ^ { ( l ) }$ into contributions from individual attention heads.

# 3.3.1 The Challenge: Non-Linearity

Standard Multi-Head Attention concatenates the outputs of $H$ independent heads and projects them via an output matrix $\mathbf { W } _ { O } ^ { ( l ) }$ . Mathematically, by partitioning $\mathbf { W } _ { O } ^ { ( l ) }$ into head-specific sub-matrices, this operation is strictly equivalent to the sum of projected head outputs:

$$
\mathbf {h} _ {\mathrm {a t t}} ^ {(l)} = \sum_ {h = 1} ^ {H} \underbrace {\left(\mathbf {A} _ {h} ^ {(l)} [ n _ {t} , : ] \mathbf {V} _ {h} ^ {(l)}\right)} _ {\mathbf {o} _ {h} ^ {(l)}} \mathbf {W} _ {O} ^ {(l, h)} \tag {9}
$$

where $\mathbf { o } _ { h } ^ { ( l ) }$ o is the head output vector at the target position, derived from the attention row ${ \bf A } _ { h } ^ { ( l ) } [ n _ { t } , : ]$ and the value matrix (l) $\mathbf { V } _ { h } ^ { ( l ) }$

Eq. (9) establishes that the attention output is linear with respect to individual heads in the hidden state space. However, our goal is to attribute the probability change ∆P (l)att t $\Delta P _ { \mathrm { a t t } } ^ { ( l ) }$ o each head $h$ . Since the probe function $\Phi ( \cdot )$ employs a non-linear Softmax operation, the sum of probability changes calculated by probing individual heads does not equal the attention block contribution. This inequality prevents us from calculating head contributions by simply probing each head individually, motivating our shift to the logit space.

# 3.3.2 Logit-Based Apportionment

To bypass the non-linearity of the Softmax, we analyze contributions in the logit space. Let $\Delta z _ { h , y } ^ { ( l ) }$ zh,y denote the scalar contribution of head $h$ to the logit

of the target token $y$ . This is calculated as the dot product between the projected head output and the target token’s unembedding vector ${ \bf w } _ { U , y }$ :

$$
\Delta z _ {h, y} ^ {(l)} = \left(\mathbf {o} _ {h} ^ {(l)} \mathbf {W} _ {O} ^ {(l, h)}\right) \cdot \mathbf {w} _ {U, y} \tag {10}
$$

We then apportion the complete probability contribution ∆P (l) $\Delta \hat { P } _ { \mathrm { a t t } } ^ { ( l ) }$ (derived in Section 3.2) to each head $h$ proportional to its exponential logit contribution:

$$
\Delta P _ {h} ^ {(l)} = \Delta P _ {\mathrm {a t t}} ^ {(l)} \cdot \frac {\exp \left(\Delta z _ {h , y} ^ {(l)}\right)}{\sum_ {j = 1} ^ {H} \exp \left(\Delta z _ {j , y} ^ {(l)}\right)} \tag {11}
$$

# 3.3.3 Theoretical Justification

We ground the logit-based apportionment using a first-order Taylor expansion similar to (Montavon et al., 2019). This approximates how logit changes affect the final probability.

Proposition 1 (Linear Decomposition). The total attention contribution ∆P (l)att $\Delta P _ { a t t } ^ { ( l ) }$ is approximated by the sum of head logits $\Delta z _ { h , y } ^ { ( l ) }$ s caled by a gradient $\mathcal { G } ^ { ( l ) }$ :

$$
\Delta P _ {a t t} ^ {(l)} \approx \mathcal {G} ^ {(l)} \sum_ {h = 1} ^ {H} \Delta z _ {h, y} ^ {(l)} + \mathcal {E} \tag {12}
$$

Proof. See Appendix.

While Proposition 1 implies a linear relationship, direct attribution is unstable when head logits sum to zero. To resolve this, we employ the Softmax normalization in Eq. (11). This ensures numerical stability and constrains the sum of head scores to exactly match the layer total ∆P (l)att . $\Delta P _ { \mathrm { a t t } } ^ { ( l ) }$ · Thus, it preserves the conservation principle in Theorem 1.

Regarding the approximation error $\mathcal { E }$ , we rely on the first-order term for efficiency. This is effective because hallucinations are often high-confidence (Kadavath et al., 2022), which suppresses higherorder terms. For low-confidence scenarios, prior work identifies low probability itself as a strong hallucination signal (Guerreiro et al., 2023). Our framework naturally captures this feature because the attribution scores sum exactly to the final output probability (Theorem 1). Therefore, TPA inherently incorporates this critical probability signal and effectively detects such hallucinations.

# 3.3.4 Source Mapping and The 7-Source Split

Having isolated the head contribution ∆P (l)h , $\Delta P _ { h } ^ { ( l ) }$ · we can now answer the guiding question by tracing attention back to input tokens using the attention

matrix ${ \bf A } _ { h } ^ { \left( l \right) }$ . We categorize inputs into four source types: $\mathcal { S } = \{ \sf { Q r y } , \sf { R A G } , \sf { P a s t } , \sf { S e l f } \}$ . Let $T _ { q } =$ $\left| \mathbf { x } _ { \mathrm { q r y } } \right|$ and $T _ { r } = | \mathbf { x } _ { \mathrm { r a g } } |$ , so $T _ { 0 } = T _ { q } + T _ { r }$ and $n _ { t } =$ $T _ { 0 } + t - 1$ . We partition the causal attention range $[ 1 , n _ { t } ]$ into four disjoint index sets: $\mathcal { T } _ { \mathrm { Q r y } } = [ 1 , T _ { q } ]$ , $\mathcal { T } _ { \mathtt { R A G } } = [ T _ { q } + 1 , T _ { 0 } ]$ , $\mathcal { T } _ { \mathsf { P a s t } } = [ T _ { 0 } + 1 , n _ { t } - 1 ]$ , and $\mathcal { T } _ { \mathsf { S e l f } } \ = \ \{ n _ { t } \}$ . For a source type $S$ containing token indices $\mathcal { T } _ { S }$ , the aggregated contribution is:

$$
\Delta P _ {S} ^ {(l)} = \sum_ {h = 1} ^ {H} \left(\Delta P _ {h} ^ {(l)} \cdot \sum_ {k \in \mathcal {I} _ {S}} \mathbf {A} _ {h} ^ {(l)} [ n _ {t}, k ]\right) \tag {13}
$$

By aggregating these components, we achieve a complete partition of the final probability $P _ { \mathrm { f i n a l } } ( y )$ into seven distinct sources:

$$
\begin{array}{l} P _ {\text {f i n a l}} (y) = \Delta P _ {\text {i n i t i a l}} (y) + \Delta P _ {\mathrm {L N}} \\ + \sum_ {l = 1} ^ {L} \left(\Delta P _ {\mathrm {f f n}} ^ {(l)} + \sum_ {S \in \mathcal {S}} \Delta P _ {S} ^ {(l)}\right) \tag {14} \\ \end{array}
$$

The Guiding Question: We have now derived a 7-dimensional attribution vector for every token. However, raw attribution scores lack context: a high FFN contribution might be normal for a function word but suspicious for a proper noun. How to contextualize these scores with syntactic priors?

# 3.4 Syntax-Aware Feature Engineering

To resolve this ambiguity, we employ Part-of-Speech (POS) tagging as a lightweight syntactic prior. Specifically, we assign a POS tag by Spacy (Honnibal et al., 2020) to each generated token and aggregate the attribution scores for each grammatical category. By profiling which attribution sources (e.g., RAG) the LLM relies on for different parts of speech, we detect hallucination effectively.

# 3.4.1 Tag Propagation Strategy

A mismatch problem arises because LLMs may split a single word into multiple tokens while standard POS taggers process whole words. We resolve this granularity issue via tag propagation: generated sub-word tokens inherit the POS tag of their parent word. For instance, if the noun "modification" is tokenized into [modi, fication], both subtokens are assigned the NOUN tag.

# 3.4.2 Aggregation

We first define the attribution vector $\mathbf { v } _ { t } \in \mathbb { R } ^ { 7 }$ for each token $y _ { t }$ as the concatenation of its 7 source components derived in Section 3.3.4. Then we

compute the mean attribution for each POS tag $\tau$ :

$$
\bar {\mathbf {v}} _ {\tau} = \frac {\sum_ {t : \mathrm {P O S} (y _ {t}) = \tau} \mathbf {v} _ {t}}{| \{t \mid \mathrm {P O S} (y _ {t}) = \tau \} |} \tag {15}
$$

The final feature vector $\mathbf { f } \in \mathbb { R } ^ { 7 \times | \mathrm { P O S } | }$ is the concatenation of these POS-specific vectors. This feature combines source attribution with linguistic structure, forming a basis for hallucination detection.

# 4 Experiments

# 4.1 Experimental Setup

We treat hallucination detection as a supervised binary classification task. We employ an ensemble of 5 XGBoost (Chen and Guestrin, 2016) classifiers initialized with different random seeds. The input is a 126-dimensional syntax-aware feature vector, constructed by aggregating the 7-source attribution scores across 18 universal POS tags (e.g., NOUN, VERB) defined by SpaCy (Honnibal et al., 2020). See more implementation details in Appendix.

# 4.2 Dataset and Baselines

To ensure a fair comparison, we utilize the public RAG hallucination benchmark established by Sun et al. (2025). This benchmark consists of the Dolly (AC) and RAGTruth datasets. The former includes responses from Llama2 (7B/13B) and Llama3 (8B), while the latter covers these models in addition to Mistral-7B. Implementation details are provided in Appendix. We compare our method against representative approaches from three categories introduced in Section 2. The introduction of baselines is provided in Appendix. For Mistral-7B, we only compare against TSV and Novo as prior baselines (e.g., ReDeEP) did not report results on this architecture and Dolly does not include Mistral data.

# 4.3 Comparison with Baselines

We evaluate TPA against baselines on RAGTruth (Llama2, Llama3, Mistral) and Dolly (AC) (Llama2, Llama3). As shown in Table 1 and Table 2, TPA is competitive across benchmarks.

On RAGTruth, TPA achieves statistically significant Rank-1 results $( p < 0 . 0 5 )$ on Llama2-7B and Llama2-13B for both F1 and AUC. The largest improvement appears on Mistral-7B, where TPA reaches 0.8702 F1, outperforming Novo by $7 \%$ , indicating good transfer to newer architectures with sliding-window attention. On Llama3-8B, TPA

Table 1: Results on RAGTruth and Dolly (AC). TPA results are averaged over 5 random seeds. The dagger symbol † indicates statistically significant improvement $( p < 0 . 0 5 )$ over the strongest baseline. Bold values indicate the best performance and underlined values indicate the second-best. Full results in Appendix.   

<table><tr><td rowspan="2">Method</td><td colspan="9">RAGTruth</td></tr><tr><td colspan="3">LLaMA2-7B</td><td colspan="3">LLaMA2-13B</td><td colspan="3">LLaMA3-8B</td></tr><tr><td>Metric</td><td>AUC</td><td>Recall</td><td>F1</td><td>AUC</td><td>Recall</td><td>F1</td><td>AUC</td><td>Recall</td><td>F1</td></tr><tr><td>ChainPoll (Friel and Sanyal, 2023)</td><td>0.6738</td><td>0.7832</td><td>0.7066</td><td>0.7414</td><td>0.7874</td><td>0.7342</td><td>0.6687</td><td>0.4486</td><td>0.5813</td></tr><tr><td>RAGAS (Es et al., 2024)</td><td>0.7290</td><td>0.6327</td><td>0.6667</td><td>0.7541</td><td>0.6763</td><td>0.6747</td><td>0.6776</td><td>0.3909</td><td>0.5094</td></tr><tr><td>Trulens (TrueLens, 2024)</td><td>0.6510</td><td>0.6814</td><td>0.6567</td><td>0.7073</td><td>0.7729</td><td>0.6867</td><td>0.6464</td><td>0.3909</td><td>0.5053</td></tr><tr><td>RefCheck (Hu et al., 2024)</td><td>0.6912</td><td>0.6280</td><td>0.6736</td><td>0.7857</td><td>0.6800</td><td>0.7023</td><td>0.6014</td><td>0.3580</td><td>0.4628</td></tr><tr><td>EigenScore (Chen et al., 2024)</td><td>0.6045</td><td>0.7469</td><td>0.6682</td><td>0.6640</td><td>0.6715</td><td>0.6637</td><td>0.6497</td><td>0.7078</td><td>0.6745</td></tr><tr><td>SEP (Han et al., 2024)</td><td>0.7143</td><td>0.7477</td><td>0.6627</td><td>0.8089</td><td>0.6580</td><td>0.7159</td><td>0.7004</td><td>0.7333</td><td>0.6915</td></tr><tr><td>ITI (Li et al., 2023)</td><td>0.7161</td><td>0.5416</td><td>0.6745</td><td>0.8051</td><td>0.5519</td><td>0.6838</td><td>0.6534</td><td>0.6850</td><td>0.6933</td></tr><tr><td>ReDeEP (Sun et al., 2025)</td><td>0.7458</td><td>0.8097</td><td>0.7190</td><td>0.8244</td><td>0.7198</td><td>0.7587</td><td>0.7285</td><td>0.7819</td><td>0.6947</td></tr><tr><td>TSV (Park et al., 2025)</td><td>0.6609</td><td>0.5526</td><td>0.6632</td><td>0.8123</td><td>0.8068</td><td>0.6987</td><td>0.7769</td><td>0.5546</td><td>0.6442</td></tr><tr><td>Novo (Ho et al., 2025)</td><td>0.7608</td><td>0.8274</td><td>0.7057</td><td>0.8506</td><td>0.7826</td><td>0.7733</td><td>0.8258</td><td>0.7737</td><td>0.7801</td></tr><tr><td>TPA</td><td>0.7873†</td><td>0.8328</td><td>0.7238†</td><td>0.8681†</td><td>0.7913</td><td>0.7975†</td><td>0.8211</td><td>0.7860</td><td>0.7843</td></tr><tr><td rowspan="2">Method</td><td colspan="9">Dolly (AC)</td></tr><tr><td colspan="3">LLaMA2-7B</td><td colspan="3">LLaMA2-13B</td><td colspan="3">LLaMA3-8B</td></tr><tr><td>Metric</td><td>AUC</td><td>Recall</td><td>F1</td><td>AUC</td><td>Recall</td><td>F1</td><td>AUC</td><td>Recall</td><td>F1</td></tr><tr><td>ChainPoll (Friel and Sanyal, 2023)</td><td>0.3502</td><td>0.4138</td><td>0.5581</td><td>0.4758</td><td>0.4364</td><td>0.6000</td><td>0.2691</td><td>0.3415</td><td>0.4516</td></tr><tr><td>RAGAS (Es et al., 2024)</td><td>0.2877</td><td>0.5345</td><td>0.6392</td><td>0.2840</td><td>0.4182</td><td>0.5476</td><td>0.3628</td><td>0.8000</td><td>0.5246</td></tr><tr><td>Trulens (TrueLens, 2024)</td><td>0.3198</td><td>0.5517</td><td>0.6667</td><td>0.2565</td><td>0.3818</td><td>0.4941</td><td>0.3352</td><td>0.3659</td><td>0.5172</td></tr><tr><td>RefCheck (Hu et al., 2024)</td><td>0.2494</td><td>0.3966</td><td>0.5412</td><td>0.2869</td><td>0.2545</td><td>0.3944</td><td>-0.0089</td><td>0.1951</td><td>0.2759</td></tr><tr><td>EigenScore (Chen et al., 2024)</td><td>0.2428</td><td>0.7500</td><td>0.7241</td><td>0.2948</td><td>0.8181</td><td>0.7200</td><td>0.2065</td><td>0.7142</td><td>0.5952</td></tr><tr><td>SEP (Han et al., 2024)</td><td>0.2605</td><td>0.6216</td><td>0.7023</td><td>0.2823</td><td>0.6545</td><td>0.6923</td><td>0.0639</td><td>0.6829</td><td>0.6829</td></tr><tr><td>ITI (Li et al., 2023)</td><td>0.0442</td><td>0.5816</td><td>0.6281</td><td>0.0646</td><td>0.5385</td><td>0.6712</td><td>0.0024</td><td>0.3091</td><td>0.4250</td></tr><tr><td>ReDeEP (Sun et al., 2025)</td><td>0.5136</td><td>0.8245</td><td>0.7833</td><td>0.5842</td><td>0.8518</td><td>0.7603</td><td>0.3652</td><td>0.8392</td><td>0.7100</td></tr><tr><td>TSV (Park et al., 2025)</td><td>0.7454</td><td>0.8728</td><td>0.7684</td><td>0.7552</td><td>0.5952</td><td>0.6043</td><td>0.7347</td><td>0.6467</td><td>0.6695</td></tr><tr><td>Novo (Ho et al., 2025)</td><td>0.6423</td><td>0.8070</td><td>0.7244</td><td>0.6909</td><td>0.7222</td><td>0.6903</td><td>0.7418</td><td>0.5854</td><td>0.6316</td></tr><tr><td>TPA</td><td>0.7134</td><td>0.7897</td><td>0.7527</td><td>0.8159†</td><td>0.9741†</td><td>0.8075†</td><td>0.7608†</td><td>0.6561</td><td>0.7529†</td></tr></table>

Table 2: Performance comparison on the RAGTruth dataset using the Mistral-7B model. Bold indicates the best performance, and † indicates statistically significant improvement over the strongest baseline $( p < 0 . 0 5 )$ . The TPA results are averaged over 5 runs.   

<table><tr><td>Method</td><td>F1</td><td>AUC</td><td>Recall</td></tr><tr><td>TSV (Park et al., 2025)</td><td>0.6764</td><td>0.7972</td><td>0.5538</td></tr><tr><td>Novo (Ho et al., 2025)</td><td>0.8000</td><td>0.8419</td><td>0.8765</td></tr><tr><td>TPA</td><td>0.8702†</td><td>0.9096†</td><td>0.9200†</td></tr></table>

ranks first in F1 and Recall but is statistically comparable to the strongest baselines, suggesting a smaller margin on newer models.

On Dolly (AC), results show a scaling trend. TPA trails baselines (e.g., ReDeEP) on Llama2- 7B, but becomes stronger as model capacity increases: it secures significant Rank-1 performance on Llama2-13B and the best F1 on Llama3-8B.

# 5 Ablation Study Analysis

We conduct an ablation study on RAGTruth to validate TPA (Figure 4). The full method generally achieves the best performance. Removing core components like RAG or FFN consistently degrades accuracy (e.g., a $3 . 0 1 \%$ F1 drop on Llama-2-7B), confirming the importance of the retrievalparameter conflict. Crucially, previously overlooked sources are distinctively vital. For instance, removing LN causes a sharp $5 . 8 3 \%$ drop on Llama-3-8B. While excluding specific components yields marginal gains in several cases (e.g., Self on Mistral-7B), we retain the complete feature set to maintain a unified framework robust across diverse architectures. We exclude Dolly as its small sample size makes fine-grained feature evaluation unstable.

# 6 Interpretability Analysis

We apply SHAP analysis to classifiers trained on RAGTruth to validate our design principles. Results for Llama2 are shown in Figure 3, while re-

![](images/fc1f80fc0fd41bef66850555e3e622dfe5b1e57718d0f30f600ff63cbed28d78.jpg)  
(a) Llama2-7B

![](images/e721080c3aaac6b7479523cc3b6e0d8935225ad4bb4466180dbc624ad1166cf1.jpg)  
(b) Llama2-13B

![](images/b57779631ae621f868027c122c596c9a131f3db5146f5da5406ade7cbe4e0e44.jpg)  
Figure 3: SHAP summary plots illustrating the decision logic. We visualize the top-10 features for classifiers trained on the RAGTruth subsets corresponding to Llama2-7B and Llama2-13B. Plots for Llama3-8B and Mistral-7B are provided in Appendix. The x-axis represents the SHAP value. Positive values indicate a push towards classifying the response as a Hallucination. The color represents the feature value (Red $=$ High attribution, Blue $=$ Low).   
Figure 4: F1 Score Drop by Removing Components.

sults for Llama3-8B and Mistral-7B are detailed in Appendix due to space constraints. We obtain three observations from this analysis.

Observation 1: Fine-grained attribution is necessary. Relying solely on the binary conflict between internal FFN knowledge and external RAG context is insufficient for robust detection. As shown in Figure 3, the classifier frequently depends on features derived from other components. For instance, the feature LN_NUM plays a decisive role in Llama2-7B. This pattern indicates that the specific contribution from the Final LayerNorm to Numeral tokens serves as a critical signal. Similarly, query-based features like QUERY_NOUN appear as top predictors in Mistral-7B. These findings confirm that accurate hallucination detection requires a complete decomposition of the generative process.

Observation 2: Syntactic aggregation captures model-specific grounding logic. Different architectures ground information via distinct syntactic structures. While RAG_NOUN dominates in Llama2 (Figure 3) and Mistral, Llama3-8B relies heavily on

relational structures, with RAG_ADP (Adpositions) ranking highest. TPA’s POS-based aggregation is thus essential to capture these diverse patterns.

Observation 3: Hallucination fingerprints vary across architectures. Our analysis shows that hallucination signals are model-specific rather than universal. A clear example is LN_NUM: high attribution is a strong hallucination signal in Llama2-7B, but this pattern reverses in Llama2-13B (Figure 3b), where higher values correlate with factuality (negative SHAP values). This reversal suggests that larger models may use LayerNorm differently to regulate output distributions, motivating a learnable, syntax-aware detector.

# 7 Conclusion and Future Work

We introduced TPA to attribute token probability into seven distinct sources. By combining these with syntax-aware features, our framework effectively detects RAG hallucinations and outperforms baselines. Our results show that hallucination signals vary across models. This confirms the need for a learnable approach rather than static heuristics.

Future work will proceed in two directions. First, we will extend TPA beyond token-level attribution to phrase- or span-level units so as to improve efficiency. Second, we will explore active mitigation: instead of only diagnosing hallucinations after generation, we will monitor source contributions online and intervene when risky patterns appear, e.g., abnormal reliance on FFN or LN.

# 8 Limitations

Our framework presents three limitations. First, it relies on white-box access to model internals, preventing application to closed-source APIs. Second, the decomposition incurs higher computational overhead than simple scalar probes. Third, our feature engineering depends on external linguistic tools (POS taggers), which may limit generalization to specialized domains like code generation where standard syntax is less defined.

# Acknowledgements

This work was supported by the Australian Research Council through the Laureate Fellow Project under Grant FL190100149.

# References

Amos Azaria and Tom Mitchell. 2023. The internal state of an llm knows when it’s lying. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 967–976.   
Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. 2022. Discovering latent knowledge in language models without supervision. arXiv preprint arXiv:2212.03827.   
Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. 2024. Inside: Llms’ internal states retain the power of hallucination detection.   
Tianqi Chen and Carlos Guestrin. 2016. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, pages 785– 794.   
Yuyan Chen, Zehao Li, Shuangjie You, Zhengyu Chen, Jingwen Chang, Yi Zhang, Weinan Dai, Qingpei Guo, and Yanghua Xiao. 2025. Attributive reasoning for hallucination diagnosis of large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 23660–23668.   
Roi Cohen, May Hamri, Mor Geva, and Amir Globerson. 2023. Lm vs lm: Detecting factual errors via cross examination. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12621–12640.   
Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. 2024. Ragas: Automated evaluation of retrieval augmented generation. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 150–158.

Robert Friel and Atindriyo Sanyal. 2023. Chainpoll: A high efficacy method for llm hallucination detection. arXiv preprint arXiv:2310.18344.   
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.   
Nuno M Guerreiro, Elena Voita, and André FT Martins. 2023. Looking for a needle in a haystack: A comprehensive study of hallucinations in neural machine translation. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 1059–1075.   
Jiatong Han, Jannik Kossen, Muhammed Razzak, Lisa Schut, Shreshth A Malik, and Yarin Gal. 2024. Semantic entropy probes: Robust and cheap hallucination detection in llms. In ICML 2024 Workshop on Foundation Models in the Wild.   
Zheng Yi Ho, Siyuan Liang, Sen Zhang, Yibing Zhan, and Dacheng Tao. 2025. Novo: Norm voting off hallucinations with attention heads in large language models. In The Thirteenth International Conference on Learning Representations.   
Matthew Honnibal, Ines Montani, Sofie Van Landeghem, Adriane Boyd, and 1 others. 2020. spacy: Industrial-strength natural language processing in python.   
Xiangkun Hu, Dongyu Ru, Lin Qiu, Qipeng Guo, Tianhang Zhang, Yang Xu, Yun Luo, Pengfei Liu, Yue Zhang, and Zheng Zhang. 2024. Refchecker: Reference-based fine-grained hallucination checker and benchmark for large language models. arXiv preprint arXiv:2405.14486.   
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and 1 others. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55.   
Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, and 1 others. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.   
Hakyung Lee, Keon-Hee Park, Hoyoon Byun, Jeyoon Yeom, Jihee Kim, Gyeong-Moon Park, and Kyungwoo Song. 2024. Ced: Comparing embedding differences for detecting out-of-distribution and hallucinated text. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 14866– 14882.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, and 1 others. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459– 9474.   
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inferencetime intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451–41530.   
Weitang Liu, Xiaoyun Wang, John Owens, and Yixuan Li. 2020. Energy-based out-of-distribution detection. Advances in neural information processing systems, 33:21464–21475.   
Andrey Malinin and Mark Gales. 2021. Uncertainty estimation in autoregressive structured prediction. In International Conference on Learning Representations.   
Potsawee Manakul, Adian Liusie, and Mark Gales. 2023. Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 9004– 9017.   
Grégoire Montavon, Alexander Binder, Sebastian Lapuschkin, Wojciech Samek, and Klaus-Robert Müller. 2019. Layer-wise relevance propagation: an overview. Explainable AI: interpreting, explaining and visualizing deep learning, pages 193–209.   
Cheng Niu, Yuanhao Wu, Juno Zhu, Siliang Xu, Kashun Shum, Randy Zhong, Juntong Song, and Tong Zhang. 2024. Ragtruth: A hallucination corpus for developing trustworthy retrieval-augmented language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10862–10878.   
nostalgebraist. 2020. interpreting gpt: the logit lens. LessWrong.   
Seongheon Park, Xuefeng Du, Min-Hsuan Yeh, Haobo Wang, and Yixuan Li. 2025. Steer LLM latents for hallucination detection. In Forty-second International Conference on Machine Learning.   
Jie Ren, Jiaming Luo, Yao Zhao, Kundan Krishna, Mohammad Saleh, Balaji Lakshminarayanan, and Peter J Liu. 2023. Out-of-distribution detection and selective generation for conditional language models. In The Eleventh International Conference on Learning Representations.   
Adi Simhi, Itay Itzhak, Fazl Barez, Gabriel Stanovsky, and Yonatan Belinkov. 2025. Trust me, i’m wrong: Llms hallucinate with certainty despite knowing the answer. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 14665– 14688.

Hsuan Su, Ting-Yao Hu, Hema Swetha Koppula, Kundan Krishna, Hadi Pouransari, Cheng-Yu Hsieh, Cem Koc, Joseph Yitan Cheng, Oncel Tuzel, and Raviteja Vemulapalli. 2025. Learning to reason for hallucination span detection. arXiv preprint arXiv:2510.02173.   
Zhongxiang Sun, Xiaoxue Zang, Kai Zheng, Jun Xu, Xiao Zhang, Weijie Yu, Yang Song, and Han Li. 2025. Redeep: Detecting hallucination in retrievalaugmented generation via mechanistic interpretability. In ICLR.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.   
TrueLens. 2024. Truelens: Evaluate and track llm applications.   
Yakir Yehuda, Itzik Malkiel, Oren Barkan, Jonathan Weill, Royi Ronen, and Noam Koenigstein. 2024. Interrogatellm: Zero-resource hallucination detection in llm-generated answers. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9333– 9347.   
Tianhang Zhang, Lin Qiu, Qipeng Guo, Cheng Deng, Yue Zhang, Zheng Zhang, Chenghu Zhou, Xinbing Wang, and Luoyi Fu. 2023. Enhancing uncertaintybased hallucination detection with stronger focus. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 915– 932.

# 9 Proof of Theorem 1

We expand the right-hand side (RHS) of Eq. 8 by substituting the definitions of each term.

First, consider the summation term. By substituting ∆P (l)att $\Delta P _ { \mathrm { a t t } } ^ { ( l ) }$ and ∆P (l), $\Delta P _ { \mathrm { f f n } } ^ { ( l ) }$ the intermediate probe value $\Phi ( \mathbf { h } _ { \mathrm { m i d } } ^ { ( l ) } )$ cancels out within each layer:

$$
\begin{array}{l} \sum_ {l = 1} ^ {L} \left(\Delta P _ {\mathrm {a t t}} ^ {(l)} + \Delta P _ {\mathrm {f f n}} ^ {(l)}\right) \\ = \sum_ {l = 1} ^ {L} \left(\left[ \Phi \left(\mathbf {h} _ {\mathrm {m i d}} ^ {(l)}\right) \right. \right. \\ \left. - \Phi \left(\mathbf {h} ^ {(l - 1)}\right) \right] + \left[ \Phi \left(\mathbf {h} ^ {(l)}\right) - \Phi \left(\mathbf {h} _ {\mathrm {m i d}} ^ {(l)}\right) \right]) \\ = \sum_ {l = 1} ^ {L} \left(\Phi \left(\mathbf {h} ^ {(l)}\right) - \Phi \left(\mathbf {h} ^ {(l - 1)}\right)\right) \\ \end{array}
$$

This summation forms a telescoping series where

adjacent terms cancel:

$$
\begin{array}{l} \sum_ {l = 1} ^ {L} (\Phi (\mathbf {h} ^ {(l)}) - \Phi (\mathbf {h} ^ {(l - 1)})) \\ = (\Phi (\mathbf {h} ^ {(1)}) - \Phi (\mathbf {h} ^ {(0)})) + \dots \\ + \left(\Phi \left(\mathbf {h} ^ {(L)}\right) - \Phi \left(\mathbf {h} ^ {(L - 1)}\right)\right) \\ = \Phi (\mathbf {h} ^ {(L)}) - \Phi (\mathbf {h} ^ {(0)}) \\ \end{array}
$$

Now, substituting this result, along with the definitions of $\Delta P _ { \mathrm { i n i t i a l } }$ and $\Delta P _ { \mathrm { L N } }$ , back into the full RHS expression:

$$
\begin{array}{l} \text {R H S} = \underbrace {\Phi (\mathbf {h} ^ {(0)})} _ {\Delta P _ {\text {i n i t i a l}}} + \underbrace {\left(P _ {\text {f i n a l}} - \Phi (\mathbf {h} ^ {(L)})\right)} _ {\Delta P _ {\text {L N}}} \\ + \underbrace {(\Phi (\mathbf {h} ^ {(L)}) - \Phi (\mathbf {h} ^ {(0)}))} _ {\text {S u m m a t i o n}} = P _ {\text {f i n a l}} \\ \end{array}
$$

The RHS simplifies exactly to $P _ { \mathrm { f i n a l } } ( y )$ , which completes the proof.

# 10 Proof of Proposition 1

Proof. We consider the $l$ -th layer of the Transformer. Let $\mathbf { h } ^ { ( l - 1 ) } \in \mathbb { R } ^ { d }$ be the input hidden state vector at the target position $n _ { t }$ . The Multi-Head Attention mechanism computes a residual update $\Delta \mathbf { h }$ by summing the outputs of $H$ heads:

$$
\Delta \mathbf {h} = \sum_ {h = 1} ^ {H} \mathbf {u} _ {h} \tag {16}
$$

where ${ \mathbf { u } } _ { h } \in \mathbb { R } ^ { d }$ is the projected output of the $h$ -th head.

The probe function $\Phi ( \mathbf { h } , y )$ computes the probability of the target token $y$ by projecting the hidden state onto the vocabulary logits $\mathbf { z } \in \mathbb { R } ^ { | \nu | }$ and applying the Softmax function:

$$
\Phi (\mathbf {h}, y) = \frac {\exp \left(\mathbf {z} _ {y}\right)}{\sum_ {v \in \mathcal {V}} \exp \left(\mathbf {z} _ {v}\right)}, \text {w h e r e} \mathbf {z} _ {v} = \mathbf {h} \cdot \mathbf {w} _ {U, v} \tag {17}
$$

Here, ${ \bf w } _ { U , v }$ is the unembedding vector for token $v$ from matrix $\mathbf { W } _ { U }$ . For brevity, let $p _ { y } ~ =$ $\Phi ( \mathbf { h } ^ { ( l - 1 ) } , y )$ denote the probability of the target token at the current state.

We approximate the change in probability, ∆P (l), $\Delta P _ { \mathrm { a t t } } ^ { ( l ) }$ using a first-order Taylor expansion of $\Phi$ with respect to h around $\mathbf { h } ^ { ( l - 1 ) }$ :

$$
\begin{array}{l} \Delta P _ {\mathrm {a t t}} ^ {(l)} = \Phi (\mathbf {h} ^ {(l - 1)} + \Delta \mathbf {h}, y) - \Phi (\mathbf {h} ^ {(l - 1)}, y) \\ \approx \nabla_ {\mathbf {h}} \Phi (\mathbf {h} ^ {(l - 1)}, y) ^ {\top} \cdot \Delta \mathbf {h} \\ = \sum_ {h = 1} ^ {H} \left(\nabla_ {\mathbf {h}} \Phi \left(\mathbf {h} ^ {(l - 1)}, y\right) ^ {\top} \cdot \mathbf {u} _ {h}\right) \tag {18} \\ \end{array}
$$

To compute the gradient $\nabla _ { \mathbf { h } } \Phi$ , we apply the chain rule through the logits $z _ { v }$ . The partial derivative of the Softmax output $p _ { y }$ with respect to any logit $z _ { v }$ is given by $p _ { y } ( \delta _ { y v } - p _ { v } )$ , where $\delta$ is the Kronecker delta. The gradient of the logit $z _ { v }$ with respect to h is simply ${ \bf w } _ { U , v }$ . Thus:

$$
\begin{array}{l} \nabla_ {\mathbf {h}} \Phi = \sum_ {v \in \mathcal {V}} \frac {\partial \Phi}{\partial z _ {v}} \frac {\partial z _ {v}}{\partial \mathbf {h}} \\ = \sum_ {v \in \mathcal {V}} p _ {y} \left(\delta_ {y v} - p _ {v}\right) \mathbf {w} _ {U, v} \tag {19} \\ = p _ {y} (1 - p _ {y}) \mathbf {w} _ {U, y} - \sum_ {v \neq y} p _ {y} p _ {v} \mathbf {w} _ {U, v} \\ \end{array}
$$

Substituting this gradient back into Eq. (18) for a specific head contribution term (denoted as ${ \mathrm { T e r m } } _ { h }$ ):

$$
\begin{array}{l} \mathrm {T e r m} _ {h} = \nabla_ {\mathbf {h}} \Phi^ {\top} \cdot \mathbf {u} _ {h} \\ = \underbrace {p _ {y} \left(1 - p _ {y}\right)} _ {\mathcal {G} ^ {(l)}} \left(\mathbf {w} _ {U, y} ^ {\top} \cdot \mathbf {u} _ {h}\right) - \tag {20} \\ \underbrace{\sum_{v\neq y}p_{y}p_{v}(\mathbf{w}_{U,v}^{\top}\cdot\mathbf{u}_{h})}_{\mathcal{E}_{h}} \\ \end{array}
$$

We observe that the dot product w⊤U,y · uh is strictly $\mathbf { w } _ { U , y } ^ { \top } \cdot \mathbf { u } _ { h }$ equivalent to the scalar logit contribution $\Delta z _ { h , y } ^ { ( l ) }$ defined in Eq. 10. The factor $\mathcal { G } ^ { ( l ) } = p _ { y } ( 1 - p _ { y } )$ represents the gradient common to all heads, depending only on the layer input $\mathbf { h } ^ { ( l - 1 ) }$ . Therefore, ear term the contribution of head $\mathcal { G } ^ { ( l ) } \cdot \Delta z _ { h , y } ^ { ( l ) }$ , subject to the off-target error $h$ is dominated by the linterm $\mathcal { E } _ { h }$ . □

# 11 Implementation Details

Environment and Models. All experiments were conducted on a computational node equipped with a single NVIDIA A100 (40GB) GPU and 200GB of RAM. Our software stack uses CUDA 12.8, PyTorch 3.10, and HuggingFace Transformers 4.56.1. We evaluate our framework using three Large Language Models: Llama2-7b-chat, Llama2- 13b-chat (Touvron et al., 2023), and Llama3-8binstruct (Grattafiori et al., 2024).

Due to GPU memory constraints (40GB), we implement TPA using a sequential prefix-replay procedure (token-by-token), rather than a fully parallel teacher-forced pass. On our hardware, generating the full TPA feature vector for one response takes approximately 20 seconds on average. Since each

Algorithm 1 TPA Part I: Token Probability Attribution (Teacher-Forced Pass / sequential process)  
Require: Transformer $\mathcal{M}$ , Query Tokens $\mathbf{x}_{\mathrm{qry}}$ , Retrieved Context Tokens $\mathbf{x}_{\mathrm{rag}}$ Ensure: Sequence of 7-source attribution vectors $\mathcal{V} = (\mathbf{v}_1, \ldots, \mathbf{v}_T)$ 1: Initialize contexts $\{\mathbf{C}_t\}_{t=1}^T$ and attribution storage $\mathcal{V} \gets \emptyset$ 2: $\mathbf{C}_1 \gets [\mathbf{x}_{\mathrm{qry}}, \mathbf{x}_{\mathrm{rag}}]$ 3: for position $t = 2 \ldots T$ do  
4: $\mathbf{C}_t \gets [\mathbf{C}_{t-1}, y_{t-1}]$ 5: end for  
6: for position $t = 1 \ldots T$ do  
7: 1. Forward Pass & State Caching  
8: Run the model on $\mathbf{C}_t$ and cache states at the last position $n_t = |\mathbf{C}_t|$ .  
9: Cache $\mathbf{h}^{(0)}, \mathbf{h}_{\mathrm{mid}}^{(l)}, \mathbf{h}^{(l)}$ , and Attention Maps $\mathbf{A}^{(l)}$ for all layers $l \in [1, L]$ .  
10: 2. Coarse Decomposition (Residual Stream)  
11: Initialize $\mathbf{v}_t \in \mathbb{R}^7$ with zeros.  
12: $\mathbf{v}_t[\mathrm{Init}] \gets \mathrm{Probe}(\mathbf{h}^{(0)}, y_t)$ {Contribution from initial embedding}  
13: $\mathbf{v}_t[\mathrm{LN}] \gets P_{\mathrm{final}}(y_t) - \mathrm{Probe}(\mathbf{h}^{(L)}, y_t)$ {Contribution from final LayerNorm}  
14: 3. Layer-wise Attribution  
15: for layer $l = 1$ to $L$ do  
16: $\mathbf{v}_t[\mathrm{FFN}] += \mathrm{Probe}(\mathbf{h}^{(l)}, y_t) - \mathrm{Probe}(\mathbf{h}_{\mathrm{mid}}^{(l)}, y_t)$ 17: $\Delta P_{\mathrm{att}} \gets \mathrm{Probe}(\mathbf{h}_{\mathrm{mid}}^{(l)}, y_t) - \mathrm{Probe}(\mathbf{h}^{(l-1)}, y_t)$ 18: 4. Fine-Grained Decomposition (Head & Source)  
19: for head $h = 1$ to $H$ do  
20: Let $\mathbf{o}_h$ be the output vector of head $h$ .  
21: Compute logit update: $\Delta z_h \gets (\mathbf{o}_h \mathbf{W}_O^{(l,h)}) \cdot \mathbf{w}_{U,y_t}$ 22: Compute ratio $\omega_h \gets \frac{\exp(\Delta z_h)}{\sum_j \exp(\Delta z_j)}$ {Logit-based apportionment}  
23: $\Delta P_h \gets \Delta P_{\mathrm{att}} \times \omega_h$ 24: for source $S \in \{\mathrm{Qry}, \mathrm{RAG}, \mathrm{Past}, \mathrm{Self}\}$ do  
25: Sum attention weights on indices of $S$ : $a_{h,S} \gets \sum_{k \in I_S} A_h^{(l)}[n_t, k]$ 26: $\mathbf{v}_t[S] += \Delta P_h \cdot a_{h,S}$ 27: end for  
28: end for  
29: end for  
30: Store $\mathbf{v}_t$ in attribution matrix $\mathcal{V}$ .  
31: end for  
32: return $\mathcal{V}$ , Generated Tokens y

dataset contains fewer than 3,000 samples, the total feature-extraction cost is on the order of ${ \sim } 1 7$ GPU-hours per dataset (per evaluated LLM).

Hallucination detection is performed using an ensemble of five XGBoost classifiers; inference is typically well below one second per response, and the total classification cost per dataset is only a few CPU-hours, negligible compared to feature extraction.

For POS tagging, we use spaCy with en_core_web_sm and disable NER.

Feature Extraction and Classifier. For each response, we extract the 7-dimensional attribution

vector for every token and aggregate them based on 18 universal POS tags (e.g., NOUN, VERB) defined by the SpaCy library (Honnibal et al., 2020). This results in a fixed-size feature vector $7 \times 1 8 = 1 2 6$ dimensions) for each sample.

Training and Evaluation Protocols. To ensure fair comparison and statistical robustness, we tailor our training strategies to the data availability of each dataset. We implement strict data isolation to prevent leakage. Crucially, to mitigate the variance inherent in small-data scenarios, we adopt a Multi-Seed Ensemble Strategy. For every experiment, we repeat the entire evaluation process using 5 distinct

Algorithm 2 TPA Part II: Syntax-Aware Feature Aggregation with Sub-word Tag Propagation   
Require: Generated Tokens $\mathbf{y} = (y_{1},\dots ,y_{T})$ , Attribution Vectors $\nu = (\mathbf{v}_1,\dots ,\mathbf{v}_T)$ Ensure: Syntax-Aware Feature Vector $\mathbf{f}\in \mathbb{R}^{126}$ 1: 1. String Reconstruction & Alignment Map   
2: Decode tokens y into a complete string text.   
3: Construct an alignment map $M$ where $M[i]$ contains the list of token indices corresponding to the i-th word in text.   
4: 2. POS Tagging & Propagation   
5: Initialize tag list $\mathcal{T}$ of length $T$ 6: Run POS tagger (e.g., SpaCy) on string text to obtain words $W_{1},\ldots ,W_{K}$ and tags tag1,..,tagK.   
7: for each word index $k = 1$ to $K$ do   
8: Get corresponding token indices: $\mathcal{I}_{\mathrm{tokens}}\gets M[k]$ 9: Get POS tag for the word: $c\gets \mathrm{tag}_k$ 10: for each token index $t\in \mathcal{I}_{\mathrm{tokens}}$ do   
11: $\tau_t\gets c$ {Propagate parent word's tag to all sub-word tokens}   
12: end for   
13: end for   
14: 3. Syntax-Aware Aggregation   
15: Initialize feature vector $\mathbf{f}\gets \emptyset$ 16: Define set of Universal POS tags $\mathcal{P}_{\mathrm{univ}}$ 17: for each POS category $c\in \mathcal{P}_{\mathrm{univ}}$ do   
18: Identify tokens belonging to this category: $\mathcal{I}_c = \{t\mid \tau_t = c\}$ 19: if $\mathcal{I}_c\neq \emptyset$ then   
20: // Compute mean attribution profile for this syntactic category   
21: $\bar{\mathbf{v}}_c\leftarrow \frac{1}{|\mathcal{I}_c|}\sum_{t\in \mathcal{I}_c}\mathbf{v}_t$ 22: else   
23: $\bar{\mathbf{v}}_c\gets 0_7$ {Fill with zeros if category is absent in response}   
24: end if   
25: f $\leftarrow$ Concatenate(f, $\bar{\mathbf{v}}_c)$ 26: end for   
27: return f

outer random seeds. For each seed, we construct an ensemble of 5 XGBoost classifiers. The final prediction is derived via Hard Voting (majority rule) for binary classification metrics (F1, Recall) and Soft Voting (probability averaging) for AUC.

Protocol I: Standard Split (RAGTruth Llama2- 7b/13b/Mistral). For datasets with official splits, we utilize the standard training and test sets.

• Optimization: We employ Optuna with a TPE sampler to optimize hyperparameters. We run 50 trials maximizing the F1-score using 5-fold Stratified Cross-Validation on the training set.   
• Training: For each of the 5 outer seeds, we train an ensemble of 5 models. Each ensemble member is trained on a distinct $8 5 \% / 1 5 \%$ split of the training data to facilitate diversity and Early Stopping (patience $\mathord { : } = 5 0$ ).

• Evaluation: Predictions are aggregated via voting on the held-out test set.

Protocol II: Stratified 20-Fold CV (RAGTruth Llama3-8b). As the Llama3-8b subset lacks a training split, we adopt a Stratified 20-Fold Cross-Validation.

• Optimization: Hyperparameters are optimized via Optuna on the available data prior to the cross-validation loop.   
• Training: We iterate through 20 folds. Within each fold, we train the 5-member XGBoost ensemble on the training partition (using diverse internal splits for early stopping).   
• Aggregation: Predictions from all folds are concatenated to compute the final performance metrics for each outer seed.

Protocol III: Nested Leave-One-Out CV (Dolly). Given the limited size of the Dolly dataset $N =$ 100), we implement a rigorous Nested Leave-One-Out (LOO) Cross-Validation.

• Outer Loop: We iterate 100 times, isolating a single sample for testing in each iteration.   
• Inner Loop (Optimization): On the remaining 99 samples, we conduct independent hyperparameter searches using Optuna (50 trials).   
• Ensemble Training: For each LOO step, we train 5 XGBoost models on the 99 training samples. To handle class imbalance, we dynamically adjust the scale_pos_weight parameter.   
• Inference: The final verdict for the single test sample is determined by the hard vote of the 5 ensemble members. This process is repeated for all 5 outer seeds to verify statistical significance.

Implementation Note regarding Memory Constraints. While TPA is theoretically designed for single-pass parallel execution via teacher forcing, storing the full attention matrices $\mathcal { O } ( T ^ { 2 } )$ and computational graphs for long sequences can be memory-intensive. In our specific experiments, due to GPU memory limitations (NVIDIA A100 40G), we implemented the attribution process sequentially (token-by-token). We emphasize that this implementation is mathematically equivalent to the parallel version due to the causal masking mechanism of Transformer decoders. The choice between serial and parallel implementation represents a trade-off between efficiency and memory usage, without affecting the attribution values or detection performance reported in this paper.

Hyperparameter Search and Best Value Discussion. We utilize the Optuna framework with a Tree-structured Parzen Estimator (TPE) sampler to perform automated hyperparameter tuning. For each model and data split, we run 50 trials to maximize the F1-score. The comprehensive search space is presented in Table 3. Regarding the best-found values, our analysis reveals that the optimal configuration is highly dependent on the specific LLM and dataset size. We observed a consistent preference for moderate tree depths ( $4 \leq$ max_depth $\leq 6$ ) and stronger regularization

$\lambda \geq 1 . 5$ , $\gamma \geq 0 . 2$ ) across most experiments, indicating that preventing overfitting is critical given the high dimensionality of our feature space relative to the dataset size. Conversely, the optimal learning rate varied significantly (0.01 to 0.1) depending on the base model (e.g., Llama-2 vs. Llama-3). Therefore, rather than fixing a single set of hyperparameters, we adopt a dynamic optimization strategy where the best parameters are re-evaluated for each fold and seed. This approach ensures that our reported results reflect the robust performance of the method rather than a specific tuning artifact.

Table 3: Hyperparameter search space for the XGBoost classifier in TPA.   

<table><tr><td>Hyperparameter</td><td>Search Values</td></tr><tr><td>Learning Rate</td><td>{0.01, 0.02, 0.05, 0.1}</td></tr><tr><td>Max Depth</td><td>{4, 5, 6, 7}</td></tr><tr><td>Subsample</td><td>{0.6, 0.7, 0.8}</td></tr><tr><td>Colsample By Tree</td><td>{0.7, 0.8, 0.9}</td></tr><tr><td>Gamma (γ)</td><td>{0.1, 0.2, 0.5}</td></tr><tr><td>Reg Alpha (α)</td><td>{0.01, 0.1, 0.5}</td></tr><tr><td>Reg Lambda (λ)</td><td>{1, 1.5, 2}</td></tr><tr><td>Fixed Parameters</td><td>n_estimators=1000, patience=50</td></tr></table>

Artifacts and intended use. We use publicly available benchmarks (RAGTruth and Dolly (AC)) and open-access LLM checkpoints strictly for research evaluation, consistent with their intended research use. We do not redistribute any original datasets or model weights; our released artifact is research code and documentation for reproducing the experiments, and it instructs users to obtain the datasets/models from their official sources.

# 12 Baselines Introduction

1. EigenScore/INSIDE(Chen et al., 2024) Focus on detecting hallucination by evaluating response’s semantic consistency, which is defined as the logarithm determinant of convariance matrix LLM’s internal states during generating the response.   
2. SEP(Han et al., 2024) Proposed a linear model to detect hallucination based on semantic entropy in test time whithout requiring multiple responses.   
3. SAPLMA(Azaria and Mitchell, 2023) Detecting hallucination based on the hidden layer activations of LLMs.

4. ITI(Li et al., 2023) Detecting hallucination based on the hidden layer activations of LLMs.   
5. Ragtruth Prompt(Niu et al., 2024) Provdes prompts for a LLM-as-judge to detect hallucination in RAG setting.   
6. ChainPoll(Friel and Sanyal, 2023) Provdes prompts for a LLM-as-judge to detect hallucination in RAG setting.   
7. RAGAS(Es et al., 2024) It use a LLM to split the response into a set of statements and verify each statement is supported by the retrieved documents. If any statement is not supported, the response is considered hallucinated.   
8. Trulens(TrueLens, 2024) Evaluating the overlap between the retrieved documents and the generated response to detect hallucination by a LLM.   
9. P(True)(Kadavath et al., 2022) The paper detects hallucinations by having the model estimate the probability that its own generated answer is correct, based on the key assumption that it is often easier for a model to recognize a correct answer than to generate one.   
10. SelfCheckGPT(Manakul et al., 2023) Self-CheckGPT detects hallucinations by checking for informational consistency across multiple stochastically sampled responses, based on the assumption that factual knowledge leads to consistent statements while hallucinations lead to divergent and contradictory ones.   
11. LN-Entropy(Malinin and Gales, 2021) This paper detects hallucinations by quantifying knowledge uncertainty, which it measures primarily with a novel metric called Reverse Mutual Information that captures the disagreement across an ensemble’s predictions, with high RMI indicating a likely hallucination.   
12. Energy(Liu et al., 2020) This paper detects hallucinations by using an energy score, derived directly from the model’s logits, as a more reliable uncertainty measure than softmax confidence to identify out-of-distribution inputs that cause the model to hallucinate.   
13. Focus(Zhang et al., 2023) This paper detects hallucinations by calculating an uncertainty

score focused on keywords, and then refines it by propagating penalties from unreliable context via attention and correcting token probabilities using entity types and inverse document frequency to mitigate both overconfidence and underconfidence.   
14. Perplexity(Ren et al., 2023) This paper detects hallucinations by separately measuring the Relative Mahalanobis Distance for both input and output embeddings, based on the assumption that in-domain examples will have embeddings closer to their respective foreground (in-domain) distributions than to a generic background distribution.   
15. REFCHECKER(Hu et al., 2024) It use a LLM to extract claim-triplets from a response and verify them by another LLM to detect hallucination.   
16. REDEEP(Sun et al., 2025) It detects hallucination by analyzing the balance between the contributions from Copying Heads that process external context and Knowledge FFNs that inject internal knowledge, based on the finding that RAG hallucinations often arise from conflicts between these two sources. This method has two version: token level and chunk level. We compare with the latter since it has better performance generally.   
17. NoVo(Ho et al., 2025) It leverages the L2 norms of specific attention heads as reliable indicators of truthfulness. By identifying a subset of truth-correlated heads from a small reference set, it employs a voting mechanism based on these head norms to detect hallucinations without requiring model parameter updates.   
18. TSV(Park et al., 2025) It introduces a lightweight steering vector to reshape the LLM’s latent space during inference. By actively intervening to enhance the linear separability between truthful and hallucinated representations in the hidden states, it enables effective detection using a simple classifier on the steered embeddings.

# 13 Complexity Analysis of the Attribution Process

In this section, we rigorously analyze the computational overhead of our attribution framework. We

report complexity in terms of analysis passes and asymptotic costs, as wall-clock varies substantially with caching strategies and kernel implementations. We focus strictly on the attribution extraction process for a generated response of length $T$ . Let $L , d$ , $| \nu |$ , and $H$ denote the number of layers, hidden dimension, vocabulary size, and attention heads (per layer), respectively. The standard inference complexity for a Transformer is $\mathcal { O } ( L \cdot T \cdot d ^ { 2 } + L \cdot H \cdot T ^ { 2 } )$ . Our attribution process introduces post-hoc computations, decomposed into three specific stages:

1. Complete Probability Decomposition. To satisfy Theorem 1, we must compute the complete probability changes using the probe function $\Phi ( \mathbf { h } , y )$ . The bottleneck is the calculation of the global partition function (denominator) in Softmax.

• Mechanism: The probe function $\Phi ( \mathbf { h } , y ) =$ Softma $\begin{array} { r l r } { \mathfrak { c } ( { \bf h } { \bf W } _ { U } ) _ { y } } & { = } & { \frac { \exp ( { \bf w } _ { U , y } ^ { \top } { \bf h } ) } { \sum _ { v \in \mathcal { V } } \exp ( { \bf w } _ { U , v } ^ { \top } { \bf h } ) } } \end{array}$ requires projecting the hidden state h to the full vocabulary logits $\mathbf { z } = \mathbf { h } \mathbf { W } _ { U }$ .   
• Single Probe Complexity: For a single hidden state $\mathbf { h } \in \mathbb { R } ^ { d }$ , the matrix-vector multiplication with the unembedding matrix ${ \bf W } _ { U } \in \mathbb $ $\mathbb { R } ^ { d \times | \nu | }$ costs $O ( | \nu | \cdot d )$ .   
• Total Calculation: We must apply this probe at multiple points:

1. Global Components: For $\Delta P _ { \mathrm { i n i t i a l } }$ and $\Delta P _ { \mathrm { L N } }$ , the probe is called once per generation step. Cost: $\mathcal { O } ( T \cdot | \mathcal { V } | \cdot d )$ .   
2. Layer Components: For ∆P (l)att $\Delta P _ { \mathrm { a t t } } ^ { ( l ) }$ and ∆P (l), $\Delta P _ { \mathrm { f f n } } ^ { ( l ) }$ the probe is invoked twice per layer (before and after the residual update). Summing over $L$ layers, this costs $\mathcal { O } ( L \cdot T \cdot | \mathcal { V } | \cdot d )$ .

• Stage Complexity: Combining these terms, the dominant complexity is $\mathcal { O } ( L \cdot T \cdot | \mathcal { V } | \cdot d )$ .

2. Head-wise Attribution. Once ∆P (l)att $\Delta P _ { \mathrm { a t t } } ^ { ( l ) }$ is obtained, we apportion it to individual heads based on their contribution to the target logit.

• Mechanism: This attribution requires projecting the target token vector ${ \bf w } _ { U , y }$ back into the hidden state space using the layer’s output projection matrix W(l)O $\mathbf { W } _ { O } ^ { ( l ) } \in \mathbb { R } ^ { d \times d }$ .   
• Step Complexity: The calculation proceeds in two sub-steps:

1. Projection: We compute the projected target vector $\mathbf { g } = ( \bar { \mathbf { W } } _ { O } ^ { ( l ) } ) ^ { \top } \mathbf { w } _ { U , y }$ . Since $\mathbf { W } _ { O } ^ { ( l ) }$ is a $d \times d$ matrix, this matrix-vector multiplication costs $\mathcal { O } ( d ^ { 2 } )$ .

2. Assignment: We distribute the contribution to $H$ heads by performing dot products between the head outputs $\mathbf { o } _ { h }$ and the corresponding segments of g. For $H$ heads, this sums to $\mathcal O ( d )$ .

• Stage Complexity: The projection step $( \mathcal { O } ( d ^ { 2 } ) )$ dominates the assignment step $( \mathcal { O } ( d ) )$ . Integrating over $L$ layers and $T$ tokens, the total complexity is $\mathcal { O } ( L \cdot T \cdot d ^ { 2 } )$ .

3. Mapping Attention to Input Sources. Finally, we map head contributions to the four sources by aggregating attention weights ${ \textbf { A } } \in$ $\mathbb { R } ^ { H \times | \mathbf { s } | \times | \mathbf { s } | }$ . This involves two distinct sub-steps for each generated token at step $t$ within a single layer:

• Step 1: Summation. For each head $h$ , we sum the attention weights corresponding to specific source indices (e.g., $I _ { R A G } )$ ):

$$
w _ {h, S} = \sum_ {k \in \mathcal {I} _ {S}} \mathbf {A} _ {h} [ n _ {t}, k ]
$$

This requires iterating over the causal range up to $n _ { t }$ . For $H$ heads, the cost is $\mathcal { O } ( H \cdot n _ { t } )$ .

• Step 2: Normalization & Weighting. We calculate the final source contribution by weighting the head contributions:

$$
\Delta P _ {S} = \sum_ {h = 1} ^ {H} \Delta P _ {h} \cdot \frac {w _ {h , S}}{\sum_ {\text {a l l s o u r c e s}} w _ {h ,}}
$$

This involves scalar operations proportional to the number of heads $H$ . Cost: $\mathcal O ( H )$ .

• Stage Complexity: The summation step $( { \mathcal { O } } ( H \cdot n _ { t } ) )$ ) dominates. We sum this cost across all $L$ layers, and then accumulate over the generation steps $t = 1$ to $T$ . The calculation is $\begin{array} { r } { \sum _ { t = 1 } ^ { T } ( L \cdot \bar { H } \cdot n _ { t } ) \approx \mathcal { O } ( L \cdot H \cdot T ^ { 2 } ) } \end{array}$ .

Overall Efficiency. The total computational cost is the sum of these three stages:

$$
\mathcal {C} _ {\text {t o t a l}} = \mathcal {O} (\underbrace {L \cdot T \cdot | \mathcal {V} | \cdot d} _ {\text {P r o b . D e c o m p .}} + \underbrace {L \cdot T \cdot d ^ {2}} _ {\text {H e a d A t t r .}} + \underbrace {L \cdot H \cdot T ^ {2}} _ {\text {S o u r c e M a p}})
$$

Runtime Efficiency. It is worth noting that theoretical complexity does not directly equate to wallclock latency. Standard text generation is serial (token-by-token), which limits GPU parallelization. In contrast, our framework can process the full sequence of length $T$ in a single parallel teacherforced pass, enabling efficient GPU matrix operations. When implemented this way, it avoids the $K$ sequential generation passes required by baselines like SelfCheckGPT.

# 14 AI Assistance Disclosure

We used AI-based tools to assist with language editing and draft refinement. All technical content, experiments, and conclusions were produced and verified by the authors.

Table 4: Full Results on RAGTruth and Dolly (AC) datasets. TPA results are reported as the Mean and Standard Deviation over 5 random seeds, obtained using an ensemble of 5 XGBoost models. † indicates that the improvement over the strongest baseline is statistically significant $( p < 0 . 0 5 )$ under a one-sample t-test. P-values are reported for metrics where TPA achieves Rank 1; otherwise, a dash (-) is shown. Bold values indicate the best performance and underlined values indicate the second-best.   

<table><tr><td rowspan="2">Model</td><td colspan="9">RAGTruth</td></tr><tr><td colspan="3">LLaMA2-7B</td><td colspan="3">LLaMA2-13B</td><td colspan="3">LLaMA3-8B</td></tr><tr><td>Metric</td><td>AUC</td><td>Recall</td><td>F1</td><td>AUC</td><td>Recall</td><td>F1</td><td>AUC</td><td>Recall</td><td>F1</td></tr><tr><td>SelfCheckGPT (Manakul et al., 2023)</td><td>—</td><td>0.4642</td><td>0.4642</td><td>—</td><td>0.4642</td><td>0.4642</td><td>—</td><td>0.5111</td><td>0.5111</td></tr><tr><td>Perplexity (Ren et al., 2023)</td><td>0.5091</td><td>0.5190</td><td>0.6749</td><td>0.5091</td><td>0.5190</td><td>0.6749</td><td>0.6235</td><td>0.6537</td><td>0.6778</td></tr><tr><td>LN-Entropy (Malinin and Gales, 2021)</td><td>0.5912</td><td>0.5383</td><td>0.6655</td><td>0.5912</td><td>0.5383</td><td>0.6655</td><td>0.7021</td><td>0.5596</td><td>0.6282</td></tr><tr><td>Energy (Liu et al., 2020)</td><td>0.5619</td><td>0.5057</td><td>0.6657</td><td>0.5619</td><td>0.5057</td><td>0.6657</td><td>0.5959</td><td>0.5514</td><td>0.6720</td></tr><tr><td>Focus (Zhang et al., 2023)</td><td>0.6233</td><td>0.5309</td><td>0.6622</td><td>0.7888</td><td>0.6173</td><td>0.6977</td><td>0.6378</td><td>0.6688</td><td>0.6879</td></tr><tr><td>Prompt (Niu et al., 2024)</td><td>—</td><td>0.7200</td><td>0.6720</td><td>—</td><td>0.7000</td><td>0.6899</td><td>—</td><td>0.4403</td><td>0.5691</td></tr><tr><td>ChainPoll (Friel and Sanyal, 2023)</td><td>0.6738</td><td>0.7832</td><td>0.7066</td><td>0.7414</td><td>0.7874</td><td>0.7342</td><td>0.6687</td><td>0.4486</td><td>0.5813</td></tr><tr><td>RAGAS (Es et al., 2024)</td><td>0.7290</td><td>0.6327</td><td>0.6667</td><td>0.7541</td><td>0.6763</td><td>0.6747</td><td>0.6776</td><td>0.3909</td><td>0.5053</td></tr><tr><td>Trulens (TrueLens, 2024)</td><td>0.6510</td><td>0.6814</td><td>0.6567</td><td>0.7073</td><td>0.7729</td><td>0.6867</td><td>0.6464</td><td>0.3909</td><td>0.5053</td></tr><tr><td>RefCheck (Hu et al., 2024)</td><td>0.6912</td><td>0.6280</td><td>0.6736</td><td>0.7857</td><td>0.6800</td><td>0.7023</td><td>0.6014</td><td>0.3580</td><td>0.4628</td></tr><tr><td>P(True) (Kadavath et al., 2022)</td><td>0.7093</td><td>0.5194</td><td>0.5313</td><td>0.7998</td><td>0.5980</td><td>0.7032</td><td>0.6323</td><td>0.7083</td><td>0.6835</td></tr><tr><td>EigenScore (Chen et al., 2024)</td><td>0.6045</td><td>0.7469</td><td>0.6682</td><td>0.6640</td><td>0.6715</td><td>0.6637</td><td>0.6497</td><td>0.7078</td><td>0.6745</td></tr><tr><td>SEP (Han et al., 2024)</td><td>0.7143</td><td>0.7477</td><td>0.6627</td><td>0.8089</td><td>0.6580</td><td>0.7159</td><td>0.7004</td><td>0.7333</td><td>0.6915</td></tr><tr><td>SAPLMA (Azaria and Mitchell, 2023)</td><td>0.7037</td><td>0.5091</td><td>0.6726</td><td>0.8029</td><td>0.5053</td><td>0.6529</td><td>0.7092</td><td>0.5432</td><td>0.6718</td></tr><tr><td>ITI (Li et al., 2023)</td><td>0.7161</td><td>0.5416</td><td>0.6745</td><td>0.8051</td><td>0.5519</td><td>0.6838</td><td>0.6534</td><td>0.6850</td><td>0.6933</td></tr><tr><td>ReDeEP (Sun et al., 2025)</td><td>0.7458</td><td>0.8097</td><td>0.7190</td><td>0.8244</td><td>0.7198</td><td>0.7587</td><td>0.7285</td><td>0.7819</td><td>0.6947</td></tr><tr><td>TSV (Park et al., 2025)</td><td>0.6609</td><td>0.5526</td><td>0.6632</td><td>0.8123</td><td>0.8068</td><td>0.6987</td><td>0.7769</td><td>0.5546</td><td>0.6442</td></tr><tr><td>Novo (Ho et al., 2025)</td><td>0.7608</td><td>0.8274</td><td>0.7057</td><td>0.8506</td><td>0.7826</td><td>0.7733</td><td>0.8258</td><td>0.7737</td><td>0.7801</td></tr><tr><td>TPA</td><td>0.7873†</td><td>0.8328</td><td>0.7238†</td><td>0.8681†</td><td>0.7913</td><td>0.7975†</td><td>0.8211</td><td>0.7860</td><td>0.7843</td></tr><tr><td>Std</td><td>0.0007</td><td>0.0145</td><td>0.0039</td><td>0.0075</td><td>0.0086</td><td>0.0076</td><td>0.0025</td><td>0.0068</td><td>0.0053</td></tr><tr><td>P-val</td><td>&lt;0.001</td><td>=0.227</td><td>=0.025</td><td>=0.003</td><td>-</td><td>=0.001</td><td>-</td><td>=0.125</td><td>=0.074</td></tr><tr><td></td><td colspan="9">Dolly (AC)</td></tr><tr><td>Model</td><td colspan="3">LLaMA2-7B</td><td colspan="3">LLaMA2-13B</td><td colspan="3">LLaMA3-8B</td></tr><tr><td>Metric</td><td>AUC</td><td>Recall</td><td>F1</td><td>AUC</td><td>Recall</td><td>F1</td><td>AUC</td><td>Recall</td><td>F1</td></tr><tr><td>SelfCheckGPT (Manakul et al., 2023)</td><td>—</td><td>0.1897</td><td>0.3188</td><td>0.2728</td><td>0.1897</td><td>0.3188</td><td>0.1095</td><td>0.2195</td><td>0.3600</td></tr><tr><td>Perplexity (Ren et al., 2023)</td><td>0.2728</td><td>0.7719</td><td>0.7097</td><td>0.2728</td><td>0.7719</td><td>0.7097</td><td>0.1095</td><td>0.3902</td><td>0.4571</td></tr><tr><td>LN-Entropy (Malinin and Gales, 2021)</td><td>0.2904</td><td>0.7368</td><td>0.6772</td><td>0.2904</td><td>0.7368</td><td>0.6772</td><td>0.1150</td><td>0.5365</td><td>0.5301</td></tr><tr><td>Energy (Liu et al., 2020)</td><td>0.2179</td><td>0.6316</td><td>0.6261</td><td>0.2179</td><td>0.6316</td><td>0.6261</td><td>-0.0678</td><td>0.4047</td><td>0.4440</td></tr><tr><td>Focus (Zhang et al., 2023)</td><td>0.3174</td><td>0.5593</td><td>0.6534</td><td>0.1643</td><td>0.7333</td><td>0.6168</td><td>0.1266</td><td>0.6918</td><td>0.6874</td></tr><tr><td>Prompt (Niu et al., 2024)</td><td>—</td><td>0.3965</td><td>0.5476</td><td>—</td><td>0.4182</td><td>0.5823</td><td>—</td><td>0.3902</td><td>0.5000</td></tr><tr><td>ChainPoll (Friel and Sanyal, 2023)</td><td>0.3502</td><td>0.4138</td><td>0.5581</td><td>0.4758</td><td>0.4364</td><td>0.6000</td><td>0.2691</td><td>0.3415</td><td>0.4516</td></tr><tr><td>RAGAS (Es et al., 2024)</td><td>0.2877</td><td>0.5345</td><td>0.6392</td><td>0.2840</td><td>0.4182</td><td>0.5476</td><td>0.3628</td><td>0.8000</td><td>0.5246</td></tr><tr><td>Trulens (TrueLens, 2024)</td><td>0.3198</td><td>0.5517</td><td>0.6667</td><td>0.2565</td><td>0.3818</td><td>0.4941</td><td>0.3352</td><td>0.3659</td><td>0.5172</td></tr><tr><td>RefCheck (Hu et al., 2024)</td><td>0.2494</td><td>0.3966</td><td>0.5412</td><td>0.2869</td><td>0.2545</td><td>0.3944</td><td>-0.0089</td><td>0.1951</td><td>0.2759</td></tr><tr><td>P(True) (Kadavath et al., 2022)</td><td>0.1987</td><td>0.6350</td><td>0.6509</td><td>0.2009</td><td>0.6180</td><td>0.5739</td><td>0.3472</td><td>0.5707</td><td>0.6573</td></tr><tr><td>EigenScore (Chen et al., 2024)</td><td>0.2428</td><td>0.7500</td><td>0.7241</td><td>0.2948</td><td>0.8181</td><td>0.7200</td><td>0.2065</td><td>0.7142</td><td>0.5952</td></tr><tr><td>SEP (Han et al., 2024)</td><td>0.2605</td><td>0.6216</td><td>0.7023</td><td>0.2823</td><td>0.6545</td><td>0.6923</td><td>0.0639</td><td>0.6829</td><td>0.6829</td></tr><tr><td>SAPLMA (Azaria and Mitchell, 2023)</td><td>0.0179</td><td>0.5714</td><td>0.7179</td><td>0.2006</td><td>0.6000</td><td>0.6923</td><td>-0.0327</td><td>0.4040</td><td>0.5714</td></tr><tr><td>ITI (Li et al., 2023)</td><td>0.0442</td><td>0.5816</td><td>0.6281</td><td>0.0646</td><td>0.5385</td><td>0.6712</td><td>0.0024</td><td>0.3091</td><td>0.4250</td></tr><tr><td>ReDeEP (Sun et al., 2025)</td><td>0.5136</td><td>0.8245</td><td>0.7833</td><td>0.5842</td><td>0.8518</td><td>0.7603</td><td>0.3652</td><td>0.8392</td><td>0.7100</td></tr><tr><td>TSV (Park et al., 2025)</td><td>0.7454</td><td>0.8728</td><td>0.7684</td><td>0.7552</td><td>0.5952</td><td>0.6043</td><td>0.7347</td><td>0.6467</td><td>0.6695</td></tr><tr><td>Novo (Ho et al., 2025)</td><td>0.6423</td><td>0.8070</td><td>0.7244</td><td>0.6909</td><td>0.7222</td><td>0.6903</td><td>0.7418</td><td>0.5854</td><td>0.6316</td></tr><tr><td>TPA</td><td>0.7134</td><td>0.7897</td><td>0.7527</td><td>0.8159†</td><td>0.9741†</td><td>0.8075†</td><td>0.7608</td><td>0.6561</td><td>0.7529†</td></tr><tr><td>Std</td><td>0.0215</td><td>0.0227</td><td>0.0199</td><td>0.0210</td><td>0.0096</td><td>0.0137</td><td>0.0164</td><td>0.0452</td><td>0.0337</td></tr><tr><td>P-val</td><td>-</td><td>-</td><td>-</td><td>&lt;0.001</td><td>&lt;0.001</td><td>&lt;0.001</td><td>0.0025</td><td>-</td><td>=0.001</td></tr></table>

![](images/a1ba49c2260fe6978a1b6cd39884d21a0f2204c1849a3b0c471383830b9e3fda.jpg)  
(a) Llama3-8B

![](images/9354f935718eb334904ef009a21d2156d57d24d98592b5d37598ca7a24cfb5d4.jpg)  
(b) Mistral-7B   
Figure 5: SHAP summary plots illustrating the decision logic. We visualize the top-10 features for classifiers trained on the RAGTruth subsets corresponding to Llama3-8B and Mistral-7B. The x-axis represents the SHAP value. Positive values indicate a push towards classifying the response as a Hallucination. The color represents the feature value (Red $=$ High attribution, Blue $=$ Low).
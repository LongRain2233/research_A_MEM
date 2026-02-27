# Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models

Zhen Qin 1 Weigao Sun 1 Dong Li 1 Xuyang Shen 1 Weixuan Sun 1 Yiran Zhong 1

# Abstract

Linear attention is an efficient attention mechanism that has recently emerged as a promising alternative to conventional softmax attention. With its ability to process tokens in linear computational complexities, linear attention, in theory, can handle sequences of unlimited length without sacrificing speed, i.e., maintaining a constant training speed for various sequence lengths with a fixed memory consumption. However, due to the issue with cumulative summation (cumsum), current Linear Attention algorithms cannot demonstrate their theoretical advantage in a casual setting. In this paper, we present Lightning Attention-2, the first linear attention implementation that enables linear attention to realize its theoretical computational benefits. To achieve this, we leverage the thought of tiling, separately handling the intrablock and inter-block components in linear attention calculation. Specifically, we utilize the conventional attention computation mechanism for the intra-blocks and apply linear attention kernel tricks for the inter-blocks. A tiling technique is adopted through both forward and backward procedures to take full advantage of the GPU hardware. We implement our algorithm in Triton to make it IO-aware and hardware-friendly. Various experiments are conducted on different model sizes and sequence lengths. Lightning Attention-2 retains consistent training and inference speed regardless of input sequence length and is significantly faster than other attention mechanisms. The source code is available at Lightning Attention-2.

# 1. Introduction

The Transformer architecture has achieved widespread adoption, particularly in the domain of large language models

(LLM) (Brown et al., 2020; Touvron et al., 2023a;b; Peng et al., 2023; Qin et al., 2023b) and multi-modal models (Li et al., 2022; 2023a; Liu et al., 2023; Radford et al., 2021; Li et al., 2023b; Lu et al., 2022; Mao et al., 2023; Shen et al., 2023; Zhou et al., 2023; Sun et al., 2023a; Hao et al., 2024). However, its computational complexity grows quadratically with the length of the input sequence, making it challenging to model extremely long sequences.

Unlimited sequence length stands out as a noteworthy aspect within the realm of LLM, attracting considerable attention from researchers who seek intelligent solutions. The potential applications of LLM with unlimited sequence length are diverse, encompassing extended conversations in various professional domains and handling a vast number of tokens in multimodal modeling tasks.

In response to the quadratic complexity challenge, a promising resolution emerges in the form of linear attention. This method involves the elimination of the softmax operation and capitalizes on the associativity property of matrix products. Consequently, it significantly accelerates both training and inference procedures. To elaborate, linear attention reduces the computational complexity from $O ( n ^ { 2 } )$ to $O ( n )$ by leveraging the kernel trick (Katharopoulos et al., 2020b; Choromanski et al., 2020; Peng et al., 2021; Qin et al., 2022b) to compute the attention matrices, where $n$ represents the sequence length. This avenue holds substantial promise for augmenting the efficiency of transformer-style models across a broad spectrum of applications.

It is important to note that the notable reduction in complexity from $O ( n ^ { 2 } )$ to $O ( n )$ in linear attention is only theoretical and may not directly translate to a proportional improvement in computational efficiency on hardware in practice. The realization of practical wall-clock speedup faces challenges, primarily stemming from two issues: 1). the dominance of memory access (I/O) on the GPU could impact the overall computation speed of attention. 2). the cumulative summation (cumsum) needed by the linear attention kernel trick prevents it from reaching its theoretical training speed in the causal setting.

The first issue has been successfully addressed by Lightning Attention-1 (Qin et al., 2023b). In this paper, we introduce

![](images/4bab3e2b7eadf9e2576e9c610bd05643e86def166912cd7cea53bac2761ae81e.jpg)

![](images/d61317a146a57d78499d492fe804fd9cd3c3802251ecf9bd660420c080409a4b.jpg)

![](images/301ce14c4595d1df5a69abdee30f71b3706366e44f931e373defe8722cdec0a1.jpg)  
Figure 1. Speed Showdown: FlashAttention vs. Lightning Attention in Expanding Sequence Lengths and Model Sizes. The diagram above provides a comparative illustration of training speed, Token per GPU per Second (TGS) for LLaMA with FlashAttention-2, TransNormerLLM with Lightning Attention-1 and TransNormerLLM with Lightning Attention-2, implemented across three model sizes: 400M, 1B, and 3B from left to right. It is strikingly evident that Lightning Attention-2 manifests a consistent training speed irrespective of the increasing sequence length. Conversely, the other methods significantly decline training speed as the sequence length expands.

Lightning Attention-2 to solve the second issue. The key idea is to leverage the concept of "divide and conquer" by separately handling the intra block and inter block components in linear attention calculation. Specifically, for the intra blocks, we maintain the use of conventional attention computation mechanism to compute the product of QKV, while for the inter blocks, we employ the linear attention kernel trick (Katharopoulos et al., 2020b). Tiling techniques are implemented in both forward and backward procedures to fully leverage GPU hardware capabilities. As a result, the Lightning Attention-2 can train LLMs with unlimited sequence length without extra cost1, as its computational speed remains constant with increasing sequence length under fixed memory consumption.

We performed a comprehensive evaluation of Lightning Attention-2 across a diverse range of sequence lengths to assess its accuracy and compare its computational speed and memory utilization with FlashAttention-2 (Dao, 2023) and Lightning Attention-1. The findings indicate that Lightning Attention-2 exhibits a notable advantage in computational speed, attributed to its innovative intra-inter separation strategy. Additionally, Lightning Attention-2 demonstrates a reduced memory footprint compared to its counterparts without compromising performance.

# 2. Related Work

# 2.1. Linear Attention

Linear Transformer architectures discard the Softmax Attention mechanism, replacing it with distinct approximations (Katharopoulos et al., 2020a; Choromanski et al., 2020; Peng et al., 2021; Qin et al., 2022b;a). The key idea is to

leverage the “kernel trick" to accelerate the attention matrix computation, i.e., compute the product of keys and values first to circumvent the $n \times n$ matrix multiplication. Multiple methods have been proposed to replace the softmax operation. For instance, Katharopoulos et al. (2020a) employ the $\mathrm { 1 + e l u }$ activation function, Qin et al. (2022b) utilize the cosine function to approximate softmax properties, and Ke et al. (2021); Zheng et al. (2022; 2023) leverage sampling strategies to directly mimic softmax operation. Despite having a theoretical complexity of $O ( n d ^ { 2 } )$ , the practical computational efficiency of linear attention diminishes notably in causal attention scenarios, primarily due to the necessity for cumsum operations (Hua et al., 2022).

# 2.2. IO-aware Attention

The FlashAttention series (Dao et al., 2022; Dao, 2023) focuses on system-level optimizations for the efficient implementation of the standard attention operator on GPU platforms. Extensive validation has demonstrated its effectiveness. The approach employs tiling strategies to minimize the volume of memory reads/writes between the GPU’s high bandwidth memory (HBM) and on-chip SRAM.

To address the issue of slow computation for Linear Attention in the causal setting, Lightning Attention 1 (Qin et al., 2023b) employs the approach of FlashAttention-1/2, which involves segmenting the inputs $\mathbf { Q } , \mathbf { K } , \mathbf { V }$ into blocks, transferring them from slow HBM to fast SRAM, and then computing the attention output with respect to these blocks. Subsequently, the final results are accumulated. Although this method is much more efficient than the PyTorch implementation, it does not take advantage of the computational characteristics inherent to Linear Attention, and the theoretical complexity remains $O ( n ^ { 2 } d )$ .

# 2.3. Long Sequence Handling in LLM

A widely adopted strategy to tackle challenges related to length extrapolation involves the integration of Relative Positional Encoding (RPE) techniques (Su et al., 2021; Qin et al., 2023c), strategically directing attention towards neighboring tokens. ALiBi (Press et al., 2022) utilizes linear decay biases in attention mechanisms to mitigate the impact of distant tokens. Roformer (Su et al., 2021) introduces a novel Rotary Position Embedding (RoPE) method, widely embraced in the community, effectively leveraging positional information for transformer-based language model learning. Kerple (Chi et al., 2022) explores shift-invariant conditionally positive definite kernels within RPEs, introducing a suite of kernels aimed at enhancing length extrapolation properties, with ALiBi recognized as one of its instances. Furthermore, Sandwich (Chi et al., 2023) postulates a hypothesis elucidating the mechanism behind ALiBi, empirically validating it by incorporating the hypothesis into sinusoidal positional embeddings. (Qin et al., 2024) explored the sufficient conditions for additive relative position encoding to have extrapolation capabilities.

Instead of investigating the length extrapolation capability of transformers, some works also attempt to directly increase the context window sizes. Chen et al. (2023) introduces Position Interpolation (PI), extending context window sizes of RoPE-based pretrained Large Language Models (LLMs) such as LLaMA models to up to 32768 with minimal finetuning (within 1000 steps). StreamingLLM (Xiao et al., 2023) proposes leveraging the attention sink phenomenon, maintaining the Key and Value information of initial tokens to substantially recover the performance of window attention. As the sequence grows longer, the performance degrades. These methods can only extend sequence length in fine-tuning or testing phases, while our method allows training models in long sequence lengths from scratch with no additional cost.

# 3. Method

# 3.1. Preliminary

We first recall the formulation of linear attention and then introduce our proposed Lightning Attention-2. In the case of NormAttention within TransNormer (Qin et al., 2022a), attention computation deviates from the conventional Transformer structure (Vaswani et al., 2017) by eschewing the costly softmax and scaling operations. The NormAttention mechanism can be expressed as follows:

$$
\mathbf {O} = \operatorname {N o r m} ((\mathbf {Q K} ^ {\top}) \mathbf {V}), \tag {1}
$$

where Q, K, and $\mathbf { V } \in \mathbb { R } ^ { n \times d }$ are the query, key, and value matrices, respectively, with $n$ denoting sequence length and $d$ representing feature dimension. To Leverage the compu-

tational efficiency inherent in right matrix multiplication, the above equation can be seamlessly and mathematically equivalently transformed into its linear variant, as dictated by the properties of matrix multiplication:

$$
\mathbf {O} = \operatorname {N o r m} (\mathbf {Q} (\mathbf {K} ^ {\top} \mathbf {V})), \tag {2}
$$

This linear formulation facilitates recurrent prediction with a commendable complexity of $O ( n d ^ { 2 } )$ , rendering it efficient during training relative to sequence length. Furthermore, employing linear attention ensures a constant computation complexity of $O ( d ^ { 2 } )$ irrespective of sequence length, thereby enabling inference over unlimited long sequences. This achievement is realized by updating $\mathbf { K } ^ { \top } \mathbf { V }$ recurrently without the need for repeated computation of the entire attention matrix. In contrast, the standard softmax attention entails a computational complexity of $O ( m d ^ { 2 } )$ during the inference process, where $m$ denotes the token index.

Nevertheless, when dealing with causal prediction tasks, the effectiveness of the right product is compromised, leading to the requirement for the computation of cumsum (Hua et al., 2022). This impediment hinders the potential for highly efficient parallel computation. Consequently, we persist with the conventional left matrix multiplication in Lightning Attention-1. This serves as the promotion behind the introduction of Lightning Attention-2, specifically crafted to address the challenges associated with the right product in such contexts.

# 3.2. Lightning Attention-2

Lightning Attention-2 employs a tiling methodology throughout its whole computation process. Given the huge variance in memory bandwidth between HBM and SRAM within GPU, Lightning Attention-2 applies a distinct strategy for leveraging them. In each iteration $i$ , matrices $\mathbf { Q } _ { i } , \mathbf { K } _ { i } , \mathbf { V } _ { i }$ undergo segmentation into blocks, subsequently transferred to SRAM for computation. The intra- and interblock operations are segregated, with intra-blocks employing the left product and inter-blocks utilizing the right product. This approach optimally exploits the computational and memory efficiencies associated with the right product, enhancing overall execution speed. The intermediate activation KV is iteratively saved and accumulated within SRAM. Subsequently, the outputs of intra-blocks and interblocks are summed within SRAM, and the results are written back to HBM. This method aims to capitalize on the distinct advantages of each memory component, optimizing the computational workflow. The structural framework of Lightning Attention-2 is well illustrated in Fig. 2.

The intricate details of the Lightning Attention-2 implementation are explicated through Algorithm 1 (forward pass) and Algorithm 2 (backward pass). These algorithms serve to encapsulate the nuanced computational procedures in-

![](images/649aec83b63d72ac683e8ca7023a15562d345d486bbeaec246789a55104f77e5.jpg)  
Figure 2. Structural framework of Lightning Attention-2 is detailed in its algorithmic schematic. During the $i$ -th iteration, the tiling blocks of matrices $\mathbf { Q } _ { i } , \mathbf { K } _ { i } , \mathbf { V } _ { i }$ are transferred from High Bandwidth Memory (HBM) to Static Random-Access Memory (SRAM). Within the SRAM, the outputs $\mathbf { O } _ { \mathrm { { i n t r a } } }$ and $\mathbf { O } _ { \mathrm { i n t e r } }$ are computed independently, followed by an update to the KV matrix. Subsequently, the final output $\mathbf { O } _ { i }$ , which is the sum of $\mathbf { O } _ { \mathrm { { i n t r a } } }$ and $\mathbf { O } _ { \mathrm { i n t e r } }$ , is written back from SRAM to HBM.

tegral to Lightning Attention-2. Additionally, we provide a comprehensive derivation to facilitate a more profound comprehension of Lightning Attention-2. The derivations are systematically presented for both the forward pass and the backward pass, contributing to a thorough understanding of the underlying mechanisms.

# 3.2.1. FORWARD PASS

We ignore the $\operatorname { N o r m } ( { \mathord { \cdot } } )$ operator in eq. (2) to simplify the derivations. During forward pass of Lightning Attention-2, the $t$ -th output can be formulated as

$$
\mathbf {o} _ {t} = \mathbf {q} _ {t} \sum_ {s \leq t} \lambda^ {t - s} \mathbf {k} _ {s} ^ {\top} \mathbf {v} _ {s}. \tag {3}
$$

In a recursive form, the above equation can be rewritten as

$$
\mathbf {k v} _ {0} = 0 \in \mathbb {R} ^ {d \times d},
$$

$$
\mathbf {k} \mathbf {v} _ {t} = \lambda \mathbf {k} \mathbf {v} _ {t - 1} + \mathbf {k} _ {t} ^ {\top} \mathbf {v} _ {t}, \tag {4}
$$

$$
\mathbf {o} _ {t} = \mathbf {q} _ {t} (\mathbf {k v} _ {t}),
$$

where

$$
\mathbf {k} \mathbf {v} _ {t} = \sum_ {s \leq t} \lambda^ {t - s} \mathbf {k} _ {s} ^ {\top} \mathbf {v} _ {s}. \tag {5}
$$

To perform tiling, let us write the equations in block form. Given the total sequence length $n$ and block size $B$ , X is divided into $\begin{array} { r } { T = \frac { n } { B } } \end{array}$ blocks $\{ \mathbf { X } _ { 1 } , \mathbf { X } _ { 2 } , \ldots , \mathbf { X } _ { T } \}$ of size $B \times d$ each, where $\mathbf { X } \in \{ \mathbf { Q } , \mathbf { K } , \mathbf { V } , \mathbf { O } \}$ .

# Algorithm 1 Lightning Attention-2 Forward Pass

Input: $\mathbf { Q } , \mathbf { K } , \mathbf { V } \in \mathbb { R } ^ { n \times d }$ , decay rate $\lambda \in \mathbb { R } ^ { + }$ , block sizes $B$ . Divide X into $\begin{array} { r } { T = { \frac { n } { B } } } \end{array}$ blocks $\mathbf { X } _ { 1 } , \mathbf { X } _ { 2 } , . . . \mathbf { X } _ { T }$ of size $B \times d$ each, where $\mathbf { X } \in \{ \mathbf { Q } , \mathbf { \bar { K } } , \mathbf { V } , \mathbf { O } \}$ .

Initialize mask $\mathbf { M } \in \mathbb { R } ^ { B \times B }$ , where $\mathbf { M } _ { i j } = \lambda ^ { i - j }$ , if $i \geq j$ , else 0.

Initialize $\Lambda = \operatorname { d i a g } \{ \lambda , \lambda ^ { 2 } , \dotsc , \lambda ^ { B } \} \in \mathbb { R } ^ { B \times B }$ .

Initialize $\mathbf { K } \mathbf { V } = 0 \in \mathbb { R } ^ { d \times d }$ .

for $1 \leq i \leq T$ do

Load $\mathbf { Q } _ { i } ^ { - } , \mathbf { K } _ { i } , \mathbf { V } _ { i } \in \mathbb { R } ^ { B \times d }$ from HBM to on-chip SRAM.

On chip, compute $\mathbf { O } _ { \mathrm { i n t r a } } = [ ( \mathbf { Q } _ { i } \mathbf { K } _ { i } ^ { \top } ) \odot \mathbf { M } ] \mathbf { V } _ { i }$

On chip, compute $\mathbf { O } _ { \mathrm { i n t e r } } = \Lambda \mathbf { Q } _ { i } ( \mathbf { K } \mathbf { V } )$ .

On chip, compute $\mathbf { K } \mathbf { V } = \lambda ^ { B } \mathbf { K } \mathbf { V } + ( \lambda ^ { B } \Lambda ^ { - 1 } \mathbf { K } _ { i } ) ^ { \top } \mathbf { V } _ { i }$

Write $\mathbf { O } _ { i } = \mathbf { O } _ { \mathrm { i n t r a } } + \mathbf { O } _ { \mathrm { i n t e r } }$ to HBM as the $_ { i }$ -th block of $\mathbf { o }$ end for

return O.

We first define

$$
\mathbf {K V} _ {0} = \mathbf {0} \in \mathbb {R} ^ {d \times d}, \mathbf {K V} _ {t} = \sum_ {s \leq t B} \lambda^ {t B - s} \mathbf {k} _ {s} ^ {\top} \mathbf {v} _ {s}. \tag {6}
$$

Given $\mathbf { K V } _ { t }$ , the output of $( t + 1 )$ -th block, i.e., $t B + r$ , with $1 \leq r \leq B$ is

$$
\begin{array}{l} \mathbf {0} _ {t B + r} \\ = \mathbf {q} _ {t B + r} \sum_ {s \leq t B + r} \lambda^ {t B + r - s} \mathbf {k} _ {s} ^ {\top} \mathbf {v} _ {s} \\ = \mathbf {q} _ {t B + r} \left(\sum_ {s = t B + 1} ^ {t B + r} \lambda^ {t B + r - s} \mathbf {k} _ {s} ^ {\top} \mathbf {v} _ {s} + \lambda^ {r} \sum_ {s \leq t B} \lambda^ {t B - s} \mathbf {k} _ {s} ^ {\top} \mathbf {v} _ {s}\right) \\ = \mathbf {q} _ {t B + r} \sum_ {s = t B + 1} ^ {t B + r} \lambda^ {t B + r - s} \mathbf {k} _ {s} ^ {\top} \mathbf {v} _ {s} + \lambda^ {r} \mathbf {q} _ {t B + r} \mathbf {k v} _ {t B}. \tag {7} \\ \end{array}
$$

Rewritten in matrix form, we have

$$
\begin{array}{l} \mathbf {O} _ {t + 1} = \underbrace {\left[ \left(\mathbf {Q} _ {t + 1} \mathbf {K} _ {t + 1} ^ {\top}\right) \odot \mathbf {M} \right] \mathbf {V} _ {t + 1}} _ {\text {I n t r a B l o c k}} \tag {8} \\ + \underbrace {\Lambda \mathbf {Q} _ {t + 1} (\mathbf {K V} _ {t})} _ {\text {I n t e r B l o c k}}, \\ \end{array}
$$

where

$$
\begin{array}{l} \mathbf {M} _ {s t} = \left\{ \begin{array}{l l} \lambda^ {s - t} & s \geq t \\ 0 & s <   t \end{array} , \right. \tag {9} \\ \Lambda = \operatorname {d i a g} \{1, \dots , \lambda^ {B - 1} \}. \\ \end{array}
$$

# Algorithm 2 Lightning Attention-2 Backward Pass

Input: $\mathbf { Q } , \mathbf { K } , \mathbf { V } , \mathbf { d O } \in \mathbb { R } ^ { n \times d }$ , decay rate $\lambda \in \mathbb { R } ^ { + }$ , block sizes $B$ .

Divide $\mathbf { X }$ into $\begin{array} { r } { T = { \frac { n } { B } } } \end{array}$ blocks $\mathbf { X } _ { 1 } , \mathbf { X } _ { 2 } , . . . \mathbf { X } _ { T }$ of size $B \times d$ each, where $\mathbf { X } \in \{ \mathbf { Q } , \mathbf { \breve { K } } , \mathbf { V } \}$ .

Divide dX into $\begin{array} { r } { T \ = \ \frac { n } { R } } \end{array}$ blocks $\mathbf { 1 X } _ { 1 } , \mathbf { d X } _ { 2 } , . . . \mathbf { d X } _ { T }$ of size $B \times d$ each, where $\mathbf { X } \in \left\{ \mathbf { Q } , \mathbf { K } , \mathbf { V } , \mathbf { O } \right\}$ .

Initialize mask $\mathbf { M } \in \mathbb { R } ^ { B \times B }$ , where $\mathbf { M } _ { i j } = \lambda ^ { i - j }$ , if $i \geq j$ , else 0.

Initialize $\Lambda = \operatorname { d i a g } \{ \lambda , \lambda ^ { 2 } , \dotsc , \lambda ^ { B } \} \in \mathbb { R } ^ { B \times B }$

Initialize KV = 0, $\mathbf { d K V } = 0 \in \mathbb { R } ^ { d \times d }$

for $i = 1 , \dots , T$ do

Load $\mathbf { K } _ { i } , \mathbf { V } _ { i } , \mathbf { O } _ { i } , \mathbf { d O } _ { i } \ \in \ \mathbb { R } ^ { B \times d }$ from HBM to on-chip SRAM.

On chip, compute ${ \bf d } { \bf Q } _ { \mathrm { i n t r a } } = [ ( { \bf d } { \bf O } _ { i } { \bf V } _ { i } ^ { \top } ) \odot { \bf M } ] { \bf K } _ { i }$

On chip, compute $\mathbf { d Q } _ { \mathrm { i n t e r } } = \Lambda \mathbf { d O } _ { i } ( \mathbf { K V } ) ^ { \top }$

On chip, compute $\mathbf { K } \mathbf { V } = \lambda ^ { B } \mathbf { K } \mathbf { V } + ( \lambda ^ { B } \Lambda ^ { - 1 } \mathbf { K } _ { i } ) ^ { \top } \mathbf { V } _ { i }$

Write $\mathbf { d Q } _ { i } = \mathbf { d Q } _ { \mathrm { { i n t r a } } } + \mathbf { d Q } _ { \mathrm { { i n t e r } } }$ to HBM as the $_ { i }$ -th block of dQ.

end for

for $i = T , \dots , 1$ do

Load $\mathbf { Q } _ { i } , \mathbf { K } _ { i } , \mathbf { V } _ { i } , \mathbf { O } _ { i } , \mathbf { d O } _ { i } \in \mathbb { R } ^ { B \times d }$ from HBM to on-chip SRAM.

On chip, compute $\mathbf { d K } _ { \mathrm { { i n t r a } } } = [ ( \mathbf { d O } _ { i } \mathbf { V } _ { i } ^ { \top } ) \odot \mathbf { M } ] ^ { \top } \mathbf { Q } _ { i }$

On chip, compute $\mathbf { d K } _ { \mathrm { i n t e r } } = \left( \lambda ^ { B } \Lambda ^ { - 1 } \mathbf { V } _ { i } \right) ( \mathbf { d K V } ) ^ { \top }$

On chip, compute ${ \bf d } { \bf V } _ { \mathrm { i n t r a } } = \left[ ( { \bf Q } _ { i } { \bf K } _ { i } ^ { \top } ) \odot \mathbf { M } \right] ^ { \top } { \bf d } { \bf O } _ { i }$

On chip, compute $\mathbf { d } \mathbf { V } _ { \mathrm { i n t e r } } = ( \lambda ^ { B } \Lambda ^ { - 1 } \mathbf { K } _ { i } ) \mathbf { d } \mathbf { K } \mathbf { V } .$

On chip, compute $\mathbf { d K V } = \lambda ^ { B } \mathbf { d K V } + \left( \Lambda \mathbf { Q } _ { i } \right) ^ { \top } \mathbf { d O } _ { i }$

Write $\mathbf { d K } _ { i } = \mathbf { K } _ { \mathrm { i n t r a } } + \mathbf { K } _ { \mathrm { i n t e r } }$ , $\mathbf { d V } _ { i } = \mathbf { V } _ { \mathrm { i n t r a } } + \mathbf { V } _ { \mathrm { i n t e r } }$ to HBM as the $i$ -th block of dK, dV.

end for

return dQ, dK, dV.

And the KV at $( t + 1 )$ -th block can be written as

$$
\begin{array}{l} \mathbf {K V} _ {t + 1} = \sum_ {s \leq (t + 1) B} \lambda^ {(t + 1) B - s} \mathbf {k} _ {s} ^ {\top} \mathbf {v} _ {s} \\ = \lambda^ {B} \sum_ {s \leq t B} \lambda^ {t B - s} \mathbf {k} _ {s} ^ {\top} \mathbf {v} _ {s} + \sum_ {s = t B + 1} ^ {(t + 1) B} \lambda^ {(t + 1) B - s} \mathbf {k} _ {s} ^ {\top} \mathbf {v} _ {s} \\ = \lambda^ {B} \mathbf {K V} _ {t} + \left(\operatorname {d i a g} \{\lambda^ {B - 1}, \dots , 1 \} \mathbf {K} _ {t}\right) ^ {\top} \mathbf {V} _ {t} \\ = \lambda^ {B} \mathbf {K} \mathbf {V} _ {t} + \left(\lambda^ {B} \boldsymbol {\Lambda} ^ {- 1} \mathbf {K} _ {t}\right) ^ {\top} \mathbf {V} _ {t}. \tag {10} \\ \end{array}
$$

The complete expression of the forward pass of Lightning Attention-2 can be found in Algorithm 1.

# 3.2.2. BACKWARD PASS

For backward pass, let us consider the reverse process. First given $\mathbf { d o } _ { t }$ , we have

$$
\begin{array}{l} \mathbf {d q} _ {t} = \mathbf {d o} _ {t} (\mathbf {k v} _ {t}) ^ {\top} \in \mathbb {R} ^ {1 \times d}, \\ \mathbf {d} \mathbf {k} _ {t} = \mathbf {v} _ {t} (\mathbf {d} \mathbf {k} \mathbf {v} _ {t}) ^ {\top} \in \mathbb {R} ^ {1 \times d}, \\ \mathbf {d} \mathbf {v} _ {t} = \mathbf {k} _ {t} (\mathbf {d} \mathbf {k} \mathbf {v} _ {t}) \in \mathbb {R} ^ {1 \times d}, \tag {11} \\ \mathbf {d k v} _ {t} = \sum_ {s \geq t} \lambda^ {s - t} \mathbf {q} _ {s} ^ {\top} \mathbf {d o} _ {s} \in \mathbb {R} ^ {d \times d}. \\ \end{array}
$$

By writing $\mathbf { d k } { \mathbf { v } } _ { t }$ in a recursive form, we get

$$
\begin{array}{l} \mathbf {d k v} _ {n + 1} = 0 \in \mathbb {R} ^ {d \times d}, \\ \mathbf {d k v} _ {t - 1} = \lambda \mathbf {d k v} _ {t} + \mathbf {q} _ {t - 1} ^ {\top} \mathbf {d o} _ {t - 1}. \tag {12} \\ \end{array}
$$

To facilitate the understanding of tiling, let us consider the above equations in block style. Given the total sequence length $n$ and block size B, X is divided into $\begin{array} { r } { T \ = \ \frac { n } { B } } \end{array}$ blocks $\{ \mathbf { X } _ { 1 } , \mathbf { X } _ { 2 } , \ldots , \mathbf { X } _ { T } \}$ of size $B \times d$ each, where $\bar { \mathbf { X } } \in \{ \mathbf { Q } , \mathbf { K } , \mathbf { V } , \mathbf { O } , \mathbf { d O } \}$ .

We first define

$$
\begin{array}{l} \mathbf {d K V} _ {T + 1} = \mathbf {0} \in \mathbb {R} ^ {d \times d}, \\ \mathbf {d} \mathbf {K V} _ {t} = \sum_ {s > t B} \lambda^ {s - t B} \mathbf {q} _ {s} ^ {\top} \mathbf {d o} _ {s}. \tag {13} \\ \end{array}
$$

Then for the $( t + 1 )$ -th block, i.e., $t B + r , 0 \leq r < B$ , we have

$$
\begin{array}{l} \mathbf {d q} _ {t B + r} \\ = \mathbf {d o} _ {t B + r} \sum_ {s \leq t B + r} \lambda^ {t B + r - s} \mathbf {v} _ {s} ^ {\top} \mathbf {k} _ {s} \\ = \mathbf {d o} _ {t B + r} \left(\sum_ {s = t B + 1} ^ {t B + r} \lambda^ {t B + r - s} \mathbf {v} _ {s} ^ {\top} \mathbf {k} _ {s} + \lambda^ {r} \sum_ {s \leq t B} \lambda^ {t B - s} \mathbf {v} _ {s} ^ {\top} \mathbf {k} _ {s}\right) \\ = \mathbf {d o} _ {t B + r} \sum_ {s = t B + 1} ^ {t B + r} \lambda^ {t B + r - s} \mathbf {v} _ {s} ^ {\top} \mathbf {k} _ {s} + \lambda^ {r} \mathbf {d o} _ {t B + r} \mathbf {k v} _ {t B} ^ {\top}. \tag {14} \\ \end{array}
$$

In matrix form, we have

$$
\begin{array}{l} \mathbf {d} \mathbf {Q} _ {t + 1} = \underbrace {\left[ \left(\mathbf {d} \mathbf {O} _ {t + 1} \mathbf {V} _ {t + 1} ^ {\top}\right) \odot \mathbf {M} \right] \mathbf {K} _ {t + 1}} _ {\text {I n t r a B l o c k}} \\ + \underbrace {\Lambda \mathbf {d} \mathbf {O} _ {t + 1} \left(\mathbf {K V} _ {t} ^ {\top}\right)} _ {\text {I n t e r B l o c k}}. \tag {15} \\ \end{array}
$$

Since the recursion of $\mathbf { d K } _ { t }$ steps from $t + 1$ to $t$ , given $\mathbf { K V } _ { t + 1 }$ , $\mathbf { d K } _ { t }$ for the $t { \cdot }$ -th block, i.e., at positions $( t - 1 ) B +$ $r , 0 < r \leq B$ is

$$
\begin{array}{l} \mathbf {d k} _ {(t - 1) B + r} \\ = \mathbf {v} _ {(t - 1) B + r} \sum_ {s \geq (t - 1) B + r} \lambda^ {s - (t - 1) B - r} \mathbf {d o} _ {s} ^ {\top} \mathbf {q} _ {s} \\ = \mathbf {v} _ {(t - 1) B + r} \left(\sum_ {s = (t - 1) B + r} ^ {t B} \lambda^ {t B + r - s} \mathbf {d o} _ {s} ^ {\top} \mathbf {q} _ {s}\right) \tag {16} \\ + \mathbf {v} _ {(t - 1) B + r} \left(\lambda^ {B - r} \sum_ {s > t B} \lambda^ {s - t B} \mathbf {d o} _ {s} ^ {\top} \mathbf {q} _ {s}\right) \\ = \mathbf {v} _ {(t - 1) B + r} \sum_ {s = (t - 1) B + r} ^ {t B} \lambda^ {t B + r - s} \mathbf {d o} _ {s} ^ {\top} \mathbf {q} _ {s} \\ + \lambda^ {B - r} \mathbf {v} _ {(t - 1) B + r} d \mathbf {K V} _ {t} ^ {\top}. \\ \end{array}
$$

In matrix form, we get

$$
\begin{array}{l} \mathbf {d} \mathbf {K} _ {t - 1} = \underbrace {\left[ \left(\mathbf {d} \mathbf {O} _ {t - 1} \mathbf {V} _ {t - 1} ^ {\top}\right) \odot \mathbf {M} \right] ^ {\top} \mathbf {Q} _ {t - 1}} _ {\text {I n t r a B l o c k}} \tag {17} \\ + \underbrace {\lambda^ {B} \Lambda^ {- 1} \mathbf {V} _ {t - 1} (\mathbf {d K V} _ {t} ^ {\top})} _ {\text {I n t e r B l o c k}}. \\ \end{array}
$$

Considering $\mathbf { d V } _ { t }$ for the $t$ -th block, i.e., at positions $( t -$ $1 ) B + r , 0 < r \leq B$ , we have

$$
\begin{array}{l} \mathbf {d v} _ {(t - 1) B + r} \\ = \mathbf {k} _ {(t - 1) B + r} \sum_ {s \geq (t - 1) B + r} \lambda^ {s - (t - 1) B - r} \mathbf {q} _ {s} ^ {\top} \mathbf {d o} _ {s} \\ = \mathbf {k} _ {(t - 1) B + r} \left(\sum_ {s = (t - 1) B + r} ^ {t B} \lambda^ {t B + r - s} \mathbf {q} _ {s} ^ {\top} \mathbf {d o} _ {s}\right) \tag {18} \\ + \lambda^ {B - r} \left(\sum_ {s > t B} \lambda^ {s - t B} \mathbf {q} _ {s} ^ {\top} \mathbf {d o} _ {s}\right) \\ = \mathbf {k} _ {(t - 1) B + r} \sum_ {s = (t - 1) B + r} ^ {t B} \lambda^ {t B + r - s} \mathbf {q} _ {s} ^ {\top} \mathbf {d o} _ {s} \\ + \lambda^ {B - r} \mathbf {k} _ {(t - 1) B + r} \mathrm {d} \mathbf {K V} _ {t}. \\ \end{array}
$$

In matrix form, we get

$$
\begin{array}{r} \mathbf {d V} _ {t - 1} = \underbrace {\left[ \left(\mathbf {Q} _ {t - 1} \mathbf {K} _ {t - 1} ^ {\top}\right) \odot \mathbf {M} \right] ^ {\top} \mathbf {d O} _ {t}} _ {\text {I n t r a B l o c k}} \\ + \underbrace {\lambda^ {B} \Lambda^ {- 1} \mathbf {K} _ {t - 1} (\mathbf {d K V} _ {t})} _ {\text {I n t e r B l o c k}}. \end{array} \tag {19}
$$

Finally, the recursive relation for $\mathbf { d K V } _ { t }$ is

$$
\begin{array}{l} \mathbf {d} \mathbf {K V} _ {t} = \sum_ {s > t B} \lambda^ {s - t B} \mathbf {q} _ {s} ^ {\top} \mathbf {d o} _ {s} \\ = \lambda^ {B} \sum_ {s > (t + 1) B} \lambda^ {s - (t + 1) B} \mathbf {q} _ {s} ^ {\top} \mathbf {d o} _ {s} \tag {20} \\ + \sum_ {s = t B + 1} ^ {(t + 1) B} \lambda^ {s - t B} \mathbf {q} _ {s} ^ {\top} \mathbf {d o} _ {s} \\ = \lambda^ {B} \mathbf {d} \mathbf {K} \mathbf {V} _ {t + 1} + (\Lambda \mathbf {Q} _ {t}) ^ {\top} \mathbf {d} \mathbf {O} _ {t}. \\ \end{array}
$$

Algorithm 2 describes the backward pass of Lightning Attention-2 in more detail.

Discussion A recent method, GLA (Yang et al., 2023) models sequences using linear attention with data-dependent decay. Its chunk-wise Block-Parallel Algorithm employs tiling and IO-aware concepts. However, unlike Lightning Attention-2, it uses parallel computations for each block, which leads to higher memory usage. Retnet (Sun et al., 2023b) is very similar in structure to TransNormerLLM (Qin

et al., 2023b) and uses the chunk-wise retention algorithm. This algorithm is comparable to the forward pass of Lightning Attention-2 but does not consider IO-aware or the backward pass.

# 4. Experiments

To comprehensively assess Lightning Attention-2’s performance, speed, and memory utilization, we conducted extensive experiments on the TransNormerLLM model, with Lightning Attention-2 integrated. Our implementation utilizes the Metaseq framework (Zhang et al., 2022), a PyTorchbased sequence modeling framework (Paszke et al., 2019). All experiments are executed on the GPU cluster featuring 128 A100 80G GPUs. The deployment of Lightning Attention-2 is implemented in Triton (Tillet et al., 2019).

# 4.1. Attention Module Evaluation

We conducted a comparison of speed and memory usage among attention modules Lightning Attention-1, Lightning Attention-2, and FlashAttention-2, all under a single A100 80G GPU. As depicted in Figure 3, the analysis focuses on the runtime, measured in milliseconds, for the separated forward and backward propagation. The baseline runtime demonstrates a quadratic growth relative to the sequence length. In contrast, Lightning Attention-2 exhibits a markedly superior performance with linear growth. Notably, as the sequence length increases, this disparity in runtime becomes increasingly apparent. In addition to speed enhancements, our method also maintains a significant advantage in memory usage with the increase in sequence length.

# 4.2. Lightning Attention-2 in Large Language Model

Table 2. Language Modeling Comparison between TransNormer-LLM with Lightning Attention-1 and Lightning Attention-2.   

<table><tr><td>Model</td><td>Attention</td><td>Params</td><td>Updates</td><td>Loss</td></tr><tr><td>TNL-LA1</td><td>LA1</td><td>0.4B</td><td>100k</td><td>2.229</td></tr><tr><td>TNL-LA2</td><td>LA2</td><td>0.4B</td><td>100k</td><td>2.228</td></tr></table>

Performance Evaluation In Table 2, we evaluated the performance of the TransNormerLLM-0.4B model under 2K contexts, comparing two variants: one equipped with Lightning Attention-1 and the other with Lightning Attention-2. These experiments were carried out using $8 \times \mathbf { A } 1 0 0$ 80G GPUs. After 100,000 iterations, using the sampled corpus from our corpus with 300B tokens and initial seed, we observed a marginal performance difference. Specifically, the variant with Lightning Attention-2 demonstrated a performance decrement of 0.001 compared to its counterpart with Lightning Attention-1.

Furthermore, our analysis extended to benchmarking the top-tier efficient large language models, including LLaMA-

![](images/966767889550aaf26ce446ee8d04aefa999fdc30c909ba245e5211ffa0b45deb.jpg)

![](images/a6d33e1b01aa1068dfa4dfad7c43487e17923d625d4104225afce300ca3b1f1d.jpg)

![](images/89da0a0dd315deba43cd14a6ed776dfbaccd6102997698dcb3328e7a7655a613.jpg)

![](images/6796186f9a102fcb3e72ced5d66ba5deac714b674f15e0053fd4e5baf70c9b12.jpg)  
Figure 3. Comparative Analysis of Speed and Memory Usage: FlashAttention vs. Lightning Attention. Upper Section: Runtime in milliseconds for the forward and backward pass across varying sequence lengths. Lower Section: Memory utilization during the forward and backward pass at different sequence lengths.

Table 1. Efficiency Comparison of LLaMA with FlashAttention2, TransNormerLLM with Lightning Attention-1, and TransNormerLLM with Lightning Attention-2. The statistical analysis was performed using $2 \times \mathbf { A } 1 0 0$ 80G GPUs. The table reports Tokens per GPU per Second (TGS) across three different model sizes, within context ranges spanning from 1K to 92K. OOM stands for out of GPU memory.   

<table><tr><td>Model</td><td>PS</td><td>1024</td><td>2048</td><td>4096</td><td>8192</td><td>16384</td><td>32768</td><td>65536</td><td>81920</td><td>94208</td></tr><tr><td>LLaMA-FA2</td><td>0.4B</td><td>35931</td><td>32453</td><td>28184</td><td>21996</td><td>15479</td><td>9715</td><td>5643</td><td>4604</td><td>4078</td></tr><tr><td>TNL-LA1</td><td>0.4B</td><td>41789</td><td>39043</td><td>34894</td><td>28627</td><td>21112</td><td>13852</td><td>8247</td><td>6824</td><td>6012</td></tr><tr><td>TNL-LA2</td><td>0.4B</td><td>38615</td><td>38680</td><td>38714</td><td>38172</td><td>37755</td><td>37364</td><td>38278</td><td>38457</td><td>38596</td></tr><tr><td>LLaMA-FA2</td><td>1B</td><td>14897</td><td>13990</td><td>12644</td><td>10887</td><td>8468</td><td>5836</td><td>3820</td><td>3167</td><td>OOM</td></tr><tr><td>TNL-LA1</td><td>1B</td><td>21195</td><td>20128</td><td>18553</td><td>16012</td><td>12594</td><td>8848</td><td>5611</td><td>4625</td><td>OOM</td></tr><tr><td>TNL-LA2</td><td>1B</td><td>20052</td><td>19967</td><td>20009</td><td>19841</td><td>19805</td><td>19691</td><td>20077</td><td>20186</td><td>OOM</td></tr><tr><td>LLaMA-FA2</td><td>3B</td><td>7117</td><td>6708</td><td>6008</td><td>4968</td><td>3755</td><td>2558</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>TNL-LA1</td><td>3B</td><td>8001</td><td>7649</td><td>7117</td><td>6152</td><td>4859</td><td>3512</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>TNL-LA2</td><td>3B</td><td>7524</td><td>7593</td><td>7599</td><td>7559</td><td>7545</td><td>7545</td><td>OOM</td><td>OOM</td><td>OOM</td></tr></table>

FA2 (Touvron et al., 2023a; Dao, 2023), TNL-LA2, HGRN (Qin et al., 2023d), and TNN (Qin et al., 2023a). This benchmarking focused on training loss using a 30B subset of our uniquely assembled corpus, scaling from 1 to 3 billion parameters. As depicted in Figure 4, the TNL-LA2 model achieved marginally lower loss compared to the other models under review in both 1B and 3B parameters.

Efficiency Evaluation In Table 1, we present a comparative analysis of training speeds under the same corpora and hardware setups. This comparison encompasses three variants: TransNormerLLM with Lightning Attention-2 (TNL-LA2), TransNormerLLM with Lightning Attention-1 (TNL-LA1), and LLaMA with FlashAttention2 (LLaMA-FA2). Our findings show that during both the forward and back-

ward passes, the TGS (tokens per GPU per second) for TNL-LA2 remains consistently high, while the other two models exhibit a rapid decline when the sequence length is scaled from 1K to 92K. This pattern suggests that Lightning Attention-2 offers a significant advancement in managing unlimited sequence lengths in LLM.

# 4.3. Benchmarking Lightning Attention-2 in Large Language Model

To evaluate the performance of the Lightning Attention-2, we conducted an analysis of the TransNormerLLM-15B (Qin et al., 2023b), a model comprising 15 billion parameters. The TransNormerLLM-15B is characterized by its 42 layers, 40 attention heads, and an overall embed-

![](images/c7414b026460d7a8e11cba9db0114469197578e5fa8162146676e5014dc81111.jpg)  
Figure 4. Performance Comparison of HGRN, TNN, LLaMA with FlashAttention2 and TransNormerLLM with Lightning Attention-2. For the 1B model, we used $1 6 \times \mathrm { A } 8 0 0$ 80G GPUs with a batch size of 12 per GPU; for the 3B model, we scaled up to $3 2 \times \mathrm { A 8 0 0 } 8 0 \mathrm { G }$ GPUs and a batch size of 30 per GPU. The training context length was set to 2K.

Table 3. Performance Comparison on Commonsense Reasoning and Aggregated Benchmarks. TNL-LA2: TransNormerLLM with Lightning Attention-2. PS: parameter size (billion). T: tokens (billion). HS: HellaSwag. WG: WinoGrande.   

<table><tr><td>Model</td><td>PS</td><td>T</td><td>BoolQ</td><td>PIQA</td><td>HS</td><td>WG</td><td>ARC-e</td><td>ARC-c</td><td>OBQA</td><td>CSR</td><td>C-Eval</td><td>MMLU</td><td>C-Eval</td><td>MMLU</td></tr><tr><td></td><td>B</td><td>B</td><td>acc</td><td>acc</td><td>acc_norm</td><td>acc</td><td>acc</td><td>acc_norm</td><td>acc_norm</td><td>avg.</td><td>acc-0shot</td><td>acc-0shot</td><td>acc-5shot</td><td>acc-5shot</td></tr><tr><td>Pythia</td><td>12</td><td>50.3</td><td>62.14</td><td>71.76</td><td>51.89</td><td>55.64</td><td>59.22</td><td>28.75</td><td>32.80</td><td>51.74</td><td>22.36</td><td>25.80</td><td>21.43</td><td>26.10</td></tr><tr><td>TNL-LA2</td><td>15</td><td>49.8</td><td>62.08</td><td>72.52</td><td>55.55</td><td>57.14</td><td>62.12</td><td>31.14</td><td>32.40</td><td>53.28</td><td>25.55</td><td>26.60</td><td>26.18</td><td>27.50</td></tr><tr><td>Pythia</td><td>12</td><td>100.6</td><td>62.20</td><td>73.23</td><td>58.83</td><td>59.35</td><td>63.76</td><td>31.91</td><td>32.80</td><td>54.58</td><td>24.00</td><td>24.80</td><td>24.45</td><td>24.40</td></tr><tr><td>TNL-LA2</td><td>15</td><td>99.7</td><td>63.98</td><td>74.70</td><td>61.09</td><td>61.33</td><td>65.95</td><td>34.64</td><td>35.60</td><td>56.76</td><td>26.70</td><td>26.90</td><td>25.38</td><td>27.40</td></tr></table>

ding dimension of 5120. The model will be trained on a corpus of more than 1.3 trillion tokens with a sequence length of 6,144. Notably, the model achieved a processing speed of 1,620 tokens per GPU per second. Given that the comprehensive pre-training phase is scheduled to span three months, we hereby present the most recent results from the latest checkpoint for inclusion in Table 3.

This evaluation is conducted using the lm-evaluationharness framework (Gao et al., 2023). Our benchmark focuses on two key areas: Commonsense Reasoning (CSR) and Multiple Choice Questions (MCQ). For comparative analysis, we also evaluated the Pythia-12B (Biderman et al., 2023) model under the same benchmarks.

Commonsense Reasoning We report BoolQ (Clark et al., 2019), PIQA (Bisk et al., 2019), SIQA (Sap et al., 2019), HellaSwag (Zellers et al., 2019), WinoGrande (Sakaguchi et al., 2019), ARC easy and challenge (Clark et al., 2018), OpenBookQA (Mihaylov et al., 2018) and their average. In all CSR tasks, the performance of TransNormerLLM-15B surpassed Pythia-12B by about $2 \%$ . Furthermore, TransNormerLLM-15B-100B showed an approximate $3 . 5 \%$ improvement over its 50 billion-token stage, especially in the HellaSwag task, with over a $5 \%$ performance increase.

Aggregated Benchmarks We report the overall results for MMLU (Hendrycks et al., 2021) and C-Eval (Huang et al., 2023) with both 0-shot and 5-shot settings. In the C-Eval tasks, TransNormerLLM-15B is about $2 \%$ higher than Pythia-12B. In the 0-shot and 5-shot tests in both Chinese

(C-Eval) and English (MMLU), TransNormerLLM-15B’s performance also exceeded the $2 5 \%$ baseline (the probability of random selection in a 4-choice scenario). We also noticed fluctuations in the 5-shot MCQ tasks, with an average MCQ score of around $2 6 . 5 \%$ .

# 5. Conclusion

In this paper, we introduced Lightning Attention-2, a pioneering implementation of linear attention that effectively harnesses its theoretical computational advantages, particularly in the causal setting. Our approach, which adopts the concepts of "divide and conquer" and tiling techniques, successfully addresses the limitations of current linear attention algorithms, especially the challenges associated with cumulative summation. By separating the computation into intrablock and inter-block components, we effectively leverage GPU hardware to its fullest potential, ensuring efficiency. Our extensive experiments across various model sizes and sequence lengths demonstrate that Lightning Attention-2 not only maintains consistent training speeds regardless of input sequence length but also outperforms existing state-ofthe-art attention mechanisms in terms of speed and accuracy. This breakthrough has profound implications for the future of large language models, particularly those requiring the processing of long sequences. Looking ahead, we intend to introduce sequence parallelism in conjunction with Lightning Attention-2, which aims to facilitate the training of extra-long sequences, effectively overcoming existing hardware constraints.

# Acknowledgement

This work is partially supported by the National Key R&D Program of China (NO.2022ZD0160100). We thank Songlin Yang for the helpful discussions.

# References

Biderman, S., Schoelkopf, H., Anthony, Q., Bradley, H., O’Brien, K., Hallahan, E., Khan, M. A., Purohit, S., Prashanth, U. S., Raff, E., Skowron, A., Sutawika, L., and van der Wal, O. Pythia: A suite for analyzing large language models across training and scaling, 2023.   
Bisk, Y., Zellers, R., Bras, R. L., Gao, J., and Choi, Y. Piqa: Reasoning about physical commonsense in natural language, 2019.   
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems, 33: 1877–1901, 2020.   
Chen, S., Wong, S., Chen, L., and Tian, Y. Extending context window of large language models via positional interpolation, 2023.   
Chi, T.-C., Fan, T.-H., Ramadge, P. J., and Rudnicky, A. I. Kerple: Kernelized relative positional embedding for length extrapolation, 2022.   
Chi, T.-C., Fan, T.-H., Rudnicky, A. I., and Ramadge, P. J. Dissecting transformer length extrapolation via the lens of receptive field analysis, 2023.   
Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlós, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, L., Belanger, D., Colwell, L. J., and Weller, A. Rethinking attention with performers. ArXiv, abs/2009.14794, 2020.   
Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. Boolq: Exploring the surprising difficulty of natural yes/no questions, 2019.   
Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge, 2018.   
Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.   
Dao, T., Fu, D. Y., Ermon, S., Rudra, A., and Ré, C. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022.

Gao, L., Tow, J., Abbasi, B., Biderman, S., Black, S., DiPofi, A., Foster, C., Golding, L., Hsu, J., Le Noac’h, A., Li, H., McDonell, K., Muennighoff, N., Ociepa, C., Phang, J., Reynolds, L., Schoelkopf, H., Skowron, A., Sutawika, L., Tang, E., Thite, A., Wang, B., Wang, K., and Zou, A. A framework for few-shot language model evaluation, 12 2023. URL https://zenodo.org/records/ 10256836.   
Hao, D., Mao, Y., He, B., Han, X., Dai, Y., and Zhong, Y. Improving audio-visual segmentation with bidirectional generation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2024.   
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., and Steinhardt, J. Measuring massive multitask language understanding, 2021.   
Hua, W., Dai, Z., Liu, H., and Le, Q. V. Transformer quality in linear time. arXiv preprint arXiv:2202.10447, 2022.   
Huang, Y., Bai, Y., Zhu, Z., Zhang, J., Zhang, J., Su, T., Liu, J., Lv, C., Zhang, Y., Lei, J., Fu, Y., Sun, M., and He, J. C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models, 2023.   
Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are rnns: Fast autoregressive transformers with linear attention. In International Conference on Machine Learning, pp. 5156–5165. PMLR, 2020a.   
Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are rnns: Fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13- 18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pp. 5156–5165. PMLR, 2020b. URL http://proceedings.mlr.press/ v119/katharopoulos20a.html.   
Ke, G., He, D., and Liu, T.-Y. Rethinking positional encoding in language pre-training. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=09-528y2Fgf.   
Li, J., Li, D., Xiong, C., and Hoi, S. BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In Chaudhuri, K., Jegelka, S., Song, L., Szepesvari, C., Niu, G., and Sabato, S. (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 12888–12900. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr. press/v162/li22n.html.   
Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In arXiv, 2023a.

Li, W., Li, D., Li, W., Wang, Y., Jie, H., and Zhong, Y. MAP: Low-data regime multimodal learning with adapter-based pre-training and prompting. In Breitholtz, E., Lappin, S., Loaiciga, S., Ilinykh, N., and Dobnik, S. (eds.), Proceedings of the 2023 CLASP Conference on Learning with Small Data (LSD), pp. 185–190, Gothenburg, Sweden, September 2023b. Association for Computational Linguistics. URL https://aclanthology.org/ 2023.clasp-1.19.   
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In arXiv, 2023.   
Lu, K., Liu, Z., Wang, J., Sun, W., Qin, Z., Li, D., Shen, X., Deng, H., Han, X., Dai, Y., and Zhong, Y. Linear video transformer with feature fixation, 2022.   
Mao, Y., Zhang, J., Xiang, M., Zhong, Y., and Dai, Y. Multimodal variational auto-encoder based audio-visual segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 954–965, October 2023.   
Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? a new dataset for open book question answering, 2018.   
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al. Pytorch: An imperative style, high-performance deep learning library. Advances in neural information processing systems, 32, 2019.   
Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., Cao, H., Cheng, X., Chung, M., Derczynski, L., Du, X., Grella, M., Gv, K., He, X., Hou, H., Kazienko, P., Kocon, J., Kong, J., Koptyra, B., Lau, H., Lin, J., Mantri, K. S. I., Mom, F., Saito, A., Song, G., Tang, X., Wind, J., Wo´zniak, S., Zhang, Z., Zhou, Q., Zhu, J., and Zhu, R.-J. RWKV: Reinventing RNNs for the transformer era. In Bouamor, H., Pino, J., and Bali, K. (eds.), Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 14048–14077, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp. 936. URL https://aclanthology.org/2023. findings-emnlp.936.   
Peng, H., Pappas, N., Yogatama, D., Schwartz, R., Smith, N. A., and Kong, L. Random feature attention. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021. OpenReview.net, 2021. URL https://openreview. net/forum?id=QtTKTdVrFBB.   
Press, O., Smith, N., and Lewis, M. Train short, test long: Attention with linear biases enables input length extrapo-

lation. In International Conference on Learning Representations, 2022. URL https://openreview.net/ forum?id=R8sQPpGCv0.   
Qin, Z., Han, X., Sun, W., Li, D., Kong, L., Barnes, N., and Zhong, Y. The devil in linear transformer. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 7025–7041, Abu Dhabi, United Arab Emirates, December 2022a. Association for Computational Linguistics. URL https:// aclanthology.org/2022.emnlp-main.473.   
Qin, Z., Sun, W., Deng, H., Li, D., Wei, Y., Lv, B., Yan, J., Kong, L., and Zhong, Y. cosformer: Rethinking softmax in attention. In International Conference on Learning Representations, 2022b. URL https: //openreview.net/forum?id=Bl8CQrx2Up4.   
Qin, Z., Han, X., Sun, W., He, B., Li, D., Li, D., Dai, Y., Kong, L., and Zhong, Y. Toeplitz neural network for sequence modeling. In The Eleventh International Conference on Learning Representations, 2023a. URL https: //openreview.net/forum?id $=$ IxmWsm4xrua.   
Qin, Z., Li, D., Sun, W., Sun, W., Shen, X., Han, X., Wei, Y., Lv, B., Yuan, F., Luo, X., et al. Scaling transnormer to 175 billion parameters. arXiv preprint arXiv:2307.14995, 2023b.   
Qin, Z., Sun, W., Lu, K., Deng, H., Li, D., Han, X., Dai, Y., Kong, L., and Zhong, Y. Linearized relative positional encoding. Transactions on Machine Learning Research, 2023c.   
Qin, Z., Yang, S., and Zhong, Y. Hierarchically gated recurrent neural network for sequence modeling. In NeurIPS, 2023d.   
Qin, Z., Zhong, Y., and Deng, H. Exploring transformer extrapolation. In Proceedings of the AAAI Conference on Artificial Intelligence, 2024.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. In arXiv, 2021.   
Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale, 2019.   
Sap, M., Rashkin, H., Chen, D., LeBras, R., and Choi, Y. Socialiqa: Commonsense reasoning about social interactions, 2019.

Shen, X., Li, D., Zhou, J., Qin, Z., He, B., Han, X., Li, A., Dai, Y., Kong, L., Wang, M., Qiao, Y., and Zhong, Y. Finegrained audible video description. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 10585–10596, June 2023.   
Su, J., Lu, Y., Pan, S., Wen, B., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.   
Sun, W., Qin, Z., Deng, H., Wang, J., Zhang, Y., Zhang, K., Barnes, N., Birchfield, S., Kong, L., and Zhong, Y. Vicinity vision transformer. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(10):12635–12649, 2023a. doi: 10.1109/TPAMI.2023.3285569.   
Sun, Y., Dong, L., Huang, S., Ma, S., Xia, Y., Xue, J., Wang, J., and Wei, F. Retentive network: A successor to transformer for large language models, 2023b.   
Tillet, P., Kung, H.-T., and Cox, D. D. Triton: an intermediate language and compiler for tiled neural network computations. Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, 2019.   
Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., and Lample, G. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.   
Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Ferrer, C. C., Chen, M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., Fuller, B., Gao, C., Goswami, V., Goyal, N., Hartshorn, A., Hosseini, S., Hou, R., Inan, H., Kardas, M., Kerkez, V., Khabsa, M., Kloumann, I., Korenev, A., Koura, P. S., Lachaux, M.-A., Lavril, T., Lee, J., Liskovich, D., Lu, Y., Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog, I., Nie, Y., Poulton, A., Reizenstein, J., Rungta, R., Saladi, K., Schelten, A., Silva, R., Smith, E. M., Subramanian, R., Tan, X. E., Tang, B., Taylor, R., Williams, A., Kuan, J. X., Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A., Kambadur, M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S., and Scialom, T. Llama 2: Open foundation and fine-tuned chat models, 2023b.   
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks, 2023.

Yang, S., Wang, B., Shen, Y., Panda, R., and Kim, Y. Gated linear attention transformers with hardware-efficient training, 2023.   
Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence?, 2019.   
Zhang, S., Roller, S., Goyal, N., Artetxe, M., Chen, M., Chen, S., Dewan, C., Diab, M., Li, X., Lin, X. V., Mihaylov, T., Ott, M., Shleifer, S., Shuster, K., Simig, D., Koura, P. S., Sridhar, A., Wang, T., and Zettlemoyer, L. Opt: Open pre-trained transformer language models, 2022.   
Zheng, L., Wang, C., and Kong, L. Linear complexity randomized self-attention mechanism. In International Conference on Machine Learning, pp. 27011–27041. PMLR, 2022.   
Zheng, L., Yuan, J., Wang, C., and Kong, L. Efficient attention via control variates. In International Conference on Learning Representations, 2023. URL https:// openreview.net/forum?id=G-uNfHKrj46.   
Zhou, J., Shen, X., Wang, J., Zhang, J., Sun, W., Zhang, J., Birchfield, S., Guo, D., Kong, L., Wang, M., and Zhong, Y. Audio-visual segmentation with semantics, 2023.
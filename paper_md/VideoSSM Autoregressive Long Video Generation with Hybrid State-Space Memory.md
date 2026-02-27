# VideoSSM: Autoregressive Long Video Generation with Hybrid State-Space Memory

Yifei $\mathbf { V } \mathbf { u } ^ { 1 , \dagger , * }$ , Xiaoshan $\mathbf { W } \mathbf { u } ^ { 1 , * }$ , Xinting $\mathbf { H } \mathbf { u } ^ { 1 }$ , Tao $\mathbf { H } \mathbf { u } ^ { 2 , \ P }$ , Yang-Tian $\mathbf { S u n ^ { 1 } }$ , Xiaoyang Lyu1, Bo Wang1, Lin Ma2, Yuewen $\mathbf { M } \mathbf { a } ^ { 2 , \ddagger }$ , Zhongrui Wang3,‡, Xiaojuan $\mathbf { Q } \mathbf { i } ^ { 1 , \ddag }$ 1HKU 2PICO, ByteDance 3SUSTech † Work done during internship at PICO ∗ Equal contribution ¶ Project lead ‡ Corresponding author

![](images/ed8228382ee91d3e81a1aa37535656d5bbfa83d7bb9eb709058a489b7dbd8651.jpg)

![](images/81da632e06f8c335e574619c1fbdde0838000bc914b9c953a78fc3253bfc7fb2.jpg)

![](images/8e9989b2f54d9066654564eca194b206f74f9e4bc653e301e133c49f64f38741.jpg)  
Figure 1. We introduce VideoSSM, an AR video diffusion model equipped with a novel hybrid memory architecture that combines a causal sliding-window local lossless cache with an SSM-based global compressed memory. Compared with prior AR Diffusion Transformers (AR DiTs) that use only window attention, which suffer from quality degradation and temporal drifting, or add sink frames, which reduce drift but cause content repetition and a lack of dynamism, our hybrid memory yields videos that remain both long-term consistent and progressively dynamic. Trained via Self Forcing distillation [25] via DMD loss [59] from a bidirectional teacher, VideoSSM supports highly stable long video generation and adaptive, interactive prompt-based video generation.

# Abstract

Autoregressive (AR) diffusion enables streaming, interactive long-video generation by producing frames causally, yet maintaining coherence over minute-scale horizons remains challenging due to accumulated errors, motion drift, and content repetition. We approach this problem from a memory perspective, treating video synthesis as a recurrent dynamical process that requires coordinated shortand long-term context. We propose VideoSSM, a Long Video Model that unifies AR diffusion with a hybrid statespace memory. The state-space model (SSM) serves as an evolving global memory of scene dynamics across the entire sequence, while a context window provides local memory for motion cues and fine details. This hybrid design preserves global consistency without frozen, repetitive patterns, supports prompt-adaptive interaction, and scales in linear time with sequence length. Experiments on short-

and long-range benchmarks demonstrate state-of-the-art temporal consistency and motion stability among autoregressive video generator especially at minute-scale horizons, enabling content diversity and interactive promptbased control, thereby establishing a scalable, memoryaware framework for long video generation.

# 1. Introduction

Long-video generation is a longstanding goal for advancing generative visual intelligence [7, 41, 57]. Beyond short-clip synthesis, the objective is to simulate evolving visual worlds with persistent identity and temporal coherence– powering applications in digital storytelling, robotics simulation, and world modeling. Despite rapid progress from largescale diffusion–transformer architectures [14, 36], current systems are fundamentally limited by short temporal con-

text and the quadratic cost of full attention, which constrain scalability and impede prompt-adaptive updates in real time [29, 45, 58].

Autoregressive (AR) diffusion offers an alternative– it creates frames causally and enables streaming, interactive synthesis. However, pushing AR diffusion to minute- or hour-long horizons exposes persistent bottlenecks: error accumulation, motion drift, and content repetition [25, 31, 57]. Recent work addresses these issues by narrowing the train–test gap and enlarging the context. Self-Forcing [25] performs short AR rollouts during training, aligning the model with its own predictions via rolling key–value (KV) caches as short-term memory to suppress immediate drift. Rolling-Forcing [31] extends this with context rolling and cache reuse to propagate information over longer windows. LongLive [57] introduces a global attention-sink that reuses the earliest frames as fixed anchors, improving long-term stability. Yet these designs remain narrow in temporal adaptability: by repeatedly attending to static early frames, they freeze global memory, which encourages scene looping and repetition over long sequences.

Humans naturally rely on dynamic memory– continuously updating past experience as new events unfold– to reason coherently over long timescales. Inspired by this, recent world-model research [24, 30, 37, 53, 61] augments video simulators with spatial memory. For example, VMem [30] builds a surfel-indexed 3D view memory, while WorldMem [53] and Context-as-Memory [61] maintain cached scene tokens indexed by 3D camera poses for interactive generation. In contrast, long-video generation seeks open-ended, perceptually realistic synthesis over minutes, typically with free camera motion and highly dynamic scenes– without an explicit world state. Consequently, such 3D-coupled memory designs transfer poorly to long-horizon, free-view settings. What’s missing is a continuously updated global memory in latent space– one that jointly captures local motion and global scene evolution without explicit 3D assumptions.

We introduce VideoSSM, an autoregressive long video diffusion model equipped with a hybrid memory architecture: a causal sliding-window local cache as the local memory and a State-Space Model (SSM) compressed global memory. Our design treats video generation as a recurrent dynamical process, where the compressed state via SSM continuously evolves with each generative step to retain and update the holistic scene state. To complement, the context window serves as local lossless memory, capturing temporal motion cues and visual details. This hybrid memory structure enables the model to balance stability and adaptability– maintaining coherence across long horizons while dynamically responding to newly emerged content. Unlike the static attention-sink [31, 57], which fixes early-frame tokens as history anchors, VideoSSM updates

memory continuously to preserve global context– avoiding freezing global memory and content repetition. Overall, it yields long-range coherence, content diversity, and interactive adaptability with linear-time cost scalability.

Extensive experiments on VBench [26] as short- and long-range benchmarks show that VideoSSM achieves state-of-the-art temporal consistency and motion stability among popular autoregressive video generators. At minutescale horizons, it substantially reduces cumulative error, motion drift, and content repetition. Interactive evaluations with prompt switching further demonstrate smoother transitions, fewer residual semantics, and higher user preference via our user study. Together, these results position VideoSSM as a scalable, memory-aware framework for long-context and interactive video generation.

# 2. Related Work

# 2.1. Autoregressive Video Generation

Different from methods adopting bidirectional attention to simultaneously generate all video frames [3, 19–21, 23, 34, 35, 58], AR video generation methods enable streaming and frame-wise video synthesis [25, 41, 60], making them particularly suitable for real-time and interactive prompt control [53, 57]. Similar to the AR paradigm in large language models, some methods first discretize videos into spatiotemporal tokens and then train next-token or next-block predictors [4, 16, 28, 39, 48–50, 55]. During inference, spatiotemporal tokens are generated sequentially to compose a complete video. Other some works modify the diffusion objective by assigning different noise levels to individual frames during training [6, 43], where the current frame is noisier than previous ones. They thereby achieve AR-style generation by feeding synthesized frames back as context during inference [7, 17, 42, 60].

# 2.2. Long Video Generation

To extend diffusion models to longer durations, some works generate overlapping clips with temporal constraints or adopt hierarchical keyframe-to-interpolation pipelines [9, 18, 44, 47, 50], which remains computation-heavy for long videos. Training-free extrapolation methods [32, 33, 38, 64] extend video length at inference by adjusting positional encodings, noise schedules, or temporal frequencies, yet unsuitable for real-time generation. AR models have shown strong scalability for variable-length and real-time generation [6–8, 17, 22, 35, 40, 42, 63]. To mitigate long-term drift, AR-diffusion methods adopt train–test alignment via rollout with KV caching [25] and DMD loss [59], extend context through KV rolling or recaching [12, 57], and integrate attention with sink tokens [31]. Yet when extend to minute-scale, these methods still suffer from severe content repetition.

![](images/3f1aa0e98715c8d2df69bbfc153c667afe106b9cca28a42b5f58b1b62786dbe5.jpg)

![](images/122ceba24088710443c421c427d5b6ce2e6524da32bfcb95913d03956fd5c375.jpg)

![](images/c40e110e21029db822250794eba6aaf076b8d8142b814e2c9c7d9bd10686aa1e.jpg)  
Figure 2. Comparison of DiT block architectures for autoregressive video generation. (a) Standard DiT block with full self-attention, which supports long-context modeling but lacks causality and streaming capability. (b) AR DiT block with masked causal attention, enabling autoregressive and streaming generation at the cost of weakened long-context consistency. (c) Our AR DiT block with a hybrid memory module and router, which combines local causal attention with a learnable global memory to achieve causal generation, streaming, and long-context support.

# 2.3. Memory Mechanism in Generative Models

Memory in generative models can be categorized into local and global. Local memory is typically implemented via sliding-window attention and a KV cache [10, 17, 62] but inherently loses information outside the window [63]. Global memory aggregates history outside the local window. Some methods utilize early-frame for attention sinks to provide a persistent reference [31, 52, 54, 57], but they tend to freeze the global state and cause content repetition. Other approaches select key historical tokens from a memory pool [5] or compress the history into a reusable implicit state to maintain long-range context [15, 35, 37]. Specifically, geometric memory, used in recent world models, solves long-term consistency with explicit 3D structures and precise camera control [24, 30, 51, 61]. While showing potential for interactive simulation with viewpoint revisits, the 3D-coupled memory transfers poorly to open-ended, long-horizon, and free-view synthesis. In this work, we introduce a SSM that functions as a dynamic, continuously evolving global memory.

# 3. Preliminary: From DiT to AR DiT

Diffusion Transformers for Video. We begin by reviewing the standard Diffusion Transformer (DiT) formulation for video generation. Given a clean video sample $\mathbf { x } _ { 0 } \sim$ $q ( \mathbf { x } _ { 0 } )$ , the forward diffusion process gradually corrupts $\mathbf { x } _ { \mathrm { 0 } }$ via a Markov chain:

$$
q \left(\mathbf {x} _ {t} \mid \mathbf {x} _ {t - 1}\right) = \mathcal {N} \left(\mathbf {x} _ {t}; \sqrt {1 - \beta_ {t}} \mathbf {x} _ {t - 1}, \beta_ {t} \mathbf {I}\right), \tag {1}
$$

where $\beta _ { t }$ denotes the noise schedule. The reverse process learns to approximate the true posterior using a denoiser pa-

rameterized by a noise-prediction network $\epsilon _ { \theta }$ , trained with the standard objective:

$$
\mathcal {L} = \mathbb {E} _ {\mathbf {x} _ {0}, \boldsymbol {\epsilon}, t} \| \boldsymbol {\epsilon} - \boldsymbol {\epsilon} _ {\theta} (\mathbf {x} _ {t}, t) \| ^ {2}, \mathbf {x} _ {t} = \sqrt {\bar {\alpha} _ {t}} \mathbf {x} _ {0} + \sqrt {1 - \bar {\alpha} _ {t}} \boldsymbol {\epsilon}, \tag {2}
$$

with $\begin{array} { r } { \bar { \alpha } _ { t } = \prod _ { s = 1 } ^ { t } ( 1 - \beta _ { s } ) } \end{array}$ . In standard DiTs (Fig. 2 (a)), $\epsilon _ { \theta }$ is implemented by a bidirectional vision Transformer that operates on spatiotemporal tokens obtained from video frames, and each token attends to all others through full selfattention.

Autoregressive DiT for Long Video. To enable longhorizon video generation, DiT is converted into a causal autoregressive AR model (Fig. 2 (b)). Let $\mathbf { c } _ { t }$ denote a conditioning signal derived from past frames. The denoiser then becomes conditional, $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t , \mathbf { c } _ { t } )$ , and the Transformer backbone is upgraded to a causal AR DiT by restricting self-attention along the temporal sequence, so that the new frame can only attend to previous frame tokens. At inference time, the model generates video in an autoregressive manner: each newly synthesized frame is fed back to update $\mathbf { c } _ { t }$ , so that future frames are conditioned only on previously generated content rather than ground-truth frames.

However, pushing AR diffusion to minute- or hour-long horizons leads to error accumulation, motion drift, and content repetition. In this work, we augment the AR DiT in Fig. 2 (b) with a hybrid memory architecture (Fig. 2 (c)).

# 4. VideoSSM

We propose VideoSSM, a state-space long video model that augments autoregressive DiT with a hybrid memory architecture for coherent and scalable long-horizon synthesis.

![](images/13035e4015cbc83e1a93285a18edbf856d6e10b631e5ab1156e03262e87d4760.jpg)

![](images/0545417fd6fcd3197d176d33d980131f11cfcbd3408852737deefe5ea8a2cce1.jpg)

![](images/005b46a6b1f39494c9969fe28b5750d77cd016f3c7377f5e1ca5558e5ad87e4f.jpg)

![](images/081e52796303c89d0f067b896a7a224e6ad705dd2321fd4ddf4432942ae0f729.jpg)

![](images/d98a9449df8392d865637ea63efb9c16029b6357ddbc17bfad9ad2eddfcdbce1.jpg)

![](images/25b282cecfeca98df3a22c17c8754f17c7d7948690720ec6dd1e84794678e12b.jpg)

![](images/4db3f9bce50d893d16becc0526c35e712a1e6a7b28450ce336b6297d889c0db8.jpg)

![](images/36a0fc0a468b2be4ac4fa9e91ac18451f29df26c42b8de4763429265a4ad80b4.jpg)

![](images/13f5c8340fed87050a9c87fd79cd57bf2b9db88fb523e3872aad7589b9325ea1.jpg)  
Figure 3. Illustration of attention mechanisms in AR DiT. Let $T$ be the video token length and $L$ the sliding-window size. (a) Causal Attention: Each query attends to all past tokens. It captures the full context with quadratic $\mathrm { O } ( \mathrm { T } ^ { 2 } )$ complexity, impractical for long sequences. (b) Window Attention: Localized attention within a local sliding window. It enables efficient O(TL) complexity for streaming but causes information drift as early tokens are evicted. (c) Attention Sink: Adds fixed initial “sink” tokens to the window. It improves long-range consistency with O(TL) complexity, but the static memory leads to repetitive generation and fails to adapt to new content (d) Ours (Hybrid Memory): Augments window attention with a learnable memory that compresses evicted tokens. This maintains O(TL) efficiency while providing a dynamic global context, balancing long-term consistency and adaptability.

Section 4.1 analyzes the limitations of existing AR architectures by examining their attention and caching strategies (Fig. 3). Section 4.2 introduces our hybrid memory design, which combines a learnable global memory with local sliding-window context. Section 4.3 presents a causal distillation framework that efficiently transfers knowledge from a bidirectional teacher to our causal AR model.

# 4.1. Motivations

In an AR DiT model, causal attention enables temporally streaming generation but requires maintaining a KV cache whose size increases linearly with the number of past tokens (Fig. 3(a)). To control memory growth, many systems adopt sliding-window attention with a rolling KV cache [52, 54] (Fig. 3(b)). This design is memory- and latency-efficient but suffers from drifting and error accumulation when early tokens are evicted [52]. To stabilize long-range dependencies, attention-sink mechanisms preserve a small set of initial “sink” tokens (Fig. 3(c)). In long-video generation, the earliest frames are often used as sink tokens [31, 57], combined with recent tokens to ensure stable attention computation.

However, our experiments show that sink-based attention frequently produces repetitive content or frozen generation patterns, especially in long videos. The fixed sink tokens over-stabilize the KV cache, overshadowing the contributions of the evolving sliding-window context.

# 4.2. Hybrid State-Space Memory

We propose VideoSSM, an autoregressive diffusion model equipped with a hybrid memory architecture (Fig. 3(d)). Rather than discarding out-of-window tokens or relying on fixed attention sinks, we introduce a dedicated memory

module that explicitly manages both short-term and longterm information.

Inspired by the hierarchical structure of human memory [1, 2, 11]– where working memory retains fine details and long-term memory stores compressed, abstract representations, we decompose model memory into two complementary components and integrate them into the attention mechanism:

• Local Memory. A causal attention window with cached KV states (Fig. 3(b)) that preserves precise, lossless representations, essential for capturing fine-grained motion and appearance details (Sec. 4.2.1).   
• Global Memory. An evolving memory module (Fig. 2(c)) that absorbs tokens evicted from the local window and recurrently compresses them into a compact, fixed-size state (Fig. 3(d)), providing a continuously updated summary of all past context (Sec. 4.2.2).

As illustrated in Fig. 3(d), this hybrid attention mechanism allows the model to access the entire history while maintaining $O ( T L )$ complexity and full streaming capability. The design meets the core requirements of modern longvideo generators: it is causal, naturally streamable, and capable of leveraging long-range temporal context efficiently.

# 4.2.1. Local Memory: Sliding Window Self-Attention

Let $L$ be the sliding-window size. Given the input hidden state $\mathbf { H } _ { t } ^ { \mathrm { i n } }$ for the current frame $t$ , queries $( \mathbf { Q } _ { t } )$ , keys $( \mathbf { K } _ { t } )$ and values $\left( \mathbf { V } _ { t } \right)$ are computed as:

$$
\left\{\mathbf {Q} _ {t}, \mathbf {K} _ {t}, \mathbf {V} _ {t} \right\} = \left\{\mathbf {H} _ {t} ^ {\text {i n}} \mathbf {W} _ {Q}, \mathbf {H} _ {t} ^ {\text {i n}} \mathbf {W} _ {K}, \mathbf {H} _ {t} ^ {\text {i n}} \mathbf {W} _ {V} \right\}, \tag {3}
$$

where $\mathbf { K } _ { t }$ and $\mathbf { V } _ { t }$ pairs are appended to the local KV Cache. The cache retains only key-value pairs of the sink-

![](images/e4314512979d604718b81e444a286ffdb2c7bf43791258c302d271f17a78386f.jpg)

![](images/470a93bfbd2a93ef8e37a0d18a77b51e09b01d38e758cb548152aa228471b956.jpg)  
Figure 4. Illustration of how sink, evicted, and window tokens are arranged at different timesteps in a causal DiT with slidingwindow attention. Here window length $L = 3$ .

ing token and the $L$ most recent tokens, forming ${ \bf K } _ { t } ^ { \mathrm { l o c a l } } =$ $[ \mathbf { K } _ { \mathrm { s i n k } } , \mathbf { K } _ { t - L + 1 } : \mathbf { K } _ { t } ]$ and $\mathbf { V } _ { t } ^ { \mathrm { l o c a l } } = [ \mathbf { V } _ { \mathrm { s i n k } } , \mathbf { V } _ { t - L + 1 } : \mathbf { V } _ { t } ]$ . $\mathbf { H } _ { t } ^ { \mathrm { l o c a l } }$ based on local memory is computed with a standard causal self-attention mechanism:

$$
\mathbf {H} _ {t} ^ {\text {l o c a l}} = \operatorname {S e l f A t t e n t i o n} \left(\mathbf {Q} _ {t}, \mathbf {K} _ {t} ^ {\text {l o c a l}}, \mathbf {V} _ {t} ^ {\text {l o c a l}}\right). \tag {4}
$$

# 4.2.2. Global Memory: Dynamic State Computation

The global memory module compresses the full history of out-of-window (evicted) tokens into a fixed-size, continuously evolving representation. It operates through four key components: gate caching, state updates, memory retrieval, and output gating.

Synchronized Gate Caching. To integrate information from an evicted token into the global memory, we maintain a recurrent compressed state governed by two learnable gates at each timestep $t$ : an injection gate $\beta _ { t }$ and a decay gate $\pmb { \alpha } _ { t }$ . $\alpha _ { t } , \beta _ { t } \in \mathbb { R } ^ { d }$ , both matching the dimensionality $d$ of latent tokens. The injection gate controls how strongly the evicted token should update the global state, while the decay gate determines how quickly past memory should fade. Both gates are computed from the hidden state $\mathbf { H } _ { t } ^ { \mathrm { i n } }$ before the token exits the local window:

$$
\boldsymbol {\beta} _ {t} = \sigma \left(\mathbf {W} _ {\beta} \mathbf {H} _ {t} ^ {\text {i n}}\right) \tag {5}
$$

$$
\boldsymbol {\alpha} _ {t} = - \exp (\mathbf {A}) \cdot \operatorname {S o f t P l u s} \left(\mathbf {W} _ {\alpha} \mathbf {H} _ {t} ^ {\text {i n}} + \mathbf {B}\right) \tag {6}
$$

where $\sigma$ is the sigmoid function and SoftPlus is a smooth variant of the ReLU activation function; A, $\mathbf { W } _ { \alpha }$ , and $\mathbf { W } _ { \beta }$ are learnable weights; and B is learnable bias. The gates $\alpha _ { t } , \beta _ { t }$ are stored in a Gates Cache that is updated in sync with the rolling KV cache.

Global Memory State Update. For a query at time $t$ , tokens outside the $L$ -length window, indexed by $[ \sinh \mathbf { k } + 1 : t -$ $L ]$ are considered evicted. We denoted $\{ \mathbf { K } , \mathbf { V } , \alpha , \beta \} _ { t } ^ { \mathrm { e v t } } =$ $\arg [ \{ { \bf K } , { \bf V } , \alpha , \beta \} _ { \mathrm { s i n k + 1 } } : \{ { \bf K } , { \bf V } , \alpha , \beta \} _ { t - L } ]$ . A summary of all information up to the most recently evicted token is compacted into the global state ${ { \bf { M } } _ { t } }$ . We update $\mathbf { M } _ { t }$ using the Gated $\Delta$ -rule [56], which extracts only the novel

![](images/fc26c1d88754fff6ba52d83cb026fcd0079a03132fda028ebd6cc9809b0ba81f.jpg)  
Figure 5. Architecture of the proposed hybrid memory module. The input $H _ { t } ^ { \mathrm { i n } }$ is processed in two streams. The local path (top) uses windowed attention with a sliding KV cache to compute $H _ { t } ^ { \mathrm { l o c a l } }$ . The global path (bottom) uses a State-Space Model (SSM) to recurrently compress historical information into a memory state $M$ , which is retrieved to produce $H _ { t } ^ { \mathrm { g l o b a l } }$ . A router then dynamically fuses the local and global outputs.

component of the incoming information before integrating it into the state:

$$
\mathbf {V} _ {\text {n e w}, t} ^ {\text {e v t}} = \mathbf {V} _ {t} ^ {\text {e v t}} - \operatorname {P r e d i c t} \left(\mathbf {M} _ {t - 1}, \mathbf {K} _ {t} ^ {\text {e v t}}, \boldsymbol {\beta} _ {t} ^ {\text {e v t}}\right),
$$

$$
\mathbf {M} _ {t} = \exp (\bar {\mathbf {g}} _ {t}) \cdot \mathbf {M} _ {t - 1} + \mathbf {K} _ {t} ^ {\mathrm {e v t}} \cdot \left(\mathbf {V} _ {\text {n e w}, t} ^ {\mathrm {e v t}}\right) ^ {T},
$$

where ${ \bf M } _ { 0 } = { \bf 0 }$ , Predict(·) estimates the predictable portion of the evicted value from the previous state $\mathbf { M } _ { t - 1 }$ , so that $\mathbf { V } _ { \mathrm { n e w } , t } ^ { \mathrm { e v t } }$ retains only the unpredictable, novel component of the input, and $\begin{array} { r } { \bar { \bf g } _ { t } = \sum _ { s = 0 } ^ { t } \alpha _ { s } ^ { \mathrm { e v t } } } \end{array}$ Pt s is a cumulative negative gate (with $\alpha _ { s } ^ { \mathrm { e v t } } < 0 $ ) that controls state decay. The first term, $\exp ( \bar { \bf g } _ { t } ) \cdot { \bf M } _ { t - 1 }$ , ensures controlled forgetting and long-term stability, while the second term integrates only the unpredictable, new information. By combining selective update and gated decay, the global memory maintains a compact, stable, and continuously evolving representation that grows with the video rather than collapsing or drifting– providing robust long-range context for AR diffusion.

Global Memory Retrieval. We retrieve a query-aligned memory response by projecting the current query $\mathbf { Q } _ { t }$ onto the compressed state ${ { \bf { M } } _ { t } }$ , which is then refined by an output gate that regulates how much global information is exposed to the backbone. The final global context $\mathbf { H } _ { t } ^ { \mathrm { g l o b a l } }$ is computed as:

$$
\mathbf {g} _ {t} ^ {\text {o u t}} = \operatorname {L i n e a r} \left(\mathbf {H} _ {t} ^ {\text {i n}}\right),
$$

$$
\mathbf {H} _ {t} ^ {\text {g l o b a l}} = \operatorname {S w i s h} \left(\mathbf {g} _ {t} ^ {\text {o u t}} \odot \operatorname {R M S N o r m} \left(\mathbf {Q} _ {t} \mathbf {M} _ {t}\right)\right), \tag {8}
$$

where $\mathbf { H } _ { t } ^ { \mathrm { i n } }$ is the current hidden state, ${ \bf g } _ { t } ^ { \mathrm { o u t } }$ is a vector of output gate coefficients controlling information flow, and

$\mathrm { S w i s h } ( x ) = x \cdot \sigma ( x )$ denotes the Swish activation. It first normalizes the memory response with RMSNorm, then applies element-wise gating, allowing the model to selectively filter and modulate global information before passing it to subsequent layers.

# 4.2.3. Position-Aware Gated Fusion

Finally, we fuse the local and global memory streams. We introduce a position-aware router that dynamically controls the strength of global memory injection. This is crucial for stable generation: early in the sequence, the model should rely primarily on local information, whereas global memory should play a larger role as more context accumulates.

We define a relative position ratio $\rho _ { t } = ( t + 1 ) / T$ , where $t$ is the current frame index and $T$ is the total context length. A memory gate $\gamma _ { t } \in \mathbb { R } ^ { d }$ , matching the token dimensionality $d$ , is then computed as

$$
\gamma_ {t} = \sigma \left(\boldsymbol {w} _ {\text {r o u t e r}} \log \left(\rho_ {t}\right) + \boldsymbol {b} _ {\text {r o u t e r}}\right), \tag {9}
$$

where $w _ { \mathrm { r o u t e r } } , b _ { \mathrm { r o u t e r } } \in \mathbb { R } ^ { d }$ are learnable vectors. As $t $ 0, we have $\rho _ { t } \ \to \ 0$ , $\log ( \rho _ { t } )  - \infty$ , and thus $\gamma _ { t } \ \to \ 0$ , effectively suppressing the global memory at the beginning of the sequence. As $t  T$ , $\rho _ { t } \to 1$ , $\log ( \rho _ { t } ) \to 0$ , and $\gamma _ { t }$ approaches the learned value $\sigma ( b _ { \mathrm { r o u t e r } } )$ , enabling the global memory stream.

The final fused hidden state $\mathbf { H } _ { t } ^ { \mathrm { f u s e d } }$ is a simple gated sum:

$$
\mathbf {H} _ {t} ^ {\text {f u s e d}} = \mathbf {H} _ {t} ^ {\text {l o c a l}} + \gamma_ {t} \cdot \mathbf {H} _ {t} ^ {\text {g l o b a l}}, \tag {10}
$$

which is then passed to the subsequent cross-attention and feed-forward layers of the DiT block. This fusion mechanism allows VideoSSM to retain the $O ( L )$ efficiency of sliding-window attention while incorporating a dynamic, compressed global context—mitigating both the catastrophic forgetting of pure windowed attention and the rigidity of static attention sinks, and yielding dynamics that are both temporally coherent over long horizons and responsive to evolving scene content.

# 4.3. Training with Memory

To enable real-time generation, we distill a high-fidelity teacher into a causal framework based on Self-Forcing [25]. Our model incorporates a hybrid memory for streaming consistency and uses a rolling memory recipe for efficient long-horizon training and interactive prompt switching.

Stage 1: Causal Model Distillation. We initialize the causal student model $G _ { \theta }$ from a pre-trained bidirectional teacher $T _ { \phi }$ (i.e., Wan 2.1 [45]) following the CausVid strategy [60]. Using the teacher’s ODE sampling trajectories, the student is trained to match the teacher’s short-clip expertise on 5-second segments. The student regresses these trajectories causally, minimizing $\mathcal { L } = \Vert \hat { \mathbf { x } } _ { 0 } - T _ { \phi } ( \mathbf { x } _ { t } , t ) \Vert ^ { 2 }$ , where $\hat { \mathbf { x } } _ { 0 } = G _ { \theta } ( \mathbf { x } _ { t } , t )$ . Gradients are computed selectively

and propagated across steps to the hybrid memory, mitigating exposure bias via self-generated histories. This stage equips $G _ { \theta }$ with high-quality short-term dynamics before introducing long-horizon autoregressive behavior. By incorporating our hybrid memory, the model gains long-range capabilities even trained only on short segments.

Stage 2: Long Video Training. The second stage mitigates long-horizon degradation by training the hybrid memory for effective operation in streaming, autoregressive scenarios. We extend the SF-style distillation using DMD, allowing the student model to self-correct errors during training that mimics inference with rolling KV caches and memory.

1. Long Self-Rollout: $G _ { \theta }$ autoregressively produces a long sequence $\hat { x } ^ { 1 : N }$ (e.g., $N = 6 0$ seconds) in chunks, surpassing the teacher’s 5-second horizon. It fills its local KV Cache and global Memory Cache (via $\beta$ and $\alpha$ gates) exclusively from self-generated outputs, employing a rolling KV cache with global memory to sustain fixed context length and avert drift, thereby emulating inference.   
2. Windowed Teacher Correction: Uniformly sample a short window $K$ $K = 5$ seconds). Apply the DMD loss over this window:

$$
\mathcal {L} _ {\mathrm {D M D}} = \mathbb {E} _ {t, i \sim \operatorname {U n i f} (1, N - K)} \left[ \nabla_ {\theta} K L \left(p _ {\theta , t} ^ {S} \left(z _ {i}\right) \mid \mid p _ {t} ^ {T} \left(z _ {i}\right)\right) \right] \tag {11}
$$

where $z _ { i }$ is the window starting at frame $i$ . This harnesses the teacher’s short-clip proficiency to rectify long-range errors, facilitating recovery from degraded states.

# 5. Experiments

# 5.1. Settings

We implement VideoSSM based on the Wan 2.1-T2V-1.3B model [45], a flow-matching model that generates 5-second videos at 16 FPS with $8 3 2 \times 4 8 0$ resolution. The distilled chunk-wise autoregressive model uses 4-step diffusion and with chunk size of 3, generating a chunk of 3 latent frames at a time. For both ODE initialization and Long Video Distillation, we sample text prompts from a filtered and LLMextended version of VidProM [46].

We evaluate on VBench [26] for short videos (5 seconds) and long videos (minute-long), measuring dimensions including subject consistency, background consistency, motion smoothness, aesthetic quality, and imaging quality. In the user study, we collected preferences from 40 participants. Each participant was shown 8 prompts, and for each prompt, they ranked 4 generated one-minute videos generated with three AR video generation models: Selfforcing [25], CausVid [60], Longlive [57], and ours.

# 5.2. Video Quality Evaluation

We first evaluate short-video (5-second) generation on the VBench benchmark [25]. As shown in Table 1, we compare VideoSSM against leading AR models, including fewstep distilled generators [25, 60] and their long-range variants [12, 31, 57], alongside other strong baselines [7, 13, 27, 42]. For reference, we also list SOTA bidirectional models [20, 45]. VideoSSM achieves the highest Total (83.95) and Quality (84.88) scores among all AR models, outperforming strong competitors like LongLive and the 4.5B-parameter MAGI-1. This demonstrates that our hybrid memory mechanism effectively enhances short-video fidelity.

Table 1. Comparison with relevant baselines. We compare VideoSSM with representative open-source video generation models of similar parameter sizes and resolutions.1   

<table><tr><td rowspan="2">Model</td><td rowspan="2">#Params</td><td colspan="3">Evaluation scores ↑</td></tr><tr><td>Total</td><td>Quality</td><td>Semantic</td></tr><tr><td colspan="5">Bidirectional Diffusion models</td></tr><tr><td>LTX-Video [20]</td><td>1.9B</td><td>80.00</td><td>82.30</td><td>70.79</td></tr><tr><td>Wan2.1 [45]</td><td>1.3B</td><td>84.26</td><td>85.30</td><td>80.09</td></tr><tr><td colspan="5">Autoregressive models</td></tr><tr><td>SkyReels-V2 [7]</td><td>1.3B</td><td>82.67</td><td>84.70</td><td>74.53</td></tr><tr><td>MAGI-1 [42]</td><td>4.5B</td><td>79.18</td><td>82.04</td><td>67.74</td></tr><tr><td>CausVid [60]</td><td>1.3B</td><td>81.20</td><td>84.05</td><td>69.80</td></tr><tr><td>NOVA [13]</td><td>0.6B</td><td>80.12</td><td>80.39</td><td>79.05</td></tr><tr><td>Pyramid Flow [27]</td><td>2B</td><td>81.72</td><td>84.74</td><td>69.62</td></tr><tr><td>Self Forcing [25]</td><td>1.3B</td><td>83.00</td><td>83.71</td><td>80.14</td></tr><tr><td>LongLive [57]</td><td>1.3B</td><td>83.52</td><td>84.26</td><td>80.53</td></tr><tr><td>Self Forcing ++ [12]</td><td>1.3B</td><td>83.11</td><td>83.79</td><td>80.37</td></tr><tr><td>Rolling Forcing [31]</td><td>1.3B</td><td>81.22</td><td>84.08</td><td>69.78</td></tr><tr><td>VideoSSM (Ours)</td><td>1.4B</td><td>83.95</td><td>84.88</td><td>80.22</td></tr></table>

# 5.3. Long Video Generation

To assess long-video capabilities, we evaluate minute-long generations using single prompts on VBench. We evaluate the train-short test-long setting, where models trained on 5- second clips must generalize to long videos, a regime that exposes catastrophic drift. As shown in Table 2, VideoSSM achieves the highest Subject and Background Consistency among all AR models, demonstrating the effectiveness of our hybrid memory in preventing error accumulation. Importantly, this consistency does not collapse into static or frozen outputs, as VideoSSM attains a markedly higher Dynamic Degree (50.50) than LongLive and Self Forcing, demonstrating its ability to maintain long-term coherence while preserving natural temporal evolution.

Figure 6 provides qualitative validation. In the burger example, VideoSSM (Ours) maintains the subject’s identity

Table 2. Performance comparisons of AR models on 60s long videos. Bold highlights the highest, underline the second highest.   

<table><tr><td>Metric</td><td>Self Forcing</td><td>LongLive</td><td>VideoSSM (Ours)</td></tr><tr><td>Temporal Flickering ↑</td><td>97.86</td><td>97.24</td><td>97.70</td></tr><tr><td>Subject Consistency ↑</td><td>88.25</td><td>91.09</td><td>92.51</td></tr><tr><td>Background Consistency ↑</td><td>91.73</td><td>93.23</td><td>93.95</td></tr><tr><td>Motion Smoothness ↑</td><td>98.67</td><td>98.38</td><td>98.60</td></tr><tr><td>Dynamic Degree ↑</td><td>35.00</td><td>37.50</td><td>50.50</td></tr><tr><td>Aesthetic Quality ↑</td><td>60.02</td><td>55.74</td><td>60.45</td></tr></table>

and structure for the full 60 seconds, whereas SkyReels-V2 suffers complete content collapse and Self Forcing exhibits severe drifting. The underwater scene is more revealing: our model successfully captures the dynamic, forward-swimming motion while maintaining high subject consistency. In contrast, competing methods fail this balance. CausVid avoids drift but succumbs to motion stagnation, with the child becoming nearly static in later frames. LongLive, which uses a fixed attention-sink, initially preserves the subject but suffers degradation where it hallucinates a second instance of the boy. This combined evidence proves our hybrid memory’s superior ability to enable stable, high-fidelity, and truly dynamic generation far beyond the short clips seen during training.

# 5.4. Interactive Video Generation

Our model also supports interactive long video generation with seamless prompt switching after long-horizon training. By enabling KV recache[57], the system can efficiently refresh its internal local memory when the user provides new instructions, preventing outdated semantics from lingering and ensuring that each transition responds cleanly to the updated prompt. This allows the model to maintain scene coherence while adapting to new narrative directives in real time. As illustrated in Fig. 7, VideoSSM produces smooth, natural, and dynamically coherent transitions across prompt changes. Additional qualitative examples are provided in the supplementary material.

# 5.5. User Study

To evaluate perceptual qualities beyond automated metrics, we conducted a user study with 40 participants. We generated 32 unique, minute-long videos by running our method and three baselines (LongLive [57], Self Forcing [25], CausVid [60]) on 8 different text prompts. For each prompt, participants were shown the four resulting videos in a randomized order and asked to rank them from 1 (best) to 4 (worst). The ranking criteria included overall visual quality, temporal and physical consistency, and adherence to the prompt. As shown in Table 3, our approach achieved the highest preference. VideoSSM received the most Rank 1

![](images/6a545b5a2239579ea6796f6d3761627c2fed8bb88aa46b0f0a2dab45a7c576d5.jpg)

![](images/bf4e6a4825c4fd9cb2729bb8a2dae1652c0291183da74d1133168a8ec90497f7.jpg)  
Figure 6. Qualitative comparison of 60s-long video generation. Baseline methods with windowed attention often suffer from error accumulation, leading to drifting artifacts, while methods using attention sink may produce repeated content or nearly static scenes. Our approach generates videos with more coherent motion and stable temporal consistency.   
Figure 7. Demonstration of interactive long video generation.

votes $( 4 1 . 0 7 \% )$ and secured the best (lowest) average rank of 1.85. While LongLive also achieves high consistency via a static attention-sink mechanism (low Dynamic Degree 37.50) that may minimize perceptual artifacts, VideoSSM’s higher Dynamic Degree enables complex, non-repetitive motion, yielding superior user preference by balancing dynamic realism with long-term consistency.

Table 3. Vote percentages for each model across different ranks. Cell color intensity indicates higher percentages. The last column shows the average rank.   

<table><tr><td>Model</td><td>Rank 1 (%)</td><td>Rank 2 (%)</td><td>Rank 3 (%)</td><td>Rank 4 (%)</td><td>Avg Rank</td></tr><tr><td>Self Forcing</td><td>11.79</td><td>13.21</td><td>23.21</td><td>51.79</td><td>3.18</td></tr><tr><td>CausVid</td><td>7.50</td><td>16.07</td><td>42.14</td><td>34.29</td><td>3.03</td></tr><tr><td>LongLive</td><td>39.64</td><td>36.43</td><td>15.00</td><td>8.93</td><td>1.92</td></tr><tr><td>Ours</td><td>41.07</td><td>34.29</td><td>19.64</td><td>5.00</td><td>1.85</td></tr></table>

# 6. Conclusion

In this work, we introduced VideoSSM, a long-video autoregressive diffusion model that reframes autoregressive diffusion as a recurrent dynamical process with hybrid memory: an SSM as evolving global memory and a context window as local memory. This design preserves longhorizon consistency while adapting to new content, achieving linear-time scalability. Experiments demonstrate that VideoSSM substantially reduces error accumulation, motion drift, and content repetition, enabling minute-scale coherence and robust prompt-adaptive interactive generation. Future directions include integrating explicit multi-modal conditioning, incorporating camera-aware and geometric priors, and extending the framework to controllable longform video editing.

# References

[1] Richard C Atkinson and Richard M Shiffrin. Human memory: A proposed system and its control processes. In Psychology of learning and motivation, pages 89–195. Elsevier, 1968. 4   
[2] Alan Baddeley. Working memory. Memory, pages 71–111, 2020. 4   
[3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023. arXiv preprint arXiv:2311.15127. 2   
[4] Jake Bruce, Michael Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Luke Hughes, Matthew Lai, Aditi Mavalankar, Chris Todd, Danny Springborn, Yusra Ghazi, Evgeny Gladchenko, James Molloy, Utsav Prabhu, John Nguyen, Matthew Aitchison, Gabriele Recchia, Izzeddin Gur, Rob Fergus, Aleksandra Faust, and Pierre Sermanet. Genie: Generative interactive environments. In ICML, 2024. 2   
[5] Shengqu Cai, Ceyuan Yang, Lvmin Zhang, Yuwei Guo, Junfei Xiao, Ziyan Yang, Yinghao Xu, Zhenheng Yang, Alan Yuille, Leonidas Guibas, Maneesh Agrawala, Lu Jiang, and Gordon Wetzstein. Mixture of contexts for long video generation, 2025. arXiv preprint arXiv:2508.21058. 3   
[6] Boyuan Chen, Diego Mart’i Mons’o, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. In Advances in Neural Information Processing Systems, 2024. 2   
[7] Guibin Chen, Dixuan Lin, Jiangping Yang, Chunze Lin, Junchen Zhu, Mingyuan Fan, Hao Zhang, Sheng Chen, Zheng Chen, Chengcheng Ma, et al. Skyreels-v2: Infinite-length film generative model. arXiv preprint arXiv:2504.13074, 2025. 1, 2, 7   
[8] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025. 2   
[9] Xinyuan Chen, Yaohui Wang, Lingjun Zhang, Shaobin Zhuang, Xin Ma, Jiashuo Yu, Yali Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Seine: Short-to-long video diffusion model for generative transition and prediction. In International Conference on Learning Representations, 2023. 2   
[10] Xinle Cheng, Tianyu He, Jiayi Xu, Junliang Guo, Di He, and Jiang Bian. Playing with transformer at $^ { 3 0 + }$ fps via next-frame diffusion. arXiv preprint arXiv:2506.01380, 2025. 3   
[11] Nelson Cowan. What are the differences between long-term, short-term, and working memory? Progress in brain research, 169:323–338, 2008. 4   
[12] Justin Cui, Jie Wu, Ming Li, Tao Yang, Xiaojie Li, Rui Wang, Andrew Bai, Yuanhao Ban, and Cho-Jui Hsieh. Selfforcing $^ { + + }$ : Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283, 2025. 2, 7

[13] Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169, 2024. 7   
[14] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik ¨ Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis. arXiv preprint arXiv:2403.03206, 2024. 1   
[15] Yunhao Fang, Weihao Yu, Shu Zhong, Qinghao Ye, Xuehan Xiong, and Lai Wei. Artificial hippocampus networks for efficient long-context modeling. arXiv preprint arXiv:2510.07318, 2025. 3   
[16] Songwei Ge, Thomas Hayes, Harry Yang, Xi Yin, Guan Pang, David Jacobs, Jia-Bin Huang, and Devi Parikh. Long video generation with time-agnostic vqgan and timesensitive transformer. In ECCV, 2022. 2   
[17] Yuchao Gu, weijia Mao, and Mike Zheng Shou. Longcontext autoregressive video modeling with next-frame prediction. arXiv preprint arXiv:2503.19325, 2025. 2, 3   
[18] Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. Long context tuning for video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025. 2   
[19] David Ha and Jurgen Schmidhuber. Recurrent world models ¨ facilitate policy evolution. Advances in neural information processing systems, 31, 2018. 2   
[20] Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, et al. Ltx-video: Realtime video latent diffusion, 2024. arXiv preprint arXiv:2501.00103. 7   
[21] Yingqing He, Tianyu Yang, Yong Zhang, Ying Shan, and Qifeng Chen. Latent video diffusion models for high-fidelity long video generation, 2022. arXiv preprint arXiv:2211.13221. 2   
[22] Roberto Henschel, Levon Khachatryan, Daniil Hayrapetyan, Hayk Poghosyan, Vahram Tadevosyan, Zhangyang Wang, Shant Navasardyan, and Humphrey Shi. Streamingt2v: Consistent, dynamic, and extendable long video generation from text, 2024. arXiv preprint arXiv:2403.14773. 2   
[23] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J Fleet. Video diffusion models. In Advances in Neural Information Processing Systems, 2022. 2   
[24] Tianyu Huang, Wangguandong Zheng, Tengfei Wang, Yuhao Liu, Zhenwei Wang, Junta Wu, Jie Jiang, Hui Li, Rynson WH Lau, Wangmeng Zuo, et al. Voyager: Long-range and world-consistent video diffusion for explorable 3d scene generation. arXiv preprint arXiv:2506.04225, 2025. 2, 3   
[25] Xun Huang, Zhengqi Li, Guande He, Mingyuan Zhou, and Eli Shechtman. Self forcing: Bridging the traintest gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009, 2025. 1, 2, 6, 7

[26] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, et al. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 21807–21818, 2024. 2, 6   
[27] Yang Jin, Zhicheng Sun, Ningyuan Li, Kun Xu, Hao Jiang, Nan Zhuang, Quzhe Huang, Yang Song, Yadong Mu, and Zhouchen Lin. Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954, 2024. 7   
[28] Dan Kondratyuk, Lijun Yu, Xiuye Gu, Jose Lezama, ´ Jonathan Huang, Rachel Hornung, Hartwig Adam, Hassan Akbari, Yair Alon, Vighnesh Birodkar, Yong Cheng, Ming-Chang Chiu, Josh Dillon, Irfan Essa, Agrim Gupta, Meera Hahn, Anja Hauth, David Hendon, Alonso Martinez, David Minnen, David Ross, Grant Schindler, Mikhail Sirotenko, Kihyuk Sohn, Krishna Somandepalli, Huisheng Wang, David Yang, Bryan Seybold, and Lu Jiang. Videopoet: A large language model for zero-shot video generation. In ICML, 2024. 2   
[29] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models, 2024. arXiv preprint arXiv:2412.03603. 2   
[30] Runjia Li, Philip Torr, Andrea Vedaldi, and Tomas Jakab. Vmem: Consistent interactive video scene generation with surfel-indexed view memory. arXiv preprint arXiv:2506.18903, 2025. 2, 3   
[31] Kunhao Liu, Wenbo Hu, Jiale Xu, Ying Shan, and Shijian Lu. Rolling forcing: Autoregressive long video diffusion in real time, 2025. arXiv preprint arXiv:2509.25161. 2, 3, 4, 7   
[32] Yu Lu and Yi Yang. Freelong++: Training-free long video generation via multi-band spectralfusion. arXiv preprint arXiv:2507.00162, 2025. 2   
[33] Yu Lu, Yuanzhi Liang, Linchao Zhu, and Yi Yang. Freelong: Training-free long video generation with spectralblend temporal attention. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. 2   
[34] Michael Mathieu, Camille Couprie, and Yann LeCun. Deep multi-scale video prediction beyond mean square error. In International Conference on Learning Representations, 2016. 2   
[35] Yuta Oshima, Shohei Taniguchi, Masahiro Suzuki, and Yutaka Matsuo. Ssm meets video diffusion models: Efficient long-term video generation with structured state spaces, 2024. arXiv preprint arXiv:2403.07711. 2, 3   
[36] William Peebles and Saining Xie. Scalable diffusion models with transformers. arXiv preprint arXiv:2212.09748, 2022.   
[37] Ryan Po, Yotam Nitzan, Richard Zhang, Berlin Chen, Tri Dao, Eli Shechtman, Gordon Wetzstein, and Xun Huang. Long-context state-space video world models. arXiv preprint arXiv:2505.20171, 2025. 2, 3   
[38] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling, 2023. 2

[39] Shuhuai Ren, Shuming Ma, Xu Sun, and Furu Wei. Next block prediction: Video generation via semi-autoregressive modeling. arXiv preprint arXiv:2502.07737, 2025. 2   
[40] Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. History-guided video diffusion. arXiv preprint arXiv:2502.06764, 2025. 2   
[41] Meituan LongCat Team, Xunliang Cai, Qilong Huang, Zhuoliang Kang, Hongyu Li, Shijun Liang, Liya Ma, Siyu Ren, Xiaoming Wei, Rixu Xie, and Tong Zhang. Longcatvideo technical report, 2025. 1, 2   
[42] Hansi Teng, Hongyu Jia, Lei Sun, Lingzhi Li, Maolin Li, Mingqiu Tang, Shuai Han, Tianning Zhang, WQ Zhang, Weifeng Luo, et al. Magi-1: Autoregressive video generation at scale. arXiv preprint arXiv:2505.13211, 2025. 2, 7   
[43] Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, and Jan Kautz. Mocogan: Decomposing motion and content for video generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018. 2   
[44] Ruben Villegas, Mohammad Babaeizadeh, Pieter-Jan Kindermans, Hernan Moraldo, Han Zhang, Mohammad Taghi Saffar, Santiago Castro, Julius Kunze, and Dumitru Erhan. Phenaki: Variable length video generation from open domain textual descriptions. In International Conference on Learning Representations, 2023. 2   
[45] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 2, 6, 7   
[46] Wenhao Wang and Yi Yang. Vidprom: A million-scale real prompt-gallery dataset for text-to-video diffusion models. Advances in Neural Information Processing Systems, 37: 65618–65642, 2024. 6   
[47] Yaohui Wang, Xinyuan Chen, Xin Ma, Shangchen Zhou, Ziqi Huang, Yi Wang, Ceyuan Yang, Yinan He, Jiashuo Yu, Peiqing Yang, et al. Lavie: High-quality video generation with cascaded latent diffusion models. IJCV, 2024. 2   
[48] Yuqing Wang, Tianwei Xiong, Daquan Zhou, Zhijie Lin, Yang Zhao, Bingyi Kang, Jiashi Feng, and Xihui Liu. Loong: Generating minute-level long videos with autoregressive language models. arXiv preprint arXiv:2410.02757, 2024. 2   
[49] Dirk Weissenborn, Oscar Tackstr¨ om, and Jakob Uszkoreit.¨ Scaling autoregressive video models. In ICLR, 2020.   
[50] Chenfei Wu, Jian Liang, Xiaowei Hu, Zhe Gan, Jianfeng Gao, Lijuan Wang, Zicheng Liu, Yuejian Fang, and Nan Duan. Nuwa-infinity: Autoregressive over autoregressive generation for infinite visual synthesis. In NeurIPS, 2022. 2   
[51] Tong Wu, Shuai Yang, Ryan Po, Yinghao Xu, Ziwei Liu, Dahua Lin, and Gordon Wetzstein. Video world models with long-term spatial memory. arXiv preprint arXiv:2506.05284, 2025. 3   
[52] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv, 2023. 3, 4   
[53] Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-

term consistent world simulation with memory. arXiv preprint arXiv:2504.12369, 2025. 2   
[54] Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. Streamingvlm: Real-time understanding for infinite video streams. arXiv preprint arXiv:2510.09608, 2025. 3, 4   
[55] Wilson Yan, Yuchen Zhang, Pieter Abbeel, and Aravind Srinivas. Videogpt: Video generation using vq-vae and transformers, 2021. 2   
[56] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464, 2024. 5   
[57] Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Yingcong Chen, Yao Lu, et al. Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622, 2025. 1, 2, 3, 4, 6, 7   
[58] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, Da Yin, Xiaotao Gu, Yuxuan Zhang, Weihan Wang, Yean Cheng, Ting Liu, Bin Xu, Yuxiao Dong, and Jie Tang. Cogvideox: Text-to-video diffusion models with an expert transformer, 2024. arXiv preprint arXiv:2408.06072. 2   
[59] Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shecht- ¨ man, Fredo Durand, William T Freeman, and Taesung Park. ´ One-step diffusion with distribution matching distillation. In CVPR, 2024. 1, 2   
[60] Tianwei Yin, Qiang Zhang, Richard Zhang, William T Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In CVPR, 2025. 2, 6, 7   
[61] Jiwen Yu, Jianhong Bai, Yiran Qin, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Context as memory: Scene-consistent interactive long video generation with memory retrieval. arXiv preprint arXiv:2506.03141, 2025. 2, 3   
[62] Shangjin Zhai, Zhichao Ye, Jialin Liu, Weijian Xie, Jiaqi Hu, Zhen Peng, Hua Xue, Danpeng Chen, Xiaomeng Wang, Lei Yang, Nan Wang, Haomin Liu, and Guofeng Zhang. Stargen: A spatiotemporal autoregression framework with video diffusion model for scalable and controllable scene generation, 2025. 3   
[63] Lvmin Zhang and Maneesh Agrawala. Packing input frame context in next-frame prediction models for video generation. arXiv preprint arXiv:2504.12626, 2025. 2, 3   
[64] Min Zhao, Guande He, Yixiao Chen, Hongzhou Zhu, Chongxuan Li, and Jun Zhu. Riflex: A free lunch for length extrapolation in video diffusion transformers. arXiv preprint arXiv:2502.15894, 2025. 2
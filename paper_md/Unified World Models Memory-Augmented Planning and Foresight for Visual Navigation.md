# UNIFIED WORLD MODELS: MEMORY-AUGMENTED PLANNING AND FORESIGHT FOR VISUAL NAVIGATION

Yifei $\mathbf { D o n g ^ { 1 * } }$ , Fengyi $\mathbf { W _ { u } } ^ { 1 * }$ , Guangyu Chen1∗, Zhi-Qi Cheng1,†, Qiyu $\mathbf { H } \mathbf { u } ^ { 1 }$ , Yuxuan Zhou1, Jingdong $\mathbf { \bar { s u n } ^ { 2 } }$ , Jun-Yan $\mathbf { H e } ^ { 1 }$ , Qi Dai3, Alexander G. Hauptmann2

1University of Washington 2Carnegie Mellon University 3Microsoft Research

# ABSTRACT

Enabling embodied agents to effectively imagine future states is critical for robust and generalizable visual navigation. Current state-of-the-art approaches, however, adopt modular architectures that separate navigation planning from visual world modeling, leading to state–action misalignment and limited adaptability in novel or dynamic scenarios. To overcome this fundamental limitation, we propose UniWM, a unified, memory-augmented world model integrating egocentric visual foresight and planning within a single multimodal autoregressive backbone. Unlike modular frameworks, UniWM explicitly grounds action decisions in visually imagined outcomes, ensuring tight alignment between prediction and control. A hierarchical memory mechanism further integrates detailed short-term perceptual cues with longer-term trajectory context, enabling stable, coherent reasoning over extended horizons. Extensive experiments across four challenging benchmarks (Go Stanford, ReCon, SCAND, HuRoN) demonstrate that UniWM substantially improves navigation success rates by up to $30 \%$ , significantly reduces trajectory errors compared to strong baselines, and exhibits impressive zero-shot generalization on the unseen TartanDrive dataset. These results highlight UniWM as a principled step toward unified, imagination-driven embodied navigation. Code is available at https://github.com/F1y1113/UniWM.

# 1 INTRODUCTION

Visual navigation is a fundamental capability for embodied AI and autonomous systems Mirowski et al. (2016); Chaplot et al. (2020); Fu et al. (2022); Sridhar et al. (2024), enabling intelligent agents to interpret egocentric visual inputs and sequentially select actions toward goals within complex environments Karnan et al. (2022). This skill underlies critical real-world applications such as robotic delivery, autonomous driving, and assistive technologies, demanding robust perception, precise planning, and the capacity to anticipate environmental evolution resulting from potential actions. Humans naturally excel at such imaginative reasoning, routinely performing mental simulations to plan routes effectively through both familiar and novel scenarios Bar et al. (2025).

Despite rapid progress in visual navigation, existing approaches remain constrained by fundamental limitations (Figs. 1). (a) Direct policy methods (e.g., GNM Shah et al. (2022), VINT Shah et al. (2023), NoMaD Sridhar et al. (2024)) map observations directly to action sequences. Although effective within familiar distributions, such policies are rigidly tied to training data and fail to adapt in novel environments Song et al. (2025). (b) Modular pipelines seek to remedy this by coupling a planner with a separate world model: NavCoT Lin et al. (2024) textualizes future observations, inevitably discarding spatial fidelity, while NWM Bar et al. (2025) employs diffusion models to generate candidate visual rollouts that are subsequently ranked. Yet, when prediction and control are learned in isolation and trajectory memory is absent, state–action misalignment emerges, and errors accumulate under partial observability and extended horizons Ding et al. (2024); Xiao et al. (2025). (c) Unified autoregressive frameworks offer a more principled alternative by interleaving “imagining the next view” with “predicting the next action,” grounding decisions in envisioned

![](images/9fdbc42fea6bb8ee8a1a6fef743aad4f1cedc2abef4c5c5857eb5a2b3ed67ff7.jpg)  
(a) NoMaD

![](images/a86f25ccbb15f21fe231234d5ab4f537ef890778e92c1ac0a8536378389e2170.jpg)  
(b) NWM

![](images/271482daf3058b5af07d56e052adabb011293a8a7238e559c43ecdf083505935.jpg)  
(c) UniWM without Memory

![](images/d61262628acb5fec55d00449955e5d8e7a09bc973b872b26ba0eaecde8400a73.jpg)  
(d) UniWM with Memory   
Figure 1: Comparison of goal-conditioned visual navigation methods. All panels use the same start/goal observations; headers report navigation performance $S R \uparrow$ , ATE↓, and $\mathrm { R P E } \downarrow$ on HuRoN Hirose et al. (2023) dataset. (a) Navigation policy methods like NoMaD Sridhar et al. (2024) directly predict action sequences $A _ { T }$ . (b) World model for navigation like NWM Bar et al. (2025) uses a world model to visualize future observations, enhancing a separate navigation planner. (c) UniWM (no memory) unifies planning and visualization within one multimodal backbone, and actions are grounded in the imagined next observation while generating $A _ { T }$ autoregressively. (d) UniWM (with hierarchical memory) adds intra-step and cross-step memory banks, stabilizing longer-horizon rollouts and consistently yielding the highest SR and lowest errors (ATE/RPE).

outcomes and thereby reducing misalignment (Fig. 1c). However, unification alone cannot halt the gradual drift inherent to longer-horizon reasoning. (d) Hierarchical memory provides the missing inductive bias: by retaining both immediate perceptual cues and longer-range trajectory context, it endows model with temporal coherence, yielding the highest SR and lowest errors in challenging settings (Fig. 1d). In essence, navigation demands not only ability to imagine while acting but also to remember over time. The central challenge, therefore, is to couple planning and imagination within a unified backbone while embedding temporal structure to ensure stable longer-horizon performance.

In response, we propose UniWM, a unified memory-augmented world model that integrates navigation planning and visual imagination within a single multimodal autoregressive backbone (Fig. 2; Sec. 2.1). During training we interleave planner and world-model samples and jointly optimize bin-token classification for actions and reconstruction for images in a shared tokenization space for actions, text, pose, and vision; the framework scales with parameter-efficient fine-tuning such as LoRA (Fig. 2 (a); Sec. 2.2). At inference UniWM alternates between predicting the next action and imagining the next egocentric view, which explicitly grounds control in predicted visual outcomes and mitigates state and action misalignment (Fig. 2 (b); Sec. 2.3). Additionally, a hierarchical twolevel memory that combines an intra-step cache with a cross-step trajectory store augments attention through similarity gating and temporal decay, sustaining coherent longer-horizon rollouts and improving stability (Fig. 3). Together these components unify planning and imagination within one backbone and provide a practical recipe for memory-augmented foresight in visual navigation.

Empirically, UniWM improves Success Rate and reduces ATE and RPE across Go Stanford, ReCon, SCAND, and HuRoN relative to GNM, VINT, NoMaD, Anole-7B, and NWM. For example, on Go Stanford the SR increases from 0.45 to 0.75 (Table 1; Fig. 4). On the unseen TartanDrive, UniWM generalizes without fine-tuning and attains an SR of 0.42 (Table 6; Fig. 6). UniWM also delivers stronger one-step and rollout visualization quality, with higher SSIM and PSNR and lower LPIPS and DreamSim (Table 2). Ablation studies clarify the sources of improvement: reconstruction enhances imagination fidelity and indirectly aids navigation; the bin-token loss directly improves action accuracy; hierarchical memory is essential for longer-horizon stability; and token budget, memory layer selection, and substep interleaving explain the remaining gains (Tables 3, 4, 5; Fig. 5).

In summary, this paper provides the following key contributions:

• Unified architecture. We propose UniWM, the first unified, memory-augmented world model integrating visual navigation planning and imagination within a single multimodal autoregressive backbone, effectively addressing representational fragmentation inherent in modular approaches.   
• Unified training. We propose an end-to-end interleaved training strategy that unifies planner and world-model instances within a single autoregressive backbone, jointly optimizing discretized action prediction and visual reconstruction to tightly align imagination with control.

• Hierarchical memory. We introduce a hierarchical memory mechanism that fuses short-term perceptual details and longer-term trajectory context through similarity-based retrieval and temporal weighting, enabling stable and coherent predictions over extended navigation horizons.   
• Comprehensive validation. Extensive experiments validate UniWM’s significant improvement over state-of-the-art methods across multiple benchmarks, demonstrating superior imagination fidelity, enhanced navigation performance, and robust generalization to novel scenarios.

# 2 UNIWM FRAMEWORK

We present UniWM, a unified, memory-augmented world model that performs planning and visualization within an autoregressive multimodal backbone. We first introduce navigation preliminaries and a unified formulation that replaces the disjoint planner–world-model pair with one multimodal LLM augmented by hierarchical memory (Eq.2, Sec.2.1). We then describe the unified training scheme, i.e. multimodal tokenization and role-specific objectives for planning and world modeling (Sec.2.2), and hierarchical memory for stable longer-horizon rollouts at inference (Sec.2.3).

# 2.1 PRELIMINARIES AND UNIFIED FORMULATION

Given an egocentric RGB observation $o _ { s }$ at the start, the initial agent pose $p _ { 0 } ~ \in ~ \mathbb { R } ^ { 3 }$ (position and yaw), and the goal observation $o _ { g }$ , the agent must predict a sequence of navigation actions $A _ { T } \overset { \cdot } { = } \{ \hat { a } _ { 1 } , \hat { a } _ { 2 } , \dots , \hat { \hat { a } } _ { T } \}$ that leads to the goal Sridhar et al. (2024). Each action $\hat { a } _ { t }$ is either a continuous control command $\left( \mathbf { u } _ { t } , \phi _ { t } \right)$ or a terminal $\operatorname { S t } \operatorname { o p }$ , where ${ \mathbf { u } } _ { t } \in \mathbb { R } ^ { 2 }$ encodes planar translation (forward/backward, left/right) and $\phi _ { t } \in \mathbb { R }$ encodes yaw rotation Bar et al. (2025). Actions are executed sequentially, and agent is required to make monotonic progress toward $o _ { g }$ until issuing Stop.

World Models for Navigation. World models Ha & Schmidhuber (2018) predict future environment states (often represented as image frames or video segments), conditioned on the current state and conditional variables. Formally, this can be written as $\hat { s } _ { t + 1 } = \mathcal { W } ( \hat { s } _ { t } , \mathbf { c } )$ , where $\hat { s } _ { t }$ is the current state, $\hat { s } _ { t + 1 }$ the predicted next state, and $\mathcal { W }$ the learned world model. The conditioning context c may include the executed action $a _ { t }$ , natural-language instructions, history of past observations, or other environmental factors Russell et al. (2025). In navigation, world models serve as imagination engines that anticipate future observations to support action planning. This typically involves two coupled modules Bar et al. (2025): a planner, which selects next action given current observation and goal; and a world model, which simulates the consequent observation conditioned on the chosen action and contextual cues such as start and goal views. Their interaction can be formalized as:

$$
\hat {a} _ {t + 1} = \mathcal {P} \left(\hat {o} _ {t}, o _ {s}, o _ {g}\right), \quad \hat {o} _ {t + 1} = \mathcal {W} \left(\hat {o} _ {t}, \hat {a} _ {t + 1}, o _ {s}, o _ {g}\right) \tag {1}
$$

where $\hat { o } _ { t }$ is the current observation, $\hat { a } _ { t + 1 }$ the action proposed by the planner $\mathcal { P }$ , and $\hat { o } _ { t + 1 }$ the next observation visualized by the world model $\mathcal { W }$ . The start and goal observations $\left( o _ { s } , o _ { g } \right)$ provide the global navigation context. The two modules operate in a closed loop: $\mathcal { P }$ selects $\hat { a } _ { t + 1 }$ conditioned on $\hat { o } _ { t }$ and $\left( o _ { s } , o _ { g } \right)$ , while $\mathcal { W }$ predicts $\hat { o } _ { t + 1 }$ given $\hat { o } _ { t }$ and $\hat { a } _ { t + 1 }$ , which is then fed back into $\mathcal { P }$ . This iterative cycle enables imagination-based planning, allowing agents to simulate prospective action–observation trajectories before execution in the real environment. However, the modular training of $\mathcal { P }$ and $\mathcal { W }$ often leads to state–action misalignment, which degrades performance in complex and partially observable settings Ding et al. (2024). (Refer to Appx. A for more related works.)

Unified World Model with Memory. To overcome these limitations, we replace the modular pair $( \mathcal { P } , \mathcal { W } )$ with a single multimodal backbone, UniWM, that couples planning and visualization. A hierarchical memory bank $\mathcal { M } _ { t }$ , comprising an intra-step $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ and a cross-step $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ , fuses short-term evidence with longer-range context (Fig. 2). At each step, UniWM performs:

$$
\left(\hat {a} _ {t + 1}, \hat {o} _ {t + 1}\right) = \mathbf {U n i W M} \left(\hat {o} _ {t}, o _ {s}, o _ {g}, p _ {0}, \mathcal {M} _ {t}\right), \tag {2}
$$

where UniWM simultaneously functions as a navigation planner and a world model, alternating between these roles to propose actions and visualization outcomes until the goal is reached. Within a unified framework, we instantiate UniWM as a single multimodal large language model (MLLM) $F _ { \theta }$ , which interleaves two substeps at each iteration: (i) action prediction (planner role) and (ii) navigation imagination (world-model role). Both substeps are executed by the same backbone $F _ { \theta }$ , jointly trained on planner and world-model data with tailored objectives (Fig. 2(a), Sec. 2.2). During inference, $F _ { \theta }$ is augmented with a hierarchical memory that integrates immediate evidence with longer-horizon context (Fig. 2(b), Sec. 2.3), ensuring temporally consistent predictions across steps.

![](images/a12a6a56efb2d81e130f26e337bcfb3efa9eb12c3d66b34e5738e98ff8a77949.jpg)  
Figure 2: UniWM framework. (a) Training: planner and world-model samples are interleaved within a single unified multimodal autoregressive backbone, optimized jointly with the discretized bin-token loss ${ \mathcal { L } } _ { \mathrm { p l a n } }$ and the reconstruction loss $\mathcal { L } _ { \mathrm { w o r l d } }$ ; bin/text/image tokenizers map actions, pose, and observations to tokens. (b) Inference: a hierarchical memory supplies intra- and cross-step KV states $( \mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ caches the current observation; $\boldsymbol { \mathcal { M } } _ { t } ^ { \mathrm { c r o s s } }$ accumulates prior steps) to augment attention, yielding robust trajectory-consistent alternating predictions of $\hat { a } _ { t }$ (next action) and $\hat { o } _ { t }$ (next observation). See Fig. 3 for the detailed memory mechanism.

• Navigation Planner (Action Prediction): Given current observation $\hat { o } _ { t }$ , conditioned on start and goal observations $( o _ { s } , o _ { g } )$ , initial pose $p _ { 0 }$ , and memory bank $\mathcal { M } _ { t }$ , $F _ { \theta }$ predicts next action $\hat { a } _ { t + 1 }$ :

$$
\hat {a} _ {t + 1} = F _ {\theta} \left(\hat {o} _ {t}, o _ {s}, o _ {g}, p _ {0}, \mathcal {M} _ {t}\right). \tag {3}
$$

• World Model (Navigation Visualization): Given current observation $\hat { o } _ { t }$ and action $\hat { a } _ { t + 1 }$ , conditioned on $\left( o _ { s } , o _ { g } \right)$ , p0, and $\mathcal { M } _ { t }$ , $F _ { \theta }$ predicts the next observation $\hat { o } _ { t + 1 }$ after executing $\hat { a } _ { t + 1 }$ :

$$
\hat {o} _ {t + 1} = F _ {\theta} \left(\hat {o} _ {t}, \hat {a} _ {t + 1}, o _ {s}, o _ {g}, p _ {0}, \mathcal {M} _ {t}\right). \tag {4}
$$

This design allows $F _ { \theta }$ to act jointly as a navigation planner and a world model, alternating between roles until a terminal $\operatorname { S t } \operatorname { o p }$ is issued. During training, planner and world-model samples are interleaved so that $F _ { \theta }$ learns both behaviors within a single autoregressive framework. At inference, a hierarchical memory bank augments $F _ { \theta }$ by caching key–value states at both intra- and cross-step levels, enabling the integration of immediate observations with longer-range trajectory context. This unified formulation ensures consistent, memory-augmented world modeling throughout navigation.

# 2.2 UNIFIED TRAINING SCHEME

Next, we turn to autoregressive MLLMs that utilize text and image tokens, enabling a unified training scheme for UniWM. We build upon Chameleon and Anole architectures Team (2024); Chern et al. (2024), which integrate a unified Transformer for joint processing of multimodal tokens (Fig. 2 (a)).

Data Preprocessing. Each navigation trajectory yields two complementary sample types aligned with Eqs. 3 and 4. For the navigation planner, a sample consists of $\left( o _ { s } , o _ { g } , o _ { t } , p _ { 0 } \right)$ with target $\hat { a } _ { t + 1 }$ . For the world model, inputs additionally include $a _ { t + 1 }$ and the target is $\hat { o } _ { t + 1 }$ . Visual observations are encoded as <image> placeholders in structured multimodal prompts, using a sliding window to extract multiple samples per trajectory (See Appx. B.1 for prompt examples). During training, samples from both substeps are interleaved in the same batch to encourage shared representations.

Multimodal Tokenization. We employ three tokenizers to unify visual and textual inputs. Following Gafni et al. (2022); Team (2024), a vector-quantized (VQ) image tokenizer discretizes images $( o _ { s } , o _ { g } , o _ { t } )$ into visual tokens via a learned codebook, while a byte-pair encoding (BPE) tokenizer Team (2024) encodes pose $p _ { o }$ and text prompts into text tokens. Actions $a _ { t }$ are mapped to discrete bin tokens using the bin tokenizer, which we discuss below. The resulting token sequences are fed to a causal Transformer for joint multimodal modeling.

Training Objective. To optimize our model for the distinct characteristics of the navigation planner and world model, we introduce tailored training objectives. At each iteration, our autoregressive MLLM jointly processes samples from both roles, producing logits across the unified vocabulary.

• Discretized Bin Token Loss (Navigation Planner). We propose a new classification-based approach for training the planner, which formulates continuous action prediction as multi-class classification over discretized motion bins. Each navigation action $a _ { t } \in \mathbb { R } ^ { 3 }$ is represented as $( x _ { t } , y _ { t } , \phi _ { t } )$ ,

![](images/5e80e1e0ddfacf7d46b9c0ecdda88de16531659cecf055fd4b56d71ce0a41f57.jpg)  
Figure 3: Overview of hierarchical memory bank mechanism $( \mathcal { M } _ { t } ^ { \mathbf { i n t r a } }$ & $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ ). (a) KV (keys/values) extracted from selected layers are deposited into $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ at the beginning of each step $t$ (Eq. 7). (b)(c) $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ is merged with the accumulated cross-step memory $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ via top- $k$ similarity gating (Eq. 8) and exponential temporal decay (Eq. 9), yielding a fused memory (Eq. 10) that augments attention for both the planner and the world-model substeps (Eq. 11) to promote trajectory-consistent predictions. At the end of step $t$ , $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ (with timestamp $t$ ) is appended to $\boldsymbol { \mathcal { M } } _ { t } ^ { \mathrm { c r o s s } }$ for reliable reuse at step $t { + } 1$ , enabling robustly efficient rollouts.

where $x _ { t }$ and $y _ { t }$ denote planar translations and $\phi _ { t }$ denotes yaw rotation. We uniformly partition each dimension into fixed-size bins with size $b = 0 . 0 1$ , computing bin index as $\lfloor \left| v \right| / b \rfloor$ for value $v$ . We use separate positive and negative token prefixes to encode the sign and another prefix for the target dimension. For example, $x$ -axis translation with $v = 0 . 0 3$ is encoded as <dx pos bin $. 0 3 >$ . This scheme represents all three dimensions as special bin tokens from disjoint token sets: $\mathcal { T } _ { x }$ , $\mathcal { T } _ { y }$ , and $\mathcal { T } _ { \phi }$ . Let $P ( t _ { i } )$ denote the model’s predicted distribution over all vocabulary tokens at decoding position $i$ . We supervise the planner using discretized bin token loss over each action dimension:

$$
\mathcal {L} _ {\text {p l a n}} = \frac {1}{3} \sum_ {k \in \{x, y, \phi \}} \left(- \log P \left(t _ {i} = t _ {k} ^ {*} \mid t _ {i} \in \mathcal {T} _ {k}\right)\right) + \mathcal {L} _ {\mathrm {C E}}, \tag {5}
$$

where $t _ { k } ^ { * }$ is the ground-truth bin token in dimension $k$ , and $\mathcal { L } _ { \mathrm { C E } }$ is the cross-entropy loss for output text tokens as output may also include text action Stop.

• Reconstruction Loss (World Model). We introduce a reconstruction loss to enforce fidelity in the predicted future observations to encourage accurate navigation visualization. Given groundtruth visual embedding $\mathbf { v } _ { i }$ for token $i$ (out of $n$ tokens in the next observation $\hat { o } _ { t + 1 }$ ) and the visual codebook embeddings $\mathcal { E } = \{ \mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { N } \}$ where $N$ is the total number of visual token vocabulary:

$$
\mathcal {L} _ {\text {w o r l d}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \| \mathbf {v} _ {i}, \mathcal {E} \| ^ {2} \cdot P \left(t _ {i}\right), \tag {6}
$$

where $\scriptstyle \sum _ { i = 1 } ^ { n } | | \mathbf { v } _ { i } , { \mathcal { E } } | | ^ { 2 }$ is the similarity vector indicating distances between $\mathbf { v } _ { i }$ and all codebook embeddings, with lower similarity referring to larger distances, and $P ( t _ { i } ) \in \mathbb { R } ^ { 1 \times N }$ denotes predicted probability distribution over visual tokens at position i. Throughout training, all tokenizers remain frozen, and only Transformer parameters are updated under autoregressive next-token prediction.

# 2.3 INFERENCE WITH MEMORY BANK

At the inference phase, UniWM alternates between two substeps: action prediction and navigation visualization. As illustrated in Fig. 3 and Alg. 1, UniWM employs a hierarchical two-level memory bank mechanism. The intra-step memory $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ caches key, value $( K , V )$ pairs extracted from the current observation $\hat { o } _ { t - 1 }$ at selected Transformer decoder layers, while the cross-step memory $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ accumulates all past intra-step memories $\mathcal { M } _ { m } ^ { \mathrm { i n t r a } }$ , where $( m \in { 1 , . . . , t - 1 } )$ together with their associated step indices $t _ { m }$ . This design allows $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ to maintain a persistent trajectory-level context, thereby enabling $F _ { \theta }$ to integrate both short-term and longer-term dependencies across steps.

Two-level Cache Design. At the beginning of each step $t$ , the intra-step memory bank $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ is reset to avoid contamination from the previous step: $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }  \emptyset$ . Given the tokenized multimodal input, we identity the span of the current observation $\hat { o } _ { t - 1 }$ by marking its token sequence with two special boundary tokens, ${ < } \mathsf { b o s s } >$ and ${ < } \mathrm { e o s s } >$ , thereby yielding the index set $\mathcal { T } _ { t }$ . We then extract $K , V$

pairs to form $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ only from this span at a selected subset of decoder layers $L _ { \mathrm { s a v e } } = \{ l _ { 0 } , . . . , l _ { 3 1 } \}$

$$
\mathcal {M} _ {t} ^ {\text {i n t r a}} = \left\{K _ {t} ^ {(l)}, V _ {t} ^ {(l)} \right\} = \left\{f _ {K} ^ {(l)} \left(\mathbf {x} _ {\mathcal {I} _ {t}}\right), f _ {V} ^ {(l)} \left(\mathbf {x} _ {\mathcal {I} _ {t}}\right) \right\}, \quad \text {w h e r e} l \in L _ {\text {s a v e}}, \tag {7}
$$

where $\{ K _ { t } ^ { ( l ) } , V _ { t } ^ { ( l ) } \}$ denotes keys and values obtained from the $l$ -th decoder layer at step $t$ , x represents the hidden states of the multimodal input sequence at that layer, $\mathbf { x } _ { \mathcal { T } _ { t } }$ refers to the slice of hidden states indexed by $\mathcal { T } _ { t }$ , and $f _ { K } ^ { ( l ) }$ and $f _ { V } ^ { ( l ) }$ are the key and value projection mappings in layer $l$ . In parallel, as demoncaches from previous ed in Fig. 3, the cross-steps with timestamps $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ d intra-step. $t - 1$ $t _ { m } \colon \mathcal { M } _ { t } ^ { \mathrm { c r o s s } } = \{ ( K _ { m } ^ { ( l ) } , V _ { m } ^ { ( l ) } , t _ { m } ) \} _ { l \in L _ { \mathrm { s a v c } } } ,$

Spatio-temporal Fusion. At each action prediction substep of step $t$ , the intra-step memory $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ is merged with the accumulated cross-step memory $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ to construct a fused memory $\tilde { \mathcal { M } } _ { t }$ , which subsequently enhances the attention mechanism for both substeps. This fusion incorporates spatial similarity selection and temporal recency weighting as shown in Fig. 3:

(i) Similarity gating. We flatten both current and historical keys and compute entry-wise cosine similarity $s _ { m } ^ { ( l ) }$ . The indices of the top- $k$ most similar entries are collected into the set $h _ { t } ^ { ( l ) }$ :

$$
s _ {m} ^ {(l)} = \cos \left(K _ {t} ^ {(l)}, K _ {m} ^ {(l)}\right), \quad h _ {t} ^ {(l)} = \operatorname {t o p - k} \left(s _ {m} ^ {(l)}\right), \quad \text {w h e r e} m \in \{1, \dots , t - 1 \}. \tag {8}
$$

(ii) Temporal decay. Each selected entry is weighted by an exponential decay factor determined by its recency gap $\Delta t _ { m } = t - t _ { m }$ , that larger weights correspond to stronger influence on subsequent predictions. Here we set $\gamma = 0 . 2$ , which biases the weighting toward more recent steps:

$$
\alpha_ {m} ^ {(l)} = \frac {\exp (- \gamma \Delta t _ {m})}{\sum_ {j \in h _ {t} ^ {(l)}} \exp (- \gamma \Delta t _ {j})}. \tag {9}
$$

(iii) Memory fusion. The fused memory $\tilde { \mathcal { M } } _ { t } = \{ \tilde { K } _ { t } ^ { ( l ) } , \tilde { V } _ { t } ^ { ( l ) } \} _ { l \in L _ { \mathrm { s a v e } } }$ is formed by concatenating the current intra-step memory with the weighted historical entries so that historical contributions are explicitly modulated by both spatial similarity and temporal recency:

$$
\tilde {K} _ {t} ^ {(l)} = \mathbf {C o n c a t} \left(K _ {t} ^ {(l)}, \alpha_ {h} ^ {(l)} K _ {h} ^ {(l)}\right), \tilde {V} _ {t} ^ {(l)} = \mathbf {C o n c a t} \left(V _ {t} ^ {(l)}, \alpha_ {h} ^ {(l)} V _ {h} ^ {(l)}\right), \text {w h e r e} h \in h _ {t} ^ {(l)}, l \in L _ {\text {s a v e}}. \tag {10}
$$

Memory-augmented Attention. The fused memory $\tilde { \mathcal { M } } _ { t }$ then directly engage in cross-attention computation. The attention mechanism can be formally described as scaled dot-product attention:

$$
\tilde {Q} _ {t} ^ {(l)} = \operatorname {A t t} \left(Q _ {t} ^ {(l)}, \tilde {K} _ {t} ^ {(l)}, \tilde {V} _ {t} ^ {(l)}\right) = \operatorname {s o f t m a x} \left(\frac {Q _ {t} ^ {(l)} \tilde {K} _ {t} ^ {(l) \top}}{\sqrt {d _ {k}}}\right) \tilde {V} _ {t} ^ {(l)}, \tag {11}
$$

where $Q _ { t } ^ { ( l ) }$ denotes the current query at layer $l$ , and $d _ { k }$ is the key dimension. $\tilde { Q } _ { t } ^ { ( l ) }$ subsequently propagate through subsequent predictions. This mechanism equips UniWM with trajectory-consistent reasoning by leveraging both current observations and temporally structured historical memories.

Rollout Procedure. The full inference process of one trajectory can be summarized as: at each step $t$ , UniWM resets the intra-step memory $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ , extracts $K$ , $V$ pairs from the current observation $\hat { o } _ { t - 1 }$ (Eq. 7), and fuses them with the cross-step memory $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ using similarity gating (Eq. 8) and temporal decay (Eq. 9) to form $\tilde { \mathcal { M } } _ { t }$ (Eq. 10). The planner then predicts the next action $\hat { a } _ { t }$ using enhanced attentions (Eqs. 3 and 11), and the world model generates the next observation $\hat { o } _ { t }$ (Eqs. 4 and 11). Finally, $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ with its timestamp $t$ is appended to $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ for future use. This process iterates until a Stop action is emitted to terminate this navigation trajectory.

# 3 EXPERIMENTS

Datasets. We use four robotics datasets (Go Stanford Hirose et al. (2018), ReCon Shah et al. (2021), SCAND Karnan et al. (2022), and HuRoN Hirose et al. (2023)) for training and in-domain evaluation, and reserve TartanDrive Triest et al. (2022) as an unseen test set. We select these datasets because they cover complementary aspects of real-world navigation; for instance, ReCon targets open-world settings, whereas SCAND emphasizes socially compliant navigation across varied environments. To standardize action magnitudes across embodiments, we normalize per-frame displacement by average step size (in meters), filter out backward motions and trajectories shorter than three steps like Bar et al. (2025); Sridhar et al. (2024), and segment each trajectory’s visual stream into semantically coherent sub-scenes using Qwen-VL-2.5 Bai et al. (2025). After filtering, we obtain the following numbers of trajectories: Go Stanford (train/eval: 4457/496), ReCon (4652/517), SCAND (2560/285), HuRoN (4642/516), and TartanDrive (eval only: 500).

![](images/2bbaa5a265a3202e69314c1d5f95a966373c1a6f31cafd6a584e2b53ecb2aebd.jpg)  
Figure 4: Qualitative Comparisons on Go Stanford and HuRoN across UniWM, NWM, and NoMaD. The central trajectory plots highlight difference between predicted $A _ { T }$ and the ground-truth.

Table 1: Comparison with SOTA Methods upon Goal-Conditioned Visual Navigation on evaluation splits of Go Stanford, ReCon, SCAND, and HuRoN with SR, ATE, and RPE.   

<table><tr><td rowspan="2">Method</td><td colspan="3">Go Stanford</td><td colspan="3">ReCon</td><td colspan="3">SCAND</td><td colspan="3">HuRoN</td></tr><tr><td>SR ↑</td><td>ATE ↓</td><td>RPE ↓</td><td>SR ↑</td><td>ATE ↓</td><td>RPE ↓</td><td>SR ↑</td><td>ATE ↓</td><td>RPE ↓</td><td>SR ↑</td><td>ATE ↓</td><td>RPE ↓</td></tr><tr><td>GNM Shah et al. (2022)</td><td>0.27</td><td>1.11</td><td>0.31</td><td>0.72</td><td>0.70</td><td>0.20</td><td>0.49</td><td>0.51</td><td>0.21</td><td>0.36</td><td>1.07</td><td>0.35</td></tr><tr><td>VINT Shah et al. (2023)</td><td>0.29</td><td>1.09</td><td>0.35</td><td>0.68</td><td>0.84</td><td>0.28</td><td>0.45</td><td>0.58</td><td>0.28</td><td>0.30</td><td>1.19</td><td>0.43</td></tr><tr><td>NoMaD Sridhar et al. (2024)</td><td>0.33</td><td>0.94</td><td>0.30</td><td>0.71</td><td>0.77</td><td>0.21</td><td>0.50</td><td>0.54</td><td>0.23</td><td>0.37</td><td>0.92</td><td>0.33</td></tr><tr><td>Anole-7B Chern et al. (2024)</td><td>0.18</td><td>2.18</td><td>0.73</td><td>0.41</td><td>1.74</td><td>0.69</td><td>0.29</td><td>1.37</td><td>0.71</td><td>0.20</td><td>1.92</td><td>0.78</td></tr><tr><td>NWM Bar et al. (2025)</td><td>0.45</td><td>0.80</td><td>0.27</td><td>0.79</td><td>0.58</td><td>0.17</td><td>0.55</td><td>0.41</td><td>0.19</td><td>0.41</td><td>0.73</td><td>0.28</td></tr><tr><td>UniWM (w/o M)</td><td>0.71</td><td>0.32</td><td>0.10</td><td>0.82</td><td>0.35</td><td>0.12</td><td>0.61</td><td>0.36</td><td>0.14</td><td>0.70</td><td>0.42</td><td>0.15</td></tr><tr><td>UniWM (with only Mtina)</td><td>0.73</td><td>0.29</td><td>0.09</td><td>0.85</td><td>0.38</td><td>0.13</td><td>0.64</td><td>0.33</td><td>0.13</td><td>0.74</td><td>0.44</td><td>0.15</td></tr><tr><td>UniWM (with Mtina &amp; Mtcross)</td><td>0.75</td><td>0.22</td><td>0.09</td><td>0.93</td><td>0.34</td><td>0.11</td><td>0.68</td><td>0.32</td><td>0.13</td><td>0.76</td><td>0.38</td><td>0.13</td></tr></table>

Evaluation Metrics. We evaluate performance using two suites of metrics. (1) Navigation quality. We report Absolute Trajectory Error (ATE), Relative Pose Error (RPE) Sturm et al. (2012), and Success Rate (SR). SR deems a trajectory successful if final distance to goal is smaller than the agent’s average step size (in meters). (2) Visualization quality. For navigation visualization, we use structural/perceptual metrics SSIM Wang et al. (2004), PSNR Hore & Ziou (2010), LPIPS Zhang et al. (2018), and DreamSim Fu et al. (2023). To assess longer-horizon stability under rollout, we introduce four metrics: $\mathrm { S S I M } @ \mathrm { n }$ , PSNR $@ \mathbf { n }$ , LPIPS@n, and DreamSim $@ \mathfrak { n }$ . (Details in Appx. C.1.)

Implementation Details. UniWM is fine-tuned upon GAIR Anole-7B Chern et al. (2024) (4096- token context) while freezing the text and image tokenizers as well as the bin-token encoder. Input images are resized to $4 4 8 \times 4 4 8$ (height $\times$ width) and discretized into 784 visual tokens. During training, only the LoRA Hu et al. (2022) adapters (rank $= 1 6$ ) in the Transformer’s qkv-projections are updated Liu et al. (2023). Optimization is performed with AdamW for 20 epochs using a learning rate of $2 \times 1 0 ^ { - 4 }$ . Training runs on 4×NVIDIA A100 GPUs (80GB each) with a global batch size of 8 (per-GPU batch $= 1$ , gradient accumulation $= 2$ ). For inference, we designate two special tokens ${ \tt { < b o s s > } }$ and ${ < } \mathrm { e o s s } >$ (token IDs 8196 and 8197) to trigger the key–value (KV) deposit of intra-step memory bank, and extract KV for enhancement from decoder layers $\{ l _ { 0 } , l _ { 7 } , l _ { 1 5 } , l _ { 2 3 } , l _ { 3 1 } \}$ .

# 3.1 COMPARISON TO STATE-OF-THE-ART METHODS

Navigation performance. Table 1 reports goal-conditioned visual navigation results on four indomain datasets (Go Stanford, ReCon, SCAND, and HuRoN), while Fig. 4 provides qualitative comparison results (more results in Appx. C.2). We compare UniWM with traditional navigation policies GNM Shah et al. (2022), VINT Shah et al. (2023), NoMaD Sridhar et al. (2024), and Anole-7B Chern et al. (2024) under direct prompting (zero-shot). We include NWM Bar et al. (2025), which leverages world modeling through CDiT within MPC framework. UniWM consistently delivers superior results compared to all SOTA baselines. Without memory augmentation, UniWM achieves substantial gains in SR (e.g., 0.71 vs. 0.45 for NWM on Go Stanford) and ATE/RPE across datasets. Equipping UniWM with intra-step memory stabilizes predictions, while cross-step memory enhances longer-horizon consistency, leading to best overall performance.

Table 2: Comparison with SOTA methods on visualization performance, averaged over evaluation splits of Go Stanford, ReCon, SCAND, and HuRoN.   

<table><tr><td>Method</td><td>SSIM↑</td><td>PSNR↑</td><td>LPIPS↓</td><td>DreamSIM↓</td><td>SSIM@5↑</td><td>PSNR@5↑</td><td>LPIPS@5↓</td><td>DreamSIM@5↓</td></tr><tr><td>Diamond Alonso et al. (2024)</td><td>0.311</td><td>9.837</td><td>0.410</td><td>0.131</td><td>0.186</td><td>6.352</td><td>0.582</td><td>0.252</td></tr><tr><td>NWM Bar et al. (2025)</td><td>0.389</td><td>11.420</td><td>0.318</td><td>0.089</td><td>0.256</td><td>7.755</td><td>0.494</td><td>0.174</td></tr><tr><td>UniWM</td><td>0.457</td><td>13.607</td><td>0.254</td><td>0.041</td><td>0.350</td><td>10.874</td><td>0.435</td><td>0.126</td></tr></table>

Table 3: Impact of Context Size and Image Token Length on both navigation and visualization performance, averaged over four datasets. All settings are evaluated without memory banks.   

<table><tr><td rowspan="2">Context</td><td rowspan="2">Token Len.</td><td colspan="3">Navigation</td><td colspan="6">Visualization</td></tr><tr><td>SR ↑</td><td>ATE ↓</td><td>RPE ↓</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>DreamSIM ↓</td><td>SSIM@5 ↑</td><td>LPIPS@5 ↓</td><td>DreamSIM@5 ↓</td></tr><tr><td>1</td><td>784</td><td>0.71</td><td>0.36</td><td>0.13</td><td>0.457</td><td>0.254</td><td>0.041</td><td>0.350</td><td>0.435</td><td>0.126</td></tr><tr><td>2</td><td>625</td><td>0.68</td><td>0.39</td><td>0.15</td><td>0.448</td><td>0.247</td><td>0.051</td><td>0.336</td><td>0.451</td><td>0.137</td></tr><tr><td>2</td><td>484</td><td>0.55</td><td>0.53</td><td>0.26</td><td>0.365</td><td>0.328</td><td>0.084</td><td>0.258</td><td>0.515</td><td>0.192</td></tr><tr><td>4</td><td>484</td><td>0.64</td><td>0.44</td><td>0.19</td><td>0.425</td><td>0.285</td><td>0.052</td><td>0.315</td><td>0.462</td><td>0.141</td></tr></table>

![](images/76b8a2ee16b4845fe20d570760beb76a66b1b54fcef94ff0b3a78761d6a94d50.jpg)

![](images/9b27a76d7f65303a5b2d8626d8b947e6c30a0cb844e54e612e610e163da21bf7.jpg)  
Figure 5: Impact of discretized bin-token loss $( \mathcal { L } _ { \mathrm { p l a n } } )$ and reconstruction Loss $\scriptstyle ( { \mathcal { L } } _ { \mathrm { w o r l d } } )$ on navigation (left) and visualization (right) performance, averaged over evaluation splits of Go Stanford, ReCon, SCAND, and HuRoN. X-axis arrows indicate whether higher or lower values are preferable.

Visualization performance. Table 2 evaluates UniWM’s visualization ability. We compare against Diamond Alonso et al. (2024), a diffusion-based world model on UNet, and NWM Bar et al. (2025) with CDiT. UniWM achieves competitive results across all metrics. On one-step predictions, it delivers highest structural similarity $( \mathrm { S S I M } = 0 . 4 5 7 )$ ) and perceptual alignment $( \mathrm { D r e a m S i m } = 0 . 0 4 1 )$ ). Under open-loop rollouts, UniWM maintains stability with $\mathrm { S S I M @ 5 = 0 . 3 5 0 }$ , preserving semantic consistency and mitigating compounding errors in longer-horizon evaluations.

# 3.2 ABLATION STUDIES

# 1. How do context size and token length affect navigation and visualization performance?

Table 3 analyzes varying context size and token length per image. Since Anole-7B has a fixed 4096 token context window, increasing context frames requires reducing tokens per frame, creating a trade-off between temporal coverage and spatial resolution. Both navigation and visualization performance improve as context size or token length increases (e.g., $( 2 \times 4 8 4 , 4 \times 4 8 4 )$ and $( 2 \times 4 8 4$ , $2 \times 6 2 5$ ). Comparing $1 \times 7 8 4$ with $2 \times 6 2 5$ and $4 \times 4 8 4$ shows higher token length can outweigh additional context, suggesting spatial resolution has stronger overall impact under fixed token budget.

# 2. Do discretized bin token loss $\mathcal { L } _ { \bf p l a n }$ and reconstruction loss ${ \mathcal { L } } _ { \mathrm { w o r l d } }$ help training?

To evaluate the impact of different training objectives on navigation and visualization performance, we compare ${ \mathcal { L } } _ { \mathrm { p l a n } }$ for action tokens and ${ \mathcal { L } } _ { \mathrm { w o r l d } }$ for image tokens against a label smoothing loss $( \mathcal { L } _ { \mathrm { L S } } )$ . From Fig. 5 (right), replacing $\mathcal { L } _ { \mathrm { { L S } } }$ with ${ \mathcal { L } } _ { \mathrm { w o r l d } }$ on image tokens markedly enhances visualization quality and gains persist under rollout, while Fig. 5 (left) demonstrates that both ${ \mathcal { L } } _ { \mathrm { w o r l d } }$ and $\mathcal { L } _ { \mathrm { { p l a n } } }$ benefit navigation. Fig. 5 (left) also shows the best performance arises when combining $\mathcal { L } _ { \mathrm { { p l a n } } }$ with ${ \mathcal { L } } _ { \mathrm { w o r l d } }$ , and the navigation gains from ${ \mathcal { L } } _ { \mathrm { p l a n } }$ are larger than those from ${ \mathcal { L } } _ { \mathrm { w o r l d } }$ under matched conditions (SR $+ 0 . 1 2$ vs. $+ 0 . 1 0 )$ . A plausible explanation is that ${ \mathcal { L } } _ { \mathrm { w o r l d } }$ improves navigation indirectly by enhancing visualization quality, whereas $\mathcal { L } _ { \mathrm { p l a n } }$ optimizes UniWM’s action decisions directly.

# 3. Do we need both intra-step and cross-step memory bank during inference?

We compare using three UniWM variants in Table 1: no memory, intra-step memory bank only, and intra+cross memory banks. Adding intra-step memory improves SR on all four datasets and generally stabilizes pose estimates—RPE decreases or remains comparable. Further augmenting with cross-step memory yields the best SR and RPE (overall: $0 . 7 8 / 0 . 1 1 )$ across all datasets. These

Table 4: Impact of number of selected layers included in memory bank on navigation performance of UniWM (with $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ & $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ ) on evaluation splits of four in-domain datasets.   

<table><tr><td rowspan="2">Layer Num</td><td colspan="3">Go Stanford</td><td colspan="3">ReCon</td><td colspan="3">SCAND</td><td colspan="3">HuRoN</td></tr><tr><td>SR ↑</td><td>ATE ↓</td><td>RPE ↓</td><td>SR ↑</td><td>ATE ↓</td><td>RPE ↓</td><td>SR ↑</td><td>ATE ↓</td><td>RPE ↓</td><td>SR ↑</td><td>ATE ↓</td><td>RPE ↓</td></tr><tr><td>1</td><td>0.71</td><td>0.30</td><td>0.10</td><td>0.84</td><td>0.36</td><td>0.12</td><td>0.62</td><td>0.35</td><td>0.14</td><td>0.72</td><td>0.41</td><td>0.14</td></tr><tr><td>3</td><td>0.74</td><td>0.27</td><td>0.09</td><td>0.89</td><td>0.35</td><td>0.11</td><td>0.66</td><td>0.33</td><td>0.13</td><td>0.75</td><td>0.39</td><td>0.14</td></tr><tr><td>5</td><td>0.75</td><td>0.22</td><td>0.09</td><td>0.93</td><td>0.34</td><td>0.11</td><td>0.68</td><td>0.32</td><td>0.13</td><td>0.76</td><td>0.38</td><td>0.13</td></tr><tr><td>7</td><td>0.74</td><td>0.25</td><td>0.09</td><td>0.91</td><td>0.35</td><td>0.12</td><td>0.69</td><td>0.31</td><td>0.13</td><td>0.74</td><td>0.38</td><td>0.14</td></tr><tr><td>16</td><td>0.61</td><td>0.46</td><td>0.23</td><td>0.70</td><td>0.58</td><td>0.26</td><td>0.52</td><td>0.49</td><td>0.24</td><td>0.57</td><td>0.55</td><td>0.22</td></tr><tr><td>32</td><td>0.58</td><td>0.52</td><td>0.26</td><td>0.67</td><td>0.64</td><td>0.29</td><td>0.49</td><td>0.55</td><td>0.27</td><td>0.54</td><td>0.61</td><td>0.25</td></tr></table>

Table 5: Comparison of navigation performance under different step strategies across four datasets.   
Table 6: Zero-shot navigation perfor- Figure 6: Qualitative Results in unseen environments (Tartanmance evaluated on TartanDrive (unseen). Drive) with UniWM. Red boxes denote ego-robot parts.   

<table><tr><td rowspan="2">Step Strategy</td><td colspan="3">Go Stanford</td><td colspan="3">ReCon</td><td colspan="3">SCAND</td><td colspan="3">HuRoN</td></tr><tr><td>SR↑</td><td>ATE↓</td><td>RPE↓</td><td>SR↑</td><td>ATE↓</td><td>RPE↓</td><td>SR↑</td><td>ATE↓</td><td>RPE↓</td><td>SR↑</td><td>ATE↓</td><td>RPE↓</td></tr><tr><td>Predict both</td><td>0.65</td><td>0.38</td><td>0.13</td><td>0.80</td><td>0.41</td><td>0.15</td><td>0.57</td><td>0.39</td><td>0.17</td><td>0.63</td><td>0.47</td><td>0.20</td></tr><tr><td>Interleave</td><td>0.71</td><td>0.32</td><td>0.10</td><td>0.82</td><td>0.35</td><td>0.12</td><td>0.61</td><td>0.36</td><td>0.14</td><td>0.70</td><td>0.42</td><td>0.15</td></tr></table>

None of the baselines is fine-tuned.   

<table><tr><td>Method</td><td>SR ↑</td><td>ATE ↓</td><td>RPE ↓</td></tr><tr><td>GNM Shah et al. (2022)</td><td>0.16</td><td>2.45</td><td>0.79</td></tr><tr><td>VINT Shah et al. (2023)</td><td>0.13</td><td>2.38</td><td>0.79</td></tr><tr><td>NoMaD Sridhar et al. (2024)</td><td>0.18</td><td>2.23</td><td>0.77</td></tr><tr><td>Anole-7B Chern et al. (2024)</td><td>0.15</td><td>2.12</td><td>0.83</td></tr><tr><td>NWM Bar et al. (2025)</td><td>0.27</td><td>1.61</td><td>0.62</td></tr><tr><td>UniWM (w/o M)</td><td>0.35</td><td>1.20</td><td>0.46</td></tr><tr><td>UniWM (with only Mtina)</td><td>0.38</td><td>1.04</td><td>0.41</td></tr><tr><td>UniWM (with Mtina &amp; Mtcross)</td><td>0.42</td><td>0.95</td><td>0.37</td></tr></table>

results indicate that both intra-step and cross-step memories are important at inference time, with cross-step memory providing longer-horizon gains on top of intra-step stabilization.

# 4. How does the number of selected layers included in the memory bank affect inference?

We evaluate the impact of integrating memory at different numbers of selected layers (with both $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ and $\boldsymbol { \mathcal { M } } _ { t } ^ { \mathrm { c r o s s } }$ enabled) and report navigation metrics in Table 4. Increasing from a single shallow layer to a moderate multi-depth integration (3–7 layers) progressively improves SR/ATE/RPE, with the 5-layer setting achieving strong results across all four datasets, indicating that multi-depth integration facilitates iterative feature refinement through the network hierarchy. In contrast, dense integration (16–32 layers) degrades performance and incurs higher compute and KV overhead. Balancing accuracy and efficiency, we adopt the 5-layer configuration for inference.

# 5. Why UniWM predict action and observation at different substep?

Table 5 compares two step strategies of UniWM: predict both vs. interleave (our choice). In predict both, each training sample contains $\left( o _ { s } , o _ { g } , o _ { t } , p _ { 0 } \right)$ and the model jointly predicts the next action $\hat { a } _ { t + 1 }$ and next observation $\hat { o } _ { t + 1 }$ in a single forward pass. In interleave, we provide two sample types corresponding to the planner and world-model sub-steps and alternate them; inference follows the same alternation. Across all datasets, interleave yields higher SR and lower ATE/RPE, which empirically verifies UniWM’s design choice to predict actions and observations in different substeps.

# 3.3 GENERALIZATION IN UNSEEN ENVIRONMENTS

We evaluate zero-shot generalization on the unseen TartanDrive split without any fine-tuning in Table 6 and Fig 6. UniWM consistently delivers competitive results compared to all baselines. Even without memory augmentation, UniWM achieves substantial gains in SR and reduces pose errors. Equipping UniWM with $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ further stabilizes predictions, while adding $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ enhances longerhorizon consistency, yielding the best overall performance (SR 0.42, ATE 0.95, RPE 0.37). These results confirm strong generalization of UniWM in unseen environments.

Error cases and limitations analysis. On TartanDrive, egocentric observation occasionally contains visible ego-robot parts (e.g., bumper/hood). Fig. 6 shows UniWM’s first-step prediction preserves these ego cues, but during rollouts they fade and disappear. We attribute this to domain gap: our training sets lack visible ego-robot regions, so the model treats them as background and ”inpaints” them away. This causes inconsistencies with ground-truth frames in unseen environments.

# 4 CONCLUSION

We present UniWM, a unified memory-augmented world model that couples visual imagination with navigation planning in a single multimodal autoregressive architecture. By jointly modeling perception, prediction, and planning, UniWM closes state–action misalignment. A hierarchical memory fuses short-term observations with longer-range context, stabilizing longer-horizon rollouts. Experiments across four benchmarks and zero-shot evaluation on the TartanDrive dataset show higher SR and lower ATE/RPE than strong baselines. UniWM represents a promising direction toward scalable and generalizable visual navigation systems. Current limitations include domain shift (e.g., ego-robot artifacts) and a fixed token budget, which future work can address through adaptive token allocation, uncertainty-aware planning, and closed-loop deployment on real robots.

# REFERENCES

Niket Agarwal, Arslan Ali, Maciej Bala, Yogesh Balaji, Erik Barker, Tiffany Cai, Prithvijit Chattopadhyay, Yongxin Chen, Yin Cui, Yifan Ding, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.   
Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos J Storkey, Tim Pearce, and Franc¸ois Fleuret. Diffusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems, 37:58757–58791, 2024.   
Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15619–15629, 2023.   
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.   
Federico Baldassarre, Marc Szafraniec, Basile Terver, Vasil Khalidov, Francisco Massa, Yann Le-Cun, Patrick Labatut, Maximilian Seitzer, and Piotr Bojanowski. Back to the features: Dino as a foundation for video world models. arXiv preprint arXiv:2507.19468, 2025.   
Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 15791–15801, 2025.   
Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.   
Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, et al. Video generation models as world simulators. OpenAI Blog, 1(8):1, 2024.   
Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, Matthew Lai, Aditi Mavalankar, Richie Steigerwald, Chris Apps, et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.   
Devendra Singh Chaplot, Dhiraj Gandhi, Saurabh Gupta, Abhinav Gupta, and Ruslan Salakhutdinov. Learning to explore using active neural slam. arXiv preprint arXiv:2004.05155, 2020.   
Zhi-Qi Cheng, Yifei Dong, Aike Shi, Wei Liu, Yuzhi Hu, Jason O’Connor, Alexander G Hauptmann, and Kate S Whitefoot. Shield: Llm-driven schema induction for predictive analytics in ev battery supply chain disruptions. arXiv preprint arXiv:2408.05357, 2024.   
Ethan Chern, Jiadi Su, Yan Ma, and Pengfei Liu. Anole: An open, autoregressive, native large multimodal models for interleaved image-text generation. arXiv preprint arXiv:2407.06135, 2024.

Jingtao Ding, Yunke Zhang, Yu Shang, Yuheng Zhang, Zefang Zong, Jie Feng, Yuan Yuan, Hongyuan Su, Nian Li, Nicholas Sukiennik, et al. Understanding world or predicting future? a comprehensive survey of world models. ACM Computing Surveys, 2024.   
Yifei Dong, Fengyi Wu, Qi He, Heng Li, Minghan Li, Zebang Cheng, Yuxuan Zhou, Jingdong Sun, Qi Dai, Zhi-Qi Cheng, et al. Ha-vln: A benchmark for human-aware navigation in discretecontinuous environments with dynamic multi-human interactions, real-world validation, and an open leaderboard. arXiv preprint arXiv:2503.14229, 2025a.   
Yifei Dong, Fengyi Wu, Sanjian Zhang, Guangyu Chen, Yuzhi Hu, Masumi Yano, Jingdong Sun, Siyu Huang, Feng Liu, Qi Dai, et al. Securing the skies: A comprehensive survey on anti-uav methods, benchmarking, and future directions. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6659–6673, 2025b.   
J Frey, M Mattamala, N Chebrolu, C Cadena, M Fallon, and M Hutter. Fast traversability estimation for wild visual navigation. In Robotics: Science and Systems, volume 19. Robotics: Science and Systems, 2023.   
Stephanie Fu, Netanel Tamir, Shobhita Sundaram, Lucy Chai, Richard Zhang, Tali Dekel, and Phillip Isola. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. arXiv preprint arXiv:2306.09344, 2023.   
Zipeng Fu, Ashish Kumar, Ananye Agarwal, Haozhi Qi, Jitendra Malik, and Deepak Pathak. Coupling vision and proprioception for navigation of legged robots. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 17273–17283, 2022.   
Oran Gafni, Adam Polyak, Oron Ashual, Shelly Sheynin, Devi Parikh, and Yaniv Taigman. Makea-scene: Scene-based text-to-image generation with human priors. In European conference on computer vision, pp. 89–106. Springer, 2022.   
David Ha and Jurgen Schmidhuber. World models. ¨ arXiv preprint arXiv:1803.10122, 2(3), 2018.   
Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. arXiv preprint arXiv:1912.01603, 2019.   
Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models, 2022. URL https://arxiv.org/abs/2010.02193.   
Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models, 2024. URL https://arxiv.org/abs/2301.04104.   
Noriaki Hirose, Amir Sadeghian, Marynel Vazquez, Patrick Goebel, and Silvio Savarese. Gonet: A ´ semi-supervised deep learning approach for traversability estimation. In 2018 IEEE/RSJ international conference on intelligent robots and systems (IROS), pp. 3044–3051. IEEE, 2018.   
Noriaki Hirose, Dhruv Shah, Ajay Sridhar, and Sergey Levine. Sacson: Scalable autonomous control for social navigation. IEEE Robotics and Automation Letters, 9(1):49–56, 2023.   
Alain Hore and Djemel Ziou. Image quality metrics: Psnr vs. ssim. In 2010 20th international conference on pattern recognition, pp. 2366–2369. IEEE, 2010.   
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.   
Haresh Karnan, Anirudh Nair, Xuesu Xiao, Garrett Warnell, Soren Pirk, Alexander Toshev, Justin ¨ Hart, Joydeep Biswas, and Peter Stone. Socially compliant navigation dataset (scand): A largescale dataset of demonstrations for social navigation. IEEE Robotics and Automation Letters, 7 (4):11807–11814, 2022.   
Jing Yu Koh, Honglak Lee, Yinfei Yang, Jason Baldridge, and Peter Anderson. Pathdreamer: A world model for indoor navigation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 14738–14748, 2021.

Heng Li, Minghan Li, Zhi-Qi Cheng, Yifei Dong, Yuxuan Zhou, Jun-Yan He, Qi Dai, Teruko Mitamura, and Alexander G Hauptmann. Human-aware vision-and-language navigation: Bridging simulation to reality with dynamic human interactions. Advances in Neural Information Processing Systems, 37:119411–119442, 2024.   
Bingqian Lin, Yunshuang Nie, Ziming Wei, Jiaqi Chen, Shikui Ma, Jianhua Han, Hang Xu, Xiaojun Chang, and Xiaodan Liang. Navcot: Boosting llm-based vision-and-language navigation via learning disentangled reasoning. arXiv preprint arXiv:2403.07376, 2024.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.   
Piotr Mirowski, Razvan Pascanu, Fabio Viola, Hubert Soyer, Andrew J Ballard, Andrea Banino, Misha Denil, Ross Goroshin, Laurent Sifre, Koray Kavukcuoglu, et al. Learning to navigate in complex environments. arXiv preprint arXiv:1611.03673, 2016.   
Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. Gaia-2: A controllable multi-view generative world model for autonomous driving. arXiv preprint arXiv:2503.20523, 2025.   
Dhruv Shah, Benjamin Eysenbach, Gregory Kahn, Nicholas Rhinehart, and Sergey Levine. Rapid exploration for open-world navigation with latent goal models. arXiv preprint arXiv:2104.05859, 2021.   
Dhruv Shah, Ajay Sridhar, Arjun Bhorkar, Noriaki Hirose, and Sergey Levine. Gnm: A general navigation model to drive any robot. arXiv preprint arXiv:2210.03370, 2022.   
Dhruv Shah, Ajay Sridhar, Nitish Dashora, Kyle Stachowicz, Kevin Black, Noriaki Hirose, and Sergey Levine. Vint: A foundation model for visual navigation. arXiv preprint arXiv:2306.14846, 2023.   
Mingchen Song, Xiang Deng, Zhiling Zhou, Jie Wei, Weili Guan, and Liqiang Nie. A survey on diffusion policy for robotic manipulation: Taxonomy, analysis, and future directions. Authorea Preprints, 2025.   
Ajay Sridhar, Dhruv Shah, Catherine Glossop, and Sergey Levine. Nomad: Goal masked diffusion policies for navigation and exploration. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 63–70. IEEE, 2024.   
Jurgen Sturm, Wolfram Burgard, and Daniel Cremers. Evaluating egomotion and structure-from- ¨ motion approaches using the tum rgb-d benchmark. In Proc. of the Workshop on Color-Depth Camera Fusion in Robotics at the IEEE/RJS International Conference on Intelligent Robot Systems (IROS), volume 13, pp. 6, 2012.   
Chameleon Team. Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818, 2024.   
Samuel Triest, Matthew Sivaprakasam, Sean J Wang, Wenshan Wang, Aaron M Johnson, and Sebastian Scherer. Tartandrive: A large-scale dataset for learning off-road dynamics models. In 2022 International Conference on Robotics and Automation (ICRA), pp. 2546–2552. IEEE, 2022.   
Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837, 2024.   
Zhou Wang, A.C. Bovik, H.R. Sheikh, and E.P. Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004. doi: 10.1109/TIP.2003.819861.   
Fengyi Wu, Yifei Dong, Zhi-Qi Cheng, Yilong Dai, Guangyu Chen, Hang Wang, Qi Dai, and Alexander G Hauptmann. Govig: Goal-conditioned visual navigation instruction generation. arXiv preprint arXiv:2508.09547, 2025.

Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. Worldmem: Long-term consistent world simulation with memory. arXiv preprint arXiv:2504.12369, 2025.   
Eric Xing, Mingkai Deng, Jinyu Hou, and Zhiting Hu. Critiques of world models, 2025.   
Xuan Yao, Junyu Gao, and Changsheng Xu. Navmorph: A self-evolving world model for visionand-language navigation in continuous environments. arXiv preprint arXiv:2506.23468, 2025.   
Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. Gamefactory: Creating new games with generative interactive videos. arXiv preprint arXiv:2501.08325, 2025.   
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 586–595, 2018.   
Guosheng Zhao, Xiaofeng Wang, Zheng Zhu, Xinze Chen, Guan Huang, Xiaoyi Bao, and Xingang Wang. Drivedreamer-2: Llm-enhanced world models for diverse driving video generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 10412–10420, 2025.

# APPENDIX

# A RELATED WORK

World models have emerged as a unifying paradigm for learning predictive representations of environment dynamics, with applications spanning simulation, decision-making, and embodied navigation. In this section, we review two lines of related research: (i) advances in generic world model architectures, and (ii) their application to goal-conditioned visual navigation.

World Models. World models Ha & Schmidhuber (2018) have become a central paradigm for learning predictive representations of environment dynamics Ding et al. (2024), evolving from compact recurrent structures to large-scale generative and multimodal systems. Early works such as World Model Ha & Schmidhuber (2018) and Dreamer Hafner et al. (2019; 2022; 2024) employed RNNbased latent dynamics to capture temporal transitions. Transformer-based designs (I-JEPA Assran et al. (2023), V-JEPA Bardes et al. (2024), DINO-WM Baldassarre et al. (2025)) introduced scalable attention mechanisms for richer spatio-temporal abstraction. More recently, diffusion-based generators (Sora Brooks et al. (2024), Cosmos Agarwal et al. (2025), Genie Bruce et al. (2024)) have been adapted for environment dynamics, enabling high-fidelity simulation and downstream planning Alonso et al. (2024); Valevski et al. (2024); Bar et al. (2025); Yu et al. (2025), though at the cost of efficiency and limited integration with policy learning Xiao et al. (2025). Parallel efforts exploit LLMs to simulate dynamics via prompting Zhao et al. (2025); Xing et al. (2025), but face modality misalignment, temporal inconsistency, and grounding challenges Ding et al. (2024); Dong et al. (2025b), with context length limits causing memory degradation over longer horizons Xiao et al. (2025). Our work builds on these advances by introducing the hierarchical memory bank mechanism and a unified paradigm that jointly couples perception, prediction, and decision-making, mitigating the alignment and stability issues in prior modular designs.

World Models for Navigation. Goal-conditioned navigation is a natural testbed for world models, as it requires tight coupling between perception and policy Frey et al. (2023). Policy-centric methods Shah et al. (2022; 2023); Sridhar et al. (2024) map observations directly to actions without explicitly modeling environment dynamics. In contrast, navigation-oriented world models predict future observations to support temporally informed planning Yao et al. (2025). Early works such as PathDreamer Koh et al. (2021) used GANs to simulate indoor vision–language navigation but depended on auxiliary inputs (e.g., semantic maps), limiting generalization Lin et al. (2024). More recent approaches (e.g., NWM Bar et al. (2025)) integrate raw video prediction into the navigation loop to produce realistic rollouts, yet still decouple planning from perception, relying on separate policy modules and failing to reason jointly over actions and observations. Building on these advances, we propose a unified multimodal backbone that aligns action prediction with observation imagination, enabling end-to-end navigation through temporally grounded dynamics modeling.

# Action Prediction

# Input

# Task: Action Prediction

Description: Based on the current first-person observation, starting point observation and coordinate, goal point observation, predict the next action to take.

# Inputs:

Starting Pose: (-90.16528149, -187.79242581, 0.15229973)

![](images/d960fcf4c36f119cfd128720fecfa9a99832ecc57e1731e92979b1e6934e89e6.jpg)  
Goal observation:

![](images/1e1eaad976982f069915954a4473d308d30a2893fc7c6c60ef2d8392d2e0a59e.jpg)

Start observation:

![](images/5048d7569c6ad0e1697ad42512a148b0217aca40c35448a199b3852fc3e43d73.jpg)  
Figure 7: Prompt design details and examples on action prediction (context size $= 1$ )

# Current observation:

Action Format: The action can be the language command ’Stop’, indicating the end of the trajectory. Alternatively, the action can be shifts composed of three components: - dx: displacement along the agent’s facing direction), - dy: displacement perpendicular to the facing direction), - dyaw: change in heading angle (i.e., how much the agent rotates). All components are discretized into bin tokens: for example, - ‘dx pos bin $0 2 ^ { \cdot }$ : $\mathrm { d } \mathrm { x } = + 0 . 0 2$ meters, - ‘dy neg bin $2 3 \colon \mathrm { d y } = - 0 . 2 3$ meters, - ‘dyaw pos bin $2 6 ^ { \circ }$ : counterclockwise rotation of $+ 0 . 2 6$ radians. If the agent reaches the goal or believes it has reached, it should predict ’Stop’. -Output format: Move by dx: ${ < } \mathrm { d } \mathbf { x } >$ , dy: ${ \mathrm { < d y > } }$ , dyaw: <dyaw>

Goal: Predict the next action to approach the goal observation.

# Response

# Predicted Action:

Move by dx: <dx pos bin $. 1 8 >$ , dy: <dy pos bin $. 0 5 >$ , dyaw: <dyaw pos bin 07>

# B METHOD DETAILS

# B.1 PROMPT DESIGN AND EXAMPLES

We examine the detailed prompt formulation Cheng et al. (2024); Wu et al. (2025) and response behaviors of two substeps: action prediction and navigation visualization in Figs. 7 and 8. These examples illustrate how multimodal inputs guide both the navigation planner and the world model in visually grounded navigation.

# B.2 PSEUDO-CODE FOR HIERARCHICAL MEMORY BANK MECHANISM

Alg. 1 details the inference process of UniWM, which systematically employs the hierarchical memory bank. The algorithm begins by initializing the intra-step memory $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ and the persistent crossstep memory $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ as empty sets (Line 9). It also defines a subset of decoder layers, $L _ { \mathrm { s a v e } }$ , from which Key-Value (KV) pairs will be extracted (Line 10). The main logic operates in a loop for each step $t$ from 1 to $T$ (Line 12), divided into two substeps:

Action Prediction. At the start of each step, the intra-step memory is cleared to prevent contamination from the previous state (Line 14). The ExtractKV function (Line 5, corresponding to Eq. 7) is invoked to extract KV pairs from the current observation $\hat { o } _ { t - 1 }$ , which are then stored in $\mathcal { M } _ { t } ^ { \mathrm { i n t r a } }$ (Lines 15-16). This new intra-step memory is then fused with the historical cross-step memory $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ using the Merge function (Line 19), which encapsulates the spatio-temporal fusion logic

# Navigation Visualization

# Input

# Task: Navigation Single Step Visualization

Description: Given the current first-person observation, predict the next first-person view observation after the agent executes a specified navigation action. To assist your prediction, you may refer to the start observation and pose (position: x, y and heading: yaw), as well as the goal and current observation.

# Inputs:

Next Action: Move by dx: 0.18, dy: 0.05, dyaw: 0.07

Starting Pose: (-90.16528149, -187.79242581, 0.15229973)

![](images/1fe4e1ca816df9bfbebb21023114fc5adae7a29d49d8cd70a3053bb6689cf24a.jpg)

![](images/53df3c234cc764add867ce5cf5fcea039832bbe4efc3d4d958547754e0c4cab7.jpg)  
Goal Observation:

Start Observation:

![](images/c5023c46377a664fd9589df22b3366d4b51c4e68026a2999f9acc7d9d4ce2681.jpg)

# Current Observation:

Action Format: The action can be the language command ’Stop’, indicating the end of the trajectory. Alternatively, the action can be shifts composed of three components: - dx: displacement along the agent’s facing direction), - dy: displacement perpendicular to the facing direction), - dyaw: change in heading angle (i.e., how much the agent rotates). All components are discretized into bin tokens: for example, - ‘dx pos bin $0 2 ^ { \cdot }$ : $\mathrm { d } \mathrm { x } = + 0 . 0 2$ meters, - ‘dy neg bin $2 3 ^ { \mathfrak { \bullet } }$ : dy = -0.23 meters, - ‘dyaw pos bin $2 6 ^ { \circ }$ : counterclockwise rotation of $+ 0 . 2 6$ radians.

Spatial Interpretation: - The magnitude of [dx, dy] reflects how far the agent moves in this step — larger values indicate greater positional shift, leading to larger visual changes. - dyaw controls the agent’s rotation (change in heading). A positive dyaw indicates a left turn (counter-clockwise), while a negative dyaw indicates a right turn (clockwise).

Goal: Predict the most likely next first-person observation, considering how the movement and rotation implied by ‘dx‘, ‘dy‘, and ‘dyaw‘ would affect what the agent sees next.

# Response

Predicted observation:

![](images/9086fba8b5f25b86e92b6e50ec386d7cb1e42b02ad2b85a7f948c4f153b65648.jpg)  
Figure 8: Prompt design details and examples on navigation visualization (context size $= 1$

from Eqs. 8, 9, and 10. At the first step $t = 1$ ), when $\boldsymbol { \mathcal { M } } _ { t } ^ { \mathrm { c r o s s } }$ is empty, the fused memory $\tilde { \mathcal { M } } _ { t }$ is simply the intra-step memory (Line 18). Finally, the model predicts the action $\hat { a } _ { t }$ using an enhanced attention mechanism conditioned on the fused memory $\tilde { \mathcal { M } } _ { t }$ , as described in Eq. 11 (Line 21).

Navigation Visualization. Following action prediction, the model generates the next observation $\hat { o } _ { t }$ . This process reuses the same fused memory $\tilde { \mathcal { M } } _ { t }$ from the action prediction substep, ensuring contextual consistency. The generation is conditioned on the prior state and the newly predicted action $\hat { a } _ { t }$ (Line 23). After both substeps, the intra-step memory $\mathbf { \dot { \mathcal { M } } } _ { t } ^ { \mathrm { i n t r a } }$ is appended to the cross-step bank $\mathcal { M } _ { t } ^ { \mathrm { c r o s s } }$ , preserving the context of the current step for future predictions (Line 24).

This iterative process continues until the trajectory concludes, at which point the algorithm returns the complete sequences of predicted actions and observations (Line 27).

Algorithm 1 Inference with Intra-step and Cross-step Memory Banks in UniWM  
Input: Start position \( p_0 \), start observation \( o_s \), goal observation \( o_g \); Decoder layers \( L = \{l_0, \dots, l_{31}\} \)  
Output: Action sequence \( A_T = \{\hat{a}_1, \dots, \hat{a}_T\} \), observation sequence \( \mathcal{O}_T = \{\hat{o}_1, \dots, \hat{o}_T\} \)  
1: Definitions (helpers)  
2: ResetIntra(): clear intra-step memory bank \( \mathcal{M}_t^{\text{intra}} \)  
3: AppendIntra(\( K_t^{(l)}, V_t^{(l)} \}) to KV to \( \mathcal{M}_t^{\text{intra}} \)  
4: AppendCross(\( \mathcal{M}_t^{\text{intra}} \)): push intra-step bank \( \mathcal{M}_t^{\text{intra}} \) to cross-step bank \( \mathcal{M}^{\text{cross}} \)  
5: ExtractKV(token seq.) -> \( \{K_t^{(l)}, V_t^{(l)}\} \)) to extract KV at selected layers (Eq. 7)  
6: Merge(\( \mathcal{M}_t^{\text{cross}}, \mathcal{M}_t^{\text{intra}} \)) -> \( \tilde{\mathcal{M}}_t \): memory fusion (Eqs. 8, 9, and 10)  
7: EnhanceAndDecodecond, \( \mathcal{M}_t^{\text{intra}}, \tilde{\mathcal{M}}_t \)) -> predict with enhanced attention (Eq. 11)  
8: Initialization  
9: \( \mathcal{M}_t^{\text{intra}} \gets \emptyset \), \( \mathcal{M}_t^{\text{cross}} \gets \emptyset \)  
10: \( \hat{o}_0 \gets o_s \), \( L_{\text{save}} \gets \{l_0, l_7, l_{15}, l_{23}, l_{31}\} \)  
11: for \( t = 1 \) to \( T \) do  
12: ResetIntra()  
13: \( \{K_t^{(l)}, V_t^{(l)}\} \gets \text{ExtractKV}(p_0, o_s, o_g, \hat{o}_{t-1}) \)  
14: AppendIntra(\( \{K_t^{(l)}, V_t^{(l)}\} \))  
15: Substep A: Action prediction at step \( t \)  
16: if \( \mathcal{M}_t^{\text{cross}} = \emptyset \) then  
17: \( \tilde{\mathcal{M}}_t \gets \mathcal{M}_t^{\text{intra}} \)  
18: else  
19: \( \tilde{\mathcal{M}}_t \gets \text{Merge}(\mathcal{M}_t^{\text{cross}}, \mathcal{M}_t^{\text{intra}}) \)  
20: end if  
21: \( \hat{a}_t \gets \text{EnhanceAndDecode}((p_0, o_s, o_g, \hat{o}_{t-1}), \tilde{\mathcal{M}}_t) \)  
22: Substep B: Navigation Visualization at step \( t \)  
23: \( \hat{o}_t \gets \text{EnhanceAndDecode}((p_0, o_s, o_g, \hat{o}_{t-1}, \hat{a}_t), \tilde{\mathcal{M}}_t) \)  
24: AppendCross(\( \mathcal{M}_t^{\text{intra}} \))  
25: end for  
26: return \( A_T = \{\hat{a}_1, \dots, \hat{a}_T\} \), \( O_T = \{\hat{o}_1, \dots, \hat{o}_T\} \)

# C EXPERIMENTS AND RESULTS

# C.1 EVALUATION METRIC DETAILS

We evaluate overall system performance using two complementary categories of metrics:

Navigation Quality: For goal-conditioned visual navigation performance, the Success Rate (SR) Li et al. (2024); Dong et al. (2025a) defines a trajectory as successful if its final distance $d$ to the goal is smaller than the agent’s average step size $\bar { s }$ (in meters). Formally, for trajectory $i$ among $N$ trajectories, with terminal estimate $\hat { p } _ { T } ^ { ( i ) }$ and goal position $p _ { g } ^ { ( i ) }$ , SR is computed as

$$
\mathbf {S R} = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbf {1} \left[ d \left(\hat {p} _ {T} ^ {(i)}, p _ {g} ^ {(i)}\right) <   \bar {s} \right],
$$

Absolute Trajectory Error (ATE) quantifies global trajectory accuracy by measuring the Euclidean distance between aligned points of the predicted and reference trajectories. Relative Pose Error (RPE) instead captures local consistency, computed as the deviation in relative motion between successive estimated and ground-truth poses Sturm et al. (2012).

(2) Visualization Quality: For navigation visualization, visual predictions are evaluated with a combination of standard structural and perceptual measures, namely SSIM Wang et al. (2004), PSNR Hore & Ziou (2010), LPIPS Zhang et al. (2018), and DreamSim Fu et al. (2023). The latter two are deep perceptual metrics specifically designed to more closely approximate human judgments. To assess longer-horizon stability under rollout, we introduce four metrics: SSIM@n, PSNR@n, LPIPS@n and DreamSim $@ n$ . Standard one-step metrics compare ground-truth next frame $o _ { t + 1 }$ with one-step prediction $\hat { o } _ { t + 1 } ^ { ( 1 ) }$ obtained from ground truth current observation and action $\left( o _ { t } , a _ { t + 1 } \right)$ . For horizon $n$ , we perform open-loop rollout that recursively feeds the model’s pre-

![](images/3f4fd04f16e29a3a7162a2568f442cb9d72551e7cb923d6e858564f343bf4e9c.jpg)  
Figure 9: Qualitative Comparisons on Go Stanford across UniWM, NWM, and NoMaD. The central trajectory plots highlight difference between predicted $A _ { T }$ and the ground-truth.

dicted observations back as inputs while conditioning on ground-truth action sequence $a _ { t + 1 : t + n + 1 }$ : $\mathrm { S S I M @ } n = \mathrm { S S I M } \big ( o _ { t + n } , \hat { o } _ { t + n } ^ { ( n ) } \big )$ , where $\hat { o } _ { t + n } ^ { ( n ) }$ is the observation prediction after $n$ rollouts, with $\mathrm { P S N R } @ n$ , $\mathrm { L P I P S } \ @ n$ and DreamSim $@ n$ defined analogously by replacing SSIM with the corresponding measure. We also provide detailed calculations for LPIPS and DreamSim here.

LPIPS: The Learned Perceptual Image Patch Similarity quantifies perceptual resemblance by computing weighted distances between deep feature activations extracted from pretrained vision backbones (e.g., AlexNet, VGG). By operating in a learned feature space, LPIPS better captures perceptually relevant differences than conventional low-level pixel-level measures.

DreamSim: DreamSim extends perceptual evaluation to the multimodal domain by measuring semantic alignment between generated images and a target text description. Given images $\{ I _ { i } \} _ { i = 1 } ^ { N }$ and a prompt $T$ , it is defined as:

$$
\operatorname {D r e a m S i m} \left(I _ {1: N}, T\right) = \frac {1}{N} \sum_ {i = 1} ^ {N} \frac {\left\langle f _ {\text {i m g}} \left(I _ {i}\right) , f _ {\text {t e x t}} (T) \right\rangle}{\left\| f _ {\text {i m g}} \left(I _ {i}\right) \right\| \cdot \left\| f _ {\text {t e x t}} (T) \right\|}. \tag {12}
$$

DreamSim leverages fused or fine-tuned visual–textual features (e.g., CLIP, OpenCLIP, DINO) trained on synthetic human similarity judgments, thereby further enhancing sensitivity to nuanced perceptual and semantic correspondences. By combining LPIPS and DreamSim, our evaluation jointly accounts for low-level visual fidelity and high-level semantic coherence, offering a balanced and human-aligned assessment across both structural and semantic dimensions.

# C.2 MORE QUALITATIVE RESULTS

We provide more qualitative results in Figs. 9, 10 and 11.

![](images/285f5fe6d863ae3c40317f8c58c1c56492303235989a487502a2168ce1892645.jpg)  
Figure 10: Qualitative Comparisons on ReCon and Scand across UniWM, NWM, and NoMaD. The central trajectory plots highlight difference between predicted $A _ { T }$ and the ground-truth.

![](images/42b3b1060c210f97c3f4480d2944e094998df463fffd13671da0d5bdf2d6a0aa.jpg)  
Figure 11: Qualitative Comparisons on HuRoN across UniWM, NWM, and NoMaD. The central trajectory plots highlight difference between predicted $A _ { T }$ and the ground-truth.
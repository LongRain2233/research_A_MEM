# Zeting Liu∗ Zida Yang∗ Zeyu Zhang∗† Hao Tang‡

Peking University

∗Equal contribution. †Project lead. ‡Corresponding author: bjdxtanghao@gmail.com.

![](images/3f34b4884cb7650f1100873ea27d15943712763abe22bcc76f83274c197afcb7.jpg)  
Build a block bridge: Place two green bars to form the bridge S structure, then fill the middle with purple blocks. bl

![](images/8235f56011770549dd57f051f74e606b871dd16ed32292491d0a62b72a341ea5.jpg)  
tack the blocks: Place the green block on top of the blue ock, then place the red block on top of the green block.

![](images/f7e4e921af0105893084c2b04a180cb2b2c912e0b43001a8be9e90aa6bc797f9.jpg)  
User-defined combined task: First stack the cups in order, then place the banana into the last cup.   
Figure 1. EvoVLA: Addressing stage hallucination in long-horizon manipulation. Our framework combines Stage-Aligned Reward (SAR), Pose-Based Object Exploration (POE), and Long-Horizon Memory to achieve robust performance on complex manipulation tasks. Example results shown: Block Bridge, Stack, and Cup Stacking with Insertion tasks across simulation and real-world deployment.

# Abstract

Long-horizon robotic manipulation remains challenging for Vision-Language-Action (VLA) models despite recent progress in zero-shot generalization and Sim2Real transfer. Current VLA models suffer from stage hallucination, where agents exploit coarse evaluation signals to shortcut multi-step tasks, reporting high progress without actual task completion. We present EvoVLA, a self-supervised VLA framework addressing this through three synergistic components: Stage-Aligned Reward (SAR) uses triplet contrastive learning with Gemini-generated hard negatives to prevent visual shortcuts; Pose-Based Object Exploration (POE) grounds curiosity in relative object-gripper pose rather than pixels; and Long-Horizon Memory with selective context and gated fusion stabilizes intrinsic shaping. Extensive evaluations on Discoverse-L, a long-horizon manipulation benchmark with three multi-stage tasks, demonstrate that EvoVLA improves average success by 10.2 percentage points over the strongest baseline (OpenVLA-OFT), reaching $6 9 . 2 \%$ , achieves $I . 5 \times$ better sample efficiency, and reduces stage hallucination from $3 8 . 5 \%$ to $1 4 . 8 \%$ . Realworld deployment on physical robots achieves $5 4 . 6 \%$ average success across four manipulation tasks, outperforming OpenVLA-OFT by 11.0 points, demonstrating effective Sim2Real transfer and robust generalization. Code:

https://github.com/AIGeeksGroup/EvoVLA.

Website: https://aigeeksgroup.github.io/ EvoVLA.

# 1. Introduction

Vision-Language-Action (VLA) models promise generalist robot policies by unifying perception, language, and control within a single large-scale backbone. Early systems such as Do-As-I-Can [2], Inner Monologue [22], and Code-as-Policies [28] demonstrated that language grounding can guide imitation and reinforcement learning, while recent foundation models, including PaLM-E [11], RT-X/RT-2 [5, 6], OpenVLA [25], and the $\pi _ { 0 }$ family [4, 40] with FAST tokenization [39], scale this recipe to web-scale corpora and thousands of robot demonstrations, achieving compelling zero-shot behavior on many short-horizon skills. Yet when tasks require dozens of semantically distinct stages and persistent state tracking, current VLAs still rely on brittle waypoints or ad-hoc progress classifiers; without explicit stage semantics and temporal grounding, autonomy often collapses despite strong pretraining.

We identify two long-horizon obstacles. First, reinforcement learning remains sample-inefficient: sparse rewards and high-dimensional observations induce “stage hallucination”, where the policy farms superficial visual cues to

appear successful. Self-supervised reinforcement learning (SSRL) offers dense intrinsic feedback, yet existing curiosity formulations [7, 38] treat vision-only novelty detection and quickly collapse in cluttered scenes. Second, longhorizon memory is fragile. Most “memory-augmented” VLAs compress history by averaging or truncation, causing catastrophic forgetting. Worse, the community lacks a benchmark that exposes memory drift or enables systematic training—without the right data, improved memory modules cannot be meaningfully evaluated.

These challenges motivate EvoVLA (Figure 1). To combat hallucination, we introduce a self-supervised reinforcement learning (SSRL) pipeline that couples stage-aware rewards with pose-grounded curiosity, delivering dense signals that respect task semantics. To curb memory forgetting, we take a system view: (i) a Long-Horizon Memory module selects, gates, and writes back only utility-critical context, and (ii) Discoverse-L, a long-horizon manipulation benchmark with ground-truth stage events for evaluation and taskaligned normalization, provides the training/evaluation substrate required to stress-test memory designs.

Our contributions are threefold. First, we present an SSRL framework that fuses stage-aligned rewards, posebased curiosity, and counterfactual shaping to suppress hallucination while retaining exploration. Second, we systematically tackle long-horizon memory by combining a context-selected gating module with the Discoverse-L benchmark, enabling both training and evaluation of memory-heavy VLAs. Third, we deliver comprehensive experiments across simulation and real robot deployments to validate robustness and sample efficiency.

Hence, we summarize our contributions as follows:

• We propose EvoVLA, a self-supervised long-horizon learning method, where SAR combined with Geminidriven hard negatives and POE provides dense, semantically consistent intrinsic feedback, enabling scalable VLA fine-tuning without additional labels.   
• We target long-horizon forgetting through a contextselected Long-Horizon Memory mechanism with selective attention and gated fusion, coupled with the Discoverse-L benchmark featuring three multi-stage manipulation tasks (18–74 stages), which together provide the community with both a method and a dataset/measurement suite for studying memory in longhorizon manipulation.   
• Extensive evaluations show EvoVLA lifts Discoverse-L success to $6 9 . 2 \%$ $( + 1 0 . 2$ points over OpenVLA-OFT [24]), reaches the $50 \%$ success threshold with $1 . 5 \times$ fewer environment steps, and attains $5 4 . 6 \%$ real-robot success $( + 1 1 . 0$ points vs. OpenVLA-OFT [24], $+ 1 6 . 9$ vs. $\pi _ { 0 }$ -FAST [40]) while keeping hallucination at $1 4 . 8 \%$ .

# 2. Related Work

VLA for Embodied Intelligence. Vision-Language-Action (VLA) models replace staged pipelines with end-toend multimodal learning [19, 31, 48, 49, 53, 54]. PaLM-E [11] and RT-X/RT-2 [5, 6] scale trajectory pretraining with frozen ViT encoders and action heads, improving generalization and Sim2Real transfer. OpenVLA [25] and Octo [34] advance open-source finetuning for real tasks, while RoboMamba [30] introduces SSM-based architectures for long sequences. Recent work on optimized finetuning [24] explores parallel decoding, action chunking, and continuous representations to boost VLA efficiency and performance. Yet two issues remain: stage hallucination (exploiting Vision-Language Model (VLM) scores without actual completion) and reward modeling dilemmas in highdimensional POMDPs. EvoVLA addresses both via stagealigned rewards and pose-grounded SSRL.

RL for VLA Fine-Tuning and Exploration in Embodied Tasks. Reinforcement learning (RL) improves VLA finetuning for long-horizon tasks. PPO [43] and SAC [15] optimize action heads but depend on careful reward design. RT-1/RT-2 [5, 6] combine policy optimization with large demonstration corpora, yet remain limited by sparse reward brittleness and hallucination. Recent 2025 VLA RL fine-tuning works explore consistency policies, chunked offline regimes, temporal feedback, and verified rewards [9, 14, 17, 18, 26, 27, 32, 47, 56]. Traditional RL suffers from sparse signals and annotation costs, causing hallucinations and instability. Self-supervised reinforcement learning (SSRL) adds dense intrinsic feedback for efficiency. ICM [38] introduces intrinsic rewards via forward dynamics prediction, Plan2Explore [44] plans curiosity via trajectory ensembles, and RaMP [8] designs random features for zero-shot world-model transfer. Recent VLAoriented SSRL [12, 29] jointly learns policies and representations, yet pixel-based curiosity remains brittle in cluttered scenes. EvoVLA grounds exploration in geometric pose structure rather than in pixels.

Counterfactual Shaping for Enhanced Reliability. Potential-based reward shaping [16, 33] mitigates reward bias and stage hallucination by adjusting returns without altering optimal policies. Reflexion [46] adds language introspection for long-horizon computation. Memoryaugmented systems such as JARVIS-1 [52] enable openworld multi-task planning; yet, they often rely on non-endto-end frameworks or short memory windows. EvoVLA bridges this gap by extending counterfactual shaping via SSRL post-training and fusing it with geometric pose-based exploration for data-efficient learning.

![](images/037074cadd56bc6527d8217171ebe5307864f91698dbfe0534a2da32e8dccee3.jpg)  
Figure 2. EvoVLA Data Engine. Aligned with Discoverse-L and the video-driven stage discovery pipeline to close the data–reward–policy loop.

# 3. The Proposed Method

# 3.1. Overview

EvoVLA couples a task-aligned data pipeline (Figure 2) with long-horizon control modules (Figure 3). Discoverse-L supplies three AIRBOT-Play tasks with 50 scripted rollouts each; we store trajectories in RLDS [1], compute task-aligned normalization, and run a multi-step prompting workflow with Gemini 2.5 Pro [13] to produce unified stage dictionaries of positive, negative, and hard-negative predicates (template, validation heuristics, and failure handling receive detailed coverage in Appendix 15). The same semantics supervise simulation training and Sim2Real deployment, where policies execute from language prompts; for unseen tasks we regenerate the dictionary from 50 teleoperated demonstrations.

The architecture centers on three interacting modules. SAR scores image-text triplets with CLIP [41] contrast and temporal smoothing, delivering stage-faithful dense rewards. POE grounds curiosity in relative gripper-to-object poses via paired forward/inverse world models to reduce spurious novelty from visual distractors. Long-Horizon Memory selects salient history tokens via attention-based context selection and gates intrinsic shaping so progress signals remain stable. All modules share the OpenVLA-OFT backbone and are optimized jointly via PPO.

Training follows a modular recipe: optionally warm-

start with supervised learning on the rollouts, then apply PPO [43] with task-aligned normalization while SAR, POE, and memory remain active. The following subsections detail each component.

# 3.2. Self-Supervised Reinforcement Learning

Our self-supervised learning strategy enables the agent to learn efficiently without manual reward engineering. Following prior work on curiosity-driven exploration [7, 38], we generate intrinsic rewards from pretext tasks. However, we ground these signals in stage-aligned evaluation and geometric structure rather than raw pixels, yielding more reliable learning signals. The combined reward is:

$$
\tilde {r} _ {t} = r _ {t} ^ {e} + \rho \left(r _ {t} ^ {\text {s t a g e}} + r _ {t} ^ {\text {c u r}} + r _ {t} ^ {\text {p r o g}}\right). \tag {1}
$$

where $\boldsymbol { r } _ { t } ^ { e }$ is the sparse external task reward, $r _ { t } ^ { \mathrm { s t a g e } }$ is the tstage-aligned reward, stems from POE, f comes $\boldsymbol { r } _ { t } ^ { \mathrm { c u r } }$ $r _ { t } ^ { \mathrm { p r o g } }$ from Long-Horizon Memory, and $\rho$ is the fixed intrinsic weight (we use $\rho = 0 . 6$ , selected via sensitivity analysis; Supp. Mat. Fig. 10). Below, we detail each component.

Stage-Aligned Reward. We evaluate stage completion through image-text contrastive learning, producing dense reward signals to guide policy learning. The key insight is that standard positive-negative contrastive learning is insufficient, so we need hard negatives that capture near-miss states to prevent policies from exploiting visual shortcuts.

Triplet text generation. For each stage $k$ , the Gemini 2.5 Pro [13] stage dictionary provides a triplet: positive $T _ { k } ^ { + }$ (completion state), mutually-exclusive negative $T _ { k } ^ { - }$ , and counterfactual hard negative $T _ { k } ^ { h - }$ (near-miss). The hard negative uses spatial/contact predicates (e.g., “grasping non-target object”) rather than visual features (e.g., “grasping blue cup”), ensuring robustness to appearance variations.

Image-text contrastive scoring. Given VLM encoder $\phi$ (CLIP [20, 21, 41]), we compute stage-aligned scores by comparing image observations against the triplet text descriptions:

$$
u _ {k} (t) = \sigma \left(\tau \left[ s _ {k} ^ {+} (t) - \max  \left\{s _ {k} ^ {-} (t), s _ {k} ^ {h} (t) \right\} \right]\right), \tag {2}
$$

$$
u _ {k} (t) \in [ 0, 1 ],
$$

where $\begin{array} { r l r } { s _ { k } ^ { + } ( t ) } & { { } = } & { \langle \phi _ { \mathrm { i m g } } ( o _ { t } ) , \phi _ { \mathrm { t e x t } } ( T _ { k } ^ { + } ) \rangle } \end{array}$ (similarly for $s _ { k } ^ { - } ( t ) , s _ { k } ^ { h } ( t ) )$ , $\tau$ is the temperature parameter, and $\sigma$ is the sigmoid function. This contrastive formulation forces the VLM to distinguish between near-miss states and actual completion.

Temporal smoothing. To stabilize score fluctuations and prevent noisy reward signals, we maintain a running average $\bar { u } _ { k }$ that is updated smoothly over time:

$$
\bar {u} _ {k} (t) = (1 - \alpha) \bar {u} _ {k} (t - 1) + \alpha u _ {k} (t). \tag {3}
$$

![](images/3f9df79924a20dd94361a70cdfca81e5a8573abd700df53f5e28a6ca9b522642.jpg)  
Figure 3. EvoVLA overview. Built on OpenVLA-OFT backbone, EvoVLA integrates three modules: Stage-Aligned Reward (SAR) with hard negatives and temporal smoothing, Pose-Based Object Exploration (POE) via world models, and Long-Horizon Memory with context selection and gated fusion. The framework couples with Discoverse-L for training and deploys to real robots.

where $\alpha$ is the smoothing coefficient. The stage reward is then computed as the progress in these smoothed scores:

$$
r _ {t} ^ {\text {s t a g e}} = \bar {u} _ {\kappa_ {t}} (t) - \bar {u} _ {\kappa_ {t}} (t - 1). \tag {4}
$$

where $\kappa _ { t }$ is the current active stage. We advance to stage $\kappa _ { t } + 1$ when a sliding window (size $m = 8$ ) of recent $r ^ { \mathrm { s t a g e } }$ values exceeds a threshold, indicating sustained progress in the current stage (Appendix 16). This approach filters out transient score fluctuations while preventing premature transitions.

Pose-Based Object Exploration. Pixel-based curiosity suffers from spurious novelty, with exploration driven by irrelevant changes, such as lighting variations or camera motion, rather than task progress. POE addresses this by grounding curiosity in task-relevant interaction dynamics between the object and the parallel-jaw gripper, focusing on how the gripper’s motion and contact evolve during manipulation.

Latent manipulation dynamics. We represent the manipulation state as $\overline { { z _ { t } = \psi ( T _ { \mathrm { e e } } ^ { - 1 } T _ { \mathrm { o b j } } ) } } \in \mathbb { R } ^ { 6 }$ , where $T _ { \mathrm { e e } }$ is the end-effector pose, $T _ { \mathrm { o b j } }$ is the object pose, and $\psi$ converts the relative transformation to a 6D vector in axis-angle plus translation representation (3D translation $+ ~ 3 \mathrm { D }$ axis-angle rotation). We train forward and inverse models:

$$
\hat {z} _ {t + 1} = f _ {\phi} \left(z _ {t}, a _ {t}\right), \tag {5}
$$

$$
\hat {a} _ {t} = g _ {\psi} \left(z _ {t}, z _ {t + 1}\right). \tag {6}
$$

These lightweight MLP models $2 \times 2 5 6$ units) learn the

dynamics of object manipulation without being distracted by visual nuisances.

Task-relevant intrinsic rewards. The curiosity reward encourages exploring states where the forward model makes large prediction errors, while the progress reward captures learning improvements:

$$
r _ {t} ^ {\operatorname {c u r}} = \frac {\eta}{2} \| \operatorname {s g} \left(\hat {z} _ {t + 1}\right) - z _ {t + 1} \| _ {2} ^ {2}, \tag {7}
$$

$$
r _ {t} ^ {\text {b a s e}} = \operatorname {R e L U} \left(\overline {{\mathcal {L} _ {F}}} (t - 1) - \overline {{\mathcal {L} _ {F}}} (t)\right). \tag {8}
$$

where $\operatorname { s g } ( \cdot )$ denotes stop-gradient (preventing policy from exploiting model errors), $\overline { { \mathcal { L } _ { F } } }$ is the forward loss with temporal smoothing, and $\eta \ : = \ : 1 . 0$ is the curiosity scale. By focusing on geometric task structure, POE substantially reduces spurious exploration caused by lighting changes and camera motion. The base progress term $r _ { t } ^ { \mathrm { b a s e } }$ is forwarded to Long-Horizon Memory for memory-conditioned gating before contributing to the combined reward.

# 3.3. Long-Horizon Memory

Memory Mechanism: Beyond Adjacent-Averaging Consolidation. Long-horizon tasks challenge memory management. Memory-augmented architectures range from Neural Map [37] to recent VLA-oriented MemoryVLA [45], yet many still compress sequences via averaging or truncation, which blurs stage-specific details and weakens temporal identifiability. In high-dimensional embodied settings, accumulated long-tail noise can amplify stage hallucination and hinder credit assignment in multi-step tasks. We

instead treat memory as context: at each step we select Top- $K$ history items as independent context tokens, fuse them with the current latent via a learned gate, and perform lightweight write-back with utility-guided eviction (usage, recency, redundancy) rather than adjacent averaging. This retains phase-specific information while keeping the store compact.

Context Selection via Attention. We extract a latent representation $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { d }$ from the OpenVLA-OFT backbone at each step (by pooling over the LLM’s hidden states), and we maintain a memory store $\mathcal { M } = \{ m _ { i } \in \mathbb { R } ^ { d } \} _ { i = 1 } ^ { L }$ with timestamps $\{ \tau _ { i } \}$ , where $L$ is the store capacity. To select relevant history, we define query, key, and value projections:

$$
q _ {t} = W _ {q} x _ {t},
$$

$$
k _ {i} = W _ {k} \left(m _ {i} + \operatorname {P E} \left(\tau_ {i}\right)\right), \tag {9}
$$

$$
v _ {i} = W _ {v} m _ {i},
$$

where $W _ { q } , W _ { k } , W _ { v } \in \mathbb R ^ { d \times d }$ are learnable linear maps, and $\mathrm { P E } ( \cdot )$ is a sinusoidal temporal positional encoding. Attention scores and the selected index set are computed as:

$$
a _ {i} = \frac {\exp \left(\left\langle q _ {t} , k _ {i} \right\rangle / \sqrt {d}\right)}{\sum_ {j = 1} ^ {L} \exp \left(\left\langle q _ {t} , k _ {j} \right\rangle / \sqrt {d}\right)}, \tag {10}
$$

$$
\mathcal {S} _ {t} = \operatorname {T o p K} \left(\left\{a _ {i} \right\} _ {i = 1} ^ {L}, K\right),
$$

where $K$ selects the most relevant history items. We prepend $\{ v _ { i } \ | \ i \in S _ { t } \}$ to the current sequence as independent context tokens, replacing static adjacent-averaging consolidation used in prior work.

Gated Fusion and Reward Modulation. Let $\begin{array} { r l } { \hat { h } _ { t } } & { { } = } \end{array}$ $\textstyle \sum _ { i \in S _ { t } } a _ { i } v _ { i }$ be the weighted context embedding. We fuse it with the current latent $x _ { t }$ via a learned gate:

$$
g _ {t} ^ {\mathrm {m e m}} = \sigma \left(w _ {g} ^ {\top} \left[ \hat {h} _ {t}; x _ {t} \right]\right), \tag {11}
$$

$$
\tilde {x} _ {t} = \left(1 - g _ {t} ^ {\mathrm {m e m}}\right) x _ {t} + g _ {t} ^ {\mathrm {m e m}} h _ {t}.
$$

where $w _ { g } \in \mathbb { R } ^ { 2 d }$ is a learnable weight vector and $[ \cdot ; \cdot ]$ denotes concatenation. The gate $g _ { t } ^ { \mathrm { m e m } }$ modulates how much history context influences the current representation. The memory gate also modulates the base progress reward rbaset : $r _ { t } ^ { \mathrm { b a s e } }$

$$
r _ {t} ^ {\text {p r o g}} = g _ {t} ^ {\text {m e m}} \cdot r _ {t} ^ {\text {b a s e}}. \tag {12}
$$

suppressing spurious progress signals when history context indicates unstable manipulation patterns (e.g., repeated failures, oscillations). We block gradients through $g _ { t - 1 } ^ { \mathrm { m e m } }$ to prevent reward hacking. Memory update writes back the fused representation $\tilde { x } _ { t }$ only to the selected items $S _ { t }$ , and eviction removes the least useful entry based on a utility score combining usage frequency, recency, and redundancy with existing items.

POE Integration and Effect. In contrast to Titans-MAC [3] and MemoryVLA [45], Long-Horizon Memory

uses selective context with gated fusion for reward-level modulation. The core mechanism operates on backbone latents $x _ { t }$ and history items $m _ { i }$ via the attention and gating mechanisms above. When combined with POE (full model, Table 2 row 5 vs. row 4 standalone), POE’s geometric pose tracking provides complementary grounding that enhances memory’s ability to distinguish task-relevant manipulation contexts from spurious visual changes, further improving robustness. This mechanism complements SAR temporal smoothing and aligns with the training objective $\mathcal { L }$ (13). Empirically, Long-Horizon Memory in standalone mode contributes $+ 2 . 4$ points of success and a 3.9-point hallucination reduction, with POE integration adding an additional $+ 3 . 1$ points, highlighting the value of both selective context and geometric grounding.

# 3.4. Training Objective

The full training objective combines policy gradient loss, value losses, and world model losses:

$$
\mathcal {L} = \mathcal {L} _ {\mathrm {P P O}} (\tilde {r}) + \lambda_ {F} \mathcal {L} _ {F} + \lambda_ {I} \mathcal {L} _ {I} - \lambda_ {\text {e n t}} H (\pi). \tag {13}
$$

where $\mathcal { L } _ { F }$ and $\mathcal { L } _ { I }$ are the MSE losses for the forward and inverse world models, $H ( \pi )$ is policy entropy, and $\lambda _ { F } , \lambda _ { I } , \lambda _ { \mathrm { e n t } }$ weight the auxiliary objectives (see Table 4 for values). We use PPO (Proximal Policy Optimization) [43] with dual critics $V _ { e } , V _ { i }$ for extrinsic/intrinsic returns, fusing advantages as $A _ { t } = ( 1 - \rho ) A _ { t } ^ { e } + \rho A _ { t } ^ { i }$ using GAE (Generalized Advantage Estimation) [42]. World models and the memory-conditioned GRU are trained via separate optimizers with stop-gradient to prevent reward exploitation.

# 4. Discoverse-L Benchmark

Benchmark Overview. Discoverse-L builds on the DIS-COVERSE simulator [23] with the AIRBOT-Play platform [10], offering 3 multi-stage manipulation tasks with varying difficulty: (1) Block Bridge (74 stages): place two bars to form a bridge structure, then fill with multiple blocks, the most complex task requiring precise multiobject coordination; (2) Stack (18 stages): stack three colored blocks in sequence, demanding accurate alignment; (3) Jujube-Cup (19 stages): place a jujube fruit into a cup and move the cup onto a plate, a relatively simpler setup with fewer contact-critical stages. Stage counts reflect our Gemini [13]-prompted dictionaries; the DISCOVERSE simulator provides fine-grained ground-truth annotations (e.g., 79 for Bridge) used only for validation of stage-discovery coverage. The 74-stage Bridge dictionary merges five simulator micro-adjustments into semantically equivalent steps $9 3 . 7 \%$ recall; Appendix 15), so reward supervision remains faithful while prompts stay compact. We generate 50 simulator rollouts per task with randomized initial conditions (Supp. Mat. Appendix 12 provides task-aligned normalization and evaluation protocol details).

# 5. Experiments

# 5.1. Experimental Design and Evaluation

For comparability across methods on Discoverse-L, we recommend initializing policies with the provided task-aligned normalization and optionally bootstrapping from the 50 demonstrations via supervised learning before engaging in interactive PPO training. Evaluation typically uses 50 interactive episodes per task with held-out seeds, identical camera setups, and fixed episode budgets.

Implementation Details. We build upon the OpenVLA-OFT architecture [24], which extends the original Open-VLA [25] with optimized fine-tuning techniques including parallel action decoding, action chunking, and L1 regression loss. The policy backbone follows OpenVLA-OFT: multi-view ViT towers (SigLIP, ICCV 2023 [55] + DI-NOv2, arXiv 2023 [36]) and a language model (Llama-2 7B [50]) with FiLM modulation and chunked parallel action decoding. For stage evaluation we use a separate frozen CLIP encoder (ViT-B/16 [41], $2 2 4 \times 2 2 4$ input) to compute image–text similarity for SAR; separating the evaluator prevents reward overfitting. The HR computation uses the same frozen CLIP with threshold $\theta \ = \ 0 . 7$ , selected via sensitivity analysis (Supp. Mat. Fig. 9) to balance falsepositive and false-negative rates. World models for posebased exploration are lightweight MLPs. Long-Horizon Memory uses a single-layer GRU with a sigmoid readout to modulate the progress reward. Training uses 8 parallel DISCOVERSE environments with PPO for 2M timesteps per seed across 3 random seeds. Hardware: $4 \times$ NVIDIA H20 GPUs (96 GB VRAM each); wall-clock training time is approximately 24 hours per seed. Complete hyperparameters are provided in the Supp. Mat.

Stage dictionaries are generated via the video-driven discovery workflow described above: for the three Discoverse-L tasks, we use ten simulator demonstrations per task; for the unseen real-world assembly task, we run the same pipeline on its 50 teleoperated demonstrations. The resulting semantics supervise both stage gating and SAR triplet selection across domains. We also load the task-aligned normalization statistics from Discoverse-L and inject them at policy initialization (matching the active task) to ensure feature magnitudes remain consistent between training and evaluation. Following prior practice, many implementations optionally apply a short behavior cloning warm-start before interactive RL fine-tuning. All simulation results report mean over three independent random seeds.

Evaluation Metrics. Success Rate (SR): Percentage of episodes completing the task within 400 steps. We cap episodes at 400 to accommodate the most complex task (Bridge, 74 stages) while preventing indefinite rollouts; successful runs typically finish in 50–100 steps for simpler tasks (Stack, Jujube-Cup) and 150–250 steps for Bridge.

Table 1. Discoverse-L benchmark (3 tasks; mean of 3 seeds, 50 evaluation episodes). All models use task-aligned normalization; EvoVLA adds SAR, POE, and Long-Horizon Memory to OpenVLA-OFT. The last row highlights improvements over OpenVLA-OFT.   

<table><tr><td>Model</td><td>Bridge</td><td>Jujube-Cup</td><td>Stack</td><td>Avg.</td></tr><tr><td>Octo [34]</td><td>24.8</td><td>33.7</td><td>29.1</td><td>29.2</td></tr><tr><td>OpenVLA [25]</td><td>32.6</td><td>42.0</td><td>37.5</td><td>37.4</td></tr><tr><td>π0[4, 40]</td><td>41.1</td><td>50.5</td><td>46.2</td><td>45.9</td></tr><tr><td>π0-FAST [39, 40]</td><td>47.4</td><td>56.9</td><td>52.8</td><td>52.4</td></tr><tr><td>OpenVLA-OFT [24]</td><td>54.1</td><td>63.5</td><td>59.4</td><td>59.0</td></tr><tr><td>EvoVLA (OURS)</td><td>65.3</td><td>72.6</td><td>69.7</td><td>69.2</td></tr><tr><td>Δ vs. OFT</td><td>+11.2</td><td>+9.1</td><td>+10.3</td><td>+10.2</td></tr></table>

Sample Efficiency: Number of environment timesteps required to reach a $50 \%$ success rate.

Hallucination Rate (HR): The fraction of high VLM scores corresponding to failed actual completion. Formally, let $u _ { k } ( t ) \in [ 0 , 1 ]$ be the VLM score for stage $k$ at time $t$ , and $c _ { k } ( t ) \ \in \ \{ 0 , 1 \}$ be the ground-truth completion indicator. Hallucination rate (HR) is:

$$
\mathrm {H R} = \frac {\mathbb {E} \left[ \mathbb {1} \left\{u _ {k} (t) > \theta \wedge c _ {k} (t) = 0 \right\} \right]}{\mathbb {E} \left[ \mathbb {1} \left\{u _ {k} (t) > \theta \right\} \right]}. \tag {14}
$$

where $u _ { k } ( t ) > \theta$ (we use $\theta = 0 . 7$ ) indicates “high VLM score” and $c _ { k } ( t ) = 0$ indicates failed completion, verified using task success signals from the DISCOVERSE simulator. The expectation averages over evaluation episodes, timesteps, and active stages. We emphasize that $c _ { k } ( t )$ is obtained solely at evaluation time from DISCOVERSE stageevent triggers to compute the metric; it is never used for training or model selection.

Real-World Deployment. Beyond simulation, we validate EvoVLA on a physical AIRBOT-Play robot (details in Section 5.3).

# 5.2. Simulation Results

Baselines. We benchmark against state-of-the-art VLA backbones trained on Discoverse-L: Octo [34], Open-VLA [25], $\pi _ { 0 }$ and $\pi _ { 0 }$ -FAST [39, 40], and OpenVLA-OFT [24]. EvoVLA is obtained by augmenting the OpenVLA-OFT architecture with our SAR module, POE module, Long-Horizon Memory module, and task-aligned normalization to address stage hallucination.

Main Simulation Results. Table 1 summarizes quantitative comparisons against the baselines. Starting from Octo $( 2 9 . 2 \% )$ and OpenVLA $( 3 7 . 4 \% )$ , progressively stronger models achieve $4 5 . 9 \%$ $\left( \pi _ { 0 } \right)$ ), $5 2 . 4 \%$ $\scriptstyle { \pi _ { 0 } }$ -FAST), and $5 9 . 0 \%$ (OpenVLA-OFT) through richer pretraining and architectural improvements. EvoVLA lifts average success to $6 9 . 2 \%$ , improving upon the strongest baseline by $+ 1 0 . 2$ absolute points. Per-task analysis shows consistent gains: Bridge $( + 1 1 . 2$ points, $6 5 . 3 \%$ vs. $5 4 . 1 \%$ ), Jujube-Cup $( + 9 . 1$

![](images/0df8a8e3876e7330d02eb5c6a7e1d29f37f78146d8c70183342e0a46e8389561.jpg)  
Figure 4. Simulation and real-world rollouts. Columns depict temporal progression, rows pair the same task family across domains. Left: dual-camera Discoverse-L rollouts for Stack, Bridge, Jujube-Cup; right: AIRBOT-Play deployments for Stack, Bridge, Insert. The alignment highlights consistent gripper–object interactions and viewpoints after Sim2Real transfer.

points, $7 2 . 6 \%$ vs. $6 3 . 5 \%$ ), and Stack $( + 1 0 . 3$ points, $6 9 . 7 \%$ vs. $5 9 . 4 \%$ ), demonstrating that our approach generalizes across tasks with varying complexity (18–74 stages) while using the same backbone architecture. Figure 4 complements these numbers by juxtaposing dual-camera rollouts in simulation and on the AIRBOT-Play robot, highlighting that EvoVLA maintains consistent gripper-object interactions across domains. Full-resolution sequences for simulation-only and real-world-only views are provided in Supp. Mat. Figures 7 and 8.

Sample Efficiency. EvoVLA reaches a $50 \%$ average success rate in approximately $6 \times 1 0 ^ { 5 }$ environment steps (averaged over 3 seeds), whereas OpenVLA-OFT requires approximately $9 \times 1 0 ^ { 5 }$ steps, demonstrating $1 . 5 \times$ better sample efficiency. This acceleration stems from dense intrinsic feedback: SAR provides stage-level guidance at every timestep, POE grounds exploration in task-relevant geometric structures, and Long-Horizon Memory stabilizes learning signals.

Hallucination Reduction. Simulation rollouts report a $1 4 . 8 \%$ Hallucination Rate versus $3 8 . 5 \%$ for OpenVLA-OFT, i.e., a 23.7-point drop (Table 2). The precise contribution of each module—hard negatives, temporal smoothing, Long-Horizon Memory, and POE—is analyzed in Section 5.4, so we defer further discussion there.

# 5.3. Real-World Evaluation

We deploy EvoVLA on a physical AIRBOT-Play robot (additional rollout snapshots shown in Figure 4) to assess its real-world performance.

We benchmark against two baselines: $\pi _ { 0 }$ -FAST [39, 40] and OpenVLA-OFT [24], both trained with the same real-world setup. Evaluation comprises two parts: Sim2Real transfer of simulation-trained policies to the three

Discoverse-L tasks, and training a new policy from scratch on an unseen assembly task.

Sim2Real Transfer (Three Tasks). For the three Discoverse-L tasks (Block Bridge, Jujube-Cup, Stack), we directly deploy the simulation-trained policies to the physical robot without any real-world retraining. Each policy receives only a language prompt describing the task (e.g., “Build block bridge: place two bars, fill with blocks”). Third-person and wrist-mounted camera recordings confirm successful task completion across all three scenarios, demonstrating effective Sim2Real transfer enabled by the video-driven stage semantics and task-aligned normalization learned in simulation.

Unseen Task: Cup Stacking with Insertion. We further evaluate a novel assembly task that stacks four cups and inserts a banana-shaped object into the final stack. Fifty teleoperated demonstrations provide stage semantics via Gemini 2.5 Pro and refreshed normalization statistics. The policy is warm-started with BC, then fine-tuned on-robot with PPO [43] for roughly 5k steps under safety monitoring. During fine-tuning, SAR queries CLIP [41] against the wrist view, POE tracks AprilTag-based [35] relative poses, and Long-Horizon Memory gates the progress reward, with third-person and wrist-camera videos documenting successful execution.

Main Real-World Results. Figure 6 and Table 3 show that EvoVLA attains $5 4 . 6 \%$ average success across the three Sim2Real tasks plus the on-robot insertion task, surpassing OpenVLA-OFT by $+ 1 1 . 0$ points overall and by $+ 9 . 2 -$ $+ 1 3 . 4$ points on each task. The unseen insertion task sees the largest gain $5 5 . 2 \%$ vs. $4 1 . 8 \%$ ). Third-person and wristmounted recordings confirm successful executions for all tasks, highlighting consistent Sim2Real behavior.

Real-World Qualitative Case Study. Figure 5 provides a

![](images/22c8603d06f4e600d822c7590d4bfc65ad2d8886b67500ead7d3448704e96af5.jpg)  
Sub-task 1: Stack leftmost block on middle block   
Sub-task 2: Stack rightmost block on top   
Figure 5. Stack task qualitative comparison. Columns (1–5) cover sub-task 1 (left block onto middle) then sub-task 2 (right block on top). Top: OpenVLA-OFT (cross markers) opens before contact, dithers, misaligns, and drops the block. Bottom: EvoVLA (check marks) delays opening until contact, aligns within a few corrections, and leaves a stable stack, matching the hallucination reductions in Section 5.3.

![](images/5f12915f1431a7d0894d3f29ee9d5fa50584de06fbd030907f7c4c14c6b34f2b.jpg)  
Figure 6. Real-world evaluation. Green arrows: absolute gains over OpenVLA-OFT $( + 9 . 2 \mathrm { - } + 1 3 . 4 $ points). Blue labels: relative gains $( + 1 8 . 4 \% - + 3 2 . 1 \%$ ). Horizontal dashed lines mark model averages $( 3 7 . 7 \%$ , $4 3 . 6 \%$ , $5 4 . 6 \%$ ); the vertical dashed line splits Sim2Real transfers (left) from the on-robot Insert task (right).

qualitative case study on the Stack task. OpenVLA-OFT exhibits premature gripper opening, prolonged dithering, and misalignment (top-row columns marked with cross symbols), whereas EvoVLA maintains contact-aware control and precise placement (bottom-row check marks), yielding a stable three-block tower. These observations align with the quantitative Stack gains in Table 2 and reinforce that stage-aligned shaping and pose-grounded exploration mitigate stage hallucination on real robots.

# 5.4. Ablation Studies

Table 2 quantifies the cumulative gains: hard negatives add $+ 2 . 8$ SR / -7.3 HR by forcing discrimination of near-miss states; temporal smoothing contributes $+ 1 . 9 \ / \ - 7 . 8$ $+ 1 . 9$ by filtering score spikes; Long-Horizon Memory supplies $+ 2 . 4 /$ -3.9 via selective context retention; POE adds $+ 3 . 1 / - 4 . 7$ b y grounding curiosity in task geometry. Together they deliver $+ 1 0 . 2$ points SR and -23.7 points HR over OpenVLA-OFT,

Table 2. Component ablation. Progressive addition (averaged across 3 tasks, 3 seeds). Starting from OpenVLA-OFT baseline, we cumulatively add: hard negatives, temporal smoothing, Long-Horizon Memory, and POE.   

<table><tr><td>Method</td><td>SR (%)</td><td>HR (%)</td></tr><tr><td>OpenVLA-OFT (Baseline)</td><td>59.0</td><td>38.5</td></tr><tr><td>+ Hard Negatives (Th−)</td><td>61.8</td><td>31.2</td></tr><tr><td>+ Temporal Smoothing</td><td>63.7</td><td>23.4</td></tr><tr><td>+ Long-Horizon Memory</td><td>66.1</td><td>19.5</td></tr><tr><td>+ Pose-based Exploration</td><td>69.2</td><td>14.8</td></tr></table>

and removing any single component forfeits at least 2.4 success points. Figure 5 visualizes these improvements on the Stack task.

Hyperparameter sensitivity analyzes (Supp. Mat. Figures 9, 10) demonstrate robustness: the CLIP threshold $\theta$ remains effective within [0.65, 0.75], and intrinsic weight $\rho$ shows stable performance $\pm 2 . 4$ points) across [0.3, 0.9], confirming that our choices of $\theta \ : = \ : 0 . 7$ and $\rho ~ = ~ 0 . 6$ are well-justified.

# 6. Conclusion

We identified and addressed stage hallucination, where policies fool VLM evaluators without task completion. Our EvoVLA framework introduces stage-aligned rewards with triplet contrastive learning and temporal smoothing, longhorizon memory that stabilizes intrinsic shaping over extended interactions, and pose-based exploration, which grounds curiosity in the relative object-gripper pose space. These components work synergistically to prevent visual shortcuts and substantially reduce spurious exploration. Comprehensive ablation studies demonstrate that each component contributes significantly to performance. Experi-

ments demonstrate a $6 9 . 2 \%$ average success rate $( + 1 0 . 2$ points over the strongest baseline), $1 . 5 \times$ improved sample efficiency (reaching $50 \%$ success with fewer environment steps), and a Hallucination Rate reduction to $1 4 . 8 \%$ across the three Discoverse-L tasks.

# References

[1] Rishabh Agarwal, Alhussein Fawzi, Andre Barreto, Erwin Coumans, Roy Fox, Mohammad Gheshlaghi Azar, Yuval Tassa, Hado van Hasselt, Matteo Hessel, Remi Munos, et al. Rlds: Reusable, extensible, and composable datasets for reinforcement learning. In Advances in Neural Information Processing Systems, pages 5317–5330, 2022. 3   
[2] Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022. 1   
[3] Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time. arXiv preprint arXiv:2501.00663, 2024. 5   
[4] Kevin Black, Noah Brown, et al. $\pi _ { 0 }$ : A vision-languageaction flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 1, 6   
[5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022. 1, 2   
[6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pages 2165–2183. PMLR, 2023. 1, 2   
[7] Yuri Burda, Harrison Edwards, Amos Storkey, and Oleg Klimov. Exploration by random network distillation. arXiv preprint arXiv:1810.12894, 2018. 2, 3   
[8] Boyuan Chen, Chuning Zhu, Pulkit Agrawal, Kaiqing Zhang, and Abhishek Gupta. Self-supervised reinforcement learning that transfers using random features. Advances in Neural Information Processing Systems, 36:56411–56436, 2023. 2   
[9] Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, and Dongbin Zhao. Conrft: A reinforced fine-tuning method for vla models via consistency policy. arXiv preprint arXiv:2502.05450, 2025. 2   
[10] Discover Robotics. Airbot play intelligent light manipulator. https://airbots.online/zh, 2024. Accessed: 2025-11-08. 5   
[11] Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palme: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. 1, 2   
[12] Yilun Du, Chuang Gan, and Phillip Isola. Curious representation learning for embodied intelligence. In Proceedings of

the IEEE/CVF International Conference on Computer Vision (ICCV), pages 10388–10397, 2021. 2   
[13] Google DeepMind. Gemini api documentation. https: //ai.google.dev/gemini-api/docs, 2024. Accessed: 2025-11-08. 3, 5, 1, 4   
[14] Yanjiang Guo, Jianke Zhang, Xiaoyu Chen, Xiang Ji, Yen-Jen Wang, Yucheng Hu, and Jianyu Chen. Improving visionlanguage-action model with online reinforcement learning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025. arXiv:2501.16664. 2   
[15] Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, et al. Soft actor-critic algorithms and applications. arXiv preprint arXiv:1812.05905, 2018. 2   
[16] Anna Harutyunyan, Sam Devlin, Peter Vrancx, and Ann Nowe. Expressing arbitrary reward functions as potential- ´ based advice. In Proceedings of the AAAI Conference on Artificial Intelligence, 2015. 2   
[17] Jiaheng Hu, Rose Hendrix, Ali Farhadi, Aniruddha Kembhavi, Roberto Mart´ın-Mart´ın, Peter Stone, Kuo-Hao Zeng, and Kiana Ehsani. Flare: Achieving masterful and adaptive robot policies with large-scale reinforcement learning finetuning. In 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025. arXiv:2409.16578, accepted to ICRA 2025. 2   
[18] Dongchi Huang, Zhirui Fang, Tianle Zhang, Yihang Li, Lin Zhao, and Chunhe Xia. Co-rft: Efficient fine-tuning of vision-language-action models through chunked offline reinforcement learning. arXiv preprint arXiv:2508.02219, 2025. 2   
[19] Ting Huang, Zeyu Zhang, and Hao Tang. 3d-r1: Enhancing reasoning in 3d vlms for unified scene understanding. arXiv preprint arXiv:2507.23478, 2025. 2   
[20] Ting Huang, Zeyu Zhang, Yemin Wang, and Hao Tang. 3d coca: Contrastive learners are 3d captioners. arXiv preprint arXiv:2504.09518, 2025. 3   
[21] Ting Huang, Zeyu Zhang, Ruicheng Zhang, and Yang Zhao. Dc-scene: Data-centric learning for 3d scene understanding. arXiv preprint arXiv:2505.15232, 2025. 3   
[22] Wenlong Huang, Fei Xia, Ted Xiao, Harris Chan, Jacky Liang, Pete Florence, Andy Zeng, Jonathan Tompson, Igor Mordatch, Yevgen Chebotar, et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2022. 1   
[23] Yufei Jia, Guangyu Wang, Yuhang Dong, Junzhe Wu, Yupei Zeng, Haonan Lin, Zifan Wang, Haizhou Ge, Weibin Gu, Kairui Ding, Zike Yan, Yunjie Cheng, Yue Li, Ziming Wang, Chuxuan Li, Wei Sui, Lu Shi, Guanzhong Tian, Ruqi Huang, and Guyue Zhou. Discoverse: Efficient robot simulation in complex high-fidelity environments. arXiv preprint arXiv:2507.21981, 2025. Code available at the URL above. 5   
[24] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. In Robotics: Science and Systems (RSS), Los Angeles, USA, 2025. arXiv:2502.19645. 2, 6, 7, 1

[25] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An opensource vision-language-action model. In Proceedings of the 8th Conference on Robot Learning, pages 2679–2713. PMLR, 2025. Code available at the URL above. 1, 2, 6   
[26] Hengtao Li, Pengxiang Ding, Runze Suo, Yihao Wang, Zirui Ge, Dongyuan Zang, Kexian Yu, Mingyang Sun, Hongyin Zhang, Donglin Wang, et al. Vla-rft: Vision-language-action reinforcement fine-tuning with verified rewards in world simulators. arXiv preprint arXiv:2510.00406, 2025. 2   
[27] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhaohui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, et al. Simplevla-rl: Scaling vla training via reinforcement learning. arXiv preprint arXiv:2509.09674, 2025. 2   
[28] Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, and Andy Zeng. Code as policies: Language model programs for embodied control. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pages 9493–9500. IEEE, 2023. 1   
[29] Xinran Liang, Anthony Han, Wilson Yan, Aditi Raghunathan, and Pieter Abbeel. Alp: Action-aware embodied learning for perception. arXiv preprint arXiv:2306.10190, 2023. 2   
[30] Jiaming Liu, Mengzhen Liu, Zhenyu Wang, Pengju An, Xiaoqi Li, Kaichen Zhou, Senqiao Yang, Renrui Zhang, Yandong Guo, and Shanghang Zhang. Robomamba: Efficient vision-language-action model for robotic reasoning and manipulation. Advances in Neural Information Processing Systems, 37:40085–40110, 2024. 2   
[31] Qingxiang Liu, Ting Huang, Zeyu Zhang, and Hao Tang. Nav-r1: Reasoning and navigation in embodied scenes. arXiv preprint arXiv:2509.10884, 2025. 2   
[32] Guanxing Lu, Wenkai Guo, Chubin Zhang, Yuheng Zhou, Haonan Jiang, Zifeng Gao, Yansong Tang, and Ziwei Wang. Vla-rl: Towards masterful and general robotic manipulation with scalable reinforcement learning. arXiv preprint arXiv:2505.18719, 2025. 2   
[33] Andrew Y Ng, Daishi Harada, and Stuart Russell. Policy invariance under reward transformations: Theory and application to reward shaping. In ICML, pages 278–287, 1999. 2   
[34] Octo Contributors. Octo: Open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024. Project: https : / / octo - models . github . io/, Code: https://github.com/octo-models/octo. 2, 6   
[35] Edwin Olson. Apriltag: A robust and flexible visual fiducial system. In 2011 IEEE International Conference on Robotics and Automation, pages 3400–3407. IEEE, 2011. 7   
[36] Maxime Oquab, Jean-Baptiste Darcet, Theodoros Moutakanni, Divya Mane, Jiaze Chen, Daniel Haziza, Chen Sun, Alexandra Sasha Luccioni, Daniel Miller, Sebastien ´ Lachapelle, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 6

[37] Emilio Parisotto and Ruslan Salakhutdinov. Neural map: Structured memory for deep reinforcement learning. In International Conference on Learning Representations, 2018. 4   
[38] Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In International conference on machine learning, pages 2778–2787. PMLR, 2017. 2, 3   
[39] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for visionlanguage-action models. arXiv preprint arXiv:2501.09747, 2025. 1, 6, 7   
[40] Physical Intelligence Lab. Openpi: Open-source $\pi$ -series vision-language-action policies. https : / / github . com/Physical-Intelligence/openpi, 2024. Accessed: 2025-11-08. 1, 2, 6, 7   
[41] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, pages 8748–8763, 2021. 3, 6, 7   
[42] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015. 5   
[43] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. 2, 3, 5, 7   
[44] Ramanan Sekar, Oleh Rybkin, Kostas Daniilidis, Pieter Abbeel, Danijar Hafner, and Deepak Pathak. Planning to explore via self-supervised world models. In International conference on machine learning, pages 8583–8592. PMLR, 2020. 2   
[45] Hao Shi, Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu, Tiancai Wang, Erjin Zhou, Haoqiang Fan, Xiangyu Zhang, and Gao Huang. Memoryvla: Perceptual-cognitive memory in vision-language-action models for robotic manipulation. arXiv preprint arXiv:2508.19236, 2025. 4, 5   
[46] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023. 2   
[47] Junyang Shu, Zhiwei Lin, and Yongtao Wang. Rftf: Reinforcement fine-tuning for embodied agents with temporal feedback. arXiv preprint arXiv:2505.19767, 2025. 2   
[48] Zirui Song, Guangxian Ouyang, Meng Fang, Hongbin Na, Zijing Shi, Zhenhao Chen, Fu Yujie, Zeyu Zhang, Shiyu Jiang, Miao Fang, et al. Hazards in daily life? enabling robots to proactively detect and resolve anomalies. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7399–7415, 2025. 2   
[49] Zirui Song, Guangxian Ouyang, Mingzhe Li, Yuheng Ji, Chenxi Wang, Zixiang Xu, Zeyu Zhang, Xiaoqing Zhang,

Qian Jiang, Zhenhao Chen, et al. Maniplvm-r1: Reinforcement learning for reasoning in embodied manipulation with large vision-language models. arXiv preprint arXiv:2505.16517, 2025. 2   
[50] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Sharan Batra, Prajjwal Bhargava, Moya Chen Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023. 6   
[51] Gu Wang, Fabian Manhardt, Federico Tombari, and Xiangyang Ji. Gdr-net: Geometry-guided direct regression network for monocular 6d object pose estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16611–16621, 2021. 1   
[52] Zihao Wang, Shaofei Cai, Anji Liu, Yonggang Jin, Jinbing Hou, Bowei Zhang, Haowei Lin, Zhaofeng He, Zilong Zheng, Yaodong Yang, et al. Jarvis-1: Open-world multitask agents with memory-augmented multimodal language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024. 2   
[53] Zhengri Wu, Yiran Wang, Yu Wen, Zeyu Zhang, Biao Wu, and Hao Tang. Stereoadapter: Adapting stereo depth estimation to underwater scenes. arXiv preprint arXiv:2509.16415, 2025. 2   
[54] Angen Ye, Zeyu Zhang, Boyuan Wang, Xiaofeng Wang, Dapeng Zhang, and Zheng Zhu. Vla-r1: Enhancing reasoning in vision-language-action models. arXiv preprint arXiv:2510.01623, 2025. 2   
[55] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 6   
[56] Hongyin Zhang, Zifeng Zhuang, Han Zhao, Pengxiang Ding, Hongchao Lu, and Donglin Wang. Reinbot: Amplifying robot visual-language manipulation with reinforcement learning. arXiv preprint arXiv:2505.07395, 2025. 2

# Supplementary Material

# 7. Limitations and Discussion

# Limitations

• Sim2Real gap: Quantitative simulation results are complemented by real-robot demonstrations; systematic real-world evaluation remains future work.   
• Dependency on 6D poses: Requires accurate object tracking (simulator ground-truth or AprilTag markers). Future work will integrate learned 6D pose estimation [51].   
• Video-driven stage discovery coverage: Automatic stage dictionaries reduce manual verification, yet still rely on demonstration diversity and prompt quality; rare corner cases may require additional data curation.   
• Compute cost: Training requires $4 \times \mathrm { H } 2 0$ GPUs $\times 2 4 \mathrm { h }$ per seed (72h total for 3 seeds). Optimizing sample efficiency remains important for scalability.

Threats to Validity Results may vary under dataset or simulator changes (e.g., asset versions), implementation choices for baselines (pre-processing and augmentations), or unseen distribution shifts. We mitigate these threats via strict budget matching (2M steps across methods), identical backbones and input resolutions, and averaging over 3 independent random seeds.

# 8. Reproducibility Statement

We will release EvoVLA training and evaluation code, stage dictionaries, and RLDS-formatted trajectories upon publication. Experiments were conducted on $4 \times$ NVIDIA H20 GPUs with deterministically seeded runs. Table 4 provides complete hyperparameter settings for all components.

# 9. Real-World Detailed Results

Table 3. Real-world evaluation on four tasks. Each model–task pair uses at least 20 valid physical trials (episodes without safety aborts); some configurations include additional valid trials, yielding one-decimal precision. The three Discoverse-L tasks deploy simulation-trained policies, whereas the insertion task is trained on-robot (BC warm-start + PPO).   

<table><tr><td>Model</td><td>Bridge</td><td>Jujube-Cup</td><td>Stack</td><td>Insert (OOD)</td><td>Avg.</td></tr><tr><td>π0-FAST [39, 40]</td><td>30.5</td><td>43.9</td><td>41.0</td><td>35.3</td><td>37.7</td></tr><tr><td>OpenVLA-OFT [24]</td><td>35.3</td><td>50.1</td><td>47.2</td><td>41.8</td><td>43.6</td></tr><tr><td>EvoVLA (OURS)</td><td>45.6</td><td>59.3</td><td>58.4</td><td>55.2</td><td>54.6</td></tr></table>

# 10. Additional Rollout Visualizations

This section expands the qualitative evidence from Figure 4 by presenting the full rollout sequences. Figure 7 provides high-resolution Discoverse-L execution traces for the place, bridge, and stack tasks, illustrating how EvoVLA maintains stable gripper-object contact throughout each stage. Figure 8 mirrors this analysis on the AIRBOT-Play platform for stack, bridge, and insert executions, highlighting consistent Sim2Real behavior across synchronized third-person and wrist-mounted views.

# 11. Problem Formulation

POMDP Setting We model long-horizon robotic manipulation as a partially observable Markov decision process (POMDP) defined by the tuple $( \mathcal { S } , \mathcal { A } , \mathcal { O } , T , R , \Omega , \gamma )$ , where $s$ is the latent state space (object poses and robot configuration), $\mathcal { A }$ the continuous action space (6-DoF end-effector twist deltas plus a 1-D gripper command; total 7-D), $\mathcal { O }$ the observation space (multi-view RGB $^ +$ proprioception), $T \colon S \times { \mathcal { A } } \to S$ the transition function, $R \colon S \times { \mathcal { A } } \to \mathbb { R }$ the extrinsic reward (sparse task success), $\Omega \colon S  \mathcal { O }$ the observation function, and $\gamma \in [ 0 , 1 )$ the discount factor.

Task Specification and Stage Decomposition Each task is specified by a natural-language instruction $\ell$ (e.g., “Build block bridge: place two bars, fill with blocks”) and decomposes into $K$ sequential stages indexed by $k \in \{ 1 , \ldots , K \}$ . We obtain this decomposition via video-driven discovery: given $N$ demonstration rollouts for task $\ell$ , we apply Gemini 2.5 Pro [13] to align them and extract a stage dictionary

$$
\mathcal {D} = \left\{\left(T _ {k} ^ {+}, T _ {k} ^ {-}, T _ {k} ^ {h -}\right) \right\} _ {k = 1} ^ {K}. \tag {15}
$$

where $T _ { k } ^ { + }$ is a text predicate describing successful completion of stage $k$ (e.g., “gripper contacts target bar; fingers closed; stable grasp”), $T _ { k } ^ { - }$ is a mutually exclusive failure predicate (e.g., “gripper far from all objects; fingers open at init”), and $T _ { k } ^ { h - }$ is a counterfactual near-miss predicate that differs by a single spatial or contact constraint (e.g., “gripper near target bar but no contact; fingers open”). See Appendix 15 for complete examples from the Bridge task (Gemini produced 74 stages out of 79 simulator groundtruth stages), Stack task (18 stages), and Jujube-Cup task (19 stages).

During policy execution, we maintain a stage tracker that consumes the observation history $\mathcal { H } _ { t } = \{ o _ { 0 } , . . . , o _ { t } \}$ and outputs the current stage index $\kappa _ { t }$ . The tracker advances

![](images/7bac19ad276bfc7e4e6eaea9be6395b840cf725bd6cb0021c011ee03b9b4a8bb.jpg)  
Figure 7. Supplementary Discoverse-L rollouts. Each row shows a full episode for place, bridge, and stack with dual third-person/wrist views; columns denote temporal progression.

![](images/2b305ccad05cb9f28ae284ea678d79d6708e6830adc107aa8a6fb2b2135e7a81.jpg)  
Figure 8. Supplementary AIRBOT-Play rollouts. Each row shows stack, bridge, and insert executions with synchronized third-person and wrist cameras in the real world.

$\kappa _ { t } ~  ~ \kappa _ { t } + 1$ when a sliding window of $m \ : = \ : 8$ recent stage rewards (computed via image-text similarity between $o _ { t }$ and $T _ { \kappa _ { t } } ^ { + }$ ) exceeds a threshold, as detailed in Appendix 16. Crucially, no ground-truth stage labels from the simulator are consumed during training or evaluation; stage progression relies solely on the learned VLM scores and the stage dictionary semantics. In practice, the stage dictionary $\mathcal { D }$ supervises SAR during simulation training and is not reestimated for Sim2Real deployment on the same three realworld tasks; the learned policy implicitly internalizes these semantics. For the unseen real-world task trained on-robot, we generate its own dictionary from 50 teleoperated demonstrations.

Policy objective. The agent learns a policy $\pi _ { \theta } \colon { \mathcal { O } } \ \times$ $\{ 1 , \dots , K \} \to { \mathcal { A } }$ that conditions on the current observation and stage index. We maximize the expected discounted return

$$
J (\theta) = \mathbb {E} _ {\xi \sim \pi_ {\theta}} \left[ \sum_ {t = 0} ^ {H} \gamma^ {t} \tilde {r} _ {t} \right]. \tag {16}
$$

where ${ \boldsymbol \xi } = ( o _ { 0 } , a _ { 0 } , \tilde { r } _ { 0 } , \dots )$ is a trajectory, $H$ is the episode horizon (400 steps in our DISCOVERSE experiments), $\gamma =$ 0.995 is the discount factor (standard for long-horizon manipulation), and $\tilde { r } _ { t }$ is the combined reward from the main paper, which balances sparse task-success signal $( r _ { t } ^ { e } )$ , dense stage-aligned feedback from SAR $( r _ { t } ^ { \mathrm { s t a g e } } )$ , and exploration bonuses from POE $( r _ { t } ^ { \mathrm { c u r } } , r _ { t } ^ { \mathrm { p r o g } } )$ with fixed weight $\rho = 0 . 6$ . Policies are initialized from an OpenVLA-OFT backbone (7B parameters: frozen SigLIP+DINOv2 towers, trainable Llama-2-7B language head, and action decoder) and finetuned via PPO for 2M environment steps across 8 parallel simulators.

# 12. Benchmark Technical Details

Task-Aligned Normalization Cross-task action and state distributions vary notably, so we compute task-specific normalization statistics $( ^ { \circ } { \bf q 0 1 } / { \bf q 9 9 } / { \bf m e a n } / { \bf s t d ^ { \circ } } )$ from the 50 demonstrations per task and release them alongside the trajectories. During our experiments the environment wrapper

injects the corresponding statistics for the active task to keep feature magnitudes consistent between training and evaluation, but external users may choose to recompute or replace these statistics as needed.

Evaluation Protocol Discoverse-L serves as a taskaligned benchmarking suite supporting interactive evaluation and complements existing long-horizon benchmarks by providing a lightweight, reproducible auxiliary reference. The released assets include demonstration rollouts and normalization files. Evaluation commonly resets the simulator with held-out seeds and runs interactive episodes until termination, reporting Success Rate, Hallucination Rate, and Sample Efficiency as defined in Section 5.1.

# 13. Hyperparameter Sensitivity Analysis

This section provides detailed sensitivity analyses for key hyperparameters. We evaluate robustness to CLIP threshold $\theta$ (used for stage gating and HR computation) and intrinsic reward weight $\rho$ (balancing extrinsic and intrinsic signals). Results confirm that our choices of $\theta = 0 . 7$ and $\rho = 0 . 6$ are well-justified and robust to variations.

# 13.1. CLIP Threshold θ Sensitivity

![](images/e3840c1119612e26b889ac5663e00207aff18077a571db13fdc265638cac30a6.jpg)

![](images/e12fb46efd0de2904599becc5fd38396ed4599033207885ce5b9511187e8ecc7.jpg)  
Figure 9. CLIP threshold sensitivity for EvoVLA (full model) averaged across three Discoverse-L tasks and three seeds. (a) Success Rate remains stable within $\pm 0 . 5$ points around $\theta \ = \ 0 . 7$ $( 6 8 . 7 \mathrm { - } 6 9 . 2 \% )$ . (b) Hallucination Rate decreases monotonically with higher thresholds; $\theta = 0 . 7$ balances false-positive and falsenegative rates while maintaining peak success. Results demonstrate that our choice is robust to threshold variations within [0.65, 0.75].

CLIP threshold $\theta$ sensitivity: Success Rate peaks at $\theta = 0 . 7$ with $6 9 . 2 \%$ and remains stable $( \pm 0 . 5$ points) within [0.65, 0.75]. Setting $\theta$ too low (0.5–0.6) causes premature stage transitions and increases hallucination to $1 9 \mathrm { - } 2 8 \%$ . Conversely, $\theta > 0 . 8$ overly restricts progression, reducing success to $6 3 \mathrm { - } 6 6 \%$ . Our choice balances false-positive and false-negative rates.

# 13.2. Intrinsic Weight $\rho$ Sensitivity

![](images/c43746ad82afba2f54015d06dacd76fd9fc55fa7414e9f802ae7e7d8cf703bfa.jpg)

![](images/cc37358e6c8b3918e91cbd8e07520fb30a0f43de923d2bccbfba19383f335308.jpg)  
Figure 10. Intrinsic weight $\rho$ sensitivity. Performance with $\rho \in \{ 0 . 3 , 0 . 6 , 0 . 9 \}$ (averaged across 3 tasks, 3 seeds). (a) Success Rate: stable within $\pm 2 . 5 \%$ , peaking at ρ = 0.6 (69.2%). (b) Training efficiency: $\rho = 0 . 6$ achieves fastest convergence to $50 \%$ success. Results confirm robustness to intrinsic weight selection.

Intrinsic weight $\rho$ sensitivity: Testing $\rho \in \{ 0 . 3 , 0 . 6 , 0 . 9 \}$ , Success Rate remains stable with $\rho = 0 . 3$ yielding $6 7 . 1 \%$ , $\rho ~ = ~ 0 . 6$ achieving peak $( 6 9 . 2 \% )$ , and $\rho ~ = ~ 0 . 9$ reaching $6 6 . 8 \%$ . Training efficiency shows $\rho ~ = ~ 0 . 6$ converges to $50 \%$ success in $6 . 0 \times 1 0 ^ { 5 }$ steps, while $\rho = 0 . 3$ and $\rho = 0 . 9$ require $7 . 2 \times 1 0 ^ { 5 }$ and $7 . 5 \times 1 0 ^ { 5 }$ steps respectively. The variation within $\pm 2 . 4$ percentage points confirms $\rho = 0 . 6$ provides both optimal performance and fastest convergence.

# 14. Complete Hyperparameter Table

Table 4. Key hyperparameters for DISCOVERSE simulation experiments.   

<table><tr><td>Component</td><td>Parameter</td><td>Value</td></tr><tr><td>SAR</td><td>Temperature τ</td><td>0.05</td></tr><tr><td>SAR</td><td>Smoothing coeff. α</td><td>0.05</td></tr><tr><td>Stage gating</td><td>Window size m</td><td>8</td></tr><tr><td>Stage gating</td><td>Threshold θκ</td><td>0.15</td></tr><tr><td>HR evaluation</td><td>CLIP threshold θ</td><td>0.7</td></tr><tr><td>POE world model</td><td>Learning rate</td><td>1e-3</td></tr><tr><td>POE world model</td><td>Hidden width</td><td>256</td></tr><tr><td>Memory</td><td>GRU hidden width</td><td>128</td></tr><tr><td>Memory</td><td>Window length L</td><td>16</td></tr><tr><td>PPO</td><td>Discount γ</td><td>0.995</td></tr><tr><td>PPO</td><td>GAE λ</td><td>0.95</td></tr><tr><td>PPO</td><td>Learning rate</td><td>3e-4</td></tr><tr><td>Training</td><td>Forward loss weight λF</td><td>1.0</td></tr><tr><td>Training</td><td>Inverse loss weight λI</td><td>0.1</td></tr><tr><td>Training</td><td>Entropy weight λent</td><td>0.01</td></tr><tr><td>Training</td><td>Intrinsic weight ρ</td><td>0.6</td></tr><tr><td>Training</td><td>Total timesteps</td><td>2M</td></tr></table>

Other PPO settings follow standard, widely-used defaults (batch size 256, clip ratio 0.2, entropy coefficient 0.01, max grad norm 0.5) and remain fixed across experiments.

# 15. Video-driven Stage Discovery: Real Example

We provide a real example of the Gemini prompt pipeline [13] that automatically discovers stage semantics and supplies SAR triplets from multi-video demonstrations.

Task and Multi-Video Input For the DISCOVERSE block-bridge placement task (“Build block bridge: place two green bars to form bridge structure, then fill with purple blocks”), we collect ten simulator demonstrations covering diverse initial end-effector poses, object placements, and camera viewpoints. The demonstrations are submitted in a single API call with structured prompts.

System/Developer role (goal setting). We define the agent as an “embodied manipulation stage decomposition expert” tasked with cross-video alignment to automatically determine a reasonable number of stages. The system instructions emphasize: (i) outputs must be structured, machine-readable, and reproducible; (ii) stage semantics must rely on executable spatial/contact/relative-pose predicates; (iii) avoid fragile features such as colors or textures.

User role (task requirements and constraints). The user prompt provides: (i) task description, (ii) notification that the videos belong to the same task but differ in initial states, (iii) instruction to determine minimal-sufficient stages via cross-video alignment (preferring merging over per-video overfitting), (iv) requirement to output unified stage IDs shared across all videos, (v) specification that each stage’s semantics field must contain mutually exclusive positive, hard negative, and negative predicates expressed as quantifiable spatial/contact constraints (e.g., “end-effector alignment error $< \mathrm { ~ 1 ~ c m / 5 ^ { \circ , * } }$ , “lift height $> 2 \mathrm { c m } ^ { \prime \prime }$ ), and (vi) instruction to self-check for conflicts, temporal gaps, or appearance-based descriptions before finalizing the JSON output.

Complete prompt template (our implementation). Building on the above design principles, our full prompt template instructs the model to: (1) perform cross-video adaptive stage segmentation with minimal-sufficient principle (preferring merging over per-video overfitting); (2) generate unified stage IDs shared across all videos; (3) provide mutually exclusive triplets (positive, hard-negative, negative) using geometric/contact predicates while avoiding color/texture cues; (4) self-check for conflicts, temporal gaps, or fragile appearance-based descriptions before outputting the final JSON. Each stage’s semantics field contains: positive (must-satisfy spatial/contact/relativepose predicates, quantifiable such as “gripper closed and contacts target; object center height $> ~ 2 ~ \mathrm { c m }$ ; alignment error $< \mathrm { ~ 1 ~ c m / 5 ^ { \circ } ~ } ^ { , }$ ), hard negative (near-miss cases such as “gripping but not contacting target”, “grasping nontarget”, “insufficient lift height”, “pose alignment $< 5 ^ { \circ \cdots }$ ),

and negative (clearly failed conditions such as far from objects, misaligned, no contact, reverse motion).

Output Schema. The model responds with a JSON dictionary containing task metadata and a triplets field (also called semantics in the workflow description) storing stage semantics. Each stage entry provides positive, hard negative, and negative predicates that we directly reuse for SAR. Below is an excerpt showing 13 representative stages selected from the complete 74-stage output (stages 2–3, 5–7, 10–14, 16–26, 28–37, 39–41, 43–47, 49– 58, 60–62, 64–72 omitted for brevity):

```json
{
    "task_name": "block_bridge_place",
    "task_description": "Build bridge: place two bars,
    fill with blocks",
    "num_stages": 74,
    "triplets": {
        "stage_0": {
            "positive": "Gripper above bridge structure;
            fingers open; ready to operate",
            "negative": "Gripper far from all structures;
            fingers closed; static",
            "hard_negative": "Gripper beside bridge structure;
            fingers open; ready"
       },
    "stage_1": {
        "positive": "Gripper above target bar;
        fingers open; ready to approach",
        "negative": "Gripper far from all objects;
        fingers closed at init",
        "hard_negative": "Gripper beside target bar;
        fingers open; ready to approach"
   },
    "stage_4": {
        "positive": "Gripper contacts target bar;
        fingers closed; stable grasp formed",
        "negative": "Gripper far from all objects;
        fingers fully open at init",
        "hard_negative": "Gripper near target bar but no contact;
        fingers open; ready"
   },
    "stage_8": {
        "positive": "Gripper lowers bar beside bridge base;
        bar contacts surface",
        "negative": "Gripper far from all objects;
        fingers closed at init",
        "hard_negative": "Gripper lowers bar above bridge
        not beside base"
   },
    "stage_9": {
        "positive": "Gripper releases block; no contact;
        block rests on bar below",
        "negative": "Gripper far from all objects;
        closed at init",
        "hard_negative": "Gripper near block but incomplete
        release; partial contact"
   },
    "stage_15": {
        "positive": "Gripper contacts target bar;
        fingers closed; applying grasp force",
        "negative": "Gripper far from all objects;
        fingers open at init",
        "hard_negative": "Gripper contacts non-target bar;
        fingers closed; grasp force"
   },
    "stage_27": {
        "positive": "Cube placed on top of bar;
        bottom surface contacts bar top",
        "negative": "Gripper far from all objects;
        fingers closed at init",
        "hard_negative": "Cube placed beside bar;
        side contacts bar not on top"
   },
    "stage_38": { 
```

```jsonl
"positive": "Gripper releases block; no contact; block rests on bar below", "negative": "Gripper far from all objects; fingers closed at init", "hard_negative": "Gripper contacts block but incomplete release; partial support" }, "stage_42": { "positive": "Gripper contacts target cube; fingers closed; stable grasp formed", "negative": "Gripper far from all objects; fingers open at init", "hard_negative": "Gripper contacts non-target cube; fingers closed; grasp formed" }, "stage_48": { "positive": "Cube placed on bar top; bottom fully contacts bar top surface", "negative": "Gripper far from all objects; static at init", "hard_negative": "Cube placed beside bar; side contacts bar side surface" }, "stage_59": { "positive": "Gripper separates from block; fingers open; block stable on bridge", "negative": "Gripper far from all objects; fingers closed at init", "hard_negative": "Gripper contacts block but incomplete separation; partial closure" }, "stage_63": { "positive": "Gripper contacts target cube; fingers closed; applying grasp force", "negative": "Gripper far from all objects; fingers fully open at init", "hard_negative": "Gripper contacts non-target cube; fingers closed; grasp force" }, "stage_73": { "positive": "Gripper contacts target cube; fingers closed; grasp force applied", "negative": "Gripper far from all objects; fingers open at init", "hard_negative": "Gripper contacts target cube; fingers open; no force applied" } } 
```

Key Characteristics Gemini API prompts [13] adaptively determined 74 stages by aligning repeated manipulation patterns across the ten videos (the excerpt above shows 13 representative stages spanning the full task trajectory; the complete 74-stage dictionary is used in training). Notably, the DISCOVERSE simulator provides 79 groundtruth stages for this task. We compute agreement by matching discovered stages to ground-truth boundaries via temporal intersection-over-union (IoU) on the demonstration set, yielding 74 of 79 stages successfully discovered $( 9 3 . 7 \%$ recall). The remaining five simulator annotations correspond to micro-adjustments (e.g., infinitesimal pre-grasp recentering) that Gemini merges with adjacent stages, demonstrating robust semantic compression. Predicates emphasize geometric/contact cues (e.g., “fingers closed; stable grasp”, “contacts surface”) and avoid appearance features. Hard negatives capture near-miss states such as “gripper near target but no contact” or “cube beside bar instead of on top”, enabling SAR to discriminate shortcuts from genuine com-

pletion.

Validation and QC Before accepting a dictionary, we run automatic heuristics that check mutual exclusivity, enforce minimum stage length (6 frames), and reject outputs with unresolved TODO markers or appearance-only predicates. We then sample $10 \%$ of the stages for manual inspection; if any predicate is ambiguous we re-prompt Gemini with additional constraints. Appendix 16 lists the automated filters and remediation prompts so the workflow remains reproducible without hidden human annotations.

# 16. Additional Implementation Details

Memory latent extraction. For Long-Horizon Memory, we extract a compact latent representation $\boldsymbol { x } _ { t } \in \mathbb { R } ^ { d }$ (with $d =$ 128 matching the GRU hidden width) from the OpenVLA-OFT backbone’s LLM hidden states. Specifically, we apply mean pooling over the last layer’s hidden states corresponding to visual tokens, then project to dimension $d$ via a learned linear layer. This latent serves as the current context for memory attention (Eq. 10) and is stored in the memory buffer $\mathcal { M }$ after gated fusion. The pooling and projection layers are trained end-to-end with the policy via PPO.

Stage gating mechanism. For stage transitions, we maintain a slidingward values $m = 8$ of recent stage re-en the window av-$\{ r _ { t - m + 1 } ^ { \mathrm { s t a g e } } , \ldots , r _ { t } ^ { \mathrm { s t a g e } } \}$ , rstaget }. Wh erage exceeds a threshold $\theta _ { \kappa }$ for the current stage $\kappa$ , we advance to the next stage: $\kappa  \operatorname* { m i n } ( \kappa + 1 , K )$ . Since $r _ { t } ^ { \mathrm { s t a g e } } = \bar { u } _ { \kappa _ { t } } ( t ) - \bar { u } _ { \kappa _ { t } } ( t - 1 )$ rt measures progress in smoothed scores (Eq. 4), using stage reward for gating captures sustained task advancement more directly than raw scores. This prevents premature transitions while allowing robust progression.

Hard negative generation. Counterfactual hard negatives h $T _ { k } ^ { h - }$ are generated by the same Gemini 2.5 Pro [13] prompting pipeline with few-shot demonstrations. We provide: (1) task description, (2) stage $k$ positive description $T _ { k } ^ { + }$ , (3) instruction to create a near-miss description differing by one predicate. Crucially, the LLM generates descriptions using spatial and contact predicates (e.g., “gripper near cup but not touching” vs. “gripper touching cup”, or “grasping non-target object” vs. “grasping target cup”) that avoid reliance on visual appearance such as colors or textures. This ensures the text-based hard negatives remain robust when the VLM performs image-text similarity evaluation.

Automated validation filters. The quality control workflow referenced in Appendix 15 includes: (1) Mutual exclusivity check: verify that positive and negative predicates cannot simultaneously hold (no logical overlap); (2) Minimum stage length: reject dictionaries with stages shorter than 6 frames to avoid over-segmentation; (3) Appearance filter: flag predicates containing color/texture key-

words (e.g., “red”, “shiny”) and require spatial rephrasing; (4) Completeness check: ensure all $K$ stages have valid triplets without placeholder text. Failed outputs trigger automatic re-prompting with clarified instructions (e.g., “Replace color mentions with spatial relations”). After automated filtering, we sample $10 \%$ of stages for spot-checking by annotators; ambiguous predicates are manually refined and fed back to improve few-shot examples in subsequent tasks.

# 17. Network Architecture Details

The main paper describes the overall architecture (OpenVLA-OFT backbone with SAR and POE modules). Here we provide dimensionality and layer specifications for reproducibility.

Visual Encoding Pipeline The policy processes two camera views: SigLIP-SO400M/14 (third-person, $3 8 4 \times 3 8 4$ input) outputs 1152-D features; DINOv2-Large (wrist, $2 2 4 \times 2 2 4$ input) outputs 1024-D features. A learned projection layer maps the concatenated 2176-D visual embedding to the Llama-2-7B token space (4096-D). Language instructions are tokenized and prepended to the visual sequence. FiLM conditioning layers inject language embeddings into visual features before the language model processes the combined sequence. During fine-tuning, SigLIP and DINOv2 encoders remain frozen; only the Llama-2- 7B language model, projection layer, and action decoder are trainable, reducing computational cost while leveraging pretrained visual representations.

Action Decoding The action head predicts $C$ -step chunks in parallel (we use $C \ = \ 8$ ) via $C$ separate linear layers ${ 4 0 9 6 \mathrm { - } \mathrm { } D }  \mathrm { } 7 \mathrm { - } 1$ D each), stacked to produce a $( C , 7 )$ output tensor. Each row corresponds to one future timestep’s 7-D action (6-DoF end-effector twist deltas $+ ~ 1 { - } \mathrm { D }$ gripper command). During inference we execute only the first action and recompute at the next step (receding-horizon control). Following OpenVLA-OFT [24], the action decoder uses L1 regression loss for training, which provides stable gradients for continuous action prediction. The chunked action prediction enables temporal coherence while maintaining computational efficiency compared to autoregressive decoding.

World Model Specifics Forward model $f _ { \phi }$ : input 13-D (6-D state + 7-D action) $ 2 5 6  – \mathrm { D }$ hidden (ReLU + LayerNorm) $ 2 5 6  – \mathrm { D }$ hidden $ 6 \mathrm { - D }$ output. Inverse model $g _ { \psi }$ : input 12-D (two consecutive 6-D states) same architecture $ 7$ -D action output. Both use separate Adam optimizers (lr=1e-3, no weight decay) and update every PPO epoch with MSE loss on transitions sampled from the replay buffer. The lightweight architecture (256-D hidden)

ensures fast online updates during policy training without introducing significant computational overhead.

Model Size and Efficiency The full EvoVLA model inherits the 7B-parameter Llama-2-7B backbone plus pretrained vision encoders (SigLIP-SO400M and DINOv2- Large). During fine-tuning, the vision encoders remain frozen while the Llama-2 language model, projection layers, and task-specific modules (POE world models, Long-Horizon Memory) are trainable, reducing effective trainable parameters to approximately 7B and enabling efficient finetuning on standard GPU hardware.

# 18. Computational Cost

Training EvoVLA on Discoverse-L requires approximately 24 hours wall-clock time per seed on 4×NVIDIA H20 GPUs (96 GB VRAM each) for 2M environment steps, using 8 parallel DISCOVERSE simulators. Across 3 random seeds, total training time is approximately 72 hours (3 days). Memory footprint peaks at approximately 80 GB per GPU during policy updates.
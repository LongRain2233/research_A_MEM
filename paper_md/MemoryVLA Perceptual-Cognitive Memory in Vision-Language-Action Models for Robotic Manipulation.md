![](images/3ed7f518593157e3908c15b0d294b9d34d4b4c000c1fdd060f446be21153aa77.jpg)

# ORY IN VISION-LANGUAGE-ACTION MODELS FOR ROBOTIC MANIPULATION

Hao Shi1† Bin Xie2 Yingfei Liu2 Lin Sun4† Fengrong Liu5† Tiancai Wang2 Erjin Zhou2 Haoqiang Fan2 Xiangyu Zhang3,6 Gao Huang1B

1Department of Automation, BNRist, Tsinghua University 2Dexmal

3MEGVII Technology 4Tianjin University 5Harbin Institute of Technology 6StepFun shi-h23@mails.tsinghua.edu.cn gaohuang@tsinghua.edu.cn

† Work done during interning at Dexmal. B Corresponding author.

# ABSTRACT

Temporal context is essential for robotic manipulation because such tasks are inherently non-Markovian, yet mainstream VLA models typically overlook it and struggle with long-horizon, temporally dependent tasks. Cognitive science suggests that humans rely on working memory to buffer short-lived representations for immediate control, while the hippocampal system preserves verbatim episodic details and semantic gist of past experience for long-term memory. Inspired by these mechanisms, we propose MemoryVLA, a Cognition-Memory-Action framework for long-horizon robotic manipulation. A pretrained VLM encodes the observation into perceptual and cognitive tokens that form working memory, while a Perceptual-Cognitive Memory Bank stores low-level details and highlevel semantics consolidated from it. Working memory retrieves decision-relevant entries from the bank, adaptively fuses them with current tokens, and updates the bank by merging redundancies. Using these tokens, a memory-conditioned diffusion action expert yields temporally aware action sequences. We evaluate MemoryVLA on $1 5 0 +$ simulation and real-world tasks across three robots. On SimplerEnv-Bridge, Fractal, LIBERO-5 suites and Mikasa-Robo, it achieves $7 1 . 9 \%$ , $7 2 . 7 \%$ , $9 6 . 5 \%$ , and $4 1 . 2 \%$ success rates, respectively, all outperforming state-of-the-art baselines CogACT and $\pi _ { 0 }$ , with a notable $+ 1 4 . 6$ gain on Bridge and $+ 1 1 . 8$ gain on Mikasa-Robo. On 12 real-world tasks spanning general skills and long-horizon temporal dependencies, MemoryVLA achieves $8 4 . 0 \%$ success score, with long-horizon tasks showing a $+ 2 6$ improvement over state-of-the-art baseline. Project Page: https://shihao1895.github.io/MemoryVLA

# 1 INTRODUCTION

Vision-Language-Action (VLA) models (Brohan et al., 2023; Kim et al., 2024; Black et al., 2024; Li et al., 2024a; Sun et al., 2025; Xie et al., 2025), powered by large-scale cross-embodiment robotic datasets (O’Neill et al., 2024; Brohan et al., 2022; Khazatsky et al., 2024; Bu et al., 2025a) and pretrained Vision-Language Models (VLMs) (Karamcheti et al., 2024; Liu et al., 2023b; Bai et al., 2023a), have achieved remarkable progress in robotic manipulation. However, mainstream VLA models such as OpenVLA (Kim et al., 2024) and $\pi _ { 0 }$ (Black et al., 2024) rely solely on the current observation, thereby overlooking temporal dependencies and performing poorly on long-horizon temporal manipulation tasks. As shown in Fig. 1 (a), Push Buttons tasks exhibit almost no visual difference before and after pushing, making it difficult to determine whether the action has already been completed. This highlights the non-Markovian nature of manipulation, where earlier actions influence later decisions, calling for temporal modeling. A naive strategy is to concatenate consecutive frames as input to the VLM. However, it faces two critical limitations: (1) The quadratic complexity of self-attention severely limits the usable temporal context length; (2) Sequential frame inputs are misaligned with the model’s single-frame robotic pretraining distribution.

![](images/014755c3cec4beb3cf93e0d91f1da05ebd1ee6723a79e2c68a7aabf899371548.jpg)  
(a) Example of Temporal Confusion   
Figure 1: (a) In Push Buttons tasks, pre- and post-push states look nearly identical, calling for temporal modeling. (b) Humans handle manipulation tasks via a dual-memory system: working memory (neural activity) supports short-term control, while episodic memory (hippocampus) preserves long-term experience. (c) Inspired by this, MemoryVLA introduces a Perceptual–Cognitive Memory Bank that consolidates low-level perceptual details and high-level cognitive semantics for temporally aware decision making. (d) MemoryVLA outperforms state-of-the-art baselines.

Research in cognitive science (Baddeley & Hitch, 1974; Tulving et al., 1972; Reyna & Brainerd, 1995) demonstrates that humans handle manipulation tasks through a dual-memory system (Fig. 1 (b)). The brain encodes multi-modal sensory inputs into both perceptual and cognitive representations. These representations are buffered in working memory via transient neural activity, providing short-term retention for immediate decision-making. Concurrently, episodic memory, the long-term memory system supported by hippocampus, encodes past experiences with temporal index in two forms: verbatim representations preserving precise details and gist representations capturing abstract semantics. During execution, working memory retrieves decision-relevant contexts from episodic memory and integrates them with current representations to guide actions through cerebellar control, while simultaneously consolidating new experiences into episodic memory.

Drawing on cognitive science insights, we propose MemoryVLA (Fig. 1 (c)), a Cognition-Memory-Action framework for robotic manipulation that explicitly models temporal dependencies through a Perceptual–Cognitive Memory Bank (PCMB). First, a vision encoder extracts perceptual tokens from observation, while a large language model (LLM) processes them together with the language instruction, leveraging commonsense priors to produce cognitive tokens. Perceptual and cognitive tokens jointly form the working memory. Second, the PCMB stores both low-level perceptual details and high-level cognitive semantics over long horizons. During retrieval, working memory buffers current tokens and queries the PCMB with temporal positional encodings to fetch decision-relevant historical contexts, which are adaptively fused with current tokens via a gating mechanism while simultaneously updating the PCMB. When capacity is reached, temporally adjacent and semantically similar entries are consolidated to preserve essential information compactly. Finally, a memoryconditioned diffusion action expert is conditioned on cognitive tokens, with perceptual tokens enriching them with fine-grained details, to produce temporally aware robotic action sequences.

We conduct extensive evaluations of MemoryVLA across 3 robots and $1 5 0 +$ tasks with $5 0 0 +$ variations in simulation and real world. For SimplerEnv (Li et al., 2024b), MemoryVLA achieves $7 1 . 9 \%$ and $7 2 . 7 \%$ success rates on Bridge and Fractal suites, surpassing CogACT by 14.6 and 4.6 points, further outperforming $\pi _ { 0 }$ . For LIBERO (Liu et al., 2023a), it achieves $9 6 . 5 \%$ success rate across 5

suites (Spatial, Object, Goal, Long-10, and Long-90), exceeding both CogACT and $\pi _ { 0 }$ . For Mikasa-Robo (Cherepanov et al., 2025), it achieves $4 1 . 2 \%$ success rate, outperforming $\pi _ { 0 }$ by 11.8 points. For real-world evaluations, we introduce 12 tasks across Franka and WidowX robots, spanning 6 general tasks and 6 long-horizon temporal tasks. MemoryVLA achieves $8 5 \%$ and $83 \%$ scores on general and temporal tasks, outperforming CogACT by 9 and 26 points, and substantially surpassing $\pi _ { 0 }$ . Moreover, MemoryVLA exhibits strong robustness and generalization under out-of-distribution conditions involving varied backgrounds, distractors, objects, containers, lighting and occlusion.

Our contributions are summarized as follows:

• Inspired by human memory systems from cognitive science, we propose MemoryVLA, a Cognition-Memory-Action framework that leverages VLM commonsense priors, a perceptualcognitive memory mechanism, and a diffusion action expert to capture long-horizon temporal dependencies for robotic manipulation.   
• We design a Perceptual–Cognitive Memory Bank with working memory that enables memory retrieval of decision-relevant contexts across high-level cognition and low-level perception, memory fusion that adaptively integrates them with current representations, and memory consolidation that merges temporally adjacent, semantically similar entries.   
• MemoryVLA achieves state-of-the-art performance on SimplerEnv, LIBERO, Mikasa-Robo, and real-world. It also demonstrates strong robustness and generalization. On challenging longhorizon real-world tasks, it outperforms CogACT and $\pi _ { 0 }$ by significant margins, underscoring the importance of temporal memory modeling.

# 2 RELATED WORKS

Vision-Language-Action Models Driven by advances in visual foundation models (Radford et al., 2021; Caron et al., 2021; Liu et al., 2024a; Zheng et al., 2024a; 2025a; Zhang et al., 2025b; Wu et al., 2025; Wang et al., 2025), robot imitation learning has progressed rapidly yet remains confined to small, task-specific policies with limited generalization (Shridhar et al., 2023; Zhao et al., 2023; Chi et al., 2023; Goyal et al., 2023; Shi et al., 2025). To overcome these, the success of VLMs (Achiam et al., 2023; Touvron et al., 2023; Liu et al., 2023b; Bai et al., 2023b; Guo et al., 2025) and largescale robot datasets (e.g., OXE (O’Neill et al., 2024), Agibot (Bu et al., 2025a)) spawned the visionlanguage-action (VLA) paradigm (Kim et al., 2024; Black et al., 2024; Yue et al., 2024; Zhong et al., 2025; Gao et al., 2025; Yang et al., 2025b). RT-2 (Zitkovich et al., 2023) and OpenVLA (Kim et al., 2024) tokenize continuous actions into discrete tokens and use VLMs for autoregressive prediction as if generating language. In contrast, $\pi _ { 0 }$ (Black et al., 2024), CogACT (Li et al., 2024a), DexVLA (Wen et al., 2025) and HybridVLA (Liu et al., 2025c) adopt diffusion-based policies (Chi et al., 2023; Liu et al., 2024b) as action heads, leveraging iterative denoising to sample continuous control trajectories that capture diverse multimodal behaviors. However, none of these methods explicitly model temporal dependencies. Robotic manipulation is inherently non-Markovian, and neglecting history leads to failures on long-horizon temporal tasks.

Temporal Modeling in Robotics Temporal modeling has been extensively studied in computer vision and autonomous driving (Wang et al., 2023; Liu et al., 2023c; Feng et al., 2023; Zhou et al., 2024), yet it has not been fully explored in robotic manipulation. Octo (Mees et al., 2024), RoboVLMs (Liu et al., 2025b), and Interleave-VLA (Fan et al., 2025) adapt the VLM paradigm to model robotic video data in an interleaved image-text format. While conceptually elegant, this format is complex to implement and computationally expensive, hindering its widespread application. RoboFlamingo (Li et al., 2023) compresses vision-language representation into a latent token and propagate it via LSTM (Hochreiter & Schmidhuber, 1997). The latent representation is obtained in a relatively coarse manner and the fine-grained perceptual history is largely discarded. TraceVLA (Zheng et al., 2024b) takes a different route, painting historical states as trajectories on the current frame, yet discards rich semantic details. UniVLA (Bu et al., 2025b) incorporates past actions into input prompts, making an initial attempt at temporal modeling. However, it merely serves as a Chain-of-Thought (Wei et al., 2022) process without effectively utilizing historical information. In contrast, we model both high-level cognitive semantics and fine-grained perceptual details within a memory framework, enabling effective temporal modeling for long-horizon manipulation.

![](images/2cebe26f11847b96045bc37121af65f9a477178a5e441d37011dcaf224cd0d8f.jpg)  
Figure 2: Overall architecture of MemoryVLA. RGB observation and language instruction are encoded by a 7B VLM into perceptual and cognitive tokens, forming short-term working memory. The working memory queries a perceptual-cognitive memory bank (PCMB) to retrieve relevant historical context, including high-level semantics and low-level visual details, adaptively fuses it with current tokens, and consolidates the PCMB by merging the most similar neighbors. The memoryaugmented tokens then condition a diffusion transformer to predict a sequence of future actions.

# 3 METHOD

# 3.1 OVERVIEW OF MEMORYVLA

Problem Formulation We formulate robotic manipulation in VLA models as a sequential trol actions for real world interaction. Given the current RGB image decision-making process, where visual observations and language instructions are mapped to con- $I \in \mathbb { R } ^ { H \times W \times 3 }$ and a language instruction $L$ , a parameterized policy $\pi$ outputs a sequence of future actions

$$
\mathcal {A} = \left(a _ {1}, \dots , a _ {T}\right) = \pi (I, L), \tag {1}
$$

where each action $a _ { t } = [ \Delta x , \Delta y , \Delta z , \Delta \theta _ { x } , \Delta \theta _ { y } , \Delta \theta _ { z } , g ] ^ { \top }$ consists of relative translation, relative rotation (Euler angles), and a binary gripper state $g \in \{ 0 , 1 \}$ .

Overview MemoryVLA is an end-to-end framework for robotic manipulation, as shown in Fig. 2. The current RGB observation and language instruction are first encoded by a VLM into perceptual and cognitive tokens, forming a working memory, analogous to neural activity in the visual and prefrontal cortex associated with short-term memory. To complement this short-term store, we introduce the Perceptual–Cognitive Memory Bank (PCMB), inspired by the hippocampus, which maintains long-term high-level semantics and fine-grained perceptual details. Working-memory embeddings query the PCMB to retrieve decision-relevant history, adaptively fuse it with current representations via gating, and consolidate the memory by merging temporally adjacent and semantically similar entries when capacity is reached. The resulting representations are then fed into a memory-conditioned diffusion action expert to generate a sequence of $N$ future 7-DoF actions.

# 3.2 VISION-LANGUAGE COGNITION MODULE

We build upon a 7B–parameter Prismatic VLM (Karamcheti et al., 2024), which is further pretrained on the large-scale cross-embodiment real robot dataset Open-X Embodiment (O’Neill et al., 2024). For visual encoding, we adopt parallel DINOv2 (Oquab et al., 2023) and SigLIP (Zhai et al., 2023) backbones on the current third-person RGB image $I$ , concatenating their features into raw visual tokens. A perceptual compression module, implemented via a SE-bottleneck (Hu et al., 2018), then compresses these tokens into a compact set of perceptual tokens $p \in \mathbb { R } ^ { N _ { p } \times d _ { p } }$ with $N _ { p } = 2 5 6$ . In parallel, the raw visual tokens are projected via a linear layer into the language embedding space and concatenated with the tokenized instruction before being fed into the LLaMA-7B (Touvron et al., 2023). The output at the end-of-sentence (EOS) position is taken as the cognitive token $c \in \mathbb { R } ^ { 1 \times d _ { c } }$ , representing high-level cognitive semantics in compact form. Finally, the perceptual tokens $p$ and cognitive token c are combined to form the short-term working memory for downstream modules.

![](images/d5129ffa61afde80b96b2c7079db91c88e8f78363e22f28f933608402ed93984.jpg)  
(a) Memory Retrieval

![](images/5b04cc5a6751ddfe18be901122eebdf9857bab2438b60030cccfd78922554a66.jpg)  
(b) Memory Gate Fusion

![](images/c22c0d3264daf4ca3f582198cd16b568613e7cae27e4dc09e122a797d03bfaca.jpg)  
(c) Memory Consolidation   
Figure 3: Details of memory module. (a) Retrieval: current perceptual and cognitive tokens query the PCMB via cross-attention with timestep positional encoding to fetch relevant historical features. (b) Gate fusion: current and retrieved tokens are adaptively fused via a gate mechanism. (c) Consolidation: the fused tokens are updated into PCMB. When PCMB reaches its capacity, we compute similarities between adjacent entries and merge the most similar pair to maintain compactness.

# 3.3 PERCEPTUAL-COGNITIVE MEMORY MODULE

The Vision–Language Cognition Module yields a working memory

$$
M _ {\mathrm {w k}} = \left\{p \in \mathbb {R} ^ {N _ {p} \times d _ {p}}, c \in \mathbb {R} ^ {1 \times d _ {c}} \right\}, \tag {2}
$$

where $p$ and $c$ represent the current perceptual tokens and cognition token, respectively. However, this working memory only reflects the present timestep and lacks temporal dependencies.

To address this, inspired by the hippocampus in human memory systems, we introduce the Perceptual–Cognitive Memory Bank (PCMB):

$$
M _ {\mathrm {p c m b}} = \left\{m ^ {x} \mid x \in \{\text {p e r}, \operatorname {c o g} \} \right\}, \tag {3}
$$

$$
m ^ {x} = \left\{m _ {i} ^ {x} \in \mathbb {R} ^ {N _ {x} \times d _ {x}} \right\} _ {i = 1} ^ {L}, \quad x \in \{\text {p e r , c o g} \}, \tag {4}
$$

where each perceptual entry $m _ { i } ^ { p }$ stores fine-grained visual details and each cognitive entry $m _ { i } ^ { c }$ encodes a high-level semantic summary. The bank maintains up to $L$ entries per stream.

Memory Retrieval At each timestep, the working memory $M _ { \mathrm { w k } }$ , comprising current perceptual tokens $\dot { p } \in \mathbb { R } ^ { N _ { p } \times d _ { p } }$ and cognition token $c \in \mathbb { R } ^ { 1 \times \widetilde { d _ { c } } }$ , acts as a dual query to retrieve historical information required for the current decision from the Perceptual-Cognitive Memory Bank $M _ { \mathrm { p c m b } }$ as illustrated in Fig. 3 (a). Each memory entry is associated with its episode timestep via a sinusoidal embedding $\operatorname { T E } ( \cdot )$ , which is added as positional encoding. We then stack all perceptual memories into a tensor $\in \dot { \mathbb { R } } ^ { L N _ { p } \times d _ { p } }$ and cognitive memories into a tensor $\in \mathbb { R } ^ { L \times d _ { c } }$ . Scaled dot-product attention between the current tokens and these memory tensors produces raw outputs for both streams:

$$
K ^ {x} = \left[ m _ {1} ^ {x} + \operatorname {T E} \left(t _ {1}\right); \dots ; m _ {L} ^ {x} + \operatorname {T E} \left(t _ {L}\right) \right], \quad V ^ {x} = \left[ m _ {1} ^ {x}; \dots ; m _ {L} ^ {x} \right], \tag {5}
$$

$$
\hat {H} ^ {x} = \operatorname {s o f t m a x} \left(\frac {q ^ {x} \left(K ^ {x}\right) ^ {\top}}{\sqrt {d _ {x}}}\right) V ^ {x}, \quad q ^ {x} \in \{p, c \}, x \in \{\text {p e r , c o g} \}. \tag {6}
$$

This attention operation is followed by a feed-forward network to complete one Transformer layer, and applying two such layers yields the final retrieved embeddings $H ^ { p }$ and $H ^ { c }$ .

Memory Gate Fusion As illustrated in Fig. 3 (b), the gate fusion process integrates the retrieved embeddings $H ^ { p }$ and $H ^ { c }$ with the current working memory representations through learned gates. For both the perceptual $( x = p$ ) and cognitive ( $x = c$ ) streams, a gating vector is computed as

$$
g ^ {x} = \sigma (\operatorname {M L P} (\operatorname {c o n c a t} [ x, H ^ {x} ])), \tag {7}
$$

and applied to obtain the memory-augmented representation

$$
\tilde {x} = g ^ {x} \odot H ^ {x} + (1 - g ^ {x}) \odot x. \tag {8}
$$

Here, $\sigma$ denotes the sigmoid activation and $\odot$ denotes element-wise multiplication. The resulting memory-augmented features $\tilde { p }$ and $\tilde { c }$ are then forwarded to the memory consolidation stage.

Memory Consolidation After gate fusion, the memory-augmented representations $\tilde { p }$ and $\tilde { c }$ are passed to the Memory-conditioned Action Expert and simultaneously updated to the PCMB. When the number of stored entries exceeds $L$ , cosine similarities are computed within each stream (perceptual and cognitive) between adjacent entries. The pair with the highest similarity in each stream is selected and merged by averaging their vectors, thereby reducing redundancy.

$$
i _ {x} ^ {*} = \arg \max  _ {i = 1, \dots , L - 1} \cos (\tilde {x} _ {i}, \tilde {x} _ {i + 1}), \quad m _ {i _ {x} ^ {*}} ^ {x} \leftarrow \frac {1}{2} \bigl (\tilde {x} _ {i _ {x} ^ {*}} + \tilde {x} _ {i _ {x} ^ {*} + 1} \bigr), \quad x \in \{\text {p e r , c o g} \}. \tag {9}
$$

This consolidation mechanism (Fig. 3 (c)) mitigates memory bloat by reducing redundancy, while preserving the most salient perceptual details and semantic abstractions, thereby maintaining a compact representation that supports efficient long-term memory.

# 3.4 MEMORY-CONDITIONED ACTION EXPERT

Leveraging the memory-augmented working memory $\{ \tilde { p } , \tilde { c } \}$ , which integrates historical perceptual and cognitive information, the action expert predicts a sequence of future actions $\{ a _ { 1 } , a _ { 2 } , \ldots , a _ { T } \}$ , with $T = 1 6$ . This prediction allows the model to anticipate multi-step trajectories, reduce cumulative error, and provide foresight for long-horizon execution. Since real-world robotic actions lie in a continuous multimodal control space, we adopt a diffusion-based Transformer (DiT) (Peebles & Xie, 2023) implemented with Denoising Diffusion Implicit Models (DDIM) (Song et al., 2020), using 10 denoising steps for efficient yet accurate trajectory generation. This architecture progressively denoises a sequence of noisy action tokens, yielding precise continuous actions.

Specifically, at each denoising step, the noisy action tokens are injected with the sinusoidal encoding of the denoising timestep and concatenated with the cognitive representation c˜. A cognition-attention layer conditions the process with high-level semantic guidance, while a perception-attention layer supplements fine-grained visual details from the perceptual features $\tilde { p }$ . The combined representation is then refined through a feed-forward network to obtain the denoised action at that step. The model is trained with mean squared error (MSE) loss between the predicted and target actions, and the final denoised vectors are passed through an MLP to generate continuous 7-DoF robotic actions.

# 4 EXPERIMENTS

To comprehensively evaluate MemoryVLA, we organize experiments around six core questions: (1) How does MemoryVLA compare with state-of-the-art methods on SimplerEnv benchmark? (Sec. 4.2) (2) How does it perform on LIBERO benchmark? (Sec. 4.3) (3) How does it perform on Mikasa-Robo benchmark? (Sec. 4.4) (4) Can it handle both general manipulation and long-horizon temporal tasks on real robots? (Sec. 4.5) (5) What is the impact of each component? (Sec. 4.6) (6) How robust and generalizable is it under diverse environmental conditions? (Appendix B)

# 4.1 EXPERIMENTAL SETUPS

Simulation and Real-world Benchmarks. Fig. 4 overviews our evaluation across simulation and real-world, covering 3 robots, 6 benchmarks, $1 5 0 +$ tasks with $5 0 0 +$ variations. SimplerEnv (Li et al., 2024b) includes Bridge suite with a WidowX robot and Fractal suite with a Google robot. Fractal provides two settings: Visual Matching (VM) and Visual Aggregation (VA). LIBERO (Liu et al., 2023a) uses a Franka robot and spans five suites (Spatial, Object, Goal, Long-10, and Long-90). Mikasa-Robo (Cherepanov et al., 2025) uses a Franka robot. In real-world, we evaluate General and Long-horizon Temporal suites on Franka and WidowX robots. Task details for each benchmark and additional qualitative results are provided in Appendix K and Appendix M.

Implementation Details We train on 8 NVIDIA A100 GPUs with PyTorch FSDP, using 32 samples per GPU for a global batch of 256 and a learning rate of $2 \times 1 0 ^ { - 5 }$ . The model takes a single third-person RGB frame at $2 2 4 \times 2 2 4$ together with the language instruction and outputs 7-DoF actions. The LLM is 7B, and the diffusion action expert has ${ \sim } 3 0 0 \mathbf { M }$ parameters. At inference we use DDIM (Song et al., 2020) with 10 sampling steps and a classifier-free guidance(CFG) (Ho & Salimans, 2022) guidance scale of 1.5. Additional details are provided in Appendix $\textrm { C }$ and D.

3 Robots, 6 Benchmarks, 150+ Tasks, 500+ Variations

![](images/3d3684b75fc17460ae462c39e7d21c42fc1f5acad4bb4c0c074dc2944a33346f.jpg)  
SimplerEnv-Bridge   
Spoon on Towel

![](images/4dd1aebfdfe46669b0285dcee10bd998f27a9ab67955cd37b336c416a7931710.jpg)  
Carrot on Plate

![](images/75e18e4d71cc7f264780188eec3faa7d977d3faa7654f627c2d4bee03075a3bb.jpg)  
Stack Cube

![](images/5c35ad22cbdb0d201a050a4f612e573e7d8d009cf8877ab7c163aa03837b8f5e.jpg)  
Eggplant in Basket

![](images/3dc6bff9e5793466a27b2a29f495e332f113ae3b4487984dedcf1ef2d82d08ba.jpg)  
SimplerEnv-Fractal   
Coke Can

![](images/dae46abb927d399c28a8f139ed7df74baf95a3bc47bce23842a9b1d6773ed1b0.jpg)  
Move Near

![](images/39fd1579695a944b7d757bbd3cfa8edfc9e612d507035ebf98a9765114895a12.jpg)  
Open/Close Drawer

![](images/be310db78861c409c870d5d9a46ce8f62c0194d77302ee9168da910e8c8ca8ac.jpg)  
Place in Drawer

![](images/5505ce75bd3af628e75c92a43dc4426bec1fad573b225cf33da9f5d5fa772fe2.jpg)  
LIBERO   
Spatial Suite

![](images/884ac5a03bfc52dd948c2a51abe2c61357d9603d9c0574b98bd9e0095948950b.jpg)  
Object Suite

![](images/32ed20570a4a65bd90dec7ec670962bd26afce429bbccfe4e9d929666e2b9c8c.jpg)  
Goal Suite

![](images/cb08bdc00d940777cd880152d9fcc0df13f675637a412520a855c0f56f53a315.jpg)  
Long-100 Suite

![](images/ba7ca28591e64394310c4b7b43ffb1c7a76dbd1c34f0b1fb655dfe77e242d16e.jpg)  
Mikasa-Robo

![](images/6290f003c06c03bbb5a854e5bccc515ad1b561f437cd7f8d98134e7129f1d5ed.jpg)  
Shell Game   
RemColor3

![](images/6b94b7b6e4f30a87ebded839ffef3685d74ccdec16c615877002783e2841917d.jpg)

![](images/6f44385fc5e8c4dcf3e11f458f575e781368f64a753a8b74c81b05a50acf852f.jpg)  
Intercept   
RemColor9

![](images/93c1aafdacfd6c03dc06a8b88156dafd32343d50de70a5bb14a3e74ca9d19d53.jpg)  
Real-world Robots   
Insert Circle

![](images/b9d1e8ecd7865c9d3b9cfdb12b31d43358add710f036f5ef811775b319be6746.jpg)  
Egg in Pan

![](images/23267f486bd36e1393c24685f330721f79ce3951b43324f03e2f40d24e042b15.jpg)  
Egg in Oven

![](images/bfbaaf0043c4a58f68a96d4bde98e29bca01efe7c5b1ab128a84c9013001f3ec.jpg)  
Stack Cups

![](images/215694cb54b287e75db3b5fd5d825b0dd3e4012e3d48c083e0f31c46bcb3ff72.jpg)  
Stack Blocks

![](images/f5201f2512d8225b5c5ad64642628dcbdfc96803223ed4519f6e17306e1f67f0.jpg)  
Pick Diverse Fruits

![](images/759cf0de02bf1d4f8350cb474bc0c92cb953c0dfebabac8feea73412c5026a0a.jpg)  
Seq Push Buttons

![](images/5e4b2e3f9aec184fd02a5b4128e300f778fd516d375e3a4ca9ff66a551996fdf.jpg)  
Change Food

![](images/850257c47cfd83098061bdf634e0985f296fd2f136938bc1b1fcbe8103a85bdc.jpg)  
Guess Where

![](images/00973067976099b63c67b73b987060c0b21f1e6db5770a969d72fafd30e616f5.jpg)  
Clean Table Count

![](images/f6a250b832cff2e80ad4d64c5e6aa03849eaee499b75af5bf513cb47f6a4c100.jpg)  
Pick Place Order

![](images/421fdabc4da22efc51a2c6cdf83d2b50d8a88eb04474af225de09f6aa711e5cc.jpg)  
Clean Rest. Table

![](images/34c78c3bd6791711190e9a3153a89fc1d23dffba54a4b04cd18cfb4f03b95bc5.jpg)  
Robustness & Generalization   
Background

![](images/d99118174a07b7a097e99ac1d1f32148048ba8bcf314a07ed889088db49ddfee.jpg)  
Distractors

![](images/f7247176585d47161109740f3b6677fb9ba6d6da15939b5627fb830e19c5180c.jpg)  
Lighting

![](images/cee674008a522da53967fdadb4f8b502505aa6e9f9d86f6bb3a590ded81e1fbc.jpg)

![](images/1f77ac6f2fd696ab25bed05729b39392989eb53c0f72e0dc75f2415ce5f92c12.jpg)  
Container

![](images/76b27a045bec4073d365aef4b86b131b46dce2e4f6b75f6cccc57f5490d1bbf1.jpg)  
Occlusion   
Figure 4: Experimental setup overview. Top: Four simulation benchmarks, SimpleEnv-Bridge, SimpleEnv-Fractal, LIBERO, and Mikasa-Robo. Bottom: real-world evaluation (General and Longhorizon Temporal), real-world robustness and generalization evaluation. In total, we evaluate 3 robots across 6 benchmarks, spanning over 150 tasks and 500 variations.

# 4.2 SIMULATED EVALUATION ON SIMPLERENV

Training and Evaluation Setup We evaluate on two SimplerEnv suites: Bridge and Fractal. For SimplerEnv-Bridge, we train on Bridge v2 dataset (Walke et al., 2023) for 50k steps with validation every $2 . 5 \mathrm { k }$ steps. Results are reported at the best validation step, and each task is evaluated with 24 trials to compute success rates. For SimplerEnv-Fractal, we train on the RT-1 dataset (Brohan et al., 2022) for 80k steps with validation every 5k steps. Evaluation covers Visual Matching (VM) and Visual Aggregation (VA) settings. VM mirrors the real setup to reduce sim-to-real gap, while VA stress-tests robustness by altering background, lighting, distractors, and table textures. The Fractal testbed includes 336 variants, yielding 2,856 trials in total.

Evaluation Results on SimplerEnv-Bridge As shown in Tab. 1, MemoryVLA achieves an average success rate of $71 . 9 \%$ , a $\mathbf { + 1 4 . 6 }$ point gain over the CogACT-Large baseline, and surpasses recent state-of-the-art VLAs including $\pi _ { 0 }$ (Black et al., 2024). Per task, success rates are $7 5 . 0 \%$ on Spoon on Towel, $7 5 . 0 \%$ on Carrot on Plate, $3 7 . 5 \%$ on Stack Cube, and $1 0 0 . 0 \%$ on Eggplant in Basket.

Evaluation Results on SimplerEnv-Fractal Tab. 2 reports results under Visual Matching and Visual Aggregation settings. MemoryVLA achieves an overall success rate of $7 2 . 7 \%$ , improving CogACT by $+ 4 . 6$ points and surpassing $\pi _ { 0 }$ . By setting, the averages are $7 7 . 7 \%$ on VM and $6 7 . 7 \%$ on VA, gains of $+ 2 . 9$ and $+ 6 . 4$ points over CogACT, respectively. On Open/Close Drawer (VM), it reaches $8 4 . 7 \%$ , a $+ 1 2 . 9$ point improvement over CogACT; under VA we observe larger gains, including $+ 2 4 . 9$ on Open/Close Drawer and $+ 1 1 . 7$ on Put in Drawer.

# 4.3 SIMULATED EVALUATION ON LIBERO

Training and Evaluation Setup We evaluate on the LIBERO (Liu et al., 2023a) benchmark with a Franka robot across five suites: Spatial, Object, Goal, Long-10, and Long-90. The first four suites contain 10 tasks each, and Long-90 contains 90 tasks. Following OpenVLA (Kim et al., 2024), 50 demonstrations per task are used. Separate models are trained for Spatial, Object, and Goal for 20k steps each, while Long-10 and Long-90 are trained jointly for 40k steps. Validation is performed

Table 1: Performance comparison on SimplerEnv-Bridge (Li et al., 2024b) with WidowX robot. CogACT-Large is our re-evaluated baseline using official weight, and MemoryVLA achieves a $+ 1 4 . 6$ gain in average success. Entries marked with * are reproduced from open-pi-zero, which leverage additional proprioceptive state inputs; they also adopt Uniform/Beta timestep sampling.   

<table><tr><td>Method</td><td>Spoon on Towel</td><td>Carrot on Plate</td><td>Stack Cube</td><td>Eggplant in Basket</td><td>Avg. Success</td></tr><tr><td>RT-1-X (O’Neill et al., 2024)</td><td>0.0</td><td>4.2</td><td>0.0</td><td>0.0</td><td>1.1</td></tr><tr><td>OpenVLA (Kim et al., 2024)</td><td>4.2</td><td>0.0</td><td>0.0</td><td>12.5</td><td>4.2</td></tr><tr><td>Octo-Base (Team et al., 2024)</td><td>15.8</td><td>12.5</td><td>0.0</td><td>41.7</td><td>17.5</td></tr><tr><td>TraceVLA (Zheng et al., 2024b)</td><td>12.5</td><td>16.6</td><td>16.6</td><td>65.0</td><td>27.7</td></tr><tr><td>RoboVLMs (Liu et al., 2025b)</td><td>45.8</td><td>20.8</td><td>4.2</td><td>79.2</td><td>37.5</td></tr><tr><td>SpatialVLA (Qu et al., 2025)</td><td>16.7</td><td>25.0</td><td>29.2</td><td>100.0</td><td>42.7</td></tr><tr><td>Magma (Yang et al., 2025a)</td><td>37.5</td><td>29.2</td><td>20.8</td><td>91.7</td><td>44.8</td></tr><tr><td>CogACT-Base (Li et al., 2024a)</td><td>71.7</td><td>50.8</td><td>15.0</td><td>67.5</td><td>51.3</td></tr><tr><td>π0-Uniform* (Black et al., 2024)</td><td>63.3</td><td>58.8</td><td>21.3</td><td>79.2</td><td>55.7</td></tr><tr><td>CogACT-Large (Li et al., 2024a)</td><td>58.3</td><td>45.8</td><td>29.2</td><td>95.8</td><td>57.3</td></tr><tr><td>CronusVLA (Li et al., 2025a)</td><td>66.7</td><td>54.2</td><td>20.8</td><td>100.0</td><td>60.4</td></tr><tr><td>π0-Beta* (Black et al., 2024)</td><td>84.6</td><td>55.8</td><td>47.9</td><td>85.4</td><td>68.4</td></tr><tr><td>MemoryVLA (Ours)</td><td>75.0</td><td>75.0</td><td>37.5</td><td>100.0</td><td>71.9 (+14.6)</td></tr></table>

Table 2: Performance comparison on SimplerEnv-Fractal (Li et al., 2024b) with Google robot. Success rates $( \% )$ are reported for Visual Matching (VM) and Visual Aggregation (VA) suites. MemoryVLA achieves an overall $+ 4 . 6$ gain over CogACT. O./C. denotes Open/Close, and * follow Tab. 1.   

<table><tr><td rowspan="2">Method</td><td colspan="5">Visual Matching (VM)</td><td colspan="5">Visual Aggregation (VA)</td><td rowspan="2">Overall</td></tr><tr><td>Coke Can</td><td>Move Near</td><td>O. / C.Drawer</td><td>Put in DRAW</td><td>Avg.</td><td>Coke Can</td><td>Move Near</td><td>O. / C. Drawn</td><td>Put in DRAW</td><td>Avg.</td></tr><tr><td>Octo-Base (Team et al., 2024)</td><td>17.0</td><td>4.2</td><td>22.7</td><td>0.0</td><td>11.0</td><td>0.6</td><td>3.1</td><td>1.1</td><td>0.0</td><td>1.2</td><td>6.1</td></tr><tr><td>RT-1-X (O&#x27;Neil et al., 2024)</td><td>56.7</td><td>31.7</td><td>59.7</td><td>21.3</td><td>42.4</td><td>49.0</td><td>32.3</td><td>29.4</td><td>10.1</td><td>30.2</td><td>36.3</td></tr><tr><td>OpenVLA (Kim et al., 2024)</td><td>18.0</td><td>56.3</td><td>63.0</td><td>0.0</td><td>34.3</td><td>60.8</td><td>67.7</td><td>28.8</td><td>0.0</td><td>39.3</td><td>36.8</td></tr><tr><td>RoboVLMs (Liu et al., 2025b)</td><td>76.3</td><td>79.0</td><td>44.9</td><td>27.8</td><td>57.0</td><td>50.7</td><td>62.5</td><td>10.3</td><td>0.0</td><td>30.9</td><td>44.0</td></tr><tr><td>TraceVLA (Zheng et al., 2024b)</td><td>45.0</td><td>63.8</td><td>63.1</td><td>11.1</td><td>45.8</td><td>64.3</td><td>60.6</td><td>61.6</td><td>12.5</td><td>49.8</td><td>47.8</td></tr><tr><td>RT-2-X (O&#x27;Neil et al., 2024)</td><td>78.7</td><td>77.9</td><td>25.0</td><td>3.7</td><td>46.3</td><td>82.3</td><td>79.2</td><td>35.5</td><td>20.6</td><td>54.4</td><td>50.4</td></tr><tr><td>Magma (Yang et al., 2025a)</td><td>75.0</td><td>53.0</td><td>58.9</td><td>8.3</td><td>48.8</td><td>68.6</td><td>78.5</td><td>59.0</td><td>24.0</td><td>57.5</td><td>53.2</td></tr><tr><td>SpatialVLA (Qu et al., 2025)</td><td>79.3</td><td>90.0</td><td>54.6</td><td>0.0</td><td>56.0</td><td>78.7</td><td>83.0</td><td>39.2</td><td>6.3</td><td>51.8</td><td>53.9</td></tr><tr><td>π0-Uniform* (Black et al., 2024)</td><td>88.0</td><td>80.3</td><td>56.0</td><td>52.2</td><td>69.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>π0-Beta* (Black et al., 2024)</td><td>97.9</td><td>78.7</td><td>62.3</td><td>46.6</td><td>71.4</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CogACT (Li et al., 2024a)</td><td>91.3</td><td>85.0</td><td>71.8</td><td>50.9</td><td>74.8</td><td>89.6</td><td>80.8</td><td>28.3</td><td>46.6</td><td>61.3</td><td>68.1</td></tr><tr><td>MemoryVLA (Ours)</td><td>90.7</td><td>88.0</td><td>84.7</td><td>47.2</td><td>77.7</td><td>80.5</td><td>78.8</td><td>53.2</td><td>58.3</td><td>67.7</td><td>72.7 (+4.6)</td></tr></table>

every 1k steps and results are reported at the best validation step. Each task is evaluated with 50 trials and per-suite average success rates are reported.

Evaluation Results on LIBERO As shown in Tab. 3, MemoryVLA achieves an overall success rate of $9 6 . 5 \%$ , improving CogACT by $+ 3 . 3$ points and surpassing $\pi _ { 0 }$ . Per-suite success rates are $9 8 . 4 \%$ on Spatial, $9 8 . 4 \%$ on Object, $9 6 . 4 \%$ on Goal, $9 3 . 4 \%$ on Long-10, and $9 5 . 6 \%$ on Long-90. Note that MemoryVLA uses only third-person RGB, without wrist views or proprioceptive states.

# 4.4 SIMULATED EVALUATION ON MIKASA-ROBO

Training and Evaluation Setup We evaluate on the Mikasa-Robo (Cherepanov et al., 2025) benchmark. In our experiments, we compare MemoryVLA with VLA-style baselines used in Mikasa-Robo, including OpenVLA-OFT (Kim et al., 2025), PI-0 (Black et al., 2024), and SpatialVLA (Qu et al., 2025). We additionally reproduce CronusVLA (Li et al., 2025a), a contemporaneous VLA model that uses temporal context. Following the standard Mikasa-Robo protocol using 250 demonstrations per task at $1 2 8 \times 1 2 8$ resolution, 100 evaluation episodes per task and endeffector control. Standard 5 tasks are trained jointly for 20k steps, and validation is performed every 1k steps and results are reported at the best validation step.

Table 3: Performance comparison on LIBERO (Liu et al., 2023a) with Franka robot. Success rates $( \% )$ are reported across five suites. * indicates methods using additional proprioceptive and wrist-camera inputs. CogACT results are reproduced by us. For methods without LIBERO-90 results, we report the average over the first four suites.   

<table><tr><td>Method</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long-10</td><td>Long-90</td><td>Avg. Success</td></tr><tr><td>Diffusion Policy (Chi et al., 2023)</td><td>78.3</td><td>92.5</td><td>68.3</td><td>50.5</td><td>-</td><td>72.4</td></tr><tr><td>Octo (Team et al., 2024)</td><td>78.9</td><td>85.7</td><td>84.6</td><td>51.1</td><td>-</td><td>75.1</td></tr><tr><td>MDT (Reuss et al., 2024)</td><td>78.5</td><td>87.5</td><td>73.5</td><td>64.8</td><td>-</td><td>76.1</td></tr><tr><td>UniACT (Zheng et al., 2025b)</td><td>77.0</td><td>87.0</td><td>77.0</td><td>70.0</td><td>73.0</td><td>76.8</td></tr><tr><td>MaIL (Jia et al., 2024)</td><td>74.3</td><td>90.1</td><td>81.8</td><td>78.6</td><td>-</td><td>83.5</td></tr><tr><td>SpatialVLA (Qu et al., 2025)</td><td>88.2</td><td>89.9</td><td>78.6</td><td>55.5</td><td>46.2</td><td>71.7</td></tr><tr><td>TraceVLA (Zheng et al., 2024b)</td><td>84.6</td><td>85.2</td><td>75.1</td><td>54.1</td><td>-</td><td>74.8</td></tr><tr><td>OpenVLA (Kim et al., 2024)</td><td>84.7</td><td>88.4</td><td>79.2</td><td>53.7</td><td>73.5</td><td>75.9</td></tr><tr><td>CoT-VLA (Zhao et al., 2025)</td><td>87.5</td><td>91.6</td><td>87.6</td><td>69.0</td><td>-</td><td>81.1</td></tr><tr><td>π0-FAST* (Pertsch et al., 2025)</td><td>96.4</td><td>96.8</td><td>88.6</td><td>60.2</td><td>83.1</td><td>85.0</td></tr><tr><td>CronusVLA (Li et al., 2025a)</td><td>90.1</td><td>94.7</td><td>91.3</td><td>68.7</td><td></td><td>86.2</td></tr><tr><td>TriVLA (Liu et al., 2025d)</td><td>91.2</td><td>93.8</td><td>89.8</td><td>73.2</td><td>-</td><td>87.0</td></tr><tr><td>4D-VLA (Zhang et al., 2025a)</td><td>93.8</td><td>92.8</td><td>95.6</td><td>86.5</td><td>-</td><td>92.2</td></tr><tr><td>CogACT (Li et al., 2024a)</td><td>97.2</td><td>98.0</td><td>90.2</td><td>88.8</td><td>92.1</td><td>93.2</td></tr><tr><td>π0* (Black et al., 2024)</td><td>96.8</td><td>98.8</td><td>95.8</td><td>85.2</td><td>-</td><td>94.2</td></tr><tr><td>MemoryVLA (Ours)</td><td>98.4</td><td>98.4</td><td>96.4</td><td>93.4</td><td>95.6</td><td>96.5 (+3.3)</td></tr></table>

Table 4: Performance comparison on Mikasa-Robo (Cherepanov et al., 2025) with Franka robot. Success rates $( \% )$ are reported. CronusVLA results are reproduced by us.   

<table><tr><td>Model</td><td>ShellGame Touch</td><td>Intercept Medium</td><td>Remb. Color3</td><td>Remb. Color5</td><td>Remb. Color9</td><td>Avg. Success</td></tr><tr><td>CronusVLA (Li et al., 2025a)</td><td>32</td><td>5</td><td>31</td><td>13</td><td>9</td><td>18.0</td></tr><tr><td>SpatialVLA (Qu et al., 2025)</td><td>23</td><td>27</td><td>27</td><td>17</td><td>11</td><td>21.0</td></tr><tr><td>OpenVLA-OFT (Kim et al., 2025)</td><td>47</td><td>14</td><td>59</td><td>16</td><td>6</td><td>28.4</td></tr><tr><td>PI-0 (Black et al., 2024)</td><td>33</td><td>42</td><td>35</td><td>22</td><td>15</td><td>29.4</td></tr><tr><td>MemoryVLA (Ours)</td><td>88</td><td>24</td><td>44</td><td>30</td><td>20</td><td>41.2 (+11.8)</td></tr></table>

Evaluation Results on Mikasa-Robo MemoryVLA consistently achieves the highest performance across all tasks, with an average of $1 1 . 8 \%$ improvement over the previous state-of-the-art, especially $+ 4 1 . 0 \%$ on ShellGameTouch task. The results are shown in Tab. 4.

# 4.5 REAL-WORLD EVALUATION

Training and Evaluation Setup We evaluate two real-robot suites, General and Long-horizon Temporal, on Franka and WidowX robots. Both use an Intel RealSense D435 RGB camera mounted in a fixed front view. Images are captured at $6 4 0 \times 4 8 0$ and downsampled to $2 2 4 \times 2 2 4$ . The system is integrated via ROS. For General, each task uses 50-150 demonstrations and is evaluated from randomized initial states. Pick Diverse Fruits comprises five variants with 5 trials per variant (25 total); all other General tasks use 15 trials. For Long-horizon Temporal, each task uses 200-300 demonstrations and is evaluated with 10-15 trials using step-wise scoring to reflect progress over sub-goals. Training runs for approximately 5k–20k steps depending on the task and data size.

Evaluation Results on Real-world As shown in Tab. 5, MemoryVLA achieves average success scores of $85 \%$ on general tasks and $83 \%$ on long-horizon temporal tasks, exceeding CogACT by $\mathbf { + 9 }$ and $\mathbf { + 2 6 }$ percentage points, respectively, and surpassing $\pi _ { 0 }$ across both suites. On general tasks, it exceeds the strongest baseline on every task, with notable gains on Egg in Pan $( + 1 3 )$ and Egg in Oven $( + 2 0 )$ . On long-horizon temporal tasks, improvements are larger, including $+ 4 3$ on Seq. Push Buttons, $+ 3 8$ on Change Food, $+ 3 2$ on Guess Where, and $+ 1 7$ on Clean Table & Count. These results demonstrate strong real-world competence and highlight the benefits of temporal memory.

Table 5: Performance comparison on real-world experiments with Franka and WidowX robots. Success scores $( \% )$ are reported over six general tasks and six long-horizon temporal tasks. All methods are evaluated with only third-person RGB observation and language instruction.   

<table><tr><td rowspan="2">Method</td><td colspan="7">General Tasks</td></tr><tr><td>Insert 
Circle</td><td>Egg 
in Pan</td><td>Egg 
in Oven</td><td>Stack 
Cups</td><td>Stack 
Blocks</td><td>Pick Diverse 
Fruits</td><td>Avg. 
Success</td></tr><tr><td>OpenVLA (Kim et al., 2024)</td><td>47</td><td>27</td><td>53</td><td>40</td><td>13</td><td>4</td><td>31</td></tr><tr><td>π0 (Black et al., 2024)</td><td>67</td><td>73</td><td>73</td><td>87</td><td>53</td><td>80</td><td>72</td></tr><tr><td>CogACT (Li et al., 2024a)</td><td>80</td><td>67</td><td>60</td><td>93</td><td>80</td><td>76</td><td>76</td></tr><tr><td>MemoryVLA (Ours)</td><td>87</td><td>80</td><td>80</td><td>93</td><td>87</td><td>84</td><td>85 (+9)</td></tr><tr><td rowspan="2">Method</td><td colspan="7">Long-horizon Temporal Tasks</td></tr><tr><td>Seq. Push 
Buttons</td><td>Change 
Food</td><td>Guess 
Where</td><td>Clean Table 
&amp; Count</td><td>Pick Place 
Order</td><td>Clean Rest. 
Table</td><td>Avg. 
Success</td></tr><tr><td>OpenVLA (Kim et al., 2024)</td><td>6</td><td>3</td><td>0</td><td>15</td><td>27</td><td>0</td><td>9</td></tr><tr><td>π0 (Black et al., 2024)</td><td>25</td><td>42</td><td>24</td><td>61</td><td>82</td><td>80</td><td>52</td></tr><tr><td>CogACT (Li et al., 2024a)</td><td>15</td><td>47</td><td>40</td><td>67</td><td>90</td><td>84</td><td>57</td></tr><tr><td>MemoryVLA (Ours)</td><td>58</td><td>85</td><td>72</td><td>84</td><td>100</td><td>96</td><td>83 (+26)</td></tr></table>

Table 6: Ablation on memory type and length. We report average success rates $( \% )$ on SimplerEnv-Bridge tasks.   

<table><tr><td></td><td>Variant</td><td>Avg. Success</td></tr><tr><td rowspan="3">Memory Type</td><td>Cognitive Mem.</td><td>63.5</td></tr><tr><td>Perceptual Mem.</td><td>64.6</td></tr><tr><td>Both</td><td>71.9</td></tr><tr><td rowspan="3">Memory Length</td><td>4</td><td>67.7</td></tr><tr><td>16</td><td>71.9</td></tr><tr><td>64</td><td>67.7</td></tr></table>

Table 7: Ablation on memory retrieval, fusion, consolidation. We report average success rates $( \% )$ on SimplerEnv-Bridge tasks.   

<table><tr><td></td><td>Variant</td><td>Avg. Success</td></tr><tr><td rowspan="2">Retrieval</td><td>w/o Timesteps PE</td><td>69.8</td></tr><tr><td>w/ Timesteps PE</td><td>71.9</td></tr><tr><td rowspan="2">Fusion</td><td>Add</td><td>67.7</td></tr><tr><td>Gate</td><td>71.9</td></tr><tr><td rowspan="2">Consolidation</td><td>FIFO</td><td>66.7</td></tr><tr><td>Token Merge</td><td>71.9</td></tr></table>

# 4.6 ABLATION STUDIES

We ablate memory design on SimplerEnv-Bridge to quantify each choice. As shown in Tab. 6, combining perceptual and cognitive memory attains $7 1 . 9 \%$ , compared with $6 3 . 5 \%$ for cognitiveonly and $6 4 . 6 \%$ for perceptual-only. A memory length of 16 performs best at $7 1 . 9 \%$ , whereas 4 and 64 drop to $6 7 . 7 \%$ . Tab. 7 evaluates retrieval, fusion, and consolidation. Adding timestep positional encoding increases performance from $6 9 . 8 \%$ to $7 1 . 9 \%$ . Gate fusion reaches $7 1 . 9 \%$ , compared with $6 7 . 7 \%$ for simple addition. Token-merge consolidation achieves $7 1 . 9 \%$ versus $6 6 . 7 \%$ with FIFO.

# 5 CONCLUSION

Inspired by cognitive science, we propose MemoryVLA, a Cognition-Memory-Action framework for robotic manipulation. It uses a hippocampus-like Perceptual–Cognitive Memory Bank that cooperates with working memory to capture temporal dependencies. VLM commonsense priors further support high-level cognition, while a memory-conditioned diffusion action expert generates temporally aware actions. Across $1 5 0 +$ tasks with $5 0 0 +$ variations on 3 robots spanning SimplerEnv, LIBERO, and real-world, MemoryVLA consistently surpasses CogACT and $\pi _ { 0 }$ , achieves stateof-the-art performance, with notable gains on challenging long-horizon temporal tasks. It also demonstrates strong robustness and generalization under diverse OOD conditions. Future directions include (i) developing memory reflection, aligning long-term memory to the LLM input space to enable embedding-space chain-of-thought reasoning; and (ii) building lifelong memory through biologically inspired consolidation that distills frequently reused experiences into permanent representations, thereby supporting scalable generalization across scenes, tasks, and embodiments.

# ACKNOWLEDGMENTS

This work was supported by the National Science and Technology Major Project of China under Grant No. 2023ZD0121300, the National Natural Science Foundation of China under Grants U24B20173 and 62321005, the Scientific Research Innovation Capability Support Project for Young Faculty under Grant ZYGXQNJSKYCXNLZCXM-I20.

# REPRODUCIBILITY STATEMENT

The main paper specifies the model architectures, training setups, and experimental protocols, and the appendix provides additional details on hyperparameters, datasets, preprocessing and data augmentation, training procedures, evaluation methods and scoring rules, robustness and generalization analyses, task specifications, and qualitative results. For full reproducibility, we release:

• Code: https://github.com/shihao1895/MemoryVLA   
• Models: https://huggingface.co/collections/shihao1895/memoryvla   
• Logs: https://huggingface.co/collections/shihao1895/memoryvla   
• Robotic videos: https://shihao1895.github.io/MemoryVLA

# REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
Alan D Baddeley and Graham James Hitch. Working memory (vol. 8). New York: GA Bower (ed), Recent advances in learning and motivation, 1974.   
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023a.   
Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023b.   
Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. pi-0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.   
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Joseph Dabis, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, Jasmine Hsu, et al. Rt-1: Robotics transformer for real-world control at scale. arXiv preprint arXiv:2212.06817, 2022.   
Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.   
Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025a.   
Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, and Hongyang Li. Univla: Learning to act anywhere with task-centric latent actions. arXiv preprint arXiv:2505.06111, 2025b.   
Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou, Julien Mairal, Piotr Bojanowski, and´ Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021.   
Egor Cherepanov, Nikita Kachaev, Alexey K Kovalev, and Aleksandr I Panov. Memory, benchmark & robots: A benchmark for solving complex tasks with reinforcement learning. arXiv preprint arXiv:2502.10550, 2025.   
Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, pp. 02783649241273668, 2023.   
Cunxin Fan, Xiaosong Jia, Yihang Sun, Yixiao Wang, Jianglan Wei, Ziyang Gong, Xiangyu Zhao, Masayoshi Tomizuka, Xue Yang, Junchi Yan, et al. Interleave-vla: Enhancing robot manipulation with interleaved image-text instructions. arXiv preprint arXiv:2505.02152, 2025.   
Tingliang Feng, Hao Shi, Xueyang Liu, Wei Feng, Liang Wan, Yanlin Zhou, and Di Lin. Open compound domain adaptation with object style compensation for semantic segmentation. Advances in Neural Information Processing Systems, 36:63136–63149, 2023.   
Chongkai Gao, Zixuan Liu, Zhenghao Chi, Junshan Huang, Xin Fei, Yiwen Hou, Yuxuan Zhang, Yudi Lin, Zhirui Fang, Zeyu Jiang, et al. Vla-os: Structuring and dissecting planning representations and paradigms in vision-language-action models. arXiv preprint arXiv:2506.17561, 2025.

Ankit Goyal, Jie Xu, Yijie Guo, Valts Blukis, Yu-Wei Chao, and Dieter Fox. Rvt: Robotic view transformer for 3d object manipulation. In Conference on Robot Learning, pp. 694–710. PMLR, 2023.   
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.   
Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. ¨ Neural computation, 9(8): 1735–1780, 1997.   
Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 7132–7141, 2018.   
Xiaogang Jia, Qian Wang, Atalay Donat, Bowen Xing, Ge Li, Hongyi Zhou, Onur Celik, Denis Blessing, Rudolf Lioutikov, and Gerhard Neumann. Mail: Improving imitation learning with selective state space models. In 8th Annual Conference on Robot Learning, 2024.   
Siddharth Karamcheti, Suraj Nair, Ashwin Balakrishna, Percy Liang, Thomas Kollar, and Dorsa Sadigh. Prismatic vlms: Investigating the design space of visually-conditioned language models. In Forty-first International Conference on Machine Learning, 2024.   
Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.   
Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.   
Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.   
Hao Li, Shuai Yang, Yilun Chen, Yang Tian, Xiaoda Yang, Xinyi Chen, Hanqing Wang, Tai Wang, Feng Zhao, Dahua Lin, et al. Cronusvla: Transferring latent motion across time for multi-frame prediction in manipulation. arXiv preprint arXiv:2506.19816, 2025a.   
Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, et al. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation. arXiv preprint arXiv:2411.19650, 2024a.   
Runhao Li, Wenkai Guo, Zhenyu Wu, Changyuan Wang, Haoyuan Deng, Zhenyu Weng, Yap-Peng Tan, and Ziwei Wang. Map-vla: Memory-augmented prompting for vision-language-action model in robotic manipulation. arXiv preprint arXiv:2511.09516, 2025b.   
Xinghang Li, Minghuan Liu, Hanbo Zhang, Cunjun Yu, Jie Xu, Hongtao Wu, Chilam Cheang, Ya Jing, Weinan Zhang, Huaping Liu, et al. Vision-language foundation models as effective robot imitators. arXiv preprint arXiv:2311.01378, 2023.   
Xuanlin Li, Kyle Hsu, Jiayuan Gu, Karl Pertsch, Oier Mees, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, et al. Evaluating real-world robot manipulation policies in simulation. arXiv preprint arXiv:2405.05941, 2024b.   
Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023a.   
Chenghao Liu, Jiachen Zhang, Chengxuan Li, Zhimu Zhou, Shixin Wu, Songfang Huang, and Huiling Duan. Ttf-vla: Temporal token fusion via pixel-attention integration for vision-languageaction models. arXiv preprint arXiv:2508.19257, 2025a.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023b.   
Huaping Liu, Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, and Hanbo Zhang. Towards generalist robot policies: What matters in building vision-language-action models. arXiv preprint arXiv:2412.14058, 2025b.   
Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, Mengzhen Liu, et al. Hybridvla: Collaborative diffusion and autoregression in a unified vision-language-action model. arXiv preprint arXiv:2503.10631, 2025c.   
Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pp. 38–55. Springer, 2024a.   
Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a diffusion foundation model for bimanual manipulation. arXiv preprint arXiv:2410.07864, 2024b.   
Yingfei Liu, Junjie Yan, Fan Jia, Shuailin Li, Aqi Gao, Tiancai Wang, and Xiangyu Zhang. Petrv2: A unified framework for 3d perception from multi-camera images. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3262–3272, 2023c.   
Zhenyang Liu, Yongchong Gu, Sixiao Zheng, Xiangyang Xue, and Yanwei Fu. Trivla: A unified triple-system-based unified vision-language-action model for general robot control. arXiv preprint arXiv:2507.01424, 2025d.   
Oier Mees, Dibya Ghosh, Karl Pertsch, Kevin Black, Homer Rich Walke, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, et al. Octo: An open-source generalist robot policy. In First Workshop on Vision-Language Models for Navigation and Manipulation at ICRA 2024, 2024.   
Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, ´ Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.   
Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pp. 6892–6903. IEEE, 2024.   
William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.   
Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokenization for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025.   
Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, et al. Spatialvla: Exploring spatial representations for visuallanguage-action model. arXiv preprint arXiv:2501.15830, 2025.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.   
Moritz Reuss, Omer Erdinc¸ Ya ¨ gmurlu, Fabian Wenzel, and Rudolf Lioutikov. Multimodal ˘ diffusion transformer: Learning versatile behavior from multimodal goals. arXiv preprint arXiv:2407.05996, 2024.   
Valerie F Reyna and Charles J Brainerd. Fuzzy-trace theory: An interim synthesis. Learning and individual Differences, 7(1):1–75, 1995.

Hao Shi, Bin Xie, Yingfei Liu, Yang Yue, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, and Gao Huang. Spatialactor: Exploring disentangled spatial representations for robust robotic manipulation. arXiv preprint arXiv:2511.09555, 2025.   
Mohit Shridhar, Lucas Manuelli, and Dieter Fox. Perceiver-actor: A multi-task transformer for robotic manipulation. In Conference on Robot Learning, pp. 785–799. PMLR, 2023.   
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.   
Lin Sun, Bin Xie, Yingfei Liu, Hao Shi, Tiancai Wang, and Jiale Cao. Geovla: Empowering 3d representations in vision-language-action models. arXiv preprint arXiv:2508.09071, 2025.   
Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, et al. Octo: An open-source generalist robot policy. arXiv preprint arXiv:2405.12213, 2024.   
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.   
Endel Tulving et al. Episodic and semantic memory. Organization of memory, 1(381-403):1, 1972.   
Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pp. 1723–1736. PMLR, 2023.   
Shihao Wang, Yingfei Liu, Tiancai Wang, Ying Li, and Xiangyu Zhang. Exploring object-centric temporal modeling for efficient multi-view 3d object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 3621–3631, 2023.   
Yulin Wang, Yang Yue, Yang Yue, Huanqian Wang, Haojun Jiang, Yizeng Han, Zanlin Ni, Yifan Pu, Minglei Shi, Rui Lu, et al. Emulating human-like adaptive vision for efficient and flexible machine visual perception. Nature Machine Intelligence, pp. 1–19, 2025.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
Junjie Wen, Yichen Zhu, Jinming Li, Zhibin Tang, Chaomin Shen, and Feifei Feng. Dexvla: Vision-language model with plug-in diffusion expert for general robot control. arXiv preprint arXiv:2502.05855, 2025.   
Dongming Wu, Yanping Fu, Saike Huang, Yingfei Liu, Fan Jia, Nian Liu, Feng Dai, Tiancai Wang, Rao Muhammad Anwer, Fahad Shahbaz Khan, et al. Ragnet: Large-scale reasoning-based affordance segmentation benchmark towards general grasping. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 11980–11990, 2025.   
Bin Xie, Erjin Zhou, Fan Jia, Hao Shi, Haoqiang Fan, Haowei Zhang, Hebei Li, Jianjian Sun, Jie Bin, Junwen Huang, et al. Dexbotic: Open-source vision-language-action toolbox. arXiv preprint arXiv:2510.23511, 2025.   
Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, et al. Magma: A foundation model for multimodal ai agents. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 14203–14214, 2025a.   
Shuai Yang, Hao Li, Yilun Chen, Bin Wang, Yang Tian, Tai Wang, Hanqing Wang, Feng Zhao, Yiyi Liao, and Jiangmiao Pang. Instructvla: Vision-language-action instruction tuning from understanding to manipulation. arXiv preprint arXiv:2507.17520, 2025b.   
Yang Yue, Yulin Wang, Bingyi Kang, Yizeng Han, Shenzhi Wang, Shiji Song, Jiashi Feng, and Gao Huang. Deer-vla: Dynamic inference of multimodal large language models for efficient robot execution. Advances in Neural Information Processing Systems, 37:56619–56643, 2024.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986, 2023.   
Jiahui Zhang, Yurui Chen, Yueming Xu, Ze Huang, Yanpeng Zhou, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, Xingyue Quan, Hang Xu, et al. 4d-vla: Spatiotemporal vision-language-action pretraining with cross-scene calibration. arXiv preprint arXiv:2506.22242, 2025a.   
Yani Zhang, Dongming Wu, Hao Shi, Yingfei Liu, Tiancai Wang, Haoqiang Fan, and Xingping Dong. Grounding beyond detection: Enhancing contextual understanding in embodied 3d grounding. arXiv preprint arXiv:2506.05199, 2025b.   
Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, et al. Cot-vla: Visual chain-of-thought reasoning for visionlanguage-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1702–1713, 2025.   
Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.   
Henry Zheng, Hao Shi, Yong Xien Chng, Rui Huang, Zanlin Ni, Tianyi Tan, Qihang Peng, Yepeng Weng, Zhongchao Shi, and Gao Huang. Denseg: Alleviating vision-language feature sparsity in multi-view 3d visual grounding. In Autonomous Grand Challenge CVPR 2024 Workshop, volume 2, pp. 6, 2024a.   
Henry Zheng, Hao Shi, Qihang Peng, Yong Xien Chng, Rui Huang, Yepeng Weng, zhongchao shi, and Gao Huang. Densegrounding: Improving dense language-vision semantics for ego-centric 3d visual grounding. In The Thirteenth International Conference on Learning Representations, 2025a.   
Jinliang Zheng, Jianxiong Li, Dongxiu Liu, Yinan Zheng, Zhihao Wang, Zhonghong Ou, Yu Liu, Jingjing Liu, Ya-Qin Zhang, and Xianyuan Zhan. Universal actions for enhanced embodied foundation models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 22508–22519, 2025b.   
Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daume III, Andrey Kolobov, ´ Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies. arXiv preprint arXiv:2412.10345, 2024b.   
Yifan Zhong, Fengshuo Bai, Shaofei Cai, Xuchuan Huang, Zhang Chen, Xiaowei Zhang, Yuanfei Wang, Shaoyang Guo, Tianrui Guan, Ka Nam Lui, et al. A survey on vision-language-action models: An action tokenization perspective. arXiv preprint arXiv:2507.01925, 2025.   
Junbao Zhou, Ziqi Pang, and Yu-Xiong Wang. Rmem: Restricted memory banks improve video object segmentation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18602–18611, 2024.   
Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning, pp. 2165–2183. PMLR, 2023.

# APPENDIX

A LLM Usage 18   
B Robustness and Generalization Evaluation 18

B.1 Real-world Evaluation 18   
B.2 Simulation Evaluation . 18

C Additional Training Details 19

C.1 Hyper-parameters 19   
C.2 Training Data 20   
C.3 Training Setup . 22   
C.4 Data Augmentation 22

D Additional Evaluation Details 23

D.1 SimplerEnv 23   
D.2 LIBERO 23   
D.3 Mikasa-Robo 23   
D.4 Real-world 23

E Case Study of Memory Retrieval 24

F Data Length Statistics 25   
G Additional Ablation Study 25   
H Inference Efficiency 26

I Zero-shot Task Generalization 27   
J Compare with Temporal Context VLA Baselines 27   
K Task Details 27

K.1 Real-world Tasks 29   
K.2 SimplerEnv Tasks 29   
K.3 LIBERO Tasks 30

L Visualization of Memory Dependence in Real-World Tasks 30   
M Qualitative Results 32

M.1 Real-world Evaluation 32   
M.2 Simulation Evaluation 33

# A LLM USAGE

Large Language Models (LLMs) were used only for grammar and wording polishing. All research content and contributions are entirely the responsibility of the authors, with no involvement of LLMs.

# B ROBUSTNESS AND GENERALIZATION EVALUATION

# B.1 REAL-WORLD EVALUATION

![](images/ecf6e82d28eed8acd574b68459ebb07e90dcb2470293735d95d24d50c011b0d4.jpg)  
(a) Various OOD Variants of Pick Place Order   
Base

![](images/97af3889aea93eadcecc5a3fbf71d080769715a19e5436451ed9deb481c9770b.jpg)  
Unseen Background

![](images/7b5a9799ccb70ed7e9bfca420472197919788e5462eb4ae35de70a9b8dc2c891.jpg)  
Unseen Distractors

![](images/d011f8910b2417d4a00a748703676384db3fe5308122c343ce6635214ec3f3f5.jpg)  
Unseen Lighting

![](images/e72a98252803e9059a4064b2dd6ee2cadcc990e3ede0b3ead3b5957798653cd0.jpg)  
Unseen Object

![](images/03f893fa879313c1f59088b9896f2b2444b6cb9ce8352c484edb4d4bf983332e.jpg)  
Unseen Container

![](images/8b1d9e2a5b000354e9abd4d942a84f62ecf52ecc607abb74773668fbdc7982d5.jpg)  
Occlusion

![](images/28eaa6f97fe846abb5fa012f6b368bfabe5a4d84d3f5c51d0fcb6be3b694c7c6.jpg)  
(b) Various OOD Variants of Clean Restaurant Table   
Base

![](images/9d2fef4cda2f36af5b96d34591773157f31f16010106207673bbe09626d6a2d7.jpg)  
Unseen Background

![](images/6d675b6b317686e56a4c487c8140b568fd57075633e7820cf8f2e465f5f3d0aa.jpg)  
Unseen Distractors

![](images/36b84ccd77b9a231d9976508c355c5082e0098a49f3912ab0712f0415794d0aa.jpg)  
Unseen Lighting

![](images/54dcc7a1a662966320e5439907a0f7f5b311c5a701c7d0c5a4243721ff21b10f.jpg)  
Unseen Object

![](images/789d8f5d09a70889d2ffcb2cd20a72d0304a21a586f8edffe725f1e1669ecbd8.jpg)  
Unseen Container

![](images/c7830dd0d355e23cf784204763028dae76ecce0c7b8ec7a49c4fb78ed6cea12d.jpg)  
Occlusion

![](images/f9814a774efcdc88d963466ea0c60279ed38607f5613a420a07b3910c2e1bd47.jpg)  
(c) Generalization of Pick Place Order

![](images/448916a783e570361a5834f3a09fe577621d7ab1e3fc6456384f31be7b311134.jpg)  
(d) Generalization of Clean Restaurant Table   
Figure 5: Robustness and generalization under out-of-distribution (OOD) conditions in realworld. (a,b) Examples of OOD variants for two representative tasks (Pick Place Order and Clean Restaurant Table), including unseen backgrounds, distractors, lighting, novel objects/containers, and occlusion. (c,d) Quantitative results showing that MemoryVLA maintains high success rates across these OOD variants, demonstrating strong robustness and generalization in real-world environments.

We further assess the robustness and generalization of MemoryVLA in real-world environments under diverse out-of-distribution (OOD) variants. Fig. 5 shows two representative tasks, Pick Place Order and Clean Restaurant Table, evaluated under unseen backgrounds, distractors, novel objects/containers, lighting variations, and occlusions.

For Pick Place Order, MemoryVLA attains near-perfect success under the base setting $( 1 0 0 \% )$ , unseen background $( 1 0 0 \% )$ , unseen distractors $( 9 2 \% )$ , unseen lighting $( 9 6 \% )$ , unseen container $( 1 0 0 \% )$ , and occlusion $( 9 6 \% )$ , with a moderate drop on unseen objects $( 8 9 \% )$ . For Clean Restaurant Table, the base success rate is $96 \%$ , with unseen background $( 9 2 \% )$ , unseen distractors $( 8 6 \% )$ , unseen lighting $( 9 4 \% )$ , unseen object $( 9 4 \% )$ , unseen container $( 9 6 \% )$ , and occlusion $( 9 4 \% )$ .

These results confirm that MemoryVLA maintains consistently high performance across a wide range of real-world OOD conditions, demonstrating strong robustness and generalization.

# B.2 SIMULATION EVALUATION

We further conduct robustness and generalization experiments in simulation, considering both pickand-move tasks and hinge-like object manipulation tasks. Fig. 6 presents results on Pick Coke Can

![](images/8c4482f779d53abbb52210a44b4d555d9f19d10965def758345502b746214199.jpg)  
(a) Various OOD Variants of Pick Coke Can

![](images/dafd5e6ab8f28975dfe4e22c0b11cb0b323f01111220c8c7e709bc19075fa36e.jpg)  
Base

![](images/ea58c3811426b4e788e96dd9b4e5ee01723bfd991c33373deeb03c43c24ae260.jpg)  
Unseen Backgroun

![](images/c3a67425a7b1f51226b5225a8a6cbec34d5b093e157697f128830ed9f91b2b34.jpg)  
Unseen Distractors

![](images/0f679c1b7e8d102bc5a9fabe597e6c1170f3eaa5464a6a83430119cedad419e6.jpg)  
Unseen Distractors

![](images/9dcc17dcd85b4c5ecf2d3e4271f71fcbd3181146063dddc25ae405693558a613.jpg)  
Unseen Lighting

![](images/bcd2f2725ec0cf8ffd25771bbbe881e6926abcf6086840e3903f04f1de72cdc7.jpg)  
Unseen Lighting

![](images/73628dc0f31d927a0fd99ea4ef94ec9f9c480c774d944f25d6a17932a11950f8.jpg)  
Unseen Camera View

![](images/810759002ba19cf455ccec425675c2d3d34326ddc03888dc7e1a29f22e40d106.jpg)  
Unseen Camera View

![](images/dcbbf7d026d40d8d1eb169a85255d1e6c3f9d2f5bd68df4096813165acbbeae5.jpg)  
(b) Various OOD Variants of Move Near

![](images/c3fdc2d30284399cf7f9d0e4a9faa6822e0f3ed07cbbb31f2228ceae278a6ed4.jpg)  
Base

![](images/69f1af1f9179d3dd1ca26e142d3a51cf5a8c423366bf4058fbd3f236a6a9b684.jpg)  
Unseen Distractors

![](images/aca56cccb47872b09b742fff1d06920273027e02c485e98331b47e77510a8de9.jpg)  
Unseen Background

![](images/2fdcf6aff2b389e734c5e078c14f4f83b7fa62ca8d1f72fdeb6053923887e4b5.jpg)  
Unseen Background

![](images/de57055f3ed584e47b1b8ff54ba970e513c5f9282f9bdfe004384adaf4b49dc9.jpg)  
Unseen Lighting

![](images/364d6de984ac2afc0709756119fe4c863fbd69ce21e17b00c294449b2b598dd2.jpg)  
Unseen Lighting

![](images/e117ab9b36f293bb7c581503525f81a4fea0f0caf2e37a50a229e5ced6379cb2.jpg)  
Unseen Camera View

![](images/b788b01dd0aeb40549fbadc0ce4d7e1436e5fe2c0b5a0d866a36ba4301b3d2bc.jpg)  
Unseen Camera View

![](images/fe50fe9f5a26316068f9c7d929723a1a5a51154c3fa2420fe6c1082ff1bf3290.jpg)  
Unseen Texture

![](images/e5415645adce2de4719035dbc35bd7773eb23fb4b7ff2bceec84af81d3897bea.jpg)  
Unseen Texture

![](images/89b79edde0dfef7d084068f9df0274eee10893293e0dad4ad0af9596bbe26bcf.jpg)  
Figure 6: Robustness and generalization under out-of-distribution (OOD) variants in simulation: Pick and Move tasks. (a) Pick Coke Can and (b) Move Near tasks evaluated under unseen backgrounds, distractors, lighting, textures, and camera views. Bar plots report the corresponding success rates, showing that MemoryVLA maintains strong performance across most shifts, with the largest degradation under unseen camera views.

and Move Near, while Fig. 7 covers Open/Close Drawer and Place Apple Into Drawer. These tasks are evaluated under unseen backgrounds, distractors, lighting, textures, and camera views.

For Pick Coke Can, MemoryVLA achieves a base success rate of $9 2 . 0 \%$ , with unseen distractors $( 9 0 . 7 \% )$ , unseen background $( 8 6 . 7 \% )$ , unseen lighting $( 9 0 . 7 \% )$ , and unseen texture $( 8 6 . 7 \% )$ , while performance drops substantially under unseen camera views $( 4 2 . 0 \% )$ . For Move Near, the base success rate is $7 6 . 0 \%$ , with unseen distractors $( 8 4 . 0 \% )$ , unseen background $( 8 6 . 0 \% )$ , unseen lighting $( 8 4 . 0 \% )$ , unseen camera view $( 5 8 . 0 \% )$ , and unseen texture $( 8 6 . 0 \% )$ . For hinge-like object manipulation, Open/Close Drawer yields a base success rate of $4 6 . 3 \%$ , unseen background $( 5 6 . 4 \% )$ , unseen lighting $( 4 9 . 1 \% )$ , and unseen texture $( 5 7 . 4 \% )$ . For Place Apple Into Drawer, the base success rate is $7 2 . 0 \%$ , with unseen background $( 6 6 . 0 \% )$ , unseen lighting $( 5 2 . 0 \% )$ , and unseen texture $( 5 0 . 0 \% )$ .

These results show that MemoryVLA generalizes well across moderate distribution shifts such as distractors, backgrounds, and textures, but suffers more under severe changes, especially unseen camera views.

# C ADDITIONAL TRAINING DETAILS

# C.1 HYPER-PARAMETERS

We summarize the main hyperparameters used in our experiments. The global batch size is 256 $3 2 \times 8$ GPUs), the learning rate is $2 \times 1 0 ^ { - 5 }$ , and gradients are clipped at a max norm of 1.0. The policy predicts a 16-step action chunk, and perceptual tokens use 256 channels. The diffusion policy

![](images/463fcbed3e7d4b1e49a4c0ada8df19028321cb1e859c9d442a561237bf1fc32f.jpg)  
Figure 7: Robustness and generalization under out-of-distribution (OOD) variants in simulation: Hinge-like object manipulation. (a) OOD variants of Open/Close Drawer and (b) Place Apple Into Drawer tasks, including unseen backgrounds, distractors, lighting, textures, and camera views. Quantitative results indicate that MemoryVLA generalizes well under moderate shifts, while performance drops notably with camera views changes.

Table 8: Training and model hyperparameters.   

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Batch size</td><td>32 × 8</td></tr><tr><td>Learning rate</td><td>2 × 10-5</td></tr><tr><td>Repeated diffusion steps</td><td>4</td></tr><tr><td>Action trunking size</td><td>16</td></tr><tr><td>Perceptual token channels</td><td>256</td></tr><tr><td>Max grad. norm</td><td>1.0</td></tr><tr><td>CFG scale (classifier-free guidance)</td><td>1.5</td></tr></table>

uses 4 repeated diffusion steps during training; inference uses DDIM with 10 sampling steps and a classifier-free guidance scale of 1.5. See Tab. 8 for a concise summary.

# C.2 TRAINING DATA

Bridge v2 For the SimplerEnv-Bridge benchmark, we train on BridgeData v2 (Walke et al., 2023), a large-scale, language-conditioned real-robot manipulation dataset of roughly 60,000 teleoperated trajectories collected on WidowX robots across diverse tabletop settings. Episodes pair language instructions with demonstrations of skills such as picking, placing, pushing, stacking, and folding.

RT-1 For the SimplerEnv-Fractal benchmark, we use RT-1 (Brohan et al., 2022), a large-scale real-world dataset of roughly 130,000 episodes spanning $^ { 7 0 0 + }$ tasks, collected over 17 months by the Google Robot fleet and paired with natural-language instructions.

![](images/d9d1573786dc923a5968fddb2e4dd2084f739333538ef62e7f75c714e67df249.jpg)  
Figure 8: Franka robot setup.

![](images/1f2f0da199e7036a4a57a25d65fee8ab9d049dce86176f3947c89db2427ae771.jpg)  
Figure 9: WidowX robot setup.

LIBERO LIBERO (Liu et al., 2023a) provides simulation tasks with a Franka robot across five suites: Spatial, Object, Goal, Long-10, and Long-90, totaling 130 language-conditioned tasks. Each task supplies 50 demonstrations.

Mikasa-Robo Mikasa-Robo (Cherepanov et al., 2025) comprises five memory-dependent manipulation tasks, each with 250 officially provided demonstrations, using $\Delta$ end-effector control.

Real-world We collect real demonstrations on Franka and WidowX robots using a fixed thirdperson RGB setup, as shown in Fig. 8 and 9. A front-facing Intel RealSense D435 captures $6 4 0 \times 4 8 0$ RGB at 30 fps. Franka uses a single end-effector per experiment, either the stock parallel gripper or a Robotiq parallel gripper. Demonstrations are gathered by joystick teleoperation. The General suite uses 50-150 demonstrations per task, and the Long-horizon Temporal suite uses 200-300 per task. The system is integrated in ROS.

After collection, we perform a standardized preprocessing pipeline. Frames are downsampled to $2 2 4 \times 2 2 4$ . We then subsample the video stream by retaining a frame whenever the end-effector translation since the last kept frame exceeds $0 . 0 1 \mathrm { m }$ or the orientation change exceeds 0.4 rad, and we also enforce a maximum gap of 120 frames between kept frames. The processed episodes are converted into the RLDS format for downstream training.

# C.3 TRAINING SETUP

SimplerEnv-Bridge On Bridge v2, models are trained for 50k steps with a stream dataloader. Each episode is unpacked into consecutive frames tagged with its episode ID. During training, batches are filled sequentially with frames from a single episode whenever possible. If an episode ends before the batch is complete, the remaining slots are filled with frames from the following episode. A new batch then continues from the position where the previous one stopped, ensuring that in-episode temporal order is always preserved. The memory length is fixed to 16.

SimplerEnv-Fractal Models are trained for 80k steps on RT-1. The benchmark defines two protocols: Visual Matching (VM), which mirrors the real-robot setup, and Visual Aggregation (VA), which perturbs background, lighting, distractors, and textures to test robustness. The dataloader design and memory length follow the same setup as in SimplerEnv-Bridge.

LIBERO Following OpenVLA (Kim et al., 2024), we train with 50 demonstrations per task after removing failed trajectories from the dataset. Spatial, Object, and Goal suites are trained separately for 20k steps each, while Long-10 and Long-90 are treated as a single family of long-horizon data and trained jointly for 40k steps. The dataloader adopts a grouped sampling strategy: in each iteration, 16 frames are randomly sampled from within a single episode, matching the memory length of 16 used throughout training. The dataloader adopts a grouped sampling strategy in which each batch is divided into multiple groups, and each group consists of several frames drawn from a single episode. Frames within a group are kept in temporal order. The memory length is set to 16.

Mikasa-Robo Following Mikasa-Robo (Cherepanov et al., 2025), we adopt the standard protocol with five tasks and train jointly on all 1,250 demonstrations for 20k steps, using $1 2 8 \times 1 2 8$ RGB observations and $\Delta$ end-effector control. We reuse the same dataloader setup as in LIBERO, and set the memory length to 16.

Real-world Models are trained for $5 \mathrm { k } { - } 2 0 \mathrm { k }$ steps depending on task and dataset size. The general tasks contain 50-150 demonstrations per task, while long-horizon temporal tasks use 200-300 demonstrations per task. The memory length is set to 16 for general tasks and 256 for long-horizon temporal tasks.

# C.4 DATA AUGMENTATION

We apply standard per-frame augmentations to the third-person RGB stream during training. Augmentations are applied in a fixed order: random resized crop, random brightness, random contrast, random saturation, and random hue. The crop samples $9 0 \%$ of the image area with aspect ratio 1.0 and resizes to $2 2 4 \times 2 2 4$ . Brightness is perturbed with magnitude 0.2, contrast and saturation are scaled in [0.8, 1.2], and hue is shifted by up to 0.05. All augmentations are disabled at evaluation.

# D ADDITIONAL EVALUATION DETAILS

# D.1 SIMPLERENV

Evaluation follows the official CogACT protocol (Li et al., 2024a). We adopt the same evaluation scripts and use the adaptive action ensemble strategy introduced in CogACT, with ensemble coefficient $\alpha = 0 . 1$ , ensemble horizon set to 7 for Bridge and 2 for Fractal. For Bridge, models are trained for 50k steps and validated every $2 . 5 \mathrm { k }$ steps, since the denoising objective of diffusion models does not reliably indicate policy quality, we report success rates at the best validation step. For Fractal, training runs for 80k steps with validation every 5k steps, and evaluation covers 336 variants in total (Tab. 19), we similarly report success rates at the best validation step, VM and VA settings are evaluated separately.

Since the original paper only reported per-task success rates for CogACT-Base but not for CogACT-Large, we re-evaluated the released CogACT-Large checkpoint in our setup and report those numbers for fairness. For $\pi _ { 0 }$ , results are taken from the open-source reproduction open-pi-zero, which provides implementations with both uniform and beta timestep sampling strategies in flow matching. We report results under float32 precision as in the public release. Note that open-pi-zero does not provide numbers for the Fractal Visual Aggregation setting, and thus these are missing.

# D.2 LIBERO

Evaluation on LIBERO (Liu et al., 2023a) is conducted across all five suites (Spatial, Object, Goal, Long-10, and Long-90). Models are validated every 1k steps, and each task is evaluated with 50 trials. Success rates are reported at the best validation step. Unlike SimplerEnv, no action ensemble strategy is used in our LIBERO experiments.

CogACT results are reproduced using the official codebase for fairness. For $\pi _ { 0 }$ and $\pi _ { 0 }$ -FAST, we adopt reported numbers, noting that both methods leverage additional wrist-camera views and proprioceptive states, while our approach relies solely on a single third-person RGB. Despite this difference in input modalities, our method consistently surpasses, achieving stronger performance without extra sensory inputs.

# D.3 MIKASA-ROBO

For Mikasa-Robo (Cherepanov et al., 2025), we evaluate with 100 episodes per task. Validation is performed every 1k training steps, and success rates are reported at the checkpoint with the best validation performance. As in LIBERO, we do not use any action ensemble strategy in our Mikasa-Robo experiments.

# D.4 REAL-WORLD

Evaluation uses 15-25 trials for General tasks and 10-15 trials for Long-horizon Temporal tasks. For General tasks, Pick Diverse Fruits contains five variants (apple, orange, banana, chili, grape), each evaluated with 5 trials (25 total). All other General tasks are evaluated with 15 trials each, and we report task-level success rates.

For Long-horizon Temporal tasks, Seq. Push Buttons includes three button orders (blue-pink-green, blue-green-pink, green-blue-pink), each tested with 5 trials. All other tasks are evaluated with 10 trials, and step-wise scoring is adopted to capture partial progress. The scoring rules are as follows:

• Seq. Push Buttons: pressing each correct button yields 30, with a bonus of 10 if all three are correct. Loose matching is allowed (slight contact counts as a press).   
• Change Food: lifting and removing the initial food (30), grasping the new food (30), and placing it on the plate (30), with a 10 bonus for full success.   
• Guess Where: grasping the cover (30), covering the block (30), and uncovering it (40).   
• Clean Table & Count: five objects in total. For each object, clearing yields 10 points and pressing the counter yields 10. Small counting errors (incomplete press / one extra press) earn 5; major errors (missed count / multiple extras) earn 0. Empty grasps with clear counting intent incur a 5-point penalty.

• Pick Place Order: carrot, banana, and orange must be picked and placed in sequence. Each correct step earns 30, with a 10 bonus for full completion. Any order violation terminates the attempt.   
• Clean Restaurant Table: five objects in total. Each correctly sorted into trash bin or storage bin scores 20. Misplacement earns 10, and merely lifting without correct placement earns 5.

# E CASE STUDY OF MEMORY RETRIEVAL

To provide a direct view of how the memory mechanism functions, Fig. 10 visualizes the retrieved memory elements and their attention weights on the real-world and simulation tasks. The model consistently attends to past frames that resolve decision-relevant ambiguities absent from the current observation.

In the real-world Change Food task, after the first food item is placed aside, the current frame contains two food items on the table, making it impossible to determine from this single observation which one should be picked next. MemoryVLAtherefore attends strongly to the nearby frames reflecting the recent motion trend, as well as the last decisive frame before the ambiguity arises. In the Shell Game Touch task from Mikasa-Robo Simulation Benchmark, the robot is briefly shown the cube location before it is covered by cups. The model consistently attends to the initial revealing frames, which provide the only reliable cue for identifying the correct cup. These results demonstrate that MemoryVLAretrieves meaningful temporal cues essential for disambiguating the next action, rather than simply recalling redundant visual history.

![](images/23bf558b86eb6133d8acf05b0ea1b8278cd6c7ee402ae3ae56badb3d26bd4178.jpg)

![](images/c32ae229daa272f02af8446a00a5a3a30e2c29d78eab862b8239957157bf51f6.jpg)  
Figure 10: Case study of memory retrieval in real-world and simulated tasks. The figure visualizes the retrieved memory elements and their attention weights on the real-world Change Food task (top) and the simulated Shell Game Touch task in Mikasa-Robo (bottom). In both settings, the model consistently attends to past frames that resolve decision-relevant ambiguities absent from the current observation.

# F DATA LENGTH STATISTICS

Tab. 9 reports the maximum, minimum, median, and average action lengths across all task suites, including SimplerEnv Evaluation (Bridge, Fractal), LIBERO (Spatial/Object/Goal and 10/90 task suites), and both real-world general and temporal tasks. For the real-world tasks, we additionally provide filtered statistics based on a motion-magnitude threshold (translation $> 1$ cm or rotation $>$ 0.4 rad between consecutive frames) to remove frames where the end-effector motion is small.

Table 9: Action Length Statistics across all simulation (SimplerEnv Bridge/Fractal, LIBERO Spatial/Object/Goal, LIBERO-10/90) and real-world (General, Temporal) task suites. For real-world tasks, the “Filtered” versions remove frames whose end-effector motion is negligible (translation $< 1$ cm and rotation $< 0 . 4$ rad).   

<table><tr><td>Task Suite</td><td>Max</td><td>Min</td><td>Median</td><td>Average</td></tr><tr><td>SimplerEnv-Bridge / Fractal</td><td>200</td><td>80</td><td>117</td><td>119</td></tr><tr><td>LIBERO-Spatial / Object / Goal</td><td>270</td><td>75</td><td>131</td><td>130</td></tr><tr><td>LIBERO-10 / 90</td><td>505</td><td>58</td><td>144</td><td>156</td></tr><tr><td>Real-General (Original)</td><td>1575</td><td>281</td><td>575</td><td>575</td></tr><tr><td>Real-General (Filtered)</td><td>213</td><td>40</td><td>81</td><td>84</td></tr><tr><td>Real-Temporal (Original)</td><td>7704</td><td>412</td><td>981</td><td>1672</td></tr><tr><td>Real-Temporal (Filtered)</td><td>902</td><td>72</td><td>236</td><td>288</td></tr></table>

# G ADDITIONAL ABLATION STUDY

For memory length, it is closely tied to the episode length of each benchmark. As shown in Tab. 9, SimplerEnv, LIBERO, and our real-world general tasks have similar episode lengths, whereas realworld temporal tasks are substantially longer, even after heavy frame filtering. This naturally motivates using a larger memory length for temporal tasks. We further conducted an ablation on a representative real-world temporal task (Clean Table & Count). A moderate memory length (256) performs best, as shown in Tab. 10.

We also added an ablation on the number of cognitive tokens. As shown in Tab. 11, increasing the count from 1 to 4 brings no performance gain. Perhaps the single 4096-dim EOS token already captures sufficient semantic information, so adding more tokens does not provide additional benefit.

In addition, we extended the ablations on fusion type (Tab. 12) and consolidation strategy (Tab. 13) to LIBERO-Long-90 and a real-world long-horizon task.

Table 10: Additional ablation on memory length for both real-world and LIBERO-90 tasks.   
(a) Real-World: Clean Table & Count   

<table><tr><td>Memory Length</td><td>Success Rate</td></tr><tr><td>64</td><td>78</td></tr><tr><td>256 (Base)</td><td>84</td></tr><tr><td>512</td><td>81</td></tr></table>

(b) LIBERO-Long-90 Tasks   

<table><tr><td>Memory Length</td><td>Success Rate</td></tr><tr><td>8</td><td>94.2</td></tr><tr><td>16 (Base)</td><td>95.6</td></tr><tr><td>32</td><td>95.6</td></tr></table>

Table 11: Ablation on the Number of Cognitive Tokens. Increasing the number of cognitive tokens from 1 to 4 does not improve performance. A single 4096-dim EOS token already provides sufficient semantic capacity.   

<table><tr><td>Num. Cog Token</td><td>Spoon on Towel</td><td>Carrot on Plate</td><td>Stack Cube</td><td>Eggplant in Basket</td><td>Avg. Success</td></tr><tr><td>1 (Base)</td><td>75.0</td><td>75.0</td><td>37.5</td><td>100.0</td><td>71.9</td></tr><tr><td>4</td><td>79.2</td><td>66.7</td><td>37.5</td><td>95.8</td><td>69.8</td></tr></table>

Table 12: Additional ablation on memory fusion type for both real-world and LIBERO-90.   

<table><tr><td>Fusion Type</td><td>Clean Table &amp; Count (Real-World)</td><td>LIBERO-Long-90 Tasks</td></tr><tr><td>Add</td><td>78</td><td>93.8</td></tr><tr><td>Gate</td><td>84</td><td>95.6</td></tr></table>

Table 13: Additional ablation on memory consolidation for both real-world and LIBERO-90.   
Tab. 14 provides an extended version of Tab. 6 and 7, reporting per-task success rates on SimplerEnv-Bridge for all ablation settings. Gray rows indicate the default configuration.   

<table><tr><td>Consolidation Type</td><td>Clean Table &amp; Count (Real-World)</td><td>LIBERO-Long-90 Tasks</td></tr><tr><td>FIFO</td><td>76</td><td>94.9</td></tr><tr><td>Token Merge</td><td>84</td><td>95.6</td></tr></table>

Table 14: Details of ablation studies. We report average success rates $( \% )$ on SimplerEnv-Bridge when varying five factors: (a) memory type, (b) memory length, (c) memory retrieval, (d) memory fusion, and (e) memory consolidation. Gray rows indicate the default configuration.   

<table><tr><td colspan="2">Method</td><td>Spoon on Towel</td><td>Carrot on Plate</td><td>Stack Cube</td><td>Eggplant in Basket</td><td>Avg. Success</td></tr><tr><td rowspan="3">(a) Memory Type</td><td>Cog. Mem.</td><td>70.8</td><td>58.3</td><td>29.2</td><td>95.8</td><td>63.5</td></tr><tr><td>Per. Mem.</td><td>83.3</td><td>54.2</td><td>20.8</td><td>100.0</td><td>64.6</td></tr><tr><td>Both</td><td>75.0</td><td>75.0</td><td>37.5</td><td>100.0</td><td>71.9</td></tr><tr><td rowspan="3">(b) Memory Length</td><td>4</td><td>79.2</td><td>75.0</td><td>25.0</td><td>91.7</td><td>67.7</td></tr><tr><td>16</td><td>75.0</td><td>75.0</td><td>37.5</td><td>100.0</td><td>71.9</td></tr><tr><td>64</td><td>79.2</td><td>54.2</td><td>37.5</td><td>100.0</td><td>67.7</td></tr><tr><td rowspan="2">(c) Memory Retrieval</td><td>w/o Timestep PE</td><td>83.3</td><td>62.5</td><td>50.0</td><td>83.3</td><td>69.8</td></tr><tr><td>w/ Timestep PE</td><td>75.0</td><td>75.0</td><td>37.5</td><td>100.0</td><td>71.9</td></tr><tr><td rowspan="2">(d) Memory Fusion</td><td>Add</td><td>75.0</td><td>62.5</td><td>33.3</td><td>100.0</td><td>67.7</td></tr><tr><td>Gate</td><td>75.0</td><td>75.0</td><td>37.5</td><td>100.0</td><td>71.9</td></tr><tr><td rowspan="2">(e) Memory Update</td><td>FIFO</td><td>66.7</td><td>66.7</td><td>33.3</td><td>100.0</td><td>66.7</td></tr><tr><td>Token Merge</td><td>75.0</td><td>75.0</td><td>37.5</td><td>100.0</td><td>71.9</td></tr></table>

# H INFERENCE EFFICIENCY

As shown in Tab. 15, latency, throughput, and GPU memory measurements comparing our method with the baseline on two commonly used GPUs, RTX 4090 and HGX H20. The results below are averaged over 300 runs in bfloat16.

Our method achieves a latency of 0.194 s and a throughput of $8 2 . 5 \mathrm { H z }$ on RTX 4090, corresponding to only a $3 . 6 \%$ increase in overhead compared to the baseline. The memory usage is 16.6 GB, which is merely $+ 0 . 8$ GB over the baseline. These results show that the memory module is lightweight and introduces only negligible extra cost during inference.

The overhead remains small because the memory module is deliberately lightweight: each cognitive memory entry is represented by a single cognitive token, the perceptual memory is compressed to 256 channels through a perceptual compression module, and the memory consolidation step continuously merges redundant entries. Together, these designs keep both the retrieved memory size and cross-attention cost minimal, resulting in negligible additional latency and memory usage during inference.

Table 15: Inference efficiency comparison. Latency, throughput, and GPU memory are measured over 300 runs in bfloat16 with action chunk length set to 16 on RTX 4090 and HGX H20 GPU. MemoryVLA introduces only minor overhead compared to the baseline.   

<table><tr><td>Model</td><td>Latency (RTX 4090)</td><td>Throughput (RTX 4090)</td><td>Latency (HGX H20)</td><td>Throughput (HGX H20)</td><td>Memory</td></tr><tr><td>Baseline</td><td>0.187 s</td><td>85.6 Hz</td><td>0.236 s</td><td>67.8 Hz</td><td>15.8 GB</td></tr><tr><td>MemoryVLA</td><td>0.194 s</td><td>82.5 Hz</td><td>0.246 s</td><td>65.0 Hz</td><td>16.6 GB</td></tr></table>

# I ZERO-SHOT TASK GENERALIZATION

In addition to visual OOD tests, we have added task generalization experiments to evaluate zero shot performance on unseen task categories. As shown in Tab. 16, We use Apple To Basket as the base task and test on three unseen tasks: Eggplant To Basket, Blush To Basket, and Apple To Plate. As shown in the table below, our method achieves good zero shot task generalization.

Table 16: Zero-shot task generalization results. Apple To Basket is used as the base task, and evaluation is conducted on three unseen task categories.   

<table><tr><td>Model</td><td>Apple To Basket (Base Task)</td><td>Eggplant To Basket (OOD Task)</td><td>Blush To Basket (OOD Task)</td><td>Apple To Plate (OOD Task)</td><td>Avg. Success (OOD Task)</td></tr><tr><td>MemoryVLA</td><td>4 / 5</td><td>3 / 5</td><td>2 / 5</td><td>2 / 5</td><td>0.47</td></tr></table>

# J COMPARE WITH TEMPORAL CONTEXT VLA BASELINES

As shown in Tab. 17, we have added temporal-context baselines for each benchmark, including Mikasa-Robo (Cherepanov et al., 2025), LIBERO (Liu et al., 2023a), and SimplerEnv (Li et al., 2024b), to provide a fair and comprehensive comparison.

In the Mikasa-Robo benchmark, a memory-dependent manipulation benchmark, no temporalcontext VLA baselines are provided in the official release. To ensure a fair comparison, we therefore reproduced CronusVLA (Li et al., 2025a), a contemporaneous strong VLA model that explicitly leverages historical information by aggregating multi-frame VLM features through a slidingwindow module. This provides Mikasa-Robo with a temporal-context baseline, and MemoryVLAsubstantially outperforms it across all tasks.

For the LIBERO benchmark, we include the main temporal-context VLA baselines commonly used in prior work, including TTF-VLA (Liu et al., 2025a), TraceVLA (Zheng et al., 2024b), 4D-VLA (Zhang et al., 2025a), CronusVLA (Li et al., 2025a), and MAP-VLA (Li et al., 2025b). These methods represent the strongest publicly available temporal-context VLA models that have reported results on LIBERO. MemoryVLAobtains higher average success rates than these baselines.

For the SimplerEnv benchmark, we include the temporal-context VLA baselines reported in prior work, including TraceVLA (Zheng et al., 2024b), RoboVLMs (Liu et al., 2025b), and CronusVLA (Li et al., 2025a). These methods incorporate explicit temporal modeling, and MemoryVLAachieves higher success rates than these baselines.

# K TASK DETAILS

To ensure comprehensive evaluation across simulation and real-world settings, we summarize the task design of each benchmark. We provide task templates, variation types, and the number of variations per task to clarify the diversity and difficulty of evaluation.

Table 17: Comparison with temporal-context VLA methods across diverse benchmarks. MemoryVLA outperforms temporal-context VLA baselines across both memory-focused benchmarks and general manipulation benchmarks, including Mikasa-Robo, SimplerEnv, and LIBERO.   
(a) SimplerEnv Benchmark   

<table><tr><td>Method</td><td>Spoon on Towel</td><td>Carrot on Plate</td><td>Stack Cube</td><td>Eggplant in Basket</td><td>Avg. Success</td></tr><tr><td>TraceVLA (Zheng et al., 2024b)</td><td>12.5</td><td>16.6</td><td>16.6</td><td>65.0</td><td>27.7</td></tr><tr><td>RoboVLMs (Liu et al., 2025b)</td><td>45.8</td><td>20.8</td><td>4.2</td><td>79.2</td><td>37.5</td></tr><tr><td>CronusVLA (Li et al., 2025a)</td><td>66.7</td><td>54.2</td><td>20.8</td><td>100.0</td><td>60.4</td></tr><tr><td>MemoryVLA (Ours)</td><td>75.0</td><td>75.0</td><td>37.5</td><td>100.0</td><td>71.9</td></tr></table>

(b) LIBERO Benchmark   

<table><tr><td>Method</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long-10</td><td>Avg. Success</td></tr><tr><td>TTF-VLA (Liu et al., 2025a)</td><td>73.0</td><td>84.0</td><td>81.0</td><td>58.0</td><td>74.0</td></tr><tr><td>TraceVLA (Zheng et al., 2024b)</td><td>84.6</td><td>85.2</td><td>75.1</td><td>54.1</td><td>74.8</td></tr><tr><td>4D-VLA (Zhang et al., 2025a)</td><td>88.9</td><td>95.2</td><td>90.9</td><td>79.1</td><td>88.6</td></tr><tr><td>CronusVLA (Li et al., 2025a)</td><td>93.8</td><td>92.8</td><td>95.6</td><td>86.5</td><td>92.2</td></tr><tr><td>MAP-VLA (Li et al., 2025b)</td><td>96.3</td><td>98.4</td><td>95.4</td><td>83.4</td><td>93.4</td></tr><tr><td>MemoryVLA (Ours)</td><td>98.4</td><td>98.4</td><td>96.4</td><td>93.4</td><td>96.7</td></tr></table>

(c) Mikasa-Robo Benchmark   

<table><tr><td>Model</td><td>Shell Game Touch</td><td>Intercept Medium</td><td>Remb. Color3</td><td>Remb. Color5</td><td>Remb. Color9</td><td>Avg. Success</td></tr><tr><td>CronusVLA (Li et al., 2025a)</td><td>32</td><td>5</td><td>31</td><td>13</td><td>9</td><td>18.0</td></tr><tr><td>MemoryVLA (Ours)</td><td>88</td><td>24</td><td>44</td><td>30</td><td>20</td><td>41.2</td></tr></table>

Table 18: Real-world tasks details. We list the instruction template, number of variations, and the corresponding variation types for each task.   

<table><tr><td>Task Name</td><td>Language Instruction Template</td><td># Variations</td><td>Variation Type</td></tr><tr><td>Seq Push Buttons</td><td>“Push the {color 1, color 2, and color 3} buttons in sequence”</td><td>3</td><td>Button color order</td></tr><tr><td>Change Food</td><td>“Move food off the plate, then put the other food on it”</td><td>2</td><td>Food object type in plate</td></tr><tr><td>Guess Where</td><td>“Place a cover over the block, then remove the cover”</td><td>1</td><td>-</td></tr><tr><td>Clean Table &amp; Count</td><td>“Clean the table item by item, and push the button after each item cleaned”</td><td>1</td><td>-</td></tr><tr><td>Pick Place Order</td><td>“Pick up carrot, banana and orange in order and place them in the basket”</td><td>7</td><td>Background, distractors, lighting, object, container, occlusion</td></tr><tr><td>Clean Restaurant Table</td><td>“Place all trash into the trash bin and all tableware into the storage bin”</td><td>7</td><td>Background, distractors, lighting, object, container, occlusion</td></tr><tr><td>Insert Circle</td><td>“Insert the circle on the square”</td><td>1</td><td>-</td></tr><tr><td>Egg In Pan</td><td>“Put the egg into the pan”</td><td>1</td><td>-</td></tr><tr><td>Egg In Oven</td><td>“Put the egg into the oven”</td><td>1</td><td>-</td></tr><tr><td>Stack Cup</td><td>“Stack the green cup on the other cup”</td><td>1</td><td>-</td></tr><tr><td>Stack Block</td><td>“Stack the yellow block on the red block”</td><td>1</td><td>-</td></tr><tr><td>Pick Diverse Fruit</td><td>“Pick up {fruit} and place it in the basket”</td><td>5</td><td>Fruit category</td></tr></table>

Table 19: SimplerEnv tasks details. $V M =$ visual-matching, $V A =$ variant-aggregation. For Fractal, we report VM and VA separately; all Bridge tasks use a single setting.

Tab. 18 shows the 12 tasks used in real-world evaluation, divided into General and Long-horizon Temporal suites.   

<table><tr><td></td><td>Task Name</td><td>Language Instruction Template</td><td># Variations</td><td>Variation Type</td></tr><tr><td rowspan="4">Bridge</td><td>Spoon On Tower</td><td>“Put the spoon on the tower”</td><td>1</td><td>-</td></tr><tr><td>Carrot On Plate</td><td>“Put the carrot on the plate”</td><td>1</td><td>-</td></tr><tr><td>Stack Cube</td><td>“Stack the green cube on the yellow cube”</td><td>1</td><td>-</td></tr><tr><td>Eggplant In Basket</td><td>“Put the eggplant in the basket”</td><td>1</td><td>-</td></tr><tr><td rowspan="8">Fractal</td><td rowspan="2">Pick Coke Can</td><td rowspan="2">“Pick up the coke can”</td><td>VM: 12</td><td>VM: can position×3, URDF×4.</td></tr><tr><td>VA: 33</td><td>VA: base, backgrounds×2, lighting×2, textures×2, camera views×2, distractors×2, can position×3.</td></tr><tr><td rowspan="2">Move Near</td><td rowspan="2">“Move the {object} near the {reference}”</td><td>VM: 4</td><td>VM: URDF×4.</td></tr><tr><td>VA: 10</td><td>VA: base, distractors, backgrounds×2, lighting×2, textures×2, camera views×2.</td></tr><tr><td rowspan="2">Open/Close Printer</td><td rowspan="2">“Open/close the {level} printer”</td><td>VM: 216</td><td>VM: env {open/closed}×{top, middle, bottom}; URDF×4; initial pose×9.</td></tr><tr><td>VA: 42</td><td>VA: base, backgrounds×2, lighting×2, cabinet styles×2, env×6.</td></tr><tr><td rowspan="2">Put In Printer</td><td rowspan="2">“Put the {object} into the {level} printer”</td><td>VM: 12</td><td>VM: URDF×4; initial pose×3.</td></tr><tr><td>VA: 7</td><td>VA: base, backgrounds×2, lighting×2, cabinet styles×2.</td></tr></table>

# K.1 REAL-WORLD TASKS

General Tasks. Insert Circle: insert a circle onto a vertical pillar, requiring accurate positioning and insertion. Egg in Pan: place an egg into a shallow frying pan, testing grasp stability and gentle placement. Egg in Oven: put an egg into a small oven container, involving more constrained placement than the pan. Stack Cups: stack one plastic cup on top of another, evaluating vertical alignment and balance. Stack Blocks: stack a yellow block on top of a red block, focusing on precise spatial alignment. Pick Diverse Fruits: pick a specified fruit from a tabletop with more than ten different fruit types and place it into a basket, testing semantic understanding, visual diversity, and instruction following.

Long-horizon Temporal Tasks. Seq. Push Buttons: push three buttons in a specified color sequence, stressing ordered memory and resistance to temporal confusion. Change Food: remove a food item from a plate and replace it with another, requiring multi-step sequencing and correct temporal ordering. Guess Where: cover a block with a container and later uncover it, testing reversible actions and consistent tracking over time. Clean Table & Count: clear items from the table one by one while pressing a counter button after each removal, combining manipulation with explicit progress monitoring. Pick Place Order: pick up carrot, banana, and orange in a fixed order and place them into a basket, enforcing sequence-sensitive planning under temporal dependencies. Clean Restaurant Table: sort table items by category, placing trash into a trash bin and tableware into a storage bin, representing a long-horizon task with semantic reasoning and complex multi-stage sequencing.

# K.2 SIMPLERENV TASKS

Tab. 19 summarizes the tasks in the SimplerEnv benchmark, which consists of two suites: Bridge and Fractal.

Table 20: LIBERO tasks details. We list the language instruction templates and the total number of tasks per suite.   

<table><tr><td>Suite</td><td>Language Instruction Templates</td><td>#Tasks</td></tr><tr><td>Spatial</td><td>pick up the OBJ SPATIAL_REL and place it on the TARGET</td><td>10</td></tr><tr><td>Object</td><td>pick up the FOOD and place it in the CONTAINER</td><td>10</td></tr><tr><td>Goal</td><td>open/close the CONTAINER
open the DRAWER and put the OBJ inside
put the OBJ on/in the TARGET
push the OBJ to the POSITION of the TARGET
turn on the APPLIANCE</td><td>10</td></tr><tr><td>Long-10</td><td>put both OBJ1 and OBJ2 in the CONTAINER
turn on the APPLIANCE and put the OBJ on it
put the OBJ in the CONTAINER/APPLIANCE and close it
place OBJ1 on TARGET1 and OBJ2 on TARGET2/at REL of TARGET2
pick up the OBJ and place it in the caddy COMPARTMENT</td><td>10</td></tr><tr><td>Long-90</td><td>open/close CONTAINER/APPLIANCE [and put OBJ on/in it; optionally sequence with another open/close]
open CONTAINER and put OBJ in it
put/place OBJ in/on/under TARGET or at REL_POS
stack OBJ1 on OBJ2 [optionally place them in CONTAINER]
pick up OBJ and put it in CONTAINER (basket/tray)
place MUG on left/right PLATE or BOOK in caddy/on/under shelf
turn on/off APPLIANCE [optionally put OBJ on it]</td><td>90</td></tr></table>

The Bridge suite contains four tabletop manipulation tasks on WidowX robot: Spoon on Towel, Carrot on Plate, Stack Cube, and Eggplant in Basket. Each task is paired with a single language template, focusing on object placement and stacking primitives.

The Fractal suite builds on RT-1 data with Google robot and defines four tasks: Pick Coke Can, Move Near, Open/Close Drawer, and Put in Drawer. Each task is evaluated under two protocols. Visual Matching (VM) mirrors the real-world setup by varying object positions and URDFs, ensuring alignment between simulation and deployment. Visual Aggregation (VA) introduces substantial visual perturbations, including changes in backgrounds, textures, lighting, distractors, and camera views, to stress-test robustness and generalization. Together, VM and VA yield 336 variants, producing 2,352 evaluation trials.

# K.3 LIBERO TASKS

Tab. 20 outlines the five suites of the LIBERO benchmark: Spatial, Object, Goal, Long-10, and Long-90. LIBERO-Spatial consists of tasks where the same object must be placed across varying target positions. LIBERO-Object focuses on handling diverse objects within a fixed scene layout. LIBERO-Goal contains heterogeneous operations such as opening containers, placing objects, or turning on appliances, performed in an unchanged environment. Long-10 introduces ten extended tasks that require multiple sub-goals across different scenes, while Long-90 expands this setting to ninety tasks, providing a substantially more challenging benchmark. In total, LIBERO offers 130 tasks in simulation with a Franka robot.

# L VISUALIZATION OF MEMORY DEPENDENCE IN REAL-WORLD TASKS

To clarify why real-world temporal tasks require memory, Fig. 11 highlights the key moments in several tasks where the correct action depends on past information rather than the current observation. Fig. 12 further provides a step-by-step example from the Change Food task, showing a case where the next action becomes ambiguous from a single frame and can only be resolved by recalling

earlier steps. These examples illustrate that the stronger gains on real-robot tasks stem from their inherently memory-dependent nature.

![](images/30cc166bb79855c2dfda8577318134d127fc75347615bcf29e007b7ce13942d3.jpg)  
(a) Clean Table & Count

![](images/25f444804f2ef927206a10600f900c524c77ffdf3499218ecb5909935e674af6.jpg)

![](images/6dd77ab0e26f0c46aef94d97715d6db43d9929bb428861a89694ba2a58b0a61e.jpg)

![](images/e8424c7d1bd2c08763acfe7e6648e565f9b5a9009250e54bf7023d57e3b32409.jpg)

Have I pressed it before?

![](images/f65f6ce6d951c50518f4fdc5947789276266fb9c2b7d5bcee71f1d17f1111ba7.jpg)

![](images/98810f3964e622791ed39de243f24ec1404b0203e996291ed1c8a6b9af7fd079.jpg)

![](images/94ba1fb809609221c445fbbe67bd5e530ef88b96c3b850f03dd142bddaafbd1d.jpg)

![](images/8e417131a8ae1de91178cec32fe665a28b935a6015a9c7244ff5d6f98e4ae92c.jpg)  
?

![](images/8099e256f7dd90561905aeb3ec9f9efd1e7184da66f3ea92a0949833b4c1df04.jpg)

![](images/17fa5cbb6028ee29af2cec11ce444db2f6486fce35f8b081e4bc2d0cbf5fa2d6.jpg)

![](images/35f33d05a67f0fc05468910b87be60598faad04155c8e7374b3094b6c642406c.jpg)

![](images/81a7237cd463cb8e565f60592e20d735f8865f461aa35af4611bf30ef22ce6b1.jpg)

![](images/7f3f731a15bf95826c05ae55f6137bd45d543f39f5bc2347ee1932da9c9b4f86.jpg)  
Which cover is the block under? block under?   
(b) Guess Where

![](images/82f057d44d7a79e5a01bbb6777d0ad2f5dcf49b60eec9960d792ea05c7069f3c.jpg)

![](images/048c540398d6dfa5a3adbdf684e122e2318fba0aa0dabd10cdf16215186c793f.jpg)

![](images/e418bd2ddbd2ef1c963813532f833970ff2f3a20269d535521a83cbd75e6d67f.jpg)  
Place a cover over the block, then remove the cover

![](images/88121853743aaf59f1765c93e4631818663d6a2f35db9a52a751627a691a3dcc.jpg)  
(c) Change Food   
What food did I just pick & place? bick&place? Which food should I pick now? picknow?

![](images/5d6a62a422f7d3bb0d46b584ea392f955439ea5a49479348d06e7016617d1e58.jpg)

![](images/bcc1354f84876fa77d2d8e7b2366cca3dadf0b634a956ec1b92a5c978f40fde6.jpg)

![](images/00d0478dd5d09af3401c0729be9db3da1f54d3c5eea63a698be3adec40e0685d.jpg)

![](images/8272a85bae28ffbf573ae5bb4ccabbd0c5813d2de4fd7da1f363b07dbe8fa593.jpg)  
Move food off the plate, then put the other food on it

![](images/2da69adeb80d7a036fdd12763fe509b318312b3ff321f0890b1a7a0209526eb4.jpg)  
Figure 11: Memory-dependent aspects of three real-world robotic tasks. The tasks Clean Table & Count, Guess Where, and Change Food all require tracking past events to make correct decisions, highlighting their inherent temporal dependence.

![](images/8122ca11122de3a97179ec64e63c692cbfd409aebeaab1ad50c3fe0179ed25f4.jpg)

![](images/63533ea84abe281c60af82eb1d59caf4511e3a388b44398b5c81979c1668728f.jpg)  
Just begun, or almost over?

![](images/cb716f3190e67a1bc60aa7f1d55dbf1ac379bb426bd119b5929dd34a3eceda27.jpg)

![](images/8f44a793113dd6aa33dcc009cfca05a7ce9f6440d7a3d3b412d26be1598d9831.jpg)  
Take away or place it on plate?

![](images/538972943ebced005d88ac94169bc1932c0c8e944acbd2aff4d46977d8bb2fa6.jpg)

![](images/4257c28115789c38af0309accc8928caf2c3a26b032f1108f6d76e95751111ed.jpg)  
Place corn or pick corn?

![](images/6512d4ce6e0403ee15741ae10281fecfb893cd3b8e666ce65fec2913ed455a66.jpg)

![](images/aeb5c8f04843d830529734fef620779546275eb1ea716f2d98c66934d03ae211.jpg)  
Move left or right? What did I just pick

![](images/1f375706e9e3c96f86792c290c61d2e57123d3ff9f75ac085367bd8e793d244b.jpg)  
  
Pick Another

![](images/5029b2ea52b388fadce6db3054aeed519e976dc70846a222104deb92a80a6550.jpg)

![](images/a6049e71c474ad8b18c6cefcfc4b6dfe82ea13e8901b8e3ae6797b9a527cb7a6.jpg)  
Same confusion like “Take Away”

![](images/9530b3774d6a326f0efa5e043aadfa25daa2196be3b612ab0892c9cce8ed31f2.jpg)

![](images/3e9bbdb027c66abdb2a42c99d0b37f3b698cb6ab4e8b44e898adf638fe2d5670.jpg)

![](images/3fd3a57d957aaf70f726403ad97a6ccc6a580c202814a2807c684e9375604e52.jpg)  
Same confusion like “Pick Up”

![](images/bca6b9b903020e8050a73007d89754ccbe0e7214b8c1602470b477935e8a89a9.jpg)

![](images/d7083f2f53f8fea6cfbd7bd407d0be3cb6498bebfc664f2e166063e453926ccf.jpg)  
Same confusion like “Ready GO”

![](images/2511f14d7e2a8bc2b4896690cf4de17700aa6aa7573a72868da4bc1ef55d08cf.jpg)  
Language Prompt   
Move food off the plate, then put the other food on it   
Initial States in Training Data

![](images/f7b7cebd2d7ce1ef4ec2358c2bc05a51251b7d287b95789ff9a4be42e05ded6e.jpg)

![](images/98339847912869386fbe27d692e1ee0204c95ae5691b96420086efbeceea68e8.jpg)

![](images/3cf91b9d51f77f84b977cf43283703f8e1ddb2f55126ff64a97a431c64954a4c.jpg)  
  
Figure 12: Step-by-step example of memory-dependent behavior. A sequence from the realworld Change Food task showing a case where the next action becomes ambiguous without recalling earlier steps.

# M QUALITATIVE RESULTS

# M.1 REAL-WORLD EVALUATION

We present qualitative examples to complement the quantitative evaluation. Fig. 13 and 14 illustrate rollouts on long-horizon temporal tasks in real-world. Fig. 15 shows general manipulation tasks in real-world.

![](images/3f843ff5ffadeab3a549e094c6165f8028034f9718e78d42daedf406ac204b33.jpg)  
(a) Seq Push Buttons: push the blue, green, and pink buttons in sequence

![](images/8b30c3d3f455f2d6522efc26bca847157d83f9ee4e40757f5e01b98c2b69ff19.jpg)  
(b) Change Food: move food off the plate, then put the other food on it

![](images/dc56b0d224eaa7c5a3664964e698605a6d7225fe9b21469f3f4ef52ca3447d3a.jpg)  
(c) Guess Where: place a cover over the block, then remove the cover   
Figure 13: Qualitative results of MemoryVLA on real-world long-horizon temporal tasks (I). Representative examples include Seq Push Buttons, Change Food, and Guess Where tasks.

![](images/629a484ccf01eac7e16855a19238421baafb8b0c0b43038cb7d12c7745596a33.jpg)  
(d) Clean Table & Count: clean the table one by one, and press the button for each item cleaned

![](images/d14cda7ad47081278b1ccd02cf1188a1a7565d5f21027d498a2ab3220925cf64.jpg)  
(e) Pick Place Order: pick up carrot, banana, and orange in order and place them in the basket

![](images/56b9e1c8b90ee0478526287c96bcf4f3bda2db6e5a89f6a5e0dbbe356b5830c0.jpg)  
(f) Clean Restaurant Table: place all trash items into the trash bin and all tableware into the storage bin   
Figure 14: Qualitative results of MemoryVLA on real-world long-horizon temporal tasks (II). Representative examples include Clean Table & Count, Pick Place Order, and Clean Restaurant Table tasks.

# M.2 SIMULATION EVALUATION

Results on simulated environments are visualized in Fig. 16 and 17, covering both Bridge and Fractal suites. Finally, Fig. 18 provides representative trajectories on LIBERO, spanning all five suites.

![](images/a06e0518365254bfd0f1a488b63410e6525be474b44f14b2b31ee35ab5ffedec.jpg)

![](images/75b0affc91aceb226d4d9a9c9fc8ed43dc3ec7350266eac5af9781f15977bebd.jpg)  
(a) Insert Circle: insert the circle on the square   
(b) Egg In Pan: put egg in pan

![](images/d6de03e9d2dfb52e354a8e50ef2e9bbf75175c62e03b7bccd9dc926d0f6f51ca.jpg)

![](images/ff0cfd05f0254582ff0d63494b7da945c601ad1ad9a868c213fdfdf2fb3d8e72.jpg)  
(c) Egg In Oven: put egg in oven   
(d) Stack Cup: stack the green cup on the other cup

![](images/37fccdd2c1a0fa59cd3b6bb1fcc16007c94fd8bfb4e14e7437c7a67421fb01c6.jpg)  
(e) Stack Block: stack the yellow block on the red block

![](images/cb0835d3b5cebcb5691536a3e2d350256a355da5db05dec9ad532321207a8688.jpg)  
(f) Pick Diverse Fruit: pick up the apple and place it in the basket   
Figure 15: Qualitative results of MemoryVLA on real-world general tasks. Representative examples include Insert Circle, Egg in Pan, Egg in Oven, Stack Cups, Stack Blocks, and Pick Diverse Fruits tasks.

![](images/ca3f1690660e406cef63a3184f16b1b6b146b267648d907999d46793a946a9b0.jpg)  
(a) Spoon On Tower: put the spoon on the tower

![](images/6aa9c7bfefdc174e594010bf2d0879d68725dbfb25bea9c61368a0cdecec1ac2.jpg)  
(b) Carrot On Plate: put the carrot on the plate

![](images/18eb467463c40b614f361598c85a2e1ea6849f68a9aa08ab202af2d9c494c8ef.jpg)  
(c) Stack Cube: stack the green cube on the yellow cube

![](images/4cc57b5d51130104fc1d3dca980151662b64d08d6e008e1c7b2668b43260ebb6.jpg)  
(d) Eggplant In Basket: put the eggplant in the basket   
Figure 16: Qualitative results of MemoryVLA on SimplerEnv-Bridge tasks. Representative examples include Spoon on Tower, Carrot on Plate, Stack Cube, and Eggplant in Basket tasks.

![](images/5b83a65d2f08318075f32692ceff20427d1ba5424a6533226d44a4b1a161391d.jpg)

![](images/208fc39f3704fc77ee3315974c6cef237384363e4489dc9fc03f31ec963b4adf.jpg)

![](images/d36ecb3f00f87a6f1d5f14db85533304ec4e52067f730c828c08841201bf1646.jpg)  
(a) Coke Can: pick up the coke can

![](images/e0713ab20675bb8fa77069c2947f69f136186b37c36a609affcebaf62604c8a6.jpg)

![](images/6ddf5753790c8ea96983b6aa1d002ad0cd4e13c1a15d57b17a1333a2063daa52.jpg)  
(b) Move Near: move the orange near the Pepsi

![](images/63a05e11e2832c17fda6a986c8e1e87994fb1fe55bdb079f7f215b28659b9363.jpg)

![](images/2cbd2e4172fdf831a9031ae9ec24f921df0e7905c7602b45a59d78d4dfdbb87e.jpg)  
(c) Open / Close Drawer: open / close the top drawer   
（d)Put In Drawer: put the apple into the top drawer   
Figure 17: Qualitative results of MemoryVLA on SimplerEnv-Fractal tasks. Representative examples include Pick Coke Can, Move Near, Open/Close Drawer, and Put in Drawer tasks.

![](images/561955e3eadb922e4aecdb08b2c0fff90eab1d3520bf936e55a94fb578a7b255.jpg)

![](images/5d42ade1f4d76d3b5815ca039b6e327169477bcdc31534a4c568417cf4c58dc4.jpg)  
(a) Spatial: pick up the black bowl on the wooden cabinet and place it on the plate   
(b) Object: pick up the tomato sauce and place it in the basket

![](images/3ff0d3d84c47521e438770dab7585b1197b85799fc6d1de144ce2311170e03a7.jpg)  
(c) Goal: open the top drawer and put the bowl inside

![](images/5df5e24c331e6c4541bbebdb910cfb9e8cd20e4f4378ae4bc120ccaaa34cb7b8.jpg)  
(d) Long: turn on the stove and put the moka pot on it

![](images/9e70a7d845670704db4a78141e04f0e27e141b60177ceb0cdd72d8b8d70c5b6d.jpg)  
(e) LIBERO-90: close the microwave   
Figure 18: Qualitative results of MemoryVLA on LIBERO tasks. Representative examples include tasks from Spatial, Object, Goal, Long-10, and Long-90 suites.
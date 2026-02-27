# MemVerse: Multimodal Memory for Lifelong Learning Agents

Junming Liu*, Yifei Sun*, Weihua Cheng, Haodong Lei, Yirong Chen, Licheng Wen, Xuemeng Yang, Daocheng Fu, Pinlong Cai, Nianchen Deng, Yi Yu, Shuyue Hu, Botian Shi, Ding Wang† Shanghai Artificial Intelligence Laboratory

liujunming@pjlab.org.cn sunyifei@pjlab.org.cn wangding@pjlab.org.cn

https://github.com/KnowledgeXLab/MemVerse

![](images/a8478c2273509e0c49a8f82b75c94bb7153b00032257b311191702bcb5f3a387.jpg)

![](images/a6bfe3f2fc1c9d0db59bb90947dfcf0cd0b443c573a0e8f3c86f331e1b5328e8.jpg)  
(a) Auditory Memory   
(c) Video Memory

![](images/4bc2904378e44e353b0e51704bdb6bf496791e832f36487c4df5533411c59477.jpg)  
(b) Image Memory

In Pirates of the Caribbean, how would you describe Captain Jack Sparrow and what makes him stand out among other pirates? Jack Sparrow is a clever and unpredictable pirate who always manages to turn trouble into opportunity. With his quick wit and strange sense of charm, he escapes every...

What is a classic line from Pirates of the Caribbean that best captures its adventurous spirit?

![](images/da3d4a368875d5ad79bdd840a7bd24231e1523690d21e5181e834c382a02544a.jpg)

One of Jack Sparrow’s most famous quotes is, "Not all treasure is silver and gold, mate." It reminds us that the real rewards in life...

![](images/346a6175a55f6a9bbbbcdc8b64d53a449857a51bd02e18800ceaab7246f70c44.jpg)  
(d) Text Memory

Retrieve Memories from MemVerse... Retrieve Memories from MemVerse... Accessing Short-term Memory... Accomplished! Accessing Short-tem Memory.. Accomplished! Accessing Paramatric Memory... Accomplished! AccessingParamatric Memory.Accomplished! Accessing Long-term Memory... Accomplished! Accomplished! se A fluffy British Shorthair kitten with A fluffy British Shorthair kiten with a light gray and white coat, wearing light gray and white coat, wearing round glasses, who belongs to Mia. √ Mia is the proud owner of the British Shorthair kitten Mia is the proud owner of the British Shorthair kitten wearing those stylish round glasses. She found him at wearing those stylish round glasses. She found him at a cozy little pet shop on a rainy afternoon, instantly draw cozy little pet shop on a rainy afternoon, instantly drawn to his fluffy gray fur and wise little eyes. Ever since, the to his fluffy gray fur and wise little eyes. Ever since, the kitten has been Mia’s constant companion, lounging on kitten has been Mia’s constant companion, Iounging on her desk while she works, occasionally swiping at the her desk while she works, occasionally swiping at the tassels on her notebook, all while looking effortlessly tassels on her notebook, all while loking effortesly sophisticated in his signature glasses. sonhisticated in his sionature glasses

# Without Memory

Who owns the British Shorthair kitten that’s wearing glasses?

![](images/eb9b6822f024b224c7d7c9de917973fda76cee24dcc0d8055c407c0b92eb9c90.jpg)

![](images/9963c5618c1066f34da9df5b76b851c9c2aed3a621719336ff39b54932dab93b.jpg)

![](images/9653615b04bf88ac5af3ae420090f2f389f5e4f2254f13a5c2c41fe8931e9fcf.jpg)

Who owns Truffles?She was rescued by and adopted by an optician named Danielle Crull, who runs a children’s eyewear practice in Pennsylvania.

# With MemVerse

Who owns the British Shorthair kitten that’s wearing glasses?

![](images/05145b21af49d7ca49b7113fd063a467a5b510c97c364ef66007621fa3e06a6f.jpg)

![](images/42919a46cbb7089dc937f6c40c14c654187f9fd2ca97eb8778fbfa0bf0f1a20d.jpg)

![](images/50971f3771cd33f5aa7fd691768d0d0b16e22d932d3ce13c8f4ec0a95e3b942d.jpg)

![](images/a75b4d30571157b36dea188a9b78180e9b93f6d2fcd4f0cf6e84482d0f049f65.jpg)

![](images/2398b14b884b573d31dec23dbdd3b71d8b96f760ffcede4e43d1274ee68ce33a.jpg)

![](images/66c32855299bf5a008308fe2bce7583329ee197f3c2fb318491504de52f0eb1b.jpg)

![](images/aad5eca727bc900bc8617630a3d39490604d3de282ffaf9dd9b39971439d76fe.jpg)

![](images/79c5dc5a6da0efdbfc2547c64e4b653d50e4552daabe10ba9dfdf84bf27cddc0.jpg)  
Figure 1. (1) The left panels (a-d) illustrate how MemVerse equips agents with multimodal memories such as auditory, visual, video, and textual modalities that evolve, interact, and enrich contextual understanding. (2) The right panels compare two reasoning outcomes: a baseline LLM without memory (top) tends to hallucinate and lose context, whereas with MemVerse plugged in (bottom), the model can retrieve grounded multimodal evidence and generate accurate, context-aware answers.

# Abstract

Despite rapid progress in large-scale language and vision models, AI agents still suffer from a fundamental limitation: they cannot remember. Without reliable memory, agents catastrophically forget past experiences, struggle with longhorizon reasoning, and fail to operate coherently in multimodal or interactive environments. We introduce Mem-Verse, a model-agnostic, plug-and-play memory framework that bridges fast parametric recall with hierarchical retrieval-based memory, enabling scalable and adaptive multimodal intelligence. MemVerse maintains short-

term memory for recent context while transforming raw multimodal experiences into structured long-term memories organized as hierarchical knowledge graphs. This design supports continual consolidation, adaptive forgetting, and bounded memory growth. To handle real-time demands, MemVerse introduces a periodic distillation mechanism that compresses essential knowledge from long-term memory into the parametric model, allowing fast, differentiable recall while preserving interpretability. Extensive experiments demonstrate that MemVerse significantly improves multimodal reasoning and continual learning efficiency, empowering agents to remember, adapt, and reason coherently across extended interactions.

# 1. Introduction

Despite remarkable progress in large-scale language and vision models, AI agents remain fundamentally constrained by their lack of memory [63], particularly in visual reasoning [9], robotic interaction [3], and multimodal scene understanding [60]. Memory is essential for enabling agents to adapt to dynamic environments, integrate new experiences with prior knowledge, and perform long-horizon reasoning. Unlike humans, whose intelligence emerges from the continuous accumulation and abstraction of multimodal experiences [7], current AI agents operate in a largely stateless manner, treating each task as independent [22, 42]. Consequently, developing a robust mechanism to organize multimodal experiences is a fundamental step in agent design, especially as AI applications shift from single-task inference to continuous interaction in real-world settings.

Current memory solutions for AI agents fall into two dominant paradigms, both inadequate for multimodal [6, 42, 47] and lifelong learning [10, 45] scenarios. The first is parameter-embedded memory, where knowledge is encoded directly into model weights through fine-tuning, prompt tuning, or in-context learning [34, 39]. While this approach leverages the model’s inherent capacity for compression and associative recall, it suffers from rigidity: memory capacity is bounded by fixed parameters, updates require costly retraining, and newly acquired information often interferes with existing knowledge. Moreover, such parameter-embedded memory is inherently a black box that limits interpretability and control. The second paradigm relies on external static storage, typically implemented through Retrieval-Augmented Generation (RAG)-style systems that maintain raw interaction logs or documents [23]. Memory access is handled via embedding-based retrieval [12] or keyword matching [20], but the absence of structured abstraction leads to redundancy and inefficiency. As the database grows, retrieval becomes increasingly noisy and computationally expensive, and the agent struggles to distill or generalize from accumulated experiences.

These limitations give rise to three critical and interconnected challenges that constrain the development of capable AI agents in multimodal settings. First, memory needs to be decoupled from model parameters. Parameter-embedded memory is tightly bound to model weights, making it difficult to scale or adapt [19, 52]. Agents cannot expand their memory capacity without altering the underlying model, nor can they learn new tasks without the risk of catastrophic forgetting. This rigidity fundamentally impedes lifelong learning, where agents must continuously assimilate new multimodal experiences. Second, static external memory requires structured organization and abstraction. RAG-style external storage typically logs raw interactions without abstraction, failing to transform linear, redundant data into structured, actionable knowledge [31]. Without

mechanisms for hierarchical organization, adaptive pruning, or semantic abstraction, agents incur escalating computational costs and retrieval inefficiencies. Third, memory must support multimodal reasoning. Most existing memory systems remain text-centric, overlooking the inherently multimodal nature of real-world experience [50, 57]. The inability to align or associate information across visual, linguistic, and other sensory modalities results in flat memories that lack contextual grounding and cross-modal reasoning capability—ultimately constraining agent performance in perception-rich environments.

To overcome these limitations, we introduce Mem-Verse, a model-agnostic memory framework designed for AI agents engaged in multimodal reasoning and lifelong learning. MemVerse provides a unified interface that integrates hierarchical retrieval-based long-term memory with a lightweight parametric memory model, enabling scalable and context-aware reasoning. Drawing inspiration from the complementary roles of fast and slow thinking in biological cognition [14], the framework employs a dual-path architecture. The fast pathway, implemented as the parametric memory model, provides immediate, differentiable recall by periodically distilling essential knowledge from long-term memory. The slow pathway, realized as hierarchical retrieval-based memory, continuously accumulates, abstracts, and organizes multimodal experiences into specialized memory types structured as hierarchical knowledge graphs. These include core memory, which stores userspecific knowledge; episodic memory, which logs detailed, time-ordered experiences; and semantic memory, which maintains structured, generalizable knowledge about entities and their relationships. This design allows MemVerse to consolidate knowledge efficiently and adaptively forget irrelevant information.

In summary, our work introduces MemVerse, a unified memory framework that bridges parametric and nonparametric mechanisms for multimodal agents. The key contributions are as follows:

• MemVerse is a model-agnostic, plug-and-play memory system that integrates hierarchical retrieval-based longterm memory with a lightweight parametric memory model, providing a unified interface for lifelong multimodal reasoning.   
• MemVerse transforms raw multimodal experiences into specialized memory types that are organized as hierarchical knowledge graphs. This enables continual abstraction, adaptive forgetting, and bounded memory growth, turning unstructured histories into interpretable knowledge.   
• We introduce a periodic distillation mechanism that compresses essential long-term memory into a lightweight parametric model, enabling fast, differentiable recall while preserving transparency and controllability.

# 2. Related Work

# 2.1. Memory for LLM Agents.

Memory is a fundamental module in LLM-based agents, shaping their ability to adapt, generalize, and perform across complex tasks [62]. It is commonly classified into parametric and non-parametric paradigms. Parametric memory embeds knowledge directly into model parameters, either via trajectory-based fine-tuning, as in Fire-Act [2], modular fine-tuning of separate components, as in AgentLumos [58], or through latent memory pools within transformers, as in MemoryLLM [44], enabling models to retain and update knowledge post-training. Hybrid approaches combine parametric mechanisms with algorithmic or latent memory structures to handle long contexts, such as MemAgent [59], which retains high-reward episodes via multi-conversation reinforcement optimization, and Mem-Gen [61], which employs generative latent memory networks. Non-parametric memory decouples storage from the model, using external databases or memory services to provide persistent, adaptable context; examples include MemGPT [33], MemoryBank [65], and MemoRAG [35], which implement tiered retrieval, temporal relevance, and dual-system retrieval to maintain personalized and accurate long-term memory. Production-grade memory layers, such as Mem0 [6] and SuperMemory [38], provide multilevel summarization, compression, and fast read/write operations for scalable retrieval-augmented deployments. Collectively, these works illustrate the evolution of agent memory from static parametric encoding to dynamic, retrievalaugmented, and generative mechanisms, while highlighting remaining challenges in real-time adaptation and rapid task switching.

# 2.2. Multimodal Knowledge Retrieval

Memory and retrieval are essential for knowledge-intensive tasks, particularly in multimodal settings. Dense retrieval methods, such as Dense Passage Retrieval (DPR) [15], learn vector representations that often outperform sparse retrieval in open-domain question answering. Retrieval-Augmented Generation (RAG) [16] integrates parametric LLMs with non-parametric retrieval to improve factual accuracy and adaptivity. For multimodal retrieval, contrastive imagetext pretraining methods, such as CLIP [36], and efficient vision-language models, such as BLIP and BLIP-2 [17], enable scalable cross-modal representations for image-text retrieval and grounding. At a higher structural level, multimodal knowledge graphs organize heterogeneous signals into nodes and relations, supporting cross-modal reasoning and retrieval. Surveys and construction methods outline pipelines, verification strategies, and applications for these graphs [5, 66]. Integrating multimodal retrieval with LLM memory and multi-agent coordination allows agents

to perform context-aware, knowledge-rich reasoning across modalities.

# 3. Methodology

In this section, we describe the technical design of Mem-Verse, illustrated in Figure 2. At the center of the architecture is the memory orchestrator, which governs all interactions between the hierarchical retrieval-based memory and the parametric memory. The orchestrator operates through rule-based control logic without introducing additional trainable parameters, and executes all memory operations, including addition, update, deletion, and retrieval, under a unified interface.

The hierarchical retrieval-based memory consists of short-term and long-term memory components. Short-term memory stores recent context and supports local consistency during ongoing interactions. Long-term memory provides persistent knowledge storage and is structured as knowledge graphs. Parallel to the retrieval-based memory, the parametric memory offers a fast, differentiable pathway for knowledge recall and generalization. It maintains compact neural representations that are periodically distilled from the long-term memory, enabling rapid access to essential information without relying solely on retrieval-based search.

# 3.1. Hierarchical Retrival-based Memory

Multimodal Processing. To handle arbitrary multimodal inputs, such as images, videos, and audio, we first employ pretrained MLLMs to convert the raw data into textual representations. Specifically, each input modality $M$ is encoded into a sequence of descriptive tokens:

$$
S = \mathcal {D} _ {\text {t e x t}} \left(\mathcal {A} \left(\mathcal {E} _ {\text {m o d}} (M)\right)\right), \tag {1}
$$

where $M$ denotes the input modality, ${ \mathcal { E } } _ { \mathrm { m o d } }$ is the modalityspecific encoder that extracts features from $M$ , $\mathcal { A }$ performs cross-modal alignment or interaction, and $\mathcal { D } _ { \mathrm { t e x t } }$ generates the textual tokens.

The resulting text $S$ serves as a text chunk corresponding to the original multimodal input $M$ . This explicit alignment allows any symbolic entity or relation derived from $S$ to be directly linked back to the original multimodal data [22].

Short-Term Memory. To avoid redundant retrieval and continuous updates to long-term storage, we introduce a short-term memory mechanism that caches recent interaction states. Specifically, given a dialogue sequence $\{ q _ { 1 } , q _ { 2 } , . . . , q _ { t } \}$ , STM retains the most recent $K$ queries within a sliding window:

$$
\mathcal {M} _ {\mathrm {S T M}} = \left\{q _ {t - K + 1}, q _ {t - K + 2}, \dots , q _ {t} \right\}. \tag {2}
$$

![](images/5ff6875031499c33a99a81cc2af095f9e3b59cb3134d72da5ca88fa459db823a.jpg)  
Figure 2. MemVerse integrates three memory components: short-term memory for recent conversational contexts, long-term memory structured as a multimodal knowledge graph with entities and semantic relationships, and parametric memory as a lightweight neural model for fast context encoding. A central memory orchestrator manages retrieval and storage across these components, enabling the agent to process multimodal inputs and support lifelong learning.

Since the contextual information in a short dialogue session is already captured within this window, frequent updates to the external memory or implicit models are unnecessary. Instead, consolidation is performed periodically or when sufficient new knowledge accumulates, ensuring both efficiency and stability.

Long-Term Memory. We implement LTM as a paired structure ${ \mathcal { M } } = ( \{ { \mathcal { G } } _ { k } \} , { \mathcal { C } } )$ , where each $\mathcal { G } _ { k } = ( \nu _ { k } , \mathcal { R } _ { k } )$ is a knowledge graph corresponding to a specific type of memory (core, episodic, or semantic), and $\mathcal { C }$ stores the original dialogue text chunks:

$$
\mathcal {M} = \left(\left\{\mathcal {G} _ {k} \right\}, \mathcal {C}\right), \quad k \in \{\text {c o r e , e p i s o d i c , s e m a n t i c} \}. \tag {3}
$$

Core memory stores durable, user-specific facts and preferences to support personalization; episodic memory captures time-ordered, event-based interactions as detailed entries; and semantic memory maintains abstract, generalizable knowledge about concepts and objects. Together, these subgraphs allow the system to accumulate, organize, and refine knowledge over extended time horizons.

Directly storing raw dialogue text as memory is both inefficient and prone to retrieval errors. The number of text chunks can grow rapidly, and simple vector-based retrieval may fail to capture complex multi-hop relations or the topological structure inherent in the interactions. To address this, we first compress the original dialogue text $\mathcal { C }$ into salient memory descriptions using an LLM, capturing

only the most essential information, such as key entities, relations, and critical context. These memory descriptions are then structured into a Multimodal Knowledge Graph (MMKG), which preserves relational connectivity and enables efficient multi-hop reasoning.

Formally, given textual chunks $\mathcal { C } = \{ c _ { i } \}$ , the LLM extracts entities and typed relations to construct the knowledge graph:

$$
\mathcal {G} = \Phi_ {\mathrm {L L M}} (\mathcal {C}) = (\mathcal {V}, \mathcal {R}), \tag {4}
$$

where each node $v \in \mathcal V$ and each relation $r \in \mathcal { R }$ maintain persistent references to their supporting text chunks:

$$
\ell_ {v}: \mathcal {V} \rightarrow \mathcal {P} (\mathcal {C}), \quad \ell_ {v} (v) = \left\{\text {C h u n k s s u p p o r t i n g} v \right\}, \tag {5}
$$

$$
\ell_ {r}: \mathcal {R} \rightarrow \mathcal {P} (\mathcal {C}), \quad \ell_ {r} (r) = \left\{\text {C h u n k s s u p p o r t i n g} r \right\}. \tag {6}
$$

Activating an entity or relation in $\mathcal { G }$ simultaneously triggers the corresponding original dialogue text $\mathcal { C }$ and any associated multimodal data $\mathcal { M }$ , ensuring that both symbolic and perceptual knowledge remain accessible and grounded. This design compresses raw memory into a compact, graphstructured form while preserving traceability and enabling rich relational reasoning.

# 3.2. Parametric Memory

Parametric memory is stored as weights in a lightweight language model, and the memory in the LTM can be updated through periodic supervised fine-tuning. This parameterized model aims to alleviate the significant computa-

tional and storage overhead introduced during RAG inference by mimicking the behavior of nonparametric retrievers and enabling plug-and-play domain memory adaptation through context.

Memory Encoding. The parametric memory module is realized as a small-scale language model $\mathcal { M } _ { \mathrm { L L M } }$ , which stores internalized knowledge within its parameters. Unlike the explicit knowledge graph that preserves symbolic relations, the parametric memory encodes knowledge in a distributed manner through gradient-based parameter updates. The model learns to simulate the retrieval process by directly generating the retrieved answer from a given question. For each training instance consisting of a question $q$ and its corresponding retrieved context $\mathcal { R }$ , the model produces an answer $\hat { \mathcal { R } }$ as:

$$
\hat {\mathcal {R}} = \mathcal {M} _ {\mathrm {L L M}} (q, \mathcal {R}), \tag {7}
$$

where $\mathcal { R }$ is obtained from the explicit knowledge graph via a retrieval-augmented generation (RAG) module. In practice, we construct supervision pairs $( q , \mathcal { R } )$ such that the model learns to internalize retrieved knowledge as part of its parametric representation.

Memory Initialization. The parametric memory is initialized with the original pretrained parameters of the base language model, denoted as:

$$
\mathcal {M} _ {\text {p a r a m e t r i c}} ^ {0} = \Theta_ {\text {p r e t r a i n e d}}, \tag {8}
$$

where Θpretrained represents the original model weights .The base model $f _ { \mathrm { b a s e } }$ can be instantiated from any pretrained multimodal or language model (e.g., LLM or VLM), and our implementation optionally adopts a 7B-scale transformer for efficiency. At this stage, the model has no taskspecific memory. The initialization provides a neutral state from which subsequent fine-tuning gradually embeds retrieved knowledge into the parameters.

Supervised Fine-tuning. To enable the model to internalize retrieved knowledge, we perform supervised fine-tuning on the constructed dataset of $( \boldsymbol { q } , \mathcal { R } , \hat { \mathcal { R } } )$ triplets. Each training example consists of a question $q$ , multiple-choice context, and a retrieved answer $\mathcal { R }$ from the explicit memory. The model is optimized to generate $\mathcal { R }$ given $q$ , effectively learning the retrieval process itself. The training objective minimizes the token-level cross-entropy loss between the generated sequence $\hat { \mathcal { R } }$ and the target $\mathcal { R }$ :

To enable the model to internalize retrieved knowledge from the LTM, we perform supervised fine-tuning on a constructed dataset of $\mathsf { \bar { \Phi } } ( q , \mathcal { R } , \hat { \mathcal { R } } )$ triplets derived from the LTM retrieval process. Each training instance consists of a question $q$ , its multiple-choice context, and the corresponding

retrieved answer $\mathcal { R }$ provided by the explicit memory module. The model is optimized to generate $\mathcal { R }$ conditioned on $q$ , thereby learning to emulate the retrieval behavior of the LTM within its parametric space.

Formally, the training objective minimizes the tokenlevel cross-entropy loss between the generated sequence $\hat { \mathcal { R } }$ and the target $\mathcal { R }$ :

$$
\mathcal {L} _ {\text {u p d a t e}} = - \sum_ {t = 1} ^ {T} \log P _ {\Theta} \left(r _ {t} \mid q, r _ {<   t}\right), \tag {9}
$$

where $r _ { t }$ denotes the $t$ -th token of the target sequence $\mathcal { R }$ , $r _ { < t }$ represents its preceding tokens, and $\Theta$ are the current model parameters. This process embeds retrieved knowledge directly into the parameter space, serving as the parametric memory update mechanism.

Dynamic Memory Expansion. Parametric memory is dynamically updated as the explicit knowledge graph expands. At each update step $t$ , newly retrieved pairs $\left( q _ { t } , \mathcal { R } _ { t } \right)$ are used to fine-tune the model, producing an updated parameter state:

$$
\mathcal {M} _ {\text {p a r a m e t r i c}} ^ {t + 1} = \mathcal {M} _ {\text {p a r a m e t r i c}} ^ {t} + \Delta \Theta_ {t}, \tag {10}
$$

where $\Delta \Theta _ { t }$ represents the gradient-based weight update induced by fine-tuning on newly accumulated data. This continual process enables the model to progressively encode new knowledge, ensuring that parametric memory evolves in synchrony with the explicit memory without requiring full retraining.

Training Implementation. The fine-tuning process is implemented using a compact autoregressive language model architecture with parameter-efficient optimization. To ensure scalability, training is conducted in mixed precision (bfloat16) with gradient checkpointing to reduce memory consumption. Automatic mixed-precision training and gradient scaling are employed to improve computational efficiency. The optimization is performed using AdamW with a cosine learning rate scheduler for stable convergence. Each training batch consists of input prompts of the form:

$$
\text {P r o m p t :} “ \text {Q u e s t i o n :} q \text {C h o i c e s :} c _ {1}, c _ {2}, \dots , c _ {n} ”, \tag {11}
$$

paired with the corresponding retrieved content $\mathcal { R }$ from the explicit memory. The model learns to generate $\mathcal { R }$ conditioned on the prompt, effectively transforming the nonparametric retrieval process into a parameterized parametric memory representation through supervised fine-tuning.

# 4. Experiment

# 4.1. Setups

Evaluation Datasets. We evaluate the proposed method on three multimodal reasoning benchmarks with distinct characteristics:

• ScienceQA [25]. This dataset contains 21,208 multimodal science questions combining textual and visual contexts, with $4 8 . 7 \%$ of instances containing images. Questions span physics, chemistry, and biology domains, requiring cross-modal reasoning between textual concepts and visual diagrams. Additionally, ScienceQA offers image captions to aid text-only LLMs in reasoning, allowing a comparison of unimodal approaches.   
• LoCoMo [30]. This dataset contains 10 high-quality, very long-term multi-modal dialogues, with each conversation encompassing approximately 600 turns and 16,618 tokens on average, distributed over up to 32 sessions. Dialogues are generated by LLM-based agents grounded in unique personas and temporal event graphs, and are equipped with image-sharing and reaction capabilities.   
• MSR-VTT [49]. A large-scale video–text benchmark comprising 10,000 YouTube video clips across 20 diverse categories, each paired with 20 human-annotated captions, totaling 200K video–sentence pairs. MSR-VTT is widely adopted for video–text retrieval and cross-modal understanding, assessing a model’s ability to align dynamic visual content with natural-language semantics.

Baselines. We compare the proposed method against a broad range of state-of-the-art LLMs and VLMs to assess both unimodal and multimodal reasoning capabilities. For ScienceQA, we compare models across general-domain reasoning in zero- and few-shot settings, covering both textonly and multimodal paradigms. Specifically, text-based LLMs include GPT Model [25], CoT [25], HoT-T5-Large [56] and DDCoT [64], while multimodal VLMs encompass, LaVIN [27], BLIP-2, CCOT [32], and GraphVis [8]. We also include the tool-augmented VLM Chameleon [26] and the RAG-enhanced model VaLiK [22] as strong multimodal reasoning baselines. For LoCoMo, we include GPT-3.5-Turbo [1] and Qwen2.5-7B-Instruct [53], covering strong proprietary and open-source models with robust reasoning and grounding abilities. All models are tested under identical prompting and evaluation protocols to ensure fair cross-modal comparison. For MSR-VTT, We compare our approach with a comprehensive set of representative methods, including InternVideo [43], UMT-L [18], CLIP-VIP [51], mPLUG-2 [48], VAST [4], CLIP2TV [11], DRL [41], TS2-Net [24], CLIP4Clip [28], X-CLIP [29], DMAE [13], Cap4Video $^ { + + }$ [46], TeachClip [40], ExCae [54] and the CLIP [36] baseline. Our baseline uses only the CLIP text encoder, which yields an extremely lightweight model

compared with video based architectures.

Implementation. For multimodal processing, we handle each modality separately. For images, we use GPT-4o-mini to convert them into captions. For audio, we employ Whisper [37] to obtain textual representations. For videos, we sample frames and process them with a VLM to generate captions. We construct three types of memory: core, semantic, and episodic. Entities and relations are then extracted from these memories to build a MMKG, where each entity node and relation edge is linked to multiple text blocks corresponding to the original multimodal data [22].To ensure fairness, in the LoCoMo dataset we exclude the fifth category from comparisons.GPT-4o-mini is used as the LLM for both knowledge graph construction and retrieval.

We construct MMKGs using the training set of ScienceQA and the LoCoMo dialogue dataset, which serves as the generation basis for parameterized memory. For model training, we adopt Qwen2.5-7B as the backbone model and employ supervised fine-tuning to encode the knowledge from ScienceQA and LoCoMo datasets into the model’s weight parameters. The fine-tuning follows the causal language modeling paradigm, with the input-output format designed as ”Question-Retrieved” and loss computed exclusively on the output segment. Key training configurations are set as follows: sequence length of 2048, AdamW optimizer with a learning rate of $2 \times 1 0 ^ { - 6 }$ , linear learning rate scheduler with $10 \%$ warm-up steps, and gradient clipping with a maximum norm of 1.0. All experiments are conducted using at most a single A100 80G GPU.

# 4.2. Main Results

Table 1 presents the performance of various methods on the ScienceQA benchmark. Overall, our MemVerse enhanced models achieve the state-of-the-art performance across most evaluation metrics. Specifically, the highest accuracy in terms of average score is $\mathbf { 8 5 . 4 8 \% }$ , achieved by the GPT-4omini equipped with MemVerse. For subject-specific scores, the MemVerse-enhanced model also obtains the top performance in natural science with $8 5 . 2 6 \%$ , social science with $8 1 . 5 5 \%$ , and language with $8 9 . 0 9 \%$ . In terms of context modalities, it achieves the best results for both text context with $8 3 . 2 8 \%$ and image caption with $7 8 . 1 9 \%$ , demonstrating that our memory mechanism effectively leverages multimodal knowledge.

It is worth noting that, on the ScienceQA dataset, shortterm memory contributes relatively little, as the test questions are largely non-sequential with limited contextual dependencies. The parametric memory, trained using the long-term memory, achieves comparable accuracy to longterm retrieval but with significantly faster access. Specifically, using the RAG approach requires on average 20.17 seconds per question, while retrieving from the compressed

Table 1. Performance comparison $( \% )$ on ScienceQA benchmark. #T-Params denotes trainable parameters. Categories: NAT (natural science), SOC (social science), LAN (language), TXT (text context), IMG-Cap (image caption), NO (no context), G1-6 (grades 1-6), G7- 12 (grades 7-12). Method groups: (1) Human performance baseline, (2) Zero/Few-shot text-only LLMs, (3) Zero/Few-shot Multimodal VLMs, (4) LLMs/VLMs enhanced with memory mechanisms or external knowledge for multimodal reasoning. The best results are marked in bold, the second-best results are underlined.   

<table><tr><td rowspan="2">Method</td><td rowspan="2">#T-Param</td><td colspan="3">Subject</td><td colspan="3">Context Modality</td><td colspan="2">Grade</td><td rowspan="2">Average</td></tr><tr><td>NAT</td><td>SOC</td><td>LAN</td><td>TXT</td><td>IMG</td><td>NO</td><td>G1-6</td><td>G7-12</td></tr><tr><td>Human [25]</td><td>-</td><td>90.23</td><td>84.97</td><td>87.48</td><td>89.60</td><td>87.50</td><td>88.10</td><td>91.59</td><td>82.42</td><td>88.40</td></tr><tr><td>GPT-4 [21]</td><td>-</td><td>84.06</td><td>73.45</td><td>87.36</td><td>81.87</td><td>70.75</td><td>90.73</td><td>84.69</td><td>79.10</td><td>82.69</td></tr><tr><td>CoT (GPT-3) [25]</td><td>173B</td><td>75.44</td><td>70.87</td><td>78.09</td><td>74.68</td><td>67.43</td><td>79.93</td><td>78.23</td><td>69.68</td><td>75.17</td></tr><tr><td>CoT (UnifiedQA) [25]</td><td>223M</td><td>71.00</td><td>76.04</td><td>78.91</td><td>66.42</td><td>66.53</td><td>81.81</td><td>77.06</td><td>68.82</td><td>74.11</td></tr><tr><td>CoT (GPT-4) [26]</td><td>1T+</td><td>85.48</td><td>72.44</td><td>90.27</td><td>82.65</td><td>71.49</td><td>92.89</td><td>86.66</td><td>79.04</td><td>83.99</td></tr><tr><td>DDCoT [64]</td><td>175B</td><td>80.15</td><td>76.72</td><td>82.82</td><td>78.89</td><td>72.53</td><td>85.02</td><td>82.86</td><td>75.21</td><td>80.15</td></tr><tr><td>HoT-T5-Large [56]</td><td>738M</td><td>84.46</td><td>79.08</td><td>84.64</td><td>82.89</td><td>75.81</td><td>88.15</td><td>83.88</td><td>82.47</td><td>83.38</td></tr><tr><td>Chameleon (ChatGPT) [26]</td><td>175B+</td><td>81.62</td><td>70.64</td><td>84.00</td><td>79.77</td><td>70.80</td><td>86.62</td><td>81.86</td><td>76.53</td><td>79.93</td></tr><tr><td>LaVIN-13B [55]</td><td>13B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>77.54</td></tr><tr><td>BLIP-2 [55]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>74.17</td></tr><tr><td>CCOT [32]</td><td>7B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>76.84</td></tr><tr><td>GraphVis [8]</td><td>7B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>73.18</td></tr><tr><td>Qwen2.5-7B [22]</td><td>7B</td><td>76.20</td><td>67.83</td><td>77.27</td><td>74.49</td><td>65.79</td><td>79.02</td><td>77.72</td><td>69.35</td><td>74.72</td></tr><tr><td>Qwen2.5-7B (Mmkg) [22]</td><td>7B</td><td>73.98</td><td>66.37</td><td>78.18</td><td>71.65</td><td>64.30</td><td>79.65</td><td>76.51</td><td>68.03</td><td>73.47</td></tr><tr><td>Qwen2.5-7B (Visual Genome) [22]</td><td>7B</td><td>76.78</td><td>67.04</td><td>78.09</td><td>74.05</td><td>66.19</td><td>79.72</td><td>78.08</td><td>69.68</td><td>75.08</td></tr><tr><td>Qwen2.5-7B (MemVerse)</td><td>7B</td><td>74.51</td><td>68.5</td><td>78.73</td><td>75.92</td><td>66.19</td><td>81.95</td><td>79.70</td><td>64.73</td><td>75.62</td></tr><tr><td>Qwen2.5-72B [22]</td><td>72B</td><td>79.64</td><td>67.10</td><td>84.90</td><td>77.56</td><td>65.00</td><td>87.93</td><td>80.25</td><td>74.85</td><td>78.37</td></tr><tr><td>Qwen2.5-72B (MemVerse)</td><td>72B</td><td>77.53</td><td>68.95</td><td>85.36</td><td>78.68</td><td>66.39</td><td>89.20</td><td>82.31</td><td>77.76</td><td>80.25</td></tr><tr><td>GPT-4o-mini</td><td>-</td><td>77.31</td><td>73.45</td><td>86.91</td><td>74.05</td><td>66.86</td><td>87.93</td><td>83.37</td><td>71.85</td><td>76.82</td></tr><tr><td>GPT-4o-mini (MemVerse)</td><td>-</td><td>85.26</td><td>81.55</td><td>89.09</td><td>83.28</td><td>78.19</td><td>91.50</td><td>88.11</td><td>80.75</td><td>85.48</td></tr></table>

long-term memory only takes 8.26 seconds on average. The parametric memory further reduces the average retrieval time to 2.28 seconds, achieving an acceleration of approximately $89 \%$ compared to RAG and $72 \%$ compared to long-term retrieval, while maintaining similar performance. This demonstrates that parametric memory offers a practical trade-off between speed and effectiveness in knowledgeaugmented reasoning.

Furthermore, we observe that the impact of memory enhancement differs between Qwen and GPT-4o-mini. For Qwen, adding memory results in only modest improvements, whereas GPT-4o-mini benefits significantly from MemVerse. Our analysis suggests that GPT-based models are more capable of leveraging retrieved knowledge, carefully integrating it into reasoning steps. In contrast, Qwen struggles to connect retrieved content with the question context, which can result in errors even when correct information is retrieved. These findings indicate that effective memory utilization in downstream tasks requires not only highquality retrieval but also careful prompt design to guide the model in learning and applying relevant knowledge, rather than simply injecting information indiscriminately.

To further evaluate the generalization ability of our method, we conduct additional experiments on the MSR-

VTT dataset. At test time, we first use the query text to retrieve relevant memory entries from the knowledge graph and then rewrite the query by concatenating it with the retrieved information before performing the final matching. As shown in Table 2, our MemVerse approach achieves an $\mathbf { R } \ @ 1$ score of $9 0 . 4 \%$ for text-to-video retrieval and $8 9 . 2 \%$ for video-to-text retrieval, compared with $2 9 . 7 \%$ and $2 1 . 4 \%$ for the CLIP baseline, respectively. This corresponds to an improvement of $6 0 . 7 \%$ and $6 7 . 8 \%$ percentage points, demonstrating that incorporating a memory-based knowledge graph and rewriting queries with retrieved information dramatically enhances semantic matching on MSR-VTT.

Notably, our approach achieves these improvements without exposing the ground-truth alignment between captions and videos to the knowledge graph or the paramatric module, ensuring fair evaluation. During memory construction, pairs of captions (the original caption and the caption generated from the video) are partially aligned and connected through GPT-4o-mini’s powerful understanding, judgment, and reasoning capabilities, forming linked representations that are stored in the memory. During retrieval, the query text can effectively leverage these stored associations, resulting in highly accurate matching. As a result, this memory-based approach overcomes the limitations of

Table 2. Comparison with SOTA methods on MSR-VTT dataset (underline marks the best ViT-based results)   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="3">text-to-video</td><td colspan="3">video-to-text</td></tr><tr><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td><td>R@1↑</td><td>R@5↑</td><td>R@10↑</td></tr><tr><td colspan="8">Pre-trained foundation model</td></tr><tr><td>InternVideo [43]</td><td>ViT-H/14</td><td>55.2</td><td>79.6</td><td>87.5</td><td>57.9</td><td>79.2</td><td>86.4</td></tr><tr><td>UMT-L [18]</td><td>ViT-L/14</td><td>58.8</td><td>81.0</td><td>87.1</td><td>58.6</td><td>81.6</td><td>86.5</td></tr><tr><td>CLIP-VIP [51]</td><td>ViT-B/16</td><td>57.7</td><td>80.5</td><td>88.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>mPLUG-2 [48]</td><td>ViT-L/14</td><td>53.1</td><td>77.6</td><td>84.7</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VAST [4]</td><td>ViT-G/14</td><td>63.9</td><td>84.3</td><td>89.6</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="8">ViT-based</td></tr><tr><td>CLIP2TV [11]</td><td>ViT-B/16</td><td>49.3</td><td>74.7</td><td>83.6</td><td>46.9</td><td>75.0</td><td>85.1</td></tr><tr><td>DRL [41]</td><td>ViT-B/16</td><td>49.4</td><td>76.4</td><td>84.2</td><td>47.0</td><td>77.1</td><td>84.4</td></tr><tr><td>TS2-Net [24]</td><td>ViT-B/16</td><td>47.8</td><td>76.8</td><td>85.2</td><td>47.8</td><td>76.0</td><td>84.6</td></tr><tr><td>Clip4Clip [28]</td><td>ViT-B/16</td><td>46.4</td><td>72.1</td><td>82.0</td><td>45.4</td><td>73.4</td><td>82.4</td></tr><tr><td>X-CLIP [29]</td><td>ViT-B/16</td><td>49.3</td><td>75.8</td><td>84.8</td><td>48.9</td><td>76.8</td><td>84.5</td></tr><tr><td>DMAE [13]</td><td>ViT-B/16</td><td>49.9</td><td>75.8</td><td>85.5</td><td>49.6</td><td>76.3</td><td>85.0</td></tr><tr><td>Cap4Video++ [46]</td><td>ViT-B/16</td><td>52.3</td><td>76.8</td><td>85.8</td><td>50.0</td><td>75.9</td><td>86.0</td></tr><tr><td>TeachClip [40]</td><td>ViT-B/16</td><td>48.0</td><td>75.9</td><td>83.5</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ExCae [54]</td><td>ViT-G/14</td><td>67.7</td><td>92.7</td><td>96.2</td><td>69.3</td><td>92.5</td><td>96.3</td></tr><tr><td>CLIP [36]</td><td>-</td><td>29.7</td><td>48.9</td><td>58.8</td><td>21.4</td><td>38.6</td><td>44.3</td></tr><tr><td>MemVerse</td><td>-</td><td>90.4</td><td>95.6</td><td>98.1</td><td>89.2</td><td>92.7</td><td>99.0</td></tr></table>

directly using large LLMs or VLMs for retrieval over massive multimodal candidate pools. Training such large video models is extremely costly due to the scale of the data, and their limited context windows prevent them from processing all candidates simultaneously. By storing semantic relationships extracted with the understanding, judgment, and reasoning capabilities of models like GPT-4o-mini, our method allows lightweight embedding-based models to efficiently retrieve highly relevant entries while still leveraging the rich reasoning captured in the memory. This effectively combines the scalability of embedding-based retrieval with the semantic power of large pretrained models, demonstrating the key advantage of our approach.

# 4.3. Further Analysis

Due to space constraints, the full quantitative and qualitative results on the Locomo dataset are provided in Appendix C. We investigate the periodic update strategy of the parametric memory on the ScienceQA test set. Using a grouped partitioning scheme, we simulate incremental knowledge accumulation and examine how different update intervals influence the stability and long-term retention of distilled memory. The detailed setup, ablations, and temporal performance curves are presented in Appendix D. Furthermore, to assess the scalability of MemVerse, we conduct comprehensive experiments across primary model sizes ranging from Qwen2.5 1.5B to 72B, while keeping the parametric memory module fixed as a Qwen2.5-7B model. This design allows us to isolate the impact of scaling the main model

and analyze how memory efficiency and reasoning accuracy evolve with increasing capacity, with all additional results and discussions presented in Appendix E. These studies provide deeper insights into the scalability and adaptability of MemVerse across different tasks and configurations.

# 5. Conclusion

We introduced MemVerse, a model-agnostic, plug-and-play memory framework that brings AI agents closer to lifelong multimodal intelligence. MemVerse unifies fast parametric recall with slow, hierarchical retrieval-based memory, mirroring the complementary roles of intuition and deliberation. This dual-path design lets agents maintain recent context, transform raw multimodal experiences into structured knowledge, and retrieve information efficiently without unbounded memory growth. Across diverse multimodal tasks, MemVerse yields significant improvements in reasoning accuracy, stability, and long-horizon coherence. These results demonstrate that scalable memory, rather than ever-larger models, is the missing ingredient for continuous-learning agents. Looking ahead, we plan to explore more adaptive memory-control strategies and to deploy MemVerse in open-world environments across a variety of domains.

# References

[1] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom

Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. Language models are few-shot learners. In Advances in Neural Information Processing Systems, pages 1877–1901. Curran Associates, Inc., 2020. 6   
[2] Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. Fireact: Toward language agent fine-tuning. arXiv preprint arXiv:2310.05915, 2023. 3   
[3] Hanzhi Chen, Boyang Sun, Anran Zhang, Marc Pollefeys, and Stefan Leutenegger. Vidbot: Learning generalizable 3d actions from in-the-wild 2d human videos for zero-shot robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 27661–27672, 2025. 2   
[4] Sihan Chen, Handong Li, Qunbo Wang, Zijia Zhao, Mingzhen Sun, Xinxin Zhu, and Jing Liu. Vast: A vision-audio-subtitle-text omni-modality foundation model and dataset. Advances in Neural Information Processing Systems, 36, 2024. 6, 8   
[5] Z. Chen et al. Knowledge graphs meet multi-modal learning, 2024. arXiv:2402.05391. 3   
[6] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025. 2, 3   
[7] Ian J. Deary, Lars Penke, and Wendy Johnson. The neuroscience of human intelligence differences. Nature Reviews Neuroscience, 11(3):201–211, 2010. 2   
[8] Yihe Deng, Chenchen Ye, Zijie Huang, Mingyu Derek Ma, Yiwen Kou, and Wei Wang. Graphvis: Boosting llms with visual knowledge graph integration. In Advances in Neural Information Processing Systems, pages 67511–67534. Curran Associates, Inc., 2024. 6, 7   
[9] Yuhao Dong, Zuyan Liu, Hai-Long Sun, Jingkang Yang, Winston Hu, Yongming Rao, and Ziwei Liu. Insight-v: Exploring long-chain visual reasoning with multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9062–9072, 2025. 2   
[10] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. In Computer Vision – ECCV 2024, pages 75–92, Cham, 2025. Springer Nature Switzerland. 2   
[11] Zijian Gao, Jingyu Liu, Sheng Chen, Dedan Chang, et al. Clip2tv: An empirical study on transformer-based methods for video-text retrieval. arXiv preprint arXiv:2111.05610, 1 (2):6, 2021. 6, 8   
[12] Jui-Ting Huang, Ashish Sharma, Shuying Sun, Li Xia, David Zhang, Philip Pronin, Janani Padmanabhan, Giuseppe Ottaviano, and Linjun Yang. Embedding-based retrieval in facebook search. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Min-

ing, page 2553–2561, New York, NY, USA, 2020. Association for Computing Machinery. 2   
[13] Chen Jiang, Hong Liu, Xuzheng Yu, et al. Dual-modal attention-enhanced text-video retrieval with triplet partial margin contrastive learning. In ACM International Conference on Multimedia, pages 4626–4636, 2023. 6, 8   
[14] Daniel Kahneman. Thinking, Fast and Slow. Farrar, Straus and Giroux, New York, NY, 2011. 2   
[15] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick ˘ Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wentau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of EMNLP, pages 6769–6781. ACL, 2020. 3   
[16] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, et al. Retrievalaugmented generation for knowledge-intensive nlp tasks. In NeurIPS, 2020. arXiv:2005.11401. 3   
[17] Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In Proceedings of the International Conference on Machine Learning (ICML) / PMLR, 2023. arXiv:2301.12597. 3   
[18] Kunchang Li, Yali Wang, Yizhuo Li, Yi Wang, Yinan He, Limin Wang, and Yu Qiao. Unmasked teacher: Towards training-efficient video foundation models. In IEEE/CVF International Conference on Computer Vision, pages 19948– 19960, 2023. 6, 8   
[19] Yuqi Li, Chuanguang Yang, Hansheng Zeng, Zeyu Dong, Zhulin An, Yongjun Xu, Yingli Tian, and Hao Wu. Frequency-aligned knowledge distillation for lightweight spatiotemporal forecasting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7262– 7272, 2025. 2   
[20] Fang Liu, Clement Yu, Weiyi Meng, and Abdur Chowdhury. Effective keyword search in relational databases. In Proceedings of the 2006 ACM SIGMOD International Conference on Management of Data, page 563–574, New York, NY, USA, 2006. Association for Computing Machinery. 2   
[21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In Advances in Neural Information Processing Systems, pages 34892–34916. Curran Associates, Inc., 2023. 7   
[22] Junming Liu, Siyuan Meng, Yanting Gao, Song Mao, Pinlong Cai, Guohang Yan, Yirong Chen, Zilin Bian, Ding Wang, and Botian Shi. Aligning vision to language: Annotation-free multimodal knowledge graph construction for enhanced llms reasoning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 981–992, 2025. 2, 3, 6, 7   
[23] Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, and Jun Ma. Hm-rag: Hierarchical multi-agent multimodal retrieval augmented generation. In Proceedings of the 33rd ACM International Conference on Multimedia, page 2781–2790, New York, NY, USA, 2025. Association for Computing Machinery. 2   
[24] Yuqi Liu, Pengfei Xiong, Luhui Xu, et al. Ts2-net: Token shift and selection transformer for text-video retrieval. In

European Conference on Computer Vision, pages 319–335, 2022. 6, 8   
[25] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, pages 2507–2521. Curran Associates, Inc., 2022. 6, 7   
[26] Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao. Chameleon: Plug-and-play compositional reasoning with large language models. In Advances in Neural Information Processing Systems, pages 43447–43478. Curran Associates, Inc., 2023. 6, 7   
[27] Gen Luo, Yiyi Zhou, Tianhe Ren, Shengxin Chen, Xiaoshuai Sun, and Rongrong Ji. Cheap and quick: Efficient visionlanguage instruction tuning for large language models. In Advances in Neural Information Processing Systems, pages 29615–29627. Curran Associates, Inc., 2023. 6   
[28] Huaishao Luo, Lei Ji, Ming Zhong, Yang Chen, et al. Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning. Neurocomputing, 508:293– 304, 2022. 6, 8   
[29] Yiwei Ma, Guohai Xu, Xiaoshuai Sun, et al. X-clip: Endto-end multi-grained contrastive learning for video-text retrieval. In ACM International Conference on Multimedia, pages 638–647, 2022. 6, 8   
[30] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. arXiv preprint arXiv:2402.17753, 2024. 6   
[31] Siyuan Meng, Junming Liu, Yirong Chen, Song Mao, Pinlong Cai, Guohang Yan, Botian Shi, and Ding Wang. From ranking to selection: A simple but efficient dynamic passage selector for retrieval augmented generation. arXiv preprint arXiv:2508.09497, 2025. 2   
[32] Chancharik Mitra, Brandon Huang, Trevor Darrell, and Roei Herzig. Compositional chain-of-thought prompting for large multimodal models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14420–14431, 2024. 6, 7   
[33] Charles Packer, Vivian Fang, Shishir G Patil, Kevin Lin, Sarah Wooders, and Joseph E Gonzalez. Memgpt: Towards llms as operating systems., 2023. 3   
[34] Alexander Pritzel, Benigno Uria, Sriram Srinivasan, Adria` Puigdomenech Badia, Oriol Vinyals, et al. Neural episodic ` control. In Proceedings of the 34th International Conference on Machine Learning (ICML), pages 2827–2836. PMLR, 2017. 2   
[35] Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun Huang. Memorag: Boosting long context processing with global memory-enhanced retrieval augmentation. In Proceedings of the ACM on Web Conference 2025, page 2366–2377, New York, NY, USA, 2025. Association for Computing Machinery. 3   
[36] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, et al. Learning

transferable visual models from natural language supervision. arXiv:2103.00020, 2021. 3, 6, 8   
[37] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Proceedings of the 40th International Conference on Machine Learning, pages 28492–28518. PMLR, 2023. 6   
[38] Dhravya Shah, Mahesh Sanikommu, Yash, et al. Supermemory. https://supermemory.ai/, 2025. Accessed: 2025-11-05. 3   
[39] Ying Tai, Jian Yang, Xiaoming Liu, and Chunyan Xu. Memnet: A persistent memory network for image restoration. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2017. 2   
[40] Kaibin Tian, Ruixiang Zhao, Zijie Xin, Bangxiang Lan, and Xirong Li. Holistic features are almost sufficient for textto-video retrieval. In CVPR, pages 17138–17147, 2024. 6, 8   
[41] Qiang Wang, Yanhao Zhang, Yun Zheng, Pan Pan, and Xian-Sheng Hua. Disentangled representation learning for textvideo retrieval. arXiv preprint arXiv:2203.07111, 2022. 6, 8   
[42] Yu Wang and Xi Chen. Mirix: Multi-agent memory system for llm-based agents. arXiv preprint arXiv:2507.07957, 2025. 2   
[43] Yi Wang, Kunchang Li, Yizhuo Li, et al. Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191, 2022. 6, 8   
[44] Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, et al. Memoryllm: Towards self-updatable large language models. arXiv preprint arXiv:2402.04624, 2024. 3   
[45] Chao-Yuan Wu, Yanghao Li, Karttikeya Mangalam, Haoqi Fan, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Memvit: Memory-augmented multiscale vision transformer for efficient long-term video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 13587–13597, 2022. 2   
[46] Wenhao Wu, Xiaohan Wang, Haipeng Luo, Jingdong Wang, Yi Yang, and Wanli Ouyang. Cap4video++: Enhancing video understanding with auxiliary captions. TPAMI, pages 1–15, 2024. 6, 8   
[47] Xi Xiao, Yunbei Zhang, Xingjian Li, Tianyang Wang, Xiao Wang, Yuxiang Wei, Jihun Hamm, and Min Xu. Visual instance-aware prompt tuning. Proceedings of the 33rd ACM International Conference on Multimedia, 2025. 2   
[48] Haiyang Xu, Qinghao Ye, Ming Yan, Yaya Shi, Jiabo Ye, Yuanhong Xu, Chenliang Li, Bin Bi, et al. mplug-2: A modularized multi-modal foundation model across text, image and video. In International Conference on Machine Learning, pages 38728–38748, 2023. 6, 8   
[49] Jun Xu, Tao Mei, Ting Yao, and Yong Rui. Msr-vtt: A large video description dataset for bridging video and language. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 6   
[50] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents, 2025. 2

[51] Hongwei Xue, Yuchong Sun, Bei Liu, Jianlong Fu, Ruihua Song, Houqiang Li, and Jiebo Luo. Clip-vip: Adapting pretrained image-text model to video-language representation alignment. In International Conference on Learning Representation, 2023. 6, 8   
[52] Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Kristian Kersting, Jeff Z Pan, Hinrich Schutze, et al. Memory-r1: En- ¨ hancing large language model agents to manage and utilize memories via reinforcement learning. arXiv preprint arXiv:2508.19828, 2025. 2   
[53] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. 6   
[54] Baoyao Yang, Junxiang Chen, Wanyun Li, Wenbin Yao, and Yang Zhou. Expertized caption auto-enhancement for videotext retrieval. arXiv preprint arXiv:2502.02885, 2025. 6, 8   
[55] Xiaocui Yang, Wenfang Wu, Shi Feng, Ming Wang, Daling Wang, Yang Li, Qi Sun, Yifei Zhang, Xiaoming Fu, and Soujanya Poria. Mm-bigbench: Evaluating multimodal models on multimodal content comprehension tasks. arXiv preprint arXiv:2310.09036, 2023. 7   
[56] Fanglong Yao, Changyuan Tian, Jintao Liu, Zequn Zhang, Qing Liu, Li Jin, Shuchao Li, Xiaoyu Li, and Xian Sun. Thinking like an expert:multimodal hypergraph-of-thought (hot) reasoning to boost foundation modals, 2023. 6, 7   
[57] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023. 2   
[58] Da Yin, Faeze Brahman, Abhilasha Ravichander, Khyathi Chandu, Kai-Wei Chang, Yejin Choi, and Bill Yuchen Lin. Agent lumos: Unified and modular training for open-source language agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12380–12403, Bangkok, Thailand, 2024. Association for Computational Linguistics. 3   
[59] Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, et al. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259, 2025. 3   
[60] Hanxun Yu, Wentong Li, Song Wang, Junbo Chen, and Jianke Zhu. Inst3d-lmm: Instance-aware 3d scene understanding with multi-modal instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 14147–14157, 2025. 2   
[61] Guibin Zhang, Muxin Fu, and Shuicheng Yan. Memgen: Weaving generative latent memory for self-evolving agents. arXiv preprint arXiv:2509.24704, 2025. 3   
[62] Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language modelbased agents. ACM Trans. Inf. Syst., 43(6), 2025. 3   
[63] Zeyu Zhang, Quanyu Dai, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A

survey on the memory mechanism of large language modelbased agents. ACM Trans. Inf. Syst., 43(6), 2025. 2   
[64] Ge Zheng, Bin Yang, Jiajin Tang, Hong-Yu Zhou, and Sibei Yang. Ddcot: Duty-distinct chain-of-thought prompting for multimodal reasoning in language models. In Advances in Neural Information Processing Systems, pages 5168–5191. Curran Associates, Inc., 2023. 6, 7   
[65] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. Memorybank: Enhancing large language models with long-term memory. Proceedings of the AAAI Conference on Artificial Intelligence, 38(17):19724–19731, 2024. 3   
[66] Xiangru Zhu, Zhixu Li, Xiaodan Wang, Xueyao Jiang, Penglei Sun, et al. Multi-modal knowledge graph construction and application. arXiv:2202.05786, 2022. 3
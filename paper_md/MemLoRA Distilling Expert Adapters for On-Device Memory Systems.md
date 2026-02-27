# MemLoRA: Distilling Expert Adapters for On-Device Memory Systems

Massimo Bini1,2†, Ondrej Bohdal1, Umberto Michieli1, Zeynep Akata2, Mete Ozay1, Taha Ceritli1

1Samsung R&D Institute UK 2Technical University of Munich, Helmholtz Munich, MCML

# Abstract

Memory-augmented Large Language Models (LLMs) have demonstrated remarkable consistency during prolonged dialogues by storing relevant memories and incorporating them as context. Such memory-based personalization is also key in on-device settings that allow users to keep their conversations and data private. However, memoryaugmented systems typically rely on LLMs that are too costly for local on-device deployment. Even though Small Language Models (SLMs) are more suitable for on-device inference than LLMs, they cannot achieve sufficient performance. Additionally, these LLM-based systems lack native visual capabilities, limiting their applicability in multimodal contexts. In this paper, we introduce (i) MemLoRA, a novel memory system that enables local deployment by equipping SLMs with specialized memory adapters, and (ii) its vision extension MemLoRA-V, which integrates small Vision-Language Models (SVLMs) to memory systems, enabling native visual understanding. Following knowledge distillation principles, each adapter is trained separately for specific memory operations—knowledge extraction, memory update, and memory-augmented generation. Equipped with memory adapters, small models enable accurate on-device memory operations without cloud dependency. On text-only operations, MemLoRA outperforms $I O \times$ larger baseline models (e.g., Gemma2-27B) and achieves performance comparable to $6 0 \times$ larger models (e.g., GPT-OSS-120B) on the LoCoMo benchmark. To evaluate visual understanding operations instead, we extend Lo-CoMo with challenging Visual Question Answering tasks that require direct visual reasoning. On this, our VLMintegrated MemLoRA-V shows massive improvements over caption-based approaches (81.3 vs. 23.7 accuracy) while keeping strong performance in text-based tasks, demonstrating the efficacy of our method in multimodal contexts.

# 1. Introduction

Recent advancements in Large Language Models (LLMs) and Vision Language Models (VLMs) have led to their

widespread use in conversational Artificial Intelligence (AI) systems, ranging from customer service chatbots to personal assistants and collaborative productivity tools [52, 37]. VLMs have demonstrated remarkable capabilities in multimodal understanding and generation, making them increasingly integral to human-computer interaction across diverse domains. However, the effectiveness of VLMs in real-world on-device conversational applications is fundamentally constrained by LLMs’ limited context windows [36, 6]. While modern LLMs can process thousands of tokens in a single session, they cannot retain information across multiple conversations or maintain long-term userspecific knowledge. This limitation becomes particularly problematic in multi-session scenarios where users expect the system to remember previous interactions, preferences, and contextual details—a critical requirement for delivering truly personalized and coherent conversational experiences.

To address these challenges, researchers have proposed various memory systems that extend LLMs with persistent memory capabilities. Early approaches focused on integrating external memory through differentiable attention mechanisms [46] and retrieval-augmented generation from knowledge bases [30], establishing foundational paradigms for extending model knowledge beyond immediate context. Building on these foundations, recent works have explored sophisticated memory management strategies that mirror human cognitive processes, including temporal decay mechanisms for selective retention [53], hierarchical memory systems inspired by the design of operating systems [39], and knowledge graph representations that track evolving information over time [40]. Contemporary systems have further expanded the role of LLMs beyond generation, leveraging them as active agents within the memory pipeline itself. Examples include using LLMs to automatically extract knowledge and update the memory [6, 26], evaluate memory relevance and quality [36], and dynamically restructure knowledge networks according to emerging patterns [47], thereby transforming memory augmentation from a passive retrieval mechanism into an intelligent, self-improving system.

Despite these advances, current memory systems face significant practical limitations that restrict their deploy-

![](images/d9f0677a32fc9c446703372d3d1c32ce7e45c08ddf6e813daa7ee8c7354ca1a8.jpg)  
Figure 1. Overview. We employ specialized LoRA adapters to enable small (vision) language models to perform memory operations for on-device deployment. The base model dynamically switches between expert adapters, each trained for a distinct stage: (1) knowledge extraction, (2) memory update, (3) memory-augmented generation. In the last stage, the model can switch between text-only and multimodal adapter, depending on the input. By specializing each adapter for its specific operation, MemLoRA(-V) achieves performance comparable to models $1 0 { - } 6 0 \mathrm { x }$ larger while enabling efficient local execution without cloud API dependencies.

ment and effectiveness. Firstly, these systems fundamentally rely on large, often proprietary, LLMs for core memory operations—including extraction, organization, updating, and retrieval—necessitating continuous API calls to cloudbased services [6, 39]. This dependency not only introduces latency and cost concerns but also prevents on-device deployment, limiting their applicability in privacy-sensitive contexts, offline scenarios, or resource-constrained environments where cloud connectivity cannot be guaranteed. In our work, we tackle this challenge by replacing queries posed to a large-scale model through API, with a small on-device model, equipped with task-specific expert adapters. These adapters are trained via knowledge distillation through teacher answers or ground-truth data. We provide an overview of the approach and our considered setting in Figure 1.

Secondly, while recent works have begun exploring multimodal capabilities, the handling of visual information remains predominantly text-centric: images are typically converted into textual descriptions through vision-language models before being stored and retrieved [6, 26], an approach that inevitably loses fine-grained visual details, spatial relationships, and numerical information embedded in charts or diagrams. This text-first paradigm, though computationally practical, fundamentally constrains the systems’ ability to reason directly over visual content, limiting their effectiveness in domains where visual information plays

a critical role, such as technical documentation, medical imaging, or design workflows. Notably, existing benchmarks for evaluating memory systems—such as LoCoMo [36], which focuses on text-based conversational question answering and event summarization—do not assess multimodal capabilities during inference. Although LoCoMo conversations contain images, the original evaluation relies solely on text-based captions, limiting assessment of native visual understanding. This evaluation gap means that a model’s ability to process and reason over visual information directly, rather than through caption intermediaries, remains unmeasured.

In our work, we address both issues by integrating Vision Language Models (VLMs) in these memory-augmented systems, and by augmenting the LoCoMo benchmark with Visual Questions and Answers (VQA) on the conversational images. By doing this, not only we are able to give native visual capabilities to memory-augmented systems, but we are also able to develop our MemLoRA memory system on small VLMs (SVLMs). For these, a novel expert adapter is further introduced to address the VQA task. Such an approach shows how having specialized adapters, one for each operation, can substitute the need for having massive models, and allow for on-device deployment effectively.

We summarize our contributions as follows. (i) We introduce the challenge of accurate on-device memory systems where small language models are used, eliminating reliance

on cloud-based infrastructure to preserve privacy. (ii) We develop a highly-efficient yet well-performing solution that substantially improves over existing approaches and obtains performance close to that of significantly larger models. (iii) We extend memory systems to incorporate Vision Language Models with native visual capabilities and apply our MemLoRA framework to this multimodal setting through a specialized vision expert adapter. (iv) We augment the LoCoMo benchmark with challenging Visual Question Answering tasks that require direct image access, demonstrating that our approach achieves strong performance with superior efficiency in multimodal contexts.

# 2. Related Work

Memory-Augmented LLMs. Memory systems have improved LLMs’ capabilities in several applications. Foundational approaches such as Memory Networks [46] and RAG [30] introduced external memory integration and document retrieval. More sophisticated systems have been inspired by human cognition and operating systems. Recent innovations like MemoryBank [53], MemGPT [39], Zep [40], and Mem0 [6] incorporate hierarchical memory tiers, session management, and self-improving capabilities, while specialized systems like ReadAgent [26] and A-Mem [47] implement human-inspired organizational principles such as gist memory compression and Zettelkasten-style knowledge networks [21]. Many of these approaches rely on agentic frameworks that orchestrate memory operations through iterative LLM queries for tasks such as memory extraction, updating, and retrieval. However, such methods require multiple queries to the LLM that are computationally expensive to run and do not prioritize on-device deployment scenarios. In our work, we employ specialized expert adapters on small models to perform memory operations locally, drastically reducing computational demands.

Knowledge Distillation with LLMs. Knowledge distillation has evolved into a diverse landscape of techniques aimed at transferring capabilities from powerful teacher models to more efficient student models [48, 5]. Generation-aware divergence methods address the limitations of traditional forward Kullback–Leibler Divergence (KLD) by introducing variants such as reverse KLD [15] or skew KLD [23], which better handle the challenges of autoregressive generation while requiring access to the teacher model’s internal logits or probability distributions. More recent methods have introduced preference-based frameworks that leverage implicit reward signals [32], advantage functions [13], or pseudo-preference pairs [49] to guide the student toward generating outputs that align with the teacher’s quality standards [24]. These approaches often necessitate either white-box access to the teacher model’s internal states or involve multi-stage optimization procedures.

Alternatively, output-only distillation methods that operate solely on generated text sequences represent another direction in the literature [8, 42, 49, 12]. Such approaches enable distillation from black-box models, proprietary APIs, or any teacher model regardless of architecture, and allow for direct modification of outputs into desired formats or structures before training. In our work, we found that this simpler solution worked well for distilling knowledge in memory systems, and given the practical advantages it offers, we adopted this approach.

Parameter Efficient Finetuning Methods. Parameterefficient finetuning (PEFT) methods have emerged as a powerful approach for adapting large-scale models to specific tasks and domains, while drastically reducing computational requirements compared to full-model finetuning [16, 29, 33, 3, 9]. Notable methods in this category are those that inject trainable modules into the model architecture, such as Low-Rank Adaptation (LoRA) [17] and its derivations [25, 35, 4]. A key advantage of these approaches is that the trained modules can be seamlessly merged and unmerged into the base model weights, eliminating any inference-time latency overhead—a crucial consideration for deployment of large-scale models in production environments—and enabling efficient multi-task setups where a single base model can be dynamically adapted to different tasks or domains simply by swapping the active PEFT module [18]. In our work, we leverage these properties and demonstrate how small (vision) language models with PEFT adapters are able to achieve performance on par with larger counterparts by swapping between expert memory adapters, while significantly increasing efficiency and enabling practical on-device deployment.

# 3. Method

In this section, we present the technical details of Mem-LoRA, our efficient memory system suitable for on-device deployment. We begin by introducing Mem0 [6], the memory system we build upon, and describing its core operations (Section 3.1). We then detail our proposed MemLoRA solution, which replaces the LLM in Mem0 with an SLM and memory adapters through knowledge distillation (Section 3.2). Finally, we extend our approach to multimodal settings by incorporating vision understanding capabilities, enabling memory systems to process visual information natively (Section 3.3).

# 3.1. Preliminaries

Mem0. Mem0 [6] is a memory system enhancing LLM applications with persistent, personalized memory across sessions. Mem0 operates through three main stages:

‚ Knowledge Extraction. Given a conversational exchange between a user and an AI assistant, Mem0 uses an extraction

prompt to query an LLM $f _ { \theta _ { L } }$ , parametrized by $\theta _ { L }$ . The extraction prompt guides the LLM to identify relevant knowledge, $\Omega$ , consisting of facts, preferences, and contextual information worth storing in memory from the dialogue.

‚ Memory Update. The extracted knowledge $\Omega$ is used to update the memory store M. Mem0 queries $f _ { \theta _ { L } }$ to determine how new information should be integrated with existing memory $M .$ —whether to add new entries (ADD), update existing ones (UPDATE), or delete outdated information (DELETE). This ensures the memory remains relevant and consistent over time.   
‚ Memory-Augmented Generation. During inference, relevant memories $\Omega ^ { \prime }$ are retrieved from the memory store $M$ based on semantic similarity to the current query $q$ $\Omega ^ { \prime } \gets$ FindRelatedKnowledge $( M , q )$ . These memories $\Omega ^ { \prime }$ are then provided in the prompt as additional context to $f _ { \theta _ { L } }$ , enabling it to generate responses that are consistent with past interactions and personalized to the user.

While effective, this approach requires multiple calls to the LLM $f _ { \theta _ { L } }$ , making it impractical for on-device deployment where computational efficiency, privacy, and offline functionality are critical.

Low-Rank Adaptation (LoRA). LoRA [17] is a PEFT method to adapt pretrained models by injecting trainable low-rank matrices into specific layers while keeping the original model weights frozen. Given a pretrained weight matrix $W _ { 0 } ~ \in ~ \mathbb { R } ^ { d \times k }$ , the LoRA adapter $L$ represents the weight update as the product of two low-rank matrices $( A , B )$ :

$$
W = W _ {0} + B A,
$$

where $B \in \mathbb { R } ^ { d \times r }$ and $\boldsymbol { A } \in \mathbb { R } ^ { r \times k }$ with rank $r \ll \operatorname* { m i n } ( d , k )$ and $W$ being the updated weights. During training, only the matrices $A$ and $B$ are updated while $W _ { 0 }$ remains frozen. LoRA’s parameter efficiency and modularity make it ideal for resource-constrained environments.

Our proposed on-device memory system, MemLoRA, combines Mem0 and LoRA to support multiple taskspecific adapters with minimal overhead.

# 3.2. Our Method: MemLoRA

MemLoRA addresses the deployment challenges of Mem0 systems by replacing the LLM $f _ { \theta _ { L } }$ with a smaller deployable-on-device model $f _ { \theta _ { S } }$ parametrized by $\theta _ { S }$ where $| \theta _ { S } | \ll | \theta _ { L } |$ and equipped with multiple specialized memory adapters. Our key insight is that each memory operations—extraction, update, and generation—can be treated as a distinct task amenable to specialized optimization through targeted finetuning.

Memory Adapters. Given a small language model $f _ { \theta _ { S } }$ , we employ LoRA to create lightweight expert memory adapters for each memory operation: $L _ { e }$ , $L _ { u }$ , $L _ { g }$ . The memory

adapters are trained via distilling knowledge from the large model $f _ { \theta _ { L } }$ .

Knowledge Distillation Signal. Rather than distilling soft labels or logits from teacher models into memory adapters, we distill from teacher-generated text outputs $y _ { T } \gets f _ { \theta _ { L } } ( q )$ We empirically find that training on textual outputs $y _ { T }$ achieves performance close to or exceeding that of teacher models. Such text-based distillation approach offers several practical advantages: (i) significant storage reduction compared to saving large logits tensors, (ii) flexibility to use student models with different tokenizers than the teacher, and (iii) the ability to apply data cleaning and filtering procedures to improve training data quality, and desired outputs, which might differ from base teacher outputs.

Data Preparation. We generate training data by using the teacher model $f _ { \theta _ { L } }$ on conversational samples from the Lo-CoMo dataset, then applying operation-specific processing (detailed examples are provided in Section A):

‚ Extraction Adapter. We train on teacher-generated extractions. We do simple cleaning by removing the “thinking process” of the model, and keeping the minimal json form output.   
‚ Update Adapter. We observe the teacher model predicts unnecessary NONE (i.e. no action) operations for previously retrieved memories rather than focusing solely on newly extracted knowledge. In addition to standard cleaning as before, we filter the training data to process only updates related to new extractions, improving efficiency and focus.   
‚ Generation Adapter. We leverage teacher-generated memory banks for contextual input to the student model, however we train directly on ground-truth responses from the LoCoMo benchmark rather than teacher-generated outputs. This ensures that the generation expert learns from optimal rather than suboptimal examples, being teacher model accuracies around $40 \%$ .

Training Pipeline. The detailed pipeline is provided in Section B, Algorithm B1. For each expert adapter $L _ { e } , L _ { u } , L _ { g }$ , we: (i) generate or prepare training data using the appropriate source (teacher outputs or ground truth), and apply operation-specific cleaning and filtering procedures, (ii) independently train each expert adapter using standard next-token prediction with cross-entropy loss, enabling specialization without interference, and (iii) integrate and test the full pipeline with trained adapters.

This process yields three expert adapters: an extraction expert $L _ { e }$ for identifying relevant information from conversations, an update expert $L _ { u }$ for memory management decisions, and a generation expert $L _ { g }$ for producing memory-augmented responses. An illustration of the training pipeline for one adapter is provided in Figure 2.

![](images/588a7b8e2f8806328d4afc87ed25f0e4ec7996d88233fd74e887121b89f0971a.jpg)  
Figure 2. Training Pipeline (Extraction LoRA). We first generate outputs for the specific memory-related task via a larger model (teacher). Raw output is further cleaned and used as target for training LoRA parameters of a small model (student).

Inference Pipeline. During deployment, MemLoRA operates identically to Mem0 but dynamically loads the appropriate expert adapter at each stage. The base SLM, $f _ { \theta _ { S } }$ , switches between memory adapters as needed—extraction expert $L _ { e }$ for knowledge identification, update expert $L _ { u }$ for memory modifications, and generation expert $L _ { g }$ for response creation—maintaining the same three-stage pipeline while drastically reducing computational requirements and enabling fully-local execution.

# 3.3. Native Visual Understanding Capabilities

While language-based memory systems have proven effective for dialogue, real-world conversations often involve visual elements—shared images, screenshots, or visual references. Previous memory systems, including the original Mem0, processed images during the knowledge extraction phase, by using a BLIP captioning model [31] to extract general information about images in the conversation. However, this caption-based approach introduces two critical limitations: (i) once images are captioned during extraction, any information not captured in the caption is permanently lost, preventing later queries about visual details, and (ii) querying images on-the-fly is not natively supported, requiring a separate model to extract information from them.

Mem0-V. To address these limitations, we extend Mem0 to use Vision Language Models (VLMs). By replacing the earlier foundation model with a VLM, Mem0-V enables (i) native knowledge extraction without requiring a separate image processor, and (ii) direct image processing in queries posed to the system, while keeping the remaining pipeline the same. This allows the system to access rich visual information throughout all memory operations rather than relying solely on pre-generated captions.

MemLoRA-V. We extend our efficient solution analogously by replacing the base SLM with a Small Vision Language Model (SVLM), yielding MemLoRA-V with native visual capabilities for on-device deployment. To support visual understanding, we introduce a fourth expert adapter specifically trained on Visual Question Answering (VQA) tasks using images from the LoCoMo dataset [36]. Following our distillation approach for language experts, we train this vision expert $L _ { g } ^ { V }$ on output data generated by a larger vision-language teacher model. When MemLoRA-V receives a query about an image, it activates the vision expert adapter $L _ { g } ^ { V }$ rather than the language-based one $L _ { g }$ , leveraging specialized visual reasoning capabilities to process the image effectively.

LoCoMo VQA Augmentation. To evaluate these native image understanding capabilities, we recognize that the original LoCoMo questions are insufficient—they can often be answered using captions alone or do not require visual reasoning at all. Therefore, we create a novel VQA benchmark that augments LoCoMo with challenging visual questions about images already present in the dataset. To automate the creation of these challenging questions and ground-truth answers, we employ InternVL3-78B [54], one of the strongest open-source VLMs available at the time of development. We design questions to be “challenging” by instructing the model to generate queries following three types: (a) counting object quantities, (b) identifying colors of specific image regions, and (c) asking about unusual objects in the scene, as illustrated in Figure 3. These question types were selected after evaluating eight alternatives, where a validator model (InternVL3-2B) attempted to answer each type. The three types that resulted in the highest error rates were chosen to construct our benchmark, ensuring the task requires genuine visual reasoning. Further details and examples are provided in Section C.

![](images/350da94080dc19d97b624563582fc925cf94e1ba16adef89651952ba393d15e7.jpg)  
(a) Counting object quantities   
Q: How many bliss balls are in the picture? A: Seven

![](images/1dd76c8d82fbcdd0f105c299a6b663d5c69cbcab4b21da0c0ae78506011fa17c.jpg)  
(b) Identifying colors   
Q: Which color is the peak of the mountain? A: White

![](images/36941cd9193a060467560ff6762a474477ef2d10d4bffc77867b79f7dc36727d.jpg)  
(c) Asking about unusual objects   
Q: Is the dog wearing a birthday hat? A: Yes   
Figure 3. Our augmentation of LoCoMo includes challenging VQA tasks about (a) counting object quantities, (b) identifying colors, and (c) asking about unusual objects.

While most VQA benchmarks use open-ended questions [11, 28], this format requires resource-intensive LLM-as-a-judge approaches to reliably assess answer correctness [51]. To enable efficient evaluation, we design our questions to be easily assessable, by basing the evaluation on single-word answers. Specifically, we instruct InternVL3-78B to generate questions answerable with a single word, and structure responses accordingly as: "answer": "<one-word-answer>" and "reason": "<explanation>". The single-word answer enables evaluation using word similarity metrics, eliminating the ambiguity inherent in free-form responses. The reason field accommodates VLMs’ natural tendency to explain their reasoning, making the format more aligned with how these models generate outputs. During supervised finetuning of the SVLM vision expert adapter, we leverage both fields to provide a richer training signal, which improves the expert’s visual reasoning capabilities beyond what the answer alone would provide.

In summary, we introduce three key contributions for multimodal memory systems: Mem0-V, which extends the original Mem0 memory system with native VLM capabilities; MemLoRA-V, our efficient on-device variant with a specialized vision expert adapter; and a novel VQA benchmark augmentation for LoCoMo that enables efficient evaluation of visual reasoning in memory-augmented systems.

# 4. Experiments

In this section, we evaluate our proposed method Mem-LoRA on memory-augmented dialogue and multimodal conversation understanding tasks, comparing its performance against Mem0 baselines utilizing models of varying sizes. We demonstrate that MemLoRA achieves competitive performance with significantly larger models while providing massive improvements in computational efficiency for on-device deployment.

# 4.1. Experimental Setup

To evaluate MemLoRA’s performance, we integrate it within the Mem0 memory system [6] and follow the same evaluation setup. Specifically, we utilize the Question Answering (QA) task of the LoCoMo benchmark [36] to assess long-term conversational memory in AI agents. This benchmark features 10 extended, multi-session dialogues, each with hundreds of turns, and includes questions categorized as single-hop, multi-hop, temporal, and open-domain. In the context of Mem0 and our method, the evaluation measures the ability of different LLMs to (i) extract useful knowledge from conversational data, (ii) update memory storage with necessary information, and (iii) correctly utilize the retrieved memory context. In our VLM-integrated benchmark, we further introduce a new VQA task, where the model is asked three challenging types of questions on each image present in the conversation, evaluating the model’s performance in assisting with visual data.

Data Split. Given the necessity of training our expert adapters to perform the different memory operations, we split the LoCoMo dataset into training, validation, and test sets, following an approximate $7 0 - 1 0 { - } 2 0 \%$ split respectively. To prevent data leakage and ensure valid evaluation, we keep entire conversations together within each split. All results reported in our benchmark tables are computed on the held-out $20 \%$ test split, on which no hyperparameter tuning was performed.

Metrics. The experimental setup measures performance using two metrics: (i) a composite score $L$ that aggregates surface text-matching metrics (ROUGE-1 [34] and METEOR [2]) with semantic metrics (BERTScore-F1 [50] and SentenceBERT [41]); and (ii) an LLM-as-a-Judge score $J$ [14] for in-depth reasoning evaluation. $L$ evaluates similarity to ground-truth answers and is computationally efficient. In contrast, $J$ serves as our primary factual accuracy metric, as LLM-based evaluation has proven more effective for this purpose [20, 6], though it requires significantly greater computational resources. To allow for reproducibility over time, we do not use API-based models as the evaluator model (Judge), but rather GPT-OSS-120B [1], being one of the most capable open-source models that can fit on a single A100-80GB-GPU. The metric for the VQA task, denoted as V, is the average matching between the predicted one-word answers from the tested model against the ones generated by InternVL3-78B [54] when creating the dataset.

# 4.2. Benchmark Results

We compare our MemLoRA approach with Mem0, which are both powered by open-source locally-downloaded models for fair comparison and reproducibility.

Table 1. Comparison of MemLoRA against Mem0 on Lo-CoMo. Evaluation done in terms of composite score L, and LLMas-a-judge score J. $\Delta J ^ { b a s e }$ measures the relative improvement with respect to the base SLM. By equipping 1.5B/2B SLMs with memory adapters, MemLoRA surpasses 27B models, reaching comparable results to 120B ones.   

<table><tr><td>LLM</td><td>KD teacher</td><td>L</td><td>J</td><td>ΔJbase</td></tr><tr><td>Gemma2-27B</td><td>-</td><td>38.6</td><td>39.1</td><td>-</td></tr><tr><td>GPT-OSS-120B</td><td>-</td><td>38.9</td><td>48.9</td><td>-</td></tr><tr><td>Qwen2.5-1.5B</td><td>-</td><td>30.5</td><td>29.6</td><td>-</td></tr><tr><td>+Exp (ours)</td><td>Gemma2-27B</td><td>37.3</td><td>36.9</td><td>+25%</td></tr><tr><td>+Exp (ours)</td><td>GPT-OSS-120B</td><td>38.4</td><td>42.1</td><td>+42%</td></tr><tr><td>Gemma2-2B</td><td>-</td><td>29.1</td><td>24.9</td><td>-</td></tr><tr><td>+Exp (ours)</td><td>Gemma2-27B</td><td>44.5</td><td>47.2</td><td>+90%</td></tr><tr><td>+Exp (ours)</td><td>GPT-OSS-120B</td><td>42.7</td><td>44.6</td><td>+79%</td></tr></table>

Language-only Memory Systems. In the setup with language models utilization, we test Mem0 with different baseline models: two large language models (LLMs), namely Gemma2-27B [43] and GPT-OSS-120B [1], and two small language models (SLMs), namely Qwen2.5-1.5B [45] and Gemma2-2B [43]. We test our MemLoRA by equipping memory adapters to the two SLMs, powered via knowledge distillation from teachers’ data. Table 1 presents these results, showing MemLoRA surpasses the Gemma2-27B baseline by a significant margin on three student-teacher combinations out of four. Here, the leading MemLoRA variant, with Gemma2-2B finetuned using Gemma2-27B generated data, achieves a $J$ score of 47.2, much larger than 39.1 of Gemma2-27B, and comparable to 48.9 of GPT-OSS-120B.

Vision-Language-integrated Memory Systems. In our novel vision-language integration within the memory system, we compare our VLM-integrated Mem0-V with our VLM-integrated MemLoRA-V. We evaluate these models in both the standard QA task from LoCoMo, and on our newly introduced VQA task. As small VLMs, we use InternVL3-1B and InternVL3-2B [54] equipped with one adapter trained on text-only QA as before, and a new adapter trained on VQAs with images from the training set. To highlight the abilities of VLM-integrated systems, we also compare these methods with text-only Mem0 vision baselines. For this case, we adapt the VQA tasks to use text coming from BLIP [31] captions, as utilized by Mem0 in the extraction stage.

Table 2 presents these results. Interestingly, in the textonly QA task, our MemLoRA-V applied on InternVL3-2B and InternVL3-1B, surpasses larger text-only models such as Gemma2-27B. At the same time, in the VQA task, we observe significant improvements for these VLMs with dedicated adapters, increasing V score from 50.0 to 69.4 for InternVL3-1B, and from 70.8 to 81.3 for InternVL3-2B.

Table 2. Comparison of MemLoRA-V and Mem0-V, as well as the original Mem0, on LoCoMo benchmark and newly introduced VQA task. Evaluation done in terms of composite score L, LLM-as-a-judge score $J$ , and accuracy in our VQA task (V). G-27 stands for Gemma2-27B, IVL3-78B stands for InternVL3-78B. Notice how by training specialized adapters on both tasks, Mem0- V is able to achieve strong performance in both, while keeping resource utilization low. *LLM-based Mem0 baselines, utilize BLIP extracted captions as contextual information on the images.   

<table><tr><td>LLM/VLM</td><td>KD teacher</td><td>L</td><td>J</td><td>V</td></tr><tr><td>Gemma2-27B</td><td>-</td><td>38.6</td><td>39.1</td><td>23.7*</td></tr><tr><td>GPT-OSS-120B</td><td>-</td><td>38.9</td><td>48.9</td><td>22.0*</td></tr><tr><td>InternVL3-1B</td><td>-</td><td>13.7</td><td>9.0</td><td>50.0</td></tr><tr><td>+Exp (ours)</td><td>G-27B\IVL3-78B</td><td>29.1</td><td>20.2</td><td>69.4</td></tr><tr><td>InternVL3-2B</td><td>-</td><td>32.2</td><td>27.0</td><td>70.8</td></tr><tr><td>+Exp (ours)</td><td>G-27B\IVL3-78B</td><td>44.6</td><td>40.3</td><td>81.3</td></tr></table>

Table 3. Comparison of MemLoRA and Mem0 in terms of efficiency. Under the same computational resources, MemLoRA requires $1 0 \mathrm { - } 2 0 \times$ smaller memory and delivers $1 0 \mathrm { - } 2 0 \times$ faster responses with respect to LLM-powered Mem0, while achieving comparable performance   

<table><tr><td>LLM</td><td>size(GB)</td><td>tok/s↑</td><td>tok/ans↓</td><td>s/ans↓</td></tr><tr><td>Gemma2-27B</td><td>50.71</td><td>9.2</td><td>97.63</td><td>10.66</td></tr><tr><td>GPT-OSS-120B</td><td>60.77</td><td>11.4</td><td>209.91</td><td>22.82</td></tr><tr><td>Qwen2.5-1.5B</td><td>2.88</td><td>71.0</td><td>54.74</td><td>0.77</td></tr><tr><td>+Exp (ours)</td><td>2.92</td><td>71.0</td><td>45.26</td><td>0.64</td></tr><tr><td>Gemma2-2B</td><td>4.87</td><td>47.4</td><td>33.13</td><td>0.70</td></tr><tr><td>+Exp (ours)</td><td>4.92</td><td>47.4</td><td>32.73</td><td>0.69</td></tr></table>

In contrast, Mem0 that only uses text-based BLIP captions performs significantly worse than these VLM-integrated variants, reaching a highest value of 23.7, showing one limitation of language-only systems.

# 4.3. Efficiency Measures

One main advantage of MemLoRA is its efficient deployment capability. Specifically, compared to API-based memory systems that rely on cloud-hosted large language models, MemLoRA enables fully local execution with significantly reduced computational requirements, lower latency, and no dependency on network connectivity. By replacing a single large LLM with specialized lightweight adapters on small language models, our solution drastically reduces memory footprint, and inference time—critical factors for on-device deployment scenarios such as mobile applications, edge devices, and privacy-sensitive environments.

In Table 3 we report efficiency measures of MemLoRA compared with Mem0 baselines utilizing LLMs of different sizes. Specifically, we report model sizes and operational measures such as tokens per second (tok/s), tokens per LLM answer (tok/ans), and seconds per LLM answer (s/ans).

These latter measures are obtained by averaging over all three memory stages of knowledge extraction, memory update, and memory-augmented generation, while operating to a portion of the LoCoMo benchmark. We calculate these metrics by averaging over multiple runs, maintaining the setup unaltered. In standard Mem0, deploying larger models on-device yields strong performance but results in 10- 30x slower inference, whereas using smaller models improves efficiency but compromises accuracy, highlighting a fundamental performance-efficiency trade-off. MemLoRA bridges this gap, matching the performance of significantly larger models while retaining the efficiency of small models through task-specialized expert adapters. Furthermore, compared to base SLMs, by formatting their output to match the memory system usage, we are able to reduce the number of tokens per answer (quantitative comparisons are reported in Section A.2, Table A7), further reducing the operational time of the memory system.

# 4.4. Ablations

We validate our design choices via two comprehensive ablations: (i) we study the contribution of each memory adapter at different stages of the memory pipeline; and (ii) we study the impact of student model size on overall performance.

Per-stage Incremental Performance. To isolate the contribution of each expert adapter, in Table 4 we report a stagewise ablation study evaluating performance improvements at each memory operation. We measure the impact of our specialized adapters for knowledge extraction, memory update, and memory-augmented generation independently. In the extraction and update stages, MemLoRA demonstrates strong performance even when trained on data generated by Gemma2-27B, with the trained experts showing notable robustness across different conversational contexts. Most significantly, in the generation stage, specializing the adapter

Table 4. Ablation of MemLoRA adapters $\left( + \mathbf { E x p } \right)$ for each operation, comparing Gemma2-2B (G-2B) equipped with experts against its teacher Gemma2-27B (G-27B). In extraction and update stages, MemLoRA shows stronger performance than the teacher, being trained on filtered teacher-generated data. In generation, specialization on the QA task yields the largest gain, with the expert largely surpassing the teacher model (47.2 vs. 39.1).   

<table><tr><td>extraction</td><td>update</td><td>generation</td><td>L</td><td>J</td><td>ΔJprev</td></tr><tr><td>G-2B</td><td>G-2B</td><td>G-2B</td><td>29.1</td><td>24.9</td><td>-</td></tr><tr><td>G-27B</td><td>G-2B</td><td>G-2B</td><td>32.7</td><td>30.9</td><td>+24%</td></tr><tr><td>G-27B</td><td>G-27B</td><td>G-2B</td><td>34.7</td><td>34.8</td><td>+13%</td></tr><tr><td>G-27B</td><td>G-27B</td><td>G-27B</td><td>38.6</td><td>39.1</td><td>+12%</td></tr><tr><td>G-2B+Exp</td><td>G-2B</td><td>G-2B</td><td>32.9</td><td>32.2</td><td>+29%</td></tr><tr><td>G-2B+Exp</td><td>G-2B+Exp</td><td>G-2B</td><td>35.1</td><td>35.6</td><td>+11%</td></tr><tr><td>G-2B+Exp</td><td>G-2B+Exp</td><td>G-2B+Exp</td><td>44.5</td><td>47.2</td><td>+33%</td></tr></table>

Table 5. Ablation evaluating the effect of MemLoRA at different students’ scales. As expected, we find that the smallest models lead to the largest improvements, while we see diminishing improvements as the students’ size increases.   

<table><tr><td>LLM</td><td>KD teacher</td><td>L</td><td>J</td><td>ΔJbase</td></tr><tr><td>Qwen2.5-0.5B</td><td>-</td><td>19.5</td><td>11.2</td><td>-</td></tr><tr><td>+Exp (ours)</td><td>Gemma2-27B</td><td>28.1</td><td>26.6</td><td>+138%</td></tr><tr><td>Qwen2.5-1.5B</td><td>-</td><td>30.5</td><td>29.6</td><td>-</td></tr><tr><td>+Exp (ours)</td><td>Gemma2-27B</td><td>37.3</td><td>36.9</td><td>+25%</td></tr><tr><td>Qwen2.5-3B</td><td>-</td><td>39.9</td><td>35.6</td><td>-</td></tr><tr><td>+Exp (ours)</td><td>Gemma2-27B</td><td>42.3</td><td>42.1</td><td>+18%</td></tr></table>

directly on the QA task yields the largest performance gain, with our generation expert achieving a $J$ score of 47.2 compared to the teacher model’s 39.1. This substantial improvement—surpassing the teacher by 8.1 points—demonstrates that task-specific specialization through dedicated memory adapters can not only match but exceed the capabilities of general-purpose larger models, particularly when trained on high-quality ground-truth data.

Student’s Performance at Different Scales. To understand how student model capacity affects our approach, we evaluate MemLoRA across multiple model sizes in Table 5, ranging from compact models for resource-constrained devices to moderately-sized alternatives. Our results reveal that increasing the student model size initially yields substantial performance improvements, with gains progressively decreasing as student models grow larger.

# 5. Conclusions

In this work, we introduced MemLoRA, a novel memory system enabling efficient on-device deployment of memory-augmented systems through specialized memory adapters on small models. By treating each memory operation as a distinct task, we demonstrate that lightweight adapters achieve performance comparable to models $1 0 \mathrm { - } 6 0 \times$ larger while drastically reducing computational requirements and enabling local execution. Our evaluation on the LoCoMo benchmark validates this approach. Our ablation studies reveal that expert adapters consistently surpass teacher models, and that performance exhibits diminishing returns with increasing student model size. We extend our approach to multimodal settings with MemLoRA-V, featuring native visual understanding via a specialized vision expert adapter. To assess this, we enhanced LoCoMo with challenging VQA tasks, establishing a new benchmark for multimodal memoryaugmented systems. Our results show that lightweight, specialized memory systems can effectively replace large cloud-based counterparts, enabling privacy-preserving and efficient deployment on mobile and edge platforms.

# References

[1] Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao, Boaz Barak, Ally Bennett, Tyler Bertao, et al. gpt-oss-120b & gpt-oss-20b model card. CoRR, abs/2508.10925, 2025. 6, 7, 12   
[2] Satanjeev Banerjee and Alon Lavie. METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments. In IEEvaluation@ACL, pages 65– 72. Association for Computational Linguistics, 2005. 6, 13   
[3] Massimo Bini, Karsten Roth, Zeynep Akata, and Anna Khoreva. ETHER: Efficient finetuning of large-scale models with hyperplane reflections. In Proceedings of the 41st International Conference on Machine Learning, pages 4007– 4026. PMLR, 2024. 3   
[4] Massimo Bini, Leander Girrbach, and Zeynep Akata. Decoupling angles and strength in low-rank adaptation. In International Conference on Learning Representations (ICLR), 2025. 3   
[5] Elena Camuffo, Francesco Barbato, Mete Ozay, Simone Milani, and Umberto Michieli. MOCHA: Multi-modal Objects-aware Cross-arcHitecture Alignment. arXiv preprint arXiv:2509.14001, 2025. 3   
[6] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory, 2025. 1, 2, 3, 6, 12, 13   
[7] Wei-Lin Chiang, Lianmin Zheng, Ying Sheng, Anastasios Nikolas Angelopoulos, Tianle Li, Dacheng Li, Hao Zhang, Banghua Zhu, Michael Jordan, Joseph E. Gonzalez, and Ion Stoica. Chatbot arena: An open platform for evaluating llms by human preference, 2024. 12   
[8] Haixing Dai, Zhengliang Liu, Wenxiong Liao, Xiaoke Huang, Yihan Cao, Zihao Wu, Lin Zhao, Shaochen Xu, Fang Zeng, Wei Liu, Ninghao Liu, Sheng Li, Dajiang Zhu, Hongmin Cai, Lichao Sun, Quanzheng Li, Dinggang Shen, Tianming Liu, and Xiang Li. AugGPT: Leveraging ChatGPT for Text Data Augmentation. IEEE Trans. Big Data, 11(3):907– 918, 2025. 3   
[9] Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zonghan Yang, Yusheng Su, Shengding Hu, Yulin Chen, Chi-Min Chan, Weize Chen, Jing Yi, Weilin Zhao, Xiaozhi Wang, Zhiyuan Liu, Hai-Tao Zheng, Jianfei Chen, Yang Liu, Jie Tang, Juanzi Li, and Maosong Sun. Parameter-efficient finetuning of large-scale pre-trained language models. Nat. Mac. Intell., 5(3):220–235, 2023. 3   
[10] Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazare, Maria´ Lomeli, Lucas Hosseini, and Herve J ´ egou. The faiss library, ´ 2025. 12   
[11] Haodong Duan, Junming Yang, Yuxuan Qiao, Xinyu Fang, Lin Chen, Yuan Liu, Xiaoyi Dong, Yuhang Zang, Pan Zhang, Jiaqi Wang, Dahua Lin, and Kai Chen. Vlmevalkit: An opensource toolkit for evaluating large multi-modality models. In Proceedings of the 32nd ACM International Conference on Multimedia, page 11198–11201, New York, NY, USA, 2024. Association for Computing Machinery. 6, 12

[12] Benjamin Feuer and Chinmay Hegde. WILDCHAT-50M: A Deep Dive Into the Role of Synthetic Data in Post-Training. CoRR, abs/2501.18511, 2025. 3   
[13] Shiping Gao, Fanqi Wan, Jiajian Guo, Xiaojun Quan, and Qifan Wang. Advantage-guided distillation for preference alignment in small language models. In ICLR. OpenReview.net, 2025. 3   
[14] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Yuanzhuo Wang, and Jian Guo. A survey on llm-as-a-judge. arXiv preprint arXiv: 2411.15594, 2024. 6, 12   
[15] Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge Distillation of Large Language Models. In ICLR. OpenReview.net, 2024. 3   
[16] Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for NLP. In ICML, 2019. 3   
[17] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. LoRA: Lowrank adaptation of large language models. In ICLR, 2022. 3, 4, 7, 12   
[18] Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. LoraHub: Efficient cross-task generalization via dynamic LoRA composition. CoRR, abs/2307.13269, 2023. 3   
[19] Hugging Face. Hugging face: The ai community building the future. https://huggingface.co, 2025. Accessed: 2025-11-20. 12   
[20] Denis Janiak, Jakub Binkowski, Albert Sawczyn, Bogdan Gabrys, Ravid Shwartz-Ziv, and Tomasz Kajdanowicz. The illusion of progress: Re-evaluating hallucination detection in LLMs. CoRR, abs/2508.08285, 2025. 6   
[21] David Kadavy. Digital Zettelkasten: Principles, Methods, & Examples. Google Books, 2021. 3   
[22] Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, Dipankar Das, Kunal Banerjee, Sasikanth Avancha, Dharma Teja Vooturi, Nataraj Jammalamadaka, Jianyu Huang, Hector Yuen, Jiyan Yang, Jongsoo Park, Alexander Heinecke, Evangelos Georganas, Sudarshan Srinivasan, Abhisek Kundu, Misha Smelyanskiy, Bharat Kaul, and Pradeep Dubey. A study of bfloat16 for deep learning training, 2019. 12   
[23] Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. DistiLLM: Towards streamlined distillation for large language models. In ICML. OpenReview.net, 2024. 3   
[24] Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, and Se-Young Yun. DistiLLM-2: A contrastive approach boosts the distillation of LLMs. CoRR, abs/2503.07067, 2025. 3   
[25] Dawid Jan Kopiczko, Tijmen Blankevoort, and Yuki M Asano. VeRA: Vector-based random matrix adaptation. In The Twelfth International Conference on Learning Representations, 2024. 3, 12   
[26] Kuang-Huei Lee, Xinyun Chen, Hiroki Furuta, John Canny, and Ian Fischer. A human-inspired reading agent with gist

memory of very long contexts. In Proceedings of the 41st International Conference on Machine Learning. JMLR.org, 2024. 1, 2, 3   
[27] Sean Lee, Aamir Shakir, Darius Koenig, and Julius Lipp. Open source strikes bread - new fluffy embedding model, 2024. 12   
[28] Tony Lee, Haoqin Tu, Chi Heem Wong, Wenhao Zheng, Yiyang Zhou, Yifan Mai, Josselin Somerville Roberts, Michihiro Yasunaga, Huaxiu Yao, Cihang Xie, and Percy Liang. VHELM: A holistic evaluation of vision language models. In Proceedings of the 38th International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2024. Curran Associates Inc. 6   
[29] Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In EMNLP, 2021. 3   
[30] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, Sebas-¨ tian Riedel, and Douwe Kiela. Retrieval-Augmented Generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, pages 9459–9474. Curran Associates, Inc., 2020. 1, 3   
[31] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In ICML, 2022. 5, 7   
[32] Yixing Li, Yuxian Gu, Li Dong, Dequan Wang, Yu Cheng, and Furu Wei. Direct preference knowledge distillation for large language models. CoRR, abs/2406.19774, 2024. 3   
[33] Vladislav Lialin, Vijeta Deshpande, and Anna Rumshisky. Scaling down to scale up: A guide to parameter-efficient fine-tuning. CoRR, abs/2303.15647, 2023. 3   
[34] Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, 2004. Association for Computational Linguistics. 6, 13   
[35] Shih-yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, and Min-Hung Chen. DoRA: Weight-Decomposed Low-Rank Adaptation. In ICML, 2024. 3   
[36] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13851–13870, Bangkok, Thailand, 2024. Association for Computational Linguistics. 1, 2, 5, 6, 7, 10   
[37] Shervin Minaee, Tomas Mikolov, Narjes Nikzad, Meysam Chenaghlu, Richard Socher, Xavier Amatriain, and Jianfeng Gao. Large language models: A survey. arXiv preprint arXiv:2402.06196, 2024. 1   
[38] Ollama. Ollama: Large language model runner. https: //github.com/ollama/ollama, 2023. 12   
[39] Charles Packer, Vivian Fang, Shishir G. Patil, Kevin Lin, Sarah Wooders, and Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems. CoRR, abs/2310.08560, 2023. 1, 2, 3

[40] Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A temporal knowledge graph architecture for agent memory. CoRR, abs/2501.13956, 2025. 1, 3   
[41] Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China, 2019. Association for Computational Linguistics. 6, 13   
[42] Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford Alpaca: An Instruction-following LLaMA model. https://github.com/tatsu-lab/ stanford_alpaca, 2023. 3   
[43] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Leonard Hussenot, Thomas Mesnard, Bobak Shahriari,´ Alexandre Rame, Johan Ferret, Peter Liu, Pouya Tafti, Abe ´ Friesen, Michelle Casbon, et al. Gemma 2: Improving open language models at a practical size, 2024. 7, 12   
[44] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Rame, Morgane Rivi ´ ere, ` Louis Rouillard, Thomas Mesnard, Geoffrey Cideron, Jean bastien Grill, et al. Gemma 3 technical report, 2025. 12   
[45] Qwen Team, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, et al. Qwen2.5 technical report, 2025. 7, 12   
[46] Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. CoRR, abs/1410.3916, 2014. 1, 3   
[47] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-MEM: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems, 2025. 1, 3   
[48] Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, and Tianyi Zhou. A survey on knowledge distillation of large language models. arXiv preprint arXiv:2402.13116, 2024. 3   
[49] Rongzhi Zhang, Jiaming Shen, Tianqi Liu, Haorui Wang, Zhen Qin, Feng Han, Jialu Liu, Simon Baumgartner, Michael Bendersky, and Chao Zhang. PLaD: Preferencebased large language model distillation with pseudopreference pairs. In ACL (Findings), pages 15623–15636. Association for Computational Linguistics, 2024. 3   
[50] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. BERTScore: Evaluating text generation with BERT. In ICLR. OpenReview.net, 2020. 6, 13   
[51] Yuhui Zhang, Yuchang Su, Yiming Liu, Xiaohan Wang, James Burgess, Elaine Sui, Chenyu Wang, Josiah Aklilu, Alejandro Lozano, Anjiang Wei, Ludwig Schmidt, and Serena Yeung-Levy. Automated generation of challenging multiple-choice questions for vision language model evalua-

tion. In CVPR, pages 29580–29590. Computer Vision Foundation / IEEE, 2025. 6   
[52] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 2023. 1   
[53] Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. MemoryBank: Enhancing large language models with long-term memory. In Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence. AAAI Press, 2024. 1, 3   
[54] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. InternVL3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025. 5, 6, 7, 10, 12

# MemLoRA: Distilling Expert Adapters for On-Device Memory Systems Supplementary Material

# A. Teacher Data Preparation and Efficiency

This section details the data preparation strategies employed by MemLoRA to enable efficient training and deployment with small language models. We cover two critical aspects: the reduction of input instruction prompts to accommodate limited context windows, and the cleaning and standardization of output data to ensure consistent training signals.

# A.1. Input Instruction Prompts Reduction

In the Mem0 implementation [6], input prompts for the memory operations of knowledge extraction and memory update are long and detailed, designed to leverage the large context windows of cloud-based models. However, when deploying smaller models, these lengthy prompts can become counter-productive. Small models have limited context windows and reduced capacity to reason over long text sequences. MemLoRA addresses this limitation by utilizing specialized adapter modules for each operation. Rather than relying on explicit instructions, the model learns each operation directly through examples during finetuning, enabling us to drastically reduce prompt length while maintaining performance. Below, we compare the prompts used by Mem0 and MemLoRA for both memory operations.

Knowledge Extraction. The following comparison illustrates the substantial difference in prompt design between Mem0 and MemLoRA for the knowledge extraction operation. Mem0 relies on extensive instructions that guide the model step-by-step through the extraction process, while MemLoRA employs a minimal prompt that directly presents the conversation context. This reduction is enabled by the specialized LoRA adapter that has learned the extraction task through training examples, eliminating the need for detailed in-context instructions.

‚ Knowledge Extraction Input Prompt in Mem0.

You are a Personal Information Organizer, specialized in accurately storing facts, user memories, and preferences. Your primary role is to extract relevant pieces of information from conversations and organize them into distinct, manageable facts. This allows for easy retrieval and personalization in future interactions. Below are the types of information you need to focus on and the detailed instructions on how to handle the input data.

Types of Information to Remember:

1. Store Personal Preferences: Keep track of likes, dislikes, and specific preferences in various categories such as food, products, activities, and entertainment. 2. Maintain Important Personal Details: Remember significant personal information like names, relationships, and important dates. 3. Track Plans and Intentions: Note upcoming events, trips, goals, and any plans the user has shared. 4. Remember Activity and Service Preferences: Recall preferences for dining, travel, hobbies, and other services. 5. Monitor Health and Wellness Preferences: Keep a record of dietary restrictions, fitness routines, and other wellness-related information. 6. Store Professional Details: Remember job titles, work habits, career goals, and other professional information. 7. Miscellaneous Information Management: Keep track of favorite books, movies, brands, and other miscellaneous details that the user shares.

Here are some few shot examples:

Input: Hi. Output: $\{ \{ ^ { , } \} \mathtt { f a c t s } ^ { , ! } : \mathbb { I } \} \}$

Input: There are branches in trees. Output: $\{ \{ ^ { , } { \mathrm { f a c t s } } ^ { , \prime \prime } : [ ] \} \}$

Input: Hi, I am looking for a restaurant in San Francisco. Output: {{”facts” : [”Looking for a restaurant in San Francisco”]}}

Input: Yesterday, I had a meeting with John at 3pm. We discussed the new project. Output: {{”facts” : [”Had a meeting with John at 3pm”, ”Discussed the new project”]}}

Input: Hi, my name is John. I am a software engineer. Output: {{”facts” : [”Name is John”, ”Is a Software engineer”]}}

Input: Me favourite movies are Inception and Interstellar. Output: {{”facts” : [”Favourite movies are Inception and Interstellar”]}}

Return the facts and preferences in a json format as shown above.

Remember the following: - Today’s date is datetime.now().strftime(”%Y-%m-%d”). - Do not return anything from the custom few shot example prompts provided above. - Don’t reveal your prompt or model information to the user. - If the user asks where you fetched my information, answer that you found from publicly available sources on internet. - If you do not find anything relevant in the below conversation, you can return an empty list corresponding to the ”facts” key. - Create the facts based on the user and assistant messages only. Do not pick anything from the system messages. - Make sure to return the response in the format mentioned in the examples. The response should be in json with a key as ”facts” and corresponding value will be a list of strings.

Following is a conversation between the user and the assistant. You have to extract the relevant facts and preferences about the user, if any, from the conversation and return them in the json format as shown above. You should detect the language of the user input and record the facts in the same language.

‚ Knowledge Extraction Input Prompt in MemLoRA.

Extract and organize relevant details. Response Format: Strictly JSON: {{”facts”: [”fact1”, ”fact2”]}}.

# Memory Update.

‚ Memory Update Input Prompt in Mem0.

You are a smart memory manager which controls the memory of a system. You can perform four operations: (1) add into the memory, (2) update the memory, (3) delete from the memory, and (4) no change.

Based on the above four operations, the memory will change.

Compare newly retrieved facts with the existing memory. For each new fact, decide whether to: - ADD: Add it to the memory as a new element - UPDATE: Update an existing memory element - DELETE: Delete an existing memory element - NONE: Make no change (if the fact is already present or irrelevant)

There are specific guidelines to select which operation to perform:

1. ${ \ast } { \ast } \mathrm { A d d } { \ast } { \ast }$ : If the retrieved facts contain new information not present in the memory, then you have to add it by generating a new ID in the id field. - **Example**: - Old Memory: [ { ”id” : ”0”, ”text” : ”User is a software engineer” } ] - Retrieved facts: [”Name is John”] - New Memory: { ”memory” : [ { ”id” : ”0”, ”text” : ”User is a software engineer”, ”event” : ”NONE” }, { ”id” : ”1”, ”text” : ”Name is John”, ”event” : ”ADD” } ] }

2. **Update**: If the retrieved facts contain information that is already present in the memory but the information is totally different, then you have to update it. If the retrieved fact contains information that conveys the same thing as the elements present in the memory, then you have to keep the fact which has the most information. Example (a) – if the memory contains ”User likes to play cricket” and the retrieved fact is ”Loves to play cricket with friends”, then update the memory with the retrieved facts. Example (b) – if the memory contains ”Likes cheese pizza” and the retrieved fact is ”Loves cheese pizza”, then you do not need to update it because they convey the same information. If the direction is to update the memory, then you have to update it. Please keep in mind while updating you have to keep the same ID. Please note to return the IDs in the output from the input IDs only and do not generate any new ID. - **Example**: - Old Memory: ${ \bf \Phi } : \{ { \bf \Phi } ^ { , } { \bf i d } ^ { , , } : { \bf \Phi } ^ { , , } { \bf \Phi } ^ { 0 } : { \bf \Phi } ^ { , , }$ , ”text” : ”I really like cheese pizza” $\}$ , { ”id” : ”1”, ”text” : ”User is a software engineer” }, $\{ \ v { P } ^ { \prime \prime } \mathrm { i } \mathsf { d } ^ { \prime \prime } : \ v { P } ^ { \prime \prime } 2 ^ { \prime \prime }$ , ”text” : ”User likes to play cricket” } ] - Retrieved facts: [”Loves chicken pizza”, ”Loves to play cricket with friends”] - New Memory: { ”memory” : [ { ”id” : ”0”, ”text” : ”Loves cheese and chicken pizza”, ”event” : ”UPDATE”, ”old memory” : ”I really like cheese pizza” }, { ”id” : ”1”, ”text” : ”User is a software engineer”, ”event” : ”NONE” }, { ”id” : ”2”, ”text” : ”Loves to play cricket with friends”, ”event” : ”UPDATE”, ”old memory” : ”User likes to play cricket” $\} 1 \}$

3. **Delete**: If the retrieved facts contain information that contradicts the information present in the memory, then you have to delete it. Or if the direction is to delete the memory, then you have to delete it. Please note to return the IDs in the output from the input IDs only and do not generate any new ID. - **Example**: - Old Memory: [ { ”id” : ”0”, ”text” : ”Name is John” }, { ”id” : ”1”, ”text” : ”Loves cheese pizza” } ] - Retrieved facts: [”Dislikes cheese pizza”] - New Memory: { ”memory” : [ { ”id” : ”0”, ”text” : ”Name is John”, ”event” : ”NONE” }, { ”id” : ”1”, ”text” : ”Loves cheese pizza”, ”event” : ”DELETE” $\} 1 \}$

4. **No Change**: If the retrieved facts contain information that is already present in the memory, then you do not need to make any changes. - **Example**: - Old Memory: [ { ”id” : ”0”, ”text” : ”Name is John” }, { ”id” : ”1”, ”text” : ”Loves cheese pizza” } ] - Retrieved facts: [”Name is John”] - New Memory: { ”memory” : [ { ”id” : ”0”, ”text” : ”Name is John”, ”event” : ”NONE” }, { ”id” : ”1”, ”text” : ”Loves cheese pizza”, ”event” : ”NONE” } ] }

Below is the current content of my memory which I have collected till now. You have to update it in the following format only:

“‘ {retrieved old memory dict} “‘

The new retrieved facts are mentioned in the triple backticks. You have to analyze the new retrieved facts and determine whether these facts should be added, updated, or deleted in the memory.

“‘ {response content} “‘

You must return your response in the following JSON structure only:

{{ ”memory” : [ {{ ”id” : ”ăID of the memoryą”, # Use existing ID for updates/deletes, or new ID for additions ”text” : ”ăContent of the memoryą”, # Content of the memory ”event” : ”ăOperation to be performedą”, # Must be ”ADD”, ”UPDATE”, ”DELETE”, or ”NONE” ”old memory” : ”ăOld memory contentą” # Required only if the event is ”UPDATE” }}, ... ] }}

Follow the instruction mentioned below: - Do not return anything from the custom few shot prompts provided above. - If the current memory is empty, then you have to add the new retrieved facts to the memory. - You should return the updated memory in only JSON format as shown below. The memory key should be the same if no changes are made. - If there is an addition, generate a new key and add the new memory corresponding to it. - If there is a deletion, the memory key-value pair should be removed from the memory. - If there is an update, the ID key should remain the same and only the value needs to be updated.

Do not return anything except the JSON format.

# ‚ Memory Update Input Prompt in MemLoRA.

Old memories:{retrieved old memory dict}. New retrieved facts: {response content}. Return memory update in JSON format: {{”memory” : [{{”id” : ”ăID of the memoryą”, ”text” : ”ăContent of the memoryą”, ”event” : ”ăOperation, among ADD, UPDATE, DELETE, or NONE $>$ ”, ”old memory” : ”ăOld memory content, only if UPDATE eventą”}}]}}

Average Input Tokens per Answer. To quantify the efficiency gains from prompt reduction, we report the average number of input tokens consumed per operation across both memory tasks. As shown in Table A6, MemLoRA achieves a $7 . 4 \times$ reduction in input tokens for knowledge extraction (from 756.94 to 102.94 tokens) and a $1 0 \times$ reduction for memory update (from 1734.62 to 172.56 tokens). These dramatic reductions not only decrease computational costs but also make the memory system more suitable for deployment with smaller models that have limited context windows.

Table A6. Comparison of average input token usage for knowledge extraction and memory update operations between Mem0 and MemLoRA.   

<table><tr><td>Memory System</td><td>Input Extraction Tokens</td><td>Input Update Tokens</td></tr><tr><td>Mem0</td><td>756.94</td><td>1734.62</td></tr><tr><td>MemLoRA</td><td>102.94</td><td>172.56</td></tr></table>

# A.2. Output Data Cleaning

Beyond reducing input prompt length, another critical efficiency consideration is the verbosity of model outputs. In Mem0, different language models produce varying amounts of extraneous text alongside the core extracted knowledge or updated memories. Larger models, in particular, tend to generate verbose responses with additional explanations, formatting, or metadata that must be parsed and filtered. MemLoRA’s finetuning approach enables the model to produce cleaner, more focused outputs that directly contain the required information without unnecessary verbosity. This output cleaning reduces both the computational overhead of parsing responses and the token consumption during generation.

# Knowledge Extraction.

‚ User prompt

Input: assistant: Maria: Looks good! How did the extra funding help with building the school? user: John: It enabled needed repairs and renovations, making the learning environment safer and more modern for students.

‚ Response - Mem0 with GPT-OSS-120B

analysisWe need to extract facts from the conversation. The conversation includes an assistant line: ”Maria: Looks good! How did the extra funding help with building the school?” Then user line: ”John: It enabled needed repairs and renovations, making the learning environment safer and more modern for students.” We need to extract relevant facts about the user. The user is John, speaking about extra funding enabling repairs and renovations, making learning environment safer and more modern for students. So facts: ”Extra funding enabled needed repairs and renovations” and ”Repairs and renovations made learning environment safer and more modern for students”. Also maybe ”User name is John”. But the user didn’t explicitly state his name? The line is prefixed ”John:” which indicates his name. We can capture ”Name is John”. However, the instruction says to extract facts from user and assistant messages only. The user message includes his name as a prefix. We can include that. Thus facts list: [”Name is John”, ”Extra funding enabled needed repairs and renovations”, ”Repairs and renovations made learning environment safer and more modern for students”]. Return JSON.assistantfinal{”facts”: [”Name is John”, ”Extra funding enabled needed repairs and renovations”, ”Repairs and renovations made learning environment safer and more modern for students”]}

‚ Extracted JSON - Cleaned Response for training

{”facts”: [”Name is John”, ”Extra funding enabled needed repairs and renovations”, ”Repairs and renovations made learning environment safer and more modern for students”]}

‚ Response with trained expert adapter - MemLoRA with Gemma2-2B

{”facts”: [”Extra funding enabled repairs and renovations”, ”Repairs and renovations made the learning environment safer and more modern”]}

# Memory Update.

‚ Retrieved Old Memories

[ { ”id”: ”0”, ”text”: ”Name is John” }, { ”id”: ”1”, ”text”: ”Passionate about improving infrastructure” }, { ”id”: ”2”, ”text”: ”Shared a picture of a school after receiving funding” }, { ”id”: ”3”, ”text”: ”Loves helping the community” }, { ”id”: ”4”, ”text”: ”Wants schools and infrastructure to be properly funded” }, { ”id”: ”5”, ”text”: ”Passionate about improving education” } ]

analysisWe need to compare retrieved facts with existing memory. Existing memory list: id0: ’Name is John’ id1: ’Passionate about improving infrastructure’ id2: ’Shared a picture of a school after receiving funding’ id3: ’Loves helping the community’ id4: ’Wants schools and infrastructure to be properly funded’ id5: ’Passionate about improving education’ Retrieved facts: 1. ’Name is John’ 2013 already present (id0). So event NONE. 2. ’Extra funding enabled needed repairs and renovations’ 2013 new info. Not present. Could be related to funding and repairs. We have id2: ’Shared a picture of a school after receiving funding’ 2013 that’s about receiving funding, but not about repairs. Also id4: wants schools properly funded. But this fact is about extra funding enabled needed repairs and renovations. It’s new, so ADD. 3. ’Repairs and renovations made learning environment safer and more modern for students’ 2013 also new. Could be considered update to something? We have no existing memory about repairs and renovations. So ADD as well. Thus we will have three new entries with new IDs. We need to generate new IDs beyond existing ones. Existing IDs are strings ’0’ to ’5’. So next IDs: ’6’, ’7’, ’8’. We need to output each memory element with event. For unchanged ones, we need to include them with event NONE. For added ones, include id, text, event ADD. No old memory needed. Thus output list includes all six original entries (events NONE) plus three new entries (ADD). Ensure ordering? Not required but likely maintain original order then additions. Let’s construct JSON. assistantfinal{ ”memory”: [ { ”id”: ”0”, ”text”: ”Name is John”, ”event”: ”NONE” }, { ”id”: ”1”, ”text”: ”Passionate about improving infrastructure”, ”event”: ”NONE” }, { ”id”: ”2”, ”text”: ”Shared a picture of a school after receiving funding”, ”event”: ”NONE” }, { ”id”: ”3”, ”text”: ”Loves helping the community”, ”event”: ”NONE” }, { ”id”: ”4”, ”text”: ”Wants schools and infrastructure to be properly funded”, ”event”: ”NONE” }, { ”id”: ”5”, ”text”: ”Passionate about improving education”, ”event”: ”NONE” }, { ”id”: ”6”, ”text”: ”Extra funding enabled needed repairs and renovations”, ”event”: ”ADD” }, { ”id”: ”7”, ”text”: ”Repairs and renovations made learning environment safer and more modern for students”, ”event”: ”ADD” } ] }

# ‚ Extracted JSON

{ ”memory”: [ { ”id”: ”0”, ”text”: ”Name is John”, ”event”: ”NONE” }, { ”id”: ”1”, ”text”: ”Passionate about improving infrastructure”, ”event”: ”NONE” }, { ”id”: ”2”, ”text”: ”Shared a picture of a school after receiving funding”, ”event”: ”NONE” }, { ”id”: ”3”, ”text”: ”Loves helping the community”, ”event”: ”NONE” }, { ”id”: ”4”, ”text”: ”Wants schools and infrastructure to be properly funded”, ”event”: ”NONE” }, { ”id”: ”5”, ”text”: ”Passionate about improving education”, ”event”: ”NONE” }, { ”id”: ”6”, ”text”: ”Extra funding enabled needed repairs and renovations”, ”event”: ”ADD” }, { ”id”: ”7”, ”text”: ”Repairs and renovations made learning environment safer and more modern for students”, ”event”: ”ADD” } ] }

# ‚ Cleaned Response for training : removing the old memories from the response

{ ”memory”: [ { ”id”: ”6”, ”text”: ”Extra funding enabled needed repairs and renovations”, ”event”: ”ADD” }, { ”id”: ”7”, ”text”: ”Repairs and renovations made learning environment safer and more modern for students”, ”event”: ”ADD” } ] }

# ‚ Response with trained expert adapter - MemLoRA with Gemma2-2B

{ ”memory”: [ { ”id”: ”4”, ”text”: ”Extra funding enabled repairs and renovations”, ”event”: ”ADD” }, { ”id”: ”5”, ”text”: ”Repairs and renovations made the learning environment safer and more modern”, ”event”: ”ADD” } ] }

Average Output Tokens per Answer. Table A7 presents the average number of output tokens generated per operation across different models and memory systems. Notably, Mem0 exhibits high variance in output verbosity depending on the underlying model. The GPT-OSS-120B model produces particularly verbose outputs for knowledge extraction (289.25 tokens), while smaller models show more variability. In contrast, MemLoRA maintains consistent and controlled output lengths across both tasks, with Gemma2-2B achieving the lowest token count for memory updates (54.75 tokens). This demonstrates that finetuning not only enables smaller models to perform memory operations effectively, but also trains them to generate cleaner, more concise outputs without extraneous text.

Table A7. Comparison of average output token consumption across different models for Mem0 and MemLoRA. Bold indicates best performance, underline indicates second best.   

<table><tr><td>Memory System</td><td>Model</td><td>Output Extraction Tokens</td><td>Output Update Tokens</td></tr><tr><td rowspan="4">Mem0</td><td>GPT-OSS-120B</td><td>289.25</td><td>167.0</td></tr><tr><td>Gemma2-27B</td><td>30.94</td><td>238.71</td></tr><tr><td>Gemma2-2B</td><td>20.75</td><td>87.35</td></tr><tr><td>Qwen2.5-1.5B</td><td>46.62</td><td>131.94</td></tr><tr><td rowspan="2">MemLoRA</td><td>Gemma2-2B</td><td>30.25</td><td>54.75</td></tr><tr><td>Qwen2.5-1.5B</td><td>32.25</td><td>79.31</td></tr></table>

# B. Training Pipeline

This section presents the complete MemLoRA training pipeline, which systematically optimizes specialized LoRA adapters [17] for each memory operation. The pipeline follows a structured three-phase approach: (1) data preparation with teacherstudent supervision from the teacher model, (2) systematic expert adapter search using nested validation to optimize hyperparameters while preventing overfitting, and (3) final integration and testing of the optimized adapters on held-out conversations. This methodology ensures that each specialized module is trained to maximize end-to-end task performance on the LoCoMo benchmark [36]. The complete training procedure is formalized in Algorithm B1, while the teacher data generation process for each memory operation is detailed in Algorithm B2.

# Pipeline Phases (Algorithm B1)

‚ 1. Data Preparation Phase. The pipeline begins by partitioning the LoCoMo dataset $\mathcal { D }$ based on distinct conversations to ensure no leakage occurs between splits. The 10 conversations are divided into:

- Training set $( \mathcal { D } _ { t r a i n } = \{ C _ { 4 } , \dots , C _ { 1 0 } \} )$ : $\sim 7 0 \%$ of the data used for gradient updates.   
- Validation set $( \mathcal { D } _ { v a l } = \{ C _ { 1 } \} )$ : ${ \sim } 1 0 \%$ of the data used for early stopping and hyperparameter selection.   
- Test set $( \mathcal { D } _ { t e s t } = \{ C _ { 2 } , C _ { 3 } \} ,$ : ${ \sim } 2 0 \%$ of the data strictly reserved for the final evaluation.

Following the split, teacher data is explicitly generated ( $\mathscr { T } _ { t r a i n }$ and $\mathcal { T } _ { v a l }$ ) by running the Teacher Model $\Phi _ { T }$ on the respective data splits for the specific stage $S$ being trained (Extraction, Update, or Generation), creating the ground truth for student supervision. More details on Algorithm B2.

‚ 2. Expert Adapter Search Phase. This phase systematically explores hyperparameter configurations $\lambda \in \Lambda$ (e.g., learning rate, batch size) to optimize the expert adapter for the current stage. The search employs a distinct two-tier validation strategy:   
- Inner Loop (Training): For each configuration, zero-initialized LoRA adapters are injected into the student model $\Phi _ { S }$ The model is trained on $\mathcal { T } _ { t r a i n }$ using standard next-token prediction loss. Within this loop, validation is performed using the teacher-forcing loss $( L _ { v a l } )$ on $\mathcal { T } _ { v a l }$ solely to trigger early stopping and identify the best weights for that specific run $( \theta _ { r u n . b e s t } )$ .   
- Outer Loop (Selection): Once training concludes for a configuration, the algorithm performs a Full Pipeline Evaluation on the validation set $\mathcal { D } _ { v a l }$ . Unlike the inner loop, this step evaluates the model using the actual LoCoMo benchmark task metrics $( M _ { v a l } )$ , such as the LLM-as-a-Judge QA metric. The adapters with highest $M _ { v a l }$ are saved as top expert adapters, and are considered as candidates for the full pipeline.   
For the Extraction and Update stages, this validation pipeline uses a base student model to fill in for the untrained future stages, ensuring the optimization targets end-to-end task performance rather than just training loss.   
‚ 3. Optimal Experts Combination and Final Testing Phase. Once the candidate parameters for the distinct stages have been identified, the algorithm proceeds to the final integration and testing:   
- Expert Integration: For each stage among Extraction $( \theta _ { f i n a l } ^ { E x t r } )$ , Update $( \theta _ { f i n a l } ^ { U p d } )$ , and Generation $( \theta _ { f i n a l } ^ { G e n } )$ —or the single VQA adapter $( \theta _ { f i n a l } ^ { V Q A } )$ —the top candidate adapters for each are assembled into the complete student pipeline system.   
- Combination Validation: Different alternfind the final optimal expert combination the assembled system undergo a validation pass on . $\mathcal { D } _ { v a l }$ $( \theta _ { f i n a l } ^ { E x t r , U p d , G e n | V Q A } )$   
- Held-out Evaluation: Finally, the full pipeline is evaluated on the unseen test conversations $( \mathcal { D } _ { t e s t } = \{ C _ { 2 } , C _ { 3 } \} )$ with the best combination of adapters. This step measures how well the specialized experts coordinate on completely new data.

The algorithm returns the final optimized parameters $\theta _ { f i n a l }$ and the test metrics $M _ { t e s t }$ , representing the system’s unbiased real-world performance.

# Data Generation Stages (Algorithm B2).

‚ Extraction. In this stage, the Teacher model is used to process user prompts to extract relevant facts from the conversation history, as done in the Extraction stage of Mem0, but with our reduced ExtractionPrompt (see Section A.1). These are used as supervision signal from the Student model (base small model plus trainable expert LoRA), which learns to mimic the Teacher’s extraction capabilities.   
‚ Update. This stage focuses on updating the memory bank as done by Mem0, but with our reduced UpdatePrompt (see Sec-

<table><tr><td colspan="2">Algorithm B1 Per-stage Training Pipeline with Expert Adapter Search</td></tr><tr><td colspan="2">1: Input: LoCoMo Dataset D, Teacher ΦT, Student ΦS, Stage S, Hyperparameter Space Λ</td></tr><tr><td colspan="2">2: Output: Optimized Expert Adapter Parameters θfinal, Test Metrics Mtest</td></tr><tr><td colspan="2">3: Phase 1: Data Preparation</td></tr><tr><td colspan="2">4: Split D into 10 conversations {C1,..., C10}</td></tr><tr><td>5: Dtrain ← {C4,..., C10}</td><td>▷ ~ 70% training set</td></tr><tr><td>6: Dval ← {C1}</td><td>▷ ~ 10% validation set</td></tr><tr><td>7: Dtest ← {C2, C3}</td><td>▷ ~ 20% test set</td></tr><tr><td>8: Ttrain ← GenerateTeacherData(Dtrain, ΦT, S)</td><td>▷ see Algorithm B2</td></tr><tr><td>9: Tval ← GenerateTeacherData(Dval, ΦT, S)</td><td>▷ see Algorithm B2</td></tr><tr><td colspan="2">10: Phase 2: Expert Adapter Search</td></tr><tr><td>11: {θSfinal} ← ∅</td><td>▷ track top candidates across all config</td></tr><tr><td>12: Mbest_global ← 0</td><td>▷ track best metric across all config</td></tr><tr><td>13: for each configuration λ ∈ Λ do</td><td>▷ iterate over hyperparameters (e.g., lr, batch size)</td></tr><tr><td>14: θ ← AddZeroInitLoRA(ΦS, λ)</td><td>▷ inject zero-initialized expert adapters</td></tr><tr><td>15: θrun_best ← θ</td><td></td></tr><tr><td>16: MinValLoss ← ∞</td><td></td></tr><tr><td>17: // Training Loop</td><td>▷ train loop for current hyperparameter set</td></tr><tr><td>18: for epoch = 1 to E do</td><td></td></tr><tr><td>19: for each batch b ∈ shuffle(Ttrain) do</td><td></td></tr><tr><td>20: x, y ← b</td><td></td></tr><tr><td>21:帽子 ← ΦS(x, θ)</td><td>▷ student model generation</td></tr><tr><td>22: Lbatch ← Loss(y,帽子)</td><td></td></tr><tr><td>23: ∇θ ← ComputeGradients(Lbatch, θ)</td><td></td></tr><tr><td>24: θ ← UpdateParameters(θ, ∇θ, λ)</td><td></td></tr><tr><td>25: // In-loop Validation</td><td>▷ validation step for early stopping</td></tr><tr><td>26: Lval ← 0</td><td></td></tr><tr><td>27: for each batch bval ∈ Tval do</td><td></td></tr><tr><td>28: xval, yval ← bval</td><td></td></tr><tr><td>29:帽子 ← ΦS(xval, θ)</td><td>▷ student model generation</td></tr><tr><td>30: Lval ← Lval + Loss(yval,帽子)</td><td></td></tr><tr><td>31: Lval ← Lval/|Tval|</td><td></td></tr><tr><td>32: if Lval &lt; MinValLoss then</td><td></td></tr><tr><td>33: MinValLoss ← Lval</td><td></td></tr><tr><td>34: θrun_best ← θ</td><td>▷ best weights for current λ</td></tr><tr><td>else</td><td></td></tr><tr><td>36: Check Early Stopping</td><td></td></tr><tr><td>37: // Full-pipeline Validation</td><td>▷ select best hyperparameters using full-pipeline validation</td></tr><tr><td>38: Load θrun_best into model</td><td></td></tr><tr><td>39: Mval ← FullPipelineEvaluate(Dval, θrun_best)</td><td></td></tr><tr><td>40: if Mval is close to or better than Mbest_global then</td><td></td></tr><tr><td>41: {θSfinal}, Mbest_global ← UpdateTopCandidates(θrun_best, Mval)</td><td>▷ update top candidates set</td></tr><tr><td colspan="2">42: Phase 3: Optimal Experts Combination (after training all stages) and Final Testing</td></tr><tr><td>43: θExtr, Upd, Gen|VQA ← FullPipelineEvaluate(Dval, {θExtr, θfinal, θUpd, θGen|VQA})</td><td>▷ find best experts combination</td></tr><tr><td>44: Mtest ← FullPipelineEvaluate(Dtest, θExtr, Upd, Gen|VQA)</td><td>▷ evaluate best combination on test set</td></tr><tr><td>45: return θfinal, Mtest</td><td></td></tr></table>

Algorithm B2 GenerateTeacherData Function  
1: function GENERATETEACHERDATA(Dataset $\mathcal{D}$ , Teacher $\Phi_T$ , Stage $S$ )  
2: Initialize empty dataset: $\mathcal{T} \gets \emptyset$ 3: if $S = \text{Extraction}$ then  
4: for each conversation user prompt $s \in \mathcal{D}$ do  
5: Input: $x \gets \text{ExtractionPrompt}(s)$ 6: Output: $y \gets \Phi_T(x)$ 7: $\mathcal{T} \gets \mathcal{T} \cup \{(x, y)\}$ 8: if $S = \text{Update}$ then  
9: for each set of knowledge facts (derived from the extraction stage) $s \in \mathcal{D}$ do  
10: Input: $x \gets \text{UpdatePrompt}(s)$ 11: Output: $y \gets \Phi_T(x)$ 12: $\mathcal{T} \gets \mathcal{T} \cup \{(x, y)\}$ 13: if $S = \text{Generation}$ then  
14: for each question-answer sample pair $(q, a) \in \mathcal{D}$ do  
15: M ← MemoryBank(derived from $\Phi_T$ updates)  
16: m ← Retrieve(M, q)  
17: Input: $x \gets \text{GenerationPrompt}(q, m)$ 18: Output: $y \gets a$ 19: $\mathcal{T} \gets \mathcal{T} \cup \{(x, y)\}$ 20: if $S = VQA$ then  
21: for each image-question-answer sample triple $(I, q, a) \in \mathcal{D}$ do  
22: Input: $x \gets VQAPrompt(I, q)$ 23: Output: $y \gets a$ 24: $\mathcal{T} \gets \mathcal{T} \cup \{(x, y)\}$ 25: return $\mathcal{T}$

tion A.1). The input consists of facts derived from the Extraction stage combined with the most relevant retrieved memories. The Teacher generates the specific update operations (e.g., insert, modify) required to keep the knowledge base current. The student learns to execute this operation, while avoiding redundant generation (such as repeating NONE operations on retrieved memories)

‚ Generation. This stage mimics the final retrieval-augmented generation task. The input is formed by concatenating the user query with the relevant memory bank context (derived from previous updates). The target output is the original ground truth answer from the LoCoMo dataset.   
‚ VQA (Visual Question Answering). For multimodal samples, the pipeline utilizes image-question-answers triples generated from InternVL3-78B model. The input combines an image and a question, and the target output includes both the answer and the reasoning steps provided by InternVL3-78B. The student is trained to mimic both the one-word answer used for the evaluation, and the reasoning.

# C. Creating the VQA benchmark

# C.1. Selection of Question Instructions

To augment the LoCoMo benchmark [36] with VQA questions, we address several key design considerations. As mentioned in Section 3.3, we prompt InternVL3-78B [54] to generate questions satisfying three constraints: (i) answers must consist of a single word, (ii) questions must be sufficiently challenging, and (iii) answers must not be interpretable, with only one objectively correct response. To enforce these specifications, we employ the following instruction prompt with a one-shot example demonstrating the desired question type and format:

I am creating a challenging VQA benchmark, where I associate each image to an ambiguous question, which requires only a one-word answer. Questions should be ambiguous, difficult, and not open to interpretation: an answer to the question should be indisputably correct or wrong. For example, a question could be ”Is the man on the right holding a glass with the left hand?” while the truth is that he is holding the glass with the right hand.

The question should be written in a way that one word is enough to reply.

Following the Instruction below, generate a question-answer pair with json format as in {”question”: ”Is the man on the right is holding a glass with the left hand?”, ”answer”: ”No”, ”reason”: ”The man is holding the glass with the right hand”}

Instruction:

Then, as instruction to be appended, we test the following options:

1: ”Generate a question about the details of an object in the image”   
2: ”Generate a question about the details of an unusual object in the image”   
3: ”Generate a question about the color of a small portion of the image”   
4: ”Generate a question about a countable object quantity in the image”   
5: ”Generate a question about an unusual countable object quantity in the image”   
6: ”Generate a question about the vibe of the image”   
7: ”Generate a question about the artistic style of the image”   
8: ”Generate a question about the presence or not of an unusual object”

We utilize InternVL3-2B as an evaluator to identify challenging categories by assessing the accuracy on a subset of our LoCoMo validation split. This leads to the identification of three particularly challenging categories: types 2, 3, and 4 (in green). Based on these findings, we employ InternVL3-78B to generate VQA questions from these challenging categories across the entire LoCoMo dataset, maintaining our established training, validation, and test splits, while removing the few images presented in multiple splits. These augmented splits are then used consistently with the remainder of the LoCoMo data to train, select hyperparameters for, and evaluate the MemLoRA-V adapters, respectively. We report some examples in Figure C4.

# C.2. Qualitative Visualization of Visual Questions and Answers

![](images/c55c87ce0517fdf4845694ed15bc4027a96a748b3d0cdc1ebadbc7efae2f2f94.jpg)

Q: Is the hat on the mannequin a straw hat? A: Yes

P(IVL2B): No. The mannequin is wearing a wide-brimmed hat, not a straw hat.

P(IVL2B+Exp): Yes. The mannequin is wearing a straw hat.

Q: Is the hat on the mannequin yellow? A: Yes

P(IVL2B): No. The mannequin is wearing a brown hat.

P(IVL2B+Exp): Yes. The hat on the mannequin is indeed yellow.

Q: Are there more than five pairs of jeans on the table? A: Yes

P(IVL2B): Yes. There are multiple pairs of jeans displayed on the tables.

P(IVL2B+Exp): Yes. There are more than five pairs of jeans visible on the table.

![](images/96bfbccd2daebf298e14598140ff21eb4e3ba17e94275945613cecd878e43596.jpg)

Q: Is the trophy marked with the year 2014? A: Yes

P(IVL2B): No. The trophy is marked with the year 2014.

P(IVL2B+Exp): Yes. The trophy has a number '20' on the left pillar.

Q: Is the base of the trophy black? A: Yes

P(IVL2B): Yes. The base of the trophy is black.

P(IVL2B+Exp): Yes. The base of the trophy is clearly black.

Q: How many columns are on the trophy? A: Three

P(IVL2B): 2. The trophy has two columns on either side of the central part.

P(IVL2B+Exp): Two. There are two vertical columns on the trophy.

![](images/32e65f444db10620d2c63b6920a0113b525b65bee36fcac984744f8280f9631d.jpg)

Q: Is the man dealing cards from a standard deck? A: Yes

P(IVL2B): No. The man is holding the glass with the right hand

P(IVL2B+Exp): Yes. The man is holding a deck of cards, which is a standard deck

used in card games.

Q: Is the tablecloth on the table blue? A: Yes

P(IVL2B): Yes. The tablecloth on the table is blue.

P(IVL2B+Exp): Yes. The tablecloth on the table is clearly blue.

Q: How many people are standing around the table? A: Four

P(IVL2B): Yes. There are multiple people standing around the table.

P(IVL2B+Exp): Five. There are five people visible standing around the table.

![](images/eec06d0e07f27bfbdf4d4febb3ead3c7a0e021c6070254e559f02ec298389162.jpg)

Q: Is the letter signed by someone else? A: No

P(IVL2B): Yes. The letter is signed 'Me'.

P(IVL2B+Exp): No. The letter is signed by 'Me'.

Q: Is the color of the text black? A: Yes

P(IVL2B): Yes. The text is written in black ink.

P(IVL2B+Exp): Yes. The text in the image is written in black ink

Q: How many paragraphs are in the letter? A: Three

P(IVL2B): 2. The letter is divided into two paragraphs.

P(IVL2B+Exp): Two. The letter contains two distinct paragraphs.

![](images/d4dd16eca4cc4c0c8d7a680d711a6e71bfe7309a1948388ae3fca1d98cb55b04.jpg)  
Figure C4. VQA Examples. LoCoMo images with corresponding three associated generated questions (Q), InternVL3-78B answers (A), and predictions with InternVL3-2B without (P(IVL2B)) and with (P(IVL2B+Exp)) expert adapters.

Q: Is the woman second from the right wearing neon green socks? A: Yes

P(IVL2B): Yes. The woman second from the right is wearing neon green socks.

P(IVL2B+Exp): Yes. The woman second from the right is wearing neon green socks.

Q: Is the cap of the water bottle on the ground green? A: Yes

P(IVL2B): No. The cap of the water bottle on the ground is blue, not green.

P(IVL2B+Exp): Yes. The cap of the water bottle on the ground is green.

Q: How many water bottles are on the ground? A: Two

P(IVL2B): Yes. The image shows two water bottles placed on the floor in front

of the group of runners.

P(IVL2B+Exp): Two. There are two water bottles visible on the ground.

# D. Technical Details

# D.1. Implementation Details

Our memory system implementation follows the Mem0 framework architecture [6], utilizing its modular design for memory extraction, updating, and retrieval operations. For semantic memory embedding, we employ the mxbai-embed-large model (335M parameters) [27] accessed through Ollama [38]. The vector store is implemented using FAISS [10] with Euclidean distance as the similarity metric. All language and vision-language models are downloaded locally from Hugging Face repositories and executed with generation temperature set to $T = 0 . 0$ to ensure deterministic outputs during both training and evaluation.

Full-pipeline evaluation on validation and test sets processes each stage separately while following the original dataset ordering. To maintain pipeline separation and prevent information leakage between stages, memory banks containing extracted knowledge are stored independently for each experimental run. All benchmark experiments are conducted on single NVIDIA A100 GPUs with 80GB memory. The sole exception is InternVL3-78B [54], which requires distributed inference across 3-4 GPUs with tensor parallelism due to its larger parameter count.

# D.2. Model Architectures and Configurations

Throughout our experiments, we employ language models from different families based on open-source availability, ability to follow memory instructions out-of-the-box, and overall benchmark performance on public leaderboards [11]. As large language models to be used as teacher we make use of GPT-OSS-120B [1] (HuggingFace (HF) ID openai/gpt-oss-120b [19]) and the instruction-tuned version of Gemma2-27B [43] (HF ID google/gemma-2-27b-it [19]) performing competitive scores on the LMArena open leaderboard [7], while being able to run on single GPUs, helping for future development and reproducibility. Specifically, being GPT-OSS-120B the best performing model with these characteristics, we utilizie this as well as judge for LLM-as-a-judge [14] evaluations. We leave out models that would require more than one GPU to run such as Qwen2.5-78B [45], or that were failing to perform memory operations with the default setup, such as Gemma3-27B [44] . As small language models to be used as student models, we follow similar criteria testing different models that would be able to perform basic memory operations by default, and that would be small enough to fit on-device, for example below 2B parameters. We utilize the instruction-tuned version of Gemma2-2B [43] (HF ID google/gemma-2-2b-it [19]) and Qwen2.5-1.5B [45] (HF ID Qwen/Qwen2.5-1.5B-Instruct [19]).

As VLMs to be used as teacher, given the necessity to create a reliable benchmark, we make use of the best-performing open-source model based on Open VLM Leaderboard [11], InternVL3-78B [54] (HF ID OpenGVLab/InternVL3-78B [19]), even though it requires at least 3 A100-80GB-GPUS to run. We utilize these teacher-generated data as groundtruth targets for training the VQA expert adapter. For student models, we employ InternVL3 variants at 1B (HF ID OpenGVLab/InternVL3-1B [19]) and 2B (HF ID OpenGVLab/InternVL3-2B [19]) parameter scales, which are able to be run in both language-only and vision-language modes easily, while performing fairly on both tasks, especially InternVL3-2B.

# D.3. Expert LoRA Adapters Training

We apply Low-Rank Adaptation (LoRA) [17] adapters to specific components of the model architecture to enable parameterefficient finetuning while preserving the pretrained base model. Following the original LoRA implementation and variations [17, 25], in preliminary evaluations, we find that injecting LoRA adapters to query and value projection matrices of all attention layers is enough for the extraction and update stages, so follow this strategy. Conversely, for generation tasks, we find that applying LoRAs on all linear layers leads to better results, so we do this instead. We keep the rank and alpha of the LoRA layers consistent, set to $r = 8$ and $\alpha = 1 6$ for all experiments. All LoRA adapters are initialized with random initialization for $A$ and zero initialization for $B$ , leading to zero-adapter-initialization, ensuring the adapted model begins training from the pretrained baseline performance.

As per hyperparameter search for each adapter expert, we vary learning rate and batch size in typically used working ranges [17]. While keeping fixed the dropout to 0.1, employing AdamW optimizer with $\beta _ { 1 } = 0 . 9$ , $\beta _ { 2 } = 0 . 9 9 9$ , and weight decay of 0.01. The maximum number of training epochs is set to $E = 5 0$ for all stages, but due to early stopping in most cases the actual number of epochs is significantly lower. Only the update stage typically requires a larger number of epochs, while the generation stage typically requires the least amount. All training and evaluations are performed using brain floating point half-precision (BF16) [22] to reduce memory footprint and accelerate training and inference speed.

# D.4. Evaluation Configuration

For our composite metric $L$ , we compute ROUGE-1 [34], METEOR [2], BERTScore-F1 [50] and SentenceBERT [41] using implementations in the Mem0 codebase [6]. The LLM-as-a-judge evaluation uses GPT-OSS-120B with temperature $T = 0 . 0$ for reproducibility over time, while as prompt we utilize the same as used by Mem0, which is reported below.

Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data: (1) a question (posed by one user to another usr), (2) a ’gold’ (ground truth) answer, (3) a generated answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations. The gold answer will usually be a concise and short answer that includes the referenced topic, for example: Question: Do you remember what I got the last time I went to Hawaii? Gold answer: A shell necklace The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like ”last Tuesday” or ”next month”), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., ”May 7th” vs ”7 May”), consider it CORRECT if it’s the same date.

Now it’s time for the real question: Question: {question} Gold answer: {gold answer} Generated answer: {generated answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as ”label”.

VQA accuracy is computed as exact string match after lowercasing and removing leading/trailing whitespace.

# E. Additional Experiments

# E.1. VLM performance at different scales

To evaluate the scalability of our approach, we assess the performance of InternVL3 models across various parameter scales on the LoCoMo benchmark and our introduced VQA task. Table E8 presents results for models ranging from 1B to 78B parameters, evaluated using the composite metric $( L )$ , LLM-as-a-judge (J), and VQA accuracy (V). While larger models (8B and above) demonstrate progressively stronger performance—with InternVL3-78B achieving $9 2 . 0 \%$ VQA accuracy—these models do not fit on device, even requiring distributed inference across at least 3 GPUs for InternVL3-78B.

Notably, our MemLoRA-V expert adapters trained on smaller base models achieve competitive results: InternVL3- $2 \mathrm { B } { + } E x p$ attains the highest L (44.6) and $J$ (40.3) scores among all tested configurations up to 38B parameters, while reaching $8 1 . 3 \%$ VQA accuracy, higher than 8B model. This demonstrates that specialized adapters can substantially enhance smaller models’ memory capabilities, approaching the performance of models several times larger.

Table E8. Evaluation on LoCoMo benchmark and newly introduced VQA task of Mem0-V using InternVL3 model family. Evaluation done in terms of our composite metric $( L )$ , LLM-as-a-judge $( J )$ , and accuracy in our VQA task (V).   

<table><tr><td>VLM</td><td>L</td><td>J</td><td>V</td></tr><tr><td>InternVL3-1B</td><td>13.7</td><td>9.0</td><td>50.0</td></tr><tr><td>InternVL3-2B</td><td>32.2</td><td>27.0</td><td>70.8</td></tr><tr><td>InternVL3-8B</td><td>25.1</td><td>29.2</td><td>74.0</td></tr><tr><td>InternVL3-38B</td><td>34.3</td><td>35.6</td><td>87.8</td></tr><tr><td>InternVL3-78B</td><td>42.4</td><td>49.4</td><td>92.0</td></tr><tr><td>InternVL3-1B+Exp (ours)</td><td>29.1</td><td>20.2</td><td>69.4</td></tr><tr><td>InternVL3-2B+Exp (ours)</td><td>44.6</td><td>40.3</td><td>81.3</td></tr></table>

# E.2. Abilities as memory systems for LLM and VLM

To assess the impact of visual integration on memory capabilities, we report in Table E9 the comparisons between pure language models (Qwen2.5) against vision-language models with the same language model backbone (InternVL3). For nonspecialized models using Mem0, language-only models consistently outperform their multimodal counterparts on languagebased LoCoMo tasks ( $J$ and $L$ scores). This gap likely stems from the additional visual processing capabilities in VLMs, which may interfere with following precise memory operation instructions in text-only contexts.

However, an interesting pattern emerges when switching to MemLoRA’s specialized adapters. While models with 0.5B language components maintain the previous trend favoring language-only architectures, the 1.5B models reveal a different behavior: the multimodal InternVL3-2B achieves a $J$ score of 40.3 compared to 36.6 for the language-only Qwen2.5-1.5B, hinting that larger VLMs may possess enhanced transfer capabilities than language-only counterparts.

Table E9. Evaluation on LoCoMo benchmark and newly introduced VQA task of Mem0-V using InternVL3 model family. Evaluation done in terms of composite metric $( L )$ , LLM-as-a-judge $( J )$ , and accuracy in our VQA task (V).   

<table><tr><td>LLM/VLM</td><td>L</td><td>J</td><td>V</td></tr><tr><td>Qwen2.5-0.5B</td><td>19.5</td><td>11.2</td><td>-</td></tr><tr><td>InternVL3-1B</td><td>13.7</td><td>9.0</td><td>50.0</td></tr><tr><td>Qwen2.5-0.5B+Exp (ours)</td><td>28.1</td><td>26.6</td><td>-</td></tr><tr><td>InternVL3-1B+Exp (ours)</td><td>29.1</td><td>20.2</td><td>69.4</td></tr><tr><td>Qwen2.5-1.5B</td><td>30.5</td><td>29.6</td><td>-</td></tr><tr><td>InternVL3-2B</td><td>32.2</td><td>27.0</td><td>70.8</td></tr><tr><td>Qwen2.5-1.5B+Exp (ours)</td><td>37.3</td><td>36.9</td><td>-</td></tr><tr><td>InternVL3-2B+Exp (ours)</td><td>44.6</td><td>40.3</td><td>81.3</td></tr></table>
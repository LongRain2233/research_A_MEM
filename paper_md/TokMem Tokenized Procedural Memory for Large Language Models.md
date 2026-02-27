# TOKMEM: TOKENIZED PROCEDURAL MEMORY FOR LARGE LANGUAGE MODELS

Zijun $\mathbf { W } \mathbf { u } ^ { 1 }$ , Yongchang $\mathbf { H a o } ^ { 1 }$ , Lili Mou1,2

1Dept. Computing Science & Alberta Machine Intelligence Institute (Amii), University of Alberta 2Canada CIFAR AI Chair

{zijun4,yongcha1}@ualberta.ca, doublepower.mou@gmail.com

# ABSTRACT

Large language models rely heavily on prompts to specify tasks, recall knowledge and guide reasoning. However, this reliance is inefficient as prompts must be re-read at each step, scale poorly across tasks, and lack mechanisms for modular reuse. We introduce TokMem, a tokenized procedural memory that stores recurring procedures as compact, trainable embeddings. Each memory token encodes both an address to a procedure and a control signal that steers generation, enabling targeted behavior with constant-size overhead. To support continual adaptation, TokMem keeps the backbone model frozen, allowing new procedures to be added without interfering with existing ones. We evaluate TokMem on 1,000 tasks for atomic recall, and on function-calling tasks for compositional recall, where it consistently outperforms retrieval-augmented generation while avoiding repeated context overhead, and fine-tuning with far fewer parameters. These results establish TokMem as a scalable and modular alternative to prompt engineering and fine-tuning, offering an explicit procedural memory for LLMs.

# 1 INTRODUCTION

Large language models (LLMs) have become the foundation of modern natural language processing, powering a wide range of applications in text understanding, generation, and coding (Brown et al., 2020; Grattafiori et al., 2024). Prompting is a widely adopted way to steer LLM behavior, where in-context learning enables adaptation to new tasks without parameter updates (Brown et al., 2020). Consequently, prompt and context engineering has emerged as a dominant mechanism for specifying tasks, obtaining relevant information, and guiding multi-step reasoning or tool invocation (Wei et al., 2022b; Yao et al., 2023; Sahoo et al., 2025).

Despite its success, this reliance on long prompts is inherently inefficient. Constructing and maintaining prompts are labor-intensive and difficult to scale across many tasks (Liu et al., 2023). At inference, long prompts increase computational cost because the attention mechanism scales quadratically with sequence length (Vaswani et al., 2017), and they reduce the effective context window available for inputs and outputs, often leading to truncation and loss of details (Liu et al., 2024a). These limitations make it difficult to manage expanding tasks and to execute procedures efficiently.

To address these issues, recent approaches offload prompts into retrieval-based memory. Retrievalaugmented generation (RAG) (Lewis et al., 2020) and memory systems such as MemGPT(Packer et al., 2023) fetch and reinsert documents or conversational state at inference time. While retrieving in-context learning demonstrations (Wei et al., 2022b) can provide procedural cues that guide the model’s behavior, the mechanism still largely aligns with declarative memory in cognitive science: knowledge remains as explicit text that must be repeatedly interpreted. This creates two challenges: (1) retrieved content still occupies the context window, reintroducing quadratic compute and truncation pressure, and (2) frequently used procedures are repeatedly re-read as text rather than compiled into compact, reusable procedures, missing the compression opportunity suggested by minimum description length principles (Grunwald, 2007). ¨

We propose Tokenized Memory (TokMem), an explicit form of procedural memory that encodes recurring procedures as compact, trainable tokens while keeping the backbone frozen. Each memory

![](images/3ad1cbc5478f917a4f5274c7e871eaa55c9137e01feb47902d9da35f10ce0e58.jpg)  
Figure 1: Overview of TokMem. (a) New memory (colored) tokens are interleaved with text sequences, learning with next-token-prediction while the LLM backbone remains frozen. (b) An example of inference, a query recalls and chains memory tokens (parse, search, format), enabling multi-step procedural behavior without long prompts.

token serves both as an address to a procedure and as a control signal that steer generation, enabling targeted behavior with constant-size overhead. Specifically, rather than front-loading procedures as long prompts, TokMem integrates memory tokens directly into the generation process. As shown in Figure 1b, memory tokens can be invoked and chained across stages: after producing one response segment, the model retrieves the next relevant token, which conditions the next stage, enabling the composition of multi-step behaviors such as parsing, searching, and formatting.

A key advantage of TokMem is that its memory tokens are parameter-isolated from the backbone. This design ensures that the learned procedural knowledge is fully stored in dedicated tokens, allowing new procedures to be added without interfering with existing ones. TokMem thus naturally supports continual learning, where the model can accumulate procedural skills over time while preserving stability. This capability mirrors human procedural memory, where skills are gradually acquired through practice and later invoked by contextual cues (Anderson & Lebiere, 1998). In this way, TokMem enables both efficient learning and continual expansion of procedural knowledge.

We evaluate TokMem in two complementary settings. In the atomic memory recall setting, each task from Super-Natural Instructions (Wang et al., 2022a) is treated as a distinct procedure, Tok-Mem stores and retrieves 1,000 such procedures efficiently, without catastrophic forgetting. In the compositional memory recall setting based on function-calling tasks (Liu et al., 2024b), each tool invocation is modeled as an atomic procedure, and solving a query requires chaining multiple procedures together. TokMem supports this process by composing memory tokens, enabling the model to assemble procedures into coherent multi-step behaviors. Across both settings and multiple LLM backbones, TokMem consistently outperforms retrieval-based baselines and surpasses parametric fine-tuning while using far fewer trainable parameters.

# 2 METHOD

We begin by reviewing how Transformer-based LLMs process input sequences and then describe how TokMem departs from existing approaches to enable procedural memory.

# 2.1 TEXTUALIZED CONTEXT ENGINEERING

A Transformer (Vaswani et al., 2017) processes a sequence of tokens $( a _ { 1 } , \ldots , a _ { n } ) \in \mathbb { N } ^ { n }$ , where each $a _ { i }$ is an integer representing the index of a token (usually sub-words). The model retrieves

the corresponding embedding vector from the embedding layer and produces an input sequence $( \pmb { x } _ { 1 } , \dots , \pmb { \dot { x } } _ { n } ) \in \tilde { \mathbb { R } } ^ { n \times d }$ , which is then consumed to predict the next token in sequence.

Recent advances in prompting can be viewed as textualized context engineering, where the goal is to carefully choose input tokens that steer the model toward improved behavior. For example, chain-of-thought prompting (Wei et al., 2022a) augments input with intermediate reasoning steps to strengthen logical inference. Retrieval-based methods such as RAG (Lewis et al., 2020) and memory-augmented approaches like MemGPT (Packer et al., 2023) provide relevant information in text form by retrieving external memory.

Concretely, to evoke a procedural response $( r _ { 1 } , \ldots , r _ { n } )$ for a query $( q _ { 1 } , \dots , q _ { k } )$ , these methods may prepend textual context $\left( \boldsymbol { c } _ { 1 } , \ldots , \boldsymbol { c } _ { m } \right)$ that describes the procedure:

$$
\left(c _ {1}, \dots , c _ {m}, \quad q _ {1}, \dots , q _ {k}\right) \xrightarrow {\text {L L M}} \left(r _ {1}, \dots , r _ {n}\right), \quad m \gg k. \tag {1}
$$

While effective, these approaches are costly: the same procedural context must be re-read at every invocation, which can exhaust the limited context window (Agarwal et al., 2024).

# 2.2 TOKMEM: PROCEDURAL MEMORY AS A TOKEN

Our key idea is that frequently reused procedures can be effectively “compressed” and stored by encoding them into an internalized memory token, bypassing repeated textual specification. Specifically, we create a memory bank of $l$ special embeddings:

$$
M = \left[ \begin{array}{c} \boldsymbol {m} _ {1} ^ {\top} \\ \vdots \\ \boldsymbol {m} _ {l} ^ {\top} \end{array} \right] \in \mathbb {R} ^ {l \times d}, \quad \boldsymbol {m} _ {i} \in \mathbb {R} ^ {d}. \tag {2}
$$

Each $\mathbf { m } _ { i }$ is a trainable vector with no direct textual translation and represents a unique procedure. For simplicity, we label each memory token $\mathbf { m } _ { i }$ to have a special index $a _ { m _ { i } } \in \mathbb { N }$ . In contrast to textualized context engineering, TokMem evokes a procedural response without any explicit context:

$$
\left(q _ {1}, \dots , q _ {k}\right) \xrightarrow {\mathrm {L L M} + M} \left(r _ {1}, \dots , r _ {n}\right), \tag {3}
$$

where the model recalls the appropriate memory token $a _ { m _ { i } }$ internally based on the query. This eliminates redundant context engineering and achieves $O ( 1 )$ procedural invocation.

To connect these tokens with training, we define a procedure–response pair as the concatenation of a memory token $a _ { m _ { i } }$ and its corresponding response tokens $( r _ { i 1 } , \ldots , r _ { i n } )$ . Importantly, each training instance may contain multiple such pairs to model multi-stage reasoning or composition (Khot et al., 2023; Zhou et al., 2023). Formally, the sequentialized training sequence has the layout

$$
\boldsymbol {a} = \left(q _ {1}, \dots , q _ {k}, \underbrace {a _ {m _ {i}} , a _ {r _ {i 1}} , a _ {r _ {i 2}} , \dots} _ {\text {p r o c e d u r e - r e s p o n s e p a i r}}, \underbrace {a _ {m _ {j}} , a _ {r _ {j 1}} , a _ {r _ {j 2}} , \dots} _ {\text {p r o c e d u r e - r e s p o n s e p a i r}}, \dots\right). \tag {4}
$$

We adopt the standard next-token prediction loss:

$$
\mathcal {L} (\boldsymbol {a}; M) = - \sum_ {i > k} \log \Pr (a _ {i} \mid \boldsymbol {a} _ {<   i}; M). \tag {5}
$$

During optimization, only the memory embeddings $( m _ { 1 } , \hdots , m _ { l } )$ are updated, while the Transformer backbone remains frozen. We visualize our training process in Figure 1a.

# 2.3 STABILIZING NEW MEMORIES

TokMem allows new procedures to be added incrementally to the memory bank, mimicking how humans continually form new procedural memories without disrupting existing skills (Anderson & Lebiere, 1998; Squire, 2009). This design enables practical deployment scenarios, where an LLM can steadily accumulate routines across domains and tasks rather than retraining from scratch.

However, adding new tokens poses stability challenges. If all procedural memories are introduced at once, the model risks overfitting to spurious patterns. Conversely, when new embeddings are added

gradually, they often develop inflated norms that dominate routing logits and suppress older memories. To address this, we introduce renormalization, which is a lightweight post-update calibration to the memory bank M ∈ Rl×d. $\dot { M } \in \mathbb { R } ^ { l \times d }$

Let $A$ and $I$ denote the indices of the active (new) and inactive (existing) procedural memories, respectively. We estimate the prevailing scale from the inactive set:

$$
\bar {n} _ {I} = \frac {1}{| I |} \sum_ {j \in I} \| \boldsymbol {m} _ {j} \| _ {2}, \tag {6}
$$

and rescale each active embedding as

$$
\boldsymbol {m} _ {i} \leftarrow \boldsymbol {m} _ {i} \cdot \frac {\bar {n} _ {I}}{\| \boldsymbol {m} _ {i} \| _ {2} + \varepsilon}, \quad \forall i \in A. \tag {7}
$$

This operation preserves the directions of newly added embeddings while aligning their magnitudes to the established scale of the memory bank, ensuring smooth integration without overwhelming the routing dynamics. The computational overhead is negligible, scaling as $O ( | A | d )$ .

# 3 EXPERIMENTS

We evaluate TokMem in two complementary scenarios that test scalability and compositionality:

• Atomic Memory Recall: Each task from the Super-Natural Instructions dataset (Wang et al., 2022a) is framed as a standalone procedure, where a query directly maps to the desired response. This tests TokMem’s ability to store many independent procedures without catastrophic forgetting.   
• Compositional Memory Recall: Evaluated on the function-calling dataset (Liu et al., 2024b), where invoking a tool is treated as a procedure and solving a query requires composing several function calls. This tests whether learned memory tokens can be flexibly chained.

Experiments are conducted on Qwen (Qwen & et al., 2025) and Llama (Grattafiori et al., 2024) model families, ranging from the 0.5B-parameter Qwen to the 8B-parameter Llama model.

# 3.1 EXPERIMENTAL SETUP

Baselines. Across both settings, we compare TokMem with textualized context engineering, retrieval-augmented memory, and parameter-efficient fine-tuning.

• Base: In the atomic setting, the model answers queries without demonstrations, providing a nonparametric lower bound highlighting the need to recall task knowledge.   
• ICL: In the compositional setting, we augment input with all tool descriptions and prepend two compositional procedure–response demonstrations, representing a context engineering baseline.   
• RAG: We retrieve relevant demonstrations or tool usages with Sentence-BERT (Reimers & Gurevych, 2019) and prepend them to the query, following memory-augmented generation (Packer et al., 2023; Chhikara et al., 2025; Xu et al., 2025).   
• Fine-tuning: We use low-rank adapters (Hu et al., 2022) inserted into the query and key projections of the transformer, updating millions of parameters depending on model sizes. This serves as a parametric form of procedural memory, but is prone to forgetting as new tasks are introduced.   
• Replay Memory: To mitigate catastrophic forgetting in fine-tuning, we follow the idea of experience replay (Mnih et al., 2015) by maintaining a buffer of previously seen precedures and mixing them with the current training data.

Training Details. All methods are implemented in HuggingFace Transformers and trained on a single NVIDIA A6000 GPU with 48GB memory using mixed-precision (bfloat16) training. The backbone models remain frozen; for fine-tuning, only the adapter weights (rank $r = 8$ ) are updated, while for TokMem, only the embeddings of the newly added procedure IDs are trainable. Tokenizer vocabulary is expanded with these procedure IDs, and their embeddings are initialized by averaging the pretrained embeddings (Hewitt, 2021). For Replay Memory, we mix $2 0 \%$ of replayed samples

Table 1: Atomic recall performance on SNI, reported with Rouge-L. TokMem consistently outperforms fine-tuning and RAG across models and scales, maintaining strong results even at 1,000 tasks.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="5">Number of Tasks</td><td rowspan="2">Avg.</td></tr><tr><td>10</td><td>50</td><td>200</td><td>500</td><td>1000</td></tr><tr><td rowspan="6">Qwen 2.5 0.5B</td><td>Base</td><td>33.9</td><td>39.0</td><td>38.8</td><td>39.1</td><td>38.5</td><td>37.9</td></tr><tr><td>RAG</td><td>50.4</td><td>43.2</td><td>38.8</td><td>36.2</td><td>34.7</td><td>40.7</td></tr><tr><td>Fine-Tuong</td><td>52.4</td><td>48.0</td><td>40.6</td><td>41.7</td><td>43.2</td><td>45.2</td></tr><tr><td>Replay Memory</td><td>52.4</td><td>49.5</td><td>47.2</td><td>47.7</td><td>46.7</td><td>48.7</td></tr><tr><td>TokMem</td><td>52.8</td><td>51.3</td><td>49.3</td><td>50.2</td><td>50.0</td><td>50.7</td></tr><tr><td>TokMem+DC</td><td>53.8</td><td>50.5</td><td>50.2</td><td>50.9</td><td>50.0</td><td>51.1</td></tr><tr><td rowspan="6">Llama 3.2 3B</td><td>Base</td><td>16.6</td><td>19.9</td><td>20.0</td><td>18.7</td><td>18.2</td><td>18.7</td></tr><tr><td>RAG</td><td>60.0</td><td>48.7</td><td>45.8</td><td>42.3</td><td>39.9</td><td>47.3</td></tr><tr><td>Fine-Tuong</td><td>67.1</td><td>59.1</td><td>59.5</td><td>58.4</td><td>57.9</td><td>60.4</td></tr><tr><td>Replay Memory</td><td>67.1</td><td>61.1</td><td>60.6</td><td>61.4</td><td>60.0</td><td>62.0</td></tr><tr><td>TokMem</td><td>68.0</td><td>62.3</td><td>61.2</td><td>61.5</td><td>61.5</td><td>62.9</td></tr><tr><td>TokMem+DC</td><td>68.8</td><td>62.5</td><td>58.7</td><td>61.7</td><td>61.1</td><td>62.6</td></tr><tr><td rowspan="6">Llama 3.1 8B</td><td>Base</td><td>27.2</td><td>27.8</td><td>30.4</td><td>29.6</td><td>29.5</td><td>28.9</td></tr><tr><td>RAG</td><td>63.8</td><td>53.9</td><td>49.1</td><td>45.3</td><td>42.6</td><td>50.9</td></tr><tr><td>Fine-Tuong</td><td>75.8</td><td>64.3</td><td>63.2</td><td>58.7</td><td>61.6</td><td>64.7</td></tr><tr><td>Replay Memory</td><td>75.8</td><td>65.2</td><td>64.5</td><td>63.4</td><td>63.6</td><td>66.5</td></tr><tr><td>TokMem</td><td>75.4</td><td>65.5</td><td>65.1</td><td>64.4</td><td>64.8</td><td>67.0</td></tr><tr><td>TokMem+DC</td><td>75.6</td><td>65.8</td><td>63.7</td><td>64.2</td><td>64.4</td><td>66.7</td></tr></table>

with the current batch, using a buffer of 500 examples refreshed every 10 tasks in the atomic setting and 1,000 examples updated each round in the compositional setting.

We optimize with AdamW using a learning rate $5 \times 1 0 ^ { - 5 }$ for fine-tuning and $5 \times 1 0 ^ { - 3 }$ for TokMem; weight decay is $1 0 ^ { - 2 }$ for fine-tuning and zero for TokMem. Training runs for one epoch with batch size 4 and maximum sequence length 1024, using teacher forcing and applying the loss only to memory-token and response positions.

# 3.2 ATOMIC MEMORY RECALL

Dataset Details. We evaluate on the Super-Natural Instructions (SNI) dataset (Wang et al., 2022a), which provides diverse QA-style natural language tasks. Here, each task is treated as an individual procedure: a query directly invokes the learned procedure to produce the desired response. We sample 1,000 English tasks, each task contains 500 training and 50 test examples. To reflect how memories are typically acquired over time, we introduce tasks sequentially during training rather than all at once. We scale the number of tasks from 10 up to 1,000, and record checkpoints after training on $\{ 1 0 , 5 0 , 2 0 0 , 5 0 0 , 1 , 0 0 0 \}$ tasks. This resembles incremental domain adaptation (Asghar et al., 2020), where at each checkpoint, the performance is evaluated across all previously seen tasks. Additional tasks details are provided in Appendix D.1.

We follow Wang et al. (2022a) and use Rouge-L (Lin, 2004) to evaluate generation quality. For the methods with explicit memory routing (RAG and TokMem), we additionally report accuracy, reflecting whether the correct procedure was selected and applied.

Decoupled Memory Embeddings. We also include an ablation variant where memory tokens are decoupled to an address token and a steering token. This decoupling separates the roles of a memory token and also increase the capacity of TokMem. We refer to this variant as TokMem with decoupled embeddings (TokMem+DC), with further details provided in Appendix A.

Results and Findings. TokMem provides the most consistent and scalable performance across models and task scales. As shown in Table 1, non-parametric methods such as Base shows stable but fail to achieve competitive performance. RAG performs reasonably well on when memory is not heavy but quickly degrade as the number of task memory increases, indicating its sensitivity to retriever quality. Parametric methods such as fine-tuning achieve stronger initial accuracy but

Table 2: Task routing accuracy. TokMem achieves near-perfect routing accuracy at 1,000 tasks, far exceeding RAG retriever whose accuracy falls below $80 \%$ .   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="5">Number of Tasks</td></tr><tr><td>10</td><td>50</td><td>200</td><td>500</td><td>1000</td></tr><tr><td>Sentence-Bert</td><td>RAG</td><td>99.6</td><td>92.6</td><td>88.7</td><td>83.2</td><td>79.7</td></tr><tr><td rowspan="2">Qwen 2.5 0.5B</td><td>TokMem</td><td>99.4</td><td>98.6</td><td>97.4</td><td>96.9</td><td>94.7</td></tr><tr><td>TokMem+DC</td><td>99.4</td><td>99.2</td><td>98.4</td><td>97.2</td><td>96.1</td></tr><tr><td rowspan="2">Llama 3.2 3B</td><td>TokMem</td><td>100.0</td><td>99.9</td><td>98.3</td><td>97.1</td><td>96.1</td></tr><tr><td>TokMem+DC</td><td>99.8</td><td>99.3</td><td>97.2</td><td>96.2</td><td>95.4</td></tr><tr><td rowspan="2">Llama 3.1 8B</td><td>TokMem</td><td>99.8</td><td>99.6</td><td>98.9</td><td>97.7</td><td>97.5</td></tr><tr><td>TokMem+DC</td><td>99.7</td><td>99.4</td><td>97.8</td><td>97.2</td><td>97.2</td></tr></table>

![](images/88a3a4b71a09b054660f4aed0a4493b2b85bcad7f53c0e4c422b91ff6e8837e2.jpg)  
Figure 2: Sample efficiency on a 10-task mixture from SNI. TokMem consistently outperforms finetuning in the low-data regime. TokMem can surpass RAG with only 10 training samples, demonstrating strong few-shot learning capability.

suffer from forgetting as tasks accumulate; replay memory alleviates this issue but still falls short of TokMem. By contrast, we see that TokMem maintains high accuracy with minimal performance drop when acquiring new task memories, achieving the best average results across all settings. The decoupled variant (TokMem+DC) provides modest gains for the smaller Qwen 0.5B model but provides no improvement for larger Llama models, and in some cases underperforms TokMem when scaling to many tasks.

Table 2 further highlights TokMem’s robustness in memory routing. Its accuracy remains above $9 4 \%$ even at 1,000 tasks with the smallest 0.5B model, significantly outperforming the Sentence-BERT retriever used in RAG, whose accuracy drops below $8 0 \%$ when have the stress to route for 1,000 tasks. This high-fidelity memory routing enables TokMem to sustain strong performance without relying on external retrieval mechanisms or fine-tuning, demonstrating its advantage in continual and large-scale task acquisition.

Analysis of Training Efficiency. We compare the training sample efficiency of LoRA fine-tuning and TokMem on the first 10 tasks from SNI, a mixture setup that removes the impact of forgetting. We set the adapter rank to $r = 1$ , which helps prevent overfitting and aligns its parameter scale with TokMem. The result in Figure 2 shows that TokMem consistently achieves higher performance than fine-tuning across all sample budgets, with the greatest advantage appearing in the low-data regime. The decoupled variant (TokMem+DC) offers small but consistent improvements, particularly when more samples are available. Overall, these results highlight TokMem’s ability to learn new procedures effectively with limited data, making it a both parameter and data efficient approach to memory acquisition.

Table 3: Compositional tool-use performance on APIGen. TokMem achieves strong tool selection and argument F1 across multiple calls, outperforming ICL and RAG with lower input-augmentation complexity, and surpassing fine-tuning with far fewer trainable parameters.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2">#Params</td><td colspan="4">Tool Selection</td><td colspan="4">Argument</td></tr><tr><td>2 calls</td><td>3 calls</td><td>4 calls</td><td>Avg.</td><td>2 calls</td><td>3 calls</td><td>4 calls</td><td>Avg.</td></tr><tr><td rowspan="5">Llama 3.2 1B</td><td>ICL</td><td>-</td><td>27.6</td><td>11.1</td><td>10.5</td><td>16.4</td><td>0.6</td><td>0.7</td><td>0.0</td><td>0.4</td></tr><tr><td>RAG</td><td>-</td><td>29.5</td><td>10.8</td><td>10.5</td><td>16.9</td><td>7.2</td><td>1.0</td><td>0.0</td><td>2.7</td></tr><tr><td>Fine-Tuong</td><td>0.85M</td><td>10.4</td><td>9.5</td><td>7.0</td><td>9.0</td><td>77.3</td><td>72.6</td><td>55.8</td><td>68.6</td></tr><tr><td>TokMem</td><td>0.10M</td><td>98.4</td><td>98.0</td><td>98.9</td><td>98.4</td><td>84.3</td><td>84.3</td><td>87.8</td><td>85.5</td></tr><tr><td>-adapt</td><td>0.10M</td><td>86.8</td><td>80.9</td><td>90.8</td><td>86.2</td><td>68.9</td><td>61.1</td><td>73.0</td><td>67.7</td></tr><tr><td rowspan="5">Llama 3.2 3B</td><td>ICL</td><td>-</td><td>66.8</td><td>59.2</td><td>59.6</td><td>61.9</td><td>42.2</td><td>42.3</td><td>38.8</td><td>44.1</td></tr><tr><td>RAG</td><td>-</td><td>78.1</td><td>71.2</td><td>69.3</td><td>72.8</td><td>54.8</td><td>53.1</td><td>62.7</td><td>56.9</td></tr><tr><td>Fine-Tuong</td><td>2.29M</td><td>98.7</td><td>98.1</td><td>96.8</td><td>97.9</td><td>87.9</td><td>86.6</td><td>82.9</td><td>85.8</td></tr><tr><td>TokMem</td><td>0.15M</td><td>99.2</td><td>98.2</td><td>100.0</td><td>99.2</td><td>85.9</td><td>86.7</td><td>88.3</td><td>86.3</td></tr><tr><td>-adapt</td><td>0.15M</td><td>82.6</td><td>79.3</td><td>67.2</td><td>76.4</td><td>65.4</td><td>57.2</td><td>50.2</td><td>57.6</td></tr><tr><td rowspan="5">Llama 3.1 8B</td><td>ICL</td><td>-</td><td>79.7</td><td>72.9</td><td>75.4</td><td>76.0</td><td>51.5</td><td>52.6</td><td>57.3</td><td>53.8</td></tr><tr><td>RAG</td><td>-</td><td>79.6</td><td>75.3</td><td>93.0</td><td>82.6</td><td>53.3</td><td>57.1</td><td>69.2</td><td>59.9</td></tr><tr><td>Fine-Tuong</td><td>3.41M</td><td>98.8</td><td>97.2</td><td>98.2</td><td>98.1</td><td>87.7</td><td>86.8</td><td>88.2</td><td>87.6</td></tr><tr><td>TokMem</td><td>0.20M</td><td>99.4</td><td>97.9</td><td>100.0</td><td>99.1</td><td>88.1</td><td>86.5</td><td>93.4</td><td>89.3</td></tr><tr><td>-adapt</td><td>0.20M</td><td>84.9</td><td>82.0</td><td>81.6</td><td>82.8</td><td>65.8</td><td>56.7</td><td>65.9</td><td>62.8</td></tr></table>

# 3.3 COMPOSITIONAL MEMORY RECALL

Dataset Details. We construct a benchmark from the APIGen dataset (Liu et al., 2024b) by sampling the 50 frequently used tools. Here, each tool invocation is treated as an atomic procedure, and solving a query requires composing multiple such procedures. We synthesize 5,000 training queries and 500 test queries, both capped at four calls. Details of the dataset can be found in Appendix D.2

We report performance using two F1 metrics: (i) Tool Prediction F1, which measures whether the correct tools are invoked; and (ii) Argument Generation F1, which evaluates the correctness of function call arguments. For robustness to semantic equivalence, both gold and predicted outputs are normalized into Abstract Syntax Trees before scoring (Patil et al., 2025).

Compositional Adaptation Fine-tuning. We found that TokMem benefits from a brief adaptation phase that exposes the backbone with the compositional structures of memory tokens. Concretely, we fine-tine the backbone on a held-out auxiliary tool set using the same LoRA fine-tuning setup as the baseline. The adapted weights are then merged, after which the backbone remains frozen for memory acquisition and evaluation (see Appendix B.1 for details).

Results and Findings. Table 3 shows that TokMem achieves the strongest overall performance. Without adaptation for compositionality, it outperforms RAG while avoiding the added complexity from an external retrieval mechanism. Non-parametric baselines such as ICL and RAG perform poorly on both tool prediction and argument generation, particularly with the smaller Llama 1B model, likely due to its weak instruction-following ability.

Compared with parametric baselines, TokMem consistently matches or surpasses LoRA fine-tuning while requiring an order of magnitude fewer trainable parameters. Notably, TokMem exhibits stronger interpretability between tool selection and argument generation, with improvements in the former translating directly into the latter. By contrast, LoRA fine-tuning shows weaker alignment. For example, as seen with the 1B model, it often generates plausible arguments even when tool selection is incorrect, indicating that its argument generation is not properly grounded in the chosen tools.

Compositional Generalization. We observe that TokMem provides clear advantages in compositional generalization over fine-tuning. Table 4 reports Argument F1 when the Llama 3B model is evaluated on queries requiring more function calls than those observed during training.

Notably, when trained solely on single-call data, TokMem achieves much stronger performance than fine-tuning when evaluating on 2 to 4 calls test data. This demonstrates that memory tokens

Table 4: TokMem shows strong out-of-domain compositional generalization, significantly outperforming fine-tuning when tested on longer call chains than seen during training. Shaded cells indicate out-of-domain test scenarios.   

<table><tr><td rowspan="2">Training</td><td rowspan="2">Method</td><td colspan="3">Argument</td><td rowspan="2">Avg.</td></tr><tr><td>2 calls</td><td>3 calls</td><td>4 calls</td></tr><tr><td rowspan="2">1-call</td><td>Fine-tuning</td><td>34.9</td><td>21.3</td><td>14.1</td><td>23.4</td></tr><tr><td>TokMem</td><td>60.3</td><td>54.3</td><td>48.9</td><td>54.5 (+31.1)</td></tr><tr><td rowspan="2">2-call</td><td>Fine-tuning</td><td>86.2</td><td>78.8</td><td>64.8</td><td>76.6</td></tr><tr><td>TokMem</td><td>82.0</td><td>81.8</td><td>82.3</td><td>82.0 (+5.4)</td></tr><tr><td rowspan="2">3-call</td><td>Fine-tuning</td><td>86.9</td><td>85.5</td><td>80.3</td><td>84.2</td></tr><tr><td>TokMem</td><td>86.8</td><td>84.0</td><td>84.7</td><td>85.2 (+1.0)</td></tr><tr><td rowspan="2">4-call</td><td>Fine-tuning</td><td>87.9</td><td>86.6</td><td>82.9</td><td>85.8</td></tr><tr><td>TokMem</td><td>85.9</td><td>86.7</td><td>88.3</td><td>86.3 (+0.5)</td></tr></table>

![](images/068181dde58bb93161925145683e34fe324e1aa1f724d6064886d749e50e15b6.jpg)

![](images/8f24cfa74f824928a39c19213a0415109a0dee0492ae516251beadf6f73fcb2b.jpg)

![](images/02b00545a57c576cdcf76af1a55d76481f49bbc2798747d7abf3cbe671b6a9ad.jpg)  
Figure 3: Forgetting analysis in continual adaptation. As new tools are introduced, fine-tuning with replay memory suffers sharp drops on earlier tasks, while TokMem maintains stable performance. Larger models show stronger retention due to greater capacity.

trained for atomic procedures can be effectively composed at test time, enabling strong zero-shot generalization to multi-step behavior.

As the training regime is expanded to include more calls (e.g., 3 or 4), the performance gap narrows, but TokMem remains competitive or slightly ahead across all configurations.

Analysis of Forgetting. We compare TokMem with replay memory in a continual adaptation setting, where tools are introduced sequentially over five training rounds (e.g., tools 1–10 in the first round, 11–20 in the second, and so forth). As shown in Figure 3, we see that replay memory struggles to prevent catastrophic forgetting as new tools are introduced. By contrast, TokMem maintains higher performance across tool groups, with only mild declines that primarily reflect the growing number of tools. Larger models exhibit better retention for both approaches, likely due to their expanded parameter capacity, which reduces the risk of interference with previously learned tools.

We further investigate the effect of the renormalization step introduced in Section 2.3 on newly added memory tokens, whose norms may otherwise dominate older tokens in the softmax. As seen in Figure 4, TokMem without renormalization shows noticeable forgetting especially when the size of the model is small. However, larger models are more robust to forgetting even without renormalization, again due to their greater embedding capacity. Overall, renormalization improves TokMem’s resistance to forgetting by balancing routing between both new and old memory tokens. Additional analysis on the benefits of keeping the backbone frozen for continual memory acquisition is provided in Appendix B.2.

# 3.4 ANALYSIS ON MEMORY PLACEMENT

An important design choice in TokMem is the placement of memory tokens within the input sequence, which directly influences how the backbone model attends to and integrates procedural

![](images/32b346c5a26f1d36326340547829c40ffb2fce87cb55515aeb6650344e83e2f8.jpg)  
Figure 4: Effect of renormalization on TokMem. Without renormalization, new tokens dominate and older ones are forgotten, particularly in smaller models with limited embedding capacity. Renormalization effectively mitigates this by balancing norms across tokens.

Table 5: Comparison of TokMem vs. prefix tuning on memorizing text from the Fanfics dataset. TokMem converges faster and achieves lower perplexity than prefix tuning, particularly with few memory tokens.   

<table><tr><td rowspan="2">Method</td><td colspan="2">1024 tokens</td><td colspan="2">2048 tokens</td><td colspan="2">4096 tokens</td></tr><tr><td>Steps@90%Best ↓</td><td>PPL ↓</td><td>Steps@90%Best ↓</td><td>PPL ↓</td><td>Steps@90%Best ↓</td><td>PPL ↓</td></tr><tr><td>Prefix tuning-1</td><td>1700</td><td>3.81</td><td>2300</td><td>8.77</td><td>2200</td><td>14.32</td></tr><tr><td>TokMem-1</td><td>1200</td><td>3.28</td><td>1400</td><td>7.07</td><td>1700</td><td>12.27</td></tr><tr><td>Prefix tuning-2</td><td>500</td><td>1.13</td><td>1700</td><td>3.51</td><td>1800</td><td>8.38</td></tr><tr><td>TokMem-2</td><td>600</td><td>1.09</td><td>1300</td><td>2.75</td><td>1700</td><td>7.21</td></tr><tr><td>Prefix tuning-5</td><td>300</td><td>1.07</td><td>500</td><td>1.17</td><td>1400</td><td>2.39</td></tr><tr><td>TokMem-5</td><td>200</td><td>1.03</td><td>500</td><td>1.15</td><td>1400</td><td>1.91</td></tr></table>

knowledge. While TokMem introduces a memory routing mechanism for generating tokens, its effectiveness also depends on this placement strategy. In the absence of routing, TokMem reduces to a prompt-tuning method (Li & Liang, 2021; Lester et al., 2021) with learnable embeddings, but it distinctively adopts an infix placement: query $\oplus$ MEM $\oplus$ response. Our experiments indicate that this infix design allows memory tokens to be activated after the context has been encoded, enabling context-aware conditioning and natural composition of multiple procedures.

However, it is unclear whether this memory placement is strictly better than the more common pre-$\mathit { \ f u x }$ formulation: MEM $\oplus$ query $\oplus$ response used in prior prompt-tuning work, where prefix tokens influence generation without having observed the query. To study the impact of placement, we compare prefix and infix placements under matched token budgets in the single-task setting.

Setup. We compare TokMem with infix memory placement against prefix tuning by stress-testing the capacity of memory tokens using the recent Fanfics dataset collected after the pretraining of LLMs (Kuratov et al., 2025). We fix the sequence length to 128 tokens and vary the batch size from 8 to 32, compressing batches of 1024 to 4096 response tokens into 1 to 5 memory tokens. For each sequence, we prepend a randomly generated query that serves only as a marker to distinguish the two placements, while the actual target to be learned remains the response.

We measure learning speed using Steps $@90 \%$ Best, defined as the number of training steps (evaluated every 100 steps) required to reach $90 \%$ of the best perplexity. Results are averaged over five runs. Additional experiments evaluating generalization on a math reasoning dataset are provided in Appendix C.

Results. Table 5 shows that TokMem consistently achieves lower perplexity and often converges faster than prefix tuning. With a single token, TokMem reaches $9 0 \%$ of the best perplexity roughly $3 0 \%$ sooner than prefix tuning, indicating that conditioning memory after the query helps the model learn more efficiently. Interestingly, when more tokens are available (e.g., five tokens), the performance gap narrows. This suggests that prior work (Li & Liang, 2021), which typically uses dozens

or even hundreds of tokens, may have underestimated the importance of memory placement in lowtoken regimes, where each token must compress more procedural information.

# 4 RELATED WORK

Equipping LLMs with memory has been explored through multiple directions. Most existing approaches emphasize declarative memory, where the objective is to store and retrieve explicit information such as facts or conversation history (Packer et al., 2023; Chhikara et al., 2025; Zhong et al., 2024). In contrast, parameter-based approaches internalize task-specific behaviors within model parameters, resembling procedural memory. TokMem builds on this latter view while emphasizing modularity and compositionality.

Text-based External Memory. A common approach is to externalize memory as textual content retrieved at inference time. Retrieval-augmented generation (RAG)(Lewis et al., 2020) and its variants(Guu et al., 2020; Karpukhin et al., 2020; Borgeaud et al., 2022; Khandelwal et al., 2020) attach relevant textual chunks during inference, while RET-LLM (Modarressi et al., 2023) encodes knowledge as symbolic triplets. Building on these ideas, more recent systems such as MemGPT (Packer et al., 2023), Mem0 (Chhikara et al., 2025), and A-Mem (Xu et al., 2025) extend these ideas to conversational settings through hierarchical or summarization-based memory states. While effective for factual recall, these approaches are not optimized for procedural control and often incur significant inference-time overhead due to the re-read of textual memory.

Parameter-based Memory. Another line of work encodes memory directly into model parameters. Fine-tuning and multitask instruction tuning (Wei et al., 2022a; Sanh et al., 2021), as well as parameter-efficient variants such as LoRA (Hu et al., 2022) allow models to acquire new procedures, but task knowledge is entangled. MemoryLLM (Zhong et al., 2024) introduces latent memory pools but remian entangled. Prompt-based methods such as prompt tuning (Lester et al., 2021; Wu et al., 2024; 2025) store knowledge implicitly as global embeddings without selective routing, and L2P (Wang et al., 2022b) introduces modular prompt pools but still relies on an external controller to determine which prompts are retrieved. ToolGen (Wang et al., 2025) compresses tools into virtual tokens but focuses on post-training the backbone through multi-stage fine-tuning. By contrast, TokMem keeps the backbone frozen, and introduces discrete, composable memory units that can be added or composed without retraining, supporting continual adaptation.

Compositional Memory. A complementary direction explores how models compose skills from simpler building blocks. Chain-of-thought prompting (Wei et al., 2022b) and tool-augmented reasoning frameworks such as Toolformer (Schick et al., 2023) enable multi-step reasoning, but rely on textual instructions that must be re-interpreted at each step. Modular parameter methods (Rosenbaum et al., 2018; Pfeiffer et al., 2021) create specialized adapters that can be recombined, but composition requires parameter merging or heuristic routing. TokMem differs by representing procedures as discrete tokens that can be chained directly in context, enabling lightweight parameterisolated composition.

# 5 CONCLUSION AND FUTURE DIRECTIONS

We introduced Tokenized Memory (TokMem), a parameter-efficient framework that encodes procedural memory as compact tokens. TokMem enables selective recall and compositional use of procedures without modifying backbone parameters, achieving strong performance across multitask and tool-augmented reasoning benchmarks.

Our experiments are conducted on the SNI and APIGen datasets, which allow controlled analysis of atomic and compositional recall. While these settings demonstrate the effectiveness of tokenized procedural memory without backbone training, they do not fully capture the diversity of real-world procedures. In particular, richer forms of composition, such as interleaving function calls with NLP tasks from SNI, as illustrated in Figure 1b, and multi-turn interactions remain unexplored, but could be supported with curated datasets. Overall, advancing TokMem toward practical deployment will require more realistic benchmarks or user-driven data collection pipelines that better reflect opendomain procedural knowledge.

Additional promising directions include incorporating reinforcement learning to improve generalization for complex compositional structures, and enabling personalization by allowing users to attach their own memory banks while keeping the backbone frozen. Together, these extensions pave the way for scalable, compact, and user-adaptive memory systems in large language models.

# REFERENCES

Rishabh Agarwal, Avi Singh, Lei Zhang, Bernd Bohnet, Luis Rosias, Stephanie Chan, Biao Zhang, Ankesh Anand, Zaheer Abbas, Azade Nova, John D. Co-Reyes, Eric Chu, Feryal Behbahani, Aleksandra Faust, and Hugo Larochelle. Many-shot in-context learning. In Advances in Neural Information Processing Systems, pp. 76930–76966, 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/ file/8cb564df771e9eacbfe9d72bd46a24a9-Paper-Conference.pdf.   
John R. Anderson and Christian Lebiere. The Atomic Components of Thought. Lawrence Erlbaum Associates, 1998.   
Nabiha Asghar, Lili Mou, Kira A. Selby, Kevin D. Pantasdo, Pascal Poupart, and Xin Jiang. Progressive memory banks for incremental domain adaptation. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ BkepbpNFwr.   
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. Improving language models by retrieving from trillions of tokens. arXiv preprint arXiv:2112.04426, 2022. URL https://arxiv.org/abs/2112.04426.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020. URL https://papers.nips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html.   
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory, 2025. URL https://arxiv. org/abs/2504.19413.   
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv. org/abs/2110.14168.   
Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, and et al. The Llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.   
Peter D Grunwald.¨ The minimum description length principle. MIT Press, 2007. URL https://direct.mit.edu/books/monograph/3813/ The-Minimum-Description-Length-Principle.   
Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. Retrieval augmented language model pre-training. In International Conference on Machine Learning, pp. 3929–3938, 2020. URL http://proceedings.mlr.press/v119/guu20a.html.   
John Hewitt. Initializing new word embeddings for pretrained language models, 2021. URL https://nlp.stanford.edu/˜johnhew//vocab-expansion.html.   
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi ˘ Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 6769–6781, 2020. URL https://aclanthology.org/2020.emnlp-main.550/.   
Urvashi Khandelwal, Omer Levy, Dan Jurafsky, Luke Zettlemoyer, and Mike Lewis. Generalization through memorization: Nearest neighbor language models. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ HklBjCEKvH.   
Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, and Ashish Sabharwal. Decomposed prompting: A modular approach for solving complex tasks. In The Eleventh International Conference on Learning Representations, 2023. URL https: //openreview.net/forum?id=_nGgzQjzaRy.   
Yuri Kuratov, Mikhail Arkhipov, Aydar Bulatov, and Mikhail Burtsev. Cramming 1568 tokens into a single vector and back again: Exploring the limits of embedding space capacity. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 19323–19339. Association for Computational Linguistics, 2025. URL https: //aclanthology.org/2025.acl-long.948/.   
Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt tuning. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 3045–3059, 2021. URL https://aclanthology.org/2021. emnlp-main.243/.   
Patrick Lewis, Ethan Perez, Aleksandar Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt ¨ aschel, et al. Retrieval-augmented genera- ¨ tion for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33: 9459–9474, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 6b493230205f780e1bc26945df7481e5-Abstract.html.   
Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, pp. 4582–4597, 2021. URL https://aclanthology.org/2021.acl-long.353/.   
Chin-Yew Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pp. 74–81. Association for Computational Linguistics, 2004. URL https:// aclanthology.org/W04-1013/.   
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, pp. 157–173, 2024a. URL https: //aclanthology.org/2024.tacl-1.9/.   
Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig. Pretrain, prompt, and predict: A systematic survey of prompting methods in natural language processing. ACM Comput. Surv., 55(9), 2023. URL https://doi.org/10.1145/3560815.   
Yujia Liu, Jiacheng Zhang, Ziming Wang, Xiaohan Li, Jing Li, and Hao Wang. APIGen: Automated API code generation for function-calling capabilities in large language models, 2024b. URL https://arxiv.org/abs/2406.18518.   
Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Humanlevel control through deep reinforcement learning. nature, pp. 529–533, 2015. URL https: //www.nature.com/articles/nature14236.   
Ali Modarressi, Ayyoob Imani, Mohsen Fayyaz, and Hinrich Schutze.¨ RET-LLM: Towards a general read-write memory for large language models. In Advances in Neural Information Processing Systems, volume 36, pp. 15558–15571, 2023. URL https://proceedings.neurips.cc/paper_files/paper/2023/hash/ 6a4cd50db0cad92c4c8d9e6ee01ac8c6-Abstract-Conference.html.

Charles Packer, Vivian Fang, Shishir Gururaj Patil, Kevin Lin, Sarah Wooders, and Joseph E Gonzalez. MemGPT: Towards LLMs as operating systems. arXiv preprint arXiv:2310.08560, 2023. URL https://arxiv.org/abs/2310.08560.   
Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ 2GmDdhBdDk.   
Jonas Pfeiffer, Aishwarya Kamath, Andreas Ruckl ¨ e, Kyunghyun Cho, and Iryna Gurevych. Adapter- ´ Fusion: Non-destructive task composition for transfer learning. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pp. 487–503. Association for Computational Linguistics, 2021. URL https:// aclanthology.org/2021.eacl-main.39/.   
Qwen and et al. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412. 15115.   
Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bertnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2019. URL https://arxiv.org/ abs/1908.10084.   
Clemens Rosenbaum, Tim Klinger, and Matthew Riemer. Routing networks: Adaptive selection of non-linear functions for multi-task learning. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ ry8dvM-R-.   
Pranab Sahoo, Ayush Kumar Singh, Sriparna Saha, Vinija Jain, Samrat Mondal, and Aman Chadha. A systematic survey of prompt engineering in large language models: Techniques and applications, 2025. URL https://arxiv.org/abs/2402.07927.   
Victor Sanh, Albert Webson, Colin Raffel, Stephen H Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, et al. Multitask prompted training enables zero-shot task generalization. arXiv preprint arXiv:2110.08207, 2021. URL https://arxiv.org/abs/2110.08207.   
Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Thirty-seventh Conference on Neural Information Processing Systems, 2023. URL https://openreview.net/forum?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ Yacmpz84TH.   
Larry R Squire. Memory and brain systems: 1969–2009. Journal of Neuroscience, 29:12711–12716, 2009. URL https://www.jneurosci.org/content/29/41/12711.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30:5998–6008, 2017. URL https://papers.nips.cc/paper_ files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract. html.   
Renxi Wang, Xudong Han, Lei Ji, Shu Wang, Timothy Baldwin, and Haonan Li. ToolGen: Unified tool retrieval and calling via generation. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ XLMAMmowdY.   
Yizhong Wang, Swaroop Mishra, Pegah Alipoormolabashi, Yeganeh Kordi, Amirreza Mirzaei, Atharva Naik, Arjun Ashok, Arut Selvan Dhanasekaran, Anjana Arunkumar, David Stap, et al. Super-NaturalInstructions: Generalization via declarative instructions on $1 6 0 0 +$ NLP tasks. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 5085–5109, 2022a. URL https://aclanthology.org/2022.emnlp-main.340/.   
Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang, Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot, Jennifer Dy, and Tomas Pfister. Learning to prompt for continual learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp.

139–149, 2022b. URL https://openaccess.thecvf.com/content/CVPR2022/ papers/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_ paper.pdf.   
Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. Finetuned language models are zero-shot learners. International Conference on Learning Representations, 2022a. URL https://openreview.net/ forum?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ gEZrGCozdqR.   
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian ichter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022b. URL https://openreview. net/forum?id=_VjQlMeSB_J.   
Zijun Wu, Yongkang Wu, and Lili Mou. Zero-shot continuous prompt transfer: Generalizing task semantics across language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=26XphugOcS.   
Zijun Wu, Yongchang Hao, and Lili Mou. ULPT: Prompt tuning with ultra-low-dimensional optimization, 2025. URL https://arxiv.org/abs/2502.04501.   
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025. URL https://arxiv.org/ pdf/2502.12110.   
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum? id=WE_vluYUL-X.   
Yu Zhong, Longyue Wang, Jiajun Liu, Guangdong Chen, Minjun Wu, Qifan Zhou, Zerui Wang, Xianzhi Wang, et al. MemoryLLM: Towards self-updatable large language models. arXiv preprint arXiv:2402.04624, 2024. URL https://arxiv.org/abs/2402.04624.   
Denny Zhou, Nathanael Scharli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schu- ¨ urmans, Claire Cui, Olivier Bousquet, Quoc V Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id= WZH7099tgfM.

![](images/1d50e5cca373bf6dab631f56e02281e351696e98c96e8010385290ed90e71320.jpg)  
Figure 5: Overview of Decoupled TokMem embeddings, which learns separate memory matrices for address of memories and generation steering.

# A DECOUPLED EMBEDDING FOR TOKMEM

In the standard TokMem formulation, each memory token embedding ${ m } _ { i } \in \mathbb { R } ^ { d }$ is shared across two roles: (1) addressing for memory routing and (2) steering for generation. We consider a decoupled (DC) variant that separates these functions into two embedding matrices:

$$
M ^ {\text {a d d r}} = \left[ \boldsymbol {u} _ {1}, \dots , \boldsymbol {u} _ {l} \right] \in \mathbb {R} ^ {l \times d}, \quad M ^ {\text {s t e e r}} = \left[ \boldsymbol {s} _ {1}, \dots , \boldsymbol {s} _ {l} \right] \in \mathbb {R} ^ {l \times d}. \tag {8}
$$

Here, $M ^ { \mathrm { a d d r } }$ provides address embeddings at the output layer. When a memory token is predicted, the model produces a distribution over indices $i$ according to $M ^ { \mathrm { a d d r } }$ . The chosen index $i$ is then used to retrieve the corresponding steering embedding $\mathbf { \boldsymbol { s } } _ { i }$ from $M ^ { \mathrm { s t e e r } }$ , which is injected into the input sequence and influences subsequent generation.

Training follows the standard next-token prediction objective, analogous to Equation 5:

$$
\mathcal {L} \left(\boldsymbol {a}; M ^ {\text {a d d r}}, M ^ {\text {s t e e r}}\right) = - \sum_ {i > k} \log \Pr \left(a _ {i} \mid \boldsymbol {a} _ {<   i}; M ^ {\text {a d d r}}, M ^ {\text {s t e e r}}\right), \tag {9}
$$

where $k$ denotes the query length. During optimization, only $M ^ { \mathrm { a d d r } }$ and $M ^ { \mathrm { s t e e r } }$ are updated; the backbone remains frozen. In addition, the renormalization treatment introduced in Section 2.2 is only applied to the address embeddings in $M ^ { \mathrm { a d d r } }$ .

This decoupled formulation provides a clean separation of functionality: routing is handled via $M ^ { \mathrm { a d d r } }$ , while steering is controlled by $M ^ { \mathrm { s t e e r } }$ . While conceptually clean, our experiments do not show consistent improvements over the coupled formulation, particularly on larger models, where embedding capacity is sufficient to jointly support both roles.

# B DETAILS FOR COMPOSITIONAL MEMORY RECALL

# B.1 DETAILS OF ADAPTATION PHASE

In compositional scenarios, the model should not only recall individual procedures but also compose them to solve multi-step queries. To prepare TokMem for such use, we construct a held-out auxiliary training set of 50 tools (5,000 samples) following Section 3.3. The backbone is then fine-tuned for one epoch on this set using LoRA, jointly with the temporary memory embeddings, before the adapted weights are merged and frozen.

The intuition for this adaptation phase to let the LLM learn to align its routing and generation behavior with compositional memory recall. After adaptation, the temporary embeddings are discarded, while the adapted backbone is retained for inference with new tasks. This procedure provides a general inductive bias for modular composition, enabling it to generalize to new tools and procedures without further retraining.

Algorithm 1 summarizes this lightweight procedure. Temporary memory embeddings are inserted into the input sequence, the loss is optimized jointly over memory and response tokens, and once the backbone has adapted, the temporary memory bank is discarded.

Algorithm 1 Adaptation Phase for Compositional Memory Recall   
Require: Pretrained backbone $f_{\theta_0}$ , adaptation traces $\mathcal{D}_{\mathrm{adapt}}$ from held-out procedures  
1: Initialize backbone $\theta \gets \theta_0$ and temporary memory embeddings $\mathcal{M}$ 2: Set learning rates $\eta_{\theta}$ and $\eta_{\mathcal{M}}$ 3: for each minibatch in $\mathcal{D}_{\mathrm{adapt}}$ do  
4: Insert $\mathcal{M}$ into sequence; forward pass with $f_{\theta}$ 5: Compute loss $\mathcal{L}$ on memory and response tokens  
6: $\theta \gets \theta - \eta_{\theta} \nabla_{\theta} \mathcal{L}$ 7: $\mathcal{M} \gets \mathcal{M} - \eta_{\mathcal{M}} \nabla_{\mathcal{M}} \mathcal{L}$ 8: end for  
9: Discard temporary memory $\mathcal{M}$ and freeze backbone $\theta$ 10: return adapted backbone $f_{\theta}$

![](images/51186341c34034ab8c64d027a0fac1dc0abf76c4bda80c93286148bb71e63ec1.jpg)  
Figure 6: Comparison between Freezing and unfreezing the backbone. Allowing the backbone to update when adding new tool memories causes severe forgetting. Freezing preserves prior tools while enabling new ones.

# B.2 ANALYSIS OF UNFREEZING LLM BACKBONE FOR TOKMEM

We further examine the importance of freezing the backbone when adding new tool memories, reflecting real-world usage where procedural knowledge grows incrementally over time. This setting contrasts with recent approaches (Wang et al., 2025) that compress tool usage into virtual tokens by post-training the backbone. While such methods improve retrieval efficiency at scale, they rely on modifying backbone parameters, which hinders continual adaptability and risks overwriting prior knowledge.

As shown in Figure 6, unfreezing the backbone during TokMem adaptation leads to severe forgetting of previously learned tools, consistent with catastrophic interference in continual learning. By contrast, freezing the backbone preserves prior capabilities while allowing new tool memories to be incorporated without loss, highlighting TokMem’s advantage for incremental adaptation. Notably, unfreezing offers no meaningful performance gains after the initial training round, suggesting that TokMem strikes an effective balance between performance and continual adaptation.

# C ADDITIONAL ANALYSIS ON MEMORY PLACEMENT

We have stress-tested the effect of memory token placement (prefix vs. infix) with randomly generated queries and with varying length of memory tokens in Section 3.4. We now turn to the GSM8K math reasoning dataset (Cobbe et al., 2021), to evaluate generalization and training efficiency.

Our experiments run on Llama 3.2 1B and 3B as backbone models and compare prefix tuning against TokMem under two training setups: using only $2 0 \%$ of the training set that represents a low-data regime, or the full dataset. We report two evaluation metrics.

• Compliance measures whether the model follows the required answer format, i.e., producing the final answer after the delimiter “####”. This metric isolates the recall of procedural memory from the reasoning abilities already present in the backbone models.

Table 6: Comparison of prefix tuning vs. TokMem condition embedding on GSM8K with two different size of Llama models. TokMem achieves higher compliance with required output formats and stronger exact-match accuracy than prefix tuning, especially in low-data regimes.   

<table><tr><td rowspan="2">Data%</td><td rowspan="2">Method</td><td colspan="2">Llama 3.2 1B</td><td colspan="2">Llama 3.2 3B</td></tr><tr><td>Compliance↑</td><td>EM ↑</td><td>Compliance↑</td><td>EM ↑</td></tr><tr><td rowspan="2">20%</td><td>Prefix tuning</td><td>0.0</td><td>0.0</td><td>45.9</td><td>33.1</td></tr><tr><td>TokMem</td><td>98.0</td><td>37.7</td><td>94.6</td><td>65.6</td></tr><tr><td rowspan="2">100%</td><td>Prefix tuning</td><td>82.8</td><td>30.0</td><td>97.2</td><td>64.1</td></tr><tr><td>TokMem</td><td>97.4</td><td>39.1</td><td>98.2</td><td>66.9</td></tr></table>

• Exact Match (EM) measures the correctness of the final answer after standard normalization (e.g., removing commas or extraneous symbols).

As shown in Table 6, TokMem significantly outperforms prefix tuning, particularly in the low-data setting. With only $2 0 \%$ of the data, prefix-tuning fails to provide meaningful results, yielding zero compliance and EM on the 1B model and underperforming on the 3B model. By contrast, TokMem achieves near-perfect compliance and substantially higher EM scores across both backbones. When trained on the full dataset, prefix tuning improves considerably, yet TokMem continues to deliver stronger compliance and higher EM, underscoring its superior data efficiency and more reliable procedural control.

# D DETAILS OF DATASETS

# D.1 DETAILS OF SUPER-NATURAL INSTRUCTION

We sample 1,000 English tasks from the SNI dataset, where each task is labeled with a task ID and a short descriptive name. The full list of sampled tasks is provided in Table 7. Tasks are introduced to the model sequentially in ascending order of their IDs (e.g., the model first sees task 1, then task 2, and so on).

After training on the first $k$ tasks, we save a checkpoint and evaluate performance on the test sets of all $k$ tasks encountered so far. This simulates a continual learning setup where the model is expected to acquire new procedures while retaining previously learned ones. Once the model has been trained on all 1,000 tasks, it should be able to perform all of them without forgetting earlier tasks.

# D.2 DETAILS OF FUNCTION CALLING DATASET

For evaluating compositional memory recall, we sample 50 tools from the APIGen dataset (Liu et al., 2024b). The list of tools and their corresponding descriptions is provided in Table 7.

For each tool, we collect 50 query–call pairs, some of which may involve multiple calls to the same tool. This yields a total of $5 0 \times 5 0 = 2 { , } 5 0 0$ samples representing the non-compositional use of tools. To avoid data leakage, we split these samples into training and test sets with a 9:1 ratio.

On top of this, we synthesize complex queries by combining calls across different tools. These multi-step queries require the model to invoke multiple tools in sequence. We cap the number of synthesized samples at 5,000 for training and 500 for testing.

Table 7: Details of the sampled tools from the APIGen dataset, including their names and descriptions.   

<table><tr><td>ID</td><td>Tool</td><td>Description</td></tr><tr><td>1</td><td>auto_COMPLETE</td><td>Fetch auto-complete suggestions for a given query using the Wayfair API.</td></tr><tr><td>2</td><td>binary_addition</td><td>Adds two binary numbers and returns the result as a binary string.</td></tr><tr><td>3</td><td>binary_search</td><td>Performs binary search on a sorted list to find the index of a target value.</td></tr><tr><td>4</td><td>cagr</td><td>Calculates the Compound Annual Growth Rate (CAGR) of an investment.</td></tr><tr><td>5</td><td>calculate_factorial</td><td>Calculates the factorial of a non-negative integer.</td></tr><tr><td>6</td><td>calculate_grade</td><td>Calculates the weighted average grade based on scores and their corresponding weights.</td></tr><tr><td>7</td><td>calculate_median</td><td>Calculates the median of a list of numbers.</td></tr><tr><td>8</td><td>can_attend_all Meetings</td><td>Determines if a person can attend all meetings given a list of meeting time intervals.</td></tr><tr><td>9</td><td>cosine_similarity</td><td>Calculates the cosine similarity between two vectors.</td></tr><tr><td>10</td><td>count_bits</td><td>Counts the number of set bits (1&#x27;s) in the binary representation of a number.</td></tr><tr><td>11</td><td>create.histogram</td><td>Create a histogram based on provided data.</td></tr><tr><td>12</td><td>directions_between_2 Locations</td><td>Fitches the route information between two geographical locations including distance, duration, and steps.</td></tr><tr><td>13</td><td>fibonacci</td><td>Calculates the nth Fibonacci number.</td></tr><tr><td>14</td><td>final Velocity</td><td>Calculates the final velocity of an object given its initial velocity, acceleration, and time.</td></tr><tr><td>15</td><td>find_equilibrium_index</td><td>Finds the equilibrium index of a list, where the sum of elements on the left is equal to the sum of elements on the right.</td></tr><tr><td>16</td><td>find_first_non_repeating_char</td><td>Finds the first non-repeating character in a string.</td></tr><tr><td>17</td><td>find_longest_word</td><td>Finds the longest word in a list of words.</td></tr><tr><td>18</td><td>find_max_subarray_sum</td><td>Finds the maximum sum of a contiguous subarray within a list of integers.</td></tr><tr><td>19</td><td>find_minimumRotated Sorted_array</td><td>Finds the minimum element in a rotated sorted array.</td></tr><tr><td>20</td><td>flatten_list</td><td>Flattens a nested list into a single-level list.</td></tr><tr><td>21</td><td>format_date</td><td>Converts a date string from one format to another.</td></tr><tr><td>22</td><td>generate_password</td><td>Generates a random password of specified length and character types.</td></tr><tr><td>23</td><td>generate_random_string</td><td>Generates a random string of specified length and character types.</td></tr><tr><td>24</td><td>get_city_from_zipcode</td><td>Retrieves the city name for a given ZIP code using the Ziptastic API.</td></tr><tr><td>25</td><td>get_pokémon_move_info</td><td>Retrieves information about a Pokémon&#x27;s move using the Pokémon API.</td></tr><tr><td>26</td><td>get_product</td><td>Fetched product details from an API using the given product ID.</td></tr><tr><td>27</td><td>get_products_in_category</td><td>Fetched products in a specified category from the demo project&#x27;s catalog.</td></tr><tr><td>28</td><td>greatest_common_divisor</td><td>Computes the greatest common divisor (GCD) of two non-negative integers.</td></tr><tr><td>29</td><td>integrate</td><td>Calculate the area under a curve for a specified function between two x values.</td></tr><tr><td>30</td><td>investment_profit</td><td>Calculates the profit from an investment based on the initial amount, annual return rate, and time.</td></tr><tr><td>31</td><td>is_anagram Phrase</td><td>Checks if two phrases are anagrams of each other, ignoring whitespace and punctuation.</td></tr><tr><td>32</td><td>is_leap_year</td><td>Checks if a year is a leap year.</td></tr><tr><td>33</td><td>isPALINDrome</td><td>Checks if a string is a palindrome.</td></tr><tr><td>34</td><td>is_power</td><td>Checks if a number is a power of a given base.</td></tr><tr><td>35</td><td>isRotation</td><td>Checks if one string is a rotation of another string.</td></tr><tr><td>36</td><td>is_valid_ip_address</td><td>Checks if a string is a valid IP address (IPv4).</td></tr><tr><td>37</td><td>is_validPALINDrome</td><td>Checks if a string is a valid palindrome, considering only alphanumeric characters and ignoring case.</td></tr><tr><td>38</td><td>is_valid_sudoku</td><td>Checks if a 9x9 Sudoku board is valid.</td></tr><tr><td>39</td><td>monthly_mortgage-payment</td><td>Calculates the monthly mortgage payment based on the loan amount, annual interest rate, and loan term.</td></tr><tr><td>40</td><td>note_duration</td><td>Calculates the duration between two musical notes based on their frequencies and the tempo.</td></tr><tr><td>41</td><td>place_safeway_order</td><td>Order specified items from a Safeway location.</td></tr><tr><td>42</td><td>polygon_area_shoelace</td><td>Calculates the area of a polygon using the shoelace formula.</td></tr><tr><td>43</td><td>potential_energy</td><td>Calculates the electrostatic potential energy given the charge and voltage.</td></tr><tr><td>44</td><td>project_populaton</td><td>Projects the population size after a specified number of years.</td></tr><tr><td>45</td><td>reverse_string</td><td>Reverses the characters in a string.</td></tr><tr><td>46</td><td>solve_quadratic</td><td>Computes the roots of a quadratic equation given its coefficients.</td></tr><tr><td>47</td><td>trapezoidal_integration</td><td>Calculates the definite integral of a function using the trapezoidal rule.</td></tr><tr><td>48</td><td>whois</td><td>Fetch the WhoIS lookup data for a given domain using the specified Toolbench RapidAPI key.</td></tr><tr><td>49</td><td>whole_foods_order</td><td>Places an order at Whole Foods.</td></tr><tr><td>50</td><td>wire_resistance</td><td>Calculates the resistance of a wire based on its length, cross-sectional area, and material resistivity.</td></tr></table>

![](images/d2d81b9427223ebcdd26b8d15e542ed434ab0cedd8e78dfcb60cb583d6183510.jpg)  
Figure 7: Overview of the 1,000 English tasks from the SNI dataset used in the atomic recall setting.
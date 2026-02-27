# Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions

Yuanzhe Hu∗, Yu Wang∗, Julian McAuley

UC San Diego

![](images/114f42af8bfd301fb2c02d5b4464dcc1c4b024089445ffddf0d48827106aa17f.jpg)

Datasets

![](images/288aea52e42ef6acdd7fb56f6ea0f714289650a3a103e666a94ad293e78754af.jpg)

Source Code

# Abstract

Recent benchmarks for Large Language Model (LLM) agents primarily focus on evaluating reasoning, planning, and execution capabilities, while another critical component—memory, encompassing how agents memorize, update, and retrieve long-term information—is under-evaluated due to the lack of benchmarks. We term agents with memory mechanisms as memory agents. In this paper, based on classic theories from memory science and cognitive science, we identify four core competencies essential for memory agents: accurate retrieval, test-time learning, long-range understanding, and selective forgetting. Existing benchmarks either rely on limited context lengths or are tailored for static, long-context settings like book-based QA, which do not reflect the interactive, multi-turn nature of memory agents that incrementally accumulate information. Moreover, no existing benchmarks cover all four competencies. We introduce MemoryAgentBench, a new benchmark specifically designed for memory agents. Our benchmark transforms existing long-context datasets and incorporates newly constructed datasets into a multi-turn format, effectively simulating the incremental information processing characteristic of memory agents. By carefully selecting and curating datasets, our benchmark provides comprehensive coverage of the four core memory competencies outlined above, thereby offering a systematic and challenging testbed for assessing memory quality. We evaluate a diverse set of memory agents, ranging from simple context-based and retrieval-augmented generation (RAG) systems to advanced agents with external memory modules and tool integration. Empirical results reveal that current methods fall short of mastering all four competencies, underscoring the need for further research into comprehensive memory mechanisms for LLM agents.

# 1 Introduction

Large Language Model (LLM) agents have rapidly transitioned from proof-of-concept chatbots to end-to-end systems that can write software (Wang et al., 2024c), control browsers (Müller & Žunič, 2024), and reason over multi-modal inputs. Frameworks such as Manus, OWL (Hu et al., 2025), OpenHands (Wang et al., 2024c), and Codex routinely solve complex, tool-rich tasks and achieve state-of-the-art results on agentic benchmarks like GAIA (Mialon et al., 2023) and SWE-Bench (Jimenez et al., 2023). Yet these evaluations focus almost exclusively on reasoning (planning, tool using, code synthesis) and leave the equally important question of memorization (abstraction, storing, updating, retrieving) largely under-explored. Recent memory-centric architectures—ranging from parametric memory systems like MemoryLLM (Wang et al., 2024d), SELF-PARAM (Wang et al.), and M+(Wang et al., 2025) to commercial token-level memory solutions such as MemGPT(Packer et al., 2023; Lin et al., 2025), Mem0(Chhikara et al., 2025), Cognee(Markovic et al., 2025), and Zep(Rasmussen et al., 2025)—employ diverse strategies for storing and retrieving past information. Despite growing interest, their real-world

![](images/a6a4b3aacacbc91b8ecc432a60a299d6a7b7dda47112fa09e0497cba1c4e364a.jpg)

I went to the zoo and saw elephants.

... (Long long conversation)

![](images/1349a964a253737eba4e9426de6eb5d9a029f5a8659d5393a20a5a2ba9fd815f.jpg)

What did I see at the zoo?

Accurate Retrieval

![](images/2af7d1435dc4aacbb09b2099c05b93169f2cf678962ee08c2b9288d7298d57d0.jpg)

I'm reading this book: Harry Potter grippedhis wand,ready for whatever came next,..

... (L.ong long conversation)

![](images/ca4e290bd89682c4ea75a7c8dbe3e5679072c32fbf8544e188a2d1fa3d6021fc.jpg)

Help me summarize the story.

Long Range Understanding

![](images/dc273d7da836ff43fa09f2cb5c176a8b53ab31a3ae29e140ff8260b7e410316e.jpg)

$A _ { 1 }$ is class 1; $B _ { 1 }$ is class 2;

$A _ { 2 }$ is class 1; $B _ { 2 }$ is class 2;

... (lLong long conversation)

![](images/f33b963458f8a5df77365173d940ce97265e28f4979c71abc0d9073712bcd0a7.jpg)

Which class is $A _ { 5 } ?$ Which class is $B _ { 5 } ?$

Test-Time Learning

![](images/ad7f6b95b7490526aa46e9a36120bf7f68462f3d08ffcfc9084329f359b10fe6.jpg)

I love the pear.

... (Long long conversation)

Did I say I love pears? That might be a typo. I don't like fruits. I love peas.

![](images/7d0d4ca48bca62c8337cff6ea482f27f1571e176da0c4c9bde207446ce574d92.jpg)  
Figure 1: Four complementary competencies that memory agents should have.

Do I love pears?

Selective Forgetting

effectiveness remains largely anecdotal, and there is currently no unified benchmark for systematically evaluating the quality of memory in agents. In this paper, we refer to agents equipped with memory mechanisms as Memory Agents, where memory can take various forms, including parameters, vectors, textual histories, or external databases. In this paper, we primarily focus on memory agents that utilize textual histories and external databases, as these approaches are most commonly deployed in real-world applications. In contrast, memory encoded in model parameters (Wang et al., 2024d; 2025; Yin et al., 2024) remains largely within academic research and is typically less capable than proprietary memory systems equipped on closed-sourced API models.

Based on some classic theories in memory and cognitive science (James, 1890; McClelland et al., 1995; Anderson & Neely, 1996; Wimber et al., 2015), we identify four complementary competencies (Examples shown in Figure 1) to evaluate memory agents: (1) Accurate Retrieval (AR): The ability to extract the correct snippet in response to a query. This can involve one-hop or multi-hop retrieval, as long as the relevant information can be accessed with a single query. (2) Test-Time Learning (TTL): The capacity to incorporate new behaviors or acquire new skills during deployment, without additional training. (3) Long-Range Understanding (LRU): The ability to integrate information distributed across extended contexts (≥ 100k tokens) and answer questions requiring a global understanding of the entire sequence. (4) Selective Forgetting (SF): The skill to revise, overwrite, or remove previously stored information when faced with contradictory evidence, aligning with goals in model editing and knowledge unlearning tasks (Meng et al., 2023; Wang et al., 2024e). For these fuor competencies, we provide more detailed definitions in Appendix A.

Previous datasets developed to evaluate memory in language models have notable limitations. Early benchmarks such as LOCOMO (Maharana et al., 2024) (∼ 9k tokens), LooGLE(Li et al., 2023) ( $\sim$ 24k tokens), and LongBench(Bai et al., 2023) ( $\sim$ 20k tokens) feature relatively short contexts that no longer challenge current models. More recent datasets like NovelQA(Wang et al., 2024a) ( $\sim$ 200k tokens), NOCHA(Karpinska et al., 2024) (∼127k tokens), Loong(Wang et al., 2024b) ( $\sim$ 100k tokens), and $\infty$ -Bench(Zhang et al., 2024) (∼150k tokens) extend the context length to evaluate global reasoning and retrieval capabilities. However, these datasets were primarily designed for evaluating long-context language models rather than memory agents. The reason that long-context benchmarks cannot be directly used to evaluate memory agents is as follows. There is a fundamental distinction between memory and long context: memory serves as a compressed and distilled representation of past information. Rather than storing all historical content verbatim, memory selectively extracts salient details, removes irrelevant information, and often incorporates new inferences derived from prior experiences. Consequently, memory agents are designed to process context incrementally—absorbing input piece by piece, ab-

stracting and consolidating information over time, generating new inferences, and learning novel rules from accumulated history. For this reason, datasets that provide the entire context in a single block are not directly applicable to evaluating memory agents. A more recent effort, LongMemEval (Wu et al., 2025), seeks to address this limitation by using synthetic long-form conversations, which can be injected into memory gradually, session by session. Nonetheless, its evaluation framework remains constrained by limited topical diversity and less realistic interaction patterns, reducing its applicability to real-world memory agent scenarios.

To address these limitations, we introduce a unified benchmark framework, MemoryAgentBench, specifically designed to evaluate a broad spectrum of memory mechanisms in agent systems. We also provide a framework for memory agent evaluation. In this framework, agents are presented with sequences of textual inputs that simulate multi-turn interactions with users. We reconstructed existing datasets originally developed for longcontext LLM evaluation by segmenting and reconstructing inputs into multiple dialogue chunks and feeding them incrementally to the agent in a time order. However, since these datasets do not fully capture all four targeted memory competencies, we also introduce two new datasets: EventQA and FactConsolidation, designed to evaluate accurate retrieval and selective forgetting, respectively. Our benchmark includes evaluations of state-of-theart commercial memory agents (such as MIRIX and MemGPT), long-context agents that treat the full input as memory, and RAG agents that extend their memory through retrieval methods. We examine how techniques developed for long-context models and RAG transfer to the memory agent setting. By providing a consistent evaluation protocol across diverse agent architectures and datasets, MemoryAgentBench delivers comprehensive insights into agent performance across the four core memory competencies.

Our contributions are summarized as follows:

• Datasets: We reconstruct existing datasets and create two new datasets to construct a comprehensive benchmark, covering four distinct memory competencies.   
• Framework: We provide a unified evaluation framework, and open-source the codebase and datasets to encourage reproducibility and further research.   
• Empirical Study: We implement various simple agents with diverse memory mechanisms, adopt commercial agents, and evaluate these agents on our proposed benchmark. With our results, we show that existing memory agents, while effective in some tasks, still face significant challenges on some aspects.

# 2 Related Work

# 2.1 Benchmarks with Long Input

In this section, we review prior work on long-context benchmarks. Early benchmarks designed for long-context evaluation include LongBench(Bai et al., 2023) and LooGLE(Li et al., 2023), with average input lengths of approximately 20k and 24k tokens, respectively. More recent benchmarks—such as $\infty$ -Bench (Zhang et al., 2024), HELMET(Yen et al., 2024), RULER(Hsieh et al., 2024), NOCHA(Karpinska et al., 2024), NoLiMa (Modarressi et al., 2025) and LongBench V2(Bai et al., 2024)—extend context lengths to over 100k tokens and are primarily intended to evaluate the capabilities of long-context models. However, despite their scale, these benchmarks are not designed to assess memory agents, and no prior work has repurposed them for that goal. More recently, LOCOMO (Maharana et al., 2024), LongMemEval (Wu et al., 2025), RealTalk (Lee et al., 2025) and StoryBench (Wan & Ma, 2025) have been proposed specifically for evaluating memory agents. While promising, LOCOMO still features relatively short conversations (∼9k), and LongMemEval uses synthetic conversations with limited topical diversity, making the dialogues less realistic and potentially less representative of real-world memory use cases. Meanwhile, the evaluation scope of the above benchmarks is not sufficient to comprehensively assess long-term memory from multiple dimensions.

# 2.2 Agents with Memory Mechanisms

Memory mechanisms are attracting more and more attention lately (Wang et al., 2025/02). Recent advancements in LLMs have demonstrated the capability to process extended context lengths, ranging from 100K to over 1 million tokens. For instance, models such as GPT-4o (OpenAI, 2025b) and Claude 3.7 (Anthropic, 2025) can handle inputs of approximately 100K to 200K tokens, while models like Gemini 2.0 Pro (DeepMind, 2025) and the GPT-4.1 series extend this capacity beyond 1 million tokens. These strong long-context capabilities enable a simple yet effective form of memory: storing information directly within the context window. However, this approach is inherently constrained by a hard limit—once the context window is exceeded, earlier information must be discarded.

In parallel, RAG continues to serve as a dominant paradigm for managing excessive context. By retrieving relevant information from earlier context and feeding it to the LLM, RAG allows systems to overcome context length limitations. For example, OpenAI’s recent memory functionality1 combines explicit user preference tracking with retrieval-based methods that reference prior interactions. RAG methods can be broadly classified into three categories: Simple RAG: These methods rely on string-matching techniques such as TF-IDF, BM25 (Robertson & Walker, 1994), and BMX (Li et al., 2024), which are entirely non-neural and operate on string-level similarity. Embedding-based RAG: This class leverages neural encoders, primarily transformers, to map text into dense vector representations (Wu et al., 2022). Early methods like DPR (Karpukhin et al., 2020) and Contriever (Izacard et al., 2021) are based on BERT (Devlin et al., 2019), while more recent models such as Qwen3-Embedding (Zhang et al., 2025) achieve significantly improved retrieval performance. Structure-Augmented RAG: These approaches enhance retrieval with structural representations such as graphs or trees. Representative systems include GraphRAG (Edge et al., 2024), RAPTOR (Sarthi et al., 2024), HippoRAG-V2 (Gutiérrez et al., 2025), Cognee, Zep (Rasmussen et al., 2025), MemoRAG (Qian et al., 2025), Mem0 (Chhikara et al., 2025), MemoryOS (Kang et al., 2025), Memary (kingjulio8238 & Memary contributors, 2024) and Memobase (memodb-io & Memobase contributors, 2025). Despite their effectiveness, RAGbased methods face challenges with ambiguous queries, multi-hop reasoning, and long-range comprehension. When questions require integrating knowledge across an entire session or learning from long, skill-encoding inputs, the retrieval mechanism—limited to the top-k most relevant passages—may fail to surface the necessary information. To address these limitations, Agentic Memory Agents introduce an iterative, decision-driven framework. Rather than relying on a single-pass retrieval, these agents dynamically process the query, retrieve evidence, reflect, and iterate through multiple retrieval and reasoning cycles. Examples include MemGPT (Packer et al., 2023), Self-RAG (Asai et al., 2023), Auto-RAG (Yu et al., 2024), A-MEM (Xu et al., 2025), Mem1 (Zhou et al., 2025), MemAgent (Yu et al., 2025), and MIRIX (Wang & Chen, 2025). This agentic design is particularly effective for resolving ambiguous or multi-step queries. Nonetheless, these methods remain fundamentally constrained by the limitations of RAG—namely, the inability to fully understand or learn from long-range context that is inaccessible via retrieval alone.

# 3 MemoryAgentBench

# 3.1 Dataset Preperation

In this section, we describe how we reconstruct existing datasets and build new ones for evaluating each competency aspect. All datasets with their categories are shown in Table 1. We introduce the details in datasets curation in Appendix A.

Datasets for Accurate Retrieval (AR) We adopt four datasets to evaluate the accurate retrieval capability of memory agents. Three are reconstructed from existing benchmarks, and one is newly created: (1) Document Question Answering: This is a NIAHstyle QA task where a long passage contains single (SH-QA) or multiple (MH-QA) documents answering the input question. The agent must identify and extract relevant snippets from the extended context. (2) LongMemEval: This benchmark evaluates memory agents on long dialogue histories. Although task types like information extraction (IE) or multi-

Table 1: Overview of evaluation datasets. We select datasets that cover various important long-context capabilities. In the table, we underline the datasets we constructed ourselves. AvgL.: Average Context Length (measured using the GPT-4o-mini model’s tokenizer).   

<table><tr><td>Category</td><td>Dataset</td><td>Metrics</td><td>AvgL.</td><td>Description</td></tr><tr><td rowspan="4">Accurate Retrieval</td><td>SH-Doc QA</td><td rowspan="4">Accuracy</td><td>197K</td><td>Single-Hop Gold passage retrieval QA.</td></tr><tr><td>MH-Doc QA</td><td>421K</td><td>Multiple-Hop Gold passage retrieval QA.</td></tr><tr><td>LongMemEval (S*)</td><td>355K</td><td>Dialogues based QA.</td></tr><tr><td>EventQA</td><td>534K</td><td>Novel multiple-choice QA on characters events.</td></tr><tr><td rowspan="6">Test-time Learning</td><td>BANKING77</td><td rowspan="5">Accuracy</td><td rowspan="5">103K</td><td>Banking intent classification, 77 labels.</td></tr><tr><td>CLINC150</td><td>Intent classification, 151 labels.</td></tr><tr><td>NLU</td><td>Task intent classification, 68 labels.</td></tr><tr><td>TREC Coarse</td><td>Question type classification, 6 labels.</td></tr><tr><td>TREC Fine</td><td>Question type classification, 50 labels.</td></tr><tr><td>Movie Recommendation</td><td>Recall@5</td><td>1.44M</td><td>Recommend movies based on provided dialogues examples.</td></tr><tr><td rowspan="2">Long Range Understanding</td><td>∞Bench-Sum</td><td>F1-Score</td><td>172K</td><td>Novel summarization with entity replacement.</td></tr><tr><td>Detective QA</td><td>Accuracy</td><td>124K</td><td>Long-range reasoning QA on detective novels.</td></tr><tr><td rowspan="2">Selective Forgetting</td><td>FactConsolidation-SH</td><td rowspan="2">Accuracy</td><td rowspan="2">262K</td><td>Single hop reasoning in facts judgment.</td></tr><tr><td>FactConsolidation-MH</td><td>Multiple hop reasoning in facts judgment.</td></tr></table>

session reasoning are included, most tasks can be reformulated as single-retrieval problems requiring agents to retrieve the correct segments spanning a long multi-turn conversation. We reformulated chat history into five long dialogues ( $\sim$ 355K tokens) with 300 questions (LongMemEval (S*) in Table 1). We create LongMemEval (S $^ *$ ) specifically for increasing the number of questions per context, mitigating the exhaustive needs of reconstructing the memory for each question. (3) EventQA (ours): We introduce EventQA this reasoning style NIAH task to evaluate agents’ ability to recall and reason about temporal sequences in long-form narratives. In this dataset, the agent is required to read a novel and select the correct event from a series of candidates after receiving up-to five previous events. Unlike other long-range narrative text datasets that require extensive manual annotation (Zhang et al., 2024; Xu et al., 2024), our dataset is built through a fully automated pipeline, making the process more efficient and scalable. Moreover, this pipeline can be directly applied to other novel-style texts.

Datasets for Test-Time Learning (TTL) We evaluate TTL via two task categories: (1) Multi-Class Classification (MCC): We reconstructed five classification datasets used in prior TTL work (Bertsch et al., 2024; Yen et al., 2024): BANKING77 (Casanueva et al., 2020), CLINC150 (Larson et al., 2019), TREC-Coarse, TREC-Fine (Li & Roth, 2002), and NLU (Liu et al., 2019). Each task requires the agent to map sentences to class labels, leveraging previously seen labeled examples in context. (2) Recommendation: Based on the setup from (Li et al., 2018; He et al., 2023), we construct a dataset to evaluate movie recommendation via dialogue history. The agent is exposed to thousands of movierelated dialogue turns and is asked to recommend twenty relevant movies based on the long interaction history.

Datasets for Long Range Understanding (LRU) We evaluate LRU via two tasks: (1) Novel Summarization (Summ.): We adopt the Summarization task En.Sum from $\infty$ -Bench (Zhang et al., 2024). The agent is required to analyze and organize the plot and characters of the novel, and then compose a summary of 1000 to 1200 words. (2) Detective QA (Det QA): We also create a difficult question set from Detective QA (Xu et al., 2024), which include ten novels with 71 questions and these questions require agents to do reasoning over a longer narrative range.

Datasets for Selective Forgetting (SF) To assess whether an agent can forget out of date memory and reason over them, we construct a new dataset called FactConsolidation. Specifically, We build this benchmark using counterfactual edit pairs from MQUAKE (Zhong et al., 2023). Each pair contains a true fact and a rewritten, contradictory version. These are ordered such that the rewritten (new) fact appears after the original, simulating a realistic update scenario. We concatenate multiple such edit pairs to create long contexts of length 6K, 32K, 64K, 262K. We then adpot MQUAKE’s original questions and categorize them into: (1) FactConsolidation-SH (Ours) (SH means Single-Hop), requiring direct factual recall (e.g., “Which country was tool $A$ created in?”), and (2) FactConsolidation-MH (Ours) (MH refers to Multi-Hop), requiring inference over multiple facts (e.g., “What is the location of death of the spouse of person $B$ ?”). Agents

are prompted to prioritize later information in case of conflict and reason based on the final memory state. This setup directly evaluates the strength and consistency of selective forgetting over long sequences.

# 3.2 Different Categories of Memory Agents

We evaluate three major types of memory agents that reflect common strategies for handling long-term information: Long-Context Agents, RAG Agents, and Agentic Memory Agents. These approaches differ in how they store, retrieve, and reason over past inputs.

(1) Long Context Agents Modern language models often support extended context windows ranging from 128K to over 1M tokens. A straightforward strategy for memory is to maintain a context buffer of the most recent tokens. For example, in a model with a 128Ktoken limit, the agent concatenates all incoming chunks until the total exceeds the window size. Once the limit is reached, the earliest chunks are evicted in a FIFO (first-in, first-out) manner. This agent design relies solely on positional recency and assumes the model can attend effectively over the current context window. (2) RAG Agents RAG-based agents address context limitations by storing past information in an external memory pool and retrieving relevant content as needed. We consider three RAG variants: Simple RAG Agents: All input chunks are stored as raw text. During inference, a keyword or rule-based string matching mechanism retrieves relevant passages. Embedding-based RAG Agents: Each input chunk is embedded and saved. At query time, the agent embeds the query and performs retrieval using cosine similarity between embeddings. Structure-Augmented RAG Agents: After ingesting all input chunks, the agent constructs a structured representation (e.g., knowledge graph or event timeline). Subsequent queries are answered based on this structured memory. (3) Agentic Memory Agents Agentic memory agents extend beyond static memory stores by employing agentic loops—iterative reasoning cycles in which the agent may reformulate questions, perform memory lookups, and update its working memory. These agents are designed to simulate a more human-like process of recalling, verifying, and integrating knowledge.

# 3.3 Datasets and Agents Formulation

Datasets Formulation We standardize all datasets into the format: $c _ { 1 } , c _ { 2 } , \cdots , c _ { n }$ (chunks), $q _ { 1 } , q _ { 2 } , \cdots , q _ { m }$ (questions), and $a _ { 1 } , a _ { 2 } , \cdots , a _ { m }$ (answers), where $c _ { i }$ denotes the $i$ -th chunk wrapped to construct a user message with instructions of memorizing the content in a sequential input, and $c _ { 1 } , c _ { 2 } , \cdots , c _ { n }$ represents a single conversation. Each chunk is accompanied by instructions prompting the agent to memorize its contents. Example prompts are provided in Appendix C.1. When curating datasets like EventQA and FactConsolidation, we deliberately design scenarios where multiple questions follow a single context. This allows us to probe the model’s memory multiple times with one sequential injection. For example, in LME (S*), five contexts are paired with 300 questions (shown in Table 5 in Appendix A). This design choice reflects a key trend: as LLMs support increasingly long context windows and memory agents become more capable of handling extended inputs, evaluation datasets must also scale accordingly. Injecting 1M tokens for just one question is resource-inefficient, whereas associating the same input with many questions provides significantly higher utility.

Agents Formulation In our framework, all agents are required to take the chunks one by one, absorb them into memory, and incrementally update the memory. After seeing all the chunks, we ask the agent to answer the related questions.

# 4 Experiments

# 4.1 Experimental Setup

The datasets are split into four categories and the statistics of all datasets are also shown in Table 5. The evaluation metrics for all datasets are shown in Table 1, along with more dataset details. For the agents, as described in Section 3.2, we consider three categories of agents: Long-Context Agents, RAG agents and Agentic Memory Agents, where $R A G$

Table 2: Overall Performance Comparison. In the absence of a specified model, All RAG agents and commercial memory agents use GPT-4o-mini as the backbone. Thus we highlight the performance of GPT-4o-mini as the reference. FC-SH and FC-MH mean FactConsolidation Single Hop and FactConsolidation Multi Hop, respectively. Best viewed in colors.   

<table><tr><td rowspan="2">Agent Type</td><td colspan="4">AR</td><td colspan="4">TTL</td><td colspan="3">LRU</td><td colspan="3">SF</td><td rowspan="2">Overall Scores</td></tr><tr><td>SH-QA</td><td>MH-QA</td><td>LME(S*)</td><td>EventQA</td><td>Avg.</td><td>MCC</td><td>Recom.</td><td>Avg.</td><td>Summ.</td><td>DetQA</td><td>Avg.</td><td>FC-SH</td><td>FC-MH</td><td>Avg.</td></tr><tr><td colspan="16">Long-Context Agents</td></tr><tr><td>GPT-4o</td><td>72.0</td><td>51.0</td><td>32.0</td><td>77.2</td><td>58.1</td><td>87.6</td><td>12.3</td><td>50.0</td><td>32.2</td><td>77.5</td><td>54.9</td><td>60.0</td><td>5.0</td><td>32.5</td><td>48.8</td></tr><tr><td>GPT-4o-mini</td><td>64.0</td><td>43.0</td><td>30.7</td><td>59.0</td><td>49.2</td><td>82.0</td><td>15.1</td><td>48.6</td><td>28.9</td><td>63.4</td><td>46.2</td><td>45.0</td><td>5.0</td><td>25.0</td><td>42.2</td></tr><tr><td>GPT-4.1-mini</td><td>83.0</td><td>66.0</td><td>55.7</td><td>82.6</td><td>71.8</td><td>75.6</td><td>16.7</td><td>46.2</td><td>41.9</td><td>56.3</td><td>49.1</td><td>36.0</td><td>5.0</td><td>20.5</td><td>46.9</td></tr><tr><td>Gemini-2.0-Flash</td><td>87.0</td><td>59.0</td><td>47.0</td><td>67.2</td><td>65.1</td><td>84.0</td><td>8.7</td><td>46.4</td><td>23.9</td><td>59.2</td><td>41.6</td><td>30.0</td><td>3.0</td><td>16.5</td><td>42.4</td></tr><tr><td>Claude-3.7-Sonnet</td><td>77.0</td><td>53.0</td><td>34.0</td><td>74.6</td><td>59.7</td><td>89.4</td><td>18.3</td><td>53.9</td><td>52.5</td><td>71.8</td><td>62.2</td><td>43.0</td><td>2.0</td><td>22.5</td><td>49.6</td></tr><tr><td>GPT-4o-mini</td><td>64.0</td><td>43.0</td><td>30.7</td><td>59.0</td><td>49.2</td><td>82.0</td><td>15.1</td><td>48.6</td><td>28.9</td><td>63.4</td><td>46.2</td><td>45.0</td><td>5.0</td><td>25.0</td><td>42.3</td></tr><tr><td colspan="16">Simple RAG Agents</td></tr><tr><td>BM25</td><td>66.0</td><td>56.0</td><td>45.3</td><td>74.6</td><td>60.5</td><td>75.4</td><td>13.6</td><td>44.5</td><td>19.0</td><td>52.1</td><td>35.6</td><td>48.0</td><td>3.0</td><td>25.5</td><td>41.5</td></tr><tr><td colspan="16">Embedding RAG Agents</td></tr><tr><td>Contriever</td><td>22.0</td><td>31.0</td><td>15.7</td><td>66.8</td><td>33.9</td><td>70.6</td><td>15.2</td><td>42.9</td><td>17.2</td><td>42.3</td><td>29.8</td><td>18.0</td><td>7.0</td><td>12.5</td><td>29.8</td></tr><tr><td>Text-Embed-3-Small</td><td>60.0</td><td>44.0</td><td>48.3</td><td>63.0</td><td>53.8</td><td>70.0</td><td>15.3</td><td>42.7</td><td>17.7</td><td>54.9</td><td>36.3</td><td>28.0</td><td>3.0</td><td>15.5</td><td>37.1</td></tr><tr><td>Text-Embed-3-Large</td><td>54.0</td><td>44.0</td><td>50.3</td><td>70.0</td><td>54.6</td><td>72.4</td><td>16.2</td><td>44.3</td><td>18.2</td><td>56.3</td><td>37.3</td><td>28.0</td><td>4.0</td><td>16.0</td><td>38.0</td></tr><tr><td>Qwen3-Embedding-4B</td><td>57.0</td><td>47.0</td><td>43.3</td><td>71.4</td><td>54.7</td><td>78.0</td><td>12.2</td><td>45.1</td><td>14.8</td><td>59.2</td><td>37.0</td><td>29.0</td><td>3.0</td><td>16.0</td><td>38.2</td></tr><tr><td colspan="16">Structure-Augmented RAG Agents</td></tr><tr><td>RAPTOR</td><td>29.0</td><td>38.0</td><td>34.3</td><td>45.8</td><td>36.8</td><td>59.4</td><td>12.3</td><td>35.9</td><td>13.4</td><td>42.3</td><td>27.9</td><td>14.0</td><td>1.0</td><td>7.5</td><td>27.0</td></tr><tr><td>GraphRAG</td><td>47.0</td><td>47.0</td><td>35.0</td><td>34.4</td><td>40.9</td><td>39.8</td><td>9.8</td><td>24.8</td><td>0.4</td><td>39.4</td><td>19.9</td><td>14.0</td><td>2.0</td><td>8.0</td><td>23.4</td></tr><tr><td>MemoRAG</td><td>29.0</td><td>33.0</td><td>20.0</td><td>56.0</td><td>34.5</td><td>77.0</td><td>13.1</td><td>45.1</td><td>9.2</td><td>50.7</td><td>30.0</td><td>21.0</td><td>7.0</td><td>14.0</td><td>30.9</td></tr><tr><td>HippoRAG-v2</td><td>76.0</td><td>66.0</td><td>50.7</td><td>67.6</td><td>65.1</td><td>61.4</td><td>10.2</td><td>35.8</td><td>14.6</td><td>57.7</td><td>36.2</td><td>54.0</td><td>5.0</td><td>29.5</td><td>41.6</td></tr><tr><td>Mem0</td><td>25.0</td><td>32.0</td><td>36.0</td><td>37.5</td><td>32.6</td><td>32.4</td><td>10.0</td><td>21.2</td><td>4.8</td><td>36.6</td><td>20.7</td><td>18.0</td><td>2.0</td><td>10.0</td><td>21.1</td></tr><tr><td>Cognee</td><td>31.0</td><td>26.0</td><td>29.3</td><td>26.8</td><td>28.3</td><td>35.4</td><td>10.1</td><td>22.8</td><td>2.3</td><td>29.6</td><td>16.0</td><td>28.0</td><td>3.0</td><td>15.5</td><td>20.6</td></tr><tr><td>Zep</td><td>44.0</td><td>25.0</td><td>38.3</td><td>42.5</td><td>37.5</td><td>62.8</td><td>12.1</td><td>37.5</td><td>4.2</td><td>28.2</td><td>16.2</td><td>7.0</td><td>3.0</td><td>5.0</td><td>24.0</td></tr><tr><td colspan="16">Agentic Memory Agents</td></tr><tr><td>Self-RAG</td><td>35.0</td><td>42.0</td><td>25.7</td><td>31.8</td><td>33.6</td><td>11.6</td><td>12.8</td><td>12.2</td><td>0.9</td><td>35.2</td><td>18.1</td><td>19.0</td><td>3.0</td><td>11.0</td><td>18.7</td></tr><tr><td>MemGPT</td><td>41.0</td><td>38.0</td><td>32.0</td><td>26.2</td><td>34.3</td><td>67.6</td><td>14.0</td><td>40.8</td><td>2.5</td><td>42.3</td><td>22.4</td><td>28.0</td><td>3.0</td><td>15.5</td><td>28.3</td></tr><tr><td>MIRIX</td><td>62.0</td><td>61.0</td><td>37.3</td><td>29.8</td><td>47.5</td><td>38.4</td><td>9.8</td><td>24.1</td><td>9.9</td><td>40.8</td><td>25.4</td><td>14.0</td><td>2.0</td><td>8.0</td><td>26.2</td></tr><tr><td>MIRIX (4.1-mini)</td><td>73.0</td><td>75.0</td><td>51.0</td><td>53.0</td><td>63.0</td><td>61.0</td><td>10.3</td><td>35.7</td><td>18.9</td><td>62.0</td><td>40.5</td><td>20.0</td><td>3.0</td><td>11.5</td><td>37.7</td></tr></table>

Agents can be further split into Simple RAG Agents, Embedding-based RAG Agents and Structure-Augmented RAG Agents. We give the detailed introduction of each memory agent in Appendix B. For chunk size settings, we choose a chunk size of 512 for the SH-Doc QA, MH-Doc QA, and LME(S*) tasks in AR, as well as for all tasks in SF. This is mainly because these tasks are composed of long texts synthesized from multiple short texts. For other tasks, we use a chunk size of 4096. Considering computational overhead and API cost, we uniformly use a chunk size of 4096 for the Mem0, Cognee, Zep, and MIRIX. We report the settings of the chunk size in Table 14 in Appendix D.

# 4.2 Overall Performance Comparison

Table 2 presents the overall performance across different benchmarks. We summarize the key findings as follows: (1) Superiority of RAG methods in Accurate Retrieval Tasks. Most RAG Agents are better than the backbone model “GPT-4o-mini” in the tasks within the Accurate Retrieval Category. This matches our intuition where RAG agents typically excel at extracting a small snippet of text that is crucial for answering the question. (2) Superiority of Long-Context Models in Test-Time Learning and Long-Range Understanding. Long-context models achieve the best performance on TTL and LRU. This highlights a fundamental limitation of RAG methods and commercial memory agents, which still follow an agentic RAG paradigm. These systems retrieve only partial information from the past context, lacking the ability to capture a holistic understanding of the input—let alone perform learning across it. (3) Limitation of All Existing Methods on Selective Forgetting. Although being a well-discussed task in model-editing community (Mitchell et al., 2022; Fang et al., 2024), forgetting out-of-date memory poses a significant challenge on memory agents. We observe that all methods fail on the multi-hop situation (with achieving at most 7% accuracy). Only long context agents can achieve fairly reasonable results on single-hop scenarios. In Section 4.3.4, we show that current reasoning models can have much better performance, while it does not change the conclusion that Selective Forgetting still poses a significant challenge to all memory mechanisms.

# 4.3 Analysis and Ablation Study

In this section, we present experiments and analysis along five dimensions: input chunk size, retrieval top- $k$ , backbone model, dataset validation, and computational latency. Additional

![](images/d8a330b457c87030de2ae5df4a1796aab650b53f2e1f00feaa6d2f02bcbc7156.jpg)  
(a) SH-Doc QA performance

![](images/b35613a131699c2c34713d9fa5a2a3433b37fcf6dec0f7bf97aa5b54e6d50e90.jpg)  
(b) $\infty$ Bench-Sum performance

![](images/f4f2dfb7669a9e9a1500b60228ae85fc3841bd074fd0b4ffc6fbd7c9becd3586.jpg)  
Figure 2: Performances on SH-Doc QA and $\infty$ -Bench-Sum with different chunk sizes.   
Figure 3: The accuracies on different benchmarks when varying the retrieval top-k to be 2, 5 and 10.

results are provided in the appendix, including context length analysis (Appendix D.4), latency and GPU memory usage comparisons (Appendix D.5, D.6), as well as further details on chunk size and top- $k$ ablations (Appendix D.2, D.3).

# 4.3.1 Ablation Study on Input Chunk Size

To understand how chunk size impacts performance, particularly for RAG methods and agentic memory agents, we conduct an additional analysis where we vary the chunk size while fixing the number of retrieved chunks to 10. The results are presented in Figure 2. From the figure, we observe that when resources permit, using smaller chunk sizes and increasing the number of retrieval calls during memory construction can improve performance on Accurate Retrieval (AR) tasks. Finer-grained segmentation enhances the relevance of retrieved information, particularly for embedding-based methods. However, for tasks requiring Long-Range Understanding (LRU), varying the chunk size hurts the performance. This is likely because RAG methods are inherently less suited for tasks that demand integration of information across a large, coherent context.

# 4.3.2 Ablation Study on Retrieval TopK

In our experiments, although we report most results with the number of retrieved chunks set to 10 in Table 2, we also conducted ablation studies with varying retrieval sizes. A subset of these results is visualized in Figure 3, with the full results provided in Table 8 in Appendix D. The results indicate that increasing the number of retrieved chunks generally improves performance across most tasks. It is worth noting that, with a chunk size of 4096 tokens, retrieving 10 chunks already yields an input of approximately 40k tokens. This places significant demands on model capacity. Due to this high token volume, we do not evaluate settings with 20 retrieved chunks.

# 4.3.3 Ablation Study on Backbone Model

To investigate how different backbone models impact the performance of various memory agents, we experimented with three different backbone models and selected four representative methods from both the RAG Agents and Agentic Memory categories. The complete

Table 3: Performance comparison on three different backbone LLMs and four representative memory agents. We choose one dataset from every competency to evaluate agent performance.   

<table><tr><td>Agent Type</td><td>Backbone Model</td><td>EventQA</td><td>Recom</td><td>∞Bench-Sum</td><td>FactCon-SH</td><td>Avg.</td></tr><tr><td rowspan="3">BM25</td><td>GPT-4o-mini</td><td>74.6</td><td>13.6</td><td>19.0</td><td>48.0</td><td>38.8</td></tr><tr><td>GPT-4.1-mini</td><td>76.4</td><td>14.0</td><td>19.4</td><td>51.0</td><td>40.2</td></tr><tr><td>Gemini-2.0-Flash</td><td>70.8</td><td>10.0</td><td>18.9</td><td>47.0</td><td>36.7</td></tr><tr><td rowspan="3">Text-Embed-3-Small</td><td>GPT-4o-mini</td><td>63.0</td><td>15.3</td><td>17.7</td><td>28.0</td><td>31.0</td></tr><tr><td>GPT-4.1-mini</td><td>62.0</td><td>15.5</td><td>17.9</td><td>30.0</td><td>31.4</td></tr><tr><td>Gemini-2.0-Flash</td><td>64.0</td><td>10.3</td><td>17.2</td><td>27.0</td><td>29.6</td></tr><tr><td rowspan="3">GraphRAG</td><td>GPT-4o-mini</td><td>34.4</td><td>9.8</td><td>0.4</td><td>14.0</td><td>14.7</td></tr><tr><td>GPT-4.1-mini</td><td>39.0</td><td>10.3</td><td>1.2</td><td>16.0</td><td>16.6</td></tr><tr><td>Gemini-2.0-Flash</td><td>36.2</td><td>7.2</td><td>0.8</td><td>13.0</td><td>14.3</td></tr><tr><td rowspan="2">MIRIX</td><td>GPT-4o-mini</td><td>29.8</td><td>9.8</td><td>9.9</td><td>14.0</td><td>15.9</td></tr><tr><td>GPT-4.1-mini</td><td>53.0 (23.2↑)</td><td>10.3 (0.5↑)</td><td>18.9 (9.0↑)</td><td>20.0 (6.0↑)</td><td>25.6 (9.7↑)</td></tr></table>

experimental results are presented in Table 3. Our findings show that for RAG Agents, once the backbone is sufficiently strong, it no longer serves as the main performance bottleneck. Compared to the default setup, upgrading to a more powerful model like GPT-4.1-mini yields only marginal improvements. In contrast, the main results in Table 2 for the MIRIX method under the Agentic Memory category, using a stronger backbone leads to substantial performance gains. This suggests that future advances in backbone models could further boost the effectiveness of Agentic Memory methods.

# 4.3.4 Validation of Dataset FactConsolidation

As the performance of different models on this dataset remains drastically low, we turn to the stronger reasoning model o4-mini and validate our dataset by checking the performance of o4-mini on a smaller version of this dataset. The results are shown in Table 4. We found that on the 6K version of the FactCon-SH dataset, both models perform well and are

generally able to complete the task effectively. However, their performance drops when the context length increases to 32K. Similarly, on the 6K version of the FactCon-MH dataset, the stronger O4-mini reasoning model achieves a decent score of 80.0, but its performance significantly drops to 14.0 when the context window reaches 32K. This indicates that our dataset is solvable under short-context settings, but current memory agents still lack strong long-range reasoning capabilities, making them unable to handle the task when presented with longer historical inputs.

Table 4: Performances of reasoning models on the dataset FactConsolidation.   

<table><tr><td></td><td colspan="2">FactCon-SH</td><td colspan="2">FactCon-MH</td></tr><tr><td></td><td>6K</td><td>32K</td><td>6K</td><td>32K</td></tr><tr><td>GPT-4o</td><td>92.0</td><td>88.0</td><td>28.0</td><td>10.0</td></tr><tr><td>O4-mini</td><td>100.0</td><td>61.0</td><td>80.0</td><td>14.0</td></tr></table>

# 5 Conclusion

In this paper, we introduce MemoryAgentBench, a unified benchmark designed to evaluate memory agents across four essential competencies: accurate retrieval, test-time learning, long-range understanding, and selective forgetting. While prior benchmarks focus largely on skill execution or long-context question answering, MemoryAgentBench fills a critical gap by assessing how agents store, update, and utilize long-term information across multiturn interactions. To build this benchmark, we restructure existing datasets and propose two new ones—EventQA and FactConsolidation—tailored to stress specific memory behaviors often overlooked in prior work. We evaluate a wide spectrum of agents, including long-context models, RAG-based systems, and commercial memory agents, under a consistent evaluation protocol. Our results reveal that, despite recent advances, current memory agents still exhibit substantial limitations when faced with tasks requiring dynamic memory updates and long-range consistency. One limitation of our work is that due to budget constraints, so we could only conduct experiments on some relatively representative Memory Agents. As future work, we aim to provide more evaluation results for more memory agents.

# References

Michael C. Anderson and James H. Neely. Interference and inhibition in memory retrieval. In Elizabeth Ligon Bjork and Robert A. Bjork (eds.), Memory, Handbook of Perception and Cognition, pp. 237–313. Academic Press, San Diego, CA, 2 edition, 1996. URL https://memorycontrol.net/an1996.pdf.   
Anthropic. Claude 3.7 sonnet, 2025. URL https://www.anthropic.com/news/ claude-3-7-sonnet. This announcement introduces Claude 3.7 Sonnet, described as Anthropic’s most intelligent model to date and the first hybrid reasoning model generally available on the market.   
Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In The Twelfth International Conference on Learning Representations, 2023.   
Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. arXiv preprint arXiv:2308.14508, 2023.   
Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, et al. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. arXiv preprint arXiv:2412.15204, 2024.   
Amanda Bertsch, Maor Ivgi, Emily Xiao, Uri Alon, Jonathan Berant, Matthew R Gormley, and Graham Neubig. In-context learning with long-context models: An in-depth exploration. arXiv preprint arXiv:2405.00200, 2024.   
Iñigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić. Efficient intent detection with dual sentence encoders. In Tsung-Hsien Wen, Asli Celikyilmaz, Zhou Yu, Alexandros Papangelis, Mihail Eric, Anuj Kumar, Iñigo Casanueva, and Rushin Shah (eds.), Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, pp. 38–45, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.nlp4convai-1.5. URL https://aclanthology.org/2020. nlp4convai-1.5/.   
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.   
DeepMind. Gemini pro, 2025. URL https://deepmind.google/technologies/gemini/ pro/. This page provides an overview of Gemini Pro, highlighting its advanced capabilities and applications in various fields.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.   
Hermann Ebbinghaus. Memory: A contribution to experimental psychology. Annals of Neurosciences, 20(4):155–156, 2013. doi: 10.5214/ans.0972.7531.200408. URL https: //pubmed.ncbi.nlm.nih.gov/25206041/.   
Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024.   
Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Shi Jie, Xiang Wang, Xiangnan He, and Tat-Seng Chua. Alphaedit: Null-space constrained knowledge editing for language models. arXiv preprint arXiv:2410.02355, 2024.

Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From rag to memory: Non-parametric continual learning for large language models. arXiv preprint arXiv:2502.14802, 2025.   
Zhankui He, Zhouhang Xie, Rahul Jha, Harald Steck, Dawen Liang, Yesu Feng, Bodhisattwa Prasad Majumder, Nathan Kallus, and Julian McAuley. Large language models as zero-shot conversational recommenders. In Proceedings of the 32nd ACM international conference on information and knowledge management, pp. 720–730, 2023.   
Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. RULER: What’s the Real Context Size of Your Long-Context Language Models?, August 2024. URL http://arxiv.org/abs/2404.06654. arXiv:2404.06654 [cs].   
Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Bowei Xia, Tao Sun, Ziyu Ye, Zhaoxuan Jin, Yingru Li, Zeyu Zhang, Yifeng Wang, Qianshuo Ye, Ping Luo, and Guohao Li. Owl: Optimized workforce learning for general multi-agent assistance in real-world task automation, 2025. URL https://github.com/camel-ai/owl.   
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118, 2021.   
William James. The Principles of Psychology, volume 1. Macmillan, London, 1890. URL https://books.google.com/books?id=JO1RL9BcI44C.   
Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? arXiv preprint arXiv:2310.06770, 2023.   
Jiazheng Kang, Mingming Ji, Zhe Zhao, and Ting Bai. Memory os of ai agent. arXiv preprint arXiv:2506.06326, 2025.   
Marzena Karpinska, Katherine Thai, Kyle Lo, Tanya Goyal, and Mohit Iyyer. One thousand and one pairs: A" novel" challenge for long-context language models. arXiv preprint arXiv:2406.16264, 2024.   
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In EMNLP (1), pp. 6769–6781, 2020.   
kingjulio8238 and Memary contributors. Memary: The open source memory layer for ai agents, 2024. URL https://github.com/kingjulio8238/Memary. GitHub repository.   
Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, and Jason Mars. An evaluation dataset for intent classification and out-of-scope prediction. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 1311–1316, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1131. URL https://aclanthology.org/D19-1131/.   
Dong-Ho Lee, Adyasha Maharana, Jay Pujara, Xiang Ren, and Francesco Barbieri. Realtalk: A 21-day real-world dataset for long-term conversation. arXiv preprint arXiv:2502.13270, 2025.   
Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. Loogle: Can long-context language models understand long contexts? arXiv preprint arXiv:2311.04939, 2023.   
Raymond Li, Samira Ebrahimi Kahou, Hannes Schulz, Vincent Michalski, Laurent Charlin, and Chris Pal. Towards deep conversational recommendations. Advances in neural information processing systems, 31, 2018.

Xianming Li, Julius Lipp, Aamir Shakir, Rui Huang, and Jing Li. Bmx: Entropy-weighted similarity and semantic-enhanced lexical search. arXiv preprint arXiv:2408.06643, 2024.   
Xin Li and Dan Roth. Learning question classifiers. In COLING 2002: The 19th International Conference on Computational Linguistics, 2002. URL https://aclanthology. org/C02-1150/.   
Kevin Lin, Charlie Snell, Yu Wang, Charles Packer, Sarah Wooders, Ion Stoica, and Joseph E Gonzalez. Sleep-time compute: Beyond inference scaling at test-time. arXiv preprint arXiv:2504.13171, 2025.   
Xingkun Liu, Arash Eshghi, Pawel Swietojanski, and Verena Rieser. Benchmarking natural language understanding services for building conversational agents, 2019. URL https: //arxiv.org/abs/1903.05566.   
Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of llm agents. arXiv preprint arXiv:2402.17753, 2024.   
Vasilije Markovic, Lazar Obradovic, Laszlo Hajdu, and Jovan Pavlovic. Optimizing the interface between knowledge graphs and llms for complex reasoning. arXiv preprint arXiv:2505.24478, 2025.   
James L. McClelland, Bruce L. McNaughton, and Randall C. O’Reilly. Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory. Psychological Review, 102(3):419–457, 1995. doi: 10.1037/0033-295X.102.3.419. URL https://pubmed. ncbi.nlm.nih.gov/7624455/.   
memodb-io and Memobase contributors. Memobase: Profile-based long-term memory for ai applications, 2025. URL https://github.com/memodb-io/memobase. GitHub repository.   
Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a transformer. In ICLR. OpenReview.net, 2023.   
Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. In The Twelfth International Conference on Learning Representations, 2023.   
Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, and Chelsea Finn. Memory-based model editing at scale. In ICML, volume 162 of Proceedings of Machine Learning Research, pp. 15817–15831. PMLR, 2022.   
Ali Modarressi, Hanieh Deilamsalehy, Franck Dernoncourt, Trung Bui, Ryan A Rossi, Seunghyun Yoon, and Hinrich Schütze. Nolima: Long-context evaluation beyond literal matching. arXiv preprint arXiv:2502.05167, 2025.   
Magnus Müller and Gregor Žunič. Browser use: Enable ai to control your browser, 2024. URL https://github.com/browser-use/browser-use.   
OpenAI. New embedding models and api updates, 2024. URL https://openai.com/index/ new-embedding-models-and-api-updates/.   
OpenAI. Introducing gpt-4.1 in the api, 2025a. URL https://openai.com/index/ gpt-4-1/.   
OpenAI. Gpt-4o system card, 2025b. URL https://openai.com/index/ gpt-4o-system-card/. This report outlines the safety work carried out prior to releasing GPT-4o including external red teaming, frontier risk evaluations according to our Preparedness Framework, and an overview of the mitigations we built in to address key risk areas.

Charles Packer, Vivian Fang, Shishir_G Patil, Kevin Lin, Sarah Wooders, and Joseph_E Gonzalez. Memgpt: Towards llms as operating systems. 2023.   
Hongjin Qian, Zheng Liu, Peitian Zhang, Kelong Mao, Defu Lian, Zhicheng Dou, and Tiejun Huang. Memorag: Boosting long context processing with global memory-enhanced retrieval augmentation. In Proceedings of the ACM on Web Conference 2025, pp. 2366– 2377, 2025.   
Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025.   
Stephen E Robertson and Steve Walker. Some simple effective approximations to the 2- poisson model for probabilistic weighted retrieval. In SIGIR’94: Proceedings of the Seventeenth Annual International ACM-SIGIR Conference on Research and Development in Information Retrieval, organised by Dublin City University, pp. 232–241. Springer, 1994.   
Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D Manning. Raptor: Recursive abstractive processing for tree-organized retrieval. In The Twelfth International Conference on Learning Representations, 2024.   
Luanbo Wan and Weizhi Ma. Storybench: A dynamic benchmark for evaluating long-term memory with multi turns. arXiv preprint arXiv:2506.13356, 2025.   
Cunxiang Wang, Ruoxi Ning, Boqi Pan, Tonghui Wu, Qipeng Guo, Cheng Deng, Guangsheng Bao, Qian Wang, and Yue Zhang. Novelqa: A benchmark for long-range novel question answering. arXiv preprint arXiv:2403.12766, 2024a.   
Minzheng Wang, Longze Chen, Fu Cheng, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan Xu, Lei Zhang, Run Luo, et al. Leave no document behind: Benchmarking long-context llms with extended multi-doc qa. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 5627–5646, 2024b.   
Xingyao Wang, Boxuan Li, Yufan Song, Frank F Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, et al. Openhands: An open platform for ai software developers as generalist agents. In The Thirteenth International Conference on Learning Representations, 2024c.   
Yu Wang and Xi Chen. Mirix: Multi-agent memory system for llm-based agents. arXiv preprint arXiv:2507.07957, 2025.   
Yu Wang, Xinshuang Liu, Xiusi Chen, Sean O’Brien, Junda Wu, and Julian McAuley. Selfupdatable large language models by integrating context into model parameters. In The Thirteenth International Conference on Learning Representations.   
Yu Wang, Yifan Gao, Xiusi Chen, Haoming Jiang, Shiyang Li, Jingfeng Yang, Qingyu Yin, Zheng Li, Xian Li, Bing Yin, et al. Memoryllm: Towards self-updatable large language models. arXiv preprint arXiv:2402.04624, 2024d.   
Yu Wang, Ruihan Wu, Zexue He, Xiusi Chen, and Julian McAuley. Large scale knowledge washing. arXiv preprint arXiv:2405.16720, 2024e.   
Yu Wang, Dmitry Krotov, Yuanzhe Hu, Yifan Gao, Wangchunshu Zhou, Julian McAuley, Dan Gutfreund, Rogerio Feris, and Zexue He. M+: Extending memoryLLM with scalable long-term memory. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=OcqbkROe8J.   
Yu Wang, Chi Han, Tongtong Wu, Xiaoxin He, Wangchunshu Zhou, Nafis Sadeq, Xiusi Chen, Zexue He, Wei Wang, Gholamreza Haffari, Heng Ji, and Julian J. McAuley. Towards lifespan cognitive systems. TMLR, 2025/02.

Maria Wimber, Arjen Alink, Ian Charest, Nikolaus Kriegeskorte, and Michael C. Anderson. Retrieval induces adaptive forgetting of competing memories via cortical pattern suppression. Nature Neuroscience, 18(4):582–589, 2015. doi: 10.1038/nn.3973. URL https://www.nature.com/articles/nn.3973.   
Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. Longmemeval: Benchmarking chat assistants on long-term interactive memory. In The Thirteenth International Conference on Learning Representations, 2025.   
Qiyu Wu, Chongyang Tao, Tao Shen, Can Xu, Xiubo Geng, and Daxin Jiang. Pcl: Peercontrastive learning with diverse augmentations for unsupervised sentence embeddings. arXiv preprint arXiv:2201.12093, 2022.   
Wujiang Xu, Kai Mei, Hang Gao, Juntao Tan, Zujie Liang, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.   
Zhe Xu, Jiasheng Ye, Xiaoran Liu, Xiangyang Liu, Tianxiang Sun, Zhigeng Liu, Qipeng Guo, Linlin Li, Qun Liu, Xuanjing Huang, et al. Detectiveqa: Evaluating long-context reasoning on detective novels. arXiv preprint arXiv:2409.02465, 2024.   
Howard Yen, Tianyu Gao, Minmin Hou, Ke Ding, Daniel Fleischer, Peter Izsak, Moshe Wasserblat, and Danqi Chen. Helmet: How to evaluate long-context language models effectively and thoroughly. arXiv preprint arXiv:2410.02694, 2024.   
Zhangyue Yin, Qiushi Sun, Qipeng Guo, Zhiyuan Zeng, Qinyuan Cheng, Xipeng Qiu, and Xuan-Jing Huang. Explicit memory learning with expectation maximization. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 16618–16635, 2024.   
Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, et al. Memagent: Reshaping longcontext llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259, 2025.   
Tian Yu, Shaolei Zhang, and Yang Feng. Auto-rag: Autonomous retrieval-augmented generation for large language models. arXiv preprint arXiv:2411.19443, 2024.   
Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Hao, Xu Han, Zhen Thai, Shuo Wang, Zhiyuan Liu, et al. ∞bench: Extending long context evaluation beyond 100k tokens. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15262–15277, 2024.   
Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.   
Zexuan Zhong, Zhengxuan Wu, Christopher D Manning, Christopher Potts, and Danqi Chen. Mquake: Assessing knowledge editing in language models via multi-hop questions. arXiv preprint arXiv:2305.14795, 2023.   
Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. Mem1: Learning to synergize memory and reasoning for efficient long-horizon agents. arXiv preprint arXiv:2506.15841, 2025.

# A Details of Dataset

Here we provide a detailed introduction to the datasets used for evaluating the four core competencies, including the dataset curation, corresponding metrics, average context length, and a brief description. Details are shown in Table 1.

# A.1 Accurate Retrieval (AR)

# A.1.1 Definition of AR

The task of accurately retrieving information has been extensively explored in prior work. In the domain of long-context modeling, the Needle-in-a-Haystack (NIAH) task is widely used to evaluate a model’s ability to locate the specific value based on a given key within a lengthy input. In the RAG setting, this corresponds to document-based QA, where the model must identify and extract relevant snippets from one or more documents to answer a query. These snippets might reside in a single location or be distributed across multiple documents. In this paper, we focus on agentic settings, where the “long context” or “multiple documents” become long-form conversations. We define Accurate Retrieval (AR) as the ability of an agent to identify and retrieve important information that may be dispersed throughout a long dialogue history.

# A.1.2 Details on AR datasets

We use four datasets to evaluate the accurate retrieval capability of memory agents.

(1) Document Question Answering We improved two QA datasets from (Hsieh et al., 2024). These datasets provide multiple synthetic contexts of varying lengths, ranging from 3K to over 200K tokens. We select 100 questions from the datasets with shorter context length. For each of these 100 questions, we collect the context and remove duplicate short documents, and then shuffle and concatenate them to create new long documents of 197K or 421K tokens, making sure the new context containing the gold passages. Since most answers are short informational entities, such as years, names, or yes/no responses, we use substring exact match (SubEM) to calculate the accuracy of QA. SubEM measures whether the predicted answer exactly matches the gold answer as a sub-string, which is a common standard in question answering systems.   
(2) LongMemEval This is a dialogue-based QA dataset. For LME(S*), we use multiple historical conversation data segments, arrange them in chronological order, and finally concatenate them to create five long conversation histories, each with a length of approximately 355K tokens. Since some of the questions have open-ended answers, we adopt the approach used in previous work and employ the GPT-4o model to assess whether the agent’s responses meet the requirements. If a response is deemed satisfactory, it is marked as True. Finally, we calculate the proportion of satisfactory responses as the evaluation metric. Wu et al. (2025) reported in Table 6 that a prompt-engineered GPT-4o judge achieves $9 8 . 0 \%$ accuracy and demonstrates very high stability.   
(3) EventQA Using five books from $\infty$ -Bench (each contains more than 390K tokens, counted using the gpt-4o-mini tokenizer), we identify the ten most frequently mentioned characters via SpaCy NER. We extract 101 events experienced by key characters using gpt-4o. For each event, we construct a 6-way multiple-choice question by pairing the true event with five distractors generated via gpt-4o. The agent receives up-to five previous events and must identify the correct continuation. We report the mean accuracy over 100 such questions per book, and ultimately present the average accuracy across all five books.

# A.2 Test-time Learning (TTL)

# A.2.1 Definition of TTL

An essential capability for real-world agents is the ability to acquire new skills dynamically through interaction with users. This mirrors the concept of In-Context Learning (ICL) in LLMs, where the model learns from a prompt containing a small number of examples, often framed as few-shot classification tasks. Ideally, performance improves with additional examples in the prompt. In the conversational agent setting, prompts are replaced by dialogue histories. We define Test-Time Learning (TTL) as the agent’s ability to learn to perform new tasks directly from the conversation. This property is crucial for enabling self-evolving agents that can continuously adapt and improve in real-world deployments.

# A.2.2 Details on TTL datasets

We evaluate TTL via two task categories:

(1) Multi-Class Classification (MCC) We adopt five classification datasets used in prior TTL work. For dataset curation, we use thousands of sentence samples from different categories, with each type of sample assigned a number as its label. Following the format "{sentence} \n Label: {label} \n", we concatenate all the sentences into a long context and shuffle them to prevent samples of the same type from being too concentrated. In this task, the agent needs to refer to a long context and correctly classify the input content. Therefore, we use average accuracy as the evaluation metric.   
(2) Recommendation (Recom.) We concatenate multiple short dialogues about movie recommendations from the original dataset, remove duplicate dialogues, and create a long context containing over a thousand recommendation instances. In this task, the agent is required to recommend 20 movies based on the content of the dialogue. We evaluate the recommendations by calculating Recall@5, which measures the overlap between the top 5 recommended movies and the ground truth.

# A.3 Long-Range Understanding (LRU)

# A.3.1 Definition of LRU

Long-range understanding refers to the agent’s ability to form abstract, high-level comprehension over extended conversations. For example, when a user narrates a long story, the agent should retain the content and derive a holistic understanding rather than just recall isolated facts. We define Long-Range Understanding (LRU) as the ability to reason about long-form inputs and answer high-level questions that require an understanding of the overall content, rather than detailed recall. An example question might be: “Summarize the main experiences of Harry Potter.”

# A.3.2 Details on LRU datasets

We evaluate LRU via the Summarization task En.Sum from $\infty$ -Bench (Zhang et al., 2024). We follow the settings from (Yen et al., 2024) and use the GPT-4o model in evaluating the summarized text. In this process, we assess the fluency of the input text (scored as 0 or 1) and use the dot product of this score with the F1 score as the final evaluation metric.

# A.4 Selective Forgetting (SF)

# A.4.1 Definition of SF

In long-term interactions, agents often face evolving or conflicting information—whether about the external world (e.g., changes in political leadership) or user-specific facts (e.g., a new occupation). This challenge is closely related to model editing (Meng et al., 2023; Fang et al., 2024) and knowledge unlearning (Wang et al., 2024e), which focus on modifying or removing factual knowledge from language models. We define Selective Forgetting (SF) as

the agent’s ability to detect and resolve contradictions between out of date knowledge and newly acquired information, ensuring the agent remains aligned with current realities and user states. SF is distinct from Abstractive Retrieval (AR) in two key ways. (1) Certain questions requiring SF cannot be answered solely through AR. As illustrated in Figure 1, an agent that retrieves all facts related to pears may fail to identify the updated information in the second message. (2) In AR, earlier messages remain relevant and should be retained, even when multiple pieces of evidence are required. In contrast, SF involves identifying outdated or incorrect information and discarding it. That is, AR requires preservation of all related content, whereas SF requires overwriting prior facts to reflect the most up-to-date truth.

# A.4.2 Details on SF datasets

We use counterfactual edit pairs from the MQUAKE (Zhong et al., 2023) dataset. Each sentence containing information was assigned a number. For each edit pair, the sentence representing outdated information (the distractor) is given a smaller number, while the sentence representing more recent information (the one containing the answer) is given a larger number. We then concatenate these sentences into a long context in order according to their assigned numbers. We evaluate the SF via two datasets: Single-Hop FactConsolidation and Multi-Hop FactConsolidation. In these tasks, the agent’s responses are mostly informational entities. Therefore, we also use SubEM (Substring Exact Match) as the evaluation metric to calculate the accuracy of QA.

# A.5 Justification for competencies based on cognitive science

Accurate retrieval is central to human memory research, as evidenced by classical forgetting curves and recall tests that foreground fidelity of recall (Ebbinghaus, 2013). However, a sole focus on accuracy obscures another fundamental axis: the timescale of learning and consolidation. Ebbinghaus observed that an initial, fleeting grasp rarely endures without reinforcement (Ebbinghaus, 2013), and James (1890) distinguished primary (immediate) from secondary (enduring) memory. These classic distinctions ground our notions of testtime learning (incorporation of new information via memory) and long-range understanding (durable, integrated knowledge). Consistent with this, the Complementary Learning Systems (CLS) framework delineates a hippocampal system for rapid episodic learning and a neocortical system for gradual, structured knowledge accumulation, underscoring the need to assess both quick memorization and long-horizon retention (McClelland et al., 1995).

Beyond the acquisition–consolidation axis, another equally fundamental challenge is selective forgetting. Overlapping or contradictory traces can impede retrieval, and interference has long been recognized as a primary driver of forgetting in cognitive psychology (Anderson $\&$ Neely, 1996). Neurocognitive evidence further shows that the brain engages targeted control mechanisms to resolve such interference at retrieval time (Wimber et al., 2015). We therefore include selective forgetting—the ability to handle interference and contradictions—as a core dimension.

In sum, our four categories—accurate retrieval, test-time learning, long-range understanding, and selective forgetting—align with key dimensions of memory identified in cognitive science and AI memory systems, covering the essential capabilities that any robust memory mechanism must support in practice.

# B Detailed Memory Agents Description

We give detailed description of the memory agents used in experiments in this section.

# B.1 Long-Context Agents

We evaluate five modern long-context LLMs: GPT-4o (OpenAI, 2025b) serves as the highperformance, low-latency model with better cost efficiency than prior generations. While GPT-4o-mini is a lightweight, budget-friendly alternative that enables large-scale evalua-

Table 5: Datasets categorized by the specific aspects of evaluation. Here 1K is 1024.   

<table><tr><td>Capability</td><td>Tasks</td><td># of Sequences : QAs</td><td>Avg Len</td></tr><tr><td rowspan="4">Accurate Retrieval</td><td>SH-Doc QA</td><td>1 : 100</td><td>197K</td></tr><tr><td>MH-Doc QA</td><td>1 : 100</td><td>421K</td></tr><tr><td>LongMemEval (S*)</td><td>5 : 300</td><td>355K</td></tr><tr><td>EventQA</td><td>5 : 500</td><td>534K</td></tr><tr><td rowspan="6">Test-Time Learning</td><td>BANKING-77</td><td>1 : 100</td><td></td></tr><tr><td>CLINC-150</td><td>1 : 100</td><td></td></tr><tr><td>NLU</td><td>1 : 100</td><td>103K</td></tr><tr><td>TREC (Coarse)</td><td>1 : 100</td><td></td></tr><tr><td>TREC (Fine)</td><td>1 : 100</td><td></td></tr><tr><td>Movie-Rec Redial</td><td>1 : 200</td><td>1.44M</td></tr><tr><td rowspan="2">Long-Range Understanding</td><td>∞Bench-Sum</td><td>100 : 100</td><td>172K</td></tr><tr><td>Detective QA</td><td>10 : 71</td><td>124K</td></tr><tr><td rowspan="2">Selective Forgetting</td><td>FactConsolidation-SH</td><td>1 : 100</td><td rowspan="2">262K</td></tr><tr><td>FactConsolidation-MH</td><td>1 : 100</td></tr></table>

tions by delivering faster responses and lower per-token costs. Notably, the GPT-4.1 (OpenAI, 2025a) family strengthens instruction following and maintains strong performance at very large context windows (reported up to 1M tokens). Considering the higher token cost, we choose the GPT-4.1-mini in evaluation. Gemini-2.0-Flash (DeepMind, 2025) targets high throughput and the use of built-in tools, offering a 1M token context window for efficient long-context processing. Claude-3.7-Sonnet (Anthropic, 2025) is a hybrid-reasoning model with optional visible “extended thinking,” strong math/coding skills, and developercontrolled thinking budgets.

# B.2 RAG Agents

We consider three RAG variants: Simple RAG Agents, Embedding-based RAG Agents, and Structure-Augmented RAG Agents.

(1) Simple RAG Agents We implement a BM25 (Robertson & Walker, 1994) retriever as a strong lexical baseline: it scores documents by term frequency with saturation and inverse document frequency, with length normalization controlled by parameters $k _ { 1 }$ and $b$ . BM25 remains competitive for exact-match queries and complements dense retrievers with robust precision on keyworded questions.   
(2) Embedding-based RAG Agents Contriever (Izacard et al., 2021) is an unsupervised dense retriever trained via contrastive learning on large text corpora, enabling semantic matching without labeled pairs. Text-Embedding-3-Small/Large (OpenAI, 2024) are OpenAI’s general-purpose embedding models offering a cost–quality trade-off (e.g., 1,536 vs. 3,072 dimensions) for search and retrieval. Qwen3-Embedding-4B (Zhang et al., 2025) is a 4B-parameter embedding/reranking model family geared toward multilingual retrieval and long-text understanding.   
(3) Structure-Augmented RAG Agents RAPTOR (Sarthi et al., 2024) is method building a hierarchical tree of recursive summaries (bottom-up clustering and abstraction) and retrieves across levels for long-document QA. GraphRAG (Edge et al., 2024) extracts a knowledge graph and community hierarchy, then performs graph-aware retrieval and summarization. MemoRAG (Qian et al., 2025) introduces a dual-system pipeline with a light “global-memory” model to guide retrieval and a stronger model for final answers. HippoRAG-v2 (Gutiérrez et al., 2025) extends hippocampal-inspired retrieval to improve factual, sense-making, and associative memory tasks over standard RAG. We also evaluate three open-sourced memory agents: Mem0, Cognee and Zep. Mem0 (Chhikara et al., 2025) provides a persistent agent memory layer for storing/retrieving user-specific knowledge to

enhance personalization. Cognee (Markovic et al., 2025) is an open-source memory engine that builds structured (graph-native) memories via ECL pipelines to power graph-aware RAG. Zep (Rasmussen et al., 2025) is a temporal knowledge-graph memory platform for agents, designed to assemble and retrieve long-term conversational and business context.

# B.3 Agentic Memory Agents

For Agentic Memory Agents, We evaluate the Self-RAG (Asai et al., 2023), MemGPT (Packer et al., 2023), and MIRIX (Wang & Chen, 2025) on our benchmark. Self-RAG use LLMs as the agent to decide when/what to retrieve and to critique its own outputs. MemGPT operates the hierarchical memory management, paging relevant snippets between short-term and long-term stores and using event-driven interrupts to maintain coherence and evolvability over extended interactions. MIRIX adopts a multi-agent memory architecture with six specialized memory types (Core, Episodic, Semantic, Procedural, Resource, Knowledge Vault) and a coordinator that orchestrates updates/retrieval across agents.

For comparability, we standardize prompts, tool access, and settings (like retrieval TopK and input chunk size) across above all systems.

# C Prompts

We introduce the examples of prompt used memory construction and task execution in this section.

# C.1 Instructions for Memory Construction

When processing long-context inputs, we split the content into chunks of a specified size and feed these chunks into the agent as memory. The agent can then extract relevant information from its memory based on the query to assist with query execution. This chunking approach helps organize and manage large amounts of contextual information, making retrieval and reasoning more efficient. In Figure 4, we provide several example instructions that require the agent to memorize the corresponding context.

# C.2 Instructions for Task Execution

In Figure 5, we provide the examples of instructions used on different of datasets when handling the input queries. For some existing datasets, we refer the prompt settings from previous work such as (Hsieh et al., 2024; Wu et al., 2025). For the dataset ∞Bench-Sum, we also insert two answer examples as $\langle { \mathrm { d e m o } } \rangle$ in the prompt to help the agent better understand the questions and standardize its outputs.

# D Detailed Experimental Results

In this section, we provide detailed versions of the results presented in the main text.

# D.1 Detailed Results on TTL

We give detailed results on Multi-Class Classification (MCC) task in Table 6. For all three types of tasks, RAG-based agents generally underperform compared to their respective GPT-4o-mini backbones. This observation highlights certain limitations inherent to the RAG approach. For instance, in TTL tasks, RAG-based methods often struggle to more accurately retrieve context from memory that is closely associated with the input.

# Prompts Used for Memory Construction on Various Tasks

# Document Question Answering (SH-Doc QA or MH-Doc QA):

Here is a dialogue between User and Assistant on ⟨time⟩:

⟨User⟩: The following context consists the documents I have read: ⟨chunk⟩. Please memorize it and I will ask some questions based on it in future.

⟨Assistant⟩: Sure! I have learned the documents and I will answer the question you ask.

# $L M E ( S ^ { * } )$ :

Here is a dialogue between User and Assistant on ⟨time⟩:

⟨User⟩: The following context is the dialogue history that I have talked with a ChatBoT: ⟨chunk⟩.

Please memorize it and I will ask some questions based on it in future.

⟨Assistant⟩: Sure! I have memorized the dialogue history and I will answer the question you ask.

# EventQA:

Here is a dialogue between User and Assistant on ⟨time⟩:

⟨User⟩: The following context is the sections in a novel that I have read: ⟨chunk⟩. Please memorize it and I will ask some questions based on it in future.

⟨Assistant⟩: Sure! I have memorized the novel sections and I will answer the question you ask.

# Multi-Class Classification (MCC):

Here is a dialogue between User and Assistant on ⟨time⟩:

⟨User⟩: The following context is the examples I have learned: ⟨chunk⟩. Please memorize it and I will ask some questions based on it in future.

⟨Assistant⟩: Sure! I have memorized the examples and I will answer the question you ask.

# Recommendation (Recom):

Here is a dialogue between User and Assistant on ⟨time⟩:

⟨User⟩: The following context is the dialogue history that I have talked with a recommender system: ⟨chunk⟩. Please memorize it and I will ask some questions based on it in future.

⟨Assistant⟩: Sure! I have memorized the dialogues and I will answer the question you ask.

# Novel Summarization:

Here is a dialogue between User and Assistant on ⟨time⟩:

⟨User⟩: The following context is the sections in a novel that I have read: ⟨chunk⟩. Please memorize it and I will need you to summarize based on it in future.

⟨Assistant⟩: Sure! I have memorized the novel sections and I will summarize them when you ask.

# Detective QA (Det QA):

Here is a dialogue between User and Assistant on ⟨time⟩:

⟨User⟩: The following context is the sections in a novel that I have read: ⟨chunk⟩. Please memorize it and I will need you to answer the questions based on the entire novel.

⟨Assistant⟩: Sure! I have memorized the novel sections and I will answer the questions you ask.

# Selective Forgetting (SF):

Here is a dialogue between User and Assistant on ⟨time⟩:

⟨User⟩: The following context is the facts that I have learned: ⟨chunk⟩. Please memorize it and I will need you to answer the questions based on the order of facts.

⟨Assistant⟩: Sure! I have memorized the facts and I will answer the questions you ask.

Figure 4: The prompts we use for the agents to create the memory.

# D.2 Results on Input Chunk Size Ablation Study

In Table 7, we report the detailed results on evaluating the RAG-based Agents on different chunk sizes and datasets. We selected chunk sizes from the two sets {512, 4096} and {512, 1024, 2048, 4096}.

# D.3 Results on Retrieval TopK Ablation Study

In Table 8, we report the detailed results of the selected RAG-based Agents evaluated on five datasets. We choose different TopK ranging from {2, 5, 10}.

# Examples of Prompts Used for Task Execution on Various Dataset

# Document Question Answering (SH-Doc QA or MH-Doc QA)

The context is given as below: ⟨memory⟩. \n Answer the question based on the context. Only give me the answer and do not output any other words. \n Now Answer the Question: ⟨question⟩ \n Answer:

# LME(S*)

Here is the context of dialogue history: ⟨memory⟩ \n. Based on the relevant chat history, answer the question concisely, using a single phrase if possible.\n Current Date: ⟨question_date⟩, \n Now Answer the Question: ⟨question⟩ \n Answer:

# EventQA

The context is given as below: ⟨memory⟩. \n Based on the context above, complete the task below: \n ⟨question⟩ \n Your task is to choose from the above events which event happens next based on the book excerpt. In your response to me, only include the answer without anything else. \n The event that happens next is:

# Multi-Class Classification (MCC)

The context is given as below: ⟨memory⟩. \n Use the provided mapping examples from the context to numerical label to assign a numerical label to the context. Only output "label: $\{ \{ \mathrm { l a b e l } \} \} ^ { \flat }$ and nothing else. \n Question: ⟨question⟩ \n label:

# Recommendation (Recom)

Here is the context of dialogue history: ⟨memory⟩. \n Pretend you are a movie recommender system. You need to recommend movies based on the above dialogue history. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n For Example:\n [Conversation] \n The recommendations are: \n 1.movie1 \n 2.movie2 \n ...\n Here is the conversation: ⟨question⟩ \n The recommendations are:

# Novel Summarization

The book is given as below: ⟨memory⟩ \n You are given a book above and you are tasked to summarize it. Write a summary of about 1000 to 1200 words. Only write about the plot and characters of the story. Do not discuss the themes or background of the book. Do not provide any analysis or commentary. \n ⟨demo⟩ \n Now summarize the book.

# Detective QA (Det QA)

The context is given as below: ⟨memory⟩. \n Based on the context above, complete the task below: You are required to answer the question based on the strict output format.\n ⟨question⟩ \n

# Fact Consolidation

Here is a knowledge pool with lots of new facts: ⟨memory⟩. \n Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n For example: \n [Knowledge Pool] \n Question: Based on the provided Knowledge Pool, what is the name of the current president of Country R? \n Answer: Person D. \n Now Answer the Question: Based on the provided Knowledge Pool, ⟨question⟩ \n Answer:

Figure 5: The example prompts we use for the Memory Agents in Table 2. Here ⟨memory⟩ refers to the accumulated text from the sequential inputs.

# D.4 Results on Different Context Length Ablation Study

In Table 9, we report the performances of different agents when scaling the input length. We measure the average context length via the tokenizer of GPT-4o-mini and here 1K is 1024. For Long-Context Agents, tasks in the AR series generally achieve satisfactory performance at relatively small context lengths (e.g., around 50K tokens). However, as the context length increases, the performance of these agents declines accordingly. In contrast, for the RAG-based agents Mem0 and Cognee, their performance is significantly lower than that of their backbone, GPT-4o-mini, even when the context length is relatively small.

Table 6: Overall performance comparison on the datasets for TTL. All RAG agents and commercial memory agents use GPT-4o-mini as the backbone.   

<table><tr><td>Agent Type</td><td>BANKING</td><td>CLINC</td><td>NU</td><td>TREC C</td><td>TREC F</td></tr><tr><td colspan="6">Long-Context Agents</td></tr><tr><td>GPT-4o</td><td>96.0</td><td>96.0</td><td>90.0</td><td>87.0</td><td>69.0</td></tr><tr><td>GPT-4o-mini</td><td>93.0</td><td>93.0</td><td>87.0</td><td>73.0</td><td>66.0</td></tr><tr><td>GPT-4.1-mini</td><td>93.0</td><td>82.0</td><td>85.0</td><td>68.0</td><td>50.0</td></tr><tr><td>Gemini-2.0-Flash</td><td>91.0</td><td>90.0</td><td>84.0</td><td>88.0</td><td>67.0</td></tr><tr><td>Claude-3.7-Sonnet</td><td>97.0</td><td>98.0</td><td>86.0</td><td>87.0</td><td>79.0</td></tr><tr><td>GPT-4o-mini</td><td>93.0</td><td>93.0</td><td>87.0</td><td>73.0</td><td>66.0</td></tr><tr><td colspan="6">Simple RAG Agents</td></tr><tr><td>BM25</td><td>89.0</td><td>89.0</td><td>84.0</td><td>62.0</td><td>53.0</td></tr><tr><td colspan="6">Embedding RAG Agents</td></tr><tr><td>Contriever</td><td>89.0</td><td>88.0</td><td>80.0</td><td>55.0</td><td>41.0</td></tr><tr><td>Text-Embed-3-Small</td><td>88.0</td><td>89.0</td><td>83.0</td><td>54.0</td><td>36.0</td></tr><tr><td>Text-Embed-3-Large</td><td>90.0</td><td>91.0</td><td>80.0</td><td>55.0</td><td>46.0</td></tr><tr><td>Qwen3-Embedding-4B</td><td>90.0</td><td>88.0</td><td>86.0</td><td>67.0</td><td>59.0</td></tr><tr><td colspan="6">Structure-Augmented RAG Agents</td></tr><tr><td>RAPTOR</td><td>78.0</td><td>75.0</td><td>73.0</td><td>48.0</td><td>23.0</td></tr><tr><td>GraphRAG</td><td>64.0</td><td>54.0</td><td>49.0</td><td>24.0</td><td>6.0</td></tr><tr><td>MemoRAG</td><td>90.0</td><td>87.0</td><td>86.0</td><td>66.0</td><td>56.0</td></tr><tr><td>HippoRAG-v2</td><td>81.0</td><td>86.0</td><td>73.0</td><td>38.0</td><td>29.0</td></tr><tr><td>Mem0</td><td>35.0</td><td>37.0</td><td>35.0</td><td>29.0</td><td>26.0</td></tr><tr><td>Cognee</td><td>34.0</td><td>42.0</td><td>42.0</td><td>41.0</td><td>18.0</td></tr><tr><td>Zep</td><td>83.0</td><td>74.0</td><td>70.0</td><td>50.0</td><td>37.0</td></tr><tr><td colspan="6">Agentic Memory Agents</td></tr><tr><td>Self-RAG</td><td>19.0</td><td>13.0</td><td>6.0</td><td>15.0</td><td>5.0</td></tr><tr><td>MemGPT</td><td>89.0</td><td>83.0</td><td>79.0</td><td>56.0</td><td>31.0</td></tr><tr><td>MIRIX</td><td>42.0</td><td>53.0</td><td>49.0</td><td>36.0</td><td>12.0</td></tr><tr><td>MIRIX(4.1-mini)</td><td>65.0</td><td>83.0</td><td>69.0</td><td>73.0</td><td>25.0</td></tr></table>

Table 7: Performance comparison on different datasets and chunk sizes. Here we choose chunk sizes from {512, 1024, 2048, 4096} and we use k=10 for RAG-based methods.   
Table 8: Performance comparison on different retrieve number.   

<table><tr><td></td><td colspan="4">SH-Doc QA</td><td colspan="4">MH-Doc QA</td><td colspan="4">∞Bench-Sum</td></tr><tr><td></td><td>512</td><td>1024</td><td>2048</td><td>4096</td><td>512</td><td>1024</td><td>2048</td><td>4096</td><td>512</td><td>1024</td><td>2048</td><td>4096</td></tr><tr><td>BM25</td><td>66.0</td><td>67.0</td><td>68.0</td><td>66.0</td><td>56.0</td><td>54.0</td><td>52.0</td><td>56.0</td><td>11.5</td><td>13.2</td><td>15.2</td><td>19.0</td></tr><tr><td>Qwen3-Embedding-4B</td><td>57.0</td><td>53.0</td><td>52.0</td><td>50.0</td><td>47.0</td><td>44.0</td><td>40.0</td><td>38.0</td><td>7.9</td><td>9.4</td><td>13.2</td><td>14.8</td></tr><tr><td>HippoRAG-v2</td><td>76.0</td><td>70.0</td><td>57.0</td><td>49.0</td><td>66.0</td><td>63.0</td><td>51.0</td><td>38.0</td><td>4.6</td><td>6.0</td><td>10.5</td><td>14.6</td></tr><tr><td>MemGPT</td><td>41.0</td><td>32.0</td><td>24.0</td><td>27.0</td><td>38.0</td><td>33.0</td><td>37.0</td><td>35.0</td><td>1.2</td><td>1.8</td><td>4.2</td><td>2.5</td></tr></table>

Table 9: Performance comparison on different context length.   

<table><tr><td rowspan="2"></td><td colspan="3">SH-Doc QA</td><td colspan="3">MH-Doc QA</td><td colspan="3">EventQA</td><td colspan="3">TTL (MCC)</td></tr><tr><td>R=2</td><td>R=5</td><td>R=10</td><td>R=2</td><td>R=5</td><td>R=10</td><td>R=2</td><td>R=5</td><td>R=10</td><td>R=2</td><td>R=5</td><td>R=10</td></tr><tr><td>BM25</td><td>50.0</td><td>60.0</td><td>66.0</td><td>49.0</td><td>54.0</td><td>56.0</td><td>66.6</td><td>71.2</td><td>74.6</td><td>67.8</td><td>74.6</td><td>75.4</td></tr><tr><td>Contriever</td><td>17.0</td><td>20.0</td><td>22.0</td><td>22.0</td><td>27.0</td><td>31.0</td><td>54.4</td><td>66.8</td><td>56.0</td><td>63.0</td><td>70.0</td><td>70.6</td></tr><tr><td>Text-Embed-3-Large</td><td>36.0</td><td>47.0</td><td>54.0</td><td>37.0</td><td>41.0</td><td>44.0</td><td>51.8</td><td>62.4</td><td>70.0</td><td>59.4</td><td>69.4</td><td>72.4</td></tr><tr><td>RAPTOR</td><td>22.0</td><td>27.0</td><td>29.0</td><td>30.0</td><td>36.0</td><td>38.0</td><td>45.8</td><td>41.8</td><td>40.4</td><td>56.3</td><td>57.4</td><td>59.4</td></tr><tr><td>HippoRAG-v2</td><td>60.0</td><td>69.0</td><td>76.0</td><td>53.0</td><td>60.0</td><td>66.0</td><td>58.8</td><td>67.6</td><td>67.4</td><td>58.8</td><td>61.4</td><td>61.4</td></tr><tr><td>Self-RAG</td><td>27.0</td><td>33.0</td><td>35.0</td><td>34.0</td><td>39.0</td><td>42.0</td><td>28.2</td><td>30.6</td><td>31.8</td><td>9.0</td><td>11.6</td><td>11.6</td></tr></table>

<table><tr><td rowspan="2"></td><td colspan="3">SH-Doc QA</td><td colspan="3">MH-Doc QA</td><td colspan="3">EventQA</td><td colspan="3">FactCon-SH</td><td colspan="3">FactCon-MH</td></tr><tr><td>51K</td><td>104K</td><td>197K</td><td>51K</td><td>104K</td><td>421K</td><td>51K</td><td>108K</td><td>534K</td><td>32K</td><td>64K</td><td>262K</td><td>32K</td><td>64K</td><td>262K</td></tr><tr><td>GPT-4o</td><td>91.0</td><td>84.0</td><td>72.0</td><td>72.0</td><td>68.0</td><td>51.0</td><td>96.8</td><td>94.0</td><td>77.2</td><td>88.0</td><td>85.0</td><td>60.0</td><td>10.0</td><td>13.0</td><td>5.0</td></tr><tr><td>GPT-4o-mini</td><td>84.0</td><td>83.0</td><td>64.0</td><td>58.0</td><td>54.0</td><td>43.0</td><td>90.2</td><td>85.8</td><td>59.0</td><td>63.0</td><td>58.0</td><td>45.0</td><td>10.0</td><td>5.0</td><td>5.0</td></tr><tr><td>GPT-4.1-mini</td><td>93.0</td><td>86.0</td><td>83.0</td><td>72.0</td><td>75.0</td><td>66.0</td><td>97.0</td><td>93.8</td><td>82.6</td><td>82.0</td><td>72.0</td><td>36.0</td><td>7.0</td><td>9.0</td><td>5.0</td></tr><tr><td>Gemini-2.0-Flash</td><td>92.0</td><td>87.0</td><td>87.0</td><td>69.0</td><td>61.0</td><td>59.0</td><td>93.4</td><td>88.6</td><td>67.2</td><td>49.0</td><td>62.0</td><td>30.0</td><td>7.0</td><td>9.0</td><td>3.0</td></tr><tr><td>Claude-3.7-Sonnet</td><td>90.0</td><td>82.0</td><td>77.0</td><td>67.0</td><td>59.0</td><td>53.0</td><td>96.6</td><td>95.2</td><td>74.6</td><td>46.0</td><td>45.0</td><td>43.0</td><td>2.0</td><td>2.0</td><td>2.0</td></tr><tr><td>Mem0</td><td>31.0</td><td>25.0</td><td>25.0</td><td>36.0</td><td>29.0</td><td>32.0</td><td>60.8</td><td>47.0</td><td>37.5</td><td>22.0</td><td>8.0</td><td>18.0</td><td>3.0</td><td>2.0</td><td>2.0</td></tr><tr><td>Cognee</td><td>38.0</td><td>42.0</td><td>31.0</td><td>36.0</td><td>38.0</td><td>26.0</td><td>53.4</td><td>39.0</td><td>26.8</td><td>39.0</td><td>31.0</td><td>28.0</td><td>4.0</td><td>5.0</td><td>3.0</td></tr></table>

# D.5 Results on Computational Latency Analysis

To illustrate the latency of various memory agents in terms of (1) Memory Construction (M.C.); (2) Query Execution (Q.E.), we report the latency of various memory agents on MH-QA and LME (S*). This part of experiments is done on a server with Four NVDIA

Table 10: Computational latency (in seconds) comparison on Long-Context Agents.   

<table><tr><td></td><td>MH-QA</td><td>LME (S*)</td></tr><tr><td>GPT-4o</td><td>17.0</td><td>20.1</td></tr><tr><td>GPT-4o-mini</td><td>4.9</td><td>5.4</td></tr><tr><td>GPT-4.1-mini</td><td>9.0</td><td>7.4</td></tr><tr><td>Gemini-2.0-Flash</td><td>12.4</td><td>10.1</td></tr><tr><td>Claude-3.7-Sonnet</td><td>23.3</td><td>22.7</td></tr></table>

Table 11: Computational latency (in seconds) comparison on RAG based agents. M.C. means Memory Construction and Q.E. means Query Execution. *Indicates that the time is obtained through estimation.   

<table><tr><td rowspan="3"></td><td colspan="4">MH-QA</td><td colspan="4">LME (S*)</td></tr><tr><td colspan="2">512</td><td colspan="2">4096</td><td colspan="2">512</td><td colspan="2">4096</td></tr><tr><td>M.C.</td><td>Q.E.</td><td>M.C.</td><td>Q.E.</td><td>M.C.</td><td>Q.E.</td><td>M.C.</td><td>Q.E.</td></tr><tr><td>BM25</td><td>0.12</td><td>0.47</td><td>0.11</td><td>1.7</td><td>0.09</td><td>1.1</td><td>0.08</td><td>1.9</td></tr><tr><td>Contriever</td><td>7.4</td><td>0.59</td><td>1.7</td><td>2.0</td><td>6.9</td><td>0.92</td><td>1.6</td><td>1.9</td></tr><tr><td>Text-Embed-3-Large</td><td>6.1</td><td>0.46</td><td>5.0</td><td>1.7</td><td>6.5</td><td>0.62</td><td>5.8</td><td>1.8</td></tr><tr><td>Qwen3-Embedding-4B</td><td>367</td><td>0.49</td><td>470</td><td>1.9</td><td>293</td><td>0.71</td><td>372</td><td>1.8</td></tr><tr><td>RAPTOR</td><td>193</td><td>0.41</td><td>161</td><td>0.67</td><td>108</td><td>0.60</td><td>104</td><td>0.53</td></tr><tr><td>GraphRAG</td><td>97.8</td><td>12.8</td><td>91.9</td><td>10.9</td><td>149</td><td>7.0</td><td>88.8</td><td>7.8</td></tr><tr><td>HippoRAG-v2</td><td>1089</td><td>0.71</td><td>380</td><td>1.71</td><td>544</td><td>1.5</td><td>188</td><td>3.5</td></tr><tr><td>Mem0</td><td>10804</td><td>0.79</td><td>1334</td><td>0.65</td><td>18483</td><td>1.6</td><td>2946</td><td>1.7</td></tr><tr><td>Cognee</td><td>11890</td><td>58.7</td><td>1185</td><td>4.8</td><td>4728</td><td>7.7</td><td>738</td><td>4.1</td></tr><tr><td>Self-RAG</td><td>11.4</td><td>3.1</td><td>8.1</td><td>2.4</td><td>5.3</td><td>0.82</td><td>5.2</td><td>1.0</td></tr><tr><td>MemGPT</td><td>433</td><td>9.4</td><td>101</td><td>10.5</td><td>392</td><td>11.7</td><td>85.5</td><td>12.3</td></tr><tr><td>MIRIX</td><td>29000*</td><td>-</td><td>20171</td><td>14.1</td><td>12600*</td><td>-</td><td>3258</td><td>8.7</td></tr><tr><td>MIRIX (GPT-4.1-mini)</td><td>28800*</td><td>-</td><td>21361</td><td>16.9</td><td>9000*</td><td>-</td><td>2512</td><td>9.2</td></tr></table>

Table 12: Peak GPU memory usage of embedding models (MB). We measure the memory usage on MH-QA dataset with different chunk size.   

<table><tr><td>Agents / Chunk Size</td><td>512</td><td>4096</td></tr><tr><td>HippoRAG-v2 (NV-Embed-v2)</td><td>27674</td><td>60205</td></tr><tr><td>Qwen3-Embedding-4B</td><td>16671</td><td>41262</td></tr></table>

L40 GPU and AMD EPYC 7713 64-Core CPU. We use the NV-Embed-v2 (7B) as the embedding model in HippoRAG-v2. We show the results in Table 10 and 11. From the table, we find that using a smaller chunk size requires significantly more time for memory construction, especially for methods such as HippoRAG-v2, Mem0, Cognee, and MemGPT. Meanwhile, methods such as Mem0, Cognee and MIRIX need extremely high resources when constructing the memory.

# D.6 GPU Memory Usage Comparison

In main experiments, we mostly use the LLM API as the backbone models which do not need local GPUs. In our experiments, the HippoRAG-v2 (NV-Embed-v2) and Qwen3- Embedding-4B require running the embedding model on GPU. We report their peak GPU memory usage in Table 12, where all experiments are conducted on a single A100 80GB GPU.

# E Experimental Settings

In this section, we present the experimental settings in evaluation.

Table 13: Maximum output token limits for various tasks   

<table><tr><td>Task</td><td>Max Output Tokens</td></tr><tr><td>SH-QA / MH-QA</td><td>50</td></tr><tr><td>LME(S*)</td><td>100</td></tr><tr><td>EventQA</td><td>40</td></tr><tr><td>MCC</td><td>20</td></tr><tr><td>Movie Recommendation</td><td>300</td></tr><tr><td>∞ Bench-Sum</td><td>1,200</td></tr><tr><td>Detective QA</td><td>500</td></tr><tr><td>FactConsolidation</td><td>10</td></tr></table>

Table 14: The choice of chunk size for different datasets.   

<table><tr><td>Chunk Size</td><td>512</td><td>4096</td></tr><tr><td rowspan="3">Dataset</td><td>SH-QA, MH-QA</td><td>∞Bench-Sum</td></tr><tr><td>FactCon-SH, FactCon-MH</td><td>MCC, Recom</td></tr><tr><td>LME(S*)</td><td>EventQA, Detective QA</td></tr></table>

# E.1 Max Output Tokens

We provide the token number limitation for each task in Table 13.

# E.2 Settings of the RAG Agents

For the embedding model selection in Structure-Augmented RAG Agents and Agentic Memory Agents, most approaches utilize OpenAI’s embedding models, such as Text-Embed-3- Small. While for the HippoRAG-v2 method, we follow the same experimental setting as in Gutiérrez et al. (2025), employing the NV-Embed-v2 model.

We implement three open-sourced memory agents in our main experiments. (1) For Mem0, we use memory.add() function to add the message with the content from each context chunk into the agent’s memory repository during memory consolidation. During query execution, the relevant memory elements are retrieved through memory.search() function. The retrieved memories are then integrated into the query before being processed by the GPT-4o-mini backbone model to complete the requested tasks. (2) For MemGPT, we employ the insert_passage() function during the memory consolidation phase to inject long context chunks into the Archival Memory structure. During query execution, this agent processes requests via the send_message() function which generates appropriate responses based on the archived information. (3) For Cognee, we utilize the cognee.add() and cognee.cognify() functions to construct the memory graph from input chunks wherein the memory consolidation phase. During query execution, the cognee.search() function is used to retrieve contextually relevant information from the memory graph based on the input query.

# E.3 Settings of the Chunk Size

We use smaller chunk size (512) for synthetic context used in AR and SF. For some tasks based on continuous text, such as $\infty$ Bench and EventQA, we used a larger chunk size (4096). For tasks such as MCC and Recom, considering the characteristics of these tasks and the computational cost, we also chose a larger chunk size (4096). For the memory construction methods that are more time-consuming and requiring more API cost, Mem0, Zep, Cognee and MIRIX, we uniformly used a chunk size of 4096 across all datasets. The detailed settings are presented in Table 14.
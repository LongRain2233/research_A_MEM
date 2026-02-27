# WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning

Woongyeong Yeo1* Kangsan Kim1∗ Jaehong Yoon2† Sung Ju Hwang1,3†

KAIST1 Nanyang Technological University2 DeepAuto.ai3

https://worldmm.github.io

# Abstract

Recent advances in video large language models have demonstrated strong capabilities in understanding short clips. However, scaling them to hours- or days-long videos remains highly challenging due to limited context capacity and the loss of critical visual details during abstraction. Existing memory-augmented methods mitigate this by leveraging textual summaries of video segments, yet they heavily rely on text and fail to utilize visual evidence when reasoning over complex scenes. Moreover, retrieving from fixed temporal scales further limits their flexibility in capturing events that span variable durations. To address this, we introduce WorldMM, a novel multimodal memory agent that constructs and retrieves from multiple complementary memories, encompassing both textual and visual representations. WorldMM comprises three types of memory: episodic memory indexes factual events across multiple temporal scales, semantic memory continuously updates high-level conceptual knowledge, and visual memory preserves detailed information about scenes. During inference, an adaptive retrieval agent iteratively selects the most relevant memory source and leverages multiple temporal granularities based on the query, continuing until it determines that sufficient information has been gathered. WorldMM significantly outperforms existing baselines across five long video question-answering benchmarks, achieving an average $8 . 4 \%$ performance gain over previous state-of-the-art methods, showing its effectiveness on long video reasoning.

# 1. Introduction

With the increasing deployment of video large language models (video LLMs) [1, 3, 16, 37] in real-world applications, such as AI glasses and household robots, these models are now required to process and reason over extremely long videos from several hours to even days [5, 25, 33]. Recent works [4, 12, 13] have introduced memory-based

approaches that build external memories from abstracted video representations. Such methods allow the model to focus on essential information by retrieving a small number of relevant memories, thereby reducing the number of input tokens. This is a more efficient and effective strategy compared to processing all frames in the video, requiring high computational cost as illustrated in Fig. 1(a).

Despite their promise, most existing approaches remain highly dependent on textual representations. Typically, each detected event or clip is converted into captions, summaries, or structured text descriptions for downstream retrieval and reasoning [12, 23, 33]. Although Long et al. [13] incorporates visual inputs when building memory, its use of multimodal features is limited to entity recognition and is not fully exploited during inference (Fig. 1(b)). Moreover, existing models [13, 33] typically retrieve a fixed number of clips with predetermined durations, such as retrieving three 30-second clips. These rigid architecture designs in video memory agents face the following two major limitations.

First, they fail to adaptively leverage visual information from videos in conjunction with textual memory during retrieval and generation. Visual details are essential for many real-world tasks requiring attribute recognition, spatial reasoning, or precise scene understanding, while this knowledge cannot be fully represented in text. Meanwhile, as shown in Fig. 1(c), a fixed strategy that always includes both captions and frames during response generation yields suboptimal results since excessive visual context can even distract the model. Therefore, an adaptive mechanism for selecting multimodal memories is essential for retrieving the most informative references for a given query, which remains unexplored in previous works.

Second, retrieving a fixed number of clips limits the model’s ability to handle queries that require varying temporal scopes. For instance, a question like “Where did I leave my glasses?” may require only a few seconds of video, whereas “What happened in the second half of the soccer match?” demands a much longer temporal context. Existing approaches retrieve a predetermined length of segments for simplicity [12, 13, 33], which inherently over-

![](images/e4c50713c2a1a10c7eae62d5d66b49c37053f22e07c3cf983573c70da6de8784.jpg)  
(a) Day-long (1fps)   
When did I last drink beer?

![](images/a7005011abc4fe9f5b5127f00da60536c4670bd3613950d3a206445b93915a09.jpg)  
Captions

> 80k frames

I haven’t drunk beer, just a cup of coffee.

Too many frames

(b) M3-Agent

What was the color of the can I drank?

![](images/363671f5f970c6312bcc7033f2a0450708d14f4238b17337970c91e4a48259a3.jpg)  
Text-based Retrieval   
“I picked up a coke can and drank it ...” 1   
It was red.   
Missing visual data   
(c) EgoRAG   
What will she ride?   
Video-Caption pairs   
“Amy said she would transfer buses ，   
Subway.   
Distracted by frames

![](images/bb799a893e0e63285d65bba752cdff6ae7ea0a1febaa5dab927366c18818f76f.jpg)  
(d) WorldMM (Ours)   
What magic did Jake perform?   
Multimodal Memory   
Call(Episodic, Jake magic)   
Episodic   
Semantic   
Visual   
Call(Visual, pigeon)   
Jake made pigeon to …   
0 Adaptive memory type selection   
Figure 1. Concept Figure. (a) A day-long video sampled at 1 fps has frames that exceed the context limits of video LLMs. (b) M3- Agent [13] relies on textual representation of video, which can underrepresent visual information. (c) EgoRAG [33] retrieves both captions and the corresponding visual frames, but irrelevant frames may distract model. (d) WorldMM (Ours) constructs multiple memories, incorporating both textual and visual representations, and uses adaptive memory retrieval to effectively leverage multimodal information.

looks the diverse temporal scales of real-world events, limiting flexibility. Instead, the retriever should dynamically gather information at multiple temporal scales, combining hour-level summaries with minute-level details as needed.

To fill this gap, we present WorldMM, a novel memorybased agent that constructs separate textual and visual memories and employs an adaptive retrieval agent to select the optimal memory modality and temporal granularity for each query. The textual memory comprises two components: episodic memory, which stores multiple events across different time scales, and semantic memory, which captures high-level, long-term knowledge such as relationships and habits, organized within knowledge graphs. The visual memory divides a long video into short-term segments indexed within a retrieval corpus, enabling the model to access visual information when required. The retrieval agent iteratively selects the most relevant memory across modalities and timescales, ensuring that the agent retrieves only the information necessary for each query. The proposed multimodal memory selection design therefore prevents the model from being forced to condition on paired yet unnecessary modality memories when retrieving data for a given query, minimizing potential distraction during reasoning.

In addition, WorldMM is able to retrieve information at varying levels of granularity over the appropriate time range by leveraging multiple graphs operating at different temporal scales, such as seconds, minutes, and hours. When episodic memory is selected for retrieval, the retrieval agent searches each memory to gather potentially relevant information from all temporal levels. The collected candidates are then jointly examined to determine which pieces of information should be used to answer the query. In the end, we dynamically access both short- and long-term video contexts to assemble only the necessary information for reasoning. Furthermore, the model performs retrieval in multiple turns by iteratively selecting memories and queries, thereby expanding the range of possible combinations and allowing

adaptive selection of the information for each query.

We evaluate WorldMM with five long video questionanswering benchmarks from hour- to week-long durations. The proposed approach consistently outperforms strong baselines, including long video LLMs and memoryaugmented models. Comprehensive ablation studies further demonstrate the effectiveness of our multi-memory, multiscale design. Specifically, episodic memory enables reasoning over events at multiple timescales, visual memory improves performance on object- and action-centered queries, while semantic memory enhances reasoning over long-term contexts. When all the memories are adaptively integrated, the model achieves the best overall results. These results highlight that our multimodal memory system represents a promising direction toward robust long video reasoning.

# 2. Related Work

# 2.1. Long Video Understanding

Existing video LLMs demonstrate strong understanding capabilities for short videos, and recent research has shifted toward reasoning over longer videos. Current proprietary models, such as GPT-5 [16] and Gemini 2.5 [3], have advanced to minute- or hour-level video understanding [2, 5, 12, 25, 34] by utilizing extended context lengths. However, these models still incur high computational costs and uniformly sampling every frame is often suboptimal for questions focused on localized events [17, 24].

To address these challenges, several strategies have been explored. Visual token compression [8, 9, 11, 19, 20, 26, 35] improves efficiency by reducing token counts but often loses fine-grained details, limiting the capture of subtle or sparse events. Key frame selection [22, 30] retains only the most informative frames to reduce redundancy but fails to detect relevant frames when video streams are too long and may miss rare events. More recently, reasoningcentric training and inference [28, 29] have enhanced long-

![](images/83e0b6eb6d352b6e8f031d16cdaf907675f439e4e9f51cfd53f72895a6d938c0.jpg)

![](images/09137ee62dc8e8492e285b5b870b39117bf360e3eec8e70dcb6298f05bcdae95.jpg)  
Figure 2. Overview of WorldMM. (Left) Multimodal Memory Construction: WorldMM builds three complementary memories (episodic, semantic, and visual memory) that capture temporal events, long-term relations, and visual details from video streams. (Middle) Adaptive Memory Retrieval: A retrieval agent iteratively selects and integrates relevant information from diverse memories for a given query. (Right) Response Generation: The retrieved content and reasoning history are used by a response agent to produce a grounded response.

range temporal grounding through reinforcement learning and adaptive test-time scaling, yet still face scalability limits on ultra-long videos over ten hours.

Beyond hour-level videos, emerging benchmarks push video understanding and reasoning toward day- or even week-long continuous recordings [23, 33]. The aforementioned strategies struggle in these settings due to the massive scale of frames and long-term temporal dependencies. This highlights the necessity for more efficient, context-aware, and scalable approaches to handle extremely long videos.

# 2.2. Memory-based Video LLMs

In order to effectively reason over long videos, retrievalaugmented generation (RAG) based methods that retrieve relevant frames or clips instead of sampled frames have been introduced. They typically retrieve query-relevant information using textual and visual cues, and allow the model to focus on crucial clips [10, 14, 31]. Some recent methods extend this approach by constructing graph structures to encode multimodal interactions across frames [18, 21, 32]. However, these models rely on textual representation or a simple similarity score between visual and query features, limiting their ability to perform holistic understanding and complex reasoning over long video sequences.

Beyond naive retrieval-based design, memory-based methods have emerged to construct structured knowledge over video streams. EgoRAG [33] organizes hierarchical textual memories that store events from egocentric video streams in a hierarchical manner, allowing reasoning throughout day-long activities. Ego-R1 [23] extends this by leveraging vision-centric tools with iterative reasoning to perform long-horizon reasoning. HippoMM [12] proposes a dual-process memory using semantic summaries with multimodal cues. M3-Agent [13] constructs entity-centric long-term memory by processing multimodal contexts and adopts iterative reasoning to retrieve relevant knowledge

from the memory. Despite these advances, existing works still struggle to fully integrate multimodal information and to dynamically retrieve knowledge across varying temporal scales to handle complex, long video scenarios.

# 3. WorldMM

We introduce WorldMM, a novel framework that leverages both textual and visual contexts of video streams to build a multimodal memory for comprehensive understanding and reasoning over long videos. As illustrated in Fig. 2, the model operates in three stages: multimodal memory construction, adaptive memory retrieval, and response generation. Given a long video stream, WorldMM first builds multiple memories, including two textual memories and one visual memory (Sec. 3.1). Next, a retrieval agent iteratively collects query-relevant information from different memories and timescales until sufficient knowledge is gathered to answer the question (Sec. 3.2). Finally, the query is fed into a response agent along with the retrieved contents and retrieval history to generate a response (Sec. 3.3).

# 3.1. Multimodal Memory Construction

As we described in Sec. 1, an effective memory agent for long-form video understanding must address two key requirements: 1) adaptively leveraging visual information alongside text memory, and 2) retrieving knowledge across diverse temporal ranges. To achieve this, WorldMM constructs three types of memory, each encoding complementary video knowledge across diverse modalities. Episodic memory captures diverse events over multiple dynamic timescales, semantic memory incrementally updates highlevel relational knowledge, and visual memory preserves spatial and appearance details. Together, they form a comprehensive multimodal memory that supports episodic retrieval, semantic reasoning, and visually grounded understanding of long-form videos.

Episodic Memory Construction Episodic memory consists of multiple textual graphs, each of which encodes events at different temporal resolutions. Before constructing the graphs, we first perform fine-grained captioning on the unit temporal scale $t _ { 0 }$ . We divide the video into short segments of length $t _ { 0 }$ , each converted into a caption using a video LLM. Most existing approaches rely on a fixed temporal scale during memory construction [13, 33], overlooking the diverse spans of real-world events. In contrast, we introduce a multi-scale memory composed of multiple temporal resolutions that flexibly encodes information with different levels of density:

$$
\mathcal {T} = \left\{t _ {0}, t _ {1}, \dots , t _ {N} \right\}, \quad t _ {0} <   t _ {1} <   \dots <   t _ {N}. \tag {1}
$$

For each temporal scale $t _ { i } ~ \in ~ \tau$ , the video is partitioned into non-overlapping segments of length $t _ { i }$ . The segments are captioned and transformed into factual triplets (entityaction-entity) to construct a knowledge graph (KG) $G _ { t _ { i } }$ . Finally, episodic memory is represented as a set of KGs:

$$
\mathcal {M} _ {e} = \left\{G _ {t _ {0}}, G _ {t _ {1}}, \dots , G _ {t _ {N}} \right\}. \tag {2}
$$

This multi-scale episodic memory enables temporally grounded reasoning that spans both fine-grained event details and long-range narrative understanding.

Semantic Memory Construction Semantic memory captures long-term, evolving knowledge about relationships and habits within a video. Since episodic graphs are constructed from independent events, they fail to preserve continuity across distant scenes and cannot capture high-level knowledge. Semantic memory, on the contrary, maintains an evolving graph that continuously integrates relational and habitual knowledge over time.

To build this continually updating memory, we first split the input video into coarse segments with a fixed timescale $t _ { s }$ . Textual captions are generated for each segment and converted into semantic triplets $T _ { t _ { s } } ^ { k }$ , focusing on conceptual knowledge rather than event-specific details. These triplets are incrementally integrated into an evolving semantic graph through a consolidation process that merges new knowledge while preserving stable relationships. Specifically, embedding-based similarity is first used to identify overlapping or conflicting triplets between the current graph $G _ { t _ { s } } ^ { k }$ and the newly extracted triplets $\boldsymbol { T } _ { t _ { s } } ^ { k + 1 }$ ts . The matched triplets are then provided to an LLM along with the new triplets, which determines outdated or conflicting triplets $T _ { \mathrm { r e m o v e } }$ and triplets that should be revised or added $T _ { \mathrm { u p d a t e } }$ . Formally, the consolidation process can be represented as:

$$
\operatorname {C o n s o l i d a t e} \left(G _ {t _ {s}} ^ {k}, T _ {t _ {s}} ^ {k + 1}\right) = \left(G _ {t _ {s}} ^ {k} \backslash T _ {\text {r e m o v e}}\right) \cup T _ {\text {u p d a t e}}. \tag {3}
$$

The resulting semantic memory is a continuously evolving KG $\mathcal { M } _ { s } = G _ { t _ { s } } ^ { M }$ , where $M$ denotes the final segment index, capturing the video’s long-term knowledge.

Visual Memory Construction Visual memory captures rich visual details that text cannot fully convey, including detailed object appearances, scene dynamics, and precise spatial context. WorldMM explicitly constructs a visual memory to ground reasoning in visual evidence. We consider two scenarios in which visual memory is invoked, when the retrieval agent searches for scenes associated with a specific keyword, and when the agent has timestamps identified during preceding retrieval steps to inspect the corresponding frames. Therefore, we adopt two complementary strategies for building visual memory: feature-based retrieval via natural language query, and timestamp-based retrieval for precise temporal grounding.

Specifically, we partition each video into short, fixedlength segments of duration $t _ { v }$ , encoding each segment $V _ { t _ { v } } ^ { k }$ into a visual feature $f _ { v } ^ { k }$ using a multimodal encoder, forming a feature-based visual memory as a set of embeddings:

$$
\mathcal {M} _ {v} ^ {f} = \left\{f _ {v} ^ {1}, f _ {v} ^ {2}, \dots , f _ {v} ^ {L} \right\}. \tag {4}
$$

In parallel, to support timestamp-based retrieval, each frame is paired with its corresponding timestamp:

$$
\mathcal {M} _ {v} ^ {I} = \left\{\left(t _ {i}, I _ {i}\right) \mid I _ {i} = V \left(t _ {i}\right), t _ {i} \in [ 0, \operatorname {l e n} (V) ] \right\}. \tag {5}
$$

This allows direct access to visual evidence at specific moments in the video. Finally, the complete visual memory integrates both components $\mathcal { M } _ { v } = \mathcal { M } _ { v } ^ { f } \cup \mathcal { M } _ { v } ^ { I }$ by combining feature-level embeddings and frame-level indices.

# 3.2. Adaptive Memory Retrieval

In this section, we present how WorldMM dynamically retrieves the most relevant multimodal memories from the appropriate temporal scope for a given query.

Retrieval Agent Reasoning over long-form videos requires integrating heterogeneous information from multiple memory sources. To handle this, the retrieval agent iteratively decides which memory to access and what query to issue, conditioned on the user question and retrieval history. Leveraging the distinct characteristics of each memory component, it adaptively selects the most relevant source and formulates modality-specific queries. Through successive iterations, the agent progressively refines its retrieval strategy and constructs better knowledge collection.

Formally, we define the retrieval agent $\mathcal { R }$ as a multimodal reasoning module that iteratively selects a memory source and formulates a corresponding query. At each iteration i, $\mathcal { R }$ takes an input the user query $q$ and the set of previous retrieval histories $r _ { < i } = \{ r _ { 1 } , \ldots , r _ { i - 1 } \}$ , and outputs either a memory–query pair or a STOP signal:

$$
\mathcal {R} (q, r _ {<   i}) = \left\{ \begin{array}{l l} \left(m _ {i}, q _ {i}\right) & \text {i f} r _ {<   i} \text {i n s u f f i c i e n t a n d} i \leq N, \\ \text {S T O P} & \text {o t h e r w i s e ,} \end{array} \right. \tag {6}
$$

where $m _ { i } ~ \in ~ \{ \mathcal { M } _ { e } , \mathcal { M } _ { s } , \mathcal { M } _ { v } \}$ and $N$ denotes the maximum number of iterations. If the retriever outputs a memory–query pair $( m _ { i } , q _ { i } )$ , it retrieves the relevant information from the memory $m _ { i }$ with search query $q _ { i }$ and proceeds to the next iteration with the updated context $r _ { \leq i }$ . When the retriever outputs STOP, it indicates that sufficient information has been collected. The iterative process then terminates, and all retrieved results $\{ r _ { 1 } , \ldots , r _ { n } \}$ are passed to the response agent for the final response generation.

Episodic Memory Retrieval Episodic memory retrieval is guided by a query $q$ provided by the retriever, which specifies the desired information from episodic memory. The main challenge lies in determining the appropriate temporal scope, as episodic memory contains multiple graphs covering different temporal ranges. WorldMM adopts a coarseto-fine, multi-timescale retrieval strategy. Specifically, for each temporal graph $G _ { t _ { i } }$ , the model first retrieves top- $k$ candidate captions using a graph-based retrieval framework guided by the Personalized PageRank (PPR) score and the query, following Gutierrez et al. [ ´ 7]. Subsequently, an LLM serves as a cross-scale reranker, jointly analyzing the query and retrieved candidates across all timescales. It then selects the most relevant temporal range and refines the retrieved content, producing the final top- $\mathbf { \nabla } m$ captions as output. By retrieving from multi-scale memory, the model leverages both coarse temporal context and fine-grained details.

Semantic Memory Retrieval The semantic memory, also represented as a graph, is queried using a PPR-based retrieval algorithm. In contrast to episodic memory retrieval which operates over nodes and their temporal structures, semantic retrieval focuses on relational knowledge encoded as edges between entities. Since the standard PPR score measures node-level relevance, we adapt it for edge-based reasoning by assigning each edge a score equal to the sum of the PPR values of its two connected nodes. The top- $k$ triplets corresponding to the highest-scoring edges are then selected as the final retrieved results.

Visual Memory Retrieval Following Sec. 3.1, visual memory retrieval operates in two complementary modes: feature-based search and timestamp-based access. In feature-based mode, the retrieval agent formulates a query $q$ , encodes it into a text feature $f _ { t }$ using a multimodal encoder, and retrieves the top- $k$ relevant video segments from $\mathcal { M } _ { v } ^ { f }$ based on the cosine similarity between $f _ { t }$ and the visual features. In timestamp-based mode, when specific temporal ranges are identified, typically following episodic retrieval, the corresponding frames are directly fetched from $\mathcal { M } _ { v } ^ { I }$ . By combining these two modes, WorldMM enables flexible and effective access to visual information at both semantic and temporal levels.

# 3.3. Response Generation

Finally, once the retrieval agent determines that sufficient information has been gathered, the retrieval process is terminated. The retrieval history, including the selected memories, their corresponding queries, and the retrieved results, is then passed to the response agent along with the original user query. The response agent generates the final answer by grounding its response in the retrieved information. This clear separation between the retriever and the responder allows each component to focus on its respective objective, ensuring effective retrieval and response generation.

# 4. Experiment Results

# 4.1. Experimental Setup

Datasets and Metrics We assess the performance of WorldMM across five benchmarks that require reasoning over long videos. EgoLifeQA [33] and Ego-R1 Bench [23] contain week-long videos, and HippoVlog [12] features vlog-style content, requiring comprehension of audio and visual streams. We also assess general video understanding on LVBench [25] and Video-MME (long) [5] with hourlevel videos. All benchmarks consist of multiple-choice questions, with accuracy used as the evaluation metric. Please see additional dataset details in the Sec. A.

Baselines We compare WorldMM against a comprehensive set of baselines spanning base video LLMs, long video understanding models, RAG systems, and memory-based models. Base video LLMs include GPT-5 [16], Gemini 2.5 Pro [3], and Qwen3-VL-8B-Instruct [1], while long video understanding models include VideoChat-Flash [11], Time-R1 [28], and Video-RTS [29], which all use uniformly sampled frames within their input capacity. We further evaluate RAG approaches, including text retrieval methods like LightRAG [6] and HippoRAG [7], which retrieve video captions, and Video-RAG [14], which retrieves relevant clips. Finally, we compare with memory-based frameworks for long video reasoning, including EgoRAG [33], Ego-R1 [23], HippoMM [12], and M3-Agent [13].

Implementations Details We adopt VLM2Vec-V2 [15] as a multimodal encoder for visual memory retrieval. During the memory construction, GPT-5-mini is used for building episodic and semantic memories. We experiment ours with two video LLMs, GPT-5 and Qwen3-VL-8B-Instruct, which serve as the retrieval and response agent, respectively denoted as WorldMM-GPT and WorldMM-8B. For temporal segmentation in episodic memory, we apply timescales specific to each dataset. For example, we use 30-second, 3-minute, 10-minute, and 1-hour intervals for EgoLifeQA. Configurations for other benchmarks, more experimental details, and the prompts are provided in Sec. B.

Table 1. Performance of WorldMM with various baselines across long video QA benchmarks. “–” denotes a proprietary backbone.   

<table><tr><td>Model</td><td></td><td>EgoLife QA</td><td>Ego-R1 Bench</td><td>Hippo Vlog</td><td>LV Bench</td><td>Video-MME (L)</td><td>Avg.</td></tr><tr><td colspan="8">Base Models</td></tr><tr><td>Qwen3-VL-8B [1]</td><td>8B</td><td>38.6</td><td>35.7</td><td>74.4</td><td>48.3</td><td>61.0</td><td>51.6</td></tr><tr><td>Gemini 2.5 Pro [3]</td><td>-</td><td>46.4</td><td>46.7</td><td>72.0</td><td>57.0</td><td>55.7</td><td>55.6</td></tr><tr><td>GPT-5 [16]</td><td>-</td><td>48.6</td><td>46.3</td><td>75.7</td><td>60.4</td><td>74.3</td><td>61.1</td></tr><tr><td colspan="8">Long Video LLMs</td></tr><tr><td>VideoChat-Flash [11]</td><td>7B</td><td>34.2</td><td>42.7</td><td>58.0</td><td>33.2</td><td>44.1</td><td>42.4</td></tr><tr><td>Time-R1 [28]</td><td>3B</td><td>48.8</td><td>48.0</td><td>54.6</td><td>31.1</td><td>37.6</td><td>44.0</td></tr><tr><td>Video-RTS [29]</td><td>7B</td><td>48.2</td><td>47.4</td><td>59.0</td><td>39.8</td><td>47.9</td><td>48.6</td></tr><tr><td colspan="8">RAG-based Video LLMs</td></tr><tr><td>LightRAG [6]</td><td>-</td><td>48.8</td><td>52.3</td><td>47.4</td><td>30.4</td><td>46.6</td><td>45.1</td></tr><tr><td>HippoRAG [7]</td><td>-</td><td>59.6</td><td>56.0</td><td>63.2</td><td>54.0</td><td>52.1</td><td>57.0</td></tr><tr><td>Video-RAG [14]</td><td>-</td><td>55.4</td><td>49.7</td><td>65.1</td><td>33.1</td><td>55.4</td><td>51.7</td></tr><tr><td colspan="8">Memory-based Video LLMs</td></tr><tr><td>EgoRAG [33]</td><td>-</td><td>52.0</td><td>49.0</td><td>57.5</td><td>32.2</td><td>41.1</td><td>46.4</td></tr><tr><td>Ego-R1 [23]</td><td>3B</td><td>53.0</td><td>52.0</td><td>58.8</td><td>34.1</td><td>42.7</td><td>48.1</td></tr><tr><td>HippoMM [12]</td><td>-</td><td>54.6</td><td>53.0</td><td>71.9</td><td>38.2</td><td>41.6</td><td>51.8</td></tr><tr><td>M3-Agent [13]</td><td>7B</td><td>53.5</td><td>52.0</td><td>65.5</td><td>49.3</td><td>55.3</td><td>55.1</td></tr><tr><td colspan="8">WorldMM (Ours)</td></tr><tr><td>WorldMM-8B</td><td>8B</td><td>56.4</td><td>52.0</td><td>69.7</td><td>55.4</td><td>66.0</td><td>59.9</td></tr><tr><td>WorldMM-GPT</td><td>-</td><td>65.6</td><td>65.3</td><td>78.3</td><td>61.9</td><td>76.6</td><td>69.5</td></tr></table>

# 4.2. Main Results

Tab. 1 presents the evaluation results of the proposed WorldMM and baseline models. WorldMM significantly outperforms all baselines across various long video understanding benchmarks. In particular, WorldMM-GPT achieves an average score of $6 9 . 5 \%$ , exceeding the strongest baseline by $8 . 4 \%$ . Compared with base models, both variants of our model surpass their corresponding baselines by more than $8 \%$ on average, highlighting the effectiveness of our framework in leveraging strong reasoning capabilities without requiring full video processing. On the other hand, models in the long video LLM category show the weakest performance, with all results falling below $50 \%$ on EgoLifeQA and Ego-R1 Bench, indicating that these approaches are not effective in days-long videos.

Meanwhile, retrieval- and memory-based approaches, including ours, achieve scores mostly above $52 \%$ on Ego-LifeQA and Ego-R1 Bench, suggesting that selective retrieval of relevant segments is more effective for long video understanding than processing full video sequences. Compared with other retrieval-based models such as HippoRAG and HippoMM, which also rely on GPT backbones, our model achieves markedly higher accuracy on average $( 6 9 . 5 \%$ vs. $5 7 . 0 \%$ and $5 1 . 8 \%$ ). These demonstrate that integrating textual and visual memory and adaptively selecting temporal scopes are crucial for effective video reasoning.

# 4.3. Efficacy of Multimodal Memory

To examine the contribution of each memory in our framework, we perform an ablation study that varies the composition of available memories. The evaluation results, summarized in Tab. 2, show a consistent improvement in performance as additional memories are incorporated. This

![](images/bbf5e7aca267bc6bcd115fd80c2bbf7b5ddef70bfea580fa184ef6c7cb1f2012.jpg)  
Figure 3. Memory type utilization of WorldMM on five distinctive categories in EgoLifeQA.

finding confirms that different memories capture complementary forms of knowledge. All following experiments in this section are conducted using WorldMM-GPT.

Effect of Episodic Memory To examine the performance differences arising from the retrieved data modality in WorldMM, we evaluate models using only episodic memory (E) and only visual memory (V), and report the results in Tab. 2. Using only episodic memory shows $20 \%$ higher performance than using only visual memory on average. This is mostly because textual information can be more readily organized into a graph, which enables effective retrieval, while indexing visual frames into a structured representation remains challenging.

Effect of Visual Memory Visual memory plays a particularly important role in categories that demand perceptual understanding, such as object recognition or action interpretation. In Tab. 2, visual memory significantly enhances accuracy in categories like EntityLog and EventRecall of EgoLifeQA and Ego-R1 Bench, as well as Visual and Audio+Visual of HippoVlog. The full configuration $\left( \mathrm { E } { + } \mathrm { S } { + } \mathrm { V } \right)$ surpasses the non-visual configuration $\left( \mathrm { E } { + } \mathrm { S } \right)$ by an average margin of $4 . 2 \%$ . This improvement arises because visual information preserves spatial and perceptual details that are difficult to represent in text. As shown in Fig. 4(a), when relying solely on episodic memory, the model fails to capture object details such as the type of baked item, leading to an incorrect response. In contrast, visual memory provides access to corresponding frames that contain a complete scene, enabling accurate interpretation of objects and activities.

Effect of Semantic Memory Semantic memory proves the most beneficial for categories that require reasoning over long-term dependencies and abstract relationships. This effect is evident in the HabitInsight and Relation-Map categories of EgoLifeQA and Ego-R1 Bench. The model equipped with full memory achieves $7 6 . 9 \%$ accuracy in HabitInsight, representing a $23 \%$ improvement over the setting without semantic memory $\mathrm { ( E + V ) }$ . This substantial gain indicates that semantic memory serves as a structured

Table 2. Performance of WorldMM across multiple benchmarks using different memory types. E, S, and V denote episodic, semantic, and visual memories, respectively. Combinations with “+” indicate multiple memory types are used.   

<table><tr><td rowspan="2">Model</td><td colspan="6">EgoLifeQA</td><td colspan="6">Ego-R1 Bench</td><td colspan="6">HippoVlog</td><td>LVBench</td><td>Video-MME (L)</td><td>Avg.</td></tr><tr><td>Ent.</td><td>EvR.</td><td>Hab.</td><td>Rel.</td><td>Task</td><td>Avg.</td><td>Ent.</td><td>EvR.</td><td>Hab.</td><td>Rel.</td><td>Task</td><td>Avg.</td><td>Aud.</td><td>Vis.</td><td>A+V</td><td>Summ.</td><td>Avg.</td><td></td><td></td><td></td><td></td></tr><tr><td>E</td><td>57.6</td><td>61.1</td><td>70.5</td><td>61.6</td><td>69.8</td><td>62.6</td><td>54.5</td><td>70.7</td><td>53.9</td><td>52.6</td><td>57.9</td><td>57.0</td><td>72.4</td><td>73.2</td><td>68.4</td><td>80.4</td><td>73.6</td><td>60.6</td><td>72.7</td><td>64.9</td><td></td></tr><tr><td>V</td><td>40.8</td><td>35.7</td><td>36.1</td><td>34.4</td><td>39.7</td><td>37.2</td><td>36.5</td><td>34.1</td><td>23.1</td><td>31.6</td><td>28.2</td><td>34.2</td><td>35.2</td><td>66.4</td><td>54.8</td><td>48.8</td><td>51.3</td><td>47.4</td><td>64.2</td><td>44.9</td><td></td></tr><tr><td>E+S</td><td>56.8</td><td>61.9</td><td>73.8</td><td>62.4</td><td>71.4</td><td>63.4</td><td>59.3</td><td>68.3</td><td>69.2</td><td>57.9</td><td>60.5</td><td>61.0</td><td>70.8</td><td>75.2</td><td>68.8</td><td>80.4</td><td>73.8</td><td>58.8</td><td>74.1</td><td>66.8</td><td></td></tr><tr><td>E+V</td><td>59.2</td><td>63.5</td><td>70.5</td><td>60.8</td><td>68.8</td><td>63.3</td><td>65.1</td><td>68.3</td><td>53.9</td><td>47.4</td><td>57.9</td><td>63.0</td><td>73.2</td><td>77.2</td><td>70.8</td><td>79.6</td><td>75.2</td><td>59.8</td><td>76.0</td><td>66.9</td><td></td></tr><tr><td>E+S+V</td><td>62.4</td><td>64.3</td><td>75.4</td><td>62.4</td><td>71.4</td><td>65.6</td><td>64.6</td><td>70.7</td><td>76.9</td><td>57.9</td><td>63.2</td><td>65.3</td><td>75.6</td><td>81.6</td><td>73.2</td><td>82.8</td><td>78.3</td><td>61.9</td><td>76.6</td><td>69.5</td><td></td></tr></table>

(a) Episodic + Visual Memory

Now Tasha suggested finding a container to put the eggs in and start baking. What did we bake in the oven last time? (a) Sweet Potato (b) Bread (c) Egg Tart (d) Pizza

Turn #1 Call(Episodic, Oven Baking)

I chat as Tasha brings out baked goods and the warm aroma washes over me. Shure heads out again while I work at the computer.

Visual context is not represented in text

![](images/235b50aefc5eba84d36b42862dcaa7bec3ecc51c9bce47b4102daf5da9fd4d2b.jpg)  
Turn #2 Call(Visual, Sweet Potato)

![](images/cb6f0a5d217e1540dcbb509183ea5f00a8489a56b279a6f28f806901f8b716cf.jpg)

![](images/d278c552e54f1957f76f61679fdaec38d70fe3a15038980c7f4c075c2c2ea093.jpg)

![](images/817e09e972e902778652cbd550e1676e54799fd2939054270291cce6182eead3.jpg)

![](images/5bdeed796ebfa89640a6e2694fe6cc33dcc4ce1f9477d5e04d42ca97f9dfe23b.jpg)  
anding of the scene   
(a) Sweet Potato

Turn #1 Call(Episodic, Wiping kitchenware after washing)   
![](images/51c061d854676e257ee63916b75628a0f8cca14e7878d8d9e0b3444a6b6fe3fc.jpg)  
What do I habitually use to wipe after washing kitchenware? (a) Kitchen wet wipes (b) Face towel (c) Dry paper towels (d) Cloth

(b) Episodic + Semantic Memory

I wash the dishes. Tasha says, "Okay, so it's real then." … Tasha laughs. I turn my head right, then left. Alice says…

Retrieval Agent

I wasn’t able to identify any habitual evidence related to wiping kitchenware. I’ll refer to the semantic memory to determine the most likely object used for this task.

Habit cannot be captured within a single episode

Turn #2 Call(Semantic, Kitchen wet wipes)

(I, often use, kitchen wet wipes) (Wet wipes, can be used as an alternative to, paper towels)

Semantic memory can capture long-term habits   
Figure 4. Qualitative results. (a) Episodic memory alone cannot capture detailed visual context. The retrieval agent dynamically retrieves from visual memory, enabling access to fine-grained visual details. (b) To address the limitations of episodic memory in representing relationships or habitual behaviors, the retrieval agent proactively accesses semantic memory, allowing it to incorporate habitual knowledge.   
![](images/ccfde6fb9004deb1b2780b40663d6892d2850af21c65117211d2375e4f057732.jpg)  
(a) Kitchen wetwipes

Response Agent

knowledge base that captures relational or habitual knowledge accumulated over time. The qualitative example in Fig. 4(b) illustrates this behavior, where episodic memory alone fails to infer habitual actions that extend beyond a single event. By contrast, semantic memory captures the repeated use of kitchen wet wipes, allowing the model to infer the correct answer through long-term reasoning.

Adaptive Retrieval on Multimodal Memory To further analyze how categories differ in their reliance on distinct memory modalities in WorldMM, we quantify the usage proportion of each memory across all retrieval iterations per category. As shown in Fig. 3, while episodic memory plays a foundational role across all tasks, certain categories tend to select it more frequently than other memory types: HabitInsight and RelationMap depend primarily on semantic memory, reflecting their reliance on reasoning over longterm patterns. In contrast, EntityLog and EventRecall benefit more from visual memory, which provides fine-grained perceptual details not fully captured by text. This selective utilization suggests that the model dynamically emphasizes the most relevant memory type for each category, leveraging the strength of each type in a context-dependent manner. These results confirm that different types of memory contribute distinct yet complementary strengths.

# 4.4. Dynamic Temporal Scope Retrieval

We evaluate episodic memory retrieval performance across diverse temporal scales of events using temporal intersection over union (tIoU), which measures the overlap between retrieved and ground truth segments as the ratio of their intersection to their union duration. We compare WorldMM with various models in temporal grounding, single-modality retrieval, long-form egocentric video retrieval, and keyframe selection. Details about baselines are given in Sec. C.1. As shown in Tab. 3, WorldMM significantly superior tIoU scores than strong baselines. Notably, reasoning-based retrieval and keyframe selection methods exhibit lower tIoU values, indicating difficulty in handling long input contexts. Moreover, Fig. 5 demonstrates that the superior tIoU is directly correlated with higher overall accuracy, particularly in understanding long videos.

# 4.5. Efficacy of Multi-turn Retrieval

We validate the effectiveness of our model’s multi-turn approach by limiting the maximum number of retrieval steps. The results in Fig. 7 show that performance consistently improves as the number of iterations increases across all benchmarks. Notably, on the EgoLifeQA benchmark, allowing a maximum of five steps yields a $9 . 3 \%$ improvement

Table 3. Average tIoU $( \% )$ across three benchmarks.   

<table><tr><td>Model</td><td>EgoLifeQA</td><td>Ego-R1 Bench</td><td>LVBench</td></tr><tr><td>Time-R1 [28]</td><td>0.58</td><td>0.59</td><td>2.70</td></tr><tr><td>Qwen3 Emb. [36]</td><td>4.35</td><td>2.87</td><td>4.54</td></tr><tr><td>HippoRAG [7]</td><td>4.00</td><td>3.28</td><td>4.30</td></tr><tr><td>InternVideo2 [27]</td><td>3.36</td><td>2.60</td><td>3.55</td></tr><tr><td>EgoRAG [33]</td><td>3.60</td><td>2.73</td><td>3.50</td></tr><tr><td>Ego-R1 [23]</td><td>3.70</td><td>2.89</td><td>3.60</td></tr><tr><td>AKS [22]</td><td>2.75</td><td>2.30</td><td>3.52</td></tr><tr><td>WorldMM (Ours)</td><td>10.09</td><td>9.17</td><td>9.57</td></tr></table>

Table 4. Comparison of model variants by changing each module.   

<table><tr><td rowspan="2">Model</td><td colspan="6">EgoLifeQA</td><td>LVBench</td></tr><tr><td>Ent.</td><td>EvR.</td><td>Hab.</td><td>Rel.</td><td>Task</td><td>Avg.</td><td>Acc.</td></tr><tr><td colspan="8">Episodic Memory</td></tr><tr><td>Fixed Timescale</td><td>44.8</td><td>51.6</td><td>60.7</td><td>51.2</td><td>58.7</td><td>51.8</td><td>47.9</td></tr><tr><td>Embedding Retrieval</td><td>45.6</td><td>52.4</td><td>59.0</td><td>54.4</td><td>52.4</td><td>52.0</td><td>50.9</td></tr><tr><td colspan="8">Semantic Memory</td></tr><tr><td>w/o Consolidation</td><td>48.8</td><td>53.2</td><td>57.4</td><td>51.2</td><td>60.7</td><td>53.0</td><td>54.2</td></tr><tr><td colspan="8">Visual Memory</td></tr><tr><td>Feature Retrieval</td><td>45.6</td><td>51.6</td><td>62.3</td><td>58.4</td><td>55.6</td><td>53.6</td><td>52.4</td></tr><tr><td>Timestamp Retrieval</td><td>41.6</td><td>50.0</td><td>63.9</td><td>56.8</td><td>54.0</td><td>51.8</td><td>52.9</td></tr><tr><td>WorldMM (Ours)</td><td>49.6</td><td>56.4</td><td>63.9</td><td>58.4</td><td>58.7</td><td>56.4</td><td>55.4</td></tr></table>

![](images/75b88f747ce715889d459928900da116d905070bd3bc0382a313c1467c0fdc97.jpg)  
Figure 5. Average tIoU and performance of WorldMM and baselines.

![](images/6ed0893c64a8585da26d8ba1f45a7fa8f0ad9187b958f6439831c91fab588511.jpg)  
Figure 6. Average latency and performance of WorldMM and baselines.

![](images/c81e7e759d188571f66e2a6d82bc1672d4ea29d100fed21bcaf7d1934c352c48.jpg)  
Figure 7. Accuracy of WorldMM with different maximum retrieval steps.

over single-step retrieval. This gain arises because multiple iterations enable the retrieval agent to gather additional relevant information and refine its retrieval strategy when earlier attempts are suboptimal. An example of this refinement is shown in Sec. E.2, where the model corrects an initially irrelevant retrieval to produce a more accurate and contextually grounded response.

# 4.6. Analysis on Efficiency

To assess the efficiency of our framework, we measure the end-to-end latency of WorldMM on 100 randomly sampled queries from EgoLifeQA. As shown in Fig. 6, our method achieves a superior latency–accuracy trade-off compared with baselines. Long-video LLMs incur significantly higher inference latency, while still exhibiting relatively low performance. Although RAG- or memory-based approaches offer better latency, they often require substantial preprocessing and show a significant performance gap. In contrast, by allowing the retrieval agent to adaptively finish iterations and by retrieving only the relevant segments, WorldMM achieves low latency and substantially higher accuracy.

# 4.7. Efficacy of Memory Modules

To enable effective long video reasoning, WorldMM applies different strategies for each memory. Tab. 4 reports the results of WorldMM-8B under various module configurations with details about each method in Sec. C.2. For episodic memory, using a fixed single timescale or embed-

dings instead of graphs results in a $6 . 1 \%$ and $4 . 4 \%$ drop in average accuracy, respectively, highlighting the importance of multi-scale structured knowledge. For semantic memory, removing the consolidation process results in approximately $7 \%$ drops for the category that requires long-term reasoning, demonstrating the need for continuous integration of knowledge to support long-term reasoning. Finally, for visual memory, disabling its dual-mode retrieval leads to an accuracy drop of about $3 \%$ , indicating that each mode contributes complementary benefits for retrieving particular scenes or accessing broader temporal ranges.

# 5. Conclusion

We propose WorldMM, a novel memory agent designed to perceive and remember the world as represented in long video streams. To address the challenges of long video reasoning, we introduce a multimodal, multi-scale memory that integrates textual and visual information through adaptive retrieval. By constructing separate memories across different modalities and timescales, together with a retrieval agent that iteratively identifies relevant information, our approach enables effective and flexible reasoning over long videos. We validate our model on multiple benchmarks ranging from hour- to week-long videos, demonstrating that WorldMM provides a promising solution capable of robust performance across various long video reasoning tasks.

# References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 1, 5, 6, 15   
[2] Keshigeyan Chandrasegaran, Agrim Gupta, Lea M. Hadzic, Taran Kota, Jimming He, Cristobal Eyzaguirre, Zane Du- ´ rante, Manling Li, Jiajun Wu, and Li Fei-Fei. Hourvideo: 1-hour video-language understanding. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. 2   
[3] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025. 1, 2, 5, 6, 15   
[4] Yue Fan, Xiaojian Ma, Rujie Wu, Yuntao Du, Jiaqi Li, Zhi Gao, and Qing Li. Videoagent: A memory-augmented multimodal agent for video understanding. In European Conference on Computer Vision, pages 75–92, 2024. 1   
[5] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24108–24118, 2025. 1, 2, 5, 11, 12   
[6] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrieval-augmented generation. arXiv preprint arXiv:2410.05779, 2024. 5, 6, 12, 15   
[7] Bernal Jimenez Guti´ errez, Yiheng Shu, Weijian Qi, Sizhe´ Zhou, and Yu Su. From rag to memory: Non-parametric continual learning for large language models. arXiv preprint arXiv:2502.14802, 2025. 5, 6, 8, 12, 15, 16   
[8] Bo He, Hengduo Li, Young Kyun Jang, Menglin Jia, Xuefei Cao, Ashish Shah, Abhinav Shrivastava, and Ser-Nam Lim. Ma-lmm: Memory-augmented large multimodal model for long-term video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13504–13514, 2024. 2   
[9] Sunil Hwang, Jaehong Yoon, Youngwan Lee, and Sung Ju Hwang. Everest: Efficient masked video autoencoder by removing redundant spatiotemporal tokens. In International Conference on Machine Learning, 2024. 2   
[10] Soyeong Jeong, Kangsan Kim, Jinheon Baek, and Sung Ju Hwang. VideoRAG: Retrieval-augmented generation over video corpus. In Findings of the Association for Computational Linguistics: ACL 2025, pages 21278–21298, Vienna, Austria, 2025. Association for Computational Linguistics. 3   
[11] Xinhao Li, Yi Wang, Jiashuo Yu, Xiangyu Zeng, Yuhan Zhu, Haian Huang, Jianfei Gao, Kunchang Li, Yinan He, Chenting Wang, et al. Videochat-flash: Hierarchical com-

pression for long-context video modeling. arXiv preprint arXiv:2501.00574, 2024. 2, 5, 6, 15

[12] Yueqian Lin, Qinsi Wang, Hancheng Ye, Yuzhe Fu, Hai Li, Yiran Chen, et al. Hippomm: Hippocampal-inspired multimodal memory for long audiovisual event understanding. arXiv preprint arXiv:2504.10739, 2025. 1, 2, 3, 5, 6, 11, 15   
[13] Lin Long, Yichen He, Wentao Ye, Yiyuan Pan, Yuan Lin, Hang Li, Junbo Zhao, and Wei Li. Seeing, listening, remembering, and reasoning: A multimodal agent with long-term memory. arXiv preprint arXiv:2508.09736, 2025. 1, 2, 3, 4, 5, 6, 12, 14, 15   
[14] Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, and Rongrong Ji. Video-rag: Visually-aligned retrievalaugmented long video comprehension. arXiv preprint arXiv:2411.13093, 2024. 3, 5, 6, 12, 15   
[15] Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, et al. Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590, 2025. 5   
[16] OpenAI. Gpt-5 system card, 2025. 1, 2, 5, 6, 15   
[17] Jongwoo Park, Kanchana Ranasinghe, Kumara Kahatapitiya, Wonjeong Ryu, Donghyun Kim, and Michael S Ryoo. Too many frames, not all useful: Efficient strategies for longform video qa. arXiv preprint arXiv:2406.09396, 2024. 2   
[18] Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. Videorag: Retrieval-augmented generation with extreme long-context videos. arXiv preprint arXiv:2502.01549, 2025. 3   
[19] Saul Santos, Antonio Farinhas, Daniel C McNamee, and ´ Andre FT Martins. ´ $\infty$ -video: A training-free approach to long video understanding via continuous-time memory consolidation. arXiv preprint arXiv:2501.19098, 2025. 2   
[20] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 2   
[21] Xiaoqian Shen, Wenxuan Zhang, Jun Chen, and Mohamed Elhoseiny. Vgent: Graph-based retrieval-reasoningaugmented generation for long video understanding, 2025. 3   
[22] Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive keyframe sampling for long video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29118–29128, 2025. 2, 8, 13, 16   
[23] Shulin Tian, Ruiqi Wang, Hongming Guo, Penghao Wu, Yuhao Dong, Xiuying Wang, Jingkang Yang, Hao Zhang, Hongyuan Zhu, and Ziwei Liu. Ego-r1: Chain-of-toolthought for ultra-long egocentric video reasoning. arXiv preprint arXiv:2506.13654, 2025. 1, 3, 5, 6, 8, 11, 12, 15, 16   
[24] Shaoguang Wang, Ziyang Chen, Yijie Xu, Weiyu Guo, and Hui Xiong. Less is more: Token-efficient video-qa via adap-

tive frame-pruning and semantic graph integration. arXiv preprint arXiv:2508.03337, 2025. 2   
[25] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, et al. Lvbench: An extreme long video understanding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22958–22967, 2025. 1, 2, 5, 11   
[26] Xidong Wang, Dingjie Song, Shunian Chen, Junyin Chen, Zhenyang Cai, Chen Zhang, Lichao Sun, and Benyou Wang. Longllava: Scaling multi-modal llms to 1000 images efficiently via a hybrid architecture. arXiv preprint arXiv:2409.02889, 2024. 2   
[27] Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pages 396–416, 2024. 8, 12, 16   
[28] Ye Wang, Ziheng Wang, Boshen Xu, Yang Du, Kejun Lin, Zihan Xiao, Zihao Yue, Jianzhong Ju, Liang Zhang, Dingyi Yang, et al. Time-r1: Post-training large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025. 2, 5, 6, 8, 12, 15, 16   
[29] Ziyang Wang, Jaehong Yoon, Shoubin Yu, Md Mohaiminul Islam, Gedas Bertasius, and Mohit Bansal. Video-rts: Rethinking reinforcement learning and test-time scaling for efficient and enhanced video reasoning. arXiv preprint arXiv:2507.06485, 2025. 2, 5, 6, 15   
[30] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3272– 3283, 2025. 2   
[31] Zeyu Xu, Junkang Zhang, Qiang Wang, and Yi Liu. E-vrag: Enhancing long video understanding with resource-efficient retrieval augmented generation, 2025. 3   
[32] Zhucun Xue, Jiangning Zhang, Xurong Xie, Yuxuan Cai, Yong Liu, Xiangtai Li, and Dacheng Tao. Adavideorag: Omni-contextual adaptive retrieval-augmented efficient long video understanding. arXiv preprint arXiv:2506.13589, 2025. 3   
[33] Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, et al. Egolife: Towards egocentric life assistant. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28885–28900, 2025. 1, 2, 3, 4, 5, 6, 8, 11, 12, 15, 16   
[34] Emmanouil Zaranis, Antonio Farinhas, Saul Santos, Beat-´ riz Canaverde, Miguel Moura Ramos, Aditya K Surikuchi, Andre Viveiros, Baohao Liao, Elena Bueno-Benito, Nithin´ Sivakumaran, Pavlo Vasylenko, Shoubin Yu, Sonal Sannigrahi, Wafaa Mohammed, Ben Peters, Danae Sanchez Vil-´ legas, Elias Stengel-Eskin, Giuseppe Attanasio, Jaehong Yoon, Stella Frank, Alessandro Suglia, Chrysoula Zerva, Desmond Elliott, Mariella Dimiccoli, Mohit Bansal, Oswald Lanz, Raffaella Bernardi, Raquel Fernandez, Sandro´

Pezzelle, Vlad Niculae, and Andre F. T. Martins. Movie facts´ and fibs (mf2): A benchmark for long movie understanding. arXiv preprint arXiv:2506.06275, 2025. 2   
[35] Pan Zhang, Xiaoyi Dong, Yuhang Cao, Yuhang Zang, Rui Qian, Xilin Wei, Lin Chen, Yifei Li, Junbo Niu, Shuangrui Ding, et al. Internlm-xcomposer2.5-omnilive: A comprehensive multimodal system for long-term streaming video and audio interactions. arXiv preprint arXiv:2412.09596, 2024. 2   
[36] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025. 8, 12, 16   
[37] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun MA, Ziwei Liu, and Chunyuan Li. LLaVA-video: Video instruction tuning with synthetic data. Transactions on Machine Learning Research, 2025. 1

# WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning Supplementary Material

# A. Additional Details on Dataset

In this section, we provide additional details for each dataset used in our experiments. Tab. 5 summarizes the datasets, including the number of queries, domain categories, and the average video duration.

Table 5. Summary of benchmark datasets used in experiments.   

<table><tr><td>Dataset</td><td># Queries</td><td>Domain</td><td>Avg. Video Length</td></tr><tr><td>EgoLifeQA [33]</td><td>500</td><td>Egocentric</td><td>44.3h</td></tr><tr><td>Ego-R1 Bench [23]</td><td>300</td><td>Egocentric</td><td>44.3h</td></tr><tr><td>HippoVlog [12]</td><td>1,000</td><td>Vlog</td><td>0.45h</td></tr><tr><td>LVBench [25]</td><td>1,534</td><td>General</td><td>1.14h</td></tr><tr><td>Video-MME (L) [5]</td><td>900</td><td>General</td><td>0.69h</td></tr></table>

# A.1. EgoLifeQA

EgoLifeQA [33] is a set of questions designed to test the capability of models to understand and remember everyday life from week-long video recordings. It includes questions that require recalling past events, tracking object locations, and reasoning over long-term activities. In our experiments, we use questions from the perspective of a single participant (A1: JAKE), along with his corresponding video stream, which spans 44.3 hours. The benchmark is organized into five distinct categories as follows.

EntityLog (Ent.) Questions that require recalling information about objects, such as their locations, states, or interactions. (Example: “Who used the screwdriver first?”)

EventRecall (EvR.) Questions that ask about specific past events, including what happened, when it occurred, and relevant context. (Example: “Shure mentioned Tiramisu, when was the last time we discussed making Tiramisu?”)

HabitInsight (Hab.) Questions aimed at identifying a person’s recurring behaviors or long-term activity patterns. (Example: “What food does Alice love to eat?”)

RelationMap (Rel.) Questions involving understanding social relationships and interactions between people. (Example: “Who usually sings when Shure plays the guitar?”)

TaskMaster (Task) Questions focused on ongoing or pending tasks that require reasoning about what actions still need to be completed. (Example: “What are we planning to do in the afternoon?”)

# A.2. Ego-R1 Bench

Ego-R1 Bench [23] is designed as a complementary evaluation to EgoLifeQA, but with a distinct focus on model reasoning. While both benchmarks focus on the same week-long egocentric video, Ego-R1 Bench targets multistep, tool-augmented reasoning over ultra-long video. We reorganize query types of Ego-R1 Bench to the category adopted by EgoLifeQA, as shown in Tab. 6.

Table 6. Classification of queries under the EgoLifeQA category.   

<table><tr><td>Category</td><td>Ego-R1 Category</td></tr><tr><td>EntityLog</td><td>EntityLog, FoodLog, HealthLog, TechLog</td></tr><tr><td>EventRecall</td><td>EventRecall, Event Recollection, Event Memory</td></tr><tr><td>HabitInsight</td><td>HabitInsight, Behavior Habit(s)</td></tr><tr><td>RelationMap</td><td>RelationMap, Interpersonal Relationships</td></tr><tr><td>TaskMaster</td><td>TaskMaster, Future Plan(s)</td></tr></table>

# A.3. HippoVlog

HippoVlog [12] contains 25 daily vlog videos with 1,000 multiple-choice questions for continuous audiovisual event understanding. The benchmark evaluates a model’s ability to handle modality-specific information, with Auditory (Aud.) questions requiring reasoning over the audio stream (or transcript) and Visual (Vis.) questions focusing on the visual content. Auditory+Visual $( \mathbf { A } { + } \mathbf { V } )$ queries test the model’s ability to integrate information across both modalities, while Summarization (Summ.) questions assess higher-level reasoning over long temporal spans, requiring synthesis of events and semantic understanding from the continuous video.

# A.4. LVBench

LVBench [25] consists of 103 long videos, typically longer than an hour, with 1,549 multiple-choice questions for extreme long video understanding. The videos cover a general and diverse set of domains. Questions include both visual perception for recognizing entities or events in short segments and summarization for higher-level reasoning across extended sequences, evaluating models’ ability to integrate information over both local and long-horizon contexts. In our experiments, we categorize questions into three groups based on their segment length, defined as the duration of video required to answer the question: Short $( < 3 0 \mathrm { s } )$ , Medium (Med.) $( 3 0 \mathrm { s } \sim 5 \mathrm { m i n } )$ , and Long $( > 5 \mathrm { { m i n } ) }$ ). We excluded 15 questions without segment tags, leaving 1,534 questions in total for evaluation.

# A.5. Video-MME

Video-MME [5] is a comprehensive video understanding benchmark with 2,700 questions and varying video durations. In this experiment, we use only the long subset $( > 3 0 \mathrm { m i n } )$ , containing 900 questions, to assess the model’s capability on long video reasoning. We adopt the categories provided by the benchmark, with acronyms as follows: Action Reasoning (ARES), Action Recognition (AREC), Attribute Perception (ATTR), Counting Problem (CNT), Information Synopsis (ISYN), OCR Problems (OCR), Object Reasoning (ORES), Object Recognition (OREC), Spatial Perception (SPER), Spatial Reasoning (SRES), Temporal Perception (TPER), and Temporal Reasoning (TRES).

# B. Additional Implementation Details

We provide additional details on the baseline setup (Sec. B.1), the configuration of our proposed WorldMM (Sec. B.2), and the prompts used (Sec. B.3).

# B.1. Baseline Setup

Base Models & Long Video LLMs For all base models and long video LLMs, the video input is uniformly sampled at 0.5 fps and capped at 768 frames, since we cannot process all frames due to context limit, as mentioned in Sec. 1. For Time-R1 [28], we employ the 7B checkpoint1.

RAG-based Video LLMs For text-based RAG video models, we construct a knowledge base from video captions. Specifically, each video is segmented into 30 second chunks, and set of captions from these segments serve as retrieval pool. LightRAG [6] performs dual-level retrieval, selecting either fine-grained (low-level) or abstracted (high-level) information from the knowledge graph generated from set of captions depending on the query. HippoRAG [7], in contrast, retrieves raw captions ranked by their PPR scores, treating each caption as a separate document. For Video-RAG [14] model, retrieval is performed directly on the raw video using tools such as optical character recognition (OCR) and automatic speech recognition (ASR) to extract textual signals. Unless otherwise stated, we follow the retrieval specifications described in each model’s corresponding paper or implementation.

Memory-based Video LLMs Memory-based video LLMs construct explicit memories from the video stream. For EgoRAG [33] and Ego-R1 [23], which build hierarchical textual memories, we use the same temporal granularity applied when constructing WorldMM’s memory. For models that perform iterative reasoning, including Ego-R1 [23] and M3-Agent [13], we evaluate the checkpoints released

by authors and set the maximum number of reasoning iterations to 5 to ensure consistent evaluation across all systems. All other implementation details follow the official specifications provided by the respective authors.

# B.2. WorldMM

To construct multi-scale episodic memory, we tailor the temporal resolutions to each dataset’s duration. For Ego-LifeQA and Ego-R1 Bench, which contain week-long videos, we use four broad timescales: 30 seconds, 3 minutes, 10 minutes, and 1 hour. For HippoVlog, LVBench, and Video-MME, which contain shorter recordings averaging about an hour, we adopt shorter timescales of 10 seconds, 30 seconds, 3 minutes, and 10 minutes to better match their temporal structure. For semantic memory, triplets with a similarity score above 0.6 are consolidated using an LLM, and the top 10 triplets are retrieved at query time. The retrieval agent is limited to a maximum of five iterations, consistent with the baseline evaluation setting.

# B.3. Prompts

To construct and retrieve memory, and to generate the final response of WorldMM, we employ carefully optimized prompts for use with an LLM. In particular, we use prompts for episodic triple extraction (Figs. 9 and 10) and multiscale memory construction (Fig. 11), adapted from Yang et al. [33]. Furthermore, we utilize prompts for multiscale memory retrieval (Fig. 12), semantic triple extraction (Fig. 13), semantic consolidation (Fig. 14), iterative reasoning by the retrieval agent (Fig. 15), and final response generation (Fig. 16).

# C. Additional Description on Experiments

In this section, we provide additional description of the settings used in our ablation experiments.

# C.1. Dynamic Temporal Scope Retrieval (Sec. 4.4)

To evaluate performance on dynamic temporal reasoning with WorldMM, we employ several approaches, including temporal grounding model, embedding-based retrieval models, hierarchical retrieval models, and keyframe selection method. For each method, we measure tIoU using either the returned timestamps or the timestamps of the selected content. For the temporal grounding model, we use Time-R1 [28], with a slightly modified prompt that enables it to return both the evidence timestamps and the corresponding grounded responses. We sample videos at 0.5 fps and provide up to 768 frames. For embedding-based and hierarchical retrieval models, we follow the configurations described in Sec. B.1. Additionally, we include Qwen3 Emb., which applies the Qwen3-Embedding-4B [36] text encoder for caption retrieval, and InternVideo2, which encodes each segment using InternVideo2 [27] as an video en-

coder with uniform 16 frame averaging to enable segmentlevel retrieval. Both methods retrieve 30 second segments based on similarity search. For key frame selection, we apply AKS [22], which selects keyframes from the 0.5 fps sampled sequence. For tIoU evaluation, we interpret frames as representing their corresponding 30 second segments.

# C.2. Efficacy of Memory Modules (Sec. 4.7)

To assess the contribution of each component within WorldMM’s multimodal memory system, we evaluate several ablated variants in Sec. 4.7. In this section, we detail each variant of WorldMM created by selectively disabling a specific component. For episodic memory variants, we first construct a fixed timescale variant by replacing hierarchical episodic memory with a single fixed timescale memory. Specifically, we use the episodic memory of the finest granularity timescale. We also experiment an embedding retrieval variant in which the model’s graph-based episodic retrieval is replaced with an embedding-based similarity search using Qwen-Embedding-4B. To examine the effect of semantic consolidation, we use a w/o consolidation version that bypasses the consolidation procedure to update the memory and instead store the raw extracted triplets without any update to existing memory. Finally, for visual memory, we ablate components of dual-retrieval mechanism by evaluating systems that rely exclusively on either feature retrieval through natural-language keyword search or timestamp retrieval based purely on temporal indices.

# D. Detailed Experimental Results

Main results Tabs. 7 and 8 present the category-wise performance breakdown of WorldMM and baseline methods. Beyond overall benchmark averages, WorldMM consistently outperforms existing approaches across most categories. Notably, the gains are especially larger in categories that rely on visual information. For instance, in the EntityRecall category of EgoLifeQA, where visual cues can help answering, WorldMM exceeds the previous best method, Ego-R1, by a substantial $1 1 . 2 \%$ . Similarly, on HippoVlog, our model achieves a $4 \%$ improvement in the Aud. and $_ { \mathrm { A + V } }$ categories, both of which require visual reasoning. These margins are greater than those observed in categories that do not explicitly depend on visual content, highlighting the strong advantage of our multimodal multi-memory architecture.

Efficacy of multimodal memory Fig. 8 shows memory type utilization of our model on HippoVlog benchmark, where categories are grouped by their modality requirements. The Audio category requires reasoning over spoken content and therefore is expected to depend primarily on textual memory derived from caption transcripts, while the

Visual category focuses on visual understanding and correspondingly is designed to rely more on visual memory. Our results clearly support these expectations, showing that the Audio category predominantly activates textual memory while the Visual category relies heavily on visual memory, indicating that each category effectively leverages the required memory. Moreover, the Summarization category, which requires long-term reasoning, utilizes semantic memory more than any other category, demonstrating the complementary roles and effectiveness of each memory module in handling different reasoning demands. Together with this distribution of memory usage and the demonstrated performance gains in Tab. 2, these underscore the effectiveness of our multimodal multi-memory framework.

Dynamic temporal scope retrieval Tabs. 9 and 10 detail the per-category tIoU and accuracy results for WorldMM and baseline methods. While WorldMM significantly outperforms existing baselines on average, the results on LVBench particularly highlight the effectiveness of our dynamic episodic memory. In LVBench’s Long category, where answering requires reasoning over more than five minutes of video, WorldMM outperforms the baselines by a notably larger margin than in categories that require shorter timescale, underscoring its ability to flexibly retrieve and integrate information over diverse temporal spans.

# E. Qualitative Results

# E.1. Memory Construction

Tab. 11 presents an example of episodic triplet extraction. Given a caption generated from sampled frames of a segment along with its corresponding transcript, an LLM is prompted (using the prompt in Fig. 10) to extract episodic triplets. Semantic triplets are extracted using a different prompt (Fig. 13), designed to focus on long-term dependencies and capture more abstract relationships across the segments, as shown in Tab. 12. To better capture persistent knowledge across segments, we introduce semantic consolidation, which incrementally updates the semantic graph by integrating new triplets and resolving conflicts. Using embedding-based matching and an LLM, duplicated or conflicting triplets are removed, and new or revised ones are added, generating an evolving semantic memory, as shown in Tab. 13. For instance, the new triplet “[I, uses WeChat for, money transfers]” is merged with the existing triplet to consolidate redundant information, and conflicting triplets, such as “[Lucia, dislikes, overly sweet food]” versus “[Lucia, likes, sweet desserts]”, are removed to ensure consistency in the semantic memory.

# E.2. Multi-turn Refinement

WorldMM demonstrates the effectiveness of multi-turn reasoning by progressively refining its retrieval strategy to answer questions, as shown in Tab. 14. In this example, the first round retrieves episodic memory using a narrow keyword focused on the “discussion” of the air conditioning, but it provides insufficient detail about the activity. In the second round, the model expands to a more general keyword, “air conditioning”, which enables retrieval of every scene where the air conditioning is involved to obtain sufficient textual evidence. Moreover, in the third round, since the textual evidence fails to capture specific visual details of the scene, WorldMM refines its strategy to retrieve video frames corresponding to the relevant timestamp. Through this stepwise process, WorldMM effectively refines its search strategy with different keyword strategies and memory types to respond to the question.

# F. Limitation and Broader Impact

While WorldMM serves as an effective multimodal memory agent for long video reasoning, it still requires careful preprocessing, including video captioning, triplet extraction, and semantic consolidation. Yet, this limitation is not unique to our approach but a broader constraint shared by existing memory-based video LLMs. For example, M3- Agent [13] incurs even heavier preprocessing due to its reliance on entity recognition, and many other approaches operate with offline preprocessing. In contrast, WorldMM is designed for online operation. Memories are updated at fixed intervals (e.g., every 10 seconds), and the required preprocessing for each segment can be performed within these windows. Moreover, new information can be seamlessly integrated into the knowledge graph, and our consolidation mechanism efficiently refines the knowledge base without requiring the reconstruction of memory from scratch.

With strong long-term reasoning capabilities and support for real-time updates, WorldMM serves as a practical solution for streaming scenarios such as egocentric assistants and embodied agents. This foundation enables richer and more persistent assistance for everyday tasks and accessibility. However, the continuous accumulation of structured knowledge over periods of time raises serious privacy and security concerns. Real-world deployments must therefore enforce safeguard policies, including strict access controls, secure data handling, and privacy protections.

Table 7. Category-wise performance breakdown of WorldMM and baselines on EgoLifeQA, Ego-R1 Bench, HippoVlog, and LVBench.   

<table><tr><td rowspan="2">Model</td><td colspan="6">EgoLifeQA</td><td colspan="6">Ego-R1 Bench</td><td colspan="6">HippoVlog</td><td colspan="4">LVBench</td></tr><tr><td>Ent.</td><td>EvR.</td><td>Hab.</td><td>Rel.</td><td>Task</td><td>Avg.</td><td>Ent.</td><td>EvR.</td><td>Hab.</td><td>Rel.</td><td>Task</td><td>Avg.</td><td>Aud.</td><td>Vis.</td><td>A+V</td><td>Summ.</td><td>Avg.</td><td>Short</td><td>Med.</td><td>Long</td><td>Avg.</td><td></td></tr><tr><td colspan="22">Base Models</td><td></td></tr><tr><td>Qwen3-VL-8B [1]</td><td>35.2</td><td>30.2</td><td>39.3</td><td>46.4</td><td>46.0</td><td>38.6</td><td>31.8</td><td>41.5</td><td>38.5</td><td>42.1</td><td>44.7</td><td>35.7</td><td>73.6</td><td>74.0</td><td>69.2</td><td>80.8</td><td>74.4</td><td>48.8</td><td>44.4</td><td>53.4</td><td>48.3</td><td></td></tr><tr><td>Gemini 2.5 Pro [3]</td><td>43.2</td><td>40.5</td><td>41.0</td><td>55.2</td><td>52.4</td><td>46.4</td><td>43.9</td><td>56.1</td><td>53.9</td><td>47.4</td><td>47.4</td><td>46.7</td><td>69.2</td><td>75.2</td><td>63.6</td><td>80.0</td><td>72.0</td><td>57.1</td><td>52.2</td><td>65.2</td><td>57.0</td><td></td></tr><tr><td>GPT-5 [16]</td><td>47.2</td><td>42.1</td><td>47.5</td><td>53.6</td><td>55.6</td><td>48.6</td><td>41.8</td><td>58.5</td><td>53.9</td><td>52.6</td><td>50.0</td><td>46.3</td><td>73.6</td><td>75.6</td><td>69.2</td><td>84.4</td><td>75.7</td><td>59.1</td><td>59.1</td><td>69.1</td><td>60.4</td><td></td></tr><tr><td colspan="22">Long Video LLMs</td><td></td></tr><tr><td>VideoChat-Flash [11]</td><td>28.8</td><td>32.5</td><td>37.7</td><td>37.6</td><td>38.1</td><td>34.2</td><td>43.4</td><td>43.9</td><td>38.5</td><td>31.6</td><td>44.7</td><td>42.7</td><td>60.8</td><td>59.2</td><td>56.4</td><td>55.6</td><td>58.0</td><td>34.9</td><td>23.1</td><td>44.6</td><td>33.2</td><td></td></tr><tr><td>Time-R1 [28]</td><td>39.2</td><td>50.8</td><td>65.6</td><td>48.8</td><td>47.6</td><td>48.8</td><td>49.2</td><td>48.8</td><td>46.2</td><td>42.1</td><td>44.7</td><td>48.0</td><td>58.2</td><td>58.2</td><td>49.4</td><td>52.4</td><td>54.6</td><td>32.1</td><td>23.6</td><td>40.2</td><td>31.1</td><td></td></tr><tr><td>Video-RTS [29]</td><td>40.8</td><td>48.4</td><td>62.3</td><td>48.8</td><td>47.6</td><td>48.2</td><td>47.6</td><td>46.3</td><td>53.9</td><td>52.6</td><td>47.4</td><td>48.0</td><td>58.8</td><td>62.0</td><td>56.8</td><td>58.4</td><td>59.0</td><td>43.4</td><td>25.7</td><td>49.5</td><td>39.8</td><td></td></tr><tr><td colspan="22">RAG-based Video LLMs</td><td></td></tr><tr><td>LightRAG [6]</td><td>40.8</td><td>48.4</td><td>67.2</td><td>50.4</td><td>44.4</td><td>48.8</td><td>54.0</td><td>61.0</td><td>46.2</td><td>42.1</td><td>42.1</td><td>52.3</td><td>51.6</td><td>46.0</td><td>44.8</td><td>47.2</td><td>47.4</td><td>30.2</td><td>28.6</td><td>34.3</td><td>30.4</td><td></td></tr><tr><td>HippoRAG [7]</td><td>48.8</td><td>60.3</td><td>70.5</td><td>60.8</td><td>66.7</td><td>59.6</td><td>54.5</td><td>65.9</td><td>69.2</td><td>52.6</td><td>50.0</td><td>56.0</td><td>72.4</td><td>53.2</td><td>54.0</td><td>73.2</td><td>63.2</td><td>54.9</td><td>47.5</td><td>62.3</td><td>54.0</td><td></td></tr><tr><td>Video-RAG [14]</td><td>49.6</td><td>56.3</td><td>67.2</td><td>55.2</td><td>54.0</td><td>55.4</td><td>48.7</td><td>58.5</td><td>53.9</td><td>47.4</td><td>44.7</td><td>49.7</td><td>63.2</td><td>64.8</td><td>63.6</td><td>68.8</td><td>65.1</td><td>32.9</td><td>30.2</td><td>39.7</td><td>33.1</td><td></td></tr><tr><td colspan="22">Memory-based Video LLMs</td><td></td></tr><tr><td>EgoRAG [33]</td><td>40.0</td><td>56.3</td><td>62.3</td><td>54.4</td><td>52.4</td><td>52.0</td><td>46.6</td><td>56.1</td><td>46.2</td><td>47.4</td><td>55.3</td><td>49.0</td><td>64.8</td><td>53.2</td><td>47.6</td><td>64.4</td><td>57.5</td><td>32.4</td><td>32.0</td><td>31.9</td><td>32.2</td><td></td></tr><tr><td>Ego-R1 [23]</td><td>51.2</td><td>53.2</td><td>63.9</td><td>50.4</td><td>50.8</td><td>53.0</td><td>50.8</td><td>63.4</td><td>38.5</td><td>36.8</td><td>57.9</td><td>52.0</td><td>57.2</td><td>58.8</td><td>52.0</td><td>67.2</td><td>58.8</td><td>32.5</td><td>36.5</td><td>37.3</td><td>34.1</td><td></td></tr><tr><td>HippoMM [12]</td><td>45.6</td><td>53.2</td><td>70.5</td><td>55.2</td><td>58.7</td><td>54.6</td><td>51.9</td><td>56.1</td><td>46.2</td><td>52.6</td><td>57.9</td><td>53.0</td><td>68.8</td><td>77.6</td><td>59.2</td><td>82.0</td><td>71.9</td><td>40.7</td><td>33.3</td><td>35.8</td><td>38.2</td><td></td></tr><tr><td>M3-Agent [13]</td><td>44.4</td><td>54.8</td><td>62.3</td><td>56.8</td><td>54.0</td><td>53.5</td><td>52.4</td><td>58.5</td><td>38.5</td><td>42.1</td><td>52.6</td><td>52.0</td><td>68.4</td><td>72.4</td><td>50.8</td><td>70.4</td><td>65.5</td><td>53.0</td><td>40.7</td><td>48.5</td><td>49.3</td><td></td></tr><tr><td colspan="22">WorldMM (Ours)</td><td></td></tr><tr><td>WorldMM-8B</td><td>49.6</td><td>56.4</td><td>63.9</td><td>58.4</td><td>58.7</td><td>56.4</td><td>48.2</td><td>63.4</td><td>53.9</td><td>52.6</td><td>57.9</td><td>52.0</td><td>69.6</td><td>73.6</td><td>65.2</td><td>70.4</td><td>69.7</td><td>55.0</td><td>54.1</td><td>59.8</td><td>55.4</td><td></td></tr><tr><td>WorldMM-GPT</td><td>62.4</td><td>64.3</td><td>75.4</td><td>62.4</td><td>71.4</td><td>65.6</td><td>64.6</td><td>70.7</td><td>76.9</td><td>57.9</td><td>63.2</td><td>65.3</td><td>75.6</td><td>81.6</td><td>73.2</td><td>82.8</td><td>78.3</td><td>58.3</td><td>65.4</td><td>72.1</td><td>61.9</td><td></td></tr></table>

Table 8. Category-wise performance breakdown of WorldMM and baselines on Video-MME (L).   

<table><tr><td>Model</td><td>ARES</td><td>AREC</td><td>ATTR</td><td>CNT</td><td>ISYN</td><td>OCR</td><td>ORES</td><td>OREC</td><td>SPER</td><td>SRES</td><td>TPER</td><td>TRES</td><td>Avg.</td></tr><tr><td colspan="14">Base Models</td></tr><tr><td>Qwen3-VL-8B [1]</td><td>62.2</td><td>54.0</td><td>51.9</td><td>43.8</td><td>68.1</td><td>42.9</td><td>62.9</td><td>57.4</td><td>33.3</td><td>45.5</td><td>33.3</td><td>67.0</td><td>61.0</td></tr><tr><td>Gemini 2.5 Pro [3]</td><td>56.9</td><td>47.6</td><td>66.7</td><td>41.7</td><td>71.8</td><td>57.1</td><td>53.3</td><td>40.7</td><td>0.0</td><td>72.7</td><td>66.7</td><td>48.4</td><td>55.7</td></tr><tr><td>GPT-5 [16]</td><td>71.1</td><td>69.8</td><td>70.4</td><td>47.9</td><td>88.3</td><td>57.1</td><td>75.8</td><td>74.1</td><td>33.3</td><td>72.7</td><td>50.0</td><td>75.8</td><td>74.3</td></tr><tr><td colspan="14">Long Video LLMs</td></tr><tr><td>VideoChat-Flash [11]</td><td>35.0</td><td>42.9</td><td>37.0</td><td>31.3</td><td>34.4</td><td>42.9</td><td>60.0</td><td>46.3</td><td>33.3</td><td>54.5</td><td>33.3</td><td>46.2</td><td>44.1</td></tr><tr><td>Time-R1 [28]</td><td>20.6</td><td>28.6</td><td>25.9</td><td>35.4</td><td>31.9</td><td>35.7</td><td>53.3</td><td>48.2</td><td>33.3</td><td>36.4</td><td>50.0</td><td>44.0</td><td>37.6</td></tr><tr><td>Video-RTS [29]</td><td>43.3</td><td>52.4</td><td>40.7</td><td>39.6</td><td>33.7</td><td>42.9</td><td>60.8</td><td>53.7</td><td>33.3</td><td>45.5</td><td>50.0</td><td>49.5</td><td>47.9</td></tr><tr><td colspan="14">RAG-based Video LLMs</td></tr><tr><td>LightRAG [6]</td><td>41.7</td><td>30.2</td><td>40.7</td><td>35.4</td><td>54.0</td><td>50.0</td><td>46.7</td><td>61.1</td><td>33.3</td><td>45.5</td><td>50.0</td><td>52.8</td><td>46.6</td></tr><tr><td>HippoRAG [7]</td><td>45.6</td><td>47.6</td><td>40.7</td><td>37.5</td><td>52.2</td><td>42.9</td><td>52.9</td><td>64.8</td><td>66.7</td><td>54.5</td><td>50.0</td><td>70.3</td><td>52.1</td></tr><tr><td>Video-RAG [14]</td><td>51.7</td><td>47.6</td><td>37.0</td><td>39.6</td><td>49.7</td><td>57.1</td><td>62.1</td><td>68.5</td><td>66.7</td><td>45.5</td><td>50.0</td><td>68.1</td><td>55.4</td></tr><tr><td colspan="14">Memory-based Video LLMs</td></tr><tr><td>EgoRAG [33]</td><td>31.1</td><td>55.6</td><td>33.3</td><td>22.9</td><td>41.1</td><td>28.6</td><td>44.6</td><td>48.2</td><td>33.3</td><td>54.5</td><td>66.7</td><td>48.4</td><td>41.1</td></tr><tr><td>Ego-R1 [23]</td><td>37.2</td><td>52.4</td><td>40.7</td><td>35.4</td><td>38.0</td><td>35.7</td><td>42.1</td><td>51.9</td><td>66.7</td><td>63.6</td><td>50.0</td><td>52.8</td><td>42.7</td></tr><tr><td>HippoMM [12]</td><td>41.1</td><td>42.9</td><td>55.6</td><td>35.4</td><td>38.7</td><td>35.7</td><td>37.9</td><td>53.7</td><td>33.3</td><td>54.5</td><td>50.0</td><td>47.3</td><td>41.6</td></tr><tr><td>M3-Agent [13]</td><td>52.2</td><td>57.1</td><td>59.3</td><td>45.8</td><td>51.5</td><td>42.9</td><td>54.6</td><td>64.8</td><td>33.3</td><td>45.5</td><td>50.0</td><td>71.4</td><td>55.3</td></tr><tr><td colspan="14">WorldMM (Ours)</td></tr><tr><td>WorldMM-8B</td><td>65.0</td><td>66.7</td><td>59.3</td><td>41.7</td><td>72.4</td><td>42.9</td><td>67.5</td><td>72.2</td><td>33.3</td><td>54.5</td><td>66.7</td><td>69.2</td><td>66.0</td></tr><tr><td>WorldMM-GPT</td><td>81.1</td><td>73.0</td><td>70.4</td><td>54.2</td><td>85.3</td><td>42.9</td><td>75.0</td><td>77.8</td><td>33.3</td><td>72.7</td><td>66.7</td><td>79.1</td><td>76.6</td></tr></table>

![](images/ee9bde65ee7a7e81e599e55095d4941c35b4f2bfdac59b973b3f5ff46f224e6d.jpg)  
Figure 8. Memory type utilization of WorldMM on four distinctive categories in HippoVlog.

Table 9. Category-wise average tIoU $( \% )$ ) breakdown of WorldMM and dynamic temporal scope retrieval baselines.   

<table><tr><td rowspan="2">Model</td><td colspan="6">EgoLifeQA</td><td colspan="6">Ego-R1 Bench</td><td colspan="4">LVBench</td></tr><tr><td>Ent.</td><td>EvR.</td><td>Hab.</td><td>Rel.</td><td>Task</td><td>Avg.</td><td>Ent.</td><td>EvR.</td><td>Hab.</td><td>Rel.</td><td>Task</td><td>Avg.</td><td>Short</td><td>Med.</td><td>Long</td><td>Avg.</td></tr><tr><td>Time-R1 [28]</td><td>0.34</td><td>0.72</td><td>1.07</td><td>0.52</td><td>0.41</td><td>0.58</td><td>0.27</td><td>0.84</td><td>0.71</td><td>1.15</td><td>1.58</td><td>0.59</td><td>3.10</td><td>2.60</td><td>1.00</td><td>2.70</td></tr><tr><td>Qwen3 Emb. [36]</td><td>2.87</td><td>4.31</td><td>5.58</td><td>2.98</td><td>8.91</td><td>4.35</td><td>2.68</td><td>2.74</td><td>3.85</td><td>2.74</td><td>3.70</td><td>2.87</td><td>4.48</td><td>6.20</td><td>1.75</td><td>4.54</td></tr><tr><td>HippoRAG [7]</td><td>3.02</td><td>4.19</td><td>4.99</td><td>2.12</td><td>8.36</td><td>4.00</td><td>3.32</td><td>2.85</td><td>3.28</td><td>2.23</td><td>4.07</td><td>3.28</td><td>4.23</td><td>5.76</td><td>1.88</td><td>4.30</td></tr><tr><td>InternVideo2 [27]</td><td>2.09</td><td>4.42</td><td>6.04</td><td>2.00</td><td>3.88</td><td>3.36</td><td>2.71</td><td>2.55</td><td>3.09</td><td>1.85</td><td>2.32</td><td>2.60</td><td>3.66</td><td>4.71</td><td>0.87</td><td>3.55</td></tr><tr><td>EgoRAG [33]</td><td>3.20</td><td>3.38</td><td>4.62</td><td>3.10</td><td>4.82</td><td>3.60</td><td>2.40</td><td>3.07</td><td>4.08</td><td>2.19</td><td>3.78</td><td>2.73</td><td>4.10</td><td>3.38</td><td>0.91</td><td>3.50</td></tr><tr><td>Ego-R1 [23]</td><td>3.31</td><td>3.52</td><td>5.03</td><td>2.87</td><td>5.18</td><td>3.70</td><td>2.57</td><td>2.83</td><td>4.13</td><td>2.83</td><td>4.12</td><td>2.89</td><td>4.08</td><td>3.72</td><td>1.14</td><td>3.60</td></tr><tr><td>AKS [22]</td><td>2.42</td><td>2.77</td><td>3.08</td><td>2.93</td><td>2.67</td><td>2.75</td><td>2.03</td><td>2.48</td><td>2.99</td><td>2.58</td><td>3.04</td><td>2.30</td><td>3.81</td><td>4.11</td><td>1.10</td><td>3.52</td></tr><tr><td>WorldMM (Ours)</td><td>9.79</td><td>10.43</td><td>11.85</td><td>7.73</td><td>12.97</td><td>10.09</td><td>8.91</td><td>9.85</td><td>8.86</td><td>9.63</td><td>9.58</td><td>9.17</td><td>7.53</td><td>14.41</td><td>10.02</td><td>9.57</td></tr></table>

Table 10. Category-wise performance breakdown of WorldMM and dynamic temporal scope retrieval baselines.   

<table><tr><td rowspan="2">Model</td><td colspan="6">EgoLifeQA</td><td colspan="6">Ego-R1 Bench</td><td colspan="4">LVBench</td></tr><tr><td>Ent.</td><td>EvR.</td><td>Hab.</td><td>Rel.</td><td>Task</td><td>Avg.</td><td>Ent.</td><td>EvR.</td><td>Hab.</td><td>Rel.</td><td>Task</td><td>Avg.</td><td>Short</td><td>Med.</td><td>Long</td><td>Avg.</td></tr><tr><td>Time-R1 [28]</td><td>39.2</td><td>50.8</td><td>65.6</td><td>48.8</td><td>47.6</td><td>48.8</td><td>49.2</td><td>48.8</td><td>46.2</td><td>42.1</td><td>44.7</td><td>48.0</td><td>32.1</td><td>23.6</td><td>40.2</td><td>31.1</td></tr><tr><td>Qwen3 Emb. [36]</td><td>44.0</td><td>59.5</td><td>70.5</td><td>58.4</td><td>68.3</td><td>57.8</td><td>51.9</td><td>65.9</td><td>61.5</td><td>57.9</td><td>47.4</td><td>54.0</td><td>52.9</td><td>49.1</td><td>62.3</td><td>53.2</td></tr><tr><td>HippoRAG [7]</td><td>48.8</td><td>60.3</td><td>70.5</td><td>60.8</td><td>66.7</td><td>59.6</td><td>54.5</td><td>65.9</td><td>69.2</td><td>52.6</td><td>50.0</td><td>56.0</td><td>54.9</td><td>47.5</td><td>62.3</td><td>54.0</td></tr><tr><td>InternVideo2 [27]</td><td>40.8</td><td>54.0</td><td>60.7</td><td>51.2</td><td>52.4</td><td>50.6</td><td>50.3</td><td>56.1</td><td>46.2</td><td>47.4</td><td>52.6</td><td>51.0</td><td>47.4</td><td>37.3</td><td>53.4</td><td>45.7</td></tr><tr><td>EgoRAG [33]</td><td>40.0</td><td>56.3</td><td>62.3</td><td>54.4</td><td>52.4</td><td>52.0</td><td>46.6</td><td>56.1</td><td>46.2</td><td>47.4</td><td>55.3</td><td>49.0</td><td>32.4</td><td>32.0</td><td>31.9</td><td>32.2</td></tr><tr><td>Ego-R1 [23]</td><td>51.2</td><td>53.2</td><td>63.9</td><td>50.4</td><td>50.8</td><td>53.0</td><td>50.8</td><td>63.4</td><td>38.5</td><td>36.8</td><td>57.9</td><td>52.0</td><td>32.5</td><td>36.5</td><td>37.3</td><td>34.1</td></tr><tr><td>AKS [22]</td><td>41.6</td><td>51.6</td><td>63.9</td><td>51.2</td><td>52.4</td><td>50.6</td><td>51.3</td><td>63.4</td><td>46.2</td><td>36.8</td><td>50.0</td><td>51.7</td><td>43.3</td><td>33.9</td><td>39.2</td><td>40.4</td></tr><tr><td>WorldMM (Ours)</td><td>62.4</td><td>64.3</td><td>75.4</td><td>62.4</td><td>71.4</td><td>65.6</td><td>64.6</td><td>70.7</td><td>76.9</td><td>57.9</td><td>63.2</td><td>65.3</td><td>58.3</td><td>65.4</td><td>72.1</td><td>61.9</td></tr></table>

Table 11. Example of episodic triplet extraction.   

<table><tr><td>Caption</td><td>I stand and walk to the other side of the dining table. Katrina asks, “Is this for tomorrow’s game?” “Yes—let’s think about what to do tomorrow,” I say. I raise my right hand as Katrina walks toward me. Lucia asks, “Using ancient poems? Or what else?” Katrina says, “I’m not good with ancient poems.” Tasha asks, “Then what else to use?” Katrina says, “I’ll be out in the first round. My room is already cleaned up.” “Okay,” I say. I turn toward the stairs, put down my phone, look back at the living room door, and walk into the second-floor living room. Lucia adds, “For example, not coming out.” Katrina says, “Let me check that place we’re going to.” Tasha asks, “I just want to ask which fields it has expanded into.” Lucia says, “Okay.”</td></tr><tr><td>Extracted Triplets</td><td>[I, stand at, dining table] [I, walk to, other side of the dining table] [Katrina, asks about, tomorrow] [I, confirm, tomorrow] [I, raise, right hand] [Katrina, walks toward, I] [Lucia, asks about, using ancient poems] [Katrina, says, not good with ancient poems] [Tasha, asks, what else to use] [Katrina, says, I will be out in the first round] [Katrina, has, room already cleaned up] [I, turn toward, stairs] [I, put down, phone] [I, look back at, living room door] [I, walk into, second-floor living room] [Lucia, adds, not coming out as an example] [Katrina, says, let me check that place we’re going to] [Lucia, says, Okay]</td></tr></table>

Table 12. Example of semantic triplet extraction.   

<table><tr><td>Caption</td><td>I got up, moved my phone, and checked it before turning it off. Alice expressed her feelings towards me, and I responded by checking my phone&#x27;s chat interface. Alice then questioned her appearance, and I turned off the phone, looking around at the snacks and utensils on the table. I stood up, grabbed a pack of snacks, and proceeded to my room to enjoy them. Alice asked about something being fancy, and I fetched my glasses, placing them on the table. ... I managed my phone, swiping through pages, and interacted with others as I went about my tasks. I observed Alice and Tasha, discussing what to feed a cat, and continued interacting with my phone. As the environment darkened, I engaged with the surroundings, noting the layout and structures. Finally, I moved towards a house with blue-green walls, managing my power bank and surveying the area.</td></tr><tr><td>Extracted Triplets</td><td>[I, assigns tasks to, Katrina][I, handles reimbursements for, Alice][I, uses WeChat for, money transfers][I, often eats, snacks][I, wears, glasses][Lucia, dislikes, overly sweet food][Alice, expresses romantic feelings toward, I][Katrina, helps with, expense tracking][I, requires PDFs for, reimbursement][Tasha, participates in, house demolition tasks][Lucia, participates in, house demolition tasks]</td></tr></table>

Table 13. Example of semantic consolidation.   

<table><tr><td>Original Triplets</td><td>[I, uses WeChat to send money] 
[I, wears, glasses] 
[I, often eats, fruits] 
[Lucia, likes, sweet desserts] 
[Tasha, participates in, household projects]</td><td></td></tr><tr><td>New Triplets</td><td>[I, assigns tasks to, Katrina] 
[I, handles reimbursements for, Alice] 
[I, uses WeChat for, money transfers] 
[I, often eats, snacks] 
[I, wears, glasses] 
[Lucia, dislikes, overly sweet food] 
[Alice, expresses romantic feelings toward, I] 
[Katrina, helps with, expense tracking] 
[I, requires PDFs for, reimbursement] 
[Tasha, participates in, house demolition tasks] 
[Lucia, participates in, house demolition tasks]</td><td>% conflicts with existing “likes sweet desserts”</td></tr><tr><td>Consolidated Triplets</td><td>[I, assigns tasks to, Katrina] 
[I, handles reimbursements for, Alice] 
[I, uses, WeChat to send money] 
[I, often eats, fruits and snacks] 
[I, wears, glasses] 
[Alice, expresses romantic feelings toward, I] 
[Katrina, helps with, expense tracking] 
[I, requires PDFs for, reimbursement] 
[Tasha, participates in, house demolition tasks] 
[Lucia, participates in, house demolition tasks]</td><td>% merged with “uses WeChat for money transfers” 
% merged and generalized “fruits” and “snacks” 
% merged with duplicate in new triples 
% refined general triple “household projects”</td></tr></table>

Table 14. Example of the multi-turn refinement of WorldMM.   

<table><tr><td>Question</td><td>What were we doing last time we discussed the air conditioning temperature?
(A) Eating hot pot (B) Shopping (C) Playing piano (D) Picking up a package</td></tr><tr><td>Answer</td><td>A</td></tr><tr><td>Round 1</td><td>Decision: Search // Memory: Episodic
Search Query: discussing the air conditioning temperature
Retrieved:
[DAY2 13:36:00-13:39:00]
I joined my colleagues at the dining table and engaged in a discussion about the food while preparing to eat. ... During the meal, we discussed the air conditioner situation in the rooms.
Tasha and Katrina talked about where the remotes were and who had access, with Shure mentioning that his room&#x27;s AC had been on continuously due to a borrowed remote from me. I listened to their conversation, ...
[DAY2 17:48:00-17:51:00]
I coordinated with Shure and Lucia to find a suitable spot for my power bank, initially placing it on a flower pot as Shure suggested, but realizing it was unstable. ... We discussed the room environment, including turning off the air conditioner that was blowing cold air. Shure clarified that the remote we had was for the TV, not the air conditioner, prompting me to inquire about the correct remote. ...</td></tr><tr><td>Round 2</td><td>Decision: Search // Memory: Episodic
Search Query: air conditioning
Retrieved:
[DAY2 17:48:30-17:49:00]
“Flip it and put it underneath,” I say. ... “Wow, this, this...” Lucia murmurs. “How do you turn off this air conditioner? It’s blowing so cold up here.” I adjust the power bank for a better position. Shure says, “Just tell it to turn off, it’ll turn off.” “Really? It’s voice-controlled?” Lucia asks. I secure the power bank and turn to Shure and Lucia. Shure adds, “No, this is the TV remote.” “Where is it?” I ask, and I chat with them.
[DAY2 18:34:01-18:34:29]
I watch Shure set the air conditioner to 26 degrees. “Save the Earth,” I say. “Everyone’s responsibility,” I laugh. I kneel, then sit on the sofa. Shure asks, “Seriously, you think it’s good to record this?” I look at my friends chatting and watch the movie. Shure says, “We’re still using the AC.” Shure says, “He really can’t hear a thing.” I watch Shure take a few steps. I laugh. Shure says, “Hahaha, it’s like we’re watching a silent movie.” I smile and keep watching the movie. “Whoa, this is classic,” I say. I stand up. “Come on, let’s have a drink,” I say.
[DAY2 18:33:00-18:33:30]
I walk in front of the fill light, pass behind Choiszt, and move toward the curtain. ... “The popularity is always high,” I say. Shure: “Ha.” Choiszt: “Turn it on.” “Turn on the air conditioner?” I ask. Lucia: “I’ll just, I suggest I drink water directly from a bowl.” Lucia: “So later I’ll eat with this bowl.”</td></tr><tr><td>Round 3</td><td>Decision: Search // Memory: Visual
Search Query: DAY2 18:34:01-18:34:29
Retrieved:
Author Note: They are gathered around a table cooking and sharing a hot pot meal while watching a projection on the wall.)</td></tr><tr><td>Round 4</td><td>Decision: Answer</td></tr><tr><td>Response</td><td>A</td></tr></table>

Your task is to extract named entities from the given paragraph. Respond with a JSON list of entities.

Example:

Radio City is India’s first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.

```json
{"namedentities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]} 
```

Figure 9. Prompt for named entity recognition (NER). Recognized named entities are used to extract episodic triplets as shown in Fig. 10.

Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists. Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.

Pay attention to the following requirements:

• Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.   
• When resolving pronouns, if the pronoun refers to the first-person (e.g., I, me, my), keep it as “I” instead of replacing with terms like “speaker” or “narrator”. For other pronouns, clearly resolve them to their specific names to maintain clarity.

Convert the paragraph into a JSON dict, it has a named entity list and a triple list.

Example:

Radio City is India’s first private FM radio station and was started on 3 July 2001. It plays Hindi, English and regional songs. Radio City recently forayed into New Media in May 2008 with the launch of a music portal - PlanetRadiocity.com that offers music related news, videos, songs, and other music-related features.

```jsonl
{ "named Entities": ["Radio City", "India", "3 July 2001", "Hindi", "English", "May 2008", "PlanetRadiocity.com"]   
}   
{ "triples": [ ["Radio City", "located in", "India"], ["Radio City", "is", "private FM radio station"], ["Radio City", "started on", "3 July 2001"], ["Radio City", "plays songs in", "Hindi"], ["Radio City", "plays songs in", "English"], ["Radio City", "forayed into", "New Media"], ["Radio City", "launched", "PlanetRadiocity.com"], ["PlanetRadiocity.com", "launched in", "May 2008"], ["PlanetRadiocity.com", "is", "music portal"], ["PlanetRadiocity.com", "offers", "news"], ["PlanetRadiocity.com", "offers", "videos"], ["PlanetRadiocity.com", "offers", "songs"]   
]} 
```

Figure 10. Prompt for episodic triplet extraction.

As an Event Summary Documentation Specialist, your role is to systematically structure and summarize event information, ensuring that all key actions of major characters are captured while maintaining clear event logic and completeness. Your focus is on concise and factual summarization rather than detailed transcription.

# # Specific Requirements

1. Structure the Events Clearly

- Merge related events: Consolidate similar content into major events and arrange them in chronological order to ensure a smooth logical flow.

- Logical segmentation: Events can be grouped based on location, task, or theme. Each event should have a clear starting point, progression, and key turning points without any jumps or fragmentation in the information.

2. Retain Key Information

- The primary character’s (“I”) decisions and actions must be fully presented, including all critical first-person activities. Transitions between different parts, such as moving between floors or starting/ending a task, should be seamless.

- Any discussions, decisions, and task execution involving the primary character and other key individuals that impact the main storyline must be reflected. This includes recording, planning, and confirming matters, but in a concise manner.

- The purpose and method of key actions must be recorded, such as “ordering takeout using a phone” or “documenting a plan on a whiteboard.”

3. Concise Expression, Remove Redundancies

- Keep the facts clear, avoiding descriptions of atmosphere, emotions, or abstract content.

- Remove trivial conversations and extract only the core topics and conclusions of discussions. If a discussion is lengthy, summarize it into task arrangements, decision points, and specific execution details.

4. Strictly Adhere to Facts, No Assumptions

- Do not make assumptions or add interpretations—strictly organize content based on available information, ensuring accuracy. Every summarized point must have a basis in the original information, with no unnecessary additions.

- Maintain the correct chronological order of events. The sequence of developments must strictly follow their actual occurrence without any inconsistencies.

# Output Format

Each paragraph should represent one major event, structured in a summary-detail-summary format. Strictly output below 500 words in total. Do not report the word count in the output.

Figure 11. Prompt for episodic memory construction to generate coarser-level caption.

You are an expert assistant that helps filter and select relevant video captions based on a given query. Your task is to analyze the retrieved video captions and determine which ones are most relevant to answer the question.

Given the following question and retrieved video captions, select and rank the most relevant captions that should be used to answer the question.

# Instructions:

1. Consider the nature of the question when selecting captions:   
- e.g., for queries about specific events, focus on finer granularities; for habitual, relationship, or general queries, consider coarser granularities.   
- Note that coarser granularity captions may provide broader context, but finer granularity captions often contain more specific details.   
2. Each caption shows its time range (start time to end time)   
3. Analyze each caption for relevance to the question   
4. Select captions that directly help answer the question   
5. Return the IDs in ranked order (most relevant first)   
6. Only include captions that are truly relevant

Return ONLY a JSON array of caption IDs in order of relevance (most relevant first), without additional justification.

Figure 12. Prompt for episodic memory retrieval to select from multiple timescales.

You are tasked with extracting semantic knowledge from episodic triples. Your goal is to infer generalizable information that extends beyond the specific episode. Focus on capturing valid semantic triples that can guide reasoning about behavior, relationships, or preferences.

# # What to Extract

1. Relationships: social bonds or roles between entities that persist over time (e.g., “Alice is a friend with Bob”, “Jason is a teacher of Alice”).   
2. Attributes & Preferences: tendencies, likes/dislikes, personality-like traits, or behavioral habits (e.g., “Alice prefers not having dessert”, “Bob enjoys music”).   
3. Habits & Capabilities: actions or patterns that suggest what an entity often does, can do, or tends to do (e.g., “Alice often helps friends”, “Jason can give advice”).   
4. Conceptual Knowledge: directly useful facts that support reasoning, but avoid overly broad taxonomic statements (e.g., “Alice’s office is near Cafe X”, “Bob’s gym is closed on Sundays”).

# # What to Avoid

- One-off events or transient states (e.g., “ate pizza yesterday”, “was late once”) unless explicitly declared as a preference/role   
- Broad taxonomy or trivia unrelated to behavior (e.g., “a laptop is electronics”, “Paris is in France”)   
- Speculative or mind-reading inferences without textual support (e.g., motives, beliefs not evidenced)

# # Important Notes

- Prefer to base semantic triples on multiple supporting episodes.   
- BUT if a single episode clearly reflects a role, preference, habit, or capability, it is valid to include it.   
- Each semantic triple MUST have at least one supporting episodic triple.   
- Reduce duplication. If multiple episodic triples support the same or very similar semantic knowledge, merge them into one semantic triple rather than repeating.   
- The ‘episodic evidence[i]’ list must always point to the indices that support ‘semantic triples[i]’.   
- Aim for broad coverage: extract as many valid semantic triples as reasonably supported by the input.

# # Output Format

- Return ONLY a JSON object with the following two keys:   
- ‘semantic triples’ (List[List[str]]): Each item is a triple [subject, predicate, object].   
- ‘episodic evidence’ (List[List[int]]) : Each item is a list of 0-based indices pointing to the input episodic triples that support the corresponding semantic triple at the same position.   
- The two lists MUST have the same length and aligned order.   
- If no semantic knowledge is inferable, return: {“semantic triples”: [], “episodic evidence”: []}

# Example:

# Episodic triples:

0. [“Alice”, “talks to”, “Bob”],   
1. [“Alice”, “laughs with”, “Bob”],   
2. [“Alice”, “doesn’t eat cake”, “at restaurant”],   
3. [“Alice”, “shares personal stories with”, “Bob”],   
4. [“Alice”, “brings coffee to”, “Bob”],   
5. [“Jason”, “talks to”, “Alice”],   
6. [“Alice”, “declines dessert”, “at friend’s house”]

```txt
Output:   
{ "semantic_triples":[ ["Alice", "is a friend with", "Bob"], ["Alice", "prefers", "not having dessert"] ], "episodic_evidence": [ [0,1,3], [2,6] ]   
} 
```

Figure 13. Prompt for semantic triplet extraction.

You are tasked with consolidating semantic knowledge by processing a new semantic triple against relevant existing knowledge from previous timestamps.

Your job is to make two decisions:

1. Which existing triples to remove/pop — those that should be merged with the new triple or conflict with it   
2. How to update the new triple — to capture merged information or resolve conflicts

# Consolidation Rules

1. Merge Similar Information: If existing triples express very similar information to the new triple, remove them and update the new triple to contain the most complete/accurate form.   
2. Resolve Conflicts: If the new triple conflicts with existing ones, decide which is more accurate/recent and remove outdated ones.   
3. Update with Context: Use information from existing triples to make the new triple more specific or more accurate.   
4. Preserve Unique Information: Only remove existing triples when they are redundant or conflicting.

# Output Format

Return ONLY a JSON object with the following two keys:

- ‘updated triple’ (List[str]): The new triple, possibly updated [subject, predicate, object].   
- ‘triples to remove’ (List[int]): Indices of existing triples to remove (empty list if none).

Example:

New triple: [“Alice”, “enjoys”, “coffee”]

Existing triples:

0. [“Alice”, “likes”, “beverages”]   
1. [“Alice”, “favors”, “to have coffee after dinner”]   
2. [“Alice”, “prefers”, “hot drinks”]   
3. [“Alice”, “likes to drink”, “coffee”]

```txt
Output:  
{ "updated triple": ["Alice", "likes", "coffee"], "triples_to_remove": [1, 3] } 
```

Figure 14. Prompt for semantic memory consolidation.

You are a reasoning agent for a video memory retrieval system. Your job is to decide whether to stop and answer, or to search memory for more evidence. When searching, you must select exactly one memory type and form a query.

# # Decision Modes

1. search: Retrieve memory to begin, continue, or extend progress toward the answer   
- Choose one memory type and form a keyword(phrase)-style search query.   
2. answer: Stop searching because the accumulated results are sufficient.   
- No memory type selection is needed.

# # Memory Types

1. Episodic: Specific events/actions. Stores memories of past events and actions. Query by EVENT/ACTION.   
2. Semantic: Entities/relationships. Stores factual knowledge about entities and their relationships, roles, and habits. Query by ENTITY/CONCEPT.   
3. Visual: Scene/setting snapshots. Stores visual snapshots of scenes and settings. Query by SCENE/SETTING or TIMESTAMP RANGE.

- For timestamp range queries, return in the format: DAY X HH:MM:SS - DAY Y HH:MM:SS.

# # Context Inputs

- Current Query   
- Round History: Log of past retrieval rounds. Each round is written in this format:

### Round N

Decision: <search|answer>

Memory: <episodic|semantic|visual>

Search Query: <query text>

Retrieved: <retrieved items>

# # Strict Output Rules

- If decision $=$ “search”: Must include “selected memory” with exactly one memory type and one query.   
- If decision $=$ “answer”: Do NOT include “selected memory”.   
- Always output in valid JSON only, no extra commentary.

# # Output Format

```json
{ "decision": "search" | "answer", "selected_memory": { "memory_type": "episodic" | "semantic" | "visual", "search_query": <str> } # Omit if decision = "answer"   
} 
```

(Few-shot examples given)

Figure 15. Prompt for retrieval agent to decide retrieval strategy.

You are an AI assistant that answers questions about egocentric video experiences using retrieved memory context. Your task is to answer multiple choice questions based on this accumulated context. Always choose the most relevant answer from the given choices based on the evidence provided.

# # Guidelines

- Analyze all provided context carefully.   
- Choose the answer that best matches the evidence.   
- If evidence is unclear, make the most reasonable inference.

# # Output Format

Provide your answer as a single letter (A, B, C, or D) based on the evidence.

Figure 16. Prompt for response agent to generate response based on retrieved results.
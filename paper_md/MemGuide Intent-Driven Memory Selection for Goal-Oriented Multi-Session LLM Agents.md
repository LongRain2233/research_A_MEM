# MemGuide: Intent-Driven Memory Selection for Goal-Oriented Multi-Session LLM Agents

Yiming $\mathbf { D } \mathbf { u } ^ { 1 * }$ , Bingbing Wang2*, Yang $\mathbf { H e } ^ { 3 }$ , Bin Liang1, Baojun Wang4, Zhongyang $\mathbf { L i } ^ { 4 }$ , Lin Gui5, Jeff Z. Pan6, Ruifeng $\mathbf { X } \mathbf { u } ^ { 2 }$ , Kam-Fai Wong1

1The Chinese University of Hong Kong, 2Harbin Institute of Technology, Shenzhen

3The Hong Kong University of Science and Technology, 4Huawei Noah’s Ark Lab, 5King’s College London,Multi-Session Memory

6The University of Edinburgh, {ydu, kfwong}@se.cuhk.edu.hk, bingbing.wang@stu.hit.edu.cn

# Abstract

Modern task-oriented dialogue (TOD) systems increasingly rely on large language model (LLM) agents, leveraging Retrieval-Augmented Generation (RAG) and long-context capabilities for long-term memory utilization. However, these methods are primarily based on semantic similarity, overlooking task intent and reducing task coherence in multisession dialogues. To address this challenge, we introduce MemGuide, a two-stage framework for intent-driven memory selection. (1) Intent-Aligned Retrieval matches the current dialogue context with stored intent descriptions in the memory bank, retrieving QA-formatted memory units that share the same goal. (2) Missing-Slot Guided Filtering employs a chain-of-thought slot reasoner to enumerate unfilled slots, then uses a fine-tuned LLaMA-8B filter to re-rank the retrieved units by marginal slot-completion gain. The resulting memory units inform a proactive strategy that minimizes conversational turns by directly addressing information gaps. Based on this framework, we introduce the $\mathbf { M S - T O D } ^ { \mathrm { i } }$ , the first multi-session TOD benchmark comprising 132 diverse personas, 956 task goals, and annotated intent-aligned memory targets, supporting efficient multi-session task completion. Evaluations on MS-TOD show that MemGuide raises the task success rate by $11 \%$ $( 8 8 \%  9 9 \%$ ) and reduces dialogue length by 2.84 turns in multi-session settings, and maintains parity with single-session benchmarks.

# Introduction

Modern task-oriented dialogue (TOD) systems increasingly integrate large language models (LLMs) to enhance generalization, context understanding, and response generation (Nguyen et al. 2025; Xu et al. 2024a,b; Chung et al. 2023; Hudecek and Dusek 2023). To utilize historical dialogue ˇ states across turns, two dominant strategies have emerged: retrieval-augmented generation (RAG), which supplements the LLM with relevant task descriptions or prior dialogue states (Xu et al. 2024a,b), and the use of long-context models, which encode the entire history directly as input (Nguyen et al. 2025). However, they are both limited to surface-level semantic similarity, often ignoring the taskspecific intent and slot-level continuity crucial for coherent multi-session TOD.

![](images/4317211c9dccd2908f725de497e7be2618289d2ae27ea69ed03fb6f30ce3123b.jpg)  
Figure 1: Task-oriented dialogue, without (left) vs. with (right) multi-session memory; the former demands more turns of conversation.

While memory-augmented methods (Lu et al. 2023) have emerged, they are often evaluated on single-session benchmarks that lack support for long-term goal tracking and multi-session memory supervision. Unlike open-domain settings that focus on free-form recall (Zhong et al. 2024), taskoriented dialogue demands structured slot tracking, evolving intent management, and consistent state maintenance. Crucially, real-world users frequently interact with assistants across multiple sessions to accomplish complex goals, yet most existing TOD models and datasets (Budzianowski et al. 2018; Rastogi et al. 2020; Stacey et al. 2024; Liu et al. 2024a) are confined to single-session settings, highlighting a fundamental gap between current TOD systems and the demands of persistent, goal-driven dialogue settings. As shown in Figure 1, traditional single-session TOD systems require users to restate details (e.g., flight times, seat preferences) in every session, leading to inefficiency and frustration.

To address this, we propose MemGuide, a novel framework that utilizes long-term memory in the multi-session TOD task. Unlike prior methods that rely solely on semantic similarity for memory selection, MemGuide incorporates task intent and slot-level guidance to enhance memory relevance and response quality. It consists of two core phases: (1) Intent-Aligned Retrieval, where an LLM generates an intent hypothesis based on the current dialogue state and retrieves memory units aligned with the predicted goal, ensuring retrieved history is both semantically similar and goalconsistent to support cross-session task tracking robustly. (2) Missing-Slot Guided Filtering, which first employs a chain-of-thought (CoT) slot reasoner to detect missing slot values, followed by a filtering module that removes irrelevant or redundant Question Answering (QA) memory units to distill slot-level content for response generation. These two phases enable MemGuide to convert long-term user history into actionable context, supporting minimal-turn, taskconsistent dialogue generation across sessions.

To enable systematic evaluation, we construct the Multi-Session Task-oriented Dialogue (MS-TOD), a new benchmark comprising 132 simulated speakers, each engaging in over 20 sessions covering diverse task goals derived from Schema-Guided Dialogue (SGD) (Rastogi et al. 2020). MS-TOD supports evaluation of slot continuity and long-term memory retrieval across sessions. Unlike opendomain benchmarks focused on retrieving dialogue summaries (Zhong et al. 2024; Li et al. 2024a; Du et al. 2024), multi-session TOD introduces additional challenges. Systems must recall key slot-value pairs, track evolving user intents, and proactively resolve missing or outdated information, while minimizing redundant interactions. To support automatic evaluation, we introduce a proactive response generation module to simulate user engagement and evaluate system performance in resolving missing information. Experimental results demonstrate that MemGuide significantly improves dialogue coherence, response quality, task success rate, and overall efficiency in multi-session TOD. The main contributions include:

• We propose MemGuide, a two-stage framework that distills and leverages cross-session memory for efficient, minimal-turn task completion.   
• We introduce MS-TOD, the first multi-session TOD dataset and benchmark task for evaluating long-term memory integration across sessions.   
• We demonstrate that MemGuide consistently outperforms strong baselines across multiple metrics, demonstrating the effectiveness of intent-aware memory retrieval and slot-guided filtering.

# Related Work

# Task-Oriented Dialogue Dataset

TOD datasets are typically constructed via either Machineto-Machine (M2M) (Shah et al. 2018; Rastogi et al. 2020) or Wizard-of-Oz (WOz) setups (Wen et al. 2017; Budzianowski et al. 2018). M2M datasets (e.g., SGD, STAR) provide schema-driven task flows, while WOz-based

Table 1: Evaluation of confirmation-type response generation under different prompting and retrieval strategies.   

<table><tr><td>Settings</td><td>GPT-4 Score</td><td>Slot Acc.</td></tr><tr><td colspan="3">No Retrieval (Direct Prompting)</td></tr><tr><td>Current Session Context</td><td>2.60</td><td>0.13</td></tr><tr><td>Full Context</td><td>4.76</td><td>0.61</td></tr><tr><td colspan="3">Retrieval-based Methods</td></tr><tr><td>BM25-Based Retrieval</td><td>5.90</td><td>0.53</td></tr><tr><td>Embedding-Based Retrieval</td><td>7.01</td><td>0.67</td></tr><tr><td>Hybrid Retrieval</td><td>7.04</td><td>0.68</td></tr><tr><td colspan="3">Oracle (Upper Bound)</td></tr><tr><td>Oracle</td><td>8.51</td><td>0.82</td></tr></table>

datasets (e.g., MultiWOZ, FRAMES) offer more natural but annotation-heavy dialogues. Recent efforts aim to improve realism and domain diversity (Hu et al. 2023; Dai et al. 2022; Xu et al. 2024b; Li et al. 2024b), yet existing benchmarks primarily assume single-session tasks. There remains a notable gap in datasets designed for multi-session TOD, where tracking long-range goals and user intents is essential.

# Task-Oriented Dialogue Systems

Traditional TOD systems adopt modular pipelines for NLU, DST, and response generation (Wu et al. 2019a; Peng et al. 2018), later unified into end-to-end models trained on annotated dialogues (Wen et al. 2017; Wang et al. 2020). With the rise of LLMs, recent work explores their use in zero-shot and fine-tuned TOD (Madotto et al. 2021; Bang, Lee, and Koo 2023), often achieving strong results on intent recognition and slot filling. In parallel, long-term memory (LTM) methods such as ChitChat (Li et al. 2024c), MemoryBank (Zhong et al. 2024), and LoCoMo (Maharana et al. 2024) support extended context retention through summarization or heuristic filtering, but lack structured memory aligned with task goals. Most assume single-session dialogues and overlook challenges in maintaining multi-session goal continuity. This work addresses these gaps by introducing MemGuide for long-range, goal-aware tracking.

# Preliminary Experiments

To motivate our framework, we first examine the limitations of direct prompting in multi-session TOD and explore the potential of retrieval-based strategies. Since existing TOD datasets lack long-term dependencies, we construct an evaluation set focused on confirmation-type response generation. This evaluation set adopts the same input–output formulation, evaluation criteria, and sample set as the MS-TOD benchmark to ensure consistency and comparability.

To clarify the evaluation setting, we formulate a confirmation-style response generation task. Formally, given the current user utterance $u _ { t }$ , the dialogue context $c =$ $\left\{ u _ { 1 } , r _ { 1 } , \ldots , u _ { t - 1 } , r _ { t - 1 } \right\}$ from the current session, and history dialogue sessions $H$ in the corresponding memory bank (retrieved or concatenated), the model generates a confirmation-style response $r _ { t }$ . The task goal $g$ is held out for reference during evaluation. This setting mimics a common scenario in TOD where the system determines whether

![](images/68fcffb636892165c27b06c8cf3dc6ca66d93b795c64c9c35c5c353b1f7746b4.jpg)

Task dialog history

![](images/73cb2a0d9004586b1782194adb787fc5b9a1486c24e3a882ad2901046a0a8ee4.jpg)

Task Completion Evaluation

![](images/7c8256e3c1ad75587277caa707c2218354fb4194236f54fef010f3a8c8d96ac1.jpg)  
Figure 2: An Example of MS-TOD dataset.

Domain: Travel Intent: BookHotel

User Task_Goal

check_in_date: March 1st check_out_date : March 8th where_to: San Francisco number_of_adults: one adult

User: Hi, I'm looking to book a room in San Francisco.

Assistant: Sure! I can help with that. When would you like to check in?

User: I'd like to check in on March 1st.

Assistant: Got it. And when will you be checking out?

User: I'll be checking out on March 8th.

Assistant: Alright. How many adults will be staying?

User: Just one adult.

Session #1

Domain: Travel Intent: BookHotel

User Task_Goal

check_in_date: March 1st check_out_date : March 8th where_to : San Francisco number_of_adults: one adult

User: Hi. Did you find any good hotel options for my stay in San Francisco?

Assistant: Hi! Yes, I did. The first option is Hotel ABC, which offers free breakfast and Wi-Fi. The second option is Hotel XYZ, located near popular tourist spots and includes a gym facility. Which one sounds better to you.

User: Hotel XYZ sounds better.

Assistant: Just to confirm, you would like to book a room at Hotel XYZ in San Francisco for one adult from March 1st to March 8th, correct?

Session #24

the dialogue contains sufficient information to proceed with task execution. For all settings, we use GPT-4o-mini as the unified generator, allowing a fair comparison across input strategies. We compare two strategies:

(1) Retrieval-based methods, including sparse (BM25 (Robertson and Zaragoza 2009)), dense (text-embeddingsmall- $3 ^ { 2 }$ ), and a hybrid retrieval. Each selects top- $k$ history sessions $H$ relevant to the current utterance $u _ { t }$ with the model input $x = [ H ; c ; u _ { t } ]$ . (2) Direct prompting, where the full dialogue history is concatenated with the current user utterance $u _ { t }$ without retrieval. The input is $x = [ H ; C ; u _ { t } ]$ . The model generates a confirmation response $r _ { t } = \mathrm { L L M } ( x )$ .

As Table 1 shows, retrieval-based methods consistently outperform direct prompting. For instance, dense retrievalbased method achieves 0.67 slot accuracy and 7.01 GPT-4 score, surpassing full-context prompting (0.61 and 4.76). This significant gap highlights how direct prompting struggles with context limitations and the ”lost-in-the-middle” (Liu et al. 2023), where irrelevant history overwhelms key information, motivating our development of MemGuide for advanced intent-aligned retrieval.

# Dataset

MS-TOD is a multi-session benchmark for evaluating longterm memory through user-specific memory banks, with over 20 sessions from a single user. This supports assessment of memory retrieval, slot tracking, and intent continuity. For evaluation, we include held-out sessions with manually annotated confirmation-type responses (Figure 2). MS-TOD is built in four stages: (1) multi-session dialogue generation; (2) confirmation-type response annotation; (3) QAstyle memory bank construction; and (4) human validation.

Table 2: MS-TOD dataset statistics for evaluation.   

<table><tr><td>Attribute</td><td>Evaluation</td></tr><tr><td>Domains</td><td>16</td></tr><tr><td>Intentions</td><td>19</td></tr><tr><td>Task goals</td><td>956</td></tr><tr><td>Dialogues</td><td>2,861</td></tr><tr><td>Utterances</td><td>18,530</td></tr><tr><td>Avg. slots per task goal</td><td>4.24</td></tr><tr><td>Number of individuals</td><td>132</td></tr><tr><td>Avg. intents per individual</td><td>5.45</td></tr><tr><td>Avg. sessions per individual</td><td>21.67</td></tr><tr><td>Avg. utterances per individual</td><td>140.38</td></tr></table>

# Multi-Session Dialogue Generation

We begin by generating multi-session dialogues for each task goal sampled from the SGD dataset (Rastogi et al. 2020). For every task, we synthesize three temporally ordered sessions using GPT-4, each conditioned on the slotfilling status of the previous session. This simulates how users revisit and revise the same task across time. Specifically, Session 1 presents an incomplete task with missing slots; Session 2 introduces updates; and Session 3 concludes with final confirmation. This staged construction reflects real-world dialogue dynamics while avoiding overfitting, as most SGD tasks involve fewer than ten slots.

# Confirmation-Type Response Annotation

To evaluate long-term task fulfillment in dialogue systems, we annotate the final session of each task with confirmationtype responses. Each marks the utterance confirming task completion and associated slot-value goal with manually labels (confirmation/non-confirmation). These annotations serve two purposes: (1) Supervising Memory Selection, indicating when to trigger memory retrieval; and (2) Supporting Evaluation, evaluating if the system recalls goalrelevant content and generates accurate confirmations.

# Memory Bank Construction

Since multi-session interactions are organized around individuals, we group dialogues into individual memory banks (Figure 2), where each bank stores temporally ordered sessions. During construction, we ensure that intents within the same bank are distinct and non-conflicting, enabling consistent memory usage and avoiding cross-intent interference. Each bank contains over 20 sessions spanning at least six task intents (Table 2), with one held-out evaluation session per intent for confirmation-type assessment. For each completed session, we use GPT-4 to generate an intent description along with a set of QA pairs, each capturing a slotspecific fact for retrieval. This QA-style format is motivated by prior work (Chen et al. 2023) showing that question–answer structures facilitate more accurate and efficient retrieval compared to unstructured text. To maintain causal consistency, memory access is limited to sessions prior to the current evaluation point.

![](images/b202b5af3c2d0913442bdd359040bf49f23cc40b1dab8592f92648884d687c8c.jpg)  
Figure 3: Overflow of our MemGuide framework, which comprises Intent-Aligned Retrieval and Missing-Slot Guided Filtering.

# Human Validation

To ensure coherence, correctness, and usability, we apply a structured multi-stage validation process involving three annotators experienced in NLP. The process includes: (1) Verifying each session preserves the intended goal and correct slot values; (2) Removing dialogues with excessive redundancy across sessions; (3) Verifying that confirmation-type utterances match the expected slot-value goals; (4) Removing sessions that fail to complete defined task goals or lack confirmation turns; and (5) Excluding episodes with unnatural repetition of similar intents. We additionally conduct inter-annotator agreement evaluation to ensure labeling consistency.

# MemGuide

MemGuide is a two-stage framework for multi-session TOD, as shown in Figure 3. It first performs Intent-Aligned Retrieval, extracting the current user intent via an LLM and retrieving QA-formatted memory units from the memory bank that align with this intent using semantic similarity. Then, Missing-Slot Guided Filtering identifies unfilled task slots through a CoT reasoner and re-ranks retrieved memories with a fine-tuned LLaMA-8B filter based on their ability to fill these slots. Finally, Response Generation leverages the filtered memories to produce proactive responses, reducing conversational turns and boosting task success.

Given a dialogue context $c = \{ u _ { 1 } , r _ { 1 } , \ldots , u _ { t - 1 } , r _ { t - 1 } , u _ { t } \}$ and a user-specific long-term memory bank $M$ , the goal is to generate the next system response $r _ { t }$ . The memory bank $M$ is a collection of memories from past sessions, structured as $M = \{ ( k _ { i } , V _ { i } ) \} _ { i = 1 } ^ { N }$ and $t$ is the number of turns. Each entry consists of a high-level intent description $k _ { i }$ (e.g., ”book a flight to San Francisco”) represents the high-level intent and a corresponding set of QA-formatted memory units $V _ { i } = \{ ( q _ { i , j } , a _ { i , j } ) \} _ { j = 1 } ^ { n }$ , where $q _ { i , j }$ is the $j$ -th question about a task slot within the $i$ -th intent (e.g., ”What is the departure date?”) and $a _ { i , j }$ is the corresponding answer (e.g., ”July 28, 2025”). The optimal response $r$ should coherently continue the conversation while strategically utilizing information from $M$ to progress towards task completion. It is

generated by an LLM conditioned on the context and a carefully selected memory subset $M _ { s e l } \subseteq M$ .

# Intent-Aligned Retrieval

The stage aims to broadly identify past sessions relevant to the current user goal by retrieving memory entries from the long-term bank that share a consistent high-level intent, ensuring thematic alignment.

Current Intent Extraction. Given the dialogue context c, we use GPT-4o-mini to generate a high-level intent description $d _ { i n t }$ , which summarizes the user’s objective in the current session. The model is prompted to generate a short, command-like phrase summarizing the user’s goal (e.g., ”find a local Italian restaurant”). This distilled phrase serves as a current intent key, $k _ { c u r }$ , providing a canonical task representation that is standardized for retrieval.

Semantic Retrieval. We then retrieve memory units from the bank $M$ , which contain stored intent keys $\{ k _ { i } \} _ { i = 1 } ^ { N }$ . and corresponding QA pairs that are semantically closest to the extracted current intent $k _ { c u r }$ . We use an embedding model (e.g., text-embedding-3-small) to compute dense vector representations for all memory units. The relevance score between $k _ { c u r }$ and each memory unit is calculated using cosine similarity. The top- $K$ memory units $( k _ { i } , V _ { i } )$ with the highest scores are selected to form the candidate memory set, $M _ { c a n d }$ . This procedure ensures that we only consider memories from sessions with a shared high-level objective, guaranteeing thematic alignment for subsequent stages.

# Missing-Slot Guided Filtering.

While the retrieved memories in $M _ { c a n d }$ are thematically aligned with the user’s intent, their immediate utility for the current dialogue turn can vary significantly. To address this, this stage performs a fine-grained filtering, prioritizing memory units that are most likely to resolve the immediate information needs of the dialogue.

Information Gap Identification. To guide the filtering process, we first must precisely identify what informa-

tion is required to advance the current task. We leverage an LLM configured as a CoT reasoner (Wei et al. 2022) to analyze the dialogue context $c$ and the overall goal corresponding with intent key $k _ { c u r }$ , so as to enumerate a list of essential task slots that have not yet been filled or confirmed. The CoT prompt is structured to guide the LLM through a logical sequence. First, it enumerates all required slots for the intent, then checks the dialogue history to determine which of these slots have already been filled or confirmed, and finally outputs only those slots that remain unresolved. The result is a list of hypothesized missing slots, ${ \cal L } _ { m i s s } ~ = ~ \{ s l o t _ { 1 } , s l o t _ { 2 } , \ldots \}$ For example, in a flight booking task where the user has only specified a destination, $L _ { m i s s }$ might be identified as {departure date, return date, seat preference}.

Re-ranking by Marginal Slot-Completion Gain. Having identified the information gaps represented by $L _ { m i s s }$ we then re-rank each QA pair $( q _ { i , j } , a _ { i , j } )$ within the candidate set $M _ { c a n d }$ based on its potential to fill one of these gaps. This process is driven by the principle of selecting information with the highest marginal slot-completion gain.

To operationalize this, we fine-tune a smaller, efficient LLaMA-8B model (Meta AI 2024) to act as a specialized filter. This filter estimates the probability that a given QA pair provides an answer for one of the missing slots. For each QA pair $( q _ { i , j } , a _ { i , j } )$ from $M _ { c a n d }$ , the model computes:

$$
s _ {i, j} = P (y = 1 \mid c, L _ {\text {m i s s}}, q _ {i, j}, a _ {i, j}) \tag {1}
$$

where $y = 1$ signifies that the answer $a _ { i , j }$ successfully fills a slot presented in $L _ { m i s s }$ . To supervise this filter, we construct a training dataset using the same pipeline as MS-TOD memory bank generation. Specifically, for each held-out session, we simulate missing-slot contexts and label each QA pair from prior sessions as positive (if it fills a missing slot) or negative (otherwise). Detailed training configurations and dataset are provided in Appendix. The model is optimized using a standard binary cross-entropy loss function:

$$
\mathcal {L} = - \sum_ {i, j} \left[ y _ {i, j} \log s _ {i, j} + \left(1 - y _ {i, j}\right) \log \left(1 - s _ {i, j}\right) \right] \tag {2}
$$

To balance the initial semantic relevance from the initial semantic retrieval (denoted as $s _ { i , j } ^ { \mathrm { p r e } }$ ) with the slot-filling utility score $s _ { i , j }$ from our filter, we compute a final score:

$$
s _ {\text {f i n a l}, i j} = \alpha \cdot s _ {i, j} ^ {\text {p r e}} + (1 - \alpha) \cdot s _ {i, j} \tag {3}
$$

where $\alpha$ is a hyperparameter. We select the top- $K$ (e,g,. ${ \mathrm { K } } { = } 5$ ) QA pairs with the highest $s _ { \mathrm { f i n a l } , i j }$ scores.

# Response Generation

The final response is generated using the top- $K$ memorys $A _ { \mathrm { c o r e } } = \{ a _ { 1 } , a _ { 2 } , \ldots , a _ { K } \}$ , omitting auxiliary questions $q _ { i , j }$ An LLM reader receives the dialogue context $c$ , the core facts $A _ { \mathrm { c o r e } }$ , and missing slots $L _ { \mathrm { m i s s } }$ as prompt inputs.

$$
r = \text {L L M R e a d e r} (\text {p r o m p t} (c, A _ {\text {c o r e}}, L _ {\text {m i s s}})) \tag {4}
$$

The prompt instructs the model to: (1) continue the conversation naturally, and (2) proactively address $L _ { \mathrm { m i s s } }$ using

$A _ { \mathrm { c o r e } }$ , e.g., by confirming or suggesting stored values. For example, if memory provided a preferred airline, the system might respond, ”I see you’re flying to Montreal. Last time you flew with Air Canada. Would you like to book with them again?” This proactive strategy, directly informed by our two-stage memory selection, minimizes redundant questions and accelerates task completion. All prompt templates used in our method are provided in Appendix.

# Experiments

To enable fine-grained assessment of long-term memory utilization and task completion, we evaluate MemGuide on sessions from the MS-TOD benchmark that are annotated with confirmation-type responses, in which the final utterance explicitly confirms that the user goal has been achieved. Each session is associated with a gold-standard task goal and corresponding slot-value set, enabling precise evaluation using standard metrics of task success and response quality.

# Experimental Setups

Evaluation Metrics. We use four core automatic metrics and human evaluation to evaluate response performance: 1) GPT-4 score, (1–10) 3 evaluates response quality in terms of fluency, coherence, and informativeness; 2) Joint Goal Accuracy (JGA) measures slot prediction accuracy; 3) Dialogue Turn Efficiency (DTE) captures the number of turns required to complete a task, and 4) Success Rate (S.R.) indicates whether the user goal is achieved. 5) Human evaluation further assesses Accuracy, Informativeness, and Coherence, with A.I.C. denoting their average.

Baselines. We evaluate MemGuide against three representative categories of baselines: 1) General-purpose LLMs. We assess full-context prompt (FCP)-based dialogue performance using instruction-tuned models including LLaMA3- 8B (Touvron et al. 2024), Qwen2.5-7B (Team 2024c), Mistral-7B (Team 2024a), and GPT-4o-mini (Team 2024b). 2)Traditional Task-Oriented Dialogue Systems. To evaluate MemGuide under structured DST conditions, we include task-specific baselines such as BERT-DST (Chao and Lane 2019), LDST (Feng et al. 2024), and AutoTOD (Xu et al. 2024a), where the latter incorporates an external memory module for cross-turn goal tracking. While these models were not originally designed for multi-session scenarios, they represent the strongest available TOD pipelines when adapted to this setting. 3) Long-term Summarization. We implement a summarization-based baseline inspired by ChatCite (Li et al. 2024c), which condenses session histories into concise summaries used during inference.

# Main Results

Comparision with General-purpose LLMs. Compared to FCP, MemGuide leverages the same underlying LLM as a memory reader to retrieve and utilize relevant long-term memory, yielding substantial improvements across critical metrics. As illustrated in Table 3, our approach consistently

Table 3: Combined results comparing FCP and MemGuide across LLMs. GPT-4, JGA, DTE, and S.R. are automatic metrics; A., I., C., and A.I.C. are human evaluation metrics for accuracy, informativeness, coherence, and their average score.   

<table><tr><td>Model</td><td>Setting</td><td>GPT-4</td><td>JGA</td><td>DTE</td><td>S.R.</td><td>A.</td><td>I.</td><td>C.</td><td>A.I.C.</td></tr><tr><td rowspan="2">LLaMA3-8B</td><td>FCP</td><td>4.89</td><td>0.64</td><td>5.37</td><td>0.82</td><td>0.56</td><td>1.47</td><td>1.74</td><td>1.26</td></tr><tr><td>MemGuide</td><td>6.39</td><td>0.63</td><td>3.46</td><td>0.92</td><td>0.61</td><td>1.98</td><td>2.16</td><td>1.58</td></tr><tr><td rowspan="2">Qwen-7B</td><td>FCP</td><td>6.26</td><td>0.66</td><td>4.93</td><td>0.83</td><td>0.43</td><td>1.24</td><td>1.85</td><td>1.17</td></tr><tr><td>MemGuide</td><td>6.81</td><td>0.66</td><td>4.31</td><td>0.87</td><td>0.54</td><td>1.70</td><td>2.30</td><td>1.51</td></tr><tr><td rowspan="2">Mistral-7B</td><td>FCP</td><td>6.20</td><td>0.73</td><td>2.52</td><td>1.00</td><td>0.58</td><td>1.63</td><td>1.99</td><td>1.40</td></tr><tr><td>MemGuide</td><td>6.48</td><td>0.80</td><td>1.21</td><td>1.00</td><td>0.61</td><td>2.06</td><td>2.08</td><td>1.58</td></tr><tr><td rowspan="2">GPT-4o-mini</td><td>FCP</td><td>6.93</td><td>0.67</td><td>6.03</td><td>0.88</td><td>0.62</td><td>1.83</td><td>1.90</td><td>1.78</td></tr><tr><td>MemGuide</td><td>7.14</td><td>0.70</td><td>3.19</td><td>0.99</td><td>0.65</td><td>2.38</td><td>2.48</td><td>2.17</td></tr></table>

Table 4: Results of traditional TOD models, summary-based methods, and MemGuide. Models marked with ∗ focus on DST only. † indicates a simplified AutoTOD pipeline.   

<table><tr><td>Model</td><td>GPT-4</td><td>JGA</td><td>DTE</td><td>S.R.</td></tr><tr><td>Bert-DST*</td><td>-</td><td>0.07</td><td>-</td><td>-</td></tr><tr><td>LDST*</td><td>-</td><td>0.23</td><td>-</td><td>-</td></tr><tr><td>\( AutoTOD^† \)</td><td>6.49</td><td>0.44</td><td>7.80</td><td>0.81</td></tr><tr><td>ChatCite</td><td>6.59</td><td>0.660</td><td>4.71</td><td>0.84</td></tr><tr><td>MemGuide</td><td>7.14</td><td>0.70</td><td>3.19</td><td>0.99</td></tr></table>

outperforms FCP across all tested models, demonstrating robust gains in task accuracy, response quality, and interaction efficiency. For example, when using Mistral-7B as the LLM Reader, MemGuide increases JGA from 0.73 to 0.80 and reduces DTE from 2.52 to 1.21, representing a $52 \%$ reduction. LLaMA3-8B achieves the largest improvement in GPT-4 score (from 4.89 to 6.39), while GPT-4o-mini reduces dialogue turns from 6.03 to 3.19, corresponding to a $4 7 . 1 \%$ decrease.. Similar gains are observed with Qwen-7B and other models. These consistent gains across models confirm that integrating memory-guided reasoning with the same base model enhances not only task accuracy but also response relevance and interaction fluency, validating the effectiveness of intent-aligned memory selection in multisession task-oriented dialogue.

Comparison with Traditional TOD and Summarization Baselines. As existing models are not explicitly designed for multi-session TOD, we compare MemGuide with two representative categories: (1) traditional Dialogue State Tracking (DST)-focused models (BERT-DST, LDST, Auto-TOD), and (2) a summarization-based approach inspired by ChatCite. As shown in Table 4, MemGuide consistently outperforms all baselines across key metrics. Compared to AutoTOD, it increases JGA from 0.440 to 0.698 and reduces DTE from 7.80 to 3.19. Against the summarization baseline, MemGuide achieves clear improvements across all metrics: the GPT-4 score increases from 6.59 to 7.14, JGA improves from 0.66 to 0.70, DTE decreases from 4.71 to 3.19, and the success rate rises from 0.84 to 0.99. These results demonstrate the effectiveness of MemGuide in improving both task performance and dialogue efficiency.

Human Evaluation. We conduct human evaluation

Table 5: Results of different methods on SGD and MultiWOZ 2.2. MemGuide∗ is a single-session variant of MemGuide, where the missing slot guided filtering is disabled while retaining the QA memory.   

<table><tr><td>Dataset</td><td>Methods</td><td>JGA</td><td>AGA</td></tr><tr><td rowspan="6">SGD</td><td>SGD Baseline</td><td>0.254</td><td>0.906</td></tr><tr><td>GOLOMB</td><td>0.465</td><td>0.750</td></tr><tr><td>SGP-DST</td><td>0.722</td><td>0.913</td></tr><tr><td>TS-DST</td><td>0.786</td><td>0.956</td></tr><tr><td>LDST</td><td>0.845</td><td>0.994</td></tr><tr><td>MemGuide*</td><td>0.846</td><td>0.965</td></tr><tr><td rowspan="8">MultiWOZ 2.2</td><td>SGD Baseline</td><td>0.420</td><td>-</td></tr><tr><td>TRADE</td><td>0.454</td><td>-</td></tr><tr><td>DS-DST</td><td>0.517</td><td>-</td></tr><tr><td>TripPy</td><td>0.530</td><td>-</td></tr><tr><td>TOATOD</td><td>0.638</td><td>-</td></tr><tr><td>SDP-DST</td><td>0.576</td><td>0.985</td></tr><tr><td>LDST</td><td>0.607</td><td>0.988</td></tr><tr><td>MemGuide*</td><td>0.879</td><td>0.976</td></tr></table>

to assess the effectiveness of MemGuide in generating confirmation-type responses after memory-guided dialogue planning. Human annotators rate each response along three dimensions: Accuracy (binary), Informativeness (scored from 0 to 3), and Coherence (scored from 0 to 3). The average of these scores, denoted as A.I.C., provides an overall measure of perceived response quality. As shown in Table 4, MemGuide consistently improves human-judged quality across all metrics. All evaluations are conducted under a blind review protocol.

Generalization to Single-Session DST Tasks. To assess the generalization of MemGuide in single-session settings, we focus on dialogue state tracking (DST), a core task that supports downstream components like policy planning and response generation. DST is a widely used and welldefined task that depends on context understanding and supports downstream dialogue components. We evaluate it on two widely-used single-session DST benchmarks, SGD and MultiWOZ2.2. While both benchmarks focus on DST, differences in annotation schemes and domain coverage result in distinct baseline configurations across SGD and Multi-WOZ2.2 (Table 5). On SGD, MemGuide achieves a state-

Table 6: Comparison of GPT-4 scores using retrieved history vs. intent-QA memory across different LLM settings.   

<table><tr><td>Setting</td><td>w/ Raw History</td><td>w/ Intent-QA Memory</td></tr><tr><td>LLaMA3-8B</td><td>5.09</td><td>6.34</td></tr><tr><td>Qwen-7B</td><td>6.38</td><td>6.56</td></tr><tr><td>Mistral-7B</td><td>5.86</td><td>6.71</td></tr><tr><td>GPT-4o-mini</td><td>7.01</td><td>7.14</td></tr></table>

of-the-art JGA of 0.846, surpassing strong baselines such as LDST (Feng et al. 2023), GOLOMB (Gulyaev et al. 2020), SGP-DST (Ruan et al. 2020), and TS-DST (Du et al. 2022), and performs comparably to LDST on Average Goal Accuracy (AGA) (Rastogi et al. 2020). On MultiWOZ2.2, MemGuide* attains a JGA of 0.879, significantly outperforming prior models including TRADE (Wu et al. 2019b), TripPy (Heck et al. 2020), and SDP-DST (Lee, Cheng, and Ostendorf 2021). We attribute the superior performance to QA memory’s ability to capture slot dependencies more effectively in smaller domain settings, confirming its adaptability and robustness across datasets.

# Ablation Study

Effect of intent-aligned Retrieval. Table 6 shows that our intent-aligned retrieval with structured QA memory consistently improves response quality, outperforming unstructured baselines by up to 1.29 points (e.g., LLaMa3-8B: 5.05 $ 6 . 3 4 \AA$ ). We attribute this enhancement to the alignment between the retrieved structured context and the model’s generative reasoning process. By providing intent-anchored QA pairs, the system can reason over task-relevant content with greater precision and mitigated ambiguity, thereby corroborating the theoretical premise that structured supervision is key to enhancing long-context utility in dialogue systems.

Effect of Missing-Slot Guided Filtering. We assess the impact of the missing-slot guided filtering module by removing it while retaining the same hybrid RAG retrieval. As shown in Figure 4, the absence of filtering results in a significant performance drop: JGA on Qwen2.5-7B drops from 0.74 to 0.41, and DTE on GPT-4o-mini increases from 3.19 to 4.30. This highlights the critical role of fine-grained memory selection in both accuracy and interaction efficiency. This module operates in two stages: the CoT reasoner identifies missing task slots, which guide the memory filter to prioritize QA pairs that fill these gaps. This joint reasoning and filtering strategy significantly improves retrieval quality: compared to semantic-only retrieval, Recall $\textcircled { a } 5$ improves by $7 . 7 \%$ on average, raising the performance of textembedding-3-small from 0.792 to 0.832. By bridging retrieval with task-specific gaps, the filter prioritizes slot-value pairs that are more likely to advance the dialogue.

# Case Study

In our case study, we compared four methods for generating confirmation responses: (1) FCP, (2) Hybrid RAG, (3) ChatCite, and (4) MemGuide. As shown in Figure 5, FCP often introduces irrelevant or outdated content due to un-

![](images/f03f311b34aacfc849ae60f2ea1e8ae01537c32260787ebe67cd05258aec5f45.jpg)  
Figure 5: Case study.

Figure 4: Comparison of Hybrid RAG (blue) vs. MemGuide (purple) across four LLMs on two metrics: JGA and DTE.

<table><tr><td colspan="2">Session 23 Evaluation - Intention: The user intends to listen to songs.</td></tr><tr><td>Dialogue Context</td><td>User: Can you play &#x27;Drive&#x27; again? Assistant:</td></tr><tr><td>FCP</td><td>Sure! Just to confirm, you&#x27;d like me to play the song &quot;Drive&quot; on the bedroom speaker again. Should I start it now? Also, would you like me to check out some tickets for ... (Extra Noise)</td></tr><tr><td>Hybrid RAG</td><td>Sure! Let me play &quot;Drive&quot; for you once more. By the way, have you decided on the number of tickets you&#x27;d like to buy for Bill Callahan&#x27;s concert? It&#x27;s coming up on March 10th in Los Angeles! (Wrong Answer)</td></tr><tr><td>ChatCite</td><td>Sure! I&#x27;ve started playing &quot;Drive&quot; again. Is there anything else you&#x27;d like? (Key Slot Omission)</td></tr><tr><td>MemGuide</td><td>Sure! Just to confirm, you&#x27;d like me to play the song &quot;Drive&quot; on the bedroom speaker again. Should I start it now? (Right Answer)</td></tr></table>

filtered long context, while Hybrid RAG and ChatCite frequently miss critical slot values such as dates or preferences, reflecting retrieval noise or lack of task-specific reasoning. In contrast, MemGuide consistently produces accurate and concise responses by combining intent-aligned retrieval with missing-slot guided filtering. Notably, it achieves higher slot coverage and fluency than other methods, as verified by both GPT-4 scores and manual annotation. These results highlight the value of intent-aware retrieval and task-specific filtering for enhancing response quality in multi-session TOD.

# Conclusion

We present MemGuide, a two-stage memory-guided framework for multi-session LLM agents. By combining intent-aligned retrieval with missing-slot guided filtering, MemGuide enables task-aware, slot-specific memory selection that surpasses traditional semantic similarity. Evaluated on MS-TOD, our novel benchmark for multi-session TOD, MemGuide significantly improves task success, shortens dialogues, and enhances interaction coherence. These results confirm that structured memory supervision and goal-aware reasoning are critical for developing effective LLM agents.

# References

Bang, N.; Lee, J.; and Koo, M.-W. 2023. Task-Optimized Adapters for an End-to-End Task-Oriented Dialogue System. In Findings of the Association for Computational Linguistics: ACL 2023, 7355–7369.   
Budzianowski, P.; Wen, T.-H.; Tseng, B.-H.; Casanueva, I.; Ultes, S.; Ramadan, O.; and Gasic, M. 2018. MultiWOZ-A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 5016–5026.   
Chao, G.-L.; and Lane, I. 2019. BERT-DST: Scalable End-to-End Dialogue State Tracking with Bidirectional Encoder Representations from Transformer. arXiv preprint arXiv:1907.03040.   
Chen, W.; Verga, P.; de Jong, M.; Wieting, J.; and Cohen, W. W. 2023. Augmenting Pre-trained Language Models with QA-Memory for Open-Domain Question Answering. In Vlachos, A.; and Augenstein, I., eds., Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, 1597–1610. Dubrovnik, Croatia: Association for Computational Linguistics.   
Chung, W.; Cahyawijaya, S.; Wilie, B.; Lovenia, H.; and Fung, P. 2023. Instructtods: Large language models for end-to-end task-oriented dialogue systems. arXiv preprint arXiv:2310.08885.   
Dai, Y.; He, W.; Li, B.; Wu, Y.; Cao, Z.; An, Z.; Sun, J.; and Li, Y. 2022. CGoDial: A Large-Scale Benchmark for Chinese Goal-oriented Dialog Evaluation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 4097–4111.   
Du, M.; Cheng, L.; Xu, B.; Wang, Z.; Wang, S.; Yuan, J.; and Pan, C. 2022. TS-DST: A Two-Stage Framework for Schema-Guided Dialogue State Tracking with Selected Dialogue History. In 2022 International Joint Conference on Neural Networks (IJCNN), 1–8.   
Du, Y.; Wang, H.; Zhao, Z.; Liang, B.; Wang, B.; Zhong, W.; Wang, Z.; and Wong, K.-F. 2024. PerLTQA: A Personal Long-Term Memory Dataset for Memory Classification, Retrieval, and Fusion in Question Answering. In Wong, K.-F.; Zhang, M.; Xu, R.; Li, J.; Wei, Z.; Gui, L.; Liang, B.; and Zhao, R., eds., Proceedings of the 10th SIGHAN Workshop on Chinese Language Processing (SIGHAN-10), 152–164. Bangkok, Thailand: Association for Computational Linguistics.   
Feng, H.; Zhang, W.; Liu, J.; and Sun, M. 2024. LDST: A LLaMA-based Dialogue State Tracking Framework. arXiv preprint arXiv:2403.12345.   
Feng, Y.; Lu, Z.; Liu, B.; Zhan, L.; and Wu, X.-M. 2023. Towards LLM-driven Dialogue State Tracking. arXiv:2310.14970.   
Gulyaev, P.; Elistratova, E.; Konovalov, V.; Kuratov, Y.; Pugachev, L.; and Burtsev, M. 2020. Goal-Oriented Multi-Task BERT-Based Dialogue State Tracker. arXiv:2002.02450.   
Heck, M.; Lubis, N.; Ruppik, B.; Vukovic, R.; Feng, S.; Geishauser, C.; Lin, H.-C.; van Niekerk, C.; and Gasiˇ c, M. ´

2023. ChatGPT for Zero-shot Dialogue State Tracking: A Solution or an Opportunity? arXiv:2306.01386.   
Heck, M.; van Niekerk, C.; Lubis, N.; Geishauser, C.; Lin, H.-C.; Moresi, M.; and Gasiˇ c, M. 2020. TripPy: A Triple ´ Copy Strategy for Value Independent Neural Dialog State Tracking. arXiv:2005.02877.   
Hu, S.; Zhou, H.; Hergul, M.; Gritta, M.; Zhang, G.; Iacobacci, I.; Vulic, I.; and Korhonen, A. 2023. Multi 3 woz: A ´ multilingual, multi-domain, multi-parallel dataset for training and evaluating culturally adapted task-oriented dialog systems. Transactions of the Association for Computational Linguistics, 11: 1396–1415.   
Hudecek, V.; and Dusek, O. 2023. Are Large Language ˇ Models All You Need for Task-Oriented Dialogue? In Stoyanchev, S.; Joty, S.; Schlangen, D.; Dusek, O.; Kennington, C.; and Alikhani, M., eds., Proceedings of the 24th Annual Meeting of the Special Interest Group on Discourse and Dialogue, 216–228. Prague, Czechia: Association for Computational Linguistics.   
Jang, J.; Boo, M.; and Kim, H. 2023. Conversation Chronicles: Towards Diverse Temporal and Relational Dynamics in Multi-Session Conversations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 13584–13606. Singapore: Association for Computational Linguistics.   
Lee, C.-H.; Cheng, H.; and Ostendorf, M. 2021. Dialogue State Tracking with a Language Model using Schema-Driven Prompting. arXiv:2109.07506.   
Li, H.; Yang, C.; Zhang, A.; Deng, Y.; Wang, X.; and Chua, T.-S. 2024a. Hello Again! LLM-powered Personalized Agent for Long-term Dialogue. arXiv preprint arXiv:2406.05925.   
Li, M.; Peng, B.; Gao, J.; and Zhang, Z. 2024b. Opera: Harmonizing task-oriented dialogs and information seeking experience. ACM Transactions on the Web, 18(4): 1–27.   
Li, Y.; Chen, L.; Liu, A.; Yu, K.; and Wen, L. 2024c. ChatCite: LLM agent with human workflow guidance for comparative literature summary. arXiv preprint arXiv:2403.02574.   
Liu, N. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua, M.; Petroni, F.; and Liang, P. 2023. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172.   
Liu, Y.; Fang, Y.; Vandyke, D.; and Collier, N. 2024a. TOAD: Task-Oriented Automatic Dialogs with Diverse Response Styles. arXiv preprint arXiv:2402.10137.   
Liu, Y.; Fang, Y.; Vandyke, D.; and Collier, N. 2024b. TOAD: Task-Oriented Automatic Dialogs with Diverse Response Styles. In Ku, L.-W.; Martins, A.; and Srikumar, V., eds., Findings of the Association for Computational Linguistics: ACL 2024, 8341–8356. Bangkok, Thailand: Association for Computational Linguistics.   
Lu, J.; An, S.; Lin, M.; Pergola, G.; He, Y.; Yin, D.; Sun, X.; and Wu, Y. 2023. Memochat: Tuning llms to use memos for consistent long-range open-domain conversation. arXiv preprint arXiv:2308.08239.

Madotto, A.; Lin, Z.; Winata, G. I.; and Fung, P. 2021. Fewshot bot: Prompt-based learning for dialogue systems. arXiv preprint arXiv:2110.08118.   
Maharana, A.; Lee, D.-H.; Tulyakov, S.; Bansal, M.; Barbieri, F.; and Fang, Y. 2024. Evaluating very long-term conversational memory of llm agents. arXiv preprint arXiv:2402.17753.   
Meta AI. 2024. Meta LLaMA 3.1-8B-Instruct. https: //huggingface.co/meta-llama/Llama-3.1-8B-Instruct.   
Nguyen, Q.-V.; Nguyen, Q.-C.; Pham, H.; and Bui, K.-H. N. 2025. Spec-TOD: A Specialized Instruction-Tuned LLM Framework for Efficient Task-Oriented Dialogue Systems. arXiv preprint arXiv:2507.04841.   
Peng, B.; Li, X.; Gao, J.; Liu, J.; and Wong, K.-F. 2018. Deep Dyna-Q: Integrating Planning for Task-Completion Dialogue Policy Learning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2182–2192.   
Rastogi, A.; Zang, X.; Sunkara, S.; Gupta, R.; and Khaitan, P. 2020. Towards scalable multi-domain conversational agents: The schema-guided dialogue dataset. In Proceedings of the AAAI conference on artificial intelligence, volume 34, 8689–8696.   
Robertson, S.; and Zaragoza, H. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4): 333–389.   
Ruan, Y.-P.; Ling, Z.-H.; Gu, J.-C.; and Liu, Q. 2020. Fine-Tuning BERT for Schema-Guided Zero-Shot Dialogue State Tracking. arXiv:2002.00181.   
Shah, P.; Hakkani-Tur, D.; T ¨ ur, G.; Rastogi, A.; Bapna, A.; ¨ Nayak, N.; and Heck, L. 2018. Building a conversational agent overnight with dialogue self-play. arXiv preprint arXiv:1801.04871.   
Stacey, J.; Cheng, J.; Torr, J.; Guigue, T.; Driesen, J.; Coca, A.; Gaynor, M.; and Johannsen, A. 2024. LUCID: LLM-Generated Utterances for Complex and Interesting Dialogues. arXiv preprint arXiv:2403.00462.   
Team, M. A. 2024a. Mistral 7B: A High-Performance Language Model. arXiv preprint arXiv:2402.12345.   
Team, O. 2024b. GPT-4: OpenAI’s Advanced Language Model. OpenAI Research.   
Team, Q. 2024c. Qwen-2.5: Advanced Large Language Model with Enhanced Capabilities. arXiv preprint arXiv:2409.12345.   
Touvron, H.; Lavril, T.; Izacard, G.; Martinet, X.; Lachaux, M.-A.; Lacroix, T.; Roziere, B.; Goyal, N.; Batra, A.; ` Randriamihaja, S.; et al. 2024. LLaMA 3: Open and Efficient Foundation Language Models. arXiv preprint arXiv:2401.00778.   
Wang, K.; Tian, J.; Wang, R.; Quan, X.; and Yu, J. 2020. Multi-Domain Dialogue Acts and Response Co-Generation. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 7125–7134.   
Wei, J.; Wang, X.; Schuurmans, D.; Bosma, M.; Chi, E.; Le, Q.; and Zhou, D. 2022. Chain of Thought Prompting Elicits Reasoning in Large Language Models. arXiv preprint arXiv:2201.11903.

Wen, T.-H.; Vandyke, D.; Mrksiˇ c, N.; Gasic, M.; Bara- ´ hona, L. M. R.; Su, P.-H.; Ultes, S.; and Young, S. 2017. A Network-based End-to-End Trainable Task-oriented Dialogue System. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, 438–449.   
Wu, C.-S.; Madotto, A.; Hosseini-Asl, E.; Xiong, C.; Socher, R.; and Fung, P. 2019a. Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 808–819.   
Wu, C.-S.; Madotto, A.; Hosseini-Asl, E.; Xiong, C.; Socher, R.; and Fung, P. 2019b. Transferable Multi-Domain State Generator for Task-Oriented Dialogue Systems. arXiv:1905.08743.   
Wu, D.; Wang, H.; Yu, W.; Zhang, Y.; Chang, K.-W.; and Yu, D. 2025. LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory. In Proceedings of the International Conference on Learning Representations (ICLR).   
Xu, H.-D.; Mao, X.-L.; Yang, P.; Sun, F.; and Huang, H.-Y. 2024a. Rethinking Task-Oriented Dialogue Systems: From Complex Modularity to Zero-Shot Autonomous Agent. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2748–2763.   
Xu, J.; Szlam, A.; and Weston, J. 2022. Beyond Goldfish Memory: Long-Term Open-Domain Conversation. In Muresan, S.; Nakov, P.; and Villavicencio, A., eds., Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 5180–5197. Dublin, Ireland: Association for Computational Linguistics.   
Xu, W.; Huang, Z.; Hu, W.; Fang, X.; Cherukuri, R.; Nayyar, N.; Malandri, L.; and Sengamedu, S. 2024b. HR-MultiWOZ: A Task Oriented Dialogue (TOD) Dataset for HR LLM Agent. In Proceedings of the First Workshop on Natural Language Processing for Human Resources (NLP4HR 2024), 59–72.   
Zhong, W.; Guo, L.; Gao, Q.; Ye, H.; and Wang, Y. 2024. MemoryBank: Enhancing Large Language Models with Long-Term Memory. Proceedings of the AAAI Conference on Artificial Intelligence, 38(17): 19724–19731.

# Appendix A

# Prompt of dialogue generation

We designed a multi-session dialogue prompt (as shown in Figure A.1) that generates multi-session dialogue data based on input dialogue intent, task goal, and target session count. Additionally, during the generation process, we annotate whether each utterance is a confirmation response. These annotations, after manual verification, will be used in the main experiment for confirmation-type response generation.

# Prompt of Task Slot Query Generation

During the evaluation process, we design a prompt (as shown in Figure A.2) that generates a query corresponding to the missing task attributes based on the current dialogue context and task objectives. The input to this prompt is the dialogue context history and the generated task goal. This query is then used as input to the memory judger to assist in selecting QA memory units that align with the task objectives.

# Prompts of Confirmation Response Generation

In the evaluation process, we employed a confirmationtype response generation approach to assess the integration performance of multi-session memory in task-oriented dialogues. We designed the prompt as shown in Figure A.3, which leverages the dialogue context, task objectives, and activated memory units to generate responses.

# Prompts of GPT4 Evaluation

During the evaluation process, we employed a GPT-4 prompt (as shown in Figure A.5) to assess the quality of confirmation-type responses. This prompt evaluates the response holistically from four perspectives: requirement alignment, content accuracy, language quality, and comparison to the reference answer. The input to this prompt includes the dialogue history, task objectives, the reference response, and the model-generated response. This design ensures that the evaluation of the response is not solely based on the dataset’s reference reply but also takes into account multiple factors such as whether the task objectives are met and the overall quality of the response. Such an evaluation approach is more comprehensive.

# Prompts of Dialogue State Tracking

we used a prompt modified from (Heck et al. 2023) (as shown in Figure A.4) that generates the dialogue state for each user turn in the dialogue. Let

$$
A _ {1} = P \oplus \text {s y s t e m}: M _ {1} \oplus \text {u s e r}: U _ {1}
$$

$$
A _ {t} = A _ {t - 1} \oplus s y s t e m: M _ {t} \oplus \mathrm {u s e r}: U _ {t}, \quad \forall t \in [ 2, T ]
$$

where P is the task description which provides the model with instructions for how to process a dialogue between a system M and a user U. In contrast to (Heck et al. 2023), P does not include the detailed description for slots to challenge ChatGPT’s ability to understand the meaning of the slots. Apart from that, ChatGPT often generated answers with excessively detailed explanations, deviating from the

Table 7: MS-TOD Training Dataset Statistics for Memory Fileter.   

<table><tr><td>Attribute</td><td>Train</td></tr><tr><td>Domains</td><td>16</td></tr><tr><td>Intentions</td><td>22</td></tr><tr><td>Task goals</td><td>4,534</td></tr><tr><td>Dialogues</td><td>13,441</td></tr><tr><td>Utterances</td><td>89,152</td></tr><tr><td>Avg. slots per task goal</td><td>4.49</td></tr><tr><td>Number of individuals</td><td>565</td></tr><tr><td>Avg. intentions per individual</td><td>6.24</td></tr><tr><td>Avg. sessions per individual</td><td>23.79</td></tr><tr><td>Avg. Utterances per individual</td><td>157.80</td></tr></table>

expected response format. To address this issue, a prompt that includes ”No explanation!” as an instruction to Chat-GPT not to provide detailed explanations was introduced (Feng et al. 2023) and we added this to our prompt.

# Appendix B

# Training Dataset for Memory Filter

To ensure that the memory filter generalizes across different domains and scenarios, we generated the training dataset(as shown in Table 7) using the same method described in the main text. The dataset spans 16 domains, 4,534 task goals, and 13,411 dialogues, involving a total of 565 individuals, each with an average of 6.24 intentions. Beyond training the memory filter, this dataset can also serve as an alternative evaluation set for broader benchmarking.

# Dataset Structure

MS-TOD encompasses multiple individual task-oriented dialogue datasets, each consisting of several sessions. We present an example of one session (as shown in Figure A.6) from an individual. This session includes a session id, where a larger value indicates a more recent timestamp. The domain represents the specific field or area of the dialogue. The reference dialogue id corresponds to the dialogue id in the original SGD dataset that shares the same task objective. The exist conf irmation indicates whether the session contains a confirmation-type response and whether it is an evaluation target. The intent represents the specific purpose or goal of the dialogue. The content stores the actual dialogue text. The task goal includes task slots and their corresponding attribute values. Each individual contains dozens of session data structured as described above.

# Human Validation Protocol

To ensure the realism, coherence, and usability of MS-TOD, we apply a structured human validation process during dataset construction. This process involves three research assistants with prior experience in natural language

Table 8: Comparison of MS-TOD with representative Task Oriented Dialogue (TOD) and Open Domain (OD) datasets along memory-related attributes.   

<table><tr><td>Dataset</td><td>Task Type</td><td>Multi-Session?</td><td>GroundedMemory?</td><td>UserIntention?</td><td>RetrievalSupport?</td><td>MemoryFormat</td></tr><tr><td>MULTIWOZ (Hu et al. 2023)</td><td>TOD</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>-</td></tr><tr><td>SGD (Rastogi et al. 2020)</td><td>TOD</td><td>X</td><td>✓</td><td>✓</td><td>X</td><td>schema</td></tr><tr><td>TOAD (Liu et al. 2024b)</td><td>TOD</td><td>X</td><td>X</td><td>✓</td><td>X</td><td>-</td></tr><tr><td>LUCID (Stacey et al. 2024)</td><td>TOD</td><td>X</td><td>✓</td><td>✓</td><td>X</td><td>latent goal</td></tr><tr><td>MSC (Xu, Szlam, and Weston 2022)</td><td>OD</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>dialogue history</td></tr><tr><td>CC (Jang, Boo, and Kim 2023)</td><td>OD</td><td>✓</td><td>✓</td><td>X</td><td>✓</td><td>persona/dialogue history</td></tr><tr><td>MEMORYBANK (Zhong et al. 2024)</td><td>OD</td><td>✓</td><td>✓</td><td>X</td><td>✓</td><td>dialogue history</td></tr><tr><td>LOCOMO (Maharana et al. 2024)</td><td>OD</td><td>✓</td><td>✓</td><td>X</td><td>✓</td><td>dialogue history</td></tr><tr><td>LONGMEMEVAL (Wu et al. 2025)</td><td>OD</td><td>✓</td><td>✓</td><td>X</td><td>✓</td><td>dialogue history</td></tr><tr><td>MS-TOD (ours)</td><td>TOD</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>qa memory/dialogue history</td></tr></table>

processing and dialogue systems. The validation pipeline includes the following stages:

1. Intent and Slot Accuracy Check. For each dialogue turn derived from the SGD intent annotations, annotators verify whether the intent is preserved and whether all required slot values are present and semantically correct.   
2. Redundancy Removal. Annotators manually review and remove multi-session dialogues that contain excessive repetition across sessions, which could undermine diversity and realism.   
3. Confirmation Accuracy Validation. For final-session confirmation-type utterances, annotators examine whether the confirmed slot values align with the task goal. Mismatched, ambiguous, or hallucinated confirmations are flagged and discarded.   
4. Dialogue Coherence Filtering. Dialogues that fail to complete any defined task goal are considered incoherent. Sessions missing necessary confirmation-type turns are also excluded to ensure logical task flow.   
5. Intent Redundancy Filtering. Episodes exhibiting unnatural repetition of similar intents across turns or sessions are excluded, as such patterns deviate from realistic multi-session user behavior.

This multi-stage quality control procedure yields a filtered evaluation subset used for system benchmarking. The validation process ensures that the dataset aligns with realistic task-oriented dialogue patterns and supports the evaluation of multi-session memory-aware dialogue systems.

# Intent-driven QA Memory

For each historical session, we generated an intent description and the corresponding QA memory (as shown in Figure A.7) for the objectives of that intent description. The QA memory consists of multiple QA pairs, where each query is a question about a task attribute under that intent, and the answer is the slot value corresponding to that task attribute.

# Dataset Design Rationale

Choice of Seed Dataset. We select the Schema-Guided Dialogue (SGD) dataset as the foundation for constructing

MS-TOD. Compared to other popular benchmarks like MultiWOZ, SGD provides broader domain coverage, a larger and more diverse set of user intents, and a schema-driven annotation format that supports extensibility and dynamic intent representation. These characteristics make SGD more suitable for modeling realistic, multi-domain, and multisession interactions. A detailed comparison is shown in Table 9.

Design of Memory Bank Structure. Each MS-TOD memory bank contains 20 sessions involving more than six distinct user intents. This structure is informed by two factors. First, prior multi-session datasets such as LoCoMo () typically use memory segments with $^ { 2 0 + }$ sessions, providing a reference for session scale under long-term memory settings. Second, based on analysis of the SGD schema, each user intent generally corresponds to fewer than 10 slot types. Organizing 3 sessions per intent enables natural progression while minimizing redundancy. As a result, grouping 6–8 distinct intents yields a total of around 20 sessions per memory bank, balancing diversity, realism, and memory demand.

# Appendix C

# Memory Retrieval Comparision

Table 11 compares the performance of different retrieval methods. text-embed3-small achieves the highest recall across all thresholds, with 0.702 at Recall $\textcircled { \alpha } 3$ , 0.792 at Recall $\textcircled { \alpha } 5$ , and 0.905 at Recall $@ 1 0$ , demonstrating superior retrieval capability. Among other models, nv-embedv2 and bge-large-en-v1.5 also perform well, while traditional retrieval methods like BM25 remain competitive at Recall $@ 1 0$ but lag behind embedding-based methods at lower recall levels. T5-base and BERT-based models exhibit lower recall, suggesting that general pre-trained models are less effective for specialized memory retrieval. These results highlight text-embed3-small as the most effective choice for long-term memory activation in multi-session dialogues.

# Effectiveness of the Proactive Response Strategy

To better understand the impact of the proactive response strategy, we present a complementary analysis that exam-

Table 9: Comparison between MultiWOZ and SGD datasets.   

<table><tr><td>Dimension</td><td>MultiWOZ</td><td>SGD</td></tr><tr><td># Domains</td><td>7</td><td>20</td></tr><tr><td>Avg. Intents per Domain</td><td>8–10</td><td>10–15</td></tr><tr><td>Total Intents</td><td>~60</td><td>~200</td></tr><tr><td>Annotation Structure</td><td>Fixed, manually updated</td><td>Schema-driven, extensible</td></tr><tr><td>Cross-Domain Intent Interaction</td><td>Limited (2–3 domain combos)</td><td>Rich (multi-domain intent chains)</td></tr></table>

Table 10: Effectiveness of the Proactive Response Strategy. Slot accuracy is measured before correction, and task rate reflects the final success after proactive clarification.   

<table><tr><td>Model</td><td>Slot Acc. (Pre)</td><td>Task Rate (Post)</td></tr><tr><td>LLaMA3-8B</td><td>0.62</td><td>0.92</td></tr><tr><td>Qwen-7B</td><td>0.48</td><td>0.87</td></tr><tr><td>Mistral-7B</td><td>0.59</td><td>1.00</td></tr><tr><td>GPT4o-mini</td><td>0.61</td><td>0.99</td></tr></table>

ines two distinct metrics: slot accuracy measured during the confirmation phase and the final task success rate. Although these metrics reflect different aspects of system performance—localized slot-level correctness versus overall goal completion—they jointly capture the effectiveness of proactive correction.

As shown in Table 10, slot accuracy remains relatively low (ranging from 0.48 to 0.62) before correction, indicating frequent omission or mismatch in predicted slot values. Nevertheless, the final task success rates reach 0.87 or higher across all models after proactive correction is applied. This pattern suggests that the proactive response strategy plays a critical role in bridging the gap between partial slot-level understanding and complete task execution by enabling the system to recover from intermediate errors through user interaction.

# Human Evaluation Details

Table 14 presents the results of human evaluation, including accuracy, informativeness, and coherency scores. Accuracy is rated on a scale of 0 to 1, while informativeness and coherency are rated from 0 to 3. The average scores in Table 3 are computed using a weighted sum with weights of 1, 1/3, and 1/3. All evaluations were conducted in a blind review manner to compare the response quality of FCP and MemGuide. Additionally, the Confirmation-type Response type assesses the response quality after memory-guided dialogue planning, while the multi-turn evaluation focuses on dialogues under the proactive response strategy, continuing until task completion or forced termination.

# Additional Evaluation Metrics

Table 12 compares the performance of task-oriented dialogue models with and without memory-augmented processing (MemGuide) across Slot Accuracy, BLEU, and ROUGE metrics. The results reveal a trade-off between

Table 11: Performance evaluation of activation modules on memory retrieval   

<table><tr><td>Activation Module</td><td>Recall@3</td><td>Recall@5</td><td>Recall@10</td></tr><tr><td>bm25</td><td>0.642</td><td>0.721</td><td>0.842</td></tr><tr><td>t5-base</td><td>0.443</td><td>0.575</td><td>0.773</td></tr><tr><td>bert-base</td><td>0.463</td><td>0.584</td><td>0.785</td></tr><tr><td>bert-large</td><td>0.401</td><td>0.530</td><td>0.730</td></tr><tr><td>nv-embed-v2</td><td>0.668</td><td>0.769</td><td>0.896</td></tr><tr><td>bge-large-en-v1.5</td><td>0.681</td><td>0.761</td><td>0.888</td></tr><tr><td>text-embed3-small</td><td>0.702</td><td>0.792</td><td>0.905</td></tr></table>

Table 12: Performance comparison of task-oriented dialogue models with and without long-term memory integration: Slot Accuracy, BLEU, and ROUGE metrics.   

<table><tr><td>Model</td><td>Setting</td><td>Slot Accuracy</td><td>BLEU</td><td>ROUGE</td></tr><tr><td rowspan="2">LLaMA3-8B</td><td>FCP</td><td>0.62</td><td>10.47</td><td>28.59</td></tr><tr><td>MemGuide</td><td>0.56</td><td>9.86</td><td>30.39</td></tr><tr><td rowspan="2">Qwen-7B</td><td>FCP</td><td>0.48</td><td>10.33</td><td>29.77</td></tr><tr><td>MemGuide</td><td>0.55</td><td>10.90</td><td>31.28</td></tr><tr><td rowspan="2">Mistral-7B</td><td>FCP</td><td>0.59</td><td>10.09</td><td>28.42</td></tr><tr><td>MemGuide</td><td>0.56</td><td>6.66</td><td>24.64</td></tr><tr><td rowspan="2">GPT4o-mini</td><td>FCP</td><td>0.61</td><td>20.30</td><td>43.49</td></tr><tr><td>MemGuide</td><td>0.68</td><td>13.6</td><td>35.20</td></tr></table>

Table 13: Performance comparison on Slot Accuracy, BLEU, and ROUGE.   

<table><tr><td>Model</td><td>Slot Accuracy</td><td>BLEU</td><td>ROUGE</td></tr><tr><td>AutoTOD</td><td>0.61</td><td>3.34</td><td>24.07</td></tr><tr><td>MemGuide</td><td>0.68</td><td>5.47</td><td>25.03</td></tr></table>

structured slot accuracy and response fluency. In most models, MemGuide slightly reduces slot accuracy, as seen in LLaMA3-8B, which drops from 0.62 to 0.56, and Mistral-7B, which decreases from 0.59 to 0.56. However, GPT4omini benefits from MemGuide, achieving the highest slot accuracy of 0.68. BLEU scores generally decline, suggesting that MemGuide shifts responses away from verbatim accuracy towards greater contextual adaptability. Mistral-7B drops from 10.90 to 6.66, and LLaMA3-8B decreases from

Table 14: Comparison of different models on human evaluation metrics: accuracy, informativeness, and coherence. The results are presented for both confirmation-type responses and multi-turn dialogue settings, comparing standard inference (‘FCP‘) with memory-augmented processing (‘MemGuide‘).   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Setting</td><td colspan="3">Confirmation-type Response</td><td colspan="3">Multi-Turn</td></tr><tr><td>Accuracy</td><td>Informativeness</td><td>Coherency</td><td>Accuracy</td><td>Informativeness</td><td>Coherency</td></tr><tr><td rowspan="2">GPT4o-mini</td><td>FCP</td><td>0.62</td><td>1.83</td><td>1.90</td><td>0.81</td><td>1.92</td><td>2.44</td></tr><tr><td>MemGuide</td><td>0.65</td><td>2.38</td><td>2.48</td><td>0.87</td><td>1.93</td><td>2.74</td></tr><tr><td rowspan="2">LLaMA</td><td>FCP</td><td>0.56</td><td>1.47</td><td>1.74</td><td>0.78</td><td>1.64</td><td>2.36</td></tr><tr><td>MemGuide</td><td>0.61</td><td>1.98</td><td>2.16</td><td>0.88</td><td>2.51</td><td>2.71</td></tr><tr><td rowspan="2">Qwen</td><td>FCP</td><td>0.43</td><td>1.24</td><td>1.85</td><td>0.82</td><td>1.60</td><td>2.02</td></tr><tr><td>MemGuide</td><td>0.54</td><td>1.70</td><td>2.30</td><td>0.92</td><td>1.93</td><td>2.47</td></tr><tr><td rowspan="2">Mistral</td><td>FCP</td><td>0.58</td><td>1.63</td><td>1.99</td><td>0.89</td><td>2.49</td><td>2.72</td></tr><tr><td>MemGuide</td><td>0.61</td><td>2.06</td><td>2.08</td><td>0.93</td><td>2.74</td><td>2.85</td></tr></table>

10.47 to 9.86. Conversely, ROUGE scores improve with MemGuide in several cases. LLaMA3-8B increases from 28.59 to 30.39, and Qwen-7B rises from 29.77 to 31.28, indicating enhanced informativeness and coherence. However, Mistral-7B experiences a slight decrease in ROUGE from 28.42 to 24.64. Overall, the results suggest that MemGuide enhances response informativeness while slightly compromising slot accuracy and BLEU, highlighting a trade-off between structured information retention and more natural, contextually aware responses.

Table 13 presents the performance comparison between AutoTOD and MemGuide on Slot Accuracy, BLEU, and ROUGE. The results indicate that MemGuide consistently outperforms AutoTOD across all three metrics, demonstrating its effectiveness in enhancing dialogue quality. Slot Accuracy improves from 0.61 to 0.68, indicating better tracking of task-specific information. BLEU increases from 3.34 to 5.47, reflecting more precise and fluent responses. ROUGE also shows a slight improvement, rising from 24.07 to 25.03, suggesting that MemGuide enhances informativeness and coherence. These results highlight the advantages of memory-augmented processing, which enables more accurate and contextually relevant dialogue generation.

# Appendix D

# Multi-session Dialogue Context Comparison

Figure ?? presents four different configurations of conversation contexts not shown in the main paper. Specifically, (1) Full conversation history includes every session from the dialogue history as prompt input to the reader. (2) Retrievalbased methods retrieve the dialogue sessions most relevant to the current session (Session 23) and append them to the reader’s context (3) Retrieving a summary compiles a summary of past sessions (Sessions 1 to 22) for inclusion alongside the current context. Finally, (4) MemGuide integrates QA memory with the Session 23 context to generate responses. By illustrating these detailed contexts, Figure ?? provides further insights into how each approach manages multi-session dialogue.

# MemGuide vs. RAG

To better understand how CoT reasoning and memory reranking affect confirmation response generation, we present a step-by-step case study comparing MemGuide and standard RAG (Appendix Table 15). In this example, the user attempts to confirm a restaurant reservation. While both systems retrieve similar QA memory candidates, the standard RAG model fails to detect missing slot information (e.g., number of people), resulting in an incomplete and partially inaccurate response. In contrast, MemGuide use Chain-of-Thought explicitly identifies missing task information (e.g., time, headcount) through reasoning, refines the retrieved memories via the Memory Judger, and generates a more complete and contextually appropriate confirmation. This illustrates how structured reasoning and selective memory grounding improve slot coverage and reduce factual errors in multi-turn dialogue.

# Prompts of the Dataset Generation

# User Prompt:

Help me generate an English conversation under the {dialogue intent} intent, where {task goal}. The conversation should be between a user and an assistant, and it should be split into {task goal length} sessions at different points in time, with continuity and connection between the sessions and each session should not less than 6 turns. Additionally, the final session must include a assistant response containing a complete confirmation-type utterance before the user confirms, and this utterance should be marked with ‘is confirmation‘ set to ‘True‘. and the user must provide a final confirmation response at the end of the final session. For all other sessions, the conversation should end with an assistant’s polite declarative statement.

# System Prompt:

””” You are dialogue generator assistant.

The sessions should be clearly separated, and the conversation should be formatted as follows:

Each turn should be a dictionary entry.

The conversation should be in the format of a list of sessions, where each session is a list of dictionaries representing each turn.

Each dictionary entry should have two keys: speaker (either ’user’ or ’assistant’) and text (the spoken dialogue).

Except for final session, each session should be a seperate dialogue and include a complete dialogue structure, beginning with a greeting from the user and ending with an assistant’s polite declarative statement.

Feel free to expand the dialogue with additional relevant details, but avoid redundant expressions or repeating the same phrases.

Reponse me with a json format

```jsonl
1 {   
2 "sessions": [   
3 [   
4 {   
5 "speaker": "xx",   
6 "text": "xx"   
7 },   
8 {   
9 "speaker": "xx",   
10 "text": "xx"   
11 }   
12 ]   
13 ]   
14 } 
```

Figure A.1: Prompts of the Dataset Generation

# Prompts of the Task Slot Querying Generation

Please help me generate questions, based on the provided {conversation history}, that correspond to unanswered attributes in the task goal {task attributes}.

1. The questions should start with ’What,’ ’When,’ ’Why,’ ’How,’ or ’Where.’   
2. Ensure that the generated questions are in third person.   
fill the following json: { [Question], }

Figure A.2: Prompts of the Task Slot Querying Generation

# Prompts of Confirmation Response Generation

””” You are an dialogue assistant.

Generate a confirmation response based on the users utterance. Include any relevant task goals [TASK GOALS] ´ identified in the dialogue or related memory [MEMORY]. If [MEMORY] is unavailable, construct your response accurately and comprehensively using the provided conversation details. Ensure your reply acknowledges the users´ request clearly and incorporates relevant information from both the dialogue and the related memory units [MEMORY]. [TASK GOAL] {task goal} [MEMORY] {memory}

Figure A.3: Prompt of Confirmation Response Generation

Table 15: Step-by-step comparison of MemGuide vs. standard RAG in confirmation response generation.   

<table><tr><td>Process</td><td>MemGuide</td><td>RAG</td></tr><tr><td colspan="3">Input and Intent</td></tr><tr><td>Dialogue History</td><td>User: Have you completed the reservation at Gen Korean BBQ House?</td><td>User: Have you completed the reservation at Gen Korean BBQ House?</td></tr><tr><td>Intention Description</td><td>The user wants to confirm restaurant reservation.</td><td>The user wants to confirm restaurant reservation.</td></tr><tr><td colspan="3">Memory Retrieval</td></tr><tr><td>Retrieved QA Memory Candidates</td><td colspan="2">Rank 1:
·Q: What is the time of the reservation? A: March 1st
·Q: What is the address of the reservation? A: Los Angeles
Rank 2:
·Q: What is the time of reservation? A: March 4th
·Q: How many people are there? A: 2
·Q: What is the address of the restaurant? A: Gen Korean BBQ House in Milpitas
·Q: What is the time of reservation? A: March 1st
·Q: What is the address of the restaurant? A: Gen Korean BBQ House in Milpitas</td></tr><tr><td colspan="3">CoT Reasoning</td></tr><tr><td>Task Goal</td><td>Reserve Restaurant</td><td>—</td></tr><tr><td>Missing Slots</td><td>Time, Number of people</td><td>—</td></tr><tr><td>Missing Query</td><td>When is the time of reservation? How many pe- ple are there?</td><td>—</td></tr><tr><td colspan="3">Memory Filter (Reranking)</td></tr><tr><td>Reranked Memory Units</td><td>·Q: What is the time of reservation? A: March 4th
·Q: How many people are there? A: 2
·Q: What is the address of the restaurant? A: Gen Korean BBQ House in Milpitas</td><td>Same as retrieved</td></tr><tr><td colspan="3">Refinement-grounded Response Generation</td></tr><tr><td>Confirmation Response</td><td>Just to confirm, it&#x27;s a reservation for 2 at Gen Korean BBQ House in Milpitas on March 4th at 12:15 pm, with a request for a quieter table. Is that correct?</td><td>To confirm, it&#x27;s a reservation for 2 at Gen Korean BBQ House Los Angeles on March 1st. Is that correct?</td></tr></table>

”””Consider the following list of concepts, called ”slots” provided to you as a json list.

Prompt of Dialogue State Tracking on MultiWOZ 2.2   
```txt
"slots": {
    "attraction-area",
    "attraction-name",
    "attraction-type",
    "bus-day",
    "bus-departure",
    "bus-destination",
    "bus-leaveat",
    "hospital-department",
    "hotel-area",
    "hotel-bookday",
    "hotel-bookpeople",
    "hotel-bookstay",
    "hotel-internet",
    "hotel-name",
    "hotel-parking",
    "hotel-pricerange",
    "hotel-stars",
    "hotel-type",
    "restaurant-area",
    "restaurant-bookday",
    "restaurant-bookpeople",
    "restaurant-booktime",
    "restaurant-food",
    "restaurant-name",
    "restaurant-pricerange",
    "taxi-arriveby",
    "taxi-departure",
    "taxi-destination",
    "taxi-leaveat",
    "train-arriveby",
    "train-bookpeople",
    "train-day",
    "train-departure",
    "train-destination",
    "train-leaveat",
} 
```

Now consider the following dialogue between two parties called the ”system” and ”user”. Can you tell me which of the ”slots” were updated by the ”user” in its latest response to the ”system”? Present the updates in JSON format. If no ”slots” were updated, return an empty JSON list. If you encounter ”slots” that were requested by the ”user” then fill them with ”?”. If the user informed that he did not care about a ”slot”, fill it with ”dontcare”. Return the output in JSON format and no explanation!

```txt
{dialogue} 
```

Figure A.4: Prompt of Dialogue State Tracking on MultiWOZ 2.2

# Prompts of GPT4 Evaluation

””” You are a strict and objective evaluator. Your task is to assess the quality of the final predicted response using the provided conversation context, the user’s target goal attributes, and a reference answer. Your evaluation should be fair, professional, and reflect an expert judgment of the response’s quality.

```txt
[Dialogue Context]  
{{conversation history}}  
[Task Goal]  
{{task_goal}}  
[reference_answer]  
{{reference_answer}}  
[predict_answer]  
{{predict_answer}}  
Evaluation Criteria: 
```

Requirement Alignment: Does the final predict answer meet the user’s task goal?

Content Accuracy: Is the information in the final response correct, clear, and logically organized?

Language Quality: Is the language fluent, coherent, and readable? Are there any obvious grammatical or word choice errors?

Comparison to Reference Answer: Compared to the reference answer, how does the final response differ in terms of completeness, professionalism, and clarity?

Overall Score: Assign a score from 1 to 10 (10 being the best), considering all of the above factors.

The evaluation must be structured in the following JSON format:

```snap
```
```
```
```
```
```
```
```
```
```
```
```
`` 
```

Figure A.5: Prompts of GPT4 Evaluation

MS-TOD dialogue session structure   
```jsonl
1 {   
2 "session_id": 9,   
3 "domain": "Travel",   
4 "referencedinologue_id": "66_00101",   
5 "exist_confomation": true,   
6 "intent": "ReserveHotel",   
7 "content": [   
8 {   
9 "speaker": "user",   
10 "utterance": "Hi again, I'm ready to finalize the booking for Aloft Portland Airport At Cascade Station.",   
11 "is Confirmation": false   
12 },   
13 {   
14 "speaker": "assistant",   
15 "utterance": "Just to confirm, you are booking 1 room at Aloft Portland Airport At Cascade Station, Portland, from March 5th to March 7th. The room is a standard king room with free Wi-Fi and a 24-hour cancellation policy. Is that correct?",   
16 "is Confirmation": true   
17 },   
18 {   
19 "speaker": "user",   
20 "utterance": "Yes, that is correct.",   
21 "is Confirmation": false   
22 },   
23 {   
24 "speaker": "assistant",   
25 "utterance": "Excellent! Your room has been successfully booked. You will receive a confirmation email shortly.",   
26 "is Confirmation": false   
27 },   
28 {   
29 "speaker": "user",   
30 "utterance": "Thank you so much for your help!",   
31 "is Confirmation": false   
32 },   
33 {   
34 "speaker": "assistant",   
35 "utterance": "You're welcome! Have a great stay in Portland.",   
36 "is Confirmation": false   
37 }   
38 ],   
39 "task_goal": {   
40 "hotel_name": "Aloft Portland Airport At Cascade Station",   
41 "location": "Portland",   
42 "check_in_date": "March 5th",   
43 "check_out_date": "March 7th",   
44 "number_of Rooms": 1   
45 }   
46 } 
```

Figure A.6: MS-TOD Session Structure.

# MS-TOD Intent Description and QA Memory

```jsonl
1 {   
2 "9":{   
3 "intent_description": "The user's intent is to finalize and confirm a hotel booking for a specific room at Aloft Portland Airport At Cascade Station, including details about the stay dates and room type.",   
4 "qa_summery":[   
5 {   
6 "Question": "What type of room did the user book?",   
7 "Answer": "The user booked a standard king room."   
8 },   
9 {   
10 "Question": "When is the user's reservation?",   
11 "Answer": "The user's reservation is from March 5th to March 7th."   
12 },   
13 {   
14 "Question": "Where is the user's reservation located?",   
15 "Answer": "The user's reservation is located at Aloft Portland Airport At Cascade Station."   
16 },   
17 {   
18 "Question": "What amenities are included in the user's reservation?",   
19 "Answer": "The user's reservation includes free Wi-Fi."   
20 },   
21 {   
22 "Question": "What is the cancellation policy for the user's booking?",   
23 "Answer": "The cancellation policy for the user's booking is 24 hours."   
24 }   
25 ]   
26 }   
27 } 
```

Figure A.7: Intent description and QA Memory in MT-TOD.
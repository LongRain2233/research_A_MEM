# FINMEM: A Performance-Enhanced LLM Trading Agent With Layered Memory and Character Design

Yangyang $\mathrm { Y u } ^ { * } \{ \varnothing \}$ , Haohang Li∗ , Zhi Chen∗ , Yuechen Jiang∗ , Yang Li∗ , Jordan W. Suchow ， Denghui Zhang , and Khaldoun Khashanah

Abstract—We introduce FINMEM, a novel Large Language Models (LLM)-based agent framework for financial trading, designed to address the need for automated systems that can transform realtime data into executable decisions. FINMEM comprises three core modules: Profile for customizing agent characteristics, Memory for hierarchical financial data assimilation, and Decision-making for converting insights into investment choices. The Memory module, which mimics human traders’ cognitive structure, offers interpretability and real-time tuning while handling the critical timing of various information types. It employs a layered approach to process and prioritize data based on its timeliness and relevance, ensuring that the most recent and impactful information is given appropriate weight in decision-making. FINMEM’s adjustable cognitive span allows retention of critical information beyond human limits, enabling it to balance historical patterns with current market dynamics. This framework facilitates self-evolution of professional knowledge, agile reactions to investment cues, and continuous refinement of trading decisions in financial environments. When compared against advanced algorithmic agents using a largescale real-world financial dataset, FINMEM demonstrates superior performance across classic metrics like Cumulative Return and Sharpe ratio. Further tuning of the agent’s perceptual span and character setting enhances its trading performance, positioning FINMEM as a cutting-edge solution for automated trading.

Index Terms—Financial AI, large language models, trading algorithms, deep learning, financial technology.

# I. INTRODUCTION

W ITH the influx of diverse financial data streams from theweb, traders face a deluge of information from various web, traders face a deluge of information from various sources. This requires them rapidly to understand, memorize, and filtrate crucial events for investment decisions. However, innate cognitive limitations restrict human traders from processing information within their perception and memory capacity, a span much narrower than the actual volume of available information [1]. Consequently, insufficiently considering or even dismissing critical events affecting trading decisions becomes

Received 1 December 2023; revised 26 September 2024; accepted 17 July 2025. Date of publication 4 August 2025; date of current version 13 November 2025. Recommended for acceptance by G. Yang. ( ∗Yangyang Yu, Haohang Li, Zhi Chen, Yuechen Jiang, and Yang Li contributed equally to this work.) (Corresponding author: Yangyang Yu.)

The authors are with the Stevens Institute of Technology, Hoboken, NJ 07030 USA (e-mail: yyu44@stevens.edu; hli113@stevens.edu; zchen100@stevens. edu; yjiang52@stevens.edu; yli269@stevens.edu; jws@stevens.edu; dzhang $4 2 @$ stevens.edu; kkhashan@stevens.edu).

This article has supplementary downloadable material available at https://doi.org/10.1109/TBDATA.2025.3593370, provided by the authors.

Digital Object Identifier 10.1109/TBDATA.2025.3593370

increasingly concerning as data availability expands. To overcome the physical limitations in the memory systems of human traders, researchers have been consistently working on designing autonomous trading agent systems. These systems need to thoroughly integrate all available information and possess a sophisticated design in the agent’s backbone algorithm to deliver enhanced trading performance.

The evolution of autonomous trading systems has transitioned from the initial rule-based trading strategies [2] to more advanced machine-learning-based algorithms [3]. In recent years, Reinforcement Learning (RL)-based agents [4], especially those employing Deep Reinforcement Learning (DRL) [5] as backbone algorithms, garner joint attention of both academia and industry. Leveraging both RL principles and deep learning, DRL agents effectively handle and learn from scalable and diverse financial data, including stock prices, key financial indicators, and market sentiments. They utilize deep neural networks to extract expressive features from input data, representing the complex financial market environment, which enhances their comprehension ability. Retaining the key features of RL agents, they learn through interaction with a predefined environment to maximize investment gain over time. Research suggests DRLs can meet the crucial needs of trading agents to process and make informed decisions from large volumes of data. However, certain inherent features of DRL algorithms exhibit notable deficiencies in financial applications. Firstly, DRL agents exhibit a lack of interpretability concerning the rationale behind their decisions [6]. They are often described as “Closed system,” where the internal processes and computational layers leading to a specific decision are neither easily understandable nor transparent. Secondly, DRL agents find it challenging to effectively integrate textual data with numerical features. Text data plays a vital role in finance, since the majority of market information is conveyed through news articles and financial reports. However, transforming text data to embeddings considerably increases input space dimensionality, making learning more computationally demanding. Plus, empirical studies have shown that combining textual representations with numerical financial indicators often leads to convergence challenges [7]. Therefore, a backbone algorithm with transparent reasoning and the enhanced ability to capture investment-related textual insights comprehensively is essential.

Recent advancements in Large Language Models (LLMs), like Generative Pre-trained Transformers (GPTs) [8], offer viable solutions for developing trading agents, alleviating previous

concerns. With carefully designed prompts, the LLM-based agent is able to provide reasons and outcomes in plain text. This allows for immediate observation and prompt adjustment of its reasoning process. Employing LLMs as the backbone algorithms for agents also overcomes the constraint of isolated environments. This is achieved through their vast pre-existing knowledge and the effective integration of valuable insights from a variety of data sources, including both textual and numerical. When equipped with suitable prompt templates, this approach significantly enhances decision-making capabilities [9]. Studies indicate that prompt-guided reasoning significantly improves problem-solving rates across various domains [10]. Notably, a growing body of research has focused on utilizing LLMs to make informed trading decisions for stocks and funds by continuously interacting with financial environment information [11], [12]. However, in these approaches, LLMs primarily serve in a question-answering (QA) capacity rather than functioning as autonomous agents. Despite outperforming traditional trading benchmarks, they generally process information indiscriminately through QA iterations, lacking the ability to prioritize and retain crucial information. Moreover, these approaches heavily rely on the uncertain and resource-intensive process of LLM fine-tuning. These LLM-based methods often oversimplify the complex cognitive functions essential to traders. For instance, they struggle to recognize the varying relevance and timeliness of different data types or to dynamically adjust risk tolerance based on changing market conditions. Consequently, they may fail to effectively prioritize and retain critical information or events in their decision-making process. In scenarios requiring sequential decision-making, such as stock trading in a volatile environment, these limitations can lead to suboptimal performance. The inability to maintain a nuanced understanding of market dynamics and adapt strategies accordingly hampers the LLMs’ effectiveness in replicating the sophisticated decisionmaking processes of human traders.

We introduce FINMEM, a novel LLM-based autonomous trading agent. We claim that it bridges these existing research gaps in the following aspects:

- FINMEM provides a language agent whose reasoning and decision-making processes are completely transparent, interpreted through plain texts. This significantly enhances the interpretability of the rationale behind its recommended investment actions.   
- FINMEM efficiently produces high-quality stock trading decisions by effectively integrating numeric and textual data, requiring significantly less training time and data volume. This method provides a robust solution for the automated trading of stocks from recently IPOed companies.   
- FINMEM flexibly adapts to market volatility by offering a self-adaptive character setting in its profile module. FIN-MEM’s profile module includes a dynamic character setting feature, offering seed information about professional backgrounds and adjustable risk inclinations. Additionally, it continuously updates domain knowledge as the trading experience grows, which reinforces FINMEM’s professional experience to further augment its decision-making quality.

- FINMEM firstly introduces a cutting-edge LLM-based trading agent framework equipped with a human-aligned memory mechanism. This innovative approach mimics the cognitive architecture of human traders, featuring working and layered long-term memory mechanisms. This design excels at capturing essential investment insights from diverse financial data characterized by multi-timeliness, thereby enhancing trading performance. Beyond a simple similarity-based search, FINMEM retrieves information using three key metrics: similarity, recency, and importance. The latter two metrics are maintained using distinct decay ratios for each layer, enhancing the system’s ability to prioritize and access relevant information across various time scales.

- FINMEM leverages its distinctive features to expand the agent’s perceptual range beyond the human limitation to make well-informed trading decisions. Cognitive research shows that human working memory can recall only five to nine events at a time [13]. This limitation, while preventing information overload, can yield insufficient insight for precise decision-making. In contrast, FINMEM’s memory module transcends this constraint. It allows adjusting cognitive load by selecting a flexible number of top-ranked events from each layer of its hierarchical long-term memory, allowing FINMEM to maintain agility and deliver superior trading decisions in data-rich contexts.

Inspired by the modular design of the classic generative agent framework by Park et al. [14], FINMEM introduces innovative profile and memory modules. The profile module provides FIN-MEM with a trade-specific professional background and a selfadaptive risk option, sharpening its trading focus and enhancing resilience to market fluctuations. Moreover, FINMEM’s memory module, featuring working and layered long-term memory, is tailored for processing information based on its importance and timeliness. These innovations address the challenges in Park et al.’s framework regarding the accurate interpretation of timesensitive and critical data. They enhance FINMEM’s ability to distinguish data timeliness, optimize information retrieval, and offer insightful feedback for refining future decisions, making it exceptionally suitable for financial decision-making contexts. FINMEMs working memory acts as a dynamic “workspace,” ´ enabling operations like summarization, observation, and reflection on multi-source information to facilitate trading decisions. Its long-term memory, structured into shallow, intermediate, and deep layers [15], manages varied decay rates to satisfy the need to retain distinct types of financial information within different time scales based on their corresponding timeliness. For instance, daily news, with its immediate effects on stock markets, is channeled into the shallow processing layer. Meanwhile, annual company reports, exerting a more prolonged impact, are processed in the deep layer by FINMEM. Each layer in FINMEM prioritizes memory events based on the assemble of recency, relevancy, and importance close to Park et al.’s method. However, it introduces new measurements for recency and importance, specifically tailored to better rank financial data according to their unique time sensitivity. FINMEM’s memory

mechanism can also transit significantly impactful investment memory events to deeper processing layers, ensuring their retention for extended periods. FINMEM’s memory module can mirror the human cognitive system [16] and facilitate agile, real-time decisions [17]. It enables continuous evolution in professional knowledge through structured summarizing, retrospecting past experiences, and reacting to new trading scenarios. Additionally, FINMEM includes a decision-making module capable of deriving investment decisions by considering top-ranked memory events and current market conditions.

In this paper, we begin by explaining the three core modules of FINMEM. Subsequently, we emphasize its superior trading performance compared to a range of representative algorithmic agents. We further highlight FINMEM’s optimal performance by comparing it with advanced algorithmic-based agent frameworks using widely adapted financial evaluation metrics like cumulative return and Sharpe ratio. We last examine the effectiveness of three critical components of FINMEM—backbone algorithms, working memory capacity, and character settings—in enhancing its decision quality through ablation studies. Through experiments, we show that FINMEM exhibits an outstanding ability to stratify and leverage the various levels of market insights, significantly improving the quality of trading decisions.

# II. RELATED WORK

# A. Deep Learning and Reinforcement Learning for Trading

The development of trading agents has evolved significantly over decades, driven by advancements in technology, finance, and computational methodologies. This evolution progressed from traditional rule-based algorithms operating on predefined sets of rules [18], [19], [20], to statistical learning methods like time series forecasting [21], [22], [23], [24], [25], [26], [27], [28]. It has further advanced to deep learning techniques [29], [30] that enhance predictive decision-making by leveraging fine-grained financial features. Compared to prior approaches, deep learning algorithms more effectively identify nuanced historical market patterns critical for trading decisions, excelling in generating sophisticated, higher-dimensional representations of market features. The implementation of Reinforcement Learning (RL) offers a robust method for developing agent systems tailored for sequential financial decision-making tasks. This approach transcends mere predictive modeling, facilitating a dynamic learning environment where the system continuously adapts to market dynamics through interactions and feedback [31], [32]. Advanced forms of RL, such as Deep Reinforcement Learning (DRL) - including Deep Q-Network (DQN) [33], Advantage Actor-Critic (A2C) [34], and Proximal Policy Optimization (PPO) [35] - further enhance these systems through feedback-driven learning [31], [32]. However, DRL faces challenges such as limited interpretability of decisions [6] and difficulties in incorporating high-dimensional textual information directly. DRL agents often rely on extracting textual sentiment [36], [37], [38], potentially missing other essential market information embedded in news and macroeconomic texts [39], [40].

# B. LLM for Financial Decision-Making

With the emergence of large language models (LLMs), a significant body of research has explored their capabilities for textual summarization and reasoning in complex tasks like financial sentiment analysis and professional knowledge assessment in the financial sector [12], [41], [42], [43], [44]. Subsequently, [11] examined the use of LLMs to summarize investment-related insights and extract financial sentiment from extensive media news through a question-answering (Q&A) framework. This work initially demonstrates the potential of LLMs to generate trading recommendations, showcasing their analytical proficiency. The subsequent study by [45] further explores various cognitive biases that naturally affect human investors in financial decision-making, and shows that these biases also influence the decision-making processes of LLMs [45]. Specifically, risk preference is a critical factor affecting the decision outcomes of LLMs, necessitating careful management.

# C. LLM Agent for Financial Decision-Making

There have been extensive studies demonstrating LLMs agents’ impressive capabilities in various decision-making tasks across various domains [9], [46], [47], [48], [49], [50], [51] such as healthcare and technology service. These works have demonstrated the necessity and advantage of accomplishing complex decision-making by having agent frameworks on top of LLMs to accomplish more complex and sequential decision-making tasks. A few recent studies [44], [52], [53] have tried to design language agent frameworks crafted for decision-making cases in the financial domain, featuring an open-ended and much more volatile environment. Ref. [53]’s approach uses multiple simple LLM agents to generate trading decisions, but this incurs high costs due to extensive coordination. Ref. [52] propose a more efficient single-agent framework with memory and reflection capabilities. However, their oversimplified memory retrieval process may lead to decisions based on outdated market signals. Meanwhile, [44] generate trading recommendations as part of their financial report generation process. However, their framework and evaluation are designed primarily for textual documentation. That is, they lack a specialized agent structure for decision-making tasks. Additionally, their recommended decisions share the frequency with their financial reports, which are typically issued monthly at most. This frequency does not support higher-frequency decision-making needs, such as daily updates. Due to these constraints, they do not offer performance evaluations for their investment recommendation. In summary, given the potential issues to be improved, there is a need to develop a more nuanced framework that better handles multi-timeliness financial data to support more intelligent decision-making.

# III. ARCHITECTURE OF FINMEM

In this section, we comprehensively detail the three core components of FINMEM, namely the profile, memory, and decisionmaking modules. The profile module empowers FINMEM to adaptively tailor character setting for specific trading tasks. Memory module leverages diverse time-efficiency attributes of

financial data, enhancing its trading efficacy. The decisionmaking module enables FINMEM to synchronize its memory streams with market facts, facilitating high-quality trading decisions. The details and notations associated with these three modules are provided in the subsequent sections.

# A. Profile Module

The profile module empowers FINMEM to develop a dynamic agent character specifically designed to navigate the complex dynamics of financial markets effectively.

The dynamic character of FINMEM comprises two principal components: firstly, a foundational professional knowledge base akin to a trading expert, and secondly, an agent with three distinct investment risk inclinations. The first component includes two types of information: an introduction to the primary trading sectors relevant to the company stock FINMEM will trade in, and a concise overview of the historical financial performance of the specified ticker, spanning from the beginning to the end of the training period. Before initiating trades in a new company’s stock, FINMEM accesses and updates this sector-specific and historical financial data from a backend database. This professional background setting narrows down information and memory events pertinent to specific trading tasks. The detailed prompt template for constructing the profile module is presented in Fig. 9 in Section VI-D, using TSLA trading as an example.

The second component of FINMEM’s design encompasses three distinct risk inclination options: risk-seeking, risk-averse, and a self-adaptive risk character. The risk-seeking setting gears FINMEM towards an aggressive, high-reward approach, while the risk-averse setting gears it towards a conservative, lower-risk strategy. A distinctive aspect of FINMEM is its ability to dynamically alternate between these risk settings in response to current market conditions. Specifically, it shifts risk preferences when the Cumulative Return falls to below zero within a brief period, such as three days, and reversely. This flexible design functions as a protective mechanism, mitigating prolonged downturns in turbulent market environments. During the initial stage of the training phase, FINMEM is configured with a chosen risk preference, each supplemented with comprehensive textual explanations through LLM prompts. These guidelines shape how FINMEM processes incoming messages and determines its subsequent actions in alignment with its designated risk inclination. The system maintains a catalog of all risk inclinations and their detailed explanations in a backlog, enabling seamless adaptation to different stocks by switching among these risk profiles as needed.

The dynamic character setting in FINME’s profile module provides subjective and professional background knowledge and flexible choice of risk inclinations. It provides crucial context for filtering and retrieving trading-relevant information and memory events, thus improving accurate inferencing and adaptability to fluctuating market conditions.

# B. Memory Module

The memory module of FINMEM emulates a human trader’s cognitive system so that it can efficiently process hierarchical

financial information and prioritize the critical messages for high-quality investment decisions. Furthermore, it adjusts the memory span flexibly, enabling the agent to operate on a wider range of events over a longer retrieval period. FINMEM’s memory module, illustrated in Fig. 1, comprises working and long-term memory with layered processing capability and is initiated by a specific investment inquiry.

1) Working Memory: Working memory refers to the human cognitive system’s functions for temporary storage and diverse operations. We incorporate this concept into FINMEM’s memory module development, creating a central workspace for informed decision-making. Unlike human working memory, having a maximum capacity of seven plus or minus two memory events [13], FINMEM has the ability to expand the capacity based on specific requirements. Tailored for converting financial data into trading actions, FINMEM’s working memory encompasses three key operations: summarization, observation, and reflection. The mechanisms by which they interact and operate as an integrated decision-making workflow are detailed in the middle box of Fig. 1. Additionally, the LLM prompt template that supports these processes is detailed in the case study presented in Figs. 9 and 10 in Section VI-D below.

Summarization: FINMEM leverages external market data to derive critical investment insights and sentiments tailored to specific stock trading queries, such as “Can you make an investment decision on TSLA on 10/25/2022?”. As illustrated in Fig. 1(1), this system condenses the original text into a compact yet informative paragraph, thereby enhancing FINMEM’s processing efficiency. It efficiently extracts and summarizes pertinent data and sentiments for stock investment decisions, demonstrated here using Tesla Inc. as an example. Subsequently, FINMEM directs these insights to an appropriate layer within its long-term memory architecture, selecting the layer based on the time sensitivity of the information.

Observation: Triggered the same inquiry, FINMEM initiates an observation operation to gather market facts. The information available to FINMEM varies between the training and testing phases.

During the training phase, FINMEM has access to comprehensive stock price data within the specified period. Upon receiving trading inquiries that specify a stock ticker and date, FINMEM focuses on the daily adjusted closing price differences, comparing the following day’s price with the current day’s. These price differences are utilized as market ground labels. Specifically, a decrease in price suggests a “Sell” action, while an increase or no change in price indicates a “Buy” action.

During the testing phase, at a specific time point, FINMEM loses the ability to access future price data. Its focus shifts to the analysis of historical stock price movements, depending on a retrospective evaluation of the cumulative return from the last $M$ trading days to infer future market trends. This phase, characterized by the absence of foreseen market grounds, serves as a critical assessment of FINMEM’s development. It tests whether the system has adequately established logical connections between stock price trends and various financial information sources, such as news, reports, and indicators. This stage is key in evaluating FINMEM’s capability of independently

![](images/0a58a6990eb2346517c9d0f22f3dace231a4f4e196aa5b29676b9ad655fe6c85.jpg)

![](images/71b8a7f0287464de3f0fb9745b38f56b063680bece0972d96b9c7a23a775e910.jpg)  
Fig. 1. (1) FINMEM’s memory module interacts with the market environment to distill multi-source financial information and facilitate investment decisions. It contains two core components – Working Memory and Layered Long-term Memory. (2) The outline of FINMEM’s decision-making workflow for retrieving critical memory events and market observations to inform specific investment decisions.

evolving its trading strategies for subsequent tasks, leveraging its analysis and interpretation of historical data patterns.

Reflection: Two types of reflections exist, immediate and extended reflection. (a) Immediate reflection is activated upon receiving a daily trading inquiry for a specific ticker. Using LLM and specific prompts exemplified in Fig. 1(1), the agent merges market indications and top- $K$ -ranked events from each long-term memory layer. Market indications are derived from the outcomes of the observation operation and differ between the training and testing phases. During testing, this process yields three types of outputs: the trading direction (“Buy”, “Sell”, or “Hold”), the underlying rationale for this decision, and the most influential memory events, along with their IDs from each layer that informed the decision. In the training phase, specifying the trading direction is unnecessary, as FINMEM is already informed of future stock movement directions. The top- $K$ -ranked memory events encapsulate key insights and sentiments derived from critical investment-related incoming messages, all distilled by FINMEM’s advanced summarization capabilities.

(b) Extended reflection reevaluates immediate reflection outcomes for a ticker over a specified $M$ -day trace period. It encompasses data like stock price trends, trading returns, and action rationales from multiple immediate reflections. While immediate reflection enables direct trading execution and records current feedback, extended reflection summarizes market trends and reassesses recent Cumulative Return on investment. Extended reflection is eventually transmitted and stored in the deep processing layer to emphasize its criticality (detailed introduced in Section III-B2) of long-term memory. $K$ and $M$ are hyperparameters to adjust FINMEM’s working memory capacity and information retrieval ability. FINMEM gains the flexibility of integrating comprehensive information into well-informed decisions by fine-tuning them.

2) LAYERED LONG-TERM MEMORY: FINMEM’s long-term memory organizes hierarchical financial data insights in a stratified structure, as illustrated in the lower section of Fig. 1. Drawing inspiration from the varying decay speeds in the human cognitive system’s information processing layers [15], FINMEM employs a layered structure to accommodate the diverse time sensitivities inherent to different types of financial data. This structure categorizes summarized insights by their timeliness and decay rates. Insights are derived by the working memory’s summarization operation. Those directed to deeper layers receive smaller decay rates, indicating longer retention, while those in shallower layers are assigned larger decay rates for shorter retention.

$$
\gamma_ {l} ^ {E} = S _ {\text {R e c e n c y} _ {l}} ^ {E} + S _ {\text {R e l e v a n c y} _ {l}} ^ {E} + S _ {\text {I m p o r t a n c e} _ {l}} ^ {E}, \tag {1}
$$

where each memory event is only associated with one score and can only belong to a single layer.

Upon receiving an investment inquiry, FINMEM retrieves the top- $K$ pivotal memory events from each layer and channels them to the immediate reflection component of the working memory. These events are chosen according to the descending order of their information retrieval score, denoted as $\gamma _ { l } ^ { E }$ , where l belongs to the set shallow, intermediate, deep, as specified in (1). $E$ denotes a given memory event. This score, adapted from Park et al. [14] but with modified recency and importance computations, especially tailoring to handle data with various timelines. It encapsulates three metrics: recency, relevancy, and importance. Individual metric scores exceeding 1.0 are scaled to the [0,1] range before being summed. The modification is to achieve the layered processing function and represent the various periodicity of the financial environment.

$$
S _ {\text {R e c e n c y} _ {l}} ^ {E} = e ^ {- \frac {\delta E}{Q _ {l}}}, \quad \delta^ {E} = t _ {\mathrm {P}} - t _ {E}, \tag {2}
$$

where $\delta ^ { E }$ refers to the time difference between the memory event occurrence and the trading inquiry arrival. $Q _ { \mathrm { s h a l l o w } } = 1 4$ , $Q _ { \mathrm { i n t e r m e d i a t e } } = 9 0 $ , and $Q _ { \mathrm { d e e p } } = 3 6 5$ correspond to day counts of two weeks, a quarter, and a year for shallow, intermediate, and deep processing layers, respectively.

Upon a trade inquiry $P$ arrival in processing layer $l$ via LLM prompt, the agent computes the recency score $S _ { \mathrm { R e c e n c y } _ { l } } ^ { E }$ per (2). $S _ { \mathrm { R e c e n c y } _ { l } } ^ { E }$ l inversely correlates with the time gap between the inquiry and the event’s memory timestamp, mirroring Ebbinghaus’s forgetting curve [54]. The stability term $Q _ { l }$ in (2) partially controls memory decay rates across layers, indicating longer memory persistence in the long-term layer with a higher stability value. In the context of trading, company annual reports, such as Form 10-Ks, are considered to have more extended timeliness compared to daily financial news. Therefore, they are assigned a higher stability value and are categorized within the deeper processing layer. This classification reflects their extended relevance and impact in financial decision-making scenarios.

$$
S _ {\text {R e l e v a n c y} _ {l}} ^ {E} = \frac {\mathbf {m} _ {\mathbf {E}} \cdot \mathbf {m} _ {\mathbf {P}}}{\left\| \mathbf {m} _ {\mathbf {E}} \right\| _ {2} \times \left\| \mathbf {m} _ {\mathbf {P}} \right\| _ {2}} \tag {3}
$$

The relevancy score, denoted as SErelevancyl , quantifies the co- $S _ { \mathrm { r e l e v a n c y } _ { l } } ^ { E }$ sine similarity between the embedding vectors. These vectors are derived from the textual content of the memory event, $\mathbf { m } _ { \mathbf { E } }$ , and the LLM prompt query, mP, using OpenAI’s “text-embeddingada- $. 0 0 2 ^ { \circ }$ model, as depicted in (3). The LLM prompt query incorporates inputs related to trading inquiries and the trading agent’s character setting.

The importance score $S _ { \mathrm { I m p o r t a n c e } _ { l } } ^ { E }$ is computed using the value $v _ { l } ^ { E }$ from a uniform piecewise scoring function (Formula 4), multiplied by a degrading ratio $\theta _ { l }$ (Formula 5) as per (6). The likelihood of higher $v _ { l } ^ { E }$ values increases from shallow to deep layers. $\theta _ { l }$ measures the diminishing importance of an event over time, which has a close form design of [14]. But our approach tailors $\theta _ { l }$ to the stratified structure of long-term memory. It adopts unique exponential functions for each layer. The base $\alpha _ { l }$ for each layer is a hyperparameter, set to follow the sequence: $\alpha _ { s h a l l o w } < \alpha _ { i n t e r m e d i a t e } < \alpha _ { d e e p }$ $< \alpha _ { d e e p }$ . These values correlate with the rate at which their importance degrades after a certain period, providing another angle to measure importance variances across different memory types. Through experimentation, we set $\alpha _ { s h a l l o w } = 0 . 9$ , $\alpha _ { i n t e r m e d i a t e } = 0 . 9 6 7$ and $\alpha _ { d e e p } = 0 . 9 8 8$ . This ensures that $\theta _ { l }$ decreases to a threshold score of 5 after intervals of 30, 90, and 365 days for shallow, intermediate, andfor $S _ { \mathrm { I m p o r t a n c e } _ { l } } ^ { E }$ rs, reand $S _ { \mathrm { R e c e n c y } _ { l } } ^ { E }$ y. The three-piece-wise functionsenable FINMEM to have layered processing in theare purged when $S _ { \mathrm { R e c e n c y } _ { l } } ^ { E }$ m memory compois below 0.05 or $S _ { \mathrm { I m p o r t a n c e } _ { l } } ^ { E }$ ory eventsis under 5 (pre-scaling).

$$
v _ {l} ^ {E} = \left\{ \begin{array}{l l} 4 0 & \text {w i t h p r o b a b i l i t y} p _ {1} \\ 6 0 & \text {w i t h p r o b a b i l i t y} p _ {2} \\ 8 0 & \text {w i t h p r o b a b i l i t y} p _ {3} \end{array} \right. \tag {4}
$$

$$
\theta_ {l} = \left(\alpha_ {l}\right) ^ {\delta^ {E}}, \quad l = \text {s h a l l o w}, \text {i n t e r m e d i a t e}, \text {d e e p}, \tag {5}
$$

where $p _ { 1 } + p _ { 2 } + p _ { 3 } = 1$ , but their values vary by shallow, intermediate, and deep processing. when shallow processing $p _ { 1 } , p _ { 2 } , p _ { 3 } = \{ 0 . 8 , 0 . 1 5 , 0 . 0 5 \}$ , intermediate processing, $p _ { 1 } , p _ { 2 } , p _ { 3 } = \{ 0 . 0 5 , 0 . 8 , 0 . 1 5 \}$ and deep processing, $p _ { 1 } , p _ { 2 } , p _ { 3 } =$ $\{ 0 . 0 5 , 0 . 1 5 , 0 . 8 \}$ .

$$
S _ {\text {I m p o r t a n c e} _ {l}} ^ {E} = v _ {l} ^ {E} * \theta_ {l}, \tag {6}
$$

Furthermore, an access counter function oversees the transfer of memory events among layers, ensuring that significant events influencing trading decisions ascend from shallower to deeper layers for extended retention and recurrent access by FINMEM. Conversely, less pertinent events gradually diminish. This process is facilitated by the LLM validation tool Guardrails AI [55], which monitors critical memory IDs across different layers. An event identified as pivotal for investment success receives an additional 5 points in its importance score $S _ { \mathrm { I m p o r t a n c e } _ { l } } ^ { E }$ . Upon meeting the criteria for upgrading to a deeper layer, an event’s recency score SERecency $S _ { \mathrm { R e c e n c y } _ { l } } ^ { E }$ is reset to 1.0, emphasizing its importance and preventing rapid decay. By implementing this access counter, FINMEM effectively identifies and prioritizes key events, taking into account their nature and frequency of retrieval.

# C. Decision-Making Module

The decision-making module of FINMEM efficiently integrates operational outcomes from the profile and memory modules to support well-informed investment decisions, as depicted in Fig. 1(1). In its daily trading decisions, FINMEM is asked to select from three distinct actions for a single share of a specific stock by Guardrails AI text validation function: “Buy”, “Sell”, or “Hold”. Additionally, the inputs and results required by FIN-MEM’s decision-making module vary between its training and testing phases, with each phase’s specifics detailed as follows:

During the training phase, FINMEM accesses a wide array of multi-source information relevant to the entire time period. When FINMEM is prompted with trading inquiries containing stock ticker and date, as well as trader character-related texts, it concurrently initiates observation and summarization operations in its working memory. FINMEM observes the market ground labels mentioned in the description about the observation operation in Section III-B1, which involve daily adjusted price differences between consecutive days, indicative of “Buy” or “Sell” actions. Utilizing these price change signals, FINMEM identifies and prioritizes the top- $K$ memories, ranking them based on retrieval scores from each long-term memory layer. This procedure enables FINMEM to produce comprehensive reflections that provide a well-founded rationale and in-depth inference of the correlation between market ground labels and the memories retrieved. Through repeated trading operations, reflections, and memory events with significant impact, transition to a deeper memory processing layer, getting preserved for guiding future investment decisions during the testing phase.

In the testing phase, where FINMEM cannot access future price data, it relies on the Cumulative Return over the previous $M$ trading days to anticipate future market trends. To compensate for the absence of future market price information, FINMEM

utilizes enhanced reflections derived from immediate reflections spanning an $M$ -trading-day period as supplementary references. When faced with a specific trading inquiry, FINMEM integrates insights from various sources, including historical Cumulative Return, outcomes from extended reflection, and the Top- $K$ retrieved memories. This comprehensive approach enables FIN-MEM to execute well-informed trading decisions.

It should be noted that FINMEM generates executable actions exclusively in the immediate reflection operation of the testing phase. Since the trading direction is guided by the actual price trend, the training phase of FINMEM does not make investment decisions. Instead, this phase is dedicated to accumulating trading experience through comparing market trends with incoming multi-source financial messages. Additionally, during this phase, FINMEM develops a memory module enriched with a comprehensive knowledge base, thereby evolving its capability for independent decision-making in future trading activities.

# IV. EXPERIMENTS SETUPS

We aim to evaluate the trading performance of FINMEM. And we further illustrate its unique advantages of requiring significantly less historical trading time window to train and take full use of key financial data time series as well as textual information. Specifically, we conducted several experiments to study the following research questions (RQs):

- RQ1: Is FINMEM capable of outperforming contemporary state-of-the-art algorithmic trading agents?   
- RQ2: Is FINMEM able to provide reliable performance on stocks with limited training data?   
- RQ3: Which LLM is best suited to form the backbone framework of FINMEM?   
- RQ4: Does the various risk inclination options that FIN-MEM’s profile module offers truly differentiate its trading performance?   
RQ5: Does FINMEM’s unique feature of an adjustable cognitive span effectively facilitate informed trading decisions?

In the rest of the section, we begin by introducing the real-world financial dataset used in our experiments. We then describe the comparative algorithmic agents and list several widely used financial metrics. Our experiments fall into two categories: 1) The comparative experiments of FINMEM versus other algorithmic trading agents, and FINMEM using different LLMs as backbone algorithms. 2) The ablation studies evaluate the effects of FINMEM’s adjustable cognitive span and the role of the trader’s dynamic character settings, particularly the risk inclinations, on its trading performance. Through experiments, FINMEM demonstrates to outperform other comparative algorithmic agents. Furthermore, we are able to show that its profile and memory modules are sophisticated and tailored to effectively address the intricacies of the financial landscape, resulting in superior trading performance.

# A. Datasets and Database Structure:

We assessed FINMEM’s performance using a backtesting strategy running on multi-source financial data from August

![](images/a5ef43f18f7278e271a20a2b0ec9065573fc52a2a72f0862bb35664269956e16.jpg)  
Fig. 2. FINMEM’s data warehouse architecture and data pipelines. Multisource data were collected from various APIs and organized into corresponding data warehouses to construct its layered long-term memory module.

15, 2021, to April 25, 2023. These data are collected through reputable financial databases and APIs like Yahoo Finance (via yfinance) and Alpaca News API, detailed explained in Table 2. The stock tickers used in our comparative experiments are detailed in Fig. 11 of Appendix A, online available. These were selected because they are among those with the highest volumes of accessible news text data, and they are spread across various trading sectors. This selection provides ample data to evaluate FINMEM’s generalization capabilities. Additionally, Tesla, Inc. (TSLA) was selected for ablation studies because it is associated with a substantial amount of textual data, providing ample information to thoroughly assess the robustness of FINMEM’s performance. We conducted a sensitivity analysis on TSLA to demonstrate the effectiveness of its adjustable cognitive span functions and the reinforcement of critical memory insights through an access count mechanism for the same reasons. Note that while we conducted backtesting—a standard and practical method for evaluating trading strategies—the FINMEM framework is fully equipped to execute real-time trading once it integrates streaming financial data API calls. Furthermore, the results of our ablation studies and sensitivity analysis demonstrate that FINMEM’s mechanism can promptly reinforce critical market trend signals, facilitating informed decision-making, which shed light on its adaptability to sudden market changes in real-time investment.

The multi-source financial data, derived from various data sources and APIs, collectively form the ”Market Environment Data Warehouse”. While the daily stock open-high-low-closevolume (OHLCV) is channeled to FINMEM’s working memory to convert into texts interpreted market facts through its observation operation, the rest of the data are then funneled into FINMEM’s ”Layered Long-term Memory Data Warehouse,” sorted by timeliness. This is achieved through the summarization operation of the working memory, utilizing LLM prompts, as illustrated in Fig. 2. The deep processing layer holds annual reports (Form 10 K’s) insights, the intermediate layer contains quarterly reports (Form 10Q’s) insights, and the shallow layer accommodates daily financial news insights. Given the daily frequency of this financial news dataset, we evaluated FIN-MEM’s performance on a daily basis. This is a frequency is tent with prevalent trading practices, and it can complement

high-frequency trading that operates at microsecond intervals in real-world fund firms, thereby enhancing investment diversity. But considering the typical response time of LLM models, the FINMEM mechanism is capable of generating sequential trading decisions at the granularity of minute-level intervals.

We leveraged the open-source vector database FAISS [56] for constructing the memory warehouse of FINMEM, benefiting from its rapid querying in high-dimensional vectors and compatibility with OpenAI for cosine-similarity-based semantic searches on specific tickers. This setup facilitates efficient topranked event retrieval. Data categorization and memory module workflow are also illustrated in Fig. 1.

# B. Baseline and Comparative Models:

We assess FINMEM’s trading performance in comparison to five advanced algorithmic agents and a commonly accepted baseline trading strategy. Among these, three models employ Deep Reinforcement Learning (DRL) approaches, while the remaining two are based on Large Language LLMs. Brief descriptions of each are provided below, with a detailed introduction elaborated in Appendix B, online available:

Buy-and-Hold strategy (B&H): A passive investment approach, where an investor purchases stocks and holds onto them for an extended period regardless of market fluctuations, is commonly used as a baseline for comparison of stock trading strategies.

DRL trading agents: As the FINMEM is practiced and examined on the basis of single stock trading and discrete trading actions, we choose three advanced DRL algorithms fitting into the same scenarios according to the previous and shown expressive performance in the work of Liu et al. [57], [58]. The three DRL algorithms are Proximal Policy Optimization (PPO) [59], Deep Q-Network (DQN) [60] and Advantage Actor-Critic (A2C) [61]. The DRL training agents only take numeric features as inputs.

LLM trading agents: We evaluate FINMEM against two LLM agents in the context of stock trading. The first LLM agent – General-Purpose Generative Agents (GA) [62], known for its proficiency in general-purpose tasks, serves as a baseline. The second agent – LLM trading agents(FINGPT) [11], a leadingedge LLM in trading, has been acclaimed for its promising performance in stock market operations.

# C. Evaluation Metrics:

We employ five widely-used metrics in finance to compare the investment rewards of FINMEM against other algorithmic trading agents. Here are their introductions:

Cumulative Return (CR) $\%$ [63]: Cumulative Return is a key trading performance metric because it provides a comprehensive insight into investment performance, especially for strategies that emphasize long-term growth and reinvestment. The effectiveness of different investment strategies is evaluated based on their Cumulative Returns, which reflect the total change in value over time. In this study, we compute Cumulative Returns over the specified period by summing daily logarithmic returns, as outlined in (7). This method is widely accepted in the finance area due

to its ability to precisely capture minor price fluctuations and symmetrically address gains and losses. In essence, a higher Cumulative Return typically indicates a more effective strategy.

$$
\mathbf {C R} = \sum_ {t = 1} ^ {n} r _ {i} = \sum_ {t = 1} ^ {n} \left[ \ln \left(\frac {p _ {t + 1}}{p _ {t}}\right) \cdot \text {a c t i o n} _ {t} \right] \tag {7}
$$

where $r _ { i }$ represents the logarithmic return for day $t + 1 , p _ { t }$ is the closing price on day t, $p _ { t + 1 }$ is the closing price on day $t + 1$ , and actiont denotes the trading decision made by the model for that day.

- Sharpe Ratio (SR) [64]: Sharpe Ratio is another core metric for evaluating investment performance and adjusting returns for risk. It is calculated by dividing the portfolio’s average excess return $( R _ { p } )$ over the risk-free rate $( R _ { f } )$ by its volatility $( \sigma _ { p } )$ , as shown in (8). This metric adjusts returns for risk, with a higher ratio indicating better risk-adjusted performance. Essential in comparing different portfolios or strategies, it contextualizes performance against similar investments. Although a Sharpe Ratio above 1 is typically considered favorable and above 2 as excellent, these benchmarks can vary depending on the context of comparison.

$$
\mathbf {S R} = \frac {R _ {p} - R _ {f}}{\sigma_ {p}} \tag {8}
$$

- Annualized Volatility (AV) $\%$ and Daily Volatility (DV) % [65]: Annualized Volatility is calculated as the Daily Volatility (standard deviation of daily logarithmic returns) multiplied by the square root of the typical number of trading days in a year (252) as outlined in (9), is vital for assessing investment risk. This measure reflects the extent of fluctuation in a return series over a year, indicating potential deviations from average returns.

$$
\mathbf {A V} = \mathbf {D V} \times \sqrt {2 5 2} \tag {9}
$$

- Max Drawdown (MDD) $\%$ [66]: Max Drawdown is a metric for assessing risk. It represents the most significant decrease in a portfolio’s value, from its highest $( P _ { \mathrm { p e a k } } )$ to its lowest point $( P _ { \mathrm { t r o u g h } } )$ until a new peak emerges, detailed in (10). Indicative of investment strategy robustness, a smaller Max Drawdown suggests reduced risk.

$$
\mathbf {M D D} = \max  \left(\frac {P _ {\text {p e a k}} - P _ {\text {t r o u g h}}}{P _ {\text {p e a k}}}\right) \tag {10}
$$

In our experiments and ablation studies, we recorded the metric outcomes as an average from five repeated trials.

# V. EXPERIMENTS

# A. Implementation Details:

In the Trading Agents Comparison, FINMEM employs GPT-4 as its backbone algorithm. The temperature parameter of the model is set at 0.7 to maintain a balance between response content consistency and model creativity. It was trained on financial data from August 17, 2021, to October 05, 2022, and tested with data from October 06, 2022, to April 10, 2023. The

![](images/7b00100cfd365373e973fb397c97144c127f5214a3aaed57c14889e015b04868.jpg)

![](images/7a4c10643ef274e909c3cc7f1def529c273235fc318ad800ff9a3cba00788a8d.jpg)

![](images/8145bbe2eccd0c2f8a2878c878d91d5982a537c616bd87eaeba0389b4870ddcf.jpg)

![](images/cba4c8ee4689a7e0b0ab1c9491ff724b10df2067be0b0bf901f458791451c878.jpg)  
Fig. 3. CR comparisons between FINMEM and other algorithmic agents for TSLA, NFLX, AMZN, and MSFT in the testing phase. FINMEM won the highest CRs for all of them by the end of the testing phase.

training period was chosen to account for the seasonal nature of corporate financial reporting and the duration of data retention in FINMEM’s memory module. The selected training duration ensures the inclusion of at least one publication cycle of either Form 10-Q, classified as intermediate memory, or Form 10-K, regarded as deep memory, or in some instances, both. This strategy ensures that the experiences retained in FINMEM are still influential during the testing phase for a significant period. Additionally, the training duration allowed FINMEM sufficient time to establish inferential links between financial news, market indicators, and stock market trends, thereby accumulating substantial experience. Furthermore, we set the number of top memory events retrieved from each layer of long-term memory at 5. We ran FINMEM using each of the three available risk inclination settings. The reported performance outcomes are based on the setting that achieved the highest cumulative return during the testing phase.

To maintain consistency in the comparison, the training and testing phases for the other two LLM-based agents were aligned with those of FINMEM. For parameters of other LLM-based agents that are not encompassed by FINMEM’s configuration, they were kept in accordance with their original settings as specified in their respective source codes.

Considering that DRL algorithms need extensive training data for stable and converged results, and given our daily evaluation of trading performance, we extend the DRL agents’ training period to roughly a 10-year span, from January 1, 2012, to October 05, 2022, for a fair comparison. The testing period is kept consistent with the other models. The DRL algorithms are implemented using Stable Baselines 3 [67].

FINMEM’s performance was benchmarked against that of the most effective comparative model, using Cumulative Return and Sharpe Ratio as the primary evaluation metrics. The statistical significance of FINMEM’s superior performance was ascertained through the non-parametric Wilcoxon signed-rank test, which is apt for the non-Gaussian distributed data.

# B. Algorithmic Trading Agents Comparison (RQ1 & RQ2)

In this experiment, we assessed the stock trading performance of FINMEM against other models, focusing on stocks from five companies in different trading sectors: Tesla, Inc. (TSLA), Netflix, Inc. (NFLX), Amazon.com, Inc. (AMZN), Microsoft Corporation (MSFT), and Coinbase Global, Inc. (COIN). The performance of all algorithmic trading agents across five key metrics was consolidated in Table I. Given the pivotal role of Cumulative Return in evaluating trading performance over time, we presented detailed time series plots in Figs. 3 and 4. It’s important to note that the trading performance of FINMEM for COIN was exclusively compared with LLM trading agents and the baseline. This was because Coinbase Global, Inc. had completed its IPO in April 2021 and, as a result, had not accumulated enough trading data to facilitate stable outcomes with DRL algorithms. These plots illustrate the changes in Cumulative Return for each of the five companies throughout the testing phase, offering an in-depth comparison of performance.

In response to RQ1, the trading outcomes presented in Table I revealed that FINMEM outperformed all other algorithmic trading agents and the B&H baseline strategy in terms

TABLE I TRADING PERFORMANCE COMPARISON DURING THE TESTING PERIOD BETWEEN FINMEM AND OTHER ALGORITHMIC AGENTS ACROSS FIVE STOCKS   

<table><tr><td>Ticker</td><td>Model</td><td>CR (%)</td><td>SR</td><td>DV (%)</td><td>AV (%)</td><td>MDD (%)</td></tr><tr><td rowspan="7">TSLA</td><td>Buy and Hold</td><td>-18.6312</td><td>-0.5410</td><td>4.4084</td><td>69.9818</td><td>55.3208</td></tr><tr><td>FINMEM</td><td>61.7758*</td><td>2.6789</td><td>2.9522</td><td>46.8649</td><td>10.7996</td></tr><tr><td>Generative Agents</td><td>13.4636</td><td>0.5990</td><td>2.8774</td><td>45.6774</td><td>24.3177</td></tr><tr><td>FinGPT</td><td>-7.4554</td><td>-0.2795</td><td>3.4145</td><td>54.2027</td><td>42.3993</td></tr><tr><td>A2C</td><td>13.7067</td><td>0.3979</td><td>4.4096</td><td>70.0009</td><td>52.3308</td></tr><tr><td>PPO</td><td>1.2877</td><td>0.0374</td><td>4.4110</td><td>70.0232</td><td>54.3264</td></tr><tr><td>DQN</td><td>33.3393</td><td>0.9694</td><td>4.4027</td><td>69.8900</td><td>52.0033</td></tr><tr><td rowspan="7">NFLX</td><td>Buy and Hold</td><td>35.5111</td><td>1.4109</td><td>3.1964</td><td>50.7410</td><td>20.9263</td></tr><tr><td>FINMEM</td><td>36.4485*</td><td>2.0168</td><td>2.2951</td><td>36.4342</td><td>15.8495</td></tr><tr><td>Generative Agents</td><td>32.0058</td><td>1.5965</td><td>2.5460</td><td>40.4168</td><td>16.9893</td></tr><tr><td>FinGPT</td><td>9.0090</td><td>0.4266</td><td>2.6819</td><td>42.5732</td><td>28.2705</td></tr><tr><td>A2C</td><td>14.6155</td><td>0.5788</td><td>3.2071</td><td>50.9112</td><td>25.0184</td></tr><tr><td>PPO</td><td>8.4121</td><td>0.3330</td><td>3.2086</td><td>50.9344</td><td>25.0184</td></tr><tr><td>DQN</td><td>-12.2067</td><td>-0.4833</td><td>3.2078</td><td>50.9217</td><td>28.7017</td></tr><tr><td rowspan="7">AMZN</td><td>Buy and Hold</td><td>-10.7739</td><td>-0.4980</td><td>2.7697</td><td>43.9674</td><td>33.6828</td></tr><tr><td>FINMEM</td><td>4.8850*</td><td>0.2327</td><td>2.6872</td><td>42.6576</td><td>22.9294</td></tr><tr><td>Generative Agents</td><td>-13.9271</td><td>-0.9981</td><td>1.7864</td><td>28.3576</td><td>27.7334</td></tr><tr><td>FinGPT</td><td>-29.6781</td><td>-2.1756</td><td>1.7464</td><td>27.7225</td><td>28.4838</td></tr><tr><td>A2C</td><td>-6.3591</td><td>-0.2938</td><td>2.7706</td><td>43.9819</td><td>26.1275</td></tr><tr><td>PPO</td><td>-8.4194</td><td>-0.3891</td><td>2.7702</td><td>43.9761</td><td>33.6828</td></tr><tr><td>DQN</td><td>-29.9820</td><td>-1.3906</td><td>2.7603</td><td>43.8177</td><td>38.3740</td></tr><tr><td rowspan="7">MSFT</td><td>Buy and Hold</td><td>14.6949</td><td>0.8359</td><td>2.2326</td><td>35.4411</td><td>15.0097</td></tr><tr><td>FINMEM</td><td>23.2613*</td><td>1.4402</td><td>2.0512</td><td>32.5617</td><td>14.9889</td></tr><tr><td>Generative Agents</td><td>-18.1031</td><td>-1.6057</td><td>1.4318</td><td>22.7285</td><td>24.2074</td></tr><tr><td>FinGPT</td><td>5.7356</td><td>0.4430</td><td>1.6442</td><td>26.1008</td><td>12.8459</td></tr><tr><td>A2C</td><td>0.4598</td><td>0.0261</td><td>2.2357</td><td>35.4913</td><td>23.6781</td></tr><tr><td>PPO</td><td>12.8067</td><td>0.7282</td><td>2.2333</td><td>35.4532</td><td>19.5355</td></tr><tr><td>DQN</td><td>14.7397</td><td>0.8385</td><td>2.2326</td><td>35.4408</td><td>25.1845</td></tr><tr><td rowspan="7">COIN</td><td>Buy and Hold</td><td>-30.0071</td><td>-0.5150</td><td>6.7517</td><td>107.1795</td><td>60.5084</td></tr><tr><td>FINMEM</td><td>34.9832*</td><td>0.7170</td><td>5.6538</td><td>89.7515</td><td>35.7526</td></tr><tr><td>Generative Agents</td><td>3.4627</td><td>0.0896</td><td>4.4783</td><td>71.0908</td><td>32.0957</td></tr><tr><td>FinGPT</td><td>-88.7805</td><td>-1.9507</td><td>5.2736</td><td>83.7153</td><td>73.5774</td></tr><tr><td>A2C</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PPO</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DQN</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

FINMEM significantly outperforms the second-best algorithmic trading agent in terms ofCRs and SRs. * indicates that theresultoftheWilcoxonsigned-ranktestisstatisticallysignificant.Theboldnumbersinthisandsubsequenttables signify the best performance for the respective metrics.

![](images/735af450ca68708ef9e0d370ec5ad21872f876aafeb7336ff0ee95ffd64a916a.jpg)  
Fig. 4. Comparison of CRs between FINMEM and other LLM-based agents for COIN during the testing phase shows that FINMEM leads in trading tasks for recently IPO-ed stocks that can be managed by language agents. This presents a challenge to DRL-based agents.

of Cumulative Return and Sharpe Ratio. FINMEM’s superiority was statistically significant when compared to the second-best trading strategy. Specifically, for TSLA and NFLX, FINMEM’s strategy achieved Sharpe Ratios exceeding 2.0 and Cumulative Returns surpassing 0.35 while maintaining the lowest Volatility and Max Drawdown. These indicators underscored FINMEM’s ability to generate higher returns per unit of risk. In the case of MSFT and NFLX, FINMEM also recorded a Sharpe Ratio

1The bold numbers in this and subsequent tables signify the best performance for the respective metrics.

![](images/1c4d8a03efda57d8edfd0453a6d3f5178a8c11880a37478c9c250874e0d28b22.jpg)  
Fig. 5. CRs of FINMEM on trading TSLA over an extended testing period. FINMEM consistently shows high performance.

above 1.0 and a Cumulative Return over 0.2, coupled with relatively low Volatility and Max Drawdown, demonstrating its impressive trading performance. For AMZN and COIN, FIN-MEM consistently delivered positive Cumulative Returns and superior Sharpe Ratios, outperforming other strategies that yielded negative values for these metrics. Additionally, its Volatility and Max Drawdown were on the lower end. Hence, these results collectively demonstrated FINMEM’s robust trading performance across a diverse range of trading sectors. Specifically, FINMEM exhibited superior performance compared to the two other LLM agents in our study, FINGPT and the general-purpose generative agent developed by Park et al. This underscored the effectiveness of FINMEM’s unique profile and memory structure, which are particularly tailored for LLM agents dealing with financial data, significantly enhancing their investment decision-making capabilities.

In response to RQ2, the main challenge for DRL trading agents is that they require training data with a large volume and extended time span, which are hard to achieve when operating on stocks with limited historical data. As shown in Table I, our experiments revealed that FINMEM achieved superior trading performance with a much shorter training duration compared to DRL trading agents trained on data spanning nearly a decade. This efficiency makes FINMEM particularly useful for newly public companies like Coinbase Global, Inc., which have limited trading histories. DRL agents often face convergence issues due to inadequate training data in such cases. Moreover, even among LLM-based trading agents suited for shorter training periods, FINMEM’s performance stood out, as illustrated in Fig. 4.

To further assess FINMEM’s adaptability to limited training data, we narrowed the training period down to an even shorter period, spanning from August 17, 2021, to February 10, 2022. We then extended the testing phase to cover from February 11, 2022, to April 25, 2023. This evaluation focused on the stock of Tesla, Inc., which has the largest volume of news data. The trading performance of FINMEM during this period is depicted in Fig. 5. Remarkably, using less than six months of daily frequency data for training, which encompassed the publication of one Form 10-K and one Form 10-Q, FINMEM consistently ranked high in gains and attained the highest cumulative return after the latter half of December 2022.

TABLE II TRADING PERFORMANCE COMPARISON OF FINMEM USING DIFFERENT LLMS AS BACKBONES   

<table><tr><td>Model</td><td>CR (%)</td><td>SR</td><td>DV (%)</td><td>AV (%)</td><td>MDD (%)</td></tr><tr><td>B&amp;H</td><td>-66.9497</td><td>-2.0845</td><td>3.8050</td><td>60.4020</td><td>67.3269</td></tr><tr><td>GPT 3.5-Turbo</td><td>16.1501</td><td>2.1589</td><td>0.8862</td><td>14.0683</td><td>1.1073</td></tr><tr><td>GPT4</td><td>62.6180</td><td>2.2251</td><td>3.3339</td><td>52.9237</td><td>17.4012</td></tr><tr><td>GPT4-Turbo</td><td>54.6958</td><td>2.4960</td><td>2.5960</td><td>41.2100</td><td>12.5734</td></tr><tr><td>davinci-003</td><td>1.6308</td><td>0.8515</td><td>0.2269</td><td>3.6018</td><td>0.8408</td></tr><tr><td>Llama2-70b-chat</td><td>-52.7233</td><td>-2.8532</td><td>2.1891</td><td>34.7503</td><td>44.7168</td></tr></table>

GPT-4 wins the highest CR and the second-highest SR (> 2.0).

In general, the consistently strong trading performance of FINMEM can be attributed to its innovative profile and memory module design. This design enables FINMEM to effectively integrate, comprehend, and prioritize key information from both textual and numerical data. Additionally, the memory module’s core features, including varied retention times for different information types and critical memory transitions, equip FINMEM to capture essential information for well-informed investment decisions.

# VI. ABLATION STUDIES

We conducted four distinct ablation studies to evaluate key component alternatives in FINMEM. The first three studies concentrated on the backbone algorithm, the memory module’s cognitive capacity, and the character setting in the profile module, particularly examining the aspect of risk inclination. These studies were done using the stock of Tesla, Inc., with a more compact training period from March 14, 2022, to June 15, 2022, and a testing period from June 16, 2022, to December 28, 2022. This shorter duration was chosen for budgetary efficiency, yet it remained sufficient to differentiate the functionality of each component. The final study provides a comprehensive case analysis demonstrating FINMEM’s end-to-end decision-making workflow on a specific trading day.

# A. FINMEM Backbone Algorithm Comparison (RQ3)

In our first study, we evaluated the trading performance of FINMEM using various LLMs as its backbone algorithms. The LLMs under consideration included davinci-003, GPT 3.5- Turbo, GPT4, GPT4-Turbo, and Llama2-70b-chat. The parameter settings were consistent with its optimal performance in the comparative experiment detailed in Section V, and the risk inclination was configured to be self-adaptive. All other model settings were maintained as outlined in Section V-A. The results of this evaluation are compiled in Table II.

Addressing RQ3, the findings demonstrate that FINMEM, powered by GPT-4 and GPT-4 Turbo, delivered superior trading results during the test phase. Specifically, GPT-4 recorded the highest cumulative return, while GPT-4-Turbo exhibited the most favorable Sharpe Ratio. GPT 3.5-Turbo’s performance was also noteworthy, following closely behind. As depicted in Fig. 6, though slightly lower than market baseline (B&H), FINMEM with GPT-4-Turbo led in cumulative returns before October 2022. This period was characterized by relative stability and a modest upward trend in TSLA stock. After October 2022, with

![](images/f56cafeb65ca9ee1f0665f69396142ff7aacbc67ed901fb563a4083227645609.jpg)  
Fig. 6. Comparison of CRs over time for FINMEM using different LLMs as backbones. FINMEM with GPT-4 achieves the highest CR by the end of the testing phase.

![](images/eb771fd559f636345b5455be4e6cb2b473cd2155d968803846226b708ccfc2a2.jpg)  
Fig. 7. Comparison of CRs with different risk inclination settings in FINMEM’s profile module. The self-adaptive risk setting helps FINMEM achieve the highest and the only positive CR by the end of the testing phase.

TSLA undergoing increased volatility and a notable downward trend, the cumulative return trajectory for FINMEM with GPT-4-Turbo exhibited significantly lower volatility and sustained stable returns not markedly lower than those of GPT-4. These results indicate that GPT-4 Turbo is the most suitable backbone algorithm for FINMEM.

FINMEM configured with davinci-003 and Llama2-70b-chat exhibited the lowest Annualized Volatility and Max Drawdown, yet their Cumulative Return and Sharpe Ratio were underwhelming. As illustrated in Fig. 6, both models defaulted to a “Hold” strategy beyond a certain point during periods of intense fluctuation in TSLA stock. The unsatisfactory performance of davinci-003 may be attributed to its limited capability, as an earlier generation language model, to capture and understand nuanced yet decisive information.

We selected Llama2-70b-chat as it was deemed to possess stronger in-context learning and instruction-following capabilities compared to other Llama family models with fewer parameters, as noted in Zhao et al. [68]. Nonetheless, in the context of stock trading, it still demonstrated challenges in adequately comprehending key messages necessary for effective trading decisions. The comparatively poorer performance of Llama2- 70b-chat can also be attributed to its shorter context window,

![](images/2bd77828c75856b0c146691bc55bcb95f2ba39bfd216e17be9bbece8f9858be0.jpg)  
Fig. 8. Comparison of CRs for FINMEM with different working memory capacity settings. The largest working memory capacity setting enabled FINMEM to achieve the highest CR by the end of the testing phase.

TABLE III TRADING PERFORMANCE COMPARISON WITH DIFFERENT RISK INCLINATION SETTINGS IN FINMEM’S PROFILE MODULE   

<table><tr><td>Model</td><td>CR (%)</td><td>SR</td><td>DV (%)</td><td>AV (%)</td><td>MDD (%)</td></tr><tr><td>B&amp;H</td><td>-66.9497</td><td>-2.0845</td><td>3.9527</td><td>3.8050</td><td>67.3269</td></tr><tr><td>Self-Adaptive</td><td>54.6958</td><td>2.4960</td><td>2.7419</td><td>2.5960</td><td>12.5734</td></tr><tr><td>Risk Seeking</td><td>-19.4132</td><td>-0.7866</td><td>3.2722</td><td>2.9236</td><td>45.0001</td></tr><tr><td>Risk Averse</td><td>-12.4679</td><td>-1.5783</td><td>1.7744</td><td>0.9358</td><td>15.9882</td></tr></table>

The self-adaptive risk setting helped FINMEM obtain the highest CR and SR.

especially when compared to the GPT models. When integrated with FINMEM, it needed to simplify prompts and shorten the length of retrieved memory insights, which could potentially result in some loss of context. The exceptional trading result demonstrated by GPT-4-Turbo across all models was a main factor in choosing it as the backbone algorithm for FINMEM in our earlier comparative analysis with other algorithmic trading agents.

# B. Influence About Varying the FINMEM Character Design (RQ4)

In our second study, we focused on evaluating the influence of FINMEM’s profile module on its trading effectiveness. Specifically, our assessment centered on the effects of customizing trader profiles according to specific stock trading, with a particular focus on risk inclination. As depicted in Fig. 13 in Appendix D, online available, we equipped FINMEM with three distinct risk profiles: risk-seeking, risk-averse, and a self-adaptive character. We executed a comparative analysis of FINMEM’s performance across these risk profiles, maintaining consistency in all other settings as outlined in Section V-A.

In response to RQ4, Table III illustrates the varied trading performance of TSLA across different risk profiles. The selfadaptive profile enabled FINMEM to achieve the most favorable trading performance, as it was the only one to secure a positive Cumulative Return and a Sharpe Ratio exceeding 2.0, along with the least Max Drawdown by the end of the testing phase. It further illustrates that FINMEM with self-adaptive risk setting can adeptly navigate substantial stock price volatility and strategically modulate its trading behavior when necessary. In contrast, the risk-seeking profile exhibited increased volatility

TABLE IV TRADING PERFORMANCE COMPARISON WITH VARIOUS SETTINGS OF WORKING MEMORY CAPACITY   

<table><tr><td>Model</td><td>CR (%)</td><td>SR</td><td>DV (%)</td><td>AV (%)</td><td>MDD (%)</td></tr><tr><td>B&amp;H</td><td>-66.9497</td><td>-2.0845</td><td>3.8050</td><td>60.4020</td><td>67.3269</td></tr><tr><td>Top1</td><td>52.0936</td><td>1.8642</td><td>3.3105</td><td>52.5529</td><td>25.2355</td></tr><tr><td>Top3</td><td>29.4430</td><td>1.1214</td><td>3.1105</td><td>49.3779</td><td>27.0972</td></tr><tr><td>Top5</td><td>54.6958</td><td>2.4960</td><td>2.5960</td><td>41.2100</td><td>12.5734</td></tr><tr><td>Top10</td><td>79.4448</td><td>2.7469</td><td>3.4262</td><td>54.3891</td><td>17.1360</td></tr></table>

The CR and SR show that expanding the working memory capacity helps FINMEM to improve trading performance.

and a decline in the face of a market downturn. The risk-averse profile, on the other hand, maintained a more conservative stance, often opting to hold positions. This approach resulted in a Cumulative Return trajectory that generally lagged behind the market baseline, reflecting a degree of overcaution that limited trading activity and potential gains, particularly in a bullish market.

The results presented in Fig. 13 in Appendix D, online available demonstrate that FINMEM’s profile module offers the flexibility to adjust risk inclinations, enhancing its ability to exploit rising market trends while safeguarding assets during downturns. For most stocks, FINMEM achieved optimal trading results under a self-adaptive risk setting. However, MSFT proved an exception, with a risk-seeking inclination yielding marginally better outcomes. Specifically, the Cumulative Return for MSFT under risk-seeking and self-adaptive settings were $2 3 . 2 6 1 3 \%$ and $1 9 . 1 4 7 \%$ respectively, while the corresponding Sharpe Ratios were 1.4402 and 1.03872. Both configurations outperformed other trading strategies. These outcomes resonated with MSFT’s generally bullish trend during the testing phase. In a predominantly positive market environment with substantial stock price increases, the risk-seeking profile enabled FINMEM to make more consistent and effective ‘Buy’ decisions, better capitalizing on market trends. This approach proved less susceptible to small historical price fluctuations or noisy market signals that can otherwise trigger ‘Hold’ decisions. Nevertheless, for the rest of the stocks experiencing downward or mixed trends, the self-adaptive configuration proved more effective. This setting allowed FINMEM to adopt a cautious strategy when faced with negative short-term cumulative returns. When short-term returns turned positive, FINMEM could switch to a more optimistic and assertive approach, thus avoiding excessive passivity while maintaining prudent risk management. This adaptive capability underscores FINMEM’s robustness across varied market conditions, demonstrating its potential to optimize trading strategies based on individual stock conditions and prevailing market trends.

# C. Impact of Adjusting the Capacity of FINMEM Working Memory (RQ5)

In our third study, we explored whether appropriately tuning the memory retrieval bandwidth of FINMEM could enhance its trading performance. This bandwidth is tied to the working memory’s capacity within its memory module. As depicted in Fig. 1, FINMEM retrieves the top- $K$ memory events from its

# Initialize Profile

# 1. Operations:

- Provide a performance overview of the trading stock based on available data.   
- Set up the risk inclination as the key character of the trading agent.   
2.Range:Financialinformationsuchasthefancialsectors,historicalperformance,andpreviousstocktrendsof thetradingstock   
3.Prompts:Youarean experiencedtrading managerandinvestmentfirm.Your task is tomakeinformed decisionsonthegiven stock based on the provided information.

(1).Self-AdaptiveCharacter Seting: When historical momentum is positive,you arearisk-seeking investor.But when historical momentum is negative, you are a risk-averse investor.   
(2). Risk-Seeking Character Setting: You are a risk-seeking investor.   
(3).Risk-Averse Character Setting:You are a risk-averse investor.   
4. General background setting:

Youhaveaccumulatedalotof informationabout the followingsectors,soyouareespeciallygoodattradingthem:l)Electric Vehicles (AutomotiveSector).2)EnergyGenerationandStorage.Fromyear2021to22September,Teslascontinuedgrowth and solid financial performance over the defined period ..

# Summarize

#

- Summarize different types of input information.

- Distribute them to corresponding layers of the long-term memory database.

2.Range: Daily market news,Long Documents such as company 10-K and 10-Q reports

# 3. Prompts:

- (1). Summarize the contents: Summarize the following documents into 10oo words.

- (2).Comprehend theinvestment sentimentof news insights:The positive,neutraland negativescores are forunderstandingthe investmentsntiments,opiion,oremotions.Forexample,positivenewsaboutacompanycanliftinvestorsentiment,enouraging more buying activity,which in turn can push stock prices higher...

# 4. Outputs:

# (1).To Shallow Memory Layer:

-[News (ID:261)] Here's How Much You Would Have Made Owning Tesla Stock In The Last10 Years Tesla (NASDAQ:TSLA) has outperformed the market over the past 1O years by $5 0 . 6 9 \%$ on an annualized basis producing an average annual return of $6 0 . 7 6 \%$ .Currently, Tesla has a market capitalization of $\$ 683.54$ billion.. The sentiment is {positive}.   
-[News (ID:278)]Tesla Q3 Earnings Are Imminent.Can Nio Foreshadow What'sToCome?WhatTo Know Before The Print TeslaInc (NASDAQ:TSLA)shares were tradingdownslightlyWednesdayafternoonaheadoftheautomakersthird-quarterreport, but the stock is up $6 \%$ over the last five sessions...The sentiment is {positive}.

# (2).To Intermediate Memory Layer:

- [Form 10-Q (ID:222)] Tesla Q3 2022 revenues were $\$ 21.5$ billion,up $56 \%$ year-over-year. Automotive sales revenue grew $56 \%$ to $\$ 123$ billion driven by higher Model 3/Y and Model S/X deliveries.Gross automotive margin declined to $2 7 . 9 \%$ due to cost inflation and factory ramps.Net income was $\$ 3.3$ billion,up $102 \%$ year-over-year.Positive free cash flow was $6.1 billion..

-[News (ID:275)TeslaQ3Earnings Highlights:RecordRevenue,Operating Margin AndFreeCashFlow,TeslaSemiDeliveries ComingInDecemberElectricvehicle leader TeslaInc (NASDAQ:TSLA)reportedthird-quarterfinancial resultsafter marketclose Wednesday... The sentiment is $\{ { \mathrm { n e u t r a l } } \}$

- [News (ID:274)]Tesla PrepsFor 2023 Cybertruck Launch,WillMake Batery Packs In California TheCybertruck isoneof Tesla Inc.(NASDAQ:TSLA) most hotly anticipated,but also most delayed, products.-.The sentiment is{negative}.

# (3). To Deep Memory Layer:

- [News(ID:161)] Tesla Whale Trades Spoted Awhale with alot of moneyto spend has takena noticeablybearish stance on Tesla.Looking atthe options historyforTesla (NASDAQ:TSLA)we detected 477 strange trades.Thesentiment is{positive}.   
-[Self-reflection (ID:226)] Given theshort-termpositivenews scoreinthe marketforTSLAandapositivecumulativereturn, thereisahighprobabilityofcontiuedgrowth intheshortterm.However,investorshouldbeawareof potential threats inthe mid-term market with competitors like General Motors,and Nio...

# Observe

1. Operations: Access and interpret market indicators such as current stock prices and historical momentum data.

2.Range: Stock's daily adjusted closing price, historical momentum in the past $k$ days $k = 3$ in this case), etc.

#

-The information below providesasummaryof stock pricefluctuations over the previous fewdays,which is the"momentum”of astock.Itreflectsthetrendofastock.Momentumisbasedontheideathatsecuritiesthathaveperformedwellinthepastwil continue to perform welland conversely,securities that have performed poorly will continue to perform poorly.

#

- (1). The daily adjusted closing price of TSLA on $\cdot$ is $\{ \$ 222.42\}$   
- (2).Train:On{20-10-25},temomentumofTLA, indicatedbytheprice diferencebetweenthecurrentandthenexttrading day,is $\{ \$ 2.22\}$

Test: O {2O22-10-25},the historical momentum of TSLA,as measuredbyits cumulative logarithmic returns in the past {3} days, was $\cdot$

Fig. 9. First section of FINMEM’s workflow for perceiving and processing multi-source information from market environment.

#

#

Train:-Inferthereasoning from theretrieved memories insights fortherecommendedtradingactions the increaseordecreaseof the trading stock price.

Test:- Generate recommended trading actions. - Infer the reasoning for making such trading action.

Train& Test: - Track the memory insight IDs that contributed to the current trading decision.

2.Range: Interpretation of observed market indicators and top $k$ insights from each layer of the long-term memory database.

#

Train:Giventhe follwinginformation,canyouexplainwhythefiancial marketfluctuation fromthecurentdaytothe nextday behaveslikethisSummarizethereason for thedecision.Youshould providesummary informationandtheIDsoftheinformation to support your summary.

Test:Given the information,canyou makeaninvestmentdecision?Justsummarizethereasonforthedecision.Consideronlythe available short-term information,themid-terminformation,the long-terminformation,ndthereflection-term information.

Consider themomentumof the historical stock price.Also,consider how much shareof the stock the investorholds now.You shouldprovideexactlyoneofthefollowinginvestment decisions:buyorsell.Youcould gowiththe’holdoptionifnecessary You also need to provide the IDs of the information to support your decision.

# 4. Outputs:

# Train:

-Recommended Action Reason:Therecommendedtrading actionis‘Buy'.Thiscan beatributed to positive market sentiment drivenbyarangeoffvorablefactors.otablyesla’sstrongbaterysuppychainefectiemarketexpansionplansineany anditsabilitytomaintainmarketstrengthdespitepotentialcompetitivethreatspositivelyinfluencedinvestorconfidence, contributing to the stock price rise.

- Reflected Contributed Insights IDs: {261,278, 275, 274,161... }

# Test:

# (1).Self-Adaptive Character Setting:

- Trading Action: {Buy}

-Recommended Action Reason:Consideringtheshort-term positiverevenuegrowth,higherModel3/Yand ModelS/Xdeliveries, positive freecash flow,and increasedcashandcashequivalents,Teslaappears to have strong financial health.The momentum being positive,italso encourages arisk-seeking strategyThe significant positive signsfrom news such as theimminent Q3 earningsaddtothefavorableviewof theinvestment.Despitethemid-termneutralsentiment,long-terminformationshowsonly marginaldropsandthestockisstilloking strongoverall.These,alongwithpositivereflection-terminformation,formthe reasoning behind the 'buy’decision...

- Reflected Contributed Insights IDs: {261,278,275,274,161... }

# (2).Risk-Seeking Character Setting:

Trading Action: {Buy)

-Recommended Action Reason:Considering the positivemomentum,strongthird-quarterresults,andthepotential forfuture growth fromnewfactoryproductionsand productlaunches,itseemstobeagoodtimetoinvestinTesla.Additionalyoveral positive short-term and reflection-term information support the buy decision.

-Reflected Contributed Insights IDs: {261, 278,161...}

# (3). Risk-Averse Character Setting:

- Trading Action: {Hold}

-Recommended Action Reason:Consideringthe short-term succesful financial metrics,positive momentum,and continuous growth,itseemspromsingforLA.Howeverosideringesaversearacteristicoftheinvestoradigificantgative sentiment inthemid-termandthe bearish stancebyawhale inthelong-term news,it'ssafer to hold the stocks for now.   
- Reflected Contributed Insights IDs: {278,275,274,161,226... }

Fig. 10. Second section of FINMEM’s workflow for generating trading action, reasoning and reflection.

long-term memory in response to a trading inquiry. The working memory capacity is thus set at $3 \times K$ , mirroring the human cognitive system’s limit of processing immediate messages upon specific stimuli (3 refers to the three processing layers in longterm memory). By varying the $K$ hyperparameter, FINMEM can expand this capacity far beyond the human cognitive scope. We aimed to determine whether such flexibility in adjusting memory bandwidth translates to improvements in FINMEM’s performance.

As demonstrated in Table IV, we adjusted the hyperparameter $K$ to alter the number of memory events retrieved from shallow, intermediate, and deep long-term memory layers in FINMEM. We tested $K$ values of 1, 3, 5, and 10, exploring FINMEM’s working memory capabilities at levels below, near, and above the human cognitive limit. For all these $K$ settings, we maintained a selfadaptive risk inclination, while other settings were consistent with those described in Section V-A.

Across all $K$ configurations, FINMEM outperformed the Buy & Hold baseline, indicating the effectiveness of its memory module in processing diverse information and capturing critical

events, which subsequently enhanced its trading performance, as evidenced by positive Cumulative Returns and Sharpe Ratios. Notably, higher $K$ values, like 5 and 10, enabled FINMEM to achieve the best Cumulative Returns and Sharpe Ratios exceeding 2.0. With $K$ set to 1, FINMEM still performed moderately well by capturing the most critical memory events of each layer.

An in-depth analysis in Fig. 8, which shows the Cumulative Return over time for various $K$ settings, reveals that a $K$ value of 5 is optimal for trading TSLA stock, consistently delivering robust performance with the lowest Volatility and Max-Drawdown. Before mid-October 2022, when the stock market was relatively stable and slightly upward, FINMEM’s trading actions aligned well with market trends (referring to B&H) and avoided significant losses. During periods of high volatility and continuous downturns (post-mid-October 2022), it maintained earnings by reducing ”Buy” actions and favoring more ”Hold” and ”Sell” strategies. However, setting $K$ to 10, while effective during market volatility, resulted in significant losses in stable market conditions. The issue may stem from the disproportionately loose capacity constraints on incoming

information relative to the volume of incoming data. The broad memory retrieval bandwidth might have mixed trivial messages with critical ones, hampering FINMEM’s decision precision. This challenge becomes especially evident in neutral market conditions, where the influx of information includes a mix of varying market sentiments and trends.

Addressing RQ5, appropriately tuning the number of memory events (Top- $K$ ) in the FINMEM memory module can significantly enhance its trading performance. The aforementioned study illustrates that FINMEM can achieve optimal results by effectively assimilating key signals from a sufficient quantity of filtered memories across each layer. However, the optimal value for $K$ may vary depending on the volume and quality of incoming information.

# D. Case Study: Forecast for TSLA on 2022-10-25 to Predict Trading Decision on 2022-10-26

In the last study, we presented the trading decision-making process for TSLA on October 25, 2022, to forecast whether to ’buy’ or ’sell’ on October 26, 2022. We outlined the workflow, presented the prompt template for FINMEM, and compared outcomes across three distinct risk profiles.

We present the complete workflow of FINMEM in two sections. The first section, depicted in Fig. 9, illustrates how FINMEM utilizes its layered memory module and customized prompt templates to effectively manage various operational mechanisms. These mechanisms are designed to perceive, organize, and distill trading-relevant insights from multi-source market information. The second section, illustrated in Fig. 10, outlines the steps FINMEM takes after extracting investmentrelated insights from the market, as described in the first section. It elaborates on reflection outcomes generated upon three risk profile options of FINMEM.

As shown in Figs. 9 and 10, FINMEM processes the latest daily news about TSLA through its shallow processing layer within the long-term memory database. The Form 10-Q released on October 24, 2022, is processed through the intermediate layer, while access-counter reinforced insights and reflective reasoning are accelerated in the deep processing layer. Given the same input information, FINMEM executes distinct trading actions under varying risk profiles. With positive historical momentum and generally favorable market signals, FINMEM adopts a risk-tolerant stance and executes a ’Buy’ action, consistent with its risk-seeking profile. This decision proves profitable, as the stock price increases by $\$ 2.22$ the following trading day. However, the rationale behind this decision and the insight IDs recognized as key contributors vary between the two profile options. FINMEM under a risk-seeking profile displays a degree of impulsiveness by disregarding negative market information in its decision-making process. In contrast, the self-adaptive setting demonstrates a more balanced approach, considering both positive and negative aspects of available insights. This more holistic view is achieved by incorporating reminders of the ’risk-averse’ condition in the prompt for the self-adaptive profile. Meanwhile, in a risk-averse profile, FINMEM opts for a ’Hold’ decision, prioritizing caution over higher profits.

# VII. CONCLUSION AND FUTURE WORK

In this paper, we introduce FINMEM, an innovative automated trading agent framework featuring an adjustable cognitive memory structure and dynamic character design. Our experiments demonstrate its exceptional proficiency in sequential financial decision-making tasks, particularly stock trading. Compared to other SOTA algorithmic trading agents, FINMEM achieves significantly higher key performance metrics on realworld financial datasets across multiple stock assets. Through ablation studies, we thoroughly assess the efficacy of each critical component within FINMEM, revealing their substantial contributions to enhancing stock trading performance.

Its unique features of human-like cognitive memory modules and dynamic character design enable it to tackle the complexities of financial environments and respond aptly to new situations. Compared to other LLM trading agents, FINMEM’s memory module equips it with the capability to better comprehend financial data featured by various timeliness and organize them as a self-evolving long-term memory layer. The dynamic character design endows FINMEM with critical professional insights, enabling efficient filtering of impactful messages from incoming financial data for trading actions. Additionally, the integration of multiple risk profiles enhances FINMEM’s adaptability to a range of market conditions.

FINMEM’s exceptional performance demonstrates its ability to transform diverse financial data into well-informed investment strategies. Its proficiency in integrating various data types is further highlighted by a reduced training duration, which benefits trading with newly established companies. We used a limited range and quality of financial news and reports, employing general-purpose LLMs as the backbone algorithms. Yet, FINMEM is fully compatible with LLMs fine-tuned for financial applications. We anticipate its trading efficacy will improve with access to a more comprehensive and higher-quality dataset, along with LLMs tailored for financial contexts.

Primarily designed for financial decision-making, the huge potential for FINMEM to handle various tasks could be expanded in future research across multiple dimensions. From the perspective of broadening the range of trading assets, FINMEM could extend beyond stock trading to include other financial products such as cryptocurrencies and Exchange-Traded Funds (ETFs). Additionally, from the standpoint of diversifying decisionmaking applications, FINMEM could be adapted to address more complex tasks like portfolio management and stock selection. This expansion would require further development of prompt templates and the integration of external tool-use functions in LLMs, such as API calls and tabular data reading capabilities. Moreover, FINMEM offers an innovative and flexible framework that extends beyond decision-making to accommodate various financial tasks. Its adaptable profile module, built upon LLM prompts, is naturally suited for articulating nuanced risk characteristics in explainable texts for market participant agents. Additionally, the memory module, aligned with human cognitive architecture, can be replicated across multiple agents to facilitate peer-to-peer communication and self-reflection. These features enable FINMEM to conduct financial market simulations in a more interpretable manner.

# REFERENCES

[1] F. Black, “Noise,” J. Finance, vol. 41, no. 3, pp. 528–543, 1986.   
[2] R. D. Edwards, J. Magee, and W. C. Bassetti, Technical Analysis of Stock Trends. Boca Raton, FL, USA: CRC Press, 2018.   
[3] B. Huang, Y. Huan, L. D. Xu, L. Zheng, and Z. Zou, “Automated trading systems statistical and machine learning methods and hardware implementation: A survey,” Enterprise Inf. Syst., vol. 13, no. 1, pp. 132–144, 2019.   
[4] T. G. Fischer, “Reinforcement learning in financial markets-a survey,” FAU Discussion Papers Economics, Tech. Rep. (No. 12/2018), 2018.   
[5] A. Millea, “Deep reinforcement learning for trading—a critical survey,” Data, vol. 6, no. 11, 2021, Art. no. 119.   
[6] S. Balhara et al., “A survey on deep reinforcement learning architectures, applications and emerging trends,” IET Commun., 2022.   
[7] S. J. Gershman and B. P. Ölveczky, “The neurobiology of deep reinforcement learning,” Curr. Biol., vol. 30, no. 11, pp. R629–R632, 2020.   
[8] OpenAI, “Gpt-4 technical report,” 2023.   
[9] L. Wang et al., “A survey on large language model-based autonomous agents,” Front. Comput. Sci., vol. 18, no. 6, 2024, Art. no. 186345.   
[10] Y. Li et al., “Making language models better reasoners with step-aware verifier,” in Proc. 61st Annu. Meeting Assoc. Comput. Linguistics, 2023, pp. 5315–5333.   
[11] H. Yang, X. -Y. Liu, and C. D. Wang, “Fingpt: Open-source financial large language models,” 2023, arXiv:2306.06031.   
[12] S. Wu et al., “Bloomberggpt: A large language model for finance,” 2023, arXiv:2303.17564.   
[13] G. A. Miller, “The magical number seven, plus or minus two: Some limits on our capacity for processing information,” Psychol. Rev., vol. 63, no. 2, 1956, Art. no. 81.   
[14] J. S. Park, J. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, “Generative agents: Interactive simulacra of human behavior,” in Proc. 36th Annu. ACM Symp. User Interface Softw. Technol., New York, NY, USA, in UIST ’23, 2023, pp. 1–22, doi: 10.1145/3586183.3606763.   
[15] F. I. Craik and R. S. Lockhart, “Levels of processing: A framework for memory research,” J. Verbal Learn. Verbal Behav., vol. 11, no. 6, pp. 671–684, 1972.   
[16] J. Sweller, “Human cognitive architecture: Why some instructional procedures work and others do not,” in APA Educational Psychology Handbook, Vol. 1. Theories, Constructs, and Critical Issues. K. R. Harris, S. Graham, T. Urdan, C. B. McCormick, G. M. Sinatra, and J. Sweller, Eds., American Psychological Association, 2012, pp. 295–325.   
[17] R. Sun, “Desiderata for cognitive architectures,” Philos. Psychol., vol. 17, no. 3, pp. 341–373, 2004.   
[18] T. -L. Chen, “Forecasting the Taiwan stock market with a novel momentum-based fuzzy time-series,” Rev. Econ. Finance, vol. 2, pp. 38–50, 2012.   
[19] R. Vaidya, “Moving average convergence-divergence (MACD) trading rule: An application in Nepalese stock market ‘NEPSE’,” Quantitative Econ. Manage. Stud., vol. 1, no. 6, pp. 366–374, 2020.   
[20] E. Pätäri and M. Vilska, “Performance of moving average trading strategies over varying stock market conditions: The finnish evidence,” Appl. Econ., vol. 46, no. 24, pp. 2851–2872, 2014.   
[21] S. Ali, M. Naveed, A. Saleem, and M. W. Nasir, “Time-frequency comovement between COVID-19 and Pakistan’s stock market: Empirical evidence from wavelet coherence analysis,” Ann. Financial Econ., vol. 17, no. 04, 2022, Art. no. 2250026.   
[22] M. Naveed, S. Ali, K. Iqbal, and M. K. Sohail, “Role of financial and non-financial information in determining individual investor investment decision: A signaling perspective,” South Asian J. Bus. Stud., vol. 9, no. 2, pp. 261–278, 2020.   
[23] F. Abbas, S. Ali, S. Moudud-Ul-Huq, and M. Naveed, “Nexus between bank capital and risk-taking behaviour: Empirical evidence from us commercial banks,” Cogent Bus. Manage., vol. 8, no. 1, 2021, Art. no. 1947557.   
[24] M. F. Farah, M. Naveed, and S. Ali, “Blockchain-enabled banking services and customers’ perceived financial well-being: A structural nexus,” in Proc. Nat. Brand Private Label Marketing Conf., 2023, pp. 41–49.   
[25] D. Y. Aharon, S. Ali, and M. Naved, “Too big to fail: The aftermath of Silicon Valley Bank (SVB) collapse and its impact on financial markets,” Res. Int. Bus. Finance, vol. 66, 2023, Art. no. 102036.   
[26] M. Naveed, S. Ali, M. Gubareva, and A. Omri, “When giants fall: Tracing the ripple effects of Silicon Valley Bank (SVB) collapse on global financial markets,” Res. Int. Bus. Finance, vol. 67, 2024, Art. no. 102160.   
[27] S. Ali, M. Naveed, H. Hanif, and M. Gubareva, “The resilience of shariahcompliant investments: Probing the static and dynamic connectedness between gold-backed cryptocurrencies and GCC equity markets,” Int. Rev. Financial Anal., vol. 91, 2024, Art. no. 103045.

[28] P. Zhang, X. Shi, and S. U. Khan, “QuantCloud: Enabling Big Data complex event processing for quantitative finance through a datadriven execution,” IEEE Trans. Big Data, vol. 5, no. 4, pp. 564–575, Dec. 2019.   
[29] M. Prata et al., “Lob-based deep learning models for stock price trend prediction: A benchmark study,” Artif. Intell. Rev., vol. 57, no. 5, pp. 1–45, 2024.   
[30] G. Sonkavde, D. S. Dharrao, A. M. Bongale, S. T. Deokate, D. Doreswamy, and S. K. Bhat, “Forecasting stock market prices using machine learning and deep learning models: A systematic review, performance analysis and discussion of implications,” Int. J. Financial Stud., vol. 11, no. 3, 2023, Art. no. 94.   
[31] Q. -V. Dang, “Reinforcement learning in stock trading,” in Proc. Int. Conf. Comput. Sci., Appl. Math. Appl., 2019, pp. 311–322.   
[32] O. Jangmin, J. Lee, J. W. Lee, and B. -T. Zhang, “Adaptive stock trading with dynamic asset allocation using reinforcement learning,” Inf. Sci., vol. 176, no. 15, pp. 2121–2147, 2006.   
[33] Y. Shi, W. Li, L. Zhu, K. Guo, and E. Cambria, “Stock trading rule discovery with double deep q-network,” Appl. Soft Comput., vol. 107, 2021, Art. no. 107320.   
[34] H. Yang, X. -Y. Liu, S. Zhong, and A. Walid, “Deep reinforcement learning for automated stock trading: An ensemble strategy,” in Proc. 1st ACM Int. Conf. AI Finance, 2020, pp. 1–8.   
[35] X.-Y. Liu et al., “FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance,” in Proc. 2nd ACM Int. Conf. AI Finance, 2021, pp. 1–9.   
[36] T. -V. Pricope, “Deep reinforcement learning in quantitative algorithmic trading: A review,” 2021, arXiv:2106.00123.   
[37] Y. -F. Chen and S. -H. Huang, “Sentiment-influenced trading system based on multimodal deep reinforcement learning,” Appl. Soft Comput., vol. 112, 2021, Art. no. 107788.   
[38] L. Avramelou, P. Nousi, N. Passalis, and A. Tefas, “Deep reinforcement learning for financial trading using multi-modal features,” Expert Syst. Appl., vol. 238, 2023, Art. no. 121849.   
[39] J. Devlin, M. -W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proc. 2019 Conf. North Amer. Chapter Assoc. Comput. Linguistics: Human Lang. Technol., 2019, pp. 4171–4186.   
[40] K. Ethayarajh, “How contextual are contextualized word representations? Comparing the geometry of BERT, ELMo, and GPT-2 embeddings,” in Proc. 2019 Conf. Empir. Methods Natural Lang. Process. 9th Int. Joint Conf. Natural Lang. Process., Hong Kong, China, 2019, pp. 55–65.   
[41] Q. Xie et al., “Pixiu: A comprehensive benchmark, instruction dataset and large language model for finance,” in Proc. Adv. Neural Inform. Process. Syst., 2023, vol. 36, pp. 33469–33484.   
[42] Y. Li, S. Wang, H. Ding, and H. Chen, “Large language models in finance: A survey,” in Proc. 4th ACM Int. Conf. AI Finance, 2023, pp. 374–382.   
[43] Y. Nie et al., “A survey of large language models for financial applications: Progress, prospects and challenges,” 2024, arXiv:2406.11903.   
[44] H. Yang et al., “Finrobot: An open-source AI agent platform for financial applications using large language models,” 2024, arXiv:2405.14767.   
[45] Y. Zhou et al., “Are large language models rational investors?,” 2024, arXiv:2402.12713.   
[46] J. S. Park, J. C. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, “Generative agents: Interactive simulacra of human behavior,” in Proc. 36th Annu. ACM Symp. User Interface Softw. Technol., 2023, pp. 1–22.   
[47] J. Li et al., “Agent hospital: A simulacrum of hospital with evolvable medical agents,” 2024, arXiv:2405.02957.   
[48] S. Hong et al., “MetaGPT: Meta programming for multi-agent collaborative framework,” in Proc. 12th Int. Conf. Learn. Representations, 2023.   
[49] Z. Xi et al., “The rise and potential of large language model based agents: A survey,” Sci. China Inform. Sci., vol. 68, no. 2, 2025, Art. no. 121101.   
[50] Q. Wu et al., “Autogen: Enabling next-gen LLM applications via multiagent conversation framework,” in Proc. 1st Conf. Lang. Model., 2024.   
[51] S. Qiao et al., “Autoact: Automatic agent learning from scratch via selfplanning,” 2024, arXiv:2401.05268.   
[52] W. Zhang et al., “Finagent: A multimodal foundation agent for financial trading: Tool-augmented, diversified, and generalist,” in Proc. 30th ACM SIGKDD Conf. Knowl. Discov. Data Mining, 2024, pp. 4314–4325.   
[53] X. Liu et al., “When ai meets finance (stockagent): Large language modelbased stock trading in simulated real-world environments,” ResearchGate Preprint, 2024, doi: 10.13140/RG.2.2.29976.20489.   
[54] J. M. Murre and J. Dros, “Replication and analysis of Ebbinghaus’ forgetting curve,” PLoS One, vol. 10, no. 7, 2015, Art. no. e0120644.   
[55] “AI guardrails,” open source library for interacting with large language models, (n.d.). [Online]. Available: https://docs.guardrailsai.com

[56] J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity search with GPUs,” IEEE Trans. Big Data, vol. 7, no. 3, pp. 535–547, Jul. 2021.   
[57] X. -Y. Liu, H. Yang, J. Gao, and C. D. Wang, “FinRL: Deep reinforcement learning framework to automate trading in quantitative finance,” in Proc. ACM Int. Conf. AI Finance, 2021, pp. 1–9.   
[58] X. -Y. Liu et al., “Finrl-meta: Market environments and benchmarks for data-driven financial reinforcement learning,” in Proc. Adv. Neural Inf. Process. Syst., 2022, vol. 35, pp. 1835–1849.   
[59] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” 2017, arXiv:1707.06347.   
[60] J. Chung, “Playing atari with deep reinforcement learning,” Comput. Ence, vol. 21, no. 351-362, 2013, Art. no. 2.   
[61] V. Mnih et al., “Asynchronous methods for deep reinforcement learning,” in Proc. Int. Conf. Mach. Learn., 2016, pp. 1928–1937.   
[62] J. S. Park, L. Popowski, C. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, “Social simulacra: Creating populated prototypes for social computing systems,” in Proc. 35th Annu. ACM Symp. User Interface Softw. Technol., 2022, pp. 1–18.   
[63] J. Hull, Risk Management and Financial Institutions. Hoboken, NJ, USA: John Wiley & Sons, Inc., 2007.   
[64] W. F. Sharpe, “The sharpe ratio,” J. Portfolio Manage., vol. 21, no. 1, pp. 49–58, 1994.   
[65] J. H. Cochrane, “Volatility tests and efficient markets: A review essay,” J. Monetary Econ., vol. 22, no. 3, pp. 463–485, 1988.   
[66] A. Ang and J. Chen, “Downside risk,” J. Portfolio Manage., vol. 29, no. 4, pp. 103–112, 2003.   
[67] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-baselines3: Reliable reinforcement learning implementations,” J. Mach. Learn. Res., vol. 22, no. 268, pp. 1–8, 2021. [Online]. Available: http://jmlr.org/papers/v22/20-1364.html   
[68] W. X. Zhao et al., “A survey of large language models,” 2023, arXiv:2303.18223.

![](images/e2929909bc2da6ccd72cac9df3b0c14eb3d249167a91c3858a639636a0844f61.jpg)

Yangyang Yu received the bachelor’s degree in applied mathematics from Xiamen University, and the master’s degree in applied statistics from Syracuse University. She is currently working toward the PhD degree in data science with the Stevens Institute of Technology. Her research interests include cognitive science, language agent design, bayesian statistics, and multi-modal learning, focusing on FinTech applications. She is a program committee member for workshops at IJCAI and COLING.

![](images/7774bf35cdf226a503fe08c63a380a3d64aedd9a4a48c86bfc7ea2e15ca6e98d.jpg)

Haohang Li received the MS degree in financial engineering (with distinction) from the Stevens Institute of Technology, where he is currently working toward the PhD degree in data science. His research interests include natural language processing, large language models, and their applications in FinTech. He is an organizer for COLING and IJCAI workshops and a program committee member for the INFORMS Workshop.

![](images/b6726cc129d9927db7928adede43e11199aa039c8c19a3a0e007b604354b4e1e.jpg)

Zhi Chen received the BS degree in physics from Guangzhou University, China, and the MS degree in financial engineering from the Stevens Institute of Technology, USA, where he is currently working toward the PhD degree in financial engineering. He has authored or coauthored paper in the journal Research in International Business and Finance. His research interests include explainable AI, fintech, ESG, and large language models.

![](images/22ac435849674f220d77193d9b0ff0ba0ce0d674f21bfcaaee4ec97ee0c039b1.jpg)

Yuechen Jiang received the BS degree in financial engineering from the Shenyang University of Technology, China, and the two MS degrees in financial engineering and machine learning from the Stevens Institute of Technology. She is currently working toward the PhD degree with the NJIT’s Ying Wu College of Computing, focusing on time-series with graph neural networks in financial forecasting. Her research interests include time series with graph neural networks and multi-agent LLM systems.

![](images/a3ee2f9ac7968e7fff8dbaeebdab74e29696785e9ef81d2577af7b920640aa29.jpg)

Yang Li received the master’s degree in mathematical finance from the University of Southern California, in 2019. He is currently working toward the PhD degree in financial engineering with the Stevens Institute of Technology. His current research interests include the intersection of high-frequency trading, neural networks, and blockchain technology in financial markets.

![](images/54abaa718faa35b893657dc6bd979d47c0ac99c0c813ba92cf08e2f2c336ca69.jpg)

Jordan W. Suchow is currently an assistant professor of information systems with the School of Business, Stevens Institute of Technology, focusing on the intersection of cognitive science and information systems in his research and teaching. In UC Berkeley, his affiliations extended to the Institute of Cognitive and Brain Sciences, the Social Science Matrix, the Berkeley Artificial Intelligence Research lab, and the Center for Technology, Society & Policy. His research interests include information systems, cognitive science, A.I., and behavioral experiments.

![](images/ae69efb369f24893b49f63c96c763096ffc78e28860e16d7b6102c7d7be4f3b5.jpg)

Denghui Zhang received the BE degree from the University of Science and Technology, Beijing, China, in 2015, the MS degree from the Chinese Academy of Sciences, China, in 2018, and the PhD degree from Rutgers, the State University of New Jersey, in 2023. He is currently an assistant professor with the Stevens Institute of Technology. He has authored or coauthored prolifically in refereed journals and conference proceedings, such as IEEE TRANS-ACTIONS ON KNOWLEDGE AND DATA ENGINEERING, ACM SIGKDD, AAAI, SIGIR, CIKM, and ICDM.

His research interests include data mining, business analytics, natural language processing, and representation learning.

![](images/7f85f72c9ebeaf70799e39c805662809f2350d000952e7ef34697a213053829e.jpg)

Khaldoun Khashanah is currently the director with Financial Engineering Division, Stevens Institute of Technology. He has helped to develop one of the largest programs in the U.S. He established a comprehensive financial engineering curriculum, offering a graduate certificate, master’s degree, and in 2009, one of the early PhD programs in financial engineering. He is also the Co-Project Director of the Hanlon Financial Systems Laboratory.
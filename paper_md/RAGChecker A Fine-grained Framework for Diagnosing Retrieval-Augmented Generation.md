# RAGCHECKER: A Fine-grained Framework for Diagnosing Retrieval-Augmented Generation

Dongyu $\mathbf { R } \mathbf { u } ^ { 1 * }$ Lin Qiu1∗ Xiangkun $\mathbf { H } \mathbf { u } ^ { 1 * }$ Tianhang Zhang1∗ Peng Shi1∗ Shuaichen Chang1∗ Cheng Jiayang1† Cunxiang Wang1† Shichao Sun2 Huanyu $\mathbf { L i } ^ { 2 }$ Zizhao Zhang1† Binjie Wang1† Jiarong Jiang1 Tong He1 Zhiguo Wang1 Pengfei Liu2 Yue Zhang3 Zheng Zhang1

1Amazon AWS AI 2Shanghai Jiaotong University 3Westlake University

# Abstract

Despite Retrieval-Augmented Generation (RAG) showing promising capability in leveraging external knowledge, a comprehensive evaluation of RAG systems is still challenging due to the modular nature of RAG, evaluation of long-form responses and reliability of measurements. In this paper, we propose a fine-grained evaluation framework, RAGCHECKER, that incorporates a suite of diagnostic metrics for both the retrieval and generation modules. Meta evaluation verifies that RAGCHECKER has significantly better correlations with human judgments than other evaluation metrics. Using RAGCHECKER, we evaluate 8 RAG systems and conduct an indepth analysis of their performance, revealing insightful patterns and trade-offs in the design choices of RAG architectures. The metrics of RAGCHECKER can guide researchers and practitioners in developing more effective RAG systems3.

# 1 Introduction

Retrieval-Augmented Generation (RAG) systems [18, 7] enhance Large Language Models (LLMs) by incorporating external knowledge bases, enabling more precise and contextually relevant responses [7, 53, 13]. As these systems become integral to a variety of applications [54, 2, 8], it’s imperative to develop robust and comprehensive evaluation frameworks to assess their performance and identify areas for improvement. Evaluating RAG systems, however, presents several challenges:

(1) modular complexity: The modular nature of RAG systems, comprising both a retriever and a generator, complicates the design of effective evaluation metrics. It is crucial to establish metrics that can holistically assess the entire system as well as evaluate the individual modules and their interplay [53], allowing for fully understanding the sources of the errors and misses and how they are generated. (2) metric limitation: Existing metrics for evaluating RAG systems, which are often rule-based or coarse-grained, fall short in providing accurate and interpretable results. Specifically, traditional metrics like recall $@ \mathbf { k }$ and MRR [44] for retrievers depend on annotated chunks and a rigid chunking approach, missing out on the full semantic scope of the knowledge base. For generators, typical measures such as n-gram-based (e.g., BLEU [30], ROUGE [19]), embedding-based (e.g., BERTScore [56]), and LLM-based methods [45] perform well with concise answers but fail to detect finer distinctions in longer responses. To bridge these gaps, it is essential to develop detailed, semantic-based evaluation metrics that effectively capture the intricacies and overall quality of both the retrieval and generation components in RAG systems. (3) metric reliability: the reliability

of existing metrics for RAG remains under-explored. Effective evaluation metrics must not only accurately reflect system performance but also align with human judgments to ensure their utility in real-world scenarios.

To overcome these challenges, we introduce RAGCHECKER, an innovative evaluation framework designed for detailed analysis of both retrieval and generation processes. RAGCHECKER is based on claim-level entailment checking which involves operations of extracting claims from the response and ground truth answer and checking them against other texts. This approach enables fine-grained evaluation instead of response-level assessment. RAGCHECKER processes the user query, retrieved context, response, and ground truth answer, producing a suite of metrics:

1. Overall Metrics to provide a holistic view of the system performance, assessing the overall quality of the generated responses.   
2. Diagnostic Retriever Metrics to evaluate the effectiveness of the retriever, identifying its strengths and weaknesses in finding relevant information from the knowledge base.   
3. Diagnostic Generator Metrics to assess the performance of the generator, diagnosing how well the generator utilizes the retrieved context, handles noisy information, and generates accurate and faithful responses.

Compared to existing evaluation frameworks, RAGCHECKER provides a more comprehensive assessment of RAG systems. While some frameworks offer fine-grained evaluation only on certain metrics (e.g., RAGAS [5], TruLens [6], ARES [35]) or evaluate specific aspects of RAG (e.g., RGB [4], RECALL [22], NoMIRACL [40]), RAGCHECKER’s metrics are all based on fine-grained claim-level checking and are designed to provide actionable insights into the sources of errors.

To ensure the reliability of RAGCHECKER, we annotate a human judgment dataset to assess the correlations between the proposed metrics and human judgments. This meta-evaluation validates the effectiveness of RAGCHECKER in capturing the quality and reliability of RAG systems from a human perspective. We demonstrate the effectiveness of RAGCHECKER through comprehensive experiments evaluating 8 state-of-the-art RAG systems on a benchmark repurposed from public datasets across 10 domains. In-depth analysis of the evaluation results reveals that RAGCHECKER provides insightful diagnostic signals (Sec. 4.3) pointing the directions for improvements of RAG systems (Sec. 4.4).

The main contributions of this paper are as follows:

• We propose RAGCHECKER, a novel RAG evaluation framework that offers fine-grained evaluation for both the retriever and generator components, introducing new diagnostic metrics to provide actionable insights into the sources of errors.   
• We conduct meta evaluation and verified RAGCHECKER has significantly better correlations with human judgements than other evaluation metrics.   
• We perform extensive experiments evaluating 8 RAG systems on our curated benchmark across 10 domains, and uncover valuable insights, such as the trade-off between retrieval improvement and noise introduction, and the tendency of faithful open-source models to blind trust on context.

# 2 Related Work

# 2.1 Retrieval Augmented Generation

Large Language Models (LLMs) demonstrate strong capabilities in generating text, but there are also obstacles such as outdated information and the potential to hallucinate [42, 46, 12]. To address these issues, RAG retrieves external knowledge to generate responses with improved accuracy and factuality [7, 53, 13]. Integrating external knowledge is especially crucial in fields like legal, medical and finance, where precision and reliability are essential [24, 50, 55].

RAG systems have shown impressive performance across a range of tasks, including open-domain question answering [27, 10, 18], code generation [32, 57, 38] and dialogue [37, 16, 41]. Additionally, real world products like Bing Search4 and Langchain [3] have integrated applications based on RAG.

# 2.2 Evaluation of RAG

Existing evaluation practices for RAG systems can be categorized into two main approaches: evaluating essential capabilities of generators only and assessing end-to-end performance of RAG systems.

Within the two components of a RAG system, the retriever has been well studied in recent years, thus a line of recent work focused on evaluating essential generator capabilities. RGB [4] evaluated 4 fundamental abilities required for generators including Noise Robustness, Negative Rejection, Information Integration and Counterfactual Robustness by manually constructed test sets. RECALL [22] introduced manually edited counterfactual contexts into QA and text generation datasets to evaluate the counterfactual robustness of LLMs. NoMIRACL [40] evaluated LLMs’ robustness against firststage retrieval errors of RAG systems with manually judged relevant and non-relevant datasets. Wu et al. [49] quantified the tug-of-war between LLMs’ faithfulness and internal prior by introducing varying levels of perturbations on the provided contexts. FaaF [15] introduced a fine-grained fact verification formulation to improve previous prompting-based approaches in evaluating factuality of generators. However, we argue that above generator-only evaluation approaches with manually constructed datasets cannot serve as a general RAG evaluation framework to reveal the entanglement of between generation results and different retrieval behaviors, as shown in the analysis of Sec. 4.3.

Another line of work focused on assessing end-to-end quality scores of RAG systems. TruLens [6] introduced the concept of RAG Triad, which decompose the quality scores into three aspects: context relevance, groundedness and answer relevance, then predicted the score by prompting LLMs or using NLI models. RAGAS [5] and ARES [35] followed the RAG Triad concept and improved the score prediction approaches on different datasets. CRUD-RAG [25] refered to the CRUD (Create, Read, Update and Delete) actions between users and knowledge bases to develop corresponding datasets and evaluation metrics for RAG systems. We compare the above four evaluation frameworks with RAGCHECKER in the meta evaluation of Sec. 4.2.

Besides, the following work also provided good insight or high quality datasets for end-to-end RAG evaluation. Liu et al. [21] conducted human evaluation to audit four popular generative search engines in terms of fluency, perceived utility, and verifiability. MEDRAG [50] constructed a medical RAG benchmark from medical QA datasets and evaluated medical RAG systems with QA accuracy. MultiHop-RAG [39] generated multi-hop queries from news articles and evaluated RAG systems with QA accuracy. CDQA [52] proposed a novel approach to generate dynamic QA questions which requires latest information to answer. However, the evaluation metrics used in the work mentioned above rely either on human evaluation or simple textual accuracy, making them incapable of complex RAG scenarios that require long answer evaluation. Therefore, we do not include them in the meta evaluation.

# 3 RAGCHECKER Framework

Formulation Define a modular RAG system as ${ \bf R A G } = \{ { \bf R } , { \bf G } \}$ , where R is the retriever and G is the generator. Given a query $q$ and documents $D$ , it first retrieves top- $k$ relevant context $\{ \mathrm { c h u n k } _ { j } \} \bar { { \bf \Phi } } = { \bf R } ( q , D , k )$ , and then generates a model response $\mathbf { m } = \mathbf { G } ( \{ \mathrm { c h u n k } _ { j } \} , q )$ . For simplicity, we can also represent the overall RAG generation process as $\mathbf { m } = \mathbf { R A G } ( q , D )$ .

Design Principle Given the compositional nature of RAG, we observe there are two major personae using a RAG evaluation framework. The first persona is a user that cares about the overall performance of RAGs and might choose a system with the best performance. Such a persona prefers a single value metric to compare and rank among RAG systems against a benchmark. The second persona is a developer that focuses on improving a RAG system with the need to identify causes of mistakes and potential rooms for improvements. Causes of errors in response can be classified into 1) retrieval errors, where the retriever fails to return complete and relevant context, and 2) generator errors, where the generator struggles to identify and leverage relevant information from context.

Consequently, metrics that reveal error causes should be different from those for overall performance, in the sense that error causes are module-specific or even reflected only by a certain behavior of a module. To help both personae to assess RAG performance, we design RAGCHECKER , a evaluation framework of RAG systems that consists of a benchmark with rich annotations and a set of diversely-purposed fine-grained metrics.

# 3.1 Inputs to RAGCHECKER

We prepare each sample in our benchmark dataset in the format of a tuple $\langle q , D , g t \rangle$ representing query, documents, and ground-truth answer, where query is the input question to a RAG system, documents form the database providing possible context and are processed into chunks with the same number of tokens, and ground-truth answer is a complete and correct answer for the input question. Further information is provided in Sec. 4.1.

# 3.2 Fine-grained Evaluation with Claim Entailment

As illustrated in Fig. 1, a response generated by a RAG system might be a mixture of correct $\left( \bigcirc \right)$ and incorrect claims $( \times )$ , while also missing some in-ground-truth claims ( ). In this sense, evaluating responses at a finer granularity is crucial to comprehensively assess the quality of an answer. For this purpose, we introduce two components: 1) a text-to-claim extractor that decomposes a given text $T$ into a set of claims $\left\{ c _ { i } \right\}$ , and 2) a claim-entailment checker to determine whether a given claim $c$ is entailed $( \in )$ in a reference text $R e f$ or not $( \not \in )$ .

# 3.3 RAGCHECKER Metrics

With the annotation and claim-level entailment functions specified, we next define the metrics. For a RAG user, we design metrics to compare the performance among RAG systems, including a single-value F1 score as an overall metric. For a RAG developer, on the other hand, we propose two sets of modular metrics for the retriever and the generator in a RAG system respectively, that aim to decompose the system and diagnose the source of errors. In the rest of this section, we will first introduce the overall metrics and then go over modular metrics for retriever and generator separately. The formulas for each metric are summarized in Appendix B.

![](images/891a8c047846821f5e41e4bbc78f64013ab2208d2c6c1f7e5b2cd0917f6b953c.jpg)  
Figure 1: Illustration of the proposed metrics in RAGCHECKER . The upper Venn diagram depicts the comparison between a model response and the ground truth answer, showing possible correct( ), incorrec $( \times )$ , and missing claims( ). The retrieved chunks are classified into two categories based on the type of claims they contain. Below, we define the overall, retriever, and generator metrics, illustrating how each component of the RAG system is evaluated for its performance.

# 3.3.1 Overall Metrics

To assess the overall response quality of a RAG system from a user’s perspective, we can compute the precision and recall at claim level for each model generated response against its paired ground-truth . Spe and y, we first extract claims from a model responserespectively. Then, we define correct claims in $m$ and a ground-e response as , $\{ c _ { i } ^ { ( m ) } \}$ $\{ c _ { i } ^ { ( g t ) } \}$ $\{ c _ { i } ^ { ( m ) } | c _ { i } ^ { ( m ) } \in g t \}$ ci and correct claims in the ground-truth answer as $\{ c _ { i } ^ { ( g t ) } | c _ { i } ^ { ( g t ) } \in m \}$ . Two metrics can be computed directly: precision is the proportion of correct claims in all response claims, and recall is the proportion of correct claims in all ground-truth answer claims. Further, the harmonic average of precision and recall gives the F1 score, as the overall performance metric.

# 3.3.2 Retriever Metrics

Ideally, a perfect retriever returns precisely all claims needed to generate the ground-truth answer. Completeness-wise, we can measure how many claims made in the ground-truth answer are covered by retrieved chunks. With retrieved chunks as the reference text, we compute claim recall as the proportion of $\{ c _ { i } ^ { ( g t ) } | c _ { i } ^ { ( g t ) } \in \{ \mathrm { c h u n k } _ { j } \} \}$ .

Differently, we define the retriever precision at chunk-level instead of claim-level. A retrieved chunk is called relevant chunk (r-chunk), if any ground-truth claim is entailed in it. In other words, chunkj is a relevant chunk if ∃i, s.t. c(gt)i $\exists i$ $c _ { i } ^ { ( g t ) } \in \mathrm { c h u n k } _ { j }$ . The rest retrieved chunks are called irrelevant chunk (irr-chunk). The retriever’s context precision is defined as $| \{ \mathrm { r } \mathrm { - c h u n k } _ { j } \} | / k$ , where $k$ is the number of all retrieved chunks.

Note that a chunk-level precision provides better interpretability than a claim-level one, because in practice RAG systems usually work with documents processed to be text chunks in a fixed size. That being said, it is likely that a chunk may contain relevant claims and irrelevant or misleading information at the same time. As a result, the best possible retriever can only achieve a claim-level precision score lower than $100 \%$ , and such an upper-bound varies depending on the actual text distribution in $D$ and chunking strategy.

# 3.3.3 Generator Metrics

Given $k$ retrieved chunks (possibly mixing relevant and irrelevant information), a perfect generator would identify and include all ground-truth-relevant claims and ignore any that are not. Because the generator’s results have dependency on retrieved chunks, we provide in total six metrics characterizing different aspects of its performance.

Given a model response $m$ and its claims $\{ c _ { i } ^ { ( m ) } \}$ , we first compute the proportion of $c _ { i } ^ { ( m ) }$ that are entailed in retrieved chunks. This metric is faithfulness, as it describes how faithful the generator is to the provided context, thus the higher the better.

Next, we examine three types of incorrect response claims, i.e. $\{ c _ { i } ^ { ( m ) } | c _ { i } ^ { ( m ) } \notin g t \}$

1. The first type includes incorrect claim that are entailed in a relevant chunk, then it indicates the generator is sensitive to noise coupled with useful information. The proportion of this type of claims to all $\{ c _ { i } ^ { ( m ) } \}$ is relevant noise sensitivity.   
2. The second type includes incorrect claim that are entailed in an irrelevant chunk, then it indicates the generator is also sensitive to noise even in an irrelevant context. The proportion of these incorrect claims is irrelevant noise sensitivity.   
3. Finally, the third type includes incorrect claims that are not entailed in any retrieved chunk, meaning all such claims are generated by the generator itself. Its proportion is hallucination.

Note that for simplicity we group the two noise sensitivities in Fig. 1, but later in Sec. 4.3 we can see that generators generally has different sensitivity to relevant and irrelevant noise.

Finally, we characterize how a generator uses information sources to produce correct claims. A correct claim not entailed by any chunk can only be based on generator’s self-knowledge, thus the proportion of these claims reflects how many correct claims are generated on its own. A lower self-knowledge score is better, when the generator is expected to fully depend on retrieved context only in a RAG system. On the other hand, we also check how much retrieved relevant information is

used by the generator. Retrieved relevant information is measured by the number of ground-truth answer claims entailed in retrieved chunks, while the evidence of being used by generator is reflected by entailment in model response. Therefore, the context utilization is computed as the ratio between $| \{ c _ { i } ^ { ( g t ) } | c _ { i } ^ { ( g t ) } \in \{ \mathrm { c h u n k } _ { j } \} $ and $c _ { i } ^ { ( g t ) } \in m \} |$ and $| \{ c _ { i } ^ { ( g t ) } | c _ { i } ^ { ( g t ) } \in \{ \mathrm { c h u n k } _ { j } \} |$ . Generally a higher context utilization is preferred.

# 4 Experiments

# 4.1 Experimental Setup

Baseline RAG Systems We apply RAGCHECKER to 8 customized RAG systems to demonstrate how these metrics reflect the properties and differences among them, and how they guide the refinement of these systems. The 8 RAG systems are combinations with 2 retrievers and 4 generators. For retrievers, we choose BM25 [33], a representative classic sparse retrieval framework, and E5-Mistral [48], the SOTA open-source dense retriever. Our four generators are GPT-4 [29], Mixtral-8x7B [14], Llama3-8B, and Llama3-70B [1], covering open-source and proprietary LLMs in various sizes. Further details are deferred to Appendix D. We employ Llama3-70B as both the claim extractor and checker models implemented by an open-sourced framework RefChecker5 [11]. As a validation of its performance on the RefChecker’s hallucination detection benchmark, this setup outperforms the best purely open-sourced combinations reported in RefChecker’s paper (see Appendix G).

Benchmark Datasets For comprehensive evaluations, we curate a benchmark containing 4,162 queries across 10 domains. This benchmark is repurposed from public datasets of open domain question answering, spanning domains of Wikipedia, AI science, novel, biomedical, finance, lifestyle, recreation, science, technology and writing. We convert the short answers to long-form answers in the datasets to align with the current LLM-based RAG systems. Please refer to Appendix A for the details of the benchmark curation process. The statistics of the benchmark are shown in Tab. 1.

# 4.2 Meta Evaluation

We first conduct the meta evaluation to verify the soundness of RAGCHECKER and compare with existing baseline RAG evaluation frameworks.

Baseline RAG Evaluation Frameworks We include a total of 10 metrics from Trulens [6], RA-GAS [5], ARES [35] and CRUD-RAG [25] in the meta evaluation, as they are capable to evaluate endto-end performance with long answers. Metrics selected for comparison along with their descriptions are summarized in Tab. 4 of Appendix C. To ensure a fair comparison, we use Llama3-70B-Instruct as the LLM backbone when applicable. Since models in the Llama3 family don’t provide an embedding model, baseline metrics requiring embedding capability still use their corresponding default LLM backbones. In addition to the 10 metrics detailed in the table, we also incorporate BLEU [31], ROUGE-L [20], and BERTScore [56] to assess the correlation between the generated responses and the ground truth answers.

Meta Evaluation Dataset All baseline metrics are designed with different aspects and functionalities to a certain degree, thus making an exact comparison over metric scores inapplicable. However, we argue that a good metric should reflect the relative human preference over different RAG systems. In this spirit, we construct the meta evaluation dataset with sampled instances from the generated responses of 8 baseline RAG systems introduced in Sec. 4.1 on our benchmark. Each meta evaluation instance is a pair of responses from two baseline RAG systems given the same query. By considering all combinations over 10 domains and 28 baseline pairs, we end up with 280 instances for pairwise human preference labeling. For each instance, annotators compare a pair of responses based on correctness, completeness, and overall assessment. For each aspect, annotators measure their preferences as one out of five relative choices, including significantly better, slightly better, tie, slightly worse and significantly worse. For quality control, each instance is annotated by two annotators, and their overall agreement and correlation are measured. To conclude, we build a meta evaluation dataset with 280 instances, each instance is labeled by two annotators with their preference in terms of correctness, completeness and overall assessment.

Table 1: Statistics of the RAG benchmark. This benchmark is repurposed from public datasets across 10 domains, containing 4,162 questions. For the domains of Finance, Lifestyle, Recreation, Technology, Science and Novel, the short answers are extended to long-form answers with GPT-4.   

<table><tr><td>Dataset</td><td>Domain</td><td># Query</td><td># Doc.</td><td>Source</td><td>Example Query</td></tr><tr><td>ClapNQ</td><td>Wikipedia</td><td>300</td><td>4,293</td><td>ClapNQ</td><td>Difference between russian blue and british blue cat</td></tr><tr><td>NovelQA</td><td>Novel</td><td>280</td><td>19</td><td>NovelQA</td><td>When do the Ewell kids go to school?</td></tr><tr><td>RobustQA Writing</td><td>Writing</td><td>500</td><td>199,994</td><td>LoTTE, RobustQA</td><td>What is the difference between online and internet?</td></tr><tr><td>RobustQA BioASQ</td><td>Biomedical</td><td>511</td><td>197,816</td><td>BioASQ</td><td>What hand deformities do patients with Apert syndrome present with?</td></tr><tr><td>RobustQA Finance</td><td>Finance</td><td>500</td><td>57,638</td><td>FiQA, RobustQA</td><td>Is it safer to send credit card number via unsecured website form or by e-mail? What safer options are there?</td></tr><tr><td>RobustQA Lifestyle</td><td>Lifestyle</td><td>500</td><td>119,461</td><td>LoTTE, RobustQA</td><td>Can i eat a day old peanut butter sandwich?</td></tr><tr><td>RobustQA Recreation</td><td>Recreation</td><td>500</td><td>166,975</td><td>LoTTE, RobustQA</td><td>Why are so many american (spy) movies set in europe?</td></tr><tr><td>RobustQA Science</td><td>Science</td><td>500</td><td>125,368</td><td>LoTTE, RobustQA</td><td>Where is the flaw in this proof that 1=2? (derivative of repeated addition)</td></tr><tr><td>RobustQA Technology</td><td>Technology</td><td>500</td><td>638,509</td><td>LoTTE, RobustQA</td><td>Why not use larger cipher keys?</td></tr><tr><td>KIWI</td><td>AI Science</td><td>71</td><td>429</td><td>KIWI</td><td>What are the prior approaches proposed to improve faithfulness of the reasoning steps generated by LLMs and what tasks are they applied on?</td></tr></table>

Meta Evaluation Process and Results Based on the meta evaluation dataset, we perform the following evaluation process. Since the human preference labels can be seen as the score difference of a response pair: $\bar { h _ { i } } \bar { = } H ( r _ { i } ^ { 2 } ) - H ( r _ { i } ^ { 1 } ) \in \{ - 2 , - 1 , 0 , 1 , 2 \}$ , with a baseline RAG evaluation model $E$ , we compute a normalized score difference as $e _ { i } = f ( \bar { E } ( r _ { i } ^ { 2 } ) - E ( r _ { i } ^ { 1 } ) ) \in [ - 2 , 2 ]$ , where $f$ is a linear normalization function. Our meta evaluation is the correlation between $h _ { i }$ and $e _ { i }$ overall 280 instances as reported in Tab. 2, together with the correlation between $h _ { i }$ and $h _ { i } ^ { \prime }$ from two annotators as the upper-bound. In addition, we further compute human agreement rate as the proportion of instances satisfying $\mathrm { a b s } ( h _ { i } - h _ { i } ^ { \prime } ) \leq 1$ , and the result is $9 0 . 9 5 \%$ .

Table 2: Correlation results with Human Evaluation of Correctness, Completeness, and Overall Assessment. We only show the metric with the best correlation for each baseline framework. Full results can be found in Tab. 5 of Appendix C.   

<table><tr><td rowspan="2">Baseline</td><td rowspan="2">Metric</td><td colspan="2">Correctness</td><td colspan="2">Completeness</td><td colspan="2">Overall Assessment</td></tr><tr><td>Pearson</td><td>Spearman</td><td>Pearson</td><td>Spearman</td><td>Pearson</td><td>Spearman</td></tr><tr><td>BLEU</td><td>BLEU-avg</td><td>38.89</td><td>35.32</td><td>32.13</td><td>21.85</td><td>35.14</td><td>29.42</td></tr><tr><td>ROUGE</td><td>ROUGE-L</td><td>31.75</td><td>31.72</td><td>47.88</td><td>45.67</td><td>43.10</td><td>43.21</td></tr><tr><td>BERTScore</td><td>BERTScore</td><td>30.34</td><td>27.05</td><td>37.93</td><td>40.05</td><td>33.51</td><td>35.57</td></tr><tr><td>TruLens</td><td>Answer Relevance</td><td>35.01</td><td>27.37</td><td>37.24</td><td>37.91</td><td>35.15</td><td>33.59</td></tr><tr><td>ARES</td><td>Answer Relevance</td><td>18.63</td><td>16.84</td><td>20.13</td><td>18.13</td><td>17.81</td><td>16.26</td></tr><tr><td>RAGAS</td><td>Answer Similarity</td><td>41.07</td><td>43.21</td><td>53.16</td><td>61.35</td><td>48.31</td><td>57.23</td></tr><tr><td>CRUD-RAG</td><td>Recall</td><td>30.93</td><td>27.13</td><td>45.11</td><td>43.76</td><td>41.25</td><td>39.71</td></tr><tr><td>RAGChecker</td><td>Same metric as human</td><td>49.66</td><td>46.95</td><td>60.67</td><td>58.11</td><td>61.93</td><td>60.90</td></tr><tr><td>Human</td><td>Annotator correlation</td><td>63.67</td><td>59.19</td><td>71.91</td><td>68.36</td><td>70.09</td><td>68.89</td></tr></table>

From the table, we can observe that RAGCHECKER has the strongest correlation with human preference in terms of three aspects. Among other baseline metrics, Answer Similarity of RAGAS, which is based on the stronger backbone model text-embedding-ada-002 [28], shows the best performance. We also provide a detailed comparison between RAGCHECKER and this strongest baseline in Fig. 4

of Appendix C. As an upper bound, the human correlations at the bottom show that there is still a clear gap between model predictions and human annotators.

# 4.3 Main Results

We present the averaged evaluation results for 8 RAG systems across 10 diverse domain datasets in Tab. 3. Additional results for all datasets are provided in Appendix E. The RAG system that exhibited the best performance in our experiments is E5-Mistral_GPT-4, owing to the strong retrieval capability of E5-Mistral coupled with the adept comprehension abilities of GPT-4. Next, we provide a list of insights induced from Tab. 3, along with their interpretation and possible directions for improvements.

Table 3: The averaged evaluation results for different RAG systems across 10 datasets. The overall performance of the RAG system is quantified using precision (Prec.), recall (Rec.), and F1 scores. The retriever component is evaluated based on claim recall (CR) and context precision (CP), while the generator component is diagnosed through context utilization (CU), relevant noise sensitivity (NS(I)), irrelevant noise sensitivity (NS(II)), hallucination (Hallu.), self-knowledge (SK), and faithfulness (Faith.). Additionally, the average number of response claims for each RAG system is provided.   

<table><tr><td rowspan="2">RAG systems</td><td colspan="3">Overall</td><td colspan="2">Retriever</td><td colspan="6">Generator</td><td rowspan="2">#Claim</td></tr><tr><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>CR↑</td><td>CP↑</td><td>CU↑</td><td>NS(I)↓</td><td>NS(II)↓</td><td>Hallu.↓</td><td>SK↓</td><td>Faith.↑</td></tr><tr><td>BM25_GPT-4</td><td>61.0</td><td>49.7</td><td>50.3</td><td>74.0</td><td>52.3</td><td>61.4</td><td>26.2</td><td>4.1</td><td>8.7</td><td>3.4</td><td>87.9</td><td>12</td></tr><tr><td>BM25_Llama3-8b</td><td>52.1</td><td>43.9</td><td>42.1</td><td>74.0</td><td>52.3</td><td>54.9</td><td>31.3</td><td>6.1</td><td>9.8</td><td>1.8</td><td>88.4</td><td>11</td></tr><tr><td>BM25_Llama3-70b</td><td>59.1</td><td>44.9</td><td>46.3</td><td>74.0</td><td>52.3</td><td>56.2</td><td>30.4</td><td>5.3</td><td>5.1</td><td>1.7</td><td>93.2</td><td>9</td></tr><tr><td>BM25_Mixtral-8x7b</td><td>52.5</td><td>44.3</td><td>42.9</td><td>74.0</td><td>52.3</td><td>54.9</td><td>34.3</td><td>5.8</td><td>6.2</td><td>1.8</td><td>92.0</td><td>9</td></tr><tr><td>E5-Mistral_GPT-4</td><td>62.0</td><td>53.0</td><td>52.7</td><td>83.5</td><td>61.8</td><td>60.4</td><td>28.9</td><td>3.5</td><td>5.7</td><td>1.4</td><td>92.9</td><td>12</td></tr><tr><td>E5-Mistral_Llama3-8b</td><td>53.8</td><td>48.3</td><td>45.0</td><td>83.5</td><td>61.8</td><td>55.0</td><td>33.5</td><td>5.5</td><td>6.6</td><td>0.8</td><td>92.7</td><td>11</td></tr><tr><td>E5-Mistral_Llama3-70b</td><td>60.6</td><td>50.4</td><td>50.2</td><td>83.5</td><td>61.8</td><td>57.6</td><td>31.7</td><td>4.3</td><td>3.3</td><td>0.8</td><td>95.9</td><td>10</td></tr><tr><td>E5-Mistral_Mixtral-8x7b</td><td>53.1</td><td>48.6</td><td>45.7</td><td>83.5</td><td>61.8</td><td>55.2</td><td>36.5</td><td>5.1</td><td>4.0</td><td>0.8</td><td>95.2</td><td>10</td></tr></table>

Retriever Matters Consistently. The quality of retrieval is crucial, as evidenced by the notable differences in overall Precision, Recall and F1 scores when comparing BM25 with E5-Mistral with the generator fixed. This improvement is agnostic to the specific choice of generator, suggesting a consistent benefit from employing a better retriever.

Generator Model Size Brings All-Round Improvement. Paired to the same retriever, Llama3-70B consistently achieves better overall performance than Llama3-8B. More concretely, this superiority is supported by a better performance over every generator metric, such as improved context utilization, reduced noise sensitivity, and less hallucination.

Stable and Performant Context Utilization is Key. Among all generator metrics, we observe that context utilization strongly correlates to the overall F1 score, while such correlation is relatively weaker for other generator metrics. Also, generators’ context utilization are relatively stable between the two retrievers, meaning their overall recall can be improved with a better retriever. These observations indicate that the capability to fully utilize retrieved context is key, which is intuitive because the generator in a RAG system is expected to leverage context to surpass its self-knowledge.

Informative Context Improves Faithfulness and Reduces Hallucination. As E5-Mistral achieves better claim recall, we observe generators paired to it achieves better faithfulness, indicating generators are all capable to identify and leverage information in context. Similarly, hallucination and selfknowledge are both reduced as well.

Retriever Recall Trades-off with Generator Noise Sensitivity. Claim recall for a retriever characterizes the coverage of all information necessary to produce ground-truth answer. In practice, however, because of the fixed-size chunking strategy, retrieved relevant chunks may inevitably also carry over noise as part of the context. As retriever claim recall increases, all generators become more sensitive to such noise, which can be explained as their faithfulness to certain context is not discriminative enough. This observation shows that generators’ capability to precisely leverage relevant context is still a challenge.

Relevant Noise Sensitivity is More Challenging. For every baseline RAG system, there’s an apparent gap between its relevant and irrelevant noise sensitivity. In correlation to the last paragraph, it further enhance the point that generators demonstrate a chunk-level faithfulness. It means a relevant

chunk is trusted as a whole, while an irrelevant one only has minimal impact. This subtle yet significant distinction supports and explains the importance of the quality and specification of the database for a RAG system.

Open-Source Models are Worse at Distinguishing Accurate Information from Noise. GPT-4 has both higher context utilization and lower noise sensitivity than the other three open source models. Open source models are faithful but tend to trust the context blindly especially when retrieval gets better. This observation raises the need for improving open source models’ reasoning ability.

# 4.4 Diagnosis on RAG Settings for Improvements

Guided by observations in Sec. 4.3, we modify settings commonly tuned in RAG systems that may lead to improvements, diagnose their working mechanisms with RAGCHECKER metrics, and provide suggestions for improvements on certain aspects. We experiment with different numbers of chunks, chunk sizes, chunk overlap ratios, and generation prompts. We highlight our main findings and suggestions as below, please refer to Appendix F for detailed analysis and results.

More Context Enhances Faithfulness. Increasing the number $( k )$ and size of chunks improves the recall of more useful information (claim recall $6 1 . 5 \substack {  } 7 7 . 6$ with $k$ $5  2 0$ , $7 0 . 3 \substack {  } 7 7 . 6$ with size $1 5 0  3 0 0$ ). Consequently, this provides more context for the generators to be more faithful to (faithfulness $8 8 . 1  9 2 . 2 $ with k $5  2 0$ , $9 1 . 2 \substack {  } 9 2 . 2$ with size $1 5 0 \to 3 0 0$ ), though at the same time they also become more sensitive to additional noise (noise sensitivity $3 4 . 0 {  } 3 5 . 4 $ with k $5  2 0$ , $3 4 . 5  3 5 . 4$ with size $1 5 0  3 0 0$ ). Improvements in the overall performance $F I 5 1 . 7 {  } 5 3 . 4 $ with $k$ $5 \to 2 0$ , $5 2 . 6  5 3 . 4$ with size $1 5 0 \to 3 0 0$ ) indicates benefits from more context.

Explicit Requirements in Prompts Affect Generation Preferences. When prompts introduces explicit requirements for better faithfulness, context utilization, and lower noise sensitivity, generators show improvements in faithfulness $( 9 2 . 2 \substack {  } 9 3 . 6 $ ), but struggle with the subtle tension between context utilization $( 5 9 . 2 \mathrm {  } 6 3 . 7 $ ) and noise sensitivity $( 3 5 . 4 \mathrm { \ - } ) 3 8 . 1 $ ).

Chunk Overlap Does Not Matter a Lot. The chunk overlap ratio is usually set to be non-zero to help generators better utilize surrounding information and identify chunks with coherent logic. However, it minimally affects generation performance, as retrieving more chunks sharing similar useful information (increased context precision $6 9 . 3 \substack {  } 7 1 . 1$ ) does not necessarily increase the total amount of retrieved useful information (comparable claim recall $7 7 . 8 \to 7 8 . 1$ ).

# Suggestions to RAG Builders

Improving the retriever is an effective way to enhance overall performance. While a better embedding model leads to improvements in both precision and recall, moderately increasing the number and size of chunks improves recall and thus $F l$ with minimal efforts in practice. Note that the effect saturates as the total amount of relevant information is fixed, so they need not be too large for a balanced cost-performance. On the other hand, given a limited number of context, larger chunk sizes with fewer chunks are preferred for better context precision. However, when targeting better context utilization or reduced noise sensitivity, opposite adjustments should be made to alleviate the influence of noise.

When tuning the generator, the trilemma of context utilization, noise sensitivity, and faithfulness makes it difficult to improve all aspects simultaneously. RAG builders should prioritize certain aspects in the prompt based on their targets, user preferences and the generator’s capability.

# 5 Conclusion

This paper presents RAGCHECKER , a novel evaluation framework designed for RAG systems. We validate our comprehensive suite of metrics, both overall and modular, through rigorous human assessments, demonstrating a strong correlation with evaluations conducted by human annotators. We have undertaken a detailed evaluation of eight distinct RAG systems using these metrics, yielding pivotal insights into the behaviors of the retriever and generator components and the trade-offs inherent in RAG system designs. These findings not only deepen our understanding of RAG system architectures but also furnish critical guidance for future advancements in RAG applications.

# References

[1] AI@Meta. Llama 3 model card. 2024.   
[2] A. Asai, S. Min, Z. Zhong, and D. Chen. Retrieval-based language models and applications. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 6: Tutorial Abstracts), pages 41–46, 2023.   
[3] H. Chase. LangChain. https://github.com/langchain-ai/langchain, Oct. 2022.   
[4] J. Chen, H. Lin, X. Han, and L. Sun. Benchmarking large language models in retrievalaugmented generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 17754–17762, 2024.   
[5] S. Es, J. James, L. Espinosa-Anke, and S. Schockaert. Ragas: Automated evaluation of retrieval augmented generation. arXiv preprint arXiv:2309.15217, 2023.   
[6] J. Ferrara, Ethan-Tonic, and O. M. Ozturk. The RAG Triad, January 2024. https://www. trulens.org/trulens_eval/core_concepts_rag_triad/.   
[7] Y. Gao, Y. Xiong, X. Gao, K. Jia, J. Pan, Y. Bi, Y. Dai, J. Sun, and H. Wang. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2023.   
[8] Y. Guo, Z. Li, X. Jin, Y. Liu, Y. Zeng, W. Liu, X. Li, P. Yang, L. Bai, J. Guo, et al. Retrieval-augmented code generation for universal information extraction. arXiv preprint arXiv:2311.02962, 2023.   
[9] R. Han, P. Qi, Y. Zhang, L. Liu, J. Burger, W. Wang, Z. Huang, B. Xiang, and D. Roth. Robustqa: Benchmarking the robustness of domain adaptation for open-domain question answering. In ACL Findings 2023, 2023.   
[10] X. He, Y. Tian, Y. Sun, N. V. Chawla, T. Laurent, Y. LeCun, X. Bresson, and B. Hooi. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. arXiv preprint arXiv:2402.07630, 2024.   
[11] X. Hu, D. Ru, L. Qiu, Q. Guo, T. Zhang, Y. Xu, Y. Luo, P. Liu, Y. Zhang, and Z. Zhang. Refchecker: Reference-based fine-grained hallucination checker and benchmark for large language models. 2024.   
[12] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. arXiv preprint arXiv:2311.05232, 2023.   
[13] Y. Huang and J. Huang. A survey on retrieval-augmented text generation for large language models. arXiv preprint arXiv:2404.10981, 2024.   
[14] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B. Hanna, F. Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.   
[15] V. Katranidis and G. Barany. Faaf: Facts as a function for the evaluation of rag systems. arXiv preprint arXiv:2403.03888, 2024.   
[16] M. Komeili, K. Shuster, and J. Weston. Internet-augmented dialogue generation. arXiv preprint arXiv:2107.07566, 2021.   
[17] T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, J. Devlin, K. Lee, K. Toutanova, L. Jones, M. Kelcey, M.-W. Chang, A. M. Dai, J. Uszkoreit, Q. Le, and S. Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019.   
[18] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.   
[19] C.-Y. Lin. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain, July 2004. Association for Computational Linguistics.   
[20] C.-Y. Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004.

[21] N. F. Liu, T. Zhang, and P. Liang. Evaluating verifiability in generative search engines. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 7001–7025, 2023.   
[22] Y. Liu, L. Huang, S. Li, S. Chen, H. Zhou, F. Meng, J. Zhou, and X. Sun. Recall: A benchmark for llms robustness against external counterfactual knowledge. arXiv preprint arXiv:2311.08147, 2023.   
[23] K. Lo, L. L. Wang, M. Neumann, R. Kinney, and D. Weld. S2ORC: The semantic scholar open research corpus. In D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, editors, Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4969–4983, Online, July 2020. Association for Computational Linguistics.   
[24] A. Louis, G. van Dijck, and G. Spanakis. Interpretable long-form legal question answering with retrieval-augmented large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 22266–22275, 2024.   
[25] Y. Lyu, Z. Li, S. Niu, F. Xiong, B. Tang, W. Wang, H. Wu, H. Liu, T. Xu, and E. Chen. Crud-rag: A comprehensive chinese benchmark for retrieval-augmented generation of large language models. arXiv preprint arXiv:2401.17043, 2024.   
[26] M. Maia, S. Handschuh, A. Freitas, B. Davis, R. McDermott, M. Zarrouk, and A. Balahur. Www’18 open challenge: financial opinion mining and question answering. In Companion proceedings of the the web conference 2018, pages 1941–1942, 2018.   
[27] Y. Mao, P. He, X. Liu, Y. Shen, J. Gao, J. Han, and W. Chen. Generation-augmented retrieval for open-domain question answering. arXiv preprint arXiv:2009.08553, 2020.   
[28] A. Neelakantan, T. Xu, R. Puri, A. Radford, J. M. Han, J. Tworek, Q. Yuan, N. Tezak, J. W. Kim, C. Hallacy, et al. Text and code embeddings by contrastive pre-training. arXiv preprint arXiv:2201.10005, 2022.   
[29] OpenAI, :, J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman, S. Anadkat, R. Avila, I. Babuschkin, S. Balaji, V. Balcom, P. Baltescu, H. Bao, M. Bavarian, J. Belgum, I. Bello, J. Berdine, G. Bernadett-Shapiro, C. Berner, L. Bogdonoff, O. Boiko, M. Boyd, A.-L. Brakman, G. Brockman, T. Brooks, M. Brundage, K. Button, T. Cai, R. Campbell, A. Cann, B. Carey, C. Carlson, R. Carmichael, B. Chan, C. Chang, F. Chantzis, D. Chen, S. Chen, R. Chen, J. Chen, M. Chen, B. Chess, C. Cho, C. Chu, H. W. Chung, D. Cummings, J. Currier, Y. Dai, C. Decareaux, T. Degry, N. Deutsch, D. Deville, A. Dhar, D. Dohan, S. Dowling, S. Dunning, A. Ecoffet, A. Eleti, T. Eloundou, D. Farhi, L. Fedus, N. Felix, S. P. Fishman, J. Forte, I. Fulford, L. Gao, E. Georges, C. Gibson, V. Goel, T. Gogineni, G. Goh, R. Gontijo-Lopes, J. Gordon, M. Grafstein, S. Gray, R. Greene, J. Gross, S. S. Gu, Y. Guo, C. Hallacy, J. Han, J. Harris, Y. He, M. Heaton, J. Heidecke, C. Hesse, A. Hickey, W. Hickey, P. Hoeschele, B. Houghton, K. Hsu, S. Hu, X. Hu, J. Huizinga, S. Jain, S. Jain, J. Jang, A. Jiang, R. Jiang, H. Jin, D. Jin, S. Jomoto, B. Jonn, H. Jun, T. Kaftan, Łukasz Kaiser, A. Kamali, I. Kanitscheider, N. S. Keskar, T. Khan, L. Kilpatrick, J. W. Kim, C. Kim, Y. Kim, H. Kirchner, J. Kiros, M. Knight, D. Kokotajlo, Łukasz Kondraciuk, A. Kondrich, A. Konstantinidis, K. Kosic, G. Krueger, V. Kuo, M. Lampe, I. Lan, T. Lee, J. Leike, J. Leung, D. Levy, C. M. Li, R. Lim, M. Lin, S. Lin, M. Litwin, T. Lopez, R. Lowe, P. Lue, A. Makanju, K. Malfacini, S. Manning, T. Markov, Y. Markovski, B. Martin, K. Mayer, A. Mayne, B. Mc-Grew, S. M. McKinney, C. McLeavey, P. McMillan, J. McNeil, D. Medina, A. Mehta, J. Menick, L. Metz, A. Mishchenko, P. Mishkin, V. Monaco, E. Morikawa, D. Mossing, T. Mu, M. Murati, O. Murk, D. Mély, A. Nair, R. Nakano, R. Nayak, A. Neelakantan, R. Ngo, H. Noh, L. Ouyang, C. O’Keefe, J. Pachocki, A. Paino, J. Palermo, A. Pantuliano, G. Parascandolo, J. Parish, E. Parparita, A. Passos, M. Pavlov, A. Peng, A. Perelman, F. de Avila Belbute Peres, M. Petrov, H. P. de Oliveira Pinto, Michael, Pokorny, M. Pokrass, V. Pong, T. Powell, A. Power, B. Power, E. Proehl, R. Puri, A. Radford, J. Rae, A. Ramesh, C. Raymond, F. Real, K. Rimbach, C. Ross, B. Rotsted, H. Roussez, N. Ryder, M. Saltarelli, T. Sanders, S. Santurkar, G. Sastry, H. Schmidt, D. Schnurr, J. Schulman, D. Selsam, K. Sheppard, T. Sherbakov, J. Shieh, S. Shoker, P. Shyam, S. Sidor, E. Sigler, M. Simens, J. Sitkin, K. Slama, I. Sohl, B. Sokolowsky, Y. Song, N. Staudacher, F. P. Such, N. Summers, I. Sutskever, J. Tang, N. Tezak, M. Thompson, P. Tillet, A. Tootoonchian, E. Tseng, P. Tuggle, N. Turley, J. Tworek, J. F. C. Uribe, A. Vallone, A. Vijayvergiya, C. Voss, C. Wainwright, J. J. Wang, A. Wang, B. Wang, J. Ward, J. Wei, C. Weinmann, A. Welihinda, P. Welinder, J. Weng, L. Weng, M. Wiethoff, D. Willner, C. Winter,

S. Wolrich, H. Wong, L. Workman, S. Wu, J. Wu, M. Wu, K. Xiao, T. Xu, S. Yoo, K. Yu, Q. Yuan, W. Zaremba, R. Zellers, C. Zhang, M. Zhang, S. Zhao, T. Zheng, J. Zhuang, W. Zhuk, and B. Zoph. Gpt-4 technical report, 2023.   
[30] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu. Bleu: a method for automatic evaluation of machine translation. In P. Isabelle, E. Charniak, and D. Lin, editors, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics.   
[31] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318, 2002.   
[32] M. R. Parvez, W. U. Ahmad, S. Chakraborty, B. Ray, and K.-W. Chang. Retrieval augmented code generation and summarization. arXiv preprint arXiv:2108.11601, 2021.   
[33] S. Robertson, H. Zaragoza, et al. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389, 2009.   
[34] S. Rosenthal, A. Sil, R. Florian, and S. Roukos. Clapnq: Cohesive long-form answers from passages in natural questions for rag systems, 2024.   
[35] J. Saad-Falcon, O. Khattab, C. Potts, and M. Zaharia. Ares: An automated evaluation framework for retrieval-augmented generation systems, 2023.   
[36] K. Santhanam, O. Khattab, J. Saad-Falcon, C. Potts, and M. Zaharia. ColBERTv2: Effective and efficient retrieval via lightweight late interaction. In M. Carpuat, M.-C. de Marneffe, and I. V. Meza Ruiz, editors, Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3715–3734, Seattle, United States, July 2022. Association for Computational Linguistics.   
[37] K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston. Retrieval augmentation reduces hallucination in conversation. arXiv preprint arXiv:2104.07567, 2021.   
[38] H. Tan, Q. Luo, L. Jiang, Z. Zhan, J. Li, H. Zhang, and Y. Zhang. Prompt-based code completion via multi-retrieval augmented generation. arXiv preprint arXiv:2405.07530, 2024.   
[39] Y. Tang and Y. Yang. Multihop-rag: Benchmarking retrieval-augmented generation for multihop queries. arXiv preprint arXiv:2401.15391, 2024.   
[40] N. Thakur, L. Bonifacio, X. Zhang, O. Ogundepo, E. Kamalloo, D. Alfonso-Hermelo, X. Li, Q. Liu, B. Chen, M. Rezagholizadeh, et al. Nomiracl: Knowing when you don’t know for robust multilingual retrieval-augmented generation. arXiv preprint arXiv:2312.11361, 2023.   
[41] D. Thulke, N. Daheim, C. Dugast, and H. Ney. Efficient retrieval augmented generation from unstructured knowledge for task-oriented dialog. arXiv preprint arXiv:2102.04643, 2021.   
[42] S. Tonmoy, S. Zaman, V. Jain, A. Rani, V. Rawte, A. Chadha, and A. Das. A comprehensive survey of hallucination mitigation techniques in large language models. arXiv preprint arXiv:2401.01313, 2024.   
[43] G. Tsatsaronis, G. Balikas, P. Malakasiotis, I. Partalas, M. Zschunke, M. R. Alvers, D. Weissenborn, A. Krithara, S. Petridis, D. Polychronopoulos, et al. An overview of the bioasq large-scale biomedical semantic indexing and question answering competition. BMC bioinformatics, 16:1–28, 2015.   
[44] E. M. Voorhees et al. The trec-8 question answering track report. In Trec, volume 99, pages 77–82, 1999.   
[45] C. Wang, S. Cheng, Q. Guo, Y. Yue, B. Ding, Z. Xu, Y. Wang, X. Hu, Z. Zhang, and Y. Zhang. Evaluating open-qa evaluation. Advances in Neural Information Processing Systems, 36, 2024.   
[46] C. Wang, X. Liu, Y. Yue, X. Tang, T. Zhang, C. Jiayang, Y. Yao, W. Gao, X. Hu, Z. Qi, et al. Survey on factuality in large language models: Knowledge, retrieval and domain-specificity. arXiv preprint arXiv:2310.07521, 2023.   
[47] C. Wang, R. Ning, B. Pan, T. Wu, Q. Guo, C. Deng, G. Bao, Q. Wang, and Y. Zhang. Novelqa: A benchmark for long-range novel question answering, 2024.   
[48] L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, and F. Wei. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368, 2023.

[49] K. Wu, E. Wu, and J. Zou. How faithful are rag models? quantifying the tug-of-war between rag and llms’ internal prior. arXiv preprint arXiv:2404.10198, 2024.   
[50] G. Xiong, Q. Jin, Z. Lu, and A. Zhang. Benchmarking retrieval-augmented generation for medicine. arXiv preprint arXiv:2402.13178, 2024.   
[51] F. Xu, K. Lo, L. Soldaini, B. Kuehl, E. Choi, and D. Wadden. Kiwi: A dataset of knowledgeintensive writing instructions for answering research questions, 2024.   
[52] Z. Xu, Y. Li, R. Ding, X. Wang, B. Chen, Y. Jiang, X. Deng, J. Ma, H.-T. Zheng, W. Lu, et al. Let llms take on the latest challenges! a chinese dynamic question answering benchmark. arXiv preprint arXiv:2402.19248, 2024.   
[53] H. Yu, A. Gan, K. Zhang, S. Tong, Q. Liu, and Z. Liu. Evaluation of retrieval-augmented generation: A survey. arXiv preprint arXiv:2405.07437, 2024.   
[54] C. Zakka, R. Shad, A. Chaurasia, A. R. Dalal, J. L. Kim, M. Moor, R. Fong, C. Phillips, K. Alexander, E. Ashley, et al. Almanac—retrieval-augmented language models for clinical medicine. NEJM AI, 1(2):AIoa2300068, 2024.   
[55] B. Zhang, H. Yang, T. Zhou, M. Ali Babar, and X.-Y. Liu. Enhancing financial sentiment analysis via retrieval augmented large language models. In Proceedings of the Fourth ACM International Conference on AI in Finance, pages 349–356, 2023.   
[56] T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675, 2019.   
[57] S. Zhou, U. Alon, F. F. Xu, Z. Wang, Z. Jiang, and G. Neubig. Docprompting: Generating code by retrieving the docs. arXiv preprint arXiv:2207.05987, 2022.

# A Details for Benchmark Curation

In this section, we introduce the benchmark datasets and the curation process for RAG evaluation. This benchmark datasets are derived from existing open-domain question answering (ODQA) datasets, including RobustQA [9], KIWI [51], ClapNQ [34], and NovelQA [47]. However, most of the ground truth answers in existing ODQA datasets are short answers, while the answers provided by modern LLM-based RAG systems tend to be long-form answers. Therefore, we repurpose the ODQA datasets by eliminating overly simple questions and converting the short answers into long-form answers to match the capabilities of current RAG systems. The statistics of the benchmark are summarized in Tab. 1. In the rest of this section, we describe the datasets we use and the curation process for each domain.

# A.1 Data Sources

RobustQA We choose 7 domains from RobustQA’s collection of datasets: Biomedical, Finance, Lifestyle, Recreation, Technology, Science, and Writing. For the Biomedical domain, following RobustQA, we employ the BioASQ [43] dataset, which contains human expert-written questionanswer pairs and ground truth documents based on abstracts of articles from PubMed. We use the test sets for Task b from 2014 to 2023 and the corpus of v.2022 to construct the benchmark. We keep QA pairs whose answers are relatively long (more than 50 words), obtaining 511 QA pairs for the biomedical domain. The other 6 domains are sourced from FiQA [26] and LoTTE [36], each of their question is annotated with a list of short answers that are spans of ground truth passages. We convert the short answers to long-form answers using GPT-4 and only keep the generated answers with no hallucinations, as checked by RefChecker. Finally, we sample 500 examples for each domain.

ClapNQ Derived from NaturalQuestions (NQ) [17], an ODQA dataset based on Wikipedia, ClapNQ has long-form answers annotated for a subset of NQ for evaluating RAG. We employ the dev set of ClapNQ in our benchmark and take the annotated long-form answers as the ground truth.

KIWI is constructed by asking LLMs research questions about a set of NLP papers and guiding the LLMs to reach satisfactory long-form answers. The authors validated the quality of the generated answers by rating them as “good”, “neutral”, or “bad”. We take the answers labeled “good” as the ground truth answers and query the full text of the papers from S2ORC [23] as the corpus. As a result, we obtain 71 QA pairs and 429 papers as the corpus.

NovelQA is a benchmark for question answering over long novels containing over 100K tokens on average. Originally designed for benchmarking long-context LLMs, we repurpose it for evaluating RAG. In contrast with the other domains, each question in NovelQA is associated with a single novel, so when we use this dataset for RAG, we constrain the retrieval to within the corresponding novel. We select 19 copyright-free novels and convert the corresponding short answers to long-form answers following the same process for RobustQA.

# A.2 Long-form Answer Generation

We employ GPT-4 (gpt-4-turbo-2024-04-09) to convert the human annotated short answers to long-form answers in the dataset of RobustQA and NovelQA. For RobustQA, the short answers are spans of the annotated ground truth passages, we take all the annotated short answers and the corresponding passages in the prompt and ask GPT-4 to convert them to one single long-form answer. For NovelQA, we take the human written evidences as the ground truth passage content and the human written short answers for the long-form answer generation. The prompt is shown in Fig. 2.

For quality control, we ask GPT-4 to generate the passage IDs associated with the long-form answer. We use RefChecker to check whether all the claims of a long-form answer are entailed by these passages, and we only keep the long-form answers that meet this criteria. The RefChecker we used here are described in Appendix G.

```txt
Here is a question and human annotated short answers, the short answers are extracted from a passage, you should help me to convert the short answers to a long-form answer based on the provided passage. There could be more than one annotation, so you should identify the best answers and merge them. The long-form answer should only depend on the provide information, you should not hallucinate anything.   
Question {question}   
Annotations   
Annotation 1 [Passage ID]:{passage_id} [Passage Content]:{passage_content} [Short Answers]: {answers}   
Annotation 2 [Passage ID]:{passage_id} [Passage Content]:{passage_content} [Short Answers]: {answers}   
Your should output the converted long-form answer and the passage IDs you used. Always follow this format for your response: [Long-form answer]: <the content of the long-form answer> [Passage IDs]:ID1,ID2,... 
```

Figure 2: The prompt used for converting short answers to long-form answers for the domains of Novel, Finance, Lifestyle, Recreation, Technology, Science, and Writing.

# A.3 Corpus Downsampling for Science and Biomedical Domains

In addition to long-form answer generation, we also perform downsampling for the corpora of Science and Biomedical domains as they are much larger than the others, with over 1 million documents each. Building indexes for a dense retriever is very costly for large corpora, so we downsample these domains to lower the evaluation cost for the community. For the biomedical domain, we first use BM25 retriever to obtain top 400 documents for each question. The subsampled corpus is formed by combining all documents from the retriever with annotated relevant documents from the datasets. Based on our initial study, we observe that the BM25 retriever yeild competitive performance against the dense retriever, so we decide to only use the BM25 retriever for downsampling purpose to save compuation cost. For the science domain, we leverage both the BM25 retriever and e5-mistral-7b-instruct based dense retriever to obtain document candidates. Specifically, we retrieve the top 200 documents from both retrievers (400 documents in total before deduplication). Similarly, the combination of all documents from the retrievers and annotated relevant documents from datasets forms the downsampled corpus.

# A.4 License of The Datasets

The annotations from RobustQA, ClapNQ and NovelQA are under Apache-2.0 License. The corpora of Finance and annotations of KIWI are under CC-BY-SA-4.0. BioASQ is under CC BY 2.5 license. The license for the corpora of LoTTE are not specified.

# B The complete formula for all metrics

Denote the model response as $m$ , the ground truth answer as $g t$ , and the retrieved chunks as $\{ \mathrm { c h u n k } _ { j } \}$ . Leveraging RefChecker, we decompose the text into a set of claims $\left\{ c _ { i } \right\}$ and assess whether a specific claim $c _ { i }$ can entail $( \in )$ or not entail $( \not \in )$ a given reference text $R e f$ , where $R e f$ may represent m, gt, or $\{ \mathrm { c h u n k } _ { j } \}$ . We assign an entailment label to each ground-truth claim relative to a chunk, and subsequently classify these chunks into relevant chunks $\left\{ \operatorname { r - c h u n k } _ { j } \right\}$ and irrelevant chunks

$\{ \mathrm { i r r - c h u n k } _ { j } \}$ . Specifically, chunkj is considered relevant if it contains at least one claim $c _ { i } ^ { ( g t ) }$ such that $c _ { i } ^ { ( g t ) } \in \mathrm { c h u n k } _ { j }$ .

In accordance with the definitions provided in Section 3.3, we compute each metric using the following formulations:

# B.1 Overall Metrics

$$
\text {P r e c i s i o n} = \frac {\left| \left\{c _ {i} ^ {(m)} \mid c _ {i} ^ {(m)} \in g t \right\} \right|}{\left| \left\{c _ {i} ^ {(m)} \right\} \right|}
$$

$$
\text {R e c a l l} = \frac {\left| \left\{c _ {i} ^ {(g t)} \mid c _ {i} ^ {(g t)} \in m \right\} \right|}{\left| \left\{c _ {i} ^ {(g t)} \right\} \right|}
$$

# B.2 Retriever Metrics

$$
\text {C l a i m R e c a l l} = \frac {\left| \left\{c _ {i} ^ {(g t)} \mid c _ {i} ^ {(g t)} \in \{\text {c h u n k} _ {j} \} \right\} \right|}{\left| \left\{c _ {i} ^ {(g t)} \right\} \right|}
$$

$$
\text {C o n t e x t P r e c i s i o n} = \frac {\left| \left\{\mathrm {r} - \operatorname {c h u n k} _ {j} \right\} \right|}{k}
$$

# B.3 Generator Metrics

$$
\text {F a i t h f u l n e s s} = \frac {\left| \left\{c _ {i} ^ {(m)} \mid c _ {i} ^ {(m)} \in \left\{\operatorname {c h u n k} _ {j} \right\} \right\} \right|}{\left| \left\{c _ {i} ^ {(m)} \right\} \right|}
$$

$$
\text {R e l e v a n t N o i s e S e n s i t i v i t y} = \frac {\left| \left\{c _ {i} ^ {(m)} \mid c _ {i} ^ {(m)} \notin g t \text {a n d} c _ {i} ^ {(m)} \in \{\mathrm {r - c h u n k} _ {j} \} \right\} \right|}{\left| \left\{c _ {i} ^ {(m)} \right\} \right|}
$$

$$
\text {I r r e l e v a n N o i s e S e n s i t i v i t y} = \frac {\left| \right.\left\{ \right.c _ {i} ^ {(m)} \left. \right| c _ {i} ^ {(m)} \notin g t \text {a n d} c _ {i} ^ {(m)} \in \{\text {i r r - c h u n k} _ {j} \} \left. \right\}\left. \right|}{\left|\left\{c _ {i} ^ {(m)} \right\}\right|}
$$

$$
\text {H a l l u c i n a t i o n} = \frac {\left| \right.\left\{ \right.c _ {i} ^ {(m)} \left. \right| c _ {i} ^ {(m)} \notin g t \text {a n d} c _ {i} ^ {(m)} \notin \left\{\text {c h u n k} _ {j} \right\}\left. \right\}\left. \right|}{\left|\left\{c _ {i} ^ {(m)} \right\}\right|}
$$

$$
\text {S e l f - k n o w l e d g e} = \frac {\left| \left\{c _ {i} ^ {(m)} \mid c _ {i} ^ {(m)} \in g t \text {a n d} c _ {i} ^ {(m)} \notin \left\{\operatorname {c h u n k} _ {j} \right\} \right\} \right|}{\left| \left\{c _ {i} ^ {(m)} \right\} \right|}
$$

$$
\text {C o n t e x T u l i z a t i o n} = \frac {\left| \left\{c _ {i} ^ {(g t)} \mid c _ {i} ^ {(g t)} \in \{\text {c h u n k} _ {j} \} \text {a n d} c _ {i} ^ {(g t)} \in m \} \right| \right.}{\left| \left\{c _ {i} ^ {(g t)} \mid c _ {i} ^ {(g t)} \in \{\text {c h u n k} _ {j} \} \right\} \right|}
$$

# C Details of Meta Evaluation

In the meta evaluation, we ask 10 annotators compare two responses from the RAG system for each instance in the meta evaluation dataset. Seven of the annotators are in-house annotators, and three of them are graduate students. We pay the students 15 USD per hour and totally cost 255 dollars.

Annotators are required to choose their preference from five options: significantly better, slightly better, tie, slightly worse, or significantly worse. The annotation is based on three metrics: correctness, completeness, and overall assessment. The annotation interface with instructions are shown in Fig. 3

To make sure the human evaluation to be agnostic to specific evaluation metrics, we provide the annotators with a detailed annotation guideline which contains detailed instruction and 5 examples.

![](images/f63cb0136eb100fc0dbdc353910a25b39ea7cbfe50f277a50bdce0fcb949ff0c.jpg)  
Figure 3: The human annotation interface and instructions of the meta evaluation dataset.

In the UI of the annotation tool, each response is shown with critiques generated by GPT-4 and we ask the annotators to refer to the content of the response and the critiques for labeling. The critiques are generated by prompting GPT-4 to compare the response with ground truth answer to ease the annotation job. In addition, each of the example the guideline are shown with a human-written explanation for the labeling.

The 10 metrics included in the meta evaluation are selected from Trulens [6], RAGAS [5], ARES [35] and CRUD-RAG [25] as explained in Sec. 4.2. Their descriptions are summarized in Tab. 4. As a supplement of Tab. 2, the full correlation results of meta evaluation is shown in Tab. 5. For a detailed comparison between RAGCHECKER and the strongest baseline metric, RAGAS Answer Similarity, we plot the prediction score distribution of two metrics in Fig. 4. From the prediction score distribution and the mean line (dashed line) of the plot, we can observe a stronger correlation of RAGCHECKER than RAGAS Answer Similarity.

# D Details of the Experiment Setup

Models in Baseline RAG Systems We use the version of e5-mistral-7b-instruct for the E5-Mistral retriever. For the generators, we use gpt-4-turbo-2024-04-09 version for GPT-4, Llama3-8B-Instruct for Llama3-8B and Llama3-70B-Instruct for Llama3-70B, and Mixtral-8x7B-Instruct-v0.1 for Mixtral-8x7B.

We adopt OpenSearch6 as the tool to implement the inverted index for BM25 and the approximate KNN search for dense retrieval. We use a g5.48xlarge instance with 8 NVIDIA A10G GPUs on AWS for inference of open-source models. We split documents in the corpus to chunks of 300 tokens with an overlap ratio of 0.2 by default. We use the tokenizer of E5-Mistral for both retrievers to control the chunking. For each query, top-20 chunks ranked by retrievers are used as context for LLM generation. The default prompt for all generators is shown in Fig. 5. We set the generation temperature to 0.0 (deterministic) and the maximum generation length to 2,048 tokens when calling proprietary LLMs.

# E Detailed Experiment Results

The detailed evaluation results for all our benchmark datasets can be found in Tab. 6 to Tab. 15.

Table 4: Summary of the metrics included in the meta evaluation.   

<table><tr><td>Baseline</td><td>Metric</td><td>Description</td></tr><tr><td rowspan="2">TruLens</td><td>Groundedness</td><td>Assesses the overlap between each statement in the response and the provided context using an LLM.</td></tr><tr><td>Answer Relevance</td><td>Prompts an LLM to give a relevance score between the response and question.</td></tr><tr><td rowspan="4">RAGAS</td><td>Faithfulness</td><td>Measures the proportion of claims in the response that can be inferred from the context.</td></tr><tr><td>Answer Relevance</td><td>Computes the mean cosine similarity between the original question and a series of LLM-generated questions derived from the response and context.</td></tr><tr><td>Answer Similarity</td><td>Measures the semantic similarity between the response and the ground truth answer based on text-embedding-ada-002 [28].</td></tr><tr><td>Answer Correctness</td><td>Quantifies both the semantic similarity and the factual overlap between the response and the ground truth answer.</td></tr><tr><td rowspan="2">ARES</td><td>Answer Faithfulness</td><td>Prompts an LLM to determine whether the response is faithful to the context.</td></tr><tr><td>Answer Relevance</td><td>Prompts an LLM to measure whether the response addresses all aspects of the question and provides only correct information from the context.</td></tr><tr><td rowspan="2">CRUD-RAG</td><td>Recall</td><td>Computes the ratio of all questions generated from ground truth answers that can be answered by response.</td></tr><tr><td>Precision</td><td>Evaluates if the generated text is accurate and consistent with the ground truth answer.</td></tr></table>

Table 5: Full Correlation results with Human Evaluation of Correctness, Completeness, and Overall Assessment   

<table><tr><td rowspan="2">Baseline</td><td rowspan="2">Metric</td><td colspan="2">Correctness</td><td colspan="2">Completeness</td><td colspan="2">Overall Assessment</td></tr><tr><td>Pearson</td><td>Spearman</td><td>Pearson</td><td>Spearman</td><td>Pearson</td><td>Spearman</td></tr><tr><td>BLEU</td><td>BLEU-avg</td><td>38.89</td><td>35.32</td><td>32.13</td><td>21.85</td><td>35.14</td><td>29.42</td></tr><tr><td>ROUGE</td><td>ROUGE-L</td><td>31.75</td><td>31.72</td><td>47.88</td><td>45.67</td><td>43.10</td><td>43.21</td></tr><tr><td>BERTScore</td><td>BERTScore</td><td>30.34</td><td>27.05</td><td>37.93</td><td>40.05</td><td>33.51</td><td>35.57</td></tr><tr><td>TruLens</td><td>Groundedness</td><td>21.11</td><td>18.21</td><td>14.01</td><td>6.02</td><td>19.45</td><td>14.42</td></tr><tr><td>TruLens</td><td>Answer Relevance</td><td>35.01</td><td>27.37</td><td>37.24</td><td>37.91</td><td>35.15</td><td>33.59</td></tr><tr><td>ARES</td><td>Answer Relevance</td><td>18.63</td><td>16.84</td><td>20.13</td><td>18.13</td><td>17.81</td><td>16.26</td></tr><tr><td>ARES</td><td>Answer Faithfulness</td><td>9.46</td><td>7.60</td><td>10.25</td><td>8.99</td><td>8.80</td><td>7.58</td></tr><tr><td>RAGAS</td><td>Faithfulness</td><td>8.22</td><td>7.53</td><td>4.90</td><td>1.19</td><td>7.83</td><td>5.55</td></tr><tr><td>RAGAS</td><td>Answer Correctness</td><td>39.11</td><td>36.30</td><td>36.42</td><td>36.04</td><td>38.01</td><td>37.14</td></tr><tr><td>RAGAS</td><td>Answer Similarity</td><td>41.07</td><td>43.21</td><td>53.16</td><td>61.35</td><td>48.31</td><td>57.23</td></tr><tr><td>RAGAS</td><td>Answer Relevance</td><td>11.59</td><td>8.19</td><td>9.39</td><td>13.57</td><td>10.27</td><td>11.83</td></tr><tr><td>CRUD-RAG</td><td>Precision</td><td>20.73</td><td>15.67</td><td>25.58</td><td>20.33</td><td>25.59</td><td>19.63</td></tr><tr><td>CRUD-RAG</td><td>Recall</td><td>30.93</td><td>27.13</td><td>45.11</td><td>43.76</td><td>41.25</td><td>39.71</td></tr><tr><td>RAGChecker</td><td>Same metric as human</td><td>49.66</td><td>46.95</td><td>60.67</td><td>58.11</td><td>61.93</td><td>60.90</td></tr><tr><td>Human</td><td>Annotator sanity check</td><td>63.67</td><td>59.19</td><td>71.91</td><td>68.36</td><td>70.09</td><td>68.89</td></tr></table>

![](images/a2f78741adaddd64f5c19e66118d5fc170df7655d0acf398371f84a3eb6a3329.jpg)

![](images/dfb25b029ee04c9776c9380ca49bba8272fe80e58fd831a48735587ce8d3fcc7.jpg)

![](images/d2162e1c89d2cbe80065109fd63fea5902b5dda3a07d51954bd41c2a1fb881a5.jpg)  
Figure 4: Comparison of prediction score distribution between RAGCHECKER and RAGAS Answer Similarity. Each point in the plot represents an instance in the meta evaluation dataset, where the x-axis is the human preference label under corresponding aspect and y-axis is the prediction score of RAGCHECKER and RAGAS Answer Similarity. The distribution of prediction score is represented by the colored area and the dashed line is the mean line.

![](images/81d8e6d879234d70788659277bec0bc63deeedc4d29f61a4c025be5cda4ea31d.jpg)  
Figure 5: The default prompt used for response generation in the main experiments for the 8 RAG baseline systems.

# F Diagnosis on RAG for Improvements

We modify hyper-parameters commonly tuned in RAG systems to observe performance variance under the metrics defined by RAGCHECKER . We focus on how RAGCHECKER explains this variance and provides tuning suggestions for improvements on certain aspects. In this section, we evaluate three RAG baselines (BM25_GPT-4, E5-Mistral_GPT-4, and E5-Mistral_Llama3-70B) across three domains with increasing difficulty: Writing, Finance, and KIWI. We use our default settings (Appendix D) in main experiments as controls. We experiment with different numbers of chunks selected as context $k \in \{ 5 , 1 0 , 2 0 \}$ , different chunk size $\{ 1 5 0 , 3 0 0 , 6 0 0 \} ^ { 7 }$ , different chunk overlap ratio $\{ 0 . 0 , 0 . 2 , 0 . 4 \}$ , and different generation prompts.

More Context Enhances Faithfulness Top-k selection and chunk size both balance the amount of noise and useful information presented to the generator, but in different manners. Corresponding results are demonstrated in Fig. 6 and Fig. 7. Increasing $k$ adds more context that could be less relevant, while increasing chunk size provides more surrounding context of relevant facts. Thus context precision decreases with larger $k$ but increases with larger chunk sizes. Despite this, they both lead to better claim recall in Retrieval.

Generators tend to be more faithful when provided with more context, though this trend is less pronounced for Llama3, which already exhibits high faithfulness. Context utilization generally worsens with more context due to increasing noise, leading to higher relevant noise sensitivity.

Overall, the end-to-end RAG performance is slightly better with more context, primarily due to improved recall. We recommend moderately increasing the two parameters for more faithful generation, noting that saturation occurs at high values as the amount of useful information is limited. Given a limited context length, a larger chunk size with a smaller k is preferred, especially for easier datasets (Finance, Writing). This is evident when comparing a chunk size of 150 with $k { = } 2 0$ against a chunk size of 300 with $k { = } 1 0$ .

Explicit Requirements in Prompts Affect Generation Preferences To validate the effect of the generation prompt, we added more detailed requirements to guide the generation for better faithfulness, context utilization, and lower noise sensitivity. The optimized prompt is shown in Fig. 9.

As shown in Fig. 8, we observed a general improvement in context utilization. However, as a counterpart to context utilization, noise sensitivity generally worsened. It demonstrates the difficulty of meeting all prompt requirements when there are subtle tension between them.

For the two generators, GPT-4 generally showes improvements in metrics related to faithfulness (hallucination, self-knowledge, faithfulness), whereas Llama3 does not exhibit the same behavior. This aligns with our previous observation (Sec. 4.3) that Llama3 already performs well on faithfulness,

![](images/82d08621f7753d470ec0c911636a5a7b4d1b22938f625d1a7c9aa57a9c59d6a3.jpg)

![](images/7853049460d21b554445477bbd3f0cc8c914daec26b0be6eff23db2bf3c0b95f.jpg)

![](images/bca06bebb34516bf9644329d0a46e55917d38cd46b587641c6e6a37bf6012986.jpg)

![](images/25e99fac48de465af63d16050d1be5b4a128574d71c21e6a6428dcee4a5e5ec7.jpg)

![](images/9bb73389a1468d42bf27eacc26877388512e3583971c1769f064efc2f63363e1.jpg)

![](images/d9dfc60c98a30a4f9ef5ab8985a3ba93414e1f09cf1ba22a2df746fcada9ded5.jpg)

![](images/c847307e904fa9c6073aeb3706e1df63fa7a0d7afdfb9a9ca0683c473515b6b2.jpg)

![](images/96f74608e852d71a6bf6ccd190b39b30c2cac421c7b7e3a4ad04e6bf518c17eb.jpg)

![](images/d3e3787bba8f0ad6de693b977347f3647ebb827fb1153ea3f3e19197c292662e.jpg)  
Figure 6: Diagnosis on Top-k Selection

while GPT-4 tends to rely on self-knowledge without explicit requirements. Consequently, there is a steady improvement in overall $F l$ for GPT-4 when switched to the optimized prompt, while the difference for Llama3 is negligible.

RAG builders can optimize prompts by combining performance on modular metrics provided by RAGCHECKER with user preferences and generator capabilities on different aspects.

Chunk Overlap Does Not Matter a Lot Chunk overlap ratio between adjacent chunks is usually set to be non-zero to help the generator better utilize surrounding information and identify chunks with coherent logic, thus alleviating the impact of hard splits in significant semantics.

According to our results in Fig. 10, higher overlap ratios generally lead to improved context precision. However, this does not necessarily translate to an increase in the total amount of useful information retrieved. This phenomenon can be attributed to the retrieval of more chunks that contain the same segment of useful information. Consequently, we observed that overlap ratio adjustments do not have a significant impact on other performance metrics in a consistent and obvious manner. This suggests that the overlap ratio may not require extensive tuning in practice.

# G Performance Validation of RefChecker with Llama3 Extractor and Checker

We use Llama3-70B-Instruct for the extractor and checker in RefChecker. To validate the effectiveness of this combination, we test its performance on the RefChecker benchmark. As shown in Tab. 16, Llama 3 based RefChecker outperforms the best purely open-sourced combinations reported in the RefChecker paper in all the three context settings.

![](images/8e019b679e70a0d575b6de896f2bc17f920bdec6b5a8ec82b3e110090ad3ec61.jpg)

![](images/0d835207779b6d5b6ecf61601a3b3dded282863fd950fb1a112fb29aed98039d.jpg)

![](images/30c9603c4b9213e395c9d3fdc132e7dce4e64d3184e5e329d90a0340b8303fe0.jpg)

![](images/b59a0b73d14bc7029b732ef892eccd4731428758751887e95be7dbe643deb0cf.jpg)

![](images/a417570cb50837f66257d915e969342cb5cf5ea64d4b100aab44116cf7907a19.jpg)

![](images/a3c07423d9d450649afb260e894027032b1e8b3520c6258cf3fce1ee58e65bf1.jpg)

![](images/e9a331c19aa9702e8f1a3c7a58fbda851aa0353f4ab742c30c228ab1abce503a.jpg)

![](images/f0566b650fdeae27605c541e08dfa01efe511deeb86b80f50b7110e449974058.jpg)

![](images/2c76212ddc6883f18d04864aca7679b92131c5ecb0964b42c9807966897ae602.jpg)  
Figure 7: Diagnosis on Chunk Size

# H Limitations

While RAGCHECKER provides a comprehensive evaluation framework for RAG systems, it has a few limitations that should be acknowledged and addressed in future research.

First, the diagnostic metrics for the retriever component are less insightful compared to those for the generator. The retrieval metrics primarily focus on the recall of ground truth claims and precision of retrieved context, but they may not fully capture the nuances and complexities of the retrieval process. Developing more sophisticated metrics that consider factors such as the information density, diversity and coherence of the retrieved context could provide deeper insights into the retriever’s performance.

Second, the metrics proposed in RAGCHECKER do not differentiate between Neutral and Contradiction checking results from RefChecker when evaluating the generated responses. These two types of results may have different impacts on the final response quality, and treating them equally could lead to an incomplete assessment. Future work should explore ways to incorporate the distinction between neutral and contradiction results into the evaluation metrics, potentially assigning different weights or penalties based on their severity.

Finally, the evaluation benchmark used in this study is curated based on existing text-only datasets and is limited to English queries and corpus. While this allows for a focused evaluation of RAG systems, it may not fully represent the diverse range of tasks and languages that RAG systems can be applied to. Expanding the benchmark to include datasets from different modalities (e.g., images, audio) and languages would provide a more comprehensive assessment of RAG systems’ capabilities and generalization. Additionally, creating benchmark datasets specifically designed for evaluating RAG systems, rather than repurposing existing ones, could help to better capture the unique challenges and requirements of this task.

![](images/1158fc6290d650342a8ea61be6b4e00fe1683af110af11dd4cea9e508ad50cf9.jpg)  
Figure 8: Diagnosis on Generation Prompts

By refining the diagnostic metrics, incorporating the impact of different checking results, and expanding the evaluation benchmark, researchers can gain an even more comprehensive understanding of RAG systems’ performance and identify targeted areas for improvement.

# I Potential Negative Societal Impacts

The RAGCHECKER evaluation framework, while beneficial for assessing RAG systems, could inadvertently lead to several negative societal impacts. There is a risk that developers may focus on optimizing for RAGCHECKER’s specific metrics to the detriment of broader utility and ethical considerations. The computational and financial requirements to meet RAGCHECKER standards could disadvantage smaller organizations, potentially centralizing innovation among well-resourced entities. Moreover, an overreliance on quantitative measures might neglect qualitative factors like user experience and ethical implications.

Table 6: Evaluation results for different RAG systems on ClapNQ dataset   

<table><tr><td rowspan="2">RAG systems</td><td colspan="3">Overall</td><td colspan="2">Retriever</td><td colspan="6">Generator</td><td rowspan="2">#Claim</td></tr><tr><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>CR↑</td><td>CP↑</td><td>CU↑</td><td>NS(I)↓</td><td>NS(II)↓</td><td>Hallu.↓</td><td>SK↓</td><td>Faith.↑</td></tr><tr><td>BM25_GPT-4</td><td>56.9</td><td>50.0</td><td>46.7</td><td>81.1</td><td>41.3</td><td>56.4</td><td>29.4</td><td>5.9</td><td>7.5</td><td>2.2</td><td>90.3</td><td>8</td></tr><tr><td>BM25_Llama3-8b</td><td>49.6</td><td>48.6</td><td>42.2</td><td>81.1</td><td>41.3</td><td>55.2</td><td>31.9</td><td>7.5</td><td>10.8</td><td>2.0</td><td>87.2</td><td>10</td></tr><tr><td>BM25_Llama3-70b</td><td>56.9</td><td>48.7</td><td>45.3</td><td>81.1</td><td>41.3</td><td>55.7</td><td>30.1</td><td>7.1</td><td>5.9</td><td>1.6</td><td>92.4</td><td>7</td></tr><tr><td>BM25_Mixtral-8x7b</td><td>47.9</td><td>49.6</td><td>42.1</td><td>81.1</td><td>41.3</td><td>55.8</td><td>36.9</td><td>7.3</td><td>6.9</td><td>2.3</td><td>90.9</td><td>9</td></tr><tr><td>E5-Mistral_GPT-4</td><td>59.7</td><td>51.1</td><td>47.9</td><td>81.5</td><td>43.6</td><td>59.9</td><td>31.1</td><td>3.8</td><td>5.4</td><td>2.3</td><td>92.3</td><td>9</td></tr><tr><td>E5-Mistral_Llama3-8b</td><td>50.4</td><td>50.9</td><td>43.5</td><td>81.5</td><td>43.6</td><td>59.4</td><td>33.2</td><td>6.4</td><td>10.0</td><td>1.5</td><td>88.5</td><td>10</td></tr><tr><td>E5-Mistral_Llama3-70b</td><td>58.7</td><td>52.8</td><td>48.1</td><td>81.5</td><td>43.6</td><td>61.4</td><td>32.0</td><td>5.1</td><td>4.2</td><td>2.1</td><td>93.7</td><td>8</td></tr><tr><td>E5-Mistral_Mixtral-8x7b</td><td>51.1</td><td>54.4</td><td>45.7</td><td>81.5</td><td>43.6</td><td>63.2</td><td>37.0</td><td>5.2</td><td>5.5</td><td>1.5</td><td>93.0</td><td>10</td></tr></table>

```txt
You are an accurate and reliable AI assistant capable of answering questions using external documents. Always be faithful to the provided documents and leverage relevant, accurate information from them as much as possible. Be aware that external documents might contain noisy or factually incorrect data. Apply critical reasoning to discern and use the correct information from these sources. <context> <content> {chunk_1} </content> <content> {chunk_2} </content> ... <content> {chunk_k} </content> </context> Question: {question} Please answer the question and tag your answer with <answer></answer>. 
```

Figure 9: The optimized prompt for response generation. In this prompt, we explicitly instruct the LLMs to be faithful to the context and identify relevant information as possible.

Table 7: Evaluation results for different RAG systems on NovelQA dataset   

<table><tr><td rowspan="2">RAG systems</td><td colspan="3">Overall</td><td colspan="2">Retriever</td><td colspan="6">Generator</td><td rowspan="2">#Claim</td></tr><tr><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>CR↑</td><td>CP↑</td><td>CU↑</td><td>NS(I)↓</td><td>NS(II)↓</td><td>Hallu.↓</td><td>SK↓</td><td>Faith.↑</td></tr><tr><td>BM25_GPT-4</td><td>71.0</td><td>56.2</td><td>56.4</td><td>82.1</td><td>42.6</td><td>64.9</td><td>17.6</td><td>5.4</td><td>6.1</td><td>2.2</td><td>91.7</td><td>4</td></tr><tr><td>BM25_Llama3-8b</td><td>60.2</td><td>47.8</td><td>45.9</td><td>82.1</td><td>42.6</td><td>55.2</td><td>23.1</td><td>7.1</td><td>9.6</td><td>1.5</td><td>88.8</td><td>3</td></tr><tr><td>BM25_Llama3-70b</td><td>65.0</td><td>51.8</td><td>51.9</td><td>82.1</td><td>42.6</td><td>59.6</td><td>21.4</td><td>7.5</td><td>6.1</td><td>2.1</td><td>91.8</td><td>3</td></tr><tr><td>BM25_Mixtral-8x7b</td><td>56.0</td><td>50.2</td><td>46.0</td><td>82.1</td><td>42.6</td><td>58.4</td><td>24.8</td><td>6.4</td><td>10.9</td><td>2.3</td><td>86.8</td><td>4</td></tr><tr><td>E5-Mistral_GPT-4</td><td>69.4</td><td>56.2</td><td>55.7</td><td>82.7</td><td>45.1</td><td>63.6</td><td>19.4</td><td>6.1</td><td>5.1</td><td>1.7</td><td>93.2</td><td>4</td></tr><tr><td>E5-Mistral_Llama3-8b</td><td>58.7</td><td>48.1</td><td>45.7</td><td>82.7</td><td>45.1</td><td>55.1</td><td>23.5</td><td>8.1</td><td>9.2</td><td>1.5</td><td>89.3</td><td>3</td></tr><tr><td>E5-Mistral_Llama3-70b</td><td>64.5</td><td>50.2</td><td>49.6</td><td>82.7</td><td>45.1</td><td>56.9</td><td>23.7</td><td>5.5</td><td>6.0</td><td>1.5</td><td>92.4</td><td>3</td></tr><tr><td>E5-Mistral_Mixtral-8x7b</td><td>54.2</td><td>48.3</td><td>43.6</td><td>82.7</td><td>45.1</td><td>54.7</td><td>29.6</td><td>6.9</td><td>7.5</td><td>1.6</td><td>90.9</td><td>4</td></tr></table>

Table 8: Evaluation results for different RAG systems on RobustQA - Writing dataset   

<table><tr><td rowspan="2">RAG systems</td><td colspan="3">Overall</td><td colspan="2">Retriever</td><td colspan="6">Generator</td><td rowspan="2">#Claim</td></tr><tr><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>CR↑</td><td>CP↑</td><td>CU↑</td><td>NS(I)↓</td><td>NS(II)↓</td><td>Hallu.↓</td><td>SK↓</td><td>Faith.↑</td></tr><tr><td>BM25_GPT-4</td><td>76.3</td><td>63.6</td><td>66.0</td><td>86.3</td><td>64.3</td><td>70.0</td><td>17.5</td><td>1.0</td><td>5.1</td><td>4.0</td><td>90.9</td><td>10</td></tr><tr><td>BM25_Llama3-8b</td><td>65.0</td><td>59.7</td><td>57.7</td><td>86.3</td><td>64.3</td><td>66.1</td><td>26.0</td><td>1.8</td><td>6.2</td><td>2.3</td><td>91.4</td><td>10</td></tr><tr><td>BM25_Llama3-70b</td><td>72.2</td><td>62.1</td><td>63.6</td><td>86.3</td><td>64.3</td><td>68.4</td><td>23.1</td><td>1.5</td><td>3.2</td><td>2.2</td><td>94.7</td><td>8</td></tr><tr><td>BM25_Mixtral-8x7b</td><td>67.0</td><td>60.1</td><td>59.8</td><td>86.3</td><td>64.3</td><td>66.1</td><td>25.2</td><td>1.5</td><td>4.0</td><td>2.2</td><td>93.8</td><td>8</td></tr><tr><td>E5-Mistral_GPT-4</td><td>77.1</td><td>65.0</td><td>67.3</td><td>91.7</td><td>66.3</td><td>69.0</td><td>17.9</td><td>1.2</td><td>3.8</td><td>1.3</td><td>94.9</td><td>10</td></tr><tr><td>E5-Mistral_Llama3-8b</td><td>67.4</td><td>62.8</td><td>60.6</td><td>91.7</td><td>66.3</td><td>66.8</td><td>25.5</td><td>2.2</td><td>4.5</td><td>0.6</td><td>95.0</td><td>9</td></tr><tr><td>E5-Mistral_Llama3-70b</td><td>73.1</td><td>65.7</td><td>66.2</td><td>91.7</td><td>66.3</td><td>70.1</td><td>23.5</td><td>1.9</td><td>1.5</td><td>0.5</td><td>98.0</td><td>9</td></tr><tr><td>E5-Mistral_Mixtral-8x7b</td><td>66.4</td><td>62.2</td><td>61.3</td><td>91.7</td><td>66.3</td><td>66.4</td><td>26.3</td><td>2.0</td><td>3.2</td><td>0.4</td><td>96.4</td><td>9</td></tr></table>

![](images/a57de094290920488c766af57c6704564b6356e7339407a5e13b925c945d289b.jpg)  
Figure 10: Diagnosis on Chunk Overlap Ratio

Table 9: Evaluation results for different RAG systems on RobustQA - BioASQ dataset   

<table><tr><td rowspan="2">RAG systems</td><td colspan="3">Overall</td><td colspan="2">Retriever</td><td colspan="6">Generator</td><td rowspan="2">#Claim</td></tr><tr><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>CR↑</td><td>CP↑</td><td>CU↑</td><td>NS(I)↓</td><td>NS(II)↓</td><td>Hallu.↓</td><td>SK↓</td><td>Faith.↑</td></tr><tr><td>BM25_GPT-4</td><td>66.1</td><td>43.2</td><td>46.5</td><td>82.1</td><td>39.4</td><td>51.0</td><td>25.0</td><td>5.4</td><td>3.4</td><td>0.7</td><td>95.9</td><td>10</td></tr><tr><td>BM25_Llama3-8b</td><td>64.1</td><td>34.5</td><td>38.5</td><td>82.1</td><td>39.4</td><td>41.0</td><td>21.6</td><td>5.8</td><td>5.7</td><td>0.5</td><td>93.8</td><td>7</td></tr><tr><td>BM25_Llama3-70b</td><td>70.7</td><td>35.4</td><td>42.1</td><td>82.1</td><td>39.4</td><td>41.8</td><td>20.3</td><td>6.0</td><td>2.9</td><td>0.6</td><td>96.4</td><td>7</td></tr><tr><td>BM25_Mixtral-8x7b</td><td>60.6</td><td>43.7</td><td>45.0</td><td>82.1</td><td>39.4</td><td>50.7</td><td>28.3</td><td>6.5</td><td>4.5</td><td>0.9</td><td>94.6</td><td>9</td></tr><tr><td>E5-Mistral_GPT-4</td><td>65.7</td><td>44.0</td><td>46.7</td><td>84.4</td><td>45.2</td><td>50.5</td><td>26.2</td><td>5.9</td><td>2.2</td><td>0.3</td><td>97.5</td><td>10</td></tr><tr><td>E5-Mistral_Llama3-8b</td><td>65.1</td><td>36.0</td><td>40.0</td><td>84.4</td><td>45.2</td><td>41.4</td><td>22.5</td><td>5.8</td><td>3.7</td><td>0.3</td><td>96.0</td><td>8</td></tr><tr><td>E5-Mistral_Llama3-70b</td><td>69.9</td><td>37.3</td><td>43.5</td><td>84.4</td><td>45.2</td><td>43.0</td><td>22.2</td><td>5.7</td><td>2.2</td><td>0.4</td><td>97.3</td><td>7</td></tr><tr><td>E5-Mistral_Mixtral-8x7b</td><td>58.2</td><td>44.5</td><td>44.6</td><td>84.4</td><td>45.2</td><td>50.3</td><td>31.6</td><td>7.0</td><td>2.9</td><td>0.7</td><td>96.4</td><td>10</td></tr></table>

Table 10: Evaluation results for different RAG systems on RobustQA - Finance dataset   

<table><tr><td rowspan="2">RAG systems</td><td colspan="3">Overall</td><td colspan="2">Retriever</td><td colspan="6">Generator</td><td rowspan="2">#Claim</td></tr><tr><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>CR↑</td><td>CP↑</td><td>CU↑</td><td>NS(I)↓</td><td>NS(II)↓</td><td>Hallu.↓</td><td>SK↓</td><td>Faith.↑</td></tr><tr><td>BM25_GPT-4</td><td>54.0</td><td>50.1</td><td>48.2</td><td>69.4</td><td>52.1</td><td>62.3</td><td>26.8</td><td>3.9</td><td>15.4</td><td>4.7</td><td>79.9</td><td>15</td></tr><tr><td>BM25_Llama3-8b</td><td>44.0</td><td>42.6</td><td>39.3</td><td>69.4</td><td>52.1</td><td>55.0</td><td>35.5</td><td>6.6</td><td>13.5</td><td>2.5</td><td>84.0</td><td>14</td></tr><tr><td>BM25_Llama3-70b</td><td>53.8</td><td>43.6</td><td>44.5</td><td>69.4</td><td>52.1</td><td>56.7</td><td>34.3</td><td>5.2</td><td>6.5</td><td>2.1</td><td>91.4</td><td>10</td></tr><tr><td>BM25_Mixtral-8x7b</td><td>46.4</td><td>41.3</td><td>39.3</td><td>69.4</td><td>52.1</td><td>53.0</td><td>37.4</td><td>6.4</td><td>6.7</td><td>2.2</td><td>91.1</td><td>11</td></tr><tr><td>E5-Mistral_GPT-4</td><td>56.0</td><td>54.6</td><td>51.8</td><td>86.3</td><td>67.4</td><td>60.2</td><td>31.7</td><td>2.7</td><td>9.5</td><td>1.4</td><td>89.1</td><td>15</td></tr><tr><td>E5-Mistral_Llama3-8b</td><td>46.9</td><td>50.1</td><td>43.9</td><td>86.3</td><td>67.4</td><td>55.5</td><td>38.4</td><td>4.6</td><td>9.5</td><td>1.1</td><td>89.4</td><td>14</td></tr><tr><td>E5-Mistral_Llama3-70b</td><td>56.0</td><td>51.4</td><td>49.9</td><td>86.3</td><td>67.4</td><td>57.5</td><td>35.2</td><td>3.9</td><td>4.9</td><td>0.6</td><td>94.4</td><td>12</td></tr><tr><td>E5-Mistral_Mixtral-8x7b</td><td>49.0</td><td>49.0</td><td>45.0</td><td>86.3</td><td>67.4</td><td>54.7</td><td>39.0</td><td>5.3</td><td>3.9</td><td>0.6</td><td>95.5</td><td>12</td></tr></table>

Table 11: Evaluation results for different RAG systems on RobustQA - Lifestyle dataset   

<table><tr><td rowspan="2">RAG systems</td><td colspan="3">Overall</td><td colspan="2">Retriever</td><td colspan="6">Generator</td><td rowspan="2">#Claim</td></tr><tr><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>CR↑</td><td>CP↑</td><td>CU↑</td><td>NS(I)↓</td><td>NS(II)↓</td><td>Hallu.↓</td><td>SK↓</td><td>Faith.↑</td></tr><tr><td>BM25_GPT-4</td><td>63.3</td><td>50.5</td><td>53.0</td><td>70.2</td><td>47.0</td><td>64.8</td><td>24.0</td><td>2.7</td><td>10.0</td><td>6.0</td><td>84.0</td><td>12</td></tr><tr><td>BM25_Llama3-8b</td><td>49.7</td><td>44.8</td><td>43.8</td><td>70.2</td><td>47.0</td><td>59.2</td><td>33.2</td><td>6.1</td><td>11.1</td><td>2.2</td><td>86.8</td><td>12</td></tr><tr><td>BM25_Llama3-70b</td><td>59.6</td><td>44.4</td><td>47.5</td><td>70.2</td><td>47.0</td><td>58.5</td><td>30.7</td><td>3.8</td><td>5.9</td><td>2.4</td><td>91.8</td><td>9</td></tr><tr><td>BM25_Mixtral-8x7b</td><td>52.8</td><td>43.5</td><td>44.1</td><td>70.2</td><td>47.0</td><td>56.8</td><td>34.5</td><td>4.9</td><td>6.8</td><td>2.3</td><td>90.9</td><td>10</td></tr><tr><td>E5-Mistral_GPT-4</td><td>66.4</td><td>57.6</td><td>58.9</td><td>89.7</td><td>64.0</td><td>62.5</td><td>26.2</td><td>2.2</td><td>5.2</td><td>1.4</td><td>93.4</td><td>13</td></tr><tr><td>E5-Mistral_Llama3-8b</td><td>54.6</td><td>55.0</td><td>51.3</td><td>89.7</td><td>64.0</td><td>59.7</td><td>34.6</td><td>4.4</td><td>6.3</td><td>0.6</td><td>93.0</td><td>14</td></tr><tr><td>E5-Mistral_Llama3-70b</td><td>64.8</td><td>56.7</td><td>57.3</td><td>89.7</td><td>64.0</td><td>61.7</td><td>29.3</td><td>3.3</td><td>2.6</td><td>0.7</td><td>96.7</td><td>12</td></tr><tr><td>E5-Mistral_Mixtral-8x7b</td><td>56.2</td><td>51.8</td><td>50.5</td><td>89.7</td><td>64.0</td><td>56.5</td><td>34.8</td><td>4.6</td><td>3.0</td><td>0.8</td><td>96.2</td><td>11</td></tr></table>

Table 12: Evaluation results for different RAG systems on RobustQA - Recreation dataset   

<table><tr><td rowspan="2">RAG systems</td><td colspan="3">Overall</td><td colspan="2">Retriever</td><td colspan="6">Generator</td><td rowspan="2">#Claim</td></tr><tr><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>CR↑</td><td>CP↑</td><td>CU↑</td><td>NS(I)↓</td><td>NS(II)↓</td><td>Hallu.↓</td><td>SK↓</td><td>Faith.↑</td></tr><tr><td>BM25_GPT-4</td><td>62.9</td><td>51.9</td><td>53.1</td><td>70.4</td><td>37.5</td><td>65.7</td><td>21.4</td><td>4.6</td><td>11.1</td><td>5.2</td><td>83.7</td><td>11</td></tr><tr><td>BM25_Llama3-8b</td><td>50.6</td><td>45.1</td><td>43.1</td><td>70.4</td><td>37.5</td><td>58.1</td><td>31.0</td><td>10.1</td><td>8.1</td><td>1.8</td><td>90.2</td><td>11</td></tr><tr><td>BM25_Llama3-70b</td><td>58.1</td><td>45.7</td><td>46.5</td><td>70.4</td><td>37.5</td><td>60.2</td><td>30.4</td><td>6.3</td><td>4.9</td><td>2.0</td><td>93.1</td><td>8</td></tr><tr><td>BM25_Mixtral-8x7b</td><td>50.5</td><td>44.3</td><td>42.4</td><td>70.4</td><td>37.5</td><td>56.2</td><td>33.1</td><td>8.3</td><td>6.7</td><td>1.8</td><td>91.5</td><td>9</td></tr><tr><td>E5-Mistral_GPT-4</td><td>62.4</td><td>57.0</td><td>56.1</td><td>85.1</td><td>51.1</td><td>64.2</td><td>27.8</td><td>4.1</td><td>5.7</td><td>1.5</td><td>92.8</td><td>12</td></tr><tr><td>E5-Mistral_Llama3-8b</td><td>51.1</td><td>52.5</td><td>47.3</td><td>85.1</td><td>51.1</td><td>59.4</td><td>34.1</td><td>9.2</td><td>5.3</td><td>0.7</td><td>94.0</td><td>13</td></tr><tr><td>E5-Mistral_Llama3-70b</td><td>60.4</td><td>53.7</td><td>52.7</td><td>85.1</td><td>51.1</td><td>60.3</td><td>30.8</td><td>6.2</td><td>2.6</td><td>0.7</td><td>96.7</td><td>10</td></tr><tr><td>E5-Mistral_Mixtral-8x7b</td><td>52.1</td><td>51.8</td><td>47.9</td><td>85.1</td><td>51.1</td><td>58.6</td><td>34.1</td><td>8.2</td><td>3.9</td><td>0.6</td><td>95.5</td><td>11</td></tr></table>

Table 13: Evaluation results for different RAG systems on RobustQA - Science dataset   

<table><tr><td rowspan="2">RAG systems</td><td colspan="3">Overall</td><td colspan="2">Retriever</td><td colspan="6">Generator</td><td rowspan="2">#Claim</td></tr><tr><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>CR↑</td><td>CP↑</td><td>CU↑</td><td>NS(I)↓</td><td>NS(II)↓</td><td>Hallu.↓</td><td>SK↓</td><td>Faith.↑</td></tr><tr><td>BM25_GPT-4</td><td>58.7</td><td>52.6</td><td>51.8</td><td>71.3</td><td>62.6</td><td>66.6</td><td>27.6</td><td>2.4</td><td>11.3</td><td>4.3</td><td>84.5</td><td>14</td></tr><tr><td>BM25_Llama3-8b</td><td>47.9</td><td>45.2</td><td>41.7</td><td>71.3</td><td>62.6</td><td>58.2</td><td>32.2</td><td>5.3</td><td>14.2</td><td>1.7</td><td>84.1</td><td>14</td></tr><tr><td>BM25_Llama3-70b</td><td>55.8</td><td>45.0</td><td>45.5</td><td>71.3</td><td>62.6</td><td>57.7</td><td>33.0</td><td>4.5</td><td>6.7</td><td>1.6</td><td>91.7</td><td>10</td></tr><tr><td>BM25_Mixtral-8x7b</td><td>51.5</td><td>45.1</td><td>44.0</td><td>71.3</td><td>62.6</td><td>58.5</td><td>34.8</td><td>5.4</td><td>7.1</td><td>1.8</td><td>91.1</td><td>10</td></tr><tr><td>E5-Mistral_GPT-4</td><td>57.9</td><td>55.0</td><td>53.0</td><td>85.0</td><td>71.8</td><td>61.5</td><td>31.5</td><td>2.3</td><td>8.4</td><td>1.5</td><td>90.2</td><td>15</td></tr><tr><td>E5-Mistral_Llama3-8b</td><td>48.8</td><td>48.5</td><td>43.5</td><td>85.0</td><td>71.8</td><td>54.8</td><td>39.1</td><td>4.6</td><td>6.6</td><td>0.6</td><td>92.8</td><td>13</td></tr><tr><td>E5-Mistral_Llama3-70b</td><td>56.7</td><td>51.3</td><td>49.6</td><td>85.0</td><td>71.8</td><td>57.7</td><td>36.1</td><td>3.9</td><td>3.2</td><td>0.5</td><td>96.3</td><td>11</td></tr><tr><td>E5-Mistral_Mixtral-8x7b</td><td>54.5</td><td>49.2</td><td>47.4</td><td>85.0</td><td>71.8</td><td>55.3</td><td>37.1</td><td>3.7</td><td>4.4</td><td>0.6</td><td>95.0</td><td>11</td></tr></table>

Table 14: Evaluation results for different RAG systems on RobustQA - Technology dataset   

<table><tr><td rowspan="2">RAG systems</td><td colspan="3">Overall</td><td colspan="2">Retriever</td><td colspan="6">Generator</td><td rowspan="2">#Claim</td></tr><tr><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>CR↑</td><td>CP↑</td><td>CU↑</td><td>NS(I)↓</td><td>NS(II)↓</td><td>Hallu.↓</td><td>SK↓</td><td>Faith.↑</td></tr><tr><td>BM25_GPT-4</td><td>57.5</td><td>48.6</td><td>48.9</td><td>69.5</td><td>63.8</td><td>63.4</td><td>28.1</td><td>3.1</td><td>11.2</td><td>4.3</td><td>84.5</td><td>14</td></tr><tr><td>BM25_Llama3-8b</td><td>47.2</td><td>46.3</td><td>42.1</td><td>69.5</td><td>63.8</td><td>61.2</td><td>36.3</td><td>5.8</td><td>10.1</td><td>1.7</td><td>88.2</td><td>14</td></tr><tr><td>BM25_Llama3-70b</td><td>55.9</td><td>44.4</td><td>45.5</td><td>69.5</td><td>63.8</td><td>59.2</td><td>35.6</td><td>4.4</td><td>4.0</td><td>1.6</td><td>94.4</td><td>10</td></tr><tr><td>BM25_Mixtral-8x7b</td><td>51.7</td><td>43.4</td><td>42.9</td><td>69.5</td><td>63.8</td><td>58.2</td><td>36.5</td><td>5.4</td><td>5.6</td><td>1.7</td><td>92.7</td><td>11</td></tr><tr><td>E5-Mistral_GPT-4</td><td>59.9</td><td>55.0</td><td>53.6</td><td>83.7</td><td>76.4</td><td>63.3</td><td>31.8</td><td>2.3</td><td>6.0</td><td>1.2</td><td>92.8</td><td>15</td></tr><tr><td>E5-Mistral_Llama3-8b</td><td>47.9</td><td>51.1</td><td>44.8</td><td>83.7</td><td>76.4</td><td>59.0</td><td>40.4</td><td>5.6</td><td>5.9</td><td>0.5</td><td>93.6</td><td>15</td></tr><tr><td>E5-Mistral_Llama3-70b</td><td>56.9</td><td>53.5</td><td>50.7</td><td>83.7</td><td>76.4</td><td>62.1</td><td>36.7</td><td>3.8</td><td>2.6</td><td>0.3</td><td>97.1</td><td>13</td></tr><tr><td>E5-Mistral_Mixtral-8x7b</td><td>52.6</td><td>51.5</td><td>47.9</td><td>83.7</td><td>76.4</td><td>59.6</td><td>40.5</td><td>3.4</td><td>3.2</td><td>0.2</td><td>96.6</td><td>12</td></tr></table>

Table 15: Evaluation results for different RAG systems on KIWI dataset   

<table><tr><td rowspan="2">RAG systems</td><td colspan="3">Overall</td><td colspan="2">Retriever</td><td colspan="6">Generator</td><td rowspan="2">#Claim</td></tr><tr><td>Prec.↑</td><td>Rec.↑</td><td>F1↑</td><td>CR↑</td><td>CP↑</td><td>CU↑</td><td>NS(I)↓</td><td>NS(II)↓</td><td>Hallu.↓</td><td>SK↓</td><td>Faith.↑</td></tr><tr><td>BM25_GPT-4</td><td>42.8</td><td>30.0</td><td>32.4</td><td>57.8</td><td>72.5</td><td>49.1</td><td>45.0</td><td>6.2</td><td>6.0</td><td>0.7</td><td>93.3</td><td>18</td></tr><tr><td>BM25_Llama3-8b</td><td>43.0</td><td>24.7</td><td>26.6</td><td>57.8</td><td>72.5</td><td>39.6</td><td>41.8</td><td>4.8</td><td>9.1</td><td>1.7</td><td>89.2</td><td>16</td></tr><tr><td>BM25_Llama3-70b</td><td>42.7</td><td>27.4</td><td>31.0</td><td>57.8</td><td>72.5</td><td>43.8</td><td>45.2</td><td>6.8</td><td>5.2</td><td>0.7</td><td>94.0</td><td>18</td></tr><tr><td>BM25_Mixtral-8x7b</td><td>40.2</td><td>21.6</td><td>23.8</td><td>57.8</td><td>72.5</td><td>35.5</td><td>51.0</td><td>5.8</td><td>3.1</td><td>0.3</td><td>96.6</td><td>13</td></tr><tr><td>E5-Mistral_GPT-4</td><td>45.5</td><td>34.0</td><td>36.0</td><td>64.6</td><td>86.7</td><td>49.0</td><td>44.9</td><td>4.0</td><td>5.5</td><td>1.5</td><td>93.0</td><td>20</td></tr><tr><td>E5-Mistral_Llama3-8b</td><td>47.2</td><td>27.8</td><td>29.8</td><td>64.6</td><td>86.7</td><td>39.1</td><td>43.4</td><td>4.2</td><td>4.6</td><td>0.3</td><td>95.2</td><td>15</td></tr><tr><td>E5-Mistral_Llama3-70b</td><td>45.2</td><td>30.9</td><td>34.0</td><td>64.6</td><td>86.7</td><td>45.3</td><td>47.5</td><td>3.7</td><td>3.6</td><td>0.3</td><td>96.1</td><td>18</td></tr><tr><td>E5-Mistral_Mixtral-8x7b</td><td>36.7</td><td>23.1</td><td>23.5</td><td>64.6</td><td>86.7</td><td>32.2</td><td>55.0</td><td>4.9</td><td>2.9</td><td>0.5</td><td>96.7</td><td>14</td></tr></table>

Table 16: Performance of RefChecker on the RefChecker benchmark using Llama 3 70B Instruct as both the extractor and checker. We compare the results with the best performed purely open-sourced combinations reported in the RefChecker paper.   

<table><tr><td></td><td>Accuracy</td><td>Fact. F1</td><td>Non-Fact. F1</td><td>Pearson</td><td>Spearman</td></tr><tr><td colspan="6">Zero Context</td></tr><tr><td>Mistral-SFT + RepC</td><td>89.38</td><td>80.43</td><td>92.72</td><td>77.14</td><td>76.74</td></tr><tr><td>Llama3 + Llama3</td><td>91.89</td><td>83.06</td><td>94.67</td><td>81.77</td><td>80.83</td></tr><tr><td colspan="6">Noisy Context</td></tr><tr><td>Mistral-SFT + NLI</td><td>70.82</td><td>75.12</td><td>64.72</td><td>52.21</td><td>45.61</td></tr><tr><td>Llama3 + Llama3</td><td>71.75</td><td>76.69</td><td>64.15</td><td>57.67</td><td>50.31</td></tr><tr><td colspan="6">Accurate Context</td></tr><tr><td>Mistral-SFT + AlignScore</td><td>74.12</td><td>81.6</td><td>56.38</td><td>46.34</td><td>43.22</td></tr><tr><td>Llama3 + Llama3</td><td>78.35</td><td>84.87</td><td>61.92</td><td>59.48</td><td>52.03</td></tr></table>
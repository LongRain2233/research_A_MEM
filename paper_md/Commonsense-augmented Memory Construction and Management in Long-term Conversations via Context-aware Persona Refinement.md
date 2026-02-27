# Commonsense-augmented Memory Construction and Management in Long-term Conversations via Context-aware Persona Refinement

Hana ${ \bf K i m } ^ { 1 }$ Kai Tzu-iunn $\mathbf { O n g ^ { 2 } }$ Seoyeon ${ \bf K i m ^ { 2 } }$ Dongha Lee2 Jinyoung Yeo2 Department of Computer Science1, Artificial Intelligence2, Yonsei University {hana.kim,ktio89,emseoyk,donalee,jinyeo}@yonsei.ac.kr

# Abstract

Memorizing and utilizing speakers’ personas is a common practice for response generation in long-term conversations. Yet, human-authored datasets often provide uninformative persona sentences that hinder response quality. This paper presents a novel framework that leverages commonsense-based persona expansion to address such issues in long-term conversation. While prior work focuses on not producing personas that contradict others, we focus on transforming contradictory personas into sentences that contain rich speaker information, by refining them based on their contextual backgrounds with designed strategies. As the pioneer of persona expansion in multi-session settings, our framework facilitates better response generation via human-like persona refinement. The supplementary video of our work is available at https://caffeine-15bbf.web.app/.

# 1 Introduction

Memorizing participants’ personal information and conversing accordingly is important for dialogue systems to maintain long-term intimacy with users (Adiwardana et al., 2020). For that, studies have proposed datasets of long-term conversations, which require dialogue systems to memorize and utilize speakers’ personas from past dialogue sessions to generate proper responses (Xu et al., 2022; Bae et al., 2022). Regardless, human-authored personas can be generic and over-simplified, limiting the generation of diverse and engaging responses.

Intuitively, this can be addressed by expanding existing personas with commonsense expansion (Majumder et al., 2020). However, such a naive remedy can raise contradiction between personas (e.g., “I am lazy” and “I clean my room every day”), especially as sessions accumulate (Figure 7), hindering consistent response generation. While we can simply get rid of contradictory personas utilizing external modules such as models for natural language inference (NLI), it yields sub-optimal

![](images/cc58923fe3621bf3afe11472ccd6a43a12536062a361531ac7240113ab174372.jpg)  
Figure 1: Contradictory personas can co-exist and provide rich speaker information for the conversation when their contexts are considered (an empirical example).

results (Section 3.2). Also, avoiding contradictory personas (Bae et al., 2022; Kim et al., 2023) does not align with human personality traits. Since human personality is context-dependent (van Oers et al., 2005), we naturally exhibit different personalities and behaviors in different contexts, allowing personas with contradictory interpretations to coexist as one’s personas, as shown in Figure 1.

Motivated by these, in this paper, we tackle such bottleneck of persona expansion in longterm conversations. Specifically, we focus on transforming contradictory personas into sentences that contain richer speaker information. To this end, we present $\circledast$ CAFFEINE, a Context-Aware reFinement Framework for contradictory pErsonas IN long-tErm conversations. CAFFEINE leverages large language models (LLMs) to iteratively refine the contradictory personas within/across the session(s) based on their contextual background with designed strategies. Our contributions are two-fold: (i) To the best of our knowledge, we are the first to explore commonsense-based persona expansion in multi-session settings; (ii) CAFFEINE enables better response generation in long-term conversations in both automatic and human evaluations. Also, it refines contradictory personas in a human-like manner, eliciting persona sentences that are superior in

![](images/71fe6ce52ccb8f01338295f6594b94786c9f9ceed1e4cc8a91b99e76645dc492.jpg)  
Figure 2: At the end of each dialogue session, CAFFEINE refines contradictory personas within/across the session(s) and saves the refined version to the dialogue model’s memory for response generation in the next session.

various criteria while being cost- and time-efficient.

# 2 Approach

Long-term conversations involve multiple dialogue sessions. At the end of each session, we perform:

# 2.1 Commonsense-based Persona Expansion

Following Majumder et al. (2020), we perform commonsense expansion on personas derived from the conversation using COMET (Hwang et al., 2020). COMET generates commonsense knowledge based on cause-effect relation types (e.g., XNEED and XWANT). For example, “I drink coffee” $ ^ { \infty } I$ want to stay awake”. Implementation details on COMET expansion are in Appendix A.1.

# 2.2 $\circledast$ CAFFEINE

We present the overview of CAFFEINE in Figure 2.

# 2.2.1 Preparation: Graph Construction for Iterative Persona Refinement

After expansion, we identify contradictory personas by computing the probability of contradiction $\delta$ between all personas with an external NLI model. To refine contradictory personas cost- and time-efficiently, we adopt iterative refinement with a graph structure: Contradictory pairs with $\delta$ larger than a threshold $\mu$ are added as nodes $V$ (edges $E = ( \delta _ { 1 } , \delta _ { 2 } , \cdot \cdot \cdot , \delta _ { | E | } )$ ) to the refinement graph $G$ .1 Then, we locate the node (persona) $p _ { 1 }$ with the largest $\Sigma \delta$ within its neighborhood. We select $p _ { 1 }$ and the adjacent node $p _ { 2 }$ with the highest $\delta$ with $p _ { 1 }$ for the first refinement iteration (Algorithm 1).

# 2.2.2 Context-aware Persona Refinement

As shown in Figure 1, personas causing contradiction can be logically acceptable and beneficial for conversations if contextual cues from their origin context are appended via commonsense reasoning. For that, we propose the following refinement strategies for the LLM to choose from:

Strategy I: Resolution. Inspired by entity resolution (Benjelloun et al., 2009), persona resolution resolves the contradiction between personas by seamlessly merging them into one informative sentence based on the contextual background from where they are derived (Figure 3 (a), Figure 1 is also an example of persona resolution).

Strategy II: Disambiguation. Contradiction between two statements can stem from the lack of contexts, known as pragmatic ambiguity (Macagno and Bigi, 2018). Drawn from entity disambiguation (Dredze et al., 2010), persona disambiguation specifies each persona with relevant information from their contextual backgrounds (Figure 3 (b)).

![](images/2f71c824dc728b22173a3504a511a6e2074d55068a97a0e351cc67b6aed3b437.jpg)  
Figure 3: Empirical demonstration of our strategies. Top: relevant contexts; Mid: contradictory personas; Bottom: refined persona(s).

Strategy III: Preservation. Due to the limitation of NLI models, personas predicted as contradictory

Table 1: Performance in response generation. Bold and underline show the best and second-highest in each column.   

<table><tr><td rowspan="2">Settings</td><td colspan="3">Session 2</td><td colspan="3">Session 3</td><td colspan="3">Session 4</td><td colspan="3">Session 5</td></tr><tr><td>B-1</td><td>R-1</td><td>R-L</td><td>B-1</td><td>R-1</td><td>R-L</td><td>B-1</td><td>R-1</td><td>R-L</td><td>B-1</td><td>R-1</td><td>R-L</td></tr><tr><td>No Memory</td><td>20.75</td><td>19.38</td><td>15.16</td><td>20.42</td><td>19.53</td><td>15.09</td><td>19.88</td><td>19.56</td><td>14.98</td><td>19.87</td><td>20.16</td><td>15.33</td></tr><tr><td>GOLD</td><td>21.19</td><td>19.86</td><td>15.50</td><td>21.24</td><td>20.16</td><td>15.47</td><td>20.57</td><td>19.94</td><td>15.16</td><td>20.49</td><td>20.53</td><td>15.55</td></tr><tr><td>+ NLI-remove</td><td>20.81</td><td>19.98</td><td>15.26</td><td>21.04</td><td>20.28</td><td>15.52</td><td>21.33</td><td>20.69</td><td>15.91</td><td>21.43</td><td>20.75</td><td>15.95</td></tr><tr><td>+ NLI-recent</td><td>20.87</td><td>20.09</td><td>15.39</td><td>21.14</td><td>20.52</td><td>15.71</td><td>21.46</td><td>20.79</td><td>15.97</td><td>21.60</td><td>20.97</td><td>16.11</td></tr><tr><td>+ CAFFEINE</td><td>20.93</td><td>20.18</td><td>15.47</td><td>21.41</td><td>20.72</td><td>15.86</td><td>21.67</td><td>21.00</td><td>16.15</td><td>21.92</td><td>21.23</td><td>16.31</td></tr><tr><td>COMET-EXP</td><td>21.23</td><td>19.82</td><td>15.44</td><td>20.95</td><td>19.90</td><td>15.38</td><td>20.33</td><td>20.02</td><td>15.18</td><td>20.00</td><td>20.27</td><td>15.37</td></tr><tr><td>+ NLI-remove</td><td>20.72</td><td>19.96</td><td>15.27</td><td>21.12</td><td>20.40</td><td>15.56</td><td>21.66</td><td>20.77</td><td>15.88</td><td>21.77</td><td>20.91</td><td>16.01</td></tr><tr><td>+ NLI-recent</td><td>20.73</td><td>20.00</td><td>15.33</td><td>21.16</td><td>20.40</td><td>15.64</td><td>21.57</td><td>20.77</td><td>15.89</td><td>21.78</td><td>20.99</td><td>16.09</td></tr><tr><td>+ CAFFEINE</td><td>20.97</td><td>20.06</td><td>15.32</td><td>21.63</td><td>20.73</td><td>15.86</td><td>21.97</td><td>21.10</td><td>16.18</td><td>22.26</td><td>21.32</td><td>16.37</td></tr></table>

may be consistent and may not require refinement. Thus, we allow the LLM to preserve personas as they are when their contexts suggest so.

In practice, with contradictory personas $\mathcal { P } =$ $( p _ { 1 } , p _ { 2 } )$ and relevant dialogue contexts $\begin{array} { r l } { \mathcal { D } } & { { } = } \end{array}$ $( d _ { 1 } , d _ { 2 } )$ from where $\mathcal { P }$ are derived, we prompt the LLM to choose one out of the three strategies $S$ with rationale and generate the refinement $\mathcal { R }$ :

$$
\begin{array}{l} \mathcal {S} ^ {*} = \underset {\mathcal {S}} {\operatorname {a r g m a x}} P _ {\mathrm {L L M}} (\mathcal {S} | \mathcal {P}, \mathcal {D}) (1) \\ \Rightarrow \mathcal {R} ^ {*} = \underset {\mathcal {R}} {\operatorname {a r g m a x}} P _ {\mathrm {L L M}} (\mathcal {R} | \mathcal {P}, \mathcal {D}, \mathcal {S} ^ {*}) (2) \\ \end{array}
$$

where $\Rightarrow$ denotes a sequential generation of tokens. $D$ consists of $w$ consecutive utterances.2 When $p$ is a persona generated by COMET, we use $D$ of its original persona and concatenate the original persona with $D$ . After refinement, we save $\mathcal { R } ^ { * }$ to long-term memory $\mathcal { M }$ and remove $\mathcal { P }$ from the graph $G$ , and start the next iteration (Algorithm 1).

# Algorithm 1 Iterative Graph Refinement

Require: Refinement graph $G ( V , E )$

Ensure: The dialogue model’s long-term memory $\mathcal { M }$

1: $\mathcal { M }  \mathcal { M } \setminus V$   
2: while $G \neq \emptyset$ do   
3: Select $p _ { 1 }$ in $V$ with the highest $\Sigma \delta$   
4: Select $p _ { 2 }$ , a neighbor of $p _ { 1 }$ with the highest $\delta$   
$( S ^ { \ast } , \mathcal { R } ^ { \ast } ) \gets \mathrm { R e f i n e } ( p _ { 1 } , p _ { 2 } )$   
6: M ← M ∪ R∗   
7: Remove $p _ { 1 } , p _ { 2 }$ from $G$   
8: Remove isolated nodes from $G$   
9: return $\mathcal { M }$

# 3 Experiments

# 3.1 Experimental Settings

Dataset. We use Multi-Session Chat (MSC) (Xu et al., 2022) to conduct experiments. MSC takes

the dialogues from Persona-Chat (Zhang et al., 2018) and extends their follow-up conversations throughout several sessions. Each session comes with speakers’ personas authored by humans.

Models and baselines. In this work, we use Chat-GPT (OpenAI, 2023) for CAFFEINE and response generation (RG),3 and Contriever (Izacard et al., 2021) to retrieve top- $k$ relevant personas from longterm memory.4 As for the NLI model, we use RoBERTa (Liu et al., 2019) fine-tuned on the MNLI dataset (Williams et al., 2017). To evaluate the effectiveness of CAFFEINE in RG, we apply it to: (i) COMET-EXP, human-authored personas with COMET expansion; (ii) GOLD, human-authored personas. We include this setting as a contradiction can also exist among un-expanded personas.5 Also, to justify our choice to refine rather than remove, we compare CAFFEINE with two baselines: NLI-remove and NLI-recent.6 The NLI-remove approach filters out personas that contradict at least one other persona with $\delta \ge 0 . 8$ via the NLI model. Similarly, the NLI-recent approach also uses the NLI model, but it differs by keeping the most recent persona in contradictory persona pairs and removing the older one (Bae et al., 2022), thereby prioritizing updated personas over time.

# 3.2 Results and Discussion

We present the empirical findings of the following research questions guiding our experiments:

Table 2: Comparison of generated responses. We report CAFFEINE’s winning rate. (*: p-value $< 0 . 0 5$ )   

<table><tr><td>CAFFEINE vs.</td><td>GOLD</td><td>COMET-EXP</td><td>NLI-remove</td></tr><tr><td>Naturalness</td><td>73%*</td><td>71%*</td><td>79%*</td></tr><tr><td>Consistency</td><td>66%*</td><td>62%*</td><td>67%*</td></tr><tr><td>Specificity</td><td>55%</td><td>53%</td><td>51%</td></tr><tr><td>Engagingness</td><td>63%*</td><td>64%*</td><td>66%*</td></tr><tr><td>Overall</td><td>62%*</td><td>63%*</td><td>67%*</td></tr></table>

RQ1: Does CAFFEINE benefit response generation in long-term conversations?

RQ2: Does CAFFEINE refine personas in a way that aligns with human judgment?

RQ3: Is CAFFEINE cost- and time-efficient?

CAFFEINE improves response generation (RQ1). To evaluate the efficacy of CAFFEINE, we conduct experiments on response generation (RG) using sessions 2 to 5 of each dialogue from MSC. Table 1 shows the results of RG in MSC with BLEU-1 (B-1), ROUGE-1 (R-1), and ROUGE-L (R-L) (Papineni et al., 2002; Lin, 2004). Applying CAFFEINE yields performance gains, which are more significant as sessions increase. Also, CAFFEINE consistently outperforms NLI-remove and NLI-recent, showing that leveraging contradictory personas elicits a more informative memory for RG than removing them. Compared to NLIremove, the improved efficacy of NLI-recent is attributed to its focus on the recency of personas. By eliminating outdated personas from contradictory pairs, NLI-recent enhances RG, yielding responses more aligned with the current dialogue context. However, despite the enhancements in NLIrecent performance, CAFFEINE still exhibits superior performance. Furthermore, the performance brought by CAFFEINE exhibits a continuously rising trend as the number of previous sessions increases, while baselines yield a flat or downward tendency. These demonstrate the effectiveness of CAFFEINE in multi-session conversations. Table 2 shows the human evaluation results of randomly sampled 50 responses conducted by 3 judges from Amazon Mechanical Turk (Appendix F). CAF-FEINE yields responses that are better (i.e., winning) in several criteria. We provide examples of RG in Appendix E.

CAFFEINE elicits personas that align with human preference (RQ2). We sample 100 persona pairs refined with “resolution” or “disambiguation” and ask 3 judges “ whether they are contradictory

![](images/d2331fd9aaba89c333dcce0fa7eb75badac81a06b4191ccf1f8d1574bf325a00.jpg)  
Figure 4: Human evaluation results on (i) refined personas and (ii) the refinement process (p-value $< 0 . 0 5$ ).

before refinement from a human standpoint”. 89 samples that receive “yes” from all judges are used for the evaluation. Judges compare the refined version with its un-refined version and vote if they agree: it is less contradictory (Consistency); it provides more speaker information (Specificity); it is more useful when having a conversation with this person (Helpfulness); it has better quality (Overall); the refinement process is reasonable (Humanlikeness). Figure 4 shows that personas refined by CAFFEINE are greater in all criteria, especially helpfulness. This supports our argument that contradictory personas become sentences with rich speaker information for the conversation if cues from their relevant contexts are included, and explains the performance gain in RG. Also, a $69 \%$ agreement on human-likeness demonstrates that CAFFEINE’s refinement is in line with human judgment. Refinement examples are presented in Appendix E.

CAFFEINE refines personas in a cost- and timeefficient manner (RQ3). In CAFFEINE, we remove refined $( p _ { 1 } , p _ { 2 } )$ from $G$ after refinement. Figure 5 compares this with a setting without such removal, i.e., all $| E |$ contradictory persona pairs in $G$ are all refined (denoted as ALL). While yielding similar RG performance (Session 2-5), ours requires significantly fewer API calls per dialogue per session, especially as the sessions accumulate (9-fold $ 2 1$ -fold more cost- and time-efficient).

![](images/95bc32af2b79e692080d8d96d17a8b76a99bf8a656cacd6734ab1f544481a30b.jpg)  
Figure 5: Cost and time efficiency of our algorithm.

# 4 Related Work

Many studies have utilized commonsense knowledge for response generation. For instance: leveraging knowledge from a general-purpose knowledge model (Zhou et al., 2022b; Wu et al., 2022;

Liu et al., 2022; Li et al., 2023); training commonsense generators for dialogues via humanannotated dataset (Ghosal et al., 2022); formulating commonsense-linking between knowledge graphs and dialogues (Gao et al., 2022); modeling speakers’ mutual beliefs before a response (Zhou et al., 2022a); integrating implicit information in dialogues into rationale for more effective response generation (Chae et al., 2023). While most work focuses on speaker utterances, we leverage speaker personas to address commonsense knowledge in response generation.

# 5 Conclusion

This work pioneers commonsense-based persona expansion in multi-session settings and presents a context-aware refinement framework that leverages contradictory personas to elicit a memory with richer speaker details. $\circledast$ CAFFEINE improves response generation in long-term conversations and demonstrates human-like refinement of contradictory personas while being cost- and time-efficient.

# 6 Limitations

Our study has the following limitations: (1) Apart from the proposed CAFFEINE, our results can be affected by the quality of commonsense models and the knowledge graph on which they are trained. As future work, we plan to leverage LLM for persona expansion; (2) Our refinement graph stores contradictory personas that are predicted as contradiction with a probability higher than a pre-defined threshold by the NLI model. Our framework may miss personas that actually need a refinement due to the limitation of the NLI model; (3) While we pioneer the commonsense-based persona expansion in multi-session settings, we only consider one speaker’s persona at a time in our refinement framework. Since different people can demonstrate different personality traits and behaviors in the same commonly experienced event (e.g., discussed topic), we acknowledge there can be potential performance gain in response generation if such modeling is included; (4) In this work, we employ LLMs to generate responses based on the dialogue context and retrieved memories (i.e., both speakers’ personas) in a zero-shot setting. However, since the refined personas tend to be longer and contain more information, it is possible that the LLM can not fully utilize the presented personas in its inputs as they get longer (Liu et al., 2023). We

plan to address a better utilization of LLM’s input texts for response generation in future work.

# 7 Ethical Statement

LLMs and COMET can generate sensual, harmful, biased, offensive, or violent content. Authors avoid such content from appearing in the main text, figure, and appendix. We guarantee fair compensation for workers we hire on Amazon Mechanical Turk. We ensure an effective pay rate higher than $\$ 18$ per hour based on the estimated time required to complete the tasks.

# 8 Acknowledgements

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT)(No.2020-0-01361, Artificial Intelligence Graduate School Program (Yonsei University)) and (No.2021-0-02068, Artificial Intelligence Innovation Hub) and (No.2022-0-00077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data). Jinyoung Yeo is a corresponding author.

# References

Daniel Adiwardana, Minh-Thang Luong, David R So, Jamie Hall, Noah Fiedel, Romal Thoppilan, Zi Yang, Apoorv Kulshreshtha, Gaurav Nemade, Yifeng Lu, et al. 2020. Towards a human-like open-domain chatbot. arXiv preprint arXiv:2001.09977.   
Sanghwan Bae, Donghyun Kwak, Soyoung Kang, Min Young Lee, Sungdong Kim, Yuin Jeong, Hyeri Kim, Sang-Woo Lee, Woomyoung Park, and Nako Sung. 2022. Keep me updated! memory management in long-term conversations. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 3769–3787, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.   
Omar Benjelloun, Hector Garcia-Molina, David Menestrina, Qi Su, Steven Euijong Whang, and Jennifer Widom. 2009. Swoosh: a generic approach to entity resolution. The VLDB Journal, 18:255–276.   
Hyungjoo Chae, Yongho Song, Kai Ong, Taeyoon Kwon, Minjin Kim, Youngjae Yu, Dongha Lee, Dongyeop Kang, and Jinyoung Yeo. 2023. Dialogue chain-of-thought distillation for commonsense-aware conversational agents. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5606–5632, Singapore. Association for Computational Linguistics.

Mark Dredze, Paul McNamee, Delip Rao, Adam Gerber, and Tim Finin. 2010. Entity disambiguation for knowledge base population. In Proceedings of the 23rd International Conference on Computational Linguistics (Coling 2010), pages 277–285, Beijing, China. Coling 2010 Organizing Committee.   
Silin Gao, Jena D. Hwang, Saya Kanno, Hiromi Wakaki, Yuki Mitsufuji, and Antoine Bosselut. 2022. Com-Fact: A benchmark for linking contextual commonsense knowledge. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 1656–1675, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.   
Deepanway Ghosal, Siqi Shen, Navonil Majumder, Rada Mihalcea, and Soujanya Poria. 2022. CICERO: A dataset for contextualized commonsense inference in dialogues. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5010–5028, Dublin, Ireland. Association for Computational Linguistics.   
Jena D. Hwang, Chandra Bhagavatula, Ronan Le Bras, Jeff Da, Keisuke Sakaguchi, Antoine Bosselut, and Yejin Choi. 2020. Comet-atomic 2020: On symbolic and neural commonsense knowledge graphs. In AAAI Conference on Artificial Intelligence.   
Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2021. Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118.   
Donghyun Kim, Youbin Ahn, Wongyu Kim, Chanhee Lee, Kyungchan Lee, Kyong-Ho Lee, Jeonguk Kim, Donghoon Shin, and Yeonsoo Lee. 2023. Persona expansion with commonsense knowledge for diverse and consistent response generation. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 1139–1149, Dubrovnik, Croatia. Association for Computational Linguistics.   
Minju Kim, Beong-woo Kwak, Youngwook Kim, Hong-in Lee, Seung-won Hwang, and Jinyoung Yeo. 2022. Dual task framework for improving persona-grounded dialogue dataset. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 10912–10920.   
Siheng Li, Wangjie Jiang, Pengda Si, Cheng Yang, Qiu Yao, Jinchao Zhang, Jie Zhou, and Yujiu Yang. 2023. Enhancing dialogue generation with conversational concept flows. In Findings of the Association for Computational Linguistics: EACL 2023, pages 1514– 1525, Dubrovnik, Croatia. Association for Computational Linguistics.   
Chin-Yew Lin. 2004. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172.   
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.   
Yiting Liu, Liang Li, Beichen Zhang, and Qingming Huang. 2022. Think beyond words: Exploring context-relevant visual commonsense for diverse dialogue generation. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 3106–3117, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.   
Fabrizio Macagno and Sarah Bigi. 2018. Types of dialogue and pragmatic ambiguity. Argumentation and Language—Linguistic, Cognitive and Discursive Explorations, pages 191–218.   
Bodhisattwa Prasad Majumder, Harsh Jhamtani, Taylor Berg-Kirkpatrick, and Julian McAuley. 2020. Like hiking? you probably enjoy nature: Personagrounded dialog with commonsense expansions. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9194–9206, Online. Association for Computational Linguistics.   
OpenAI. 2023. Chatgpt. https://openai.com/blog/ chatgpt.   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730–27744.   
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pages 311–318.   
Kees van Oers, Margreet Klunder, and Piet J Drent. 2005. Context dependence of personalities: risktaking behavior in a social and a nonsocial situation. Behavioral Ecology, 16(4):716–723.   
Sean Welleck, Jason Weston, Arthur Szlam, and Kyunghyun Cho. 2019. Dialogue natural language inference. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3731–3741, Florence, Italy. Association for Computational Linguistics.   
Adina Williams, Nikita Nangia, and Samuel R Bowman. 2017. A broad-coverage challenge corpus for sentence understanding through inference. arXiv preprint arXiv:1704.05426.

Sixing Wu, Ying Li, Ping Xue, Dawei Zhang, and Zhonghai Wu. 2022. Section-aware commonsense knowledge-grounded dialogue generation with pretrained language model. In Proceedings of the 29th International Conference on Computational Linguistics, pages 521–531, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.   
Jing Xu, Arthur Szlam, and Jason Weston. 2022. Beyond goldfish memory: Long-term open-domain conversation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5180–5197, Dublin, Ireland. Association for Computational Linguistics.   
Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. 2018. Personalizing dialogue agents: I have a dog, do you have pets too? In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2204–2213, Melbourne, Australia. Association for Computational Linguistics.   
Pei Zhou, Hyundong Cho, Pegah Jandaghi, Dong-Ho Lee, Bill Yuchen Lin, Jay Pujara, and Xiang Ren. 2022a. Reflect, not reflex: Inference-based common ground improves dialogue response quality. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 10450– 10468, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.   
Pei Zhou, Karthik Gopalakrishnan, Behnam Hedayatnia, Seokhwan Kim, Jay Pujara, Xiang Ren, Yang Liu, and Dilek Hakkani-Tur. 2022b. Think before you speak: Explicitly generating implicit commonsense knowledge for response generation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1237–1252, Dublin, Ireland. Association for Computational Linguistics.

# A Implementation Details

# A.1 Commonsense Expansion with COMET

At the end of each dialogue session, we augment personas derived from the current session with new personas via COMET (Hwang et al., 2020), a widely used commonsense model generating rich and diverse commonsense expansions of a given statement based on cause-effect relation. Among the 23 possible candidate relation types, following prior works on commonsense-based persona expansion (Majumder et al., 2020; Kim et al., 2022), we choose 9 relation types: XATTR, XEFFECT, XINTENT, XNEED, XREACT, XWANT, OEFFECT, OREACT, and OWANT for our expansion, where the prefix ‘x’ indicates an effect or cause on that person and ‘o’ denotes others. After persona expansion via COMET, we leverage an external NLI model to initially filter out improper expansion. Specifically, when a new persona $p ^ { n }$ is generated based on an original persona $p ^ { o }$ (1 original persona yields nine 9 personas), we filter it out if the NLI model predicts the logical relationship between $p ^ { n }$ and $p ^ { o }$ is contradiction with $\delta > 0 . 3 3$ . Note that this is different from the NLI-remove baseline, as here we solely address a one-to-one relationship between a generated persona and its corresponding original persona, while the latter addresses the contradiction among all possible combinations of personas within/across the dialogue session(s). We report the statistics of this initial filtering in Table 3.

Table 3: Initial filtering of improper expansion.   

<table><tr><td>At the End of</td><td>Filtered (%)</td><td>Total</td></tr><tr><td>Session #1</td><td>2830 (6.84 %)</td><td>41391</td></tr><tr><td>Session #2</td><td>2715 (7.39 %)</td><td>36718</td></tr><tr><td>Session #3</td><td>2935 (7.43 %)</td><td>39523</td></tr><tr><td>Session #4</td><td>2971 (7.58 %)</td><td>39198</td></tr></table>

# A.2 Contriever

In our experiments on persona-grounded response generation (RG), we adopt Contriever (Izacard et al., 2021) as the memory retriever to retrieve top- $k$ relevant personas from long-term memory based on the current conversation. Contriever is a dense information retriever trained with unsupervised contrastive learning. Even without supervision, it has shown remarkable capabilities in information retrieval tasks, particularly in demonstrating competitiveness with BM25 in Recall at

100 $( \mathbf { R } @ \mathbf { l } 0 0 )$ on the benchmark for zero-shot retrieval.

# A.3 Large language model

In this work, we employ ChatGPT for the proposed CAFFEINE and response generation. ChatGPT is an LLM with 175B parameters based on Instruct-GPT (Ouyang et al., 2022)7. ChatGPT is trained to follow instructions given by users and return requested information in a conversational manner. We use LangChain8 to send API calls to OpenAI API. The prompt used in CAFFEINE and response generation are in Table 6 and Table 7, respectively.

# A.4 Linking Personas to their Contextual Backgrounds

In the adopted MSC dataset, human annotators summarize information in a speaker’s utterance and use it to derive a persona sentence. As demonstrated in Figure 6, since not every utterance contains enough information to conclude a persona for that speaker, some utterances are not paired with a persona sentence. In our experiment for context-aware persona refinement, we utilize contradictory personas $\begin{array} { r } { \mathcal { P } = \left( p _ { 1 } , p _ { 2 } \right) } \end{array}$ and their contextual backgrounds, i.e., relevant dialogue contexts $\mathcal { D } = ( d _ { 1 } , d _ { 2 } )$ from where they are derived. $d$ consists of $w$ consecutive sentences. In practice, $w$ can differ, as we link each persona with their relevant dialogue context by separating the past conversation into dialogue fragments based on utterances that have corresponding persona sentences. For instance, the $d _ { i }$ for persona $p _ { i }$ will be $d _ { i } = ( u _ { 1 } , u _ { 2 } )$ , and $d _ { i + 1 } = ( u _ { 3 } , u _ { 4 } , \cdot \cdot \cdot , u _ { 6 } )$ for $p _ { i + 1 }$ .

![](images/1dd10d03dc2a9de95463087efe49aff74c48d2ebed496d289f6304ac3ee27eb4.jpg)  
Figure 6: Demonstration of personas and their contextual backgrounds in the MSC dataset.

# A.5 Computational Resources and API Cost

We run Contriever and the NLI model on eight NVIDIA RTX A5000 GPUs. For ChatGPT API usage, we use $\$ 35.52$ on CAFFEINE’s refinement, and $\$ 27.09$ on response generation.

# B Performance in Response Generation

In response generation, top- $k$ relevant persona sentences are retrieved from the long-term memory to assist response generation. In the main text, we report the mode performance in response generation with $k = 2 0$ , the results with $k = 1 2$ and $k = 3 0$ are presented in Table 4.

# C Contradictory Personas in Multi-session Conversations

As human personalities are context-dependent, we display different personalities in different contexts and adapt to new situations. This naturally leads to personas with contradictory literal interpretations to co-exist as one’s persona. Such a phenomenon does not harm human conversations. However, contradictions between personas can lead to inconsistent response generation, hindering user interest in the dialogue systems.

In our study on the Multi-session Chat dataset, we first find that contradictory personas exist in human-authored personas (Figure 7 (a)). Then, we show that expanding existing human-authored personas via commonsence expansion can lead to orders-of-magnitude more contradictory personas that hinder user interest in the conversation (Figure 7 (b)) (Kim et al., 2023).

Personas can contradict other personas from the same sessions (intra-session) and from the previous sessions (inter-session). When comparing COMET-EXP with the human-authored personas (GOLD), we observe that as the number of previous sessions increases, the intra-session contradiction slightly increases, whereas the inter-session contradiction skyrockets significantly. Although such a rising trend appears similarly in GOLD and COMET-EXP, the total count in COMET-EXP is order-ofmagnitude larger. This supports the necessity of CAFFEINE, which refines the contradictory personas in the long-term memory of dialogue models in multi-session settings.

# D CAFFEINE vs. NLI models

Noteworthily, Figure 8 shows that CAFFEINE determines that $6 5 . 4 5 \%$ of contradictory personas

![](images/eb0505ef015cd58f3cc7359f2e70ccf57f8e142297581fed326226b335e63cd3.jpg)  
(a) GOLD

![](images/e78320a207697d1380889f9d1c69f771251d9465dee4675c53cba33d1ec35852.jpg)  
(b) COMET-EXP  
Figure 7: Contradiction among human-authored original personas (GOLD) and expanded personas (COMET-EXP). The blue, red, and green lines represent the intrasession, inter-session, and total contradictory persona pairs, respectively.

(with $\delta \ge 0 . 8 0 )$ ) can be consistent without requiring any refinement when their contextual backgrounds are taken into account, indicating that our contextaware refinement can address the simplification of NLI models where they often solely compare the semantic representation of two statements without reasoning over their contexts. We employ two NLI models: the MNLI model (referred to as ‘NLI model’) and the DNLI model. Results with the DNLI model are presented in Table 5.

![](images/a0417b9e334f84de9ff38b68b48ed31eea84098c7ee195deb13df55d853f4306.jpg)  
Figure 8: Proportion of selected strategies highlights the limitation of the NLI model.

# E Examples of Refinement and Response Generation

We provide examples of response generation in Figure 9 and Figure 10. We can observe that while baselines provide short personas and yield inconsistent or unconstructive responses (green underlines), CAFFEINE offers informative personas (color red) and leads to a response that provides constructive suggestion for Persona A’s Spanish learning (Figure 9) and a response that reflects Person B’s situation and what B is looking for in a car (Figure 10).

Apart from the empirical examples demonstrated in figures in the main text, we have provided more examples for CAFFEINE’s refinement in Table 8, 9, and 10.

# F Details on Human Evaluation.

# F.1 Response Quality

We outsource a human evaluation comparing the generated responses from our setting and those from the baselines via Amazon Mechanical Turk (AMT). We show the interface for the evaluation in Figure 11. We ask the human judges to compare the responses based on the following criteria:

• Naturalness: Which response is more humanlike?   
• Consistency: Which response is more consistent (aligned) with the dialogue context?   
• Specificity: Which response contains more speaker information?   
• Engagingness: Which response is more interesting?

# F.2 Refinement Quality

We outsource a human evaluation comparing the personas before/after CAFFEINE via Amazon Mechanical Turk (AMT). We show the interface for this evaluation in Figure 12.

We ask the human judges to compare the persona before and after refinement based on the following criteria:

• Consistency: Is the refined version less contradictory or more reasonable than before?   
• Specificity: Does the refined version describe a person more specifically?   
• Helpfulness: Can the refined version be more helpful if you are having a conversation with this person?   
• Overall: Overall, do you prefer the refined version?   
• Human-likeness: Is the refinement process (generated rationales) reasonable?

Note that before assessing CAFFEINE’s refinement, we ask workers to determine whether the personas before refinement are actually contradictory from human standpoints. The assessment of the above criteria only begins if the answer is positive. Therefore, the reported human evaluation results are based on 89 out of 100 samples we provided.

Figure 9: Example of Response Generation #1. While baselines yield short personas and inconsistent or unconstructive responses (green underlines), CAFFEINE offers informative personas (color red) and leads to a response that provides constructive suggestion for Person A’s Spanish learning.   

<table><tr><td colspan="3">Dialogue Context</td></tr><tr><td colspan="3">A: Do you know where to get a copy of the The Fugitive movies?B: Sure! if you go to justwatch.com it will tell you where to buy a digital copy or you can go to Walmart.com or amazon. How&#x27;s your Spanish learning going?A: Perfect, thank you! I will look at all those places. I am still learning, and it is slow but I am trying!</td></tr><tr><td colspan="2">Speaker A&#x27;s Personas (draw from retrieved top-20 memories)</td><td>Generated Response</td></tr><tr><td>GOLD</td><td>I am trying to learn Spanish better.I&#x27;m learning Spanish.The Fugitive is my all time favorite movie.My Spanish language progress is not much I&#x27;d expect.I like to speak in Spanish.I like to watch that movie.</td><td>B: That&#x27;s great that you&#x27;re putting in the effort to learn Spanish! The Fugitive is a fantastic movie, by the way. How&#x27;s your progress in Spanish been so far?</td></tr><tr><td>COMET-EXP</td><td>I am trying to learn Spanish better.I&#x27;m learning Spanish.The Fugitive is my all time favorite movie.I want to learn Spanish.I need to learn Spanish.</td><td>B: That&#x27;s great to hear! Learning a new language can take time, but as long as you&#x27;re putting in the effort, you&#x27;ll make progress.Keep it up!</td></tr><tr><td>COMET-EXP+CAFFEINE</td><td>I am trying to learn Spanish better.I am trying to learn Spanish better by watching shows and movies with Spanish subtitles.The Fugitive is my all time favorite movie.I&#x27;m learning Spanish.I&#x27;m learning Spanish by watching movies.I need to read books to learn Spanish faster.I need to go to the library to learn Spanish, and my girlfriend might be interested in learning too.</td><td>B: That&#x27;s great to hear! Learning a new language can take time, but as long as you&#x27;re putting in the effort, you&#x27;ll get there.Have you tried watching any Spanish movies with subtitles to practice?</td></tr></table>

# Dialogue Context

A: I always think used is the way to go; new cars lose so much value the second they leave the lot. I am all for them! If I didn't live in a tiny town with no charging stations I would've definitely gotten an electric car. Were you thinking about getting one?   
B: I totally agree! I thought about getting electric because I've been thinking about the environment and the earth lately. I think I might change my major.   
A: I really think you should do it! You can even get used ones, especially hybrids, I've seen more and more popping up for sale. Oh really, to what?

Figure 10: Example of Response Generation #2. While baselines yield short personas, CAFFEINE offers informative personas (color red) and leads to a response that reflects Person B’s situation and what B is looking for in a car.   

<table><tr><td colspan="2">Speaker B&#x27;s Personas (draw from retrieved top-20 memories)</td><td>Generated Response</td></tr><tr><td>GOLD</td><td>I prefer imported cars.
I want to buy a car.
I am looking for a car.
I cannot afford a hyundai genesis.
I used to have a Ford Edge.</td><td>B: I cannot afford a hybrid at the moment, but maybe I&#x27;ll consider it in the future. I was actually considering a pre-owned luxury car. Maybe something like a used BMW or Mercedes-Benz. What do you think?</td></tr><tr><td>COMET-EXP</td><td>I get a new car.
I want to have a good car.
I want to have a car.
I want to get a car.
I need to have a car.
I prefer imported cars.
I want to buy a car.
I am looking for a car.
I want to buy a new car.</td><td>B: I want to get a new car, preferably an imported one. I&#x27;m looking for something that combines good value and performance. Maybe a new BMW or Mercedes-Benz.</td></tr><tr><td>COMET-EXP +CAFFEINE</td><td>I want to have a car.
As I work on my PhD, I&#x27;m considering buying a used car like a Honda Fit for its affordability and spaciousness.
I want to get a car.
I prefer imported cars.
Since I&#x27;ve been laid off from my job at Cisco, I&#x27;m hoping I can still buy a car.
I cannot afford a hyundai genesis.
I want to buy a car.</td><td>B: I am considering sticking with a used car for now, especially given my recent employment situation. A Honda Fit seems like a practical choice, given its affordability and spaciousness. It might be a sensible option considering my current circumstances.</td></tr></table>

We are surveying qualities for responses from dialogue.

Specificallyyoullivenaeceofaloguecontextandresponsetofollowthdalogue.You'beskedtoomparewich response is better in terms of different aspects,and specify which aspect was most important for judging.

# Guidelines:

1. [Q1~5] First, choose which response is better regarding the given aspect.   
。Try to focus on quality over quantity. Contentful/high-quality response doesn't need to be lengthy.   
2. [Q6] Second, choose which aspect influenced you the most when judging the overall quality.   
。If some factor other thanthe ones in Question 1~6 had the biggest influence,please select "Other" and specify

Dialogue Context

${context}

Response candidate 1

${response_ours}

Response candidate 2

${response_other}

Question 1. Which response is more natural (human-like)?

![](images/89241ce967f9ce2a938778ad8d14815f03dd47284026611feb19f3e60804a065.jpg)

Question 2. Which response is more interesting?

![](images/e9a737af9a1e2192f65b48137b6a9aeb83ddc6da7d714f77f0404f2df7f7b6ce.jpg)

Question 3.Which response is more consistent (well aligned) with the dialogue context?

![](images/a28eaf4cf501636e3ec7031930dbe80628fe709a8ec36298bb9c95d97bbbff73.jpg)

Question 4. Which response is more specific(contains more information about speakers)?

![](images/1fcd740d36daf102e899f52835eb0c32f20447d871814d74ccbc6f952551f074.jpg)

Question 5.Which response do you like more overall?

![](images/b092a53e8bbca9f1bf5aaf772f5bba91bfef37259bd7f036a3a67c3e5ea7cfd1.jpg)  
Figure 11: Interface for human evaluation on response quality.

Question 6. Which aspect afected you the most when judging the overall quality?

O Naturalness

Interesting

Consistency

O Specificity

O Other:

Optional feedback? (expand/collapse)

Submit

We are surveying qualities of personas from dialogues.

Specificallyyu'lleivenersonaandtatarepotentiallyontradictoryitsdaloguecontextanditsfiedpersona.ach personais expanded from Source persona.Refinedpersona is eitherasingle merged persona,or two individuallyspecified personaofPersona1and2.Youllbeasked tocompare whichpersona(s)isbetrin termsof diferentaspects,andassess rationale that was used for refinement.

# Guidelines:

There are four choices: Definitely Disagree/Agree and Slightly Disagree/Agree Please trust yourinstinctsandchooseDefinitelyifyouwouldfelmore confident giving oneresponse,versustheotherone.

1. [Q1] First,answer if Persona 1 and 2 contradict each other.   
2. [Q2-6] Second,assess refined persona in terms of different aspects.

Persona 1 (Before refinement)

${persona_1}

Persona 2 (Before refinement)

${persona_2}

Question 1. Are Persona 1 and Persona 2 contradictory?

Definitely Disagree

Slightly Disagree

● Slightly Agree

Definitely Agree

# Refined Persona

${refined}

Question 2.

(If there is one refined persona) Is refined persona reasonable?

(If thereare two refined personas) Are refined personas less contradictory?

●Definitely Disagree

●Slightly Disagree

● Slightly Agree

●Definitely Agree

Question 3. Is refined persona more specific to decribe a person?

● Definitely Disagree

● Slightly Disagree

● Slightly Agree

●Definitely Agree

Dialog Context of Persona 1

${context_1}

Dialog Context of Persona 2

${context_2}

Question 4. Is refined persona more useful when having a dialogue with this person?

Definitely Disagree

Slightly Disagree

● Slightly Agree

Definitely Agree

Rationale forrefinement

${rationale}

Question 5.Do you think this rationale is appropriate for refining contradiction in Persona 1 and 2?

Definitely Disagree

Slightly Disagree

● Slightly Agree

Definitely Agree

Question 6.Is Refined persona more preferrable overall than personas before refinement?

● Definitely Disagree

Slightly Disagree

● Slightly Agree

Definitely Agree

Optional feedback? (expand/collapse)

Submit

Figure 12: Interface for human evaluation on refinement quality.

Table 4: Performance in response generation. Bold and underline show the best and second-highest in each column.   

<table><tr><td rowspan="2">Memory</td><td colspan="3">Session 2</td><td colspan="3">Session 3</td><td colspan="3">Session 4</td><td colspan="3">Session 5</td></tr><tr><td>B-1</td><td>R-1</td><td>R-L</td><td>B-1</td><td>R-1</td><td>R-L</td><td>B-1</td><td>R-1</td><td>R-L</td><td>B-1</td><td>R-1</td><td>R-L</td></tr><tr><td>None</td><td>20.75</td><td>19.38</td><td>15.16</td><td>20.42</td><td>19.53</td><td>15.09</td><td>19.88</td><td>19.56</td><td>14.98</td><td>19.87</td><td>20.16</td><td>15.33</td></tr><tr><td>k=12</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GOLD</td><td>21.18</td><td>19.78</td><td>15.46</td><td>21.26</td><td>20.11</td><td>15.43</td><td>20.58</td><td>19.97</td><td>15.17</td><td>20.38</td><td>20.40</td><td>15.42</td></tr><tr><td>+ NLI-remove</td><td>20.74</td><td>19.83</td><td>15.19</td><td>21.05</td><td>20.27</td><td>15.51</td><td>21.09</td><td>20.35</td><td>15.62</td><td>21.22</td><td>20.56</td><td>15.78</td></tr><tr><td>+ Caffeine</td><td>20.91</td><td>20.03</td><td>15.33</td><td>21.20</td><td>20.52</td><td>15.74</td><td>21.46</td><td>20.77</td><td>15.94</td><td>21.62</td><td>20.97</td><td>16.11</td></tr><tr><td>COMET-EXP</td><td>21.04</td><td>19.63</td><td>15.32</td><td>20.89</td><td>19.88</td><td>15.27</td><td>20.20</td><td>19.84</td><td>15.14</td><td>20.12</td><td>20.43</td><td>15.50</td></tr><tr><td>+ NLI-remove</td><td>20.68</td><td>19.89</td><td>15.19</td><td>21.04</td><td>20.21</td><td>15.42</td><td>21.49</td><td>20.70</td><td>15.81</td><td>21.57</td><td>20.73</td><td>15.88</td></tr><tr><td>+ Caffeine</td><td>20.99</td><td>20.05</td><td>15.32</td><td>21.41</td><td>20.55</td><td>15.71</td><td>21.66</td><td>20.83</td><td>15.93</td><td>21.86</td><td>20.96</td><td>16.07</td></tr><tr><td>k=20</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GOLD</td><td>21.19</td><td>19.86</td><td>15.50</td><td>21.24</td><td>20.16</td><td>15.47</td><td>20.57</td><td>19.94</td><td>15.16</td><td>20.49</td><td>20.53</td><td>15.55</td></tr><tr><td>+ NLI-remove</td><td>20.81</td><td>19.98</td><td>15.26</td><td>21.04</td><td>20.28</td><td>15.52</td><td>21.33</td><td>20.69</td><td>15.91</td><td>21.43</td><td>20.75</td><td>15.95</td></tr><tr><td>+ Caffeine</td><td>20.93</td><td>20.18</td><td>15.47</td><td>21.41</td><td>20.72</td><td>15.86</td><td>21.67</td><td>21.00</td><td>16.15</td><td>21.92</td><td>21.23</td><td>16.31</td></tr><tr><td>COMET-EXP</td><td>21.23</td><td>19.82</td><td>15.44</td><td>20.95</td><td>19.90</td><td>15.38</td><td>20.33</td><td>20.02</td><td>15.18</td><td>20.00</td><td>20.27</td><td>15.37</td></tr><tr><td>+ NLI-remove</td><td>20.72</td><td>19.96</td><td>15.27</td><td>21.12</td><td>20.40</td><td>15.56</td><td>21.66</td><td>20.77</td><td>15.88</td><td>21.77</td><td>20.91</td><td>16.01</td></tr><tr><td>+ Caffeine</td><td>20.97</td><td>20.06</td><td>15.32</td><td>21.63</td><td>20.73</td><td>15.86</td><td>21.97</td><td>21.10</td><td>16.18</td><td>22.26</td><td>21.32</td><td>16.37</td></tr><tr><td>k=30</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GOLD</td><td>20.88</td><td>19.65</td><td>15.45</td><td>21.09</td><td>20.18</td><td>15.56</td><td>20.50</td><td>19.89</td><td>15.09</td><td>20.41</td><td>20.47</td><td>15.46</td></tr><tr><td>+ NLI-remove</td><td>20.65</td><td>19.85</td><td>15.14</td><td>21.09</td><td>20.43</td><td>15.68</td><td>21.50</td><td>20.83</td><td>15.98</td><td>21.59</td><td>20.93</td><td>16.07</td></tr><tr><td>+ Caffeine</td><td>20.89</td><td>20.13</td><td>15.43</td><td>21.42</td><td>20.77</td><td>15.96</td><td>21.73</td><td>21.07</td><td>16.19</td><td>22.01</td><td>21.29</td><td>16.32</td></tr><tr><td>COMET-EXP</td><td>21.40</td><td>19.89</td><td>15.52</td><td>21.06</td><td>20.10</td><td>15.40</td><td>20.38</td><td>20.03</td><td>15.27</td><td>20.06</td><td>20.50</td><td>15.59</td></tr><tr><td>+ NLI-remove</td><td>20.60</td><td>19.86</td><td>15.20</td><td>21.02</td><td>20.33</td><td>15.57</td><td>21.35</td><td>20.62</td><td>15.83</td><td>21.71</td><td>20.97</td><td>16.05</td></tr><tr><td>+ Caffeine</td><td>20.96</td><td>20.11</td><td>15.37</td><td>21.73</td><td>20.85</td><td>15.99</td><td>22.17</td><td>21.24</td><td>16.27</td><td>22.57</td><td>21.58</td><td>16.54</td></tr></table>

Table 5: Performance in response generation with DNLI model. Bold and underline show the best and secondhighest in each column.   

<table><tr><td rowspan="2">Settings</td><td colspan="3">Session 2</td><td colspan="3">Session 3</td><td colspan="3">Session 4</td><td colspan="3">Session 5</td></tr><tr><td>B-1</td><td>R-1</td><td>R-L</td><td>B-1</td><td>R-1</td><td>R-L</td><td>B-1</td><td>R-1</td><td>R-L</td><td>B-1</td><td>R-1</td><td>R-L</td></tr><tr><td>No Memory</td><td>20.75</td><td>19.38</td><td>15.16</td><td>20.42</td><td>19.53</td><td>15.09</td><td>19.88</td><td>19.56</td><td>14.98</td><td>19.87</td><td>20.16</td><td>15.33</td></tr><tr><td>GOLD</td><td>21.19</td><td>19.86</td><td>15.50</td><td>21.24</td><td>20.16</td><td>15.47</td><td>20.57</td><td>19.94</td><td>15.16</td><td>20.49</td><td>20.53</td><td>15.55</td></tr><tr><td>+ DNLI-remove</td><td>20.87</td><td>20.07</td><td>15.31</td><td>21.15</td><td>20.50</td><td>15.70</td><td>21.37</td><td>20.82</td><td>15.97</td><td>21.52</td><td>20.95</td><td>16.08</td></tr><tr><td>+ DNLI-recent</td><td>20.92</td><td>20.09</td><td>15.36</td><td>21.16</td><td>20.58</td><td>15.80</td><td>21.36</td><td>20.83</td><td>16.01</td><td>21.60</td><td>21.08</td><td>16.20</td></tr><tr><td>+ CaffeINE</td><td>20.94</td><td>20.15</td><td>15.41</td><td>21.33</td><td>20.69</td><td>15.89</td><td>21.54</td><td>21.01</td><td>16.17</td><td>21.75</td><td>21.18</td><td>16.30</td></tr><tr><td>COMET-EXP</td><td>21.23</td><td>19.82</td><td>15.44</td><td>20.95</td><td>19.90</td><td>15.38</td><td>20.33</td><td>20.02</td><td>15.18</td><td>20.00</td><td>20.27</td><td>15.37</td></tr><tr><td>+ DNLI-remove</td><td>20.81</td><td>20.01</td><td>15.26</td><td>21.13</td><td>20.46</td><td>15.73</td><td>21.53</td><td>20.96</td><td>16.12</td><td>21.66</td><td>21.06</td><td>16.18</td></tr><tr><td>+ DNLI-recent</td><td>20.92</td><td>20.10</td><td>15.41</td><td>21.35</td><td>20.69</td><td>15.91</td><td>21.51</td><td>20.93</td><td>16.11</td><td>21.72</td><td>21.20</td><td>16.34</td></tr><tr><td>+ CaffeINE</td><td>20.89</td><td>20.10</td><td>15.40</td><td>21.37</td><td>20.62</td><td>15.81</td><td>21.82</td><td>21.06</td><td>16.19</td><td>22.07</td><td>21.21</td><td>16.30</td></tr></table>

# Prompt

You will be provided with two contradictory persona sentences, along with their source personas and the dialogue fragments from which these persona sentences were derived.

Your task is to resolve the contradiction between the two persona sentences based on the dialogue fragments and the source persona of each contradictory persona. You can use these two strategies:

[Resolution]: If the two personas are based on the same event but change over time (possibly due to a temporal difference or other events in between), adjust and aggregate them into one new persona sentence.

[Disambiguation]: If the contradiction between them isn’t due to time changes or they are derived from unrelated events in the first place, utilize dialogue fragments to rewrite (clarify/specify) each persona.

First, provide a rationale for your choice ([Resolution] or [Disambiguation]). Then, based on this rationale, generate refined persona sentence(s).

If the two personas are not contradictory, generate [NO_CONFLICT].

# Example 1:

Persona 1: I am a programmer.

Dialogue fragment of Persona 1:

A: As a computer programmer, I spend a lot of time writing and debugging code. It’s rewarding to see my work contribute to the development of functional and efficient software.

Source Persona: I am a programmer.

Persona 2: I get fire.

Dialogue fragment of Persona 2:

A: I got fired. It was quite unexpected, and I’m still processing everything.

B: I’m really sorry to hear that. Do you want to talk about what happened?

Source Persona: I don’t have a job right now.

Rationale: There is a temporal connection between the two personas. Persona 1 is about being a programmer,

whereas Persona 2 is about having been fired. Both personas can exist over time with Persona 2 occurring after Persona 1.

[Resolution]: I am a programmer who has recently been fired.

# Example 2:

Persona 1: I feel happy.

Dialogue fragment of Persona 1:

A: My favorite baseball team won this season! I’m so excited!

B: Wow, really? That’s great news!

Source Persona: My team won the league.

Persona 2: I have been feeling quite stressed.

Dialogue fragment of Persona 2:

A: Work has been overwhelming with tight deadlines and high expectations. It’s been hard to find time for relaxation and self-care.

B: With stress piling up, it’s important to find ways to unwind and destress. Maybe I’ll try meditation or yoga to help manage my anxiety.

Source Persona: I work very hard.

Rationale: The two personas do not reflect changes over time but rather different emotional states in response to separate circumstances; one, a moment of happiness due to a favorite team winning, and the other, underlying stress caused by work pressures.

# [Disambiguation]:

- Persona 1: I feel happy when my favorite baseball team wins.

- Persona 2: I am a person dealing with work-related stress and looking for ways to manage anxiety.

# Example 3:

Persona 1: I am a vegetarian.

Dialogue fragment of Persona 1:

A: I’ve been a vegetarian for about 5 years now. It started for health reasons, but now it’s also about sustainability and animal welfare.

Source Persona: I am a vegetarian.

Persona 2: I enjoy reading fiction books.

Dialogue fragments of Persona 2:

A: One of my favorite pastimes is reading fiction. I love getting lost in different worlds and lives through the pages of a good novel.

Source Persona: I enjoy reading fiction books.

Rationale: The two persona sentences do not contradict each other as they pertain to different aspects of the speaker’s identity. One persona is about dietary preference (being a vegetarian), and the other is about a hobby or interest (enjoying reading fiction books). There is no inherent conflict between being a vegetarian and enjoying reading fiction, so the two persona sentences can coexist without the need for resolution or disambiguation.

# [NO_CONFLICT]

Example 4: ...

Table 6: The prompt for CAFFEINE (Five-shot setting, Examples 4 and 5 are omitted in this table). The “preservation” strategy is represented as [NO_CONFLICT] in our prompt.

# Prompt

You will be generating the next turn of a given dialogue context between Speaker A and Speaker B. Alongside the dialogue context, you’ll be given persona statements about both speakers. Your response should be 1-2 sentences, utilizing the persona statements as guidance to create an appropriate reply. Generate appropriate answers using given persona statements as memory.

What is the most appropriate next utterance (3 sentences max)?

Persona Statements of A: {A’s personas within the top- $k$ retrieved personas from long-term memory}

Persona Statements of B: {B’s personas within the top- $k$ retrieved personas from long-term memory}

Dialogue: {dialogue context}

Response: {generation result}

Table 7: The prompt for response generation (zero-shot).

# Example 1 - [Resolution]

# Before refinement:

B’s Persona 1: I feel happy.   
B’s Persona 2: I feel sad.

Contextual Backgrounds (relevant dialogue contexts):

In this example, both personas are derived from the same dialogue context.

A: It’s really bad from the first of the year until tax deadline day but it keeps me pretty busy the rest of the year as well. I will need to think it over. It does get lonely sometimes.   
B: I can only imagine. That was another thing about being a vet that was hard. People expected you to be on call all the time, so I didn’t have a chance for any sort of social life. That’s gotten a lot better since I quit. I’ve found a good group of friends to hang out with at least a few times a week at a local cafe.

# CAFFEINE’s output:

Explanation: The two persona sentences reflect contradictory emotional states. Persona 1 expresses happiness due to having a good group of friends to hang out with, whereas Persona 2 expresses sadness and loneliness from lacking a social life. Both personas are based on the same events and experiences, indicating a change in emotional state over time.

After refinement: I used to feel sad and lonely when I was a vet, but now I feel happy because I have a good group of friends to hang out with at a cafe every week.

# Example 2 - [Disambiguation]

# Before refinement:

B’s Persona 1: I want to play with the dogs.   
B’s Persona 2: I want to eat them.

Contextual Backgrounds (relevant dialogue contexts):

A: Yeah definitely, that’s great to hear you’re enjoying your new vegan diet. Oh, cute! I love labs. Yes, my puppies keep me busy as well. The training phase is definitely a lot of work. How old are your labs now?   
B: Almost 4. Still act like puppies running around like crazy. I think I wouldn’t get any exercise if it weren’t for them though, so that is a good thing for me.

A: That sounds delicious!!! I agree, a recipe exchange would be great. True, vegan cheese does help with that. Have you found any favorite brands for vegan cheese? I found a couple that I like, and some that I didn’t enjoy as much.   
B: I have tried a couple, none that I’ve enjoyed so far. the next on my list to try is cashew cheese, which I have heard is superior, so I am excited to try it. I hear it makes great nachos, and I miss nachos so much!

# CAFFEINE’s output:

Explanation: The two personas do not reflect changes over time but rather different topics of discussion. Persona 1 is about the speaker’s dogs and their age, while Persona 1 is about discussing vegan cheese and favorite brands.

# After refinement:

B’s Persona 1: My 4-year-old labs (labrador) keep me active and it’s a good thing for me.   
B’s Persona 2: I enjoy trying different brands of vegan cheese, although I haven’t found a favorite yet.

Table 8: More examples of persona refinement done by CAFFEINE. Color magenta highlights the contextual cues from the relevant dialogue contexts.

# Example 1 - [Resolution]

# Before refinement:

A’s Persona 1: I feel excited.

A’s Persona 2: I get tired.

Contextual Backgrounds (relevant dialogue contexts):

In this example, both personas are derived from the same dialogue context.

B: I am sure you feel so small and at peace up there. When will your mission make its trip? I imagine training for space is long and strenuous.

A: It absolutely is! The trip is scheduled for around this time next year. I’m going to have to get on a pretty strict diet and exercise routine to make sure that I am fit enough for the trip.

# CAFFEINE’s output:

Explanation: The two persona sentences contradict each other in terms of the emotional state of the speaker. One persona expresses excitement, while the other expresses tiredness. However, both personas are based on the same event or circumstance of training for a space mission. Thus, there is a temporal aspect to this contradiction.

After refinement: I am currently feeling tired due to the strenuous training for my upcoming space mission, but I am also excited about the opportunity.

# Example 2 - [Disambiguation]

# Before refinement:

A’s Persona 1: I feel relaxed.

A’s Persona 2: I feel tired.

Contextual Backgrounds (relevant dialogue contexts):

B: That sounds like so much fun. I have never been fishing in my life. Is it difficult to learn?

A: There’s a bit of a knack to the rod, but once you get that down, it’s pretty simple. It’s a lot of just sitting around and waiting. It’s very peaceful and relaxing.

B: We are always on the lookout for a new restaurant. Have you tried a great place recently?

A: I have been so busy with work, that I haven’t! I was hoping you might have a recommendation. We definitely need to go somewhere with great crab legs, or else a nice sushi joint. Somewhere with delicious cocktails as well!

# CAFFEINE’s output:

Explanation: The two persona sentences do not reflect changes over time but rather different emotional states and interests. Persona 1 expresses feeling happy, while Persona 2 expresses being bored. The contradiction lies in the difference in emotional experiences and preferences.

# After refinement:

A’s Persona 1: I feel relaxed when I go fishing.

A’s Persona 2: I feel tired because I spend a lot of time at work.

Table 9: More examples of persona refinement done by CAFFEINE. Color magenta highlights the contextual cues from the relevant dialogue contexts.

# Example 1 - [Preservation]

# Before refinement:

B’s Persona 1: I love punk music.   
B’s Persona 2: I enjoy romantic comedies and would like to watch some cop shows.

Contextual Backgrounds (relevant dialogue contexts):

A: I am great. Residing in Arizona, skydiving is awesome, Barnes & Noble is my favorite, you?   
B: I like movies over books, love punk music and just love to cruise around!

A: I read the Tori Spelling novel you told me about!   
B: Awesome! What did you think of it?   
A: I thought it was really good. Thanks for recommending it. You also said you watch a lot of movies. Can you recommend a good movie to me too?   
B: Well, I’ve been watching a lot of romantic comedies lately. We should watch a couple the next time we have a movie night! I also wanted to ask, are there any cop shows you’d recommend? I’d like to try watching them too.

# CAFFEINE’s output:

Explanation: The two persona sentences do not contradict each other as they pertain to different areas of interest. Persona 1 expresses a love for punk music, while Persona 2 expresses a liking for romantic comedies and an interest in watching cop shows. There is no inherent conflict between loving punk music and enjoying romantic comedies or cop shows, so the two persona sentences can coexist without the need for resolution or disambiguation.

# Example 2 - [Preservation]

# Before refinement:

B’s Persona 1: I got a new car.   
B’s Persona 2: I like to drink water.

Contextual Backgrounds (relevant dialogue contexts):

A: How was the drive up to San Diego? Did you bring your kids or are you flying solo? I hear the beaches are beautiful there.   
B: The drive was long, but the scenery was gorgeous! Took a lot longer to make the trip up than I expected because with the kids I had to make more stops to stretch our legs, feed them, and give them a little time to burn off some energy. I think I always took for granted the luxury of traveling sans kids in the past, but honestly I don’t mind the company! It was the first time traveling with my kids in my new car. How’s it going for you?

A: I have not been to San Diego, but I have always wanted to go. You will have to tell me all about it.   
B: I will. I will be on the lookout for the best iced tea! Haha. A: You know it is one of my favorites to drink. Besides tea I only really drink water   
B: I love water, too. I wish I could get my kids to drink more water.

# CAFFEINE’s output:

Explanation: The two personas do not contradict each other as they pertain to different aspects of the speaker’s life. Persona 1 is about getting a new car and taking a trip to San Diego, while Persona 2 is about the speaker’s preference for drinking water. The two persona sentences can coexist without the need for resolution or disambiguation.

Table 10: Examples of “preservation” addressing the sub-optimal performance of NLI models that solely rely on the persona sentences without contextual backgrounds. Color magenta highlights the contextual cues.
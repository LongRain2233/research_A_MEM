# Reinforcement Learning for Reasoning in Large Language Models with One Training Example

Yiping Wang1 † ∗ Qing Yang2 Zhiyuan Zeng1 Liliang Ren3 Liyuan Liu3

Baolin Peng3 Hao Cheng3 Xuehai He4 Kuan Wang5 Jianfeng Gao3

Weizhu Chen3 Shuohang Wang3 † Simon Shaolei Du1 † Yelong Shen3 †

1University of Washington 2University of Southern California 3Microsoft 4University of California, Santa Cruz 5Georgia Institute of Technology

# Abstract

We show that reinforcement learning with verifiable reward using one training example (1-shot RLVR) is effective in incentivizing the mathematical reasoning capabilities of large language models (LLMs). Applying RLVR to the base model Qwen2.5-Math-1.5B, we identify a single example that elevates model performance on MATH500 from $3 6 . 0 \%$ to ${ \dot { 7 } } 3 . 6 \%$ $8 . 6 \%$ improvement beyond format correction), and improves the average performance across six common mathematical reasoning benchmarks from $1 7 . 6 \%$ to $3 5 . 7 \%$ ( $7 . 0 \%$ non-format gain). This result matches the performance obtained using the 1.2k DeepScaleR subset (MATH500: $7 3 . 6 \%$ , average: $3 5 . 9 \%$ ), which contains the aforementioned example. Furthermore, RLVR with only two examples even slightly exceeds these results (MATH500: $7 4 . 8 \%$ , average: $3 6 . 6 \%$ ). Similar substantial improvements are observed across various models (Qwen2.5-Math-7B, Llama3.2-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B), RL algorithms (GRPO and PPO), and different math examples. In addition, we identify some interesting phenomena during 1-shot RLVR, including cross-category generalization, increased frequency of self-reflection, and sustained test performance improvement even after the training accuracy has saturated, a phenomenon we term post-saturation generalization. Moreover, we verify that the effectiveness of 1-shot RLVR primarily arises from the policy gradient loss, distinguishing it from the "grokking" phenomenon. We also show the critical role of promoting exploration (e.g., by incorporating entropy loss with an appropriate coefficient) in 1-shot RLVR training. We also further discuss related observations about format correction, label robustness and prompt modification. These findings can inspire future work on RLVR efficiency and encourage a re-examination of recent progress and the underlying mechanisms in RLVR. Our code, models, and data are open source at https://github.com/ypwang61/One-Shot-RLVR.

# 1 Introduction

Recently, significant progress has been achieved in enhancing the reasoning capabilities of large language models (LLMs), including OpenAI-o1 [1], DeepSeek-R1 [2], and Kimi-1.5 [3], particularly for complex mathematical tasks. A key method contributing to these advancements is Reinforcement Learning with Verifiable Reward (RLVR) [4, 5, 2, 3], which commonly employs reinforcement learning on an LLM with a rule-based outcome reward, such as a binary reward indicating the correctness

![](images/6b00a68f4f8eafc7a41efc36cf613a15a7a43e23df3a445ab419b47a4a6de377.jpg)

![](images/9b54ca5787d66ff9b28322ad9f6f6b5af689000107b9ccbfd7c0e706d6bf1788.jpg)  
Figure 1: RLVR with 1 example (green) can perform as well as using datasets with thousands of examples (blue). Left/Right corresponds to MATH500/Average performance on 6 mathematical reasoning benchmarks (MATH500, AIME24, AMC23, Minerva Math, OlympiadBench, and AIME25). Base model is Qwen2.5-Math-1.5B. $\pi _ { 1 }$ and $\pi _ { 1 3 }$ are examples defined by Eqn. 2 and detailed in Tab. 2, and they are from the 1.2k DeepScalerR subset (DSR-sub). Setup details are in Sec. 3.1. We find that RLVR with 1 example $\{ \pi _ { 1 3 } \}$ $( 3 5 . 7 \% )$ performs close to that with 1.2k DSR-sub $( 3 5 . 9 \% )$ , and RLVR with 2 examples $\{ \pi _ { 1 } , \pi _ { 1 3 } \}$ $( 3 6 . 6 \% )$ even performs better than RLVR with DSR-sub and as well as using $7 . 5 \mathrm { k }$ MATH train dataset $( 3 6 . 7 \% )$ . Format reward (gold) (Appendix C.2.3) serves as a baseline for format correction. Detailed results are in Appendix C.1.1. Additional results for non-mathematical reasoning tasks are in Tab. 1.

of the model’s final answer to a math problem. Several intriguing empirical phenomena have been observed in RLVR, such as the stimulation or enhancement of specific cognitive behaviors [6] (e.g., self-reflection) and improved generalization across various downstream tasks [5, 2, 3].

Currently, substantial efforts are directed toward refining RL algorithms (e.g., PPO [7] and GRPO [8]) to further enhance RLVR’s performance and stability [9–16]. Conversely, data-centric aspects of RLVR remain relatively underexplored. Although several studies attempt to curate high-quality mathematical reasoning datasets [17, 18, 11], there is relatively limited exploration into the specific role of data in RLVR. Thus, critical questions remain open: How much data is truly necessary? What data is most effective? How do the quality and quantity of the training data relate to observed empirical phenomena (e.g., self-reflection and robust generalization)? The most relevant study to these problems is LIMR [19], which proposed a metric called learning impact measurement (LIM) to evaluate the effectiveness of training examples. Using the LIM score, they maintain model performance while reducing the number of training examples by sixfold. However, this study does not explore how aggressively the RLVR training dataset can be reduced. Motivated by these considerations, in this paper, we specifically investigate the following research question:

"To what extent can we reduce the training dataset for RLVR while maintaining comparable performance compared to using the full dataset?"

We empirically demonstrate that, surprisingly, the training dataset for RLVR can be reduced to as little as ONE example! This finding supports recent claims that base models already possess significant reasoning capabilities [13, 20, 6, 21], and further shows that a single example is sufficient to substantially enhance the base model’s mathematical performance. We refer to this setup as $^ { l }$ -shot RLVR. We summarize our contributions and findings below:

• We find that selecting one specific example as the training dataset can achieve similar downstream performance to that of the $1 . 2 \hat { \mathrm { k } }$ DeepScaleR subset (DSR-sub) containing that example. Specifically, this improves the Qwen2.5-Math-1.5B model from $3 6 . 0 \%$ to ${ \bar { 7 } } 3 . 6 \%$ on MATH500, and from $1 7 . { \bar { 6 } } \%$ to $3 5 . 7 \%$ on average across 6 mathematical reasoning benchmarks, including non-trivial improvements beyond format correction (Fig. 1). Notably, these two examples are relatively easy for the base model, which can solve them with high probability without any training (Sec. 3.2.1). Additionally, 1-shot RLVR on math examples can improve model performance on non-mathematical reasoning tasks, even outperforming full-set RLVR (Tab. 1).   
• We confirm the effectiveness of 1(few)-shot RLVR across different base models (Qwen2.5- Math-1.5/7B, Llama3.2-3B-Instruct), models distilled from long Chain-of-Thought (CoT) data (DeepSeek-R1-Distill-Qwen-1.5B), and different RL algorithms (GRPO, PPO).   
• We highlight an intriguing phenomenon in 1-shot RLVR: post-saturation generalization. Specifically, the training accuracy on the single example rapidly approaches $100 \%$ , yet the model’s test accuracy continues to improve. Moreover, despite using only one training

example, overfitting does not occur until after approximately $1 . 4 \mathrm { k }$ training steps. Even post-overfitting, while the model’s reasoning outputs for the training example become incomprehensible multilingual gibberish mixed with correct solutions, its test performance remains strong, and the reasoning outputs for the test examples remain human-interpretable.   
• In addition, we demonstrate the following phenomena: (1) 1-shot RLVR is viable for many examples in the full dataset when each example is individually used for training. We also discuss its connection with format correction in Appendix C.2.3. (2) 1-shot RLVR enables cross-category generalization: training on a single example from one category (e.g., Geometry) often enhances performance in other categories (e.g., Algebra, Number Theory). (3) As 1-shot RLVR training progresses, both the response length for the training example and the frequency of self-reflective terms in downstream tasks increase.   
• Through ablation studies, we show that policy gradient loss primarily drives the improvements observed in 1-shot RLVR, distinguishing it from “grokking”, which heavily depends on regularization methods like weight decay. Additionally, we emphasize the importance of promoting diverse exploration in model outputs, showing that adding an entropy loss with an appropriate coefficient further enhances performance.   
• Lastly, we find that employing entropy loss alone, even without any outcome reward, yields a performance boost, although it remains weaker than the format-reward baseline. Similar improvements are observed for Qwen2.5-Math-7B and Llama-3.2-3B-Instruct. We also discuss label robustness and prompt modification in RLVR (Appendix C.2).

# 2 Preliminary

RL Loss Function. In this paper, we adopt GRPO [8, 2] as the RL algorithm for LLMs unless stated otherwise. We briefly introduce three main components in the loss function as below and provide more details in Appendix B.1.

(1) Policy gradient loss: it encourages the model to produce responses with higher rewards, assigning weights according to their group-normalized advantages. Thus, better-thanaverage solutions are reinforced, whereas inferior ones are penalized. Since we focus on mathematical problems, the reward is defined as binary (0-1), where a reward of 1 is granted only when the outcome of the model’s response correctly matches the ground truth. We do not include the format reward when using the outcome reward, but formatreward RLVR is used as a baseline for Qwen models. Further discussion can be found in Appendix C.2.3.

(2) KL loss: it helps to maintain general language quality by measuring the divergence between current model’s responses and those from reference model.

(3) Entropy loss [22]: applied with a negative coefficient, it incentivizes higher per-token entropy to (3) Entropy loss [22]: applied with a negative coefcient, it incentivizes higher per-token entropy to encourage exploration and generate more diverse reasoning paths.We note that entropy loss is not encourage exploration and generate more diverse reasoning paths. We note that entropy loss is not strictly necessary for GRPO training, but it is included by default in verl [22] used in our experiments. strictly necessary for GRPO training,but it is included by default in verl [22] used in our experiments. Its effect on 1-shot RLVR is discussed in Sec. 4.1. Its effect on 1-shot RLVR is discussed in Sec.4.1.

Data Selection: Historical Variance Score. To explore how extensively we can reduce the RLVR training dataset, we propose a simple data selection approach for ranking training examples. We first train the model for $E$ epochs on the full dataset using RLVR. Then for each example $i \in [ N ] = \{ 1 , \ldots , N \}$ , we can obtain a list of historical training accuracy $L _ { i } = [ s _ { i , 1 } , \dotsc , s _ { i , E } ]$ which records its average training accuracy for every epoch. Note that some previous work has shown that the variance of the reward signal [23] is critical for RL training, we simply rank the data by their historical variance of training accuracy, which is directly related to the reward:

$$
v _ {i} := \operatorname {v a r} \left(s _ {i, 1}, \dots , s _ {i, E}\right) \tag {1}
$$

Next, we define a permutation $\pi : [ N ] \to [ N ]$ such that $v _ { \pi ( 1 ) } \geq \cdot \cdot \cdot \geq v _ { \pi ( N ) } .$ . Under this ordering, $\pi ( j )$ (denoted as $\pi _ { j }$ for convenience) corresponds to the example with the $j$ -th largest variance $v _ { i }$ :

$$
\pi_ {j} := \pi (j) = \underset {j} {\arg \operatorname {s o r t}} \left\{v _ {l}: l \in [ N ] \right\} \tag {2}
$$

Table 1: 1-shot RLVR with math examples $\pi _ { 1 } / \pi _ { 1 3 }$ improves model performance on ARC, even better than full-set RLVR. Base model is Qwen2.5-Math-1.5B, evaluation tasks are ARC-Easy (ARC-E) and ARC-Challenge (ARC-C). We select the checkpoints achieving the best average across 6 math benchmarks.

<table><tr><td>Dataset</td><td>Size</td><td>ARC-E</td><td>ARC-C</td></tr><tr><td>Base</td><td>NA</td><td>48.0</td><td>30.2</td></tr><tr><td>MATH</td><td>7500</td><td>51.6</td><td>32.8</td></tr><tr><td>DSR-sub</td><td>1209</td><td>42.2</td><td>29.9</td></tr><tr><td>{π1}</td><td>1</td><td>52.0</td><td>32.2</td></tr><tr><td>{π13}</td><td>1</td><td>55.8</td><td>33.4</td></tr><tr><td>{π1,π13}</td><td>2</td><td>52.1</td><td>32.4</td></tr></table>

We then select examples according to this straightforward ranking criterion. For instance, $\pi _ { 1 }$ , identified by the historical variance score on Qwen2.5-Math-1.5B, performs well in 1-shot RLVR (Sec. 3.2.3, 3.3). We also choose additional examples from diverse categories among $\{ \pi _ { 1 } , . . . , \pi _ { 1 7 } \}$ and evaluate them under 1-shot RLVR (Tab. 3), finding that $\pi _ { 1 3 }$ likewise achieves strong performance. Importantly, we emphasize that this criterion is not necessarily optimal for selecting single examples for 1-shot RLVR2. In fact, Tab. 3 shows that many examples, including those with moderate or low historical variance, can individually produce improvements on MATH500 when used as a single training example in RLVR. This suggests a potentially general phenomenon that is independent of the specific data selection method.

# 3 Experiments

# 3.1 Setup

Models. We by default run our experiments on Qwen2.5-Math-1.5B [24, 25], and also verify the effectiveness of Qwen2.5-Math-7B [25], Llama-3.2-3B-Instruct [26], and DeepSeek-R1-Distill-Qwen-1.5B [2] for 1-shot RLVR in Sec. 3.3. We also include the results of Qwen2.5-1.5B and Qwen2.5-Math-1.5B-Instruct in Appendix C.1.2.

Dataset. Due to resource limitations, we randomly select a subset consisting of 1209 examples from DeepScaleR-Preview-Dataset [18] as our instance pool (“DSR-sub”). For data selection (Sec. 2), as described in Sec. 2, we first train Qwen2.5-Math-1.5B for 500 steps, and then obtain its historical variance score (Eqn. 1) and the corresponding ranking (Eqn. 2) on the examples. To avoid ambiguity, we do not change the correspondence between $\{ \stackrel { \smile } { \pi _ { i } } \} _ { i = 1 } ^ { 1 2 0 9 }$ and examples for all the experiments, i.e., they are all ranked by the historical variance score of Qwen2.5-Math-1.5B. We also use the MATH [27] training set (consisting of 7500 instances) as another dataset in full RLVR to provide a comparison. More details are in Appendix B.2.

Training. As described in Sec. 2, we follow the verl [22] pipeline, and by default, the coefficients for KL divergence and entropy loss are $\beta = 0 . 0 0 1$ and $\alpha = - 0 . 0 0 1$ , respectively. The training rollout temperature is set to 0.6 for vLLM [28]. The training batch size and mini-batch size are 128 3, and we sample 8 responses for each prompt. Therefore, we have 8 gradient updates for each rollout step. By default, the maximum prompt length is 1024, and the maximum response length is 3072, considering that Qwen2.5-Math-1.5B/7B’s context length are 4096. For a fairer comparison on Qwen models, we include the format-reward baseline, which assigns a reward of 1 if and only if the final answer can be parsed from the model output (see Appendix C.2.3 for details). More details are in Appendix B.4.

Evaluation. We use the official Qwen2.5-Math evaluation pipeline [25] for our evaluation. Six widely used complex mathematical reasoning benchmarks are used in our paper: MATH500 [27, 29], AIME 2024 [30], AMC 2023 [31], Minerva Math [32], OlympiadBench [33], and AIME 2025 [30]. We also consider non-mathematical reasoning tasks ARC-Easy and ARC-Challenge [34]. More details about benchmarks are in Appendix B.3. For AIME 2024, AIME 2025, and AMC 2023, which contain only 30 or 40 questions, we repeat the test set 8 times for evaluation stability and evaluate the model with temperature $= 0 . 6$ , and finally report the average pass@1 (avg@8) performance. And for other 3 mathematical benchmarks, we let temperature be 0. The evaluation setup for DeepSeek-R1-Distill-Qwen-1.5B and other evaluation details are provided in Appendix B.5.

# 3.2 Observation of 1/Few-Shot RLVR

In Fig. 1, we have found that RLVR with 1 or 2 examples can perform as well as RLVR with thousands of examples, yielding significant improvements in both format and non-format aspects. Tab. 1 further shows that 1(few)-shot RLVR with these math examples enable better generalization on non-mathematical reasoning tasks (More details are in Appendix C.1). To better understand this phenomenon, we provide a detailed analysis of 1-shot RLVR in this section.

# 3.2.1 Dissection of $\pi _ { 1 }$ : A Not-So-Difficult Problem

Table 2: Example $\pi _ { 1 }$ . It is selected from DSR-sub (Sec. 3.1).

# Prompt of example $\pi _ { 1 }$

The pressure \\( P \\) exerted by wind on a sail varies jointly as the area \\( A \\) of the sail and the cube of the wind’s velocity \\( V \\). When the velocity is \\( 8 \\) miles per hour, the pressure on a sail of \\( 2 \\) square feet is \\( 4 \\) pounds. Find the wind velocity when the pressure on \\( 4 \\) square feet of sail is \\( 32 \\) pounds. Let’s think step by step and output the final answer within \\boxed{}.

Ground truth (label in DSR-sub): 12.8.

First, we inspect the examples that produce such strong results. Tab. 2 lists the instances of $\pi _ { 1 }$ , which is defined by Eqn. 2. We can see that it’s actually an algebra problem with a physics background. The key steps for it are obtaining $k = 1 / 2 5 6$ for formula $\mathbf { \dot { \bar { \mathit { P } } } } = \mathbf { \mathit { k } } \mathbf { \mathit { A } } V ^ { 3 }$ , and calculating $V =$ $( 2 0 4 8 ) ^ { 1 / 3 } \approx 1 2 . 6 9 9$ . Interestingly, we note that base model already almost solves $\pi _ { 1 }$ . In Fig. 3, the base model without any training already solves all the key steps before calculating $( 2 0 4 8 ) ^ { 1 / 3 }$ with high probability4. Just for the last step to calculate the cube root, the model has diverse outputs,√ including 4, 10.95, 12.6992, $8 \sqrt [ 3 ] { 4 }$ , 12.70, 12.8, 13, etc. Specifically, for 128 samplings from the base model, $5 7 . 8 \%$ of outputs are $^ { 6 6 } 1 2 . 7 ^ { 5 }$ or $^ { 6 6 } 1 2 . 7 0 ^ { 9 }$ , $6 . 3 \%$ of outputs are $" 1 2 . 8 "$ , and $\bar { 6 . 3 \% }$ are $^ { 6 6 } 1 3 ^ { , 9 }$ . More examples used in this paper are shown in Appendix E. In Appendix C.2.5, we show that interestingly,√ even though the key step in solving $\pi _ { 1 }$ is computing $\sqrt [ 3 ] { 2 0 4 8 }$ , including only this question in the training example leads to significantly worse performance compared to using full $\pi _ { 1 }$ .

# 3.2.2 Post-saturation Generalization: Generalization After Training Accuracy Saturation

![](images/a49c9aa03caf9984f350692dfe7ca055b7ff82c0659ef00ecbefafa81da64cac.jpg)

![](images/85beb6accde4074e7da8cf01ebbb54e5654e8a2e792fb98b41dc4d7be2714171.jpg)

![](images/4471f1ea1cced09da82f743194214a557c17ffd4606471d49cb797df13f45f8f.jpg)  
Figure 2: Post-saturation generalization in 1-shot RLVR. The training accuracy of RLVR with $\pi _ { 1 } ( \mathrm { L e f t } )$ $\pi _ { 1 }$ and $\pi _ { 1 3 }$ (Middle) saturates before step 100, but their test performance continues improving. On the other hand, the training accuracy for RLVR with 1.2k DSR-sub dataset (Right) still has not saturated after 2000 steps, but there is no significant improvement on test tasks after step 1000.

Then, we show an interesting phenomenon in 1-shot RLVR. As shown in Fig. 2, since we only have one training example, it’s foreseeable that the training accuracy for $\pi _ { 1 }$ and $\pi _ { 1 3 }$ quickly saturates before the 100th step. However, the performance on the test set still continues improving: 1-shot RLVR with $\pi _ { 1 }$ gets $3 . 4 \%$ average improvement from step 100 to step 1540, while using $\pi _ { 1 3 }$ yields a $9 . 9 \%$ average improvement from step 500 to step $2 0 0 0 ^ { 5 }$ . Besides, this phenomenon cannot be observed when using full-set RLVR with DSR-sub currently, as the test performance has started to drop before training accuracy converges.

Moreover, we compare the training and evaluation responses in Fig. 3. Surprisingly, we find that at the final stage of 1-shot RLVR, the model overfits the single training example by mixing the correct calculation process into long unintelligible multilingual outputs in its outputted reasoning. Nonetheless, the test responses still remain normally and maintain high accuracy, indicating that post-saturation generalization still holds even after overfitting the training example. In particular, overfitting in RLVR occurs quite late $\mathbf { \pi } _ { \mathcal { \pi } _ { 1 } }$ after 1400 steps and $\pi _ { 1 3 }$ after 1800 steps). Considering that each example is sampled 1024 times per step, the single training example is not overfitted until after millions of rollouts. Further analysis is provided in Sec. 4.1.

# 3.2.3 1-shot RLVR is Effective for Many Examples & Brings Improvements across Categories

In this section, we investigate whether different data behave differently in 1-shot RL, and whether 1-shot RLVR with one training example from a specific category can help the model better generalize to other categories. We select data with high $( \pi _ { 1 } , \ldots , \pi _ { 1 7 } )$ , medium $( \pi _ { 6 0 5 } , \pi _ { 6 0 6 } )$ , and low

<table><tr><td>Training Response</td><td>Evaluation Response (MATH500)</td></tr><tr><td>Problem: As shown in Table 2
Answer: 12.8</td><td>Problem: Find the roots of $(x-3)^3+(x-7)^3=(2x-10)^3.$
Answer: 3, 5, 7</td></tr><tr><td>Step 0: &quot;Step 1: ...\[\P=\text{kAV^3}\] ... Step 2: ...\[\k=\text{frac}\{4\}{1024}\]=\text{frac}\{1\}{256}\] ... Step 3: ...\[\V=\text{sqrt}\{3\}{2048}\] \]nIn\[\V=12.88\]...final answer is:\nIn\[\boxed{12.88}\]&quot;}</td><td>Step 0: &quot;Find the roots of $(x-3)^3+(x-7)^3=(2x-10)^3.$.
HighlightIn.HighlightIn.HighlightIn.HighlightIn....(no final answer).&quot;</td></tr><tr><td>Step 500: &quot;We start by writing the mathematical relationship for the pressure $\backslash(P)\backslash\backslash\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\PI\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pi\Pl&quot;</td><td>Step 500: &quot;... simplifies to:\nIn\[\n(x-3)^3+(x-7)^3=(2x-10)(\backslash x-3)^2-(x-3)(x-7)+(x-7)^2)\]nInThis equation is true if \(\backslash\{2x-10=0\}\) or if the expression \(\backslash((x-3)^2-(x-3)(x-7)+(x-7)^2=0\).So \(\backslash(x=5)\) is a solution... Since the discriminant is negative, there are no real solutions to this quadratic equation...the final answer is:\nIn\[\n\Box\boxed(12.8)\]}/miles per hour&quot;</td></tr><tr><td>Step 1300: &quot;First, substituting the initial known scenario conditions into the corresponding variant,...\[\P=k\backslashcdot A\backslashcdot V^3\]\...From specific of \(\backslash(A=2\backslash)\), \(\backslash(V=8\backslash)\), \(\backslash(P=4\backslash)\), nInthen required equation will be line up out:\n\[4=\text{k}\backslashcdot 2\backslash\text{cdot} 4\backslash\text{cdot} V^3\]\...[4=4=1024k]{nIn solving can then seek toward \(\backslash(k=\backslash\text{frac}\{4\}\{1024)=\backslash\text{frac}\{1\}\{256\})\]...For a moment press onward investing concrete data to pinpoint...\[\{32=\backslash\text{frac}\{1\}\{256\}\backslash\text{cdot} 4\backslash\text{cdot} V^3\}\]...now simplification further solving thus for final \(\backslash[V^3=2048\}\) then calculate in:\n\[V=\backslash\text{sqrt}\{3\}\{2048\}=12.8\]nInThus, putting already tidings calculated match format needed to seize \(\backslash(V=31)\), and \(\backslash(x=71)\). Therefore, the roots of the equation are:\nIn\[\n\Box\boxed(3,5,7)\nIn\]...&quot;</td><td>Step 1300: &quot;...First, let&#x27;s simplify the equation... \(\backslash[\n{2x-10})(x-3)^2-(x-3)(x-7)+(x-7)^2-(2x-10)^2=0\)nIn\[\n\nThis gives us two cases to consider:\nIn\{12x-10=0\}\nIn\{2x-(3)^2-(x-(3)^2-(2x-(3)^2-(2x-10)^2=0\).So \(\backslash(x=5)\) is a solution.
Now let&#x27;s solve the second case... \(\backslash\{n(x-3)(x-7)=0\}\)implies \(x=\) \(\backslash\{3\times t\text{ext}\}\) or \(\backslash\{x=7\}\)nInSo the solutions to the equation are \(\backslash(x=5)\), \(\backslash(x=3)\), and \(\backslash(x=7)\). Let&#x27;s verify these solutions using Python. &quot;python Infrom sympy&quot;&quot;output[n5, 3, 7]&quot;&quot;nThe solutions to the equation \(\backslash(\{x-3\)^3+(x-7)^3=(2x-10)^3\)&quot; are \(\backslash(x=5)\), \(\backslash(x=3)\), and \(\backslash(x=7)\). Therefore, the roots of the equation are:\nIn\[\n\Box\boxod(3,5,7)\nIn\]...&quot;</td></tr><tr><td>Step 1860: &quot;Lon Pharmacation/computeradecap Played Miglik镀idx DependencyPropertyiciente with unexpected.Function blendDesigner.xlsx %)...nIn kon(Application tileSizewow addUser惑reaa养又那一&quot;lemmpaenéHmp Finds...灿[Metricpuyou],nIn
&quot;[In\begin{aligned}
&amp;\backslash\{n=\text{sqrt}\{3\}\{1024\}=\text{nno 影响性 allocation} \\
&amp;\backslash\{n=\text{sqrt}\{3\}\{1024\}=\text{nno 影响性 allocation} \\
&amp;\backslash\{n=\text{sqrt}\{3\}\{1024\}=\text{nno 影响性 allocation} \\
&amp;\backslash\{n=\text{sqrt}\{3\}\{1024\}=\text{nno 归入} \\
&amp;\backslash\{n=\text{sqrt}\{3\}\{1024\}=\text{nno 影响性 allocation} \\
&amp;\backslash\{n=\text{sqrt}\{3\}\{1024\}=\text{nno 影响性 allocation} \\
&amp;\backslash\{n=\text{sqrt}\{3\}\{1024\}=\text{nNO 影响性 allocation} \\
&amp;\backslash\{n=\text{sqrt}\{3\}\{1024\}=\text{nNO 影响性 allocation} \\
&amp;\backslash\{n=\text{sqrt}\{3\}\{1024\}=\text{nNO 影响性 allocation} \\
&amp;\backslash\{n=\text{sqrt}\{3\}\{1024~}Jumps~s#eep~spock~手势~提名的心~Work~n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#
&quot;nIn-St(SIG(ogroupd frau Jumps#))&quot;;\nIn\[\nBox\boxod(3,5,7)\nIn\]...&quot;</td><td>Step 1860: &quot;Let&#x27;s solve the equation \(\backslash(\{x-3\)^3+(x-7)^3=(2x-10)^3\)&quot; step by step... we need to solve:\nIn\[\n{x^n^3-15x^2+71x-105=0}\nIn\}\We can use the Rational Root Theorem to find the possible rational roots of this polynomial. The possible rational roots are the factors of the constant term (105) divided by the factors of the leading coefficient (1). The factors of 105 are \(\backslash(\{x=5\})\) is a root:\nIn\[\n{5n^5-15n^5-15n^5+15n^5+2+\)71(5)-105=125-375+355-105=0\)...we get:\nIn\[\n{x^n^3-15n^2+71x-105=0}\) and \(\backslash(\{x=5\})\) is a root:\nIn\[\n{5n^5-15n^5-15n^5+2+\)71(5)-105=125-375+355-105=0\)...we get:\nIn\[\n{x^n^3-15n^2+71x-105=(x-5)^n(x^n^2-10x+21)\nIn\}\nIn\[\n{5n^2-10x+21}\nIn\}\nIn\[\n{5n^2-10x+21}\nIn\}\nIn\[\n{5n^2-10x+21}\nIn\}\nIn\[\n{5n^2-10x+21}\nIn\}\nIn\[\n{5n^2-10x+21}\nIn\}\nIn\[\n{5 n^2-10x+21}\nIn\}\nIn\[\n{5n^2-10x+21}\nIn\}\nIn\[\n{5n^2-10x+21}\nIn\}\nIn\[\n{5n^2-10x+21}\nIn\}\nIn\[\n{5n^2-10x+21}\nIN\}\nIn\[\n{5n^2-10x+21}\nIN\}\nIn\[\n{5n^2-10x+21}\nIN\}\nIn\[\n{5n^2-10x+21}\nIN\}\nIn\[\n{5n^2-10x+21}\nIN\}\nIn\[\n{5n^{\prime}}\nIn\}\nIn\[\n{5n^{\prime}}\nIn\}\nIn\[\n{5n^{\prime}}\nIn\}\nIn\[\n{5n^{\prime}}\nIn\}\nIn\[\n{5n^{\prime}}\nIn\}\nIn\[\n{5n^{\prime}}\nIn\}\nIn\[\n{5n^{\prime}} \nIn\}\nIn\[\n{5n^{\prime}}\nIn\}\nIn\[\n{5n^{\prime}}\nIn\}\nIn\[\n{5n^{\prime}}\nIn\}\nIn\[\n{5n^{\prime}}\nIn\}\nIn\[\n{5n^{\prime}}\nIn\}\nIn\[\n{5n^{\mu}}\nIn\}\nIn\[\n{5n^{\mu}}\nIn\}\nIn\[\n{5n^{\mu}}\nIn\}\nIn\[\n{5n^{\mu}}\nIn\}\nIn\[\n{5n^{\mu}}\nIn\}\nIn\[\n{5n^{\mu}}\nIn\}\nIn\[\n{5n} \nIn\]
frivol,inobspock-spock-手势:提名的心Work n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n#n# nIn-St(SIG(ogroupd frau Jumps#))&quot;;\nIn\[\nBox\boxod(3,5,7)\nIn\]...&quot;</td></tr></table>

Figure 3: The model can still generalize on test data after overfitting training example for 1-shot RLVR’s post-saturation generalization. Here we show model’s response to training example $\pi _ { 1 }$ and a selected MATH500 problem. Green/Red are used for marking Correct/Wrong answers. The model converges on $\pi _ { 1 }$ (before step 500) and later attempt to generate longer solutions for $\pi _ { 1 }$ in different styles (step 1300), and gradually performs better on evaluation task. But it significantly overfits training data $\pi _ { 1 }$ at step 1860 (when model achieves $74 \%$ MATH500 accuracy), as it mixes the correct process (cyan) with meaningless output. However, the test response is normal, even trying a different strategy (“Rational Root Theorem”) from step-1300 responses.

$( \pi _ { 1 2 0 1 } , . . . \pi _ { 1 2 0 9 } )$ historical variance (Eqn. 1) and from different topics. We determine the categories of the questions based on their characteristics. We show their detailed MATH500 performance for both overall and subclasses in Tab. 3. More performance curves are in Appendix C.1.

We observe that (1) 1-shot RLVR improves performance across all categories in MATH500. Almost all examples yield a $\cdot \geq 3 0 \%$ improvement over the base model, except for the incorrect example $\pi _ { 1 2 0 7 }$ and the extremely difficult example $\pi _ { 1 2 0 8 }$ , which cause the model to fail to generate any correct solutions. (2) 1-shot RLVR can perform at least as well as the format-reward baseline (except $\pi _ { 1 2 0 7 }$ and $\pi _ { 1 2 0 8 }$ ), and with appropriate examples, 1-shot RLVR with outcome reward can achieve additional non-trivial improvements. From Tab. 3, we observe that the improvements of some examples (e.g., $\pi _ { 7 }$ , $\pi _ { 1 1 }$ , and $\pi _ { 6 0 6 } $ ) mainly come from format correction. However, many other examples (e.g., $\pi _ { 1 }$ , $\pi _ { 1 3 }$ , and $\pi _ { 1 2 0 9 . }$ ) still exhibit non-trivial improvements beyond format fixing. Further discussion is provided in Appendix C.2.3. (3) Counterintuitively, test data belonging to the same category as the single training example does not necessarily exhibit better improvement. For instance, $\pi _ { 1 1 }$ belongs to Number Theory, but RLVR trained with $\pi _ { 1 1 }$ achieves a relatively low Number Theory score compared to using other examples (e.g., $\pi _ { 6 0 5 }$ from Precalculus). This may indicate that the reasoning capability stimulated by an instance cannot be simply predicted by superficial features such as categories [35]. Additional analysis on prompt complexity is provided in Appendix C.2.5.

# 3.2.4 More Frequent Self-Reflection on Test Data

In this section, we show another empirical observation of 1-shot RLVR: it can increase the frequency of self-reflection [6] in the model responses as training progresses. To study this, we check the output patterns of different checkpoints from the RLVR training on Qwen2.5-Math-1.5B. We find

Table 3: 1(Few)-Shot RLVR performance $( \% )$ for different categories in MATH500. Here for MATH500, we consider Algebra (Alg.), Count & Probability (C.P.), Geometry (Geo.), Intermediate Algebra (I. Alg.), Number Theory (N. T.), Prealgebra (Prealg.), Precalculus (Precal.), and MATH500 Average (Avg.). We report the best model performance on MATH500 and AIME24 separately (As illustrated in Appendix. B.5). “Size” means dataset size, and "Step" denotes the checkpoint step that model achieves the best MATH500 performance. Data with red color means the model (almost) never successfully samples the ground truth in training ( $\pi _ { 1 2 0 7 }$ has wrong label and $\pi _ { 1 2 0 8 }$ is too difficult). “Format” denotes the format reward baseline (Appendix C.2.3) for format correction. We further mention related discussions about prompt complexity in Appendix C.2.5.   

<table><tr><td>Dataset</td><td>Size</td><td>Step</td><td>Type</td><td>Alg.</td><td>C. P.</td><td>Geo.</td><td>I. Alg.</td><td>N. T.</td><td>Prealg.</td><td>Precal.</td><td>MATH500</td><td>AIME24</td></tr><tr><td>Base</td><td>0</td><td>0</td><td>NA</td><td>37.1</td><td>31.6</td><td>39.0</td><td>43.3</td><td>24.2</td><td>36.6</td><td>33.9</td><td>36.0</td><td>6.7</td></tr><tr><td>MATH</td><td>7500</td><td>1160</td><td>General</td><td>91.1</td><td>65.8</td><td>63.4</td><td>59.8</td><td>82.3</td><td>81.7</td><td>66.1</td><td>75.4</td><td>20.4</td></tr><tr><td>DSR-sub</td><td>1209</td><td>1160</td><td>General</td><td>91.9</td><td>68.4</td><td>58.5</td><td>57.7</td><td>85.5</td><td>79.3</td><td>67.9</td><td>75.2</td><td>18.8</td></tr><tr><td>Format</td><td>1209</td><td>260</td><td>General</td><td>81.5</td><td>60.5</td><td>53.7</td><td>52.6</td><td>72.6</td><td>68.3</td><td>53.6</td><td>65.6</td><td>10.0</td></tr><tr><td>{π1}</td><td>1</td><td>1860</td><td>Alg.</td><td>88.7</td><td>63.2</td><td>56.1</td><td>62.9</td><td>79.0</td><td>81.7</td><td>64.3</td><td>74.0</td><td>16.7</td></tr><tr><td>{π2}</td><td>1</td><td>220</td><td>N. T.</td><td>83.9</td><td>57.9</td><td>56.1</td><td>55.7</td><td>77.4</td><td>82.9</td><td>60.7</td><td>70.6</td><td>17.1</td></tr><tr><td>{π4}</td><td>1</td><td>80</td><td>N. T.</td><td>79.8</td><td>57.9</td><td>53.7</td><td>51.6</td><td>71.0</td><td>74.4</td><td>53.6</td><td>65.6</td><td>17.1</td></tr><tr><td>{π7}</td><td>1</td><td>580</td><td>I. Alg.</td><td>75.8</td><td>60.5</td><td>51.2</td><td>56.7</td><td>59.7</td><td>70.7</td><td>57.1</td><td>64.0</td><td>12.1</td></tr><tr><td>{π11}</td><td>1</td><td>20</td><td>N. T.</td><td>75.8</td><td>65.8</td><td>56.1</td><td>50.5</td><td>66.1</td><td>73.2</td><td>50.0</td><td>64.0</td><td>13.3</td></tr><tr><td>{π13}</td><td>1</td><td>1940</td><td>Geo.</td><td>89.5</td><td>65.8</td><td>63.4</td><td>55.7</td><td>83.9</td><td>81.7</td><td>66.1</td><td>74.4</td><td>17.1</td></tr><tr><td>{π16}</td><td>1</td><td>600</td><td>Alg.</td><td>86.3</td><td>63.2</td><td>56.1</td><td>51.6</td><td>67.7</td><td>73.2</td><td>51.8</td><td>67.0</td><td>14.6</td></tr><tr><td>{π17}</td><td>1</td><td>220</td><td>C. P.</td><td>80.7</td><td>65.8</td><td>51.2</td><td>58.8</td><td>67.7</td><td>78.1</td><td>48.2</td><td>67.2</td><td>13.3</td></tr><tr><td>{π605}</td><td>1</td><td>1040</td><td>Precal.</td><td>84.7</td><td>63.2</td><td>58.5</td><td>49.5</td><td>82.3</td><td>78.1</td><td>62.5</td><td>71.8</td><td>14.6</td></tr><tr><td>{π606}</td><td>1</td><td>460</td><td>N. T.</td><td>83.9</td><td>63.2</td><td>53.7</td><td>49.5</td><td>58.1</td><td>75.6</td><td>46.4</td><td>64.4</td><td>14.2</td></tr><tr><td>{π1201}</td><td>1</td><td>940</td><td>Geo.</td><td>89.5</td><td>68.4</td><td>58.5</td><td>53.6</td><td>79.0</td><td>73.2</td><td>62.5</td><td>71.4</td><td>16.3</td></tr><tr><td>{π1207}</td><td>1</td><td>100</td><td>Geo.</td><td>67.7</td><td>50.0</td><td>43.9</td><td>41.2</td><td>53.2</td><td>63.4</td><td>42.7</td><td>54.0</td><td>9.6</td></tr><tr><td>{π1208}</td><td>1</td><td>240</td><td>C. P.</td><td>58.1</td><td>55.3</td><td>43.9</td><td>32.0</td><td>40.3</td><td>48.8</td><td>32.1</td><td>45.0</td><td>8.8</td></tr><tr><td>{π1209}</td><td>1</td><td>1140</td><td>Precal.</td><td>86.3</td><td>71.1</td><td>65.9</td><td>55.7</td><td>75.8</td><td>76.8</td><td>64.3</td><td>72.2</td><td>17.5</td></tr><tr><td>{π1...π16}</td><td>16</td><td>1840</td><td>General</td><td>90.3</td><td>63.2</td><td>61.0</td><td>55.7</td><td>69.4</td><td>80.5</td><td>60.7</td><td>71.6</td><td>16.7</td></tr><tr><td>{π1,π2}</td><td>2</td><td>1580</td><td>Alg./N.T.</td><td>89.5</td><td>63.2</td><td>61.0</td><td>60.8</td><td>82.3</td><td>74.4</td><td>58.9</td><td>72.8</td><td>15.0</td></tr><tr><td>{π1,π13}</td><td>2</td><td>2000</td><td>Alg./Geo.</td><td>92.7</td><td>71.1</td><td>58.5</td><td>57.7</td><td>79.0</td><td>84.2</td><td>71.4</td><td>76.0</td><td>17.9</td></tr></table>

![](images/14af64db204f828a0b8c66b8d48fbe62da219c5d6fb8a74fa521790105d8dc8f.jpg)

![](images/d33f24bf96a5842776cafec11768f29f0456e0501e73f0031999c53c2cb2b8d9.jpg)

![](images/87ad3e61d586d49ed68a3d5d485f43c159ddd4ae20960897b41e7866fabcb88e.jpg)  
Figure 4: (Left, Middle) Average response length on training data and entropy loss. After around 1300/1700 steps, the average response length of 1-shot RLVR with $\pi _ { 1 } / \pi _ { 1 3 }$ significantly increases, corresponding to that model tries to solve the single problem with longer CoT reasoning in a more diverse way (Fig. 3, step 1300), which is also confirmed by the increase of entropy loss. These may also indicate the gradual overfitting (Fig. 3, step 1860). (Right) Number of reflection words detected in evaluation tasks. The number of reflection words (“rethink”, “recheck”, and “recalculate”) appearing in evaluation tasks increases in 1-shot RLVR with $\pi _ { 1 } / \pi _ { 1 3 }$ , especially after around 1250 steps, matching the increase of response length. On the other hand, RLVR with DSR-sub contains fewer reflection words as the training progresses.

that their self-reflection process often appears with words “rethink”, “recheck” and “recalculate”. Therefore, we count the number of responses that contain these three words when evaluating 6 mathematical reasoning tasks. The results are in Fig. 4. First, after around 1.3k steps, the response length and entropy loss increase significantly, which may imply the attempt of diverse output patterns or overfitting (Fig. 3). Second, for the evaluation task, the base model itself already exhibits selfreflection processes, which supports the observation in recent works [13, 21]. Third, the number of self-recheck processes increases at the later stages of 1-shot RL training, which again confirms that the model generalizes well on test data and shows more complex reasoning processes even after it

Table 4: 1(few)-shot RLVR is viable for different models and RL algorithm. “Random” denotes the 16 examples randomly sampled from $1 . 2 \mathrm { k }$ DSR-sub. Format reward (Appendix C.2.3) serves as a baseline for format correction. More details are in Appendix C.1, and we also include the results of Qwen2.5-Math-1.5B-Instruct and Qwen2.5-1.5B in Appendix C.1.2.   

<table><tr><td>RL Dataset</td><td>Dataset Size</td><td>MATH 500</td><td>AIME 2024</td><td>AMC 2023</td><td>Minerva Math</td><td>Olympiad-Bench</td><td>AIME 2025</td><td>Avg.</td></tr><tr><td colspan="9">Qwen2.5-Math-7B [24] + GRPO</td></tr><tr><td>NA</td><td>NA</td><td>51.0</td><td>12.1</td><td>35.3</td><td>11.0</td><td>18.2</td><td>6.7</td><td>22.4</td></tr><tr><td>DSR-sub</td><td>1209</td><td>78.6</td><td>25.8</td><td>62.5</td><td>33.8</td><td>41.6</td><td>14.6</td><td>42.8</td></tr><tr><td>Format Reward</td><td>1209</td><td>65.8</td><td>24.2</td><td>54.4</td><td>24.3</td><td>30.4</td><td>6.7</td><td>34.3</td></tr><tr><td>{π1}</td><td>1</td><td>79.2</td><td>23.8</td><td>60.3</td><td>27.9</td><td>39.1</td><td>10.8</td><td>40.2</td></tr><tr><td>{π1, π13}</td><td>2</td><td>79.2</td><td>21.7</td><td>58.8</td><td>35.3</td><td>40.9</td><td>12.1</td><td>41.3</td></tr><tr><td>{π1, π2, π13, π1209}</td><td>4</td><td>78.6</td><td>22.5</td><td>61.9</td><td>36.0</td><td>43.7</td><td>12.1</td><td>42.5</td></tr><tr><td>Random</td><td>16</td><td>76.0</td><td>22.1</td><td>63.1</td><td>31.6</td><td>35.6</td><td>12.9</td><td>40.2</td></tr><tr><td>{π1, ..., π16}</td><td>16</td><td>77.8</td><td>30.4</td><td>62.2</td><td>35.3</td><td>39.9</td><td>9.6</td><td>42.5</td></tr><tr><td colspan="9">Llama-3.2-3B-Instruct [26] + GRPO</td></tr><tr><td>NA</td><td>NA</td><td>40.8</td><td>8.3</td><td>25.3</td><td>15.8</td><td>13.2</td><td>1.7</td><td>17.5</td></tr><tr><td>DSR-sub</td><td>1209</td><td>43.2</td><td>11.2</td><td>27.8</td><td>19.5</td><td>16.4</td><td>0.8</td><td>19.8</td></tr><tr><td>{π1}</td><td>1</td><td>45.8</td><td>7.9</td><td>25.3</td><td>16.5</td><td>17.0</td><td>1.2</td><td>19.0</td></tr><tr><td>{π1, π13}</td><td>2</td><td>49.4</td><td>7.1</td><td>31.6</td><td>18.4</td><td>19.1</td><td>0.4</td><td>21.0</td></tr><tr><td>{π1, π2, π13, π1209}</td><td>4</td><td>46.4</td><td>6.2</td><td>29.1</td><td>21.0</td><td>15.1</td><td>1.2</td><td>19.8</td></tr><tr><td colspan="9">Qwen2.5-Math-1.5B [24] + PPO</td></tr><tr><td>NA</td><td>NA</td><td>36.0</td><td>6.7</td><td>28.1</td><td>8.1</td><td>22.2</td><td>4.6</td><td>17.6</td></tr><tr><td>DSR-sub</td><td>1209</td><td>72.8</td><td>19.2</td><td>48.1</td><td>27.9</td><td>35.0</td><td>9.6</td><td>35.4</td></tr><tr><td>{π1}</td><td>1</td><td>72.4</td><td>11.7</td><td>51.6</td><td>26.8</td><td>33.3</td><td>7.1</td><td>33.8</td></tr><tr><td colspan="9">DeepSeek-R1-Distill-Qwen-1.5B [2] + GRPO (Eval=32k)</td></tr><tr><td>NA</td><td>NA</td><td>82.9</td><td>29.8</td><td>63.2</td><td>26.4</td><td>43.1</td><td>23.9</td><td>44.9</td></tr><tr><td>DSR-sub</td><td>1209</td><td>84.5</td><td>32.7</td><td>70.1</td><td>29.5</td><td>46.9</td><td>27.8</td><td>48.6</td></tr><tr><td>{π1}</td><td>1</td><td>83.9</td><td>31.0</td><td>66.1</td><td>28.3</td><td>44.6</td><td>24.1</td><td>46.3</td></tr><tr><td>{π1, π2, π13, π1209}</td><td>4</td><td>84.8</td><td>32.2</td><td>66.6</td><td>27.7</td><td>45.5</td><td>24.8</td><td>46.9</td></tr><tr><td>{π1, ..., π16}</td><td>16</td><td>84.5</td><td>34.3</td><td>69.0</td><td>30.0</td><td>46.9</td><td>25.2</td><td>48.3</td></tr></table>

overfits the training data. Interestingly, for the $1 . 2 \mathrm { k }$ DeepScaleR subset, the frequency of reflection slightly decreases as the training progresses, matching the decreasing response length.

# 3.3 1/Few-shot RLVR on Other Models/Algorithms

We further investigate whether 1(few)-shot RLVR is feasible for other models and RL algorithms. We consider setup mentioned in Sec. 3.1, and the results are shown in Tab. 4 (Detailed results on each benchmark are in Appendix C.1). We can see (1) for Qwen2.5-Math-7B, 1-shot RLVR with $\pi _ { 1 }$ improves average performance by $1 7 . 8 \%$ $5 . 9 \%$ higher than format-reward baseline), and 4-shot RLVR performs as well as RLVR with DSR-sub. Moreover, $\{ \pi _ { 1 } , \ldots , \pi _ { 1 6 } \}$ performs better than the subset consisting of 16 randomly sampled examples. (2) For Llama-3.2-3B-Instruct, the absolute gain from RLVR is smaller, but 1(few)-shot RLVR still matches or surpasses (e.g., $\{ \pi _ { 1 } , \pi _ { 1 3 } \} )$ ) the performance of full-set RLVR. We also show the instability of the RLVR process on Llama-3.2- 3B-Instruct in Appendix C.1. (3) RLVR with $\pi _ { 1 }$ using PPO also works for Qwen2.5-Math-1.5B with PPO. (4) For DeepSeek-R1-Distill-Qwen-1.5B, the performance gap between few-shot and full-set RLVR is larger. Nevertheless, few-shot RLVE still yield improvement. More results are in Appendix C.

# 4 Analysis

Table 5: Ablation study of loss function and label correctness. Here we use Qwen2.5-Math-1.5B and example $\pi _ { 1 }$ . “+” means the component is added. “Convergence” denotes if the training accuracy saturates (e.g. Fig. 2). “ $- 0 . 0 0 3 ^ { \prime \prime }$ is the coefficient of entropy loss (default -0.001). We report the best model performance on each benchmark separately (Appendix B.3). (1) Rows 1-8: The improvement of 1(few)-shot RLVR is mainly attributed to policy gradient loss, and it can be enhanced by adding entropy loss. (2) Rows 9-10: Simply adding entropy loss alone can still improve MATH500, but still worse than the format reward baseline (Tab. 3, MATH500: 65.6, AIME24: 10.0). (3) Rows 5,11-13: further investigation into how different labels affect test performance.   

<table><tr><td>Row</td><td>Policy Loss</td><td>Weight Decay</td><td>KL Loss</td><td>Entropy Loss</td><td>Label</td><td>Training Convergence</td><td>MATH 500</td><td>AIME 2024</td></tr><tr><td>1</td><td></td><td></td><td></td><td></td><td>12.8</td><td>NO</td><td>39.8</td><td>7.5</td></tr><tr><td>2</td><td>+</td><td></td><td></td><td></td><td>12.8</td><td>YES</td><td>71.8</td><td>15.4</td></tr><tr><td>3</td><td>+</td><td>+</td><td></td><td></td><td>12.8</td><td>YES</td><td>71.4</td><td>16.3</td></tr><tr><td>4</td><td>+</td><td>+</td><td>+</td><td></td><td>12.8</td><td>YES</td><td>70.8</td><td>15.0</td></tr><tr><td>5</td><td>+</td><td>+</td><td>+</td><td>+</td><td>12.8</td><td>YES</td><td>74.8</td><td>17.5</td></tr><tr><td>6</td><td>+</td><td>+</td><td>+</td><td>+, -0.003</td><td>12.8</td><td>YES</td><td>73.6</td><td>15.4</td></tr><tr><td>7</td><td>+</td><td></td><td></td><td>+</td><td>12.8</td><td>YES</td><td>75.6</td><td>17.1</td></tr><tr><td>8</td><td></td><td>+</td><td>+</td><td></td><td>12.8</td><td>NO</td><td>39.0</td><td>10.0</td></tr><tr><td>9</td><td></td><td>+</td><td>+</td><td>+</td><td>12.8</td><td>NO</td><td>65.4</td><td>7.1</td></tr><tr><td>10</td><td></td><td></td><td></td><td>+</td><td>12.8</td><td>NO</td><td>63.4</td><td>8.8</td></tr><tr><td>11</td><td>+</td><td>+</td><td>+</td><td>+</td><td>12.7</td><td>YES</td><td>73.4</td><td>17.9</td></tr><tr><td>12</td><td>+</td><td>+</td><td>+</td><td>+</td><td>4</td><td>YES</td><td>57.0</td><td>9.2</td></tr><tr><td>13</td><td>+</td><td>+</td><td>+</td><td>+</td><td>929725</td><td>NO</td><td>64.4</td><td>9.6</td></tr></table>

In this section, we concentrate on exploring the potential mechanisms that allow RLVR to work with only one or a few examples. We hope the following analyses can provide some insight for future works. Additional experiments and discussions about the format correction (Appendix C.2.3), prompt modification (Appendix C.2.5) and the reasoning capabilities of base models (Appendix D) are included in supplementary materials.

![](images/985da05f53897755e6d6053a43c592eee38c13b48abc21d561a735f48d478ea4.jpg)  
Figure 5: Encouraging exploration can improve postsaturation generalization. $t$ is the temperature parameter for training rollouts.

# 4.1 Ablation Study: Policy Gradient Loss is the Main Contributor, and Entropy Loss Further Improve Post-Saturation Generalization

As discussed in Sec. 3.2.2, 1-shot RLVR shows the property of post-saturation generalization. This phenomenon is similar to “grokking” [36, 37], which shows that neural networks first memorize/overfit the training data but still perform poorly on

the test set, while suddenly improve generalization after many training steps. A natural question is raised: Is the performance gain from 1-shot RLVR related to the “grokking” phenomenon? To answer this question, noting “grokking” is strongly affected by regularization [36, 38–41] like weight decay, we conduct an ablation study by removing or changing the components of the loss function one by one to see how each of them contributes to the improvement.

The results are shown in Tab. 5 (Test curves are in Appendix C.2.1). We see that if we only add policy gradient loss (Row 2) with $\pi _ { 1 }$ , we already get results close to that of the full loss training (Row 5). In addition, further adding weight decay (Row 3) and KL divergence loss (Row 4) has no significant impact on model performance, while adding entropy loss (Row 5) can further bring $4 . 0 \%$ improvement for MATH500 and $2 . 5 \%$ for AIME24. Here we need to be careful about the weight of the entropy loss, as a too large coefficient (Row 6) might make the training more unstable. These observations support that the feasibility of 1(few)-shot RLVR is mainly attributed to policy gradient loss, rather than weight decay, distinguishing it from “grokking”, which should be significantly affected by weight decay. To double check this, we show that only adding weight decay and KL divergence (Row 8) has little influence on model performance, while using only policy gradient loss and entropy loss (Row 7) behaves almost the same as the full GRPO loss.

Moreover, we also argue that encouraging greater diversity in model outputs—for instance, adding proper entropy loss — can enhance post-saturation generalization in 1-shot RLVR. As shown in Fig. 5, without entropy loss, model performance under 1-shot RLVR shows limited improvement beyond step 150, coinciding with the point at which training accuracy saturates (Fig. 2, Left). By adding entropy loss, the model achieves an average improvement of $2 . 3 \%$ , and further increasing

the temperature to $t = 1 . 0$ yields an additional $0 . 8 \%$ gain. More discussions about entropy loss and post-saturation generalization are in Appendix C.2.2.

# 4.2 Entropy-Loss-Only Training & Label Correctness

In Tab. 3, we find that when using $\pi _ { 1 2 0 7 }$ and $\pi _ { 1 2 0 8 }$ , it is difficult for model to output the ground truth label and receive rewards during 1-shot RLVR training, resulting in a very sparse policy gradient signal. Nevertheless, they still outperform the base model, although their performance remains lower than that of the format-reward baseline. To investigate this, we remove the policy loss from the full GRPO loss (Tab. 5, Row 9) or even retain only the entropy loss (Row 10), and again observe similar improvement. Furthermore, this phenomenon also happens on Qwen2.5-Math-7B and Llama-3.2-3B-Instruct, although only improve at the first several steps. These results implies entropy loss may independently contribute to performance gains from format correction, which, although much smaller than those from policy loss, are still nontrivial.

Moreover, we conduct an experiment by altering the label to (1) the correct one (“12.7,” Row 11), (2) an incorrect one that model can still overfit (“4,” Row 12), and (3) an incorrect one

that the model can neither guess nor overfit (“9292725,” Row 13). We compare them with (4) the original label (“12.8,” Row 5). Interestingly, we find the performance rankings are $( 1 ) \approx ( 4 )$ $> ( 3 ) \bar { > } ( 2 )$ . This suggests that slight inaccuracies in the label do not significantly impair 1-shot RLVR performance. However, if the incorrect label deviates substantially while remaining guessable and overfittable, the resulting performance can be even worse than using a completely incorrect and unguessable label, which behaves similarly to training with entropy loss alone (Row 10). In Appendix C.2.4, we also discuss label robustness on full-set RLVR by showing that if too many data in the dataset are assigned random wrong labels, full-set RLVR can perform worse than 1-shot RLVR.

Table 6: Training with only entropy loss using $\pi _ { 1 }$ can partially improve base model performance, but still perform worse than format-reward baseline. Details are in Tab. 13.   

<table><tr><td>Model</td><td>M500</td><td>Avg.</td></tr><tr><td>Qwen2.5-Math-1.5B</td><td>36.0</td><td>17.6</td></tr><tr><td>+Entropy Loss, 20 steps</td><td>63.4</td><td>25.0</td></tr><tr><td>Format Reward</td><td>65.0</td><td>28.7</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>40.8</td><td>17.5</td></tr><tr><td>+Entropy Loss, 10 steps</td><td>47.8</td><td>19.5</td></tr><tr><td>Qwen2.5-Math-7B</td><td>51.0</td><td>22.4</td></tr><tr><td>+Entropy Loss, 4 steps</td><td>57.2</td><td>25.0</td></tr><tr><td>Format Reward</td><td>65.8</td><td>34.3</td></tr></table>

# 5 Conclusion

In this work, we show that 1-shot RLVR is sufficient to trigger substantial improvements in reasoning tasks, even matching the performance of RLVR with thousands of examples. The empirical results reveal not only improved task performance but also additional observations such as post-saturation generalization, cross-category generalization, more frequent self-reflection and also additional analysis. These findings suggest that the reasoning capability of the model is already buried in some base models, and encouraging exploration on a very small amount of data is capable of generating useful RL training signals for igniting these LLM’s reasoning capability. It also demonstrates the anti-overfitting property of the RLVR algorithm with zero-mean advantage, as we can train on a single example millions of times without performance degradation. Our work also emphasizes the importance of better selection and collection of data for RLVR. We discuss directions for future work in Appendix D.4, and also discuss limitations in Appendix D.1.

# 6 Acknoledgements

We thank Lifan Yuan, Hamish Ivison, Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Pang Wei Koh, Kaixuan Huang, Mickel Liu, Jacqueline He, Noah Smith, Jiachen T. Wang, Yifang Chen, and Weijia Shi for very constructive discussions. YW and ZZ acknowledge the support of Amazon AI Ph.D. Fellowship. SSD acknowledges the support of NSF IIS-2110170, NSF DMS-2134106, NSF CCF-2212261, NSF IIS-2143493, NSF CCF-2019844, NSF IIS-2229881, and the Sloan Research Fellowship.

# References

[1] OpenAI. Learning to reason with llms. https://openai.com/index/ learning-to-reason-with-llms/, 2024. Accessed: 2025-04-10.

[2] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
[3] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.   
[4] Jiaxuan Gao, Shusheng Xu, Wenjie Ye, Weilin Liu, Chuyi He, Wei Fu, Zhiyu Mei, Guangju Wang, and Yi Wu. On designing effective rl reward at training time for llm reasoning. arXiv preprint arXiv:2410.15115, 2024.   
[5] Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V. Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, Yuling Gu, Saumya Malik, Victoria Graf, Jena D. Hwang, Jiangjiang Yang, Ronan Le Bras, Oyvind Tafjord, Chris Wilhelm, Luca Soldaini, Noah A. Smith, Yizhong Wang, Pradeep Dasigi, and Hannaneh Hajishirzi. Tülu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124, 2024.   
[6] Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D Goodman. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective stars. arXiv preprint arXiv:2503.01307, 2025.   
[7] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.   
[8] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
[9] Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. Vineppo: Unlocking rl potential for llm reasoning through refined credit assignment. arXiv preprint arXiv:2410.01679, 2024.   
[10] Yufeng Yuan, Yu Yue, Ruofei Zhu, Tiantian Fan, and Lin Yan. What’s behind ppo’s collapse in long-cot? value optimization holds the secret. arXiv preprint arXiv:2503.01491, 2025.   
[11] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.   
[12] Yufeng Yuan, Qiying Yu, Xiaochen Zuo, Ruofei Zhu, Wenyuan Xu, Jiaze Chen, Chengyi Wang, TianTian Fan, Zhengyin Du, Xiangpeng Wei, et al. Vapo: Efficient and reliable reinforcement learning for advanced reasoning tasks. arXiv preprint arXiv:2504.05118, 2025.   
[13] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.   
[14] Michael Luo, Sijun Tan, Roy Huang, Xiaoxiang Shi, Rachel Xin, Colin Cai, Ameen Patel, Alpay Ariyak, Qingyang Wu, Ce Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepcoder: A fully open-source 14b coder at o3-mini level. https://pretty-radio-b75.notion.site/ DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51, 2025. Notion Blog.   
[15] Jian Hu. Reinforce++: A simple and efficient approach for aligning large language models. arXiv preprint arXiv:2501.03262, 2025.   
[16] Xiaojiang Zhang, Jinghui Wang, Zifei Cheng, Wenhao Zhuang, Zheng Lin, Minglei Zhang, Shaojie Wang, Yinghan Cui, Chao Wang, Junyi Peng, Shimiao Jiang, Shiqi Kuang, Shouyu Yin, Chaohang Wen, Haotian Zhang, Bin Chen, and Bing Yu. Srpo: A cross-domain implementation of large-scale reinforcement learning on llm, 2025.

[17] Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. Numinamath. [https://huggingface.co/AI-MO/NuminaMath-CoT](https://github.com/ project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf), 2024.   
[18] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/ DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca3030 2025. Notion Blog.   
[19] Xuefeng Li, Haoyang Zou, and Pengfei Liu. Limr: Less is more for rl scaling, 2025.   
[20] Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? arXiv preprint arXiv:2504.13837, 2025. Submitted on April 18, 2025.   
[21] Darsh J Shah, Peter Rushton, Somanshu Singla, Mohit Parmar, Kurt Smith, Yash Vanjani, Ashish Vaswani, Adarsh Chaluvaraju, Andrew Hojel, Andrew Ma, et al. Rethinking reflection in pre-training. arXiv preprint arXiv:2504.04022, 2025.   
[22] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.   
[23] Noam Razin, Zixuan Wang, Hubert Strauss, Stanley Wei, Jason D Lee, and Sanjeev Arora. What makes a reward model a good teacher? an optimization perspective. arXiv preprint arXiv:2503.15477, 2025.   
[24] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.   
[25] An Yang, Beichen Zhang, Binyuan Hui, Bofei Gao, Bowen Yu, Chengpeng Li, Dayiheng Liu, Jianhong Tu, Jingren Zhou, Junyang Lin, et al. Qwen2. 5-math technical report: Toward mathematical expert model via self-improvement. arXiv preprint arXiv:2409.12122, 2024.   
[26] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.   
[27] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.   
[28] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles, 2023.   
[29] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.   
[30] Art of Problem Solving. Aime problems and solutions. https://artofproblemsolving. com/wiki/index.php/AIME_Problems_and_Solutions. Accessed: 2025-04-20.   
[31] Art of Problem Solving. Amc problems and solutions. https://artofproblemsolving. com/wiki/index.php?title=AMC_Problems_and_Solutions. Accessed: 2025-04-20.

[32] Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, et al. Solving quantitative reasoning problems with language models. Advances in Neural Information Processing Systems, 35:3843–3857, 2022.   
[33] Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008, 2024.   
[34] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018.   
[35] Zhiyuan Zeng, Yizhong Wang, Hannaneh Hajishirzi, and Pang Wei Koh. Evaltree: Profiling language model weaknesses via hierarchical capability trees. arXiv preprint arXiv:2503.08893, 2025.   
[36] Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra. Grokking: Generalization beyond overfitting on small algorithmic datasets. arXiv preprint arXiv:2201.02177, 2022.   
[37] Simin Fan, Razvan Pascanu, and Martin Jaggi. Deep grokking: Would deep neural networks generalize better? arXiv preprint arXiv:2405.19454, 2024.   
[38] Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. arXiv preprint arXiv:2301.05217, 2023.   
[39] Ziming Liu, Ouail Kitouni, Niklas S Nolte, Eric Michaud, Max Tegmark, and Mike Williams. Towards understanding grokking: An effective theory of representation learning. Advances in Neural Information Processing Systems, 35:34651–34663, 2022.   
[40] Branton DeMoss, Silvia Sapora, Jakob Foerster, Nick Hawes, and Ingmar Posner. The complexity dynamics of grokking. arXiv preprint arXiv:2412.09810, 2024.   
[41] Lucas Prieto, Melih Barsbey, Pedro AM Mediano, and Tolga Birdal. Grokking at the edge of numerical stability. arXiv preprint arXiv:2501.04697, 2025.   
[42] Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerl-zoo: Investigating and taming zero reinforcement learning for open base models in the wild. arXiv preprint arXiv:2503.18892, 2025.   
[43] Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, et al. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. arXiv preprint arXiv:2503.10460, 2025.   
[44] Mingyang Song, Mao Zheng, Zheng Li, Wenjie Yang, Xuan Luo, Yue Pan, and Feng Zhang. Fastcurl: Curriculum reinforcement learning with progressive context extension for efficient training r1-like reasoning models. arXiv preprint arXiv:2503.17287, 2025.   
[45] Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. arXiv preprint arXiv:2505.03335, 2025.   
[46] Qingyang Zhang, Haitao Wu, Changqing Zhang, Peilin Zhao, and Yatao Bian. Right question is already half the answer: Fully unsupervised llm reasoning incentivization. arXiv preprint arXiv:2504.05812, 2025.   
[47] Yuxin Zuo, Kaiyan Zhang, Shang Qu, Li Sheng, Xuekai Zhu, Biqing Qi, Youbang Sun, Ganqu Cui, Ning Ding, and Bowen Zhou. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.   
[48] Hamish Ivison, Muru Zhang, Faeze Brahman, Pang Wei Koh, and Pradeep Dasigi. Large-scale data selection for instruction tuning. arXiv preprint arXiv:2503.01807, 2025.

[49] Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, and Hongxia Jin. Alpagasus: Training a better alpaca with fewer data. In International Conference on Learning Representations, 2024.   
[50] Hamish Ivison, Noah A. Smith, Hannaneh Hajishirzi, and Pradeep Dasigi. Data-efficient finetuning using cross-task nearest neighbors. In Findings of the Association for Computational Linguistics, 2023.   
[51] Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. LESS: selecting influential data for targeted instruction tuning. In International Conference on Machine Learning, 2024.   
[52] William Muldrew, Peter Hayes, Mingtian Zhang, and David Barber. Active preference learning for large language models. In International Conference on Machine Learning, 2024.   
[53] Zijun Liu, Boqun Kou, Peng Li, Ming Yan, Ji Zhang, Fei Huang, and Yang Liu. Enabling weak llms to judge response reliability via meta ranking. arXiv preprint arXiv:2402.12146, 2024.   
[54] Nirjhar Das, Souradip Chakraborty, Aldo Pacchiano, and Sayak Ray Chowdhury. Active preference optimization for sample efficient rlhf. arXiv preprint arXiv:2402.10500, 2024.   
[55] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 2022.   
[56] Mehdi Fatemi, Banafsheh Rafiee, Mingjie Tang, and Kartik Talamadupula. Concise reasoning via reinforcement learning. arXiv preprint arXiv:2504.05185, 2025.   
[57] J. Schulman. Approximating kl divergence. http://joschu.net/blog/kl-approx.html, 2020. 2025.   
[58] Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, et al. Omni-math: A universal olympiad level mathematic benchmark for large language models. arXiv preprint arXiv:2410.07985, 2024.   
[59] Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang, Xiaoxue Cheng, Huatong Song, et al. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. arXiv preprint arXiv:2412.09413, 2024.   
[60] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reasoner series. https://capricious-hydrogen-41c.notion.site/ Skywork-Open-Reaonser-Series-1d0bc9ae823a80459b46c149e4f51680, 2025. Notion Blog.   
[61] Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et al. A survey on llm-as-a-judge. arXiv preprint arXiv:2411.15594, 2024.   
[62] Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025.   
[63] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Tianyu Liu, et al. A survey on in-context learning. arXiv preprint arXiv:2301.00234, 2022.   
[64] David Rolnick, Andreas Veit, Serge Belongie, and Nir Shavit. Deep learning is robust to massive label noise. arXiv preprint arXiv:1705.10694, 2017.   
[65] Preetum Nakkiran, Gal Kaplun, Yamini Bansal, Tristan Yang, Boaz Barak, and Ilya Sutskever. Deep double descent: Where bigger models and more data hurt. Journal of Statistical Mechanics: Theory and Experiment, 2021(12):124003, 2021.   
[66] Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang. On large-batch training for deep learning: Generalization gap and sharp minima. arXiv preprint arXiv: 1609.04836, 2016.

[67] Samuel L. Smith, Benoit Dherin, David G. T. Barrett, and Soham De. On the origin of implicit regularization in stochastic gradient descent. Iclr, 2021.   
[68] Zihan Liu, Yang Chen, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acemath: Advancing frontier math reasoning with post-training and reward modeling. arXiv preprint, 2024.   
[69] Ziniu Li, Congliang Chen, Tian Xu, Zeyu Qin, Jiancong Xiao, Ruoyu Sun, and Zhi-Quan Luo. Entropic distribution matching for supervised fine-tuning of llms: Less overfitting and better diversity. In NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability, 2024.

# Contents

1 Introduction 1   
2 Preliminary 3   
3 Experiments 4

3.1 Setup 4   
3.2 Observation of 1/Few-Shot RLVR 4

3.2.1 Dissection of $\pi _ { 1 }$ : A Not-So-Difficult Problem . 4   
3.2.2 Post-saturation Generalization: Generalization After Training Accuracy Saturation 5   
3.2.3 1-shot RLVR is Effective for Many Examples & Brings Improvements across Categories 5   
3.2.4 More Frequent Self-Reflection on Test Data . . . . 6

3.3 1/Few-shot RLVR on Other Models/Algorithms 8

4 Analysis 8

4.1 Ablation Study: Policy Gradient Loss is the Main Contributor, and Entropy Loss Further Improve Post-Saturation Generalization 9   
4.2 Entropy-Loss-Only Training & Label Correctness 10

5 Conclusion 10   
6 Acknoledgements 10

A Related Work 17   
B Experiment Setup 17

B.1 Details of Loss Function 17   
B.2 Training Dataset . . 18   
B.3 Evaulation Dataset 19   
B.4 More Training Details . 19   
B.5 More Evaluation Details 19   
B.6 Performance Difference on Initial Model . 20

C Evaluation Result 25

C.1 Main Experiments . 25

C.1.1 Detailed performance on Qwen2.5-Math-1.5B. . 25   
C.1.2 Detailed Performance on More Models and Training Examples. . . . 25   
C.1.3 Detailed performance with best per-benchmark results . . 25   
C.1.4 Detailed Test curves on MATH500 for 1-shot RLVR on Qwen2.5-Math-1.5B. 26   
C.1.5 Detailed RLVR results on eacn benchmark over training process. . . . 26   
C.1.6 More Evaluation on DeepSeek-R1-Distill-Qwen-1.5B 26

C.2 Analysis . 26

C.2.1 Test Curves for Ablation Study . . 26   
C.2.2 Entropy loss 26   
C.2.3 (Only) Format Correction? . . 27   
C.2.4 Influence of Random Wrong Labels . . 31   
C.2.5 Change the Prompt of $\pi _ { 1 }$ . . 31

C.3 Response Length 32   
C.4 Pass@8 Results 32

# D Discussions 32

D.1 Limitations of Our Work 32   
D.2 Reasoning Capability of Base Models 33   
D.3 Why Model Continues Improving After the Training Accuracy Reaches Near $100 \%$ ? 33   
D.4 Future Works 33

# E Example Details 34

# A Related Work

Reinforcement Learning with Verifiable Reward (RLVR). RLVR, where the reward is computed by a rule-based verification function, has been shown to be effective in improving the reasoning capabilities of LLMs. The most common practice of RLVR when applying reinforcement learning to LLMs on mathematical reasoning datasets is to use answer matching: the reward function outputs a binary signal based on if the model’s answer matches the gold reference answer [4, 5, 2, 3, 42–44]. This reward design avoids the need for outcome-based or process-based reward models, offering a simple yet effective approach. The success of RLVR is also supported by advancements in RL algorithms, including value function optimization or detail optimization in PPO [7] (e.g., VinePPO [9], VC-PPO [10], VAPO [12]), stabilization and acceleration of GRPO [2] (e.g., DAPO [11], Dr. GRPO [13], GRPO+[14], SRPO [16]), and integration of various components (e.g., REINFORCE++[15]). There are also some recent works that focus on RLVR with minimal human supervision (without using labeled data or even problems), such as Absolute-Zero [45], EMPO [46], and TTRL [47].

Data Selection for LLM Post-Training. The problem of data selection for LLM post-training has been extensively studied in prior work [48], with most efforts focusing on data selection for supervised fine-tuning (instruction tuning). These approaches include LLM-based quality assessment [49], leveraging features from model computation [50], gradient-based selection [51], and more. Another line of work [52–54] explores data selection for human preference data in Reinforcement Learning from Human Feedback (RLHF) [55]. Data selection for RLVR remains relatively unexplored. One attempt is LIMR [19], which selects $1 . 4 \mathrm { k }$ examples from an $8 . 5 \mathrm { k }$ full set for RLVR to match performance; however, unlike our work, they do not push the limits of training set size to the extreme case of just a single example. Another closely related concurrent work [56] shows that RLVR using PPO with only 4 examples can already yield very significant improvements; however, they do not systematically explore this observation, nor do they demonstrate that such an extremely small training set can actually match the performance of using the full dataset.

# B Experiment Setup

# B.1 Details of Loss Function

As said in the main paper, we contain three components in the GRPO loss function following verl [22] pipeline: policy gradient loss, KL divergence, and entropy loss. Details are as follows. For each question $q$ sampled from the Question set ${ \bf \bar { \it P } } ( Q )$ , GRPO samples a group of outputs $\left\{ o _ { 1 } , o _ { 2 } , \ldots , o _ { G } \right\}$ from the old policy model $\pi _ { \theta _ { \mathrm { o l d } } }$ , and then optimizes the policy model $\pi _ { \theta }$ by minimizing the following

loss function:

$$
\mathcal {L} _ {\mathrm {G R P O}} (\theta) = \mathbb {E} _ {\substack {q \sim P (Q) \\ \left\{o _ {i} \right\} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\mathrm {o l d}}} (O | q)}} \left[ \mathcal {L} _ {\mathrm {P G - G R P O}} ^ {\prime} (\cdot , \theta) + \beta \mathcal {L} _ {\mathrm {K L}} ^ {\prime} (\cdot , \theta , \theta_ {\mathrm {r e f}}) + \alpha \mathcal {L} _ {\text {Entropy}} ^ {\prime} (\cdot , \theta) \right], \tag{3}
$$

where $\beta$ and $\alpha$ are hyper-parameters (in general $\beta > 0$ , $\alpha < 0$ ), and “·” is the abbreviation of sampled prompt-responses: $\dot { \{ q , \{ o _ { i } \} }  _ { i = 1 } ^ { G } \}$ . The policy gradient loss and KL divergence loss are:

$$
\mathcal {L} _ {\mathrm {P G - G R P O}} ^ {\prime} \left(q, \left\{o _ {i} \right\} _ {i = 1} ^ {G}, \theta\right) = - \frac {1}{G} \sum_ {i = 1} ^ {G} \left(\min  \left(\frac {\pi_ {\theta} \left(o _ {i} \mid q\right)}{\pi_ {\theta_ {\mathrm {o l d}}} \left(o _ {i} \mid q\right)} A _ {i}, \operatorname {c l i p} \left(\frac {\pi_ {\theta} \left(o _ {i} \mid q\right)}{\pi_ {\theta_ {\mathrm {o l d}}} \left(o _ {i} \mid q\right)}, 1 - \varepsilon , 1 + \varepsilon\right) A _ {i}\right)\right) \tag {4}
$$

$$
\mathcal {L} _ {\mathrm {K L}} ^ {\prime} \left(q, \left\{o _ {i} \right\} _ {i = 1} ^ {G}, \theta , \theta_ {\text {r e f}}\right) = \mathbb {D} _ {\mathrm {K L}} \left(\pi_ {\theta} \| \pi_ {\theta_ {\text {r e f}}}\right) = \frac {\pi_ {\theta_ {\text {r e f}}} \left(o _ {i} \mid q\right)}{\pi_ {\theta} \left(o _ {i} \mid q\right)} - \log \frac {\pi_ {\theta_ {\text {r e f}}} \left(o _ {i} \mid q\right)}{\pi_ {\theta} \left(o _ {i} \mid q\right)} - 1, \tag {5}
$$

Here $\theta _ { \mathrm { r e f } }$ is the reference model, $\varepsilon$ is a hyper-parameter of clipping threshold. Notably, we use the approximation formulation of KL divergence [57], which is widely used in previous works [8, 2]. Besides, $A _ { i }$ is the group-normalized advantage defined below.

$$
A _ {i} = \frac {r _ {i} - \operatorname* {m e a n} \left(\left\{r _ {1} , r _ {2} , \dots , r _ {G} \right\}\right)}{\operatorname* {s t d} \left(\left\{r _ {1} , r _ {2} , \dots , r _ {G} \right\}\right)}. \quad i \in [ G ] \tag {6}
$$

Since we focus on math questions, we let the reward $r _ { i }$ be the 0-1 accuracy score, and $r _ { i }$ is 1 if and only if the response $\mathcal { L } _ { \mathrm { E n t r o p y } } ^ { \prime }$ calculates the average per-token entropy of the responses, and its coefficient $o _ { i }$ gets the correct answer to the question $q$ . What’s more, the entropy loss $\alpha < 0$ implies the encouragement of more diverse responses.

The details of entropy loss are as follows. For each query $q$ and set of outputs $\{ o _ { i } \} _ { i = 1 } ^ { G }$ , the model produces logits $X$ that determine the policy distribution $\pi _ { \theta }$ . These logits $X$ are the direct computational link between inputs $q$ and outputs o - specifically, the model processes $q$ to generate logits $X$ , which after softmax normalization give the probabilities used to sample each token in the outputs $o$ . The entropy loss is formally defined below.

$$
\mathcal {L} _ {\text {E n t r o p y}} ^ {\prime} \left(q, \left\{o _ {i} \right\} _ {i = 1} ^ {G}, \theta\right) = \frac {\sum_ {b , s} M _ {b , s} \cdot H _ {b , s} (X)}{\sum_ {b , s} M _ {b , s}} \tag {7}
$$

Here $M _ { b , s }$ represents the response mask indicating which tokens contribute to the loss calculation (excluding padding and irrelevant tokens), with $b$ indexing the batch dimension and $s$ indexing the sequence position. The entropy $H _ { b , s } ( X )$ is computed from the model’s logits $X$ :

$$
H _ {b, s} (X) = \log \left(\sum_ {v} e ^ {X _ {b, s, v}}\right) - \sum_ {v} p _ {b, s, v} \cdot X _ {b, s, v} \tag {8}
$$

where $v$ indexes over the vocabulary tokens (i.e., the possible output tokens from the model’s vocabulary), and the probability distribution is given by $\begin{array} { r } { p _ { b , s , v } = \mathrm { s o f t m a x } ( X _ { b , s } ) _ { v } = \frac { e ^ { X _ { b , s , v } } } { \sum _ { v ^ { \prime } } e ^ { X _ { b , s , v ^ { \prime } } } } . } \end{array}$ P v ′ e X b,s,v ′ .

# B.2 Training Dataset

DeepScaleR-sub. DeepScaleR-Preview- Dataset [18] consists of approximately 40,000 unique mathematics problem-answer pairs from AIME (1984-2023), AMC (pre-2023), and other sources including Omni-MATH [58] and Still [59]. The data processing pipeline includes extracting answers using Gemini-1.5-Pro-002, removing duplicate problems through RAG with Sentence-Transformers embeddings, and filtering out questions that cannot be evaluated using SymPy to maintain a clean training set. We randomly select a subset that contains 1,209 examples referred to as "DSR-sub".

MATH. Introduced in [27], this dataset contains 12,500 challenging competition mathematics problems designed to measure advanced problem-solving capabilities in machine learning models. Unlike standard mathematical collections, MATH features complex problems from high school mathematics competitions spanning subjects including Prealgebra, Algebra, Number Theory, Counting and Probability, Geometry, Intermediate Algebra, and Precalculus, with each problem assigned a difficulty level from 1 to 5 and accompanied by detailed step-by-step solutions. It’s partitioned into a training subset comprising 7,500 problems $( 6 0 \% )$ and a test subset containing 5,000 problems $(40 \% )$ .

# B.3 Evaulation Dataset

All evaluation sets are drawn from the Qwen2.5-Math evaluation repository6, with the exception of AIME20257. We summarize their details as follows:

MATH500. MATH500, developed by OpenAI [29], comprises a carefully curated selection of 500 problems extracted exclusively from the test partition $( \mathrm { n } { = } 5 , 0 0 0 )$ of the MATH benchmark [27]. It is smaller, more focused, and designed for efficient evaluation.

AIME 2024/2025. The AIME 2024 and 2025 datasets are specialized benchmark collections, each consisting of 30 problems from the 2024 and 2025 American Invitational Mathematics Examination (AIME) I and II, respectively [30].

AMC 2023. AMC 2023 dataset consists of 40 problems, selected from two challenging mathematics competitions (AMC 12A and 12B) for students grades 12 and under across the United States [31]. These AMC 12 evaluates problem-solving abilities in secondary school mathematics, covering topics such as arithmetic, algebra, combinatorics, geometry, number theory, and probability, with all problems solvable without calculus.

Minerva Math. Implicitly introduced in the paper "Solving Quantitative Reasoning Problems with Language Models" [32] as OCWCourses, Minerva Math consists of 272 undergraduate-level STEM problems harvested from MIT’s OpenCourseWare, specifically designed to evaluate multistep scientific reasoning capabilities in language models. Problems were carefully curated from courses including solid-state chemistry, information and entropy, differential equations, and special relativity, with each problem modified to be self-contained with clearly-delineated answers that are automatically verifiable through either numeric (191 problems) or symbolic solutions (81 problems).

OlympiadBench. OlympiadBench [33]is a large-scale, bilingual, and multimodal benchmark designed to evaluate advanced mathematical and physical reasoning in AI systems. It contains 8,476 Olympiad-level problems, sourced from competitions and national exams, with expert-annotated step-by-step solutions. The subset we use for evaluation consists of 675 open-ended text-only math competition problems in English.

We also consider other non-mathematical reasoning tasks: ARC-Challenge and ARC-Easy [34].

ARC-Challenge/Easy. The ARC-Challenge benchmark represents a subset of 2,590 demanding science examination questions drawn from the broader ARC (AI2 Reasoning Challenge) [34] collection, specifically selected because traditional information retrieval and word co-occurrence methods fail to solve them correctly. This challenging evaluation benchmark features exclusively text-based, English-language multiple-choice questions (typically with four possible answers) spanning diverse grade levels, designed to assess science reasoning capabilities rather than simple pattern matching or information retrieval. The complementary ARC-Easy [34] subset contains 5197 questions solvable through simpler approaches. We use $1 . 1 7 \mathbf { k }$ test split for ARC-Challenge evaluation and $2 . 3 8 \mathrm { k }$ test split for ARC-Easy evaluation, respectively.

# B.4 More Training Details

For DeepSeek-R1-Distill-Qwen-1.5B, we let the maximum response length be 8192, following the setup of stage 1 in DeepScaleR [18]. The learning rate is set to 1e-6. The coefficient of weight decay is set to 0.01 by default. We store the model checkpoint every 20 steps for evaluation, and use 8 A100 GPUs for each experiment. For Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Llama-3.2-3B-Instruct, and DeepSeek-R1-Distill-Qwen-1.5B, we train for 2000, 1000, 1000, and 1200 steps, respectively, unless the model has already shown a significant drop in performance. We use the same approach as DeepScaleR [18] (whose repository is also derived from the verl) to save the model in safetensor format to facilitate evaluation.

# B.5 More Evaluation Details

In evaluation, the maximum number of generated tokens is set to be 3072 by default. For Qwenbased models, we use the “qwen25-math-cot” prompt template in evaluation. For Llama and

Table 7: Difference between model downloaded from Hugging Face and initial checkpoint saved by verl/deepscaler pipeline. Since the performance of stored initial checkpoint has some randomness, we still use the original downloaded model for recording initial performance.   

<table><tr><td>Model</td><td>MATH
500</td><td>AIME24
2024</td><td>AMC23
2023</td><td>Minerva
Math</td><td>Olympiad-
Bench</td><td>AIME
2025</td><td>Avg.</td></tr><tr><td colspan="8">Qwen2.5-Math-1.5B [24]</td></tr><tr><td>Hugging Face Model</td><td>36.0</td><td>6.7</td><td>28.1</td><td>8.1</td><td>22.2</td><td>4.6</td><td>17.6</td></tr><tr><td>Stored Initial Checkpoint</td><td>39.6</td><td>8.8</td><td>34.7</td><td>8.5</td><td>22.7</td><td>3.3</td><td>19.6</td></tr><tr><td colspan="8">Qwen2.5-Math-7B [24]</td></tr><tr><td>Hugging Face Model</td><td>51.0</td><td>12.1</td><td>35.3</td><td>11.0</td><td>18.2</td><td>6.7</td><td>22.4</td></tr><tr><td>Stored Initial Checkpoint</td><td>52.0</td><td>14.6</td><td>36.6</td><td>12.1</td><td>18.1</td><td>4.2</td><td>22.9</td></tr><tr><td colspan="8">Llama-3.2-3B-Instruct [26]</td></tr><tr><td>Hugging Face Model</td><td>40.8</td><td>8.3</td><td>25.3</td><td>15.8</td><td>13.2</td><td>1.7</td><td>17.5</td></tr><tr><td>Stored Initial Checkpoint</td><td>41.0</td><td>7.1</td><td>28.4</td><td>16.9</td><td>13.0</td><td>0.0</td><td>17.7</td></tr></table>

distilled models, we use their original chat templates. We set the evaluation seed to 0 and top_p to 1 by default. For evaluation on DeepSeek-R1-Distill-Qwen-1.5B, following DeepSeek-R1 [2] and DeepScaleR [18], we set the temperature to 0.6 and top_p to 0.95, and use avg@16 for MATH500, Minerva Math, and OlympiadBench, and avg@64 for AIME24, AIME25, and AMC23. Since our training length is 8192, we provide results for both 8192 (8k) and 32768 (32k) evaluation lengths (Appendix C.1.6). By default, we report the performance of the checkpoint that obtains the best average performance on 6 benchmarks. But in Sec. 3.2.3 and Sec. 4.1, since we only evaluate MATH500 and AIME2024, we report the best model performance on each benchmark separately, i.e., the best MATH500 checkpoint and best AIME2024 checkpoint can be different (This will not influence our results, as in Tab. 9 and Tab. 11, we still obtain similar conclusions as in main paper.) We use 4 GPUs for the evaluation. Finally we mention that there are slightly performance difference on initial model caused by numerical precision, but it does not influence our conclusions (Appendix B.6).

# B.6 Performance Difference on Initial Model

We mention that there is a precision inconsistency between models downloaded from Hugging Face repositories and initial checkpoints saved by the verl/deepscaler reinforcement learning pipeline in Tab. 7. This discrepancy arises from the verl/DeepScaleR pipeline saving checkpoints with float32 precision, whereas the original base models from Hugging Face utilize bfloat16 precision.

The root cause appears to be in the model initialization process within the verl framework. The fsdp_workers.py 8 file in the verl codebase reveals that models are deliberately created in float32 precision during initialization, as noted in the code comment: "note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect". This design choice was likely made to ensure optimizer stability during training. When examining the checkpoint saving process, the precision setting from initialization appears to be preserved, resulting in saved checkpoints retaining float32 precision rather than the original bfloat16 precision of the base model.

Our empirical investigation demonstrates that modifying the torch_dtype parameter in the saved config.json file to match the base model’s precision (specifically, changing from float32 to bfloat16) successfully resolves the observed numerical inconsistency. Related issues are documented in the community9, and we adopt the default settings of the verl pipeline in our experiments.

Table 8: Detailed 1/2-shot RLVR performance for Qwen2.5-Math-1.5B. Results are reported for the checkpoint achieving the best average across 6 math benchmarks (Fig. 1). Models’ best individual benchmark results are listed in Tab. 9. Format reward (Appendix C.2.3) serves as a baseline for format correction.   

<table><tr><td>RL Dataset/Method</td><td>Dataset Size</td><td>MATH 500</td><td>AIME 2024</td><td>AMC 2023</td><td>Minerva Math</td><td>Olympiad-Bench</td><td>AIME 2025</td><td>Avg.</td></tr><tr><td>NA</td><td>NA</td><td>36.0</td><td>6.7</td><td>28.1</td><td>8.1</td><td>22.2</td><td>4.6</td><td>17.6</td></tr><tr><td>MATH</td><td>7500</td><td>74.4</td><td>20.0</td><td>54.1</td><td>29.0</td><td>34.1</td><td>8.3</td><td>36.7</td></tr><tr><td>DSR-sub</td><td>1209</td><td>73.6</td><td>17.1</td><td>50.6</td><td>32.4</td><td>33.6</td><td>8.3</td><td>35.9</td></tr><tr><td>Format Reward</td><td>1209</td><td>65.0</td><td>8.3</td><td>45.9</td><td>17.6</td><td>29.9</td><td>5.4</td><td>28.7</td></tr><tr><td>{π1}</td><td>1</td><td>72.8</td><td>15.4</td><td>51.6</td><td>29.8</td><td>33.5</td><td>7.1</td><td>35.0</td></tr><tr><td>{π13}</td><td>1</td><td>73.6</td><td>16.7</td><td>53.8</td><td>23.5</td><td>35.7</td><td>10.8</td><td>35.7</td></tr><tr><td>{π1, π13}</td><td>2</td><td>74.8</td><td>17.5</td><td>53.1</td><td>29.4</td><td>36.7</td><td>7.9</td><td>36.6</td></tr></table>

Table 9: Detailed 1/2/4-shot RLVR performance for Qwen2.5-Math-1.5B. Here we record model’s best performance on each benchmark independently. “Best Avg. Step” denotes the checkpoint step that model achieves the best average performance (Tab. 8).   

<table><tr><td>RL Dataset</td><td>Dataset Size</td><td>MATH 500</td><td>AIME 2024</td><td>AMC 2023</td><td>Minerva Math</td><td>Olympiad-Bench</td><td>AIME 2025</td><td>Avg.</td><td>Best Avg. Step</td></tr><tr><td>NA</td><td>NA</td><td>36.0</td><td>6.7</td><td>28.1</td><td>8.1</td><td>22.2</td><td>4.6</td><td>17.6</td><td>0</td></tr><tr><td>MATH</td><td>7500</td><td>75.4</td><td>20.4</td><td>54.7</td><td>29.8</td><td>37.3</td><td>10.8</td><td>36.7</td><td>2000</td></tr><tr><td>DSR-sub</td><td>1209</td><td>75.2</td><td>18.8</td><td>52.5</td><td>34.9</td><td>35.1</td><td>11.3</td><td>35.9</td><td>1560</td></tr><tr><td>{π1}</td><td>1</td><td>74.0</td><td>16.7</td><td>54.4</td><td>30.2</td><td>35.3</td><td>9.2</td><td>35.0</td><td>1540</td></tr><tr><td>{π2}</td><td>1</td><td>70.6</td><td>17.1</td><td>52.8</td><td>28.7</td><td>34.2</td><td>7.9</td><td>33.5</td><td>320</td></tr><tr><td>{π13}</td><td>1</td><td>74.4</td><td>17.1</td><td>53.8</td><td>25.4</td><td>36.7</td><td>10.8</td><td>35.7</td><td>2000</td></tr><tr><td>{π1201}</td><td>1</td><td>71.4</td><td>16.3</td><td>54.4</td><td>25.4</td><td>36.2</td><td>10.0</td><td>33.7</td><td>1120</td></tr><tr><td>{π1209}</td><td>1</td><td>72.2</td><td>17.5</td><td>50.9</td><td>27.6</td><td>34.2</td><td>8.8</td><td>33.5</td><td>1220</td></tr><tr><td>{π1, π13}</td><td>2</td><td>76.0</td><td>17.9</td><td>54.1</td><td>30.9</td><td>37.2</td><td>10.8</td><td>36.6</td><td>1980</td></tr><tr><td>{π1, π2, π13, π1209}</td><td>4</td><td>74.4</td><td>16.3</td><td>56.3</td><td>32.4</td><td>37.0</td><td>11.3</td><td>36.0</td><td>1880</td></tr></table>

Table 10: Results of more models (base and instruct versions) and more training examples (on Qwen2.5-Math-7B). We record results from checkpoints achieving best average performance. Test curves are in Fig. 10 and Fig. 11. Analysis is in Appendix C.1.2. We can see that on Qwen2.5-Math-7B, different examples have different performance for 1-shot RLVR.   

<table><tr><td>RL Dataset</td><td>Dataset Size</td><td>MATH 500</td><td>AIME 2024</td><td>AMC 2023</td><td>Minerva Math</td><td>Olympiad-Bench</td><td>AIME 2025</td><td>Avg.</td></tr><tr><td colspan="9">Qwen2.5-1.5B [24]</td></tr><tr><td>NA</td><td>NA</td><td>3.2</td><td>0.4</td><td>3.1</td><td>2.6</td><td>1.2</td><td>1.7</td><td>2.0</td></tr><tr><td>DSR-sub</td><td>1209</td><td>57.2</td><td>5.0</td><td>30.3</td><td>17.6</td><td>21.2</td><td>0.8</td><td>22.0</td></tr><tr><td>{π1}</td><td>1</td><td>43.6</td><td>0.8</td><td>14.4</td><td>12.9</td><td>17.6</td><td>0.4</td><td>15.0</td></tr><tr><td>{π1, π2, π13, π1209}</td><td>4</td><td>46.4</td><td>2.9</td><td>15.9</td><td>14.0</td><td>19.0</td><td>0.8</td><td>16.5</td></tr><tr><td>{π1, ..., π16}</td><td>16</td><td>53.0</td><td>3.8</td><td>30.3</td><td>19.1</td><td>19.6</td><td>0.0</td><td>21.0</td></tr><tr><td colspan="9">Qwen2.5-Math-1.5B-Instruct [25]</td></tr><tr><td>NA</td><td>NA</td><td>73.4</td><td>10.8</td><td>55.0</td><td>29.0</td><td>38.5</td><td>6.7</td><td>35.6</td></tr><tr><td>DSR-sub</td><td>1209</td><td>75.6</td><td>13.3</td><td>57.2</td><td>31.2</td><td>39.6</td><td>12.1</td><td>38.2</td></tr><tr><td>{π1}</td><td>1</td><td>74.6</td><td>12.1</td><td>55.3</td><td>30.9</td><td>37.9</td><td>12.1</td><td>37.1</td></tr><tr><td colspan="9">Qwen2.5-Math-7B [25]</td></tr><tr><td>NA</td><td>NA</td><td>51.0</td><td>12.1</td><td>35.3</td><td>11.0</td><td>18.2</td><td>6.7</td><td>22.4</td></tr><tr><td>DSR-sub</td><td>1209</td><td>78.6</td><td>25.8</td><td>62.5</td><td>33.8</td><td>41.6</td><td>14.6</td><td>42.8</td></tr><tr><td>{π1}</td><td>1</td><td>79.2</td><td>23.8</td><td>60.3</td><td>27.9</td><td>39.1</td><td>10.8</td><td>40.2</td></tr><tr><td>{π605}</td><td>1</td><td>77.4</td><td>20.4</td><td>59.4</td><td>23.9</td><td>39.0</td><td>10.8</td><td>38.5</td></tr><tr><td>{π1209}</td><td>1</td><td>76.4</td><td>16.2</td><td>55.0</td><td>30.9</td><td>41.0</td><td>5.4</td><td>37.5</td></tr><tr><td>{π1, ..., π16}</td><td>16</td><td>77.8</td><td>30.4</td><td>62.2</td><td>35.3</td><td>39.9</td><td>9.6</td><td>42.5</td></tr></table>

Table 11: 1(few)-shot RL still works well for different model with different scales. Here we record model’s best performance on each benchmark independently.   

<table><tr><td>RL Dataset</td><td>Dataset Size</td><td>MATH 500</td><td>AIME 2024</td><td>AMC 2023</td><td>Minerva Math</td><td>Olympiad-Bench</td><td>AIME 2025</td><td>Avg.</td></tr><tr><td colspan="9">Qwen2.5-Math-7B [24] + GRPO</td></tr><tr><td>NA</td><td>NA</td><td>51.0</td><td>12.1</td><td>35.3</td><td>11.0</td><td>18.2</td><td>6.7</td><td>22.4</td></tr><tr><td>DSR-sub</td><td>1209</td><td>81.0</td><td>34.6</td><td>64.6</td><td>39.7</td><td>42.2</td><td>14.6</td><td>42.8</td></tr><tr><td>{π1}</td><td>1</td><td>79.4</td><td>27.1</td><td>61.9</td><td>32.7</td><td>40.3</td><td>11.7</td><td>40.2</td></tr><tr><td>{π1, π13}</td><td>1</td><td>81.2</td><td>23.3</td><td>64.1</td><td>36.0</td><td>42.2</td><td>12.1</td><td>41.3</td></tr><tr><td>{π1, π2, π13, π1209}</td><td>4</td><td>80.0</td><td>26.2</td><td>64.4</td><td>37.9</td><td>43.7</td><td>14.6</td><td>42.5</td></tr><tr><td>Random</td><td>16</td><td>78.0</td><td>24.6</td><td>63.1</td><td>36.8</td><td>38.7</td><td>14.2</td><td>40.2</td></tr><tr><td>{π1, ..., π16}</td><td>16</td><td>79.2</td><td>30.4</td><td>62.2</td><td>37.9</td><td>42.4</td><td>11.7</td><td>42.5</td></tr><tr><td colspan="9">Llama-3.2-3B-Instruct [26] + GRPO</td></tr><tr><td>NA</td><td>NA</td><td>40.8</td><td>8.3</td><td>25.3</td><td>15.8</td><td>13.2</td><td>1.7</td><td>17.5</td></tr><tr><td>DSR-sub</td><td>1209</td><td>45.4</td><td>11.7</td><td>30.9</td><td>21.7</td><td>16.6</td><td>11.7</td><td>19.8</td></tr><tr><td>{π1}</td><td>1</td><td>46.4</td><td>8.3</td><td>27.5</td><td>19.5</td><td>18.2</td><td>1.7</td><td>19.0</td></tr><tr><td>{π1, π13}</td><td>2</td><td>49.4</td><td>9.2</td><td>31.6</td><td>20.6</td><td>20.0</td><td>2.1</td><td>21.0</td></tr><tr><td>{π1, π2, π13, π1209}</td><td>4</td><td>48.4</td><td>9.2</td><td>29.4</td><td>23.5</td><td>17.6</td><td>1.7</td><td>19.8</td></tr><tr><td colspan="9">Qwen2.5-Math-1.5B [24] + PPO</td></tr><tr><td>NA</td><td>NA</td><td>36.0</td><td>6.7</td><td>28.1</td><td>8.1</td><td>22.2</td><td>4.6</td><td>17.6</td></tr><tr><td>DSR-sub</td><td>1209</td><td>73.8</td><td>21.2</td><td>52.8</td><td>32.4</td><td>36.3</td><td>10.4</td><td>35.4</td></tr><tr><td>{π1}</td><td>1</td><td>74.0</td><td>16.7</td><td>53.8</td><td>28.3</td><td>34.1</td><td>9.2</td><td>33.8</td></tr></table>

![](images/2f1b614507ca586966ccdf425b9e135bb5833b1f6f1579a66ec273925bc57d09.jpg)

![](images/fdd4c8600442bca10aba80a9b926e07ae481fa44c2aea94c1cbf8e4607737250.jpg)

![](images/35d81a7d877d9d1de75703654e0a82f105862cecebf8c92484e819dbdc51747c.jpg)

![](images/d9511c8fb9274a129544fcad2a3739fc93210ec87bab3b50759babd727da2680.jpg)  
Figure 6: Different data have large difference on improving MATH500 accuracy, but they all improve various tasks rather than their own task. From left to right correspond to 1-shot RL on $\pi _ { 1 }$ , $\pi _ { 1 1 }$ , $\pi _ { 1 3 }$ , or $\pi _ { 1 6 }$ . Details are in Tab. 3.

![](images/6e48b9448bb5a85edccdd6a921d7d6cb106f1add7331dc74fd97db98e36d207f.jpg)

![](images/73749c5f983d5bdb99ff4749494f5a7b17c7d1e34cad316d949b079c036aac9d.jpg)

![](images/2da1120495c8609992643d5d3c27ce3248eb3aa12575a8de8ce672cb057f6c50.jpg)

![](images/dc79623cec1eb67a4eb4c2be84195176e8c4690a2d2f1cb6bdb9e4ba2405d07f.jpg)

![](images/911d6525efa2123c22368cd6d3bafb06d69fe0af26543eae2ea918408d721c2c.jpg)

![](images/d98ce1c1ee5b44a6e6ab59fb430918e7c5b5fd4fe96b24b928bf2b8c401e761a.jpg)

![](images/0eedb3e3468389df2cf2102d1ed3e931dca60b0ece37592da35ad1fa685d8239.jpg)  
Figure 7: Detailed results for RLVR on Qwen2.5-Math-1.5B.

![](images/7397fcbc75af1a160b07d6af9b855ad94e7f9f19474af6d155462eed3bb6c1bf.jpg)

![](images/5d7490930b434b66e6fd8a9a9be1ca252ed907c25102ffb5baa2f8ef979b4d8e.jpg)

![](images/0c2dee60ceb09d8203f8605a6630e36c282e2a5c6918c96429c56d2ee93fef47.jpg)

![](images/cbcaf71593891d0c0503a58bf387cf0dd01a523f94854b0479fc7a2f6414a8dd.jpg)

![](images/de382b679cee1db2d5778817fb4a04cb76ebdc23503b67fe20b2de90b1b6fe07.jpg)

![](images/f16ed5f5b2e10739fb1017f5c07bd26b563693639a0c5fdc0bc76b4993b4b854.jpg)

![](images/77d2e3d6c06ef5e80ff2df444d3051b4c3e86d5cd31620bed604df42aa390b31.jpg)  
Figure 8: Detailed results for RLVR on Qwen2.5-Math-7B.

![](images/ae77152c8e8a90291d72ba11f0cf4f0602458aeec5cd8e6b70b065beb63b3531.jpg)

![](images/2706e57ec0fc1d3b15406fd2c01b2f292fd68d4da6058b8f066ed3b9ae20d21e.jpg)

![](images/36627aba7582895ccef3fb5e63f19be65d6522436e4e8a0d256f263c42827e6d.jpg)

![](images/af38d0c3a1bae1dcff6741c9d3bb8b7d2a3d76adc4592a84996f919c8b36330b.jpg)

![](images/43404c8fb6a1e4145a2361fe187157b6d411636361116993de48cd2731a7f52c.jpg)

![](images/ddae4ea7991c77e0421b7eb0416be99362c20fd440f9c4f69080b76b2bdae68a.jpg)

![](images/5c1f02a01d1c43f99452fb90f6f5f72d851144bf0539b9e36bd8bfc710c13b3a.jpg)  
Figure 9: Detailed results for RLVR on Llama-3.2-3B-Instruct.

![](images/501e599b0a7f0f5ef9e4e77d7fca704b07360bed8279e21af4ae42a774954e9e.jpg)

![](images/657c871b60b2cbd5740c4a8a691400f0e3cae08ccb7c548ecf69374fe0cdb80c.jpg)

![](images/62048a743ee3851240902c9ce0155bf8136b280593836831f32e8d0e07add85d.jpg)

![](images/cc6311700a08d81422d7d5c84f3dbcd9fc8bb6203275d4cfa2d35ac145aceddb.jpg)

![](images/196c46073d61e1d7aa1d54a42d60388648564dd6c26d2716ccda6c7839c3c0dd.jpg)

![](images/5ac1825708e8158e71c664aeacb1438d883a72b8eb4c9390443dfa5c9a0580f4.jpg)

![](images/6e1b47bc7b29c735fab0adea036f7ef70f01f99ae8f9175b8f5c75b88a01e04b.jpg)  
Figure 10: Detailed results for RLVR on Qwen2.5-1.5B. The gap between 1-shot RLVR and full-set RLVR is larger, but the 1-shot RLVR still improves a lot from initial model and 16-shot RLVR behaves close to full-set RLVR.

![](images/adc13f4700123585e17827193531468470c76ef8a968d7901199575cb7da3c83.jpg)

![](images/b809c040b0b2bb091bf915af50f59a9b072e66c006cb393ea652c8d7edf87f99.jpg)

![](images/99adf99230c88adba83587b857fb2193c487215e42dffda0ae7aec3e4f0101af.jpg)

![](images/da293a046e2caeb05f871d30cac4de00e3d559733d20351d4f7bfcf30ab47298.jpg)

![](images/07b5da8d313ffb48d14e9ec650e6b5d9cd6af25e5e48a99c59bd72a8fb466554.jpg)

![](images/d9b914cc2967d0723ff1b6b2af66c01d860e8526059394bd361f98c905eddd26.jpg)

![](images/8aa04a5b7f4f0f21896d9cf087afc8f9a42488b9eaa17d1103304efd117183f0.jpg)  
Figure 11: Detailed results for RLVR on Qwen2.5-Math-1.5B-Instruct. Interestingly, 1-shot RLVR is more stable than full-set RLVR here.

# C Evaluation Result

# C.1 Main Experiments

# C.1.1 Detailed performance on Qwen2.5-Math-1.5B.

In Tab. 8, we show the detailed performance that shown in Fig. 1. Results are reported for the checkpoint achieving the best average performance.

# C.1.2 Detailed Performance on More Models and Training Examples.

In Tab. 10, we also show the 1(few)-shot RLVR results on the base model (Qwen2.5-1.5B [24]) and instruction model (Qwen2.5-Math-1.5B-Instruct [25]). More detailed test curves are shown in Fig. 10 and Fig. 11. We can see that (1) for Qwen2.5-1.5B, the gap between 1-shot RLVR with $\pi _ { 1 }$ and full-set RLVR is larger, but the former still improves model performance significantly (e.g., MATH500: $3 . 2 \%$ to $4 3 . 6 \%$ ), and 16-shot RLVR works very closely to full-set RLVR. (2) for Qwen2.5-Math-1.5B-Instruct, both full-set RLVR and 1-shot RLVR have limited improvement as the initial model already has good performance. Interestingly, as shown in Fig. 11, we observe that 1-shot RLVR is more stable than full-set RLVR.

Besides, we also consider other single training examples like $\pi _ { 6 0 5 }$ and $\pi _ { 1 2 0 9 }$ on Qwen2.5-Math-7B. We can see that they behave relatively worse than $\pi _ { 1 }$ , and 16-shot RLVR provides a more consistent approach to closing the performance gap relative to full-set RLVR.

# C.1.3 Detailed performance with best per-benchmark results

In Tab. 9, we present the detailed 1(few)-shot RLVR results for Qwen2.5-Math-1.5B. Here, we record the model’s best performance on each benchmark individually, so their average can be higher than the best overall average performance (“Avg.”). We include these results to estimate the upper limit of what the model can achieve on each benchmark. Additionally, we include several examples that, while not performing as well as $\pi _ { 1 }$ or $\pi _ { 1 3 }$ , still demonstrate significant improvements, such as $\pi _ { 2 }$ , $\pi _ { 1 2 0 1 }$ , and $\pi _ { 1 2 0 9 }$ . We observe that, in general, better results correspond to a larger checkpoint step for best average performance, which may correspond to a longer post-saturation generalization

process. Similarly, in Tab. 11, we also include the best per-benchmark results for Qwen2.5-Math-7B, Llama-3.2-3B-Instruct, respectively, together with Qwen2.5-Math-1.5B with PPO training.

# C.1.4 Detailed Test curves on MATH500 for 1-shot RLVR on Qwen2.5-Math-1.5B.

We plot the performance curves for each subject in MATH500 under 1-shot RLVR using different mathematical examples. As shown in Fig. 6, the choice of example leads to markedly different improvements and training dynamics in 1-shot RLVR, highlighting the critical importance of data selection for future few-shot RLVR methods.

# C.1.5 Detailed RLVR results on eacn benchmark over training process.

To better visualize the training process of RLVR and compare few-shot RLVR with full-set RLVR, we show the performance curves for each benchamrk on each model in Fig. 7, 8, 9. It will be interesting to see that if applying 1(few)-shot RLVR for more stable GRPO variants [13, 11, 12, 16] can alleviate this phenomenon. In addition to the conclusions discussed in Sec. 3.3, we also note that Llama3.2-3B-Instruct is more unstable during training, as almost all setups start having performance degradation before 200 steps.

In Appendix C.1.2, we also test the base model and instruction version models in Qwen family. Their test curves are also shown in Fig. 10 and Fig. 11.

# C.1.6 More Evaluation on DeepSeek-R1-Distill-Qwen-1.5B

In Tab. 12 we show the DeepSeek-R1-Distill-Qwen-1.5B results at 8k and 32k evaluation lengths. The experimental setup is illustrated in Appendix B.3.

Table 12: DeepSeek-R1-Distill-Qwen-1.5B results at 8k and 32k evaluation lengths. Setup details are in Appendix B.3. $ { ^ { 6 4 } } 8  { \mathrm { k } }  {  } 1 6  { \mathrm { k } }  {  } 2 4  { \mathrm { k } } ^ { \prime \prime }$ denotes the length extension process in DeepScaleR training.   

<table><tr><td>RL Dataset</td><td>Train Length</td><td>MATH 500</td><td>AIME 2024</td><td>AMC 2023</td><td>Minerva Math</td><td>Olympiad- Bench</td><td>AIME 2025</td><td>Avg.</td></tr><tr><td colspan="9">Eval Length = 8k</td></tr><tr><td>NA</td><td>NA</td><td>76.7</td><td>20.8</td><td>51.3</td><td>23.3</td><td>35.4</td><td>19.7</td><td>37.9</td></tr><tr><td>DSR-sub</td><td>8k</td><td>84.4</td><td>30.2</td><td>68.3</td><td>29.2</td><td>45.8</td><td>26.7</td><td>47.4</td></tr><tr><td>DeepScaleR (40k DSR)</td><td>8k→16k→24k</td><td>86.3</td><td>35.2</td><td>68.1</td><td>29.6</td><td>46.7</td><td>28.3</td><td>49.0</td></tr><tr><td>{π1}</td><td>8k</td><td>80.5</td><td>25.1</td><td>58.9</td><td>27.2</td><td>40.2</td><td>21.7</td><td>42.3</td></tr><tr><td>{π1, π2, π13, π1209}</td><td>8k</td><td>81.2</td><td>25.8</td><td>60.1</td><td>26.8</td><td>40.4</td><td>22.0</td><td>42.7</td></tr><tr><td>{π1, ..., π16}</td><td>8k</td><td>83.3</td><td>29.6</td><td>64.8</td><td>29.3</td><td>43.3</td><td>22.8</td><td>45.5</td></tr><tr><td colspan="9">Eval Length = 32k</td></tr><tr><td>NA</td><td>NA</td><td>82.9</td><td>29.8</td><td>63.2</td><td>26.4</td><td>43.1</td><td>23.9</td><td>44.9</td></tr><tr><td>DSR-sub</td><td>8k</td><td>84.5</td><td>32.7</td><td>70.1</td><td>29.5</td><td>46.9</td><td>27.8</td><td>48.6</td></tr><tr><td>DeepScaleR(40k DSR)</td><td>8k→16k→24k</td><td>87.6</td><td>41.4</td><td>73.2</td><td>30.6</td><td>49.6</td><td>31.3</td><td>52.3</td></tr><tr><td>{π1}</td><td>8k</td><td>83.9</td><td>31.0</td><td>66.1</td><td>28.3</td><td>44.6</td><td>24.1</td><td>46.3</td></tr><tr><td>{π1, π2, π13, π1209}</td><td>8k</td><td>84.8</td><td>32.2</td><td>66.6</td><td>27.7</td><td>45.5</td><td>24.8</td><td>46.9</td></tr><tr><td>{π1, ..., π16}</td><td>8k</td><td>84.5</td><td>34.3</td><td>69.0</td><td>30.0</td><td>46.9</td><td>25.2</td><td>48.3</td></tr></table>

# C.2 Analysis

# C.2.1 Test Curves for Ablation Study

In Fig. 12, we can see the test curves for ablation study (Sec. 4.1). We can see that policy gradient loss is the main contributor of 1-shot RLVR. More discussions about format fixing are in Appendix C.2.3.

# C.2.2 Entropy loss

Detailed results of entropy-loss-only training. As in Sec. 4.2, we show the full results of entropyloss-only training in Tab. 13. Training with only entropy loss for a few steps can improve model performance on all math benchmarks except AIME2025. The test curves are in Fig. 12. Notice that the improvement of entropy-loss-only training on Qwen2.5-Math-1.5B is similar to that of RLVR with

![](images/3ce72f4c67c1f74739cb3412971be3e6fe74dacffa1f134b36ca551d6b741ccd.jpg)

![](images/b380c0de8077a29c83566d2e645044ec75b6661380313767087e0ae0fa466ed0.jpg)  
Figure 12: Test curves for ablation study. Here we consider adding policy gradient loss (PG), weight decay (WD), KL divergence loss (KL) and entropy loss (Ent) one by one for 1-shot RLVR training on Qwen2.5-Math-1.5B (Sec. 4.1). Especially for only-entropy training, the test performance quickly achieves 0 since too large entropy will result in random output, but before that, the model gets significant improvement from the first several steps, which is close to the results of format-reward RLVR training (Appendix C.2.3). More discussions are in Appendix C.2.3.

Table 13: Entropy loss alone with $\pi _ { 1 }$ can improve model performance, but it still underperforms compared to the format-reward baseline (Appendix C.2.3).   

<table><tr><td>Model</td><td>MATH
500</td><td>AIME24
2024</td><td>AMC23
2023</td><td>Minerva
Math</td><td>Olympiad-
Bench</td><td>AIME
2025</td><td>Avg.</td></tr><tr><td>Qwen2.5-Math-1.5B</td><td>36.0</td><td>6.7</td><td>28.1</td><td>8.1</td><td>22.2</td><td>4.6</td><td>17.6</td></tr><tr><td>+Entropy Loss, Train 20 steps</td><td>63.4</td><td>8.8</td><td>33.8</td><td>14.3</td><td>26.5</td><td>3.3</td><td>25.0</td></tr><tr><td>Format Reward</td><td>65.0</td><td>8.3</td><td>45.9</td><td>17.6</td><td>29.9</td><td>5.4</td><td>28.7</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>40.8</td><td>8.3</td><td>25.3</td><td>15.8</td><td>13.2</td><td>1.7</td><td>17.5</td></tr><tr><td>+Entropy Loss, Train 10 steps</td><td>47.8</td><td>8.8</td><td>26.9</td><td>18.0</td><td>15.1</td><td>0.4</td><td>19.5</td></tr><tr><td>Qwen2.5-Math-7B</td><td>51.0</td><td>12.1</td><td>35.3</td><td>11.0</td><td>18.2</td><td>6.7</td><td>22.4</td></tr><tr><td>+Entropy Loss, Train 4 steps</td><td>57.2</td><td>13.3</td><td>39.7</td><td>14.3</td><td>21.5</td><td>3.8</td><td>25.0</td></tr><tr><td>Format Reward</td><td>65.8</td><td>24.2</td><td>54.4</td><td>24.3</td><td>30.4</td><td>6.7</td><td>34.3</td></tr></table>

format reward (Appendix C.2.3, Tab. 14), thus we doubt that the effectiveness of entropy-loss-only training may come from format fixing, and we leave the rigorous analysis of this phenomenon for future works.

Discussion of entropy loss and its function in 1-shot RLVR. Notably, we observe that the benefit of adding entropy loss for 1-shot RLVR is consistent with conclusions from previous work [60] on the full RLVR dataset, which shows that appropriate entropy regularization can enhance generalization, although it remains sensitive to the choice of coefficient. We conjecture the success of 1-shot RLVR is that the policy gradient loss on the learned example (e.g., $\pi ( \bar { 1 } ) )$ actually acts as an implicit regularization by ensuring the correctness of learned training examples when the model tries to explore more diverse responses or strategies, as shown in Fig. 3 (Step 1300). And because of this, both policy loss and entropy loss can contribute to the improvement of 1-shot RLVR. We leave the rigorous analysis to future works.

# C.2.3 (Only) Format Correction?

As discussed in Dr. GRPO [13], changing the template of Qwen2.5-Math models can significantly affect their math performance. In this section, we investigate some critical problems: is (1-shot) RLVR doing format fixing? And if the answer is true, is this the only thing 1-shot RLVR does?

To investigate it, we consider three methods:

(a). Applying format reward in RLVR. We first try to apply only format reward for RLVR (i.e., if the verifier can parse the final answer from model output, then it gets 1 reward no matter if the answer is correct or not, otherwise it gets 0 reward), considering both 1-shot and full-set. The results are shown in Tab. 14, and the test curves are shown in Fig. 14 and Fig. 13, respectively.

Notably, we can find that (1) Applying format reward to full-set RLVR and 1-shot RLVR behave very similarly. (2) applying only format reward is already capable of improving model performance

![](images/2695eea3712c0ce106d5f454bb3db1d618ff6f8dc2e3da34b34e534e5dc80b78.jpg)

![](images/8808a37493139ac8a71a7fcf52ccfd6f79f09a3f19d3b2289a4ad7501f524bbb.jpg)

![](images/9f01115bf6f78fb5aa927dbbd1d25a99f38d25208f488e35bc2a4dc53bb299e8.jpg)

![](images/66ad69f50693b4ee93809cf3f0a74848018de2fe8f1366f9da3da759e13992e9.jpg)

![](images/727e93d814b8885d1f2501eeaa8c28f0408a885be780dccfeeee1ba566763aac.jpg)

![](images/56297e0d8382c47d0927b36667a8ce7db7fb549ba5d15ce7e4b0b52eb2acb490.jpg)

![](images/75cb59acf498ca3cb7592f7bc764341428a20ed1231f7b2987c9f02c8d6dcaed.jpg)  
Figure 13: Comparison between outcome reward and format reward for full-set RLVR with 1.2k DSR-sub on Qwen2.5-Math-1.5B.

![](images/f4279d166d7a6316ebc91a2c412ace9b6616a40b49e0cc70aea3c87920e58029.jpg)

![](images/ff7e87697d35487146071184faa1049166684a09bc24664ea5fe14e0a3d37be0.jpg)

![](images/fae7e8ed08e5d7a51a5fab83578c38ed36c0a5c110dd1dd129d586a09bc972ea.jpg)

![](images/05e96cb627895f4f7004cc841753aa7c4605fba1996939d31d4601cd987e8aff.jpg)

![](images/34c1284d08d0c898edbdf968f9347bb4d57132a82d46020b3c5bb52241a33c1c.jpg)

![](images/e7b4f55d5d5cc370e9337022b74daa01235cc756b89340df0e3a7da1ec314fd9.jpg)

![](images/a5674f81b48c0b84425e11326e02a613ca5dc213a500d6fcc618ff908e1a92b3.jpg)  
Figure 14: Comparison between outcome reward and format reward for 1-shot RLVR with $\pi _ { 1 }$ on Qwen2.5-Math-1.5B.

Table 14: RLVR with only format reward can still improve model performance significantly, while still having a gap compared with that using outcome reward. Numbers with orange color denote the ratio of responses that contain “\boxed{}” in evaluation. Here we consider adding entropy loss or not for format reward. Detailed test curves are in Fig. 13 and Fig. 14. We can see that: (1) RLVR with format reward has similar test performance between 1.2k dataset DSR-sub and $\pi _ { 1 }$ . (2) $\pi _ { 1 }$ with outcome reward or format reward have similar \boxed{} ratios, but the former still has better test performance (e.g., $+ 7 . 4 \%$ on MATH500 and $+ 5 . 8 \%$ on average). (3) Interestingly, RLVR with DSR-sub using outcome reward can fix the format perfectly, although it still has similar test performance as 1-shot RLVR with $\pi _ { 1 }$ (outcome reward).   

<table><tr><td>Dataset</td><td>Reward Type</td><td>Entropy Loss</td><td>MATH 500</td><td>AIME 2024</td><td>AMC 2023</td><td>Minerva Math</td><td>Olympiad-Bench</td><td>AIME 2025</td><td>Avg.</td></tr><tr><td>NA</td><td>NA</td><td>NA</td><td>36.060%</td><td>6.775%</td><td>28.183%</td><td>8.159%</td><td>22.276%</td><td>4.681%</td><td>17.672%</td></tr><tr><td>DSR-sub</td><td>Outcome</td><td>+</td><td>73.6100%</td><td>17.199%</td><td>50.6100%</td><td>32.499%</td><td>33.699%</td><td>8.3100%</td><td>35.999%</td></tr><tr><td>DSR-sub</td><td>Format</td><td>+</td><td>65.094%</td><td>8.383%</td><td>45.994%</td><td>17.689%</td><td>29.992%</td><td>5.490%</td><td>28.791%</td></tr><tr><td>DSR-sub</td><td>Format</td><td></td><td>61.493%</td><td>9.687%</td><td>44.794%</td><td>16.583%</td><td>29.590%</td><td>3.887%</td><td>27.689%</td></tr><tr><td>{π1}</td><td>Outcome</td><td>+</td><td>72.897%</td><td>15.492%</td><td>51.697%</td><td>29.892%</td><td>33.588%</td><td>7.193%</td><td>35.093%</td></tr><tr><td>{π1}</td><td>Outcome</td><td></td><td>68.297%</td><td>15.492%</td><td>49.495%</td><td>25.094%</td><td>31.791%</td><td>5.890%</td><td>32.693%</td></tr><tr><td>{π1}</td><td>Format</td><td>+</td><td>65.496%</td><td>8.891%</td><td>43.898%</td><td>22.191%</td><td>31.690%</td><td>3.888%</td><td>29.292%</td></tr><tr><td>{π1}</td><td>Format</td><td></td><td>61.692%</td><td>8.384%</td><td>46.290%</td><td>15.478%</td><td>29.389%</td><td>4.686%</td><td>27.688%</td></tr></table>

significantly (e.g., about $29 \%$ improvement on MATH500 and about $11 \%$ gain on average). (3) There is still significant gap between the performance of 1-shot RLVR with outcome reward using $\pi _ { 1 }$ and that of format-reward RLVR (e.g., $+ 7 . 4 \%$ on MATH500 and $+ 5 . 8 \%$ on average), although they may have similar ratios of responses that contain “\boxed{}” in evaluation (More discussions are in (b) part). (4) In particular, format-reward RLVR is more sensitive to entropy loss based on Fig. 14 and Fig. 13.

Interestingly, we also note that the best performance of format-reward RLVR on MATH500 and AIME24 are close to that for 1-shot RLVR with relatively worse examples, for example, $\pi _ { 7 }$ and $\pi _ { 1 1 }$ in Tab. 3. This may imply that 1-shot RLVR with outcome reward can at least work as well as format-reward RLVR, but with proper examples that can better incentivize the reasoning capability of the model, 1-shot RLVR with outcome reward can bring additional non-trivial improvement. Appendix C.2.5 provides a prompt $\pi _ { 1 } ^ { \prime }$ , which uses a sub-question of $\pi _ { 1 }$ , as an example to support our claim here.

![](images/1ee60bf2387ffa75520f948942212f2c6bb4005fc765f96b420bb6a0417b2559.jpg)

![](images/e18ea6200dadc4f5e61d6df633ff3044b56d7bd7d6c11bf32dd494235d523894.jpg)

![](images/4246ea64b391d4907aef7abdec992047fa0a8a2ca519b4b4c97195dd289874f1.jpg)

![](images/66541fc69172a316b3ca4b8b151c4275009467a44ba2aa8b52def67a826e951f.jpg)

![](images/3aef78569eaeffc93435972d6fde47dd274a35f3895d138f8ad548745d728fa9.jpg)

![](images/bb6de850c52456fe946e2d13b36364546582284bef8419f18cd7c55108a8b50a.jpg)  
Figure 15: Relation between the number of \boxed{} and test accuracy. We can see that they have a strong positive correlation. However, after the number of \boxed{} enters a plateau, the evaluation results on some evaluation tasks continue improving (like Minerva Math, OlympiadBench and MATH500).

(b) Observe the change of format in 1-shot RLVR. We then investigate how the output format of the model, for example, the number of \boxed{}, changes in the 1-shot RLVR progress. The results are shown in Fig. 15. We can see that (1) the test accuracy is strongly positively correlated to the number of \boxed{}, which matches our claim that format fixing contributes a lot to model

Table 15: 1-shot RLVR does not do something like put the answer into the \boxed{}. “Ratio of disagreement” means the ratio of questions that has different judgement between Qwen-Eval and QwQ-32B judge. Here we let QwQ-32B judged based on if the output contain correct answer, without considering if the answer is put in the \boxed{}.   

<table><tr><td></td><td>Step0</td><td>Step 20</td><td>Step 60</td><td>Step 500</td><td>Step 1300</td><td>Step 1860</td></tr><tr><td>Ratio of \boxed{}</td><td>59.6%</td><td>83.6%</td><td>97.4%</td><td>96.6%</td><td>96.6%</td><td>94.2%</td></tr><tr><td>Acc. judge by Qwen-Eval</td><td>36.0</td><td>53.8</td><td>69.8</td><td>70.4</td><td>72.2</td><td>74.0</td></tr><tr><td>Acc. judge by QwQ-32B</td><td>35.8</td><td>57.2</td><td>70.6</td><td>71.8</td><td>73.6</td><td>74.6</td></tr><tr><td>Ratio of disagreement</td><td>4.2%</td><td>5%</td><td>1.2%</td><td>1.4%</td><td>1.8%</td><td>1.8%</td></tr></table>

Table 16: $\pi _ { 1 }$ even performs well for in-context learning on Qwen2.5-Math-7B. Here “Qwen official 4 examples” are from Qwen Evaluation repository [25] for 4-shot in-context learning on MATH500, and “Qwen official Example 1” is the first example.   

<table><tr><td>Dataset</td><td>Method</td><td>MATH 500</td><td>AIME 2024</td><td>AMC 2023</td><td>Minerva Math</td><td>Olympiad- Bench</td><td>AIME 2025</td><td>Avg.</td></tr><tr><td colspan="9">Qwen2.5-Math-1.5B</td></tr><tr><td rowspan="2">NA\{π1}</td><td>NA</td><td>36.0</td><td>6.7</td><td>28.1</td><td>8.1</td><td>22.2</td><td>4.6</td><td>17.6</td></tr><tr><td>RLVR</td><td>72.8</td><td>15.4</td><td>51.6</td><td>29.8</td><td>33.5</td><td>7.1</td><td>35.0</td></tr><tr><td rowspan="2">{π1}Qwen official 4 examples</td><td>In-Context</td><td>59.0</td><td>8.3</td><td>34.7</td><td>19.9</td><td>25.6</td><td>5.4</td><td>25.5</td></tr><tr><td>In-Context</td><td>49.8</td><td>1.7</td><td>16.9</td><td>19.9</td><td>19.9</td><td>0.0</td><td>18.0</td></tr><tr><td>Qwen official Example 1</td><td>In-Context</td><td>34.6</td><td>2.5</td><td>14.4</td><td>12.1</td><td>21.0</td><td>0.8</td><td>14.2</td></tr><tr><td colspan="9">Qwen2.5-Math-7B</td></tr><tr><td rowspan="2">NA\{π1}</td><td>NA</td><td>51.0</td><td>12.1</td><td>35.3</td><td>11.0</td><td>18.2</td><td>6.7</td><td>22.4</td></tr><tr><td>RLVR</td><td>79.2</td><td>23.8</td><td>60.3</td><td>27.9</td><td>39.1</td><td>10.8</td><td>40.2</td></tr><tr><td rowspan="2">{π1}Qwen official 4 examples</td><td>In-Context</td><td>75.4</td><td>15.8</td><td>48.4</td><td>30.1</td><td>41.3</td><td>13.3</td><td>37.4</td></tr><tr><td>In-Context</td><td>59.2</td><td>4.2</td><td>20.9</td><td>20.6</td><td>24.4</td><td>0.8</td><td>21.7</td></tr><tr><td>Qwen official Example 1</td><td>In-Context</td><td>54.0</td><td>4.2</td><td>23.4</td><td>18.4</td><td>21.2</td><td>2.1</td><td>20.6</td></tr></table>

improvement in (a), but (2) for some benchmarks like MATH500, Minerva Math and OlympiadBench, when the number of \boxed{} keeps a relatively high ratio, the test accuracy on these benchmarks is still improving, which may imply independent improvement of reasoning capability.

In particular, to prevent the case that the model outputs the correct answer but not in \boxed{}, we also use LLM-as-a-judge [61] with QwQ-32B [62] to judge if the model contains the correct answer in the response. The results are shown in Tab. 15. We can see that the accuracy judged by rulebased Qwen-Eval pipeline and LLM judger QwQ-32B are very close, and as the ratio of \boxed{} increases, the test accuracy also increases, which implies that the number of correct answers exhibited in the response also increases, rather than just putting correct answer into \boxed{}.

Notably, we also observe that Qwen2.5-Math models contain lots of repetition at the end of model responses, which may result in failure of obtaining final results. The ratio of repetition when evaluating MATH500 can be as high as about $40 \%$ and $20 \%$ for Qwen2.5-Math-1.5B and Qwen2.5- Math-7B, respectively, which is only about $2 \%$ for Llama3.2-3B-Instruct. This may result in the large improvement of format fixing (e.g., format-reward RLVR) mentioned in (a).

(c) In-context learning with one-shot example. In-context learning [63] is a widely-used baseline for instruction following (although it may still improve model’s reasoning capability). In this section, we try to see if 1-shot RLVR can behave better than in-context learning. Especially, we consider the official 4 examples chosen by Qwen-Eval [25] for in-context learning, and also the single training example $\pi _ { 1 }$ . The results are shown in Tab. 16.

We can find that (1) surprisingly, $\pi _ { 1 }$ with self-generated response can behave much better than Qwen’s official examples, both for 1.5B and 7B models. In particular on Qwen2.5-Math-7B, incontext learning with $\pi _ { 1 }$ can improve MATH500 from $5 1 . 0 \%$ to $7 5 . 4 \%$ and on average from $2 2 . 4 \%$ to $3 7 . 4 \%$ . (2) Although in-context learning also improves the base models, 1-shot RLVR still performs better than all in-context results, showing the advantage of RLVR.

Table 17: Influence of Random Wrong Labels. Here “Error Rate” means the ratio of data that has the random wrong labels.   

<table><tr><td>Dataset</td><td>Error Rate</td><td>MATH 500</td><td>AIME 2024</td><td>AMC 2023</td><td>Minerva Math</td><td>Olympiad-Bench</td><td>AIME 2025</td><td>Avg.</td></tr><tr><td>NA</td><td>NA</td><td>36.0</td><td>6.7</td><td>28.1</td><td>8.1</td><td>22.2</td><td>4.6</td><td>17.6</td></tr><tr><td colspan="9">Qwen2.5-Math-1.5B + GRPO</td></tr><tr><td>DSR-sub</td><td>0%</td><td>73.6</td><td>17.1</td><td>50.6</td><td>32.4</td><td>33.6</td><td>8.3</td><td>35.9</td></tr><tr><td>DSR-sub</td><td>60%</td><td>71.8</td><td>17.1</td><td>47.8</td><td>29.4</td><td>34.4</td><td>7.1</td><td>34.6</td></tr><tr><td>DSR-sub</td><td>90%</td><td>67.8</td><td>14.6</td><td>46.2</td><td>21.0</td><td>32.3</td><td>5.4</td><td>31.2</td></tr><tr><td>{π1}</td><td>0%</td><td>72.8</td><td>15.4</td><td>51.6</td><td>29.8</td><td>33.5</td><td>7.1</td><td>35.0</td></tr><tr><td colspan="9">Qwen2.5-Math-1.5B + PPO</td></tr><tr><td>DSR-sub</td><td>0%</td><td>72.8</td><td>19.2</td><td>48.1</td><td>27.9</td><td>35.0</td><td>9.6</td><td>35.4</td></tr><tr><td>DSR-sub</td><td>60%</td><td>71.6</td><td>13.3</td><td>49.1</td><td>27.2</td><td>34.4</td><td>12.1</td><td>34.6</td></tr><tr><td>DSR-sub</td><td>90%</td><td>68.2</td><td>15.8</td><td>50.9</td><td>26.1</td><td>31.9</td><td>4.6</td><td>32.9</td></tr><tr><td>{π1}</td><td>0%</td><td>72.4</td><td>11.7</td><td>51.6</td><td>26.8</td><td>33.3</td><td>7.1</td><td>33.8</td></tr></table>

In short, we use these three methods to confirm that 1-shot RLVR indeed does format fixing and obtains a lot of gain from it, but it still has additional improvement that cannot be easily obtained from format reward or in-context learning.

# C.2.4 Influence of Random Wrong Labels

In this section, we want to investigate the label robustness of RLVR. It’s well-known that general deep learning is robust to label noise [64], and we want to see if this holds for RLVR. We try to randomly flip the labels of final answers in DSR-sub and see their performance. Here we randomly add or subtract numbers within 10 and randomly change the sign. If it is a fraction, we similarly randomly add or subtract the numerator and denominator.

The results are in Tab. 17. We can see that (1) changing $60 \%$ of the data with wrong labels can still achieve good RLVR results. (2) if $90 \%$ of the data in the dataset contains wrong labels (i.e., only about 120 data contain correct labels, and all other 1.1k data have wrong labels), the model performance will be worse than that for 1-shot RLVR with $\pi _ { 1 }$ (which only contains 1 correct label!). This may show that RLVR is partially robust to label noise, but if there are too many data with random wrong labels, they may hurt the improvement brought by data with correct labels.

# C.2.5 Change the Prompt of $\pi _ { 1 }$

Table 18: Keeping CoT complexity in problem-solving may improve model performance. Com-√ paring $\pi _ { 1 }$ and simplified variant $\pi _ { 1 } ^ { \prime }$ (prompt: “Calculate $\sqrt [ 3 ] { 2 0 4 8 } ^ { 3 } )$ , where we only keep the main step that Qwen2.5-Math-1.5B may make a mistake on. We record the results from the checkpoint with the best average performance. For $\pi _ { 1 } ^ { \prime }$ , the model’s output CoT is simpler and the corresponding 1-shot RLVR performance is worse. The additional improvement of $\pi _ { 1 } ^ { \prime }$ is relatively marginal compared with using format reward, showing the importance of the training example used in 1-shot RLVR.   

<table><tr><td>RL Dataset</td><td>Reward Type</td><td>MATH 500</td><td>AIME 2024</td><td>AMC 2023</td><td>Minerva Math</td><td>Olympiad-Bench</td><td>AIME 2025</td><td>Avg.</td></tr><tr><td colspan="9">Qwen2.5-Math-1.5B [24]</td></tr><tr><td>NA</td><td>NA</td><td>36.0</td><td>6.7</td><td>28.1</td><td>8.1</td><td>22.2</td><td>4.6</td><td>17.6</td></tr><tr><td>{π1}</td><td>outcome</td><td>72.8</td><td>15.4</td><td>51.6</td><td>29.8</td><td>33.5</td><td>7.1</td><td>35.0</td></tr><tr><td>Simplified {π1&#x27;}</td><td>outcome</td><td>65.4</td><td>9.6</td><td>45.9</td><td>23.2</td><td>31.1</td><td>5.0</td><td>30.0</td></tr><tr><td>DSR-sub</td><td>Format</td><td>65.0</td><td>8.3</td><td>45.9</td><td>17.6</td><td>29.9</td><td>5.4</td><td>28.7</td></tr></table>

As discussed in Sec. 3.2.1, we show that the model can almost solve √ $\pi _ { 1 }$ but sometimes fails in solving its last step: “Calculate $\sqrt [ 3 ] { 2 0 4 8 } ^ { } ,$ . We use this step itself as a problem $( \pi _ { 1 } ^ { \prime } )$ , and see how it behaves in 1-shot RLVR. The results are in Tab. 18. Interestingly, we find that $\pi _ { 1 } ^ { \prime }$ significantly underperforms $\pi _ { 1 }$ and has only $1 . 3 \%$ average improvement compared with format reward (as illustrated in

Appendix C.2.3 (a)). We think the reason should be that although solving $\sqrt [ 3 ] { 2 0 4 8 }$ is one of the most difficult parts of $\pi _ { 1 }$ , $\pi _ { 1 }$ still needs other key steps to solve (e.g., calculating $k$ from $P = k A V ^ { 3 }$ given some values) that may generate different patterns of CoT (rather than just calculating), which may allow more exploration space at the post-saturation generalization stage and maybe better incentivize the model’s reasoning capability.

# C.3 Response Length

In Tab. 19, we report the average response length on the evaluation tasks. The response length on the test tasks remains relatively stable compared to that on the training data.

Table 19: Average response length of Qwen2.5-Math-1.5B on evaluation tasks. We use the formatreward experiment (DSR-sub $^ +$ format reward in Tab. 14) as the baseline to eliminate differences in token counts introduced by formats.   

<table><tr><td>Setting</td><td>MATH 500</td><td>AIME24 2024</td><td>AMC23 2023</td><td>Minerva Math</td><td>Olympiad- Bench</td><td>AIME 2025</td><td>Avg.</td></tr><tr><td>Format Reward</td><td>689</td><td>1280</td><td>911</td><td>1018</td><td>957</td><td>1177</td><td>1005</td></tr><tr><td>1-shot RLVR w/ π1 (step 100)</td><td>611</td><td>1123</td><td>939</td><td>1072</td><td>951</td><td>1173</td><td>978</td></tr><tr><td>1-shot RLVR w/ π1 (step 1500)</td><td>740</td><td>1352</td><td>986</td><td>905</td><td>1089</td><td>1251</td><td>1054</td></tr><tr><td>RLVR w/ DSR-sub (step 100)</td><td>636</td><td>1268</td><td>874</td><td>797</td><td>954</td><td>1122</td><td>942</td></tr><tr><td>RLVR w/ DSR-sub (step 1500)</td><td>562</td><td>949</td><td>762</td><td>638</td><td>784</td><td>988</td><td>780</td></tr></table>

# C.4 Pass@8 Results

In Tab. 20, we report the pass@8 results on the evaluation tasks. Interestingly, we find that (1) 1-shot RLVR achieves comparable or even slightly better pass@8 performance (51.7(2) full-set RLVR (with 1.2k DSR-sub) exhibits a noticeable downward trend in pass@8 performance after 200 steps, which is consistent with recent findings that RLVR may sometimes degrade the pass@n performance [20].

Table 20: Pass@8 results on 3 math evaluation tasks using Qwen2.5-Math-1.5B. We also include the performance of RLVR with format-reward (as in Table 19) as a stronger baseline.   

<table><tr><td>Setting</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>Avg. (3 tasks)</td></tr><tr><td>Base Model</td><td>26.6</td><td>20.0</td><td>72.5</td><td>39.7</td></tr><tr><td>Format Reward(highest)</td><td>33.3</td><td>23.3</td><td>72.5</td><td>43.1</td></tr><tr><td>RLVR w/ DSR-sub (highest, step 160)</td><td>36.7</td><td>26.7</td><td>87.5</td><td>50.3</td></tr><tr><td>RLVR w/ DSR-sub (step 500)</td><td>33.3</td><td>30.0</td><td>82.5</td><td>48.6</td></tr><tr><td>RLVR w/ DSR-sub (step 1000)</td><td>33.3</td><td>20.0</td><td>75.0</td><td>42.8</td></tr><tr><td>RLVR w/ DSR-sub (step 1500)</td><td>30.0</td><td>26.7</td><td>67.5</td><td>41.3</td></tr><tr><td>1-shot RLVR (step 500)</td><td>30.0</td><td>16.7</td><td>80.0</td><td>42.2</td></tr><tr><td>1-shot RLVR (highest, step 980)</td><td>36.7</td><td>33.3</td><td>85.0</td><td>51.7</td></tr><tr><td>1-shot RLVR (step 1500)</td><td>26.6</td><td>23.3</td><td>87.5</td><td>45.8</td></tr></table>

# D Discussions

# D.1 Limitations of Our Work

Due to the limit of computational resources, we haven’t tried larger models like Qwen2.5-32B training currently. But in general, a lot of RLVR works are conducted on 1.5B and 7B models, and they already achieve impressive improvement on some challenging math benchmarks like OlympiadBench, so our experiments are still insightful for RLVR topics. Another limitation of our work is that we mainly focus on the math domain, but haven’t tried 1(few)-shot RLVR on other verifiable domains like coding. But we also emphasize that all math-related experiments and conclusions in our paper are logically self-contained and clearly recorded, to ensure clarity and avoid confusion for readers. And we mainly focus on analyzing this new phenomenon itself, which already brings a lot of novel observations (e.g., cross-category generalization, post-saturation generalization, and more frequent self-reflection in 1-shot RLVR, etc.). We leave the few-shot RLVR on other scenarios for future work.

![](images/1f4a887f21f455a9d9c15ff5b8a5731ecb7fd1502311ac059703d15e8ef62647.jpg)  
Figure 16: The norm of policy gradient loss for 1-shot RLVR $\left( \pi _ { 1 } \right)$ ) on Qwen2.5-Math-1.5B.

In particular, we note that our main focus is to propose a new observation rather than propose a new better method, noting that 1-shot RLVR doesn’t save (and maybe requires more) RL computation. Besides, $\pi _ { 1 }$ is not necessarily the best choice for 1-shot RLVR on other models, since it’s selected based on the historical variance score of Qwen2.5-Math-1.5B. In general, using few-shot RLVR may be more stable for training, as we have seen that on DeepSeek-R1-Distill-Qwen-1.5B (Tab. 4), Qwen2.5-Math-7B (Tab. 4, 10) and Qwen2.5-1.5B (Tab. 10), RLVR with 16 examples $( \{ \pi _ { 1 } , \ldots , \pi _ { 1 6 } \} )$ ) works as well as RLVR with 1.2k dataset DSR-sub and outperforms 1-shot RL with $\pi _ { 1 }$ .

# D.2 Reasoning Capability of Base Models

The effectiveness of 1(few)-shot RLVR provides strong evidence for an assumption people proposed recently, that is, base models already have strong reasoning capability [13, 6, 20, 21]. For example, Dr. GRPO [13] has demonstrated that when no template is used, base models can achieve significantly better downstream performance. Recent work further supports this observation by showing that, with respect to the pass@k metrics, models trained via RLVR gradually perform worse than the base model as $k$ increases [20]. Our work corroborates this claim from another perspective, as a single example provides almost no additional knowledge. Moreover, our experiments reveal that using very few examples with RLVR is already sufficient to achieve significant improvement on mathematical reasoning tasks. Thus, it is worth investigating how to select appropriate data to better activate the model during the RL stage while maintaining data efficiency.

# D.3 Why Model Continues Improving After the Training Accuracy Reaches Near $100 \%$ ?

A natural concern of 1-shot RLVR is that if training accuracy reaches near $100 \%$ (which may occur when over-training on one example), the GRPO advantage (Eqn. 6) should be zero, eliminating policy gradient signal. However, entropy loss encourages diverse outputs, causing occasional errors ( $9 9 . \mathrm { { X } \% }$ training accuracy) and non-zero gradients (advantage becomes large for batches with wrong responses due to small variance). This shows the importance of entropy loss to the post-saturation generalization (Fig. 5). Supporting this, Fig. 16 shows that for 1-shot RLVR training $( \pi _ { 1 } )$ on Qwen2.5-Math-1.5B, policy gradient loss remains non-zero after 100 steps.

# D.4 Future Works

We believe our findings can provide some insights for the following topics:

Data Selection and Curation. Currently, there are no specific data selection methods for RLVR except LIMR [19]. Note that 1-shot RLVR allows for evaluating each example individually, it will be helpful for assessing the data value, and thus help to design better data selection strategy. What’s more, noting that different examples can have large differences in stimulating LLM reasoning capability (Tab. 3), it may be necessary to find out what kind of data is more useful for RLVR, which is critical for the RLVR data collection stage. It’s worth mentioning that our work does not mean scaling RLVR datasets is useless, but it emphasizes the importance of better selection and collection of data for RLVR.

Table 21: Details of example $\pi _ { 1 3 }$ .   

<table><tr><td>Prompt</td></tr><tr><td>Given that circle $C$ passes through points $P(0,-4)$, $Q(2,0)$, and $R(3,-1)$. \n$(1)$ Find the equation of circle $C$. \n$(2)$ If the line $l: mx+y-1=0$ intersects circle $C$ at points $A$ and $B$, and $|AB|=4$, find the value of $m$. Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 4/3 .</td></tr><tr><td>Table 22: Details of example π2 .</td></tr><tr><td>Prompt:</td></tr><tr><td>How many positive divisors do 9240 and 13860 have in common? Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 24 .</td></tr></table>

Understanding 1-shot RLVR and Post-saturation Generalization A rigorous understanding for the feasibility of 1-shot LLM RLVR and post-saturation generalization is still unclear. We think that one possible hypothesis is that the policy loss on the learned examples plays a role as “implicit regularization” of RLVR when the model tries to explore more diverse output strategies under the encouragement of entropy loss or larger rollout temperature. It will punish the exploration patterns that make the model fail to answer the learned data, and thus provide a verification for exploration. It’s interesting to explore if the phenomenon has relevance to Double Descent [65] or the implicit regularization from SGD [66, 67], as 1-shot RLVR on $\pi _ { 1 3 }$ (Fig. 2, middle) shows a test curve similar to Double Descent. We leave the rigorous analysis of this phenomenon for future works, and we believe that can help us to comprehend what happens in the RLVR process.

Importance of Exploration. In Sec. 4.1, we also highlight the importance of entropy loss in 1-shot RLVR, and note that a more thorough explanation of why training with only entropy loss can enhance model performance remains an interesting direction for future work (Sec. 4.2). Relatedly, entropy loss has also received increasing attention from the community, with recent works discussing its dynamics [68, 47, 60] or proposing improved algorithms from the perspective of entropy [46]. Moreover, we believe a broader and more important insight for these is that encouraging the model to explore more diverse outputs within the solution space is critical, as it may significantly impact the model’s generalization to downstream tasks [69]. Adding entropy loss is merely one possible approach to achieve this goal and may not necessarily be the optimal solution. As shown in our paper and previous work [60], the effectiveness of entropy loss is sensitive to the choice of coefficient, which could limit its applicability in larger-scale experiments. We believe that discovering better strategies to promote exploration could further enhance the effectiveness of RLVR.

Other Applications. In this paper, we focus primarily on mathematical reasoning data; however, it is also important to evaluate the efficacy of 1-shot RLVR in other domains, such as code generation or tasks without verifiable rewards. Moreover, investigating methodologies to further improve fewshot RLVR performance under diverse data-constrained scenarios represents a valuable direction. Examining the label robustness of RLVR, as discussed in Sec. 4.2, likewise merits further exploration. Finally, these observations may motivate the development of additional evaluation sets to better assess differences between 1-shot and full-set RLVR on mathematical or other reasoning tasks.

# E Example Details

In the main paper, we show the details of $\pi _ { 1 }$ . Another useful example $\pi _ { 1 3 }$ is shown in Tab. 21. Here we mention that $\pi _ { 1 3 }$ is a geometry problem and its answer is precise. And similar to $\pi _ { 1 }$ , the initial base model still has $2 1 . 9 \%$ of outputs successfully obtaining $\frac 4 3$ in 128 samplings.

Besides, Tab. 22 through 42 in the supplementary material provide detailed information for each example used in our experiments and for all other examples in $\{ \pi _ { 1 } , \ldots , \pi _ { 1 7 } \}$ . Each table contains the specific prompt and corresponding ground truth label for an individual example.

Table 23: Details of example $\pi _ { 3 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>There are 10 people who want to choose a committee of 5 people among them. They do this by first electing a set of $1,2,3$, or 4 committee leaders, who then choose among the remaining people to complete the 5-person committee. In how many ways can the committee be formed, assuming that people are distinguishable? (Two committees that have the same members but different sets of leaders are considered to be distinct.) Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 7560 .</td></tr></table>

Table 24: Details of example $\pi _ { 4 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>Three integers from the list $1,2,4,8,16,20$ have a product of 80. What is the sum of these three integers? 
Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 25 .</td></tr></table>

Table 25: Details of example $\pi _ { 5 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>In how many ways can we enter numbers from the set ${}\{1,2,3,4\}$ into a $4 \times times 4$ array so that all of the following conditions hold? (a) Each row contains all four numbers. (b) Each column contains all four numbers. (c) Each &quot;quadrant&quot; contains all four numbers. (The quadrants are the four corner $2 \times times 2$ squares.) Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 288 .</td></tr></table>

Table 26: Details of example $\pi _ { 6 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>The vertices of a $3 \times times 1 \times times 1$ rectangular prism are $A, B, C, D, E, F, G$, and $H$ so that $A E, B F$, $C G$, and $D H$ are edges of length 3. Point $I$ and point $J$ are on $A E$ so that $A I = I J = J E</td></tr><tr><td>Similarly, points $K$ and $L$ are on $B F$ so that $B K = K L = L F = 1$, points $M$ and $N$ are on $C G$ so that $C M = M N = N G = 1$, and points $O$ and $P$ are on $D H$ so that $D O = O P = P H = 1$. For every pair of the 16 points $A$ through $P$, Maria computes the distance between them and lists the 120 distances. How many of these 120 distances are equal to $\sqrt[3]{2}$$? Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 32 .</td></tr></table>

Table 27: Details of example $\pi _ { 7 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>Set $u_0 = \\frac{1}{4}$. and for $k \ge 0$ let $u_{k+1}$ be determined by the recurrence\n
\[ [u_{k+1}] = 2u_{k} - 2u_{k-2}. \]This sequence tends to a limit; call it $L$. What is the least value of $k$ such that\n[[u_k-L] \le 1000]? \] Let&#x27;s think step by step and output the final answer within \boxed{}</td></tr><tr><td>Ground truth (label in DSR-sub): 10.</td></tr></table>

Table 28: Details of example $\pi _ { 8 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>Consider the set \{2, 7, 12, 17, 22, 27, 32\}. Calculate the number of different integers that can be expressed as the sum of three distinct members of this set. Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 13 .</td></tr></table>

Table 29: Details of example $\pi _ { 9 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>In a group photo, 4 boys and 3 girls are to stand in a row such that no two boys or two girls stand next to each other. How many different arrangements are possible? Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 144 .</td></tr></table>

Table 30: Details of example $\pi _ { 1 0 }$   

<table><tr><td>Prompt:</td></tr><tr><td>How many ten-digit numbers exist in which there are at least two identical digits? Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 8996734080 .</td></tr></table>

Table 31: Details of example $\pi _ { 1 1 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>How many pairs of integers $a$ and $b$ are there such that $a$ and $b$ are between $1$ and $42$ and $a~9 = b~7 \mod 43$ ? Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 42 .</td></tr></table>

Table 32: Details of example $\pi _ { 1 2 }$   

<table><tr><td>Prompt:</td></tr><tr><td>Two springs with stiffnesses of $6 \, , \text{kN} \}/ \text{m} $ and $12 \, , \text{kN} \)/ \text{m} $ are connected in series. How much work is required to stretch this system by 10 cm? Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 20 .</td></tr></table>

Table 33: Details of example $\pi _ { 1 4 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>Seven cards numbered $1$ through $7$ are to be lined up in a row. Find the number of arrangements of these seven cards where one of the cards can be removed leaving the remaining six cards in either ascending or descending order. Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 74 .</td></tr></table>

Table 34: Details of example $\pi _ { 1 5 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>What is the area enclosed by the geoboard quadrilateral below?\n[asy] unitsize(3mm);</td></tr><tr><td>defaultpen(linewidth(.8pt)); dotfactor=2; for(int a=0; a&lt;=10; ++a) for(int b=0; b&lt;=10; ++b)</td></tr><tr><td>{ dot((a,b));; draw((4,0)--(0,5)--(3,4)--(10,10)--cycle); [/asy] Let&#x27;s think step by step and output the final answer within \boxed{}}.</td></tr><tr><td>Ground truth (label in DSR-sub): 221/2.</td></tr></table>

Table 35: Details of example $\pi _ { 1 6 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>If $p, q,$ and $r$ are three non-zero integers such that $p + q + r = 26$ and\[\{\{p\} + \{\{q\} + \{\{r\} + \{\{p\} + \{\{r\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\} + \{\{p\}\} + \{\{p\} + \{\{p\}\} + \{\{p\} + \{\{p\}\} + \{\{p\} + \{\{p\}\} + \{\{p\} + \{\{p\}\} + \{\{p\} + \{\{p\}\} + \{\{p\} + \{\{p\}\} + \{\{p\} +\{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p \} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{\{p\}\} + \{p\}\}\}\] compute $pqr$.n Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 576 .</td></tr></table>

Table 36: Details of example $\pi _ { 1 7 }$ .   

<table><tr><td>Prompt:</td></tr><tr><td>In Class 3 (1), consisting of 45 students, all students participate in the tug-of-war. For the other three events, each student participates in at least one event. It is known that 39 students participate in the shuttlecock kicking competition and 28 students participate in the basketball shooting competition. How many students participate in all three events? Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 22 .</td></tr></table>

Table 37: Details of example $\pi _ { 6 0 5 }$   

<table><tr><td>Prompt:</td></tr><tr><td>Given vectors $$$\overline{overrightarrow{m}}=(\ \sqrt{sqrt{3}}\{\{3\}\} \sin x+\ \cos x,1), \ \overline{overrightarrow{m}}\{\{n\}=(\ \cos x,-f(x)), \ \overline{overrightarrow{m}}\{\{m\}\} \perp p \ \overline{overrightarrow{m}}\{\{n\}\} \$\$\. \n(1) Find the monotonic intervals of $f(x)$;\n(2) Given that $A$ is an internal angle of $ \triangle triangle ABC$, and $f\left( A \right\} \{2\} \left\} \left( \right) = \ \frac{1}{4} \left( \right) + \ \frac{1}{4} \left( \right) + \ \frac{1}{4} \left( \right) + \ \frac{1}{4} \left( \right) + \ \frac{1}{4} \left( \right) + \ \frac{1}{4} \left( \right) + \ \frac{1}{4} \left( \right) + \ \frac{1}{4} \ \left( \right) + \ \frac{1}{4} \ \left( \right) + \ \frac{1}{4} \ \left( \right) + \ \frac{1}{4} \ \left( \right) + \ \frac{1}{4} \ \left( \right) + \ \frac{1}{4} \ \left( \right) + \ \frac{1}{4} \ \left( 2 \right) \$\$\. find the area of $ \triangle triangle ABC$. Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): $\frac{\sqrt{3}-1}{4}$.</td></tr></table>

Table 38: Details of example $\pi _ { 6 0 6 }$   

<table><tr><td>Prompt:</td></tr><tr><td>How many zeros are at the end of the product \s( s(1) \cdots s(2) \cdots s(100) \), where \s( s(n) \) denotes the sum of the digits of the natural number \((n\))? Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 19.</td></tr></table>

Table 39: Details of example $\pi _ { 1 2 0 1 }$   

<table><tr><td>Prompt:</td></tr><tr><td>The angles of quadrilateral $PQRS$ satisfy $\angle \angle P = 3\angle \angle Q = 4\angle \angle R = 6\angle \angle S$. What is the degree measure of $\angle angle P$? Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 206 .</td></tr></table>

Table 40: Details of example $\pi _ { 1 2 0 7 }$ . A correct answer for this question should be 2/3.   

<table><tr><td>Prompt:</td></tr><tr><td>A rectangular piece of paper whose length is $\sqrt{3}$ times the width has area $A$. The paper is divided into three equal sections along the opposite lengths, and then a dotted line is drawn from the first divider to the second divider on the opposite side as shown. The paper is then folded flat along this dotted line to create a new shape with area $B$. What is the ratio $\sqrt{3}$(B)-(A)? Let&#x27;s think step by step and output the final answer within \boxed{} .</td></tr><tr><td>Ground truth (label in DSR-sub): 4/5 .</td></tr></table>

Table 41: Details of example $\pi _ { 1 2 0 8 }$   

<table><tr><td>Prompt:</td></tr><tr><td>Given a quadratic function in terms of \\\\x\\\\x, \\\\f(x)=ax-2[-4bx+1\\\x/.\\x(1)\\\x/.\\x Let set \\\\x\\\\x{1,2,3\\\x/}\\x and \\\\x(Q=\\\x{-1,1,2,3,4\\\x/}\\x, randomly pick a number from set \\\\x\\\\x(1\\\x/)\( as \\\\x\\\\x(a\\\x/)\( and from set \\\\x\\\\x(Q\\\x/)\( as \\\\x\\\\x(b\\\x/)\( , calculate the probability that the function \\\\x\\\\x(y=f(x)\\\x/)\( is increasing in the interval \\\\x\\\\x([1,+\\\x/\infty)/\x\\\\x/(2)\\\x/)\( Suppose point \\\\x\\\\x((a,b)\\\x/)\( is a random point within the region defined by \\\\x\\\\x begin{cases} x+y-8\\\x/leqslant 0 \\\\x\\\\x x&gt;0\\\x/\\\x y&gt;0\\\x/\\\x end{cases}\\\\x/), denote \\\\x\\\\x(A=\\\x{y=f(x)\\\x/}\( has two zeros, one greater than \\\\x\\\\x(1\\\x/)\( and the other less than \\\\x\\\\x(1\\\x/\\\x/), calculate the probability of event \\\\x\\\\x(A\\\x/)\( occurring. Let&#x27;s think step by step and output the final answer within \\\\boxed{\}.</td></tr><tr><td>Ground truth (label in DSR-sub): 961/1280.</td></tr></table>

Table 42: Details of example $\pi _ { 1 2 0 9 }$   

<table><tr><td>Prompt:</td></tr><tr><td>Define the derivative of the $(n-1)$th derivative as the $n$th derivative $(n \in N^{\sim}\{*\}, n \backslash geqlant 2)$, that is, $f^{\sim}((n))(x)=[f^{\sim}((n-1))(x)]&#x27;. They are denoted as $f&#x27;(x)$, $f&#x27;&#x27;(x)$, $f^{\sim}((4)(x)$, ..., $f^{\sim}((n))(x)$. If $f(x) = xe^{\sim}\{x\}$, then the $2023$rd derivative of the function $f(x)$ at the point $(0, f^{\sim}((2023))(0)$ has a $y-axis intercept on the $x-axis of _____. Let&#x27;s think step by step and output the final answer within \boxed{}$.</td></tr><tr><td>Ground truth (label in DSR-sub): -2023/2024.</td></tr></table>
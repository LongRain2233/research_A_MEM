# Long-VMNet: Accelerating Long-Form Video Understanding via Fixed Memory

Saket Gurukar

Samsung Research America

s.gurukar@samsung.com

Asim Kadav*

asim.kadav@gmail.com

# Abstract

Long-form video understanding is essential for various applications such as video retrieval, summarizing, and question answering. Yet, traditional approaches demand substantial computing power and are often bottlenecked by GPU memory. To tackle this issue, we present Long-Video Memory Network, Long-VMNet, a novel video understanding method that employs a fixed-size memory representation to store discriminative patches sampled from the input video. Long-VMNet achieves improved efficiency by leveraging a neural sampler that identifies discriminative tokens. Additionally, Long-VMNet only needs one scan through the video, greatly boosting efficiency. Our results on the Rest-ADL dataset demonstrate an $I 8 x - 7 5 x$ improvement in inference times for long-form video retrieval and answering questions, with a competitive predictive performance.

# 1. Introduction

Long-form video understanding is important for various applications such as video retrieval, summarizing, and question answering. For example, for automated checkout in retail applications, a video understanding system needs to understand the temporal order of important actions such as grabbing an object, and limit the attention to actions such as browsing items, and interacting with other people, to efficiently process long duration shopping videos [40, 48].

Modern methods for long-form video understanding such as transformer based models can be inefficient and require significant computational resources [52]. These methods often require building an intermediate representation for the entire video in memory and consume large amounts of GPU memory limiting the maximum length of videos that can be processed, especially for understanding tasks that require joint spatio-temporal analysis of different scene elements [16]. These tasks necessitate understanding approaches that repeatedly access and manipulate different scene elements, as dictated by complex intermedi-

![](images/c59a222be04597035066729ce9dbb0dfa3dc92209bbb2d5d23968d957e928a11.jpg)  
Figure 1. shows a activity query video and a histogram of the number of times a single frame is reloaded in GPU memory for the video from the video understanding, ReST-ADL dataset $\mathrm { ( F P S { = } 1 ) }$ ).

ate computational or scene graphs [13, 24]. Furthermore, when these understanding models are scaled, such as using Vision-Language Models (VLMs), they require even more compute and GPU memory [5]. Hence, given a limited compute budget, VLMs only operate on a few images and often struggle to efficiently perform dense understanding of videos, limiting video sizes to a few minutes [50].

Even though several efficient approaches exist, these methods often sample a fixed number of frames [30] affecting model performance for certain actions, or perform a clip based aggregation losing the order of short term actions [10]. Existing token sampling and pruning methods condense background tokens in the spatial domain, and do not store or re-use tokens in memory that can affect efficiency for dense spatio-temporal tasks [4, 12].

Figure 1 illustrates the computational challenge associated with video understanding models. As shown in Figure 1, TubeDETR [56] repeatedly processes a large number of frames while handling approximately 6000 activity queries on long videos during inference. The average duration of these long videos spans is 27 minutes. A histogram

of the frame processing counts is presented in Figure 1. This observation demonstrates an opportunity for memory-based approaches, which load the frames of long videos into GPU memory, to improve computational efficiency.

To address the above issues, we present Long-VMNet which uses a fixed-size memory representation to identify and store discriminative patches sampled from the input video. The memory patches are identified using a neural sampler, that improves efficiency while maintaining the discriminability of memory representation. Additionally, Long-VMNet only requires a single pass of the video over the memory, further improving the overall efficiency.

In our results over the Rest-ADL dataset, we demonstrate an 18X speedup during inference with 1 FPS and and 75x improvement with 5 FPS, for long form video retrieval for answering questions over long $( > 3 0 \mathrm { m i n }$ ) videos to answer activity, object, and temporal queries and achieve competitive performance.

# 2. Related Work

Token efficiency methods In order to reduce token costs, approaches such as token merging [4, 43], adaptive token sampling in classification domain [12], token turing machines [36], spatio-temporal token selection [46] have been explored. BLIP-3 [55] uses Perceiver based token sampler to project input image to a fixed number of tokens. However, token pruning methods often can deduplicate tokens whereas, Long-VMNet identifies discriminative tokens using a neural sampler. Crucially, Long-VMNet uses a fixed memory, re-using token representations across queries significantly reducing inference time.

Video understanding Deep learning based video understanding methods have evolved from using 3D convolution based methods [25] to 2D-CNNs [9, 14, 38], with additional blocks such as object/ROI features [19, 30], convolution-transformer approaches [18] and transformeronly approaches [1, 3, 11, 29]. Transformer based approaches often tokenize the video by chopping the input into a series of 2D spatial frames or into 3D spatio-temporal cubes on a regular grid. This approach can provide high accuracy but requires significant amounts of compute and memory due to large number of tokens and their parallel processing in transformer architecture. In contrast, our method uses transformer based tokens and samples the tokens to significantly reduce processing costs.

Long-form video understanding Long-form video understanding approaches have been studied [39, 41, 47, 52, 53]. MeMViT caches representations of previous clips to extend temporal context without increasing per-step compute. However, these approaches are often limited to less

than few minutes, largely due to lack of long form video datasets $\mathrm { \Omega } > 3 0 \mathrm { m i n s }$ ). Additionally, existing understanding based datasets largely focus around QA [58] or temporal action retrieval [6, 20, 59]. In contrast, our paper is focused on videos with duration of $3 0 +$ mins with a focus on tasks that require a joint analysis of activities, objects and time, which requires complex understanding.

Efficient transformer architectures Efficient transformer architectures have focused on reducing the cost of quadratic attention costs with respect to sequence lengths [42], pruning [31, 33, 45] and reduction of vision tokens as an input to decoders downstream. Previous work has analyzed sparse attention patterns to reduce complexity from attention to linear [2, 60], and approximated attention using kernel methods achieving linear time and memory complexity [8, 37]. Many hierarchical approaches use a hierarchical token structure to process inputs at multiple resolutions, reducing overall computation [15, 23, 27].

Long-context VLMs Most VLMs processes only a few minutes of videos due to limited context length. Video processing requires a large number of tokens to be processed; for example, deploying a 7B Llama model that can process 10 million tokens requires 8 A100 GPUs (640GB memory), even with advanced serving optimizations [22]. Even larger proprietary models such as Gemini 1.5 Pro can process 10 million tokens which roughly translates to approximately 10 hours of video duration [34]. Gemini 1.5 model architecture and number of parameters are unknown. However, Gemini 1.5 model is most likely compute intensive – based on its API pricing [17]. HEM-LLM [7] designs novel adaptive sequence segmentation scheme to segment multiple events in long videos and relies to VLM for video understanding. In contrast, Long-VMNet consists of around 300 million parameters $\leq 1$ GB with FP16) and can process a single 10 hour video into a fixed sized memory $\leq 1$ GB with FP16). Long-VMNet can be deployed on edge device.

# 3. Preliminaries

We use the Relational Space-Time Query (ReST) dataset [57] to evaluate long-form video understanding. ReST consists of three kinds of relational space-time queries: activity query, object query, and time-query. Each query asks questions on a single property (e.g. activity) by providing the other two properties (e.g. object and time) as input. The templates of queries are as follows:

1. Activity query - what activities did I perform with a particular object during a given time?   
2. Object query - on which objects I perform with a particular activity during a given time?

![](images/928b4e472e482e8fd860f4b195ea82beca52d339f66b4ca0ea005a799ed820af.jpg)  
Figure 2. Overview of the Long-VMNet training : We sample tokens from input videos and store them in memory to efficiently process long-form videos. The inference steps are shown in the Figure 3.

3. Time query - at what time did I perform a particular activity with a particular object?

ReST consists of long videos with average duration of 27 minutes in length. The relational space-time queries over the videos are further categorized into three types based on query time duration – short (around 5 minutes), medium (around 15 minutes), and long (around 30 minutes). Note that the short queries in our paper are longer than those typically employed by existing models, which usually last about 3 minutes [57].

There are four-time representations in our setup: long video time $( v _ { s } , v _ { e } )$ , ReST query time $( q _ { s } , q _ { e } )$ , time-property time $( t _ { s } , t _ { e } )$ and clip time $\left( c _ { s } , c _ { e } \right)$ . The long video time represents the complete duration of the input video clip (up to an hour long). The ReST query time $( q _ { s } , q _ { e } )$ represents the duration of the relational space time query. The ReST query time has the following constraint: $v _ { s } \ \leq \ q _ { s } \ \leq \ q _ { e } \ \leq \ v _ { e }$ . The time-property time $( t _ { s } , t _ { e } )$ of a query is the duration when an activity occurs on an object within query time $( q _ { s } , q _ { e } )$ . The time-property time has following constraint: $v _ { s } \leq q _ { s } \leq t _ { s } \leq t _ { e } \leq q _ { e } \leq v _ { e }$ . The clip time represents the sampled clip from a long video such that the sampled frames from the clip can be loaded into the available GPU memory. It has following constraint: $v _ { s } \leq c _ { s } \leq c _ { e } \leq v _ { e }$ . The ReST dataset contains multiple queries per video where $q _ { j } ^ { i }$ represents the $j ^ { t h }$ query in the ReST dataset that belongs to video $i$ .

# 4. Methodology

We refer to video understanding as a model’s ability to understand three properties: what activity is being performed on what object over what time. We test a model’s video understanding ability by asking queries where we provide input two properties of the video and then ask the model to predict the third property.

To build an efficient architecture, we draw inspiration from human memory that uses multiple memory representations and uses attention as a gatekeeper for the memory, guided by the high level goals [21, 49]. Long-VMNet performs understanding over long videos by using attention to store specific information within a fixed memory. It achieves this through a trained neural sampler that extracts discriminative visual patches from the video and stores them in memory as shown in Figure 2. During inference, Long-VMNet uses this pre-populated memory to respond to queries without needing to revisit the original video, significantly reducing inference time. This architecture is wellsuited for answering multiple queries from a single video, enabling fast and effective video understanding.

# Long-VMNet Input Encoding

Activity, Object, and Time properties in ReST are represented in three different representation spaces: activity is represented as one of $\mathcal { C }$ classes, the object is represented as an image, and time is represented by start and end times.

The activity input is represented as a one-dimensional vector $a _ { j } ~ \in ~ \mathbb { R } ^ { 1 \times \mathcal { C } }$ . This vector is then passed through a feed-forward layer to obtain a $d$ -dimensional representation $a _ { j } ^ { h } \in \mathbb { R } ^ { 1 \times \bar { d } }$ . The object input is an instance image $o _ { j } \in \breve { \mathbb R } ^ { o _ { h } \times o _ { w } }$ . The image is passed through a frozen image backbone (pre-trained Swin Transformer [28]). The time input $( t _ { j , s } , t _ { j , e } )$ from the ReST query is passed into a videolevel temporal positional encoding layer [44] that outputs a latent representation thj ∈ R(tj,e−tj,s+1)×d. $t _ { j } ^ { h } \in \mathbb { R } ^ { ( t _ { j , e } - \bar { t } _ { j , s } + 1 ) \times d }$

# Neural Sampler

The objective of the differentiable neural sampler is to populate memory with the representative visual tokens from the whole long video $v _ { i }$ . The memory stores a fixed number of $m _ { i } ~ \in ~ \mathbb { R } ^ { m \times d }$ visual tokens per video. The input to the neural sampler is $m _ { i }$ memory tokens and $k$ visual tokens from the sampled clip $c _ { j } ^ { i } \in \mathbb { R } ^ { T \times H \times W \times d }$ of the query $q _ { j } ^ { i }$ where $T , H$ , and $W$ are duration of clip, height and weight of the image frames, respectively. The neural sampler outputs $m$ discriminative visual tokens. The sampled clip $c _ { j } ^ { i } \mathbf { \bar { \Pi } } \in \mathbb { R } ^ { T \times H \times W \times d }$ is passed through a pre-trained, frozen Swin transformer based image backbone [28] that results in $k \in \mathbb { R } ^ { T h ^ { \prime } w ^ { \prime } \times d }$ visual tokens where $h ^ { \prime }$ and $w ^ { \prime }$ are computed based on patch size. We use the same image backbone for object image and clip frames. The sampled $m _ { i }$ tokens are then passed through the 2D spatial positional encoding layer and video-level temporal positional encoding layer. Our proposed framework is independent of the choice of neural sampler [32, 54].

The input to the neural sampler is $k$ (clip) tokens and $m$ (memory) tokens. Inside neural sampler, we pass the $m + k$ tokens through a transformer encoder followed by a MLP layer that outputs the scores. The neural sampler [54] samples $m$ tokens where sampling is a discrete operation. To backpropagate the gradients to the encoder and MLP layer, the neural sampler adds Gumbel noise to the scores and utilizes reparameterization trick [26]. The sampler is trained based on ReST queries predictons where the loss is higher if the sampler samples non-discriminative tokens.

# Encoder-Decoder

The input $\boldsymbol { x } _ { j } ^ { i }$ to the transformer encoder includes $m _ { i } \in \mathsf { \Gamma }$ $\mathbb { R } ^ { m \times d }$ memory tokens along with the query specific input. For example, in the case of activity query, the input includes latent representation of instance image $o _ { j } ^ { h }$ so the input is $x _ { j } \in \mathbb { R } ^ { ( m + o h ^ { \prime } o w ^ { \prime } ) \times d }$ . Before passing the input $x _ { j }$ to the encoder, we perform element-wise multiplication of instance image and frame tokens. Let $n f$ be number of frames, then m = nf × (oh′ow′). Therefore, xj ∈ R((nf+1)⊙oh′ow′)×d. $m = n f \times ( o h ^ { \prime } o w ^ { \prime } )$ $x _ { j } \in \mathbb { R } ^ { ( ( n f + 1 ) \odot o h ^ { \prime } o w ^ { \prime } ) \times d }$ In case of object query, the input includes latent representation of activity $a _ { j } ^ { h }$ so input is and $x _ { j } \ \in \ \mathbb { R } ^ { ( m + 1 ) \times d }$ . In case of time query, the input includes both latent representation of activity $a _ { j } ^ { h }$ and instance image $o _ { j } ^ { h }$ so input is

$x _ { j } \in \mathbb { R } ^ { ( m + o h ^ { \prime } o w ^ { \prime } + 1 ) \times d }$ and after element-wise multiplication $x _ { j } \ \in \ \mathbb { R } ^ { ( ( ( n f + 1 ) \odot o h ^ { \prime } o w ^ { \prime } ) + 1 ) \times d }$ . The transformer decoder accepts input in the form of key and value representations from the transformer encoder. The queries input to the transformer decoder are initialized based on videolevel temporal positional encoding, with an input given by $t _ { j } ^ { h } \ \in \ \mathbb { R } ^ { \left( \hat { t } _ { j , e } - \hat { t _ { j , s } } + 1 \right) \times d }$ . The output of the decoder is a learned representation of $\hat { t } _ { j } ^ { h }$ , which are learned by contextualizing memory tokens and the ReST queries’ input representations through the use of time query representations.

# Long-VMNet Output Encoding

The learned representation of $\hat { t } _ { j } ^ { h }$ from the previous step is used for prediction tasks. The prediction is carried out using a query-specific multi-layer perceptron (MLP) head. For an activity query, we apply mean pooling to $\hat { t } _ { j } ^ { h } \ \in \ \mathbb { R } ^ { ( t _ { j , e } - t _ { j , s } + 1 ) \times \bar { d } }$ to obtain $\hat { t } ^ { \prime } \in \mathbb { R } ^ { 1 \times d }$ . This representation is then fed into an activity prediction MLP, which predicts the activities $\hat { a } ~ \in ~ \mathbb { R } ^ { 1 \times \dot { \mathcal { C } } }$ . For an object query, bounding box predictions are computed for each sampled frame. To do this, we pass the learned query representation $\hat { t } _ { j } ^ { h } \in \mathbb { R } ^ { ( t _ { j , e } - t _ { j , s } + 1 ) \times d }$ through a specific MLP layer tailored for object queries. This object-specific MLP layer predicts normalized bounding boxes $\hat { o } _ { j } \in \mathbb { R } ^ { ( t _ { j , e } - t _ { j , s } + 1 ) \times 4 }$ for each sampled frame. In the case of a time query, we pass the learned query representation $\hat { t } _ { j } ^ { h } \in \mathbb { R } ^ { ( t _ { j , e } - t _ { j , s } + 1 ) \times d }$ through two separate MLP layers to predict start and end times.

# Memory Read/Write operations

In Long-VMNet the memory $m _ { i }$ is allocated per long video $v _ { i }$ . A long video $v _ { i }$ can possibly have multiple associated ReST queries where each query could focus on a clip (for example, $c _ { j } ^ { i }$ which corresponds to $j ^ { t h }$ clip of video $v _ { i }$ ). Since we train our model through sampled clips it becomes important, how we form a batch through a data sampler. A naive data sampler can form a batch with two or more clips belonging to the same video. In this case, the neural sampler would read $m$ tokens from video $v _ { i }$ and $k$ tokens from each clip $c _ { j } ^ { i }$ (say $c _ { j 1 } ^ { i }$ and $c _ { j 2 } ^ { i }$ ). The neural sampler would then output $m$ tokens for $c _ { j 1 } ^ { i }$ and $m$ tokens for $c _ { j 2 } ^ { i }$ to be written to the $i ^ { t h }$ video memory slot thereby creating a race condition 1.

To avoid this race condition on a single GPU, we design a data sampler such that a batch has ReST queries with no two queries belonging to the same long video. In distributed training with multiple GPUs, our data sampler ensures that all the batches have ReST queries with no two queries belonging to the same video. This data-sampling constraint ensures there is no memory corruption. At the end of each iteration, the written memory tokens are synchronized across all devices. In distributed training setup

![](images/165e77f38f1a41cfcb34f5a6a0436f4b3ffa2f7aa3167f728fcb31bec3dcf1de.jpg)  
(a) Inference Stage 1: The whole long video is passed clip by clip through the trained neural sampler which populates the memory.

![](images/0399836664a41f65f96f5258c1c623b2a50b6a9effebcd160bb699186611ef3b.jpg)  
(b) Inference Stage 2: The ReST queries responses are predicted by our trained model by only reviewing the pre-computed memory tokens.   
Figure 3. Two stage Inference pipeline of Long-VMNet .

with $r$ devices and $n$ number of videos, the maximum batch size on a single GPU becomes $\frac { n } { r }$ to avoid race condition.

# Inference

The inference of Long-VMNet enables faster processing of queries than existing long video understanding models. In existing models such as TubeDETR [56], for answering $q$ queries from a single video, one has to pass the query clip’s frames, $q$ number of times. The clip processing – passing the clip frames through the image backbone and then passing the latent representations through encoder-decoder modules – is compute intensive and results in a significant delay in generating the query’s response. Moreover, $q$ queries are processed independently so if multiple queries share a small region of clip, there is no potential to offset the

clip processing load. In contrast, in our proposed model, as shown in the Figure 3, we first populate video $v _ { i }$ specific memory $m _ { i }$ through our trained neural sampler. All the responses to the queries that belong to video $v _ { i }$ are generated using sampled memory $m _ { i }$ .

The memory $m _ { i } \in \mathbb { R } ^ { ( m , d ) } - m$ is the number of tokens and $d$ represents latent dimension of image backbone – for a particular video $v _ { i }$ is populated as follows: we first initialize $m _ { i }$ with video tokens sampled randomly. We then extract clips from video $v _ { i }$ through a sliding window with two clips having zero overlap. Each clip is then passed through the image backbone that outputs $k$ tokens. The neural sampler takes $m$ memory tokens and $k$ clip tokens and outputs $m$ tokens that are written to memory. The populated $m _ { i }$ is fed to the encoder for answering queries belonging to $i ^ { t h }$ video.

An advantage of our proposed model is that it can be deployed on an edge device with limited memory. The inference can be performed in a streaming fashion where we can store $m$ memory tokens and $k$ tokens from the current query clip. The sampler here would take input $m + k$ tokens and output $m$ discriminative tokens. These $m$ memory tokens are used to generate responses for multiple queries.

# Training loss

The input training data is the ReST query’s clip frames. In the case of activity query, the input is object instance and time-property time $( t _ { s } , t _ { e } )$ . The task in an activity query is to predict the activity from the available $C$ classes. In an activity query, multiple activities can happen on an object instance within time-property time, so we model the activity prediction as a multi-label classification. We use focal loss $( a , \hat { a } )$ where $a \in \mathbb { R } ^ { 1 \times \mathcal { C } }$ represents ground truth.

In the case of object query, the input is two properties: activity class and time-property time $( t _ { s } , t _ { e } )$ . The task in the object query is to predict bounding boxes for each sampled frame in $( t _ { s } , t _ { e } )$ . Given ground truth bounding boxes, o where $o \in [ 0 , 1 ] ^ { 4 ( t _ { e } - t _ { s } + 1 ) }$ and predicted bounding boxes $\hat { o }$ the object query loss is given as

$$
\sum_ {i \in \text {o b j e c t - q u e r i e s}} \lambda_ {1} \mathcal {L} _ {1} (\hat {o} _ {j}, o _ {j}) + \lambda_ {g I o U} \mathcal {L} _ {g I o U} (\hat {o} _ {j}, o _ {j}) \tag {1}
$$

where $\mathcal { L } _ { 1 }$ is $\mathcal { L } _ { 1 }$ loss on bounding boxes coordinates and $\mathcal { L } _ { g I o U }$ is generalized intersection over union loss on the bounding boxes [35]. $\lambda _ { 1 }$ and $\lambda _ { g I o U }$ are scalar weights.

In the case of time query, the input is three properties: activity class, object instance, and query time $( q t _ { s } , q t _ { e } )$ . The task in time query is to predict the time-property time $( t _ { s } , t _ { e } )$ within $( q t _ { s } , q t _ { e } )$ . The ground truth is represented through two vectors – $\boldsymbol { v } _ { t _ { s } } ~ \in ~ \mathbb { R } ^ { 1 \times l }$ for start time $t _ { s }$ and $v _ { t _ { e } } \in \mathbb { R } ^ { 1 \times l }$ for end time $t _ { e }$ . Here, $l$ is set to $( t _ { e } - t _ { s } ) / { \mathrm { t a r g e t } }$ - fps. We compute Cross Entropy loss $\mathcal { L } _ { C E } ( \hat { v } _ { t _ { s } } , v _ { t _ { s } } ) \ + $ $\mathcal { L } _ { C E } ( \hat { v } _ { t _ { e } } , v _ { t _ { e } } )$ for training the model on time query.

# Online Continual learning loss

We train the neural sampler based on the above described training loss. However, with this training setup, the neural sampler is biased towards sampling $q _ { j } ^ { i }$ ’s clip visual tokens instead of memory tokens – since the training loss computed on $q _ { j } ^ { i }$ ’s predictions is minimized by sampling visual tokens from the $q _ { j } ^ { i }$ ’s clip. Hence, the model cannot identify tokens that capture the global view of the long video. This bias contrasts against our goal of training a neural sampler that would process the long video once and populate discriminative tokens into memory.

We propose online continual learning loss to address this sampling bias (shown in Figure 4). Here, we store past $p$ ReST queries in a heap of size $p$ where the oldest query is ejected when the heap is full. The query $q _ { j } ^ { i }$ and past $p$ ReST

queries belonging to the video $v _ { i }$ are passed through the shared transformer encoder-decoder and the training loss is computed on both the current query’s predictions and past $p$ query’s predictions. This auxiliary loss addresses the sampling bias of the neural sampler. We make a design choice of performing the continual learning in an online fashion – training based on recent past $p$ queries – instead of randomly sampling $p$ queries from all the previous queries. This is because the initial probability of past $p$ queries’ relevant tokens being in memory is high. With online continual learning, we reinforce the neural sampler to give those relevant tokens high scores regardless of their relevance to the current query $q _ { j } ^ { i }$ .

# Experiments

We performed experiments on the ReST-ADL dataset. The queries in the test set focus on the long-videos which are not seen during training.

# Evaluation metrics

We follow [57] and use recall $@ 1 \mathbf { x }$ metric for evaluation. The metric measures the percentage of ground truth labels identified in top $x$ predictions where $x$ stands for the number of ground truth predictions. In case of object query, we follow [56] and define $\begin{array} { r l } { { v I o U } _ { j } } & { { } = } \end{array}$ Su $\begin{array} { r } { \frac { 1 } { S _ { \ast } } \sum _ { f \in t _ { e } - t _ { s } + 1 } I o U ( \hat { o } _ { j , f } , o _ { j , f } ) } \end{array}$ . The prediction is positive if $v I o U _ { j } ~ > ~ R$ otherwise a zero value is assigned to the prediction. Following [56], we set $R \ : = \ : 0 . 3$ . For time query, we again compute $t I o U$ using ground truth start-end time and predicted start-end time. A prediction is positive if $t I o U _ { j } > 0 . 3$ otherwise a zero value is assigned to the prediction.

# Baselines

We compare our proposed method with the ReST [57] that uses a multi-stage differentiable learning model and endto-end TubeDETR method [56]. We modify the last MLP layer of TubeDETR for activity prediction outputs. Tube-DETR operates on clips that can be loaded into GPU memory. For clips with durations greater than 4 minutes (1 FPS), TubeDETR requires sampling a fixed number of frames to meet GPU memory requirements. We follow the clip-based training and inference recipe outlined in the PyTorchVideo library [10] for TubeDETR. Specifically, during training, given a long clip, we randomly sample a sub-clip whose duration, with the selected FPS, results in a predefined fixed number of frames. During inference, we follow these steps: Divide the long clip into non-overlapping short clips, pass each short clip along with the object image through the trained TubeDETR model, which outputs the activity class logits and aggregate the logits across all clips and perform prediction.

![](images/5f70a2a725e7b1a9cb9c6494a819e531a78f975a9c70dda44146d0a9b91afaf0.jpg)  
Figure 4. Auxillary online continual learning loss (shown in red color).

Table 1. Running time: Benchmarked over a single A100 with inference batch size selected to maximize 80GB GPU memory for both methods. The Target FPS is set to one for activity and time query and set to five for object query as ground truth is available at five FPS for object query.   

<table><tr><td></td><td>Activity Query</td><td>Object Query</td><td>Time Query</td></tr><tr><td colspan="4">Short Queries</td></tr><tr><td>Modified TubeDETR</td><td>264 mins</td><td>99 mins</td><td>11 mins</td></tr><tr><td>Long-VMNet</td><td>14 mins (18x)</td><td>6 mins (16.5x)</td><td>7 mins</td></tr><tr><td colspan="4">Medium Queries</td></tr><tr><td>Modified TubeDETR</td><td>180 mins</td><td>663 mins</td><td>31 mins</td></tr><tr><td>Long-VMNet</td><td>16 mins (11.2x)</td><td>15 mins (44x)</td><td>14 mins</td></tr><tr><td colspan="4">Long Queries</td></tr><tr><td>Modified TubeDETR</td><td>174 mins</td><td>756 mins</td><td>19 mins</td></tr><tr><td>Long-VMNet</td><td>15 mins (11.6x)</td><td>10 mins (75x)</td><td>10 mins</td></tr></table>

In the case of the object query, during training, we apply object detection loss mentioned in the Equation 1 on the sampled clip. During inference, we divide the long clip of query $q _ { j }$ into non-overlapping short clips, pass the activity hidden representation along with each clip and compute the $v I o U _ { j }$ score per query.

In the case of the time query, where the task requires predicting the start and end times, we follow TubeDETR and sample a fixed number of frames [56].

# Results

We report our experimental results in Tables 1 and 2. We perform the inference running time computation on the identical A100 instances. For running time of Long-VMNet, we include the time taken by both stages of inference. The source code of ReST method (with any optimizations) is not publicly available. Batch size for both the methods is selected to maximize the GPU memory utilization. As shown in Table 1, Long-VMNet, outperforms the TubeDETR model in terms of inference speed. In the case of activity query, Long-VMNet achieves speedups of 18X, 11.2X, and 11.6X over TubeDETR on short, medium, and long activity queries, respectively. When processing the ReST-ADL dataset, which consists of approximately 6000 test activity queries across four long videos, Long-VMNet passes each video through its neural sampler once to create four video-specific memories. All subsequent test queries are then processed using this memory, resulting in efficiency gains.

In contrast, TubeDETR treats each query independently and requires a separate inference process for every query, as outlined in the baselines section. This approach leads to redundant processing of frames when multiple queries refer to overlapping or identical long clips. While it might be possible to optimize this by processing all frame representations once and storing them on disk, this approach would still require additional steps: i. Divide long video clip into non-overlapping short clips ii. Load these clips’ frame rep-

Table 2. Prediction Performance (Recall@1x) over short, medium, and long queries using ReST, TubeDETR, and Long-VMNet.   

<table><tr><td></td><td>Activity Query</td><td>Object Query</td><td>Time Query</td></tr><tr><td colspan="4">Short Queries</td></tr><tr><td>ReST system</td><td>48.1</td><td>9.6</td><td>31.3</td></tr><tr><td>Modified TubeDETR</td><td>45.3</td><td>27.5</td><td>35.0</td></tr><tr><td>Long-VMNet</td><td>32.4</td><td>26.4</td><td>22.9</td></tr><tr><td colspan="4">Medium Queries</td></tr><tr><td>ReST system</td><td>50.7</td><td>10.0</td><td>31.8</td></tr><tr><td>Modified TubeDETR</td><td>31.6</td><td>25.4</td><td>6.7</td></tr><tr><td>Long-VMNet</td><td>26.1</td><td>11.9</td><td>11.9</td></tr><tr><td colspan="4">Long Queries</td></tr><tr><td>ReST system</td><td>46.3</td><td>10.0</td><td>30.0</td></tr><tr><td>Modified TubeDETR</td><td>29.9</td><td>24.6</td><td>12.8</td></tr><tr><td>Long-VMNet</td><td>22.8</td><td>21.3</td><td>8.6</td></tr></table>

resentations in memory iii. Pass them along with the object image through the trained TubeDETR model to obtain logits and iv. Aggregate these logits. The latter steps are particularly time-consuming and result in slow inference times.

In the case of object query, the ground truth is available at five FPS, hence the target fps is set to five for all the models. We observe 75x improvement in inference speed as compared to TubeDETR since at five FPS, TubeDETR has to process 5X more frames. From Table 2, we observe Long-VMNet performs competitively as compared to Tube-DETR.

In the case of the time query, due to the nature of predicting start and end times, we follow TubeDETR and sample a fixed number of frames [56]. We observe that the modified TubeDETR had a shorter running time for time queries compared to activity and object queries due to frame sampling. However, for medium (15-minute) and long (30- minute) queries, the performance of the modified Tube-DETR deteriorates because it is not explicitly designed for long videos [56]. In contrast, Long-VMNet stores a global view of each video in memory and passes this along with object images through its trained encoder-decoder model, which outputs activity class logits without requiring any additional aggregation. As shown in Table 2, our system performs competitively compared to other methods.

# 4.1. Ablation studies

# 4.1.1. Random sampling video tokens vs sampler

We study the impact of the neural sampler in sampling video tokens as compared to the uniform sampling of tokens in Table 3. We quantitatively show that the neural sampler is able to identify discriminative tokens as compared to a uniform random sampling of tokens. The uniform random sampling would sample a lot of background tokens as compared to the trained neural sampler thereby resulting in a significant

<table><tr><td></td><td>Recall@1x</td><td>Recall@3x</td></tr><tr><td colspan="3">Short Queries</td></tr><tr><td>Long-VMNet</td><td>32.38</td><td>56.78</td></tr><tr><td>Long-VMNet-random</td><td>21.42</td><td>43.20</td></tr><tr><td colspan="3">Medium Queries</td></tr><tr><td>Long-VMNet</td><td>26.12</td><td>44.80</td></tr><tr><td>Long-VMNet-random</td><td>18.23</td><td>40.57</td></tr><tr><td colspan="3">Long Queries</td></tr><tr><td>Long-VMNet</td><td>22.81</td><td>45.39</td></tr><tr><td>Long-VMNet-random</td><td>18.42</td><td>38.45</td></tr></table>

Table 3. Activity Query: Neural vs random sampling.   
Table 4. Activity Query: Continual vs Non-Continual learning   

<table><tr><td></td><td>Recall@1x</td><td>Recall@3x</td></tr><tr><td colspan="3">Short Queries</td></tr><tr><td>Long-VMNet</td><td>32.38</td><td>56.78</td></tr><tr><td>Long-VMNet-non-continual</td><td>26.39</td><td>47.31</td></tr><tr><td colspan="3">Medium Queries</td></tr><tr><td>Long-VMNet</td><td>26.12</td><td>44.80</td></tr><tr><td>Long-VMNet-non-continual</td><td>24.81</td><td>44.63</td></tr><tr><td colspan="3">Long Queries</td></tr><tr><td>Long-VMNet</td><td>22.81</td><td>45.39</td></tr><tr><td>Long-VMNet-non-continual</td><td>18.28</td><td>44.54</td></tr></table>

reduction in the predictive performance of activity query.

# 4.1.2. Continual Learning

We perform an ablation experiment where we report the performance of Long-VMNet with and without continual learning in Table 4. We can see that adding continual learning helps improve the performance of Long-VMNet in a significant manner.

# 4.2. Long-VMNet details

The temporal positional encoding layer is standard positional encoding [44] where the sequence length is set to the maximum long-video length in seconds times the target FPS. We set the target frame per second (FPS) to 1 for activity and time query while the target FPS is set to 5 for object query. We sample 120 number of frames set in a clip. We select the frozen pre-trained image backbone as Swin transformer [28]. We set the following hyper-parameters: $T = 1 2 0 , d = 2 0 4 8 , N = 2 , \mathcal { L } _ { 1 } = 5 , \lambda _ { g I o U } = 2 .$ . We train our model and baseline models for 10 epochs. The models are trained on 4 A100 GPUs with an effective batch size of 4. We initialize the parameters of Long-VMNet using modified TubeDETR. The learning rate of the neural sampler is set to 1e-5 while the rest of the parameters learning rate is set to 1e-7. We reset the memory bank after every

training epoch. We set the number of past continual learning queries $p$ value to 2. The memory size is set to 5880 tokens. The Swin transformer outputs 49 tokens per frame. With 120 number of frames, the number of clip tokens and memory tokens has the same capacity of tokens. The number of layers in all MLPs is set to 1. We use the TIMM library [51] for Swin transformer backbone with model id: swinv2 cr small ns 224. We perform data augmentations – horizontal flip, posterize, photometric distortion – with a probability of 0.25. The dropout value is set 0.2. To encourage exploration during the initial stage of neural sampler training, we set the temperature to 1.5 and slowly decrease the value of the temperature to 1.

# Conclusion

Long-form video understanding has been a long-standing challenge for the computer vision community. While many approaches exist, they cannot be efficiently applied to longer videos over 30 minutes. In this paper, we present Long-VMNet that demonstrates an efficient network for long-form video understanding using an external memory. The external memory is populated using differentiable neural sampler that samples tokens and builds an effective condensed representation. In our results, we demonstrate an 18-75X faster inference over the state of the art in the ReST ADL video understanding benchmark. The inference speedup is primarily due to the fact that our proposed Long-VMNet performs a single pass over a long video to populate memory and can provide answers to multiple queries from the long video using the populated memory.

# References

[1] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luciˇ c, and Cordelia Schmid. Vivit: ´ A video vision transformer. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6836–6846, 2021. 2   
[2] Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020. 2   
[3] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, volume 2, pp. 4, 2021. 2   
[4] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022. 1, 2   
[5] Florian Bordes, Richard Yuanzhe Pang, Anurag Ajay, Alexander C Li, Adrien Bardes, Suzanne Petryk, Oscar Manas, Zhiqiu Lin, Anas Mahmoud, Bargav Ja- ˜ yaraman, et al. An introduction to vision-language modeling. arXiv preprint arXiv:2405.17247, 2024. 1

[6] Yu-Wei Chao, Sudheendra Vijayanarasimhan, Bryan Seybold, David A Ross, Jia Deng, and Rahul Sukthankar. Rethinking the faster r-cnn architecture for temporal action localization. In CVPR, pp. 1130– 1139, 2018. 2   
[7] Dingxin Cheng, Mingda Li, Jingyu Liu, Yongxin Guo, Bin Jiang, Qingbin Liu, Xi Chen, and Bo Zhao. Enhancing long video understanding via hierarchical event-based memory. arXiv preprint arXiv:2409.06299, 2024. 2   
[8] Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, et al. Rethinking attention with performers. arXiv preprint arXiv:2009.14794, 2020. 2   
[9] Jeffrey Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell. Long-term recurrent convolutional networks for visual recognition and description. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2625–2634, 2015. 2   
[10] Haoqi Fan, Tullie Murrell, Heng Wang, Kalyan Vasudev Alwala, Yanghao Li, Yilei Li, Bo Xiong, Nikhila Ravi, Meng Li, Haichuan Yang, et al. Pytorchvideo: A deep learning library for video understanding. In Proceedings of the 29th ACM international conference on multimedia, pp. 3783–3786, 2021. 1, 6   
[11] Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, and Christoph Feichtenhofer. Multiscale vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 6824–6835, 2021. 2   
[12] Mohsen Fayyaz, Soroush Abbasi Koohpayegani, Farnoush Rezaei Jafari, Sunando Sengupta, Hamid Reza Vaezi Joze, Eric Sommerlade, Hamed Pirsiavash, and Jurgen Gall. Adaptive token sampling for ¨ efficient vision transformers. In European Conference on Computer Vision, pp. 396–414. Springer, 2022. 1, 2   
[13] Hao Fei, Shengqiong Wu, Wei Ji, Hanwang Zhang, Meishan Zhang, Mong-Li Lee, and Wynne Hsu. Video-of-thought: Step-by-step video reasoning from perception to cognition. In Forty-first International Conference on Machine Learning, 2024. 1   
[14] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2019. 2   
[15] Leo Feng, Frederick Tung, Hossein Hajimirsadeghi, Yoshua Bengio, and Mohamed Osama Ahmed. Tree

cross attention. arXiv preprint arXiv:2309.17388, 2023. 2   
[16] Quentin Fournier, Gaetan Marceau Caron, and Daniel ´ Aloise. A practical survey on faster and lighter transformers. ACM Computing Surveys, 55(14s):1–40, 2023. 1   
[17] GeminiAPI. Gemini api pricing. https://ai. google.dev/pricing, 2024. Accessed: Oct 1st, 2024. 2   
[18] Rohit Girdhar, Joao Carreira, Carl Doersch, and Andrew Zisserman. Video action transformer network. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 244–253, 2019. 2   
[19] Georgia Gkioxari, Ross Girshick, Piotr Dollar, and ´ Kaiming He. Detecting and recognizing human-object interactions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 8359– 8367, 2018. 2   
[20] Meera Hahn, Asim Kadav, James M Rehg, and Hans Peter Graf. Tripping through time: Efficient localization of activities in videos. arXiv preprint arXiv:1904.09936, 2019. 2   
[21] Thomas E Hazy, Michael J Frank, and Randall C O’Reilly. Banishing the homunculus: making working memory work. Neuroscience, 139(1):105–118, 2006. 3   
[22] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization. arXiv preprint arXiv:2401.18079, 2024. 2   
[23] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In International conference on machine learning, pp. 4651–4664. PMLR, 2021. 2   
[24] Jingwei Ji, Ranjay Krishna, Li Fei-Fei, and Juan Carlos Niebles. Action genome: Actions as compositions of spatio-temporal scene graphs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10236–10247, 2020. 1   
[25] Shuiwang Ji, Wei Xu, Ming Yang, and Kai Yu. 3d convolutional neural networks for human action recognition. IEEE transactions on pattern analysis and machine intelligence, 35(1):221–231, 2012. 2   
[26] Diederik P Kingma. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 4   
[27] Song Liu, Haoqi Fan, Shengsheng Qian, Yiru Chen, Wenkui Ding, and Zhongyuan Wang. Hit: Hierarchical transformer with momentum contrast for videotext retrieval. In Proceedings of the IEEE/CVF inter-

national conference on computer vision, pp. 11915– 11925, 2021. 2   
[28] Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 10012–10022, 2021. 4, 8   
[29] Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, and Han Hu. Video swin transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3202–3211, 2022. 2   
[30] Chih-Yao Ma, Asim Kadav, Iain Melvin, Zsolt Kira, Ghassan AlRegib, and Hans Peter Graf. Attend and interact: Higher-order object interactions for video understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6790– 6800, 2018. 1, 2   
[31] Lingchen Meng, Hengduo Li, Bor-Chun Chen, Shiyi Lan, Zuxuan Wu, Yu-Gang Jiang, and Ser-Nam Lim. Adavit: Adaptive vision transformers for efficient image recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12309–12318, 2022. 2   
[32] Adeel Pervez et al. Scalable subset sampling with neural conditional poisson networks. In ICLR, 2022. 4   
[33] Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, and Cho-Jui Hsieh. Dynamicvit: Efficient vision transformers with dynamic token sparsification. Advances in neural information processing systems, 34:13937–13949, 2021. 2   
[34] Machel Reid, Nikolay Savinov, Denis Teplyashin, Dmitry Lepikhin, Timothy Lillicrap, Jean-baptiste Alayrac, Radu Soricut, Angeliki Lazaridou, Orhan Firat, Julian Schrittwieser, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024. 2   
[35] Rezatofighi et al. Generalized intersection over union: A metric and a loss for bounding box regression. In CVPR, 2019. 6   
[36] Michael S Ryoo, Keerthana Gopalakrishnan, Kumara Kahatapitiya, Ted Xiao, Kanishka Rao, Austin Stone, Yao Lu, Julian Ibarz, and Anurag Arnab. Token turing machines. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19070–19081, 2023. 2   
[37] Imanol Schlag, Kazuki Irie, and Jurgen Schmidhuber. ¨ Linear transformers are secretly fast weight programmers. In International Conference on Machine Learning, pp. 9355–9366. PMLR, 2021. 2   
[38] Karen Simonyan and Andrew Zisserman. Twostream convolutional networks for action recognition

in videos. Advances in neural information processing systems, 27, 2014. 2   
[39] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tian Ye, Yanting Zhang, et al. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18221–18232, 2024. 2   
[40] Ombretta Strafforello, Klamer Schutte, and Jan Van Gemert. Are current long-term video understanding datasets long-term? In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2967–2976, 2023. 1   
[41] Yuchong Sun, Hongwei Xue, Ruihua Song, Bei Liu, Huan Yang, and Jianlong Fu. Long-form videolanguage pre-training with multimodal temporal contrastive learning. Advances in neural information processing systems, 35:38032–38045, 2022. 2   
[42] Yi Tay, Mostafa Dehghani, Dara Bahri, and Donald Metzler. Efficient transformers: A survey.(2020). arXiv preprint cs.LG/2009.06732, 2020. 2   
[43] Hoai-Chau Tran, Duy MH Nguyen, Duy M Nguyen, Trung-Tin Nguyen, Ngan Le, Pengtao Xie, Daniel Sonntag, James Y Zou, Binh T Nguyen, and Mathias Niepert. Accelerating transformers with spectrumpreserving token merging. NeurIPS, 2024. 2   
[44] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 4, 8   
[45] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head selfattention: Specialized heads do the heavy lifting, the rest can be pruned. arXiv preprint arXiv:1905.09418, 2019. 2   
[46] Junke Wang, Xitong Yang, Hengduo Li, Li Liu, Zuxuan Wu, and Yu-Gang Jiang. Efficient video transformers with spatial-temporal token selection. In European Conference on Computer Vision, pp. 69–86. Springer, 2022. 2   
[47] Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. arXiv preprint arXiv:2403.10517, 2024. 2   
[48] Kirti Wankhede, Bharati Wukkadada, and Vidhya Nadar. Just walk-out technology and its challenges: A case of amazon go. In 2018 International Conference on Inventive Research in Computing Applications (ICIRCA), pp. 254–257. IEEE, 2018. 1   
[49] Sebastian Watzl. Structuring mind: The nature of at-

tention and how it shapes consciousness. Oxford University Press, 2017. 3   
[50] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. arXiv preprint arXiv:2404.03384, 2024. 1   
[51] Ross Wightman. Pytorch image models. https:// github.com/rwightman/pytorch-imagemodels, 2019. 9   
[52] Chao-Yuan Wu and Philipp Krahenbuhl. Towards long-form video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 1884–1894, 2021. 1, 2   
[53] Chao-Yuan Wu, Yanghao Li, Karttikeya Mangalam, Haoqi Fan, Bo Xiong, Jitendra Malik, and Christoph Feichtenhofer. Memvit: Memory-augmented multiscale vision transformer for efficient long-term video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13587–13597, 2022. 2   
[54] Sang Michael Xie et al. Reparameterizable subset sampling via continuous relaxations. IJCAI, 2019. 4   
[55] Le Xue, Manli Shu, Anas Awadalla, Jun Wang, An Yan, Senthil Purushwalkam, Honglu Zhou, Viraj Prabhu, Yutong Dai, Michael S Ryoo, et al. xgen-mm (blip-3): A family of open large multimodal models. arXiv preprint arXiv:2408.08872, 2024. 2   
[56] Antoine Yang et al. Tubedetr: Spatio-temporal video grounding with transformers. In CVPR, 2022. 1, 5, 6, 7, 8   
[57] Xitong Yang, Fu-Jen Chu, Matt Feiszli, Raghav Goyal, Lorenzo Torresani, and Du Tran. Relational space-time query in long-form videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 6398–6408, 2023. 2, 3, 6   
[58] Kexin Yi, Chuang Gan, Yunzhu Li, Pushmeet Kohli, Jiajun Wu, Antonio Torralba, and Joshua B Tenenbaum. Clevrer: Collision events for video representation and reasoning. arXiv preprint arXiv:1910.01442, 2019. 2   
[59] Jun Yuan, Bingbing Ni, Xiaokang Yang, and Ashraf A Kassim. Temporal action localization with pyramid of score distribution features. In CVPR, pp. 3093–3102, 2016. 2   
[60] Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, et al. Big bird: Transformers for longer sequences. Advances in neural information processing systems, 33:17283–17297, 2020. 2
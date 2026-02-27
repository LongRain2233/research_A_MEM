# Context as Memory: Scene-Consistent Interactive Long Video Generation with Memory Retrieval

JIWEN $\mathsf { Y } \mathsf { U } ^ { \star }$ , The University of Hong Kong, China

JIANHONG BAI∗, Zhejiang University, China

YIRAN QIN, The University of Hong Kong, China

QUANDE LIU†, XINTAO WANG, PENGFEI WAN, and DI ZHANG, Kling Team, Kuaishou Technology, China XIHUI LIU†, The University of Hong Kong, China

![](images/5bc2a69697b29c2b16dd89041855d2e96aaa2f6188bd7d8132f64c5285a496c4.jpg)

![](images/e055631f25d8cb969355c204f0029c7df3c57b872a9aa35b175481425f2dc736.jpg)  
Frame 30

![](images/fcb1ce32d9d958b3726d37ed9b2819837d9dea665cd8251fb31ae0111269e756.jpg)  
Frame 254

![](images/0fd2b5f213ceee16ae6659de1b995187b9cb0091d309127aee9da593fb403c30.jpg)  
Frame 387

![](images/9694dd29e9e0c78e37d82153d91d13dd8b772298d81fae2912f3773dae59ef29.jpg)  
Frame 705

![](images/73ecf175b234883732eeecf81951a354cc3a715c141b5af6ff3d9625e448fc5a.jpg)  
Frame 919   
(1) Pr ompt: an industrial-style warehouse, oil drums, gas cylinders, a yellow-black construction vehicle, ...

![](images/42b92643b884f6ef4a27a6654b8c978095a17484c6408c925b8584137d37da59.jpg)  
Frame 2

![](images/6f9b0dd35c7b39bb61e549517d04ef46eb2cbd10acce9ed35f9b234a157da798.jpg)  
Frame 363

![](images/e8e458cfc4141968ef060e0a83915421adee9a04c278f7dee06827edf83c3a4c.jpg)  
Frame 776

![](images/751adbd0c805a399bc9a63b536d113919f4403d29a23590b653a4154bf3d1d34.jpg)  
Frame 1142

![](images/8fe55242c9ec91f8481be07c1a6770dd4959a7b250c45167226f6ce93e2d581c.jpg)  
Frame 1281   
(2) Pr ompt: An ancient temple with a reddish-brown stone structure and dome sits in a lush forested valley, surrounded by mountains,

![](images/d8b52f87c900cdda9a07942e033335576868f41aef653f3aa622949845eec265.jpg)  
Frame 2

![](images/8eff8b7176fa1d90dbb36be3261bee66b803868e88bf897c5d1d812cca24a1c2.jpg)  
Frame 428

![](images/f09079bcc22e13ab9c045532db429d2a4f1460c6918c5d7642ca92691dbb1957.jpg)  
Frame 683

![](images/d6834b123be170bc9c3a4ea94568c5cee01f5b7bc685a49941b4d776ffc1317d.jpg)  
Frame 853

![](images/db76827764a2a941bc81024c619a8e6a3234d690818e92988d46ca9a69b617b8.jpg)  
(3) Pr ompt: A charming coastal village scene featuring brightly painted wooden houses, a lighthouse, and winding stone paths, ..

![](images/0af3ffb62999f663b7e1233faad66e38290395c16636b794779e4ece768aa1b2.jpg)  
Frame 2

![](images/5ab70954152d17ea149113f0302c1faeabc06ab3df952c0a4ab93e6a64fd2a43.jpg)  
Frame 173

![](images/cc8f3902659b9e44a3d10896974b5155e27ca7945af16996652e53d6d7d09fa9.jpg)  
Frame 703

![](images/dae44cc08f5a5b69cd3431ccb04a4f8d6f97aaab5c28bb1662b200044b20e8bf.jpg)  
Frame 1178

![](images/faab0afc03eb3b57696d79cdbb557d7e9336ec1b65eea17b8cc1ce12bdb660ef.jpg)  
(4) Pr ompt: A traditional Japanese townscape featuring a vivid red arched bridge, torii gates, wooden boats, and classic tiled-roof buildings, ..   
Fig. 1. Teaser Demonstration. We propose Context-as-Memory, which utilizes history context as memory to guide the generation of new frames, thereby achieving scene-consistent long video generation. This figure shows key frames in generated videos under long camera trajectories. The cameras of the key frames are marked in red. It can be observed that different frames maintain good consistency when viewing the same scene from different viewpoints.

Recent advances in interactive video generation have shown promising results, yet existing approaches struggle with scene-consistent memory capabilities in long video generation due to limited use of historical context. In this work, we propose Context-as-Memory, which utilizes historical context as memory for video generation. It includes two simple yet effective designs: (1) storing context in frame format without additional post-processing; (2) conditioning by concatenating context and frames to be predicted along the frame dimension at the input, requiring no external control modules. Furthermore, considering the enormous computational overhead of incorporating all historical context, we propose the Memory Retrieval module to select truly relevant context frames by determining FOV (Field of View) overlap between camera poses, which significantly reduces the number of

candidate frames without substantial information loss. Experiments demonstrate that Context-as-Memory achieves superior memory capabilities in interactive long video generation compared to SOTAs, even generalizing effectively to open-domain scenarios not seen during training. The link of our project page is https://context-as-memory.github.io/.

# 1 INTRODUCTION

Recent breakthroughs in video generation models [Kong et al. 2024; OpenAI 2024; Runway 2024; Wang et al. 2025; Yang et al. 2024a] have shown remarkable progress. Due to their powerful generative capabilities developed through training on large-scale real-world datasets, these models are considered to have the potential to become world models capable of modeling reality [OpenAI 2024; Qin

et al. 2024; Yang et al. 2023, 2024b]. Among various research directions in this field, interactive long video generation has emerged as a crucial one since many applications, such as gaming [DeepMind 2024a; Valevski et al. 2024; Yu et al. 2025c] and simulation [Gao et al. 2024; Hu et al. 2023; Russell et al. 2025], require interactive long video generation, where the videos are generated in a streaming manner controlled by user interactions. Recent works on long video generation [Chen et al. 2024; Deng et al. 2024; Gu et al. 2025; Kondratyuk et al. 2023; Song et al. 2025; Wang et al. 2024b; Zhang and Agrawala 2025] have significantly facilitated research in this field.

Despite these advances, current approaches still face significant challenges in terms of memory capabilities [Yu et al. 2025b,a], which refers to a model’s ability to maintain content consistency during continuous video generation, such as preserving the scene when the camera returns to a previously viewed location. Take Oasis [Decart 2024] as an example: while it can generate lengthy Minecraft gameplay videos, even simple operations like turning left and then immediately right result in completely different scenes. This issue is prevalent across various state-of-the-art methods [Kanervisto et al. 2025; Song et al. 2025; Valevski et al. 2024; Yu et al. 2025c], suggesting that while current approaches can generate videos of extended duration, they struggle to maintain coherent long-term memory of scene content and spatial relationships.

However, in our view, these methods’ limitations in memory capabilities are not surprising. This is because when generating each new video frame, these methods can only predict based on a limited number of previous frames. For instance, Diffusion Forcing [Chen et al. 2024; Song et al. 2025] can only utilize context from a fixed window of several dozen frames. While this setup works for video continuation, it fails to maintain long-term consistency. In the case of video generation, if each frame to be generated could reference all previously generated frames, the generative model could actively select and replicate relevant content from historical frames into the current frame being generated, thus it would be possible to maintain scene consistency in long videos. In other words, all previously generated context frames serve as the memory.

However, the idea of "all historical context as memory" seems intuitive but is impractical for three main reasons: (1) Including all historical frames in computation would be extremely resourceintensive. (2) Processing all historical frames is computationally wasteful since only a small fraction is relevant to the current frame generation. (3) Processing irrelevant historical frames adds noise that may hinder rather than help current frame generation. There fore, a reasonable approach is to retrieve a small number of relevant frames from historical context as conditions for current generation, which we call "Memory Retrieval".

In this work, we propose Context-as-Memory as a solution for scene-consistent interactive long video generation, which includes two simple yet effective designs: (1) Storage format: directly store generated context frames as memory, requiring no post-processing such as feature embedding extraction or 3D reconstruction; (2) Conditioning method: directly incorporate as part of the input through concatenation for context learning, without requiring additional control modules like external adapters or cross attention. To effectively reduce unnecessary computational overhead and only condition on truly relevant context, we propose Memory Retrieval. Specifically,

we introduce a rule-based approach based on camera trajectories. With a camera-controlled video generation model, we can annotate all context frames with camera information based on user’s camera control. We can determine co-visibility by checking the FOV (Field of View) overlap based on camera poses at each timestamp along the trajectory, and then use this co-visibility relationship to decide which relevant frames to retrieve. To implement this solution, we collected a new scene-consistent memory learning dataset using Unreal Engine 5, featuring long videos with precise camera annotations across diverse scenes and camera trajectories. The same regions are captured across different viewpoints and times, enabling both FOV-based retrieval and long-term consistency supervision.

Our main contributions can be summarized as follows:

• We propose Context-as-Memory, highlighting the direct storage of frames as memory and conditioning via historical context learning for scene-consistent video generation.   
• To effectively utilize relevant history frames while minimizing costs, we design Memory Retrieval, a specialized rule-based approach using FOV overlap of camera trajectory.   
• We introduce a long, scene-consistent video dataset with precise camera annotations for memory training, featuring diverse scenes and captions.   
• Our experiments show superior long video generation memory, significantly outperforming SOTAs and achieving effective memory even in unseen, open-domain scenarios.

# 2 RELATED WORK

# 2.1 Interactive Long Video Generation

In following parts, we will review related work from four aspects:

Video Generation Model. Video generation models can generate video sequences $\mathbf { x } = \{ x ^ { 0 } , x ^ { 1 } , . . . , \bar { x } ^ { t } \}$ , where $x ^ { i }$ indicates the ??-th frame. The current mainstream model architecture is based on diffusion models [Ho et al. 2020; Lipman et al. 2022; Liu et al. 2022; Song and Ermon 2019; Song et al. 2021], which excel in generating high-quality content and have been widely adopted in video generation [Bao et al. 2024; DeepMind 2024b; Kling 2024; Kong et al. 2024; OpenAI 2024; Runway 2024; Wang et al. 2025; Yang et al. 2024a]. Other alternative architectures include next-token prediction [Kondratyuk et al. 2023; Wang et al. 2024b; Yan et al. 2021] and various hybrid approaches [Chen et al. 2024; Deng et al. 2024; Li et al. 2024].

Controllable Video Generation. This task can be formulated as $p ( \mathbf { x } | c )$ , where ?? represents different types of control signals. The most representative control signals include: camera motion control [Bai et al. 2025, 2024; Fu et al. 2025; He et al. 2024; Wang et al. 2024a], and agent action control in games or simulators [Decart 2024; DeepMind 2024a; Feng et al. 2024; Valevski et al. 2024; Yu et al. 2025c]. These control signals greatly enhance user interactive experience, enabling free exploration in the created virtual worlds.

Streaming Video Generation. Streaming video generation can condition on previously generated frames to continuously generate new video frames, which can be expressed as $p ( x ^ { 0 } , x ^ { 1 } , . . . , x ^ { n } ) \ =$ $\textstyle \prod _ { i = 0 } ^ { n } p ( x ^ { i } | x ^ { 0 } , x ^ { 1 } , . . . , x ^ { i - 1 } )$ , where $x ^ { i }$ indicates the ??-th frame. Representative approaches include Diffusion-based methods [Chen et al.

2024; Gu et al. 2025; Song et al. 2025; Yu et al. 2025c; Zhang and Agrawala 2025] and GPT-like next token prediction methods [Kanervisto et al. 2025; Kondratyuk et al. 2023; Wang et al. 2024b]. Diffusion-based methods generally achieve higher visual quality and faster sampling speed, thus we focus on diffusion models for long video generation in this work. Although these SOTA methods generally fail to generate long videos with scene-consistent memory, instead only producing long videos with short-term continuity.

Memory Capability for Video Generation. Many related works’ demos [Decart 2024; Kanervisto et al. 2025; Song et al. 2025; Valevski et al. 2024] have shown that current long video generation methods generally lack memory capability: while maintaining frame-toframe continuity, the scenes continuously change. One potential approach [Ma et al. 2024; Ren et al. 2025; Yu et al. 2024a,b] is to leverage 3D reconstruction to build explicit 3D representations from generated videos, then render initial frames from these 3D representations as conditions for new video generation. However, this method is limited by the accuracy and speed of 3D reconstruction, particularly in continuously expanding large scenes where accumulated 3D reconstruction errors become intolerable. Moreover, these works focus on 3D generation and merely borrow priors from video generation models, which differs from our scope. WorldMem [Xiao et al. 2025] attempts to implement memory by injecting historical frames through cross attention, and has been validated on video lengths of around 10 seconds in Minecraft scenarios.

# 2.2 Context Learning for Video Generation

Recently, some works [Gu et al. 2025; Guo et al. 2025; Zhang and Agrawala 2025] have begun to explore the role of long-context in video generation. LCT [Guo et al. 2025] performs long-context tuning on pre-trained single-shot video diffusion models to achieve consistency in multi-shot video generation. FAR [Gu et al. 2025] proposes Long-Term and Short-Term context windows to condition video generation models for long video generation. FramePack [Zhang and Agrawala 2025] introduces a hierarchical method to compress context frames into a fixed number of frames as conditioning for video generation models to achieve long video generation. However, their compression method loses too much information from temporally distant frames. In this work, we further highlight the significance of context, emphasizing that all history context serves as memory for scene-consistent long video generation.

# 3 METHOD

As discussed in Section 1, we propose that historical context frames can serve as memory for scene-consistent interactive long video generation. This section will detail how we implement this approach. Specifically: Sec. 3.1 introduces preliminaries. Sec. 3.2 describes how to inject context frames as conditions for video generation. Sec. 3.3 presents our Memory Retrieval method, which selects most relevant context frames to guide the generation of new frames. This section includes alternative approaches and our proposed search method based on camera trajectories. Sec. 3.4 introduces our long video dataset collected using Unreal Engine 5, which features precise camera pose annotations, diverse scenes, and caption annotations.

![](images/ef054afc186b77bea71338634f4f6b6d9b9092b2503f5591cc119b614c728fa5.jpg)  
Fig. 2. Model Architecture. We concatenate the context to be conditioned and the predicted frames along the frame dimension. This method of injecting context is simple and effective, requiring no additional modules.

# 3.1 Preliminaries

Full-Sequence Text-to-Video Base Model. Our work is based on a full-sequence text-to-video model, specifically, a latent video diffusion model consisting of a causal 3D VAE [Kingma et al. 2013] and a Diffusion Transformer (DiT) [Peebles and Xie 2023]. Each DiT block sequentially consists of spatial (2D) attention, spatialtemporal (3D) attention, cross-attention, and FFN modules. Let x represent a sequence of video frames, the Encoder of 3D VAE compresses it temporally and spatially to obtain the latent representation $\mathbf { z } = E n c o d e r ( \mathbf { x } )$ . With a temporal compression factor of $r$ , the original $1 + n r$ frames of $\mathbf { x } = \{ x ^ { \bar { 0 } } , x ^ { 1 } , . . . x ^ { n \bar { r } } \}$ are compressed into $1 + n$ latents of $\textbf { z } = \{ z ^ { 0 } , z ^ { 1 } , . . . , z ^ { n } \}$ . During training, random Gaussian noise $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ is added to the clean latent $\mathbf { z } _ { 0 }$ to obtain noisy latent $\mathbf { z } _ { t }$ at timestep ??. The network $\epsilon _ { \phi } ( \cdot )$ is trained to predict the added noise, with the following loss function:

$$
\mathcal {L} (\phi) = \mathbb {E} \left[ | | \epsilon_ {\phi} \left(\mathbf {z} _ {t}, \mathbf {p}, t\right) - \boldsymbol {\epsilon} | | \right], \tag {1}
$$

where $\phi$ represents the parameters and p is the given text prompt. Then we can use the predicted noise $\epsilon _ { \phi }$ to denoise the noisy latent. During inference, a clean latent z can be sampled from a randomly sampled Gaussian noise, then the Decoder of 3D VAE decodes it into video sequence $\mathbf { x } = D e c o d e r ( \mathbf { z } )$ .

Camera-Conditioned Video Generation. In our work, we incorporate camera control mechanisms [Bai et al. 2025; Wang et al. 2024a] into the video generation model to implement interactive video generation. By providing camera trajectories as conditioning for video generation, we can know the camera poses of each context frame in advance. Let cam represent the camera poses, where $f$ denotes the total number of frames. Following the mechanism proposed in ReCamMaster [Bai et al. 2025], in order to inject $\mathbf { c a m } = [ R , t ] \in \mathbb { R } ^ { f \times ( 3 \times 4 ) }$ , we first map it to the same dimension as the model’s feature channels through a camera encoder $\mathcal { E } _ { c } ( \cdot )$ , followed by adding them together:

$$
\mathbf {F} _ {i} = \mathbf {F} _ {o} + \mathcal {E} _ {c} (\mathbf {c a m}), \tag {2}
$$

![](images/9493264cf5b8d307ccd1fbd7a4b3ff230c50c51a8872b11466c57bd794ff837b.jpg)

![](images/167e6dbd3530648be9c5e882c2d967ec1cdf722d56afa3497c208841782e1bb1.jpg)  
Fig. 3. Method Demonstration. (a) We propose Context-as-Memory, where all historical context frames serve as memory conditions in the generation of predicted frames, with Memory Retrieval extracting relevant information from all context frames. (b) Our proposed Memory Retrieval method is a search algorithm based on camera trajectories. It selects relevant frames by evaluating the overlap between camera views of different frames.

where $\mathbf { F } _ { o }$ is the output of spatial attention module, $\mathbf { F } _ { i }$ is the input of 3D attention module and $\mathcal { E } _ { c } ( \cdot )$ is one layer of MLP with $\phi _ { M L P }$ as learnable parameters. During the training of camera control, we use the original diffusion loss as follows:

$$
\mathcal {L} _ {\mathbf {c a m}} \left(\phi , \phi_ {M L P}\right) = \mathbb {E} \left[ \left| \left| \epsilon_ {\boldsymbol {\phi}, \phi_ {M L P}} \left(\mathbf {z} _ {t}, \mathbf {p}, \mathbf {c a m}, t\right) - \boldsymbol {\epsilon} \right| \right|. \right. \tag {3}
$$

# 3.2 Context Frames Learning Mechanism for Memory

Suppose the latent of context that needs to be conditioned is $\mathbf { z } ^ { c }$ , and we need to learn the conditional denoiser $p ( \mathbf { z } _ { t - 1 } | \mathbf { z } _ { t } , \mathbf { z } ^ { c } )$ . Considering that the context grows continuously during the generation process (i.e., the context is variable-length), methods designed for singleframe or fixed-length frame conditions, such as Adapter [Mou et al. 2024; Zhang et al. 2023] and channel-wise concatenation [Xing et al. 2023], are not applicable. Similar to ReCamMaster [Bai et al. 2025], we propose to inject context through concatenation along the frame dimension (shown in Fig. 2), which can flexibly support variablelength context conditions. Specifically, the clean context latents $\mathbf { z } ^ { c }$ participate equally with the noisy predicted latents $\mathbf { z } _ { t }$ in the attention computation within DiT Blocks. During output, we only update the noisy latents $\mathbf { z } _ { t }$ using the predicted noise $\epsilon _ { \phi } ( \{ \mathbf { z } _ { t } , \mathbf { z } ^ { c } \} , \mathbf { p } , t )$ while keeping the clean context latents $\mathbf { z } ^ { c }$ unchanged.

Another challenge is how to handle positional encoding along the frame dimension in video diffusion models after context frame expansion. Since our method is based on a pre-trained full-sequence text-to-video model, to preserve the original model’s generation capability and facilitate easier adaptation to the context-conditioned generation setting, we maintain the same positional encoding for predicted latents $\mathbf { z } _ { t }$ as in the pre-training phase, while assigning new positional encodings to the newly conditioned context latents $\mathbf { z } ^ { c }$ . Our base model employs RoPE [Su et al. 2024], which can conveniently adapt to variable-length position encodings.

# 3.3 Memory Retrieval

As analyzed in Sec. 1, including all context frames in computation is impractical due to computational overhead and may introduce irrelevant information that causes interference. A reasonable approach is to filter out valuable frames from the context, specifically frames that share overlapping visible regions with the frames to be generated. To this end, we propose Memory Retrieval to accomplish

this task as shown in Fig. 3 (a). Below, we first introduce several alternative implementation methods, followed by our solution.

Alternative method #1: random selection. A baseline randomly selects frames from context. This works well in early generation when context size is small, as adjacent frames’ natural redundancy reduces the risk of missing important information. However, with hundreds of context frames, random selection fails to identify valuable frames.

Alternative method #2: neighbor frames within a window. Another approach selects consecutive recent frames within a window near the current predicted frames. While common in existing methods [Decart 2024; Song et al. 2025; Yu et al. 2025c], this has key limitations. First, adjacent frames’ redundancy means multiple consecutive frames add little new information beyond the most recent frame. Second, ignoring temporally distant frames prevents awareness of previously seen scenes, leading to continuous generation of new scenes and ultimately breaking scene consistency.

Alternative method #3: hierarchical compression. FramePack [Zhang and Agrawala 2025] proposes a hierarchical compression method for context frames into a minimal set (e.g., 2-3 frames). For twoframe compression, it allocates space proportionally: the most recent frame gets one full frame, the second most recent gets half, the third gets a quarter, and so on, totaling two frames. While achieving high compression, this exponential decay significantly loses historical information. Though the authors suggest manually preserving certain key frames uncompressed, they don’t specify the selection criteria.

Our method: camera-trajectory-based search. The fundamental limitation of these methods lies in their inability to identify truly valuable frames from the large number of context frames. They either introduce many redundant frames or lose too much useful information, especially from the old frames that are temporally distant. We leverage the known camera trajectory of the context to search for valuable frames, specifically those that share high-overlap visible regions with predicted frame as shown in Fig. 3 (b).

The first question is how to obtain the camera trajectory of the context video. Since we have introduced camera control into our video generation model in Sec. 3.1, these context frames are generated with user-provided camera poses. These conditioning camera

![](images/8ac5bc1598e839202353286ba656702395c684813de76724e594180e39ebc183.jpg)  
Fig. 4. Examples of FOV Overlap. We simplify FOV overlap detection to checking intersections between four rays from two camera origins. A practical rule that works for most cases requires: both left and right ray pairs intersect (a, b). However, we must filter out cases where intersection points are either too near (d) or too distant (c) from cameras. While this rule may not cover all scenarios and some corner cases exist (e, f ), occasional missed or incorrect candidates don’t substantially affect overall performance.

poses can serve as camera annotations for the generated context, eliminating the need for an additional camera pose estimator.

The second question is how to determine co-visibility between two frames given their camera poses. We attempt to determine this by checking if there is an overlapping region between the fan-shaped areas corresponding to the Fields of View (FOV) of the two cameras. Specifically, since we restrict camera movement to the XY plane, we only need to consider the left and right rays shooting from each camera’s origin. By checking the intersection of these four rays from two cameras, we can quickly determine the FOV overlap as shown in Fig. 4. Additionally, we calculate the distance between the predicted frame’s camera and the calculated intersection points to eliminate cases where the cameras are too far apart (which typically indicates no actual overlap or very small overlap). This FOV overlap detection is not perfect, as it may fail in cases with occlusions. However, this method effectively reduces the number of candidate context frames.

The final question is: after FOV co-visibility filtering, if the number of filtered frames still exceeds the context condition limit, how should we further filter them? A baseline approach would be random selection, but we also provide some more insightful strategies: (1) Considering the redundancy between adjacent frames, we randomly select only one frame from each group of consecutive frames in the filtered context. This design is highly effective, significantly reducing the number of candidate frames while preserving most of the valuable information. (2) Building upon the first strategy, we can additionally select a few context frames that are furthest apart either spatially or temporally. This helps to supplement potentially missing long-term information (both spatial and temporal). However, in most cases, this additional selection may not be necessary.

Implementation details in training and inference. Assume the maximum number of retrieved context frames is $k$ . During training, we read a long ground truth video (containing thousands of frames) and randomly select a segment as the sequence to be predicted. We then apply our Memory Retrieval method to select $k - 1$ context frames from the remaining frames. The overlapping relationships between frames have been pre-computed, eliminating the need for repeated calculations. The first frame of the prediction sequence is also included as an additional context frame to ensure video continuity. Additionally, there is a $1 0 \%$ probability during training that only the recent context frame is used, simulating the beginning of long video generation where no context frames are available. During inference, for each video segment to be predicted, we search

ALGORITHM 1: Training Process of Context-as-Memory   
Input: Video sequence $\mathcal{X}$ and camera annotations $C$ in training dataset, context size $k$ 1 while not converged do  
2 Randomly select predicted video sequence $\mathbf{x}_0$ from $\mathcal{X}$ ;  
3 Retrieve $k$ frames as context $\mathbf{x}^c$ ;  
4 Obtain camera poses $\{\mathbf{cam}_0,\mathbf{cam}^c\}$ for $\{\mathbf{x}_0,\mathbf{x}^c\}$ from $C$ ;  
5 Obtain latent embeddings $\{\mathbf{z}_0,\mathbf{z}^c\} \gets \mathrm{Encoder}(\{\mathbf{x}_0,\mathbf{x}^c\})$ ;  
6 Sample $t \sim U(1,T)$ and $\epsilon \sim \mathcal{N}(0,\mathbf{I})$ , then corrupt $\mathbf{z}_0$ to $\mathbf{z}_t$ ;  
7 Train $p(\mathbf{z}_{t-1} \mid \mathbf{z}_t,\mathbf{z}^c,\mathbf{cam}_0,\mathbf{cam}^c,t)$ using diffusion loss;

ALGORITHM 2: Inference Process of Context-as-Memory   
Input: Initial frame set $\mathcal{X} = \{\mathbf{x}_{\mathrm{init}}\}$ and camera poses $C = \{\mathbf{cam}_{\mathrm{init}}\}$ Output: Generated video sequence X   
1 while generation not finished do   
2 User provides next target camera pose $\mathbf{cam}^t$ .   
3 Retrieve context frames $\mathbf{x}^c\subset \mathcal{X}$ and $\mathbf{cam}^c\subset C$ by checking FOV overlap with $\mathbf{cam}^t$ .   
4 Compute context latent $\mathbf{z}^c\gets$ Encoder $(\mathbf{x}^{c})$ .   
5 Sample noise $\epsilon \sim \mathcal{N}(0,\mathbf{I})$ and infer latent $\mathbf{z}^t\sim p(\mathbf{z}^t\mid \epsilon ,\mathbf{z}^c,\mathbf{cam}^t,\mathbf{cam}^c)$ .   
6 Decode generated frames $\mathbf{x}^t\gets$ Decoder $(\mathbf{z}^t)$ .   
7 Append $\mathbf{x}^t$ to $\mathcal{X}$ and $\mathbf{cam}^t$ to $C$

$k - 1$ context frames from the previously generated frames using FOV-based Memory Retrieval and add the most recently generated frame to the context. The training and inference procedures are outlined in Algorithm 1 and 2, respectively.

# 3.4 Data Collection

To validate our method, we require long video datasets with camera pose annotations. However, currently available datasets with camera pose information typically consist of short video clips [Bai et al. 2025; Zhou et al. 2018]. To obtain long-duration data with precise camera annotations, we utilized a simulation environment, specifically Unreal Engine 5. We generated randomized camera trajectories navigating through different scenes and rendered corresponding long videos. Our dataset comprises 100 videos of 7,601 frames each, featuring 12 distinct scene styles, with captions annotated by a multimodal LLM [Yao et al. 2024] every 77 frames. To simplify the problem while still effectively validating our method, we constrained the camera trajectory’s position changes to a 2D plane and limited rotation to only around the z-axis, which still presents sufficient complexity for camera trajectory control. Additional details about the dataset are provided in the supplementary materials.

# 4 EXPERIMENTS

# 4.1 Experiment Settings

Implementation Details. Our method is implemented on an internal 1B-parameter pre-trained text-to-video Diffusion Transformer, developed for research purposes. The resolution of generated videos is $6 4 0 \times 3 5 2$ . The model supports generation of 77-frame videos, with a temporal compression ratio of 4 in the causal 3D VAE, resulting

![](images/f40498a3a50d4d552f1a412f1ae118c1691a8363a958681955a272119659a8cc.jpg)

![](images/1060548590e78584a05b1cac8cde8273690d07066a604fe2da0a9c70e77589bd.jpg)  
Fig. 5. Qualitative Comparison Results. Among them, Context-as-Memory demonstrated the best memory capabilities and the highest visual quality, indicating the effectiveness of sufficient context information conditioning. Other methods exhibit scene inconsistency issues due to limited context utilization.

Table 1. Quantitative Comparison results. Due to learning abundant context, Context-as-Memory demonstrates the best memory capabilities and highest quality of generated videos. In contrast, DFoT [Song et al. 2025] and FramePack [Zhang and Agrawala 2025], which can only utilize the most recent contexts, show relatively inferior performance, even worse than random context selection. This is because although random selection cannot guarantee the selection of useful information, on average it tends to obtain more information compared to methods that only learn from the most recent context.   

<table><tr><td rowspan="2">Methods</td><td colspan="4">Ground Truth Comparison</td><td colspan="4">History Context Comparison</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>FID↓</td><td>FVD↓</td><td>PSNR↑</td><td>LPIPS↓</td><td>FID↓</td><td>FVD↓</td></tr><tr><td>1st Frame as Context</td><td>15.72</td><td>0.5282</td><td>127.55</td><td>937.51</td><td>14.53</td><td>0.5456</td><td>157.44</td><td>1029.71</td></tr><tr><td>1st Frame + Random Context</td><td>17.70</td><td>0.4847</td><td>115.94</td><td>853.13</td><td>17.07</td><td>0.3985</td><td>119.31</td><td>882.36</td></tr><tr><td>DFoT [Song et al. 2025]</td><td>17.63</td><td>0.4528</td><td>112.96</td><td>897.87</td><td>15.70</td><td>0.5102</td><td>121.18</td><td>919.75</td></tr><tr><td>FramePack [Zhang and Agrawala 2025]</td><td>17.20</td><td>0.4757</td><td>121.87</td><td>901.58</td><td>15.65</td><td>0.4947</td><td>131.59</td><td>974.52</td></tr><tr><td>Context-as-Memory (Ours)</td><td>20.22</td><td>0.3003</td><td>107.18</td><td>821.37</td><td>18.11</td><td>0.3414</td><td>113.22</td><td>859.42</td></tr></table>

in 20-frame video latents generation. We set the context size to 20, meaning 20 RGB frames are selected as context. Since these frames lack temporal continuity, they are individually compressed using the causal 3D VAE, also resulting in 20 frames of video latents. The model was trained on our collected dataset for over 10, 000 iterations with a batch size of 64 on 8 NVIDIA A100 GPUs. During sampling, we employ Classifier-Free Guidance [Ho and Salimans 2022] for text prompts, with 50 sampling steps.

Evaluation Methods. To evaluate our method, we held out $5 \%$ of the dataset containing diverse scenes for testing. Our evaluation metrics include: (1) FID and FVD for video quality assessment; (2) PSNR and LPIPS for quantifying memory capability through pixel-wise differences between frames. Given the lack of memory evaluation methods, we propose two approaches: (1) Ground truth comparison: evaluating whether predicted frames match ground truth based on context selected from ground truth frames; (2) History context comparison: comparing newly generated frames with previously generated ones in long video sequences. This second approach provides stronger evidence of memory capability as it evaluates consistency in newly generated content. In our implementation, we test on simple trajectories where the camera rotates n degrees and returns, allowing easy identification of corresponding frames for PSNR/LPIPS calculation.

# 4.2 Comparison Results

In this section, we evaluate video generation memory capabilities across baseline methods, SOTA approaches, and our Context-as-Memory. The compared methods include: (1) Single-frame context using the first frame; (2) Multi-frame context using the first frame plus random historical frames; (3) Diffusion Forcing Transformer (DFoT) [Song et al. 2025], using a fixed-size window of most recent frames; (4) FramePack [Zhang and Agrawala 2025], which hierarchically compresses previous context into two frames, with each frame’s height or width halved compared to its predecessor. While theoretically supporting all historical frames, compression becomes impractical after several frames as latent size reduces to $1 \times 1 .$ . For fair comparison, all methods were implemented on our base model and dataset with identical training configurations and iterations. Results are presented in Table 1 and Figure 5.

PSNR and LPIPS metrics demonstrate our Memory-as-Context’s advantages over other approaches. It effectively retrieves and utilizes useful context information, while other methods have limited context access. Random context selection outperforms DFoT and FramePack, possibly because although it cannot guarantee selecting useful context, it still performs better on average than methods limited to only recent frames. DFoT and FramePack’s performance limitations stem from adjacent frame redundancy. Despite accessing

Table 2. Ablation of Context Size. Larger context sizes contain more useful information and lead to better memory capability, but also incur higher computational overhead, necessitating an optimal trade-off choice.   

<table><tr><td rowspan="2">Context Size</td><td colspan="2">GT Comp.</td><td colspan="2">HC Comp.</td><td rowspan="2">Speed (fps)↑</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>1</td><td>15.72</td><td>0.5282</td><td>14.53</td><td>0.5456</td><td>1.60</td></tr><tr><td>5</td><td>17.37</td><td>0.4825</td><td>15.97</td><td>0.5063</td><td>1.40</td></tr><tr><td>10</td><td>19.14</td><td>0.3554</td><td>17.75</td><td>0.3985</td><td>1.20</td></tr><tr><td>20</td><td>20.22</td><td>0.3003</td><td>18.11</td><td>0.3414</td><td>0.97</td></tr><tr><td>30</td><td>20.31</td><td>0.3137</td><td>18.19</td><td>0.3319</td><td>0.79</td></tr></table>

Table 3. Ablation of Memory Retrieval Strategy. The filtering methods of "FOV" and "Non-adj" (where only one frame from continuous frame sequences is selected as a candidate) effectively filter out useless and redundant information, leading to significant improvements in memory capability.   

<table><tr><td rowspan="2">Strategy</td><td colspan="2">GT Comp.</td><td colspan="2">HC Comp.</td></tr><tr><td>PSNR↑</td><td>LPIPS↓</td><td>PSNR↑</td><td>LPIPS↓</td></tr><tr><td>Random</td><td>17.70</td><td>0.4847</td><td>17.07</td><td>0.3985</td></tr><tr><td>FOV+Random</td><td>19.17</td><td>0.3825</td><td>17.47</td><td>0.3896</td></tr><tr><td>FOV+Non-adj</td><td>20.11</td><td>0.3075</td><td>18.19</td><td>0.3571</td></tr><tr><td>FOV+Non-adj+Far-space-time</td><td>20.22</td><td>0.3003</td><td>18.11</td><td>0.3414</td></tr></table>

dozens of recent frames, the inherent redundancy limits effective information utilization. FramePack’s exponential information decay further weakens its memory capabilities compared to DFoT.

Moreover, FID and FVD show our Context-as-Memory achieves the best generation quality among all methods. Sufficient context conditioning not only enhances memory but also improves generation quality by reducing error accumulation in long videos. This improvement stems from two factors: (1) context provides stronger conditional guidance by reducing generation uncertainty, and (2) earlier generated frames used as context contain fewer accumulated errors, helping minimize error propagation in new frames.

Additionally, History Context Comparison proves more challenging than Ground Truth Comparison. Even with simple "rotate forward and rotate backward" trajectories, the performance gaps between methods are significant. DFoT and FramePack can only utilize the most recent context, causing them to continuously generate new content. Only by having access to global context and extracting useful relevant information from it can memory-aware new video generation be achieved.

# 4.3 Ablation Study

Ablation of Context Size. We studied how context size affects memory capability. Larger contexts theoretically provide more useful information, improving memory performance as shown in Tab. 2. However, this comes with increased computational cost and slower generation speed. When context size reaches 30, there’s a notable speed drop compared to size 1. Balancing performance and speed, a context size of 20 offers a good trade-off. Future improvements in context compression techniques may help reduce the optimal context size further.

Ablation of Memory Retrieval Strategy. We ablated different memory retrieval strategies to analyze their effects. "Random" refers

![](images/58a91be5f8b47e7a58910b23566c3ab1c71d05c2bf6b8e3946117cb1ffd5bbbf.jpg)  
(1) Prompt:

![](images/04210dbba6fff988b40ddb2b15455b9c29758c20a1a016e58c97aa8715272dad.jpg)  
(2) Prompt:

![](images/c6ffa244ca44af8fb2c9f83d91cf9b3c185f0f40079c9480bcaac3006a1958bf.jpg)  
(3) Prompt:   
Fig. 6. Open-Domain Results. We collected open-domain images from the internet and used them as the first frame to generate subsequent long videos. Under the trajectory of "rotate away and rotate back," even when generating new content, it still demonstrates good memory capability.

to randomly selecting context; "FOV+Random" means first filtering using the FOV-based method, then randomly selecting from the remaining candidates; "Non-adj" means only one frame from continuous frame sequences will be selected as a candidate; "farspace-time" means frames that are more distant in time or space are more likely to be selected. The results in Tab. 3 demonstrate the effectiveness of "FOV" and "Non-adj" methods in removing useless and redundant information, which significantly increases the probability of selecting useful context and thereby enhances memory capability. The impact of "Far-space-time" is relatively minor.

# 4.4 Open-Domain Results

Due to our diverse training dataset and the various visual priors learned by our base model during pre-training, our method has the potential to generalize to open-domain scenarios not present in the training set. We selected images of different styles from the internet and used them as the first frame to generate long videos. We validated using the trajectory of "rotate away and rotate back," which is suitable for verifying memory consistency in generated content. Results in Fig. 6 demonstrate that our method indeed possesses good memory capability in open-domain scenarios.

# 5 CONCLUSION

In this work, we propose Context-as-Memory, highlighting that using historical generated frames as memory is key to achieving scene-consistent long video generation. Our method design is simple yet effective, directly saving context frames as memory and inputting the context together with the predicted frame as conditions. Furthermore, to avoid high computational overhead caused by lengthy context, we propose Memory Retrieval to dynamically select truly valuable context based on the predicted video frames.

Limitations and Future Work. Although our method has made significant progress in achieving memory capability for long video generation, several limitations remain: (1) Our method is limited to static scenes, while memory retrieval for dynamic scenes poses greater challenges; (2) In complex scenarios, particularly those with multiple occlusions (e.g., interconnected indoor rooms), FOV overlap may struggle to effectively identify truly relevant context frames;

(3) The inherent error accumulation problem in long video generation persists, which currently can only be addressed through larger datasets, more extensive training, and more powerful base models. In the future, we will continue to develop memory capabilities for open-domain long video generation on larger-scale base models, supporting more complex trajectories, broader scene ranges, and longer generation sequences.

# REFERENCES

Jianhong Bai, Menghan Xia, Xiao Fu, Xintao Wang, Lianrui Mu, Jinwen Cao, Zuozhu Liu, Haoji Hu, Xiang Bai, Pengfei Wan, et al. 2025. ReCamMaster: Camera-Controlled Generative Rendering from A Single Video. arXiv preprint arXiv:2503.11647 (2025).   
Jianhong Bai, Menghan Xia, Xintao Wang, Ziyang Yuan, Xiao Fu, Zuozhu Liu, Haoji Hu, Pengfei Wan, and Di Zhang. 2024. SynCamMaster: Synchronizing Multi-Camera Video Generation from Diverse Viewpoints. arXiv:2412.07760 [cs.CV] https: //arxiv.org/abs/2412.07760   
Fan Bao, Chendong Xiang, Gang Yue, Guande He, Hongzhou Zhu, Kaiwen Zheng, Min Zhao, Shilong Liu, Yaole Wang, and Jun Zhu. 2024. Vidu: a highly consistent, dynamic and skilled text-to-video generator with diffusion models. arXiv preprint arXiv:2405.04233 (2024).   
Boyuan Chen, Diego Marti Monso, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. 2024. Diffusion forcing: Next-token prediction meets full-sequence diffusion. arXiv preprint arXiv:2407.01392 (2024).   
Etched Decart. 2024. Oasis: A Universe in a Transformer. https://oasis-model.github.io/.   
Google DeepMind. 2024a. Genie 2: A large-scale foundation world model. https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-worldmodel/.   
Google DeepMind. 2024b. Veo 2: Our state-of-the-art video generation model. https: //deepmind.google/technologies/veo/veo-2/.   
Haoge Deng, Ting Pan, Haiwen Diao, Zhengxiong Luo, Yufeng Cui, Huchuan Lu, Shiguang Shan, Yonggang Qi, and Xinlong Wang. 2024. Autoregressive Video Generation without Vector Quantization. arXiv preprint arXiv:2412.14169 (2024).   
Ruili Feng, Han Zhang, Zhantao Yang, Jie Xiao, Zhilei Shu, Zhiheng Liu, Andy Zheng, Yukun Huang, Yu Liu, and Hongyang Zhang. 2024. The Matrix: Infinite-Horizon World Generation with Real-Time Moving Control. arXiv preprint arXiv:2412.03568 (2024).   
Xiao Fu, Xian Liu, Xintao Wang, Sida Peng, Menghan Xia, Xiaoyu Shi, Ziyang Yuan, Pengfei Wan, Di Zhang, and Dahua Lin. 2025. 3DTrajMaster: Mastering 3D Trajectory for Multi-Entity Motion in Video Generation. In ICLR.   
Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. 2024. Vista: A generalizable driving world model with high fidelity and versatile controllability. arXiv preprint arXiv:2405.17398 (2024).   
Yuchao Gu, Weijia Mao, and Mike Zheng Shou. 2025. Long-Context Autoregressive Video Modeling with Next-Frame Prediction. arXiv preprint arXiv:2503.19325 (2025).   
Yuwei Guo, Ceyuan Yang, Ziyan Yang, Zhibei Ma, Zhijie Lin, Zhenheng Yang, Dahua Lin, and Lu Jiang. 2025. Long context tuning for video generation. arXiv preprint arXiv:2503.10589 (2025).   
Hao He, Yinghao Xu, Yuwei Guo, Gordon Wetzstein, Bo Dai, Hongsheng Li, and Ceyuan Yang. 2024. Cameractrl: Enabling camera control for text-to-video generation. arXiv preprint arXiv:2404.02101 (2024).   
Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising diffusion probabilistic models. Advances in neural information processing systems (2020).   
Jonathan Ho and Tim Salimans. 2022. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598 (2022).   
Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. 2023. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080 (2023).   
Anssi Kanervisto, Dave Bignell, Linda Yilin Wen, Martin Grayson, Raluca Georgescu, Sergio Valcarcel Macua, Shan Zheng Tan, Tabish Rashid, Tim Pearce, Yuhan Cao, et al. 2025. World and human action models towards gameplay ideation. Nature 638, 8051 (2025), 656–663.   
Diederik P Kingma, Max Welling, et al. 2013. Auto-encoding variational bayes. Kling. 2024. Kling AI: Next-Generation AI Creative Studio. https://app.klingai.com/.   
Dan Kondratyuk, Lijun Yu, Xiuye Gu, José Lezama, Jonathan Huang, Grant Schindler, Rachel Hornung, Vighnesh Birodkar, Jimmy Yan, Ming-Chang Chiu, et al. 2023. Videopoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125 (2023).   
Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. 2024. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024).   
Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. 2024. Autoregressive Image Generation without Vector Quantization. arXiv preprint arXiv:2406.11838 (2024).

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. 2022. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022).   
Xingchao Liu, Chengyue Gong, and Qiang Liu. 2022. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022).   
Baorui Ma, Huachen Gao, Haoge Deng, Zhengxiong Luo, Tiejun Huang, Lulu Tang, and Xinlong Wang. 2024. You See it, You Got it: Learning 3D Creation on Pose-Free Videos at Scale. arXiv preprint arXiv:2412.06699 (2024).   
Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. 2024. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings of the AAAI conference on artificial intelligence.   
OpenAI. 2024. Creating video from text. https://openai.com/index/sora/. William Peebles and Saining Xie. 2023. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision.   
Yiran Qin, Zhelun Shi, Jiwen Yu, Xijun Wang, Enshen Zhou, Lijun Li, Zhenfei Yin, Xihui Liu, Lu Sheng, Jing Shao, et al. 2024. Worldsimbench: Towards video generation models as world simulators. arXiv preprint arXiv:2410.18072 (2024).   
Xuanchi Ren, Tianchang Shen, Jiahui Huang, Huan Ling, Yifan Lu, Merlin Nimier-David, Thomas Müller, Alexander Keller, Sanja Fidler, and Jun Gao. 2025. Gen3c: 3d-informed world-consistent video generation with precise camera control. arXiv preprint arXiv:2503.03751 (2025).   
Runway. 2024. Runway : Tools for human imagination. https://runwayml.com/. Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. 2025. GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving. arXiv preprint arXiv:2503.20523 (2025).   
Kiwhan Song, Boyuan Chen, Max Simchowitz, Yilun Du, Russ Tedrake, and Vincent Sitzmann. 2025. History-Guided Video Diffusion. arXiv preprint arXiv:2502.06764 (2025).   
Yang Song and Stefano Ermon. 2019. Generative modeling by estimating gradients of the data distribution. Advances in neural information processing systems (2019).   
Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2021. Score-based generative modeling through stochastic differential equations. International Conference on Learning Representations (2021).   
Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing 568 (2024), 127063.   
Dani Valevski, Yaniv Leviathan, Moab Arar, and Shlomi Fruchter. 2024. Diffusion models are real-time game engines. arXiv preprint arXiv:2408.14837 (2024).   
Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, et al. 2025. Wan: Open and Advanced Large-Scale Video Generative Models. arXiv preprint arXiv:2503.20314 (2025).   
Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. 2024b. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869 (2024).   
Zhouxia Wang, Ziyang Yuan, Xintao Wang, Yaowei Li, Tianshui Chen, Menghan Xia, Ping Luo, and Ying Shan. 2024a. Motionctrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH 2024 Conference Papers.   
Zeqi Xiao, Yushi Lan, Yifan Zhou, Wenqi Ouyang, Shuai Yang, Yanhong Zeng, and Xingang Pan. 2025. WORLDMEM: Long-term Consistent World Simulation with Memory. arXiv preprint arXiv:2504.12369 (2025).   
Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Xintao Wang, Tien-Tsin Wong, and Ying Shan. 2023. DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors. arXiv:2310.12190   
Wilson Yan, Yunzhi Zhang, Pieter Abbeel, and Aravind Srinivas. 2021. Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157 (2021).   
Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Jonathan Tompson, Dale Schuurmans, and Pieter Abbeel. 2023. Learning Interactive Real-World Simulators. arXiv preprint arXiv:2310.06114 (2023).   
Sherry Yang, Jacob C Walker, Jack Parker-Holder, Yilun Du, Jake Bruce, Andre Barreto, Pieter Abbeel, and Dale Schuurmans. 2024b. Position: Video as the New Language for Real-World Decision Making. In Proceedings of the 41st International Conference on Machine Learning.   
Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. 2024a. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arXiv preprint arXiv:2408.06072 (2024).   
Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. 2024. MiniCPM-V: A GPT-4V Level MLLM on Your Phone. arXiv preprint arXiv:2408.01800 (2024).   
Hong-Xing Yu, Haoyi Duan, Junhwa Hur, Kyle Sargent, Michael Rubinstein, William T Freeman, Forrester Cole, Deqing Sun, Noah Snavely, Jiajun Wu, et al. 2024a. Wonderjourney: Going from anywhere to everywhere. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6658–6667.   
Jiwen Yu, Yiran Qin, Haoxuan Che, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, Kun Gai, Hao Chen, and Xihui Liu. 2025b. A Survey of Interactive Generative Video.

arXiv preprint arXiv:2504.21853 (2025).   
Jiwen Yu, Yiran Qin, Haoxuan Che, Quande Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. 2025a. Position: Interactive generative video as next-generation game engine. arXiv preprint arXiv:2503.17359 (2025).   
Jiwen Yu, Yiran Qin, Xintao Wang, Pengfei Wan, Di Zhang, and Xihui Liu. 2025c. Game-Factory: Creating New Games with Generative Interactive Videos. arXiv:2501.08325   
Wangbo Yu, Jinbo Xing, Li Yuan, Wenbo Hu, Xiaoyu Li, Zhipeng Huang, Xiangjun Gao, Tien-Tsin Wong, Ying Shan, and Yonghong Tian. 2024b. Viewcrafter: Taming video diffusion models for high-fidelity novel view synthesis. arXiv preprint arXiv:2409.02048 (2024).   
Lvmin Zhang and Maneesh Agrawala. 2025. Packing Input Frame Context in Next-Frame Prediction Models for Video Generation. arXiv preprint arXiv:2504.12626 (2025).   
Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. 2023. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision. 3836–3847.   
Tinghui Zhou, Richard Tucker, John Flynn, Graham Fyffe, and Noah Snavely. 2018. Stereo magnification: Learning view synthesis using multiplane images. arXiv preprint arXiv:1805.09817 (2018).

# A INTRODUCTION OF THE BASE TEXT-TO-VIDEO GENERATION MODEL

We use a transformer-based latent diffusion model as the base T2V generation model, as illustrated in Fig. 7. We employ a 3D-VAE to transform videos from the pixel space to a latent space, upon which we construct a transformer-based video diffusion model. Unlike previous models that rely on UNets or transformers, which typically incorporate an additional 1D temporal attention module for video generation, such spatially-temporally separated designs do not yield optimal results. We replace the 1D temporal attention with 3D self-attention, enabling the model to effectively perceive and process spatiotemporal tokens, thereby achieving a high-quality and coherent video generation model. Specifically, before each attention or feed-forward network (FFN) module, we map the timestep to a scale, thereby applying RMSNorm to the spatiotemporal tokens.

# B DETAILS OF COLLECTED DATASET

In this section, we provide a detailed description of the rendered dataset used to train our model.

3D Environments. We collect 12 different 3D environments assets from https://www.fab.com/. To minimize the domain gap between rendered data and real-world videos, we primarily select visually realistic 3D scenes, while choosing a few stylized or surreal 3D scenes as a supplement. To ensure data diversity, the selected scenes cover a variety of indoor and outdoor settings, such as city streets, shopping malls, and the countryside.

Camera Trajectories. To create data that roam within a scene, we employ smoothed polylines as camera trajectories. Specifically, we begin by randomly sampling coordinate points in the 3D scene to serve as the endpoints of the polyline, and then generate B-spline curves from these points. To ensure smooth camera movement without abrupt speed changes or rotations, we limit the camera’s movement distance to the range of $[ 3 \mathrm { m } , 6 \mathrm { m } ]$ for each 77-frame video segment and restrict the rotation angle within the xy-plane to less than 60 degrees.

Upon completing the 3D scene collection and trajectory design, we utilized Unreal Engine 5 to batch-render 100 long videos for training. Each video features 7,601 frames (30 fps) of continuous camera movement. Additionally, we record the camera’s extrinsic

and intrinsic parameters for each frame. The camera is configured with a focal length of $2 4 \mathrm { m m }$ , an aperture of 10, and a field of view (FOV) of 52.67 degrees.

# C ADDITIONAL OPEN-DOMAIN RESULTS

In Fig. 8 and Fig. 9, we present additional open-domain results. Using diverse images collected from the internet as initial frames, we demonstrate long video generation with "rotate away and rotate back" trajectories. These source images, representing various styles and scenes, can be found in the provided Data.

Our method achieves generalization capability in open-domain scenarios due to two main factors: (1) Training on diverse scenes enables the model to develop generalizable context utilization skills; (2) The pre-trained base model possesses strong generative priors from exposure to various data types during pre-training.

However, our method still faces significant limitations in opendomain generalization that require future research: (1) The 1Bparameter base model’s capabilities are insufficient, only showing good results on simple trajectories. For complex trajectories, the base model struggles to generate high-quality content from the initial frame, leading to unacceptable error accumulation in long video generation. Validating our approach with larger-scale base models remains a future research direction. (2) The method cannot yet support more complex, diverse, and dynamic long-term scene exploration in open-domain settings. Our ideal goal is to enable free, extended navigation from any given image while maintaining memory consistency. This is a challenging objective, though the "context as memory" concept shows promise.

![](images/f362be8fd25dd45b409fdb79a4d1826fc060cbd4610866c9887a7c977f60a66b.jpg)

Fig. 7. Overview of the base text-to-video generation model.   
![](images/0eee2a3b94b4910b581b0ba04e7d69dafd15f883b7669bc49732049e24ae7d22.jpg)  
(1) Prompt: Japanese landscape, Mount Fuji, blue sky, red and white pagoda, cherry blossom trees, cultural harmony, natural beauty.

![](images/28f9a68303637c1d5fa6fb4c8995f93602690571cec91f0069ed0ec96e11845f.jpg)  
(2) Prompt: Black Myth Wukong, snow-covered stone pathway, ancient temple, wintry forest, towering trees, solemn statues, sacred ambiance, icicles, ...

![](images/756e9029cd381b2cc6a1c67b3ec3f8229e55d77dd2e1117d766795f3e45a00ef.jpg)  
(3) Prompt: Black Myth Wukong, mist-laden forest, weathered wooden structure, overgrown vegetation, moss-covered stone steps, ornate tiled roof, ...

![](images/0bf42abeaf599ab0fb1b20d3001c595954baeefce10c267a18cc021aca0c382e.jpg)  
(4) Prompt: Black Myth Wukong, a barren landscape of red and grey features a desolate shrine with weathered East Asian architecture, a lone stone statue, ...

![](images/7c5e2198e18ab8a753731a42d3191036668b3587646500e3a70701516bb8ce39.jpg)  
(5) Prompt: A sunlit alpine valley shows logs by a rustic cabin, surrounded by meadows, forests, and towering mountains. The breeze gently stirs the grass, ...

![](images/039466101fd30c3c4575a5ec4f97bda539a7f20d2a9573f657fad665bb945a97.jpg)  
(6) Prompt: The Legend of Zelda, vibrant green fields, wildflowers, sprawling landscape, winding rivers, distant ruins, towering mountains, clear blue sky, ...

![](images/7f0ddab84dc4ac4a709981dde0e3ac4e0a47d8ecb4cf926c2e1cc9657c7716da.jpg)  
(7) Prompt: Genshin Impact, a festival pier glows with lanterns, featuring a poised girl in traditional attire amid dragon-shaped lanterns and swaying pavilions, ...

Fig. 8. Open-Domain Results.   
![](images/48cd89e6a2384f0a97c2dc647197c4eb79ef9e77cbb122fa87322bdb626f4ec6.jpg)  
(8) Prompt: Genshin Impact, a gentle pan reveals a vibrant countryside village in golden morning light, with a windmill turning above timber-framed houses and ..

![](images/8e775581b9b5386edbbaf829dd7736457afd5fd73b7eb4188807a8af7449a081.jpg)  
(1) Prompt: A smooth tracking shot sweeps through a sunlit high-rise office, showcasing ergonomic workstations and a stunning cityscape view, ...

![](images/705f3be1cbbf63bec6cf4c526ceaadcf5055b8d739c481af044dad6306eb5be4.jpg)  
(2) Prompt: A bright, open-plan office features minimalist desks, ergonomic chairs, and calming greenery, all under natural and LED lighting,

![](images/0c09cea31ab9a179fad1716bd7f3216c9f29fefd76a6e80ea4beb86dc9ffc8e4.jpg)  
(3) Prompt: A serene bedroom features a neatly made bed with cream bedding, wooden nightstands, and minimalist decor. Soft natural light and framed artwork ...

![](images/cdc2b3755c918f32c4988bd43b96ac9778012144d2941244cd9c7ead37ba49ef.jpg)  
(4) Prompt: A slow glide through a formal garden reveals a circular stone pedestal, symmetrical hedges, and blooming rose bushes, ..

![](images/27fd4c3e957cecb6f9b2cacc6b32d0fb19edfd9a8abd4dff7f73d95e8df82079.jpg)  
(5) Prompt: An aerial view glides over a winding path in an alpine meadow, with golden sunlight and mist creating a serene morning scene, ...

![](images/612fce298b10c0d8e1cf5346b1452e46025790c30d0453257977614c3f2abe98.jpg)  
(6) Prompt: A slow zoom captures a tranquil alpine forest and a mountain range glowing in the sunset. A still body of water reflects snow-capped peaks ...

![](images/616a647cbe83c37a1e1141c91a56f9361f7f2d9e25314b2913400daf2b647f40.jpg)  
(7) Prompt: A sunlit view of a frozen alpine basin surrounded by snow-covered peaks. The camera pans, highlighting the contrast between sunlit ridges ...

![](images/bd9d69cc157fc1d72b77bd6ffd16ebfe9e3900ebe4c185f10b7006dd0006aa2d.jpg)  
(8) Prompt: A tranquil lakeside scene reflects a snow-capped mountain, with golden light revealing shoreline buildings amidst vibrant autumn foliage ..

![](images/d2e56e4252e9e81c1ac2ee4ac3cdfe2442a89c9d88632ebefc9fe371340d7b9b.jpg)  
(9) Prompt: A sunlit mountain path winds through a lush valley with wildflowers and a rustic cabin. Snow-capped peaks rise above, set against a crisp blue sky, .

![](images/3575fdc7a7319c4482a012b8fdc9d0e8b4e4d449ebacbcd809ae843d5f5b5a0f.jpg)  
(10) Prompt: A sunny alpine path curves along a hillside, bordered by evergreens and a stone wall. A fence overlooks a valley with rooftops, ...

Fig. 9. Open-Domain Results.   
![](images/8dda3468913ec89e1c437f2a85c5b8e9479c1e3f0103b65fe12f26e591afd84d.jpg)  
(11) Prompt: In an arid desert under a burnt-orange sky, a dusty six-wheeled rover stands amid scattered components, evoking a Mars expedition base.
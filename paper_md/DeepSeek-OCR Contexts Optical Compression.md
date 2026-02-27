# DeepSeek-OCR: Contexts Optical Compression

Haoran Wei, Yaofeng Sun, Yukun Li

DeepSeek-AI

# Abstract

We present DeepSeek-OCR as an initial investigation into the feasibility of compressing long contexts via optical 2D mapping. DeepSeek-OCR consists of two components: DeepEncoder and DeepSeek3B-MoE-A570M as the decoder. Specifically, DeepEncoder serves as the core engine, designed to maintain low activations under high-resolution input while achieving high compression ratios to ensure an optimal and manageable number of vision tokens. Experiments show that when the number of text tokens is within 10 times that of vision tokens (i.e., a compression ratio $< 1 0 \times$ ), the model can achieve decoding (OCR) precision of $9 7 \%$ . Even at a compression ratio of $2 0 \times .$ , the OCR accuracy still remains at about $6 0 \%$ . This shows considerable promise for research areas such as historical long-context compression and memory forgetting mechanisms in LLMs. Beyond this, DeepSeek-OCR also demonstrates high practical value. On OmniDocBench, it surpasses GOT-OCR2.0 (256 tokens/page) using only 100 vision tokens, and outperforms MinerU2.0 $6 0 0 0 +$ tokens per page on average) while utilizing fewer than 800 vision tokens. In production, DeepSeek-OCR can generate training data for LLMs/VLMs at a scale of $2 0 0 \mathrm { k } +$ pages per day (a single A100-40G). Codes and model weights are publicly accessible at http://github.com/deepseek-ai/DeepSeek-OCR.

![](images/e46a6b5e5ffc2871e363782c3195c17caf2c3f2c095ddc764c1ae0afd6a22e8d.jpg)  
(a) Compression on Fox benchmark

![](images/e929d957ed1740fdd7d31bf962a28f021eeb1ecf541711e46fd2a225124d7e23.jpg)  
(b) Performance on Omnidocbench   
Figure 1 | Figure (a) shows the compression ratio (number of text tokens in ground truth/number of vision tokens model used) testing on Fox [21] benchmark; Figure (b) shows performance comparisons on OmniDocBench [27]. DeepSeek-OCR can achieve state-of-the-art performance among end-to-end models enjoying the fewest vision tokens.

# Contents

# 1 Introduction 3

# 2 Related Works 4

2.1 Typical Vision Encoders in VLMs 4   
2.2 End-to-end OCR Models . 4

# 3 Methodology 5

3.1 Architecture 5   
3.2 DeepEncoder 5

3.2.1 Architecture of DeepEncoder . 5   
3.2.2 Multiple resolution support . . 6

3.3 The MoE Decoder . . 7   
3.4 Data Engine 7

3.4.1 OCR 1.0 data 7   
3.4.2 OCR 2.0 data 8   
3.4.3 General vision data . 9   
3.4.4 Text-only data . 9

3.5 Training Pipelines . . 9

3.5.1 Training DeepEncoder . . 10   
3.5.2 Training DeepSeek-OCR . 10

# 4 Evaluation 10

4.1 Vision-text Compression Study 10   
4.2 OCR Practical Performance 12

4.3 Qualitative Study . 12

4.3.1 Deep parsing 12   
4.3.2 Multilingual recognition . . 16   
4.3.3 General vision understanding . . . 17

# 5 Discussion 18

# 6 Conclusion 19

# 1. Introduction

Current Large Language Models (LLMs) face significant computational challenges when processing long textual content due to quadratic scaling with sequence length. We explore a potential solution: leveraging visual modality as an efficient compression medium for textual information. A single image containing document text can represent rich information using substantially fewer tokens than the equivalent digital text, suggesting that optical compression through vision tokens could achieve much higher compression ratios.

This insight motivates us to reexamine vision-language models (VLMs) from an LLM-centric perspective, focusing on how vision encoders can enhance LLMs’ efficiency in processing textual information rather than basic VQA [12, 16, 24, 32, 41] what humans excel at. OCR tasks, as an intermediate modality bridging vision and language, provide an ideal testbed for this visiontext compression paradigm, as they establish a natural compression-decompression mapping between visual and textual representations while offering quantitative evaluation metrics.

Accordingly, we present DeepSeek-OCR, a VLM designed as a preliminary proof-of-concept for efficient vision-text compression. Our work makes three primary contributions:

First, we provide comprehensive quantitative analysis of vision-text token compression ratios. Our method achieves $9 6 \% +$ OCR decoding precision at $9 { - } 1 0 \times$ text compression, ${ \sim } 9 0 \%$ a t $1 0 – 1 2 \times$ compression, and $\sim 6 0 \%$ at $2 0 \times$ compression on Fox [21] benchmarks featuring diverse document layouts (with actual accuracy being even higher when accounting for formatting differences between output and ground truth), as shown in Figure 1(a). The results demonstrate that compact language models can effectively learn to decode compressed visual representations, suggesting that larger LLMs could readily acquire similar capabilities through appropriate pretraining design.

Second, we introduce DeepEncoder, a novel architecture that maintains low activation memory and minimal vision tokens even with high-resolution inputs. It serially connects window attention and global attention encoder components through a $1 6 \times$ convolutional compressor. This design ensures that the window attention component processes a large number of vision tokens, while the compressor reduces vision tokens before they enter the dense global attention component, achieving effective memory and token compression.

Third, we develop DeepSeek-OCR based on DeepEncoder and DeepSeek3B-MoE [19, 20]. As shown in Figure 1(b), it achieves state-of-the-art performance within end-to-end models on OmniDocBench while using the fewest vision tokens. Additionally, we equip the model with capabilities for parsing charts, chemical formulas, simple geometric figures, and natural images to enhance its practical utility further. In production, DeepSeek-OCR can generate 33 million pages of data per day for LLMs or VLMs using 20 nodes (each with 8 A100-40G GPUs).

In summary, this work presents a preliminary exploration of using visual modality as an efficient compression medium for textual information processing in LLMs. Through DeepSeek-OCR, we demonstrate that vision-text compression can achieve significant token reduction $( 7 - 2 0 \times )$ for different historical context stages, offering a promising direction for addressing long-context challenges in large language models. Our quantitative analysis provides empirical guidelines for VLM token allocation optimization, while the proposed DeepEncoder architecture showcases practical feasibility with real-world deployment capabilities. Although focused on OCR as a proof-of-concept, this paradigm opens new possibilities for rethinking how vision and language modalities can be synergistically combined to enhance computational efficiency in large-scale text processing and agent systems.

![](images/d810f1f1a1612fc04c5458a3fbce10ee35c57446224542d43c8927942047af6c.jpg)

![](images/2b8aa3a94d27d909a506de360b2fac9207f228b5d124618c427ff73692bc7757.jpg)

![](images/b53b5a2b389ff7f8cafee17bda6f0a136529e75bf39ec118cf73139d34132362.jpg)  
Figure 2 | Typical vision encoders in popular VLMs. Here are three types of encoders commonly used in current open-source VLMs, all of which suffer from their respective deficiencies.

# 2. Related Works

# 2.1. Typical Vision Encoders in VLMs

Current open-source VLMs employ three main types of vision encoders, as illustrated in Figure 2. The first type is a dual-tower architecture represented by Vary [36], which utilizes parallel SAM [17] encoder to increase visual vocabulary parameters for high-resolution image processing. While offering controllable parameters and activation memory, this approach suffers from significant drawbacks: it requires dual image preprocessing that complicates deployment and makes encoder pipeline parallelism challenging during training. The second type is tile-based method exemplified by InternVL2.0 [8], which processes images by dividing them into small tiles for parallel computation, reducing activation memory under high-resolution settings. Although capable of handling extremely high resolutions, this approach has notable limitations due to its typically low native encoder resolution (below $5 1 2 { \times } 5 1 2 ,$ ), causing large images to be excessively fragmented and resulting in numerous vision tokens. The third type is adaptive resolution encoding represented by Qwen2-VL [35], which adopts the NaViT [10] paradigm to directly process full images through patch-based segmentation without tile parallelization. While this encoder can handle diverse resolutions flexibly, it faces substantial challenges with large images due to massive activation memory consumption that can cause GPU memory overflow, and sequence packing requires extremely long sequence lengths during training. Long vision tokens will slow down both prefill and generation phases of inference.

# 2.2. End-to-end OCR Models

OCR, particularly document parsing task, has been a highly active topic in the image-to-text domain. With the advancement of VLMs, a large number of end-to-end OCR models have emerged, fundamentally transforming the traditional pipeline architecture (which required separate detection and recognition expert models) by simplifying OCR systems. Nougat [6] first employs end-to-end framework for academic paper OCR on arXiv, demonstrating the potential of models in handling dense perception tasks. GOT-OCR2.0 [38] expands the scope of OCR2.0 to include more synthetic image parsing tasks and designs an OCR model with performance-efficiency trade-offs, further highlighting the potential of end-to-end OCR researches. Additionally, general vision models such as Qwen-VL series [35], InternVL series [8], and many their derivatives continuously enhance their document OCR capabilities to explore dense visual perception boundaries. However, a crucial research question that current models have not addressed is: for a document containing 1000 words, how many vision tokens are at least needed for decoding? This question holds significant importance for research in the principle that "a picture is worth a thousand words."

![](images/af7df717d906bdec5af432dbe5f080cc2b5bbfc5d5f13f2dc82fecd4bbe85339.jpg)  
Figure 3 | The architecture of DeepSeek-OCR. DeepSeek-OCR consists of a DeepEncoder and a DeepSeek-3B-MoE decoder. DeepEncoder is the core of DeepSeek-OCR, comprising three components: a SAM [17] for perception dominated by window attention, a CLIP [29] for knowledge with dense global attention, and a $1 6 \times$ token compressor that bridges between them.

# 3. Methodology

# 3.1. Architecture

As shown in Figure 3, DeepSeek-OCR enjoys a unified end-to-end VLM architecture consisting of an encoder and a decoder. The encoder (namely DeepEncoder) is responsible for extracting image features and tokenizing as well as compressing visual representations. The decoder is used for generating the required result based on image tokens and prompts. DeepEncoder is approximately 380M in parameters, mainly composed of an 80M SAM-base [17] and a 300M CLIP-large [29] connected in series. The decoder adopts a 3B MoE [19, 20] architecture with 570M activated parameters. In the following paragraphs, we will delve into the model components, data engineering, and training skills.

# 3.2. DeepEncoder

To explore the feasibility of contexts optical compression, we need a vision encoder with the following features: 1.Capable of processing high resolutions; 2.Low activation at high resolutions; 3.Few vision tokens; 4.Support for multiple resolution inputs; 5. Moderate parameter count. However, as described in the Section 2.1, current open-source encoders cannot fully satisfy all these conditions. Therefore, we design a novel vision encoder ourselves, named DeepEncoder.

# 3.2.1. Architecture of DeepEncoder

DeepEncoder mainly consists of two components: a visual perception feature extraction component dominated by window attention, and a visual knowledge feature extraction component with dense global attention. To benefit from the pretraining gains of previous works, we use SAM-base (patch-size 16) and CLIP-large as the main architectures for the two components respectively. For CLIP, we remove the first patch embedding layer since its input is no longer images but output tokens from the previous pipeline. Between the two components, we borrow from Vary [36] and use a 2-layer convolutional module to perform $1 6 \times$ downsampling of vision tokens. Each convolutional layer has a kernel size of 3, stride of 2, padding of 1, and channels increase from 256 to 1024. Assuming we input a $1 0 2 4 { \times } 1 0 2 4$ image, the DeepEncoder will segment it into $1 0 2 4 / 1 6 { \times } 1 0 2 4 / 1 6 { = } 4 0 9 6$ patch tokens. Since the first half of encoder is dominated by window attention and only 80M, the activation is acceptable. Before entering global attention,

![](images/15a41c91e145dc87b61ea928905650363667a0fe1424722cd5b116f1eecaa216.jpg)

![](images/b5117da7dc32b264b24e4a4e5e9ea8387dc1195572b068655edc1ae8a6549923.jpg)  
Figure 4 | To test model performance under different compression ratios (requiring different numbers of vision tokens) and enhance the practicality of DeepSeek-OCR, we configure it with multiple resolution modes.

the 4096 tokens go through the compression module and the token count becomes 4096/16=256, thus making the overall activation memory controllable.

Table 1 | Multi resolution support of DeepEncoder. For both research and application purposes, we design DeepEncoder with diverse native resolution and dynamic resolution modes.   

<table><tr><td rowspan="2">Mode</td><td colspan="4">Native Resolution</td><td colspan="2">Dynamic Resolution</td></tr><tr><td>Tiny</td><td>Small</td><td>Base</td><td>Large</td><td>Gundam</td><td>Gundam-M</td></tr><tr><td>Resolution</td><td>512</td><td>640</td><td>1024</td><td>1280</td><td>640+1024</td><td>1024+1280</td></tr><tr><td>Tokens</td><td>64</td><td>100</td><td>256</td><td>400</td><td>n×100+256</td><td>n×256+400</td></tr><tr><td>Process</td><td>resize</td><td>resize</td><td>padding</td><td>padding</td><td>resize + padding</td><td>resize + padding</td></tr></table>

# 3.2.2. Multiple resolution support

Suppose we have an image with 1000 optical characters and we want to test how many vision tokens are needed for decoding. This requires the model to support a variable number of vision tokens. That is to say the DeepEncoder needs to support multiple resolutions.

We meet the requirement aforementioned through dynamic interpolation of positional encodings, and design several resolution modes for simultaneous model training to achieve the capability of a single DeepSeek-OCR model supporting multiple resolutions. As shown in Figure 4, DeepEncoder mainly supports two major input modes: native resolution and dynamic resolution. Each of them contains multiple sub-modes.

Native resolution supports four sub-modes: Tiny, Small, Base, and Large, with corresponding resolutions and token counts of $5 1 2 \times 5 1 2$ (64), $6 4 0 { \times } 6 4 0$ (100), $1 0 2 4 \times 1 0 2 4$ (256), and $1 2 8 0 \times 1 2 8 0$ (400) respectively. Since Tiny and Small modes have relatively small resolutions, to avoid wasting vision tokens, images are processed by directly resizing the original shape. For Base and Large modes, in order to preserve the original image aspect ratio, images are padded to the corresponding size. After padding, the number of valid vision tokens is less than the actual number of vision tokens, with the calculation formula being:

$$
N _ {\text {v a l i d}} = \lceil N _ {\text {a c t u a l}} \times [ 1 - ((\max (w, h) - \min (w, h)) / (\max (w, h))) ] \rceil \tag {1}
$$

where $w$ and $h$ represent the width and height of the original input image.

Dynamic resolution can be composed of two native resolutions. For example, Gundam mode consists of $\mathrm { n } { \times } 6 4 0 { \times } 6 4 0$ tiles (local views) and a $1 0 2 4 \times 1 0 2 4$ global view. The tiling method following InternVL2.0 [8]. Supporting dynamic resolution is mainly for application considerations, especially for ultra-high-resolution inputs (such as newspaper images). Tiling is a form of secondary window attention that can effectively reduce activation memory further. It’s worth noting that due to our relatively large native resolutions, images won’t be fragmented too much under dynamic resolution (the number of tiles is controlled within the range of 2 to 9). The vision token number output by the DeepEncoder under Gundam mode is: $n \times 1 0 0 + 2 5 6 .$ , where ?? is the number of tiles. For images with both width and height smaller than 640, $n$ is set to 0, i.e., Gundam mode will degrade to Base mode.

Gundam mode is trained together with the four native resolution modes to achieve the goal of one model supporting multiple resolutions. Note that Gundam-master mode $1 0 2 4 \times 1 0 2 4$ local views $+ 1 2 8 0 { \times } 1 2 8 0$ global view) is obtained through continued training on a trained DeepSeek-OCR model. This is mainly for load balancing, as Gundam-master’s resolution is too large and training it together would slow down the overall training speed.

# 3.3. The MoE Decoder

Our decoder uses the DeepSeekMoE [19, 20], specifically DeepSeek-3B-MoE. During inference, the model activates 6 out of 64 routed experts and 2 shared experts, with about 570M activated parameters. The 3B DeepSeekMoE is very suitable for domain-centric (OCR for us) VLM research, as it obtains the expressive capability of a 3B model while enjoying the inference efficiency of a 500M small model.

The decoder reconstructs the original text representation from the compressed latent vision tokens of DeepEncoder as:

$$
f _ {\mathrm {d e c}}: \mathbb {R} ^ {n \times d _ {\text {l a t e n t}}} \rightarrow \mathbb {R} ^ {N \times d _ {\text {t e x t}}}; \quad \hat {\mathbf {X}} = f _ {\mathrm {d e c}} (\mathbf {Z}) \quad \text {w h e r e} n \leq N \tag {2}
$$

where $\mathbf { Z } \in \mathbb { R } ^ { n \times d _ { \mathrm { l a t e n t } } }$ are the compressed latent(vision) tokens from DeepEncoder and $\hat { \mathbf { X } } \in \mathbb { R } ^ { N \times d _ { \mathrm { t e x t } } }$ is the reconstructed text representation. The function $f _ { \mathrm { d e c } }$ represents a non-linear mapping that can be effectively learned by compact language models through OCR-style training. It is reasonable to conjecture that LLMs, through specialized pretraining optimization, would demonstrate more natural integration of such capabilities.

# 3.4. Data Engine

We constructe complex and diverse training data for DeepSeek-OCR, including OCR 1.0 data, which mainly consists of traditional OCR tasks such as scene image OCR and document OCR; OCR 2.0 data, which mainly includes parsing tasks for complex artificial images, such as common charts, chemical formulas, and plane geometry parsing data; General vision data, which is mainly used to inject certain general image understanding capabilities into DeepSeek-OCR and preserve the general vision interface.

# 3.4.1. OCR 1.0 data

Document data is the top priority for DeepSeek-OCR. We collect 30M pages of diverse PDF data covering about 100 languages from the Internet, with Chinese and English accounting for approximately 25M and other languages accounting for 5M. For this data, we create two types of ground truth: coarse annotations and fine annotations. Coarse annotations are extracted

![](images/2d9ece5b2d4575dc770827c33d3696838a56a309c08dd6064a4a541b8da48f16.jpg)  
Prvi dan trcanja Tomislav moze izabrati na 7 razlicitih nacina. Drugi dan trcanja moze izabrati na 4 razlicita nacina postujuci uvjet da ne trci dva dana za redom. Time dobiva ukupno 7 · 4 = 28 mogucnosti no svaka od njih je na taj nacin brojana dva puta (npr. PO-SR i SR-PO). Stoga je ukupan broj razlicitih rasporeda trcanja:   
14. Masa Zeli popuniti tablicu tako da u svaku celiju upise jedan broj. Za sada je upisala dva broja kako je prikazano na slici. Tablicu Zeli popuniti takodaje zbroj svih upisanih brojeva 35, zbroj brojeva u prve tri celije ie22.azbroibroievaunosliednietriceliie25Kolikiieumnozakbroieyakoieceunisatiusiveceliie?

![](images/854ae8d733f24d69fba6845b9e5d7d33d67cc2d979f49813e2f8a75fa8eed3d2.jpg)  
C0   
D)48   
E)39

# Rjesenje: A) 63

Sive celije su druga i Cetvrta pa trazimo brojeve koje ce Masa u njih upisati. Kako zbroj brojeva utablici mora biti 35 to je zbroj brojeva u drugoj,trecoji etvrtoj celiji35–3−4 = 28. Kako zbroj brojeva u prve tri celije mora biti 22 toje zbroj brojeva u drugoj itrecoj celiji 22-3=19. Kako zbroj brojevauposljednje tricelijemora biti 25toje zbroj brojevautrecoj icetvrtoj celiji25-4=21. Toznacidaje brojutrecoj eliji 19+21-28=12.Ondaje brojudrugoj celiji19-12=7,a broj ucetvrtoj celiji 21–12 = 9.Umnozak tih brojeva je 63.

# 2. nacin:

Oznacimo s a, b i c brojeve koji nedostaju u tablici

![](images/35e8010c35d31b4e8752566e19431900ff6d2d8f880b417d5df86a6e354c7e1f.jpg)  
Figure 5 | OCR 1.0 fine annotations display. We format the ground truth into an interleaved layout and text format, where each paragraph of text is preceded by the coordinates and label of it in the original image. All coordinates are normalized into 1000 bins.

# Trazimo umnozak brojeva a i c.

Kako zbroj brojeva u tablici mora biti 35 to je 3 + a + b + c + 4 = 35 odnosno: (1）a+b+c=28.

Kako zbroj brojeva uprve tri celije mora biti 22 toje3 +a +b =22 odnosno:

(2)𝛼 +𝑏=19.

Kako zbrojbrojevau posljednje tricelije mora biti 25 to je b + c + 4 = 25 odnosno (3)b+c=21.

# (a) Ground truth image

<|ref >text<|/ref|><|det|>[55，43，130,60]]<|/det|> 2.nacin:

<|ref|>image<|/ref|><|det|>[70,93，450,360]]<|/det|>

<|ref|>text<|/ref|><|det|>[[460，95，896，132]]<|/det|>

Prvi dan tranja Tomislav moze izabrati na 7 razliitih nacina.

<|ref|>text<|/ref|><|det|>[[460,131,880,168]]<|/det|>

Drugi dan tr&anja moze izabrati na 4 razlicita nacina poätujuci uvjet da ne tri dva dana za redom.

<|ref|>text<|/ref|><|det|>[[460，166，941，220]]<|/det|>

Time dobiva ukupno \(7 \cdot 4 = 28\） mogucnosti no svaka od njih je na taj nain brojana dva puta (npr.Po-SR i SR-PO).Stoga je ukupan broj razlicitih rasporeda trcanja:

<| ref|>equation<|/ref|><|det|>[[460，217,550，256]]<|/det|>

[\frac{7 \cdot 4}{2} = 14.]

<|ref|>text<|/ref|><|det|>[[55，397，931，452]]<|/det|>

14.Masa zeli popuniti tablicu tako da u svaku celiju upise jedan broj，Za sada je upisala dva broja kako je prikazano na slici. Tablicu zeli popuniti tako da je zbroj svih upisanih brojeva 35,zbroj brojeva u prve tri celije je 22,a zbroj brojeva u posljednje tri celije 25.Koliki je umnozak brojeva koje ce upisati u sive celije?

<|ref|>table<|/ref|><|det|>[57,450，360,500]]<|/det|>

<table><tr><td>3</td><td></td><td></td><td>4</td></tr></table>

<|ref|>text<|/ref|><|det|>[[55，515，110，534]]<|/det|>A）63

<|ref|>text<|/ref|><|det|>[[230,515，293,534]]<|/det|>B）108

<|ref|>text<|/ref|><|det|>[[405,515,450,534]]<|/det|> c）0

<|ref|>text<|/ref|><|det|>[[581,515,636,534]]<|/det|> D）48

# (b) Fine annotations with layouts

directly from the full dataset using fitz, aimed at teaching the model to recognize optical text, especially in minority languages. Fine annotations include 2M pages each for Chinese and English, labeled using advanced layout models (such as PP-DocLayout [33]) and OCR models (such as MinuerU [34] and GOT-OCR2.0 [38]) to construct detection and recognition interleaved data. For minority languages, in the detection part, we find that the layout model enjoys certain generalization capabilities. In the recognition part, we use fitz to create small patch data to train a GOT-OCR2.0, then use the trained model to label small patches after layout processing, employing a model flywheel to create 600K data samples. During the training of DeepSeek-OCR, coarse labels and fine labels are distinguished using different prompts. The ground truth for fine annotation image-text pairs can be seen in Figure 5. We also collect 3M Word data, constructing high-quality image-text pairs without layout by directly extracting content. This data mainly brings benefits to formulas and HTML-formatted tables. Additionally, we select some open-source data [28, 37] as supplements.

For natural scene OCR, our model mainly supports Chinese and English. The image data sources come from LAION [31] and Wukong [13], labeled using PaddleOCR [9], with 10M data samples each for Chinese and English. Like document OCR, natural scene OCR can also control whether to output detection boxes through prompts.

# 3.4.2. OCR 2.0 data

Following GOT-OCR2.0 [38], we refer to chart, chemical formula, and plane geometry parsing data as OCR 2.0 data. For chart data, following OneChart [7], we use pyecharts and matplotlib

![](images/b0d6a6b4a3ef7cdfd56cb7fa9c152c09d1887270092febdfb2063d19fc6913bf.jpg)  
(a) Image-text ground truth of chart

![](images/13a79d858b3413934402e45afd209d2633b2d0c69824bb01f06a7317585e38f9.jpg)  
(b) Image-text ground truth of geometry   
Figure 6 | For charts, we do not use OneChart’s [7] dictionary format, but instead use HTML table format as labels, which can save a certain amount of tokens. For plane geometry, we convert the ground truth to dictionary format, where the dictionary contains keys such as line segments, endpoint coordinates, line segment types, etc., for better readability. Each line segment is encoded using the Slow Perception [39] manner.

to render 10M images, mainly including commonly used line, bar, pie, and composite charts. We define chart parsing as image-to-HTML-table conversion task, as shown in Figure 6(a). For chemical formulas, we utilize SMILES format from PubChem as the data source and render them into images using RDKit, constructing 5M image-text pairs. For plane geometry images, we follow Slow Perception [39] for generation. Specifically, we use perception-ruler size as 4 to model each line segment. To increase the diversity of rendered data, we introduce geometric translation-invariant data augmentation, where the same geometric image is translated in the original image, corresponding to the same ground truth drawn at the centered position in the coordinate system. Based on this, we construct a total of 1M plane geometry parsing data, as illustrated in Figure 6(b).

# 3.4.3. General vision data

DeepEncoder can benefit from CLIP’s pretraining gains and has sufficient parameters to incorporate general visual knowledge. Therefore, we also prepare some corresponding data for DeepSeek-OCR. Following DeepSeek-VL2 [40], we generate relevant data for tasks such as caption, detection, and grounding. Note that DeepSeek-OCR is not a general VLM model, and this portion of data accounts for only $2 0 \%$ of the total data. We introduce such type of data mainly to preserve the general vision interface, so that researchers interested in our model and general vision task can conveniently advance their work in the future.

# 3.4.4. Text-only data

To ensure the model’s language capabilities, we introduced $1 0 \%$ of in-house text-only pretrain data, with all data processed to a length of 8192 tokens, which is also the sequence length for DeepSeek-OCR. In summary, when training DeepSeek-OCR, OCR data accounts for $7 0 \%$ general vision data accounts for $2 0 \%$ , and text-only data accounts for $1 0 \%$ .

# 3.5. Training Pipelines

Our training pipeline is very simple and consists mainly of two stages: a).Training DeepEncoder independently; b).Training the DeepSeek-OCR. Note that the Gundam-master mode is obtained by continuing training on a pre-trained DeepSeek-OCR model with 6M sampled data. Since the training protocol is identical to other modes, we omit the detailed description hereafter.

# 3.5.1. Training DeepEncoder

Following Vary [36], we utilize a compact language model [15] and use the next token prediction framework to train DeepEncoder. In this stage, we use all OCR 1.0 and 2.0 data aforementioned, as well as 100M general data sampled from the LAION [31] dataset. All data is trained for 2 epochs with a batch size of 1280, using the AdamW [23] optimizer with cosine annealing scheduler [22] and a learning rate of 5e-5. The training sequence length is 4096.

# 3.5.2. Training DeepSeek-OCR

After DeepEncoder is ready, we use data mentioned in Section 3.4 to train the DeepSeek-OCR. with the entire training process conducted on the HAI-LLM [14] platform. The entire model uses pipeline parallelism (PP) and is divided into 4 parts, with DeepEncoder taking two parts and the decoder taking two parts. For DeepEncoder, we treat SAM and the compressor as the vision tokenizer, place them in PP0 and freeze their parameters, while treating the CLIP part as input embedding layer and place it in PP1 with unfrozen weights for training. For the language model part, since DeepSeek3B-MoE has 12 layers, we place 6 layers each on PP2 and PP3. We use 20 nodes (each with 8 A100-40G GPUs) for training, with a data parallelism (DP) of 40 and a global batch size of 640. We use the AdamW optimizer with a step-based scheduler and an initial learning rate of 3e-5. For text-only data, the training speed is 90B tokens/day, while for multimodal data, the training speed is 70B tokens/day.

Table 2 | We test DeepSeek-OCR’s vision-text compression ratio using all English documents with 600-1300 tokens from the Fox [21] benchmarks. Text tokens represent the number of tokens after tokenizing the ground truth text using DeepSeek-OCR’s tokenizer. Vision Tokens ${ = } 6 4$ or 100 respectively represent the number of vision tokens output by DeepEncoder after resizing input images to $5 1 2 \times 5 1 2$ and $6 4 0 { \times } 6 4 0$ .

<table><tr><td rowspan="2">Text Tokens</td><td colspan="2">Vision Tokens =64</td><td colspan="2">Vision Tokens=100</td><td rowspan="2">Pages</td></tr><tr><td>Precision</td><td>Compression</td><td>Precision</td><td>Compression</td></tr><tr><td>600-700</td><td>96.5%</td><td>10.5×</td><td>98.5%</td><td>6.7×</td><td>7</td></tr><tr><td>700-800</td><td>93.8%</td><td>11.8×</td><td>97.3%</td><td>7.5×</td><td>28</td></tr><tr><td>800-900</td><td>83.8%</td><td>13.2×</td><td>96.8%</td><td>8.5×</td><td>28</td></tr><tr><td>900-1000</td><td>85.9%</td><td>15.1×</td><td>96.8%</td><td>9.7×</td><td>14</td></tr><tr><td>1000-1100</td><td>79.3%</td><td>16.5×</td><td>91.5%</td><td>10.6×</td><td>11</td></tr><tr><td>1100-1200</td><td>76.4%</td><td>17.7×</td><td>89.8%</td><td>11.3×</td><td>8</td></tr><tr><td>1200-1300</td><td>59.1%</td><td>19.7×</td><td>87.1%</td><td>12.6×</td><td>4</td></tr></table>

# 4. Evaluation

# 4.1. Vision-text Compression Study

We select Fox [21] benchmarks to verify DeepSeek-OCR’s compression-decompression capability for text-rich documents, in order to preliminarily explore the feasibility and boundaries of contexts optical compression. We use the English document portion of Fox, tokenize the ground truth text with DeepSeek-OCR’s tokenizer (vocabulary size of approximately 129k), and select documents with 600-1300 tokens for testing, which happens to be 100 pages. Since the number of text tokens is not large, we only need to test performance in Tiny and Small modes, where Tiny mode corresponds to 64 tokens and Small mode corresponds to 100 tokens. We use the prompt

Table 3 | We use OmniDocBench [27] to test the performance of DeepSeek-OCR on real document parsing tasks. All metrics in the table are edit distances, where smaller values indicate better performance. "Tokens" represents the average number of vision tokens used per page, and "†200dpi" means using fitz to interpolate the original image to 200dpi. For the DeepSeek-OCR model, the values in parentheses in the "Tokens" column represent valid vision tokens, calculated according to Equation 1.   

<table><tr><td rowspan="2">Model</td><td rowspan="2">Tokens</td><td colspan="5">English</td><td colspan="5">Chinese</td></tr><tr><td>overall</td><td>text</td><td>formula</td><td>table</td><td>order</td><td>overall</td><td>text</td><td>formula</td><td>table</td><td>order</td></tr><tr><td colspan="12">Pipeline Models</td></tr><tr><td>Dolphin [11]</td><td>-</td><td>0.356</td><td>0.352</td><td>0.465</td><td>0.258</td><td>0.35</td><td>0.44</td><td>0.44</td><td>0.604</td><td>0.367</td><td>0.351</td></tr><tr><td>Marker [1]</td><td>-</td><td>0.296</td><td>0.085</td><td>0.374</td><td>0.609</td><td>0.116</td><td>0.497</td><td>0.293</td><td>0.688</td><td>0.678</td><td>0.329</td></tr><tr><td>Mathpix [2]</td><td>-</td><td>0.191</td><td>0.105</td><td>0.306</td><td>0.243</td><td>0.108</td><td>0.364</td><td>0.381</td><td>0.454</td><td>0.32</td><td>0.30</td></tr><tr><td>MinerU-2.1.1 [34]</td><td>-</td><td>0.162</td><td>0.072</td><td>0.313</td><td>0.166</td><td>0.097</td><td>0.244</td><td>0.111</td><td>0.581</td><td>0.15</td><td>0.136</td></tr><tr><td>MonkeyOCR-1.2B [18]</td><td>-</td><td>0.154</td><td>0.062</td><td>0.295</td><td>0.164</td><td>0.094</td><td>0.263</td><td>0.179</td><td>0.464</td><td>0.168</td><td>0.243</td></tr><tr><td>PPstructure-v3 [9]</td><td>-</td><td>0.152</td><td>0.073</td><td>0.295</td><td>0.162</td><td>0.077</td><td>0.223</td><td>0.136</td><td>0.535</td><td>0.111</td><td>0.11</td></tr><tr><td colspan="12">End-to-end Models</td></tr><tr><td>Nougat [6]</td><td>2352</td><td>0.452</td><td>0.365</td><td>0.488</td><td>0.572</td><td>0.382</td><td>0.973</td><td>0.998</td><td>0.941</td><td>1.00</td><td>0.954</td></tr><tr><td>SmolDocling [25]</td><td>392</td><td>0.493</td><td>0.262</td><td>0.753</td><td>0.729</td><td>0.227</td><td>0.816</td><td>0.838</td><td>0.997</td><td>0.907</td><td>0.522</td></tr><tr><td>InternVL2-76B [8]</td><td>6790</td><td>0.44</td><td>0.353</td><td>0.543</td><td>0.547</td><td>0.317</td><td>0.443</td><td>0.29</td><td>0.701</td><td>0.555</td><td>0.228</td></tr><tr><td>Qwen2.5-VL-7B [5]</td><td>3949</td><td>0.316</td><td>0.151</td><td>0.376</td><td>0.598</td><td>0.138</td><td>0.399</td><td>0.243</td><td>0.5</td><td>0.627</td><td>0.226</td></tr><tr><td>OLMOCR [28]</td><td>3949</td><td>0.326</td><td>0.097</td><td>0.455</td><td>0.608</td><td>0.145</td><td>0.469</td><td>0.293</td><td>0.655</td><td>0.652</td><td>0.277</td></tr><tr><td>GOT-OCR2.0 [38]</td><td>256</td><td>0.287</td><td>0.189</td><td>0.360</td><td>0.459</td><td>0.141</td><td>0.411</td><td>0.315</td><td>0.528</td><td>0.52</td><td>0.28</td></tr><tr><td>OCRFlux-3B [3]</td><td>3949</td><td>0.238</td><td>0.112</td><td>0.447</td><td>0.269</td><td>0.126</td><td>0.349</td><td>0.256</td><td>0.716</td><td>0.162</td><td>0.263</td></tr><tr><td>GPT4o [26]</td><td>-</td><td>0.233</td><td>0.144</td><td>0.425</td><td>0.234</td><td>0.128</td><td>0.399</td><td>0.409</td><td>0.606</td><td>0.329</td><td>0.251</td></tr><tr><td>InternVL3-78B [42]</td><td>6790</td><td>0.218</td><td>0.117</td><td>0.38</td><td>0.279</td><td>0.095</td><td>0.296</td><td>0.21</td><td>0.533</td><td>0.282</td><td>0.161</td></tr><tr><td>Qwen2.5-VL-72B [5]</td><td>3949</td><td>0.214</td><td>0.092</td><td>0.315</td><td>0.341</td><td>0.106</td><td>0.261</td><td>0.18</td><td>0.434</td><td>0.262</td><td>0.168</td></tr><tr><td>dots.ocr [30]</td><td>3949</td><td>0.182</td><td>0.137</td><td>0.320</td><td>0.166</td><td>0.182</td><td>0.261</td><td>0.229</td><td>0.468</td><td>0.160</td><td>0.261</td></tr><tr><td>Gemini2.5-Pro [4]</td><td>-</td><td>0.148</td><td>0.055</td><td>0.356</td><td>0.13</td><td>0.049</td><td>0.212</td><td>0.168</td><td>0.439</td><td>0.119</td><td>0.121</td></tr><tr><td>MinerU2.0 [34]</td><td>6790</td><td>0.133</td><td>0.045</td><td>0.273</td><td>0.15</td><td>0.066</td><td>0.238</td><td>0.115</td><td>0.506</td><td>0.209</td><td>0.122</td></tr><tr><td>dots.ocr+200dpi [30]</td><td>5545</td><td>0.125</td><td>0.032</td><td>0.329</td><td>0.099</td><td>0.04</td><td>0.16</td><td>0.066</td><td>0.416</td><td>0.092</td><td>0.067</td></tr><tr><td colspan="12">DeepSeek-OCR (end2end)</td></tr><tr><td>Tiny</td><td>64</td><td>0.386</td><td>0.373</td><td>0.469</td><td>0.422</td><td>0.283</td><td>0.361</td><td>0.307</td><td>0.635</td><td>0.266</td><td>0.236</td></tr><tr><td>Small</td><td>100</td><td>0.221</td><td>0.142</td><td>0.373</td><td>0.242</td><td>0.125</td><td>0.284</td><td>0.24</td><td>0.53</td><td>0.159</td><td>0.205</td></tr><tr><td>Base</td><td>256(182)</td><td>0.137</td><td>0.054</td><td>0.267</td><td>0.163</td><td>0.064</td><td>0.24</td><td>0.205</td><td>0.474</td><td>0.1</td><td>0.181</td></tr><tr><td>Large</td><td>400(285)</td><td>0.138</td><td>0.054</td><td>0.277</td><td>0.152</td><td>0.067</td><td>0.208</td><td>0.143</td><td>0.461</td><td>0.104</td><td>0.123</td></tr><tr><td>Gundam</td><td>795</td><td>0.127</td><td>0.043</td><td>0.269</td><td>0.134</td><td>0.062</td><td>0.181</td><td>0.097</td><td>0.432</td><td>0.089</td><td>0.103</td></tr><tr><td>Gundam-M+200dpi</td><td>1853</td><td>0.123</td><td>0.049</td><td>0.242</td><td>0.147</td><td>0.056</td><td>0.157</td><td>0.087</td><td>0.377</td><td>0.08</td><td>0.085</td></tr></table>

without layout: "<image>\nFree OCR." to control the model’s output format. Nevertheless, the output format still cannot completely match Fox benchmarks, so the actual performance would be somewhat higher than the test results.

As shown in Table 2, within a $1 0 \times$ compression ratio, the model’s decoding precision can reach approximately $9 7 \%$ , which is a very promising result. In the future, it may be possible to achieve nearly $1 0 \times$ lossless contexts compression through text-to-image approaches. When the compression ratio exceeds $1 0 \times ,$ , performance begins to decline, which may have two reasons: one is that the layout of long documents becomes more complex, and another reason may be that long texts become blurred at $5 1 2 \times 5 1 2$ or $6 4 0 { \times } 6 4 0$ resolution. The first issue can be solved by rendering texts onto a single layout page, while we believe the second issue will become

a feature of the forgetting mechanism. When compressing tokens by nearly $2 0 \times ,$ , we find that precision can still approach $6 0 \%$ . These results indicate that optical contexts compression is a very promising and worthwhile research direction, and this approach does not bring any overhead because it can leverage VLM infrastructure, as multimodal systems inherently require an additional vision encoder.

Table 4 | Edit distances for different categories of documents in OmniDocBench. The results show that some types of documents can achieve good performance with just 64 or 100 vision tokens, while others require Gundam mode.   

<table><tr><td>Type Mode</td><td>Book Slides</td><td>Financial Report</td><td>Textbook</td><td>Exam Paper</td><td>Magazine</td><td>Academic Papers</td><td>Notes</td><td>Newspaper</td><td>Overall</td></tr><tr><td>Tiny</td><td>0.147</td><td>0.116</td><td>0.207</td><td>0.173</td><td>0.294</td><td>0.201</td><td>0.395</td><td>0.297</td><td>0.94</td></tr><tr><td>Small</td><td>0.085</td><td>0.111</td><td>0.079</td><td>0.147</td><td>0.171</td><td>0.107</td><td>0.131</td><td>0.187</td><td>0.744</td></tr><tr><td>Base</td><td>0.037</td><td>0.08</td><td>0.027</td><td>0.1</td><td>0.13</td><td>0.073</td><td>0.052</td><td>0.176</td><td>0.645</td></tr><tr><td>Large</td><td>0.038</td><td>0.108</td><td>0.022</td><td>0.084</td><td>0.109</td><td>0.06</td><td>0.053</td><td>0.155</td><td>0.353</td></tr><tr><td>Gundam</td><td>0.035</td><td>0.085</td><td>0.289</td><td>0.095</td><td>0.094</td><td>0.059</td><td>0.039</td><td>0.153</td><td>0.122</td></tr><tr><td>Guandam-M</td><td>0.052</td><td>0.09</td><td>0.034</td><td>0.091</td><td>0.079</td><td>0.079</td><td>0.048</td><td>0.1</td><td>0.099</td></tr></table>

# 4.2. OCR Practical Performance

DeepSeek-OCR is not only an experimental model; it has strong practical capabilities and can construct data for LLM/VLM pretraining. To quantify OCR performance, we test DeepSeek-OCR on OmniDocBench [27], with results shown in Table 3. Requiring only 100 vision tokens $6 4 0 { \times } 6 4 0$ resolution), DeepSeek-OCR surpasses GOT-OCR2.0 [38] which uses 256 tokens; with 400 tokens (285 valid tokens, $1 2 8 0 \times 1 2 8 0$ resolution), it achieves on-par performance with stateof-the-arts on this benchmark. Using fewer than 800 tokens (Gundam mode), DeepSeek-OCR outperforms MinerU2.0 [34] which needs nearly 7,000 vision tokens. These results demonstrate that our DeepSeek-OCR model is powerful in practical applications, and because the higher tokens compression, it enjoys a higher research ceiling.

As shown in Table 4, some categories of documents require very few tokens to achieve satisfactory performance, such as slides which only need 64 vision tokens. For book and report documents, DeepSeek-OCR can achieve good performance with only 100 vision tokens. Combined with the analysis from Section 4.1, this may be because most text tokens in these document categories are within 1,000, meaning the vision-token compression ratio does not exceed $1 0 \times$ . For newspapers, Gundam or even Gundam-master mode is required to achieve acceptable edit distances, because the text tokens in newspapers are 4-5,000, far exceeding the $1 0 \times$ compression of other modes. These experimental results further demonstrate the boundaries of contexts optical compression, which may provide effective references for researches on the vision token optimization in VLMs and context compression, forgetting mechanisms in LLMs.

# 4.3. Qualitative Study

# 4.3.1. Deep parsing

DeepSeek-OCR possesses both layout and OCR 2.0 capabilities, enabling it to further parse images within documents through secondary model calls, a feature we refer to as "deep parsing". As shown in Figures 7,8,9,10, our model can perform deep parsing on charts, geometry, chemical formulas, and even natural images, requiring only a unified prompt.

![](images/2f7e98a4d24089bb5558f4f657b85bf401cc2d29eb2b0ec4b13603781c5d1f08.jpg)  
Input image

![](images/70a9cfa7f017e6315e5fb7ce09356970259a5fa76846a5e5567589646339bf6d.jpg)

![](images/ba98bb4edb8510c6ea53fbdcaf3547d4ef7547ddbe5d24c89995291b170917e0.jpg)  
Deep Parsing

![](images/e0ac2d140a05c81b0d01b8ee5fb1ec007b491cedacc9bb0e96d1176173c65e28.jpg)  
Rendering   
Figure 7 | In the field of financial research reports, the deep parsing mode of DeepSeek-OCR can be used to obtain structured results of charts within documents. Charts are a crucial form of data representation in finance and scientific fields, and the chart structured extraction is an indispensable capability for future OCR models.

![](images/bc6641547fcee388934c2fe00dbb0a1d673f12b848ecebd876c5350f3dea6df9.jpg)  
Figure 8 | For books and articles, the deep parsing mode can output dense captions for natural images in the documents. With just a prompt, the model can automatically identify what type of image it is and output the required results.

WO 2013/171642

PCT/IB2013/053771

[00369] The title compound was prepared in an analogous fashion to that described in Stage 2.1 using 5-bromo-6-chloro-N-(4-(chlorodifluoromethoxy)phenyl)nicotinamide(Stage 22.2)and 2-methylamino-etanol toafrd awhite crystalinesolid. HPLC(Condiion $4 ) t _ { \mathrm { R } } = 5 . 7 2$ min, UPLC-MS (Condition $3 ) \mathfrak { t } _ { \mathtt { R } } =$ 1.14 min, m/z = 452.2 [M+H]+.

# Example 24

N-(4-(Chlorodifluoromethoxy )phenvn-6-(ethyl(2-hvdroxyethyl )amino)-5-(IH-pyrazol-5-

![](images/19208ef6ba9b2d2ba9e87f79d3b5c033d95cad8a69b6944bbd9a2cf31b7bef45.jpg)  
vDnicotinamide

[00370] The title compound was prepared in an analogous fashion to that described in Example 26 using 5-bromo-N-(4-(chlorodifluoromethoxy)phenyl)-6-(ethyl(2-

hydroxyethyl)amino)nicotinamide (Stage 24.1) and 1-(tetrahydro-2H-pyran-2-yl)-5-(4,4,5,5- tetramethyl-1,3,2-dioxaborolan-2-yl)-lH-pyrazoleto afford a yellow solid. UPLC-MS (Conditior 3)tg = 1.02 min, m/z = 452.2 [M+H]+,m/z = 450.1 [M-H]-; 1H-N MR (400 MHz, DMSO-d 6) δ ppm 0.93 (t,J= 7.09 Hz,3H) 3.17 -3.27 (m,2 H)3.35 -3.43 (m,2 H) 3.43-3.53 (m,2 H) 4.59 (br.s,1H)6.53 (d,J= 1.96 Hz,1H) 7.33 (d,J= 9.05 Hz,2 H) 7.76 (br.s,1H) 7.82- 7.95 (m,2 H) 8.13 (d,J = 2.45 Hz, 1H) 8.72 (d,J= 2.45 Hz,1H) 10.29 (s,1H) 12.98 (br.s,1H)

[00371] Stage 24.1 5-Bromo-N-(4-(chlorodifluoromethoxy)phenyl)-6-(ethyl(2- hydroxyethyl)amino)nicotinamide

![](images/a5d181ea08c5b104a5c65af555db291afcd42532b9a4c03ff1160ab3ace67a93.jpg)

[00372] The title compound was prepared in an analogous fashion to that described in Stage 22.1using5-bromo-6-chloro-N-(4-(chlorodifluoromethoxy)phenyl)nicotinamide(Stage

# <image>\n<|grounding|>Convert the document to markdown.

WO 2013/171642

PCT/IB2013/053771

[00369] The title compound was prepared in an analogous fashion to that described in Stage 22.1 using 5-bromo-6-chloro-N-(4-(chlorodifluoromethoxy)phenyl)nicotinamide(Stage 22.2) and 2-methylamino-ethanol to affrd a white crystalline solid. HPLC (Condition 4)t = 5.72 min,UPLC-MS (Condition 3)tg=1.14min,m/z =452.2[M+H]+

# Example 24

N-(4-(Chlorodifluoromethoxy)phenvn-6-(ethyl(2-hvdroxyethyl)amino)-5-(IH-pyrazol-5-

![](images/c22c92eae4f87651cb1b2b288fe16b23330e9d59cf3a32b297e6adad789d4b4a.jpg)  
vDnicotinamide

[00370] The titlecompound was prepared in an analogous fashion to that described in Example26using5-bromo-N-(4-(chlorodifluoromethoxy)phenyl)-6-(ethyl(2

hydroxyethyl)amino)nicotinamide(Stage 24.1) and1-(tetrahydro-2H-pyran-2-yl)-5-(4,4,5,5 tetramethyl-13,2-dioxaborolan-2-yl)-H-pyrazoletoaffordayellow solid.UPLC-MS(Condition 3)tg= 1.02 min, m/z = 452.2 [M+H]+,m/z = 450.1 [M-H]°;1H-NMR (400 MHz, DMSO-d6)δ ppm 0.93 (t,J= 7.09 Hz,3 H)3.17 -3.27 (m,2 H) 3.35-3.43 (m,2H) 3.43 -3.53 (m,2 H) 4.59 (br. s,1H) 6.53 (d,J= 1.96 Hz,1H) 7.33 (d,J= 9.05 Hz,2 H) 7.76 (br.s,1H) 7.82 -7.95 (m,2 H)8.13 (d,J = 2.45 Hz,1H)8.72 (d, J = 2.45 Hz,1H) 10.29 (s,1H) 12.98 (br. s,1H).

[00371] Stage24.15-Bromo-N-(4-(chlorodifluoromethoxy)phenyl)-6-(ethyl(2 hydroxyethyl)amino)nicotinamide

![](images/265ecc858564a673dbab1274bf0cda3c2563773385aa9726fff0a55f7c05bc1a.jpg)

[00372] Thetitecompound was prepared inananalogousfashionto that described in Stage22.1using5-bromo-6-chloro-N-(4-(chlorodifluoromethoxy)phenyl)nicotinamide(Stage

![](images/0ea4ed239a2a04065fa8937935d71aa619ee5457b3416f0cf429a0658c543d90.jpg)  
Input image   
<image>\nParse the figure.

![](images/acb4ebc7fbe678f6a7bfe0700c5e9714865befcbff6751786abf1c1449651b4a.jpg)

![](images/63220bb98d1a5435ce0fb6d61d91d34465f7aaa6dea28d9b115d67bf307a8ee6.jpg)

![](images/65d50a9f46e79fba1ba4ff68b2979051c0544b71c1f4813a38037afb2d5a050a.jpg)  
Result

toafford a white crystalline solid.HPLC(Condition4)tg=5.72min，UPLC-MS(Condition3)

# Example24

![](images/8daeedbc64d965095baa671e69e230c0e0fddf569f051deae84a479a1a77792b.jpg)

[00370]The title compound was preparedinananalogous fashion to that described in Example 26 using 5- bromo-N-(4-（orodifluorometho)pl)-6-（ethy(2-roxetamino）nicotinamide(Stage24.1）and

J=1.96Hz，1H）7.33（d，J=9.05Hz,2H）7.76（br.s1H）7.82-7.95（m,2H)8.13（d，J=2.45Hz,1H）8.72

![](images/f6a82e6015db5450f95ee17d65a19e69113e567073c0c08a0ad576603128f757.jpg)  
Deep Parsing   
Rendering   
Figure 9 | DeepSeek-OCR in deep parsing mode can also recognize chemical formulas within chemical documents and convert them to SMILES format. In the future, OCR $1 . 0 { + } 2 . 0 $ technology may play a significant role in the development of VLM/LLM in STEM fields.

brom6N4(otlidetge

![](images/cf35fee81ac7e59fc006fd9f765c869c52518845889fd9395688746beef51740.jpg)  
Input image

![](images/b9c451c44ac654a197f724ce3d88a4f564eb0976133a0bf8e2fecbe48099d8fc.jpg)  
Result

![](images/b09c9e0fe7c1fe6a611c308e848c04f9426665b3599a48b36bfee63852dad63c.jpg)  
Figure 10 | DeepSeek-OCR also possesses the capability to copy (structure) simple planar geometric figures. Due to the intricate interdependencies among line segments in geometric shapes, parsing geometry task is extremely challenging and has a long way to go.

# 4.3.2. Multilingual recognition

PDF data on the Internet contains not only Chinese and English, but also a large amount of multilingual data, which is also crucial when training LLMs. For PDF documents, DeepSeek-OCR can handle nearly 100 languages. Like Chinese and English documents, multilingual data also supports both layout and non-layout OCR formats. The visualization results are shown in Figure 11, where we select Arabic and Sinhala languages to demonstrate results.

# <image>\nFree OCR.

# sLiys

-22 14/04tyy c（ANGEM · .(2004) ul24NA .5 √ · √ √

# ：yu-I

# L 1

# llu01l

![](images/ea048e2ffbcbfcc67cac0d0263458e209a80f4f9ff8b79c3be69cec0d3005d09.jpg)

#

# 2

![](images/784bb9de28b336c85b2a1f608c0a4de92d0756d4c9ce534214bdd0366a227e46.jpg)

#

(1994)iululiogo s aliaiallg aoloukil ≥loyl lc|lgl lc

ilcjliogU atbg aJh

#

#

<table><tr><td>1234567890</td><td>2221</td><td>%</td></tr><tr><td>4234567890</td><td>83</td><td>100</td></tr><tr><td>4234567890</td><td>0</td><td>0</td></tr><tr><td>1234567890</td><td>83</td><td>100</td></tr></table>

#

Cll gg Jg>all uJ usyhilaas le sg aiyllauaud Olglll Ai gegaKagipgaylyig

# <image>\n<|grounding|>Convert the document to markdown.

@aem8 e2031. .

8m20268305822.82.86s@s（EU） 2021/2307（8）@6e@8日8.（EU）2022/2238306δ20222258818

（8）（EU）2021/2307

（9）282820223028e0.8202281088

（10）@ 6s 6f2园@s

@（EU） 2021/2119 2s 6s②2s）eo@2

g（EU） 2021/2119 8@8 @2

8gfTRACEsed

5 520238188

）E278

88202263022.8. 6s5@55（U2021/2307（8）. .（u）2022/223830620222 5888885188 656

8）u202308

9）2880e026x585@8820228302% @50565.8892202215 10e6g8bmmg@

6F2021/211985@@9

E20212119s@8@

<lre‡|>text<|/sef|><|det|>[[137,84,B06,12011<l/det ]> <>e<l/ef||t|>[37,84,806120<l/ds> zs mg gg 5 ss ogdad (p) (s) d

(5)e|>ext</e|>,[[132541</et>q00e0e da gdbo a6e -euoee aoda snd see -eapmenaa ca oeedDe-dse s6and0sd. s6amd0ssod mnetnə mo qsoaad:0d oeommodiaa 2023 g8 1

<lre‡|>text<|/sef|><|det|>[[117,275,840,29311<|/det|>

(ee> s0d:-a25 g6mn8 (Em 2022/2238 (7) a00 sd ssssd mao ro0eal eo a@o n0 8e6d000)202/228(7)0e2  8a anaab 3 cogbe sdn ct. 8d dm. am&sd mnd s 6alsn soaa coqg8 D221/20782/2  0 5 2022 te2 cl n5 2022 et: n e=ca 0 0) 2022/2 B o od go 1ee c scaed  dage anoo ogddn0 od angb 18 d em2_cs sexl8aadim @abz dateazd dnoo

(8)e0)et 0/e8 (0) 2021/2307zd6x658 etD 880s/0g 06 g0g

8ep z65e 8sstpabz, eee em20a 2022 g8 1 0zn 8o 8o:m20 see 8a god

(10)e@8s ogg8scd cx8a5 zd 5os@o oDge lelses mags-D 8od0 qgxaob

<|red[>text<|/re

e008(Fm)2021/2119 gd5sx8n ,a,o0g/0

<|ret|>text<|/ref|><|det|>[[117,808,752,826]]<|/det|>

(lref[>text<]/zef]><]det]>[117,837,539,B53

Figure 11 | To endow the capability of processing widely crawled PDFs (multilingual data), we train our model with OCR capabilities for nearly 100 languages. Minority language documents can also support both layout and non-layout outputs through different prompts.

# 4.3.3. General vision understanding

We also provide DeepSeek-OCR with a certain degree of general image understanding capabilities. The related visualization results are shown in Figure 12.

![](images/93d24af0c72f9ac2cfbc2de5140f8c976dd29709d4fa98f696a94577e89d9d36.jpg)  
<image>\nLocate <|ref|>11-2=<|/ref|> in the image.

<image>\nDescribe this image in detail.   
![](images/31a3d3966db97400f4079eb3513246791abdfe52d2be9b4ec2c4af207b3c7bc3.jpg)  
A green plastic container labeled for food storage purposes. The lid is yellow and appears slightly worn out.

On the front label:

- At the top left corner, some Chinese characters.

- In bold black letters near the center-top reads "6月香" (June Fragrance).   
- Below that, smaller white font states "鲜甜咸香" which translates roughly to "Fresh Sweet Salty Fragrant."   
- Dominating the middle part of the label is large lettering stating "豆瓣酱," meaning "Bean Paste Sauce."

Additional Information includes:

- A small circular logo indicating quality assurance ("优质品").

- An illustration showing what seems like a dish made using bean paste sauce.

- Text below the main title says "专注25年，只做原酿酱" translating to "Focusing since 1998, only making authentic fermented bean paste."

- On the bottom right side, additional product specifications include "净含量：500g"

Overall Style/Type:

It looks like commercial packaging designed specifically for preserving food items

within the jar. It has clear labeling intended for easy identification and understanding among consumers who read Mandarin.

Note: There might be slight variations due to handwriting styles but ove maintains readability regarding brand identity and contents description.

![](images/369faf7bdde61f992beab5d88bb10f15478e46005faf2777b89603f7cda0081e.jpg)  
<image>\nLocate <|ref|>the teacher<|/ref|> in the image.   
MAZKANDEZSON   
WWW.ANDERTOONS.COM   
"I like how Bonnie's not stepping on my last nerve."

![](images/4e701a3d8dc73b260ab612174e1c158f6cfa6dc2343c0b624b97632d28b089e7.jpg)  
<image>\nIdentify all objects in the image and output them in bounding boxes.

# <image>\n这是一张

![](images/5fdf8fb62781a99bb4221ce195c6d6a471bf029a7ce2cb20ec6cef259c095f99.jpg)

照片，展示了一辆红色的消防栓。消防栓上有一个笑脸，显得非常友好和亲切。消防栓的顶部有一个黑色的盖子，周围有一些金属铆钉。在消防栓的底部，有一个粉红色的贴纸，上面写着“bowtie”。背景中可以看到一条街道，街道上有几辆停放的汽车和一些树木。整体画面给人一种温馨和友好的感觉。

<image>\n<|grounding|>OCR the image.

![](images/413b06fe0f61876cb8471b00f40456c726d53026d622b37a9465efb9cf220e53.jpg)  
Figure 12 | We retain DeepSeek-OCR’s capabilities in general visual understanding, mainly including image description, object detection, grounding, etc. Meanwhile, due to the inclusion of text-only data, DeepSeek-OCR’s language capabilities are also retained. Note that since we do not include SFT (Supervised Fine-Tuning) stage, the model is not a chatbot, and some capabilities need completion prompts to be activated.

# 君不见，黄河之水天上来

，奔流到海不复回。君不见，高堂明镜悲白发，朝如青丝暮成雪。人生得意须尽欢，莫使金樽空对月。天生我材必有用，千金散尽还复来。烹羊宰牛且为乐，会须一饮三百杯。岑夫子，丹丘生，将进酒，杯莫停。与君歌一曲，请君为我倾耳听。钟鼓馔玉不足贵，但愿长醉不愿醒。古来圣贤皆寂寞，惟有饮者留其名。陈王昔时宴平乐，斗酒十千恣欢谑。主人何为言少钱，径须沽取对君酌。五花马，千金裘，呼儿将出换美酒，与尔同销万古愁。

# 5. Discussion

Our work represents an initial exploration into the boundaries of vision-text compression, investigating how many vision tokens are required to decode $N$ text tokens. The preliminary results are encouraging: DeepSeek-OCR achieves near-lossless OCR compression at approximately $1 0 \times$ ratios, while $2 0 \times$ compression still retains $6 0 \%$ accuracy. These findings suggest promising directions for future applications, such as implementing optical processing for dialogue histories beyond $k$ rounds in multi-turn conversations to achieve $1 0 \times$ compression efficiency.

![](images/6bde3557f71636dea77635bbab03e866b18a4fdf6bcd9dc3f3b2ec56de7bcf99.jpg)  
Figure 13 | Forgetting mechanisms constitute one of the most fundamental characteristics of human memory. The contexts optical compression approach can simulate this mechanism by rendering previous rounds of historical text onto images for initial compression, then progressively resizing older images to achieve multi-level compression, where token counts gradually decrease and text becomes increasingly blurred, thereby accomplishing textual forgetting.

For older contexts, we could progressively downsizing the rendered images to further reduce token consumption. This assumption draws inspiration from the natural parallel between human memory decay over time and visual perception degradation over spatial distance—both exhibit similar patterns of progressive information loss, as shown in Figure 13. By combining these mechanisms, contexts optical compression method enables a form of memory decay that mirrors biological forgetting curves, where recent information maintains high fidelity while distant memories naturally fade through increased compression ratios.

While our initial exploration shows potential for scalable ultra-long context processing, where recent contexts preserve high resolution and older contexts consume fewer resources, we acknowledge this is early-stage work that requires further investigation. The approach suggests a path toward theoretically unlimited context architectures that balance information retention with computational constraints, though the practical implications and limitations of such vision-text compression systems warrant deeper study in future research.

# 6. Conclusion

In this technical report, we propose DeepSeek-OCR and preliminarily validate the feasibility of contexts optical compression through this model, demonstrating that the model can effectively decode text tokens exceeding 10 times the quantity from a small number of vision tokens. We believe this finding will facilitate the development of VLMs and LLMs in the future. Additionally, DeepSeek-OCR is a highly practical model capable of large-scale pretraining data production, serving as an indispensable assistant for LLMs. Of course, OCR alone is insufficient to fully validate true context optical compression and we will conduct digital-optical text interleaved pretraining, needle-in-a-haystack testing, and other evaluations in the future. From another perspective, optical contexts compression still offers substantial room for research and improvement, representing a promising new direction.

# References

[1] Marker. URL https://github.com/datalab-to/marker.   
[2] Mathpix. URL https://mathpix.com/.   
[3] Ocrflux, 2025. URL https://github.com/chatdoc-com/OCRFlux.   
[4] G. AI. Gemini 2.5-pro, 2025. URL https://gemini.google.com/.   
[5] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.   
[6] L. Blecher, G. Cucurull, T. Scialom, and R. Stojnic. Nougat: Neural optical understanding for academic documents. arXiv preprint arXiv:2308.13418, 2023.   
[7] J. Chen, L. Kong, H. Wei, C. Liu, Z. Ge, L. Zhao, J. Sun, C. Han, and X. Zhang. Onechart: Purify the chart structural extraction via one auxiliary token. In Proceedings of the 32nd ACM International Conference on Multimedia, pages 147–155, 2024.   
[8] Z. Chen, W. Wang, H. Tian, S. Ye, Z. Gao, E. Cui, W. Tong, K. Hu, J. Luo, Z. Ma, et al. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821, 2024.   
[9] C. Cui, T. Sun, M. Lin, T. Gao, Y. Zhang, J. Liu, X. Wang, Z. Zhang, C. Zhou, H. Liu, et al. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595, 2025.   
[10] M. Dehghani, J. Djolonga, B. Mustafa, P. Padlewski, J. Heek, J. Gilmer, A. Steiner, M. Caron, R. Geirhos, I. Alabdulmohsin, et al. Patch n’ pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:3632–3656, 2023.   
[11] H. Feng, S. Wei, X. Fei, W. Shi, Y. Han, L. Liao, J. Lu, B. Wu, Q. Liu, C. Lin, et al. Dolphin: Document image parsing via heterogeneous anchor prompting. arXiv preprint arXiv:2505.14059, 2025.   
[12] Y. Goyal, T. Khot, D. Summers-Stay, D. Batra, and D. Parikh. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913, 2017.   
[13] J. Gu, X. Meng, G. Lu, L. Hou, N. Minzhe, X. Liang, L. Yao, R. Huang, W. Zhang, X. Jiang, et al. Wukong: A 100 million large-scale chinese cross-modal pre-training benchmark. Advances in Neural Information Processing Systems, 35:26418–26431, 2022.   
[14] High-flyer. HAI-LLM: Efficient and lightweight training tool for large models, 2023. URL https://www.high-flyer.cn/en/blog/hai-llm.   
[15] S. Iyer, X. V. Lin, R. Pasunuru, T. Mihaylov, D. Simig, P. Yu, K. Shuster, T. Wang, Q. Liu, P. S. Koura, et al. Opt-iml: Scaling language model instruction meta learning through the lens of generalization. arXiv preprint arXiv:2212.12017, 2022.   
[16] S. Kazemzadeh, V. Ordonez, M. Matten, and T. Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787–798, 2014.

[17] A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao, S. Whitehead, A. C. Berg, W.-Y. Lo, et al. Segment anything. arXiv preprint arXiv:2304.02643, 2023.   
[18] Z. Li, Y. Liu, Q. Liu, Z. Ma, Z. Zhang, S. Zhang, Z. Guo, J. Zhang, X. Wang, and X. Bai. Monkeyocr: Document parsing with a structure-recognition-relation triplet paradigm. arXiv preprint arXiv:2506.05218, 2025.   
[19] A. Liu, B. Feng, B. Wang, B. Wang, B. Liu, C. Zhao, C. Dengr, C. Ruan, D. Dai, D. Guo, et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.   
[20] A. Liu, B. Feng, B. Xue, B. Wang, B. Wu, C. Lu, C. Zhao, C. Deng, C. Zhang, C. Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.   
[21] C. Liu, H. Wei, J. Chen, L. Kong, Z. Ge, Z. Zhu, L. Zhao, J. Sun, C. Han, and X. Zhang. Focus anywhere for fine-grained multi-page document understanding. arXiv preprint arXiv:2405.14295, 2024.   
[22] I. Loshchilov and F. Hutter. Sgdr: Stochastic gradient descent with warm restarts. arXiv preprint arXiv:1608.03983, 2016.   
[23] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In ICLR, 2019.   
[24] A. Masry, D. X. Long, J. Q. Tan, S. Joty, and E. Hoque. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022.   
[25] A. Nassar, A. Marafioti, M. Omenetti, M. Lysak, N. Livathinos, C. Auer, L. Morin, R. T. de Lima, Y. Kim, A. S. Gurbuz, et al. Smoldocling: An ultra-compact vision-language model for end-to-end multi-modal document conversion. arXiv preprint arXiv:2503.11576, 2025.   
[26] OpenAI. Gpt-4 technical report, 2023.   
[27] L. Ouyang, Y. Qu, H. Zhou, J. Zhu, R. Zhang, Q. Lin, B. Wang, Z. Zhao, M. Jiang, X. Zhao, et al. Omnidocbench: Benchmarking diverse pdf document parsing with comprehensive annotations. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 24838–24848, 2025.   
[28] J. Poznanski, A. Rangapur, J. Borchardt, J. Dunkelberger, R. Huff, D. Lin, C. Wilhelm, K. Lo, and L. Soldaini. olmocr: Unlocking trillions of tokens in pdfs with vision language models. arXiv preprint arXiv:2502.18443, 2025.   
[29] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.   
[30] Rednote. dots.ocr, 2025. URL https://github.com/rednote-hilab/dots.ocr.   
[31] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million imagetext pairs. arXiv preprint arXiv:2111.02114, 2021.

[32] A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019.   
[33] T. Sun, C. Cui, Y. Du, and Y. Liu. Pp-doclayout: A unified document layout detection model to accelerate large-scale data construction. arXiv preprint arXiv:2503.17213, 2025.   
[34] B. Wang, C. Xu, X. Zhao, L. Ouyang, F. Wu, Z. Zhao, R. Xu, K. Liu, Y. Qu, F. Shang, et al. Mineru: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024.   
[35] P. Wang, S. Bai, S. Tan, S. Wang, Z. Fan, J. Bai, K. Chen, X. Liu, J. Wang, W. Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.   
[36] H. Wei, L. Kong, J. Chen, L. Zhao, Z. Ge, J. Yang, J. Sun, C. Han, and X. Zhang. Vary: Scaling up the vision vocabulary for large vision-language model. In European Conference on Computer Vision, pages 408–424. Springer, 2024.   
[37] H. Wei, L. Kong, J. Chen, L. Zhao, Z. Ge, E. Yu, J. Sun, C. Han, and X. Zhang. Small language model meets with reinforced vision vocabulary. arXiv preprint arXiv:2401.12503, 2024.   
[38] H. Wei, C. Liu, J. Chen, J. Wang, L. Kong, Y. Xu, Z. Ge, L. Zhao, J. Sun, Y. Peng, et al. General ocr theory: Towards ocr-2.0 via a unified end-to-end model. arXiv preprint arXiv:2409.01704, 2024.   
[39] H. Wei, Y. Yin, Y. Li, J. Wang, L. Zhao, J. Sun, Z. Ge, X. Zhang, and D. Jiang. Slow perception: Let’s perceive geometric figures step-by-step. arXiv preprint arXiv:2412.20631, 2024.   
[40] Z. Wu, X. Chen, Z. Pan, X. Liu, W. Liu, D. Dai, H. Gao, Y. Ma, C. Wu, B. Wang, et al. Deepseek-vl2: Mixture-of-experts vision-language models for advanced multimodal understanding. arXiv preprint arXiv:2412.10302, 2024.   
[41] W. Yu, Z. Yang, L. Li, J. Wang, K. Lin, Z. Liu, X. Wang, and L. Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.   
[42] J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, H. Tian, Y. Duan, W. Su, J. Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.
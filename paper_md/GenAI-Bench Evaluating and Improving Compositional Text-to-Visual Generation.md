# GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation

Baiqi Li1∗ Zhiqiu Lin1,2∗ Deepak Pathak1 Jiayao Li1 Yixin Fei1 Kewen Wu1 Tiffany Ling1 Xide Xia2† Pengchuan Zhang2† Graham Neubig1† Deva Ramanan1† 1Carnegie Mellon University 2Meta

# Abstract

While text-to-visual models now produce photo-realistic images and videos, they struggle with compositional text prompts involving attributes, relationships, and higher-order reasoning such as logic and comparison. In this work, we conduct an extensive human study on GenAI-Bench to evaluate the performance of leading image and video generation models in various aspects of compositional text-to-visual generation. We also compare automated evaluation metrics against our collected human ratings and find that VQAScore – a metric measuring the likelihood that a VQA model views an image as accurately depicting the prompt – significantly outperforms previous metrics such as CLIPScore. In addition, VQAScore can improve generation in a black-box manner (without finetuning) via simply ranking a few (3 to 9) candidate images. Ranking by VQAScore is $2 \mathbf { x }$ to 3x more effective than other scoring methods like PickScore, HPSv2, and ImageReward at improving human alignment ratings for DALL-E 3 and Stable Diffusion, especially on compositional prompts that require advanced visio-linguistic reasoning. We release a new GenAI-Rank benchmark with over 40,000 human ratings to evaluate scoring metrics on ranking images generated from the same prompt. Lastly, we discuss promising areas for improvement in VQAScore, such as addressing fine-grained visual details. We will release all human ratings (over 80,000) to facilitate scientific benchmarking of both generative models and automated metrics.

# 1 Introduction

State-of-the-art text-to-visual models like Stable Diffusion [56], DALL-E 3 [1], Gen2 [16], and Sora [63] generate images and videos with exceptional realism and quality. Due to their rapid advancement, traditional evaluation metrics and benchmarks (e.g., FID scores on COCO [21, 36] and CLIPScores on PartiPrompt [20, 80]) are becoming insufficient [39, 50]. For instance, benchmarks should include more compositional text prompts [41] that involve attribute bindings, object relationships, and logical reasoning, among other visio-linguistic reasoning skills (Figure 1). Moreover, it’s crucial for automated evaluation metrics to measure how well the generated images (or videos) align with such compositional text prompts. Yet, widely used metrics like CLIPScore [20] function as bag-of-words [38, 69, 81] and cannot produce reliable alignment (faithfulness [23]) scores. Therefore, to guide the scientific benchmarking of generative models, we conduct a comprehensive evaluation of compositional text-to-visual generation alongside automated alignment metrics [5, 20].

Evaluating text-to-visual generation. We evaluate generative models using our collected GenAI-Bench benchmark [39], which consists of 1,600 challenging real-world text prompts sourced from professional designers. Compared to benchmarks [23, 27, 43] such as PartiPrompt [80] and T2I-CompBench [24] (see Table 1), GenAI-Bench captures a wider range of aspects in compositional

![](images/4b27e9d75e380ec4e59e58ec2798cdebd0fce22466fc3dee8f42cdc10a968639.jpg)  
A row of colorful townhouses,

![](images/11be8438dd8ab0a2a33ce8b94c23787a85a7d7a95ee9f2be2e27a063bd0be7a8.jpg)  
townhouses on a sunny street.

![](images/5bde384d5164e272b8a5a7f6d1f86d8f1b5299dc1ba201444952f1590d7db758.jpg)  
A fork lies to the right of a spoon on a wooden table. Let's give them colors. (Attribute)

![](images/3f7baaccb3f231c3ce2ed64dfb36bf25c3a67551e63455c75c86edc117f816a2.jpg)  
right of a goldenspoon

![](images/11e90e1ea1bc8787b54a84980db9f495634a7a7855ae63155f91e031a0f7ff1e.jpg)  
A kitten and a butterfly.

![](images/424445839060fa9213626ce50ed91982b97763453d8c44e8dde936def7e73f52.jpg)  
bell collar on the left, batting at a butterfly on theright.

![](images/64ffdab0c95337c5bc054502c1890728e23712f2e619ef74bec883270619bda8.jpg)  
White seagulls flying over a blue lake. I want only three seagulls! (Counting)

![](images/18f77e9d16d6967423a0bc5289ddc42db05a0653e3ac0935e0472b17224762ae.jpg)

A birthday party with presents A birthday party with presents bionthetableuwitpened. on the table unopened. ® Let's keep all the Let's keep all the boxes closed. boxesclosed. (Univeclality) (Universality)

![](images/ac30d4965f84a3d25281134539f5219f5df4c7817f86c75ce24e21f18b235f95.jpg)

![](images/1a87f4f7e3d407ddbd79c99ad71daa02998e4c5e68a712fcc9d9ec0f5d85c1b3.jpg)

![](images/a25ea531bb03ac1224ef2a24d541760ccb276d95c1f23f0ec6a05216b4727bb0.jpg)  
A little ant carrying a backpack. ® Moteitsbckpack → (Comparison)

![](images/ff99a5583d50eea12735cb5dbedee8cc7e7b0de74a68020fb85edf817207a65d.jpg)  
backpackbiggerthan itself.

![](images/29b907a31e0854c3150ac5e2a5a24c6fdb1911ed9768fe3c6c9cbb83a0082115.jpg)  
Threeflowers on the ground. ® Coloreach flower (oifrengtatinon Color ech twe

![](images/6ebab64b158b523285287a7a6fbcc7a29a0d3a43760de639de50834f1f3d8de6.jpg)  
ground;onered, another yellow,and the thirdblue.

![](images/8c72cdcc3965d00ed6b8220d0896798e84363bf324ec3eb69f3bfb579dccb715.jpg)  
A bookshelf with picture frames. Idon'twant books (Negation)

![](images/2faf77167aed78764e7f34b9686d8b089b6e23c2a853649bfbdf68994de459a2.jpg)  
Figure 1: Compositional text prompts of our GenAI-Bench (highlighted in green) reflect how real-world users (e.g., designers) seek precise control in text-to-visual generation. For example, users often write detailed prompts that specify compositions of basic visual entities and properties (highlighted in gray), such as scenes, attributes, and relationships (spatial/action/part). Moreover, user prompts may require advanced visio-linguistic reasoning (highlighted in blue), such as counting, comparison, differentiation, and logic (negation/universality). Appendix B provides skill definitions with more examples. Table 1 compares GenAI-Bench with previous benchmarks [24, 27, 58, 80].

# Step 1:

Source prompts from designers who use generation tools in their profession.

![](images/a16c42f80cdcc4f6460b9cce92ae9aca90c6f9eb1f38c0e8bf5ba17fc980c3b4.jpg)

# Step 2:

Tag prompts with relevant visio-linguistic compositional reasoning skills.

One cat sleeping under five bright stars.

Spatial

A small cat waves a wand.

The girl with glasses is drawing,and the girl withoutglassesissinging.

# Step 3:

Collect alignment ratings in 1-to-5 Likert scale from human annotators.

One cat sleeping nderfivebrightstars.

![](images/099c9e8fc6336ca7fcb6c80616c2dbe4af428a642c129a405768b28159c87122.jpg)

How well does the image match thedescription?

![](images/023cb4f0eb33e3ecb503abc626fdf0d2a61435dd6382a56c3daa727e1b6fabf1.jpg)

![](images/074912a72c490aebcfdc3ad910651e27c24898b6dba1c52b61956f7ac69fdaef.jpg)

![](images/f799e2e388b19af05534d0fcc1af9dc9d79cb2396023313507a014c0840965fa.jpg)  
Annotators   
Figure 2: Collecting GenAI-Bench. To reliably evaluate generative models, we source prompts from professional designers who use tools such as Midjourney [47] and CIVITAI. This ensures the prompts encompass practical skills relevant to real-world applications and are free of subjective or inappropriate content. Each GenAI-Bench prompt is carefully tagged with all its evaluated skills. We then generate images and videos using state-of-the-art models like SD-XL [56] and Gen2 [16]. We follow the recommended annotation protocol [50] to collect 1-to-5 Likert scale ratings for how well the generated visuals align with the input text prompts.

text-to-visual generation [32], ranging from basic (scene, attribute, relation) to advanced (counting, comparison, differentiation, logic). We collect a total of 38,400 human alignment ratings (1-to-5 Likert scales [50]) on images and videos generated by ten leading models2, such as Stable Diffusion [56], DALL-E 3 [1], Midjourney v6 [47], Pika v1 [52], and Gen2 [16]. Figure 2 illustrates the evaluation process. Our human study shows that while these models can often accurately generate basic compositions (e.g., attributes and relations), they still struggle with advanced reasoning (e.g., logic and comparison). For instance, for “basic” prompts that do not require advanced reasoning, the state-of-the-art DALL-E 3 (most preferred by humans) achieves a remarkable average rating of 4.3, meaning its images range from having “a few minor discrepancies” to ‘matching exactly” with the prompts. However, its rating on “advanced” prompts drops to 3.4, indicating “several discrepancies”.

Evaluating automated metrics. We also use the human ratings to benchmark automated metrics (e.g., CLIPScore [20], PickScore [27], and Davidsonian [5]) that measure the alignment between an image and a text prompt. Specifically, we show that a simple metric, VQAScore [39], which computes the likelihood of generating a “Yes” answer to a question like “Does this figure show {text}?” from a VQA model, significantly surpasses previous metrics in correlating with human judgments.

VQAScore can be calculated end-to-end from off-the-shelf VQA models, without finetuning on human feedback [27, 78] or decomposing prompts into QA pairs [5, 79]. VQAScore is strong because it leverages the compositional reasoning capabilities of recent multimodal large language models (LLMs) [8, 40] trained for VQA. For instance, our study adopts the open-source CLIP-FlanT5 model [39], which uses a bidirectional encoder that allows the image and question embeddings to “see” each other. VQAScore based on CLIP-FlanT5 sets a new state-of-the-art on both GenAI-Bench and previous benchmarks like TIFA160 [23] and Winoground [69]. As such, we recommend VQAScore over the “bag-of-words” CLIPScore, which has been widely misused in our community [25, 81]. We will release all human ratings to facilitate the development of automated metrics.

Improving generation with VQAScore. We show that text-to-image generation can be improved by selecting the candidate image with the highest VQAScore (from a set of candidates). This rankingbased approach does not require any finetuning and can operate in a fully black-box manner [42], needing only an image generation API. Notably, simply ranking between 3 to 9 images can already enhance the average human ratings for DALL-E 3 and SD-XL by 0.2 to 0.3 (on a 1-to-5 Likert scale), setting the new closed-source and open-source SOTAs on GenAI-Bench. VQAScore significantly outperforms other metrics; for instance, using CLIPScore for ranking often leads to the same or lower human ratings. We present qualitative examples in Figure 7. Overall, VQAScore emerges as the most effective ranking metric, surpassing other metrics that rely on costly human feedback (e.g., PickScore [27]) or ChatGPT for prompt decomposition (e.g., Davidsonian [5]) by 2x to 3x.

Limitations. Lastly, we explore the implications of Goodhart’s Law [17], particularly limitations of VQAScore in detecting fine-grained visual details and resolving linguistic ambiguity. Despite these mild limitations, we strongly urge the research community to adopt VQAScore as a reproducible supplement to non-reproducible human studies [50], or as a more reliable alternative to CLIPScore, which has ceased to be effective [25, 38, 81].

Table 1: Comparing GenAI-Bench to existing text-to-visual benchmarks. GenAI-Bench covers more essential skills of compositional text-to-visual generation, emphasizing advanced reasoning skills (highlight in blue) that are required to parse complex user prompts. Moreover, GenAI-Bench tags each prompt with all its evaluated skills, whereas most benchmarks assign no tags or only one or two per prompt (even when multiple skills are involved). GenAI-Bench also provides human ratings for both image and video generative models to benchmark automated metrics.   

<table><tr><td rowspan="2">Benchmarks</td><td colspan="8">Skills Covered in Compositional Text-to-Visual Generation</td><td rowspan="2">Tagging</td><td rowspan="2">Human Annotation</td></tr><tr><td>Scene</td><td>Attribute</td><td>Relation</td><td>Count</td><td>Negation</td><td>Universal</td><td>Compare</td><td>Differ</td></tr><tr><td>PartiPrompt (P2) [80]</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>2 Tags</td><td>X</td></tr><tr><td>DrawBench [58]</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>1 Tag</td><td>X</td></tr><tr><td>EditBench [72]</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>TIFAv1 [23]</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>All Tags</td><td>Images</td></tr><tr><td>Pick-a-pic [27]</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>Images</td></tr><tr><td>T2I-CompBench [24]</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>1 Tag</td><td>Images</td></tr><tr><td>HPDv2 [76]</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>Images</td></tr><tr><td>EvalCrafter [43]</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>Videos</td></tr><tr><td>GenAI-Bench (Ours)</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>All Tags</td><td>Images &amp; Videos</td></tr></table>

# Contribution summary.

1. We conduct an extensive human study on compositional text-to-visual generation using GenAI-Bench, revealing limitations of leading open-source and closed-source models.   
2. We present a simple black-box approach that improves generation by ranking images with VQAScore, significantly surpassing other scoring methods by 2x to 3x.   
3. We will release GenAI-Rank with over 40,000 human ratings to benchmark methods that rank images generated from the same prompt.

# 2 Related Works

Text-to-visual benchmarks. Early benchmarks mostly rely on captions from existing datasets like COCO [6, 23, 36, 55], focusing on generating simple objects, attributes, and scenes. Other benchmarks, such as HPDv2 [76] and Pick-a-pic [27], primarily evaluate image quality (aesthetic) using simpler text prompts. Recently, benchmarks like DrawBench [58], PartiPrompt [80], and

T2I-CompBench [24] have shifted the focus to compositional text-to-image generation with an emphasis on attribute bindings and object relationships. Our GenAI-Bench escalates the challenge by incorporating more practical prompts that require “advanced” reasoning (e.g., logic and comparison) to benchmark next-generation text-to-visual models.

Automated metrics. Perceptual metrics like IS [59], FID [21] and LPIPS [82] use pre-trained networks to assess the quality of generated imagery using reference images. To evaluate visionlanguage alignment (also referred to as faithfulness or consistency [9, 23, 45]), recent studies [4, 14, 15, 26, 29, 46, 57, 61, 73] primarily report CLIPScore [20], which measures (cosine) similarity of the embedded image and text prompt. However, CLIP cannot reliably process compositional text prompts due to its “bag-of-words” encoding [25, 38, 81]. Human preference models like ImageReward [78], PickScore [27], and HPSv2 [76] further leverage human feedback to improve models like CLIP by finetuning on large-scale human ratings. Another popular line of works [7, 23, 24, 62, 74] uses LLMs like ChatGPT to decompose texts into simpler components for analysis, e.g., via question generation and answering (QG/A) [5]. For example, Davidsonian Scene Graph [5] decomposes a text prompt into simpler QA pairs and outputs a score as the accuracy of answers generated by a VQA model. However, Lin et al. [38, 39] show that such methods still struggle with complex text prompts and propose VQAScore, an end-to-end metric that better correlates with human judgments.

# 3 GenAI-Bench for Text-to-Visual Evaluation

In this section, we review GenAI-Bench [39], a challenging benchmark featuring real-world user prompts that cover essential aspects of compositional text-to-visual generation.

Skill taxonomy. Prior literature on text-to-visual generation [24, 58, 80] focuses on generating “basic” objects, attributes, relations, and scenes. However, as shown in Figure 1, user prompts often need “advanced” compositional reasoning, including comparison, differentiation, counting, and logic. These “advanced” compositions extend beyond the “basic” ones. For example, user prompts may involve counting not just objects, but also complex object-attribute-relation compositions, e.g., “three white seagulls flying over a blue lake”. Also, users often want to generate multiple objects of the same category but differentiate them with varied properties, e.g., “three flowers on the ground; one red, another yellow, and the third blue”. To this end, we collaborate with designers who use text-to-image tools [47] in their profession to design a taxonomy that includes both “basic” (objects, scenes, attributes, and spatial/action/part relations) and “advanced” skills (counting, comparison, differentiation, negation, and universality). Table 1 shows that GenAI-Bench uniquely covers all these essential skills. Appendix B provides definitions and more examples.

GenAI-Bench. We also collect 1,600 prompts from these professional designers. Importantly, collecting prompts from actual designers ensures that they are free from subjective (non-visual) or toxic contents. For example, we observe that ChatGPT-generated prompts from T2I-CompBench [24] can include subjective phrases like “a natural symbol of rebirth and renewal”. We also find that the Pick-a-pic [27] contains inappropriate prompts (e.g., those suggesting NSFW images of celebrities) created by malicious web users. To avoid legal issues and test general model capabilities, we instruct the designers to create prompts with generic characters and subjects (e.g., food, vehicles, humans, pets, plants, household items, and mythical creatures) instead of celebrities or copyrighted characters. Appendix C details our collection procedure and discuss how we avoid these issues. For a fine-grained analysis, we tag each prompt with all its evaluated skills. This contrasts with previous benchmarks that provide no tags [27, 43, 76] or only assign only one or two tags per prompt [24, 58, 80]. In total, GenAI-Bench provides over 5,000 human-verified tags with an approximately balanced distribution of skills: about half of the prompts involve only “basic” compositions, while the other half poses greater challenges by incorporating both “basic” and ”advanced” compositions.

# 4 Human Evaluation via GenAI-Bench

We now present an extended human study of ten popular image and video generative models.

Human evaluation. We evaluate six text-to-image models: Stable Diffusion [56] (SD v2.1, SD-XL, SD-XL Turbo), DeepFloyd-IF [11], Midjourney v6 [47], DALL-E 3 [1]; along with four text-to-video models: ModelScope [71], Floor33 [13], Pika v1 [52], Gen2 [16]. Next, we hire three annotators to

rate on a 1-to-5 Likert scale for image-text or video-text alignment using the recommended annotation protocol of [50]:

How well does the image (or video) match the description?

1. Does not match at all.   
2. Has significant discrepancies.   
3. Has several minor discrepancies.   
4. Has a few minor discrepancies.   
5. Matches exactly.

![](images/d063f3b263f7c7e40dfad7239537f7eb6118589607e4101c35b5b7a1af7468be.jpg)  
(a) GenAI-Bench (Image Ratings)

![](images/0feaf6f62aca8763cbf50c28fc80fdeb0b51f8493e4a2c1b8774aee8884abe44.jpg)  
(b) GenAI-Bench (Video Ratings)   
Figure 3: Human evaluation on GenAI-Bench. We show the average human alignment ratings on ten popular image and video generative models. We highlight closed-source models (e.g., DALL-E 3 [1]) in green. We find that (1) “advanced” prompts that require higher-order reasoning (e.g., negation and comparison) challenge all models more, (2) models using better text embeddings or captions (DeepFloyd-IF [11] and DALL-E 3 [1]) outperform others (SD-XL [56]), (3) open-source and video-generative models [16, 56] still lag behind their closed-sourced and image-generative counterparts, suggesting room for improvement.

![](images/f6f1e7b2c2de917a6320b0f8c36a28550904672a5149a786a80104ec7f1d8627.jpg)  
Figure 4: VQAScore (based on CLIP-FlanT5 [39]) versus CLIPScore on samples from GenAI-Bench. VQAScore shows a significantly stronger agreement with human ratings compared to CLIPScore [20], making it a more reliable tool for automatic text-to-visual evaluation, especially on user prompts that involve complex compositional reasoning.

Our collected human ratings indicate a high level of inter-rater agreement, with Krippendorff’s Alpha reaching 0.72 for image ratings and 0.70 for video ratings, suggesting substantial agreement [23]. The use of the Likert scale also makes the final rating interpretable. For example, a score near 5 implies that the model’s generated images almost always “match exactly” with the input prompts.

Analysis. Figure 3 presents human ratings for basic, advanced, and overall prompts. Notably, advanced prompts that require complex visio-linguistic reasoning are much harder. For example, the top-performing DALL-E 3 scores 4.3 on basic prompts, indicating “a few minor discrepancies”. However, on advanced prompts, its score drops to 3.4, indicating “several minor discrepancies”. Interestingly, models (e.g., DeepFloyd-IF and DALL-E 3) using stronger text embeddings from LLMs (e.g., T5 [54]) outperform those using CLIP text embeddings (e.g., SD-XL). Lastly, we observe that open-source and video-generative models lag behind their closed-source and image-generative counterparts, suggesting room for future innovation. In Appendix C, we detail model performance across various skills, highlighting challenges in higher-order reasoning like negation and comparison.

# 5 Evaluating Automated Metrics

We use our human ratings to benchmark automated alignment metrics [5, 20, 79] on GenAI-Bench.

Table 2: Evaluating the correlation of automated metrics with human ratings on GenAI-Bench. We report Pairwise accuracy [12], Pearson, and Kendall, with higher scores indicating better performance for all. VQAScore based on the CLIP-FlanT5 model [39] achieves the strongest agreement with human ratings on images and videos, significantly surpassing popular metrics like CLIPScore [20], PickScore [27], and Davidsonian [5].   

<table><tr><td rowspan="2">Method</td><td colspan="3">GenAI-Bench (Image)</td><td colspan="3">GenAI-Bench (Video)</td></tr><tr><td>Pairwise</td><td>Pearson</td><td>Kendall</td><td>Pairwise</td><td>Pearson</td><td>Kendall</td></tr><tr><td>CLIPScore [20]</td><td>50.8</td><td>16.4</td><td>11.8</td><td>53.6</td><td>25.3</td><td>18.0</td></tr><tr><td>BLIPv2Score [20]</td><td>52.2</td><td>17.2</td><td>14.7</td><td>54.6</td><td>25.3</td><td>20.1</td></tr><tr><td>ImageReward [78]</td><td>56.6</td><td>35.0</td><td>24.0</td><td>60.0</td><td>42.9</td><td>31.4</td></tr><tr><td>PickScore [27]</td><td>57.1</td><td>35.4</td><td>25.0</td><td>56.8</td><td>34.6</td><td>24.8</td></tr><tr><td>HPSv2 [76]</td><td>49.6</td><td>13.9</td><td>9.6</td><td>51.5</td><td>18.4</td><td>13.7</td></tr><tr><td>LLMScore [44]</td><td>53.2</td><td>15.4</td><td>13.6</td><td>53.2</td><td>19.4</td><td>17.7</td></tr><tr><td>BLIP-VQA [24]</td><td>54.3</td><td>27.1</td><td>23.0</td><td>55.1</td><td>29.8</td><td>22.5</td></tr><tr><td>VQ2 [79]</td><td>51.9</td><td>13.3</td><td>12.0</td><td>52.8</td><td>18.0</td><td>15.5</td></tr><tr><td>Davidsonian [5]</td><td>54.6</td><td>29.3</td><td>22.4</td><td>55.9</td><td>32.3</td><td>23.5</td></tr><tr><td>VQAScore [39]</td><td>64.1</td><td>49.9</td><td>39.8</td><td>63.2</td><td>50.6</td><td>38.2</td></tr></table>

VQAScore. Given an image and text, we calculate the probability of a “Yes” answer to a simple question like “Does this figure show ‘{text}’? Please answer yes or no.”:

$$
P (\text {` `} \text {Y e s} ^ {\prime \prime} | \text {i m a g e}, \text {q u e s t i o n}) \tag {1}
$$

We implement VQAScore using the open-source3 CLIP-FlanT5 model [39] trained on 665K public VQA data [40]. For video-text pairs, we average the scores across all video frames following [61].

Evaluation setup. To evaluate automated metrics on GenAI-Bench, we follow [23] to report the Pearson and Kendall coefficients, which measures the correlation of the metric score with human judgment. However, [12] (EMNLP’23 outstanding paper) show several issues with these metrics. For example, Pearson assumes a linear relationship between metric and human scores, while Kendall skips ties common in 1-to-5 Likert scales. As such, we report Pairwise accuracy [12], which is designed to address these issues. We refer readers to [12] for detailed equations and provide an overview below. For a dataset with $M$ items (e.g., image-text pairs), there are two $M$ -size score vectors: one for human ratings and one for metric scores. Pairwise accuracy (a value between 0 and 1) evaluates the percentage of agreement across all $M \times M$ pairs of items, that is, if one item scores higher, lower, or ties with another item in both human and metric scores. Lastly, we apply the tie calibration technique from [12] to find the optimal tie threshold for each metric.

Results. Table 2 shows that VQAScore significantly outperforms previous metrics such as CLIP-Score [20], models trained with extensive human feedback [27, 76, 78], and QG/A methods that use the same CLIP-FlanT5 VQA model [5, 79]. In addition, we implement BLIP-VQA [24] and LLMScore [44] using their official codebase. Appendix C shows that VQAScore achieves the best performance across all “basic” and “advanced” skills on GenAI-Bench. Appendix D shows that VQAScore also achieves the state-of-the-art performance on seven more alignment benchmarks such as TIFA160 [23] and Winoground [69]. Figure 4 qualitatively compare VQAScore against CLIPScore on random samples from GenAI-Bench. The strong performance of VQAScore makes it a more reliable tool for the future automated evaluation of text-to-visual models.

# 6 Improving Text-to-Visual Generation

VQAScore’s superior performance in evaluating text-to-visual generation suggests its potential to improve generation as well. We now show that VQAScore can improve the alignment of DALL-E 3 [1] and SD-XL [56] by simply ranking candidate images. We also collect a GenAI-Rank benchmark to evaluate scoring metrics on ranking images from the same prompt.

Ranking images by VQAScore. Given the same prompt, most text-to-visual models produce vastly different images with each run. As such, we adopt a black-box method [42] that improves text-toimage generation by selecting the highest-VQAScore image from a few generated candidates. This

![](images/f7c0d6a208c619aa3f956e8d517a6d9ba6769e51ffe9b93771c56b15fd44ffa7.jpg)  
Figure 5: VQAScore can select images generated by SD-XL [56] that outperform DALL-E 3’s [1]. Although less powerful in prompt alignment than DALL-E 3, SD-XL [56] can still be improved by selecting the highest VQAScore image from merely three candidates. We provide examples of how VQAScore ranks SD-XL images in Appendix A.

![](images/4bc6eb05b65ec374888a63dc718941f97f5c16c309c9341c499ed426eb1c92e7.jpg)  
(a) Improving DALL-E 3 by image ranking

![](images/da5f7997af929cd66b610886c0bd01ada12701937cb39da6e5e6002d5d743d59.jpg)  
(b) Improving SD-XL by image ranking   
Figure 6: Improving text-to-visual generation by ranking nine candidate images. We show the performance gains over the Random baseline (no ranking) in green and decreases in red. Notably, selecting the highest-VQAScore images from nine candidates significantly boosts the overall human alignment ratings. In contrast, ranking by CLIPScore [20] results in the same or lower performance. Overall, VQAScore is 2x to 3x more effective than other methods that rely on costly human feedback (PickScore [27]) or decompose texts using ChatGPT (Davidsonian [5]). Table 3 reports more scoring methods.

ranking-based approach is simple yet surprisingly effective. For instance, despite SD-XL’s weaker prompt alignment compared to DALL-E 3, Figure 5 shows how VQAScore can select the best SD-XL images (from three candidates) that outperform DALL-E 3’s. Figure 7 shows that VQAScore can also improve the closed-source (black-box) DALL-E 3 by correctly selecting the most prompt-aligned images from three candidates.

GenAI-Rank: A benchmark for text-to-image ranking. To compare against other ranking metrics (e.g., CLIPScore and PickScore), we hire three annotators to rate nine generated images for each prompt. In this study, we randomly select 800 prompts from GenAI-Bench and collect 43,200 human ratings for 14,400 images generated by DALL-E 3 and SD-XL. We will release this benchmark (termed GenAI-Rank) for reproducibility and to facilitate the evaluation of future ranking metrics.

VQAScore achieves superior performance gains. Figure 6 confirms that ranking by VQAScore delivers the most significant improvements in human ratings. While ranking by CLIPScore [20] results in the same or even lower performance, VQAScore consistently improves with more images to rank. VQAScore is also $2 \mathbf { x }$ to 3x more effective than other ranking metrics that rely on expensive human feedback (e.g., PickScore [27]) or decompose texts via ChatGPT (e.g., Davidsonian [5]). Table 3 details the performance gains for ranking 3 to 9 images across basic, advanced, and all prompts. VQAScore notably improves the prompt alignment of DALL-E 3 and SD-XL by about 0.3 on “advanced” prompts that require complex visio-linguistic reasoning, such as counting, comparison, and logic. Lastly, although we use the open-source CLIP-FlanT5 model for this study, future work may use stronger models such as GPT-4o for improved performance.

![](images/be2ce956dc54383b4923d5bf7cf5d62340aefb470bccda6fa5b471a24bac08aa.jpg)  
Figure 7: Ranking DALL-E 3 generated images with VQAScore or CLIPScore. VQAScore outperforms CLIPScore in ranking candidate images generated by DALL-E 3, particularly for prompts that involve attributes, relationships, and higher-order reasoning. This indicates that VQAScore can already improve text-to-image generation using only an image generation API [42].

Table 3: Comparing scoring methods for image ranking on GenAI-Rank. We present the average human ratings of 7 popular scoring methods across basic, advanced, and all prompts. Performance gains over the Random baseline (no ranking) are highlighted in green, while decreases are marked in red. We first note that ranking by CLIPScore [20] can lead to a performance drop. For instance, CLIPScore results in a 0.04 drop when given more images to rank (from 3 to 9). In contrast, VQAScore demonstrates consistent and significant improvements with more images. VQAScore improves performance on the more challenging “advanced” prompts that require complex visiolinguistic reasoning skills like counting, comparison, and logic. For these “advanced” prompts, VQAScore boosts DALL-E 3 by 0.30 and SD-XL by 0.27 by ranking nine images, outperforming the second-best method PickScore [27] by $2 \mathbf { x }$ to 3x. For reference, we include human (oracle) performance (ranking by ground-truth human ratings). GenAI-Rank releases all human ratings to help benchmark future ranking metrics.

<table><tr><td rowspan="2">Method</td><td colspan="2">Basic</td><td colspan="2">Advanced</td><td colspan="2">Overall</td></tr><tr><td>3 Images</td><td>9 Images</td><td>3 Images</td><td>9 Images</td><td>3 Images</td><td>9 Images</td></tr><tr><td>Random</td><td>4.51</td><td>4.51</td><td>3.77</td><td>3.77</td><td>4.03</td><td>4.03</td></tr><tr><td>Human Oracle</td><td>\( {4.77}_{+{26}} \)</td><td>\( {4.89}_{+{38}} \)</td><td>\( {4.18}_{+{41}} \)</td><td>\( {4.46}_{+{69}} \)</td><td>\( {4.39}_{+{36}} \)</td><td>\( {4.61}_{+{58}} \)</td></tr><tr><td>CLIPScore [20]</td><td>\( {4.54}_{+{03}} \)</td><td>\( {4.53}_{+{02}} \)</td><td>\( {3.79}_{+{02}} \)</td><td>\( {3.73}_{-{04}} \)</td><td>\( {4.05}_{+{02}} \)</td><td>\( {4.01}_{-{02}} \)</td></tr><tr><td>ImageReward [78]</td><td>\( {4.56}_{+{05}} \)</td><td>\( {4.52}_{+{01}} \)</td><td>\( {3.82}_{+{05}} \)</td><td>\( {3.83}_{+{06}} \)</td><td>\( {4.08}_{+{05}} \)</td><td>\( {4.08}_{+{05}} \)</td></tr><tr><td>PickScore [27]</td><td>\( {4.58}_{+{07}} \)</td><td>\( {4.60}_{+{09}} \)</td><td>\( {3.82}_{+{05}} \)</td><td>\( {3.81}_{+{04}} \)</td><td>\( {4.09}_{+{06}} \)</td><td>\( {4.09}_{+{06}} \)</td></tr><tr><td>HPSv2 [76]</td><td>\( {4.57}_{+{06}} \)</td><td>\( {4.60}_{+{09}} \)</td><td>\( {3.80}_{+{03}} \)</td><td>\( {3.78}_{+{01}} \)</td><td>\( {4.07}_{+{04}} \)</td><td>\( {4.07}_{+{04}} \)</td></tr><tr><td>VQ2 [79]</td><td>\( {4.54}_{+{03}} \)</td><td>\( {4.55}_{+{04}} \)</td><td>\( {3.79}_{+{02}} \)</td><td>\( {3.79}_{+{02}} \)</td><td>\( {4.05}_{+{02}} \)</td><td>\( {4.06}_{+{03}} \)</td></tr><tr><td>Davidsonian [5]</td><td>\( {4.56}_{+{05}} \)</td><td>\( {4.61}_{+{10}} \)</td><td>\( {3.83}_{+{05}} \)</td><td>\( {3.84}_{+{07}} \)</td><td>\( {4.09}_{+{06}} \)</td><td>\( {4.12}_{+{09}} \)</td></tr><tr><td>VQAScore</td><td>\( {4.59}_{+{08}} \)</td><td>\( {4.62}_{+{11}} \)</td><td>\( {3.92}_{+{15}} \)</td><td>\( {4.05}_{+{28}} \)</td><td>\( {4.16}_{+{13}} \)</td><td>\( {4.25}_{+{22}} \)</td></tr></table>

(a) Improving DALL-E 3 by image ranking

<table><tr><td rowspan="2">Method</td><td colspan="2">Basic</td><td colspan="2">Advanced</td><td colspan="2">Overall</td></tr><tr><td>3 Img</td><td>9 Img</td><td>3 Img</td><td>9 Img</td><td>3 Img</td><td>9 Img</td></tr><tr><td>Random</td><td>3.80</td><td>3.80</td><td>3.02</td><td>3.02</td><td>3.30</td><td>3.30</td></tr><tr><td>Human (Oracle)</td><td>4.17+ .37</td><td>4.41+ .61</td><td>3.38+ .36</td><td>3.70+ .68</td><td>3.66+ .36</td><td>3.95+ .65</td></tr><tr><td>CLIPScore [20]</td><td>3.86+ .06</td><td>3.92+ .12</td><td>3.06+ .04</td><td>3.06+ .04</td><td>3.34+ .04</td><td>3.37+ .07</td></tr><tr><td>ImageReward [78]</td><td>3.94+ .14</td><td>3.96+ .16</td><td>3.10+ .08</td><td>3.15+ .13</td><td>3.40+ .10</td><td>3.44+ .14</td></tr><tr><td>PickScore [27]</td><td>3.94+ .14</td><td>4.02+ .22</td><td>3.13+ .11</td><td>3.17+ .15</td><td>3.42+ .12</td><td>3.47+ .17</td></tr><tr><td>HPSv2 [76]</td><td>3.91+ .11</td><td>3.99+ .19</td><td>3.11+ .09</td><td>3.15+ .13</td><td>3.39+ .09</td><td>3.45+ .15</td></tr><tr><td>VQ2 [79]</td><td>3.83+ .03</td><td>3.88+ .08</td><td>3.06+ .04</td><td>3.08+ .06</td><td>3.33+ .03</td><td>3.37+ .07</td></tr><tr><td>Davidsonian [5]</td><td>3.85+ .05</td><td>3.88+ .08</td><td>3.06+ .04</td><td>3.11+ .09</td><td>3.34+ .04</td><td>3.39+ .09</td></tr><tr><td>VQAScore</td><td>3.94+ .14</td><td>4.06+ .26</td><td>3.15+ .13</td><td>3.29+ .27</td><td>3.43+ .13</td><td>3.56+ .26</td></tr></table>

(b) Improving SD-XL by image ranking

# 7 Goodhart’s Law Still Applies

When a measure becomes a target, it ceases to be a good measure.

— Marilyn Strathern [64]

This quote conveys the essence of Goodhart’s Law [17, 18]: an over-optimized metric inevitably loses its effectiveness. This phenomenon is well-documented in fields such as machine learning [22, 68], economics [10, 18], and education [2, 28]. Acknowledging that VQAScore is also subject to this law, we examine its limitations as an automated metric and suggest avenues for future improvements.

![](images/6e7546fd61ff12463aa4a9a9b319b32d95d7ce3c0a51edde825bedb6db9d707a.jpg)  
(a) Too many objects

![](images/e81cb313cd240752a5819e495199d9408a0e9dcdfdbca9c08f82daf67d49be93.jpg)  
(b) Fine-grained visual details

![](images/f1bb7d7a2d340f821c339a7b47eb4b782eefc8f601ebc838c5fe0fe201bc43fe.jpg)  
(c) Linguistic ambiguity   
Figure 8: Limitations of VQAScore (please zoom into the figures for a detailed view). We identify three failure cases of VQAScore (based on CLIP-FlanT5). (a) While VQAScore can reasonably count objects in small quantities, it struggles with larger numbers. (b) VQAScore can overlook small visual details, such as entities that occupy only a small portion of the image. (c) VQAScore may not understand ambiguous prompts, misinterpreting “two shoes” as “two pairs of shoes”, or “towards the left (of the viewer)” as “towards the left (of the swan)”.

Limitations of VQAScore. We conduct a qualitative study by manually examining samples where VQAScore and human ratings disagree. Figure 8 identifies three failure cases: (1) miscounting when there are too many objects, (2) overlooking fine-grained visual details, and (3) misinterpreting linguistic ambiguity. We posit that VQA models with higher image resolution [60] and more capable language models [49, 67] may improve on these challenging aspects. Despite these mild limitations, we strongly recommend adopting VQAScore as a more reliable alternative to CLIPScore, which has already ceased to be an effective metric [25, 38, 81]. We believe VQAScore also serves well as a reproducible supplement to non-reproducible human studies [50].

# 8 Conclusion

Limitations and future work. Currently, GenAI-Bench does not evaluate several vital aspects of generative models [31, 43, 51, 75], such as toxicity, bias, aesthetics, and video motion. Future work may also incorporate other interesting aspects of visual generation, such as mixed media, optical effects, reflection, and world knowledge, as explored in datasets such as DOCCI [48]. Although our ranking-based approach is effective, future work may explore white-box finetuning techniques [3, 53, 77] for more efficient inference.

Summary. We have conducted an extensive human study with GenAI-Bench, focusing on both compositional text-to-visual generation and automated evaluation metrics. We show a straightforward ranking-based method that improves the prompt alignment of black-box generative models. By discussing Goodhart’s Law, we hope to encourage further research into automated evaluation techniques, which is essential to the scientific progression of this field.

# References

[1] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. https://cdn.openai. com/papers/dall-e-3.pdf, 2023.   
[2] Mario Biagioli. Watch out for cheats in citation game. Nature, 535(7611):201–201, 2016.   
[3] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.   
[4] Huiwen Chang, Han Zhang, Jarred Barber, AJ Maschinot, Jose Lezama, Lu Jiang, Ming-Hsuan Yang, Kevin Murphy, William T Freeman, Michael Rubinstein, et al. Muse: Text-to-image generation via masked generative transformers. arXiv preprint arXiv:2301.00704, 2023.   
[5] Jaemin Cho, Yushi Hu, Roopal Garg, Peter Anderson, Ranjay Krishna, Jason Baldridge, Mohit Bansal, Jordi Pont-Tuset, and Su Wang. Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-image generation. arXiv preprint arXiv:2310.18235, 2023.   
[6] Jaemin Cho, Abhay Zala, and Mohit Bansal. Dall-eval: Probing the reasoning skills and social biases of text-to-image generation models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3043–3054, 2023.   
[7] Jaemin Cho, Abhay Zala, and Mohit Bansal. Visual programming for text-to-image generation and evaluation. arXiv preprint arXiv:2305.15328, 2023.   
[8] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.   
[9] Xiaoliang Dai, Ji Hou, Chih-Yao Ma, Sam Tsai, Jialiang Wang, Rui Wang, Peizhao Zhang, Simon Vandenhende, Xiaofang Wang, Abhimanyu Dubey, et al. Emu: Enhancing image generation models using photogenic needles in a haystack. arXiv preprint arXiv:2309.15807, 2023.   
[10] Jón Danıelsson. The emperor has no clothes: Limits to risk modelling. Journal of Banking & Finance, 26 (7):1273–1296, 2002.   
[11] Deepfloyd IF. Deepfloyd IF. https://github.com/deep-floyd/IF, 2024.   
[12] Daniel Deutsch, George Foster, and Markus Freitag. Ties matter: Meta-evaluating modern metrics with pairwise accuracy and tie calibration. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12914–12929, 2023.   
[13] Floor33. Floor33. https://www.morphstudio.com/, 2023.   
[14] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. arXiv preprint arXiv:2208.01618, 2022.   
[15] Rinon Gal, Or Patashnik, Haggai Maron, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. Stylegannada: Clip-guided domain adaptation of image generators. ACM Transactions on Graphics (TOG), 41(4): 1–13, 2022.   
[16] Gen2. Gen2. https://research.runwayml.com/gen2, 2024.   
[17] Charles Goodhart. Goodhart’s law. Edward Elgar Publishing Cheltenham, UK, 2015.   
[18] Charles AE Goodhart and CAE Goodhart. Problems of monetary management: the UK experience. Springer, 1984.   
[19] Jianshu Guo, Wenhao Chai, Jie Deng, Hsiang-Wei Huang, Tian Ye, Yichen Xu, Jiawei Zhang, Jenq-Neng Hwang, and Gaoang Wang. Versat2i: Improving text-to-image models with versatile reward. arXiv preprint arXiv:2403.18493, 2024.   
[20] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. Clipscore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.   
[21] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

[22] Jennifer Hsia, Danish Pruthi, Aarti Singh, and Zachary C Lipton. Goodhart’s law applies to nlp’s explanation benchmarks. arXiv preprint arXiv:2308.14272, 2023.   
[23] Yushi Hu, Benlin Liu, Jungo Kasai, Yizhong Wang, Mari Ostendorf, Ranjay Krishna, and Noah A Smith. Tifa: Accurate and interpretable text-to-image faithfulness evaluation with question answering. arXiv preprint arXiv:2303.11897, 2023.   
[24] Kaiyi Huang, Kaiyue Sun, Enze Xie, Zhenguo Li, and Xihui Liu. T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation. arXiv preprint arXiv:2307.06350, 2023.   
[25] Amita Kamath, Jack Hessel, and Kai-Wei Chang. Text encoders bottleneck compositionality in contrastive vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4933–4944, 2023.   
[26] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. Imagic: Text-based real image editing with diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6007–6017, 2023.   
[27] Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Pick-a-pic: An open dataset of user preferences for text-to-image generation. 2023.   
[28] Vladlen Koltun and David Hafner. The h-index is no longer an effective correlate of scientific reputation. PLoS One, 16(6):e0253397, 2021.   
[29] Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image diffusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1931–1941, 2023.   
[30] Kimin Lee, Hao Liu, Moonkyung Ryu, Olivia Watkins, Yuqing Du, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, and Shixiang Shane Gu. Aligning text-to-image models using human feedback. arXiv preprint arXiv:2302.12192, 2023.   
[31] Tony Lee, Michihiro Yasunaga, Chenlin Meng, Yifan Mai, Joon Sung Park, Agrim Gupta, Yunzhi Zhang, Deepak Narayanan, Hannah Benita Teufel, Marco Bellagente, et al. Holistic evaluation of text-to-image models. arXiv preprint arXiv:2311.04287, 2023.   
[32] Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Li, Yixin Fei, Kewen Wu, Xide Xia, Pengchuan Zhang, Graham Neubig, and Deva Ramanan. Evaluating and improving compositional text-to-visual generation. In The First Workshop on the Evaluation of Generative Foundation Models at CVPR, 2024.   
[33] Baiqi Li, Zhiqiu Lin, Deepak Pathak, Jiayao Emily Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. GenAI-bench: A holistic benchmark for compositional text-to-visual generation. In Synthetic Data for Computer Vision Workshop @ CVPR 2024, 2024.   
[34] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023.   
[35] Jiachen Li, Weixi Feng, Wenhu Chen, and William Yang Wang. Reward guided latent consistency distillation. arXiv preprint arXiv:2403.11027, 2024.   
[36] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part V 13, pages 740–755. Springer, 2014.   
[37] Zhiqiu Lin, Samuel Yu, Zhiyi Kuang, Deepak Pathak, and Deva Ramanan. Multimodality helps unimodality: Cross-modal few-shot learning with multimodal models, 2023.   
[38] Zhiqiu Lin, Xinyue Chen, Deepak Pathak, Pengchuan Zhang, and Deva Ramanan. Revisiting the role of language priors in vision-language models. arXiv preprint arXiv:2306.01879, 2024.   
[39] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan. Evaluating text-to-visual generation with image-to-text generation. arXiv preprint arXiv:2404.01291, 2024.   
[40] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744, 2023.

[41] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. In European Conference on Computer Vision, pages 423–439. Springer, 2022.   
[42] Shihong Liu, Zhiqiu Lin, Samuel Yu, Ryan Lee, Tiffany Ling, Deepak Pathak, and Deva Ramanan. Language models as black-box optimizers for vision-language models. arXiv preprint arXiv:2309.05950, 2024.   
[43] Yaofang Liu, Xiaodong Cun, Xuebo Liu, Xintao Wang, Yong Zhang, Haoxin Chen, Yang Liu, Tieyong Zeng, Raymond Chan, and Ying Shan. Evalcrafter: Benchmarking and evaluating large video generation models. arXiv preprint arXiv:2310.11440, 2023.   
[44] Yujie Lu, Xianjun Yang, Xiujun Li, Xin Eric Wang, and William Yang Wang. Llmscore: Unveiling the power of large language models in text-to-image synthesis evaluation. arXiv preprint arXiv:2305.11116, 2023.   
[45] Oscar Mañas, Pietro Astolfi, Melissa Hall, Candace Ross, Jack Urbanek, Adina Williams, Aishwarya Agrawal, Adriana Romero-Soriano, and Michal Drozdzal. Improving text-to-image consistency via automatic prompt optimization. arXiv preprint arXiv:2403.17804, 2024.   
[46] Chenlin Meng, Yutong He, Yang Song, Jiaming Song, Jiajun Wu, Jun-Yan Zhu, and Stefano Ermon. Sdedit: Guided image synthesis and editing with stochastic differential equations. arXiv preprint arXiv:2108.01073, 2021.   
[47] Midjourney. Midjourney. https://www.midjourney.com, 2024.   
[48] Yasumasa Onoe, Sunayana Rane, Zachary Berger, Yonatan Bitton, Jaemin Cho, Roopal Garg, Alexander Ku, Zarana Parekh, Jordi Pont-Tuset, Garrett Tanzer, et al. Docci: Descriptions of connected and contrasting images. arXiv preprint arXiv:2404.19753, 2024.   
[49] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
[50] Mayu Otani, Riku Togashi, Yu Sawai, Ryosuke Ishigami, Yuta Nakashima, Esa Rahtu, Janne Heikkilä, and Shin’ichi Satoh. Toward verifiable and reproducible human evaluation for text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14277– 14286, 2023.   
[51] Shubham Parashar, Zhiqiu Lin, Tian Liu, Xiangjue Dong, Yanan Li, Deva Ramanan, James Caverlee, and Shu Kong. The neglected tails of vision-language models. arXiv preprint arXiv:2401.12425, 2024.   
[52] Pika. Pika. https://www.pika.art/, 2024.   
[53] Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-to-image diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023.   
[54] Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020.   
[55] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.   
[56] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.   
[57] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22500–22510, 2023.   
[58] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-toimage diffusion models with deep language understanding. Advances in Neural Information Processing Systems, 35:36479–36494, 2022.   
[59] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.

[60] Baifeng Shi, Ziyang Wu, Maolin Mao, Xin Wang, and Trevor Darrell. When do we not need larger vision models? arXiv preprint arXiv:2403.13043, 2024.   
[61] Uriel Singer, Adam Polyak, Thomas Hayes, Xi Yin, Jie An, Songyang Zhang, Qiyuan Hu, Harry Yang, Oron Ashual, Oran Gafni, et al. Make-a-video: Text-to-video generation without text-video data. arXiv preprint arXiv:2209.14792, 2022.   
[62] Jaskirat Singh and Liang Zheng. Divide, evaluate, and refine: Evaluating and improving text-to-image alignment with iterative vqa feedback. arXiv preprint arXiv:2307.04749, 2023.   
[63] Sora. Sora. https://openai.com/sora, 2024.   
[64] Marilyn Strathern. ‘improving ratings’: audit in the british university system. European review, 5(3): 305–321, 1997.   
[65] Jiao Sun, Deqing Fu, Yushi Hu, Su Wang, Royi Rassin, Da-Cheng Juan, Dana Alon, Charles Herrmann, Sjoerd van Steenkiste, Ranjay Krishna, et al. Dreamsync: Aligning text-to-image generation with image understanding feedback. arXiv preprint arXiv:2311.17946, 2023.   
[66] Jiao Sun, Deqing Fu, Yushi Hu, Su Wang, Royi Rassin, Da-Cheng Juan, Dana Alon, Charles Herrmann, Sjoerd van Steenkiste, Ranjay Krishna, et al. Dreamsync: Aligning text-to-image generation with image understanding feedback. arXiv preprint arXiv:2311.17946, 2023.   
[67] Yi Tay, Mostafa Dehghani, Vinh Q Tran, Xavier Garcia, Dara Bahri, Tal Schuster, Huaixiu Steven Zheng, Neil Houlsby, and Donald Metzler. Unifying language learning paradigms. arXiv preprint arXiv:2205.05131, 2022.   
[68] Damien Teney, Ehsan Abbasnejad, Kushal Kafle, Robik Shrestha, Christopher Kanan, and Anton Van Den Hengel. On the value of out-of-distribution testing: An example of goodhart’s law. Advances in neural information processing systems, 33:407–417, 2020.   
[69] Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross. Winoground: Probing vision and language models for visio-linguistic compositionality. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5238–5248, 2022.   
[70] Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. arXiv preprint arXiv:2311.12908, 2023.   
[71] Jiuniu Wang, Hangjie Yuan, Dayou Chen, Yingya Zhang, Xiang Wang, and Shiwei Zhang. Modelscope text-to-video technical report. arXiv preprint arXiv:2308.06571, 2023.   
[72] Su Wang, Chitwan Saharia, Ceslee Montgomery, Jordi Pont-Tuset, Shai Noy, Stefano Pellegrini, Yasumasa Onoe, Sarah Laszlo, David J Fleet, Radu Soricut, et al. Imagen editor and editbench: Advancing and evaluating text-guided image inpainting. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18359–18369, 2023.   
[73] Jay Zhangjie Wu, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie, and Mike Zheng Shou. Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 7623–7633, 2023.   
[74] Jay Zhangjie Wu, Guian Fang, Haoning Wu, Xintao Wang, Yixiao Ge, Xiaodong Cun, David Junhao Zhang, Jia-Wei Liu, Yuchao Gu, Rui Zhao, et al. Towards a better metric for text-to-video generation. arXiv preprint arXiv:2401.07781, 2024.   
[75] Tong Wu, Guandao Yang, Zhibing Li, Kai Zhang, Ziwei Liu, Leonidas Guibas, Dahua Lin, and Gordon Wetzstein. Gpt-4v (ision) is a human-aligned evaluator for text-to-3d generation. arXiv preprint arXiv:2401.04092, 2024.   
[76] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.   
[77] Xiaoshi Wu, Yiming Hao, Manyuan Zhang, Keqiang Sun, Zhaoyang Huang, Guanglu Song, Yu Liu, and Hongsheng Li. Deep reward supervisions for tuning text-to-image diffusion models. arXiv preprint arXiv:2405.00760, 2024.

[78] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. arXiv preprint arXiv:2304.05977, 2023.   
[79] Michal Yarom, Yonatan Bitton, Soravit Changpinyo, Roee Aharoni, Jonathan Herzig, Oran Lang, Eran Ofek, and Idan Szpektor. What you see is what you read? improving text-image alignment evaluation. arXiv preprint arXiv:2305.10400, 2023.   
[80] Jiahui Yu, Yuanzhong Xu, Jing Yu Koh, Thang Luong, Gunjan Baid, Zirui Wang, Vijay Vasudevan, Alexander Ku, Yinfei Yang, Burcu Karagol Ayan, et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.   
[81] Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou. When and why vision-language models behave like bags-of-words, and what to do about it? In The Eleventh International Conference on Learning Representations, 2022.   
[82] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.

# GenAI-Bench: Evaluating and Improving Compositional Text-to-Visual Generation

Supplementary Material

# Outline

This document supplements the main paper with benchmark and method details. Below is the outline:

• Section A presents additional qualitative studies.   
• Section B details GenAI-Bench’s skill taxonomy.   
• Section C describes how we collect GenAI-Bench.   
• Section D discusses other baseline methods.

# A Additional Examples

Ranking SD-XL images by VQAScore. Figure 9 shows that ranking by VQAScore can also improve the prompt alignment of SD-XL using only its image generation API. We encourage future work to explore other white-box techniques for finetuning [19, 30, 35, 37, 57, 66, 70].

![](images/4810361665f015f16420ed5b8d8bd85ee3309f5b77f4ee6d3e1fbd8282f65e18.jpg)  
Figure 9: Ranking SD-XL generated images with VQAScore and CLIPScore. VQAScore outperforms CLIPScore in ranking candidate images generated by SD-XL, particularly for advanced prompts that involve complex visio-linguistic reasoning.

# B Skill taxonomy of GenAI-Bench

We now detail the evaluated skills of GenAI-Bench.

Skill definitions. Previous benchmarks for text-to-visual generation [6, 23, 24, 58, 80] primarily focus on generating basic objects, attributes, relations, and scenes. While these “basic” visual compositions still pose challenges, real-world user prompts often introduce greater complexity, such as higher-order reasoning beyond basic compositions. For example, while existing benchmarks focus only on counting objects [23, 80], real-world prompts often require counting complex objectattribute-relation compositions, like “one person wearing a white shirt and the other five wearing blue shirts”. To this end, after reviewing relevant literature [24, 47, 69, 80] and discussing with professional designers, we introduce a set of compositional reasoning skills and categorize them into “basic” and “advanced”, where the latter can build upon the former. For logical reasoning, we focus on “negation” and “universality”. We find these are the two most common types of logic in user prompts. Other logical operators such as conjunction [41] are not included because the logical “AND” is usually implied in prompts that involve multiple objects, and “OR” is rarely useful as designers tend to specify a more precise prompt. Lastly, we plan to evaluate image styles in future work. We detail the definitions for “basic” skills in Table 4 and “advanced” skills in Table 5.

Table 4: Skill definitions and examples for basic compositions.   

<table><tr><td>Skill Type</td><td>Definition</td><td>Examples</td></tr><tr><td>Object</td><td>Basic Compositions
Basic entities within an image, such as person, animal, food, items, vehicles, or text symbols (e.g., “A”, “1+1”).</td><td>a dog, a cat and a chicken on a table; a young man with a green bat and a blue ball; a ‘No Parking’ sign on a busy street.</td></tr><tr><td>Attribute</td><td>Visual attributes (properties) of entities, such as color, material, emotion, size, shape, age, gender, state, and so on.</td><td>a silver spoon lies to the left of a golden fork on a wooden table; a green pumpkin is smiling happily, a red pumpkin is sitting sadly.</td></tr><tr><td>Scene</td><td>Backgrounds or settings of an image, such as weather and location.</td><td>A child making a sandcastle on a beach in a cloudy day; a grand fountain surrounded by historic buildings in a town square.</td></tr><tr><td>Spatial Relation</td><td>Physical arrangements of multiple entities relative to each other, e.g., on the right, on top, facing, towards, inside, outside, near, far, and so on.</td><td>a bustling city street, a neon ‘Open 24 Hours’ sign glowing above a small diner; a teacher standing in front of a world map in a classroom; tea steams in a cup, next to a closed diary with a pen resting on its cover.</td></tr><tr><td>Action Relation</td><td>Action interactions between entities, e.g., pushing, kissing, hugging, hitting, helping, and so on.</td><td>a group of children playing on the beach; a boat glides across the ocean, dolphins leaping beside it and seagulls soaring overhead.</td></tr><tr><td>Part Relation</td><td>Part-whole relationships between entities – one entity is a component of another, such as body part, clothing, and accessories.</td><td>a pilot with aviator sunglasses; a baker with a cherry pin on a polka dot apron.; a young lady wearing a T-shirt puts her hand on a puppy’s head.</td></tr></table>

Skill coverage in benchmarks. We find the skill categorization in benchmarks like PartiPrompt [80] to be somewhat confusing. For instance, PartiPrompt introduces two categories “complex” and “finegrained detail”. The former refers to “...fine-grained, interacting details or relationships between multiple participants”, while the latter refers to “...attributes or actions of entities or objects in a scene”. Upon closer examination, the categorization of spatial, action, and part relations into these categories are arbitrary. To address this, we attempt to compare the skill coverage across all benchmarks by our unified set of compositional reasoning skills. For benchmarks (PartiPrompt/T2I-CompBench) with pre-defined skill categories, we map their skills to our definitions. For the other benchmarks that do not have a comprehensive skill set, we manually annotate a random subset of samples. Finally, we calculate the skill proportions in each benchmark, identifying skills that constitute more than $2 \%$ as genuinely present.

Table 5: Skill definitions and examples for advanced compositions.   

<table><tr><td>Skill Type</td><td>Definition</td><td>Examples</td></tr><tr><td rowspan="2">Counting</td><td colspan="2">Advanced Compositions</td></tr><tr><td>Determining the quantity, size, or volume of entities, e.g., objects, attribute-object pairs, and object-relation-object triplets.</td><td>two cats playing with a single ball; five enthusiastic athletes and one tired coach; one pirate ship sailing through space, crewed by five robots; three pink peonies and four white daisies in a garden.</td></tr><tr><td>Differentiation</td><td>Differentiating objects within a category by their attributes or relations, such as distinguishing between “old” and “young” people by age, or “the cat on top of the table” versus “the cat under the table” by their spatial relations.</td><td>one cat is sleeping on the table and the other is playing under the table; there are two men in the living room, the taller one to the left of the shorter one; a notebook lies open in the grass, with sketches on the left page and blank space on the right; there are two shoes on the grass, the one without laces looks newer than the one with laces.</td></tr><tr><td>Comparison</td><td>Comparing characteristics like number, attributes, area, or volume between entities.</td><td>there are more people standing than sitting; between the two cups on the desk, the taller one holds more coffee than the shorter one, which is half-empty; three little boys are sitting on the grass, and the boy in the middle looks the strongest.</td></tr><tr><td>Negation</td><td>Specifying the absence or contradiction of elements, as indicated by “no”, “not”, or “without”, e.g., entities not present or actions not taken.</td><td>a bookshelf with no books, only picture frames.; a person with short hair is crying while a person with long hair is not; a smiling girl with short hair and no glasses; a cute dog without a collar.</td></tr><tr><td>Universality</td><td>Specifying when every member of a group shares a specific attribute or is involved in a common relation, indicated by words like “every”, “all”, “each”, “both”.</td><td>in a room, all the chairs are occupied except one; a bustling kitchen where every chef is preparing a dish; in a square, several children are playing, each wearing a red T-shirt; a table laden with apples and bananas, where all the fruits are green; the little girl in the garden has roses in both hands.</td></tr></table>

# C GenAI-Bench

This section describes how we collect GenAI-Bench.

Details of GenAI-Bench. GenAI-Bench consists of 1,600 diverse prompts. To ensure our prompts reflect real-world applications, we collaborate with graphic designers who regularly use text-to-visual tools like Midjourney [47]. First, we collaborate with them to refine our skill taxonomy, identifying practical skills that current models still struggle with. They then collect compositional prompts relevant to their professional needs. To avoid copyright issues, we advise them to write prompts about generic subjects. We provide designers with sample prompts from existing benchmarks like PartiPrompt for inspiration and encourage the use of ChatGPT to brainstorm prompt variants across diverse visual domains. Crucially, these designers ensure that the prompts are objective. This contrasts with [24], whose prompts are almost entirely auto-generated. For example, in the “texture” category of T2I-CompBench, an overwhelming $40 \%$ of the 1000 auto-generated prompts use “metallic” as the attribute, which limits their diversity. Other T2I-CompBench’s prompts generated by ChatGPT often contain subjective phrases. For instance, in the prompt “the delicate, fluttering wings of the butterfly signaled the arrival of spring, a natural symbol of rebirth and renewal”, the “rebirth and renewal” can convey different meanings to different people. Similarly, in “the soft, velvety texture of the rose petals felt luxurious against the fingertips, a romantic symbol of love and affection”, the “love and affection” is open to diverse interpretations. Thus, we guide the designers to avoid such prompts. Lastly, each prompt in GenAI-Bench is tagged with all its evaluated skills. In total, we collect over 5,000 human-verified tags with a balanced distribution of “basic” and “advanced” skills.

Collecting human ratings. We evaluate six text-to-image models: Stable Diffusion [56] (SD v2.1, SD-XL, SD-XL Turbo), DeepFloyd-IF [11], Midjourney v6 [47], DALL-E 3 [1]; along with four text-to-video models: ModelScope [71], Floor33 [13], Pika v1 [52], Gen2 [16]. Due to the lack

of APIs for Floor33 [13], Pika v1 [52], and Gen2 [16], we manually download videos from their websites. For image generative models, we generate images using all 1,600 GenAI-Bench prompts. We use a coreset of 800 prompts to collect videos for the four video models. The same 800 prompts are used to collect the GenAI-Rank benchmark in the main paper. In total, we collect over 80,000 human ratings, greatly exceeding the scale of human annotations in previous work [5, 23], e.g., TIFA160 collected 2,400 ratings. We pay the local minimum wage of 12 dollars per hour for a total of about 800 annotator hours.

GenAI-Bench performance. We detail the performance of the ten image and video generative models across all skills in Table 6. Both humans and VQAScores rate DALL-E 3 [1] higher than the other models in nearly all skills, except for negation. In addition, prompts requiring “advanced” compositions are rated significantly lower by both humans and VQAScores, with negation being the most challenging skill. Lastly, current video models do not perform as well as image models, suggesting room for improvement.

Performance of automated metrics across skill dimensions. To provide a more comprehensive comparison between VQAScore and other metrics, we report Pairwise Accuracy across basic and advanced skill dimensions on GenAI-Bench (Image) in Table 7.

Table 6: Performance breakdown on GenAI-Bench. We present the averaged human ratings and VQAScores (based on CLIP-FlanT5) for “basic” and “advanced” prompts. Human ratings use a 1-5 Likert scale, and VQAScore ranges from 0 to 1, with higher scores indicating better performance for both. Generally, both human ratings and VQAScores favor DALL-E 3 over other models, with DALL-E 3 preferred across almost all skills except for negation. We find that “advanced” prompts that require higher-order reasoning present significant challenges. For instance, the state-of-the-art DALL-E 3 receives a remarkable average human rating of 4.3 for “basic” prompts, indicating the images and prompts range from “having a few minor discrepancies” to “matching exactly”. However, it scores only 3.4 for “advanced” prompts, suggesting “several minor discrepancies”. In addition, video models receive significantly lower scores than image models. Overall, VQAScores closely match human ratings.

(a) Human ratings on “basic” prompts   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Attribute</td><td rowspan="2">Scene</td><td colspan="3">Relation</td><td rowspan="2">Avg</td></tr><tr><td>Spatial</td><td>Action</td><td>Part</td></tr><tr><td colspan="7">Image models</td></tr><tr><td>SD v2.1</td><td>3.2</td><td>3.4</td><td>3.1</td><td>3.2</td><td>2.9</td><td>3.2</td></tr><tr><td>SD-XL Turbo</td><td>3.5</td><td>3.6</td><td>3.4</td><td>3.4</td><td>3.2</td><td>3.5</td></tr><tr><td>SD-XL</td><td>3.6</td><td>3.8</td><td>3.5</td><td>3.6</td><td>3.4</td><td>3.6</td></tr><tr><td>DeepFloyd-IF</td><td>3.6</td><td>3.7</td><td>3.6</td><td>3.6</td><td>3.4</td><td>3.6</td></tr><tr><td>Midjourney v6</td><td>4.0</td><td>4.1</td><td>4.0</td><td>4.1</td><td>3.8</td><td>4.0</td></tr><tr><td>DALL-E 3</td><td>4.3</td><td>4.4</td><td>4.2</td><td>4.3</td><td>4.2</td><td>4.3</td></tr><tr><td colspan="7">Video models</td></tr><tr><td>ModelScope</td><td>3.1</td><td>3.1</td><td>2.8</td><td>3.0</td><td>3.1</td><td>3.0</td></tr><tr><td>Floor33</td><td>3.2</td><td>3.2</td><td>2.9</td><td>3.2</td><td>3.1</td><td>3.1</td></tr><tr><td>Pika v1</td><td>3.4</td><td>3.4</td><td>3.1</td><td>3.3</td><td>3.2</td><td>3.3</td></tr><tr><td>Gen2</td><td>3.6</td><td>3.7</td><td>3.4</td><td>3.6</td><td>3.6</td><td>3.6</td></tr></table>

(b) VQAScores on “basic” prompts   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Attribute</td><td rowspan="2">Scene</td><td colspan="3">Relation</td><td rowspan="2">Avg</td></tr><tr><td>Spatial</td><td>Action</td><td>Part</td></tr><tr><td colspan="7">Image models</td></tr><tr><td>SD v2.1</td><td>0.75</td><td>0.79</td><td>0.73</td><td>0.73</td><td>0.71</td><td>0.75</td></tr><tr><td>SD-XL Turbo</td><td>0.81</td><td>0.82</td><td>0.78</td><td>0.79</td><td>0.78</td><td>0.80</td></tr><tr><td>SD-XL</td><td>0.82</td><td>0.85</td><td>0.80</td><td>0.80</td><td>0.81</td><td>0.82</td></tr><tr><td>DeepFloyd-IF</td><td>0.82</td><td>0.83</td><td>0.80</td><td>0.81</td><td>0.81</td><td>0.82</td></tr><tr><td>Midjourney v6</td><td>0.86</td><td>0.88</td><td>0.86</td><td>0.87</td><td>0.85</td><td>0.86</td></tr><tr><td>DALL-E 3</td><td>0.91</td><td>0.91</td><td>0.90</td><td>0.90</td><td>0.91</td><td>0.90</td></tr><tr><td colspan="7">Video models</td></tr><tr><td>ModelScope</td><td>0.69</td><td>0.69</td><td>0.65</td><td>0.65</td><td>0.70</td><td>0.66</td></tr><tr><td>Floor33</td><td>0.70</td><td>0.71</td><td>0.64</td><td>0.66</td><td>0.67</td><td>0.67</td></tr><tr><td>Pika v1</td><td>0.78</td><td>0.80</td><td>0.74</td><td>0.72</td><td>0.76</td><td>0.75</td></tr><tr><td>Gen2</td><td>0.79</td><td>0.81</td><td>0.74</td><td>0.76</td><td>0.83</td><td>0.77</td></tr></table>

(c) Human ratings on “advanced” prompts   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Count</td><td rowspan="2">Differ</td><td rowspan="2">Compare</td><td colspan="2">Logical</td><td rowspan="2">Avg</td></tr><tr><td>Negate</td><td>Universal</td></tr><tr><td colspan="7">Image models</td></tr><tr><td>SD v2.1</td><td>2.8</td><td>2.5</td><td>2.6</td><td>2.9</td><td>3.2</td><td>2.9</td></tr><tr><td>SD-XL Turbo</td><td>2.9</td><td>2.6</td><td>2.6</td><td>2.9</td><td>3.3</td><td>3.0</td></tr><tr><td>SD-XL</td><td>3.0</td><td>2.7</td><td>2.6</td><td>2.9</td><td>3.3</td><td>3.0</td></tr><tr><td>DeepFloyd-IF</td><td>3.2</td><td>2.9</td><td>2.9</td><td>2.9</td><td>3.5</td><td>3.1</td></tr><tr><td>Midjourney v6</td><td>3.4</td><td>3.2</td><td>3.2</td><td>3.0</td><td>3.7</td><td>3.4</td></tr><tr><td>DALL-E 3</td><td>3.6</td><td>3.5</td><td>3.4</td><td>3.0</td><td>3.8</td><td>3.4</td></tr><tr><td colspan="7">Video models</td></tr><tr><td>ModelScope</td><td>2.4</td><td>2.4</td><td>2.2</td><td>2.6</td><td>2.8</td><td>2.5</td></tr><tr><td>Floor33</td><td>2.7</td><td>2.7</td><td>2.5</td><td>2.8</td><td>3.2</td><td>2.8</td></tr><tr><td>Pika v1</td><td>2.7</td><td>2.7</td><td>2.6</td><td>2.9</td><td>3.3</td><td>2.9</td></tr><tr><td>Gen2</td><td>2.8</td><td>2.7</td><td>2.6</td><td>2.9</td><td>3.3</td><td>2.9</td></tr></table>

(d) VQAScores on “advanced” prompts   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Count</td><td rowspan="2">Differ</td><td rowspan="2">Compare</td><td colspan="2">Logical</td><td rowspan="2">Avg</td></tr><tr><td>Negate</td><td>Universal</td></tr><tr><td colspan="7">Image models</td></tr><tr><td>SD v2.1</td><td>0.66</td><td>0.64</td><td>0.65</td><td>0.51</td><td>0.63</td><td>0.60</td></tr><tr><td>SD-XL Turbo</td><td>0.71</td><td>0.68</td><td>0.69</td><td>0.52</td><td>0.66</td><td>0.63</td></tr><tr><td>SD-XL</td><td>0.72</td><td>0.70</td><td>0.69</td><td>0.50</td><td>0.67</td><td>0.63</td></tr><tr><td>DeepFloyd-IF</td><td>0.70</td><td>0.70</td><td>0.71</td><td>0.50</td><td>0.65</td><td>0.63</td></tr><tr><td>Midjourney v6</td><td>0.77</td><td>0.77</td><td>0.76</td><td>0.50</td><td>0.73</td><td>0.68</td></tr><tr><td>DALL-E 3</td><td>0.80</td><td>0.80</td><td>0.77</td><td>0.49</td><td>0.75</td><td>0.69</td></tr><tr><td colspan="7">Video models</td></tr><tr><td>ModelScope</td><td>0.58</td><td>0.61</td><td>0.57</td><td>0.52</td><td>0.52</td><td>0.55</td></tr><tr><td>Floor33</td><td>0.60</td><td>0.64</td><td>0.59</td><td>0.53</td><td>0.55</td><td>0.57</td></tr><tr><td>Pika v1</td><td>0.65</td><td>0.64</td><td>0.63</td><td>0.55</td><td>0.63</td><td>0.61</td></tr><tr><td>Gen2</td><td>0.69</td><td>0.69</td><td>0.64</td><td>0.54</td><td>0.58</td><td>0.62</td></tr></table>

(a) Pairwise Accuracy on “basic” skills   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Attribute</td><td rowspan="2">Scene</td><td colspan="3">Relation</td><td rowspan="2">Overall</td></tr><tr><td>Spatial</td><td>Action</td><td>Part</td></tr><tr><td>VQAScore (CLIP-FlanT5)</td><td>63.8</td><td>62.1</td><td>64.5</td><td>63.7</td><td>65.0</td><td>63.7</td></tr><tr><td>BLIP-VQA [24]</td><td>58.9</td><td>57.5</td><td>56.8</td><td>56.6</td><td>58.9</td><td>56.7</td></tr><tr><td>LLMScore [44]</td><td>54.6</td><td>56.2</td><td>55.4</td><td>57.2</td><td>54.7</td><td>55.4</td></tr><tr><td>VQAScore (InstructBLIP)</td><td>60.9</td><td>60.3</td><td>62.2</td><td>61.9</td><td>61.9</td><td>61.1</td></tr><tr><td>VQAScore (LLaVA-1.5)</td><td>59.4</td><td>57.5</td><td>59.2</td><td>59.8</td><td>60.3</td><td>59.0</td></tr><tr><td>Davidsonian [5]</td><td>53.5</td><td>52.9</td><td>55.1</td><td>53.7</td><td>55.3</td><td>54.2</td></tr><tr><td>VQ2 [79]</td><td>51.7</td><td>51.8</td><td>52.2</td><td>52.7</td><td>53.9</td><td>52.5</td></tr><tr><td>HPSv2 [76]</td><td>48.4</td><td>48.6</td><td>49.6</td><td>49.4</td><td>54.4</td><td>49.4</td></tr><tr><td>CLIPScore [20]</td><td>46.8</td><td>45.1</td><td>49.3</td><td>47.9</td><td>49.4</td><td>47.4</td></tr><tr><td>BLIPv2Score [34]</td><td>47.5</td><td>45.9</td><td>51.6</td><td>50.2</td><td>53.2</td><td>48.5</td></tr><tr><td>ImageReward [78]</td><td>55.8</td><td>55.1</td><td>58.3</td><td>57.2</td><td>59.3</td><td>56.3</td></tr><tr><td>PickScore [27]</td><td>57.7</td><td>57.9</td><td>57.2</td><td>58.4</td><td>59.3</td><td>58.2</td></tr><tr><td>VQAScore (GPT4-o)</td><td>69.3</td><td>68.4</td><td>69.3</td><td>68.1</td><td>69.7</td><td>69.2</td></tr></table>

(b) Pairwise Accuracy on “advanced” skills   
Table 7: Evaluating human correlation across skills on GenAI-Bench (Image). For a comprehensive comparison, we report the Pairwise Accuracy of different automated metrics across skills. Overall, VQAScore based on our in-house CLIP-FlanT5 or the proprietary GPT4-o surpasses other metrics across all skills by a large margin. Lastly, all automated metrics perform much worse on GenAI-Bench’s “advanced” prompts that require higher-order reasoning, suggesting room for improvement.   

<table><tr><td rowspan="2">Method</td><td rowspan="2">Count</td><td rowspan="2">Differ</td><td rowspan="2">Compare</td><td colspan="2">Logical</td><td rowspan="2">Overall</td></tr><tr><td>Negate</td><td>Universal</td></tr><tr><td>VQAScore (CLIP-FlanT5)</td><td>59.2</td><td>58.0</td><td>57.3</td><td>58.3</td><td>59.1</td><td>59.6</td></tr><tr><td>BLIP-VQA [24]</td><td>55.3</td><td>49.8</td><td>51.9</td><td>44.1</td><td>54.1</td><td>51.4</td></tr><tr><td>LLMScore [44]</td><td>54.6</td><td>55.4</td><td>56.2</td><td>57.2</td><td>54.7</td><td>55.4</td></tr><tr><td>VQAScore (InstructBLIP)</td><td>56.8</td><td>55.1</td><td>54.8</td><td>58.6</td><td>58.2</td><td>58.5</td></tr><tr><td>VQAScore (LLaVA-1.5)</td><td>58.1</td><td>54.8</td><td>53.0</td><td>57.3</td><td>59.6</td><td>58.9</td></tr><tr><td>Davidsonian [5]</td><td>53.5</td><td>55.1</td><td>52.9</td><td>53.7</td><td>55.3</td><td>54.2</td></tr><tr><td>VQ2 [79]</td><td>51.7</td><td>52.2</td><td>51.8</td><td>52.7</td><td>53.9</td><td>52.5</td></tr><tr><td>HPSv2 [76]</td><td>53.6</td><td>51.7</td><td>51.1</td><td>44.4</td><td>47.7</td><td>49.1</td></tr><tr><td>CLIPScore [20]</td><td>50.6</td><td>50.7</td><td>47.3</td><td>46.0</td><td>50.0</td><td>49.8</td></tr><tr><td>BLIPv2Score [34]</td><td>52.3</td><td>56.5</td><td>51.5</td><td>47.1</td><td>52.3</td><td>51.6</td></tr><tr><td>ImageReward [78]</td><td>56.3</td><td>55.3</td><td>56.0</td><td>46.6</td><td>52.5</td><td>53.2</td></tr><tr><td>PickScore [27]</td><td>56.4</td><td>54.0</td><td>53.9</td><td>47.9</td><td>52.4</td><td>53.7</td></tr><tr><td>VQAScore (GPT4-o)</td><td>64.0</td><td>62.3</td><td>63.0</td><td>64.0</td><td>68.3</td><td>66.2</td></tr></table>

# D Details of Baseline Methods

In this section, we detail the implementation of the baseline methods. Table 8 reports VQAScore performance on seven more benchmarks that measures correlation with human judgments.

CLIPScore and BLIPv2Score. To calculate CLIPScore, we use the CLIP-L-336 model [20]. To calculate BLIPv2Score, we use the ITMScore of BLIPv2-vit-G [34].

Metrics finetuned on human feedback (PickScore/ImageReward/HPSv2). We use the official code and model checkpoints to calculate these metrics. PickScore [27] and HPSv2 [76] finetune

Table 8: VQAScore on image-text alignment benchmarks. We show Group Score for Winoground and EqBen; AUROC for DrawBench, EditBench, and COCO-T2I; pairwise accuracy [12] for TIFA160 and GenAI-Bench; and binary accuracy for Pick-a-Pick, with higher scores indicating better performance for all metrics. VQAScore (based on CLIP-FlanT5) outperforms all prior art across all benchmarks.   

<table><tr><td>Method</td><td>Models</td><td>Winoground</td><td>EqBen</td><td>DrawBench</td><td>EditBench</td><td>COCO-T2I</td><td>TIFA160</td><td>Pick-a-Pic</td><td>GenAI-Bench</td></tr><tr><td colspan="10">Based on vision-language models</td></tr><tr><td>CLIPScore [20]</td><td>CLIP-L-14</td><td>7.8</td><td>25.0</td><td>49.1</td><td>60.6</td><td>63.7</td><td>54.1</td><td>76.0</td><td>50.8</td></tr><tr><td colspan="10">Finetuned on human feedback</td></tr><tr><td>PickScore [27]</td><td>CLIP-H-14 (finetuned)</td><td>6.8</td><td>23.6</td><td>72.3</td><td>64.3</td><td>61.5</td><td>59.4</td><td>70.0</td><td>56.2</td></tr><tr><td>ImageReward [78]</td><td>BLIPv2 (finetuned)</td><td>12.8</td><td>26.4</td><td>70.4</td><td>70.3</td><td>77.0</td><td>67.3</td><td>75.0</td><td>55.8</td></tr><tr><td>HPSv2 [76]</td><td>CLIP-H-14 (finetuned)</td><td>4.0</td><td>17.0</td><td>63.1</td><td>64.1</td><td>60.3</td><td>55.2</td><td>69.0</td><td>49.6</td></tr><tr><td colspan="10">OG/A methods</td></tr><tr><td>VQ2 [79]</td><td>FlanT5, LLaVA-1.5</td><td>10.0</td><td>20.0</td><td>52.8</td><td>52.8</td><td>47.7</td><td>48.7</td><td>73.0</td><td>51.9</td></tr><tr><td>Davidsonian [5]</td><td>ChatGPT, LLaVA-1.5</td><td>15.5</td><td>20.0</td><td>78.8</td><td>69.0</td><td>76.2</td><td>54.3</td><td>70.0</td><td>54.6</td></tr><tr><td colspan="10">VQAScore (ours) using open-source VQA models</td></tr><tr><td>VQAScore</td><td>InstructBLIP</td><td>28.5</td><td>38.6</td><td>82.6</td><td>75.7</td><td>83.0</td><td>70.1</td><td>83.0</td><td>62.3</td></tr><tr><td>VQAScore</td><td>LLaVA-1.5</td><td>29.8</td><td>35.0</td><td>82.2</td><td>70.6</td><td>79.4</td><td>66.4</td><td>76.0</td><td>61.7</td></tr><tr><td colspan="10">VQAScore (ours) using our VQA model</td></tr><tr><td>VQAScore</td><td>CLIP-FlanT5</td><td>46.0</td><td>47.9</td><td>85.3</td><td>77.0</td><td>85.0</td><td>71.2</td><td>84.0</td><td>64.1</td></tr></table>

the CLIP-H model, and ImageReward [78] finetunes the BLIPv2 [34], using costly human feedback from either random web users or expert annotators. These metrics use discriminative pre-trained VLMs, which bottleneck their performance due to bag-of-words encodings. Also, their finetuning datasets may lack compositional texts. Finally, human annotations can be noisy, especially when these annotators are not well trained (e.g., random web users of the Pick-a-pic dataset [27]).

QG/A methods (VQ2/Davidsonian). These divide-and-conquer methods are the most popular in recent text-to-visual evaluation [1, 24, 65, 74]. VQ2 [79] uses a finetuned FlanT5 to generate free-form QA pairs and computes the average score of P(answer | image, question). Davidsonian uses a more sophisticated pipeline by prompting ChatGPT to generate yes-or-no QA pairs while avoiding inconsistent questions. For example, given the text “the moon is over the cow”, if a VQA model already answers “No” to “Is there a cow?”, it then skips the follow-up question “Is the moon over the cow?”. However, these methods often generate nonsensical QA pairs, as shown in Table 9 on real-world user prompts from GenAI-Bench.

Table 9: Failure cases of divide-and-conquer methods (VQ2/Davidsonian). We show generated question-and-answer pairs of VQ2 and Davidsonian on three GenAI-Bench prompts. These methods often generate irrelevant or erroneous QA pairs (highlighted in red), especially with more compositional texts.   

<table><tr><td>Method</td><td>Generated questions</td><td>Candidate answers (correct answer choice in bold)</td></tr><tr><td>VQ2</td><td>Text: &quot;a snowy landscape with a cabin, but no smoke from the chimney&quot; 
What is the name of the landscape on which it&#x27;s a cabin? 
In this landscape what does the fire not go off?</td><td>a snowy landscape 
a cabin</td></tr><tr><td>Davidsonian</td><td>Is there a landscape? 
Is there no smoke from the chimney? 
Is the cabin in the landscape?</td><td>yes, no 
yes, no 
yes, no</td></tr><tr><td>VQ2</td><td>Text: &quot;six people wear white shirts and no people wear red shirts&quot; 
What does the average American wear? 
What kind of clothes do not all people wear?</td><td>white shirts 
red shirts</td></tr><tr><td>Davidsonian</td><td>Are there people? 
Are the shirts red? 
Are the shirts white?</td><td>yes, no 
yes, no 
yes, no</td></tr><tr><td>Text: &quot;in the classroom there are two boys standing together, the boy in the red jumper is taller than the boy in the white t-shirt&quot; 
VQ2</td><td>Where do two tall kids stand? 
Which color of jumper is the tallest?</td><td>jumper is taller than the boy in the white t-shirt&quot; 
the classroom 
the red jumper</td></tr><tr><td>Davidsonian</td><td>Is the boy in the red jumper wearing a red jumper? 
Is the boy in the white t-shirt wearing a white t-shirt? 
Are the boys standing together?</td><td>yes, no 
yes, no 
yes, no</td></tr></table>
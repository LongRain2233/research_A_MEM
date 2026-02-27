# PyVision: Agentic Vision with Dynamic Tooling

Shitian Zhao1,§,*, Haoquan Zhang1,3,*, Shaoheng Lin1,*, Ming Li1,*, Qilong Wu4,*, Kaipeng Zhang1,5,†, Chen Wei2,†

1Shanghai AI Lab, 2Rice University, 3CUHK, 4NUS, 5SII

![](images/f4933b33de9612f149ad83332968f27b8b4c42e7801e2a7dd04cd5769d9f5f0a.jpg)

Project Page

![](images/fdf65110afd5f13e9bcb6e8d4f6cb6a1d6b61c4e3ac37263c2cdd95957ceed38.jpg)

Inference Code

![](images/23bd9c0545ff950bc2819986caae8599b85999e4ddc476a40f69d643ab1c1062.jpg)

PyVision Demo

![](images/ba69c753f3cca05fde66c3c200705d25d62f8a48fa88f7abe7adb3d7ff0cbc1e.jpg)  
Figure 1 | Overcoming Illusory Heuristics with Code. This visual puzzle mimics the wellknown Ebbinghaus illusion [19], but with a twist: it reverses the typical size context, making the correct answer visually obvious to humans. Yet, a standard MLLM [35] mistakenly recalls the well-documented illusion template to answer “same size”. In contrast, PyVision behaves agentically, probing pixel values, segmenting objects, and computing the actual sizes via onthe-fly Python code to reach the correct answer. This example highlights how dynamic tooling enables adaptive, grounded, verifiable visual reasoning beyond superficial pattern matching.

# Abstract

LLMs are increasingly deployed as agents, systems capable of planning, reasoning, and dynamically calling external tools. However, in visual reasoning, prior approaches largely remain limited by predefined workflows and static toolsets. In this report, we present PyVision, an interactive, multi-turn framework that enables MLLMs to autonomously generate, execute, and refine Python-based tools tailored to the task at hand, unlocking flexible and interpretable problem-solving. We develop a taxonomy of the tools created by PyVision and analyze their usage across a diverse set of benchmarks. Quantitatively, PyVision achieves consistent performance gains, boosting GPT-4.1 by $+ 7 . 8 \%$ on $\mathrm { V } ^ { \ast }$ and Claude-4.0-Sonnet by $+ 3 1 . 1 \%$ o n VLMsAreBlind-mini. These results point to a broader shift: dynamic tooling allows models not just to use tools, but to invent them, advancing toward more agentic visual reasoning.

# 1. Introduction

The idea of AI agents, systems that can autonomously plan and execute tasks, is rapidly gaining traction in modern AI research. Large language models (LLMs), originally built for text generation, have quickly evolved into capable agents that can formulate plans, interact with environments, and call external tools or functions to solve complex problems with minimal human oversight [1, 4, 14, 16, 27–30, 32–34, 40]. But beyond simply using tools, the more profound leap lies in an agent’s ability to invent them, such as dynamically generating code snippets tailored to its task or environment. This capacity to create problem-solving tools on the fly is not just powerful, but foundational to intelligence. As Benjamin Franklin remarked, “Man is a tool-making animal”.

Interestingly, the idea of using external computational modules for complex reasoning is not new, particularly in the vision domain. Early works such as Neural Module Networks [2] introduced a parser that orchestrated a set of predefined functions, embracing a neuro-symbolic approach to visual reasoning. This line of work inspired a series of influential successors (Tab. 1). Unlike end-to-end models, these systems explicitly represent each reasoning step and producing transparent and inspectable intermediate outputs, offering a promising path for tackling complex and compositional visual reasoning.

However, prior works typically rely on predefined workflows and static toolsets within single-turn frameworks, limiting the flexibility, creativity, and adaptability that modern LLM agents can achieve through dynamic tooling. With the growing coding and reasoning capabilities of today’s MLLMs, we can now move beyond these constraints in visual reasoning: models can dynamically generate code snippets in a multi-turn setup, building tools on the fly that are tailored to the task at hand.

Recent developments like OpenAI’s “Thinking with Images” [37] highlight this potential, but they offer limited visibility into how this process actually works. In this report, we present and analyze how advanced MLLMs with strong coding abilities, in our case, GPT-4.1 [35] and Claude-4.0-Sonnet [3], can dynamically create and leverage Python-based visual tools. We introduce PyVision, an interactive framework in which the model autonomously generates, executes, and iteratively refines Python code in response to multimodal user queries. To support this dynamic tooling loop, we build on Python’s rich ecosystem of mature libraries and carefully engineer both the system prompts and the runtime environment to enable seamless, multi-turn interaction between the MLLM and Python interpreter.

Table 1 | Comparison between PyVision and previous tool-using methods for visual reasoning.   

<table><tr><td>Methods</td><td>Dynamic Workflow</td><td>Dynamic Tool Generation</td><td>Multi-Turn Framework</td></tr><tr><td>NMN [2]</td><td>×</td><td>×</td><td>×</td></tr><tr><td>IEP [20]</td><td>×</td><td>×</td><td>×</td></tr><tr><td>VisProg [12]</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Visual ChatGPT [51]</td><td>×</td><td>×</td><td>✓</td></tr><tr><td>ViperGPT [47]</td><td>×</td><td>×</td><td>×</td></tr><tr><td>MM-REACT [56]</td><td>×</td><td>×</td><td>×</td></tr><tr><td>HuggingGPT [42]</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Image-of-Thought [61]</td><td>×</td><td>×</td><td>×</td></tr><tr><td>Visual Sketchpad [17]</td><td>✓</td><td>×</td><td>✓</td></tr><tr><td>VAT [24]</td><td>×</td><td>×</td><td>×</td></tr><tr><td>PyVision</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

We then analyze the tools generated by PyVision in depth. To do so, we construct a taxonomy that classifies the tools into four broad categories: basic image processing, advanced image processing, visual prompting and sketching, and numerical and statistical analysis, alongside a long tail of creative, task-specific operations (Fig. 1). This framework enables us to examine how different benchmarks and domains elicit distinct patterns of tool usage. For instance, perception-heavy tasks often trigger operations like cropping and contrast enhancement, while math and logic benchmarks rely more on visual sketching and numerical analysis. These findings highlight the power of dynamic tool generation: it equips the model with the flexibility to adapt its strategy to the unique demands of each task and domain.

Results across major benchmarks reveal that PyVision consistently improves the performance of strong backend models. Notable improvements include a $+ 7 . 8 \%$ boost on $\mathrm { V } ^ { \ast }$ [52] with PyVision-GPT-4.1, an $+ 8 . 3 \%$ gain on Visual Puzzles [43], and a dramatic leap on VLMsAreBlindmini [41], where PyVision-Claude-4.0-Sonnet improves from $4 8 . 1 \%$ to $7 9 . 2 \%$ , marking a remarkable $+ 3 1 . 1 \%$ increase. Our results suggest that PyVision acts as an amplifier of the backend model’s innate strengths: gaining more at perception tasks when paired with perceptually strong models like GPT-4.1, and at abstract reasoning when paired with Claude-4.0-Sonnet. In short, dynamic tooling does not override model capabilities. It unlocks them.

Ultimately, the agentic PyVision with dynamic tooling not only provides practical performance benefits, it also signals a broader shift in multimodal reasoning. By empowering models to invent new computational tools on the fly, we move closer to versatile, autonomous, and genuinely creative AI systems capable of adapting in real-world visual reasoning scenarios.

# 2. PyVision

We propose PyVision, an interactive, multi-turn framework for multimodal reasoning. PyVision empowers an MLLM with the ability to dynamically generate and execute Python code during inference. In each session, the MLLM receives an input, generates Python code in response, and executes it within an isolated Python runtime. The resulting output—textual, visual, or both—is fed back into the MLLM’s context, allowing it to iterate and refine its reasoning over multiple turns until it produces a final answer.

Unlike prior approaches that rely on a fixed toolset, such as detection [25] or segmentation [21] models, PyVision provides only Python as building blocks for tools. This design leverages Python’s rich ecosystem of scientific and vision libraries, for example, OpenCV [7], Pillow [8], NumPy [13], Pandas [31], Scikit-learn [38], and Scikit-image [48]. With access to such a versatile ecosystem, the model can generate highly adaptive tools tailored to diverse tasks.

![](images/df30cff2bb39dfcccb51e715df5c438ae7dfe46457504f3c17ffeaf826bbe860.jpg)  
Figure 2 | PyVision, an interactive and multi-turn framework capable of dynamic tool generation, designed for multimodal reasoning. In an inference session, PyVision performs $\tt { n } { + } 1$ interaction turns with the Python interpreter. In the figure, code_block_i refers to the generated Python code by the MLLM in the i-th turn, and mm_clue_i the executed multi-modal outputs by the Python interpreter. This loop continues until the MLLM outputs a final answer.

System Prompt Design. To guide the MLLM’s reasoning and code generation, PyVision uses a carefully constructed system prompt in addition to user queries. The system prompts encode operational instructions that specify how to access input images, structure code, and return final answers. Key components include:

• Encouraging the MLLM to generate code to solve the task.   
• Input images or video frames are pre-loaded as variables named image_clue_i, where i denotes the image index. This allows the model to reference the images without additional loading code. We also provide image resolution that helps operations like cropping.   
• Output from the code is expected via specific functions: print() for textual results and plt.show() for image visualizations.   
• Each generated code block is wrapped in a <code> tag to enable reliable parsing.   
• Final answers are enclosed in a <answer> tag for consistent evaluation.

With this design, the two MLLMs we experiment with, GPT-4.1 [35] and Claude-4.0- Sonnet [3], can reliably generate parsable and executable code blocks that rarely crash. The full system prompt is included in appendix A.

Multi-Turn Interaction between Runtime and the MLLM. As illustrated in Fig. 2, PyVision operates as a multi-turn agentic loop between the MLLM and an isolated Python runtime. In the i-th turn, the MLLM generates a code block code_block_i, which is executed to produce multimodal results mm_clue_i. These results are appended to the MLLM’s context, enabling it to update its reasoning in the next turn. This loop continues until the MLLM automatically decides to output a final boxed answer.

To support robust and effective multi-turn interaction between the MLLM and the runtime environment of Python, PyVision incorporates several design principles:

• Process isolation: Each code snippet is executed in a subprocess dynamically spawned by the main process, ensuring that crashes or side effects in one execution do not impact the overall inference session.   
• Cross-turn persistence: The runtime environment retains variables and state across turns. This allows the model to reuse or modify intermediate Python code execution results in previous turns, e.g., first cropping an image, then applying filters, and finally computing geometric features to complete a task.   
• File-system safe I/O: Communication between the runtime and the MLLM is handled through structured variable passing [9, 11, 53], guided by system prompts. This avoids direct dependencies on the host file system.

Together, these mechanisms enable PyVision to serve as a flexible, secure, and powerful platform for dynamic tool generation in multi-modal reasoning tasks.

# 3. Dynamically Generated Tools

Examples in Different Tasks and Domains. We start our analysis by presenting examples of PyVision across diverse tasks and domains in Figs. 4 to 8. These examples illustrate how PyVision autonomously creates task-specific and domain-specific tools tailored to each unique challenge, emerging voluntarily from PyVision’s multi-turn code generation and execution.

# 3.1. Tooling Taxonomy

To better understand the types of tools generated by PyVision, we construct a taxonomy based on the code it produces across various tasks and domains (Sec. 4). Specifically, we collect the generated code snippets from inference sessions, embed them using text-embedding-3-large [36] via OpenAI’s API, and cluster the embeddings to identify emergent tool categories. By inspecting and interpreting the resulting clusters, we identify four major classes of tools: (1) basic image processing, (2) advanced image processing, (3) visual prompting and sketching, (4) numerical and statistical analysis, and (5) long-tailed operations. We detail each below.

Basic Image Processing. These tools serve as the foundation for visual manipulation and perception. They enable the model to clean, align, and highlight image content in ways that improve downstream reasoning.

Cropping: For high-resolution or cluttered inputs, PyVision often crops and zooms into regions of interest. By selecting coordinates through reasoning, it effectively performs soft object detection, focusing attention where it matters most. (Fig. 3)   
• Rotation: Misaligned images (e.g., rotated maps, skewed documents) can confuse even strong models. PyVision rotates inputs to canonical orientations, making text, spatial layouts, or directional cues easier to interpret.   
• Enhancement: In visually subtle domains like medical imaging, PyVision applies contrast adjustments and other enhancements to make latent structures more salient. (Fig. 4)

Advanced Image Processing. These tools reflect PyVision’s ability to perform mid to highlevel vision tasks, but designed and executed dynamically, on demand.

Segmentation: By isolating specific regions via thresholding or edge detection, PyVision can extract foreground objects from background noise.   
• Detection: PyVision generates bounding boxes or edge detection to localize objects in the scene. This supports follow-up operations like counting or measuring. (Fig. 5)   
OCR: Without relying on external APIs, PyVision extract textual content (e.g., signage, labels) by itself, enabling hybrid visual-linguistic reasoning. (Fig. 3)

Visual Prompting and Sketching. In some tasks, it is not enough to perceive the image—the model must “think visually” [5, 15, 54, 59]. To help itself reason, PyVision annotates the image with auxiliary markings, essentially creating visual notes or sketches.

• Rendering Marks: In object counting or enumeration task, PyVision often marks items with dots or symbols. This external memory acts as a tallying aid, helping it keep track of what’s been counted. (Fig. 6)   
• Rendering Lines: In geometric or spatial tasks (e.g., mazes), PyVision draws auxiliary lines to assist reasoning, such as showing the moving directions in a maze.

Numerical and Statistical Analysis. To go beyond perception and into interpretation, PyVision invokes tools for quantitative reasoning over visual inputs.

Image Histogram: By plotting pixel intensity distributions, PyVision can analyze lighting, contrast, and more, critical for domains where histogram carry meaning. (Fig. 4)   
• Numerical Analysis: When solving visual math problems or compare quantities, PyVision writes scripts to compute areas, lengths, or other metrics for symbolic reasoning. (Fig. 5)

![](images/646e8cd602e1f4af0786b1ab2ab477646301c76ba2956d3a116b68cdd265cb7a.jpg)

![](images/4bc29b54c0369101ff35b720390ecfb6ba7127a2f511fc38fc650ff7e0747667.jpg)

![](images/741297f1f5a14f2a92e8f5fdf96b8c3cf800863f5d92f7e84e11ca052b9b100a.jpg)  
Figure 3 | Case Study: Visual Search requires language-based world knowledge, vision-based contextual understanding, and iterative refinement to precisely locate specific visual elements. In this case from $\mathrm { V } ^ { \ast }$ [52], the agent must identify text on a small advertising board in a complex outdoor scene. The target occupies a minimal image area, requiring adaptive cropping. This showcases how dynamic tool generation enables flexible spatial reasoning.

The agent generates custom Python codes across three iterative turns, refining coordinatebased cropping tools based on visual feedback, and eventually isolating the advertising board required by the user query. The agent then performs OCR and extracts key text such as “YOGA" and correctly identifies the business as a yoga studio.

Long-Tail Operations. PyVision also invents novel tools not easily classified. These one-off operations showcase its ability to reason creatively under novel constraints. For example, in Fig. 7, to solve a “spot the difference” task, PyVision directly subtracts pixel values between two images and visualizes the result. This kind of zero-shot problem decomposition and tool synthesis reflects both the power and flexibility of dynamic tooling for visual reasoning.

Video Reasoning with Agentic Tooling. Video understanding poses unique challenges compared to static image tasks. PyVision demonstrates strong potential in this setting by treating video not as a monolithic input but as a sequence of decision points. In Fig. 8, rather than exhaustively analyzing all frames, PyVision dynamically selects and processes only those frames containing distinct types of tables. It then extracts visual evidence and support reasoning. This agentic, multi-step workflow enables PyVision to operate more like a human analyst: skimming, sampling, and refining its understanding based on intermediate results.

# User Query

What is the specific type of abnormality shown in this image? A.The abnormality shown in this image is a blockage in the blood vessels. B.There are no specific abnormalities observed in this image. C.The abnormality shown in this image is an infection. D.The abnormality shown in this image is a blood clot.

![](images/95ff6a0410a489cb089bc60c1015d07b12559d36ed6746ac1b7e974c49fa91f3.jpg)  
System Prompt

![](images/742f94f1c2eabc2899007d7a5f18244e3110ed96efe60f4d4ff61805667315e1.jpg)

# 1

Let's proceed step by step: Step 1: Understand the Question and Image - We're given a fundus (retina) image of the eye. - The question asks for the specific type of abnormality shown in the image, with four options.

Step 2: Plan the Approach - Analyze the image for any signs of abnormality: - Blockage in blood vessels (look for vessel occlusions, whitening, or dropout) - Infection (look for exudates, hemorrhages, or unusual patterns) - Blood clot (look for localized hemorrhages or vascula occlusion) - No abnormalities (normal optic disc, vessels, and macula) - Use matplotlib to enhance contrast and zoom into key areas (optic disc, macula blood vessels).

Step 3: Visualize the Image - Display the original image and enhance contrast to check for subtle abnormalities. Let's write code to visualize and enhance the image to better inspect for abnormalities.

# First Turn - Reasoning, Coding & Execution

import matplotlib.pyplot as plt import numpy as np

$$
\# \text {S n o w} \quad \text {c h i n e a r i g h t m a p l . f i g u r e (f i g s i z e = (8 , 8))}
$$

$$
\begin{array}{l} \text {p l t . i m s h o w (i m a g e _ {c l u e} 0)} \\ \text {p l t . t h i t e ("O r i g i n a l F u n d u a l I m a g e")} \end{array}
$$

$$
\begin{array}{c} \text {p l t . a x i s (} \\ \text {p l t . s h o l (} \end{array}
$$

$$
\# \text {E n h a n c e} \# \text {e q u a l i z a t i o n (o n e c h a n n e l)}
$$

$$
\begin{array}{l} \text {d e f} \\ \text {a r r o r} \end{array}
$$

$$
\text {e n h a n e d} = \mathrm {n p . z e r o s l i k e} (\text {a r r})
$$

$$
\text {f o r} \mathrm {I} \text {i n} \text {r a n g e} (\mathrm {S}): \# \text {f o r} \mathrm {R}, \mathrm {G}, \mathrm {B} \text {c h a n n e l s}
$$

$$
\text {h i s t , b i n s} = \text {n p . h i s t g r a p t (} \text {c h a n n e l . f l a t t e n (} \text {, 2 5 6 ,}
$$

$$
\begin{array}{l} \text {c d f = h i s t . c u m s u m ()} \\ \text {c d f n o r m a l i z e d = (c d f + h i s t . m a x) /} \end{array}
$$

$$
\begin{array}{l} \text {c d f _ {m} = n p . m a . m a x e d , e q u a l , (c d f , 0)} \\ \text {c d f _ {m} = n p . m a . m a x e d , e q u a l , (c d f , 0)} \end{array}
$$

$$
\operatorname {c a f} _ {\text {m}} = \left(\frac {\operatorname {c a f} _ {\text {m}} - \operatorname {c a f} _ {\text {m . m i n}})}{\operatorname {c d f} _ {\text {m . m a x}} (\operatorname {- c d f} _ {\text {m . m i n}})}\right) ^ {* 2 5 7}
$$

$$
\begin{array}{l} \text {c d f} = (\text {n p . m a . f i l l e d (c d f} _ {\text {m}} 0), \text {a s t y p e} (\text {u n t 8})) \\ \text {e n h a n c e d [ , . . . , ] = c d f [ c h a n n e l ]} \end{array}
$$

$$
\begin{array}{c} \text {r e t u r n e n h a n c e d} \\ \text {e n s h o r e d i n m u l t e r} \end{array}
$$

$$
\text {p l t . f i g u r e (f i g s i z e = (8 , 8))}
$$

$$
\begin{array}{c} \text {p l t . i m s h o w (e n h a n c e d . i m g)} \\ \text {p l t . t i t l e} (\text {C o n t r a s t . E n h a n c e d}) \end{array}
$$

$$
\begin{array}{c} \text {p l . a x i s t (e f f)} \\ \text {p l . t i x . a x i s t (o f f ^ {\prime})} \end{array}
$$

$$
p i t. s h o w ()
$$

![](images/fab577c8f5046a8cdd51e86e82b79b88e4d848a34c1b5e5569e18b87ed7ea0cb.jpg)

![](images/28fe67f345cf916aef8fe23a8ddfb0123925cd13bdb172c4eec82f5cb92c0534.jpg)

# 2

# Second Turn - Reasoning, Coding & Execution

# Reflection:

- The original and contrast-enhanced images show a clear view of the retinal

blood vessels, optic disc (left), and macula (center).

- There are no visible signs of kage), he hage, o exudates.

- The vessels are continuous, with no abrupt cut-offs, whitening, or signs of

- The optic disc appears normal, with clear margins.

- The macula appears normal, without edema or hemorrhage.

- To be thorough, let's check the intensity histogram to see if there are any

subtle abnormalities (e.g., localized bright or dark spots that may have been

missed visually).

# Analyze the intensity histogram

# to look for abnormalities

np.asarray(image_clue_0), axis=2 plt.figure(figsize=(6,4))

plt.hist(gray_img.flatten(),

bins=50, color='gray')

"Histogram")

plt.xlabel("Intensity") plt.ylabel("Pixel Count")

plt.show()

![](images/5dabffa24bb85a992c51e6cc49102a7b20d82b4dd3f6554f4690c5f5afed5e94.jpg)

# T3

# hird Turm - Reasoning (Answer)

# Reflection on Histogram:

- The intensity histogram shows the majority of pixel values

are within a normal range for a fundus image. - There are no abnormal peaks or distributions suggesting abnormal bright or dark regions (e.g., large hemorrhages or exudates).

# Summary of Findings:

- Vessel structure is continuous and normal.   
- No visible blood clots, blockages, or infection signs.   
- No abnormal intensity regions. - The image appears to be a healthy retina.

# Conclusion:

- The correct answer is: B. There are no specific abnormalities observed in this image.

Figure 4 | Case Study: Medical Imaging Analysis often requires high visual sensitivity and domain-specific reasoning to solve subtle medical problems. This case from OmniMedVQA [18] involves identifying a specific abnormality in a fundus image of the retina. PyVision is tasked with diagnosing the image without prior domain-specific modules.

To begin, PyVision generates a visualization tool that applies histogram equalization to enhance contrast in key regions, a standard technique in medical imaging analysis. It then creates a second tool to plot the intensity histogram, confirming the absence of abnormal peaks. Within a few iterations, PyVision constructs a custom diagnostic pipeline from scratch, correctly concluding that no specific abnormalities are present. This highlights PyVision’s ability to generate interpretable, domain-adapted tools for complex medical tasks.

![](images/06d60160f49e757fb6285a7a05ae2ac90294eb61c8dec8f9a52f6bd0038ab997.jpg)  
Figure 5 | Case Study: Symbolic Visual Puzzles require precise spatial reasoning and the recognition of overlapping or closely positioned geometric primitives, a task that is simple for humans, yet surprisingly difficult for powerful MLLMs when asked directly [41]. In contrast, the regular structure of such puzzles makes them well-suited to code-based solutions.

This is a nested squares counting task from VLMsAreBlind [41]. First, PyVision applies edge detection using skimage.measure.find_contours and identifies ten contours. Then, it infers that the number of contours corresponds to five nested squares, as each square contributes an inner and outer edge. To validate this, the model performs numerical analysis and prints the sorted perimeters of the detected contours. In the third stage, it performs a double-check and confidently confirms the correct answer: five nested squares.

![](images/8b9215f12e87a7aac534733f802c370f621bfc025823a2bd16f2458617800524.jpg)  
Figure 6 | Case Study: Visual Sketching is a valuable strategy for humans to solve scientific problems, including those in mathematics and physics. It can also enhance AI model performance by enabling precise numerical calculations and visual reasoning [17].

In this example from MathVision [49], PyVision is asked to compute how many people can sit around a row of four tables. The model first sketches the table arrangement, rendering marks to represent four connected tables and people sitting around them. From the sketch, it infers that ten people can be seated. In a second turn, PyVision generates a schematic diagram to explain and validate the seating logic, identifying positions that are unsuitable for seating.

![](images/80e4fb83f23ec871999c6e5e73b412e61cf010e4d32c6f34bb1fcfaa28547b44.jpg)

![](images/9db30be2ae7ac48a9a91cfb9469a0cd3b6073559e465c03451a542e6624b2b3e.jpg)  
Figure 7 | Case Study: Spot-the-Difference showcases structured visual comparison. Given a side-by-side image pair, PyVision is asked to identify all visual discrepancies.

PyVision first plans a multi-step strategy: it splits the image into left and right halves, computes the absolute pixel-level difference, and generates a difference map to highlight changes. It then displays both original images alongside the computed difference visualization to aid analysis. Based on this, PyVision proceeds to enumerate the identified differences. Although the final answers are not completely correct, the model’s initiative to employ pixel-level differencing and organize a reasoning pipeline is notable. This example illustrates both the creative potential of agentic visual reasoning and the ongoing challenge of mitigating hallucinations.1

![](images/c38f1cfd3cf36fd6f010326ff908bb2775a74e91bf6bf37144bbc703fc1e7c33.jpg)  
Figure 8 | Case Study: Video Understanding benefits from an agentic pipeline that integrates reasoning across frames and targeted frame selection. In this example from VSI-Bench [55], a 3-minute egocentric video of an indoor scene is presented, and PyVision is tasked with identifying the number of tables in the room.

PyVision begins by analyzing the video to detect candidate frames containing tables. It then selects and displays key frames, each showing a different table, including dining table, desk, coffee table, and side table, to support its reasoning. By synthesizing visual evidence and textual inference across multiple views, PyVision concludes there are four distinct tables in the room.

![](images/bc06cfd71b0327542d4f7d20cc6717a6b0269fb43e18447a8aa37cb10380ba54.jpg)  
MathVision

Basic Image Process: 250 (26.0%)   
crop: 238 (95.2%)   
□ rotate: 10 (4.0%)   
■ enhance: 2 (0.8%)   
OAdvanced Image Process: 43 (4.5%)   
1 Lsegmentation:29 (67.4%)   
detection: 8 (18.6%)   
Locr: 6 (14.0%)   
Add Visual Prompts: 122 (12.7%)   
L render_marks: 8   
Lrender_lines: 34 (27.9%)   
Numerical and Statistical Analysis: 291 (30.2%)   
■ L numerical analvsis: 291 (100.0%)   
Others: 256 (26.6%)   
L long-tail operations: 256 (100.0%)

![](images/c2334ac1f2646feb4c38af8b9330d66d0b5cbc5bd49bda8616e900b7bd4eb858.jpg)  
MathVista

Basic Image Process: 565 (20.2%)   
广 Lcrop:540 (95.6%)  
rotate: 6 (1.1%)   
Lenhance: 19 (3.4%)   
Advanced Image Process: 336 (12.0%)   
L segmentation: 110 (32.7%)   
1 detection: 198 (58.9%)   
ocr: 28 (8.3%)   
OAdd Visual Prompts: 231 (8.3%)   
Lrender_marks: 146 (63.2%)   
Lrender_lines: 85 (36.8%)   
ONumerical and Statistical Analysis: 964 (34.5%)   
histogram: 20 (2.1%   
L numerical analvsis: 944 (97.9%)   
Others: 695 (24.9%)   
■long-tailoperations: 695 (100.0%)

![](images/e87f3d15fd0391d5bf909ffd5b808c7d2546240f43a44be5f6fb0c8e8a7fde06.jpg)  
MMMU

. Basic Image Process: 403 (16.5%)   
□ crop: 333 (82.6%)   
□ L enhance: 70 (17.4%)   
Advanced Image Process: 41 (1.7%)   
1 L segmentation: 4 (9.8%)   
1 Ldetection:6 (14   
■ ocr: 31 (75.6%)   
Add Visual Prompts: 52 (2.1%)   
Lrender_marks:40 (76.9%)   
L render_lines: 12 (23.1%)   
Numerical and Statistical Analvsis: 1207 (49 5%)   
histogram: 16 (1.3%)   
■ L numerical_analysis: 1191 (98.7%)   
Others: 733 (30.1%)   
L long-tail operations: 733 (100.0%)

![](images/22344291dcad7980ab29c4495265682b5407352b71c38ab5ebae46c82c4202fd.jpg)  
VisualPuzzles

Basic Image Process: 1850 (44.1%)   
1 crop: 1795 (97.0%)   
Lrotate: 25 (1.4%)   
enhance: 30 (1.6%)   
Advanced Image Process: 346 (8.3%   
1 L segmentation: 225 (65.0%)   
1 detection: 99 (28.6%)   
1 gcr:22 (6.4%)   
Add Visual Prompts: 362 (8.6%)   
Lrender marks: 206 (56 .9%)   
1 Lrender_lines: 156 (43.1%)   
Numerical and Statistical Analvsis: 853 (20.3%)   
Lhistogram: 8 (0.9%)   
Lnumerical analvsis: 845 (99.1%) nalv   
Others: 782 (18.7%)   
L long-tail operations: 782 (100.0%) (100.0%

![](images/a735071dc17f94528d1b7d2d60d81cbe9de1fbb8a0cc489e444fc0d80e150386.jpg)  
VLMsareblind

OBasic Image Process: 499 (7.9%)   
crop: 423 (84.8%)   
enhance: 76 (15.2%)   
Advanced Image Process: 2583 (40.7%)   
1 L segmentation: 1535 (59 4%)   
1 detection: 1018   
Locr: 30 (1.2%)   
Add Visual Prompts: 204 (3.2%)   
Lrender marks: 155 (76.0%)   
Lrender lines: 49 (24.0%)   
Numerical and Statistical Analvsis: 1918 (30 2%)   
histogram: 79 (4.1%)   
Lnumerical analvsis: 1839 (95.9%)   
Others: 1150 (18.1%)   
L long-tail operations: 1150 (100.0%)

![](images/b88c0e65096101cf212d15d3ff6ea8d32845a8fa315d278ddfc2b00410e92eb2.jpg)  
V*

Basic Image Process: 801 (83.4%)   
1 crop: 799 (99.8%)   
1 Lenhance: 2 (0.2%)   
Advanced Image Process: 6 (0.6%)   
1 segmentation:6 (100.0%)   
Add Visual Prompts: 18 (1.9%)   
1 Lrender_marks:18(100.0%）  
Numerical and Statistical Analysis: 53 (5.5%)   
Lbie 3./5.7%   
- histogram: 3 (5.7% busiey50/04.3%/.)   
numerical ana   
Others: 82 (8.5%) L long-tail operations: 82 (100.0%)

![](images/731e455927633762e6ae9b3196083d9fd53afa1932bd80dd13811e18798871af.jpg)  
Medical

. Basic Image Process: 1174 (51.0%)   
1 crop: 375 (31.9%)   
Lenhance: 799 (68.1%)   
Advanced Image Process: 91 (3.9%)   
Lsegmentation:82 (90.1%)   
detection: 9 (9.9%)   
Add Visual Prompts: 18 (0.8%)   
render marks:18 (100.0%)   
Numerical and Statistical Analvsis: 457 (19.8%)   
histogram: 147 (32.2%)   
二 numerical_analysis:310 (67.8%)   
Others:564 (24.5%)   
Llong-tail operations: 564 (100.0%)

Remote Sensing

![](images/5ef7e4c844db83e852da871de8962d1186563c31d0d4c5c7613a6fed1ae519c5.jpg)  
Figure 9 | Taxonomy Distribution Across Benchmarks and Domains. Tool usage varies significantly across different tasks and domains.

Racic Imago Brococs: 3.065 /31 0% )   
1 Lcrop:1260 (64.1%)   
Lenhance:705 (35.9%   
Advapced Image Process: 1296 (21.1%)   
1 L segmentation: 1153 (89.0%)   
1 Ldetection: 143 (11.0%)   
O.Add Visual Prompts: 480 (7.8%)   
1 L render marks: 438 (91.2%)   
Lrender lines: 42 (8.8%）  
Numerical and Statistical Analvsis: 639 (10.4%)   
Lhistogram: 296 (46.3%)   
Lnumerical analvsis: 343 (53.7%)   
Others: 1772 (28.8%)   
− long-tail operations: 1772 (100.0%)

For math- and logic-related benchmarks, e.g., MathVision [49], MathVista [26], MMMU [57], VisualPuzzles [43], numerical and statistical tools constitute a major portion of the usage and visual prompts are used relatively more often. In the symbolic vision task VLMsAreBlind [41], advanced image processing tools dominate. For visual search in $\mathrm { V } ^ { \ast }$ [52], PyVision primarily relies on cropping to facilitate detailed visual querying, which takes over $8 3 \%$ of all tools used.

Tooling preferences are also domain-sensitive: On medical images [18], contrast-enhancement tools are frequently invoked. In remote sensing [23], segmentation tools are more common.

These observations highlight the importance of flexible and dynamic tooling to support the diverse demands of real-world vision tasks.

# 3.2. Analyzing Tooling Patterns Across Tasks and Domains

Benchmarks. To evaluate the effectiveness of PyVision on versatile benchmarks and domains, we select six benchmarks. The details are listed as follows:

• Multi-Modal Math: MathVista [26] and MathVision [49] challenge models with math problems that combine visual perception and numerical reasoning.   
• Domain and Logic Reasoning: MMMU [57] tests subject-specific reasoning across disciplines using multi-modal input, often requiring college-level knowledge. VisualPuzzles [43] focuses on logic, with tasks covering algorithmic, analogical, deductive, inductive, and spatial reasoning, minimizing domain dependency while maximizing abstraction.   
• Symbolic Vision: VLMs Are Blind [41] consists of designed symbolic visual puzzles, probing the limits of parsing and reasoning over abstract, structured visual primitives.   
• Fine-Grained Visual Search: $\mathrm { V } ^ { \ast }$ [52] features 191 high-resolution samples that require pinpointing subtle visual details based on nuanced queries, making it a strong testbed for attention and spatial reasoning.

We also evaluate two special domains, Medical Imaging VQA [18] and Remote Sensing VQA [23] to probe the tooling patterns in different domains.

Distribution of Tools. To understand how PyVision adapts its tooling to different problems, we analyze the distribution of tool categories across benchmarks and domains in Fig. 9.

The results reveal strong task- and domain-specific preferences. In math and logic-heavy benchmarks like MathVista [26], MathVision [49], MMMU [57], and VisualPuzzles [43], PyVision frequently generates numerical and statistical tools to support symbolic and quantitative reasoning. These are often accompanied by visual prompting and sketching that help ground abstract logic in visual cues. In symbolic visual tasks such as VLMsAreBlind [41], advanced image processing tools are predominant, reflecting the need for structure extraction and visual parsing. For fine-grained visual search tasks like $\mathrm { V } ^ { \ast }$ [52], cropping overwhelmingly dominates, accounting for over $8 3 \%$ of all tools, as the model focuses attention on localized regions.

Domain also plays a significant role: on medical images [18], contrast enhancement is commonly used to reveal subtle visual patterns, while in remote sensing [23], segmentation tools help delineate objects in large-scale scenes.

These results underscore the importance of dynamic tool generation, allowing the model to flexibly tailor its strategy to the task at hand.

# 4. Results on Versatile Benchmarks

Baselines. To evaluate PyVision’s effectiveness on diverse multi-modal scenarios, we test it on versatile benchmarks with MLLMs including GPT-4.1 [35] and Claude-4.0-Sonnet [3] as the backend. We use plain chain-of-thought prompting [22, 50] as our baseline. The inference parameter settings and the prompt details are in appendix A.

Results. Tab. 2 highlights how adding PyVision’s dynamic tooling consistently boosts two strong back-end models across a diverse benchmark suite. For GPT-4.1, PyVision yields uniform gains on every dataset, from modest improvements on math-centric tasks: $+ 1 . 8 \%$ on MathVista and $+ 2 . 4 \%$ on MMMU, to a sizeable $+ 7 . 8 \%$ on the fine-grained visual-search benchmark $\mathrm { V } ^ { \ast }$ . Claude-4.0-Sonnet shows a sharper pattern: while math and general-reasoning tasks improve by roughly $+ 3 \%$ to $+ 5 \%$ , symbolic-vision performance on VLMsAreBlind-mini soars by $+ 3 1 . 1 \%$ . In short, dynamic tool generation delivers broad, task-dependent gains, which also depends on the backend model’s capability, discussed next.

Table 2 | Performance on six benchmarks. Improvements over each base model appear beneath the scores. We highlight a $+ 7 . 8 \%$ gain on $\mathrm { V } ^ { \ast }$ by PyVision-GPT-4.1, $+ 8 . 3 \%$ on VisualPuzzles and $+ 3 1 . 1 \%$ on VLMsAreBlind-mini by PyVision-Claude. *GPT-4.1 results are self-collected with plain chain-of-though prompting (appendix A.2) in June 2025.   

<table><tr><td></td><td>MathVista</td><td>MathVision-mini</td><td>MMMU</td><td>VisualPuzzles</td><td>VLMsAreBlind-mini</td><td>V*</td></tr><tr><td>GPT-4o</td><td>61.4</td><td>-</td><td>68.7</td><td>41.1</td><td>-</td><td>73.9</td></tr><tr><td>o1</td><td>71.8</td><td>-</td><td>77.6</td><td>51.8</td><td>-</td><td>69.7</td></tr><tr><td>o3</td><td>86.8</td><td>-</td><td>82.9</td><td>54.0</td><td>-</td><td>95.7</td></tr><tr><td>GPT-4.1</td><td>69.9*</td><td>46.4</td><td>71.9*</td><td>44.9</td><td>67.1</td><td>68.1</td></tr><tr><td rowspan="2">PyVision-GPT-4.1</td><td>71.7</td><td>48.7</td><td>74.3</td><td>47.4</td><td>69.7</td><td>75.9</td></tr><tr><td>+1.8</td><td>+2.3</td><td>+2.4</td><td>+2.5</td><td>+2.6</td><td>+7.8</td></tr><tr><td>Claude-4.0-Sonnet</td><td>71.4</td><td>48.0</td><td>74.4</td><td>42.7</td><td>48.1</td><td>56.5</td></tr><tr><td rowspan="2">PyVision-Claude</td><td>76.2</td><td>51.3</td><td>74.6</td><td>51.0</td><td>79.2</td><td>56.8</td></tr><tr><td>+4.8</td><td>+3.3</td><td>+0.2</td><td>+8.3</td><td>+31.1</td><td>+0.3</td></tr></table>

PyVision Amplifies What the Backend MLLM Does Best, Reasoning or Perception. To better understand the relationship between PyVision’s performance gains and the inherent strengths of backend models, we focus on two representative benchmarks: MathVision-mini [49], which emphasizes abstract reasoning, and $V ^ { * }$ [52], which highlights perception ability. Claude-4.0- Sonnet, stronger in abstract reasoning as shown by its higher MathVision-mini performance $( 4 8 . 0 \%$ vs. $4 6 . 4 \%$ for GPT-4.1), experiences a larger boost from PyVision $\left( + 3 . 3 \% \right)$ compared to GPT-4.1’s more modest gain $( + 2 . 3 \% )$ . Conversely, GPT-4.1, superior in perceptual tasks like $\mathrm { V } ^ { \ast }$ ( $6 8 . 1 \%$ vs. Claude-4.0-Sonnet’s $5 6 . 5 \%$ ), achieves a significantly greater improvement with PyVision $( + 7 . 8 \%$ vs. only $+ 0 . 3 \%$ ). This complementary pattern suggests that the effectiveness of dynamic tooling provided by PyVision depends critically on the backend model’s foundational reasoning and perception strengths.

Further supporting this hypothesis, experiments with Qwen2.5-VL-72B [6] yield similar findings: weaker abstract reasoning capabilities $1 8 . 4 \%$ on MathVision-mini) lead to limited improvement $( + 1 . 7 \% )$ , while stronger perceptual performance $( 6 7 . 0 \%$ on $\mathrm { V } ^ { \ast }$ ) translates into substantial gains $( + 1 0 . 0 \% )$ . These insights underline that PyVision amplifies existing backend model strengths, making the interplay of reasoning and perception crucial for unlocking the full potential of dynamic multimodal tooling.

How Often and How Much MLLMs Generate Code? Fig. 10 shows the distribution of the number of code blocks generated per user query across six benchmarks, comparing PyVision backed by GPT-4.1 and Claude-4.0-Sonnet. Each subplot visualizes how frequently the model uses code during multi-turn inference, with the legend indicating the percentage of query sessions that include any code generation. We observe that Claude-4.0-Sonnet consistently generates more code than GPT-4.1 across all domains, often with longer toolchains per query and reaching $1 0 0 \%$ code coverage. Conversely, GPT-4.1 tends to use fewer code blocks. These trends suggest a difference in agentic behavior, reflecting underlying differences in how each MLLM parses complexity and utilizes code to support reasoning.

# 5. Related Work

Multi-Modal Tool Using. To solve the compositional Visual Question Answering (VQA) task in a more transparent and interpretable fashion, early work NMN [2] use a heuristic method while IEP [20] train an LSTM network as the program generator. In the era of LLMs, a pretrained LLM, e.g., GPT-4, is used to generate programs.

![](images/014c3ccbd572c0411bfadb2a768e29176ea7325f018e06536f1d3190d67aaec8.jpg)

![](images/da7bf117f76c3a978a33fca9bf15151e400e078100200c641456a199fc9cee00.jpg)

![](images/bfcafdc1ea2d5de86a56010d76e432cb3589d10f25d9cb180885f7a6f323545e.jpg)

![](images/2fce705adbc3fe3354247a6758a67bf4a11f043a0891eb36749172fd015632e4.jpg)

![](images/fe61e8399180d033943a82d9cc09c2106eb70699c52f5efa42ec9c893a5fc117.jpg)

Figure 10 | Multi-Turn Interaction Patterns Across Tasks and Backend Models. The histograms show the distribution of the number of generated code blocks per query across six benchmarks. PyVision-GPT-4.1 (blue) and PyVision-Claude-4.0-Sonnet (red) exhibit distinct interaction patterns, with Claude consistently generating code more frequently and with more turns. The legend in each subplot indicates the percentage of samples that involved at least one code block.   
![](images/976a01389c13e26cbbbbef4512313116c0f70dcfb280324d2f88fccb8f96740f.jpg)  
PyVision-GPT-4.1 PyVision-Claude-4.0-Sonnet

Visual ChatGPT [51], MM-REACT [56], HuggingGPT [42], Image-of-Thought [61], and VAT [24] design workflows to process VQA inputs and produce final answers. In VisProg [12] and ViperGPT [47], researchers predefine a static toolset for specific vision tasks and prompt the LLMs or MLLMs to generate programs that invoke these tools to support reasoning. As LLMs’ coding abilities improve, Visual Sketchpad [17] predefines a toolset and prompts the LLM to program and execute code on the fly, offering more flexibility. These prior works rely on a static toolset containing various visual parsers [10], e.g., detection models (GroundingDINO [25]) and segmentation models (SAM [21]), which limits generality across vision tasks and makes the external models a bottleneck. In contrast, PyVision uses Python as the sole primitive tool. With the advanced coding and multimodal understanding abilities of today’s MLLMs, e.g., Claude-4.0 [3] and GPT-4.1 [35], they can write Python code to construct and execute complex tools on the fly, enabling more general and flexible reasoning.

Thinking with Images. In o3’s [37] blog, thinking with images is presented as an attractive feature. CoGCoM [39] synthesizes program-integrated data and teaches the MLLM to use predefined tools during inference. DeepEyes [60], Pixel Reasoner [44], OpenThinkIMG [45, 46], and Chain-of-Focus [58] incentivize MLLMs to develop the ability to “think with images using predefined tools” through reinforcement learning. In PyVision, we support thinking with images by using Python as the tool creation interface, enabling the MLLM to self-generate more complex and adaptive tools based on varying scenarios.

# 6. Conclusion

We propose PyVision, an agentic framework enabling MLLMs to generate and execute Python code on the fly. Different from previous visual programming works [12, 17, 47], PyVision needs no visual parsers and predefined static toolset, it generates tools dynamically from the specific query and visual input. We evaluate its effectiveness and flexibility on various benchmarks and visual reasoning scenarios, e.g., medical, multi-modal math problems, remote sensing and visual puzzles. It shows significant performance improvement on versatile benchmarks.

Acknowledgement We thank Yuxiang Lai and Jike Zhong for providing test samples in the initial stage of this project and Yunfei Xie for his feedback on the manuscript.

# References

[1] MoonShot AI. Kimi k2: Open agentic intelligence, 2025.   
[2] Jacob Andreas, Marcus Rohrbach, Trevor Darrell, and Dan Klein. Deep compositional question answering with neural module networks. arXiv:1511.02799, 2015.   
[3] Anthropic. Introducing claude 4, 2025.   
[4] Axel Backlund and Lukas Petersson. Vending-bench: A benchmark for long-term coherence of autonomous agents. arXiv preprint arXiv:2502.15840, 2025.   
[5] Hao Bai, Yifei Zhou, Jiayi Pan, Mert Cemri, Alane Suhr, Sergey Levine, and Aviral Kumar. Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning. NeurIPS, 2024.   
[6] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv:2502.13923, 2025.   
[7] Gary Bradski. The opencv library. Dr. Dobb’s Journal: Software Tools for the Professional Programmer, 2000.   
[8] Alex Clark et al. Pillow (pil fork) documentation. readthedocs, 2015.   
[9] Jiazhan Feng, Shijue Huang, Xingwei Qu, Ge Zhang, Yujia Qin, Baoquan Zhong, Chengquan Jiang, Jinxin Chi, and Wanjun Zhong. Retool: Reinforcement learning for strategic tool use in llms. arXiv:2504.11536, 2025.   
[10] Ross Girshick. The parable of the parser, 2024.   
[11] Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen. Tora: A tool-integrated reasoning agent for mathematical problem solving. arXiv preprint arXiv:2309.17452, 2023.   
[12] Tanmay Gupta and Aniruddha Kembhavi. Visual programming: Compositional visual reasoning without training. In CVPR, 2023.   
[13] Charles R Harris, K Jarrod Millman, Stéfan J Van Der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J Smith, et al. Array programming with numpy. Nature, 2020.   
[14] Sirui Hong, Xiawu Zheng, Jonathan Chen, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, et al. Metagpt: Meta programming for multi-agent collaborative framework. arXiv preprint arXiv:2308.00352, 2023.

[15] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In CVPR, 2024.   
[16] Mengkang Hu, Yuhang Zhou, Wendong Fan, Yuzhou Nie, Bowei Xia, Tao Sun, Ziyu Ye, Zhaoxuan Jin, Yingru Li, Qiguang Chen, et al. Owl: Optimized workforce learning for general multi-agent assistance in real-world task automation. arXiv preprint arXiv:2505.23885, 2025.   
[17] Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Ranjay Krishna. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. In NeurIPS, 2024.   
[18] Yutao Hu, Tianbin Li, Quanfeng Lu, Wenqi Shao, Junjun He, Yu Qiao, and Ping Luo. OmniMedVQA: A new large-scale comprehensive evaluation benchmark for medical lvlm. In CVPR, 2024.   
[19] Ted Jaeger and Kyle Klahs. The ebbinghaus illusion: New contextual effects and theoretical considerations. Perceptual and motor skills, 2015.   
[20] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Judy Hoffman, Li Fei-Fei, C. Lawrence Zitnick, and Ross B. Girshick. Inferring and executing programs for visual reasoning. In ICCV, 2017.   
[21] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything, 2023.   
[22] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. NeurIPS, 2022.   
[23] Kartik Kuckreja, Muhammad Sohail Danish, Muzammal Naseer, Abhijit Das, Salman Khan, and Fahad Shahbaz Khan. Geochat: Grounded large vision-language model for remote sensing. In CVPR, 2024.   
[24] Dairu Liu, Ziyue Wang, Minyuan Ruan, Fuwen Luo, Chi Chen, Peng Li, and Yang Liu. Visual abstract thinking empowers multimodal reasoning. arXiv:2505.20164, 2025.   
[25] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In ECCV, 2024.   
[26] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv:2310.02255, 2023.   
[27] Pan Lu, Bowen Chen, Sheng Liu, Rahul Thapa, Joseph Boen, and James Zou. Octotools: An agentic framework with extensible tools for complex reasoning. arXiv preprint arXiv:2502.11271, 2025.   
[28] Michael Luo, Naman Jain, Jaskirat Singh, Sijun Tan, Ameen Patel, Qingyang Wu, Alpay Ariyak, Colin Cai, Shang Zhu Tarun Venkat, Ben Athiwaratkun, Manan Roongta, Ce Zhang, Li Erran Li, Raluca Ada Popa, Koushik Sen, and Ion Stoica. Deepswe: Training a state-ofthe-art coding agent from scratch by scaling rl, 2025. Notion Blog.   
[29] MainFunc. Meet genspark super agent, 2025.

[30] Manus. Leave it to manus, 2025.   
[31] Wes McKinney et al. pandas: a foundational python library for data analysis and statistics. Python for high performance and scientific computing, 2011.   
[32] MiniMax. Minimax-agent, 2025.   
[33] OpenAI. Computer-using agent, 2025.   
[34] OpenAI. Introducing codex, 2025.   
[35] OpenAI. Introducing gpt-4.1 in the api, 2025.   
[36] OpenAI. New embedding models and api updates, 2025.   
[37] OpenAI. Thinking with images, 2025.   
[38] Fabian Pedregosa, Gaël Varoquaux, Alexandre Gramfort, Vincent Michel, Bertrand Thirion, Olivier Grisel, Mathieu Blondel, Peter Prettenhofer, Ron Weiss, Vincent Dubourg, et al. Scikit-learn: Machine learning in python. the Journal of machine Learning research, 2011.   
[39] Ji Qi, Ming Ding, Weihan Wang, Yushi Bai, Qingsong Lv, Wenyi Hong, Bin Xu, Lei Hou, Juanzi Li, Yuxiao Dong, and Jie Tang. Cogcom: A visual language model with chain-ofmanipulations reasoning. In ICLR, 2025.   
[40] Jiahao Qiu, Xuan Qi, Tongcheng Zhang, Xinzhe Juan, Jiacheng Guo, Yifu Lu, Yimin Wang, Zixin Yao, Qihan Ren, Xun Jiang, et al. Alita: Generalist agent enabling scalable agentic reasoning with minimal predefinition and maximal self-evolution. arXiv preprint arXiv:2505.20286, 2025.   
[41] Pooyan Rahmanzadehgervi, Logan Bolton, Mohammad Reza Taesiri, and Anh Totti Nguyen. Vision language models are blind. In ACCV, 2024.   
[42] Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang. Hugginggpt: Solving ai tasks with chatgpt and its friends in hugging face. In NeurIPS, 2023.   
[43] Yueqi Song, Tianyue Ou, Yibo Kong, Zecheng Li, Graham Neubig, and Xiang Yue. Visualpuzzles: Decoupling multimodal reasoning evaluation from domain knowledge. arXiv:2504.10342, 2025.   
[44] Alex Su, Haozhe Wang, Weiming Ren, Fangzhen Lin, and Wenhu Chen. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning, 2025.   
[45] Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, et al. Openthinkimg: Learning to think with images via visual tool reinforcement learning. arXiv preprint arXiv:2505.08617, 2025.   
[46] Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, et al. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918, 2025.   
[47] Dídac Surís, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. In ICCV, 2023.   
[48] Stefan Van der Walt, Johannes L Schönberger, Juan Nunez-Iglesias, François Boulogne, Joshua D Warner, Neil Yager, Emmanuelle Gouillart, and Tony Yu. scikit-image: image processing in python. PeerJ, 2014.

[49] Ke Wang, Junting Pan, Weikang Shi, Zimu Lu, Houxing Ren, Aojun Zhou, Mingjie Zhan, and Hongsheng Li. Measuring multimodal mathematical reasoning with math-vision dataset. NeurIPS, 2024.   
[50] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. NeurIPS, 2022.   
[51] Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv:2303.04671, 2023.   
[52] Penghao Wu and Saining Xie. V*: Guided visual search as a core mechanism in multimodal llms. In CVPR, 2024.   
[53] Zhenghai Xue, Longtao Zheng, Qian Liu, Yingru Li, Zejun Ma, and Bo An. Simpletir: End-to-end reinforcement learning for multi-turn tool-integrated reasoning, 2025. Notion Blog.   
[54] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-ofmark prompting unleashes extraordinary visual grounding in gpt-4v, 2023.   
[55] Jihan Yang, Shusheng Yang, Anjali Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in Space: How Multimodal Large Language Models See, Remember and Recall Spaces. arXiv:2412.14171, 2024.   
[56] Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. arXiv:2303.11381, 2023.   
[57] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In CVPR, 2024.   
[58] Xintong Zhang, Zhi Gao, Bofei Zhang, Pengxiang Li, Xiaowen Zhang, Yang Liu, Tao Yuan, Yuwei Wu, Yunde Jia, Song-Chun Zhu, and Qing Li. Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl, 2025.   
[59] Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. Gpt-4v(ision) is a generalist web agent, if grounded, 2024.   
[60] Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing "thinking with images" via reinforcement learning, 2025.   
[61] Qiji Zhou, Ruochen Zhou, Zike Hu, Panzhong Lu, Siyang Gao, and Yue Zhang. Image-ofthought prompting for visual reasoning refinement in multimodal large language models. arXiv:2405.13872, 2024.

# Appendix Contents

A. Additional Evaluation Details 21

A.1. System Prompt Details 21   
A.2. Evaluation Parameters Details 22

B. Examples of Generated Tools 22

B.1. Code Snippet of C R O P Tool 22   
B.2. Code Snippet of R O TAT E Tool 22   
B.3. Code Snippet of E N H A N C E C O N T R A S T Tool 23   
B.4. Code Snippet of S E G M E N TAT I O N Tool 23   
B.5. Code Snippet of D E T E C T I O N Tool 24   
B.6. Code Snippet of O C R Tool 24   
B.7. Code Snippet of R E N D E R M A R K S Tool 25   
B.8. Code Snippet of R E N D E R A U X I L I A R Y L I N E S Tool 25   
B.9. Code Snippet of V I S U L I Z E I M A G E H I S T O G R A M Tool .26   
B.10. Code Snippet of N U M E R I C A L A N A LY S I S Tool 26

# A. Additional Evaluation Details

# A.1. System Prompt Details

# System Prompt Template of PyVision

You are an agent - please keep going until the user’s query is completely resolved, before ending your turn and yielding back to the user. Only terminate your turn when you are sure that the problem is solved.

Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process. The Python code will be executed by an external sandbox.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls. DO NOT do this entire process by making function calls only, as this can impair your ability to solve the problem and think insightfully.

For all the provided images, in order, the i-th image has already been read into the global variable "image_clue_i" using the "PIL.Image.open()" function. When writing Python code, you can directly use these variables without needing to read them again.

Since you are dealing with the vision-related question answering task, you MUST use the python tool (e.g., matplotlib library) to analyze or transform images whenever it could improve your understanding or aid your reasoning. This includes but is not limited to zooming in, rotating, adjusting contrast, computing statistics, or isolating features.

Note that when you use matplotlib to visualize data or further process images, you need to use "plt.show()" to display these images; there is no need to save them. Do not use image processing libraries like cv2 or PIL. If you want to check the value of a variable, you MUST use "print()" to check it.

The output (wrapped in "<interpreter>output_str</interpreter>") can be returned to aid your reasoning and help you arrive at the final answer. The Python code should be complete scripts, including necessary imports.

Each code snippet is wrapped with: <code> python code snippet </code>

The last part of your response should be in the following format:

<answer> \boxed{"The final answer goes here."} </answer>

*image resolution:*

Image Width: {width}; Image Height: {height}

*user question:*

Answer the following Problem with an image provided and put the answer in the format of \boxed{answer} {"query"}

Remember to place the final answer in the last part using the format:

<answer> \boxed{"The final answer goes here."} </answer>

# A.2. Evaluation Details

Inference Parameters. In the evaluation stage, we set the temperature to 0.6. Here is the chain-of-thought prompt template used for evaluation.

# Chain-of-Thought Prompt Template

You are a helpful assistant. And you are dealing with the VQA tasks. Solve the visual questions step by step and give the correct answer. Note: put your answer in the format of "\boxed{the right answer here}"

*user question*: {query}

Illustration of GPT-4.1’s result in Tab. 2. To keep the consistent evaluation setting with Claude-4.0-Sonnet, we evaluated GPT-4.1 on MathVista [26] and MMMU [57] by ourselves with the above-mentioned CoT prompt.

Illustration of Qwen2.5-VL-72B’s result in Sec. 4. When evaluating Qwen2.5-VL-72B on $\mathrm { V } ^ { \ast }$ [52] and MathVision-testmini [49], to keep the consistent evaluation setting with other models, e.g., GPT-4.1 and Claude-4.0-Sonnet, we wrap the visual question with the above-mentioned CoT prompt template as input. Since Qwen2.5-VL-72B was not specifically optimized for CoT reasoning [6], the results with CoT prompt may be damaged.

# B. Examples of Generated Tools

# B.1. Code Snippet of C R O P Tool

```python
The board is towards the center-right. Based on the apparent position, let's crop the region centered at (center-right), roughly: # x: 900-1050, y: 520-700 (after visually inspecting the source image proportions) cropped = image_clue_0crop((900, 520, 1050, 700)) plt.imshow(cropped) plt.title('Zoom on Advertising Board') plt.axis('off') plt.show() 
```

Listing 1 | A code snippet example of generated C R O P tool. In this example, PyVision was queried to answer the type of the shop based on the advertising board’s information. It generated C R O P tool to focus on the board area.

# B.2. Code Snippet of R O TAT E Tool

```python
import matplotlib.pyplot as plt  
import pytesseract  
from PIL import Image  
# Step 1: Rotate the image 180 degrees to correct orientation  
rotated_image = image_clue_0rotate(180)  
# Display the rotated image to verify orientation  
plt.imshow(rotated_image)  
plt.axis('off') 
```

```txt
plt.show() 
```

Listing 2 | A code snippet example of generated R O TAT E tool. In this example, PyVision was queried to answer what is written on an upside-down image. It generated R O TAT E tool to make the image rotate 180 degree.

B.3. Code Snippet of E N H A N C E C O N T R A S T Tool   
Visualize the Image - Display the original image and enhance contrast to check for subtle abnormalities.   
#Let's write code to visualize and enhance the image to better inspect for abnormalities.   
def enhance Contrast(img): arr $=$ np.asarray(img) enhanced $=$ np.zeros_like(arr) for i in range(3): # For R, G, B channels channel $=$ arr[.,i] hist,bins $=$ np.histogram( channel.flatten(),256，[0,256]) cdf $=$ hist.cumsum() cdf_normalized $\equiv$ (cdf\*hist.max() /cdf.max()) cdf_m $=$ np.mamaskedequalcdf,0) cdf_m $=$ ((cdf_m-cdf_m.min()\*255/ (cdf_m.max()-cdf_m.min()))） cdf $=$ (np.maFilledcdf_m,0).astype('uint8')) enhanced[.,i] $=$ cdf频道] return enhanced

Listing 3 | A code snippet example of the generated E N H A N C E C O N T R A S T tool. In this example, PyVision was queried to identify retinal abnormalities in a fundus image. It generated the E N H A N C E C O N T R A S T tool to increase image contrast, making subtle pathological features more visible.

B.4. Code Snippet of S E G M E N TAT I O N Tool   
import matplotlib.pyplot as plt   
import numpy as np   
from skimage.filters import threshold_otsu   
from scipy.ndimage import label, binary_opening, binary_closing   
# Convert to grayscale   
image = imageCLAue_0   
gray $=$ np.array(image.convert('L'))   
# Otsu's thresholding on grayscale   
otsu thresh $=$ threshold_otsu(gray)   
binary_otsu $=$ gray $>$ otsu thresh   
# Let's apply morphological operations and count the number of buildings.   
# Morphological operations to clean up the mask cleaned $=$ binary_opening(binary_otsu, structure=np.ones((5,5))) cleaned $=$ binary_closing(cleaned, structure=np.ones((7,7)))

```python
# Label connected components  
labeled, num_features = label(cleaned)  
print("Number of detected buildings:", num_features)  
# Calculate area of each region  
region_sizes = np.sumulative labeled.flatten())  
min(building_area = 1000 # Adjust as needed for urban scale  
# Ignore the background (label 0)  
num_buildings = np.sum(region_sizes[1:] > min(building_area)  
print("Number of buildings (filtered by area):", num_buildings) 
```

Listing 4 | A code snippet example of generated S E G M E N TAT I O N tool. In this example, PyVision was queried to count the number of buildings in a satelite remote sensing image. The entire python tool chain (merged into a single python code snippet for conciseness) generated by PyVisionachieved a similar functionality as segmentation models by thresholding, clustering and filtering step by step.

# B.5. Code Snippet of D E T E C T I O N Tool

```python
The table is roughly at the center, estimate the coordinates to crop
# Image size: 1920x930, table is roughly between (700, 250) and (1200, 700)
left, top, right, bottom = 700, 250, 1200, 700
table_area = image_clue_0crop((left, top, right, bottom)) 
```

Listing 5 | A code snippet example of generated D E T E C T I O N tool. In this example, PyVision was queried to recognize a small cup of water in the image. It generated D E T E C T I O N tool to help cropping.

# B.6. Code Snippet of O C R Tool

import easyocr   
import numpy as np   
# Convert rotated PIL image to numpy array for easyocr img_np $=$ np.array(rotated_img)   
# Initialize EasyOCR reader reader $=$ easyocr-reader([en])   
# Perform OCR result $=$ reader.readtext(img_np, detail=0)   
# Print the recognized text print(result)

Listing 6 | A code snippet example of generated O C R tool. In this example, PyVision was queried to answer what is the text on the image. It generated O C R tool , using EasyOCR to recognize the text on the image.

B.7. Code Snippet of R E N D E R M A R K S Tool   
```python
# Based on my visual analysis, I can identify the following people in solid red shirts:  
# Let me mark them with circles and numbers  
# Person 1: Far left, standing person in bright red shirt  
plt.plot(45, 140, 'yo', markersize=15, markedegcolor='yellow', markeredgewidth=3)  
plt.text(45, 120, '1', fontsize=16, color='yellow', fontweight='bold', ha='center')  
# Person 2: Left side, sitting person in red top  
plt.plot(110, 195, 'yo', markersize=15, markedegcolor='yellow', markeredgewidth=3)  
plt.text(110, 175, '2', fontsize=16, color='yellow', fontweight='bold', ha='center')  
# Person 3: Center-right area, person in red shirt sitting  
plt.plot(380, 200, 'yo', markersize=15, markedegcolor='yellow', markeredgewidth=3)  
plt.text(380, 180, '3', fontsize=16, color='yellow', fontweight='bold', ha='center')  
plt.axis('off')  
plt.show() 
```

Listing 7 | A code snippet example of generated R E N D E R M A R K S tool. In this example, PyVision was queried to count the people wearing solid red shirts. It generated a R E N D E R M A R K S tool, labeling the people wearing red shirts.

B.8. Code Snippet of R E N D E R A U X I L I A R Y L I N E S Tool   
```python
import matplotlib.pyplot as plt  
import matplotlib)patches as patches  
# Load image  
img = image_clue_0  
# Set up plot  
fig, ax = plt.subplot(figsize=(8,6))  
ax.imshow(img)  
# Green square location (col, row): (2,4) in O-based grid (counting from top left)  
# Move left 1 step: (1,4)  
start_col, start_row = 2,4  
end_col, end_row = start_col - 1, start_row  
# Convert to image coordinates  
# Each cell: width = 640/5 = 128, height = 480/5 = 96  
cell_w, cell_h = 128, 96  
# Center of starting cell  
start_x = start_col * cell_w + cell_w // 2  
start_y = start_row * cell_h + cell_h // 2  
# Center of ending cell 
```

```txt
end_x = end_col * cell_w + cell_w // 2  
end_y = end_row * cell_h + cell_h // 2  
# Draw arrow from green square to left  
axArrow(start_x, start_y, end_x - start_x, end_y - start_y, head_width=20, head_length=20, fc='red', ec='red', lengthincludes_head=True)  
# Add annotation at end point  
ax.text(end_x, end_y, 'HERE', color='red', fontsize=16, fontweight='bold', ha='center', va='center')  
plt.axis('off')  
plt.show() 
```

Listing 8 | A code snippet example of generated R E N D E R A U X I L I A R Y L I N E S tool. In this example, PyVision was queried to answer what is the position after the next step. It generated R E N D E R A U X I L I A R Y L I N E S tool to help grounding the position via sketching.

# B.9. Code Snippet of V I S U L I Z E I M A G E H I S T O G R A M Tool

```python
Plot the histogram of pixel values  
img_array = np.array(image_clue_0)  
plt.figure(figsize=(6,4))  
plt.hist(img_array.ravel(), bins=256, color='gray')  
plt.title('Pixel Value Histogram')  
plt.xlabel('Pixel Intensity')  
pltylabel('Frequency')  
plt.show() 
```

Listing 9 | A code snippet example of generated V I S U L I Z E I M A G E H I S T O G R A M tool. In this example, PyVision was queried to if there is some specific pattern. It generated V I S U L I Z E I M A G E H I S T O G R A M tool to help detect the pattern.

# B.10. Code Snippet of N U M E R I C A L A N A LY S I S Tool

import numpy as np   
# Convert the glass area to a numpy array glass_np $=$ np.array(glass_area)   
# Calculate mean RGB values mean_rgb $=$ glass_np.mean(axis=(0,1))   
print("Mean RGB values of the glass area:"，mean_rgb)

Listing 10 | A code snippet example of generated N U M E R I C A L A N A LY S I S tool. In this example, PyVision was queried to answer the the color of the specific area in the image. It generated N U M E R I C A L A N A LY S I S tool to help analysis the color.
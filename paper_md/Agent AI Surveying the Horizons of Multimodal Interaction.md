# AGENT AI:

# SURVEYING THE HORIZONS OF MULTIMODAL INTERACTION

Zane Durante1†*, Qiuyuan Huang2‡∗, Naoki Wake2∗, Ran Gong3†, Jae Sung Park4†, Bidipta Sarkar1†, Rohan Taori1†, Yusuke Noda5, Demetri Terzopoulos3, Yejin Choi4, Katsushi Ikeuchi2, Hoi $\mathbf { V o } ^ { 5 }$ , Li Fei-Fei1, Jianfeng Gao2

1Stanford University; 2Microsoft Research, Redmond; 3University of California, Los Angeles; 4University of Washington; 5Microsoft Gaming

![](images/f1c146605a55f756d8ff391baf9196b67c53fef8a7ccb2fd36b18bf1c610653b.jpg)  
Figure 1: Overview of an Agent AI system that can perceive and act in different domains and applications. Agent AI is emerging as a promising avenue toward Artificial General Intelligence (AGI). Agent AI training has demonstrated the capacity for multi-modal understanding in the physical world. It provides a framework for reality-agnostic training by leveraging generative AI alongside multiple independent data sources. Large foundation models trained for agent and action-related tasks can be applied to physical and virtual worlds when trained on cross-reality data. We present the general overview of an Agent AI system that can perceive and act in many different domains and applications, possibly serving as a route towards AGI using an agent paradigm.

# ABSTRACT

Multi-modal AI systems will likely become a ubiquitous presence in our everyday lives. A promising approach to making these systems more interactive is to embody them as agents within physical and virtual environments. At present, systems leverage existing foundation models as the basic building blocks for the creation of embodied agents. Embedding agents within such environments facilitates the ability of models to process and interpret visual and contextual data, which is critical for the creation of more sophisticated and context-aware AI systems. For example, a system that can perceive user actions, human behavior, environmental objects, audio expressions, and the collective sentiment of a scene can be used to inform and direct agent responses within the given environment. To accelerate research on agent-based multimodal intelligence, we define “Agent AI” as a class of interactive systems that can perceive visual stimuli, language inputs, and other environmentallygrounded data, and can produce meaningful embodied actions. In particular, we explore systems that aim to improve agents based on next-embodied action prediction by incorporating external knowledge, multi-sensory inputs, and human feedback. We argue that by developing agentic AI systems in grounded environments, one can also mitigate the hallucinations of large foundation models and their tendency to generate environmentally incorrect outputs. The emerging field of Agent AI subsumes the broader embodied and agentic aspects of multimodal interactions. Beyond agents acting and interacting in the physical world, we envision a future where people can easily create any virtual reality or simulated scene and interact with agents embodied within the virtual environment.

# Contents

1 Introduction 5

1.1 Motivation 5   
1.2 Background 5   
1.3 Overview 6

2 Agent AI Integration 7

2.1 Infinite AI agent 7   
2.2 Agent AI with Large Foundation Models 8

2.2.1 Hallucinations 8   
2.2.2 Biases and Inclusivity 9   
2.2.3 Data Privacy and Usage 10   
2.2.4 Interpretability and Explainability . 11   
2.2.5 Inference Augmentation 12   
2.2.6 Regulation . 13

2.3 Agent AI for Emergent Abilities . 14

3 Agent AI Paradigm 15

3.1 LLMs and VLMs 15   
3.2 Agent Transformer Definition 15   
3.3 Agent Transformer Creation 16

4 Agent AI Learning 17

4.1 Strategy and Mechanism . 17

4.1.1 Reinforcement Learning (RL) 17   
4.1.2 Imitation Learning (IL) 18   
4.1.3 Traditional RGB 18   
4.1.4 In-context Learning 18   
4.1.5 Optimization in the Agent System 18

4.2 Agent Systems (zero-shot and few-shot level) 19

4.2.1 Agent Modules 19   
4.2.2 Agent Infrastructure 19

4.3 Agentic Foundation Models (pretraining and finetune level) 19

# 5 Agent AI Categorization 20

5.1 Generalist Agent Areas 20   
5.2 Embodied Agents 20

5.2.1 Action Agents 20   
5.2.2 Interactive Agents 21

5.3 Simulation and Environments Agents 21   
5.4 Generative Agents 21

5.4.1 AR/VR/mixed-reality Agents 22

5.5 Knowledge and Logical Inference Agents 22

5.5.1 Knowledge Agent 23   
5.5.2 Logic Agents 23   
5.5.3 Agents for Emotional Reasoning 23   
5.5.4 Neuro-Symbolic Agents . 24

5.6 LLMs and VLMs Agent . 24

# 6 Agent AI Application Tasks 24

6.1 Agents for Gaming 24

6.1.1 NPC Behavior 24   
6.1.2 Human-NPC Interaction . 25   
6.1.3 Agent-based Analysis of Gaming 25   
6.1.4 Scene Synthesis for Gaming 27   
6.1.5 Experiments and Results 27

6.2 Robotics 28

6.2.1 LLM/VLM Agent for Robotics. 30   
6.2.2 Experiments and Results. 31

6.3 Healthcare 35   
6.3.1 Current Healthcare Capabilities 36

6.4 Multimodal Agents 36

6.4.1 Image-Language Understanding and Generation 36   
6.4.2 Video and Language Understanding and Generation 37   
6.4.3 Experiments and Results 39

6.5 Video-language Experiments 41   
6.6 Agent for NLP 45

6.6.1 LLM agent 45   
6.6.2 General LLM agent 45   
6.6.3 Instruction-following LLM agents 46   
6.6.4 Experiments and Results 46

# 7 Agent AI Across Modalities, Domains, and Realities 48

7.1 Agents for Cross-modal Understanding 48   
7.2 Agents for Cross-domain Understanding 48   
7.3 Interactive agent for cross-modality and cross-reality 49   
7.4 Sim to Real Transfer . 49

# 8 Continuous and Self-improvement for Agent AI 49

8.1 Human-based Interaction Data 49   
8.2 Foundation Model Generated Data 50

# 9 Agent Dataset and Leaderboard 50

9.1 “CuisineWorld” Dataset for Multi-agent Gaming 50

9.1.1 Benchmark 51   
9.1.2 Task 51   
9.1.3 Metrics and Judging 51   
9.1.4 Evaluation 51

9.2 Audio-Video-Language Pre-training Dataset. 51

# 10 Broader Impact Statement 52

# 11 Ethical Considerations 53

12 Diversity Statement 53

References 55

Appendix 69

A GPT-4V Agent Prompt Details 69   
B GPT-4V for Bleeding Edge 69   
C GPT-4V for Microsoft Fight Simulator 69   
D GPT-4V for Assassin’s Creed Odyssey 69   
E GPT-4V for GEARS of WAR 4 69   
F GPT-4V for Starfield 75   
Author Biographies 77   
Acknowledgemets 80

# 1 Introduction

# 1.1 Motivation

Historically, AI systems were defined at the 1956 Dartmouth Conference as artificial life forms that could collect information from the environment and interact with it in useful ways. Motivated by this definition, Minsky’s MIT group built in 1970 a robotics system, called the “Copy Demo,” that observed “blocks world” scenes and successfully reconstructed the observed polyhedral block structures. The system, which comprised observation, planning, and manipulation modules, revealed that each of these subproblems is highly challenging and further research was necessary. The AI field fragmented into specialized subfields that have largely independently made great progress in tackling these and other problems, but over-reductionism has blurred the overarching goals of AI research.

To advance beyond the status quo, it is necessary to return to AI fundamentals motivated by Aristotelian Holism. Fortunately, the recent revolution in Large Language Models (LLMs) and Visual Language Models (VLMs) has made it possible to create novel AI agents consistent with the holistic ideal. Seizing upon this opportunity, this article explores models that integrate language proficiency, visual cognition, context memory, intuitive reasoning, and adaptability. It explores the potential completion of this holistic synthesis using LLMs and VLMs. In our exploration, we also revisit system design based on Aristotle’s Final Cause, the teleological “why the system exists”, which may have been overlooked in previous rounds of AI development.

With the advent of powerful pretrained LLMs and VLMs, a renaissance in natural language processing and computer vision has been catalyzed. LLMs now demonstrate an impressive ability to decipher the nuances of real-world linguistic data, often achieving abilities that parallel or even surpass human expertise (OpenAI, 2023). Recently, researchers have shown that LLMs may be extended to act as agents within various environments, performing intricate actions and tasks when paired with domain-specific knowledge and modules (Xi et al., 2023). These scenarios, characterized by complex reasoning, understanding of the agent’s role and its environment, along with multi-step planning, test the agent’s ability to make highly nuanced and intricate decisions within its environmental constraints (Wu et al., 2023; Meta Fundamental AI Research (FAIR) Diplomacy Team et al., 2022).

Building upon these initial efforts, the AI community is on the cusp of a significant paradigm shift, transitioning from creating AI models for passive, structured tasks to models capable of assuming dynamic, agentic roles in diverse and complex environments. In this context, this article investigates the immense potential of using LLMs and VLMs as agents, emphasizing models that have a blend of linguistic proficiency, visual cognition, contextual memory, intuitive reasoning, and adaptability. Leveraging LLMs and VLMs as agents, especially within domains like gaming, robotics, and healthcare, promises not just a rigorous evaluation platform for state-of-the-art AI systems, but also foreshadows the transformative impacts that Agent-centric AI will have across society and industries. When fully harnessed, agentic models can redefine human experiences and elevate operational standards. The potential for sweeping automation ushered in by these models portends monumental shifts in industries and socio-economic dynamics. Such advancements will be intertwined with multifaceted leader-board, not only technical but also ethical, as we will elaborate upon in Section 11. We delve into the overlapping areas of these sub-fields of Agent AI and illustrate their interconnectedness in Fig.1.

# 1.2 Background

We will now introduce relevant research papers that support the concepts, theoretical background, and modern implementations of Agent AI.

Large Foundation Models: LLMs and VLMs have been driving the effort to develop general intelligent machines (Bubeck et al., 2023; Mirchandani et al., 2023). Although they are trained using large text corpora, their superior problem-solving capacity is not limited to canonical language processing domains. LLMs can potentially tackle complex tasks that were previously presumed to be exclusive to human experts or domain-specific algorithms, ranging from mathematical reasoning (Imani et al., 2023; Wei et al., 2022; Zhu et al., 2022) to answering questions of professional law (Blair-Stanek et al., 2023; Choi et al., 2023; Nay, 2022). Recent research has shown the possibility of using LLMs to generate complex plans for robots and game AI (Liang et al., 2022; Wang et al., 2023a,b; Yao et al., 2023a; Huang et al., 2023a), marking an important milestone for LLMs as general-purpose intelligent agents.

Embodied AI: A number of works leverage LLMs to perform task planning (Huang et al., 2022a; Wang et al., 2023b; Yao et al., 2023a; Li et al., 2023a), specifically the LLMs’ WWW-scale domain knowledge and emergent zero-shot embodied abilities to perform complex task planning and reasoning. Recent robotics research also leverages LLMs to perform task planning (Ahn et al., 2022a; Huang et al., 2022b; Liang et al., 2022) by decomposing natural language instruction into a sequence of subtasks, either in the natural language form or in Python code, then using a low-level controller to execute these subtasks. Additionally, they incorporate environmental feedback to improve task performance (Huang et al., 2022b), (Liang et al., 2022), (Wang et al., 2023a), and (Ikeuchi et al., 2023).

Interactive Learning: AI agents designed for interactive learning operate using a combination of machine learning techniques and user interactions. Initially, the AI agent is trained on a large dataset. This dataset includes various types of information, depending on the intended function of the agent. For instance, an AI designed for language tasks would be trained on a massive corpus of text data. The training involves using machine learning algorithms, which could include deep learning models like neural networks. These training models enable the AI to recognize patterns, make predictions, and generate responses based on the data on which it was trained. The AI agent can also learn from real-time interactions with users. This interactive learning can occur in various ways: 1) Feedback-based learning: The AI adapts its responses based on direct user feedback (Li et al., 2023b; Yu et al., 2023a; Parakh et al., 2023; Zha et al., 2023; Wake et al., 2023a,b,c). For example, if a user corrects the AI’s response, the AI can use this information to improve future responses (Zha et al., 2023; Liu et al., 2023a). 2) Observational Learning: The AI observes user interactions and learns implicitly. For example, if users frequently ask similar questions or interact with the AI in a particular way, the AI might adjust its responses to better suit these patterns. It allows the AI agent to understand and process human language, multi-model setting, interpret the cross reality-context, and generate human-users’ responses. Over time, with more user interactions and feedback, the AI agent’s performance generally continuous improves. This process is often supervised by human operators or developers who ensure that the AI is learning appropriately and not developing biases or incorrect patterns.

# 1.3 Overview

Multimodal Agent AI (MAA) is a family of systems that generate effective actions in a given environment based on the understanding of multimodal sensory input. With the advent of Large Language Models (LLMs) and Vision-Language Models (VLMs), numerous MAA systems have been proposed in fields ranging from basic research to applications. While these research areas are growing rapidly by integrating with the traditional technologies of each domain (e.g., visual question answering and vision-language navigation), they share common interests such as data collection, benchmarking, and ethical perspectives. In this paper, we focus on the some representative research areas of MAA, namely multimodality, gaming (VR/AR/MR), robotics, and healthcare, and we aim to provide comprehensive knowledge on the common concerns discussed in these fields. As a result we expect to learn the fundamentals of MAA and gain insights to further advance their research. Specific learning outcomes include:

• MAA Overview: A deep dive into its principles and roles in contemporary applications, providing researcher with a thorough grasp of its importance and uses.   
• Methodologies: Detailed examples of how LLMs and VLMs enhance MAAs, illustrated through case studies in gaming, robotics, and healthcare.   
• Performance Evaluation: Guidance on the assessment of MAAs with relevant datasets, focusing on their effectiveness and generalization.   
• Ethical Considerations: A discussion on the societal impacts and ethical leader-board of deploying Agent AI, highlighting responsible development practices.   
• Emerging Trends and Future leader-board: Categorize the latest developments in each domain and discuss the future directions.

Computer-based action and generalist agents (GAs) are useful for many tasks. A GA to become truly valuable to its users, it can natural to interact with, and generalize to a broad range of contexts and modalities. We aims to cultivate a vibrant research ecosystem and create a shared sense of identity and purpose among the Agent AI community. MAA has the potential to be widely applicable across various contexts and modalities, including input from humans. Therefore, we believe this Agent AI area can engage a diverse range of researchers, fostering a dynamic Agent AI community and

shared goals. Led by esteemed experts from academia and industry, we expect that this paper will be an interactive and enriching experience, complete with agent instruction, case studies, tasks sessions, and experiments discussion ensuring a comprehensive and engaging learning experience for all researchers.

This paper aims to provide general and comprehensive knowledge about the current research in the field of Agent AI. To this end, the rest of the paper is organized as follows. Section 2 outlines how Agent AI benefits from integrating with related emerging technologies, particularly large foundation models. Section 3 describes a new paradigm and framework that we propose for training Agent AI. Section 4 provides an overview of the methodologies that are widely used in the training of Agent AI. Section 5 categorizes and discusses various types of agents. Section 6 introduces Agent AI applications in gaming, robotics, and healthcare. Section 7 explores the research community’s efforts to develop a versatile Agent AI, capable of being applied across various modalities, domains, and bridging the sim-to-real gap. Section 8 discusses the potential of Agent AI that not only relies on pre-trained foundation models, but also continuously learns and self-improves by leveraging interactions with the environment and users. Section 9 introduces our new datasets that are designed for the training of multimodal Agent AI. Section 11 discusses the hot topic of the ethics consideration of AI agent, limitations, and societal impact of our paper.

# 2 Agent AI Integration

Foundation models based on LLMs and VLMs, as proposed in previous research, still exhibit limited performance in the area of embodied AI, particularly in terms of understanding, generating, editing, and interacting within unseen environments or scenarios (Huang et al., 2023a; Zeng et al., 2023). Consequently, these limitations lead to sub-optimal outputs from AI agents. Current agent-centric AI modeling approaches focus on directly accessible and clearly defined data (e.g. text or string representations of the world state) and generally use domain and environment-independent patterns learned from their large-scale pretraining to predict action outputs for each environment (Xi et al., 2023; Wang et al., 2023c; Gong et al., 2023a; Wu et al., 2023). In (Huang et al., 2023a), we investigate the task of knowledge-guided collaborative and interactive scene generation by combining large foundation models, and show promising results that indicate knowledge-grounded LLM agents can improve the performance of 2D and 3D scene understanding, generation, and editing, alongside with other human-agent interactions (Huang et al., 2023a). By integrating an Agent AI framework, large foundation models are able to more deeply understand user input to form a complex and adaptive HCI system. Emergent ability of LLM and VLM works invisible in generative AI, embodied AI, knowledge augmentation for multi-model learning, mix-reality generation, text to vision editing, human interaction for 2D/3D simulation in gaming or robotics tasks. Agent AI recent progress in foundation models present an imminent catalyst for unlocking general intelligence in embodied agents. The large action models, or agent-vision-language models open new possibilities for general-purpose embodied systems such as planning, problem-solving and learning in complex environments. Agent AI test further step in metaverse, and route the early version of AGI.

# 2.1 Infinite AI agent

AI agents have the capacity to interpret, predict, and respond based on its training and input data. While these capabilities are advanced and continually improving, it’s important to recognize their limitations and the influence of the underlying data they are trained on. AI agent systems generally possess the following abilities: 1) Predictive Modeling: AI agents can predict likely outcomes or suggest next steps based on historical data and trends. For instance, they might predict the continuation of a text, the answer to a question, the next action for a robot, or the resolution of a scenario. 2) Decision Making: In some applications, AI agents can make decisions based on their inferences. Generally, the agent will base their decision on what is most likely to achieve a specified goal. For AI applications like recommendation systems, an agent can decide what products or content to recommend based on its inferences about user preferences. 3) Handling Ambiguity: AI agents can often handle ambiguous input by inferring the most likely interpretation based on context and training. However, their ability to do so is limited by the scope of their training data and algorithms. 4) Continuous Improvement: While some AI agents have the ability to learn from new data and interactions, many large language models do not continuously update their knowledge-base or internal representation after training. Their inferences are usually based solely on the data that was available up to the point of their last training update.

We show augmented interactive agents for multi-modality and cross reality-agnostic integration with an emergence mechanism in Fig. 2. An AI agent requires collecting extensive training data for every new task, which can be costly or impossible for many domains. In this study, we develop an infinite agent that learns to transfer memory information from

![](images/7a6ce783073a43f17c46d071696d7790f83e034895f552db966b8c612dd249f3.jpg)  
Figure 2: The multi-model agent AI for 2D/3D embodied generation and editing interaction in cross-reality.

general foundation models (e.g., GPT-X, DALL-E) to novel domains or scenarios for scene understanding, generation, and interactive editing in physical or virtual worlds.

An application of such an infinite agent in robotics is RoboGen (Wang et al., 2023d). In this study, the authors propose a pipeline that autonomously run the cycles of task proposition, environment generation, and skill learning. RoboGen is an effort to transfer the knowledge embedded in large models to robotics.

# 2.2 Agent AI with Large Foundation Models

Recent studies have indicated that large foundation models play a crucial role in creating data that act as benchmarks for determining the actions of agents within environment-imposed constraints. For example, using foundation models for robotic manipulation (Black et al., 2023; Ko et al., 2023) and navigation (Shah et al., 2023a; Zhou et al., 2023a). To illustrate, Black et al. employed an image-editing model as a high-level planner to generate images of future sub-goals, thereby guiding low-level policies (Black et al., 2023). For robot navigation, Shah et al. proposed a system that employs a LLM to identify landmarks from text and a VLM to associate these landmarks with visual inputs, enhancing navigation through natural language instructions (Shah et al., 2023a).

There is also growing interest in the generation of conditioned human motions in response to language and environmental factors. Several AI systems have been proposed to generate motions and actions that are tailored to specific linguistic instructions (Kim et al., 2023; Zhang et al., 2022; Tevet et al., 2022) and to adapt to various 3D scenes (Wang et al., 2022a). This body of research emphasizes the growing capabilities of generative models in enhancing the adaptability and responsiveness of AI agents across diverse scenarios.

# 2.2.1 Hallucinations

Agents that generate text are often prone to hallucinations, which are instances where the generated text is nonsensical or unfaithful to the provided source content (Raunak et al., 2021; Maynez et al., 2020). Hallucinations can be split into two categories, intrinsic and extrinsic (Ji et al., 2023). Intrinsic hallucinations are hallucinations that are contradictory

to the source material, whereas extrinsic hallucinations are when the generated text contains additional information that was not originally included in the source material.

Some promising routes for reducing the rate of hallucination in language generation involve using retrieval-augmented generation (Lewis et al., 2020; Shuster et al., 2021) or other methods for grounding natural language outputs via external knowledge retrieval (Dziri et al., 2021; Peng et al., 2023). Generally, these methods seek to augment language generation by retrieving additional source material and by providing mechanisms to check for contradictions between the generated response and the source material.

Within the context of multi-modal agent systems, VLMs have been shown to hallucinate as well (Zhou et al., 2023b). One common cause of hallucination for vision-based language-generation is due to the over-reliance on co-occurrence of objects and visual cues in the training data (Rohrbach et al., 2018). AI agents that exclusively rely upon pretrained LLMs or VLMs and use limited environment-specific finetuning can be particularly vulnerable to hallucinations since they rely upon the internal knowledge-base of the pretrained models for generating actions and may not accurately understand the dynamics of the world state in which they are deployed.

# 2.2.2 Biases and Inclusivity

AI agents based on LLMs or LMMs (large multimodal models) have biases due to several factors inherent in their design and training process. When designing these AI agents, we must be mindful of being inclusive and aware of the needs of all end users and stakeholders. In the context of AI agents, inclusivity refers to the measures and principles employed to ensure that the agent’s responses and interactions are inclusive, respectful, and sensitive to a wide range of users from diverse backgrounds. We list key aspects of agent biases and inclusivity below.

• Training Data: Foundation models are trained on vast amounts of text data collected from the internet, including books, articles, websites, and other text sources. This data often reflects the biases present in human society, and the model can inadvertently learn and reproduce these biases. This includes stereotypes, prejudices, and slanted viewpoints related to race, gender, ethnicity, religion, and other personal attributes. In particular, by training on internet data and often only English text, models implicitly learn the cultural norms of Western, Educated, Industrialized, Rich, and Democratic (WEIRD) societies (Henrich et al., 2010) who have a disproportionately large internet presence. However, it is essential to recognize that datasets created by humans cannot be entirely devoid of bias, since they frequently mirror the societal biases and the predispositions of the individuals who generated and/or compiled the data initially.   
• Historical and Cultural Biases: AI models are trained on large datasets sourced from diverse content. Thus, the training data often includes historical texts or materials from various cultures. In particular, training data from historical sources may contain offensive or derogatory language representing a particular society’s cultural norms, attitudes, and prejudices. This can lead to the model perpetuating outdated stereotypes or not fully understanding contemporary cultural shifts and nuances.   
• Language and Context Limitations: Language models might struggle with understanding and accurately representing nuances in language, such as sarcasm, humor, or cultural references. This can lead to misinterpretations or biased responses in certain contexts. Furthermore, there are many aspects of spoken language that are not captured by pure text data, leading to a potential disconnect between human understanding of language and how models understand language.   
• Policies and Guidelines: AI agents operate under strict policies and guidelines to ensure fairness and inclusivity. For instance, in generating images, there are rules to diversify depictions of people, avoiding stereotypes related to race, gender, and other attributes.   
• Overgeneralization: These models tend to generate responses based on patterns seen in the training data. This can lead to overgeneralizations, where the model might produce responses that seem to stereotype or make broad assumptions about certain groups.   
• Constant Monitoring and Updating: AI systems are continuously monitored and updated to address any emerging biases or inclusivity issues. Feedback from users and ongoing research in AI ethics play a crucial role in this process.

• Amplification of Dominant Views: Since the training data often includes more content from dominant cultures or groups, the model may be more biased towards these perspectives, potentially underrepresenting or misrepresenting minority viewpoints.   
• Ethical and Inclusive Design: AI tools should be designed with ethical considerations and inclusivity as core principles. This includes respecting cultural differences, promoting diversity, and ensuring that the AI does not perpetuate harmful stereotypes.   
• User Guidelines: Users are also guided on how to interact with AI in a manner that promotes inclusivity and respect. This includes refraining from requests that could lead to biased or inappropriate outputs. Furthermore, it can help mitigate models learning harmful material from user interactions.

Despite these measures, AI agents still exhibit biases. Ongoing efforts in agent AI research and development are focused on further reducing these biases and enhancing the inclusivity and fairness of agent AI systems. Efforts to Mitigate Biases:

• Diverse and Inclusive Training Data: Efforts are made to include a more diverse and inclusive range of sources in the training data.   
• Bias Detection and Correction: Ongoing research focuses on detecting and correcting biases in model responses.   
• Ethical Guidelines and Policies: Models are often governed by ethical guidelines and policies designed to mitigate biases and ensure respectful and inclusive interactions.   
• Diverse Representation: Ensuring that the content generated or the responses provided by the AI agent represent a wide range of human experiences, cultures, ethnicities, and identities. This is particularly relevant in scenarios like image generation or narrative construction.   
• Bias Mitigation: Actively working to reduce biases in the AI’s responses. This includes biases related to race, gender, age, disability, sexual orientation, and other personal characteristics. The goal is to provide fair and balanced responses that do not perpetuate stereotypes or prejudices.   
• Cultural Sensitivity: The AI is designed to be culturally sensitive, acknowledging and respecting the diversity of cultural norms, practices, and values. This includes understanding and appropriately responding to cultural references and nuances.   
• Accessibility: Ensuring that the AI agent is accessible to users with different abilities, including those with disabilities. This can involve incorporating features that make interactions easier for people with visual, auditory, motor, or cognitive impairments.   
• Language-based Inclusivity: Providing support for multiple languages and dialects to cater to a global user base, and being sensitive to the nuances and variations within a language (Liu et al., 2023b).   
• Ethical and Respectful Interactions: The Agent is programmed to interact ethically and respectfully with all users, avoiding responses that could be deemed offensive, harmful, or disrespectful.   
• User Feedback and Adaptation: Incorporating user feedback to continually improve the inclusivity and effectiveness of the AI agent. This includes learning from interactions to better understand and serve a diverse user base.   
• Compliance with Inclusivity Guidelines: Adhering to established guidelines and standards for inclusivity in AI agent, which are often set by industry groups, ethical boards, or regulatory bodies.

Despite these efforts, it’s important to be aware of the potential for biases in responses and to interpret them with critical thinking. Continuous improvements in AI agent technology and ethical practices aim to reduce these biases over time. One of the overarching goals for inclusivity in agent AI is to create an agent that is respectful and accessible to all users, regardless of their background or identity.

# 2.2.3 Data Privacy and Usage

One key ethical consideration of AI agents involves comprehending how these systems handle, store, and potentially retrieve user data. We discuss key aspects below:

Data Collection, Usage and Purpose. When using user data to improve model performance, model developers access the data the AI agent has collected while in production and interacting with users. Some systems allow users to view their data through user accounts or by making a request to the service provider. It is important to recognize what data the AI agent collects during these interactions. This could include text inputs, user usage patterns, personal preferences, and sometimes more sensitive personal information. Users should also understand how the data collected from their interactions is used. If, for some reason, the AI holds incorrect information about a particular person or group, there should be a mechanism for users to help correct this once identified. This is important for both accuracy and to be respectful of all users and groups. Common uses for retrieving and analyzing user data include improving user interaction, personalizing responses, and system optimization. It is extremely important for developers to ensure the data is not used for purposes that users have not consented to, such as unsolicited marketing.

Storage and Security. Developers should know where the user interaction data is stored and what security measures are in place to protect it from unauthorized access or breaches. This includes encryption, secure servers, and data protection protocols. It is extremely important to determine if agent data is shared with third parties and under what conditions. This should be transparent and typically requires user consent.

Data Deletion and Retention. It is also important for users to understand how long user data is stored and how users can request its deletion. Many data protection laws give users the right to be forgotten, meaning they can request their data be erased. AI agents must adhere to data protection laws like GDPR in the EU or CCPA in California. These laws govern data handling practices and user rights regarding their personal data.

Data Portability and Privacy Policy. Furthermore, developers must create the AI agent’s privacy policy to document and explain to users how their data is handled. This should detail data collection, usage, storage, and user rights. Developers should ensure that they obtain user consent for data collection, especially for sensitive information. Users typically have the option to opt-out or limit the data they provide. In some jurisdictions, users may even have the right to request a copy of their data in a format that can be transferred to another service provider.

Anonymization. For data used in broader analysis or AI training, it should ideally be anonymized to protect individual identities. Developers must understand how their AI agent retrieves and uses historical user data during interactions. This could be for personalization or improving response relevance.

In summary, understanding data privacy for AI agents involves being aware of how user data is collected, used, stored, and protected, and ensuring that users understand their rights regarding accessing, correcting, and deleting their data. Awareness of the mechanisms for data retrieval, both by users and the AI agent, is also crucial for a comprehensive understanding of data privacy.

# 2.2.4 Interpretability and Explainability

Imitation Learning Decoupling. Agents are typically trained using a continuous feedback loop in Reinforcement Learning (RL) or Imitation Learning (IL), starting with a randomly initialized policy. However, this approach faces leader-board in obtaining initial rewards in unfamiliar environments, particularly when rewards are sparse or only available at the end of a long-step interaction. Thus, a superior solution is to use an infinite-memory agent trained through IL, which can learn policies from expert data, improving exploration and utilization of unseen environmental space with emergent infrastructure as shown in Fig. 3. With expert characteristics to help the agent explore better and utilize the unseen environmental space. Agent AI, can learn policies and new paradigm flow directly from expert data.

Traditional IL has an agent mimicking an expert demonstrator’s behavior to learn a policy. However, learning the expert policy directly may not always be the best approach, as the agent may not generalize well to unseen situations. To tackle this, we propose learning an agent with in-context prompt or a implicit reward function that captures key aspects of the expert’s behavior, as shown in Fig. 3. This equips the infinite memory agent with physical-world behavior data for task execution, learned from expert demonstrations. It helps overcome existing imitation learning drawbacks like the need for extensive expert data and potential errors in complex tasks. The key idea behind the Agent AI has two parts: 1) the infinite agent that collects physical-world expert demonstrations as state-action pairs and 2) the virtual environment that imitates the agent generator. The imitating agent produces actions that mimic the expert’s behavior, while the agent learns a policy mapping from states to actions by reducing a loss function of the disparity between the expert’s actions and the actions generated by the learned policy.

Decoupling Generalization. Rather than relying on a task-specific reward function, the agent learns from expert demonstrations, which provide a diverse set of state-action pairs covering various task aspects. The agent then learns a policy that maps states to actions by imitating the expert’s behavior. Decoupling in imitation learning refers to separating the learning process from the task-specific reward function, allowing the policy to generalize across different tasks without explicit reliance on the task-specific reward function. By decoupling, the agent can learn from expert demonstrations and learn a policy that is adaptable to a variety of situations. Decoupling enables transfer learning, where a policy learned in one domain can adapt to others with minimal fine-tuning. By learning a general policy that is not tied to a specific reward function, the agent can leverage the knowledge it acquired in one task to perform well in other related tasks. Since the agent does not rely on a specific reward function, it can adapt to changes in the reward function or environment without the need for significant retraining. This makes the learned policy more robust and generalizable across different environments. Decoupling in this context refers to the separation of two tasks in the learning process: learning the reward function and learning the optimal policy.

Generalization Emergent Behavior. Generalization explains how emergent properties or behaviors can arise from simpler components or rules. The key idea lies in identifying the basic elements or rules that govern the behavior of the system, such as individual neurons or basic algorithms. Consequently, by observing how these simple components or rules interact with one another. These interactions of these components often lead to the emergence of complex behaviors, which are not predictable by examining individual components alone. Generalization across different levels of complexity allows a system to learn general principles applicable across these levels, leading to emergent properties. This enables the system to adapt to new situations, demonstrating the emergence of more complex behaviors from simpler rules. Furthermore, the ability to generalize across different complexity levels facilitates knowledge transfer from one domain to another, which contributes to the emergence of complex behaviors in new contexts as the system adapts.

![](images/88cf79408b1d65bb8b070e58b61b753a1fcaa6d9f5c12c058bad3dfa4fe78e14.jpg)  
Figure 3: Example of the Emergent Interactive Mechanism using an agent to identify text relevant to the image from candidates. The task involves using a multi-modal AI agent from the web and human-annotated knowledge interaction samples to incorporate external world information.

# 2.2.5 Inference Augmentation

The inference ability of an AI agent lies in its capacity to interpret, predict, and respond based on its training and input data. While these capabilities are advanced and continually improving, it’s important to recognize their limitations and the influence of the underlying data they are trained on. Particularly, in the context of large language models, it refers to its capacity to draw conclusions, make predictions, and generate responses based on the data it has been trained on and the input it receives. Inference augmentation in AI agents refers to enhancing the AI’s natural inference abilities with additional tools, techniques, or data to improve its performance, accuracy, and utility. This can be particularly important in complex decision-making scenarios or when dealing with nuanced or specialized content. We denote particularly important sources for inference augmentation below:

Data Enrichment. Incorporating additional, often external, data sources to provide more context or background can help the AI agent make more informed inferences, especially in areas where its training data may be limited. For example, AI agents can infer meaning from the context of a conversation or text. They analyze the given information and use it to understand the intent and relevant details of user queries. These models are proficient at recognizing patterns in data. They use this ability to make inferences about language, user behavior, or other relevant phenomena based on the patterns they’ve learned during training.

Algorithm Enhancement. Improving the AI’s underlying algorithms to make better inferences. This could involve using more advanced machine learning models, integrating different types of AI (like combining NLP with image recognition), or updating algorithms to better handle complex tasks. Inference in language models involves understanding and generating human language. This includes grasping nuances like tone, intent, and the subtleties of different linguistic constructions.

Human-in-the-Loop (HITL). Involving human input to augment the AI’s inferences can be particularly useful in areas where human judgment is crucial, such as ethical considerations, creative tasks, or ambiguous scenarios. Humans can provide guidance, correct errors, or offer insights that the agent would not be able to infer on its own.

Real-Time Feedback Integration. Using real-time feedback from users or the environment to enhance inferences is another promising method for improving performance during inference. For example, an AI might adjust its recommendations based on live user responses or changing conditions in a dynamic system. Or, if the agent is taking actions in a simulated environment that break certain rules, the agent can be dynamically given feedback to help correct itself.

Cross-Domain Knowledge Transfer. Leveraging knowledge or models from one domain to improve inferences in another can be particularly helpful when producing outputs within a specialized discipline. For instance, techniques developed for language translation might be applied to code generation, or insights from medical diagnostics could enhance predictive maintenance in machinery.

Customization for Specific Use Cases. Tailoring the AI’s inference capabilities for particular applications or industries can involve training the AI on specialized datasets or fine-tuning its models to better suit specific tasks, such as legal analysis, medical diagnosis, or financial forecasting. Since the particular language or information within one domain can greatly contrast with the language from other domains, it can be beneficial to finetune the agent on domain-specific information.

Ethical and Bias Considerations. It is important to ensure that the augmentation process does not introduce new biases or ethical issues. This involves careful consideration of the sources of additional data or the impact of the new inference augmentation algorithms on fairness and transparency. When making inferences, especially about sensitive topics, AI agents must sometimes navigate ethical considerations. This involves avoiding harmful stereotypes, respecting privacy, and ensuring fairness.

Continuous Learning and Adaptation. Regularly updating and refining the AI’s capabilities to keep up with new developments, changing data landscapes, and evolving user needs.

In summmary, winference augmentation in AI agents involves methods in which their natural inference abilities can be enhanced through additional data, improved algorithms, human input, and other techniques. Depending on the use-case, this augmentation is often essential for dealing with complex tasks and ensuring accuracy in the agent’s outputs.

# 2.2.6 Regulation

Recently, Agent AI has made significant advancements, and its integration into embodied systems has opened new possibilities for interacting with agents via more immersive, dynamic, and engaging experiences. To expedite the process and ease the cumbersome work in agent AI developing, we are proposing to develop the next-generation AI-empowered pipeline for agent interaction. Develop a human-machine collaboration system where humans and machines can communicate and interact meaningfully. The system can leverage the LLM’s or VLM dialog capabilities and vast action to talk with human players and identify human needs. Then it will perform proper actions to help human players upon request.

When employing LLM/VLMs for a human-machine collaboration system, it is essential to note that these operate as black boxes, generating unpredictable output. This uncertainty can become crucial in a physical setup, such as operating actual robotics. An approach to address this challenge is constraining the focus of the LLM/VLM through prompt engineering. For instance, in robotic task planning from instructions, providing environmental information within the prompt has been reported to yield more stable outputs than relying solely on text (Gramopadhye and Szafir, 2022). This

![](images/b02a34444cbb310cee955097203bb96266107cf041d98632e3967f0383bcadfb.jpg)

![](images/f9ec2c183d04aa13bf94cae704b894eec9753d6397bde2eea32dc3ec881ffa64.jpg)  
Figure 4: A robot teaching system developed in (Wake et al., 2023c). (Left) The system workflow. The process involves three steps: Task planning, where ChatGPT plans robotic tasks from instructions and environmental information; Demonstration, where the user visually demonstrates the action sequence. All the steps are reviewed by the user, and if any step fails or shows deficiencies, the previous steps can be revisited as necessary. (Right) A web application that enables uploading of demonstration data and the interaction between the user and ChatGPT.

report is supported by the Minsky’s frame theory of AI (Minsky, 1975), suggesting that the problem space to be solved by LLM/VLMs is defined by the given prompts. Another approach is designing prompts to make LLM/VLMs include explanatory text to allow users understand what the model has focused on or recognized. Additionally, implementing a higher layer that allows for pre-execution verification and modification under human guidance can facilitate the operation of systems working under such guidance (Fig. 4).

# 2.3 Agent AI for Emergent Abilities

Despite the growing adoption of interactive agent AI systems, the majority of proposed methods still face a challenge in terms of their generalization performance in unseen environments or scenarios. Current modeling practices require developers to prepare large datasets for each domain to finetune/pretrain models; however, this process is costly and even impossible if the domain is new. To address this issue, we build interactive agents that leverage the knowledge-memory of general-purpose foundation models (ChatGPT, Dall-E, GPT-4, etc.) for a novel scenario, specifically for generating a collaboration space between humans and agents. We discover an emergent mechanism— which we name Mixed Reality with Knowledge Inference Interaction—that facilitates collaboration with humans to solve challenging tasks in complex real-world environments and enables the exploration of unseen environments for adaptation to virtual reality. For this mechanism, the agent learns i) micro-reactions in cross-modality: collecting relevant individual knowledge for each interaction task (e.g., understanding unseen scenes) from the explicit web source and by implicitly inferring from the output of pretrained models; ii) macro-behavior in reality-agnostic: improving interactive dimensions and patterns in language and multi-modality domains, and make changes based on characterized roles, certain target variable, influenced diversification of collaborative information in mixed-reality and LLMs. We investigate the task of knowledge-guided interactive synergistic effects to collaborated scene generation with combining various OpenAI models, and show promising results of how the interactive agent system can further boost the large foundation models in

![](images/35ccb3dc15164d21929eb3cc72dfc33f48c2aebdb8aa2d783af22021400e69bb.jpg)  
Figure 5: Our proposed new agent paradigm for a multi-modal generalist agent. There are 5 main modules as shown in the figures: 1) Environment and Perception with task-planning and skill observation; 2) Agent learning; 3) Memory; 4) Agent action; 5) Cognition.

our setting. It integrates and improves the depth of generalization, conscious and interpretability of a complex adaptive AI systems.

# 3 Agent AI Paradigm

In this section, we discuss a new paradigm and framework for training Agent AI. We seek to accomplish several goals with our proposed framework:

• Make use of existing pre-trained models and pre-training strategies to effectively bootstrap our agents with effective understanding of important modalities, such as text or visual inputs.   
• Support for sufficient long-term task-planning capabilities.   
• Incorporate a framework for memory that allows for learned knowledge to be encoded and retrieved later.   
• Allow for environmental feedback to be used to effectively train the agent to learn which actions to take.

We show a high-level new agent diagram outlining the important submodules of such a system in Fig. 5.

# 3.1 LLMs and VLMs

We can use the LLM or VLM model to bootstrap the components of the Agent as showed in Fig. 5. In particular, LLMs have been shown to perform well for task-planning (Gong et al., 2023a), contain significant world knowledge (Yu et al., 2023b), and display impressive logical reasoning capabilities (Creswell et al., 2022). Additionally, VLMs such as CLIP (Radford et al., 2021) provide a general visual encoder that is language-aligned, as well as providing zero-shot visual recognition capabilities. For example, state-of-the-art open-source multi-modal models such as LLaVA (Liu et al., 2023c) and InstructBLIP (Dai et al., 2023) rely upon frozen CLIP models as visual encoders.

# 3.2 Agent Transformer Definition

Instead of using frozen LLMs and VLMs for the AI agent, it is also possible to use a single-agent transformer model that takes visual tokens and language tokens as input, similar to Gato (Reed et al., 2022). In addition to vision and language, we add a third general type of input, which we denote as agent tokens. Conceptually, agent tokens are used to reserve a specific subspace of the input and output space of the model for agentic behaviors. For robotics or game playing, this may be represented as the input action space of the controller. When training agents to use specific tools, such as image-generation or image-editing models, or for other API calls, agent tokens can also be used. As showed in Fig. 7, we can combine the agent tokens with visual and language tokens to generate a unified interface for training multi-modal agent AI. Compared to using large, proprietary LLMs as agents, there are several advantages to using an agent transformer. Firstly, the model can be easily customized to very specific agentic tasks that may be difficult to represent in natural language (e.g. controller inputs or other specific actions). Thus, the agent can learn from environmental interactions and domain-specific data to improve performance. Secondly, it can be easier to understand why the model does or does not take specific actions by having access to the probabilities of the agent tokens. Thirdly,

![](images/a13db0a0ddccc937d86de4681ac0e5479ca188f0707c5561ccaf11824478715f.jpg)  
Figure 6: We show the current paradigm for creating multi-modal AI agents by incorporating a Large Language Model (LLM) with a Large Vision Model (LVM). Generally, these models take visual or language inputs and use pre-trained and frozen visual and language models, learning smaller sub-network that connect and bridge modalities. Examples include Flamingo (Alayrac et al., 2022), BLIP-2 (Li et al., 2023c), InstructBLIP (Dai et al., 2023), and LLaVA (Liu et al., 2023c).

![](images/9179fbcef13084f4478b091eb58d495babd522b394164c18e72acbf156ba28c2.jpg)  
Figure 7: The unified agent multi-modal transformer model. Instead of connecting frozen submodules and using existing foundation models as building blocks, we propose a unified and end-to-end training paradigm for agent systems. We can still initialize the submodules with LLMs and LVMs as in Figure 6 but also make use of agent tokens, specialized tokens for training the model to perform agentic behaviors in a specific domain (e.g., robotics). For more details about agent tokens, see Section 3.2

there are certain domains such as healthcare and law that have strict data privacy requirements. Finally, a relatively smaller agent transformer can potentially be significantly cheaper than a larger proprietary language model.

# 3.3 Agent Transformer Creation

As shown above in Fig. 5, we can use the new agent paradigm with LLM and VLM-bootstrapped agents, as well as leveraging data generated from large foundation models to train the agent transformer model for learning to execute specific goals. Within this process, the agent model is trained to be specialized and tailored for specific tasks and domains. This approach allows you to leverage a pre-existing, foundation model’s learned features and knowledge. We show a simplified overview of the process in two steps below:

Define Objectives within the Domain. In order to train the agent transformer, the objectives and the action-space of the agent within the context of each specific environment needs to be clearly defined. This includes determining which specific tasks or actions the agent needs to perform and assigning unique agent tokens for each. Furthermore, any automatic rules or procedures that can be used to identify successful completion of tasks can significantly improve the amount of data available for training. Otherwise, foundation-model generated or human-annotated data will be required

for training the model. After the data is collected and it is possible to evaluate the performance of the agent, the process of continuous improvement can begin.

Continuous Improvement. Continuous monitoring of the model’s performance and collection of feedback are essential steps in the process. Feedback should be used for further fine-tuning and updates. It is also crucial to ensure that the model does not perpetuate biases or unethical outcomes. This necessitates a careful examination of the training data, regular checks for biases in outputs, and, if needed, training the model to recognize and avoid biases. Once the model achieves satisfactory performance, it can be deployed for the intended application. Continuous monitoring remains vital to ensure that the model performs as expected and to facilitate necessary adjustments. More details on this process, sources of training data, and details surrounding continous learning for agent AI can be found in Section 8.

# 4 Agent AI Learning

# 4.1 Strategy and Mechanism

The strategy of interactive AI on different domains which extends the paradigm of calling large foundation models with a trained agent that actively seeks to collect user feedback, action information, useful knowledge for generation and interaction. Some times, the LLM/VLM models are not need to trained again, and we improve their performance by providing improved contextual prompts at test time for an agent. On the other hand, it always involves a knowledge/reasoning/commonsense/inference interactive modeling through a combination of triple systems - one performing knowledge retrieval from multi-model query, second performing interactive generation from the relevant agent, and last one the trained a new, informative self-supervised training or pre-training with reinforcement learning or imitation learning with improved way.

# 4.1.1 Reinforcement Learning (RL)

There is a rich history of leveraging reinforcement learning (RL) to train interactive agents that exhibits intelligent behaviors. RL is a methodology to learn the optimal relationship between states and actions based on rewards (or penalties) received as a result of its actions. RL is a highly scalable framework that has been applied to numerous applications including robotics, however, it generally faces several leader-board and LLM/VLMs have shown their potential to mitigate or overcome some of those difficulties:

• Reward designing The efficiency of policy learning greatly depends on the design of the reward function. Designing the reward function requires not only knowledge of RL algorithms but also a deep understanding of the nature of the task, and thus often necessitates crafting the function based on expert experience. Several studies explored the use of LLM/VLMs for designing reward functions (Yu et al., 2023a; Katara et al., 2023; Ma et al., 2023).   
• Data collection and efficiency Given its exploratory nature, RL-based policy learning requires a significant amount of data (Padalkar et al., 2023). The necessity for extensive data becomes particularly evident when the policy involves managing long sequences or integrating complex actions. This is because these scenarios demand more nuanced decision-making and learning from a wider range of situations. In recent studies, efforts have been directed towards enhancing data generation to support policy learning (Kumar et al., 2023; Du et al., 2023). Additionally, in some studies, these models have been integrated into the reward function to improve policy learning (Sontakke et al., 2023). Parallel to these developments, another strand of research has focused on achieving parameter efficiency in learning processes using VLMs (Tang et al., 2023; Li et al., 2023d) and LLMs (Shi et al., 2023)   
• Long-horizon steps In relation to the issue of data efficiency, RL becomes more challenging as the length of action sequences increases. This is due to the ambiguity in the relationship between actions and rewards, known as the credit assignment problem, and the increase in the number of states to be explored, necessitating a significant amount of time and data. One typical approach for long and complex tasks is to break them down into a sequence of subgoals and apply pretrained policies to solve each subgoal (e.g., (Takamatsu et al., 2022)). This idea falls within the framework called the task and motion planning (TAMP)(Garrett et al., 2021). TAMP is composed of two primary components: task planning, which entails identifying sequences of high-level

actions, and motion planning, which involves finding physically consistent, collision-free trajectories to achieve the objectives of the task plan.

LLMs are well-suited to TAMP, and recent research has often adopted an approach where LLMs are used to execute high-level task planning, while low-level controls are addressed with RL-based policies (Xu et al., 2023; Sun et al., 2023a; Li et al., 2023b; Parakh et al., 2023). The advanced capabilities of LLMs enable them to effectively decompose even abstract instructions into subgoals (Wake et al., 2023c), contributing to the enhancement of language understanding abilities in robotic systems.

# 4.1.2 Imitation Learning (IL)

While RL aims to train a policy based on exploratory behavior and maximizing rewards through interactions with the environment, imitation learning (IL) seeks to leverage expert data to mimic the actions of experienced agents or experts. For example, in robotics, one of the major frameworks based on IL is Behavioral Cloning (BC). BC is an approach where a robot is trained to mimic the actions of an expert by directly copying them. In this approach, the expert’s actions in performing specific tasks are recorded, and the robot is trained to replicate these actions in similar situations. Recent BC-based methods often incorporate technologies from LLM/VLMs, enabling more advanced end-to-end models. For example, Brohan et al. proposed RT-1 (Brohan et al., 2022) and RT-2 (Brohan et al., 2023), transformer-based models that output an action sequence for the base and arm, taking a series of images and language as input. These models are reported to show high generalization performance as the result of training on a large amount of training data.

# 4.1.3 Traditional RGB

Learning intelligent agent behavior leveraging image inputs has been of interest for many years (Mnih et al., 2015). The inherent challenge of using RGB input is the curse of dimensionality. To solve this problem, researchers either use more data (Jang et al., 2022; Ha et al., 2023) or introduce inductive biases into the model design to improve sample efficiency. In particular, authors incorporate 3D structures into the model architecture for manipulations (Zeng et al., 2021; Shridhar et al., 2023; Goyal et al., 2023; James and Davison, 2022). For robot navigation, authors (Chaplot et al., 2020a,b) leverage maps as a representation. Maps can either be learned from a neural network aggregating all previous RGB inputs or through 3D reconstruction methods such as Neural Radiance Fields (Rosinol et al., 2022).

To obtain more data, researchers synthesize synthetic data using graphics simulators (Mu et al., 2021; Gong et al., 2023b), and try to close the sim2real gap (Tobin et al., 2017; Sadeghi and Levine, 2016; Peng et al., 2018). Recently, there has been some collective effort to curate large-scale dataset that aims to resolve the data scarcity problem (Padalkar et al., 2023; Brohan et al., 2023). On the other hand, to improve sample complexity, data augmentation techniques have been extensively studied as well (Zeng et al., 2021; Rao et al., 2020; Haarnoja et al., 2023; Lifshitz et al., 2023).

# 4.1.4 In-context Learning

In-context learning was shown to be an effective method for solving tasks in NLP with the advent of large language models like GPT-3 (Brown et al., 2020; Min et al., 2022). Few-shot prompts were seen to be an effective way to contextualize model output’s across a variety of tasks in NLP by providing examples of the task within the context of the LLM prompt. Factors like the diversity of examples and quality of examples shown for the in-context demonstrations may improve the quality of model outputs (An et al., 2023; Dong et al., 2022). Within the context of multi-modal foundation models, models like Flamingo and BLIP-2 (Alayrac et al., 2022; Li et al., 2023c) have been shown to be effective at a variety of visual understanding tasks when given only given a small number of examples. In context learning can be further improved for agents within environments by incorporating environment-specific feedback when certain actions are taken (Gong et al., 2023a).

# 4.1.5 Optimization in the Agent System

The optimization of agent systems can be divided into spatial and temporal aspects. Spatial optimization considers how agents operate within a physical space to execute tasks. This includes inter-robot coordination, resource allocation, and keeping an organized space.

In order to effectively optimize agent AI systems, especially systems with large numbers of agents acting in parallel, previous works have focused on using large batch reinforcement learning (Shacklett et al., 2023). Since datasets of

multi-agent interactions for specific tasks are rare, self-play reinforcement learning enables a team of agents to improve over time. However, this may also lead to very brittle agents that can only work under self-play and not with humans or other independent agents since they over-fit to the self-play training paradigm. To address this issue, we can instead discover a diverse set of conventions (Cui et al., 2023; Sarkar et al., 2023), and train an agent that is aware of a wide range of conventions. Foundation models can further help to establish conventions with humans or other independent agents, enabling smooth coordination with new agents.

Temporal optimization, on the other hand, focuses on how agents execute tasks over time. This encompasses task scheduling, sequencing, and timeline efficiency. For instance, optimizing the trajectory of a robot’s arm is an example of efficiently optimizing movement between consecutive tasks (Zhou et al., 2023c). At the level of task scheduling, methods like LLM-DP (Dagan et al., 2023) and ReAct (Yao et al., 2023a) have been proposed to solve efficient task planning by incorporating environmental factors interactively.

# 4.2 Agent Systems (zero-shot and few-shot level)

# 4.2.1 Agent Modules

Our foray into the agent paradigm involves the development of Agent AI "Modules" for interactive multi-modal agents using LLMs or VLMs. Our initial Agent Modules facilitate training or in-context learning and adopt a minimalist design for the purposes of demonstrating the agent’s ability to schedule and coordinate effectively. We also explored initial prompt-based memory techniques that facilitate better planning and inform future actions approaches within the domain. To illustrate, our “MindAgent" infrastructure comprises 5 main modules: 1) environment perception with task planning, 2) agent learning, 3) memory, 4) general agent action prediction and 5) cognition, as shown in Figure 5.

# 4.2.2 Agent Infrastructure

Agent-based AI is a large and fast-growing community within the domains of entertainment, research, and industry. The development of large foundation models has significantly improved the performance of agent AI systems. However, creating agents in this vein is limited by the increasing effort necessary to create high-quality datasets and overall cost. At Microsoft, building high-quality agent infrastructure has significantly impacted multi-modal agent copilots by using advanced hardware, diverse data sources, and powerful software libraries. As Microsoft continues to push the boundaries of agent technology, AI agent platforms are poised to remain a dominant force in the world of multimodal intelligence for years to come. Nevertheless, agent AI interaction is currently still a complex process that requires a combination of multiple skills. The recent advancements in the space of large generative AI models have the potential to greatly reduce the current high cost and time required for interactive content, both for large studios, as well as empowering smaller independent content creators to design high quality experiences beyond what they are currently capable of. The current human-machine interaction systems inside multi-modal agents are primarily rule-based. They do have intelligent behaviors in response to human/user actions and possess web knowledge to some extent. However, these interactions are often limited by software development costs to enable specific behaviors in the system. In addition, current models are not designed to help human to achieve a goal in the case of users’ inability to achieve specific tasks. Therefore, there is a need for an agent AI system infrastructure to analyze users behaviors and provide proper support when needed.

# 4.3 Agentic Foundation Models (pretraining and finetune level)

The use of pre-trained foundation models offers a significant advantage in their wide applicability across diverse use cases. The integration of these models enables the development of customized solutions for various applications, circumventing the need for extensive labeled datasets for each specific task.

A notable example in the field of navigation is the LM-Nav system (Shah et al., 2023a), which incorporates GPT-3 and CLIP in a novel approach. It effectively uses textual landmarks generated by the language model, anchoring them in images acquired by robots for navigation. This method demonstrates a seamless fusion of textual and visual data, significantly enhancing the capabilities of robotic navigation, while maintaining wide applicability.

In robot manipulation, several studies have proposed the use of off-the-shelf LLMs (e.g., ChatGPT) while using open vocabulary object detectors. The combination of LLM and advanced object detectors (e.g., Detic (Zhou et al., 2022)) facilitates the understanding of human instruction while grounding the textual information in scenery information (Parakh

et al., 2023). Furthermore, the latest advancements showcase the potential of using prompt engineering with advanced multi-modal models such as GPT-4V(ision) (Wake et al., 2023b). This technique opens avenues for multi-modal task planning, underscoring the versatility and adaptability of pre-trained models in a variety of contexts.

# 5 Agent AI Categorization

# 5.1 Generalist Agent Areas

Computer-based action and generalist agents (GAs) are useful for many tasks. Recent progress in the field of large foundation models and interactive AI has enabled new functionalities for GAs. However, for a GA to become truly valuable to its users, it must be natural to interact with, and generalize to a broad range of contexts and modalities. We high-quality extended main Chapters on Agent foundation AI in Sec.6, especially in areas relevant to the themes in general of these topics:

Multimodal Agent AI (MMA) is an upcoming forum2 for our research and industry communities to engage with each other and with the broader research and technology communities in Agent AI. Recent progress in the field of large foundation models and interactive AI has enabled new functionalities for generalist agents (GAs), such as predicting user actions and task planning in constrained settings (e.g., MindAgent (Gong et al., 2023a), fine-grained multimodal video understanding (Luo et al., 2022), Robotics (Ahn et al., 2022b; Brohan et al., 2023)), or providing a chat companion for users that incorporates knowledge feedback (e.g., website customer support for healthcare systems (Peng et al., 2023)). More details about the representative works and most recent representative works are shown below. We hope to discuss our vision for the future of MAA and inspire future researchers to work in this space. This article and our forum covers the following main topics, but is not limited exclusively to these:

• Primary Subject Topics: Multimodal Agent AI, General Agent AI   
• Secondary Subject Topics: Embodied Agents, Action Agents, Language-based Agents, Vision & Language Agents, Knowledge and Inference Agents, Agents for Gaming, Robotics, Healthcare, etc.   
• Extend Subject Topics: Visual Navigation, Simulation Environments, Rearrangement, Agentic Foundation Models, VR/AR/MR, Embodied Vision & Language.

Next, we present a specific lists of representative agent categories as follows:

# 5.2 Embodied Agents

Our biological minds live in bodies, and our bodies move through a changing world. The goal of embodied artificial intelligence is to create agents, such as robots, which learn to creatively solve challenging tasks requiring interaction with the environment. While this is a significant challenge, important advances in deep learning and the increasing availability of large datasets like ImageNet have enabled superhuman performance on a variety of AI tasks previously thought intractable. Computer vision, speech recognition and natural language processing have experienced transformative revolutions at passive input-output tasks like language translation and image classification, and reinforcement learning has similarly achieved world-class performance at interactive tasks like game playing. These advances have supercharged embodied AI, enabling a growing collection of users to make rapid progress towards intelligent agents can interactive with machine.

# 5.2.1 Action Agents

Action agents refer to the agents that need to execute physical actions in the simulated physical environment or real world. In particular, they need to be actively engaging in activities with the environment. We broadly classify action agents into two different categories based on their application domains: gaming AI and robotics.

In gaming AI, the agents will interact with the game environment and other independent entities. In these settings, natural language can enable smooth communication between agents and humans. Depending on the game, there may be a specific task to accomplish, providing a true reward signal. For instance, in the competitive Diplomacy game, training

a language model using human conversation data along with an action policy with RL enables human-level play (Meta Fundamental AI Research (FAIR) Diplomacy Team et al., 2022).

There are also settings where we agents act as normal residents in a town (Park et al., 2023a), without trying to optimize a specific goal. Foundation models are useful in these settings because they can model interactions that appear more natural by mimicking human behavior. When augmented with external memory, they produce convincing agents that can have conversations, daily schedules, form relationships, and have a virtual life.

# 5.2.2 Interactive Agents

Interactive agents simply refer to agents that can interact with the world, a broader class of agents than action agents. Their forms of interaction do not necessarily require physical actions, but may involve communicating information to users or modifying the environment. For instance, an embodied interactive agent may answer a user’s questions about a topic through dialogue or help users parse through existing information similar to a chatbot. By extending an agent’s capabilities to include information sharing, the core designs and algorithms of Agent AI can be effectively adapted for a range of applications, such as diagnostic (Lee et al., 2023) and knowledge-retrieval (Peng et al., 2023) agents.

# 5.3 Simulation and Environments Agents

An effective approach for AI agents to learn how to act in an environment is to go through trial-and-error experiences via interactions with the environment. A representative method is RL, which requires extensive experience of failures to train an agent. Although there exist approaches that use physical agents (Kalashnikov et al., 2018), using physical agents is time-consuming and costly. Furthermore, training in the physical environment is often feasible when failure in actual environments can be dangerous (e.g., autonomous driving, underwater vehicles). Hence, using simulators to learn policies is a common approach.

Many simulation platforms have been proposed for research in embodied AI, ranging from navigation (Tsoi et al., 2022; Deitke et al., 2020; Kolve et al., 2017) to object manipulation (Wang et al., 2023d; Mees et al., 2022; Yang et al., 2023a; Ehsani et al., 2021). One example is Habitat (Savva et al., 2019; Szot et al., 2021), which provides a 3D indoor environment where human- and robotic-agents can perform various tasks such as navigation, instruction following, and question answering. Another representative simulation platform is VirtualHome (Puig et al., 2018), supporting human avatars for object manipulation in 3D indoor environments. In the field of gaming, Carroll et al. have introduced "Overcooked-AI," a benchmark environment designed to study cooperative tasks between humans and AI (Carroll et al., 2019). Along similar lines, several works aim to incorporate real human intervention beyond the focus of interaction between agents and the environment (Puig et al., 2023; Li et al., 2021a; Srivastava et al., 2022). These simulators contribute to the learning of policies in practical settings involving agent and robot interactions, and IL-based policy learning utilizing human demonstrative actions.

In certain scenarios, the process of learning a policy may necessitate the integration of specialized features within simulators. For example, in the case of learning image-based policies, realistic rendering is often required to facilitate adaptability to real environments (Mittal et al., 2023; Zhong et al., 2023). Utilizing a realistic rendering engine is effective for generating images that reflect various conditions, such as lighting environments. Moreover, simulators employing physics engines are required to simulate physical interactions with objects (Liu and Negrut, 2021). The integration of physics engines in simulation has been shown to facilitate the acquisition of skills that are applicable in real-world scenarios (Saito et al., 2023).

# 5.4 Generative Agents

The recent advancements in the space of large generative AI models have the potential to greatly reduce the current high cost and time required for interactive content, both for large gaming studios, as well as empower smaller independent studios to create high quality experiences beyond what they are currently capable of. Additionally, embedding large AI models within a sandbox environment will allow users to author their own experiences and express their creativity in ways that are currently out of reach.

The goals of this agent go beyond simply adding interactive 3d content to scenes, but also include:

• Adding arbitrary behavior and rules of interactions to the objects, allowing the user to create their own VR rules with minimal prompting.   
• Generating whole level geometry from a sketch on a piece of paper, by using the multimodal GPT4-v model, as well as other chains of models involving vision AI models   
• Retexturing content in scenes using diffusion models   
• Creating custom shaders and visual special effects from simple user prompts

One potential application in the short term is the VR creation of a storyboarding/prototype tool allowing a single user to create a rough (but functional) sketch of an experience/game an order of magnitude faster than currently feasible. Such a prototype then could be expanded and made more polished using these tools as well.

# 5.4.1 AR/VR/mixed-reality Agents

AR/VR/mixed-reality (jointly referred to as XR) settings currently require skilled artists and animators to create characters, environments, and objects to be used to model interactions in virtual worlds. This is a costly process that involves concept art, 3D modeling, texturing, rigging, and animation. XR agents can assist in this process by facilitating interactions between creators and building tools to help build the final virtual environment.

Our early experiments have already demonstrated that GPT models can be used in the few-shot regime inside of the Unity engine (without any additional fine-tuning) to call engine-specific methods, use API calls to download 3d models from the internet and place them into the scene, and assign state trees of behavior and animations to them (Huang et al., 2023a). This behavior likely emerges due to the presence of similar code in open source game repositories that use Unity. Therefore, GPT models are capable of building rich visual scenes in terms of loading in many objects into the scene from a simple user prompt.

The aim of this category of agents is to build a platform and a set of tools that provide an efficient interface between large AI models (both GPT-family ones as well as diffusion image models) and a rendering engine. We explore two primary avenues here:

• Integration of large models into the various editor tools in the agent infrastructure, allowing for significant speedups in development.   
• Controlling the rendering engine from within a user experience, by generating code that follows user instruction and then compiling it at runtime, allowing for users to potentially edit the VR/simulation they are interacting with in arbitrary ways, even by introducing new agent mechanics.

Introducing an AI copilot focused on XR settings would be useful for XR creators, who can use the copilot to complete tedious tasks, like providing simple assets or writing code boilerplate, freeing creators to focus on their creative vision and quickly iterate on ideas.

Furthermore, agents can help users interactively modify the environment by adding new assets, changing the dynamics of the environment, or building new settings. This form of dynamic generation during runtime can also be specified by a creator, enabling the user’s experience to feel fresh and continue evolving over time.

# 5.5 Knowledge and Logical Inference Agents

The capacity to infer and apply knowledge is a defining feature of human cognition, particularly evident in complex tasks such as logical deduction, and understanding theory of mind 3. Making inferences on knowledge ensures that the AI’s responses and actions are consistent with known facts and logical principles. This coherence is a crucial mechanism for maintaining trust and reliability in AI systems, especially in critical applications like medical diagnosis or legal analysis. Here, we introduce agents that incorporate the interplay between knowledge and inference that address specific facets of intelligence and reasoning.

# 5.5.1 Knowledge Agent

Knowledge Agents reason over their acquired knowledge systems in two directions: implicit and explicit. Implicit knowledge is typically what large-scale language models like the GPT series (Brown et al., 2020; OpenAI, 2023) encapsulate after being trained on vast amounts of text data. These models can generate responses that give the impression of understanding, as they draw on patterns and information implicitly learned during training. Explicit knowledge, conversely, is structured and can be directly queried, such as the information found in knowledge bases or databases, which was traditionally used to enhance AI reasoning capabilities by referencing verifiable external resources.

Despite the advancements in language models, their implicit knowledge is static and becomes outdated as the world evolves (Lewis et al., 2020; Peng et al., 2023). This limitation necessitates the integration of explicit knowledge sources that are updated continuously, ensuring that AI systems can provide accurate and current responses. The fusion of implicit and explicit knowledge equips AI agents with a more nuanced understanding and the ability to apply knowledge contextually, akin to human intelligence (Gao et al., 2022). Such integration is crucial for crafting knowledge-centric AI agents that not only possess information but can also understand, explain, and employ it, thereby narrowing the chasm between extensive learning and profound knowledge (Marcus and Davis, 2019; Gao et al., 2020). These agents are designed to reason with flexibility and dynamic information about the world, enhancing their robustness and adaptability (Marcus, 2020).

# 5.5.2 Logic Agents

Generally, a logic agent is a component of a system designed to apply logical reasoning to process data or solve tasks specific to logical inference or logical reasoning. Logic agents within the context of large foundation models like GPT-4 refers to a specialized component or submodules designed to handle logical reasoning tasks. These tasks often involve understanding and manipulating abstract concepts, deducing conclusions from given premises, or solving problems that require a structured, logical approach. Broadly, foundation models like GPT-4 are trained on a vast corpus of text data and learn to perform a wide range of tasks, including those that require some form of logical reasoning. Thus, their capability for logical reasoning is integrated into the overall architecture, and they generally do not possess a distinct, isolated "Logic agent". While GPT-4 and similar models can perform tasks that involve logic, their approach is fundamentally different from how humans or traditional logic-based systems operate. They do not follow formal logical rules or have an explicit understanding of logic; rather, they generate responses based on patterns learned from the training data. As a result, their performance in logical tasks can be impressive, but it can also be inconsistent or limited by the nature of the training data and the inherent limitations of the model’s design. One example of embedding a separate logical submodule into the architecture is (Wang et al., 2023e), which modifies the token embedding process used by LLMs during pre-training by parsing text into logical segments and explicitly modeling logical hierarchies in the token embeddings.

# 5.5.3 Agents for Emotional Reasoning

Emotional understanding and empathy are important skills for agents in many human-machine interactions. To illustrate, one important goal for creating engaging dialogue agents is to have the agents act with increased emotion and empathy while minimizing socially inappropriate or offensive outputs. To advance towards this goal for dialogue agents, we released the Neural Image Commenting with Empathy (NICE) dataset (Chen et al., 2021) consisting of almost two million images and the corresponding human-generated comments and a set of human emotion annotations. We also provided a novel pre-training model - Modeling Affect Gneration for Image Comments (MAGIC) (Chen et al., 2021) - which aims to generate comments for images, conditioned on linguistic representations that capture style and affect, and to help generate more empathetic, emotional, engaging and socially appropriate comments. Our experiments show that the approach is effective in training a more human-like and engaging image comment agent. Developing empathy-aware agents is a promising direction for interactive agents, and it is important to create agents with emotional understanding capabilities across a wide range of groups and populations, especially considering that many current language models exhibit bias in their emotional understanding and empathetic reasoning capabilities (Mao et al., 2022; Wake et al., 2023d).

# 5.5.4 Neuro-Symbolic Agents

Neuro-Symbolic agents operate on a hybrid system of neurons and symbols (d’Avila Garcez and Lamb, 2020). To solve problems stated in natural language is a challenging task because it requires explicitly capturing discrete symbolic structural information implicit in the input. However, most general neural sequence models do not explicitly capture such structural information, limiting their performance on these tasks. The work (Chen et al., 2020) propose a new encoder-decoder model based on a structured neural representation agent, The encoder of TP-N2F employs TPR ‘binding’ to encode natural-language symbolic structure in vector space and the decoder uses TPR ‘unbinding’ to generate, in symbolic space, a sequential program represented by relational tuples, each consisting of a relation (or operation) and a number of arguments.

Instruction following vision-language (VL) models like GPT-4 offer a flexible interface that supports a broad range of multimodal tasks in a zero-shot fashion. However, interfaces that operate on full images do not directly enable the user to “point to” and access specific regions within images. This capability is important not only to support reference-grounded VL benchmarks, but also, for practical applications that require precise within-image reasoning. In (Park et al., 2023b), we build Localized Visual Commonsense model which allows users to specify (multiple) regions-as-input. We train our model by sampling localized commonsense knowledge from a large language model (LLM): specifically, we prompt a LLM to collect common sense knowledge given a global literal image description and a local literal region description automatically generated by a set of VL models. This pipeline is scalable and fully automatic, as no aligned or human-authored image and text pairs are required. With a separately trained critic model that selects high quality examples, we find that training on the localized commonsense corpus expanded solely from images can successfully distill existing VL models to support a reference-as-input interface. Empirical results and human evaluations in zero-shot settings demonstrate that our distillation method results in more precise VL models of reasoning compared to a baseline of passing a generated referring expression.

# 5.6 LLMs and VLMs Agent

A number of works leverage LLMs as agents to perform task planning (Huang et al., 2022a; Wang et al., 2023b; Yao et al., 2023a; Li et al., 2023a), and leverage the LLMs’ large internet-scale domain knowledge and zero-shot planning abilities to perform agentic tasks like planning and reasoning. Recent robotics research also leverages LLMs to perform task planning (Ahn et al., 2022a; Huang et al., 2022b; Liang et al., 2022) by decomposing natural language instruction into a sequence of subtasks, either in the natural language form or in Python code , then using a low-level controller to execute these subtasks. Additionally, (Huang et al., 2022b), (Liang et al., 2022), and (Wang et al., 2023a) also incorporate environmental feedback to improve task performance. There have also been a number of works that demonstrate the ability of general-purpose visually-aligned large language models trained on large-scale text, image, and video data to serve as a foundation for creating multi-modal agents that are embodied and can act in various environments (Baker et al., 2022; Driess et al., 2023; Brohan et al., 2023).

# 6 Agent AI Application Tasks

# 6.1 Agents for Gaming

Games provide a unique sandbox to test the agentic behavior of LLMs and VLMs, pushing the boundaries of their collaborative and decision-making abilities. We describe three areas in particular that highlight agent’s abilities to interact with human players and other agents, as well as their ability to take meaningful actions within an environment.

# 6.1.1 NPC Behavior

In modern gaming systems, the behavior of Non-Player Characters (NPCs) is predominantly dictated by predefined scripts crafted by developers. These scripts encompass a range of reactions and interactions based on various triggers or player actions within the gaming environment. However, this scripted nature often results in predictable or repetitive NPC behavior which fails to evolve in response to player’s actions or the dynamic environment of the game. This rigidity hampers the immersive experience intended in a dynamic gaming environment. Therefore, there is a burgeoning interest in leveraging LLMs to induce autonomy and adaptability in NPC behavior, making interactions more nuanced and engaging. AI-driven NPCs can learn from player behavior, adapt to varying strategies, and provide a more challenging

![](images/d3b5d985769622c7fb99c304b67f9032b157da83501b7f2eae1839f61e00da69.jpg)

![](images/7ac984ba790c2e6d0655d0a50ceaff33e1cf4c20fc3e978407f5323312395eb6.jpg)

![](images/929aa08028236a659952bff31ea63ac5a849304786321f296fb61273dc13c1e0.jpg)  
Figure 8: The embodied agent for user interactive gaming action prediction and interactive editing with Minecraft Dungeons gaming sense simulation and generation via GPT-4V.

and less predictable gameplay experience. Large Language Models (LLMs) can significantly contribute to evolving NPC behavior in games. By processing vast amounts of text, LLMs can learn patterns and generate responses that are more varied and human-like. They can be utilized to create dynamic dialogue systems, making interactions with NPCs more engaging and less predictable. Furthermore, LLMs can be trained on player feedback and in-game data to continually refine NPC behaviors, making them more attuned to player expectations and game dynamics.

# 6.1.2 Human-NPC Interaction

The interaction between human players and NPCs is a crucial aspect of the gaming experience. The conventional interaction paradigm is primarily one-dimensional, with NPCs reacting in a preset manner to player inputs. This limitation stifles the potential for a more organic and enriching interaction, akin to human-human interaction within the virtual realm. The advent of LLM and VLM technologies holds the promise of transforming this paradigm. By employing these technologies, gaming systems can analyze and learn from human behavior to provide more human-like interactions. This not only enhances the realism and engagement of the game but also provides a platform for exploring and understanding human-machine interaction in a controlled yet complex setting.

# 6.1.3 Agent-based Analysis of Gaming

Gaming is an integral part of daily life, estimated to engage half of the world’s population4. Additionally, it exhibits a positive impact on mental health5. However, contemporary game systems exhibit a deficiency in interactions with human players since their behaviors are primarily hand-crafted by game developers. These pre-programmed behaviors frequently fail to adapt to players’ needs. Consequently, there exists a need for new AI systems in games that can analyze player behaviors and furnish appropriate support when necessary. Intelligent interactive systems bear the potential to revolutionize how gamers interact with gaming systems in general. NPCs’ interactions with gamers are no longer confined by the restricted rule sets designed by game developers. They have the potential to adapt seamlessly to gamers’ experiences, providing timely feedback to enrich the gaming experience and elevate the synergy of human-machine interaction.

# GPT-4-Vision

System prompt: You are playing minecraft, please generate future actions.

User prompt: The current action is moving to the right, and moving forward. The goal is to craft a crafting table. Please describe the future actions

# Video input:

![](images/f3b20df755ff538cb107e7f5281f20549b2e983eca62fae5cab1bd3826200431.jpg)

# Output:

Since you’ve collected wood, the next steps to craft a crafting table in Minecraft are as follows:

1. **Open your inventory** (`E` key by default) to bring up the crafting screen.   
2. **Place the wood logs** you’ve collected into one of the crafting slots to turn them into wooden planks. Each log will create 4 wooden planks.   
3. **Collect the wooden planks** that you’ve created by dragging them into your inventory.   
4. **Fill all four crafting slots** in the 2x2 crafting grid (available in your inventory screen) with wooden planks to create one crafting table.   
5. **Collect the crafting table** by dragging it into your inventory.   
6. Close your inventory and **select the crafting table** in your hotbar.   
7. **Right-click on the ground** to place the crafting table.

Once the crafting table is placed, you can interact with it to use the 3x3 crafting grid for creating a wide array of more complex items. That’s the basic process for crafting a crafting table in Minecraft. Now you’re ready to begin crafting more advanced items and tools.

Figure 9: GPT-4V can effectively predict the high-level next actions when given the “action history" and a “gaming target" in the prompt. Furthermore, GPT-4V accurately recognized that the player is holding wooden logs in their hand and can incorporate this perceived information into its plan for future actions. Although GPT-4V appears to be capable of predicting some low-level actions (such as pressing ‘E‘ to open the inventory), the model’s outputs are not inherently suitable for raw low-level action prediction (including mouse movements) and likely requires supplemental modules for low-level action control.

LLMs can serve as a robust tool for analyzing in-game text data, including chat logs, player feedback, and narrative content. They can help in identifying patterns of player behavior, preferences, and interactions which can be invaluable for game developers to improve game mechanics and narratives. Additionally, VLMs can parse through large quantities of image and video data from gaming sessions to help analyze user intent and actions within the game world. Moreover, LLMs and VLMs can facilitate the development of intelligent agents within games that can communicate with players and other agents in a sophisticated and human-like manner, enhancing the overall gaming experience. Beyond LLMs and VLMs, user input data, provides a promising avenue for creating game-playing agents that model perception, game playing, and game understanding by imitating human players. By incorporating a combination of player interactions and feedback, pixel inputs, and natural language planning and understanding, agent models can assist in the continuous improvement of game dynamics, driving a more player-centric evolution of the gaming environment.

# 6.1.4 Scene Synthesis for Gaming

Scene synthesis is a vital component in the creation and enhancement of immersive gaming environments. It entails the automatic or semi-automatic generation of threedimensional (3D) scenes and environments within a game. This process includes the generation of terrain, placement of objects, creation of realistic lighting, and sometimes even dynamic weather systems.

Modern games often feature vast, open-world environments. Manually designing these landscapes can be incredibly time-consuming and resource-intensive. Automated terrain generation, often leveraging procedural or AI-driven techniques, can produce complex, realistic landscapes with less manual effort. LLMs and VLMs can utilize the internet scale knowledge to formulate rules to design non-repeating landscapes that are visually impressive and unique. Additionally, LLMs and VLMs can be used to ensure the semantic consistency and variability of generated assets. Placing objects such as buildings, vegetation, and other elements within a scene in a realistic and aesthetically pleasing manner is crucial for immersion.

VLMs and LLMs can assist in object placement by adhering to predefined or learned rules and aesthetics, thus

speeding up the level design process. VLMs and LLMs can be further trained to understand the principles of design and aesthetics, aiding in the procedural generation of content. They can help formulate rules or guidelines that procedural algorithms can follow to generate objects, and scenes that are both visually appealing and contextually appropriate.

Realistic lighting and atmospheric effects are fundamental for creating a believable and engaging gaming environment. Advanced algorithms can simulate natural lighting conditions and dynamic weather effects, enhancing the realism and mood of the scene. LLMs can help develop systems to acheive more realistic lighting and atmospheric effects in several innovative ways. VLMs can analyze vast datasets from real-world lighting and atmospheric conditions to help develop more realistic algorithms for simulating these effects in games. By understanding the patterns and intricacies of natural lighting and weather, these models can contribute to the development of algorithms that mimic reality closely. LLMs and VLMs could also be used to develop systems that adjust lighting and atmospheric effects in real-time based on player actions, game states, or external inputs. They can process natural language commands from players to modify the game environment, providing a more interactive and immersive experience.

# Unseen Video Prediction Results

![](images/a14cdbe6f31dc419cfaddf9af0cca001fe6972d3b356836fa9317f7346ab3485.jpg)

![](images/8aa0d94576db4d657d518b968e42f35b6ba0c061c9a8935765822f4e01487153.jpg)

![](images/3cca552e19f836c97006dda66c819fe2c25208667637f6ff646193029ae75e97.jpg)

![](images/55bbc3b85d9326b96fa47ba049ac58e1ce10bc97d16617d6da17d538028630ce.jpg)

![](images/b1c816cb2477eef6dcd6cda889ad9922637337f5d758d2e13c604ce7ed68da01.jpg)

![](images/6ff4e3390ef17c78560c5b783efa7a9c99e9ea1194ebd90f2e2ae0fbd8e9bb0a.jpg)

![](images/b8f93f1cacc61e80368f2fb5b4703989345d7c89b920dd65d6ee827aac0a7cdf.jpg)

![](images/527215d6447ae6d17dd1b1d978926997abc210b316449d14822b32af824734c9.jpg)

![](images/0a9a169125574bee67d2d905e70a09ac7079e5412b4f8df0b72c7b7a31bb9181.jpg)

![](images/2bd6f1bbba6958f802ba92923517e0c7e6e202bb7437a6a93b8c6124dbefb4f8.jpg)

![](images/01e1e199975f319b8df3a3c1833646ab9876c0fa13477a81cf2858c0c3c15caa.jpg)

![](images/263e006ff2c19abc3c1ff1e9742bff5e6cbf974219f937d36a953b3f7a5e8b1f.jpg)

![](images/63b91250d879464673cfef0138ef41f094d91fd15da702fdccc8b94e542526e7.jpg)

![](images/b76b0d4a29c4bf0dd299f8542c8b0fe9470b49d798658f0f1295a5d2d0ade7c1.jpg)

![](images/3a36a0c8c24100c88488cfd9c5dc89474063ef02acb4caa96ce1796bc098fbf4.jpg)

![](images/244f70282facf3f6a028f35de7ce600ee3c08f9edf70e7e1a54ee9a01edadb1b.jpg)  
Figure 10: Masked video prediction on unseen Minecraft videos. From left to right: the original frame, the masked frame, the reconstructed frame, and the reconstructed frame with patches.

# 6.1.5 Experiments and Results

Zero-shot/Few-shot Learning with LLM or LVM. As we showed in the Fig. 8 and Fig. 9, we used GPT-4V for high-level description and action prediction. Fig. 8 showed some qualitative examples of action description generation and editing with GPT-4V. Agent-enhanced text opens up a novel method of generating 3D scenes with game action priors to help improve the naturalness of the scene. Consequently, GPT-4V generates relevant high-level descriptions that are appropriate for the gaming videos.

Small Agent Pretraining Model. To showcase our agent vision-language architecture, we first study its application in a widely used domain for gaming agents by pretraining on Minecraft data. As shown in Fig. 7, given an input action agent, key frame of video, and corresponding text, a standard encoder-decoder can be employed to convert the agent action and image into action text token and image patch token and then use the agent-vision-language decoder to convert it into a action prediction sentence. The overall architecture is depicted in Fig. 7. We evaluate our approach with several Minecraft demonstrations. The Minecraft video data consists of 5min clips, and we use for pretraining contains 78K videos, and we used 5K videos $6 \%$ of pretraining data) for the first round pretraining. We train a 250M parameter model on 16 NVIDIA v100 GPUs for one day and visualize our model outputs in Fig. 10 and Fig. 11. Fig. 10 shows that our relatively small agent architecture can produce reasonable outputs for Minecraft scenes unseen during training. Fig. 11 showed the model’s predictions compared to the ground truth human player actions indicating potential low-level understanding for our small agent model.

Multi-Agent Infrastructure. As showed in the agent

paradigm in Fig. 5, we designed a novel infrastructure for a new gaming scenario called “CuisineWorld" (Gong et al., 2023a). We detail our approach in Fig. 12. Our infrastructure allows for multi-agent collaboration by leveraging GPT-4 as a central planner and works across multiple gaming domains. We investigated our system’s multi-agent planning capabilities, and we deployed the infrastructure into real-world video games to demonstrate its multi-agent and human-AI collaboration effectiveness. Additionally, we presented “Cuisineworld", a text-based multi-agent collaboration benchmark that provides a new auto-metric Collaboration Score (CoS) to quantify collaboration efficiency.

Please refer to the Appendix for more examples and details for gaming description, high-level action prediction, and GPT-4V prompting. We show examples for Bleeding Edge in Fig. 32 and Appendix B, Microsoft Flight Simulator in Fig. 33 and Appendix C, ASSASSIN’s CREED ODYSSEY in Fig. 34 and Appendix D, GEARS of WAR 4 in Fig. 35 and Appendix E, and Starfield in Fig. 36 and Appendix F. We also provide a detailed screenshot of the prompting process for GPT4V used to generate Minecraft examples with Fig. 31 in Appendix A.

# 6.2 Robotics

Robots are representative agents that necessitate effective interaction with their environment. In this section, we will introduce key elements essential for efficient robotic operation, review research topics where the latest LLM/VLM technologies have been applied, and share findings from our most recent studies.

Visual Motor Control. Visual Motor Control refers to the integration of visual perception and motor action to execute tasks effectively in a robotic system. This integration is paramount as it enables robots to interpret the visual data from their environment and accordingly adjust their motor actions to interact with the environment accurately. For instance, in an assembly line, a robot equipped with visual motor control can perceive the position and orientation of objects and accurately align its manipulator to interact with these objects. This capability is essential for ensuring the precision and effectiveness of robotic operations across a myriad of applications, ranging from industrial automation to assisting the elderly in their daily chores. Moreover, visual motor control facilitates robots in adapting to dynamic environments

![](images/f057944914dc5651aa6ab002c674870b9156178289ee6c8bab7aaf5ae49bc03d.jpg)  
Figure 11: The low-level next step action prediction with the small agent pretraining model in gaming Minecraft scene.

![](images/b11ad036276b0070d34046d97163b3eeddb887edde46a9ee54a119ad795dd3f9.jpg)  
Figure 12: The MindAgent of in-context learning gaming Infrastructure. Planning Skill and Tool Use: The game environment requires diverse planning skills and tool use to complete tasks. It generates relevant game information and converts the game data into a structured text format that the LLMs can process. LLM: The main workhorse of our infrastructure makes decisions, thus serving as a dispatcher for the multi-agent system. Memory History: A storage utility for relevant information. Action Module: Extracts actions from text inputs and convertd them into domain-specific language and validates DSLs so that they cause no errors during execution.

where the state of the environment may change rapidly, requiring real-time adjustments to motor actions based on visual feedback.

Additionally, within the context of safe operation, visual information is crucial for detecting execution errors and confirming the pre- and post-conditions of each robot action. In uncontrolled environments, such as unknown domestic settings, robots are more likely to face unexpected outcomes due to unpredictable factors like changing furniture shapes, varied lighting, and slippage. Executing a pre-planned action plan solely in a feedforward manner can pose significant risks in these settings. Therefore, utilizing visual feedback to continually verify outcomes at each step is key to ensuring robust and reliable operation of robotic systems.

Language Conditioned Manipulation. Language Conditioned Manipulation entails the ability of a robotic system to interpret and execute tasks based on language instructions. This aspect is particularly crucial for creating intuitive and user-friendly interfaces for human-robot interaction. Through natural language commands, users can specify goals and tasks to robots in a manner similar to human-human communication, thereby lowering the barrier to operating robotic systems. In a practical scenario, for instance, a user could instruct a service robot to “pick up the red apple from the table,” and the robot would parse this instruction, identify the referred object and execute the task of picking it up (Wake et al., 2023c). The core challenge lies in developing robust natural language processing and understanding algorithms that can accurately interpret a wide array of instructions, ranging from direct commands to more abstract directives, and enable the robot to convert these instructions into actionable tasks. Furthermore, ensuring that robots can generalize these instructions across diverse tasks and environments is critical for enhancing their versatility and utility in real-world applications. The use of language input to guide robot’s task planning has gained attention in the context of a robot framework called Task and Motion Planning (Garrett et al., 2021).

Skill Optimization. Recent studies highlight the effectiveness of LLMs in robotic task planning. However the optimal execution of tasks, especially those involving physical interactions like grasping, requires a deeper understanding of the environment that goes beyond simply interpreting human instructions. For example, robot grasping necessitates precise contact points (Wake et al., 2023e) and arm posture (Sasabuchi et al., 2021) to efficiently execute subsequent actions.

While these elements—precise contact points and arm posture—are intuitive for humans, articulating them through language is challenging. Despite advances in internet-scale VLMs, capturing these nuanced indirect cues from scenes and translating them effectively into robotic skills remains a significant challenge. In response, the robotics community is increasingly focusing on collecting enhanced datasets(e.g., (Wang et al., 2023d; Padalkar et al., 2023)) or developing methodologies for direct skill acquisition from human demonstrations (Wake et al., 2021a). Frameworks including Learning-from-Demonstration and Imitation Learning are leading these developments, playing a crucial role in the optimization of physical skills.

# 6.2.1 LLM/VLM Agent for Robotics.

Recent research has demonstrated the potential of LLM/VLMs for robotic agents that involve interactions with humans in an environment. Research topics that aim to leverage latest LLM/VLM technologies include:

Multimodal Systems: Recent research has been actively focusing on developing end-to-end systems that incorporate the latest LLM and VLM technologies as encoders for input information. Particularly, there is a significant trend towards modifying these foundation models to process multimodal information. (Jiang et al., 2022; Brohan et al., 2023, 2022; Li et al., 2023d; Ahn et al., 2022b; Shah et al., 2023b; Li et al., 2023e). This adaptation aims to guide robotic actions based on both linguistic instructions and visual cues, thus achieving an effective embodiment.

Task Planning and Skill Training: In contrast to end-to-end systems, Task And Motion Planning (TAMP) based systems first compute a high-level task plan and then achieve them with low-level robot control, known as skills.

The advanced language processing abilities of LLMs have demonstrated the capability to interpret instructions and decompose them into robot action steps, greatly advancing task planning technologies (Ni et al., 2023; Li et al., 2023b; Parakh et al., 2023; Wake et al., 2023c). For skill training, several studies have explored the use of LLMs/VLMs for designing reward functions (Yu et al., 2023a; Katara et al., 2023; Ma et al., 2023), generating data to facilitate policy learning (Kumar et al., 2023; Du et al., 2023), or serving as part of a reward function (Sontakke et al., 2023). Together with training frameworks such as RL and IL, these efforts will contribute to the development of efficient robot controllers.

On-site Optimization: Executing long task steps in robotics can be difficult due to unexpected and unpredictable environmental conditions. Therefore, a significant challenge in the field of robotics involves dynamically adapting and refining robotic skills by integrating task plans with real-time environmental data. For instance, (Ahn et al., 2022b) proposed an approach that calculates the feasibility of actions (i.e., affordance) from visual information and compares it with planned tasks. Additionally, there are approaches that focus on enabling LLMs to output the pre-conditions and post-conditions (e.g., states of objects and their interrelationships) of task steps to optimize their execution (Zhou et al., 2023c) and detect pre-condition errors for necessary revisions to the task plan (Raman et al., 2023). These strategies seek to achieve environment-grounded robot execution by integrating environmental information and adjusting the robot’s actions at the task plan or controller level.

Conversation Agents: In creating conversational robots, LLMs can contribute to natural, context-sensitive interactions with humans (Ye et al., 2023a; Wake et al., 2023f). These models process and generate responses that mimic human conversation, allowing robots to participate in meaningful dialogues. Additionally, LLMs play a significant role in the estimation of conceptual (Hensel et al., 2023; Teshima et al., 2022) and emotional attributes (Zhao et al., 2023; Yang et al., 2023b; Wake et al., 2023d) of utterances. Those attributes facilitate the understanding of human intent and meaningful gesture generation, thus contributing to the naturalness and efficacy of human-robot communication.

Navigation Agents: Robot navigation has a long history of research, focusing on core aspects such as map-based path planning and Simultaneous Localization and Mapping (SLAM) for creating environmental maps. These functionalities have become standard in widely used robot middleware like the Robot Operating System (ROS) (Guimarães et al., 2016).

While classic navigation techniques remain prevalent in many robotics applications, they typically rely on static or pre-created maps. Recently, there has been an increased interest in advanced technologies that enable robots to navigate in more challenging environments, leveraging breakthroughs in fields like computer vision and natural language processing. One representative task is object navigation (Chaplot et al., 2020a; Batra et al., 2020; Gervet et al., 2023; Ramakrishnan et al., 2022; Zhang et al., 2021), where robots use object names for navigation instead of map coordinates, requiring the visual grounding of object names in the environment. Furthermore, recent attention has been given to

# ChatGPT-empowered task planning

Instruction: “Put the juice on top of the shelf."

```txt
Environmental description:  
{  
    "<spam>": "on_something(<table>)",  
    "<juice>": on_something(<shelf_bottom>)"  
} 
```

![](images/5c3c3f27102e949b32c33db1ed6aab0e9e8f3307a5f5c9049e272728afb79b9a.jpg)

ChatGPT

![](images/9c51cb909c2dc72aec52f77229a6cf6263776562597d1cf9a51a5c6660efc2a5.jpg)

```txt
Task sequence with explanations:   
{ "move_hand(), # move the hand near the juice, "grasp_object(), # grasp the juice "move_object(<juice>), # take the juice from the bottom shelf "move_object(<juice>)", # move the juice upward "put_down_object(<juice>), # place the juice on the shelf "release_object(<juice>" # release the juice   
} 
```

# Skill parameterization from demonstration

![](images/606f152d6ec96e5644c4a21c835bf04bfd0ff32e4e4c986825914c4a71e2db2a.jpg)  
Visual demonstration

![](images/4f9b68d42fe88300020868e0bd2b9e0cd1ba9a1c1059967a26321ab39338eda5.jpg)

![](images/cea4f4c270c60671594e9829d869a2c8bdfe11e01f26fbf24d8cf10385aa6cf5.jpg)

Vision analyzer

# Parameterized tasks:

“grasp_object(<juice>)":

{ 3D positionof the object:[x,y,z]

Handless of the manipulation: “right"

Approach direction to the object: [x,y，z]

Grasp type:“power”,

}

![](images/0706903333f16c2b77cc15be4f98762203bdf9917dec8539cdf4db5b3b12dbe8.jpg)

![](images/a88abdf417c7c91afcebcf0c2f8e7f90423ebeb1c673e93fc7ee2189d6d6c2ff.jpg)  
Figure 13: Overview of the robot teaching system that integrates a ChatGPT-empowered task planner. The process involves two steps: Task planning, where the user employs the task planner to create an action sequence and adjusts the result through feedback as necessary, and Demonstration, where the user visually demonstrates the action sequence to provide information needed for robot operation. The vision system collects visual parameters that will be used for robot execution.

technologies that navigate robots in entirely unfamiliar new environments on a zero-shot basis, on top of foundation models, so-called zero-shot object navigation (Gadre et al., 2023; Dorbala et al., 2023; Cai et al., 2023). Additionally, Vision-Language Navigation (VLN) (Anderson et al., 2018a) is a representative task, where the task involves navigating an agent by natural language instructions in previously unseen, real-world environments (Shah et al., 2023a; Zhou et al., 2023a; Dorbala et al., 2022; Liang et al., 2023; Huang et al., 2023b). VLN interprets sentences rather than object names, such as “go to the bathroom on your left.,” thus it requires a higher functionality to parse input text (Wang et al., 2019).

The advent of foundation models contributes to the development of such adaptive, on-the-fly navigation technologies by enhancing the understanding of human language instructions and the visual interpretation of environmental information. More detailed explanations of representative VLN research are provided in 6.2.2.

# 6.2.2 Experiments and Results.

An accumulating body of evidence suggests that recent VLMs and LLMs have promising capabilities for symbolic task planning (e.g., what-to-do). However, each task requires low-level control policy (e.g., how-to-do) to achieve successful interaction between the environment. While reinforcement learning and imitation learning are promising approach to learn policies in a data-driven manner, another promising approach is to obtain the strategy directly from humans through on-site demonstration, an approach called Learning-from-Observation (Wake et al., 2021a; Ikeuchi et al., 0). In this section, we introduce a study where we employ ChatGPT for task planning and enrich the plan by parameterizing it with affordance information to facilitate effective and precise execution (Fig. 13).

The pipeline was composed of two modules: task planning and parameterization. In task planning, the system is fed with language instructions and the description of the working environment. These instructions, along with a predefined set of robot actions and output specifications, are compiled into a comprehensive prompt provided to ChatGPT, which then generates a sequence of decomposed tasks with their textual descriptions (Fig. 13; left pane). Notably, we employ a few-shot approach, meaning ChatGPT is not trained on this task, offering an advantage in applicability as it eliminates the need for hardware-dependent data collection and model training. Additionally, the textual descriptions in the output

# "take the pie on the table and warm it using the stove"

1．"Walktowards(<kitchentable_231>)",   
2．"Grab(<pie_320>)",   
3."Walktowards(<stove_312>)",   
4."Put (<pie_320>,<stove_312>)",   
5."Close(<stove_312>)

1."Walktowards(<kitchentable ${ 2 3 1 > } ) "$   
2."Grab(<pie_320>)",   
3."Walktowards(<stove $_ { - } 3 1 2 > ) "$   
4."Open(<stove_312>)",   
5."Putln(<pie_320>,<stove_312>)",   
6."Close(<stove_312>)"

1．"Walktowards(<kitchentable_231>)",   
2."Grab(<pie_320>)”   
3."Walktowards(<stove_312>)",   
4."Open(<stove_312>),   
5."Putin(<pie_320>,<stove_312>)”,   
6."Close(<stove_312>)   
7."SwitchOn(<stove_312>)"

1st feedback

"You arewrong! Modify your   
answer. You need to open and   
close an openable object when you   
'putin'something intoit."

2nd feedback

"Youarewrong! Modify your

answer. Do not forget to turn on the

switch in the end."

![](images/80fa9d4106d3d2bdac6aa0af074e5e5abf83f21dfe4dcd89a1217536a3252be9.jpg)

![](images/361c54773479afb28116970b92b8d5e78a441bfa7f587310537a2416650cf7d3.jpg)

![](images/0a5e99e7a3d19ce4cb7aaf2425bc184977eee411fd35817ac43745b4d13c3d08.jpg)

![](images/f36185247ab1cb48f1ac881d633f552d93c92b1a268e0f0a6d1faaf7f56e5807.jpg)

![](images/c1d540736184fc393ed6213531d0c73cf258a3e9313f53e41146c63f8799797a.jpg)

![](images/8380576ee9e199ace52fe963e819804abd66f23c243acf306ff291ca77c0006d.jpg)

![](images/24c979213f5f08b38cce5c123176261f22df813fa12b44a6e949be1ed9b55399.jpg)  
Figure 14: Example of adjusting an output sequence through auto-generated feedback. We use an open-sourced simulator, VirtualHome for the experiment. Given an instruction “Take the pie on the table and warm it using the stove.,” the task planner plans a sequence of functions that are provided in VirtualHome. If an error in execution is detected, the task planner correct its output based on the auto-generated error message.

enable the user to check and adjust the results as necessary, which is a crucial feature for a safe and robust operation. Fig. 14 shows the qualitative results conducted for an agentic simulation on top of VirtualHome (Puig et al., 2018). The results demonstrate a reasonable task plan and its flexibility in adjusting outputs, indicating the broad applicability of our approach.

While the task planner guarantees coherency between the task sequences, successful operation in reality requires detailed parameters. For example, grasp type is crucial for carrying a container while spilling out the content, such a parameter is often ignored in a simulators (see Fig. 14 in grasping a pie). In our robot system, therefore, users are asked to demonstrate each action visually (Fig. 13; right pane). The tasks had predefined parameters necessary for execution, which our vision system extracts from the videos (Wake et al., 2021b). Notably, our robotic system is not designed for exact replication of human motions (i.e., teleoperation) but rather to handle varying real-world conditions, such as changes in object locations. Hence, the parameters extracted from human demonstrations encompass not precise motion paths but affordance information that dictates effective environmental movement (e.g., waypoints for collision avoidance (Wake et al., 2023a), grasp types (Wake et al., 2023e), and upper-limbs postures (Sasabuchi et al., 2021; Wake et al., 2021a)). The posture of the upper limbs is critical in robots with high degrees of freedom and is designed to assume predictable postures for humans coexisting with the operational robot. The task sequence endowed with affordances is transformed into a sequence of reusable robot skills acquired through reinforcement learning and executed by the robot (Takamatsu et al., 2022).

LLM-empowered task planning can be extended to a more versatile robotic system by integrating it with VLMs. Here, we show an example where we use the GPT-4V(ision) to broaden the aforementioned task planner in a multimodal input context (Fig. 15), a human performs actions that are intended to be replicated by the robot. In this paper, only part of the prompt is shown. The whole prompt is available at microsoft.github.io/GPT4Vision-Robot-Manipulation-Prompts.

This pipeline takes demonstration videos and text, then outputs a sequence of robot actions. A vision analyzer aims to understand the actions performed by humans in the video. We used GPT-4V and provided a prompt to generate text instructions in a style typical of human-to-human communication.Fig. 16 demonstrates how the usage of text

![](images/f351bf4b48eb8ad872d5cb7d95097b01b7f3e7618112948ff4d2ae0a4e7754d5.jpg)  
Figure 15: Overview of the multimodal task planner that leverages GPT-4V and GPT-4. The system processes video demonstrations and text instructions, generating task plans for robotic execution.

![](images/334db7a40651219e3cb5d4c530536fc0302de10cb60dd94d453e1447da1f131a.jpg)  
Reach for the can on the table, grab it,and then place it on the tray nearby.

![](images/332ca627eeef3af4ebc3a259cfdec2972cb15e66be5331485f351b4f720bb217.jpg)  
Please move your hand to the fridge handle, grab it, and open the fridge door.   
Figure 16: Examples of the output of the video analyzer. The five frames are extracted at regular intervals and fed into GPT-4V. We describe the entire pipeline in Section 6.2.2.

input allows user to give feedback on GPT-4V’s recognition results for correction purposes. Such a feature, aiming at improving the accuracy of the recognition results, also enables more robust operation.

Next, the scene analyzer compiles the expected work environment into the text information based on the instructions and the first frame of the video data (or an image of the environment). This environmental information includes a list of object names recognized by GPT-4V, the graspable properties of objects, and the spatial relationships between objects. Although these computational processes are a black box within GPT-4V, the information is output based on the knowledge of GPT-4V and the image/text input. Fig. 17 shows the example outputs of our scene analyzer. As shown in the figure, GPT-4V successfully selects the objects that are related to the manipulation. For example, a table is included in the output when the human is relocating a spam container on the table, while the table is ignored for the fridge opening task. These results suggest that the scene analyzer encodes the scene information with respect to the human’s actions. We prompted GPT-4V to explain the results of the object selection process and the reasons behind those choices. In practice, we found this approach resulted in reasonable outputs. Finally, based on the given text instructions and environmental information, the task planner outputs a sequence of tasks (Wake et al., 2023c).

Instruction: “Reach for the can on the table,grab it,and then place it on the tray nearby.

![](images/c2f8ddeb83fcf5b9a2b0a8744f195d22d7a8def7f9fd9c2a5902d37c1f115004.jpg)

```txt
"objects": ["<can>" , "<table>", "<tray>", "object_propertyss": { "<can>": ["GRABBABLE"], "<table>": [], "<tray": [] 
```

```txt
},
"spatial Relations": {
<can>": ["on(<table>")",
<table>": [], 
<tray>": ["on(<table>")"]
}, 
```

"your_explanation": "The human is expected to grab the can that is on the table and place it on the tray.The table and the tray are included as they are the surfaces involved in this action. The can is marked as GRABBABLE because it is the item being manipulated.Other objects and details in the environment not related to the action,such as the human,the background elements,and the clothing,are omitted because they do not influence the task of moving the can from table to tray."

Instruction: “Please move your hand to the fridge handle, grab it, and open the fridge door."

![](images/840d23b4dc3b2c266c7b4de51122c088d1441a4a974381ccefbd6a002c722729.jpg)  
Figure 17: Examples of the outputs of the scene analyzer that leverages GPT-4V. We describe our entire pipeline in Section 6.2.2.

```python
"objects": ["<fridge>", "<fridge_handle>]". "object_propertyss": { "<fridge>": [], "<fridge_handle>: ["GRABBABLE"] 
```

```txt
"spatial Relations":{ "<fridge_handle>":"on(<fridge>)", "<fridge>":[] } 
```

"your_explanation": "Human is moving their hand to the fridge handle, grabbing it,and opening the fridge door. Objects included are the fridge and fridge handle, as they are directly involved in the action. lomitted the computer monitor and other unmanipulated objects in the setting because they are not being interacted with for this task."

Embodied Agents for Robotics Navigation. Vision-language navigation (VLN) is the task of navigating an embodied agent to carry out natural language instructions inside real 3D environments. Navigation in 3D environments (Zhu et al., 2017a; Mirowski et al., 2016; Mousavian et al., 2018; Hemachandra et al., 2015) is an essential capability of a mobile intelligent system that functions in the physical world. In the past few years, a plethora of tasks and evaluation protocols (Savva et al., 2017; Kolve et al., 2017; Song et al., 2017; Xia et al., 2018; Anderson et al., 2018a) have been proposed as summarized in (Anderson et al., 2018b). VLN (Anderson et al., 2018a) focuses on language-grounded navigation in the real 3D environment. In order to solve the VLN task, (Anderson et al., 2018a) set up an attention-based sequence-to-sequence baseline model. Then (Wang et al., 2018) introduced a hybrid approach that combines model-free and model-based reinforcement learning (RL) to improve the model’s generalizability. Lastly, (Fried et al., 2018) proposed a speaker-follower model that adopts data augmentation, a panoramic action space and modified beam search for VLN, establishing the current state-of-the-art performance on the Room-to-Room dataset. Extending prior work, we propose a Reinforced Cross-Modal Matching (RCM) for VLN in (Wang et al., 2019). The RCM model is built upon (Fried et al., 2018) but differs in many significant aspects: (1) RCM combines a novel multi-reward RL with imitation learning for VLN while Speaker-Follower models (Fried et al., 2018) only uses supervised learning as in (Anderson et al., 2018a). (2) The RCM reasoning navigator performs cross-modal grounding rather than the temporal attention mechanism on single-modality input. (3) The RCM matching critic is similar to the Speaker in terms of the architecture design, but the former is used to provide the cycle-reconstruction intrinsic reward for both RL and SIL training while the latter is used to augment training data for supervised learning. In (Wang et al., 2019), we study how to address three critical leader-board for this task: the cross-modal grounding, the ill-posed feedback, and the generalization problem. As shown in Fig. 18, we propose a novel Reinforced Cross-Modal Matching approach that enforces cross-modal

# Instruction

Turn right and head towards the kitchen. Then turn left, pass a table and enter the hallway. Walk down the hallway and turn into the entry way to your right without doors. Stop in front of the toilet.

Local visual scene

![](images/1c4240372dcf68d90b2d5a87fe9002d323c47b8deffe0a0a07d0e12d0ba65074.jpg)

![](images/24506925d3e469b8b93b441889aedd4be3898b3328d7d6b5cf9ca6e7312c17fb.jpg)

![](images/79f57f92871abbc46a8c2aa02c0c8512b31a091ce15518623bb2481a72d57945.jpg)

![](images/edae18b7d3f21f97548a71ae409d410cb09a2e78c3071a7c7cc876c05ce488ca.jpg)

![](images/8e13dd9bd39fbed194a24d1b0530cfacd129ee32c351799a4b54dfc538bbd0ce.jpg)

Initial	Position

Target	Position

![](images/d87defbb9088a3fc4cee071a6463619bce952f101e519c91f2eca7bc790b3d5e.jpg)  
Figure 18: Demonstration of embodied agent for the VLN task (Wang et al., 2019). The instruction, the local visual scene, and the global trajectories in a top-down view is shown. The agent does not have access to the top-down view. Path A is the demonstration path following the instruction. Path B and C are two different paths executed by the agent.

Demonst

Executed

Executed

grounding both locally and globally via reinforcement learning (RL). Particularly, a matching critic is used to provide an intrinsic reward to encourage global matching between instructions and trajectories, and a reasoning navigator is employed to perform cross-modal grounding in the local visual scene. Evaluation on a VLN benchmark dataset shows that our RCM model significantly outperforms previous methods by $10 \%$ on SPL and achieved a new state-of-the-art performance. To improve the generalizability of the learned policy, we further introduce a Self-Supervised Imitation Learning (SIL) method to explore unseen environments by imitating its own past, good decisions. We demonstrate that SIL can approximate a better and more efficient policy, which tremendously minimizes the success rate performance gap between seen and unseen environments (from $3 0 . 7 \%$ to $1 1 . 7 \%$ ). Moreover, in (Wang et al., 2019) we introduce a self-supervised imitation learning method for exploration in order to explicitly address the generalization issue, which is a problem not well-studied in prior work. Concurrent to the work, (Thomason et al., 2018; Ke et al., 2019; Ma et al., 2019a,b) studies the VLN tasks from various aspects, and (Nguyen et al., 2018) introduces a variant of the VLN task to find objects by requesting language assistance when needed. Note that we are the first to propose to explore unseen environments for the VLN task.

# 6.3 Healthcare

In healthcare, LLMs and VLMs can act as diagnostic agents, patient care assistants, or even therapy aids, but they come with unique leader-board and responsibilities. With the tremendous potential for AI agents to improve patient care and save lives comes an equally dangerous possibility that their misuse or hasty deployment could endanger thousands or millions of people worldwide. We discuss some of the promising routes for AI agents within the context of healthcare and also discuss some of the key leader-board faced.

Diagnostic Agents. Using LLMs as medical chatbots for patient diagnosis has recently attracted great attention due to the high-demand for medical experts and the potential for LLMs to help triage and diagnose patients (Lee et al., 2023). Dialogue agents, especially those that can effectively communicate important medical information to a broad range of people from diverse patient populations, have the potential to provide equitable healthcare access to historically disadvantaged or marginalized groups. Furthermore, doctors and healthcare systems across the world are largely over-burdened and under-resourced, resulting in insufficient access to medical care for hundreds of millions of people worldwide (World Health Organization and World Bank, 2015). Diagnostic agents provide a particularly advantageous pathway to improve healthcare for millions since they have they can be built with the capability to understand a variety

of languages, cultures, and health conditions. Initial results have shown that healthcare-knowledgeable LMMs can be trained by utilizing large-scale web data (Li et al., 2023f). Although an exciting direction, the promise of diagnostic agents does not come without risks. We highlight the risks of hallucination within medical contexts, as well as potential pathways for solutions in the following section.

Knowledge Retrieval Agents. Within the medical context, model hallucinations are particularly dangerous and may even result in serious patient harm or death, depending on the severity of the error. For instance, if a patient mistakenly receives a diagnosis suggesting they are free of a condition they actually have, it can lead to catastrophic outcomes. These include postponed or inappropriate treatments, or in some cases, a total lack of necessary medical intervention. The gravity of undiagnosed or misdiagnosed conditions can lead to escalated healthcare expenses, extended therapies causing further physical strain, and in extreme scenarios, severe harm or even death. Thus, approaches that can use agents to more reliably retrieve knowledge (Peng et al., 2023) or generate text in a retrieval-based manner (Guu et al., 2020) are promising directions. Pairing a diagnostic agent with a medical knowledge retrieval agent has the potential to significantly reduce hallucinations while simultaneously improving the quality and preciseness of the responses of the diagnostic dialogue agent.

Telemedicine and Remote Monitoring. Agent-based AI also has great potential within the world of Telemedicine and Remote Monitoring by improving the access to healthcare, improving communications between healthcare providers and patients, as well as improving the efficiency and reducing the costs of frequent doctor-patient interactions (Amjad et al., 2023). Primary care clinicians spend significant amounts of time sifting through patient messages, reports, and emails that are often irrelevant or unnecessary for them to view. There is significant potential to allow for support agents to help triage messages from doctors, patients, and other healthcare providers and to help highlight important messages for all parties. By enabling agentic AI systems to coordinate with patients, clinicians, and other AI agents, there is a massive potential to revolutionize the remote healthcare and digital health industry.

# 6.3.1 Current Healthcare Capabilities

Image understanding. We demonstrate the current capabilities and limitations of modern multimodal agents such as GPT-4V within the context of healthcare in Fig. 19. We can see that although GPT-4V possesses significant internal knowledge of the equipment and procedures involved in hospital care, it does not always respond to more prescriptive or diagnostic queries by the user.

Video understanding. We investigate the performance of VLM agents for medical video understanding in two contexts. First, we investigate the ability for VLM agents to identify important patient care activities in clinical spaces. Secondly, we explore the usage of of VLMs for more technical videos such as ultrasounds. Specifically, in Figure 20, we demonstrate some of the current capabilities and limitations of GPT-4V for hospital care and medical video analysis.

# 6.4 Multimodal Agents

The integration of visual and linguistic understanding is crucial for developing sophisticated multimodal AI agents. This includes tasks such as image captioning, visual question answering, video language generation, and video understanding, amongst others. We aim to delve into these visual-language tasks, exploring the leader-board and opportunities they present in the context of AI agents.

# 6.4.1 Image-Language Understanding and Generation

Image-language understanding is a task that involves the interpretation of visual content in a given image with language and the generation of associated linguistic descriptions. This task is critical to the development of AI agents that can interact with the world in a more human-like manner. Some of most popular ones are image captioning (Lin et al., 2014; Sharma et al., 2018; Young et al., 2014; Krishna et al., 2016), referring expression (Yu et al., 2016; Karpathy et al., 2014), and visual question answering (Antol et al., 2015; Ren et al., 2015; Singh et al., 2019).

More recently, knowledge-intensive Visual Question Answering tasks such as OKVQA (Marino et al., 2019), KB-VQA (Wang et al., 2015), FVQA (Wang et al., 2017), and WebQA (Chang et al., 2021) have been introduced. Multimodal

![](images/7f95beacce550febb72935f22ae385966c07a78e8520b44db7a2698eb5bf548e.jpg)  
Figure 19: Example prompts and responses when using GPT-4V within the domain of healthcare image understanding. From left to right: (1) an image of a nurse and doctor conducting a CT scan, (2) a synthetic image of an irregular EKG scan, and (3) an image from the ISIC (Codella et al., 2018) skin lesion dataset. We can see that GPT-4V possesses significant medical knowledge and is able to reason about medical images. However, due to safety training, it is unable to make diagnoses for some medical images.

agents should capable of identifying objects in an image, comprehending their spatial relationships, generating accurate descriptive sentences about the scene, and utilizing reasoning skills to handle knowledge-intensive visual reasoning. This requires not just object recognition capabilities, but also a deep understanding of spatial relationships, visual semantics, and the ability to map these visual elements to linguistic constructs with integration of the world knowledge.

# 6.4.2 Video and Language Understanding and Generation

Video-language generation. Video captioning or video storytelling is the task of generating a sequence of coherent sentences for a stream of video frames. Inspired by the successful use of recurrent large foundation models employed in video and language tasks, variants of agent driven enhanced models have shown promising results on the task of video-lanaguage generation. The fundamental challenge is that the strong performance of neural encoder-decoder models does not generalize well for visual storytelling, because the task requires a full understanding of the content of each image as well as the relation among different frames. One important goal for the field is to create an agent-aware text-synthesis model that can efficiently encode the sequence of frames and generate a topically coherent multi-sentence paragraph.

Video Understanding. Video understanding extends the scope of image understanding to dynamic visual content. This involves interpretation and reasoning about the sequence of frames in a video, often in conjunction with accompanying audio or textual information. An agent should be able interact with various modalities from visual, text, and also audio modalities to demonstrate their advanced comprehension of video content. Tasks in this domain include video captioning, video question answering, and activity recognition, amongst others. The leader-board in video understanding are manifold. They include the temporal alignment of visual and linguistic content, the handling of long sequences of frames, and the interpretation of complex activities that unfold over time. Regarding audio, the agent could process spoken words, background noises, music, and tone of voice to comprehend the mood, setting, and subtleties of the video content.

![](images/b06bdeb83bb32ffae2b50a5cae9d94895e8e38b6ff9c3e0811d35818d16a238d.jpg)  
Figure 20: Example prompts and responses when using GPT-4V within the domain of healthcare video understanding. We input the example videos as $2 \mathbf { x } 2$ grids with overlaid text indicating the order of frames. In the first two examples, we prompt GPT-4V to examine the frames in the video to detect the clinical bedside activities performed on the volunteer patients. For the final example, we attempt to prompt GPT-4V to assess an echocardiogram video, however due to GPT-4V’s safety training, it does not provide a detailed response. For clarity, we bold text that describes the activity of interest, and abbreviate model responses that are unnecessary. We gray-out faces from the individuals to preserve their privacy.

![](images/94105adcf58d524f3b7f48bf29453fae260706729549b3952355caf60f64c748.jpg)  
Figure 21: Interactive multimodal agents include four main pillars: Interaction, Speech, Vision, and Language. Co-pilot agents are made up of different services. 1) Interaction services help make a unified platform for automated actions, cognition, and decision-making. 2) Audio services integrate audio and speech processing into apps and services. 3) Vision services identify and analyze content within images, videos, and digital ink. 4) Language services extract meaning from structured and unstructured text.

Previous works have focused on employing existing video-language training data available online for establishing video foundational models (Li et al., 2020, 2021b; Fu et al., 2022; Bain et al., 2021; Zellers et al., 2021, 2022; Fu et al., 2023). Supporting such training pipelines and functionalities is, however, difficult due to the limited and often inconsistent nature of these datasets. Video foundational models are designed with masked and contrastive pretraining objectives and later tuned on their respective tasks. Despite showing remarkable results in multimodal benchmarks, these models encounter difficulties in video-only tasks such as action recognition due to their dependency on limited video-text data built from noisy audio transcriptions. This limitation also leads to the lack of robustness and fine-grained reasoning skills that large language models generally possess.

Other methods, similar to those used in image-language understanding, have drawn on the strong reasoning skills and broad knowledge of large language models to improve different facets of video interpretation. The task of video understanding is simplified by language only models like ChatGPT and GPT4 or image-language models like GPT4-V, which treat the audio, video, and language modalities as individual interpretable input data types and position the agents as strong open-source models. For example, (Huang et al., 2023c; Li et al., 2023g) transformed video understanding into a natural language processing (NLP) question-answering formulation by textualizing video content with open-source vision classification/detection/caption models. (Lin et al., 2023) integrated GPT4-V with specialized tools in vision, audio, and speech, to facilitate complex video understanding tasks, such as scripting character movements and actions in long-form videos.

Parallel research explores generating scaled datasets from large models, then applying visual instruction tuning (Liu et al., 2023c; Li et al., 2023c; Zhu et al., 2023) on the generated data. Considerable audio, speech, and visual expert perception models are subsequently used to verbalize videos. Speech is transcribed with automatic speech recognition tools, and video descriptions and related data are produced with various tagging, grounding, and captioning models (Li et al., $2 0 2 3 \mathrm { g }$ ; Maaz et al., 2023; Chen et al., 2023; Wang et al., 2023f). These techniques demonstrate how instruction tuning video-language models on generated datasets may lead to enhanced video-reasoning and communication abilities.

# 6.4.3 Experiments and Results

• Knowledge-Intensive Models: As introduced in INK (Park et al., 2022), and KAT (Gui et al., 2022a), an intensive neural knowledge task that incorporates required knowledge annotated by humans to support knowledge-intensive retrieval task.   
• Multimodal-Agents: There has been a growing interest in multimodal language models like Chameleon (Lu et al., 2023) and MM-React (Yang et al., 2023c).   
• Visual Instruction Tuning: VCL(Gui et al., 2022b), Mini-GPT4 (Zhu et al., 2023), MPLUG-OWL (Ye et al., 2023b), LSKD (Park et al., 2023c) generate image-level instruction tuning dataset.

Knowledge-Intensive Agent. As showed in Fig. 22 and Fig. 23, Knowledge-based visual question answering and vision-language retrieval tasks are challenging tasks in multi-modal machine learning that requires outside knowledge beyond image contents. Recent studies on large-scale transformers have primarily focused on maximizing the efficiency of the model’s parameters to store information. This line of research explores a different aspect: whether multimodal transformers can use explicit knowledge in their decision-making process. Pretraining methods based on transformers have shown remarkable success in implicitly learning knowledge representations across multiple modalities. However, traditional methods, mainly unimodal, have investigated knowledge retrieval and subsequent answer prediction, raising questions about the quality and relevance of the knowledge retrieved and the integration of reasoning processes using both implicit and explicit knowledge. To tackle these issues, we introduce the Knowledge Augmented Transformer (KAT), which outperforms others by $6 \%$ on the 2022 OK-VQA open-domain multimodal task. KAT combines implicit knowledge from GPT3 with explicit knowledge from websites using an encoder-decoder structure, and allows for concurrent reasoning with both knowledge types during answer generation. Furthermore, incorporating explicit knowledge enhances the interpretability of the model’s predictions. The code and pre-trained models are available at https://github.com/guilk/KAT.

Vision-language Transformer Agent. Next, we introduce the "Training Vision-Language Transformers from Captions" (VLC) model (Gui et al., 2022b), a transformer that has been pretrained exclusively with image-caption pairs. Despite using just a simple linear projection layer for image embeddings, VLC attains competitive results across various vision-language tasks, in contrast to other methods that depend on object detectors or supervised CNN/ViT networks.

![](images/bba755baca5b83459e90ac30810f086490b329428c0d05b11c0dd8264fe51859.jpg)  
Figure 22: Example of Intensive Neural Knowledge (INK) (Park et al., 2022) task that uses knowledge to identify text relevant to the image from a set of text candidates. Our task involves leveraging visual and text knowledge retrieved from web and human-annotated knowledge.

![](images/7283019d6a57119b3ab20983b4ab0f72cb2c4774165c9d210b83625d63f4db6d.jpg)  
Figure 23: The KAT model (Gui et al., 2022a) uses a contrastive-learning-based module to retrieve knowledge entries from an explicit knowledge base and uses GPT-3 to retrieve implicit knowledge with supporting evidence. The integration of knowledge is processed by the respective encoder transformer and jointly with reasoning module and the decoder transformer via end-to-end training for answer generation.

![](images/b41b0e8ca79d81c77001d52f0bcdb20823534a45a908f33847add0b6beb384a7.jpg)  
Figure 24: The overall architecture of the VLC model (Gui et al., 2022b). Our model consists of three modules: (1) Modality-specific projection. We use a simple linear projection to embed patched images and a word embedding layer to embed tokenized text; (2) Multi-modal encoder. We use a 12-layer ViT (Dosovitskiy et al., 2021) initialized from MAE (He et al., 2022) (ImageNet-1K without labels) as our backbone; (3) Task-specific decoder. We learn our multi-modal representations by masked image/language modeling and image-text matching which are only used during pre-training. We use a 2-layer MLP to fine-tune our multi-modal encoder for downstream tasks. Importantly, we find that the masked image modeling objective is important throughout second-stage pre-training, not only for initialization of the visual transformer.

Through extensive analysis, we explore the potential of VLC as a vision-language transformer agent. For instance, we show that VLC’s visual representations are highly effective for ImageNet-1K classification, and our visualizations confirm that VLC can accurately match image patches to corresponding text tokens. The scalability of performance with more training data highlights the promising potential for developing large-scale, weakly-supervised, open-domain vision-language models.

# 6.5 Video-language Experiments

To understand the practicality of converting pre-trained image-LLMs for video understanding, we temporally expand and fine-tune InstructBLIP (Dai et al., 2023) for video captioning. Specifically, we expand the visual encoder of InstructBLIP (EVA-CLIP-G (Sun et al., 2023b)) using the same divided space-time attention scheme as Frozen in Time (Bain et al., 2021) and keep the Q-former and LLM (Flan-T5-XL (Chung et al., 2022)) frozen during training. We freeze all spatial layers of the visual encoder, while keeping the temporal layers unfrozen during captioning training. This allows for our model to take image and videos as input (matching the image-level performance of InstructBLIP). We train on a 5 million video-caption subset of WebVid10M (Bain et al., 2021). We visualize two example outputs in Figure 25. However, existing agents fail to fully comprehend precise, fine-grained visual details in the video content. A similar limitation is seen by visual instruction tuning methods, where they lack the general, human-level perception abilities that are remain to be solved by multimodal models and agents.

The instruction-tuned models show promise in accurately summarizing visible actions within videos and identifying actions like "person sitting on a bench" effectively in Fig. 25. However, they sometimes add incorrect details, such as "person smiling to the camera," revealing a shortfall in capturing conversation topics or the video’s ambiance, elements that are readily apparent to human observers. This shortfall underscores another key limitation: the omission of audio and speech modalities that would enrich the video understanding with context, aiding in more accurate interpretation and preventing such misrepresentations. Bridging this gap requires a holistic integration of available modalities, allowing multimodal agents to reach a level of comprehension akin to human perception and ensuring a fully multimodal approach to video interpretation.

![](images/cede1c5c4ad5118296f3b112ed6fca0ccc6b4f67221b66e86821575e890cc52d.jpg)  
What would you recommend for this person to do to feel better?

![](images/6097f00763abe2e34aabb7802977077df6579c28edb9506f9cdd221d326214d8.jpg)

In the image, there is a person sitting on a bench in a park with a friend. The person is wearing a blue shirt and a pair of blue jeans. The person is holding a bottle of water in one hand and a cup of coffee in the other. The person is looking at the camera and smiling to the camera. The person may be feeling depressed, anxious, or stressed due to a recent event or situation. The person may want to take a break from their busy schedule and relax for a while. They may want to take a walk, read a book, listen to music, or take a nap to help them feel better. The person may also want to talk to a trusted friend or family member to share their feelings and provide emotional support.

![](images/a5a4b2884b9fb2b5794e5b7aac7b51e12f154966106694e9afbd230d4fec2ad9.jpg)  
Provide a short description of the scene

![](images/40bf93c3c3252fc547d40307a6e75936d8b05631f4cf5e97d39ce915cb034b16.jpg)

![](images/1a064e1ec51a130fe5e17a85619d0f9a318bf1ecc0e03357b5dab9f0975cdacf.jpg)  
Red fox running in the snow   
What does the fox do at the end?

![](images/8cee8edcd150ffe3b442940d0289b4e701ebc558299f78c28567982f0a451588.jpg)  
Jump.

![](images/de36b595008ae5399e37bb64ecd7b48ce5f87b57cf015abaefbf7544ca4a5fb1.jpg)  
Figure 25: Example prompts and responses when using a video fine-tuned variant of InstructBLIP (method described in Section 6.5). Our model is able to produce long-form textual responses that describe scenes and is able to answer questions related to the temporality of events in the videos.

# Video Frames

![](images/0060e9f1de3a61033117fd8b2229f49a0577da11231048c2e12cd4ef5a927c3d.jpg)

# GPT4-V Frame Captions

A man with a serious expression stands holding a yellow measuring stick in a parking lot, with a car and trees in the background.

![](images/efb3e80a3a3eb45eca563e487c57fa132ace11f5b4fe8131ffce0bb0ffad2796.jpg)

A man ina "New York City"tshirt holds a tall, yellow measuring stick upright in a sunny parking lot setting.

![](images/5199deecb42d67a79ba3b60b8c21cc05b2e468df93d0360c44d190b47e714d6f.jpg)

A man is humorously biting a large yellow measuring stick while extending it with his left arm ina parking lot with a backdrop of greenery and a car.

![](images/1a70b335f812b4a5c7d3c9b5316f72901035cf3f2e9e6535de7d80520c8a171b.jpg)  
Figure 26: The audio-multimodal agent described in Section 6.5. Hallucinated content are highlighted in red. We use GPT-4V to generate 1) the videochat summary with video frames; 2) the video summary with the frame captions; 3) the video summary with frame captioning and audio information.

# VideoChat Summary

The video shows a man wearing a blue shirt who is holding a yellow pole and demonstrating how to use it.He is seen holding the pole and talking to the camera while explaining the process.The man is wearing a blue shirt and is holding a yellow pole. He is seen holding the pole and talking to the camera while explaining the process.

# GPT-4V Summary with Frame Captions

In the video,a man appears in a series of sequences where he interacts with a large measuring stick in a parking lot. Initially, he stands with the stickat his side,presenting it like a novelty item.As the video progresses, he engages more actively with the stick, extending it outward as if sizing up some unseen object or distance.In a playful twist, he bites down on it while holding it horizontally,showcasinga humorousapproach to what might be a mundane task. Throughout the video, the man's demeanor suggests he's involved ina light-hearted activity,possibly a humorous take on measurementand scale inan everyday suburban setting.The background with cars,greenery,and signage fora dental clinic remains consistent,anchoring the sequence of events in a single location and providingacasual,almost candid atmosphere to theactivities unfolding.

# Audio

# ASR Transcriptions

This is your tree height measuring lesson so l have an ordinary broomstick here.

you hold your hand out like this perpendicular to your body and holdingthe broomstick like that.

you rotate the broomstick down like thisand you adjust the length of the broomstick such that the top of thebroomstickthispointright here is parallel with your eye

# GPT-4V Video Summary with Frame Captions $\mathbf { + }$ Audio

The video is a tutorial on how to measure the height of a tree using a simple method with a broomstick. The instructor demonstrates how to hold the broomstick perpendicular to the body and rotate it down so that when it's parallel to the eye,the length of the broomstick is adjusted accordingly.The stick is then flipped back up and held perpendicular again. The distance from the hand to the eye is said to be equal to the distance from the hand to the end of the stick.The instructor then explains how to use this measurement to create an isosceles triangle when sighting the bottom of the tree with one eye and aligning the top of the tree with the top of the stick.This alignment allows the measurer to determine that they are exactly as far from the tree as the tree is tall.Finally,the instructor advises to measure the distance from the point on the ground to the base of thetree withatapemeasure tofindtheheight ofthetree,completing the measurement process.

![](images/864290ff0273083f4abf75f5c0e6110610cef71cbd9ca1bbfcb884a73257694a.jpg)  
Figure 27: An interactive multimodal agent that incorporates visual, audio, and text modalities for video understanding. Our pipeline mines hard negative hallucinations to produce difficult queries for the VideoAnalytica challenge. More the related details of interactive audio-video-language agent dataset are described in Section 9.2.

Audio-Video-Language Agents with GPT-4V. We then evaluate the capabilities of GPT-4V as a multimodal agent that integrates vision, audio, and speech for a nuanced and precise understanding of videos, following the methodology outlined in (Lin et al., 2023). Results depicted in Fig. 26 compare the performance of various video agents on the task of video summarization. The video-instruction tuned model (Li et al., 2023g) provides accurate content but falls short on comprehensiveness and detail, missing specific actions like the methodical use of a broomstick to measure a tree’s height.

To enhance the accuracy of video descriptions, we employ GPT-4V to caption frames, while audio and its transcriptions are sourced from the OpenAI Whisper model. We then prompt GPT-4V to create video summaries using only frame captions and then using both frame captions and audio transcriptions. Initially, we observe that frame captions alone can lead to fabricated events, such as a person biting down on a stick in the third segment. These inaccuracies persist in the video summary, with descriptions like "in a playful twist, he bites down on it while holding it horizontally." Without audio input, the agent cannot correct these captioning errors, resulting in descriptions that are semantically correct but visually misleading.

However, when we provide the audio transcriptions to the agent, it manages to accurately depict the content, even capturing detailed physical actions like "holding the broomstick perpendicular to the body and rotating it downwards." This level of detail is significantly more informative and gives viewers a clearer understanding of the video’s purpose and key details. These findings highlight the importance of integrating audio, video, and language interactions to develop high-quality multimodal agents. GPT-4V emerges as a promising foundation for such advanced multimodal understanding and interaction.

Embodied Multi-modal Agents with GPT-4V. As shown in Fig. 27, We mainly used StackOverflow to get the initial Question, then we used the “Bing search" API to retrieve a related video and audio corresponding to the question. Next, we mainly use GPT-4V to get the relevant text information and high-level video description. On the other hand, we transfer the key frame audio to a low-level segment description of the key frames via ASR. Finally, we use GPT-4V to generate convincing "hallucinations" that serve as hard negative queries for video-question and answer tasks. We support interactions and question answering in the current frame of the video, as well as summarization for the overall high-level video description. During inference, we also combine external knowledge information via web search to improve answering capapbilities.

The main prompt information for GPT-4V is described as below. The entire prompt is indented for clarity; it is over one page long.

GPT-4V are an assistant to provide descriptive, informative, and full comprehensive details in the video for the visually impaired who can hear the video but cannot see. The job is to create high-quality, dense descriptions of the video by synthesizing the given annotations and output them as JSON. Specifically, GPT-4V will be given original query used to search the video, the video title, description, audio transcription, and potentially noisy descriptions for specific time in the video. Different segments of same video is annotated as "[time start - time end (in seconds)] ’text’ ". Utilize the transcriptions and descriptions all together to reason about the exact detail and visual demonstration that might be happening in the video. GPT-4V will to combine or segment the timestamps as necessary to provide the best segmentation of the video.

# Expectations for GPT-4V Output:

1. Action-Oriented Descriptions: Prioritize plausible actions, motions, and physical demonstrations that the audio implies, enriching your narrative with dynamic visual cues.   
2. Complete Video Coverage: Provide a continuous and consistent audio-descriptive experience that covers every moment of the video’s duration, ensuring no content is left undescribed.   
3. Concise Segmentation: Construct your descriptions in focused, succinct segments of 1-2 sentences each to effectively communicate visual actions without overwhelming detail.   
4. Contextual Audio-Visual Synthesis: Seamlessly blend the spoken audio content with inferred visual elements to form a narrative that reflects potential onscreen activities.   
5. Imaginative and Plausible Speculation: Infuse your descriptions with creative yet believable visual details that correspond with the audio, enhancing scene comprehension.   
6. Accurate Timecode Correspondence: Align your descriptive segments with corresponding timecodes, ensuring that speculative visual details synchronize with the audio narrative’s timeline.   
7. Confident Narrative Delivery: Present the descriptions with assurance, as though the speculated visuals are occurring, to instill confidence in the listener.   
8. Omit Implausible Details: Exclude descriptions of objects or events that do not reasonably fit within the context established by the audio and visual information provided.   
The final output should be structured in a JSON format containing a list of dictionaries, each detailing a segment of the video.   
The final output should be structured in a JSON format containing a list of dictionaries, each detailing a segment of the video.

[ ‘start’: <start-time-in-seconds>, ‘end’: <end-time-in-seconds>, ‘text’: “<Your detailed single-sentence, audio-visual description here>" ]

For MC Creation: our task is to create multiple-choice questions for video-to-text retrieval tasks that is trivially solved by looking at the title and reading through audio transcriptions. To do so, we will be given original query to get the video, description, audio transcription, and potentially noisy descriptions for specific time in the video.

• Format of audio transcription: -[start-end time in seconds] “transcription"   
• Format of noisy description: - [time in seconds] “description"

We kindly ask GPT-4V to generate four queries, where the primary query is aligned with the video content, and the other three negatives are subtly different from our primary one. Selecting the primary one should not simply involve listening to audio transcriptions e.g. the text original query is contained in audio transcriptions. The negatives should be closely related but not fully aligned with the video content, requiring visual understanding of the video to differentiate. For example, modify the semantics in nuanced way so that one needs to watch the video than just listening to select the original query. Compile four queries in caption-like statement, with the first one being the rephrased original.

Think step by step how you can come up with negative statements using the information from the video. And justify the negative queries are incorrect but still compelling choices that demand nuanced understanding of the video. And how humans would not accidentally choose the negatives over the original query.

Finally, we present the work in the following format of analyses and 4 queries. No need to generate how you translated the original query.

• Video Analysis: xxx   
• Queries: [query1, query2, query3, query4]   
• Justification: xxx

# 6.6 Agent for NLP

# 6.6.1 LLM agent

Recognizing task directives and taking action has been a fundamental challenge in interactive AI and natural language processing for decades. With the recent advances in deep learning, there is a growing interest in studying these areas jointly to improve human-agent collaboration. We identify three specific directions, among others, to improve language-grounded agents:

• Tool use and querying from knowledge bases. This direction emphasizes the importance of integrating external knowledge bases, web search, or other helpful tools into the reasoning processes of AI agents. By leveraging structured and unstructured data from various sources, agents can enhance their understanding and provide more accurate and context-aware responses. Furthermore, it fosters the agent’s ability to proactively seek out information when faced with unfamiliar scenarios or queries, ensuring more comprehensive and informed responses. Examples include Toolformer (Schick et al., 2023) and Retrieve What You Need (Wang et al., 2023g).   
• Improved agent reasoning and planning. Enhancing the agent’s ability to reason and plan is pivotal for effective human-agent collaboration. This involves the development of models that can understand complex instructions, infer user intentions, and predict potential future scenarios. This can be accomplished by asking the agent to reflect on past actions and failures as in ReAct (Yao et al., 2023a), or by structuring the agent thought process as a form of search (Yao et al., 2023b). By simulating different outcomes and assessing the ramifications of various actions, agents can make more informed context-aware decisions.   
• Incorporating system and human feedback. AI agents can frequently operate in two primary contexts: environments that provide explicit signals about the effectiveness of their actions (system feedback), and settings where they collaborate with humans who can offer verbal critiques (human feedback). This direction underscores the need for adaptive learning mechanisms that allow agents to refine their strategies and rectify mistakes, such as in AutoGen (Wu et al., 2023). The ability to continuously learn and adapt from diverse feedback sources ensures that agents remain helpful and aligned for user needs.

# 6.6.2 General LLM agent

Recognizing and understanding agent content and natural language has been a fundamental challenge in interactive AI and natural language processing for decades. With the recent advances in deep learning, there is a growing interest

# General Alpaca Training Framework

![](images/2cf4dfad9b6f5fa003c2f4878e5dfda7a5c5d12fd791fa5220567e8f70a64909.jpg)  
Figure 28: The training recipe used to train the Alpaca model (Taori et al., 2023). At a high level, existing LLMs are used to generate a large pool of instruction-following examples from a smaller set of seed tasks. The generated instruction-following examples are then used to instruction-tune an LLM where the underlying model weights are available.

in studying these two areas jointly for deep understanding of both agent planning or human feedback for knowledgeinference and natural language generation. These are the key components of many human-machine-interaction agents, such as “AutoGen"(Wu et al., 2023) and “Retrieve What You Need"(Wang et al., 2023g).

# 6.6.3 Instruction-following LLM agents

Furthermore, the creation of LLM Agents that can be trained to effectively follow human instructions has become an important area of research. Initial models used human feedback to train a proxy reward model to simulate human preferences, through a process known as Reinforcement Learning with Human Feedback (RLHF) (Ouyang et al., 2022). This process produced models such as InstructGPT and ChatGPT. In order to more efficiently train instruction-following LLM agents without needing human labels, researchers developed a more efficient method for instruction-tuning that trains the LLM agent directly on instruction/response pairs, either generated by humans like Dolly $2 . 0 ^ { 6 }$ or automatically from LLMs like Alpaca (Taori et al., 2023). We show the overall Alpaca training pipeline in Figure 28.

# 6.6.4 Experiments and Results

Despite the growing adoption of conversational and self-feedback systems, these forms of AI still do not perform well with regard to generating factually correct responses from their own implicit knowledge and therefore often use external tools like web search and knowledge retrieval mechanisms at inference-time to augment their response as a consequence. Addressing this would help create more engaging experiences for users in many real-life applications. In social conversations (such as those on social media platforms like Instagram and Facebook), or with $\mathrm { Q + A }$ websites (such as Ask or Quora), people usually engage with others through a series of comments and by web-searching for information and knowledge relevant to the discussion. Thus, the task of generating conversational turns in this context is not to simply bootstrap upon traditional NLP models and tasks, but to use agents to generate dialogue through intelligent behaviors that reflect knowledge search and acquisition (Peng et al., 2023). In this way, intelligent agents for

![](images/ead85165ec20be59e6cef63b4dcadbbe8ca6964d979337bb253a9cbf879b22c8.jpg)  
Figure 29: The logic transformer agent model (Wang et al., 2023e). We integrate a logical reasoning module into the transformer-based abstractive summarization model in order to endow the logic agent the ability to reason over text and dialogue logic, so that it can generate better-quality abstractive summarizations and reduce factuality errors.

These logic embeddings vectors are fully parameterizedand arefully trainable during thefine-tuning training processon dialogue summarization datasets.The only restrictions that we put on their values are:

· Logic embeddings with same color(i.e.same logic role and type)must have the same vector values;   
·Logicembeddings with diferent colors (i.e.diferent logic roles and types)can have different vector values.

NLP tasks extends the task description and improves upon the interpretability of the response by adding an explicit knowledge search and retrieval step during dialogue. Incorporating these web search and retrieval agents as feedback during dialogue will help to engage further and deeper the social interactions between humans and agents (Wang et al., 2023e). As the Fig 29 showed, we introduced a new modeling paradigm for transformer language models that detects and extracts important logical structures and information from input texts and then integrates them into the input embeddings through carefully designed multi-layer hierarchical logical projections to infuse logical structures into pre-trained language models as one kind of NLP agent. (Wang et al., 2023e) propose a novel approach to construct logic-aware input embeddings for transformer language models through a combination of logic detection, logic mapping and hierarchical logical projections, and then develop a corresponding new modeling paradigm that can upgrade all existing transformer language models into logical transformers to consistently boost their performance. The proposed logical transformer agent consistently achieve superior performance over their baseline transformer models through a deeper understanding of the logical structures of texts. To human users, it is often these aspects that are more important for delivering a meaningful and interesting conversation via a agent-based coordination between dialogue and information retrieval. Delving deep into natural language processing, this topic will discuss the advancements and leader-board in making LLMs more agentic and better suited for various language-centered tasks.

An open-domain question answering (QA) system usually follows a retrieve-then-read paradigm, in which a retriever is used to retrieve relevant passages from a large corpus, and then a reader generates answers based on the retrieved passages and the original question. In (Wang et al., 2023g), we propose a simple and novel mutual learning framework to improve the performance of retrieve-then-read-style models via an intermediate module named the knowledge selector agent, which we train with reinforcement learning. The fine-grained knowledge selector into the retrieve-thenreader paradigm, whose goal is to construct a small subset of passages which retain question-relevant information. As showed in Figure 30, The knowledge selector agent is trained as a component of our novel mutual learning framework, which iteratively trains the knowledge selector and the reader. We adopt a simple and novel approach employing policy gradients to optimize the knowledge selector agnet, using feedback from the reader to train it to select a small and

![](images/e2d598f8e374eabaa29ff4ec5128c42910da6f1a0906fbd5bca85e2154b4234a.jpg)

![](images/935755174734bfa5381f4e2985349842a9c8e4377d9693111d8b4ce8aa9539d4.jpg)  
Figure 30: Architecture of one proposed NLP agent (Wang et al., 2023g) mutual learning framework. In each epoch, Phase 1 and Phase 2 are executed alternately. During Phase 1, the parameters of the reader model remain fixed, and only the weights of the knowledge selector are updated. Conversely, during Phase 2, the reader model’s parameters are adjusted, while the knowledge selector’s weights remain frozen.

informative set of passages. This approach avoids brute-force search or manually-designed heuristics, without requiring any annotated query-document pairs for supervision. We show that iteratively training the reader and the knowledge selector agent leads to better predictive performance on some public open-domain question answering benchmarks.

# 7 Agent AI Across Modalities, Domains, and Realities

# 7.1 Agents for Cross-modal Understanding

Multi-modal understanding is a significant challenge for creating generalist AI agents due to the lack of large-scale datasets that contain vision, language, and agent behavior. More generally, training data for AI agents is often modality specific. This results in most modern multi-modal systems using a combination of frozen submodules. Some notable examples are Flamingo (Alayrac et al., 2022), BLIP-2 (Li et al., 2023c), and LLaVA (Liu et al., 2023c), all of which utilize a frozen LLM and frozen visual encoder. These submodules are trained individually on separate datasets, and then adaptation layers are trained to encode the visual encoder into the LLM embedding space. In order to make further progress for cross-modal understanding for AI agents, it is likely that the strategy of using frozen LLMs and visual encoders will need to change. Indeed, RT-2, a recent visual-language model that is capable of taking actions within the domain of robotics showed significantly improved performance when jointly tuning the visual encoder and LLM for robotics and visual-language tasks (Brohan et al., 2023).

# 7.2 Agents for Cross-domain Understanding

A key challenge for creating generalist agents is the distinctive visual appearance and disparate action spaces across different domains. Humans possess the capability to interpret images and videos from various sources, including the real world, video games, and specialized domains such as robotics and healthcare, once they become familiar with the specific details of these areas. However, existing LLMs and VLMs often demonstrate significant differences between the data they were trained on and the varied domains in which they are applied. And notably, training agent models to predict specific actions presents a considerable challenge when trying to develop a single policy that can effectively learn multiple control systems across domains. Generally, the approach most modern works take when applying systems within specific domains is to start from a pretrained foundation model and then finetune a separate model for each specific domain. This fails to capture any commonalities between domains and results in a smaller total set of data used for training instead of leveraging each domain’s data.

# 7.3 Interactive agent for cross-modality and cross-reality

Developing AI agents that can successfully understand and perform tasks across different realities is an on-going challenge that has seen some recent success for image and scene generation (Huang et al., 2023a). In particular, it is challenging for agents to simultaneously understand real-world and virtual reality environments due to their visual dissimilarities and separate environment physics. Within the context of cross-reality, Sim to Real transfer is a particularly important problem when using simulation-trained policies for real-world data, which we discuss in the next section.

# 7.4 Sim to Real Transfer

Techniques which enable models trained in simulation to be deployed in the real world. Embodied agents, especially one based on RL policies, are typically trained in simulated environments. These simulations do not fully replicate the characteristics of the real world (e.g., disturbances, light, gravity, and other physical properties). Due to this discrepancy between simulation and reality, models trained in simulation often struggle to perform well when applied in the real world. This issue is known as the “sim-to-real” problem. To solve this problem, several approaches can be taken:

• Domain randomization: domain randomization is a technique that trains a model while randomly varying parameters within a simulation environment (e.g., object appearance, sensor noise, and optical properties) in anticipation of the uncertainties and variations of the real world (Tobin et al., 2017). For instance, in the context of training a RL-based grasping skills, introducing randomness in the shapes of objects can lead to a policy capable of adapting to objects with somewhat different shapes (Saito et al., 2022).   
• Domain adaptation: Domain adaptation, or domain transfer is a technique that bridges the gap between simulated and real-world domains by training models with a large number of simulated images and a smaller set of real-world images. In practical settings, unpaired image-to-image translation methods such as CycleGAN (Zhu et al., 2017b) are employed due to the difficulty in preparing paired images across domains. Several enhanced versions exist for reinforcement learning, including RL-CycleGAN (Rao et al., 2020), and for imitation learning, such as RetinaGAN (Ho et al., 2021).   
• Improvement of simulation: Realistic simulation is a key for sim-to-real transfer. Part of this effort is achieved by a system identification techniques (Zhu et al., 2017c; Allevato et al., 2020), which aims to identify simulation parameters to mimic the real-world environments. Additionally, use of photorealistic simulators would be effective in image-based reinforcement learning (Martinez-Gonzalez et al., 2020; Müller et al., 2018; Shah et al., 2018; Sasabuchi et al., 2023).

The sim-to-real transfer remains a central challenge in the study of Embodied Agents, as approaches keep evolving. Both theoretical and empirical research are essential to advance these technologies further.

# 8 Continuous and Self-improvement for Agent AI

Currently, foundation model based AI agents have the capacity to learn from multiple different data sources, which allow for more flexible sources for data for training. Two key consequences of this are (1) user and human-based interaction data can be used to further refine and improve the agent and (2) existing foundation models and model artifacts can be used to generate training data. We discuss each of these in more detail in the following sections, but we note that since current AI Agents are largely tied to existing pretrained foundation models, they generally do not learn from continuous interaction with their environments. We think this is an exciting future direction, and initial work by Bousmalis et al. has shown that self-improving agents for robotic control are able to continuous learn and improve through environmental interactions without supervision (Bousmalis et al., 2023).

# 8.1 Human-based Interaction Data

The core idea behind using human-based interaction data is to leverage a large number of of agent-human interactions to train and improve future iterations of the agent. There are several strategies used to improve agents from human-agent interactions.

• Additional training data Perhaps the simplest usage of human-agent interactions is to use the interaction examples themselves as training data for a future iteration of the agent. This generally requires filtering

strategies to differentiate successful agent examples from unsuccessful interaction examples. Filtering can be rules-based (e.g., reaching some desired end goal state), model-based (e.g., classifying successful vs unsuccessful interactions), or manually selected after a posthoc inspection and/or modification of the interaction examples.

• Human preference learning During interaction with the user, the agent system can prompt the user with several different model outputs and allow for the user to select the best output. This is commonly used by LLMs like ChatGPT and GPT-4, whereby users can select one output (out of several) that aligns best with their preferences.   
• Safety training (red-teaming) Red-teaming within the context of Agent AI refers to having a dedicated team of adversaries (either human or computer) that seek to exploit and expose weaknesses and vulnerabilities within the Agent AI system. Although adversarial in nature, red-teaming is commonly used as a means for understanding how to improve AI safety measures and reduce the occurrence of harmful outputs. The core principle is to discover consistent methods for inducing unwanted agent outputs so that the model can be trained on data that explicitly corrects this behavior.

# 8.2 Foundation Model Generated Data

With the advent of powerful foundation model artifacts produced by academia and industry, there have been a variety of methods developed to extract and generate meaningful training data from these artifacts using a variety of prompting and data-pairing techniques.

• LLM Instruction-tuning Methods for generating instruction-following training data from LLMs have allowed for the finetuning of smaller, open-source models based on the outputs of larger proprietary LLMs (Wang et al., 2022b). For example, Alpaca (Taori et al., 2023) and Vicuna (Zheng et al., 2023) are LLMs based on the open-source LLaMA family (Touvron et al., 2023) that have been tuned on various outputs from ChatGPT and human participants. This method of instruction tuning can be viewed as a form of knowledge distillation, where the larger LLM serves as a teacher model to a smaller student model. Importantly, although LLM instruction-tuning has been shown to transfer the writing style and some instruction-following capabilities of the teacher model to the student model, significant gaps still exist between the factuality and capabilities of the teacher and student models (Gudibande et al., 2023).   
• Vision-language pairs A number of recent works have sought to increase the number of diversity of pretraining data available to visual-language models by automatically generating captions and other text for visual content. For example, LLaVA (Liu et al., 2023c) uses 150,000 examples of instruction-following behavior from textual and visual inputs that are mainly LLM-generated. Other work has shown that using VLMs to re-caption images can improve the training data and subsequent quality of image generation models (Segalis et al., 2023). Within the realm of video understanding, using VLMs and LLMs to recaption videos has been shown to improve the performance and quality of subsequent VLMs trained on the recaptioned videos (Wang et al., 2023f; Zhao et al., 2022).

# 9 Agent Dataset and Leaderboard

To accelerate research in this domain, we propose two benchmarks respectively for multi-agent gaming and agentic visual language tasks. We will release two new datasets - “CuisineWorld” and “VideoAnalytica” - and a set of baseline models, encouraging participants to explore new models, systems, and submit their results on the test set of our leaderboard.

# 9.1 “CuisineWorld” Dataset for Multi-agent Gaming

CuisineWorld is a text-based game reminiscent of Overcooked! It offers a platform for AI-powered agents to cooperate and play in tandem. This dataset will test the collaboration efficiency of multi-agent systems, offering insights into how well LLMs and other systems can work together in dynamic scenarios. In particular, the dataset will focus on how well the agents understand goals, and how well the agents can coordinate among themselves. Two types of modes are supported in this dataset: a centralized dispatcher mode and a decentralized mode. Participants can choose a play mode and make a submission to our leaderboard.

# 9.1.1 Benchmark

For our competition, we will release a benchmark, the CuisineWorld benchmark, which includes a text interface that includes extendable task definition files, and an interface for multi-agent interaction, and human-machine interactions. We introduce the gaming interaction task in which the goal is to generate relevant, appropriate, multi-agent collaboration strategies that can maximize collaboration efficiency. We evaluate the collaboration efficiency with the proposed evaluation metric: CoS.

The “CuisineWorld" dataset was collected by Microsoft, UCLA, and Stanford University. The goal of the competition is to explore how different, existing and novel, grounded-LLM and interactive techniques perform with this benchmark and establish strong baselines for the task of multi-agent gaming infrastructure.

The dataset of CuisineWorld includes:

- A selection of well-defined multi-agent collaboration tasks.   
- An API system to facilitate agent interactions.   
- An automatic evaluation system.

(The link for downloading the dataset will soon be made available and this article will be updated to include it here.)

# 9.1.2 Task

• We provide a dataset and related the benchmark, called Microsoft MindAgent and and correspondingly release a dataset “CuisineWorld” to the to the research community.   
• We will provide benchmarks to evaluate and rank the submitted “MindAgent" algorithms. We will also provide baseline results generated using popular infrastructures.

# 9.1.3 Metrics and Judging

The quality of multi-agent collaboration efficiency is determined by the new “cos" auto-metric (from MindAgent (Gong et al., 2023a)). The final rating of out metric is calculated as an average over the evaluated collaboration efficiency metrics of the multi-agent system on all tasks. Human evaluators will be asked to rate individual responses as well as provide subjective judgement of the engagement, breadth and an overall quality of the users’ interactions with the agents.

# 9.1.4 Evaluation

• Automated Evaluation. We plan to release a leaderboard, starting on the release date (TBA), registered participants will be asked to submit their results on the task associated with the dataset “CuisineWorld" (our publicly released dataset for the leaderboard). Submission of results will be closed on the end date (TBA). Each team will be required to submit their generated results on the testing set for automated evaluation of the “cos" metric.   
• Human Evaluation on our leaderboard. The leaderboard participants will need to provide a submission file generated by evaluation scripts locally. We will use the evalAI system to check the submission file and optionally rerun the code for top challenge contenders. Therefore, teams must also submit their code with a Readme file on how to run their code. Human evaluation will be performed by the organization team.   
• Winner Announcement. We will make an announcement of the winners and post the final ratings of the submissions on our leaderboard.

# 9.2 Audio-Video-Language Pre-training Dataset.

We introduce VideoAnalytica: a new benchmark for analytical video demonstration comprehension. VideoAnalytica focuses on leveraging video demonstrations as aids to better understand complex, high-level reasoning embedded within long-formed instructional videos. The objective is to evaluate the cognitive reasoning abilities of video language models, pushing them beyond mere recognition tasks and basic comprehension, towards a more sophisticated and nuanced understanding of videos. Crucially, VideoAnalytica emphasizes the integration of multiple modalities, such as

audio, video, and language, as well as the ability of models to apply domain-specific knowledge, to contextualize and interpret the information presented in the videos. Specifically, VideoAnalytica involves two primary tasks:

1. Video Text Retrieval: This task involves accurately retrieving relevant text from the instructional videos. The challenge lies in distinguishing between relevant and irrelevant information, thus requiring a deep understanding of the video content, and analysis of the demonstration to retrieve the correct query. To further increase the complexity of these tasks, we introduce hard negatives into our datasets generated by large language models. We run human validation on the generated negatives and remove instances that make the task invalid and unfair (e.g. negatives being valid).   
2. Video Assisted Informative Question Answering: This task requires the model to answer questions based on the information extracted from the videos. The focus is on complex questions that require analytical reasoning and a thorough comprehension of the video demonstration.

To facilitate the development of an audio-video-language agent for analytical video understanding, we introduce a benchmark leaderboard for the two tasks from VideoAnalytica.

• The leaderboard participants will need to submit their solutions for evaluation. The evaluation will be based on the model’s performance on the two tasks, and the results will be displayed on the leaderboard. Participants are required to submit their code, along with a detailed explanation of their approach and methodology.   
• Ethical considerations: The leaderboard focuses on understanding and interpreting video content, which could potentially be used in surveillance or other privacy-invasive applications. Therefore, it’s crucial to consider the ethical implications and potential misuse of the technology. We encourage participants to consider these aspects in their submissions and promote the ethical use of AI.

# 10 Broader Impact Statement

This article and our associated forum 7 aim to be a catalyst for innovative research, fostering collaborations that will drive the next wave of AI applications. By focusing on multimodal agents, we emphasize the future direction of human-AI interactions, leader-board, and solutions. We detail three ways in which we make significant contributions to the broader community.

Firstly, we hope our forum grounds AI researchers to develop solutions motivated by real-world problems in gaming, robotics, healthcare, and long-video understanding. Specifically, the development of multimodal agents in gaming could lead to more immersive and personalized gaming experiences, thereby transforming the gaming industry. In robotics, the development of adaptive robotic systems could revolutionize industries ranging from manufacturing to agriculture, potentially addressing labor shortages and improving efficiency. In healthcare, the use of LLMs and VLMs as diagnostic agents or patient care assistants could lead to more accurate diagnoses, improved patient care, and increased accessibility to medical services, particularly in underserved areas. Furthermore, the ability of these models to interpret long-form videos could have far-reaching applications, from enhancing online learning to improving technical support services. In general, the topics covered in our forum will have significant downstream effects on a wide range of industries and humans across the world.

Secondly, we hope our forum stands as a valuable resource for AI practitioners and researchers alike, serving as a platform to explore and deeply comprehend the diverse and complex leader-board that come with implementing AI agents across a wide variety of environments and situations. This exploration includes, for instance, understanding the specific limitations and potential hazards linked to Agentic AI systems when they are developed for specialized sectors such as healthcare diagnostics. In this domain, issues like dangerous hallucinations in AI behavior can pose significant risks, highlighting the critical need for meticulous design and testing. However, these specific leader-board may not be equally relevant or noticeable when considering AI agents crafted for the gaming industry. In such recreational fields, developers might instead prioritize tackling different hurdles, such as the need for AI to perform more open-ended generation and exhibit creativity, adapting dynamically to unpredictable gameplay scenarios and player interactions. By attending the forum, participants will gain insights into how these varied environments dictate the focus and direction of AI development, and how best to tailor AI solutions to meet these distinct needs and overcome the pertinent leader-board.

Thirdly, the various elements of our event, including the expert presentations, informative posters, and notably the winners of our two leader-board, are set to offer a substantive yet succinct overview of the latest and significant trends, research directions, and innovative concepts in the realm of multimodal agents. These presentations will encapsulate pivotal findings and developments, shining a light on new systems, ideas, and technologies in the field of mulitmodal agent AI. This assortment of knowledge is not only beneficial for the attendees of our forum, who are looking to deepen their understanding and expertise in this domain, but it also serves as a dynamic and rich resource board. Those visiting our forum’s website can tap into this reservoir of information to discover and understand the cutting-edge advancements and creative ideas steering the future of multimodal agent AI. We strive to serve as a useful knowledge base for both newcomers and veterans in the field. By engaging with these resources, we hope participants and online visitors alike can remain informed of the transformative changes and novel approaches that are shaping the exciting landscape surrounding multimodal agent AI.

# 11 Ethical Considerations

Multimodal Agent AI systems have many applications. In addition to interactive AI, grounded multimodal models could help drive content generation for bots and AI agents, and assist in productivity applications, helping to re-play, paraphrase, action prediction or synthesize 3D or 2D scenario. Fundamental advances in agent AI help contribute towards these goals and many would benefit from a greater understanding of how to model embodied and empathetic in a simulate reality or a real world. Arguably many of these applications could have positive benefits.

However, this technology could also be used by bad actors. Agent AI systems that generate content can be used to manipulate or deceive people. Therefore, it is very important that this technology is developed in accordance with responsible AI guidelines. For example, explicitly communicating to users that content is generated by an AI system and providing the user with controls in order to customize such a system. It is possible the Agent AI could be used to develop new methods to detect manipulative content - partly because it is rich with hallucination performance of large foundation model - and thus help address another real world problem.

For examples, 1) in health topic, ethical deployment of LLM and VLM agents, especially in sensitive domains like healthcare, is paramount. AI agents trained on biased data could potentially worsen health disparities by providing inaccurate diagnoses for underrepresented groups. Moreover, the handling of sensitive patient data by AI agents raises significant privacy and confidentiality concerns. 2) In the gaming industry, AI agents could transform the role of developers, shifting their focus from scripting non-player characters to refining agent learning processes. Similarly, adaptive robotic systems could redefine manufacturing roles, necessitating new skill sets rather than replacing human workers. Navigating these transitions responsibly is vital to minimize potential socio-economic disruptions.

Furthermore, the agent AI focuses on learning collaboration policy in simulation and there is some risk if directly applying the policy to the real world due to the distribution shift. Robust testing and continual safety monitoring mechanisms should be put in place to minimize risks of unpredictable behaviors in real-world scenarios. Our “VideoAnalytica" dataset is collected from the Internet and considering which is not a fully representative source, so we already go through-ed the ethical review and legal process from both Microsoft and University Washington. Be that as it may, we also need to understand biases that might exist in this corpus. Data distributions can be characterized in many ways. In this workshop, we have captured how the agent level distribution in our dataset is different from other existing datasets. However, there is much more than could be included in a single dataset or workshop. We would argue that there is a need for more approaches or discussion linked to real tasks or topics and that by making these data or system available.

We will dedicate a segment of our project to discussing these ethical issues, exploring potential mitigation strategies, and deploying a responsible multi-modal AI agent. We hope to help more researchers answer these questions together via this paper.

# 12 Diversity Statement

By examining the adaptability of AI agent models in various domains, we inherently embrace a diversity of leader-board, perspectives, and solutions. In this vein, our project aims to build a diverse community by exploring the wide array of subjects in multimodal and agentic AI.

With these principles in mind, this project focuses on advanced multimodal systems that interact effectively within both physical and virtual environments and facilitate effective interaction with humans. As such, we intend to engage a broad range of experts and practitioners across a wide-range of technical specialities, cultures, countries, and scholarly fields to discuss important topics, including but not limited to:

• Application of foundation models: the development of agents with integrated modalities (audio, image, text, sensor inputs), aiming to enhance their recognition and response capabilities for a wide variety of applications.   
• General-purpose end-to-end systems: the development of end-to-end models that are trained with large-scale data, seeking to create versatile and adaptable AI solutions.   
• Methodologies for grounding modalities: integrating information across various modalities, enhancing the coherence and efficacy of data processing.   
• Intuitive human interface: the development of effective and meaningful interaction between humans and agents.   
• Taming LLM/VLMs: exploring new approaches to address common issues in large-scale models, such as hallucinations and biases in their outputs.

We aspire to broaden our collective understanding of the potential and limitations of agentic AI by leveraging our unique and diverse perspectives. We strongly believe that this approach will not only enrich individual perspectives, but will also enhance the community’s collective knowledge and promote a holistic view that is more inclusive of the wide-ranging leader-board faced by multimodal AI agents.

# References

M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. Fu, K. Gopalakrishnan, K. Hausman, A. Herzog, D. Ho, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, E. Jang, R. J. Ruano, K. Jeffrey, S. Jesmonth, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, K.-H. Lee, S. Levine, Y. Lu, L. Luu, C. Parada, P. Pastor, J. Quiambao, K. Rao, J. Rettinghouse, D. Reyes, P. Sermanet, N. Sievers, C. Tan, A. Toshev, V. Vanhoucke, F. Xia, T. Xiao, P. Xu, S. Xu, M. Yan, and A. Zeng, “Do as i can and not as i say: Grounding language in robotic affordances,” in arXiv preprint arXiv:2204.01691, 2022.   
M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog et al., “Do as i can, not as i say: Grounding language in robotic affordances,” arXiv preprint arXiv:2204.01691, 2022.   
J.-B. Alayrac, J. Donahue, P. Luc, A. Miech, I. Barr, Y. Hasson, K. Lenc, A. Mensch, K. Millican, M. Reynolds et al., “Flamingo: a visual language model for few-shot learning,” Advances in Neural Information Processing Systems, vol. 35, pp. 23 716–23 736, 2022.   
A. Allevato, E. S. Short, M. Pryor, and A. Thomaz, “Tunenet: One-shot residual tuning for system identification and sim-to-real robot task transfer,” in Conference on Robot Learning. PMLR, 2020, pp. 445–455.   
A. Amjad, P. Kordel, and G. Fernandes, “A review on innovation in healthcare sector (telehealth) through artificial intelligence,” Sustainability, vol. 15, no. 8, p. 6655, 2023.   
S. An, Z. Lin, Q. Fu, B. Chen, N. Zheng, J.-G. Lou, and D. Zhang, “How do in-context examples affect compositional generalization?” arXiv preprint arXiv:2305.04835, 2023.   
P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik, R. Mottaghi, M. Savva et al., “On evaluation of embodied navigation agents,” arXiv preprint arXiv:1807.06757, 2018.   
P. Anderson, Q. Wu, D. Teney, J. Bruce, M. Johnson, N. Sünderhauf, I. Reid, S. Gould, and A. Van Den Hengel, “Vision-and-language navigation: Interpreting visually-grounded navigation instructions in real environments,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3674–3683.   
S. Antol, A. Agrawal, J. Lu, M. Mitchell, D. Batra, C. L. Zitnick, and D. Parikh, “Vqa: Visual question answering,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 2425–2433.   
M. Bain, A. Nagrani, G. Varol, and A. Zisserman, “Frozen in time: A joint video and image encoder for end-to-end retrieval,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021, pp. 1728–1738.   
B. Baker, I. Akkaya, P. Zhokov, J. Huizinga, J. Tang, A. Ecoffet, B. Houghton, R. Sampedro, and J. Clune, “Video pretraining (vpt): Learning to act by watching unlabeled online videos,” Advances in Neural Information Processing Systems, vol. 35, pp. 24 639–24 654, 2022.   
D. Batra, A. Gokaslan, A. Kembhavi, O. Maksymets, R. Mottaghi, M. Savva, A. Toshev, and E. Wijmans, “Objectnav revisited: On evaluation of embodied agents navigating to objects,” arXiv preprint arXiv:2006.13171, 2020.   
K. Black, M. Nakamoto, P. Atreya, H. Walke, C. Finn, A. Kumar, and S. Levine, “Zero-shot robotic manipulation with pretrained image-editing diffusion models,” arXiv preprint arXiv:2310.10639, 2023.   
A. Blair-Stanek, N. Holzenberger, and B. Van Durme, “Can gpt-3 perform statutory reasoning?” arXiv preprint arXiv:2302.06100, 2023.   
K. Bousmalis, G. Vezzani, D. Rao, C. Devin, A. X. Lee, M. Bauza, T. Davchev, Y. Zhou, A. Gupta, A. Raju et al., “Robocat: A self-improving foundation agent for robotic manipulation,” arXiv preprint arXiv:2306.11706, 2023.   
A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu et al., “Rt-1: Robotics transformer for real-world control at scale,” arXiv preprint arXiv:2212.06817, 2022.   
A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” arXiv preprint arXiv:2307.15818, 2023.

T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., “Language models are few-shot learners,” Advances in neural information processing systems, vol. 33, pp. 1877–1901, 2020.   
S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamar, P. Lee, Y. T. Lee, Y. Li, S. Lundberg et al., “Sparks of artificial general intelligence: Early experiments with gpt-4,” arXiv preprint arXiv:2303.12712, 2023.   
W. Cai, S. Huang, G. Cheng, Y. Long, P. Gao, C. Sun, and H. Dong, “Bridging zero-shot object navigation and foundation models through pixel-guided navigation skill,” arXiv preprint arXiv:2309.10309, 2023.   
M. Carroll, R. Shah, M. K. Ho, T. Griffiths, S. Seshia, P. Abbeel, and A. Dragan, “On the utility of learning about humans for human-ai coordination,” Advances in neural information processing systems, vol. 32, 2019.   
Y. Chang, M. Narang, H. Suzuki, G. Cao, J. Gao, and Y. Bisk, “WebQA: Multihop and Multimodal QA,” arXiv preprint arXiv:2109.00590, 2021.   
D. S. Chaplot, D. P. Gandhi, A. Gupta, and R. R. Salakhutdinov, “Object goal navigation using goal-oriented semantic exploration,” Advances in Neural Information Processing Systems, vol. 33, pp. 4247–4258, 2020.   
D. S. Chaplot, R. Salakhutdinov, A. Gupta, and S. Gupta, “Neural topological slam for visual navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 12 875–12 884.   
G. Chen, Y.-D. Zheng, J. Wang, J. Xu, Y. Huang, J. Pan, Y. Wang, Y. Wang, Y. Qiao, T. Lu, and L. Wang, “Videollm: Modeling video sequence with large language models,” 2023.   
K. Chen, Q. Huang, H. Palangi, P. Smolensky, K. D. Forbus, and J. Gao, “Mapping natural-language problems to formal-language solutions using structured neural representations,” in ICML 2020, July 2020.   
K. Chen, Q. Huang, D. McDuff, X. Gao, H. Palangi, J. Wang, K. Forbus, and J. Gao, “Nice: Neural image commenting with empathy,” in EMNLP 2021, October 2021. [Online]. Available: https: //www.microsoft.com/en-us/research/publication/nice-neural-image-commenting-with-empathy/   
J. H. Choi, K. E. Hickman, A. Monahan, and D. Schwarcz, “Chatgpt goes to law school,” Available at SSRN, 2023.   
H. W. Chung, L. Hou, S. Longpre, B. Zoph, Y. Tay, W. Fedus, Y. Li, X. Wang, M. Dehghani, S. Brahma et al., “Scaling instruction-finetuned language models,” arXiv preprint arXiv:2210.11416, 2022.   
N. C. F. Codella, D. Gutman, M. E. Celebi, B. Helba, M. A. Marchetti, S. W. Dusza, A. Kalloo, K. Liopyris, N. Mishra, H. Kittler, and A. Halpern, “Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging (isbi), hosted by the international skin imaging collaboration (isic),” in 2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018), 2018, pp. 168–172.   
A. Creswell, M. Shanahan, and I. Higgins, “Selection-inference: Exploiting large language models for interpretable logical reasoning,” arXiv preprint arXiv:2205.09712, 2022.   
B. Cui, A. Lupu, S. Sokota, H. Hu, D. J. Wu, and J. N. Foerster, “Adversarial diversity in hanabi,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/forum?id=uLE3WF3-H_5   
G. Dagan, F. Keller, and A. Lascarides, “Dynamic planning with a llm,” arXiv preprint arXiv:2308.06391, 2023.   
W. Dai, J. Li, D. Li, A. M. H. Tiong, J. Zhao, W. Wang, B. Li, P. Fung, and S. Hoi, “Instructblip: Towards generalpurpose vision-language models with instruction tuning,” 2023.   
A. d’Avila Garcez and L. C. Lamb, “Neurosymbolic ai: The 3rd wave,” 2020.   
M. Deitke, W. Han, A. Herrasti, A. Kembhavi, E. Kolve, R. Mottaghi, J. Salvador, D. Schwenk, E. VanderBilt, M. Wallingford et al., “Robothor: An open simulation-to-real embodied ai platform,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020, pp. 3164–3174.

Q. Dong, L. Li, D. Dai, C. Zheng, Z. Wu, B. Chang, X. Sun, J. Xu, and Z. Sui, “A survey for in-context learning,” arXiv preprint arXiv:2301.00234, 2022.   
V. S. Dorbala, G. Sigurdsson, R. Piramuthu, J. Thomason, and G. S. Sukhatme, “Clip-nav: Using clip for zero-shot vision-and-language navigation,” arXiv preprint arXiv:2211.16649, 2022.   
V. S. Dorbala, J. F. Mullen Jr, and D. Manocha, “Can an embodied agent find your" cat-shaped mug"? llm-based zero-shot object navigation,” arXiv preprint arXiv:2303.03480, 2023.   
A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” ICLR, 2021.   
D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu et al., “Palm-e: An embodied multimodal language model,” arXiv preprint arXiv:2303.03378, 2023.   
Y. Du, M. Yang, P. Florence, F. Xia, A. Wahid, B. Ichter, P. Sermanet, T. Yu, P. Abbeel, J. B. Tenenbaum et al., “Video language planning,” arXiv preprint arXiv:2310.10625, 2023.   
N. Dziri, A. Madotto, O. Zaiane, and A. J. Bose, “Neural path hunter: Reducing hallucination in dialogue systems via path grounding,” arXiv preprint arXiv:2104.08455, 2021.   
K. Ehsani, W. Han, A. Herrasti, E. VanderBilt, L. Weihs, E. Kolve, A. Kembhavi, and R. Mottaghi, “Manipulathor: A framework for visual object manipulation,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 4497–4506.   
D. Fried, R. Hu, V. Cirik, A. Rohrbach, J. Andreas, L.-P. Morency, T. Berg-Kirkpatrick, K. Saenko, D. Klein, and T. Darrell, “Speaker-follower models for vision-and-language navigation,” in Advances in Neural Information Processing Systems (NIPS), 2018.   
T.-J. Fu, L. Li, Z. Gan, K. Lin, W. Y. Wang, L. Wang, and Z. Liu, “Violet : End-to-end video-language transformers with masked visual-token modeling,” 2022.   
—, “An empirical study of end-to-end video-language transformers with masked visual modeling,” 2023.   
S. Y. Gadre, M. Wortsman, G. Ilharco, L. Schmidt, and S. Song, “Cows on pasture: Baselines and benchmarks for language-driven zero-shot object navigation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 23 171–23 181.   
J. Gao, B. Peng, C. Li, J. Li, S. Shayandeh, L. Liden, and H.-Y. Shum, “Robust conversational ai with grounded text generation,” arXiv preprint arXiv:2009.03457, 2020.   
J. Gao, C. Xiong, P. Bennett, and N. Craswell, “Neural approaches to conversational information retrieval,” arXiv preprint arXiv:2201.05176, 2022.   
C. R. Garrett, R. Chitnis, R. Holladay, B. Kim, T. Silver, L. P. Kaelbling, and T. Lozano-Pérez, “Integrated task and motion planning,” Annual review of control, robotics, and autonomous systems, vol. 4, pp. 265–293, 2021.   
T. Gervet, S. Chintala, D. Batra, J. Malik, and D. S. Chaplot, “Navigating to objects in the real world,” Science Robotics, vol. 8, no. 79, p. eadf6991, 2023.   
R. Gong, J. Huang, Y. Zhao, H. Geng, X. Gao, Q. Wu, W. Ai, Z. Zhou, D. Terzopoulos, S.-C. Zhu et al., “Arnold: A benchmark for language-grounded task learning with continuous states in realistic 3d scenes,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.   
R. Gong, Q. Huang, X. Ma, H. Vo, Z. Durante, Y. Noda, Z. Zheng, S.-C. Zhu, D. Terzopoulos, L. Fei-Fei et al., “Mindagent: Emergent gaming interaction,” arXiv preprint arXiv:2309.09971, 2023.   
A. Goyal, J. Xu, Y. Guo, V. Blukis, Y.-W. Chao, and D. Fox, “Rvt: Robotic view transformer for 3d object manipulation,” arXiv preprint arXiv:2306.14896, 2023.

M. Gramopadhye and D. Szafir, “Generating executable action plans with environmentally-aware language models,” arXiv preprint arXiv:2210.04964, 2022.   
A. Gudibande, E. Wallace, C. Snell, X. Geng, H. Liu, P. Abbeel, S. Levine, and D. Song, “The false promise of imitating proprietary llms,” arXiv preprint arXiv:2305.15717, 2023.   
L. Gui, Q. Huang, A. Hauptmann, Y. Bisk, and J. Gao, “Vlc: Training vision-language transformers from captions,” May 2022.   
L. Gui, B. Wang, Q. Huang, A. Hauptmann, Y. Bisk, and J. Gao, “Kat: A knowledge augmented transformer for vision-and-language,” in NAACL 2022. Long paper, Oral., January 2022.   
R. L. Guimarães, A. S. de Oliveira, J. A. Fabro, T. Becker, and V. A. Brenner, “Ros navigation: Concepts and tutorial,” Robot Operating System (ROS) The Complete Reference (Volume 1), pp. 121–160, 2016.   
K. Guu, K. Lee, Z. Tung, P. Pasupat, and M. Chang, “Retrieval augmented language model pre-training,” in International conference on machine learning. PMLR, 2020, pp. 3929–3938.   
H. Ha, P. Florence, and S. Song, “Scaling up and distilling down: Language-guided robot skill acquisition,” arXiv preprint arXiv:2307.14535, 2023.   
T. Haarnoja, B. Moran, G. Lever, S. H. Huang, D. Tirumala, M. Wulfmeier, J. Humplik, S. Tunyasuvunakool, N. Y. Siegel, R. Hafner et al., “Learning agile soccer skills for a bipedal robot with deep reinforcement learning,” arXiv preprint arXiv:2304.13653, 2023.   
K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked autoencoders are scalable vision learners,” CVPR, 2022.   
S. Hemachandra, F. Duvallet, T. M. Howard, N. Roy, A. Stentz, and M. R. Walter, “Learning models for following natural language directions in unknown environments,” arXiv preprint arXiv:1503.05079, 2015.   
J. Henrich, S. J. Heine, and A. Norenzayan, “The weirdest people in the world?” Behavioral and Brain Sciences, vol. 33, no. 2-3, p. 61–83, 2010.   
L. B. Hensel, N. Yongsatianchot, P. Torshizi, E. Minucci, and S. Marsella, “Large language models in textual analysis for gesture selection,” in INTERNATIONAL CONFERENCE ON MULTIMODAL INTERACTION, 2023, pp. 378–387.   
D. Ho, K. Rao, Z. Xu, E. Jang, M. Khansari, and Y. Bai, “Retinagan: An object-aware approach to sim-to-real transfer,” in 2021 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2021, pp. 10 920–10 926.   
C. Huang, O. Mees, A. Zeng, and W. Burgard, “Visual language maps for robot navigation,” in 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023, pp. 10 608–10 615.   
Q. Huang, J. S. Park, A. Gupta, P. Bennett, R. Gong, S. Som, B. Peng, O. K. Mohammed, C. Pal, Y. Choi et al., “Ark: Augmented reality with knowledge interactive emergent ability,” arXiv preprint arXiv:2305.00970, 2023.   
W. Huang, P. Abbeel, D. Pathak, and I. Mordatch, “Language models as zero-shot planners: Extracting actionable knowledge for embodied agents,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, Eds., vol. 162. PMLR, 17–23 Jul 2022, pp. 9118–9147. [Online]. Available: https://proceedings.mlr.press/v162/huang22a.html   
W. Huang, F. Xia, T. Xiao, H. Chan, J. Liang, P. Florence, A. Zeng, J. Tompson, I. Mordatch, Y. Chebotar, P. Sermanet, N. Brown, T. Jackson, L. Luu, S. Levine, K. Hausman, and B. Ichter, “Inner monologue: Embodied reasoning through planning with language models,” in arXiv preprint arXiv:2207.05608, 2022.   
Z. Huang, H. Feng, Z. Chongzhi, L. Sheng, L. Ziwei, and J. Shao, “Dolphin: General video interaction platform based on llms,” 2023, https://github.com/kaleido-lab/dolphin.

K. Ikeuchi, N. Wake, K. Sasabuchi, and J. Takamatsu, “Semantic constraints to represent common sense required in household actions for multimodal learning-from-observation robot,” The International Journal of Robotics Research, vol. 0, no. 0, p. 02783649231212929, 0.   
K. Ikeuchi, J. Takamatsu, K. Sasabuchi, N. Wake, and A. Kanehiro, “Applying learning-from-observation to household service robots: three common-sense formulation,” arXiv preprint arXiv:2304.09966, 2023.   
S. Imani, L. Du, and H. Shrivastava, “Mathprompter: Mathematical reasoning using large language models,” arXiv preprint arXiv:2303.05398, 2023.   
S. James and A. J. Davison, “Q-attention: Enabling efficient learning for vision-based robotic manipulation,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 1612–1619, 2022.   
E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, “Bc-z: Zero-shot task generalization with robotic imitation learning,” in Conference on Robot Learning. PMLR, 2022, pp. 991–1002.   
Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, and P. Fung, “Survey of hallucination in natural language generation,” ACM Computing Surveys, vol. 55, no. 12, pp. 1–38, 2023.   
Y. Jiang, A. Gupta, Z. Zhang, G. Wang, Y. Dou, Y. Chen, L. Fei-Fei, A. Anandkumar, Y. Zhu, and L. Fan, “Vima: General robot manipulation with multimodal prompts,” arXiv, 2022.   
D. Kalashnikov, A. Irpan, P. Pastor, J. Ibarz, A. Herzog, E. Jang, D. Quillen, E. Holly, M. Kalakrishnan, V. Vanhoucke et al., “Scalable deep reinforcement learning for vision-based robotic manipulation,” in Conference on Robot Learning. PMLR, 2018, pp. 651–673.   
A. Karpathy, A. Joulin, and L. F. Fei-Fei, “Deep fragment embeddings for bidirectional image sentence mapping,” Advances in neural information processing systems, vol. 27, 2014.   
P. Katara, Z. Xian, and K. Fragkiadaki, “Gen2sim: Scaling up robot learning in simulation with generative models,” arXiv preprint arXiv:2310.18308, 2023.   
L. Ke, X. Li, B. Yonatan, A. Holtzman, Z. Gan, J. Liu, J. Gao, Y. Choi, and S. Srinivasa, “Tactical rewind: Selfcorrection via backtracking in vision-and-language navigation,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019.   
J. Kim, J. Kim, and S. Choi, “Flame: Free-form language-based motion synthesis & editing,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 7, 2023, pp. 8255–8263.   
P.-C. Ko, J. Mao, Y. Du, S.-H. Sun, and J. B. Tenenbaum, “Learning to act from actionless videos through dense correspondences,” arXiv preprint arXiv:2310.08576, 2023.   
E. Kolve, R. Mottaghi, W. Han, E. VanderBilt, L. Weihs, A. Herrasti, M. Deitke, K. Ehsani, D. Gordon, Y. Zhu et al., “Ai2-thor: An interactive 3d environment for visual ai,” arXiv preprint arXiv:1712.05474, 2017.   
R. Krishna, Y. Zhu, O. Groth, J. Johnson, K. Hata, J. Kravitz, S. Chen, Y. Kalantidis, L.-J. Li, D. A. Shamma, M. Bernstein, and L. Fei-Fei, “Visual genome: Connecting language and vision using crowdsourced dense image annotations,” in arXiv:1602.07332, 2016.   
K. N. Kumar, I. Essa, and S. Ha, “Words into action: Learning diverse humanoid robot behaviors using language guided iterative motion refinement,” arXiv preprint arXiv:2310.06226, 2023.   
P. Lee, S. Bubeck, and J. Petro, “Benefits, limits, and risks of gpt-4 as an ai chatbot for medicine,” New England Journal of Medicine, vol. 388, no. 13, pp. 1233–1239, 2023.   
P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel et al., “Retrieval-augmented generation for knowledge-intensive nlp tasks,” Advances in Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.   
B. Li, P. Wu, P. Abbeel, and J. Malik, “Interactive task planning with language models,” arXiv preprint arXiv:2310.10645, 2023.

C. Li, F. Xia, R. Martín-Martín, M. Lingelbach, S. Srivastava, B. Shen, K. Vainio, C. Gokmen, G. Dharan, T. Jain et al., “igibson 2.0: Object-centric simulation for robot learning of everyday household tasks,” arXiv preprint arXiv:2108.03272, 2021.   
C. Li, C. Wong, S. Zhang, N. Usuyama, H. Liu, J. Yang, T. Naumann, H. Poon, and J. Gao, “Llava-med: Training a large language-and-vision assistant for biomedicine in one day,” arXiv preprint arXiv:2306.00890, 2023.   
G. Li, H. A. A. K. Hammoud, H. Itani, D. Khizbullin, and B. Ghanem, “Camel: Communicative agents for" mind" exploration of large scale language model society,” arXiv preprint arXiv:2303.17760, 2023.   
J. Li, Q. Gao, M. Johnston, X. Gao, X. He, S. Shakiah, H. Shi, R. Ghanadan, and W. Y. Wang, “Mastering robot manipulation with multimodal prompts through pretraining and multi-task fine-tuning,” arXiv preprint arXiv:2310.09676, 2023.   
J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models,” arXiv preprint arXiv:2301.12597, 2023.   
K. Li, Y. He, W. Yi, Y. Li, W. Wang, P. Luo, Y. Wang, L. Wang, and Y. Qiao, “Videochat: Chat-centric video understanding,” arXiv preprint arXiv:2305.06355, 2023.   
L. Li, Y.-C. Chen, Y. Cheng, Z. Gan, L. Yu, and J. Liu, “Hero: Hierarchical encoder for video+language omnirepresentation pre-training,” 2020.   
L. Li, J. Lei, Z. Gan, L. Yu, Y.-C. Chen, R. Pillai, Y. Cheng, L. Zhou, X. E. Wang, W. Y. Wang, T. L. Berg, M. Bansal, J. Liu, L. Wang, and Z. Liu, “Value: A multi-task benchmark for video-and-language understanding evaluation,” 2021.   
X. Li, M. Liu, H. Zhang, C. Yu, J. Xu, H. Wu, C. Cheang, Y. Jing, W. Zhang, H. Liu et al., “Vision-language foundation models as effective robot imitators,” arXiv preprint arXiv:2311.01378, 2023.   
J. Liang, W. Huang, F. Xia, P. Xu, K. Hausman, B. Ichter, P. Florence, and A. Zeng, “Code as policies: Language model programs for embodied control,” in arXiv preprint arXiv:2209.07753, 2022.   
X. Liang, L. Ma, S. Guo, J. Han, H. Xu, S. Ma, and X. Liang, “Mo-vln: A multi-task benchmark for open-set zero-shot vision-and-language navigation,” arXiv preprint arXiv:2306.10322, 2023.   
S. Lifshitz, K. Paster, H. Chan, J. Ba, and S. McIlraith, “Steve-1: A generative model for text-to-behavior in minecraft,” arXiv preprint arXiv:2306.00937, 2023.   
K. Lin, F. Ahmed, L. Li, C.-C. Lin, E. Azarnasab, Z. Yang, J. Wang, L. Liang, Z. Liu, Y. Lu, C. Liu, and L. Wang, “Mm-vid: Advancing video understanding with gpt-4v(ision),” 2023.   
T.-Y. Lin, M. Maire, S. Belongie, L. Bourdev, R. Girshick, J. Hays, P. Perona, D. Ramanan, C. L. Zitnick, and P. Dollár, “Microsoft coco: Common objects in context,” Proceedings of ECCV, 2014.   
C. K. Liu and D. Negrut, “The role of physics-based simulators in robotics,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 4, pp. 35–58, 2021.   
H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” 2023.   
H. Liu, A. Chen, Y. Zhu, A. Swaminathan, A. Kolobov, and C.-A. Cheng, “Interactive robot learning from verbal correction,” arXiv preprint arXiv:2310.17555, 2023.   
Y. Liu, W. Held, and D. Yang, “Dada: Dialect adaptation via dynamic aggregation of linguistic rules,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023.   
P. Lu, B. Peng, H. Cheng, M. Galley, K.-W. Chang, Y. N. Wu, S.-C. Zhu, and J. Gao, “Chameleon: Plug-and-play compositional reasoning with large language models,” 2023.

Z. Luo, Z. Durante, L. Li, W. Xie, R. Liu, E. Jin, Z. Huang, L. Y. Li, J. Wu, J. C. Niebles et al., “Moma-lrg: Languagerefined graphs for multi-object multi-actor activity parsing,” Advances in Neural Information Processing Systems, vol. 35, pp. 5282–5298, 2022.   
C.-Y. Ma, J. Lu, Z. Wu, G. AlRegib, Z. Kira, R. Socher, and C. Xiong, “Self-monitoring navigation agent via auxiliary progress estimation,” arXiv preprint arXiv:1901.03035, 2019.   
C.-Y. Ma, Z. Wu, G. AlRegib, C. Xiong, and Z. Kira, “The regretful agent: Heuristic-aided navigation through progress estimation,” arXiv preprint arXiv:1903.01602, 2019.   
Y. J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar, “Eureka: Human-level reward design via coding large language models,” arXiv preprint arXiv:2310.12931, 2023.   
M. Maaz, H. Rasheed, S. Khan, and F. S. Khan, “Video-chatgpt: Towards detailed video understanding via large vision and language models,” 2023.   
R. Mao, Q. Liu, K. He, W. Li, and E. Cambria, “The biases of pre-trained language models: An empirical study on prompt-based sentiment analysis and emotion detection,” IEEE Transactions on Affective Computing, 2022.   
G. Marcus, “The next decade in ai: four steps towards robust artificial intelligence,” arXiv preprint arXiv:2002.06177, 2020.   
G. Marcus and E. Davis, Rebooting AI: Building artificial intelligence we can trust. Pantheon, 2019.   
K. Marino, M. Rastegari, A. Farhadi, and R. Mottaghi, “Ok-vqa: A visual question answering benchmark requiring external knowledge,” in CVPR, 2019.   
P. Martinez-Gonzalez, S. Oprea, A. Garcia-Garcia, A. Jover-Alvarez, S. Orts-Escolano, and J. Garcia-Rodriguez, “Unrealrox: an extremely photorealistic virtual reality environment for robotics simulations and synthetic data generation,” Virtual Reality, vol. 24, pp. 271–288, 2020.   
J. Maynez, S. Narayan, B. Bohnet, and R. McDonald, “On faithfulness and factuality in abstractive summarization,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds. Online: Association for Computational Linguistics, Jul. 2020, pp. 1906–1919. [Online]. Available: https://aclanthology.org/2020.acl-main.173   
O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard, “Calvin: A benchmark for language-conditioned policy learning for long-horizon robot manipulation tasks,” IEEE Robotics and Automation Letters, vol. 7, no. 3, pp. 7327–7334, 2022.   
Meta Fundamental AI Research (FAIR) Diplomacy Team, A. Bakhtin, N. Brown, E. Dinan, G. Farina, C. Flaherty, D. Fried, A. Goff, J. Gray, H. Hu et al., “Human-level play in the game of Diplomacy by combining language models with strategic reasoning,” Science, vol. 378, no. 6624, pp. 1067–1074, 2022.   
S. Min, X. Lyu, A. Holtzman, M. Artetxe, M. Lewis, H. Hajishirzi, and L. Zettlemoyer, “Rethinking the role of demonstrations: What makes in-context learning work?” arXiv preprint arXiv:2202.12837, 2022.   
M. L. Minsky, “Minsky’s frame system theory,” in Proceedings of the 1975 Workshop on Theoretical Issues in Natural Language Processing, ser. TINLAP ’75. USA: Association for Computational Linguistics, 1975, p. 104–116. [Online]. Available: https://doi.org/10.3115/980190.980222   
S. Mirchandani, F. Xia, P. Florence, B. Ichter, D. Driess, M. G. Arenas, K. Rao, D. Sadigh, and A. Zeng, “Large language models as general pattern machines,” arXiv preprint arXiv:2307.04721, 2023.   
P. Mirowski, R. Pascanu, F. Viola, H. Soyer, A. J. Ballard, A. Banino, M. Denil, R. Goroshin, L. Sifre, K. Kavukcuoglu et al., “Learning to navigate in complex environments,” arXiv preprint arXiv:1611.03673, 2016.   
M. Mittal, C. Yu, Q. Yu, J. Liu, N. Rudin, D. Hoeller, J. L. Yuan, R. Singh, Y. Guo, H. Mazhar et al., “Orbit: A unified simulation framework for interactive robot learning environments,” IEEE Robotics and Automation Letters, 2023.

V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K. Fidjeland, G. Ostrovski et al., “Human-level control through deep reinforcement learning,” nature, vol. 518, no. 7540, pp. 529–533, 2015.   
A. Mousavian, A. Toshev, M. Fiser, J. Kosecka, and J. Davidson, “Visual representations for semantic target driven navigation,” arXiv preprint arXiv:1805.06066, 2018.   
T. Mu, Z. Ling, F. Xiang, D. Yang, X. Li, S. Tao, Z. Huang, Z. Jia, and H. Su, “Maniskill: Generalizable manipulation skill benchmark with large-scale demonstrations,” arXiv preprint arXiv:2107.14483, 2021.   
M. Müller, V. Casser, J. Lahoud, N. Smith, and B. Ghanem, “Sim4cv: A photo-realistic simulator for computer vision applications,” International Journal of Computer Vision, vol. 126, pp. 902–919, 2018.   
J. J. Nay, “Law informs code: A legal informatics approach to aligning artificial intelligence with humans,” Nw. J. Tech. & Intell. Prop., vol. 20, p. 309, 2022.   
K. Nguyen, D. Dey, C. Brockett, and B. Dolan, “Vision-based navigation with language-based assistance via imitation learning with indirect intervention,” arXiv preprint arXiv:1812.04155, 2018.   
Z. Ni, X.-X. Deng, C. Tai, X.-Y. Zhu, X. Wu, Y.-J. Liu, and L. Zeng, “Grid: Scene-graph-based instruction-driven robotic task planning,” arXiv preprint arXiv:2309.07726, 2023.

OpenAI, “GPT-4 technical report,” OpenAI, Tech. Rep., 2023.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in Neural Information Processing Systems, vol. 35, pp. 27 730–27 744, 2022.   
A. Padalkar, A. Pooley, A. Jain, A. Bewley, A. Herzog, A. Irpan, A. Khazatsky, A. Rai, A. Singh, A. Brohan et al., “Open x-embodiment: Robotic learning datasets and rt-x models,” arXiv preprint arXiv:2310.08864, 2023.   
M. Parakh, A. Fong, A. Simeonov, A. Gupta, T. Chen, and P. Agrawal, “Human-assisted continual robot learning with foundation models,” arXiv preprint arXiv:2309.14321, 2023.   
J. S. Park, J. Hessel, K. Chandu, P. P. Liang, X. Lu, P. West, Q. Huang, J. Gao, A. Farhadi, and Y. Choi, “Multimodal agent – localized symbolic knowledge distillation for visual commonsense models,” in NeurIPS 2023, October 2023.   
J. S. Park, J. Hessel, K. Chandu, P. P. Liang, X. Lu, P. West, Y. Yu, Q. Huang, J. Gao, A. Farhadi, and Y. Choi, “Localized symbolic knowledge distillation for visual commonsense models,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023. [Online]. Available: https://openreview.net/forum?id=V5eG47pyVl   
J. Park, Q. Huang, Y. Bisk, J. Yang, S. Som, A. Farhadi, Y. Choi, and J. Gao, “Ink: Intensive neural knowledge,” July 2022.   
J. S. Park, J. C. O’Brien, C. J. Cai, M. R. Morris, P. Liang, and M. S. Bernstein, “Generative agents: Interactive simulacra of human behavior,” arXiv preprint arXiv:2304.03442, 2023.   
B. Peng, M. Galley, P. He, H. Cheng, Y. Xie, Y. Hu, Q. Huang, L. Liden, Z. Yu, W. Chen et al., “Check your facts and try again: Improving large language models with external knowledge and automated feedback,” arXiv preprint arXiv:2302.12813, 2023.   
X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, “Sim-to-real transfer of robotic control with dynamics randomization,” in 2018 IEEE international conference on robotics and automation (ICRA). IEEE, 2018, pp. 3803–3810.   
X. Puig, K. Ra, M. Boben, J. Li, T. Wang, S. Fidler, and A. Torralba, “Virtualhome: Simulating household activities via programs,” in 2018 IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 8494–8502.   
X. Puig, E. Undersander, A. Szot, M. D. Cote, T.-Y. Yang, R. Partsey, R. Desai, A. W. Clegg, M. Hlavac, S. Y. Min et al., “Habitat 3.0: A co-habitat for humans, avatars and robots,” arXiv preprint arXiv:2310.13724, 2023.

A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.   
S. K. Ramakrishnan, D. S. Chaplot, Z. Al-Halah, J. Malik, and K. Grauman, “Poni: Potential functions for objectgoal navigation with interaction-free learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 18 890–18 900.   
S. S. Raman, V. Cohen, D. Paulius, I. Idrees, E. Rosen, R. Mooney, and S. Tellex, “Cape: Corrective actions from precondition errors using large language models,” in 2nd Workshop on Language and Robot Learning: Language as Grounding, 2023.   
K. Rao, C. Harris, A. Irpan, S. Levine, J. Ibarz, and M. Khansari, “Rl-cyclegan: Reinforcement learning aware simulation-to-real,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11 157–11 166.   
V. Raunak, A. Menezes, and M. Junczys-Dowmunt, “The curious case of hallucinations in neural machine translation,” arXiv preprint arXiv:2104.06683, 2021.   
S. Reed, K. Zolna, E. Parisotto, S. G. Colmenarejo, A. Novikov, G. Barth-Maron, M. Gimenez, Y. Sulsky, J. Kay, J. T. Springenberg et al., “A generalist agent,” arXiv preprint arXiv:2205.06175, 2022.   
M. Ren, R. Kiros, and R. Zemel, “Exploring models and data for image question answering,” Advances in neural information processing systems, vol. 28, 2015.   
A. Rohrbach, L. A. Hendricks, K. Burns, T. Darrell, and K. Saenko, “Object hallucination in image captioning,” arXiv preprint arXiv:1809.02156, 2018.   
A. Rosinol, J. J. Leonard, and L. Carlone, “Nerf-slam: Real-time dense monocular slam with neural radiance fields,” arXiv preprint arXiv:2210.13641, 2022.   
F. Sadeghi and S. Levine, “Cad2rl: Real single-image flight without a single real image,” arXiv preprint arXiv:1611.04201, 2016.   
D. Saito, K. Sasabuchi, N. Wake, J. Takamatsu, H. Koike, and K. Ikeuchi, “Task-grasping from a demonstrated human strategy,” in 2022 IEEE-RAS 21st International Conference on Humanoid Robots (Humanoids), 2022, pp. 880–887.   
D. Saito, K. Sasabuchi, N. Wake, A. Kanehira, J. Takamatsu, H. Koike, and K. Ikeuchi, “Constraint-aware policy for compliant manipulation,” 2023.   
B. Sarkar, A. Shih, and D. Sadigh, “Diverse conventions for human-AI collaboration,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023.   
K. Sasabuchi, N. Wake, and K. Ikeuchi, “Task-oriented motion mapping on robots of various configuration using body role division,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 413–420, 2021.   
K. Sasabuchi, D. Saito, A. Kanehira, N. Wake, J. Takamatsu, and K. Ikeuchi, “Task-sequencing simulator: Integrated machine learning to execution simulation for robot manipulation,” arXiv preprint arXiv:2301.01382, 2023.   
M. Savva, A. X. Chang, A. Dosovitskiy, T. Funkhouser, and V. Koltun, “Minos: Multimodal indoor simulator for navigation in complex environments,” arXiv preprint arXiv:1712.03931, 2017.   
M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun, J. Malik et al., “Habitat: A platform for embodied ai research,” in Proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 9339–9347.   
T. Schick, J. Dwivedi-Yu, R. Dessì, R. Raileanu, M. Lomeli, L. Zettlemoyer, N. Cancedda, and T. Scialom, “Toolformer: Language models can teach themselves to use tools,” 2023.   
E. Segalis, D. Valevski, D. Lumen, Y. Matias, and Y. Leviathan, “A picture is worth a thousand words: Principled recaptioning improves image generation,” arXiv preprint arXiv:2310.16656, 2023.

B. Shacklett, L. G. Rosenzweig, Z. Xie, B. Sarkar, A. Szot, E. Wijmans, V. Koltun, D. Batra, and K. Fatahalian, “An extensible, data-oriented architecture for high-performance, many-world simulation,” ACM Trans. Graph., vol. 42, no. 4, 2023.   
D. Shah, B. Osinski, S. Levine ´ et al., “Lm-nav: Robotic navigation with large pre-trained models of language, vision, and action,” in Conference on Robot Learning. PMLR, 2023, pp. 492–504.   
R. Shah, R. Martín-Martín, and Y. Zhu, “Mutex: Learning unified policies from multimodal task specifications,” arXiv preprint arXiv:2309.14320, 2023.   
S. Shah, D. Dey, C. Lovett, and A. Kapoor, “Airsim: High-fidelity visual and physical simulation for autonomous vehicles,” in Field and Service Robotics: Results of the 11th International Conference. Springer, 2018, pp. 621–635.   
P. Sharma, N. Ding, S. Goodman, and R. Soricut, “Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning,” Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, 2018.   
R. Shi, Y. Liu, Y. Ze, S. S. Du, and H. Xu, “Unleashing the power of pre-trained language models for offline reinforcement learning,” arXiv preprint arXiv:2310.20587, 2023.   
M. Shridhar, L. Manuelli, and D. Fox, “Perceiver-actor: A multi-task transformer for robotic manipulation,” in Conference on Robot Learning. PMLR, 2023, pp. 785–799.   
K. Shuster, S. Poff, M. Chen, D. Kiela, and J. Weston, “Retrieval augmentation reduces hallucination in conversation,” arXiv preprint arXiv:2104.07567, 2021.   
A. Singh, V. Natarajan, M. Shah, Y. Jiang, X. Chen, D. Batra, D. Parikh, and M. Rohrbach, “Towards vqa models that can read,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 8317–8326.   
S. Song, F. Yu, A. Zeng, A. X. Chang, M. Savva, and T. Funkhouser, “Semantic scene completion from a single depth image,” IEEE Conference on Computer Vision and Pattern Recognition, 2017.   
S. A. Sontakke, J. Zhang, S. M. Arnold, K. Pertsch, E. Bıyık, D. Sadigh, C. Finn, and L. Itti, “Roboclip: One demonstration is enough to learn robot policies,” arXiv preprint arXiv:2310.07899, 2023.   
S. Srivastava, C. Li, M. Lingelbach, R. Martín-Martín, F. Xia, K. E. Vainio, Z. Lian, C. Gokmen, S. Buch, K. Liu et al., “Behavior: Benchmark for everyday household activities in virtual, interactive, and ecological environments,” in Conference on Robot Learning. PMLR, 2022, pp. 477–490.   
J. Sun, Q. Zhang, Y. Duan, X. Jiang, C. Cheng, and R. Xu, “Prompt, plan, perform: Llm-based humanoid control via quantized imitation learning,” arXiv preprint arXiv:2309.11359, 2023.   
Q. Sun, Y. Fang, L. Wu, X. Wang, and Y. Cao, “Eva-clip: Improved training techniques for clip at scale,” arXiv preprint arXiv:2303.15389, 2023.   
A. Szot, A. Clegg, E. Undersander, E. Wijmans, Y. Zhao, J. Turner, N. Maestre, M. Mukadam, D. Chaplot, O. Maksymets, A. Gokaslan, V. Vondrus, S. Dharur, F. Meier, W. Galuba, A. Chang, Z. Kira, V. Koltun, J. Malik, M. Savva, and D. Batra, “Habitat 2.0: Training home assistants to rearrange their habitat,” in Advances in Neural Information Processing Systems (NeurIPS), 2021.   
J. Takamatsu, K. Sasabuchi, N. Wake, A. Kanehira, and K. Ikeuchi, “Learning-from-observation system considering hardware-level reusability,” arXiv preprint arXiv:2212.09242, 2022.   
C. Tang, D. Huang, W. Ge, W. Liu, and H. Zhang, “Graspgpt: Leveraging semantic knowledge from a large language model for task-oriented grasping,” IEEE Robotics and Automation Letters, 2023.   
R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto, “Stanford alpaca: An instruction-following llama model,” https://github.com/tatsu-lab/stanford_alpaca, 2023.

H. Teshima, N. Wake, D. Thomas, Y. Nakashima, H. Kawasaki, and K. Ikeuchi, “Deep gesture generation for social robots using type-specific libraries,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2022, pp. 8286–8291.   
G. Tevet, S. Raab, B. Gordon, Y. Shafir, D. Cohen-Or, and A. H. Bermano, “Human motion diffusion model,” arXiv preprint arXiv:2209.14916, 2022.   
J. Thomason, D. Gordan, and Y. Bisk, “Shifting the baseline: Single modality performance on visual navigation & qa,” arXiv preprint arXiv:1811.00613, 2018.   
J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel, “Domain randomization for transferring deep neural networks from simulation to the real world,” in 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2017, pp. 23–30.   
H. Touvron, T. Lavril, G. Izacard, X. Martinet, M.-A. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar et al., “Llama: Open and efficient foundation language models,” arXiv preprint arXiv:2302.13971, 2023.   
N. Tsoi, A. Xiang, P. Yu, S. S. Sohn, G. Schwartz, S. Ramesh, M. Hussein, A. W. Gupta, M. Kapadia, and M. Vázquez, “Sean 2.0: Formalizing and generating social situations for robot navigation,” IEEE Robotics and Automation Letters, vol. 7, no. 4, pp. 11 047–11 054, 2022.   
N. Wake, R. Arakawa, I. Yanokura, T. Kiyokawa, K. Sasabuchi, J. Takamatsu, and K. Ikeuchi, “A learning-fromobservation framework: One-shot robot teaching for grasp-manipulation-release household operations,” in 2021 IEEE/SICE International Symposium on System Integration (SII). IEEE, 2021.   
N. Wake, I. Yanokura, K. Sasabuchi, and K. Ikeuchi, “Verbal focus-of-attention system for learning-from-observation,” in 2021 IEEE International Conference on Robotics and Automation (ICRA), 2021, pp. 10 377–10 384.   
N. Wake, A. Kanehira, K. Sasabuchi, J. Takamatsu, and K. Ikeuchi, “Interactive task encoding system for learning-fromobservation,” in 2023 IEEE/ASME International Conference on Advanced Intelligent Mechatronics (AIM), 2023, pp. 1061–1066.   
—, “Bias in emotion recognition with chatgpt,” arXiv preprint arXiv:2310.11753, 2023.   
—, “Gpt models meet robotic applications: Co-speech gesturing chat system,” arXiv preprint arXiv:2306.01741, 2023.   
—, “Gpt-4v(ision) for robotics: Multimodal task planning from human demonstration,” arXiv preprint arXiv:2311.12015, 2023.   
—, “Chatgpt empowered long-step robot control in various environments: A case application,” IEEE Access, vol. 11, pp. 95 060–95 078, 2023.   
N. Wake, D. Saito, K. Sasabuchi, H. Koike, and K. Ikeuchi, “Text-driven object affordance for guiding grasp-type recognition in multimodal robot teaching,” Machine Vision and Applications, vol. 34, no. 4, p. 58, 2023.   
B. Wang, Q. Huang, B. Deb, A. L. Halfaker, L. Shao, D. McDuff, A. Awadallah, D. Radev, and J. Gao, “Logical transformers: Infusing logical structures into pre-trained language models,” in Proceedings of ACL 2023, May 2023.   
D. Wang, Q. Huang, M. Jackson, and J. Gao, “Retrieve what you need: A mutual learning framework for open-domain question answering,” March 2023. [Online]. Available: https://www.microsoft.com/en-us/research/publication/ retrieve-what-you-need-a-mutual-learning-framework-for-open-domain-question-answering/   
G. Wang, Y. Xie, Y. Jiang, A. Mandlekar, C. Xiao, Y. Zhu, L. Fan, and A. Anandkumar, “Voyager: An open-ended embodied agent with large language models,” arXiv preprint arXiv:2305.16291, 2023.   
L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin et al., “A survey on large language model based autonomous agents,” arXiv preprint arXiv:2308.11432, 2023.   
P. Wang, Q. Wu, C. Shen, A. v. d. Hengel, and A. Dick, “Explicit knowledge-based reasoning for visual question answering,” arXiv preprint arXiv:1511.02570, 2015.

P. Wang, Q. Wu, C. Shen, A. Dick, and A. Van Den Hengel, “Fvqa: Fact-based visual question answering,” TPAMI, vol. 40, no. 10, pp. 2413–2427, 2017.   
X. Wang, W. Xiong, H. Wang, and W. Y. Wang, “Look before you leap: Bridging model-free and model-based reinforcement learning for planned-ahead vision-and-language navigation,” in The European Conference on Computer Vision (ECCV), September 2018.   
X. Wang, Q. Huang, A. Celikyilmaz, J. Gao, D. Shen, Y.-F. Weng, W. Y. Wang, and L. Zhang, “Reinforced cross-modal matching and self-supervised imitation learning for vision-language navigation,” in CVPR 2019, June 2019.   
Y. Wang, Y. He, Y. Li, K. Li, J. Yu, X. Ma, X. Chen, Y. Wang, P. Luo, Z. Liu, Y. Wang, L. Wang, and Y. Qiao, “Internvid: A large-scale video-text dataset for multimodal understanding and generation,” 2023.   
Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, and H. Hajishirzi, “Self-instruct: Aligning language model with self generated instructions,” arXiv preprint arXiv:2212.10560, 2022.   
Y. Wang, Z. Xian, F. Chen, T.-H. Wang, Y. Wang, K. Fragkiadaki, Z. Erickson, D. Held, and C. Gan, “Robogen: Towards unleashing infinite data for automated robot learning via generative simulation,” arXiv preprint arXiv:2311.01455, 2023.   
Z. Wang, Y. Chen, T. Liu, Y. Zhu, W. Liang, and S. Huang, “Humanise: Language-conditioned human motion generation in 3d scenes,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 14 959–14 971. [Online]. Available: https://proceedings.neurips.cc/paper_files/paper/2022/file/ 6030db5195150ac86d942186f4abdad8-Paper-Conference.pdf   
Z. Wang, S. Cai, A. Liu, X. Ma, and Y. Liang, “Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents,” arXiv preprint arXiv:2302.01560, 2023.   
J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in Neural Information Processing Systems, vol. 35, pp. 24 824–24 837, 2022.   
World Health Organization and World Bank, “Tracking universal health coverage: First global monitoring report,” www.who.int/healthinfo/universal_health_coverage/report/2015/en, Jun 2015.   
Q. Wu, G. Bansal, J. Zhang, Y. Wu, S. Zhang, E. E. Zhu, B. Li, L. Jiang, X. Zhang, and C. Wang, “Autogen: Enabling next-gen llm applications via multi-agent conversation,” Microsoft, Tech. Rep. MSR-TR-2023-33, August 2023. [Online]. Available: https://www.microsoft.com/en-us/research/publication/ autogen-enabling-next-gen-llm-applications-via-multi-agent-conversation-framework/   
Z. Xi, W. Chen, X. Guo, W. He, Y. Ding, B. Hong, M. Zhang, J. Wang, S. Jin, E. Zhou et al., “The rise and potential of large language model based agents: A survey,” arXiv preprint arXiv:2309.07864, 2023.   
F. Xia, A. R. Zamir, Z.-Y. He, A. Sax, J. Malik, and S. Savarese, “Gibson Env: real-world perception for embodied agents,” in Computer Vision and Pattern Recognition (CVPR), 2018 IEEE Conference on. IEEE, 2018.   
M. Xu, P. Huang, W. Yu, S. Liu, X. Zhang, Y. Niu, T. Zhang, F. Xia, J. Tan, and D. Zhao, “Creative robot tool use with large language models,” arXiv preprint arXiv:2310.13065, 2023.   
J. Yang, Y. Dong, S. Liu, B. Li, Z. Wang, C. Jiang, H. Tan, J. Kang, Y. Zhang, K. Zhou et al., “Octopus: Embodied vision-language programmer from environmental feedback,” arXiv preprint arXiv:2310.08588, 2023.   
K. Yang, S. Ji, T. Zhang, Q. Xie, and S. Ananiadou, “On the evaluations of chatgpt and emotion-enhanced prompting for mental health analysis,” arXiv preprint arXiv:2304.03347, 2023.   
Z. Yang, L. Li, J. Wang, K. Lin, E. Azarnasab, F. Ahmed, Z. Liu, C. Liu, M. Zeng, and L. Wang, “Mm-react: Prompting chatgpt for multimodal reasoning and action,” 2023.

S. Yao, D. Yu, J. Zhao, I. Shafran, T. L. Griffiths, Y. Cao, and K. Narasimhan, “Tree of thoughts: Deliberate problem solving with large language models,” 2023.   
S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” 2023.   
Q. Ye, H. Xu, G. Xu, J. Ye, M. Yan, Y. Zhou, J. Wang, A. Hu, P. Shi, Y. Shi, C. Li, Y. Xu, H. Chen, J. Tian, Q. Qi, J. Zhang, and F. Huang, “mplug-owl: Modularization empowers large language models with multimodality,” 2023.   
Y. Ye, H. You, and J. Du, “Improved trust in human-robot collaboration with chatgpt,” IEEE Access, 2023.   
P. Young, A. Lai, M. Hodosh, and J. Hockenmaier, “From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions,” Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2014.   
J. Yu, X. Wang, S. Tu, S. Cao, D. Zhang-Li, X. Lv, H. Peng, Z. Yao, X. Zhang, H. Li et al., “Kola: Carefully benchmarking world knowledge of large language models,” arXiv preprint arXiv:2306.09296, 2023.   
L. Yu, P. Poirson, S. Yang, A. C. Berg, and T. L. Berg, “Modeling context in referring expressions,” in Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part II 14. Springer, 2016, pp. 69–85.   
W. Yu, N. Gileadi, C. Fu, S. Kirmani, K.-H. Lee, M. G. Arenas, H.-T. L. Chiang, T. Erez, L. Hasenclever, J. Humplik et al., “Language to rewards for robotic skill synthesis,” arXiv preprint arXiv:2306.08647, 2023.   
R. Zellers, X. Lu, J. Hessel, Y. Yu, J. S. Park, J. Cao, A. Farhadi, and Y. Choi, “Merlot: Multimodal neural script knowledge models,” 2021.   
R. Zellers, J. Lu, X. Lu, Y. Yu, Y. Zhao, M. Salehi, A. Kusupati, J. Hessel, A. Farhadi, and Y. Choi, “Merlot reserve: Neural script knowledge through vision and language and sound,” 2022.   
A. Zeng, P. Florence, J. Tompson, S. Welker, J. Chien, M. Attarian, T. Armstrong, I. Krasin, D. Duong, V. Sindhwani et al., “Transporter networks: Rearranging the visual world for robotic manipulation,” in Conference on Robot Learning. PMLR, 2021, pp. 726–747.   
A. Zeng, M. Liu, R. Lu, B. Wang, X. Liu, Y. Dong, and J. Tang, “Agenttuning: Enabling generalized agent abilities for llms,” 2023.   
L. Zha, Y. Cui, L.-H. Lin, M. Kwon, M. G. Arenas, A. Zeng, F. Xia, and D. Sadigh, “Distilling and retrieving generalizable knowledge for robot manipulation via language corrections,” arXiv preprint arXiv:2311.10678, 2023.   
M. Zhang, Z. Cai, L. Pan, F. Hong, X. Guo, L. Yang, and Z. Liu, “Motiondiffuse: Text-driven human motion generation with diffusion model,” arXiv preprint arXiv:2208.15001, 2022.   
S. Zhang, X. Song, Y. Bai, W. Li, Y. Chu, and S. Jiang, “Hierarchical object-to-zone graph for object navigation,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 15 130–15 140.   
W. Zhao, Y. Zhao, X. Lu, S. Wang, Y. Tong, and B. Qin, “Is chatgpt equipped with emotional dialogue capabilities?” arXiv preprint arXiv:2304.09582, 2023.   
Y. Zhao, I. Misra, P. Krähenbühl, and R. Girdhar, “Learning video representations from large language models,” in arXiv preprint arXiv:2212.04501, 2022.   
L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, “Judging llm-as-a-judge with mt-bench and chatbot arena,” 2023.   
Z. Zhong, J. Cao, S. Gu, S. Xie, W. Gao, L. Luo, Z. Yan, H. Zhao, and G. Zhou, “Assist: Interactive scene nodes for scalable and realistic indoor simulation,” arXiv preprint arXiv:2311.06211, 2023.   
G. Zhou, Y. Hong, and Q. Wu, “Navgpt: Explicit reasoning in vision-and-language navigation with large language models,” arXiv preprint arXiv:2305.16986, 2023.

H. Zhou, M. Ding, W. Peng, M. Tomizuka, L. Shao, and C. Gan, “Generalizable long-horizon manipulations with large language models,” arXiv preprint arXiv:2310.02264, 2023.   
X. Zhou, R. Girdhar, A. Joulin, P. Krähenbühl, and I. Misra, “Detecting twenty-thousand classes using image-level supervision,” in ECCV, 2022.   
Y. Zhou, C. Cui, J. Yoon, L. Zhang, Z. Deng, C. Finn, M. Bansal, and H. Yao, “Analyzing and mitigating object hallucination in large vision-language models,” arXiv preprint arXiv:2310.00754, 2023.   
D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” 2023.   
J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2223–2232.   
S. Zhu, A. Kimmel, K. E. Bekris, and A. Boularias, “Fast model identification via physics engines for data-efficient policy search,” arXiv preprint arXiv:1710.08893, 2017.   
X. Zhu, J. Wang, L. Zhang, Y. Zhang, R. Gan, J. Zhang, and Y. Yang, “Solving math word problem via cooperative reasoning induced language models,” arXiv preprint arXiv:2210.16257, 2022.   
Y. Zhu, R. Mottaghi, E. Kolve, J. J. Lim, A. Gupta, L. Fei-Fei, and A. Farhadi, “Target-driven visual navigation in indoor scenes using deep reinforcement learning,” in Robotics and Automation (ICRA), 2017 IEEE International Conference on. IEEE, 2017, pp. 3357–3364.

# Appendix for

# Agent AI

# A GPT-4V Agent Prompt Details

Unless specified, we use the default system prompt for GPT-4V. We show a detailed description of the process to prompt GPT-4V for Minecraft and how we generate GPT-4V’s responses in Fig. 31.

# B GPT-4V for Bleeding Edge

Bleeding Edge is a third person team-based combat game where players attempt to capture objective points or collect more resources than the enemy team. We show an example input and output when prompting GPT-4V for the game Bleeding Edge in Fig. 32. Compared to Minecraft, we qualitatively find that GPT-4V has a less thorough understanding of the visual content and game rules. This is likely due to the (1) the amount of minecraft data present in GPT-4V’s training data and (2) the visual complexity of Bleeding Edge compared to Minecraft.

# C GPT-4V for Microsoft Fight Simulator

As shown in Fig. 33, a GPT-4V based agent can provide the high-level action description for the player in Microsoft Flight Simulator. It describes how the player is in the process of flying an aircraft, shown by the cockpit perspective and external views of the plane, managing various flight controls and instruments to maintain appropriate airspeed and altitude while navigating through the virtual airspace.

# D GPT-4V for Assassin’s Creed Odyssey

As shown in Fig. 34, the GPT-4V agent provides the high-level action description of the character in the images being engaged in nighttime combat during a historical battle, wielding a spear and utilizing special abilities, such as the Bull Rush, to fight against multiple enemies marked with level indicators, while managing their health and ability meters in a third-person action role-playing setting. GPT-4V also describes that the player is engaged in a large-scale melee combat scenario, controlling a character with red accents on their armor, and is prompted to use a "Bull Rush Ability" as indicated by the on-screen instructions. The environment is dark and fiery, suggesting a battlefield, and the player’s character is facing multiple enemies marked with level indicators and health bars overhead. The action is part of a tutorial or combat sequence where the player learns and executes combat abilities.

# E GPT-4V for GEARS of WAR 4

As shown in Fig. 35, Based on the sequence of 10 frames of video provided, the GPT-4V agent can provide a general description of the player’s actions in the video game:

“The player is navigating a character through an industrial environment at night, taking strategic cover behind various objects like crates and walls. The character is armed and engages in combat with enemy entities. The player uses firearms and possibly melee attacks, as indicated by close-quarters combat. They appear to manage their ammunition and health, indicated by visual cues such as a flashing red indicator on the screen, which often signifies damage to the character.

Additionally, the player seems to be moving tactically, shifting between different cover positions to advance or defend against the enemy. There is evidence of a protective shield being activated at some point, which suggests the player is responding to incoming fire by using defensive abilities. The character also seems to be reloading their weapon, suggesting an element of resource management during combat. Blood and impact effects indicate successful hits on the enemies.

The player’s character collaborates with allies, as other characters are seen in the vicinity, engaging in similar combat actions. The environment has interactive elements, such as exploding barrels, which the player might use to their advantage.

![](images/1d119c97a9f6d5387f9b2ab4713bcf04182b62d00f76a4ce3681db7d9fbade7e.jpg)  
Figure 31: We provide code snippets to show the complete prompting process for calling GPT-4V for Minecraft videos. First, we encode the frames of the video and then call the GPT-4V endpoint. The model’s response is shown on the bottom of the screen.

# GPT-4-Vision

# Prompt:

These are frames of a video of a Bleeding Edge player ordered from left to right and top to bottom as a grid. Give a simple, but precise description of what the player is doing in 1 sentence. Be specific about important items, entities, and actions. In your description do not mention specific frame numbers or the name of the game.

# Video input:

![](images/22762d17d2ec937154c23fb3090b146d72b0b73ef3e75c7aceecd89dea55dad4.jpg)  
Figure 32: GPT-4V is able to produce meaningful content for games like Bleeding Edge that have 3rd person viewpoints and visually complex scenes. In order to input a large number of frames (48) to GPT-4V, we input the frames as a grid with frame numbers overlaid on each frame (as shown above).

# Output:

The player begins by running around the map, passing through different checkpoints and interacting with several capture points, then fights against an enemy player, and finally captures an objective while being attacked by another enemy.

# GPT-4-Vision

# Prompt:

These are frames of a video of a Flight Simulator gaming player ordered from left to right and top to bottom. Give a simple, but precise description of what the player is doing in 1 sentence.Be specific about important items, entities, and actions. In your description do not mention specific frame numbers or the name of the game.

# Video input:

![](images/02dfbcb30a24164db94c8c9296806d6a24e218cb46f86eccdf23b8d675b195d7.jpg)

# Output :

In the images from the flight simulator game， the player is piloting an aircraft, taking off from an airstrip over water, adjusting the cockpit controls，and navigating through the game's interface to manage the flight， while dealing with an alert about low airspeed and the need to increase it to avoid a stall.

Figure 33: GPT-4V is able to produce meaningful content description for Microsoft Fight Simulator games. The GPT-4V agent can provide a high-level action description as the player in the flight simulator is in the process of flying an aircraft, shown by the cockpit perspective and external views of the plane, managing various flight controls and instruments to maintain appropriate airspeed and altitude while navigating through the virtual airspace. In order to input a large video, we select a number of key frames (6) to send to GPT-4V. We input each key frame individually and do not use a grid (as shown above).

# GPT-4-Vision

# Prompt:

These are frames of a video of an ASSASSIN's CREED ODYSSEY gaming player ordered from left to right and top to bottom. Give a simple, but precise description of what the player is doing. Be specific about important items, entities, and actions. In your description do not mention specific frame numbers or the name of the game.

# Video input:

![](images/fb26144b27d19eece6d415158b41455dc894890e6f750c2b7c09cdaf1325bd0f.jpg)

# Outp

The player is engaged in a large-scale melee combat scenario， controlling a character with red accents on their armor， and is prompted to use a "Bull Rush Ability" as indicated by the onscreen instructions. The environment is dark and fiery， suggesting a battlefield, and the player's character is facing multiple enemies marked with level indicators and health bars overhead.The action is part of a tutorial or combat sequence where the player learns and executes combat abilities.

Figure 34: GPT-4V is able to produce meaningful content description for the game, Assassin’s Creed Odyssey. The GPT-4V-based agent can also provide a high-level action description of "the character in the images [being] engaged in nighttime combat during a historical battle, wielding a spear and utilizing special abilities, such as the Bull Rush, to fight against multiple enemies marked with level indicators, while managing their health and ability meters in a third-person action role-playing setting." In order to input a large video, we select a number of key frames (8) to send to GPT-4V. We input each key frame individually and do not use a grid (as shown above).

# GPT-4-Vision

# Prompt:

These are frames of a video of the GEARS of WAR 4 gaming player ordered from left to right and top to bottom. Give asimple,but precise descriptionof what the player isdoing.Be specific about important items,entities,and actions.In your description do not mention specific frame numbers or the name of the game.

Video input:

![](images/d2f8eadf9c12edd5a143bc1d1f6e951a95149c84d8505758173129b52082fb36.jpg)

# Output:

Based on the sequence of images provided，here is a general description of the player's actions in a video game:

The player is navigating a character through an industrial environment at night， taking strategic cover behind various objects like crates and walls. The character is armed and engages in combat with enemy entities. The player uses firearms and possibly melee attacks，as indicated by close-quarters combat. They appear to manage their ammunition and health, indicated by visual cues such as a flashing red indicator on the screen，which often signifies damage to the character.

Additionally， the player seems to be moving tactically， shifting between different cover positions to advance or defend against the enemy. There is evidence of a protective shield being activated at some point， which suggests the player is responding to incoming fire by using defensive abilities. The character also seems to be reloading their weapon， suggesting an element of resource management during combat.Blood and impact effects indicate successful hits on the enemies.

The player's character collaborates with allies，as other characters are seen in the vicinity, engaging in similar combat actions. The environment has interactive elements，such as exploding barrels，which the player might use to their advantage.

The scenes depict fast-paced action with a focus on strategic movement， shooting，and situational awareness to overcome adversaries and possibly achieve a specific objective within the game.

Figure 35: GPT-4V is able to produce meaningful content description for GEARS of WAR 4 games. The GPT-4V agent provides the high-level action description of the player being engaged in a combat scenario within a visually rich, futuristic environment, reminiscent of industrial or military complexes. The character, armored and equipped with a firearm, is navigating through this area, using cover and aiming systems to engage with enemies. Actions captured include the player character aiming from behind cover, firing upon enemies, taking damage as indicated by visual cues on the screen, and performing close-quarters combat. The environment is marked by night-time lighting with artificial sources highlighting areas of interest or action. There are also moments where the player’s character is seen in a transparent, outlined form, which could denote a special ability or a temporary state like invincibility or stealth. In order to input a large video, we select a number of key frames (10) to send to GPT-4V. We input each key frame individually and do not use a grid (as shown above).

The scenes depict fast-paced action with a focus on strategic movement, shooting, and situational awareness to overcome adversaries and possibly achieve a specific objective within the game."

# F GPT-4V for Starfield

As showed in Fig. 36, based on the sequence of 10 frames of video provided for Starfield, the GPT-4V agent provides a general description of the player’s actions in a video game. According to GPT-4V: "The images depict a series of events within the game Starfield, showcasing the player’s interactions within a mining environment. Here’s a description of the sequence:

1) The player stands in a dimly lit underground cavern with stalactites, accompanied by another character in similar space gear. They face a highlighted waypoint, indicating an objective or location of interest.   
2) Moving through an industrial interior, likely a mining station, the player approaches another character. The HUD (Heads-Up Display) shows a dialogue line from "Supervisor Lin," which suggests NPC (non-player character) interaction and possibly a mission briefing or update.   
3) The player seems to be following the NPC, Supervisor Lin, through the cavern, hinting at a guided segment of gameplay where the NPC leads the player to a specific location or task.   
4) A dialogue from Supervisor Lin suggests a decision point or mission update, possibly directing the player to a new objective related to a "big vein" of resources.   
5) The player encounters Supervisor Lin again, who provides instruction or information about metal deposits, implying the start of a mining task or exploration of a new area.   
6) In this frame, the player appears to be observing or interacting with a piece of machinery, with another character, Isabelle, indicating an action or dialogue related to equipment operation or technical work.   
7) The player looks down at a rough terrain with two glowing objects, which could represent resources to be collected or points of interest for the mission.   
8) The player examines a specific resource, indicated by a HUD marker as "Beryllium Deposit," suggesting a resource gathering or mining activity.   
9) The player traverses through the caverns, potentially searching for additional resources or navigating to the next objective, as indicated by another character in space gear ahead.   
10) The player is positioned in a darker area of the cavern, with two glowing markers indicating a path or objective points to be reached.

Throughout these images, the player seems to be engaged in a mission involving exploration, NPC interaction, resource collection, and possibly machinery operation within a mining environment. The HUD elements like waypoints, dialogue text, and mission updates provide context to the player’s actions and the game’s objectives."

# GPT-4-Vision

# Prompt:

Theseare framesofavideoofthe Starfield Consolegame playerorderedfromlefttorightandtoptobotom.Giveasimple,but precisedescriptionofwhattheplayerisdoing.Bespecificabout importantitems,ntities,andactions.Inyourdescriptiodoot mention specific frame numbers or the name of the game.

Video input:

![](images/61135ef5cc1becd3c8f6b3266750a2aa03d632c64ef47fa95745e740d6016bd0.jpg)

# Output:

The images depict a series of events within the game Starfield， showcasing the player's interactions within a mining environment. Here's a description of the sequence:

1.The player stands in a dimly lit underground cavern with stalactites，accompanied by another character in similar space gear. They face a highlighted waypoint, indicating an objective or location of interest.   
2.Moving through an industrial interior，likely a mining station, the player approaches another character.The HUD (Heads-Up Display） shows a dialogue line from "Supervisor Lin," which suggests NPC (non-player character) interaction and possibly a mission briefing or update.   
3.The player seems to be following the NPC， Supervisor Lin， through the cavern, hinting at a guided segment of gameplay where the NPC leads the player to a specific location or task.   
4.A dialogue from Supervisor Lin suggests a decision point or mission update, possibly directing the player to a new objective related to a "big vein" of resources.   
5.The player encounters Supervisor Lin again，who provides instruction or information about metal deposits, implying the start of a mining task or exploration of a new area.   
6.In this frame，the player appears to be observing or interacting with a piece of machinery，with another character，Isabelle,indicating an action or dialogue related to equipment operation or technical work.   
7.The player looks down at a rough terrain with two glowing objects，which could represent resources to be collected or points of interest for the mission.   
8.The player examines a specific resource， indicated by a HUD marker as "Beryllium Deposit," suggesting a resource gathering or mining activity.   
9.The player traverses through the caverns， potentially searching for additional resources or navigating to the next objective，as indicated by another character in space gear ahead.   
10.The player is positioned in a darker area of the cavern， with two glowing markers indicating a path or objective points to be reached. Throughout these images， the player seems to be engaged in a mission involving exploration， NPC interaction， resource collection, and possibly machinery operation within a mining environment.The HUD elements like waypoints，dialogue text，and mission updates provide context to the player's actions and the game's objectives.

Figure 36: GPT-4V is able to produce meaningful content description for the X-Box console game Starfield. The GPT-4V agent provides a high-level action description for the space-themed exploration game. In order to input a large video, we select a number of key frames (10) to send to GPT-4V. We input each key frame individually and do not use a grid (as shown above).

# Author Biographies

Zane Durante PhD Student in Computer Science at Stanford University

Email: durante@stanford.edu

Web: https://zanedurante.github.io/

Zane Durante is a PhD student at Stanford Vision Lab advised by Dr. Fei-Fei Li. H His research interests include video understanding, connecting computer vision with natural language, and AI applications in hospital care. He is leading an ongoing collaboration between Stanford Medical School and Stanford Vision Lab to develop a dataset of natural language descriptions of video clips of hospital scenes. His PhD work is supported by NSF’s Graduate Research Fellowship.

Qiuyuan Huang Principal Researcher at Microsoft Research, Redmond, WA, USA.

Email: qihua $@$ microsoft.com

Web: https://www.microsoft.com/en-us/research/people/qihua/

Qiuyuan Huang is a principal researcher in the deep learning group at Microsoft Research (MSR), Redmond, WA. Her current research interests are mainly in the deep learning, multi-modality, and natural language processing areas, specifically on Agent AI for Gaming, Robotics and Healthcare; Knowledge-reasoning Intelligence for Interactive AI; Neuro-symbolic Computation for Inference Reasoning; and Large Foundation models for NLP and Multi-modality.

Naoki Wake Researcher at Microsoft, Redmond, WA, USA.

Email: naoki.wake@microsoft.com

Web: https://www.microsoft.com/en-us/research/people/nawake/

Naoki Wake is a researcher in the Applied Robotics Research group at Microsoft. His current research involves the development of multimodal perception systems for robots and co-speech gesturing systems. His past research has spanned auditory neuroscience, neuro-rehabilitation, and speech processing. Naoki received his B.S. degree in Engineering in 2014, and his Ph.D. in Information Science and Technology in 2019, both from the University of Tokyo.

Ran Gong PhD Student in Computer Science at University of California, Los Angeles.

Email: nikepupu@ucla.edu

Web: https://nikepupu.github.io

Ran Gong, is a PhD student at the UCLA VCLA Lab. His research lies in the intersection of Robotics, Computer Vision, Computer Graphics, and Machine Learning. His research focuses on embodied simulation and interaction with a goal of creating intelligent behaviors that can solve diverse tasks in diverse environments as well as well as the capability of collaborating with humans. He received his B.S. degree in Computer Science and Engineering at the University of California, Los Angeles.

Jae Sung Park PhD Student at University of Washington

Email: jspark96@cs.washington.edu

Web: https://homes.cs.washington.edu/~jspark96/

Jae Sung is a PhD student advised by Yejin Choi and Ali Farhadi. His research focuses on developing models with multimodal commonsense reasoning. He is interested in equipping models with grounding linguistic concepts to visual modalities, and having them understand multimedia content in a way that humans process the visual information. Jae Sung received his B.S. degree in Computer Science at University of California, Berkeley.

Bidipta Sarkar Undergraduate Student at Stanford University

Email: bidiptas@stanford.edu

Web: https://bsarkar321.github.io/

Bidipta Sarkar is a senior undergraduate student at Stanford University and a member of Stanford’s ILIAD lab. His research focuses on creating AI agents that can interact with their environment and safely work alongside humans and other autonomous agents.

Rohan Taori PhD Student in Computer Science at Stanford University

Email: rtaori@cs.stanford.edu

Web: https://www.rohantaori.com/

Rohan Taori is a PhD student at the Stanford AI Lab. His research focuses on studying the foundations of machine learning in the context of real-world systems. Most recently, Rohan has pushed forward the frontier of open-source large language models, finetuning them to be helpful, general-purpose assistants. Rohan is also very interested in augmenting language models with multi-modality capability to allow them to reason over images and videos. Rohan received his Bachelor’s degree in Computer Science at UC Berkeley.

Yusuke Noda Principal Software Engineer. Microsoft Gaming, Redmond, WA.

Email: yusuke.noda@microsoft.com

Web: https://www.linkedin.com/in/yusuke-noda-908797/

Yusuke Noda is a principal software engineer at Microsoft Gaming Platform. He has led the development of gaming infrastructure for Xbox One and Xbox Cloud Gaming and has over 16 years of experience developing efficient infrastructure for cloud and gaming technologies.

Demetri Terzopoulos UCLA Distinguished Professor, Academy Award winning computer scientist and entrepreneur.

Email: dt@cs.ucla.edu

Web: https://web.cs.ucla.edu/~dt/

Dr. Demetri Terzopoulos is a Distinguished Professor and Chancellor’s Professor of Computer Science at the University of California, Los Angeles, where he directs the UCLA Computer Graphics & Vision Laboratory. He is also Co-Founder and Chief Scientist of VoxelCloud, Inc., a multinational healthcare AI company. He is or was a Guggenheim Fellow, a Fellow of the Royal Society (FRS) of London and the Royal Society of Canada (FRSC), a Fellow of the Association for Computing Machinery (ACM), a Life Fellow of the Institute of Electrical and Electronics Engineers (IEEE), a Fellow of the Asia-Pacific Artificial Intelligence Association (AAIA), a Distinguished Fellow of the International Engineering and Technology Institute (IETI), a Member of the European Academy of Sciences (EAS) and the New York Academy of Sciences (NYAS), and a Life Member of Sigma Xi.

Yejin Choi Wisnner-Slivaka Chair and Brett Helsel Professor at University of Washington, Senior Research Manager at Allen Institute for Artificial Intelligence, and MacArthur Fellow.

Email: yejin@cs.washington.edu

Web: https://homes.cs.washington.edu/~yejin/

Yejin is a Wisnner-Slivaka Chair and Brett Helsel Professor at University of Washington and Senior Research Manager at Allen Institute of Artifical Intelligence. She has won the Anita Borg Early Career Award in 2018. She was the recipient of MacArthur Fellow foundation fellowship in 2020. She has received outstanding paper award in AAAI 2020, Neurips 2021, ICML 2022, and ACL 2023, and the best paper award in NAACL 2022 and ACL 2023. She is one of the main organizers of COLM 2024, an academic venue focused on the study of language modeling. Her main research interests are commonsense reasoning in the fields of Natural Language Processing, Machine Learning, Artificial Intelligence, with broader interests in Computer Vision and Digital Humanities.

Fei-Fei Li Professor of Computer Science at Stanford University

Email: feifeili $@$ stanford.edu

Web: https://profiles.stanford.edu/fei-fei-li/

Fei-Fei is the inaugural Sequoia Professor in the Computer Science Department at Stanford University, and Co-Director of Stanford’s Human-Centered AI Institute. She served as the Director of Stanford’s AI Lab from 2013 to 2018. And during her sabbatical from Stanford from January 2017 to September 2018, Dr. Li was Vice President at Google and served as Chief Scientist of AI/ML at Google Cloud. Since then she has served as a Board member or advisor in various public or private companies.

Katsushi Ikeuchi Senior Principal Research Manager at Microsoft and an IEEE Life Fellow. Microsoft Research, Redmond, WA. IEEE Life Fellow.

Email: katsuike@microsoft.com

Web: https://www.microsoft.com/en-us/research/people/katsuike/

Dr. Katsushi Ikeuchi received the BE degree in Mechanical Engineering from Kyoto University in 1973 and the PhD degree in Information Engineering from the University of Tokyo in 1978. After working at the Artificial Intelligence Laboratory, Massachusetts Institute of Technology for three years, Electrotechnical Laboratory, Ministry of International Trade and Industry, Japanese government for five years, and Robotics Institute, Carnegie Mellon University for 10 years,

Institute of Industrial Science, the University of Tokyo for 19 years, he joined Microsoft as a Principal Researcher in 2015. During this tenure of CMU and UTokyo, he supervised more than 50 PhD students.

His research interest spans computer vision, robotics, and computer graphics. In these research fields, he has received several best paper awards, including the David Marr Prize in computational vision and IEEE Robotics and Automation Society K. S. Fu memorial best transaction paper.

His community service includes: general chair of IROS95, ITSC99, IV01, ICCV05, ACCV07, ICCV17; program chair of CVPR96, ICCV03, ICRA09, ICPR12, ICCV15; EIC of IJCV (2000-2017), EIC of IJ ITS (2012-2014), associate editor of IEEE Trans. RA, IEEE Trans. PAMI; and a distinguished lecturer of IEEE Signal Processing Society in 2000-2002, IEEE Robotics and Automation Society in 2004-2006 and IEEE Computer Society in 2008-2010.

Through these research and society service, he was awarded a (life) fellow from IEEE, IEICE, IPSJ and RSJ. He received the Distinguished Researcher Award from IEEE-PAMI, the Medal of Honor with Purple Ribbon from Japanese Emperor, the Okawa prize from Okawa foundation as well as a couple of research achievement awards from Japanese professional societies.

Hoi Vo Technical Fellow. Microsoft Gaming and X-Box Emerging Technologies, Redmond, WA.

Email: hoiv@microsoft.com

Web: https://www.linkedin.com/in/hoi-vo-193420/

Hoi Vo is a Technical Fellow from MS Gaming Division focusing on bridging AGI with new Gaming IP. He has played a pivotal role in establishing the Xbox Cloud Gaming service has led cloud and gaming efforts across teams in Microsoft, including Microsoft’s Windows Azure platform. His focus is on the optimization of AI models to run efficiently at the edge while leveraging the cloud for scenarios that exceed the hardware capabilities, including fine-tuning models to be more engaging with various playing experiences.

Jianfeng Gao Distinguished Scientist and Vice President at Microsoft and an IEEE Fellow. Microsoft Research, Redmond, WA. IEEE Fellow.

Email: jfgao@microsoft.com

Web: https://www.microsoft.com/en-us/research/people/jfgao/

Dr. Jianfeng Gao is Distinguished Scientist and Vice President at Microsoft Research, IEEE Fellow, ACM Distinguished Member, and the current head of the Deep Learning Group at Microsoft Research. From 2014 to 2017, he was Partner Research Manager in Business AI at Microsoft AI Research and at Deep Learning Technology Center (DLTC) at Microsoft Research, Redmond. He lead the development of AI solutions to Predictive Sales and Marketing. He also works on deep learning for text and image processing (see ACL/SIGIR 2018 Tutorial, Deep Learning 2017 Tutorial and IJCAI 2016 Tutorial or MS internal site) and lead the development of AI systems for dialogue, machine reading comprehension (MRC), and question answering (QA). From 2022, he leads the research of self-improving AI where LLMs (e.g., ChatGPT/GPT4) are augmented and adapted for the development of commercial AI systems.

# Acknowledgements

We are especially grateful to Peter Lee, Doug Burger, Desney Tan, Johannes Gehrke, Ryen White, Ece Kamar, Subhojit Som, and Kareem Choudhry for their advices, enormous support, and encouragement. We thank to Haiyan Zhang, Spencer Perreault, Dave Bignell, Katja Hofmann, Sam Devlin, Shanzheng Tan, Raluca Georgescu, Bill Dolan, Nebojsa Jojic, Sudha Rao, Adrian Brown, Andrzej Banburski-Fahey, Jianwei Yang for the early insightful discussions and helps of Gaming. We appreciate Kiran Muthabatulla, Antonio Criminisi, Tom Cashman, Nguyen Bach, Jennifer Marsman, Jaron Lanier from Mesh team, OCTO of Microsoft, and Microsoft office team for their mix-reality work, dataset work, and their generous helps and feedback for the project. We special thanks to Paul Bennett, Corby Rosset, Michel Galley, Chenglong Wang, Baolin Peng, Hao Chen, Silviu Cucerzan, Ahmed Awadallah, Saleema Amershi for their suggestion and comments for the NLP part. The authors gratefully acknowledge Paul Smolensky, Yonatan Bisk, Kezhen Chen, Borui Wang, Liangke Gui, Dingmin Wang, Xin (Eric) Wang, Zhe Gan, Xiaojian Ma, Zilong Zheng, Song-chun Zhu, Dragomir R. Radev, Daniel McDuff, Harry Shum for the related previous works, comments, suggestions, painstaking multiple reviews of this paper, and their pointers to the literature. Finally, we would like to really appreciate Microsoft Holulens team, Microsoft X-box team, and Meta Quest team for their generous provision of the equipment; MSR Central Engineering (CE) team, Microsoft 343 team for the data collection and sharing; Microsoft AOAI and GCR team for their Azure-OpenAI endpoint supporting.

We would like to thank our colleagues from Stanford’s Partnership in AI-assisted Care, who helped inform the medical applications explored in this work. In particular, we would like to thank Ehsan Adeli, Paul Tang, Amit Kaushal, Roger Bohn, Kevin Schulman, and Arnold Milstein for their clinical expertise and guidance.

This research was supported by Microsoft Research project Fair 2023, Microsoft HackBox 2023, OCTO team.
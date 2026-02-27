# UFO2: The Desktop AgentOS

<table><tr><td>Chaoyun Zhang* 
Microsoft</td><td>He Huang 
Microsoft</td><td>Chiming Ni† 
ZJU-UIUC Institute</td><td>Jian Mu† 
Nanjing University</td><td>Si Qin 
Microsoft</td></tr><tr><td>Shilin He 
Microsoft</td><td>Lu Wang 
Microsoft</td><td>Fangkai Yang 
Microsoft</td><td>Pu Zhao 
Microsoft</td><td>Bo Qiao 
Microsoft</td></tr><tr><td>Chao Du 
Microsoft</td><td>Liqun Li 
Microsoft</td><td>Yu Kang 
Microsoft</td><td>Zhao Jiang 
Microsoft</td><td>Suzhen Zheng 
Microsoft</td></tr><tr><td>Rujia Wang 
Microsoft</td><td>Jiaxu Qian† 
Peking University</td><td>Minghua Ma 
Microsoft</td><td>Jian-Guang Lou 
Microsoft</td><td>Qingwei Lin 
Microsoft</td></tr><tr><td></td><td>Saravan 
Rajmohan 
Microsoft</td><td>Dongmei Zhang 
Microsoft</td><td></td><td></td></tr></table>

# Abstract

Recent Computer-Using Agents (CUAs), powered by multimodal large language models (LLMs), offer a promising direction for automating complex desktop workflows through natural language. However, most existing CUAs remain conceptual prototypes, hindered by shallow OS integration, fragile screenshot-based interaction, and disruptive execution.

We present $\mathbf { U F O ^ { 2 } }$ , a multiagent AgentOS for Windows desktops that elevates CUAs into practical, system-level automation. UFO2 features a centralized HostAgent for task decomposition and coordination, alongside a collection of application-specialized AppAgents equipped with native APIs, domain-specific knowledge, and a unified GUI–API action layer. This architecture enables robust task execution while preserving modularity and extensibility. A hybrid control detection pipeline fuses Windows UI Automation (UIA) with vision-based parsing to support diverse interface styles. Runtime efficiency is further enhanced through speculative multi-action planning, reducing per-step LLM overhead. Finally, a Picture-in-Picture (PiP) interface enables automation within an isolated virtual desktop, allowing agents and users to operate concurrently without interference.

We evaluate $\mathrm { U F O ^ { 2 } }$ across over 20 real-world Windows applications, demonstrating substantial improvements in robustness and execution accuracy over prior CUAs. Our results show that deep OS integration unlocks a scalable path toward reliable, user-aligned desktop automation.

The source code of $\mathrm { U F O ^ { 2 } }$ is publicly available at https: //github.com/microsoft/UFO/, with comprehensive documentation provided at https://microsoft.github.io/UFO/.

Keywords: Computer Using Agent, Large Language Model, Desktop Automation, Windows System

![](images/1fd16c4d7ebc3e96832ffe88a187158ef7e5b1aebee67ba8cd3dcfe43d702b94.jpg)

![](images/6d52f26528e5672880c410af6e816922e033eba3b1b7000861d78bd8fcffd81c.jpg)  
Figure 1. A comparison of (a) existing CUAs and (b) desktop AgentOS UFO2.

# ACM Reference Format:

Chaoyun Zhang et al.. 2025. UFO2: The Desktop AgentOS. In . ACM, New York, NY, USA, 24 pages. https://doi.org/10.1145/nnnnnnn. nnnnnnn

# 1 Introduction

Automation of desktop applications has long been central to improving workforce productivity. Commercial Robotic Process Automation (RPA) platforms such as UiPath [1], Automation Anywhere [2], and Microsoft Power Automate [3] exemplify this trend, using predefined scripts to replicate repetitive user interactions through the graphical user interface (GUI) [4, 5]. However, these script-based approaches often prove fragile in dynamic, continuously evolving environments [6]. Minor interface changes can break the underlying automation scripts, requiring manual updates and extensive maintenance effort. As software ecosystems grow increasingly complex and heterogeneous, the brittleness of

script-based automation severely limits scalability, adaptability, and practicality.

Recently, Computer-Using Agents (CUAs) [7] have emerged as a promising alternative. These systems leverage advanced multimodal large language models (LLMs) [8, 9] to interpret diverse user instructions, perceive GUI interfaces, and generate adaptive actions (e.g., mouse clicks, keyboard inputs) without fixed scripting [7]. Early prototypes such as UFO [10], Anthropic Claude [11], and OpenAI Operator [12] demonstrate that such LLM-driven agents can robustly automate tasks too complex or ambiguous for conventional RPA pipelines. Yet, despite these advances, current CUA implementations remain largely conceptual: they mainly optimize visual grounding or language reasoning [12–16], but give little attention to system-level integration with desktop operating systems (OSs) and application internals (Figure 1 (a)).

Reliance on raw GUI screenshots and simulated input events has several drawbacks. First, visual-only inputs can be noisy and redundant, increasing cognitive overhead for LLMs and reducing execution efficiency [17]. Second, existing CUAs rarely leverage the OS’s native accessibility interfaces, application-level APIs, or detailed process states—a missed opportunity to significantly enhance decision accuracy, reduce latency, and enable more reliable execution. Finally, simulating mouse and keyboard events on the primary user desktop locks users out during automation, imposing a poor user experience (UX). These limitations must be resolved before CUAs can mature from intriguing prototypes into robust, scalable solutions for real-world desktop automation, and motivates our central research question:

How can we build a robust, deeply integrated system for desktop automation that flexibly adapts to evolving interfaces, reliably orchestrates diverse applications, and minimizes disruption to user workflows?

In response, we present $\mathbf { U F O ^ { 2 } }$ , a new AgentOS for Windows that reimagines desktop automation as a first-class operating system abstraction. Unlike prior CUAs that treat automation as a layer atop screenshots and simulated input events, $\mathrm { U F O ^ { 2 } }$ is architected as a deeply integrated, multiagent execution environment—embedding OS capabilities, application-specific introspection, and domain-aware planning into the core automation loop, as illustrated in Figure 1(b).

At its foundation, $\mathrm { U F O ^ { 2 } }$ provides a modular, system-level substrate for natural language-driven automation. A centralized coordinator, the HostAgent, interprets user instructions, decomposes them into semantically meaningful subtasks, and dynamically dispatches execution to specialized AppAgents —expert modules tailored for specific Windows applications. Each AppAgent is equipped with an extensible toolbox of application-specific APIs, a hybrid GUI–API

action interface, and integrated knowledge about the application’s capabilities and semantics. This architecture enables robust orchestration across multiple concurrent applications, supporting workflows that span Excel, Outlook, Edge, and beyond.

To enable reliable execution across the full spectrum of application UIs, $\mathrm { U F O ^ { 2 } }$ introduces a hybrid control detection pipeline, combining Windows UI Automation (UIA) APIs with advanced visual grounding models [18]. This allows agents to introspect and act on both standard and custom UI components, bridging the gap between structured accessibility trees and pixel-level perception. Moreover, UFO2 continuously incorporates external documentation, patch notes, and past execution traces into a unified vectorized memory layer, enabling each AppAgent to incrementally refine its behavior without retraining.

At the interaction layer, UFO2 exposes a unified GUI–API execution model, where agents seamlessly combine traditional GUI actions (e.g., clicks, keystrokes) with native Windows or application-specific APIs. This hybrid approach improves execution efficiency, reduces brittleness to UI layout changes, and enables more expressive, higher-level operations. To further minimize the latency associated with LLM-based action planning, $\mathrm { U F O ^ { 2 } }$ incorporates a speculative multi-action execution engine that proactively infers and validates action sequences using lightweight control-state checks at a single inference step—substantially reducing inference overhead without compromising correctness.

Finally, to ensure a practical and non-intrusive user experience, $\mathrm { U F O ^ { 2 } }$ introduces a novel Picture-in-Picture (PiP) interface: a secure, nested desktop environment where agents can execute independently of the user’s main session. Built atop Windows’ native remote desktop loopback infrastructure, PiP enables seamless, side-by-side user–agent multitasking without disruption on the user’s main desktop, addressing one of the most persistent UX limitations of existing CUAs.

Together, these design principles position $\mathrm { U F O ^ { 2 } }$ not just as a smarter agent, but as a new OS-level abstraction for automation—transforming desktop workflows into programmable, composable, and robust entities. In summary, this paper makes the following contributions:

• Deep OS Integration: We design and implement UFO2, a multiagent AgentOS that deeply embeds within the Windows OS, orchestrating desktop applications through introspection, API access, and fine-grained execution control.   
• Unified GUI–API Action Layer: We propose a hybrid action interface that bridges traditional GUI interactions with application-native API calls, enabling flexible, efficient, and robust automation.   
• Hybrid Control Detection: We introduce a fusion pipeline combining UIA metadata with vision-based

detection to achieve reliable control grounding even in non-standard interfaces.

• Continuous Knowledge Integration: We build a retrieval-augmented memory that integrates documentation and historical execution logs, allowing agents to improve autonomously over time without retraining.   
• Speculative Multi-Action Execution: We reduce LLM invocation overhead by predicting and validating action sequences ahead of time using UI state signals.   
• Non-Disruptive UX: We develop a nested virtual desktop environment that allows automation to proceed in parallel with user activity, avoiding interference and improving adoptability.   
• Comprehensive Evaluation: We evaluate $\mathrm { U F O ^ { 2 } }$ across $2 0 +$ real-world Windows applications, showing consistent improvements in success rate, execution efficiency, and usability over state-of-the-art CUAs like Operator.

Overall, $\mathrm { U F O ^ { 2 } }$ advances the vision of OS-native automation by shifting the paradigm from GUI scripting to structured, programmable application control. Even when paired with general-purpose models like GPT-4o, $\mathrm { U F O ^ { 2 } }$ outperforms dedicated CUAs by over $1 0 \%$ , highlighting the transformative impact of system-level integration and architectural design.

# 2 Background

# 2.1 The Fragility of Traditional Desktop Automation

For decades, desktop automation has relied on brittle techniques to replicate human interactions with GUI-based applications. Commercial RPA (Robotic Process Automation) tools—such as UiPath [1], Automation Anywhere [2], and Microsoft Power Automate [3]—operate by recording and replaying mouse movements, keystrokes, or rule-based scripts. These systems rely heavily on surface-level GUI cues (e.g., pixel regions, window titles), offering little introspection into application state.

While widely deployed in enterprise settings, traditional RPA systems exhibit poor robustness and scalability [19]. Even minor UI updates—such as reordering buttons or relabeling menus—can silently break automation scripts. Maintaining correctness requires frequent human intervention. Furthermore, these tools lack semantic understanding of application workflows and cannot reason about or adapt to novel tasks. As a result, RPA tools remain constrained to narrow, repetitive workflows in stable environments, far from general-purpose automation.

# 2.2 Rise of Computer-Using Agents

Recent advances in large language models (LLMs) and multimodal perception have enabled a new class of automation systems, referred to as Computer-Using Agents (CUAs) [7– 9]. CUAs aim to generalize across applications and tasks by leveraging LLMs to interpret user instructions, perceive GUI layouts, and synthesize actions such as clicks and keystrokes.

Early CUAs like UFO [10] demonstrated that multimodal models (e.g., GPT-4V [20]) could map natural language requests to sequences of GUI actions with no hand-crafted scripts. More recent industry prototypes, including Claude-3.5 (Computer Use) [11] and OpenAI Operator [12], have pushed the envelope further, performing realistic desktop workflows across multiple applications.

These CUAs represent a promising evolution from static RPA scripts to adaptive, general-purpose automation. However, despite their sophistication, current CUAs largely remain research prototypes, constrained by architectural and systems-level limitations that impede real-world deployment.

# 2.3 Systems Challenges in CUAs

Current CUAs fall short in three fundamental ways, which we argue stem from missing operating system abstractions:

(1) Lack of OS-Level Integration. Most CUAs interact with the system through screenshots and low-level input emulation (mouse and keyboard events). They ignore rich system interfaces such as accessibility APIs, application process state, and native inter-process communication mechanisms (e.g., shell commands, COM interfaces [21]). This superficial interaction model limits reliability and efficiency—every action must be inferred from pixels rather than structured state.   
(2) Absence of Application Introspection. CUAs typically operate as generalists with limited awareness of applicationspecific capabilities. They treat all interfaces uniformly, lacking the ability to leverage built-in APIs or vendor documentation. As a result, they cannot reason over high-level concepts unless such flows are explicitly embedded in the model. This rigidity limits their generalization and makes maintenance expensive.   
(3) Disruptive and Unsafe Execution Model. Most CUAs drive automation directly on the user’s desktop session, hijacking the real mouse and keyboard. This design prevents users from interacting with their system during execution, introduces interference risk, and violates isolation principles fundamental to safe system design. Long-running tasks—especially those involving multiple LLM queries—can monopolize the session for minutes at a time.

# 2.4 Missing Abstraction: OS Support for Automation

Despite growing demand for intelligent, language-driven automation, existing operating systems offer no first-class abstraction for exposing GUI application control to external agents. In contrast to system calls, files, or sockets, GUI workflows remain opaque and non-programmable. As a result, both RPA and CUA systems are forced to operate as ad-hoc layers atop the GUI, with no unified substrate for execution, coordination, or introspection.

![](images/e8522f74b651186919725786a0a31340cf114b9130e388b64672ea42c2dea081.jpg)  
Figure 2. An overview of the architecture of $\mathrm { U F O ^ { 2 } }$ .

This paper argues that automation should be elevated to a system primitive. We present $\mathbf { U F O ^ { 2 } }$ , a new AgentOS for Windows that addresses these limitations by embedding automation as a deeply integrated OS abstraction—exposing GUI controls, application APIs, and task orchestration as programmable, inspectable, and composable system services.

# 3 System Design of UFO2

Motivated by the challenges highlighted in Section 2, UFO2 is designed to seamlessly interpret natural language user requests and reliably automate tasks across a wide range of Windows applications. This section provides an architectural overview of $\mathrm { U F O ^ { 2 } }$ (Section 3.1) and explains how each component is deeply integrated with the underlying OS to overcomes the pitfalls of current CUAs, ultimately enabling a practical, robust AgentOS for desktop automation.

# 3.1 UFO2 as a System Substrate for Automation

Figure 2 presents the high-level architecture of $\mathrm { U F O ^ { 2 } }$ , which provides a structured runtime environment for task-oriented automation on Windows desktops. Deployed as a local daemon, $\mathrm { U F O ^ { 2 } }$ enables users to issue natural language requests that are translated into coordinated workflows spanning multiple GUI applications. The system provides core abstractions for orchestration, introspection, control execution, and agent collaboration—exposing these as system-level services analogous to those in traditional OSes.

At the heart of $\mathrm { U F O ^ { 2 } }$ is a central control plane, the HostAgent, responsible for parsing user intent, managing system state, and dispatching subtasks to a collection of specialized runtime modules called AppAgents. Each AppAgent is dedicated to a particular application (e.g., Excel, Outlook, File Explorer) and encapsulates all logic needed to observe and control that application, including API bindings, UI detectors, and knowledge bases. These modules act as isolated execution contexts with application-specific semantics.

Upon receiving a user request, HostAgent decomposes it into a series of subtasks, each mapped to the application

best suited to fulfill it. If the corresponding application is not already running, HostAgent launches it using native Windows APIs and instantiates the corresponding AppAgent. Execution proceeds through a structured loop: each AppAgent continuously observes the application state (via accessibility APIs and vision-based detectors), reasons about the next operation using a ReAct-style planning cycle [22], and invokes the appropriate action—either a GUI event or a native API call. This loop continues until the subtask terminates, either successfully or due to an unrecoverable error.

UFO2 implements shared memory and control flow via a global blackboard interface, allowing HostAgent and AppAgents to exchange intermediate results, dependency states, and execution metadata. This architecture supports complex workflows across application boundaries—for instance, extracting data from a spreadsheet and using it to populate fields in a web form—without requiring hand-crafted scripts or coordination logic. Crucially, all interactions occur within a virtualized, PiP-based desktop environment, ensuring process-level isolation and safe multi-application concurrency.

Design Rationale: Centralized Multiagent Runtime. $\mathrm { U F O ^ { 2 } }$ adopts a centralized multiagent [10, 23–25] runtime to support both reliability and extensibility. The centralized HostAgent acts as a control plane, simplifying task-level orchestration, error handling, and lifecycle management. Meanwhile, each AppAgent is architected as a loosely coupled executor that encapsulates deep, application-specific functionality.

Modularity at the AppAgent level allows developers and third-party contributors to incrementally expand UFO2’s capabilities by authoring new application interfaces and API bindings. These agents are discoverable, self-contained, and dynamically instantiated by the runtime as needed. From a security and evolvability perspective, this separation of concerns ensures that application logic can evolve independently of the core task orchestration engine.

Together, the HostAgent –AppAgent model allows UFO2 to function as a scalable, pluggable runtime substrate for GUI automation—abstracting away the complexity of heterogeneous interfaces and providing a unified system interface to structured application behavior.

# 3.2 HostAgent: System-Level Orchestration and Execution Control

The HostAgent serves as the centralized control plane of $\mathrm { U F O ^ { 2 } }$ . It is responsible for interpreting user-specified goals, decomposing them into structured subtasks, instantiating and dispatching AppAgent modules, and coordinating their progress across the system. HostAgent provides systemlevel services for introspection, planning, application lifecycle management, and multi-agent synchronization.

![](images/cb40b66e564169f78c6738560d4d5eea12ee234016bb05f751aef85f78176e60.jpg)  
Figure 3. High-level architecture of HostAgent as a controlplane orchestrator.

Figure 3 outlines the architecture of HostAgent. Operating atop the native Windows substrate, HostAgent monitors active applications, issues shell commands to spawn new processes as needed, and manages the creation and teardown of application-specific AppAgent instances. All coordination occurs through a persistent state machine, which governs the transitions across execution phases.

Responsibilities and Interfaces. HostAgent exposes the following system services:

• Task Decomposition. Given a user’s natural language input, HostAgent identifies the underlying task goal and decomposes it into a dependency-ordered subtask graph.   
• Application Lifecycle Management. For each subtask, HostAgent inspects system process metadata (via UIA APIs) to determine whether the target application is running. If not, it launches the program and registers it with the runtime.   
• AppAgent Instantiation. HostAgent spawns the corresponding AppAgent for each active application, providing it with task context, memory references, and relevant toolchains (e.g., APIs, documentation).   
• Task Scheduling and Control. The global execution plan is serialized into a finite state machine (FSM), allowing HostAgent to enforce execution order, detect failures, and resolve dependencies across agents.   
• Shared State Communication. HostAgent reads from and writes to a global blackboard, enabling interagent communication and system-level observability for debugging and replay.

System Perception and Introspection. To perform its control functions, HostAgent fuses two layers of system introspection:

1. Visual Layer. Captures pixel-level screenshots of the desktop workspace, enabling coarse-grained layout understanding.

![](images/8fb00ea4ca648c386c2906780672c6b07f343e5d197704f185cd15c252aae088.jpg)  
Figure 4. Control-state transitions managed by HostAgent.

2. Semantic Layer. Queries Windows UIA APIs to extract structural metadata about applications, windows, and control hierarchies.

This dual perception enables HostAgent to resolve ambiguities, detect runtime inconsistencies, and guide agents with context-aware decisions.

Structured Output Interface. HostAgent produces structured outputs to drive downstream execution:

• Subtask Plan: A high-level execution plan detailing decomposed subtasks.   
• Shell Command: A sequence of shell-level invocations for managing application lifecycles.   
• Assigned Application: The process name and index of the application selected for instantiating the AppAgent, which will be used to execute the next subtask.   
• Agent Messages: Context-specific instructions passed to AppAgent instances for localized execution.   
• User Prompts: Interactive clarification requests in cases of ambiguity or failure.   
• HostAgent State: The current state within the HostAgent’s internal FSM.

Execution via Finite-State Controller. The core logic of HostAgent is modeled as a finite-state controller (Figure 4), with the following states:

• CONTINUE: Main execution loop; evaluates which subtasks are ready to launch or resume.   
• ASSIGN: Selects an available application process and spawns the corresponding AppAgent agent.   
• PENDING: Waits for user input to resolve ambiguity or gather additional task parameters.   
• FINISH: All subtasks complete; cleans up agent instances and finalizes session state.   
• FAIL: Enters recovery or abort mode upon irrecoverable failure.

This explicit FSM structure enables HostAgent to robustly orchestrate dynamic workflows while maintaining high-level guarantees over task completion and fault isolation.

Memory and State Management. HostAgent maintains two classes of persistent state:

• Private State: Tracks user intent, plan progress, and the control flow of the current session.   
• Shared Blackboard: A concurrent, append-only memory space that facilitates transparent agent communication by recording key observations, intermediate results, and execution metadata accessible to all AppAgent instances.

This separation ensures that local context remains encapsulated, while global coordination is visible and consistent across the system.This separation ensures that each agent retains clean, scoped state while benefiting from a globally consistent view for collaborative task execution.

Overall, HostAgent abstracts away the complexity of managing concurrent, stateful, cross-application workflows in desktop environments [26]. Its control-plane role enables modular execution, coordinated progress, and robust task lifecycle management—all critical features in scaling desktop automation to real-world deployments.

# 3.3 AppAgent: Application-Specialized Execution Runtime

The AppAgent is the core execution runtime in $\mathrm { U F O ^ { 2 } }$ , responsible for carrying out individual subtasks within a specific Windows application. Each AppAgent functions as an isolated, application-specialized worker process launched and orchestrated by the central HostAgent (Section 3.2). Unlike monolithic CUAs that treat all GUI contexts uniformly, each AppAgent is tailored to a single application and operates with deep knowledge of its API surface, control semantics, and domain logic.

Figure 5 outlines the architecture of an AppAgent. Upon receiving a subtask and execution context from the HostAgent, the AppAgent initializes a ReAct-style control loop [22], where it iteratively senses the current application state, reasons about the next step, and executes either a GUI or APIbased action. This hybrid execution layer—implemented via a Puppeteer interface—enables reliable control over dynamic and complex UIs by favoring structured APIs whenever available, while retaining fallback to GUI-based interaction when necessary.

Perception Layer. Each AppAgent fuses multiple streams of perception:

• Visual Input: Captures GUI screenshots for layout understanding and control grounding.   
• Semantic Metadata: Extracted from Windows UIA APIs, including control type, label, hierarchy, and enabled state.   
• Symbolic Annotation: Uses Set-of-Mark (SoM) techniques [27] to annotate the control on screenshots.

These fused signals are converted into a structured observation object containing both the GUI screenshot and the set of candidate control elements. This multi-modal representation enables a more comprehensive understanding of the application state, going well beyond raw visual input alone.

Structured Output. Based on this state, the AppAgent produces a structured output:

• Target control (if applicable)   
• Action type (e.g., click, type, call API)   
• Arguments or payload   
• Reasoning trace and planning with Chain-of-Thought (CoT) [28, 29]   
• Current state in local FSM

This design decouples perception from actuation, enabling deterministic replay, offline debugging, and fine-grained observability.

Execution via Finite-State Controller. Each AppAgent maintains a local finite-state machine (Figure 6) that governs its behavior within the assigned application context:

• CONTINUE: Default state for action planning and execution.   
• PENDING: Invoked for safety-critical actions (e.g., destructive operations); requires user confirmation.   
• FINISH: Task completed; execution ends.   
• FAIL: Irrecoverable failure detected (e.g., application crash, permission error).

This bounded execution model isolates failures to the current task and enables safe preemption, retry, or delegation. The FSM also supports interruptible workflows, which can resume from intermediate checkpoints.

Memory and State Coordination. To enable stateful execution and maintain contextual awareness, each AppAgent maintains:

• Private State: A local log of all executed actions, control decisions, and CoT traces.   
• Shared State: Updates to the system-wide blackboard, including intermediate outputs, encountered errors, and application-level insights.

This dual-memory design enables AppAgents to act autonomously on behalf of HostAgent while remaining synchronized with the broader system. It also supports composability: one AppAgent’s output may become another’s input in downstream subtasks.

Application-Aware SDK and Extensibility. To support rapid onboarding of new applications, $\mathrm { U F O ^ { 2 } }$ exposes an SDK that encapsulates the development and maintenance of AppAgents. Developers can register application-specific APIs via a declarative interface, including function metadata, argument schemas, and prompt bindings. Domain-specific help documents and patch notes can be ingested into a searchable knowledge base that agents query at runtime.

![](images/09859ef978aacf89122db6da7b1babef1f6c55b630b600c66378d19a686695fa.jpg)  
Figure 5. Architecture of an AppAgent, the per-application execution runtime in UFO2.

![](images/f63d8f07fd146e711b5511d272e9068b842be40edd289325c7172a81eb4a23fb.jpg)  
Figure 6. Control-state transitions for an AppAgent runtime.

This modular abstraction allows third-party vendors or power users to extend UFO2’s capabilities without retraining any models. New functionality can be integrated by updating the application’s AppAgent module, isolating changes from the rest of the system and minimizing regression risk.

Summary. As a per-application execution runtime, each AppAgent provides modular, domain-aware control that surpasses generic GUI agents in both efficiency and robustness. Its hybrid perception-action loop, plugin-based extensibility, and local fault containment enable $\mathrm { U F O ^ { 2 } }$ to scale to large application ecosystems with minimal system-wide disruption.

# 3.4 Hybrid Control Detection

Reliable perception of GUI elements is fundamental to enabling AppAgents to interact with application interfaces in a deterministic and safe manner. However, real-world GUI environments exhibit substantial heterogeneity: some applications expose well-structured accessibility data via Windows UI Automation (UIA) APIs, while others—especially legacy or custom applications—render critical controls using non-standard toolkits that bypass UIA entirely.

To address this disparity, $\mathrm { U F O ^ { 2 } }$ introduces a hybrid control detection subsystem that fuses UIA-based metadata with

vision-based grounding [18, 30, 31] to construct a unified and comprehensive control graph for each application window (Figure 7). This design ensures both coverage and reliability, forming a resilient perceptual foundation for downstream action planning and execution.

UIA-Layer Detection. When available, UIA offers a semantically rich and high-precision interface for enumerating on-screen controls. The detection pipeline first queries the accessibility tree to extract controls satisfying a set of runtime predicates (e.g., is_visible(), is_enabled()). These controls are annotated with their attributes (type, label, bounding box) and assigned stable identifiers, forming the initial control graph.

Vision-Layer Augmentation. To augment the perception pipeline for UI-invisible or custom-rendered controls, we integrate OmniParser-v2 [18], a vision-based grounding model designed for fast and accurate GUI parsing. OmniParserv2 combines a lightweight YOLO-v8 [32] detector with a fine-tuned Florence-2 (0.23B) [33] encoder to process raw application screenshots and identify additional interactive elements. Each detection includes the control type, confidence score, and spatial bounding box.

Fusion and Deduplication. We unify these two streams by performing deduplication based on bounding-box overlap. Visual detections with Intersection-over-Union (IoU) greater than $1 0 \%$ against any UIA-derived control are discarded. Remaining visual-only detections are converted into pseudo-UIA objects using a lightweight UIAWrapper abstraction, allowing them to seamlessly integrate into the rest of the AppAgent pipeline. This fused control set is passed downstream to the SoM-based annotation module [27]. Figure 7 shows a typical scenario involving a hybrid-rendered GUI. Yellow bounding boxes denote standard UIA-detected elements, while blue bounding boxes represent visual-only detections. Both are integrated into a single actionable control graph consumed by the AppAgent.

![](images/f9cd36335a2561365c7f1e120631e2f7ff862262bbd4481d8cbcb521877bef33.jpg)  
Figure 7. The hybrid control detection approach employed in UFO2.

![](images/6200f87dc0d25d8b658a09939b4bd294f2702abfe0f13de8b886c00792fdcd7a.jpg)  
Figure 8. Puppeteer serves as a unified execution engine that harmonizes GUI actions and native API calls.

# 3.5 Unified GUI–API Action Orchestrator

AppAgents can interact with application environments that expose two distinct classes of interfaces: GUI frontends, which are universally observable but often brittle; and native APIs, which are high-fidelity but require explicit integration [17]. To unify these heterogeneous execution backends under a single runtime abstraction, $\mathrm { U F O ^ { 2 } }$ introduces the Puppeteer, a modular execution orchestrator that dynamically selects between GUI-level automation and applicationspecific APIs for each action step (Figure 8). This design significantly improves task robustness, latency, and maintainability. Tasks that would otherwise require long GUI interaction sequences (e.g., iteratively selecting and formatting cells in Excel) can often be collapsed into a single API

![](images/f5c6b393e5b396d5ec42d3c880360b9dca187831a4f824d9c012ace93a8cf7b3.jpg)  
Figure 9. Example API registration for Excel.

call, reducing both execution time and the surface area for failure [17, 34].

Puppeteer supports a lightweight API registration mechanism that enables developers to expose high-level operations in target applications. APIs are registered using a simple Python decorator interface, as shown in Figure 9. Each function is wrapped with metadata—name, argument schema, and application binding—and automatically incorporated into the AppAgent’s runtime action space.

At runtime, UFO2 prompts AppAgent to employs a decisionmaking policy to choose the most appropriate execution path for each operation. If a semantically equivalent API is available, Puppeteer prefers it over GUI automation for reliability and atomicity.If an API fails or is unavailable (e.g., missing bindings or runtime permission errors), the system gracefully falls back to GUI-based control via simulated clicks or keystrokes. This runtime flexibility allows AppAgent to preserve robustness across heterogeneous environments without sacrificing generality.

Puppeteer transforms action execution in $\mathrm { U F O ^ { 2 } }$ from a monolithic GUI-centric model into a flexible, OS-integrated

![](images/1f8a673e7106154f6dac6f390529de05c0416d53aaf1f572ce2ad5d58bfc2f94.jpg)  
Figure 10. Overview of the knowledge substrate in $\mathrm { U F O ^ { 2 } }$ , combining static documentation with dynamic execution history.

control layer that mixes perceptual agility with semantic precision. This hybrid execution model not only improves system performance and stability but also lays the groundwork for sustainable integration of application-specific capabilities in future desktop agents.

# 3.6 Continuous Knowledge Integration Substrate

Unlike traditional CUAs, which rely heavily on static training corpora, $\mathrm { U F O ^ { 2 } }$ introduces a persistent and extensible knowledge substrate that supports runtime augmentation of application-specific understanding. As illustrated in Figure 10, this substrate enables each AppAgent to retrieve, interpret, and apply external documentation and prior execution traces without requiring retraining of the underlying models. This hybrid memory design functions analogously to an OS-level metadata manager, abstracting over two key knowledge flows: static references (e.g., user manuals) and dynamic experience (e.g., execution logs).

Bootstrapping from Documentation. Most real-world desktop applications expose substantial task-level documentation via user guides, help menus, or online tutorials. UFO2 capitalizes on this resource by offering a one-click interface to parse and ingest such documents into an applicationspecific vector store. Documents are structured as json records with a natural language description in the request field and detailed execution guidance in the guidance field.

At runtime, when an AppAgent receives a subtask, it queries this indexed store to retrieve relevant guidance, which is then injected into the agent’s prompt. This mechanism

effectively mitigates cold-start issues—especially when handling novel applications or infrequent operations—by enriching the agent’s reasoning context with domain-grounded procedural knowledge.

Reinforcing from Experience. Beyond static knowledge, $\mathrm { U F O ^ { 2 } }$ continuously learns from its own execution history. Each automation run produces structured logs—including natural language task descriptions, executed action sequences, application screenshots, and final outcomes. Periodically, these logs are mined offline by a summarization module that distills successful trajectories into reusable Example records.

Each record contains a task signature and an associated step-by-step plan, stored in an application-specific example database. When a similar task is encountered in the future, AppAgent uses In-Context Learning (ICL) [35–38] to retrieve relevant demonstrations and improve execution fidelity. This dynamic reinforcement pipeline transforms the system into a long-lived agent that improves with use, without introducing the brittleness or operational cost of fine-tuning [39].

Runtime RAG Integration. At the system level, the knowledge substrate acts as a Retrieval-Augmented Generation (RAG) layer [40–43] that bridges the gap between pre-trained language models and application-specific requirements. Because both help documents and examples are indexed with semantic embeddings, the retrieval pipeline is fast, interpretable, and cache-friendly. Additionally, versioned indexing ensures that knowledge artifacts can evolve alongside software updates, preventing model obsolescence and supporting robust execution across long deployment cycles.

By integrating static and experiential knowledge into a unified RAG pipeline, $\mathrm { U F O ^ { 2 } }$ transforms CUAs from brittle, training-time constructs into dynamic, evolving agents. This substrate plays a foundational role in enabling sustainable automation across complex, heterogeneous application ecosystems.

# 3.7 Speculative Multi-Action Execution

Conventional CUAs suffer from a fundamental execution bottleneck: each automation step is executed in isolation, requiring a full LLM inference for every single GUI action. This step-wise inference loop introduces excessive latency, inflates system resource usage, and increases cumulative error rates—especially when interacting with complex or multi-phase workflows [17]. The root cause is the dynamic and uncertain nature of GUI environments, where any single action may alter the interface and invalidate future plans.

To overcome these limitations, $\mathrm { U F O ^ { 2 } }$ introduces a systemlevel optimization called speculative multi-action execution, inspired by classical ideas from speculative execution in processor design and instruction pipelining. Rather than issuing one action per LLM call, $\mathrm { U F O ^ { 2 } }$ speculatively generates a batch of likely next steps using a single inference pass and

![](images/4932ef9340d1737971efd04910931d0175bb01eac76f64b361faf315468c23f8.jpg)  
Figure 11. Speculative multi-action execution in $\mathrm { U F O ^ { 2 } }$ : batched inference with online validation.

Algorithm 1 Speculative Multi-Action Execution in UFO2

Require: Initial UI context $C _ { 0 }$ , batch size $k$

Ensure: List Executed of actions completed so far

# Stage 1: Batch Prediction

1: ?? ← LLM_Predict $( C _ { 0 } , k )$ $\mathsf { \Sigma } \circ A = [ ( c t r l _ { i } , o p _ { i } ] _ { i = 1 } ^ { k }$   
2: Executed $ [ ]$ ; $C  C _ { 0 }$

# Stage 2 & 3: Sequential Validate-Execute Loop

3: for $i \gets 1$ to $k$ do   
4: $( c t r l , o p , \_ ) \gets A [ i ]$

// Validate in the current context

5: if not UIA_IsEnabled $( c t r l , C )$ ∨ not UIA_IsVisible $( c t r l , C )$ then   
6: break ⊲ validation failed early stop   
7: end if //Execute and refresh context   
8: Execute(????????, ????)   
9: append $( c t r l , o p )$ to Executed   
10: $C \gets$ UIA_GetContext() ⊲ UI changed

11: end for   
12: if |Executed| $< k$ then   
13: ReportPartial(Executed)   
14: Replan(??)   
15: end if

validates their applicability at runtime through tight OS integration. We present an algorithm in 1.

The speculative executor operates in three stages:

1. Action Prediction: The AppAgent issues a single LLM query to predict multiple plausible actions under its current context. Each predicted step includes a target control, intended operation, and rationale.   
2. Runtime Validation: For each action, the system consults the Windows UIA API to verify the action’s preconditions (e.g., is_enabled(), is_visible()). This

check ensures that each target control is still valid and interactive.

3. Sequential Execution and Early Exit: Actions are executed in order, halting immediately if any validation fails due to interface change (e.g., control no longer exists or is disabled). The executor then reports a partial result set and prompts the agent to replan.

We show an illustrative example of speculative multi-action execution in Figure 11. In this case, the AppAgent initially plans to execute three actions in a single step: clicking Paste, then Quick Style, and finally Grid Filled. However, after the second action, the control validator detects that the control required for the third action (Grid Filled) is no longer present—likely because the GUI layout changed as a result of the previous step. The Puppeteer then terminates execution at that point and returns the partial results. This example highlights how UFO2 safely handles speculative execution by validating each control before acting, ensuring robustness even in the face of dynamic interface changes.

Overall, this strategy drastically reduces LLM invocation frequency and amortizes the cost of action planning across multiple steps, while preserving the correctness guarantees of per-step validation. Critically, validation is performed by trusted OS-level APIs instead of vision models, ensuring high reliability and eliminating spurious interactions.

# 4 Picture-in-Picture Interface

A key design objective of $\mathrm { U F O ^ { 2 } }$ is to deliver high-throughput automation while preserving the responsiveness and usability of the primary desktop environment. Existing CUAs often monopolize the user’s workspace, seizing mouse and keyboard control for extended periods and making the system effectively unusable during task execution. To overcome this, $\mathrm { U F O ^ { 2 } }$ introduces a Picture-in-Picture (PiP) interface: a lightweight, virtualized desktop window powered by Remote Desktop loopback, enabling fully isolated agent execution

![](images/95b12d97f26cb08655f2721dcef29c17a567e7ae4605764f0dc5f75d26003812.jpg)  
Figure 12. The Picture-in-Picture interface: a virtual desktop window for non-disruptive automation.

in parallel with active user workflows, as illustrated in Figure 12.

# 4.1 Virtualized User Environment with Minimal Disruption

Unlike conventional CUAs that operate in the main desktop session, the PiP interface presents a resizable, movable window containing a fully functional replica of the user’s desktop. Internally, this is implemented via Windows’ native Remote Desktop Protocol (RDP) loopback [44], creating a distinct virtual session hosted on the same machine. Applications launched within the PiP session inherit the user’s identity, credentials, settings, and network context, ensuring consistency with foreground operations.

From the user’s perspective, the PiP window behaves like a sandboxed workspace: the automation executes in the background, visible but unobtrusive. The user retains full control of the primary desktop and can minimize or reposition the PiP window at will. This enables $\mathrm { U F O ^ { 2 } }$ to perform longrunning or repetitive workflows (e.g., data entry, batch file processing) without blocking user interaction or degrading responsiveness.

# 4.2 Robust Input and State Isolation

To ensure robust separation between agent actions and user activities, $\mathrm { U F O ^ { 2 } }$ leverages the RDP subsystem to maintain distinct input queues and device contexts across sessions. Mouse and keyboard events generated within the PiP desktop are fully scoped to that session and cannot interfere with the primary desktop. Similarly, GUI changes and focus transitions are restricted to the virtual environment.

This level of input isolation is critical for preventing accidental interference—either by the user or the agent—and ensures that automation sequences remain stable, even during simultaneous foreground activity. The architecture also supports controlled error recovery: failures or unexpected UI states within the PiP session do not propagate to the primary desktop, preserving the integrity of the user’s environment.

# 4.3 Secure Cross-Session Coordination

Although visually and operationally distinct, the PiP session must remain logically connected to the host environment. To enable this, $\mathrm { U F O ^ { 2 } }$ establishes a secure inter-process communication (IPC) channel between the PiP agent runtime and a host-side coordinator. We implement this using Windows

![](images/6c32697d21fae9d2dbec8ae556a04d38acb9ba76cb917fd0c6b34a9b22016277.jpg)  
Figure 13. The interactive Session model in UFO2 supports multi-round refinement.

Named Pipes, authenticated and encrypted using per-session credentials [45].

This IPC layer supports two-way messaging:

• From the host to the PiP: task assignment, progress polling, cancellation, and user clarifications.   
• From the PiP to the host: status updates, completion reports, and exception notifications.

Users interact with the automation pipeline through a lightweight frontend panel on the host desktop, enabling real-time visibility and partial control without needing to directly access the PiP window. This transparent yet secure communication channel ensures trust and usability, particularly in long-running or partially supervised workflows.

# 4.4 System-Level Implications

The PiP interface represents more than a UX refinement—it is a system-level abstraction that reconciles concurrency, usability, and safety. It decouples automation execution from foreground interactivity, introduces a new isolation primitive for GUI-based agents, and simplifies failure recovery by sandboxing side effects. By exploiting existing RDP capabilities with minimal system overhead, the PiP interface offers a practical and backwards-compatible approach to scalable desktop automation.

# 5 Implementation and Specialized Engineering Design

We implement $\mathrm { U F O ^ { 2 } }$ as a full-stack desktop automation framework spanning over 30,000 lines of Python and C# code. Python serves as the core runtime environment for agent orchestration, control logic, and API integration, while C# supports GUI development, debugging interfaces, and Windows-specific operations such as the Picture-in-Picture desktop. To support retrieval-augmented reasoning, UFO2 leverages Sentence Transformers [46] for embedding-based document and experience retrieval.

Beyond its core functionality, $\mathrm { U F O ^ { 2 } }$ incorporates multiple specialized engineering components that target critical systems goals: composability, interactivity, debuggability, and scalable deployment. We highlight several key mechanisms below.

![](images/7e2362e82b3ce997932a1aedf1e8661ea67cd5f8d2f036078ba32e7f5ad36e2c.jpg)  
Figure 14. The safeguard mechanism employed in $\mathrm { U F O ^ { 2 } }$ .

# 5.1 Multi-Round Task Execution

Unlike stateless one-shot agents, $\mathrm { U F O ^ { 2 } }$ adopts a sessionbased execution model to support iterative, interactive workflows (Figure 13). Each Session maintains persistent contextual memory—including intermediate results, task progress, and application state—across multiple Rounds of execution. Users can refine prior instructions, launch follow-up tasks, or intervene when agents encounter ambiguous or unsafe operations.

This multi-round interaction paradigm facilitates progressive convergence on complex tasks while preserving transparency and human oversight. It enables $\mathrm { U F O ^ { 2 } }$ to support human-in-the-loop refinement strategies, bridging static LLM workflows with dynamic user guidance.

# 5.2 Safeguard Mechanism

While automation substantially boosts productivity, any CUA carries inherent risks of executing unsafe actions that may adversely affect user data or system stability [10, 47]. Examples include deleting critical files, terminating applications prematurely (resulting in unsaved data loss), or activating sensitive devices such as webcams without explicit consent. These actions pose severe risks, potentially causing irrecoverable damage or security breaches.

To mitigate such risks, $\mathrm { \Delta U F O ^ { 2 } }$ incorporates an explicit safeguard mechanism, designed to actively detect potentially dangerous actions, as shown in Figure 14. Specifically, whenever an AppAgent identifies an action matching predefined risk criteria, it transitions into a dedicated PENDING state, pausing execution and actively prompting the user for confirmation. Only upon receiving explicit user consent does the agent proceed; otherwise, the action is aborted to prevent harm. The definition and scope of what constitutes a risky action are fully customizable through a straightforward prompt-based interface, enabling users and system administrators to precisely tailor safeguard behavior according to their organization’s specific risk policies. This flexibility allows the safeguard system to be dynamically adapted as automation requirements evolve.

![](images/326acbe2add906db3d0be07cf3e0cd0f06145e5178c9c137c44bc6dc10b49f28.jpg)

![](images/3e0a13a31b4789047cbc72620bc0a630917645a5ad0dedca4b37b2d0dee7994e.jpg)  
Figure 15. The agent registry supports seamless wrapping of third-party components into the AppAgent framework.   
Figure 16. The client-server deployment model used in AgentOS-as-a-Service.

Through this proactive safety-checking framework, UFO2 significantly reduces the likelihood of executing harmful operations, thus enhancing overall system safety, user trust, and robustness in real-world deployments.

# 5.3 Everything-as-an-AppAgent

To support ecosystem extensibility, UFO2 introduces an agent registry mechanism that encapsulates arbitrary third-party components as pluggable AppAgents (Figure 15). Through a simple registration API, external automation solutions—such as domain-specific copilots or proprietary tools—can be wrapped with lightweight compatibility shims that expose a unified interface to the HostAgent.

This design enables HostAgent to treat native and external AppAgents interchangeably, dispatching subtasks based on capability and specialization. We find that even minimal wrappers (e.g., for OpenAI Operator [12]) lead to tangible performance gains, highlighting the system’s modularity and its ability to incorporate diverse execution backends with minimal engineering overhead.

# 5.4 AgentOS-as-a-Service

$\mathrm { U F O ^ { 2 } }$ adopts a client-server architecture to support practical deployment at scale (Figure 16). A lightweight client resides on the user’s machine and is responsible for GUI operations and application-side sensing. Meanwhile, a centralized server (running on-premises or in the cloud) hosts the HostAgent/AppAgent logic, orchestrates workflows, and handles LLM queries.

This separation of control and execution offers several systems-level benefits:

• Security: Sensitive orchestration and model execution are isolated from user devices.   
• Maintainability: Server-side updates propagate without modifying the client.   
• Scalability: The system can support multiple concurrent clients with centralized scheduling and load management.

The client-server boundary enforces a clean service abstraction, promoting modularity and simplifying rollout in enterprise environments.

# 5.5 Comprehensive Logging and Debugging Infrastructure

Robust observability is essential for diagnosing failures and supporting ongoing system improvement. To this end, UFO2 implements a comprehensive logging and debugging framework. Each session captures fine-grained traces of execution: prompts, LLM outputs, control metadata, UI state snapshots, and error events.

At the end of each session, $\mathrm { U F O ^ { 2 } }$ compiles these artifacts into a structured, Markdown-formatted execution log. Developers can inspect action-by-action agent decisions, visualize interface state transitions, and replay behavior for debugging. The framework also supports prompt editing and selective replay for targeted hypothesis testing, significantly accelerating the debugging cycle. We show an example of these tools in Figure 17.

This observability layer functions as a lightweight provenance system for agent behavior, fostering transparency, accountability, and rapid iteration during deployment.

# 5.6 Automated Task Evaluator

To provide structured feedback and facilitate continuous improvement, $\mathrm { U F O ^ { 2 } }$ includes an automated task evaluation engine based on LLM-as-a-judge [48]. As shown in Figure 18, the evaluator parses session traces—including actions, rationales, and screenshots—and applies CoT reasoning to decompose tasks into evaluation criteria.

It assigns partial scores and synthesizes an overall result: success, partial, or failure. This structured outcome feeds into downstream dashboards and debugging tools. It also supports self-monitoring and offline analysis of failure cases, closing the loop between execution, diagnosis, and improvement.

Summary. These engineering components demonstrate UFO2’s commitment to operational robustness and extensibility. From session-based execution and pluggable agents to serviceoriented deployment and observability infrastructure, each module reflects a design focused on bridging conceptual LLM agent architectures with the systems realities of deployment at scale.

![](images/fa01e55536a8ff73435a1cdba06d2af899b524258b9c3cc0c7e08ac7f07f1ca6.jpg)

![](images/b251f0c7144a5ce970f0a2bd6a00273201e8b0d67ac9de0da00ff1d0de0e2d69.jpg)  
Figure 17. An illustration of the markdown-formatted log viewer and debugging tool in UFO2.

![](images/7cb725ef2fdd2e5dad96c8e33cb4f278ea0c28b114cc202af076059abe0257a0.jpg)  
Figure 18. The LLM-based task evaluator applies CoT reasoning to structured session logs.

# 6 Evaluation

We tested $\mathrm { U F O ^ { 2 } }$ rigorously across more than 20 Windows applications, including office suites, file explorers, and custom enterprise tools to assess performance, efficiency, and robustness. Our experiments show:

1. UFO2 achieves a $1 0 \%$ higher task completion rate, a $5 0 \%$ relative improvement—over the best-performing current CUA Operator, enabled by deeper OS-level integration.   
2. The hybrid UIA–vision approach identifies custom or nonstandard GUI elements missed by UIA alone, boosting success in interfaces with proprietary widgets.   
3. Allowing AppAgents to invoke native APIs or GUI interactions to improve completion rate by over $8 \%$ ,

cuts latency and reduces the fragility seen in purely click-based workflows.

4. Leveraging external documents and execution logs increases $\mathrm { U F O ^ { 2 } }$ ’s ability to handle unfamiliar features without retraining.   
5. Speculative multi-action execution consolidates multiple steps into a single LLM call, lowering inference cost by up to $5 1 . 5 \%$ without compromising reliability.   
6. By enabling Everything-as-an-AppAgent (e.g., Operator), $\mathrm { U F O ^ { 2 } }$ both boosts overall performance and uncovers the full potential of each individual agent.

Overall, these results confirm that $\mathrm { U F O ^ { 2 } } ^ { , }$ s deeper integration with Windows and application-level APIs yields both higher performance and reduced overhead, making a compelling case for an OS-native approach to desktop automation.

# 6.1 Experimental Setup

Deployment Environment. The benchmark environments are hosted on isolated VMs with 8 AMD Ryzen 7 CPU cores and 8 GB of memory, matching typical deployment conditions. All GPT-family models (GPT-4V, GPT-4o, o1, and Operator) are accessed via Azure OpenAI services, while the OmniParser-v2 vision model operates on a separate virtual machine provisioned with an NVIDIA A100 80GB GPU to support efficient and high-throughput visual grounding.

Benchmarks. We evaluate $\mathrm { U F O ^ { 2 } }$ using two established Windows-centric automation benchmarks:

• Windows Agent Arena (WAA) [49]: Consists of 154 live automation tasks across 15 commonly used Windows applications, including office productivity tools, web browsers, system utilities, development environments, and multimedia apps. Each task includes a custom verification script for automated correctness checking.   
• OSWorld-W [50]: A targeted subset of the OSWorld benchmark specifically tailored for Windows, comprising 49 live tasks across office applications, browser

interactions, and file-system operations. Tasks are similarly equipped with handcrafted verification scripts for reliable outcome validation.

Each task runs independently, and verification strictly follows the original scripts provided by each benchmark.1

Baselines. We compare $\mathrm { U F O ^ { 2 } }$ with five representative state-of-the-art CUAs, each leveraging GPT-4o as the inference engine:

• UFO [10]: A pioneering multiagent, GUI-focused automation system designed explicitly for Windows, integrating UIA and visual perception.   
• NAVI [49]: A single-agent baseline from WAA, utilizing screenshots and accessibility data for GUI understanding.   
• OmniAgent [18]: Employs OmniParser for visual grounding combined with GPT-based action planning.   
• Agent S [51]: Features a multiagent architecture with experience-driven hierarchical planning, optimized for complex, multi-step tasks.   
• Operator [12]: A recent, high-performance CUA from OpenAI, simulating human-like mouse and keyboard interactions via screenshots.

These baselines were selected for their representativeness of diverse architectural and design paradigms (e.g., single-agent vs. multiagent, GUI-only vs. hybrid approaches). To ensure fairness, each agent is restricted to a maximum of 30 execution steps per task, reflecting practical user expectations and preventing excessively long task executions. Additionally, we evaluate a base version of $\mathrm { U F O ^ { 2 } }$ (termed $\mathrm { U F O ^ { 2 } }$ -base) using only UIA detection, GUI-based interactions, and without dynamic knowledge integration, alongside the full implementation of $\mathrm { U F O ^ { 2 } }$ featuring hybrid control detection, combined GUI-API interactions, and continuous knowledge augmentation. API integrations were selectively implemented for three office applications within OSWorld-W as illustrative examples; no APIs were introduced for the WAA tasks. Further implementation details are available in Section 6.4.

Evaluation Metrics. We utilize two primary metrics for performance evaluation:

• Success Rate (SR): Defined as the percentage of tasks successfully completed, validated via the benchmarks’ own verification scripts.   
• Average Completion Steps (ACS): Measures the average number of LLM-involved action inference steps required per task. Fewer steps correspond to higher efficiency, directly correlating with lower inference latency and reduced computational overhead.

Table 1. Comparison of success rates (SR) across agents on WAA and OSWorld-W benchmarks.   

<table><tr><td>Agent</td><td>Model</td><td>WAA</td><td>OSWorld-W</td></tr><tr><td>UFO</td><td>GPT-4o</td><td>19.5%</td><td>12.2%</td></tr><tr><td>NAVI</td><td>GPT-4o</td><td>13.3%</td><td>10.2%</td></tr><tr><td>OmniAgent</td><td>GPT-4o</td><td>19.5%</td><td>8.2%</td></tr><tr><td>Agent S</td><td>GPT-4o</td><td>18.2%</td><td>12.2%</td></tr><tr><td>Operator</td><td>computer-use</td><td>20.8%</td><td>14.3%</td></tr><tr><td>UFO2-base</td><td>GPT-4o</td><td>23.4%</td><td>16.3%</td></tr><tr><td>UFO2-base</td><td>o1</td><td>25.3%</td><td>16.3%</td></tr><tr><td>UFO2</td><td>GPT-4o</td><td>27.9%</td><td>28.6%</td></tr><tr><td>UFO2</td><td>o1</td><td>30.5%</td><td>32.7%</td></tr></table>

These metrics effectively reflect both functional effectiveness and practical efficiency, providing clear indicators of realworld automation performance.

# 6.2 Success Rate Comparison

Table 1 summarizes the success rates (SR) of all evaluated agents across the WAA and OSWorld-W benchmarks, as verified by each benchmark’s automated validation scripts. Notably, even the basic configuration (UFO2-base)—which relies solely on standard UI Automation and GUI-driven actions—consistently surpasses prior state-of-the-art CUAs. Specifically, with GPT-4o, UFO2-base achieves an SR of $2 3 . 4 \%$ on WAA, outperforming the best existing baseline, Operator $( 2 0 . 8 \% )$ , by $2 . 6 \%$ . This margin widens significantly when employing the stronger o1 LLM, lifting $\mathrm { U F O ^ { 2 } }$ -base’s performance to $2 5 . 3 \%$ .

Moreover, the complete version of $\mathrm { U F O ^ { 2 } }$ , incorporating hybrid GUI–API action execution, advanced visual grounding, and continuous knowledge integration, further amplifies these performance gains. With GPT-4o, $\mathrm { U F O ^ { 2 } }$ achieves a $2 7 . 9 \%$ SR on WAA, exceeding Operator by a substantial $7 . 1 \%$ . The performance gap becomes even more pronounced on OSWorld-W, where $\mathrm { U F O ^ { 2 } }$ achieves a $2 8 . 6 \%$ SR compared to Operator’s $1 4 . 3 \%$ , effectively doubling its success rate. Utilizing the stronger o1 model further improves $\mathrm { U F O ^ { 2 } } ^ { , }$ s performance to $3 0 . 5 \%$ (WAA) and $3 2 . 7 \%$ (OSWorld-W), solidifying its leading position.

These significant performance improvements clearly underscore the advantages of UFO2’s deep integration with OSlevel mechanisms and its unified system architecture. While prior CUAs primarily emphasize model-level optimization or singular reliance on visual interfaces, our results demonstrate that robust, system-level orchestration—combining structured OS APIs, specialized application knowledge, and hybrid GUI–API interaction—is instrumental in achieving higher task reliability and broader automation coverage. Crucially, even a general-purpose, less-specialized model like GPT-4o can surpass highly specialized CUAs (such as Operator) when integrated within the comprehensive UFO2

framework. This insight reinforces the value of architectural design and OS integration as key drivers of practical, deployable desktop automation solutions.

Performance Breakdown. Table 2 presents a detailed breakdown of success rates (SR) by application type on the WAA and OSWorld-W benchmarks, enabling deeper understanding of where $\mathrm { U F O ^ { 2 } }$ achieves particularly strong results and identifying areas for further system-level improvements. Across multiple categories, $\mathrm { U F O ^ { 2 } }$ consistently demonstrates superior performance compared to baseline CUAs, particularly in application scenarios demanding deeper OS integration or sophisticated multi-step task execution.

Notably, $\mathrm { U F O ^ { 2 } }$ excels in tasks involving web browsers and coding environments. For instance, the strongest configuration (UFO2 with o1) attains an impressive $4 0 . 0 \%$ SR for web browser tasks—markedly outperforming the next-best baseline (OmniAgent) by over $1 2 \%$ . Similarly, in coding-related workflows, $\mathrm { U F O ^ { 2 } }$ (GPT-4o) achieves the highest SR of $5 8 . 3 \%$ , significantly exceeding all competing CUAs. These results underscore the effectiveness of $\mathrm { U F O ^ { 2 : } }$ ’s hybrid GUI-API approach and continuous knowledge integration, which enable more precise action inference, reduce brittleness due to GUI changes, and substantially elevate reliability in multi-step workflows.

The breakdown further reveals a clear correlation between application complexity, popularity, and system-level support. Tasks involving LibreOffice (in the Office category of WAA) uniformly yield lower SRs across all evaluated CUAs, largely due to inadequate adherence to accessibility standards and incomplete UIA support. Conversely, OSWorld-W tasks predominantly utilize Microsoft 365 Office applications, which offer richer OS-native APIs and structured accessibility data, resulting in improved SRs (up to $5 1 . 9 \%$ for UFO2-o1). This discrepancy highlights the critical role that robust OS-level integration and API availability play in achieving high-quality desktop automation.

Cross-application tasks, especially prominent in OSWorld-W, present an even greater challenge. Such tasks inherently require sophisticated task decomposition and robust interagent coordination, pushing CUAs—and even human users—to their limits. Here, UFO2’s multiagent architecture, led by the centralized HostAgent and specialized AppAgents, demonstrates notable promise, outperforming other baselines with a $9 . 1 \%$ SR. Although performance remains relatively modest, it clearly illustrates the strength of systematic multiagent collaboration and centralized orchestration in addressing complex scenarios that cross traditional application boundaries.

Overall, these detailed breakdown results validate the system-level design principles of $\mathrm { U F O ^ { 2 } }$ , particularly its emphasis on deep OS and application-specific integration, multiagent coordination, and flexible action orchestration. While there remains significant potential for further enhancements

![](images/cb8ca0384aaa5a9eed6a7790a5d764a713c5fb02acb0f513ccfb7fb0617f57d9.jpg)

![](images/2fa6a3d553bbe8eabf263128f671f4794de204d9cdd415c799f0eac93aa4fcf7.jpg)

![](images/a66c885cea3a702957160ab0b397c9047b8e63f2fcb1cff8e187fb295cbe147d.jpg)

![](images/5bc24dbd60cf60b98e752fd3651ada0ccaaf31b2a150e2dc099369250a0a5b79.jpg)

![](images/7a484d0a7e8cfbb4fecd813fd54e88b89e1c17b9982556e5d93d8ce4a2262cda.jpg)  
Figure 19. Error analysis of $\mathrm { U F O ^ { 2 } }$ -base (GPT-4o) on the two benchmarks.

in niche or less-supported application domains (e.g., custom or legacy software with limited API availability), UFO2’s current architecture already provides a substantial, measurable improvement in practical, real-world desktop automation tasks.

Error Analysis. To systematically understand the limitations of $\mathrm { U F O ^ { 2 } }$ and identify opportunities for further improvement, we conducted a detailed manual review of all failure cases for $\mathrm { U F O ^ { 2 } }$ -base (GPT-4o) on both benchmarks. Following a classification framework similar to Agashe et al., [51], each failure was categorized into one of three distinct system-level categories:

• Plan Errors: Failures arising from inadequate highlevel task understanding, typically reflected by incomplete or incorrect action plans. These errors indicate gaps in the agent’s task comprehension or insufficient grounding in application-specific workflows.   
• Execution Errors: Cases where the high-level plan is reasonable but the execution is flawed (e.g., selecting an incorrect control, performing unintended actions). Execution errors often stem from inaccurate visual reasoning, incorrect associations between GUI elements and actions, or erroneous inference by the LLM.   
• Control Detection Failures: Instances where the agent fails to detect or identify critical GUI controls required to complete a task, usually due to non-standard or custom-rendered UI elements that are not fully accessible via standard OS APIs.

Figure 19 summarizes our findings for $\mathrm { U F O ^ { 2 } }$ -base. On the WAA benchmark, more than $6 2 \%$ of failures were attributed to Control Detection Failures, highlighting significant gaps in standard UIA API coverage—especially for third-party applications (e.g., LibreOffice) that do not strictly adhere to accessibility standards. Conversely, the OSWorld-W benchmark exhibited a higher incidence of Plan Errors, underscoring that tasks in this set frequently involve more complex workflows, necessitating deeper domain knowledge or advanced contextual reasoning capabilities beyond simple visual recognition.

These observations provide concrete evidence of specific system-level shortcomings, directly motivating the enhancements incorporated into the complete version of UFO2. The high frequency of Control Detection Failures validates our

Table 2. SR breakdown by application type on WAA and OSWorld-W.   

<table><tr><td rowspan="2">Agent</td><td rowspan="2">Model</td><td colspan="6">WAA</td><td colspan="2">OSWorld-W</td></tr><tr><td>Office</td><td>Web Browser</td><td>Windows System</td><td>Coding</td><td>Media &amp; Video</td><td>Windows Utilities</td><td>Office</td><td>Cross-App</td></tr><tr><td>UFO</td><td>GPT-4o</td><td>0.0%</td><td>23.3%</td><td>33.3%</td><td>29.2%</td><td>33.3%</td><td>8.3%</td><td>18.5%</td><td>4.5%</td></tr><tr><td>NAVI</td><td>GPT-4o</td><td>0.0%</td><td>20.0%</td><td>29.2%</td><td>9.1%</td><td>25.3%</td><td>0.0%</td><td>18.5%</td><td>0.0%</td></tr><tr><td>OmniAgent</td><td>GPT-4o</td><td>0.0%</td><td>27.3%</td><td>33.3%</td><td>27.3%</td><td>30.3%</td><td>8.3%</td><td>14.8%</td><td>0.0%</td></tr><tr><td>Agent S</td><td>GPT-4o</td><td>0.0%</td><td>13.3%</td><td>45.8%</td><td>29.2%</td><td>19.1%</td><td>22.2%</td><td>22.2%</td><td>0.0%</td></tr><tr><td>Operator</td><td>computer-use</td><td>7%</td><td>26.7%</td><td>29.2%</td><td>29.2%</td><td>28.6%</td><td>8.3%</td><td>22.2%</td><td>4.5%</td></tr><tr><td>UFO2-base</td><td>GPT-4o</td><td>2.3%</td><td>36.7%</td><td>29.2%</td><td>41.7%</td><td>33.3%</td><td>0.0%</td><td>22.2%</td><td>9.1%</td></tr><tr><td>UFO2-base</td><td>o1</td><td>2.3%</td><td>30.0%</td><td>37.5%</td><td>50.0%</td><td>33.3%</td><td>8.3%</td><td>22.2%</td><td>9.1%</td></tr><tr><td>UFO2</td><td>GPT-4o</td><td>4.7%</td><td>30.0%</td><td>41.7%</td><td>58.3%</td><td>33.3%</td><td>8.3%</td><td>44.4%</td><td>9.1%</td></tr><tr><td>UFO2</td><td>o1</td><td>4.7%</td><td>40.0%</td><td>45.8%</td><td>50.0%</td><td>38.1%</td><td>16.7%</td><td>51.9%</td><td>9.1%</td></tr></table>

Table 3. Comparison of SR and CRR across control detection mechanisms.   

<table><tr><td rowspan="2">Control Detector</td><td rowspan="2">Model</td><td colspan="2">WAA</td><td colspan="2">OSWorld-W</td></tr><tr><td>SR</td><td>CRR</td><td>SR</td><td>CRR</td></tr><tr><td>UIA</td><td>GPT-4o</td><td>23.4%</td><td>-</td><td>22.4%</td><td>-</td></tr><tr><td>OmniParser-v2</td><td>GPT-4o</td><td>26.6%</td><td>7.0%</td><td>14.3%</td><td>0%</td></tr><tr><td>Hybrid</td><td>GPT-4o</td><td>26.6%</td><td>9.9%</td><td>22.4%</td><td>12.5%</td></tr><tr><td>UIA</td><td>o1</td><td>25.3%</td><td>-</td><td>24.5%</td><td>-</td></tr><tr><td>OmniParser-v2</td><td>o1</td><td>20.8%</td><td>7.0%</td><td>14.3%</td><td>0%</td></tr><tr><td>Hybrid</td><td>o1</td><td>27.9%</td><td>9.9%</td><td>28.6%</td><td>25.0%</td></tr></table>

![](images/1c28c7ebb11809c3f2f6fe49a05e1a271ad611d1155bb8c5efa929978fa3a3bb.jpg)  
Figure 20. The number of detected controls of different approaches.

choice of adopting a hybrid GUI detection pipeline that supplements standard UIA data with advanced visual grounding techniques. Similarly, the prevalence of Plan Errors underscores the critical role of integrating richer external documentation, domain-specific knowledge bases, and applicationlevel APIs to strengthen task understanding and action inference. In the subsequent sections, we explicitly demonstrate how these incremental system-level improvements progressively mitigate each identified category of errors, thereby substantially boosting UFO2’s overall task completion effectiveness.

# 6.3 Evaluation on Hybrid Control Detection

As shown in Figure 19, a considerable fraction of failures arise from Control Detection Failures, where non-standard UI elements do not comply with UIA guidelines. To quantify the effectiveness of different detection strategies, we compare UIA-only, OmniParser-v2–only, and our hybrid method (Section 3.4). We introduce a Control Recovery Ratio (CRR) to measure how many UIA-only failures are “recovered” (i.e., become successful completions) under OmniParser or the hybrid approach.

Table 3 presents the results on both benchmarks, across multiple model configurations. The hybrid method consistently outperforms either UIA-only or OmniParser-only settings, raising the overall success rate and converting up to $9 . 8 6 \%$ of previously irrecoverable cases into completions. This gain highlights the complementary strengths of the two detection pipelines, as the hybrid approach bridges coverage gaps in UIA while avoiding OmniParser’s limitations in more standardized GUIs.

In Figure 20, we report the average number of controls detected from each source (UIA, OmniParser-v2, and the merged set) under the hybrid approach. Owing to differences in application coverage, the total number of detected controls is generally higher in OSWorld-W than in WAA. Notably, both UIA and OmniParser-v2 identify substantial subsets of controls, and after merging, $2 7 . 9 \%$ and $5 6 . 7 \%$ of OmniParser-v2 detections are discarded due to overlap with UIA. These observations indicate that OmniParser-v2 provides a valuable complement to UIA by recovering nonstandard or custom elements. At the same time, the merging step removes redundancies and prevents double-counting, ultimately reducing control detection failures in the hybrid scheme.

# 6.4 Effectiveness of GUI + API Integration

We now evaluate how unifying API-based actions with standard GUI interactions in the Puppeteer impacts performance

![](images/1b984fe98d0829742715e1eb7ddd2653bd13851c34ca00e729bfae183b2a24fe.jpg)  
Figure 21. A case study comparing the completion of the same task using GUI-only actions vs. GUI $^ +$ API actions.

Table 4. APIs supported across Office applications.   

<table><tr><td>API</td><td>Application</td><td>Description</td></tr><tr><td>select_text</td><td>Word</td><td>Select matched text in the document.</td></tr><tr><td>selectParagraph</td><td>Word</td><td>Select a paragraph in the document.</td></tr><tr><td>set_font</td><td>Word</td><td>Set the font size and style of selected text.</td></tr><tr><td>save_as</td><td>Word</td><td>Save the current document to a desired format.</td></tr><tr><td>insert_excel_table</td><td>Excel</td><td>Insert a table at the desired position.</td></tr><tr><td>select_table_range</td><td>Excel</td><td>Select a range within a table.</td></tr><tr><td>reorder_column</td><td>Excel</td><td>Reorder columns of a table.</td></tr><tr><td>save_as</td><td>Excel</td><td>Save the current sheet to a desired format.</td></tr><tr><td>setbackground_color</td><td>PowerPoint</td><td>Set the background color of slide(s).</td></tr><tr><td>save_as</td><td>PowerPoint</td><td>Save the current presentation to a desired format.</td></tr></table>

(Section 3.5). To do so, we focus on the 27 office-related tasks in OSWorld and manually develop 12 APIs for Word, Excel, and PowerPoint. These applications provide COM interfaces that facilitate the creation of custom functions, making them ideal exemplars for deeper OS- and application-level integration. Importantly, many of these operations would require

Table 5. Performance comparison of GUI-only vs. GUI $^ +$ API actions.   

<table><tr><td>Action</td><td>Model</td><td>SR</td><td>PRR</td><td>ERR</td><td>CRR</td><td>ACS</td></tr><tr><td>GUI-only</td><td>GPT-4o</td><td>16.3%</td><td>-</td><td>-</td><td>-</td><td>13.8</td></tr><tr><td>GUI+API</td><td>GPT-4o</td><td>22.4%</td><td>5.9%</td><td>14.3%</td><td>25.0%</td><td>12.9</td></tr><tr><td>GUI-only</td><td>o1</td><td>16.3%</td><td>-</td><td>-</td><td>-</td><td>16.0</td></tr><tr><td>GUI+API</td><td>o1</td><td>24.5%</td><td>17.7%</td><td>0.0%</td><td>12.5%</td><td>6.6</td></tr></table>

cumbersome multi-step GUI procedures but become straightforward single calls via these APIs (e.g., select paragraphs). Table 4 details the implemented APIs.

Table 5 compares (i) overall Success Rate (SR), (ii) Plan Error Recovery rate (PRR), (iii) Execution Error Recovery rate (ERR), (iv) Control Detection Failure Recovery rate (CRR), and (v) Average Completion Steps (ACS) for two configurations: GUI-only versus GUI + API. We calculate ACS on the subset of tasks that both configurations successfully complete, ensuring a fair comparison.

The results show that integrating API actions boosts SR for both GPT-4o $( + 6 . 1 \% )$ and o1 $( + 8 . 2 \% )$ , underscoring the effectiveness of mixing GUI and API interactions. Notably, GPT-4o benefits most from APIs in recovering from Control Detection Failures by circumventing unannotated GUI elements. In contrast, o1 more frequently addresses Plan Errors through API “shortcuts”, reflecting the model’s stronger reasoning capabilities and preference for concise solutions.

Table 6. Performance comparison with and without knowledge integration.   

<table><tr><td rowspan="2">Knowledge Enhancement</td><td rowspan="2">Model</td><td colspan="2">WAA</td><td colspan="2">OSWorld-W</td></tr><tr><td>SR</td><td>PRR</td><td>SR</td><td>PRR</td></tr><tr><td>None</td><td>GPT-4o</td><td>23.4%</td><td>-</td><td>22.4%</td><td>-</td></tr><tr><td>Help Document</td><td>GPT-4o</td><td>26.6%</td><td>10.34%</td><td>26.5%</td><td>11.8%</td></tr><tr><td>Self-Experience</td><td>GPT-4o</td><td>26.6%</td><td>13.79%</td><td>24.5%</td><td>11.8%</td></tr><tr><td>None</td><td>o1</td><td>25.3%</td><td>-</td><td>24.5%</td><td>-</td></tr><tr><td>Help Document</td><td>o1</td><td>27.9%</td><td>3.5%</td><td>28.5%</td><td>17.7%</td></tr><tr><td>Self-Experience</td><td>o1</td><td>20.8%</td><td>13.79%</td><td>26.5%</td><td>17.7%</td></tr></table>

Beyond higher success rates, GUI + API also reduces the effort required to complete tasks. $\mathrm { U F O ^ { 2 } }$ achieves a $6 . 5 \%$ step savings with GPT-4o and an impressive $5 8 . 5 \%$ reduction for o1 on identical tasks. The latter improvement stems from o1’s ability to strategically call API functions, bypassing multiple GUI-based steps. Overall, these findings confirm the advantages of mixing GUI automation with API calls, both in terms of robustness and efficiency, and showcase the importance of deep system integration for desktop automation.

Case Study. To illustrate how the GUI + API approach streamlines task execution, Figure 21 shows the completion trajectory for exporting an Excel file to CSV format in a case of OSWorld-W, using either GUI-only or GUI + API interactions. Although both configurations eventually succeed, the GUI-only setting requires five steps to open the Save dialog, select the file format, and confirm the action. In contrast, a single call to the save_as API completes the task immediately. Beyond improving efficiency, this one-step solution also reduces the risk of compounding errors across multiple GUI interactions—a clear demonstration of the advantages of deeper OS and application-level integration.

# 6.5 Continuous Knowledge Integration Evaluation

We next evaluate the impact of continuous knowledge integration (Section 3.6) on $\mathrm { U F O ^ { 2 } }$ ’s performance. Specifically, we augment $\mathrm { U F O ^ { 2 } }$ with external documentation and executionderived insights to dynamically improve its domain understanding without retraining. We create 34 help documents tailored to benchmark tasks, each containing precise stepby-step instructions, enabling $\mathrm { U F O ^ { 2 } }$ to retrieve the most relevant guidance (maximum of one per task) at runtime. Additionally, we implement an automated pipeline that summarizes successful execution trajectories—validated by our Task Evaluator and archives them into a retrievable knowledge database. For subsequent tasks, $\mathrm { U F O ^ { 2 } }$ dynamically retrieves up to three relevant past execution logs to guide task planning and execution. Given that knowledge integration primarily addresses failures arising from insufficient planning (Plan Errors), we employ the Plan Recovery Ratio (PRR) to measure the proportion of previously failed planning cases successfully resolved by integrating new knowledge.

Table 7. The SR and ACS comparison between single action and speculative multi-action mode.   

<table><tr><td rowspan="2">Action Execution</td><td rowspan="2">Model</td><td colspan="3">WAA</td><td colspan="3">OSWorld-W</td></tr><tr><td>SR</td><td>ACS</td><td>Success Subset</td><td>SR</td><td>ACS</td><td>Success Subset</td></tr><tr><td>Single</td><td>GPT-4o</td><td>23.4%</td><td>10.00</td><td rowspan="2">30</td><td>22.4%</td><td>13.30</td><td rowspan="2">10</td></tr><tr><td>Speculative</td><td>GPT-4o</td><td>23.4%</td><td>8.78</td><td>24.5%</td><td>7.40</td></tr><tr><td>Single</td><td>o1</td><td>25.3%</td><td>9.95</td><td rowspan="2">32</td><td>24.5%</td><td>6.80</td><td rowspan="2">10</td></tr><tr><td>Speculative</td><td>o1</td><td>24.7%</td><td>8.85</td><td>26.5%</td><td>3.30</td></tr></table>

Table 6 compares the overall SR and PRRs across two benchmarks, highlighting significant performance improvements attributable to knowledge integration. Both live helpdocument retrieval and self-experience summarization yield noticeable gains, reducing planning failures by up to $1 7 . 7 \%$ Notably, self-experience enhancements using the stronger model (o1) achieve consistent improvements across both benchmarks, underscoring the efficacy of leveraging prior successes for adaptive improvement. While help documents occasionally result in modest gains, their effectiveness depends on task complexity and document specificity.

These findings underscore the value of systematic knowledge integration, demonstrating that continuous augmentation of the agent’s knowledge base can substantially enhance its robustness, scalability, and adaptability in real-world deployments. Moreover, as $\mathrm { U F O ^ { 2 } }$ continues to accumulate execution experience and documentation over time, it inherently evolves toward higher reliability and improved autonomy, marking a clear path for ongoing enhancement in desktop automation.

# 6.6 Effectiveness of Speculative Multi-Action Execution

Next, we evaluate how speculative multi-action execution (Section 3.7) affects task completion rates and efficiency. Table 7 compares two modes for UFO2: generating and executing one action per inference (single-action) versus inferring multiple consecutive actions in one step (speculative multiaction). To ensure a fair comparison, we compute the Average Completion Steps (ACS) only on the subset of tasks that both modes complete successfully.

The results show that speculative multi-action execution retains a comparable Success Rate (SR) to single-action mode while notably cutting the average steps—by up to $1 0 \%$ on WAA and an impressive $5 1 . 5 \%$ on OSWorld-W. Because each step requires an LLM call, reducing the number of steps significantly lowers both latency and cost. This finding confirms that speculative multi-action planning enhances efficiency without compromising reliability, further highlighting UFO2’s ability to optimize resource utilization in practical desktop automation.

Case Study. Figure 22 illustrates how speculative multiaction execution operates in practice. When the user requests

![](images/b497d18aadd8d6676cb5de086ac656882395f30ea6a447effa2fb0e5bf334781.jpg)  
Figure 22. A case study of the successful speculative multiaction execution.

![](images/8ee488db35a0f3a204f64e9007e016372799e62ba415a29ec10ce38607842c1b.jpg)  
Figure 23. Comparison of Operator vs. $\mathrm { U F O ^ { 2 } + }$ Operator on WAA and OSWorld-W.

that $\mathrm { U F O ^ { 2 } }$ center-align a heading in a Word document, the sequence of steps would typically require selecting the heading text and then clicking the Center icon. These actions that are sequentially dependent but do not interfere with each other. Instead of treating each action as a separate LLM inference, $\mathrm { U F O ^ { 2 } }$ predicts both actions in a single step, leveraging speculative multi-action planning. Consequently, it completes the task with just one LLM call, significantly enhancing efficiency while maintaining accuracy.

# 6.7 Operator as an AppAgent

To demonstrate the “Everything-as-an-AppAgent” capability (Section 5.3), we conducted an experiment where UFO2’s HostAgent orchestrator uses only Operator as the AppAgent. In other words, all native AppAgents were disabled, leaving Operator to accept subtasks and communicate via the standard UFO2 messaging protocol. The only adjustment to Operator’s perception layer was restricting it to screenshots of the selected application window, rather than the full desktop.

![](images/ed7b1493ab335112c04ca49e9c26d8293ab335ad5fd9548dc173d65666167dab.jpg)  
Figure 24. Average time cost per-stage of a single execution step.

Figure 23 shows that $\mathrm { { U F O ^ { 2 } \Sigma + } }$ Operator achieves higher success rates than running Operator alone, particularly on WAA $2 6 . 0 \%$ vs. $2 0 . 8 \%$ ). We attribute these gains to three key factors. First, the HostAgent breaks down complex user instructions into clearer, single-application subtasks, reducing ambiguity. Second, HostAgent messages include additional tips that improve Operator’s decision making. Finally, limiting Operator’s view to a single, active application window reduces visual noise and simplifies control detection. Taken together, these results underscore the benefits of UFO2’s multiagent design, while demonstrating how “Everythingas-an-AppAgent” can elevate the performance of an existing CUA.

# 6.8 Efficiency Analysis

To comprehensively understand the performance characteristics of $\mathrm { U F O ^ { 2 } }$ , we conducted detailed profiling of task execution efficiency, focusing specifically on step count and latency.

Step Count Profiling. Table 8 summarizes the average number of execution steps performed by the HostAgent and AppAgents across both benchmark suites. The steps reported are computed on tasks that were successfully completed across all model configurations, ensuring fair comparisons. Two key insights emerge:

First, the fully integrated $\mathrm { U F O ^ { 2 } }$ configuration consistently reduces the average number of steps required compared to the baseline (UFO2-base), achieving reductions of up to $5 0 \%$ This substantial efficiency gain demonstrates how deep OS integration, specifically the hybrid GUI–API action orchestration and advanced control detection strategies, significantly streamline execution paths.

Second, utilizing a more powerful reasoning model (e.g.,, o1 versus GPT-4o) further reduces step counts, indicating that enhanced reasoning capability enables the agent to identify and exploit more efficient action sequences. For instance, stronger models can better leverage direct API interactions or avoid unnecessary intermediate GUI interactions. This underscores the complementary role of both robust system integration and advanced LLM reasoning in minimizing execution overhead.

Table 8. Step count statistic for $\mathrm { U F O ^ { 2 } }$ .   

<table><tr><td rowspan="2">Agent</td><td rowspan="2">Model</td><td colspan="4">WAA</td><td colspan="4">OSWorld-W</td></tr><tr><td>HOSTAGENT</td><td>APPAGENT</td><td>Total</td><td>Success Subset</td><td>HOSTAGENT</td><td>APPAGENT</td><td>Total</td><td>Success Subset</td></tr><tr><td>UFO2-base</td><td>GPT-4o</td><td>2.21</td><td>8.11</td><td>10.32</td><td rowspan="2">31</td><td>1.80</td><td>10.80</td><td>12.60</td><td rowspan="2">7</td></tr><tr><td>UFO2</td><td>GPT-4o</td><td>2.32</td><td>7.89</td><td>10.21</td><td>2.80</td><td>7.20</td><td>10.00</td></tr><tr><td>UFO2-base</td><td>o1</td><td>2.14</td><td>7.00</td><td>9.14</td><td rowspan="2">34</td><td>2.50</td><td>8.83</td><td>11.33</td><td rowspan="2">8</td></tr><tr><td>UFO2</td><td>o1</td><td>2.00</td><td>4.05</td><td>6.05</td><td>2.00</td><td>3.50</td><td>5.50</td></tr></table>

Latency Breakdown. Figure 24 provides a detailed breakdown of average latency per execution step in UFO2, separated into five key phases: (i) Screenshot Capture, (ii) Control Detection via UIA APIs (and) OmniParser-v2, (iii) Prompt Preparation, including retrieval of relevant help documents and historical execution experiences, (iv) LLM Inference, and (v) Action Execution on target applications.

Across all configurations, the LLM inference phase dominates total latency, averaging around 10 seconds per inference. This bottleneck highlights a clear opportunity for optimization by deploying smaller, specialized models or employing more powerful inference hardware—strategies that remain viable but are beyond our current evaluation scope.

Excluding LLM inference overhead, the baseline system (UFO2-base) achieves highly efficient execution, incurring around 10 seconds per step on average. In contrast, the fully integrated $\mathrm { U F O ^ { 2 } }$ incurs only an additional 1 second per step for its hybrid control detection pipeline, largely due to OmniParser-v2 visual parsing. This added overhead represents a deliberate trade-off, significantly enhancing the robustness and accuracy of GUI control detection at a modest latency cost.

Taken together, these results indicate that the substantial reduction in total steps required per task ensures overall task completion times remain practical (approximately 1 minute per task). These profiling insights reinforce that UFO2’s comprehensive system-level integration balances latency, accuracy, and efficiency, offering a scalable and performant solution for real-world desktop automation.

# 6.9 Model Ablation

Table 25 compares the performance of $\mathrm { U F O ^ { 2 } }$ and $\mathrm { U F O ^ { 2 } }$ -base across four large language models. GPT-4V and GPT-4o generate direct answers without exposing an explicit CoT, whereas Gemini 2.0 (Flash Thinking) and o1 embed reasoning steps internally before producing final outputs. Overall, models with built-in reasoning typically achieve higher success rates (SR), highlighting the value of more deliberative or CoT-driven processes in desktop automation [52].

This result underscores a promising direction for CUAs: fine-tuning advanced reasoning models specifically for desktop automation tasks. By allowing agents to formulate and refine multi-step plans—especially when integrated with

![](images/5255b3ce79e774904464b7b44f9de51173e5cfcd0b0ce77f582647082ed65a4a.jpg)  
Figure 25. Comparison of different LLMs used in $\mathrm { U F O ^ { 2 } }$ and $\mathrm { U F O ^ { 2 } }$ -base on WAA and OSWorld-W.

deeper OS-level signals—UFO2 can address complex or ambiguous situations more reliably. As LLM-based reasoning continues to mature, we expect further gains in both accuracy and generality from UFO2’s model-agnostic design.

# 7 Discussion & Future Work

Latency and Responsiveness. UFO2 currently invokes LLM inference at each decision step, incurring a latency typically ranging from several seconds to tens of seconds per action. Despite various engineering optimizations, complex tasks comprising multiple sequential actions can accumulate to execution times of 1–2 minutes, which remains acceptable but still inferior to skilled human performance. To alleviate user-perceived delay, we introduced the Picture-in-Picture (PiP) interface, enabling $\mathrm { U F O ^ { 2 } }$ to execute tasks unobtrusively within an isolated virtual desktop, thus substantially reducing user inconvenience during longer-running automations. In future work, we aim to further lower latency by investigating the deployment of specialized, lightweight Large Action Models (LAMs) [13], optimized for task-specific inference to enhance both responsiveness and scalability.

Closing the Gap with Human-Level Performance. Our comprehensive evaluations indicate that $\mathrm { U F O ^ { 2 } }$ , while robust and effective, does not yet consistently achieve human-level performance across all Windows applications. Bridging this gap will necessitate advances primarily along two critical dimensions. First, enhancing foundational visual-language

models through fine-tuning on extensive, diverse GUI interaction datasets will significantly improve agents’ capabilities and generalization across varied applications. Second, tighter integration with OS-level APIs, native application interfaces, and comprehensive, structured documentation sources will deepen contextual understanding and bolster execution reliability. Given $\mathrm { U F O ^ { 2 } }$ ’s modular architecture, these enhancements can be incrementally adopted, continuously refining performance towards human-equivalent proficiency across diverse application scenarios.

Generalization Across Operating Systems. While UFO2 targets Windows OS due to its widespread market adoption (over $7 0 \%$ market share2), the modular, layered system design facilitates straightforward adaptation to other desktop platforms, such as Linux and macOS. The core approach—leveraging accessibility frameworks like Windows UI Automation (UIA)—has direct analogs on Linux (AT-SPI [53]) and macOS (Accessibility API [54]). Consequently, the existing design principles, agent decomposition strategy, and unified GUI-API orchestration model generalize naturally, enabling rapid, platform-specific customizations. Exploring cross-platform deployments will be an important area of future work, potentially laying the groundwork for a unified ecosystem of desktop automation solutions spanning diverse operating environments.

# 8 Related Work

Integrating LLMs into OS represents a growing, yet nascent area of research. In this section, we discuss prior work that intersects with our research on system-level integration of multimodal LLM-based desktop automation agents.

# 8.1 Computer-Using Agents (CUAs)

Recent advancements in multimodal LLMs have significantly accelerated the development of Computer-Using Agents (CUAs), which automate desktop workflows by simulating GUI interactions at the OS level. Early pioneering systems, such as UFO [10], leveraged multimodal models (e.g., GPT-4V [20]) alongside UIA APIs to interpret graphical interfaces and execute complex tasks via natural language instructions. UFO notably introduced a multi-agent architecture, enhancing the reliability and capability of CUAs to handle cross-application and long-term workflows.

Subsequent efforts have focused primarily on refining underlying multimodal models and extending platform capabilities. For example, CogAgent [55], built upon the visionlanguage model CogVLM [56], specialized in GUI understanding across multiple platforms (PC, web, Android), representing one of the earliest dedicated multimodal CUAs. Industry interest has similarly accelerated with Anthropic’s Claude-3.5 (Computer Use) [11], an agent relying entirely

on screenshot-based GUI interactions, and OpenAI’s Operator [12], which significantly improved desktop automation performance through advanced multimodal reasoning.

However, these existing CUAs remain largely prototype demonstrations, often lacking deep integration with the OS and native application capabilities. In contrast, our work in $\mathrm { U F O ^ { 2 } }$ directly addresses these fundamental system-level limitations through a modular AgentOS architecture, deep OS and API integration, hybrid GUI detection, and a nondisruptive execution model, bridging the gap between conceptual CUAs and practical desktop automation.

# 8.2 LLMs for Operating Systems

Another promising research direction involves embedding LLMs directly within OS architectures, aiming to substantially enhance automation, adaptability, and usability. Ge et al., first proposed the conceptual framework AIOS [57], positioning an LLM at the center of OS design to orchestrate high-level user interactions and automated decision-making. In their vision, agents resemble OS applications, each exposing specialized capabilities accessible via natural language, effectively enabling users to "program" their OS intuitively.

Building on this conceptual foundation, Mei et al., [58] realized AIOS as a concrete prototype, encapsulating LLM interactions and tool APIs within a privileged OS kernel. This design provides core OS functionalities such as process scheduling, memory management, I/O handling, and access control, leveraging LLMs to simplify agent development through a dedicated SDK. Rama et al., [59] extended this paradigm, introducing semantic file management capabilities directly within traditional OS environments through AIOS-based agents, further demonstrating practical systemlevel integration.

Complementing these high-level OS integrations, AutoOS [60] applied LLMs for automatic tuning of kernel-level parameters in Linux, achieving substantial efficiency gains through autonomous exploration and optimization. This highlights another dimension where LLM integration can directly enhance core system performance and management.

Collectively, these research efforts illustrate an emerging paradigm shift where LLMs become integral components of operating systems, enabling powerful automation, enhanced user interaction, and adaptive system behavior. Our work with $\mathrm { U F O ^ { 2 } }$ extends this line of research specifically to desktop automation, offering a deeply integrated, scalable, and practical AgentOS that leverages multimodal LLMs in conjunction with robust OS-level mechanisms.

# 9 Conclusion

We introduced $\mathbf { U F O ^ { 2 } }$ , a practical, OS-integrated Windows desktop automation AgentOS that transforms CUAs from conceptual prototypes into robust, user-oriented solutions.

Unlike prior CUAs, $\mathrm { U F O ^ { 2 } }$ leverages deep system-level integration through a modular, multiagent architecture consisting of a centralized HostAgent and application-specialized AppAgents. Each AppAgent seamlessly combines GUI interactions with native APIs and continually integrates applicationspecific knowledge, substantially improving reliability and execution efficiency. $\mathrm { U F O ^ { 2 } }$ can operate on a PiP virtual desktop interface further enhances usability, enabling concurrent user-agent workflows without interference.

Our comprehensive evaluation across over 20 real-world Windows applications demonstrated that $\mathrm { U F O ^ { 2 } }$ achieves significant improvements in robustness, accuracy, and scalability compared to state-of-the-art CUAs. Notably, by coupling our integrated framework with robust OS-level features, even less specialized foundation models (e.g., GPT-4o) surpass specialized CUAs such as Operator.

# References

[1] UiPath. Uipath: Automation platform, 2025. Accessed: 2025-03-18.   
[2] Automation Anywhere. Automation anywhere: Automation 360 platform, 2025. Accessed: 2025-03-18.   
[3] Microsoft. Microsoft power automate, 2025. Accessed: 2025-03-18.   
[4] Peter Hofmann, Caroline Samp, and Nils Urbach. Robotic process automation. Electronic markets, 30(1):99–106, 2020.   
[5] Somayya Madakam, Rajesh M Holmukhe, and Durgesh Kumar Jaiswal. The future digital work force: robotic process automation (rpa). JISTEM-Journal of Information Systems and Technology Management, 16:e201916001, 2019.   
[6] Dhanya Pramod. Robotic process automation for industry: adoption status, benefits, challenges and research agenda. Benchmarking: an international journal, 29(5):1562–1586, 2022.   
[7] Chaoyun Zhang, Shilin He, Jiaxu Qian, Bowen Li, Liqun Li, Si Qin, Yu Kang, Minghua Ma, Guyue Liu, Qingwei Lin, et al. Large language model-brained gui agents: A survey. arXiv preprint arXiv:2411.18279, 2024.   
[8] Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 1(2), 2023.   
[9] Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan Su, Chenhui Chu, and Dong Yu. Mm-llms: Recent advances in multimodal large language models. arXiv preprint arXiv:2401.13601, 2024.   
[10] Chaoyun Zhang, Liqun Li, Shilin He, Xu Zhang, Bo Qiao, Si Qin, Minghua Ma, Yu Kang, Qingwei Lin, Saravan Rajmohan, et al. Ufo: A ui-focused agent for windows os interaction. arXiv preprint arXiv:2402.07939, 2024.   
[11] Anthropic. Introducing computer use, a new claude 3.5 sonnet, and claude 3.5 haiku, 2024. Accessed: 2024-10-26.   
[12] OpenAI. Computer-using agent: Introducing a universal interface for ai to interact with the digital world. 2025.   
[13] Lu Wang, Fangkai Yang, Chaoyun Zhang, Junting Lu, Jiaxu Qian, Shilin He, Pu Zhao, Bo Qiao, Ray Huang, Si Qin, et al. Large action models: From inception to implementation. arXiv preprint arXiv:2412.10047, 2024.   
[14] Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. Uitars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326, 2025.   
[15] Jiani Zheng, Lu Wang, Fangkai Yang, Chaoyun Zhang, Lingrui Mei, Wenjie Yin, Qingwei Lin, Dongmei Zhang, Saravan Rajmohan, and Qi Zhang. Vem: Environment-free exploration for training gui agent

with value environment model. arXiv preprint arXiv:2502.18906, 2025.   
[16] Runliang Niu, Jindong Li, Shiqi Wang, Yali Fu, Xiyu Hu, Xueyuan Leng, He Kong, Yi Chang, and Qi Wang. Screenagent: a vision language model-driven computer control agent. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, pages 6433–6441, 2024.   
[17] Chaoyun Zhang, Shilin He, Liqun Li, Si Qin, Yu Kang, Qingwei Lin, and Dongmei Zhang. Api agents vs. gui agents: Divergence and convergence. arXiv preprint arXiv:2503.11069, 2025.   
[18] Yadong Lu, Jianwei Yang, Yelong Shen, and Ahmed Awadallah. Omniparser for pure vision based gui agent. arXiv preprint arXiv:2408.00203, 2024.   
[19] Julia Siderska, Lili Aunimo, Thomas Süße, John von Stamm, Damian Kedziora, and Suraya Nabilah Binti Mohd Aini. Towards intelligent automation (ia): literature review on the evolution of robotic process automation (rpa), its challenges, and future trends. Engineering Management in Production and Services, 15(4), 2023.   
[20] Zhengyuan Yang, Linjie Li, Kevin Lin, Jianfeng Wang, Chung-Ching Lin, Zicheng Liu, and Lijuan Wang. The dawn of lmms: Preliminary explorations with gpt-4v (ision). arXiv preprint arXiv:2309.17421, 9(1):1, 2023.   
[21] David N Gray, John Hotchkiss, Seth LaForge, Andrew Shalit, and Toby Weinberg. Modern languages and microsoft’s component object model. Communications of the ACM, 41(5):55–65, 1998.   
[22] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR), 2023.   
[23] Bo Qiao, Liqun Li, Xu Zhang, Shilin He, Yu Kang, Chaoyun Zhang, Fangkai Yang, Hang Dong, Jue Zhang, Lu Wang, et al. Taskweaver: A code-first agent framework. arXiv preprint arXiv:2311.17541, 2023.   
[24] Shanshan Han, Qifan Zhang, Yuhang Yao, Weizhao Jin, Zhaozhuo Xu, and Chaoyang He. Llm multi-agent systems: Challenges and open problems. arXiv preprint arXiv:2402.03578, 2024.   
[25] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6):186345, 2024.   
[26] Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model based agents. arXiv preprint arXiv:2404.13501, 2024.   
[27] Jianwei Yang, Hao Zhang, Feng Li, Xueyan Zou, Chunyuan Li, and Jianfeng Gao. Set-of-mark prompting unleashes extraordinary visual grounding in gpt-4v. arXiv preprint arXiv:2310.11441, 2023.   
[28] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.   
[29] Ruomeng Ding, Chaoyun Zhang, Lu Wang, Yong Xu, Minghua Ma, Wei Zhang, Si Qin, Saravan Rajmohan, Qingwei Lin, and Dongmei Zhang. Everything of thoughts: Defying the law of penrose triangle for thought generation. In Findings of the Association for Computational Linguistics ACL 2024, pages 1638–1662, 2024.   
[30] Kanzhi Cheng, Qiushi Sun, Yougang Chu, Fangzhi Xu, Li YanTao, Jianbing Zhang, and Zhiyong Wu. Seeclick: Harnessing gui grounding for advanced visual gui agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9313–9332, 2024.   
[31] Boyu Gou, Ruohan Wang, Boyuan Zheng, Yanan Xie, Cheng Chang, Yiheng Shu, Huan Sun, and Yu Su. Navigating the digital world as humans do: Universal visual grounding for gui agents. arXiv preprint arXiv:2410.05243, 2024.

[32] Dillon Reis, Jordan Kupec, Jacqueline Hong, and Ahmad Daoudi. Real-time flying object detection with yolov8. arXiv preprint arXiv:2305.09972, 2023.   
[33] Bin Xiao, Haiping Wu, Weijian Xu, Xiyang Dai, Houdong Hu, Yumao Lu, Michael Zeng, Ce Liu, and Lu Yuan. Florence-2: Advancing a unified representation for a variety of vision tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4818–4829, 2024.   
[34] Yueqi Song, Frank Xu, Shuyan Zhou, and Graham Neubig. Beyond browsing: Api-based web agents. arXiv preprint arXiv:2410.16464, 2024.   
[35] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, et al. A survey on in-context learning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 1107–1128, 2024.   
[36] Sewon Min, Xinxi Lyu, Ari Holtzman, Mikel Artetxe, Mike Lewis, Hannaneh Hajishirzi, and Luke Zettlemoyer. Rethinking the role of demonstrations: What makes in-context learning work? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11048–11064, 2022.   
[37] Chaoyun Zhang, Zicheng Ma, Yuhao Wu, Shilin He, Si Qin, Minghua Ma, Xiaoting Qin, Yu Kang, Yuyi Liang, Xiaoyu Gou, et al. Allhands: Ask me anything on large-scale verbatim feedback via large language models. arXiv preprint arXiv:2403.15157, 2024.   
[38] Man Luo, Xin Xu, Yue Liu, Panupong Pasupat, and Mehran Kazemi. Incontext learning with retrieved demonstrations for language models: A survey. Transactions on Machine Learning Research.   
[39] Siyuan Wang, Zhuohan Long, Zhihao Fan, Xuan-Jing Huang, and Zhongyu Wei. Benchmark self-evolving: A multi-agent framework for dynamic llm evaluation. In Proceedings of the 31st International Conference on Computational Linguistics, pages 3310–3328, 2025.   
[40] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.   
[41] Jun Liu, Chaoyun Zhang, Jiaxu Qian, Minghua Ma, Si Qin, Chetan Bansal, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. Large language models can deliver accurate and interpretable time series anomaly detection. arXiv preprint arXiv:2405.15370, 2024.   
[42] Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Haofen Wang, and Haofen Wang. Retrievalaugmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997, 2, 2023.   
[43] Yuxuan Jiang, Chaoyun Zhang, Shilin He, Zhihao Yang, Minghua Ma, Si Qin, Yu Kang, Yingnong Dang, Saravan Rajmohan, Qingwei Lin, et al. Xpert: Empowering incident management with query recommendations via large language models. In Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, pages 1–13, 2024.   
[44] Karissa Miller and Mahmoud Pegah. Virtualization: virtually at the desktop. In Proceedings of the 35th annual ACM SIGUCCS fall conference, pages 255–260, 2007.   
[45] Aditya Venkataraman and Kishore Kumar Jagadeesha. Evaluation of inter-process communication mechanisms. Architecture, 86(64), 2015.   
[46] Nils Reimers and Iryna Gurevych. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, 2019.   
[47] Ido Levy, Ben Wiesel, Sami Marreed, Alon Oved, Avi Yaeli, and Segev Shlomov. St-webagentbench: A benchmark for evaluating safety and trustworthiness in web agents. arXiv preprint arXiv:2410.06703, 2024.   
[48] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. Mllmas-a-judge: Assessing multimodal llm-as-a-judge with vision-language

benchmark. In Forty-first International Conference on Machine Learning, 2024.   
[49] Rogerio Bonatti, Dan Zhao, Dillon Dupont, Sara Abdali, Yinheng Li, Yadong Lu, Justin Wagle, Kazuhito Koishida, Arthur Bucker, Lawrence Keunho Jang, et al. Windows agent arena: Evaluating multimodal os agents at scale. In NeurIPS 2024 Workshop on Open-World Agents.   
[50] Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Jing Hua Toh, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, et al. Osworld: Benchmarking multimodal agents for open-ended tasks in real computer environments. Advances in Neural Information Processing Systems, 37:52040–52094, 2024.   
[51] Saaket Agashe, Jiuzhou Han, Shuyu Gan, Jiachen Yang, Ang Li, and Xin Eric Wang. Agent s: An open agentic framework that uses computers like a human. In The Thirteenth International Conference on Learning Representations.   
[52] Zhengxi Lu, Yuxiang Chai, Yaxuan Guo, Xi Yin, Liang Liu, Hao Wang, Guanjing Xiong, and Hongsheng Li. Ui-r1: Enhancing action prediction of gui agents by reinforcement learning, 2025.   
[53] Linux From Scratch. At-spi - assistive technology service provider interface. https://www.linuxfromscratch.org/blfs/view/5.1/gnome/atspi.html. Accessed: 2025-03-31.   
[54] Apple Inc. Accessibility api. https://developer.apple.com/ documentation/accessibility/accessibility-api, 2025. Accessed: 2025- 03-31.   
[55] Wenyi Hong, Weihan Wang, Qingsong Lv, Jiazheng Xu, Wenmeng Yu, Junhui Ji, Yan Wang, Zihan Wang, Yuxiao Dong, Ming Ding, et al. Cogagent: A visual language model for gui agents. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14281–14290, 2024.   
[56] Weihan Wang, Qingsong Lv, Wenmeng Yu, Wenyi Hong, Ji Qi, Yan Wang, Junhui Ji, Zhuoyi Yang, Lei Zhao, Song XiXuan, et al. Cogvlm: Visual expert for pretrained language models. Advances in Neural Information Processing Systems, 37:121475–121499, 2024.   
[57] Yingqiang Ge, Yujie Ren, Wenyue Hua, Shuyuan Xu, Juntao Tan, and Yongfeng Zhang. Llm as os, agents as apps: Envisioning aios, agents and the aios-agent ecosystem. arXiv preprint arXiv:2312.03815, 2023.   
[58] Kai Mei, Xi Zhu, Wujiang Xu, Wenyue Hua, Mingyu Jin, Zelong Li, Shuyuan Xu, Ruosong Ye, Yingqiang Ge, and Yongfeng Zhang. Aios: Llm agent operating system. arXiv preprint arXiv:2403.16971, 2024.   
[59] Balaji Rama, Kai Mei, and Yongfeng Zhang. Cerebrum (aios sdk): A platform for agent development, deployment, distribution, and discovery, 2025.   
[60] Huilai Chen, Yuanbo Wen, Limin Cheng, Shouxu Kuang, Yumeng Liu, Weijia Li, Ling Li, Rui Zhang, Xinkai Song, Wei Li, et al. Autoos: make your os more powerful by exploiting large language models. In Forty-first International Conference on Machine Learning, 2024.
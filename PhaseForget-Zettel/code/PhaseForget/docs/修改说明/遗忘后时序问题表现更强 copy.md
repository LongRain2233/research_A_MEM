**问题背景**
本实验是基于A-mem的源码进行修改的实验，在源实验的基础上增加了遗忘机制。
源实验代码:D:\research\research_A_MEM\A-mem-ollma\A-mem  
源实验论文D:\research\research_A_MEM\PhaseForget-Zettel\code\PhaseForget\docs\A-MEM Agentic Memory for LLM Agents.md
新实验代码:D:\research\research_A_MEM\PhaseForget-Zettel\code\PhaseForget\src\phaseforget
**现在的问题**
我想知道当开启了遗忘机制之后，因为会把不需要的记忆节点删除了，本实验回答问题逻辑是什么样，会检索来自哪的记忆，和源实验的区别是什么。源实验应该是从相关记忆节点提取记忆构成上下文喂给大模型来回答。
我现在想知道新实验会怎么做，会结合抽象出来的高层记忆进行回答吗，还有新实验找相似节点的逻辑是什么，会考虑到抽象出来的更高层的记忆吗

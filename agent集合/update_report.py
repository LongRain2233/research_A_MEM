# -*- coding: utf-8 -*-
with open(r'D:\文档\论文\agent集合\Agent记忆领域研究方向分析报告.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 更新1.2文献总览说明行
content = content.replace(
    '与 **Agent 记忆（Memory）** 直接相关的核心文献有 **27 篇**（含 2026-02 补充的 SimpleMem）',
    '与 **Agent 记忆（Memory）** 直接相关的核心文献有 **28 篇**（含 2026-02 补充的 SimpleMem 和 O-Mem）'
)

# 2. 更新1.3 记忆存储结构设计 (8篇 -> 9篇)
content = content.replace(
    '8\u7bc7 | A-MEM, H-MEM, FinMem, LightMem, MIRIX, RET-LLM, Zep, EverMemOS |',
    '9\u7bc7 | A-MEM, H-MEM, FinMem, LightMem, MIRIX, RET-LLM, Zep, EverMemOS, **O-Mem** |'
)

# 3. 更新1.3 记忆检索与访问机制 - 已经改为9篇，更新代表文献
content = content.replace(
    '9\u7bc7 | H-MEM, AriGraph, Zep, MAICC, M3-Agent, RMM, **SimpleMem** |',
    '9\u7bc7 | H-MEM, AriGraph, Zep, MAICC, M3-Agent, RMM, **SimpleMem**, **O-Mem**\uff08\u5e76\u884c\u4e09\u8def+IDF\u5173\u952e\u8bcd\uff09 |'
)

# 4. 更新1.3 记忆构建与写入策略 (5篇 -> 6篇)
content = content.replace(
    '5\u7bc7 | PREMem, Mem-\u03b1, A-MEM, EverMemOS, **SimpleMem** |',
    '6\u7bc7 | PREMem, Mem-\u03b1, A-MEM, EverMemOS, **SimpleMem**, **O-Mem**\uff08Add/Ignore/Update+\u8fd1\u90bb\u56fe\u805a\u7c7b\uff09 |'
)

# 5. 在记忆构建与写入策略行后插入新的"个性化记忆"行
old_line = '| **\u8bb0\u5fc6\u66f4\u65b0\u4e0e\u7ef4\u62a4** | 4\u7bc7'
new_insertion = '| **\u4e2a\u6027\u5316\u8bb0\u5fc6\u4e0e\u4e3b\u52a8\u7528\u6237\u5efa\u6a21** | 2\u7bc7 | **O-Mem**\uff08\u4e3b\u52a8\u7528\u6237\u753b\u50cf+\u4e09\u7ec4\u4ef6\u67b6\u6784\uff09, PREMem\uff08\u4e2a\u6027\u5316\u5bf9\u8bdd\u8bb0\u5fc6\uff09 | \u2605\u2605\u2605 \u4e2d\u7b49 | \u2605\u2605 \u8f83\u65b0 |\n|| **\u8bb0\u5fc6\u66f4\u65b0\u4e0e\u7ef4\u62a4** | 4\u7bc7'
content = content.replace(old_line, new_insertion)

# 6. 更新记忆可信度与鲁棒性 0篇描述
content = content.replace(
    '**\u8bb0\u5fc6\u53ef\u4fe1\u5ea6\u4e0e\u9c81\u68d2\u6027** | 0\u7bc7 | \u2014 |',
    '**\u8bb0\u5fc6\u53ef\u4fe1\u5ea6\u4e0e\u9c81\u68d2\u6027** | 0\u7bc7 | \u2014\uff08O-Mem\u660e\u786e\u6307\u51faPersona Memory\u5e7b\u89c9\u98ce\u9669\u672a\u89e3\u51b3\uff09 |'
)

with open(r'D:\文档\论文\agent集合\Agent记忆领域研究方向分析报告.md', 'w', encoding='utf-8') as f:
    f.write(content)

print("Done! Checking changes...")
with open(r'D:\文档\论文\agent集合\Agent记忆领域研究方向分析报告.md', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines[10:15], start=11):
    print(f"L{i}: {line.rstrip()[:100]}")
print("...")
for i, line in enumerate(lines[49:75], start=50):
    print(f"L{i}: {line.rstrip()[:120]}")

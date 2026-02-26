import os
import sys
import re
import fitz

sys.stdout.reconfigure(encoding='utf-8')

# 已下载的论文
papers_dir = r"D:\research\research_A_MEM\papers_533"
downloaded_files = sorted([f for f in os.listdir(papers_dir) if f.endswith('.pdf')])

print("=" * 70)
print("已下载的论文列表 (共 {} 篇)".format(len(downloaded_files)))
print("=" * 70)
for i, f in enumerate(downloaded_files, 1):
    size = os.path.getsize(os.path.join(papers_dir, f)) // 1024
    print(f"{i:2d}. {f} ({size} KB)")

# 从PDF中提取5.3.3节引用的论文
doc = fitz.open(r'D:\research\research_A_MEM\MemoryintheAgeofAIAgents.pdf')
full_text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    full_text += page.get_text()
doc.close()

# 找到5.3.3节的实际内容 - 搜索包含大量引用的部分
# 5.3.3节标题后面应该有实际内容
section_start = full_text.find("Retrieval Strategies")
print(f"\n找到'Retrieval Strategies'在位置: {section_start}")

# 获取该位置附近的内容
if section_start != -1:
    # 获取前后文
    context = full_text[max(0, section_start-200):min(len(full_text), section_start+20000)]
    print(f"\n上下文长度: {len(context)} 字符")
    print("\n前500字符:")
    print(context[:500])
    
    # 在这个上下文中搜索引用
    citations = re.findall(r'\(([A-Z][a-z]+ et al\., \d{4}[a-z]?)\)', context)
    citations += re.findall(r'\(([A-Z][a-z]+ and [A-Z][a-z]+, \d{4}[a-z]?)\)', context)
    citations += re.findall(r'\(([A-Z][a-z]+, \d{4}[a-z]?)\)', context)
    
    unique_citations = sorted(set(citations))
    print(f"\n在Retrieval Strategies附近找到 {len(unique_citations)} 个引用:")
    for i, cite in enumerate(unique_citations, 1):
        print(f"  {i:2d}. {cite}")

# 映射到已下载的论文
print("\n" + "=" * 70)
print("论文对应检查")
print("=" * 70)

# 定义5.3.3节引用的论文 (根据之前的分析)
citation_map = {
    "Tang et al., 2025d": ("Agent KB", "Tang"),
    "Wang et al., 2025p": ("Mem-α", "MemAlpha"),
    "Pan et al., 2025": ("SeCom", "SeCom"),
    "Reimers and Gurevych, 2019": ("Sentence-BERT", "SentenceBERT"),
    "Radford et al., 2021": ("CLIP", "CLIP"),
    "Lewis et al., 2020": ("RAG", "RAG"),
    "Anokhin et al., 2024": ("AriGraph", "AriGraph"),
    "Wang et al., 2024l": ("EMG-RAG", "EMGRAG"),
    "Chhikara et al., 2025": ("Mem0/Mem0g", "Mem0"),
    "Wu et al., 2025h": ("SGMem", "SGMem"),
    "Gutierrez et al., 2024": ("HippoRAG", "HippoRAG"),
    "Li et al., 2025g": ("CAM", None),  # 未找到
    "Lei et al., 2025": ("D-SMART", "DSMART"),
    "Rasmussen et al., 2025": ("Zep", "Zep"),
    "Tan et al., 2025b": ("MemoTime", "MemoTime"),
    "Tay et al., 2022": ("DSI/GenRet", "DSI"),
    "Wang and Chen, 2025": ("MIRIX", "MIRIX"),
    "Wang et al., 2022b": ("NCI", "NCI"),
    "Chatterjee and Agarwal, 2025": ("Semantic Anchoring", "SemanticAnchoring"),
    "Kaiya et al., 2023": ("Lyfe Agents", "LyfeAgents"),
    "Jiang et al., 2025c": ("MAICC", "MAICC"),
    "Ward, 2025": ("MemoriesDB", "MemoriesDB"),
}

found = []
missing = []

for cite, (name, keyword) in citation_map.items():
    if keyword is None:
        missing.append((cite, name))
        continue
    # 检查是否有匹配的文件
    matching_files = [f for f in downloaded_files if keyword in f]
    if matching_files:
        found.append((cite, name, matching_files[0]))
    else:
        missing.append((cite, name))

print(f"\n✓ 已找到对应论文 ({len(found)} 篇):")
for cite, name, filename in found:
    print(f"  • [{cite}] {name}")
    print(f"    → {filename}")

if missing:
    print(f"\n✗ 缺少的论文 ({len(missing)} 篇):")
    for cite, name in missing:
        print(f"  • [{cite}] {name}")

print("\n" + "=" * 70)
print("总结")
print("=" * 70)
print(f"5.3.3节应包含 {len(citation_map)} 篇论文")
print(f"已下载: {len(found)} 篇")
print(f"缺少: {len(missing)} 篇")

if missing:
    print("\n缺少的论文详情:")
    for cite, name in missing:
        print(f"  - {name} ({cite})")
        if cite == "Li et al., 2025g":
            print("    作者: Rui Li, Zeyu Zhang, Xiaohe Bo, Zihang Tian, Xu Chen,")
            print("          Quanyu Dai, Zhenhua Dong, Ruiming Tang")
            print("    标题: CAM: A constructivist view of agentic memory for llm-based reading comprehension")
            print("    时间: October 2025")
            print("    状态: 在PDF参考文献中未提供arXiv ID或URL，需要手动搜索")

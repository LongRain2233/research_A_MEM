"""
将两个文件夹中 is_related_to_agent_memory: true 的论文合并为单个 Markdown 文档
"""
import os
import json
import sys
from pathlib import Path

# 解决 Windows 控制台中文输出问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# ================= 配置区域 =================
DIR1 = r"D:\research\research_A_MEM\paper2024_txt1_json"
DIR2 = r"D:\research\research_A_MEM\533_md_json"
OUTPUT_PATH = r"D:\research\research_A_MEM\docs\All_Papers_Review_with_github.md"
# ============================================

def load_json_file(filepath):
    """加载单个 JSON 文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ 加载失败 {filepath}: {e}")
        return None

def is_agent_memory_related(data):
    """检查是否为智能体记忆相关论文"""
    return data.get("is_related_to_agent_memory") is True

def format_paper_as_markdown(data, source_dir):
    """将论文数据格式化为 Markdown"""
    title = data.get("title", "Unknown Title")
    source_file = data.get("source_file", "Unknown")
    has_github = data.get("has_github_link", False)

    # 添加来源目录标识
    dir_name = os.path.basename(source_dir)

    # GitHub 链接状态显示
    github_status = "🔗 有 GitHub" if has_github else "❌ 无 GitHub"

    md = f"## 📄 {title}\n"
    md += f"**来源**: `{dir_name}` | **文件**: {source_file} | **{github_status}**\n\n"

    md += "### 一、问题与动机\n"
    md += f"{data.get('problem_and_motivation', '')}\n\n"

    md += "### 二、核心方法与技术创新\n"
    md += f"{data.get('core_method', '')}\n\n"

    md += "### 三、关键实验与结论\n"
    md += f"{data.get('key_experiments_and_results', '')}\n\n"

    md += "### 四、局限性与致命缺陷\n"
    md += f"{data.get('limitations_and_critique', '')}\n\n"

    md += "### 五、对其他AI的启发与研究契机\n"
    md += f"{data.get('ai_inspiration_and_opportunities', '')}\n\n"

    md += "---\n\n"
    return md

def main():
    all_papers = []

    # 处理两个目录
    for dir_path in [DIR1, DIR2]:
        print(f"\n📂 正在扫描: {dir_path}")
        dir_path = Path(dir_path)

        if not dir_path.exists():
            print(f"⚠️ 目录不存在: {dir_path}")
            continue

        json_files = list(dir_path.glob("*.json"))
        print(f"   发现 {len(json_files)} 个 JSON 文件")

        related_count = 0
        for json_file in json_files:
            data = load_json_file(json_file)
            if data and is_agent_memory_related(data):
                all_papers.append((data, str(dir_path)))
                related_count += 1

        print(f"   ✅ 其中 {related_count} 篇与 Agent Memory 相关")

    # 按标题排序
    all_papers.sort(key=lambda x: x[0].get("title", ""))

    print(f"\n📊 总计找到 {len(all_papers)} 篇相关论文")

    # 生成 Markdown
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write("# 📚 Agent Memory 论文综述\n\n")
        f.write(f"共 {len(all_papers)} 篇相关论文\n\n")
        f.write("---\n\n")

        for data, source_dir in all_papers:
            md = format_paper_as_markdown(data, source_dir)
            f.write(md)

    print(f"\n✅ 已生成合并文档: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
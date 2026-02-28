import json
import os
from pathlib import Path

def find_titles_with_flag(folder_path, flag_value=True):
    """查找指定文件夹中 is_related_to_agent_memory 为指定值的文件的 title"""
    titles = {}
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"文件夹不存在: {folder_path}")
        return titles
    
    json_files = list(folder.glob("**/*.json"))
    print(f"在 {folder_path} 中找到 {len(json_files)} 个 JSON 文件")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查 is_related_to_agent_memory 字段
            if isinstance(data, dict) and 'is_related_to_agent_memory' in data:
                if data['is_related_to_agent_memory'] == flag_value:
                    title = data.get('title', '')
                    if title:
                        titles[title] = str(json_file)
        except Exception as e:
            print(f"读取文件 {json_file} 出错: {e}")
    
    return titles

def main():
    folder1 = r"D:\research\research_A_MEM\533_md_json"
    folder2 = r"D:\research\research_A_MEM\paper2024_txt1_json"
    
    print("=" * 60)
    print("开始分析 JSON 文件")
    print("=" * 60)
    
    # 查找两个文件夹中 is_related_to_agent_memory 为 true 的文件的 title
    print("\n[1] 扫描 533_md_json 文件夹...")
    titles_folder1 = find_titles_with_flag(folder1, True)
    print(f"在 533_md_json 中找到 {len(titles_folder1)} 个 is_related_to_agent_memory=true 的文件")
    
    print("\n[2] 扫描 paper2024_txt1_json 文件夹...")
    titles_folder2 = find_titles_with_flag(folder2, True)
    print(f"在 paper2024_txt1_json 中找到 {len(titles_folder2)} 个 is_related_to_agent_memory=true 的文件")
    
    # 查找相同的 title
    print("\n[3] 比较两个文件夹中的 title...")
    common_titles = set(titles_folder1.keys()) & set(titles_folder2.keys())
    print(f"找到 {len(common_titles)} 个相同的 title")
    
    if common_titles:
        print("\n相同的 title 列表:")
        for i, title in enumerate(common_titles, 1):
            print(f"  {i}. {title}")
    
    # 删除 533_md_json 中匹配的文件
    print("\n[4] 删除 533_md_json 中匹配的文件...")
    deleted_files = []
    
    for title in common_titles:
        file_path = titles_folder1[title]
        try:
            os.remove(file_path)
            deleted_files.append({
                'title': title,
                'path': file_path
            })
            print(f"  已删除: {title}")
        except Exception as e:
            print(f"  删除失败 {title}: {e}")
    
    # 输出结果摘要
    print("\n" + "=" * 60)
    print("删除结果摘要")
    print("=" * 60)
    print(f"总共删除了 {len(deleted_files)} 个文件")
    print("\n删除的文件详情:")
    for i, file_info in enumerate(deleted_files, 1):
        print(f"\n  [{i}] Title: {file_info['title']}")
        print(f"      路径: {file_info['path']}")

if __name__ == "__main__":
    main()

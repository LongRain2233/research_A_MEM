import requests
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

output_dir = r"D:\research\research_A_MEM\papers_533"
os.makedirs(output_dir, exist_ok=True)

def download_arxiv(arxiv_id, filename_prefix, paper_title):
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    output_filename = f"{filename_prefix}_{arxiv_id}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_path):
        print(f"  ✓ 已存在: {output_filename}")
        return True
    
    print(f"\n[下载] {paper_title}")
    print(f"  arXiv: {arxiv_id}")
    print(f"  URL: {pdf_url}")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(pdf_url, stream=True, headers=headers, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  ✓ 下载成功: {output_filename} ({os.path.getsize(output_path) // 1024} KB)")
        return True
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return False

def download_openreview(forum_id, filename_prefix, paper_title):
    """下载OpenReview上的论文PDF"""
    # OpenReview PDF下载URL格式
    pdf_url = f"https://openreview.net/pdf?id={forum_id}"
    output_filename = f"{filename_prefix}_{forum_id}.pdf"
    output_path = os.path.join(output_dir, output_filename)
    
    if os.path.exists(output_path):
        print(f"  ✓ 已存在: {output_filename}")
        return True
    
    print(f"\n[下载] {paper_title}")
    print(f"  OpenReview ID: {forum_id}")
    print(f"  URL: {pdf_url}")
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/pdf,*/*'
        }
        response = requests.get(pdf_url, stream=True, headers=headers, timeout=60)
        response.raise_for_status()
        
        # Check if it's actually a PDF
        content_type = response.headers.get('Content-Type', '')
        print(f"  Content-Type: {content_type}")
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        file_size = os.path.getsize(output_path)
        print(f"  ✓ 下载成功: {output_filename} ({file_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return False

print("=" * 60)
print("下载剩余论文")
print("=" * 60)

# 1. MAICC - arXiv 2511.10030
download_arxiv(
    arxiv_id="2511.10030",
    filename_prefix="Jiang_2025c_MAICC",
    paper_title="MAICC: Multi-agent in-context coordination via decentralized memory retrieval"
)

# 2. SeCom - OpenReview ICLR 2025
download_openreview(
    forum_id="xKDZAW0He3",
    filename_prefix="Pan_2025_SeCom",
    paper_title="SeCom: On memory construction and retrieval for personalized conversational agents"
)

# 3. CAM - Try arXiv search (October 2025, no explicit ID in PDF)
# Try some candidate IDs
cam_candidates = ["2510.12345", "2510.14567"]  # placeholder - need web search
print("\n[信息] CAM (Li et al., 2025g)")
print("  该论文在PDF参考文献中未提供arXiv ID或URL")
print("  作者: Rui Li, Zeyu Zhang, Xiaohe Bo, Zihang Tian, Xu Chen, Quanyu Dai, Zhenhua Dong, Ruiming Tang")
print("  标题: CAM: A constructivist view of agentic memory for llm-based reading comprehension")
print("  时间: October 2025")
print("  需要手动在arXiv或Google Scholar上搜索")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)

# List downloaded files
files = [f for f in os.listdir(output_dir) if f.endswith('.pdf')]
print(f"\n当前已下载 {len(files)} 篇论文:")
for i, f in enumerate(sorted(files), 1):
    size = os.path.getsize(os.path.join(output_dir, f)) // 1024
    print(f"  {i:2d}. {f} ({size} KB)")

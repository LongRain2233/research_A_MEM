import fitz
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

doc = fitz.open(r'D:\research\research_A_MEM\MemoryintheAgeofAIAgents.pdf')

full_text = ""
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    full_text += page.get_text()

doc.close()

# Search for CAM paper's arXiv link
print("=== CAM paper full reference entry ===")
for m in re.finditer(r'(?i)rui li.*?cam.*?constructivist|cam.*?constructivist.*?reading comprehension', full_text):
    ctx = full_text[max(0,m.start()-100):m.end()+800]
    print(repr(ctx))
    print("---")

# Search with specific author names
print("\n=== Rui Li, Zeyu Zhang reference ===")
for m in re.finditer(r'Rui Li, Zeyu Zhang', full_text):
    ctx = full_text[max(0,m.start()-100):m.end()+800]
    print(repr(ctx))
    print("---")

# Search for SeCom arXiv ID
print("\n=== SeCom paper entry ===")
for m in re.finditer(r'(?i)secom|zhuoshi pan', full_text):
    ctx = full_text[max(0,m.start()-100):m.end()+700]
    print(repr(ctx[:800]))
    print("---")

import pymupdf
import sys

doc = pymupdf.open(r'D:\文档\论文\agent集合\Liu 等 - 2026 - SimpleMem Efficient lifelong memory for LLM agents.pdf')
text = ''
for page in doc:
    text += page.get_text()

with open(r'D:\文档\论文\agent集合\simplemem_text.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Extracted {len(text)} characters")

import os, sys
os.chdir(r'D:\research\research_A_MEM')
sys.stdout.reconfigure(encoding='utf-8')

import PyPDF2
fname = 'Wu 等 - 2025 - LongMemEval Benchmarking chat assistants on long-term interactive memory.pdf'
with open(fname, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    print(f"Total pages: {len(reader.pages)}")
    # Extract pages - focus on results/analysis sections (likely pages 5-15)
    for i in range(min(18, len(reader.pages))):
        print(f"\n=== PAGE {i+1} ===")
        text = reader.pages[i].extract_text()
        if text:
            print(text[:4000])

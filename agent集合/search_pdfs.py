import fitz
import os
import re

folder = '.'
search_terms = [
    r'memory.{0,20}gat(?:e|ing)',
    r'write.{0,20}gat(?:e|ing)',
    r'selective.{0,20}(?:memory|storage|memoriz)',
    r'memory.{0,20}filter',
    r'information.{0,20}filter',
    r'importance.{0,20}scor',
    r'memory.{0,20}quality',
    r'what.{0,20}(?:to|should).{0,20}(?:remember|memoriz|store|forget)',
    r'memory.{0,20}write.{0,20}control',
    r'memory.{0,20}select',
    r'quality.{0,20}(?:gate|gating|control|filter)',
    r'pre.storage',
    r'storage.{0,20}reason',
    r'memory.{0,20}importance',
    r'forget(?:ting)?.{0,20}gate',
    r'write.{0,20}control',
    r'memory.{0,20}(?:assess|evaluat)',
    r'(?:decide|deciding|decision).{0,20}(?:what|which).{0,20}(?:to|should).{0,20}(?:store|remember|memoriz)',
    r'memory.{0,20}(?:value|worth)',
    r'(?:discard|prun).{0,20}memory',
    r'memory.{0,20}(?:discard|prun)',
    r'(?:store|write).{0,20}(?:decision|criteria|threshold)',
    r'memory.{0,20}(?:consolidat)',
    r'(?:relevant|irrelevant).{0,20}(?:memory|information).{0,20}(?:store|write|save)',
    r'(?:update|updating).{0,20}gate',
    r'(?:input|write).{0,20}gate',
    r'forget.{0,20}mechanism',
    r'memory.{0,20}management',
    r'(?:remember|memoriz).{0,20}(?:mechanism|strateg)',
]

pdfs = sorted([f for f in os.listdir(folder) if f.endswith('.pdf')])
print(f'Total PDFs: {len(pdfs)}')

results = {}
for i, pdf_name in enumerate(pdfs):
    try:
        doc = fitz.open(os.path.join(folder, pdf_name))
        text = ''
        for page in doc:
            text += page.get_text()
        doc.close()
        
        found_terms = {}
        text_lower = text.lower()
        for pattern in search_terms:
            matches = re.findall(pattern, text_lower)
            if matches:
                found_terms[pattern] = matches[:5]
        
        if found_terms:
            results[pdf_name] = found_terms
            print(f'[{i+1}/{len(pdfs)}] MATCH: {pdf_name[:80]}')
            for pat, matches in found_terms.items():
                print(f'  Pattern: {pat}')
                for m in matches[:3]:
                    print(f'    -> "{m}"')
        else:
            print(f'[{i+1}/{len(pdfs)}] no match: {pdf_name[:60]}')
    except Exception as e:
        print(f'[{i+1}/{len(pdfs)}] ERROR: {pdf_name[:60]}: {e}')

print(f'\n=== SUMMARY: {len(results)} papers with matches ===')
for name in results:
    print(f'  - {name[:100]}')

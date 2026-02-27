# -*- coding: utf-8 -*-
import os
import glob

base_dir = os.path.dirname(os.path.abspath(__file__))
pdf_files = glob.glob(os.path.join(base_dir, "*.pdf"))

keywords = ["PRIME", "Mem-", "prospect", "MemAgent", "Hierarchical memory", "LightMem"]

for f in pdf_files:
    fname = os.path.basename(f)
    for kw in keywords:
        if kw in fname:
            print(f"MATCH: {fname}")
            break

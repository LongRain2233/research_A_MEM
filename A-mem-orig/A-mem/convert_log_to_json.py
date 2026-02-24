import re
import json
import os
import sys
from collections import defaultdict
import statistics
from utils import calculate_metrics, aggregate_metrics

# 配置路径
LOG_FILE = r"D:\research\A-mem-orig\A-mem\logs\eval_ours_openai_gpt-4o-mini_openrouter_ratio1.0_2026-02-12-18-24.log"
OUTPUT_FILE = r"D:\research\A-mem-orig\A-mem\results_restored.json"

def clean_line(line):
    # Remove standard log prefix
    # Patterns: 
    # 2026-02-12 20:01:36,888 - INFO - ...
    # INFO:locomo_eval: ...
    line = line.strip()
    if " - INFO - " in line:
        return line.split(" - INFO - ", 1)[1].strip()
    if "INFO:locomo_eval:" in line:
        return line.split("INFO:locomo_eval:", 1)[1].strip()
    return line

def parse_log():
    entries = []
    current = {}
    current_sample_id = -1
    
    print(f"Reading log file: {LOG_FILE}")
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        cleaned = clean_line(line)
        
        # Match "Processing QA for sample 14/45"
        # Log line: Processing QA for sample 1/446
        # Need to be careful with exact format
        if "Processing QA for sample" in cleaned:
            # Try to extract numbers
            m = re.search(r'Processing QA for sample (\d+)', cleaned)
            if m:
                current_sample_id = int(m.group(1)) - 1
        
        # Match "Question 123: Text"
        # Note: Question number in log is global question count (1..N)
        q_match = re.match(r'^Question (\d+): (.*)', cleaned)
        if q_match:
            # Save previous if it has essential fields
            if 'question' in current:
                entries.append(current)
            
            current = {
                'sample_id': current_sample_id,
                'question_id': int(q_match.group(1)),
                'question': q_match.group(2),
                'prediction': "",
                'reference': "",
                'category': 0
            }
            continue
            
        if 'question' not in current:
            continue
            
        if cleaned.startswith("Prediction:"):
            current['prediction'] = cleaned[11:].strip()
        elif cleaned.startswith("Reference:"):
            current['reference'] = cleaned[10:].strip()
        elif cleaned.startswith("Category:"):
            try:
                # Handle cases where category might be followed by other text or just a number
                val = cleaned[9:].strip()
                current['category'] = int(val)
            except:
                pass

    # Add last entry
    if 'question' in current:
        entries.append(current)
        
    return entries

def main():
    if not os.path.exists(LOG_FILE):
        print(f"Error: Log file not found at {LOG_FILE}")
        return

    entries = parse_log()
    print(f"Parsed {len(entries)} entries.")
    
    if len(entries) == 0:
        print("No entries found. Check log file format.")
        return

    final_results = []
    all_metrics = []
    all_categories = []
    category_counts = defaultdict(int)
    
    print("Calculating metrics...")
    for i, entry in enumerate(entries):
        pred = entry.get('prediction', "")
        ref = entry.get('reference', "")
        cat = entry.get('category', 0)
        
        # Calculate metrics using utils.py function
        metrics = calculate_metrics(pred, ref)
        
        all_metrics.append(metrics)
        all_categories.append(cat)
        category_counts[cat] += 1
        
        final_results.append({
            "sample_id": entry.get('sample_id', i),
            "question": entry['question'],
            "prediction": pred,
            "reference": ref,
            "category": cat,
            "metrics": metrics
        })

    print("Aggregating metrics...")
    aggregate = aggregate_metrics(all_metrics, all_categories)
    
    output = {
        "model": "restored_from_log",
        "dataset": "restored_from_log",
        "total_questions": len(entries),
        "category_distribution": {str(k): v for k, v in category_counts.items()},
        "aggregate_metrics": aggregate,
        "individual_results": final_results
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"Successfully saved results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

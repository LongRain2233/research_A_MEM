import json
import os

def sort_results():
    filepath = r'D:\research\research_A_MEM\PhaseForget-Zettel\code\PhaseForget\data\full_compare.json'
    if not os.path.exists(filepath):
        # Try relative path if absolute fails
        filepath = os.path.join('data', 'full_compare.json')
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return

    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter out entries without metrics if any
    data = [entry for entry in data if 'metrics' in entry]

    # Sort by avg_f1 descending
    sorted_data = sorted(data, key=lambda x: x['metrics'].get('avg_f1', 0), reverse=True)

    print(f"{'Rank':<5} | {'Avg F1':<10} | {'Trial ID'}")
    print("=" * 100)

    for i, entry in enumerate(sorted_data):
        trial_id = entry.get('trial_id', 'N/A')
        metrics = entry['metrics']
        avg_f1 = metrics.get('avg_f1', 0)
        params = entry.get('params', {})
        by_category = metrics.get('by_category', {})
        memory_stats = metrics.get('memory_stats', {})

        # Format category F1s
        cat_f1_str = ", ".join([f"Cat {cat}: {m.get('avg_f1', 0):.4f}" for cat, m in by_category.items()])
        
        print(f"{i+1:<5} | {avg_f1:<10.4f} | {trial_id}")
        print(f"      Params: {params}")
        print(f"      Category F1s: {cat_f1_str}")
        print(f"      Memory Stats: {memory_stats}")
        print("-" * 100)

if __name__ == "__main__":
    sort_results()

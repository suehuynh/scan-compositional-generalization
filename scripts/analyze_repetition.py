# scripts/analyze_repetition.py

import pandas as pd
from collections import defaultdict

# Load errors
errors_df = pd.read_csv('results/error_breakdown/compositional_analysis/merged_errors.csv')
repetition_errors = errors_df[errors_df['error_type'] == 'REPETITION']

# Analyze by modifier type
modifier_analysis = defaultdict(lambda: {'count': 0, 'examples': []})

for _, row in repetition_errors.iterrows():
    input_text = row['input']
    
    # Extract modifiers
    modifiers = []
    if 'twice' in input_text:
        modifiers.append('twice')
    if 'thrice' in input_text:
        modifiers.append('thrice')
    if 'four times' in input_text or 'around' in input_text:
        modifiers.append('four_times')
    
    for mod in modifiers or ['unknown']:
        modifier_analysis[mod]['count'] += 1
        if len(modifier_analysis[mod]['examples']) < 3:
            modifier_analysis[mod]['examples'].append({
                'input': row['input'],
                'target': row['target'],
                'pred': row['prediction']
            })

# Print findings
for mod, stats in modifier_analysis.items():
    print(f"\n{mod.upper()}: {stats['count']} errors")
    for ex in stats['examples']:
        print(f"  Input: {ex['input']}")
        print(f"  Pred:  {ex['pred']}")
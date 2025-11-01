import pandas as pd
import numpy as np

ai_models = pd.read_csv('data/ai_models/all_ai_models.csv')

# Get models with both compute and params
complete = ai_models[
    ai_models['Training compute (FLOP)'].notna() & 
    ai_models['Parameters'].notna()
].copy()

# Calculate tokens implied by compute
# Chinchilla: compute = 6 * params * tokens
# So: tokens = compute / (6 * params)
complete['implied_tokens'] = complete['Training compute (FLOP)'] / (6 * complete['Parameters'])
complete['tokens_per_param'] = complete['implied_tokens'] / complete['Parameters']

# Focus on large models (>= 1B params)
large = complete[complete['Parameters'] >= 1e9].copy()
large = large.sort_values('Publication date')

print('COMPUTE-TO-PARAMETER RELATIONSHIPS (Large Models):')
print('='*100)

for _, row in large.tail(30).iterrows():
    tpp = row['tokens_per_param']
    print(f'{row["Model"]:40s} | {str(row["Publication date"]):12s} | '
          f'{row["Parameters"]:12.2e} params | {row["Training compute (FLOP)"]:15.2e} FLOPs | '
          f'{tpp:12.1f} tok/param')

print()
print('SUMMARY STATISTICS:')
print(f'  Median tokens/param: {large["tokens_per_param"].median():.1f}')
print(f'  Mean tokens/param: {large["tokens_per_param"].mean():.1f}')
print(f'  Chinchilla optimal: 20.0')
print(f'  Sample size: {len(large)} models')


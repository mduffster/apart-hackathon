"""
Step 1: Match METR Models to AI Models Dataset
Goal: Link each METR model to its training compute and parameters
"""

import pandas as pd
import json
import numpy as np

# Load METR runs data to get unique model names
print("Loading METR data...")
metr_runs = []
with open('data/METR/all_runs.jsonl', 'r') as f:
    for line in f:
        metr_runs.append(json.loads(line))

metr_df = pd.DataFrame(metr_runs)

# Get unique models - use both 'model' and 'alias' fields
metr_models_raw = metr_df['model'].unique()
metr_aliases = metr_df['alias'].unique()

print(f"Found {len(metr_models_raw)} unique METR model IDs")
print(f"Found {len(metr_aliases)} unique METR model aliases")

# Create mapping from alias to model ID
alias_to_id = dict(zip(metr_df['alias'], metr_df['model']))

# Load AI models dataset (using all_ai_models.csv as it had best coverage)
print("\nLoading AI models dataset...")
ai_models = pd.read_csv('data/ai_models/all_ai_models.csv')

# Filter to models with required fields
ai_models_filtered = ai_models[
    ai_models['Training compute (FLOP)'].notna() &
    ai_models['Parameters'].notna()
].copy()

print(f"AI models with compute+params data: {len(ai_models_filtered)} of {len(ai_models)}")

# Manual mapping for known models - map METR alias to EXACT AI model name
MANUAL_MAPPINGS = {
    # GPT-4 family - use GPT-4 as proxy for all variants
    'GPT-4o': 'GPT-4',
    'GPT-4 Turbo': 'GPT-4',
    'GPT-4 1106': 'GPT-4',
    'GPT-4 0314': 'GPT-4',
    'GPT-4 0125': 'GPT-4',
    
    # o1 family - no data available, will need special handling
    # 'o1': None,
    # 'o1-preview': None,
    
    # GPT-2 - use the main GPT-2 (124M) model
    'GPT-2': 'GPT-2 (124M)',
    
    # GPT-3 family
    'davinci-002 (GPT-3)': 'GPT-3 175B (davinci)',
    'gpt-3.5-turbo-instruct': 'GPT-3 175B (davinci)',  # Closest proxy
    
    # Claude family - no compute+params data available
    # 'Claude 3 Opus': None,
    # 'Claude 3.5 Sonnet (New)': None,
    # 'Claude 3.5 Sonnet (Old)': None,
    # 'Claude 3.7 Sonnet': None,
}

def find_ai_model(metr_alias, ai_df):
    """Find matching AI model using manual mappings and exact model names"""
    
    # Try manual mapping first (exact match)
    if metr_alias in MANUAL_MAPPINGS:
        exact_model_name = MANUAL_MAPPINGS[metr_alias]
        mask = ai_df['Model'] == exact_model_name
        if mask.any():
            return ai_df[mask].iloc[0]
    
    # Try direct exact match
    mask = ai_df['Model'] == metr_alias
    if mask.any():
        return ai_df[mask].iloc[0]
    
    return None

# Match METR models to AI models
print("\nMatching METR models to AI models...")
matches = []

for metr_alias in metr_aliases:
    # Skip "human" - not a model
    if metr_alias.lower() == 'human':
        continue
    
    ai_match = find_ai_model(metr_alias, ai_models_filtered)
    
    if ai_match is not None:
        matches.append({
            'metr_alias': metr_alias,
            'metr_model_id': alias_to_id[metr_alias],
            'ai_model_name': ai_match['Model'],
            'release_date': ai_match.get('Publication date', None),
            'training_compute_flops': ai_match['Training compute (FLOP)'],
            'parameters': ai_match['Parameters'],
            'organization': ai_match.get('Organization', None),
            'domain': ai_match.get('Domain', None)
        })
    else:
        print(f"  ⚠️  No match found for: {metr_alias}")

matches_df = pd.DataFrame(matches)

print(f"\nMatching complete!")
print(f"  Total METR models: {len(metr_aliases)}")
print(f"  Matched: {len(matches_df)}")
print(f"  Unmatched: {len(metr_aliases) - len(matches_df) - 1}")  # -1 for "human"

# Check data coverage
with_dates = matches_df['release_date'].notna().sum()
with_compute = matches_df['training_compute_flops'].notna().sum()
with_params = matches_df['parameters'].notna().sum()

print(f"\nData coverage:")
print(f"  With release dates: {with_dates}/{len(matches_df)}")
print(f"  With compute data: {with_compute}/{len(matches_df)}")
print(f"  With parameter data: {with_params}/{len(matches_df)}")

# Save results
matches_df.to_csv('outputs/metr_ai_model_matches.csv', index=False)
print(f"\nSaved matches to outputs/metr_ai_model_matches.csv")

# Display matches for review
print("\n" + "="*100)
print("MATCHED MODELS:")
print("="*100)
display_cols = ['metr_alias', 'ai_model_name', 'training_compute_flops', 'parameters', 'release_date']
for _, row in matches_df.iterrows():
    print(f"\n{row['metr_alias']:30s} → {row['ai_model_name']}")
    print(f"  Compute: {row['training_compute_flops']:.2e} FLOPs")
    print(f"  Params:  {row['parameters']:.2e}")
    print(f"  Date:    {row['release_date']}")

# Summary statistics
print("\n" + "="*100)
print("SUMMARY STATISTICS:")
print("="*100)
print(f"\nCompute range:")
print(f"  Min: {matches_df['training_compute_flops'].min():.2e} FLOPs")
print(f"  Max: {matches_df['training_compute_flops'].max():.2e} FLOPs")
print(f"  Mean: {matches_df['training_compute_flops'].mean():.2e} FLOPs")

print(f"\nParameter range:")
print(f"  Min: {matches_df['parameters'].min():.2e}")
print(f"  Max: {matches_df['parameters'].max():.2e}")
print(f"  Mean: {matches_df['parameters'].mean():.2e}")

print(f"\nDate range:")
print(f"  Earliest: {matches_df['release_date'].min()}")
print(f"  Latest: {matches_df['release_date'].max()}")

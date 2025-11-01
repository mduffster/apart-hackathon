"""
STEP 2: Build datasets for hierarchical model

Creates three datasets:
1. Stage 1 (Compute → ECI): ~80 Epoch models with compute and ECI scores
2. Stage 2 (ECI → METR): ~8 overlap models with both ECI and METR
3. Current state: Latest frontier models for baseline

Output:
- stage1_compute_eci.csv: Epoch models (C → ECI)
- stage2_eci_metr.csv: Overlap models (ECI → METR)
- current_state.csv: Current frontier for conditioning
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("="*100)
print("STEP 2: BUILD DATASETS FOR HIERARCHICAL MODEL")
print("="*100)

# ============================================================================
# Load METR capability (from Step 1)
# ============================================================================

metr_capability = pd.read_csv('outputs/metr_capability_summary.csv')
print(f"\nLoaded METR capability for {len(metr_capability)} models")

# ============================================================================
# Load Epoch ECI data
# ============================================================================

eci_file = Path('data/benchmark_data/epoch_capabilities_index.csv')
eci_data = pd.read_csv(eci_file)

print(f"\n{'='*100}")
print("STAGE 1: COMPUTE → ECI")
print("="*100)

# Filter for models with both compute and ECI
stage1_data = eci_data[
    eci_data['Training compute (FLOP)'].notna() & 
    eci_data['ECI Score'].notna()
].copy()

stage1_data = stage1_data.rename(columns={
    'Training compute (FLOP)': 'compute',
    'ECI Score': 'eci'
})

stage1_data = stage1_data[['Model version', 'compute', 'eci', 'Release date']].copy()
stage1_data = stage1_data.rename(columns={'Model version': 'Model', 'Release date': 'date'})
stage1_data['log_compute'] = np.log10(stage1_data['compute'])

print(f"\nStage 1 dataset: {len(stage1_data)} models")
print(f"  Compute range: {stage1_data['compute'].min():.2e} to {stage1_data['compute'].max():.2e}")
print(f"  ECI range: {stage1_data['eci'].min():.1f} to {stage1_data['eci'].max():.1f}")

# Quick OLS for reference
from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(
    stage1_data['log_compute'], 
    stage1_data['eci']
)
print(f"\nOLS: ECI = {intercept:.2f} + {slope:.2f} * log10(C)")
print(f"  R² = {r_value**2:.4f}")

# Save
stage1_file = 'outputs/stage1_compute_eci.csv'
stage1_data.to_csv(stage1_file, index=False)
print(f"\n✅ Saved: {stage1_file}")

# ============================================================================
# Find overlap: Models in both ECI and METR
# ============================================================================

print(f"\n{'='*100}")
print("STAGE 2: ECI → METR (Overlap)")
print("="*100)

# Manual mapping of METR model IDs to Epoch model names
metr_to_epoch = {
    'gpt2': 'GPT-2',
    'gpt_3_5_turbo': 'GPT-3.5',
    'gpt_4': 'GPT-4',
    'gpt_4_turbo': 'GPT-4 Turbo',
    'gpt_4o': 'GPT-4o',
    'o1_preview': 'o1-preview',
    'claude_3_opus': 'Claude 3 Opus',
    'claude_3_5_sonnet': 'Claude 3.5 Sonnet',
    'claude_3_7_sonnet': 'Claude 3.7 Sonnet',
}

# Get ECI scores for overlap models
overlap_records = []
for metr_model, epoch_model in metr_to_epoch.items():
    # Get METR capability
    metr_row = metr_capability[metr_capability['model'] == metr_model]
    if len(metr_row) == 0:
        continue
    
    metr_cap = metr_row['metr_capability'].iloc[0]
    compute = metr_row['compute'].iloc[0]
    
    # Get ECI score (try exact match first, then fuzzy)
    eci_row = eci_data[eci_data['Model version'] == epoch_model]
    if len(eci_row) == 0:
        # Try fuzzy match
        eci_row = eci_data[eci_data['Model version'].str.contains(epoch_model.split()[0], case=False, na=False)]
    
    if len(eci_row) > 0:
        eci_score = eci_row['ECI Score'].iloc[0]
        overlap_records.append({
            'metr_model': metr_model,
            'epoch_model': epoch_model,
            'compute': compute,
            'eci': eci_score,
            'metr': metr_cap
        })
        print(f"  ✓ {metr_model} → {epoch_model}: ECI={eci_score:.1f}, METR={metr_cap:.3f}")
    else:
        print(f"  ✗ {metr_model} → {epoch_model}: No ECI match")

stage2_data = pd.DataFrame(overlap_records)

# Estimate missing ECI values from compute using Stage 1 relationship
missing_eci = stage2_data['eci'].isna()
if missing_eci.any():
    print(f"\nEstimating ECI from compute for {missing_eci.sum()} models...")
    for idx in stage2_data[missing_eci].index:
        compute = stage2_data.loc[idx, 'compute']
        eci_est = intercept + slope * np.log10(compute)
        stage2_data.loc[idx, 'eci'] = eci_est
        print(f"  {stage2_data.loc[idx, 'metr_model']}: ECI ≈ {eci_est:.1f} (from compute)")

if len(stage2_data) == 0:
    print("\n⚠️  WARNING: No overlap found! Using compute-based ECI estimates.")
    # Fallback: estimate ECI from compute using Stage 1 relationship
    for metr_model in metr_capability['model']:
        metr_row = metr_capability[metr_capability['model'] == metr_model]
        compute = metr_row['compute'].iloc[0]
        metr_cap = metr_row['metr_capability'].iloc[0]
        
        # Estimate ECI from compute
        eci_est = intercept + slope * np.log10(compute)
        
        overlap_records.append({
            'metr_model': metr_model,
            'epoch_model': f"{metr_model} (estimated)",
            'compute': compute,
            'eci': eci_est,
            'metr': metr_cap
        })
    
    stage2_data = pd.DataFrame(overlap_records)

print(f"\nStage 2 dataset: {len(stage2_data)} models")
print(f"  ECI range: {stage2_data['eci'].min():.1f} to {stage2_data['eci'].max():.1f}")
print(f"  METR range: {stage2_data['metr'].min():.3f} to {stage2_data['metr'].max():.3f}")

# OLS for ECI → METR (on logit scale)
from scipy.special import logit
stage2_data['logit_metr'] = logit(stage2_data['metr'].clip(0.001, 0.999))

slope2, intercept2, r_value2, p_value2, std_err2 = linregress(
    stage2_data['eci'], 
    stage2_data['logit_metr']
)
print(f"\nOLS: logit(METR) = {intercept2:.3f} + {slope2:.4f} * ECI")
print(f"  R² = {r_value2**2:.4f}")

# Save
stage2_file = 'outputs/stage2_eci_metr.csv'
stage2_data.to_csv(stage2_file, index=False)
print(f"\n✅ Saved: {stage2_file}")

# ============================================================================
# Current state (for conditioning)
# ============================================================================

print(f"\n{'='*100}")
print("CURRENT STATE")
print("="*100)

# Get best current models
current_state = metr_capability.nlargest(2, 'compute').copy()

print(f"\nCurrent frontier models:")
for _, row in current_state.iterrows():
    print(f"  {row['model']}: Compute={row['compute']:.2e}, METR={row['metr_capability']:.3f}")

# Use the best one for conditioning
best_current = current_state.iloc[0]
conditioning_compute = best_current['compute']
conditioning_metr = best_current['metr_capability']

print(f"\nConditioning point (best current model):")
print(f"  Model: {best_current['model']}")
print(f"  Compute: {conditioning_compute:.2e} FLOPs")
print(f"  METR: {conditioning_metr:.3f}")
print(f"  Upper bound for conditioning: {conditioning_metr + 0.07:.2f}")

# Save
current_file = 'outputs/current_state.csv'
current_state.to_csv(current_file, index=False)
print(f"\n✅ Saved: {current_file}")

print(f"\n{'='*100}")
print("✅ STEP 2 COMPLETE")
print("="*100)

print(f"\n💡 SUMMARY:")
print(f"  • Stage 1 (C→ECI): {len(stage1_data)} Epoch models")
print(f"  • Stage 2 (ECI→METR): {len(stage2_data)} overlap models")
print(f"  • Current frontier METR: {conditioning_metr:.3f}")
print(f"  • Conditioning upper bound: {conditioning_metr + 0.07:.2f}")
print(f"\n📊 Next step: Run 03_bayesian_hierarchical_model.py to fit the model")


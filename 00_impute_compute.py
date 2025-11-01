#!/usr/bin/env python3
"""
STEP 0: Principled Compute Imputation for METR Models

Priority:
1. Use exact/fuzzy match from Epoch all_ai_models.csv
2. Estimate from ECI using Stage 1 inverse: log10(C) = (ECI - a) / b
3. Use temporal trends as last resort

Output: compute_assignments.json with full provenance
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
from thefuzz import process
import json

print("="*100)
print("STEP 0: PRINCIPLED COMPUTE IMPUTATION")
print("="*100)

# ============================================================================
# Load Epoch Data
# ============================================================================

all_models = pd.read_csv('data/ai_models/all_ai_models.csv')
eci_data = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')

print(f"\nEpoch all_ai_models: {len(all_models)} models")
print(f"Epoch ECI dataset: {len(eci_data)} models")

# ============================================================================
# Stage 1: Fit Compute → ECI on clean data
# ============================================================================

# Get models with both compute and ECI
eci_clean = eci_data[
    eci_data['Training compute (FLOP)'].notna() & 
    eci_data['ECI Score'].notna()
].copy()

compute_eci = eci_clean['Training compute (FLOP)'].values
eci_scores = eci_clean['ECI Score'].values
log_compute_eci = np.log10(compute_eci)

# Fit ECI = a + b * log10(C)
slope, intercept, r_value, p_value, std_err = linregress(log_compute_eci, eci_scores)

print(f"\n{'='*100}")
print("STAGE 1: Compute → ECI Relationship")
print("="*100)
print(f"\nLinear fit: ECI = {intercept:.2f} + {slope:.2f} * log10(C)")
print(f"  R² = {r_value**2:.4f}")
print(f"  Std error = {std_err:.3f}")
print(f"  Based on {len(eci_clean)} models")

# ============================================================================
# METR Models: Impute Compute
# ============================================================================

print(f"\n{'='*100}")
print("IMPUTING COMPUTE FOR METR MODELS")
print("="*100)

metr_models = {
    'gpt2': {},
    'gpt_3_5_turbo': {},
    'gpt_4': {},
    'gpt_4_turbo': {},
    'gpt_4o': {},
    'o1_preview': {},
    'claude_3_opus': {},
    'claude_3_5_sonnet': {},
    'claude_3_7_sonnet': {},
}

compute_assignments = {}

for metr_name in metr_models.keys():
    print(f"\n{metr_name}:")
    
    # ------------------------------------------------------------------------
    # Method 1: Fuzzy match to all_ai_models.csv
    # ------------------------------------------------------------------------
    
    model_list = all_models['Model'].dropna().tolist()
    matches = process.extract(metr_name, model_list, limit=1)
    
    if len(matches) > 0:
        best_match = matches[0]
        match_name, score = best_match[0], best_match[1]
        
        if score >= 90:  # High confidence match
            row = all_models[all_models['Model'] == match_name].iloc[0]
            compute_val = row.get('Training compute (FLOP)')
            
            if pd.notna(compute_val):
                print(f"  ✓ METHOD 1: Matched '{match_name}' (score={score})")
                print(f"    Compute: {compute_val:.2e} FLOPs")
                compute_assignments[metr_name] = {
                    'compute': float(compute_val),
                    'method': 'epoch_match',
                    'source': match_name,
                    'confidence': score,
                    'provenance': f'Matched to Epoch all_ai_models.csv'
                }
                continue
    
    # ------------------------------------------------------------------------
    # Method 2: Estimate from ECI (if available)
    # ------------------------------------------------------------------------
    
    # Try to find ECI for this model
    eci_matches = process.extract(metr_name, eci_data['Model version'].dropna().tolist(), limit=1)
    
    if len(eci_matches) > 0:
        eci_match_name, eci_score = eci_matches[0][0], eci_matches[0][1]
        
        if eci_score >= 80:
            eci_row = eci_data[eci_data['Model version'] == eci_match_name].iloc[0]
            eci_val = eci_row.get('ECI Score')
            
            if pd.notna(eci_val):
                # Inverse Stage 1: log10(C) = (ECI - a) / b
                log_compute_est = (eci_val - intercept) / slope
                compute_est = 10 ** log_compute_est
                
                # Estimate uncertainty from residuals
                residuals = eci_scores - (intercept + slope * log_compute_eci)
                sigma_eci = np.std(residuals)
                
                # Propagate uncertainty to compute
                log_compute_low = (eci_val - sigma_eci - intercept) / slope
                log_compute_high = (eci_val + sigma_eci - intercept) / slope
                compute_low = 10 ** log_compute_low
                compute_high = 10 ** log_compute_high
                
                print(f"  ✓ METHOD 2: Estimated from ECI")
                print(f"    ECI: {eci_val:.1f} (from '{eci_match_name}', score={eci_score})")
                print(f"    Compute: {compute_est:.2e} FLOPs")
                print(f"    Uncertainty: [{compute_low:.2e}, {compute_high:.2e}]")
                
                compute_assignments[metr_name] = {
                    'compute': float(compute_est),
                    'compute_low': float(compute_low),
                    'compute_high': float(compute_high),
                    'method': 'eci_estimate',
                    'eci': float(eci_val),
                    'eci_source': eci_match_name,
                    'confidence': eci_score,
                    'provenance': f'Estimated from ECI using C→ECI inverse'
                }
                continue
    
    # ------------------------------------------------------------------------
    # Method 3: Use informed prior (last resort)
    # ------------------------------------------------------------------------
    
    print(f"  ⚠ METHOD 3: Using informed prior")
    
    # Manual assignments based on domain knowledge
    manual_estimates = {
        'gpt2': {
            'compute': 1.92e21,
            'method': 'manual_epoch',
            'provenance': 'gpt2-xl from Epoch all_ai_models.csv',
            'note': 'Used gpt2-xl as proxy for gpt2'
        },
        'gpt_4_turbo': {
            'compute': 2.1e25,
            'method': 'manual_proxy',
            'provenance': 'GPT-4 from Epoch (turbo is tuned version)',
            'note': 'Assumed same compute as GPT-4'
        }
    }
    
    if metr_name in manual_estimates:
        manual = manual_estimates[metr_name]
        print(f"    Compute: {manual['compute']:.2e} FLOPs ({manual['note']})")
        compute_assignments[metr_name] = manual
    else:
        compute_assignments[metr_name] = {
            'compute': None,
            'method': 'missing',
            'provenance': 'No data available - requires manual input',
            'warning': 'This model has no compute or ECI data'
        }

# ============================================================================
# Save Results
# ============================================================================

output_file = 'outputs/compute_assignments.json'
with open(output_file, 'w') as f:
    json.dump(compute_assignments, f, indent=2)

print(f"\n{'='*100}")
print("SUMMARY")
print("="*100)

n_epoch = sum(1 for v in compute_assignments.values() if v.get('method') == 'epoch_match')
n_eci = sum(1 for v in compute_assignments.values() if v.get('method') == 'eci_estimate')
n_manual = sum(1 for v in compute_assignments.values() if v.get('method') in ['manual_epoch', 'manual_proxy'])
n_missing = sum(1 for v in compute_assignments.values() if v.get('method') == 'missing')

print(f"\n  Epoch matches: {n_epoch}")
print(f"  ECI estimates: {n_eci}")
print(f"  Manual (informed): {n_manual}")
print(f"  Missing: {n_missing}")

print(f"\n✅ Saved: {output_file}")

if n_missing > 0:
    print(f"\n⚠ WARNING: {n_missing} models have no compute data")
    print("  These will need manual estimates or should be excluded from analysis")

print(f"\n{'='*100}")
print("✅ COMPUTE IMPUTATION COMPLETE")
print("="*100)

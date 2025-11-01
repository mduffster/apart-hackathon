"""
Use ECI as Bridge to Estimate Missing Data
Params ↔ ECI ↔ Compute, then link to METR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import json

print("="*100)
print("ECI AS BRIDGE: Params ↔ ECI ↔ Compute")
print("="*100)

# Load ECI data
eci_data = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')
print(f"\nECI dataset: {len(eci_data)} models")

# Load Epoch AI models for params
ai_models = pd.read_csv('data/ai_models/all_ai_models.csv')

# Merge to get params for ECI models
eci_with_params = eci_data.merge(
    ai_models[['Model', 'Parameters']],
    left_on='Display name',
    right_on='Model',
    how='left'
)

print(f"  With compute: {eci_data['Training compute (FLOP)'].notna().sum()}")
print(f"  With ECI scores: {eci_data['ECI Score'].notna().sum()}")
print(f"  With parameters: {eci_with_params['Parameters'].notna().sum()}")

# Clean data
eci_clean = eci_with_params.dropna(subset=['ECI Score']).copy()

print(f"\n{'='*100}")
print("BUILDING RELATIONSHIPS:")
print(f"{'='*100}")

# Relationship 1: Params → ECI
params_eci_data = eci_clean.dropna(subset=['Parameters', 'ECI Score']).copy()
params_eci_data['log_params'] = np.log10(params_eci_data['Parameters'])

print(f"\n1. PARAMS → ECI:")
print(f"   Data: {len(params_eci_data)} models")

if len(params_eci_data) > 5:
    slope_p2e, intercept_p2e, r_p2e, _, _ = linregress(
        params_eci_data['log_params'],
        params_eci_data['ECI Score']
    )
    
    print(f"   ECI = {slope_p2e:.2f} * log10(Params) + {intercept_p2e:.2f}")
    print(f"   R = {r_p2e:.3f}")
    
    # Show examples
    print(f"\n   Example predictions:")
    for params in [1e9, 1e10, 1e11, 1.76e12]:
        pred_eci = slope_p2e * np.log10(params) + intercept_p2e
        print(f"     {params:.1e} params → ECI {pred_eci:.1f}")
else:
    print(f"   ✗ Not enough data")
    slope_p2e = None

# Relationship 2: Compute → ECI (load from modern scaling law)
with open('outputs/modern_scaling_law.json', 'r') as f:
    modern_law = json.load(f)

eci_inf, A_eci, alpha_eci = modern_law['params']

def compute_to_eci(compute):
    return eci_inf - A_eci * np.power(compute, -alpha_eci)

def eci_to_compute(eci):
    """Inverse: solve for compute given ECI"""
    if eci >= eci_inf:
        return np.inf
    return np.power(A_eci / (eci_inf - eci), 1.0 / alpha_eci)

print(f"\n2. COMPUTE → ECI (from modern scaling law):")
print(f"   {modern_law['formula']}")
print(f"   R² = {modern_law['r2']:.3f}")

print(f"\n   Example predictions:")
for compute in [1e24, 1e25, 1e26]:
    pred_eci = compute_to_eci(compute)
    print(f"     {compute:.0e} FLOPs → ECI {pred_eci:.1f}")

# Relationship 3: ECI → Compute (inverse)
print(f"\n3. ECI → COMPUTE (inverse):")
for eci in [120, 140, 160, 180]:
    pred_compute = eci_to_compute(eci)
    if pred_compute < 1e50:
        print(f"     ECI {eci} → {pred_compute:.2e} FLOPs")
    else:
        print(f"     ECI {eci} → inf (exceeds ceiling {eci_inf:.1f})")

# Relationship 4: Params → Compute (via ECI bridge)
if slope_p2e:
    print(f"\n4. PARAMS → COMPUTE (via ECI bridge):")
    print(f"   Params → ECI → Compute")
    
    for params in [1e9, 1e10, 1e11, 1.76e12]:
        pred_eci = slope_p2e * np.log10(params) + intercept_p2e
        pred_compute = eci_to_compute(pred_eci)
        if pred_compute < 1e50:
            print(f"     {params:.1e} params → ECI {pred_eci:.1f} → {pred_compute:.2e} FLOPs")

# Save estimation functions
bridge_model = {
    'params_to_eci': {
        'slope': float(slope_p2e) if slope_p2e else None,
        'intercept': float(intercept_p2e) if slope_p2e else None,
        'r': float(r_p2e) if slope_p2e else None,
        'n': len(params_eci_data) if slope_p2e else 0
    },
    'compute_to_eci': {
        'eci_inf': float(eci_inf),
        'A': float(A_eci),
        'alpha': float(alpha_eci),
        'formula': modern_law['formula']
    }
}

with open('outputs/eci_bridge_model.json', 'w') as f:
    json.dump(bridge_model, f, indent=2)

print(f"\n{'='*100}")
print("APPLYING TO METR MODELS:")
print(f"{'='*100}")

# Load METR comparison
metr_comparison = pd.read_csv('outputs/metr_vs_eci_comparison.csv')

print(f"\nMETR models:")
for _, row in metr_comparison.iterrows():
    model_name = row['model']
    has_eci = not pd.isna(row['eci_score'])
    has_compute = not pd.isna(row['training_compute'])
    
    print(f"\n{model_name}:")
    eci_str = f"{row['eci_score']:.1f}" if has_eci else 'N/A'
    compute_str = f"{row['training_compute']:.2e}" if has_compute else 'N/A'
    print(f"  Has ECI: {has_eci} ({eci_str})")
    print(f"  Has compute: {has_compute} ({compute_str})")
    
    # If has ECI but missing compute
    if has_eci and not has_compute:
        est_compute = eci_to_compute(row['eci_score'])
        if est_compute < 1e50:
            print(f"  → Estimated compute: {est_compute:.2e} FLOPs")
        else:
            print(f"  → Cannot estimate (ECI too high)")
    
    # If has compute but missing ECI
    if has_compute and not has_eci:
        est_eci = compute_to_eci(row['training_compute'])
        print(f"  → Estimated ECI: {est_eci:.1f}")

# Check: Can we estimate for models with neither?
print(f"\n{'='*100}")
print("MODELS NEEDING FULL CHAIN:")
print(f"{'='*100}")

# Models from METR that have NEITHER compute NOR ECI
# Need to check if they have parameters
print(f"\nSearching for models with only parameters...")

# Load METR runs to get all model names
metr_runs = pd.read_json('data/METR/all_runs.jsonl', lines=True)
metr_models = metr_runs['model'].unique()

print(f"METR has {len(metr_models)} unique models")

# Check which ones we can estimate
# (This would require manual parameter lookups or web search)

print(f"\n💡 KEY INSIGHT:")
print(f"   ECI acts as a 'common currency' between:")
print(f"   • Parameters (model size)")
print(f"   • Compute (training cost)")
print(f"   • METR (task performance)")
print(f"\n   For any model with 1 of these, we can estimate the others!")

print(f"\n{'='*100}")
print("✅ ECI BRIDGE MODEL SAVED")
print(f"{'='*100}")


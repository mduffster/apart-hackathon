"""
Estimate Training Compute for Models Missing Data
Based on empirical relationships from known models (ECI dataset)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import json

print("="*100)
print("ESTIMATING COMPUTE FOR MODELS MISSING DATA")
print("="*100)

# Load Epoch AI models (more comprehensive than ECI alone)
ai_models = pd.read_csv('data/ai_models/all_ai_models.csv')
ai_models = ai_models.dropna(subset=['Training compute (FLOP)', 'Publication date'])
ai_models['Publication date'] = pd.to_datetime(ai_models['Publication date'])

print(f"\nEpoch AI dataset: {len(ai_models)} models with compute data")

# Extract features
ai_models['log_compute'] = np.log10(ai_models['Training compute (FLOP)'])
ai_models['year'] = ai_models['Publication date'].dt.year
ai_models['month'] = ai_models['Publication date'].dt.month
ai_models['date_numeric'] = ai_models['year'] + ai_models['month']/12

# Check features
has_params = ai_models['Parameters'].notna().sum()

print(f"  With parameters: {has_params}")

# Load ECI for capability scores
eci_data = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')
eci_data = eci_data.dropna(subset=['ECI Score'])
print(f"  ECI dataset: {len(eci_data)} models with capability scores")

print(f"\n{'='*100}")
print("FITTING ESTIMATION MODELS:")
print(f"{'='*100}")

# Model 1: Compute vs Time (temporal trend)
recent_data = ai_models[ai_models['date_numeric'] >= 2020].copy()

slope_time, intercept_time, r_time, _, _ = linregress(
    recent_data['date_numeric'],
    recent_data['log_compute']
)

print(f"\n1. TEMPORAL TREND (2020+):")
print(f"   log10(Compute) = {slope_time:.3f} * year + {intercept_time:.3f}")
print(f"   R = {r_time:.3f}")
print(f"   → Compute grows {10**slope_time:.2f}x per year")

# Model 2: Compute vs Parameters (Chinchilla-style)
params_data = ai_models.dropna(subset=['Parameters']).copy()
params_data['log_params'] = np.log10(params_data['Parameters'])

if len(params_data) > 10:
    slope_params, intercept_params, r_params, _, _ = linregress(
        params_data['log_params'],
        params_data['log_compute']
    )
    
    print(f"\n2. PARAMETER RELATIONSHIP:")
    print(f"   log10(Compute) = {slope_params:.3f} * log10(Params) + {intercept_params:.3f}")
    print(f"   R = {r_params:.3f}")
    print(f"   → Compute ~ Params^{slope_params:.2f}")
else:
    print(f"\n2. PARAMETER RELATIONSHIP: Not enough data ({len(params_data)} models)")
    slope_params = None

# Model 3: Compute vs ECI Score (need to join datasets)
# Merge ai_models with eci_data to get ECI scores
ai_with_eci = ai_models.merge(
    eci_data[['Display name', 'ECI Score']], 
    left_on='Model', 
    right_on='Display name', 
    how='inner'
)

if len(ai_with_eci) > 10:
    slope_eci, intercept_eci, r_eci, _, _ = linregress(
        ai_with_eci['ECI Score'],
        ai_with_eci['log_compute']
    )
    
    print(f"\n3. CAPABILITY RELATIONSHIP:")
    print(f"   log10(Compute) = {slope_eci:.4f} * ECI + {intercept_eci:.3f}")
    print(f"   R = {r_eci:.3f}")
    print(f"   (Based on {len(ai_with_eci)} models with both compute and ECI)")
else:
    print(f"\n3. CAPABILITY RELATIONSHIP: Not enough data ({len(ai_with_eci)} models)")
    slope_eci = None

# Model 4: Organization patterns
print(f"\n4. ORGANIZATION PATTERNS:")
org_stats = ai_models.groupby('Organization')['log_compute'].agg(['mean', 'std', 'count'])
org_stats = org_stats[org_stats['count'] >= 2].sort_values('mean', ascending=False)

print(f"\n   Top organizations by avg compute:")
for org, row in org_stats.head(5).iterrows():
    print(f"   {org:30s}: {10**row['mean']:.2e} FLOPs (±{row['std']:.1f} log10)")

# Save estimation models
estimation_models = {
    'temporal': {
        'slope': float(slope_time),
        'intercept': float(intercept_time),
        'r': float(r_time),
        'formula': f'log10(C) = {slope_time:.3f} * year + {intercept_time:.3f}'
    }
}

if slope_params is not None:
    estimation_models['parameters'] = {
        'slope': float(slope_params),
        'intercept': float(intercept_params),
        'r': float(r_params),
        'formula': f'log10(C) = {slope_params:.3f} * log10(P) + {intercept_params:.3f}'
    }

if slope_eci is not None:
    estimation_models['capability'] = {
        'slope': float(slope_eci),
        'intercept': float(intercept_eci),
        'r': float(r_eci),
        'formula': f'log10(C) = {slope_eci:.4f} * ECI + {intercept_eci:.3f}'
    }

estimation_models['organization_means'] = {
    org: float(10**row['mean']) 
    for org, row in org_stats.iterrows()
}

with open('outputs/compute_estimation_models.json', 'w') as f:
    json.dump(estimation_models, f, indent=2)

print(f"\n{'='*100}")
print("ESTIMATING COMPUTE FOR METR MODELS:")
print(f"{'='*100}")

# Load METR comparison data
metr_comparison = pd.read_csv('outputs/metr_vs_eci_comparison.csv')

# Models we want to estimate
target_models = [
    {'name': 'Claude 3.7 Sonnet', 'date': '2025-02-01', 'org': 'Anthropic', 'params': None, 'eci': 139, 'notes': 'Latest Claude'},
    {'name': 'GPT-4 0125', 'date': '2023-11-01', 'org': 'OpenAI', 'params': 1.76e12, 'eci': None, 'notes': 'GPT-4 Turbo'},
]

# Also add any from comparison that are missing compute
for _, row in metr_comparison.iterrows():
    if pd.isna(row['training_compute']) and not pd.isna(row['eci_score']):
        target_models.append({
            'name': row['model'],
            'date': None,
            'org': None,
            'params': None,
            'eci': row['eci_score'],
            'notes': 'From METR comparison'
        })

estimates = []

for model in target_models:
    print(f"\n{model['name']}:")
    
    # Try multiple estimation methods
    estimate_methods = []
    
    # Method 1: Temporal
    if model['date']:
        date_numeric = pd.to_datetime(model['date']).year + pd.to_datetime(model['date']).month/12
        log_c_temporal = slope_time * date_numeric + intercept_time
        c_temporal = 10 ** log_c_temporal
        estimate_methods.append(('Temporal', c_temporal, r_time))
        print(f"  Temporal estimate: {c_temporal:.2e} FLOPs (R={r_time:.2f})")
    
    # Method 2: Parameters
    if model['params'] and slope_params:
        log_c_params = slope_params * np.log10(model['params']) + intercept_params
        c_params = 10 ** log_c_params
        estimate_methods.append(('Parameters', c_params, r_params))
        print(f"  Parameter estimate: {c_params:.2e} FLOPs (R={r_params:.2f})")
    
    # Method 3: Capability (ECI)
    if model['eci'] and slope_eci:
        log_c_eci = slope_eci * model['eci'] + intercept_eci
        c_eci = 10 ** log_c_eci
        estimate_methods.append(('Capability', c_eci, r_eci))
        print(f"  Capability estimate: {c_eci:.2e} FLOPs (R={r_eci:.2f})")
    
    # Method 4: Organization average
    if model['org'] and model['org'] in estimation_models['organization_means']:
        c_org = estimation_models['organization_means'][model['org']]
        estimate_methods.append(('Organization', c_org, 0.5))  # Arbitrary R for org
        print(f"  Organization estimate: {c_org:.2e} FLOPs")
    
    # Weighted average by R²
    if estimate_methods:
        weights = np.array([r**2 for _, _, r in estimate_methods])
        weights = weights / weights.sum()
        
        values = np.array([c for _, c, _ in estimate_methods])
        weighted_estimate = np.exp(np.sum(weights * np.log(values)))  # Geometric mean with weights
        
        print(f"  → Weighted estimate: {weighted_estimate:.2e} FLOPs")
        print(f"     Methods used: {', '.join([m for m, _, _ in estimate_methods])}")
        
        # Confidence: min/max range
        conf_low = values.min()
        conf_high = values.max()
        print(f"     Range: {conf_low:.2e} - {conf_high:.2e} FLOPs")
        
        estimates.append({
            'model': model['name'],
            'estimated_compute': weighted_estimate,
            'conf_low': conf_low,
            'conf_high': conf_high,
            'methods': [m for m, _, _ in estimate_methods],
            'notes': model['notes']
        })
    else:
        print(f"  ✗ Cannot estimate (insufficient data)")

# Save estimates
estimates_df = pd.DataFrame(estimates)
estimates_df.to_csv('outputs/compute_estimates.csv', index=False)

print(f"\n{'='*100}")
print(f"✅ SAVED {len(estimates)} COMPUTE ESTIMATES")
print(f"{'='*100}")

print(f"\nNext steps:")
print(f"  1. Review estimates for reasonableness")
print(f"  2. Add manual constraints (e.g., 'o1 > GPT-4')")
print(f"  3. Merge with existing compute data")
print(f"  4. Rebuild LUCR analysis with expanded dataset")


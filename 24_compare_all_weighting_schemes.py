"""
Compare all 4 METR weighting schemes for ECI→METR mapping
Find the most reasonable one for AGI forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime, timedelta

print("="*100)
print("COMPARING ALL METR WEIGHTING SCHEMES")
print("="*100)

# Load the existing comparison data that has all weightings already calculated
comparison = pd.read_csv('outputs/metr_vs_eci_comparison.csv')
comparison = comparison.dropna(subset=['eci_score']).copy()

# Rename columns to match expected format
comparison = comparison.rename(columns={
    'metr_score_simple': 'metr_equal',
    'metr_score': 'metr_frontier'
})

# Need to recalculate remaining and binary from raw data
print("\nRecalculating remaining and binary weights...")
runs = pd.read_json('data/METR/all_runs.jsonl', lines=True)

# Models we care about (use the actual IDs from all_runs.jsonl)
target_models = {
    'Claude 3.5 Sonnet (Old)': 'claude_3_5_sonnet',  # June version
    'Claude 3.5 Sonnet (New)': 'claude_3_5_sonnet_20241022',  # October version
    'GPT-4 0314': 'gpt-4-0314',
    'GPT-4o': 'gpt_4o',
    'o1-preview': 'o1_preview'
}

remaining_scores = []
binary_scores = []

for display_name, metr_id in target_models.items():
    model_data = runs[runs['model'] == metr_id]
    if len(model_data) > 0:
        task_scores = model_data.groupby('task_family')['score_cont'].mean()
        
        # Remaining weight (1 - score)
        remaining_weights = 1 - task_scores
        remaining_weights = remaining_weights / remaining_weights.sum()
        remaining_score = (task_scores * remaining_weights).sum()
        
        # Binary weight (1 if score < 0.9, else 0.1)
        binary_weights = np.where(task_scores < 0.9, 1.0, 0.1)
        binary_weights = binary_weights / binary_weights.sum()
        binary_score = (task_scores * binary_weights).sum()
        
        remaining_scores.append({'model': display_name, 'metr_remaining': remaining_score})
        binary_scores.append({'model': display_name, 'metr_binary': binary_score})

remaining_df = pd.DataFrame(remaining_scores)
binary_df = pd.DataFrame(binary_scores)

# Merge into comparison
comparison = comparison.merge(remaining_df, on='model', how='left')
comparison = comparison.merge(binary_df, on='model', how='left')

print(f"\nModels with both METR and ECI: {len(comparison)}")
print("\nComparison data:")
print(comparison[['model', 'eci_score', 'metr_equal', 'metr_remaining', 'metr_frontier', 'metr_binary']].to_string(index=False))

# Fit linear mappings for all 4 schemes
schemes = ['equal', 'remaining', 'frontier', 'binary']
mappings = {}

print(f"\n{'='*100}")
print("LINEAR FITS FOR EACH SCHEME:")
print(f"{'='*100}")

for scheme in schemes:
    col = f'metr_{scheme}'
    m, c = np.polyfit(comparison['eci_score'], comparison[col], 1)
    r = np.corrcoef(comparison['eci_score'], comparison[col])[0, 1]
    
    mappings[scheme] = {
        'slope': m,
        'intercept': c,
        'r_squared': r**2
    }
    
    print(f"\n{scheme.upper()}:")
    print(f"  METR = {m:.6f} * ECI + {c:.6f}")
    print(f"  R² = {r**2:.4f}")

# Calculate AGI targets (METR = 0.9) for each scheme
target_metr = 0.9

print(f"\n{'='*100}")
print(f"AGI TARGETS (METR = {target_metr}):")
print(f"{'='*100}")

# Kaplan ceiling
with open('outputs/kaplan_horsepower_linear.json', 'r') as f:
    kaplan = json.load(f)
L_inf = kaplan['kaplan_constants']['L_inf']
a_linear = kaplan['linear_fit']['a']
b_linear = kaplan['linear_fit']['b']
max_eci_kaplan = a_linear * L_inf + b_linear

for scheme in schemes:
    m = mappings[scheme]['slope']
    c = mappings[scheme]['intercept']
    
    target_eci = (target_metr - c) / m
    achievable_pct = (max_eci_kaplan / target_eci) * 100 if target_eci > 0 else 999
    
    print(f"\n{scheme.upper():12s}: Target ECI = {target_eci:6.1f} (Kaplan can achieve {achievable_pct:5.1f}%)")

# Current progress for each scheme
current_eci = comparison['eci_score'].max()

print(f"\n{'='*100}")
print(f"CURRENT PROGRESS (Latest model: ECI {current_eci:.1f}):")
print(f"{'='*100}")

for scheme in schemes:
    m = mappings[scheme]['slope']
    c = mappings[scheme]['intercept']
    
    current_metr = m * current_eci + c
    progress_pct = (current_metr / target_metr) * 100
    gap = (target_metr - current_metr) * 100
    
    print(f"\n{scheme.upper():12s}: METR {current_metr*100:5.1f}% (Progress: {progress_pct:5.1f}%, Gap: {gap:5.1f} pp)")

# Save all mappings
with open('outputs/eci_metr_all_schemes.json', 'w') as f:
    json.dump(mappings, f, indent=2)

print(f"\nSaved all mappings to outputs/eci_metr_all_schemes.json")

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors = {
    'equal': '#2E86AB',
    'remaining': '#A23B72',
    'frontier': '#F18F01',
    'binary': '#06A77D'
}

# Plot each scheme
for idx, scheme in enumerate(schemes):
    ax = axes[idx // 2, idx % 2]
    
    col = f'metr_{scheme}'
    m = mappings[scheme]['slope']
    c = mappings[scheme]['intercept']
    r2 = mappings[scheme]['r_squared']
    
    # Scatter
    ax.scatter(comparison['eci_score'], comparison[col] * 100,
              s=150, alpha=0.7, color=colors[scheme], edgecolors='black', linewidth=2)
    
    # Fit line
    eci_range = np.linspace(120, 160, 100)
    metr_pred = (m * eci_range + c) * 100
    ax.plot(eci_range, metr_pred, '--', linewidth=2.5, color=colors[scheme], alpha=0.7)
    
    # AGI target
    target_eci = (target_metr - c) / m
    ax.axhline(90, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='AGI (90%)')
    ax.axvline(max_eci_kaplan, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Kaplan Ceiling')
    
    if target_eci > 0 and target_eci < 300:
        ax.scatter([target_eci], [90], s=300, marker='*', color='gold', 
                  edgecolors='black', linewidth=2, zorder=10, label='Target')
    
    ax.set_xlabel('ECI Score', fontsize=11, fontweight='bold')
    ax.set_ylabel('METR Score (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'{scheme.upper()} Weighting (R²={r2:.3f})', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
    ax.set_xlim([120, 160])

plt.tight_layout()
plt.savefig('outputs/eci_metr_all_schemes_comparison.png', dpi=300, bbox_inches='tight')
print("Saved visualization to outputs/eci_metr_all_schemes_comparison.png")

# Summary table
summary = []
for scheme in schemes:
    m = mappings[scheme]['slope']
    c = mappings[scheme]['intercept']
    target_eci = (target_metr - c) / m
    current_metr = m * current_eci + c
    progress = (current_metr / target_metr) * 100
    kaplan_achievable = (max_eci_kaplan / target_eci) * 100 if target_eci > 0 else 999
    
    summary.append({
        'scheme': scheme,
        'r_squared': mappings[scheme]['r_squared'],
        'current_metr': current_metr,
        'target_eci': target_eci,
        'progress_pct': progress,
        'kaplan_achievable_pct': kaplan_achievable
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv('outputs/weighting_schemes_summary.csv', index=False)

print(f"\n{'='*100}")
print("SUMMARY TABLE:")
print(f"{'='*100}")
print(summary_df.to_string(index=False))

print(f"\n{'='*100}")
print("✅ COMPARISON COMPLETE!")
print(f"{'='*100}")

print(f"\n💡 RECOMMENDATIONS:")
print(f"\n  EQUAL: R²={mappings['equal']['r_squared']:.3f}, Progress={summary_df[summary_df['scheme']=='equal']['progress_pct'].values[0]:.0f}%")
print(f"    → Optimistic, AGI seems very close")

print(f"\n  REMAINING: R²={mappings['remaining']['r_squared']:.3f}, Progress={summary_df[summary_df['scheme']=='remaining']['progress_pct'].values[0]:.0f}%")
print(f"    → Balanced between solved and unsolved tasks")

print(f"\n  FRONTIER: R²={mappings['frontier']['r_squared']:.3f}, Progress={summary_df[summary_df['scheme']=='frontier']['progress_pct'].values[0]:.0f}%")
print(f"    → Pessimistic, heavily weights hardest tasks")

print(f"\n  BINARY: R²={mappings['binary']['r_squared']:.3f}, Progress={summary_df[summary_df['scheme']=='binary']['progress_pct'].values[0]:.0f}%")
print(f"    → Conservative, treats tasks as solved/unsolved")


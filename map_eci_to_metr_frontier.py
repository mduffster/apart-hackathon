"""
Map ECI scale to Frontier-Weighted METR scale
Goal: Create conversion function so we can use ECI's 80 models with compute data
      but interpret results in METR's meaningful 0-1 scale
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load METR data
print("Loading METR data...")
metr_runs = []
with open('data/METR/all_runs.jsonl', 'r') as f:
    for line in f:
        metr_runs.append(json.loads(line))

metr_df = pd.DataFrame(metr_runs)

# Calculate frontier-weighted METR scores per model
print("Calculating frontier-weighted METR scores...")
metr_scores = []

for alias in metr_df['alias'].unique():
    if alias == 'human':
        continue
    
    model_runs = metr_df[metr_df['alias'] == alias]
    
    # Frontier weighting: (1 - score)^2
    scores = model_runs['score_cont']
    weights = (1 - scores) ** 2
    
    frontier_weighted = np.average(scores, weights=weights)
    
    metr_scores.append({
        'model': alias,
        'metr_frontier_weighted': frontier_weighted,
        'metr_equal_weighted': scores.mean(),
        'n_tasks': len(model_runs)
    })

metr_scores_df = pd.DataFrame(metr_scores)

# Load ECI data
print("Loading ECI data...")
eci = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')

# Manual matching (models in both datasets)
matches = [
    ('Claude 3.5 Sonnet (Old)', 'Claude 3.5 Sonnet (Jun 2024)'),
    ('Claude 3.5 Sonnet (New)', 'Claude 3.5 Sonnet (Oct 2024)'),
    ('Claude 3.7 Sonnet', 'Claude 3.7 Sonnet (1k thinking)'),
    ('GPT-4 0314', 'GPT-4 (Mar 2023)'),
    ('GPT-4 1106', 'GPT-4 Turbo Preview (Nov 2023)'),
    ('GPT-4o', 'GPT-4o (May 2024)'),
    ('o1-preview', 'o1-preview'),
]

# Create comparison dataframe
comparison = []

for metr_name, eci_name in matches:
    metr_row = metr_scores_df[metr_scores_df['model'] == metr_name]
    eci_row = eci[eci['Display name'] == eci_name]
    
    if len(metr_row) > 0 and len(eci_row) > 0 and pd.notna(eci_row.iloc[0]['ECI Score']):
        comparison.append({
            'model': metr_name,
            'eci_name': eci_name,
            'metr_frontier': metr_row.iloc[0]['metr_frontier_weighted'],
            'metr_equal': metr_row.iloc[0]['metr_equal_weighted'],
            'eci_score': eci_row.iloc[0]['ECI Score'],
        })

comparison_df = pd.DataFrame(comparison)

print("\n" + "="*100)
print("ECI vs FRONTIER-WEIGHTED METR:")
print("="*100)
print(comparison_df[['model', 'eci_score', 'metr_frontier']].to_string(index=False))

# Fit linear mapping: metr_frontier = a * eci + b
X = comparison_df['eci_score'].values
y = comparison_df['metr_frontier'].values

# Simple linear regression
coeffs = np.polyfit(X, y, 1)
a, b = coeffs

print(f"\n\nLINEAR MAPPING:")
print(f"  METR_frontier = {a:.6f} * ECI + {b:.6f}")

# Calculate R²
y_pred = a * X + b
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
r2 = 1 - (ss_res / ss_tot)

print(f"  R² = {r2:.4f}")

# Calculate what ECI scores map to key METR milestones
milestones = {
    'Current best (10%)': 0.10,
    'GPT-4 level (20%)': 0.20,
    'Half human-level (50%)': 0.50,
    'Near human-level (90%)': 0.90,
    'Human-level (100%)': 1.00,
}

print("\n\nKEY MILESTONES (METR → ECI):")
print("="*100)
for name, metr_val in milestones.items():
    # Solve for ECI: eci = (metr - b) / a
    eci_val = (metr_val - b) / a
    print(f"  {name:30s} | METR: {metr_val:.2f} → ECI: {eci_val:.1f}")

# Plot the mapping
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Scatter with fit line
ax1 = axes[0]
ax1.scatter(comparison_df['eci_score'], comparison_df['metr_frontier'], 
           s=200, alpha=0.7, edgecolors='black', linewidth=2, color='#2E86AB')

# Add fit line
X_smooth = np.linspace(X.min(), X.max(), 100)
y_smooth = a * X_smooth + b
ax1.plot(X_smooth, y_smooth, 'r--', linewidth=2.5, alpha=0.8, 
        label=f'METR = {a:.4f}*ECI + {b:.4f}\n(R² = {r2:.3f})')

# Add labels
for _, row in comparison_df.iterrows():
    ax1.annotate(row['model'], 
                xy=(row['eci_score'], row['metr_frontier']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax1.set_xlabel('ECI Score', fontsize=12, fontweight='bold')
ax1.set_ylabel('METR Frontier-Weighted Score', fontsize=12, fontweight='bold')
ax1.set_title('Mapping ECI to Frontier-Weighted METR', fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, max(y.max() * 1.1, 0.1)])

# Plot 2: Show conversion for full ECI range
ax2 = axes[1]

# Get ECI range from data
eci_full = eci[eci['ECI Score'].notna()]['ECI Score']
eci_range = np.linspace(eci_full.min(), eci_full.max(), 200)
metr_range = a * eci_range + b

ax2.plot(eci_range, metr_range, linewidth=3, color='#06A77D')
ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Human-level (1.0)')
ax2.axhline(0.5, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Half human-level (0.5)')

# Mark current frontier
current_eci = X.max()
current_metr = (a * current_eci + b)
ax2.scatter([current_eci], [current_metr], s=300, color='red', zorder=10, 
           edgecolors='black', linewidth=2, label='Current Frontier')

ax2.set_xlabel('ECI Score', fontsize=12, fontweight='bold')
ax2.set_ylabel('METR Frontier-Weighted Score', fontsize=12, fontweight='bold')
ax2.set_title('ECI → METR Conversion Function', fontsize=14, fontweight='bold', pad=20)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, min(1.1, metr_range.max() * 1.1)])

plt.tight_layout()
plt.savefig('outputs/eci_to_metr_mapping.png', dpi=300, bbox_inches='tight')
print("\nSaved mapping plot to outputs/eci_to_metr_mapping.png")

# Save mapping parameters
mapping_params = {
    'slope': a,
    'intercept': b,
    'r_squared': r2,
    'n_models': len(comparison_df),
    'formula': f'METR_frontier = {a:.6f} * ECI + {b:.6f}'
}

import json
with open('outputs/eci_to_metr_mapping.json', 'w') as f:
    json.dump(mapping_params, f, indent=2)

print("Saved mapping parameters to outputs/eci_to_metr_mapping.json")

print("\n" + "="*100)
print("✅ MAPPING COMPLETE!")
print("="*100)
print(f"\nNow we can:")
print(f"  1. Build scaling law using ECI's 80 models with compute data")
print(f"  2. Convert predictions to METR frontier scale using: METR = {a:.4f}*ECI + {b:.4f}")
print(f"  3. Calculate LUCR in meaningful terms (target = 1.0 = human-level)")


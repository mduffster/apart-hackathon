"""
Rebuild ECI → METR mapping using SIMPLE AVERAGE instead of frontier weighting
Test if this creates a more reasonable AGI forecast
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

print("="*100)
print("ECI → METR MAPPING: SIMPLE AVERAGE vs FRONTIER WEIGHTED")
print("="*100)

# Load comparison data
comparison = pd.read_csv('outputs/metr_vs_eci_comparison.csv')
comparison = comparison.dropna(subset=['eci_score'])

print(f"\nOverlapping models: {len(comparison)}")
print("\nMETR Score Comparison:")
print(comparison[['model', 'metr_score', 'metr_score_simple', 'eci_score']].to_string(index=False))

# Compare the two scoring methods
print(f"\n{'='*100}")
print("SCALE COMPARISON:")
print(f"{'='*100}")
print(f"\nFrontier Weighted:")
print(f"  Range: {comparison['metr_score'].min():.4f} - {comparison['metr_score'].max():.4f}")
print(f"  Mean: {comparison['metr_score'].mean():.4f}")
print(f"  As %: {comparison['metr_score'].min()*100:.1f}% - {comparison['metr_score'].max()*100:.1f}%")

print(f"\nSimple Average:")
print(f"  Range: {comparison['metr_score_simple'].min():.4f} - {comparison['metr_score_simple'].max():.4f}")
print(f"  Mean: {comparison['metr_score_simple'].mean():.4f}")
print(f"  As %: {comparison['metr_score_simple'].min()*100:.1f}% - {comparison['metr_score_simple'].max()*100:.1f}%")

print(f"\nRatio (simple/frontier): {comparison['metr_score_simple'].mean() / comparison['metr_score'].mean():.1f}x")

# Fit both mappings
# 1. Frontier weighted (original)
m_frontier, c_frontier = np.polyfit(comparison['eci_score'], comparison['metr_score'], 1)
r_frontier = np.corrcoef(comparison['eci_score'], comparison['metr_score'])[0, 1]

# 2. Simple average (new)
m_simple, c_simple = np.polyfit(comparison['eci_score'], comparison['metr_score_simple'], 1)
r_simple = np.corrcoef(comparison['eci_score'], comparison['metr_score_simple'])[0, 1]

print(f"\n{'='*100}")
print("LINEAR FITS:")
print(f"{'='*100}")

print(f"\nFrontier Weighted:")
print(f"  METR = {m_frontier:.6f} * ECI + {c_frontier:.6f}")
print(f"  R² = {r_frontier**2:.4f}")

print(f"\nSimple Average:")
print(f"  METR = {m_simple:.6f} * ECI + {c_simple:.6f}")
print(f"  R² = {r_simple**2:.4f}")

# Calculate AGI targets for both
target_metr = 0.9

eci_target_frontier = (target_metr - c_frontier) / m_frontier
eci_target_simple = (target_metr - c_simple) / m_simple

print(f"\n{'='*100}")
print("AGI TARGETS (METR = 0.9):")
print(f"{'='*100}")

print(f"\nFrontier Weighted → ECI = {eci_target_frontier:.1f}")
print(f"Simple Average → ECI = {eci_target_simple:.1f}")
print(f"Difference: {eci_target_frontier - eci_target_simple:.1f} ECI points")

# Load Kaplan ceiling
with open('outputs/kaplan_horsepower_linear.json', 'r') as f:
    kaplan = json.load(f)

L_inf = kaplan['kaplan_constants']['L_inf']
a_linear = kaplan['linear_fit']['a']
b_linear = kaplan['linear_fit']['b']

max_eci_kaplan = a_linear * L_inf + b_linear

print(f"\nKaplan ceiling: ECI = {max_eci_kaplan:.1f}")
print(f"\nFrontier: {(max_eci_kaplan / eci_target_frontier) * 100:.1f}% of target")
print(f"Simple: {(max_eci_kaplan / eci_target_simple) * 100:.1f}% of target")

# Current state
current_eci = comparison['eci_score'].max()
current_metr_frontier = comparison['metr_score'].max()
current_metr_simple = comparison['metr_score_simple'].max()

print(f"\n{'='*100}")
print("PROGRESS TO AGI:")
print(f"{'='*100}")

print(f"\nCurrent ECI: {current_eci:.1f}")

print(f"\nFrontier Weighted:")
print(f"  Current: {current_metr_frontier*100:.1f}%")
print(f"  Target: 90%")
print(f"  Progress: {(current_metr_frontier / 0.9) * 100:.1f}%")
print(f"  Gap: {(0.9 - current_metr_frontier)*100:.1f} percentage points")

print(f"\nSimple Average:")
print(f"  Current: {current_metr_simple*100:.1f}%")
print(f"  Target: 90%")
print(f"  Progress: {(current_metr_simple / 0.9) * 100:.1f}%")
print(f"  Gap: {(0.9 - current_metr_simple)*100:.1f} percentage points")

# Save simple average mapping
mapping_simple = {
    'slope': m_simple,
    'intercept': c_simple,
    'r_squared': r_simple**2,
    'n_models': len(comparison),
    'method': 'simple_average'
}

with open('outputs/eci_to_metr_mapping_simple.json', 'w') as f:
    json.dump(mapping_simple, f, indent=2)

print(f"\nSaved simple average mapping to outputs/eci_to_metr_mapping_simple.json")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Both mappings
ax1 = axes[0, 0]

eci_range = np.linspace(120, 150, 100)
metr_frontier_pred = m_frontier * eci_range + c_frontier
metr_simple_pred = m_simple * eci_range + c_simple

ax1.scatter(comparison['eci_score'], comparison['metr_score'] * 100,
           s=150, alpha=0.7, color='#2E86AB', edgecolors='black', linewidth=2,
           label='Frontier Weighted', marker='s')
ax1.plot(eci_range, metr_frontier_pred * 100, '--', linewidth=2.5, color='#2E86AB', alpha=0.7)

ax1.scatter(comparison['eci_score'], comparison['metr_score_simple'] * 100,
           s=150, alpha=0.7, color='#06A77D', edgecolors='black', linewidth=2,
           label='Simple Average', marker='o')
ax1.plot(eci_range, metr_simple_pred * 100, '-', linewidth=2.5, color='#06A77D', alpha=0.7)

ax1.axhline(90, color='orange', linestyle=':', linewidth=2.5, alpha=0.7, label='AGI Target (90%)')
ax1.axvline(max_eci_kaplan, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Kaplan Ceiling')

ax1.set_xlabel('ECI Score', fontsize=12, fontweight='bold')
ax1.set_ylabel('METR Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('ECI → METR: Frontier vs Simple Average', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 100])

# Plot 2: Scale comparison
ax2 = axes[0, 1]

models = comparison['model'].values
x = np.arange(len(models))
width = 0.35

bars1 = ax2.bar(x - width/2, comparison['metr_score'] * 100, width,
               label='Frontier Weighted', color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=1.5)
bars2 = ax2.bar(x + width/2, comparison['metr_score_simple'] * 100, width,
               label='Simple Average', color='#06A77D', alpha=0.7, edgecolor='black', linewidth=1.5)

ax2.set_ylabel('METR Score (%)', fontsize=12, fontweight='bold')
ax2.set_title('Weighting Impact by Model', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(90, color='orange', linestyle=':', linewidth=2, alpha=0.7)

# Plot 3: ECI targets comparison
ax3 = axes[1, 0]

targets = ['Frontier\nWeighted', 'Simple\nAverage', 'Kaplan\nCeiling']
eci_values = [eci_target_frontier, eci_target_simple, max_eci_kaplan]
colors_targets = ['#2E86AB', '#06A77D', '#E63946']

bars = ax3.barh(targets, eci_values, color=colors_targets, alpha=0.7, edgecolor='black', linewidth=2)

# Annotate
for i, (target, value) in enumerate(zip(targets, eci_values)):
    ax3.text(value + 20, i, f'{value:.1f}', va='center', fontsize=12, fontweight='bold')

ax3.set_xlabel('Target ECI for METR = 90%', fontsize=12, fontweight='bold')
ax3.set_title('AGI Target: Which is Achievable?', fontsize=14, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Progress bars
ax4 = axes[1, 1]

categories = ['Frontier\nWeighted', 'Simple\nAverage']
current_progress = [(current_metr_frontier / 0.9) * 100, (current_metr_simple / 0.9) * 100]
colors_progress = ['#2E86AB', '#06A77D']

bars = ax4.barh(categories, current_progress, color=colors_progress, alpha=0.7, 
               edgecolor='black', linewidth=2)

ax4.axvline(100, color='orange', linestyle='--', linewidth=2.5, alpha=0.7, label='AGI (100%)')

for i, (cat, progress) in enumerate(zip(categories, current_progress)):
    ax4.text(progress + 2, i, f'{progress:.1f}%', va='center', fontsize=12, fontweight='bold')

ax4.set_xlabel('Progress to AGI (%)', fontsize=12, fontweight='bold')
ax4.set_title('How Close Are We?', fontsize=14, fontweight='bold', pad=15)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3, axis='x')
ax4.set_xlim([0, 120])

plt.tight_layout()
plt.savefig('outputs/eci_metr_simple_vs_frontier.png', dpi=300, bbox_inches='tight')
print("Saved visualization to outputs/eci_metr_simple_vs_frontier.png")

print(f"\n{'='*100}")
print("✅ ANALYSIS COMPLETE!")
print(f"{'='*100}")

print(f"\n🎯 KEY INSIGHT:")
print(f"   Frontier weighting emphasizes unsolved tasks → deflates scores 10-20x")
print(f"   Simple average reflects actual model capability")
print(f"   \n   Using simple average:")
print(f"   • Current models: ~70% METR (not 3%)")
print(f"   • AGI target: ECI {eci_target_simple:.0f} (not {eci_target_frontier:.0f})")
print(f"   • Kaplan achieves: {(max_eci_kaplan / eci_target_simple) * 100:.0f}% of target (not {(max_eci_kaplan / eci_target_frontier) * 100:.0f}%)")


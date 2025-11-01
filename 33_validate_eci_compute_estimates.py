"""
Validate ECI → Compute Estimates
Test in-sample accuracy on models with both ECI and compute
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

print("="*100)
print("VALIDATING ECI → COMPUTE ESTIMATES")
print("="*100)

# Load modern scaling law
with open('outputs/modern_scaling_law.json', 'r') as f:
    modern_law = json.load(f)

eci_inf, A_eci, alpha_eci = modern_law['params']

def eci_to_compute(eci):
    """Inverse: solve for compute given ECI"""
    if eci >= eci_inf:
        return np.inf
    return np.power(A_eci / (eci_inf - eci), 1.0 / alpha_eci)

def compute_to_eci(compute):
    """Forward: ECI from compute"""
    return eci_inf - A_eci * np.power(compute, -alpha_eci)

# Load ECI data (used to FIT the model)
eci_data = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')
eci_data = eci_data.dropna(subset=['ECI Score', 'Training compute (FLOP)'])

print(f"\nTraining data: {len(eci_data)} models with both ECI and compute")

# For each model, predict compute from ECI
eci_data['predicted_compute'] = eci_data['ECI Score'].apply(eci_to_compute)
eci_data['actual_compute'] = eci_data['Training compute (FLOP)']

# Calculate errors
eci_data['log_predicted'] = np.log10(eci_data['predicted_compute'])
eci_data['log_actual'] = np.log10(eci_data['actual_compute'])
eci_data['log_error'] = eci_data['log_predicted'] - eci_data['log_actual']
eci_data['abs_log_error'] = np.abs(eci_data['log_error'])
eci_data['ratio'] = eci_data['predicted_compute'] / eci_data['actual_compute']

# Statistics
print(f"\n{'='*100}")
print("IN-SAMPLE VALIDATION:")
print(f"{'='*100}")

print(f"\nLog10 Error Statistics:")
print(f"  Mean: {eci_data['log_error'].mean():.3f}")
print(f"  Median: {eci_data['log_error'].median():.3f}")
print(f"  Std: {eci_data['log_error'].std():.3f}")
print(f"  MAE: {eci_data['abs_log_error'].mean():.3f}")

print(f"\nRatio Statistics (Predicted/Actual):")
print(f"  Median: {eci_data['ratio'].median():.2f}x")
print(f"  25th percentile: {eci_data['ratio'].quantile(0.25):.2f}x")
print(f"  75th percentile: {eci_data['ratio'].quantile(0.75):.2f}x")

# Within what range?
within_2x = (eci_data['ratio'] >= 0.5) & (eci_data['ratio'] <= 2.0)
within_5x = (eci_data['ratio'] >= 0.2) & (eci_data['ratio'] <= 5.0)
within_10x = (eci_data['ratio'] >= 0.1) & (eci_data['ratio'] <= 10.0)

print(f"\nAccuracy:")
print(f"  Within 2x: {within_2x.sum()}/{len(eci_data)} ({within_2x.sum()/len(eci_data)*100:.1f}%)")
print(f"  Within 5x: {within_5x.sum()}/{len(eci_data)} ({within_5x.sum()/len(eci_data)*100:.1f}%)")
print(f"  Within 10x: {within_10x.sum()}/{len(eci_data)} ({within_10x.sum()/len(eci_data)*100:.1f}%)")

# Worst predictions
print(f"\n{'='*100}")
print("WORST PREDICTIONS (Top 10):")
print(f"{'='*100}")

worst = eci_data.nlargest(10, 'abs_log_error')[['Display name', 'ECI Score', 'actual_compute', 'predicted_compute', 'ratio']]
for idx, row in worst.iterrows():
    print(f"\n{row['Display name']}:")
    print(f"  ECI: {row['ECI Score']:.1f}")
    print(f"  Actual: {row['actual_compute']:.2e} FLOPs")
    print(f"  Predicted: {row['predicted_compute']:.2e} FLOPs")
    print(f"  Ratio: {row['ratio']:.2f}x {'(over)' if row['ratio'] > 1 else '(under)'}")

# Best predictions
print(f"\n{'='*100}")
print("BEST PREDICTIONS (Top 5):")
print(f"{'='*100}")

best = eci_data.nsmallest(5, 'abs_log_error')[['Display name', 'ECI Score', 'actual_compute', 'predicted_compute', 'ratio']]
for idx, row in best.iterrows():
    print(f"\n{row['Display name']}:")
    print(f"  ECI: {row['ECI Score']:.1f}")
    print(f"  Actual: {row['actual_compute']:.2e} FLOPs")
    print(f"  Predicted: {row['predicted_compute']:.2e} FLOPs")
    print(f"  Ratio: {row['ratio']:.2f}x")

# Test on METR models specifically
print(f"\n{'='*100}")
print("METR MODELS VALIDATION:")
print(f"{'='*100}")

metr_comparison = pd.read_csv('outputs/metr_vs_eci_comparison.csv')
metr_with_both = metr_comparison.dropna(subset=['eci_score', 'training_compute'])

print(f"\nMETR models with both ECI and compute: {len(metr_with_both)}")

for _, row in metr_with_both.iterrows():
    actual = row['training_compute']
    predicted = eci_to_compute(row['eci_score'])
    ratio = predicted / actual
    
    print(f"\n{row['model']}:")
    print(f"  ECI: {row['eci_score']:.1f}")
    print(f"  Actual: {actual:.2e} FLOPs")
    print(f"  Predicted: {predicted:.2e} FLOPs")
    print(f"  Ratio: {ratio:.2f}x")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Predicted vs Actual (scatter)
ax1 = axes[0, 0]

ax1.scatter(eci_data['actual_compute'], eci_data['predicted_compute'],
           s=50, alpha=0.5, color='#2E86AB', edgecolors='black', linewidth=0.5)

# Perfect prediction line
min_val = min(eci_data['actual_compute'].min(), eci_data['predicted_compute'].min())
max_val = max(eci_data['actual_compute'].max(), eci_data['predicted_compute'].max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

# 2x bounds
ax1.plot([min_val, max_val], [min_val*2, max_val*2], 'orange', linestyle=':', linewidth=1.5, alpha=0.7, label='2x over')
ax1.plot([min_val, max_val], [min_val*0.5, max_val*0.5], 'orange', linestyle=':', linewidth=1.5, alpha=0.7, label='2x under')

ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Actual Compute (FLOPs)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Predicted Compute (FLOPs)', fontsize=12, fontweight='bold')
ax1.set_title('ECI → Compute Predictions', fontsize=14, fontweight='bold', pad=15)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, which='both')

# Plot 2: Error distribution
ax2 = axes[0, 1]

ax2.hist(eci_data['log_error'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect (0 error)')
ax2.axvline(eci_data['log_error'].median(), color='orange', linestyle='-', linewidth=2, 
           label=f'Median ({eci_data["log_error"].median():.2f})')

ax2.set_xlabel('Log10 Error (Predicted - Actual)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title('Prediction Error Distribution', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Error vs ECI
ax3 = axes[1, 0]

ax3.scatter(eci_data['ECI Score'], eci_data['log_error'],
           s=50, alpha=0.5, color='#06A77D', edgecolors='black', linewidth=0.5)

ax3.axhline(0, color='red', linestyle='--', linewidth=2)
ax3.set_xlabel('ECI Score', fontsize=12, fontweight='bold')
ax3.set_ylabel('Log10 Error', fontsize=12, fontweight='bold')
ax3.set_title('Error vs ECI (bias check)', fontsize=14, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3)

# Plot 4: Ratio distribution
ax4 = axes[1, 1]

# Remove extreme outliers for visualization
ratio_clipped = eci_data['ratio'].clip(0.01, 100)

ax4.hist(np.log10(ratio_clipped), bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect (1x)')
ax4.axvline(np.log10(eci_data['ratio'].median()), color='blue', linestyle='-', linewidth=2,
           label=f'Median ({eci_data["ratio"].median():.2f}x)')

# Add vertical lines for 2x, 5x, 10x
for x_val, label in [(np.log10(2), '2x'), (np.log10(5), '5x'), (np.log10(10), '10x')]:
    ax4.axvline(x_val, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax4.axvline(-x_val, color='gray', linestyle=':', linewidth=1, alpha=0.5)

ax4.set_xlabel('Log10(Predicted/Actual)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
ax4.set_title('Prediction Ratio Distribution', fontsize=14, fontweight='bold', pad=15)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/eci_compute_validation.png', dpi=300, bbox_inches='tight')
print(f"\n\nSaved visualization to outputs/eci_compute_validation.png")

# Save validation stats
validation_stats = {
    'n_models': len(eci_data),
    'log_error_mean': float(eci_data['log_error'].mean()),
    'log_error_median': float(eci_data['log_error'].median()),
    'log_error_std': float(eci_data['log_error'].std()),
    'mae': float(eci_data['abs_log_error'].mean()),
    'median_ratio': float(eci_data['ratio'].median()),
    'within_2x_pct': float(within_2x.sum()/len(eci_data)*100),
    'within_5x_pct': float(within_5x.sum()/len(eci_data)*100),
    'within_10x_pct': float(within_10x.sum()/len(eci_data)*100)
}

with open('outputs/eci_compute_validation.json', 'w') as f:
    json.dump(validation_stats, f, indent=2)

print(f"\n{'='*100}")
print("✅ VALIDATION COMPLETE")
print(f"{'='*100}")

print(f"\n💡 INTERPRETATION:")
if validation_stats['within_2x_pct'] > 70:
    print(f"   ✓ GOOD: {validation_stats['within_2x_pct']:.0f}% of predictions within 2x")
    print(f"   → Estimates are reliable for out-of-sample models")
elif validation_stats['within_5x_pct'] > 80:
    print(f"   ⚠️  MODERATE: {validation_stats['within_5x_pct']:.0f}% within 5x, but only {validation_stats['within_2x_pct']:.0f}% within 2x")
    print(f"   → Estimates give rough magnitude, use with uncertainty bounds")
else:
    print(f"   ✗ POOR: Only {validation_stats['within_5x_pct']:.0f}% within 5x")
    print(f"   → Estimates may not be reliable")


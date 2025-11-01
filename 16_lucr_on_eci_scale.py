"""
Calculate LUCR on ECI scale
Compare Kaplan horsepower (compute → loss → ECI) to actual ECI from data
This keeps the analysis on the scale with 80 data points
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

# Load horsepower parameters
with open('outputs/kaplan_horsepower_params.json', 'r') as f:
    horsepower_params = json.load(f)

L_inf = horsepower_params['kaplan_constants']['L_inf']
A = horsepower_params['kaplan_constants']['A']
alpha = horsepower_params['kaplan_constants']['alpha']

k_fit = horsepower_params['sigmoid_fit']['k']
L_0_fit = horsepower_params['sigmoid_fit']['L_0']
ECI_min_fit = horsepower_params['sigmoid_fit']['ECI_min']
ECI_max_fit = horsepower_params['sigmoid_fit']['ECI_max']

def kaplan_loss(compute):
    return L_inf + A * np.power(compute, -alpha)

def sigmoid_loss_to_eci(loss):
    sigmoid = 1.0 / (1.0 + np.exp(-k_fit * (L_0_fit - loss)))
    return ECI_min_fit + (ECI_max_fit - ECI_min_fit) * sigmoid

def kaplan_horsepower_eci(compute):
    """Predict ECI from compute using Kaplan scaling"""
    loss = kaplan_loss(compute)
    return sigmoid_loss_to_eci(loss)

# Load ECI data
print("Loading ECI data...")
eci = pd.read_csv('data/benchmark_data/epoch_capabilities_index.csv')

# Filter to models with both ECI score and compute
eci_complete = eci[
    eci['ECI Score'].notna() & 
    eci['Training compute (FLOP)'].notna()
].copy()

print(f"Models with ECI score and compute: {len(eci_complete)}")

# Calculate Kaplan predictions
eci_complete['kaplan_predicted_eci'] = eci_complete['Training compute (FLOP)'].apply(kaplan_horsepower_eci)

# Calculate LUCR (as difference)
eci_complete['lucr_eci_diff'] = eci_complete['kaplan_predicted_eci'] - eci_complete['ECI Score']

# Calculate LUCR as compute tax
# For each model: what compute would Kaplan need to achieve this ECI?
def inverse_kaplan_eci(target_eci):
    """Find compute needed to achieve target ECI via Kaplan"""
    # Binary search in log space
    log_compute_range = np.linspace(20, 35, 10000)
    compute_range = np.power(10, log_compute_range)
    eci_range = np.array([kaplan_horsepower_eci(c) for c in compute_range])
    
    # Find closest
    idx = np.argmin(np.abs(eci_range - target_eci))
    return compute_range[idx]

print("\nCalculating inverse Kaplan (compute needed for observed ECI)...")
eci_complete['kaplan_needed_compute'] = eci_complete['ECI Score'].apply(inverse_kaplan_eci)
eci_complete['lucr_compute_tax'] = eci_complete['Training compute (FLOP)'] / eci_complete['kaplan_needed_compute']

# Sort by release date
eci_complete['Release date'] = pd.to_datetime(eci_complete['Release date'])
eci_complete = eci_complete.sort_values('Release date')

# Display results
print("\n" + "="*100)
print("LUCR ON ECI SCALE:")
print("="*100)

print(f"\nSample models:")
for _, row in eci_complete.head(10).iterrows():
    print(f"\n{row['Display name']}:")
    print(f"  Actual compute: {row['Training compute (FLOP)']:.2e} FLOPs")
    print(f"  Actual ECI: {row['ECI Score']:.2f}")
    print(f"  Kaplan predicts ECI: {row['kaplan_predicted_eci']:.2f}")
    print(f"  LUCR diff: {row['lucr_eci_diff']:.2f} ({"below" if row['lucr_eci_diff'] > 0 else "above"} prediction)")
    print(f"  LUCR tax: {row['lucr_compute_tax']:.2f}x ({"less" if row['lucr_compute_tax'] < 1 else "more"} efficient than Kaplan)")

# Analyze trends
print("\n" + "="*100)
print("LUCR TRENDS:")
print("="*100)

# Trend over time
time_idx = range(len(eci_complete))
diff_trend = np.polyfit(time_idx, eci_complete['lucr_eci_diff'].values, 1)
tax_trend = np.polyfit(time_idx, eci_complete['lucr_compute_tax'].values, 1)

print(f"\nLUCR difference trend: {diff_trend[0]:.4f} ECI points per model")
print(f"LUCR tax trend: {tax_trend[0]:.4f} per model")

if diff_trend[0] > 0:
    print(f"  → Kaplan predictions are getting MORE optimistic relative to reality")
    print(f"  → Efficiency is declining")
else:
    print(f"  → Kaplan predictions are getting LESS optimistic relative to reality")  
    print(f"  → Reality is outpacing Kaplan (algorithmic improvements)")

# Summary statistics
print(f"\n" + "="*100)
print("SUMMARY STATISTICS:")
print("="*100)

print(f"\nLUCR difference (Kaplan - Actual ECI):")
print(f"  Mean: {eci_complete['lucr_eci_diff'].mean():.2f}")
print(f"  Median: {eci_complete['lucr_eci_diff'].median():.2f}")
print(f"  Std: {eci_complete['lucr_eci_diff'].std():.2f}")

print(f"\nLUCR compute tax (Actual/Kaplan):")
print(f"  Mean: {eci_complete['lucr_compute_tax'].mean():.2f}x")
print(f"  Median: {eci_complete['lucr_compute_tax'].median():.2f}x")

if eci_complete['lucr_compute_tax'].mean() < 1:
    print(f"  → On average, we're {(1 - eci_complete['lucr_compute_tax'].mean())*100:.1f}% MORE efficient than Kaplan predicts")
else:
    print(f"  → On average, we need {(eci_complete['lucr_compute_tax'].mean() - 1)*100:.1f}% MORE compute than Kaplan predicts")

# Save
eci_complete.to_csv('outputs/lucr_eci_scale.csv', index=False)
print(f"\nSaved LUCR analysis to outputs/lucr_eci_scale.csv")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Kaplan vs Actual ECI
ax1 = axes[0, 0]
ax1.scatter(eci_complete['kaplan_predicted_eci'], eci_complete['ECI Score'],
           s=100, alpha=0.7, edgecolors='black', linewidth=1.5, c=eci_complete['Release date'].astype(int),
           cmap='viridis')

# Diagonal line (perfect match)
eci_range = [eci_complete['kaplan_predicted_eci'].min(), eci_complete['kaplan_predicted_eci'].max()]
ax1.plot(eci_range, eci_range, 'r--', linewidth=2.5, alpha=0.7, label='Perfect Match')

ax1.set_xlabel('Kaplan Predicted ECI', fontsize=12, fontweight='bold')
ax1.set_ylabel('Actual ECI', fontsize=12, fontweight='bold')
ax1.set_title('Kaplan vs Actual: ECI Scale', fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: LUCR difference over time
ax2 = axes[0, 1]
colors = ['#E63946' if x > 0 else '#06A77D' for x in eci_complete['lucr_eci_diff']]
ax2.scatter(eci_complete['Release date'], eci_complete['lucr_eci_diff'],
           s=100, alpha=0.7, c=colors, edgecolors='black', linewidth=1.5)

# Trend line
trend_line = diff_trend[0] * time_idx + diff_trend[1]
ax2.plot(eci_complete['Release date'], trend_line, 'r--', linewidth=2.5, alpha=0.7,
        label=f'Trend ({diff_trend[0]:.3f} ECI/model)')

ax2.axhline(0, color='black', linewidth=2, linestyle='--')
ax2.set_xlabel('Release Date', fontsize=12, fontweight='bold')
ax2.set_ylabel('LUCR (Kaplan - Actual ECI)', fontsize=12, fontweight='bold')
ax2.set_title('LUCR Difference Over Time', fontsize=14, fontweight='bold', pad=20)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Compute vs ECI (both curves)
ax3 = axes[1, 0]

# Actual data
ax3.scatter(eci_complete['Training compute (FLOP)'], eci_complete['ECI Score'],
           s=150, alpha=0.7, color='#06A77D', edgecolors='black', linewidth=1.5,
           label='Actual Data', zorder=10)

# Kaplan curve
compute_range = np.logspace(20, 27, 500)
eci_kaplan = [kaplan_horsepower_eci(c) for c in compute_range]
ax3.plot(compute_range, eci_kaplan, '--', linewidth=3, color='#E63946',
        alpha=0.7, label='Kaplan "Horsepower"')

ax3.set_xscale('log')
ax3.set_xlabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax3.set_ylabel('ECI Score', fontsize=12, fontweight='bold')
ax3.set_title('ECI vs Compute: Kaplan vs Reality', fontsize=14, fontweight='bold', pad=20)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, which='both')

# Plot 4: LUCR tax distribution
ax4 = axes[1, 1]
ax4.hist(eci_complete['lucr_compute_tax'], bins=30, color='#F18F01', 
        alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.axvline(1.0, color='red', linestyle='--', linewidth=2.5, label='Kaplan Baseline')
ax4.axvline(eci_complete['lucr_compute_tax'].median(), color='blue', linestyle='--', 
           linewidth=2.5, label=f'Median ({eci_complete["lucr_compute_tax"].median():.2f}x)')

ax4.set_xlabel('LUCR Compute Tax (Actual/Kaplan)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax4.set_title('Distribution of LUCR Tax', fontsize=14, fontweight='bold', pad=20)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/lucr_eci_analysis.png', dpi=300, bbox_inches='tight')
print("Saved LUCR ECI analysis to outputs/lucr_eci_analysis.png")

print("\n" + "="*100)
print("✅ LUCR ON ECI SCALE COMPLETE!")
print("="*100)
print("\nNext: Compare LUCR trends to METR frontier projections separately")


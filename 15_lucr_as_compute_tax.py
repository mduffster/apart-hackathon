"""
Recalculate LUCR as a Compute Tax
LUCR = Actual_compute / Kaplan_predicted_compute_for_this_capability

This measures: "How much more (or less) compute do we need than Kaplan predicts?"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import fsolve

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

eci_to_metr_slope = horsepower_params['eci_to_metr']['slope']
eci_to_metr_intercept = horsepower_params['eci_to_metr']['intercept']

def kaplan_loss(compute):
    return L_inf + A * np.power(compute, -alpha)

def sigmoid_loss_to_eci(loss):
    sigmoid = 1.0 / (1.0 + np.exp(-k_fit * (L_0_fit - loss)))
    return ECI_min_fit + (ECI_max_fit - ECI_min_fit) * sigmoid

def kaplan_horsepower_metr(compute):
    """Given compute, predict METR using Kaplan"""
    loss = kaplan_loss(compute)
    eci = sigmoid_loss_to_eci(loss)
    metr = eci_to_metr_slope * eci + eci_to_metr_intercept
    return metr

def inverse_kaplan(target_metr):
    """Given target METR, find the compute Kaplan predicts is needed"""
    # Solve numerically
    def objective(log_compute):
        compute = 10 ** log_compute
        predicted_metr = kaplan_horsepower_metr(compute)
        return predicted_metr - target_metr
    
    # Search in log space from 1e20 to 1e35
    try:
        result = fsolve(objective, 25.0)  # Start at 1e25
        return 10 ** result[0]
    except:
        return np.nan

# Load METR frontier data
print("Loading METR frontier data...")
lucr_df = pd.read_csv('outputs/lucr_time_series.csv')
lucr_df['date'] = pd.to_datetime(lucr_df['date'])

print(f"Data points: {len(lucr_df)}")

# For each time point, calculate LUCR as compute tax
print("\nCalculating LUCR as compute tax...")

lucr_tax_data = []

for _, row in lucr_df.iterrows():
    actual_compute = row['max_compute']
    actual_metr = row['metr_actual']
    
    # What compute does Kaplan predict for this METR?
    kaplan_predicted_compute = inverse_kaplan(actual_metr)
    
    if pd.notna(kaplan_predicted_compute) and kaplan_predicted_compute > 0:
        # LUCR tax = how much more compute we actually used
        lucr_tax = actual_compute / kaplan_predicted_compute
        
        lucr_tax_data.append({
            'date': row['date'],
            'actual_compute': actual_compute,
            'actual_metr': actual_metr,
            'kaplan_predicted_compute': kaplan_predicted_compute,
            'lucr_tax': lucr_tax
        })

lucr_tax_df = pd.DataFrame(lucr_tax_data)

print(f"✓ Calculated LUCR tax for {len(lucr_tax_df)} time points")

# Display results
print("\n" + "="*100)
print("LUCR AS COMPUTE TAX:")
print("="*100)

print(f"\nCurrent state ({lucr_tax_df['date'].iloc[-1].date()}):")
print(f"  Actual compute: {lucr_tax_df['actual_compute'].iloc[-1]:.2e} FLOPs")
print(f"  Actual METR: {lucr_tax_df['actual_metr'].iloc[-1]:.4f} ({lucr_tax_df['actual_metr'].iloc[-1]*100:.2f}%)")
print(f"  Kaplan predicts this needs: {lucr_tax_df['kaplan_predicted_compute'].iloc[-1]:.2e} FLOPs")
print(f"  LUCR tax: {lucr_tax_df['lucr_tax'].iloc[-1]:.2f}x")

if lucr_tax_df['lucr_tax'].iloc[-1] > 1:
    print(f"  → INEFFICIENT: We used {lucr_tax_df['lucr_tax'].iloc[-1]:.2f}x more compute than Kaplan predicts")
else:
    print(f"  → EFFICIENT: We used {lucr_tax_df['lucr_tax'].iloc[-1]:.2f}x less compute than Kaplan predicts")

# Analyze trend
tax_trend_coeffs = np.polyfit(range(len(lucr_tax_df)), lucr_tax_df['lucr_tax'].values, 1)
tax_slope = tax_trend_coeffs[0]
tax_intercept = tax_trend_coeffs[1]

print(f"\nLUCR tax trend:")
print(f"  Slope: {tax_slope:.6f} per month")

if tax_slope > 0:
    print(f"  → Tax is INCREASING: Becoming less efficient over time")
    print(f"  → Each capability increment requires more compute than Kaplan predicts")
else:
    print(f"  → Tax is DECREASING: Becoming more efficient over time")
    print(f"  → Algorithmic improvements are beating Kaplan predictions")

# Save
lucr_tax_df.to_csv('outputs/lucr_compute_tax.csv', index=False)
print(f"\nSaved LUCR tax data to outputs/lucr_compute_tax.csv")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: LUCR tax over time
ax1 = axes[0, 0]
colors = ['#E63946' if x > 1 else '#06A77D' for x in lucr_tax_df['lucr_tax']]
ax1.bar(lucr_tax_df['date'], lucr_tax_df['lucr_tax'], width=20, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax1.axhline(1.0, color='black', linewidth=2, linestyle='--', label='Kaplan Baseline')

# Add trend line
trend_line = tax_slope * np.arange(len(lucr_tax_df)) + tax_intercept
ax1.plot(lucr_tax_df['date'], trend_line, 'r--', linewidth=2.5, alpha=0.7, 
        label=f'Trend ({tax_slope:.4f}/month)')

ax1.set_ylabel('LUCR Compute Tax (×Kaplan)', fontsize=12, fontweight='bold')
ax1.set_title('LUCR as Compute Tax: Efficiency Over Time', fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Actual vs Predicted Compute
ax2 = axes[0, 1]
ax2.scatter(lucr_tax_df['kaplan_predicted_compute'], lucr_tax_df['actual_compute'],
           s=100, alpha=0.7, edgecolors='black', linewidth=1.5, color='#2E86AB')

# Add diagonal (perfect efficiency)
comp_range = [lucr_tax_df['kaplan_predicted_compute'].min(), 
              lucr_tax_df['kaplan_predicted_compute'].max()]
ax2.plot(comp_range, comp_range, 'r--', linewidth=2, alpha=0.7, label='Perfect Efficiency')

ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Kaplan Predicted Compute (FLOPs)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Actual Compute Used (FLOPs)', fontsize=11, fontweight='bold')
ax2.set_title('Actual vs Kaplan-Predicted Compute', fontsize=13, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: METR vs Compute (showing inefficiency)
ax3 = axes[1, 0]

# Kaplan curve
compute_range = np.logspace(20, 28, 500)
metr_kaplan = [kaplan_horsepower_metr(c) for c in compute_range]
ax3.plot(compute_range, np.array(metr_kaplan)*100, '--', linewidth=2.5, 
        color='#E63946', alpha=0.7, label='Kaplan Prediction')

# Actual data
ax3.scatter(lucr_tax_df['actual_compute'], lucr_tax_df['actual_metr']*100,
           s=150, alpha=0.7, edgecolors='black', linewidth=1.5, 
           color='#06A77D', label='Actual Frontier', zorder=10)

ax3.set_xscale('log')
ax3.set_xlabel('Training Compute (FLOPs)', fontsize=11, fontweight='bold')
ax3.set_ylabel('METR Frontier Score (%)', fontsize=11, fontweight='bold')
ax3.set_title('Capability vs Compute: Kaplan vs Reality', fontsize=13, fontweight='bold', pad=15)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, which='both')

# Plot 4: Tax distribution
ax4 = axes[1, 1]
ax4.hist(lucr_tax_df['lucr_tax'], bins=20, color='#F18F01', alpha=0.7, edgecolor='black')
ax4.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Kaplan Baseline')
ax4.axvline(lucr_tax_df['lucr_tax'].mean(), color='blue', linestyle='--', linewidth=2, 
           label=f'Mean ({lucr_tax_df["lucr_tax"].mean():.2f}x)')

ax4.set_xlabel('LUCR Compute Tax (×Kaplan)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Distribution of LUCR Tax', fontsize=13, fontweight='bold', pad=15)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('outputs/lucr_compute_tax_analysis.png', dpi=300, bbox_inches='tight')
print("Saved LUCR tax analysis to outputs/lucr_compute_tax_analysis.png")

print("\n" + "="*100)
print("✅ LUCR COMPUTE TAX ANALYSIS COMPLETE!")
print("="*100)
print("\nNext: Use tax trend to project compute requirements for METR = 0.9")


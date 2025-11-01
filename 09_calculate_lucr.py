"""
Steps 4 & 5: Build Horsepower Curve and Calculate LUCR
Goal: Compare actual capability (frontier-weighted index) to predicted capability (Kaplan scaling)
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load scaling law parameters
with open('outputs/scaling_law_params.json', 'r') as f:
    scaling_params = json.load(f)

# Get log law parameters (our best fit)
a_log = scaling_params['log_law']['a']
b_log = scaling_params['log_law']['b']

print("="*100)
print("KAPLAN SCALING LAW:")
print("="*100)
print(f"score = {a_log:.6f} * log10(compute) + {b_log:.6f}")
print(f"R² = {scaling_params['log_law']['r2']:.4f}")

# Define prediction function
def predict_score_from_compute(compute_flops):
    """Predict METR score from training compute using fitted scaling law"""
    return a_log * np.log10(compute_flops) + b_log

# Load frontier time series (from Step 2)
frontier_df = pd.read_csv('outputs/frontier_time_series.csv')
frontier_df['date'] = pd.to_datetime(frontier_df['date'])

# Build horsepower curve: predicted capability from frontier compute
print("\n" + "="*100)
print("BUILDING HORSEPOWER CURVE:")
print("="*100)

frontier_df['horsepower_predicted'] = frontier_df['max_compute_flops'].apply(predict_score_from_compute)

print(f"Horsepower curve built for {len(frontier_df)} time points")
print(f"Date range: {frontier_df['date'].min().date()} to {frontier_df['date'].max().date()}")

# Load actual capability index (from Step 5: frontier-weighted)
# We need to recreate this from the METR data, grouped by date

print("\n" + "="*100)
print("BUILDING ACTUAL CAPABILITY CURVE:")
print("="*100)

# Load METR runs
metr_runs = []
with open('data/METR/all_runs.jsonl', 'r') as f:
    for line in f:
        metr_runs.append(json.loads(line))

metr_df = pd.DataFrame(metr_runs)

# Load model matches to get release dates
matches = pd.read_csv('outputs/metr_ai_model_matches.csv')
matches['release_date'] = pd.to_datetime(matches['release_date'])

# Create mapping: model alias → release date
alias_to_date = dict(zip(matches['metr_alias'], matches['release_date']))

# Add release dates to METR runs
metr_df['release_date'] = metr_df['alias'].map(alias_to_date)

# Filter to runs with release dates
metr_with_dates = metr_df[metr_df['release_date'].notna()].copy()

print(f"Runs with release dates: {len(metr_with_dates)}")

# For each time point in frontier_df, calculate actual capability
# Use only models released by that date

actual_capability = []

for date in frontier_df['date']:
    # Get runs from models released by this date
    available_runs = metr_with_dates[metr_with_dates['release_date'] <= date]
    
    if len(available_runs) == 0:
        actual_capability.append(np.nan)
        continue
    
    # For each task family, use the BEST score from any available model
    best_scores = available_runs.groupby('task_family')['score_cont'].max()
    
    # Apply frontier weighting: prioritize incomplete tasks
    weights = (1 - best_scores) ** 2
    
    # Calculate weighted average
    overall_capability = np.average(best_scores, weights=weights)
    actual_capability.append(overall_capability)

frontier_df['actual_capability'] = actual_capability

# Calculate LUCR = Predicted - Actual
# Positive LUCR = underperforming (inefficient)
# Negative LUCR = overperforming (efficient)

frontier_df['LUCR'] = frontier_df['horsepower_predicted'] - frontier_df['actual_capability']

# Save results
frontier_df.to_csv('outputs/lucr_time_series.csv', index=False)
print(f"\nSaved LUCR time series to outputs/lucr_time_series.csv")

# Display key statistics
print("\n" + "="*100)
print("LUCR SUMMARY:")
print("="*100)

# Filter to dates with actual capability data
valid_data = frontier_df[frontier_df['actual_capability'].notna()]

print(f"\nCurrent state ({valid_data['date'].max().date()}):")
print(f"  Frontier Compute: {valid_data['max_compute_flops'].iloc[-1]:.2e} FLOPs")
print(f"  Horsepower (Predicted): {valid_data['horsepower_predicted'].iloc[-1]:.4f}")
print(f"  Actual Capability: {valid_data['actual_capability'].iloc[-1]:.4f}")
print(f"  LUCR: {valid_data['LUCR'].iloc[-1]:.4f}")

if valid_data['LUCR'].iloc[-1] > 0:
    print(f"  → INEFFICIENT: Capability below Kaplan prediction")
else:
    print(f"  → EFFICIENT: Capability exceeds Kaplan prediction")

print(f"\nLUCR over time:")
print(f"  Mean: {valid_data['LUCR'].mean():.4f}")
print(f"  Std: {valid_data['LUCR'].std():.4f}")
print(f"  Min: {valid_data['LUCR'].min():.4f} on {valid_data.loc[valid_data['LUCR'].idxmin(), 'date'].date()}")
print(f"  Max: {valid_data['LUCR'].max():.4f} on {valid_data.loc[valid_data['LUCR'].idxmax(), 'date'].date()}")

# Check if LUCR is growing over time
lucr_trend = np.polyfit(range(len(valid_data)), valid_data['LUCR'].values, 1)[0]
print(f"\nLUCR trend: {lucr_trend:.6f} per time step")
if lucr_trend > 0:
    print("  → LUCR is INCREASING (efficiency declining over time)")
    print("  → Could challenge 'doubling every 7 months' narrative")
else:
    print("  → LUCR is DECREASING (efficiency improving over time)")

# Visualize LUCR
fig, axes = plt.subplots(3, 1, figsize=(15, 14))

# Plot 1: Horsepower vs Actual Capability
ax1 = axes[0]
ax1.plot(valid_data['date'], valid_data['horsepower_predicted'], 
         'o-', linewidth=2.5, markersize=8, color='#E63946', label='Horsepower (Kaplan Prediction)')
ax1.plot(valid_data['date'], valid_data['actual_capability'], 
         's-', linewidth=2.5, markersize=8, color='#06A77D', label='Actual Capability (Frontier-Weighted)')
ax1.fill_between(valid_data['date'], 
                 valid_data['horsepower_predicted'], 
                 valid_data['actual_capability'],
                 alpha=0.2, color='gray', label='LUCR Gap')
ax1.set_ylabel('METR Score', fontsize=12, fontweight='bold')
ax1.set_title('AGI Progress: Horsepower vs Actual Capability', fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=11, loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: LUCR over time
ax2 = axes[1]
colors = ['#E63946' if x > 0 else '#06A77D' for x in valid_data['LUCR']]
ax2.bar(valid_data['date'], valid_data['LUCR'], width=20, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(0, color='black', linewidth=2, linestyle='--')
ax2.set_ylabel('LUCR (Predicted - Actual)', fontsize=12, fontweight='bold')
ax2.set_title('Loss to Utility Conversion Rate (LUCR) Over Time', fontsize=14, fontweight='bold', pad=20)
ax2.grid(True, alpha=0.3, axis='y')

# Add annotations
ax2.text(0.02, 0.95, 'LUCR > 0: Inefficient (Underperforming)', 
        transform=ax2.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='#E63946', alpha=0.3))
ax2.text(0.02, 0.05, 'LUCR < 0: Efficient (Overperforming)', 
        transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='#06A77D', alpha=0.3))

# Plot 3: Frontier compute evolution
ax3 = axes[2]
ax3.plot(valid_data['date'], valid_data['max_compute_flops'], 
         'o-', linewidth=2.5, markersize=8, color='#2E86AB')
ax3.set_yscale('log')
ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
ax3.set_ylabel('Frontier Compute (FLOPs)', fontsize=12, fontweight='bold')
ax3.set_title('Frontier Compute Evolution', fontsize=14, fontweight='bold', pad=20)
ax3.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('outputs/lucr_analysis.png', dpi=300, bbox_inches='tight')
print(f"\nSaved LUCR analysis plot to outputs/lucr_analysis.png")

print("\n" + "="*100)
print("✅ LUCR CALCULATION COMPLETE!")
print("="*100)

print(f"\n📊 KEY INSIGHT:")
print(f"   Current LUCR = {valid_data['LUCR'].iloc[-1]:.4f}")
if valid_data['LUCR'].iloc[-1] > 0:
    efficiency_pct = (1 - valid_data['actual_capability'].iloc[-1] / valid_data['horsepower_predicted'].iloc[-1]) * 100
    print(f"   We are {efficiency_pct:.1f}% BELOW Kaplan prediction")
    print(f"   → Compute scaling is NOT translating fully to capability gains")
else:
    efficiency_pct = (valid_data['actual_capability'].iloc[-1] / valid_data['horsepower_predicted'].iloc[-1] - 1) * 100
    print(f"   We are {efficiency_pct:.1f}% ABOVE Kaplan prediction")
    print(f"   → Algorithmic improvements are outpacing raw compute scaling")


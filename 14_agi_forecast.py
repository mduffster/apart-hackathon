"""
Final AGI Forecast: Project compute and timeline for METR = 0.9
Using LUCR trend to account for efficiency erosion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
import json

# Load LUCR time series
lucr_df = pd.read_csv('outputs/lucr_time_series.csv')
lucr_df['date'] = pd.to_datetime(lucr_df['date'])

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
    loss = kaplan_loss(compute)
    eci = sigmoid_loss_to_eci(loss)
    metr = eci_to_metr_slope * eci + eci_to_metr_intercept
    return metr

print("="*100)
print("AGI FORECAST: Computing requirements for METR = 0.9")
print("="*100)

# Current state
current_compute = lucr_df['max_compute'].iloc[-1]
current_metr = lucr_df['metr_actual'].iloc[-1]
current_lucr = lucr_df['lucr'].iloc[-1]

print(f"\nCurrent state:")
print(f"  Compute: {current_compute:.2e} FLOPs")
print(f"  METR: {current_metr:.4f} ({current_metr*100:.2f}%)")
print(f"  LUCR: {current_lucr:.4f}")

# Analyze LUCR trend
lucr_trend_coeffs = np.polyfit(range(len(lucr_df)), lucr_df['lucr'].values, 1)
lucr_slope = lucr_trend_coeffs[0]
lucr_intercept = lucr_trend_coeffs[1]

print(f"\nLUCR trend:")
print(f"  Slope: {lucr_slope:.6f} per month")
print(f"  Current: {current_lucr:.4f}")

# Project LUCR into the future
def project_lucr(months_ahead):
    """Project LUCR value N months from now"""
    current_idx = len(lucr_df) - 1
    future_idx = current_idx + months_ahead
    return lucr_slope * future_idx + lucr_intercept

# Scenario: What compute is needed for METR = 0.9?
target_metr = 0.9

print(f"\n\nSCENARIO: Computing requirements for METR = {target_metr:.1f} (near-AGI)")
print("="*100)

# We need to solve: target_metr = kaplan_horsepower(C) + LUCR_adjustment
# Where LUCR_adjustment accounts for efficiency erosion

# Try different timeframes
timeframes_years = [1, 2, 5, 10, 20, 50]

projections = []

for years in timeframes_years:
    months = years * 12
    
    # Project LUCR at this timeframe
    future_lucr = project_lucr(months)
    
    # Account for LUCR: actual_metr = kaplan_predicted - LUCR
    # So: kaplan_predicted = actual_metr + LUCR
    kaplan_needed = target_metr + future_lucr
    
    # Solve for compute: kaplan_horsepower(C) = kaplan_needed
    # This is complex, so we'll search numerically
    
    # Try compute values from 1e25 to 1e35
    compute_range = np.logspace(25, 35, 10000)
    kaplan_preds = np.array([kaplan_horsepower_metr(c) for c in compute_range])
    
    # Find closest match
    idx = np.argmin(np.abs(kaplan_preds - kaplan_needed))
    required_compute = compute_range[idx]
    achieved_metr = kaplan_preds[idx] - future_lucr  # Subtract LUCR to get actual
    
    # Compute scaling factor
    scaling_factor = required_compute / current_compute
    
    projections.append({
        'years': years,
        'months': months,
        'future_lucr': future_lucr,
        'kaplan_needed': kaplan_needed,
        'required_compute': required_compute,
        'achieved_metr': achieved_metr,
        'scaling_factor': scaling_factor
    })
    
    print(f"\n{years} years from now:")
    print(f"  Projected LUCR: {future_lucr:.4f}")
    print(f"  Required compute: {required_compute:.2e} FLOPs")
    print(f"  Scaling factor: {scaling_factor:.0f}x current")
    print(f"  Achieved METR: {achieved_metr:.4f} ({achieved_metr*100:.1f}%)")

projections_df = pd.DataFrame(projections)

# Find when we reach METR = 0.9
# Assuming compute doubles every X months
compute_doubling_time_months = 6  # Estimate

print(f"\n\n" + "="*100)
print(f"TIMELINE PROJECTION (assuming compute doubles every {compute_doubling_time_months} months):")
print("="*100)

for _, proj in projections_df.iterrows():
    months_needed_for_compute = compute_doubling_time_months * np.log2(proj['scaling_factor'])
    years_needed = months_needed_for_compute / 12
    
    print(f"\nTo reach {proj['scaling_factor']:.0f}x compute:")
    print(f"  Time needed: {years_needed:.1f} years ({months_needed_for_compute:.0f} months)")
    print(f"  Target METR: {proj['achieved_metr']*100:.1f}%")
    
    if 0.85 <= proj['achieved_metr'] <= 0.95:
        print(f"  ✓ This achieves near-AGI (METR ≈ 0.9)")

# Save projections
projections_df.to_csv('outputs/agi_compute_projections.csv', index=False)
print(f"\n\nSaved projections to outputs/agi_compute_projections.csv")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: METR vs Compute (with projection)
ax1 = axes[0, 0]

# Historical data
ax1.scatter(lucr_df['max_compute'], lucr_df['metr_actual']*100, 
           s=100, alpha=0.7, color='#06A77D', edgecolors='black', linewidth=1.5, 
           label='Actual History', zorder=10)

# Kaplan horsepower curve
compute_smooth = np.logspace(20, 35, 500)
kaplan_smooth = np.array([kaplan_horsepower_metr(c)*100 for c in compute_smooth])
ax1.plot(compute_smooth, kaplan_smooth, '--', linewidth=2, color='#E63946', 
        alpha=0.7, label='Kaplan "Horsepower"')

# Projections
future_computes = projections_df['required_compute'].values
future_metrs = projections_df['achieved_metr'].values * 100
ax1.scatter(future_computes, future_metrs, s=200, marker='*', 
           color='#F18F01', edgecolors='black', linewidth=2, 
           label='Projections (with LUCR)', zorder=10)

# Annotate projections
for _, proj in projections_df.iterrows():
    if proj['years'] in [1, 5, 10, 20]:
        ax1.annotate(f"{proj['years']}y", 
                    xy=(proj['required_compute'], proj['achieved_metr']*100),
                    xytext=(10, 10), textcoords='offset points', fontsize=9)

ax1.axhline(90, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Near-AGI (90%)')
ax1.set_xscale('log')
ax1.set_xlabel('Training Compute (FLOPs)', fontsize=11, fontweight='bold')
ax1.set_ylabel('METR Frontier Score (%)', fontsize=11, fontweight='bold')
ax1.set_title('AGI Forecast: Compute Requirements', fontsize=13, fontweight='bold', pad=15)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_ylim([0, 100])

# Plot 2: Time to METR milestones
ax2 = axes[0, 1]

milestones = [0.5, 0.7, 0.9, 1.0]
milestone_times = []

for milestone in milestones:
    # Find projection closest to milestone
    idx = np.argmin(np.abs(projections_df['achieved_metr'] - milestone))
    proj = projections_df.iloc[idx]
    
    months_needed = compute_doubling_time_months * np.log2(proj['scaling_factor'])
    years_needed = months_needed / 12
    
    milestone_times.append({
        'milestone': milestone,
        'years': years_needed,
        'compute': proj['required_compute']
    })

milestone_df = pd.DataFrame(milestone_times)

ax2.barh(range(len(milestone_df)), milestone_df['years'], color=['#2E86AB', '#06A77D', '#F18F01', '#E63946'])
ax2.set_yticks(range(len(milestone_df)))
ax2.set_yticklabels([f"METR {m*100:.0f}%" for m in milestone_df['milestone']])
ax2.set_xlabel('Years from Now', fontsize=11, fontweight='bold')
ax2.set_title('Timeline to AGI Milestones', fontsize=13, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, axis='x')

for i, (years, compute) in enumerate(zip(milestone_df['years'], milestone_df['compute'])):
    ax2.text(years + 0.5, i, f'{years:.1f}y\\n{compute:.1e}', 
            va='center', fontsize=9)

# Plot 3: LUCR projection
ax3 = axes[1, 0]

# Historical LUCR
ax3.plot(lucr_df['date'], lucr_df['lucr']*100, 'o-', linewidth=2, markersize=6, 
        color='#2E86AB', label='Historical LUCR')

# Project LUCR forward
future_months = np.arange(len(lucr_df), len(lucr_df) + 240)  # 20 years
future_lucr = lucr_slope * future_months + lucr_intercept
future_dates = pd.date_range(lucr_df['date'].iloc[-1], periods=241, freq='MS')[1:]

ax3.plot(future_dates, future_lucr*100, '--', linewidth=2, color='#E63946', 
        alpha=0.7, label='Projected LUCR')

ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
ax3.set_ylabel('LUCR (%)', fontsize=11, fontweight='bold')
ax3.set_title('LUCR Projection: Efficiency Erosion', fontsize=13, fontweight='bold', pad=15)
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Scaling factors
ax4 = axes[1, 1]

ax4.plot(projections_df['years'], projections_df['scaling_factor'], 
        'o-', linewidth=2.5, markersize=8, color='#F18F01')
ax4.set_yscale('log')
ax4.set_xlabel('Years from Now', fontsize=11, fontweight='bold')
ax4.set_ylabel('Required Compute (×current)', fontsize=11, fontweight='bold')
ax4.set_title('Compute Scaling Requirements', fontsize=13, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('outputs/agi_forecast.png', dpi=300, bbox_inches='tight')
print("Saved AGI forecast plot to outputs/agi_forecast.png")

print("\n" + "="*100)
print("✅ AGI FORECAST COMPLETE!")
print("="*100)


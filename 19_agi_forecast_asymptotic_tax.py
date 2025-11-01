"""
Final AGI Forecast with Asymptotic Tax Model
Tax approaches 1.0 as algorithmic efficiency improves
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from datetime import datetime, timedelta

print("="*100)
print("AGI FORECAST WITH ASYMPTOTIC COMPUTE TAX")
print("="*100)

# Load LUCR compute tax data
print("\nLoading compute tax data...")
lucr_eci = pd.read_csv('outputs/lucr_eci_scale.csv')
lucr_eci['Release date'] = pd.to_datetime(lucr_eci['Release date'])
lucr_eci = lucr_eci.sort_values('Release date')

# Remove outliers in tax (anything > 10x is likely numerical error)
lucr_eci = lucr_eci[lucr_eci['lucr_compute_tax'] < 10].copy()

print(f"Models: {len(lucr_eci)}")
print(f"Compute tax range: {lucr_eci['lucr_compute_tax'].min():.2f}x - {lucr_eci['lucr_compute_tax'].max():.2f}x")
print(f"Median tax: {lucr_eci['lucr_compute_tax'].median():.2f}x")

# Fit asymptotic decay model
# Tax(t) = 1.0 + (initial_tax - 1.0) * exp(-decay_rate * t)
def asymptotic_tax(t, initial_tax, decay_rate):
    """Asymptotic decay approaching 1.0"""
    return 1.0 + (initial_tax - 1.0) * np.exp(-decay_rate * t)

# Fit to data
time_idx = np.arange(len(lucr_eci))
tax_values = lucr_eci['lucr_compute_tax'].values

try:
    # Initial guess: start at median, decay moderately
    p0 = [tax_values[0], 0.01]
    popt, pcov = curve_fit(asymptotic_tax, time_idx, tax_values, p0=p0, maxfev=10000)
    initial_tax_fit, decay_rate_fit = popt
    
    # Calculate R²
    tax_pred = asymptotic_tax(time_idx, *popt)
    ss_res = np.sum((tax_values - tax_pred) ** 2)
    ss_tot = np.sum((tax_values - tax_values.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n✓ Asymptotic tax model fit:")
    print(f"  Initial tax: {initial_tax_fit:.4f}x")
    print(f"  Decay rate: {decay_rate_fit:.6f} per model")
    print(f"  Asymptote: 1.0x (perfect Kaplan efficiency)")
    print(f"  R² = {r2:.4f}")
    
except Exception as e:
    print(f"\n✗ Asymptotic fit failed: {e}")
    print("Using median tax instead")
    initial_tax_fit = tax_values.median()
    decay_rate_fit = 0.0
    r2 = 0.0

# Load linear Kaplan parameters
print("\nLoading Kaplan horsepower (linear)...")
with open('outputs/kaplan_horsepower_linear.json', 'r') as f:
    kaplan_params = json.load(f)

L_inf = kaplan_params['kaplan_constants']['L_inf']
A = kaplan_params['kaplan_constants']['A']
alpha = kaplan_params['kaplan_constants']['alpha']
a_linear = kaplan_params['linear_fit']['a']
b_linear = kaplan_params['linear_fit']['b']

def kaplan_loss(compute):
    return L_inf + A * np.power(compute, -alpha)

def kaplan_horsepower_eci(compute):
    loss = kaplan_loss(compute)
    return a_linear * loss + b_linear

# Load ECI→METR mapping
with open('outputs/eci_to_metr_mapping.json', 'r') as f:
    mapping = json.load(f)

eci_to_metr_slope = mapping['slope']
eci_to_metr_intercept = mapping['intercept']

def eci_to_metr(eci):
    return eci_to_metr_slope * eci + eci_to_metr_intercept

# Current state
current_date = lucr_eci['Release date'].max()
current_compute = lucr_eci['Training compute (FLOP)'].iloc[-1]
current_eci = lucr_eci['ECI Score'].iloc[-1]
current_metr = eci_to_metr(current_eci)

print(f"\nCurrent state ({current_date.date()}):")
print(f"  Compute: {current_compute:.2e} FLOPs")
print(f"  ECI: {current_eci:.2f}")
print(f"  METR: {current_metr:.4f} ({current_metr*100:.2f}%)")

# AGI target
target_metr = 0.9
target_eci = (target_metr - eci_to_metr_intercept) / eci_to_metr_slope

print(f"\n" + "="*100)
print("AGI TARGET:")
print("="*100)
print(f"  METR = {target_metr:.2f} → ECI = {target_eci:.1f}")
print(f"  Current ECI: {current_eci:.1f}")
print(f"  Gap: {target_eci - current_eci:.1f} ECI points")

# Find Kaplan-predicted compute for target ECI
# Solve: target_eci = a_linear * (L_inf + A * C^(-alpha)) + b_linear
# This requires numerical search
def find_kaplan_compute(target_eci):
    """Find compute needed for target ECI via Kaplan"""
    compute_range = np.logspace(20, 40, 100000)
    eci_range = np.array([kaplan_horsepower_eci(c) for c in compute_range])
    idx = np.argmin(np.abs(eci_range - target_eci))
    return compute_range[idx]

kaplan_compute_needed = find_kaplan_compute(target_eci)
print(f"\nKaplan predicts need: {kaplan_compute_needed:.2e} FLOPs for ECI = {target_eci:.1f}")

# Project timeline with asymptotic tax
print(f"\n" + "="*100)
print("TIMELINE PROJECTIONS:")
print("="*100)

compute_doubling_months = 6  # Compute doubles every 6 months
models_per_year = 4  # Conservative estimate of frontier models per year

projections = []

for years_ahead in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50]:
    months = years_ahead * 12
    
    # Compute growth
    compute_multiplier = 2 ** (months / compute_doubling_months)
    available_compute = current_compute * compute_multiplier
    
    # Tax at this time (based on number of models)
    num_models_ahead = years_ahead * models_per_year
    current_model_idx = len(lucr_eci)
    future_model_idx = current_model_idx + num_models_ahead
    
    future_tax = asymptotic_tax(future_model_idx, initial_tax_fit, decay_rate_fit)
    
    # Actual compute needed = Kaplan × Tax
    actual_compute_needed = kaplan_compute_needed * future_tax
    
    # Can we achieve it?
    achieved = available_compute >= actual_compute_needed
    
    projections.append({
        'years': years_ahead,
        'date': current_date + timedelta(days=365*years_ahead),
        'available_compute': available_compute,
        'compute_multiplier': compute_multiplier,
        'tax': future_tax,
        'actual_compute_needed': actual_compute_needed,
        'achieved': achieved
    })
    
    status = "✓✓✓ AGI ACHIEVED!" if achieved else "Not yet"
    
    print(f"\n{years_ahead} years ({(current_date + timedelta(days=365*years_ahead)).year}):")
    print(f"  Available compute: {available_compute:.2e} FLOPs ({compute_multiplier:.0f}x)")
    print(f"  Tax: {future_tax:.3f}x (approaching 1.0)")
    print(f"  Needed compute: {actual_compute_needed:.2e} FLOPs")
    print(f"  Gap: {actual_compute_needed / available_compute:.2f}x short" if not achieved else f"  {status}")

projections_df = pd.DataFrame(projections)

# Find AGI date (first year where achieved = True)
agi_projections = projections_df[projections_df['achieved']]

if len(agi_projections) > 0:
    agi_year = agi_projections.iloc[0]
    
    print("\n" + "="*100)
    print("🎯 AGI TIMELINE:")
    print("="*100)
    print(f"\n  METR = {target_metr:.1f} (near-AGI) achievable by:")
    print(f"  📅 {agi_year['date'].date()}")
    print(f"  ⏱️  ~{agi_year['years']:.0f} years from now")
    print(f"  💻 Compute: {agi_year['available_compute']:.2e} FLOPs ({agi_year['compute_multiplier']:.0f}x current)")
    print(f"  📉 Tax at that time: {agi_year['tax']:.3f}x")
else:
    print("\n⚠️  AGI not achieved within 50-year projection window")
    print("    Either need:")
    print("    • Faster compute scaling")
    print("    • Faster tax improvement")
    print("    • Algorithmic breakthroughs")

# Save results
projections_df.to_csv('outputs/agi_timeline_asymptotic_tax.csv', index=False)
print(f"\nSaved projections to outputs/agi_timeline_asymptotic_tax.csv")

# Visualize
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: Tax decay over time
ax1 = fig.add_subplot(gs[0, :])

# Historical
ax1.scatter(lucr_eci['Release date'], lucr_eci['lucr_compute_tax'],
           s=100, alpha=0.7, color='#2E86AB', edgecolors='black', linewidth=1.5,
           label='Historical Data', zorder=10)

# Fitted curve (past)
time_smooth = np.linspace(0, len(lucr_eci)-1, 200)
tax_smooth = asymptotic_tax(time_smooth, initial_tax_fit, decay_rate_fit)
dates_smooth = pd.date_range(lucr_eci['Release date'].min(), lucr_eci['Release date'].max(), periods=200)
ax1.plot(dates_smooth, tax_smooth, '-', linewidth=2.5, color='#E63946', alpha=0.7, label='Fitted Asymptotic')

# Future projection
future_years = 30
future_model_indices = np.arange(len(lucr_eci), len(lucr_eci) + future_years * models_per_year)
future_tax = asymptotic_tax(future_model_indices, initial_tax_fit, decay_rate_fit)
future_dates = pd.date_range(lucr_eci['Release date'].max(), periods=len(future_model_indices), freq='3MS')
ax1.plot(future_dates, future_tax, '--', linewidth=2.5, color='#F18F01', alpha=0.7, label='Projected')

ax1.axhline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Perfect Efficiency (1.0x)')

ax1.set_ylabel('Compute Tax (Actual/Kaplan)', fontsize=13, fontweight='bold')
ax1.set_title('LUCR Compute Tax: Asymptotic Improvement Toward Kaplan Efficiency', 
             fontsize=15, fontweight='bold', pad=20)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.9, max(lucr_eci['lucr_compute_tax'].max(), 2.0)])

# Plot 2: Compute requirements vs availability
ax2 = fig.add_subplot(gs[1, 0])

years_plot = projections_df['years'].values
ax2.plot(years_plot, projections_df['available_compute'], 
        'o-', linewidth=3, markersize=8, color='#06A77D', label='Available Compute (6mo doubling)')
ax2.plot(years_plot, projections_df['actual_compute_needed'],
        's-', linewidth=3, markersize=8, color='#E63946', label='Needed for AGI (with tax)')

ax2.axhline(kaplan_compute_needed, color='blue', linestyle='--', linewidth=2, 
           alpha=0.7, label='Kaplan Baseline (no tax)')

ax2.set_yscale('log')
ax2.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax2.set_ylabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax2.set_title('Compute Requirements for AGI', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, which='both')

# Highlight AGI achievement
if len(agi_projections) > 0:
    ax2.scatter([agi_year['years']], [agi_year['available_compute']],
               s=500, marker='*', color='gold', edgecolors='black', linewidth=3,
               zorder=15, label='AGI Achieved')

# Plot 3: Tax improvement over time
ax3 = fig.add_subplot(gs[1, 1])

ax3.plot(projections_df['years'], projections_df['tax'],
        'o-', linewidth=3, markersize=8, color='#F18F01')
ax3.axhline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.7)

ax3.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax3.set_ylabel('Compute Tax', fontsize=12, fontweight='bold')
ax3.set_title('Tax Approaching Perfect Efficiency', fontsize=14, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3)

# Plot 4: Compute gap (needed/available ratio)
ax4 = fig.add_subplot(gs[2, :])

gap_ratio = projections_df['actual_compute_needed'] / projections_df['available_compute']
colors = ['#06A77D' if r <= 1.0 else '#E63946' for r in gap_ratio]

ax4.bar(projections_df['years'], gap_ratio, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.axhline(1.0, color='black', linestyle='--', linewidth=2.5)

ax4.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax4.set_ylabel('Compute Gap (Needed/Available)', fontsize=12, fontweight='bold')
ax4.set_title('AGI Feasibility: Gap < 1.0 = Achievable', fontsize=14, fontweight='bold', pad=15)
ax4.set_yscale('log')
ax4.grid(True, alpha=0.3, which='both', axis='y')

plt.savefig('outputs/agi_forecast_final.png', dpi=300, bbox_inches='tight')
print("Saved forecast visualization to outputs/agi_forecast_final.png")

print("\n" + "="*100)
print("✅ AGI FORECAST COMPLETE!")
print("="*100)


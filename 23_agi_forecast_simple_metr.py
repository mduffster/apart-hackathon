"""
AGI Forecast with Simple Average METR (not frontier weighted)
Much more achievable target: ECI 148 instead of 635
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
from datetime import datetime, timedelta

print("="*100)
print("AGI FORECAST: SIMPLE AVERAGE METR")
print("="*100)

# Load simple average mapping
with open('outputs/eci_to_metr_mapping_simple.json', 'r') as f:
    mapping = json.load(f)

slope = mapping['slope']
intercept = mapping['intercept']
r_squared = mapping['r_squared']

print(f"\nECI → METR (simple average):")
print(f"  METR = {slope:.6f} * ECI + {intercept:.6f}")
print(f"  R² = {r_squared:.4f}")

# Load Kaplan parameters
with open('outputs/kaplan_horsepower_linear.json', 'r') as f:
    kaplan = json.load(f)

L_inf = kaplan['kaplan_constants']['L_inf']
A = kaplan['kaplan_constants']['A']
alpha = kaplan['kaplan_constants']['alpha']
a_linear = kaplan['linear_fit']['a']
b_linear = kaplan['linear_fit']['b']

def kaplan_loss(compute):
    return L_inf + A * np.power(compute, -alpha)

def kaplan_eci(compute):
    return a_linear * kaplan_loss(compute) + b_linear

def eci_to_metr(eci):
    return slope * eci + intercept

# Load LUCR tax model
lucr_eci = pd.read_csv('outputs/lucr_eci_scale.csv')
lucr_eci = lucr_eci[lucr_eci['lucr_compute_tax'] < 10].copy()

# Filter out speculative models
lucr_eci = lucr_eci[lucr_eci['Confidence'] != 'Speculative'].copy()
print(f"\nFiltered to non-speculative models: {len(lucr_eci)}")

# Fit asymptotic tax
def asymptotic_tax(t, initial_tax, decay_rate):
    return 1.0 + (initial_tax - 1.0) * np.exp(-decay_rate * t)

time_idx = np.arange(len(lucr_eci))
tax_values = lucr_eci['lucr_compute_tax'].values
popt, _ = curve_fit(asymptotic_tax, time_idx, tax_values, p0=[tax_values[0], 0.01], maxfev=10000)
initial_tax_fit, decay_rate_fit = popt

# Current state
current_eci = lucr_eci['ECI Score'].max()
current_metr = eci_to_metr(current_eci)
current_compute = lucr_eci['Training compute (FLOP)'].max()
current_date = pd.to_datetime(lucr_eci['Release date'].max())

print(f"\nCurrent state ({current_date.date()}):")
print(f"  ECI: {current_eci:.1f}")
print(f"  METR: {current_metr:.2f} ({current_metr*100:.1f}%)")
print(f"  Compute: {current_compute:.2e} FLOPs")

# AGI target
target_metr = 0.9
target_eci = (target_metr - intercept) / slope

print(f"\n{'='*100}")
print("AGI TARGET:")
print(f"{'='*100}")
print(f"  METR = {target_metr:.2f} → ECI = {target_eci:.1f}")
print(f"  Current ECI: {current_eci:.1f}")
print(f"  Gap: {target_eci - current_eci:.1f} ECI points")
print(f"  Progress: {(current_metr / target_metr) * 100:.1f}%")

# Find Kaplan compute needed
def find_kaplan_compute(target_eci):
    compute_range = np.logspace(20, 50, 100000)
    eci_range = np.array([kaplan_eci(c) for c in compute_range])
    idx = np.argmin(np.abs(eci_range - target_eci))
    return compute_range[idx], eci_range[idx]

kaplan_compute_needed, eci_achieved = find_kaplan_compute(target_eci)

print(f"\nKaplan horsepower:")
print(f"  Need: {kaplan_compute_needed:.2e} FLOPs for ECI {target_eci:.1f}")
print(f"  Can achieve: ECI {eci_achieved:.1f}")

max_eci_kaplan = a_linear * L_inf + b_linear
print(f"  Ceiling: ECI {max_eci_kaplan:.1f}")
print(f"  Target is {(target_eci / max_eci_kaplan) * 100:.1f}% of ceiling ✅")

# Project timeline
print(f"\n{'='*100}")
print("TIMELINE PROJECTIONS:")
print(f"{'='*100}")

compute_doubling_months = 6
models_per_year = 4

projections = []

for years_ahead in [1, 2, 3, 5, 7, 10, 15, 20]:
    months = years_ahead * 12
    
    # Compute growth
    compute_multiplier = 2 ** (months / compute_doubling_months)
    available_compute = current_compute * compute_multiplier
    
    # Tax at this time
    num_models_ahead = years_ahead * models_per_year
    future_model_idx = len(lucr_eci) + num_models_ahead
    future_tax = asymptotic_tax(future_model_idx, initial_tax_fit, decay_rate_fit)
    
    # Actual compute needed = Kaplan × Tax
    actual_compute_needed = kaplan_compute_needed * future_tax
    
    # Can we achieve it?
    achieved = available_compute >= actual_compute_needed
    
    # What ECI/METR can we achieve?
    available_eci, _ = find_kaplan_compute(available_compute / future_tax)  # Reverse calc
    available_eci_pred = kaplan_eci(available_compute / future_tax)
    available_metr_pred = eci_to_metr(available_eci_pred)
    
    projections.append({
        'years': years_ahead,
        'date': current_date + timedelta(days=365*years_ahead),
        'available_compute': available_compute,
        'compute_multiplier': compute_multiplier,
        'tax': future_tax,
        'actual_compute_needed': actual_compute_needed,
        'available_eci': available_eci_pred,
        'available_metr': available_metr_pred,
        'achieved': achieved
    })
    
    status = "✓ AGI!" if achieved else ""
    
    print(f"\n{years_ahead} years ({(current_date + timedelta(days=365*years_ahead)).year}):")
    print(f"  Available: {available_compute:.2e} FLOPs ({compute_multiplier:.0f}x)")
    print(f"  Tax: {future_tax:.3f}x")
    print(f"  → ECI {available_eci_pred:.1f} → METR {available_metr_pred*100:.1f}% {status}")
    if achieved:
        print(f"  🎯 AGI ACHIEVED!")

projections_df = pd.DataFrame(projections)

# Find AGI date more precisely
agi_achieved = projections_df[projections_df['achieved']]

if len(agi_achieved) > 0:
    agi_year = agi_achieved.iloc[0]
    
    # Interpolate to find month
    if agi_year['years'] > 1:
        prev_row = projections_df[projections_df['years'] == agi_year['years'] - 1].iloc[0]
        
        # Binary search for exact month
        for months in range(int((agi_year['years']-1)*12), int(agi_year['years']*12)+1):
            compute_mult = 2 ** (months / compute_doubling_months)
            avail = current_compute * compute_mult
            
            years_frac = months / 12
            num_models = years_frac * models_per_year
            tax = asymptotic_tax(len(lucr_eci) + num_models, initial_tax_fit, decay_rate_fit)
            
            needed = kaplan_compute_needed * tax
            
            if avail >= needed:
                agi_months = months
                agi_years_precise = months / 12
                agi_date_precise = current_date + timedelta(days=365*years_frac)
                break
    else:
        agi_months = agi_year['years'] * 12
        agi_years_precise = agi_year['years']
        agi_date_precise = agi_year['date']
    
    print(f"\n{'='*100}")
    print("🎯 AGI TIMELINE:")
    print(f"{'='*100}")
    print(f"\n  METR = {target_metr:.1f} (near-AGI) achievable by:")
    print(f"  📅 {agi_date_precise.strftime('%B %Y')}")
    print(f"  ⏱️  ~{agi_years_precise:.1f} years from now")
    print(f"  💻 Compute: {agi_year['available_compute']:.2e} FLOPs")
    print(f"  📉 Tax: {agi_year['tax']:.3f}x")
else:
    print(f"\n⚠️  AGI not achieved within 20-year window")

# Save
projections_df.to_csv('outputs/agi_timeline_simple_metr.csv', index=False)
print(f"\nSaved to outputs/agi_timeline_simple_metr.csv")

# Visualize
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: METR progress over time
ax1 = fig.add_subplot(gs[0, :])

years = projections_df['years'].values
metr_progress = projections_df['available_metr'].values * 100

ax1.plot(years, metr_progress, 'o-', linewidth=3, markersize=10, 
        color='#06A77D', label='Projected METR')
ax1.axhline(90, color='orange', linestyle='--', linewidth=2.5, alpha=0.7, label='AGI Target (90%)')
ax1.axhline(current_metr * 100, color='blue', linestyle=':', linewidth=2, alpha=0.7, 
           label=f'Current ({current_metr*100:.1f}%)')

# Fill area
ax1.fill_between(years, current_metr * 100, metr_progress, alpha=0.2, color='#06A77D')

if len(agi_achieved) > 0:
    ax1.scatter([agi_years_precise], [90], s=500, marker='*', color='gold', 
               edgecolors='black', linewidth=3, zorder=15, label='AGI!')

ax1.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax1.set_ylabel('METR Score (%)', fontsize=12, fontweight='bold')
ax1.set_title('Path to AGI: METR Progress (Simple Average)', fontsize=15, fontweight='bold', pad=20)
ax1.legend(fontsize=11, loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([current_metr * 100 - 5, 100])

# Plot 2: ECI progress
ax2 = fig.add_subplot(gs[1, 0])

eci_progress = projections_df['available_eci'].values

ax2.plot(years, eci_progress, 'o-', linewidth=3, markersize=10, color='#2E86AB')
ax2.axhline(target_eci, color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Target ({target_eci:.0f})')
ax2.axhline(max_eci_kaplan, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Kaplan Ceiling ({max_eci_kaplan:.0f})')
ax2.axhline(current_eci, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'Current ({current_eci:.0f})')

ax2.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax2.set_ylabel('ECI Score', fontsize=12, fontweight='bold')
ax2.set_title('ECI Growth', fontsize=14, fontweight='bold', pad=15)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Compute requirements vs availability
ax3 = fig.add_subplot(gs[1, 1])

ax3.plot(years, projections_df['available_compute'], 'o-', linewidth=3, markersize=8,
        color='#06A77D', label='Available (6mo doubling)')
ax3.plot(years, projections_df['actual_compute_needed'], 's-', linewidth=3, markersize=8,
        color='#E63946', label='Needed (with tax)')

ax3.axhline(kaplan_compute_needed, color='blue', linestyle='--', linewidth=2, alpha=0.7,
           label='Kaplan baseline')

ax3.set_yscale('log')
ax3.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax3.set_ylabel('Training Compute (FLOPs)', fontsize=12, fontweight='bold')
ax3.set_title('Compute Requirements', fontsize=14, fontweight='bold', pad=15)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, which='both')

# Plot 4: Gap over time
ax4 = fig.add_subplot(gs[2, :])

gap = (0.9 - projections_df['available_metr']) * 100
colors_gap = ['#E63946' if g > 0 else '#06A77D' for g in gap]

bars = ax4.bar(years, gap, color=colors_gap, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.axhline(0, color='black', linestyle='-', linewidth=2)

ax4.set_xlabel('Years from Now', fontsize=12, fontweight='bold')
ax4.set_ylabel('Gap to AGI (percentage points)', fontsize=12, fontweight='bold')
ax4.set_title('Closing the Gap: AGI = METR 90%', fontsize=14, fontweight='bold', pad=15)
ax4.grid(True, alpha=0.3, axis='y')

plt.savefig('outputs/agi_forecast_simple_metr.png', dpi=300, bbox_inches='tight')
print("Saved visualization to outputs/agi_forecast_simple_metr.png")

print(f"\n{'='*100}")
print("✅ AGI FORECAST COMPLETE!")
print(f"{'='*100}")

print(f"\n💡 KEY FINDINGS:")
print(f"   • Using simple average METR (not frontier weighted)")
print(f"   • Current models: {current_metr*100:.1f}% METR")
print(f"   • Target: 90% METR (ECI {target_eci:.0f})")
print(f"   • Progress: {(current_metr/0.9)*100:.1f}%")
if len(agi_achieved) > 0:
    print(f"   • Timeline: ~{agi_years_precise:.1f} years ({agi_date_precise.year})")
print(f"   • Kaplan can achieve {(max_eci_kaplan/target_eci)*100:.0f}% of target ✅")

